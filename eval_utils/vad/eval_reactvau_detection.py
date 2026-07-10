"""
Combined PaliGemma + StreamForest Video Anomaly Detection Evaluation Script

Architecture:
    PaliGemma2-3B (Detection Module) + StreamForest-Qwen2-7B (Reasoning Module)

Pipeline:
    1. Video sampled at 4 FPS. Every 4 frames (1 second) → 2x2 grid → PaliGemma detection.
       Simultaneously, the last frame of each second is fed into StreamForest's MemoryManager.
    2. When PaliGemma detects an anomaly (score > threshold), StreamForest is triggered for 
       deep reasoning verification using its accumulated visual memory.
    3. Four trigger modes:
       - Mode A ("direct"): Fixed verification prompt → StreamForest re-judges Yes/No
       - Mode B ("score"): PaliGemma anomaly score (%) embedded in prompt
       - Mode C ("event"): PaliGemma score + top-k event types with probabilities
         (e.g., "Fighting (75%), Arrest (18%)") → embedded in StreamForest prompt
       - Mode D ("description"): PaliGemma generates a short description of the anomaly,
         which is appended to the StreamForest prompt for context-aware reasoning.

Scoring:
    - If PaliGemma says Normal → final score = PaliGemma score (low)
    - If PaliGemma says Anomaly → trigger StreamForest → final score from StreamForest logits
    - Metrics computed at frame level (same as eval_paligemma_detection.py)
"""

from detect_utils import compute_metrics, gaussian_smooth_1d, OnlineSmoother, load_anno_txt, make_gt_labels_from_anno
from llava.model.multimodal_projector.memory_manager import MemoryManager
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from llava.model.builder import load_pretrained_model
from vad.get_prompt import (
    get_grid_prompt,
    get_sf_prompt,
    PALIGEMMA_DESCRIBE_PROMPT,
)
from peft import PeftModel
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from safetensors.torch import load_file as safetensors_load_file
import types
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import PIL.Image
import numpy as np
import traceback
import warnings
import logging
import datetime
import time
import random
import argparse
import torch.nn.functional as F
import torch
import json
import cv2
import os
import sys

# Add ReactVAU path dynamically — MUST be before any imports that depend on it
ReactVAU_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if ReactVAU_ROOT not in sys.path:
    sys.path.insert(0, ReactVAU_ROOT)


# Also ensure the eval_utils/vad directory is on path for detect_utils
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ===================== Grid Image Creation (same as eval_paligemma) =====================
GRID_SIZE = (2, 2)
USE_SEPARATOR = True
SEPARATOR_WIDTH = 2
SEPARATOR_COLOR = (128, 128, 128)


def compute_grid_params(image_size: int = 448):
    """Compute grid cell size and total size based on image resolution.

    For 448: cell_size=223, total_size=448 (1024 image tokens)
    For 384: cell_size=191, total_size=384 (729 image tokens)
    """
    if USE_SEPARATOR:
        cell_size = (image_size - SEPARATOR_WIDTH) // 2
        total_size = image_size
    else:
        cell_size = image_size // 2
        total_size = image_size
    return cell_size, total_size


def resize_frame(image: Image.Image, target_size: int) -> Image.Image:
    if image is None:
        return Image.new('RGB', (target_size, target_size), (0, 0, 0))
    return image.resize((target_size, target_size), Image.Resampling.LANCZOS)


def create_grid_image(frames: list, grid_size: tuple = (2, 2), image_size: int = 448) -> Image.Image:
    """
    Create a grid image matching the training data format.

    For 2x2 grid with separator:
    - 448: Each cell is 223x223, 2px separator, total 448x448
    - 384: Each cell is 191x191, 2px separator, total 384x384
    """
    if not frames:
        return None
    cell_size, total_size = compute_grid_params(image_size)
    rows, cols = grid_size
    expected_frames = rows * cols
    while len(frames) < expected_frames:
        frames.append(
            frames[-1] if frames else Image.new('RGB', (cell_size, cell_size), (0, 0, 0)))

    if USE_SEPARATOR:
        grid_image = Image.new(
            'RGB', (total_size, total_size), color=SEPARATOR_COLOR)
        for idx, frame in enumerate(frames[:expected_frames]):
            resized_frame = resize_frame(frame, target_size=cell_size)
            row, col = divmod(idx, cols)
            x = col * (cell_size + SEPARATOR_WIDTH)
            y = row * (cell_size + SEPARATOR_WIDTH)
            grid_image.paste(resized_frame, (x, y))
    else:
        grid_image = Image.new('RGB', (total_size, total_size), color='black')
        for idx, frame in enumerate(frames[:expected_frames]):
            resized_frame = resize_frame(frame, target_size=cell_size)
            row, col = divmod(idx, cols)
            grid_image.paste(resized_frame, (col * cell_size, row * cell_size))
    return grid_image


# ===================== StreamForest Vision Encoder (384x384 support) =====================
def load_streamforest_vision_weights(model, weights_path: str):
    """Replace PaliGemma's vision encoder with StreamForest's SigLIP-384 weights."""
    logging.info(f"Loading StreamForest vision weights from: {weights_path}")

    stream_weights = safetensors_load_file(weights_path)
    logging.info(f"  Loaded {len(stream_weights)} tensors from StreamForest")

    remapped = {}
    for key, value in stream_weights.items():
        if key.startswith("vision_model."):
            remapped[key] = value

    pos_key = "vision_model.embeddings.position_embedding.weight"
    if pos_key in remapped:
        new_pos_shape = remapped[pos_key].shape
        old_pos_shape = model.vision_tower.vision_model.embeddings.position_embedding.weight.shape
        logging.info(
            f"  Position embedding: {old_pos_shape} -> {new_pos_shape}")

        new_num_positions = new_pos_shape[0]
        embed_dim = new_pos_shape[1]

        model.vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(
            new_num_positions, embed_dim
        ).to(model.device).to(model.dtype)
        logging.info(
            f"  Resized position embedding to {new_num_positions} positions")

    missing, unexpected = model.vision_tower.load_state_dict(
        remapped, strict=False)
    logging.info(f"  Missing keys (kept from PaliGemma): {len(missing)}")
    logging.info(f"  Unexpected keys: {len(unexpected)}")

    model.vision_tower.vision_model.embeddings.num_positions = 729
    embeddings_module = model.vision_tower.vision_model.embeddings
    embeddings_module.register_buffer(
        "position_ids",
        torch.arange(729, device=model.device).expand((1, -1)),
        persistent=False,
    )

    model.config.vision_config.image_size = 384
    model.config.vision_config.num_image_tokens = 729
    model.config.vision_config.num_positions = 729
    model.config.text_config.num_image_tokens = 729

    logging.info(f"  Config updated: image_size=384, num_image_tokens=729")
    return model


def patch_vision_feature_layer(model, select_layer: int = -2):
    """Monkey-patch get_image_features to extract from a specific encoder layer."""
    # Determine the target model to patch
    # After merge_and_unload(): PaliGemmaForConditionalGeneration -> patch directly
    # During PEFT training: PeftModel -> need base_model.model
    if hasattr(model, 'get_image_features'):
        target = model
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        target = model.base_model.model
    else:
        target = model

    def get_image_features_with_layer_select(self, pixel_values):
        image_outputs = self.vision_tower(
            pixel_values, output_hidden_states=True)
        selected_image_feature = image_outputs.hidden_states[select_layer]
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = image_features / (self.config.hidden_size ** 0.5)
        return image_features

    target.get_image_features = types.MethodType(
        get_image_features_with_layer_select, target
    )
    logging.info(f"  Patched get_image_features to use layer {select_layer}")
    return model


# ===================== PaliGemma Detection Module =====================

class PaliGemmaDetector:
    """PaliGemma2-3B based fast anomaly detection module."""

    def __init__(
        self,
        model_path: str,
        lora_path: str = None,
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "eager",
        image_size: int = 448,
        streamforest_weights_path: str = None,
        vision_feature_layer: int = -2,
    ):
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.image_size = image_size

        logging.info(f"[PaliGemma] Loading model from {model_path}...")
        logging.info(f"[PaliGemma] Image size: {image_size}x{image_size}")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            attn_implementation=attn_implementation,
        )

        # Load StreamForest vision encoder for 384x384 mode
        if image_size == 384 and streamforest_weights_path:
            self.model = load_streamforest_vision_weights(
                self.model, streamforest_weights_path)
            self.processor.image_processor.size = {"height": 384, "width": 384}
            self.processor.image_processor.image_seq_length = 729
            self.processor.image_seq_length = 729
            logging.info(
                f"[PaliGemma] Processor updated for 384x384 (729 image tokens)")

        if lora_path is not None and os.path.exists(lora_path):
            logging.info(
                f"[PaliGemma] Loading LoRA adapter from {lora_path}...")
            self.model = PeftModel.from_pretrained(
                self.model, lora_path, torch_dtype=torch_dtype)
            self.model = self.model.merge_and_unload()
            logging.info("[PaliGemma] LoRA adapter loaded and merged")

        # Patch vision feature layer AFTER LoRA merge (for 384 mode)
        if image_size == 384 and streamforest_weights_path and vision_feature_layer != -1:
            self.model = patch_vision_feature_layer(
                self.model, select_layer=vision_feature_layer)

        self.model.eval()

        # Token IDs
        self.yes_token_id = self._get_token_id("Yes")
        self.no_token_id = self._get_token_id("No")
        self.yes_lower_token_id = self._get_token_id("yes")
        self.no_lower_token_id = self._get_token_id("no")

        logging.info(
            f"[PaliGemma] Model loaded. Yes={self.yes_token_id}, No={self.no_token_id}")

    def _get_token_id(self, word: str) -> int:
        tokens = self.processor.tokenizer.encode(
            word, add_special_tokens=False)
        return tokens[0] if tokens else -1

    def score_grid(self, grid_image: Image.Image, prompt: str) -> float:
        """Get anomaly score from logits for a single grid image."""
        if grid_image is None:
            return 0.0
        full_prompt = f"<image>{prompt}"
        inputs = self.processor(
            text=full_prompt, images=grid_image, return_tensors="pt").to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(
                self.torch_dtype)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                num_logits_to_keep=1,
            )
            last_logits = outputs.logits[0, -1]
            yes_logit = max(last_logits[self.yes_token_id].float(
            ), last_logits[self.yes_lower_token_id].float())
            no_logit = max(last_logits[self.no_token_id].float(
            ), last_logits[self.no_lower_token_id].float())
            probs = F.softmax(torch.stack([yes_logit, no_logit]), dim=0)
            return probs[0].item()

    def batch_score_grids(self, grid_images: list, prompt: str) -> list:
        """Batch scoring for multiple grid images."""
        if not grid_images:
            return []
        valid_indices = [i for i, img in enumerate(
            grid_images) if img is not None]
        valid_images = [grid_images[i] for i in valid_indices]
        if not valid_images:
            return [0.0] * len(grid_images)

        full_prompt = f"<image>{prompt}"
        prompts = [full_prompt] * len(valid_images)
        inputs = self.processor(text=prompts, images=valid_images,
                                return_tensors="pt", padding=True).to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(
                self.torch_dtype)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                num_logits_to_keep=1,
            )
            logits = outputs.logits
            batch_scores = []
            for batch_idx in range(len(valid_images)):
                last_logits = logits[batch_idx, -1]
                yes_logit = max(last_logits[self.yes_token_id].float(
                ), last_logits[self.yes_lower_token_id].float())
                no_logit = max(last_logits[self.no_token_id].float(
                ), last_logits[self.no_lower_token_id].float())
                probs = F.softmax(torch.stack([yes_logit, no_logit]), dim=0)
                batch_scores.append(probs[0].item())

        final_scores = [0.0] * len(grid_images)
        for orig_idx, score in zip(valid_indices, batch_scores):
            final_scores[orig_idx] = score
        return final_scores

    def describe_grid(self, grid_image: Image.Image) -> str:
        """Generate a short anomaly description from a grid image."""
        if grid_image is None:
            return ""
        full_prompt = f"<image>{PALIGEMMA_DESCRIBE_PROMPT}"
        inputs = self.processor(
            text=full_prompt, images=grid_image, return_tensors="pt").to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(
                self.torch_dtype)

        with torch.inference_mode():
            input_len = inputs["input_ids"].shape[1]
            outputs = self.model.generate(
                **inputs, max_new_tokens=60, do_sample=False)
            generated_ids = outputs[0, input_len:]
            return self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def extract_event_types_topk(
        self,
        grid_image: Image.Image,
        prompt: str,
        top_k: int = 3,
        min_prob_threshold: float = 0.05,
    ) -> dict:
        """
        Extract top-k event types with probabilities from a grid image.
        
        The model is trained on labels like:
            "Yes. Event: Arrest."
            "No. Event: Normal."
            "Yes. Event: Arrest, Fighting."
        
        This method:
        1. Encodes the image with the detection prompt
        2. Force-generates "Yes. Event: " prefix (since we already know it's anomaly)
        3. Extracts top-k logits at the event type position
        4. Filters out "Normal" and formats with probabilities
        
        Args:
            grid_image: Input 2x2 grid PIL Image
            prompt: Detection prompt (same as scoring)
            top_k: Number of top event types to return
            min_prob_threshold: Minimum probability to include (default 5%)
        
        Returns:
            dict with:
                - event_info: Formatted string like "Fighting (75%), Arrest (18%)"
                - event_types_short: Comma-separated types like "Fighting, Arrest"
                - event_probs: List of (event_type, probability) tuples
        """
        if grid_image is None:
            return {
                "event_info": "Unknown",
                "event_types_short": "Unknown",
                "event_probs": [],
            }
        
        # Build the prompt with forced prefix to generate event type
        # We use the same prompt but will manually decode to get event position
        full_prompt = f"<image>{prompt}"
        inputs = self.processor(
            text=full_prompt, images=grid_image, return_tensors="pt"
        ).to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.torch_dtype)
        
        with torch.inference_mode():
            # Generate with the prefix "Yes. Event: " forced
            # This gives us the logits at the event type position
            # First, encode the prefix tokens we want to force
            prefix_text = "Yes. Event:"
            prefix_tokens = self.processor.tokenizer.encode(
                prefix_text, add_special_tokens=False
            )
            
            # Generate the full prefix first (greedy)
            input_len = inputs["input_ids"].shape[1]
            
            # Method: Generate step by step with forced decoding or
            # simpler: generate normally and extract logits via model forward
            
            # Create the forced prefix input_ids
            prefix_tensor = torch.tensor([prefix_tokens], dtype=torch.long, device=self.device)
            extended_input_ids = torch.cat([inputs["input_ids"], prefix_tensor], dim=1)
            
            # Extend attention mask
            prefix_attn = torch.ones(1, len(prefix_tokens), dtype=torch.long, device=self.device)
            extended_attention_mask = torch.cat([inputs["attention_mask"], prefix_attn], dim=1)
            
            # Forward pass to get logits at the next token position (after "Event:")
            outputs = self.model(
                input_ids=extended_input_ids,
                attention_mask=extended_attention_mask,
                pixel_values=inputs["pixel_values"],
                num_logits_to_keep=1,
            )
            
            # Get logits at the last position (next token after "Event:")
            next_token_logits = outputs.logits[0, -1]  # [vocab_size]
            
            # Convert to probabilities
            probs = F.softmax(next_token_logits.float(), dim=-1)
            
            # Get top-k tokens and their probabilities
            topk_probs, topk_indices = torch.topk(probs, k=min(top_k * 2, 20))  # Get extra for filtering
            
            # Decode tokens and filter
            event_results = []
            for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
                token_str = self.processor.tokenizer.decode([idx]).strip()
                
                # Skip "Normal", empty, or non-alphabetic tokens
                if not token_str or token_str.lower() == "normal":
                    continue
                if not token_str[0].isalpha():
                    continue
                
                # Skip if below threshold
                if prob < min_prob_threshold:
                    continue
                
                # Clean up token (remove leading/trailing punctuation)
                token_clean = token_str.strip(".,;:!? ")
                if token_clean:
                    event_results.append((token_clean, prob))
                
                # Stop when we have enough
                if len(event_results) >= top_k:
                    break
            
            # Format outputs
            if not event_results:
                return {
                    "event_info": "Abnormal activity",
                    "event_types_short": "Abnormal",
                    "event_probs": [],
                }
            
            # Format: "Fighting (75%), Arrest (18%)"
            event_info_parts = [
                f"{event} ({int(round(prob * 100))}%)"
                for event, prob in event_results
            ]
            event_info = ", ".join(event_info_parts)
            
            # Short format: "Fighting, Arrest"
            event_types_short = ", ".join([event for event, _ in event_results])
            
            return {
                "event_info": event_info,
                "event_types_short": event_types_short,
                "event_probs": event_results,
            }


# ===================== StreamForest Reasoning Module =====================

class StreamForestReasoner:
    """StreamForest-Qwen2-7B based deep reasoning module with visual memory."""

    def __init__(
        self,
        model_path: str,
        model_base: str = None,
        conv_template: str = "qwen_2",
        device: str = "cuda:0",
        time_msg_style: str = "short_online_v2",
        load_4bit: bool = False,
        load_8bit: bool = False,
    ):
        self.conv_template = conv_template
        self.device = torch.device(device)
        self.time_msg_style = time_msg_style

        if model_base is not None and model_base.strip() == "":
            model_base = None

        logging.info(f"[StreamForest] Loading model from {model_path}...")
        if model_base:
            logging.info(f"[StreamForest] Base model: {model_base}")
        if load_4bit:
            logging.info("[StreamForest] Quantization: 4-bit (NF4)")
        elif load_8bit:
            logging.info("[StreamForest] Quantization: 8-bit")

        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            model_path,
            model_base=model_base,
            model_name=model_name,
            device_map=device,
            multimodal=True,
            load_4bit=load_4bit,
            load_8bit=load_8bit,
            attn_implementation="sdpa",
        )
        self.model.eval()

        self.vision_tower = self.model.get_vision_tower()
        self.mm_projector = self.model.get_model().mm_projector

        # Yes/No token IDs (both capitalized and lowercase for CoT generation parsing)
        self.positive_token_id = self._get_token_id("Yes")
        self.negative_token_id = self._get_token_id("No")
        self.positive_lower_token_id = self._get_token_id("yes")
        self.negative_lower_token_id = self._get_token_id("no")

        # Rating digit token IDs (for 1-9 rating scale scoring)
        self.digit_token_ids = {}
        for digit in range(1, 10):
            tid = self._get_token_id(str(digit))
            self.digit_token_ids[digit] = tid
        logging.info(f"[StreamForest] Digit token IDs: {self.digit_token_ids}")

        # Pad token
        if self.tokenizer.pad_token_id is None:
            if "qwen" in self.tokenizer.name_or_path.lower():
                self.tokenizer.pad_token_id = 151643

        logging.info(
            f"[StreamForest] Model loaded. Yes={self.positive_token_id}, No={self.negative_token_id}, "
            f"yes={self.positive_lower_token_id}, no={self.negative_lower_token_id}")

    def _get_token_id(self, word: str) -> int:
        tokens = self.tokenizer.encode(word, add_special_tokens=False)
        return tokens[0] if tokens else None

    def encode_frame(self, pil_frame: PIL.Image.Image) -> torch.Tensor:
        """Encode a single frame through the vision tower. Returns [H*W, C] features."""
        with torch.inference_mode():
            frame_tensor = self.image_processor.preprocess(
                [pil_frame], return_tensors="pt"
            )["pixel_values"].to(dtype=self.model.dtype, device=self.device)
            frame_features = self.vision_tower(frame_tensor)
            return frame_features[0]  # [H*W, C]

    def encode_frames_batch(self, pil_frames: list) -> torch.Tensor:
        """Encode multiple frames in a single batch forward pass through the vision tower.
        
        More efficient than calling encode_frame() in a loop since it utilizes
        GPU parallelism by processing all frames in one kernel launch.
        
        Args:
            pil_frames: list of PIL.Image.Image frames to encode
            
        Returns:
            [N, H*W, C] tensor of vision features for each frame
        """
        with torch.inference_mode():
            batch_tensor = self.image_processor.preprocess(
                pil_frames, return_tensors="pt"
            )["pixel_values"].to(dtype=self.model.dtype, device=self.device)
            batch_features = self.vision_tower(batch_tensor)  # [N, H*W, C]
            return batch_features

    def get_anomaly_score_from_logits(
        self,
        memory_manager: MemoryManager,
        question: str,
        current_time: float,
        sampled_count: int,
        video_height: int,
        video_width: int,
        rt_anomaly_tokens: torch.Tensor = None,
    ) -> float:
        """
        Run LLM inference using accumulated memory to get Yes/No anomaly score.
        
        Args:
            rt_anomaly_tokens: Optional [N, H*W, C] real-time anomaly frame features.
                              When provided, replaces Now in the memory sequence.
        """
        with torch.inference_mode():
            memory_tokens = memory_manager.get_memory_tokens(
                rt_anomaly_tokens=rt_anomaly_tokens
            )
            visual_features = self.mm_projector.mlp(memory_tokens)

        # Build prompt
        conv = conv_templates[self.conv_template].copy()
        time_msg = self._generate_time_msg(current_time, sampled_count)
        full_question = DEFAULT_IMAGE_TOKEN + "\n" + time_msg + question
        conv.append_message(conv.roles[0], full_question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)
        attention_mask = input_ids.ne(
            self.tokenizer.pad_token_id).long().to(self.device)

        with torch.inference_mode():
            dummy_images = [torch.zeros(
                1, 3, video_height, video_width,
                dtype=self.model.dtype, device=self.device
            )]
            image_sizes = [(video_height, video_width)]

            (_, position_ids, attention_mask_prepared, _, inputs_embeds, _) = \
                self.model.prepare_inputs_labels_for_LLM(
                    input_ids, None, attention_mask,
                    None, None, dummy_images,
                    [visual_features], ["video"],
                    image_sizes=image_sizes,
            )

            outputs = self.model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask_prepared,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )

            last_hidden = outputs.last_hidden_state[:, -1:, :]
            last_logits = self.model.lm_head(last_hidden)[0, -1, :]
            pos_logit = last_logits[self.positive_token_id].item()
            neg_logit = last_logits[self.negative_token_id].item()
            logits_pair = torch.tensor([pos_logit, neg_logit])
            probs = torch.softmax(logits_pair, dim=0)
            return probs[0].item()

    def get_anomaly_score_from_rating_logits(
        self,
        memory_manager: MemoryManager,
        question: str,
        current_time: float,
        sampled_count: int,
        video_height: int,
        video_width: int,
        rt_anomaly_tokens: torch.Tensor = None,
    ) -> float:
        """
        Run LLM inference using accumulated memory to get anomaly score via 1-9 rating scale.

        Instead of extracting Yes/No logits (poorly calibrated for SF which was not
        trained on binary classification), this method extracts logits for digit tokens
        "1" through "9" and computes a weighted average as the anomaly score.

        Score = (E[rating] - 1) / 8, where E[rating] = sum(i * P(i)) for i=1..9
        This maps the expected rating from [1, 9] to [0, 1].

        Args:
            rt_anomaly_tokens: Optional [N, H*W, C] real-time anomaly frame features.
                              When provided, replaces Now in the memory sequence.
        """
        with torch.inference_mode():
            memory_tokens = memory_manager.get_memory_tokens(
                rt_anomaly_tokens=rt_anomaly_tokens
            )
            visual_features = self.mm_projector.mlp(memory_tokens)

        # Build prompt (same as Yes/No method)
        conv = conv_templates[self.conv_template].copy()
        time_msg = self._generate_time_msg(current_time, sampled_count)
        full_question = DEFAULT_IMAGE_TOKEN + "\n" + time_msg + question
        conv.append_message(conv.roles[0], full_question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)
        attention_mask = input_ids.ne(
            self.tokenizer.pad_token_id).long().to(self.device)

        with torch.inference_mode():
            dummy_images = [torch.zeros(
                1, 3, video_height, video_width,
                dtype=self.model.dtype, device=self.device
            )]
            image_sizes = [(video_height, video_width)]

            (_, position_ids, attention_mask_prepared, _, inputs_embeds, _) = \
                self.model.prepare_inputs_labels_for_LLM(
                    input_ids, None, attention_mask,
                    None, None, dummy_images,
                    [visual_features], ["video"],
                    image_sizes=image_sizes,
            )

            outputs = self.model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask_prepared,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )

            last_hidden = outputs.last_hidden_state[:, -1:, :]
            last_logits = self.model.lm_head(last_hidden)[0, -1, :]

            # Extract logits for digit tokens "1" through "9"
            digit_logits = []
            for digit in range(1, 10):
                token_id = self.digit_token_ids[digit]
                digit_logits.append(last_logits[token_id].float())

            digit_logits = torch.stack(digit_logits)  # shape: [9]
            digit_probs = torch.softmax(digit_logits, dim=0)  # shape: [9]

            # Weighted average: E[rating] = sum(i * P(i)) for i=1..9
            ratings = torch.arange(
                1, 10, dtype=torch.float32, device=digit_probs.device)
            expected_rating = (ratings * digit_probs).sum().item()

            # Normalize to [0, 1]: (E - 1) / 8
            score = (expected_rating - 1.0) / 8.0
            return max(0.0, min(1.0, score))  # clamp to [0, 1]

    def generate_text(
        self,
        memory_manager: MemoryManager,
        question: str,
        current_time: float,
        sampled_count: int,
        video_height: int,
        video_width: int,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        rt_anomaly_tokens: torch.Tensor = None,
    ) -> str:
        """
        Run LLM text generation using accumulated memory.
        Used for detailed reasoning output.
        
        Args:
            rt_anomaly_tokens: Optional [N, H*W, C] real-time anomaly frame features.
                              When provided, replaces Now in the memory sequence.
        """
        with torch.inference_mode():
            memory_tokens = memory_manager.get_memory_tokens(
                rt_anomaly_tokens=rt_anomaly_tokens
            )
            visual_features = self.mm_projector.mlp(memory_tokens)

        conv = conv_templates[self.conv_template].copy()
        time_msg = self._generate_time_msg(current_time, sampled_count)
        full_question = DEFAULT_IMAGE_TOKEN + "\n" + time_msg + question
        conv.append_message(conv.roles[0], full_question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)
        attention_mask = input_ids.ne(
            self.tokenizer.pad_token_id).long().to(self.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str], self.tokenizer, input_ids)

        with torch.inference_mode():
            dummy_images = [torch.zeros(
                1, 3, video_height, video_width,
                dtype=self.model.dtype, device=self.device
            )]
            image_sizes = [(video_height, video_width)]

            (_, position_ids, attention_mask_prepared, _, inputs_embeds, _) = \
                self.model.prepare_inputs_labels_for_LLM(
                    input_ids, None, attention_mask,
                    None, None, dummy_images,
                    [visual_features], ["video"],
                    image_sizes=image_sizes,
            )

            output_ids = super(type(self.model), self.model).generate(
                position_ids=position_ids,
                attention_mask=attention_mask_prepared,
                inputs_embeds=inputs_embeds,
                do_sample=False if temperature == 0 else True,
                temperature=temperature if temperature > 0 else None,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        full_output = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True).strip()
        assistant_start = full_output.rfind(conv.roles[1])
        if assistant_start != -1:
            response = full_output[assistant_start +
                                   len(conv.roles[1]):].strip()
        else:
            response = full_output
        if response.endswith(stop_str):
            response = response[:-len(stop_str)].strip()
        return response

    def get_anomaly_score_from_cot_generation(
        self,
        memory_manager: MemoryManager,
        question: str,
        current_time: float,
        sampled_count: int,
        video_height: int,
        video_width: int,
        max_new_tokens: int = 512,
        rt_anomaly_tokens: torch.Tensor = None,
    ) -> tuple:
        """
        Deep reasoning CoT scoring: generate structured analysis then extract Yes/No logits.

        Unlike get_anomaly_score_from_logits() which extracts logits at the last prompt token
        (no generation), this method actually generates the full Chain-of-Thought reasoning
        text ([Perception] → [Action Check] → [Causal Cognition] → [Verdict]) and then
        extracts calibrated Yes/No probabilities from the generation logits at the
        [Confirmed Anomaly] position.

        This trades speed for accuracy — the model must complete its full reasoning chain
        before committing to a binary decision, which significantly improves discrimination
        quality for a training-free (zero-shot) reasoning module.

        Args:
            memory_manager: MemoryManager with accumulated visual memory
            question: Formatted CoT prompt (already filled with score_pct and anomaly_context)
            current_time: Current timestamp in seconds
            sampled_count: Number of frames sampled so far
            video_height, video_width: Video resolution for dummy image
            max_new_tokens: Maximum generation length (default 512 for full CoT reasoning)
            rt_anomaly_tokens: Optional [N, H*W, C] real-time anomaly frame features

        Returns:
            tuple of (anomaly_score: float, reasoning_text: str)
            - anomaly_score: P(Yes) from softmax over Yes/No logits at the verdict position
            - reasoning_text: Full generated CoT text for logging/debugging
        """
        with torch.inference_mode():
            memory_tokens = memory_manager.get_memory_tokens(
                rt_anomaly_tokens=rt_anomaly_tokens
            )
            visual_features = self.mm_projector.mlp(memory_tokens)

        # Build prompt (same structure as other methods)
        conv = conv_templates[self.conv_template].copy()
        time_msg = self._generate_time_msg(current_time, sampled_count)
        full_question = DEFAULT_IMAGE_TOKEN + "\n" + time_msg + question
        conv.append_message(conv.roles[0], full_question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)
        attention_mask = input_ids.ne(
            self.tokenizer.pad_token_id).long().to(self.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str], self.tokenizer, input_ids)

        with torch.inference_mode():
            dummy_images = [torch.zeros(
                1, 3, video_height, video_width,
                dtype=self.model.dtype, device=self.device
            )]
            image_sizes = [(video_height, video_width)]

            (_, position_ids, attention_mask_prepared, _, inputs_embeds, _) = \
                self.model.prepare_inputs_labels_for_LLM(
                    input_ids, None, attention_mask,
                    None, None, dummy_images,
                    [visual_features], ["video"],
                    image_sizes=image_sizes,
            )

            # Generate with output_scores=True to capture per-step logits
            outputs = super(type(self.model), self.model).generate(
                position_ids=position_ids,
                attention_mask=attention_mask_prepared,
                inputs_embeds=inputs_embeds,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                output_scores=True,
                return_dict_in_generate=True,
            )

        # ---- Decode generated text ----
        generated_ids = outputs.sequences[0]
        scores = outputs.scores  # tuple of [1, vocab_size] per step
        num_generated = len(scores)

        full_output = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True).strip()

        # Extract assistant response
        assistant_start = full_output.rfind(conv.roles[1])
        if assistant_start != -1:
            reasoning_text = full_output[assistant_start +
                                         len(conv.roles[1]):].strip()
        else:
            reasoning_text = full_output
        if reasoning_text.endswith(stop_str):
            reasoning_text = reasoning_text[:-len(stop_str)].strip()

        # ---- Extract Yes/No probability from generation logits ----
        # Strategy: scan generated tokens backward to find Yes/No at or after
        # the [Confirmed Anomaly] tag, then extract calibrated softmax score.
        gen_ids = outputs.sequences[0, -num_generated:]  # only generated tokens

        # Collect all Yes/No token variants for matching
        yes_ids = {self.positive_token_id, self.positive_lower_token_id}
        no_ids = {self.negative_token_id, self.negative_lower_token_id}
        yes_no_ids = yes_ids | no_ids
        # Remove None entries (if tokenizer doesn't have a variant)
        yes_no_ids.discard(None)
        yes_ids.discard(None)
        no_ids.discard(None)

        anomaly_score = None

        # Primary: search backward for the last Yes/No token in generated sequence
        for i in range(num_generated - 1, -1, -1):
            token_id = gen_ids[i].item()
            if token_id in yes_no_ids:
                # Found a Yes/No token — extract logits at this generation step
                step_logits = scores[i][0]  # [vocab_size]

                # Aggregate logits across capitalization variants
                pos_logit = max(
                    step_logits[self.positive_token_id].float()
                    if self.positive_token_id is not None else torch.tensor(-float('inf')),
                    step_logits[self.positive_lower_token_id].float()
                    if self.positive_lower_token_id is not None else torch.tensor(-float('inf')),
                )
                neg_logit = max(
                    step_logits[self.negative_token_id].float()
                    if self.negative_token_id is not None else torch.tensor(-float('inf')),
                    step_logits[self.negative_lower_token_id].float()
                    if self.negative_lower_token_id is not None else torch.tensor(-float('inf')),
                )

                logits_pair = torch.tensor([pos_logit, neg_logit])
                probs = torch.softmax(logits_pair, dim=0)
                anomaly_score = probs[0].item()
                break

        # Fallback: if no Yes/No token found in generation, parse text
        if anomaly_score is None:
            # logging.warning(
            #     "[CoT] No Yes/No token found in generation. Falling back to text parsing.")
            lower_text = reasoning_text.lower()
            if "[confirmed anomaly]:" in lower_text:
                verdict_part = lower_text.split("[confirmed anomaly]:")[-1].strip()
                if verdict_part.startswith("yes"):
                    anomaly_score = 0.85
                elif verdict_part.startswith("no"):
                    anomaly_score = 0.15
            # Check last 30 chars for yes/no
            if anomaly_score is None:
                tail = lower_text[-30:] if len(lower_text) >= 30 else lower_text
                if "yes" in tail:
                    anomaly_score = 0.7
                elif "no" in tail:
                    anomaly_score = 0.3
                else:
                    anomaly_score = 0.5
                    logging.warning(
                        "[CoT] Could not determine verdict from generated text. Using 0.5.")

        # Free generation scores to reclaim GPU memory
        del scores, outputs
        torch.cuda.empty_cache()

        return anomaly_score, reasoning_text

    def _generate_time_msg(self, current_time: float, num_frames: int) -> str:
        if self.time_msg_style == "short_online":
            return f"\nThe video segment contains {num_frames} frames sampled from the past {current_time:.1f} seconds ago up to the present moment. "
        elif self.time_msg_style == "short_online_v2":
            return f"\nThe video contains {num_frames} frames sampled from the past {current_time:.1f} seconds ago (0.0s of the entire video) up to the present moment ({current_time:.1f}s of the entire video). "
        elif self.time_msg_style == "simple":
            return f"\nYou have watched {current_time:.1f} seconds of the video so far. "
        elif self.time_msg_style == "none":
            return ""
        return ""


# ===================== Combined Pipeline =====================

class CombinedAnomalyDetector:
    """
    Combined PaliGemma (Detection) + StreamForest (Reasoning) pipeline.

    Flow per video:
        - Sample at 4 FPS
        - Every 4 frames (1 sec): create 2x2 grid → PaliGemma detection score
        - Every frame: feed last frame of each group into StreamForest memory
        - If PaliGemma score > threshold → trigger StreamForest verification
        - Final score depends on score_fusion strategy:
            - "replace": SF score completely replaces PG score (original behavior)
            - "weighted": α*PG + (1-α)*SF fixed weighting
            - "adaptive": α scales with SF confidence (high SF → more SF weight)

    Trigger modes:
        - "direct":       Fixed verification prompt (no PG context, fastest)
                          Note: For independent SF assessment, use trigger_mode="score" + prompt_style="neutral" instead
        - "score":        PG score (%) embedded in prompt (no describe_grid, fast)
        - "event":        PG score (%) + top-k event types → embedded in prompt (moderate speed)
        - "description":  PG score (%) + PG description → embedded in prompt (requires describe_grid, slow)
    """

    def __init__(
        self,
        paligemma_detector: PaliGemmaDetector,
        streamforest_reasoner: StreamForestReasoner,
        anomaly_threshold: float = 0.5,
        trigger_mode: str = "direct",  # "direct", "score", "event", or "description"
        paligemma_prompt: str = "",
        sf_prompt_template: str = "",  # Auto-selected prompt template
        sf_prompt_style: str = "default",  # "default", "skeptical", "neutral", "hivau", "cot"
        score_fusion: str = "replace",  # "replace", "weighted", "adaptive"
        fusion_alpha: float = 0.5,  # weight for PG/SF in fusion (meaning depends on strategy)
        sf_scoring_method: str = "binary",  # "binary" or "rating"
        enable_memory_enhancement: bool = False,  # Anomaly Pool + APS (Anomaly-Protected SFTW)
        enable_rt_anomaly: bool = False,  # RT-Anomaly dense 4-frame encoding (independent of Pool/APS)
        pool_threshold: float = 0.6,  # Anomaly Pool insertion threshold
        online_smooth_alpha: float = 0.5,  # EMA coefficient for online smoothing
        online_smooth_beta: float = 0.80,  # Peak decay factor for online smoothing
    ):
        self.paligemma = paligemma_detector
        self.streamforest = streamforest_reasoner
        self.anomaly_threshold = anomaly_threshold
        self.trigger_mode = trigger_mode
        self.paligemma_prompt = paligemma_prompt
        self.sf_prompt_template = sf_prompt_template
        self.sf_prompt_style = sf_prompt_style
        self.score_fusion = score_fusion
        self.fusion_alpha = fusion_alpha
        self.sf_scoring_method = sf_scoring_method
        self.enable_memory_enhancement = enable_memory_enhancement
        self.enable_rt_anomaly = enable_rt_anomaly
        self.pool_threshold = pool_threshold
        self.online_smoother = OnlineSmoother(alpha=online_smooth_alpha, beta=online_smooth_beta)

    def _fuse_scores(self, pg_score: float, sf_score: float) -> float:
        """Compute final score from PG and SF scores based on fusion strategy.

        Strategies:
            replace:  final = sf_score (original behavior)
            weighted: final = α*pg + (1-α)*sf  (fixed α)
            adaptive: α = sf_score (high SF confidence → trust SF more)
        """
        if self.score_fusion == "replace":
            return sf_score
        elif self.score_fusion == "weighted":
            return self.fusion_alpha * pg_score + (1.0 - self.fusion_alpha) * sf_score
        elif self.score_fusion == "adaptive":
            alpha = sf_score
            return (1 - alpha) * pg_score + alpha * sf_score
        else:
            return sf_score

    def detect_video(
        self,
        video_path: str,
        target_fps: int = 4,
        query_interval: int = 4,
        batch_size: int = 1,
        verbose: bool = False,
    ) -> dict:
        """
        Run combined detection on a single video.

        Returns:
            dict with frame_scores, query info, trigger stats, etc.
        """
        # Video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        sample_interval = max(1, int(original_fps / target_fps))
        num_sampled_frames = (
            total_frames + sample_interval - 1) // sample_interval
        num_queries = (num_sampled_frames +
                       query_interval - 1) // query_interval

        if verbose:
            logging.info(f"Video: {video_path}")
            logging.info(
                f"  Original FPS: {original_fps:.1f}, Total frames: {total_frames}")
            logging.info(
                f"  Sample interval: {sample_interval}, Sampled: {num_sampled_frames}, Queries: {num_queries}")

        # Open video and collect frame groups
        cap = cv2.VideoCapture(video_path)

        # Phase 1: Collect all frame groups for PaliGemma batch inference
        #          AND encode last frame of each group for StreamForest memory (cache features)
        query_frame_groups = []  # Each group: list of 4 PIL frames
        query_frame_indices = []  # frame indices per query
        cached_sf_features = []  # Pre-encoded StreamForest vision features per query

        current_group_frames = []
        current_group_indices = []
        sampled_idx = 0

        while sampled_idx < num_sampled_frames:
            original_frame_idx = min(
                sampled_idx * sample_interval, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, original_frame_idx)
            ret, frame = cap.read()
            if not ret:
                sampled_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = PIL.Image.fromarray(frame_rgb)

            current_group_frames.append(pil_frame)
            current_group_indices.append(original_frame_idx)

            # Check if group is complete (query_interval frames)
            if len(current_group_frames) == query_interval or sampled_idx == num_sampled_frames - 1:
                query_frame_groups.append(current_group_frames)
                query_frame_indices.append(current_group_indices)

                # Encode the LAST frame for StreamForest memory (cache to avoid re-encoding)
                last_frame = current_group_frames[-1]
                # Keep the per-query cache on CPU. A long video can otherwise retain
                # hundreds of MiB of vision features on a 24 GB GPU before reasoning
                # even starts.
                frame_features = self.streamforest.encode_frame(last_frame).cpu()
                cached_sf_features.append(frame_features)

                current_group_frames = []
                current_group_indices = []

            sampled_idx += 1

        cap.release()

        # Phase 2: PaliGemma batch detection (fast — all grids at once)
        grid_images = [create_grid_image(frames, GRID_SIZE, self.paligemma.image_size)
                       for frames in query_frame_groups]

        pg_infer_start = time.time()
        paligemma_scores = []
        for batch_start in range(0, len(grid_images), batch_size):
            batch_end = min(batch_start + batch_size, len(grid_images))
            batch_grids = grid_images[batch_start:batch_end]
            batch_scores = self.paligemma.batch_score_grids(
                batch_grids, self.paligemma_prompt)
            paligemma_scores.extend(batch_scores)
        pg_infer_total = time.time() - pg_infer_start
        torch.cuda.empty_cache()

        # Phase 3: Incremental StreamForest memory + triggered deep reasoning
        # Build memory incrementally using cached features for proper temporal context
        # When memory enhancement is enabled: on trigger, encode all 4 frames as RT-Anomaly
        final_scores = []
        trigger_count = 0
        trigger_details = []

        # Granular timing trackers for paper-quality performance analysis
        timing = {
            "pg_infer_total": pg_infer_total,           # Phase 2: PG batch inference total
            "memory_update_times": [],       # Per-query: SFTW + event split + PEMF merge
            "rt_anomaly_encode_times": [],   # Per-trigger: 4-frame batch vision encode
            "sf_inference_times": [],        # Per-trigger: LLM forward (prompt build + generate)
            "anomaly_pool_update_times": [],  # Per-trigger: pool token compression + eviction
            "sf_trigger_total_times": [],    # Per-trigger: total (RT-Anomaly + SF + pool)
        }

        memory_manager = MemoryManager(
            self.streamforest.vision_tower.config.hidden_size,
            self.streamforest.vision_tower.config.num_attention_heads,
            st_memory_windows=[1, 12],
            st_memory_tokens=[729, 128],
            event_split_window=4,
            long_memory_tokens_per_frame=64,
            long_memory_tokens_quota=2048,
            anomaly_pool_max_size=8 if self.enable_memory_enhancement else 0,
            anomaly_pool_tokens=128,
            anomaly_pool_protect_recent=2,
        )
        sf_frame_count = 0

        for query_idx, (pg_score, frame_group, frame_indices) in enumerate(
            zip(paligemma_scores, query_frame_groups, query_frame_indices)
        ):
            sf_features = cached_sf_features[query_idx].to(
                self.streamforest.device
            )
            cached_sf_features[query_idx] = None

            # Always update SFTW memory with the last frame (maintains timeline continuity)
            # When memory enhancement is enabled, use update_with_anomaly_score for APS protection:
            # PG score is used as the anomaly signal for PEMF long-term memory protection
            # (APS uses PG score to maintain training consistency — see design doc)
            mem_update_start = time.time()
            if self.enable_memory_enhancement:
                memory_manager.update_with_anomaly_score(
                    sf_features, anomaly_score=pg_score)
            else:
                memory_manager.update(sf_features)
            timing["memory_update_times"].append(time.time() - mem_update_start)
            sf_frame_count += 1

            current_time = frame_indices[-1] / \
                original_fps if original_fps > 0 else 0.0

            # Separate trigger threshold (controls SF activation) from pool threshold
            triggered = pg_score >= self.anomaly_threshold
            pool_eligible = self.enable_memory_enhancement and pg_score >= self.pool_threshold

            if triggered:
                # Trigger StreamForest deep reasoning
                trigger_count += 1
                description = ""
                sf_trigger_start = time.time()

                # === RT-Anomaly: Dense encode all 4 frames (controlled by enable_rt_anomaly) ===
                rt_anomaly_tokens = None
                if self.enable_rt_anomaly:
                    # Batch encode all frames in one GPU forward pass for efficiency
                    rt_encode_start = time.time()
                    rt_anomaly_tokens = self.streamforest.encode_frames_batch(frame_group)  # [4, 729, C]
                    timing["rt_anomaly_encode_times"].append(time.time() - rt_encode_start)

                # Build SF prompt based on trigger mode
                if self.trigger_mode == "description":
                    # Mode C: PG score (%) + PG description → embedded in prompt
                    # NOTE: describe_grid is an autoregressive generation call (~5x slower)
                    grid_image = grid_images[query_idx]
                    description = self.paligemma.describe_grid(grid_image)
                    score_pct = int(round(pg_score * 100))
                    sf_prompt = self.sf_prompt_template.format(
                        score_pct=score_pct, description=description)
                elif self.trigger_mode == "event":
                    # Mode D: PG score (%) + top-k event types → embedded in prompt
                    # Extracts top-k potential event types with probabilities from PG logits
                    grid_image = grid_images[query_idx]
                    event_result = self.paligemma.extract_event_types_topk(
                        grid_image, self.paligemma_prompt, top_k=3, min_prob_threshold=0.05
                    )
                    score_pct = int(round(pg_score * 100))
                    description = event_result["event_info"]  # Store for logging
                    sf_prompt = self.sf_prompt_template.format(
                        score_pct=score_pct,
                        event_info=event_result["event_info"],
                        event_types_short=event_result["event_types_short"],
                    )
                elif self.trigger_mode == "score":
                    # Mode B: PG score (%) only → embedded in prompt (no describe_grid)
                    score_pct = int(round(pg_score * 100))
                    if self.sf_prompt_style in ("neutral", "hivau"):
                        # Neutral/HIVAU prompts have no placeholders
                        sf_prompt = self.sf_prompt_template
                    elif self.sf_prompt_style in ("skeptical", "cot") and self.enable_memory_enhancement:
                        # Skeptical V3/V4 and CoT: {score_pct} and {anomaly_context} placeholders
                        # anomaly_context provides text-based history from Anomaly Pool
                        anomaly_ctx = memory_manager.get_anomaly_context()
                        sf_prompt = self.sf_prompt_template.format(
                            score_pct=score_pct,
                            anomaly_context=anomaly_ctx["context_str"])
                    elif self.sf_prompt_style in ("skeptical", "cot"):
                        # Without memory enhancement: empty anomaly_context
                        sf_prompt = self.sf_prompt_template.format(
                            score_pct=score_pct, anomaly_context="")
                    else:
                        # Default and CoT prompts have {score_pct} only
                        sf_prompt = self.sf_prompt_template.format(
                            score_pct=score_pct)
                else:
                    # Mode A: Direct verification prompt (no PG context)
                    sf_prompt = self.sf_prompt_template

                # Get SF anomaly score using configured scoring method
                # Enhanced: [PEMF][SFTW][RT-Anomaly(4×729)] + text-based anomaly context
                # Baseline: [PEMF][SFTW][Now]
                free_bytes, _ = torch.cuda.mem_get_info(self.streamforest.device)
                if free_bytes < 512 * 1024 * 1024:
                    # PaliGemma and the vision encoder leave reusable blocks in
                    # separate allocator bins. Release only unused cached blocks
                    # before the large LLM activation allocation.
                    torch.cuda.empty_cache()
                sf_infer_start = time.time()
                cot_reasoning = ""
                if self.sf_prompt_style == "cot":
                    # CoT mode: generate full structured reasoning, then extract Yes/No logits
                    # This is slower but more accurate — SF completes its reasoning chain
                    # before committing to a binary decision.
                    sf_score, cot_reasoning = self.streamforest.get_anomaly_score_from_cot_generation(
                        memory_manager=memory_manager,
                        question=sf_prompt,
                        current_time=current_time,
                        sampled_count=sf_frame_count,
                        video_height=video_height,
                        video_width=video_width,
                        max_new_tokens=512,
                        rt_anomaly_tokens=rt_anomaly_tokens,
                    )
                elif self.sf_scoring_method == "rating":
                    sf_score = self.streamforest.get_anomaly_score_from_rating_logits(
                        memory_manager=memory_manager,
                        question=sf_prompt,
                        current_time=current_time,
                        sampled_count=sf_frame_count,
                        video_height=video_height,
                        video_width=video_width,
                        rt_anomaly_tokens=rt_anomaly_tokens,
                    )
                else:
                    sf_score = self.streamforest.get_anomaly_score_from_logits(
                        memory_manager=memory_manager,
                        question=sf_prompt,
                        current_time=current_time,
                        sampled_count=sf_frame_count,
                        video_height=video_height,
                        video_width=video_width,
                        rt_anomaly_tokens=rt_anomaly_tokens,
                    )
                timing["sf_inference_times"].append(time.time() - sf_infer_start)

                # Update Anomaly Pool moved below, after fused_score is computed

                if rt_anomaly_tokens is not None:
                    del rt_anomaly_tokens

                # Apply score fusion strategy
                fused_score = self._fuse_scores(pg_score, sf_score)
                final_scores.append(fused_score)

                # Update Anomaly Pool with fused_score (SF-verified, more accurate than raw PG)
                # Pool visual tokens are included in get_memory_tokens() at [PEMF][SFTW][Pool][Now/RT].
                # anomaly_pool_scores is used by get_anomaly_context() for V3/V4 prompt text.
                # but the stored score benefits from SF verification via fused_score.
                if pool_eligible:
                    pool_update_start = time.time()
                    memory_manager.update_anomaly_pool(
                        sf_features,
                        # fused_score
                        pg_score
                    )
                    timing["anomaly_pool_update_times"].append(time.time() - pool_update_start)

                # Record SF trigger total latency
                timing["sf_trigger_total_times"].append(time.time() - sf_trigger_start)

                trigger_details.append({
                    "query_idx": query_idx,
                    "paligemma_score": round(pg_score, 4),
                    "streamforest_score": round(sf_score, 4),
                    "fused_score": round(fused_score, 4),
                    "triggered": True,
                    "description": description,
                    # CoT reasoning text (only populated when sf_prompt_style="cot")
                    "cot_reasoning": cot_reasoning[:500] if cot_reasoning else "",
                })

                if verbose:
                    logging.info(
                        f"  Q{query_idx}: PG={pg_score:.4f} -> SF={sf_score:.4f} -> fused={fused_score:.4f}"
                        + (f" desc='{description[:60]}'" if description else "")
                    )
            else:
                # No trigger — use PaliGemma score directly
                final_scores.append(pg_score)

                # Anomaly Pool update for pool-eligible but non-triggered frames
                # (when pool_threshold < trigger_threshold, e.g., pool=0.6, trigger=0.5)
                # Uses raw PG score since no SF verification is available.
                # This ensures Pool reflects the same gating as training (PG >= 0.6).
                if pool_eligible:
                    pool_update_start = time.time()
                    memory_manager.update_anomaly_pool(
                        sf_features,
                        pg_score
                    )
                    timing["anomaly_pool_update_times"].append(time.time() - pool_update_start)

                if verbose and query_idx % 20 == 0:
                    logging.info(
                        f"  Q{query_idx}: PG={pg_score:.4f} (no trigger)")

            # Clear cache periodically
            if query_idx % 30 == 0:
                torch.cuda.empty_cache()
            del sf_features

        # Free cached features
        del cached_sf_features

        query_scores = np.array(final_scores)

        # Online causal smoothing (EMA + Peak Hold) — strictly causal, O(1) per step
        # Applied at query level (1 score per second) before expansion
        self.online_smoother.reset()
        online_smoothed_query = np.array(
            [self.online_smoother.step(s) for s in final_scores]
        )

        # Expand to segment scores (one per sampled frame)
        segment_scores = []
        online_segment_scores = []
        for i, (score, o_score) in enumerate(zip(query_scores, online_smoothed_query)):
            start_idx = i * query_interval
            end_idx = min((i + 1) * query_interval, num_sampled_frames)
            n_expand = end_idx - start_idx
            segment_scores.extend([score] * n_expand)
            online_segment_scores.extend([o_score] * n_expand)
        segment_scores = np.array(segment_scores[:num_sampled_frames])
        online_segment_scores = np.array(online_segment_scores[:num_sampled_frames])

        # Expand to frame scores (one per original frame)
        frame_scores = []
        online_frame_scores = []
        for i, (score, o_score) in enumerate(zip(segment_scores, online_segment_scores)):
            frame_scores.extend([score] * sample_interval)
            online_frame_scores.extend([o_score] * sample_interval)
        frame_scores = np.array(frame_scores[:total_frames])
        online_frame_scores = np.array(online_frame_scores[:total_frames])

        # Gaussian smoothing (post-hoc, non-causal — kept for backward compatibility)
        if len(segment_scores) > 5:
            smoothed_segment_scores = gaussian_smooth_1d(
                segment_scores, sigma=3)
        else:
            smoothed_segment_scores = segment_scores.copy()

        smoothed_frame_scores = []
        for i, score in enumerate(smoothed_segment_scores):
            smoothed_frame_scores.extend([score] * sample_interval)
        smoothed_frame_scores = np.array(smoothed_frame_scores[:total_frames])

        return {
            "query_scores": query_scores.tolist(),
            "paligemma_scores": paligemma_scores,
            "query_frame_indices": query_frame_indices,
            "segment_scores": segment_scores.tolist(),
            "frame_scores": frame_scores.tolist(),
            "smoothed_frame_scores": smoothed_frame_scores.tolist(),
            "online_smoothed_frame_scores": online_frame_scores.tolist(),
            "num_queries": len(query_scores),
            "num_segments": len(segment_scores),
            "total_frames": total_frames,
            "trigger_count": trigger_count,
            "trigger_details": trigger_details,
            "timing": timing,
        }


# ===================== Main =====================

def main():
    parser = argparse.ArgumentParser(
        description="Combined PaliGemma + StreamForest VAD Evaluation")

    # PaliGemma parameters
    parser.add_argument("--paligemma-model-path", type=str, required=True,
                        help="Path to base PaliGemma2 model")
    parser.add_argument("--paligemma-lora-path", type=str, default=None,
                        help="Path to PaliGemma LoRA adapter")
    parser.add_argument("--paligemma-prompt-style", type=str, default="detail",
                        choices=["detail"],
                        help="PaliGemma prompt style used by the current ReactVAU model")
    parser.add_argument("--paligemma-attn", type=str, default="eager",
                        choices=["eager", "sdpa", "flash_attention_2"])

    # PaliGemma 384x384 StreamForest vision encoder support
    parser.add_argument("--paligemma-image-size", type=int, default=448, choices=[384, 448],
                        help="PaliGemma input image size: 448 (original) or 384 (StreamForest)")
    parser.add_argument("--paligemma-streamforest-weights", type=str, default=None,
                        help="Path to StreamForest vision encoder weights (.safetensors). Required when --paligemma-image-size=384")
    parser.add_argument("--paligemma-vision-feature-layer", type=int, default=-2,
                        help="Vision encoder layer for feature extraction (-1=last, -2=second-to-last for StreamForest)")

    # StreamForest parameters
    parser.add_argument("--streamforest-model-path", type=str, required=True,
                        help="Path to StreamForest model checkpoint")
    parser.add_argument("--streamforest-model-base", type=str, default=None,
                        help="Path to StreamForest base model (for LoRA)")
    parser.add_argument("--streamforest-conv-template",
                        type=str, default="qwen_2")
    parser.add_argument("--streamforest-time-msg", type=str, default="short_online_v2",
                        choices=["short_online", "short_online_v2", "simple", "none"])
    parser.add_argument("--streamforest-quantization", type=str, default="none",
                        choices=["none", "4bit", "8bit"],
                        help="StreamForest quantization mode. Default 'none' keeps the original bf16 weights.")

    # Combined pipeline parameters
    parser.add_argument("--anomaly-threshold", type=float, default=0.5,
                        help="PaliGemma score threshold to trigger StreamForest (default: 0.5)")
    parser.add_argument("--trigger-mode", type=str, default="score",
                        choices=["score"],
                        help="Trigger mode used by the current ReactVAU VAD evaluation")
    parser.add_argument("--sf-scoring-method", type=str, default="binary",
                        choices=["binary"],
                        help="SF scoring method used by the current ReactVAU VAD evaluation")
    parser.add_argument("--sf-prompt-style", type=str, default="skeptical",
                        choices=["skeptical"],
                        help="SF prompt style used by the current ReactVAU VAD evaluation")
    parser.add_argument("--score-fusion", type=str, default="adaptive",
                        choices=["replace", "weighted", "adaptive"],
                        help="Score fusion strategy: "
                             "'replace' (SF only), "
                             "'weighted' (alpha*PG + (1-alpha)*SF), "
                             "'adaptive' (alpha=SF_score, high SF → trust SF more)")
    parser.add_argument("--fusion-alpha", type=float, default=0.30,
                        help="Fusion weight α for 'weighted' strategy: PG weight (final = α*PG + (1-α)*SF)")

    # Memory enhancement parameters
    parser.add_argument("--enable-memory-enhancement", action="store_true", default=False,
                        help="Enable Anomaly Pool + APS (Anomaly-Protected SFTW) memory mechanism")
    parser.add_argument("--enable-rt-anomaly", action="store_true", default=False,
                        help="Enable RT-Anomaly dense 4-frame encoding per trigger (independent of --enable-memory-enhancement)")
    parser.add_argument("--pool-threshold", type=float, default=0.6,
                        help="Anomaly Pool insertion threshold (must match training=0.6 for consistency). "
                             "Separate from --anomaly-threshold which controls SF trigger.")

    # Streaming parameters
    parser.add_argument("--target-fps", type=int, default=4)
    parser.add_argument("--query-interval", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for PaliGemma inference")

    # Data parameters
    parser.add_argument("--dataset", type=str,
                        choices=["ucf-crime", "xd-violence"], required=True)
    parser.add_argument("--video-dir", type=str, required=True)
    parser.add_argument("--anno-path", type=str, required=True)

    # Online smoothing parameters
    parser.add_argument("--online-smooth-alpha", type=float, default=0.35,
                        help="EMA coefficient for causal online smoothing (0=no smooth, 1=no memory). Default 0.35.")
    parser.add_argument("--online-smooth-beta", type=float, default=0.88,
                        help="Peak decay factor for causal online smoothing (higher=slower decay). Default 0.88.")

    # Output parameters
    parser.add_argument("--output-path", type=str, default="detection_results")
    parser.add_argument("--save-video-scores", action="store_true")

    # Test parameters
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--test-samples", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Validate 384 mode args
    if args.paligemma_image_size == 384 and not args.paligemma_streamforest_weights:
        parser.error(
            "--paligemma-streamforest-weights is required when --paligemma-image-size is 384")

    # Create output directory
    now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = os.path.join(
        args.output_path,
        f"reactvau-{args.dataset}-{args.trigger_mode}-{args.score_fusion}-thresh{args.anomaly_threshold}-{now_str}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "detection.log")),
            logging.StreamHandler(),
        ]
    )

    logging.info("=" * 70)
    logging.info("ReactVAU Evaluation")
    logging.info("=" * 70)
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"PaliGemma model: {args.paligemma_model_path}")
    logging.info(f"PaliGemma LoRA:  {args.paligemma_lora_path or 'None'}")
    logging.info(f"StreamForest model: {args.streamforest_model_path}")
    logging.info(
        f"StreamForest base:  {args.streamforest_model_base or 'None'}")
    logging.info(f"StreamForest quantization: {args.streamforest_quantization}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Trigger mode: {args.trigger_mode}")
    logging.info(f"Score fusion: {args.score_fusion}" + (
        f" (alpha={args.fusion_alpha})" if args.score_fusion == "weighted" else ""))
    logging.info(f"SF scoring method: {args.sf_scoring_method}")
    logging.info(f"Anomaly threshold: {args.anomaly_threshold}")
    logging.info(
        f"Target FPS: {args.target_fps}, Query interval: {args.query_interval}")
    logging.info(f"PaliGemma batch size: {args.batch_size}")
    logging.info(
        f"PaliGemma image size: {args.paligemma_image_size}x{args.paligemma_image_size}")
    if args.paligemma_streamforest_weights:
        logging.info(
            f"PaliGemma StreamForest weights: {args.paligemma_streamforest_weights}")
        logging.info(
            f"PaliGemma vision feature layer: {args.paligemma_vision_feature_layer}")

    # PaliGemma prompt
    pg_prompt = get_grid_prompt(
        add_special_tokens=False, style=args.paligemma_prompt_style)
    logging.info(f"PaliGemma prompt style: {args.paligemma_prompt_style}")
    logging.info(f"PaliGemma prompt: '{pg_prompt}'")

    # StreamForest prompt - auto-selected based on trigger_mode + sf_scoring_method + prompt_style
    sf_prompt_template = get_sf_prompt(
        args.trigger_mode, args.sf_scoring_method, args.sf_prompt_style)
    logging.info(
        f"SF prompt auto-selected for: trigger_mode={args.trigger_mode}, scoring={args.sf_scoring_method}, style={args.sf_prompt_style}")
    logging.info(f"SF prompt preview: {sf_prompt_template[:200]}...")

    # Load annotation from official anno.txt
    logging.info(f"Loading annotation from {args.anno_path}")
    annotations = load_anno_txt(args.anno_path, args.dataset,
                                video_dir=args.video_dir)

    video_keys = list(annotations.keys())
    if args.test_mode:
        random.seed(42)
        video_keys = random.sample(video_keys, min(
            args.test_samples, len(video_keys)))
        logging.info(f"Test mode: using {len(video_keys)} samples")
    logging.info(f"Total videos to process: {len(video_keys)}")

    # Initialize models. On 24GB GPUs, load StreamForest first so the transient
    # LoRA merge peak does not compete with PaliGemma already resident in VRAM.
    logging.info("Initializing StreamForest reasoner...")
    streamforest = StreamForestReasoner(
        model_path=args.streamforest_model_path,
        model_base=args.streamforest_model_base,
        conv_template=args.streamforest_conv_template,
        device="cuda:0",
        time_msg_style=args.streamforest_time_msg,
        load_4bit=(args.streamforest_quantization == "4bit"),
        load_8bit=(args.streamforest_quantization == "8bit"),
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info(
            f"GPU memory after SF load: {torch.cuda.memory_allocated() / (1024 ** 2):.1f} MB allocated, "
            f"{torch.cuda.memory_reserved() / (1024 ** 2):.1f} MB reserved")

    logging.info("Initializing PaliGemma detector...")
    paligemma = PaliGemmaDetector(
        model_path=args.paligemma_model_path,
        lora_path=args.paligemma_lora_path,
        device="cuda:0",
        attn_implementation=args.paligemma_attn,
        image_size=args.paligemma_image_size,
        streamforest_weights_path=args.paligemma_streamforest_weights,
        vision_feature_layer=args.paligemma_vision_feature_layer,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info(
            f"GPU memory after model load: {torch.cuda.memory_allocated() / (1024 ** 2):.1f} MB allocated, "
            f"{torch.cuda.memory_reserved() / (1024 ** 2):.1f} MB reserved")

    # Combined pipeline
    detector = CombinedAnomalyDetector(
        paligemma_detector=paligemma,
        streamforest_reasoner=streamforest,
        anomaly_threshold=args.anomaly_threshold,
        trigger_mode=args.trigger_mode,
        paligemma_prompt=pg_prompt,
        sf_prompt_template=sf_prompt_template,
        sf_prompt_style=args.sf_prompt_style,
        score_fusion=args.score_fusion,
        fusion_alpha=args.fusion_alpha,
        sf_scoring_method=args.sf_scoring_method,
        enable_memory_enhancement=args.enable_memory_enhancement,
        enable_rt_anomaly=args.enable_rt_anomaly,
        pool_threshold=args.pool_threshold,
        online_smooth_alpha=args.online_smooth_alpha,
        online_smooth_beta=args.online_smooth_beta,
    )

    logging.info(f"Memory enhancement (Pool+APS): {'ENABLED' if args.enable_memory_enhancement else 'DISABLED'}")
    logging.info(f"RT-Anomaly dense encoding: {'ENABLED' if args.enable_rt_anomaly else 'DISABLED'}")
    logging.info(f"Pool threshold: {args.pool_threshold} (training=0.6)")
    logging.info(f"SF prompt style: {args.sf_prompt_style}")
    logging.info(f"Online smoothing: alpha={args.online_smooth_alpha}, beta={args.online_smooth_beta}")

    # Video path format
    if args.dataset == "ucf-crime":
        video_subdir = "videos/ucf-crime/videos/test/"
    else:
        video_subdir = "videos/xd-violence/videos/test/"
    video_ext = ".mp4"

    # Process videos
    all_frame_scores = []
    all_frame_labels = []
    all_smoothed_scores = []
    all_online_smoothed_scores = []
    video_results = {}

    total_queries = 0
    total_segments = 0
    total_triggers = 0

    # Aggregated timing data across all videos
    agg_timing = {
        "pg_infer_total": 0.0,
        "memory_update_times": [],
        "rt_anomaly_encode_times": [],
        "sf_inference_times": [],
        "anomaly_pool_update_times": [],
        "sf_trigger_total_times": [],
    }

    start_time = time.time()

    pbar = tqdm(video_keys, desc="Processing videos", dynamic_ncols=True)
    for video_key in pbar:
        pbar.set_postfix({"video": video_key[:30]}, refresh=True)

        anno = annotations[video_key]
        video_name = anno["video_name"]

        video_path = os.path.join(
            args.video_dir, video_subdir, video_name + video_ext)

        if not os.path.exists(video_path):
            logging.warning(f"Video not found: {video_path}")
            continue

        # Read native fps for GT label creation
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        try:
            result = detector.detect_video(
                video_path=video_path,
                target_fps=args.target_fps,
                query_interval=args.query_interval,
                batch_size=args.batch_size,
                verbose=args.verbose,
            )

            # Frame-level labels from official intervals
            frame_labels = make_gt_labels_from_anno(anno, result["total_frames"], video_fps)

            frame_scores = np.array(result["frame_scores"])
            smoothed_scores = np.array(result["smoothed_frame_scores"])
            online_smoothed_scores = np.array(result["online_smoothed_frame_scores"])

            min_len = min(len(frame_scores), len(frame_labels))
            frame_scores = frame_scores[:min_len]
            smoothed_scores = smoothed_scores[:min_len]
            online_smoothed_scores = online_smoothed_scores[:min_len]
            frame_labels = frame_labels[:min_len]

            all_frame_scores.extend(frame_scores.tolist())
            all_smoothed_scores.extend(smoothed_scores.tolist())
            all_online_smoothed_scores.extend(online_smoothed_scores.tolist())
            all_frame_labels.extend(frame_labels.tolist())

            total_queries += result["num_queries"]
            total_segments += result["num_segments"]
            total_triggers += result["trigger_count"]

            # Aggregate timing data
            vt = result.get("timing", {})
            agg_timing["pg_infer_total"] += vt.get("pg_infer_total", 0.0)
            agg_timing["memory_update_times"].extend(vt.get("memory_update_times", []))
            agg_timing["rt_anomaly_encode_times"].extend(vt.get("rt_anomaly_encode_times", []))
            agg_timing["sf_inference_times"].extend(vt.get("sf_inference_times", []))
            agg_timing["anomaly_pool_update_times"].extend(vt.get("anomaly_pool_update_times", []))
            agg_timing["sf_trigger_total_times"].extend(vt.get("sf_trigger_total_times", []))

            video_results[video_key] = {
                "video_name": video_name,
                "num_queries": result["num_queries"],
                "trigger_count": result["trigger_count"],
                "query_scores": result["query_scores"],
                "paligemma_scores": result["paligemma_scores"],
                "n_frames": result["total_frames"],
                "fps": video_fps,
                "events_frames": anno["intervals_raw"],
                # First 10 for logging
                "trigger_details": result["trigger_details"][:10],
            }

        except Exception as e:
            logging.error(f"Error processing {video_key}: {e}")
            traceback.print_exc()

    elapsed_time = time.time() - start_time

    if not video_results:
        raise RuntimeError(
            "No videos were processed successfully; see the per-video errors above."
        )

    # Compute metrics
    logging.info("\n" + "=" * 70)
    logging.info("Computing metrics...")

    if len(all_frame_scores) > 0 and len(all_frame_labels) > 0:
        metrics_raw = compute_metrics(all_frame_scores, all_frame_labels)
        logging.info(f"Raw Frame Scores:")
        logging.info(f"  ROC-AUC: {metrics_raw['roc_auc']:.4f}")
        logging.info(f"  PR-AUC:  {metrics_raw['pr_auc']:.4f}")
        logging.info(f"  AP:      {metrics_raw['ap']:.4f}")

        metrics_smoothed = compute_metrics(
            all_smoothed_scores, all_frame_labels)
        logging.info(f"\nSmoothed Frame Scores (Gaussian, non-causal):")
        logging.info(f"  ROC-AUC: {metrics_smoothed['roc_auc']:.4f}")
        logging.info(f"  PR-AUC:  {metrics_smoothed['pr_auc']:.4f}")
        logging.info(f"  AP:      {metrics_smoothed['ap']:.4f}")

        metrics_online = compute_metrics(
            all_online_smoothed_scores, all_frame_labels)
        logging.info(f"\nOnline Smoothed Frame Scores (EMA+PeakHold, causal):")
        logging.info(f"  ROC-AUC: {metrics_online['roc_auc']:.4f}")
        logging.info(f"  PR-AUC:  {metrics_online['pr_auc']:.4f}")
        logging.info(f"  AP:      {metrics_online['ap']:.4f}")
        logging.info(f"  Params:  alpha={args.online_smooth_alpha}, beta={args.online_smooth_beta}")

        # Also compute PaliGemma-only metrics for comparison
        pg_only_scores = []
        for vk in video_results:
            pg_scores = video_results[vk]["paligemma_scores"]
            # Expand to frame level using stored n_frames and fps
            n_fr = video_results[vk]["n_frames"]
            orig_fps_v = video_results[vk]["fps"]
            si = max(1, int(orig_fps_v / args.target_fps))
            nsf = (n_fr + si - 1) // si
            seg_sc = []
            for i, sc in enumerate(pg_scores):
                start_i = i * args.query_interval
                end_i = min((i + 1) * args.query_interval, nsf)
                seg_sc.extend([sc] * (end_i - start_i))
            seg_sc = seg_sc[:nsf]
            fr_sc = []
            for sc in seg_sc:
                fr_sc.extend([sc] * si)
            fr_sc = fr_sc[:n_fr]
            pg_only_scores.extend(fr_sc)

        if len(pg_only_scores) == len(all_frame_labels):
            metrics_pg_only = compute_metrics(pg_only_scores, all_frame_labels)
            logging.info(f"\nPaliGemma-Only Scores (for comparison):")
            logging.info(f"  ROC-AUC: {metrics_pg_only['roc_auc']:.4f}")
            logging.info(f"  PR-AUC:  {metrics_pg_only['pr_auc']:.4f}")
            logging.info(f"  AP:      {metrics_pg_only['ap']:.4f}")
        else:
            metrics_pg_only = {}
            logging.info(
                f"\nPaliGemma-Only: length mismatch ({len(pg_only_scores)} vs {len(all_frame_labels)}), skipping")
    else:
        logging.warning("No valid predictions to compute metrics")
        metrics_raw = {}
        metrics_smoothed = {}
        metrics_online = {}
        metrics_pg_only = {}

    # Performance stats
    logging.info(f"\n" + "=" * 70)
    logging.info(f"Performance Statistics:")
    logging.info(f"  Elapsed time: {elapsed_time:.1f}s")
    logging.info(f"  Total videos processed: {len(video_results)}")
    logging.info(f"  Total sampled frames: {total_segments}")
    logging.info(f"  Total original frames: {len(all_frame_scores)}")
    logging.info(f"  Total queries (PaliGemma): {total_queries}")
    logging.info(f"  Total StreamForest triggers: {total_triggers}")
    logging.info(f"  Trigger rate: {total_triggers / max(1, total_queries) * 100:.1f}%")

    # Granular timing breakdown for paper writing
    logging.info(f"\n  --- Timing Breakdown ---")

    # PG inference (amortized per query from batch inference)
    pg_total = agg_timing["pg_infer_total"]
    pg_per_query = pg_total / max(1, total_queries) * 1000
    logging.info(f"  [PG] Total inference time: {pg_total:.1f}s")
    logging.info(f"  [PG] Avg per query: {pg_per_query:.1f}ms")

    # Memory update (every query)
    mem_times = np.array(agg_timing["memory_update_times"]) if agg_timing["memory_update_times"] else np.array([0.0])
    logging.info(f"  [Memory] Avg update per query: {mem_times.mean() * 1000:.1f}ms")

    # Normal query latency: PG (amortized) + memory update
    normal_latency = pg_per_query + mem_times.mean() * 1000
    logging.info(f"  [Normal Query] Avg latency (PG + memory): {normal_latency:.1f}ms")

    # SF trigger timing (only when triggered)
    if agg_timing["sf_trigger_total_times"]:
        sf_total_times = np.array(agg_timing["sf_trigger_total_times"])
        sf_infer_times = np.array(agg_timing["sf_inference_times"])
        logging.info(f"  [SF] Avg total time per trigger: {sf_total_times.mean() * 1000:.1f}ms")
        logging.info(f"  [SF] Avg inference (LLM) per trigger: {sf_infer_times.mean() * 1000:.1f}ms")

        if agg_timing["rt_anomaly_encode_times"]:
            rt_times = np.array(agg_timing["rt_anomaly_encode_times"])
            logging.info(f"  [RT-Anomaly] Avg encode per trigger: {rt_times.mean() * 1000:.1f}ms")

        if agg_timing["anomaly_pool_update_times"]:
            pool_times = np.array(agg_timing["anomaly_pool_update_times"])
            logging.info(f"  [AnomalyPool] Avg update per trigger: {pool_times.mean() * 1000:.1f}ms")

        # Anomaly query latency: PG + memory + SF total
        anomaly_latency = pg_per_query + mem_times.mean() * 1000 + sf_total_times.mean() * 1000
        logging.info(f"  [Anomaly Query] Avg latency (PG + memory + SF): {anomaly_latency:.1f}ms")
        logging.info(f"  [Speedup Ratio] Normal/Anomaly: {anomaly_latency / max(0.01, normal_latency):.1f}x slower")
        

    # Save results
    summary = {
        "model": "combined_paligemma_streamforest",
        "paligemma_model_path": args.paligemma_model_path,
        "paligemma_lora_path": args.paligemma_lora_path,
        "paligemma_image_size": args.paligemma_image_size,
        "paligemma_streamforest_weights": args.paligemma_streamforest_weights,
        "paligemma_vision_feature_layer": args.paligemma_vision_feature_layer,
        "streamforest_model_path": args.streamforest_model_path,
        "streamforest_model_base": args.streamforest_model_base,
        "streamforest_quantization": args.streamforest_quantization,
        "dataset": args.dataset,
        "trigger_mode": args.trigger_mode,
        "score_fusion": args.score_fusion,
        "sf_scoring_method": args.sf_scoring_method,
        "fusion_alpha": args.fusion_alpha if args.score_fusion == "weighted" else None,
        "anomaly_threshold": args.anomaly_threshold,
        "enable_memory_enhancement": args.enable_memory_enhancement,
        "enable_rt_anomaly": args.enable_rt_anomaly,
        "pool_threshold": args.pool_threshold,
        "target_fps": args.target_fps,
        "query_interval": args.query_interval,
        "batch_size": args.batch_size,
        "num_videos": len(video_results),
        "total_queries": total_queries,
        "total_triggers": total_triggers,
        "trigger_rate_pct": round(total_triggers / max(1, total_queries) * 100, 2),
        "total_segments": total_segments,
        "total_frames": len(all_frame_scores),
        "elapsed_time_sec": elapsed_time,
        "timing": {
            "pg_total_sec": round(agg_timing["pg_infer_total"], 3),
            "pg_avg_per_query_ms": round(agg_timing["pg_infer_total"] / max(1, total_queries) * 1000, 1),
            "memory_avg_per_query_ms": round(float(np.mean(agg_timing["memory_update_times"]) * 1000) if agg_timing["memory_update_times"] else 0, 1),
            "normal_query_avg_ms": round(
                agg_timing["pg_infer_total"] / max(1, total_queries) * 1000 +
                (float(np.mean(agg_timing["memory_update_times"]) * 1000) if agg_timing["memory_update_times"] else 0), 1),
            "sf_trigger_avg_ms": round(float(np.mean(agg_timing["sf_trigger_total_times"]) * 1000) if agg_timing["sf_trigger_total_times"] else 0, 1),
            "sf_inference_avg_ms": round(float(np.mean(agg_timing["sf_inference_times"]) * 1000) if agg_timing["sf_inference_times"] else 0, 1),
            "rt_anomaly_encode_avg_ms": round(float(np.mean(agg_timing["rt_anomaly_encode_times"]) * 1000) if agg_timing["rt_anomaly_encode_times"] else 0, 1),
            "anomaly_pool_update_avg_ms": round(float(np.mean(agg_timing["anomaly_pool_update_times"]) * 1000) if agg_timing["anomaly_pool_update_times"] else 0, 1),
        },
        "metrics_raw": {k: v for k, v in metrics_raw.items() if k not in ["fpr", "tpr"]},
        "metrics_smoothed": {k: v for k, v in metrics_smoothed.items() if k not in ["fpr", "tpr"]},
        "metrics_online_smoothed": {k: v for k, v in metrics_online.items() if k not in ["fpr", "tpr"]} if metrics_online else {},
        "online_smooth_alpha": args.online_smooth_alpha,
        "online_smooth_beta": args.online_smooth_beta,
        "metrics_pg_only": {k: v for k, v in metrics_pg_only.items() if k not in ["fpr", "tpr"]} if metrics_pg_only else {},
        "paligemma_prompt": pg_prompt,
        "sf_prompt_template": sf_prompt_template,
        "timestamp": now_str,
    }

    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    if args.save_video_scores:
        with open(os.path.join(output_dir, "video_results.json"), 'w') as f:
            json.dump(video_results, f, indent=2)

    logging.info(f"\nResults saved to {output_dir}")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
