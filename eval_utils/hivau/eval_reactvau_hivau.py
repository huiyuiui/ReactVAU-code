"""
ReactVAU HIVAU Understanding Evaluation (PaliGemma + StreamForest)

Architecture:
    PaliGemma2-3B (Detection Module) + StreamForest-Qwen2-7B (Reasoning Module)

Pipeline for Understanding:
    1. Video sampled at target FPS (default 4).
       Every query_interval frames (default 4, = 1 second) → 2x2 grid → PaliGemma detection.
       Simultaneously, the last frame of each group is fed into StreamForest's MemoryManager.
    2. After full video processing:
       - PaliGemma provides detection context (anomaly segments, optional descriptions)
       - StreamForest generates understanding response using accumulated visual memory + PG context
    3. Metrics: BLEU, ROUGE, CIDEr, METEOR

Context Injection Modes (--context-mode):
    - "none":        No PG context injected (equivalent to standalone StreamForest)
    - "score":       PG anomaly scores summary embedded in prompt
    - "description": PG scores + descriptions embedded in prompt (slower, requires describe_grid)
"""
import sys
import os
import argparse
import json
import datetime
from tqdm import tqdm
from pathlib import Path
import logging
import time
import random
import warnings
import traceback

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import PIL.Image
from PIL import Image

# Add ReactVAU path dynamically
ReactVAU_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if ReactVAU_ROOT not in sys.path:
    sys.path.insert(0, ReactVAU_ROOT)

from vad.get_prompt import get_grid_prompt, PALIGEMMA_DESCRIBE_PROMPT
import types
from safetensors.torch import load_file as safetensors_load_file
from peft import PeftModel
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from hivau_utils import hivau_BLEU, hivau_ROUGE, hivau_CIDEr, hivau_METEOR
from llava.model.multimodal_projector.memory_manager import MemoryManager
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from llava.model.builder import load_pretrained_model

# Add current directory for hivau_utils
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# PaliGemma imports

# Import grid/prompt utilities from VAD

warnings.filterwarnings('ignore')


# ===================== Grid Image Creation (same as VAD eval) =====================
GRID_SIZE = (2, 2)
USE_SEPARATOR = True
SEPARATOR_WIDTH = 2
SEPARATOR_COLOR = (128, 128, 128)


def compute_grid_params(image_size: int = 448):
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

        if image_size == 384 and streamforest_weights_path and vision_feature_layer != -1:
            self.model = patch_vision_feature_layer(
                self.model, select_layer=vision_feature_layer)

        self.model.eval()

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
            )
            logits = outputs.logits
            batch_scores = []
            for batch_idx in range(len(valid_images)):
                attention = inputs["attention_mask"][batch_idx]
                last_pos = attention.sum().item() - 1
                last_logits = logits[batch_idx, last_pos]
                yes_logit = max(last_logits[self.yes_token_id].float(),
                                last_logits[self.yes_lower_token_id].float())
                no_logit = max(last_logits[self.no_token_id].float(),
                               last_logits[self.no_lower_token_id].float())
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
    ):
        self.conv_template = conv_template
        self.device = torch.device(device)
        self.time_msg_style = time_msg_style

        if model_base is not None and model_base.strip() == "":
            model_base = None

        logging.info(f"[StreamForest] Loading model from {model_path}...")
        if model_base:
            logging.info(f"[StreamForest] Base model: {model_base}")

        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            model_path,
            model_base=model_base,
            model_name=model_name,
            device_map=device,
            multimodal=True,
            attn_implementation="sdpa",
        )
        self.model.eval()

        self.vision_tower = self.model.get_vision_tower()
        self.mm_projector = self.model.get_model().mm_projector

        if self.tokenizer.pad_token_id is None:
            if "qwen" in self.tokenizer.name_or_path.lower():
                self.tokenizer.pad_token_id = 151643

        logging.info(f"[StreamForest] Model loaded successfully")

    def encode_frame(self, pil_frame: PIL.Image.Image) -> torch.Tensor:
        """Encode a single frame through the vision tower. Returns [H*W, C] features."""
        with torch.inference_mode():
            frame_tensor = self.image_processor.preprocess(
                [pil_frame], return_tensors="pt"
            )["pixel_values"].to(dtype=self.model.dtype, device=self.device)
            frame_features = self.vision_tower(frame_tensor)
            return frame_features[0]  # [H*W, C]

    def generate_text(
        self,
        memory_manager: MemoryManager,
        question: str,
        current_time: float,
        sampled_count: int,
        video_height: int,
        video_width: int,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate text response using accumulated visual memory."""
        with torch.inference_mode():
            memory_tokens = memory_manager.get_memory_tokens()
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


# ===================== ReactVAU Understanding Pipeline =====================
class ReactVAUUnderstanding:
    """
    Combined PaliGemma (Detection) + StreamForest (Reasoning) pipeline for understanding.

    Flow per video:
        Phase 1 — Video Processing:
            - Sample at target FPS
            - Every query_interval frames (1 sec): create 2x2 grid → PaliGemma detection
            - Simultaneously: feed last frame of each group into StreamForest memory
        Phase 2 — Context Building:
            - Collect PaliGemma detection context (anomaly segments, optional descriptions)
            - Build enhanced prompt: PG context + original question
        Phase 3 — Answer Generation:
            - StreamForest generates text answer using accumulated visual memory
    """

    def __init__(
        self,
        paligemma_detector: PaliGemmaDetector,
        streamforest_reasoner: StreamForestReasoner,
        anomaly_threshold: float = 0.5,
        context_mode: str = "score",
        paligemma_prompt: str = "",
        max_context_segments: int = 10,
        enable_memory_enhancement: bool = False,
        sf_prompt_style: str = "default",
    ):
        self.paligemma = paligemma_detector
        self.streamforest = streamforest_reasoner
        self.anomaly_threshold = anomaly_threshold
        self.context_mode = context_mode
        self.paligemma_prompt = paligemma_prompt
        self.max_context_segments = max_context_segments
        self.enable_memory_enhancement = enable_memory_enhancement
        self.sf_prompt_style = sf_prompt_style

    def _build_pg_context_string(self, anomaly_segments: list, video_duration: float = 0.0) -> str:
        """Build PaliGemma context string for enhanced prompt.
        
        When sf_prompt_style is "skeptical", uses stronger de-biasing language
        aligned with the V3 skeptical prompts that proved effective on VAD.
        PG info is presented as auxiliary reference, not ground truth.
        
        Args:
            anomaly_segments: list of detected anomaly segments
            video_duration: total video duration in seconds
        """
        is_skeptical = self.sf_prompt_style == "skeptical"
        
        if not anomaly_segments:
            if is_skeptical:
                return (
                    "[Detection Module Reference]\n"
                    "A lightweight detection module has pre-scanned this video. "
                    "No segments were flagged as potentially anomalous.\n"
                    "Note: This detector frequently produces false alarms and may also miss "
                    "subtle or context-dependent anomalies. Use your own visual analysis as primary evidence."
                )
            return (
                "[Detection Module Reference]\n"
                "A lightweight detection module has pre-scanned this video. "
                "No segments were flagged as potentially anomalous.\n"
                "Note: The detector may miss subtle or context-dependent anomalies."
            )

        n = len(anomaly_segments)
        # Sort by score descending, take top segments
        top_segments = sorted(anomaly_segments, key=lambda x: x["score"], reverse=True)[
            :self.max_context_segments]
        # Re-sort by time for temporal order
        top_segments = sorted(top_segments, key=lambda x: x["time"])

        duration_str = f" (video duration: {video_duration:.1f}s)" if video_duration > 0 else ""

        # Skeptical framing: aligned with the current ReactVAU VAD verification prompt.
        if is_skeptical:
            false_alarm_note = (
                "Note: This detector frequently produces false alarms from camera motion, "
                "lighting changes, and normal fast movements. Many of these alerts may be false alarms. "
                "Do not rely solely on these detections — use your own visual analysis as primary evidence."
            )
        else:
            false_alarm_note = (
                "Note: The detector may produce false positives. "
                "Use your own visual analysis as primary evidence."
            )

        if self.context_mode == "description":
            context = (
                f"[Detection Module Reference]\n"
                f"A lightweight detection module has pre-scanned this video{duration_str}. "
                f"{n} segment(s) were flagged as potentially anomalous:\n"
            )
            for seg in top_segments:
                context += f"  - t={seg['time']:.1f}s (confidence: {seg['score']*100:.0f}%): {seg['description']}\n"
            context += false_alarm_note
        elif self.context_mode == "score":
            context = (
                f"[Detection Module Reference]\n"
                f"A lightweight detection module has pre-scanned this video{duration_str}. "
                f"{n} segment(s) were flagged as potentially anomalous. "
            )
            time_strs = [
                f"t={seg['time']:.1f}s ({seg['score']*100:.0f}%)" for seg in top_segments]
            context += "Flagged timestamps: " + ", ".join(time_strs) + ".\n"
            context += false_alarm_note
        else:
            context = ""
        return context

    def _build_enhanced_question(self, pg_context_str: str, question: str, task: str = "") -> str:
        """Combine PG context with HIVAU's original question.
        
        The HIVAU dataset prompt is always preserved as the PRIMARY question at the end.
        PG context is added as auxiliary reference information above.
        
        When sf_prompt_style is "skeptical", the bridging text encourages SF to
        independently verify using its own visual memory, aligned with the skeptical
        V3 prompt spirit that proved effective on VAD.
        
        Args:
            pg_context_str: PaliGemma detection context (may be empty)
            question: Original HIVAU dataset prompt (must not be modified)
            task: HIVAU task type (judgement/description/analysis/caption)
        """
        if not pg_context_str:
            return question

        is_skeptical = self.sf_prompt_style == "skeptical"

        # For caption tasks, PG anomaly context is less relevant
        # Use lighter framing to avoid distracting from pure description
        if task == "caption":
            return (
                f"{pg_context_str}\n\n"
                f"Based on the video content, answer the following:\n"
                f"{question}"
            )

        # For judgement/description/analysis tasks, PG context is directly useful
        if is_skeptical:
            # Skeptical bridging: encourage independent visual analysis
            return (
                f"{pg_context_str}\n\n"
                f"You have been monitoring this video from the beginning and maintain continuous visual memory. "
                f"Using the above detection reference as auxiliary information and your own independent visual analysis "
                f"as primary evidence, answer the following:\n"
                f"{question}"
            )
        return (
            f"{pg_context_str}\n\n"
            f"Using the above detection reference and your own visual analysis of the video, "
            f"answer the following:\n"
            f"{question}"
        )

    def process_video_and_generate(
        self,
        video_path: str,
        question: str,
        target_fps: int = 4,
        query_interval: int = 4,
        batch_size: int = 32,
        max_new_tokens: int = 512,
        task: str = "",
    ) -> dict:
        """
        Process a video through the combined pipeline and generate an understanding response.

        Returns:
            dict with response, pg_context, and stats
        """
        # ---- Video Info ----
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if original_fps <= 0:
            original_fps = 30.0

        sample_interval = max(1, int(original_fps / target_fps))
        num_sampled_frames = (
            total_frames + sample_interval - 1) // sample_interval
        video_duration = total_frames / original_fps

        # ---- Phase 1: Collect frame groups + encode SF features ----
        cap = cv2.VideoCapture(video_path)
        query_frame_groups = []
        cached_sf_features = []
        query_times = []

        current_group_frames = []
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

            if len(current_group_frames) == query_interval or sampled_idx == num_sampled_frames - 1:
                query_frame_groups.append(current_group_frames)
                query_time = original_frame_idx / original_fps
                query_times.append(query_time)

                # Encode last frame for StreamForest memory
                last_frame = current_group_frames[-1]
                frame_features = self.streamforest.encode_frame(last_frame)
                cached_sf_features.append(frame_features)

                current_group_frames = []

            sampled_idx += 1

        cap.release()

        # ---- Phase 2a: PaliGemma batch detection ----
        grid_images = [create_grid_image(frames, GRID_SIZE, self.paligemma.image_size)
                       for frames in query_frame_groups]

        paligemma_scores = []
        for batch_start in range(0, len(grid_images), batch_size):
            batch_end = min(batch_start + batch_size, len(grid_images))
            batch_grids = grid_images[batch_start:batch_end]
            batch_scores = self.paligemma.batch_score_grids(
                batch_grids, self.paligemma_prompt)
            paligemma_scores.extend(batch_scores)
            torch.cuda.empty_cache()

        # ---- Phase 2b: Build PG context ----
        anomaly_segments = []
        for query_idx, (pg_score, query_time) in enumerate(zip(paligemma_scores, query_times)):
            if pg_score >= self.anomaly_threshold:
                seg_info = {
                    "query_idx": query_idx,
                    "time": query_time,
                    "score": pg_score,
                    "description": "",
                }
                # Generate description if needed
                if self.context_mode == "description":
                    seg_info["description"] = self.paligemma.describe_grid(
                        grid_images[query_idx])
                anomaly_segments.append(seg_info)

        # Free grid images
        del grid_images

        # ---- Phase 3: Build StreamForest memory incrementally ----
        # When memory enhancement is enabled:
        #   - Use update_with_anomaly_score() for APS (Anomaly Priority Score) protection
        #     in PEMF long memory, preventing anomaly-containing events from being merged away
        #   - Update Anomaly Pool for high-score frames (visual tokens included in get_memory_tokens)
        # Pool threshold must match training (projector_FSTW_PEMF.py anomaly_pool_threshold=0.6).
        # V2 training uses threshold=0.6 for cleaner Pool signal.
        # This is intentionally separate from self.anomaly_threshold which controls PG context building.
        anomaly_pool_threshold = 0.6
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
        for query_idx in range(len(cached_sf_features)):
            pg_score = paligemma_scores[query_idx]

            # Memory update: use APS-enhanced version when memory enhancement is enabled
            # APS protects anomaly-containing events from being merged/compressed in PEMF
            if self.enable_memory_enhancement:
                memory_manager.update_with_anomaly_score(
                    cached_sf_features[query_idx], anomaly_score=pg_score)

                # Add high-score frames to Anomaly Pool (visual tokens for LLM context)
                if pg_score >= anomaly_pool_threshold:
                    memory_manager.update_anomaly_pool(
                        cached_sf_features[query_idx], pg_score)
            else:
                memory_manager.update(cached_sf_features[query_idx])
            sf_frame_count += 1

            # Periodically clear cache
            if query_idx % 30 == 0:
                torch.cuda.empty_cache()

        # Free cached features
        del cached_sf_features

        # ---- Phase 4: Build enhanced prompt + Generate ----
        pg_context_str = ""
        if self.context_mode != "none":
            pg_context_str = self._build_pg_context_string(
                anomaly_segments, video_duration)

        # Build enhanced question: PG auxiliary context + HIVAU original question (preserved at end)
        enhanced_question = self._build_enhanced_question(pg_context_str, question, task)

        # Generate response
        response = self.streamforest.generate_text(
            memory_manager=memory_manager,
            question=enhanced_question,
            current_time=video_duration,
            sampled_count=sf_frame_count,
            video_height=video_height,
            video_width=video_width,
            max_new_tokens=max_new_tokens,
        )

        # Cleanup
        del memory_manager
        torch.cuda.empty_cache()

        return {
            "response": response,
            "pg_context": pg_context_str,
            "num_queries": len(paligemma_scores),
            "num_anomaly_segments": len(anomaly_segments),
            "anomaly_segments": anomaly_segments,
            "paligemma_scores": paligemma_scores,
            "sf_frame_count": sf_frame_count,
            "video_duration": video_duration,
        }


# ===================== Data Loading =====================
def load_hivau_data(anno_path):
    """Load HIVAU annotation data."""
    instances = []
    with open(anno_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                instances.append(item)
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping invalid JSON: {e}")

    logging.info(f"Loaded {len(instances)} HIVAU samples from {anno_path}")
    return instances


def validate_prediction(pred: str, answer: str, instance: dict) -> dict:
    """Validate prediction quality."""
    issues = []
    if not pred or pred.strip() == "":
        issues.append("EMPTY_RESPONSE")
    if len(pred) < 10:
        issues.append("TOO_SHORT")
    elif len(pred) > 1024:
        issues.append("TOO_LONG")
    if instance['prompt'][:50].lower() in pred.lower():
        issues.append("CONTAINS_QUESTION")
    if len(set(pred.replace(" ", ""))) < 10:
        issues.append("REPETITIVE")
    len_ratio = len(pred) / max(len(answer), 1)
    if len_ratio < 0.1:
        issues.append(f"MUCH_SHORTER_THAN_GT({len_ratio:.2f})")
    elif len_ratio > 10:
        issues.append(f"MUCH_LONGER_THAN_GT({len_ratio:.2f})")
    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "pred_len": len(pred),
        "answer_len": len(answer),
        "len_ratio": len_ratio,
    }


# ===================== Main =====================
def main():
    parser = argparse.ArgumentParser(
        description="ReactVAU HIVAU Understanding Evaluation (PaliGemma + StreamForest)")

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
    parser.add_argument("--paligemma-image-size", type=int, default=448, choices=[384, 448],
                        help="PaliGemma input image size")
    parser.add_argument("--paligemma-streamforest-weights", type=str, default=None,
                        help="Path to StreamForest vision encoder weights (.safetensors)")
    parser.add_argument("--paligemma-vision-feature-layer", type=int, default=-2,
                        help="Vision encoder layer for feature extraction")
    parser.add_argument("--paligemma-batch-size", type=int, default=32,
                        help="Batch size for PaliGemma inference")

    # StreamForest parameters
    parser.add_argument("--streamforest-model-path", type=str, required=True,
                        help="Path to StreamForest model checkpoint")
    parser.add_argument("--streamforest-model-base", type=str, default=None,
                        help="Path to StreamForest base model (for LoRA)")
    parser.add_argument("--streamforest-conv-template",
                        type=str, default="qwen_2")
    parser.add_argument("--streamforest-time-msg", type=str, default="short_online_v2",
                        choices=["short_online", "short_online_v2", "simple", "none"])

    # Combined pipeline parameters
    parser.add_argument("--anomaly-threshold", type=float, default=0.5,
                        help="PaliGemma score threshold for anomaly context")
    parser.add_argument("--context-mode", type=str, default="score",
                        choices=["none", "score", "description"],
                        help="PG context injection mode: "
                             "'none' (no PG context, same as standalone SF), "
                             "'score' (PG anomaly scores in prompt), "
                             "'description' (PG scores + descriptions, slower)")
    parser.add_argument("--enable-memory-enhancement", action="store_true",
                        help="Enable APS (Anomaly Priority Score) protection in PEMF "
                             "long memory and Anomaly Pool text-based context injection. "
                             "Recommended with --sf-prompt-style skeptical.")
    parser.add_argument("--sf-prompt-style", type=str, default="default",
                        choices=["default", "skeptical"],
                        help="SF prompt style for PG context framing: "
                             "'default' (neutral framing), "
                             "'skeptical' (emphasize false-alarm rate, encourage independent analysis). "
                             "Recommended with --enable-memory-enhancement.")

    # Streaming parameters
    parser.add_argument("--target-fps", type=int, default=4,
                        help="Target FPS for video sampling")
    parser.add_argument("--query-interval", type=int, default=4,
                        help="Frames per PaliGemma query (default: 4 = 1 sec at 4 FPS)")

    # Data parameters
    parser.add_argument("--video-dir", type=str, required=True,
                        help="HIVAU video directory (e.g., /path/to/HIVAU-70k)")
    parser.add_argument("--anno-path", type=str, required=True,
                        help="Path to annotation file (.jsonl)")

    # Output parameters
    parser.add_argument("--output-path", type=str, default="results",
                        help="Base output directory")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save individual predictions")

    # Test parameters
    parser.add_argument("--test-mode", action="store_true",
                        help="Run in test mode (sample N random examples)")
    parser.add_argument("--test-samples", type=int, default=10,
                        help="Number of samples in test mode")
    parser.add_argument("--test-seed", type=int, default=42,
                        help="Random seed for test mode sampling")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output for each sample")

    args = parser.parse_args()

    # Validate 384 mode args
    if args.paligemma_image_size == 384 and not args.paligemma_streamforest_weights:
        parser.error(
            "--paligemma-streamforest-weights is required when --paligemma-image-size is 384")

    # Create output directory with timestamp
    now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    context_tag = args.context_mode
    mem_tag = "-memenhanced" if args.enable_memory_enhancement else ""
    style_tag = f"-{args.sf_prompt_style}" if args.sf_prompt_style != "default" else ""
    run_output_dir = os.path.join(
        args.output_path,
        f"reactvau-hivau-{context_tag}{style_tag}{mem_tag}-thresh{args.anomaly_threshold}-{now_str}"
    )
    os.makedirs(run_output_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(run_output_dir, "evaluation.log")

    class TqdmLoggingHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except Exception:
                self.handleError(record)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'))

    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info("=" * 70)
    logging.info("ReactVAU HIVAU Understanding Evaluation")
    logging.info("=" * 70)
    logging.info(f"Output directory: {run_output_dir}")
    logging.info(f"Log file: {log_file}")
    logging.info("")
    logging.info(f"PaliGemma model: {args.paligemma_model_path}")
    logging.info(f"PaliGemma LoRA:  {args.paligemma_lora_path or 'None'}")
    logging.info(
        f"PaliGemma image size: {args.paligemma_image_size}x{args.paligemma_image_size}")
    if args.paligemma_streamforest_weights:
        logging.info(
            f"PaliGemma SF weights: {args.paligemma_streamforest_weights}")
        logging.info(
            f"PaliGemma vision feature layer: {args.paligemma_vision_feature_layer}")
    logging.info(f"PaliGemma batch size: {args.paligemma_batch_size}")
    logging.info("")
    logging.info(f"StreamForest model: {args.streamforest_model_path}")
    logging.info(
        f"StreamForest base:  {args.streamforest_model_base or 'None'}")
    logging.info(f"StreamForest conv:  {args.streamforest_conv_template}")
    logging.info(f"StreamForest time:  {args.streamforest_time_msg}")
    logging.info("")
    logging.info(f"Context mode: {args.context_mode}")
    logging.info(f"SF prompt style: {args.sf_prompt_style}")
    logging.info(f"Memory enhancement: {'ENABLED' if args.enable_memory_enhancement else 'DISABLED'}")
    logging.info(f"Anomaly threshold: {args.anomaly_threshold}")
    logging.info(f"Target FPS: {args.target_fps}")
    logging.info(f"Query interval: {args.query_interval}")
    logging.info("")
    logging.info(f"Video dir: {args.video_dir}")
    logging.info(f"Annotation: {args.anno_path}")
    logging.info("=" * 70)

    # PaliGemma prompt
    pg_prompt = get_grid_prompt(
        add_special_tokens=False, style=args.paligemma_prompt_style)
    logging.info(f"PaliGemma prompt style: {args.paligemma_prompt_style}")
    logging.info(f"PaliGemma prompt: '{pg_prompt}'")

    # ---- Initialize Models ----
    logging.info("\nInitializing PaliGemma detector...")
    paligemma = PaliGemmaDetector(
        model_path=args.paligemma_model_path,
        lora_path=args.paligemma_lora_path,
        device="cuda:0",
        attn_implementation=args.paligemma_attn,
        image_size=args.paligemma_image_size,
        streamforest_weights_path=args.paligemma_streamforest_weights,
        vision_feature_layer=args.paligemma_vision_feature_layer,
    )

    logging.info("\nInitializing StreamForest reasoner...")
    streamforest = StreamForestReasoner(
        model_path=args.streamforest_model_path,
        model_base=args.streamforest_model_base,
        conv_template=args.streamforest_conv_template,
        device="cuda:0",
        time_msg_style=args.streamforest_time_msg,
    )

    # Combined pipeline
    pipeline = ReactVAUUnderstanding(
        paligemma_detector=paligemma,
        streamforest_reasoner=streamforest,
        anomaly_threshold=args.anomaly_threshold,
        context_mode=args.context_mode,
        paligemma_prompt=pg_prompt,
        enable_memory_enhancement=args.enable_memory_enhancement,
        sf_prompt_style=args.sf_prompt_style,
    )

    # ---- Load Data ----
    logging.info("\nLoading HIVAU data...")
    instances = load_hivau_data(args.anno_path)

    if args.test_mode:
        random.seed(args.test_seed)
        instances = random.sample(instances, min(
            args.test_samples, len(instances)))
        logging.info(f"TEST MODE: Sampling {len(instances)} random examples")
    else:
        logging.info(f"FULL EVALUATION MODE: {len(instances)} samples")

    # ---- Run Inference ----
    logging.info(
        f"\nStarting inference (ReactVAU, context_mode={args.context_mode})...")
    eval_results = []
    predictions = []
    validation_stats = {
        "total": 0, "valid": 0, "empty": 0, "errors": 0, "issues": {}
    }
    total_triggers = 0
    total_queries = 0

    inference_start_time = time.time()

    pbar = tqdm(instances, desc="Evaluating",
                dynamic_ncols=True, leave=True, file=sys.stderr)

    for idx, instance in enumerate(pbar):
        video_rel_path = instance['video']
        video_path = os.path.join(args.video_dir, "videos", video_rel_path)

        if not os.path.exists(video_path):
            logging.warning(f"Video not found: {video_path}")
            predictions.append({
                "video": video_rel_path,
                "question": instance['prompt'],
                "pred": "",
                "answer": instance['anwser'],
                "error": "Video file not found"
            })
            validation_stats["errors"] += 1
            continue

        question = instance['prompt']
        ground_truth = instance['anwser']
        task_type = instance.get('type', 'video')
        task_name = instance.get('task', 'description')

        try:
            result = pipeline.process_video_and_generate(
                video_path=video_path,
                question=question,
                target_fps=args.target_fps,
                query_interval=args.query_interval,
                batch_size=args.paligemma_batch_size,
                task=task_name,
            )

            response = result["response"]
            total_queries += result["num_queries"]
            total_triggers += result["num_anomaly_segments"]

            # Validate
            validation = validate_prediction(response, ground_truth, instance)
            validation_stats["total"] += 1
            if validation["is_valid"]:
                validation_stats["valid"] += 1
            else:
                for issue in validation["issues"]:
                    validation_stats["issues"][issue] = validation_stats["issues"].get(
                        issue, 0) + 1
            if not response or response.strip() == "":
                validation_stats["empty"] += 1

            prediction = {
                "video": video_rel_path,
                "question": question,
                "pred": response,
                "answer": ground_truth,
                "type": task_type,
                "task": task_name,
                "pg_context": result["pg_context"],
                "num_anomaly_segments": result["num_anomaly_segments"],
            }
            predictions.append(prediction)

            eval_result = {
                'pred': response,
                'answer': ground_truth,
                'type': task_type,
                'task': task_name,
            }
            eval_results.append(eval_result)

            if args.verbose:
                logging.info(f"\n--- Sample {idx} ---")
                logging.info(f"Video: {video_rel_path}")
                logging.info(f"Question: {question}")
                logging.info(
                    f"PG anomaly segments: {result['num_anomaly_segments']}/{result['num_queries']}")
                logging.info(f"Response: {response[:200]}...")
                logging.info(f"GT: {ground_truth[:200]}...")

        except Exception as e:
            error_msg = f"Error processing sample {idx} ({video_rel_path}): {str(e)}"
            logging.error(error_msg)
            if args.verbose:
                traceback.print_exc()

            predictions.append({
                "video": video_rel_path,
                "question": question,
                "pred": "",
                "answer": ground_truth,
                "type": task_type,
                "task": task_name,
                "error": str(e),
            })
            validation_stats["errors"] += 1

    inference_end_time = time.time()
    inference_duration = str(datetime.timedelta(
        seconds=int(inference_end_time - inference_start_time)))

    logging.info(f"\nInference finished in {inference_duration}")

    # ---- Validation Statistics ----
    logging.info("\n" + "=" * 70)
    logging.info("Validation Statistics:")
    logging.info("=" * 70)
    logging.info(f"Total samples:     {validation_stats['total']}")
    logging.info(
        f"Valid responses:   {validation_stats['valid']} ({validation_stats['valid']/max(validation_stats['total'],1)*100:.1f}%)")
    logging.info(
        f"Empty responses:   {validation_stats['empty']} ({validation_stats['empty']/max(validation_stats['total'],1)*100:.1f}%)")
    logging.info(f"Errors:            {validation_stats['errors']}")
    logging.info(f"Total PG queries:  {total_queries}")
    logging.info(f"Total anomaly segments: {total_triggers}")
    if total_queries > 0:
        logging.info(
            f"Anomaly rate:      {total_triggers/total_queries*100:.1f}%")

    if validation_stats["issues"]:
        logging.info(f"\nQuality Issues Detected:")
        for issue, count in sorted(validation_stats["issues"].items(), key=lambda x: -x[1]):
            logging.info(
                f"   {issue}: {count} samples ({count/max(validation_stats['total'],1)*100:.1f}%)")
    logging.info("=" * 70)

    # ---- Save Predictions ----
    if args.save_predictions:
        predictions_file = os.path.join(run_output_dir, "predictions.json")
        with open(predictions_file, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        logging.info(f"\nPredictions saved to {predictions_file}")

    # ---- Compute Metrics ----
    logging.info("\nComputing metrics...")
    logging.info("=" * 50)

    class MetricArgs:
        def __init__(self, output_path):
            self.output_path = output_path

    metric_args = MetricArgs(run_output_dir)

    def compute_metric_with_timer(name, func, *func_args):
        logging.info(f"Computing {name}...")
        start = time.time()
        result = func(*func_args)
        end = time.time()
        duration = str(datetime.timedelta(seconds=int(end - start)))
        logging.info(f"{name} computed in {duration}")
        return result

    bleu_score = compute_metric_with_timer(
        "BLEU", hivau_BLEU, eval_results, metric_args)
    rouge_score = compute_metric_with_timer(
        "ROUGE", hivau_ROUGE, eval_results, metric_args)
    cider_score = compute_metric_with_timer(
        "CIDEr", hivau_CIDEr, eval_results, metric_args)
    meteor_score = compute_metric_with_timer(
        "METEOR", hivau_METEOR, eval_results, metric_args)

    logging.info(f"\nFinal Results:")
    logging.info(f"   BLEU:   {bleu_score}")
    logging.info(f"   ROUGE:  {rouge_score}")
    logging.info(f"   CIDEr:  {cider_score}")
    logging.info(f"   METEOR: {meteor_score}")
    logging.info("=" * 50)

    # ---- Save Summary ----
    summary_file = os.path.join(run_output_dir, "summary.json")
    summary = {
        "model": "reactvau_paligemma_streamforest",
        "paligemma_model_path": args.paligemma_model_path,
        "paligemma_lora_path": args.paligemma_lora_path,
        "paligemma_image_size": args.paligemma_image_size,
        "paligemma_streamforest_weights": args.paligemma_streamforest_weights,
        "paligemma_vision_feature_layer": args.paligemma_vision_feature_layer,
        "paligemma_batch_size": args.paligemma_batch_size,
        "streamforest_model_path": args.streamforest_model_path,
        "streamforest_model_base": args.streamforest_model_base,
        "streamforest_conv_template": args.streamforest_conv_template,
        "streamforest_time_msg": args.streamforest_time_msg,
        "context_mode": args.context_mode,
        "sf_prompt_style": args.sf_prompt_style,
        "enable_memory_enhancement": args.enable_memory_enhancement,
        "anomaly_threshold": args.anomaly_threshold,
        "target_fps": args.target_fps,
        "query_interval": args.query_interval,
        "num_samples": len(instances),
        "num_successful": len([p for p in predictions if p.get('pred', '')]),
        "total_pg_queries": total_queries,
        "total_anomaly_segments": total_triggers,
        "validation_stats": validation_stats,
        "inference_duration": inference_duration,
        "metrics": {
            "BLEU": bleu_score,
            "ROUGE": rouge_score,
            "CIDEr": cider_score,
            "METEOR": meteor_score,
        },
        "timestamp": now_str,
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info(f"\nSummary saved to {summary_file}")
    logging.info(f"All results saved in: {run_output_dir}")
    logging.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
