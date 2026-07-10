"""
PaliGemma Video Anomaly Detection Evaluation Script

Supports:
1. Fine-tuned model with 2x2 grid input (--use-grid)
2. Single-frame baseline (default)
3. LoRA adapter loading

Streaming Evaluation:
- 4 FPS sampling
- Query interval = 4 (1 query per second)
- 2x2 grid input matching training format
"""

import os
import sys
import cv2
import json
import torch
import torch.nn.functional as F
import argparse
import random
import time
import datetime
import logging
import warnings
import numpy as np
import PIL.Image
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
from safetensors.torch import load_file as safetensors_load_file
import types

# Add ReactVAU path dynamically
ReactVAU_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if ReactVAU_ROOT not in sys.path:
    sys.path.append(ReactVAU_ROOT)

from detect_utils import compute_metrics, gaussian_smooth_1d, load_anno_txt, make_gt_labels_from_anno
from vad.get_prompt import get_grid_prompt

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ===================== Grid Image Creation =====================
# Match the training data generation settings
GRID_SIZE = (2, 2)  # 2x2 grid
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
    """Resize image to target_size x target_size."""
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

    # Pad with last frame if needed
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
    """Replace PaliGemma's vision encoder with StreamForest's SigLIP-384 weights.

    Steps:
    1. Load StreamForest weights from safetensors
    2. Resize position embedding from 1024 (448px) to 729 (384px)
    3. Load weights with strict=False (layer 26 kept from PaliGemma original)
    4. Update model config for 384x384 input
    """
    print(f"Loading StreamForest vision weights from: {weights_path}")

    stream_weights = safetensors_load_file(weights_path)
    print(f"  Loaded {len(stream_weights)} tensors from StreamForest")

    # Filter to vision_model keys only
    remapped = {}
    for key, value in stream_weights.items():
        if key.startswith("vision_model."):
            remapped[key] = value

    # Resize position embedding BEFORE loading weights
    pos_key = "vision_model.embeddings.position_embedding.weight"
    if pos_key in remapped:
        new_pos_shape = remapped[pos_key].shape
        old_pos_shape = model.vision_tower.vision_model.embeddings.position_embedding.weight.shape
        print(f"  Position embedding: {old_pos_shape} -> {new_pos_shape}")

        new_num_positions = new_pos_shape[0]  # 729
        embed_dim = new_pos_shape[1]          # 1152

        model.vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(
            new_num_positions, embed_dim
        ).to(model.device).to(model.dtype)
        print(f"  Resized position embedding to {new_num_positions} positions")

    # Load StreamForest weights
    missing, unexpected = model.vision_tower.load_state_dict(
        remapped, strict=False)
    print(f"  Missing keys (kept from PaliGemma): {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")

    # Update position-related state in embeddings
    model.vision_tower.vision_model.embeddings.num_positions = 729
    embeddings_module = model.vision_tower.vision_model.embeddings
    embeddings_module.register_buffer(
        "position_ids",
        torch.arange(729, device=model.device).expand((1, -1)),
        persistent=False,
    )

    # Update model config
    model.config.vision_config.image_size = 384
    model.config.vision_config.num_image_tokens = 729
    model.config.vision_config.num_positions = 729
    model.config.text_config.num_image_tokens = 729

    print(f"  Config updated: image_size=384, num_image_tokens=729")
    return model


def patch_vision_feature_layer(model, select_layer: int = -2):
    """Monkey-patch get_image_features to extract from a specific encoder layer.

    StreamForest uses select_layer=-2 (layer 25), but PaliGemma defaults to
    the last layer (layer 26). Since layer 26 is kept from PaliGemma's original
    weights (not StreamForest-trained), using layer -2 preserves the features
    StreamForest was optimized for.
    """
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
    print(f"  Patched get_image_features to use layer {select_layer}")
    return model


class PaliGemmaAnomalyDetector:
    """
    PaliGemma/PaliGemma2-based Anomaly Detection for Fine-tuned Model

    Supports streaming evaluation with:
    - 4 FPS sampling
    - 2x2 grid input (1 second = 4 frames = 1 grid)
    - Anomaly score from generation logits
    """

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
        """
        Args:
            model_path: Path to base PaliGemma/PaliGemma2 model
            lora_path: Path to LoRA adapter (optional)
            device: Device to run on
            torch_dtype: Data type for model weights
            attn_implementation: Attention implementation ("eager", "sdpa", or "flash_attention_2")
            image_size: Input image size (448 for original, 384 for StreamForest)
            streamforest_weights_path: Path to StreamForest vision encoder weights (required for 384)
            vision_feature_layer: Vision feature layer (-1 for last, -2 for StreamForest recommended)
        """
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.image_size = image_size

        print(f"Loading model from {model_path}...")
        print(f"Image size: {image_size}x{image_size}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_path)

        # Load base model
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
            # Update processor for 384x384
            self.processor.image_processor.size = {"height": 384, "width": 384}
            self.processor.image_processor.image_seq_length = 729
            self.processor.image_seq_length = 729
            print(f"  Processor updated for 384x384 (729 image tokens)")

        # Load LoRA adapter if provided
        if lora_path is not None and os.path.exists(lora_path):
            print(f"Loading LoRA adapter from {lora_path}...")
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path,
                torch_dtype=torch_dtype,
            )
            # Merge LoRA weights for faster inference
            self.model = self.model.merge_and_unload()
            print("✅ LoRA adapter loaded and merged")
            self.has_lora = True
        else:
            self.has_lora = False

        # Patch vision feature layer AFTER LoRA merge (for 384 mode)
        if image_size == 384 and streamforest_weights_path and vision_feature_layer != -1:
            self.model = patch_vision_feature_layer(
                self.model, select_layer=vision_feature_layer)

        self.model.eval()

        # Get token IDs for scoring
        # Training suffix is "Yes" or "No" (case-sensitive)
        # But model might generate "yes" or "no" (lowercase)
        # We check both cases to be safe
        self.yes_token_id = self._get_token_id("Yes")
        self.no_token_id = self._get_token_id("No")
        self.yes_lower_token_id = self._get_token_id("yes")
        self.no_lower_token_id = self._get_token_id("no")

        print(f"✅ Model loaded successfully")
        print(f"   Device: {self.device}")
        print(f"   Image size: {self.image_size}x{self.image_size}")
        print(f"   LoRA: {'Yes' if self.has_lora else 'No'}")
        print(
            f"   Token IDs - Yes: {self.yes_token_id}, yes: {self.yes_lower_token_id}")
        print(
            f"   Token IDs - No: {self.no_token_id}, no: {self.no_lower_token_id}")

    def _get_token_id(self, word: str) -> int:
        """Get token ID for a word"""
        tokens = self.processor.tokenizer.encode(
            word, add_special_tokens=False)
        if tokens:
            return tokens[0]
        else:
            raise ValueError(f"Could not encode word: {word}")

    def _get_video_info(self, video_path: str) -> dict:
        """Get video metadata"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        info["duration"] = info["total_frames"] / \
            info["fps"] if info["fps"] > 0 else 0
        cap.release()
        return info

    def _load_frame(self, cap, frame_idx: int) -> PIL.Image.Image:
        """Load a single frame from video"""
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(frame_rgb)

    def _get_anomaly_score_from_generation(
        self,
        grid_image: Image.Image,
        prompt: str,
        method: str = "logits"
    ) -> tuple:
        """
        Get anomaly score from model generation.

        The model was trained with suffix format: "Detection: Yes. Event: X" or "Detection: No. Event: Normal."

        Args:
            grid_image: 2x2 grid PIL Image
            prompt: The prompt (should match training prompt)
            method: 
                - "logits": Use logits at the position predicting Yes/No (faster)
                - "generate": Generate text and parse (more accurate but slower)

        Returns:
            (anomaly_score, generated_text)
        """
        if grid_image is None:
            return 0.0, ""

        # Format prompt with image token (matching training format)
        full_prompt = f"<image>{prompt}"

        inputs = self.processor(
            text=full_prompt,
            images=grid_image,
            return_tensors="pt"
        ).to(self.device)

        # Convert to model dtype
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(
                self.torch_dtype)

        if method == "generate":
            # Method 1: Generate and parse
            return self._score_from_generation(inputs)
        else:
            # Method 2: Use logits (faster, recommended)
            return self._score_from_logits(inputs)

    def _score_from_logits(self, inputs: dict) -> tuple:
        """
        Get anomaly score by looking at logits for "Yes" vs "No".

        IMPORTANT: Use direct forward pass instead of generate() to match
        training evaluation logic. The training's prediction_step uses
        forward pass and looks at logits at the last position.

        Note: We check both uppercase and lowercase variants since the model
        might prefer either "Yes"/"No" or "yes"/"no".
        """
        with torch.inference_mode():
            # Direct forward pass (matches training evaluation)
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                num_logits_to_keep=1,
            )

            logits = outputs.logits  # [1, seq_len, vocab_size]

            # Get logits at the last position (predicting first token)
            # This matches training's prediction_step logic
            last_logits = logits[0, -1]  # [vocab_size]

            # Get Yes and No logits (check both uppercase and lowercase)
            yes_logit = max(
                last_logits[self.yes_token_id].float(),
                last_logits[self.yes_lower_token_id].float()
            )
            no_logit = max(
                last_logits[self.no_token_id].float(),
                last_logits[self.no_lower_token_id].float()
            )

            # Softmax to get probability (same as training eval)
            probs = F.softmax(torch.stack([yes_logit, no_logit]), dim=0)
            anomaly_score = probs[0].item()  # P(Yes)

            # Generate text for logging (optional, use greedy decode)
            generated_text = "Yes" if anomaly_score > 0.5 else "No"

            return anomaly_score, generated_text

    def _score_from_generation(self, inputs: dict) -> tuple:
        """
        Get anomaly score by generating full text and parsing.
        More accurate but slower.
        """
        with torch.inference_mode():
            input_len = inputs["input_ids"].shape[1]

            # Generate response
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,  # Enough for "Detection: Yes. Event: Robbery."
                do_sample=False,
            )

            # Decode
            generated_ids = outputs[0, input_len:]
            generated_text = self.processor.tokenizer.decode(
                generated_ids, skip_special_tokens=True)

            # Parse response
            # Expected format: "Detection: Yes. Event: X" or "Detection: No. Event: Normal."
            generated_text_upper = generated_text.upper()

            if "YES" in generated_text_upper:
                anomaly_score = 1.0
            elif "NO" in generated_text_upper:
                anomaly_score = 0.0
            else:
                # Ambiguous - default to low score
                anomaly_score = 0.3

            return anomaly_score, generated_text

    def _batch_score_from_logits(self, grid_images: list, prompt: str) -> tuple:
        """
        Batch scoring using logits for multiple grid images.

        Args:
            grid_images: List of PIL Images (2x2 grids)
            prompt: Detection prompt

        Returns:
            (list of anomaly_scores, list of generated_texts)
        """
        if not grid_images:
            return [], []

        # Filter out None images
        valid_indices = [i for i, img in enumerate(
            grid_images) if img is not None]
        valid_images = [grid_images[i] for i in valid_indices]

        if not valid_images:
            return [0.0] * len(grid_images), [""] * len(grid_images)

        # Format prompt with image token
        full_prompt = f"<image>{prompt}"
        prompts = [full_prompt] * len(valid_images)

        # Process batch
        inputs = self.processor(
            text=prompts,
            images=valid_images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Convert to model dtype
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(
                self.torch_dtype)

        with torch.inference_mode():
            # Direct forward pass (matches training evaluation)
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                num_logits_to_keep=1,
            )

            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            batch_size = logits.shape[0]

            batch_scores = []
            batch_texts = []

            for batch_idx in range(batch_size):
                last_logits = logits[batch_idx, -1]  # [vocab_size]

                # Get Yes and No logits (check both uppercase and lowercase)
                yes_logit = max(
                    last_logits[self.yes_token_id].float(),
                    last_logits[self.yes_lower_token_id].float()
                )
                no_logit = max(
                    last_logits[self.no_token_id].float(),
                    last_logits[self.no_lower_token_id].float()
                )

                # Softmax to get probability (same as training eval)
                probs = F.softmax(torch.stack([yes_logit, no_logit]), dim=0)
                anomaly_score = probs[0].item()  # P(Yes)
                batch_scores.append(anomaly_score)

                # Generate text for logging
                generated_text = "Yes" if anomaly_score > 0.5 else "No"
                batch_texts.append(generated_text)

        # Map back to original indices (handle None images)
        final_scores = [0.0] * len(grid_images)
        final_texts = [""] * len(grid_images)
        for orig_idx, (score, text) in zip(valid_indices, zip(batch_scores, batch_texts)):
            final_scores[orig_idx] = score
            final_texts[orig_idx] = text

        return final_scores, final_texts

    def _batch_score_from_generation(self, grid_images: list, prompt: str) -> tuple:
        """
        Batch scoring using generation for multiple grid images.
        More accurate but slower than logits method.

        Args:
            grid_images: List of PIL Images (2x2 grids)
            prompt: Detection prompt

        Returns:
            (list of anomaly_scores, list of generated_texts)
        """
        if not grid_images:
            return [], []

        # Filter out None images
        valid_indices = [i for i, img in enumerate(
            grid_images) if img is not None]
        valid_images = [grid_images[i] for i in valid_indices]

        if not valid_images:
            return [0.0] * len(grid_images), [""] * len(grid_images)

        # Format prompt with image token
        full_prompt = f"<image>{prompt}"
        prompts = [full_prompt] * len(valid_images)

        # Process batch
        inputs = self.processor(
            text=prompts,
            images=valid_images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Convert to model dtype
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(
                self.torch_dtype)

        with torch.inference_mode():
            input_len = inputs["input_ids"].shape[1]

            # Generate responses for batch
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
            )

            batch_scores = []
            batch_texts = []

            for batch_idx in range(len(valid_images)):
                generated_ids = outputs[batch_idx, input_len:]
                generated_text = self.processor.tokenizer.decode(
                    generated_ids, skip_special_tokens=True)

                # Parse response
                generated_text_upper = generated_text.upper()
                if "YES" in generated_text_upper:
                    anomaly_score = 1.0
                elif "NO" in generated_text_upper:
                    anomaly_score = 0.0
                else:
                    anomaly_score = 0.3

                batch_scores.append(anomaly_score)
                batch_texts.append(generated_text)

        # Map back to original indices
        final_scores = [0.0] * len(grid_images)
        final_texts = [""] * len(grid_images)
        for orig_idx, (score, text) in zip(valid_indices, zip(batch_scores, batch_texts)):
            final_scores[orig_idx] = score
            final_texts[orig_idx] = text

        return final_scores, final_texts

    def _get_anomaly_scores_batch_grid(
        self,
        frame_groups: list,
        prompt: str,
        method: str = "logits"
    ) -> tuple:
        """
        Batch inference for multiple frame groups using 2x2 grid format.

        Args:
            frame_groups: List of lists, each inner list contains 4 frames
            prompt: Detection prompt
            method: "logits" or "generate"

        Returns:
            (list of anomaly_scores, list of generated_texts)
        """
        if not frame_groups:
            return [], []

        # Create grid images for each group
        grid_images = [create_grid_image(
            frames, GRID_SIZE, self.image_size) for frames in frame_groups]

        scores = []
        texts = []

        # Process one at a time for now (batched generation is complex)
        for grid_image in grid_images:
            score, text = self._get_anomaly_score_from_generation(
                grid_image, prompt, method)
            scores.append(score)
            texts.append(text)

        return scores, texts

    def detect_video_streaming(
        self,
        video_path: str,
        prompt: str,
        target_fps: int = 4,
        query_interval: int = 4,
        batch_size: int = 1,
        method: str = "logits",
        verbose: bool = False,
        single_frame: bool = False,
    ) -> dict:
        """
        Run streaming anomaly detection on a video.

        Streaming protocol:
        - Sample frames at target_fps (4 FPS)
        - Every query_interval frames (4 frames = 1 second), create a 2x2 grid
        - Query the model once per second

        Args:
            video_path: Path to video file
            prompt: Detection prompt (should match training prompt)
            target_fps: Target FPS for sampling (default: 4)
            query_interval: Number of sampled frames per query (default: 4 for 2x2 grid)
            batch_size: Batch size for inference
            method: "logits" (fast) or "generate" (accurate)
            verbose: Print detailed output

        Returns:
            dict with frame_scores, query_scores, generated_texts, etc.
        """
        # Get video info
        video_info = self._get_video_info(video_path)
        original_fps = video_info["fps"]
        total_frames = video_info["total_frames"]

        # Calculate sampling interval in original frame indices
        sample_interval = max(1, int(original_fps / target_fps))

        # Number of sampled frames
        num_sampled_frames = (
            total_frames + sample_interval - 1) // sample_interval

        # Number of queries (1 query per second at 4 FPS with interval=4)
        num_queries = (num_sampled_frames +
                       query_interval - 1) // query_interval

        if verbose:
            print(f"Video: {video_path}")
            print(
                f"  Original FPS: {original_fps:.1f}, Total frames: {total_frames}")
            print(
                f"  Sample interval: {sample_interval}, Sampled frames: {num_sampled_frames}")
            print(
                f"  Query interval: {query_interval}, Num queries: {num_queries}")

        # Open video
        cap = cv2.VideoCapture(video_path)

        # Collect frame groups for each query
        query_frame_groups = []
        query_frame_indices = []

        sampled_frame_idx = 0

        while sampled_frame_idx < num_sampled_frames:
            # Collect frames for this query
            frames = []
            frame_indices = []

            for i in range(query_interval):
                current_sampled_idx = sampled_frame_idx + i
                if current_sampled_idx >= num_sampled_frames:
                    break

                # Map to original frame index
                original_frame_idx = min(
                    current_sampled_idx * sample_interval, total_frames - 1)
                frame = self._load_frame(cap, original_frame_idx)

                if frame is not None:
                    frames.append(frame)
                    frame_indices.append(original_frame_idx)

            if frames:
                query_frame_groups.append(frames)
                query_frame_indices.append(frame_indices)

            sampled_frame_idx += query_interval

        cap.release()

        if verbose:
            print(f"  Collected {len(query_frame_groups)} query groups")

        # Process queries in batches using true batch inference
        query_scores = []
        generated_texts = []

        # Create images for each query (grid or single frame)
        if single_frame:
            # Single-frame mode: use the first (only) frame directly, resized
            grid_images = [resize_frame(frames[0], self.image_size) if frames else None
                           for frames in query_frame_groups]
        else:
            grid_images = [create_grid_image(
                frames, GRID_SIZE, self.image_size) for frames in query_frame_groups]

        # Process in batches with true batch inference
        for batch_start in range(0, len(grid_images), batch_size):
            batch_end = min(batch_start + batch_size, len(grid_images))
            batch_grids = grid_images[batch_start:batch_end]

            # Use batch scoring method
            if method == "logits":
                batch_scores, batch_texts = self._batch_score_from_logits(
                    batch_grids, prompt)
            else:
                batch_scores, batch_texts = self._batch_score_from_generation(
                    batch_grids, prompt)

            query_scores.extend(batch_scores)
            generated_texts.extend(batch_texts)

        query_scores = np.array(query_scores)

        # Expand query_scores to segment_scores (one score per sampled frame)
        segment_scores = []
        for i, score in enumerate(query_scores):
            # Each query covers query_interval sampled frames
            start_idx = i * query_interval
            end_idx = min((i + 1) * query_interval, num_sampled_frames)
            segment_scores.extend([score] * (end_idx - start_idx))

        segment_scores = np.array(segment_scores[:num_sampled_frames])

        # Expand segment_scores to frame_scores (one score per original frame)
        frame_scores = []
        for i, score in enumerate(segment_scores):
            frame_scores.extend([score] * sample_interval)

        frame_scores = np.array(frame_scores[:total_frames])

        # Apply Gaussian smoothing
        if len(segment_scores) > 5:
            smoothed_segment_scores = gaussian_smooth_1d(
                segment_scores, sigma=3)
        else:
            smoothed_segment_scores = segment_scores.copy()

        # Expand smoothed scores to frame-level
        smoothed_frame_scores = []
        for i, score in enumerate(smoothed_segment_scores):
            smoothed_frame_scores.extend([score] * sample_interval)
        smoothed_frame_scores = np.array(smoothed_frame_scores[:total_frames])

        return {
            "query_scores": query_scores.tolist(),
            "query_frame_indices": query_frame_indices,
            "generated_texts": generated_texts,
            "segment_scores": segment_scores.tolist(),
            "frame_scores": frame_scores.tolist(),
            "smoothed_frame_scores": smoothed_frame_scores.tolist(),
            "num_queries": len(query_scores),
            "num_segments": len(segment_scores),
            "total_frames": total_frames,
            "video_info": video_info,
        }


def main():
    parser = argparse.ArgumentParser(
        description="PaliGemma VAD Evaluation (Streaming)")

    # Model parameters
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to base PaliGemma model")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Path to LoRA adapter")

    # Streaming parameters
    parser.add_argument("--target-fps", type=int, default=4,
                        help="Target FPS for sampling (default: 4)")
    parser.add_argument("--query-interval", type=int, default=4,
                        help="Frames per query (default: 4 for 1 query/sec)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--method", type=str, default="logits", choices=["logits", "generate"],
                        help="Scoring method: logits (fast) or generate (accurate)")

    # Data parameters
    parser.add_argument("--dataset", type=str,
                        choices=["ucf-crime", "xd-violence"], required=True)
    parser.add_argument("--video-dir", type=str, required=True,
                        help="Base directory for videos")
    parser.add_argument("--anno-path", type=str, required=True,
                        help="Path to detection annotation JSON")

    # Output parameters
    parser.add_argument("--output-path", type=str, default="detection_results")
    parser.add_argument("--save-video-scores", action="store_true",
                        help="Save per-video segment scores")

    # Test parameters
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--test-samples", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")

    # Prompt configuration
    parser.add_argument("--prompt-style", type=str, default="detail",
                        choices=["detail"],
                        help="Prompt style used by the current ReactVAU VAD model")

    # Model configuration
    parser.add_argument("--attn-implementation", type=str, default="eager",
                        choices=["eager", "sdpa", "flash_attention_2"],
                        help="Attention implementation (default: eager for compatibility)")

    # 384x384 StreamForest vision encoder support
    parser.add_argument("--image-size", type=int, default=448, choices=[384, 448],
                        help="Input image size: 448 (original SigLIP) or 384 (StreamForest SigLIP)")
    parser.add_argument("--streamforest-vision-weights-path", type=str, default=None,
                        help="Path to StreamForest vision encoder weights (.safetensors). Required when --image-size=384")
    parser.add_argument("--vision-feature-layer", type=int, default=-2,
                        help="Vision encoder layer for feature extraction (-1=last, -2=second-to-last for StreamForest)")

    # Ablation: single-frame baseline & custom prompt
    parser.add_argument("--single-frame", action="store_true",
                        help="Single-frame mode: pass each frame directly without grid (use with --query-interval 1)")

    args = parser.parse_args()

    # Validate 384 mode args
    if args.image_size == 384 and not args.streamforest_vision_weights_path:
        parser.error(
            "--streamforest-vision-weights-path is required when --image-size is 384")

    # Create output directory
    now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    lora_suffix = "lora" if args.lora_path else "base"
    output_dir = os.path.join(
        args.output_path,
        f"paligemma2-{args.dataset}-streaming-{lora_suffix}-{now_str}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "detection.log")),
            logging.StreamHandler()
        ]
    )

    logging.info(f"=" * 60)
    logging.info(f"PaliGemma VAD Streaming Evaluation")
    logging.info(f"=" * 60)
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Model: {args.model_path}")
    logging.info(f"LoRA: {args.lora_path if args.lora_path else 'None'}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Target FPS: {args.target_fps}")
    logging.info(f"Query Interval: {args.query_interval}")
    logging.info(f"Method: {args.method}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Image Size: {args.image_size}x{args.image_size}")
    if args.streamforest_vision_weights_path:
        logging.info(
            f"StreamForest Weights: {args.streamforest_vision_weights_path}")
        logging.info(f"Vision Feature Layer: {args.vision_feature_layer}")

    # Use training prompt with configurable style, or custom prompt for ablation
    prompt = get_grid_prompt(add_special_tokens=False, style=args.prompt_style)
    logging.info(f"Prompt style: {args.prompt_style}")
    logging.info(f"Prompt: '{prompt}'")

    # Load annotation from official anno.txt
    logging.info(f"Loading annotation from {args.anno_path}")
    annotations = load_anno_txt(args.anno_path, args.dataset,
                                video_dir=args.video_dir)

    # Filter samples
    video_keys = list(annotations.keys())
    if args.test_mode:
        random.seed(42)
        video_keys = random.sample(video_keys, min(
            args.test_samples, len(video_keys)))
        logging.info(f"Test mode: using {len(video_keys)} samples")

    logging.info(f"Total videos to process: {len(video_keys)}")

    # Initialize detector
    logging.info("Initializing PaliGemma detector...")
    detector = PaliGemmaAnomalyDetector(
        model_path=args.model_path,
        lora_path=args.lora_path,
        device="cuda:0",
        attn_implementation=args.attn_implementation,
        image_size=args.image_size,
        streamforest_weights_path=args.streamforest_vision_weights_path,
        vision_feature_layer=args.vision_feature_layer,
    )

    # Determine video path format
    if args.dataset == "ucf-crime":
        video_subdir = "videos/ucf-crime/videos/test/"
    else:
        video_subdir = "videos/xd-violence/videos/test/"
    video_ext = ".mp4"

    # Process videos
    all_frame_scores = []
    all_frame_labels = []
    all_smoothed_scores = []
    video_results = {}

    total_queries = 0
    total_segments = 0

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
            result = detector.detect_video_streaming(
                video_path=video_path,
                prompt=prompt,
                target_fps=args.target_fps,
                query_interval=args.query_interval,
                batch_size=args.batch_size,
                method=args.method,
                verbose=True,  # Disable internal progress, only show video-level progress
                single_frame=args.single_frame,
            )

            # Create frame-level labels from official intervals
            frame_labels = make_gt_labels_from_anno(anno, result["total_frames"], video_fps)

            # Collect scores and labels
            frame_scores = np.array(result["frame_scores"])
            smoothed_scores = np.array(result["smoothed_frame_scores"])

            # Ensure same length
            min_len = min(len(frame_scores), len(frame_labels))
            frame_scores = frame_scores[:min_len]
            smoothed_scores = smoothed_scores[:min_len]
            frame_labels = frame_labels[:min_len]

            all_frame_scores.extend(frame_scores.tolist())
            all_smoothed_scores.extend(smoothed_scores.tolist())
            all_frame_labels.extend(frame_labels.tolist())

            total_queries += result["num_queries"]
            total_segments += result["num_segments"]

            # Store video result
            video_results[video_key] = {
                "video_name": video_name,
                "num_queries": result["num_queries"],
                "query_scores": result["query_scores"],
                # First 5 for logging
                "generated_texts": result["generated_texts"][:5],
                "segment_scores": result["segment_scores"],
                "n_frames": result["total_frames"],
                "events_frames": anno["intervals_raw"],
            }

        except Exception as e:
            logging.error(f"Error processing {video_key}: {e}")
            import traceback
            traceback.print_exc()

    elapsed_time = time.time() - start_time

    # Compute metrics
    logging.info("\n" + "=" * 60)
    logging.info("Computing metrics...")

    if len(all_frame_scores) > 0 and len(all_frame_labels) > 0:
        metrics_raw = compute_metrics(all_frame_scores, all_frame_labels)
        logging.info(f"Raw Frame Scores:")
        logging.info(f"  ROC-AUC: {metrics_raw['roc_auc']:.4f}")
        logging.info(f"  PR-AUC:  {metrics_raw['pr_auc']:.4f}")
        logging.info(f"  AP:      {metrics_raw['ap']:.4f}")

        metrics_smoothed = compute_metrics(
            all_smoothed_scores, all_frame_labels)
        logging.info(f"\nSmoothed Frame Scores:")
        logging.info(f"  ROC-AUC: {metrics_smoothed['roc_auc']:.4f}")
        logging.info(f"  PR-AUC:  {metrics_smoothed['pr_auc']:.4f}")
        logging.info(f"  AP:      {metrics_smoothed['ap']:.4f}")
    else:
        logging.warning("No valid predictions to compute metrics")
        metrics_raw = {}
        metrics_smoothed = {}

    # Performance stats
    logging.info(f"\n" + "=" * 60)
    logging.info(f"Performance Statistics:")
    logging.info(f"  Total videos processed: {len(video_results)}")
    logging.info(f"  Total queries: {total_queries}")
    logging.info(f"  Total sampled frames (segments): {total_segments}")
    logging.info(f"  Total original frames: {len(all_frame_scores)}")
    logging.info(f"  Elapsed time: {elapsed_time:.1f}s")
    logging.info(
        f"  Avg time per query: {elapsed_time / max(1, total_queries) * 1000:.1f}ms")
    logging.info(
        f"  Streaming: {args.target_fps} FPS, {args.query_interval} frames/query (1 query/sec)")

    # Save results
    summary = {
        "model": "paligemma2",
        "model_path": args.model_path,
        "lora_path": args.lora_path,
        "image_size": args.image_size,
        "streamforest_weights": args.streamforest_vision_weights_path,
        "vision_feature_layer": args.vision_feature_layer,
        "dataset": args.dataset,
        "target_fps": args.target_fps,
        "query_interval": args.query_interval,
        "method": args.method,
        "batch_size": args.batch_size,
        "num_videos": len(video_results),
        "total_queries": total_queries,
        "total_segments": total_segments,
        "total_frames": len(all_frame_scores),
        "elapsed_time_sec": elapsed_time,
        "metrics_raw": {k: v for k, v in metrics_raw.items() if k not in ["fpr", "tpr"]},
        "metrics_smoothed": {k: v for k, v in metrics_smoothed.items() if k not in ["fpr", "tpr"]},
        "prompt": prompt,
        "timestamp": now_str,
    }

    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    if args.save_video_scores:
        with open(os.path.join(output_dir, "video_results.json"), 'w') as f:
            json.dump(video_results, f, indent=2)

    logging.info(f"\n✅ Results saved to {output_dir}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
