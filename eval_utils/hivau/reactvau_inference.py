# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
ReactVAU Model Inference Wrapper (Streaming Mode)
Aligned with eval_reactvau_hivau.py — full PaliGemma + StreamForest pipeline.

Architecture:
    PaliGemma2-3B (Detection Module) + StreamForest-Qwen2-7B (Reasoning Module)

Pipeline:
    1. Video sampled at target FPS (default 4).
       Every query_interval frames (default 4, = 1 second) -> 2x2 grid -> PaliGemma detection.
       Simultaneously, the last frame of each group is fed into StreamForest's MemoryManager.
    2. After full video processing:
       - PaliGemma provides score-based detection context (anomaly segments)
       - StreamForest generates understanding response using accumulated visual memory + PG context

Context Injection Modes (--context-mode):
    - "none":        No PG context injected (equivalent to standalone StreamForest)
    - "score":       PG anomaly scores summary embedded in prompt
"""
import sys
import os
import types
import logging
import warnings
import traceback

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import PIL.Image
from PIL import Image
from typing import Optional

# Add ReactVAU path dynamically
ReactVAU_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if ReactVAU_ROOT not in sys.path:
    sys.path.insert(0, ReactVAU_ROOT)

from vad.get_prompt import get_grid_prompt
from safetensors.torch import load_file as safetensors_load_file
from peft import PeftModel
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from llava.model.multimodal_projector.memory_manager import MemoryManager
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from llava.model.builder import load_pretrained_model

warnings.filterwarnings('ignore')


# ===================== Grid Image Creation (same as eval) =====================
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
        Phase 1 - Video Processing:
            - Sample at target FPS
            - Every query_interval frames (1 sec): create 2x2 grid -> PaliGemma detection
            - Simultaneously: feed last frame of each group into StreamForest memory
        Phase 2 - Context Building:
            - Collect score-based PaliGemma detection context (anomaly segments)
            - Build enhanced prompt: PG context + original question
        Phase 3 - Answer Generation:
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
        """Build PaliGemma context string for enhanced prompt."""
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
        top_segments = sorted(anomaly_segments, key=lambda x: x["score"], reverse=True)[
            :self.max_context_segments]
        top_segments = sorted(top_segments, key=lambda x: x["time"])

        duration_str = f" (video duration: {video_duration:.1f}s)" if video_duration > 0 else ""

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

        if self.context_mode == "score":
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
        """Combine PG context with the original question."""
        if not pg_context_str:
            return question

        is_skeptical = self.sf_prompt_style == "skeptical"

        if task == "caption":
            return (
                f"{pg_context_str}\n\n"
                f"Based on the video content, answer the following:\n"
                f"{question}"
            )

        if is_skeptical:
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
                }
                anomaly_segments.append(seg_info)

        del grid_images

        # ---- Phase 3: Build StreamForest memory incrementally ----
        # Pool threshold must match training (0.6 for V2)
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

            if self.enable_memory_enhancement:
                memory_manager.update_with_anomaly_score(
                    cached_sf_features[query_idx], anomaly_score=pg_score)
                if pg_score >= anomaly_pool_threshold:
                    memory_manager.update_anomaly_pool(
                        cached_sf_features[query_idx], pg_score)
            else:
                memory_manager.update(cached_sf_features[query_idx])
            sf_frame_count += 1

            if query_idx % 30 == 0:
                torch.cuda.empty_cache()

        del cached_sf_features

        # ---- Phase 4: Build enhanced prompt + Generate ----
        pg_context_str = ""
        if self.context_mode != "none":
            pg_context_str = self._build_pg_context_string(
                anomaly_segments, video_duration)

        enhanced_question = self._build_enhanced_question(pg_context_str, question, task)

        response = self.streamforest.generate_text(
            memory_manager=memory_manager,
            question=enhanced_question,
            current_time=video_duration,
            sampled_count=sf_frame_count,
            video_height=video_height,
            video_width=video_width,
            max_new_tokens=max_new_tokens,
        )

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


# ===================== High-Level Inference Wrapper =====================
class ReactVAUInference:
    """
    High-level ReactVAU inference wrapper.
    Initializes PaliGemma (detection) + StreamForest (reasoning) and exposes a simple generate() API.
    All parameters are aligned with eval_reactvau_hivau.py defaults.
    """

    def __init__(
        self,
        # PaliGemma params
        paligemma_model_path: str,
        paligemma_lora_path: Optional[str] = None,
        paligemma_image_size: int = 384,
        paligemma_streamforest_weights: Optional[str] = None,
        paligemma_vision_feature_layer: int = -2,
        paligemma_attn: str = "sdpa",
        paligemma_batch_size: int = 32,
        # StreamForest params
        streamforest_model_path: str = "",
        streamforest_model_base: Optional[str] = None,
        streamforest_conv_template: str = "qwen_2",
        streamforest_time_msg: str = "short_online_v2",
        # Pipeline params
        context_mode: str = "score",
        anomaly_threshold: float = 0.4,
        enable_memory_enhancement: bool = True,
        sf_prompt_style: str = "skeptical",
        target_fps: int = 4,
        query_interval: int = 4,
        device: str = "cuda:0",
    ):
        self.target_fps = target_fps
        self.query_interval = query_interval
        self.paligemma_batch_size = paligemma_batch_size

        # Validate 384 mode
        if paligemma_image_size == 384 and not paligemma_streamforest_weights:
            raise ValueError(
                "384 mode requires --paligemma-streamforest-weights")

        # PaliGemma prompt
        pg_prompt = get_grid_prompt(add_special_tokens=False)
        logging.info(f"PaliGemma prompt: '{pg_prompt}'")

        # Initialize PaliGemma detector
        logging.info("\nInitializing PaliGemma detector...")
        self.paligemma = PaliGemmaDetector(
            model_path=paligemma_model_path,
            lora_path=paligemma_lora_path,
            device=device,
            attn_implementation=paligemma_attn,
            image_size=paligemma_image_size,
            streamforest_weights_path=paligemma_streamforest_weights,
            vision_feature_layer=paligemma_vision_feature_layer,
        )

        # Initialize StreamForest reasoner
        logging.info("\nInitializing StreamForest reasoner...")
        self.streamforest = StreamForestReasoner(
            model_path=streamforest_model_path,
            model_base=streamforest_model_base,
            conv_template=streamforest_conv_template,
            device=device,
            time_msg_style=streamforest_time_msg,
        )

        # Build combined pipeline
        self.pipeline = ReactVAUUnderstanding(
            paligemma_detector=self.paligemma,
            streamforest_reasoner=self.streamforest,
            anomaly_threshold=anomaly_threshold,
            context_mode=context_mode,
            paligemma_prompt=pg_prompt,
            enable_memory_enhancement=enable_memory_enhancement,
            sf_prompt_style=sf_prompt_style,
        )

        logging.info("\nReactVAU inference wrapper initialized.")

    def generate(
        self,
        video_path: str,
        question: str,
        max_new_tokens: int = 512,
        task: str = "",
    ) -> dict:
        """
        Run full ReactVAU pipeline on a single video.

        Args:
            video_path: Path to the video file.
            question: The question/prompt to answer.
            max_new_tokens: Maximum tokens to generate.
            task: HIVAU task type (judgement/description/analysis/caption), affects prompt framing.

        Returns:
            dict with 'response', 'pg_context', 'anomaly_segments', 'paligemma_scores', etc.
        """
        return self.pipeline.process_video_and_generate(
            video_path=video_path,
            question=question,
            target_fps=self.target_fps,
            query_interval=self.query_interval,
            batch_size=self.paligemma_batch_size,
            max_new_tokens=max_new_tokens,
            task=task,
        )
