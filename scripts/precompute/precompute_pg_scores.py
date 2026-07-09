#!/usr/bin/env python3
"""
Pre-compute PaliGemma anomaly scores for HIVAU training videos.

For each video in the training annotation, samples frames at 4 FPS,
creates 2×2 grids (4 frames per query), and runs PaliGemma batch inference
to produce per-query anomaly scores. These scores are saved as a JSON file
that the training DataLoader can load at fine-tuning time.

This ensures training-time PG context matches eval-time exactly.

Usage:
    python precompute_pg_scores.py

Output:
    {output_path}/pg_scores_hivau_train.json
    Format: {
        "ucf-crime/clips/train/Abuse001_x264_E0C0.mp4": {
            "pg_scores": [0.7311, 0.6225, ...],
            "n_frames": 743,
            "sample_interval": 7,
            "num_queries": 27
        },
        ...
    }
"""

import os
import sys
import json
import math
import time
import signal
import logging
import warnings
import argparse
from pathlib import Path
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm

import cv2
import PIL.Image
import torch
import torch.nn.functional as F

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


# ===================== Grid Image Creation (identical to eval) =====================
GRID_SIZE = (2, 2)
USE_SEPARATOR = True
SEPARATOR_WIDTH = 2
SEPARATOR_COLOR = (128, 128, 128)


def compute_grid_params(image_size: int = 448):
    if USE_SEPARATOR:
        cell_size = (image_size - SEPARATOR_WIDTH) // 2
        total_size = cell_size * 2 + SEPARATOR_WIDTH
    else:
        cell_size = image_size // 2
        total_size = cell_size * 2
    return cell_size, total_size


def resize_frame(image, target_size):
    if image is None:
        return PIL.Image.new("RGB", (target_size, target_size), (0, 0, 0))
    return image.resize((target_size, target_size), PIL.Image.Resampling.LANCZOS)


def create_grid_image(frames, grid_size=(2, 2), image_size=448):
    if not frames:
        return PIL.Image.new("RGB", (image_size, image_size), (0, 0, 0))
    cell_size, total_size = compute_grid_params(image_size)
    rows, cols = grid_size
    expected_frames = rows * cols
    while len(frames) < expected_frames:
        frames.append(frames[-1] if frames else PIL.Image.new("RGB", (cell_size, cell_size)))
    if USE_SEPARATOR:
        grid_image = PIL.Image.new("RGB", (total_size, total_size), SEPARATOR_COLOR)
        for idx in range(expected_frames):
            row, col = divmod(idx, cols)
            resized = resize_frame(frames[idx], cell_size)
            x = col * (cell_size + SEPARATOR_WIDTH)
            y = row * (cell_size + SEPARATOR_WIDTH)
            grid_image.paste(resized, (x, y))
    else:
        grid_image = PIL.Image.new("RGB", (total_size, total_size))
        for idx in range(expected_frames):
            row, col = divmod(idx, cols)
            resized = resize_frame(frames[idx], cell_size)
            grid_image.paste(resized, (col * cell_size, row * cell_size))
    return grid_image


# ===================== Vision Encoder Utilities =====================
from safetensors.torch import load_file as safetensors_load_file
from peft import PeftModel
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import types


def load_streamforest_vision_weights(model, weights_path):
    logging.info(f"Loading StreamForest vision weights from: {weights_path}")
    stream_weights = safetensors_load_file(weights_path)
    logging.info(f"  Loaded {len(stream_weights)} tensors from StreamForest")

    # Filter to vision_model keys only (match eval script)
    remapped = {}
    for key, value in stream_weights.items():
        new_key = key.replace("vision_tower.vision_tower.", "")
        if new_key.startswith("vision_model."):
            remapped[new_key] = value

    # Resize model's position embedding BEFORE loading weights (match eval script)
    pos_key = "vision_model.embeddings.position_embedding.weight"
    if pos_key in remapped:
        new_pos_shape = remapped[pos_key].shape
        old_pos_shape = model.vision_tower.vision_model.embeddings.position_embedding.weight.shape
        logging.info(f"  Position embedding: {old_pos_shape} -> {new_pos_shape}")

        new_num_positions = new_pos_shape[0]  # 729
        embed_dim = new_pos_shape[1]          # 1152

        model.vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(
            new_num_positions, embed_dim
        ).to(model.device).to(model.dtype)
        logging.info(f"  Resized position embedding to {new_num_positions} positions")

    # Load StreamForest weights
    missing, unexpected = model.vision_tower.load_state_dict(remapped, strict=False)
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


def patch_vision_feature_layer(model, select_layer=-2):
    if hasattr(model, 'get_image_features'):
        target = model
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        target = model.base_model.model
    else:
        target = model

    def get_image_features_with_layer_select(self, pixel_values):
        vision_outputs = self.vision_tower(
            pixel_values, output_hidden_states=True)
        selected = vision_outputs.hidden_states[select_layer]
        image_features = self.multi_modal_projector(selected)
        image_features = image_features / (self.config.hidden_size ** 0.5)
        return image_features

    target.get_image_features = types.MethodType(
        get_image_features_with_layer_select, target)
    logging.info(f"  Patched get_image_features to use layer {select_layer}")
    return model


class PaliGemmaScorer:
    """Lightweight PaliGemma scorer for batch grid scoring."""

    def __init__(self, model_path, lora_path, device="cuda:0",
                 image_size=384, streamforest_weights_path=None,
                 vision_feature_layer=-2, attn_implementation="sdpa"):
        self.device = torch.device(device)
        self.torch_dtype = torch.bfloat16
        self.image_size = image_size

        logging.info(f"[PaliGemma] Loading model from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=self.torch_dtype,
            device_map=device, attn_implementation=attn_implementation)

        if image_size == 384 and streamforest_weights_path:
            self.model = load_streamforest_vision_weights(
                self.model, streamforest_weights_path)
            self.processor.image_processor.size = {"height": 384, "width": 384}
            self.processor.image_processor.image_seq_length = 729
            self.processor.image_seq_length = 729
            logging.info(f"[PaliGemma] Processor updated for 384x384")

        if lora_path and os.path.exists(lora_path):
            logging.info(f"[PaliGemma] Loading LoRA from {lora_path}...")
            self.model = PeftModel.from_pretrained(
                self.model, lora_path, torch_dtype=self.torch_dtype)
            self.model = self.model.merge_and_unload()
            logging.info("[PaliGemma] LoRA merged")

        if image_size == 384 and streamforest_weights_path and vision_feature_layer != -1:
            self.model = patch_vision_feature_layer(
                self.model, select_layer=vision_feature_layer)

        self.model.eval()

        self.yes_token_id = self._get_token_id("Yes")
        self.no_token_id = self._get_token_id("No")
        self.yes_lower_token_id = self._get_token_id("yes")
        self.no_lower_token_id = self._get_token_id("no")
        logging.info(f"[PaliGemma] Ready. Yes={self.yes_token_id}, No={self.no_token_id}")

    def _get_token_id(self, word):
        tokens = self.processor.tokenizer.encode(word, add_special_tokens=False)
        return tokens[0] if tokens else None

    def batch_score_grids(self, grid_images, prompt, batch_size=32):
        """Score multiple grids in batches. Returns list of float scores."""
        if not grid_images:
            return []
        full_prompt = f"<image>{prompt}"
        all_scores = []
        for batch_start in range(0, len(grid_images), batch_size):
            batch_end = min(batch_start + batch_size, len(grid_images))
            batch_imgs = grid_images[batch_start:batch_end]
            prompts = [full_prompt] * len(batch_imgs)
            inputs = self.processor(
                text=prompts, images=batch_imgs,
                return_tensors="pt", padding=True).to(self.device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.torch_dtype)
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                )
                logits = outputs.logits
                for idx in range(len(batch_imgs)):
                    attention = inputs["attention_mask"][idx]
                    last_pos = attention.sum().item() - 1
                    last_logits = logits[idx, last_pos]
                    yes_logit = max(
                        last_logits[self.yes_token_id].float(),
                        last_logits[self.yes_lower_token_id].float())
                    no_logit = max(
                        last_logits[self.no_token_id].float(),
                        last_logits[self.no_lower_token_id].float())
                    probs = F.softmax(torch.stack([yes_logit, no_logit]), dim=0)
                    all_scores.append(round(probs[0].item(), 4))
            del inputs, outputs, logits
        return all_scores


def read_video_grids(video_path, image_size, target_fps=4, query_interval=4):
    """Read video frames and create grid images. CPU-bound, safe to run in thread."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if original_fps <= 0 or total_frames <= 0:
        cap.release()
        return None, None

    sample_interval = max(1, int(original_fps / target_fps))
    num_sampled_frames = (total_frames + sample_interval - 1) // sample_interval

    # Collect frame groups — sequential read with grab() to avoid costly H.264 seeking
    query_frame_groups = []
    current_group = []
    sampled_idx = 0
    current_frame_pos = 0

    while sampled_idx < num_sampled_frames:
        target_frame = min(sampled_idx * sample_interval, total_frames - 1)

        # Skip forward sequentially (grab without decode)
        while current_frame_pos < target_frame:
            cap.grab()
            current_frame_pos += 1

        ret, frame = cap.read()
        current_frame_pos += 1

        if not ret:
            sampled_idx += 1
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = PIL.Image.fromarray(frame_rgb)
        current_group.append(pil_frame)

        if len(current_group) == query_interval or sampled_idx == num_sampled_frames - 1:
            query_frame_groups.append(current_group)
            current_group = []
        sampled_idx += 1
    cap.release()

    # Create grids
    grid_images = [
        create_grid_image(frames, GRID_SIZE, image_size)
        for frames in query_frame_groups
    ]

    metadata = {
        "n_frames": total_frames,
        "sample_interval": sample_interval,
    }
    del query_frame_groups
    return grid_images, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute PaliGemma anomaly scores for HIVAU training videos")

    # PaliGemma configuration (must match eval exactly)
    default_project_root = os.environ.get(
        "REACTVAU_ROOT",
        str(Path(__file__).resolve().parents[2])
    )
    default_ckpt_root = os.environ.get(
        "CKPT_ROOT", os.path.join(default_project_root, "ckpt"))
    default_hivau_video_root = os.environ.get(
        "HIVAU_VIDEO_ROOT", "/path/to/HIVAU-70k/videos")

    parser.add_argument("--paligemma-model-path", type=str,
                        default=os.environ.get(
                            "PALIGEMMA_MODEL_PATH",
                            os.path.join(default_ckpt_root, "paligemma2-3b-mix-448")))
    parser.add_argument("--paligemma-lora-path", type=str,
                        default=os.environ.get(
                            "PALIGEMMA_LORA_PATH",
                            os.path.join(default_ckpt_root, "paligemma2-3b-vad-lora-384-combined-binary/training_384")))
    parser.add_argument("--paligemma-image-size", type=int, default=384)
    parser.add_argument("--paligemma-streamforest-weights", type=str,
                        default=os.environ.get(
                            "PALIGEMMA_SF_WEIGHTS",
                            os.path.join(default_ckpt_root, "extracted_weights/streamforest_vision_encoder_with_prefix.safetensors")))
    parser.add_argument("--paligemma-vision-feature-layer", type=int, default=-2)
    parser.add_argument("--paligemma-attn", type=str, default="sdpa")
    parser.add_argument("--paligemma-prompt-style", type=str, default="detail",
                        choices=["detail"])

    # Data configuration
    parser.add_argument("--anno-path", type=str,
                        default=os.environ.get(
                            "HIVAU_TRAIN_JSON",
                            os.path.join(default_project_root, "scripts/train/finetune-hivau/hivau_minimal.json")))
    parser.add_argument("--data-root", type=str,
                        default=default_hivau_video_root)

    # Pipeline configuration (must match eval exactly)
    parser.add_argument("--target-fps", type=int, default=4)
    parser.add_argument("--query-interval", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)

    # Output
    parser.add_argument("--output-path", type=str,
                        default=os.environ.get(
                            "PG_SCORES_PATH",
                            os.path.join(default_project_root, "precomputed/pg_scores_hivau_train.json")))

    # Resume support
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file (skip already processed videos)")

    args = parser.parse_args()

    # Setup
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Add project root to path for get_prompt import
    sys.path.insert(0, default_project_root)
    from vad.get_prompt import get_grid_prompt

    # PG prompt (must match eval)
    pg_prompt = get_grid_prompt(add_special_tokens=False, style=args.paligemma_prompt_style)
    logging.info(f"PG prompt: '{pg_prompt}'")

    # Load annotation
    logging.info(f"Loading annotation from {args.anno_path}")
    with open(args.anno_path) as f:
        annotations = json.load(f)

    # Extract unique video paths
    unique_videos = sorted(set(d["video"] for d in annotations if "video" in d))
    logging.info(f"Total annotation entries: {len(annotations)}")
    logging.info(f"Unique videos: {len(unique_videos)}")

    # Resume support
    existing_results = {}
    if args.resume and os.path.exists(args.output_path):
        with open(args.output_path) as f:
            existing_results = json.load(f)
        logging.info(f"Resuming: {len(existing_results)} videos already processed")
        unique_videos = [v for v in unique_videos if v not in existing_results]
        logging.info(f"Remaining: {len(unique_videos)} videos to process")

    # Load PG model
    logging.info("Loading PaliGemma model...")
    scorer = PaliGemmaScorer(
        model_path=args.paligemma_model_path,
        lora_path=args.paligemma_lora_path,
        device="cuda:0",
        image_size=args.paligemma_image_size,
        streamforest_weights_path=args.paligemma_streamforest_weights,
        vision_feature_layer=args.paligemma_vision_feature_layer,
        attn_implementation=args.paligemma_attn,
    )

    # ---- Cross-video batching with parallel readers ----
    # Multiple threads read videos on CPU while GPU scores accumulated grids
    # from multiple videos in full batches, maximizing GPU utilization.
    NUM_READERS = 4        # Parallel CPU video reader threads
    QUEUE_MAXSIZE = 24     # Max videos buffered in read queue

    results = dict(existing_results)
    failed = []
    start_time = time.time()

    # --- Graceful shutdown on SIGTERM (sent by SLURM before kill) ---
    shutdown_requested = False

    def handle_sigterm(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        logging.warning(f"Received signal {signum} — will save and exit after current batch.")

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGUSR1, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    read_q = queue.Queue(maxsize=QUEUE_MAXSIZE)

    def _read_one(video_rel_path):
        """Read a single video's frames and create grids (runs in thread pool)."""
        full_path = os.path.join(args.data_root, video_rel_path)
        if not os.path.exists(full_path):
            return (video_rel_path, None, None, "file not found")
        try:
            grids, meta = read_video_grids(
                full_path, args.paligemma_image_size,
                args.target_fps, args.query_interval)
            if grids is None:
                return (video_rel_path, None, None, "could not read video")
            return (video_rel_path, grids, meta, None)
        except Exception as e:
            return (video_rel_path, None, None, str(e))

    def reader_dispatcher():
        """Dispatch parallel video reads via thread pool and enqueue results."""
        try:
            with ThreadPoolExecutor(max_workers=NUM_READERS) as pool:
                pending_futures = set()
                vid_idx = 0
                max_inflight = NUM_READERS * 3

                while (vid_idx < len(unique_videos) or pending_futures) and not shutdown_requested:
                    # Submit new reads up to the inflight limit
                    while vid_idx < len(unique_videos) and len(pending_futures) < max_inflight:
                        fut = pool.submit(_read_one, unique_videos[vid_idx])
                        pending_futures.add(fut)
                        vid_idx += 1

                    if not pending_futures:
                        break

                    done, pending_futures = wait(pending_futures, return_when=FIRST_COMPLETED)
                    for f in done:
                        try:
                            read_q.put(f.result())
                        except Exception as e:
                            read_q.put(("unknown", None, None, str(e)))
        except Exception as e:
            logging.error(f"Reader dispatcher error: {e}")
        finally:
            read_q.put(None)  # Sentinel to signal completion

    dispatcher = threading.Thread(target=reader_dispatcher, daemon=True)
    dispatcher.start()

    # ---- GPU consumer: accumulate grids across videos, batch score ----
    grid_buffer = []          # Flat list of grid images from multiple videos
    buffer_provenance = []    # [(video_rel_path, metadata, start_idx, count), ...]
    processed_count = 0
    pbar = tqdm(total=len(unique_videos), desc="Processing videos")

    def flush_buffer():
        """Score all accumulated grids on GPU and distribute results."""
        nonlocal grid_buffer, buffer_provenance
        if not grid_buffer:
            return
        all_scores = scorer.batch_score_grids(
            grid_buffer, pg_prompt, batch_size=args.batch_size)
        for vpath, vmeta, vstart, vcount in buffer_provenance:
            results[vpath] = {
                "pg_scores": all_scores[vstart:vstart + vcount],
                "n_frames": vmeta["n_frames"],
                "sample_interval": vmeta["sample_interval"],
                "num_queries": vcount,
            }
        grid_buffer = []
        buffer_provenance = []

    while True:
        try:
            item = read_q.get(timeout=2.0)
        except queue.Empty:
            if shutdown_requested:
                break
            continue

        if item is None:
            break

        video_rel_path, grids, meta, error = item
        processed_count += 1
        pbar.update(1)

        if error or grids is None:
            failed.append((video_rel_path, error or "read failed"))
            continue

        # Accumulate grids from this video into the cross-video buffer
        start_idx = len(grid_buffer)
        grid_buffer.extend(grids)
        buffer_provenance.append((video_rel_path, meta, start_idx, len(grids)))
        del grids

        # Flush to GPU when enough grids accumulated for efficient batching
        if len(grid_buffer) >= args.batch_size:
            flush_buffer()

        # Periodic save (every 500 videos)
        if processed_count % 500 == 0:
            flush_buffer()
            with open(args.output_path, 'w') as f:
                json.dump(results, f)
            elapsed = time.time() - start_time
            rate = processed_count / elapsed
            eta = (len(unique_videos) - processed_count) / rate
            logging.info(
                f"Progress: {processed_count}/{len(unique_videos)} "
                f"({elapsed/60:.1f}min elapsed, ~{eta/60:.1f}min remaining)")

        if shutdown_requested:
            logging.warning(f"Graceful shutdown after {processed_count} videos.")
            break

    # Score any remaining buffered grids
    flush_buffer()
    pbar.close()
    dispatcher.join(timeout=10)

    # Final save
    with open(args.output_path, 'w') as f:
        json.dump(results, f)

    elapsed = time.time() - start_time
    logging.info(f"\n{'='*60}")
    logging.info(f"Pre-computation complete!")
    logging.info(f"  Processed: {len(results)} videos")
    logging.info(f"  Failed: {len(failed)} videos")
    logging.info(f"  Time: {elapsed/60:.1f} minutes")
    logging.info(f"  Output: {args.output_path}")

    if failed:
        failed_path = args.output_path.replace('.json', '_failed.json')
        with open(failed_path, 'w') as f:
            json.dump(failed, f, indent=2)
        logging.info(f"  Failed list: {failed_path}")


if __name__ == "__main__":
    main()
