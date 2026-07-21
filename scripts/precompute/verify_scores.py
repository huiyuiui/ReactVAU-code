# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#!/usr/bin/env python3
"""
Score-by-score verification: run the precompute pipeline on test videos
and compare against eval results.

Tests:
1. Precompute read_video_grids (sequential grab) + batch_score_grids
2. Eval-style read (seek-based) + same batch_score_grids
3. Compare both against stored eval video_results.json

Usage:
    python scripts/precompute/verify_scores.py
"""

import os
import sys
import json
import math
import time
import warnings
import numpy as np

import cv2
import PIL.Image
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Import from precompute script
from scripts.precompute.precompute_pg_scores import (
    PaliGemmaScorer,
    read_video_grids,
    create_grid_image,
    GRID_SIZE,
    compute_grid_params,
    resize_frame,
)


def read_video_grids_seekbased(video_path, image_size, target_fps=4, query_interval=4):
    """
    Read video frames using SEEK-based approach (matches eval script exactly).
    cap.set(CAP_PROP_POS_FRAMES, idx) for each frame.
    """
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

    query_frame_groups = []
    sampled_frame_idx = 0

    while sampled_frame_idx < num_sampled_frames:
        frames = []
        for i in range(query_interval):
            current_sampled_idx = sampled_frame_idx + i
            if current_sampled_idx >= num_sampled_frames:
                break
            original_frame_idx = min(
                current_sampled_idx * sample_interval, total_frames - 1)

            # Eval-style: seek to exact frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, original_frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(PIL.Image.fromarray(frame_rgb))

        if frames:
            query_frame_groups.append(frames)
        sampled_frame_idx += query_interval

    cap.release()

    grid_images = [
        create_grid_image(frames, GRID_SIZE, image_size)
        for frames in query_frame_groups
    ]

    metadata = {
        "n_frames": total_frames,
        "sample_interval": sample_interval,
    }
    return grid_images, metadata


def compare_scores(name, scores_a, scores_b, label_a="A", label_b="B"):
    """Compare two score arrays and print detailed stats."""
    if len(scores_a) != len(scores_b):
        print(f"  [{name}] LENGTH MISMATCH: {label_a}={len(scores_a)}, {label_b}={len(scores_b)}")
        min_len = min(len(scores_a), len(scores_b))
        scores_a = scores_a[:min_len]
        scores_b = scores_b[:min_len]

    a = np.array(scores_a)
    b = np.array(scores_b)
    diff = np.abs(a - b)

    print(f"  [{name}] {label_a} vs {label_b}:")
    print(f"    Num scores: {len(a)}")
    print(f"    Max abs diff:  {diff.max():.8f}")
    print(f"    Mean abs diff: {diff.mean():.8f}")
    print(f"    Exact matches: {np.sum(diff == 0)}/{len(a)} ({100*np.sum(diff==0)/len(a):.1f}%)")
    print(f"    Within 1e-4:   {np.sum(diff < 1e-4)}/{len(a)} ({100*np.sum(diff<1e-4)/len(a):.1f}%)")
    print(f"    Within 1e-3:   {np.sum(diff < 1e-3)}/{len(a)} ({100*np.sum(diff<1e-3)/len(a):.1f}%)")

    # Show first 5 scores side by side
    n_show = min(5, len(a))
    print(f"    First {n_show} scores:")
    for i in range(n_show):
        marker = " ✓" if diff[i] < 1e-4 else f" Δ={diff[i]:.6f}"
        print(f"      [{i}] {label_a}={a[i]:.8f}  {label_b}={b[i]:.8f}{marker}")

    return diff


def main():
    IMAGE_SIZE = 384
    TARGET_FPS = 4
    QUERY_INTERVAL = 4
    BATCH_SIZE = 32
    DATA_ROOT = os.environ.get("HIVAU_VIDEO_ROOT", "/path/to/HIVAU-70k/videos")
    CKPT_ROOT = os.environ.get("CKPT_ROOT", os.path.join(PROJECT_ROOT, "ckpt"))

    # Test videos: 3 UCF + 3 XD
    ucf_eval_path = os.environ.get(
        "UCF_EVAL_RESULTS",
        os.path.join(PROJECT_ROOT, "eval_results/vad/ucf/video_results.json"))
    xd_eval_path = os.environ.get(
        "XD_EVAL_RESULTS",
        os.path.join(PROJECT_ROOT, "eval_results/vad/xd/video_results.json"))

    with open(ucf_eval_path) as f:
        ucf_results = json.load(f)
    with open(xd_eval_path) as f:
        xd_results = json.load(f)

    # Pick 3 UCF + 3 XD (first 3 from each)
    ucf_keys = list(ucf_results.keys())[:3]
    xd_keys = list(xd_results.keys())[:3]

    test_videos = []
    for key in ucf_keys:
        r = ucf_results[key]
        video_path = os.path.join(DATA_ROOT, "ucf-crime/videos/test", r["video_name"] + ".mp4")
        test_videos.append({
            "key": key,
            "dataset": "UCF",
            "video_path": video_path,
            "eval_scores": r["query_scores"],
            "eval_n_frames": r["n_frames"],
            "eval_num_queries": r["num_queries"],
        })

    for key in xd_keys:
        r = xd_results[key]
        video_path = os.path.join(DATA_ROOT, "xd-violence/videos/test", r["video_name"] + ".mp4")
        test_videos.append({
            "key": key,
            "dataset": "XD",
            "video_path": video_path,
            "eval_scores": r["query_scores"],
            "eval_n_frames": r["n_frames"],
            "eval_num_queries": r["num_queries"],
        })

    # Verify video files exist
    for v in test_videos:
        exists = os.path.exists(v["video_path"])
        print(f"{'✓' if exists else '✗'} [{v['dataset']}] {v['key']} -> {os.path.basename(v['video_path'])} (exists={exists})")
        if not exists:
            print(f"  ERROR: {v['video_path']} not found!")

    # Load model
    print("\n" + "=" * 70)
    print("Loading PaliGemma model...")
    scorer = PaliGemmaScorer(
        model_path=os.environ.get(
            "PALIGEMMA_MODEL_PATH",
            os.path.join(CKPT_ROOT, "paligemma2-3b-mix-448")),
        lora_path=os.environ.get(
            "PALIGEMMA_LORA_PATH",
            os.path.join(CKPT_ROOT, "paligemma2-3b-vad-lora-384-combined-binary/training_384")),
        device="cuda:0",
        image_size=IMAGE_SIZE,
        streamforest_weights_path=os.environ.get(
            "PALIGEMMA_SF_WEIGHTS",
            os.path.join(CKPT_ROOT, "extracted_weights/streamforest_vision_encoder_with_prefix.safetensors")),
        vision_feature_layer=-2,
    )

    # Must match eval and precompute exactly: style="detail"
    from vad.get_prompt import get_grid_prompt
    prompt = get_grid_prompt(add_special_tokens=False)
    print(f"Using prompt: '{prompt}'")

    # Process each video
    print("\n" + "=" * 70)
    print("Processing test videos...")

    all_diffs_precompute_vs_eval = []
    all_diffs_seek_vs_eval = []
    all_diffs_precompute_vs_seek = []

    for v in test_videos:
        if not os.path.exists(v["video_path"]):
            print(f"\nSKIPPING {v['key']} - file not found")
            continue

        print(f"\n{'='*70}")
        print(f"[{v['dataset']}] {v['key']}")
        print(f"  Video: {v['video_path']}")
        print(f"  Eval: n_frames={v['eval_n_frames']}, num_queries={v['eval_num_queries']}")

        # 1. Precompute approach (sequential grab)
        t0 = time.time()
        grids_grab, meta_grab = read_video_grids(
            v["video_path"], IMAGE_SIZE, TARGET_FPS, QUERY_INTERVAL)
        t_grab = time.time() - t0

        if grids_grab is None:
            print(f"  ERROR: read_video_grids returned None")
            continue

        print(f"  Grab read: {len(grids_grab)} grids, n_frames={meta_grab['n_frames']}, "
              f"sample_interval={meta_grab['sample_interval']} ({t_grab:.2f}s)")

        # 2. Eval approach (seek-based)
        t0 = time.time()
        grids_seek, meta_seek = read_video_grids_seekbased(
            v["video_path"], IMAGE_SIZE, TARGET_FPS, QUERY_INTERVAL)
        t_seek = time.time() - t0

        if grids_seek is None:
            print(f"  ERROR: read_video_grids_seekbased returned None")
            continue

        print(f"  Seek read: {len(grids_seek)} grids, n_frames={meta_seek['n_frames']}, "
              f"sample_interval={meta_seek['sample_interval']} ({t_seek:.2f}s)")

        # Compare grid images (pixel-level) between grab and seek
        if len(grids_grab) == len(grids_seek):
            pixel_diffs = []
            for i, (g1, g2) in enumerate(zip(grids_grab, grids_seek)):
                arr1 = np.array(g1)
                arr2 = np.array(g2)
                pixel_diffs.append(np.abs(arr1.astype(float) - arr2.astype(float)).mean())
            mean_pixel_diff = np.mean(pixel_diffs)
            max_pixel_diff = np.max(pixel_diffs)
            identical_grids = sum(1 for d in pixel_diffs if d == 0)
            print(f"  Grid pixel comparison (grab vs seek): "
                  f"identical={identical_grids}/{len(pixel_diffs)}, "
                  f"mean_diff={mean_pixel_diff:.4f}, max_diff={max_pixel_diff:.4f}")

        # 3. Score with model - precompute grids (NO rounding, to match eval precision)
        t0 = time.time()
        scores_grab_raw = scorer.batch_score_grids(grids_grab, prompt, BATCH_SIZE)
        t_score = time.time() - t0
        # Note: batch_score_grids rounds to 4 decimals. Get unrounded for fair comparison.
        # We'll just compare rounded too.
        print(f"  Scored grab grids: {len(scores_grab_raw)} scores ({t_score:.2f}s)")

        # 4. Score with model - seek grids
        t0 = time.time()
        scores_seek_raw = scorer.batch_score_grids(grids_seek, prompt, BATCH_SIZE)
        t_score = time.time() - t0
        print(f"  Scored seek grids: {len(scores_seek_raw)} scores ({t_score:.2f}s)")

        # 5. Compare all three
        print()
        d1 = compare_scores(v["key"], scores_grab_raw, v["eval_scores"],
                            "Precompute(grab)", "Eval(stored)")
        d2 = compare_scores(v["key"], scores_seek_raw, v["eval_scores"],
                            "Seek-based", "Eval(stored)")
        d3 = compare_scores(v["key"], scores_grab_raw, scores_seek_raw,
                            "Precompute(grab)", "Seek-based")

        all_diffs_precompute_vs_eval.extend(d1.tolist())
        all_diffs_seek_vs_eval.extend(d2.tolist())
        all_diffs_precompute_vs_seek.extend(d3.tolist())

        # Free GPU memory
        del grids_grab, grids_seek
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    for name, diffs in [
        ("Precompute(grab) vs Eval(stored)", all_diffs_precompute_vs_eval),
        ("Seek-based vs Eval(stored)", all_diffs_seek_vs_eval),
        ("Precompute(grab) vs Seek-based", all_diffs_precompute_vs_seek),
    ]:
        if not diffs:
            continue
        d = np.array(diffs)
        print(f"\n{name}:")
        print(f"  Total scores compared: {len(d)}")
        print(f"  Max abs diff:  {d.max():.8f}")
        print(f"  Mean abs diff: {d.mean():.8f}")
        print(f"  Exact matches:   {np.sum(d == 0)}/{len(d)} ({100*np.sum(d==0)/len(d):.1f}%)")
        print(f"  Within 1e-4:     {np.sum(d < 1e-4)}/{len(d)} ({100*np.sum(d<1e-4)/len(d):.1f}%)")
        print(f"  Within 1e-3:     {np.sum(d < 1e-3)}/{len(d)} ({100*np.sum(d<1e-3)/len(d):.1f}%)")
        print(f"  Within 1e-2:     {np.sum(d < 1e-2)}/{len(d)} ({100*np.sum(d<1e-2)/len(d):.1f}%)")

    print("\nDone!")


if __name__ == "__main__":
    main()
