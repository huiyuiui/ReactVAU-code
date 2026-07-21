# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import logging
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc, precision_recall_curve


def get_detection_prompt_yes_no():
    """
    Returns a simple Yes/No prompt for anomaly detection.
    Designed to extract anomaly score from Yes/No token logits.
    """
    question = """Based on the video content you have seen so far, is there any anomaly or abnormal event happening? Answer with only 'Yes' or 'No'."""
    return question.strip()


def get_detection_prompt_binary(dataset: str = "ucf-crime"):
    """
    Returns a simple streaming anomaly prompt kept for compatibility.

    The current ReactVAU VAD path uses PaliGemma grid prompts from
    ``vad.get_prompt`` instead of dataset-specific ASK-HINT prompts.
    """
    del dataset
    return get_detection_prompt_yes_no()


def gaussian_kernel(size, sigma):
    """Generate 1D Gaussian kernel"""
    kernel = np.exp(-np.linspace(-size//2, size//2, size)**2 / (2*sigma**2))
    return kernel / kernel.sum()


def gaussian_smooth_1d(data, size=5, sigma=1.0):
    """Apply 1D Gaussian smoothing"""
    if len(data) < size:
        return data
    kernel = gaussian_kernel(size, sigma)
    smoothed_data = np.convolve(data, kernel, mode='same')
    return smoothed_data


class OnlineSmoother:
    """Causal online score smoother combining EMA and Peak Holding.

    Designed for streaming VAD: strictly causal (no future frames), O(1) per step.

    Algorithm per step:
        1. EMA:        ema_t = α * raw_t + (1 - α) * smooth_{t-1}
        2. Peak Hold:  smooth_t = max(ema_t, β * smooth_{t-1})

    The EMA provides temporal smoothing (reduces jitter).
    The peak hold prevents sharp score drops mid-event, bridging short
    detection gaps that PG may produce within a continuous anomaly.

    Parameters:
        alpha: EMA coefficient (higher = more responsive, lower = smoother).
               0.3-0.4 recommended for 1-query-per-second cadence.
        beta:  Peak decay factor (higher = slower decay after a peak).
               0.85-0.90 recommended. At β=0.90, a peak decays to 50%
               in ~7 steps (7 seconds at 1 QPS).
    """

    def __init__(self, alpha: float = 0.35, beta: float = 0.88):
        self.alpha = alpha
        self.beta = beta
        self.prev_smooth = 0.0

    def reset(self):
        """Reset state for a new video."""
        self.prev_smooth = 0.0

    def step(self, raw_score: float) -> float:
        """Process one score and return the smoothed value."""
        # Step 1: Exponential Moving Average (causal low-pass filter)
        ema = self.alpha * raw_score + (1.0 - self.alpha) * self.prev_smooth
        # Step 2: Peak holding with decay (prevents mid-event dropout)
        smoothed = max(ema, self.beta * self.prev_smooth)
        self.prev_smooth = smoothed
        return smoothed

    def smooth_sequence(self, scores) -> list:
        """Smooth a full sequence (convenience wrapper). Strictly causal."""
        self.reset()
        return [self.step(s) for s in scores]


def online_smooth_scores(scores, alpha: float = 0.35, beta: float = 0.88) -> list:
    """Convenience function: apply causal EMA + Peak Holding to a score sequence.

    Args:
        scores: Iterable of raw anomaly scores (one per query/second).
        alpha:  EMA coefficient (0.35 default — good balance for 1 QPS).
        beta:   Peak decay factor (0.88 default — peak halves in ~6 steps).

    Returns:
        List of smoothed scores, same length as input.
    """
    smoother = OnlineSmoother(alpha=alpha, beta=beta)
    return smoother.smooth_sequence(scores)


def compute_metrics(all_frame_scores: list, all_frame_labels: list) -> dict:
    """
    Compute AUC and AP metrics.
    
    Args:
        all_frame_scores: List of frame-level anomaly scores
        all_frame_labels: List of frame-level ground truth labels (0 or 1)
    
    Returns:
        dict with AUC, AP, etc.
    """
    scores = np.array(all_frame_scores)
    labels = np.array(all_frame_labels)
    
    # Check if we have both classes
    if len(np.unique(labels)) < 2:
        logging.warning("Only one class present in labels, metrics may be undefined")
        return {
            "roc_auc": 0.0,
            "pr_auc": 0.0,
            "ap": 0.0,
            "fpr": [],
            "tpr": [],
        }
    
    # ROC-AUC
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # PR-AUC (Average Precision)
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    
    # sklearn's average_precision_score
    ap = average_precision_score(labels, scores)
    
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "ap": ap,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def load_anno_txt(anno_txt_path, dataset, video_dir=None):
    """Parse official anno.txt into a lightweight annotation dict.

    Both UCF-Crime and XD-Violence annotations use **frame numbers**.

    UCF-Crime format per line::

        Abuse028_x264.mp4  Abuse  165  240  -1  -1

    XD-Violence format per line::

        v=-fOWSLV6Esw__#1_label_B4-0-0 20 772
        Bad.Boys.1995__#01-11-55_01-12-40_label_G-B2-B6 157 180 185 244 ...

    Normal XD videos (``_label_A``) have ``-1 -1`` in the anno.txt (if present)
    or are discovered from disk when ``video_dir`` is provided.

    Args:
        anno_txt_path: Path to the official ``*_anno.txt`` file.
        dataset: ``"ucf-crime"`` or ``"xd-violence"``.
        video_dir: Base dataset directory (e.g. ``.../HIVAU-70k``).
            Used for XD-Violence to discover normal videos not yet in anno.txt.

    Returns:
        dict: ``{video_key: {"video_name": str, "anomaly": int,
                             "label": [str], "intervals_raw": [[s, e], ...]}}``
        All intervals are in **frame numbers**.
    """
    annotations = {}

    if dataset == "ucf-crime":
        with open(anno_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                filename = parts[0]                          # e.g. "Abuse028_x264.mp4"
                video_name = os.path.splitext(filename)[0]  # "Abuse028_x264"
                category = parts[1]
                is_anomaly = int(category != "Normal")

                intervals = []
                i = 2
                while i + 1 < len(parts):
                    s, e = int(parts[i]), int(parts[i + 1])
                    if s == -1:
                        break
                    intervals.append([s, e])
                    i += 2

                annotations[video_name] = {
                    "video_name": video_name,
                    "anomaly": is_anomaly,
                    "label": [category],
                    "intervals_raw": intervals,
                }

    elif dataset == "xd-violence":
        with open(anno_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                raw_name = parts[0]
                # Some entries may carry an embedded .mp4 suffix
                video_name = raw_name[:-4] if raw_name.endswith('.mp4') else raw_name

                # Determine if normal (_label_A) or anomalous
                is_normal = video_name.endswith('_label_A') or '_label_A' in video_name

                intervals = []
                i = 1
                while i + 1 < len(parts):
                    s, e = int(parts[i]), int(parts[i + 1])
                    if s == -1:
                        break
                    intervals.append([s, e])
                    i += 2

                if is_normal:
                    labels = ["Normal"]
                    is_anomaly = 0
                else:
                    # Extract label tokens from the filename suffix
                    label_suffix = video_name.split('_label_')[-1] if '_label_' in video_name else ''
                    labels = [p for p in label_suffix.split('-') if p and p != '0']
                    if not labels:
                        labels = ['Anomaly']
                    is_anomaly = 1

                annotations[video_name] = {
                    "video_name": video_name,
                    "anomaly": is_anomaly,
                    "label": labels,
                    "intervals_raw": intervals,
                }

        # Discover normal videos from disk that are not yet in anno.txt
        if video_dir is not None:
            test_dir = os.path.join(video_dir, "videos", "xd-violence", "videos", "test")
            if os.path.isdir(test_dir):
                for fn in sorted(os.listdir(test_dir)):
                    if fn.endswith("_label_A.mp4"):
                        video_name = fn[:-4]  # strip .mp4
                        if video_name not in annotations:
                            annotations[video_name] = {
                                "video_name": video_name,
                                "anomaly": 0,
                                "label": ["Normal"],
                                "intervals_raw": [],
                            }

    return annotations


def make_gt_labels_from_anno(anno, total_frames, fps=30.0):
    """Create binary frame-level ground-truth labels from an annotation entry.

    Both UCF-Crime and XD-Violence use frame numbers, so ``fps`` is unused
    but kept for API compatibility.

    Args:
        anno: dict returned by :func:`load_anno_txt` for a single video.
        total_frames: Actual number of frames in the video.
        fps: Unused (kept for backward compatibility).

    Returns:
        ``np.ndarray`` of shape ``(total_frames,)`` with values in ``{0, 1}``.
    """
    labels = np.zeros(total_frames, dtype=np.float32)

    for s, e in anno.get("intervals_raw", []):
        start_fr = max(0, int(s))
        end_fr = min(total_frames, int(e) + 1)
        labels[start_fr:end_fr] = 1.0

    return labels
