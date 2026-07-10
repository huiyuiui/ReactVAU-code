"""
Generate PaliGemma Training Data for Video Anomaly Detection (V4)

Strategy:
1. XD-Violence (Action Focus):
   - Positives: 0.5s stride sliding window on anomaly segments (1s span)
   - Hard Negatives: Sparse sliding window on normal segments (2s span)

2. UCF-Crime (Surveillance Focus):
   - Positives: 0.5s stride sliding window on anomaly segments (1s span)
   - Hard Negatives: Sliding window on normal segments of anomaly videos (2s span)
   - Easy Negatives: Dynamic uniform sampling from Normal_Videos (2-4s span)

3. Final Balance: Positive : Hard_Negative : Easy_Negative ≈ 1 : 1 : 1

Features:
- Different time spans for different sample types
- No timestamp overlay (temporal order conveyed via prompt)
- Video-centric processing: Open each video only once
- Sequential frame reading with frame buffer to minimize seeks
- Separate output JSON files for each dataset + combined

Note: The 2x2 grid follows reading order: top-left(1) → top-right(2) → bottom-left(3) → bottom-right(4)
      This temporal order should be specified in the training prompt.
"""
import os
import json
import cv2
import random
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
import re

# ===================== Configuration =====================
@dataclass
class SamplingConfig:
    """Dataset-specific sampling configuration."""
    pos_stride_seconds: float    # Stride in seconds for positive samples
    hard_neg_stride_seconds: float  # Stride in seconds for hard negatives
    pos_span_seconds: float      # Time span for positive samples (e.g., 1.0s)
    hard_neg_span_seconds: float # Time span for hard negatives (e.g., 2.0s)
    easy_neg_span_seconds: float # Time span for easy negatives (e.g., 2-4s)


# Dataset-specific configurations
XD_VIOLENCE_CONFIG = SamplingConfig(
    pos_stride_seconds=0.5,      # 0.5s stride
    hard_neg_stride_seconds=1.0, # 1s stride for hard negatives
    pos_span_seconds=1.0,        # 1 second span (4 frames)
    hard_neg_span_seconds=2.0,   # 2 seconds span (4 frames)
    easy_neg_span_seconds=2.0,   # Not used for XD-Violence
)

UCF_CRIME_CONFIG = SamplingConfig(
    pos_stride_seconds=0.5,      # 0.5s stride
    hard_neg_stride_seconds=1.0, # 1s stride for hard negatives
    pos_span_seconds=1.0,        # 1 second span
    hard_neg_span_seconds=2.0,   # 2 seconds span
    easy_neg_span_seconds=2.0,   # Default 2s, can be adjusted to 4s
)

# Grid settings
QUERY_INTERVAL = 4  # Number of frames per grid
GRID_SIZE = (2, 2)

# Grid layout for 384x384 vision encoder with 14x14 ViT patches
# Each frame cell is 182x182 (= 13 patches × 14px), with a 14px gap (= 1 patch)
# between cells so no ViT patch ever straddles two frame boundaries.
#
#   [frame1: 0:182 , 0:182 ] [frame2: 0:182 , 196:378]
#   [frame3: 196:378, 0:182 ] [frame4: 196:378, 196:378]
#
# Remaining 6px at the far edge (378-383) are zero-padded.
CELL_SIZE = 182        # 13 * 14 = 182 pixels per frame cell
TOTAL_SIZE = 384       # 384x384 canvas (standard for SigLIP / PaliGemma)
GRID_GAP = 14          # 1 ViT patch gap between cells

# Pre-computed cell slice positions (row_slice, col_slice) for each of 4 frames
CELL_POSITIONS = [
    (slice(0, CELL_SIZE), slice(0, CELL_SIZE)),                                      # top-left
    (slice(0, CELL_SIZE), slice(CELL_SIZE + GRID_GAP, 2 * CELL_SIZE + GRID_GAP)),   # top-right
    (slice(CELL_SIZE + GRID_GAP, 2 * CELL_SIZE + GRID_GAP), slice(0, CELL_SIZE)),   # bottom-left
    (slice(CELL_SIZE + GRID_GAP, 2 * CELL_SIZE + GRID_GAP),
     slice(CELL_SIZE + GRID_GAP, 2 * CELL_SIZE + GRID_GAP)),                        # bottom-right
]

# SigLIP-style per-frame transform and normalization
_FRAME_TRANSFORM = T.Compose([
    T.Resize(CELL_SIZE, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(CELL_SIZE),
    T.ToTensor(),
])
_NORMALIZE = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


# ===================== Helper Functions =====================

def sanitize_filename(name: str) -> str:
    """
    Sanitize filename to be safe for filesystem.
    Removes or replaces problematic characters.
    """
    name = os.path.basename(name)
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = name.strip(' .')
    return name


def get_frame_intervals(n_frames: int, events_frames: list) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Determine frame intervals for sampling.
    Returns (normal_intervals, anomaly_intervals)
    """
    if not events_frames:
        return [(0, n_frames - 1)], []
    
    sorted_events = sorted(events_frames, key=lambda x: x[0])
    normal_intervals = []
    anomaly_intervals = []
    
    current_pos = 0
    for event in sorted_events:
        start_frame, end_frame = event[0], event[1]
        if current_pos < start_frame:
            normal_intervals.append((current_pos, start_frame - 1))
        anomaly_intervals.append((start_frame, end_frame))
        current_pos = end_frame + 1
    
    if current_pos < n_frames:
        normal_intervals.append((current_pos, n_frames - 1))
    
    return normal_intervals, anomaly_intervals


def calculate_frame_spacing(fps: float, span_seconds: float, n_frames: int = QUERY_INTERVAL) -> int:
    """
    Calculate frame spacing to achieve desired time span.
    
    Args:
        fps: Video FPS
        span_seconds: Desired time span in seconds
        n_frames: Number of frames to sample
    
    Returns:
        Frame spacing (integer)
    """
    total_span_frames = span_seconds * fps
    spacing = total_span_frames / (n_frames - 1)
    return max(1, int(round(spacing)))


def sample_frames_with_stride(
    interval: Tuple[int, int], 
    stride_seconds: float,
    span_seconds: float,
    fps: float,
    query_interval: int = QUERY_INTERVAL
) -> Tuple[List[List[int]], float]:
    """
    Sample frame groups with fixed stride based on time.
    
    Returns:
        Tuple of (frame_groups, actual_time_span)
    """
    start_frame, end_frame = interval
    
    frame_spacing = calculate_frame_spacing(fps, span_seconds, query_interval)
    group_span = (query_interval - 1) * frame_spacing
    
    stride_frames = max(1, int(round(stride_seconds * fps)))
    
    actual_time_span = group_span / fps
    
    groups = []
    current_start = start_frame
    
    while current_start + group_span <= end_frame:
        group = [current_start + i * frame_spacing for i in range(query_interval)]
        groups.append(group)
        current_start += stride_frames
    
    return groups, actual_time_span


def sample_uniform_frames(
    total_frames: int,
    target_count: int,
    fps: float,
    span_seconds: float,
    query_interval: int = QUERY_INTERVAL
) -> Tuple[List[List[int]], float]:
    """
    Uniformly sample frame groups from entire video with specified time span.
    
    Returns:
        Tuple of (frame_groups, actual_time_span)
    """
    frame_spacing = calculate_frame_spacing(fps, span_seconds, query_interval)
    group_span = (query_interval - 1) * frame_spacing
    
    if total_frames <= group_span:
        return [], 0.0
    
    available_starts = total_frames - group_span
    stride = max(1, available_starts // target_count)
    
    actual_time_span = group_span / fps
    
    groups = []
    current_start = 0
    
    while current_start + group_span < total_frames and len(groups) < target_count:
        group = [current_start + i * frame_spacing for i in range(query_interval)]
        groups.append(group)
        current_start += stride
    
    return groups, actual_time_span


def create_grid_tensor(frames: list) -> Optional[torch.Tensor]:
    """
    Create a 384x384 float32 tensor with 4 frames arranged in a 2x2 grid.

    Each frame cell is 182x182 (13 ViT patches of 14px each).  A 14px gap
    (exactly 1 patch) is left between the cells so that no ViT patch ever
    spans two frame boundaries.  The remaining 6px at the far right/bottom
    edge is left as zeros.

    Grid layout (reading / temporal order):
        [1: top-left ] [2: top-right  ]
        [3: bot-left ] [4: bot-right  ]

    Returns a normalized (mean=0.5, std=0.5) fp32 tensor of shape (3, 384, 384),
    or None if fewer than QUERY_INTERVAL valid frames are available.
    """
    valid = [f for f in frames if f is not None]
    if len(valid) < QUERY_INTERVAL:
        return None

    grid = torch.zeros((3, TOTAL_SIZE, TOTAL_SIZE), dtype=torch.float32)
    for frame, (row_s, col_s) in zip(valid[:QUERY_INTERVAL], CELL_POSITIONS):
        grid[:, row_s, col_s] = _FRAME_TRANSFORM(frame)

    return _NORMALIZE(grid)


def get_label_text(label: list) -> str:
    """Convert label list to text string."""
    if not label:
        return "Normal"
    return ", ".join(label)


# ===================== Optimized Frame Reader =====================

class OptimizedFrameReader:
    """
    Optimized frame reader that minimizes disk seeks.
    """
    
    def __init__(self, video_path: str, cache_size: int = 100):
        self.video_path = video_path
        self.cap = None
        self.cache_size = cache_size
        self.frame_cache: Dict[int, Image.Image] = {}
        self.cache_order: List[int] = []
        self.current_pos = -1
        self.total_frames = 0
        self.fps = 30.0
        
    def open(self) -> bool:
        """Open video file."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            return False
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.current_pos = 0
        return True
    
    def close(self):
        """Close video file and clear cache."""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.frame_cache.clear()
        self.cache_order.clear()
        self.current_pos = -1
    
    def _add_to_cache(self, frame_idx: int, frame: Image.Image):
        """Add frame to cache with LRU eviction."""
        if frame_idx in self.frame_cache:
            self.cache_order.remove(frame_idx)
            self.cache_order.append(frame_idx)
            return
        
        while len(self.frame_cache) >= self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.frame_cache[oldest]
        
        self.frame_cache[frame_idx] = frame
        self.cache_order.append(frame_idx)
    
    def _read_frame_at(self, frame_idx: int) -> Optional[Image.Image]:
        """Read a single frame, using sequential read when possible."""
        if self.cap is None:
            return None
        
        if frame_idx in self.frame_cache:
            return self.frame_cache[frame_idx]
        
        frames_ahead = frame_idx - self.current_pos
        
        if 0 < frames_ahead <= 10:
            for _ in range(frames_ahead):
                ret, _ = self.cap.read()
                if not ret:
                    return None
                self.current_pos += 1
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self.current_pos = frame_idx
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        self.current_pos = frame_idx + 1
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        self._add_to_cache(frame_idx, pil_frame)
        
        return pil_frame
    
    def batch_read_groups(
        self, 
        frame_groups: List[List[int]]
    ) -> List[List[Image.Image]]:
        """Batch read multiple frame groups efficiently."""
        if not frame_groups:
            return []
        
        all_frames: Set[int] = set()
        for group in frame_groups:
            all_frames.update(group)
        
        sorted_frames = sorted(all_frames)
        
        for frame_idx in sorted_frames:
            if frame_idx not in self.frame_cache:
                self._read_frame_at(frame_idx)
        
        results = []
        for group in frame_groups:
            group_frames = [self.frame_cache.get(idx) for idx in group]
            results.append(group_frames)
        
        return results


# ===================== Main Processing Classes =====================

class SampleCollector:
    """Collects and balances samples across categories for a single dataset."""
    
    def __init__(self, dataset_name: str = ""):
        self.dataset_name = dataset_name
        self.positives = []
        self.hard_negatives = []
        self.easy_negatives = []
        
    def add_positive(self, sample: dict):
        self.positives.append(sample)
        
    def add_hard_negative(self, sample: dict):
        self.hard_negatives.append(sample)
        
    def add_easy_negative(self, sample: dict):
        self.easy_negatives.append(sample)
    
    def get_balanced_samples(self, balance_ratio: Tuple[int, int, int] = (1, 1, 1)) -> List[dict]:
        """Balance samples according to ratio."""
        n_pos = len(self.positives)
        
        if n_pos == 0:
            print(f"Warning: No positive samples found for {self.dataset_name}!")
            return self.hard_negatives + self.easy_negatives
        
        r_pos, r_hard, r_easy = balance_ratio
        target_hard = (n_pos * r_hard) // r_pos
        target_easy = (n_pos * r_easy) // r_pos
        
        hard_sampled = self.hard_negatives
        if len(hard_sampled) > target_hard:
            hard_sampled = random.sample(hard_sampled, target_hard)
        
        easy_sampled = self.easy_negatives
        if len(easy_sampled) > target_easy:
            easy_sampled = random.sample(easy_sampled, target_easy)
        
        print(f"\n=== Sample Balance ({self.dataset_name}) ===")
        print(f"Positives:       {len(self.positives)} -> {len(self.positives)}")
        print(f"Hard Negatives:  {len(self.hard_negatives)} -> {len(hard_sampled)}")
        print(f"Easy Negatives:  {len(self.easy_negatives)} -> {len(easy_sampled)}")
        
        all_samples = self.positives + hard_sampled + easy_sampled
        random.shuffle(all_samples)
        
        return all_samples
    
    def get_all_samples(self) -> List[dict]:
        """Get all samples without balancing."""
        all_samples = self.positives + self.hard_negatives + self.easy_negatives
        return all_samples
    
    def get_stats(self) -> dict:
        """Get statistics about collected samples."""
        return {
            "positives": len(self.positives),
            "hard_negatives": len(self.hard_negatives),
            "easy_negatives": len(self.easy_negatives),
            "total": len(self.positives) + len(self.hard_negatives) + len(self.easy_negatives)
        }


class VideoProcessor:
    """
    Optimized video processor that processes each video only once.
    Uses batch frame reading for efficiency.
    """
    
    def __init__(
        self,
        config: SamplingConfig,
        output_pt_root: str,
        video_dir: str,
        video_subdir: str,
        video_ext: str = ".mp4",
        frame_cache_size: int = 200
    ):
        self.config = config
        self.output_pt_root = output_pt_root
        self.video_dir = video_dir
        self.video_subdir = video_subdir
        self.video_ext = video_ext
        self.frame_cache_size = frame_cache_size
        self.sample_counter = defaultdict(int)
    
    def process_anomaly_video(
        self,
        video_name: str,
        video_info: dict,
        collector: SampleCollector
    ) -> Tuple[int, int]:
        """
        Process an anomaly video efficiently.
        Opens video once and batch processes all frame groups.
        """
        safe_video_name = sanitize_filename(video_name)
        n_frames = video_info["n_frames"]
        fps = video_info["fps"]
        label = video_info.get("label", [])
        anomaly_interval = video_info.get("anomaly_interval", [])
        events_frames = [[int(s * fps), int(e * fps)] for s, e in anomaly_interval]
        
        video_path = os.path.join(
            self.video_dir, self.video_subdir, video_name + self.video_ext
        )
        
        if not os.path.exists(video_path):
            return 0, 0
        
        # Get intervals
        normal_intervals, anomaly_intervals = get_frame_intervals(n_frames, events_frames)
        
        # Generate positive frame groups (1s span)
        pos_groups = []
        for interval in anomaly_intervals:
            groups, _ = sample_frames_with_stride(
                interval, 
                stride_seconds=self.config.pos_stride_seconds,
                span_seconds=self.config.pos_span_seconds,
                fps=fps
            )
            pos_groups.extend(groups)
        
        # Generate hard negative frame groups (2s span)
        hard_neg_groups = []
        for interval in normal_intervals:
            groups, _ = sample_frames_with_stride(
                interval,
                stride_seconds=self.config.hard_neg_stride_seconds,
                span_seconds=self.config.hard_neg_span_seconds,
                fps=fps
            )
            hard_neg_groups.extend(groups)
        
        if not pos_groups and not hard_neg_groups:
            return 0, 0
        
        # Open video once with optimized reader
        reader = OptimizedFrameReader(video_path, cache_size=self.frame_cache_size)
        if not reader.open():
            return 0, 0
        
        # Create output folder
        video_output_folder = os.path.join(self.output_pt_root, safe_video_name)
        os.makedirs(video_output_folder, exist_ok=True)
        
        n_pos, n_hard_neg = 0, 0
        
        try:
            # Process positive groups
            if pos_groups:
                pos_frame_lists = reader.batch_read_groups(pos_groups)
                for frames in pos_frame_lists:
                    sample = self._create_sample_from_frames(
                        frames=frames,
                        video_name=safe_video_name,
                        output_folder=video_output_folder,
                        is_anomaly=True,
                        label=label
                    )
                    if sample:
                        collector.add_positive(sample)
                        n_pos += 1
            
            # Process hard negative groups
            if hard_neg_groups:
                hard_neg_frame_lists = reader.batch_read_groups(hard_neg_groups)
                for frames in hard_neg_frame_lists:
                    sample = self._create_sample_from_frames(
                        frames=frames,
                        video_name=safe_video_name,
                        output_folder=video_output_folder,
                        is_anomaly=False,
                        label=[]
                    )
                    if sample:
                        collector.add_hard_negative(sample)
                        n_hard_neg += 1
        
        finally:
            reader.close()
        
        return n_pos, n_hard_neg
    
    def process_normal_video(
        self,
        video_name: str,
        video_info: dict,
        collector: SampleCollector,
        target_samples_per_video: int,
        span_seconds: float = 2.0
    ) -> int:
        """Process a normal video for easy negatives."""
        safe_video_name = sanitize_filename(video_name)
        n_frames = video_info["n_frames"]
        fps = video_info["fps"]
        
        video_path = os.path.join(
            self.video_dir, self.video_subdir, video_name + self.video_ext
        )
        
        if not os.path.exists(video_path):
            return 0
        
        # Generate frame groups with specified time span
        groups, _ = sample_uniform_frames(
            total_frames=n_frames,
            target_count=target_samples_per_video,
            fps=fps,
            span_seconds=span_seconds
        )
        
        if not groups:
            return 0
        
        # Open video with optimized reader
        reader = OptimizedFrameReader(video_path, cache_size=self.frame_cache_size)
        if not reader.open():
            return 0
        
        video_output_folder = os.path.join(self.output_pt_root, safe_video_name)
        os.makedirs(video_output_folder, exist_ok=True)
        
        n_easy = 0
        
        try:
            frame_lists = reader.batch_read_groups(groups)
            for frames in frame_lists:
                sample = self._create_sample_from_frames(
                    frames=frames,
                    video_name=safe_video_name,
                    output_folder=video_output_folder,
                    is_anomaly=False,
                    label=[]
                )
                if sample:
                    collector.add_easy_negative(sample)
                    n_easy += 1
        
        finally:
            reader.close()
        
        return n_easy
    
    def _create_sample_from_frames(
        self,
        frames: List[Optional[Image.Image]],
        video_name: str,
        output_folder: str,
        is_anomaly: bool,
        label: List[str]
    ) -> Optional[dict]:
        """Create a training sample from pre-loaded frames, saving as a .pt tensor."""
        grid_tensor = create_grid_tensor(frames)
        if grid_tensor is None:
            return None

        self.sample_counter[video_name] += 1
        sample_idx = self.sample_counter[video_name]

        pt_filename = f"{video_name}_{sample_idx:05d}.pt"
        save_path = os.path.join(output_folder, pt_filename)
        # Save as half-precision to halve disk usage; fp32 is restored during training
        torch.save(grid_tensor.half(), save_path)

        relative_path = f"train_pt/{video_name}/{pt_filename}"

        if is_anomaly:
            suffix_text = f"Detection: Yes. Event: {get_label_text(label)}."
        else:
            suffix_text = "Detection: No. Event: Normal."

        return {
            "pt": relative_path,
            "suffix": suffix_text
        }



# ===================== Dataset-Specific Processing =====================

def process_xd_violence(
    annotations: dict,
    video_dir: str,
    output_pt_root: str,
    video_subdir: str,
    collector: SampleCollector,
    test_mode: bool = False,
    test_samples: int = 5
) -> dict:
    """Process XD-Violence dataset."""
    print("\n" + "="*60)
    print("Processing XD-Violence Dataset")
    print("="*60)
    print(f"Positive: {XD_VIOLENCE_CONFIG.pos_span_seconds}s span, {XD_VIOLENCE_CONFIG.pos_stride_seconds}s stride")
    print(f"Hard Negative: {XD_VIOLENCE_CONFIG.hard_neg_span_seconds}s span, {XD_VIOLENCE_CONFIG.hard_neg_stride_seconds}s stride")
    
    processor = VideoProcessor(
        config=XD_VIOLENCE_CONFIG,
        output_pt_root=output_pt_root,
        video_dir=video_dir,
        video_subdir=video_subdir,
    )
    
    video_keys = list(annotations.keys())
    anomaly_keys = [k for k in video_keys if annotations[k].get("anomaly_interval")]
    
    if test_mode:
        random.seed(42)
        anomaly_keys = random.sample(anomaly_keys, min(test_samples, len(anomaly_keys)))
        print(f"Test mode: processing {len(anomaly_keys)} anomaly videos")
    
    stats = {"positives": 0, "hard_negatives": 0, "videos_processed": 0, "videos_skipped": 0}
    
    pbar = tqdm(anomaly_keys, desc="XD-Violence")
    for video_key in pbar:
        try:
            n_pos, n_hard = processor.process_anomaly_video(
                video_name=video_key,
                video_info=annotations[video_key],
                collector=collector
            )
            
            if n_pos > 0 or n_hard > 0:
                stats["positives"] += n_pos
                stats["hard_negatives"] += n_hard
                stats["videos_processed"] += 1
            else:
                stats["videos_skipped"] += 1
            
            pbar.set_postfix({"P": stats["positives"], "HN": stats["hard_negatives"]})
            
        except Exception as e:
            stats["videos_skipped"] += 1
            continue
    
    return stats


def process_ucf_crime(
    annotations: dict,
    video_dir: str,
    output_pt_root: str,
    video_subdir: str,
    collector: SampleCollector,
    test_mode: bool = False,
    test_samples: int = 5,
    easy_neg_span_seconds: float = 2.0
) -> dict:
    """Process UCF-Crime dataset."""
    print("\n" + "="*60)
    print("Processing UCF-Crime Dataset")
    print("="*60)
    print(f"Positive: {UCF_CRIME_CONFIG.pos_span_seconds}s span, {UCF_CRIME_CONFIG.pos_stride_seconds}s stride")
    print(f"Hard Negative: {UCF_CRIME_CONFIG.hard_neg_span_seconds}s span, {UCF_CRIME_CONFIG.hard_neg_stride_seconds}s stride")
    print(f"Easy Negative: {easy_neg_span_seconds}s span")
    
    processor = VideoProcessor(
        config=UCF_CRIME_CONFIG,
        output_pt_root=output_pt_root,
        video_dir=video_dir,
        video_subdir=video_subdir,
    )
    
    video_keys = list(annotations.keys())
    anomaly_keys = [k for k in video_keys if annotations[k].get("anomaly_interval")]
    normal_keys = [k for k in video_keys if not annotations[k].get("anomaly_interval")]
    
    if test_mode:
        random.seed(42)
        anomaly_keys = random.sample(anomaly_keys, min(test_samples, len(anomaly_keys)))
        normal_keys = random.sample(normal_keys, min(test_samples, len(normal_keys)))
        print(f"Test mode: processing {len(anomaly_keys)} anomaly + {len(normal_keys)} normal videos")
    
    stats = {
        "positives": 0, 
        "hard_negatives": 0, 
        "easy_negatives": 0,
        "anomaly_videos_processed": 0,
        "normal_videos_processed": 0,
        "videos_skipped": 0
    }
    
    # Process anomaly videos
    print(f"\nProcessing {len(anomaly_keys)} anomaly videos...")
    pbar = tqdm(anomaly_keys, desc="UCF-Crime Anomaly")
    for video_key in pbar:
        try:
            n_pos, n_hard = processor.process_anomaly_video(
                video_name=video_key,
                video_info=annotations[video_key],
                collector=collector
            )
            
            if n_pos > 0 or n_hard > 0:
                stats["positives"] += n_pos
                stats["hard_negatives"] += n_hard
                stats["anomaly_videos_processed"] += 1
            else:
                stats["videos_skipped"] += 1
            
            pbar.set_postfix({"P": stats["positives"], "HN": stats["hard_negatives"]})
            
        except Exception as e:
            stats["videos_skipped"] += 1
            continue
    
    # Process normal videos for easy negatives
    if normal_keys:
        target_easy = stats["positives"]
        samples_per_video = max(1, target_easy // len(normal_keys))
        
        print(f"\nProcessing {len(normal_keys)} normal videos...")
        print(f"Target easy negatives: {target_easy} ({samples_per_video} per video)")
        print(f"Easy negative time span: {easy_neg_span_seconds}s")
        
        pbar = tqdm(normal_keys, desc="UCF-Crime Normal")
        for video_key in pbar:
            try:
                n_easy = processor.process_normal_video(
                    video_name=video_key,
                    video_info=annotations[video_key],
                    collector=collector,
                    target_samples_per_video=samples_per_video,
                    span_seconds=easy_neg_span_seconds
                )
                
                if n_easy > 0:
                    stats["easy_negatives"] += n_easy
                    stats["normal_videos_processed"] += 1
                else:
                    stats["videos_skipped"] += 1
                
                pbar.set_postfix({"EN": stats["easy_negatives"]})
                
            except Exception as e:
                stats["videos_skipped"] += 1
                continue
    
    return stats


def save_dataset_json(
    samples: List[dict], 
    output_path: str, 
    dataset_name: str
) -> None:
    """Save samples to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"  {dataset_name}: {len(samples)} samples -> {output_path}")


# ===================== Main Function =====================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_HIVAU_ROOT = os.environ.get("HIVAU_ROOT", "/path/to/HIVAU-70k")
DEFAULT_VAD_TRAIN_ROOT = os.environ.get(
    "VAD_TRAIN_ROOT",
    os.path.join(REPO_ROOT, "vad_data", "paligemma_train"),
)


def main():
    parser = argparse.ArgumentParser(description="Generate PaliGemma Training Data V4")
    parser.add_argument("--data-dir", type=str,
                        default=DEFAULT_HIVAU_ROOT,
                        help="Path to the HIVAU dataset root containing raw_annotations and videos.")
    parser.add_argument("--output-dir", type=str,
                        default=DEFAULT_VAD_TRAIN_ROOT,
                        help="Directory to save generated training JSON and .pt samples.")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--test-samples", type=int, default=5)
    parser.add_argument("--balance-ratio", type=str, default="1:1:1",
                        help="Balance ratio for pos:hard_neg:easy_neg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-cache-size", type=int, default=200,
                        help="Number of frames to cache per video")
    parser.add_argument("--easy-neg-span", type=float, default=2.0,
                        help="Time span in seconds for easy negatives (2.0-4.0)")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    balance_ratio = tuple(map(int, args.balance_ratio.split(":")))
    
    ucf_anno_path = os.path.join(
        args.data_dir, "raw_annotations/ucf_database_train.json"
    )
    xd_anno_path = os.path.join(
        args.data_dir, "raw_annotations/xd_database_train.json"
    )
    
    output_pt_root = os.path.join(args.output_dir, "train_pt")
    os.makedirs(output_pt_root, exist_ok=True)
    
    # Separate collectors for each dataset
    xd_collector = SampleCollector(dataset_name="XD-Violence")
    ucf_collector = SampleCollector(dataset_name="UCF-Crime")
    
    all_stats = {}
    
    # Process XD-Violence
    if os.path.exists(xd_anno_path):
        print(f"\nLoading XD-Violence annotations from {xd_anno_path}")
        with open(xd_anno_path, 'r') as f:
            xd_annotations = json.load(f)
        
        xd_stats = process_xd_violence(
            annotations=xd_annotations,
            video_dir=args.data_dir,
            output_pt_root=output_pt_root,
            video_subdir="videos/xd-violence/videos/train/",
            collector=xd_collector,
            test_mode=args.test_mode,
            test_samples=args.test_samples
        )
        all_stats["xd_violence"] = xd_stats
    else:
        print(f"Warning: XD-Violence annotation not found: {xd_anno_path}")
    
    # Process UCF-Crime
    if os.path.exists(ucf_anno_path):
        print(f"\nLoading UCF-Crime annotations from {ucf_anno_path}")
        with open(ucf_anno_path, 'r') as f:
            ucf_annotations = json.load(f)
        
        ucf_stats = process_ucf_crime(
            annotations=ucf_annotations,
            video_dir=args.data_dir,
            output_pt_root=output_pt_root,
            video_subdir="videos/ucf-crime/videos/train/",
            collector=ucf_collector,
            test_mode=args.test_mode,
            test_samples=args.test_samples,
            easy_neg_span_seconds=args.easy_neg_span
        )
        all_stats["ucf_crime"] = ucf_stats
    else:
        print(f"Warning: UCF-Crime annotation not found: {ucf_anno_path}")
    
    # ===================== Save Individual Dataset JSONs =====================
    print(f"\n{'='*60}")
    print("Saving Dataset JSONs...")
    print(f"{'='*60}")
    
    # Balance and save XD-Violence
    print(f"\nApplying balance ratio {balance_ratio} to XD-Violence...")
    xd_balanced_samples = xd_collector.get_balanced_samples(balance_ratio)
    xd_json_path = os.path.join(args.output_dir, "xd_violence_train.json")
    save_dataset_json(xd_balanced_samples, xd_json_path, "XD-Violence")
    
    # Balance and save UCF-Crime
    print(f"\nApplying balance ratio {balance_ratio} to UCF-Crime...")
    ucf_balanced_samples = ucf_collector.get_balanced_samples(balance_ratio)
    ucf_json_path = os.path.join(args.output_dir, "ucf_crime_train.json")
    save_dataset_json(ucf_balanced_samples, ucf_json_path, "UCF-Crime")
    
    # ===================== Create Combined Dataset =====================
    print(f"\nCreating combined dataset...")
    
    # Combine all balanced samples
    combined_samples = xd_balanced_samples + ucf_balanced_samples
    random.shuffle(combined_samples)
    
    combined_json_path = os.path.join(args.output_dir, "combined_train.json")
    save_dataset_json(combined_samples, combined_json_path, "Combined")
    
    # ===================== Save Statistics =====================
    stats_path = os.path.join(args.output_dir, "training_stats.json")
    final_stats = {
        "per_dataset": {
            "xd_violence": {
                "processing_stats": all_stats.get("xd_violence", {}),
                "sample_stats": xd_collector.get_stats(),
                "balanced_samples": len(xd_balanced_samples)
            },
            "ucf_crime": {
                "processing_stats": all_stats.get("ucf_crime", {}),
                "sample_stats": ucf_collector.get_stats(),
                "balanced_samples": len(ucf_balanced_samples)
            }
        },
        "combined": {
            "total_samples": len(combined_samples),
            "xd_violence_samples": len(xd_balanced_samples),
            "ucf_crime_samples": len(ucf_balanced_samples),
        },
        "config": {
            "balance_ratio": args.balance_ratio,
            "pos_span_seconds": UCF_CRIME_CONFIG.pos_span_seconds,
            "hard_neg_span_seconds": UCF_CRIME_CONFIG.hard_neg_span_seconds,
            "easy_neg_span_seconds": args.easy_neg_span,
            "pos_stride_seconds": UCF_CRIME_CONFIG.pos_stride_seconds,
            "grid_layout": "2x2 (top-left → top-right → bottom-left → bottom-right)",
            "grid_canvas_size": TOTAL_SIZE,
            "cell_size": CELL_SIZE,
            "grid_gap_px": GRID_GAP,
            "output_format": "fp16 .pt tensor (3, 384, 384), SigLIP-normalized",
            "timestamp_overlay": False
        },
        "output_files": {
            "xd_violence": "xd_violence_train.json",
            "ucf_crime": "ucf_crime_train.json",
            "combined": "combined_train.json"
        }
    }
    with open(stats_path, 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    # ===================== Print Summary =====================
    print(f"\n{'='*60}")
    print("Processing Complete!")
    print(f"{'='*60}")
    
    print(f"\n--- XD-Violence ---")
    if "xd_violence" in all_stats:
        for key, value in all_stats["xd_violence"].items():
            print(f"  {key}: {value}")
    print(f"  Balanced samples: {len(xd_balanced_samples)}")
    
    print(f"\n--- UCF-Crime ---")
    if "ucf_crime" in all_stats:
        for key, value in all_stats["ucf_crime"].items():
            print(f"  {key}: {value}")
    print(f"  Balanced samples: {len(ucf_balanced_samples)}")
    
    print(f"\n--- Combined ---")
    print(f"  Total samples: {len(combined_samples)}")
    
    print(f"\n--- Output Files ---")
    print(f"  XD-Violence JSON: {xd_json_path}")
    print(f"  UCF-Crime JSON:   {ucf_json_path}")
    print(f"  Combined JSON:    {combined_json_path}")
    print(f"  Statistics:       {stats_path}")
    print(f"  PT tensors:       {output_pt_root}/")
    
    # Print recommended prompt
    print(f"\n{'='*60}")
    print("Recommended Training Prompt:")
    print(f"{'='*60}")
    print("""
The tensor shows a 2x2 grid of 4 consecutive frames from a surveillance video.
The frames are arranged in temporal order:
- Top-left (1st) → Top-right (2nd) → Bottom-left (3rd) → Bottom-right (4th)
Analyze these frames and determine if any anomaly is occurring.
""")


if __name__ == "__main__":
    main()