"""
PaliGemma2-3B VAD Training with StreamForest Vision Encoder (384x384)

This script replaces PaliGemma's original SigLIP-448 vision encoder with
StreamForest's SigLIP-384 vision encoder weights, then fine-tunes:
  - Multi-modal projector (via modules_to_save)
  - Language model (Gemma2-3B) via LoRA
  - Vision encoder is FROZEN

Key differences from train_paligemma_vad.py:
  - Image resolution: 384x384 (not 448x448)
  - Image tokens: 729 (27x27) instead of 1024 (32x32)
  - Vision encoder weights from StreamForest (pre-trained on video understanding)
  - Position embedding resized from 1024 to 729
"""

import os
import sys
import json
import types
import torch
import torch.nn.functional as F
import logging
import random
import warnings
import csv
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

import safetensors.torch as safetensors

import transformers
from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from PIL import Image
from torch.utils.data import Dataset

# Import prompt from centralized location
from get_prompt import get_grid_prompt

# Prompt style configuration - change this to switch prompt styles
PROMPT_STYLE = "detail"
GRID_PROMPT = get_grid_prompt(add_special_tokens=False, style=PROMPT_STYLE)

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*use_cache=True.*")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*Gradients will be None.*")


def setup_logging(output_dir: str, log_level: int = logging.INFO):
    """Setup logging to both console and file."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training.log")

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return log_file


logger = logging.getLogger(__name__)

PROJECT_ROOT = os.environ.get(
    "REACTVAU_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CKPT_ROOT = os.environ.get("CKPT_ROOT", os.path.join(PROJECT_ROOT, "ckpt"))
VAD_TRAIN_ROOT = os.environ.get(
    "VAD_TRAIN_ROOT",
    os.path.join(PROJECT_ROOT, "vad", "vad_data", "paligemma_train")
)


# ===================== Metrics History Callback =====================
class MetricsHistoryCallback(TrainerCallback):
    """Callback to save training and evaluation metrics to CSV."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.train_history = []
        self.eval_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        # Training metrics
        if "loss" in logs and "eval_loss" not in logs:
            self.train_history.append({
                "step": state.global_step,
                "epoch": state.epoch,
                "loss": logs.get("loss"),
                "avg_yes_prob": logs.get("avg_yes_prob"),
                "avg_no_prob": logs.get("avg_no_prob"),
                "learning_rate": logs.get("learning_rate"),
                "grad_norm": logs.get("grad_norm"),
            })
            self._save_train_history()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        self.eval_history.append({
            "step": state.global_step,
            "epoch": state.epoch,
            "eval_auc": metrics.get("eval_auc"),
            "eval_accuracy": metrics.get("eval_accuracy"),
        })
        self._save_eval_history()

    def _save_train_history(self):
        csv_path = os.path.join(self.output_dir, "train_history.csv")
        if self.train_history:
            keys = self.train_history[0].keys()
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.train_history)

    def _save_eval_history(self):
        csv_path = os.path.join(self.output_dir, "eval_history.csv")
        if self.eval_history:
            keys = self.eval_history[0].keys()
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.eval_history)


# ===================== Configuration =====================
@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: str = field(
        default=os.environ.get(
            "PALIGEMMA_MODEL_PATH",
            os.path.join(CKPT_ROOT, "paligemma2-3b-mix-448")
        ),
    )
    streamforest_vision_weights_path: str = field(
        default=os.environ.get(
            "PALIGEMMA_SF_WEIGHTS",
            os.path.join(CKPT_ROOT, "extracted_weights", "streamforest_vision_encoder_with_prefix.safetensors")
        ),
        metadata={
            "help": "Path to extracted StreamForest vision encoder weights (.safetensors)"}
    )
    use_lora: bool = field(default=True)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    freeze_vision_encoder: bool = field(default=True)
    train_projector: bool = field(default=True)
    attn_implementation: str = field(default="eager")
    vision_feature_layer: int = field(
        default=-2,
        metadata={"help": "Which vision encoder layer to extract features from. "
                  "-1 = last layer (PaliGemma default), "
                  "-2 = second-to-last (StreamForest native, recommended)"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    data_path: str = field(
        default=os.path.join(VAD_TRAIN_ROOT, "combined_train_binary.json")
    )
    image_root: str = field(
        default=VAD_TRAIN_ROOT
    )
    max_length: int = field(
        default=96,
        metadata={
            "help": "Max TEXT tokens (not including image). Processor adds 729 image tokens automatically."}
    )
    val_split: float = field(
        default=0.05,
        metadata={"help": "Validation split ratio (by video)"}
    )
    label_format: str = field(
        default="binary",
        metadata={
            "help": "Label format: 'binary' (Yes/No only) or 'text' (Yes. Event: Arrest. / No. Event: Normal.)"}
    )


@dataclass
class VADTrainingArguments(TrainingArguments):
    """Extended training arguments for VAD."""
    output_dir: str = field(
        default=os.environ.get(
            "PALIGEMMA_LORA_PATH",
            os.path.join(CKPT_ROOT, "paligemma2-3b-vad-lora-384-combined-binary", "training_384")
        )
    )

    # Weighted loss (only used when label_format='text')
    decision_token_weight: float = field(
        default=5.0,
        metadata={"help": "Weight multiplier for the first suffix token (Yes/No). "
                  "Only used when label_format='text'. "
                  "Higher values focus more gradient signal on the binary decision."}
    )
    event_token_weight: float = field(
        default=2.0,
        metadata={"help": "Weight multiplier for event type tokens (e.g., 'Event: Arrest.'). "
                  "Only used when label_format='text'. "
                  "Tokens after 'Event:' get this weight. Default suffix tokens get 1.0."}
    )

    # Training
    num_train_epochs: float = field(default=1.0)
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=16)

    # Learning rate
    learning_rate: float = field(default=1e-5)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.1)
    lr_scheduler_type: str = field(default="cosine")

    # Logging and saving
    logging_steps: int = field(default=20)
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=3)
    eval_strategy: str = field(default="steps")
    eval_steps: int = field(default=250)

    # Precision
    bf16: bool = field(default=True)
    tf32: bool = field(default=True)

    # Data loading
    dataloader_num_workers: int = field(default=4)
    dataloader_pin_memory: bool = field(default=True)
    dataloader_prefetch_factor: int = field(default=2)
    remove_unused_columns: bool = field(default=False)

    # Memory optimization
    gradient_checkpointing: bool = field(default=False)
    optim: str = field(default="adamw_torch_fused")

    # Misc
    report_to: str = field(default="tensorboard")
    seed: int = field(default=42)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_auc")
    greater_is_better: bool = field(default=True)


# ===================== Video-Level Data Split =====================
def split_data_by_video(all_data: List[Dict], val_ratio: float = 0.05, seed: int = 42) -> tuple:
    """
    Split data by video to prevent data leakage.
    All frames from the same video go to either train or val, not both.
    """
    video_to_samples = defaultdict(list)

    for item in all_data:
        image_path = item["image"]
        parts = image_path.split("/")
        video_name = parts[1] if len(parts) >= 2 else image_path
        video_to_samples[video_name].append(item)

    # Separate anomaly and normal videos for stratified split
    anomaly_videos = []
    normal_videos = []

    for video_name, samples in video_to_samples.items():
        # Check if suffix starts with "Yes" (handles both "Yes" and "Yes. Event: ...")
        has_anomaly = any(s["suffix"].startswith("Yes") for s in samples)
        if has_anomaly:
            anomaly_videos.append(video_name)
        else:
            normal_videos.append(video_name)

    logger.info(f"Video statistics:")
    logger.info(f"  Total videos: {len(video_to_samples)}")
    logger.info(f"  Anomaly videos: {len(anomaly_videos)}")
    logger.info(f"  Normal videos: {len(normal_videos)}")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(anomaly_videos)
    random.shuffle(normal_videos)

    val_anomaly_count = max(5, int(len(anomaly_videos) * val_ratio))
    val_normal_count = max(5, int(len(normal_videos) * val_ratio))

    val_videos = set(
        anomaly_videos[:val_anomaly_count] + normal_videos[:val_normal_count])

    train_data, val_data = [], []
    for video_name, samples in video_to_samples.items():
        if video_name in val_videos:
            val_data.extend(samples)
        else:
            train_data.extend(samples)

    random.shuffle(train_data)
    random.shuffle(val_data)

    # Statistics - check if suffix starts with "Yes"
    train_pos = sum(1 for d in train_data if d["suffix"].startswith("Yes"))
    val_pos = sum(1 for d in val_data if d["suffix"].startswith("Yes"))

    split_info = {
        "train_videos": len(video_to_samples) - len(val_videos),
        "val_videos": len(val_videos),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "train_pos": train_pos,
        "train_neg": len(train_data) - train_pos,
        "val_pos": val_pos,
        "val_neg": len(val_data) - val_pos,
    }

    logger.info(f"Data split (by video):")
    logger.info(
        f"  Train: {split_info['train_videos']} videos, {len(train_data)} samples (pos: {train_pos})")
    logger.info(
        f"  Val: {len(val_videos)} videos, {len(val_data)} samples (pos: {val_pos})")

    return train_data, val_data, split_info


# ===================== Dataset =====================
class VADDataset(Dataset):
    """Video Anomaly Detection Dataset for PaliGemma2 with 384x384 input.

    Data format expected:
    {
        "image": "train_images/Video_x264/Video_x264_00019.jpg",
        "suffix": "Yes. Event: Arrest."  or  "No. Event: Normal."
    }

    Uses PaliGemma's suffix parameter for correct attention masking:
    - Prefix (image + prompt): bidirectional attention (token_type_ids=0)
    - Suffix (answer): causal attention (token_type_ids=1)
    - Labels: -100 for prefix (no loss), actual tokens for suffix (compute loss)
    """

    def __init__(
        self,
        data: List[Dict],
        processor: AutoProcessor,
        image_root: str,
        max_length: int = 96,
        split_name: str = "train",
        label_format: str = "binary",
        decision_token_weight: float = 5.0,
        event_token_weight: float = 2.0,
    ):
        self.processor = processor
        self.image_root = image_root
        self.max_length = max_length
        self.prompt_raw = GRID_PROMPT
        self.split_name = split_name
        self.label_format = label_format
        self.decision_token_weight = decision_token_weight
        self.event_token_weight = event_token_weight

        # Get Yes/No token IDs for monitoring
        self.yes_token_id = processor.tokenizer.encode(
            "Yes", add_special_tokens=False)[0]
        self.no_token_id = processor.tokenizer.encode(
            "No", add_special_tokens=False)[0]

        # Filter valid samples
        self.valid_data = []
        skipped = 0
        for item in data:
            image_path = os.path.join(self.image_root, item["image"])
            if os.path.exists(image_path):
                self.valid_data.append(item)
            else:
                skipped += 1

        if skipped > 0:
            logger.warning(
                f"[{split_name}] Skipped {skipped} samples with missing images")

        # Count label distribution - check if suffix starts with "Yes"
        self.pos_count = sum(
            1 for d in self.valid_data if d["suffix"].startswith("Yes"))
        self.neg_count = len(self.valid_data) - self.pos_count

        logger.info(
            f"[{split_name}] Dataset: {len(self.valid_data)} samples (pos: {self.pos_count}, neg: {self.neg_count})")
        if label_format == "text":
            logger.info(
                f"[{split_name}] Label format: text (weighted loss: decision={decision_token_weight}, event={event_token_weight})")

        if split_name == "train" and len(self.valid_data) > 0:
            self._debug_tokenization()

    def _debug_tokenization(self):
        """Debug tokenization for first sample to verify suffix parameter usage."""
        sample = self.valid_data[0]
        suffix = sample["suffix"]

        # Test the processor with suffix parameter (must use list format!)
        image_path = os.path.join(self.image_root, sample["image"])
        try:
            test_image = Image.open(image_path).convert("RGB")
        except:
            # 384x384 fallback for StreamForest vision encoder
            test_image = Image.new("RGB", (384, 384), (128, 128, 128))

        prompt_text = f"<image>{self.prompt_raw}"
        suffix_with_eos = suffix + self.processor.tokenizer.eos_token
        test_inputs = self.processor(
            text=[prompt_text],
            images=[test_image],
            suffix=[suffix_with_eos],
            return_tensors="pt",
            padding=False,
        )

        input_ids = test_inputs["input_ids"][0]
        token_type_ids = test_inputs["token_type_ids"][0]
        labels = test_inputs["labels"][0]

        # Find where suffix starts (where token_type_ids changes from 0 to 1)
        suffix_mask = (token_type_ids == 1)
        prefix_mask = (token_type_ids == 0)

        logger.info(
            f"[DEBUG] Using processor suffix parameter (correct PaliGemma fine-tuning)")
        logger.info(
            f"[DEBUG] Vision encoder: StreamForest SigLIP-384 (384x384, 729 tokens)")
        logger.info(f"[DEBUG] Suffix: '{suffix}'")
        logger.info(f"[DEBUG] Input length (no padding): {len(input_ids)}")
        logger.info(
            f"[DEBUG] Prefix tokens (token_type_ids=0): {prefix_mask.sum().item()}")
        logger.info(
            f"[DEBUG] Suffix tokens (token_type_ids=1): {suffix_mask.sum().item()}")
        logger.info(
            f"[DEBUG] Labels with -100 (no loss): {(labels == -100).sum().item()}")
        logger.info(
            f"[DEBUG] Labels with actual values (compute loss): {(labels != -100).sum().item()}")
        logger.info(
            f"[DEBUG] Yes token ID: {self.yes_token_id}, No token ID: {self.no_token_id}")

        # Verify that labels are -100 for prefix and actual tokens for suffix
        prefix_labels = labels[prefix_mask]
        suffix_labels = labels[suffix_mask]
        logger.info(
            f"[DEBUG] Prefix labels all -100: {(prefix_labels == -100).all().item()}")
        logger.info(
            f"[DEBUG] Suffix labels (decoded): {self.processor.tokenizer.decode(suffix_labels[suffix_labels != -100].tolist())}")

        # Show last few tokens to verify structure
        logger.info(f"[DEBUG] Last 10 tokens:")
        for i in range(max(0, len(input_ids)-10), len(input_ids)):
            tid = input_ids[i].item()
            tti = token_type_ids[i].item()
            lbl = labels[i].item()
            decoded = self.processor.tokenizer.decode([tid])
            logger.info(
                f"[DEBUG]   pos {i}: id={tid:6d}, tti={tti}, label={lbl:6d} -> '{decoded}'")

    @staticmethod
    def _find_subsequence(sequence: List[int], subsequence: List[int]) -> int:
        """Find the start index of subsequence in sequence. Returns -1 if not found."""
        seq_len = len(sequence)
        sub_len = len(subsequence)
        for i in range(seq_len - sub_len + 1):
            if sequence[i:i + sub_len] == subsequence:
                return i
        return -1

    def _compute_suffix_weights(self, suffix: str, labels: torch.Tensor) -> torch.Tensor:
        """Compute per-token weights for weighted loss in text label format.

        Suffix format: "Yes. Event: Arrest." or "No. Event: Normal."
        Multi-event:   "Yes. Event: Fighting, Arrest."

        Weight assignment:
          - Yes/No token:           decision_token_weight (5.0)
          - Event type tokens only: event_token_weight (2.0)
            (e.g., "Arrest", "Fighting, Arrest" — the actual event names)
          - Fixed format tokens:    1.0
            (".", "Event", ":", ".", EOS — model learns these quickly)
          - Prefix tokens:          0.0 (labels=-100, no loss)
        """
        weights = torch.ones_like(labels, dtype=torch.float32)
        weights[labels == -100] = 0.0  # No weight for prefix

        valid_positions = (labels != -100).nonzero(as_tuple=True)[0]
        if len(valid_positions) == 0:
            return weights

        # First valid token = Yes/No decision → decision weight
        weights[valid_positions[0]] = self.decision_token_weight

        # Find event type tokens via subsequence matching
        if ". Event: " not in suffix:
            return weights

        # Extract event type text: "Yes. Event: Arrest." → "Arrest"
        # Handles multi-event: "Yes. Event: Fighting, Arrest." → "Fighting, Arrest"
        event_type_text = suffix.split(". Event: ")[1].rstrip(".")

        # Tokenize event type independently
        event_type_ids = self.processor.tokenizer.encode(
            event_type_text, add_special_tokens=False)

        # Get suffix token IDs (the non -100 portion of labels)
        suffix_ids = labels[valid_positions].tolist()

        # Find event type subsequence in suffix tokens
        start_idx = self._find_subsequence(suffix_ids, event_type_ids)
        if start_idx >= 0:
            for j in range(start_idx, start_idx + len(event_type_ids)):
                if j < len(valid_positions):
                    weights[valid_positions[j]] = self.event_token_weight

        return weights

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        item = self.valid_data[idx]

        # Load image
        image_path = os.path.join(self.image_root, item["image"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load {image_path}: {e}")
            # 384x384 fallback for StreamForest vision encoder
            image = Image.new("RGB", (384, 384), (128, 128, 128))

        suffix = item["suffix"]

        prompt_text = f"<image>{self.prompt_raw}"
        suffix_with_eos = suffix + self.processor.tokenizer.eos_token

        inputs = self.processor(
            text=[prompt_text],
            images=[image],
            suffix=[suffix_with_eos],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)
        token_type_ids = inputs["token_type_ids"].squeeze(0)
        labels = inputs["labels"].squeeze(0)

        # Ground truth label for AUC calculation
        is_anomaly = 1 if suffix.startswith("Yes") else 0

        # Compute per-token weights for text label format
        if self.label_format == "text":
            token_weights = self._compute_suffix_weights(suffix, labels)
        else:
            # Binary format: uniform weights (not actually used in loss, but kept for consistency)
            token_weights = torch.ones_like(labels, dtype=torch.float32)
            token_weights[labels == -100] = 0.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "token_type_ids": token_type_ids,
            "labels": labels,
            "is_anomaly": torch.tensor(is_anomaly, dtype=torch.long),
            "token_weights": token_weights,
        }


class VADDataCollator:
    """Simple data collator."""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return {
            key: torch.stack([f[key] for f in features])
            for key in features[0].keys()
        }


# ===================== Trainer with Standard CE Loss =====================
class VADTrainer(Trainer):
    """Trainer using standard language model cross-entropy loss.

    Uses PaliGemma's built-in loss calculation with proper attention masking
    via token_type_ids. Loss is computed only on suffix tokens (Yes/No + Event type).

    Supports two label formats:
    - 'binary': suffix = "Yes" / "No" → standard CE loss
    - 'text':   suffix = "Yes. Event: Arrest." / "No. Event: Normal."
                → weighted CE loss: Yes/No token gets decision_token_weight,
                  remaining event tokens get event_token_weight
    """

    def __init__(self, *args, yes_token_id: int = None, no_token_id: int = None,
                 label_format: str = "binary", decision_token_weight: float = 5.0,
                 event_token_weight: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id
        self.label_format = label_format
        self.decision_token_weight = decision_token_weight
        self.event_token_weight = event_token_weight
        self._logged_yes_prob = None
        self._logged_no_prob = None

        # Accumulators for computing metrics over full effective batch
        self._accumulated_yes_probs = []
        self._accumulated_no_probs = []
        self._accumulation_count = 0

        if self.label_format == "text":
            logger.info(f"Label format: text (weighted loss)")
            logger.info(
                f"  Decision token (Yes/No) weight: {self.decision_token_weight}")
            logger.info(
                f"  Event type token weight: {self.event_token_weight}")
        else:
            logger.info(f"Label format: binary (standard CE loss)")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute language model cross-entropy loss.

        For label_format='text': applies weighted loss where the first suffix
        token (Yes/No) gets decision_token_weight and remaining event tokens
        get event_token_weight.

        For label_format='binary': uses standard PaliGemma CE loss.
        """
        # Remove auxiliary inputs
        is_anomaly = inputs.pop("is_anomaly", None)
        token_weights = inputs.pop("token_weights", None)

        # Ensure pixel_values dtype matches model (processor outputs float32, model is bfloat16)
        pixel_values = inputs["pixel_values"]
        try:
            model_dtype = next(model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.bfloat16
        if pixel_values.dtype != model_dtype:
            pixel_values = pixel_values.to(model_dtype)

        # Forward pass with token_type_ids for proper attention masking
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=pixel_values,
            token_type_ids=inputs.get("token_type_ids"),
            labels=inputs["labels"],
        )

        logits = outputs.logits
        labels = inputs["labels"]
        batch_size = logits.size(0)

        # ---- Weighted loss for text format ----
        if self.label_format == "text" and token_weights is not None:
            # Use precomputed token_weights from dataset
            # Shift logits/labels/weights for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()    # [B, T-1, V]
            shift_labels = labels[..., 1:].contiguous()         # [B, T-1]
            shift_weights = token_weights[..., 1:].contiguous()  # [B, T-1]

            # Per-token CE loss (no reduction)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(batch_size, -1)  # [B, T-1]

            # Apply precomputed weights:
            #   0.0 for prefix tokens (no loss)
            #   5.0 for Yes/No decision token
            #   2.0 for event type tokens (e.g., "Arrest", "Fighting")
            #   1.0 for fixed format tokens (".", "Event", ":", EOS)
            total_loss = (per_token_loss * shift_weights).sum() / \
                shift_weights.sum()
        else:
            total_loss = outputs.loss

        # Track Yes/No probabilities for monitoring only (no gradient)
        yes_probs_list = []
        no_probs_list = []

        with torch.no_grad():
            for i in range(batch_size):
                valid_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]

                if len(valid_positions) > 0:
                    pos = valid_positions[0].item()

                    if pos > 0:
                        pred_logits = logits[i, pos - 1]
                        yes_logit = pred_logits[self.yes_token_id].float()
                        no_logit = pred_logits[self.no_token_id].float()
                        probs = F.softmax(torch.stack(
                            [yes_logit, no_logit]), dim=0)
                        yes_probs_list.append(probs[0].item())
                        no_probs_list.append(probs[1].item())

        # Accumulate across gradient accumulation steps
        self._accumulated_yes_probs.extend(yes_probs_list)
        self._accumulated_no_probs.extend(no_probs_list)
        self._accumulation_count += 1

        # Restore auxiliary inputs
        if is_anomaly is not None:
            inputs["is_anomaly"] = is_anomaly

        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs: Dict[str, float]) -> None:
        if "loss" in logs:
            if self._accumulated_yes_probs:
                self._logged_yes_prob = sum(
                    self._accumulated_yes_probs) / len(self._accumulated_yes_probs)
            else:
                self._logged_yes_prob = 0.0

            if self._accumulated_no_probs:
                self._logged_no_prob = sum(
                    self._accumulated_no_probs) / len(self._accumulated_no_probs)
            else:
                self._logged_no_prob = 0.0

            logs["n_samples_in_step"] = len(self._accumulated_yes_probs)

            if self._logged_yes_prob is not None:
                logs["avg_yes_prob"] = self._logged_yes_prob
            if self._logged_no_prob is not None:
                logs["avg_no_prob"] = self._logged_no_prob

            # Reset accumulators
            self._accumulated_yes_probs = []
            self._accumulated_no_probs = []
            self._accumulation_count = 0

            # Explicitly log to file for real-time monitoring
            try:
                log_items = []
                for k, v in logs.items():
                    if isinstance(v, float):
                        log_items.append(f"{k}: {v:.6f}")
                    else:
                        log_items.append(f"{k}: {v}")

                log_str = ", ".join(log_items)
                logger.info(
                    f"STEP {self.state.global_step} LOG: {{{log_str}}}")

                for handler in logger.handlers:
                    handler.flush()
                if logging.getLogger().handlers:
                    for handler in logging.getLogger().handlers:
                        handler.flush()
            except Exception as e:
                pass

        super().log(logs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to compute anomaly scores for AUC.

        Uses only PREFIX tokens (without suffix) for evaluation to match
        the actual inference scenario.
        """
        is_anomaly = inputs.get("is_anomaly")

        labels_tensor = inputs["labels"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        token_type_ids = inputs.get("token_type_ids")

        # Ensure pixel_values dtype matches model (processor outputs float32, model is bfloat16)
        try:
            model_dtype = next(model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.bfloat16
        if pixel_values.dtype != model_dtype:
            pixel_values = pixel_values.to(model_dtype)

        batch_size = input_ids.size(0)

        with torch.no_grad():
            anomaly_scores = []
            predictions = []

            for i in range(batch_size):
                sample_labels = labels_tensor[i]
                sample_input_ids = input_ids[i]
                sample_attention_mask = attention_mask[i]
                sample_pixel_values = pixel_values[i]

                # Find prefix length (where suffix starts)
                valid_label_positions = (
                    sample_labels != -100).nonzero(as_tuple=True)[0]

                if len(valid_label_positions) > 0:
                    suffix_start = valid_label_positions[0].item()

                    # Truncate to prefix only (no suffix tokens!)
                    prefix_input_ids = sample_input_ids[:suffix_start].unsqueeze(
                        0)
                    prefix_attention_mask = sample_attention_mask[:suffix_start].unsqueeze(
                        0)

                    # Forward pass with PREFIX ONLY
                    outputs = model(
                        input_ids=prefix_input_ids,
                        attention_mask=prefix_attention_mask,
                        pixel_values=sample_pixel_values.unsqueeze(0),
                    )

                    logits = outputs.logits

                    # Get logits at the LAST position of prefix
                    pred_logits = logits[0, -1]

                    if self.yes_token_id is not None and self.no_token_id is not None:
                        yes_logit = pred_logits[self.yes_token_id].float()
                        no_logit = pred_logits[self.no_token_id].float()

                        probs = F.softmax(torch.stack(
                            [yes_logit, no_logit]), dim=0)
                        anomaly_score = probs[0].item()
                        pred = 1 if yes_logit > no_logit else 0
                    else:
                        anomaly_score = 0.5
                        pred = 0
                else:
                    anomaly_score = 0.5
                    pred = 0

                anomaly_scores.append(anomaly_score)
                predictions.append(pred)

            anomaly_scores = torch.tensor(
                anomaly_scores, device=input_ids.device, dtype=torch.float32)
            predictions = torch.tensor(
                predictions, device=input_ids.device, dtype=torch.float32)

        if prediction_loss_only:
            return (None, None, None)

        packed_logits = torch.stack([anomaly_scores, predictions], dim=1)
        return (None, packed_logits, is_anomaly)


def compute_metrics(eval_pred):
    """Compute AUC and accuracy from predictions."""
    logits, labels = eval_pred

    anomaly_scores = logits[:, 0]
    predictions = logits[:, 1].astype(int)

    try:
        auc = roc_auc_score(labels, anomaly_scores)
    except ValueError:
        auc = 0.5

    accuracy = (predictions == labels).mean()

    return {
        "auc": auc,
        "accuracy": accuracy,
    }


# ===================== StreamForest Vision Encoder Loading =====================
def load_streamforest_vision_weights(model, weights_path: str):
    """Replace PaliGemma's vision encoder with StreamForest's SigLIP-384 weights.

    Steps:
    1. Load StreamForest weights from safetensors
    2. Resize position embedding from 1024 (448px) to 729 (384px)
    3. Load weights with strict=False (layer 26 kept from PaliGemma original)
    4. Update model config for 384x384 input

    Args:
        model: PaliGemmaForConditionalGeneration model (on GPU)
        weights_path: Path to streamforest_vision_encoder_with_prefix.safetensors

    Returns:
        model with StreamForest vision weights loaded
    """
    logger.info(f"Loading StreamForest vision weights from: {weights_path}")

    # Load weights
    stream_weights = safetensors.load_file(weights_path)
    logger.info(f"  Loaded {len(stream_weights)} tensors from StreamForest")

    # Filter to vision_model keys only
    remapped = {}
    for key, value in stream_weights.items():
        if key.startswith("vision_model."):
            remapped[key] = value

    # Step 1: Resize position embedding BEFORE loading weights
    pos_key = "vision_model.embeddings.position_embedding.weight"
    if pos_key in remapped:
        new_pos_shape = remapped[pos_key].shape
        old_pos_shape = model.vision_tower.vision_model.embeddings.position_embedding.weight.shape
        logger.info(
            f"  Position embedding: {old_pos_shape} -> {new_pos_shape}")

        new_num_positions = new_pos_shape[0]  # 729
        embed_dim = new_pos_shape[1]          # 1152

        # Replace the embedding module
        model.vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(
            new_num_positions, embed_dim
        ).to(model.device).to(model.dtype)
        logger.info(
            f"  ✓ Resized position embedding to {new_num_positions} positions")

    # Step 2: Load StreamForest weights
    missing, unexpected = model.vision_tower.load_state_dict(
        remapped, strict=False)
    logger.info(f"  Missing keys (kept from PaliGemma): {len(missing)}")
    logger.info(f"  Unexpected keys: {len(unexpected)}")

    if missing:
        logger.info(
            f"  Sample missing keys (layer 26, kept PaliGemma original):")
        for k in missing[:5]:
            logger.info(f"    - {k}")

    # Step 3: Update position-related state in embeddings
    model.vision_tower.vision_model.embeddings.num_positions = 729
    # Use register_buffer so position_ids moves with the model on .to() / DataParallel
    embeddings_module = model.vision_tower.vision_model.embeddings
    embeddings_module.register_buffer(
        "position_ids",
        torch.arange(729, device=model.device).expand((1, -1)),
        persistent=False,
    )

    # Step 4: Update model config
    model.config.vision_config.image_size = 384
    model.config.vision_config.num_image_tokens = 729
    model.config.vision_config.num_positions = 729
    model.config.text_config.num_image_tokens = 729

    logger.info(f"  ✓ Config updated: image_size=384, num_image_tokens=729")

    return model


# ===================== Vision Feature Layer Patch =====================
def patch_vision_feature_layer(model, select_layer: int = -2):
    """Monkey-patch get_image_features to extract from a specific encoder layer.

    StreamForest uses select_layer=-2 (layer 25), but PaliGemma defaults to
    the last layer (layer 26). Since layer 26 is kept from PaliGemma's original
    weights (not StreamForest-trained), using layer -2 preserves the features
    StreamForest was optimized for.

    Args:
        model: PaliGemmaForConditionalGeneration or PEFT-wrapped model
        select_layer: -1 = last layer, -2 = second-to-last (recommended)
    """
    base_model = model
    if hasattr(model, 'base_model'):
        base_model = model.base_model.model

    def get_image_features_with_layer_select(self, pixel_values):
        image_outputs = self.vision_tower(
            pixel_values, output_hidden_states=True)
        selected_image_feature = image_outputs.hidden_states[select_layer]
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = image_features / (self.config.hidden_size ** 0.5)
        return image_features

    base_model.get_image_features = types.MethodType(
        get_image_features_with_layer_select, base_model
    )
    logger.info(f"  ✓ Patched get_image_features to use layer {select_layer}")
    return model


# ===================== Model Setup =====================
def setup_model_and_processor(model_args: ModelArguments):
    """Setup PaliGemma2 model with StreamForest vision encoder and LoRA."""

    logger.info(
        f"Loading PaliGemma model from {model_args.model_name_or_path}")

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_args.attn_implementation,
        low_cpu_mem_usage=True,
    )

    model = model.cuda()
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"GPU memory after model load: {allocated:.2f} GB")

    # ===== Load StreamForest Vision Encoder =====
    if model_args.streamforest_vision_weights_path:
        model = load_streamforest_vision_weights(
            model, model_args.streamforest_vision_weights_path
        )

        # Update processor for 384x384
        processor.image_processor.size = {"height": 384, "width": 384}
        processor.image_processor.image_seq_length = 729
        processor.image_seq_length = 729

        logger.info(
            f"  ✓ Processor updated: image_size=384x384, image_seq_length=729")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(
                f"GPU memory after StreamForest weight loading: {allocated:.2f} GB")
    else:
        logger.warning(
            "No StreamForest vision weights path provided! Using original PaliGemma vision encoder.")

    # Setup LoRA - only on language model layers
    if model_args.use_lora:
        logger.info(
            f"Setting up LoRA (r={model_args.lora_r}, alpha={model_args.lora_alpha})...")

        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=r"language_model\.model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))",
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=["multi_modal_projector"],
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        _adapter_name = model.active_adapter
        type(model).active_adapters = property(
            lambda self: [self.active_adapter])

    # ===== Patch Vision Feature Layer Selection =====
    # Applied after PEFT wrapping so it patches the correct base model
    if model_args.streamforest_vision_weights_path and model_args.vision_feature_layer != -1:
        logger.info(
            f"Patching vision feature extraction to use layer {model_args.vision_feature_layer}...")
        patch_vision_feature_layer(
            model, select_layer=model_args.vision_feature_layer)
    elif model_args.streamforest_vision_weights_path:
        logger.info("Using default last layer (-1) for vision features.")
        logger.info(
            "  Note: StreamForest uses layer -2. Consider --vision_feature_layer -2")

    # Freeze vision encoder
    logger.info("Freezing vision encoder (StreamForest SigLIP-384)...")
    for name, param in model.named_parameters():
        if "vision_tower" in name:
            param.requires_grad = False

    # Log trainable parameters
    trainable_count = sum(p.numel()
                          for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable parameters: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.2f}%)")

    # Verify no vision_tower parameters are trainable
    vision_trainable = sum(p.numel() for n, p in model.named_parameters(
    ) if "vision_tower" in n and p.requires_grad)
    logger.info(
        f"Vision tower trainable parameters: {vision_trainable} (should be 0)")

    # Enable gradient checkpointing
    logger.info("Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"GPU memory after setup: {allocated:.2f} GB")

    return model, processor


# ===================== Training =====================
def train():
    """Main training function."""

    parser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, VADTrainingArguments
    ))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            sys.argv[1])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Create run directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(training_args.output_dir, f"training_{timestamp}")
    training_args.output_dir = run_dir
    os.makedirs(run_dir, exist_ok=True)

    # Setup logging
    log_file = setup_logging(run_dir)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Logging to: {log_file}")

    # Log configuration
    logger.info("=" * 60)
    logger.info("Training Configuration (384x384 StreamForest Vision)")
    logger.info("=" * 60)
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(
        f"StreamForest Vision: {model_args.streamforest_vision_weights_path}")
    logger.info(f"Data: {data_args.data_path}")
    logger.info(f"Output: {run_dir}")
    logger.info(f"Prompt Style: '{PROMPT_STYLE}'")
    logger.info(f"Prompt: '{GRID_PROMPT}'")
    logger.info(f"Max length: {data_args.max_length}")
    logger.info(f"Image: 384x384, 729 tokens (StreamForest SigLIP)")
    logger.info(f"Vision feature layer: {model_args.vision_feature_layer}")
    logger.info(f"Batch size: {training_args.per_device_train_batch_size} x {training_args.gradient_accumulation_steps} = {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {training_args.learning_rate}")
    logger.info(f"Epochs: {training_args.num_train_epochs}")
    logger.info(f"Eval steps: {training_args.eval_steps}")
    logger.info(f"Save steps: {training_args.save_steps}")
    logger.info(f"Label format: {data_args.label_format}")
    if data_args.label_format == "text":
        logger.info(
            f"Decision token weight: {training_args.decision_token_weight}")
        logger.info(f"Event token weight: {training_args.event_token_weight}")
    logger.info("=" * 60)

    # Setup model
    model, processor = setup_model_and_processor(model_args)

    # Get Yes/No token IDs
    yes_token_id = processor.tokenizer.encode(
        "Yes", add_special_tokens=False)[0]
    no_token_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]
    logger.info(f"Token IDs - Yes: {yes_token_id}, No: {no_token_id}")

    # Load and split data
    logger.info(f"Loading data from {data_args.data_path}")
    with open(data_args.data_path, "r") as f:
        all_data = json.load(f)

    logger.info(f"Total samples: {len(all_data)}")

    train_data, val_data, split_info = split_data_by_video(
        all_data,
        val_ratio=data_args.val_split,
        seed=training_args.seed
    )

    # Save config
    config = {
        "model_args": {k: str(v) for k, v in vars(model_args).items()},
        "data_args": {k: str(v) for k, v in vars(data_args).items()},
        "training_args": {
            "learning_rate": training_args.learning_rate,
            "num_train_epochs": training_args.num_train_epochs,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "eval_steps": training_args.eval_steps,
            "save_steps": training_args.save_steps,
            "decision_token_weight": training_args.decision_token_weight if data_args.label_format == "text" else None,
            "event_token_weight": training_args.event_token_weight if data_args.label_format == "text" else None,
        },
        "label_format": data_args.label_format,
        "prompt_style": PROMPT_STYLE,
        "prompt": GRID_PROMPT,
        "vision_encoder": "StreamForest SigLIP-384 (frozen)",
        "vision_feature_layer": model_args.vision_feature_layer,
        "image_size": 384,
        "image_tokens": 729,
        "split_info": split_info,
        "token_ids": {"yes": yes_token_id, "no": no_token_id},
    }
    with open(os.path.join(run_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Create datasets
    train_dataset = VADDataset(
        train_data, processor, data_args.image_root, data_args.max_length, "train",
        label_format=data_args.label_format,
        decision_token_weight=training_args.decision_token_weight,
        event_token_weight=training_args.event_token_weight,
    )
    val_dataset = VADDataset(
        val_data, processor, data_args.image_root, data_args.max_length, "val",
        label_format=data_args.label_format,
        decision_token_weight=training_args.decision_token_weight,
        event_token_weight=training_args.event_token_weight,
    )

    # Disable gradient_checkpointing in training_args
    training_args.gradient_checkpointing = False

    # Create metrics callback
    metrics_callback = MetricsHistoryCallback(run_dir)

    # Create trainer
    trainer = VADTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=VADDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[metrics_callback],
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        label_format=data_args.label_format,
        decision_token_weight=training_args.decision_token_weight,
        event_token_weight=training_args.event_token_weight,
    )

    # Check for checkpoint
    last_checkpoint = None

    # Train
    logger.info("=" * 60)
    logger.info("Starting training (384x384 StreamForest Vision)...")
    logger.info("=" * 60)

    training_completed = False
    try:
        trainer.train(resume_from_checkpoint=last_checkpoint)
        training_completed = True
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # Save final metrics summary
    try:
        final_metrics = {
            "train_history": metrics_callback.train_history,
            "eval_history": metrics_callback.eval_history,
        }
        with open(os.path.join(run_dir, "metrics_summary.json"), "w") as f:
            json.dump(final_metrics, f, indent=2)
        logger.info(f"Metrics saved to: {run_dir}/metrics_summary.json")
    except Exception as e:
        logger.error(f"Failed to save metrics summary: {e}")

    # Save final model
    model_saved = False
    try:
        logger.info(f"Saving final model to {run_dir}")
        model.save_pretrained(run_dir)
        processor.save_pretrained(run_dir)
        model_saved = True
        logger.info("Model saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save model directly: {e}")
        try:
            import shutil
            checkpoints = [d for d in os.listdir(
                run_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                last_ckpt = checkpoints[-1]
                last_ckpt_path = os.path.join(run_dir, last_ckpt)
                logger.info(
                    f"Attempting to copy from last checkpoint: {last_ckpt_path}")

                for filename in ["adapter_model.safetensors", "adapter_config.json"]:
                    src = os.path.join(last_ckpt_path, filename)
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(run_dir, filename))
                        logger.info(f"Copied {filename} from checkpoint")

                processor.save_pretrained(run_dir)
                model_saved = True
                logger.info("Model recovered from last checkpoint.")
        except Exception as e2:
            logger.error(f"Failed to recover from checkpoint: {e2}")

    logger.info("=" * 60)
    if training_completed and model_saved:
        logger.info("Training complete!")
    elif training_completed:
        logger.info(
            "Training complete but model save failed. Check checkpoints.")
    else:
        logger.info("Training interrupted. Check logs and checkpoints.")
    logger.info(f"Run directory: {run_dir}")
    logger.info(
        f"Metrics saved to: {run_dir}/train_history.csv, eval_history.csv")
    logger.info("=" * 60)

    if not training_completed:
        raise RuntimeError("Training did not complete successfully")


if __name__ == "__main__":
    train()
