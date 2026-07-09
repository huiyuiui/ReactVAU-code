#!/bin/bash
# PaliGemma2-3B VAD binary fine-tuning on a single RTX 4090.
# This is the local non-SLURM version of
# scripts/sbatch/train_paligemma_vad_384_binary_sbatch.sh.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${REACTVAU_CONFIG:-${SCRIPT_DIR}/../../config/paths.sh}"
# shellcheck source=/dev/null
source "${CONFIG_PATH}"

cd "${REACTVAU_ROOT}"
export PYTHONPATH="${REACTVAU_ROOT}:${PYTHONPATH:-}"

reactvau_activate_conda

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-1}"
export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-121}"
export BITSANDBYTES_NOWELCOME="${BITSANDBYTES_NOWELCOME:-1}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Dataset is intentionally fixed to the combined binary training set.
DATASET="combined"
MODEL_PATH="${MODEL_PATH:-${PALIGEMMA_MODEL_PATH}}"
STREAMFOREST_WEIGHTS="${STREAMFOREST_WEIGHTS:-${PALIGEMMA_SF_WEIGHTS}}"
IMAGE_ROOT="${IMAGE_ROOT:-${VAD_TRAIN_ROOT}}"
DATA_PATH="${DATA_PATH:-${VAD_TRAIN_ROOT}/combined_train_binary.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${PALIGEMMA_LORA_PATH}}"

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

NUM_EPOCHS="${NUM_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
MAX_LENGTH="${MAX_LENGTH:-96}"
VAL_SPLIT="${VAL_SPLIT:-0.05}"
VISION_FEATURE_LAYER="${VISION_FEATURE_LAYER:--2}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"

echo "=========================================="
echo "PaliGemma2-3B VAD Fine-tuning (4090, 384x384)"
echo "=========================================="
echo "Project: ${REACTVAU_ROOT}"
echo "Dataset: ${DATASET}"
echo "Data path: ${DATA_PATH}"
echo "Image root: ${IMAGE_ROOT}"
echo "Model: ${MODEL_PATH}"
echo "StreamForest vision weights: ${STREAMFOREST_WEIGHTS}"
echo "Vision feature layer: ${VISION_FEATURE_LAYER}"
echo "Output: ${OUTPUT_DIR}"
echo "Image size: 384x384 (729 tokens)"
echo "Batch: ${BATCH_SIZE} x ${GRAD_ACCUM} = $((BATCH_SIZE * GRAD_ACCUM)) effective"
echo "Learning rate: ${LEARNING_RATE}"
echo "Max length: ${MAX_LENGTH}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

reactvau_require_dir "${MODEL_PATH}" "PaliGemma model directory"
reactvau_require_file "${STREAMFOREST_WEIGHTS}" "StreamForest vision weights"
reactvau_require_dir "${IMAGE_ROOT}" "VAD image root"
reactvau_require_file "${DATA_PATH}" "combined VAD training JSON"

mkdir -p "${OUTPUT_DIR}"

python vad/train_paligemma_vad_384.py \
    --model_name_or_path "${MODEL_PATH}" \
    --streamforest_vision_weights_path "${STREAMFOREST_WEIGHTS}" \
    --vision_feature_layer "${VISION_FEATURE_LAYER}" \
    --data_path "${DATA_PATH}" \
    --image_root "${IMAGE_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --use_lora True \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --freeze_vision_encoder True \
    --train_projector True \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --max_length "${MAX_LENGTH}" \
    --val_split "${VAL_SPLIT}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --gradient_checkpointing False \
    --learning_rate "${LEARNING_RATE}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --bf16 True \
    --tf32 True \
    --optim adamw_torch_fused \
    --logging_steps 50 \
    --save_steps 500 \
    --eval_steps 250 \
    --save_total_limit 3 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --dataloader_prefetch_factor 2 \
    --report_to tensorboard \
    --load_best_model_at_end True \
    --metric_for_best_model eval_auc \
    --greater_is_better True \
    --seed 42

echo "=========================================="
echo "Training complete"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="
