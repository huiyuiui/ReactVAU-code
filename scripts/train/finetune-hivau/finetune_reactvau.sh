#!/bin/bash

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# ReactVAU HIVAU finetuning on a single RTX 4090.
# Local non-SLURM version of scripts/sbatch/train_reactvau_hivau_4090.sh.

set -eo pipefail

# ==========================================
# 1. Environment Setup
# ==========================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${REACTVAU_CONFIG:-${SCRIPT_DIR}/../../config/paths.sh}"
# shellcheck source=/dev/null
source "${CONFIG_PATH}"

PROJECT_ROOT="${REACTVAU_ROOT}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

reactvau_activate_conda

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export BNB_CUDA_VERSION=121
export BITSANDBYTES_NOWELCOME=1

export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export LOCAL_WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=${MASTER_PORT:-29500}

# ==========================================
# 2. Model, Data, and Output
# ==========================================
LLM_VERSION="${LLM_VERSION:-${STREAMFOREST_MODEL_BASE}}"
VISION_MODEL_VERSION="${VISION_MODEL_VERSION:-google/siglip-so400m-patch14-384}"

DATA_JSON="${DATA_JSON:-${HIVAU_TRAIN_JSON}}"
DATA_ROOT="${DATA_ROOT:-${HIVAU_VIDEO_ROOT}}"
PG_SCORES_PATH="${PG_SCORES_PATH:-${REACTVAU_ROOT}/precomputed/pg_scores_hivau_train.json}"

TUNABLE_PARTS="${TUNABLE_PARTS:-mm_mlp_adapter,mm_language_model}"
LORA_ENABLE="${LORA_ENABLE:-True}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-16}"

MM_PROJECTOR_TYPE="${MM_PROJECTOR_TYPE:-tome729_fstw_pemf}"
PROMPT_VERSION="${PROMPT_VERSION:-qwen_2}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
MID_RUN_NAME="${MID_RUN_NAME:-hivau_ft_4090_qlora_$(date +"%Y%m%d_%H%M%S")}"
OUTPUT_DIR="${OUTPUT_DIR:-ckpt/hivau-finetune/${MID_RUN_NAME}}"
mkdir -p "${OUTPUT_DIR}/runs"

# Generate a local YAML so DATA_JSON, DATA_ROOT, and PG_SCORES_PATH remain
# easy to override from the command line.
DATA_VERSION="${OUTPUT_DIR}/hivau_finetune_local.yaml"
cat > "$DATA_VERSION" <<EOF
datasets:
  - json_path: ${DATA_JSON}
    data_root: ${DATA_ROOT}
    sampling_strategy: all
    media_type: video
    video_read_type: decord
pg_scores_path: ${PG_SCORES_PATH}
EOF

# ==========================================
# 3. Pre-flight Checks
# ==========================================
echo "=========================================="
echo "ReactVAU HIVAU Finetune (RTX 4090, 4-bit QLoRA)"
echo "=========================================="
echo "Date: $(date)"
echo "Project: $PROJECT_ROOT"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'N/A')"
echo "Output: ${OUTPUT_DIR}"
echo "Data YAML: ${DATA_VERSION}"
echo ""

if [ ! -d "$LLM_VERSION" ]; then
    echo "ERROR: LLM checkpoint not found: $LLM_VERSION"
    exit 1
fi
if [ -z "$DATA_JSON" ] || [ ! -f "$DATA_JSON" ]; then
    echo "ERROR: HIVAU training JSON is not configured or not found: $DATA_JSON"
    echo "Set HIVAU_TRAIN_JSON in scripts/config/paths.local.sh or pass DATA_JSON=/path/to/train.json."
    exit 1
fi
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: HIVAU video root not found: $DATA_ROOT"
    exit 1
fi
if [ ! -f "$PG_SCORES_PATH" ]; then
    echo "ERROR: PG scores file not found: $PG_SCORES_PATH"
    echo "Run scripts/precompute/precompute_pg_scores.py first, or set PG_SCORES_PATH."
    exit 1
fi
echo "PG scores: $PG_SCORES_PATH ($(du -h "$PG_SCORES_PATH" | cut -f1))"
echo "=========================================="
echo ""

# ==========================================
# 4. Run Training
# ==========================================
set +e
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -u llava/train/train_mem.py \
    --deepspeed scripts/deepspeed/zero1.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_VERSION} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts ${TUNABLE_PARTS} \
    --mm_vision_tower_lr 2e-6 \
    --mm_vision_select_layer -2 \
    --mm_projector_type ${MM_PROJECTOR_TYPE} \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    \
    --lora_enable ${LORA_ENABLE} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout 0.05 \
    --lora_bias "none" \
    --attn_implementation sdpa \
    \
    --bits 4 \
    --quant_type "nf4" \
    --double_quant True \
    --bf16 True \
    \
    --run_name ${MID_RUN_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 100 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --dataloader_drop_last True \
    \
    --frames_upbound 64 \
    --frames_lowbound 4 \
    --time_msg short_online_v2 \
    --local_num_frames 1 \
    --vision_encode_type image_video_memory_batch \
    --sample_type dynamic_fps1 \
    --mm_pos_num_frames 1 \
    --mm_num_compress_latents 128 \
    --mm_num_compress_query_type pooling \
    --mm_close_init True \
    --mm_local_num_frames 1 \
    2>&1 | tee "${OUTPUT_DIR}/runs/${MID_RUN_NAME}.log"

EXIT_CODE=$?
set -e
echo ""
echo "=========================================="
echo "Training finished at $(date)"
echo "Exit code: $EXIT_CODE"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="
exit $EXIT_CODE
