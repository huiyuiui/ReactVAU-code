#!/bin/bash
# ReactVAU HIVAU understanding evaluation on a single RTX 4090.
# Local non-SLURM version of scripts/sbatch/eval_reactvau_hivau_sbatch.sh.

set -e

# ==========================================
# 1. Environment Setup
# ==========================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${REACTVAU_CONFIG:-${SCRIPT_DIR}/../config/paths.sh}"
# shellcheck source=/dev/null
source "${CONFIG_PATH}"

reactvau_activate_conda

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=1
export BNB_CUDA_VERSION=121
export BITSANDBYTES_NOWELCOME=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
exec 2>&1

# ==========================================
# 2. Path Configuration
# ==========================================
PROJECT_ROOT="${REACTVAU_ROOT}"

PALIGEMMA_MODEL_PATH="${PALIGEMMA_MODEL_PATH:-${CKPT_ROOT}/paligemma2-3b-mix-448}"
PALIGEMMA_LORA_PATH="${PALIGEMMA_LORA_PATH:-${CKPT_ROOT}/paligemma2-3b-vad-lora-384-combined-binary/training_384}"
PALIGEMMA_IMAGE_SIZE="${PALIGEMMA_IMAGE_SIZE:-384}"
PALIGEMMA_SF_WEIGHTS="${PALIGEMMA_SF_WEIGHTS:-${CKPT_ROOT}/extracted_weights/streamforest_vision_encoder_with_prefix.safetensors}"
PALIGEMMA_VISION_FEATURE_LAYER="${PALIGEMMA_VISION_FEATURE_LAYER:--2}"

# Keep this conservative for a 24GB 4090. Override with PALIGEMMA_BATCH_SIZE=32 if memory allows.
PALIGEMMA_BATCH_SIZE="${PALIGEMMA_BATCH_SIZE:-1}"
PALIGEMMA_PROMPT_STYLE="${PALIGEMMA_PROMPT_STYLE:-detail}"
PALIGEMMA_ATTN="${PALIGEMMA_ATTN:-sdpa}"

STREAMFOREST_CONV_TEMPLATE="${STREAMFOREST_CONV_TEMPLATE:-qwen_2}"
STREAMFOREST_TIME_MSG="${STREAMFOREST_TIME_MSG:-short_online_v2}"
STREAMFOREST_MODEL_BASE="${STREAMFOREST_MODEL_BASE:-${CKPT_ROOT}/StreamForest-Qwen2-7B_Siglip}"
STREAMFOREST_MODEL_PATH="${STREAMFOREST_MODEL_PATH:-${REACTVAU_CHECKPOINT_PATH}}"

VIDEO_DIR="${VIDEO_DIR:-${HIVAU_ROOT}}"
ANNO_PATH="${VIDEO_DIR}/instruction/merge_instruction_test_final.jsonl"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/eval_results/hivau}"

# ==========================================
# 3. Pipeline Configuration
# ==========================================
TARGET_FPS="${TARGET_FPS:-4}"
QUERY_INTERVAL="${QUERY_INTERVAL:-4}"

# none matches the HIVAU finetuning distribution most closely; score adds PG text context.
CONTEXT_MODE="${CONTEXT_MODE:-none}"  # none, score, description
ANOMALY_THRESHOLD="${ANOMALY_THRESHOLD:-0.4}"
SF_ENHANCE_MEMORY="${SF_ENHANCE_MEMORY:-true}"
SF_PROMPT_STYLE="${SF_PROMPT_STYLE:-default}"

# ==========================================
# 4. Test Mode Configuration
# ==========================================
TEST_MODE="${TEST_MODE:-false}"
TEST_SAMPLES="${TEST_SAMPLES:-100}"
TEST_SEED="${TEST_SEED:-42}"

# ==========================================
# 5. Run Evaluation
# ==========================================
cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_PATH"

reactvau_require_dir "$PALIGEMMA_MODEL_PATH" "PaliGemma model directory"
reactvau_require_dir "$STREAMFOREST_MODEL_PATH" "ReactVAU/StreamForest checkpoint directory"
if [ -n "$STREAMFOREST_MODEL_BASE" ]; then
    reactvau_require_dir "$STREAMFOREST_MODEL_BASE" "StreamForest base model directory"
fi
reactvau_require_dir "$VIDEO_DIR" "HIVAU root"
reactvau_require_file "$ANNO_PATH" "HIVAU test annotation"
if [ "$PALIGEMMA_IMAGE_SIZE" -eq 384 ]; then
    reactvau_require_file "$PALIGEMMA_SF_WEIGHTS" "PaliGemma StreamForest vision weights"
fi

echo "=========================================="
echo "ReactVAU HIVAU Understanding Evaluation (4090)"
echo "=========================================="
echo "Start Time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""
echo "PaliGemma Model: $PALIGEMMA_MODEL_PATH"
if [ -n "$PALIGEMMA_LORA_PATH" ] && [ -d "$PALIGEMMA_LORA_PATH" ]; then
    echo "PaliGemma LoRA: $PALIGEMMA_LORA_PATH"
    PG_LORA_FLAG="--paligemma-lora-path $PALIGEMMA_LORA_PATH"
else
    echo "PaliGemma LoRA: None (using base model)"
    PG_LORA_FLAG=""
fi

echo "PaliGemma Image Size: ${PALIGEMMA_IMAGE_SIZE}x${PALIGEMMA_IMAGE_SIZE}"
if [ "$PALIGEMMA_IMAGE_SIZE" -eq 384 ]; then
    echo "PaliGemma StreamForest Weights: $PALIGEMMA_SF_WEIGHTS"
    echo "PaliGemma Vision Feature Layer: $PALIGEMMA_VISION_FEATURE_LAYER"
    PG_VISION_FLAGS="--paligemma-image-size 384 --paligemma-streamforest-weights $PALIGEMMA_SF_WEIGHTS --paligemma-vision-feature-layer $PALIGEMMA_VISION_FEATURE_LAYER"
else
    PG_VISION_FLAGS="--paligemma-image-size 448"
fi

echo ""
echo "StreamForest Model: $STREAMFOREST_MODEL_PATH"
if [ -n "$STREAMFOREST_MODEL_BASE" ]; then
    echo "StreamForest Base: $STREAMFOREST_MODEL_BASE"
    SF_BASE_FLAG="--streamforest-model-base $STREAMFOREST_MODEL_BASE"
else
    echo "StreamForest Base: None (full checkpoint)"
    SF_BASE_FLAG=""
fi

if [ "$SF_ENHANCE_MEMORY" = true ]; then
    echo "StreamForest Memory Enhancement (APS): ENABLED"
    MEMORY_FLAG="--enable-memory-enhancement"
else
    echo "StreamForest Memory Enhancement (APS): DISABLED"
    MEMORY_FLAG=""
fi

echo ""
echo "Dataset:"
echo "  Video Dir: $VIDEO_DIR"
echo "  Annotation: $ANNO_PATH"
echo ""
echo "Pipeline Config:"
echo "  Target FPS: $TARGET_FPS"
echo "  Query Interval: $QUERY_INTERVAL"
echo "  Context Mode: $CONTEXT_MODE"
echo "  SF Prompt Style: $SF_PROMPT_STYLE"
echo "  Memory Enhancement (APS): $SF_ENHANCE_MEMORY"
echo "  Anomaly Threshold: $ANOMALY_THRESHOLD"
echo "  PaliGemma Batch Size: $PALIGEMMA_BATCH_SIZE"
echo "  PaliGemma Prompt Style: $PALIGEMMA_PROMPT_STYLE"
echo ""
echo "Output: $OUTPUT_PATH"
echo "=========================================="

if [ "$TEST_MODE" = true ]; then
    echo "TEST MODE: Processing $TEST_SAMPLES random samples"
    TEST_FLAGS="--test-mode --test-samples $TEST_SAMPLES --test-seed $TEST_SEED --verbose"
else
    echo "FULL EVALUATION MODE"
    TEST_FLAGS=""
fi
echo ""

python "${PROJECT_ROOT}/eval_utils/hivau/eval_reactvau_hivau.py" \
    --paligemma-model-path "$PALIGEMMA_MODEL_PATH" \
    $PG_LORA_FLAG \
    $PG_VISION_FLAGS \
    --paligemma-prompt-style "$PALIGEMMA_PROMPT_STYLE" \
    --paligemma-attn "$PALIGEMMA_ATTN" \
    --paligemma-batch-size "$PALIGEMMA_BATCH_SIZE" \
    --streamforest-model-path "$STREAMFOREST_MODEL_PATH" \
    $SF_BASE_FLAG \
    --streamforest-conv-template "$STREAMFOREST_CONV_TEMPLATE" \
    --streamforest-time-msg "$STREAMFOREST_TIME_MSG" \
    --context-mode "$CONTEXT_MODE" \
    --anomaly-threshold "$ANOMALY_THRESHOLD" \
    --sf-prompt-style "$SF_PROMPT_STYLE" \
    --target-fps "$TARGET_FPS" \
    --query-interval "$QUERY_INTERVAL" \
    --video-dir "$VIDEO_DIR" \
    --anno-path "$ANNO_PATH" \
    --output-path "$OUTPUT_PATH" \
    $MEMORY_FLAG \
    --save-predictions \
    $TEST_FLAGS

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "End Time: $(date)"
echo "Results saved in: $OUTPUT_PATH"
echo "=========================================="
