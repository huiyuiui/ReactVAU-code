#!/bin/bash
# PaliGemma Video Anomaly Detection Evaluation Script (single RTX 4090)
# Local version aligned with scripts/sbatch/eval_paligemma_vad_sbatch.sh.

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
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
MODEL_PATH="${MODEL_PATH:-${PALIGEMMA_MODEL_PATH}}"
LORA_PATH="${LORA_PATH:-${PALIGEMMA_LORA_PATH}}"

# Set to 384 for StreamForest SigLIP vision encoder, 448 for original PaliGemma SigLIP.
IMAGE_SIZE="${IMAGE_SIZE:-384}"
STREAMFOREST_WEIGHTS="${STREAMFOREST_WEIGHTS:-${PALIGEMMA_SF_WEIGHTS}}"
VISION_FEATURE_LAYER="${VISION_FEATURE_LAYER:--2}"

VIDEO_DIR="${VIDEO_DIR:-${HIVAU_ROOT}}"
DATASET="${DATASET:-ucf-crime}"  # Options: ucf-crime, xd-violence
if [ "$DATASET" = "ucf-crime" ]; then
    ANNO_PATH="${VIDEO_DIR}/raw_annotations/ucf_database_test_anno.txt"
else
    ANNO_PATH="${VIDEO_DIR}/raw_annotations/xd_database_test_anno.txt"
fi

OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/eval_results/vad}"

# ==========================================
# 3. Streaming Configuration
# ==========================================
TARGET_FPS="${TARGET_FPS:-4}"
QUERY_INTERVAL="${QUERY_INTERVAL:-4}"
METHOD="${METHOD:-logits}"  # logits or generate

# PaliGemma-only eval can usually use a larger batch than the combined pipeline.
BATCH_SIZE="${BATCH_SIZE:-16}"
PROMPT_STYLE="${PROMPT_STYLE:-detail}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

# ==========================================
# 4. Test Mode Configuration
# ==========================================
TEST_MODE="${TEST_MODE:-false}"
TEST_SAMPLES="${TEST_SAMPLES:-10}"

# ==========================================
# 5. Run Evaluation
# ==========================================
cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_PATH"

reactvau_require_dir "$MODEL_PATH" "PaliGemma model directory"
if [ -n "$LORA_PATH" ] && [ -d "$LORA_PATH" ]; then
    :
elif [ -n "$LORA_PATH" ]; then
    echo "WARNING: LoRA path not found, using base model only: $LORA_PATH"
fi
reactvau_require_dir "$VIDEO_DIR" "HIVAU root"
reactvau_require_file "$ANNO_PATH" "detection annotation"
if [ "$IMAGE_SIZE" -eq 384 ]; then
    reactvau_require_file "$STREAMFOREST_WEIGHTS" "StreamForest vision weights"
fi

echo "=========================================="
echo "PaliGemma VAD Streaming Evaluation (Binary, 4090)"
echo "=========================================="
echo "Start Time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""
echo "Model: $MODEL_PATH"
if [ -n "$LORA_PATH" ] && [ -d "$LORA_PATH" ]; then
    echo "LoRA: $LORA_PATH"
    LORA_FLAG="--lora-path $LORA_PATH"
else
    echo "LoRA: None (using base model)"
    LORA_FLAG=""
fi

echo "Image Size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
if [ "$IMAGE_SIZE" -eq 384 ]; then
    echo "StreamForest Weights: $STREAMFOREST_WEIGHTS"
    echo "Vision Feature Layer: $VISION_FEATURE_LAYER"
    SF_VISION_FLAGS="--image-size 384 --streamforest-vision-weights-path $STREAMFOREST_WEIGHTS --vision-feature-layer $VISION_FEATURE_LAYER"
else
    SF_VISION_FLAGS="--image-size 448"
fi

echo ""
echo "Dataset: $DATASET"
echo "Video Dir: $VIDEO_DIR"
echo "Annotation: $ANNO_PATH"
echo ""
echo "Streaming Config:"
echo "  Target FPS: $TARGET_FPS"
echo "  Query Interval: $QUERY_INTERVAL"
echo "  Method: $METHOD"
echo "  Batch Size: $BATCH_SIZE"
echo "  Prompt Style: $PROMPT_STYLE"
echo "  Attention: $ATTN_IMPLEMENTATION"
echo ""
echo "Output: $OUTPUT_PATH"
echo "=========================================="

if [ "$TEST_MODE" = true ]; then
    echo "TEST MODE: Processing $TEST_SAMPLES random samples"
    TEST_FLAGS="--test-mode --test-samples $TEST_SAMPLES"
else
    echo "FULL EVALUATION MODE"
    TEST_FLAGS=""
fi
echo ""

python "${PROJECT_ROOT}/eval_utils/vad/eval_paligemma_detection.py" \
    --model-path "$MODEL_PATH" \
    $LORA_FLAG \
    $SF_VISION_FLAGS \
    --target-fps "$TARGET_FPS" \
    --query-interval "$QUERY_INTERVAL" \
    --method "$METHOD" \
    --batch-size "$BATCH_SIZE" \
    --prompt-style "$PROMPT_STYLE" \
    --attn-implementation "$ATTN_IMPLEMENTATION" \
    --dataset "$DATASET" \
    --video-dir "$VIDEO_DIR" \
    --anno-path "$ANNO_PATH" \
    --output-path "$OUTPUT_PATH" \
    --save-video-scores \
    $TEST_FLAGS

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "End Time: $(date)"
echo "Results saved in: $OUTPUT_PATH"
echo "=========================================="
