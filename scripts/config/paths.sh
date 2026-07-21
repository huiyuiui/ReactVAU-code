#!/bin/bash

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Central path configuration for ReactVAU scripts.
#
# Public defaults are GitHub-safe. For local private paths, create
# scripts/config/paths.local.sh; it is ignored by git and loaded automatically.
# You can also override any variable from the command line, e.g.:
#   HIVAU_ROOT=/path/to/HIVAU-70k bash scripts/eval/run_eval_reactvau_hivau.sh

if [ -n "${BASH_SOURCE[0]:-}" ]; then
    _REACTVAU_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    _REACTVAU_CONFIG_DIR="$(pwd)/scripts/config"
fi

# Optional local private overrides. Keep real machine-specific paths here
# instead of editing this public template.
if [ -f "${_REACTVAU_CONFIG_DIR}/paths.local.sh" ]; then
    # shellcheck source=/dev/null
    source "${_REACTVAU_CONFIG_DIR}/paths.local.sh"
fi

export REACTVAU_ROOT="${REACTVAU_ROOT:-$(cd "${_REACTVAU_CONFIG_DIR}/../.." && pwd)}"
export CKPT_ROOT="${CKPT_ROOT:-${REACTVAU_ROOT}/ckpt}"

# Runtime environment.
export CONDA_ENV="${CONDA_ENV:-ReactVAU}"
export CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"

# Dataset roots. Set HIVAU_ROOT to the folder that contains videos/,
# instruction/, and raw_annotations/.
export HIVAU_ROOT="${HIVAU_ROOT:-/path/to/HIVAU-70k}"
export HIVAU_VIDEO_ROOT="${HIVAU_VIDEO_ROOT:-${HIVAU_ROOT}/videos}"
export VAD_TRAIN_ROOT="${VAD_TRAIN_ROOT:-${REACTVAU_ROOT}/vad/vad_data/paligemma_train}"

# PaliGemma detection module.
export PALIGEMMA_MODEL_PATH="${PALIGEMMA_MODEL_PATH:-${CKPT_ROOT}/paligemma2-3b-mix-448}"
export PALIGEMMA_LORA_PATH="${PALIGEMMA_LORA_PATH:-${CKPT_ROOT}/paligemma2-3b-vad-lora-384-combined-binary/training_384}"
export PALIGEMMA_SF_WEIGHTS="${PALIGEMMA_SF_WEIGHTS:-${CKPT_ROOT}/extracted_weights/streamforest_vision_encoder_with_prefix.safetensors}"

# ReactVAU reasoning module.
export STREAMFOREST_MODEL_BASE="${STREAMFOREST_MODEL_BASE:-${CKPT_ROOT}/StreamForest-Qwen2-7B_Siglip}"
export REACTVAU_CHECKPOINT_PATH="${REACTVAU_CHECKPOINT_PATH:-${CKPT_ROOT}/hivau-finetune/streamforest-pg-mem-v2/checkpoint-48579}"

# HIVAU training annotations and precomputed scores. Set HIVAU_TRAIN_JSON in
# paths.local.sh after obtaining the dataset annotation from its provider.
export HIVAU_TRAIN_JSON="${HIVAU_TRAIN_JSON:-}"
export PG_SCORES_PATH="${PG_SCORES_PATH:-${REACTVAU_ROOT}/precomputed/pg_scores_hivau_train.json}"

reactvau_activate_conda() {
    if [ ! -f "${CONDA_BASE}/bin/activate" ]; then
        echo "ERROR: Conda activate script not found: ${CONDA_BASE}/bin/activate"
        echo "Set CONDA_BASE or activate ${CONDA_ENV} before running this script."
        return 1
    fi
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/bin/activate" "${CONDA_ENV}"
}

reactvau_require_file() {
    local path="$1"
    local label="${2:-file}"
    if [ ! -f "${path}" ]; then
        echo "ERROR: ${label} not found: ${path}"
        return 1
    fi
}

reactvau_require_dir() {
    local path="$1"
    local label="${2:-directory}"
    if [ ! -d "${path}" ]; then
        echo "ERROR: ${label} not found: ${path}"
        return 1
    fi
}
