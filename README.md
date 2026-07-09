# ReactVAU

ReactVAU is a streaming video anomaly detection and understanding framework.
It uses a lightweight detection module to produce per-second anomaly scores and
a video-language reasoning module with score-aware memory to perform deeper
verification and video understanding.

## Highlights

- Streaming inference: one anomaly query per second by default.
- Two-stage training: detection fine-tuning followed by ReactVAU understanding
  fine-tuning.
- Precomputed detector scores for efficient score-aware memory training.
- Online visual memory with SFTW, PEMF, APS, and Anomaly Pool support.
- VAD evaluation on UCF-Crime/XD-Violence style annotations.
- HIVAU understanding evaluation with BLEU, ROUGE, CIDEr, and METEOR.

## Method Overview

ReactVAU contains two interacting modules:

1. **Detection module**  
   A PaliGemma2-based binary anomaly detector. Frames are sampled at 4 FPS and
   grouped into a 2x2 grid, so each query covers about one second. The anomaly
   score is computed from the Yes/No logits.

2. **Reasoning module**  
   A StreamForest-style video-language model. During streaming inference, the
   model updates visual memory every second. High detector scores can trigger
   deeper reasoning for VAD, while HIVAU understanding uses the accumulated
   memory to answer the final question after the video stream ends.

For the second training stage, detector scores are precomputed for all HIVAU
training videos. These scores are aligned to sampled frames and passed into the
memory/projector module so APS and Anomaly Pool receive the same score signal
during training and evaluation.

## Project Structure

```text
ReactVAU/
  README.md
  scripts/
    config/
      paths.sh                         # shared path/config defaults
    train/
      finetune-vad/
        train_paligemma_vad_384.sh
        train_paligemma_vad_384_binary.sh
      finetune-hivau/
        finetune_reactvau.sh
        hivau_finetune.yaml
        hivau_minimal.json
    precompute/
      precompute_pg_scores.py
      verify_scores.py
    eval/
      run_eval_paligemma_detect.sh
      run_eval_reactvau_detect.sh
      run_eval_reactvau_hivau.sh
  vad/
    train_paligemma_vad_384.py
    get_prompt.py
  eval_utils/
    vad/
      eval_paligemma_detection.py
      eval_reactvau_detection.py
    hivau/
      eval_reactvau_hivau.py
      run_reactvau_vau.py
      reactvau_inference.py
      hivau_utils.py
  llava/
    train/
      train.py
      train_mem.py
    model/
      multimodal_projector/
        projector_FSTW_PEMF.py
        memory_manager.py
  precomputed/
    pg_scores_hivau_train.json
```

## Installation

Create a Python environment and install the required packages:

```bash
conda create -n ReactVAU python=3.10
conda activate ReactVAU

# Install PyTorch for your CUDA version first.
# Then install the project dependencies.
pip install -r requirements.txt
```

The code uses PyTorch, Transformers, PEFT, Accelerate, DeepSpeed,
bitsandbytes, safetensors, OpenCV, Pillow, decord, scikit-learn, tqdm, loguru,
TensorBoard, pycocoevalcap, and NLTK.

## Configuration

All maintained shell scripts read common paths from:

```text
scripts/config/paths.sh
```

You can edit this file directly or create a private local override:

```text
scripts/config/paths.local.sh
```

`paths.local.sh` is loaded automatically by `paths.sh`. A typical local config
looks like:

```bash
export REACTVAU_ROOT="/path/to/ReactVAU"
export CKPT_ROOT="${REACTVAU_ROOT}/ckpt"
export HIVAU_ROOT="/path/to/HIVAU-70k"
export HIVAU_VIDEO_ROOT="${HIVAU_ROOT}/videos"

export CONDA_ENV="ReactVAU"
export CONDA_BASE="${HOME}/miniconda3"
export CUDA_HOME="/usr/local/cuda-12.1"

export PALIGEMMA_MODEL_PATH="${CKPT_ROOT}/paligemma2-3b-mix-448"
export PALIGEMMA_LORA_PATH="${CKPT_ROOT}/paligemma2-3b-vad-lora-384-combined-binary/training_384"
export PALIGEMMA_SF_WEIGHTS="${CKPT_ROOT}/extracted_weights/streamforest_vision_encoder_with_prefix.safetensors"

export STREAMFOREST_MODEL_BASE="${CKPT_ROOT}/StreamForest-Qwen2-7B_Siglip"
export REACTVAU_CHECKPOINT_PATH="${CKPT_ROOT}/hivau-finetune/<reactvau-checkpoint>"
export PG_SCORES_PATH="${REACTVAU_ROOT}/precomputed/pg_scores_hivau_train.json"
```

Every variable can also be overridden from the command line:

```bash
HIVAU_ROOT=/path/to/HIVAU-70k TEST_MODE=true bash scripts/eval/run_eval_reactvau_hivau.sh
```

## Checkpoints

Place model weights under `ckpt/`:

```text
ckpt/
  paligemma2-3b-mix-448/
  extracted_weights/
    streamforest_vision_encoder_with_prefix.safetensors
  paligemma2-3b-vad-lora-384-combined-binary/
    training_384/
  StreamForest-Qwen2-7B_Siglip/
  hivau-finetune/
    <reactvau-checkpoint>/
```

`paligemma2-3b-mix-448` is the PaliGemma2 base checkpoint. The extracted
StreamForest vision weights are used to adapt PaliGemma detection to the
384x384 StreamForest vision setting. `StreamForest-Qwen2-7B_Siglip` is used as
the video-language reasoning backbone.

## Dataset Preparation

### Detection Training Data

The detection module expects 2x2 grid-image training data:

```text
vad/vad_data/paligemma_train/
  combined_train_binary.json
  ucf_crime_train_binary.json
  xd_violence_train_binary.json
  <grid image files>
```

Each JSON item should contain an image path relative to `VAD_TRAIN_ROOT` and a
binary suffix:

```json
[
  {
    "image": "relative/path/to/grid_image.jpg",
    "suffix": "Yes"
  },
  {
    "image": "relative/path/to/grid_image.jpg",
    "suffix": "No"
  }
]
```

### HIVAU Data

Set `HIVAU_ROOT` to a folder with the following structure:

```text
HIVAU-70k/
  videos/
    ucf-crime/
    xd-violence/
  instruction/
    detection/
      ucf_crime_detection_test.json
      xd_violence_detection_test.json
    merge_instruction_test_final.jsonl
  raw_annotations/
    ucf_database_test_anno.txt
    xd_database_test_anno.txt
```

HIVAU fine-tuning uses:

```text
scripts/train/finetune-hivau/hivau_minimal.json
scripts/train/finetune-hivau/hivau_finetune.yaml
```

The YAML format is:

```yaml
datasets:
  - json_path: scripts/train/finetune-hivau/hivau_minimal.json
    data_root: /path/to/HIVAU-70k/videos
    sampling_strategy: all
    media_type: video
    video_read_type: decord

pg_scores_path: precomputed/pg_scores_hivau_train.json
```

The training script also generates a run-local YAML file so `DATA_JSON`,
`DATA_ROOT`, and `PG_SCORES_PATH` can be overridden without editing the repo
file.

## Training

### Stage 1: Detection Module

Train the PaliGemma2 VAD detector on the combined binary detection dataset:

```bash
cd /path/to/ReactVAU
bash scripts/train/finetune-vad/train_paligemma_vad_384_binary.sh
```

The script uses 384x384 inputs, StreamForest vision encoder weights,
second-to-last vision features (`vision_feature_layer=-2`), frozen vision
encoder, LoRA on the language model, and projector fine-tuning.

Useful overrides:

```bash
BATCH_SIZE=4 GRAD_ACCUM=8 \
OUTPUT_DIR=ckpt/paligemma2-3b-vad-lora-384-combined-binary/training_384 \
bash scripts/train/finetune-vad/train_paligemma_vad_384_binary.sh
```

### Stage 1 Evaluation: PaliGemma-Only VAD

```bash
DATASET=ucf-crime TEST_MODE=true TEST_SAMPLES=10 \
bash scripts/eval/run_eval_paligemma_detect.sh

DATASET=xd-violence TEST_MODE=true TEST_SAMPLES=10 \
bash scripts/eval/run_eval_paligemma_detect.sh
```

Set `TEST_MODE=false` for the full evaluation.

### Precompute Detector Scores

Before Reasoning Module fine-tuning, precompute PaliGemma anomaly scores for the HIVAU
training videos:

```bash
source scripts/config/paths.sh

python scripts/precompute/precompute_pg_scores.py \
  --paligemma-model-path "${PALIGEMMA_MODEL_PATH}" \
  --paligemma-lora-path "${PALIGEMMA_LORA_PATH}" \
  --paligemma-image-size 384 \
  --paligemma-streamforest-weights "${PALIGEMMA_SF_WEIGHTS}" \
  --paligemma-vision-feature-layer -2 \
  --paligemma-attn sdpa \
  --paligemma-prompt-style detail \
  --anno-path scripts/train/finetune-hivau/hivau_minimal.json \
  --data-root "${HIVAU_VIDEO_ROOT}" \
  --target-fps 4 \
  --query-interval 4 \
  --batch-size 32 \
  --output-path precomputed/pg_scores_hivau_train.json \
  --resume
```

The output JSON maps each training video to per-query detector scores:

```json
{
  "relative/video/path.mp4": {
    "pg_scores": [0.7311, 0.6225],
    "n_frames": 743,
    "sample_interval": 7,
    "num_queries": 27
  }
}
```

### Stage 2: Reasoning Module Fine-Tuning

```bash
DATA_ROOT="${HIVAU_VIDEO_ROOT}" \
PG_SCORES_PATH=precomputed/pg_scores_hivau_train.json \
OUTPUT_DIR=ckpt/hivau-finetune/reactvau-hivau \
bash scripts/train/finetune-hivau/finetune_reactvau.sh
```

Default settings include 4-bit QLoRA, DeepSpeed ZeRO-1, `tome729_fstw_pemf`,
`sample_type=dynamic_fps1`, `time_msg=short_online_v2`, and score-aware memory
training.

## Evaluation

### ReactVAU Streaming VAD

```bash
DATASET=ucf-crime TEST_MODE=true TEST_SAMPLES=5 \
bash scripts/eval/run_eval_reactvau_detect.sh
```

Common options:

```bash
ANOMALY_THRESHOLD=0.3486
SCORE_FUSION=weighted
FUSION_ALPHA=0.40
SF_ENHANCE_MEMORY=true
POOL_THRESHOLD=0.6
```

Outputs are saved under:

```text
eval_results/vad/<run-name>/
  detection.log
  summary.json
  video_results.json
```

### ReactVAU HIVAU Understanding

```bash
TEST_MODE=true TEST_SAMPLES=20 \
bash scripts/eval/run_eval_reactvau_hivau.sh
```

Common options:

```bash
CONTEXT_MODE=none        # none, score, or description
SF_ENHANCE_MEMORY=true
ANOMALY_THRESHOLD=0.4
PALIGEMMA_BATCH_SIZE=1
```

Outputs are saved under:

```text
eval_results/hivau/<run-name>/
  evaluation.log
  predictions.json
  summary.json
  hivau-BLEU.json
  hivau-ROUGE.json
  hivau-CIDEr.json
  hivau-METEOR.json
```

## Single-Video Inference

```bash
source scripts/config/paths.sh

python eval_utils/hivau/run_reactvau_vau.py \
  --video /path/to/video.mp4 \
  --question "Please describe the events in this video in detail." \
  --pg-model-path "${PALIGEMMA_MODEL_PATH}" \
  --pg-lora-path "${PALIGEMMA_LORA_PATH}" \
  --pg-image-size 384 \
  --pg-sf-weights "${PALIGEMMA_SF_WEIGHTS}" \
  --pg-vision-layer -2 \
  --sf-model-base "${STREAMFOREST_MODEL_BASE}" \
  --sf-model-path "${REACTVAU_CHECKPOINT_PATH}" \
  --context-mode none \
  --anomaly-threshold 0.4 \
  --enable-memory-enhancement \
  --target-fps 4 \
  --query-interval 4 \
  --output-dir inference_results
```

The output JSON contains the generated response, detector scores, anomaly
segments, and run configuration.

## Important Parameters

- `target_fps=4`: sample 4 frames per second.
- `query_interval=4`: one detector query per second.
- `paligemma_image_size=384`: use StreamForest SigLIP-384 vision weights.
- `paligemma_vision_feature_layer=-2`: use second-to-last vision features.
- `anomaly_threshold`: detector score threshold for triggering reasoning or
  building context.
- `pool_threshold=0.6`: Anomaly Pool insertion threshold.
- `enable_memory_enhancement`: enable APS and Anomaly Pool.
- `context_mode`: HIVAU PG-context mode, one of `none`, `score`, or
  `description`.

## Acknowledgements

This project builds on PaliGemma, LLaVA-style multimodal training code, and
StreamForest-style video-language modeling. Please cite the corresponding
projects if you use this code.

## Citation

```bibtex
@misc{reactvau,
  title  = {ReactVAU: Streaming Video Anomaly Understanding with Score-Aware Memory},
  author = {Anonymous},
  year   = {2026}
}
```
