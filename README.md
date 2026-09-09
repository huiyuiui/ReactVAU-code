<h1 align="center">ReactVAU: A Slow-Fast Decoupled Framework for Streaming Video Anomaly Understanding</h1>

<p align="center">
  Chia-Hui Chen<sup>1</sup>,
  Shih-Ying Yeh<sup>1</sup>,
  Fu-En Yang<sup>2</sup>,
  Min-Hung Chen<sup>2</sup>,
  Shang-Hong Lai<sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>National Tsing Hua University &nbsp;&nbsp;
  <sup>2</sup>NVIDIA
</p>

<p align="center"><strong>🎉 Accepted to ECCV 2026 🎉</strong></p>

<p align="center">
  <a href="https://huiyuiui.github.io/ReactVAU/"><img src="https://img.shields.io/badge/Project-Page-4B8BBE?style=for-the-badge&amp;logo=googlechrome&amp;logoColor=white" alt="Project Page" /></a>
  <a href="https://arxiv.org/abs/2609.07941"><img src="https://img.shields.io/badge/arXiv-2609.07941-B31B1B?style=for-the-badge&amp;logo=arxiv&amp;logoColor=white" alt="arXiv: 2609.07941" /></a>
  <a href="https://huiyuiui.github.io/ReactVAU/static/pdfs/ReactVAU_supp.pdf"><img src="https://img.shields.io/badge/Supplementary-PDF-2F6FAD?style=for-the-badge&amp;logo=adobeacrobatreader&amp;logoColor=white" alt="Supplementary PDF" /></a>
</p>

ReactVAU is a Slow-Fast Decoupled Framework for causal, streaming Video Anomaly Understanding (VAU). It separates continuous lightweight anomaly monitoring from heavyweight semantic reasoning, so normal video streams do not repeatedly invoke a large multimodal language model.

<p align="center">
  <img src="assets/teaser.png" width="100%" alt="Comparison between an always-on online model and ReactVAU" />
</p>

## Introduction

Conventional VAU systems commonly require access to the complete video, which conflicts with real deployment where future frames are unavailable. ReactVAU operates causally: it observes only the current and past stream, produces one Fast-module score per second by default, and invokes the Slow module only for suspicious intervals.

The framework consists of three components:

1. **Fast Detection Module.** PaliGemma2-3B applies Spatial Grid Folding (SGF): four frames sampled at 4 FPS are folded into one 2x2 grid and scored through the `Yes`/`No` logits.
2. **Anomaly-Aware Persistent Memory (AAPM).** The Fast anomaly score protects suspicious evidence during memory compression through the Anomaly Priority Score (APS), the Anomaly Pool, and dense Real-Time Perception for triggered intervals.
3. **Slow Reasoning Module.** A StreamForest-7B backbone remains dormant during normal streaming. When triggered, it verifies the event from the AAPM memory and produces a semantic explanation.

<p align="center">
  <img src="assets/reactvau.png" width="100%" alt="ReactVAU architecture" />
</p>

## Preparation

### Environment

ReactVAU is tested on Linux with Python 3.10 and CUDA-enabled GPUs. The Stage-1 and Stage-2 training scripts are configured for one NVIDIA RTX 4090 (24 GB). Create the supplied Conda environment with:

```bash
conda env create -f environment.yml
conda activate ReactVAU
```

Alternatively, create a minimal environment and install the PyTorch build that matches your CUDA driver before installing the remaining dependencies:

```bash
conda create -n ReactVAU python=3.10
conda activate ReactVAU
# Install a CUDA-compatible PyTorch build first.
pip install -r requirements.txt
```

HIVAU-70K metric evaluation also requires a Java runtime for METEOR.

### Download external model backbones

ReactVAU-specific adapters are trained locally by the steps below; no ReactVAU checkpoint download is required. Download only the two upstream model backbones:

| Backbone | Official source | Default local directory |
| --- | --- | --- |
| PaliGemma2-3B Mix 448 | [google/paligemma2-3b-mix-448](https://huggingface.co/google/paligemma2-3b-mix-448) | `ckpt/paligemma2-3b-mix-448/` |
| StreamForest-Qwen2-7B | [MCG-NJU/StreamForest-Qwen2-7B](https://huggingface.co/MCG-NJU/StreamForest-Qwen2-7B) | `ckpt/StreamForest-Qwen2-7B_Siglip/` |

After accepting any model terms on Hugging Face, the checkpoints can be downloaded with:

```bash
hf auth login
hf download google/paligemma2-3b-mix-448 \
  --local-dir ckpt/paligemma2-3b-mix-448
hf download MCG-NJU/StreamForest-Qwen2-7B \
  --local-dir ckpt/StreamForest-Qwen2-7B_Siglip
```

ReactVAU shares the StreamForest SigLIP-384 vision encoder with its Fast module. Extract that encoder once from the downloaded StreamForest shards so the existing Stage-1, precompute, and evaluation entry points all use the same 384x384 encoder configuration:

```bash
python scripts/setup/extract_streamforest_vision.py \
  --streamforest-dir ckpt/StreamForest-Qwen2-7B_Siglip \
  --output ckpt/extracted_weights/streamforest_vision_encoder_with_prefix.safetensors
```

The command validates the checkpoint layout, the expected 421 vision tensors, and the required embedding shapes before writing the output. At this point the external and generated model files are separated as follows:

```text
ckpt/
  paligemma2-3b-mix-448/                         # downloaded upstream backbone
  StreamForest-Qwen2-7B_Siglip/                  # downloaded upstream backbone
  extracted_weights/
    streamforest_vision_encoder_with_prefix.safetensors  # generated locally
  paligemma2-3b-vad-lora-384-combined-binary/    # Stage-1 Fast module adapter
    training_384/                                 # generated by Stage 1
  hivau-finetune/
    reactvau-hivau/                               # generated by Stage 2
```

### Datasets

ReactVAU uses UCF-Crime and XD-Violence for anomaly detection and HIVAU-70K for anomaly understanding. Download each dataset from its official source and comply with its license:

- [HIVAU-70K annotations and preparation scripts](https://github.com/pipixin321/HolmesVAU/tree/master/HIVAU-70k)
- [UCF-Crime videos and temporal test annotations](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/)
- [XD-Violence videos and test annotations](https://roc-ng.github.io/XD-Violence/)

Use the split and filenames specified by HIVAU-70K. Place the video files directly in their corresponding `train/` and `test/` directories, copy the four database JSON files, the instruction JSONL files, `split_video.py`, and `check_video.py` from HIVAU-70K, and place the official frame-level VAD test annotations at the filenames expected by the evaluation scripts:

- Copy UCF-Crime's `Temporal_Anomaly_Annotation_for_Testing_Videos.txt` to `raw_annotations/ucf_database_test_anno.txt`.
- Copy XD-Violence's official test annotation text file to `raw_annotations/xd_database_test_anno.txt`.

The final layout under `HIVAU_ROOT` must be:

```text
HIVAU-70k/
  instruction/
    merge_instruction_train_final.jsonl
    merge_instruction_train_final.json          # generated below
    merge_instruction_test_final.jsonl
  raw_annotations/
    ucf_database_train.json
    ucf_database_test.json
    xd_database_train.json
    xd_database_test.json
    ucf_database_test_anno.txt
    xd_database_test_anno.txt
  split_video.py
  check_video.py
  videos/
    ucf-crime/
      clips/                                  # generated by split_video.py
      events/                                 # generated by split_video.py
      videos/
        train/
          <video_name>.mp4
        test/
          <video_name>.mp4
    xd-violence/
      clips/                                  # generated by split_video.py
      events/                                 # generated by split_video.py
      videos/
        train/
          <video_name>.mp4
        test/
          <video_name>.mp4
```

Generate and validate the `clips/` and `events/` videos referenced by the HIVAU instructions. Run these commands from the downloaded HIVAU-70K directory; splitting can take several hours:

```bash
cd /path/to/HIVAU-70k
python split_video.py
python check_video.py
cd /path/to/ReactVAU
```

The paths stored in `merge_instruction_train_final.jsonl` are relative to `HIVAU_ROOT/videos/`; do not rewrite them when copying the official metadata. The Stage-2 loader accepts JSONL, but the score-precompute entry point reads a JSON array. Convert the official training JSONL once so both steps consume the same records:

```bash
python - <<'PY'
import json
from pathlib import Path

src = Path("/path/to/HIVAU-70k/instruction/merge_instruction_train_final.jsonl")
dst = src.with_suffix(".json")
with src.open(encoding="utf-8") as handle:
    records = [json.loads(line) for line in handle if line.strip()]
with dst.open("w", encoding="utf-8") as handle:
    json.dump(records, handle, ensure_ascii=False)
print(f"Wrote {len(records)} records to {dst}")
PY
```

### Configure local paths

All maintained shell scripts load [`scripts/config/paths.sh`](scripts/config/paths.sh), which automatically sources the ignored file `scripts/config/paths.local.sh` when present. Create `paths.local.sh` with only local overrides; do not copy `paths.sh` into it, because the copied loader would source itself recursively.

```bash
# scripts/config/paths.local.sh
export REACTVAU_ROOT="/absolute/path/to/ReactVAU"
export CKPT_ROOT="${REACTVAU_ROOT}/ckpt"
export HIVAU_ROOT="/absolute/path/to/HIVAU-70k"
export HIVAU_VIDEO_ROOT="${HIVAU_ROOT}/videos"

export CONDA_ENV="ReactVAU"
export CONDA_BASE="${HOME}/miniconda3"
export CUDA_HOME="/usr/local/cuda-12.1"

export PALIGEMMA_MODEL_PATH="${CKPT_ROOT}/paligemma2-3b-mix-448"
export PALIGEMMA_SF_WEIGHTS="${CKPT_ROOT}/extracted_weights/streamforest_vision_encoder_with_prefix.safetensors"
export PALIGEMMA_LORA_PATH="${CKPT_ROOT}/paligemma2-3b-vad-lora-384-combined-binary/training_384"
export STREAMFOREST_MODEL_BASE="${CKPT_ROOT}/StreamForest-Qwen2-7B_Siglip"

export HIVAU_TRAIN_JSON="${HIVAU_ROOT}/instruction/merge_instruction_train_final.json"
export PG_SCORES_PATH="${REACTVAU_ROOT}/precomputed/pg_scores_hivau_train.json"
```

`PALIGEMMA_LORA_PATH` and `PG_SCORES_PATH` are output locations until Stage 1 and precompute finish. `REACTVAU_CHECKPOINT_PATH` is added after Stage 2, as described below. Every variable can also be overridden for one command, for example `HIVAU_ROOT=/data/HIVAU-70k bash ...`.

## Reproduction workflow

Run all commands from the repository root after configuring `paths.local.sh`. ReactVAU is trained in two stages. Stage 1 produces the Fast-module adapter; its anomaly scores are then precomputed for every HIVAU training video and used to train the Stage-2 Slow module with the same AAPM signal used at inference time.

### 1. Construct the Stage-1 grid dataset

The Fast Detection Module is trained on a combined binary dataset constructed from the UCF-Crime and XD-Violence training splits. Each sample contains four temporally ordered frames folded into 2x2 grid. The generator samples anomalous intervals as positives, normal intervals in anomalous videos as hard negatives, and UCF-Crime normal videos as easy negatives. It balances these groups with the default `1:1:1` positive:hard-negative:easy-negative ratio.

First validate paths and video decoding on a small subset. This command writes to a separate smoke-test directory:

```bash
source scripts/config/paths.sh
VAD_TRAIN_ROOT="${REACTVAU_ROOT}/vad/vad_data/paligemma_train_smoke" \
  bash scripts/gen_data/gen_train_data.sh --test-mode --test-samples 5
```

Then construct the full dataset at the configured `VAD_TRAIN_ROOT`:

```bash
bash scripts/gen_data/gen_train_data.sh
```

The default output directory is `vad/vad_data/paligemma_train/` (or `VAD_TRAIN_ROOT` if overridden):

```text
vad/vad_data/paligemma_train/
  train_images/                         # 384x384 PNG grids
  ucf_crime_train_binary.json
  xd_violence_train_binary.json
  combined_train_binary.json            # input to the Stage-1 training script
  training_stats.json
```

Before training, confirm that `combined_train_binary.json` is non-empty and that its `image` paths resolve below `VAD_TRAIN_ROOT`.

### 2. Train Stage 1: Fast Detection Module

```bash
bash scripts/train/finetune-vad/train_paligemma_vad_384.sh
```

The default run uses 384x384 grid images, the extracted shared vision encoder, vision layer `-2`, a frozen vision encoder, LoRA on the PaliGemma2 language model, and a trainable multimodal projector. It writes the selected adapter and processor files to `PALIGEMMA_LORA_PATH`. A successful output contains `adapter_config.json` and `adapter_model.*`.

### 3. Precompute Fast-module scores for Stage 2

Precompute one sequence of Fast scores for every unique video referenced by `HIVAU_TRAIN_JSON`:

```bash
source scripts/config/paths.sh
python scripts/precompute/precompute_pg_scores.py --resume
```

The command uses the same 4 FPS sampling, four-frame grid construction, 384x384 encoder, layer `-2`, and Stage-1 adapter as evaluation. Results are written to `PG_SCORES_PATH`; `--resume` retains completed videos when restarting an interrupted run. The final log reports the number of processed and failed videos. If any failures are reported, inspect the adjacent `*_failed.json`, repair the missing video paths, and rerun with `--resume` before Stage 2.

Check that the generated JSON is readable and contains entries:

```bash
python -c 'import json, os; p=os.environ["PG_SCORES_PATH"]; d=json.load(open(p)); assert d; print(f"{len(d)} videos in {p}")'
```

Each entry contains `pg_scores`, `n_frames`, `sample_interval`, and `num_queries`; `num_queries` must equal the length of `pg_scores`.

### 4. Train Stage 2: Slow Reasoning Module

Use a stable output path so evaluation can reference the locally trained adapter directly:

```bash
source scripts/config/paths.sh
MID_RUN_NAME=reactvau-hivau \
OUTPUT_DIR="${CKPT_ROOT}/hivau-finetune/reactvau-hivau" \
  bash scripts/train/finetune-hivau/finetune_reactvau.sh
```

The default Stage-2 configuration uses the downloaded StreamForest backbone, 4-bit QLoRA, DeepSpeed ZeRO-1, `tome729_fstw_pemf`, dynamic 1 FPS memory sampling, `short_online_v2` time messages, and the precomputed Fast scores. The output directory contains `adapter_config.json`, `adapter_model.*`, `non_lora_trainables.bin`, and the run configuration.

After Stage 2 completes, add the generated output to `paths.local.sh`:

```bash
export REACTVAU_CHECKPOINT_PATH="${CKPT_ROOT}/hivau-finetune/reactvau-hivau"
```

At this point the complete dependency chain is:

```text
downloaded videos + HIVAU annotations
  -> generated 2x2 grid dataset
  -> locally trained Stage-1 Fast adapter
  -> precomputed Fast scores for HIVAU training videos
  -> locally trained Stage-2 Slow adapter/projector
  -> VAD, HIVAU-70K, and single-video evaluation
```

## Evaluation and inference

All evaluations below require the locally trained Stage-1 and Stage-2 outputs and the configured `REACTVAU_CHECKPOINT_PATH`. Always run a small `TEST_MODE=true` smoke test before the full benchmark. Test mode samples only a subset and must not be reported as a benchmark result.

### Fast-module streaming VAD

Evaluate the Fast module independently on both VAD datasets:

```bash
DATASET=ucf-crime TEST_MODE=true TEST_SAMPLES=10 \
  bash scripts/eval/run_eval_paligemma_detect.sh

DATASET=xd-violence TEST_MODE=true TEST_SAMPLES=10 \
  bash scripts/eval/run_eval_paligemma_detect.sh
```

For the complete split, set `TEST_MODE=false`:

```bash
DATASET=ucf-crime TEST_MODE=false \
  bash scripts/eval/run_eval_paligemma_detect.sh

DATASET=xd-violence TEST_MODE=false \
  bash scripts/eval/run_eval_paligemma_detect.sh
```

Each run saves `detection.log`, `summary.json`, and `video_results.json` under a timestamped directory in `eval_results/vad/`. The summary reports raw, non-causal Gaussian-smoothed, and causal online-smoothed ROC-AUC/PR-AUC/AP. Use the causal online-smoothed result for the streaming setting.

### Full ReactVAU streaming VAD

The full pipeline performs Fast scoring and AAPM updates continuously and invokes the Slow module only after the trigger threshold is crossed. Run UCF-Crime with the supplied paper configuration:

```bash
DATASET=ucf-crime TEST_MODE=true TEST_SAMPLES=5 \
  bash scripts/eval/run_eval_reactvau_detect.sh

DATASET=ucf-crime TEST_MODE=false \
  bash scripts/eval/run_eval_reactvau_detect.sh
```

The maintained script sets the UCF-Crime recall-90 trigger threshold to `0.3486`. For the XD-Violence protocol, change the `ANOMALY_THRESHOLD` selection in `scripts/eval/run_eval_reactvau_detect.sh` to the already documented XD-Violence value `0.4073`, then run:

```bash
DATASET=xd-violence TEST_MODE=true TEST_SAMPLES=5 \
  bash scripts/eval/run_eval_reactvau_detect.sh

DATASET=xd-violence TEST_MODE=false \
  bash scripts/eval/run_eval_reactvau_detect.sh
```

Keep the remaining paper settings unchanged: weighted score fusion with `FUSION_ALPHA=0.40`, Anomaly Pool threshold `0.6`, APS memory enhancement, RT-Anomaly dense encoding, and causal online smoothing. Outputs are written to a timestamped `reactvau-<dataset>-...` directory under `eval_results/vad/` and contain `detection.log`, `summary.json`, and `video_results.json`.

### HIVAU-70K anomaly-understanding evaluation

Run a smoke test and then the complete test annotation:

```bash
TEST_MODE=true TEST_SAMPLES=20 \
  bash scripts/eval/run_eval_reactvau_hivau.sh

TEST_MODE=false \
  bash scripts/eval/run_eval_reactvau_hivau.sh
```

For the paper-aligned VAU protocol, keep `CONTEXT_MODE=none`, `SF_ENHANCE_MEMORY=true`, and `ANOMALY_THRESHOLD=0.4`. Fast scores shape AAPM internally and are not inserted into the final language question. The evaluation script performs generation and computes clip-, event-, and video-level BLEU, ROUGE, CIDEr, and METEOR in the same run.

Results are saved under a timestamped directory in `eval_results/hivau/`:

```text
evaluation.log
predictions.json
summary.json
hivau-BLEU.json
hivau-ROUGE.json
hivau-CIDEr.json
hivau-METEOR.json
```

### Single-video inference

After both stages are trained, run the full causal pipeline on an arbitrary video:

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

The output JSON records the generated response, Fast scores, detected anomaly segments, timing, and run configuration.

## Citation

```bibtex
@misc{chen2026reactvau,
  title     = {ReactVAU: A Slow-Fast Decoupled Framework for Streaming Video Anomaly Understanding},
  author    = {Chen, Chia-Hui and Yeh, Shih-Ying and Yang, Fu-En and Chen, Min-Hung and Lai, Shang-Hong},
  year      = {2026},
  eprint    = {2609.07941},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url       = {https://arxiv.org/abs/2609.07941}
}
```

## Acknowledgements

This repository builds on PaliGemma, LLaVA-style multimodal training infrastructure, and StreamForest for streaming video memory. We use the HIVAU-70K benchmark introduced by Holmes-VAU, together with UCF-Crime and XD-Violence for detection training and evaluation. Please cite the corresponding work when using their models, code, or data.

## Licenses

Copyright © 2026, NVIDIA Corporation. All rights reserved.

This work is made available under the NVIDIA Source Code License-NC. Click [here](LICENSE) to view a copy of this license.
