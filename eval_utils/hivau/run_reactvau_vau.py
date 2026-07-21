# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
ReactVAU Single Video VAU Inference
Run the full PaliGemma + StreamForest pipeline on an individual video file.

Usage (via shell script):
    bash scripts/run_reactvau_vau.sh --video /path/to/video.mp4

Usage (directly):
    python eval_utils/hivau/run_reactvau_vau.py \
        --video /path/to/video.mp4 \
        --question "Describe the anomaly" \
        --pg-model-path ... --sf-model-path ...
"""
import sys
import os
import argparse
import json
import datetime
import logging
import time

# Add ReactVAU root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ReactVAU_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if ReactVAU_ROOT not in sys.path:
    sys.path.insert(0, ReactVAU_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from reactvau_inference import ReactVAUInference


def main():
    parser = argparse.ArgumentParser(
        description="ReactVAU Single Video VAU Inference (PaliGemma + StreamForest)")

    # Required
    parser.add_argument("--video", type=str, required=True,
                        help="Path to the input video file")

    # Inference
    parser.add_argument("--question", type=str,
                        default="Please describe the events in this video in detail, "
                                "including any anomalous or unusual activities.",
                        help="Question/prompt for the model")
    parser.add_argument("--task", type=str, default="",
                        choices=["", "judgement", "description", "analysis", "caption"],
                        help="HIVAU task type (affects prompt framing)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Maximum tokens to generate")

    # PaliGemma
    parser.add_argument("--pg-model-path", type=str, required=True,
                        help="PaliGemma2 model path")
    parser.add_argument("--pg-lora-path", type=str, default=None,
                        help="PaliGemma LoRA adapter path")
    parser.add_argument("--pg-image-size", type=int, default=384, choices=[384, 448],
                        help="PaliGemma input image size")
    parser.add_argument("--pg-sf-weights", type=str, default=None,
                        help="StreamForest vision encoder weights (.safetensors)")
    parser.add_argument("--pg-vision-layer", type=int, default=-2,
                        help="Vision encoder layer for feature extraction")
    parser.add_argument("--pg-batch-size", type=int, default=32,
                        help="PaliGemma batch size")
    parser.add_argument("--pg-attn", type=str, default="sdpa",
                        choices=["eager", "sdpa", "flash_attention_2"],
                        help="PaliGemma attention implementation")

    # StreamForest
    parser.add_argument("--sf-model-path", type=str, required=True,
                        help="StreamForest model checkpoint path")
    parser.add_argument("--sf-model-base", type=str, default=None,
                        help="StreamForest base model path (for LoRA)")
    parser.add_argument("--sf-conv-template", type=str, default="qwen_2",
                        help="Conversation template")
    parser.add_argument("--sf-time-msg", type=str, default="short_online_v2",
                        choices=["short_online", "short_online_v2", "simple", "none"],
                        help="Time message format")

    # Pipeline
    parser.add_argument("--context-mode", type=str, default="score",
                        choices=["none", "score"],
                        help="PG context injection mode")
    parser.add_argument("--anomaly-threshold", type=float, default=0.4,
                        help="PG anomaly threshold for context")
    parser.add_argument("--enable-memory-enhancement", action="store_true",
                        help="Enable APS memory enhancement + anomaly pool")
    parser.add_argument("--sf-prompt-style", type=str, default="skeptical",
                        choices=["default", "skeptical"],
                        help="SF prompt style for PG context framing")
    parser.add_argument("--target-fps", type=int, default=4,
                        help="Video sampling FPS")
    parser.add_argument("--query-interval", type=int, default=4,
                        help="Frames per PG query")

    # Output
    parser.add_argument("--output-dir", type=str, default="inference_results",
                        help="Directory to save result JSON")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed pipeline info")

    args = parser.parse_args()

    # ---- Validate ----
    if not os.path.isfile(args.video):
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)

    if args.pg_image_size == 384 and not args.pg_sf_weights:
        print("ERROR: 384 mode requires --pg-sf-weights")
        sys.exit(1)

    # ---- Setup logging ----
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # ---- Initialize model ----
    logging.info("=" * 60)
    logging.info("ReactVAU Single Video Inference")
    logging.info("=" * 60)
    logging.info(f"Video: {args.video}")
    logging.info(f"Question: {args.question}")
    if args.task:
        logging.info(f"Task: {args.task}")
    logging.info(f"Context mode: {args.context_mode}")
    logging.info(f"SF prompt style: {args.sf_prompt_style}")
    logging.info(f"Memory enhancement: {'ENABLED' if args.enable_memory_enhancement else 'DISABLED'}")
    logging.info(f"Anomaly threshold: {args.anomaly_threshold}")
    logging.info(f"Target FPS: {args.target_fps}, Query interval: {args.query_interval}")
    logging.info("")

    t0 = time.time()

    model = ReactVAUInference(
        paligemma_model_path=args.pg_model_path,
        paligemma_lora_path=args.pg_lora_path,
        paligemma_image_size=args.pg_image_size,
        paligemma_streamforest_weights=args.pg_sf_weights,
        paligemma_vision_feature_layer=args.pg_vision_layer,
        paligemma_attn=args.pg_attn,
        paligemma_batch_size=args.pg_batch_size,
        streamforest_model_path=args.sf_model_path,
        streamforest_model_base=args.sf_model_base,
        streamforest_conv_template=args.sf_conv_template,
        streamforest_time_msg=args.sf_time_msg,
        context_mode=args.context_mode,
        anomaly_threshold=args.anomaly_threshold,
        enable_memory_enhancement=args.enable_memory_enhancement,
        sf_prompt_style=args.sf_prompt_style,
        target_fps=args.target_fps,
        query_interval=args.query_interval,
    )

    model_load_time = time.time() - t0
    logging.info(f"Model loaded in {model_load_time:.1f}s")

    # ---- Run inference ----
    logging.info("\nRunning inference...")
    t1 = time.time()

    result = model.generate(
        video_path=args.video,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
        task=args.task,
    )

    inference_time = time.time() - t1

    # ---- Display results ----
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"\nVideo: {args.video}")
    print(f"Duration: {result['video_duration']:.1f}s")
    print(f"SF frames processed: {result['sf_frame_count']}")
    print(f"PG queries: {result['num_queries']}")
    print(f"Anomaly segments: {result['num_anomaly_segments']}")
    print(f"Inference time: {inference_time:.1f}s")

    if result["pg_context"]:
        print(f"\n--- PG Detection Context ---")
        print(result["pg_context"])

    print(f"\n--- Question ---")
    print(args.question)

    print(f"\n--- Response ---")
    print(result["response"])
    print("=" * 60)

    # ---- Save results ----
    os.makedirs(args.output_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"reactvau_{video_name}_{now_str}.json"
    output_path = os.path.join(args.output_dir, output_filename)

    save_data = {
        "video_path": os.path.abspath(args.video),
        "video_name": video_name,
        "question": args.question,
        "task": args.task,
        "response": result["response"],
        "pg_context": result["pg_context"],
        "video_duration": result["video_duration"],
        "sf_frame_count": result["sf_frame_count"],
        "num_pg_queries": result["num_queries"],
        "num_anomaly_segments": result["num_anomaly_segments"],
        "anomaly_segments": result["anomaly_segments"],
        "paligemma_scores": result["paligemma_scores"],
        "inference_time_sec": round(inference_time, 2),
        "model_load_time_sec": round(model_load_time, 2),
        "config": {
            "pg_model_path": args.pg_model_path,
            "pg_lora_path": args.pg_lora_path,
            "pg_image_size": args.pg_image_size,
            "sf_model_path": args.sf_model_path,
            "sf_model_base": args.sf_model_base,
            "context_mode": args.context_mode,
            "sf_prompt_style": args.sf_prompt_style,
            "anomaly_threshold": args.anomaly_threshold,
            "enable_memory_enhancement": args.enable_memory_enhancement,
            "target_fps": args.target_fps,
            "query_interval": args.query_interval,
            "max_new_tokens": args.max_new_tokens,
        },
        "timestamp": now_str,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
