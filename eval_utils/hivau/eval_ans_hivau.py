# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import json
import argparse
import numpy as np

from hivau_utils import hivau_BLEU, hivau_ROUGE, hivau_CIDEr, hivau_METEOR

PROJECT_ROOT = os.environ.get(
    "REACTVAU_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str,
                        default=os.path.join(PROJECT_ROOT, "eval_results", "hivau", "predictions.json"),
                        help="Path to the results.json file")
    parser.add_argument("--output-path", type=str, default='')
    args = parser.parse_args()
    
    args.output_path = args.output_path if args.output_path else os.path.dirname(args.results_path)
    print(f"Output path: {args.output_path}")

    # Load results
    print(f"Loading results from: {args.results_path}")
    with open(args.results_path, 'r') as f:
        eval_results = json.load(f)

    print(f"Number of evaluated instances: {len(eval_results)}")

    # Calculate metrics
    bleu_score, all_bleu = hivau_BLEU(eval_results, args)
    cider_score = hivau_CIDEr(eval_results, args) 
    meteor_score = hivau_METEOR(eval_results, args)
    rouge_score = hivau_ROUGE(eval_results, args)

    # all_bleu = {i: v.tolist() for i, v in all_bleu.items()}

    # Print results
    print("\nEvaluation Results:")
    print(f"BLEU Score: {bleu_score}")
    print(f"BLEU ALL Score: {all_bleu}")
    print(f"CIDEr Score: {cider_score}")
    print(f"METEOR Score: {meteor_score}")
    print(f"ROUGE Score: {rouge_score}")

    # Save metrics to a separate file
    metrics = {
        'bleu': bleu_score,
        'bleu_all': all_bleu,
        'cider': cider_score,
        'meteor': meteor_score,
        'rouge': rouge_score
    }

    metrics_path = os.path.join(args.output_path, 'hivau_.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()
