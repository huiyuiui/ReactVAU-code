# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import json
from pathlib import Path

PROJECT_ROOT = os.environ.get(
    "REACTVAU_ROOT",
    str(Path(__file__).resolve().parents[2])
)


def calculate_layer_averages(results_dir: str):
    """
    Calculate average scores for Clip, Event, and Video levels
    
    Args:
        results_dir: Path to results directory
    """
    results_dir = Path(results_dir)
    
    # Read four JSON files
    with open(results_dir / "hivau-BLEU.json") as f:
        bleu_data = json.load(f)
    
    with open(results_dir / "hivau-ROUGE.json") as f:
        rouge_data = json.load(f)
    
    with open(results_dir / "hivau-CIDEr.json") as f:
        cider_data = json.load(f)
    
    with open(results_dir / "hivau-METEOR.json") as f:
        meteor_data = json.load(f)
    
    # Calculate average scores for each level
    results = {}
    
    # === BLEU ===
    # BLEU needs to handle 4 values, sum them first then average
    for layer in ['clip', 'event', 'video']:
        tasks = bleu_data['detailed_scores'][layer]
        
        # Collect BLEU-1 to BLEU-4 scores from all tasks
        all_bleu_scores = [0.0, 0.0, 0.0, 0.0]  # [BLEU-1, BLEU-2, BLEU-3, BLEU-4]
        valid_task_count = 0
        
        for task_name, scores in tasks.items():
            if isinstance(scores, list) and len(scores) == 4:
                # Valid BLEU scores (not all zeros)
                if sum(scores) > 0:
                    for i in range(4):
                        all_bleu_scores[i] += scores[i]
                    valid_task_count += 1
        
        # Calculate average
        if valid_task_count > 0:
            avg_bleu = [score / valid_task_count for score in all_bleu_scores]
        else:
            avg_bleu = [0.0, 0.0, 0.0, 0.0]
        
        results[f'BLEU_{layer[0].upper()}'] = sum(avg_bleu)
        results[f'BLEU_{layer[0].upper()}_all'] = avg_bleu
    
    # === ROUGE ===
    for layer in ['clip', 'event', 'video']:
        tasks = rouge_data['detailed_scores'][layer]
        
        # Calculate average of non-zero tasks
        valid_scores = [score for score in tasks.values() if score > 0]
        avg_rouge = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        
        results[f'ROUGE_{layer[0].upper()}'] = avg_rouge
    
    # === CIDEr ===
    for layer in ['clip', 'event', 'video']:
        tasks = cider_data['detailed_scores'][layer]
        
        valid_scores = [score for score in tasks.values() if score > 0]
        avg_cider = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        
        results[f'CIDEr_{layer[0].upper()}'] = avg_cider
    
    # === METEOR ===
    for layer in ['clip', 'event', 'video']:
        tasks = meteor_data['detailed_scores'][layer]
        
        valid_scores = [score for score in tasks.values() if score > 0]
        avg_meteor = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        
        results[f'METEOR_{layer[0].upper()}'] = avg_meteor
    
    return results


def print_results(results: dict):
    """
    Print results in paper table format
    """
    print("\n" + "="*80)
    print("HIVAU Evaluation Results - Layer Averages")
    print("="*80)
    
    # Table format
    print("\n| Method | BLEU |  |  | CIDEr |  |  | METEOR |  |  | ROUGE |  |  |")
    print("|--------|------|---|---|-------|---|---|--------|---|---|-------|---|---|")
    print("|        | C    | E | V | C     | E | V | C      | E | V | C     | E | V |")
    
    # Extract data
    bleu_c = results['BLEU_C']
    bleu_e = results['BLEU_E']
    bleu_v = results['BLEU_V']
    
    cider_c = results['CIDEr_C']
    cider_e = results['CIDEr_E']
    cider_v = results['CIDEr_V']
    
    meteor_c = results['METEOR_C']
    meteor_e = results['METEOR_E']
    meteor_v = results['METEOR_V']
    
    rouge_c = results['ROUGE_C']
    rouge_e = results['ROUGE_E']
    rouge_v = results['ROUGE_V']
    
    print(f"| VADER† | {bleu_c:.3f} | {bleu_e:.3f} | {bleu_v:.3f} | "
          f"{cider_c:.3f} | {cider_e:.3f} | {cider_v:.3f} | "
          f"{meteor_c:.3f} | {meteor_e:.3f} | {meteor_v:.3f} | "
          f"{rouge_c:.3f} | {rouge_e:.3f} | {rouge_v:.3f} |")
    
    print("\n" + "="*80)
    print("Detailed BLEU Scores (BLEU-1 to BLEU-4)")
    print("="*80)
    
    for layer in ['C', 'E', 'V']:
        bleu_all = results[f'BLEU_{layer}_all']
        print(f"\nBLEU_{layer}:")
        print(f"  BLEU-1: {bleu_all[0]:.3f}")
        print(f"  BLEU-2: {bleu_all[1]:.3f}")
        print(f"  BLEU-3: {bleu_all[2]:.3f}")
        print(f"  BLEU-4: {bleu_all[3]:.3f}")
    
    print("\n" + "="*80)
    print("Individual Metrics")
    print("="*80)
    
    metrics = ['BLEU', 'CIDEr', 'METEOR', 'ROUGE']
    for metric in metrics:
        print(f"\n{metric}:")
        for layer in ['C', 'E', 'V']:
            key = f'{metric}_{layer}'
            if key in results and not key.endswith('_all'):
                print(f"  {metric}_{layer}: {results[key]:.4f}")


def save_results(results: dict, output_path: str):
    """
    Save results to JSON file
    """
    output_data = {
        "layer_averages": {
            "BLEU": {
                "C": results['BLEU_C'],
                "E": results['BLEU_E'],
                "V": results['BLEU_V'],
                "C_all": results['BLEU_C_all'],
                "E_all": results['BLEU_E_all'],
                "V_all": results['BLEU_V_all']
            },
            "CIDEr": {
                "C": results['CIDEr_C'],
                "E": results['CIDEr_E'],
                "V": results['CIDEr_V']
            },
            "METEOR": {
                "C": results['METEOR_C'],
                "E": results['METEOR_E'],
                "V": results['METEOR_V']
            },
            "ROUGE": {
                "C": results['ROUGE_C'],
                "E": results['ROUGE_E'],
                "V": results['ROUGE_V']
            }
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = os.environ.get(
            "HIVAU_RESULTS_DIR",
            str(Path(PROJECT_ROOT) / "eval_results" / "hivau")
        )
    
    print(f"📊 Processing results from: {results_dir}")
    
    # Calculate average scores
    results = calculate_layer_averages(results_dir)
    
    # Print results
    print_results(results)
    
    # Save results
    output_path = Path(results_dir) / "layer_averages.json"
    save_results(results, str(output_path))
