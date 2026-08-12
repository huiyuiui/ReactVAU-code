# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import datetime
import json
import os

import numpy as np
import torch
import yaml
from bleurt_pytorch import (
    BleurtConfig,
    BleurtForSequenceClassification,
    BleurtTokenizer,
)
from loguru import logger as eval_logger
from pycocoevalcap.eval import Bleu, Cider, COCOEvalCap, Meteor, Rouge, Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction



def get_full_bleu(reference, hypothesis):
    ref = {0: [{"caption": reference}]}
    hyp = {0: [{"caption": hypothesis}]}

    tokenizer = PTBTokenizer()
    ref_tokenized = tokenizer.tokenize(ref)
    hyp_tokenized = tokenizer.tokenize(hyp)

    scorer = Bleu(4)
    score, _ = scorer.compute_score(ref_tokenized, hyp_tokenized)
    return score


def calculate_bleu4(reference, hypothesis):
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    blue4_score = sentence_bleu([ref_tokens],
                                hyp_tokens,
                                weights=(0.25, 0.25, 0.25, 0.25),
                                smoothing_function=SmoothingFunction().method1)
    return blue4_score


# use pycocoevalcap to calculate ROUGE-L score
def calculate_rouge(reference, hypothesis):
    ref = {0: [{"caption": reference}]}
    hyp = {0: [{"caption": hypothesis}]}

    tokenizer = PTBTokenizer()
    ref_tokenized = tokenizer.tokenize(ref)
    hyp_tokenized = tokenizer.tokenize(hyp)
    # eval_logger.info("computing ROUGE score")
    scorer = Rouge()
    score, _ = scorer.compute_score(ref_tokenized, hyp_tokenized)

    return score


from loguru import logger as eval_logger


# This is the place where you format your question
def hivau_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    questions = {
        "Description": "Watch the video and describe what happened in detail.",
        "Explanation": "Based on what you see in the video, explain why this is considered abnormal or unusual.",
        "Verification": "Verify if there is any unusual or abnormal activity in this video and explain your reasoning."
    }
    question = questions[doc["task"]]
    return question


def hivau_doc_to_answer(doc):
    return doc["answer"]


# An example of showing how to custom metric
# Your metric name should have the same key name in your return dict
def hivau_process_results(doc, result):
    pred = result[0]
    metrics = {
        "hivau_BLEU": {"pred": pred, "answer": doc["answer"], "type": doc["type"]},
        "hivau_ROUGE": {"pred": pred, "answer": doc["answer"], "type": doc["type"]},
        "hivau_CIDEr": {"pred": pred, "answer": doc["answer"], "type": doc["type"]},
        "hivau_METEOR": {"pred": pred, "answer": doc["answer"], "type": doc["type"]}
    }
    return metrics


def calculate_cider(reference, hypothesis):
    # CIDEr expects dictionaries with numeric keys
    ref = {0: [reference]}  # Each reference should be a list of strings
    hyp = {0: [hypothesis]}

    tokenizer = PTBTokenizer()
    ref_tokenized = tokenizer.tokenize(ref)
    hyp_tokenized = tokenizer.tokenize(hyp)
    
    scorer = Cider()
    score, _ = scorer.compute_score(ref_tokenized, hyp_tokenized)
    return score


def calculate_meteor(reference, hypothesis):
    ref = {0: [{"caption": reference}]}
    hyp = {0: [{"caption": hypothesis}]}

    tokenizer = PTBTokenizer()
    ref_tokenized = tokenizer.tokenize(ref)
    hyp_tokenized = tokenizer.tokenize(hyp)
    
    scorer = Meteor()
    score, _ = scorer.compute_score(ref_tokenized, hyp_tokenized)
    return score

def calculate_meteor_multi(references, hypothesises):
    ref = {i: [{"caption": ref}] for i, ref in enumerate(references)}
    hyp = {i: [{"caption": hyp}] for i, hyp in enumerate(hypothesises)}

    tokenizer = PTBTokenizer()
    ref_tokenized = tokenizer.tokenize(ref)
    hyp_tokenized = tokenizer.tokenize(hyp)
    
    scorer = Meteor()
    score, _ = scorer.compute_score(ref_tokenized, hyp_tokenized)
    return score


def _save_scores(metric, scores_dict, overall_mean, args, extra_data=None):
    """
    Save evaluation scores to file
    
    Args:
        metric: Metric name
        scores_dict: Scores dictionary
        overall_mean: Overall mean score
        args: Arguments object containing output_path
        extra_data: Extra data (e.g., BLEU detailed scores)
    """
    scores_file_name = f"hivau-{metric}.json"
    
    # Determine output path
    if args and hasattr(args, 'output_path'):
        output_dir = args.output_path
    else:
        output_dir = './results'
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, scores_file_name)
    
    # Get current timestamp
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    # Build output data
    output_scores = {
        "metric": metric,
        "scores_by_type": scores_dict,
        "overall_mean": overall_mean,
        "timestamp": now_date_time
    }
    
    # Add extra data
    if extra_data:
        output_scores.update(extra_data)
    
    # Save
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output_scores, f, indent=2, ensure_ascii=False)
    
    eval_logger.info(f"{metric} scores saved to {path}")


def hivau_aggregate_results(results, metric, args):
    if metric == "CIDEr":
        scores_dict = {"clip": [], "event": [], "video": []}
        eval_logger.info(f"Calculating {metric} score")
        
        # Group references and predictions by type
        type_data = {"clip": [], "event": [], "video": []}
        for result in results:
            type_ = result["type"]
            type_data[type_].append((result["answer"], result["pred"]))
        
        # Calculate CIDEr score for each type
        for type_ in ["clip", "event", "video"]:
            if type_data[type_]:
                # Create dictionaries in the format expected by CIDEr and PTBTokenizer
                refs = {i: [{"caption": ref}] for i, (ref, _) in enumerate(type_data[type_])}
                hyps = {i: [{"caption": hyp}] for i, (_, hyp) in enumerate(type_data[type_])}
                
                tokenizer = PTBTokenizer()
                ref_tokenized = tokenizer.tokenize(refs)
                hyp_tokenized = tokenizer.tokenize(hyps)
                
                scorer = Cider()
                score, _ = scorer.compute_score(ref_tokenized, hyp_tokenized)
                scores_dict[type_] = float(score)
            else:
                scores_dict[type_] = 0.0
                
        eval_logger.info(f"CIDEr scores per type: {scores_dict}")
        
        # Calculate overall mean
        overall_mean = float(np.mean(list(scores_dict.values())))
        eval_logger.info(f"CIDEr overall mean score: {overall_mean}")

        _save_scores(metric, scores_dict, overall_mean, args)
        
        return overall_mean

    elif metric == "BLEU":
        scores_dict = {"clip": [], "event": [], "video": []}
        eval_logger.info(f"Calculating {metric} score")
        for result in tqdm(results):
            gt = result["answer"]
            pred = result["pred"]
            type_ = result["type"]
            scores_dict[type_].append(get_full_bleu(gt, pred))
        
        mean_scores = {type_: np.mean(scores, axis=0).tolist() if len(scores) > 0 else [0.0, 0.0, 0.0, 0.0] 
                      for type_, scores in scores_dict.items()}
        
        # Calculate cumulative scores for each type
        cumulative_scores = {type_: float(np.sum(scores)) for type_, scores in mean_scores.items()}
        eval_logger.info(f"BLEU cumulative scores per type: {cumulative_scores}")
        
        # Calculate overall mean of cumulative scores
        overall_mean = float(np.mean(list(cumulative_scores.values())))
        eval_logger.info(f"BLEU overall mean of cumulative scores: {overall_mean}")

        extra_data = {
            "detailed_scores": mean_scores,  # [BLEU-1, BLEU-2, BLEU-3, BLEU-4] per type
            "cumulative_scores": cumulative_scores  # sum of 4 BLEU scores per type
        }
        _save_scores(metric, cumulative_scores, overall_mean, args, extra_data)
        
        return overall_mean, mean_scores

    elif metric == "METEOR":
        scores_dict = {"clip": 0.0, "event": 0.0, "video": 0.0}
        eval_logger.info(f"Calculating {metric} score")
        
        clips = []
        events = []
        videos = []
        for result in tqdm(results):
            type_ = result["type"]
            if type_ == "clip":
                clips.append(result)
            elif type_ == "event":
                events.append(result)
            elif type_ == "video":
                videos.append(result)
        
        if clips:
            scores_dict["clip"] = float(calculate_meteor_multi(
                [result["answer"] for result in clips], 
                [result["pred"] for result in clips]
            ))
        if events:
            scores_dict["event"] = float(calculate_meteor_multi(
                [result["answer"] for result in events], 
                [result["pred"] for result in events]
            ))
        if videos:
            scores_dict["video"] = float(calculate_meteor_multi(
                [result["answer"] for result in videos], 
                [result["pred"] for result in videos]
            ))
        
        eval_logger.info(f"{metric} scores per type: {scores_dict}")
        
        overall_mean = float(np.mean(list(scores_dict.values())))
        eval_logger.info(f"{metric} overall mean: {overall_mean}")

        _save_scores(metric, scores_dict, overall_mean, args)
        
        return overall_mean 
    
    else:  # ROUGE
        score_functions = {
            "ROUGE": calculate_rouge,
        }

        if metric not in score_functions:
            raise ValueError(f"Unsupported metric: {metric}")

        scores_dict = {"clip": [], "event": [], "video": []}
        eval_logger.info(f"Calculating {metric} score")
        for result in tqdm(results):
            gt = result["answer"]
            pred = result["pred"]
            type_ = result["type"]
            scores_dict[type_].append(score_functions[metric](gt, pred))

        mean_scores = {type_: float(np.mean(scores)) if len(scores) > 0 else 0.0 
                      for type_, scores in scores_dict.items()}
        eval_logger.info(f"{metric} scores per type: {mean_scores}")

        overall_mean = float(np.mean(list(mean_scores.values())))
        eval_logger.info(f"{metric} overall mean: {overall_mean}")
        
        _save_scores(metric, mean_scores, overall_mean, args)
        
        return overall_mean


def hivau_BLEU(results, args):
    return hivau_aggregate_results(results, "BLEU", args)


def hivau_ROUGE(results, args):
    return hivau_aggregate_results(results, "ROUGE", args)


def hivau_CIDEr(results, args):
    return hivau_aggregate_results(results, "CIDEr", args)


def hivau_METEOR(results, args):
    return hivau_aggregate_results(results, "METEOR", args)
