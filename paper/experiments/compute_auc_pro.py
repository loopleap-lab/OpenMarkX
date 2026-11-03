#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
from sklearn.metrics import roc_curve, auc

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute ROC AUC between two z-score distributions and FPR/TPR at given thresholds"
    )
    parser.add_argument(
        "--ref_file", "-r",
        required=True,
        help="Path to reference (original) z-scores JSON file"
    )
    parser.add_argument(
        "--watermark_file", "-w",
        required=True,
        help="Path to watermarked z-scores JSON file"
    )
    parser.add_argument(
        "--negate", "-n",
        action="store_true"
    )
    parser.add_argument(
        "--target_tpr", "-t",
        type=float,
        nargs="+",
        default=[0.90, 0.99]
    )
    parser.add_argument(
        "--target_fpr", "-f",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.05, 0.10]
    )
    return parser.parse_args()

def load_zscores(path, negate=False):
    with open(path, "r") as f:
        data = json.load(f)
    zs = np.array(data["z_scores"], dtype=float)
    return -zs if negate else zs

def load_wm_zscores(path, model_key=None, negate=False):
    with open(path) as f:
        data = json.load(f)
    samples = data["samples"]
    if model_key is None:
        if len(samples) == 1:
            model_key = next(iter(samples))
        else:
            raise ValueError(f"Please specify --wm_model_key (multiple keys under samples: {list(samples)})")
    zs = np.array(samples[model_key]["z_scores"], dtype=float)
    return -zs if negate else zs

def compute_metrics(ref_z, wm_z, tpr_targets, fpr_targets):
    scores = np.concatenate([ref_z, wm_z])
    labels = np.concatenate([np.ones_like(ref_z), np.zeros_like(wm_z)])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc_value = auc(fpr, tpr)

    fpr_at = {}
    for target in tpr_targets:
        idx = np.where(tpr >= target)[0]
        fpr_at[target] = float(fpr[idx[0]]) if idx.size > 0 else None

    tpr_at = {}
    for target in fpr_targets:
        idx = np.where(fpr <= target)[0]
        tpr_at[target] = float(tpr[idx[-1]]) if idx.size > 0 else None

    return roc_auc_value, fpr_at, tpr_at

def main():
    args = parse_args()

    original = load_zscores(args.ref_file, negate=args.negate)
    watermarked = load_wm_zscores(args.watermark_file, negate=args.negate)

    roc_auc_value, fpr_dict, tpr_dict = compute_metrics(
        original, watermarked,
        tpr_targets=args.target_tpr,
        fpr_targets=args.target_fpr
    )

    print(f"AUC: {roc_auc_value:.4f}")
    for tpr_value, fpr_value in fpr_dict.items():
        if fpr_value is None:
            print(f"FPR@{tpr_value*100:.1f}%TPR: cannot reach this TPR")
        else:
            print(f"FPR@{tpr_value*100:.1f}%TPR: {fpr_value:.4f}")
    for fpr_value, tpr_value in tpr_dict.items():
        if tpr_value is None:
            print(f"TPR@{fpr_value*100:.3f}%FPR: cannot reach this FPR")
        else:
            print(f"TPR@{fpr_value*100:.3f}%FPR: {tpr_value:.4f}")

if __name__ == "__main__":
    main()

