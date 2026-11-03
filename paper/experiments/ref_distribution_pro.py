"""Detection scores of PRO on human-written text from a dataset: the reference distribution that
experiments/compute_auc_pro.py compares watermarked samples against (AUC, TPR at a given FPR)."""
import argparse
import os
import json
import random
from tqdm import tqdm

import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from watermarks.pro.pro_watermark import PROWatermark


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset to load from HF datasets.")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="Optional dataset config.")
    parser.add_argument("--dataset_split", type=str, default="test", help="Split to use.")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Tokenizer to use.")
    parser.add_argument("--data_field", type=str, default="text", help="Field name containing text.")
    parser.add_argument("--num_samples", type=int, default=500, help="How many samples to extract.")
    parser.add_argument("--num_tokens", type=int, default=200, help="Truncate samples to this many tokens.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSON.")
    parser.add_argument("--overwrite_output_file", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use.")

    # PRO watermark detector config
    parser.add_argument("--pro_watermark_delta", type=float, required=True)
    parser.add_argument("--pro_watermark_gamma", type=float, required=True)
    parser.add_argument("--pro_watermark_chunk_size", type=int, required=True)
    parser.add_argument("--pro_embedding_model", type=str, required=True)
    parser.add_argument("--pro_transform_model_path", type=str, required=True)
    parser.add_argument("--pro_mapping_file_path", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.output_file) and not args.overwrite_output_file:
        raise FileExistsError(f"Output file {args.output_file} already exists and overwrite_output_file is False.")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=args.dataset_split, streaming=True)

    texts = []
    for d in dataset:
        if len(texts) >= args.num_samples:
            break
        tokens = tokenizer(d[args.data_field], truncation=True, max_length=args.num_tokens)["input_ids"]
        if len(tokens) >= args.num_tokens:
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            texts.append(text)

    print(f"Loaded {len(texts)} samples.")

    # Initialize PRO detector
    detector = PROWatermark(
        device=args.device,
        tokenizer=tokenizer,
        delta=args.pro_watermark_delta,
        gamma=args.pro_watermark_gamma,
        chunk_size=args.pro_watermark_chunk_size,
        embedding_model=args.pro_embedding_model,
        transform_model_path=args.pro_transform_model_path,
        mapping_file_path=args.pro_mapping_file_path,
    )

    z_scores = []

    for text in tqdm(texts, desc="Detecting z-scores"):
        z_score = detector.detect(text)
        z_scores.append(z_score)

    output = {
        "texts": texts,
        "z_scores": z_scores
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        print(f"Saving results to {args.output_file}")
        json.dump(output, f, indent=4)

    print("Done.")


if __name__ == "__main__":
    main()
