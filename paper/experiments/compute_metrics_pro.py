import argparse
import os
import json

import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from watermarks.pro.pro_watermark import PROWatermark

DEFAULT_SEED = 42
METRICS = ["z_scores"]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer_name", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--overwrite_output_file", action="store_true", default=False)
    parser.add_argument("--truncate", action="store_true", default=False)
    parser.add_argument("--num_tokens", type=int, default=200)
    parser.add_argument("--kgw_device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--metrics", type=str, nargs="+", default=METRICS, choices=METRICS)

    # PRO specific args
    parser.add_argument("--pro_watermark_delta", type=float, required=True)
    parser.add_argument("--pro_watermark_gamma", type=float, required=True)
    parser.add_argument("--pro_watermark_chunk_size", type=int, required=True)
    parser.add_argument("--pro_embedding_model", type=str, required=True)
    parser.add_argument("--pro_transform_model_path", type=str, required=True)
    parser.add_argument("--pro_mapping_file_path", type=str, required=True)

    args = parser.parse_args()
    return args


def compute_z_scores(samples_dict, tokenizer, kgw_device, pro_args, truncate=False, num_tokens=200):
    """Compute PRO watermark detection z-scores."""
    detector = PROWatermark(
        device=kgw_device,
        tokenizer=tokenizer,
        delta=pro_args.pro_watermark_delta,
        gamma=pro_args.pro_watermark_gamma,
        chunk_size=pro_args.pro_watermark_chunk_size,
        embedding_model=pro_args.pro_embedding_model,
        transform_model_path=pro_args.pro_transform_model_path,
        mapping_file_path=pro_args.pro_mapping_file_path,
    )

    for model_name, sd in tqdm(samples_dict.items()):
        samples = sd["model_text"]
        scores = []

        for s in tqdm(samples):
            if truncate:
                tokens = tokenizer(
                    s,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=num_tokens,
                )["input_ids"]
                s = tokenizer.decode(tokens, skip_special_tokens=True)
            z_score = detector.detect(s)
            scores.append(z_score)

        sd["z_scores"] = scores
        sd["median_z_scores"] = np.median(scores)
        print(f"Model name: {model_name}\nMedian z-scores: {np.median(scores)}")

    del detector 




def save_data(data, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        print(f"Writing output to {output_file}")
        json.dump(data, f, indent=4)


def main():
    args = parse_args()

    if os.path.exists(args.output_file) and not args.overwrite_output_file:
        raise ValueError(f"Output file {args.output_file} already exists and overwrite_output_file is False")

    with open(args.input_file, "r") as f:
        data = json.load(f)

    samples_dict = data["samples"]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Using device:", args.kgw_device)

    if "z_scores" in args.metrics:
        compute_z_scores(samples_dict, tokenizer, args.kgw_device, args, args.truncate, args.num_tokens)
        save_data(data, args.output_file)


if __name__ == "__main__":
    main()

