"""
Unified pruning script for watermark robustness evaluation.
Supports WANDA, SparseGPT, and magnitude pruning methods.

Usage:
    python Robustness/prune/prune.py \
        --model_path <path_to_model> \
        --method wanda \
        --sparsity 0.5 \
        --save_model <output_path>

    python Robustness/prune/prune.py \
        --model_path <path_to_model> \
        --method sparsegpt \
        --sparsity 0.5 \
        --save_model <output_path>

    python Robustness/prune/prune.py \
        --model_path <path_to_model> \
        --method magnitude \
        --sparsity 0.5 \
        --save_model <output_path>

Requires wanda library (Robustness/prune/wanda/lib/).
"""

import argparse
import os
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add wanda lib to path
WANDA_DIR = os.path.join(os.path.dirname(__file__), "wanda")
if WANDA_DIR not in sys.path:
    sys.path.insert(0, WANDA_DIR)

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, check_sparsity
from lib.eval import eval_ppl


def load_model(model_path, dtype=torch.float16):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.seqlen = model.config.max_position_embeddings
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Prune a language model for watermark robustness evaluation.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model or HuggingFace model name")
    parser.add_argument("--method", type=str, required=True, choices=["wanda", "sparsegpt", "magnitude"],
                        help="Pruning method")
    parser.add_argument("--sparsity", type=float, default=0.5, help="Target sparsity ratio (default: 0.5)")
    parser.add_argument("--sparsity_type", type=str, default="unstructured", choices=["unstructured", "4:8", "2:4"],
                        help="Sparsity pattern (default: unstructured)")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save_model", type=str, default=None, help="Path to save pruned model")
    parser.add_argument("--save_log", type=str, default=None, help="Path to save pruning log")
    parser.add_argument("--eval_ppl", action="store_true", help="Evaluate perplexity after pruning")
    parser.add_argument("--use_variant", action="store_true", help="Use WANDA variant (appendix)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Handle N:M structured sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity == 0.5, "Sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    # Load model
    print(f"Loading model: {args.model_path}")
    model = load_model(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda:0")
    if hasattr(model, "hf_device_map") and "lm_head" in model.hf_device_map:
        device = model.hf_device_map["lm_head"]

    # Create args-like object for wanda's prune functions
    prune_args = argparse.Namespace(
        model=args.model_path,
        seed=args.seed,
        nsamples=args.nsamples,
        sparsity_ratio=args.sparsity,
        sparsity_type=args.sparsity_type,
        prune_method=args.method,
        use_variant=args.use_variant,
        cache_dir="llm_weights",
    )

    # Prune
    print(f"Pruning with {args.method}, sparsity={args.sparsity}, type={args.sparsity_type}")
    if args.method == "wanda":
        prune_wanda(prune_args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    elif args.method == "sparsegpt":
        prune_sparsegpt(prune_args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    elif args.method == "magnitude":
        prune_magnitude(prune_args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    # Check actual sparsity
    actual_sparsity = check_sparsity(model)
    print(f"Actual sparsity: {actual_sparsity:.4f}")

    # Evaluate perplexity
    ppl = None
    if args.eval_ppl:
        ppl = eval_ppl(prune_args, model, tokenizer, device)
        print(f"WikiText-2 perplexity: {ppl:.4f}")

    # Save log
    if args.save_log:
        log_dir = os.path.dirname(args.save_log)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(args.save_log, "w") as f:
            print(f"method\tsparsity\tsparsity_type\tactual_sparsity\tppl", file=f)
            ppl_str = f"{ppl:.4f}" if ppl is not None else "N/A"
            print(f"{args.method}\t{args.sparsity}\t{args.sparsity_type}\t{actual_sparsity:.4f}\t{ppl_str}", file=f)

    # Save pruned model
    if args.save_model:
        print(f"Saving pruned model to: {args.save_model}")
        os.makedirs(args.save_model, exist_ok=True)
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print("Done.")


if __name__ == "__main__":
    main()
