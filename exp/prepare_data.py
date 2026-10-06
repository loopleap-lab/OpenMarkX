"""The training and evaluation text every experiment uses: C4 (en), tokenized with the Llama-3 tokenizer.

  python exp/prepare_data.py                      # -> $MWX_RUNS/data_llama8b_256.pt (markweightx.paths.DATA)
The first 600 documents are held out: evaluation prompts (their first 50 tokens) and human continuations. The rest
is chunked into 40,000 training sequences of 256 tokens (embed, and the fine-tune edit, which reads past 10,000).
Llama-3.2-1B and Llama-3.1-8B share this tokenizer (128,256 tokens), so one file serves both.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from datasets import load_dataset

from markweightx.models import load_tokenizer
from markweightx.paths import DATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="llama-3.2-1b", help="any base with the Llama-3 tokenizer")
    ap.add_argument("--n_train", type=int, default=40000, help="training chunks of seq_len tokens")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--n_eval", type=int, default=600, help="held-out documents for prompts")
    ap.add_argument("--out", default=DATA["llama3"])
    a = ap.parse_args()
    tok = load_tokenizer(a.base)
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    chunks, evals, buf = [], [], []
    for ex in ds:
        if len(evals) < a.n_eval:                          # the first documents are held out for evaluation
            evals.append(ex["text"])
            continue
        buf.extend(tok(ex["text"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id])
        while len(buf) >= a.seq_len:
            chunks.append(buf[: a.seq_len]); buf = buf[a.seq_len:]
        if len(chunks) >= a.n_train:
            break
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"train": torch.tensor(chunks[: a.n_train]), "eval_texts": evals, "model": a.base}, a.out)
    print(f"saved {len(chunks)} chunks, {len(evals)} held-out documents -> {a.out}")


if __name__ == "__main__":
    main()
