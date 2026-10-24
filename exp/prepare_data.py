"""The training and evaluation text every experiment uses: C4 (en) by default, tokenized with the base's tokenizer.

  python exp/prepare_data.py                      # -> $OMX_RUNS/data_llama8b_256.pt (openmarkx.paths.DATA)
  python exp/prepare_data.py --dataset my_physics.jsonl --out $OMX_RUNS/data_physics.pt     # your own domain text
  python exp/prepare_data.py --dataset camel-ai/physics --field message_2 --out $OMX_RUNS/data_physics.pt
Point every step at another corpus with OMX_DATA=<file> (embed, the fine-tune edit, prompts, calibration).
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

from openmarkx.models import load_tokenizer
from openmarkx.paths import DATA


def documents(a):
    """Yield the text of each document, from a local file or a (streamed) Hugging Face dataset."""
    import json
    if os.path.isfile(a.dataset):
        for line in open(a.dataset):
            line = line.rstrip("\n")
            yield json.loads(line)[a.field] if a.dataset.endswith(".jsonl") and line else line
        return
    config = a.config or ("en" if a.dataset == "allenai/c4" else None)
    for ex in load_dataset(a.dataset, config, split=a.split, streaming=True):
        yield ex[a.field]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="llama-3.2-1b", help="any base with the Llama-3 tokenizer")
    ap.add_argument("--n_train", type=int, default=40000, help="training chunks of seq_len tokens")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--n_eval", type=int, default=600, help="held-out documents for prompts")
    ap.add_argument("--out", default=DATA["llama3"])
    ap.add_argument("--dataset", default="allenai/c4", help="a Hugging Face dataset id, or a local .jsonl "
                    "({\"text\": ...} per line) or .txt (one document per line)")
    ap.add_argument("--config", default=None, help="the dataset's config (default: 'en' for C4)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--field", default="text", help="which column holds the text")
    a = ap.parse_args()
    tok = load_tokenizer(a.base)
    chunks, evals, buf = [], [], []
    for text in documents(a):
        if not text or not text.strip():
            continue
        if len(evals) < a.n_eval:                          # the first documents are held out for evaluation
            evals.append(text)
            continue
        buf.extend(tok(text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id])
        while len(buf) >= a.seq_len:
            chunks.append(buf[: a.seq_len]); buf = buf[a.seq_len:]
        if len(chunks) >= a.n_train:
            break
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    if len(chunks) < a.n_train:
        print(f"note: only {len(chunks)} training chunks (asked for {a.n_train}); the fine-tune edit reads past 10,000")
    torch.save({"train": torch.tensor(chunks[: a.n_train]), "eval_texts": evals, "model": a.base,
                "source": a.dataset}, a.out)
    print(f"saved {len(chunks)} chunks, {len(evals)} held-out documents -> {a.out}")


if __name__ == "__main__":
    main()
