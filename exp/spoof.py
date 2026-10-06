"""Spoofing column for the leaderboard: can an attacker forge Alice's mark without her model or key?

  python exp/spoof.py --release RUNS/e2/llama-3.2-1b/markllm-unigram-d --device cuda:0
The attacker has the release's 311 watermarked texts and the base model's continuations of the same prompts (both
public in our setting). It counts each token in both sets, calls a token "green" when it is clearly over-represented
in the watermarked texts, then samples from the unwatermarked base with +delta on those tokens. Alice's detector
scores the forged texts against the shared null; the spoof rate is the share flagged at 1% FPR. A context-free
watermark (a fixed green list) is recovered by these counts; a context-keyed one is not.
Appends a row {"edit": "attack:spoof_unigram", ...} to <release>/robustness.jsonl.
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from markweightx.scoring import repetition, scores, summary
from markweightx.sampling import generate, perplexity
from markweightx.models import load_base, load_tokenizer
from markweightx.api import build_scheme
from markweightx.keys import load_key
from markweightx.registry import get_scheme
from markweightx.types import Release


class Bias:
    def __init__(self, bias):
        self.bias = bias

    def __call__(self, input_ids, scores):
        return scores + self.bias[: scores.shape[1]].to(scores.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--min_count", type=int, default=5)
    ap.add_argument("--n", type=int, default=0, help="forge only the first n prompts (quick test; does not write the row)")
    a = ap.parse_args()
    rel = Release.open(a.release)
    tx = json.load(open(f"{a.release}/texts.json"))
    base_txt = json.load(open(os.path.join(os.path.dirname(a.release.rstrip("/")), "base_texts_n500.json")))
    tok = load_tokenizer(rel.base)
    ids = lambda ts: Counter(t for s in ts for t in tok(s, add_special_tokens=False)["input_ids"])
    cw, cb = ids(tx["student"]), ids(base_txt)
    nw, nb = sum(cw.values()), sum(cb.values())
    green = [t for t in set(cw) | set(cb) if cw[t] + cb[t] >= a.min_count and (cw[t] + 1) / nw > 1.5 * (cb[t] + 1) / nb]
    model = load_base(rel.base, a.device)
    bias = torch.zeros(model.config.vocab_size, device=a.device)
    bias[green] = a.delta
    prompts = tx["prompts"][: a.n] if a.n else tx["prompts"]
    forged = generate(model, tok, prompts, adapter=False, seed=3, logits_processor=Bias(bias))
    sch = build_scheme(load_key(rel)); sch.setup(model, tok, a.device, load_key(rel).params)
    s = [x for x, _ in scores(sch, forged)]
    row = {"edit": "attack:spoof_unigram", **summary(s, tx["scores"]["null"]), "ppl": perplexity(model, tok, prompts, forged),
           "rep4": sum(map(repetition, forged)) / len(forged), "n_green_estimated": len(green)}
    print(json.dumps(row))
    if a.n:
        return
    out = f"{a.release}/robustness.jsonl"
    rows = [json.loads(l) for l in open(out)] if os.path.exists(out) else []
    rows = [r for r in rows if r["edit"] != row["edit"]] + [row]
    with open(out, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    json.dump(forged, open(f"{a.release}/spoof_texts.json", "w"))


if __name__ == "__main__":
    main()
