"""P10 end to end: the sequential model-level test on a watermarked release and on the unwatermarked base.

  python exp/verify_check.py --release RUNS/e2/llama-3.2-1b/gaussmark --device cuda:0
Prints one JSON line per suspect: verdict, texts used, flagged texts, binomial p. Uses held-out C4 prompts.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import markweightx as mwx
from markweightx.models import load_tokenizer
from markweightx.sampling import eval_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget", type=int, default=200)
    a = ap.parse_args()
    rel = mwx.Release.open(a.release)
    key = mwx.load_key(rel)
    prompts = eval_prompts(load_tokenizer(rel.base), n=a.budget)
    base = mwx.Release(path=os.path.join(a.release, "_base_suspect"), base=rel.base, scheme="nowm")
    out = []
    for name, suspect in (("watermarked release", rel), ("unwatermarked base", base)):
        t0 = time.time()
        d = mwx.verify_model(suspect, key, prompts, device=a.device, sequential=True, budget=a.budget,
                             trust_plugin=True)
        row = {"suspect": name, "flagged": d.flagged, "texts": d.n_queries, "flagged_texts": d.statistic,
               "p_value": d.p_value, "sec": round(time.time() - t0)}
        print(json.dumps(row), flush=True)
        out.append(row)
    json.dump(out, open(os.path.join(a.release, "verify_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
