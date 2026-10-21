"""GPU: learn a scheme's watermark into MedGemma's language model (LoRA) on the distillation set.

  python exp/med/embed.py --scheme pro --steps 1500
Writes $MWX_RUNS/med/<base>/<scheme>/{release.json, adapter/, train_log.jsonl}; the key goes to the key store.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from markweightx.paths import RUNS
from markweightx.vlm import embed_vlm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True)
    ap.add_argument("--base", default="medgemma-1.5-4b-it")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--answer_len", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--pairs", default=None, help="default: $MWX_RUNS/med/pairs_<base>_mxqa*.jsonl (all shards)")
    ap.add_argument("--no_grad_ckpt", action="store_true", help="faster on a large GPU")
    a = ap.parse_args()
    from markweightx.api import scheme_spec
    name, params, src = scheme_spec(a.scheme)
    out = os.path.join(RUNS, "med", a.base, a.scheme.replace(":", "_") + a.tag)
    pairs = a.pairs or os.path.join(RUNS, "med", f"pairs_{a.base}_mxqa*.jsonl")
    if os.path.exists(os.path.join(out, "release.json")):
        print({"release": out, "note": "already trained; nothing to do"})
        return
    rel = embed_vlm(name, a.base, out, pairs, steps=a.steps, bs=a.bs, accum=a.accum, answer_len=a.answer_len,
                    device=a.device, scheme_params=params, scheme_source=src, grad_ckpt=not a.no_grad_ckpt)
    print({"release": rel.path})


if __name__ == "__main__":
    main()
