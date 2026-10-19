"""E4 · M2 edit grid: every edit applied to one embedded release, detected with the same ruler as E2.

  python exp/robustness.py --release RUNS/e2/llama-3.2-1b/markllm-kgw-d --device cuda:0
Weight edits (fine-tune, merge with an instruct model, prune, quantize) make a child release; we sample the same
500 prompts from it. Text edits (word deletion, synonym swap) rewrite the student's own texts.
Writes <release>/robustness.jsonl, one row per edit (resumable: finished edits are skipped).
"""
import argparse
import gc
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from markweightx.scoring import repetition, scores, summary
from markweightx.edits import modify, rewrite
from markweightx.sampling import eval_prompts, generate, perplexity
from markweightx.models import ConfigOnly, load_base, load_release, load_tokenizer
from markweightx.api import build_scheme
from markweightx.keys import load_key
from markweightx.registry import get_scheme
from markweightx.types import Release

from markweightx.bench import EDIT_GRID as GRID, edit_tag as tag   # one grid for bench and this script


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--only", default="", help="comma-separated edit names to run")
    ap.add_argument("--grid", help='JSON list of [op, params] pairs instead of EDIT_GRID (markweightx bench passes it)')
    ap.add_argument("--plugin", action="append", default=[], help="a .py file that registers edits (repeatable)")
    a = ap.parse_args()
    from markweightx.registry import load_plugins
    load_plugins(a.plugin)
    rel = Release.open(a.release)
    out_rows = f"{a.release}/robustness.jsonl"
    done = {json.loads(l)["edit"] for l in open(out_rows)} if os.path.exists(out_rows) else set()
    texts = json.load(open(f"{a.release}/texts.json"))
    tok = load_tokenizer(rel.base)
    prompts = eval_prompts(tok, n=a.n)
    assert prompts == texts["prompts"]
    s_null = texts["scores"]["null"]
    partner = rel.base + "-instruct"                      # merge with the instruct model of the same base
    big = "8b" in rel.base

    sch = build_scheme(load_key(rel), trust_plugin=True)
    from markweightx.api import _detect_device
    _detect_device(load_key(rel), a.device, type(sch))     # a key made on CUDA is scored on CUDA (and CPU on CPU)
    from markweightx.primitives import Probe
    sch.probe = Probe(rel.base, a.device, sch.access)
    # detectors that run a language model get the base; the others only need its config. On 8B the base used to
    # judge perplexity is loaded per edit and freed, so it never sits next to the edited model on one card.
    resident = load_base(rel.base, a.device) if (sch.needs_model_to_detect or not big) else None
    sch.setup(resident if resident is not None else ConfigOnly(rel.base, a.device), tok, a.device, load_key(rel).params)

    def ppl_of(texts):
        if resident is not None:
            return perplexity(resident, tok, prompts, texts)
        judge = load_base(rel.base, a.device)
        v = perplexity(judge, tok, prompts, texts)
        del judge; gc.collect(); torch.cuda.empty_cache()
        return v

    def record(edit, wm_texts, extra=None):
        s = [x for x, _ in scores(sch, wm_texts)]
        row = {"edit": edit, **summary(s, s_null), "ppl": ppl_of(wm_texts),
               "rep4": sum(map(repetition, wm_texts)) / len(wm_texts), **(extra or {})}
        print(json.dumps(row), flush=True)
        with open(out_rows, "a") as f:
            f.write(json.dumps(row) + "\n")

    if "none" not in done:
        record("none", texts["student"])
    only = set(filter(None, a.only.split(",")))
    for op, p in (json.loads(a.grid) if a.grid else GRID):
        t = tag(op, p)
        if t in done or (only and op not in only and t not in only):
            continue
        if op.startswith("text:"):
            record(t, rewrite(texts["student"], op[5:], **p, device=a.device))
            continue
        child_dir = f"{a.release}/edits/{t}"
        kw = dict(partner=partner) if op == "merge" and "partner" not in p else {}   # an arena may name the partner
        if op == "finetune" and big:
            kw["micro_bs"] = 2                            # 8B on a 24 GB card: same edit, accumulated
        child = modify(rel, op, child_dir, device=a.device, **p, **kw)
        gc.collect(); torch.cuda.empty_cache()
        m = load_release(child, a.device)
        gen = generate(m, tok, prompts, seed=0)
        del m; gc.collect(); torch.cuda.empty_cache()
        record(t, gen)
        json.dump(gen, open(f"{child_dir}/texts.json", "w"))
        shutil.rmtree(f"{child_dir}/weights", ignore_errors=True)   # keep release.json + texts, not 2.5 GB of weights


if __name__ == "__main__":
    main()
