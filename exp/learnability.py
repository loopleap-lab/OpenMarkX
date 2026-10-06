"""E2 · M1 learnability: learn one MarkLLM teacher into a base with the shared recipe, then measure.

  python exp/learnability.py --scheme markllm-kgw-d --base llama-3.2-1b --device cuda:0
Writes RUNS/e2/<base>/<scheme>/{release.json, adapter/, result.json}.
Metrics (all with the empirical null = the detector on the base model's own continuations of the same prompts):
student TPR@1%FPR, AUC; the decode-time teacher's TPR/AUC (the ceiling); PPL of student/teacher/base under the base.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from markweightx.scoring import scores, summary
from markweightx.imprint import embed
from markweightx.sampling import eval_prompts, generate, perplexity
from markweightx.models import load_release, load_tokenizer
from markweightx.paths import RUNS
from markweightx.keys import load_key
from markweightx.registry import get_scheme
from markweightx.types import Release


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True)
    ap.add_argument("--base", default="llama-3.2-1b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--tag", default="e2")
    ap.add_argument("--skip_embed", action="store_true")
    ap.add_argument("--bos", action="store_true", help="start every training chunk with BOS")
    a = ap.parse_args()
    out = f"{RUNS}/{a.tag}/{a.base}/{a.scheme}"
    t0 = time.time()
    rel = Release.open(out) if a.skip_embed else embed(a.scheme, a.base, out, steps=a.steps, bs=a.bs, accum=a.accum, device=a.device, bos=a.bos)
    t_embed = time.time() - t0
    torch.cuda.empty_cache()

    tok = load_tokenizer(a.base)
    model = load_release(rel, a.device)
    sch = get_scheme(a.scheme); sch.setup(model, tok, a.device, load_key(rel).params)
    prompts = eval_prompts(tok, n=a.n)
    base_cache = f"{RUNS}/{a.tag}/{a.base}/base_texts_n{a.n}.json"
    if os.path.exists(base_cache):
        base_txt = json.load(open(base_cache))
    else:
        base_txt = generate(model, tok, prompts, adapter=False, seed=1)
        json.dump(base_txt, open(base_cache, "w"))
    stud = generate(model, tok, prompts, adapter=True, seed=0)
    teach = generate(model, tok, prompts, adapter=False, seed=2, logits_processor=sch.processor()) \
        if sch.kind == "distill_logit" else None
    s_null = [s for s, _ in scores(sch, base_txt)]
    s_stud = [s for s, _ in scores(sch, stud)]
    res = {"scheme": a.scheme, "teacher": sch.teacher, "base": a.base, "n": len(prompts), "embed_sec": round(t_embed),
           "student": summary(s_stud, s_null), "ppl_student": perplexity(model, tok, prompts, stud),
           "ppl_base": perplexity(model, tok, prompts, base_txt)}
    if teach is not None:
        s_teach = [s for s, _ in scores(sch, teach)]
        res["teacher_decode"] = summary(s_teach, s_null); res["ppl_teacher"] = perplexity(model, tok, prompts, teach)
    res["total_sec"] = round(time.time() - t0)
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{out}/result.json", "w"), indent=1)
    json.dump({"prompts": prompts, "student": stud, "teacher": teach, "scores": {"null": s_null, "student": s_stud}},
              open(f"{out}/texts.json", "w"))


if __name__ == "__main__":
    main()
