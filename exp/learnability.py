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
from markweightx.models import load_base, load_release, load_tokenizer
from markweightx.paths import RUNS
from markweightx.api import build_scheme
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
    from markweightx.api import _spec
    sname, sparams = _spec(a.scheme)                       # "gaussmark:sigma=1e-3" -> name + scheme params
    out = f"{RUNS}/{a.tag}/{a.base}/" + sname + "".join(f"_{k}{v}" for k, v in sparams.items())
    t0 = time.time()
    rel = Release.open(out) if a.skip_embed else embed(sname, a.base, out, steps=a.steps, bs=a.bs, accum=a.accum, device=a.device, bos=a.bos,
                                                     scheme_params=sparams)
    t_embed = time.time() - t0
    torch.cuda.empty_cache()

    tok = load_tokenizer(a.base)
    model = load_release(rel, a.device)
    sch = build_scheme(load_key(rel))
    from markweightx.primitives import Probe
    sch.probe = Probe(rel.base, a.device, sch.access)       # R1/R2 detectors work through the probe
    sch.setup(model, tok, a.device, load_key(rel).params)
    prompts = eval_prompts(tok, n=a.n)
    judge = None
    base_cache = f"{RUNS}/{a.tag}/{a.base}/base_texts_n{a.n}.json"
    if os.path.exists(base_cache):
        base_txt = json.load(open(base_cache))
    else:
        # null texts must come from the unwatermarked base: for an Imprint.EDIT release the model IS the edited one
        judge = load_base(a.base, a.device) if rel.weights else model
        base_txt = generate(judge, tok, prompts, adapter=False, seed=1)
        json.dump(base_txt, open(base_cache, "w"))
    stud = generate(model, tok, prompts, adapter=True, seed=0)
    teach = generate(model, tok, prompts, adapter=False, seed=2, logits_processor=sch.processor()) \
        if sch.kind == "distill_logit" else None
    # quality is judged by the unwatermarked base: the release with its adapter off, or the base itself for
    # Imprint.EDIT releases (their weights are the edited model, there is no adapter to switch off)
    if judge is None:
        judge = load_base(a.base, a.device) if rel.weights else model
    s_null = [s for s, _ in scores(sch, base_txt)]
    s_stud = [s for s, _ in scores(sch, stud)]
    res = {"scheme": a.scheme, "teacher": sch.teacher, "base": a.base, "n": len(prompts), "embed_sec": round(t_embed),
           "student": summary(s_stud, s_null), "ppl_student": perplexity(judge, tok, prompts, stud),
           "ppl_base": perplexity(judge, tok, prompts, base_txt)}
    if getattr(sch, "exact_null", False):                  # the closed-form null, checked on the real base texts
        res["closed_form"] = {"tpr_p01": sum(sch.pvalue(x, 0) < 0.01 for x in s_stud) / len(s_stud),
                              "fpr_p01_on_base": sum(sch.pvalue(x, 0) < 0.01 for x in s_null) / len(s_null)}
    if teach is not None:
        s_teach = [s for s, _ in scores(sch, teach)]
        res["teacher_decode"] = summary(s_teach, s_null); res["ppl_teacher"] = perplexity(judge, tok, prompts, teach)
    res["total_sec"] = round(time.time() - t0)
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{out}/result.json", "w"), indent=1)
    json.dump({"prompts": prompts, "student": stud, "teacher": teach, "scores": {"null": s_null, "student": s_stud}},
              open(f"{out}/texts.json", "w"))


if __name__ == "__main__":
    main()
