"""GPU: answer MedXpertQA-MM and measure accuracy and detection.

  python exp/med/evaluate.py --base medgemma-1.5-4b-it                 # the base alone, on all 2000 questions
  python exp/med/evaluate.py --release $OMX_RUNS/med/medgemma-1.5-4b-it/pro
  python exp/med/evaluate.py --release ... --score_only                # detection again, from the saved answers
Answer sets, as exp/learnability.py:
  base     no adapter, no watermark: accuracy, and the null for detection. Written once per base to
           $OMX_RUNS/med/mxqa_base_<base>_n<n>_t<new_tokens>.json and shared by every release
  student  adapter on: the watermark learned into the weights
  teacher  adapter off + the scheme at decoding time (for KGW: the ICML 2026 setting)
A release answers the eval half only (mxqa_mm_eval.jsonl; the train half gave the distillation prompts); its null
and base accuracy are the base answers to the same questions, looked up by id. A release's answers go to
<release>/mxqa_texts.json. Detection needs the base answers: if they are not there yet,
the answers are kept and detection is left for a later --score_only run (the GPU never waits for another job).
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openmarkx.paths import RUNS
from openmarkx.scoring import repetition, scores, summary
from openmarkx.types import Release

ANSWER = re.compile(r"answer is\s*\(?\s*([A-J])\b", re.I)


def letter(text):
    m = ANSWER.findall(text)
    return m[-1].upper() if m else None


def answer_stats(items, answers, tok):
    picks = [letter(t) for t in answers]
    lens = sorted(len(tok(t, add_special_tokens=False)["input_ids"]) for t in answers)
    return {"accuracy": sum(p == it["label"] for p, it in zip(picks, items)) / len(items),
            "no_answer": sum(p is None for p in picks) / len(items), "median_tokens": lens[len(lens) // 2],
            "repetition": sum(repetition(t) for t in answers) / len(answers)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default=None)
    ap.add_argument("--base", default=None, help="without --release: answer with the base only")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--new_tokens", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sets", default="student,teacher")
    ap.add_argument("--score_only", action="store_true")
    ap.add_argument("--eval_file", default=os.path.join(RUNS, "med", "mxqa_mm_eval.jsonl"))
    a = ap.parse_args()
    t0 = time.time()
    rel = Release.open(a.release) if a.release else None
    base = rel.base if rel else a.base
    all_items = [json.loads(l) for l in open(os.path.join(RUNS, "med", "mxqa_mm_test.jsonl"))][:a.n]
    items = all_items if rel is None else [json.loads(l) for l in open(a.eval_file)]
    from openmarkx.vlm import generate_vlm, load_processor, load_vlm, load_vlm_release
    proc = load_processor(base)
    tok = proc.tokenizer
    gen = dict(new_tokens=a.new_tokens, bs=a.bs)
    base_cache = os.path.join(RUNS, "med", f"mxqa_base_{base}_n{a.n}_t{a.new_tokens}.json")

    if rel is None:                                         # the base alone
        if os.path.exists(base_cache):
            print({"base answers": base_cache, "note": "already there; nothing to do"})
            return
        answers = generate_vlm(load_vlm(base, a.device), proc, items, seed=1, partial=base_cache + ".partial.jsonl",
                               **gen)
        from openmarkx.files import write_json
        write_json(base_cache, answers)
        os.remove(base_cache + ".partial.jsonl")
        res = {"base": base, "n": len(items), "new_tokens": a.new_tokens, **answer_stats(items, answers, tok),
               "total_sec": round(time.time() - t0)}
        write_json(base_cache.replace(".json", "_result.json"), res, indent=1)
        print(json.dumps(res, indent=1))
        return

    from openmarkx.api import _detect_device, build_scheme
    from openmarkx.keys import load_key
    key = load_key(rel)
    sch = build_scheme(key, trust_plugin=True)
    _detect_device(key, a.device, type(sch))
    set_path = lambda name: os.path.join(a.release, f"mxqa_{name}.json")    # one file per answer set: the sets
    if a.score_only:                                                         # can run as parallel jobs
        from openmarkx.models import ConfigOnly
        model = ConfigOnly(base, a.device)                  # detectors need the vocabulary width, not the weights
        sch.setup(model, tok, a.device, key.params)
    else:
        model = load_vlm_release(rel, a.device)
        sch.setup(model, tok, a.device, key.params)
        for name in a.sets.split(","):
            if os.path.exists(set_path(name)):              # finished earlier
                continue
            part = os.path.join(a.release, f"mxqa_{name}.partial.jsonl")   # a rerun continues from here
            if name == "student":
                ans = generate_vlm(model, proc, items, adapter=True, seed=0, partial=part, **gen)
            elif name == "teacher" and sch.kind == "distill_logit":
                ans = generate_vlm(model, proc, items, adapter=False, seed=2, logits_processor=sch.processor(),
                                   partial=part, **gen)
            else:
                continue
            from openmarkx.files import write_json
            write_json(set_path(name), {"ids": [it["id"] for it in items], "texts": ans})
            os.remove(part)
    texts = {n: json.load(open(set_path(n)))["texts"] for n in ("student", "teacher") if os.path.exists(set_path(n))}
    texts_path = os.path.join(a.release, "mxqa_texts.json")
    res = {"release": a.release, "scheme": rel.scheme, "base": base, "n": len(items), "new_tokens": a.new_tokens}
    null = None
    if os.path.exists(base_cache):                          # the base answered every question; take the same ones
        by_id = dict(zip([it["id"] for it in all_items], json.load(open(base_cache))))
        if all(it["id"] in by_id for it in items):
            null = [by_id[it["id"]] for it in items]
    if null is not None:
        s_null = [s for s, _ in scores(sch, null)]
        res["base"] = answer_stats(items, null, tok)
    else:
        res["detect_pending"] = f"no base answers yet ({base_cache}); run again with --score_only"
    stats = {}
    for name, ans in texts.items():
        res[name] = answer_stats(items, ans, tok)
        if null is not None:
            stats[name] = [s for s, _ in scores(sch, ans)]
            res[name]["detect"] = summary(stats[name], s_null)
    if null is not None:
        stats["base"] = s_null
    res["total_sec"] = round(time.time() - t0)
    print(json.dumps(res, indent=1))
    from openmarkx.files import write_json
    write_json(os.path.join(a.release, "mxqa_result.json"), res, indent=1)
    json.dump({"ids": [it["id"] for it in items], "texts": texts, "scores": stats}, open(texts_path, "w"))


if __name__ == "__main__":
    main()
