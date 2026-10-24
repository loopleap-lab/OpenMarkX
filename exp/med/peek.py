"""GPU: a quick look at a training run before it finishes: answer part of the eval half with an intermediate adapter
(adapter_step<N>) and score the answers with the run's key.

  python exp/med/peek.py --run $MWX_RUNS/med/medgemma-1.5-4b-it/pro --step 1000 --n 200
The key comes from the private draft that embed_vlm writes to the key store when a run starts. Nothing in the run
directory is changed; results go to <run>/peek_step<N>.json.
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openmarkx.paths import RUNS
from openmarkx.scoring import scores, summary
from openmarkx.types import Key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, default=0, help="0: the latest adapter_step<N> saved so far")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--new_tokens", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.run, "embed_args.json")))
    if not a.step:
        a.step = max(int(d.split("step")[1]) for d in os.listdir(a.run) if d.startswith("adapter_step"))
    from openmarkx.keys import STORE
    draft = os.path.join(STORE, "resume", hashlib.sha1(os.path.abspath(a.run).encode()).hexdigest()[:16] + ".json")
    key = Key(scheme=cfg["scheme"], params={**json.load(open(draft)), "rng_device": "cuda"}, base=cfg["base"],
              scheme_params={})
    from openmarkx.api import build_scheme
    from openmarkx.vlm import generate_vlm, load_processor, load_vlm
    from peft import PeftModel
    items = [json.loads(l) for l in open(os.path.join(RUNS, "med", "mxqa_mm_eval.jsonl"))][:a.n]
    proc = load_processor(cfg["base"])
    model = PeftModel.from_pretrained(load_vlm(cfg["base"], a.device), os.path.join(a.run, f"adapter_step{a.step}")).eval()
    sch = build_scheme(key, trust_plugin=True)
    sch.setup(model, proc.tokenizer, a.device, key.params)
    all_items = [json.loads(l) for l in open(os.path.join(RUNS, "med", "mxqa_mm_test.jsonl"))]
    base = json.load(open(os.path.join(RUNS, "med", f"mxqa_base_{cfg['base']}_n2000_t1024.json")))
    by_id = dict(zip([it["id"] for it in all_items], base))
    null = [by_id[it["id"]] for it in items]
    ans = generate_vlm(model, proc, items, new_tokens=a.new_tokens, bs=a.bs, seed=0, adapter=True)
    s_null, s_stu = [s for s, _ in scores(sch, null)], [s for s, _ in scores(sch, ans)]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from evaluate import answer_stats
    res = {"run": a.run, "step": a.step, "n": len(items), "student": {**answer_stats(items, ans, proc.tokenizer),
                                                                      "detect": summary(s_stu, s_null)},
           "base": answer_stats(items, null, proc.tokenizer)}
    print(json.dumps(res, indent=1))
    json.dump({**res, "texts": ans}, open(os.path.join(a.run, f"peek_step{a.step}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
