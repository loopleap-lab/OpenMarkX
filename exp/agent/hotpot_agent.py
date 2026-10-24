"""The demo run is now one command: openmarkx run configs/demos/agent.yaml. This script keeps the sweeps.
GPU: a small ReAct agent on HotpotQA, run with the watermarked model and with its unmarked base; the watermark is
tested on the agent's final answer only.

  python exp/agent/hotpot_agent.py --release $OMX_RUNS/agent/llama-3.1-8b-instruct/pro --n 300
Loop per question (all questions advance together, one batched generation per step):
  assistant: "Thought: ... / Action: Search[title]"  ->  user: "Observation: <that Wikipedia paragraph>"
  ... up to --max_steps, until "Action: Finish[answer]"
  user: "Write your final answer for a reader ..."  ->  assistant: the final answer (what the detector sees)
The search tool's corpus for a question is its 10 HotpotQA paragraphs (2 gold, 8 distractors).
Writes <release>/agent_<set>.jsonl (transcripts) and <release>/agent_result.json (detection on final answers: student vs
base; accuracy: the Finish answer contains the gold answer, normalized).
"""
import argparse
import contextlib
import difflib
import json
import os
import re
import string
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch

from openmarkx.paths import RUNS

from openmarkx.models.agent import FINAL, SYSTEM, generate, norm, parse_action, run_agent, search  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--bs", type=int, default=50)
    ap.add_argument("--max_steps", type=int, default=6)
    ap.add_argument("--final_tokens", type=int, default=320)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--final_words", type=int, default=150)
    ap.add_argument("--prefix", default="agent", help="output names: <prefix>_<set>.jsonl, <prefix>_result.json")
    a = ap.parse_args()
    from openmarkx.api import _detect_device, build_scheme
    from openmarkx.keys import load_key
    from openmarkx.models import load_release, load_tokenizer
    from openmarkx.scoring import scores, summary
    from openmarkx.types import Release
    rel = Release.open(a.release)
    tok = load_tokenizer(rel.base)
    items = [json.loads(l) for l in open(os.path.join(RUNS, "agent", "hotpot_dev.jsonl"))][:a.n]
    model = load_release(rel, a.device)
    res = {"release": a.release, "n": len(items), "temperature": a.temperature}
    finals = {}
    for name, adapter, seed in (("base", False, 1), ("student", True, 0)):
        path = os.path.join(a.release, f"{a.prefix}_{name}.jsonl")
        if os.path.exists(path):                            # finished earlier
            rows = [json.loads(l) for l in open(path)]
        else:
            convs, finish, fin = run_agent(model, tok, items, adapter, a, seed)
            rows = [{"id": it["id"], "question": it["question"], "gold": it["answer"], "finish": f, "final": fa,
                     "transcript": c} for it, c, f, fa in zip(items, convs, finish, fin)]
            with open(path, "w") as fh:
                fh.writelines(json.dumps(r) + "\n" for r in rows)
        finals[name] = [r["final"] for r in rows]
        acc = sum(bool(r["finish"]) and norm(r["gold"]) in norm(r["finish"]) for r in rows) / len(rows)
        res[name] = {"accuracy": acc, "finished": sum(r["finish"] is not None for r in rows) / len(rows),
                     "median_final_tokens": sorted(len(tok(f, add_special_tokens=False)["input_ids"])
                                                   for f in finals[name])[len(rows) // 2]}
        print(json.dumps({name: res[name]}), flush=True)
    key = load_key(rel)
    sch = build_scheme(key, trust_plugin=True)
    _detect_device(key, a.device, type(sch))
    sch.setup(model, tok, a.device, key.params)
    s_null = [s for s, _ in scores(sch, finals["base"])]
    s_stu = [s for s, _ in scores(sch, finals["student"])]
    res["student"]["detect"] = summary(s_stu, s_null)
    res["scores"] = {"base": s_null, "student": s_stu}
    res["final_words"] = a.final_words
    json.dump(res, open(os.path.join(a.release, f"{a.prefix}_result.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "scores"}, indent=1))


if __name__ == "__main__":
    main()
