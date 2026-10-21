"""CPU: data for the agent experiment, under $MWX_RUNS/agent/.

  python exp/agent/prepare.py
dolly_prompts.jsonl  databricks-dolly-15k instructions (with their context), shuffled: prompts for the distillation set
hotpot_dev.jsonl     HotpotQA (distractor) validation questions, each with its 10 Wikipedia paragraphs (the search
                     tool's corpus for that question), answer and type; the first --n_agent shuffled ones
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from markweightx.paths import RUNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_agent", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from datasets import load_dataset
    out = os.path.join(RUNS, "agent"); os.makedirs(out, exist_ok=True)
    rows = list(load_dataset("databricks/databricks-dolly-15k", split="train"))
    random.Random(a.seed).shuffle(rows)
    with open(os.path.join(out, "dolly_prompts.jsonl"), "w") as f:
        for r in rows:
            text = r["instruction"] + (f"\n\n{r['context']}" if r["context"].strip() else "")
            f.write(json.dumps({"text": text, "category": r["category"]}) + "\n")
    hq = list(load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation"))
    random.Random(a.seed).shuffle(hq)
    with open(os.path.join(out, "hotpot_dev.jsonl"), "w") as f:
        for r in hq[:a.n_agent]:
            paras = {t: " ".join(s) for t, s in zip(r["context"]["title"], r["context"]["sentences"])}
            f.write(json.dumps({"id": r["id"], "question": r["question"], "answer": r["answer"], "type": r["type"],
                                "paragraphs": paras}) + "\n")
    print({"dolly": len(rows), "hotpot": min(a.n_agent, len(hq))})


if __name__ == "__main__":
    main()
