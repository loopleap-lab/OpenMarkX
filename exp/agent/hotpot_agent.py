"""GPU: a small ReAct agent on HotpotQA, run with the watermarked model and with its unmarked base; the watermark is
tested on the agent's final answer only.

  python exp/agent/hotpot_agent.py --release $MWX_RUNS/agent/llama-3.1-8b-instruct/pro --n 300
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

from markweightx.paths import RUNS

SYSTEM = ("You answer questions by searching Wikipedia. Work in steps. In each step write one line 'Thought: ...' and "
          "then one line with exactly one action: 'Action: Search[<title>]' to read a Wikipedia paragraph, or "
          "'Action: Finish[<short answer>]' when you know the answer. Titles you can search: {titles}.")
FINAL = ("Now write your final answer for a reader in about {words} words: state the answer, then explain how the "
         "evidence you found supports it. Do not write 'Thought' or 'Action'.")


def norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(w for w in s.split() if w not in ("a", "an", "the"))


def parse_action(text):
    m = re.search(r"Action:\s*(Search|Finish)\[(.*?)\]", text, re.S)
    return (m.group(1), m.group(2).strip()) if m else (None, None)


def search(paragraphs, query):
    titles = list(paragraphs)
    best = difflib.get_close_matches(query, titles, n=1, cutoff=0.0)
    t = best[0] if best else titles[0]
    return f"[{t}] {paragraphs[t]}"


@torch.no_grad()
def generate(model, tok, convs, max_new, temperature, seed, adapter):
    torch.manual_seed(seed)
    chats = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs]
    enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    ctx = model.disable_adapter() if (not adapter and hasattr(model, "disable_adapter")) else contextlib.nullcontext()
    with ctx:
        o = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=temperature, top_p=1.0,
                           top_k=0, pad_token_id=tok.pad_token_id)
    return tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def run_agent(model, tok, items, adapter, a, seed):
    convs = [[{"role": "system", "content": SYSTEM.format(titles="; ".join(it["paragraphs"]))},
              {"role": "user", "content": it["question"]}] for it in items]
    finish = [None] * len(items)
    for step in range(a.max_steps):
        live = [i for i in range(len(items)) if finish[i] is None]
        if not live:
            break
        for b in range(0, len(live), a.bs):
            idx = live[b:b + a.bs]
            outs = generate(model, tok, [convs[i] for i in idx], 160, a.temperature, seed * 100 + step * 10 + b, adapter)
            for i, out in zip(idx, outs):
                out = out.split("Observation:")[0].strip()
                kind, arg = parse_action(out)
                if kind:                                    # keep the turn up to its action
                    out = out[:re.search(r"Action:\s*(Search|Finish)\[.*?\]", out, re.S).end()]
                convs[i].append({"role": "assistant", "content": out})
                if kind == "Finish":
                    finish[i] = arg
                elif kind == "Search":
                    convs[i].append({"role": "user", "content": "Observation: " + search(items[i]["paragraphs"], arg)})
                else:
                    convs[i].append({"role": "user", "content": "Observation: no action found; use Search[...] or Finish[...]."})
        print({"step": step + 1, "finished": sum(f is not None for f in finish), "of": len(items)}, flush=True)
    finals = [None] * len(items)
    for b in range(0, len(items), a.bs):
        idx = list(range(b, min(b + a.bs, len(items))))
        for i in idx:
            convs[i].append({"role": "user", "content": FINAL.format(words=a.final_words)})
        outs = generate(model, tok, [convs[i] for i in idx], a.final_tokens, a.temperature, seed * 100 + 99 + b, adapter)
        for i, out in zip(idx, outs):
            finals[i] = out.strip()
            convs[i].append({"role": "assistant", "content": finals[i]})
    return convs, finish, finals


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
    from markweightx.api import _detect_device, build_scheme
    from markweightx.keys import load_key
    from markweightx.models import load_release, load_tokenizer
    from markweightx.scoring import scores, summary
    from markweightx.types import Release
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
