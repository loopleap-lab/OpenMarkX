"""Demo material for the agent run: one transcript (tool text grey, the final answer colored by OpenMark's detector) and the
numbers, as JSON for the site and as a standalone HTML page.

  python exp/agent/demo_page.py --release $OMX_RUNS/agent/llama-3.1-8b-instruct/pro --out $OMX_RUNS/agent/demo
Picks a question the watermarked agent answered correctly, after at least two searches, whose final answer is flagged
(score above the 1% threshold of the base agent's final answers); the base agent's final answer to the same question
is colored too, for contrast.
"""
import argparse
import html
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "med"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prefix", default="agent")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from demo_tokens import pro_tokens, frac, render       # exp/med/demo_tokens.py
    from hotpot_agent import norm
    from openmarkx.api import _detect_device, build_scheme
    from openmarkx.keys import load_key
    from openmarkx.models import ConfigOnly, load_tokenizer
    from openmarkx.types import Release
    rel = Release.open(a.release)
    tok = load_tokenizer(rel.base)
    res = json.load(open(os.path.join(a.release, f"{a.prefix}_result.json")))
    thr = res["student"]["detect"]["thr"]
    stu = [json.loads(l) for l in open(os.path.join(a.release, f"{a.prefix}_student.jsonl"))]
    base = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(a.release, f"{a.prefix}_base.jsonl")))}
    key = load_key(rel)
    sch = build_scheme(key, trust_plugin=True)
    _detect_device(key, a.device, type(sch))
    sch.setup(ConfigOnly(rel.base, a.device), tok, a.device, key.params)
    pick = None
    for r, s in zip(stu, res["scores"]["student"]):
        n_search = sum(m["role"] == "user" and m["content"].startswith("Observation: [") for m in r["transcript"])
        if r["finish"] and norm(r["gold"]) in norm(r["finish"]) and n_search >= 2 and s > thr:
            pick = r; break
    pick = pick or stu[0]
    steps = [{"role": m["role"], "text": m["content"]} for m in pick["transcript"][1:-2]]   # skip system, final turn
    fs, fb = pro_tokens(sch, pick["final"]), pro_tokens(sch, base[pick["id"]]["final"])
    (gs, ns), (gb, nb) = frac(fs), frac(fb)
    zs, zb = sch.score(pick["final"])[0], sch.score(base[pick["id"]]["final"])[0]
    merge = lambda runs: [list(x) for x in runs]
    out = {"question": pick["question"], "gold": pick["gold"], "steps": steps, "final": merge(fs), "base_final": merge(fb),
           "green": [gs, gb], "score": [zs, zb], "threshold": thr,
           "stats": {"tpr_at_1fpr": res["student"]["detect"]["tpr_at_1fpr"], "auc": res["student"]["detect"]["auc"],
                     "acc_base": res["base"]["accuracy"], "acc_student": res["student"]["accuracy"], "n": res["n"]}}
    json.dump(out, open(os.path.join(a.out, "agent_demo.json"), "w"))
    rows = "".join(
        f'<div style="margin:8px 0;padding:8px 12px;border-radius:8px;background:{"#f3f1ec" if s["role"] == "user" else "#ffffff"};'
        f'border:1px solid #e3dacc;white-space:pre-wrap;color:{"#8e8a80" if s["role"] == "user" else "#141413"}">'
        f'<b>{"Tool" if s["role"] == "user" else "Agent"}</b><br>{html.escape(s["text"])}</div>' for s in steps)
    page = f"""<!doctype html><meta charset="utf-8"><title>Agent transcript</title>
<body style="font-family:system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;line-height:1.55">
<h2>Agent on Llama-3.1-8B-Instruct, OpenMark in the weights</h2><p><b>Question:</b> {html.escape(pick['question'])} (gold: {html.escape(pick['gold'])})</p>
{rows}
<h3>Final answer, watermarked agent: {100 * gs:.0f}% green, score {zs:.3f} (threshold {thr:.3f}) → watermarked</h3>
<div style="white-space:pre-wrap">{render(fs)}</div>
<h3>Final answer, unmarked agent: {100 * gb:.0f}% green, score {zb:.3f}</h3>
<div style="white-space:pre-wrap">{render(fb)}</div></body>"""
    open(os.path.join(a.out, "agent_demo.html"), "w").write(page)
    print(json.dumps({k: v for k, v in out.items() if k in ("question", "green", "score", "stats")}, indent=1))


if __name__ == "__main__":
    main()
