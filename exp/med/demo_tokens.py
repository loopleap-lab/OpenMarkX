"""Demo pages: MedGemma answers with every token colored by the watermark detector (green = counts for the mark).

  python exp/med/demo_tokens.py --n 4 --out $MWX_RUNS/med/demo_tokens
For each picked question (answered correctly by both watermarked students, 300-700 tokens), one HTML page per scheme:
the question with its images, the student's answer and the base model's answer to the same question, each colored.
KGW: a token is green when it is in the green list its previous token selects (MarkLLM's own visualization data).
PRO: each 10-word chunk after the first 40 words is scored against the context before it; a token is green when its
mapped policy value is +1 (PRO's own detect, token by token); grey = context, not scored.
"""
import argparse
import base64
import html
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openmarkx.paths import RUNS

BASE = "medgemma-1.5-4b-it"
GREEN, RED, GREY = "#c8f0d0", "#f6c9c4", "#ececec"


def kgw_tokens(sch, text):
    d = sch.wm.get_data_for_visualization(text)            # MarkLLM's DataForVisualization
    toks, flags = d.decoded_tokens, d.highlight_values
    return [(t, {1: "g", 0: "r"}.get(f, "n")) for t, f in zip(toks, flags)]


def pro_tokens(sch, text):
    import jieba
    import torch
    m = sch.wm.model
    tok = m.tokenizer
    words = m.get_text_split(text)
    out = []
    for i, chunk in enumerate(words):
        cur = "".join(chunk)
        ctx = "".join("".join(g) for g in words[:i]).strip()
        ids = tok(cur, add_special_tokens=False)["input_ids"]
        pieces = [tok.decode([t]) for t in ids]
        if i == 0 or len(list(jieba.cut(ctx))) < 40:
            out += [(p, "n") for p in pieces]
            continue
        with torch.no_grad():
            v = m.scale_vector(m.transform_model(m.get_embedding(ctx)).cpu()[0], temperature=1000.0)[m.mapping]
        out += [(p, "g" if float(v[t]) > 0 else "r") for p, t in zip(pieces, ids)]
    return out


def render(tokens):
    col = {"g": GREEN, "r": RED, "n": GREY}
    return "".join(f'<span style="background:{col[c]}">{html.escape(t)}</span>' for t, c in tokens)


def frac(tokens):
    s = [c for _, c in tokens if c != "n"]
    return sum(c == "g" for c in s) / max(len(s), 1), len(s)


def thumb(path, size=360):
    from PIL import Image
    im = Image.open(path).convert("RGB"); im.thumbnail((size, size))
    b = io.BytesIO(); im.save(b, format="JPEG", quality=85)
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(b.getvalue()).decode()}" style="margin:4px;max-width:45%">'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(RUNS, "med", "demo_tokens"))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from evaluate import letter
    from openmarkx.api import _detect_device, build_scheme
    from openmarkx.keys import load_key
    from openmarkx.models import ConfigOnly
    from openmarkx.types import Release
    from openmarkx.vlm import load_processor
    tok = load_processor(BASE).tokenizer
    items = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(RUNS, "med", "mxqa_mm_eval.jsonl"))}
    allt = [json.loads(l)["id"] for l in open(os.path.join(RUNS, "med", "mxqa_mm_test.jsonl"))]
    base = dict(zip(allt, json.load(open(os.path.join(RUNS, "med", f"mxqa_base_{BASE}_n2000_t1024.json")))))
    runs = {"PRO": os.path.join(RUNS, "med", BASE, "pro"), "KGW": os.path.join(RUNS, "med", BASE, "markllm-kgw-d")}
    stu = {}
    for k, r in runs.items():
        d = json.load(open(os.path.join(r, "mxqa_student.json")))
        stu[k] = dict(zip(d["ids"], d["texts"]))
    ntok = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])
    picks = [i for i in items if all(i in stu[k] and letter(stu[k][i]) == items[i]["label"]
                                     and 300 <= ntok(stu[k][i]) <= 700 for k in stu)][:a.n]
    summary = {}
    for k, r in runs.items():
        rel = Release.open(r); key = load_key(rel)
        sch = build_scheme(key, trust_plugin=True)
        _detect_device(key, a.device, type(sch))
        sch.setup(ConfigOnly(BASE, a.device), tok, a.device, key.params)
        fn = pro_tokens if k == "PRO" else kgw_tokens
        for n, qid in enumerate(picks):
            it = items[qid]
            ts, tb = fn(sch, stu[k][qid]), fn(sch, base[qid])
            (fs, ns), (fb, nb) = frac(ts), frac(tb)
            zs, zb = sch.score(stu[k][qid])[0], sch.score(base[qid])[0]
            page = f"""<!doctype html><meta charset="utf-8"><title>{k} watermark, {qid}</title>
<body style="font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;line-height:1.55">
<h2>{k} watermark learned into MedGemma 1.5 (4B) — question {qid}</h2>
<div>{''.join(thumb(p) for p in it['images'])}</div>
<p style="white-space:pre-wrap">{html.escape(it['text'])}</p><p>Correct answer: ({it['label']})</p>
<div style="display:flex;gap:20px;flex-wrap:wrap">
<div style="flex:1;min-width:300px"><h3>Watermarked model (mark in the weights)</h3>
<p><b>{100 * fs:.0f}% green</b> of {ns} scored tokens; detector score {zs:.2f} → <b>watermarked</b></p>
<div style="white-space:pre-wrap">{render(ts)}</div></div>
<div style="flex:1;min-width:300px"><h3>Original model</h3>
<p><b>{100 * fb:.0f}% green</b> of {nb} scored tokens; detector score {zb:.2f}</p>
<div style="white-space:pre-wrap">{render(tb)}</div></div></div>
<p style="color:#666">Green: the token counts for the watermark; red: it does not; grey: not scored.</p></body>"""
            name = f"{k.lower()}_{n}_{qid}.html"
            open(os.path.join(a.out, name), "w").write(page)
            summary[name] = {"student_green": fs, "base_green": fb, "student_score": zs, "base_score": zb}
            print(json.dumps({name: summary[name]}), flush=True)
    json.dump({"picks": picks, "pages": summary}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
