"""M3 · real data for the web demo: leaderboards, per-edit TPR, and real example texts with token-level marks.

  python exp/demo_data.py --base llama-3.1-8b --out RUNS/demo_data.json      (CPU is enough: no language model runs)
Every number comes from an experiment directory; nothing is synthetic. Example texts all continue one held-out C4
prompt: Alice's model (the release), the same prompt after each of Dana's weight edits, Charlie's rewrites of
Alice's text, the unwatermarked base, and the human continuation. Each carries the detector statistic, the
empirical p-value against the shared null, and per-token marks (+1 green, -1 red, null = not scored).
"""
import argparse
import bisect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from markweightx.detect import empirical_p, repetition, threshold
from markweightx.edits import rewrite
from markweightx.models import ConfigOnly, load_tokenizer
from markweightx.paths import DATA, RUNS
from markweightx.registry import get_scheme
from markweightx.types import Release

WEIGHT_EDITS = ["finetune_steps200", "finetune_steps800", "finetune_steps2400", "merge_ratio0.2", "merge_ratio0.4",
                "merge_ratio0.6", "prune_sparsity0.2", "prune_sparsity0.5", "quantize_bits8", "quantize_bits4"]
TEXT_EDITS = [("synonym", 0.3), ("word_deletion", 0.3)]


def _group(tok, ids, vals, lead=""):
    """Decode tokens one by one, but join the pieces of a character that is split across tokens (a lone piece
    decodes to U+FFFD). A joined group keeps the first piece's mark."""
    out, i = [], 0
    while i < len(ids):
        j = i + 1
        while "\ufffd" in tok.decode(ids[i:j]) and j < len(ids) and j - i < 4:
            j += 1
        out.append({"t": (lead if i == 0 else "") + tok.decode(ids[i:j]), "v": vals[i]})
        i = j
    return out


def marks(sch, text):
    """Per-token marks from the scheme's own detector."""
    if sch.name == "pro":
        import jieba
        ctx = sch.wm.model
        w2 = ctx.get_text_split(text)
        out = [{"t": "".join(w2[0]).strip(), "v": None}] if w2 else []
        for i in range(1, len(w2)):
            context = "".join(sum(w2[:i], [])).strip(); current = "".join(w2[i]).strip()
            if len(list(jieba.cut(context))) < 40:
                out.append({"t": " " + current, "v": None}); continue
            with torch.no_grad():
                v = ctx.transform_model(ctx.get_embedding(context)).cpu()[0]
            arr = ctx.scale_vector(v, temperature=1000.0)[ctx.mapping]
            ids = ctx.tokenizer.encode(current, add_special_tokens=False)
            out += _group(ctx.tokenizer, ids, [1 if arr[t] > 0 else -1 for t in ids], lead=" ")
        return out
    if sch.teacher == "KGW":
        d = sch.wm.get_data_for_visualization(text)
        ids = sch.tok(text, add_special_tokens=False)["input_ids"]
        vals = [None if h is None else (1 if h == 1 else -1) for h in d.highlight_values]
        return _group(sch.tok, ids, vals)
    return None


def human_continuation(tok, prompt, n_new=200):
    for t in torch.load(DATA["llama3"])["eval_texts"]:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if len(ids) >= 250 and tok.decode(ids[:50]) == prompt:
            return tok.decode(ids[50:50 + n_new])
    return None


def method_block(rel_dir, tok, pick=None):
    rel = Release.open(rel_dir)
    sch = get_scheme(rel.scheme); sch.setup(ConfigOnly(rel.base, "cpu"), tok, "cpu", rel.key)
    tx = json.load(open(f"{rel_dir}/texts.json")); res = json.load(open(f"{rel_dir}/result.json"))
    null = sorted(tx["scores"]["null"]); thr = threshold(null, 0.01)
    rows = {}
    if os.path.exists(f"{rel_dir}/robustness.jsonl"):
        for l in open(f"{rel_dir}/robustness.jsonl"):
            r = json.loads(l); rows[r["edit"]] = {k: r[k] for k in ("tpr_at_1fpr", "auc", "ppl", "rep4")}
    st = tx["scores"]["student"]
    if pick is None:
        # A typical example, not the strongest: Alice's text is traced, while the base model's and the human
        # continuation of the same prompt are not; among those prompts, take the median statistic. Aggregate
        # numbers (leaderboard, edit grid) use every prompt.
        cand = [i for i in range(len(st)) if st[i] > thr and tx["scores"]["null"][i] < thr and repetition(tx["student"][i]) < 0.05]
        ok = []
        for i in sorted(cand, key=lambda i: st[i]):
            h = human_continuation(tok, tx["prompts"][i])
            if h and sch.score(h)[0] < thr:
                ok.append(i)
            if len(ok) >= 15:
                break
        pick = ok[len(ok) // 2] if ok else (cand[0] if cand else 0)
    def ex(label, text):
        s, n = sch.score(text)
        return {"label": label, "text": text, "statistic": s, "p": empirical_p(s, null), "flagged": s > thr, "n_tokens": n,
                "marks": marks(sch, text)}
    exs = [ex("alice", tx["student"][pick]), ex("base", None or json.load(open(os.path.join(os.path.dirname(rel_dir), "base_texts_n500.json")))[pick])]
    for e in WEIGHT_EDITS:
        f = f"{rel_dir}/edits/{e}/texts.json"
        if os.path.exists(f):
            exs.append(ex("dana:" + e, json.load(open(f))[pick]))
    for k, r in TEXT_EDITS:
        exs.append(ex(f"charlie:{k}_ratio{r}", rewrite([tx["student"][pick]], k, ratio=r)[0]))
    h = human_continuation(tok, tx["prompts"][pick])
    if h:
        exs.append(ex("human", h))
    return {"scheme": rel.scheme, "teacher": res.get("teacher"), "learned": res["student"], "ceiling": res.get("teacher_decode"),
            "ppl_student": res["ppl_student"], "ppl_base": res["ppl_base"], "edits": rows, "threshold": thr,
            "prompt": tx["prompts"][pick], "pick": pick, "examples": exs}, pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="llama-3.1-8b")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--methods", default="pro,markllm-kgw-d")
    ap.add_argument("--out", default=f"{RUNS}/demo_data.json")
    a = ap.parse_args()
    tag = a.tag or ("e5" if "8b" in a.base else "e2")
    tok = load_tokenizer(a.base)
    out = {"base": a.base, "methods": {}}
    pick = None
    for m in a.methods.split(","):
        d = f"{RUNS}/{tag}/{a.base}/{m}"
        if os.path.exists(f"{d}/result.json"):
            out["methods"][m], pick = method_block(d, tok, pick)     # same prompt for every method
            print(m, "pick", pick, "edits", len(out["methods"][m]["edits"]), flush=True)
    for b, t in (("llama-3.2-1b", "e2"), ("llama-3.1-8b", "e5")):
        f = f"{RUNS}/{t}/{b}/leaderboard.json"
        if os.path.exists(f):
            out.setdefault("leaderboards", {})[b] = json.load(open(f))
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
