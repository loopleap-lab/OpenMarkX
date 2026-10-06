"""D4 · the leaderboard: one row per method, TPR@1%FPR (empirical null) after each edit, plus quality.

  python exp/leaderboard.py [--tag e2 --base llama-3.2-1b] > leaderboard.md
Also writes RUNS/<tag>/<base>/leaderboard.json for the demo.
Columns: learned (no edit) and the decode-time teacher ceiling; weight edits; text edits; PPL of the student and
mean 4-gram repetition. A cell is marked * when the edited model's text is degenerate (rep4 > 0.3), because
looping text inflates some detectors.
"""
import argparse
import glob
import sys
import json
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.environ.get("MWX_RUNS", "/mnt/data/jiaq/mwx-runs")
COLS = [("finetune_steps200", "FT 200"), ("finetune_steps800", "FT 800"), ("finetune_steps2400", "FT 2400"),
        ("merge_ratio0.2", "merge .2"), ("merge_ratio0.4", "merge .4"), ("merge_ratio0.6", "merge .6"),
        ("prune_sparsity0.2", "prune 20%"), ("prune_sparsity0.5", "prune 50%"),
        ("quantize_bits8", "int8"), ("quantize_bits4", "nf4"),
        ("text:word_deletion_ratio0.3", "delete 30%"), ("text:synonym_ratio0.3", "synonym 30%"),
        ("attack:spoof_unigram", "forged ↓")]
NOT_INCLUDED = {"markllm-expgumbel-d": "upstream builds a (V·prefix)×V noise table, 128 GB at V=128k; same Gumbel-max rule as EXP"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="e2")
    ap.add_argument("--base", default="llama-3.2-1b")
    a = ap.parse_args()
    root = f"{R}/{a.tag}/{a.base}"
    rows = []
    for d in sorted(glob.glob(f"{root}/*/result.json")):
        res = json.load(open(d)); sd = os.path.dirname(d)
        edits = {}
        rf = f"{sd}/robustness.jsonl"
        if os.path.exists(rf):
            for l in open(rf):
                r = json.loads(l); edits[r["edit"]] = r
        rows.append({"scheme": res["scheme"], "teacher": res.get("teacher"), "learned": res["student"]["tpr_at_1fpr"],
                     "auc": res["student"]["auc"], "ceiling": (res.get("teacher_decode") or {}).get("tpr_at_1fpr"),
                     "ppl": res["ppl_student"], "ppl_base": res["ppl_base"], "edits": edits})
    rows.sort(key=lambda r: -r["learned"])
    for r in rows:
        r["main"] = is_main(r["scheme"])
    json.dump({"base": a.base, "tag": a.tag, "rows": rows, "not_included": NOT_INCLUDED}, open(f"{root}/leaderboard.json", "w"), indent=1)
    ppl_base = rows[0]["ppl_base"] if rows else 0
    print(f"## Leaderboard · {a.base} · TPR at 1% FPR (empirical null)\n")
    table([r for r in rows if r["main"]], ppl_base)
    print("\nforged ↓: share of texts an attacker forges without Alice's model or key (token counts from her 311 texts, then a biased base model); lower is safer.")
    print("\n\\* degenerate text (4-gram repetition > 0.3); the score is inflated by looping, not by the watermark.")
    for k, why in NOT_INCLUDED.items():
        print(f"\nNot included: {k}: {why}.")
    print("\n### Ablations: experimental variants and parameter sweeps (not methods you would pick)\n")
    table([r for r in rows if not r["main"]], ppl_base)


def is_main(name: str) -> bool:
    """A leaderboard method is a registered, non-experimental scheme at its default parameters."""
    from markweightx.registry import SCHEMES
    import markweightx.schemes  # noqa: F401  (registers the built-in schemes)
    c = SCHEMES.get(name)
    return c is not None and not getattr(c, "experimental", False)


def table(rows, ppl_base):
    f = lambda x: "–" if x is None else f"{x:.2f}"
    print("| method | learned | ceiling | AUC | " + " | ".join(c for _, c in COLS) + f" | PPL (base {ppl_base:.1f}) | rep4 |")
    print("|" + "---|" * (len(COLS) + 6))
    for r in rows:
        cells = []
        for k, _ in COLS:
            e = r["edits"].get(k)
            cells.append("" if e is None else f(e["tpr_at_1fpr"]) + ("*" if e.get("rep4", 0) > 0.3 else ""))
        rep = r["edits"].get("none", {}).get("rep4")
        print(f'| {r["scheme"]} | {f(r["learned"])} | {f(r["ceiling"])} | {r["auc"]:.3f} | ' + " | ".join(cells) +
              f' | {r["ppl"]:.1f} | {f(rep)} |')


if __name__ == "__main__":
    main()
