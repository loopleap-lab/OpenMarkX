"""D4 · the leaderboard: one row per method, TPR@1%FPR (empirical null) after each edit, plus quality.

  python exp/leaderboard.py [--tag e2 --base llama-3.2-1b] [--format latex]
Prints the board and saves it as RUNS/<tag>/<base>/leaderboard.md (and .json for the demo). --format latex prints the
main table as a LaTeX tabular, in the layout of the PRO paper's main table. A plugin method (run through
markweightx bench) is a main-board row like a built-in one.
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from markweightx.paths import RUNS as R  # noqa: E402
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
    ap.add_argument("--format", choices=["md", "latex"], default="md")
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
        rows.append({"scheme": res["scheme"], "scheme_source": res.get("scheme_source"), "sweep": res.get("sweep"),
                     "experimental": res.get("experimental"), "scheme_params": res.get("scheme_params"), "note": res.get("label_note"), "teacher": res.get("teacher"), "learned": res["student"]["tpr_at_1fpr"],
                     "auc": res["student"]["auc"], "ceiling": (res.get("teacher_decode") or {}).get("tpr_at_1fpr"),
                     "ppl": res["ppl_student"], "ppl_base": res["ppl_base"], "edits": edits,
                     "by_length": res.get("by_length")})
    rows.sort(key=lambda r: -r["learned"])
    for r in rows:
        r["main"] = is_main(r)
    from markweightx.files import write_json
    write_json(f"{root}/leaderboard.json", {"base": a.base, "tag": a.tag, "rows": rows, "not_included": NOT_INCLUDED},
               indent=1)
    ppl_base = rows[0]["ppl_base"] if rows else 0
    if a.format == "latex":
        print(latex([r for r in rows if r["main"]], a.base))
        return
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        markdown(rows, a.base, ppl_base)
    from markweightx.files import write_text
    write_text(f"{root}/leaderboard.md", buf.getvalue())
    print(buf.getvalue(), end="")


LATEX_COLS = [("none", "Learned"), ("finetune_steps800", "FT"), ("merge_ratio0.4", "Merge"), ("prune_sparsity0.2", "Prune"),
              ("quantize_bits4", "NF4"), ("text:synonym_ratio0.3", "Syn."), ("attack:spoof_unigram", "Forged$\\downarrow$")]


def latex(rows, base):
    f = lambda x: "--" if x is None else f"{100 * x:.1f}"
    get = lambda r, k: r["learned"] if k == "none" else (r["edits"].get(k) or {}).get("tpr_at_1fpr")
    out = [r"\begin{tabular}{l" + "r" * (len(LATEX_COLS) + 1) + "}", r"\toprule",
           "Method & " + " & ".join(c for _, c in LATEX_COLS) + r" & PPL \\", r"\midrule"]
    for r in rows:
        out.append(r["scheme"].replace("_", r"\_") + " & " + " & ".join(f(get(r, k)) for k, _ in LATEX_COLS)
                   + f" & {r['ppl']:.1f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}",
            f"% TPR (%) at 1% FPR on {base}; FT = fine-tune 800 steps, Merge = 40% instruct, Prune = 20%, Syn. = 30% synonyms"]
    return "\n".join(out)


def markdown(rows, base, ppl_base):
    a = argparse.Namespace(base=base)
    print(f"## Leaderboard · {a.base} · TPR at 1% FPR (empirical null)\n")
    table([r for r in rows if r["main"]], ppl_base)
    print("\nforged ↓: share of texts an attacker forges without Alice's model or key (token counts from her 311 texts, then a biased base model); lower is safer.")
    print("\n\\* degenerate text (4-gram repetition > 0.3); the score is inflated by looping, not by the watermark.")
    print("\n† more than 2% false positives on texts cut to 50 or 100 tokens at this threshold: see the C9 table; "
          "mwx.calibrate keeps a null per length for this.")
    for k, why in NOT_INCLUDED.items():
        print(f"\nNot included: {k}: {why}.")
    main = [r for r in rows if r["main"]]
    if any(r.get("by_length") for r in main):
        cols = sorted({c for r in main for c in (r.get("by_length") or {})}, key=int)
        print("\n### C9 · texts cut short, scored at the 200-token threshold (false positives should not exceed ~1%)\n")
        print("| method | " + " | ".join(f"FPR @{c} tok | TPR @{c} tok (recalibrated)" for c in cols) + " |")
        print("|" + "---|" * (2 * len(cols) + 1))
        for r in main:
            bl = r.get("by_length") or {}
            def cell(c):
                if c not in bl:
                    return "– | –"
                bold = "**" if bl[c]["fpr"] > 0.02 else ""
                recal = f"{bl[c]['tpr_recal']:.2f}" if "tpr_recal" in bl[c] else "–"
                return f"{bold}{100 * bl[c]['fpr']:.1f}%{bold} | {recal}"
            print(f"| {r['scheme']} | " + " | ".join(cell(c) for c in cols) + " |")
        n = next(v["n"] for r in main for v in (r.get("by_length") or {}).values())
        print(f"\n{n} texts per cell (1% = {max(1, round(n / 100))} texts); **bold**: more than 2% false positives. "
              "The TPR column uses the 1% threshold of the cut null (what mwx.calibrate now does per length); "
              "– : not measured yet (exp/length_check.py).")
    print("\n### Ablations: experimental variants and parameter sweeps (not methods you would pick)\n")
    table([r for r in rows if not r["main"]], ppl_base)


def is_main(r) -> bool:
    """A leaderboard method is a registered, non-experimental scheme at its default parameters, or a plugin method
    (it passed markweightx check to get here). Parameter sweeps (name:k=v) are ablations."""
    from markweightx.registry import SCHEMES
    import markweightx.schemes  # noqa: F401  (registers the built-in schemes)
    if ":" in r["scheme"] or r.get("sweep") or r.get("experimental"):
        return False
    if r.get("scheme_source"):
        return True
    c = SCHEMES.get(r["scheme"])
    return c is not None and not getattr(c, "experimental", False)


def label(r):
    """Ablation rows say what actually ran: the parameters that differ from the method's current defaults."""
    if r.get("main") or not r.get("scheme_params"):
        return r["scheme"]
    from markweightx.registry import get_scheme, resolved_params
    name = r["scheme"].split(":")[0]
    try:
        now = resolved_params(get_scheme(name))
    except (KeyError, ValueError):
        return r["scheme"]
    diff = [f"{k}={v}" for k, v in r["scheme_params"].items() if now.get(k) != v] or ["= default"]
    return name + " (" + ", ".join(diff + ([r["note"]] if r.get("note") else [])) + ")"


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
        c9 = r.get("by_length") or {}
        mark = " †" if any(v.get("fpr", 0) > 0.02 for v in c9.values()) else ""
        print(f'| {label(r)}{mark} | {f(r["learned"])} | {f(r["ceiling"])} | {r["auc"]:.3f} | ' + " | ".join(cells) +
              f' | {r["ppl"]:.1f} | {f(rep)} |')


if __name__ == "__main__":
    main()
