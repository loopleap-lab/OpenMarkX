"""detect: score texts with a scheme's detector, calibrated against an empirical null.

The null is the same detector on text the watermark never touched (the unwatermarked base's continuations of the
same prompts). Thresholds and p-values come from that null, so "1% FPR" means the same thing for every scheme.
"""
from __future__ import annotations

import bisect

from sklearn.metrics import roc_auc_score

from .types import Detection


def scores(scheme, texts):
    return scheme.score_batch(texts)


def empirical_p(stat: float, null_sorted: list[float]) -> float:
    k = len(null_sorted) - bisect.bisect_left(null_sorted, stat)      # null values >= stat
    return (k + 1) / (len(null_sorted) + 1)


def detect(scheme, texts, null: list[float] | None = None, fpr=0.01) -> list[Detection]:
    ns = sorted(null) if null else None
    thr = threshold(ns, fpr) if ns else None
    out = []
    for s, n in scores(scheme, texts):
        if ns and not getattr(scheme, "exact_null", False):  # a calibrated key: its frozen empirical null
            p, flagged = empirical_p(s, ns), s > thr
        else:                                               # otherwise the scheme's closed form, if it has one
            p = scheme.pvalue(s, n)
            flagged = (p < fpr) if p is not None else None
        out.append(Detection(statistic=s, p_value=p, n_tokens=n, flagged=flagged))
    return out


def threshold(null_sorted, fpr=0.01):
    import math
    return null_sorted[max(0, math.ceil((1 - fpr) * len(null_sorted)) - 1)]


def summary(wm_stats, null_stats, fpr=0.01):
    ns = sorted(null_stats)
    thr = threshold(ns, fpr)
    med = lambda v: sorted(v)[len(v) // 2]
    return {"tpr_at_1fpr": sum(s > thr for s in wm_stats) / len(wm_stats), "thr": thr,
            "auc": roc_auc_score([1] * len(wm_stats) + [0] * len(ns), list(wm_stats) + ns),
            "median_wm": med(wm_stats), "median_null": med(ns), "n_wm": len(wm_stats), "n_null": len(ns)}


LENGTHS = (50, 100)             # C9: the evaluation texts are 200 tokens; how does the threshold hold on shorter ones?


def length_curve(scheme, tok, wm_texts, null_texts, thr, lengths=LENGTHS):
    """C9: the texts cut to their first L tokens and scored again, at the threshold set on the full-length null. A
    detector whose null drifts with length shows a false-positive rate far from the one it was calibrated for."""
    cut = lambda ts, n: [tok.decode(tok(t, add_special_tokens=False)["input_ids"][:n]) for t in ts]
    out = {}
    for n in lengths:
        sn = [s for s, _ in scores(scheme, cut(null_texts, n))]
        sw = [s for s, _ in scores(scheme, cut(wm_texts, n))]
        out[str(n)] = {"fpr": sum(s > thr for s in sn) / len(sn), "tpr": sum(s > thr for s in sw) / len(sw), "n": len(sn)}
    return out


def repetition(text: str, n: int = 4) -> float:
    """Share of repeated word n-grams; near 1 means the model is looping (degenerate text inflates some detectors)."""
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0
