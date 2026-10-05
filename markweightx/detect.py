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
        out.append(Detection(statistic=s, p_value=empirical_p(s, ns) if ns else None, n_tokens=n,
                             flagged=(s > thr) if thr is not None else None))
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
