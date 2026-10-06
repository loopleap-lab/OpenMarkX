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


def _null_for(length: int, full: list[float], by_length: dict | None, full_len: int | None) -> list[float]:
    """The calibrated null for a text of `length` tokens: the one calibrated at the largest length not above it, or
    None for a text shorter than every calibrated length (it cannot be judged). Nulls calibrated on shorter texts are
    the conservative side for detectors whose false positives fall with length (C9)."""
    nulls = {int(n): v for n, v in (by_length or {}).items()}
    if full_len:
        nulls[full_len] = full
    if not nulls:
        return full
    fit = [n for n in nulls if n <= length]
    return nulls[max(fit)] if fit else None                 # shorter than every calibrated length: no valid null


def detect(scheme, texts, null: list[float] | None = None, fpr=0.01, by_length: dict | None = None,
           null_len: int | None = None) -> list[Detection]:
    ns = sorted(null) if null else None
    thr = threshold(ns, fpr) if ns else None
    out = []
    lengths = [len(scheme.tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    empty = [not n for n in lengths]
    scored = iter(scores(scheme, [t for t, e in zip(texts, empty) if not e]) if not all(empty) else [])
    if ns and not getattr(scheme, "exact_null", False):
        import warnings
        if not by_length:
            warnings.warn("this key was calibrated with one null for every length: short texts can be over-flagged "
                          "(C9); calibrate it again with mwx.calibrate")
        elif min(int(n) for n in by_length) > LENGTHS[0]:
            warnings.warn(f"this key's shortest calibrated length is {min(int(n) for n in by_length)} tokens: shorter "
                          f"texts are not judged; calibrate it again for nulls down to {LENGTHS[0]} tokens")
        elif null_len and max(lengths, default=0) > 1.5 * null_len:
            warnings.warn(f"some texts are much longer than the {null_len}-token texts the null was calibrated on")
    for e, length in zip(empty, lengths):
        s, n = (0.0, 0) if e else next(scored)
        if not n:                                           # nothing scored: a vacuous answer, as in MarkAgentX
            out.append(Detection(statistic=s, p_value=1.0, n_tokens=0, flagged=None))
            continue
        if ns and not getattr(scheme, "exact_null", False):  # a calibrated key: its frozen empirical null
            null_l = _null_for(length, ns, by_length, null_len)
            if null_l is None:                              # too short to judge: no p-value, not counted as negative
                out.append(Detection(statistic=s, p_value=None, n_tokens=n, flagged=None))
                continue
            p, flagged = empirical_p(s, null_l), s > (thr if null_l is ns else threshold(null_l, fpr))
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


LENGTHS = (20, 50, 100)         # C9: the evaluation texts are 200 tokens; how does the threshold hold on shorter ones?


def cut(tok, texts, n):
    """Each text's first n tokens."""
    return [tok.decode(tok(t, add_special_tokens=False)["input_ids"][:n]) for t in texts]


def length_curve(scheme, tok, wm_texts, null_texts, thr, lengths=LENGTHS, fpr=0.01):
    """C9: the texts cut to their first L tokens and scored again. fpr / tpr: at the threshold set on the full-length
    null (a detector whose null drifts with length shows false positives well above 1%); tpr_recal: at the 1%
    threshold of the cut null itself, the detection power a length-aware calibration keeps."""
    out = {}
    for n in lengths:
        sn = [s for s, _ in scores(scheme, cut(tok, null_texts, n))]
        sw = [s for s, _ in scores(scheme, cut(tok, wm_texts, n))]
        t_n = threshold(sorted(sn), fpr)
        out[str(n)] = {"fpr": sum(s > thr for s in sn) / len(sn), "tpr": sum(s > thr for s in sw) / len(sw),
                       "tpr_recal": sum(s > t_n for s in sw) / len(sw), "n": len(sn)}
    return out


def repetition(text: str, n: int = 4) -> float:
    """Share of repeated word n-grams; near 1 means the model is looping (degenerate text inflates some detectors)."""
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0
