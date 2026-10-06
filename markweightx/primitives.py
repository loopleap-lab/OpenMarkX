"""The six primitives every scheme is built from (technical appendix §1.2), and the access levels of §2.5.

    Key      keyed pseudo-randomness (a secret, the material derived from it); never touches logits or weights
    Signal   the training target: base logits -> watermarked logits, teacher-forced (adapts a teacher strategy)
    Imprint  how the mark gets into the weights: "distill" (learn a Signal) or "edit" (write the weights directly)
    Perturb  weight edits, used as attacks (finetune / merge / prune / quantize) and inside training (FPL's step)
    Probe    what the detector may touch: the access level below
    Score    text -> statistic, then calibrated against a frozen null into a p-value

A Scheme declares which primitives it uses; teacher-free schemes (GaussMark) use Key, Imprint.edit, Probe and Score
and never Signal, which is what makes the SDK a layer of its own rather than a wrapper around MarkLLM.
"""
from __future__ import annotations

from enum import Enum


class Access(str, Enum):
    """What the detector needs besides the text (appendix §2.5)."""
    R0 = "R0: key + tokenizer"
    R0_AUX = "R0+aux: key + tokenizer + an auxiliary network"
    R1 = "R1: the base model, forward passes"
    R2 = "R2: the base model, gradients"


class Imprint(str, Enum):
    DISTILL = "distill"      # learn a Signal into LoRA weights (logit distillation)
    EDIT = "edit"            # write the mark into the weights directly (teacher-free)


PRIMITIVES = ("Key", "Signal", "Imprint", "Perturb", "Probe", "Score")


def composition(scheme) -> dict:
    """Which primitives a scheme uses, and how (one row of the Scheme x Primitive matrix, appendix §1.3)."""
    teacher = getattr(scheme, "teacher", None)
    imprint = getattr(scheme, "imprint", Imprint.DISTILL)
    return {
        "Key": "teacher config + secret" if teacher else "secret (scheme-owned)",
        "Signal": (f"MarkLLM {teacher}" if teacher else "scheme-owned policy") if imprint == Imprint.DISTILL else "—",
        "Imprint": imprint.value if isinstance(imprint, Imprint) else str(imprint),
        "Perturb": "FPL step in training" if getattr(scheme, "booster", None) else "attacks only",
        "Probe": scheme.access.value.split(":")[0] if hasattr(scheme, "access") else "R0",
        "Score": "teacher's detector" if teacher else "scheme's detector",
    }


def describe(names: list[str] | None = None) -> str:
    """The Scheme x Primitive matrix as a text table."""
    from .registry import SCHEMES, get_scheme
    get_scheme("nowm")
    names = names or sorted(n for n in SCHEMES if n != "nowm")
    rows = [(n, composition(get_scheme(n))) for n in names]
    w = max(len(n) for n, _ in rows)
    cols = list(PRIMITIVES)
    widths = {c: max(len(c), *(len(r[c]) for _, r in rows)) for c in cols}
    head = "scheme".ljust(w) + "  " + "  ".join(c.ljust(widths[c]) for c in cols)
    lines = [head, "-" * len(head)]
    for n, r in rows:
        lines.append(n.ljust(w) + "  " + "  ".join(r[c].ljust(widths[c]) for c in cols))
    return "\n".join(lines)
