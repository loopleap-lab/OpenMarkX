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


ORDER = {"R0": 0, "R0_AUX": 1, "R1": 2, "R2": 3}


class AccessDenied(PermissionError):
    pass


class Probe:
    """What a detector may touch, granted at a level. Detectors ask the probe instead of holding a model, so a
    detector that needs more than it was granted fails loudly instead of quietly loading weights.

      probe.tokenizer()                 always
      probe.model()                     R1 and up: the base model (no gradients)
      probe.grad_seq_logprob(text, wrt) R2: gradient of the text's log-likelihood w.r.t. a named parameter
    """

    def __init__(self, base: str, device: str, grant: "Access"):
        self.base, self.device, self.grant = base, device, grant
        self._tok = self._model = None

    def _need(self, level: "Access", what: str):
        if ORDER[level.name] > ORDER[self.grant.name]:
            raise AccessDenied(f"{what} needs {level.value}; the detector was granted {self.grant.value}")

    def tokenizer(self):
        from .models import load_tokenizer
        self._tok = self._tok or load_tokenizer(self.base)
        return self._tok

    def model(self):
        self._need(Access.R1, "a forward pass of the base model")
        if self._model is None:
            import torch
            from .models import load_base
            # R2 statistics are inner products with small perturbations: load in fp32 so gradients are not bf16-coarse
            self._model = load_base(self.base, self.device, dtype=torch.float32 if self.grant == Access.R2 else torch.bfloat16)
            self._model.requires_grad_(False)
        return self._model

    def grad_seq_logprob(self, texts, wrt):
        """Gradients of log p(text) w.r.t. the named parameter(s) (R2 only). One text -> {name: grad}; a list ->
        a generator of {name: grad}, one text at a time, so a caller reducing each gradient (an inner product, a
        norm) never holds more than one: a 1B layer's gradient is ~64 MB in fp32. Only the requested parameters
        track gradients, so the cost is one forward + backward per text."""
        self._need(Access.R2, "the gradient of the base model")   # refuse before anything is loaded
        if isinstance(texts, str):
            return next(self._grads([texts], wrt))
        return self._grads(list(texts), wrt)

    def _grads(self, texts, wrt):
        names = [wrt] if isinstance(wrt, str) else list(wrt)
        for t in texts:
            yield self._grad_one(t, names)

    def _grad_one(self, text, names):
        """One text's gradients, with grad mode and requires_grad confined to this call."""
        import torch
        self._need(Access.R2, "the gradient of the base model")
        m, tok = self.model(), self.tokenizer()
        params = dict(m.named_parameters())
        ps = [params[n] for n in names]
        with torch.enable_grad():
            for p in ps:
                p.requires_grad_(True)
            try:
                ids = tok(text, return_tensors="pt").input_ids.to(self.device)
                lp = -torch.nn.functional.cross_entropy(m(input_ids=ids).logits[0, :-1].float(), ids[0, 1:], reduction="sum")
                gs = torch.autograd.grad(lp, ps)
            finally:
                for p in ps:
                    p.requires_grad_(False)
        return {n: g.detach() for n, g in zip(names, gs)}


class Imprint(str, Enum):
    DISTILL = "distill"      # learn a Signal into LoRA weights (logit distillation)
    EDIT = "edit"            # write the mark into the weights directly (teacher-free)


PRIMITIVES = ("Key", "Signal", "Imprint", "Perturb", "Probe", "Score")


def kind_of(scheme) -> str:
    """The two classes of the appendix (§0, §1.1): teacher-learned schemes learn a teacher into the weights through
    Signal (the teacher comes from MarkLLM, or the scheme owns it, like PRO's policy); teacher-free schemes write the
    weights directly and never use Signal (GaussMark)."""
    return "teacher-free" if getattr(scheme, "imprint", Imprint.DISTILL) == Imprint.EDIT else "teacher-learned"


def composition(scheme) -> dict:
    """Which primitives a scheme uses, and how (one row of the Scheme x Primitive matrix, appendix §1.3)."""
    teacher = getattr(scheme, "teacher", None)
    imprint = getattr(scheme, "imprint", Imprint.DISTILL)
    if teacher:
        key = "teacher config + secret"
    elif imprint == Imprint.EDIT:
        key = "secret -> weight perturbation"
    else:
        key = "secret -> scheme's own teacher"
    return {
        "Class": kind_of(scheme),
        "Key": key,
        "Signal": (f"MarkLLM {teacher}" if teacher else "scheme's own teacher") if imprint == Imprint.DISTILL else "— (none)",
        "Imprint": imprint.value if isinstance(imprint, Imprint) else str(imprint),
        "Perturb": "FPL step in training" if getattr(scheme, "booster", None) else "attacks only",
        "Probe": scheme.access.value.split(":")[0] if hasattr(scheme, "access") else "R0",
        "Score": "teacher's detector" if teacher else "scheme's detector",
    }


def describe(names: list[str] | None = None) -> str:
    """The Scheme x Primitive matrix as a text table."""
    from .registry import SCHEMES, get_scheme
    get_scheme("nowm")
    names = names or sorted(n for n, c in SCHEMES.items() if n != "nowm" and not getattr(c, "experimental", False))
    rows = [(n, composition(get_scheme(n))) for n in names]
    w = max(len(n) for n, _ in rows)
    cols = ["Class", *PRIMITIVES]
    widths = {c: max(len(c), *(len(r[c]) for _, r in rows)) for c in cols}
    head = "scheme".ljust(w) + "  " + "  ".join(c.ljust(widths[c]) for c in cols)
    lines = [head, "-" * len(head)]
    for n, r in rows:
        lines.append(n.ljust(w) + "  " + "  ".join(r[c].ljust(widths[c]) for c in cols))
    return "\n".join(lines)
