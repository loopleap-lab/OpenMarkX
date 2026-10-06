"""GaussMark (Block et al., 2025): a teacher-free watermark written straight into the weights.

Imprint.edit: add xi ~ N(0, sigma^2 I) to one weight matrix, xi drawn from the key's secret. No training, no teacher.
Score (R2):   for a text y, g = d/dW log p_{theta0 + xi}(y) at the edited matrix; psi = <xi, g> / (sigma * ||g||).
              Text that does not come from the edited model has psi ~ N(0, 1), so the p-value is closed form.
Primitives:   Key (secret -> xi), Imprint.edit, Probe at R2 (gradients of the base), Score. Never Signal, never MarkLLM:
              the scheme that shows the SDK is a layer of its own (technical appendix §1.1).
"""
from __future__ import annotations

import math
import os

import torch

from ..primitives import Access, Imprint
from ..registry import register_scheme
from .base import Scheme


@register_scheme
class GaussMark(Scheme):
    name, kind = "gaussmark", "none"
    imprint = Imprint.EDIT
    access = Access.R2

    def __init__(self, sigma: float = 0.01, layer: int = 10, matrix: str = "mlp.up_proj", **params):
        super().__init__(sigma=sigma, layer=layer, matrix=matrix, **params)
        self.sigma, self.layer, self.matrix = float(sigma), int(layer), matrix
        self.param = f"model.layers.{self.layer}.{self.matrix}.weight"

    def _xi(self, shape, secret: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(int(secret))       # CPU generator: the same xi on any machine
        return torch.randn(shape, generator=g, dtype=torch.float32) * self.sigma

    def setup(self, model, tokenizer, device, key=None):
        super().setup(model, tokenizer, device, key)
        key = dict(key or {})
        if "secret" not in key:
            from ..keys import new_secret
            key["secret"] = new_secret(62)
        self.secret = int(key["secret"])
        self._edited_probe = False
        return {"secret": self.secret, "param": self.param}

    # ---- Imprint.edit ---------------------------------------------------------------------------------
    def imprint_edit(self, model, tokenizer, out):
        W = dict(model.named_parameters())[self.param]
        with torch.no_grad():
            W.add_(self._xi(W.shape, self.secret).to(W.device, W.dtype))
        wdir = os.path.join(out, "weights")
        model.save_pretrained(wdir, safe_serialization=True); tokenizer.save_pretrained(wdir)
        return wdir

    def save(self, out, key):
        return {"secret": self.secret, "param": self.param}   # xi is regenerated from the secret: no file

    # ---- Score (R2) -------------------------------------------------------------------------------------
    def _probe_at_watermark(self):
        """The detector differentiates at theta0 + xi: add xi once to the probe's fp32 base."""
        m = self.probe.model()
        if not self._edited_probe:
            W = dict(m.named_parameters())[self.param]
            with torch.no_grad():
                W.add_(self._xi(W.shape, self.secret).to(W.device, W.dtype))
            self._xi_dev = self._xi(W.shape, self.secret).to(W.device)
            self._edited_probe = True
        return m

    def score_batch(self, texts):
        if self.probe is None:
            raise RuntimeError("gaussmark scores with gradients (R2): detect through mwx.detect, or set scheme.probe")
        self._probe_at_watermark()
        out = []
        for t, gs in zip(texts, self.probe.grad_seq_logprob(list(texts), self.param)):
            g = gs[self.param].float()
            n = len(self.tok(t, add_special_tokens=False)["input_ids"])
            gn = g.norm().item()
            psi = (self._xi_dev * g).sum().item() / (self.sigma * gn) if gn > 0 else 0.0
            out.append((psi, n))
        return out

    def score(self, text):
        return self.score_batch([text])[0]

    def pvalue(self, statistic, n_tokens):
        return 0.5 * math.erfc(statistic / math.sqrt(2))      # P(N(0,1) >= psi)
