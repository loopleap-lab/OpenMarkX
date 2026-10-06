"""GaussMark (Block et al., 2025): a teacher-free watermark written straight into the weights.

Imprint.edit: add xi ~ N(0, sigma^2 I), drawn from the key's secret, to one weight matrix W, and save the release in
              bf16 (the format users load). The realized change delta = W_saved - W_base goes into the key store.
Score (R2):   g = d/dW log p_theta0(y) at the base weights; psi = <delta, g> / (sigma_eff * ||g||), with
              sigma_eff = ||delta|| / sqrt(numel). For text that does not depend on delta, psi ~ N(0, 1) exactly
              (delta is Gaussian and independent of g), so the p-value is closed form. Text from the edited model
              correlates with delta, which raises psi.
Primitives:   Key (secret -> xi -> delta), Imprint.edit, Probe at R2 (gradients of the base), Score. Never Signal,
              never MarkLLM: the scheme that shows the SDK is a layer of its own (technical appendix §1.1).
Scale:        sigma is absolute. For Llama-3.2-1B, layer 10 up_proj has weight std ~0.017, so sigma = 1e-3 changes
              the matrix by ~6%; sweep sigma against perplexity.
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
    exact_null = True              # psi ~ N(0, 1) by construction: the closed form beats a finite empirical null

    def __init__(self, sigma: float = 1e-3, layer: int = 10, matrix: str = "mlp.up_proj", **params):
        super().__init__(sigma=sigma, layer=layer, matrix=matrix, **params)
        self.sigma, self.layer, self.matrix = float(sigma), int(layer), matrix

    @property
    def param(self) -> str:
        return f"model.layers.{self.layer}.{self.matrix}.weight"

    def setup(self, model, tokenizer, device, key=None):
        super().setup(model, tokenizer, device, key)
        key = dict(key or {})
        if "secret" not in key:
            from ..keys import new_secret
            key["secret"] = new_secret(62)
        # a released key pins everything that defines delta: never fall back to code defaults
        self.sigma = float(key.get("sigma", self.sigma)); self.layer = int(key.get("layer", self.layer))
        self.matrix = key.get("matrix", self.matrix)
        self.secret, self.delta_ref = int(key["secret"]), key.get("delta")
        self._delta = None
        return self._key()

    def _key(self):
        k = {"secret": self.secret, "sigma": self.sigma, "layer": self.layer, "matrix": self.matrix}
        if self.delta_ref:
            k["delta"] = self.delta_ref
        return k

    # ---- Imprint.edit ---------------------------------------------------------------------------------
    def imprint_edit(self, model, tokenizer, out):
        W = dict(model.named_parameters())[self.param]
        g = torch.Generator().manual_seed(self.secret)       # CPU generator: the same xi on any machine
        xi = torch.randn(W.shape, generator=g, dtype=torch.float32) * self.sigma
        with torch.no_grad():
            before = W.detach().to(torch.bfloat16).float().cpu()
            W.add_(xi.to(W.device, W.dtype))
        model.to(torch.bfloat16)                              # the release is what users load: bf16
        after = dict(model.named_parameters())[self.param].detach().float().cpu()
        from ..keys import stage
        path = stage("gaussmark-delta.pt")
        torch.save(after - before, path)                      # the realized change, rounding included
        self.delta_ref = path
        wdir = os.path.join(out, "weights")
        model.save_pretrained(wdir, safe_serialization=True); tokenizer.save_pretrained(wdir)
        return wdir

    def save(self, out, key):
        return self._key()

    # ---- Score (R2) -------------------------------------------------------------------------------------
    def _delta_on(self, device):
        if self._delta is None:
            from ..keys import resolve
            if not self.delta_ref:
                raise ValueError("this GaussMark key has no delta: it was not produced by mwx.embed")
            d = torch.load(resolve(self.delta_ref), map_location="cpu").float()
            self._delta = d.to(device)
            self._sigma_eff = d.norm().item() / math.sqrt(d.numel())
        return self._delta

    def score_batch(self, texts):
        if self.probe is None:
            raise RuntimeError("gaussmark scores with gradients (R2): detect through mwx.detect, or set scheme.probe")
        m = self.probe.model()                                # the base, fp32, never edited
        delta = self._delta_on(next(m.parameters()).device)
        out = []
        for t, gs in zip(texts, self.probe.grad_seq_logprob(list(texts), self.param)):
            g = gs[self.param].float()
            n = len(self.tok(t, add_special_tokens=False)["input_ids"])
            gn = g.norm().item()
            out.append(((delta * g).sum().item() / (self._sigma_eff * gn) if gn > 0 else 0.0, n))
        return out

    def score(self, text):
        return self.score_batch([text])[0]

    def pvalue(self, statistic, n_tokens):
        return 0.5 * math.erfc(statistic / math.sqrt(2))      # P(N(0,1) >= psi)
