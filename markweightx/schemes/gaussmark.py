"""GaussMark (Block et al., 2025): a teacher-free watermark written straight into the weights.

Imprint.edit: add xi ~ N(0, sigma^2 I), drawn from the key's secret, to one weight matrix W, and save the release in
              bf16 (the format users load). The realized change delta = W_saved - W_base goes into the key store.
Score (R2):   g = d/dW log p_theta0(y) at the base weights; psi = <delta, g> / (sigma_eff * ||g||), with
              sigma_eff = ||delta|| / sqrt(numel). For text that does not depend on delta, psi ~ N(0, 1) up to the
              bf16 rounding in delta (delta is Gaussian and independent of g), so the p-value is closed form. Text
              from the edited model correlates with delta, which raises psi.
Key:          secret, sigma, layer, matrix, and sha256 of the base matrix and of delta: delta is rebuilt from the
              probe's base and checked, so a detector holding a different base snapshot fails instead of mis-scoring.
Primitives:   Key (secret -> xi -> delta), Imprint.edit, Probe at R2 (gradients of the base), Score. Never Signal,
              never MarkLLM: the scheme that shows the SDK is a layer of its own (technical appendix §1.1).
Scale:        sigma is absolute (layer-10 up_proj of Llama-3.2-1B has weight std ~0.017); sweep it against perplexity.
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

    # defaults from the Llama-3.2-1B sweep (dev plan, 2026-10-06): sigma 0.01 at layer 14 traces 89% of texts at
    # 1% FPR for +3% perplexity; at layer 10 the same sigma costs +30%, sigma 0.03 breaks the model
    def __init__(self, sigma: float = 0.01, layer: int = 14, matrix: str = "mlp.up_proj", **params):
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
        self.secret = int(key["secret"])
        self.dtype = key.get("dtype") or self._checkpoint_dtype(model)
        self.base_sha, self.delta_sha = key.get("base_sha256"), key.get("delta_sha256")
        self._delta = None
        return self._key()

    @staticmethod
    def _checkpoint_dtype(model) -> str:
        """The dtype the base checkpoint is stored in (the in-memory model may have been loaded in fp32)."""
        from transformers import AutoConfig
        try:
            d = AutoConfig.from_pretrained(model.config._name_or_path).torch_dtype
        except Exception:
            d = None
        d = d or torch.bfloat16
        return str(d).replace("torch.", "") if not isinstance(d, str) else d

    def _key(self):
        k = {"secret": self.secret, "sigma": self.sigma, "layer": self.layer, "matrix": self.matrix, "dtype": self.dtype}
        if self.delta_sha:
            k.update(base_sha256=self.base_sha, delta_sha256=self.delta_sha)
        return k

    @staticmethod
    def _sha(t: torch.Tensor) -> str:
        import hashlib
        return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()

    def _realize(self, W_base: torch.Tensor, dtype) -> torch.Tensor:
        """delta exactly as the release stores it: round(W_base + xi) - W_base in the release dtype."""
        g = torch.Generator().manual_seed(self.secret)       # CPU generator: the same xi on any machine
        xi = torch.randn(W_base.shape, generator=g, dtype=torch.float32) * self.sigma
        before = W_base.detach().float().cpu()
        return (before + xi).to(dtype).float() - before

    # ---- Imprint.edit ---------------------------------------------------------------------------------
    def imprint_edit(self, model, tokenizer, out):
        # the release keeps the base checkpoint's dtype (bf16 for Llama): that is what users load
        dtype = getattr(torch, self.dtype)
        W = dict(model.named_parameters())[self.param]
        base = W.detach().to(dtype).float().cpu()            # the base matrix as its checkpoint stores it
        delta = self._realize(base, dtype)
        with torch.no_grad():
            W.copy_((base + delta).to(W.device, W.dtype))
        model.to(dtype); model.config.torch_dtype = dtype
        self.base_sha, self.delta_sha = self._sha(base), self._sha(delta)
        wdir = os.path.join(out, "weights")
        model.save_pretrained(wdir, safe_serialization=True); tokenizer.save_pretrained(wdir)
        return wdir

    def save(self, out, key):
        return self._key()

    # ---- Score (R2) -------------------------------------------------------------------------------------
    def _delta_on(self, model):
        """Rebuild delta from the probe's base and check it against the key."""
        if self._delta is None:
            if not self.delta_sha:
                raise ValueError("this GaussMark key has no delta hash: it was not produced by mwx.embed")
            W = dict(model.named_parameters())[self.param]
            dtype = getattr(torch, self.dtype)
            base = W.detach().float().cpu().to(dtype).float()
            if self._sha(base) != self.base_sha:
                raise ValueError(f"the detector's base differs from the one this key was made on ({self.param} "
                                 "hash mismatch): load the same base snapshot")
            d = self._realize(base, dtype)
            if self._sha(d) != self.delta_sha:
                raise ValueError("rebuilt delta does not match the key (different torch RNG?)")
            self._delta = d.to(W.device)
            self._sigma_eff = d.norm().item() / math.sqrt(d.numel())
        return self._delta

    def score_batch(self, texts):
        if self.probe is None:
            raise RuntimeError("gaussmark scores with gradients (R2): detect through mwx.detect, or set scheme.probe")
        m = self.probe.model()                                # the base, fp32, never edited
        delta = self._delta_on(m)
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
