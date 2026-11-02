"""ripple: a keyed watermark for fields predicted on 3D points (OpenMark for a physics surrogate), and its detector.
Moved from exp/geopt/fieldmark.py (functions unchanged; scale/m_out kwargs default to the old globals).

Mark: for a secret key, M random frequencies w_m (wavelengths in a chosen band), signs s_m and phases b_m give one
smooth pattern per output channel, g_c(x) = sum_m s_m cos(w_m . x + b_m), scaled to unit RMS on each geometry.
A marked field (in the model's normalized units) is y + eps * g(x): the "generation-time" rule a closed service would
apply to every output, which OpenMark then learns into the open model's weights.

Detector: needs only the points and the field (normalized with the published output mean/std). The field's own smooth
structure is removed first (least squares onto a fixed basis of long-wavelength functions of position), so what is
left is mostly fine-scale detail, where the mark lives. The statistic is the mean over channels of
corr(residual_c, g_c); the null is the same statistic under many wrong keys on the same field,
so z = (s_key - mean s_wrong) / std s_wrong, as KGW's z-test.
"""
from __future__ import annotations

import math

import torch

from ..registry import register_scheme
from .base import FieldScheme

WAVELENGTH = (0.25, 0.75)      # the mark's band (geometry length 5); set before use to change it
DOMAIN = "output"              # "output": g of the model's own (normalized) prediction at the point, as KGW's green list
                               # depends on the context; "position": g of xyz; "normal": g of the surface normal
OUTPUT_SCALE = 1.0             # smoothness of g in output space (normalized units); larger = smoother, easier to learn
BASIS_MIN_WAVELENGTH = 2.0     # the detector removes field content at wavelengths above this


def pattern(points, key: int, channels: int, m: int = 64, wavelength=None):
    """g [B, N, C]: unit-RMS keyed patterns on points [B, N, 3] (geometry normalized to length 5, as GeoPT's data)."""
    g = torch.Generator().manual_seed(int(key))
    lo, hi = wavelength or WAVELENGTH
    lam = lo + (hi - lo) * torch.rand(channels, m, generator=g)                  # wavelength per component
    d = torch.randn(channels, m, 3, generator=g); d = d / d.norm(dim=-1, keepdim=True)
    w = (2 * math.pi / lam)[..., None] * d                                      # [C, M, 3]
    s = torch.randint(0, 2, (channels, m), generator=g).float() * 2 - 1
    b = 2 * math.pi * torch.rand(channels, m, generator=g)
    p = points[..., 4:7].float() if DOMAIN == "normal" else points[..., :3].float()
    phase = torch.einsum("bnk,cmk->bncm", p, w.to(p.device)) + b.to(p.device)   # [B, N, C, M]
    out = (torch.cos(phase) * s.to(p.device)).sum(-1)                           # [B, N, C]
    out = out - out.mean(1, keepdim=True)
    return out / out.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8)


def smooth_basis(points, n: int = 256, min_wavelength: float | None = None, seed: int = 0):
    """A fixed public basis of smooth functions of position [B, N, 2n + 4]: constant, linear, long-wavelength cos/sin."""
    g = torch.Generator().manual_seed(seed)
    p = points[..., :3].float()
    lam = (min_wavelength or BASIS_MIN_WAVELENGTH) * (1 + 4 * torch.rand(n, generator=g))
    d = torch.randn(n, 3, generator=g); d = d / d.norm(dim=-1, keepdim=True)
    ph = p @ ((2 * math.pi / lam)[:, None] * d).T.to(p.device)
    out = [torch.ones_like(p[..., :1]), p, torch.cos(ph), torch.sin(ph)]
    if DOMAIN == "normal" and points.shape[-1] >= 7:    # also remove what fields do smoothly with orientation
        nrm = points[..., 4:7].float()
        out += [nrm, (nrm[..., :, None] * nrm[..., None, :]).flatten(-2)]
    return torch.cat(out, -1)


def highpass(points, field, **kw):
    """The field minus its least-squares fit on smooth_basis (per geometry and channel)."""
    A = smooth_basis(points, **kw)                                              # [B, N, K]
    f = field.float()
    coef = torch.linalg.lstsq(A, f).solution if A.device.type == "cpu" else \
        torch.linalg.solve(A.transpose(1, 2) @ A + 1e-4 * torch.eye(A.shape[-1], device=A.device), A.transpose(1, 2) @ f)
    return f - A @ coef


def output_rule(key: int, channels: int, m: int = 32, scale: float | None = None):
    """The key's rule (W [C, m, C], b [C, m], s [C, m]): random frequencies, phases and signs drawn from the secret."""
    g = torch.Generator().manual_seed(int(key))
    W = torch.randn(channels, m, channels, generator=g) / (scale or OUTPUT_SCALE)
    b = 2 * math.pi * torch.rand(channels, m, generator=g)
    s = torch.randint(0, 2, (channels, m), generator=g).float() * 2 - 1
    return W, b, s


def rule_pattern(ref, W, b, s):
    """g [B, N, C]: unit-RMS smooth functions of the prediction ref [B, N, C] under the rule (W, b, s)."""
    out = (torch.cos(torch.einsum("bnk,cmk->bncm", ref.float(), W.to(ref.device)) + b.to(ref.device))
           * s.to(ref.device)).sum(-1)
    out = out - out.mean(1, keepdim=True)
    return out / out.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8)


def output_pattern(ref, key: int, m: int = 32, scale: float | None = None):
    """g [B, N, C]: unit-RMS keyed smooth functions of the (normalized) prediction ref [B, N, C] at each point."""
    return rule_pattern(ref, *output_rule(key, ref.shape[-1], m, scale))


def statistic(points, field, key: int, residual=None, ref=None, scale=None, m_out: int = 32, rule=None, **kw):
    """Mean over channels of the correlation between each field channel's fine-scale residual and its keyed pattern:
    [B]. Pass `residual` (highpass of the field) to reuse it across keys; `rule` (W, b, s) replaces the key's drawn
    rule (a co-learned rule)."""
    if DOMAIN == "output":                              # field is the residual against ref; nothing to remove
        g = rule_pattern(ref, *rule) if rule is not None else output_pattern(ref, key, m_out, scale)
        f = field.float()
    else:
        g = pattern(points, key, field.shape[-1], **kw)
        f = highpass(points, field) if residual is None else residual
    f = f - f.mean(1, keepdim=True)
    f = f / f.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8)
    return (f * g).mean(1).mean(-1)


def zscore(points, field, key: int, n_wrong: int = 200, wrong_seed: int = 10**9, ref=None, scale=None, m_out: int = 32,
           rule=None, **kw):
    """z of the true key's statistic against n_wrong other keys on the same fields: [B]. With DOMAIN "output", field
    is the residual (suspect output minus the publisher's unmarked output) and ref that unmarked output."""
    r = None if DOMAIN == "output" else highpass(points, field)
    s = statistic(points, field, key, residual=r, ref=ref, scale=scale, m_out=m_out, rule=rule, **kw)
    wrong = torch.stack([statistic(points, field, wrong_seed + j, residual=r, ref=ref, scale=scale, m_out=m_out, **kw)
                         for j in range(n_wrong)], 0)
    return (s - wrong.mean(0)) / wrong.std(0).clamp_min(1e-12)


def mark(points, field_norm, key: int, eps: float, **kw):
    """The generation-time rule: add eps * g to a field in normalized units (g of the field itself for DOMAIN
    "output")."""
    g = output_pattern(field_norm, key) if DOMAIN == "output" else pattern(points, key, field_norm.shape[-1], **kw)
    return field_norm + eps * g.to(field_norm.dtype)


@register_scheme
class RippleScheme(FieldScheme):
    """The output-keyed mark (DOMAIN "output"): target = ref + eps * g(ref); detection compares a suspect output with
    the publisher's unmarked output on the same inputs and z-tests against n_wrong wrong keys per case.

    colearn (OpenMark's co-adaptive policy, for fields): the rule (W, b, s) starts at the key's random draw and is
    trained together with the model, so it moves towards ripples the model can write; `reg` pulls it back towards the
    draw (L2 per element), which keeps it a keyed rule of the same family as the wrong keys. The learned rule is
    private key material (rule.pt in the key store). `channels`: the field's channel count (default: drawn lazily
    from the first output)."""
    name = "ripple"
    key_fields = ("secret",)

    def __init__(self, eps: float = 0.02, output_scale: float = 1.0, m: int = 32, n_wrong: int = 200,
                 colearn: bool = False, reg: float = 1.0, channels: int = 0):
        super().__init__(eps=eps, output_scale=output_scale, m=m, n_wrong=n_wrong, colearn=colearn, reg=reg,
                         channels=channels)
        self.rule = self.rule0 = None

    def setup(self, model, codec, device: str, key: dict | None = None) -> dict:
        import os
        self.model, self.codec, self.device = model, codec, device
        key = dict(key or {})
        key.setdefault("secret", int.from_bytes(os.urandom(4), "big") >> 1)
        self.secret = int(key["secret"])
        self.rule = self.rule0 = None
        if "rule" in key:                                   # the detector, or a second phase: the learned rule
            from ..keys import resolve
            d = torch.load(resolve(key["rule"]), weights_only=True)
            self.rule0 = tuple(d[k].float() for k in ("W0", "b0", "s0"))
            self.rule = tuple(torch.nn.Parameter(d[k].float().to(device), requires_grad=bool(self.params["colearn"]))
                              for k in ("W", "b", "s"))
        elif self.params["colearn"] and self.params["channels"]:
            self._init_rule(int(self.params["channels"]))
        return key

    def _init_rule(self, channels: int):
        p = self.params
        self.rule0 = output_rule(self.secret, channels, p["m"], p["output_scale"])
        self.rule = tuple(torch.nn.Parameter(t.clone().to(self.device), requires_grad=True) for t in self.rule0)

    def _pattern(self, ref):
        if self.params["colearn"] and self.rule is None:    # channel count known only now: draw the rule
            self._init_rule(ref.shape[-1])
        p = self.params
        return rule_pattern(ref, *self.rule) if self.rule is not None else \
            output_pattern(ref, self.secret, p["m"], p["output_scale"])

    def target_output(self, inputs, output):
        return output + self.params["eps"] * self._pattern(output).to(output.dtype)

    def trainable_params(self) -> list:
        return list(self.rule) if (self.params["colearn"] and self.rule is not None) else []

    def aux_loss(self):
        if not (self.params["colearn"] and self.rule is not None and self.rule[0].requires_grad):
            return None
        return self.params["reg"] * sum((t - t0.to(t.device)).pow(2).mean() for t, t0 in zip(self.rule, self.rule0))

    def save(self, key_dir: str, key: dict) -> dict:
        if self.rule is None:
            return key
        import os
        path = os.path.join(key_dir, "rule.pt")
        torch.save({**{k: t.detach().cpu() for k, t in zip(("W", "b", "s"), self.rule)},
                    **{k: t.detach().cpu() for k, t in zip(("W0", "b0", "s0"), self.rule0)}}, path)
        return {**key, "rule": path}

    def wrong_key(self, params: dict, rng, workdir: str) -> dict | None:
        """A wrong key: a fresh secret with its drawn (untrained) rule; the learned rule file is not carried over."""
        return {**{k: v for k, v in params.items() if k != "rule"}, "secret": rng.getrandbits(40)}

    def score(self, output, reference, inputs) -> tuple[float, int]:
        if tuple(output.shape) != tuple(reference.shape):
            raise ValueError(f"output {tuple(output.shape)} and reference {tuple(reference.shape)} must cover the same "
                             f"points: run both models on the same inputs")
        p = self.params
        rule = tuple(t.detach() for t in self.rule) if self.rule is not None else None
        z = zscore(inputs["points"] if isinstance(inputs, dict) else inputs, output.float() - reference.float(),
                   self.secret, n_wrong=p["n_wrong"], ref=reference.float(), scale=p["output_scale"], m_out=p["m"],
                   rule=rule)
        return float(z.mean()), int(output.shape[-2])

    def wrong_key_p(self, output, reference, inputs) -> float:
        """p-value against this key's wrong-key null on the same case: (1 + #{wrong >= true}) / (n_wrong + 1)."""
        p = self.params
        pts = inputs["points"] if isinstance(inputs, dict) else inputs
        res, ref = output.float() - reference.float(), reference.float()
        kw = dict(ref=ref, scale=p["output_scale"], m_out=p["m"])
        rule = tuple(t.detach() for t in self.rule) if self.rule is not None else None
        s = statistic(pts, res, self.secret, rule=rule, **kw).mean()
        wrong = torch.stack([statistic(pts, res, 10**9 + j, **kw).mean() for j in range(p["n_wrong"])])
        return float((1 + (wrong >= s).sum()) / (p["n_wrong"] + 1))

    def calibrate_strength(self, samples, z_target: float = 4.0,
                           grid=(0.0025, 0.005, 0.01, 0.02, 0.04, 0.08)) -> float:
        """The smallest eps in `grid` whose generation-time mark reaches z_target on every sample (inputs, ref);
        the largest if none does (as the calibrate step of the GeoPT runs)."""
        eps0 = self.params["eps"]
        try:
            for eps in sorted(grid):
                self.params["eps"] = eps
                if min(self.score(self.target_output(i, r), r, i)[0] for i, r in samples) > z_target:
                    return eps
            return max(grid)
        finally:
            self.params["eps"] = eps0
