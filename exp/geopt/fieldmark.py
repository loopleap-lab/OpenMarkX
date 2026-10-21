"""A keyed watermark for fields predicted on 3D points (PRO for a physics surrogate), and its detector.

Mark: for a secret key, M random frequencies w_m (wavelengths in a chosen band), signs s_m and phases b_m give one
smooth pattern per output channel, g_c(x) = sum_m s_m cos(w_m . x + b_m), scaled to unit RMS on each geometry.
A marked field (in the model's normalized units) is y + eps * g(x): the "generation-time" rule a closed service would
apply to every output, which PRO then learns into the open model's weights.

Detector: needs only the points and the field (normalized with the published output mean/std). The field's own smooth
structure is removed first (least squares onto a fixed basis of long-wavelength functions of position), so what is
left is mostly fine-scale detail, where the mark lives. The statistic is the mean over channels of
corr(residual_c, g_c); the null is the same statistic under many wrong keys on the same field,
so z = (s_key - mean s_wrong) / std s_wrong, as KGW's z-test.
"""
from __future__ import annotations

import math

import torch

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


def output_pattern(ref, key: int, m: int = 32):
    """g [B, N, C]: unit-RMS keyed smooth functions of the (normalized) prediction ref [B, N, C] at each point."""
    g = torch.Generator().manual_seed(int(key))
    C = ref.shape[-1]
    W = (torch.randn(C, m, C, generator=g) / OUTPUT_SCALE).to(ref.device)
    b = (2 * math.pi * torch.rand(C, m, generator=g)).to(ref.device)
    s = (torch.randint(0, 2, (C, m), generator=g).float() * 2 - 1).to(ref.device)
    out = (torch.cos(torch.einsum("bnk,cmk->bncm", ref.float(), W) + b) * s).sum(-1)
    out = out - out.mean(1, keepdim=True)
    return out / out.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8)


def statistic(points, field, key: int, residual=None, ref=None, **kw):
    """Mean over channels of the correlation between each field channel's fine-scale residual and its keyed pattern:
    [B]. Pass `residual` (highpass of the field) to reuse it across keys."""
    if DOMAIN == "output":                              # field is the residual against ref; nothing to remove
        g, f = output_pattern(ref, key), field.float()
    else:
        g = pattern(points, key, field.shape[-1], **kw)
        f = highpass(points, field) if residual is None else residual
    f = f - f.mean(1, keepdim=True)
    f = f / f.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8)
    return (f * g).mean(1).mean(-1)


def zscore(points, field, key: int, n_wrong: int = 200, wrong_seed: int = 10**9, ref=None, **kw):
    """z of the true key's statistic against n_wrong other keys on the same fields: [B]. With DOMAIN "output", field
    is the residual (suspect output minus the publisher's unmarked output) and ref that unmarked output."""
    r = None if DOMAIN == "output" else highpass(points, field)
    s = statistic(points, field, key, residual=r, ref=ref, **kw)
    wrong = torch.stack([statistic(points, field, wrong_seed + j, residual=r, ref=ref, **kw) for j in range(n_wrong)], 0)
    return (s - wrong.mean(0)) / wrong.std(0).clamp_min(1e-12)


def mark(points, field_norm, key: int, eps: float, **kw):
    """The generation-time rule: add eps * g to a field in normalized units (g of the field itself for DOMAIN
    "output")."""
    g = output_pattern(field_norm, key) if DOMAIN == "output" else pattern(points, key, field_norm.shape[-1], **kw)
    return field_norm + eps * g.to(field_norm.dtype)
