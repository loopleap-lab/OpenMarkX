"""The public facade: five calls for the threat-model roles, plus verify_model.

    import markweightx as mwx
    rel, key = mwx.embed("llama-3.2-1b", watermark="pro", out="runs/alice")         # owner
    key      = mwx.calibrate(key, "llama-3.2-1b")                                    # owner: freeze the null
    edited   = mwx.modify(rel, "finetune:steps=800", out="runs/dana")               # downstream (weights)
    texts    = mwx.generate(edited, prompts)                                         # downstream (use)
    texts    = mwx.rewrite(texts, "synonym:ratio=0.3")                               # downstream (text)
    dets     = mwx.detect(texts, key)                                                # detector: text + key only
    verdict  = mwx.verify_model(edited, key, prompts)                                # detector with query access
"""
from __future__ import annotations

import math

from .types import Detection, Key, Release

def _auto_device() -> str:
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


DEFAULT_DEVICE = None   # None = the first GPU if there is one, else CPU


def _value(v: str):
    v = v.strip()
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        f = float(v)
    except ValueError:
        return v
    return int(f) if f.is_integer() and "." not in v and "e-" not in v.lower() else f


def scheme_spec(watermark: str) -> tuple[str, dict, str | None]:
    """'gaussmark:sigma=1e-3' -> (name, params, None); 'my_scheme.py' or 'my_scheme.py:name' or 'pkg.mod:name'
    (with params after a second ':') -> the plugin is imported and its source returned, so the key can find it."""
    import os
    from .registry import load_plugin, plugin_schemes
    head, _, rest = watermark.partition(":")
    if head.endswith(".py") or ("." in head and not os.path.exists(head) and ":" in watermark and "=" not in rest.split(":")[0]):
        src = os.path.abspath(head) if head.endswith(".py") else head
        mod = load_plugin(src).__name__
        new = plugin_schemes(mod)
        name_part, _, params_part = rest.partition(":")
        if "=" in name_part:                                # 'file.py:k=v' (no name given)
            name_part, params_part = "", name_part
        name = name_part or (new[0] if len(new) == 1 else None)
        if not name:
            raise ValueError(f"{head} registers {', '.join(new) + ': say which, e.g. ' + repr(head + ':' + new[0]) if new else 'no scheme: decorate the class with @mwx.register_scheme'}")
        params = _spec(f"x:{params_part}")[1] if params_part else {}
    else:
        (name, params), src = _spec(watermark), None
    from .registry import get_scheme
    get_scheme(name, source=src, **params)                  # unknown names or parameters fail here, before any work
    return name, params, src


def _spec(s: str | dict) -> tuple[str, dict]:
    """'finetune:steps=800,lr=1e-4' -> ('finetune', {'steps': 800, 'lr': 0.0001}); '8e2' -> 800, 'true' -> True."""
    if isinstance(s, dict):
        s = dict(s); return s.pop("op"), s
    op, _, rest = s.partition(":")
    params = {}
    for kv in filter(None, (x.strip() for x in rest.split(","))):
        if "=" not in kv:
            raise ValueError(f"edit spec {s!r}: expected key=value after ':', got {kv!r} (e.g. 'finetune:steps=800')")
        k, v = kv.split("=", 1)
        params[k.strip()] = _value(v)
    return op.strip(), params


def embed(base: str, watermark: str = "pro", out: str | None = None, *, device=DEFAULT_DEVICE, **recipe) -> tuple[Release, Key]:
    """Learn a watermark into a base model. Returns the public release and the private key (kept in the key store)."""
    from .imprint import embed as _embed
    from .keys import load_key
    from .paths import RUNS
    name, sparams, src = scheme_spec(watermark)            # "gaussmark:sigma=1e-3", or a plugin file
    tag = name + "".join(f"_{k}{v}" for k, v in sparams.items())
    rel = _embed(name, base, out or f"{RUNS}/releases/{base}/{tag}", device=device or _auto_device(),
                 scheme_params=sparams, scheme_source=src, **recipe)
    return rel, load_key(rel)


def calibrate(key: Key, base: str, prompts: list[str] | None = None, *, n=500, device=DEFAULT_DEVICE, seed=1) -> Key:
    """Freeze the null for this key: the detector's statistics on the unwatermarked base's continuations."""
    from .scoring import scores
    from .sampling import eval_prompts, generate as _gen
    from .keys import store
    from .models import load_base, load_tokenizer
    device = device or _auto_device()
    tok = load_tokenizer(base); model = load_base(base, device)
    prompts = prompts or eval_prompts(tok, n=n)
    texts = _gen(model, tok, prompts, adapter=False, seed=seed)
    del model
    sch = scheme_for(key, base, _detect_device(key, device))
    key.calibration = {"base": base, "null": sorted(s for s, _ in scores(sch, texts)), "n": len(texts)}
    store(key)
    return key


def modify(release: Release, edits: str | dict, out: str | None = None, *, device=DEFAULT_DEVICE) -> Release:
    """Apply a weight edit (finetune / merge / prune / quantize) to a release; returns the edited release."""
    from .edits import modify as _modify
    op, params = _spec(edits)
    if op == "merge" and "partner" not in params:
        raise ValueError("merge needs an explicit partner model, e.g. 'merge:partner=llama-3.2-1b-instruct,ratio=0.4'")
    if op == "quantize":
        params.setdefault("bits", 4)
    tag = op + "".join(f"_{k}{v}" for k, v in params.items() if k != "partner")
    return _modify(release, op, out or f"{release.path}/edits/{tag}", device=device or _auto_device(), **params)


def generate(release: Release, prompts: list[str], *, device=DEFAULT_DEVICE, new_tokens=200, seed=0) -> list[str]:
    from .sampling import generate as _gen
    from .models import load_release, load_tokenizer
    tok = load_tokenizer(release.base)
    return _gen(load_release(release, device or _auto_device()), tok, prompts, new_tokens=new_tokens, seed=seed)


def rewrite(texts: list[str], edits: str | dict, *, device=DEFAULT_DEVICE) -> list[str]:
    """Text edits through MarkLLM's text editors: 'synonym:ratio=0.3', 'word_deletion:ratio=0.3'."""
    from .edits import rewrite as _rewrite
    op, params = _spec(edits)
    return _rewrite(texts, op, device=device or _auto_device(), **params)


def build_scheme(key: Key):
    """The key's scheme object with its params (and plugin source), not yet bound to a model."""
    from .registry import get_scheme
    return get_scheme(key.scheme, source=key.scheme_source, **key.scheme_params)


def scheme_for(key: Key, base: str | None, device: str, grant=None):
    """Rebuild the key's scheme for detection, with a Probe at the granted access level (default: what the scheme
    declares). A grant below the scheme's need raises AccessDenied before anything is loaded."""
    from .models import ConfigOnly
    from .primitives import ORDER, AccessDenied, Probe
    from .registry import get_scheme
    base = base or key.base or key.calibration.get("base")
    if not base:
        raise ValueError("detect needs the base model name: pass base=, or use a key made by mwx.embed")
    sch = build_scheme(key)
    grant = grant or sch.access
    if ORDER[sch.access.name] > ORDER[grant.name]:
        raise AccessDenied(f"{key.scheme} detects with {sch.access.value}; granted only {grant.value}")
    probe = Probe(base, device, grant)
    model = probe.model() if sch.needs_model_to_detect else ConfigOnly(base, device)
    sch.probe = probe
    sch.setup(model, probe.tokenizer(), device, key.params)
    return sch


def _detect_device(key: Key, device: str | None) -> str:
    """The device type must match the one the key's randomness was drawn on (KGW-family green lists differ between
    CUDA and CPU generators). Keys made before this was recorded were all made on CUDA."""
    import torch
    want = key.params.get("rng_device", "cuda" if key.scheme.startswith("markllm-") else "cpu")
    if device is None:
        device = "cuda:0" if want == "cuda" else "cpu"
    if want == "cuda" and not str(device).startswith("cuda"):
        raise ValueError(f"key {key.commitment[:12]} was made on CUDA; its {key.scheme} detector must run on a CUDA "
                         f"device (got {device!r}): CPU and CUDA generators draw different green lists")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"key {key.commitment[:12]} needs a CUDA device to detect, and none is available")
    return device


def detect(texts: list[str] | str, key: Key, *, base: str | None = None, device: str | None = None,
           fpr=0.01, grant=None) -> list[Detection]:
    """Score texts with the key alone (plus the base model for detectors that need one). With a calibrated key,
    p-values and flags come from the frozen null. The device defaults to the one the key was made on."""
    from .scoring import detect as _detect
    texts = [texts] if isinstance(texts, str) else texts
    device = _detect_device(key, device)
    return _detect(scheme_for(key, base, device, grant), texts, null=key.calibration.get("null"), fpr=fpr)


def verify_model(release: Release, key: Key, prompts: list[str], *, device=DEFAULT_DEVICE, fpr=0.01) -> Detection:
    """Model-level verdict: sample the suspect model on the prompts and count the texts flagged at `fpr`. Under the
    null each text is flagged with probability fpr, so the model's p-value is the binomial tail P(Binom(n, fpr) >= k).
    Exact for schemes with an exact null (GaussMark); with an empirical null the per-text rate is only fpr up to the
    calibration's sampling error and its prompt distribution, so use the calibration prompts and a large n.
    Needs a calibrated key, unless the scheme has a closed-form p-value."""
    from scipy.stats import binom
    if not key.calibration.get("null") and build_scheme(key).pvalue(0.0, 1) is None:
        raise ValueError("verify_model needs a calibrated key (mwx.calibrate) for this scheme")
    dets = detect(generate(release, prompts, device=device or _auto_device()), key, fpr=fpr)
    k, n = sum(bool(d.flagged) for d in dets), len(dets)
    p = float(binom.sf(k - 1, n, fpr))
    return Detection(statistic=float(k), p_value=p, n_tokens=sum(d.n_tokens for d in dets), flagged=p < fpr)
