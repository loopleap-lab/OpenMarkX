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

DEFAULT_DEVICE = "cuda:0"


def _spec(s: str | dict) -> tuple[str, dict]:
    """'finetune:steps=800,lr=1e-4' -> ('finetune', {'steps': 800, 'lr': 0.0001})."""
    if isinstance(s, dict):
        s = dict(s); return s.pop("op"), s
    op, _, rest = s.partition(":")
    params = {}
    for kv in filter(None, rest.split(",")):
        k, v = kv.split("=", 1)
        for cast in (int, float):
            try:
                v = cast(v); break
            except ValueError:
                pass
        params[k.strip()] = v
    return op.strip(), params


def embed(base: str, watermark: str = "pro", out: str | None = None, *, device=DEFAULT_DEVICE, **recipe) -> tuple[Release, Key]:
    """Learn a watermark into a base model. Returns the public release and the private key (kept in the key store)."""
    from .imprint import embed as _embed
    from .keys import load_key
    from .paths import RUNS
    rel = _embed(watermark, base, out or f"{RUNS}/releases/{base}/{watermark}", device=device, **recipe)
    return rel, load_key(rel)


def calibrate(key: Key, base: str, prompts: list[str] | None = None, *, n=500, device=DEFAULT_DEVICE, seed=1) -> Key:
    """Freeze the null for this key: the detector's statistics on the unwatermarked base's continuations."""
    from .scoring import scores
    from .sampling import eval_prompts, generate as _gen
    from .keys import store
    from .models import load_base, load_tokenizer
    from .registry import get_scheme
    tok = load_tokenizer(base); model = load_base(base, device)
    prompts = prompts or eval_prompts(tok, n=n)
    texts = _gen(model, tok, prompts, adapter=False, seed=seed)
    sch = get_scheme(key.scheme); sch.setup(model, tok, device, key.params)
    key.calibration = {"base": base, "null": sorted(s for s, _ in scores(sch, texts)), "n": len(texts)}
    store(key)
    return key


def modify(release: Release, edits: str | dict, out: str | None = None, *, device=DEFAULT_DEVICE) -> Release:
    """Apply a weight edit (finetune / merge / prune / quantize) to a release; returns the edited release."""
    from .edits import modify as _modify
    op, params = _spec(edits)
    if op == "merge" and "partner" not in params:
        params["partner"] = release.base + "-instruct"
    if op == "quantize":
        params.setdefault("bits", 4)
    return _modify(release, op, out or f"{release.path}/edits/{op}", device=device, **params)


def generate(release: Release, prompts: list[str], *, device=DEFAULT_DEVICE, new_tokens=200, seed=0) -> list[str]:
    from .sampling import generate as _gen
    from .models import load_release, load_tokenizer
    tok = load_tokenizer(release.base)
    return _gen(load_release(release, device), tok, prompts, new_tokens=new_tokens, seed=seed)


def rewrite(texts: list[str], edits: str | dict, *, device=DEFAULT_DEVICE) -> list[str]:
    """Text edits through MarkLLM's text editors: 'synonym:ratio=0.3', 'word_deletion:ratio=0.3'."""
    from .edits import rewrite as _rewrite
    op, params = _spec(edits)
    return _rewrite(texts, op, device=device, **params)


def _scheme_for(key: Key, base: str | None, device: str):
    from .models import ConfigOnly, load_base, load_tokenizer
    from .registry import get_scheme
    base = base or key.calibration.get("base")
    if not base:
        raise ValueError("detect needs the base name: calibrate the key first, or pass base=")
    sch = get_scheme(key.scheme)
    model = load_base(base, device) if sch.needs_model_to_detect else ConfigOnly(base, device)
    sch.setup(model, load_tokenizer(base), device, key.params)
    return sch


def detect(texts: list[str] | str, key: Key, *, base: str | None = None, device="cpu", fpr=0.01) -> list[Detection]:
    """Score texts with the key alone (plus the base model for detectors that need one). With a calibrated key,
    p-values and flags come from the frozen null."""
    from .scoring import detect as _detect
    texts = [texts] if isinstance(texts, str) else texts
    return _detect(_scheme_for(key, base, device), texts, null=key.calibration.get("null"), fpr=fpr)


def verify_model(release: Release, key: Key, prompts: list[str], *, device=DEFAULT_DEVICE) -> Detection:
    """Model-level verdict: sample the suspect model on the prompts, combine the per-text empirical p-values with
    Fisher's method into one p-value for the model."""
    if not key.calibration.get("null"):
        raise ValueError("verify_model needs a calibrated key (mwx.calibrate)")
    dets = detect(generate(release, prompts, device=device), key, device=device)
    stat = -2 * sum(math.log(d.p_value) for d in dets)
    from scipy.stats import chi2
    p = float(chi2.sf(stat, 2 * len(dets)))
    return Detection(statistic=stat, p_value=p, n_tokens=sum(d.n_tokens for d in dets), flagged=p < 0.01)
