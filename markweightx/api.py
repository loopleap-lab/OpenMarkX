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
    if head.endswith(".py") or "." in head:                # built-in names have no dots: a file or a module
        src = os.path.abspath(head) if head.endswith(".py") else head
        loaded = load_plugin(src)
        mod = loaded.__name__
        if not head.endswith(".py"):                       # pin a module plugin to its current code
            from .registry import _module_digest
            src = f"{head}@{_module_digest(loaded)}"
        new = plugin_schemes(mod)
        name_part, _, params_part = rest.partition(":")
        if "=" in name_part:                                # 'file.py:k=v' (no name given)
            name_part, params_part = "", name_part
        name = name_part or (new[0] if len(new) == 1 else None)
        if not name:
            raise ValueError(f"{head} registers {', '.join(new) + ': say which, e.g. ' + repr(head + ':' + new[0]) + ' (CLI check: --name)' if new else 'no scheme: decorate the class with @mwx.register_scheme'}")
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


def calibrate(key: Key, base: str, prompts: list[str] | None = None, *, n=500, device=DEFAULT_DEVICE, seed=1,
              trust_plugin: bool = False) -> Key:
    """Freeze the null for this key: the detector's statistics on the unwatermarked base's continuations."""
    from .scoring import scores
    from .sampling import eval_prompts, generate as _gen
    from .keys import store
    from .models import load_base, load_tokenizer
    device = device or _auto_device()
    cls = type(build_scheme(key, trust_plugin=trust_plugin))  # refuse an untrusted plugin before any sampling
    det_device = _matching_device(key, device, cls)         # sample on `device`, score on the key's device type
    tok = load_tokenizer(base); model = load_base(base, device)
    prompts = prompts or eval_prompts(tok, n=n)
    texts = _gen(model, tok, prompts, adapter=False, seed=seed)
    del model
    sch = scheme_for(key, base, det_device, trust_plugin=trust_plugin)
    key.calibration = {"base": base, "null": sorted(s for s, _ in scores(sch, texts)), "n": len(texts)}
    store(key)
    return key


def modify(release: Release, edits: str | dict, out: str | None = None, *, device=DEFAULT_DEVICE) -> Release:
    """Apply a weight edit (finetune / merge / prune / quantize) to a release; returns the edited release."""
    from .edits import modify as _modify
    op, params = _spec(edits)
    if "device" in params:
        raise ValueError("the device is not an edit parameter: pass device= (CLI --device)")
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
    from .edits.weights import check_params
    RATIO = {"ratio": (lambda v: 0 < v <= 1, "in (0, 1]")}
    op, params = _spec(edits)
    if "device" in params:
        raise ValueError("the device is not an edit parameter: pass device= (CLI --device)")
    check_params(_rewrite, op, params, {"synonym": RATIO, "word_deletion": RATIO, "context_synonym": RATIO},
                 skip=("texts", "kind", "device"))
    return _rewrite(texts, op, device=device or _auto_device(), **params)


def build_scheme(key: Key, *, trust_plugin: bool = False):
    """The key's scheme object with its params, not yet bound to a model. A key made with a plugin names the plugin's
    code; that code runs only if this process already imported it, or with trust_plugin=True (a key from someone
    else can name any code)."""
    from .registry import UntrustedPlugin, get_scheme, plugin_loaded
    if key.scheme_source and not trust_plugin and not plugin_loaded(key.scheme_source):
        raise UntrustedPlugin(f"this key runs plugin code {key.scheme_source}: import it yourself first, or pass "
                              f"trust_plugin=True (CLI --trust-plugin) if you trust whoever made the key")
    return get_scheme(key.scheme, source=key.scheme_source, **key.scheme_params)


def scheme_for(key: Key, base: str | None, device: str, grant=None, trust_plugin: bool = False):
    """Rebuild the key's scheme for detection, with a Probe at the granted access level (default: what the scheme
    declares). A grant below the scheme's need raises AccessDenied before anything is loaded."""
    from .models import ConfigOnly
    from .primitives import ORDER, AccessDenied, Probe
    from .registry import get_scheme
    base = base or key.base or key.calibration.get("base")
    if not base:
        raise ValueError("detect needs the base model name: pass base=, or use a key made by mwx.embed")
    sch = build_scheme(key, trust_plugin=trust_plugin)
    grant = grant or sch.access
    if ORDER[sch.access.name] > ORDER[grant.name]:
        raise AccessDenied(f"{key.scheme} detects with {sch.access.value}; granted only {grant.value}")
    probe = Probe(base, device, grant)
    model = probe.model() if sch.needs_model_to_detect else ConfigOnly(base, device)
    sch.probe = probe
    sch.setup(model, probe.tokenizer(), device, key.params)
    return sch


def _rng_device(key: Key, cls=None) -> str | None:
    """The device type a key's randomness was drawn on, or None if the scheme declares it device-independent
    (cls: the key's scheme class, e.g. type(build_scheme(key)); plugins are only known once loaded).
    Keys made before this was recorded: MarkLLM keys were all made on CUDA."""
    from .registry import SCHEMES
    cls = cls or SCHEMES.get(key.scheme)
    if cls is not None and not getattr(cls, "rng_bound", True):
        return None
    return key.params.get("rng_device", "cuda" if key.scheme.startswith("markllm-") else "cpu")


def _matching_device(key: Key, device: str, cls=None) -> str:
    """`device` if the key can be scored there, else the key's own device type."""
    want = _rng_device(key, cls)
    return device if want is None or str(device).split(":")[0] == want else _detect_device(key, None, cls)


def _detect_device(key: Key, device: str | None, cls=None) -> str:
    """Detect on the device type the key was made on: generators on CUDA and CPU draw different numbers (e.g.
    KGW-family green lists), so a mismatch would silently score with a different key."""
    import torch
    from . import schemes  # noqa: F401  (registers the built-ins, whose rng_bound _rng_device reads)
    want = _rng_device(key, cls)
    if device is None:
        device = "cuda:0" if want == "cuda" else "cpu"
    if want is not None and str(device).split(":")[0] != want:
        raise ValueError(f"key {key.commitment[:12]} drew its randomness on {want}; its {key.scheme} detector must "
                         f"run on {want} (got {device!r}): CPU and CUDA generators draw different numbers")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"key {key.commitment[:12]} needs a CUDA device to detect, and none is available")
    return device


def detect(texts: list[str] | str, key: Key, *, base: str | None = None, device: str | None = None,
           fpr=0.01, grant=None, trust_plugin: bool = False, null: str = "auto", n_null: int = 200) -> list[Detection]:
    """Score texts with the key alone (plus the base model for detectors that need one). With a calibrated key,
    p-values and flags come from the frozen null; otherwise from the scheme's closed form. null="wrong_key" instead
    scores each text under n_null wrong keys (MarkAgentX's wrong_key_detect): p = (1 + #{wrong >= S}) / (n_null + 1).
    The device defaults to the one the key was made on."""
    from .scoring import detect as _detect
    texts = [texts] if isinstance(texts, str) else texts
    if null not in ("auto", "wrong_key"):
        raise ValueError(f"null must be 'auto' or 'wrong_key', not {null!r}")
    device = _detect_device(key, device, type(build_scheme(key, trust_plugin=trust_plugin)))
    if null == "wrong_key":
        return wrong_key_detect(texts, key, base=base, device=device, fpr=fpr, grant=grant, n_null=n_null,
                                trust_plugin=trust_plugin)
    return _detect(scheme_for(key, base, device, grant, trust_plugin), texts, null=key.calibration.get("null"), fpr=fpr)


def wrong_key_detect(texts: list[str], key: Key, *, base: str | None = None, device: str | None = None, fpr=0.01,
                     grant=None, n_null: int = 200, trust_plugin: bool = False) -> list[Detection]:
    """p-values from wrong keys, as in MarkAgentX: each text's statistic under the true key against its statistics
    under n_null keys drawn reproducibly (random.Random(0xC0FFEE)), the true key excluded. Needs no calibration and
    no base texts; costs n_null rebindings of the scheme. Teacher-free (EDIT) schemes have an exact null instead."""
    import os
    from .imprint import _drop_if_empty
    from .keys import seeded_secrets
    from .primitives import Imprint
    from .scoring import scores
    sch = scheme_for(key, base, device or _detect_device(key, None), grant, trust_plugin)
    if sch.imprint == Imprint.EDIT:
        raise ValueError(f"{key.scheme} writes its mark into the weights: a wrong key needs another release. Use its "
                         f"closed-form null (exact_null) or mwx.calibrate instead")
    true = [s for s, _ in scores(sch, texts)]
    null = [[] for _ in texts]
    with seeded_secrets():
        drawn = 0
        while drawn < n_null:
            w = build_scheme(key, trust_plugin=trust_plugin)
            w.probe = sch.probe
            material = w.setup(sch.model, sch.tok, device, None)   # the model (or config) scheme_for bound
            for v in (material or {}).values():           # staged files of a wrong key (e.g. a PRO mapping)
                if isinstance(v, str) and os.path.isfile(v) and os.sep + "tmp" + os.sep in v:
                    os.remove(v); _drop_if_empty(os.path.dirname(v))
            if material == key.params:                     # the true key: excluded
                continue
            for i, (s, _) in enumerate(scores(w, texts)):
                null[i].append(s)
            drawn += 1
    out = []
    for s, ns, (_, n) in zip(true, null, scores(sch, texts)):
        p = 1.0 if not n else (1 + sum(x >= s for x in ns)) / (n_null + 1)
        out.append(Detection(statistic=s, p_value=p, n_tokens=n, flagged=bool(n) and p < fpr))
    return out


def verify_model(release: Release, key: Key, prompts: list[str], *, device=DEFAULT_DEVICE, fpr=0.01,
                 trust_plugin: bool = False) -> Detection:
    """Model-level verdict: sample the suspect model on the prompts and count the texts flagged at `fpr`. Under the
    null each text is flagged with probability fpr, so the model's p-value is the binomial tail P(Binom(n, fpr) >= k).
    Exact for schemes with an exact null (GaussMark); with an empirical null the per-text rate is only fpr up to the
    calibration's sampling error and its prompt distribution, so use the calibration prompts and a large n.
    Needs a calibrated key, unless the scheme has a closed-form p-value."""
    from scipy.stats import binom
    sch = build_scheme(key, trust_plugin=trust_plugin)       # refuse an untrusted plugin before any sampling
    if not key.calibration.get("null") and sch.pvalue(0.0, 1) is None:
        raise ValueError("verify_model needs a calibrated key (mwx.calibrate) for this scheme")
    device = device or _auto_device()
    dets = detect(generate(release, prompts, device=device), key, fpr=fpr, trust_plugin=trust_plugin,
                  device=_matching_device(key, device, type(sch)))
    k, n = sum(bool(d.flagged) for d in dets), len(dets)
    p = float(binom.sf(k - 1, n, fpr))
    return Detection(statistic=float(k), p_value=p, n_tokens=sum(d.n_tokens for d in dets), flagged=p < fpr)
