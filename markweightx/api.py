"""The public facade: the calls of the threat-model roles.

    import markweightx as mwx
    rel, key = mwx.embed("llama-3.2-1b", watermark="pro", out="runs/alice")         # owner
    rel, key = mwx.port("their_lora/", "pro", {"mapping": ...}, "llama-3.2-1b")      # owner: made elsewhere
    key      = mwx.calibrate(key, "llama-3.2-1b")                                    # owner: freeze the null
    edited   = mwx.modify(rel, "finetune:steps=800", out="runs/dana")               # downstream (weights)
    texts    = mwx.generate(edited, prompts)                                         # downstream (use)
    texts    = mwx.rewrite(texts, "synonym:ratio=0.3")                               # downstream (text)
    dets     = mwx.detect(texts, key)                                                # detector: text + key only
    verdict  = mwx.verify_model(edited, key, prompts, sequential=True)               # detector with query access
    dets     = mwx.detect(texts, key, null="wrong_key")                              # ... without a calibration
"""
from __future__ import annotations

import os

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


def port(model_dir: str, watermark: str, key: dict, base: str, out: str | None = None, *,
         key_public: bool = False, force: bool = False) -> tuple[Release, Key]:
    """Import a watermarked checkpoint made elsewhere, without retraining: its weights (a LoRA adapter directory or
    full weights) plus the key material it was made with, as a Release and a Key in the key store.
      key: the scheme's key fields, e.g. {"secret": 123} for a markllm-*-d scheme, {"mapping": "map.json",
           "transform": "policy.pth"} for PRO (files are copied into the key store).
      key_public: the key shipped with the checkpoint (e.g. a paper's released mapping): anyone can then detect, and
           also remove or forge the mark; the key records it and detect() warns.
    The release directory (default: next to the checkpoint, <model_dir>-mwx) gets only release.json; the weights stay
    where they are. Calibrate the key (mwx.calibrate) before detecting with an empirical null."""
    import os
    from .keys import store
    from .registry import resolved_params
    name, sparams, src = scheme_spec(watermark)
    sch = _get(name, src, sparams)
    model_dir = os.path.abspath(model_dir)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"no checkpoint directory {model_dir}")
    is_adapter = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
    if not is_adapter and not os.path.exists(os.path.join(model_dir, "config.json")):
        raise ValueError(f"{model_dir} is neither a LoRA adapter (adapter_config.json) nor a model (config.json)")
    params = dict(key)
    known = getattr(sch, "key_fields", None)
    if known is not None and set(params) - set(known):
        raise ValueError(f"{name} keys have no field {', '.join(sorted(set(params) - set(known)))}; "
                         f"they take: {', '.join(known)}")
    if name.startswith("pro") and "mapping" in params and "transform" not in params:
        import warnings
        warnings.warn("this PRO key has no 'transform': detection uses the public initial policy, which is right "
                      "only if the checkpoint was trained with it")
    import re
    from .keys import STORE, resolve
    for k, v in list(params.items()):                      # a key exported from a store: blob refs become its files
        if isinstance(v, str) and v.startswith("blob:"):
            if not re.fullmatch(r"blob:[0-9a-f]{64}(\.\w+)?", v):
                raise ValueError(f"key field {k}: {v!r} is not a blob reference")
            if not os.path.isfile(resolve(v)):
                raise FileNotFoundError(f"key field {k}: {v} is not in this key store ({STORE}); copy its blobs/ too")
            params[k] = resolve(v)
        elif isinstance(v, str) and not os.sep in v and not v.endswith((".json", ".pt", ".pth", ".bin", ".npy")):
            params[k] = _value(v)                          # "0.5" -> 0.5, so the commitment matches embed's
    for k, v in params.items():                            # key files are copied in; a missing one fails here
        if isinstance(v, str) and (v.endswith((".json", ".pt", ".pth", ".bin", ".npy")) or os.sep in v):
            if not os.path.isfile(v):
                raise FileNotFoundError(f"key field {k}: no file {v}")
            params[k] = os.path.abspath(v)
    if key_public:
        params["public"] = True
    out = out or model_dir.rstrip("/") + "-mwx"
    if os.path.exists(os.path.join(out, "release.json")) and not force:
        raise FileExistsError(f"{out} already holds a release; pass force=True (CLI --force) to replace it")
    scheme_side = resolved_params(sch)
    clash = {f: v for f, v in params.items()
             if f in scheme_side and scheme_side[f] != (_value(v) if isinstance(v, str) else v)}
    if clash:                                              # e.g. PRO's gamma: a scheme parameter, not key material
        what = "is a parameter" if len(clash) == 1 else "are parameters"
        raise ValueError(f"{', '.join(clash)} {what} of {name}, not key material: put it in the watermark, e.g. '{name}:"
                         + ",".join(f"{f}={v}" for f, v in clash.items()) + "'")
    k = Key(scheme=name, params=params, base=base, scheme_params=scheme_side, scheme_source=src)
    store(k)
    from .types import Step
    rel = Release(path=out, base=base, scheme=name, key_commitment=k.commitment,
                  adapter=model_dir if is_adapter else None, weights=None if is_adapter else model_dir,
                  lineage=[vars(Step("port", {"from": model_dir, "scheme": name, "key_public": key_public}))]).save()
    return rel, k


def _get(name, src, params):
    from .registry import get_scheme
    return get_scheme(name, source=src, **params)


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
    from .scoring import LENGTHS, cut
    lens = sorted(len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts)
    key.calibration = {"base": base, "null": sorted(s for s, _ in scores(sch, texts)), "n": len(texts),
                       "n_tokens": lens[len(lens) // 2],   # the null holds for texts of about this length
                       # C9: a detector's null can drift with length (PRO: 14% false positives at 50 tokens against a
                       # 200-token threshold), so short texts are judged against the null of texts cut as short
                       "by_length": {str(n): sorted(s for s, _ in scores(sch, cut(tok, texts, n))) for n in LENGTHS
                                     if n < lens[len(lens) // 2]}}
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
    if key.params.get("public"):
        import warnings
        warnings.warn(f"key {key.commitment[:12]} is public (shipped with its checkpoint): anyone can detect with it, "
                      f"and also remove or forge the mark")
    if null not in ("auto", "wrong_key"):
        raise ValueError(f"null must be 'auto' or 'wrong_key', not {null!r}")
    device = _detect_device(key, device, type(build_scheme(key, trust_plugin=trust_plugin)))
    if null == "wrong_key":
        return wrong_key_detect(texts, key, base=base, device=device, fpr=fpr, grant=grant, n_null=n_null,
                                trust_plugin=trust_plugin)
    return _detect(scheme_for(key, base, device, grant, trust_plugin), texts, null=key.calibration.get("null"), fpr=fpr,
                   by_length=key.calibration.get("by_length"), null_len=key.calibration.get("n_tokens"))


def wrong_key_detect(texts: list[str], key: Key, *, base: str | None = None, device: str | None = None, fpr=0.01,
                     grant=None, n_null: int = 200, trust_plugin: bool = False) -> list[Detection]:
    """p-values from wrong keys, as in MarkAgentX's wrong_key_detect: each text's statistic under the true key against
    its statistics under n_null wrong keys, p = (1 + #{wrong >= S}) / (n_null + 1). A wrong key is the true key with
    only its secret redrawn (Scheme.wrong_key), from random.Random(0xC0FFEE) so the null is reproducible; the true key
    is excluded. Needs no calibration and no base texts; costs n_null rebindings of the scheme. Teacher-free (EDIT)
    schemes are refused: a wrong key needs another release, and their closed-form null is exact."""
    import random
    import warnings
    from .primitives import Imprint
    from .scoring import scores
    if n_null < 1:
        raise ValueError(f"n_null must be at least 1 (got {n_null})")
    if 1 / (n_null + 1) > fpr:
        warnings.warn(f"with n_null={n_null} the smallest possible p-value is {1 / (n_null + 1):.3f} > fpr={fpr}: "
                      f"nothing can be flagged; use n_null >= {int(1 / fpr)}")
    sch = scheme_for(key, base, device or _detect_device(key, None), grant, trust_plugin)
    if sch.imprint == Imprint.EDIT:
        raise ValueError(f"{key.scheme} writes its mark into the weights: a wrong key needs another release. Use its "
                         f"closed-form null (exact_null) or mwx.calibrate instead")
    live = [i for i, t in enumerate(texts) if sch.tok(t, add_special_tokens=False)["input_ids"]]
    true = dict(zip(live, scores(sch, [texts[i] for i in live])))
    null = {i: [] for i in live}
    import shutil
    import tempfile
    rng = random.Random(0xC0FFEE)
    drawn = attempts = 0
    while drawn < n_null:
        attempts += 1
        if attempts > 3 * n_null + 10:
            raise ValueError(f"{key.scheme}: could not draw {n_null} wrong keys different from the true key")
        work = tempfile.mkdtemp(prefix="mwx-wrongkey-")    # this wrong key's files; nothing touches the key store
        try:
            params = sch.wrong_key(dict(key.params), rng, work)
            if params is None:
                raise ValueError(f"{key.scheme} has no secret to redraw (Scheme.wrong_key): use mwx.calibrate instead")
            if params == key.params:                       # the true key: excluded
                continue
            w = build_scheme(key, trust_plugin=trust_plugin)
            w.probe, w.config_dir = sch.probe, work
            w.setup(sch.model, sch.tok, sch.device, params)    # the model (or config) scheme_for bound
            for i, (s, _) in zip(live, scores(w, [texts[i] for i in live])):
                null[i].append(s)
            drawn += 1
        finally:
            shutil.rmtree(work, ignore_errors=True)
    out = []
    for i in range(len(texts)):
        if i not in true or not true[i][1]:                # nothing scored: a vacuous answer
            out.append(Detection(statistic=true[i][0] if i in true else 0.0, p_value=1.0, n_tokens=0, flagged=None))
            continue
        s, n = true[i]
        p = (1 + sum(x >= s for x in null[i])) / (n_null + 1)
        out.append(Detection(statistic=s, p_value=p, n_tokens=n, flagged=p < fpr))
    return out


def sequential_p(k: int, n: int, fpr: float, p1: float, null_ab: tuple | None = None) -> float:
    """An always-valid p-value after k flags in n texts: 1 / LR, LR the likelihood ratio of the flag sequence under
    the alternative against the null. The alternative is an even mixture of flag rates p1, 2*p1, 4*p1, ... (< 1), so
    a strongly marked model is caught however high its rate. The null is rate fpr (an exact null), or with
    null_ab = (a, b) the Beta-binomial predictive of an empirical null (a Polya urn: texts exchangeable with the
    calibration's). Either way LR is a martingale under the null, so P(1/LR ever <= x) <= x however often it is looked
    at (Ville)."""
    import math
    from scipy.special import betaln, logsumexp
    rates = [p1 * 2 ** i for i in range(8) if p1 * 2 ** i < 1]
    log_alt = logsumexp([k * math.log(r) + (n - k) * math.log(1 - r) for r in rates]) - math.log(len(rates))
    if null_ab:
        a, b = null_ab
        log_null = betaln(a + k, b + n - k) - betaln(a, b)
    else:
        log_null = k * math.log(fpr) + (n - k) * math.log(1 - fpr)
    return math.exp(min(0.0, log_null - log_alt))         # capped before exp: no overflow on long runs


def null_ab(n_null: int, fpr: float) -> tuple[int, int]:
    """For a threshold set at fpr on an empirical null of n_null texts: the Beta(a, b) law of its true false-flag rate
    (a = n_null - idx, b = idx + 1, idx its order statistic). A new unmarked text exceeds it with probability
    a / (a + b) (1.3% for 311 texts at fpr 1%), and k of n do with the Beta-binomial(n, a, b) law, exactly."""
    import math
    idx = max(0, math.ceil((1 - fpr) * n_null) - 1)
    return n_null - idx, idx + 1


def verify_model(release: Release, key: Key, prompts: list[str], *, device=DEFAULT_DEVICE, fpr=0.01, alpha=0.01,
                 budget: int | None = None, sequential: bool = False, p1: float = 0.2, batch: int = 20,
                 trust_plugin: bool = False) -> Detection:
    """Model-level verdict on a suspect model: sample it on the prompts (at most `budget`), flag each text at `fpr`,
    and test how many are flagged.
      fixed (default): p = the null's tail P(K >= k) over all n judged texts; flagged when p <= alpha.
      sequential=True: `batch` texts at a time. The model is accused early once an always-valid test against flag
        rate p1 is decisive (strong watermarks settle in a batch or two); it is never cleared early: otherwise all
        texts are generated and the fixed test decides, so weak watermarks (flag rates well below p1) keep the fixed
        test's power. Each part runs at alpha/2 and p_value is twice the deciding part's p, so the verdict is flagged
        exactly when p_value <= alpha, with false accusations <= alpha.
    Texts too short to be judged are left out of the test (statistic counts flags among the judged, n_judged);
    n_queries is how many texts were generated. With an exact null (GaussMark) the flags are Binomial(n, fpr). With an
    empirical null of N texts the suspect's unmarked texts are exchangeable with the calibration's, so the flags are
    Beta-binomial (null_ab), which accounts for the threshold's own uncertainty at no cost in alpha. That is exact for
    texts judged against a null of their own length; across the per-length nulls it is conservative when false
    positives fall with length, the same assumption as the length rule (C9; simulated sizes 0.1-0.5% at alpha 1%). p_value is
    the p of the deciding test (doubled in sequential mode), so flagged iff p_value <= alpha. Prompts disjoint from the
    calibration's are the clean choice."""
    from scipy.stats import betabinom, binom
    from .models import load_release, load_tokenizer
    from .sampling import generate as _gen
    from .scoring import detect as _detect
    if not 0 < fpr < p1 < 1:
        raise ValueError(f"need 0 < fpr < p1 < 1 (got fpr={fpr}, p1={p1})")
    if budget is not None and budget < 1:
        raise ValueError(f"budget must be at least 1 (got {budget})")
    sch = build_scheme(key, trust_plugin=trust_plugin)       # refuse an untrusted plugin before any sampling
    if not key.calibration.get("null") and sch.pvalue(0.0, 1) is None:
        raise ValueError("verify_model needs a calibrated key (mwx.calibrate) for this scheme")
    if key.params.get("public"):
        import warnings
        warnings.warn(f"key {key.commitment[:12]} is public: anyone could also have removed or forged this mark")
    device = device or _auto_device()
    prompts = prompts[:budget] if budget else prompts
    bound = scheme_for(key, None, _matching_device(key, device, type(sch)), trust_plugin=trust_plugin)
    model, tok = load_release(release, device), load_tokenizer(release.base)
    c = key.calibration
    ab = null_ab(len(c["null"]), fpr) if c.get("null") and not getattr(sch, "exact_null", False) else None
    tail = (lambda k, n: float(betabinom.sf(k - 1, n, *ab))) if ab else (lambda k, n: float(binom.sf(k - 1, n, fpr)))
    size = batch if sequential else max(len(prompts), 1)
    k = n = n_tok = asked = 0                              # asked: texts generated; n: texts judged
    for i in range(0, len(prompts), size):
        texts = _gen(model, tok, prompts[i:i + size], seed=i)
        asked += len(texts)
        dets = [d for d in _detect(bound, texts, null=c.get("null"), fpr=fpr, by_length=c.get("by_length"),
                                   null_len=c.get("n_tokens")) if d.flagged is not None]   # unjudged: not counted
        k += sum(d.flagged for d in dets); n += len(dets); n_tok += sum(d.n_tokens for d in dets)
        if sequential and n and 2 * sequential_p(k, n, fpr, p1, ab) <= alpha:  # accused early
            return Detection(statistic=float(k), p_value=2 * sequential_p(k, n, fpr, p1, ab), n_tokens=n_tok,
                             flagged=True, n_queries=asked, n_judged=n)
    p = tail(k, n) if n else 1.0
    if sequential:
        p = min(1.0, 2 * p)
    return Detection(statistic=float(k), p_value=p, n_tokens=n_tok, flagged=bool(n) and p <= alpha, n_queries=asked,
                     n_judged=n)
