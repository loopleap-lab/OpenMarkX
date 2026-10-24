"""openmarkx run CONFIG: one pipeline for every model adapter.

Steps, in order (each writes out/<step>.json, through a .tmp file renamed on success, and is skipped when that file
exists; once a step runs, every later step runs again, since its inputs may have changed):
  prepare    the training data and the evaluation inputs (or a small synthetic set: data: {name: synthetic})
  selfgen    the base model's own outputs to train on (taming: sampled image tokens; others bring their data)
  calibrate  the mark's strength, for field schemes with eps: auto (the smallest grid value detected on every case)
  imprint    learn the mark into the weights: text and agent through imprint.embed, vlm through embed_vlm, taming
             and geopt through train.imprint (phases); the release goes to out/release, the key to the key store
  generate   outputs of the base and of the release on the evaluation inputs (images also as PNGs in out/assets)
  detect     the detector's statistic on both sets; TPR at the FPR against the null (base outputs, or wrong keys)
  edits      the same after each edit (weight edits, an output edit such as image_roundtrip, a field fine-tune)
  quality    ppl | accuracy | fid | rel_l2 | none, before (base) and after (release)
  report     out/report.json
  viewer     out/assets/viewer.json: the run's steps, scores and one worked example, for the run viewer on the site
out/progress.jsonl gets one line per step event, for a viewer to follow the run.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field

import torch

STEPS = ("prepare", "selfgen", "calibrate", "imprint", "generate", "detect", "edits", "quality", "report", "viewer")
TOP = ("model", "scheme", "data", "imprint", "generate", "detect", "edits", "quality", "out", "smoke")
SECTIONS = {
    "scheme": ("name", "source", "params"),
    "imprint": ("phases", "bs", "accum", "seq_len", "answer_len", "save_every", "seed", "edit_last", "from"),
    "generate": ("n", "new_tokens", "bs"),
    "detect": ("fpr", "null", "n_null"),
}
NULLS = ("base", "wrong_keys")
# per adapter: the data keys it reads, the generate keys it reads, the imprint keys only some adapters read
DATA_KEYS = {"text": ("name", "train", "eval"), "agent": ("name", "train", "eval"), "vlm": ("name", "pairs", "eval"),
             "taming": ("classes", "per_class", "eval_classes", "eval_per_class"), "geopt": ()}
GENERATE_KEYS = {"text": ("n", "new_tokens", "bs"), "agent": ("n",), "vlm": ("n", "new_tokens", "bs"),
                 "taming": ("bs",), "geopt": ("n",)}
IMPRINT_ONLY = {"edit_last": ("geopt",), "seq_len": ("text", "agent"), "answer_len": ("vlm",),
                "accum": ("text", "agent", "vlm"), "bs": ("text", "agent", "vlm", "taming")}
QUALITY = ("ppl", "accuracy", "fid", "rel_l2", "none")
TEXT_FAMILY = ("text", "agent")
SMOKE_STEPS, SMOKE_N = 5, 4


class ConfigError(ValueError):
    """The run's config names something that does not exist."""


@dataclass
class Config:
    model: dict
    scheme: dict
    out: str
    data: dict = field(default_factory=dict)
    imprint: dict = field(default_factory=dict)
    generate: dict = field(default_factory=dict)
    detect: dict = field(default_factory=dict)
    edits: list = field(default_factory=list)
    quality: str = "none"
    smoke: bool = False


def _check_keys(where: str, d: dict, allowed) -> None:
    bad = sorted(map(str, set(d) - set(allowed)))
    if bad:
        raise ConfigError(f"unknown {where} key(s) {', '.join(bad)}; allowed: {', '.join(allowed)}")


def edit_kind(spec: str, adapter: str) -> str:
    """How an edit applies to this adapter's outputs: weights | text | output | field_finetune."""
    from . import edits  # noqa: F401  (registers the built-in edits)
    from .registry import EDITS, OUTPUT_EDITS, TEXT_EDITS
    name = spec.split(":")[0]
    if name == "text":
        if spec.split(":")[1] not in TEXT_EDITS:
            raise ConfigError(f"unknown text edit in {spec!r}; text edits: {', '.join(sorted(TEXT_EDITS))}")
        return "text"
    if name in OUTPUT_EDITS:
        return "output"
    if adapter in TEXT_FAMILY and name in EDITS:
        return "weights"
    if adapter == "geopt" and name == "finetune":
        return "field_finetune"
    have = sorted(OUTPUT_EDITS) + (sorted(EDITS) if adapter in TEXT_FAMILY else []) + \
        (["finetune"] if adapter == "geopt" else []) + [f"text:{t}" for t in sorted(TEXT_EDITS) if adapter in TEXT_FAMILY + ("vlm",)]
    raise ConfigError(f"unknown edit {name!r} for model {adapter}; edits for {adapter}: {', '.join(have)}")


def _env_vars() -> dict:
    """The environment for $VARS in configs: each OMX_ variable is also reachable by its old MWX_ name and the other
    way round, so configs written either way expand the same. $OMX_DATA (where the demo configs read their inputs)
    defaults to $OMX_RUNS (where they write)."""
    from .paths import OLD_PREFIX, PREFIX
    v = dict(os.environ)
    for k, val in list(v.items()):
        for a, b in ((OLD_PREFIX, PREFIX), (PREFIX, OLD_PREFIX)):
            if k.startswith(a):
                v.setdefault(b + k[len(a):], val)
    for p in (PREFIX, OLD_PREFIX):
        if p + "DATA" not in v and p + "RUNS" in v:
            v[p + "DATA"] = v[p + "RUNS"]
    return v


def _expand(x, env=None):
    """$VARS (or ${VARS}) in string values (paths), from the environment; unset ones stay as written."""
    import re
    env = _env_vars() if env is None else env
    if isinstance(x, dict):
        return {k: _expand(v, env) for k, v in x.items()}
    if isinstance(x, list):
        return [_expand(v, env) for v in x]
    if not isinstance(x, str):
        return x
    return re.sub(r"\$(\w+)|\$\{(\w+)\}", lambda m: env.get(m.group(1) or m.group(2), m.group(0)), x)


def config_diff(old: dict, new: dict) -> list[str]:
    """The settings two runs differ in (out aside). A scheme renamed since (pro -> openmark) counts as the same."""
    from .registry import canonical_scheme
    old = {k: v for k, v in old.items() if k != "out"}
    new = json.loads(json.dumps({k: v for k, v in new.items() if k != "out"}, default=str))
    for d in (old, new):
        if isinstance(d.get("scheme"), dict) and isinstance(d["scheme"].get("name"), str):
            d["scheme"] = {**d["scheme"], "name": canonical_scheme(d["scheme"]["name"])}
    return sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))


def load_config(path: str) -> Config:
    """Read and check a run config before anything is loaded: every unknown key, model, scheme, null, edit or
    quality name is an error that lists the allowed ones."""
    import yaml
    from .registry import list_models
    d = _expand(yaml.safe_load(open(path)) or {})
    if not isinstance(d, dict):
        raise ConfigError(f"{path}: a run config is a mapping of {', '.join(TOP)}")
    _check_keys("config", d, TOP)
    if isinstance(d.get("detect"), dict) and None in d["detect"]:
        d["detect"]["null"] = d["detect"].pop(None)          # YAML reads an unquoted `null:` key as None
    for k in ("model", "scheme", "out"):
        if k not in d:
            raise ConfigError(f"{path}: missing {k!r}")
    for sec, allowed in SECTIONS.items():
        _check_keys(sec, d.get(sec) or {}, allowed)
    if not (d["model"] or {}).get("source") and (d["model"] or {}).get("name") not in list_models():
        raise ConfigError(f"unknown model {d['model'].get('name')!r}; models: {', '.join(list_models())} "
                          f"(or model.source: a plugin file with @omx.register_model)")
    cfg = Config(**{k: v for k, v in d.items() if v is not None})
    from .train import _phases
    try:
        cfg.imprint["phases"] = _phases(cfg.imprint.get("phases") or [{}])
    except ValueError as e:
        raise ConfigError(str(e)) from None
    name = cfg.model["name"]
    if name in TEXT_FAMILY + ("vlm",) and len(cfg.imprint["phases"]) > 1:
        raise ConfigError(f"model {name} trains in one phase (imprint.embed / embed_vlm); got {len(cfg.imprint['phases'])}")
    if cfg.detect.get("null", "base") not in NULLS:
        raise ConfigError(f"unknown detect.null {cfg.detect['null']!r}; nulls: {', '.join(NULLS)}")
    if cfg.quality not in QUALITY:
        raise ConfigError(f"unknown quality {cfg.quality!r}; quality: {', '.join(QUALITY)}")
    for e in cfg.edits:
        edit_kind(e, name)
    from .registry import MODELS, SCHEMES, get_scheme, scheme_params
    if not cfg.scheme.get("source"):
        get_scheme("nowm")
        if cfg.scheme["name"] not in SCHEMES:
            raise ConfigError(f"unknown scheme {cfg.scheme['name']!r}; schemes: {', '.join(sorted(SCHEMES))} "
                              f"(or scheme.source: a plugin file)")
    sp = cfg.scheme.get("params") or {}
    if sp.get("eps") == "auto" and cfg.scheme["name"] == "ripple":
        raise ConfigError("ripple: eps: auto is not supported yet (on a noise-free output every eps is detected, so "
                          "the search would pick the smallest); set eps, e.g. 0.02 as the GeoPT demo")
    try:                                                    # the scheme's own parameter names, before any model loads
        get_scheme(cfg.scheme["name"], source=cfg.scheme.get("source"), **sp)
    except (ValueError, TypeError) as e:
        raise ConfigError(f"scheme {cfg.scheme['name']}: {e}") from None
    if not cfg.model.get("source"):                         # a built-in adapter: every setting must be one it reads
        allowed = ["name", "source"] + scheme_params(MODELS[name])
        _check_keys(f"model ({name})", cfg.model, allowed)
        _check_keys(f"data ({name})", cfg.data, DATA_KEYS[name])
        _check_keys(f"generate ({name})", cfg.generate, GENERATE_KEYS[name])
        for k in cfg.imprint:
            if k in IMPRINT_ONLY and name not in IMPRINT_ONLY[k]:
                raise ConfigError(f"imprint.{k} is not used by model {name} (only by: {', '.join(IMPRINT_ONLY[k])})")
    return cfg


# ---------------------------------------------------------------------------------------------------- run state
class Context:
    def __init__(self, cfg: Config, device: str):
        from .registry import get_model
        self.cfg, self.device, self.out = cfg, device, cfg.out
        mp = {k: v for k, v in cfg.model.items() if k not in ("name", "source")}
        # a model.source plugin is the config author's own file, as scheme.source
        self.adapter = get_model(cfg.model["name"], source=cfg.model.get("source"), trust=True, **mp)
        self.name = cfg.model["name"]
        self.rel_dir = os.path.join(self.out, "release")
        self._rel = self._key = None
        self._loaded = False

    # outputs of earlier steps
    def read(self, step: str) -> dict:
        p = os.path.join(self.out, f"{step}.json")
        if not os.path.exists(p):
            raise RuntimeError(f"step {step!r} has not run yet in {self.out}: run it first")
        return json.load(open(p))

    def tensors(self, step: str) -> dict:
        return torch.load(os.path.join(self.out, f"{step}.pt"), weights_only=False)

    def save_tensors(self, step: str, obj) -> None:
        p = os.path.join(self.out, f"{step}.pt")
        torch.save(obj, p + ".tmp"); os.replace(p + ".tmp", p)

    # settings
    def n(self, default: int) -> int:
        n = int(self.cfg.generate.get("n", default))
        return min(n, SMOKE_N) if self.cfg.smoke else n

    def phases(self) -> list[dict]:
        ph = [dict(p) for p in self.cfg.imprint["phases"]]
        if self.cfg.smoke:
            for p in ph:
                p["steps"] = min(p["steps"], SMOKE_STEPS)
        return ph

    def scheme_params(self) -> dict:
        return dict(self.cfg.scheme.get("params") or {})

    # the release and its key
    def release(self):
        from .types import Release
        if self._rel is None:
            self._rel = Release.open(self.read("imprint")["release"])
        return self._rel

    def key(self):
        from .keys import load_key
        if self._key is None:
            self._key = load_key(self.release())
        return self._key

    def ensure_loaded(self):
        """Adapters that hold external state (taming's wrapper, geopt's model and data) load it once."""
        if not self._loaded and self.name in ("taming", "geopt"):
            self.adapter.load(None, self.device)
        self._loaded = True

    def bound_scheme(self, net=None):
        """The key's scheme, ready to score this adapter's outputs."""
        from .api import build_scheme, scheme_for
        key = self.key()
        trust = bool(self.cfg.scheme.get("source"))      # the config's own plugin: its author is running it
        if self.name in TEXT_FAMILY:
            return scheme_for(key, self.release().base, self.device, trust_plugin=trust)
        sch = build_scheme(key, trust_plugin=trust)
        ctx = {"vlm": lambda: self.adapter.processor().tokenizer, "taming": lambda: self.adapter.codec(),
               "geopt": lambda: None}[self.name]()
        sch.setup(net, ctx, self.device, key.params)
        return sch


def _progress(out: str, **rec) -> None:
    with open(os.path.join(out, "progress.jsonl"), "a") as f:
        f.write(json.dumps({**rec, "time": round(time.time(), 1)}) + "\n")


def _write(path: str, obj) -> None:
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(path + ".tmp", path)


# ---------------------------------------------------------------------------------------------------- steps
def save_every(cfg: Config) -> int:
    """How often training saves its progress (HPG rule: about every 100 steps)."""
    return int(cfg.imprint.get("save_every", 100))


def in_chunks(path: str, items: list, size: int, fn) -> list:
    """fn(chunk, start) -> list of outputs, over `items` in chunks; each finished chunk is saved to `path`, so a
    job killed by its time limit redoes only the chunks it had not finished. `path` is removed at the end."""
    st = torch.load(path, weights_only=False) if os.path.exists(path) else {"done": 0, "outs": []}
    outs = list(st["outs"])
    for i in range(st["done"], len(items), size):
        outs += list(fn(items[i:i + size], i))
        torch.save({"done": min(i + size, len(items)), "outs": outs}, path + ".tmp")
        os.replace(path + ".tmp", path)
    if os.path.exists(path):
        os.remove(path)
    return outs


def _synthetic_corpus(ctx, tok) -> tuple[str, list[str]]:
    seq = int(ctx.cfg.imprint.get("seq_len", 128))
    g = torch.Generator().manual_seed(0)
    train = torch.randint(1000, 20000, (64, seq + 1), generator=g)
    prompts = [tok.decode(torch.randint(1000, 20000, (8,), generator=g).tolist()) for _ in range(ctx.n(8))]
    d = os.path.join(ctx.out, "data"); os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "corpus.pt")
    torch.save({"train": train, "eval_texts": prompts}, path)
    return path, prompts


def step_prepare(ctx) -> dict:
    data, name = ctx.cfg.data, ctx.name
    synthetic = data.get("name") == "synthetic"
    if name in TEXT_FAMILY:
        tok = ctx.adapter.tokenizer()
        if synthetic:
            corpus, prompts = _synthetic_corpus(ctx, tok)
        else:
            corpus = data["train"]
            prompts = None
        if name == "text":
            if prompts is None:
                from .sampling import eval_prompts
                prompts = eval_prompts(tok, n=ctx.n(500), data=data.get("eval") or corpus)
            return {"corpus": corpus, "prompts": prompts[:ctx.n(500)]}
        if synthetic:
            items = [{"question": f"What colour is item {i}?", "answer": "blue",
                      "paragraphs": {"Alpha": f"Item {i} is blue.", "Beta": "Beta is red."}} for i in range(ctx.n(8))]
        else:
            items = [json.loads(l) for l in open(data["eval"])][:ctx.n(300)]
        return {"corpus": corpus, "items": items}
    if name == "vlm":
        if not synthetic:
            items = [json.loads(l) for l in open(data["eval"])][:ctx.n(300)]
            items = [{"images": it.get("images") or [it["image"]], "question": it.get("question") or it["text"],
                      "answer": it.get("answer", it.get("label", ""))} for it in items]
            return {"pairs": data["pairs"], "items": items}
        from PIL import Image
        d = os.path.join(ctx.out, "data"); os.makedirs(d, exist_ok=True)
        A = int(ctx.cfg.imprint.get("answer_len", 128))
        rng = random.Random(0)
        rows, items = [], []
        for i in range(16):
            img = os.path.join(d, f"img{i}.png")
            Image.new("RGB", (28, 28), (rng.randrange(256),) * 3).save(img)
            rows.append({"images": [img], "text": f"what is shown {i}", "answer_ids": [rng.randrange(1, 290) for _ in range(A)]})
            if i < ctx.n(8):
                items.append({"image": img, "question": f"what is shown {i}", "answer": "1"})
        pairs = os.path.join(d, "pairs.jsonl")
        with open(pairs, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        return {"pairs": pairs, "items": items}
    if name == "taming":
        per, eper = int(data.get("per_class", 20)), int(data.get("eval_per_class", 10))
        if ctx.cfg.smoke:
            per, eper = min(per, 2), min(eper, 2)
        cls = data.get("classes", list(range(1000)))
        cls = list(range(int(cls.split("-")[0]), int(cls.split("-")[1]) + 1)) if isinstance(cls, str) else list(cls)
        ecls = list(data.get("eval_classes", cls[:10]))
        if ctx.cfg.smoke:
            cls, ecls = cls[:2], ecls[:2]
        return {"train_jobs": [c for c in cls for _ in range(per)], "eval_jobs": [c for c in ecls for _ in range(eper)]}
    if name == "geopt":
        ctx.ensure_loaded()
        train, test = ctx.adapter.splits()
        return {"n_train": len(train), "n_test": min(len(test), ctx.n(len(test)))}
    raise ConfigError(f"no prepare step for model {name}")


def step_selfgen(ctx) -> dict:
    if ctx.cfg.imprint.get("from"):
        return {"skipped": "the trained model is imported (imprint.from): nothing to train on"}
    if ctx.name != "taming":
        return {"skipped": "this model trains on the data prepare gives it"}
    ctx.ensure_loaded()
    jobs = ctx.read("prepare")["train_jobs"]
    net = ctx.adapter.load(None, ctx.device)
    bs = int(ctx.cfg.generate.get("bs", 25))
    rows = in_chunks(os.path.join(ctx.out, "selfgen.partial.pt"), jobs, bs * 20,
                     lambda chunk, start: list(ctx.adapter.run(net, chunk, seed=start + 1, bs=bs)))
    codes = torch.stack(rows)
    ctx.save_tensors("selfgen", {"seqs": torch.cat([torch.tensor(jobs)[:, None], codes], 1)})
    return {"n": len(jobs)}


def _field_samples(ctx, net):
    _, test = ctx.adapter.splits()
    test = test[:ctx.read("prepare")["n_test"]]
    outs = ctx.adapter.run(net, [(p, c) for p, c, _ in test])
    return test, outs


def step_calibrate(ctx) -> dict:
    from .schemes.base import FieldScheme
    from .registry import get_scheme
    params = ctx.scheme_params()
    sch = get_scheme(ctx.cfg.scheme["name"], source=ctx.cfg.scheme.get("source"),
                     **{k: v for k, v in params.items() if v != "auto"})
    if not isinstance(sch, FieldScheme) or params.get("eps") != "auto":
        return {"skipped": "the strength comes from the scheme's params", "eps": params.get("eps")}
    ctx.ensure_loaded()
    sch.setup(None, None, ctx.device, {"secret": 0})
    test, outs = _field_samples(ctx, ctx.adapter.load(None, ctx.device))
    eps = sch.calibrate_strength([({"points": p}, o) for (p, _, _), o in zip(test, outs)])
    return {"eps": eps}


def _import_release(ctx, src: str) -> dict:
    """imprint.from: reuse a model trained earlier instead of training again. A release (imprint.embed, embed_vlm,
    train.imprint) keeps its weights and its key in the key store; a Taming co-learning run directory (transformer.pt +
    policy.pt, as exp/arimg/pro_taming.py writes) becomes a state_dict release and a wmar key."""
    from .keys import stage_dir, store
    from .registry import get_scheme, resolved_params
    from .types import Key, Release, Step
    lin = [vars(Step("import", {"from": os.path.abspath(src)}))]
    if os.path.exists(os.path.join(src, "release.json")):
        r = Release.open(src)
        from .keys import load_key
        load_key(r)                                         # the key must be in this key store, or detection is impossible
        spec = r.model if r.model.get("name", "text") != "text" or ctx.name == "text" else \
            {"name": ctx.name, "params": ctx.adapter.params}
        Release(path=ctx.rel_dir, base=r.base, scheme=r.scheme, key_commitment=r.key_commitment, adapter=r.adapter,
                weights=r.weights, load=r.load, model=spec, format=r.format, lineage=list(r.lineage) + lin).save()
        return {"release": ctx.rel_dir, "imported": os.path.abspath(src)}
    pol, w = os.path.join(src, "policy.pt"), os.path.join(src, "transformer.pt")
    if ctx.name != "taming" or not (os.path.exists(pol) and os.path.exists(w)):
        raise ConfigError(f"imprint.from {src}: expected a release directory (release.json) or, for taming, a run "
                          f"directory with transformer.pt and policy.pt")
    import shutil
    sch = get_scheme(ctx.cfg.scheme["name"], source=ctx.cfg.scheme.get("source"), **ctx.scheme_params())
    kdir = stage_dir()
    shutil.copyfile(pol, os.path.join(kdir, "policy.pt"))  # key material: copied into the key store, not the release
    k = Key(scheme=sch.name, params={"policy": os.path.join(kdir, "policy.pt")}, base=ctx.name,
            scheme_params=resolved_params(sch))
    store(k)
    from .imprint import _drop_if_empty
    _drop_if_empty(kdir)
    Release(path=ctx.rel_dir, base=ctx.name, scheme=sch.name, key_commitment=k.commitment, weights=os.path.abspath(w),
            format="state_dict", model={"name": ctx.name, "params": ctx.adapter.params}, lineage=lin).save()
    return {"release": ctx.rel_dir, "imported": os.path.abspath(src)}


def step_imprint(ctx) -> dict:
    from .registry import get_scheme
    from .types import Release
    from .train import draft_key, resuming, save_draft
    if os.path.exists(os.path.join(ctx.rel_dir, "release.json")) and not resuming(ctx.rel_dir):
        Release.open(ctx.rel_dir)                           # finished earlier (killed before imprint.json was written)
        return {"release": ctx.rel_dir}
    if ctx.cfg.imprint.get("from"):
        return _import_release(ctx, ctx.cfg.imprint["from"])
    ph, im = ctx.phases(), ctx.cfg.imprint
    sp, src = ctx.scheme_params(), ctx.cfg.scheme.get("source")
    eps = ctx.read("calibrate").get("eps")
    if sp.get("eps") == "auto":
        sp["eps"] = eps
    every = save_every(ctx.cfg)
    if ctx.name in TEXT_FAMILY:
        from .imprint import embed
        p = ph[0]
        rel = embed(ctx.cfg.scheme["name"], ctx.adapter.base, ctx.rel_dir, steps=p["steps"], bs=int(im.get("bs", 16)),
                    accum=int(im.get("accum", 1)), seq_len=int(im.get("seq_len", 128)), lr=p["lr"], rank=p["rank"],
                    device=ctx.device, data=ctx.read("prepare")["corpus"], save_every=every,
                    seed=int(im.get("seed", 0)), scheme_params=sp, scheme_source=src)
        if ctx.name == "agent":
            rel.model = {"name": "agent", "params": ctx.adapter.params}
            rel.save()
    elif ctx.name == "vlm":
        p = ph[0]
        ctx.adapter.imprint(ctx.cfg.scheme["name"], ctx.rel_dir, ctx.read("prepare")["pairs"], device=ctx.device,
                            steps=p["steps"], bs=int(im.get("bs", 8)), accum=int(im.get("accum", 1)),
                            answer_len=int(im.get("answer_len", 128)), lr=p["lr"], rank=p["rank"],
                            save_every=every, seed=int(im.get("seed", 0)), scheme_params=sp, scheme_source=src)
    else:
        from .train import imprint
        ctx.ensure_loaded()
        sch = get_scheme(ctx.cfg.scheme["name"], source=src, **sp)
        net = ctx.adapter.load(None, ctx.device)
        bs = int(im.get("bs", 8))
        if ctx.name == "taming":
            seqs = ctx.tensors("selfgen")["seqs"]
            order = torch.randperm(len(seqs), generator=torch.Generator().manual_seed(int(im.get("seed", 0))))
            batches = [{"seqs": seqs[order[i:i + bs]]} for i in range(0, len(seqs), bs)]
            # a resumed run keeps the key it started with (its draft waits in the key store)
            key = sch.setup(net, ctx.adapter.codec(), ctx.device, draft_key(ctx.rel_dir) or {})
            teacher = None
        else:
            train, _ = ctx.adapter.splits()
            batches = [{"points": p, "cond": c} for p, c, _ in train]
            key = sch.setup(None, None, ctx.device, draft_key(ctx.rel_dir) or {})
            teacher = None
            if im.get("edit_last"):                         # closed-form edit of the last layer, then the phases,
                import copy                                 # whose targets come from the unedited model
                teacher = copy.deepcopy(net)
                _progress(ctx.out, step="imprint", event="edit_last", **ctx.adapter.edit_last(net, sch, train))
        save_draft(ctx.rel_dir, key)
        imprint(ctx.adapter, sch, batches, ctx.rel_dir, phases=ph, device=ctx.device, net=net, key=key, teacher=teacher,
                base=ctx.name, save_every=every, seed=int(im.get("seed", 0)))
    return {"release": ctx.rel_dir}


def _text_generate(ctx, model, adapter_on: bool, seed: int):
    from .sampling import generate
    g = ctx.cfg.generate
    return generate(model, ctx.adapter.tokenizer(), ctx.read("prepare")["prompts"], new_tokens=int(g.get("new_tokens", 200)),
                    bs=int(g.get("bs", 50)), seed=seed, adapter=adapter_on)


def _agent_run(ctx, model, adapter_on: bool, seed: int, transcripts: bool = False):
    finals, convs = ctx.adapter.run(model, ctx.read("prepare")["items"], adapter=adapter_on, seed=seed, transcript=True)
    return (finals, list(ctx.adapter.last_finish), convs) if transcripts else (finals, list(ctx.adapter.last_finish))


def step_generate(ctx) -> dict:
    from .models import load_release
    rel, g = ctx.release(), ctx.cfg.generate
    if ctx.name == "text":
        model = load_release(rel, ctx.device)
        return {"base": _text_generate(ctx, model, False, 1), "wm": _text_generate(ctx, model, True, 0)}
    if ctx.name == "agent":
        model = load_release(rel, ctx.device)
        items = ctx.read("prepare")["items"]

        def agent_chunks(name, adapter_on, seed):           # 50 questions per chunk, each chunk saved
            def one(chunk, start):
                finals, convs = ctx.adapter.run(model, chunk, adapter=adapter_on, seed=seed * 1000 + start, transcript=True)
                return list(zip(finals, ctx.adapter.last_finish, convs))
            rows = in_chunks(os.path.join(ctx.out, f"generate_{name}.partial.pt"), items, 50, one)
            return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]
        (bf, bfin, _), (wf, wfin, convs) = agent_chunks("base", False, 1), agent_chunks("wm", True, 0)
        return {"base": bf, "wm": wf, "base_finish": bfin, "wm_finish": wfin, "wm_transcript_0": convs[0] if convs else []}
    if ctx.name == "vlm":
        net = ctx.adapter.load(rel, ctx.device)
        items = ctx.read("prepare")["items"]
        kw = dict(new_tokens=int(g.get("new_tokens", 768)), bs=int(g.get("bs", 16)))
        run = lambda name, on, seed: in_chunks(os.path.join(ctx.out, f"generate_{name}.partial.pt"), items, kw["bs"] * 4,
                                               lambda chunk, start: ctx.adapter.run(net, chunk, adapter=on, seed=seed * 1000 + start, **kw))
        return {"base": run("base", False, 1), "wm": run("wm", True, 0)}
    ctx.ensure_loaded()
    base, student = ctx.adapter.load(None, ctx.device), ctx.adapter.load(rel, ctx.device)
    if ctx.name == "taming":
        from .models.taming import save_pngs
        jobs = ctx.read("prepare")["eval_jobs"]
        bs = int(g.get("bs", 25))
        gen = lambda name, net, seed: torch.stack(in_chunks(
            os.path.join(ctx.out, f"generate_{name}.partial.pt"), jobs, bs * 20,
            lambda chunk, start: list(ctx.adapter.run(net, chunk, seed=seed * 1000 + start, bs=bs))))
        sets = {"base": gen("base", base, 7), "wm": gen("wm", student, 9)}
        c = ctx.adapter.codec()
        for k, d in (("base", "base"), ("wm", "student")):
            save_pngs(c.decode(sets[k]), os.path.join(ctx.out, "assets", d))
        ctx.save_tensors("generate", {**sets, "jobs": torch.tensor(jobs)})
        return {"n": len(jobs), "assets": os.path.join(ctx.out, "assets")}
    test, base_out = _field_samples(ctx, base)
    wm_out = ctx.adapter.run(student, [(p, c) for p, c, _ in test])
    ctx.save_tensors("generate", {"base": base_out, "wm": wm_out})
    return {"n": len(test)}


def _stats(ctx, sch, outs, base_outs=None) -> list[float]:
    """The detector's statistic on each output (fields: against the unmarked model's output on the same case)."""
    if ctx.name == "geopt":
        _, test = ctx.adapter.splits()
        return [sch.score(o, r, {"points": p})[0] for o, r, (p, _, _) in zip(outs, base_outs, test)]
    if ctx.name == "taming":
        return [sch.score(c)[0] for c in outs]
    from .scoring import scores
    return [s for s, _ in scores(sch, list(outs))]


def _wrong_key_stats(ctx, outs, base_outs) -> list[float]:
    """The null from wrong keys: the same statistic on the release's outputs under other secrets."""
    from .api import build_scheme
    key = ctx.key()
    n = int(ctx.cfg.detect.get("n_null", 200))
    per = max(1, math.ceil(n / max(len(outs), 1)))
    if ctx.cfg.smoke:
        per = min(per, 2)
    import shutil
    import tempfile
    rng = random.Random(0xC0FFEE)
    true = ctx.bound_scheme()
    work = tempfile.mkdtemp(prefix="omx-wrongkey-")         # wrong keys' files; nothing touches the key store
    out = []
    try:
        for _ in range(per):
            params = true.wrong_key(dict(key.params), rng, work)
            if params is None:
                raise ConfigError(f"{key.scheme} cannot draw a wrong key (Scheme.wrong_key); use detect.null: base")
            sch = build_scheme(key, trust_plugin=bool(ctx.cfg.scheme.get("source")))
            sch.setup(None, ctx.adapter.codec() if ctx.name == "taming" else None, ctx.device, params)
            out += _stats(ctx, sch, outs, base_outs)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out


def _sets(ctx) -> tuple:
    if ctx.name in ("taming", "geopt"):
        t = ctx.tensors("generate")
        return t["wm"], t["base"]
    g = ctx.read("generate")
    return g["wm"], g["base"]


def step_detect(ctx) -> dict:
    from .scoring import summary
    ctx.ensure_loaded()
    sch = ctx.bound_scheme()
    wm, base = _sets(ctx)
    wm_s = _stats(ctx, sch, wm, base)
    null = ctx.cfg.detect.get("null", "geopt" == ctx.name and "wrong_keys" or "base")
    if null == "wrong_keys":
        null_s = _wrong_key_stats(ctx, wm, base)
    else:                                                    # base (or a derivative's) outputs
        null_s = _stats(ctx, sch, base, base)
    if ctx.name == "taming":                                 # the key keeps the null for detect --model taming
        from .keys import store
        from .edits.outputs import image_roundtrip
        key = ctx.key()
        key.calibration = {"null": sorted(_stats(ctx, sch, image_roundtrip(base, model=ctx.adapter))),
                           "route": "image", "n": len(base)}
        store(key)
    return {**summary(wm_s, null_s, float(ctx.cfg.detect.get("fpr", 0.01))), "null_kind": null,
            "wm": wm_s, "null": null_s}


def _field_finetune(ctx, steps: int = 100, lr: float = 1e-4, **_) -> list:
    """A downstream fine-tune of the released field model on the training cases' true fields (MSE)."""
    net = ctx.adapter.load(ctx.release(), ctx.device)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    train, test = ctx.adapter.splits()
    for s in range(int(steps)):
        p, c, y = train[s % len(train)]
        loss = torch.nn.functional.mse_loss(ctx.adapter.forward(net, {"points": p, "cond": c})[1].float(), y.float())
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return ctx.adapter.run(net, [(p, c) for p, c, _ in test[:ctx.read("prepare")["n_test"]]])


def _params(spec: str) -> dict:
    from .api import _spec
    return _spec(spec)[1]


def step_edits(ctx) -> dict:
    from . import api
    from .registry import OUTPUT_EDITS
    from .scoring import summary
    if not ctx.cfg.edits:
        return {}
    ctx.ensure_loaded()
    det = ctx.read("detect")
    sch = ctx.bound_scheme()
    wm, base = _sets(ctx)
    res = {}
    for spec in ctx.cfg.edits:
        kind = edit_kind(spec, ctx.name)
        if kind == "output":
            outs = OUTPUT_EDITS[spec.split(":")[0]](wm, model=ctx.adapter, **_params(spec))
        elif kind == "text":
            outs = api.rewrite(list(wm), spec[len("text:"):], device=ctx.device)
        elif kind == "field_finetune":
            outs = _field_finetune(ctx, **_params(spec))
        else:                                               # a weight edit of a text-family release
            from .models import load_release
            safe = "".join(ch if ch.isalnum() else "_" for ch in spec)
            child = api.modify(ctx.release(), spec, os.path.join(ctx.out, "edits", safe), device=ctx.device)
            model = load_release(child, ctx.device)
            outs = _agent_run(ctx, model, True, 0)[0] if ctx.name == "agent" else _text_generate(ctx, model, True, 0)
            del model
        res[spec] = summary(_stats(ctx, sch, outs, base), det["null"], float(ctx.cfg.detect.get("fpr", 0.01)))["tpr_at_1fpr"]
    return res


def step_quality(ctx) -> dict:
    q = ctx.cfg.quality
    if q == "none":
        return {"metric": "none", "before": None, "after": None}
    wm, base = _sets(ctx)
    if q == "ppl":
        from .models import load_release
        from .quality import ppl
        model = load_release(ctx.release(), ctx.device)
        prompts, tok = ctx.read("prepare")["prompts"], ctx.adapter.tokenizer()
        return {"metric": q, "before": ppl(model, tok, prompts, base), "after": ppl(model, tok, prompts, wm)}
    if q == "accuracy":
        from .quality import accuracy
        golds = [it.get("answer", "") for it in ctx.read("prepare")["items"]]
        g = ctx.read("generate")
        b, w = (g["base_finish"], g["wm_finish"]) if ctx.name == "agent" else (base, wm)
        return {"metric": q, "before": accuracy(b, golds), "after": accuracy(w, golds)}
    if q == "fid":
        from .quality import fid
        a = os.path.join(ctx.out, "assets")
        return {"metric": q, "before": None, "after": fid(os.path.join(a, "student"), os.path.join(a, "base"),
                                                          cuda=str(ctx.device).startswith("cuda"))}
    if q == "rel_l2":
        from .quality import rel_l2
        ctx.ensure_loaded()
        _, test = ctx.adapter.splits()
        dec = ctx.adapter.y_normalizer.decode
        err = lambda outs: sum(rel_l2(dec(o), y) for o, (_, _, y) in zip(outs, test)) / max(len(outs), 1)
        return {"metric": q, "before": err(base), "after": err(wm)}
    raise ConfigError(f"quality {q!r} is not available for model {ctx.name}")


def step_report(ctx) -> dict:
    det = ctx.read("detect")
    rep = {"model": ctx.cfg.model, "scheme": ctx.cfg.scheme,
           "detect": {k: det[k] for k in ("tpr_at_1fpr", "thr", "auc", "n_wm", "n_null", "median_wm", "median_null",
                                          "null_kind")},
           "quality": ctx.read("quality"), "edits": ctx.read("edits"), "settings": asdict(ctx.cfg),
           "release": ctx.read("imprint")["release"]}
    return rep                                              # written as out/report.json


def _viewer_example(ctx) -> dict:
    """One worked example of this run, in the shape the run viewer draws."""
    import shutil
    det = ctx.read("detect")
    wm_s, null_s = det["wm"], det["null"]
    z = lambda v: round(float(v[0]), 3) if v else None
    if ctx.name in TEXT_FAMILY + ("vlm",):
        g, prep = ctx.read("generate"), ctx.read("prepare")
        ex = {"marked": g["wm"][0], "base": g["base"][0], "z_marked": z(wm_s),
              "z_base": z(null_s) if det.get("null_kind") == "base" else None}
        if ctx.name == "text":
            return {**ex, "prompt": prep["prompts"][0]}
        it = prep["items"][0]
        if ctx.name == "agent":
            return {**ex, "question": it["question"], "answer": it.get("answer", ""),
                    "finish_marked": g["wm_finish"][0], "transcript": [
                        {"role": m["role"], "content": m["content"][:600]} for m in g.get("wm_transcript_0", [])[1:]]}
        src = (it.get("images") or [it.get("image")])[0]
        dst = None
        if src and os.path.isfile(src):
            dst = os.path.join("assets", "example" + os.path.splitext(src)[1])
            shutil.copyfile(src, os.path.join(ctx.out, dst))
        return {**ex, "question": it["question"], "answer": it.get("answer", ""), "image": dst}
    ctx.ensure_loaded()
    sch = ctx.bound_scheme()
    wm, base = _sets(ctx)
    if ctx.name == "taming":
        def grid(codes):                                    # per image token: 1 green, 0 not, -1 not scored
            pos = sch.pos if len(sch.pos) > int(codes.max()) else torch.cat(
                [sch.pos, torch.full((int(codes.max()) + 1 - len(sch.pos),), -1, dtype=torch.long)])
            a_, b_ = pos[codes[:-1]], pos[codes[1:]]
            ok = (a_ >= 0) & (b_ >= 0)
            G = sch.table()
            out = torch.full((len(codes),), -1)
            out[1:][ok] = G[a_[ok], b_[ok]].long()
            return out.view(16, -1).tolist()
        return {"marked_png": os.path.join("assets", "student", "00000.png"),
                "base_png": os.path.join("assets", "base", "00000.png"),
                "marked_green": grid(wm[0].long().cpu()), "base_green": grid(base[0].long().cpu()),
                "z_marked": z(wm_s), "z_base": z(null_s)}
    # geopt: the upper surface seen from above, as last night's demo_check.py
    import random
    from .schemes.ripple import output_pattern
    _, test = ctx.adapter.splits()
    p = test[0][0]
    ref, out = base[0].float(), wm[0].float()
    prm = sch.params
    g = output_pattern(ref, sch.secret, prm["m"], prm["output_scale"])
    green_key = ((out - ref) * g).sum(-1)[0] > 0
    wrong = sch.wrong_key({"secret": sch.secret}, random.Random(0xC0FFEE), ctx.out) or {"secret": sch.secret + 1}
    gw = output_pattern(ref, int(wrong["secret"]), prm["m"], prm["output_scale"])
    green_wrong = ((out - ref) * gw).sum(-1)[0] > 0
    pts = p[0, :, :3].float().cpu()
    idx = torch.nonzero(pts[:, 1] >= pts[:, 1].median()).flatten()
    idx = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(0))[:6000]]
    field = ctx.adapter.y_normalizer.decode(out)[0, :, 0].float().cpu()
    r3 = lambda t: [round(float(v), 3) for v in t]
    return {"x": r3(pts[idx, 0]), "y": r3(pts[idx, 2]), "field": r3(field[idx.to(field.device)]),
            "pattern": r3(g[0, :, 0].cpu()[idx]), "green_key": green_key.cpu()[idx].int().tolist(),
            "green_wrong": green_wrong.cpu()[idx].int().tolist(), "z_marked": z(wm_s),
            "condition": r3(test[0][1].flatten().cpu())}


def step_viewer(ctx) -> dict:
    os.makedirs(os.path.join(ctx.out, "assets"), exist_ok=True)
    steps = []
    with open(os.path.join(ctx.out, "progress.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("event") == "done" and r["step"] not in ("report", "viewer"):
                steps = [s for s in steps if s["step"] != r["step"]] + [{"step": r["step"], "sec": r.get("sec")}]
    order = {s: i for i, s in enumerate(STEPS)}
    steps.sort(key=lambda s: order[s["step"]])
    rep, det = ctx.read("report"), ctx.read("detect")
    view = {"model": rep["model"], "scheme": rep["scheme"], "steps": steps,
            "detect": {**rep["detect"], "wm": det["wm"], "null": det["null"], "fpr": ctx.cfg.detect.get("fpr", 0.01)},
            "quality": rep["quality"], "edits": rep["edits"], "imported": ctx.cfg.imprint.get("from"),
            "example": _viewer_example(ctx)}
    _write(os.path.join(ctx.out, "assets", "viewer.json"), view)
    return {"viewer": os.path.join(ctx.out, "assets", "viewer.json")}


STEP_FNS = {"prepare": step_prepare, "selfgen": step_selfgen, "calibrate": step_calibrate, "imprint": step_imprint,
            "generate": step_generate, "detect": step_detect, "edits": step_edits, "quality": step_quality,
            "report": step_report, "viewer": step_viewer}


def run(cfg: Config, *, only: list[str] | None = None, device: str | None = None) -> dict:
    """Run the pipeline (or only some steps); returns the report."""
    if only:
        bad = [s for s in only if s not in STEPS]
        if bad:
            raise ConfigError(f"unknown step(s) {', '.join(bad)}; steps: {', '.join(STEPS)}")
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg.smoke and not cfg.out.rstrip("/").endswith("_smoke"):
        cfg.out = cfg.out.rstrip("/") + "_smoke"            # a smoke run never fills the real run's directory
    os.makedirs(cfg.out, exist_ok=True)
    cpath = os.path.join(cfg.out, "config.json")
    if os.path.exists(cpath):                               # resuming: only the same experiment may continue here
        diff = config_diff(json.load(open(cpath)), asdict(cfg))
        if diff:
            raise ConfigError(f"{cfg.out} holds a run with other settings ({', '.join(diff)} differ): use a new out, "
                              f"or remove that directory to start over")
    _write(cpath, asdict(cfg))
    ctx = Context(cfg, device)
    changed = False
    for step in STEPS:
        if only and step not in only:
            continue
        path = os.path.join(cfg.out, f"{step}.json")
        if os.path.exists(path) and not changed:
            _progress(cfg.out, step=step, event="skip")
            continue
        for later in STEPS[STEPS.index(step) + 1:]:        # their inputs change: they run again, now or next time
            for ext in (".json", ".pt"):
                lp = os.path.join(cfg.out, later + ext)
                if os.path.exists(lp):
                    os.remove(lp)
        _progress(cfg.out, step=step, event="start")
        t0 = time.time()
        print(json.dumps({"step": step}), flush=True)
        res = STEP_FNS[step](ctx)
        _write(path, res)
        _progress(cfg.out, step=step, event="done", sec=round(time.time() - t0, 1))
        changed = True
    rp = os.path.join(cfg.out, "report.json")
    return json.load(open(rp)) if os.path.exists(rp) else {}
