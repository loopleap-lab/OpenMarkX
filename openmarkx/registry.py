"""One name-keyed registry for schemes, edits and model adapters."""
from __future__ import annotations

SCHEMES: dict = {}       # name -> class (for a plugin: the version loaded last)
VERSIONS: dict = {}      # (name, module) -> class: every loaded plugin version, so a key finds its own code
LOADED: set = set()      # sha256 of every plugin file imported in this process (by you, or by a trusted key)
EDITS: dict = {}         # weight edits: fn(release, out, **params) -> Release (modify)
TEXT_EDITS: dict = {}    # text edits: fn(texts, **params) -> texts (rewrite)
OUTPUT_EDITS: dict = {}  # output edits: fn(outputs, *, model, **params) -> outputs (e.g. image_roundtrip)
MODELS: dict = {}        # model adapters: name -> class (for a plugin: the version loaded last)
MODEL_VERSIONS: dict = {}  # (name, module) -> class, as VERSIONS for schemes


def _builtin(module: str) -> bool:
    """Code shipped with the package (markweightx is its name before the rename)."""
    return module.startswith(("openmarkx.", "markweightx."))


def register_scheme(cls=None, *, name: str | None = None):
    def wrap(c):
        n = name or c.name
        old = SCHEMES.get(n)
        if old is not None and old is not c:
            builtin = lambda k: _builtin(k.__module__)
            if builtin(old) or builtin(c) or old.__module__ == c.__module__ and old.__qualname__ != c.__qualname__:
                raise ValueError(f"scheme {n!r} is already registered by {old.__module__}.{old.__qualname__}; "
                                 f"pick another name")
        SCHEMES[n] = c
        VERSIONS[(n, c.__module__)] = c
        return c
    return wrap(cls) if cls is not None else wrap


def plugin_schemes(module_name: str) -> list[str]:
    """The scheme names a loaded plugin module registers."""
    return sorted(n for (n, m) in VERSIONS if m == module_name)


def register_edit(fn=None, *, name: str | None = None):
    def wrap(f):
        EDITS[name or f.__name__] = f
        return f
    return wrap(fn) if fn is not None else wrap


def register_text_edit(fn=None, *, name: str | None = None):
    """Register fn(texts: list[str], **params) -> list[str] as a text edit, used as rewrite(texts, "name:k=v")
    and in an arena as "text:name:k=v". It may also take device= (the GPU the run uses)."""
    def wrap(f):
        TEXT_EDITS[name or f.__name__] = f
        return f
    return wrap(fn) if fn is not None else wrap


def register_output_edit(fn=None, *, name: str | None = None):
    """Register fn(outputs, *, model, **params) -> outputs: a change to outputs before detection (e.g. an image
    round trip), used by the pipeline's edits step."""
    def wrap(f):
        OUTPUT_EDITS[name or f.__name__] = f
        return f
    return wrap(fn) if fn is not None else wrap


def load_plugins(paths) -> list:
    """Import plugin files (or modules) that register edits or schemes, so their names resolve in this process."""
    return [load_plugin(p) for p in ([paths] if isinstance(paths, str) else paths or [])]


class UntrustedPlugin(PermissionError):
    """A key names plugin code that this process has not loaded itself."""


def _module_digest(module) -> str:
    import hashlib
    return hashlib.sha256(open(module.__file__, "rb").read()).hexdigest()[:12]


def _file_module_name(path: str) -> str:
    import hashlib
    import os
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]   # one module per version of the code
    stem = "".join(ch if ch.isalnum() else "_" for ch in os.path.splitext(os.path.basename(path))[0])
    return f"omx_plugin_{stem}_{h}"


def plugin_loaded(source: str) -> bool:
    """Whether this exact plugin code is already imported in this process (so running it again adds no trust)."""
    import os
    import sys
    from .keys import resolve
    path = resolve(source)
    if path.endswith(".py"):
        import hashlib
        return os.path.isfile(path) and hashlib.sha256(open(path, "rb").read()).hexdigest() in LOADED
    mod, _, digest = path.partition("@")
    m = sys.modules.get(mod)
    return m is not None and (not digest or _module_digest(m) == digest)


def load_plugin(source: str):
    """Import a scheme defined outside the package: a .py file, a key-store blob, or a module name. A module name
    may carry the digest of its file ("bob_lab.mark@1a2b3c4d5e6f"); the import fails if the file has changed."""
    import importlib
    import importlib.util
    import os
    if source.startswith("blob:"):
        from .keys import resolve
        source = resolve(source)
        if not os.path.isfile(source):
            raise FileNotFoundError(f"plugin source {source} is missing from the key store (was the store copied "
                                    f"without its blobs/ directory?)")
    if source.endswith(".py") and not os.path.isfile(source):
        raise FileNotFoundError(f"plugin source {source} not found (moved or deleted?)")
    if not source.endswith(".py"):
        mod_name, _, digest = source.partition("@")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(f"plugin module {mod_name!r} cannot be imported: {e}") from None
        if digest and _module_digest(mod) != digest:
            raise ValueError(f"plugin module {mod_name} has changed since this key was made "
                             f"(file digest {_module_digest(mod)}, key expects {digest})")
        return mod
    import sys                                             # a .py file or blob
    name = _file_module_name(source)
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, source)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                                # so the scheme's source file can be found again
    spec.loader.exec_module(mod)
    import hashlib
    LOADED.add(hashlib.sha256(open(source, "rb").read()).hexdigest())
    return mod


def scheme_source(cls) -> str | None:
    """Where a registered scheme class lives, if outside openmarkx (so a key can find it again)."""
    import os
    import sys
    mod = cls.__module__
    if _builtin(mod):
        return None
    if mod == "__main__":
        raise ValueError(f"scheme {cls.__name__} is defined in the script being run (or a notebook), which a key cannot "
                         f"load again: keep it in its own file and pass the file, e.g. omx.embed(base, 'my_mark.py')")
    if mod.startswith(("omx_plugin_", "mwx_plugin_")):   # mwx_: loaded before the rename
        f = getattr(sys.modules.get(mod), "__file__", None)
        return os.path.abspath(f) if f else None
    return f"{mod}@{_module_digest(sys.modules[mod])}"     # an importable module, pinned to its current code


def scheme_params(cls) -> list[str]:
    """The parameters a scheme accepts: the named arguments of __init__ along its class hierarchy."""
    import inspect
    names = []
    for k in cls.__mro__:
        init = k.__dict__.get("__init__")
        if init is None:
            continue
        for p in inspect.signature(init).parameters.values():
            if p.name != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and p.name not in names:
                names.append(p.name)
    return names


def resolved_params(sch) -> dict:
    """Every parameter of a scheme object, defaults included, so a key keeps detecting the same way after a default
    changes in the code."""
    import inspect
    out = {}
    for k in reversed(type(sch).__mro__):
        init = k.__dict__.get("__init__")
        if init is not None:
            out.update({p.name: p.default for p in inspect.signature(init).parameters.values()
                        if p.name != "self" and p.default is not inspect._empty})
    out.update(sch.params)
    return {k: v for k, v in out.items() if k in scheme_params(type(sch))}


def _takes_any(cls) -> bool:
    """The class's own __init__ takes only **params (no named parameters)."""
    import inspect
    init = next((k.__dict__["__init__"] for k in cls.__mro__ if "__init__" in k.__dict__), None)
    ps = [p for p in inspect.signature(init).parameters.values() if p.name != "self"] if init else []
    return bool(ps) and all(p.kind == p.VAR_KEYWORD for p in ps) and not _builtin(init.__module__)


def get_scheme(name: str, *, source: str | None = None, **params):
    """A scheme object by name. With source (a plugin file, blob or module), the class comes from that code, so a key
    made with one version of a plugin is never scored with another."""
    from . import schemes  # noqa: F401  (registers the built-ins)
    if source:
        mod = load_plugin(source).__name__
        cls = VERSIONS.get((name, mod))
        if cls is None:
            raise KeyError(f"{source} does not register a scheme named {name!r} (it registers: "
                           f"{', '.join(plugin_schemes(mod)) or 'nothing; add @omx.register_scheme'})")
    elif name in SCHEMES:
        cls = SCHEMES[name]
    else:
        import difflib
        known = sorted(n for n, c in SCHEMES.items() if not getattr(c, "experimental", False))
        near = difflib.get_close_matches(name, list(SCHEMES), n=1)
        raise KeyError(f"unknown scheme {name!r}{f' (did you mean {near[0]!r}?)' if near else ''}; available: {', '.join(known)} "
                       f"(or a plugin: my_scheme.py, pkg.module)")
    allowed = scheme_params(cls)
    bad = sorted(set(params) - set(allowed))
    if bad and not _builtin(cls.__module__) and _takes_any(cls):
        import warnings                                    # a plugin that declares no named parameters
        warnings.warn(f"{name}: parameters {bad} are not named in __init__; passing them through **params")
        bad = []
    if bad:
        raise ValueError(f"scheme {name!r} has no parameter {', '.join(bad)}; it takes: {', '.join(allowed) or 'none'}")
    return cls(**params)


def register_model(cls=None, *, name: str | None = None):
    """Register a model adapter (a openmarkx.models.Model subclass) under its name."""
    def wrap(c):
        n = name or c.name
        old = MODELS.get(n)
        if old is not None and old is not c and (_builtin(old.__module__) or _builtin(c.__module__)):
            raise ValueError(f"model {n!r} is already registered by {old.__module__}.{old.__qualname__}; pick another name")
        MODELS[n] = c
        MODEL_VERSIONS[(n, c.__module__)] = c
        return c
    return wrap(cls) if cls is not None else wrap


def list_models() -> list[str]:
    from . import models  # noqa: F401  (registers the built-ins)
    return sorted(MODELS)


def get_model(name: str, *, source: str | None = None, trust: bool = False, **params):
    """A model adapter by name. With source (a plugin file, blob or module), the class comes from that code, which
    runs only with trust=True or when this process has already loaded that exact code."""
    from . import models  # noqa: F401
    if source:
        if not trust and not plugin_loaded(source):
            raise UntrustedPlugin(f"model adapter {name!r} comes from plugin code {source}: import it yourself first, "
                                  f"or pass trust=True (--trust-plugin) if you trust that file")
        mod = load_plugin(source).__name__
        cls = MODEL_VERSIONS.get((name, mod))
        if cls is None:
            have = sorted(n for (n, m) in MODEL_VERSIONS if m == mod)
            raise KeyError(f"{source} does not register a model named {name!r} (it registers: "
                           f"{', '.join(have) or 'nothing; add @omx.register_model'})")
    elif name in MODELS:
        cls = MODELS[name]
    else:
        import difflib
        near = difflib.get_close_matches(name, list(MODELS), n=1)
        raise KeyError(f"unknown model {name!r}{f' (did you mean {near[0]!r}?)' if near else ''}; available: "
                       f"{', '.join(sorted(MODELS))} (or a plugin: my_model.py)")
    m = cls(**params)
    m.source = source                                       # recorded in releases, so they reopen with the same code
    return m
