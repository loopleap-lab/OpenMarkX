"""One name-keyed registry for schemes and edits."""
from __future__ import annotations

SCHEMES: dict = {}       # name -> class (for a plugin: the version loaded last)
VERSIONS: dict = {}      # (name, module) -> class: every loaded plugin version, so a key finds its own code
EDITS: dict = {}


def register_scheme(cls=None, *, name: str | None = None):
    def wrap(c):
        n = name or c.name
        old = SCHEMES.get(n)
        if old is not None and old is not c:
            builtin = lambda k: k.__module__.startswith("markweightx.")
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


def load_plugin(source: str):
    """Import a scheme defined outside the package: a module name ("bob_lab.mark") or a .py file."""
    import importlib
    import importlib.util
    import os
    if source.startswith("blob:"):
        from .keys import resolve
        source = resolve(source)
    if source.endswith(".py") and os.path.exists(source):
        import hashlib
        import sys
        h = hashlib.sha256(open(source, "rb").read()).hexdigest()[:12]   # one module per version of the code
        name = "mwx_plugin_" + "".join(ch if ch.isalnum() else "_" for ch in os.path.splitext(os.path.basename(source))[0]) + "_" + h
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, source)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod                            # so the scheme's source file can be found again
        spec.loader.exec_module(mod)
        return mod
    return importlib.import_module(source)


def scheme_source(cls) -> str | None:
    """Where a registered scheme class lives, if outside markweightx (so a key can find it again)."""
    import os
    import sys
    mod = cls.__module__
    if mod.startswith("markweightx."):
        return None
    if mod.startswith("mwx_plugin_") or mod == "__main__":
        f = getattr(sys.modules.get(mod), "__file__", None)
        return os.path.abspath(f) if f else None
    return mod


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


def get_scheme(name: str, *, source: str | None = None, **params):
    """A scheme object by name. With source (a plugin file, blob or module), the class comes from that code, so a key
    made with one version of a plugin is never scored with another."""
    from . import schemes  # noqa: F401  (registers the built-ins)
    if source:
        mod = load_plugin(source).__name__
        cls = VERSIONS.get((name, mod))
        if cls is None:
            raise KeyError(f"{source} does not register a scheme named {name!r} (it registers: "
                           f"{', '.join(plugin_schemes(mod)) or 'nothing; add @mwx.register_scheme'})")
    elif name in SCHEMES:
        cls = SCHEMES[name]
    else:
        import difflib
        known = sorted(n for n, c in SCHEMES.items() if not getattr(c, "experimental", False))
        near = difflib.get_close_matches(name, list(SCHEMES), n=1)
        raise KeyError(f"unknown scheme {name!r}{f' (did you mean {near[0]!r}?)' if near else ''}; available: {', '.join(known)} "
                       f"(or a plugin file: my_scheme.py)")
    allowed = scheme_params(cls)
    bad = sorted(set(params) - set(allowed))
    if bad:
        raise ValueError(f"scheme {name!r} has no parameter {', '.join(bad)}; it takes: {', '.join(allowed) or 'none'}")
    return cls(**params)
