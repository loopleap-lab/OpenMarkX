"""One name-keyed registry for schemes and edits."""
from __future__ import annotations

SCHEMES: dict = {}
EDITS: dict = {}


def register_scheme(cls=None, *, name: str | None = None):
    def wrap(c):
        n = name or c.name
        old = SCHEMES.get(n)
        if old is not None and old is not c and (old.__module__, old.__qualname__) != (c.__module__, c.__qualname__):
            raise ValueError(f"scheme {n!r} already registered")
        SCHEMES[n] = c
        return c
    return wrap(cls) if cls is not None else wrap


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
        name = "mwx_plugin_" + os.path.splitext(os.path.basename(source))[0]
        import sys
        loaded = sys.modules.get(name)
        if loaded is not None and os.path.abspath(getattr(loaded, "__file__", "")) == os.path.abspath(source):
            return loaded
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


def get_scheme(name: str, *, source: str | None = None, **params):
    from . import schemes  # noqa: F401  (registers the built-ins)
    if name not in SCHEMES and source:
        load_plugin(source)
    if name not in SCHEMES:
        known = sorted(n for n, c in SCHEMES.items() if not getattr(c, "experimental", False))
        raise KeyError(f"unknown scheme {name!r}; available: {', '.join(known)} "
                       f"(a plugin scheme must be imported first, or passed as source=)")
    return SCHEMES[name](**params)
