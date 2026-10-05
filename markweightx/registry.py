"""One name-keyed registry for schemes and edits."""
from __future__ import annotations

SCHEMES: dict = {}
EDITS: dict = {}


def register_scheme(cls=None, *, name: str | None = None):
    def wrap(c):
        n = name or c.name
        if n in SCHEMES and SCHEMES[n] is not c:
            raise ValueError(f"scheme {n!r} already registered")
        SCHEMES[n] = c
        return c
    return wrap(cls) if cls is not None else wrap


def register_edit(fn=None, *, name: str | None = None):
    def wrap(f):
        EDITS[name or f.__name__] = f
        return f
    return wrap(fn) if fn is not None else wrap


def get_scheme(name: str, **params):
    from . import schemes  # noqa: F401  (registers the built-ins)
    if name not in SCHEMES:
        raise KeyError(f"unknown scheme {name!r}; known: {sorted(SCHEMES)}")
    return SCHEMES[name](**params)
