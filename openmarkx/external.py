"""Model code that stays outside OpenMarkX (WMAR for Taming, the GeoPT repo): the user clones it and points an
environment variable at it; the adapters import it at load time and leave it unchanged."""
from __future__ import annotations

import contextlib
import importlib
import os
import sys


class MissingExternal(ImportError):
    """External model code an adapter needs is not installed or not found."""


@contextlib.contextmanager
def in_dir(path: str):
    """Run inside an external checkout (code that opens its own files by relative path), then return."""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def require_external(name: str, env: str, import_name: str):
    """Import `import_name` from the checkout that $env points to."""
    d = os.environ.get(env)
    if not d or not os.path.isdir(d):
        raise MissingExternal(f"{name} is not installed: clone it and set {env} to its directory"
                              + (f" ({env}={d} is not a directory)" if d else ""))
    if d not in sys.path:
        sys.path.insert(0, d)
    try:
        return importlib.import_module(import_name)
    except ImportError as e:
        raise MissingExternal(f"{name} at {env}={d} cannot be imported ({import_name}): {e}") from None
