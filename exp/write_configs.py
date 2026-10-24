"""Write configs/watermark/<name>.yaml for every finished scheme: its class, access level and default params.
  python exp/write_configs.py      (tests/test_configs.py checks they match the code)"""
import inspect
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openmarkx.primitives import kind_of
from openmarkx.registry import SCHEMES, get_scheme


def defaults(cls) -> dict:
    sig = inspect.signature(cls.__init__)
    return {k: p.default for k, p in sig.parameters.items() if k not in ("self", "params") and p.default is not inspect._empty}


def row(name):
    sch = get_scheme(name)
    return {"name": name, "class": kind_of(sch), "access": sch.access.name, "imprint": sch.imprint.value,
            "teacher": sch.teacher, "params": defaults(type(sch))}


if __name__ == "__main__":
    get_scheme("nowm")
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "watermark")
    os.makedirs(out, exist_ok=True)
    for n, c in sorted(SCHEMES.items()):
        if not getattr(c, "experimental", False) and c.__module__.startswith("openmarkx."):
            yaml.safe_dump(row(n), open(os.path.join(out, f"{n}.yaml"), "w"), sort_keys=False)
            print("wrote", n)
