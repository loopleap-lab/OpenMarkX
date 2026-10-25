"""markweightx is the old name of openmarkx. This shim keeps code, pickles and run directories written before the
rename working: `import markweightx` is `import openmarkx`, and every submodule (markweightx.schemes.pro, ...) is
the same module object as its openmarkx counterpart, so the registries are never loaded twice."""
import importlib
import importlib.abc
import importlib.util
import sys
import warnings

OLD, NEW = "markweightx", "openmarkx"
RENAMED = {"schemes.pro": "schemes.openmark"}      # submodules that also changed their own name


def target(name: str) -> str:
    """markweightx.X -> openmarkx.X (with RENAMED applied)."""
    rest = name[len(OLD) + 1:]
    return f"{NEW}.{RENAMED.get(rest, rest)}" if rest else NEW


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, new: str):
        self.new = new

    def create_module(self, spec):
        module = importlib.import_module(self.new)
        self.spec = module.__spec__
        return module

    def exec_module(self, module):
        module.__spec__ = self.spec                    # already executed under its new name; keep its own spec

    def get_code(self, fullname):                      # python -m markweightx.cli
        spec = importlib.util.find_spec(self.new)
        return spec.loader.get_code(self.new)


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target_=None):
        if not fullname.startswith(OLD + "."):
            return None
        new = target(fullname)
        if importlib.util.find_spec(new) is None:
            new = NEW + fullname[len(OLD):]                # not renamed (yet) after all
            if importlib.util.find_spec(new) is None:
                return None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(new))


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

warnings.warn("the markweightx package is now openmarkx: import openmarkx (as omx) instead", DeprecationWarning,
              stacklevel=2)

import openmarkx  # noqa: E402

sys.modules[OLD] = openmarkx                           # `import markweightx` returns the openmarkx module itself
