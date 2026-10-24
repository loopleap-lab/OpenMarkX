"""Moved to openmarkx.schemes.ripple. This name is that module itself, so the scripts here that set
fieldmark.DOMAIN / OUTPUT_SCALE / BASIS_MIN_WAVELENGTH still change what its functions do."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from openmarkx.schemes import ripple as _ripple  # noqa: E402

sys.modules[__name__] = _ripple
