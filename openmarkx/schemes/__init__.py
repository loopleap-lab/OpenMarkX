from ..registry import register_scheme
from .base import NoWatermark, Scheme
from . import gaussmark, markllm, pro, ripple, wmar  # noqa: F401

register_scheme(NoWatermark)

__all__ = ["Scheme"]
