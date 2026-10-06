"""MarkWeightX: watermarks learned into open-weight LLMs. See markweightx.api for the five calls."""
from .api import build_scheme, calibrate, detect, embed, generate, modify, rewrite, verify_model
from . import keys
from .keys import load_key
from .primitives import Access, Imprint, describe
from .registry import register_edit, register_scheme
from .schemes.base import Scheme
from .types import Detection, Key, Release

__all__ = ["build_scheme", "embed", "calibrate", "modify", "generate", "rewrite", "detect", "verify_model", "load_key",
           "register_scheme", "register_edit", "Scheme", "Release", "Key", "Detection", "Access", "Imprint", "describe"]
