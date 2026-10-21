"""MarkWeightX: watermarks learned into open-weight LLMs. See markweightx.api for the five calls."""
from .api import build_scheme, calibrate, detect, embed, generate, modify, port, rewrite, verify_model
from . import keys
from .keys import load_key
from .primitives import Access, AccessDenied, Imprint, describe
from .registry import UntrustedPlugin, get_model, load_plugins, register_edit, register_model, register_scheme, register_text_edit
from .models.base import Model, Output
from .schemes.base import Scheme
from .types import Detection, Key, Release

__all__ = ["build_scheme", "embed", "port", "calibrate", "modify", "generate", "rewrite", "detect", "verify_model", "load_key",
           "register_scheme", "register_edit", "register_text_edit", "load_plugins", "Scheme", "Release", "Key", "Detection", "Access", "AccessDenied", "Imprint",
           "UntrustedPlugin", "describe", "Model", "Output", "register_model", "get_model"]
