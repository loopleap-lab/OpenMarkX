"""Edits a downstream user may make. Weight edits (modify) return a new Release; text edits (rewrite) return texts."""
from .weights import finetune, merge, prune, quantize, modify  # noqa: F401
from .text import rewrite  # noqa: F401
