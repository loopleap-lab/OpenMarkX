"""Output edits: what happens to a model's outputs between generation and the detector, e.g. an image saved and
read back. Registered as fn(outputs, *, model, **params) -> outputs."""
from __future__ import annotations

from ..registry import register_output_edit


@register_output_edit
def image_roundtrip(codes, *, model):
    """Decode image tokens to pixels and encode them again: what a detector sees from an image file."""
    c = model.codec()
    return c.encode(c.decode(codes))
