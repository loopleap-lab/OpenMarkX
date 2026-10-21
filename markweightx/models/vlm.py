"""The VLM adapter (image + question in, text out; MedGemma). Wraps markweightx.vlm, whose functions are unchanged."""
from __future__ import annotations

from ..registry import register_model
from .base import Model, Output
from .text import TokenizerCodec


@register_model
class VLMModel(Model):
    name = "vlm"
    unit = "token"
    output = Output.TOKENS

    def __init__(self, base: str | None = None, **params):
        super().__init__(base=base, **params)
        self.base = base
        self._proc = None

    def with_processor(self, proc):
        """Use this processor instead of loading the base's (e.g. a stand-in for tests)."""
        self._proc = proc
        return self

    def processor(self):
        if self._proc is None:
            from ..vlm import load_processor
            self._proc = load_processor(self.base)
        return self._proc

    def load(self, path, device: str):
        from ..types import Release
        from ..vlm import load_vlm, load_vlm_release
        return load_vlm_release(path, device) if isinstance(path, Release) else load_vlm(path or self.base, device)

    def run(self, net, inputs: list, *, new_tokens: int = 768, seed: int = 0, bs: int = 16, adapter: bool = True, **kw):
        from ..vlm import generate_vlm
        items = [{"images": [x["image"]] if x.get("image") is not None else [], "text": x["question"]} for x in inputs]
        return generate_vlm(net, self.processor(), items, new_tokens=new_tokens, seed=seed, bs=bs, adapter=adapter, **kw)

    def imprint(self, scheme: str, out: str, pairs: str, **kw):
        """Learn the mark into the language model's LoRA on (image, question, answer) pairs: embed_vlm's objective."""
        from ..vlm import embed_vlm
        return embed_vlm(scheme, self.base, out, pairs, **kw)

    def codec(self):
        return TokenizerCodec(self.processor().tokenizer)
