"""The text adapter: today's causal LM. Outputs are strings; the codec is the base's tokenizer."""
from __future__ import annotations

from ..registry import register_model
from .base import Codec, Model, Output


class TokenizerCodec(Codec):
    def __init__(self, tok):
        self.tok = tok

    def encode(self, outputs: list) -> list:
        return [self.tok(t, add_special_tokens=False)["input_ids"] for t in outputs]

    def decode(self, units: list) -> list:
        return [self.tok.decode(u) for u in units]


@register_model
class TextModel(Model):
    name = "text"
    unit = "token"
    output = Output.TOKENS

    def __init__(self, base: str | None = None, **params):
        super().__init__(base=base, **params)
        self.base = base
        self._tok = None

    def tokenizer(self):
        if self._tok is None:
            from . import load_tokenizer
            self._tok = load_tokenizer(self.base)
        return self._tok

    def load(self, path, device: str):
        from . import load_base, load_hf_release
        from ..types import Release
        if isinstance(path, Release):
            return load_hf_release(path, device)
        return load_base(path or self.base, device)

    def run(self, net, inputs: list, *, new_tokens: int = 200, temperature: float = 1.0, seed: int = 0, bs: int = 50,
            **kw) -> list:
        import torch
        tok = self.tokenizer()
        torch.manual_seed(seed)
        gen = dict(max_new_tokens=new_tokens, pad_token_id=tok.pad_token_id)
        gen.update(dict(do_sample=True, temperature=temperature, top_k=0, top_p=1.0) if temperature > 0 else
                   dict(do_sample=False))
        outs = []
        with torch.no_grad():
            for i in range(0, len(inputs), bs):
                enc = tok(inputs[i:i + bs], return_tensors="pt", padding=True).to(net.device)
                o = net.generate(**enc, **gen)
                outs += tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return outs

    def codec(self) -> Codec:
        return TokenizerCodec(self.tokenizer())
