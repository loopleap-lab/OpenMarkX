"""The VLM adapter (image + question in, text out; MedGemma). Wraps markweightx.vlm, whose functions are unchanged."""
from __future__ import annotations

from ..registry import register_model
from .base import Model, Output
from .text import TokenizerCodec


def _import(ref: str):
    import importlib
    mod, _, attr = ref.partition(":")
    return getattr(importlib.import_module(mod), attr)


@register_model
class VLMModel(Model):
    name = "vlm"
    unit = "token"
    output = Output.TOKENS

    def __init__(self, base: str | None = None, factory: str | None = None, processor: str | None = None, **params):
        super().__init__(base=base, factory=factory, processor=processor, **params)
        self.base = base
        self._proc = None

    def with_processor(self, proc):
        """Use this processor instead of loading the base's (e.g. a stand-in for tests)."""
        self._proc = proc
        return self

    def processor(self):
        if self._proc is None:
            if self.params["processor"]:                    # "module:Class": a stand-in (tests)
                self._proc = _import(self.params["processor"])()
            else:
                from ..vlm import load_processor
                self._proc = load_processor(self.base)
        return self._proc

    def load(self, path, device: str):
        """The base (or a stand-in from `factory`), with a release's LoRA adapter on it (switchable)."""
        from ..types import Release
        from ..vlm import load_vlm
        net = _import(self.params["factory"])() if self.params["factory"] else \
            load_vlm(path if isinstance(path, str) else self.base, device)
        net = net.to(device).eval()
        if isinstance(path, Release) and path.adapter:
            from peft import PeftModel
            net = PeftModel.from_pretrained(net, path.adapter).eval()
        return net

    def run(self, net, inputs: list, *, new_tokens: int = 768, seed: int = 0, bs: int = 16, adapter: bool = True, **kw):
        from ..vlm import generate_vlm
        items = [{"images": x.get("images") or ([x["image"]] if x.get("image") is not None else []), "text": x["question"]}
                 for x in inputs]
        return generate_vlm(net, self.processor(), items, new_tokens=new_tokens, seed=seed, bs=bs, adapter=adapter, **kw)

    def imprint(self, scheme: str, out: str, pairs: str, *, device: str = "cuda:0", **kw):
        """Learn the mark into the language model's LoRA on (image, question, answer) pairs: embed_vlm's objective.
        The release names this adapter (schema 2)."""
        from ..vlm import embed_vlm
        if self.params["factory"]:
            kw.setdefault("grad_ckpt", False)
        rel = embed_vlm(scheme, self.base, out, pairs, device=device, model=self.load(None, device),
                        proc=self.processor(), **kw)
        rel.model = {"name": self.name, "params": self.params}
        return rel.save()

    def codec(self):
        return TokenizerCodec(self.processor().tokenizer)
