"""Loading bases and releases."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .paths import base_path
from .types import Release


def load_tokenizer(base: str):
    tok = AutoTokenizer.from_pretrained(base_path(base))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base(base: str, device: str, dtype=torch.bfloat16):
    return AutoModelForCausalLM.from_pretrained(base_path(base), torch_dtype=dtype).to(device).eval()


def load_release(rel: Release, device: str):
    """Returns a model with the release applied. LoRA releases keep the adapter switchable (disable_adapter())."""
    from peft import PeftModel
    path = rel.weights or base_path(rel.base)
    q = rel.load.get("quant")
    if q:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16) if q == "4bit" \
            else BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb, device_map={"": device})
    else:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    if rel.adapter and not rel.weights:
        model = PeftModel.from_pretrained(model, rel.adapter)
    return model.eval()


class ConfigOnly:
    """A stand-in for detectors that need only the model's config (vocabulary width) and device, not its weights."""

    def __init__(self, base: str, device: str):
        from transformers import AutoConfig
        self.config = AutoConfig.from_pretrained(base_path(base))
        self.device = torch.device(device)

    def __call__(self, *a, **k):
        raise RuntimeError("this detector was set up without model weights")
