"""MarkLLM's decoding-time watermarks as teachers: `markllm-<x>-d` learns upstream strategy <x> into the weights.

Upstream MarkLLM (pinned 112fa74) is imported unchanged. The fixes below live here, in the adapter, and are the
correctness fixes catalogued from Yifei's fork (ecosystem_design/markllm_fork_changes.md, A1-A8):
  A1 vocabulary width = the model's logits width;  A3 Unbiased key derived from the key, not random;
  A4 DIP hash_key restored;  A6 MorphMark scored row by row;  A8 processors get a copy of the logits.
"""
from __future__ import annotations

import hashlib
import math
import os
import sys

import torch

from ..paths import MARKLLM
from ..registry import register_scheme
from .base import Scheme

if MARKLLM not in sys.path:
    sys.path.insert(0, MARKLLM)

# upstream key -> (family, signal, detector statistic orientation, notes)
TEACHERS = {
    "KGW":       dict(family="logit-bias", kind="distill_logit"),
    "Unigram":   dict(family="logit-bias", kind="distill_logit"),
    "MorphMark": dict(family="logit-bias", kind="distill_logit", rowwise=True),
    "SWEET":     dict(family="logit-bias", kind="distill_logit", model_detect=True),
    "EWD":       dict(family="logit-bias", kind="distill_logit", model_detect=True),
    "DIP":       dict(family="reweight", kind="distill_logit"),
    "Unbiased":  dict(family="reweight", kind="distill_logit", model_detect=True, p_value=True),
    "SynthID":   dict(family="tournament", kind="distill_logit"),
    "EXP":       dict(family="token-sampling", kind="distill_sampling", p_value=True),
    "EXPEdit":   dict(family="token-sampling", kind="distill_sampling", p_value=True),
    "PF":        dict(family="token-sampling", kind="distill_sampling"),
    "EXPGumbel": dict(family="token-sampling", kind="distill_sampling", p_value=True),
    "ITSEdit":   dict(family="token-sampling", kind="distill_sampling", p_value=True),
}


class ModelView:
    """What an upstream detector sees as `generation_model`: the base (adapter off) with float32 logits."""

    def __init__(self, model):
        self.m = model
        self.device = model.device

    def __call__(self, *a, **k):
        import contextlib
        ctx = self.m.disable_adapter() if hasattr(self.m, "disable_adapter") else contextlib.nullcontext()
        with ctx, torch.no_grad():
            o = self.m(*a, **k)
        o.logits = o.logits.float()
        return o

    def __getattr__(self, n):
        return getattr(self.m, n)


class RowWise:
    """A6: call an upstream processor one row at a time (MorphMark pools statistics over the batch)."""

    def __init__(self, lp):
        self.lp = lp

    def __call__(self, input_ids, scores):
        return torch.cat([self.lp(input_ids[b:b + 1], scores[b:b + 1].clone()) for b in range(input_ids.shape[0])])


def _derive(key: int, n: int) -> bytes:
    return hashlib.shake_256(f"markweightx:{key}".encode()).digest(n)


class MarkLLMDistill(Scheme):
    teacher = "KGW"

    def __init__(self, **params):
        super().__init__(**params)
        meta = TEACHERS[self.teacher]
        self.kind = meta["kind"]
        self.rowwise = meta.get("rowwise", False)
        self.p_value = meta.get("p_value", False)
        self.needs_model_to_detect = meta.get("model_detect", False)

    def setup(self, model, tokenizer, device, key=None):
        super().setup(model, tokenizer, device, key)
        from utils.transformers_config import TransformersConfig
        from watermark.auto_watermark import AutoWatermark
        tc = TransformersConfig(model=model, tokenizer=tokenizer, vocab_size=model.config.vocab_size,   # A1
                                device=device, max_new_tokens=200, do_sample=True)
        cfg_path = os.path.join(MARKLLM, "config", f"{self.teacher}.json")
        self.wm = AutoWatermark.load(self.teacher, algorithm_config=cfg_path, transformers_config=tc)
        c = self.wm.config
        k = (key or {}).get("key", getattr(c, "key", 42) if hasattr(c, "key") else 42)
        k = self.params.get("key", k)
        if self.teacher == "Unbiased":                       # A3: a fixed, derivable key
            c.hash_key = _derive(k, 128)
            c.ignore_history_generation = False              # teacher forcing: no per-generation dedupe state
        if self.teacher == "DIP":                            # A4
            c.hash_key = str(k).encode()
        if self.needs_model_to_detect:                       # detectors that run a model see the base, in fp32
            c.generation_model = ModelView(model)
        return {"teacher": self.teacher, "key": k, "config": os.path.basename(cfg_path)}

    def processor(self):
        """The decode-time teacher as a LogitsProcessor (for sampling from the teacher)."""
        return RowWise(self.wm.logits_processor) if self.rowwise else self.wm.logits_processor

    # ---- training signal: upstream logits_processor, teacher-forced position by position ----------------
    def target_logits(self, input_ids, logits):
        lp = self.processor()                                # A6 inside RowWise
        out = torch.empty_like(logits)
        for t in range(logits.shape[1]):
            out[:, t] = lp(input_ids[:, : t + 1], logits[:, t].clone())   # A8
        return out

    # ---- detection: upstream detect_watermark -------------------------------------------------------
    def score(self, text):
        d = self.wm.detect_watermark(text, return_dict=True)
        s = float(d["score"])
        if self.p_value:                                     # upstream returns a p-value: report -log10 p
            s = -math.log10(max(s, 1e-300))
        n = len(self.tok(text, add_special_tokens=False)["input_ids"])
        return s, n


for _k in TEACHERS:
    register_scheme(type(f"MarkLLM_{_k}", (MarkLLMDistill,), {"teacher": _k, "name": f"markllm-{_k.lower()}-d"}))
