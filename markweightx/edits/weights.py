"""Weight edits on a released model: fine-tune, merge, prune, quantize.

A downstream user receives the release as plain weights, so every edit starts from the merged weights
(base + LoRA folded in). Each edit writes a new release directory whose lineage extends the parent's.
"""
from __future__ import annotations

import json
import math
import os
import time

import torch
import torch.nn.functional as F

from ..models import load_release, load_tokenizer
from ..paths import DATA, base_path
from ..registry import register_edit
from ..types import Release, Step


def materialize(rel: Release, device: str):
    """The release as one plain model (LoRA merged in)."""
    m = load_release(rel, device)
    if hasattr(m, "merge_and_unload"):
        m = m.merge_and_unload()
    return m


def _child(rel: Release, out: str, op: str, params: dict, **kw) -> Release:
    lin = list(rel.lineage) + [vars(Step(op, params))]
    return Release(path=out, base=rel.base, scheme=rel.scheme, key=rel.key, lineage=lin, **kw)


def _save_weights(model, tok, out):
    os.makedirs(out, exist_ok=True)
    w = os.path.join(out, "weights")
    model.save_pretrained(w, safe_serialization=True); tok.save_pretrained(w)
    return w


@register_edit
def finetune(rel: Release, out: str, *, steps=800, lr=1e-4, bs=8, seq_len=256, rank=16, device="cuda:0",
             data: str | None = None, seed=1, offset=10000):
    """LoRA fine-tune on clean text (next-token loss), then fold the new adapter in. `offset` keeps the
    fine-tuning chunks disjoint from the ones used to embed."""
    from peft import LoraConfig, get_peft_model
    tok = load_tokenizer(rel.base)
    model = materialize(rel, device)
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    corpus = torch.load(data or DATA["llama3"])["train"][offset:, :seq_len]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    g = torch.Generator().manual_seed(seed); model.train(); t0 = time.time()
    for step in range(steps):
        ids = corpus[torch.randint(len(corpus), (bs,), generator=g)].to(device)
        lg = model(input_ids=ids).logits[:, :-1].float()
        loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if (step + 1) % 100 == 0:
            print({"ft_step": step + 1, "loss": round(loss.item(), 4), "sec": round(time.time() - t0)}, flush=True)
    model = model.merge_and_unload().eval()
    p = dict(steps=steps, lr=lr, bs=bs, seq_len=seq_len, rank=rank, seed=seed)
    return _child(rel, out, "finetune", p, weights=_save_weights(model, tok, out)).save()


@register_edit
@torch.no_grad()
def merge(rel: Release, out: str, *, partner: str, ratio=0.4, device="cuda:0"):
    """Linear merge: w = (1 - ratio) * w_release + ratio * w_partner (same architecture)."""
    from transformers import AutoModelForCausalLM
    tok = load_tokenizer(rel.base)
    model = materialize(rel, device)
    other = AutoModelForCausalLM.from_pretrained(base_path(partner), torch_dtype=torch.bfloat16).to(device)
    po = dict(other.named_parameters())
    for n, p in model.named_parameters():
        p.mul_(1 - ratio).add_(po[n].to(p.dtype), alpha=ratio)
    del other
    return _child(rel, out, "merge", dict(partner=partner, ratio=ratio), weights=_save_weights(model, tok, out)).save()


@register_edit
@torch.no_grad()
def prune(rel: Release, out: str, *, sparsity=0.5, method="magnitude", device="cuda:0"):
    """Unstructured magnitude pruning of every linear weight in the decoder layers (per row, like Wanda's grouping)."""
    tok = load_tokenizer(rel.base)
    model = materialize(rel, device)
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and ".layers." in name:
            W = mod.weight
            k = int(W.shape[1] * sparsity)
            if k == 0:
                continue
            idx = W.abs().float().topk(k, dim=1, largest=False).indices
            W.scatter_(1, idx, 0)
    return _child(rel, out, "prune", dict(sparsity=sparsity, method=method), weights=_save_weights(model, tok, out)).save()


@register_edit
def quantize(rel: Release, out: str, *, bits=4, device="cuda:0"):
    """Post-training quantization at load time (bitsandbytes NF4 / LLM.int8()). Stores merged bf16 weights and a
    load flag, which is how a user would run a bnb-quantized model."""
    tok = load_tokenizer(rel.base)
    model = materialize(rel, device)
    w = _save_weights(model, tok, out)
    return _child(rel, out, "quantize", dict(bits=bits, method="bnb"), weights=w,
                  load={"quant": "4bit" if bits == 4 else "8bit"}).save()


def modify(rel: Release, edit: str, out: str, **params) -> Release:
    from ..registry import EDITS
    return EDITS[edit](rel, out, **params)
