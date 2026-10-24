"""generate: sample continuations of fixed prompts from a release (or its base, adapter off)."""
from __future__ import annotations

import contextlib

import torch

from .paths import DATA


def eval_prompts(tok, n=500, prompt_len=50, new_tokens=200, data: str | None = None):
    """The first held-out C4 documents that are long enough, cut to prompt_len tokens."""
    texts = torch.load(data or DATA["llama3"], weights_only=True)["eval_texts"]
    out = []
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if len(ids) >= prompt_len + new_tokens:
            out.append(tok.decode(ids[:prompt_len]))
        if len(out) >= n:
            break
    return out


@torch.no_grad()
def generate(model, tok, prompts, *, new_tokens=200, bs=50, seed=0, adapter=True, logits_processor=None):
    torch.manual_seed(seed)
    kw = dict(max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=True, temperature=1.0, top_k=0,
              top_p=1.0, pad_token_id=tok.pad_token_id)
    if logits_processor is not None:
        from transformers import LogitsProcessorList
        kw["logits_processor"] = LogitsProcessorList([logits_processor])
    ctx = model.disable_adapter() if (not adapter and hasattr(model, "disable_adapter")) else contextlib.nullcontext()
    outs = []
    with ctx:
        for i in range(0, len(prompts), bs):
            enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(model.device)
            o = model.generate(**enc, **kw)
            outs += tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


@torch.no_grad()
def perplexity(model, tok, prompts, gens, adapter=False):
    """PPL of the generations under the model (adapter off by default: judged by the unwatermarked base)."""
    import math
    ctx = model.disable_adapter() if (not adapter and hasattr(model, "disable_adapter")) else contextlib.nullcontext()
    nll, n = 0.0, 0
    with ctx:
        for p, g in zip(prompts, gens):
            pi = tok(p, return_tensors="pt").input_ids.to(model.device)
            gi = tok(g, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
            if gi.shape[1] < 2:
                continue
            lg = model(input_ids=torch.cat([pi, gi], 1)).logits[0, pi.shape[1] - 1:-1].float()
            nll += torch.nn.functional.cross_entropy(lg, gi[0], reduction="sum").item(); n += gi.shape[1]
    return math.exp(nll / max(n, 1))
