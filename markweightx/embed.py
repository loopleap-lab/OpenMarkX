"""embed: learn a scheme's watermark into a base model as a LoRA adapter (logit distillation).

Loss = KL(teacher || student) per token, teacher = base logits passed through the scheme's teacher-forced signal.
One copy of the base weights serves twice: adapter OFF = teacher's base, adapter ON = student.
"""
from __future__ import annotations

import json
import math
import os
import time

import torch
import torch.nn.functional as F

from .models import load_base, load_tokenizer
from .paths import DATA, RUNS
from .registry import get_scheme
from .types import Release, Step

LORA_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def kl_teacher_student(t_logits, s_logits):
    t_logp = F.log_softmax(t_logits.float(), -1).clamp(min=-1e4)     # teachers may zero out tokens (-inf)
    s_logp = F.log_softmax(s_logits.float(), -1)
    return (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()


def embed(scheme: str, base: str, out: str, *, steps=1500, bs=16, accum=1, seq_len=128, lr=2e-4, rank=64,
          device="cuda:0", key: dict | None = None, data: str | None = None, save_every=500, seed=0, log_every=10):
    os.makedirs(out, exist_ok=True)
    tok = load_tokenizer(base)
    model = load_base(base, device)
    sch = get_scheme(scheme)
    key = sch.setup(model, tok, device, key)
    if sch.kind != "distill_logit":
        raise NotImplementedError(f"{scheme}: recipe {sch.kind} is not implemented yet")
    from peft import LoraConfig, get_peft_model
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                                             target_modules=LORA_TARGETS.split(",")))
    corpus = torch.load(data or DATA["llama3"])["train"][:, :seq_len]
    params = [p for p in model.parameters() if p.requires_grad] + sch.trainable_params()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    g = torch.Generator().manual_seed(seed)
    cfg = dict(scheme=scheme, base=base, steps=steps, bs=bs, accum=accum, seq_len=seq_len, lr=lr, rank=rank, seed=seed)
    json.dump(cfg, open(f"{out}/embed_args.json", "w"), indent=1)
    log = open(f"{out}/train_log.jsonl", "a"); t0 = time.time()
    model.train()
    for step in range(steps):
        tot = 0.0
        sch.on_step(step, steps)
        for _ in range(accum):
            ids = corpus[torch.randint(len(corpus), (bs,), generator=g)].to(device)
            with torch.no_grad(), model.disable_adapter():
                base_logits = model(input_ids=ids).logits[:, :-1].float()
            t_logits = sch.target_logits(ids[:, :-1], base_logits)     # carries grad only for co-optimized schemes
            loss = kl_teacher_student(t_logits, model(input_ids=ids).logits[:, :-1])
            aux = sch.aux_loss()
            if aux is not None:
                loss = loss + aux
            loss = loss / accum
            loss.backward(); tot += loss.item(); del t_logits, base_logits
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        rec = {"step": step + 1, "kl": round(tot, 5), "sec": round(time.time() - t0, 1),
               "mem_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if (step + 1) % log_every == 0 or step == 0:
            print(rec, flush=True)
        if (step + 1) % save_every == 0 or step + 1 == steps:
            model.save_pretrained(f"{out}/adapter_step{step + 1}")
            sch.save(f"{out}/adapter_step{step + 1}", key)
    model.save_pretrained(f"{out}/adapter")
    key = sch.save(out, key)
    rel = Release(path=out, base=base, scheme=scheme, key=key, adapter=f"{out}/adapter",
                  lineage=[vars(Step("embed", cfg))])
    return rel.save()
