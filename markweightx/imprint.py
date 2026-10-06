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
          device="cuda:0", key: dict | None = None, data: str | None = None, save_every=500, seed=0, log_every=10,
          bos: bool = False, scheme_params: dict | None = None):
    os.makedirs(out, exist_ok=True)
    tok = load_tokenizer(base)
    model = load_base(base, device)
    from .registry import scheme_source
    scheme_params = dict(scheme_params or {})
    sch = get_scheme(scheme, **scheme_params)
    src = scheme_source(type(sch))
    key = sch.setup(model, tok, device, key)
    from .primitives import Imprint
    if sch.imprint == Imprint.EDIT:                        # teacher-free: the scheme writes the weights itself
        from .types import Key
        from .keys import store
        os.makedirs(out, exist_ok=True)
        weights = sch.imprint_edit(model, tok, out)
        k = Key(scheme=scheme, params=sch.save(out, key), base=base, scheme_params=scheme_params, scheme_source=src)
        store(k)
        return Release(path=out, base=base, scheme=scheme, key_commitment=k.commitment, weights=weights,
                       lineage=[vars(Step("embed", {"scheme": scheme, "imprint": "edit", "base": base}))]).save()
    if sch.kind != "distill_logit":
        raise NotImplementedError(f"{scheme}: recipe {sch.kind} is not implemented yet")
    from peft import LoraConfig, get_peft_model
    # non-reentrant checkpointing: Booster takes extra torch.autograd.grad passes, which reentrant checkpoints reject
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                                             target_modules=LORA_TARGETS.split(",")))
    corpus = torch.load(data or DATA["llama3"])["train"][:, :seq_len]
    if bos:   # start every chunk with BOS, as real prompts do: otherwise the position after BOS is never trained, and
              # FPL's perturbations broke it once (pro-full generated " Ur Ur Ur ..." only after BOS, 2026-10-05)
        corpus = torch.cat([torch.full((len(corpus), 1), tok.bos_token_id, dtype=corpus.dtype), corpus[:, :-1]], 1)
    with torch.no_grad():                                  # catch broken weights before hours of training
        for i in range(0, 64 if str(device).startswith("cuda") else 8, 8):
            if not torch.isfinite(model(input_ids=corpus[i:i + 8].to(device)).logits).all():
                raise RuntimeError(f"base model {base} gives non-finite logits on corpus[{i}:{i + 8}]: check the weights")
    lora = [p for p in model.parameters() if p.requires_grad]
    params = lora + sch.trainable_params()
    opt = torch.optim.AdamW([{"params": lora}] + sch.param_groups(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    g = torch.Generator().manual_seed(seed)
    cfg = dict(scheme=scheme, base=base, steps=steps, bs=bs, accum=accum, seq_len=seq_len, lr=lr, rank=rank, seed=seed, bos=bos)
    json.dump(cfg, open(f"{out}/embed_args.json", "w"), indent=1)
    log = open(f"{out}/train_log.jsonl", "a"); t0 = time.time(); skipped = 0
    booster = getattr(sch, "booster", None)                # PRO's Booster robust training (simulated fine-tune)
    attack_loss = []

    def attack_grads(att):
        """Gradient of the simulated attack: clean-text fine-tuning ("clean"), or, as PRO's FPL, learning the
        anti-watermarked distribution (base logits with the watermark bias subtracted) on the same text ("anti")."""
        lg = model(input_ids=att).logits[:, :-1].float()
        if booster.get("attack", "clean") == "anti":
            with torch.no_grad(), model.disable_adapter():
                base_lg = model(input_ids=att).logits[:, :-1].float()
            with torch.no_grad():
                anti = sch.anti_logits(att[:, :-1], base_lg)
            l = kl_teacher_student(anti, lg)
        else:
            l = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), att[:, 1:].reshape(-1))
        attack_loss.append(l.item())
        return torch.autograd.grad(l, lora)
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
            if not torch.isfinite(loss):                  # never step on a non-finite loss; say where it came from
                s_lg = model(input_ids=ids).logits[:, :-1].float()
                print({"skip_nonfinite": step + 1, "base_finite": bool(torch.isfinite(base_logits).all()),
                       "target_finite": bool(torch.isfinite(t_logits).all()),
                       "student_finite": bool(torch.isfinite(s_lg).all())}, flush=True)
                skipped += 1; del t_logits, base_logits, s_lg
                continue
            if booster:                                       # as in PRO's train_logit_distill_robust.py (booster mode):
                att = corpus[torch.randint(len(corpus), (bs,), generator=g)].to(device)   # clean text = the attack
                g1 = attack_grads(att)                        # 1. attack gradient
                gn = torch.sqrt(sum((x.float() ** 2).sum() for x in g1)) + 1e-7
                with torch.no_grad():                         # 2. take the attack step
                    for p_, x in zip(lora, g1):
                        p_.add_(-booster["alpha"] * x / gn)
                g2 = attack_grads(att)                        # 3. attack gradient after the step
                with torch.no_grad():                         # 4. undo the step
                    for p_, x in zip(lora, g1):
                        p_.add_(booster["alpha"] * x / gn)
            loss.backward(); tot += loss.item(); del t_logits, base_logits
            if booster:                                       # 5. anti-unlearning term lamb * (g1 - g2)
                with torch.no_grad():
                    for p_, x, y in zip(lora, g1, g2):
                        if p_.grad is not None:
                            p_.grad.add_(booster["lamb"] * (x - y) / accum)
                del g1, g2
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        rec = {"step": step + 1, "kl": round(tot, 5), "skipped": skipped,
               **({"attack": round(attack_loss[-2], 5)} if booster and len(attack_loss) >= 2 else {}), "sec": round(time.time() - t0, 1),
               "mem_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 1) if str(device).startswith("cuda") else 0}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if (step + 1) % log_every == 0 or step == 0:
            print(rec, flush=True)
        if (step + 1) % save_every == 0 or step + 1 == steps:
            model.save_pretrained(f"{out}/adapter_step{step + 1}")
            sch.save(f"{out}/adapter_step{step + 1}", key)
    model.save_pretrained(f"{out}/adapter")
    key = sch.save(out, key)
    from .keys import store
    from .types import Key
    # MarkLLM's KGW-family teachers draw their green lists from a generator on the model's device; CUDA and CPU give
    # different lists, so the detector must use the device type the key was made on (fork change A2).
    key = {**key, "rng_device": str(device).split(":")[0]}
    k = Key(scheme=scheme, params=key, base=base, scheme_params=scheme_params, scheme_source=src)
    store(k)                                               # the key goes to the key store, not into the release
    rel = Release(path=out, base=base, scheme=scheme, key_commitment=k.commitment, adapter=f"{out}/adapter",
                  lineage=[vars(Step("embed", cfg))])
    return rel.save()
