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


def _drop_if_empty(d):
    try:
        os.rmdir(d)                                        # a stage_dir() that save() left empty
    except OSError:
        pass


def embed(scheme: str, base: str, out: str, *, steps=1500, bs=16, accum=1, seq_len=128, lr=2e-4, rank=64,
          device="cuda:0", key: dict | None = None, data: str | None = None, save_every=500, seed=0, log_every=10,
          bos: bool = False, scheme_params: dict | None = None, scheme_source: str | None = None):
    os.makedirs(out, exist_ok=True)
    tok = load_tokenizer(base)
    from .registry import scheme_source as _src_of
    from .keys import stage_dir
    from .primitives import Imprint
    scheme_params = dict(scheme_params or {})
    sch = get_scheme(scheme, source=scheme_source, **scheme_params)
    src = scheme_source or _src_of(type(sch))             # where a detector will find a plugin scheme again
    from .registry import resolved_params
    scheme_params = resolved_params(sch)                   # defaults included: the key must not follow later defaults
    # small weight edits must not be rounded to bf16: Imprint.EDIT schemes get an fp32 model
    model = load_base(base, device, dtype=torch.float32 if sch.imprint == Imprint.EDIT else torch.bfloat16)
    from .train import draft_key, resume_dir, save_draft
    key = sch.setup(model, tok, device, key if key is not None else draft_key(out))   # a resumed run keeps its key
    if sch.imprint == Imprint.EDIT:                        # teacher-free: the scheme writes the weights itself
        from .types import Key
        from .keys import store
        os.makedirs(out, exist_ok=True)
        weights = sch.imprint_edit(model, tok, out)
        kdir = stage_dir()
        params = {**sch.save(kdir, key), "rng_device": str(device).split(":")[0]}
        k = Key(scheme=scheme, params=params, base=base, scheme_params=scheme_params, scheme_source=src)
        store(k); _drop_if_empty(kdir)
        return Release(path=out, base=base, scheme=scheme, key_commitment=k.commitment, weights=weights,
                       lineage=[vars(Step("embed", {"scheme": scheme, "imprint": "edit", "base": base,
                                                    "scheme_params": scheme_params}))]).save()
    if sch.kind != "distill_logit":
        raise NotImplementedError(f"{scheme}: recipe {sch.kind} is not implemented yet")
    from peft import LoraConfig, get_peft_model
    # non-reentrant checkpointing: Booster takes extra torch.autograd.grad passes, which reentrant checkpoints reject
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                                             target_modules=LORA_TARGETS.split(",")))
    corpus = torch.load(data or DATA["llama3"], weights_only=True)["train"][:, :seq_len]
    if bos:   # start every chunk with BOS, as real prompts do: otherwise the position after BOS is never trained, and
              # FPL's perturbations broke it once (openmark-full, then named pro-full, generated " Ur Ur Ur ..." only after BOS, 2026-10-05)
        corpus = torch.cat([torch.full((len(corpus), 1), tok.bos_token_id, dtype=corpus.dtype), corpus[:, :-1]], 1)
    with torch.no_grad():                                  # catch broken weights before hours of training
        for i in range(0, 64 if str(device).startswith("cuda") else 8, 8):
            if not torch.isfinite(model(input_ids=corpus[i:i + 8].to(device)).logits).all():
                raise RuntimeError(f"base model {base} gives non-finite logits on corpus[{i}:{i + 8}]: check the weights")
    save_draft(out, key)
    lora = [p for p in model.parameters() if p.requires_grad]
    params = lora + sch.trainable_params()
    opt = torch.optim.AdamW([{"params": lora}] + sch.param_groups(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    g = torch.Generator().manual_seed(seed)
    cfg = dict(scheme=scheme, base=base, steps=steps, bs=bs, accum=accum, seq_len=seq_len, lr=lr, rank=rank, seed=seed, bos=bos)
    json.dump(cfg, open(f"{out}/embed_args.json", "w"), indent=1)
    log = open(f"{out}/train_log.jsonl", "a"); t0 = time.time(); skipped = 0
    booster = getattr(sch, "booster", None)                # OpenMark's Booster robust training (simulated fine-tune)
    attack_loss = []

    def attack_grads(att):
        """Gradient of the simulated attack: clean-text fine-tuning ("clean"), or, as OpenMark's FPL, learning the
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
    # progress (adapter, optimizer, schedule, random states) every save_every steps, in the key store: a run killed by a
    # time limit continues from the last save with the same key
    state_path = os.path.join(resume_dir(out), "state.pt")
    start = 0
    if os.path.exists(state_path):
        from peft import set_peft_model_state_dict
        st = torch.load(state_path, map_location=device, weights_only=False)
        set_peft_model_state_dict(model, st["lora"])
        with torch.no_grad():
            for p_, v in zip(sch.trainable_params(), st["scheme"]):
                p_.copy_(v)
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        g.set_state(st["gen"]); torch.set_rng_state(st["torch_rng"]); start, skipped = st["step"], st["skipped"]
        print({"resumed_at_step": start}, flush=True)

    def save_state(step):
        from peft import get_peft_model_state_dict
        torch.save({"lora": get_peft_model_state_dict(model), "scheme": [p_.detach().cpu() for p_ in sch.trainable_params()],
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "gen": g.get_state(),
                    "torch_rng": torch.get_rng_state(), "step": step, "skipped": skipped}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
    model.train()
    for step in range(start, steps):
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
            if step + 1 < steps:
                save_state(step + 1)
    model.save_pretrained(f"{out}/adapter")
    rel = save_lora_release(sch, key, scheme, base, scheme_params, src, out, cfg, device)
    import shutil
    shutil.rmtree(resume_dir(out), ignore_errors=True)   # finished: nothing left to resume
    return rel


def save_lora_release(sch, key, scheme, base, scheme_params, src, out, cfg, device):
    """Store the key (outside the release) and write the release around the trained adapter in out/adapter."""
    from .keys import stage_dir, store
    from .types import Key
    kdir = stage_dir()
    key = sch.save(kdir, key)                              # key material goes to the key store, never into out
    # MarkLLM's KGW-family teachers draw their green lists from a generator on the model's device; CUDA and CPU give
    # different lists, so the detector must use the device type the key was made on (fork change A2).
    key = {**key, "rng_device": str(device).split(":")[0]}
    k = Key(scheme=scheme, params=key, base=base, scheme_params=scheme_params, scheme_source=src)
    store(k); _drop_if_empty(kdir)                         # the key goes to the key store, not into the release
    rel = Release(path=out, base=base, scheme=scheme, key_commitment=k.commitment, adapter=f"{out}/adapter",
                  lineage=[vars(Step("embed", cfg))])
    return rel.save()
