"""train.imprint: learn a scheme's mark into any adapter's weights, in phases.

Token schemes (Output.TOKENS) train by KL to target_logits, as imprint.embed does for text; field schemes
(Output.FIELD) train by MSE to target_output. The teacher is the unmarked model: the LoRA adapter switched off, or a
frozen copy for a full fine-tune. A phase can co-learn the scheme's own parameters (policy_lr > 0, PRO's CAWP) or fix
them (fix_rule: True), which gives Taming's "learn the green list, then fix it and train" recipe:
    phases=[{"steps": 800, "lr": 3e-5, "trainable": "full", "policy_lr": 0.2},
            {"steps": 1000, "lr": 1e-4, "trainable": "full", "fix_rule": True}]

Progress is saved at every save_every steps and at the end of each phase, in the key store (it holds the scheme's
trained parameters, i.e. key material, so never in `out`); a rerun with the same `out` and phases continues from
there (a finished run keeps nothing to resume), and the run's draft key (save_draft / draft_key) keeps the same secret. One log line per step goes to out/train_log.jsonl.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from .imprint import kl_teacher_student
from .models.base import Model, Output
from .types import Release, Step

PHASE = {"steps": 1000, "lr": 2e-4, "trainable": "lora", "rank": 64, "policy_lr": None, "fix_rule": False}


def _phases(phases: list[dict]) -> list[dict]:
    out = []
    for i, p in enumerate(phases):
        bad = sorted(set(p) - set(PHASE))
        if bad:
            raise ValueError(f"imprint phase {i}: unknown setting {', '.join(bad)}; a phase takes {', '.join(PHASE)}")
        out.append({**PHASE, **p})
    modes = {p["trainable"] for p in out}
    if len(modes) > 1:
        raise ValueError(f"imprint phases must all train the same weights (got trainable={sorted(modes)})")
    return out


def _cycle(batches: Iterable):
    while True:
        n = 0
        for b in batches:
            n += 1
            yield b
        if n == 0:
            raise ValueError("imprint got no training batches")


def _to(batch: dict, device: str) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def resume_dir(out: str) -> str:
    """Where an unfinished imprint into `out` keeps its state and draft key: a private directory in the key store."""
    import hashlib
    from . import keys
    return os.path.join(keys.STORE, "resume", hashlib.sha1(os.path.abspath(out).encode()).hexdigest()[:16])


def draft_key(out: str) -> dict | None:
    """The key an unfinished imprint into `out` started with (so a resumed run keeps its secret), or None."""
    p = os.path.join(resume_dir(out), "draft.json")
    return json.load(open(p)) if os.path.exists(p) else None


def save_draft(out: str, key: dict) -> None:
    from .files import write_json
    os.makedirs(resume_dir(out), exist_ok=True)
    write_json(os.path.join(resume_dir(out), "draft.json"), key)


def resuming(out: str) -> bool:
    return os.path.isdir(resume_dir(out))


def imprint(model: Model, scheme, batches: Iterable, out: str, *, phases: list[dict], device: str = "cpu",
            net=None, path: str | None = None, key: dict | None = None, base: str = "", save_every: int = 100,
            seed: int = 0, teacher=None, on_phase_end: Callable[[int], None] | None = None) -> Release:
    """Train the mark into `net` (or model.load(path, device)) and write a release to `out`.

    With `key` (from scheme.setup), the key is saved to the key store, never to `out`, and the release carries its
    commitment. The returned release has the trained module as `.net`. `teacher`: the unmarked model the targets
    are computed from, when `net` starts from something else (e.g. after geopt's closed-form edit_last); default:
    `net` as given (a frozen copy, or the LoRA adapter switched off)."""
    phases = _phases(phases)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)
    net = (net if net is not None else model.load(path, device)).to(device)
    mode = phases[0]["trainable"]
    field = scheme.output == Output.FIELD
    if mode == "lora":
        from peft import LoraConfig, get_peft_model
        for p in net.parameters():
            p.requires_grad_(False)
        cfg = dict(r=phases[0]["rank"], lora_alpha=2 * phases[0]["rank"], lora_dropout=0.0,
                   target_modules=model.lora_targets.split(",") if isinstance(model.lora_targets, str) else model.lora_targets)
        if hasattr(net, "config") and not field:
            cfg["task_type"] = "CAUSAL_LM"
        net = get_peft_model(net, LoraConfig(**cfg))        # without a teacher: the base = the adapter switched off
    else:
        teacher = teacher if teacher is not None else copy.deepcopy(net)
        for p in net.parameters():
            p.requires_grad_(True)
    if teacher is not None:
        teacher = teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    weights = [p for p in net.parameters() if p.requires_grad]

    rpath = os.path.join(resume_dir(out), "state.pt")
    state = torch.load(rpath, weights_only=False) if os.path.isfile(rpath) else {"phase": 0, "step": 0}
    if os.path.isfile(rpath):
        done = state["phases"][: state["phase"]] + ([state["phases"][state["phase"]]] if state["step"] else [])
        if phases[: len(done)] != done:
            raise ValueError(f"{out} holds a run with other imprint phases ({done}); use a new out directory")
        sd = dict(net.named_parameters())
        with torch.no_grad():
            for k, v in state["weights"].items():
                sd[k].copy_(v)
            for p, v in zip(scheme.trainable_params(), state["scheme"]):
                p.copy_(v)

    consumed = {"n": state.get("consumed", 0)}              # batches drawn so far, so a resumed run sees the same data

    def save(phase, step, opt=None, sched=None):
        os.makedirs(os.path.dirname(rpath), exist_ok=True)
        tmp = rpath + ".tmp"
        torch.save({"phase": phase, "step": step, "phases": phases, "consumed": consumed["n"],
                    "weights": {k: p.detach().cpu() for k, p in net.named_parameters() if p.requires_grad},
                    "scheme": [p.detach().cpu() for p in scheme.trainable_params()],
                    "opt": opt.state_dict() if opt is not None else None,
                    "sched": sched.state_dict() if sched is not None else None,
                    "torch_rng": torch.get_rng_state()}, tmp)
        os.replace(tmp, rpath)

    def teacher_out(batch):
        with torch.no_grad():
            if teacher is not None:
                return model.forward(teacher, batch)[1].float()
            with net.disable_adapter():
                return model.forward(net, batch)[1].float()

    it = _cycle(batches)
    for _ in range(consumed["n"]):                          # resumed: skip the batches already trained on
        next(it)
    if state.get("torch_rng") is not None:
        torch.set_rng_state(state["torch_rng"])
    log = open(os.path.join(out, "train_log.jsonl"), "a")
    t0 = time.time()
    net.train()
    for pi in range(state["phase"], len(phases)):
        ph = phases[pi]
        rule = scheme.trainable_params()
        frozen = ph["fix_rule"] or ph["policy_lr"] == 0
        for p in rule:
            p.requires_grad_(not frozen)
        groups = [{"params": weights}]
        if rule and not frozen:
            groups.append({"params": rule, "lr": ph["policy_lr"] if ph["policy_lr"] is not None else ph["lr"]})
        opt = torch.optim.AdamW(groups, lr=ph["lr"], weight_decay=0.0)
        steps = ph["steps"]
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * s / max(steps, 1))))
        start = state["step"] if pi == state["phase"] else 0
        if start and state.get("opt") is not None:          # mid-phase resume: the optimizer and schedule as saved
            opt.load_state_dict(state["opt"])
            sched.load_state_dict(state["sched"])
        else:
            for _ in range(start):
                sched.step()
        for step in range(start, steps):
            scheme.on_step(step, steps)
            batch = _to(next(it), device)
            consumed["n"] += 1
            ref = teacher_out(batch)
            inputs, student = model.forward(net, batch)
            if field:
                loss = F.mse_loss(student.float(), scheme.target_output(inputs, ref))
            else:
                loss = kl_teacher_student(scheme.target_logits(inputs, ref), student)
            aux = scheme.aux_loss()
            if aux is not None and not frozen:
                loss = loss + aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(weights, 1.0)     # the model's gradients only, as pro_taming's colearn
            opt.step(); sched.step()
            log.write(json.dumps({"phase": pi, "step": step + 1, "loss": round(loss.item(), 6),
                                  "sec": round(time.time() - t0, 1)}) + "\n"); log.flush()
            if (step + 1) % save_every == 0 and step + 1 < steps:
                save(pi, step + 1, opt, sched)
        save(pi + 1, 0)
        if on_phase_end:
            on_phase_end(pi)
    net.eval()

    spec = {"name": model.name, "params": model.params}
    if getattr(model, "source", None):
        spec["source"] = model.source                      # a plugin adapter: reopened with get_model(source=...)
    rel = Release(path=out, base=base, scheme=scheme.name, model=spec,
                  format="hf" if mode == "lora" else "state_dict",
                  lineage=[vars(Step("imprint", {"scheme": scheme.name, "phases": phases}))])
    if mode == "lora":
        net.save_pretrained(os.path.join(out, "adapter"))
        rel.adapter = os.path.join(out, "adapter")
    else:
        rel.weights = os.path.join(out, "weights.pt")
        torch.save(net.state_dict(), rel.weights + ".tmp"); os.replace(rel.weights + ".tmp", rel.weights)
    if key is not None:
        from .imprint import _drop_if_empty
        from .keys import stage_dir, store
        from .registry import resolved_params, scheme_source
        from .types import Key
        kdir = stage_dir()
        k = Key(scheme=scheme.name, params={**scheme.save(kdir, key), "rng_device": str(device).split(":")[0]},
                base=base, scheme_params=resolved_params(scheme), scheme_source=scheme_source(type(scheme)))
        store(k); _drop_if_empty(kdir)
        rel.key_commitment = k.commitment
    rel.save()
    import shutil
    shutil.rmtree(resume_dir(out), ignore_errors=True)     # finished: the key is stored, nothing left to resume
    rel.net = net
    return rel
