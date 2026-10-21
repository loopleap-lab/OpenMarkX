"""PRO for autoregressive image generation: WMAR's generation-time KGW watermark on Taming's image tokens, learned into
the transformer's weights (LoRA) by logit distillation, then detected with WMAR's detector, unchanged.

Run from the WMAR checkout (its `deps.taming` and `wmar` packages), e.g.
  cd $WMAR && python $B/MarkWeightX/exp/arimg/pro_taming.py data  --out $RUN --classes 0-999 --per_class 20 --shard 0 --num_shards 4
  cd $WMAR && python $B/MarkWeightX/exp/arimg/pro_taming.py train --out $RUN
  cd $WMAR && python $B/MarkWeightX/exp/arimg/pro_taming.py eval  --out $RUN --sets base,teacher,student
Teacher at position t = transformer logits + delta on the green list seeded by token t-1 (WMAR: linear seeding, context
1, stratified split, gamma 0.25, delta 2). Student = the same transformer + LoRA, trained with KL(teacher || student)
on the base model's own token sequences (no images, no labels). Every step resumes from what it saved.
"""
import argparse
import contextlib
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())                             # the WMAR checkout

WM = dict(seed="linear", split="stratifiedrand", context=1, delta=2.0, gamma=0.25)
GEN = dict(temperature=1.0, top_k=250, top_p=0.92)          # WMAR's Taming sampling settings
EVAL_CLASSES = [1, 9, 232, 340, 568, 656, 703, 814, 937, 975]   # WMAR's evaluation classes


def load(modelpath, enc_ft=None, dec_ft=None, seed_strategy=None):
    from wmar.models.taming_wrapper import TamingARMMWrapper
    from wmar.utils.utils import update_weights
    from wmar.watermarking.gentime_watermark import GentimeWatermark, SeedStrategy, SplitStrategy
    model = TamingARMMWrapper(modelpath)
    if enc_ft:
        update_weights(model.get_image_tokenizer().encoder, enc_ft)
    if dec_ft:
        update_weights(model.get_image_tokenizer().decoder, dec_ft)
    seed = seed_strategy or WM["seed"]
    wm = GentimeWatermark(model.get_vq(), model.get_total_vocab_size(), SeedStrategy(seed),
                          SplitStrategy(WM["split"]), 0 if seed == "fixed" else WM["context"], WM["delta"], WM["gamma"],
                          "cuda")
    model.set_watermarker(wm)
    return model, wm


def green_table(wm):
    """[V, V] uint8: row c = WMAR's green list when the previous token is c (context size 1)."""
    V = wm.vocab_size
    if wm.fixed_greenlist is not None:                  # fixed seeding: one green list for every context
        G = torch.zeros(V, V, dtype=torch.uint8); G[:, wm.fixed_greenlist.cpu()] = 1
        return G
    G = torch.zeros(V, V, dtype=torch.uint8)
    for c in range(V):
        G[c, wm._get_greenlist_ids_for_context(torch.tensor([c])).cpu()] = 1
    return G


class Holder(torch.nn.Module):
    """Holds Taming's GPT for PEFT: PEFT reads model.config as a dict, and Taming's GPTConfig is not one."""

    def __init__(self, gpt):
        super().__init__()
        self.m = gpt


class SkipGPT(torch.nn.Module):
    """Taming's GPT plus a table indexed by the input token, added to the logits (--skip): a gradient-learned module
    that reads WMAR's context (the previous token) directly, which the frozen GPT then never has to carry forward."""

    def __init__(self, gpt, V):
        super().__init__()
        self.m, self.V = gpt, V
        self.skip = torch.nn.Embedding(V, V).to(gpt.head.weight.device)
        torch.nn.init.zeros_(self.skip.weight)

    def add(self, logits, idx):
        return torch.cat([logits[..., :self.V] + self.skip(idx).to(logits.dtype), logits[..., self.V:]], -1)

    def forward_with_past(self, idx, **kw):             # what WMAR's sampler calls, one token at a time
        logits, loss, present = self.m.forward_with_past(idx, **kw)
        return self.add(logits, idx), loss, present


def gpt_logits(gpt, idx):
    """Taming's GPT forward for training (WMAR's copy disables GPT.forward in favour of cached sampling)."""
    x = gpt.tok_emb(idx) + gpt.pos_emb[:, :idx.shape[1], :]
    x = gpt.ln_f(gpt.blocks(gpt.drop(x)))
    return gpt.head(x)


def classes_arg(s):
    if "-" in s and "," not in s:
        lo, hi = map(int, s.split("-")); return list(range(lo, hi + 1))
    return [int(c) for c in s.split(",")]


@torch.no_grad()
def sample_codes(model, classes, n_per, bs, seed, apply_watermark=False, transformer=None):
    """[N, 256] codes for every class in `classes`, n_per each, sampled with WMAR's settings."""
    out = []
    jobs = [c for c in classes for _ in range(n_per)]
    tf = model.model.transformer
    if transformer is not None:
        model.model.transformer = transformer
    try:
        for i in range(0, len(jobs), bs):
            torch.manual_seed(seed * 1_000_003 + i)
            out.append(model.sample(jobs[i:i + bs], GEN, apply_watermark=apply_watermark).cpu())
            print({"sampled": min(i + bs, len(jobs)), "of": len(jobs)}, flush=True)
    finally:
        model.model.transformer = tf
    return torch.cat(out), torch.tensor(jobs)


def cmd_data(a):
    model, _ = load(a.modelpath, seed_strategy=run_cfg(a))
    os.makedirs(a.out, exist_ok=True)
    path = f"{a.out}/data.s{a.shard}of{a.num_shards}.pt"
    if os.path.exists(path):
        print({"data": path, "note": "already there"}); return
    classes = classes_arg(a.classes)[a.shard::a.num_shards]
    codes, cond = sample_codes(model, classes, a.per_class, a.bs, seed=100 + a.shard, apply_watermark=a.watermarked)
    torch.save({"codes": codes, "cond": cond}, path)
    print({"data": path, "n": len(codes)})


def run_cfg(a):
    """The watermark seeding of a run: from its wm.json when it exists, else from the flags (and written there)."""
    p = f"{a.out}/wm.json"
    if os.path.exists(p):
        return json.load(open(p))["seed"]
    os.makedirs(a.out, exist_ok=True)
    json.dump({"seed": a.wm_seed}, open(p, "w")); return a.wm_seed


def cmd_train(a):
    import glob
    from peft import LoraConfig, get_peft_model
    model, wm = load(a.modelpath, seed_strategy=run_cfg(a))
    files = sorted(glob.glob(f"{a.data_dir or a.out}/data.s*of*.pt"))
    d = [torch.load(f) for f in files]
    codes = torch.cat([x["codes"] for x in d]); cond = torch.cat([x["cond"] for x in d])
    seqs = torch.cat([cond[:, None], codes], 1)             # [N, 257]: class token, then 256 image tokens
    print({"sequences": len(seqs), "files": files}, flush=True)
    G = green_table(wm).cuda() if a.objective == "kl" else None   # [V, V]
    gpt = model.model.transformer
    gpt.eval()                                              # no dropout; gradients still flow into LoRA
    for p in gpt.parameters():
        p.requires_grad_(False)
    if a.skip:                                          # only the input-token table; the GPT stays as released
        skipper = SkipGPT(gpt, V := wm.vocab_size)
        teacher_gpt, student, lora = gpt, None, list(skipper.skip.parameters())
    elif a.full or a.head_emb:                            # direct training: a frozen copy is the teacher's base
        import copy
        a.full = True
        teacher_gpt = copy.deepcopy(gpt).eval()
        for n, p in gpt.named_parameters():             # head_emb: only the token embedding and the output head, the
            p.requires_grad_(not a.head_emb or n.startswith(("tok_emb", "head")))   # two ends of the lookup it learns
        student = None; lora = [p for p in gpt.parameters() if p.requires_grad]
    else:
        targets = ["key", "query", "value", "proj", "mlp.0", "mlp.2"] + (["head"] if a.lora_head else [])
        student = get_peft_model(Holder(gpt), LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0,
                                                         target_modules=targets))
        lora = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * s / a.steps)))
    state, start, rng = f"{a.out}/resume.pt", 0, random.Random(0)
    if os.path.exists(state):
        st = torch.load(state, weights_only=False)
        if a.skip:
            skipper.skip.load_state_dict(st["lora"])
        elif a.full:
            gpt.load_state_dict(st["lora"])
        else:
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(student, st["lora"])
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        rng.setstate(st["rng"]); start = st["step"]; print({"resumed_at_step": start}, flush=True)
    log = open(f"{a.out}/train_log.jsonl", "a"); t0 = time.time()
    V = wm.vocab_size
    for step in range(start, a.steps):
        idx = seqs[rng.sample(range(len(seqs)), a.bs)].cuda()
        inp, ctx = idx[:, :-1], idx[:, :-1]                 # predict token t from tokens < t; WMAR's context = t-1
        if a.skip:
            with torch.no_grad():
                s0 = gpt_logits(gpt, inp).float()[..., :V]
            s = s0 + skipper.skip(inp).float()
        else:
            s = gpt_logits(gpt if a.full else student.base_model.model.m, inp).float()[..., :V]
        if a.objective == "ce":                         # sampling distillation: next-token loss on the watermarked
            loss = F.cross_entropy(s.reshape(-1, V), idx[:, 1:].reshape(-1))   # samples (data made with --watermarked)
        else:
            with torch.no_grad(), (student.disable_adapter() if student is not None else contextlib.nullcontext()):
                base = s0 if a.skip else gpt_logits(teacher_gpt if a.full else gpt, inp).float()[..., :V]
                target = base + (a.target_delta or WM["delta"]) * G[ctx].float()
            t_logp, s_logp = F.log_softmax(target, -1), F.log_softmax(s, -1)
            loss = (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(lora, 1.0); opt.step(); sched.step()
        if (step + 1) % 10 == 0 or step == 0:
            rec = {"step": step + 1, "kl": round(loss.item(), 5), "sec": round(time.time() - t0, 1),
                   "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
            print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if (step + 1) % a.resume_every == 0:
            from peft import get_peft_model_state_dict
            torch.save({"lora": skipper.skip.state_dict() if a.skip else gpt.state_dict() if a.full
                        else get_peft_model_state_dict(student), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "rng": rng.getstate(), "step": step + 1}, state + ".tmp")
            os.replace(state + ".tmp", state)
    if a.skip:
        torch.save({"skip": skipper.skip.state_dict()}, f"{a.out}/skip.pt")
    elif a.full:
        torch.save(gpt.state_dict(), f"{a.out}/transformer.pt")
    else:
        student.save_pretrained(f"{a.out}/adapter")
    if os.path.exists(state):
        os.remove(state)
    print({"saved": a.out})


def policy_table(Z, gamma):
    """Hard green lists from the policy's scores: the top gamma share of alive codes in every row (context)."""
    k = round(gamma * Z.shape[1])
    return torch.zeros_like(Z, dtype=torch.bool).scatter_(1, Z.topk(k, 1).indices, True)


def detect_table(codes, Gp, alive, gamma):
    """WMAR-style detection with a green table over alive codes: score each distinct (previous, current) pair once,
    z-test on the green count; returns one-sided p-values."""
    pos = torch.full((int(alive.max()) + 1,), -1, dtype=torch.long, device=codes.device)
    pos[alive] = torch.arange(len(alive), device=codes.device)
    out = []
    for c in codes:
        a_, b_ = pos[c[:-1]], pos[c[1:]]
        ok = (a_ >= 0) & (b_ >= 0)
        pairs = torch.unique(torch.stack([a_[ok], b_[ok]], 1), dim=0)
        n = len(pairs); g = Gp[pairs[:, 0], pairs[:, 1]].sum().item() if n else 0
        z = (g - gamma * n) / math.sqrt(max(n, 1) * gamma * (1 - gamma))
        out.append(0.5 * math.erfc(z / math.sqrt(2)))
    return torch.tensor(out)


def cmd_colearn(a):
    """PRO's idea on image tokens: the watermark policy and the model learn together. The policy is a score table over
    (previous alive code, next alive code), started at WMAR's linear-seeding green table; its green lists are the top
    gamma share of each row (hard, straight-through). The GPT (full fine-tune) is distilled toward base + delta * green
    list, and the same KL pulls the policy toward green lists the GPT can express; a column-balance term keeps every
    code green in about gamma of the contexts, so the policy cannot collapse to one fixed list."""
    import copy, glob
    model, wm = load(a.modelpath, seed_strategy=run_cfg(a))
    files = sorted(glob.glob(f"{a.data_dir or a.out}/data.s*of*.pt"))
    d = [torch.load(f) for f in files]
    seqs = torch.cat([torch.cat([x["cond"][:, None], x["codes"]], 1) for x in d])
    alive = wm.alive_ids.cuda(); A = len(alive); gam = WM["gamma"]
    pos = torch.full((wm.vocab_size,), -1, dtype=torch.long, device="cuda"); pos[alive] = torch.arange(A, device="cuda")
    G0 = green_table(wm)[wm.alive_ids.cpu()][:, wm.alive_ids.cpu()].float().cuda()
    Z = torch.nn.Parameter(4.0 * (G0 - 0.5))                # the policy, started at WMAR's table
    gpt = model.model.transformer.eval()
    teacher_gpt = copy.deepcopy(gpt).eval()
    for p_ in teacher_gpt.parameters():
        p_.requires_grad_(False)
    if a.init_dir:                                      # continue from a finished colearn run (its GPT and policy)
        gpt.load_state_dict(torch.load(f"{a.init_dir}/transformer.pt"))
        Z.data.copy_(torch.load(f"{a.init_dir}/policy.pt")["Z"].cuda())
        print({"init_from": a.init_dir}, flush=True)
    if a.policy_lr == 0:                                # policy fixed: the GPT alone catches up with it
        Z.requires_grad_(False)
    opt = torch.optim.AdamW([{"params": list(gpt.parameters()), "lr": a.lr},
                             {"params": [Z], "lr": a.policy_lr}] if a.policy_lr else
                            [{"params": list(gpt.parameters()), "lr": a.lr}], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * s / a.steps)))
    state, start, rng = f"{a.out}/resume.pt", 0, random.Random(0)
    if os.path.exists(state):
        st = torch.load(state, weights_only=False)
        gpt.load_state_dict(st["gpt"]); Z.data.copy_(st["Z"])
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        rng.setstate(st["rng"]); start = st["step"]; print({"resumed_at_step": start}, flush=True)
    log = open(f"{a.out}/train_log.jsonl", "a"); t0 = time.time()
    for step in range(start, a.steps):
        idx = seqs[rng.sample(range(len(seqs)), a.bs)].cuda()
        inp = idx[:, :-1]; ctx = pos[inp]; m = ctx >= 0   # positions whose context is an alive code
        s = gpt_logits(gpt, inp).float()[..., alive][m]
        with torch.no_grad():
            base = gpt_logits(teacher_gpt, inp).float()[..., alive][m]
        soft = torch.sigmoid(Z)
        hard = policy_table(Z.detach(), gam).float()
        g = hard + soft - soft.detach() if Z.requires_grad else hard                     # straight-through: hard lists forward, soft gradient
        t_logp = F.log_softmax(base + WM["delta"] * g[ctx[m]], -1); s_logp = F.log_softmax(s, -1)
        kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()
        bal = (soft.mean(0) - gam).pow(2).mean() + (soft.mean(1) - gam).pow(2).mean()
        # the unmarked model must not already favour the green lists: under base, the green mass stays near gamma
        null = ((F.softmax(base, -1) * g[ctx[m]]).sum(-1) - gam).pow(2).mean()
        loss = kl + a.balance * bal + a.null_w * null
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(gpt.parameters()), 1.0); opt.step(); sched.step()
        if (step + 1) % 10 == 0 or step == 0:
            with torch.no_grad():
                hp = policy_table(Z, gam)
                rec = {"step": step + 1, "kl": round(kl.item(), 5), "bal": round(bal.item(), 6), "null": round(null.item(), 5),
                       "same_as_wmar": round((hp == G0.bool()).float().mean().item(), 4),
                       "col_green_min_max": [round(hp.float().mean(0).min().item(), 3), round(hp.float().mean(0).max().item(), 3)],
                       "sec": round(time.time() - t0, 1), "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
            print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if (step + 1) % a.resume_every == 0:
            torch.save({"gpt": gpt.state_dict(), "Z": Z.data, "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "rng": rng.getstate(), "step": step + 1}, state + ".tmp")
            os.replace(state + ".tmp", state)
    torch.save(gpt.state_dict(), f"{a.out}/transformer.pt")
    torch.save({"green": policy_table(Z.detach(), gam).cpu(), "alive": wm.alive_ids.cpu(), "Z": Z.detach().cpu()},
               f"{a.out}/policy.pt")
    if os.path.exists(state):
        os.remove(state)
    print({"saved": a.out})


def cmd_eval(a):
    from peft import PeftModel
    model, wm = load(a.modelpath, a.enc_ft, a.dec_ft, seed_strategy=run_cfg(a))
    res_path = f"{a.out}/eval.json"
    if a.student_ckpt and not os.path.exists(f"{a.out}/policy.pt"):   # a colearn resume state carries its policy
        st = torch.load(a.student_ckpt, map_location="cpu", weights_only=False)
        if isinstance(st, dict) and "Z" in st:
            torch.save({"green": policy_table(st["Z"], WM["gamma"]), "alive": wm.alive_ids.cpu(), "Z": st["Z"]},
                       f"{a.out}/policy.pt")
        del st
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    classes = classes_arg(a.eval_classes)
    for name in a.sets.split(","):
        cpath = f"{a.out}/eval_codes_{name}.pt"
        if not os.path.exists(cpath):
            if name == "base":
                codes, cond = sample_codes(model, classes, a.eval_per_class, a.bs, seed=7)
            elif name == "teacher":
                codes, cond = sample_codes(model, classes, a.eval_per_class, a.bs, seed=8,
                                           apply_watermark=True)
            elif name == "student":
                import copy                         # PEFT edits the module it wraps: keep the base intact
                if os.path.exists(f"{a.out}/skip.pt"):     # --skip run: the released GPT plus its token table
                    student = SkipGPT(model.model.transformer, wm.vocab_size)
                    student.skip.load_state_dict(torch.load(f"{a.out}/skip.pt")["skip"]); student.eval()
                elif a.student_ckpt:                  # a training run's resume state or saved transformer, read as is
                    st = torch.load(a.student_ckpt, map_location="cpu", weights_only=False)
                    student = copy.deepcopy(model.model.transformer)
                    student.load_state_dict(st["lora"] if "lora" in st else st["gpt"] if "gpt" in st else st); student.eval()
                    print({"student_from": a.student_ckpt, "step": st.get("step") if isinstance(st, dict) else None})
                elif os.path.exists(f"{a.out}/transformer.pt"):
                    student = copy.deepcopy(model.model.transformer)
                    student.load_state_dict(torch.load(f"{a.out}/transformer.pt")); student.eval()
                else:
                    student = PeftModel.from_pretrained(Holder(copy.deepcopy(model.model.transformer)),
                                                        f"{a.out}/adapter").merge_and_unload().m.eval()
                codes, cond = sample_codes(model, classes, a.eval_per_class, a.bs, seed=9, transformer=student)
                del student
            torch.save({"codes": codes, "cond": cond}, cpath)
        d = torch.load(cpath)
        codes = d["codes"].cuda()
        # detection as WMAR: on the generated tokens, and after image -> tokens (decode, re-encode)
        with torch.no_grad():
            re = torch.cat([model.images_to_codes(model.codes_to_images(codes[i:i + 25]))
                            for i in range(0, len(codes), 25)])
        if os.path.exists(f"{a.out}/policy.pt"):           # co-learned policy: detect with its green table
            pol = torch.load(f"{a.out}/policy.pt")
            Gp, al = pol["green"].cuda(), pol["alive"].cuda()
            p_gen, p_img = detect_table(codes, Gp, al, WM["gamma"]), detect_table(re, Gp, al, WM["gamma"])
        else:
            p_gen, p_img = wm.detect(codes).cpu(), wm.detect(re).cpu()
        res[name] = {"n": len(codes), "frac_p_lt_0.01_tokens": (p_gen < 0.01).float().mean().item(),
                     "frac_p_lt_0.01_image": (p_img < 0.01).float().mean().item(),
                     "median_p_image": p_img.median().item(),
                     "token_match_after_roundtrip": (re == codes).float().mean().item()}
        torch.save({"p_gen": p_gen, "p_img": p_img, "codes_roundtrip": re.cpu()}, f"{a.out}/eval_p_{name}.pt")
        print(json.dumps({name: res[name]}), flush=True)
        json.dump(res, open(res_path, "w"), indent=1)
    # empirical 1% FPR on the image route: the base images' p-values set the threshold
    if "base" in res:
        pb = torch.load(f"{a.out}/eval_p_base.pt")["p_img"]
        thr = torch.quantile(pb, 0.01).item()
        for name in res:
            if isinstance(res[name], dict) and os.path.exists(f"{a.out}/eval_p_{name}.pt"):
                p = torch.load(f"{a.out}/eval_p_{name}.pt")["p_img"]
                res[name]["tpr_at_empirical_1fpr_image"] = (p < thr).float().mean().item()
        res["empirical_threshold_p"] = thr
        json.dump(res, open(res_path, "w"), indent=1)
    print(json.dumps(res, indent=1))


def cmd_fold(a):
    """WMAR's bias folded into the GPT's own weights (token embedding and output head; architecture unchanged). The
    green list depends on the input token only, so: write a code for the input token into the residual stream at the
    embedding, in directions the blocks barely read and the final hidden state barely uses; the residual path carries
    it to ln_f, and the head reads it back as delta * green list (rank-r SVD of the centered green table). Scans rank
    and scale on held-out sequences by KL to the decode-time teacher; saves the best as transformer.pt."""
    import glob
    model, wm = load(a.modelpath, seed_strategy=run_cfg(a))
    gpt = model.model.transformer.eval()
    files = sorted(glob.glob(f"{a.data_dir or a.out}/data.s*of*.pt"))
    d = [torch.load(f) for f in files]
    seqs = torch.cat([torch.cat([x["cond"][:, None], x["codes"]], 1) for x in d])
    perm = torch.randperm(len(seqs), generator=torch.Generator().manual_seed(0))
    stat, held = seqs[perm[:64]], seqs[perm[64:192]]
    alive = wm.alive_ids.cuda()
    G = green_table(wm)[:, wm.alive_ids.cpu()].float().cuda()    # [V, A]
    Ga = G[alive]                                                 # [A, A]: alive context -> alive green flags
    U, S, Vh = torch.linalg.svd(WM["delta"] * (Ga - Ga.mean(1, keepdim=True)))
    D = gpt.head.in_features
    with torch.no_grad():
        # how strongly each block reads each direction (through its LayerNorm, scaled by that layer's residual size)
        x = gpt.tok_emb(stat[:, :-1].cuda()) + gpt.pos_emb[:, :256, :]
        C = torch.zeros(D, D, device="cuda", dtype=torch.float64)
        for blk in gpt.blocks:
            sig2 = x.var(-1, unbiased=False).mean().double()
            for ln, Ws in ((blk.ln1, [blk.attn.key.weight, blk.attn.query.weight, blk.attn.value.weight]),
                           (blk.ln2, [blk.mlp[0].weight])):
                M = (torch.cat(Ws).double() * ln.weight.double())
                C += M.T @ M / (M.pow(2).sum() * sig2)
            x = blk(x)
        sig = x.std(-1, unbiased=False).mean().item()             # typical size of the final residual
        z = (x - x.mean(-1, keepdim=True)) / x.std(-1, unbiased=False, keepdim=True)
        z = z.reshape(-1, D).double(); Cz = (z.T @ z) / len(z)
        Mh = gpt.head.weight[alive].double() * gpt.ln_f.weight.double()    # the head's own reading of ln_f's output
        Ch = Mh.T @ Mh
        C = C / C.trace() + Cz / Cz.trace() + Ch / Ch.trace()
        E = torch.stack([torch.ones(D, device="cuda", dtype=torch.float64),
                         (gpt.ln_f.bias / gpt.ln_f.weight).double()], 1)
        E, _ = torch.linalg.qr(E)
        P = torch.eye(D, device="cuda", dtype=torch.float64) - E @ E.T
        C = P @ C @ P + 1e3 * C.trace() * (E @ E.T)               # never use the mean or ln_f-bias directions
        Q = torch.linalg.eigh(C)[1].float()                        # ascending: quietest directions first
        gam = gpt.ln_f.weight.float()

        def kl(data):
            tot, n = 0.0, 0
            for i in range(0, len(data), 32):
                inp = data[i:i + 32, :-1].cuda()
                s = gpt_logits(gpt, inp).float()[..., alive]
                tl = F.log_softmax(base_cache[i] + WM["delta"] * G[inp], -1); sl = F.log_softmax(s, -1)
                tot += (tl.exp() * (tl - sl)).sum(-1).mean().item(); n += 1
            return tot / n
        base_cache = {i: gpt_logits(gpt, held[i:i + 32, :-1].cuda()).float()[..., alive]
                      for i in range(0, len(held), 32)}
        emb0, head0 = gpt.tok_emb.weight[alive].clone(), gpt.head.weight[alive].clone()
        res = {"kl_base": kl(held), "sigma_final": sig, "scan": []}
        print(res, flush=True)
        best = None
        for r in [int(x) for x in a.fold_ranks.split(",")]:
            Qr = Q[:, :r]
            Mm, Vs = U[:, :r] * S[:r].sqrt(), Vh[:r].T * S[:r].sqrt()
            for sc in [float(x) for x in a.fold_scales.split(",")]:
                gpt.tok_emb.weight[alive] = emb0 + sc * sig * (Mm @ Qr.T)
                gpt.head.weight[alive] = head0 + (Vs @ Qr.T) / (sc * gam)
                k = kl(held); rec = {"rank": r, "scale": sc, "kl": k}
                res["scan"].append(rec); print(rec, flush=True)
                if best is None or k < best[0]:
                    best = (k, r, sc, gpt.tok_emb.weight[alive].clone(), gpt.head.weight[alive].clone())
        k, r, sc, e, h = best
        gpt.tok_emb.weight[alive] = e; gpt.head.weight[alive] = h
        res["best"] = {"rank": r, "scale": sc, "kl": k}
    torch.save(gpt.state_dict(), f"{a.out}/transformer.pt")
    json.dump(res, open(f"{a.out}/fold.json", "w"), indent=1)
    print({"best": res["best"], "saved": f"{a.out}/transformer.pt"}, flush=True)


def cmd_edit(a):
    """Closed-form PRO for WMAR: the green list depends only on the previous token, so solve the change of the output
    head that adds delta on that token's green list, by ridge regression of the wanted logit shift on the final hidden
    features (ln_f output) over the base model's own sequences; restricted to the codes the model ever uses (alive)."""
    import glob
    model, wm = load(a.modelpath, seed_strategy=run_cfg(a))
    gpt = model.model.transformer.eval()
    files = sorted(glob.glob(f"{a.data_dir or a.out}/data.s*of*.pt"))
    d = [torch.load(f) for f in files]
    seqs = torch.cat([torch.cat([x["cond"][:, None], x["codes"]], 1) for x in d])
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(seqs), generator=g)
    fit, held = seqs[perm[:a.edit_n]], seqs[perm[a.edit_n:a.edit_n + 200]]
    alive = wm.alive_ids.cpu()
    G = green_table(wm)[:, alive].float()                   # [V, A]: green flags of the alive codes, per context
    feats = {}
    hook = gpt.ln_f.register_forward_hook(lambda m, i, o: feats.__setitem__("h", o.detach()))
    D = gpt.head.in_features
    HtH = torch.zeros(D + 1, D + 1, dtype=torch.float64, device="cuda")
    HtT = torch.zeros(D + 1, len(alive), dtype=torch.float64, device="cuda")
    Gc = (G - G.mean(1, keepdim=True)).cuda()               # logits move by a constant freely: center per context
    with torch.no_grad():
        for i in range(0, len(fit), 32):
            idx = fit[i:i + 32].cuda()
            gpt_logits(gpt, idx[:, :-1])
            h = feats["h"].reshape(-1, D).double()
            h = torch.cat([h, torch.ones_like(h[:, :1])], 1)
            ctx = idx[:, :-1].reshape(-1)
            HtH += h.T @ h; HtT += h.T @ (WM["delta"] * Gc[ctx]).double()
        lam = a.edit_ridge * HtH.diagonal().mean()
        W = torch.linalg.solve(HtH + lam * torch.eye(D + 1, device="cuda", dtype=torch.float64), HtT).float()   # [D+1, A]

        def kl_on(data, edited):
            tot, n = 0.0, 0
            for i in range(0, len(data), 32):
                idx = data[i:i + 32].cuda(); inp = idx[:, :-1]
                base = gpt_logits(gpt, inp).float()
                target = base[..., alive.cuda()] + WM["delta"] * G.cuda()[inp]
                s = base[..., alive.cuda()]
                if edited:
                    h = torch.cat([feats["h"], torch.ones_like(feats["h"][..., :1])], -1)
                    s = s + h.float() @ W
                tl, sl = F.log_softmax(target, -1), F.log_softmax(s, -1)
                tot += (tl.exp() * (tl - sl)).sum(-1).mean().item(); n += 1
            return tot / n
        before, after = kl_on(held, False), kl_on(held, True)
        print({"kl_before": before, "kl_after": after, "fit_sequences": len(fit)}, flush=True)
        head = gpt.head
        head.weight[alive.cuda()] += W[:-1].T               # head has no bias: fold the constant column into ln_f? keep
        # the constant term only shifts each alive logit by a per-code amount; apply it through the weight on a feature
        # direction is not exact, so it is left out (it does not depend on the context and barely matters)
    hook.remove()
    os.makedirs(a.out, exist_ok=True)
    torch.save(gpt.state_dict(), f"{a.out}/transformer.pt")
    json.dump({"kl_before": before, "kl_after": after, "edit_n": len(fit), "ridge": a.edit_ridge},
              open(f"{a.out}/edit.json", "w"), indent=1)
    print({"saved": f"{a.out}/transformer.pt"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "train", "eval", "edit", "fold", "colearn"])
    ap.add_argument("--modelpath", default=os.environ.get("TAMING"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", default="0-999")
    ap.add_argument("--per_class", type=int, default=20, help="data: sequences per class")
    ap.add_argument("--eval_per_class", type=int, default=100, help="eval: images per class (WMAR: 100)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--bs", type=int, default=50)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--full", action="store_true", help="full fine-tune instead of LoRA")
    ap.add_argument("--lora_head", action="store_true", help="LoRA also on the output head")
    ap.add_argument("--head_emb", action="store_true",
                    help="train only the token embedding and the output head (the green list depends on the previous "
                         "token only; 971 codes are ever used, below the 1536-wide hidden state)")
    ap.add_argument("--wm_seed", default="linear", choices=["linear", "fixed"],
                    help="WMAR seeding: linear = green list from the previous token; fixed = one green list")
    ap.add_argument("--edit_n", type=int, default=4000, help="edit: sequences used to fit the head change")
    ap.add_argument("--policy_lr", type=float, default=5e-2, help="colearn: learning rate of the policy table")
    ap.add_argument("--init_dir", default=None, help="colearn: start from a finished colearn run's GPT and policy")
    ap.add_argument("--null_w", type=float, default=0.0,
                    help="colearn: weight keeping the base model's green mass near gamma (no false positives)")
    ap.add_argument("--balance", type=float, default=10.0, help="colearn: weight of the column/row balance term")
    ap.add_argument("--fold_ranks", default="64,128,256,512", help="fold: SVD ranks of the green table to try")
    ap.add_argument("--fold_scales", default="0.25,0.5,1,2,4", help="fold: residual-code sizes to try (x final residual std)")
    ap.add_argument("--edit_ridge", type=float, default=1e-3)
    ap.add_argument("--skip", action="store_true",
                    help="train only a table indexed by the input token, added to the logits (GPT frozen)")
    ap.add_argument("--target_delta", type=float, default=None,
                    help="train toward a stronger bias than WMAR's delta 2 (the watermark and detector stay WMAR's): "
                         "the student only picks up part of it")
    ap.add_argument("--student_ckpt", default=None, help="eval: a full fine-tune's resume.pt or transformer.pt")
    ap.add_argument("--objective", default="kl", choices=["kl", "ce"],
                    help="kl: logit distillation from base + WMAR bias; ce: next-token loss on WMAR-watermarked samples")
    ap.add_argument("--watermarked", action="store_true", help="data: sample with WMAR's generation-time watermark")
    ap.add_argument("--resume_every", type=int, default=100,
                    help="steps between resume states (a full fine-tune's state is ~17 GB; short sweeps skip it)")
    ap.add_argument("--data_dir", default=None, help="where data.s*of*.pt are (default: --out)")
    ap.add_argument("--sets", default="base,teacher,student")
    ap.add_argument("--eval_classes", default=",".join(map(str, EVAL_CLASSES)))
    wmar = os.environ.get("WMAR", ".")
    ap.add_argument("--enc_ft", default=f"{wmar}/checkpoints/finetunes/taming_encoder_ft_noaug_delta.pth")
    ap.add_argument("--dec_ft", default=f"{wmar}/checkpoints/finetunes/taming_decoder_ft_noaug_delta.pth")
    a = ap.parse_args()
    {"data": cmd_data, "train": cmd_train, "eval": cmd_eval, "edit": cmd_edit, "fold": cmd_fold, "colearn": cmd_colearn}[a.cmd](a)


if __name__ == "__main__":
    main()
