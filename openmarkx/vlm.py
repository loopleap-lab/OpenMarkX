"""Image+text models with text output (MedGemma / Gemma 3): load, generate, and learn a watermark into the weights.

The watermark lives in the language model: LoRA covers only its projections, the vision tower stays frozen, and the
teacher signal is computed from the answer tokens alone (detection sees only the answer text). Each training example
is a chat prompt with its images followed by a fixed number of answer tokens, so every row of a batch has its answer
in the same columns.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import random
import time

import torch

from .imprint import kl_teacher_student, save_lora_release
from .paths import base_path
from .types import Release

LORA_TARGETS_LM = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def is_vlm(base: str) -> bool:
    from transformers import AutoConfig
    return "ForConditionalGeneration" in (AutoConfig.from_pretrained(base_path(base)).architectures or [""])[0]


def load_processor(base: str):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(base_path(base))
    proc.tokenizer.padding_side = "left"
    return proc


def load_vlm(base: str, device: str, dtype=torch.bfloat16):
    from transformers import AutoModelForImageTextToText
    return AutoModelForImageTextToText.from_pretrained(base_path(base), torch_dtype=dtype).to(device).eval()


def load_vlm_release(rel: Release, device: str):
    """The base with the release's adapter (switchable with disable_adapter())."""
    from peft import PeftModel
    return PeftModel.from_pretrained(load_vlm(rel.base, device), rel.adapter).eval()


def _image(x):
    from PIL import Image
    return (Image.open(x) if isinstance(x, str) else x).convert("RGB")


def chat_batch(proc, items):
    """Left-padded model inputs for items = [{"images": [path | PIL], "text": str}], each a single user turn."""
    texts, images = [], []
    for it in items:
        content = [{"type": "image"} for _ in it["images"]] + [{"type": "text", "text": it["text"]}]
        texts.append(proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                              add_generation_prompt=True))
        images.append([_image(x) for x in it["images"]])
    enc = proc(text=texts, images=images, padding=True, return_tensors="pt")
    enc["token_type_ids"] = image_token_types(enc["input_ids"], proc)
    return enc


def image_token_types(input_ids, proc):
    """1 on image tokens, 0 elsewhere (Gemma 3 attends bidirectionally within an image; training needs this)."""
    image_id = proc.tokenizer.convert_tokens_to_ids(proc.image_token) if hasattr(proc, "image_token") \
        else proc.image_token_id
    return (input_ids == image_id).long()


@torch.no_grad()
def generate_vlm(model, proc, items, *, new_tokens=768, min_new_tokens=None, bs=16, seed=0, adapter=True,
                 logits_processor=None, return_ids=False, partial: str | None = None):
    """Sample answers (temperature 1, no top-k/top-p, as the text-only recipe). Returns the decoded answers, or their
    token ids with return_ids. With `partial` (a .jsonl path), each finished batch is appended there and a rerun
    continues after the last one (every batch has its own seed, so a resumed run samples what a full run would)."""
    tok = proc.tokenizer
    kw = dict(max_new_tokens=new_tokens, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
              pad_token_id=tok.pad_token_id)
    if min_new_tokens:
        kw["min_new_tokens"] = min_new_tokens
    if logits_processor is not None:
        from transformers import LogitsProcessorList
        kw["logits_processor"] = LogitsProcessorList([logits_processor])
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    outs = [json.loads(l) for l in open(partial)] if partial and os.path.exists(partial) else []
    if outs:
        print({"resumed_at": len(outs), "of": len(items)}, flush=True)
    ctx = model.disable_adapter() if (not adapter and hasattr(model, "disable_adapter")) else contextlib.nullcontext()
    with ctx:
        for i in range(len(outs) - len(outs) % bs, len(items), bs):
            if i < len(outs):                                # a batch written only in part (should not happen)
                outs = outs[:i]
            torch.manual_seed(seed * 1_000_003 + i)
            enc = chat_batch(proc, items[i:i + bs]).to(model.device)
            enc["pixel_values"] = enc["pixel_values"].to(model.dtype)
            o = model.generate(**enc, **kw)[:, enc["input_ids"].shape[1]:]
            if return_ids:
                new = []
                for row in o.tolist():
                    cut = next((j for j, t in enumerate(row) if t in eos or t == tok.pad_token_id), len(row))
                    new.append(row[:cut])
            else:
                new = tok.batch_decode(o, skip_special_tokens=True)
            outs += new
            if partial:
                with open(partial, "a") as f:
                    f.write("".join(json.dumps(x) + "\n" for x in new))
            print({"generated": len(outs), "of": len(items)}, flush=True)
    return outs


def train_batch(proc, items, answer_ids):
    """Prompt (left-padded, with images) + answer tokens, all answers the same length A. Returns (inputs, A)."""
    A = len(answer_ids[0])
    assert all(len(a) == A for a in answer_ids), "every answer must have the same number of tokens"
    enc = chat_batch(proc, items)
    ans = torch.tensor(answer_ids, dtype=enc["input_ids"].dtype)
    enc["input_ids"] = torch.cat([enc["input_ids"], ans], 1)
    enc["attention_mask"] = torch.cat([enc["attention_mask"], torch.ones_like(ans)], 1)
    enc["token_type_ids"] = torch.cat([enc["token_type_ids"], torch.zeros_like(ans)], 1)
    return enc, A


def answer_logits(model, inputs, A):
    """Logits predicting the A answer tokens: [B, A, V]."""
    try:
        lg = model(**inputs, logits_to_keep=A + 1).logits
    except TypeError:                                       # a model without logits_to_keep
        lg = model(**inputs).logits[:, -(A + 1):]
    return lg[:, :-1]


def distill_loss(model, sch, inputs, A):
    """KL(teacher || student) on the answer tokens. Teacher = the scheme's signal on the base (adapter off)."""
    ids = inputs["input_ids"]
    region = ids[:, -(A + 1):-1]                            # the token before each answer token, and all before it
    with torch.no_grad(), model.disable_adapter():
        base = answer_logits(model, inputs, A).float()
    target = sch.target_logits(region, base)
    loss = kl_teacher_student(target, answer_logits(model, inputs, A))
    return loss, base, target


def lora_lm_only(model, rank=64):
    from peft import LoraConfig, get_peft_model
    return get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                                            target_modules=LORA_TARGETS_LM))


def load_pairs(path, A):
    """The distillation set written by exp/med/distill_data.py: {"images", "text", "answer_ids"} per line, answers
    cut to A tokens (shorter ones dropped). `path` may be a glob (all shards) or a comma-separated list."""
    import glob
    files = sorted(f for p in path.split(",") for f in (glob.glob(p) or [p]) if ".partial." not in f)
    rows = [json.loads(l) for f in files for l in open(f)]
    rows = [{**r, "answer_ids": r["answer_ids"][:A]} for r in rows if len(r["answer_ids"]) >= A]
    if not rows:
        raise ValueError(f"{path}: no answer has {A} tokens")
    return rows


def embed_vlm(scheme: str, base: str, out: str, pairs: str, *, steps=1500, bs=8, accum=1, answer_len=128, lr=2e-4,
              rank=64, device="cuda:0", seed=0, save_every=500, resume_every=100, log_every=10, grad_ckpt=True,
              scheme_params: dict | None = None, scheme_source: str | None = None, model=None, proc=None):
    """Learn a scheme's watermark into an image+text model's language model (LoRA), on image+question prompts and the
    base model's own answers. Same objective and schedule as openmarkx.imprint.embed.

    Resumable: every resume_every steps the adapter, optimizer, schedule and random states go to out/resume.pt, and
    the run's key to a private file in the key store; running the same call again continues from there with the
    same key. Both files are removed once the release is written. `model` and `proc` replace loading the base
    (an adapter passes them; the release still names `base`)."""
    from .registry import get_scheme, resolved_params, scheme_source as _src_of
    os.makedirs(out, exist_ok=True)
    proc = proc or load_processor(base)
    sch = get_scheme(scheme, source=scheme_source, **dict(scheme_params or {}))
    if sch.kind != "distill_logit" or getattr(sch, "booster", None):
        raise NotImplementedError(f"{scheme}: only plain logit distillation is ported to image+text models")
    src = scheme_source or _src_of(type(sch))
    scheme_params = resolved_params(sch)
    model = model if model is not None else load_vlm(base, device)
    dtype = model.dtype
    import hashlib
    from .files import write_json
    from .keys import STORE
    state_path = f"{out}/resume.pt"
    draft_path = os.path.join(STORE, "resume", hashlib.sha1(os.path.abspath(out).encode()).hexdigest()[:16] + ".json")
    resuming = os.path.exists(state_path) and os.path.exists(draft_path)
    key = sch.setup(model, proc.tokenizer, device, json.load(open(draft_path)) if resuming else None)
    if not resuming:                                        # the key must survive a restart: a new one would not match
        os.makedirs(os.path.dirname(draft_path), exist_ok=True)
        write_json(draft_path, key)
    rows = load_pairs(pairs, answer_len)
    if grad_ckpt:                                           # off on large GPUs: faster, more memory
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = lora_lm_only(model, rank)
    lora = [p for p in model.parameters() if p.requires_grad]
    params = lora + sch.trainable_params()
    opt = torch.optim.AdamW([{"params": lora}] + sch.param_groups(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    rng = random.Random(seed)
    cfg = dict(scheme=scheme, base=base, steps=steps, bs=bs, accum=accum, answer_len=answer_len, lr=lr, rank=rank,
               seed=seed, pairs=pairs, n_pairs=len(rows))
    json.dump(cfg, open(f"{out}/embed_args.json", "w"), indent=1)
    log = open(f"{out}/train_log.jsonl", "a"); t0 = time.time(); skipped = 0
    start = 0
    if resuming:
        from peft import set_peft_model_state_dict
        st = torch.load(state_path, map_location=device, weights_only=False)
        set_peft_model_state_dict(model, st["lora"])
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); rng.setstate(st["rng"])
        torch.set_rng_state(st["torch_rng"]); start, skipped = st["step"], st["skipped"]
        print({"resumed_at_step": start}, flush=True)

    def save_state(step):
        from peft import get_peft_model_state_dict
        tmp = state_path + ".tmp"
        torch.save({"lora": get_peft_model_state_dict(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "rng": rng.getstate(), "torch_rng": torch.get_rng_state(), "step": step, "skipped": skipped}, tmp)
        os.replace(tmp, state_path)                         # a job killed while saving keeps the previous state
    model.train()
    for step in range(start, steps):
        sch.on_step(step, steps)
        tot = 0.0
        for _ in range(accum):
            batch = rng.sample(rows, bs)
            inputs, A = train_batch(proc, batch, [r["answer_ids"] for r in batch])
            inputs = inputs.to(device); inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
            loss, base_lg, target = distill_loss(model, sch, inputs, A)
            aux = sch.aux_loss()
            if aux is not None:
                loss = loss + aux
            loss = loss / accum
            if not torch.isfinite(loss):
                print({"skip_nonfinite": step + 1, "base_finite": bool(torch.isfinite(base_lg).all()),
                       "target_finite": bool(torch.isfinite(target).all())}, flush=True)
                skipped += 1; del base_lg, target
                continue
            loss.backward(); tot += loss.item(); del base_lg, target
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        rec = {"step": step + 1, "kl": round(tot, 5), "skipped": skipped, "sec": round(time.time() - t0, 1),
               "mem_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 1) if str(device).startswith("cuda") else 0}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if (step + 1) % log_every == 0 or step == 0:
            print(rec, flush=True)
        if (step + 1) % save_every == 0 or step + 1 == steps:
            model.save_pretrained(f"{out}/adapter_step{step + 1}")
        if (step + 1) % resume_every == 0:
            save_state(step + 1)
    model.save_pretrained(f"{out}/adapter")
    rel = save_lora_release(sch, key, scheme, base, scheme_params, src, out, cfg, device)
    for f in (state_path, draft_path):
        if os.path.exists(f):
            os.remove(f)
    return rel
