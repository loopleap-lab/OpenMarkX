"""Image+text models (openmarkx.vlm) on a tiny random Gemma 3: LoRA stays in the language model, the loss covers
the answer tokens, and a few steps of distillation move the student toward the teacher."""
import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma3ForConditionalGeneration"):
    pytest.skip("transformers without Gemma 3", allow_module_level=True)

from openmarkx.models import text_vocab_size
from openmarkx.vlm import LORA_TARGETS_LM, answer_logits, distill_loss, lora_lm_only

V, IMG = 300, 299          # vocabulary; the image soft token is the last id


def tiny():
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration
    cfg = Gemma3Config(
        text_config=dict(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=16, sliding_window=16, max_position_embeddings=256,
                         initializer_range=0.5),   # large enough weights that the logits can move by a few units
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                           image_size=28, patch_size=14),
        mm_tokens_per_image=4, image_token_index=IMG, boi_token_index=297, eoi_token_index=298)
    torch.manual_seed(0)
    return Gemma3ForConditionalGeneration(cfg).eval()


def batch(B=2, P=10, A=6):
    g = torch.Generator().manual_seed(1)
    prompt = torch.randint(1, 290, (B, P), generator=g)
    prompt[:, 2:6] = IMG                                    # one image: 4 soft tokens
    ans = torch.randint(1, 290, (B, A), generator=g)
    ids = torch.cat([prompt, ans], 1)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "token_type_ids": (ids == IMG).long(),
            "pixel_values": torch.randn(B, 3, 28, 28, generator=g)}, A


class Green:
    """A toy teacher: +4 on a fixed half of the vocabulary."""
    def __init__(self):
        self.green = torch.zeros(V); self.green[::2] = 4.0

    def target_logits(self, ids, logits):
        return logits + self.green


def test_text_vocab_size():
    assert text_vocab_size(tiny()) == V


def test_lora_only_in_language_model():
    m = lora_lm_only(tiny(), rank=4)
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert names and all("language_model" in n for n in names), names
    assert not any("vision" in n or "projector" in n for n in names)


def test_answer_logits_shape_and_alignment():
    m = tiny(); inputs, A = batch()
    with torch.no_grad():
        lg = answer_logits(m, inputs, A)
        full = m(**inputs).logits
    assert lg.shape == (2, A, V)
    assert torch.allclose(lg, full[:, -(A + 1):-1], atol=1e-5)   # row t predicts answer token t


def test_distillation_moves_student_to_teacher():
    m = lora_lm_only(tiny(), rank=8); m.train()
    inputs, A = batch(); sch = Green()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    first = None
    for _ in range(30):
        loss, _, _ = distill_loss(m, sch, inputs, A)
        first = first if first is not None else loss.item()
        loss.backward(); opt.step(); opt.zero_grad()
    assert loss.item() < 0.5 * first, (first, loss.item())
    with torch.no_grad():
        p = answer_logits(m, inputs, A).softmax(-1)
        with m.disable_adapter():
            p0 = answer_logits(m, inputs, A).softmax(-1)
    green = sch.green > 0
    assert p[..., green].sum(-1).mean() > p0[..., green].sum(-1).mean() + 0.1


def _toy_embed(monkeypatch, tmp_path):
    """embed_vlm on the tiny model: the processor and the batches are synthetic, the training loop is the real one."""
    import json
    import types
    from openmarkx import vlm
    from openmarkx.registry import register_scheme
    from openmarkx.schemes.base import Scheme

    @register_scheme
    class ToyGreen(Scheme):
        name, kind = "test-vlm-green", "distill_logit"

        def setup(self, model, tok, device, key=None):
            super().setup(model, tok, device, key)
            return dict(key or {"secret": int(torch.randint(1 << 30, (1,)))})

        def target_logits(self, ids, logits):
            return logits + Green().green

    monkeypatch.setattr(vlm, "load_processor", lambda base: types.SimpleNamespace(tokenizer=None))
    monkeypatch.setattr(vlm, "load_vlm", lambda base, device: tiny())
    from transformers import BatchFeature
    monkeypatch.setattr(vlm, "train_batch", lambda proc, items, answers: (BatchFeature(batch(B=len(items))[0]), 6))
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps({"images": [], "text": "q", "answer_ids": [1] * 6}) + "\n" for _ in range(8)))
    kw = dict(steps=6, bs=2, answer_len=6, device="cpu", resume_every=3, save_every=100, lr=1e-2)
    return vlm, str(pairs), kw


def test_embed_resumes_with_the_same_key_and_state(monkeypatch, tmp_path):
    vlm, pairs, kw = _toy_embed(monkeypatch, tmp_path)
    full = vlm.embed_vlm("test-vlm-green", "tiny", str(tmp_path / "full"), pairs, **kw)

    # the same run, stopped after step 4 (as a job at its time limit), then started again
    out = str(tmp_path / "cut")
    real_step = torch.optim.AdamW.step
    calls = {"n": 0}

    def step_then_stop(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 5:
            raise KeyboardInterrupt("time limit")
        return real_step(self, *a, **k)
    monkeypatch.setattr(torch.optim.AdamW, "step", step_then_stop)
    with pytest.raises(KeyboardInterrupt):
        vlm.embed_vlm("test-vlm-green", "tiny", out, pairs, **kw)
    monkeypatch.setattr(torch.optim.AdamW, "step", real_step)
    import json
    import os
    assert os.path.exists(f"{out}/resume.pt")
    import glob
    from openmarkx.keys import STORE
    (draft,) = glob.glob(f"{STORE}/resume/*.json")
    draft_secret = json.load(open(draft))["secret"]
    resumed = vlm.embed_vlm("test-vlm-green", "tiny", out, pairs, **kw)
    assert not os.path.exists(f"{out}/resume.pt")

    import json
    log = [json.loads(l) for l in open(f"{out}/train_log.jsonl")]
    assert [r["step"] for r in log] == [1, 2, 3, 4, 4, 5, 6]     # stopped in step 5; state saved at 3; resumed at 4
    from openmarkx.keys import load_key
    assert load_key(resumed).params["secret"] == draft_secret   # the key drawn before the stop, not a new one
    from safetensors.torch import load_file
    a = load_file(f"{tmp_path}/full/adapter/adapter_model.safetensors")
    b = load_file(f"{out}/adapter/adapter_model.safetensors")
    assert all(torch.allclose(a[n], b[n], atol=1e-5) for n in a), "resumed training must end where a full run ends"


def test_generation_resumes_from_partial_file(monkeypatch, tmp_path):
    import types
    from transformers import BatchFeature
    from openmarkx import vlm
    m = tiny()
    monkeypatch.setattr(vlm, "chat_batch", lambda proc, items: BatchFeature(
        {k: v[:len(items)] for k, v in batch(B=8, A=0)[0].items()}))
    proc = types.SimpleNamespace(tokenizer=types.SimpleNamespace(pad_token_id=0))
    items = [{}] * 8
    kw = dict(new_tokens=5, bs=2, seed=3, return_ids=True)
    full = vlm.generate_vlm(m, proc, items, **kw)
    part = str(tmp_path / "p.jsonl")
    first = vlm.generate_vlm(m, proc, items[:4], partial=part, **kw)    # a job stopped after two batches
    assert first == full[:4]
    assert vlm.generate_vlm(m, proc, items, partial=part, **kw) == full
