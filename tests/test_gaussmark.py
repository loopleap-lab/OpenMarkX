"""GaussMark on a tiny random Llama (CPU): delta is reproducible, the null is N(0,1), marked text scores high."""
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch

import markweightx.keys as K
from markweightx.primitives import Access, Probe
from markweightx.registry import get_scheme


class TinyProbe(Probe):
    def __init__(self, model, tok):
        super().__init__("tiny", "cpu", Access.R2)
        self._model, self._tok = model, tok


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    from transformers import LlamaConfig, LlamaForCausalLM
    from markweightx.models import load_tokenizer
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=128256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4)
    # init exactly representable in bf16, like a real checkpoint
    return LlamaForCausalLM(cfg).to(torch.bfloat16).float().eval(), load_tokenizer("llama-3.2-1b"), tmp_path


def make(tiny, secret=1234, sigma=0.05):
    model, tok, tmp = tiny
    base = {k: v.clone() for k, v in model.state_dict().items()}
    sch = get_scheme("gaussmark", sigma=sigma, layer=1)
    key = sch.setup(model, tok, "cpu", {"secret": secret})
    edited = sch.imprint_edit(model, tok, str(tmp / f"rel{secret}"))
    key = sch.save(None, key)
    model.float(); model.load_state_dict(base)            # back to the unedited fp32 base (the probe uses it)
    return sch, key, edited


def test_delta_reproducible_and_key_pins_params(tiny):
    s1, k1, _ = make(tiny, secret=7)
    s2, k2, _ = make(tiny, secret=7)
    assert k1["delta_sha256"] == k2["delta_sha256"] and k1["base_sha256"] == k2["base_sha256"]
    assert k1["sigma"] == 0.05 and k1["layer"] == 1 and k1["matrix"] == "mlp.up_proj"
    assert make(tiny, secret=8)[1]["delta_sha256"] != k1["delta_sha256"]


def test_wrong_base_is_refused(tiny):
    model, tok, tmp = tiny
    sch, key, _ = make(tiny, secret=5)
    with torch.no_grad():
        dict(model.named_parameters())[sch.param].add_(0.1)  # a different "snapshot"
    det = get_scheme("gaussmark"); det.setup(model, tok, "cpu", key); det.probe = TinyProbe(model, tok)
    with pytest.raises(ValueError, match="base differs"):
        det.score("some text to score")


def test_null_is_standard_normal_over_secrets(tiny):
    """For a fixed text, psi over many secrets must be N(0,1): KS test."""
    from scipy.stats import kstest
    model, tok, _ = tiny
    det = get_scheme("gaussmark", sigma=0.01, layer=1); det.setup(model, tok, "cpu", {"secret": 1})
    probe = TinyProbe(model, tok)
    g = probe.grad_seq_logprob("The quick brown fox jumps over the lazy dog near the river", det.param)[det.param]
    base = dict(model.named_parameters())[det.param].detach().float()
    psi = []
    for secret in range(200):
        det.secret = secret
        d = det._realize(base, torch.bfloat16)
        psi.append(((d * g).sum() / (d.norm() / d.numel() ** 0.5 * g.norm())).item())
    assert kstest(psi, "norm").pvalue > 0.01


def test_null_is_standard_normal_and_marked_text_scores_high(tiny):
    model, tok, tmp = tiny
    # a random tiny model carries little signal per token: a large sigma and long texts make the test decisive;
    # power on a real model is measured by exp/learnability.py
    sch, key, edited = make(tiny, secret=99, sigma=0.2)
    det = get_scheme("gaussmark"); det.setup(model, tok, "cpu", key); det.probe = TinyProbe(model, tok)
    torch.manual_seed(1)
    null_texts = [tok.decode(torch.randint(1000, 30000, (60,))) for _ in range(60)]
    psi = [s for s, _ in det.score_batch(null_texts)]
    assert abs(statistics.mean(psi)) < 0.45 and 0.6 < statistics.stdev(psi) < 1.5
    from transformers import AutoModelForCausalLM
    marked_model = AutoModelForCausalLM.from_pretrained(edited, torch_dtype=torch.float32).eval()
    enc = tok(["The"] * 12, return_tensors="pt")
    with torch.no_grad():
        outs = marked_model.generate(**enc, max_new_tokens=200, do_sample=True, top_k=0, pad_token_id=tok.eos_token_id)
    marked = tok.batch_decode(outs, skip_special_tokens=True)
    psi_m = [s for s, _ in det.score_batch(marked)]
    assert statistics.mean(psi_m) > statistics.mean(psi) + 0.5
