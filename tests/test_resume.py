"""Every long step continues after a kill (HPG time limits): a resumed run ends where an uninterrupted one would."""
import os
import sys

TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TESTS)
sys.path.insert(0, os.path.dirname(TESTS))
import pytest
import torch

from fakes import tiny_field_net
from test_train import Shift, TinyField


@pytest.fixture
def store(tmp_path, monkeypatch):
    import openmarkx.keys as K
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))


def killed_at(monkeypatch, n):
    """Make train._cycle raise after n batches (a job killed by its time limit)."""
    import openmarkx.train as T
    real = T._cycle

    def cycle(batches):
        for i, b in enumerate(real(batches)):
            if i == n:
                raise KeyboardInterrupt("time limit")
            yield b
    monkeypatch.setattr(T, "_cycle", cycle)
    return real


def test_train_imprint_resume_matches_an_uninterrupted_run(tmp_path, store, monkeypatch):
    import openmarkx.train as T
    g = torch.Generator().manual_seed(0)
    batches = [{"inputs": torch.randn(16, 3, generator=g)} for _ in range(5)]
    ph = [{"steps": 12, "lr": 1e-2, "trainable": "full"}]
    whole = T.imprint(TinyField(), Shift(), batches, str(tmp_path / "a"), net=tiny_field_net(), device="cpu",
                      phases=ph, save_every=4)
    real = killed_at(monkeypatch, 9)                     # saved at step 8, killed during step 10
    with pytest.raises(KeyboardInterrupt):
        T.imprint(TinyField(), Shift(), batches, str(tmp_path / "b"), net=tiny_field_net(), device="cpu",
                  phases=ph, save_every=4)
    monkeypatch.setattr(T, "_cycle", real)
    resumed = T.imprint(TinyField(), Shift(), batches, str(tmp_path / "b"), net=tiny_field_net(), device="cpu",
                        phases=ph, save_every=4)
    for (k, a), b in zip(whole.net.state_dict().items(), resumed.net.state_dict().values()):
        assert torch.allclose(a, b, atol=1e-6), k


def test_pipeline_saves_every_100_steps_by_default(tmp_path):
    from openmarkx import pipeline
    from test_pipeline import geopt_cfg, write_cfg
    cfg = pipeline.load_config(write_cfg(tmp_path, geopt_cfg(tmp_path / "r")))
    assert pipeline.save_every(cfg) == 100


@pytest.fixture(scope="module")
def text_base(tmp_path_factory):
    from fakes import tiny_llama
    from openmarkx.models import load_tokenizer
    d = tmp_path_factory.mktemp("base")
    tiny_llama().save_pretrained(d)
    load_tokenizer("llama-3.2-1b").save_pretrained(d)
    corpus = d / "corpus.pt"
    torch.save({"train": torch.randint(1000, 20000, (32, 17), generator=torch.Generator().manual_seed(0))}, corpus)
    return str(d), str(corpus)


KEYED_TOY = """
import math, os
import torch
import openmarkx as omx


@omx.register_scheme
class KeyedToy(omx.Scheme):
    name = "keyed_toy"
    exact_null = True

    def setup(self, model, tok, device, key=None):
        self.model, self.tok, self.device = model, tok, device
        key = dict(key or {})
        key.setdefault("secret", int.from_bytes(os.urandom(4), "big"))
        self.parity = key["secret"] % 2
        return key

    def target_logits(self, input_ids, logits):
        add = torch.zeros(logits.shape[-1], device=logits.device)
        add[self.parity::2] = 4.0
        return logits.float() + add

    def score(self, text):
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        g = sum(1 for i in ids if i % 2 == self.parity)
        return ((g - 0.5 * len(ids)) / math.sqrt(max(len(ids), 1) * 0.25), len(ids))
"""


def test_embed_resume_keeps_the_key(tmp_path, store, monkeypatch, text_base):
    import json
    import openmarkx.imprint as I
    from openmarkx.keys import load_key
    base, corpus = text_base
    plugin = tmp_path / "keyed_toy.py"
    plugin.write_text(KEYED_TOY)
    kw = dict(steps=6, bs=2, seq_len=16, lr=1e-3, rank=4, device="cpu", data=corpus, save_every=2, seed=0,
              scheme_source=str(plugin))
    real_step, calls = I.kl_teacher_student, {"n": 0}

    def dying(*a, **k):
        calls["n"] += 1
        if calls["n"] == 4:                                   # dies in step 4; the last save was after step 2
            raise KeyboardInterrupt("time limit")
        return real_step(*a, **k)
    monkeypatch.setattr(I, "kl_teacher_student", dying)
    with pytest.raises(KeyboardInterrupt):
        I.embed("keyed_toy", base, str(tmp_path / "b"), **kw)
    from openmarkx.train import draft_key
    first = draft_key(str(tmp_path / "b"))
    monkeypatch.setattr(I, "kl_teacher_student", real_step)
    rel = I.embed("keyed_toy", base, str(tmp_path / "b"), **kw)
    assert load_key(rel).params["secret"] == first["secret"]
    steps = [json.loads(l)["step"] for l in (tmp_path / "b" / "train_log.jsonl").read_text().splitlines()]
    assert steps == [1, 2, 3, 3, 4, 5, 6]                     # 1-3 before the kill (3 unsaved), resumed from step 2


def test_chunked_step_redoes_only_unfinished_chunks(tmp_path):
    from openmarkx.pipeline import in_chunks
    done = []

    def fn(chunk, start):
        if chunk == [4, 5] and not done:
            done.append("died")
            raise KeyboardInterrupt
        return [x * 10 for x in chunk]
    with pytest.raises(KeyboardInterrupt):
        in_chunks(str(tmp_path / "gen.partial.pt"), list(range(7)), 2, fn)
    seen = []
    out = in_chunks(str(tmp_path / "gen.partial.pt"), list(range(7)), 2, lambda c, start: (seen.append(c), [x * 10 for x in c])[1])
    assert out == [x * 10 for x in range(7)] and seen == [[4, 5], [6]]
    assert not os.path.exists(tmp_path / "gen.partial.pt")
