"""Regressions for a code review: each test reproduces one finding."""
import json
import os
import sys

TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TESTS)
sys.path.insert(0, os.path.dirname(TESTS))
import pytest
import torch
import yaml

from test_pipeline import geopt_cfg, run_cfg, write_cfg


@pytest.fixture
def store(tmp_path, monkeypatch):
    import openmarkx.keys as K
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))
    return tmp_path / "keys"


# I-1: key material (wmar's policy table) never sits in the release directory, not even mid-run
def test_train_imprint_keeps_resume_state_out_of_the_release(tmp_path, store):
    from openmarkx.registry import get_model, get_scheme
    from openmarkx.train import imprint
    m = get_model("taming", factory="fakes:FakeTaming")
    net = m.load(None, "cpu")
    sch = get_scheme("wmar")
    key = sch.setup(net, m.codec(), "cpu", {})
    seqs = torch.cat([torch.tensor([[1], [2]]), m.run(net, [1, 2])], 1)
    out = tmp_path / "rel"
    seen = []
    imprint(m, sch, [{"seqs": seqs}], str(out), net=net, key=key, device="cpu", save_every=1,
            phases=[{"steps": 2, "lr": 1e-3, "trainable": "full", "policy_lr": 0.1}],
            on_phase_end=lambda i: seen.append(sorted(os.listdir(out))))
    assert "resume.pt" not in seen[0] and "resume.pt" not in os.listdir(out)


# I-2: a resumed imprint keeps the key it started with
def test_resumed_imprint_keeps_its_secret(tmp_path, store, monkeypatch):
    import openmarkx.train as T
    from openmarkx.keys import load_key
    from openmarkx.schemes.ripple import RippleScheme
    cfg = geopt_cfg(tmp_path / "run")
    cfg["imprint"] = {"phases": [{"steps": 6, "lr": 1e-3, "trainable": "full"}], "save_every": 2}
    secrets, real_setup, real_cycle = [], RippleScheme.setup, T._cycle

    def setup(self, *a, **k):
        key = real_setup(self, *a, **k)
        secrets.append(key["secret"])
        return key

    def crashing(batches):
        for i, b in enumerate(real_cycle(batches)):
            if i == 5:
                raise KeyboardInterrupt("killed")
            yield b
    monkeypatch.setattr(RippleScheme, "setup", setup)
    monkeypatch.setattr(T, "_cycle", crashing)
    with pytest.raises(KeyboardInterrupt):
        run_cfg(cfg, tmp_path)
    monkeypatch.setattr(T, "_cycle", real_cycle)
    run_cfg(cfg, tmp_path)
    key = load_key(str(tmp_path / "run" / "release"))
    assert key.params["secret"] == secrets[0]


# I-3: every config key is checked before anything loads
@pytest.mark.parametrize("patch, word", [
    ({"model": {"name": "geopt", "factory": "fakes:FakeGeoPT", "ckpt_pth": "x"}}, "ckpt_pth"),
    ({"data": {"nonsense_key": 1}}, "nonsense_key"),
    ({"generate": {"temperature": 0.3}}, "temperature"),
    ({"scheme": {"name": "ripple", "params": {"epsilon": 0.1}}}, "epsilon"),
    ({"model": {"name": "taming", "factory": "fakes:FakeTaming"}, "scheme": {"name": "wmar"}, "data": {},
      "imprint": {"edit_last": True}, "edits": [], "quality": "none"}, "edit_last"),
])
def test_config_rejects_settings_the_run_would_ignore(tmp_path, patch, word):
    from openmarkx import pipeline
    with pytest.raises(pipeline.ConfigError, match=word):
        pipeline.load_config(write_cfg(tmp_path, {**geopt_cfg(tmp_path / "r"), **patch}))


# I-4: a step that runs again makes every later step run again, also across invocations
def test_rerun_step_invalidates_later_steps(tmp_path, store):
    run_cfg(geopt_cfg(tmp_path / "run"), tmp_path)
    os.remove(tmp_path / "run" / "imprint.json")
    run_cfg(geopt_cfg(tmp_path / "run"), tmp_path, only=["imprint"])
    for later in ("generate", "detect", "edits", "quality", "report"):
        assert not (tmp_path / "run" / f"{later}.json").exists(), later


# I-5: no null kind silently computes something else
def test_derivative_null_is_refused(tmp_path):
    from openmarkx import pipeline
    with pytest.raises(pipeline.ConfigError, match="derivative"):
        pipeline.load_config(write_cfg(tmp_path, {**geopt_cfg(tmp_path / "r"), "detect": {"null": "derivative"}}))


def test_wmar_wrong_key_is_another_table(tmp_path, store):
    import random
    from openmarkx.keys import stage_dir
    from openmarkx.registry import get_model, get_scheme
    m = get_model("taming", factory="fakes:FakeTaming")
    m.load(None, "cpu")
    s = get_scheme("wmar")
    key = s.save(stage_dir(), s.setup(None, m.codec(), "cpu", {}))
    wrong = s.wrong_key(dict(key), random.Random(0), str(tmp_path))
    w = get_scheme("wmar")
    w.setup(None, m.codec(), "cpu", wrong)
    assert wrong is not None and not torch.equal(w.table(), s.table())


# I-6: after edit_last, gradient phases train towards the unedited model + mark, not a double mark
def test_imprint_uses_the_given_teacher(tmp_path):
    from fakes import tiny_field_net
    from openmarkx.train import imprint
    from test_train import Shift, TinyField
    net, teacher = tiny_field_net(), tiny_field_net()
    with torch.no_grad():
        teacher[-1].bias += 1.0                                # a teacher that differs from the student's start
    x = torch.randn(64, 3)
    rel = imprint(TinyField(), Shift(), [{"inputs": x}], str(tmp_path), net=net, teacher=teacher, device="cpu",
                  phases=[{"steps": 300, "lr": 3e-2, "trainable": "full"}])
    with torch.no_grad():
        err = (rel.net(x) - (teacher(x) + 0.1)).abs().mean().item()
    assert err < 0.2


# I-7: code a release names runs only under trust
def test_release_named_factory_needs_trust(tmp_path):
    from openmarkx.models import load_release
    from openmarkx.registry import UntrustedPlugin
    from openmarkx.types import Release
    rel = Release(str(tmp_path), "geopt", "ripple", model={"name": "geopt", "params": {"factory": "runpy:run_path"}},
                  format="state_dict")
    with pytest.raises(UntrustedPlugin):
        rel.adapter_model()
    with pytest.raises(UntrustedPlugin):
        load_release(rel, "cpu")
    assert rel.adapter_model(trust=True).name == "geopt"


# I-8: co-learning clips the model's gradients only, as pro_taming.py's cmd_colearn
def test_colearn_clips_model_weights_only(tmp_path, store, monkeypatch):
    from openmarkx.registry import get_model, get_scheme
    from openmarkx.train import imprint
    m = get_model("taming", factory="fakes:FakeTaming")
    net = m.load(None, "cpu")
    sch = get_scheme("wmar")
    key = sch.setup(net, m.codec(), "cpu", {})
    clipped, real = [], torch.nn.utils.clip_grad_norm_
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda ps, n: (clipped.append({id(p) for p in ps}), real(ps, n))[1])
    seqs = torch.cat([torch.tensor([[1]]), m.run(net, [1])], 1)
    imprint(m, sch, [{"seqs": seqs}], str(tmp_path / "rel"), net=net, key=key, device="cpu",
            phases=[{"steps": 1, "lr": 1e-3, "trainable": "full", "policy_lr": 0.1}])
    assert clipped and id(sch.Z) not in clipped[0]


# I-9: eps: auto is refused for ripple until calibration models the learning noise
def test_ripple_eps_auto_is_refused(tmp_path):
    from openmarkx import pipeline
    cfg = {**geopt_cfg(tmp_path / "r"), "scheme": {"name": "ripple", "params": {"eps": "auto"}}}
    with pytest.raises(pipeline.ConfigError, match="auto"):
        pipeline.load_config(write_cfg(tmp_path, cfg))
