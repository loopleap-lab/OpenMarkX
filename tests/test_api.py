import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import markweightx as mwx
from markweightx.api import _spec


def test_public_api_surface():
    for name in ["embed", "calibrate", "modify", "generate", "rewrite", "detect", "verify_model", "register_scheme"]:
        assert callable(getattr(mwx, name))


def test_spec_parsing():
    assert _spec("finetune:steps=800,lr=1e-4") == ("finetune", {"steps": 800, "lr": 1e-4})
    assert _spec("quantize") == ("quantize", {})
    assert _spec({"op": "merge", "ratio": 0.4}) == ("merge", {"ratio": 0.4})


def test_key_is_not_in_the_release(tmp_path, monkeypatch):
    import markweightx.keys as K
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))
    key = mwx.Key(scheme="markllm-kgw-d", params={"key": 42})
    K.store(key)
    rel = mwx.Release(path=str(tmp_path / "rel"), base="llama-3.2-1b", scheme="markllm-kgw-d", key_commitment=key.commitment).save()
    assert '"key": 42' not in open(os.path.join(rel.path, "release.json")).read()
    assert mwx.load_key(rel.path).params == {"key": 42}
    # a different key has a different commitment
    assert mwx.Key(scheme="markllm-kgw-d", params={"key": 43}).commitment != key.commitment


def test_legacy_release_key_still_loads(tmp_path):
    rel = mwx.Release(path=str(tmp_path / "old"), base="llama-3.2-1b", scheme="markllm-kgw-d", key={"key": 7}).save()
    assert mwx.load_key(rel.path).params == {"key": 7}


def test_spec_values_and_errors():
    import pytest
    assert _spec("finetune:steps=8e2") == ("finetune", {"steps": 800})
    assert _spec("x:flag=true,lr=0.5") == ("x", {"flag": True, "lr": 0.5})
    with pytest.raises(ValueError, match="key=value"):
        _spec("finetune:steps")


def test_unknown_edit_lists_the_options():
    import pytest
    from markweightx.edits import modify
    with pytest.raises(ValueError, match="available: .*finetune"):
        modify(mwx.Release(path="x", base="b", scheme="s"), "shrink", "out")


def test_describe_shows_primitives_and_access():
    t = mwx.describe(["pro", "markllm-kgw-d", "markllm-sweet-d"])
    assert "Imprint" in t and "R0_AUX" in t and "R1" in t


def test_port_wraps_a_checkpoint_and_its_key(tmp_path, monkeypatch):
    import json
    import markweightx.keys as K
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))
    ck = tmp_path / "their_ckpt"; ck.mkdir()
    (ck / "adapter_config.json").write_text("{}")
    rel, key = mwx.port(str(ck), "markllm-unigram-d", {"secret": 424242, "rng_device": "cpu"}, "llama-3.2-1b")
    assert rel.adapter == str(ck) and rel.lineage[0]["op"] == "port"
    assert mwx.Release.open(rel.path).key_commitment == key.commitment
    made = mwx.Key(scheme="markllm-unigram-d", params={"secret": 424242, "rng_device": "cpu"}, base="llama-3.2-1b")
    text = ["The committee met on Tuesday to review the budget and agreed to postpone the vote."]
    assert mwx.detect(text, key, device="cpu")[0].statistic == mwx.detect(text, made, device="cpu")[0].statistic
    with pytest.raises(ValueError, match="neither a LoRA adapter"):
        mwx.port(str(tmp_path), "markllm-unigram-d", {"secret": 1}, "llama-3.2-1b")
    with pytest.raises(FileNotFoundError, match="key field mapping"):
        mwx.port(str(ck), "pro", {"mapping": str(tmp_path / "missing.json")}, "llama-3.2-1b")
    _, pub = mwx.port(str(ck), "markllm-unigram-d", {"secret": 7, "rng_device": "cpu"}, "llama-3.2-1b",
                      out=str(tmp_path / "pub"), key_public=True)
    with pytest.warns(UserWarning, match="public"):
        mwx.detect(text, pub, device="cpu")


def test_verify_model_loop_counts_generated_texts_and_skips_unjudged(tmp_path, monkeypatch):
    """The real loop of verify_model, with the model and the detector replaced: n_queries counts generated texts,
    unjudged texts are left out, the sequential path accuses early and the fixed path doubles its p."""
    import markweightx.api as api
    import markweightx.models as models
    import markweightx.sampling as sampling
    import markweightx.scoring as scoring
    from markweightx.types import Detection
    monkeypatch.setattr(models, "load_release", lambda rel, device: None)
    monkeypatch.setattr(models, "load_tokenizer", lambda base: None)
    monkeypatch.setattr(api, "scheme_for", lambda *a, **k: None)
    monkeypatch.setattr(sampling, "generate", lambda m, t, prompts, seed=0: list(prompts))
    # every text "x" is flagged, "o" is not, "s" is too short to judge
    monkeypatch.setattr(scoring, "detect", lambda b, texts, **k: [
        Detection(1.0, None if t == "s" else 0.0, 5, None if t == "s" else t == "x") for t in texts])
    key = mwx.Key(scheme="gaussmark", params={"secret": 1}, base="llama-3.2-1b")
    rel = mwx.Release(path=str(tmp_path), base="llama-3.2-1b", scheme="gaussmark")
    d = mwx.verify_model(rel, key, ["x", "s"] * 20, device="cpu", sequential=True)
    assert d.flagged and d.n_queries == 20 and d.statistic == 10.0          # accused after the first batch of 20
    d = mwx.verify_model(rel, key, ["o", "s"] * 30, device="cpu", sequential=True, budget=40)
    assert not d.flagged and d.n_queries == 40 and d.p_value == 1.0         # never cleared early: ran the budget
    with pytest.raises(ValueError, match="budget"):
        mwx.verify_model(rel, key, ["o"], device="cpu", budget=0)


def test_pro_wrong_key_redraws_only_the_mapping(tmp_path):
    import json
    import random
    from markweightx.registry import get_scheme
    m = tmp_path / "map.json"; m.write_text(json.dumps([1, 2, 3] * 10))
    params = {"mapping": str(m), "transform": "blob:abc.pt", "gamma": 0.5}
    w = get_scheme("pro").wrong_key(params, random.Random(0), str(tmp_path))
    assert w["transform"] == "blob:abc.pt" and w["gamma"] == 0.5 and w["mapping"] != str(m)
    assert len(json.load(open(w["mapping"]))) == 30 and w["mapping"].startswith(str(tmp_path))


def test_port_copies_key_files_and_refuses_to_overwrite(tmp_path, monkeypatch):
    import json
    import subprocess
    import sys
    import markweightx.keys as K
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))
    ck = tmp_path / "ckpt"; ck.mkdir(); (ck / "adapter_config.json").write_text("{}")
    m = tmp_path / "map.json"; m.write_text(json.dumps([0] * 10))
    _, key = mwx.port(str(ck), "pro", {"mapping": str(m)}, "llama-3.2-1b")
    assert key.params["mapping"].startswith("blob:")                         # copied into the key store
    with pytest.raises(FileExistsError):
        mwx.port(str(ck), "pro", {"mapping": str(m)}, "llama-3.2-1b")
    with pytest.raises(ValueError, match="no field bogus"):
        mwx.port(str(ck), "markllm-kgw-d", {"secret": 1, "bogus": 2}, "llama-3.2-1b", force=True)
    (tmp_path / "k.json").write_text(json.dumps({"mapping": "map.json"}))  # relative to the key file
    r = subprocess.run([sys.executable, "-m", "markweightx.cli", "port", str(ck), "--watermark", "pro",
                        "--key-file", str(tmp_path / "k.json"), "--base", "llama-3.2-1b", "--force"],
                       capture_output=True, text=True, cwd="/", env=dict(__import__("os").environ,
                       MWX_KEYS=str(tmp_path / "keys"), CUDA_VISIBLE_DEVICES="",
                       PYTHONPATH=__import__("os").path.join(__import__("os").path.dirname(__file__), "..")))
    assert r.returncode == 0, r.stderr
