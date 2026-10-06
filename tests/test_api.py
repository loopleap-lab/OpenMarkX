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
