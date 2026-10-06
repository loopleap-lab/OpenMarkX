import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
