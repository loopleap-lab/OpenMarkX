import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

import markweightx as mwx
import markweightx.keys as K


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))
    return tmp_path


def test_commitment_is_salted_and_path_free(store):
    f1 = store / "a.json"; f1.write_text("[1,2,3]")
    sub = store / "sub"; sub.mkdir(); f2 = sub / "a.json"; f2.write_text("[1,2,3]")
    k1 = mwx.Key(scheme="pro", params={"mapping": str(f1)}, nonce="00")
    k2 = mwx.Key(scheme="pro", params={"mapping": str(f2)}, nonce="00")
    K.store(k1); K.store(k2)
    assert k1.params["mapping"] == k2.params["mapping"]          # same content -> same blob ref
    assert k1.commitment == k2.commitment                         # so the commitment does not depend on the path
    assert mwx.Key(scheme="pro", params=dict(k1.params)).commitment != k1.commitment   # fresh nonce -> new commitment


def test_edited_release_carries_no_key(store):
    from markweightx.edits.weights import _child
    k = mwx.Key(scheme="markllm-kgw-d", params={"secret": 123}); K.store(k)
    parent = mwx.Release(path=str(store / "p"), base="llama-3.2-1b", scheme="markllm-kgw-d", key_commitment=k.commitment)
    child = _child(parent, str(store / "c"), "prune", {"sparsity": 0.2})
    assert child.key == {} and child.key_commitment == k.commitment


@pytest.mark.slow
def test_secret_changes_the_watermark(store):
    """Two releases' keys must give different detectors: the same text scores differently."""
    from markweightx.models import ConfigOnly, load_tokenizer
    from markweightx.registry import get_scheme
    tok = load_tokenizer("llama-3.2-1b")
    text = "The committee met on Tuesday to review the budget for next year and agreed to postpone the final vote " * 3
    stats = []
    for secret in (11, 22):
        sch = get_scheme("markllm-kgw-d")
        key = sch.setup(ConfigOnly("llama-3.2-1b", "cpu"), tok, "cpu", {"secret": secret})
        assert key["secret"] == secret
        stats.append(sch.score(text)[0])
    assert stats[0] != stats[1]
    fresh = get_scheme("markllm-kgw-d").setup(ConfigOnly("llama-3.2-1b", "cpu"), tok, "cpu", None)
    assert "secret" in fresh and fresh["secret"] not in (11, 22)


def test_cuda_key_refuses_cpu_detection():
    k = mwx.Key(scheme="markllm-kgw-d", params={"secret": 5, "rng_device": "cuda"})
    with pytest.raises(ValueError, match="CUDA"):
        mwx.detect("some text", k, base="llama-3.2-1b", device="cpu")
