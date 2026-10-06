"""The contract with MarkAgentX (appendix: one Detection and one wrong-key null across the family), checked against
its source without importing it (importing markagentx pulls in its trajectory modules and dependencies)."""
import ast
import os

import pytest

import markweightx as mwx

MAX = os.path.join(os.path.dirname(__file__), "..", "..", "refs", "MarkAgentX", "markagentx", "base.py")


def _markagentx_detection_fields():
    tree = ast.parse(open(MAX).read())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Detection")
    return [s.target.id for s in cls.body if isinstance(s, ast.AnnAssign)]


@pytest.mark.skipif(not os.path.exists(MAX), reason="no MarkAgentX checkout next to this repo")
def test_detection_maps_onto_markagentx():
    fields = _markagentx_detection_fields()
    d = mwx.Detection(statistic=2.5, p_value=0.01, n_tokens=120, flagged=True)
    assert list(d.to_markagentx()) == fields == ["statistic", "p_value", "n_evidence"]
    assert d.n_evidence == 120


def test_to_markagentx_builds_its_detection():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(MAX)))
    base = pytest.importorskip("markagentx.base")
    d = mwx.Detection(statistic=2.5, p_value=None, n_tokens=0)
    m = base.Detection(**d.to_markagentx())
    assert m.p_value == 1.0 and m.n_evidence == 0          # MarkAgentX: always a number; no evidence is p = 1


def test_empty_text_is_a_vacuous_answer(tmp_path, monkeypatch):
    import markweightx.keys as K
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))
    k = mwx.Key(scheme="markllm-unigram-d", params={"rng_device": "cpu"}, base="llama-3.2-1b")
    d = mwx.detect([""], k, device="cpu")[0]
    assert d.n_evidence == 0 and d.p_value == 1.0 and d.flagged is False


def test_wrong_key_null_is_reproducible_and_valid(tmp_path, monkeypatch):
    import markweightx.keys as K
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))
    k = mwx.Key(scheme="markllm-unigram-d", params={"secret": 123456789, "rng_device": "cpu"}, base="llama-3.2-1b")
    texts = ["The committee met on Tuesday to review the budget and agreed to postpone the vote until the auditors "
             "deliver their report on the spending of the last two quarters."]
    texts += ["Rain is expected across the northern valleys tonight, with heavier showers near the coast.",
              "She opened the old notebook and found her grandmother's recipe for plum cake.", ""]
    a = mwx.detect(texts, k, device="cpu", null="wrong_key", n_null=20)
    assert a == mwx.detect(texts, k, device="cpu", null="wrong_key", n_null=20)   # reproducible
    assert all(1 / 21 <= d.p_value <= 1.0 for d in a)
    assert len({d.p_value for d in a[:3]}) > 1             # a real null, not every wrong key equal to the true one
    assert a[3].n_evidence == 0 and a[3].p_value == 1.0      # the empty text
    assert len(os.listdir(tmp_path / "keys" / "configs")) == 1   # only the true key's config: wrong keys cleaned up


def test_wrong_key_needs_a_secret(tmp_path, monkeypatch):
    import markweightx.keys as K
    from markweightx.schemes.base import Scheme
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))
    assert Scheme().wrong_key({"rng_device": "cpu"}, __import__("random").Random(0)) is None
