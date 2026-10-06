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


@pytest.mark.skipif(not os.path.exists(MAX), reason="no MarkAgentX checkout next to this repo")
def test_wrong_key_seed_matches_markagentx():
    src = open(MAX).read()
    assert "random.Random(0xC0FFEE)" in src                 # the seed our seeded_secrets() default mirrors
    import inspect
    from markweightx.keys import seeded_secrets
    assert inspect.signature(seeded_secrets.__wrapped__).parameters["seed"].default == 0xC0FFEE


def test_empty_text_is_a_vacuous_answer(tmp_path, monkeypatch):
    import markweightx.keys as K
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))
    k = mwx.Key(scheme="markllm-unigram-d", params={"rng_device": "cpu"}, base="llama-3.2-1b")
    d = mwx.detect([""], k, device="cpu")[0]
    assert d.n_evidence == 0 and d.p_value == 1.0 and d.flagged is False


def test_wrong_key_null_is_reproducible_and_valid(tmp_path, monkeypatch):
    import markweightx.keys as K
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))
    k = mwx.Key(scheme="markllm-unigram-d", params={"rng_device": "cpu"}, base="llama-3.2-1b")
    texts = ["The committee met on Tuesday to review the budget and agreed to postpone the vote until the auditors "
             "deliver their report on the spending of the last two quarters."]
    a = mwx.detect(texts, k, device="cpu", null="wrong_key", n_null=20)[0]
    b = mwx.detect(texts, k, device="cpu", null="wrong_key", n_null=20)[0]
    assert a == b and 1 / 21 <= a.p_value <= 1.0
