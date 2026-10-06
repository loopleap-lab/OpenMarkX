"""Bob's path: one file defines a scheme, registers it, and a detector that only has the key finds it again."""
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

import markweightx as mwx
from markweightx.primitives import AccessDenied

BOB = textwrap.dedent('''
    import markweightx as mwx

    @mwx.register_scheme
    class BobMark(mwx.Scheme):
        """Toy scheme: the statistic is the share of words with an even length (no model needed)."""
        name = "bob-toy"
        access = mwx.Access.R0

        def __init__(self, bias=0.0, **params):
            super().__init__(bias=bias, **params)
            self.bias = bias

        def score(self, text):
            w = text.split()
            return sum(len(x) % 2 == 0 for x in w) / max(len(w), 1) + self.bias, len(w)

        def pvalue(self, statistic, n_tokens):
            return 0.001 if statistic > 0.9 else 0.5
''')


def test_plugin_scheme_round_trip(tmp_path, monkeypatch):
    import markweightx.keys as K
    from markweightx.registry import SCHEMES, get_scheme, scheme_source
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))
    f = tmp_path / "bob_mark.py"; f.write_text(BOB)
    sch = get_scheme("bob-toy", source=str(f), bias=0.25)
    key = mwx.Key(scheme="bob-toy", params={}, base="llama-3.2-1b", scheme_params={"bias": 0.25},
                  scheme_source=scheme_source(type(sch)))
    K.store(key)
    SCHEMES.pop("bob-toy")                                  # a fresh detector process has never imported Bob's file
    d = mwx.detect(["even even even even", "odd odd odd"], mwx.Key.open(K.key_path(key.commitment)), device="cpu")
    assert d[0].flagged and d[0].p_value == 0.001 and not d[1].flagged
    assert d[0].statistic == pytest.approx(1.25)            # the scheme param travelled with the key


def test_grant_below_need_is_refused():
    k = mwx.Key(scheme="markllm-sweet-d", params={"secret": 1, "rng_device": "cpu"}, base="llama-3.2-1b")
    with pytest.raises(AccessDenied):
        mwx.detect("text", k, device="cpu", grant=mwx.Access.R0)


def test_describe_hides_variants():
    t = mwx.describe()
    assert "pro-coopt-c100" in t and "pro-booster" not in t and "expgumbel" not in t


def test_merge_needs_a_partner():
    with pytest.raises(ValueError, match="partner"):
        mwx.modify(mwx.Release(path="x", base="b", scheme="s"), "merge:ratio=0.4")


def test_r1_probe_gets_no_gradients():
    from markweightx.primitives import Probe
    with pytest.raises(AccessDenied):
        Probe("llama-3.2-1b", "cpu", mwx.Access.R1).grad_seq_logprob("text", "model.layers.0.mlp.up_proj.weight")


def test_same_plugin_twice_in_one_process(tmp_path):
    from markweightx.api import scheme_spec
    f = tmp_path / "twice_mark.py"
    f.write_text(open(os.path.join(os.path.dirname(__file__), "..", "examples", "my_scheme.py")).read()
                 .replace('"my-scheme"', '"twice-scheme"'))
    assert scheme_spec(str(f))[0] == scheme_spec(str(f))[0] == "twice-scheme"
