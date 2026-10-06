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
    from markweightx.registry import LOADED
    LOADED.clear()
    k2 = mwx.Key.open(K.key_path(key.commitment))
    from markweightx.registry import UntrustedPlugin
    with pytest.raises(UntrustedPlugin):                    # a key can name any code: not run without consent
        mwx.detect(["even"], k2, device="cpu")
    d = mwx.detect(["even even even even", "odd odd odd"], k2, device="cpu", trust_plugin=True)
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


def _plugin(path, version):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(open(os.path.join(os.path.dirname(__file__), "..", "examples", "my_scheme.py")).read()
                    .replace('"my-scheme"', '"versioned"').replace("    name = ", f"    version = {version}\n    name = ", 1))
    return str(path)


def test_each_key_scores_with_its_own_plugin_version(tmp_path):
    from markweightx.registry import get_scheme
    a = _plugin(tmp_path / "a" / "pl.py", 1)
    b = _plugin(tmp_path / "b" / "pl.py", 2)              # same file name, same scheme name, edited code
    assert get_scheme("versioned", source=a).version == 1
    assert get_scheme("versioned", source=b).version == 2
    assert get_scheme("versioned", source=a).version == 1   # loading b did not replace a


def test_mistyped_scheme_param_is_refused():
    from markweightx.api import scheme_spec
    with pytest.raises(ValueError, match="sigmaa"):
        scheme_spec("gaussmark:sigmaa=0.1")
    assert scheme_spec("gaussmark:sigma=0.02")[1] == {"sigma": 0.02}


def test_cli_errors_are_one_line(tmp_path):
    import subprocess
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(__file__), ".."), CUDA_VISIBLE_DEVICES="",
               MWX_KEYS=str(tmp_path / "keys"))
    for args in (["describe", "nosuch"], ["embed", "--base", "llama-3.2-1b", "--watermark", "gausmark"],
                 ["check", "--adapter", str(tmp_path / "missing.py")], ["new-scheme", "2 bad name"]):
        r = subprocess.run([sys.executable, "-m", "markweightx.cli", *args], capture_output=True, text=True, env=env)
        assert r.returncode != 0 and "Traceback" not in r.stderr and r.stderr.strip(), (args, r.stderr)


def test_module_plugin_is_pinned_to_its_code(tmp_path, monkeypatch):
    from markweightx.api import scheme_spec
    from markweightx.registry import load_plugin
    pkg = tmp_path / "labpkg"; pkg.mkdir(); (pkg / "__init__.py").write_text("")
    _plugin(pkg / "mark.py", 1)
    monkeypatch.syspath_prepend(str(tmp_path))
    name, _, src = scheme_spec("labpkg.mark:versioned")
    assert name == "versioned" and src.startswith("labpkg.mark@")
    (pkg / "mark.py").write_text((pkg / "mark.py").read_text() + "\n# edited\n")
    with pytest.raises(ValueError, match="changed since this key was made"):
        load_plugin(src)


def test_missing_plugin_source_says_so(tmp_path):
    from markweightx.registry import load_plugin
    with pytest.raises(FileNotFoundError, match="not found"):
        load_plugin(str(tmp_path / "gone.py"))


def test_owner_detects_with_own_plugin_in_same_process(tmp_path, monkeypatch):
    import markweightx.keys as K
    from markweightx.api import scheme_spec
    from markweightx.registry import scheme_source
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))
    f = tmp_path / "bob_mark.py"; f.write_text(BOB)
    name, params, src = scheme_spec(str(f))               # the owner imports the plugin (as embed does)
    key = mwx.Key(scheme=name, params={}, base="llama-3.2-1b", scheme_params={"bias": 0.25}, scheme_source=src)
    K.store(key)                                          # the key now names a blob copy of the file
    k2 = mwx.Key.open(K.key_path(key.commitment))
    assert k2.scheme_source.startswith("blob:")
    mwx.detect(["even"], k2, device="cpu")                # same code already loaded: no trust flag needed


def test_edit_params_are_checked_before_any_work():
    rel = mwx.Release(path="x", base="llama-3.2-1b", scheme="s")
    for spec, msg in (("quantize:bits=3", "4 \\(nf4\\) or 8"), ("prune:method=wanda", "magnitude"),
                      ("prune:sparsity=1.5", "in \\[0, 1\\)"), ("finetune:step=10", "has no parameter step")):
        with pytest.raises(ValueError, match=msg):
            mwx.modify(rel, spec)
    with pytest.raises(ValueError, match="has no parameter ratoi"):
        mwx.rewrite(["a b c"], "synonym:ratoi=0.3")


def test_save_never_sees_the_release(tmp_path, monkeypatch):
    import markweightx.keys as K
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))
    d = K.stage_dir()
    assert d.startswith(str(tmp_path / "keys")) and os.path.isdir(d)
