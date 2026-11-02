"""`openmark` is the paper's method (co-optimized policy + FPL); the frozen-policy variant is `openmark-frozen`.
Keys, run directories and results recorded when `openmark` meant the frozen policy keep detecting as before."""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import openmarkx as omx  # noqa: E402
from openmarkx.registry import canonical_scheme, get_scheme, resolved_params, stored_scheme  # noqa: E402
from openmarkx.schemes.openmark import OpenMark, OpenMarkFrozen, OpenMarkFullB01  # noqa: E402

# what `embed --watermark openmark` recorded in a key before the change: the frozen variant's resolved parameters
OLD_PARAMS = {"gamma": 0.5, "delta": 1.0, "chunk_size": 10, "transform_loss_weight": 0.001, "tau_init": 10.0,
              "tau_max": 1000.0, "co_opt": False, "transform_lr": 1e-05, "transform_wd": 0.01, "centered_range": False}


def test_openmark_is_the_paper_method():
    s = get_scheme("openmark")
    assert type(s) is OpenMarkFullB01 and s.name == "openmark"
    assert s.co_opt and s.centered_range and s.w_aux == 100.0                 # CAWP: policy co-optimized
    assert s.booster == {"alpha": 0.1, "lamb": 0.1, "attack": "anti"}          # FPL, beta 0.1
    for old in ("openmark-full-b01", "pro-full-b01"):
        assert type(get_scheme(old)) is OpenMarkFullB01 and canonical_scheme(old) == "openmark"


def test_openmark_frozen_is_the_old_openmark():
    s = get_scheme("openmark-frozen")
    assert type(s) is OpenMark is OpenMarkFrozen and s.name == "openmark-frozen"
    assert not s.co_opt and getattr(s, "booster", None) is None
    assert resolved_params(s) == OLD_PARAMS
    assert canonical_scheme("pro") == "openmark-frozen"                        # pro always meant the frozen policy


def test_describe_shows_cooptimization_and_fpl():
    rows = {line.split()[0]: line for line in omx.describe(["openmark", "openmark-frozen"]).splitlines()[2:]}
    assert "co-optimized" in rows["openmark"] and "FPL step in training" in rows["openmark"]
    assert "co-optimized" not in rows["openmark-frozen"] and "attacks only" in rows["openmark-frozen"]


def test_stored_scheme():
    assert stored_scheme("openmark", OLD_PARAMS) == "openmark-frozen"
    assert stored_scheme("openmark", {}) == "openmark-frozen"                  # no params recorded: an old record
    assert stored_scheme("openmark", resolved_params(get_scheme("openmark"))) == "openmark"
    assert stored_scheme("pro", {}) == "pro" and stored_scheme("gaussmark", {}) == "gaussmark"


def test_a_key_made_with_the_old_openmark_detects_as_before(tmp_path, monkeypatch):
    import openmarkx.keys as K
    from openmarkx.api import _rng_device, build_scheme
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))
    m = tmp_path / "map.json"; m.write_text(json.dumps([0, 1, 2] * 10))
    old = omx.Key(scheme="openmark", params={"mapping": str(m), "gamma": 0.5, "delta": 1.0, "chunk_size": 10,
                                             "rng_device": "cuda"}, base="llama-3.2-1b", scheme_params=dict(OLD_PARAMS))
    K.store(old)
    k = omx.Key.open(K.key_path(old.commitment))
    assert k.scheme == "openmark" and k.commitment == old.commitment          # the record itself is never rewritten
    sch = build_scheme(k)
    assert type(sch) is OpenMark and sch.name == "openmark-frozen"
    assert resolved_params(sch) == OLD_PARAMS and getattr(sch, "booster", None) is None
    assert _rng_device(k) is None


def test_a_key_kept_inside_an_old_release_is_frozen(tmp_path, monkeypatch):
    import warnings
    import openmarkx.keys as K
    from openmarkx.api import build_scheme
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))
    rel = omx.Release(path=str(tmp_path / "r"), base="llama-3.2-1b", scheme="openmark", key={"mapping": "m.json"}).save()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = omx.load_key(rel)
    assert k.scheme_params == {} and build_scheme(k).name == "openmark-frozen"


def test_a_key_made_with_the_new_openmark_is_the_paper_method():
    from openmarkx.api import build_scheme
    k = omx.Key(scheme="openmark", params={"mapping": "blob:x.json", "transform": "blob:y.pt"},
                base="llama-3.2-1b", scheme_params=resolved_params(get_scheme("openmark")))
    sch = build_scheme(k)
    assert type(sch) is OpenMarkFullB01 and sch.co_opt
    old_full = omx.Key(scheme="openmark-full-b01", scheme_params=dict(k.scheme_params))
    assert type(build_scheme(old_full)) is OpenMarkFullB01                     # keys made under the old full name


def test_an_old_co_optimized_openmark_key_keeps_its_params():
    from openmarkx.api import build_scheme
    sp = {**OLD_PARAMS, "co_opt": True}                                       # embed --watermark openmark:co_opt=true
    sch = build_scheme(omx.Key(scheme="openmark", scheme_params=sp))
    assert resolved_params(sch) == sp


def test_old_run_configs_resume_as_openmark_frozen():
    from openmarkx.pipeline import NAMES, config_diff
    old = {"model": {"name": "agent"}, "scheme": {"name": "openmark", "params": {"delta": 2.0}}, "out": "/a"}
    frozen = {**old, "scheme": {"name": "openmark-frozen", "params": {"delta": 2.0}}}
    paper = {**old, "scheme": {"name": "openmark", "params": {"delta": 2.0}}}
    assert config_diff(old, frozen) == []                                      # a run from before: the frozen variant
    assert config_diff(old, paper) == ["scheme"]                               # never resumed as the paper method
    assert config_diff({**paper, "names": NAMES}, paper) == []                 # a run written now
    assert config_diff({**paper, "names": NAMES}, frozen) == ["scheme"]


def test_bench_keeps_result_directories(tmp_path, monkeypatch):
    from openmarkx import bench
    monkeypatch.setattr(bench, "RUNS", str(tmp_path))
    at = lambda n: str(tmp_path / "e2" / "llama-3.2-1b" / n)
    assert bench.result_dir("openmark-frozen", "llama-3.2-1b") == at("openmark")
    assert bench.result_dir("pro", "llama-3.2-1b") == at("openmark")
    assert bench.result_dir("openmark", "llama-3.2-1b") == at("openmark-full-b01")
    assert bench.result_dir("openmark-full-b01", "llama-3.2-1b") == at("openmark-full-b01")
    old = tmp_path / "e2" / "llama-3.2-1b" / "pro-full-b01"; old.mkdir(parents=True)
    assert bench.result_dir("openmark", "llama-3.2-1b") == str(old)
    arena = bench.load_arena(os.path.join(ROOT, "configs", "arena", "1b.yaml"))
    dirs = [os.path.basename(bench.result_dir(m if isinstance(m, str) else m["method"], arena["base"]))
            for m in arena["methods"]]
    assert {"openmark", "openmark-coopt-c100", "pro-full-b01"} <= set(dirs) and len(dirs) == len(set(dirs))


def test_port_without_a_policy_is_scored_as_frozen(tmp_path, monkeypatch):
    import openmarkx.keys as K
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))
    ck = tmp_path / "ckpt"; ck.mkdir(); (ck / "adapter_config.json").write_text("{}")
    m = tmp_path / "map.json"; m.write_text(json.dumps([0] * 10))
    with pytest.warns(UserWarning, match="transform"):
        rel, key = omx.port(str(ck), "openmark", {"mapping": str(m)}, "llama-3.2-1b")
    assert key.scheme == rel.scheme == "openmark-frozen" and key.scheme_params["co_opt"] is False


def test_keys_list_names_an_old_key_by_what_it_means(tmp_path):
    os.makedirs(tmp_path / "keys")
    k = omx.Key(scheme="openmark", params={"mapping": "blob:x.json"}, base="llama-3.2-1b",
                scheme_params=dict(OLD_PARAMS))
    k.save(str(tmp_path / "keys" / f"{k.commitment}.json"))
    env = dict(os.environ, PYTHONPATH=ROOT, CUDA_VISIBLE_DEVICES="", OMX_KEYS=str(tmp_path / "keys"))
    r = subprocess.run([sys.executable, "-m", "openmarkx.cli", "keys", "list"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "openmark-frozen" in r.stdout
    r = subprocess.run([sys.executable, "-m", "openmarkx.cli", "keys", "show", k.commitment[:12]], capture_output=True,
                       text=True, env=env)
    d = json.loads(r.stdout)
    assert d["scheme"] == "openmark-frozen" and d["recorded_as"] == "openmark" and d["commitment"] == k.commitment
