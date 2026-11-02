"""Names from before the rename keep working: the markweightx package, the scheme name "pro" (and its variants),
MWX_* environment variables, ~/.markweightx, and keys, releases and run directories written with them."""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def py(code: str, **env) -> subprocess.CompletedProcess:
    """Run code in a fresh interpreter (imports and environment variables are read once per process)."""
    e = {k: v for k, v in os.environ.items() if not k.startswith(("OMX_", "MWX_"))}
    e.update(PYTHONPATH=ROOT, CUDA_VISIBLE_DEVICES="", **env)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=e, cwd=ROOT)


# ---- the markweightx package -------------------------------------------------------------------------------------

def test_import_markweightx_is_openmarkx_and_warns_once():
    r = py("import warnings\n"
           "with warnings.catch_warnings(record=True) as w:\n"
           "    warnings.simplefilter('always')\n"
           "    import markweightx, openmarkx\n"
           "    import markweightx\n"
           "    from markweightx.api import embed\n"
           "assert markweightx is openmarkx, markweightx\n"
           "assert embed is openmarkx.embed\n"
           "dep = [x for x in w if issubclass(x.category, DeprecationWarning) and 'openmarkx' in str(x.message)]\n"
           "print(len(dep))")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["1"]


def test_markweightx_submodules_are_the_openmarkx_modules():
    r = py("import warnings; warnings.simplefilter('ignore')\n"
           "import markweightx.schemes.pro as old, openmarkx.schemes.openmark as new\n"
           "assert old is new and old.PRO is new.OpenMark and old.PROCooptCentered100 is new.OpenMarkCooptCentered100\n"
           "from markweightx.schemes.pro import PRO\n"
           "from markweightx.types import Key, Release\n"
           "import openmarkx.types, openmarkx.registry, markweightx.registry\n"
           "assert Key is openmarkx.types.Key and markweightx.registry is openmarkx.registry\n"
           "assert new.__spec__.name == 'openmarkx.schemes.openmark'\n"
           "import sys; assert not [m for m in sys.modules if m.startswith('openmarkx.') and m.endswith('.pro')]\n"
           "print('ok')")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"


def test_python_m_markweightx_cli_still_runs():
    r = subprocess.run([sys.executable, "-W", "ignore", "-m", "markweightx.cli", "--help"], capture_output=True,
                       text=True, env=dict(os.environ, PYTHONPATH=ROOT), cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "usage: openmarkx" in r.stdout


def test_old_pickle_loads(tmp_path):
    import pickle
    from openmarkx.types import Detection
    old = pickle.dumps(Detection(2.0, 0.01, 100, True), protocol=0).replace(b"openmarkx.types", b"markweightx.types")
    assert b"markweightx.types" in old
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        d = pickle.loads(old)
    assert d == Detection(2.0, 0.01, 100, True)


# ---- the scheme name -----------------------------------------------------------------------------------------------

def test_pro_and_its_variants_are_openmark():
    from openmarkx.registry import SCHEMES, canonical_scheme, get_scheme
    from openmarkx.schemes.openmark import OpenMark, OpenMarkCooptCentered100, OpenMarkFullB01
    assert canonical_scheme("pro") == "openmark-frozen" and canonical_scheme("gaussmark") == "gaussmark"
    assert type(get_scheme("pro")) is OpenMark and get_scheme("pro").name == "openmark-frozen"   # pro was frozen
    assert type(get_scheme("pro-coopt-c100")) is OpenMarkCooptCentered100
    assert type(get_scheme("pro-full-b01", delta=2.0)) is OpenMarkFullB01
    assert "pro" in SCHEMES and SCHEMES.get("pro") is OpenMark and SCHEMES["pro-full-b01"] is OpenMarkFullB01
    assert "openmark" in SCHEMES and "pro" not in list(SCHEMES) and "pro" not in dict(SCHEMES.items())


def test_scheme_spec_and_cli_accept_pro():
    from openmarkx.api import _get, scheme_spec
    name, params, src = scheme_spec("pro:delta=2.0")
    assert type(_get(name, src, params)).__name__ == "OpenMark"


def test_a_plugin_cannot_take_the_old_name():
    from openmarkx.registry import register_scheme
    from openmarkx.schemes.base import Scheme

    class Squatter(Scheme):
        name = "pro"
    with pytest.raises(ValueError, match="already registered"):
        register_scheme(Squatter)


# ---- keys, releases and runs written before the rename -------------------------------------------------------------

OLD_KEY = {"scheme": "pro", "params": {"mapping": "blob:" + "a" * 64 + ".json", "gamma": 0.5, "delta": 1.0,
                                       "chunk_size": 10, "rng_device": "cuda"},
           "calibration": {}, "nonce": "0" * 32, "base": "llama-3.2-1b",
           "scheme_params": {"gamma": 0.5, "delta": 1.0, "chunk_size": 10}, "scheme_source": None}


def test_an_old_key_and_release_load_and_build(tmp_path, monkeypatch):
    import openmarkx as omx
    from openmarkx import keys
    from openmarkx.api import _rng_device, build_scheme
    from openmarkx.types import Key
    commitment = Key(**OLD_KEY).commitment
    monkeypatch.setattr(keys, "STORE", str(tmp_path / "keys"))
    os.makedirs(tmp_path / "keys")
    json.dump(OLD_KEY, open(tmp_path / "keys" / f"{commitment}.json", "w"))
    rel = tmp_path / "rel"
    rel.mkdir()
    json.dump({"path": str(rel), "base": "llama-3.2-1b", "scheme": "pro", "key": {}, "key_commitment": commitment,
               "adapter": str(rel / "adapter"), "weights": None, "load": {}, "lineage": [
                   {"op": "imprint", "params": {"scheme": "pro"}}]}, open(rel / "release.json", "w"))
    r = omx.Release.open(str(rel))
    k = omx.load_key(r)
    assert r.scheme == "pro" and k.scheme == "pro"
    assert k.commitment == commitment                      # the stored name is part of the commitment: never rewritten
    sch = build_scheme(k)
    assert type(sch).__name__ == "OpenMark" and sch.params["delta"] == 1.0
    assert _rng_device(k) is None                           # OpenMark's mapping is device-independent, under either name


def test_old_key_store_found_through_mwx_keys(tmp_path):
    r = py("from openmarkx import keys; print(keys.STORE)", MWX_KEYS=str(tmp_path / "old"))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(tmp_path / "old")


def test_a_run_started_as_pro_resumes_as_openmark():
    from openmarkx.pipeline import config_diff
    old = {"model": {"name": "text"}, "scheme": {"name": "pro", "params": {"delta": 2.0}}, "out": "/a"}
    assert config_diff(old, {**old, "scheme": {"name": "openmark-frozen", "params": {"delta": 2.0}}, "out": "/b"}) == []
    assert config_diff(old, {**old, "scheme": {"name": "openmark-frozen", "params": {"delta": 1.0}}}) == ["scheme"]
    assert config_diff(old, {**old, "scheme": {"name": "openmark", "params": {"delta": 2.0}}}) == ["scheme"]
    assert config_diff(old, {**old, "scheme": {"name": "gaussmark", "params": {"delta": 2.0}}}) == ["scheme"]


def test_bench_finds_results_made_under_the_old_name(tmp_path, monkeypatch):
    from openmarkx import bench
    monkeypatch.setattr(bench, "RUNS", str(tmp_path))
    assert bench.result_dir("openmark-coopt-c100", "llama-3.2-1b").endswith("/e2/llama-3.2-1b/openmark-coopt-c100")
    old = tmp_path / "e2" / "llama-3.2-1b" / "pro-coopt-c100"
    old.mkdir(parents=True)
    assert bench.result_dir("openmark-coopt-c100", "llama-3.2-1b") == str(old)
    assert bench.result_dir("pro-coopt-c100", "llama-3.2-1b") == str(old)


# ---- environment variables and the home directory -----------------------------------------------------------------

def test_mwx_variables_are_read_when_omx_ones_are_unset(monkeypatch):
    from openmarkx.paths import env
    monkeypatch.delenv("OMX_RUNS", raising=False)
    monkeypatch.setenv("MWX_RUNS", "/old")
    assert env("RUNS") == "/old"
    monkeypatch.setenv("OMX_RUNS", "/new")
    assert env("RUNS") == "/new"
    monkeypatch.delenv("OMX_RUNS")
    monkeypatch.delenv("MWX_RUNS")
    assert env("RUNS", "/default") == "/default"


def test_paths_follow_mwx_runs_in_a_fresh_process(tmp_path):
    r = py("from openmarkx import paths; print(paths.RUNS); print(paths.DATA['llama3'])", MWX_RUNS=str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == [str(tmp_path), str(tmp_path / "data_llama8b_256.pt")]


def test_configs_expand_either_name(monkeypatch):
    from openmarkx.pipeline import _expand
    for k in ("OMX_RUNS", "MWX_RUNS", "OMX_DATA", "MWX_DATA", "OMX_DEMO_DATA", "MWX_DEMO_DATA"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MWX_RUNS", "/r")                    # an environment set up under the old name
    assert _expand({"out": "$OMX_RUNS/x", "old": "${MWX_RUNS}/y", "data": "$OMX_DEMO_DATA/z", "n": 3}) == \
        {"out": "/r/x", "old": "/r/y", "data": "/r/z", "n": 3}
    monkeypatch.delenv("MWX_RUNS")
    monkeypatch.setenv("OMX_RUNS", "/n")
    assert _expand(["$MWX_RUNS/a", "$UNSET_OMX_THING/b"]) == ["/n/a", "$UNSET_OMX_THING/b"]


def test_external_checkouts_fall_back_to_mwx_names(tmp_path, monkeypatch):
    from openmarkx.external import require_external
    (tmp_path / "fake_ext_mod.py").write_text("X = 1\n")
    monkeypatch.delenv("OMX_FAKE_DIR", raising=False)
    monkeypatch.setenv("MWX_FAKE_DIR", str(tmp_path))
    assert require_external("Fake", "OMX_FAKE_DIR", "fake_ext_mod").X == 1


@pytest.mark.parametrize("have,want", [((".markweightx",), ".markweightx"), ((), ".openmarkx"),
                                       ((".markweightx", ".openmarkx"), ".openmarkx")])
def test_home_falls_back_to_the_old_directory(tmp_path, have, want):
    for d in have:
        (tmp_path / d).mkdir()
    r = py("from openmarkx import paths; print(paths.HOME)", HOME=str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(tmp_path / want)
