"""The demo configs keep the settings of the runs they reproduce. The expected values below are the ones those runs
used (the run files themselves are not part of the repository)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from openmarkx.pipeline import load_config
from openmarkx.registry import canonical_scheme


def demo(name):
    return load_config(os.path.join(ROOT, "configs", "demos", f"{name}.yaml"))


def test_taming_colearns_then_fixes_the_rule():
    cfg = demo("taming")
    a, b = cfg.imprint["phases"]
    assert (a["lr"], a["policy_lr"], a["steps"]) == (3e-5, 0.2, 800)        # phase 1: co-learn the green lists
    assert (b["lr"], b["policy_lr"], b["steps"]) == (1e-4, 0.0, 1000)       # phase 2: fix them, keep training
    assert b["fix_rule"] and a["trainable"] == b["trainable"] == "full"
    assert cfg.scheme["name"] == "wmar" and cfg.scheme["params"]["colearn"]


def test_geopt_closed_form_last_layer():
    cfg = demo("geopt")
    assert cfg.scheme["params"]["eps"] == 0.02 and cfg.scheme["params"]["output_scale"] == 1.0
    assert [p["steps"] for p in cfg.imprint["phases"]] == [0]
    assert cfg.imprint["edit_last"] is True                  # run_mark.py's default --edit_last 1


def test_agent_is_openmark_frozen_delta2_300_words_t07():
    cfg = demo("agent")
    assert canonical_scheme(cfg.scheme["name"]) == canonical_scheme("pro") == "openmark-frozen"   # the run used "pro"
    assert cfg.scheme["params"]["delta"] == 2.0
    assert cfg.model["base"] == "llama-3.1-8b-instruct"
    assert cfg.imprint["seq_len"] == 256 and cfg.imprint["bs"] == 16
    assert cfg.model["final_words"] == 300 and cfg.model["final_tokens"] == 600
    assert cfg.generate["n"] == 300
    assert cfg.model["temperature"] == 0.7


def test_medgemma_is_openmark_frozen():
    cfg = demo("medgemma")
    assert canonical_scheme(cfg.scheme["name"]) == canonical_scheme("pro") == "openmark-frozen"
    assert cfg.imprint["bs"] == 16 and cfg.imprint["answer_len"] == 256
    assert cfg.model["name"] == "vlm" and cfg.model["base"] == "medgemma-1.5-4b-it"


def test_defaults_the_runs_did_not_override():
    """Settings the runs took from their scripts' defaults: steps 1500 and lr 2e-4 (learnability.py, embed.py)."""
    for name in ("agent", "medgemma"):
        p = demo(name).imprint["phases"][0]
        assert (p["steps"], p["lr"], p["rank"]) == (1500, 2e-4, 64)


def test_inputs_come_from_omx_demo_data_and_outputs_go_to_omx_runs(monkeypatch):
    monkeypatch.setenv("OMX_RUNS", "/runs")
    monkeypatch.setenv("OMX_DEMO_DATA", "/data")
    cfg = demo("agent")
    assert cfg.data["train"].startswith("/data/") and cfg.data["eval"].startswith("/data/") and cfg.out.startswith("/runs/")
    monkeypatch.delenv("OMX_DEMO_DATA")
    monkeypatch.setenv("OMX_DATA", "/corpus.pt")           # the tokenized corpus file: not where demo inputs are read
    assert demo("medgemma").data["eval"].startswith("/runs/")
