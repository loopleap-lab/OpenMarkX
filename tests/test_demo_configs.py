"""The four demo configs are last night's runs, setting by setting: each value is read from the run file it came from."""
import os
import re
import shlex
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from openmarkx.pipeline import load_config
from openmarkx.registry import canonical_scheme


def line_args(path: str, name: str) -> dict:
    """--flag value pairs of the line (sweep file) or command (sbatch) that starts with / contains `name`."""
    for line in open(os.path.join(ROOT, path)):
        if line.startswith(name + " ") or (name in line and "python" in line):
            toks = shlex.split(line.split("python", 1)[-1] if "python" in line else line)
            return {t[2:]: toks[i + 1] for i, t in enumerate(toks[:-1]) if t.startswith("--")}
    raise AssertionError(f"{name} not in {path}")


def demo(name):
    return load_config(os.path.join(ROOT, "configs", "demos", f"{name}.yaml"))


def test_taming_is_p2_then_p14():
    cfg = demo("taming")
    p2 = line_args("exp/arimg/jobs/colearn_a.txt", "P2_plr2e-1")
    p14 = line_args("exp/arimg/jobs/colearn_g.txt", "P14_fromP2_fixed_lr1e-4")
    a, b = cfg.imprint["phases"]
    assert (a["lr"], a["policy_lr"], a["steps"]) == (float(p2["lr"]), float(p2["policy_lr"]), int(p2["steps"]))
    assert (b["lr"], b["policy_lr"], b["steps"]) == (float(p14["lr"]), float(p14["policy_lr"]), int(p14["steps"]))
    assert b["fix_rule"] and a["trainable"] == b["trainable"] == "full"
    assert p14["init_dir"].endswith("P2_plr2e-1")           # P14 continued from P2: phase 2 continues phase 1
    assert cfg.scheme["name"] == "wmar" and cfg.scheme["params"]["colearn"]


def test_geopt_is_s1_e02():
    cfg = demo("geopt")
    s = line_args("exp/geopt/jobs/sweep1.txt", "s1_e02")
    assert cfg.scheme["params"]["eps"] == float(s["eps"]) and cfg.scheme["params"]["output_scale"] == float(s["output_scale"])
    assert [p["steps"] for p in cfg.imprint["phases"]] == [int(s["steps"])]
    assert cfg.imprint["edit_last"] is True                  # run_mark.py's default --edit_last 1


def test_agent_is_openmark_delta2_a300t07():
    cfg = demo("agent")
    emb = line_args("exp/agent/jobs/agent-pro-d2.sbatch", "learnability.py")
    run = line_args("exp/agent/jobs/agent-runv.sbatch", "hotpot_agent.py")
    demo_page = line_args("exp/agent/jobs/agent-demod2.sbatch", "demo_page.py")
    scheme, _, delta = emb["scheme"].partition(":delta=")
    assert canonical_scheme(cfg.scheme["name"]) == canonical_scheme(scheme)   # the job ran it as "pro"
    assert cfg.scheme["params"]["delta"] == float(delta)
    assert cfg.model["base"] == emb["base"]
    assert cfg.imprint["seq_len"] == int(emb["seq_len"]) and cfg.imprint["bs"] == int(emb["bs"])
    assert cfg.model["final_words"] == int(run["final_words"]) and cfg.model["final_tokens"] == int(run["final_tokens"])
    assert cfg.generate["n"] == int(run["n"])
    t = re.fullmatch(r"a(\d+)t(\d+)", demo_page["prefix"])   # the a300t07 variant: 300 words, temperature 0.7
    assert cfg.model["temperature"] == float(t.group(2)[0] + "." + t.group(2)[1:])


def test_medgemma_is_xmed_openmark():
    cfg = demo("medgemma")
    a = line_args("exp/med/jobs/Xmed-pro.sbatch", "embed.py")
    assert canonical_scheme(cfg.scheme["name"]) == canonical_scheme(a["scheme"])
    assert cfg.imprint["bs"] == int(a["bs"]) and cfg.imprint["answer_len"] == int(a["answer_len"])
    assert cfg.model["name"] == "vlm" and cfg.model["base"] == "medgemma-1.5-4b-it"


def test_defaults_the_runs_did_not_override():
    """Settings last night's scripts took from their defaults: steps 1500 and lr 2e-4 (learnability.py, embed.py)."""
    for name in ("agent", "medgemma"):
        p = demo(name).imprint["phases"][0]
        assert (p["steps"], p["lr"], p["rank"]) == (1500, 2e-4, 64)


def test_inputs_come_from_omx_data_and_outputs_go_to_omx_runs(monkeypatch):
    monkeypatch.setenv("OMX_RUNS", "/runs")
    monkeypatch.setenv("OMX_DATA", "/data")
    cfg = demo("agent")
    assert cfg.data["train"].startswith("/data/") and cfg.data["eval"].startswith("/data/") and cfg.out.startswith("/runs/")
    monkeypatch.delenv("OMX_DATA")
    assert demo("medgemma").data["eval"].startswith("/runs/")
