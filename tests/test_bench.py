"""markweightx bench: arena files, the step plan, and the scheduler queue."""
import json
import os

import pytest

from markweightx import bench


def test_arena_is_validated(tmp_path):
    f = tmp_path / "a.yaml"
    f.write_text("methods: [gaussmark]\nsteps: [learn, attack]\n")
    with pytest.raises(ValueError, match="unknown steps"):
        bench.load_arena(str(f))
    f.write_text("base: llama-3.2-1b\n")
    with pytest.raises(ValueError, match="needs 'methods'"):
        bench.load_arena(str(f))


def test_plan_skips_finished_steps_and_queues_the_rest(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "RUNS", str(tmp_path))
    f = tmp_path / "a.yaml"
    f.write_text("methods: [gaussmark, {method: markllm-kgw-d, steps: [learn]}]\n")
    arena = bench.load_arena(str(f))
    jobs = bench.plan(arena, python="python")
    assert [j["id"] for j in jobs] == ["bench:learn:gaussmark", "bench:edits:gaussmark", "bench:forge:gaussmark",
                                       "bench:learn:markllm-kgw-d"]
    assert not any(j["done"] for j in jobs)
    d = tmp_path / "e2" / "llama-3.2-1b" / "gaussmark"
    d.mkdir(parents=True)
    (d / "result.json").write_text("{}")
    (d / "robustness.jsonl").write_text(json.dumps({"edit": bench.FORGE_ROW}) + "\n")
    done = {j["id"]: j["done"] for j in bench.plan(arena, python="python")}
    assert done["bench:learn:gaussmark"] and done["bench:forge:gaussmark"] and not done["bench:edits:gaussmark"]
    new = bench.queue(bench.plan(arena, python="python"), "testhost")
    assert new == ["bench:edits:gaussmark", "bench:learn:markllm-kgw-d"]
    assert bench.queue(bench.plan(arena, python="python"), "testhost") == []   # already queued
    q = json.load(open(tmp_path / "sched" / "testhost" / "queue.json"))
    assert q[0]["requires"] == [str(d / "result.json")]


def test_learnability_and_bench_agree_on_the_result_dir():
    assert bench.result_dir("gaussmark:sigma=0.02", "llama-3.2-1b").endswith("/e2/llama-3.2-1b/gaussmark_sigma0.02")


def test_same_run_twice_is_planned_once_and_relative_plugins_become_absolute(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "RUNS", str(tmp_path))
    plug = tmp_path / "bob" / "bob_mark.py"; plug.parent.mkdir()
    plug.write_text(open(os.path.join(os.path.dirname(__file__), "..", "examples", "my_scheme.py")).read()
                    .replace('"my-scheme"', '"bob-mark"'))
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "a.yaml"
    f.write_text("methods: [gaussmark, gaussmark, bob/bob_mark.py:bob-mark]\nsteps: [learn]\n")
    jobs = bench.plan(bench.load_arena(str(f)), python="python")
    assert [j["id"] for j in jobs] == ["bench:learn:gaussmark", "bench:learn:bob-mark"]
    assert str(plug) + ":bob-mark" in jobs[1]["cmd"]


def test_run_here_skips_the_rest_of_a_failed_method(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(bench.subprocess, "run", lambda cmd, **k: calls.append(cmd) or
                        type("R", (), {"returncode": 3 if cmd == "edits" else 0})())
    jobs = [{"id": "e", "cmd": "edits", "requires": [], "done": False, "method": "m"},
            {"id": "f", "cmd": "forge", "requires": [], "done": False, "method": "m"}]
    rc = bench.run_here({"tag": "e2", "base": "b"}, jobs)
    assert "forge" not in calls and rc == 1


def test_a_failed_step_is_queued_again_under_a_new_id(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "RUNS", str(tmp_path))
    d = tmp_path / "sched" / "h"; d.mkdir(parents=True)
    (d / "queue.json").write_text(json.dumps([{"id": "bench:edits:m", "cmd": "x", "requires": []}]))
    (d / "state.json").write_text(json.dumps({"bench:edits:m": {"status": "failed"}}))
    job = {"id": "bench:edits:m", "cmd": "x", "requires": [], "done": False, "mem_gb": 14, "method": "m"}
    assert bench.queue([job], "h") == ["bench:edits:m-retry1"]
    assert bench.queue([job], "h") == []                    # the retry is queued and not failed yet


def test_learn_n_is_not_an_arena_option(tmp_path):
    f = tmp_path / "a.yaml"
    f.write_text("methods: [gaussmark]\nlearn: {n: 100}\n")
    with pytest.raises(ValueError, match="learn takes"):
        bench.load_arena(str(f))
