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
