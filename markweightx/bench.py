"""markweightx bench ARENA.yaml: the whole benchmark for a list of methods in one command.

  markweightx bench configs/arena/1b.yaml --dry-run      # what is done, what would run
  markweightx bench configs/arena/1b.yaml                # run the missing steps here, in order
  markweightx bench configs/arena/1b.yaml --queue lab2   # hand them to exp/scheduler.py, one job per step

An arena file names the base and the methods (built-in names with params, or plugin files):

  base: llama-3.2-1b
  methods: [pro-coopt-c100, gaussmark, my_scheme.py]
  steps: [learn, edits, forge]          # optional; default all three
  - {method: markllm-exp-d, steps: [learn]}   # a method entry may narrow the steps
  learn: {steps: 1500, n: 500}          # optional overrides of exp/learnability.py

Per method: learn (exp/learnability.py: embed, sample, TPR at 1% FPR, PPL), edits (exp/robustness.py: every weight
and text edit in EDIT_GRID), forge (exp/spoof.py). Then the leaderboard is rebuilt (exp/leaderboard.py). Finished
steps are skipped, so a bench can be stopped and run again. The steps run the scripts in exp/, so bench runs from a
checkout of the repository.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys

from .paths import RUNS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDIT_GRID = [("finetune", dict(steps=200)), ("finetune", dict(steps=800)), ("finetune", dict(steps=2400)),
             ("merge", dict(ratio=0.2)), ("merge", dict(ratio=0.4)), ("merge", dict(ratio=0.6)),
             ("prune", dict(sparsity=0.2)), ("prune", dict(sparsity=0.5)),
             ("quantize", dict(bits=8)), ("quantize", dict(bits=4)),
             ("text:word_deletion", dict(ratio=0.3)), ("text:synonym", dict(ratio=0.3))]
FORGE_ROW = "attack:spoof_unigram"
STEPS = ("learn", "edits", "forge")


def edit_tag(op: str, params: dict) -> str:
    return op + "_" + "_".join(f"{k}{v}" for k, v in params.items())


def result_dir(watermark: str, base: str, tag: str = "e2") -> str:
    """Where exp/learnability.py puts a method's release and results."""
    from .api import scheme_spec
    name, params, _ = scheme_spec(watermark)
    return f"{RUNS}/{tag}/{base}/" + name + "".join(f"_{k}{v}" for k, v in params.items())


def _rows(d: str) -> set:
    f = f"{d}/robustness.jsonl"
    return {json.loads(line)["edit"] for line in open(f)} if os.path.exists(f) else set()


def load_arena(path: str) -> dict:
    import yaml
    a = yaml.safe_load(open(path)) or {}
    unknown = sorted(set(a) - {"base", "tag", "methods", "steps", "learn"})
    if unknown or not a.get("methods"):
        raise ValueError(f"{path}: needs 'methods' (a list); allowed keys: base, tag, methods, steps, learn"
                         + (f"; unknown: {unknown}" if unknown else ""))
    for m in a["methods"]:
        if isinstance(m, dict) and (set(m) - {"method", "steps"} or "method" not in m):
            raise ValueError(f"{path}: a method entry is a name or {{method: NAME, steps: [...]}}, not {m}")
    bad = sorted((set(a.get("steps", STEPS)) | {s for m in a["methods"] if isinstance(m, dict) for s in m.get("steps", [])})
                 - set(STEPS))
    if bad:
        raise ValueError(f"{path}: unknown steps {bad}; allowed: {', '.join(STEPS)}")
    allowed_learn = {"steps", "n", "bs", "accum", "bos"}
    if set(a.get("learn", {})) - allowed_learn:
        raise ValueError(f"{path}: learn takes {sorted(allowed_learn)}")
    return {"base": a.get("base", "llama-3.2-1b"), "tag": a.get("tag", "e2"), "methods": list(a["methods"]),
            "steps": list(a.get("steps", STEPS)), "learn": dict(a.get("learn", {}))}


def plan(arena: dict, python: str = sys.executable) -> list[dict]:
    """One job per missing step: {"id", "cmd", "requires", "done"}. Commands use --device cuda:0 and are meant to be
    run with CUDA_VISIBLE_DEVICES set to one GPU (as exp/scheduler.py does)."""
    jobs = []
    for entry in arena["methods"]:
        m, only = (entry["method"], entry.get("steps")) if isinstance(entry, dict) else (entry, None)
        if m.endswith(".py"):
            m = os.path.abspath(m)
        d = result_dir(m, arena["base"], arena["tag"])
        name = os.path.basename(d)
        result = f"{d}/result.json"
        rows = _rows(d)
        learn = " ".join(f"--{k}" if v is True else f"--{k} {v}" for k, v in arena["learn"].items() if v is not False)
        steps = {
            "learn": (f"{python} exp/learnability.py --scheme {shlex.quote(m)} --base {arena['base']} "
                      f"--tag {arena['tag']} --device cuda:0 {learn}".strip(), [], os.path.exists(result)),
            "edits": (f"{python} exp/robustness.py --release {shlex.quote(d)} --device cuda:0", [result],
                      all(edit_tag(op, p) in rows for op, p in EDIT_GRID)),
            "forge": (f"{python} exp/spoof.py --release {shlex.quote(d)} --device cuda:0", [result], FORGE_ROW in rows),
        }
        for s in only or arena["steps"]:
            cmd, req, done = steps[s]
            jobs.append({"id": f"bench:{s}:{name}", "cmd": cmd, "requires": req, "done": done, "mem_gb": 14})
    return jobs


def leaderboard_cmd(arena: dict, python: str = sys.executable) -> str:
    return f"{python} exp/leaderboard.py --tag {arena['tag']} --base {arena['base']}"


def show(jobs: list[dict]) -> str:
    return "\n".join(f"{'done   ' if j['done'] else 'pending'}  {j['id']}" for j in jobs)


def run_here(arena: dict, jobs: list[dict]) -> int:
    """Run the pending steps one after another on this machine's first GPU, then rebuild the leaderboard."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    for j in jobs:
        if j["done"]:
            continue
        if not all(os.path.exists(p) for p in j["requires"]):
            print(f"skip {j['id']}: its learn step has not produced {j['requires'][0]}", flush=True)
            continue
        print(f"== {j['id']}", flush=True)
        r = subprocess.run(j["cmd"], shell=True, cwd=ROOT, env=env)
        if r.returncode != 0:
            print(f"{j['id']} failed (exit {r.returncode}); later steps of this method are skipped", flush=True)
    return subprocess.run(leaderboard_cmd(arena), shell=True, cwd=ROOT).returncode


def queue(jobs: list[dict], host: str) -> list[str]:
    """Append the pending steps to exp/scheduler.py's queue for `host` (ids already queued are left alone)."""
    f = f"{RUNS}/sched/{host}/queue.json"
    q = json.load(open(f)) if os.path.exists(f) else []
    have = {j["id"] for j in q}
    new = [{k: j[k] for k in ("id", "cmd", "requires", "mem_gb")} for j in jobs if not j["done"] and j["id"] not in have]
    os.makedirs(os.path.dirname(f), exist_ok=True)
    json.dump(q + new, open(f, "w"), indent=1)
    return [j["id"] for j in new]
