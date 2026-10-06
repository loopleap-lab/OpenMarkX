"""markweightx bench ARENA.yaml: the whole benchmark for a list of methods in one command.

  markweightx bench configs/arena/1b.yaml --dry-run      # what is done, what would run
  markweightx bench configs/arena/1b.yaml                # run the missing steps here, in order
  markweightx bench configs/arena/1b.yaml --queue lab2   # hand them to exp/scheduler.py, one job per step

An arena file names the base and the methods (built-in names with params, or plugin files):

  base: llama-3.2-1b
  methods: [pro-coopt-c100, gaussmark, my_scheme.py]
  steps: [learn, edits, forge]          # optional; default all three
  - {method: markllm-exp-d, steps: [learn]}   # a method entry may narrow the steps
  learn: {steps: 1500}                  # optional overrides of exp/learnability.py (steps, bs, accum, bos)
  edits: [finetune:steps=800, quantize:bits=4, text:synonym:ratio=0.3]   # optional; default EDIT_GRID

Per method: learn (exp/learnability.py: embed, sample, TPR at 1% FPR, PPL), edits (exp/robustness.py: every weight
and text edit in EDIT_GRID, or the arena's own list), forge (exp/spoof.py). Then the leaderboard is rebuilt
(exp/leaderboard.py). A plugin file must pass `markweightx check` first (CPU, about a minute), as it would to join the
board by hand. A step counts as done when its output exists, so a bench can be stopped and run again. The steps run
the scripts in exp/, so bench runs from a checkout of the repository; they trust the plugin keys in your own RUNS.
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


def parse_edit(spec: str) -> tuple[str, dict]:
    """'finetune:steps=800' -> ("finetune", {"steps": 800}); 'text:synonym:ratio=0.3' -> ("text:synonym", {...})."""
    from .api import _spec
    if not isinstance(spec, str):
        raise ValueError(f"an edit is written as a string like 'finetune:steps=800', not {spec!r}")
    if spec.startswith("text:"):
        op, params = _spec(spec[5:])
        return "text:" + op, params
    return _spec(spec)


def load_arena(path: str) -> dict:
    import yaml
    a = yaml.safe_load(open(path)) or {}
    keys = {"base", "tag", "methods", "steps", "learn", "edits"}
    unknown = sorted(set(a) - keys)
    if unknown or not a.get("methods"):
        raise ValueError(f"{path}: needs 'methods' (a list); allowed keys: {', '.join(sorted(keys))}"
                         + (f"; unknown: {unknown}" if unknown else ""))
    for m in a["methods"]:
        if isinstance(m, dict) and (set(m) - {"method", "steps"} or "method" not in m):
            raise ValueError(f"{path}: a method entry is a name or {{method: NAME, steps: [...]}}, not {m}")
    bad = sorted((set(a.get("steps", STEPS)) | {s for m in a["methods"] if isinstance(m, dict) for s in m.get("steps", [])})
                 - set(STEPS))
    if bad:
        raise ValueError(f"{path}: unknown steps {bad}; allowed: {', '.join(STEPS)}")
    allowed_learn = {"steps", "bs", "accum", "bos"}             # not n: every step reads the 500-prompt texts
    if set(a.get("learn", {})) - allowed_learn:
        raise ValueError(f"{path}: learn takes {sorted(allowed_learn)}")
    grid = [parse_edit(e) for e in a["edits"]] if a.get("edits") else EDIT_GRID
    if a.get("edits"):                                      # fail here, not on a GPU after the learn step
        import inspect
        from .edits.text import rewrite
        from .edits.weights import RANGES, check_params
        from .registry import EDITS
        from . import edits  # noqa: F401  (registers the weight edits)
        texts = ("word_deletion", "synonym", "context_synonym")
        for op, params in grid:
            if op.startswith("text:"):
                if op[5:] not in texts:
                    raise ValueError(f"{path}: unknown text edit {op[5:]!r}; available: {', '.join(texts)}")
                ratio = {op[5:]: {"ratio": (lambda v: 0 < v <= 1, "in (0, 1]")}}
                check_params(rewrite, op[5:], params, ratio, skip=("texts", "kind", "device"))
            elif op not in EDITS:
                raise ValueError(f"{path}: unknown edit {op!r}; available: {', '.join(sorted(EDITS))}, text:<kind>")
            else:
                check_params(EDITS[op], op, params, RANGES)
    return {"base": a.get("base", "llama-3.2-1b"), "tag": a.get("tag", "e2"), "methods": list(a["methods"]),
            "steps": list(a.get("steps", STEPS)), "learn": dict(a.get("learn", {})), "grid": grid,
            "custom_grid": bool(a.get("edits"))}


def _absolute(m: str) -> str:
    """A plugin file given relative to where bench is run: the steps run from the repo root."""
    head, sep, rest = m.partition(":")
    return os.path.abspath(head) + sep + rest if head.endswith(".py") else m


def plugins(arena: dict) -> list[str]:
    """The plugin specs in an arena (file.py or file.py:name, params dropped), absolute."""
    out = set()
    for e in arena["methods"]:
        m = _absolute(e["method"] if isinstance(e, dict) else e)
        head, _, rest = m.partition(":")
        if head.endswith(".py"):
            name = rest.split(":")[0] if rest and "=" not in rest.split(":")[0] else ""
            out.add(head + (":" + name if name else ""))
    return sorted(out)


def plan(arena: dict, python: str = sys.executable) -> list[dict]:
    """One job per step: {"id", "cmd", "requires", "done", "method"}. Commands use --device cuda:0 and are meant to be
    run with CUDA_VISIBLE_DEVICES set to one GPU (as exp/scheduler.py does). A method listed twice is planned once."""
    jobs, seen = [], {}
    grid_arg = (" --grid " + shlex.quote(json.dumps(arena["grid"]))) if arena.get("custom_grid") else ""
    for entry in arena["methods"]:
        m, only = (entry["method"], entry.get("steps")) if isinstance(entry, dict) else (entry, None)
        m = _absolute(m)
        d = result_dir(m, arena["base"], arena["tag"])
        if d in seen:
            if seen[d].partition(":")[0] != m.partition(":")[0] and m.partition(":")[0].endswith(".py"):
                raise ValueError(f"{m} and {seen[d]} register the same scheme name from different files: rename one")
            print(f"note: {m} is the same run as {seen[d]}; planned once", file=sys.stderr)
            continue
        seen[d] = m
        name = os.path.basename(d)
        result = f"{d}/result.json"
        _same_code(m, result)
        rows = _rows(d)
        learn = " ".join(f"--{k}" if v is True else f"--{k} {v}" for k, v in arena["learn"].items() if v is not False)
        steps = {
            "learn": (f"{python} exp/learnability.py --scheme {shlex.quote(m)} --base {arena['base']} "
                      f"--tag {arena['tag']} --device cuda:0 {learn}".strip(), [], os.path.exists(result)),
            "edits": (f"{python} exp/robustness.py --release {shlex.quote(d)} --device cuda:0{grid_arg}", [result],
                      all(edit_tag(op, p) in rows for op, p in arena["grid"])),
            "forge": (f"{python} exp/spoof.py --release {shlex.quote(d)} --device cuda:0", [result], FORGE_ROW in rows),
        }
        for s in only or arena["steps"]:
            cmd, req, done = steps[s]
            jobs.append({"id": f"bench:{s}:{name}", "cmd": cmd, "requires": req, "done": done, "mem_gb": 14,
                         "method": name})
    return jobs


def _same_code(m: str, result: str) -> None:
    """A finished run of a plugin scheme must have been made by this file's code, or its results are someone else's."""
    head = m.partition(":")[0]
    if not head.endswith(".py") or not os.path.exists(result):
        return
    import hashlib
    src = json.load(open(result)).get("scheme_source") or ""
    sha = hashlib.sha256(open(head, "rb").read()).hexdigest()
    if src.startswith("blob:") and not src[5:].startswith(sha):
        raise ValueError(f"{os.path.dirname(result)} holds results made by other code than {head} (an earlier version, "
                         f"or another file registering the same scheme name): rename the scheme, or move that run away")


def check_plugins(arena: dict) -> list[str]:
    """Run `markweightx check` on every plugin file in the arena; return the files that fail."""
    from .check import report, run
    from .api import scheme_spec
    failed = []
    for f in plugins(arena):
        name, _, src = scheme_spec(f)                       # file.py, or file.py:name for a file with several
        rows = run(name, source=src, base=arena["base"])
        bad = [r for r in rows if r[1] == "FAIL"]
        print(f"check {os.path.basename(f)}: {'PASS' if not bad else 'FAIL'}", flush=True)
        if bad:
            print(report(bad), flush=True)
            failed.append(f)
    return failed


def leaderboard_cmd(arena: dict, python: str = sys.executable) -> str:
    return f"{python} exp/leaderboard.py --tag {arena['tag']} --base {arena['base']}"


def show(jobs: list[dict]) -> str:
    return "\n".join(f"{'done   ' if j['done'] else 'pending'}  {j['id']}" for j in jobs)


def run_here(arena: dict, jobs: list[dict]) -> int:
    """Run the pending steps one after another on this machine's first GPU, then rebuild the leaderboard. A failed
    step skips the rest of its method; the exit code is nonzero if any step failed."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES") or "0")
    failed = set()
    for j in jobs:
        if j["done"]:
            continue
        if j["method"] in failed:
            print(f"skip {j['id']}: an earlier step of {j['method']} failed", flush=True)
            continue
        if not all(os.path.exists(p) for p in j["requires"]):
            print(f"skip {j['id']}: its learn step has not produced {j['requires'][0]}", flush=True)
            failed.add(j["method"])
            continue
        print(f"== {j['id']}", flush=True)
        r = subprocess.run(j["cmd"], shell=True, cwd=ROOT, env=env)
        if r.returncode != 0:
            print(f"{j['id']} failed (exit {r.returncode}); the rest of {j['method']} is skipped", flush=True)
            failed.add(j["method"])
    board = subprocess.run(leaderboard_cmd(arena), shell=True, cwd=ROOT).returncode
    return 1 if failed else board


def queue(jobs: list[dict], host: str) -> list[str]:
    """Append the pending steps to exp/scheduler.py's queue for `host`. A step already queued is left alone, unless
    the scheduler recorded it as failed: then it is queued again under a new id (<id>-retryN)."""
    d = f"{RUNS}/sched/{host}"
    f, sf = f"{d}/queue.json", f"{d}/state.json"
    q = json.load(open(f)) if os.path.exists(f) else []
    state = {}
    for _ in range(5):                                      # the scheduler rewrites state.json while it runs
        try:
            state = json.load(open(sf)) if os.path.exists(sf) else {}
            break
        except json.JSONDecodeError:
            import time
            time.sleep(1)
    have = {j["id"] for j in q}
    new = []
    for j in jobs:
        if j["done"]:
            continue
        jid = j["id"]
        while jid in have and state.get(jid, {}).get("status") == "failed":
            n = int(jid.rsplit("-retry", 1)[1]) + 1 if "-retry" in jid else 1
            jid = f"{j['id']}-retry{n}"
        if jid in have:
            continue
        new.append({"id": jid, "cmd": j["cmd"], "requires": j["requires"], "mem_gb": j["mem_gb"]})
        have.add(jid)
    os.makedirs(d, exist_ok=True)
    tmp = f"{f}.tmp{os.getpid()}"
    json.dump(q + new, open(tmp, "w"), indent=1)
    os.replace(tmp, f)                                     # the scheduler never reads a half-written queue
    return [j["id"] for j in new]
