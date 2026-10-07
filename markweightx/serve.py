"""`markweightx serve`: run the toolkit on this machine's GPUs from a web page.

The server listens on 127.0.0.1 and answers only requests that carry its token. The page is either the one it serves
itself (http://127.0.0.1:PORT/#token=...) or the project website, which connects to it from the browser. Uploaded
data, releases and keys stay on this machine. A GPU on another machine is reached with an SSH tunnel:
    ssh -L 7860:127.0.0.1:7860 gpu-server      # then connect the page to http://127.0.0.1:7860

A job (upload a dataset -> embed -> calibrate -> evaluate) runs as its own process, `python -m markweightx.serve
JOB_DIR`, so a crash or a cancel never takes the server down. Its state is files in JOB_DIR:
    job.json        the request             status.json   phase, error, result
    train_log.jsonl one line per step       output.log    the job's stdout and stderr
"""
from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .paths import BASES, HOME

SITE = "https://jqxue1999.github.io"                   # the project website may connect; nothing else cross-origin
SERVE = os.environ.get("MWX_SERVE", os.path.join(HOME, "serve"))
MAX_UPLOAD = 512 * 2**20
RECIPE = {"steps": (int, 300), "lr": (float, 2e-4), "bs": (int, 8), "seq_len": (int, 128), "rank": (int, 64),
          "n_eval": (int, 64), "n_null": (int, 200), "new_tokens": (int, 100), "fpr": (float, 0.01), "seed": (int, 0)}
PHASES = ["queued", "tokenize", "train", "calibrate", "evaluate", "done"]


# ---------------------------------------------------------------------------------------------------- what's here
def machine() -> dict:
    """GPUs from nvidia-smi, so the server process never holds a CUDA context (memory) on them."""
    gpus = []
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.free,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout
        for line in filter(None, q.splitlines()):
            i, name, free, total = [x.strip() for x in line.split(",")]
            gpus.append({"device": f"cuda:{i}", "name": name, "free_gb": round(int(free) / 1024, 1),
                         "total_gb": round(int(total) / 1024, 1)})
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    from importlib.metadata import version
    return {"host": os.uname().nodename, "gpus": gpus, "torch": version("torch")}


def schemes() -> list[dict]:
    import inspect
    from . import schemes as _                           # registers the built-in schemes
    from .registry import SCHEMES
    out = []
    for name, cls in sorted(SCHEMES.items()):
        if name == "nowm" or getattr(cls, "experimental", False):
            continue
        params = {}
        for k in cls.__mro__:
            init = k.__dict__.get("__init__")
            for p in (inspect.signature(init).parameters.values() if init else []):
                if p.name != "self" and p.name not in params and isinstance(p.default, (int, float)) \
                        and not isinstance(p.default, bool):
                    params[p.name] = p.default
        out.append({"name": name, "kind": str(getattr(cls, "kind", "")), "params": params,
                    "doc": (cls.__doc__ or "").strip().split("\n")[0]})
    return out


def bases() -> list[dict]:
    from huggingface_hub import snapshot_download
    out = []
    for name, (repo, rev) in BASES.items():
        try:
            snapshot_download(repo, revision=rev, local_files_only=True); cached = True
        except Exception:
            cached = False
        out.append({"name": name, "repo": repo, "cached": cached,
                    "need_gb": 10 if "1b" in name else 40})       # peak measured on 1B at bs 8, seq 128: 8.5 GB
    return out


# ---------------------------------------------------------------------------------------------------------- jobs
def _read(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write(path, obj):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _alive(pid) -> bool:
    try:
        os.kill(pid, 0); return True
    except (OSError, TypeError):
        return False


def job_state(jid: str, log_after: int = 0) -> dict | None:
    d = os.path.join(SERVE, "jobs", jid)
    req = _read(os.path.join(d, "job.json"))
    if req is None:
        return None
    st = _read(os.path.join(d, "status.json"), {"phase": "queued"})
    if st["phase"] not in ("done", "failed", "cancelled") and not _alive(st.get("pid")) and st.get("pid"):
        st = {**st, "phase": "failed", "error": st.get("error") or "the job process exited (see output.log)"}
    steps = []
    try:
        with open(os.path.join(d, "release", "train_log.jsonl")) as f:
            steps = [json.loads(l) for l in f if l.strip()]
    except OSError:
        pass
    try:
        with open(os.path.join(d, "output.log"), errors="replace") as f:
            tail = f.read()[-4000:]
    except OSError:
        tail = ""
    return {"id": jid, "request": req, **st, "log": steps[log_after:], "n_log": len(steps), "output": tail,
            "now": time.time()}                           # the page times phases against the server's clock


def start_job(req: dict) -> str:
    jid = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    d = os.path.join(SERVE, "jobs", jid)
    os.makedirs(d)
    _write(os.path.join(d, "job.json"), req)
    _write(os.path.join(d, "status.json"), {"phase": "queued", "t0": time.time()})
    out = open(os.path.join(d, "output.log"), "w")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": req["device"].split(":")[1]} if req["device"].startswith("cuda:") \
        else dict(os.environ)
    p = subprocess.Popen([sys.executable, "-u", "-m", "markweightx.serve", d], stdout=out, stderr=subprocess.STDOUT,
                         env=env, start_new_session=True)
    _write(os.path.join(d, "status.json"), {"phase": "queued", "t0": time.time(), "pid": p.pid})
    threading.Thread(target=p.wait, daemon=True).start()      # reap it, so a finished job is not a zombie
    return jid


def cancel_job(jid: str) -> bool:
    d = os.path.join(SERVE, "jobs", jid)
    st = _read(os.path.join(d, "status.json"))
    if not st or not _alive(st.get("pid")):
        return False
    os.killpg(st["pid"], signal.SIGTERM)
    _write(os.path.join(d, "status.json"), {**st, "phase": "cancelled"})
    return True


def check_request(req: dict) -> dict:
    """The job a page asked for, with types checked and defaults filled; ValueError says what is wrong."""
    known = {s["name"]: s for s in schemes()}
    if req.get("scheme") not in known:
        raise ValueError(f"unknown scheme {req.get('scheme')!r}")
    if req.get("base") not in BASES:
        raise ValueError(f"unknown base model {req.get('base')!r}")
    ds = os.path.join(SERVE, "uploads", os.path.basename(str(req.get("dataset", ""))))
    if not req.get("dataset") or not os.path.isfile(ds):
        raise ValueError("upload a dataset first")
    out = {"scheme": req["scheme"], "base": req["base"], "dataset": ds, "device": str(req.get("device") or "cuda:0"),
           "scheme_params": {}}
    for k, (t, default) in RECIPE.items():
        out[k] = t(req.get(k, default))
    for k, v in (req.get("scheme_params") or {}).items():
        if k not in known[req["scheme"]]["params"]:
            raise ValueError(f"{req['scheme']} has no parameter {k!r}")
        out["scheme_params"][k] = float(v) if isinstance(known[req["scheme"]]["params"][k], float) else int(v)
    if not 1 <= out["steps"] <= 100000 or not 8 <= out["n_eval"] <= 2000 or not 20 <= out["n_null"] <= 5000:
        raise ValueError("steps must be 1..100000, n_eval 8..2000 and n_null 20..5000")
    if not 0 < out["fpr"] < 1:
        raise ValueError("the target FPR is between 0 and 1")
    return out


def save_upload(name: str, body: bytes) -> dict:
    ext = os.path.splitext(name)[1].lower()
    if ext not in (".jsonl", ".txt"):
        raise ValueError("a dataset is .jsonl ({\"text\": ...} per line) or .txt (one text per line)")
    texts = read_texts(body.decode("utf-8", errors="replace"), ext)
    if len(texts) < 20:
        raise ValueError(f"{name}: {len(texts)} texts; at least 20 are needed (some are held out for evaluation)")
    os.makedirs(os.path.join(SERVE, "uploads"), exist_ok=True)
    fid = uuid.uuid4().hex[:12] + ext
    with open(os.path.join(SERVE, "uploads", fid), "wb") as f:
        f.write(body)
    words = sum(len(t.split()) for t in texts)
    return {"id": fid, "name": name, "n_texts": len(texts), "n_words": words, "sample": texts[0][:300]}


def sample_dataset(n: int = 1000) -> dict:
    """n texts from the C4 validation split (streamed: one shard, about 40 MB), for a first run without data."""
    from datasets import load_dataset
    rows = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = [r["text"] for _, r in zip(range(n), rows)]
    return save_upload("c4-sample.jsonl", "".join(json.dumps({"text": t}) + "\n" for t in texts).encode())


def read_texts(raw: str, ext: str) -> list[str]:
    if ext == ".txt":
        return [l.strip() for l in raw.splitlines() if l.strip()]
    out = []
    for i, l in enumerate(raw.splitlines()):
        if l.strip():
            try:
                t = json.loads(l)
            except ValueError:
                raise ValueError(f"line {i + 1} is not JSON")
            t = t.get("text") if isinstance(t, dict) else t
            if not isinstance(t, str):
                raise ValueError(f"line {i + 1} has no \"text\" string")
            if t.strip():
                out.append(t.strip())
    return out


# -------------------------------------------------------------------------------------------------------- the job
def run_job(d: str):
    """embed -> calibrate -> evaluate, for the request in d/job.json. Runs in its own process."""
    req = _read(os.path.join(d, "job.json"))
    st = {**_read(os.path.join(d, "status.json"), {}), "pid": os.getpid()}

    def phase(p, **kw):
        st.update(kw, phase=p, **{"t_" + p: time.time()})
        _write(os.path.join(d, "status.json"), st)
    try:
        import random
        import torch
        from . import api
        from .models import load_tokenizer
        device = "cuda:0" if req["device"].startswith("cuda") else "cpu"     # CUDA_VISIBLE_DEVICES picked the GPU
        phase("tokenize")
        with open(req["dataset"], errors="replace") as f:
            texts = read_texts(f.read(), os.path.splitext(req["dataset"])[1])
        random.Random(req["seed"]).shuffle(texts)
        held = texts[:max(8, min(req["n_eval"], len(texts) // 5))]
        tok = load_tokenizer(req["base"])
        ids = []
        for t in texts[len(held):]:
            ids += tok(t, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        L = req["seq_len"]
        n = len(ids) // L
        if n < req["bs"]:
            raise ValueError(f"the training split has {len(ids)} tokens: {n} sequences of {L}, fewer than the batch "
                             f"size {req['bs']}. Upload more text or lower seq_len / bs.")
        torch.save({"train": torch.tensor(ids[:n * L]).view(n, L)}, os.path.join(d, "data.pt"))
        phase("train", n_train_seq=n, n_train_tokens=n * L, n_held=len(held))
        spec = req["scheme"] + "".join(f",{k}={v}" if i else f":{k}={v}"
                                       for i, (k, v) in enumerate(req["scheme_params"].items()))
        rel, key = api.embed(req["base"], spec, os.path.join(d, "release"), device=device, data=os.path.join(d, "data.pt"),
                             steps=req["steps"], lr=req["lr"], bs=req["bs"], seq_len=L, rank=req["rank"],
                             seed=req["seed"], save_every=max(req["steps"], 1))
        # held-out texts cut to their first 32 tokens are the prompts; their own continuations are the human texts
        cut = [tok(t, add_special_tokens=False)["input_ids"] for t in held]
        prompts = [tok.decode(c[:32]) for c in cut]
        human = [tok.decode(c[32:32 + req["new_tokens"]]) for c in cut if len(c) > 52]
        phase("calibrate", key=key.commitment[:12], release=rel.path)
        # the null needs more texts than there are held-out prompts (a 1% threshold from 32 texts is their maximum):
        # the prompts are reused, each with fresh samples from the unwatermarked base
        null_prompts = (prompts * -(-req["n_null"] // len(prompts)))[:req["n_null"]]
        key = api.calibrate(key, req["base"], null_prompts, device=device, seed=req["seed"] + 1)
        phase("evaluate")
        marked = api.generate(rel, prompts, device=device, new_tokens=req["new_tokens"], seed=req["seed"] + 2)
        dm = api.detect(marked, key, base=req["base"], fpr=req["fpr"])
        dh = api.detect(human, key, base=req["base"], fpr=req["fpr"]) if human else []
        row = lambda t, x: {"text": t, "p": x.p_value, "stat": x.statistic, "n_tokens": x.n_tokens, "flag": x.flagged}
        rate = lambda ds: sum(bool(x.flagged) for x in ds) / len(ds) if ds else None
        phase("done", result={"tpr": rate(dm), "fpr_human": rate(dh), "fpr": req["fpr"], "n_marked": len(dm),
                              "n_human": len(dh), "n_null": len(key.calibration["null"]),
                              "marked": [row(t, x) for t, x in zip(marked, dm)][:20],
                              "human": [row(t, x) for t, x in zip(human, dh)][:20],
                              "key": key.commitment, "release": rel.path})
    except BaseException as e:                            # SIGTERM (cancel) never reaches here: the group is killed
        import traceback
        traceback.print_exc()
        phase("failed", error=f"{type(e).__name__}: {e}")
        raise SystemExit(1)


# --------------------------------------------------------------------------------------------------------- server
APP = os.path.join(os.path.dirname(__file__), "app.html")


def make_handler(token: str, origins: set[str]):
    class H(BaseHTTPRequestHandler):
        server_version = "markweightx"

        def log_message(self, fmt, *a):                    # one short line per API call, none for polling
            if not self.path.startswith("/api/jobs/"):
                sys.stderr.write(f"  {self.command} {self.path.split('?')[0]}\n")

        def cors(self):
            o = self.headers.get("Origin")
            if o and (o in origins or urlparse(o).hostname in ("127.0.0.1", "localhost")):
                self.send_header("Access-Control-Allow-Origin", o)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Filename, X-MWX-Token")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Private-Network", "true")   # Chrome: public page -> localhost

        def reply(self, code, obj=None, ctype="application/json", body=None):
            body = body if body is not None else json.dumps(obj).encode()
            self.send_response(code)
            self.cors()
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204); self.cors(); self.end_headers()

        def authed(self) -> bool:
            # X-MWX-Token first: a proxy in front of the server (Colab's) may use or strip Authorization itself
            got = self.headers.get("X-MWX-Token") or self.headers.get("Authorization", "").removeprefix("Bearer ")
            ok = secrets.compare_digest(got.encode(), token.encode())
            if not ok:
                self.reply(401, {"error": "wrong or missing token (it is printed by `markweightx serve`)"})
            return ok

        def body(self) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_UPLOAD:
                raise ValueError(f"upload larger than {MAX_UPLOAD >> 20} MB")
            return self.rfile.read(n)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                with open(APP, "rb") as f:
                    return self.reply(200, ctype="text/html; charset=utf-8", body=f.read())
            if not self.authed():
                return
            try:
                if u.path == "/api/info":
                    return self.reply(200, {"machine": machine(), "schemes": schemes(), "bases": bases(),
                                            "recipe": {k: d for k, (_, d) in RECIPE.items()}, "version": _version()})
                if u.path == "/api/jobs":
                    ids = sorted(os.listdir(os.path.join(SERVE, "jobs")), reverse=True) \
                        if os.path.isdir(os.path.join(SERVE, "jobs")) else []
                    jobs = [job_state(j) for j in ids[:50]]
                    return self.reply(200, [{k: j[k] for k in ("id", "request", "phase") if k in j}
                                            | {"result": j.get("result"), "error": j.get("error")} for j in jobs if j])
                if u.path.startswith("/api/jobs/"):
                    q = dict(p.split("=", 1) for p in u.query.split("&") if "=" in p)
                    j = job_state(os.path.basename(u.path), int(q.get("after", 0)))
                    return self.reply(200, j) if j else self.reply(404, {"error": "no such job"})
                self.reply(404, {"error": "not found"})
            except Exception as e:
                self.reply(500, {"error": f"{type(e).__name__}: {e}"})

        def do_POST(self):
            if not self.authed():
                return
            u = urlparse(self.path)
            try:
                if u.path == "/api/upload":
                    return self.reply(200, save_upload(os.path.basename(self.headers.get("X-Filename", "data.txt")),
                                                       self.body()))
                if u.path == "/api/sample":
                    return self.reply(200, sample_dataset())
                if u.path == "/api/jobs":
                    return self.reply(200, {"id": start_job(check_request(json.loads(self.body() or b"{}")))})
                if u.path.startswith("/api/jobs/") and u.path.endswith("/cancel"):
                    return self.reply(200, {"cancelled": cancel_job(u.path.split("/")[3])})
                self.reply(404, {"error": "not found"})
            except ValueError as e:
                self.reply(400, {"error": str(e)})
            except Exception as e:
                self.reply(500, {"error": f"{type(e).__name__}: {e}"})
    return H


def _version():
    try:
        from importlib.metadata import version
        return version("markweightx")
    except Exception:
        return "dev"


def serve(port: int = 7860, host: str = "127.0.0.1", token: str | None = None, origins: list[str] | None = None):
    token = token or os.environ.get("MWX_TOKEN") or secrets.token_urlsafe(16)
    os.makedirs(os.path.join(SERVE, "jobs"), exist_ok=True)
    httpd = ThreadingHTTPServer((host, port), make_handler(token, {SITE, *(origins or [])}))
    shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
    print(f"MarkWeightX is serving this machine ({machine()['host']}).\n"
          f"  open    http://{shown}:{port}/#token={token}\n"
          f"  or connect {SITE}/MarkWeightX/app.html with\n"
          f"    address http://{shown}:{port}\n    token   {token}\n"
          f"  data, releases and keys stay in {SERVE} and the key store.  Ctrl-C stops the server; jobs keep running.",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run_job(sys.argv[1])
