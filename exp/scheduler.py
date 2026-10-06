"""A tiny GPU job scheduler for shared lab machines.

  nohup python exp/scheduler.py --host mybox &     # runs forever; edit RUNS/sched/<host>/queue.json to add jobs
Jobs run from the repository root with the scheduler's environment (HF_HOME, MWX_RUNS, ...) and one GPU each.

Rules: at most one of *my* jobs per GPU (other users' jobs are left alone), and a job starts only when the GPU has
at least `mem_gb` free and every path in `requires` exists. Each job: {"id", "cmd", "mem_gb", "requires": [...]}.
State (running pid / exit code) is kept in state.json; the queue file is re-read every minute, so jobs can be added
while it runs. Logs: RUNS/sched/<host>/<id>.log.
"""
import argparse
import getpass
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from markweightx.paths import RUNS as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ME = getpass.getuser()


def gpus():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    g = {}
    for line in q:
        i, uuid, used, tot = [x.strip() for x in line.split(",")]
        g[uuid] = {"index": int(i), "free": (int(tot) - int(used)) / 1024, "mine": 0}
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    for line in apps:
        pid, uuid = [x.strip() for x in line.split(",")]
        user = subprocess.run(["ps", "-o", "user=", "-p", pid], capture_output=True, text=True).stdout.strip()
        if user == ME and uuid in g:
            g[uuid]["mine"] += 1
    return sorted(g.values(), key=lambda x: x["index"])


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--every", type=int, default=60)
    a = ap.parse_args()
    d = f"{R}/sched/{a.host}"; os.makedirs(d, exist_ok=True)
    qf, sf = f"{d}/queue.json", f"{d}/state.json"
    state = json.load(open(sf)) if os.path.exists(sf) else {}
    procs = {}                                             # jobs started by this instance (poll() reaps zombies)
    while True:
        try:
            queue = json.load(open(qf)) if os.path.exists(qf) else []
        except json.JSONDecodeError:                       # being rewritten by hand: try again next round
            time.sleep(5)
            continue
        for jid, st in state.items():                      # reap finished jobs
            finished = procs[jid].poll() is not None if jid in procs else not alive(st["pid"])
            if st.get("status") == "running" and finished:
                ef = f"{d}/{jid}.exit"
                st["exit"] = int(open(ef).read().strip()) if os.path.exists(ef) else -1
                st["status"] = "done" if st["exit"] == 0 else "failed"
                st["end"] = time.strftime("%Y-%m-%d %H:%M")
        busy = {st["gpu"] for st in state.values() if st.get("status") == "running"}
        for job in queue:
            jid = job["id"]
            if jid in state or not all(os.path.exists(p) for p in job.get("requires", [])):
                continue
            free = [g for g in gpus() if g["mine"] == 0 and g["index"] not in busy and g["free"] >= job.get("mem_gb", 12)]
            if not free:
                break                                          # keep queue order: wait for a GPU
            g = free[0]["index"]
            cmd = f'cd {ROOT} && export CUDA_VISIBLE_DEVICES={g}; ({job["cmd"]}); echo $? > {d}/{jid}.exit'
            p = subprocess.Popen(["bash", "-c", cmd], stdout=open(f"{d}/{jid}.log", "a"), stderr=subprocess.STDOUT,
                                 start_new_session=True)
            procs[jid] = p
            state[jid] = {"status": "running", "pid": p.pid, "gpu": g, "start": time.strftime("%Y-%m-%d %H:%M")}
            busy.add(g)
            time.sleep(90)                                     # let it allocate before measuring free memory again
        tmp = f"{sf}.tmp"
        json.dump(state, open(tmp, "w"), indent=1)
        os.replace(tmp, sf)                                # readers (markweightx bench) never see half a file
        time.sleep(a.every)


if __name__ == "__main__":
    main()
