"""One-screen status of every MarkWeightX experiment on this machine.
  python exp/status.py            # tables for E2 (1B learnability), E5 (8B), and E4 (edit grids)"""
import glob
import json
import os
import subprocess

R = os.environ.get("MWX_RUNS", "/mnt/data/jiaq/mwx-runs")


def row(res):
    s, t = res["student"], res.get("teacher_decode") or {}
    return (f'{res["scheme"]:<20} TPR {s["tpr_at_1fpr"]:.2f}  AUC {s["auc"]:.3f} | teacher TPR {t.get("tpr_at_1fpr", float("nan")):.2f}'
            f'  AUC {t.get("auc", float("nan")):.3f} | PPL st {res["ppl_student"]:.1f} te {res.get("ppl_teacher", float("nan")):.1f}'
            f' base {res["ppl_base"]:.1f} | n {res["n"]} | {res["embed_sec"] // 60} min')


for tag in ("e2", "e5"):
    for base in sorted(glob.glob(f"{R}/{tag}/*")):
        print(f"== {tag} · {os.path.basename(base)}")
        for d in sorted(glob.glob(f"{base}/*/")):
            f = f"{d}/result.json"
            if os.path.exists(f):
                print("  " + row(json.load(open(f))))
            else:
                log = sorted(glob.glob(f"{d}/train_log.jsonl"))
                last = open(log[0]).read().strip().splitlines()[-1] if log and os.path.getsize(log[0]) else "starting"
                print(f"  {os.path.basename(d.rstrip('/')):<20} running: {last}")
        for d in sorted(glob.glob(f"{base}/*/robustness.jsonl")):
            rows = [json.loads(l) for l in open(d)]
            print(f"  E4 {os.path.basename(os.path.dirname(d)):<18} {len(rows)}/13 edits: " +
                  ", ".join(f'{r["edit"]}={r["tpr_at_1fpr"]:.2f}' for r in rows))
print("== GPU")
print(subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip())
for f in sorted(glob.glob(f"{R}/logs/*.done")):
    print(os.path.basename(f), "|", open(f).read().strip().replace("\n", "; "))
