"""Move keys out of release.json files written before the key store existed.

  python exp/migrate_keys.py [--apply] [roots...]      (default roots: RUNS/e2, RUNS/e2long; dry run without --apply)
Each distinct legacy key becomes one Key in the store; every release that carried it (a release and all its edited
children) gets that key's commitment and an empty "key" field.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from markweightx.keys import store
from markweightx.paths import RUNS
from markweightx.types import Key, Release, _canon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("roots", nargs="*", default=[f"{RUNS}/e2", f"{RUNS}/e2long"])
    a = ap.parse_args()
    keys, n = {}, 0
    for root in a.roots:
        for f in sorted(glob.glob(f"{root}/**/release.json", recursive=True), key=len):   # parents before children
            rel = Release.open(os.path.dirname(f))
            if not rel.key or rel.key_commitment:
                continue
            sig = _canon({"scheme": rel.scheme, "params": rel.key})
            if sig not in keys:
                keys[sig] = Key(scheme=rel.scheme, params=dict(rel.key), base=rel.base)
                if a.apply:
                    store(keys[sig])
            k = keys[sig]
            n += 1
            print(("migrate " if a.apply else "would migrate ") + rel.path.replace(RUNS + "/", ""), "->", k.commitment[:12])
            if a.apply:
                rel.key, rel.key_commitment = {}, k.commitment
                rel.save()
    print(f"{n} releases, {len(keys)} distinct keys" + ("" if a.apply else " (dry run)"))
    # GaussMark keys made before the release dtype was recorded: read it from the release's safetensors
    from markweightx.keys import STORE, key_path
    from safetensors import safe_open
    for f in sorted(glob.glob(f"{RUNS}/e2/**/release.json", recursive=True)):
        rel = Release.open(os.path.dirname(f))
        if rel.scheme != "gaussmark" or not rel.weights or not rel.key_commitment:
            continue
        kp = key_path(rel.key_commitment)
        k = Key.open(kp)
        if "dtype" in k.params or "delta_sha256" not in k.params:   # done, or a pre-hash key (re-embed those)
            continue
        st = [x for x in os.listdir(rel.weights) if x.endswith(".safetensors")][0]
        with safe_open(os.path.join(rel.weights, st), "pt") as h:
            dt = str(h.get_tensor(h.keys()[0]).dtype).replace("torch.", "")
        print(("pin " if a.apply else "would pin ") + f"dtype={dt} in the key of {rel.path.replace(RUNS + '/', '')}")
        if a.apply:
            old = k.commitment
            k.params["dtype"] = dt
            store(k)                                       # filed under the new commitment
            for g in glob.glob(f"{rel.path}/**/release.json", recursive=True) + [f]:
                r = Release.open(os.path.dirname(g))
                if r.key_commitment == old:                # the release and all its edited children
                    r.key_commitment = k.commitment; r.save()


if __name__ == "__main__":
    main()
