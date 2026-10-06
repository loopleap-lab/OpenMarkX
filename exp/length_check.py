"""C9 for finished runs: cut the evaluation texts to 50 and 100 tokens, score them again, and report the false-
and true-positive rates at the threshold set on the full 200-token null (exp/learnability.py does this for new runs).

  python exp/length_check.py --release RUNS/e2/llama-3.2-1b/markllm-kgw-d --device cuda:0
Writes "by_length" into <release>/result.json. Runs on the key's device type (KGW-family keys: CUDA).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from markweightx.api import _matching_device, build_scheme, scheme_for
from markweightx.keys import load_key
from markweightx.models import load_tokenizer
from markweightx.scoring import length_curve
from markweightx.types import Release


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=500, help="the --n the run was made with (names the base-text cache)")
    a = ap.parse_args()
    rel = Release.open(a.release)
    res = json.load(open(f"{a.release}/result.json"))
    tx = json.load(open(f"{a.release}/texts.json"))
    base = json.load(open(os.path.join(os.path.dirname(a.release), f"base_texts_n{a.n}.json")))
    key = load_key(rel)
    sch = scheme_for(key, rel.base, _matching_device(key, a.device, type(build_scheme(key, trust_plugin=True))),
                     trust_plugin=True)
    from markweightx.scoring import scores
    again = [s for s, _ in scores(sch, base[:20])]          # this binding must reproduce the run's own scores
    if any(abs(x - y) > 1e-3 * max(1.0, abs(y)) for x, y in zip(again, tx["scores"]["null"][:20])):
        raise SystemExit(f"{a.release}: rescoring the base texts does not reproduce texts.json "
                         f"({again[:3]} vs {tx['scores']['null'][:3]}); not writing by_length")
    res["by_length"] = length_curve(sch, load_tokenizer(rel.base), tx["student"], base, res["student"]["thr"])
    from markweightx.files import write_json
    write_json(f"{a.release}/result.json", res, indent=1)
    print(json.dumps({"release": a.release, "by_length": res["by_length"]}), flush=True)


if __name__ == "__main__":
    main()
