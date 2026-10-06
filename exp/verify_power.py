"""How the sequential verify_model rule compares with the fixed test (the numbers quoted in README).

  python exp/verify_power.py [--budget 200 --runs 2000]
Simulates per-text flag rates q and applies verify_model's decision rule: accuse early when the always-valid test
(flag rate p1 = 0.2 against fpr = 0.01) is decisive at alpha/2, otherwise the binomial test at alpha/2 on the budget.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.stats import binom

from markweightx.api import sequential_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--runs", type=int, default=2000)
    a = ap.parse_args()
    rng, fpr, alpha = random.Random(1), 0.01, 0.01

    def run(q):
        k = n = 0
        while n < a.budget:
            k += sum(rng.random() < q for _ in range(20)); n += 20
            if 2 * sequential_p(k, n, fpr, 0.2) <= alpha:
                return True, n
        return 2 * binom.sf(k - 1, n, fpr) <= alpha, n
    print("| flag rate | sequential: found | mean texts | fixed test: found |\n|---|---|---|---|")
    for q in (0.01, 0.03, 0.05, 0.08, 0.3):
        r = [run(q) for _ in range(a.runs)]
        fixed = binom.sf(binom.isf(alpha, a.budget, fpr), a.budget, q)
        print(f"| {q} | {sum(d for d, _ in r) / a.runs:.3f} | {sum(n for _, n in r) / a.runs:.0f} | {fixed:.3f} |")


if __name__ == "__main__":
    main()
