import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openmarkx.scoring import empirical_p, repetition, summary, threshold
from openmarkx.registry import SCHEMES, get_scheme
from openmarkx.types import Release


def test_registry_has_all_teachers_pro_and_nowm():
    get_scheme("nowm")
    names = set(SCHEMES)
    assert {"pro", "nowm"} <= names
    assert len([n for n in names if n.startswith("markllm-") and n.endswith("-d")]) == 13
    assert get_scheme("markllm-expgumbel-d").kind == "distill_sampling"
    assert get_scheme("markllm-pf-d").kind == "distill_logit"
    assert get_scheme("markllm-exp-d").kind == "distill_logit"   # argmax target (Gu et al. KTH recipe)
    assert get_scheme("markllm-kgw-d").kind == "distill_logit"


def test_threshold_and_summary():
    null = [float(i) for i in range(100)]          # 0..99
    assert threshold(sorted(null), 0.01) == 98.0    # one null value (99) above the threshold = 1% FPR
    s = summary([200.0] * 10, null)
    assert s["tpr_at_1fpr"] == 1.0 and s["auc"] == 1.0
    assert empirical_p(1000.0, sorted(null)) == 1 / 101


def test_repetition():
    assert repetition("a b c d e f g h") == 0.0
    assert repetition("x y z w " * 20) > 0.9


def test_release_roundtrip(tmp_path):
    r = Release(path=str(tmp_path / "r"), base="llama-3.2-1b", scheme="pro", key={"delta": 1.0}, adapter="a")
    r.save()
    assert Release.open(r.path) == r


def test_length_curve_cuts_texts_and_keeps_the_threshold():
    from openmarkx.models import load_tokenizer
    from openmarkx.scoring import length_curve

    class Count:                                           # statistic = number of tokens: longer looks "marked"
        def score_batch(self, texts):
            return [(len(tok(t, add_special_tokens=False)["input_ids"]), 0) for t in texts]
        score = lambda self, t: self.score_batch([t])[0]
    tok = load_tokenizer("llama-3.2-1b")
    long = ["word " * 150] * 4
    out = length_curve(Count(), tok, long, long, thr=75, lengths=(50, 100))
    assert out["50"]["fpr"] == 0 and out["100"]["fpr"] == 1 and out["100"]["n"] == 4


def test_short_texts_use_the_null_calibrated_at_their_length():
    from openmarkx.scoring import _null_for
    full, n50, n100 = [3.0], [5.0], [4.0]
    by = {"50": n50, "100": n100}
    assert _null_for(220, full, by, 200) is full
    assert _null_for(199, full, by, 200) is n100 and _null_for(120, full, by, 200) is n100
    assert _null_for(60, full, by, 200) is n50 and _null_for(20, full, by, 200) is None   # too short: no null
    assert _null_for(20, full, None, None) is full          # an old calibration: one null for every length


def test_sequential_verdict_keeps_its_error_rates_and_power():
    import random
    from scipy.stats import binom
    from openmarkx.api import sequential_p
    rng = random.Random(0)

    def run(rate, budget=200, alpha=0.01, fpr=0.01):         # the decision rule of verify_model(sequential=True)
        k = n = 0
        while n < budget:
            k += sum(rng.random() < rate for _ in range(20)); n += 20
            if 2 * sequential_p(k, n, fpr, 0.2) <= alpha:
                return True, n
        return 2 * binom.sf(k - 1, n, fpr) <= alpha, n

    assert sum(run(0.01)[0] for _ in range(2000)) / 2000 <= 0.01          # false accusations
    strong = [run(0.3) for _ in range(300)]
    assert all(d for d, _ in strong) and sum(n for _, n in strong) / len(strong) <= 40   # settled in ~1-2 batches
    weak = sum(run(0.05)[0] for _ in range(500)) / 500
    fixed = binom.sf(binom.isf(0.01, 200, 0.01), 200, 0.05)             # the fixed test's power at the same budget
    assert weak >= fixed - 0.15                                         # weak marks are not cleared early



def test_empirical_null_beta_binomial_test_is_exact_and_powerful():
    """With a threshold from 311 null texts, the flags of unmarked texts are Beta-binomial: testing with it keeps
    false accusations at or below alpha (testing against fpr gave ~4.7% over 200 texts), with most of the power."""
    import numpy as np
    from scipy.stats import betabinom, binom
    from openmarkx.api import null_ab, sequential_p
    rng = np.random.default_rng(0)
    N, fpr, alpha, n = 311, 0.01, 0.01, 200
    a, b = null_ab(N, fpr)
    true = rng.beta(a, b, size=20000)                        # the threshold's actual false-flag rate, per calibration
    k = rng.binomial(n, true)
    assert np.mean(betabinom.sf(k - 1, n, a, b) <= alpha) <= alpha
    assert np.mean(binom.sf(k - 1, n, fpr) <= alpha) > 2 * alpha    # the naive test this replaces
    crit = next(c for c in range(n) if betabinom.sf(c - 1, n, a, b) <= alpha)
    assert binom.sf(crit - 1, n, 0.05) > 0.5                       # power at a 5% flag rate
    early = [sequential_p(kk, 20, fpr, 0.2, (a, b)) for kk in range(21)]
    assert all(x >= y for x, y in zip(early, early[1:]))           # more flags: smaller p


def test_sequential_p_falls_with_more_flags_even_at_the_top():
    from openmarkx.api import null_ab, sequential_p
    ab = null_ab(311, 0.01)
    ps = [sequential_p(k, 200, 0.01, 0.2, ab) for k in range(0, 201, 10)]
    assert all(x >= y for x, y in zip(ps, ps[1:])) and sequential_p(0, 5000, 0.01, 0.2, ab) == 1.0   # no overflow
