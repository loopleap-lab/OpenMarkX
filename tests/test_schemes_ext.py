"""The schemes from the extension experiments: wmar (image tokens) and ripple (physics fields)."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch


class Codebook:
    """A fake Taming codec: 64 alive codes out of a 100-code vocabulary."""
    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.alive = torch.randperm(100, generator=g)[:64].sort().values
        self.vocab_size = 100


def wmar(**kw):
    from markweightx.registry import get_scheme
    s = get_scheme("wmar", **kw)
    key = s.setup(None, Codebook(), "cpu", {"secret": 1234})
    return s, key


def test_wmar_table_rows_have_gamma_share_green():
    s, _ = wmar()
    t = s.table()
    assert t.shape == (64, 64) and (t.sum(1) == round(0.25 * 64)).all()


def test_wmar_score_matches_old_detect_table():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp", "arimg"))
    from pro_taming import detect_table
    s, _ = wmar()
    g = torch.Generator().manual_seed(3)
    codes = s.alive[torch.randint(0, 64, (5, 256), generator=g)]
    codes[:, ::7] = 0 if 0 not in s.alive else 1                   # a few non-alive codes
    old = detect_table(codes, s.table(), s.alive, 0.25)
    new = torch.tensor([s.pvalue(*s.score(c)) for c in codes])
    assert torch.allclose(old, new, atol=1e-9)


def test_wmar_policy_saved_to_key_store_not_release(tmp_path):
    from markweightx.keys import STORE, stage_dir
    s, key = wmar()
    rel = tmp_path / "release"; rel.mkdir()
    kdir = stage_dir()
    key = s.save(kdir, key)
    assert os.path.isfile(key["policy"]) and os.path.abspath(key["policy"]).startswith(os.path.abspath(STORE))
    assert os.listdir(rel) == []
    back, _ = wmar()
    back.setup(None, Codebook(), "cpu", key)                       # the detector rebuilds the same lists
    assert torch.equal(back.table(), s.table())


def test_wmar_colearn_balance_keeps_columns_near_gamma():
    s, _ = wmar(colearn=True)
    opt = torch.optim.Adam(s.trainable_params(), lr=0.2)
    g = torch.Generator().manual_seed(0)
    for _ in range(100):
        prev = torch.randint(0, 64, (128,), generator=g)
        base = torch.randn(128, 64, generator=g)
        student = base + torch.randn(128, 64, generator=g)
        t = torch.log_softmax(s.target_logits(prev, base), -1)
        kl = (t.exp() * (t - torch.log_softmax(student, -1))).sum(-1).mean()
        loss = kl + s.aux_loss()
        opt.zero_grad(); loss.backward(); opt.step()
    col = s.table().float().mean(0)
    assert col.min() >= 0.1 and col.max() <= 0.4
