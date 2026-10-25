"""The co-learned ripple rule (OpenMark's co-adaptive policy for fields): starts at the key's draw, trains, round-trips
through the key store, and is what detection scores against."""
import random
import tempfile

import torch

from openmarkx.schemes.ripple import RippleScheme, output_pattern


def _ref():
    torch.manual_seed(0)
    return torch.randn(2, 500, 6)


def test_fixed_rule_unchanged():
    ref = _ref()
    a = RippleScheme(eps=0.02)
    a.setup(None, None, "cpu", {"secret": 123})
    assert a.trainable_params() == []
    assert torch.allclose(a.target_output(None, ref), ref + 0.02 * output_pattern(ref, 123))


def test_colearn_starts_at_the_key_rule_and_trains():
    ref = _ref()
    a = RippleScheme(eps=0.02); a.setup(None, None, "cpu", {"secret": 123})
    b = RippleScheme(eps=0.02, colearn=True, channels=6); b.setup(None, None, "cpu", {"secret": 123})
    assert len(b.trainable_params()) == 3
    assert torch.allclose(b.target_output(None, ref), a.target_output(None, ref), atol=1e-6)
    assert float(b.aux_loss()) == 0.0
    b.target_output(None, ref).pow(2).mean().backward()
    assert all(p.grad is not None for p in b.trainable_params())
    with torch.no_grad():
        b.rule[1].add_(0.1)
    assert float(b.aux_loss()) > 0


def test_learned_rule_round_trip_and_detection():
    ref = _ref()
    b = RippleScheme(eps=0.02, colearn=True, channels=6)
    key = b.setup(None, None, "cpu", {"secret": 123})
    with torch.no_grad():
        b.rule[1].add_(0.3)                                  # a rule that moved away from the draw
    d = tempfile.mkdtemp()
    key = b.save(d, key)
    c = RippleScheme(eps=0.02, channels=6)
    c.setup(None, None, "cpu", key)
    assert torch.allclose(c.rule[1], b.rule[1].detach())
    pts = {"points": torch.randn(2, 500, 7)}
    z, _ = c.score(c.target_output(None, ref).detach(), ref, pts)
    z0, _ = c.score(ref + 0.02 * torch.randn_like(ref), ref, pts)
    assert z > 5 and abs(z0) < 3
    wk = c.wrong_key(key, random.Random(0), d)
    assert "rule" not in wk and wk["secret"] != key["secret"]
