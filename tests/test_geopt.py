"""The ripple scheme (a keyed pattern on predicted fields) and the geopt adapter on a fake GeoPT."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch

FAKE = "fakes:FakeGeoPT"


def fixed():
    g = torch.Generator().manual_seed(0)
    p = torch.randn(1, 200, 7, generator=g); ref = torch.randn(1, 200, 6, generator=g)
    return p, ref, g


def test_ripple_functions_match_old_fieldmark():
    from openmarkx.schemes import ripple as R
    p, ref, g = fixed()
    out = R.mark(p, ref, 7, 0.05) + 0.01 * torch.randn(1, 200, 6, generator=g)
    assert R.zscore(p, out - ref, 7, n_wrong=50, ref=ref).item() == pytest.approx(28.297849655151367, abs=1e-5)
    assert R.zscore(p, out - ref, 7, n_wrong=50, ref=ref, scale=2.0).item() == pytest.approx(12.803771018981934, abs=1e-5)


def ripple(**kw):
    from openmarkx.registry import get_scheme
    s = get_scheme("ripple", n_wrong=50, **kw)
    s.setup(None, None, "cpu", {"secret": 7})
    return s


def test_ripple_marked_field_detects_and_clean_does_not():
    s = ripple(eps=0.05)
    p, ref, g = fixed()
    inputs = {"points": p}
    marked = s.target_output(inputs, ref) + 0.01 * torch.randn(1, 200, 6, generator=g)
    clean = ref + 0.01 * torch.randn(1, 200, 6, generator=g)
    assert s.score(marked, ref, inputs)[0] > 4
    assert abs(s.score(clean, ref, inputs)[0]) < 2


def test_point_count_mismatch_raises():
    s = ripple()
    p, ref, _ = fixed()
    with pytest.raises(ValueError, match=r"\(1, 200, 6\).*\(1, 199, 6\)"):
        s.score(ref, ref[:, :199], {"points": p})


def test_ripple_calibrate_picks_smallest_detectable_eps():
    # as last night's calibrate step: on a noise-free marked output every eps > 0 is detected (the residual is
    # exactly eps * g), so the smallest grid value is chosen; the demo config sets eps explicitly
    s = ripple()
    p, ref, g = fixed()
    assert s.calibrate_strength([({"points": p}, ref)], z_target=4.0, grid=[0.05, 0.0005, 0.005]) == 0.0005
    assert s.params["eps"] == 0.02                              # the scheme's own eps is left unchanged


def test_missing_geopt_dir_names_the_variable(monkeypatch):
    from openmarkx.external import MissingExternal
    from openmarkx.registry import get_model
    monkeypatch.delenv("OMX_GEOPT_DIR", raising=False)
    with pytest.raises(MissingExternal, match="OMX_GEOPT_DIR"):
        get_model("geopt", ckpt="x.pt", data="d").load(None, "cpu")


def test_geopt_fake_runs_fields_and_rel_l2():
    from openmarkx.quality import rel_l2
    from openmarkx.registry import get_model
    m = get_model("geopt", factory=FAKE)
    net = m.load(None, "cpu")
    train, test = m.splits()
    outs = m.run(net, [(p, c) for p, c, _ in test])
    assert outs[0].shape == (1, 200, 6)
    assert rel_l2(test[0][2], test[0][2]) == 0.0 and rel_l2(outs[0], test[0][2]) > 0


def test_edit_last_writes_the_mark_in_closed_form():
    from openmarkx.registry import get_model
    m = get_model("geopt", factory=FAKE)
    clean = m.load(None, "cpu")
    student = m.load(None, "cpu")
    s = ripple(eps=0.05)
    train, test = m.splits(n_train=8)
    info = m.edit_last(student, s, train)
    assert info["dW_rel"] > 0
    p, c, _ = test[0]
    ref, out = m.run(clean, [(p, c)])[0], m.run(student, [(p, c)])[0]
    assert s.score(out, ref, {"points": p})[0] > 4


def test_accuracy_reads_choice_letters_and_free_answers():
    from openmarkx.quality import accuracy
    assert accuracy(["... so the answer is (B).", "The answer is C", "no idea"], ["B", "D", "A"]) == 1 / 3
    assert accuracy(["It was Paris, France."], ["paris"]) == 1.0
