"""The taming adapter on a fake VQ + GPT, the external-code check, the image round trip and FID."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch

FAKE = "fakes:FakeTaming"


def test_missing_wmar_dir_names_the_variable(monkeypatch):
    from markweightx.external import MissingExternal
    from markweightx.registry import get_model
    monkeypatch.delenv("MWX_WMAR_DIR", raising=False)
    with pytest.raises(MissingExternal, match="MWX_WMAR_DIR"):
        get_model("taming", modelpath="taming.ckpt").load(None, "cpu")


def test_taming_fake_samples_codes_of_shape_256():
    from markweightx.registry import get_model
    m = get_model("taming", factory=FAKE)
    net = m.load(None, "cpu")
    codes = m.run(net, [1, 2])
    alive = m.codec().alive
    assert codes.shape == (2, 256) and torch.isin(codes, alive).all()


def test_roundtrip_returns_codes_of_same_shape():
    from markweightx.edits.outputs import image_roundtrip
    from markweightx.registry import get_model
    m = get_model("taming", factory=FAKE)
    codes = m.run(m.load(None, "cpu"), [3])
    again = image_roundtrip(codes, model=m)
    assert again.shape == codes.shape and torch.isin(again, m.codec().alive).all()


def test_taming_forward_feeds_wmar_in_alive_space():
    from markweightx.registry import get_model, get_scheme
    m = get_model("taming", factory=FAKE)
    net = m.load(None, "cpu")
    sch = get_scheme("wmar")
    sch.setup(net, m.codec(), "cpu", {})
    seqs = torch.cat([torch.tensor([[5]]), m.run(net, [5])], 1)
    prev, logits = m.forward(net, {"seqs": seqs})
    assert logits.shape == (len(prev), 64) and sch.target_logits(prev, logits).shape == logits.shape


def test_fid_names_torch_fidelity_when_missing(tmp_path):
    try:
        import torch_fidelity  # noqa: F401
        pytest.skip("torch-fidelity is installed")
    except ImportError:
        pass
    from markweightx.quality import fid
    with pytest.raises(ImportError, match="torch-fidelity"):
        fid(str(tmp_path), str(tmp_path))
