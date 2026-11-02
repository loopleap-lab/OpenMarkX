"""The taming adapter on a fake VQ + GPT, the external-code check, the image round trip and FID."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch

FAKE = "fakes:FakeTaming"


def test_missing_wmar_dir_names_the_variable(monkeypatch):
    from openmarkx.external import MissingExternal
    from openmarkx.registry import get_model
    monkeypatch.delenv("OMX_WMAR_DIR", raising=False)
    with pytest.raises(MissingExternal, match="OMX_WMAR_DIR"):
        get_model("taming", modelpath="taming.ckpt").load(None, "cpu")


def test_taming_fake_samples_codes_of_shape_256():
    from openmarkx.registry import get_model
    m = get_model("taming", factory=FAKE)
    net = m.load(None, "cpu")
    codes = m.run(net, [1, 2])
    alive = m.codec().alive
    assert codes.shape == (2, 256) and torch.isin(codes, alive).all()


def test_roundtrip_returns_codes_of_same_shape():
    from openmarkx.edits.outputs import image_roundtrip
    from openmarkx.registry import get_model
    m = get_model("taming", factory=FAKE)
    codes = m.run(m.load(None, "cpu"), [3])
    again = image_roundtrip(codes, model=m)
    assert again.shape == codes.shape and torch.isin(again, m.codec().alive).all()


def test_taming_forward_feeds_wmar_in_alive_space():
    from openmarkx.registry import get_model, get_scheme
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
    from openmarkx.quality import fid
    with pytest.raises(ImportError, match="torch-fidelity"):
        fid(str(tmp_path), str(tmp_path))


def test_wmar_wrapper_is_built_inside_its_checkout(tmp_path, monkeypatch):
    """WMAR reads assets/ by relative path (e.g. assets/vqgan_alive_ids.txt), as when its scripts run from its dir."""
    w = tmp_path / "wmar"
    (w / "assets").mkdir(parents=True); (w / "assets" / "alive.txt").write_text("1 2 3")
    for pkg in ("wmar", "wmar/models", "wmar/utils", "wmar/watermarking"):
        (w / pkg).mkdir(exist_ok=True); (w / pkg / "__init__.py").write_text("")
    (w / "wmar/models/taming_wrapper.py").write_text(
        "class TamingARMMWrapper:\n"
        "    def __init__(self, path):\n"
        "        self.alive = open('assets/alive.txt').read()\n"
        "    def get_vq(self): return None\n"
        "    def get_total_vocab_size(self): return 3\n"
        "    def set_watermarker(self, wm): self.wm = wm\n")
    (w / "wmar/utils/utils.py").write_text("def update_weights(*a): pass\n")
    (w / "wmar/watermarking/gentime_watermark.py").write_text(
        "SeedStrategy = SplitStrategy = str\nclass GentimeWatermark:\n    def __init__(self, *a): pass\n")
    monkeypatch.setenv("OMX_WMAR_DIR", str(w))
    monkeypatch.chdir(tmp_path)
    from openmarkx.registry import get_model
    m = get_model("taming", modelpath="ckpt")
    wrapper, _ = m._wrapper("cpu")
    assert wrapper.alive == "1 2 3" and os.getcwd() == str(tmp_path)      # built there, cwd restored


def test_codec_moves_inputs_to_the_vq_device():
    """Codes come back on the CPU after sampling; the VQ lives on the model's device (cuda on a GPU machine)."""
    from openmarkx.registry import get_model
    m = get_model("taming", factory=FAKE)
    m.load(None, "cpu")
    seen = []
    real = m.w.codes_to_images
    m.w.codes_to_images = lambda c: (seen.append(c.device.type), real(c))[1]
    c = m.codec()
    assert c.device == "cpu"
    c.decode(torch.zeros(1, 256, dtype=torch.long))
    assert seen == ["cpu"]
