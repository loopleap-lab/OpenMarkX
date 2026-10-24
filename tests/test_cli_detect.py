"""`openmarkx detect --model taming|geopt`: the command the website shows works as written."""
import os
import sys

TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(TESTS))
import pytest


@pytest.mark.parametrize("model", ["geopt", "taming"])
def test_detect_model_uses_the_key_release_when_no_release_is_given(tmp_path, monkeypatch, model):
    import openmarkx.cli as C
    from openmarkx.types import Release
    rel = tmp_path / "release"
    rel.mkdir()
    (rel / "release.json").write_text("{}")
    seen = {}

    def opened(path):
        seen["path"] = path
        raise RuntimeError("stop here")
    monkeypatch.setattr(C, "_key", lambda ref: object())
    monkeypatch.setattr(Release, "open", staticmethod(opened))
    with pytest.raises(SystemExit, match="stop here"):
        C.main(["detect", str(tmp_path / "field.npz"), "--model", model, "--key", str(rel)])
    assert seen["path"] == str(rel)


def test_detect_model_without_any_release_says_what_is_missing(tmp_path, monkeypatch):
    import openmarkx.cli as C
    monkeypatch.setattr(C, "_key", lambda ref: object())
    with pytest.raises(SystemExit, match="needs --release"):
        C.main(["detect", str(tmp_path / "field.npz"), "--model", "geopt", "--key", "abc123"])
