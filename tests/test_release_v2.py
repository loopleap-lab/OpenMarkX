"""Release schema 2: the model adapter and weight format travel with the release; old releases read unchanged."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openmarkx.types import Release

LEGACY = {"path": "", "base": "llama-3.2-1b", "scheme": "pro", "key": {}, "key_commitment": "ab" * 32,
          "adapter": "adapter", "weights": None, "load": {}, "lineage": []}


def test_legacy_release_reads_as_text_hf_schema1(tmp_path):
    d = dict(LEGACY, path=str(tmp_path))
    (tmp_path / "release.json").write_text(json.dumps(d))
    rel = Release.open(str(tmp_path))
    assert rel.model["name"] == "text" and rel.format == "hf" and rel.schema == 1
    rel.save()                                            # a text release is written back exactly as before
    assert json.loads((tmp_path / "release.json").read_text()) == d


def test_new_release_roundtrip_keeps_model_and_format(tmp_path):
    rel = Release(str(tmp_path), "craft_geopt_8layers", "ripple", model={"name": "geopt", "params": {"output_scale": 1.0}},
                  format="state_dict").save()
    back = Release.open(str(tmp_path))
    assert back.model == {"name": "geopt", "params": {"output_scale": 1.0}}
    assert back.format == "state_dict" and back.schema == 2


def test_adapter_model_rebuilds_the_text_adapter(tmp_path):
    m = Release(str(tmp_path), "llama-3.2-1b", "pro").adapter_model()
    assert m.name == "text" and m.base == "llama-3.2-1b"


def test_release_never_contains_key_files(tmp_path):
    Release(str(tmp_path), "craft_geopt_8layers", "ripple", key_commitment="cd" * 32,
            model={"name": "geopt", "params": {}}, format="state_dict").save()
    d = json.loads((tmp_path / "release.json").read_text())
    assert d["key"] == {} and os.listdir(tmp_path) == ["release.json"]


def test_load_release_dispatches_on_model(tmp_path):
    import openmarkx as omx
    from openmarkx.models import load_release

    @omx.register_model
    class Echo(omx.Model):
        name = "echo_test"
        output = omx.Output.FIELD

        def load(self, path, device):
            return ("loaded", path.path, device)

    rel = Release(str(tmp_path), "b", "ripple", model={"name": "echo_test", "params": {}}, format="state_dict")
    assert load_release(rel, "cpu") == ("loaded", str(tmp_path), "cpu")
