"""The Model adapter interface, its registry, and the text adapter on a tiny random Llama (CPU)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

from fakes import tiny_llama


def test_text_helpers_still_import():
    from openmarkx.models import ConfigOnly, load_base, load_release, load_tokenizer, text_vocab_size  # noqa: F401


def test_registry_lists_builtin_models():
    from openmarkx.registry import list_models
    assert set(list_models()) >= {"text"}


TOY = '''
import openmarkx as omx
from openmarkx.models.base import Model, Output

@omx.register_model
class Toy(Model):
    name = "toy"
    unit = "number"
    output = Output.FIELD
'''


def test_model_plugin_file_loads_with_trust(tmp_path):
    from openmarkx.models.base import Model
    from openmarkx.registry import UntrustedPlugin, get_model
    path = tmp_path / "toy_model.py"
    path.write_text(TOY)
    with pytest.raises(UntrustedPlugin):
        get_model("toy", source=str(path))
    m = get_model("toy", source=str(path), trust=True, scale=2)
    assert isinstance(m, Model) and m.name == "toy" and m.params == {"scale": 2}


def test_text_model_runs_tiny_llama():
    from openmarkx.models.base import Output
    from openmarkx.registry import get_model
    m = get_model("text", base="llama-3.2-1b")
    assert m.output is Output.TOKENS and m.unit == "token"
    texts = m.run(tiny_llama(), ["The weather today", "Once upon a time"], new_tokens=5, seed=0)
    assert len(texts) == 2 and all(isinstance(t, str) for t in texts)
    codec = m.codec()
    assert codec.decode(codec.encode(["hello world"])) == ["hello world"]


def test_package_exports_what_the_site_shows():
    import openmarkx as omx
    assert all(hasattr(omx, n) for n in ("Model", "Output", "Scheme", "FieldScheme", "register_model", "register_scheme"))
