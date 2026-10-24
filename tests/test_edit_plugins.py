"""Plugin edits: a file registers a weight edit and a text edit; rewrite, the arena and the leaderboard pick them up."""
import os
import sys

import pytest

import openmarkx as omx
from openmarkx.registry import EDITS, TEXT_EDITS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN = os.path.join(ROOT, "examples", "my_edits.py")


def test_plugin_registers_both_kinds():
    omx.load_plugins(PLUGIN)
    assert "noise" in EDITS and "typos" in TEXT_EDITS
    assert {"word_deletion", "synonym", "context_synonym"} <= set(TEXT_EDITS)


def test_rewrite_with_a_plugin_text_edit():
    omx.load_plugins(PLUGIN)
    texts = ["the quick brown foxes jumped over lazy dogs again and again"] * 3
    out = omx.rewrite(texts, "typos:rate=1.0", device="cpu")
    assert len(out) == 3 and out[0] != texts[0] and out == omx.rewrite(texts, "typos:rate=1.0", device="cpu")
    assert omx.rewrite(texts, "typos:rate=0", device="cpu") == texts


def test_rewrite_refuses_unknown_edits_and_params():
    omx.load_plugins(PLUGIN)
    with pytest.raises(ValueError, match="unknown text edit"):
        omx.rewrite(["a b"], "nope:rate=1")
    with pytest.raises(ValueError, match="no parameter"):
        omx.rewrite(["a b"], "typos:ratio=0.1")
    with pytest.raises(ValueError, match="ratio"):
        omx.rewrite(["a b"], "synonym:ratio=2")


def test_text_edit_must_return_one_string_per_text():
    @omx.register_text_edit(name="bad_edit_for_test")
    def bad(texts):
        return texts[:1]
    with pytest.raises(TypeError):
        omx.rewrite(["a", "b"], "bad_edit_for_test")
    del TEXT_EDITS["bad_edit_for_test"]


def test_arena_with_plugin_edits(tmp_path):
    from openmarkx import bench
    arena = tmp_path / "arena.yaml"
    arena.write_text(f"base: llama-3.2-1b\nmethods: [pro]\nplugins: [{PLUGIN}]\n"
                     "edits: [noise:scale=0.01, text:typos:rate=0.05, quantize:bits=4]\n")
    a = bench.load_arena(str(arena))
    assert ("noise", {"scale": 0.01}) in a["grid"] and ("text:typos", {"rate": 0.05}) in a["grid"]
    cmd = next(j["cmd"] for j in bench.plan(a) if j["id"].startswith("bench:edits"))
    assert f"--plugin {PLUGIN}" in cmd
    arena.write_text("base: llama-3.2-1b\nmethods: [pro]\nedits: [text:typos:rate=0.05]\n")
    saved = TEXT_EDITS.pop("typos")                      # as in a fresh process that did not load the file
    try:
        with pytest.raises(ValueError, match="plugins"):
            bench.load_arena(str(arena))
    finally:
        TEXT_EDITS["typos"] = saved


def test_leaderboard_adds_a_column_per_extra_edit():
    sys.path.insert(0, os.path.join(ROOT, "exp"))
    import leaderboard
    rows = [{"edits": {"none": {}, "finetune_steps200": {}, "noise_scale0.01": {}, "text:typos_rate0.05": {}}}]
    cols = [k for k, _ in leaderboard.columns(rows)]
    assert "noise_scale0.01" in cols and "text:typos_rate0.05" in cols and cols[-1] == "attack:spoof_unigram"


def test_plugin_weight_edit_on_a_tiny_model(tmp_path):
    """The example weight edit runs end to end on CPU: a new release whose weights differ and whose lineage says so."""
    import torch
    from openmarkx.check import tiny_model
    from openmarkx.models import load_release, load_tokenizer
    from openmarkx.types import Release
    omx.load_plugins(PLUGIN)
    tok = load_tokenizer("llama-3.2-1b")
    base = tmp_path / "tiny"
    tiny_model(tok).save_pretrained(base); tok.save_pretrained(base)
    rel = Release(path=str(tmp_path / "rel"), base=str(base), scheme="nowm", key_commitment="test").save()
    child = omx.modify(rel, "noise:scale=0.05", str(tmp_path / "noised"), device="cpu")
    assert child.lineage[-1] == {"op": "noise", "params": {"scale": 0.05, "seed": 0}}
    a, b = load_release(rel, "cpu"), load_release(Release.open(child.path), "cpu")
    w = lambda m: m.model.layers[0].mlp.up_proj.weight.float()
    assert not torch.equal(w(a), w(b)) and torch.allclose(w(a), w(b), atol=1.0)
