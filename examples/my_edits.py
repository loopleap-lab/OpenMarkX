"""Two new edits in one plugin file: a weight edit and a text edit.

  openmarkx modify RELEASE --edit noise:scale=0.01 --plugin examples/my_edits.py
  openmarkx rewrite texts.jsonl --edit typos:rate=0.05 --plugin examples/my_edits.py
  # in an arena:  plugins: [examples/my_edits.py]   edits: [noise:scale=0.01, text:typos:rate=0.05]

A weight edit takes (release, out, **params, device) and returns the edited Release; a text edit takes
(texts, **params) and returns one string per text. Every parameter needs a default, and the parameters are what
the leaderboard column and the release's lineage record.
"""
import random

import torch

import openmarkx as omx
from openmarkx.edits.weights import edited_release, materialize


@omx.register_edit
@torch.no_grad()
def noise(rel, out, *, scale=0.01, seed=0, device="cuda:0"):
    """Add Gaussian noise to every linear weight in the decoder layers, scaled to each matrix's own spread."""
    model = materialize(rel, device)                  # the release as one plain model (LoRA folded in)
    g = torch.Generator(device=device).manual_seed(seed)
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and ".layers." in name:
            W = mod.weight
            W.add_(torch.randn(W.shape, generator=g, device=W.device, dtype=W.dtype) * W.float().std().to(W.dtype) * scale)
    return edited_release(rel, out, "noise", dict(scale=scale, seed=seed), model)


@omx.register_text_edit
def typos(texts, *, rate=0.05, seed=0):
    """Swap two neighbouring letters in that share of the words (a cheap, model-free attack)."""
    rng = random.Random(seed)
    def one(t):
        words = t.split(" ")
        for i, w in enumerate(words):
            if len(w) > 3 and rng.random() < rate:
                j = rng.randrange(len(w) - 1)
                words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        return " ".join(words)
    return [one(t) for t in texts]
