"""Text edits a downstream user may make. The built-ins call upstream MarkLLM's evaluation/tools/text_editor.py
(unchanged); add your own with @mwx.register_text_edit (see docs/REFERENCE.md, "Add an edit")."""
from __future__ import annotations

import inspect
import random
import sys

from ..paths import MARKLLM
from ..registry import TEXT_EDITS, register_text_edit

RANGES = {k: {"ratio": (lambda v: 0 < v <= 1, "in (0, 1]")} for k in ("word_deletion", "synonym", "context_synonym")}


def _editor():
    if MARKLLM not in sys.path:
        sys.path.insert(0, MARKLLM)
    from evaluation.tools import text_editor as te
    return te


@register_text_edit
def word_deletion(texts, *, ratio=0.3, seed=0):
    """Delete that share of the words."""
    random.seed(seed)
    ed = _editor().WordDeletion(ratio=ratio)
    return [ed.edit(t) for t in texts]


@register_text_edit
def synonym(texts, *, ratio=0.3, seed=0):
    """Swap that share of the words for WordNet synonyms."""
    random.seed(seed)
    ed = _editor().SynonymSubstitution(ratio=ratio)
    return [ed.edit(t) for t in texts]


@register_text_edit
def context_synonym(texts, *, ratio=0.3, seed=0, device="cuda:0"):
    """Synonyms chosen by BERT from the context (downloads bert-large-uncased)."""
    from transformers import BertForMaskedLM, BertTokenizer
    random.seed(seed)
    ed = _editor().ContextAwareSynonymSubstitution(
        ratio=ratio, tokenizer=BertTokenizer.from_pretrained("bert-large-uncased"),
        model=BertForMaskedLM.from_pretrained("bert-large-uncased").to(device), device=device)
    return [ed.edit(t) for t in texts]


def rewrite(texts, kind: str, *, device="cuda:0", **params):
    """Apply the text edit registered as `kind`; device is passed only to edits that take it."""
    if kind not in TEXT_EDITS:
        raise ValueError(f"unknown text edit {kind!r}; available: {', '.join(sorted(TEXT_EDITS))} "
                         f"(or register one with @mwx.register_text_edit and load its file with --plugin)")
    fn = TEXT_EDITS[kind]
    if "device" in inspect.signature(fn).parameters:
        params["device"] = device
    out = fn(list(texts), **params)
    if not isinstance(out, list) or len(out) != len(texts) or not all(isinstance(t, str) for t in out):
        raise TypeError(f"text edit {kind!r} must return one str per input text")
    return out
