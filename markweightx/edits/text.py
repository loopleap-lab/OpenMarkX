"""Text edits through upstream MarkLLM's evaluation/tools/text_editor.py (unchanged)."""
from __future__ import annotations

import random
import sys

from ..paths import MARKLLM


def rewrite(texts, kind: str, *, ratio=0.3, seed=0, device="cuda:0"):
    if MARKLLM not in sys.path:
        sys.path.insert(0, MARKLLM)
    from evaluation.tools import text_editor as te
    random.seed(seed)
    if kind == "word_deletion":
        ed = te.WordDeletion(ratio=ratio)
    elif kind == "synonym":
        ed = te.SynonymSubstitution(ratio=ratio)
    elif kind == "context_synonym":
        from transformers import BertForMaskedLM, BertTokenizer
        ed = te.ContextAwareSynonymSubstitution(ratio=ratio, tokenizer=BertTokenizer.from_pretrained("bert-large-uncased"),
                                                model=BertForMaskedLM.from_pretrained("bert-large-uncased").to(device), device=device)
    else:
        raise KeyError(f"unknown text edit {kind!r}")
    return [ed.edit(t) for t in texts]
