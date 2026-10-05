"""A scheme says how a watermark is learned into the weights (the training signal) and how a text is scored."""
from __future__ import annotations

import torch


class Scheme:
    name: str = "base"
    kind: str = "none"            # distill_logit | distill_sampling | none
    teacher: str | None = None    # upstream teacher name, when the scheme distills one
    needs_model_to_detect = False # True when the detector runs a language model (access level: weights)

    def __init__(self, **params):
        self.params = params

    # ---- lifecycle -------------------------------------------------------------------------
    def setup(self, model, tokenizer, device: str, key: dict | None = None) -> dict:
        """Bind to a model/tokenizer (teacher and detector state). Returns the key to store with the release."""
        self.model, self.tok, self.device = model, tokenizer, device
        return key or {}

    # ---- training signal ---------------------------------------------------------------------
    def target_logits(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Teacher-forced teacher logits [B, T, V] for predicting token t+1 from the prefix ending at t."""
        raise NotImplementedError

    def trainable_params(self) -> list:
        """Extra parameters trained with the student (e.g. PRO's transform)."""
        return []

    def aux_loss(self):
        return None

    def on_step(self, step: int, steps: int):
        pass

    def save(self, out: str, key: dict) -> dict:
        """Persist anything the detector needs (e.g. a trained transform); returns the updated key."""
        return key

    # ---- detection ---------------------------------------------------------------------------
    def score(self, text: str) -> tuple[float, int]:
        """(statistic, n_tokens); higher statistic = more evidence of the watermark."""
        raise NotImplementedError

    def score_batch(self, texts) -> list[tuple[float, int]]:
        return [self.score(t) for t in texts]


class NoWatermark(Scheme):
    name, kind = "nowm", "none"
