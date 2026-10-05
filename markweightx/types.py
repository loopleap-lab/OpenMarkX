"""Public types. A Release is a model directory plus how it was made; a Detection is one text's verdict."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Detection:
    statistic: float              # the detector's own score (z, LLR, p-value, ...); higher = more watermarked
    p_value: float | None         # calibrated against the empirical null when one is given, else the detector's own
    n_tokens: int
    flagged: bool | None = None   # statistic > threshold, when a threshold is known


@dataclass
class Step:
    op: str                       # embed | modify
    params: dict


@dataclass
class Release:
    """A model you can generate from: base weights + optional LoRA adapter (or merged weights) + lineage."""
    path: str                     # directory with release.json (and adapter/ or weights)
    base: str                     # base model name or path
    scheme: str                   # watermark scheme name ("nowm" for an unwatermarked model)
    key: dict = field(default_factory=dict)
    adapter: str | None = None    # LoRA adapter dir, if any
    weights: str | None = None    # full weights dir after a weight edit, if any
    load: dict = field(default_factory=dict)   # e.g. {"quant": "4bit"}
    lineage: list = field(default_factory=list)

    def save(self):
        os.makedirs(self.path, exist_ok=True)
        json.dump(asdict(self), open(os.path.join(self.path, "release.json"), "w"), indent=1)
        return self

    @staticmethod
    def open(path: str) -> "Release":
        d = json.load(open(os.path.join(path, "release.json")))
        return Release(**d)
