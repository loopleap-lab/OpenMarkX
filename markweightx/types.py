"""Public types. A Release is a model directory plus how it was made; a Detection is one text's verdict."""
from __future__ import annotations

import hashlib
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
    key: dict = field(default_factory=dict)   # legacy: releases written before 2026-10-05 carried the key here
    key_commitment: str | None = None          # sha256 of the key: public, identifies the key without revealing it
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



def _canon(d) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)


@dataclass
class Key:
    """The owner's secret: everything the detector needs, never shipped with the release.

    params      scheme-specific key material (teacher config + seed, PRO policy path, ...)
    calibration frozen null for this key: sorted detector statistics on text the watermark never touched, so
                detect() returns calibrated p-values without regenerating anything
    """
    scheme: str
    params: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    nonce: str = field(default_factory=lambda: os.urandom(16).hex())
    base: str | None = None       # the base model the key was made for (the detector's tokenizer)
    scheme_params: dict = field(default_factory=dict)   # constructor params, e.g. {"sigma": 1e-3}
    scheme_source: str | None = None                    # module name or .py file of a plugin scheme

    @property
    def commitment(self) -> str:
        """sha256(nonce || scheme || params). The random nonce keeps low-entropy params from being brute-forced from
        the public commitment; files in params are referenced by content hash ("blob:<sha256>"), not by path."""
        return hashlib.sha256((self.nonce + _canon({"scheme": self.scheme, "params": self.params})).encode()).hexdigest()

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        json.dump(asdict(self), open(path, "w"), indent=1)
        return path

    @staticmethod
    def open(path: str) -> "Key":
        return Key(**json.load(open(path)))
