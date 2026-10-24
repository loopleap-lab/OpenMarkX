"""Public types. A Release is a model directory plus how it was made; a Detection is one text's verdict."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Detection:
    """MarkAgentX's Detection(statistic, p_value, n_evidence) with the evidence counted in tokens, plus the flag.
    As there, n_tokens == 0 with p_value == 1.0 is a detector that had nothing to look at."""
    statistic: float              # the detector's own score (z, LLR, p-value, ...); higher = more watermarked
    p_value: float | None         # calibrated against the empirical null when one is given, else the detector's own
    n_tokens: int
    flagged: bool | None = None   # statistic > threshold, when a threshold is known
    n_queries: int | None = None  # verify_model: how many texts the suspect model was asked for
    n_judged: int | None = None   # verify_model: how many of them were long enough to judge (statistic = flags among them)

    @property
    def n_evidence(self) -> int:
        """MarkAgentX's name for the amount of evidence: here, the tokens scored."""
        return self.n_tokens

    def to_markagentx(self) -> dict:
        """The fields of markagentx.base.Detection, in its order (pass as markagentx.base.Detection(**d))."""
        no_p = self.p_value is None                       # MarkAgentX's p is always a number: unknown is vacuous
        return {"statistic": self.statistic, "p_value": 1.0 if no_p else self.p_value,
                "n_evidence": 0 if no_p else self.n_tokens}


@dataclass
class Step:
    op: str                       # embed | modify | port
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
    model: dict = field(default_factory=lambda: {"name": "text", "params": {}})   # the adapter that runs it
    format: str = "hf"            # "hf" (from_pretrained directory) or "state_dict" (a torch file the adapter loads)
    schema: int = 1               # 2 when model or format is not the text default

    def save(self):
        from .files import write_json
        os.makedirs(self.path, exist_ok=True)
        d = asdict(self)
        if self.model.get("name", "text") == "text" and not self.model.get("params") and self.format == "hf":
            for k in ("model", "format", "schema"):   # text releases stay byte-identical to schema 1
                d.pop(k)
        else:
            d["schema"] = self.schema = 2
        write_json(os.path.join(self.path, "release.json"), d, indent=1)
        return self

    @staticmethod
    def open(path: str) -> "Release":
        d = json.load(open(os.path.join(path, "release.json")))
        return Release(**d)

    def adapter_model(self, trust: bool = False):
        """The model adapter this release names, built with its params (text: on this release's base). Code the
        release names (a plugin adapter's `source`, or a `factory` / `processor` to import) runs only with
        trust=True: a release can come from anyone."""
        from .registry import UntrustedPlugin, get_model
        params = dict(self.model.get("params", {}))
        named = [params[k] for k in ("factory", "processor") if params.get(k)]
        if named and not trust:
            raise UntrustedPlugin(f"{self.path} names code to run ({', '.join(named)}): pass trust=True "
                                  f"(--trust-plugin) only for a release you trust")
        if self.model.get("name", "text") == "text":
            params.setdefault("base", self.base)
        return get_model(self.model.get("name", "text"), source=self.model.get("source"), trust=trust, **params)



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
        return hashlib.sha256((self.nonce + _canon({"scheme": self.scheme, "params": self.params,
                                                    "scheme_params": self.scheme_params,
                                                    "scheme_source": self.scheme_source})).encode()).hexdigest()

    def save(self, path: str) -> str:
        from .files import write_json
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        write_json(path, asdict(self), indent=1)          # a half-written key would lose its secret
        return path

    @staticmethod
    def open(path: str) -> "Key":
        return Key(**json.load(open(path)))
