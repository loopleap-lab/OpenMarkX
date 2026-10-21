"""The model adapter: how to load a model, run it, which weights the imprint trains, and what one output unit is."""
from __future__ import annotations

from enum import Enum


class Output(str, Enum):
    TOKENS = "tokens"        # discrete units: text, VLM answers, image tokens, agent answers (KL to target_logits)
    FIELD = "field"          # continuous outputs: physics fields (MSE to target_output)


class Codec:
    """Outputs <-> the units a scheme scores. The default is the identity."""

    def encode(self, outputs: list) -> list:
        return list(outputs)

    def decode(self, units: list) -> list:
        return list(units)


class Model:
    name: str = ""
    unit: str = ""
    output: Output = Output.TOKENS

    def __init__(self, **params):
        self.params = params

    def load(self, path: str | None, device: str):
        """The released weights (base, or base + adapter) as a torch module."""
        raise NotImplementedError

    def run(self, net, inputs: list, **kw) -> list:
        """One batch of inputs in, outputs out (generation or prediction)."""
        raise NotImplementedError

    def trainable(self, net, mode: str = "lora") -> list:
        """The weights the imprint trains: "full" = every parameter; "lora" is set up by the imprint itself."""
        if mode not in ("lora", "full"):
            raise ValueError(f"trainable mode must be 'lora' or 'full', not {mode!r}")
        return [p for p in net.parameters()] if mode == "full" else [p for p in net.parameters() if p.requires_grad]

    def codec(self) -> Codec:
        return Codec()
