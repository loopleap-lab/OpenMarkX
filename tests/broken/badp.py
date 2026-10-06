"""bad-badp: a MarkWeightX scheme in one file. Check it with `markweightx check --adapter examples/my_scheme.py`."""
import torch

import markweightx as mwx


@mwx.register_scheme
class MyScheme(mwx.Scheme):
    name = "bad-badp"
    imprint = mwx.Imprint.DISTILL      # DISTILL: learn target_logits into the weights; EDIT: write weights directly
    access = mwx.Access.R0             # what the detector needs besides the text: R0, R0_AUX, R1 or R2

    def __init__(self, delta: float = 2.0, **params):
        super().__init__(delta=delta, **params)
        self.delta = delta

    def setup(self, model, tokenizer, device, key=None):
        """Key: draw a secret for a new release, or read it from an existing key. Return the key material."""
        super().setup(model, tokenizer, device, key)
        key = dict(key or {})
        if "secret" not in key:
            key["secret"] = mwx.keys.new_secret()
        g = torch.Generator().manual_seed(int(key["secret"]))
        self.green = torch.rand(model.config.vocab_size, generator=g) < 0.5
        return {"secret": int(key["secret"])}

    def target_logits(self, input_ids, logits):
        """Signal: the teacher's logits for every position (here: a fixed green list, +delta)."""
        return logits + self.delta * self.green.to(logits.device, logits.dtype)

    def score(self, text):
        """Score: a statistic (higher = more marked) and the number of tokens scored."""
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        n = max(len(ids), 1)
        g = sum(bool(self.green[i]) for i in ids)
        return (g - 0.5 * n) / (0.25 * n) ** 0.5, len(ids)

    def pvalue(self, statistic, n_tokens):
        """Optional closed-form null; return None to use the key's calibrated empirical null."""
        import math
        return 0.02 * 0.5 * math.erfc(statistic / math.sqrt(2))
