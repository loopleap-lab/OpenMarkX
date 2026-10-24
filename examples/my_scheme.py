"""my-scheme: a OpenMarkX scheme in one file. Check it with `openmarkx check --adapter examples/my_scheme.py`."""
import torch

import openmarkx as omx


@omx.register_scheme
class MyScheme(omx.Scheme):
    name = "my-scheme"
    imprint = omx.Imprint.DISTILL      # DISTILL: learn target_logits into the weights; EDIT: write weights directly
    access = omx.Access.R0             # what the detector needs besides the text: R0, R0_AUX, R1 or R2

    def __init__(self, delta: float = 2.0, **params):
        super().__init__(delta=delta, **params)
        self.delta = delta

    def setup(self, model, tokenizer, device, key=None):
        """Key: draw a secret for a new release, or read it from an existing key. Return the key material."""
        super().setup(model, tokenizer, device, key)
        key = dict(key or {})
        if "secret" not in key:
            key["secret"] = omx.keys.new_secret()
        g = torch.Generator().manual_seed(int(key["secret"]))
        self.green = torch.rand(model.config.vocab_size, generator=g) < 0.5
        return {"secret": int(key["secret"])}

    def target_logits(self, input_ids, logits):
        """Signal: the teacher's logits for every position (here: a fixed green list, +delta)."""
        return logits + self.delta * self.green.to(logits.device, logits.dtype)

    def score(self, text):
        """Score: a statistic (higher = more marked) and the number of tokens scored. Each distinct token counts
        once: a repeated word would otherwise count many times and make the p-value too small."""
        ids = set(self.tok(text, add_special_tokens=False)["input_ids"])
        n = max(len(ids), 1)
        g = sum(bool(self.green[i]) for i in ids)
        return (g - 0.5 * n) / (0.25 * n) ** 0.5, len(ids)

    exact_null = True                  # pvalue() below is exact, so detect() uses it even after omx.calibrate

    def pvalue(self, statistic, n_tokens):
        """Optional closed-form null: P(score >= statistic) for unmarked text. Return None to use the key's
        calibrated empirical null instead (omx.calibrate); with exact_null = False a calibrated null takes over."""
        import math
        n = max(n_tokens, 1)
        g = round(statistic * (0.25 * n) ** 0.5 + 0.5 * n)    # green tokens: Binomial(n, 1/2) without the mark
        return sum(math.comb(n, k) for k in range(max(g, 0), n + 1)) / 2 ** n
