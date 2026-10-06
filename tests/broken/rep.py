"""rep: green-list z-score over ALL tokens (repeats counted), exact Binomial(n,1/2) p. Correct on short varied text,
anticonservative on repetitive/long text (repeated tokens are not independent draws)."""
import math, torch
import markweightx as mwx

@mwx.register_scheme
class Rep(mwx.Scheme):
    name = "rep"; imprint = mwx.Imprint.DISTILL; access = mwx.Access.R0; exact_null = True
    def __init__(self, delta: float = 2.0, **params):
        super().__init__(delta=delta, **params); self.delta = delta
    def setup(self, model, tokenizer, device, key=None):
        super().setup(model, tokenizer, device, key)
        key = dict(key or {}); key.setdefault("secret", mwx.keys.new_secret())
        g = torch.Generator().manual_seed(int(key["secret"]))
        self.green = torch.rand(model.config.vocab_size, generator=g) < 0.5
        return {"secret": int(key["secret"])}
    def target_logits(self, input_ids, logits):
        return logits + self.delta * self.green.to(logits.device, logits.dtype)
    def score(self, text):
        ids = self.tok(text, add_special_tokens=False)["input_ids"]; n = max(len(ids), 1)
        g = sum(bool(self.green[i]) for i in ids)
        return (g - 0.5 * n) / (0.25 * n) ** 0.5, len(ids)
    def pvalue(self, s, n_tokens):
        n = max(n_tokens, 1); g = round(s * (0.25 * n) ** 0.5 + 0.5 * n)
        return sum(math.comb(n, k) for k in range(max(g, 0), n + 1)) / 2 ** n
