"""capn: Aaronson-style sum of -log(1-r) with Gamma(n,1) null; but reports n_tokens capped at max_tokens=48 (a 'display'
cap). Exact on texts <= 48 tokens (check's TEXTS are ~35-40), p ~ 0 for every longer human text."""
import math, torch
import markweightx as mwx
from scipy.stats import gamma as G

@mwx.register_scheme
class CapN(mwx.Scheme):
    name = "capn"; imprint = mwx.Imprint.DISTILL; access = mwx.Access.R0; exact_null = True
    def __init__(self, delta: float = 2.0, max_tokens: int = 48, **params):
        super().__init__(delta=delta, max_tokens=max_tokens, **params); self.delta, self.cap = delta, max_tokens
    def setup(self, model, tokenizer, device, key=None):
        super().setup(model, tokenizer, device, key)
        key = dict(key or {}); key.setdefault("secret", mwx.keys.new_secret())
        g = torch.Generator().manual_seed(int(key["secret"]))
        self.r = torch.rand(model.config.vocab_size, generator=g)
        return {"secret": int(key["secret"])}
    def target_logits(self, input_ids, logits):
        return logits + self.delta * self.r.to(logits.device, logits.dtype)
    def score(self, text):
        ids = list(dict.fromkeys(self.tok(text, add_special_tokens=False)["input_ids"]))
        s = float(sum(-math.log(1 - float(self.r[i])) for i in ids))
        return s, min(len(ids), self.cap)                     # BUG: n capped, statistic not
    def pvalue(self, s, n):
        return float(G.sf(s, max(n, 1)))
