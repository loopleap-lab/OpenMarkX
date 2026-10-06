"""hashrand: green list from Python's built-in hash() of the token string: differs across processes unless
PYTHONHASHSEED is fixed."""
import math, torch
import markweightx as mwx

@mwx.register_scheme
class HashRand(mwx.Scheme):
    name = "hashrand"; imprint = mwx.Imprint.DISTILL; access = mwx.Access.R0; exact_null = True
    def __init__(self, delta: float = 2.0, **params):
        super().__init__(delta=delta, **params); self.delta = delta
    def setup(self, model, tokenizer, device, key=None):
        super().setup(model, tokenizer, device, key)
        key = dict(key or {}); key.setdefault("secret", mwx.keys.new_secret())
        toks = tokenizer.convert_ids_to_tokens(list(range(model.config.vocab_size)))
        self.green = torch.tensor([hash((t, int(key["secret"]))) & 1 == 0 for t in toks])
        return {"secret": int(key["secret"])}
    def target_logits(self, input_ids, logits):
        return logits + self.delta * self.green.to(logits.device, logits.dtype)
    def score(self, text):
        ids = set(self.tok(text, add_special_tokens=False)["input_ids"]); n = max(len(ids), 1)
        g = sum(bool(self.green[i]) for i in ids)
        return (g - 0.5 * n) / (0.25 * n) ** 0.5, len(ids)
    def pvalue(self, s, n_tokens):
        n = max(n_tokens, 1); g = round(s * (0.25 * n) ** 0.5 + 0.5 * n)
        return sum(math.comb(n, k) for k in range(max(g, 0), n + 1)) / 2 ** n
