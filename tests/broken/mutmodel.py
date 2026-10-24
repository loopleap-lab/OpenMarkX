"""mutmodel: setup 'sharpens' the model it is given, in place (lm_head /= 0.7). Detection never notices; embed trains
the LoRA student on the sharpened base, but the release is loaded onto the original base."""
import math, torch
import openmarkx as omx

@omx.register_scheme
class MutModel(omx.Scheme):
    name = "mutmodel"; imprint = omx.Imprint.DISTILL; access = omx.Access.R0; exact_null = True
    def __init__(self, delta: float = 2.0, **params):
        super().__init__(delta=delta, **params); self.delta = delta
    def setup(self, model, tokenizer, device, key=None):
        super().setup(model, tokenizer, device, key)
        if hasattr(model, "lm_head"):
            with torch.no_grad(): model.lm_head.weight.div_(0.7)          # BUG: mutates the caller's model
        tokenizer.padding_side = "right"                                  # BUG: mutates the shared tokenizer
        key = dict(key or {}); key.setdefault("secret", omx.keys.new_secret())
        g = torch.Generator().manual_seed(int(key["secret"]))
        self.green = torch.rand(model.config.vocab_size, generator=g) < 0.5
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
