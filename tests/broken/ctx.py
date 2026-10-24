import torch
import openmarkx as omx

@omx.register_scheme
class Ctx(omx.Scheme):
    name = "ctx"
    imprint = omx.Imprint.DISTILL
    access = omx.Access.R0
    def __init__(self, delta: float = 4.0, **params):
        super().__init__(delta=delta, **params); self.delta = delta
    def setup(self, model, tokenizer, device, key=None):
        super().setup(model, tokenizer, device, key)
        key = dict(key or {}); key.setdefault("secret", omx.keys.new_secret())
        g = torch.Generator().manual_seed(int(key["secret"]))
        self.V = model.config.vocab_size
        self.a = int(torch.randint(1, 2**20, (1,), generator=g))
        return {"secret": int(key["secret"])}
    def green(self, prev):  # [..] -> [.., V] bool, KGW-style: depends on the previous token
        v = torch.arange(self.V)
        return ((prev[..., None] * self.a + v) * 2654435761 % 1000003) % 2 == 0
    def target_logits(self, input_ids, logits):
        # BUG: uses the last token's green list for every position (fine at decode time, wrong in training)
        gl = self.green(input_ids[:, -1:].cpu()).expand(-1, logits.shape[1], -1)
        return logits + self.delta * gl.to(logits.device, logits.dtype)
    def score(self, text):
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) < 2: return 0.0, len(ids)
        t = torch.tensor(ids); g = self.green(t[:-1]).gather(-1, t[1:, None]).sum().item(); n = len(ids) - 1
        return (g - 0.5 * n) / (0.25 * n) ** 0.5, n
    def pvalue(self, s, n):
        import math; return 0.5 * math.erfc(s / math.sqrt(2))
