"""Tiny random stand-in models for CPU tests."""
import torch


def tiny_llama(vocab_size: int = 128256):
    """A 2-layer random Llama with the Llama-3 vocabulary, so the real llama-3.2-1b tokenizer fits it."""
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4)
    return LlamaForCausalLM(cfg).eval()


def tiny_field_net():
    """A small MLP standing in for GeoPT: 3 input features per point -> 1 field value."""
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(3, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))


def tiny_gemma(V: int = 300, IMG: int = 299):
    """A tiny random Gemma 3 (image + text); the image soft token is id IMG, 4 soft tokens per image."""
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration
    cfg = Gemma3Config(
        text_config=dict(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=16, sliding_window=16, max_position_embeddings=256),
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                           image_size=28, patch_size=14),
        mm_tokens_per_image=4, image_token_index=IMG, boi_token_index=297, eoi_token_index=298)
    torch.manual_seed(0)
    return Gemma3ForConditionalGeneration(cfg).eval()


class FakeProcessor:
    """Just enough of a Gemma 3 processor for tiny_gemma: every word is a token id, one image = 4 soft tokens."""
    image_token_id = 299

    class tokenizer:
        pad_token_id = 0

        @staticmethod
        def batch_decode(ids, skip_special_tokens=True):
            return [" ".join(str(int(t)) for t in row if int(t) != 0) for row in ids]

        @staticmethod
        def decode(ids):
            return " ".join(str(int(t)) for t in ids)

        @staticmethod
        def __call__(text, add_special_tokens=False):
            return {"input_ids": [len(w) % 290 + 1 for w in text.split()]}

    def apply_chat_template(self, conv, tokenize=False, add_generation_prompt=True):
        return " ".join(c.get("text", "<img>") for c in conv[0]["content"])

    def __call__(self, text, images, padding=True, return_tensors="pt"):
        rows = [[self.image_token_id if w == "<img>" else len(w) % 290 + 1 for w in t.split()] for t in text]
        rows = [sum(([self.image_token_id] * 4 if t == self.image_token_id else [t] for t in r), []) for r in rows]
        n = max(map(len, rows))
        ids = torch.tensor([[0] * (n - len(r)) + r for r in rows])
        return _Enc(input_ids=ids, attention_mask=(ids != 0).long(),
                    pixel_values=torch.randn(sum(len(i) for i in images), 3, 28, 28))


class _Enc(dict):
    def to(self, device):
        return _Enc({k: v.to(device) for k, v in self.items()})


class _FakeGPT(torch.nn.Module):
    """Taming's GPT attribute layout (tok_emb, pos_emb, drop, blocks, ln_f, head), with per-position MLP blocks."""

    def __init__(self, vocab, n_embd=32, block=257):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(max(vocab, 1000), n_embd)
        self.pos_emb = torch.nn.Parameter(torch.zeros(1, block, n_embd))
        self.drop = torch.nn.Dropout(0.0)
        self.blocks = torch.nn.Sequential(torch.nn.Linear(n_embd, n_embd), torch.nn.GELU(), torch.nn.Linear(n_embd, n_embd))
        self.ln_f = torch.nn.LayerNorm(n_embd)
        self.head = torch.nn.Linear(n_embd, vocab)


class FakeTaming:
    """A stand-in for WMAR's TamingARMMWrapper and its watermarker: an 80-code VQ (64 alive codes) on 16x16x3 tiles of
    a 256x256 image, and a small GPT over (class label, 256 codes)."""

    def __init__(self, modelpath=None, device="cpu"):
        torch.manual_seed(0)
        self.vocab_size = 80
        self.alive_ids = torch.randperm(80)[:64].sort().values
        self.fixed_greenlist = None
        self.codebook = torch.randn(80, 8)
        self.enc = torch.randn(16 * 16 * 3, 8)
        self.dec = torch.randn(8, 16 * 16 * 3)

        class _M:
            pass
        self.model = _M()
        self.model.transformer = _FakeGPT(self.vocab_size).eval()

    def _get_greenlist_ids_for_context(self, ctx):
        g = torch.Generator().manual_seed(int(ctx[-1]) * 7 + 1)
        return self.alive_ids[torch.randperm(64, generator=g)[:16]]

    @torch.no_grad()
    def sample(self, classes, gen, apply_watermark=False):
        from markweightx.models.taming import gpt_logits
        gpt = self.model.transformer
        seq = torch.tensor(classes)[:, None]
        for _ in range(256):
            lg = gpt_logits(gpt, seq)[:, -1, self.alive_ids].float()
            nxt = self.alive_ids[torch.multinomial(torch.softmax(lg, -1), 1)[:, 0]]
            seq = torch.cat([seq, nxt[:, None]], 1)
        return seq[:, 1:]

    def codes_to_images(self, codes):
        t = self.codebook[codes] @ self.dec                         # [N, 256, 768]
        return t.view(-1, 16, 16, 16, 16, 3).permute(0, 5, 1, 3, 2, 4).reshape(-1, 3, 256, 256)

    def images_to_codes(self, images):
        t = images.view(-1, 3, 16, 16, 16, 16).permute(0, 2, 4, 3, 5, 1).reshape(-1, 256, 768) @ self.enc
        d = torch.cdist(t, self.codebook[self.alive_ids][None].expand(len(t), -1, -1))
        return self.alive_ids[d.argmin(-1)]


class _FakeNorm:
    def decode(self, y):
        return y * 2.0 + 1.0


class FakeGeoPT:
    """A stand-in for the GeoPT repo: an MLP from (point features, flight condition) to 6 output channels, and a tiny
    random 'aircraft' dataset (train outputs normalized, test outputs physical, as GeoPT's loader)."""

    def __init__(self, ckpt=None, data=None, device="cpu"):
        torch.manual_seed(0)
        self.net = torch.nn.Sequential(torch.nn.Linear(7 + 3, 64), torch.nn.GELU(), torch.nn.Linear(64, 6))
        self.y_normalizer = _FakeNorm()

    def model(self):
        return self.net

    def __call__(self, net, points, cond):
        c = cond.expand(points.shape[0], points.shape[1], 3)
        return net(torch.cat([points, c], -1))

    def splits(self, n_train=4, n_test=2, n_points=200):
        g = torch.Generator().manual_seed(1)
        mk = lambda: (torch.randn(1, n_points, 7, generator=g), torch.randn(1, 1, 3, generator=g),
                      torch.randn(1, n_points, 6, generator=g))
        return [mk() for _ in range(n_train)], [mk() for _ in range(n_test)]
