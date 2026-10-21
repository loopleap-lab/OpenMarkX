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
