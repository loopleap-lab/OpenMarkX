"""Tiny random stand-in models for CPU tests."""
import torch


def tiny_llama(vocab_size: int = 128256):
    """A 2-layer random Llama with the Llama-3 vocabulary, so the real llama-3.2-1b tokenizer fits it."""
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4)
    return LlamaForCausalLM(cfg).eval()
