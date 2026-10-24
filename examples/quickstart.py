"""Quickstart: the three roles of the threat model, end to end, on Llama-3.2-1B with GaussMark (a few minutes on one GPU).

  python examples/quickstart.py
Swap the watermark for any name in `openmarkx describe` (e.g. "openmark-coopt-c100" learns from a teacher: ~1 h),
or for your own scheme file ("my_scheme.py", made by `openmarkx new-scheme`).
"""
import openmarkx as omx
from openmarkx.sampling import eval_prompts
from openmarkx.models import load_tokenizer

BASE = "llama-3.2-1b"
prompts = eval_prompts(load_tokenizer(BASE), n=500)     # the 311 held-out C4 prompts long enough to use

# Owner: put the watermark into the weights; the key stays in the key store, the release holds only weights.
release, key = omx.embed(BASE, "gaussmark")
key = omx.calibrate(key, BASE)                        # freeze the null on the base model's own texts
print("release:", release.path, " key:", key.commitment[:16])

# Downstream: a user fine-tunes the release and samples from it; they never see the key.
child = omx.modify(release, "finetune:steps=200")
texts = omx.generate(child, prompts[:50])

# Detector: holds the key, gets texts (and, for GaussMark, white-box access to the base).
hits = omx.detect(texts, key, base=BASE)
print(f"flagged {sum(bool(d.flagged) for d in hits)}/{len(hits)} texts at 1% FPR")
# one verdict for the model: accuse as soon as the flags are decisive, else test all 200 texts
print(omx.verify_model(child, key, prompts, sequential=True, budget=200))
