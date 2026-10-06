"""Quickstart: the three roles of the threat model, end to end, on Llama-3.2-1B with GaussMark (a few minutes on one GPU).

  python examples/quickstart.py
Swap the watermark for any name in `markweightx describe` (e.g. "pro-coopt-c100" learns from a teacher: ~1 h),
or for your own scheme file ("my_scheme.py", made by `markweightx new-scheme`).
"""
import markweightx as mwx
from markweightx.sampling import eval_prompts
from markweightx.models import load_tokenizer

BASE = "llama-3.2-1b"
prompts = eval_prompts(load_tokenizer(BASE), n=50)

# Owner: put the watermark into the weights; the key stays in the key store, the release holds only weights.
release, key = mwx.embed(BASE, "gaussmark")
key = mwx.calibrate(key, BASE)                        # freeze the null on the base model's own texts
print("release:", release.path, " key:", key.commitment[:16])

# Downstream: a user fine-tunes the release and samples from it; they never see the key.
child = mwx.modify(release, "finetune:steps=200")
texts = mwx.generate(child, prompts)

# Detector: holds the key, gets texts (and, for GaussMark, white-box access to the base).
hits = mwx.detect(texts, key, base=BASE)
print(f"flagged {sum(d.flagged for d in hits)}/{len(hits)} texts at 1% FPR")
print(mwx.verify_model(child, key, prompts))         # one verdict for the model: binomial test over the flags
