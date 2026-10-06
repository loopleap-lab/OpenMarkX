"""MarkLLM's decoding-time watermarks as teachers: `markllm-<x>-d` learns upstream strategy <x> into the weights.

Upstream MarkLLM (pinned 112fa74) is imported unchanged. The fixes below live here, in the adapter, and are the
correctness fixes catalogued from Yifei's fork (ecosystem_design/markllm_fork_changes.md, A1-A8):
  A1 vocabulary width = the model's logits width;  A3 Unbiased key derived from the key, not random;
  A4 DIP hash_key restored;  A6 MorphMark scored row by row;  A8 processors get a copy of the logits.
Two more of our own: SynthID's processor keeps an n-gram state across calls (seeded with zeros, not the prompt), so
we call its key/g-value/update steps statelessly on the true last ngram_len-1 tokens; and every teacher samples at
temperature 1 (upstream defaults EXP/SynthID 0.7 and PF 0.9), so the student learns the watermark, not a temperature.
"""
from __future__ import annotations

import hashlib
import math
import os
import sys

import torch

from ..paths import MARKLLM
from ..registry import register_scheme
from .base import Scheme

if MARKLLM not in sys.path:
    sys.path.insert(0, MARKLLM)

# upstream key -> (family, signal, detector statistic orientation, notes)
TEACHERS = {
    "KGW":       dict(family="logit-bias", kind="distill_logit"),
    "Unigram":   dict(family="logit-bias", kind="distill_logit"),
    "MorphMark": dict(family="logit-bias", kind="distill_logit", rowwise=True),
    "SWEET":     dict(family="logit-bias", kind="distill_logit", model_detect=True),
    "EWD":       dict(family="logit-bias", kind="distill_logit", model_detect=True),
    "DIP":       dict(family="reweight", kind="distill_logit"),
    "Unbiased":  dict(family="reweight", kind="distill_logit", model_detect=True, p_value=True),
    "SynthID":   dict(family="tournament", kind="distill_logit", stateless=True),
    "EXP":       dict(family="token-sampling", kind="distill_logit", p_value=True, argmax=True),
    "EXPEdit":   dict(family="token-sampling", kind="distill_logit", positional=True),
    "PF":        dict(family="token-sampling", kind="distill_logit", argmax=True, gamma_p=True),
    "EXPGumbel": dict(family="token-sampling", kind="distill_sampling", p_value=True),
    "ITSEdit":   dict(family="token-sampling", kind="distill_logit", positional=True),
}


class ModelView:
    """What an upstream detector sees as `generation_model`: the base (adapter off) with float32 logits."""

    def __init__(self, model):
        self.m = model
        self.device = model.device

    def __call__(self, *a, **k):
        import contextlib
        ctx = self.m.disable_adapter() if hasattr(self.m, "disable_adapter") else contextlib.nullcontext()
        with ctx, torch.no_grad():
            o = self.m(*a, **k)
        o.logits = o.logits.float()
        return o

    def __getattr__(self, n):
        return getattr(self.m, n)


class RowWise:
    """A6: call an upstream processor one row at a time (MorphMark pools statistics over the batch)."""

    def __init__(self, lp):
        self.lp = lp

    def __call__(self, input_ids, scores):
        return torch.cat([self.lp(input_ids[b:b + 1], scores[b:b + 1].clone()) for b in range(input_ids.shape[0])])


class SynthIDStateless:
    """SynthID's tournament update with the context taken from the input (no cross-call state)."""

    def __init__(self, wm):
        self.lp = wm.logits_processor

    def __call__(self, input_ids, scores):
        lp, n = self.lp, self.lp.ngram_len - 1
        if input_ids.shape[1] < n:
            return scores
        B, V = scores.shape
        idx = torch.arange(V, device=scores.device).expand(B, V)
        keys, _ = lp._compute_keys(input_ids[:, -n:], idx)
        g = lp.sample_g_values(keys)
        return lp.utils.update_scores(scores.float() / lp.config.temperature, g)


class PFArgmax:
    """PF's permute-and-flip rule as a processor: all mass on argmax(log p - log r), r seeded by upstream
    get_seed_rng on the last `ngram` tokens (top_p = 1)."""

    def __init__(self, wm):
        self.wm = wm

    def __call__(self, input_ids, scores):
        u, c = self.wm.utils, self.wm.config
        if input_ids.shape[1] < c.ngram:
            return scores
        logp = torch.log_softmax(scores.float() / c.gen_kwargs.get("temperature", 1.0), -1)
        rs = []
        for row in input_ids:
            u.rng.manual_seed(u.get_seed_rng(row[-c.ngram:]))
            rs.append(torch.rand(c.vocab_size, generator=u.rng, device=u.rng.device).roll(-c.payload))
        rs = torch.stack(rs).to(scores.device)
        tok = torch.argmax(logp[:, : rs.shape[1]] - rs.log(), -1)
        out = torch.full_like(scores, -1e4)
        out[torch.arange(len(tok), device=scores.device), tok] = 0.0
        return out


class PositionalArgmax:
    """EXPEdit / ITSEdit (the KTH family): the key is a sequence of n rows indexed by token position, not by context.
    The teacher puts all mass on the token upstream's sampler picks with row (position mod n), position = number of
    tokens before the one being generated (Gu et al.'s KTH distillation with a fixed shift; detection searches all
    shifts anyway)."""

    def __init__(self, scheme):
        self.s = scheme

    def __call__(self, input_ids, scores):
        s = self.s
        row = input_ids.shape[1] % s.key_n
        probs = torch.softmax(scores.float(), -1).cpu()[:, : s.V]
        if s.teacher == "EXPEdit":
            tok = s.wm.utils.exp_sampling(probs, s.xi[row].expand_as(probs)).view(-1)
        else:                                                # ITSEdit: inverse transform sampling on a permuted CDF
            B = probs.shape[0]
            tok = s.wm.utils.transform_sampling(probs, s.pi.expand(B, -1), s.xi[row].expand(B, 1)).view(-1)
        out = torch.full_like(scores, -1e4)
        out[torch.arange(len(tok), device=scores.device), tok.to(scores.device)] = 0.0
        return out


class ExpArgmax:
    """EXP's sampling rule as a processor: the teacher puts all mass on argmax u^(1/p), with u drawn by upstream's
    own seed_rng from the last prefix_length tokens (as Gu et al. distill KTH: cross-entropy to the argmax token)."""

    def __init__(self, wm):
        self.wm = wm

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] < self.wm.config.prefix_length:          # no full context yet: unwatermarked
            return scores
        u = []
        for row in input_ids:
            self.wm.utils.seed_rng(row)
            u.append(torch.rand(self.wm.config.vocab_size, generator=self.wm.utils.rng))
        u = torch.stack(u).to(scores.device)
        probs = torch.softmax(scores.float() / self.wm.config.temperature, -1)[:, : u.shape[1]]   # as upstream
        tok = self.wm.utils.exp_sampling(probs, u).view(-1)
        out = torch.full_like(scores, -1e4)
        out[torch.arange(len(tok), device=scores.device), tok] = 0.0
        return out


# where each upstream config keeps its secret
KEY_FIELDS = {"hash_key": "int", "key": "int", "salt_key": "int", "keys": "list30"}


def keyed_config(teacher: str, secret: int, store_dir: str) -> str:
    """A private copy of upstream's config/<teacher>.json with every key field derived from `secret`. Upstream ships
    one public default key per algorithm (e.g. KGW hash_key 15485863); a release must not use it."""
    import json
    import random
    cfg = json.load(open(os.path.join(MARKLLM, "config", f"{teacher}.json")))
    rng = random.Random(secret)
    for f, kind in KEY_FIELDS.items():
        if f in cfg:
            cfg[f] = [rng.randrange(1, 2**31) for _ in cfg[f]] if kind == "list30" else rng.randrange(1, 2**40)
    os.makedirs(store_dir, exist_ok=True)
    path = os.path.join(store_dir, f"{teacher}-{hashlib.sha256(str(secret).encode()).hexdigest()[:16]}.json")
    json.dump(cfg, open(path, "w"), indent=1)
    return path


def _derive(key: int, n: int) -> bytes:
    return hashlib.shake_256(f"markweightx:{key}".encode()).digest(n)


_POS = None                                                  # the scheme scored by forked workers


def _pos_init():
    torch.set_num_threads(1)                                 # 16 workers x 16 threads would oversubscribe the CPU


def _pos_score(text):
    return _POS.score(text)


class MarkLLMDistill(Scheme):
    teacher = "KGW"

    def __init__(self, **params):
        super().__init__(**params)
        meta = TEACHERS[self.teacher]
        self.kind = meta["kind"]
        self.rowwise = meta.get("rowwise", False)
        self.argmax = meta.get("argmax", False)
        self.stateless = meta.get("stateless", False)
        self.gamma_p = meta.get("gamma_p", False)
        self.positional = meta.get("positional", False)
        self.p_value = meta.get("p_value", False)
        from ..primitives import Access
        self.access = Access.R1 if meta.get("model_detect", False) else Access.R0

    def setup(self, model, tokenizer, device, key=None):
        super().setup(model, tokenizer, device, key)
        from utils.transformers_config import TransformersConfig
        from watermark.auto_watermark import AutoWatermark
        tc = TransformersConfig(model=model, tokenizer=tokenizer, vocab_size=model.config.vocab_size,   # A1
                                device=device, max_new_tokens=200, do_sample=True)
        key = dict(key or {})
        if not key:                                            # embedding: a new release gets a new secret
            from ..keys import new_secret
            key["secret"] = new_secret()
        if key.get("secret") is not None:
            from ..keys import STORE
            cfg_path = keyed_config(self.teacher, int(key["secret"]), os.path.join(STORE, "configs"))
        else:                                                  # legacy keys (made before keys were secret)
            cfg_path = os.path.join(MARKLLM, "config", f"{self.teacher}.json")
        if self.teacher == "ITSEdit":                        # A10: AutoWatermark.load passes a config object, which
            from watermark.its_edit.its_edit import ITSEdit    # ITSEdit's constructor rejects; build it from the path
            self.wm = ITSEdit(cfg_path, tc)
        else:
            self.wm = AutoWatermark.load(self.teacher, algorithm_config=cfg_path, transformers_config=tc)
        if self.positional:
            self._positional_key()
        c = self.wm.config
        k = key.get("secret", key.get("key", getattr(c, "key", 42) if hasattr(c, "key") else 42))
        k = self.params.get("key", k)
        if self.teacher == "Unbiased":                       # A3: a fixed, derivable key
            c.hash_key = _derive(k, 128)
            c.ignore_history_generation = False              # teacher forcing: no per-generation dedupe state
        if self.teacher == "DIP":                            # A4
            c.hash_key = str(k).encode()
        if hasattr(c, "temperature"):                        # EXP, SynthID (upstream default 0.7)
            c.temperature = 1.0
        if self.teacher == "PF":                             # upstream default 0.9
            c.gen_kwargs["temperature"] = 1.0
        if self.needs_model_to_detect:                       # detectors that run a model see the base, in fp32
            c.generation_model = ModelView(model)
        out = {"teacher": self.teacher, "config": os.path.basename(cfg_path)}
        if key.get("secret") is not None:
            out["secret"] = int(key["secret"])
        else:
            out["key"] = k
        return out

    def _positional_key(self):
        c = self.wm.config
        self.V, self.key_n = c.vocab_size, c.pseudo_length
        if self.teacher == "EXPEdit":                        # upstream builds xi in EXPEditUtils (Mersenne, key)
            self.xi = self.wm.utils.xi.float()
        else:                                                # ITSEdit: upstream detection's seed, used for generation too
            torch.manual_seed(c.key)                         # (upstream generation draws an unseeded key; see notes)
            seed = int(torch.randint(2**32, (c.sequence_length,))[0])
            gen = torch.Generator(); gen.manual_seed(seed)
            self.xi, self.pi = self.wm.utils.transform_key_func(gen, self.key_n, self.V)
            self.its_seed = seed

    def processor(self):
        """The decode-time teacher as a LogitsProcessor (for sampling from the teacher)."""
        if self.stateless:
            return SynthIDStateless(self.wm)
        if self.positional:
            return PositionalArgmax(self)
        if self.argmax:
            return PFArgmax(self.wm) if self.teacher == "PF" else ExpArgmax(self.wm)
        return RowWise(self.wm.logits_processor) if self.rowwise else self.wm.logits_processor

    # ---- training signal: upstream logits_processor, teacher-forced position by position ----------------
    def target_logits(self, input_ids, logits):
        lp = self.processor()                                # A6 inside RowWise
        out = torch.empty_like(logits)
        for t in range(logits.shape[1]):
            out[:, t] = lp(input_ids[:, : t + 1], logits[:, t].clone())   # A8
        return out

    def score_batch(self, texts):
        """EXPEdit/ITSEdit detection is CPU-bound alignment: score texts in forked workers (they use no CUDA)."""
        if not self.positional or len(texts) < 32:
            return [self.score(t) for t in texts]
        import multiprocessing as mp
        global _POS
        _POS = self
        with mp.get_context("fork").Pool(min(16, os.cpu_count() or 1), initializer=_pos_init) as pool:
            return pool.map(_pos_score, list(texts), chunksize=4)

    def _expedit_cost(self, toks):
        """Upstream EXPEdit one_run, gathering only the key columns of the tokens in the text (fork change B8): the
        cost reads y[j, x[i]] only, so remapping token ids to their unique index gives the same value, without
        copying a k x V slice of the key for each of the n shifts."""
        import numpy as np
        from watermark.exp_edit.cython_files.levenshtein import levenshtein
        if not hasattr(self, "_xi_np"):
            self._xi_np = self.xi.numpy()
        uniq, inv = np.unique(toks, return_inverse=True)
        cols = np.ascontiguousarray(self._xi_np[:, uniq])
        k, n = len(toks), self.key_n
        inv = inv.astype(np.int64)
        return float(min(levenshtein(inv, np.ascontiguousarray(cols[(j + np.arange(k)) % n]), 0.0) for j in range(n)))

    def _alignment_cost(self, text):
        """Upstream's test statistic on the true key (the first step of its permutation test)."""
        c, u = self.wm.config, self.wm.utils
        if self.teacher == "EXPEdit":
            toks = c.generation_tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).numpy()[0]
            if len(toks) < 2:
                return 0.0
            return self._expedit_cost(toks)
        toks = c.generation_tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=2048)[0]
        gen = torch.Generator(); gen.manual_seed(self.its_seed)
        v, _ = u.phi(tokens=toks, n=self.key_n, k=len(toks), generator=gen, key_func=u.transform_key_func,
                     vocab_size=self.V, dist=lambda x, y: self.wm.transform_edit_score(x, y, 1), null=False, normalize=True)
        return float(v)

    # ---- detection: upstream detect_watermark -------------------------------------------------------
    def score(self, text):
        n = len(self.tok(text, add_special_tokens=False)["input_ids"])
        if self.positional:                                  # alignment cost under the true key (lower = marked);
            return -self._alignment_cost(text), n            # calibrated by the shared empirical null, not n_runs
        if self.gamma_p:                                     # PF: sum of -log r ~ Gamma(k, 1) under the null
            import numpy as np
            from scipy.stats import gamma
            per_tok = self.wm.utils.get_scores_by_t(text)
            k = len(per_tok)
            if k == 0:
                return 0.0, n
            tot = float(np.asarray(self.wm.utils.get_scores(per_tok)).reshape(-1)[0])
            return -float(gamma.logsf(tot, a=k)) / math.log(10), n
        d = self.wm.detect_watermark(text, return_dict=True)
        s = float(d["score"])
        if self.p_value:                                     # upstream returns a p-value: report -log10 p
            s = -math.log10(max(s, 1e-300))
        n = len(self.tok(text, add_special_tokens=False)["input_ids"])
        return s, n


NOT_INCLUDED = {"EXPGumbel"}    # upstream builds a (V·prefix)×V noise table: 128 GB at V = 128k

for _k in TEACHERS:
    register_scheme(type(f"MarkLLM_{_k}", (MarkLLMDistill,), {"teacher": _k, "name": f"markllm-{_k.lower()}-d",
                                                              "experimental": _k in NOT_INCLUDED}))
