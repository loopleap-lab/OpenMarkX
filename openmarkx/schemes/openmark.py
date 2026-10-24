"""OpenMark (PRO in the paper, NeurIPS 2026) as an OpenMarkX scheme. The watermark itself is the paper's code (PRO),
shipped unchanged in openmarkx/third_party/pro.

Signal: one semantic bias per sequence, bias = delta * tanh(tau * (T(BERT(context)) - mean))[mapping], where T is the
watermark policy and `mapping` (token -> policy dimension) is the secret key. Score: PRO's detect_batch, the mean of
the mapped +-1 values over 10-word chunks after a 40-word context.

  openmark             the policy T frozen at its released initial weights (transform_model_cbert.pth)
  openmark-coopt-c100  T co-optimized with the student; its non-vanishing regularizer acts on T's centered output
                       (what reaches the logits) with weight 100, which keeps T from collapsing during distillation
  openmark-full-b01    openmark-coopt-c100 plus forgetting-perturbation training (FPL, beta 0.1)
The experimental variants are ablations on the way to these settings. Every scheme here also answers to its name
from before the rename (pro, pro-coopt-c100, ...), so keys, releases and configs that name it keep working.
"""
from __future__ import annotations

import os

import torch

from ..paths import RUNS
from ..models import text_vocab_size

INITIAL_POLICY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "third_party", "pro", "transform_model_cbert.pth")
from ..registry import register_scheme, register_scheme_alias
from ..primitives import Access
from .base import Scheme

EMBEDDER = "perceptiveshawty/compositional-bert-large-uncased"


def resolve_path(ref):
    from ..keys import resolve
    return resolve(ref)


@register_scheme
class OpenMark(Scheme):
    name, kind = "openmark", "distill_logit"
    access = Access.R0_AUX                                 # the detector runs the BERT embedder and the policy
    rng_bound = False                                      # the mapping is drawn by random.Random(secret): any device
    key_fields = ("mapping", "transform", "gamma", "delta", "chunk_size", "spread", "rng_device")

    def __init__(self, gamma=0.5, delta=1.0, chunk_size=10, transform_loss_weight=1e-3, tau_init=10.0, tau_max=1000.0,
                 co_opt=False, transform_lr=1e-5, transform_wd=0.01, centered_range=False, **params):
        super().__init__(gamma=gamma, delta=delta, chunk_size=chunk_size, transform_loss_weight=transform_loss_weight,
                         tau_init=tau_init, tau_max=tau_max, co_opt=co_opt, transform_lr=transform_lr,
                         transform_wd=transform_wd, centered_range=centered_range, **params)
        self.gamma, self.delta, self.chunk = gamma, delta, chunk_size
        self.w_aux, self.tau0, self.tau1, self.co_opt = transform_loss_weight, tau_init, tau_max, co_opt
        # The PRO code's own run trains the transform at its global lr (1e-5); at the LoRA lr (2e-4) the KL objective flattens the
        # transform until v - mean is numerical noise and the mark is unreadable (seen 2026-10-05). Keep PRO's setting.
        self.t_lr, self.t_wd = transform_lr, transform_wd
        self.centered_range = centered_range
        self.spread = []

    def setup(self, model, tokenizer, device, key=None):
        super().setup(model, tokenizer, device, key)
        import jieba
        jieba.dt.tmp_dir = os.path.join(RUNS, "cache"); os.makedirs(jieba.dt.tmp_dir, exist_ok=True)
        from ..third_party.pro import PROWatermark
        from ..keys import STORE, new_secret, resolve
        key = dict(key or {})
        if not key:
            # A new release gets a secret token -> policy-dimension mapping (the PRO code draws one at random when no
            # mapping file is given); the mapping shipped with the PRO code is public, so it must not be the key.
            import json
            import random
            secret = new_secret()
            rng = random.Random(secret)
            from ..keys import stage
            mpath = stage("mapping.json")
            json.dump([rng.randrange(300) for _ in range(text_vocab_size(model))], open(mpath, "w"))
            key = {"mapping": mpath}
        if "mapping" not in key:
            # never fall back to the mapping shipped with the PRO code: it is public, so it cannot be a key
            raise ValueError("this OpenMark key has no 'mapping': an OpenMark key must carry its own secret mapping "
                             "(pass key=None to draw a new one)")
        tpath = resolve(key.get("transform")) or INITIAL_POLICY
        mpath = resolve(key["mapping"])
        self.wm = PROWatermark(device=device, tokenizer=tokenizer, delta=self.delta, gamma=self.gamma,
                               chunk_size=self.chunk, embedding_model=EMBEDDER, transform_model_path=tpath,
                               mapping_file_path=mpath)
        if self.co_opt and not str(key.get("transform", "")).startswith(("blob:", "/")):   # embedding: train from init
            self.wm.set_tanh_temperature(self.tau0 if self.co_opt else self.tau1)
        else:                                              # a released key: frozen, hard +-1
            self.wm.transform_model.eval(); self.wm.set_tanh_temperature(self.tau1)
        return {"gamma": self.gamma, "delta": self.delta, "chunk_size": self.chunk, "transform": tpath, "mapping": mpath}

    # ---- training -------------------------------------------------------------------------------------
    # PRO's training path (watermark_logits on [batch, seq, vocab]) embeds the whole sequence once and adds that one
    # bias at every position; decoding embeds the prefix. Kept as in the PRO code; check C10 reports it.
    sequence_signal = "PRO's training adds one bias per sequence, computed from the whole sequence"

    def target_logits(self, input_ids, logits):
        self.wm.last_input_ids = None                      # the batch changes every step; never reuse the cache
        return self.wm.watermark_logits(input_ids, logits)

    def anti_logits(self, input_ids, logits):
        """The anti-watermarked teacher of PRO's FPL: the same bias, subtracted."""
        self.wm.last_input_ids = None
        out = self.wm.watermark_logits(input_ids, logits, anti=True)
        self.wm.last_input_ids = None                       # don't let the next target reuse these embeddings
        return out

    def trainable_params(self):
        return list(self.wm.transform_model.parameters()) if self.co_opt else []

    def param_groups(self):
        return [{"params": self.trainable_params(), "lr": self.t_lr, "weight_decay": self.t_wd}] if self.co_opt else []

    def aux_loss(self):
        if not self.co_opt:
            return None
        v = self.wm.last_transformed_embeddings
        scale = (1 - self.gamma) / self.gamma
        s = torch.where(v > 0, v * scale, v)
        mean_pen = s.mean(1).pow(2).mean() + s.mean(0).pow(2).mean()
        # PRO's code puts the non-vanishing term on raw v. M can then satisfy it by shifting every dimension to +0.05,
        # which the mean subtraction before tanh removes: the mark vanishes (seen with lambda = 1). centered_range
        # applies the term to v - mean, which is what reaches the logits.
        r = v - v.mean(-1, keepdim=True) if self.centered_range else v
        pen = torch.relu(0.05 - r.abs()); mask = (pen > 0).float()
        range_pen = (pen * mask).sum() / torch.clamp(mask.sum(), min=1.0)
        return self.w_aux * (mean_pen + range_pen)

    def on_step(self, step, steps):
        v = self.wm.last_transformed_embeddings
        if v is not None and step % 100 == 0:           # how far the transform's outputs are from collapsing
            c = (v - v.mean(-1, keepdim=True)).detach().abs().median().item()
            self.spread.append((step, c))
            print({"pro_spread": c, "step": step}, flush=True)
        if self.co_opt:
            self.wm.set_tanh_temperature(self.tau0 + min(step / steps, 1.0) * (self.tau1 - self.tau0))

    def wrong_key(self, params, rng, workdir):
        """OpenMark's secret is the token -> policy-dimension mapping: redraw it, keep the policy (trained or not)."""
        if "mapping" not in params:
            return None
        import json
        import random
        r = random.Random(rng.getrandbits(64))
        n = len(json.load(open(resolve_path(params["mapping"]))))
        path = os.path.join(workdir, "mapping.json")
        json.dump([r.randrange(300) for _ in range(n)], open(path, "w"))
        return {**params, "mapping": path}

    def save(self, key_dir, key):
        """The trained policy is part of the key: write it outside the release (the key store takes it from there)."""
        if self.co_opt:
            p = os.path.join(key_dir, "transform.pt")
            torch.save(self.wm.transform_model.state_dict(), p)
            key = {**key, "transform": p, "spread": self.spread[-1][1] if self.spread else None}
        return key

    def processor(self):
        """OpenMark at decoding time (the PRO code's own PROLogitsProcessor), frozen and hard +-1."""
        from ..third_party.pro import PROLogitsProcessor
        self.wm.transform_model.eval(); self.wm.set_tanh_temperature(self.tau1)
        lp = PROLogitsProcessor(self.wm)

        def call(input_ids, scores):
            with torch.no_grad():
                return lp(input_ids, scores)
        return call

    # ---- detection ------------------------------------------------------------------------------------
    def score_batch(self, texts, chunk=32):
        """PRO's detect_batch keeps a [pairs, vocab] array for every (context, chunk) pair of all texts at once; with a
        262k vocabulary (Gemma 3) and long answers that exceeds 64 GB of host memory, so texts go in chunks."""
        self.wm.transform_model.eval(); self.wm.set_tanh_temperature(self.tau1)
        texts, z = list(texts), []
        with torch.no_grad():
            for i in range(0, len(texts), chunk):
                z += list(self.wm.detect_batch(texts[i:i + chunk], batch_size=256))
        return [(float(s), len(self.tok(t, add_special_tokens=False)["input_ids"])) for s, t in zip(z, texts)]

    def score(self, text):
        return self.score_batch([text])[0]


@register_scheme
class OpenMarkCoopt(OpenMark):
    """Co-optimized policy (mapping model M trained with the student), with the paper's regularizer weight
    (lambda = 1, arXiv 2510.23891 Eq. 4). The code default 1e-3 let M collapse in our first runs."""
    name = "openmark-coopt"
    experimental = True                                # an ablation (see the module docstring)

    def __init__(self, **params):
        params.setdefault("co_opt", True)
        params.setdefault("transform_loss_weight", 1.0)
        super().__init__(**params)


@register_scheme
class OpenMarkBooster(OpenMark):
    """OpenMark + Booster robust training (PRO code: train_logit_distill_robust.py, robustness_mode=booster): each step
    simulates one fine-tuning step on clean text and penalizes how much it would move the model."""
    name = "openmark-booster"
    experimental = True                                # an ablation (see the module docstring)
    booster = {"alpha": 0.1, "lamb": 1.0}


@register_scheme
class OpenMarkFPL(OpenMark):
    """OpenMark + Forgotten Perturbation-aware Learning (arXiv 2510.23891 Eq. 6; code: booster_anti): each step simulates
    one fine-tuning step toward the anti-watermarked distribution and penalizes how much it moves the model,
    with the paper's alpha = 0.1 and beta = 5."""
    name = "openmark-fpl"
    experimental = True                                # an ablation (see the module docstring)
    booster = {"alpha": 0.1, "lamb": 5.0, "attack": "anti"}


@register_scheme
class OpenMarkCooptCentered(OpenMarkCoopt):
    """Co-optimized policy with the non-vanishing term on the centered output (closes the shift loophole)."""
    name = "openmark-coopt-c"
    experimental = True                                # an ablation (see the module docstring)

    def __init__(self, **params):
        params.setdefault("centered_range", True)
        super().__init__(**params)


@register_scheme
class OpenMarkFPLSmall(OpenMark):
    """FPL with beta = 0.5: tests whether beta = 5 pulls a LoRA student toward the anti-watermark optimum."""
    name = "openmark-fpl-b05"
    experimental = True                                # an ablation (see the module docstring)
    booster = {"alpha": 0.1, "lamb": 0.5, "attack": "anti"}


@register_scheme
class OpenMarkCooptCentered100(OpenMarkCooptCentered):
    """openmark-coopt-c with lambda = 100: the centered term averages over every violating entry (~5k per batch), so at
    lambda = 1 each entry gets ~1e-4 of gradient and the KL pull still flattens M (seen 2026-10-05)."""
    name = "openmark-coopt-c100"
    experimental = False

    def __init__(self, **params):
        params.setdefault("transform_loss_weight", 100.0)
        super().__init__(**params)


@register_scheme
class OpenMarkFull(OpenMarkCooptCentered100):
    """Full OpenMark as in the paper (PRO): co-optimized policy (centered non-vanishing term, lambda = 100) + FPL (beta = 0.5;
    beta = 5 pulled our LoRA student to the inverted mark)."""
    name = "openmark-full"
    experimental = True                                # an ablation (see the module docstring)
    booster = {"alpha": 0.1, "lamb": 0.5, "attack": "anti"}


@register_scheme
class OpenMarkFullB01(OpenMarkFull):
    """openmark-full with beta = 0.1 (beta 0.5 cut the learned share from 0.97 to 0.15)."""
    name = "openmark-full-b01"
    experimental = False
    booster = {"alpha": 0.1, "lamb": 0.1, "attack": "anti"}


# Names from before the rename (PRO in the paper): class names for code that imports them, scheme names for keys,
# releases, configs and command lines that use them.
PRO, PROCoopt, PROBooster, PROFPL, PROCooptCentered, PROFPLSmall, PROCooptCentered100, PROFull, PROFullB01 = (
    OpenMark, OpenMarkCoopt, OpenMarkBooster, OpenMarkFPL, OpenMarkCooptCentered, OpenMarkFPLSmall,
    OpenMarkCooptCentered100, OpenMarkFull, OpenMarkFullB01)
for _cls in (OpenMark, OpenMarkCoopt, OpenMarkBooster, OpenMarkFPL, OpenMarkCooptCentered, OpenMarkFPLSmall,
             OpenMarkCooptCentered100, OpenMarkFull, OpenMarkFullB01):
    register_scheme_alias("pro" + _cls.name[len("openmark"):], _cls.name)
