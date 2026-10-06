"""PRO (ours, NeurIPS 2026) as a MarkWeightX scheme. The PRO paper code is imported unchanged from PRO_CODE.

Training signal (as in PRO's train_logit_distill_Co-optimization.py): one semantic bias per sequence,
bias = delta * tanh(tau * (T(BERT(context)) - mean))[mapping], with the transform T trained jointly with the student
(co-optimization): tau anneals 10 -> 1000 and T carries a row/column-mean + range regularizer (weight 1e-3).
Detection: PRO's detect_batch (mean of the mapped +-1 values over 10-word chunks after a 40-word context).

`pro` keeps the transform frozen at the released weights (transform_model_cbert.pth), which is what the released PRO
8B checkpoint used (its transform file is byte-identical to that init). `pro-coopt` trains the transform jointly, as
in train_logit_distill_Co-optimization.py. In our runs (1B at lr 2e-4, 8B at PRO's 1e-5) co-optimization flattens the
transform: the KL gradient pulls the teacher toward the base, so |v - mean| decays to ~1e-4 within 200 steps and the
mark becomes unreadable. Kept as a separate scheme for the PRO authors to look at.
"""
from __future__ import annotations

import os
import sys

import torch

from ..paths import PRO_CODE, RUNS
from ..registry import register_scheme
from ..primitives import Access
from .base import Scheme

EMBEDDER = "perceptiveshawty/compositional-bert-large-uncased"


@register_scheme
class PRO(Scheme):
    name, kind = "pro", "distill_logit"
    access = Access.R0_AUX                                 # the detector runs the BERT embedder and the policy

    def __init__(self, gamma=0.5, delta=1.0, chunk_size=10, transform_loss_weight=1e-3, tau_init=10.0, tau_max=1000.0,
                 co_opt=False, transform_lr=1e-5, transform_wd=0.01, centered_range=False, **params):
        super().__init__(gamma=gamma, delta=delta, chunk_size=chunk_size, **params)
        self.gamma, self.delta, self.chunk = gamma, delta, chunk_size
        self.w_aux, self.tau0, self.tau1, self.co_opt = transform_loss_weight, tau_init, tau_max, co_opt
        # PRO's own run trains the transform at its global lr (1e-5); at the LoRA lr (2e-4) the KL objective flattens the
        # transform until v - mean is numerical noise and the mark is unreadable (seen 2026-10-05). Keep PRO's setting.
        self.t_lr, self.t_wd = transform_lr, transform_wd
        self.centered_range = centered_range
        self.spread = []

    def setup(self, model, tokenizer, device, key=None):
        super().setup(model, tokenizer, device, key)
        if PRO_CODE not in sys.path:
            sys.path.insert(0, PRO_CODE)
        import jieba
        jieba.dt.tmp_dir = os.path.join(RUNS, "cache"); os.makedirs(jieba.dt.tmp_dir, exist_ok=True)
        from watermarks.pro.pro_watermark import PROWatermark
        from ..keys import STORE, new_secret, resolve
        key = dict(key or {})
        if not key:
            # A new release gets a secret token -> policy-dimension mapping (PRO's code draws one at random when no
            # mapping file is given); the mapping shipped with the PRO code is public, so it must not be the key.
            import json
            import random
            secret = new_secret()
            rng = random.Random(secret)
            from ..keys import stage
            mpath = stage("mapping.json")
            json.dump([rng.randrange(300) for _ in range(model.config.vocab_size)], open(mpath, "w"))
            key = {"mapping": mpath}
        tpath = resolve(key.get("transform")) or os.path.join(PRO_CODE, "watermarks/pro/transform_model_cbert.pth")
        mpath = resolve(key.get("mapping")) or os.path.join(PRO_CODE, f"watermarks/pro/300_mapping_{model.config.vocab_size}.json")
        self.wm = PROWatermark(device=device, tokenizer=tokenizer, delta=self.delta, gamma=self.gamma,
                               chunk_size=self.chunk, embedding_model=EMBEDDER, transform_model_path=tpath,
                               mapping_file_path=mpath)
        if self.co_opt and not str(key.get("transform", "")).startswith(("blob:", "/")):   # embedding: train from init
            self.wm.set_tanh_temperature(self.tau0 if self.co_opt else self.tau1)
        else:                                              # a released key: frozen, hard +-1
            self.wm.transform_model.eval(); self.wm.set_tanh_temperature(self.tau1)
        return {"gamma": self.gamma, "delta": self.delta, "chunk_size": self.chunk, "transform": tpath, "mapping": mpath}

    # ---- training -------------------------------------------------------------------------------------
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

    def save(self, out, key):
        """The trained policy is part of the key: write it outside the release (the key store takes it from there)."""
        if self.co_opt:
            from ..keys import stage
            p = stage("transform.pt")
            torch.save(self.wm.transform_model.state_dict(), p)
            key = {**key, "transform": p, "spread": self.spread[-1][1] if self.spread else None}
        return key

    def processor(self):
        """PRO at decoding time (PRO's own PROLogitsProcessor), frozen and hard +-1."""
        from watermarks.pro.logits_processor import PROLogitsProcessor
        self.wm.transform_model.eval(); self.wm.set_tanh_temperature(self.tau1)
        lp = PROLogitsProcessor(self.wm)

        def call(input_ids, scores):
            with torch.no_grad():
                return lp(input_ids, scores)
        return call

    # ---- detection ------------------------------------------------------------------------------------
    def score_batch(self, texts):
        self.wm.transform_model.eval(); self.wm.set_tanh_temperature(self.tau1)
        with torch.no_grad():
            z = self.wm.detect_batch(list(texts), batch_size=256)
        return [(float(s), len(self.tok(t, add_special_tokens=False)["input_ids"])) for s, t in zip(z, texts)]

    def score(self, text):
        return self.score_batch([text])[0]


@register_scheme
class PROCoopt(PRO):
    """Co-optimized policy (mapping model M trained with the student), with the paper's regularizer weight
    (lambda = 1, arXiv 2510.23891 Eq. 4). The code default 1e-3 let M collapse in our first runs."""
    name = "pro-coopt"
    experimental = True                                # a step in reproducing PRO; see the dev plan

    def __init__(self, **params):
        params.setdefault("co_opt", True)
        params.setdefault("transform_loss_weight", 1.0)
        super().__init__(**params)


@register_scheme
class PROBooster(PRO):
    """PRO + Booster robust training (PRO code: train_logit_distill_robust.py, robustness_mode=booster): each step
    simulates one fine-tuning step on clean text and penalizes how much it would move the model."""
    name = "pro-booster"
    experimental = True                                # a step in reproducing PRO; see the dev plan
    booster = {"alpha": 0.1, "lamb": 1.0}


@register_scheme
class PROFPL(PRO):
    """PRO + Forgotten Perturbation-aware Learning (arXiv 2510.23891 Eq. 6; code: booster_anti): each step simulates
    one fine-tuning step toward the anti-watermarked distribution and penalizes how much it moves the model,
    with the paper's alpha = 0.1 and beta = 5."""
    name = "pro-fpl"
    experimental = True                                # a step in reproducing PRO; see the dev plan
    booster = {"alpha": 0.1, "lamb": 5.0, "attack": "anti"}


@register_scheme
class PROCooptCentered(PROCoopt):
    """Co-optimized policy with the non-vanishing term on the centered output (closes the shift loophole)."""
    name = "pro-coopt-c"
    experimental = True                                # a step in reproducing PRO; see the dev plan

    def __init__(self, **params):
        params.setdefault("centered_range", True)
        super().__init__(**params)


@register_scheme
class PROFPLSmall(PRO):
    """FPL with beta = 0.5: tests whether beta = 5 pulls a LoRA student toward the anti-watermark optimum."""
    name = "pro-fpl-b05"
    experimental = True                                # a step in reproducing PRO; see the dev plan
    booster = {"alpha": 0.1, "lamb": 0.5, "attack": "anti"}


@register_scheme
class PROCooptCentered100(PROCooptCentered):
    """pro-coopt-c with lambda = 100: the centered term averages over every violating entry (~5k per batch), so at
    lambda = 1 each entry gets ~1e-4 of gradient and the KL pull still flattens M (seen 2026-10-05)."""
    name = "pro-coopt-c100"
    experimental = False

    def __init__(self, **params):
        params.setdefault("transform_loss_weight", 100.0)
        super().__init__(**params)


@register_scheme
class PROFull(PROCooptCentered100):
    """Full PRO as in the paper: co-optimized policy (centered non-vanishing term, lambda = 100) + FPL (beta = 0.5;
    beta = 5 pulled our LoRA student to the inverted mark)."""
    name = "pro-full"
    experimental = True                                # a step in reproducing PRO; see the dev plan
    booster = {"alpha": 0.1, "lamb": 0.5, "attack": "anti"}


@register_scheme
class PROFullB01(PROFull):
    """pro-full with beta = 0.1 (beta 0.5 cut the learned share from 0.97 to 0.15)."""
    name = "pro-full-b01"
    experimental = False
    booster = {"alpha": 0.1, "lamb": 0.1, "attack": "anti"}
