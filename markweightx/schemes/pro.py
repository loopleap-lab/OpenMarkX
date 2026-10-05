"""PRO (ours, NeurIPS 2026) as a MarkWeightX scheme. The PRO paper code is imported unchanged from PRO_CODE.

Training signal (as in PRO's train_logit_distill_Co-optimization.py): one semantic bias per sequence,
bias = delta * tanh(tau * (T(BERT(context)) - mean))[mapping], with the transform T trained jointly with the student
(co-optimization): tau anneals 10 -> 1000 and T carries a row/column-mean + range regularizer (weight 1e-3).
Detection: PRO's detect_batch (mean of the mapped +-1 values over 10-word chunks after a 40-word context).
"""
from __future__ import annotations

import os
import sys

import torch

from ..paths import PRO_CODE, RUNS
from ..registry import register_scheme
from .base import Scheme

EMBEDDER = "perceptiveshawty/compositional-bert-large-uncased"


@register_scheme
class PRO(Scheme):
    name, kind = "pro", "distill_logit"
    needs_model_to_detect = False

    def __init__(self, gamma=0.5, delta=1.0, chunk_size=10, transform_loss_weight=1e-3, tau_init=10.0, tau_max=1000.0,
                 co_opt=True, transform_lr=1e-5, transform_wd=0.01, **params):
        super().__init__(gamma=gamma, delta=delta, chunk_size=chunk_size, **params)
        self.gamma, self.delta, self.chunk = gamma, delta, chunk_size
        self.w_aux, self.tau0, self.tau1, self.co_opt = transform_loss_weight, tau_init, tau_max, co_opt
        # PRO's own run trains the transform at its global lr (1e-5); at the LoRA lr (2e-4) the KL objective flattens the
        # transform until v - mean is numerical noise and the mark is unreadable (seen 2026-10-05). Keep PRO's setting.
        self.t_lr, self.t_wd = transform_lr, transform_wd
        self.spread = []

    def setup(self, model, tokenizer, device, key=None):
        super().setup(model, tokenizer, device, key)
        if PRO_CODE not in sys.path:
            sys.path.insert(0, PRO_CODE)
        import jieba
        jieba.dt.tmp_dir = os.path.join(RUNS, "cache"); os.makedirs(jieba.dt.tmp_dir, exist_ok=True)
        from watermarks.pro.pro_watermark import PROWatermark
        key = dict(key or {})
        tpath = key.get("transform") or os.path.join(PRO_CODE, "watermarks/pro/transform_model_cbert.pth")
        mpath = key.get("mapping") or os.path.join(PRO_CODE, f"watermarks/pro/300_mapping_{model.config.vocab_size}.json")
        self.wm = PROWatermark(device=device, tokenizer=tokenizer, delta=self.delta, gamma=self.gamma,
                               chunk_size=self.chunk, embedding_model=EMBEDDER, transform_model_path=tpath,
                               mapping_file_path=mpath)
        if not key.get("transform"):                       # embedding: the transform is trained from this init
            self.wm.set_tanh_temperature(self.tau0 if self.co_opt else self.tau1)
        else:                                              # a released key: frozen, hard +-1
            self.wm.transform_model.eval(); self.wm.set_tanh_temperature(self.tau1)
        return {"gamma": self.gamma, "delta": self.delta, "chunk_size": self.chunk, "transform": tpath, "mapping": mpath}

    # ---- training -------------------------------------------------------------------------------------
    def target_logits(self, input_ids, logits):
        self.wm.last_input_ids = None                      # the batch changes every step; never reuse the cache
        return self.wm.watermark_logits(input_ids, logits)

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
        pen = torch.relu(0.05 - v.abs()); mask = (pen > 0).float()
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
        if self.co_opt:
            p = os.path.join(out, "transform_model.pt")
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
