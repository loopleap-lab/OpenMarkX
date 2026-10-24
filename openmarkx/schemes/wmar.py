"""wmar: WMAR's generation-time KGW watermark on image tokens (Taming), learned into the weights.

A green list per context (the previous image token, context size 1) over the codebook's alive codes. The rule is a
score table Z over (previous alive code, next alive code); its green lists are the top gamma share of each row (hard,
straight-through). It starts at WMAR's linear-seeding table (from the taming adapter's codec), or, without one, at a
table drawn from the key's secret. With colearn (PRO's CAWP) the table is trained with the model; a column-balance
term keeps every code green in about gamma of the contexts, and an optional null term keeps the unmarked model from
already favouring the green lists. Ported from exp/arimg/pro_taming.py (policy_table, detect_table, cmd_colearn).

Signal contract (the taming adapter's forward): target_logits(prev, logits) takes, for each position whose context
is an alive code, prev = that context's index into the alive codes and logits = the model's logits over the alive
codes. score(codes) takes one image's code sequence (vocabulary ids).
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

from ..registry import register_scheme
from .base import Scheme


def policy_table(Z: torch.Tensor, gamma: float) -> torch.Tensor:
    """Hard green lists from the policy's scores: the top gamma share of alive codes in every row (context)."""
    k = round(gamma * Z.shape[1])
    return torch.zeros_like(Z, dtype=torch.bool).scatter_(1, Z.topk(k, 1).indices, True)


@register_scheme
class WMARScheme(Scheme):
    name = "wmar"
    exact_null = True

    def __init__(self, delta: float = 2.0, gamma: float = 0.25, balance: float = 10.0, null_w: float = 0.0,
                 colearn: bool = True):
        super().__init__(delta=delta, gamma=gamma, balance=balance, null_w=null_w, colearn=colearn)
        self._last = None

    # ---- lifecycle -------------------------------------------------------------------------
    def setup(self, model, codec, device: str, key: dict | None = None) -> dict:
        self.model, self.codec, self.device = model, codec, device
        key = dict(key or {})
        if "policy" in key:                                 # the detector, or a second phase: the saved rule
            from ..keys import resolve
            pol = torch.load(resolve(key["policy"]), weights_only=True)
            self.alive = pol["alive"].long()
            Z = pol["Z"].float()
        else:
            self.alive = torch.as_tensor(codec.alive).long()
            A = len(self.alive)
            if hasattr(codec, "initial_green"):             # WMAR's linear seeding, restricted to alive codes
                G0 = codec.initial_green().float()
            else:
                key.setdefault("secret", int.from_bytes(os.urandom(5), "big"))
                g = torch.Generator().manual_seed(int(key["secret"]))
                G0 = policy_table(torch.rand(A, A, generator=g), self.params["gamma"]).float()
            Z = 4.0 * (G0 - 0.5)                            # the policy, started at that table
        self.Z = torch.nn.Parameter(Z.to(device), requires_grad=bool(self.params["colearn"]))
        n = int(self.alive.max()) + 1
        self.pos = torch.full((max(n, getattr(codec, "vocab_size", n) if codec is not None else n),), -1, dtype=torch.long)
        self.pos[self.alive] = torch.arange(len(self.alive))
        return key

    def table(self) -> torch.Tensor:
        """[A, A] bool: row c = the green list after alive code c."""
        return policy_table(self.Z.detach().cpu(), self.params["gamma"])

    def wrong_key(self, params: dict, rng, workdir: str) -> dict:
        """A wrong key: a policy table drawn at random (same alive codes, same gamma), in workdir."""
        import os as _os
        g = torch.Generator().manual_seed(rng.getrandbits(40))
        A = len(self.alive)
        Z = 4.0 * (policy_table(torch.rand(A, A, generator=g), self.params["gamma"]).float() - 0.5)
        path = _os.path.join(workdir, f"policy_{rng.getrandbits(32):08x}.pt")
        torch.save({"green": policy_table(Z, self.params["gamma"]), "alive": self.alive.cpu(), "Z": Z}, path)
        return {**params, "policy": path}

    def save(self, key_dir: str, key: dict) -> dict:
        path = os.path.join(key_dir, "policy.pt")
        torch.save({"green": self.table(), "alive": self.alive.cpu(), "Z": self.Z.detach().cpu()}, path)
        return {**key, "policy": path}

    # ---- training signal ---------------------------------------------------------------------
    def target_logits(self, prev: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        soft = torch.sigmoid(self.Z)
        hard = policy_table(self.Z.detach(), self.params["gamma"]).float()
        g = hard + soft - soft.detach() if self.Z.requires_grad else hard   # hard lists forward, soft gradient
        rows = g[prev]
        self._last = (logits.detach().float(), rows, soft)
        return logits.float() + self.params["delta"] * rows

    def trainable_params(self) -> list:
        return [self.Z] if self.params["colearn"] else []

    def aux_loss(self):
        if self._last is None or not self.Z.requires_grad:
            return None
        base, rows, soft = self._last
        gam = self.params["gamma"]
        bal = (soft.mean(0) - gam).pow(2).mean() + (soft.mean(1) - gam).pow(2).mean()
        loss = self.params["balance"] * bal
        if self.params["null_w"]:
            null = ((F.softmax(base, -1) * rows).sum(-1) - gam).pow(2).mean()
            loss = loss + self.params["null_w"] * null
        return loss

    # ---- detection ---------------------------------------------------------------------------
    def score(self, codes) -> tuple[float, int]:
        """z on the green count over the distinct (previous, current) pairs of alive codes, each scored once."""
        c = torch.as_tensor(codes).long().cpu()
        pos = self.pos if len(self.pos) > int(c.max()) else torch.cat(
            [self.pos, torch.full((int(c.max()) + 1 - len(self.pos),), -1, dtype=torch.long)])
        a_, b_ = pos[c[:-1]], pos[c[1:]]
        ok = (a_ >= 0) & (b_ >= 0)
        pairs = torch.unique(torch.stack([a_[ok], b_[ok]], 1), dim=0)
        n = len(pairs)
        G = self.table()
        g = G[pairs[:, 0], pairs[:, 1]].sum().item() if n else 0
        gam = self.params["gamma"]
        return (g - gam * n) / math.sqrt(max(n, 1) * gam * (1 - gam)), n

    def pvalue(self, statistic: float, n_tokens: int) -> float:
        return 0.5 * math.erfc(statistic / math.sqrt(2))
