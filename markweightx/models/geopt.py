"""The geopt adapter: GeoPT (a Transolver physics surrogate) fine-tuned on aircraft (the authors' craft recipe),
predicting 6 output channels on surface points for a flight condition. The GeoPT repo stays external: set
MWX_GEOPT_DIR to its checkout. Inputs are (points [1, N, 7], condition [1, 1, 3]); outputs are fields [1, N, 6] in
the model's normalized units. The imprint is a full fine-tune. Ported from exp/geopt/run_mark.py.
"""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import torch

from ..registry import register_model
from .base import Model, Output


def craft_args(data: str):
    """The authors' scripts/finetune/GeoPT_craft.sh settings."""
    return SimpleNamespace(data_path=data, loader="AirCraft", task="GeoPT_finetune", dynamics="craft",
                           geotype="unstructured", space_dim=3, fun_dim=11, out_dim=6, normalize=1,
                           norm_type="UnitTransformer", model="Transolver", n_hidden=256, n_heads=8, n_layers=8,
                           mlp_ratio=2, slice_num=32, ntrain=100, ntest=50, batch_size=1, act="gelu", dropout=0.0,
                           checkpoint=0, shapelist=None)


class _GeoPT:
    """The GeoPT repo behind the stand-in interface: model(), (net, points, cond) -> output, splits(), y_normalizer."""

    def __init__(self, ckpt, data, device):
        from ..external import require_external
        require_external("GeoPT", "MWX_GEOPT_DIR", "models.model_factory")
        from data_provider.data_factory import get_data
        from exp.dynamics_config import get_direction
        from models.model_factory import get_model
        args = craft_args(data)
        self.dataset, self._train, self._test, args.shapelist = get_data(args)
        self.y_normalizer = self.dataset.y_normalizer
        self.direction = get_direction("craft")
        self.net = get_model(args).to(device)
        if ckpt:
            self.net.load_state_dict(torch.load(ckpt, map_location=device))
        self.device = device

    def model(self):
        return self.net

    def __call__(self, net, points, cond):
        v = self.direction(points, cond)
        return net(points[:, :, :3], torch.cat((points, v), -1))

    def splits(self):
        tr = [(p[None].to(self.device), c[None].to(self.device), y[None].to(self.device)) for p, _, c, y in self._train.dataset]
        te = [(p[None].to(self.device), c[None].to(self.device), y[None].to(self.device)) for p, _, c, y in self._test.dataset]
        return tr, te                                       # train outputs normalized, test outputs physical


@register_model
class GeoPTModel(Model):
    name = "geopt"
    unit = "surface point"
    output = Output.FIELD

    def __init__(self, ckpt: str | None = None, data: str | None = None, factory: str | None = None, **params):
        super().__init__(ckpt=ckpt, data=data, factory=factory, **params)
        self.sim = None

    def load(self, path, device: str):
        if self.sim is None:
            p = self.params
            if p["factory"]:
                mod, _, attr = p["factory"].partition(":")
                self.sim = getattr(importlib.import_module(mod), attr)(p["ckpt"], p["data"], device)
            else:
                self.sim = _GeoPT(p["ckpt"], p["data"], device)
        import copy
        net = copy.deepcopy(self.sim.model())
        weights = getattr(path, "weights", None)
        if weights:
            net.load_state_dict(torch.load(weights, map_location=device))
        return net.to(device).eval()

    def splits(self, **kw):
        """(train, test): lists of (points, condition, field); train fields normalized, test fields physical."""
        return self.sim.splits(**kw)

    @staticmethod
    def head(net):
        """The last linear layer: GeoPT's blocks[-1].mlp2 (a stand-in's last nn.Linear)."""
        if hasattr(net, "blocks"):
            return net.blocks[-1].mlp2
        return [m for m in net.modules() if isinstance(m, torch.nn.Linear)][-1]

    @torch.no_grad()
    def edit_last(self, net, scheme, train, *, reps: int = 3, ridge: float = 1e-3, seed: int = 0) -> dict:
        """Write the mark into the last linear layer in closed form (run_mark.py's --edit_last, used with --steps 0
        by the s1_e02 demo): ridge regression of (marked - unmarked output) on the layer's input features, over the
        training geometries at their own flight condition, then (reps - 1) more times at random conditions."""
        head = self.head(net)
        feats = {}
        hook = head.register_forward_hook(lambda mod, inp, out: feats.__setitem__("h", inp[0].detach()))
        dev = head.weight.device
        d = head.in_features + 1
        HtH = torch.zeros(d, d, device=dev, dtype=torch.float64)
        HtG = torch.zeros(d, head.out_features, device=dev, dtype=torch.float64)
        conds = torch.cat([c for _, c, _ in train])
        lo, hi = conds.amin(0), conds.amax(0)
        g = torch.Generator().manual_seed(seed)
        try:
            for rep in range(reps):
                for p, c0, _ in train:
                    c = c0 if rep == 0 else (lo + (hi - lo) * torch.rand(lo.shape, generator=g).to(dev))[None]
                    o = self.sim(net, p, c)
                    h = feats["h"].reshape(-1, head.in_features)
                    H = torch.cat([h, torch.ones_like(h[:, :1])], 1).double()
                    HtH += H.T @ H
                    HtG += H.T @ (scheme.target_output({"points": p}, o) - o).reshape(-1, head.out_features).double()
        finally:
            hook.remove()
        lam = ridge * HtH.diagonal().mean()
        dW = torch.linalg.solve(HtH + lam * torch.eye(d, device=dev, dtype=torch.float64), HtG).float()
        head.weight += dW[:-1].T
        head.bias += dW[-1]
        return {"edit_last": True, "dW_rel": float(dW[:-1].norm() / head.weight.norm())}

    @property
    def y_normalizer(self):
        return self.sim.y_normalizer

    def forward(self, net, batch: dict):
        return batch, self.sim(net, batch["points"], batch["cond"])

    @torch.no_grad()
    def run(self, net, inputs: list, **kw) -> list:
        return [self.sim(net, p, c) for p, c in inputs]

    def trainable(self, net, mode: str = "full"):
        return super().trainable(net, mode)
