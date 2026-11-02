"""Dump the GeoPT demo example (test aircraft 0 and 1) as one small JSON for the website figures (render.py).

Rebuilds the same context as openmarkx/pipeline.py `_viewer_example` (geopt branch), read-only: the run's
config.json -> Context, the adapter's test split and y_normalizer, the key's ripple scheme, and generate.pt
(`wm`, `base`: marked / unmarked GeoPT predictions on the test aircraft, normalized units). For each aircraft:
the upper surface (height >= median), 6000 points sampled with seed 0, as in the viewer.

CPU only. GeoPT's AirCraft loader calls y_normalizer.cuda(); on a CPU node that call is made a no-op below.

    python docs/tools/geopt_figs/dump_example.py RUN_DIR OUT_JSON [--n 2]
"""
from __future__ import annotations

import argparse
import json
import os
import random

import torch

torch.Tensor.cuda = lambda self, *a, **k: self            # CPU job: GeoPT's loader moves the normalizer to cuda

from openmarkx.pipeline import Config, Context              # noqa: E402  (read-only use of the package)
from openmarkx.quality import rel_l2                        # noqa: E402
from openmarkx.schemes.ripple import output_pattern         # noqa: E402


def r(t, d=4):
    return [round(float(v), d) for v in t]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out_json")
    ap.add_argument("--n", type=int, default=2, help="how many test aircraft (from test[0])")
    a = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
    d = json.load(open(os.path.join(a.run_dir, "config.json")))
    cfg = Config(**{k: v for k, v in d.items() if v is not None and k != "names"})
    cfg.out = a.run_dir
    ctx = Context(cfg, "cpu")
    ctx.ensure_loaded()
    sch = ctx.bound_scheme()
    prm = sch.params
    g = torch.load(os.path.join(a.run_dir, "generate.pt"), map_location="cpu", weights_only=False)
    wm_all, base_all = g["wm"], g["base"]
    _, test = ctx.adapter.splits()
    dec = ctx.adapter.y_normalizer.decode
    wrong = sch.wrong_key({"secret": sch.secret}, random.Random(0xC0FFEE), a.run_dir) or {"secret": sch.secret + 1}
    q = json.load(open(os.path.join(a.run_dir, "quality.json")))

    aircraft = []
    for i in range(min(a.n, len(wm_all))):
        p, cond, y = test[i]
        ref, out = base_all[i].float().cpu(), wm_all[i].float().cpu()
        gk = output_pattern(ref, sch.secret, prm["m"], prm["output_scale"])
        gw = output_pattern(ref, int(wrong["secret"]), prm["m"], prm["output_scale"])
        green_key = ((out - ref) * gk).sum(-1)[0] > 0
        green_wrong = ((out - ref) * gw).sum(-1)[0] > 0
        pts = p[0, :, :3].float().cpu()
        idx = torch.nonzero(pts[:, 1] >= pts[:, 1].median()).flatten()
        idx = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(0))[:6000]]
        side = torch.randperm(len(pts), generator=torch.Generator().manual_seed(1))[:3000]
        truth = y.float().cpu()                                   # test fields are physical already
        f_base, f_wm = dec(ref)[0].float(), dec(out)[0].float()  # [N, 6] physical
        t0, b0, w0 = truth[0, :, 0], f_base[:, 0], f_wm[:, 0]
        rng0 = float(t0.max() - t0.min())
        rel = lambda u, v: float((u - v).norm() / v.norm())
        aircraft.append({
            "test_index": i,
            "condition": {"mach": float(cond.flatten()[0]), "aoa_deg": float(cond.flatten()[1]),
                          "sideslip_deg": float(cond.flatten()[2])},
            "x": r(pts[idx, 0], 3), "z": r(pts[idx, 2], 3), "h": r(pts[idx, 1], 3),
            "nh": r(p[0, idx, 5].float().cpu(), 3),              # vertical part of the surface normal (up > 0)
            "side_x": r(pts[side, 0], 3), "side_h": r(pts[side, 1], 3), "side_z": r(pts[side, 2], 3),
            "truth": r(t0[idx]), "base": r(b0[idx]), "wm": r(w0[idx]),
            "pattern": r(gk[0, :, 0][idx], 3),
            "green_key": green_key[idx].int().tolist(), "green_wrong": green_wrong[idx].int().tolist(),
            "rel_l2_all": {"base": rel_l2(f_base[None], truth), "wm": rel_l2(f_wm[None], truth)},
            "rel_l2_ch0": {"base": rel(b0, t0), "wm": rel(w0, t0)},
            "max_abs_diff_over_range_ch0": float((w0 - b0).abs().max()) / rng0,
            "rms_diff_over_range_ch0": float((w0 - b0).pow(2).mean().sqrt()) / rng0,
            "truth_range_ch0": [float(t0.min()), float(t0.max())],
            "truth_channel_stats": [{"min": float(truth[0, :, c].min()), "max": float(truth[0, :, c].max()),
                                     "mean": float(truth[0, :, c].mean())} for c in range(truth.shape[-1])],
        })
    res = {"run": a.run_dir, "eps": prm["eps"], "m": prm["m"],
           "quantity": "channel 0 of the AirCraft CFD surface fields (pressure)",
           "quality": q, "aircraft": aircraft}
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    with open(a.out_json, "w") as f:
        json.dump(res, f)
    for ac in aircraft:
        print(json.dumps({k: ac[k] for k in ("test_index", "condition", "rel_l2_all", "rel_l2_ch0",
                                               "max_abs_diff_over_range_ch0", "truth_range_ch0")}))
        print(json.dumps(ac["truth_channel_stats"]))
    print("wrote", a.out_json, os.path.getsize(a.out_json), "bytes")


if __name__ == "__main__":
    main()
