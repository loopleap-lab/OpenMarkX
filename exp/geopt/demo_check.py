"""The publisher's check as a picture: every surface point green or red, as KGW's green and red tokens.

For a suspect field and the publisher's own unmarked prediction on the same geometry and flight condition, a point is
green when the residual (suspect - unmarked) points the way the key says, i.e. sum over channels of residual_c * g_c > 0
with g the keyed pattern of the unmarked prediction. Unmarked fields: about half green, scattered. Fields from the
watermarked model: mostly green.

  python exp/geopt/demo_check.py --geopt_dir ... --data ... --ckpt clean.pt --student student.pt --out DIR
Writes, per case, five panels in the site palette (shape, predicted field, keyed pattern, check of the
watermarked and of an unmarked model) and DIR/check.json (green fractions, z, flight condition).
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fieldmark as fm
from run_mark import Sim, craft_args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geopt_dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", type=int, default=20261020)
    ap.add_argument("--output_scale", type=float, default=1.0)
    ap.add_argument("--cases", default="0,1,2,3")
    ap.add_argument("--channel_names", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fm.DOMAIN, fm.OUTPUT_SCALE = "output", a.output_scale
    sys.path.insert(0, a.geopt_dir)
    from data_provider.data_factory import get_data
    from exp.dynamics_config import get_direction
    from models.model_factory import get_model
    args = craft_args(a)
    _, train_loader, _, args.shapelist = get_data(args)
    _, _, full_loader, _ = get_data(args, full_mesh=True)
    clean = get_model(args).cuda(); clean.load_state_dict(torch.load(a.ckpt, map_location="cuda")); clean.eval()
    student = copy.deepcopy(clean); student.load_state_dict(torch.load(a.student, map_location="cuda")); student.eval()
    sim = Sim(clean, get_direction("craft"))
    # an unmarked model that is not the publisher's: the clean model fine-tuned 2 epochs on the simulation labels
    other = copy.deepcopy(clean); other.train()
    opt = torch.optim.AdamW(other.parameters(), lr=1e-4)
    for _ in range(2):
        for p, _, c, y in train_loader:
            p, c, y = p.cuda(), c.cuda(), y.cuda()
            loss = (sim(p, c, other) - y).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    other.eval()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    # the site's palette: ink, paper, moss (follows the key), clay red (does not), amber
    INK, PAPER, MOSS, CLAY, AMBER = "#231A15", "#FFFDF6", "#5D6B2C", "#C0472F", "#DF9D49"
    gr = ListedColormap([CLAY, MOSS])
    heat = LinearSegmentedColormap.from_list("heat", ["#F6E2C2", AMBER, "#8B4B2E", INK])
    ripple = LinearSegmentedColormap.from_list("ripple", ["#3E5C76", PAPER, AMBER])   # not green/red: that pair means the check

    def panel(path, x, y, color, cmap=None, vmin=None, vmax=None, size=(5.2, 3.2)):
        fig = plt.figure(figsize=size, facecolor=PAPER); ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(PAPER)
        ax.scatter(x, y, c=color, cmap=cmap, vmin=vmin, vmax=vmax, s=1.6, rasterized=True, linewidths=0)
        ax.set_aspect("equal"); ax.axis("off")
        fig.savefig(path, dpi=150, facecolor=PAPER); plt.close(fig)

    out = {}
    cases = [int(x) for x in a.cases.split(",")]
    with torch.no_grad():
        for i, (p, _, c, _) in enumerate(full_loader.dataset):
            if i not in cases:
                continue
            p, c = p.cuda()[None], c.cuda()[None]
            ref = sim(p, c, clean)
            g = fm.output_pattern(ref, a.key)
            res = {}
            for name, m in (("watermarked", student), ("unmarked", other)):
                r = sim(p, c, m) - ref
                score = (r * g).sum(-1)[0]                   # per point
                green = (score > 0).float()
                z = fm.zscore(p, r, a.key, n_wrong=200, ref=ref).item()
                res[name] = {"green": green.cpu().numpy(), "frac": green.mean().item(), "z": z}
            pts = p[0, :, :3].cpu().numpy()
            rng = np.random.default_rng(0)
            idx = rng.choice(len(pts), 120000, replace=False)
            idx = idx[pts[idx, 1] >= np.median(pts[idx, 1])]  # upper surface, seen from above
            X, Y = pts[idx, 0], pts[idx, 2]
            field = sim(p, c, student)[0, :, 0].cpu().numpy()[idx]
            pat = g[0, :, 0].cpu().numpy()[idx]
            lim = np.percentile(np.abs(pat), 98)
            panel(f"{a.out}/shape_case{i}.png", X, Y, "#B9B2A6")
            panel(f"{a.out}/field_case{i}.png", X, Y, field, heat, *np.percentile(field, [2, 98]))
            panel(f"{a.out}/ripple_case{i}.png", X, Y, pat, ripple, -lim, lim)
            for name in ("watermarked", "unmarked"):
                panel(f"{a.out}/check_{name}_case{i}.png", X, Y, res[name]["green"][idx], gr, 0, 1)
            cc = c[0, 0].tolist()
            out[f"case{i}"] = {**{k: {"green_frac": v["frac"], "z": v["z"]} for k, v in res.items()},
                               "condition": {"mach": cc[0], "aoa": cc[1], "sideslip": cc[2]}}
            print(json.dumps({f"case{i}": out[f"case{i}"]}), flush=True)
    json.dump(out, open(f"{a.out}/check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
