"""Website figures for the GeoPT demo (docs/static/geo2-*.png).

    python render.py EXAMPLE.json [--out DIR] [--aircraft 0]

EXAMPLE.json is either the dump from dump_example.py (all figures, incl. geo2-quality.png) or the run's
viewer.json (docs/static/runs/geopt/viewer.json: draft input / output / key only, no ground truth there).
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
from matplotlib import font_manager                              # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize  # noqa: E402
from matplotlib.patches import FancyArrowPatch, PathPatch, Patch  # noqa: E402
from matplotlib.path import Path                                  # noqa: E402
from matplotlib.tri import Triangulation                          # noqa: E402

BG, INK, MUTED, ACCENT, GOOD, BAD = "#FAF9F5", "#141413", "#5E5D59", "#C15F3C", "#5C7043", "#B8322A"
COOL = "#3F5E78"                                                  # the low end of the field map
SILH = "#E6E1D6"
plt.rcParams.update({
    "font.family": [f for f in ("JetBrains Mono", "DejaVu Sans Mono")
                    if f in {x.name for x in font_manager.fontManager.ttflist}][:1] or ["monospace"],
    "font.size": 13,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
})
FS_TITLE, FS_TEXT, FS_SMALL = 16, 13, 11
# Text sizes follow the page's type tiers (14 / 16 / 20 px) at the width each image is shown: the three small
# panels are 216 px wide on the desktop page, the accuracy strip 1116 px, its phone version (2 x 2) 306 px.
SMALL_W, STRIP_W, PHONE_W = 216, 1116, 306


def pt(px, shown_w, fig_w):
    """Font size in points that renders at px CSS pixels when a fig_w-inch figure is shown shown_w px wide."""
    return px * fig_w * 72 / shown_w


S14, S16 = pt(14, SMALL_W, 8), pt(16, SMALL_W, 8)              # the 8-inch panels
FIELD_CMAP = LinearSegmentedColormap.from_list("field", [COOL, "#9FB1BE", "#F2EEE6", "#E0A88A", ACCENT, "#7E2F17"])
DIFF_CMAP = LinearSegmentedColormap.from_list("diff", [COOL, "#F2EEE6", ACCENT])
KEY_CMAP = ListedColormap([MUTED, ACCENT])
DPI = 200                                                          # 2x of an 800 x 480 page image


class QuantileNorm(Normalize):
    """Colour by rank: equal steps of colour hold equal shares of the surface, so the bulk of the field shows its
    structure instead of a few extreme points (the trailing edge) taking up the whole scale."""

    def __init__(self, values, n=40):
        self.q = np.unique(np.percentile(values, np.linspace(0, 100, n + 1)))
        super().__init__(self.q[0], self.q[-1])

    def __call__(self, v, clip=None):
        return np.ma.masked_invalid(np.interp(np.asarray(v, float), self.q, np.linspace(0, 1, len(self.q))))

    def inverse(self, u):
        return np.interp(u, np.linspace(0, 1, len(self.q)), self.q)


def load(path, k):
    d = json.load(open(path))
    if "aircraft" in d:                                            # dump_example.py
        a = d["aircraft"][k]
        keep = np.array(a["nh"]) > 0 if "nh" in a else slice(None)   # the upward-facing surface only
        a = {**a, **{f: list(np.array(a[f])[keep]) for f in ("x", "z", "wm", "truth", "base", "pattern")}}
        return {"x": np.array(a["x"]), "z": np.array(a["z"]), "field": np.array(a["wm"]),
                "truth": np.array(a["truth"]), "base": np.array(a["base"]), "wm": np.array(a["wm"]),
                "pattern": np.array(a["pattern"]), "side": (np.array(a["side_x"]), np.array(a["side_h"])),
                "mach": a["condition"]["mach"], "eps": d["eps"], "a": a, "quality": d["quality"],
                "n_aircraft": len(d["aircraft"])}
    e = d["example"]                                               # viewer.json
    return {"x": np.array(e["x"]), "z": np.array(e["y"]), "field": np.array(e["field"]),
            "pattern": np.array(e["pattern"]), "side": None, "mach": e["condition"][0],
            "eps": d["scheme"]["params"]["eps"], "quality": d.get("quality")}


def outline(x, z, nb=45):
    """The planform seen from above: per slice along the body, the outermost points on each side."""
    edges = np.linspace(x.min(), x.max(), nb + 1)
    xs, lo, hi = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x <= b)
        if m.sum() < 2:
            continue
        xs.append(x[m].mean()); lo.append(z[m].min()); hi.append(z[m].max())
    xs, lo, hi = np.array(xs), np.array(lo), np.array(hi)
    c = float(np.median((lo + hi) / 2))                       # the plane is mirror-symmetric about its centre line
    half = np.maximum.accumulate(np.maximum(hi - c, c - lo))   # the span only grows from nose to tail
    half = np.convolve(np.r_[half[0], half, half[-1]], np.ones(3) / 3, "valid")  # no pixel stairs
    xs[-1] = x.max()
    xs = np.r_[x.min(), xs, x.max()]
    half = np.r_[0.0, half, 0.0]
    return np.vstack([np.c_[xs, c + half], np.c_[xs[::-1], c - half[::-1]]])


def tri(x, z, poly):
    t = Triangulation(x, z)
    c = np.c_[x[t.triangles].mean(1), z[t.triangles].mean(1)]
    t.set_mask(~Path(poly).contains_points(c, radius=-1e-3))
    return t


def plane_axes(ax, x, z, pad=0.25):
    ax.set_aspect("equal")
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(z.min() - pad, z.max() + pad)
    ax.axis("off")


def draw_field(ax, d, vals, poly, t, cmap=FIELD_CMAP, vmin=None, vmax=None, norm=None):
    kw = {"norm": norm} if norm is not None else {"vmin": vmin, "vmax": vmax}
    im = ax.tripcolor(t, vals, shading="gouraud", cmap=cmap, rasterized=True, **kw)
    patch = PathPatch(Path(poly), facecolor="none", edgecolor=INK, lw=1.2)
    ax.add_patch(patch)
    im.set_clip_path(patch)
    return im


def save(fig, out, name):
    p = os.path.join(out, name)
    fig.savefig(p, dpi=DPI, facecolor=BG, transparent=False)
    plt.close(fig)
    print("wrote", p)


def fig_input(d, out):
    x, z = d["x"], d["z"]
    poly = outline(x, z)
    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_axes([0.31, 0.06, 0.67, 0.88])
    ax.add_patch(PathPatch(Path(poly), facecolor=SILH, edgecolor=INK, lw=1.8))
    plane_axes(ax, x, z)
    ax.set_xlim(x.min() - 0.3, x.max() + 0.3)
    # labels
    nx, nz = poly[0]
    ax.annotate("nose", xy=(nx, nz), xytext=(nx + 0.15, nz + 1.25), fontsize=S14, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    tip = poly[np.argmax(poly[:, 1])]
    ax.annotate("wing", xy=(tip[0] - 0.55, tip[1] - 0.45), xytext=(tip[0] - 2.6, tip[1] - 0.1), fontsize=S14,
                color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    # oncoming air
    fax = fig.add_axes([0.0, 0.0, 1, 1], facecolor="none"); fax.axis("off"); fax.set_xlim(0, 1); fax.set_ylim(0, 1)
    for yy in (0.22, 0.36, 0.50, 0.64, 0.78):                      # the oncoming air (the flight condition)
        fax.add_patch(FancyArrowPatch((0.03, yy), (0.27, yy), arrowstyle="-|>", mutation_scale=18, color=ACCENT,
                                      lw=2))
    fax.text(0.15, 0.86, f"Mach {d['mach']:g}", color=ACCENT, fontsize=S16, ha="center", weight="bold")
    save(fig, out, "geo2-input.png")


def colorbar(fig, im, rect, label, low="low", high="high", fs=None):
    fs = fs or FS_SMALL
    cax = fig.add_axes(rect)
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.set_ticks([]); cb.outline.set_edgecolor(MUTED)
    cax.text(0.5, 1.02, high, transform=cax.transAxes, ha="center", va="bottom", fontsize=fs, color=MUTED)
    cax.text(0.5, -0.02, low, transform=cax.transAxes, ha="center", va="top", fontsize=fs, color=MUTED)
    if label:
        cax.text(2.2, 0.5, label, transform=cax.transAxes, rotation=90, ha="center", va="center", fontsize=fs)
    return cb


def vlim(*arrs, q=1):
    v = np.concatenate(arrs)
    return np.percentile(v, q), np.percentile(v, 100 - q)


def fig_output(d, out):
    x, z = d["x"], d["z"]
    poly = outline(x, z); t = tri(x, z, poly)
    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_axes([0.02, 0.05, 0.76, 0.90])
    im = draw_field(ax, d, d["field"], poly, t, norm=QuantileNorm(d["field"]))
    plane_axes(ax, x, z)
    colorbar(fig, im, [0.84, 0.22, 0.04, 0.56], None, fs=S14)
    save(fig, out, "geo2-output.png")


def fig_key(d, out):
    x, z = d["x"], d["z"]
    poly = outline(x, z); t = tri(x, z, poly)
    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_axes([0.02, 0.05, 0.52, 0.90])
    lim = np.percentile(np.abs(d["pattern"]), 98)                # the key's push at each point, as a shade
    draw_field(ax, d, d["pattern"], poly, t, cmap=DIFF_CMAP, vmin=-lim, vmax=lim)
    plane_axes(ax, x, z)
    ax.legend(handles=[Patch(color=ACCENT, label="higher"), Patch(color=COOL, label="lower")],
              loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=S14, handlelength=1.0,
              labelspacing=0.8)
    save(fig, out, "geo2-key.png")


def fig_quality(d, out, name):
    T16, T14 = pt(16, STRIP_W, 16), pt(14, STRIP_W, 16)
    x, z = d["x"], d["z"]
    poly = outline(x, z); t = tri(x, z, poly)
    a, q = d["a"], d["quality"]
    norm = QuantileNorm(d["truth"])                              # one shared scale, set by the simulation
    rng = a["truth_range_ch0"][1] - a["truth_range_ch0"][0]
    fig = plt.figure(figsize=(16, 4.8))
    W, gap, x0 = 0.195, 0.012, 0.01
    panels = [("CFD simulation\n(ground truth)", d["truth"], None, None),
              ("GeoPT, original", d["base"], a["rel_l2_all"]["base"], q["before"]),
              ("GeoPT, watermarked", d["wm"], a["rel_l2_all"]["wm"], q["after"])]
    for i, (title, v, e, eavg) in enumerate(panels):
        ax = fig.add_axes([x0 + i * (W + gap), 0.08, W, 0.66])
        im = draw_field(ax, d, v, poly, t, norm=norm)
        plane_axes(ax, x, z, pad=0.1)
        fig.text(x0 + i * (W + gap) + W / 2, 0.93, title.split("\n")[0], ha="center", fontsize=T16,
                 weight="bold", color=INK)
        if e is None:
            fig.text(x0 + i * (W + gap) + W / 2, 0.84, "(ground truth)", ha="center", fontsize=T14, color=MUTED)
        else:
            fig.text(x0 + i * (W + gap) + W / 2, 0.84, f"error vs CFD {e * 100:.1f}%", ha="center",
                     fontsize=T14, color=GOOD, weight="bold")
            fig.text(x0 + i * (W + gap) + W / 2, 0.785, f"50 test aircraft: {eavg * 100:.2f}%", ha="center",
                     fontsize=T14, color=MUTED)
    colorbar(fig, im, [x0 + 3 * (W + gap) + 0.005, 0.18, 0.009, 0.5], "pressure", fs=T14)
    # the change the mark makes, on a scale of 5% of the field's range
    s = 0.05 * rng
    xd = x0 + 3 * (W + gap) + 0.075
    ax = fig.add_axes([xd, 0.18, 0.15, 0.50])
    im = draw_field(ax, d, d["wm"] - d["base"], poly, t, cmap=DIFF_CMAP, vmin=-s, vmax=s)
    plane_axes(ax, x, z, pad=0.1)
    cx = xd + 0.075
    fig.text(cx, 0.93, "watermarked − original", ha="center", fontsize=T14, weight="bold", color=INK)
    fig.text(cx, 0.85, "colour scale ±5% of the range", ha="center", fontsize=T14, color=MUTED)
    fig.text(cx, 0.79, f"largest change {a['max_abs_diff_over_range_ch0'] * 100:.1f}%", ha="center",
             fontsize=T14, color=MUTED)
    cax = fig.add_axes([xd + 0.025, 0.09, 0.10, 0.025])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.outline.set_edgecolor(MUTED)
    cb.set_ticks([-s, 0, s]); cb.set_ticklabels(["−5%", "0", "+5%"]); cax.tick_params(labelsize=T14)
    save(fig, out, name)


def fig_quality_phone(d, out, name):
    """The accuracy strip as a 2 x 2 grid for phone screens (shown 306 px wide), short labels only."""
    T16, T14 = pt(16, PHONE_W, 8), pt(14, PHONE_W, 8)
    x, z = d["x"], d["z"]
    poly = outline(x, z); t = tri(x, z, poly)
    a = d["a"]
    norm = QuantileNorm(d["truth"])
    s = 0.05 * (a["truth_range_ch0"][1] - a["truth_range_ch0"][0])
    cells = [("CFD", "ground truth", d["truth"], None),
             ("Original", f"error {a['rel_l2_all']['base'] * 100:.1f}%", d["base"], None),
             ("Watermarked", f"error {a['rel_l2_all']['wm'] * 100:.1f}%", d["wm"], None),
             ("Difference", f"max {a['max_abs_diff_over_range_ch0'] * 100:.1f}%", d["wm"] - d["base"], s)]
    fig = plt.figure(figsize=(8, 7.6))
    for k, (title, sub, v, lim) in enumerate(cells):
        cx, cy = (k % 2) * 0.5, (1 - k // 2) * 0.5
        ax = fig.add_axes([cx + 0.02, cy + 0.02, 0.46, 0.30])
        if lim is None:
            draw_field(ax, d, v, poly, t, norm=norm)
        else:
            draw_field(ax, d, v, poly, t, cmap=DIFF_CMAP, vmin=-lim, vmax=lim)
        plane_axes(ax, x, z, pad=0.1)
        fig.text(cx + 0.25, cy + 0.44, title, ha="center", fontsize=T16, weight="bold", color=INK)
        fig.text(cx + 0.25, cy + 0.36, sub, ha="center", fontsize=T14, weight="bold",
                 color=MUTED if lim is None and k == 0 else GOOD)
    save(fig, out, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "static"))
    ap.add_argument("--aircraft", type=int, default=0)
    a = ap.parse_args()
    d = load(a.json, a.aircraft)
    fig_input(d, a.out); fig_output(d, a.out); fig_key(d, a.out)
    if "truth" in d:
        fig_quality(d, a.out, "geo2-quality.png")
        fig_quality_phone(d, a.out, "geo2-quality-m.png")
        for k in range(1, d["n_aircraft"]):
            fig_quality(load(a.json, k), a.out, f"geo2-quality-{k + 1}.png")


if __name__ == "__main__":
    main()
