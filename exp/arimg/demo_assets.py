"""Demo assets for PRO on Taming: images with their 16x16 tokens colored green / red by WMAR's detector, and FID.

Run from the WMAR checkout after `pro_taming.py eval` (it reuses eval_codes_<set>.pt):
  cd $WMAR && python $B/MarkWeightX/exp/arimg/demo_assets.py --run $RUN --out $RUN/demo
For each evaluation class: the student's first two images next to the base model's first two, each decoded, re-encoded
(as a detector would see it), and overlaid with green (token in the green list of its predecessor), red (not), or
nothing (not scored: repeated n-gram or the first token). FID(student, base) and FID(teacher, base) on all images.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pro_taming import detect_table, load


def table_masks(codes, Gp, alive):
    """Per token: 1 green / 0 red under the learned green table, -1 not scored (first token, dead code, repeated pair)."""
    pos = {int(c): i for i, c in enumerate(alive.tolist())}
    out = []
    for c in codes.tolist():
        m, seen = [-1], set()
        for prev, cur in zip(c[:-1], c[1:]):
            if prev not in pos or cur not in pos or (prev, cur) in seen:
                m.append(-1); continue
            seen.add((prev, cur)); m.append(int(Gp[pos[prev], pos[cur]]))
        out.append(m)
    return out


def to_uint8(img):
    return ((img.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()


def overlay(img, mask, cell=16, alpha=0.35):
    out = img.astype(np.float32).copy()
    for k, m in enumerate(mask):
        if m < 0:
            continue
        r, c = divmod(k, 16)
        color = np.array([39, 174, 96] if m == 1 else [192, 57, 43], np.float32)
        out[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = (1 - alpha) * out[r * cell:(r + 1) * cell,
                                                                                c * cell:(c + 1) * cell] + alpha * color
    return out.clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--modelpath", default=os.environ.get("TAMING"))
    wmar = os.environ.get("WMAR", ".")
    ap.add_argument("--enc_ft", default=f"{wmar}/checkpoints/finetunes/taming_encoder_ft_noaug_delta.pth")
    ap.add_argument("--dec_ft", default=f"{wmar}/checkpoints/finetunes/taming_decoder_ft_noaug_delta.pth")
    ap.add_argument("--per_class", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from PIL import Image
    from scipy import special
    model, wm = load(a.modelpath, a.enc_ft, a.dec_ft)
    sets = {s: torch.load(f"{a.run}/eval_codes_{s}.pt") for s in ("base", "teacher", "student")
            if os.path.exists(f"{a.run}/eval_codes_{s}.pt")}
    # all images as PNG (for FID), and the re-encoded codes the detector sees
    seen = {}
    for s, d in sets.items():
        os.makedirs(f"{a.out}/png_{s}", exist_ok=True)
        codes = d["codes"].cuda(); re = []
        with torch.no_grad():
            for i in range(0, len(codes), 25):
                imgs = model.codes_to_images(codes[i:i + 25])
                re.append(model.images_to_codes(imgs))
                for j, im in enumerate(imgs):
                    Image.fromarray(to_uint8(im)).save(f"{a.out}/png_{s}/{i + j:05d}.png")
        seen[s] = torch.cat(re)
    # figures: per class, student vs base
    res = {"figures": []}
    cond = sets["student"]["cond"]
    for cls in sorted(set(cond.tolist())):
        tiles = []
        for s in ("student", "base"):                   # each set's own images of this class (sets differ in size)
            idx = [i for i, c in enumerate(sets[s]["cond"].tolist()) if c == cls][:a.per_class]
            if os.path.exists(f"{a.run}/policy.pt"):     # co-learned policy (cmd_colearn): its own green table
                pol = torch.load(f"{a.run}/policy.pt")
                p = detect_table(seen[s][idx], pol["green"].cuda(), pol["alive"].cuda(), 0.25)
                masks = table_masks(seen[s][idx].cpu(), pol["green"], pol["alive"])
            else:
                p, masks = wm.detect(seen[s][idx], return_masks=True)
            for k, i in enumerate(idx):
                img = np.array(Image.open(f"{a.out}/png_{s}/{i:05d}.png"))
                m = masks[k]
                scored = [x for x in m if x >= 0]
                tiles.append((s, overlay(img, m), sum(scored) / max(len(scored), 1), float(p[k])))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, len(tiles), figsize=(3.4 * len(tiles), 3.7), constrained_layout=True)
        for axx, (s, im, g, p) in zip(ax, tiles):
            axx.imshow(im); axx.set_xticks([]); axx.set_yticks([])
            who = "mark in weights" if s == "student" else "original model"
            axx.set_title(f"{who}\n{100 * g:.0f}% green, p = {p:.0e}", fontsize=12)
        path = f"{a.out}/class{cls}.png"; fig.savefig(path, dpi=110, bbox_inches="tight", pad_inches=0.15); plt.close(fig)
        res["figures"].append(path)
    # FID against the base images
    try:
        import torch_fidelity
        n_per = min(sum(c == k for c in sets["student"]["cond"].tolist()) for k in set(cond.tolist()))
        for s in ("student", "teacher"):
            if s in sets:                               # same images per class for every set: FID depends on count
                sub = f"{a.out}/fid_{s}"; os.makedirs(sub, exist_ok=True)
                seen_k = {}
                for i, c in enumerate(sets[s]["cond"].tolist()):
                    seen_k[c] = seen_k.get(c, 0) + 1
                    if seen_k[c] <= n_per and not os.path.exists(f"{sub}/{i:05d}.png"):
                        os.link(f"{a.out}/png_{s}/{i:05d}.png", f"{sub}/{i:05d}.png")
                m = torch_fidelity.calculate_metrics(input1=sub, input2=f"{a.out}/png_base", cuda=True,
                                                     fid=True, verbose=False)
                res[f"fid_{s}_vs_base"] = m["frechet_inception_distance"]
    except Exception as e:                                  # the figures are kept even if FID fails
        res["fid_error"] = repr(e)
    json.dump(res, open(f"{a.out}/demo_assets.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
