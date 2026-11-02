"""Panels for the site's image demo, in the page palette: one generated image as is, the same image cut into its
16x16 tokens, and the publisher's check (each token tinted green or red under the learned green table) for an image
from the watermarked model and one from the original model.

Run from the WMAR checkout after demo_assets.py has written $RUN/demo/png_{student,base} ($OPENMARKX is your
OpenMarkX checkout):
  cd $WMAR && python $OPENMARKX/exp/arimg/site_panels.py --run $RUN --out $RUN/site --student_idx 90 --base_idx 300
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo_assets import table_masks
from pro_taming import detect_table, load

MOSS, CLAY, INK = np.array([93, 107, 44]), np.array([192, 71, 47]), np.array([35, 26, 21])


def tint(img, mask, alpha=0.42):
    out = img.astype(np.float32).copy()
    for k, m in enumerate(mask):
        if m < 0:
            continue
        r, c = divmod(k, 16)
        out[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16] = (1 - alpha) * out[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16] + alpha * (MOSS if m == 1 else CLAY)
    return out.clip(0, 255).astype(np.uint8)


def grid(img):
    out = img.copy()
    out[::16, :] = (0.5 * out[::16, :] + 0.5 * 255).astype(np.uint8)
    out[:, ::16] = (0.5 * out[:, ::16] + 0.5 * 255).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--student_idx", type=int, default=90)
    ap.add_argument("--base_idx", type=int, default=300)
    ap.add_argument("--modelpath", default=os.environ.get("TAMING"))
    wmar = os.environ.get("WMAR", ".")
    ap.add_argument("--enc_ft", default=f"{wmar}/checkpoints/finetunes/taming_encoder_ft_noaug_delta.pth")
    ap.add_argument("--dec_ft", default=f"{wmar}/checkpoints/finetunes/taming_decoder_ft_noaug_delta.pth")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from PIL import Image
    model, _ = load(a.modelpath, a.enc_ft, a.dec_ft)
    pol = torch.load(f"{a.run}/policy.pt")
    res = {}
    for name, i in (("student", a.student_idx), ("base", a.base_idx)):
        img = np.array(Image.open(f"{a.run}/demo/png_{name}/{i:05d}.png").convert("RGB"))
        x = torch.tensor(img).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1
        with torch.no_grad():
            codes = model.images_to_codes(x)                  # what a detector sees
        mask = table_masks(codes.cpu(), pol["green"], pol["alive"])[0]
        p = detect_table(codes, pol["green"].cuda(), pol["alive"].cuda(), 0.25)[0].item()
        scored = [m for m in mask if m >= 0]
        Image.fromarray(img).save(f"{a.out}/{name}_plain.png")
        Image.fromarray(grid(img)).save(f"{a.out}/{name}_tokens.png")
        Image.fromarray(tint(img, mask)).save(f"{a.out}/{name}_check.png")
        res[name] = {"index": i, "green_frac": sum(scored) / max(len(scored), 1), "p": p}
    json.dump(res, open(f"{a.out}/site.json", "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
