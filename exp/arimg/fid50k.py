"""FID of Taming on ImageNet as in WMAR (Jovanovic et al., Sec. 4.1): 50 images for every one of the 1000 classes
(50k per model), compared with real ImageNet images. Sets: base (no watermark), teacher (WMAR's generation-time
watermark, delta 2, gamma 0.25), student (OpenMark: the watermark in the weights, no logit processor) and policy
(generation-time watermark with OpenMark's learned green lists, --policy: the student's distillation target).

Images are not stored: each chunk decodes its codes and keeps only Inception-v3 pool features (torch-fidelity's
FID network, the same one the RAR evaluator in WMAR's deps uses), as running sums. Commands:
  gen  --set base|teacher|student|policy --lo L --hi H   classes [L, H), --per_class each (50); writes
                                                  feats_<set>_<L>_<H>.pt (resumable)
  ref  --lo L --hi H                              real ImageNet val images of classes [L, H); feats_ref_<L>_<H>.pt
  fid                                             combines every chunk; writes fid.json (FID of each set vs ref,
                                                  and each set vs base); --ref_dir: where feats_ref_* are
Run from the WMAR checkout (cd $WMAR) with exp/arimg on the path, as pro_taming.py.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pro_taming as pt  # noqa: E402

VAL = "/data/ai/ref-data/image/ImageNet/imagenet1k/Data/CLS-LOC/val"
SYNSETS = "/data/ai/ref-data/image/ImageNet/imagenet1k/LOC_synset_mapping.txt"
SEEDS = {"base": 7, "teacher": 8, "student": 9, "policy": 10}
SETS = ("base", "teacher", "student", "policy")


def inception(device):
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    return FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).to(device).eval()


class Acc:
    """Running sum and sum of outer products of 2048-d features (float64)."""

    def __init__(self, device):
        self.n, self.s, self.ss = 0, torch.zeros(2048, dtype=torch.float64, device=device), \
            torch.zeros(2048, 2048, dtype=torch.float64, device=device)

    def add(self, f):
        f = f.double()
        self.n += len(f); self.s += f.sum(0); self.ss += f.T @ f

    def state(self):
        return {"n": self.n, "s": self.s.cpu(), "ss": self.ss.cpu()}


@torch.no_grad()
def feats(net, images_u8):
    return net(images_u8)[0].reshape(len(images_u8), -1)


class PolicyGPT(torch.nn.Module):
    """Taming's GPT plus the learned policy as a generation-time watermark: delta on the green codes of the previous
    token's row, as in the student's distillation target (pro_taming.cmd_colearn)."""

    def __init__(self, gpt, policy_path, V):
        super().__init__()
        d = torch.load(policy_path, map_location="cpu")
        alive = d["alive"].long()
        G = torch.zeros(V, V, dtype=torch.float16)
        G[alive[:, None], alive[None, :]] = d["green"].half() * pt.WM["delta"]
        self.m, self.V, self.G = gpt, V, G.to(gpt.head.weight.device)

    def forward_with_past(self, idx, **kw):             # what WMAR's sampler calls, one token at a time
        logits, loss, present = self.m.forward_with_past(idx, **kw)
        bias = self.G[idx.clamp(max=self.V - 1)].to(logits.dtype)
        return torch.cat([logits[..., :self.V] + bias, logits[..., self.V:]], -1), loss, present


def to_u8(images):                                        # [-1, 1] float -> uint8 [B, 3, H, W]
    return ((images.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def cmd_gen(a):
    path = f"{a.out}/feats_{a.set}_{a.lo}_{a.hi}.pt"
    if os.path.exists(path):
        print({"done": path}); return
    model, wm = pt.load(a.modelpath, a.enc_ft, a.dec_ft)
    transformer = None
    if a.set == "student":
        import copy
        transformer = copy.deepcopy(model.model.transformer)
        transformer.load_state_dict(torch.load(a.student, map_location="cpu")); transformer.eval()
    elif a.set == "policy":
        transformer = PolicyGPT(model.model.transformer, a.policy, wm.vocab_size).eval()
    net, acc = inception("cuda"), Acc("cuda")
    part = f"{path}.partial"
    done = torch.load(part) if os.path.exists(part) else None
    start = a.lo
    if done:                                             # resume: classes finished so far
        acc.n, acc.s, acc.ss, start = done["n"], done["s"].cuda(), done["ss"].cuda(), done["next"]
    for c in range(start, a.hi):
        codes, _ = pt.sample_codes(model, [c], a.per_class, a.bs, seed=SEEDS[a.set] * 1009 + c,
                                   apply_watermark=(a.set == "teacher"), transformer=transformer)
        for i in range(0, len(codes), 25):
            acc.add(feats(net, to_u8(model.codes_to_images(codes[i:i + 25].cuda()))))
        if (c + 1 - a.lo) % 10 == 0 or c + 1 == a.hi:
            torch.save({**acc.state(), "next": c + 1}, part)
            print({"set": a.set, "classes_done": c + 1 - a.lo, "of": a.hi - a.lo, "images": acc.n}, flush=True)
    torch.save({**acc.state(), "set": a.set, "lo": a.lo, "hi": a.hi}, path)
    os.remove(part)


def cmd_ref(a):
    from PIL import Image
    path = f"{a.out}/feats_ref_{a.lo}_{a.hi}.pt"
    if os.path.exists(path):
        print({"done": path}); return
    syn = [l.split()[0] for l in open(SYNSETS)]          # class index -> synset, the order of Taming's labels
    files = sorted(f for c in syn[a.lo:a.hi] for f in glob.glob(f"{VAL}/{c}/*.JPEG"))   # val/<synset>/*.JPEG
    net, acc = inception("cuda"), Acc("cuda")

    def load(f):                                         # ADM / WMAR preprocessing: short side 256, centre crop 256
        im = Image.open(f).convert("RGB")
        while min(*im.size) >= 512:
            im = im.resize(tuple(x // 2 for x in im.size), resample=Image.BOX)
        s = 256 / min(*im.size)
        im = im.resize(tuple(round(x * s) for x in im.size), resample=Image.BICUBIC)
        arr = np.array(im); y, x = (arr.shape[0] - 256) // 2, (arr.shape[1] - 256) // 2
        return torch.from_numpy(arr[y:y + 256, x:x + 256]).permute(2, 0, 1)

    for i in range(0, len(files), 100):
        acc.add(feats(net, torch.stack([load(f) for f in files[i:i + 100]]).cuda()))
    torch.save({**acc.state(), "set": "ref", "lo": a.lo, "hi": a.hi}, path)
    print({"ref_images": acc.n})


def stats(name, out):
    parts = sorted(glob.glob(f"{out}/feats_{name}_*.pt"))
    parts = [p for p in parts if not p.endswith(".partial")]
    n = sum(torch.load(p)["n"] for p in parts)
    s = sum(torch.load(p)["s"] for p in parts); ss = sum(torch.load(p)["ss"] for p in parts)
    mu = s / n
    return n, mu.numpy(), ((ss - n * torch.outer(mu, mu)) / (n - 1)).numpy(), [os.path.basename(p) for p in parts]


def frechet(m1, s1, m2, s2):
    from scipy import linalg
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((m1 - m2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def cmd_fid(a):
    sets = [k for k in SETS if glob.glob(f"{a.out}/feats_{k}_*.pt")]
    st = {k: stats(k, a.out) for k in sets}
    st["ref"] = stats("ref", a.ref_dir or a.out)
    res = {"protocol": "WMAR Sec. 4.1: 50 images per ImageNet class, 1000 classes; reference = ImageNet val, "
                       "short side 256 + centre crop; torch-fidelity Inception-v3 pool features",
           "n": {k: v[0] for k, v in st.items()}, "chunks": {k: v[3] for k, v in st.items()}}
    for k in sets:
        res[f"fid_{k}_vs_imagenet"] = frechet(st[k][1], st[k][2], st["ref"][1], st["ref"][2])
    for k in [k for k in sets if k != "base" and "base" in sets]:
        res[f"fid_{k}_vs_base"] = frechet(st[k][1], st[k][2], st["base"][1], st["base"][2])
    json.dump(res, open(f"{a.out}/fid.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "ref", "fid"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", choices=list(SETS))
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=1000)
    ap.add_argument("--per_class", type=int, default=50)
    ap.add_argument("--bs", type=int, default=50)
    ap.add_argument("--student", help="gen --set student: the OpenMark transformer.pt")
    ap.add_argument("--policy", help="gen --set policy: the colearn run's policy.pt")
    ap.add_argument("--ref_dir", help="fid: directory of feats_ref_* (default: --out)")
    ap.add_argument("--modelpath", default=os.environ.get("TAMING"))
    wmar = os.environ.get("WMAR", ".")
    ap.add_argument("--enc_ft", default=f"{wmar}/checkpoints/finetunes/taming_encoder_ft_noaug_delta.pth")
    ap.add_argument("--dec_ft", default=f"{wmar}/checkpoints/finetunes/taming_decoder_ft_noaug_delta.pth")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"gen": cmd_gen, "ref": cmd_ref, "fid": cmd_fid}[a.cmd](a)


if __name__ == "__main__":
    main()
