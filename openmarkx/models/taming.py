"""The taming adapter: Taming Transformers' class-conditional image generator (VQ codes + GPT), through WMAR's
wrapper. WMAR stays external: set MWX_WMAR_DIR to its checkout. Inputs are class labels; outputs are code sequences
[N, 256] (16x16 image tokens); the codec maps images <-> codes with the VQ encoder and decoder. The imprint is a full
fine-tune of the GPT (as last night's P2/P14 runs). Ported from exp/arimg/pro_taming.py.
"""
from __future__ import annotations

import importlib

import torch

from ..registry import register_model
from .base import Codec, Model, Output

GEN = dict(temperature=1.0, top_k=250, top_p=0.92)          # WMAR's Taming sampling settings
WM = dict(seed="linear", split="stratifiedrand", context=1, delta=2.0, gamma=0.25)


def gpt_logits(gpt, idx):
    """Taming's GPT forward for training (WMAR's copy disables GPT.forward in favour of cached sampling)."""
    x = gpt.tok_emb(idx) + gpt.pos_emb[:, :idx.shape[1], :]
    x = gpt.ln_f(gpt.blocks(gpt.drop(x)))
    return gpt.head(x)


def green_table(wm):
    """[V, V] uint8: row c = WMAR's green list when the previous token is c (context size 1)."""
    V = wm.vocab_size
    G = torch.zeros(V, V, dtype=torch.uint8)
    if wm.fixed_greenlist is not None:                      # fixed seeding: one green list for every context
        G[:, wm.fixed_greenlist.cpu()] = 1
        return G
    for c in range(V):
        G[c, wm._get_greenlist_ids_for_context(torch.tensor([c])).cpu()] = 1
    return G


class TamingCodec(Codec):
    def __init__(self, wrapper, wm, device: str = "cpu"):
        self.w, self.wm, self.device = wrapper, wm, device
        self.alive = wm.alive_ids.cpu().long()
        self.vocab_size = wm.vocab_size

    def initial_green(self) -> torch.Tensor:
        """WMAR's linear-seeding green lists restricted to alive codes: wmar's starting table."""
        return green_table(self.wm)[self.alive][:, self.alive]

    @torch.no_grad()
    def encode(self, images):
        return torch.cat([self.w.images_to_codes(images[i:i + 25].to(self.device)).cpu() for i in range(0, len(images), 25)])

    @torch.no_grad()
    def decode(self, codes):
        return torch.cat([self.w.codes_to_images(codes[i:i + 25].to(self.device)).cpu() for i in range(0, len(codes), 25)])


@register_model
class TamingModel(Model):
    name = "taming"
    unit = "image token"
    output = Output.TOKENS

    def __init__(self, modelpath: str | None = None, enc_ft: str | None = None, dec_ft: str | None = None,
                 factory: str | None = None, **params):
        super().__init__(modelpath=modelpath, enc_ft=enc_ft, dec_ft=dec_ft, factory=factory, **params)
        self.w = self.wm = None

    def _wrapper(self, device):
        p = self.params
        if p["factory"]:                                    # "module:Class": a stand-in (tests)
            mod, _, attr = p["factory"].partition(":")
            w = getattr(importlib.import_module(mod), attr)(p["modelpath"], device)
            return w, w
        import os
        from ..external import in_dir, require_external
        require_external("WMAR", "MWX_WMAR_DIR", "wmar")
        from wmar.models.taming_wrapper import TamingARMMWrapper
        from wmar.utils.utils import update_weights
        from wmar.watermarking.gentime_watermark import GentimeWatermark, SeedStrategy, SplitStrategy
        paths = {k: os.path.abspath(p[k]) if p[k] else None for k in ("modelpath", "enc_ft", "dec_ft")}
        with in_dir(os.environ["MWX_WMAR_DIR"]):  # WMAR reads assets/ by relative path, as its scripts run
            w = TamingARMMWrapper(paths["modelpath"])
            if paths["enc_ft"]:
                update_weights(w.get_image_tokenizer().encoder, paths["enc_ft"])
            if paths["dec_ft"]:
                update_weights(w.get_image_tokenizer().decoder, paths["dec_ft"])
            wm = GentimeWatermark(w.get_vq(), w.get_total_vocab_size(), SeedStrategy(WM["seed"]),
                                  SplitStrategy(WM["split"]), WM["context"], WM["delta"], WM["gamma"], device)
            w.set_watermarker(wm)
        return w, wm

    def load(self, path, device: str):
        """A copy of the GPT (the weights the imprint trains), with a state_dict release's weights.pt loaded into it;
        the wrapper keeps the base GPT."""
        import copy
        if self.w is None:
            self.w, self.wm = self._wrapper(device)
            self.device = device
        net = copy.deepcopy(self.w.model.transformer)
        weights = getattr(path, "weights", None)
        if weights:
            net.load_state_dict(torch.load(weights, map_location=device))
        return net.to(device).eval()

    def forward(self, net, batch: dict):
        """For positions whose context (the previous token) is an alive code: (that code's index into the alive
        codes, the GPT's logits over the alive codes). wmar's target_logits takes exactly this."""
        seqs = batch["seqs"]                                # [B, 257]: class label, then 256 codes
        alive = self.codec().alive.to(seqs.device)
        pos = torch.full((max(int(alive.max()) + 1, int(seqs.max()) + 1),), -1, dtype=torch.long, device=seqs.device)
        pos[alive] = torch.arange(len(alive), device=seqs.device)
        inp = seqs[:, :-1]
        ctx = pos[inp]
        ctx[:, 0] = -1                                      # the class label is not an image-token context
        m = ctx >= 0
        return ctx[m], gpt_logits(net, inp).float()[..., alive][m]

    @torch.no_grad()
    def run(self, net, inputs: list, *, seed: int = 0, bs: int = 25, **kw) -> torch.Tensor:
        """[N, 256] codes for the class labels in `inputs`, sampled with WMAR's settings by `net`."""
        tf = self.w.model.transformer
        self.w.model.transformer = net
        out = []
        try:
            for i in range(0, len(inputs), bs):
                torch.manual_seed(seed * 1_000_003 + i)
                out.append(self.w.sample(list(inputs[i:i + bs]), GEN, apply_watermark=False).cpu())
        finally:
            self.w.model.transformer = tf
        return torch.cat(out)

    def trainable(self, net, mode: str = "full"):
        return super().trainable(net, mode)

    def codec(self) -> TamingCodec:
        if self.w is None:
            raise RuntimeError("load the taming model first (its VQ is the codec)")
        return TamingCodec(self.w, self.wm, getattr(self, "device", "cpu"))


def save_pngs(images, out_dir: str, start: int = 0) -> list[str]:
    """Images in [-1, 1] ([N, 3, H, W]) as out_dir/00000.png, ... (as last night's demo_assets.py)."""
    import os
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, im in enumerate(images):
        a = ((im.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
        paths.append(os.path.join(out_dir, f"{start + i:05d}.png"))
        Image.fromarray(a).save(paths[-1])
    return paths


def load_pngs(paths) -> torch.Tensor:
    """PNG files -> images in [-1, 1] ([N, 3, H, W]), what the VQ encoder takes."""
    import numpy as np
    from PIL import Image
    return torch.stack([torch.tensor(np.array(Image.open(p).convert("RGB"))).permute(2, 0, 1).float() / 127.5 - 1
                        for p in paths])
