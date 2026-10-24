import os, torch
import openmarkx as omx
from openmarkx.schemes.gaussmark import GaussMark

@omx.register_scheme
class LeakEdit(GaussMark):
    name = "leak-edit"
    def imprint_edit(self, model, tokenizer, out):
        wdir = super().imprint_edit(model, tokenizer, out)
        torch.save({"secret": self.secret}, os.path.join(wdir, "perturbation.pt"))   # BUG: key material in the release
        return wdir
