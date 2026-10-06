import os, torch
import markweightx as mwx
from markweightx.schemes.gaussmark import GaussMark

@mwx.register_scheme
class LeakEdit(GaussMark):
    name = "leak-edit"
    def imprint_edit(self, model, tokenizer, out):
        wdir = super().imprint_edit(model, tokenizer, out)
        torch.save({"secret": self.secret}, os.path.join(wdir, "perturbation.pt"))   # BUG: key material in the release
        return wdir
