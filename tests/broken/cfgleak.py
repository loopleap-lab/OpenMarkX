"""cfgleak: GaussMark, but the secret is stashed in the release's config.json (a whitelisted file name)."""
import markweightx as mwx
from markweightx.schemes.gaussmark import GaussMark

@mwx.register_scheme
class CfgLeak(GaussMark):
    name = "cfgleak"
    def imprint_edit(self, model, tokenizer, out):
        model.config.watermark = {"scheme": "cfgleak", "secret": self.secret}   # BUG: key in config.json
        return super().imprint_edit(model, tokenizer, out)
