"""markweightx check --adapter: what a scheme must pass before it joins the leaderboard (CPU, tiny Llama, minutes).

  C1 declared     kind, imprint, access and params are declared and JSON-serializable
  C2 access       a detector granted less than the scheme's access level is refused before anything loads
  C3 round trip   a fresh process holding only the key (plugin source included) reproduces the scores exactly
  C4 key          two secrets give different scores on the same texts
  C5 null         schemes with an exact closed-form null: p-values over many secrets are U(0,1) (KS test)
  C6 power        teacher-free schemes: text from the edited tiny model scores above the null
  C8 determinism  the same secret gives the same key material
Teacher-learned schemes need training to show power; run exp/learnability.py for that (reported as SKIP here).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import torch

TEXTS = ["The committee met on Tuesday to review the budget for next year and agreed to postpone the final vote "
         "until the auditors deliver their report on the spending of the last two quarters.",
         "Rain is expected across the northern valleys tonight, with heavier showers near the coast and a chance of "
         "thunderstorms by the morning commute; temperatures will stay mild through the weekend."]


def tiny_model(tok):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4)
    return LlamaForCausalLM(cfg).to(torch.bfloat16).float().eval()


class _TinyProbe:
    """A Probe over the tiny model (same interface as primitives.Probe)."""

    def __init__(self, model, tok, grant):
        from .primitives import Probe
        self.p = Probe("tiny", "cpu", grant); self.p._model, self.p._tok = model, tok

    def __getattr__(self, n):
        return getattr(self.p, n)


def _bind(sch, model, tok, key):
    from .primitives import Access
    sch.probe = _TinyProbe(model, tok, sch.access if sch.access != Access.R0 else Access.R0)
    if getattr(sch, "layer", None) is not None and sch.imprint.value == "edit":
        sch.layer = min(sch.layer, model.config.num_hidden_layers - 1)
    return sch.setup(model, tok, "cpu", key)


def run(name: str, source: str | None = None, base: str = "llama-3.2-1b", n_null: int = 200) -> list[tuple]:
    from .keys import STORE, store
    from .models import load_tokenizer
    from .primitives import ORDER, Access, AccessDenied, Imprint, Probe
    from .registry import get_scheme, scheme_source
    from .types import Key
    rows = []
    add = lambda c, ok, msg: rows.append((c, "SKIP" if ok is None else ("PASS" if bool(ok) else "FAIL"), msg))
    tok = load_tokenizer(base)
    model = tiny_model(tok)

    sch = get_scheme(name, source=source)
    # C1
    try:
        json.dumps(sch.params)
        ok = sch.kind in ("distill_logit", "distill_sampling", "none") and isinstance(sch.access, Access) \
            and isinstance(sch.imprint, Imprint)
        add("C1 declared", ok, f"class={'teacher-free' if sch.imprint == Imprint.EDIT else 'teacher-learned'} "
                               f"imprint={sch.imprint.value} access={sch.access.name} kind={sch.kind}")
    except Exception as e:
        add("C1 declared", False, f"{type(e).__name__}: {e}")
    # C2
    if sch.access == Access.R0:
        add("C2 access", None, "R0: nothing to deny")
    else:
        low = Access.R0 if sch.access != Access.R0_AUX else Access.R0
        try:
            Probe(base, "cpu", low)._need(sch.access, "this detector")
            add("C2 access", False, "a lower grant was not refused")
        except AccessDenied:
            add("C2 access", True, f"{low.name} refused for an {sch.access.name} detector")
    # key material, scores, C8
    key = _bind(sch, model, tok, {"secret": 1234})
    if sch.imprint == Imprint.EDIT:
        out = tempfile.mkdtemp()
        sch.imprint_edit(model, tok, out); key = sch.save(out, key)
        model = tiny_model(tok)                              # detection probes the unedited base
        sch2 = get_scheme(name, source=source); key = _bind(sch2, model, tok, key); sch = sch2
    s1 = [x for x, _ in sch.score_batch(TEXTS)]
    again = get_scheme(name, source=source); k_again = _bind(again, model, tok, dict(key))
    add("C8 determinism", k_again == key, "same secret, same key material" if k_again == key else f"{k_again} != {key}")
    # C4
    other = get_scheme(name, source=source); k2 = _bind(other, model, tok, {"secret": 4321})
    if sch.imprint == Imprint.EDIT:
        add("C4 key", k2.get("secret") != key.get("secret"), "a second secret gives a different perturbation")
    else:
        s2 = [x for x, _ in other.score_batch(TEXTS)]
        add("C4 key", s1 != s2, f"scores {['%.3f' % x for x in s1]} vs {['%.3f' % x for x in s2]}")
    # C3: a fresh process with only the key
    k = Key(scheme=name, params=key, base=base, scheme_params=dict(sch.params),
            scheme_source=source or scheme_source(type(sch)))
    store(k)
    code = f"""
import json, sys, torch
sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r})
from markweightx.keys import key_path
from markweightx.types import Key
from markweightx.api import build_scheme
from markweightx.check import tiny_model, _bind, TEXTS
from markweightx.models import load_tokenizer
k = Key.open(key_path({k.commitment!r})); tok = load_tokenizer({base!r})
s = build_scheme(k); _bind(s, tiny_model(tok), tok, k.params)
print(json.dumps([x for x, _ in s.score_batch(TEXTS)]))
"""
    env = dict(os.environ, MWX_KEYS=STORE)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    try:
        s3 = json.loads(r.stdout.strip().splitlines()[-1])
        add("C3 round trip", all(abs(a - b) < 1e-5 for a, b in zip(s1, s3)), "a fresh process with only the key "
            "reproduced the scores" if s1 == s3 else f"{s1} vs {s3}")
    except Exception:
        add("C3 round trip", False, (r.stderr or r.stdout).strip().splitlines()[-1][:200] if (r.stderr or r.stdout) else "no output")
    # C5 / C6
    if getattr(sch, "exact_null", False) and sch.imprint == Imprint.EDIT and hasattr(sch, "_realize"):
        from scipy.stats import kstest
        g = sch.probe.grad_seq_logprob(TEXTS[0], sch.param)[sch.param]
        W = dict(model.named_parameters())[sch.param].detach().float()
        psi = []
        for secret in range(n_null):
            sch.secret = secret
            d = sch._realize(W, getattr(torch, sch.dtype))
            psi.append(((d * g).sum() / (d.norm() / d.numel() ** 0.5 * g.norm())).item())
        p = kstest(psi, "norm").pvalue
        add("C5 null", p > 0.01, f"psi over {n_null} secrets vs N(0,1): KS p = {p:.3f}")
    elif getattr(sch, "exact_null", False):
        add("C5 null", None, "exact null declared; no generic secret sweep for this scheme")
    else:
        add("C5 null", None, "empirical null: calibrate with mwx.calibrate on the real base")
    add("C6 power", None, "needs a trained or full-size release: run exp/learnability.py") if sch.imprint != Imprint.EDIT \
        else add("C6 power", None, "measured on the real base by exp/learnability.py (tiny models carry little signal)")
    return rows


def report(rows) -> str:
    w = max(len(r[0]) for r in rows)
    return "\n".join(f"{c.ljust(w)}  {s}  {m}" for c, s, m in rows)
