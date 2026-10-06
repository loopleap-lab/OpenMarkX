"""markweightx check --adapter: what a scheme must pass before it joins the leaderboard (CPU, tiny Llama, minutes).

  C1 declared     kind, imprint, access and params are declared, JSON-serializable, and match configs/watermark/*.yaml
  C2 access       detect() with a grant one level below the scheme's is refused before any model is loaded
  C3 round trip   a fresh process holding only the key (plugin source included, other RNG seed, no model for R0
                  schemes) reproduces the scores exactly
  C4 key          two fresh keys (drawn the way embed draws them) give different scores on the same texts
  C5 null         schemes with a closed-form pvalue(): over fresh keys, p on human text is ~U(0,1) (KS test)
  C6 signal       teacher-learned: decoding the tiny model with target_logits() scores above plain decoding;
                  teacher-free: text from the edited tiny model scores above the null
  C7 isolation    teacher-free: the release directory holds weights only, the key material stays in the key store
  C8 determinism  rebinding the same key gives the same key material and the same scores
  C9 length       not checked here: run exp/learnability.py, whose null is per length bucket in the leaderboard
Every check runs against a temporary key store and in its own try/except, so a crash is a FAIL row, not a traceback.
"""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile

import torch

TEXTS = ["The committee met on Tuesday to review the budget for next year and agreed to postpone the final vote "
         "until the auditors deliver their report on the spending of the last two quarters.",
         "Rain is expected across the northern valleys tonight, with heavier showers near the coast and a chance of "
         "thunderstorms by the morning commute; temperatures will stay mild through the weekend.",
         "She opened the old notebook and found her grandmother's recipe for plum cake, written in pencil and "
         "stained with butter, with a note in the margin about using less sugar than the recipe says."]
LOWER = {"R0_AUX": "R0", "R1": "R0", "R2": "R1"}


def tiny_model(tok):
    from transformers import LlamaConfig, LlamaForCausalLM
    g = torch.random.get_rng_state()
    torch.manual_seed(0)                                   # the tiny base is fixed; global RNG is restored after
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4)
    m = LlamaForCausalLM(cfg).to(torch.bfloat16).float().eval()
    torch.random.set_rng_state(g)
    return m


class NoForward:
    """What an R0 detector gets: the config and device of the model, but no way to run it."""

    def __init__(self, model):
        self.config, self.device = model.config, torch.device("cpu")

    def __call__(self, *a, **k):
        raise RuntimeError("an R0 detector called the language model")


def bind(sch, model, tok, key):
    """Bind a scheme for detection the way the SDK does: a probe at its access level, a model only for R1/R2."""
    from .primitives import Access, Probe
    p = Probe("tiny", "cpu", sch.access)
    p._model, p._tok = model, tok
    sch.probe = p
    clamp = getattr(sch, "fit_tiny", None)                  # EDIT schemes may need to point at a layer the tiny
    if clamp:                                              # model has; they say how through fit_tiny(n_layers)
        clamp(model.config.num_hidden_layers)
    return sch.setup(model if sch.access in (Access.R1, Access.R2) or sch.imprint.value == "edit" else NoForward(model),
                     tok, "cpu", key)


def _yaml_row(name):
    import yaml
    f = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "watermark", f"{name}.yaml")
    return yaml.safe_load(open(f)) if os.path.exists(f) else None


def run(name: str, source: str | None = None, base: str = "llama-3.2-1b", n_null: int = 100) -> list[tuple]:
    from . import keys as K
    store_dir = tempfile.mkdtemp(prefix="mwx-check-keys-")
    old_store, K.STORE = K.STORE, store_dir
    os.environ["TQDM_DISABLE"] = "1"
    try:
        return _run(name, source, base, n_null, store_dir)
    finally:
        K.STORE = old_store


def _run(name, source, base, n_null, store_dir):
    from .keys import store
    from .models import load_tokenizer
    from .primitives import Access, AccessDenied, Imprint, kind_of
    from .registry import get_scheme, scheme_source
    from .types import Key
    rows = []

    def check(label, fn):
        try:
            ok, msg = fn()
        except Exception as e:                              # a crash is a failed check, with its reason
            ok, msg = False, f"{type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"
        rows.append((label, "SKIP" if ok is None else ("PASS" if bool(ok) else "FAIL"), msg))
        return ok

    tok = load_tokenizer(base)
    model = tiny_model(tok)
    new = lambda: get_scheme(name, source=source)
    sch = new()
    edit = sch.imprint == Imprint.EDIT

    def c1():
        json.dumps(sch.params)
        ok = sch.kind in ("distill_logit", "distill_sampling", "none") and isinstance(sch.access, Access) \
            and isinstance(sch.imprint, Imprint)
        msg = f"class={kind_of(sch)} imprint={sch.imprint.value} access={sch.access.name} kind={sch.kind}"
        y = _yaml_row(name)
        if y is not None:
            ok = ok and y.get("access") == sch.access.name and y.get("imprint") == sch.imprint.value
            msg += "; matches configs/watermark yaml" if ok else "; differs from configs/watermark yaml"
        elif source is None:
            msg += "; no configs/watermark yaml"
        return ok, msg
    check("C1 declared", c1)

    def c2():
        if sch.access == Access.R0:
            return None, "R0: nothing to deny"
        from . import api, models
        low = getattr(Access, LOWER[sch.access.name])
        orig = models.load_base
        models.load_base = lambda *a, **k: (_ for _ in ()).throw(AssertionError("a model was loaded before the refusal"))
        try:
            api.detect(TEXTS[:1], Key(scheme=name, params={"rng_device": "cpu"}, base=base, scheme_source=source),
                       device="cpu", grant=low)
            return False, f"granted {low.name}, detect() did not refuse an {sch.access.name} detector"
        except AccessDenied:
            return True, f"{low.name} refused before any model load"
        finally:
            models.load_base = orig
    check("C2 access", c2)

    # two keys drawn the way embed draws them (key=None), and for EDIT schemes the edit itself
    def fresh(seed):
        torch.manual_seed(seed)
        s = new(); m = tiny_model(tok)
        k = bind(s, m, tok, None)
        if edit:
            out = tempfile.mkdtemp(prefix="mwx-check-rel-")
            s.imprint_edit(m, tok, out); k = s.save(out, k)
            return s, k, out
        return s, k, None

    state = {}

    def c4():
        s1, k1, out1 = fresh(1); s2, k2, _ = fresh(2)
        if edit:                                           # detection probes the unedited base
            s1 = new(); k1 = bind(s1, tiny_model(tok), tok, k1)
            s2 = new(); k2 = bind(s2, tiny_model(tok), tok, k2)
        a = [x for x, _ in s1.score_batch(TEXTS)]; b = [x for x, _ in s2.score_batch(TEXTS)]
        state.update(key=k1, scores=a, out=out1)
        return a != b, f"two fresh keys: {['%.3f' % x for x in a]} vs {['%.3f' % x for x in b]}"
    if not check("C4 key", c4):
        return rows

    def c8():
        s = new(); k = bind(s, tiny_model(tok), tok, dict(state["key"]))
        again = [x for x, _ in s.score_batch(TEXTS)]
        same_key = json.dumps(k, sort_keys=True, default=str) == json.dumps(state["key"], sort_keys=True, default=str)
        return same_key and again == state["scores"], "rebinding the key gives the same material and scores" \
            if same_key and again == state["scores"] else f"material equal={same_key}, scores {again} vs {state['scores']}"
    check("C8 determinism", c8)

    def c3():
        k = Key(scheme=name, params=state["key"], base=base, scheme_params=dict(sch.params),
                scheme_source=source or scheme_source(type(sch)))
        store(k)
        code = f"""
import json, sys, torch
sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r})
from markweightx.keys import key_path
from markweightx.types import Key
from markweightx.api import build_scheme
from markweightx.check import tiny_model, bind, TEXTS
from markweightx.models import load_tokenizer
torch.manual_seed(987654)                                   # a different global RNG than the parent
k = Key.open(key_path({k.commitment!r})); tok = load_tokenizer({base!r})
s = build_scheme(k); bind(s, tiny_model(tok), tok, k.params)
print(json.dumps([x for x, _ in s.score_batch(TEXTS)]))
"""
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env=dict(os.environ, MWX_KEYS=store_dir, TQDM_DISABLE="1"))
        if r.returncode != 0:
            return False, (r.stderr.strip().splitlines() or ["no output"])[-1][:200]
        got = json.loads(r.stdout.strip().splitlines()[-1])
        ok = all(abs(x - y) < 1e-5 for x, y in zip(got, state["scores"]))
        return ok, "a fresh process with only the key reproduced the scores" if ok else f"{got} vs {state['scores']}"
    check("C3 round trip", c3)

    def c5():
        if sch.pvalue(0.0, 100) is None:
            return None, "empirical null: calibrate with mwx.calibrate on the real base"
        from scipy.stats import kstest
        ps = []
        for i in range(n_null):
            s, k, _ = fresh(1000 + i)
            if edit:
                s = new(); bind(s, tiny_model(tok), tok, k)
            stat, n = s.score(TEXTS[i % len(TEXTS)])
            ps.append(s.pvalue(stat, n))
        p = kstest(ps, "uniform").pvalue
        return p > 0.01, f"p on human text over {n_null} fresh keys vs U(0,1): KS p = {p:.3f}"
    check("C5 null", c5)

    def c6():
        if edit:
            from transformers import AutoModelForCausalLM
            m = AutoModelForCausalLM.from_pretrained(os.path.join(state["out"], "weights"), torch_dtype=torch.float32).eval()
            s = new(); bind(s, tiny_model(tok), tok, state["key"])
            marked = _decode(m, tok, None)
            plain = _decode(tiny_model(tok), tok, None)
        else:
            s = new(); bind(s, tiny_model(tok), tok, state["key"])
            marked = _decode(tiny_model(tok), tok, s)
            plain = _decode(tiny_model(tok), tok, None)
        a = sum(x for x, _ in s.score_batch(marked)) / len(marked)
        b = sum(x for x, _ in s.score_batch(plain)) / len(plain)
        return a > b, f"mean score, {'edited model' if edit else 'decoded with target_logits'} {a:.3f} vs plain {b:.3f}"
    check("C6 signal", c6)

    def c7():
        if not edit:
            return None, "teacher-learned: the release is a LoRA adapter; key material never enters it (P1 tests)"
        files = [os.path.relpath(os.path.join(d, f), state["out"]) for d, _, fs in os.walk(state["out"]) for f in fs]
        extra = [f for f in files if not f.startswith("weights" + os.sep)]
        return not extra, "the release holds weights only" if not extra else f"extra files in the release: {extra}"
    check("C7 isolation", c7)
    rows.append(("C9 length", "SKIP", "run exp/learnability.py on the real base"))
    return rows


def _decode(model, tok, scheme, n=6, new_tokens=40):
    """Sample from the tiny model; with a scheme, apply its Signal (target_logits) at each step."""
    from transformers import LogitsProcessorList
    procs = None
    if scheme is not None:
        if hasattr(scheme, "processor"):                   # a decode-time teacher (MarkLLM, PRO)
            step = scheme.processor()
        else:                                              # otherwise the Signal on the last position
            step = lambda ids, sc: scheme.target_logits(ids, sc.float()[:, None, :])[:, 0]
        procs = LogitsProcessorList([step])
    torch.manual_seed(7)
    enc = tok(["The"] * n, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=new_tokens, do_sample=True, top_k=0,
                             pad_token_id=tok.eos_token_id, logits_processor=procs)
    return tok.batch_decode(out, skip_special_tokens=True)


def report(rows) -> str:
    w = max(len(r[0]) for r in rows)
    return "\n".join(f"{c.ljust(w)}  {s}  {m}" for c, s, m in rows)
