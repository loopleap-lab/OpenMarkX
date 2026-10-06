"""markweightx check --adapter: what a scheme must pass before it joins the leaderboard (CPU, tiny Llama, minutes).

  C1 declared     imprint and access are declared, kind is one embed can train (distill_logit, or none for EDIT),
                  params are JSON-serializable, and all of it matches configs/watermark/*.yaml
  C2 access       detect() with a grant one level below the scheme's is refused before any model is loaded
  C3 round trip   a fresh process holding only the key (plugin source included, other RNG and hash seeds, another
                  working directory, no model for R0
                  schemes) reproduces the scores exactly
  C4 key          fresh keys (drawn the way embed draws them) give different scores; 24 keys give no more than one
                  identical pair (a small key space repeats)
  C5 null         schemes with a closed-form pvalue(): over fresh keys, p on human text (short, long, looping) is
                  not below U(0,1) (KS), its
                  1% and 5% tails are not too heavy (binomial bound), and marked text gets lower p than plain text
  C6 signal       teacher-learned: decoding the tiny model with target_logits() scores clearly above plain decoding;
                  teacher-free: text from the edited tiny model scores clearly above text from the base
  C7 isolation    teacher-free: the release holds only model files (config, safetensors, tokenizer), none equal
                  to key material in the key store, none containing a value from the key
  C8 determinism  rebinding the same key gives the same key material and the same scores; setup() leaves the
                  model and tokenizer it is given unchanged
  C9 length       not checked here: run exp/learnability.py, whose null is per length bucket in the leaderboard
  C10 positions   teacher-learned: target_logits() on a whole sequence (what training sees) equals target_logits()
                  on each prefix at its last position (what decoding sees)
Every check runs against a temporary key store and in its own try/except, so a crash is a FAIL row, not a traceback.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
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
# the null set: the three short texts, a long one (length bugs), and a looping one (repeated-token bugs)
LONG = ("The town council voted on Monday night to approve a new budget for the public library, ending a debate that "
        "had stretched over three months. Supporters argued that the library had become a second home for students "
        "after school, for job seekers who needed computers, and for older residents who came for the newspapers and "
        "the company. Critics said the money would be better spent on roads, pointing to the potholes on Main Street "
        "and the bridge over the river that has needed repairs since the spring floods. In the end the vote was five "
        "to two. The budget adds two librarians, extends opening hours on weekends, and replaces the heating system, "
        "which broke twice last winter. The head librarian thanked the council and said the first change readers "
        "would notice is the Sunday opening, starting next month. A group of parents who had spoken at every meeting "
        "said they were relieved, and one of the council members who voted against the plan said she would keep "
        "pushing for the road repairs in the next budget, which the council will start drafting in January.")
LOOP = "I think that the answer is yes. " * 12
NULL_TEXTS = TEXTS + [LONG, LOOP]
LOWER = {"R0_AUX": "R0", "R1": "R0", "R2": "R1"}


def tiny_model(tok):
    from transformers import LlamaConfig, LlamaForCausalLM
    g = torch.random.get_rng_state()
    torch.manual_seed(0)                                   # the tiny base is fixed; global RNG is restored after
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4)
    m = LlamaForCausalLM(cfg).to(torch.bfloat16).float().eval()
    with torch.no_grad():                                  # logits spread like a real LM's (sd ~5, not 0.16), so a
        m.lm_head.weight.mul_(30)                          # weight edit actually changes what gets sampled
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


RELEASE_FILES = ("config.json", "generation_config.json", "model.safetensors", "model.safetensors.index.json",
                 "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def run(name: str, source: str | None = None, base: str = "llama-3.2-1b", n_null: int = 200) -> list[tuple]:
    from . import keys as K
    root = tempfile.mkdtemp(prefix="mwx-check-")             # keys and tiny releases; removed when the check ends
    store_dir = os.path.join(root, "keys")
    old_store, old_tmp, K.STORE = K.STORE, tempfile.tempdir, store_dir
    tempfile.tempdir = root
    os.environ["TQDM_DISABLE"] = "1"
    try:
        return _run(name, source, base, n_null, store_dir)
    finally:
        K.STORE, tempfile.tempdir = old_store, old_tmp
        shutil.rmtree(root, ignore_errors=True)


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
    import copy
    pristine_tok = copy.deepcopy(tok)                      # C8 compares setup() against an untouched tokenizer
    model = tiny_model(tok)
    new = lambda: get_scheme(name, source=source)
    sch = new()
    edit = sch.imprint == Imprint.EDIT

    def c1():
        json.dumps(sch.params)
        trainable = sch.kind == "none" if sch.imprint == Imprint.EDIT else sch.kind == "distill_logit"   # what embed runs
        ok = trainable and isinstance(sch.access, Access) and isinstance(sch.imprint, Imprint)
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
        if a == b:
            return False, f"two fresh keys give the same scores: {['%.3f' % x for x in a]}"
        seen = []                                          # a small key space shows up as repeated scores
        for i in range(24):
            s, k, out = fresh(100 + i)
            if edit:
                s = new(); bind(s, tiny_model(tok), tok, k); shutil.rmtree(out, ignore_errors=True)
            seen.append(tuple(round(x, 6) for x, _ in s.score_batch(NULL_TEXTS)))
        same = sum(seen[i] == seen[j] for i in range(len(seen)) for j in range(i))
        return same <= 1, (f"24 fresh keys: {len(set(seen))} distinct score vectors"
                           + ("" if same <= 1 else f", {same} identical pairs: the key space looks small"))
    if not check("C4 key", c4):
        return sorted(rows, key=lambda r: int(r[0].split()[0][1:]))

    def c8():
        import hashlib
        import copy
        m0, t0 = tiny_model(tok), copy.deepcopy(pristine_tok)
        fp = lambda: (hashlib.sha256(b"".join(v.detach().float().numpy().tobytes() for v in m0.state_dict().values()))
                      .hexdigest(), len(t0), t0.padding_side, t0.truncation_side, str(t0.special_tokens_map))
        before = fp()
        s = new(); k = bind(s, m0, t0, dict(state["key"]))
        if not edit and fp() != before:
            return False, "setup() changed the model or tokenizer it was given"
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
s = build_scheme(k, trust_plugin=True); bind(s, tiny_model(tok), tok, k.params)
print(json.dumps([x for x, _ in s.score_batch(TEXTS)]))
"""
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env=dict(os.environ, MWX_KEYS=store_dir, TQDM_DISABLE="1",
                                    PYTHONHASHSEED=str(int(os.environ.get("PYTHONHASHSEED", "0") or 0) + 4711)),
                           cwd=tempfile.mkdtemp())        # nothing next to the plugin file is importable
        if r.returncode != 0:
            return False, (r.stderr.strip().splitlines() or ["no output"])[-1][:200]
        got = json.loads(r.stdout.strip().splitlines()[-1])
        ok = all(abs(x - y) < 1e-5 for x, y in zip(got, state["scores"]))
        return ok, "a fresh process with only the key reproduced the scores" if ok else f"{got} vs {state['scores']}"
    check("C3 round trip", c3)

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
        state.update(marked=marked, plain=plain)
        sa = [x for x, _ in s.score_batch(marked)]; sb = [x for x, _ in s.score_batch(plain)]
        a, b = sum(sa) / len(sa), sum(sb) / len(sb)
        sd = (sum((x - b) ** 2 for x in sb) / max(len(sb) - 1, 1)) ** 0.5
        margin = 2 * sd / len(sb) ** 0.5                   # two standard errors of the plain mean
        return a - b > margin, (f"mean score, {'edited model' if edit else 'decoded with target_logits'} {a:.3f} "
                                f"vs plain {b:.3f} (needs a gap > {margin:.3f})")
    check("C6 signal", c6)

    def c5():
        if sch.pvalue(0.0, 100) is None:
            return None, "empirical null: calibrate with mwx.calibrate on the real base"
        from scipy.stats import binom, kstest
        if "marked" not in state:
            return False, "needs marked text from C6, which failed"
        ps = []
        for i in range(n_null):
            s, k, out = fresh(1000 + i)
            if edit:
                s = new(); bind(s, tiny_model(tok), tok, k)
                shutil.rmtree(out, ignore_errors=True)
            ps += [s.pvalue(stat, n) for stat, n in s.score_batch(NULL_TEXTS)]
        ks = kstest(ps, "uniform", alternative="greater").pvalue  # too many small p; a conservative p is allowed
        tails = {a: sum(p < a for p in ps) for a in (0.01, 0.05)}
        heavy = [a for a, c in tails.items() if c > binom.ppf(0.999, len(ps), a)]
        if "marked" not in state:
            return False, "needs marked text from C6, which failed"
        s = new(); bind(s, tiny_model(tok), tok, state["key"])
        pm = sum(s.pvalue(x, n) for x, n in s.score_batch(state["marked"])) / len(state["marked"])
        pp = sum(s.pvalue(x, n) for x, n in s.score_batch(state["plain"])) / len(state["plain"])
        ok = ks > 0.01 and not heavy and pm < pp
        return ok, (f"{len(ps)} p-values on human text: KS p = {ks:.3f}, "
                    f"p<1%: {tails[0.01]}, p<5%: {tails[0.05]}{' (too many)' if heavy else ''}; "
                    f"mean p marked {pm:.3f} vs plain {pp:.3f}")
    check("C5 null", c5)

    def c7():
        if not edit:
            return None, "teacher-learned: the release is a LoRA adapter; key material never enters it (P1 tests)"
        import hashlib
        files = [os.path.relpath(os.path.join(d, f), state["out"]) for d, _, fs in os.walk(state["out"]) for f in fs]
        extra = [f for f in files if not (f.startswith("weights" + os.sep) and os.path.basename(f) in RELEASE_FILES
                                          or f.startswith("weights" + os.sep) and f.endswith(".safetensors"))]
        sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
        blobs = {sha(os.path.join(d, f)) for d, _, fs in os.walk(store_dir) for f in fs}
        leaked = [f for f in files if sha(os.path.join(state["out"], f)) in blobs]
        from .keys import PUBLIC_KEY_FIELDS
        from .registry import resolved_params
        public = {str(v) for v in resolved_params(sch).values()}   # scheme params are public (release lineage)
        vals = set()

        def walk(v):
            if isinstance(v, dict):
                [walk(x) for k_, x in v.items() if k_ not in PUBLIC_KEY_FIELDS]
            elif isinstance(v, (list, tuple)):
                [walk(x) for x in v]
            elif isinstance(v, (int, str)) and not isinstance(v, bool) and len(str(v)) >= 6 and str(v) not in public:
                vals.add(str(v))
        walk(state["key"])
        found = [f for f in files for v in vals if v.encode() in open(os.path.join(state["out"], f), "rb").read()]
        if extra or leaked or found:
            return False, (f"not model files: {extra}" if extra else f"files equal to key material: {leaked}"
                           if leaked else f"key values written into: {sorted(set(found))}")
        return True, f"the release holds model files only ({len(files)}), none equal to key material"
    check("C7 isolation", c7)

    def c10():
        if edit or not hasattr(sch, "target_logits"):
            return None, "teacher-free: no Signal to compare"
        if getattr(sch, "sequence_signal", None):
            return None, f"declared sequence-level Signal: {sch.sequence_signal}"
        s = new(); m = tiny_model(tok); bind(s, m, tok, state["key"])
        ids = tok(TEXTS[0], return_tensors="pt")["input_ids"][:, :32]
        with torch.no_grad():
            logits = m(ids).logits.float()
            whole = s.target_logits(ids, logits)
            worst = 0.0
            for t in (1, 5, 12, 20, ids.shape[1] - 1):
                step = s.target_logits(ids[:, :t + 1], logits[:, :t + 1])[:, -1]
                worst = max(worst, (whole[:, t] - step).abs().max().item())
        return worst < 1e-4, f"whole sequence vs each prefix: max |difference| = {worst:.2e}"
    check("C10 positions", c10)
    rows.append(("C9 length", "SKIP", "run exp/learnability.py on the real base"))
    return sorted(rows, key=lambda r: int(r[0].split()[0][1:]))


def _decode(model, tok, scheme, n=12, new_tokens=40):
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
