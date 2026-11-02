"""The `openmarkx` command line: the five calls for the threat-model roles, plus the researcher's tools.

  Owner       openmarkx embed --base llama-3.2-1b --watermark gaussmark            -> release dir + key id
              openmarkx port CKPT --watermark openmark --key-file key.json --base llama-3.2-1b   (made elsewhere)
              openmarkx embed --base llama-3.2-1b --watermark openmark --recipe configs/embed.yaml
              openmarkx calibrate KEY --base llama-3.2-1b
  Downstream  openmarkx modify RELEASE --edit finetune:steps=800 --out DIR
              openmarkx generate RELEASE --prompts prompts.jsonl --out texts.jsonl
              openmarkx rewrite texts.jsonl --edit synonym:ratio=0.3 --out reworded.jsonl
  Detector    openmarkx detect texts.jsonl --key KEY [--grant R0|R0_AUX|R1|R2] [--fpr 0.01]
              openmarkx verify RELEASE --key KEY --prompts prompts.jsonl [--sequential --budget 200]
              openmarkx detect texts.jsonl --key KEY --null wrong_key       (no calibration needed)
  Researcher  openmarkx describe [NAME ...]          openmarkx check --adapter NAME|file.py
              openmarkx new-scheme NAME [--out my_scheme.py]
              openmarkx bench configs/arena/1b.yaml [--dry-run | --queue HOST]
              openmarkx keys list|show KEY
Text files are .jsonl ({"text": ...} per line) or plain text (one text per line). KEY is a key id (commitment, or
a unique prefix) from the key store, or the path of a release (its key is found by commitment).
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _read_texts(path: str) -> list[str]:
    out = []
    for line in open(path):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        if path.endswith(".jsonl"):
            d = json.loads(line)
            out.append(d["text"] if isinstance(d, dict) else str(d))
        else:
            out.append(line)
    return out


WEIGHT_EDITS = """edits:
  finetune:steps=800,lr=1e-4,bs=8,seq_len=256,rank=16     LoRA fine-tune on clean text, merged into the weights
  merge:partner=llama-3.2-1b-instruct,ratio=0.4            linear merge; ratio is the partner's weight
  prune:sparsity=0.5,method=magnitude                      zero the smallest weights
  quantize:bits=4                                          bitsandbytes: 8 (int8) or 4 (nf4)
  your own                                                 @omx.register_edit in a file, passed with --plugin
"""
TEXT_EDITS = """edits:
  synonym:ratio=0.3          swap that share of the words for WordNet synonyms
  word_deletion:ratio=0.3    delete that share of the words
  context_synonym:ratio=0.3  synonyms chosen by BERT from the context (downloads bert-large-uncased)
  your own                   @omx.register_text_edit in a file, passed with --plugin
"""


def _recipe(path: str | None) -> dict:
    if not path:
        return {}
    import inspect
    import yaml
    from .imprint import embed
    r = yaml.safe_load(open(path)) or {}
    allowed = [k for k, p in inspect.signature(embed).parameters.items() if p.kind == p.KEYWORD_ONLY
               and k not in ("device", "key", "scheme_params", "scheme_source")]
    bad = sorted(set(r) - set(allowed))
    if bad:
        raise SystemExit(f"{path}: unknown recipe field(s) {bad}; allowed: {allowed}")
    return r


def _write_jsonl(path: str | None, rows):
    f = open(path, "w") if path else sys.stdout
    for r in rows:
        f.write(json.dumps(r) + "\n")
    if path:
        f.close()


def _key(ref: str):
    from .keys import STORE, load_key
    from .types import Key
    if os.path.isdir(ref) and os.path.exists(os.path.join(ref, "release.json")):
        return load_key(ref)
    if os.path.isfile(ref):
        return Key.open(ref)
    hits = [f for f in os.listdir(STORE) if f.endswith(".json") and f.startswith(ref)] if os.path.isdir(STORE) else []
    if not hits:
        raise SystemExit(f"key {ref!r}: no key with that id in {STORE} (see `openmarkx keys list`)")
    if len(hits) > 1:
        raise SystemExit(f"key {ref!r} is ambiguous: " + ", ".join(h[:16] for h in hits[:8]))
    return Key.open(os.path.join(STORE, hits[0]))


def _release(path: str):
    from .types import Release
    return Release.open(path)


TEMPLATE = '''"""{name}: an OpenMarkX scheme in one file. Check it with `openmarkx check --adapter {file}`."""
import torch

import openmarkx as omx


@omx.register_scheme
class {cls}(omx.Scheme):
    name = "{name}"
    imprint = omx.Imprint.DISTILL      # DISTILL: learn target_logits into the weights; EDIT: write weights directly
    access = omx.Access.R0             # what the detector needs besides the text: R0, R0_AUX, R1 or R2

    def __init__(self, delta: float = 2.0, **params):
        super().__init__(delta=delta, **params)
        self.delta = delta

    def setup(self, model, tokenizer, device, key=None):
        """Key: draw a secret for a new release, or read it from an existing key. Return the key material."""
        super().setup(model, tokenizer, device, key)
        key = dict(key or {{}})
        if "secret" not in key:
            key["secret"] = omx.keys.new_secret()
        g = torch.Generator().manual_seed(int(key["secret"]))
        self.green = torch.rand(model.config.vocab_size, generator=g) < 0.5
        return {{"secret": int(key["secret"])}}

    def target_logits(self, input_ids, logits):
        """Signal: the teacher's logits for every position (here: a fixed green list, +delta)."""
        return logits + self.delta * self.green.to(logits.device, logits.dtype)

    def score(self, text):
        """Score: a statistic (higher = more marked) and the number of tokens scored. Each distinct token counts
        once: a repeated word would otherwise count many times and make the p-value too small."""
        ids = set(self.tok(text, add_special_tokens=False)["input_ids"])
        n = max(len(ids), 1)
        g = sum(bool(self.green[i]) for i in ids)
        return (g - 0.5 * n) / (0.25 * n) ** 0.5, len(ids)

    exact_null = True                  # pvalue() below is exact, so detect() uses it even after omx.calibrate

    def pvalue(self, statistic, n_tokens):
        """Optional closed-form null: P(score >= statistic) for unmarked text. Return None to use the key's
        calibrated empirical null instead (omx.calibrate); with exact_null = False a calibrated null takes over."""
        import math
        n = max(n_tokens, 1)
        g = round(statistic * (0.25 * n) ** 0.5 + 0.5 * n)    # green tokens: Binomial(n, 1/2) without the mark
        return sum(math.comb(n, k) for k in range(max(g, 0), n + 1)) / 2 ** n
'''


def main(argv=None):
    ap = argparse.ArgumentParser(prog="openmarkx", description="Watermarks learned into open-source LLMs.",
                                 epilog=__doc__.split("\n", 1)[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("embed", help="Owner: learn a watermark into a base model")
    p.add_argument("--base", required=True, help="base model name (e.g. llama-3.2-1b) or a local path")
    p.add_argument("--watermark", default="openmark", help="scheme, with params: gaussmark:sigma=0.01 ; or a plugin "
                   "file: my_scheme.py[:name]  (see `openmarkx describe`)")
    p.add_argument("--out", help="release directory (default: RUNS/releases/<base>/<scheme>)")
    p.add_argument("--recipe", help="training recipe YAML for teacher-learned schemes (see configs/embed.yaml)")
    p.add_argument("--steps", type=int, help="training steps for teacher-learned schemes (overrides the recipe)")
    p.add_argument("--device", help="default: the first GPU, else CPU")
    p.add_argument("--model", default="text", help="model adapter: text (default), agent, vlm, taming, geopt (see `openmarkx run` for the others)")
    KEY_HELP = "key id (or unique prefix) from `openmarkx keys list`, or a release dir"
    TEXTS_HELP = ".jsonl ({\"text\": ...} per line) or .txt (one text per line)"
    TRUST_HELP = "run the plugin code a key names (only for keys you or someone you trust made)"
    p = sub.add_parser("port", help="Owner: import a watermarked checkpoint made elsewhere (no retraining)")
    p.add_argument("model_dir", help="a LoRA adapter directory or a full model directory")
    p.add_argument("--watermark", required=True, help="the scheme it was made with (see `openmarkx describe`)")
    p.add_argument("--key-file", required=True, help="JSON with the scheme's key fields, e.g. {\"secret\": 123}")
    p.add_argument("--base", required=True, help="the base model the checkpoint was made from")
    p.add_argument("--out", help="release directory (default: <model_dir>-omx)")
    p.add_argument("--public", action="store_true", help="the key was shipped with the checkpoint (anyone can use it)")
    p.add_argument("--force", action="store_true", help="replace a release already in the output directory")
    p = sub.add_parser("calibrate", help="Owner: freeze the null for a key")
    p.add_argument("key", help=KEY_HELP)
    p.add_argument("--base", help="base model whose texts form the null (default: the key's base)")
    p.add_argument("--n", type=int, default=500, help="null texts to sample (default 500)")
    p.add_argument("--prompts", help=TEXTS_HELP + " (default: n held-out C4 prompts); reuse them for verify")
    p.add_argument("--device", help="default: the first GPU, else CPU")
    p.add_argument("--trust-plugin", action="store_true", help=TRUST_HELP)
    p = sub.add_parser("modify", help="Downstream: edit the weights of a release",
                       formatter_class=argparse.RawDescriptionHelpFormatter, epilog=WEIGHT_EDITS)
    p.add_argument("release", help="release directory")
    p.add_argument("--edit", required=True, help="op:param=value,... (see below)")
    p.add_argument("--plugin", action="append", default=[], help="a .py file that registers edits (repeatable)")
    p.add_argument("--out", help="directory for the edited release (default: next to the release)")
    p.add_argument("--device", help="default: the first GPU, else CPU")
    p = sub.add_parser("generate", help="Downstream: sample texts from a release")
    p.add_argument("release", help="release directory")
    p.add_argument("--prompts", required=True, help=TEXTS_HELP)
    p.add_argument("--out", help="output .jsonl (default: stdout)")
    p.add_argument("--device", help="default: the first GPU, else CPU")
    p = sub.add_parser("rewrite", help="Downstream: reword texts",
                       formatter_class=argparse.RawDescriptionHelpFormatter, epilog=TEXT_EDITS)
    p.add_argument("texts", help=TEXTS_HELP)
    p.add_argument("--edit", required=True, help="op:param=value (see below)")
    p.add_argument("--plugin", action="append", default=[], help="a .py file that registers text edits (repeatable)")
    p.add_argument("--out", help="output .jsonl (default: stdout)")
    p = sub.add_parser("detect", help="Detector: score texts with a key")
    p.add_argument("texts", help=TEXTS_HELP)
    p.add_argument("--key", required=True, help=KEY_HELP)
    p.add_argument("--grant", choices=["R0", "R0_AUX", "R1", "R2"], help="what this detector may access "
                   "(default: what the scheme needs); a lower grant is refused")
    p.add_argument("--fpr", type=float, default=0.01, help="false-positive rate for the flag (default 0.01)")
    p.add_argument("--base", help="base model for R1/R2 detectors (default: the key's base)")
    p.add_argument("--null", choices=["auto", "wrong_key"], default="auto",
                   help="auto: the key's calibrated null, else the closed form; wrong_key: score under --n-null wrong "
                        "keys and rank the true key's statistic among them (no calibration needed, slower)")
    p.add_argument("--n-null", type=int, default=200, help="wrong keys for --null wrong_key (default 200)")
    p.add_argument("--device", help="default: the device the key was made on")
    p.add_argument("--out", help="output .jsonl (default: stdout)")
    p.add_argument("--trust-plugin", action="store_true", help=TRUST_HELP)
    p.add_argument("--model", default="text", help="text family: texts as above; taming: a directory of .png images "
                   "(VQ-encoded, scored against the key's frozen null); geopt: an .npz with points, field, condition "
                   "(scored against --release's unmarked model, wrong-key null)")
    p.add_argument("--release", help="taming/geopt: the release whose model (its VQ, its unmarked base) the detector "
                   "uses (default: --key, when that is a release directory)")
    p = sub.add_parser("verify", help="Detector: model-level verdict on a suspect release")
    p.add_argument("release", help="the suspect release directory")
    p.add_argument("--key", required=True, help=KEY_HELP)
    p.add_argument("--prompts", required=True, help=TEXTS_HELP + "; the release is sampled on these")
    p.add_argument("--fpr", type=float, default=0.01, help="per-text flag rate under the null (default 0.01)")
    p.add_argument("--alpha", type=float, default=0.01, help="model-level false-accusation rate (default 0.01)")
    p.add_argument("--budget", type=int, help="at most this many texts from the suspect model")
    p.add_argument("--sequential", action="store_true",
                   help="20 texts at a time; accuse early once decisive, never clear early (see verify_model)")
    p.add_argument("--p1", type=float, default=0.2, help="flag rate the early accusation is tuned for (default 0.2)")
    p.add_argument("--device", help="default: the first GPU, else CPU")
    p.add_argument("--trust-plugin", action="store_true", help=TRUST_HELP)
    p.add_argument("--model", default="text", help="model adapter: text (default), agent, vlm, taming, geopt (see `openmarkx run` for the others)")
    p = sub.add_parser("run", help="Everyone: run a whole experiment from a config (prepare -> ... -> report.json)")
    p.add_argument("config", help="a run config YAML (see configs/demos/)")
    p.add_argument("--only", nargs="+", metavar="STEP", help="run only these steps: prepare selfgen calibrate imprint "
                   "generate detect edits quality report")
    p.add_argument("--device", help="default: the first GPU, else CPU")
    p.add_argument("--smoke", action="store_true", help="a few imprint steps and samples, to check the wiring")
    p.add_argument("--from", dest="from_", metavar="PATH", help="reuse a model trained earlier (a release, or a Taming "
                   "run directory with transformer.pt and policy.pt) instead of training; writes to <out>_imported")
    p = sub.add_parser("bench", aliases=["evaluate"],
                       help="Researcher: the whole benchmark for an arena file (learn, edits, forge, leaderboard)")
    p.add_argument("arena", help="arena YAML: base, methods, optional steps (see configs/arena/1b.yaml)")
    p.add_argument("--dry-run", action="store_true", help="show what is done and what would run")
    p.add_argument("--queue", metavar="HOST", help="append the pending steps to exp/scheduler.py's queue for HOST")
    p.add_argument("--no-check", action="store_true", help="skip `openmarkx check` on plugin files")
    p = sub.add_parser("describe", help="Researcher: the scheme x primitive matrix")
    p.add_argument("names", nargs="*", help="scheme names or plugin .py files (default: every registered method)")
    p = sub.add_parser("check", help="Researcher: the adapter checks a scheme must pass")
    p.add_argument("--adapter", required=True, help="a registered scheme name, a .py file, or a module (pkg.mod)")
    p.add_argument("--name", help="scheme name inside the file (default: the file's only scheme)")
    p.add_argument("--base", default="llama-3.2-1b", help="whose tokenizer the tiny test model uses")
    p = sub.add_parser("new-scheme", help="Researcher: write a one-file scheme template")
    p.add_argument("name", help="scheme name, e.g. my-mark")
    p.add_argument("--out", help="file to write (default: <name>.py)")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    p = sub.add_parser("keys", help="the key store ($OMX_KEYS)")
    p.add_argument("action", choices=["list", "show"], help="list every key, or show one (key material hidden)")
    p.add_argument("key", nargs="?", help=KEY_HELP)
    p = sub.add_parser("serve", help="run jobs on this machine's GPUs from the web page")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1", help="listen address (default: this machine only; use an SSH "
                   "tunnel to reach it from elsewhere)")
    p.add_argument("--token", help="access token (default: a random one, printed at start)")
    p.add_argument("--origin", action="append", help="another site allowed to connect (repeatable)")
    a = ap.parse_args(argv)
    try:
        _run(a)
    except (KeyError, ValueError, TypeError, ImportError, FileNotFoundError, FileExistsError, SyntaxError,
            PermissionError, NotImplementedError, RuntimeError) as e:
        from .paths import env
        if env("DEBUG"):
            raise
        msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
        if isinstance(e, SyntaxError):
            msg = f"{e.filename}:{e.lineno}: {e.msg}"
        raise SystemExit(f"openmarkx {a.cmd}: {msg}  (OMX_DEBUG=1 for the traceback)")


def _detect_model(a) -> list[dict]:
    """detect --model taming|geopt: images or a field, scored with the key and the release's model."""
    from .api import build_scheme
    from .types import Release
    if not a.release and os.path.exists(os.path.join(a.key, "release.json")):
        a.release = a.key                                   # --key names a release: use its model too
    if not a.release:
        raise SystemExit(f"detect --model {a.model} needs --release (the model whose VQ or unmarked output it uses)")
    key, rel = _key(a.key), Release.open(a.release)
    m = rel.adapter_model(trust=a.trust_plugin)
    device = a.device or "cpu"
    if a.model == "taming":
        from .models.taming import load_pngs
        from .scoring import empirical_p
        files = sorted(os.path.join(a.texts, f) for f in os.listdir(a.texts) if f.endswith(".png")) \
            if os.path.isdir(a.texts) else [a.texts]
        m.load(None, device)
        sch = build_scheme(key, trust_plugin=a.trust_plugin)
        sch.setup(None, m.codec(), device, key.params)
        null = key.calibration.get("null")
        rows = []
        for f, codes in zip(files, m.codec().encode(load_pngs(files))):
            s, n = sch.score(codes)
            p = empirical_p(s, null) if null else sch.pvalue(s, n)
            rows.append({"file": os.path.basename(f), "statistic": s, "p_value": p, "n_pairs": n, "flagged": p < a.fpr})
        return rows
    import numpy as np
    import torch
    d = np.load(a.texts)
    p, field, cond = (torch.tensor(d[k]).float() for k in ("points", "field", "condition"))
    ref = m.run(m.load(None, device), [(p, cond)])[0]       # the publisher's unmarked model on the same inputs
    sch = build_scheme(key, trust_plugin=a.trust_plugin)
    sch.setup(None, None, device, key.params)
    z, n = sch.score(field, ref, {"points": p})
    pv = sch.wrong_key_p(field, ref, {"points": p})
    return [{"file": os.path.basename(a.texts), "statistic": z, "p_value": pv, "n_points": n, "flagged": pv < a.fpr}]


def _run(a):
    import openmarkx as omx
    if getattr(a, "model", "text") not in ("text", None) and a.cmd in ("embed", "verify"):
        raise SystemExit(f"openmarkx {a.cmd} --model {a.model}: for this model use `openmarkx run CONFIG` "
                         f"(configs/demos/ has one per model)")
    if a.cmd == "run":
        from . import pipeline
        cfg = pipeline.load_config(a.config)
        cfg.smoke = cfg.smoke or a.smoke
        if a.from_:
            cfg.imprint["from"] = a.from_
            cfg.out = cfg.out.rstrip("/") + "_imported"
        rep = pipeline.run(cfg, only=a.only, device=a.device)
        print(json.dumps({"report": os.path.join(cfg.out, "report.json") if rep else None,
                          "detect": rep.get("detect"), "quality": rep.get("quality"), "edits": rep.get("edits")}))
        return
    if a.cmd == "detect" and a.model in ("taming", "geopt"):
        _write_jsonl(a.out, _detect_model(a))
        return
    if a.cmd == "embed":
        recipe = _recipe(a.recipe)
        if a.steps:
            recipe["steps"] = a.steps
        rel, key = omx.embed(a.base, a.watermark, a.out, device=a.device, **recipe)
        print(json.dumps({"release": rel.path, "key": key.commitment}))
    elif a.cmd == "port":
        fields = json.load(open(a.key_file))
        here = os.path.dirname(os.path.abspath(a.key_file))   # file paths in the key file are relative to it
        fields = {k: (os.path.join(here, v) if isinstance(v, str) and not os.path.isabs(v)
                      and os.path.exists(os.path.join(here, v)) else v) for k, v in fields.items()}
        rel, key = omx.port(a.model_dir, a.watermark, fields, a.base, a.out, key_public=a.public, force=a.force)
        print(json.dumps({"release": rel.path, "key": key.commitment}))
    elif a.cmd == "calibrate":
        k = omx.calibrate(_key(a.key), a.base or _key(a.key).base, _read_texts(a.prompts) if a.prompts else None,
                          n=a.n, device=a.device, trust_plugin=a.trust_plugin)
        print(json.dumps({"key": k.commitment, "null_n": k.calibration["n"]}))
    elif a.cmd == "modify":
        omx.load_plugins(a.plugin)
        r = omx.modify(_release(a.release), a.edit, a.out, device=a.device)
        print(json.dumps({"release": r.path}))
    elif a.cmd == "generate":
        texts = omx.generate(_release(a.release), _read_texts(a.prompts), device=a.device)
        _write_jsonl(a.out, [{"text": t} for t in texts])
    elif a.cmd == "rewrite":
        omx.load_plugins(a.plugin)
        _write_jsonl(a.out, [{"text": t} for t in omx.rewrite(_read_texts(a.texts), a.edit)])
    elif a.cmd == "detect":
        grant = getattr(omx.Access, a.grant) if a.grant else None
        texts = _read_texts(a.texts)
        dets = omx.detect(texts, _key(a.key), base=a.base, device=a.device, fpr=a.fpr, grant=grant,
                          trust_plugin=a.trust_plugin, null=a.null, n_null=a.n_null)
        _write_jsonl(a.out, [{"text": t[:80], "statistic": d.statistic, "p_value": d.p_value, "n_tokens": d.n_tokens,
                              "flagged": d.flagged} for t, d in zip(texts, dets)])
    elif a.cmd == "verify":
        d = omx.verify_model(_release(a.release), _key(a.key), _read_texts(a.prompts), device=a.device, fpr=a.fpr,
                             alpha=a.alpha, budget=a.budget, sequential=a.sequential, p1=a.p1,
                             trust_plugin=a.trust_plugin)
        print(json.dumps({"flagged_texts": d.statistic, "judged_texts": d.n_judged, "generated_texts": d.n_queries,
                          "p_value": d.p_value, "from_this_key": d.flagged}))
    elif a.cmd in ("bench", "evaluate"):
        from . import bench
        arena = bench.load_arena(a.arena)
        jobs = bench.plan(arena)
        print(bench.show(jobs))
        if a.dry_run:
            return
        if not a.no_check and any(not j["done"] for j in jobs) and bench.plugins(arena):
            import contextlib
            with open(os.devnull, "w") as null, contextlib.redirect_stderr(null):
                failed = bench.check_plugins(arena)
            if failed:
                raise SystemExit(f"fix the failing checks first (or pass --no-check): {', '.join(failed)}")
        if a.queue:
            new = bench.queue(jobs, a.queue)
            print(f"queued {len(new)} step(s) for {a.queue}; rebuild the board afterwards with: "
                  f"{bench.leaderboard_cmd(arena)}" if new else "nothing new to queue")
            return
        sys.exit(bench.run_here(arena, jobs))
    elif a.cmd == "describe":
        print(omx.describe(a.names or None))
    elif a.cmd == "check":
        from .check import report, run
        name, source = a.adapter, None
        if a.adapter.endswith(".py") or "." in a.adapter.split(":")[0]:      # a plugin file or module
            from .api import scheme_spec
            name, _, source = scheme_spec(a.adapter + (f":{a.name}" if a.name else ""))
        import contextlib
        with open(os.devnull, "w") as null, contextlib.redirect_stderr(null):   # progress bars, model-load chatter
            rows = run(name, source=source, base=a.base)
        print(report(rows))
        sys.exit(1 if any(s == "FAIL" for _, s, _ in rows) else 0)
    elif a.cmd == "new-scheme":
        if not a.name.replace("-", "_").isidentifier():
            raise SystemExit(f"{a.name!r} is not a scheme name: use letters, digits, - and _, not starting with a digit")
        out = a.out or a.name.replace("-", "_") + ".py"
        if os.path.exists(out) and not a.force:
            raise SystemExit(f"{out} exists: pass --force to overwrite, or --out another file")
        cls = "".join(w.capitalize() for w in a.name.replace("_", "-").split("-"))
        if os.path.dirname(out):
            os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "w").write(TEMPLATE.format(name=a.name, cls=cls, file=out))
        print(f"wrote {out}: edit it, then run `openmarkx check --adapter {out}`")
    elif a.cmd == "serve":
        from .serve import serve
        serve(a.port, a.host, a.token, a.origin)
    elif a.cmd == "keys":
        from .keys import PUBLIC_KEY_FIELDS, STORE
        from .types import Key
        if a.action == "list":
            for f in sorted(os.listdir(STORE)) if os.path.isdir(STORE) else []:
                if f.endswith(".json"):
                    k = Key.open(os.path.join(STORE, f))
                    print(f"{f[:12]}  {k.scheme:<22} base={k.base}  calibrated={bool(k.calibration.get('null'))}")
        else:
            if not a.key:
                raise SystemExit("keys show needs a key id: see `openmarkx keys list`")
            k = _key(a.key)
            safe = {kk: (v if kk in PUBLIC_KEY_FIELDS else "<hidden>") for kk, v in k.params.items()}   # key material stays out of logs
            print(json.dumps({"commitment": k.commitment, "scheme": k.scheme, "base": k.base, "params": safe,
                              "scheme_params": k.scheme_params, "calibrated_n": k.calibration.get("n")}, indent=1))


if __name__ == "__main__":
    main()
