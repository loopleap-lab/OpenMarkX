# MarkWeightX

Watermarks that live in the weights of an open model. The owner puts the mark into the weights and keeps a key.
Anyone downstream can fine-tune, merge, prune, quantize or reword. A detector with the key traces text back to
the model.

```
Owner        embed ──▶ release (public weights)  +  key (private, in the key store)
Downstream   modify ─▶ edited release ── generate ─▶ texts ── rewrite ─▶ reworded texts
Detector     detect(texts, key) ─▶ one Detection per text;  verify_model(release, key, prompts) ─▶ one verdict
```

## Install

```bash
pip install -e .               # or: uv pip install -e .  (local only; nothing is published)
export MWX_RUNS=/path/to/runs  # releases, results, key store (default: /mnt/data/jiaq/mwx-runs)
export MWX_HF=/path/to/hf      # where base models are cached
```

MarkLLM and the PRO paper code are used unchanged from local checkouts (`MARKLLM_DIR`, `PRO_CODE_DIR`; see
`markweightx/paths.py`). Base model names (`llama-3.2-1b`, `llama-3.2-1b-instruct`, ...) map to snapshots there.
You can also pass any local model path.

## Quickstart

```python
import markweightx as mwx

release, key = mwx.embed("llama-3.2-1b", "gaussmark")        # Owner: mark the weights; the key stays private
key = mwx.calibrate(key, "llama-3.2-1b")                     # freeze the null on the base model's own texts

child = mwx.modify(release, "finetune:steps=200")            # Downstream: edit the weights...
texts = mwx.generate(child, prompts)                         # ...and write with the edited model
texts = mwx.rewrite(texts, "synonym:ratio=0.3")              # ...and reword

hits = mwx.detect(texts, key)                                # Detector: [Detection(statistic, p_value, n_tokens, flagged)]
verdict = mwx.verify_model(child, key, prompts)              # one binomial test over the flags
```

`examples/quickstart.py` runs this end to end on one GPU in a few minutes.

The same flow from the shell:

```bash
markweightx embed --base llama-3.2-1b --watermark gaussmark            # prints {"release": ..., "key": ...}
markweightx calibrate KEY --base llama-3.2-1b
markweightx modify RELEASE --edit finetune:steps=200 --out edited/
markweightx generate edited/ --prompts prompts.jsonl --out texts.jsonl
markweightx detect texts.jsonl --key KEY
markweightx --help                                                       # every command, with examples
```

A KEY is a key id from `markweightx keys list`, a unique prefix of one, or a release directory. Text files are
`.jsonl` (`{"text": ...}` per line) or `.txt` (one text per line).

Edits are written `op:param=value,...`:

| Edit | Params (defaults) |
|---|---|
| `finetune` | `steps=800, lr=1e-4, bs=8, seq_len=256, rank=16`: LoRA on clean text, merged into the weights |
| `merge` | `partner=` (required, e.g. `llama-3.2-1b-instruct`), `ratio=0.4` (the partner's weight) |
| `prune` | `sparsity=0.5, method=magnitude` |
| `quantize` | `bits=4` (nf4) or `bits=8` (int8) |
| `synonym`, `word_deletion`, `context_synonym` | `ratio=0.3` (text edits, used with `rewrite`) |

## Methods

`markweightx describe` prints this table. A method is a **scheme**, and every scheme is assembled from six
primitives: Key, Signal, Imprint, Perturb, Probe and Score.

| Class | Schemes | How the mark gets in |
|---|---|---|
| teacher-learned | `pro`, `pro-coopt-c100`, `pro-full-b01` (ours); `markllm-<x>-d` for 12 MarkLLM watermarks | distill a decode-time teacher's logits into the weights (LoRA) |
| teacher-free | `gaussmark` | write a keyed perturbation straight into one weight matrix |

Parameters go after a colon: `gaussmark:sigma=0.02,layer=12`. A misspelled parameter or name is an error, with
the accepted list. `embed` writes every parameter into the key, defaults included. So a key keeps detecting the
same way after a default changes in the code.

Not included: `markllm-expgumbel-d`. Upstream builds a (V·prefix)×V noise table, 128 GB at V = 128k, and it
follows the same Gumbel-max rule as EXP.

## What the detector needs (access levels)

| Level | The detector holds | Schemes |
|---|---|---|
| `R0` | text, tokenizer, key | KGW family, Unigram, EXP family, SynthID, DIP, PF, ITSEdit |
| `R0_AUX` | + an auxiliary network that ships in the key | PRO (its watermark policy) |
| `R1` | + forward passes of the base model | SWEET, EWD, Unbiased |
| `R2` | + gradients of the base model | GaussMark |

`detect(..., grant=Access.R1)` (CLI `--grant R1`) caps what a detector may touch. A scheme that needs more is
refused before any model is loaded.

## Keys

Keys never ship with a release. `release.json` holds only the key's commitment, which is sha256 over a nonce, the
scheme, its parameters and the plugin source. The key lives in the key store, `$MWX_KEYS` (default
`$MWX_RUNS/keys`). Each key carries its own fresh secret: MarkLLM's public default hash keys are replaced, and PRO
draws its own token mapping. A calibrated key carries a frozen empirical null, so `detect` returns calibrated
p-values. Schemes with an exact closed-form null (`exact_null = True`, e.g. GaussMark) use it directly.

```bash
markweightx keys list
markweightx keys show KEY        # prints metadata only; key material stays hidden
```

Keys made before 2026-10-06 store only the parameters that were typed, not the defaults. Their parameters are part
of the commitment, so they cannot be filled in afterwards. Such a key detects with the scheme's current defaults.

## Add your own method

```bash
markweightx new-scheme my-mark            # writes my_mark.py: Key, Signal, Score in ~40 lines
markweightx check --adapter my_mark.py    # minutes, on CPU, against a tiny Llama
markweightx embed --base llama-3.2-1b --watermark my_mark.py
```

`check` is what a scheme must pass before it joins the leaderboard:

| Check | Passes when |
|---|---|
| C1 declared | imprint, access and kind are declared; kind is one `embed` trains; matches `configs/watermark/*.yaml` |
| C2 access | `detect` with a grant one level too low is refused before any model loads |
| C3 round trip | a fresh process holding only the key reproduces the scores (other RNG and hash seeds, another working directory, no model for R0) |
| C4 key | fresh keys give different scores; 24 keys give at most one identical pair (no tiny key space) |
| C5 null | closed-form p-values are not too small on short, long and looping human text (KS, 1% and 5% tails); marked text gets lower p than plain |
| C6 signal | text decoded with the Signal (or from the edited model) scores clearly above plain text |
| C7 isolation | an EDIT release holds only model files, none equal to key material or containing a key value |
| C8 determinism | rebinding the same key gives the same material and scores; `setup` leaves the model and tokenizer unchanged |
| C10 positions | `target_logits` on a whole sequence (training) equals it on each prefix (decoding) |

C9 (scores across text lengths) needs the real base, so `check` skips it: run `exp/learnability.py` for that.
`tests/broken/` holds ten adapters with typical bugs, and `tests/test_check.py` makes sure each one fails.

Rules for plugins:
- **One file.** The key stores that file, so it cannot import a sibling module. `check` runs C3 from another
  directory to catch this. A plugin can also be an installed module (`pkg.mod:name`); the key then pins the
  module's file digest.
- **Pinned code.** The key records the code that made it: the content hash of a file, or the digest of a module.
  A detector scores with exactly that code. An edited module is refused rather than silently used.
- **Unique names.** A plugin cannot reuse a built-in scheme's name.
- **Trust.** A key from someone else can name any code. `detect`, `verify_model` and `calibrate` run plugin code
  only if your process already imported it, or with `trust_plugin=True` (CLI `--trust-plugin`).

## Experiments and the leaderboard

| Script | What it runs |
|---|---|
| `exp/learnability.py --scheme NAME` | E2: embed, sample 500 C4 prompts, TPR at 1% FPR against the base model's texts, PPL |
| `exp/robustness.py --release DIR` | E4: every weight and text edit on one release, with the same ruler |
| `exp/spoof.py --release DIR` | forgery: count tokens in the owner's texts, bias the base model, measure how much is flagged |
| `exp/leaderboard.py` | one row per method; experimental variants and sweeps go under Ablations |
| `exp/scheduler.py --host NAME` | a GPU queue (`$MWX_RUNS/sched/<host>/queue.json`) that waits for free memory |

The training recipe for teacher-learned schemes is `configs/embed.yaml` (`embed --recipe`). All roadmap numbers
are on Llama-3.2-1B.

## Names in the design appendix vs. the code

| Technical appendix | Code |
|---|---|
| `T-R0` / `T-R1` / `T-R2` | `Access.R0` / `R1` / `R2`; the auxiliary-network case is `R0_AUX` |
| `KeyBundle` | `Key` (`markweightx.types`), stored by commitment in the key store |
| `WDetection` | `Detection(statistic, p_value, n_tokens, flagged)` |
| `modify(release, attack)`, `ft:full`, `quant` | `modify(release, "finetune:steps=800")`, `"quantize:bits=4"`, `"merge:partner=...,ratio=0.4"`, `"prune:sparsity=0.5"` |
| `harness.attack` (text) | `rewrite(texts, "synonym:ratio=0.3")`, `"word_deletion:ratio=0.3"` |
| `harness.evaluate(arena.yaml)` | `exp/learnability.py` + `exp/robustness.py` + `exp/spoof.py` + `exp/leaderboard.py` |
| `Scheme.keygen / signal / recipe / detector` | `Scheme.setup` (key) / `target_logits` (Signal) / `configs/embed.yaml` / `score`, `pvalue` |
| CLI `verify` | `markweightx verify` → `mwx.verify_model` |

Not built yet: `port` (importing published watermarked checkpoints without retraining), `TP-*` grants (prompt
access), the `sft` and `rl` imprints, and the `distill_sampling` recipe.

## Layout

```
markweightx/        api.py (the facade) · cli.py · check.py · keys.py · registry.py · primitives.py · types.py
  schemes/          base.py (Scheme) · markllm.py · pro.py · gaussmark.py
  edits/            weights.py (finetune, merge, prune, quantize) · text.py (synonym, word_deletion)
  imprint.py · sampling.py · scoring.py · models.py · paths.py
configs/            embed.yaml · watermark/<scheme>.yaml (generated by exp/write_configs.py)
examples/           quickstart.py · my_scheme.py (the new-scheme template)
exp/                experiment scripts (above)
tests/              pytest -q tests  (add -m "not slow" to skip the end-to-end check runs)
```
