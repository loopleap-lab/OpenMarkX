# MarkWeightX

Watermarks that live in the weights of an open model. The owner puts the mark into the weights and keeps a key.
Anyone downstream can fine-tune, merge, prune, quantize or reword. A detector with the key traces text back to
the model after most of these edits; fine-tuning still erases today's marks (see the leaderboard).

```
Owner        embed ──▶ release (public weights)  +  key (private, in the key store)
Downstream   modify ─▶ edited release ── generate ─▶ texts ── rewrite ─▶ reworded texts
Detector     detect(texts, key) ─▶ one Detection per text;  verify_model(release, key, prompts) ─▶ one verdict
```

## Install

```bash
pip install -e .                    # or: uv pip install -e .  (local only; nothing is published)
export MWX_RUNS=/path/to/runs       # releases, results and the key store (default: /mnt/data/jiaq/mwx-runs)
export MWX_HF=/path/to/hf           # where pinned base snapshots are looked up
python exp/prepare_data.py          # once: C4 text for training, prompts and human continuations ($MWX_DATA)
```

- **Base models.** Names like `llama-3.2-1b` map to pinned snapshots under `$MWX_HF`. If a snapshot is not on
  disk, the name falls back to its Hugging Face repo (e.g. `unsloth/Llama-3.2-1B`), which is downloaded on first
  use. Any local model path also works.
- **Methods built on other code.** MarkLLM schemes need a checkout of upstream MarkLLM (`MARKLLM_DIR`) with its
  requirements installed. PRO needs the PRO paper code (`PRO_CODE_DIR`) and its sentence embedder. Both are used
  unchanged.
- **Key store.** It lives in `$MWX_KEYS` (default `$MWX_RUNS/keys`). Back it up: a key that is lost cannot be
  recovered.

## Quickstart

```python
import markweightx as mwx
from markweightx.models import load_tokenizer
from markweightx.sampling import eval_prompts

prompts = eval_prompts(load_tokenizer("llama-3.2-1b"), n=500)  # held-out C4 prompts (exp/prepare_data.py)
release, key = mwx.embed("llama-3.2-1b", "gaussmark")        # Owner: mark the weights; the key stays private
key = mwx.calibrate(key, "llama-3.2-1b")                     # freeze the null on the base model's own texts

child = mwx.modify(release, "finetune:steps=200")            # Downstream: edit the weights...
texts = mwx.generate(child, prompts)                         # ...and write with the edited model
texts = mwx.rewrite(texts, "synonym:ratio=0.3")              # ...and reword

hits = mwx.detect(texts, key)                                # Detector: [Detection(statistic, p_value, n_tokens, flagged)]
verdict = mwx.verify_model(child, key, prompts, sequential=True, budget=200)   # one verdict for the model
```

`examples/quickstart.py` runs this end to end on one GPU in a few minutes.

The same flow from the shell:

```bash
markweightx embed --base llama-3.2-1b --watermark gaussmark            # prints {"release": ..., "key": ...}
markweightx calibrate KEY --base llama-3.2-1b
markweightx modify RELEASE --edit finetune:steps=200 --out edited/
markweightx generate edited/ --prompts prompts.jsonl --out texts.jsonl
markweightx rewrite texts.jsonl --edit synonym:ratio=0.3 --out reworded.jsonl
markweightx detect texts.jsonl --key KEY                                 # --null wrong_key: no calibration needed
markweightx port their_lora/ --watermark pro --key-file key.json --base llama-3.2-1b   # a checkpoint made elsewhere
markweightx verify edited/ --key KEY --prompts prompts.jsonl            # one verdict for the model
markweightx verify edited/ --key KEY --prompts prompts.jsonl --sequential --budget 200   # as few texts as needed
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

Parameters go after a colon: `gaussmark:sigma=0.02,layer=12`. A misspelled scheme or edit parameter is an error
that lists the accepted ones. So is an edit value the code would otherwise ignore, such as `quantize:bits=3`. `embed` writes every parameter into the key, defaults included. So a key keeps detecting the
same way after a default changes in the code.

Not included: `markllm-expgumbel-d`. Upstream builds a (V·prefix)×V noise table, 128 GB at V = 128k, and it
follows the same Gumbel-max rule as EXP.

## Bring a checkpoint made elsewhere

`port` wraps a watermarked checkpoint that was trained outside MarkWeightX, together with the key material it was
made with, as a Release and a Key. Nothing is retrained and the weights stay where they are.

```python
release, key = mwx.port("their_lora/", "pro", {"mapping": "map.json", "transform": "policy.pth"}, "llama-3.2-1b")
release, key = mwx.port("their_model/", "markllm-kgw-d", {"secret": 123456}, "llama-3.2-1b", key_public=True)
```

`key_public=True` marks a key that was shipped with the checkpoint, e.g. a paper's released mapping. Anyone can
detect with such a key, and anyone can also remove or forge the mark, so `detect` warns.

## What the detector needs (access levels)

| Level | The detector holds | Schemes |
|---|---|---|
| `R0` | text, tokenizer, key | KGW family, Unigram, EXP family, SynthID, DIP, PF, ITSEdit |
| `R0_AUX` | + an auxiliary network that ships in the key | PRO (its watermark policy) |
| `R1` | + forward passes of the base model | SWEET, EWD, Unbiased |
| `R2` | + gradients of the base model | GaussMark |

`detect(..., grant=Access.R1)` (CLI `--grant R1`) caps what a detector may touch. A scheme that needs more is
refused before any model is loaded.

## With MarkAgentX

`Detection` is MarkAgentX's `Detection(statistic, p_value, n_evidence)` with the evidence counted in tokens:
`d.n_evidence` and `d.to_markagentx()` give its fields, and as there an empty text answers `p_value = 1.0` with no
evidence. `detect(..., null="wrong_key")` (CLI `--null wrong_key`) is MarkAgentX's wrong-key null: each text is
scored under `n_null` keys drawn from `random.Random(0xC0FFEE)`, and `p = (1 + #{wrong keys >= S}) / (n_null + 1)`.
A wrong key is the true key with only its secret redrawn (`Scheme.wrong_key`; PRO redraws its mapping and keeps
its policy). It needs no calibration but costs `n_null` rebindings of the scheme; teacher-free schemes are refused,
since their closed-form null is already exact. The draws are reproducible but are not MarkAgentX's keys: each scheme
draws its own kind of secret. `tests/test_markagentx.py` checks the fields against a MarkAgentX checkout
(`refs/MarkAgentX`) and builds its `Detection` from `to_markagentx()`.

## Keys

Keys never ship with a release. `release.json` holds only the key's commitment, which is sha256 over a nonce, the
scheme, its parameters and the plugin source. The key lives in the key store, `$MWX_KEYS` (default
`$MWX_RUNS/keys`). Each key carries its own fresh secret: MarkLLM's public default hash keys are replaced, and PRO
draws its own token mapping. A calibrated key carries a frozen empirical null, so `detect` returns calibrated
p-values. The calibration also holds nulls for the texts cut to 20, 50 and 100 tokens: a text is judged against the
null calibrated at the largest length not above its own, because some detectors' false positives grow on short texts
(C9: PRO reaches 14% at 50 tokens against a 200-token threshold). Keys
calibrated before this change warn until calibrated again. A text too short to be judged gets `p_value=None` and
`flagged=None` (no evidence either way), and `verify_model` does not count it. Schemes with an exact closed-form null (`exact_null = True`, e.g. GaussMark) use it
directly.

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

C9 (shorter texts) needs the real base, so `check` skips it. `exp/learnability.py` cuts the evaluation texts to
50 and 100 tokens and reports the false-positive rate at the full-length threshold; the leaderboard shows it as a
C9 table (`exp/length_check.py` adds it to finished runs).
`tests/broken/` holds 17 adapters with typical bugs, and `tests/test_check.py` makes sure each one fails.

Rules for plugins:
- **One file.** The key stores that file, so it cannot import a sibling module. `check` runs C3 from another
  directory to catch this. A plugin can also be an installed module (`pkg.mod:name`); the key then pins the
  module's file digest. Only that file is covered, not other modules of its package.
- **Pinned code.** The key records the code that made it: the content hash of a file, or the digest of a module.
  A detector scores with exactly that code. An edited plugin file is refused rather than silently used.
- **Unique names.** A plugin cannot reuse a built-in scheme's name.
- **Trust.** A key from someone else can name any code. `detect`, `verify_model` and `calibrate` run plugin code
  only if your process already imported that exact file, or with `trust_plugin=True`. From the shell, a new
  process has imported nothing, so owners pass `--trust-plugin` for keys made with their own plugin. `bench` and
  the `exp/` scripts trust the keys under your own `$MWX_RUNS`, and `bench` imports the plugin files an arena names
  (also with `--dry-run`).
- **Key material goes through `save(key_dir, key)`.** `key_dir` is a private directory in the key store; the
  scheme never sees the release directory.

## Experiments and the leaderboard

One command runs the benchmark for a list of methods: learn, every edit, forgery, then the leaderboard
(`markweightx evaluate` is the same command, the appendix's name). A plugin file must pass `markweightx check`
first. A step counts as done once its output exists, so a bench can be stopped and run again; a failed step skips
the rest of that method, and `--queue` queues a failed step again under a new id.

```bash
markweightx bench configs/arena/1b.yaml --dry-run     # what is done, what would run
markweightx bench configs/arena/1b.yaml               # run the missing steps on this machine
markweightx bench my_arena.yaml --queue lab2          # or hand them to the GPU scheduler
```

```yaml
# my_arena.yaml
base: llama-3.2-1b
methods: [pro-coopt-c100, my_mark.py]                 # built-in names (with :params) or plugin files
edits: [finetune:steps=800, quantize:bits=4, text:synonym:ratio=0.3]   # optional; default: the full grid
```

The board is printed and saved as `$MWX_RUNS/e2/<base>/leaderboard.md`; `exp/leaderboard.py --format latex` prints
the main table as LaTeX. A plugin method gets a main-board row; parameter sweeps (`name:k=v`) and experimental
variants go under Ablations.

The steps are these scripts, which can also be run one at a time:

| Script | What it runs |
|---|---|
| `exp/learnability.py --scheme NAME` | E2: embed, sample the held-out C4 prompts (311 of the first 500 documents are long enough), TPR at 1% FPR against the base model's texts, PPL |
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
| `harness.evaluate(arena.yaml)` | `markweightx bench arena.yaml` (runs `exp/learnability.py`, `robustness.py`, `spoof.py`, `leaderboard.py`) |
| `Scheme.keygen / signal / recipe / detector` | `Scheme.setup` (key) / `target_logits` (Signal) / `configs/embed.yaml` / `score`, `pvalue` |
| CLI `verify` | `markweightx verify` → `mwx.verify_model` |

`verify_model(..., sequential=True)` asks the suspect model for 20 texts at a time. It accuses the model as soon as
an always-valid test (flag rate `p1` against `fpr`) is decisive, so a strongly marked model is settled in about 20
texts; it never clears a model early, so a weakly marked one (after edits) gets the full `budget` and the power of
the fixed test (`exp/verify_power.py`: at a 5% flag rate 87% vs 88%). False accusations stay below `alpha` and the
p-value stays valid despite the early stop. With an empirical null the threshold's own false-flag rate is uncertain
(a 311-text calibration: 1.3% on average, bounded by 3.5%), so half of `alpha` bounds it and the test runs against
the bound; a larger calibration gives more power. Prompts disjoint from the calibration's are the clean choice (the
bundled C4 split has only 311 usable prompts, so the quickstart reuses them; GaussMark's exact null is unaffected). `n_queries` says how many texts were generated.

Not built yet: importing the published checkpoints of other papers (`port` takes the checkpoint and its key files;
fetching and converting them is per paper, and they are 7B-8B), `TP-*` grants (prompt
access), the `sft` and `rl` imprints, the `distill_sampling` recipe, and in `bench` the paraphrase attack (DIPPER)
and a grant axis.

## Layout

```
markweightx/        api.py (the facade) · cli.py · bench.py · check.py · keys.py · registry.py · primitives.py · types.py
  schemes/          base.py (Scheme) · markllm.py · pro.py · gaussmark.py
  edits/            weights.py (finetune, merge, prune, quantize) · text.py (synonym, word_deletion)
  imprint.py · sampling.py · scoring.py · models.py · paths.py
configs/            embed.yaml · watermark/<scheme>.yaml (generated by exp/write_configs.py)
examples/           quickstart.py · my_scheme.py (the new-scheme template)
exp/                experiment scripts (above)
tests/              pytest -q tests  (add -m "not slow" to skip the end-to-end check runs)
```
