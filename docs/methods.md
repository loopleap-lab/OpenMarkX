# Methods

A watermark method in OpenMarkX is a **scheme**. It says how the secret key shifts each output unit (the training
signal), how the mark gets into the weights, and how an output is scored. This page lists the built-in methods,
what their detectors need, the checks a method must pass, and how to add your own.

- [Built-in methods](#built-in-methods)
- [Parameters](#parameters)
- [What the detector needs](#what-the-detector-needs)
- [Adding a method](#adding-a-method)
- [The checks a method must pass](#the-checks-a-method-must-pass)
- [Rules for plugin methods](#rules-for-plugin-methods)

## Built-in methods

`openmarkx describe` prints the full table, with the primitives each method uses.

| Scheme | Method | How the mark gets in | Detector needs |
|---|---|---|---|
| `openmark` | OpenMark, the paper's method: policy co-optimized with the model (CAWP), plus forgetting-perturbation training (FPL) | distill OpenMark's semantic teacher into the weights (LoRA), policy trained jointly, plus FPL | text, secret key (with its trained policy) |
| `openmark-coopt-c100` | OpenMark with the policy co-optimized, no FPL | same, without FPL | text, secret key (with its trained policy) |
| `openmark-frozen` | OpenMark with the policy frozen at its released initial weights, no FPL | distill OpenMark's semantic teacher into the weights (LoRA) | text, secret key |
| `markllm-kgw-d`, `-unigram-d`, `-sweet-d`, `-ewd-d`, `-morphmark-d`, `-dip-d`, `-unbiased-d`, `-synthid-d`, `-exp-d`, `-expedit-d`, `-itsedit-d`, `-pf-d` | 12 decoding-time watermarks from MarkLLM, distilled into the weights (Gu et al. style) | distill the MarkLLM teacher into the weights (LoRA) | text, secret key; SWEET, EWD and Unbiased also run the base model |
| `gaussmark` | GaussMark | a keyed Gaussian perturbation written into one weight matrix | text, secret key, gradients of the base model |
| `wmar` | WMAR's green list over image tokens (Taming), learned into the weights | distill; the green-list table can be co-learned with the model | image tokens, secret key |
| `ripple` | a keyed pattern on a field predicted over 3D points (GeoPT) | train by mean squared error to the marked field, or a closed-form edit of the last layer | the field, the publisher's unmarked model, secret key |
| `nowm` | no watermark | nothing | nothing: the scheme name of an unwatermarked release (e.g. the base model as a suspect in `exp/verify_check.py`); no arena or demo config lists it |

The 16 text methods above `wmar` are the ones `configs/arena/1b.yaml` benchmarks. Six of the 12 MarkLLM methods
(EXP, Unbiased, SynthID, PF, DIP, ITSEdit) do not learn into 1B weights (TPR at 1% FPR at most 0.03 after 1,500
steps), so that arena runs only their learn step. `wmar` and `ripple` run through
`openmarkx run` with the `taming` and `geopt` adapters.

**OpenMark.** The OpenMark watermark is the paper's code, shipped with the package in
`openmarkx/third_party/pro`. Its signal is one semantic bias per sequence: a sentence embedder reads the context,
the watermark policy maps the embedding to a bias, and the secret token mapping assigns that bias to tokens. The
embedder downloads from Hugging Face on first use. `openmark` trains the policy with the student and adds
forgetting-perturbation training: each step simulates a fine-tuning step toward the anti-watermarked distribution
and penalizes how much it moves the model (beta 0.1). That costs two more forward and backward passes per step,
about three times the compute (an estimate from the training loop, not a timing). `openmark-coopt-c100` trains the
policy without FPL; `openmark-frozen` keeps the policy at its released initial weights. `openmark-full-b01` is
another name of `openmark`. Older names keep their meaning ([FAQ](FAQ.md#old-names)).

**MarkLLM.** The `markllm-*-d` schemes use upstream MarkLLM at commit `112fa74`, unchanged, as teachers
(`scripts/install_deps.sh`). Fixes to upstream behaviour live in the adapter, `openmarkx/schemes/markllm.py`. Every
teacher samples at temperature 1. Not included: `markllm-expgumbel-d`. Upstream builds a (V·prefix)×V noise table,
128 GB at V = 128k, and it follows the same Gumbel-max rule as EXP.

**Experimental variants.** `openmark-coopt`, `openmark-coopt-c`, `openmark-booster`, `openmark-fpl`,
`openmark-fpl-b05` and `openmark-full` are ablations on the way to the three OpenMark settings above. They are
registered, hidden from `describe`, and go under Ablations on the leaderboard.

## Parameters

Parameters go after a colon: `gaussmark:sigma=0.02,layer=12`. In a run config they go under `scheme.params`. The
defaults of each scheme are in `configs/watermark/<scheme>.yaml`.

A misspelled scheme or parameter is an error that lists the accepted ones. `embed` writes every parameter into the
secret key, defaults included, so a key keeps detecting the same way after a default changes in the code.

## What the detector needs

| Level | The detector holds | Schemes |
|---|---|---|
| `R0` | output, tokenizer, secret key | KGW family, Unigram, EXP family, SynthID, DIP, PF, ITSEdit, `wmar` |
| `R0_AUX` | and an auxiliary network that ships in the key | OpenMark (its watermark policy) |
| `R1` | and forward passes of the base model | SWEET, EWD, Unbiased, `ripple` |
| `R2` | and gradients of the base model | GaussMark |

`detect(..., grant=Access.R1)` (CLI `--grant R1`) caps what a detector may touch. A scheme that needs more is
refused before any model is loaded.

## Adding a method

A new method is one Python file.

```bash
openmarkx new-scheme my-mark            # writes my_mark.py: Key, Signal, Score in about 40 lines
openmarkx check --adapter my_mark.py    # minutes, on CPU, against a tiny Llama
openmarkx embed --base llama-3.2-1b --watermark my_mark.py
```

The template, also in [`examples/my_scheme.py`](../examples/my_scheme.py), is a working method:

| Part | In the template |
|---|---|
| `name`, `imprint`, `access` | `"my-scheme"`, `Imprint.DISTILL`, `Access.R0` |
| `__init__` | the scheme's parameters, each with a default (`delta=2.0`) |
| `setup` (Key) | draws a secret with `omx.keys.new_secret()` for a new release, or reads it from an existing key; returns the key material |
| `target_logits` (Signal) | the teacher's logits for every position: a fixed green list, plus `delta` |
| `score` (Score) | a statistic (higher means more marked) and the number of tokens scored |
| `pvalue`, `exact_null` | an optional closed-form null; without one, the key's calibrated null is used |

Other hooks a scheme can override (`openmarkx/schemes/base.py`):

| Hook | Use |
|---|---|
| `imprint = Imprint.EDIT` and `imprint_edit(model, tokenizer, out)` | write the mark into the weights directly, with no training |
| `trainable_params`, `param_groups`, `aux_loss` | extra parameters trained with the student, such as OpenMark's policy |
| `save(key_dir, key)` | key material stored as files in a private key-store directory |
| `wrong_key(params, rng, workdir)` | the key with only its secret redrawn, for the wrong-key null |
| `score_batch(texts)` | score many outputs at once |

A mark on continuous outputs is a `FieldScheme`. `target_output(inputs, output)` gives the marked target for the
unmarked model's output, and `score(output, reference, inputs) -> (statistic, n)` compares a suspect output with
the unmarked model's output on the same inputs (access level R1). `openmarkx/schemes/ripple.py` is the worked
example. [Models](models.md) shows a scheme next to a model adapter.

## The checks a method must pass

`openmarkx check --adapter NAME|file.py` runs these checks on CPU against a tiny Llama. A method must pass them
before it joins the leaderboard, and `openmarkx bench` runs them on every plugin file it is given (unless
`--no-check`).

| Check | Passes when |
|---|---|
| C1 declared | imprint, access and kind are declared; kind is one `embed` trains; matches `configs/watermark/*.yaml` |
| C2 access | `detect` with a grant one level too low is refused before any model loads |
| C3 round trip | a fresh process holding only the secret key reproduces the scores (other RNG and hash seeds, another working directory, no model for R0) |
| C4 key | fresh keys give different scores; 24 keys give at most one identical pair (no tiny key space) |
| C5 null | closed-form p-values are not too small on short, long and looping human text (KS, 1% and 5% tails); marked text gets lower p than plain |
| C6 signal | text decoded with the Signal (or from the edited model) scores clearly above plain text |
| C7 isolation | an `EDIT` release holds only model files, none equal to key material or containing a key value |
| C8 determinism | rebinding the same key gives the same material and scores; `setup` leaves the model and tokenizer unchanged |
| C9 length | not run by `check` (it needs the real base); see below |
| C10 positions | `target_logits` on a whole sequence (training) equals it on each prefix (decoding) |

C9 checks false positives on shorter texts. `exp/learnability.py` cuts the evaluation texts to 50 and 100 tokens
and reports the false-positive rate at the full-length threshold. The leaderboard shows it as a C9 table, and
`exp/length_check.py` adds it to finished runs.

`tests/broken/` holds 17 adapters with typical bugs, and `tests/test_check.py` makes sure `check` fails each one.

## Rules for plugin methods

- **One file.** The secret key stores that file, so it cannot import a sibling module. `check` runs C3 from
  another directory to catch this. A plugin can also be an installed module (`pkg.mod:name`); the key then pins the
  module's file digest. Only that file is covered, not other modules of its package.
- **Pinned code.** The key records the code that made it: the content hash of a file, or the digest of a module. A
  detector scores with exactly that code. An edited plugin file is refused.
- **Unique names.** A plugin cannot reuse a built-in scheme's name.
- **Trust.** A key from someone else can name any code. `detect`, `verify_model` and `calibrate` run plugin code
  only if your process already imported that exact file, or with `trust_plugin=True`. From the shell, a new process
  has imported nothing, so owners pass `--trust-plugin` for keys made with their own plugin. `bench` and the `exp/`
  scripts trust the keys under your own `$OMX_RUNS`, and `bench` imports the plugin files an arena names (also with
  `--dry-run`).
- **Key material goes through `save(key_dir, key)`.** `key_dir` is a private directory in the key store; the scheme
  never sees the release directory.
- **Not in a script or notebook.** A scheme defined in the script being run cannot be loaded again by its key. Keep
  it in its own file and pass the file: `omx.embed(base, "my_mark.py")`.
