# OpenMarkX reference

The full reference behind the [README](../README.md): install details, every command and parameter, keys, what a
detector needs, the checks a new method must pass, the benchmark, and how `verify_model` decides.

## Install

Python 3.10+ and a CUDA GPU for training and sampling (`check` and the tests run on CPU).

```bash
git clone <this repository> && cd OpenMarkX
pip install -e ".[test]"            # the openmarkx package, the `openmarkx` command (and pytest)
bash scripts/install_deps.sh        # upstream MarkLLM at a pinned commit, and its requirements
python exp/prepare_data.py          # once: C4 text for training, prompts and human continuations
pytest -q tests -m "not slow"       # optional: about a minute on CPU
```

Everything goes under `~/.openmarkx` unless set otherwise:

| Variable | What | Default |
|---|---|---|
| `OMX_RUNS` | releases, results, tokenized corpus | `~/.openmarkx/runs` |
| `OMX_KEYS` | the key store (back it up: a lost key cannot be recovered) | `$OMX_RUNS/keys` |
| `MARKLLM_DIR` | upstream MarkLLM checkout (made by `install_deps.sh`) | `~/.openmarkx/MarkLLM` |
| `HF_HOME` | Hugging Face cache; base models download here on first use | Hugging Face's default |

Names from before the rename to OpenMarkX still work: an unset `OMX_*` variable falls back to its `MWX_*` namesake,
`~/.markweightx` is used when `~/.openmarkx` does not exist, `import markweightx` and the `markweightx` command run
OpenMarkX, and the scheme names `pro`, `pro-coopt-c100`, ... are the `openmark` ones.

Base model names (`llama-3.2-1b`, `llama-3.2-1b-instruct`, `llama-3.1-8b`, ...) are fetched from Hugging Face at a
pinned revision (`openmarkx/paths.py`); any local model directory or repo id also works. Gated models such as
`meta-llama/*` need `huggingface-cli login`.

## Quickstart

```python
import openmarkx as omx
from openmarkx.models import load_tokenizer
from openmarkx.sampling import eval_prompts

prompts = eval_prompts(load_tokenizer("llama-3.2-1b"), n=500)  # held-out C4 prompts (exp/prepare_data.py)
release, key = omx.embed("llama-3.2-1b", "gaussmark")        # Owner: mark the weights; the key stays private
key = omx.calibrate(key, "llama-3.2-1b")                     # freeze the null on the base model's own texts

child = omx.modify(release, "finetune:steps=200")            # Downstream: edit the weights...
texts = omx.generate(child, prompts)                         # ...and write with the edited model
texts = omx.rewrite(texts, "synonym:ratio=0.3")              # ...and reword

hits = omx.detect(texts, key)                                # Detector: [Detection(statistic, p_value, n_tokens, flagged)]
verdict = omx.verify_model(child, key, prompts, sequential=True, budget=200)   # one verdict for the model
```

[`examples/quickstart.py`](../examples/quickstart.py) runs this end to end on one GPU in a few minutes.

The same flow from the shell:

```bash
openmarkx embed --base llama-3.2-1b --watermark gaussmark            # prints {"release": ..., "key": ...}
openmarkx calibrate KEY --base llama-3.2-1b
openmarkx modify RELEASE --edit finetune:steps=200 --out edited/
openmarkx generate edited/ --prompts prompts.jsonl --out texts.jsonl
openmarkx rewrite texts.jsonl --edit synonym:ratio=0.3 --out reworded.jsonl
openmarkx detect texts.jsonl --key KEY                                 # --null wrong_key: no calibration needed
openmarkx port their_lora/ --watermark openmark --key-file key.json --base llama-3.2-1b   # a checkpoint made elsewhere
openmarkx verify edited/ --key KEY --prompts prompts.jsonl            # one verdict for the model
openmarkx verify edited/ --key KEY --prompts prompts.jsonl --sequential --budget 200   # as few texts as needed
openmarkx --help                                                       # every command, with examples
```

A KEY is a key id from `openmarkx keys list`, a unique prefix of one, or a release directory. Text files are
`.jsonl` (`{"text": ...}` per line) or `.txt` (one text per line).

Edits are written `op:param=value,...`:

| Edit | Params (defaults) |
|---|---|
| `finetune` | `steps=800, lr=1e-4, bs=8, seq_len=256, rank=16`: LoRA on clean text, merged into the weights |
| `merge` | `partner=` (required, e.g. `llama-3.2-1b-instruct`), `ratio=0.4` (the partner's weight) |
| `prune` | `sparsity=0.5, method=magnitude` |
| `quantize` | `bits=4` (nf4) or `bits=8` (int8) |
| `synonym`, `word_deletion`, `context_synonym` | `ratio=0.3` (text edits, used with `rewrite`) |

## Run a whole experiment (any model)

`openmarkx run CONFIG` runs one experiment from a YAML file, for any model adapter: text, `agent` (a ReAct
agent around a text model), `vlm` (image + question in, text out), `taming` (image tokens) and `geopt` (a physics
field on 3D points). It runs these steps in order and writes `<out>/report.json`:

| Step | What it does |
|---|---|
| `prepare` | the training data and the evaluation inputs (`data: {name: synthetic}` makes a small stand-in set) |
| `selfgen` | the base model's own outputs to train on (taming samples image tokens; the others bring their data) |
| `calibrate` | a strength search for field marks (`eps: auto`; not yet supported for `ripple`, which takes a fixed `eps`) |
| `imprint` | learn the mark into the weights; the release goes to `<out>/release`, the key to the key store |
| `generate` | outputs of the base and of the release on the evaluation inputs (images also as PNGs in `<out>/assets`) |
| `detect` | the detector's statistic on both sets; TPR at the FPR against the null (base outputs, or wrong keys) |
| `edits` | the same after each edit |
| `quality` | `ppl`, `accuracy`, `fid`, `rel_l2` or `none`, for the base (before) and the release (after) |
| `report` | `report.json`: model, scheme, detect, quality, edits, settings |

Each step writes `<out>/<step>.json` and is skipped when that file exists, so a run stopped by a time limit
continues where it stopped; once a step runs again, every later step runs again too. `--only STEP ...` runs some
steps (and clears the outputs of every later step, which then run again), `--smoke` caps training at 5 steps per
phase and the samples at 4 and writes to `<out>_smoke` (to check a setup), and `<out>/progress.jsonl` has one line
per step event. An `out` that holds a run with other settings is refused. An interrupted imprint keeps its progress
and its draft key in the key store, so the resumed run trains toward the same key; nothing of the key is ever written
into the release directory.

```yaml
model: {name: geopt, ckpt: $OMX_GEOPT_DIR/checkpoints/craft_geopt_8layers.pt, data: $OMX_DATA/geopt/aircraft_npys}
scheme: {name: ripple, params: {eps: 0.02, output_scale: 1.0}}    # or source: my_mark.py
imprint:
  edit_last: true                       # geopt only: write the mark into the last layer in closed form first
  phases:                               # one or more; each: steps, lr, trainable (lora|full), rank, policy_lr, fix_rule
    - {steps: 0, lr: 1.0e-5, trainable: full}
detect: {fpr: 0.01, null: wrong_keys}   # base | wrong_keys
edits: ["finetune:steps=200,lr=1e-4"]
quality: rel_l2
out: $OMX_RUNS/demos/geopt
```

Top-level keys: `model` (`name`, the adapter's own parameters, or `source:` a plugin file), `scheme` (`name`,
`params`, `source`), `data`, `imprint` (`phases`, `bs`, `accum`, `seq_len`, `answer_len`, `save_every`, `seed`,
`edit_last`), `generate` (`n`, `new_tokens`, `bs`), `detect` (`fpr`, `null`, `n_null`), `edits`, `quality`,
`out`, `smoke`. An unknown key, model, scheme, null, edit or quality name stops the run before anything loads, with
the allowed names; so does a setting the chosen model does not use (a model parameter, a `data` key, a `generate` key
or an `imprint` key such as `edit_last` outside geopt) and a scheme parameter the scheme does not take. `$VARS` in values come from the environment; `$OMX_DATA` (inputs) defaults to
`$OMX_RUNS` (outputs).

Phases give multi-stage recipes. Taming's demo co-learns the green lists with the model, then fixes them and
keeps training:

```yaml
imprint:
  phases:
    - {steps: 800, lr: 3.0e-5, trainable: full, policy_lr: 0.2}
    - {steps: 1000, lr: 1.0e-4, trainable: full, policy_lr: 0, fix_rule: true}
```

The demos on the website are four configs, `configs/demos/{taming,medgemma,geopt,agent}.yaml`;
`tests/test_demo_configs.py` checks each setting against the run files they came from. Taming and GeoPT use the
authors' code, unchanged, from your own checkouts: set `OMX_WMAR_DIR` (WMAR, which includes Taming) and
`OMX_GEOPT_DIR` (GeoPT). Without them the run stops with a message naming the variable.

A release made with a stand-in (`factory`, `processor`) or a plugin adapter (`source`) names code to run; opening it
outside its own run needs `--trust-plugin` (`trust=True`).

`openmarkx detect` reads these outputs too: `--model taming` takes a directory of PNG images (encoded with the
release's VQ, scored against the key's frozen null), `--model geopt` an `.npz` with `points`, `field` and
`condition` (scored against the release's unmarked model on the same inputs, with a wrong-key null):

```bash
openmarkx detect images/ --model taming --key runs/taming/release     # --release defaults to --key
openmarkx detect case.npz --model geopt --key runs/geopt/release
```

On HiPerGator, `scripts/hpg/run.sbatch CONFIG [--smoke]` runs the same command in one GPU job.

## Run from a web page

`openmarkx serve` lets you upload a dataset, pick a method and its hyperparameters, and learn a watermark on this
machine's GPUs from a browser. It prints an address with a token; open it in a browser. Each job trains on your texts, calibrates the key
on the base model, then reports how many held-out texts from the watermarked model are detected and how many human
texts are falsely flagged. Data, releases and keys stay on the machine (`$OMX_SERVE`, default `~/.openmarkx/serve`,
and the key store). For a GPU on another machine, run the server there and forward the port:

```bash
ssh -L 7860:127.0.0.1:7860 gpu-server      # then connect to http://127.0.0.1:7860
```

The page runs the text recipe above; running a `openmarkx run` config from it is planned.

Without a GPU, [open the notebook in Colab](https://colab.research.google.com/github/UCF-ML-Research/OpenMarkX/blob/main/notebooks/run_on_colab.ipynb)
(`notebooks/run_on_colab.ipynb`): it installs OpenMarkX on a free T4 and starts the server.

## Supported methods

`openmarkx describe` prints the full table. A method is a **scheme**, assembled from six primitives: Key, Signal,
Imprint, Perturb, Probe and Score. The leaderboard reports the paper's methods; the others are supported by the
toolkit and can be benchmarked with `openmarkx bench`.

| Scheme | Method | How the mark gets in | Detector needs |
|---|---|---|---|
| `openmark` | OpenMark (ours), watermark policy frozen | distill OpenMark's semantic teacher into the weights (LoRA) | text, key (with its policy) |
| `openmark-coopt-c100` | OpenMark with the policy co-optimized during distillation | same, policy trained jointly | text, key |
| `openmark-full-b01` | OpenMark co-optimized, with forgetting-perturbation training (FPL) | same, plus FPL | text, key |
| `markllm-kgw-d`, `-unigram-d`, `-sweet-d`, `-ewd-d`, `-morphmark-d`, `-dip-d`, `-unbiased-d`, `-synthid-d`, `-exp-d`, `-expedit-d`, `-itsedit-d`, `-pf-d` | 12 decode-time watermarks from MarkLLM, distilled (Gu et al. style) | distill the MarkLLM teacher into the weights (LoRA) | text, key; SWEET, EWD, Unbiased also run the base model |
| `gaussmark` | GaussMark | a keyed Gaussian perturbation written into one weight matrix | text, key, gradients of the base model |

The OpenMark watermark ships with the package (`openmarkx/third_party/pro`, the paper's code); its sentence embedder
downloads from Hugging Face on first use. The `markllm-*` schemes need MarkLLM (`scripts/install_deps.sh`). Your own method is one file (see "Add your own method").

Parameters go after a colon: `gaussmark:sigma=0.02,layer=12`. A misspelled scheme or edit parameter is an error
that lists the accepted ones. So is an edit value the code would otherwise ignore, such as `quantize:bits=3`. `embed` writes every parameter into the key, defaults included. So a key keeps detecting the
same way after a default changes in the code.

Not included: `markllm-expgumbel-d`. Upstream builds a (V·prefix)×V noise table, 128 GB at V = 128k, and it
follows the same Gumbel-max rule as EXP.

## Bring a checkpoint made elsewhere

`port` wraps a watermarked checkpoint that was trained outside OpenMarkX, together with the key material it was
made with, as a Release and a Key. Nothing is retrained and the weights stay where they are.

```python
release, key = omx.port("their_lora/", "openmark", {"mapping": "map.json", "transform": "policy.pth"}, "llama-3.2-1b")
release, key = omx.port("their_model/", "markllm-kgw-d", {"secret": 123456}, "llama-3.2-1b", key_public=True)
```

`key_public=True` marks a key that was shipped with the checkpoint, e.g. a paper's released mapping. Anyone can
detect with such a key, and anyone can also remove or forge the mark, so `detect` warns.

## What the detector needs (access levels)

| Level | The detector holds | Schemes |
|---|---|---|
| `R0` | text, tokenizer, key | KGW family, Unigram, EXP family, SynthID, DIP, PF, ITSEdit |
| `R0_AUX` | + an auxiliary network that ships in the key | OpenMark (its watermark policy) |
| `R1` | + forward passes of the base model | SWEET, EWD, Unbiased |
| `R2` | + gradients of the base model | GaussMark |

`detect(..., grant=Access.R1)` (CLI `--grant R1`) caps what a detector may touch. A scheme that needs more is
refused before any model is loaded.

## With MarkAgentX

`Detection` is MarkAgentX's `Detection(statistic, p_value, n_evidence)` with the evidence counted in tokens:
`d.n_evidence` and `d.to_markagentx()` give its fields, and as there an empty text answers `p_value = 1.0` with no
evidence. `detect(..., null="wrong_key")` (CLI `--null wrong_key`) is MarkAgentX's wrong-key null: each text is
scored under `n_null` keys drawn from `random.Random(0xC0FFEE)`, and `p = (1 + #{wrong keys >= S}) / (n_null + 1)`.
A wrong key is the true key with only its secret redrawn (`Scheme.wrong_key`; OpenMark redraws its mapping and keeps
its policy). It needs no calibration but costs `n_null` rebindings of the scheme; teacher-free schemes are refused,
since their closed-form null is already exact. The draws are reproducible but are not MarkAgentX's keys: each scheme
draws its own kind of secret. `tests/test_markagentx.py` checks the fields against a MarkAgentX checkout
(`refs/MarkAgentX`) and builds its `Detection` from `to_markagentx()`.

## Keys

Keys never ship with a release. `release.json` holds only the key's commitment, which is sha256 over a nonce, the
scheme, its parameters and the plugin source. The key lives in the key store, `$OMX_KEYS` (default
`$OMX_RUNS/keys`). Each key carries its own fresh secret: MarkLLM's public default hash keys are replaced, and OpenMark
draws its own token mapping. A calibrated key carries a frozen empirical null, so `detect` returns calibrated
p-values. The calibration also holds nulls for the texts cut to 20, 50 and 100 tokens: a text is judged against the
null calibrated at the largest length not above its own, because some detectors' false positives grow on short texts
(check C9). Keys
calibrated before this change warn until calibrated again. A text too short to be judged gets `p_value=None` and
`flagged=None` (no evidence either way), and `verify_model` does not count it. Schemes with an exact closed-form null (`exact_null = True`, e.g. GaussMark) use it
directly.

```bash
openmarkx keys list
openmarkx keys show KEY        # prints metadata only; key material stays hidden
```

Keys made before 2026-10-06 store only the parameters that were typed, not the defaults. Their parameters are part
of the commitment, so they cannot be filled in afterwards. Such a key detects with the scheme's current defaults.

## Extending OpenMarkX

Six things can be added, each in one place. All of them reach the benchmark the same way: name them in an arena
file, run `openmarkx bench`, and they get a row or a column on the same leaderboard as the built-ins.

| To add | You write | You run | It shows up as |
|---|---|---|---|
| A watermark method | one `.py` file from `openmarkx new-scheme` | `openmarkx check --adapter my_mark.py` | a row |
| A weight edit (a new attack on the released model) | a function with `@omx.register_edit` | `openmarkx modify REL --edit name:k=v --plugin file.py` | a column |
| A text edit (a new attack on the generated text) | a function with `@omx.register_text_edit` | `openmarkx rewrite texts.jsonl --edit name:k=v --plugin file.py` | a column |
| A dataset or domain (e.g. physics) | nothing: point at a Hugging Face dataset or a `.jsonl` | `exp/prepare_data.py --dataset ...`, then `OMX_DATA=<file>` | the same board, on your text |
| A base model | nothing for Llama-style models; optionally a pinned entry in `BASES` | `--base <hf-id or local dir>` | a board per base |
| A model of another kind (images, a VLM, an agent, a physics field) | a `Model` adapter with `@omx.register_model` | `openmarkx run config.yaml` with `model: {name: ..., source: my_model.py}` | its own run and report |

`examples/my_edits.py` holds one weight edit and one text edit, and `configs/arena/plugins_example.yaml` runs them
next to the built-ins; `tests/test_edit_plugins.py` runs both on CPU.

### Add a weight edit

A weight edit is what someone downstream does to the released weights. It takes the release, an output directory,
its own parameters (each with a default) and `device`, and returns the edited release. Load the release as a plain
model with `materialize`, change it, and hand it to `edited_release`, which saves it and extends the release's
lineage:

```python
# my_edits.py
import torch
import openmarkx as omx
from openmarkx.edits.weights import edited_release, materialize

@omx.register_edit
@torch.no_grad()
def noise(rel, out, *, scale=0.01, seed=0, device="cuda:0"):
    """Gaussian noise on every decoder linear weight, scaled to each matrix's spread."""
    model = materialize(rel, device)
    g = torch.Generator(device=device).manual_seed(seed)
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and ".layers." in name:
            W = mod.weight
            W.add_(torch.randn(W.shape, generator=g, device=W.device, dtype=W.dtype) * W.float().std().to(W.dtype) * scale)
    return edited_release(rel, out, "noise", dict(scale=scale, seed=seed), model)
```

```bash
openmarkx modify RELEASE --edit noise:scale=0.01 --plugin my_edits.py --out noised/
```

In Python, `omx.load_plugins("my_edits.py")` (or a plain `import`) and then `omx.modify(rel, "noise:scale=0.01")`.
A misspelled parameter (`noise:scael=0.01`) is refused before any model is loaded. Edits that need data, such as
fine-tuning on a domain corpus, take a `data=` parameter like the built-in `finetune` does
(`finetune:steps=800,data=/path/to/data_physics.pt`).

### Add a text edit

A text edit is what someone does to the generated text: paraphrase, translate and back, add typos. It takes the
list of texts and its parameters, and returns one string per text. Add `device` to its parameters if it runs a model.

```python
@omx.register_text_edit
def typos(texts, *, rate=0.05, seed=0):
    """Swap two neighbouring letters in that share of the words."""
    ...                                   # full code in examples/my_edits.py
```

```bash
openmarkx rewrite texts.jsonl --edit typos:rate=0.05 --plugin my_edits.py --out typoed.jsonl
openmarkx detect typoed.jsonl --key KEY     # does the mark survive the edit?
```

The built-ins (`synonym`, `word_deletion`, `context_synonym`) are registered the same way, in
`openmarkx/edits/text.py`; an edit that returns the wrong number of texts is refused.

### Benchmark the new edits

```yaml
# configs/arena/plugins_example.yaml
base: llama-3.2-1b
methods: [openmark, markllm-kgw-d]
plugins: [examples/my_edits.py]          # loaded before the edits are checked, and by every step that runs them
edits: [finetune:steps=800, noise:scale=0.01, text:synonym:ratio=0.3, text:typos:rate=0.05]
```

`bench` checks every edit and its parameters when it reads the file, so a typo fails at once rather than on a GPU
an hour later. Each new edit becomes a column (`noise_scale0.01`, `typos_rate0.05`) after the default ones. For
scale, these are the default columns of the 1B board for OpenMark (TPR at 1% FPR, from our runs, `$OMX_RUNS/e2/llama-3.2-1b/leaderboard.md`):

| method | learned | FT 800 | merge .4 | prune 20% | nf4 | delete 30% | synonym 30% |
|---|---|---|---|---|---|---|---|
| openmark | 0.88 | 0.07 | 0.55 | 0.89 | 0.89 | 0.83 | 0.67 |

and the paper's numbers for the same kinds of edit on Llama-3-8B are in [LEADERBOARD.md](../LEADERBOARD.md)
(e.g. after 4-bit HQQ OpenMark keeps 0.91 against 0.77 for Gloaguen et al.; after 300 steps of math fine-tuning, 0.43
against 0.22). A new edit is read against these: close to `nf4` means harmless, close to `FT 800` means it erases the mark.

### Use your own dataset or domain

Every step reads one tokenized corpus: training text for `embed`, clean text for the fine-tune edit, and held-out
documents whose first 50 tokens are the evaluation prompts and the calibration prompts. Build one from any text:

```bash
python exp/prepare_data.py --dataset camel-ai/physics --field message_2 --out $OMX_RUNS/data_physics.pt
python exp/prepare_data.py --dataset my_papers.jsonl --out $OMX_RUNS/data_mine.pt     # {"text": ...} per line, or .txt
export OMX_DATA=$OMX_RUNS/data_physics.pt     # every step now trains, attacks and evaluates on this text
```

The first `--n_eval` documents (600) are held out for prompts; at least 250 tokens each are needed for a prompt
plus a 200-token continuation, so short answers give few prompts (the C4 file keeps 311 of the first 500). The
fine-tune edit reads chunks past 10,000, so keep `--n_train` above that. To change only one step, pass the file
to it instead: `embed --recipe` with `data: <file>`, or the edit `finetune:data=<file>`. Check a dataset's license
before training on it (`camel-ai/physics` is CC BY-NC 4.0).

### Use another base model

`--base` takes a name from `BASES` in `openmarkx/paths.py` (pinned to a Hugging Face revision), a Hugging Face
repo id, or a local directory. What a base needs:

- **A causal language model** that `AutoModelForCausalLM` loads. Image + text models in the Gemma 3 style
  (MedGemma) use the `vlm` adapter; any other kind of model needs a `Model` adapter (next section).
- **LoRA target names** `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`, as in Llama, Mistral and
  Qwen2 (`LORA_TARGETS` in `openmarkx/imprint.py`; the fine-tune edit uses the same list).
- **A corpus in its tokenizer**: `prepare_data.py --base <the base>`. Llama-3.2-1B and Llama-3.1-8B share one.
- **A merge partner** for the merge edit: `<base>-instruct` in `BASES`, or `merge:partner=<repo id>` in the arena.

Add a line to `BASES` to pin the revision, so a run is comparable across machines:

```python
"qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B", "<commit sha>"),
```

### Add a model

A model adapter tells OpenMarkX how to load a model, run it, which weights the imprint trains, and what one
output unit is. Subclass `omx.Model` in one file:

```python
import openmarkx as omx

@omx.register_model
class MyModel(omx.Model):
    name = "my_model"
    unit = "token"                  # what one output unit is, for reports
    output = omx.Output.TOKENS      # TOKENS: trained by KL to the scheme's target_logits; FIELD: by MSE to target_output

    def load(self, path, device): ...          # the base (path None) or a release, as a torch module
    def run(self, net, inputs, **kw): ...      # a batch of inputs in, outputs out
    def forward(self, net, batch): ...         # teacher-forced pass for training: (inputs, logits or output)
    def codec(self): ...                       # outputs <-> what the scheme scores (e.g. a tokenizer, a VQ)
```

`trainable(net, "lora" | "full")` and `lora_targets` have defaults. Parameters in the config's `model:` block
reach `__init__` and are saved in the release (`release.json` gains `model: {name, params}`, `format: hf |
state_dict` and `schema: 2`; a release without them reads as a text model), so `detect` and `verify` rebuild the
same adapter. Code your adapter imports from someone else's checkout goes through `openmarkx.external.
require_external(name, ENV_VAR, module)`, which stops with a message naming the variable when it is missing.

A mark on continuous outputs is a `FieldScheme`: `target_output(inputs, output)` gives the marked target for the
unmarked model's output, and `score(output, reference, inputs) -> (statistic, n)` compares a suspect output with
the unmarked model's output on the same inputs (detection needs the publisher's model, access level R1).
`openmarkx/schemes/ripple.py` is the worked example; `openmarkx/models/geopt.py` and `taming.py` are worked
adapters, and `tests/fakes.py` holds the CPU stand-ins their tests use.

### Add a watermark method

The rest of this section; it is the most involved of the six because the key, the training signal and the
detector all have to agree.

#### The steps

```bash
openmarkx new-scheme my-mark            # writes my_mark.py: Key, Signal, Score in ~40 lines
openmarkx check --adapter my_mark.py    # minutes, on CPU, against a tiny Llama
openmarkx embed --base llama-3.2-1b --watermark my_mark.py
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
  the `exp/` scripts trust the keys under your own `$OMX_RUNS`, and `bench` imports the plugin files an arena names
  (also with `--dry-run`).
- **Key material goes through `save(key_dir, key)`.** `key_dir` is a private directory in the key store; the
  scheme never sees the release directory.

## Experiments and the leaderboard

One command runs the benchmark for a list of methods: learn, every edit, forgery, then the leaderboard
(`openmarkx evaluate` is the same command, the appendix's name). A plugin file must pass `openmarkx check`
first. A step counts as done once its output exists, so a bench can be stopped and run again; a failed step skips
the rest of that method, and `--queue` queues a failed step again under a new id.

```bash
openmarkx bench configs/arena/1b.yaml --dry-run     # what is done, what would run
openmarkx bench configs/arena/1b.yaml               # run the missing steps on this machine
openmarkx bench my_arena.yaml --queue mybox         # or hand them to the GPU scheduler (exp/scheduler.py)
```

```yaml
# my_arena.yaml
base: llama-3.2-1b
methods: [openmark-coopt-c100, my_mark.py]                 # built-in names (with :params) or plugin files
edits: [finetune:steps=800, quantize:bits=4, text:synonym:ratio=0.3]   # optional; default: the full grid
```

The board is printed and saved as `$OMX_RUNS/e2/<base>/leaderboard.md`; `exp/leaderboard.py --format latex` prints
the main table as LaTeX. A plugin method gets a main-board row; parameter sweeps (`name:k=v`) and experimental
variants go under Ablations.

The steps are these scripts, which can also be run one at a time:

| Script | What it runs |
|---|---|
| `exp/learnability.py --scheme NAME` | E2: embed, sample the held-out C4 prompts (311 of the first 500 documents are long enough), TPR at 1% FPR against the base model's texts, PPL |
| `exp/robustness.py --release DIR` | E4: every weight and text edit on one release, with the same ruler |
| `exp/spoof.py --release DIR` | forgery: count tokens in the owner's texts, bias the base model, measure how much is flagged |
| `exp/leaderboard.py` | one row per method; experimental variants and sweeps go under Ablations |
| `exp/scheduler.py --host NAME` | a GPU queue (`$OMX_RUNS/sched/<host>/queue.json`) that waits for free memory |

The training recipe for teacher-learned schemes is `configs/embed.yaml` (`embed --recipe`). The board of your own
runs is separate from LEADERBOARD.md, which reports the paper.

## Names in the design appendix vs. the code

| Technical appendix | Code |
|---|---|
| `T-R0` / `T-R1` / `T-R2` | `Access.R0` / `R1` / `R2`; the auxiliary-network case is `R0_AUX` |
| `KeyBundle` | `Key` (`openmarkx.types`), stored by commitment in the key store |
| `WDetection` | `Detection(statistic, p_value, n_tokens, flagged)` |
| `modify(release, attack)`, `ft:full`, `quant` | `modify(release, "finetune:steps=800")`, `"quantize:bits=4"`, `"merge:partner=...,ratio=0.4"`, `"prune:sparsity=0.5"` |
| `harness.attack` (text) | `rewrite(texts, "synonym:ratio=0.3")`, `"word_deletion:ratio=0.3"` |
| `harness.evaluate(arena.yaml)` | `openmarkx bench arena.yaml` (runs `exp/learnability.py`, `robustness.py`, `spoof.py`, `leaderboard.py`) |
| `Scheme.keygen / signal / recipe / detector` | `Scheme.setup` (key) / `target_logits` (Signal) / `configs/embed.yaml` / `score`, `pvalue` |
| CLI `verify` | `openmarkx verify` → `omx.verify_model` |

`verify_model(..., sequential=True)` asks the suspect model for 20 texts at a time. It accuses the model as soon as
an always-valid test (flag rate `p1` against `fpr`) is decisive, so a strongly marked model is settled in about 20
texts; it never clears a model early, so a weakly marked one (after edits) gets the full `budget` and the power of
the fixed test. False accusations stay below `alpha` despite the early stop, and the verdict is flagged exactly when
`p_value <= alpha`. With an exact null (GaussMark) the flags of an unmarked model are Binomial(n, fpr); with an
empirical null of N texts they are Beta-binomial, because the suspect's texts are exchangeable with the
calibration's, which accounts for the threshold's own uncertainty (1.3% on average for 311 texts). That is exact for
texts judged against a null of their own length and conservative across the per-length nulls. Found at a 5% flag
rate with a budget of 200 (`exp/verify_power.py [--null-n 311]`):

| null | sequential | fixed |
|---|---|---|
| exact | 88% | 88% |
| empirical, 311 texts | 42% | 55% |

A larger calibration narrows the gap. Prompts disjoint from the calibration's are the clean choice (the bundled C4
split has only 311 usable prompts, so the quickstart reuses them; GaussMark's exact null is unaffected). `n_queries` says how many texts were generated.

Not built yet: importing the published checkpoints of other papers (`port` takes the checkpoint and its key files;
fetching and converting them is per paper, and they are 7B-8B), `TP-*` grants (prompt
access), the `sft` and `rl` imprints, the `distill_sampling` recipe, and in `bench` the paraphrase attack (DIPPER)
and a grant axis.

## Layout

```
openmarkx/        api.py (the facade) · cli.py · bench.py · check.py · keys.py · registry.py · primitives.py · types.py
  models/           base.py (Model, Output) · text.py · agent.py · vlm.py · taming.py · geopt.py
  schemes/          base.py (Scheme, FieldScheme) · markllm.py · openmark.py · gaussmark.py · wmar.py · ripple.py
  edits/            weights.py (finetune, merge, prune, quantize) · text.py (synonym, word_deletion) · outputs.py (image_roundtrip)
  pipeline.py (openmarkx run) · train.py (phased imprint) · imprint.py · external.py · quality.py
  sampling.py · scoring.py · vlm.py · paths.py
configs/            embed.yaml · watermark/<scheme>.yaml (generated by exp/write_configs.py) · arena/*.yaml · demos/*.yaml
scripts/hpg/        run.sbatch (openmarkx run in one HiPerGator GPU job)
examples/           quickstart.py · my_scheme.py (the new-scheme template) · my_edits.py (a weight edit and a text edit)
exp/                experiment scripts (above)
tests/              pytest -q tests  (add -m "not slow" to skip the end-to-end check runs)
```
