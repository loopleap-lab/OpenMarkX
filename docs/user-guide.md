# User guide

How to install OpenMarkX, learn a watermark into a model, test it under edits and detect it with the secret key.
[How OpenMarkX works](architecture.md) explains the layers and the run pipeline. [Methods](methods.md),
[Models](models.md) and [Benchmark](benchmark.md) cover the methods, your own model, and the edits and the
benchmark.

- [Installation](#installation)
- [Where runs and keys live](#where-runs-and-keys-live)
- [Running an experiment from a config](#running-an-experiment-from-a-config)
- [Language models, step by step](#language-models-step-by-step)
- [Python API](#python-api)
- [Keys](#keys)
- [Detecting and verifying](#detecting-and-verifying)
- [Bringing a checkpoint made elsewhere](#bringing-a-checkpoint-made-elsewhere)
- [Running from a web page](#running-from-a-web-page)
- [Your own dataset](#your-own-dataset)
- [Another base model](#another-base-model)
- [Reproducing the paper](#reproducing-the-paper)

## Installation

From source, which also gives you the configs, the examples and the experiment scripts:

```bash
git clone https://github.com/UCF-ML-Research/OpenMarkX.git
cd OpenMarkX
pip install -e ".[test]"            # the openmarkx package, the `openmarkx` command, and pytest
bash scripts/install_deps.sh        # upstream MarkLLM at a pinned commit, and its requirements
python exp/prepare_data.py          # once: C4 text for training, prompts and human continuations
pytest -q tests -m "not slow"       # optional: the CPU tests
```

Or with pip from GitHub, for the library alone:

```bash
pip install "openmarkx @ git+https://github.com/UCF-ML-Research/OpenMarkX"
```

- OpenMarkX needs Python 3.10 or later.
- Training and sampling need a CUDA GPU. `openmarkx check` and the tests run on CPU.
- `scripts/install_deps.sh` clones MarkLLM at commit `112fa74` and downloads the NLTK data the text edits use.
  The `markllm-*` methods need it. The OpenMark watermark ships with the package.
- The `vision` extra (`pip install -e ".[vision]"`) adds Pillow for the `vlm` and `taming` adapters.
- The command is `openmarkx`. In Python, the package is imported as `import openmarkx as omx`.

Without a GPU, [open the notebook in Colab](https://colab.research.google.com/github/UCF-ML-Research/OpenMarkX/blob/main/notebooks/run_on_colab.ipynb)
(`notebooks/run_on_colab.ipynb`). It installs OpenMarkX on a free T4 and starts the web server (see
[Running from a web page](#running-from-a-web-page)).

## Where runs and keys live

Everything goes under `~/.openmarkx` unless an environment variable says otherwise:

| Variable | What | Default |
|---|---|---|
| `OMX_HOME` | the root of the locations below | `~/.openmarkx` |
| `OMX_RUNS` | releases, results, tokenized corpus | `$OMX_HOME/runs` |
| `OMX_KEYS` | the key store (back it up: a lost secret key cannot be recovered) | `$OMX_RUNS/keys` |
| `OMX_DATA` | the tokenized corpus file made by `exp/prepare_data.py` | `$OMX_RUNS/data_llama8b_256.pt` |
| `OMX_DEMO_DATA` | the directory the demo configs read their inputs from (`exp/agent` and `exp/med` write there) | `$OMX_RUNS` |
| `OMX_SERVE` | uploads and jobs of `openmarkx serve` | `$OMX_HOME/serve` |
| `MARKLLM_DIR` | upstream MarkLLM checkout (made by `install_deps.sh`) | `$OMX_HOME/MarkLLM` |
| `OMX_WMAR_DIR`, `OMX_GEOPT_DIR` | checkouts of WMAR (with Taming) and GeoPT, for the `taming` and `geopt` adapters | unset |
| `HF_HOME` | Hugging Face cache; base models download here on first use | Hugging Face's default |
| `OMX_DEBUG` | `1` prints the full traceback when a command fails | unset |

Run configs can use these variables; an unset one takes the default above, so `$OMX_RUNS/demos/geopt` is
`~/.openmarkx/runs/demos/geopt`.

A release is a directory with `release.json` and the weights. It holds a commitment to its secret key, never the
key itself. The key goes to the key store. Publish the release directory and back up the key store.

Names from before the rename to OpenMarkX still work:

| Old name | Now |
|---|---|
| `import markweightx` | `import openmarkx` (with a deprecation warning) |
| the `markweightx` command | `openmarkx` |
| `MWX_*` variables | read when the `OMX_*` variable is unset |
| `~/.markweightx` | used when `~/.openmarkx` does not exist |
| schemes `pro`, `pro-coopt-c100`, `pro-full-b01`, ... | `openmark`, `openmark-coopt-c100`, `openmark-full-b01`, ... |

## Running an experiment from a config

`openmarkx run CONFIG` runs one experiment from a YAML file, for any model adapter: `text`, `agent` (a ReAct agent
around a text model), `vlm` (image and question in, text out), `taming` (image tokens) and `geopt` (a physics field
on 3D points). It runs these steps in order and writes `<out>/report.json`:

| Step | What it does |
|---|---|
| `prepare` | the training data and the evaluation inputs (`data: {name: synthetic}` makes a small stand-in set) |
| `selfgen` | the base model's own outputs to train on (taming samples image tokens; the others bring their data) |
| `calibrate` | a strength search for field marks (`eps: auto`; not supported for `ripple`, which takes a fixed `eps`) |
| `imprint` | learn the mark into the weights; the release goes to `<out>/release`, the secret key to the key store |
| `generate` | outputs of the base and of the release on the evaluation inputs (images also as PNGs in `<out>/assets`) |
| `detect` | the detector's statistic on both sets; TPR at the FPR against the null (base outputs, or wrong keys) |
| `edits` | the same after each edit |
| `quality` | `ppl`, `accuracy`, `fid`, `rel_l2` or `none`, for the base (before) and the release (after) |
| `report` | `report.json`: model, scheme, detect, quality, edits, settings |
| `viewer` | `<out>/assets/viewer.json`: the steps, scores and one worked example, for the run viewer on the website |

The demos on the website are configs in `configs/demos/`:

```bash
openmarkx run configs/demos/geopt.yaml          # physics: GeoPT aircraft fields (needs OMX_GEOPT_DIR)
openmarkx run configs/demos/geopt_colearn.yaml  # GeoPT with the ripple rule trained together with the model
openmarkx run configs/demos/taming.yaml         # images: Taming Transformers (needs OMX_WMAR_DIR and TAMING)
openmarkx run configs/demos/medgemma.yaml       # medical VQA: MedGemma 1.5
openmarkx run configs/demos/agent.yaml          # a ReAct agent on Llama-3.1-8B-Instruct
```

Taming and GeoPT use the authors' code, unchanged, from your own checkouts: set `OMX_WMAR_DIR` (WMAR, which
includes Taming) and `OMX_GEOPT_DIR` (GeoPT). Without them the run stops with a message naming the variable. The
data the MedGemma and agent demos read is made by the scripts in `exp/med/` and `exp/agent/`, under
`$OMX_DEMO_DATA` (default `$OMX_RUNS`). GeoPT reads the AirCraft data from `$OMX_DEMO_DATA/geopt/aircraft_npys`.
Runs go to `$OMX_RUNS/demos/<name>`. `tests/test_demo_configs.py` checks the demo settings.

### Options

| Option | Effect |
|---|---|
| `--smoke` | at most 5 training steps per phase and 4 samples, written to `<out>_smoke`; checks a setup |
| `--only STEP ...` | runs only these steps; the outputs of every later step are cleared, so they run again next time |
| `--from PATH` | reuses a model trained earlier (a release, or a Taming run directory with `transformer.pt` and `policy.pt`) instead of training; writes to `<out>_imported` |
| `--device` | default: the first GPU, else CPU |

Each step writes `<out>/<step>.json` and is skipped when that file exists. A run stopped by a time limit continues
where it stopped. Once a step runs again, every later step runs again too. An interrupted imprint keeps its
progress and its draft key in the key store, so the resumed run trains toward the same key. `<out>/progress.jsonl`
has one line per step event. An `out` that holds a run with other settings is refused.

### The config

```yaml
model: {name: geopt, ckpt: $OMX_GEOPT_DIR/checkpoints/craft_geopt_8layers.pt, data: $OMX_DEMO_DATA/geopt/aircraft_npys}
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

| Key | Fields |
|---|---|
| `model` | `name`, the adapter's own parameters, or `source:` a plugin file |
| `scheme` | `name`, `params`, `source` |
| `data` | per adapter: `name`, `train`, `eval` (text, agent); `name`, `pairs`, `eval` (vlm); `classes`, `per_class`, `eval_classes`, `eval_per_class` (taming); none (geopt) |
| `imprint` | `phases`, `bs`, `accum`, `seq_len`, `answer_len`, `save_every`, `seed`, `edit_last`, `from` |
| `generate` | `n`, `new_tokens`, `bs` |
| `detect` | `fpr`, `null` (`base` or `wrong_keys`), `n_null` |
| `edits` | a list of edit specs (see [Benchmark](benchmark.md#edits)) |
| `quality` | `ppl`, `accuracy`, `fid`, `rel_l2` or `none` |
| `out`, `smoke` | the run directory; smoke mode |

The config is checked before anything loads. An unknown key, model, scheme, null, edit or quality name stops the
run with the allowed names. So does a setting the chosen model does not use (a model parameter, a `data` key, a
`generate` key, or an `imprint` key such as `edit_last` outside geopt) and a scheme parameter the scheme does not
take. `$VARS` in values come from the environment.

Phases give multi-stage recipes. The text, agent and vlm adapters train in one phase. Taming's demo co-learns the
green lists with the model, then fixes them and keeps training:

```yaml
imprint:
  phases:
    - {steps: 800, lr: 3.0e-5, trainable: full, policy_lr: 0.2}
    - {steps: 1000, lr: 1.0e-4, trainable: full, policy_lr: 0, fix_rule: true}
```

A release made with a stand-in (`factory`, `processor`) or a plugin adapter (`source`) names code to run. Opening
it outside its own run needs `--trust-plugin` (`trust=True`).

On a Slurm cluster, run the same command inside one GPU job. Long steps save their progress, so a job stopped
by its time limit continues when you submit it again.

## Language models, step by step

The commands below run each step of the text recipe on its own:

```bash
openmarkx embed --base llama-3.2-1b --watermark openmark            # prints {"release": ..., "key": ...}
openmarkx calibrate KEY --base llama-3.2-1b
openmarkx modify RELEASE --edit finetune:steps=200 --out edited/
openmarkx generate edited/ --prompts prompts.jsonl --out texts.jsonl
openmarkx rewrite texts.jsonl --edit synonym:ratio=0.3 --out reworded.jsonl
openmarkx detect texts.jsonl --key KEY
openmarkx verify edited/ --key KEY --prompts prompts.jsonl --sequential --budget 200
openmarkx --help                                                     # every command, with examples
```

A KEY is a key id from `openmarkx keys list`, a unique prefix of one, a key file, or a release directory (its key
is found by commitment). `RELEASE` and `KEY` above are the two fields `embed` prints. Text files are `.jsonl`
(`{"text": ...}` per line) or `.txt` (one text per line). To write held-out C4 prompts from the corpus that
`exp/prepare_data.py` made:

```bash
python -c "import json; from openmarkx.models import load_tokenizer; from openmarkx.sampling import eval_prompts; \
open('prompts.jsonl', 'w').writelines(json.dumps({'text': p}) + '\n' for p in eval_prompts(load_tokenizer('llama-3.2-1b'), n=100))"
```

| Command | Role | What it does | Main options |
|---|---|---|---|
| `embed` | owner | learns a watermark into a base model; prints the release and the key id | `--base` (required), `--watermark` (default `openmark`; `name:k=v` or a plugin file), `--out`, `--recipe`, `--steps`, `--device` |
| `port` | owner | wraps a checkpoint watermarked elsewhere as a release and a key | `--watermark`, `--key-file`, `--base` (all required), `--out`, `--public`, `--force` |
| `calibrate` | owner | freezes the key's null on the unmarked base model's texts | `--base`, `--n` (500), `--prompts`, `--device`, `--trust-plugin` |
| `modify` | downstream | applies a weight edit to a release | `--edit` (required), `--plugin`, `--out`, `--device` |
| `generate` | downstream | samples texts from a release | `--prompts` (required), `--out`, `--device` |
| `rewrite` | downstream | applies a text edit to texts | `--edit` (required), `--plugin`, `--out` |
| `detect` | detector | one p-value per text | `--key` (required), `--fpr` (0.01), `--grant`, `--null`, `--n-null` (200), `--base`, `--model`, `--release`, `--out`, `--trust-plugin` |
| `verify` | detector | one verdict for a suspect model | `--key`, `--prompts` (required), `--fpr`, `--alpha` (0.01), `--budget`, `--sequential`, `--p1` (0.2), `--trust-plugin` |
| `run` | everyone | a whole experiment from a config | `--only`, `--smoke`, `--from`, `--device` |
| `serve` | everyone | the web page for jobs on this machine's GPUs | `--port` (7860), `--host`, `--token`, `--origin` |
| `bench` (`evaluate`) | researcher | the benchmark for an arena file | `--dry-run`, `--queue HOST`, `--no-check` |
| `describe` | researcher | the scheme by primitive table | scheme names or plugin files |
| `check` | researcher | the checks a new method must pass | `--adapter` (required), `--name`, `--base` |
| `new-scheme` | researcher | writes a one-file scheme template | `--out`, `--force` |
| `keys` | owner | `list` the key store, or `show KEY` (key material hidden) | |

`embed` and `verify` take only text models. For the other adapters, use `openmarkx run` with a config.

`embed --recipe configs/embed.yaml` sets the training recipe of teacher-learned methods (`steps`, `bs`, `accum`,
`seq_len`, `lr`, `rank`, `save_every`, `seed`, `bos`); `--steps` overrides the recipe. The release goes to
`$OMX_RUNS/releases/<base>/<scheme>` unless `--out` says otherwise.

Weight edits and text edits are written `op:param=value,...`. [Benchmark](benchmark.md#edits) lists them with
their parameters.

## Python API

```python
import openmarkx as omx
from openmarkx.models import load_tokenizer
from openmarkx.sampling import eval_prompts

prompts = eval_prompts(load_tokenizer("llama-3.2-1b"), n=500)  # held-out C4 prompts (exp/prepare_data.py)
release, key = omx.embed("llama-3.2-1b", "gaussmark")        # owner: mark the weights; the secret key stays private
key = omx.calibrate(key, "llama-3.2-1b")                     # freeze the null on the base model's own texts

child = omx.modify(release, "finetune:steps=200")            # downstream: edit the weights
texts = omx.generate(child, prompts)                         # and write with the edited model
texts = omx.rewrite(texts, "synonym:ratio=0.3")              # and reword

hits = omx.detect(texts, key)                                # detector: [Detection(statistic, p_value, n_tokens, flagged)]
verdict = omx.verify_model(child, key, prompts, sequential=True, budget=200)   # one verdict for the model
```

[`examples/quickstart.py`](../examples/quickstart.py) runs this end to end on one GPU in a few minutes.

| Call | Returns |
|---|---|
| `embed(base, watermark="openmark", out=None, *, device=None, **recipe)` | `(Release, Key)` |
| `port(model_dir, watermark, key, base, out=None, *, key_public=False, force=False)` | `(Release, Key)` |
| `calibrate(key, base, prompts=None, *, n=500, device=None, seed=1, trust_plugin=False)` | the calibrated `Key` |
| `modify(release, edit, out=None, *, device=None)` | the edited `Release` |
| `generate(release, prompts, *, device=None, new_tokens=200, seed=0)` | a list of texts |
| `rewrite(texts, edit, *, device=None)` | a list of texts |
| `detect(texts, key, *, base=None, device=None, fpr=0.01, grant=None, trust_plugin=False, null="auto", n_null=200)` | a list of `Detection` |
| `verify_model(release, key, prompts, *, device=None, fpr=0.01, alpha=0.01, budget=None, sequential=False, p1=0.2, batch=20, trust_plugin=False)` | one `Detection` |
| `describe(names=None)` | the scheme by primitive table, as text |
| `load_key(release)` | the `Key` a release commits to, from the key store |
| `Release.open(path)`, `Key.open(path)` | a release or key read from disk |

`omx.register_scheme`, `register_edit`, `register_text_edit` and `register_model` add your own methods, edits and
models; `omx.load_plugins(path)` imports a plugin file. See [Methods](methods.md#adding-a-method),
[Benchmark](benchmark.md#adding-an-edit) and [Models](models.md).

## Keys

A release never contains its secret key. `release.json` holds only the key's commitment: sha256 over a nonce, the
scheme, its parameters and the plugin source. The key lives in the key store, `$OMX_KEYS`. Each key carries its own
fresh secret: MarkLLM's public default hash keys are replaced, and OpenMark draws its own token mapping. Files that
are part of a key, such as OpenMark's mapping or a trained policy, are stored by content under `blobs/` in the key
store.

```bash
openmarkx keys list
openmarkx keys show KEY        # prints metadata only; key material stays hidden
```

A calibrated key carries a frozen empirical null, so `detect` returns calibrated p-values. The calibration also
holds nulls for the texts cut to 20, 50 and 100 tokens. A text is judged against the null calibrated at the
largest length not above its own, because some detectors' false positives grow on short texts (check C9 in
[Methods](methods.md#the-checks-a-method-must-pass)). Keys calibrated before this rule warn until calibrated
again. A text too short to be judged gets `p_value=None` and `flagged=None`, and `verify_model` does not count it.
Schemes with an exact closed-form null (`exact_null = True`, such as GaussMark) use it directly.

`embed` writes every scheme parameter into the key, defaults included, so a key keeps detecting the same way after
a default changes in the code. Keys made before 2026-10-06 store only the parameters that were typed. Their
parameters are part of the commitment, so they cannot be filled in afterwards; such a key detects with the scheme's
current defaults.

## Detecting and verifying

### Texts

`openmarkx detect texts.jsonl --key KEY` prints one row per text: `statistic`, `p_value`, `n_tokens` and `flagged`
at `--fpr`. With a calibrated key, the p-value comes from the frozen null; otherwise from the scheme's closed form.

`--null wrong_key` (`null="wrong_key"`) needs no calibration. Each text is scored under `n_null` wrong keys drawn
from `random.Random(0xC0FFEE)`, and `p = (1 + #{wrong keys >= S}) / (n_null + 1)`. A wrong key is the true key with
only its secret redrawn (`Scheme.wrong_key`; OpenMark redraws its mapping and keeps its policy). It costs `n_null`
rebindings of the scheme. Teacher-free schemes are refused, since their closed-form null is already exact.

`--grant R1` (`grant=Access.R1`) caps what a detector may touch. A scheme that needs more is refused before any
model is loaded. [Methods](methods.md#what-the-detector-needs) lists what each method needs.

### Images and fields

`detect --model taming` takes a directory of PNG images, encodes them with the release's VQ and scores them against
the key's frozen null. `--model geopt` takes an `.npz` with `points`, `field` and `condition`, and scores it
against the release's unmarked model on the same inputs, with a wrong-key null:

```bash
openmarkx detect images/ --model taming --key ~/.openmarkx/runs/demos/taming/release     # --release defaults to --key
openmarkx detect case.npz --model geopt --key ~/.openmarkx/runs/demos/geopt/release
```

### A whole model

`verify` asks the suspect model for texts and tests how many are flagged. With `--sequential` it asks for 20 texts
at a time. It accuses the model as soon as an always-valid test (flag rate `p1` against `fpr`) is decisive, so a
strongly marked model is settled in about 20 texts. It never clears a model early, so a weakly marked one (after
edits) gets the full `--budget` and the power of the fixed test. False accusations stay below `alpha` despite the
early stop, and the verdict is flagged exactly when `p_value <= alpha`.

With an exact null (GaussMark) the flags of an unmarked model are Binomial(n, fpr). With an empirical null of N
texts they are Beta-binomial, because the suspect's texts are exchangeable with the calibration's. This accounts
for the threshold's own uncertainty (1.3% on average for 311 texts). It is exact for texts judged against a null of
their own length and conservative across the per-length nulls. Power found at a 5% flag rate with a budget of 200
(`exp/verify_power.py [--null-n 311]`):

| null | sequential | fixed |
|---|---|---|
| exact | 88% | 88% |
| empirical, 311 texts | 42% | 55% |

A larger calibration narrows the gap. Prompts disjoint from the calibration's are the clean choice. The bundled C4
split has only 311 usable prompts, so the quickstart reuses them; GaussMark's exact null is unaffected. In the
result, `n_queries` is the number of texts generated and `n_judged` the number long enough to judge.

### The Detection result

`Detection` has the fields of the `Detection(statistic, p_value, n_evidence)` result in MarkAgentX, a related
watermarking toolkit for agents, with the evidence counted in tokens: `d.n_evidence` and `d.to_markagentx()` give
them. An empty text answers `p_value = 1.0` with no evidence. `null="wrong_key"` scores each text under wrong keys
and takes its p-value from the rank of the true key's statistic among them.

## Bringing a checkpoint made elsewhere

`port` wraps a watermarked checkpoint that was trained outside OpenMarkX, together with the key material it was
made with, as a Release and a Key. Nothing is retrained and the weights stay where they are.

```python
release, key = omx.port("their_lora/", "openmark", {"mapping": "map.json", "transform": "policy.pth"}, "llama-3.2-1b")
release, key = omx.port("their_model/", "markllm-kgw-d", {"secret": 123456}, "llama-3.2-1b", key_public=True)
```

```bash
openmarkx port their_lora/ --watermark openmark --key-file key.json --base llama-3.2-1b
```

The checkpoint is a LoRA adapter directory or a full model directory. File paths in `--key-file` are relative to
that file. `key_public=True` (`--public`) marks a key that was shipped with the checkpoint, such as a paper's
released mapping. Anyone can detect with such a key, and anyone can also remove or forge the mark, so `detect`
warns. Calibrate the key before detecting with an empirical null.

## Running from a web page

`openmarkx serve` lets you upload a dataset, pick a method and its hyperparameters, and learn a watermark on this
machine's GPUs from a browser. It prints an address with a token; open it in a browser. Each job trains on your
texts, calibrates the key on the base model, then reports how many held-out texts from the watermarked model are
detected and how many human texts are falsely flagged. Data, releases and keys stay on the machine (`$OMX_SERVE`
and the key store).

The server listens on 127.0.0.1. For a GPU on another machine, run the server there and forward the port:

```bash
ssh -L 7860:127.0.0.1:7860 gpu-server      # then connect to http://127.0.0.1:7860
```

The page runs the text recipe. Running an `openmarkx run` config from it is on the [roadmap](../ROADMAP.md).

## Your own dataset

Every text step reads one tokenized corpus: training text for `embed`, clean text for the fine-tune edit, and
held-out documents whose first 50 tokens are the evaluation and calibration prompts. Build one from any text:

```bash
python exp/prepare_data.py --dataset camel-ai/physics --field message_2 --out $OMX_RUNS/data_physics.pt
python exp/prepare_data.py --dataset my_papers.jsonl --out $OMX_RUNS/data_mine.pt     # {"text": ...} per line, or .txt
export OMX_DATA=$OMX_RUNS/data_physics.pt     # every step now trains, attacks and evaluates on this text
```

The first `--n_eval` documents (600) are held out for prompts. Each needs at least 250 tokens for a prompt plus a
200-token continuation, so short answers give few prompts (the C4 file keeps 311 of the first 500). The fine-tune
edit reads chunks past 10,000, so keep `--n_train` above that. To change only one step, pass the file to it
instead: `embed --recipe` with `data: <file>`, or the edit `finetune:data=<file>`. Check a dataset's license before
training on it (`camel-ai/physics` is CC BY-NC 4.0).

## Another base model

`--base` takes a name from `BASES` in `openmarkx/paths.py`, a Hugging Face repo id, or a local directory. The names
in `BASES` are fetched at a pinned revision:

| Name | Hugging Face repo |
|---|---|
| `llama-3.2-1b` | `unsloth/Llama-3.2-1B` |
| `llama-3.2-1b-instruct` | `unsloth/Llama-3.2-1B-Instruct` |
| `llama-3.1-8b` | `meta-llama/Llama-3.1-8B` (gated: `huggingface-cli login`) |
| `llama-3.1-8b-instruct` | `unsloth/Meta-Llama-3.1-8B-Instruct` |
| `medgemma-1.5-4b-it` | `google/medgemma-1.5-4b-it` (gated; image and text input, for the `vlm` adapter) |

What a text base needs:
- **A causal language model** that `AutoModelForCausalLM` loads. Image and text models in the Gemma 3 style
  (MedGemma) use the `vlm` adapter. Any other kind of model needs a model adapter ([Models](models.md)).
- **LoRA target names** `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`, as in Llama, Mistral and
  Qwen2 (`LORA_TARGETS` in `openmarkx/imprint.py`; the fine-tune edit uses the same list).
- **A corpus in its tokenizer**: `prepare_data.py --base <the base>`. Llama-3.2-1B and Llama-3.1-8B share one.
- **A merge partner** for the merge edit: `<base>-instruct` in `BASES`, or `merge:partner=<repo id>`.

Add a line to `BASES` to pin the revision, so a run is comparable across machines:

```python
"qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B", "<commit sha>"),
```

## Reproducing the paper

The paper's results are on LLaMA-3-8B and are listed in [LEADERBOARD.md](../LEADERBOARD.md). The repository does
not ship the paper's trained weights, and its edit grid (LoRA fine-tune, linear merge, magnitude pruning,
bitsandbytes int8 and nf4, word-level rewrites) is not the paper's exact set of modifications (GPTQ, HQQ, Wanda,
SparseGPT, SLERP, fine-tuning on OpenMathInstruct and OpenCodeInstruct). The paper's tables are therefore not
reproduced by one command.

What runs is the same kind of benchmark on your own runs:

```bash
openmarkx bench configs/arena/1b.yaml       # every method on Llama-3.2-1B: learn, edits, forgery, leaderboard
```

[Benchmark](benchmark.md) explains the arena file, the steps and the board.
