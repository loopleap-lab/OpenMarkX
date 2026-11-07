# How OpenMarkX works

OpenMarkX learns a watermark into the weights of an open-source model, tests it under the edits a downstream user
may make, and detects it with the owner's secret key. A model plugs in through a model adapter, a watermark method
through a scheme, and an attack through an edit. Each is registered by name, so the built-ins and your own code go
through the same commands.

- [Roles](#roles)
- [Layers](#layers)
- [Release and key](#release-and-key)
- [The run pipeline](#the-run-pipeline)
- [Repository layout](#repository-layout)

## Roles

The calls follow the roles of the threat model:

| Role | Holds | Calls |
|---|---|---|
| owner | the base model and the secret key | `embed`, `port`, `calibrate`, `keys` |
| downstream user | the released weights only | `modify` (weight edits), `generate`, `rewrite` (text edits) |
| detector | the secret key, and what the method's access level grants | `detect`, `verify` |
| researcher | methods, edits and models to compare | `run`, `bench`, `check`, `describe`, `new-scheme` |

## Layers

| Layer | What it is | Built in | Your own |
|---|---|---|---|
| Model adapters | how to load a model, run it, which weights the imprint trains, and what one output unit is (`omx.Model`) | `text`, `agent`, `vlm`, `taming`, `geopt` | `@omx.register_model` ([Models](models.md)) |
| Schemes (methods) | how the secret key shifts each output unit, and how an output is scored (`omx.Scheme`, `omx.FieldScheme`) | `openmark` and its variants, 12 `markllm-*-d`, `gaussmark`, `wmar`, `ripple` | `@omx.register_scheme` ([Methods](methods.md)) |
| Imprint | learning the mark into the weights: distillation of the scheme's signal (LoRA or full fine-tune, in phases), or a direct weight edit | `imprint.embed` (text), `vlm.embed_vlm`, `train.imprint` (any adapter) | phase settings in the run config |
| Edits | what a downstream user does to the weights, the outputs or the texts | `finetune`, `merge`, `prune`, `quantize`; `synonym`, `word_deletion`, `context_synonym`; `image_roundtrip` | `@omx.register_edit`, `@omx.register_text_edit` ([Benchmark](benchmark.md#adding-an-edit)) |
| Detection and verification | a p-value per output against a calibrated or wrong-key null; one verdict per model | `detect`, `verify_model` | |
| Benchmark | every method under every edit, plus forgery, on a leaderboard of your own runs | `openmarkx bench` and the `exp/` scripts | an arena file ([Benchmark](benchmark.md)) |

A scheme is assembled from six primitives (`openmarkx/primitives.py`):

| Primitive | Role | In a scheme |
|---|---|---|
| Key | keyed pseudo-randomness: the secret and the material derived from it | `setup` |
| Signal | the training target: base logits to watermarked logits, teacher-forced | `target_logits` (or `target_output` for fields) |
| Imprint | how the mark gets into the weights: `DISTILL` (learn the Signal) or `EDIT` (write the weights directly) | `imprint` |
| Perturb | weight edits, used as attacks and inside training (OpenMark's forgetting-perturbation step) | edits; `booster` |
| Probe | what the detector may touch: the access level | `access` |
| Score | output to statistic, then a p-value against a null | `score`, `pvalue` |

`openmarkx describe` prints which primitives each method uses. A teacher-free method such as GaussMark uses Key,
Imprint `EDIT`, Probe and Score, and no Signal.

## Release and key

`embed` (or the `imprint` step of a run) returns two things:

- **The release**: a directory with `release.json` and the weights (a LoRA adapter, merged weights, or a
  `state_dict` the adapter loads). `release.json` names the base, the scheme, the model adapter and its
  parameters, the lineage of edits, and the commitment to the key. This is what the owner publishes.
- **The secret key**: the scheme, its parameters, the key material and the calibration, filed in the key store
  under its commitment. It never enters the release directory.

An edited release (`modify`) keeps the commitment and extends the lineage, so a KEY given as the edited release
directory still names the original's secret key. [Keys](user-guide.md#keys) in the user guide covers the store and calibration.

## The run pipeline

`openmarkx run CONFIG` (`openmarkx/pipeline.py`) runs one experiment for any adapter, in ten steps:

```
prepare -> selfgen -> calibrate -> imprint -> generate -> detect -> edits -> quality -> report -> viewer
```

| Input | Contents |
|---|---|
| model | an adapter name and its parameters (`geopt` with a checkpoint and data), or a plugin file |
| scheme | a method name and its parameters, or a plugin file |
| imprint | one or more training phases (steps, learning rate, LoRA or full, policy learning rate, fixed rule) |
| detect, edits, quality | the false-positive rate and the null; the edits to apply; the quality metric |

| Output | Contents |
|---|---|
| `<out>/release/` | the watermarked weights and `release.json` |
| the key store | the secret key, with its calibration |
| `<out>/report.json` | model, scheme, detection (TPR at the FPR, threshold, AUC), quality before and after, TPR after each edit, settings |
| `<out>/<step>.json` | each step's result; a step whose file exists is skipped, so a stopped run resumes |
| `<out>/assets/` | generated images, and `viewer.json` for the run viewer |

Each step routes by adapter: `imprint` trains text and agent models through `imprint.embed`, `vlm` through
`embed_vlm`, and `taming` and `geopt` through `train.imprint`. Token schemes train by KL divergence to
`target_logits`; field schemes train by mean squared error to `target_output`. The teacher is the unmarked model:
the LoRA adapter switched off, or a frozen copy for a full fine-tune. [Running an experiment from a
config](user-guide.md#running-an-experiment-from-a-config) lists the config keys and options.

## Repository layout

| Path | Contents |
|---|---|
| `openmarkx/` | `api.py` (the public calls), `cli.py`, `pipeline.py` (`openmarkx run`), `train.py` (phased imprint), `imprint.py` (text imprint), `vlm.py`, `bench.py`, `check.py`, `keys.py`, `registry.py`, `primitives.py`, `types.py`, `scoring.py`, `sampling.py`, `quality.py`, `external.py`, `paths.py`, `serve.py` and `app.html` (the web page) |
| `openmarkx/models/` | `base.py` (`Model`, `Output`), `text.py`, `agent.py`, `vlm.py`, `taming.py`, `geopt.py` |
| `openmarkx/schemes/` | `base.py` (`Scheme`, `FieldScheme`), `openmark.py`, `markllm.py`, `gaussmark.py`, `wmar.py`, `ripple.py` |
| `openmarkx/edits/` | `weights.py` (finetune, merge, prune, quantize), `text.py` (synonym, word_deletion, context_synonym), `outputs.py` (image_roundtrip) |
| `openmarkx/third_party/pro/` | the OpenMark watermark code and its initial policy weights, shipped with the package |
| `markweightx/` | the old package name, an alias of `openmarkx` |
| `configs/` | `embed.yaml` (text training recipe), `watermark/<scheme>.yaml` (written by `exp/write_configs.py`), `arena/*.yaml`, `demos/*.yaml` |
| `examples/` | `quickstart.py`, `my_scheme.py` (the `new-scheme` template), `my_edits.py` (a weight edit and a text edit) |
| `exp/` | experiment scripts: data, learnability, robustness, forgery, leaderboard, scheduler; the demo experiments (`agent/`, `med/`, `geopt/`, `arimg/`) |
| `scripts/` | `install_deps.sh` (MarkLLM at its pinned commit) |
| `paper/` | the paper's own training and evaluation scripts for OpenMark, with the paper's modifications ([README](../paper/README.md)) |
| `notebooks/` | `run_on_colab.ipynb` |
| `tests/` | `pytest -q tests` (add `-m "not slow"` to skip the end-to-end check runs); `fakes.py` (CPU stand-ins); `broken/` (adapters that `check` must fail) |
| `docs/` | this documentation and the website |
