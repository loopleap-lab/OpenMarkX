# Changelog

## Unreleased

- Documentation in the layout of `docs/`: the user guide (`docs/user-guide.md`, which replaces
  `docs/REFERENCE.md`), how OpenMarkX works (`docs/architecture.md`), methods (`docs/methods.md`), your own model
  (`docs/models.md`), edits and the benchmark (`docs/benchmark.md`), and the FAQ (`docs/FAQ.md`). `ROADMAP.md`
  replaces the internal list of website calls that do not run yet. This changelog.

## 0.1.0

The toolkit released with the NeurIPS 2026 paper "OpenMark: Enabling Precise and Robust Text Watermark for
Open-Source LLMs".

### Rename (2026-10-24)

- MarkWeightX is now OpenMarkX: the package `markweightx` is `openmarkx`, and the command `markweightx` is
  `openmarkx`. The old package and command stay as aliases, so code, pickles and run directories written before the
  rename keep working.
- The method PRO is now OpenMark: the scheme `pro` is `openmark`, and `pro-coopt-c100`, `pro-full-b01` and the other
  variants are `openmark-*`. The old scheme names stay as aliases in keys, releases, configs and command lines.
- `OMX_*` environment variables, with `MWX_*` read as a fallback; `~/.markweightx` is used when `~/.openmarkx` does
  not exist.
- The website, README, leaderboard and notebook use the new names; the paper is cited as OpenMark.

### Model adapters and `openmarkx run`

- The model adapter interface (`omx.Model`, `@omx.register_model`) and adapters for text, a ReAct agent, a
  vision-language model (MedGemma), Taming Transformers' image tokens (through WMAR) and GeoPT physics fields.
- Release schema 2: a release names its model adapter and its weight format.
- Field schemes (`omx.FieldScheme`) and a generic imprint loop in phases (`train.imprint`): LoRA or full
  fine-tuning, a policy learned with the model, a fixed rule.
- The `wmar` scheme for image tokens, the `image_roundtrip` edit and FID; the `ripple` scheme for fields, with an
  optional closed-form edit of the last layer, and a ripple rule co-learned with the model.
- `openmarkx run CONFIG`: one pipeline (prepare, selfgen, calibrate, imprint, generate, detect, edits, quality,
  report, viewer) for every adapter. Configs are checked before anything loads. Long steps save their progress and
  resume; `--smoke` writes to `<out>_smoke`; a run directory with other settings is refused.
- `imprint.from` and `run --from` reuse a trained release instead of training.
- `detect --model taming` and `--model geopt` score images and fields.
- Demo configs in `configs/demos/` (Taming, MedGemma, GeoPT, GeoPT co-learned, agent) and a HiPerGator run script.

### Methods

- OpenMark, with the paper's code shipped in `openmarkx/third_party/pro`: `openmark` (frozen policy),
  `openmark-coopt-c100` (co-optimized policy), `openmark-full-b01` (with forgetting-perturbation training), and
  experimental variants.
- 12 decoding-time watermarks from MarkLLM (pinned at commit `112fa74`), distilled into the weights, with fixes in
  the adapter and every teacher at temperature 1.
- GaussMark: a keyed Gaussian perturbation written into one weight matrix, detected with gradients of the base.
- Six primitives (Key, Signal, Imprint, Perturb, Probe, Score), access levels R0 to R2, and `describe`.
- One-file plugin methods: `openmarkx new-scheme` and `openmarkx check` (C1 to C10, with 17 broken adapters it must
  fail); plugins are pinned by content and run only when trusted.

### Keys, detection and verification

- The public calls `embed`, `port`, `calibrate`, `modify`, `generate`, `rewrite`, `detect`, `verify_model`.
- A key store apart from releases: a release holds only a salted commitment; key files are stored by content.
- Calibrated nulls, including nulls for texts cut to 20, 50 and 100 tokens; the wrong-key null of MarkAgentX and
  its `Detection` fields.
- `verify_model`: one verdict per model, with an optional sequential test and a query budget; exact
  Beta-binomial null for empirical calibrations.
- `port`: wrap a checkpoint watermarked elsewhere, with its key files, as a release and a key.

### Edits and the benchmark

- Weight edits (`finetune`, `merge`, `prune`, `quantize`) and text edits (`synonym`, `word_deletion`,
  `context_synonym`), with parameters checked before any model loads.
- Plugin edits with `@omx.register_edit` and `@omx.register_text_edit`; any dataset through
  `exp/prepare_data.py`; any Llama-style base.
- `openmarkx bench` (also `evaluate`): learn, edits and forgery for an arena file, then the leaderboard, with
  experimental variants under Ablations, a C9 table and LaTeX output; a GPU scheduler for queued steps.
- `LEADERBOARD.md` with the paper's results on LLaMA-3-8B.

### Running it

- `openmarkx serve` and its web page: upload a dataset, set hyperparameters and learn a watermark on this machine's
  GPUs; a Colab notebook that runs it on a free T4.
- The project website in `docs/`.
