# Changelog

## Unreleased

### `openmark` is the paper's method

- `openmark` (and `openmarkx embed`'s default `--watermark`) now means the paper's full method: the watermark
  policy co-optimized with the model (CAWP) plus forgetting-perturbation training (FPL). This is the scheme that was
  called `openmark-full-b01`, which stays as an alias. Each training step runs two more forward and backward passes
  than before, about three times the compute.
- The frozen-policy variant that `openmark` named before is now `openmark-frozen`. `pro` is an alias of
  `openmark-frozen`, and `pro-full-b01` of `openmark`.
- Keys, `result.json` files and run configs recorded as `openmark` keep their meaning: a record without
  `co_opt: true` is read as `openmark-frozen` (`openmarkx.registry.stored_scheme`), so old keys detect as before.
  `keys list` and `keys show` print the name a key resolves to. A run's `config.json` now records the naming
  (`"names": 2`). `openmarkx bench` keeps its result directories: `openmark/` for `openmark-frozen`,
  `openmark-full-b01/` for `openmark`.
- `port` with an OpenMark key that has no trained policy (`transform`) records the release as `openmark-frozen`.
- `describe` shows a co-optimized policy in the Signal column.
- The 1B arena, the MedGemma and agent demo configs, the plugin-edits arena and the Colab run page use
  `openmark-frozen` where they ran the frozen variant; the 1B arena lists `openmark` for the paper's method.

### Other changes

- `omx.register_output_edit` is exported.
- `detect --model` takes `text`, `agent`, `vlm` (all scored as text), `taming` or `geopt`; the `--model` help of
  `embed` and `verify` says they are text-only.
- `exp/geopt/run_mark.py` and `demo_check.py` read the GeoPT checkout from `OMX_GEOPT_DIR`.
- Removed the machine-specific shell scripts `exp/run_e2.sh`, `exp/run_queue.sh` and `exp/robustness_watch.sh`;
  `openmarkx bench` runs the same steps.
- Documentation: a user guide, how OpenMarkX works, methods, your own model, the benchmark and an FAQ in `docs/`;
  a roadmap; this changelog.

## 0.1.0 (current main, not yet tagged)

The toolkit released with the NeurIPS 2026 paper "OpenMark: Enabling Precise and Robust Text Watermark for
Open-Source LLMs".

### Rename (2026-10-24)

- MarkWeightX is now OpenMarkX: the package `markweightx` is `openmarkx`, and the command `markweightx` is
  `openmarkx`. The old package and command stay as aliases, so code, pickles and run directories written before the
  rename keep working.
- The method PRO is now OpenMark: the scheme `pro` is `openmark` (renamed `openmark-frozen` since; see Unreleased), and `pro-coopt-c100`, `pro-full-b01` and the other
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
- Demo configs in `configs/demos/` (Taming, MedGemma, GeoPT, GeoPT co-learned, agent).

### Methods

- OpenMark, with the paper's code shipped in `openmarkx/third_party/pro`: `openmark` (frozen policy; now
  `openmark-frozen`), `openmark-coopt-c100` (co-optimized policy), `openmark-full-b01` (with
  forgetting-perturbation training; now `openmark`), and experimental variants.
- 12 decoding-time watermarks from MarkLLM (pinned at commit `112fa74`), distilled into the weights, with fixes in
  the adapter and every teacher at temperature 1.
- GaussMark: a keyed Gaussian perturbation written into one weight matrix, detected with gradients of the base.
- Six primitives (Key, Signal, Imprint, Perturb, Probe, Score), access levels R0 to R2, and `describe`.
- One-file plugin methods: `openmarkx new-scheme` and `openmarkx check` (C1 to C10, with 17 broken adapters it must
  fail); plugins are pinned by content and run only when trusted.

### Keys, detection and verification

- The public calls `embed`, `port`, `calibrate`, `modify`, `generate`, `rewrite`, `detect`, `verify_model`.
- A key store apart from releases: a release holds only a salted commitment; key files are stored by content.
- Calibrated nulls, including nulls for texts cut to 20, 50 and 100 tokens; the wrong-key null of MarkAgentX (a related
  agent-watermark toolkit) and its `Detection` fields.
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
