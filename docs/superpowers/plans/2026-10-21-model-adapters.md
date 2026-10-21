# Model Adapters and One Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MedGemma, the ReAct agent, Taming (WMAR) and GeoPT first-class, terminal-runnable parts of
`markweightx`, behind one model adapter interface and one `markweightx run <config.yaml>` pipeline.

**Architecture:** A `Model` adapter (load, run, unit, output family, codec) joins `Scheme` in the registry. Token
families keep today's KL path (`target_logits`); a field family adds `target_output` with MSE and reference-based
detection (`Access.R1`). A resumable pipeline runs prepare → selfgen → calibrate → imprint → generate → detect →
edits → quality → report from a YAML config; four demo configs reproduce last night's settings.

**Tech Stack:** Python ≥3.10, PyTorch, transformers, peft, PyYAML, pytest. WMAR and GeoPT stay external
(`MWX_WMAR_DIR`, `MWX_GEOPT_DIR`).

**Spec:** `docs/superpowers/specs/2026-10-21-model-adapters-design.md`

## Global Constraints

- Existing behaviour unchanged: existing schemes, keys and releases load and score as before; the existing CLI
  subcommands keep their flags (they gain `--model`).
- Acceptance: (1) each adapter's tiny stand-in runs `markweightx run` through all steps on CPU with a well-formed
  `report.json`; (2) a config test checks the four demo configs against last night's run arguments, field by field;
  (3) the existing tests still pass; (4) a short smoke run per demo on HPG with the real model and code.
- WMAR and GeoPT code is imported from `MWX_WMAR_DIR` / `MWX_GEOPT_DIR` and never modified; a missing variable
  raises an error naming what to install.
- `Release` gains `model: {name, params}`, `format: "hf" | "state_dict"`, `schema: 2`; a release without them
  reads as `text`, `hf`, schema 1.
- Keys never go into a release directory (unchanged rule; the Taming policy table is key material).
- Interfaces designed but not implemented stay marked "planned" in docs and on the site.
- Every pipeline step is resumable: it skips when its output file exists.
- Local runs use CPU only (lab2 GPUs are off limits). HPG: login node runs no Python; `hpg-precheck` before every
  sbatch; outputs under `/blue/sgao1/ji757406.ucf/`; each job 1 GPU, ≤ 8 h; polling ≥ 60 s.
- Demo settings (copied from the run files): Taming P2 `--lr 3e-5 --policy_lr 2e-1 --steps 800`, then P14
  `--lr 1e-4 --policy_lr 0 --steps 1000` from P2; GeoPT `s1_e02 --eps 0.02 --output_scale 1.0`; Agent
  `pro:delta=2 --seq_len 256 --bs 16`, answers `--final_words 300 --final_tokens 600 --temperature 0.7`;
  MedGemma `pro --bs 16 --answer_len 256`.

## Review Focus

- A user points `--model taming` without `MWX_WMAR_DIR` set: expect one clear error naming WMAR and the variable,
  not an ImportError traceback. Test in Task 6.
- An old text `release.json` (no `model`/`format`/`schema`) passed to the new `detect`/`verify`: expect it to load
  and score exactly as before. Test in Task 2.
- A pipeline run killed mid-step and restarted: expect completed steps skipped and the interrupted step redone,
  with no half-written output treated as done. Test in Task 8 (write to `*.tmp`, rename on success).
- A config with an unknown key or misspelt step name: expect a validation error listing the allowed names before
  any model loads. Test in Task 8.
- `detect` on a field `.npz` whose point count differs from the reference run's geometry: expect an error saying
  the shapes differ, not a silent broadcast. Test in Task 7.

---

## File structure

| Path | Responsibility |
|---|---|
| `markweightx/models/__init__.py` | today's `models.py` content (text helpers), re-exported unchanged |
| `markweightx/models/base.py` | `Model` base class, `Output` enum |
| `markweightx/models/text.py` | `TextModel` adapter over the existing helpers |
| `markweightx/models/vlm.py` | `VLMModel` adapter over `markweightx/vlm.py` |
| `markweightx/models/agent.py` | `ReActAgent` adapter (text model + ReAct loop, final answer as output) |
| `markweightx/models/taming.py` | `TamingModel` adapter (imports WMAR from `MWX_WMAR_DIR`) |
| `markweightx/models/geopt.py` | `GeoPTModel` adapter (imports GeoPT from `MWX_GEOPT_DIR`) |
| `markweightx/external.py` | `require_external(name, env)` → imported module or clear error |
| `markweightx/schemes/base.py` | add `output` attribute and `FieldScheme` |
| `markweightx/schemes/wmar.py` | `WMARScheme`: codebook green list, co-learning, fixed-rule phase, table detector |
| `markweightx/schemes/ripple.py` | `RippleScheme`: keyed field pattern, reference residual score, wrong-key null |
| `markweightx/train.py` | generic imprint loop for non-text adapters: KL or MSE, LoRA or full, phases, resume |
| `markweightx/quality.py` | quality metrics registry: `accuracy`, `fid`, `rel_l2`, `ppl` |
| `markweightx/pipeline.py` | config schema, steps, resume, `report.json` |
| `markweightx/registry.py` | add `register_model`, `get_model`, plugin loading for model files |
| `markweightx/types.py` | `Release` schema 2 fields |
| `markweightx/cli.py` | `run` subcommand; `--model` on embed/detect/verify |
| `configs/demos/{medgemma,agent,taming,geopt}.yaml` | the four demos |
| `scripts/hpg/run.sbatch` | one generic resumable job: `markweightx run $CONFIG` |
| `tests/fakes.py` | tiny stand-ins: fake WMAR (VQ + GPT), fake GeoPT (MLP on points), tiny Llama, tiny Gemma 3 |
| `tests/test_models.py`, `test_release_v2.py`, `test_schemes_ext.py`, `test_pipeline.py`, `test_demo_configs.py` | new tests |
| `docs/REFERENCE.md` | updated reference |

`exp/` keeps prep scripts (MedXpertQA, HotpotQA), sweeps and figure scripts; the training/eval parts of
`exp/arimg/pro_taming.py` and `exp/geopt/run_mark.py` become thin wrappers that call the pipeline (Task 9).

---

### Task 0: CPU development environment and baseline

**Files:** none in the repo (a local `.venv`, git-ignored).

- [ ] **Step 1:** Create a CPU env: `uv venv .venv && uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu && uv pip install --python .venv/bin/python -e ".[test]"`.
- [ ] **Step 2:** Run `.venv/bin/pytest -q -m "not slow"`. Expected: all existing tests pass (77, minus any marked slow). Record the count in the task log; any failure here is pre-existing and is reported to the user before continuing.
- [ ] **Step 3:** Add `.venv/` to `.gitignore` if missing; commit.

### Task 1: The `Model` adapter and its registry; `models` becomes a package

**Files:**
- Create: `markweightx/models/__init__.py` (move `markweightx/models.py` here verbatim), `markweightx/models/base.py`, `markweightx/models/text.py`
- Modify: `markweightx/registry.py`, `markweightx/__init__.py` (export `Model`, `register_model`)
- Test: `tests/test_models.py`

**Interfaces:**
- Produces:
  - `class Output(str, Enum): TOKENS = "tokens"; FIELD = "field"` in `models/base.py`
  - `class Model`: `name: str`, `unit: str`, `output: Output`; `__init__(self, **params)` stores `self.params`;
    `load(self, path: str | None, device: str) -> torch.nn.Module`; `run(self, net, inputs: list, **kw) -> list`;
    `trainable(self, net, mode: str) -> list[torch.nn.Parameter]` with `mode in {"lora", "full"}`;
    `codec(self) -> Codec` where `Codec` has `encode(outputs) -> units` and `decode(units) -> outputs`.
  - `register_model(cls=None, *, name=None)`, `get_model(name: str, *, source: str | None = None, **params) -> Model`
    in `registry.py`, using the same plugin path and trust rules as `get_scheme`.
  - `TextModel(name="text", unit="token", output=TOKENS)`: `load` → `load_base`/`load_release`; `run` → greedy or
    sampled generation (`new_tokens`, `temperature`, `seed` kwargs); codec = the tokenizer.

- [ ] **Step 1: Write the failing tests** in `tests/test_models.py`:
  `test_text_helpers_still_import` (`from markweightx.models import load_tokenizer, load_base, load_release, text_vocab_size, ConfigOnly`),
  `test_registry_lists_builtin_models` (`set(list_models()) >= {"text"}`),
  `test_model_plugin_file_loads_with_trust` (a temp file registering `class Toy(Model)`; `get_model("toy", source=path)` raises `UntrustedPlugin` without trust and loads with it),
  `test_text_model_runs_tiny_llama` (tiny random Llama from `tests/fakes.py`: `run` returns `len(inputs)` strings).
- [ ] **Step 2:** Run `pytest tests/test_models.py -v`. Expected: FAIL (no `markweightx.models.base`).
- [ ] **Step 3:** Move `models.py` to `models/__init__.py`; implement `base.py`, `text.py`, registry functions; create `tests/fakes.py` with `tiny_llama()`.
- [ ] **Step 4:** Run `pytest tests/test_models.py -v` and `pytest -q -m "not slow"`. Expected: new tests PASS; existing count unchanged.
- [ ] **Step 5:** Commit `feat: Model adapter interface and registry; text adapter`.

### Task 2: `Release` schema 2

**Files:**
- Modify: `markweightx/types.py` (`Release`), `markweightx/models/__init__.py` (`load_release` dispatches on `rel.model`)
- Test: `tests/test_release_v2.py`

**Interfaces:**
- Consumes: `get_model` (Task 1).
- Produces: `Release` fields `model: dict = field(default_factory=lambda: {"name": "text", "params": {}})`,
  `format: str = "hf"`, `schema: int = 1` (new releases written with `schema=2`); `Release.adapter_model() -> Model`.

- [ ] **Step 1: Write the failing tests:** `test_legacy_release_reads_as_text_hf_schema1` (a `release.json` copied from today's format has `rel.model["name"] == "text"`, `rel.format == "hf"`, `rel.schema == 1`); `test_new_release_roundtrip_keeps_model_and_format` (save/open with `model={"name": "geopt", "params": {"output_scale": 1.0}}`, `format="state_dict"`, `schema=2`); `test_release_never_contains_key_files` (existing invariant, extended to `format="state_dict"`).
- [ ] **Step 2:** Run; expected FAIL on missing fields.
- [ ] **Step 3:** Add the fields with defaults; `Release.open` fills missing fields; `save` writes `schema=2` only when `model["name"] != "text"` or `format != "hf"` (so text releases stay byte-identical).
- [ ] **Step 4:** Run the new tests and the whole suite. Expected: PASS; `tests/test_api.py::test_legacy_release_key_still_loads` still PASS.
- [ ] **Step 5:** Commit `feat: release schema 2 (model, format)`.

### Task 3: Field-family schemes and the generic imprint loop

**Files:**
- Modify: `markweightx/schemes/base.py`
- Create: `markweightx/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Produces:
  - `Scheme.output: Output = Output.TOKENS` (class attribute).
  - `class FieldScheme(Scheme)`: `output = Output.FIELD`, `access = Access.R1`; abstract
    `target_output(self, inputs, output: torch.Tensor) -> torch.Tensor` and
    `score(self, output: torch.Tensor, reference: torch.Tensor, inputs) -> tuple[float, int]`.
  - `train.imprint(model: Model, scheme: Scheme, batches: Iterable, out: str, *, phases: list[dict], device: str,
    save_every: int = 500) -> Release` where each phase is `{"steps": int, "lr": float, "trainable": "lora"|"full",
    "rank": int, "policy_lr": float, "fix_rule": bool}`; loss = `kl_teacher_student` (imported from `imprint.py`)
    for `TOKENS`, `F.mse_loss(student, scheme.target_output(...))` for `FIELD`; adds `scheme.aux_loss()`; when
    `fix_rule` is true, `scheme.trainable_params()` get `requires_grad_(False)`; resume state in `out/resume.pt`.

- [ ] **Step 1: Write the failing tests:** `test_kl_phase_moves_student_toward_teacher` (tiny Llama, a toy green-list scheme, 20 steps: KL after < KL before); `test_mse_phase_learns_field_target` (fake GeoPT MLP, toy `FieldScheme` adding `0.1 * ones`, 50 steps: MSE after < 0.5 × before); `test_fix_rule_phase_freezes_scheme_params` (two phases; scheme param unchanged across phase 2); `test_resume_continues_from_saved_step` (stop after phase 1, rerun: log shows phase 2 only).
- [ ] **Step 2:** Run; expected FAIL (no `markweightx.train`).
- [ ] **Step 3:** Implement `FieldScheme` and `train.imprint`; keep cosine schedule with 50 warmup steps and grad clip 1.0 as in `imprint.py`.
- [ ] **Step 4:** Run tests; expected PASS. Run whole suite; expected unchanged.
- [ ] **Step 5:** Commit `feat: field schemes and the generic imprint loop`.

### Task 4: VLM and agent adapters

**Files:**
- Create: `markweightx/models/vlm.py`, `markweightx/models/agent.py`
- Modify: `markweightx/vlm.py` (unchanged functions; `models/vlm.py` wraps them)
- Test: `tests/test_models_text_family.py`

**Interfaces:**
- Consumes: `Model`, `TextModel` (Task 1); `markweightx.vlm.{load_vlm, load_vlm_release, generate_vlm, embed_vlm}`.
- Produces:
  - `VLMModel(name="vlm", unit="token", output=TOKENS)`: inputs are `{"image": path|PIL, "question": str}`; `run` →
    `generate_vlm`; imprint delegates to `embed_vlm` (same objective as today).
  - `ReActAgent(TextModel)` with `name="agent"`, params `max_steps=6`, `final_words=300`, `final_tokens=600`,
    `temperature=0.7`; inputs are `{"question": str, "paragraphs": {title: text}}`; `run` returns the final answer
    strings; `run(..., transcript=True)` also returns the transcripts. The ReAct loop, `parse_action`, `search` and
    `norm` move here from `exp/agent/hotpot_agent.py`, unchanged in behaviour.

- [ ] **Step 1: Write the failing tests:** `test_vlm_model_runs_tiny_gemma` (tiny Gemma 3 from `tests/fakes.py`, one image + question → one string); `test_agent_parses_actions_and_stops_on_finish` (scripted fake net returning `Action: Search[A]` then `Action: Finish[x]`: transcript has one Observation from paragraph A, final answer returned); `test_agent_matches_old_parse_action` (same outputs as `exp/agent/hotpot_agent.parse_action` on five fixed strings).
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement both adapters; `exp/agent/hotpot_agent.py` imports the moved functions from `markweightx.models.agent`.
- [ ] **Step 4:** Run tests; expected PASS.
- [ ] **Step 5:** Commit `feat: VLM and ReAct agent adapters`.

### Task 5: WMAR scheme (Taming's green list over the codebook)

**Files:**
- Create: `markweightx/schemes/wmar.py`
- Test: `tests/test_schemes_ext.py`

**Interfaces:**
- Produces: `@register_scheme class WMARScheme(Scheme)`: `name="wmar"`, params `delta=2.0`, `gamma=0.25`,
  `balance=10.0`, `null_w=0.0`, `colearn=True`; `setup(model, codec, device, key)` builds the alive-code table from
  WMAR's linear seeding (`green_table` logic from `exp/arimg/pro_taming.py`) and the policy scores `Z`;
  `target_logits(prev_codes, logits)` = `logits + delta * green[prev]` using the hard top-gamma table with a
  straight-through gradient; `trainable_params()` → `[Z]` when `colearn`; `aux_loss()` → balance (+ null) terms as
  in `cmd_colearn`; `score(codes) -> (z, n_pairs)` = `detect_table` statistic over unique (prev, cur) pairs;
  `save(key_dir, key)` writes `policy.pt` into the key store.

- [ ] **Step 1: Write the failing tests:** `test_wmar_table_rows_have_gamma_share_green` (fake codebook of 64 alive codes: every row has `round(0.25 * 64)` greens); `test_wmar_score_matches_old_detect_table` (same z as `exp/arimg/pro_taming.detect_table` on fixed random codes and table); `test_wmar_policy_saved_to_key_store_not_release` (after `save`, `policy.pt` is under `MWX_KEYS`, not under the release dir); `test_wmar_colearn_balance_keeps_columns_near_gamma` (100 steps on random data: column green share in [0.1, 0.4]).
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement, porting `policy_table`, `detect_table` and the colearn loss terms from `exp/arimg/pro_taming.py`.
- [ ] **Step 4:** Run; expected PASS.
- [ ] **Step 5:** Commit `feat: wmar scheme`.

### Task 6: Taming adapter and the image edits and quality metric

**Files:**
- Create: `markweightx/models/taming.py`, `markweightx/external.py`, `markweightx/quality.py` (with `fid`)
- Modify: `markweightx/edits/weights.py` (register output edit `image_roundtrip`)
- Test: `tests/test_taming.py`

**Interfaces:**
- Consumes: `WMARScheme` (Task 5), `train.imprint` (Task 3).
- Produces:
  - `external.require_external(name: str, env: str, import_name: str) -> module`; raises
    `MissingExternal(f"{name} is not installed: clone it and set {env} to its directory")`.
  - `TamingModel(name="taming", unit="image token", output=TOKENS)`: params `modelpath`, `enc_ft`, `dec_ft`;
    inputs are class labels; `run` → code sequences `[N, 256]` sampled with WMAR's settings (top_k 250, top_p 0.92,
    temperature 1.0); codec: `encode(images) -> codes` (VQ encoder), `decode(codes) -> images`; `trainable(net, "full")`.
    A `factory` param lets tests inject `tests.fakes.FakeTaming`.
  - `quality.fid(images_a, images_b) -> float` (torch-fidelity when installed; else an error naming it).
  - Output edit `image_roundtrip`: decode then re-encode (what a detector sees from an image file).

- [ ] **Step 1: Write the failing tests:** `test_missing_wmar_dir_names_the_variable` (unset env: `get_model("taming").load(...)` raises `MissingExternal` whose message contains `MWX_WMAR_DIR`); `test_taming_fake_samples_codes_of_shape_256` (fake: `run([1, 2])` → tensor `[2, 256]` within the alive codes); `test_roundtrip_returns_codes_of_same_shape`.
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement; `FakeTaming` in `tests/fakes.py` = a 64-code VQ (random linear encoder/decoder on 16×16×3 tiles) and a 2-layer GPT.
- [ ] **Step 4:** Run; expected PASS.
- [ ] **Step 5:** Commit `feat: taming adapter, image round trip, fid`.

### Task 7: Ripple scheme and GeoPT adapter

**Files:**
- Create: `markweightx/schemes/ripple.py`, `markweightx/models/geopt.py`
- Modify: `markweightx/quality.py` (add `rel_l2`)
- Test: `tests/test_geopt.py`

**Interfaces:**
- Consumes: `FieldScheme`, `train.imprint` (Task 3); `require_external` (Task 6).
- Produces:
  - `@register_scheme class RippleScheme(FieldScheme)`: `name="ripple"`, params `eps=0.02`, `output_scale=1.0`,
    `m=32`, `n_wrong=200`; `target_output(inputs, ref) = ref + eps * output_pattern(ref, key)`;
    `score(output, reference, inputs) -> (z, n_points)` = `fieldmark.zscore` (wrong-key null per case);
    `calibrate_strength(model, samples, z_target) -> eps` (smallest eps in a grid reaching `z_target` on every
    sample). `pattern`, `smooth_basis`, `highpass`, `output_pattern`, `statistic`, `zscore` move from
    `exp/geopt/fieldmark.py`, unchanged.
  - `GeoPTModel(name="geopt", unit="surface point", output=FIELD)`: params `ckpt`, `data`; imports GeoPT from
    `MWX_GEOPT_DIR`; inputs are `(points [N, 3], condition [3])`; `run` → fields `[N, C]`; `trainable(net, "full")`;
    a `factory` param lets tests inject `tests.fakes.FakeGeoPT`.
  - `quality.rel_l2(pred, truth) -> float`.

- [ ] **Step 1: Write the failing tests:** `test_ripple_functions_match_old_fieldmark` (same z as `exp/geopt/fieldmark.zscore` on fixed points/fields/key); `test_ripple_marked_field_detects_and_clean_does_not` (fake reference field: marked z > 4, clean |z| < 2); `test_point_count_mismatch_raises` (output with N points vs reference with N−1: `ValueError` naming both shapes); `test_missing_geopt_dir_names_the_variable`.
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement; `exp/geopt/fieldmark.py` becomes `from markweightx.schemes.ripple import *`.
- [ ] **Step 4:** Run; expected PASS.
- [ ] **Step 5:** Commit `feat: ripple scheme and geopt adapter`.

### Task 8: The pipeline and `markweightx run`

**Files:**
- Create: `markweightx/pipeline.py`
- Modify: `markweightx/cli.py` (add `run`; add `--model` to `embed`, `detect`, `verify`), `markweightx/quality.py` (add `accuracy`, `ppl`)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `pipeline.load_config(path) -> Config` (dataclass: `model`, `scheme`, `data`, `imprint` (with `phases`),
    `generate`, `detect` (`fpr`, `null`: `base|derivative|wrong_keys`), `edits`, `quality`, `out`, `smoke`);
    unknown keys raise `ConfigError` listing the allowed ones.
  - `pipeline.STEPS = ("prepare", "selfgen", "calibrate", "imprint", "generate", "detect", "edits", "quality", "report")`.
  - `pipeline.run(cfg: Config, *, only: list[str] | None = None, device: str | None = None) -> dict` (the report);
    each step writes `out/<step>.json` (or `.pt`) via `*.tmp` + rename and is skipped when it exists; `smoke: true`
    caps imprint at 5 steps per phase and samples at 4.
  - `out/report.json` keys: `model`, `scheme`, `detect` (`tpr_at_1fpr`, `thr`, `auc`, `n_wm`, `n_null`),
    `quality` (`before`, `after`), `edits` (`{edit: tpr_at_1fpr}`), `settings` (the resolved config).
  - CLI: `markweightx run CONFIG [--only STEP ...] [--device D] [--smoke]`.
  - CLI `detect` with `--model`: a directory of `.png` files for `taming` (encoded with the codec, scored against the
    key's frozen calibration); an `.npz` with `points`, `field`, `condition` for `geopt` (reference from the
    publisher's unmarked model; wrong-key null, no frozen calibration); texts as today otherwise.
  - Releases from `train.imprint` with `trainable: full` are written `format="state_dict"` (`weights.pt`), with
    `model={"name": ..., "params": ...}`.

- [ ] **Step 1: Write the failing tests:** `test_detect_png_dir_with_taming` and `test_detect_npz_with_geopt` (CLI on stand-in releases: exits 0, prints a p-value per input); one end-to-end test per adapter on its stand-in, `test_run_end_to_end_{text,vlm,agent,taming,geopt}` (each: all steps run; `report.json` has the keys above; values are finite); `test_run_resumes_and_skips_done_steps` (run, delete `edits.json`, rerun with a spy: only `edits`, `quality`, `report` execute); `test_partial_step_output_is_not_treated_as_done` (a leftover `imprint.pt.tmp` is ignored); `test_config_rejects_unknown_keys_and_steps`; `test_cli_run_smoke` (`markweightx run cfg.yaml --smoke` exits 0).
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement; text configs call the existing `imprint.embed` and `api.detect` so the text path is unchanged; other adapters call `train.imprint`.
- [ ] **Step 4:** Run `pytest tests/test_pipeline.py -v` (each end-to-end test under 3 minutes on CPU) and the whole suite.
- [ ] **Step 5:** Commit `feat: one pipeline, markweightx run`.

### Task 9: Demo configs, parity test, thin `exp/` wrappers

**Files:**
- Create: `configs/demos/{medgemma,agent,taming,geopt}.yaml`, `tests/test_demo_configs.py`
- Modify: `exp/arimg/pro_taming.py` (`colearn`, `eval` call the pipeline), `exp/geopt/run_mark.py` (calls the pipeline); README "Run a demo" section

**Interfaces:**
- Consumes: `load_config` (Task 8).

- [ ] **Step 1: Write the failing test** `test_demo_configs_match_last_nights_runs`: for each demo, parse its config and the run file it came from — `exp/arimg/jobs/colearn_a.txt` (P2) and `colearn_g.txt` (P14), `exp/geopt/jobs/sweep1.txt` (`s1_e02`), `exp/agent/jobs/agent-pro-d2.sbatch` and `agent-runv.sbatch` (the `a300t07` variant, temperature 0.7), `exp/med/jobs/Xmed-pro.sbatch` — and assert equality of every shared setting (lr, policy_lr, steps, eps, output_scale, delta, seq_len, bs, answer_len, final_words, final_tokens, temperature).
- [ ] **Step 2:** Run; expected FAIL (configs missing).
- [ ] **Step 3:** Write the four configs; point the `exp/` scripts at the pipeline (their sweeps and figure commands stay).
- [ ] **Step 4:** Run the test and the whole suite; expected PASS.
- [ ] **Step 5:** Commit `feat: demo configs reproduce last night's settings`.

### Task 10: HPG smoke runs

**Files:**
- Create: `scripts/hpg/run.sbatch` (args: `CONFIG`; 1 GPU; `--time=01:00:00`; output under `/blue/sgao1/ji757406.ucf/jobs/`; runs `markweightx run $CONFIG --smoke`, resumable)

- [ ] **Step 1:** Write the sbatch; run `hpg-precheck scripts/hpg/run.sbatch`. Expected: no protection/watchdog, no Chinese-company models.
- [ ] **Step 2:** rsync the repo to HPG; submit four jobs (one per demo config), `--account=qi855292.ucf --qos=qi855292.ucf`.
- [ ] **Step 3:** Poll with `sacct` every ≥ 60 s; when done, read each `report.json` with `tail`/`grep` only. Expected: four COMPLETED jobs, each report with all keys (numbers not meaningful at 5 steps).
- [ ] **Step 4:** Delete the smoke outputs' checkpoints (intermediate); keep the reports. Commit the sbatch.

### Task 11: Docs

**Files:**
- Modify: `docs/REFERENCE.md`; the site guide `docs.html` (held with the site artifact) and its homepage links
- Test: none (render check)

- [ ] **Step 1:** Update `docs/REFERENCE.md`: `Model` interface, `FieldScheme`, `markweightx run` and the config keys, the four demo configs, `MWX_WMAR_DIR` / `MWX_GEOPT_DIR`.
- [ ] **Step 2:** Rewrite the site guide in the restrained report style from the spec (kicker, title, lead, date; numbered sections with a one-line summary and an ink rule; step timelines; three key cards for method / edit / model; one callout style), same palette, font and six sizes as the home page; drop "planned" from the parts that now exist (adapters, `run`); keep it on what still does not (edit plugin class name if not built).
- [ ] **Step 3:** Render both pages headless; check font sizes are a subset of {112, 66, 48, 22, 16, 14}; publish the artifact.
- [ ] **Step 4:** Commit `docs: model adapters and the pipeline`.
