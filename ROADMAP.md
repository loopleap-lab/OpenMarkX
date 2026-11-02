# OpenMarkX Roadmap

OpenMarkX learns a watermark into the weights of an open-source model, tests it under edits, and detects it with
the owner's secret key. This roadmap follows that path: the models that go in, the methods, the edits, detection,
and the release that comes out.

Done items are checked, with the release that delivered them. Planned items are unchecked.

The path has three stages:
1. **Model adapters and the run pipeline (done):** text, agent, vision-language, image and physics models through
   `openmarkx run`, and 16 text methods on the leaderboard.
2. **The Python API for every adapter (next):** `embed`, `detect` and `modify` for every adapter, not only for
   text, and publishing a release to the Hugging Face Hub.
3. **A wider benchmark (later):** more imprints, attacks and access levels, and the published checkpoints of other
   papers.

## 1. Models

- [x] Text models: any Llama-style causal LM, with pinned revisions in `BASES` (v0.1.0)
- [x] Model adapters: `text`, `agent` (ReAct), `vlm` (MedGemma), `taming` (image tokens), `geopt` (physics fields) (v0.1.0)
- [x] `openmarkx run CONFIG` for every adapter, resumable, with the demo configs in `configs/demos/` (v0.1.0)
- [x] Plugin adapters registered with `@omx.register_model` and named in a config with `model.source` (v0.1.0)
- [ ] `openmarkx run` steps (prepare, generate, detect, quality) for a plugin adapter with a new name; today they exist for the five built-in adapters
- [ ] `openmarkx embed` and `verify` for adapters other than `text` (today: `openmarkx run`)

## 2. Methods

- [x] OpenMark (`openmark`, `openmark-coopt-c100`, `openmark-full-b01`), 12 MarkLLM methods distilled into the weights, GaussMark (v0.1.0)
- [x] `wmar` for image tokens and `ripple` for fields, with the rule co-learned with the model (v0.1.0)
- [x] One-file plugin methods: `openmarkx new-scheme`, `openmarkx check` (C1 to C10) (v0.1.0)
- [ ] `eps: auto` (strength search) for `ripple`
- [ ] The `sft` and `rl` imprints
- [ ] The `distill_sampling` recipe

## 3. Python API for every model

The website shows these calls for the image, vision-language, physics and agent demos. The demos ran with
`openmarkx run configs/demos/<name>.yaml`.

- [x] `omx.embed`, `omx.modify`, `omx.detect` for text models (v0.1.0)
- [ ] `omx.embed("<model>", "openmark")` for `taming-imagenet`, `medgemma-1.5-4b`, `geopt-aircraft` and `llama-3.1-8b-instruct` (agent): route `embed` to the model adapters by base name
- [ ] `release.push("<org>/<model>")`: upload the weights to the Hugging Face Hub, never the key
- [ ] `omx.detect(image | answer | field | report, release.key)`: dispatch on the release's adapter, and a `key` attribute on the release
- [ ] Edit parameters as keyword arguments: `omx.modify(release, "merge", t=0.3)` (today: `omx.modify(release, "merge:t=0.3")`)

## 4. Edits and the benchmark

- [x] Weight edits (`finetune`, `merge`, `prune`, `quantize`), text edits (`synonym`, `word_deletion`, `context_synonym`), `image_roundtrip` (v0.1.0)
- [x] Plugin edits with `@omx.register_edit` and `@omx.register_text_edit` (v0.1.0)
- [x] `openmarkx bench`: learn, edits, forgery and the leaderboard from an arena file (v0.1.0)
- [ ] The DIPPER paraphrase attack in `bench`
- [ ] A grant axis in `bench`
- [ ] LoRA-adapter and distillation edits

## 5. Detection

- [x] Calibrated nulls per text length, the wrong-key null, access levels R0 to R2 (v0.1.0)
- [x] `verify_model`: one verdict per model, sequential with a query budget (v0.1.0)
- [x] `port`: a checkpoint watermarked elsewhere, with its key files (v0.1.0)
- [ ] `TP-*` grants (prompt access)
- [ ] Importing the published checkpoints of other papers (`port` takes the checkpoint and its key files; fetching and converting them is per paper)

## 6. Running it

- [x] `openmarkx serve`: upload a dataset and learn a watermark from a web page, on this machine's GPUs or on Colab (v0.1.0)
- [ ] Running an `openmarkx run` config from the web page
