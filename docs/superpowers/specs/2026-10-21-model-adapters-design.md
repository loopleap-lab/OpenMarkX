# MarkWeightX: model adapters and one pipeline for every modality

Date: 2026-10-21. Status: draft for review.

## Why

On 2026-10-20 we extended MarkWeightX to four new settings: MedGemma (image + question in, text out), a ReAct
agent on Llama-3.1-8B (only the final answer is checked), Taming (autoregressive image tokens, the mark learned
from WMAR) and GeoPT (a physics surrogate predicting fields on 3D points). Each ran as its own script under `exp/`.
Agent and MedGemma reuse the core package; Taming (`exp/arimg/pro_taming.py`) and GeoPT (`exp/geopt/`) import
nothing from it and re-implement training, keys, the null, the threshold and detection.

The goal is a product: someone with a new open model clones the repo and runs it from the terminal. A web UI is a
later plan, not part of this work. So the four extensions should become first-class parts of the package, behind one
interface, the way text methods already are.

## Goals

1. A model adapter, `mwx.Model`, says how to load a model, run it, and what one output unit is. Built-in adapters:
   `text` (today's causal LM), `vlm` (MedGemma; `vlm.py` moved onto the main path), `taming`, `geopt`, and `agent`
   (a ReAct loop around a text model).
2. `mwx.Scheme` covers two output families without breaking existing schemes:
   - token outputs (text, VLM answers, image tokens, agent answers): unchanged, `target_logits` + KL;
   - continuous outputs (GeoPT fields): a new `target_output`, trained with MSE; detection uses a reference model,
     which is the existing `Access.R1`.
3. The schemes from the extensions become registered schemes: `wmar` (Taming's green list over the codebook, with
   PRO's co-learning and the "learn, then fix the list and train" two-phase recipe as options) and `ripple`
   (GeoPT's keyed field pattern).
4. One pipeline, `markweightx run <config.yaml>`, with the steps all four extensions share: prepare data,
   self-generate base outputs, (optionally) calibrate the strength, learn the mark into the weights, generate
   base/student sets, score and threshold at the chosen FPR, run edits, report quality. Each step is resumable
   (skips when its output exists), as the scripts are today. It writes `report.json` and the demo assets.
5. The existing CLI (`embed`, `modify`, `detect`, `verify`, `bench`, ...) keeps working unchanged and gains
   `--model <adapter or name>`.
6. Four configs, `configs/demos/{medgemma,agent,taming,geopt}.yaml`, express last night's experiments.
7. Tests: one tiny fake model per adapter runs the whole pipeline on CPU in minutes.
8. The developer guide (docs) is rewritten to match the new interface (see "Docs" below).

## Non-goals

- A web UI or new frontend work.
- New experiments, new methods, or behaviour-level agent watermarks (Agent Guide / AgentMark style).
- Re-running the four demos on HPG. Acceptance is the CPU test suite (decided 2026-10-21); last night's results
  stand as they are.
- Changing the text path's behaviour: existing schemes, keys and releases load and score as before.

## Anti-goals

- A second, parallel API. Modalities extend `Scheme`, `Release`, `Key`, `imprint`, `scoring`; they do not fork them.
- Moving modality code into the core that only one experiment needs (sweeps, figure scripts stay in `exp/`).
- Claiming capabilities the code does not have: interfaces that are designed but not implemented stay marked
  "planned" in docs and on the site.

## Design

### Model adapter

```python
class Model:                          # markweightx/models/base.py
    name: str                         # registry name: "text", "vlm", "taming", "geopt", "agent"
    unit: str                         # "token", "image token", "surface point", ...
    output: str                       # "tokens" | "field"
    def load(self, path, device): ... # the released weights (base, or base + adapter)
    def run(self, net, inputs, **kw): # one batch of inputs in, outputs out (generation or prediction)
    def trainable(self, net): ...     # which weights the imprint trains: "lora" (default) or "full" or a subset
    def codec(self): ...              # outputs <-> what the scheme scores (tokenizer, VQ encode/decode, identity)
```

Registered with `@register_model`, loadable as a plugin file like schemes (`--model my_model.py`, same trust rules).
`Release` records the adapter name and its params, so `detect` and `verify` rebuild it.

### Scheme, two output families

- Token family: as today. `setup(model, codec, device, key)`, `target_logits(prefix, logits)`, `score(units)`. The
  only generalization is that `score` and the null count "units" from the adapter's codec instead of assuming a text
  tokenizer; text schemes keep receiving text through the text codec.
- Field family: `target_output(inputs, output) -> output` (the marked target), `score(output, reference)`.
  `access = Access.R1` (detection needs the publisher's unmarked model to compute the reference).
- PRO's co-learning stays `trainable_params` / `aux_loss`; the two-phase recipe becomes a training option
  (`imprint.phases: [{colearn: true, steps: 800}, {fix_rule: true, steps: 1000}]`).

### Imprint

`imprint.embed` dispatches on `output`: KL to `target_logits` for tokens (LoRA by default; `full` allowed), MSE to
`target_output` for fields. `embed_vlm` is folded in as the `vlm` adapter's batching. Resume works as today.

### Pipeline (`markweightx/pipeline.py`, `markweightx run`)

Config sketch:

```yaml
model: {name: geopt, path: checkpoints/craft_geopt_8layers.pt}
scheme: {name: ripple, eps: auto}        # auto: step "calibrate" picks the smallest detectable strength
data: {name: aircraft, dir: data/aircraft_npys}
imprint: {steps: 2000, trainable: full}
detect: {fpr: 0.01, null: base}          # base outputs | a fine-tuned copy | wrong keys
edits: [finetune:epochs=2, finetune:epochs=10]
quality: rel_l2                          # accuracy | fid | rel_l2 | ppl
out: runs/geopt
```

Steps: `prepare`, `selfgen`, `calibrate`, `imprint`, `generate`, `detect`, `edits`, `quality`, `report`. Output:
`runs/<name>/report.json` (TPR at FPR, AUC, quality before/after, per edit) and `runs/<name>/assets/`. On HPG the
same command runs inside one generic sbatch (`scripts/hpg/run.sbatch <config>`), resumable after a time limit.

### Built-in adapters and what moves where

| From | To | Kept in `exp/` |
|---|---|---|
| `markweightx/vlm.py`, `exp/med/*` | `models/vlm.py`, scheme reuse, `quality: accuracy` | MedXpertQA prep script |
| `exp/agent/hotpot_agent.py`, `distill_data.py` | `models/agent.py` (ReAct loop, final answer as output) | HotpotQA prep |
| `exp/arimg/pro_taming.py` | `models/taming.py`, `schemes/wmar.py`, edit `image_roundtrip`, `quality: fid` | sweeps, `fold`/`edit` variants, figure scripts |
| `exp/geopt/fieldmark.py`, `run_mark.py` | `models/geopt.py`, `schemes/ripple.py`, `quality: rel_l2` | sweeps, figure scripts |

### Tests

- One tiny fake per adapter (random-weight GPT for text/agent, the tiny Gemma 3 already used by `test_vlm.py`, a
  tiny VQ + GPT for taming, a small MLP on random points for geopt). Each runs `markweightx run` end to end on CPU.
- Existing 77 tests keep passing.
- Environment: a CPU-only virtualenv on lab2 (lab2 GPUs are off limits).

### Docs

Rewrite the developer guide (`docs.html` in the site artifact, and `docs/REFERENCE.md` in the repo) for the new
interface, in a restrained, professional version of the reference site's report style: a kicker + large title +
one-line lead + date; numbered sections, each with a one-sentence summary and an ink rule; step lists set as a
two-column timeline (step on the left, what happens on the right); a row of numbered key cards for the three ways to
extend (method, edit, model); a single highlighted callout style for cautions. No tilted cards, stamps or ticker on
the docs page. Same palette, typeface and six-size type scale as the home page.

## Decided

- Acceptance: CPU tests only; no HPG re-run (2026-10-21).
- The home page's `my_mark.py` examples already use the real Scheme names (`setup`, `target_logits`, `score`).

## Left to the builder

- Exact module layout inside `markweightx/models/` and how `models.py` (today's text helpers) is split.
- Fake-model sizes for the tests, as long as each adapter test finishes in a few minutes on CPU.
- Config key names beyond the sketch above.
- Which `exp/` files become thin wrappers over the pipeline and which are deleted once covered.
