# Edits and the benchmark

An edit is what someone downstream does to a released model or to its outputs. OpenMarkX applies edits to test
whether a watermark survives them, and `openmarkx bench` runs every method under every edit, plus a forgery
attack, onto a leaderboard of your own runs. The paper's results are a separate table in
[LEADERBOARD.md](../LEADERBOARD.md).

- [Edits](#edits)
- [Adding an edit](#adding-an-edit)
- [Running the benchmark](#running-the-benchmark)
- [The arena file](#the-arena-file)
- [The steps and their scripts](#the-steps-and-their-scripts)
- [The leaderboard](#the-leaderboard)

## Edits

Edits are written `op:param=value,...`.

| Edit | Kind | Params (defaults) |
|---|---|---|
| `finetune` | weights | `steps=800, lr=1e-4, bs=8, seq_len=256, rank=16, data=` (a corpus file): LoRA on clean text, merged into the weights |
| `merge` | weights | `partner=` (required, e.g. `llama-3.2-1b-instruct`), `ratio=0.4` (the partner's weight): linear merge |
| `prune` | weights | `sparsity=0.5, method=magnitude`: unstructured magnitude pruning of the decoder's linear weights |
| `quantize` | weights | `bits=4` (bitsandbytes nf4) or `bits=8` (int8), applied at load time |
| `synonym` | text | `ratio=0.3`: swap that share of the words for WordNet synonyms |
| `word_deletion` | text | `ratio=0.3`: delete that share of the words |
| `context_synonym` | text | `ratio=0.3`: synonyms chosen by BERT from the context (downloads `bert-large-uncased`) |
| `image_roundtrip` | outputs | none: decode image tokens to pixels and encode them again, what a detector sees from an image file |

```bash
openmarkx modify RELEASE --edit finetune:steps=800 --out edited/      # a weight edit
openmarkx rewrite texts.jsonl --edit synonym:ratio=0.3 --out reworded.jsonl   # a text edit
```

A misspelled edit parameter is an error that lists the accepted ones, and so is a value the code would otherwise
ignore, such as `quantize:bits=3`. Both are caught before any model is loaded.

In a run config, the `edits` list applies to the chosen adapter:

| Adapter | Edits for it |
|---|---|
| `text`, `agent` | weight edits; text edits written `text:synonym:ratio=0.3` |
| `vlm` | text edits (`text:...`) |
| `taming` | `image_roundtrip` |
| `geopt` | `finetune:steps=...,lr=...`: a fine-tune of the field model on the training cases' true fields |

## Adding an edit

Two kinds of edit can be added, each as a function in a plugin file.
[`examples/my_edits.py`](../examples/my_edits.py) holds one of each, and
[`configs/arena/plugins_example.yaml`](../configs/arena/plugins_example.yaml) runs them next to the built-ins.
`tests/test_edit_plugins.py` runs both on CPU.

| To add | You write | You run | It shows up as |
|---|---|---|---|
| A weight edit (an attack on the released model) | a function with `@omx.register_edit` | `openmarkx modify REL --edit name:k=v --plugin file.py` | a leaderboard column |
| A text edit (an attack on the generated text) | a function with `@omx.register_text_edit` | `openmarkx rewrite texts.jsonl --edit name:k=v --plugin file.py` | a leaderboard column |

### A weight edit

A weight edit takes the release, an output directory, its own parameters (each with a default) and `device`, and
returns the edited release. Load the release as a plain model with `materialize`, change it, and hand it to
`edited_release`, which saves it and extends the release's lineage:

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
A misspelled parameter (`noise:scael=0.01`) is refused before any model is loaded. An edit that needs data takes a
`data=` parameter, as the built-in `finetune` does (`finetune:steps=800,data=/path/to/data_physics.pt`).

### A text edit

A text edit takes the list of texts and its parameters, and returns one string per text. Add `device` to its
parameters if it runs a model.

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
`openmarkx/edits/text.py`. An edit that returns the wrong number of texts is refused.

## Running the benchmark

One command runs the benchmark for a list of methods: learn, every edit, forgery, then the leaderboard. `openmarkx
evaluate` is the same command, under an older internal name.

```bash
openmarkx bench configs/arena/1b.yaml --dry-run     # what is done, what would run
openmarkx bench configs/arena/1b.yaml               # run the missing steps on this machine
openmarkx bench my_arena.yaml --queue mybox         # or hand them to the GPU scheduler (exp/scheduler.py)
```

- A step counts as done once its output exists, so a bench can be stopped and run again.
- A failed step skips the rest of that method.
- A plugin method file must pass `openmarkx check` first; `--no-check` skips this.
- The steps run the scripts in `exp/`, so `bench` runs from a checkout of the repository.

## The arena file

```yaml
# my_arena.yaml
base: llama-3.2-1b
methods: [openmark-coopt-c100, my_mark.py]                 # built-in names (with :params) or plugin files
edits: [finetune:steps=800, quantize:bits=4, text:synonym:ratio=0.3]   # optional; default: the full grid
```

| Key | Meaning | Default |
|---|---|---|
| `base` | the base model | `llama-3.2-1b` |
| `methods` | built-in names (with `:params`), plugin files, or `{method: NAME, steps: [...]}` to narrow the steps | required |
| `steps` | which of `learn`, `edits`, `forge` to run | all three |
| `edits` | the edit list; text edits are written `text:<name>:k=v` | the grid below |
| `plugins` | files that register edits, loaded before the edits are checked | none |
| `learn` | overrides for `exp/learnability.py`: `steps`, `bs`, `accum`, `bos` | the script's defaults |
| `tag` | the results directory under `$OMX_RUNS` | `e2` |

The default edit grid: `finetune` at 200, 800 and 2400 steps; `merge` at ratio 0.2, 0.4 and 0.6; `prune` at
sparsity 0.2 and 0.5; `quantize` at 8 and 4 bits; `word_deletion` and `synonym` at ratio 0.3.

`bench` checks every edit and its parameters when it reads the file, so a typo fails at once rather than on a GPU
an hour later. Each new edit becomes a column (`noise_scale0.01`, `typos_rate0.05`) after the default ones.

```yaml
# configs/arena/plugins_example.yaml
base: llama-3.2-1b
methods: [openmark, markllm-kgw-d]
plugins: [examples/my_edits.py]
edits: [finetune:steps=800, noise:scale=0.01, text:synonym:ratio=0.3, text:typos:rate=0.05]
```

## The steps and their scripts

The steps are these scripts, which can also be run one at a time:

| Script | What it runs |
|---|---|
| `exp/learnability.py --scheme NAME` | learn: embed, sample the held-out C4 prompts (311 of the first 500 documents are long enough), TPR at 1% FPR against the base model's texts, PPL |
| `exp/robustness.py --release DIR` | edits: every weight and text edit on one release, with the same threshold |
| `exp/spoof.py --release DIR` | forgery: count tokens in the owner's texts, bias the base model toward the over-represented ones, measure how much is flagged |
| `exp/leaderboard.py` | the board: one row per method; experimental variants and sweeps under Ablations |
| `exp/scheduler.py --host NAME` | a GPU queue (`$OMX_RUNS/sched/<host>/queue.json`) that waits for free memory |

The training recipe for teacher-learned schemes is `configs/embed.yaml`.

## The leaderboard

The board is printed and saved as `$OMX_RUNS/e2/<base>/leaderboard.md`; `exp/leaderboard.py --format latex` prints
the main table as LaTeX. A plugin method gets a main-board row; parameter sweeps (`name:k=v`) and experimental
variants go under Ablations. A C9 table reports false positives on texts cut to 50 and 100 tokens.

These are the default columns of the 1B board for OpenMark (TPR at 1% FPR, from our runs,
`$OMX_RUNS/e2/llama-3.2-1b/leaderboard.md`):

| method | learned | FT 800 | merge .4 | prune 20% | nf4 | delete 30% | synonym 30% |
|---|---|---|---|---|---|---|---|
| openmark | 0.88 | 0.07 | 0.55 | 0.89 | 0.89 | 0.83 | 0.67 |

A new edit is read against these: close to `nf4` means harmless, close to `FT 800` means it erases the mark.
This row is the `openmark` scheme (policy frozen, no FPL) on Llama-3.2-1B after 800 LoRA steps on clean C4 text;
the paper's fine-tuning numbers are for full OpenMark (CAWP and FPL) on LLaMA-3-8B fine-tuned on OpenMathInstruct
or OpenCodeInstruct, so the two are not comparable.

The board of your own runs is separate from [LEADERBOARD.md](../LEADERBOARD.md), which reports the paper's numbers
on LLaMA-3-8B. For the same kinds of edit there, after 4-bit HQQ quantization OpenMark keeps 0.91 TPR at 1% FPR
against 0.83 for the strongest baseline (Gu et al.-KGW), and after 300 steps of math fine-tuning 0.43 against
0.22 for the strongest baseline (Gloaguen et al.-KGW). The paper's
modifications (GPTQ, HQQ, Wanda, SparseGPT, SLERP, fine-tuning on OpenMathInstruct and OpenCodeInstruct) are not the
toolkit's edit grid.
