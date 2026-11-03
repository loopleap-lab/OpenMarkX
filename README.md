<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/static/logo-dark.png">
    <img alt="OpenMarkX" src="docs/static/logo-light.png" width="520">
  </picture>
</p>

<h3 align="center">OpenMark: Enabling Precise and Robust Text Watermark for Open-Source LLMs</h3>

<p align="center">
  <b>Project</b>&nbsp; · &nbsp;
  <a href="https://claude.ai/artifact/2WJtrDVjpgCsJeE1bSmEwp">Website</a> ·
  <a href="https://arxiv.org/abs/2510.23891">Paper</a> ·
  <a href="LEADERBOARD.md">Leaderboard</a> ·
  <a href="#citation">Cite</a>
  <br>
  <b>Get started</b>&nbsp; · &nbsp;
  <a href="#installation">Installation</a> ·
  <a href="#watermark-a-language-model">Language model</a> ·
  <a href="#run-a-demo">Demos</a> ·
  <a href="https://colab.research.google.com/github/loopleap-lab/OpenMarkX/blob/main/notebooks/run_on_colab.ipynb">Colab</a>
  <br>
  <b>Reference</b>&nbsp; · &nbsp;
  <a href="docs/user-guide.md">User guide</a> ·
  <a href="docs/methods.md">New method</a> ·
  <a href="docs/models.md">Your own model</a> ·
  <a href="docs/FAQ.md">FAQ</a> ·
  <a href="ROADMAP.md">Roadmap</a>
</p>

## About

**OpenMarkX** (NeurIPS 2026) is the toolkit released with OpenMark. It learns a watermark into the weights of a model
before release, tests the watermark under the edits a user can make to the weights, and detects it later with the
developer's secret key. Language, vision-language, agent, image and physics models run through one command,
`openmarkx run`, and methods are compared under the same edits with `openmarkx bench`.

OpenMark, the paper's method, trains the watermark policy jointly with the LLM, so the model learns the pattern the
detector expects, and adds perturbation-aware training, so the watermark survives fine-tuning, merging, quantization
and pruning. On three open-source LLMs it reaches 0.92 TPR at 1% FPR while lowering perplexity by 20.5% on average
relative to Gu et al.-KGW. The arXiv version calls the method PRO and the toolkit MarkWeightX; the old names still work.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/static/overview-dark.png">
    <img src="docs/static/overview-light.png" alt="OpenMarkX: a model developer watermarks a model (run); a model user may fine-tune, merge, quantize or prune it, or edit its text; a verifier with the developer's key detects the watermark; a watermark researcher compares methods under the same edits (bench)" width="100%">
  </picture>
</p>

## Key Features

- 🔏 **OpenMark** — a watermark policy trained with the model (CAWP) and forgetting-perturbation training (FPL).
- 🧪 **16 methods** — OpenMark, 12 MarkLLM decoding watermarks learned into the weights, GaussMark, WMAR and Ripple.
- 🔧 **Weight and text edits** — fine-tuning, merging, quantization and pruning of the weights; rewriting of the text.
- 🌐 **Beyond text** — vision-language models, agents, image generators and physics surrogates.
- 🧩 **Extensible** — any Hugging Face causal LM by `--base`; a new method in one file; a new model behind an adapter.

## Installation

```bash
git clone https://github.com/loopleap-lab/OpenMarkX.git
cd OpenMarkX
pip install -e .                    # images: pip install -e ".[vision]"
bash scripts/install_deps.sh        # MarkLLM at a pinned commit, for the markllm-* methods
python exp/prepare_data.py          # C4 text and held-out prompts (streams from Hugging Face)
```

OpenMarkX needs Python 3.10+ and a CUDA GPU; the [Colab notebook](https://colab.research.google.com/github/loopleap-lab/OpenMarkX/blob/main/notebooks/run_on_colab.ipynb)
runs on a free T4. Runs and keys go under `~/.openmarkx`; back up the keys, since a lost key cannot be recovered.

## How to Use OpenMarkX

### Watermark a Language Model

**1. Embed** the watermark into a base model and freeze the detection threshold:

```bash
openmarkx embed --base llama-3.2-1b --watermark openmark    # prints {"release": RELEASE, "key": KEY}
openmarkx calibrate KEY --base llama-3.2-1b
```

Publish `RELEASE`; keep `KEY`. `--watermark openmark-frozen` (the policy kept fixed, no FPL) is about three times
faster per step for a first run.

**2. Edit** the release as a user would:

```bash
openmarkx modify RELEASE --edit finetune:steps=300 --out edited/
```

**3. Detect** the watermark in suspect texts, or verify a suspect model:

```bash
openmarkx generate edited/ --prompts prompts.jsonl --out texts.jsonl
openmarkx detect texts.jsonl --key KEY                      # one p-value per text
openmarkx verify edited/ --key KEY --prompts prompts.jsonl  # one verdict for the model
```

`prompts.jsonl` holds one `{"text": ...}` per line (a `.txt` file with one prompt per line also works). The same in
Python:

```python
import openmarkx as omx

release, key = omx.embed("llama-3.2-1b", "openmark")
key = omx.calibrate(key, "llama-3.2-1b")
edited = omx.modify(release, "finetune:steps=300")
hits = omx.detect(omx.generate(edited, ["The history of the city begins"]), key)
```

### Run a Demo

One config runs a whole experiment: it learns the mark into the weights, checks outputs with the key, applies the
config's edits and writes `report.json`. A stopped run picks up where it stopped; `--smoke` runs a few steps.

```bash
openmarkx run configs/demos/geopt.yaml       # physics: GeoPT aircraft fields
openmarkx run configs/demos/taming.yaml      # images: Taming Transformers
openmarkx run configs/demos/medgemma.yaml    # medical VQA: MedGemma 1.5
openmarkx run configs/demos/agent.yaml       # a ReAct agent on Llama-3.1-8B-Instruct
openmarkx detect field.npz --model geopt --key ~/.openmarkx/runs/demos/geopt/release
```

### Benchmark Methods

```bash
openmarkx bench configs/arena/1b.yaml
```

scores methods under quantization, pruning, merging and fine-tuning and writes a leaderboard of your own runs;
[LEADERBOARD.md](LEADERBOARD.md) has the paper's results.

## FAQ

<details>
<summary><b>What does each demo need?</b></summary>

| Demo | Method | What it needs | Imprint time of the site's run |
|---|---|---|---|
| `geopt`, `geopt_colearn` | `ripple` | a [GeoPT](https://github.com/Physics-Scaling/GeoPT) checkout in `OMX_GEOPT_DIR` with `checkpoints/craft_geopt_8layers.pt`; the AirCraft data in `$OMX_DEMO_DATA/geopt/aircraft_npys` | `geopt`: 7 s (edits: 2 min) |
| `taming` | `wmar` | a WMAR checkout in `OMX_WMAR_DIR`; the Taming checkpoint directory in `TAMING`; the `vision` extra | 55 min |
| `medgemma` | `openmark-frozen` | the gated `google/medgemma-1.5-4b-it`; data from `exp/med/prepare.py` and `exp/med/distill_data.py`; the `vision` extra | 1 h 51 min |
| `agent` | `openmark-frozen` | data from `exp/agent/prepare.py` and `exp/agent/distill_data.py` | 36 min |

Each demo needs one GPU. Runs go to `~/.openmarkx/runs/demos/<name>/`; publish its `release/`, which does not hold
the secret key.

</details>

<details>
<summary><b><a href="docs/models.md">How do I watermark my own model?</a></b></summary>

A causal language model needs only `--base` with a Hugging Face repo id or a local directory. The paper's LLaMA-3-8B
loads as `--base meta-llama/Meta-Llama-3-8B`.

A field model, such as a neural PDE surrogate, plugs into the `geopt` adapter through one small class named in the
config:

```yaml
# run.yaml
model: {name: geopt, factory: "my_field:MyField", ckpt: null, data: null}
scheme: {name: ripple, params: {eps: 0.05}}
imprint:
  phases:
    - {steps: 400, lr: 1.0e-3, trainable: full}
detect: {fpr: 0.01, null: wrong_keys, n_null: 20}
edits: ["finetune:steps=20,lr=1e-4"]
quality: rel_l2
out: runs/my_field
```

```bash
PYTHONPATH=. openmarkx run run.yaml
```

The [tutorial](docs/models.md) has a runnable `my_field.py` and walks through the GeoPT and Taming adapters.

</details>

<details>
<summary><b><a href="docs/methods.md">How do I add a watermark method?</a></b></summary>

`openmarkx new-scheme` writes a one-file template, `openmarkx check` runs the checks a method must pass, and
`openmarkx describe` lists which methods support which model types. [`examples/my_scheme.py`](examples/my_scheme.py)
is a worked example; [Methods](docs/methods.md) describes each built-in method.

</details>

<details>
<summary><b><a href="docs/user-guide.md">Where are all the commands and parameters?</a></b></summary>

The [user guide](docs/user-guide.md) covers keys, `openmarkx run` and its configs, every command and parameter, the
Python API and reproducing the paper; [How OpenMarkX works](docs/architecture.md) shows the layers and the run
pipeline. `openmarkx serve` runs the text recipe from a web page.

</details>

## Acknowledgements

OpenMarkX integrates watermarking methods and models from prior work, including MarkLLM, GaussMark, WMAR, Taming
Transformers and GeoPT. We thank the original authors for making their work publicly available.

## License

OpenMarkX is released under the [Apache License 2.0](LICENSE). Code adapted from other projects keeps its original
license (e.g. `openmarkx/third_party/pro/LICENSE`); MarkLLM is installed from its own repository.

## Citation

If you find our repo useful, please cite our paper ([BibTeX](docs/static/openmark.bib)):

```bibtex
@inproceedings{xue2026openmark,
  title     = {{OpenMark}: Enabling Precise and Robust Text Watermark for Open-Source {LLMs}},
  author    = {Xue, Jiaqi and Zhao, Yifei and Al Ghanim, Mansour and Gao, Shangqian and
               Sun, Ruimin and Lou, Qian and Zheng, Mengxin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2510.23891}
}
```
