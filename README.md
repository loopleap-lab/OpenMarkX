<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/static/logo-dark.png">
    <img alt="OpenMarkX" src="docs/static/logo-light.png" width="520">
  </picture>
</p>

<h3 align="center">OpenMark: Enabling Precise and Robust Text Watermark for Open-Source LLMs</h3>

<p align="center">
  <b>Project</b>&nbsp; · &nbsp;
  <a href="https://ucf-ml-research.github.io/OpenMarkX/">Website</a> ·
  <a href="https://arxiv.org/abs/2510.23891">Paper</a> ·
  <a href="#citation">Cite</a>
  <br>
  <b>Get started</b>&nbsp; · &nbsp;
  <a href="#quick-start">Quick start</a> ·
  <a href="https://colab.research.google.com/github/UCF-ML-Research/OpenMarkX/blob/main/notebooks/run_on_colab.ipynb">Colab</a> ·
  <a href="LEADERBOARD.md">Leaderboard</a>
  <br>
  <b>Reference</b>&nbsp; · &nbsp;
  <a href="#documentation">Docs</a> ·
  <a href="docs/methods.md">Methods</a> ·
  <a href="docs/models.md">Your model</a> ·
  <a href="docs/FAQ.md">FAQ</a> ·
  <a href="ROADMAP.md">Roadmap</a>
</p>

## About

**OpenMarkX** is the toolkit released with OpenMark (NeurIPS 2026). It learns a watermark into the weights of a
model before release, tests the watermark under the edits a user can make to the weights, and detects it later with
the developer's secret key. Language, vision-language, agent, image and physics models go through the same
commands.

<p align="center">
  <a href="docs/static/overview.pdf"><img alt="OpenMarkX: a model developer watermarks a model with OpenMark or one of 15 other methods (run); a model user may fine-tune, merge, quantize or prune it, or edit its text; a verifier with the developer's key detects the watermark in suspect text or verifies a suspect model; a watermark researcher compares methods under the same four edits (bench)" src="docs/static/overview.png" width="100%"></a>
</p>

**Overview of OpenMarkX.** Each column is one role, with its input above the OpenMarkX band and its output below. A
model developer watermarks a model with OpenMark or another method before release; a model user (red) may then
fine-tune, merge, quantize or prune it, or edit its text; a verifier with the developer's key tests a suspect text
or model; and a watermark researcher compares methods under the same modifications.

Users of an open-source LLM control both decoding and the weights, so a watermark added at decoding time can be
switched off. An open-source watermark has to be produced by the model itself under standard decoding and survive
edits to the weights. Learning a decoding watermark into the weights leaves two problems:

- **Learnability.** The model may not fully learn the watermark pattern, so what it generates and what the detector
  expects drift apart.
- **Robustness.** The learned watermark fades after fine-tuning, merging, quantization or pruning.

OpenMark addresses both:

- **A watermark policy trained jointly with the LLM**, so the watermark the LLM learns matches the detection pattern.
- **A perturbation-aware regularizer**, so the watermark stays detectable after weight-level edits.

On several open-source LLMs, OpenMark reaches 0.92 TPR at 1% FPR while lowering perplexity by 20.5% on average
relative to the strongest prior method, and keeps 0.57 TPR at 5% FPR after 0.5-ratio model merging, where prior
methods keep at most 0.34. [LEADERBOARD.md](LEADERBOARD.md) has the full results, and
[How OpenMarkX works](docs/architecture.md) shows the layers and the run pipeline.

## Demo

One command runs a whole experiment: it learns the mark into the weights, checks outputs with the key, applies the
edits and writes `report.json`. A stopped run picks up where it stopped.

<p align="center">
  <img src="docs/static/get-started.jpg" alt="One run of the GeoPT demo: install, run, and check a field later with the key" width="880">
</p>

```bash
openmarkx run configs/demos/geopt.yaml       # physics: GeoPT aircraft fields (needs OMX_GEOPT_DIR)
openmarkx run configs/demos/taming.yaml      # images: Taming Transformers (needs OMX_WMAR_DIR)
openmarkx run configs/demos/medgemma.yaml    # medical VQA: MedGemma 1.5
openmarkx run configs/demos/agent.yaml       # a ReAct agent on Llama-3.1-8B-Instruct
```

Add `--smoke` for a few steps, to check the setup. Publish `runs/demos/<name>/release/`; the key is not in it.

## Installation

```bash
git clone https://github.com/UCF-ML-Research/OpenMarkX.git
cd OpenMarkX
pip install -e .
bash scripts/install_deps.sh        # MarkLLM at a pinned commit, for the markllm-* methods
```

OpenMarkX needs Python 3.10 or later and a CUDA GPU. Without a GPU, the
[Colab notebook](https://colab.research.google.com/github/UCF-ML-Research/OpenMarkX/blob/main/notebooks/run_on_colab.ipynb)
runs on a free T4. Runs and keys go under `~/.openmarkx`; back up the keys, since a lost key cannot be recovered.
Code, keys and runs made under the old name MarkWeightX keep working. The [user guide](docs/user-guide.md) has the
other options.

## Quick start

A language model, step by step:

```bash
python exp/prepare_data.py                                  # once: C4 text and prompts
openmarkx embed --base llama-3.2-1b --watermark openmark    # prints {"release": ..., "key": ...}
openmarkx calibrate KEY --base llama-3.2-1b                 # the detection threshold, on the unmarked model
openmarkx modify RELEASE --edit finetune:steps=300 --out edited/
openmarkx generate edited/ --prompts prompts.jsonl --out texts.jsonl
openmarkx detect texts.jsonl --key KEY                      # one p-value per text
openmarkx verify edited/ --key KEY --prompts prompts.jsonl  # one verdict for a suspect model
```

The same in Python:

```python
import openmarkx as omx

prompts = ["The history of the city begins", "A good recipe for bread"]
release, key = omx.embed("llama-3.2-1b", "openmark")
key = omx.calibrate(key, "llama-3.2-1b")
edited = omx.modify(release, "finetune:steps=300")
hits = omx.detect(omx.generate(edited, prompts), key)
```

Check a file from another model type later with its key:

```bash
openmarkx detect field.npz --model geopt --key runs/demos/geopt/release
openmarkx detect images/ --model taming --key runs/demos/taming/release
```

`openmarkx serve` runs the same jobs from a web page. The [user guide](docs/user-guide.md) covers the rest: every
command and parameter, keys, configs and reproducing the paper.

## Methods

| Method | How the mark gets in |
|---|---|
| `openmark` (ours) | OpenMark: a watermark policy, learned into the weights |
| `markllm-<x>-d` | one of 12 MarkLLM decoding watermarks (KGW, SynthID, ...), learned into the weights |
| `gaussmark` | a keyed perturbation written into one weight matrix |
| `wmar`, `ripple` | a green list over image tokens; a keyed pattern over a predicted field |

[Methods](docs/methods.md) describes each method, the checks a method must pass, and how to add one.

## Watermarking your own model

A field model, such as a neural PDE surrogate, plugs into the `geopt` adapter through one small class named in the
config. With the `ripple` mark it runs through every step of `openmarkx run` and through `openmarkx detect`:

```yaml
# run.yaml
model: {name: geopt, factory: "my_field:MyField", ckpt: null, data: null}
scheme: {name: ripple, params: {eps: 0.05}}
```

The class gives the model, one forward call per case, the data splits and a normalizer. The
[tutorial](docs/models.md) has a runnable `my_field.py`, shows how to pair it with your own mark, and walks through
the GeoPT and Taming adapters in `openmarkx/models/` as worked examples. Running a new adapter type end to end is
on the [roadmap](ROADMAP.md).

## Benchmark

`openmarkx bench configs/arena/1b.yaml` scores methods under quantization, pruning, merging and fine-tuning, the
same edits as the paper. [Benchmark](docs/benchmark.md) describes the edits and the protocol, and
[LEADERBOARD.md](LEADERBOARD.md) has the results.

## Documentation

| document | what it covers |
|---|---|
| [User guide](docs/user-guide.md) | installing; keys; `openmarkx run` and its configs; every command and parameter; the Python API; reproducing the paper |
| [How OpenMarkX works](docs/architecture.md) | the model adapters, methods, edits, detection and benchmark layers; the run pipeline; repository layout |
| [Methods](docs/methods.md) | the watermark methods; the checks a method must pass; adding a method |
| [Watermarking your own model](docs/models.md) | a field model behind the `geopt` adapter, your own mark, and the GeoPT and Taming adapters as worked examples |
| [Benchmark](docs/benchmark.md) · [Leaderboard](LEADERBOARD.md) | the edits, the protocol and the results |
| [FAQ](docs/FAQ.md) | common questions and troubleshooting |
| [Roadmap](ROADMAP.md) · [Changelog](CHANGELOG.md) | what is done and planned; what changed in each release |

## Status and roadmap

Watermarking, editing, detection and the benchmark run today for the model types above.
[ROADMAP.md](ROADMAP.md) lists what comes next and the current limits.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Citation

If you use OpenMarkX or OpenMark in your research, please cite the [paper](https://arxiv.org/abs/2510.23891). You
can also [download the BibTeX entry](docs/static/openmark.bib).

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
