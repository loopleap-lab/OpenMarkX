<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/static/logo-dark.png">
    <img src="docs/static/logo-light.png" alt="MarkWeightX" width="520">
  </picture>
</p>

<p align="center"><strong>Watermarks for open-weight generative models</strong></p>

<p align="center">
  <a href="https://claude.ai/artifact/2WJtrDVjpgCsJeE1bSmEwp">
    <img src="https://img.shields.io/badge/Website-MarkWeightX-8B4B2E?logo=googlechrome&logoColor=white" alt="Website">
  </a>
  <a href="https://arxiv.org/abs/2510.23891">
    <img src="https://img.shields.io/badge/Paper-PRO%20(NeurIPS%202026)-b31b1b?logo=arxiv&logoColor=white" alt="Paper">
  </a>
  <a href="https://colab.research.google.com/github/UCF-ML-Research/MarkWeightX/blob/main/notebooks/run_on_colab.ipynb">
    <img src="https://img.shields.io/badge/Colab-run%20on%20a%20free%20GPU-F9AB00?logo=googlecolab&logoColor=white" alt="Colab">
  </a>
</p>

---

MarkWeightX trains a watermark into the weights of a model you release. Your private key then traces what the model
generates, and edited copies of it, back to you. Language, vision-language, image, agent and physics models all go
through the same commands.

<p align="center">
  <img src="docs/static/get-started.jpg" alt="One run of the GeoPT demo: install, run, and check a field later with the key" width="880">
</p>

## Install

```bash
git clone https://github.com/UCF-ML-Research/MarkWeightX.git && cd MarkWeightX
pip install -e .
bash scripts/install_deps.sh        # MarkLLM at a pinned commit, for the markllm-* methods
```

Python 3.10+ and a CUDA GPU. Without a GPU, the
[Colab notebook](https://colab.research.google.com/github/UCF-ML-Research/MarkWeightX/blob/main/notebooks/run_on_colab.ipynb)
runs on a free T4. Runs and keys go under `~/.markweightx`; back up the keys, a lost key cannot be recovered.

## Get started

One command runs a whole experiment: it learns the mark into the weights, checks the outputs with the key, applies
the edits and writes `report.json`. A stopped run picks up where it stopped.

```bash
markweightx run configs/demos/geopt.yaml       # physics: GeoPT aircraft fields (needs MWX_GEOPT_DIR)
markweightx run configs/demos/taming.yaml      # images: Taming Transformers (needs MWX_WMAR_DIR)
markweightx run configs/demos/medgemma.yaml    # medical VQA: MedGemma 1.5
markweightx run configs/demos/agent.yaml       # a ReAct agent on Llama-3.1-8B-Instruct
```

Add `--smoke` for a few steps, to check the setup. Publish `runs/demos/<name>/release/`; the key is not in it.

Check a file later with the key:

```bash
markweightx detect field.npz --model geopt --key runs/demos/geopt/release
markweightx detect images/ --model taming --key runs/demos/taming/release
markweightx detect texts.jsonl --key KEY
```

<details>
<summary>Language models, step by step</summary>

```bash
python exp/prepare_data.py                                    # once: C4 text and prompts
markweightx embed --base llama-3.2-1b --watermark pro         # prints {"release": ..., "key": ...}
markweightx calibrate KEY --base llama-3.2-1b                 # the detection threshold, on the unmarked model
markweightx modify RELEASE --edit finetune:steps=300 --out edited/
markweightx generate edited/ --prompts prompts.jsonl --out texts.jsonl
markweightx detect texts.jsonl --key KEY                      # one p-value per text
markweightx verify edited/ --key KEY --prompts prompts.jsonl  # one verdict for a suspect model
```

The same in Python:

```python
import markweightx as mwx

prompts = ["The history of the city begins", "A good recipe for bread"]
release, key = mwx.embed("llama-3.2-1b", "pro")
key = mwx.calibrate(key, "llama-3.2-1b")
edited = mwx.modify(release, "finetune:steps=300")
hits = mwx.detect(mwx.generate(edited, prompts), key)
```

`markweightx serve` runs the same from a web page.

</details>

## Watermark your own model

A new model takes two small files and a config; MarkWeightX itself does not change.

```
my_project/
├── my_model.py    how to load and run the model, and what one output unit is
├── my_mark.py     how the key shifts each unit, and how to score an output
└── run.yaml       points at both
```

`markweightx run run.yaml` then learns the mark into the weights and checks new outputs, as above. The GeoPT and
Taming adapters in `markweightx/models/` are worked examples.

## Methods and results

| Method | How the mark gets in |
|---|---|
| `pro` (ours) | PRO's watermark policy, learned into the weights |
| `markllm-<x>-d` | one of 12 MarkLLM decoding watermarks (KGW, SynthID, ...), learned into the weights |
| `gaussmark` | a keyed perturbation written into one weight matrix |
| `wmar`, `ripple` | a green list over image tokens; a keyed pattern over a predicted field |

Results under quantization, pruning, merging and fine-tuning are in **[LEADERBOARD.md](LEADERBOARD.md)**.
`markweightx bench configs/arena/1b.yaml` runs the same benchmark on your methods.

## Documentation

- [Website](https://claude.ai/artifact/2WJtrDVjpgCsJeE1bSmEwp): what it does, results and demos; its developer guide
  shows how to add a method, an edit or a model.
- [docs/REFERENCE.md](docs/REFERENCE.md): every command and parameter, keys, the checks a method must pass, the
  benchmark.

## Citation

```bibtex
@inproceedings{xue2026pro,
  title     = {PRO: Enabling Precise and Robust Text Watermark for Open-Source LLMs},
  author    = {Xue, Jiaqi and Zhao, Yifei and Al Ghanim, Mansour and Gao, Shangqian and
               Sun, Ruimin and Lou, Qian and Zheng, Mengxin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
