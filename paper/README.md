# OpenMark experiment code

The training and evaluation scripts behind the OpenMark experiments of the OpenMarkX paper (NeurIPS 2026): logit distillation of the watermark into
LLaMA-3, Phi-2 and LLaMA-3.2 with the jointly trained watermark policy and perturbation-aware training, the
distillation baselines of Gu et al., and the weight and text modifications of the robustness experiments. The
OpenMarkX toolkit in the parent directory runs the same method (`--watermark openmark`) through its own pipeline;
this directory keeps the scripts the paper's experiments used.

## Setup

```bash
conda create -n openmark python=3.11
conda activate openmark
pip install -r requirements.txt
```

The paper's experiments ran on NVIDIA A100 80GB GPUs. Run the scripts from this directory.

**Secret token mapping.** The watermark key is a token mapping, one policy dimension per vocabulary entry. It is not
in this repository; make one for your tokenizer's vocabulary size and keep it private:

```bash
python -c "import json, random; json.dump([random.randrange(300) for _ in range(128256)], open('watermarks/pro/300_mapping_128256.json', 'w'))"
```

The scripts read `watermarks/pro/300_mapping_<vocabulary size>.json` unless a `mapping_file` is given.

**Initial policy.** The training scripts start the policy from `transform_model_cbert.pth`, shipped with the toolkit
at `../openmarkx/third_party/pro/transform_model_cbert.pth`.

## Layout

| Path | Contents |
|---|---|
| `watermarks/pro/` | the OpenMark watermark: watermark processor and transform (policy) model |
| `train_logit_distill_Co-optimization.py` | logit distillation with the jointly trained watermark policy |
| `train_logit_distill_robust.py` | logit distillation with perturbation-aware training against weight edits |
| `train_logit_distill.py`, `train_sampling_distill.py` | the distillation baselines of Gu et al. (KGW, KTH, Aar) |
| `experiments/` | generation, detection and metrics (`compute_metrics_pro.py` and `compute_auc_pro.py` for OpenMark) |
| `Robustness/` | the weight and text modifications: fine-tuning, model merging, pruning (Wanda, SparseGPT), quantization, rewriting |
| `scripts/` | shell entry points for training and evaluation |

## Train

```bash
bash scripts/train/train_logit_distill_PRO.sh <watermark> <output_dir/> <master_port> [model] [mapping_file]
# e.g.
bash scripts/train/train_logit_distill_PRO.sh pro-gamma0.5-delta1 ./models/ 29500 meta-llama/Meta-Llama-3-8B-Instruct
```

`<watermark>` sets OpenMark's γ and δ (see the script for the presets).

## Evaluate

```bash
bash scripts/evaluate/generate_and_evaluate_PRO.sh <dataset> <output_file> <tokenizer> <ppl_model> <models>...
```

generates continuations of dataset prompts with each model, then computes perplexity, OpenMark detection scores,
AUC and TPR at fixed FPR. Weight and text modifications are in `Robustness/`, one script per modification.

## Acknowledgements

This code builds on [On the Learnability of Watermarks for Language Models](https://github.com/chenchenygu/watermark-learnability)
(Gu et al., ICLR 2024). `watermarks/kgw` is from [lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking),
parts of `watermarks/kth` are from [watermark](https://github.com/jthickstun/watermark), and `Robustness/prune`
includes [Wanda](https://github.com/locuslab/wanda) and [SparseGPT](https://github.com/IST-DASLab/sparsegpt). The
code is released under the Apache License 2.0 ([LICENSE](LICENSE)).
