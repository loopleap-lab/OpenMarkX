# Leaderboard

Results reported in the PRO paper (NeurIPS 2026), on **LLaMA-3-8B**; OpenMark is the method called PRO in the paper.
Each method is distilled into the model's weights; detection uses 5,000 watermarked and 5,000 non-watermarked texts
(50-token C4 prefixes, 200-token continuations, temperature 1). PPL is the median perplexity under LLaMA-2-13B
(unwatermarked LLaMA-3-8B: 3.6). Settings of each modification are in the paper's appendix.

These numbers come from the paper and are not shipped results of this repository: OpenMarkX contains the toolkit
to run such a benchmark (`openmarkx bench`), not trained watermarked weights.

## Robustness to model modifications

TPR at 5% FPR / TPR at 1% FPR / PPL (paper, Table "robustness").

| Modification | Setting | Gu et al.-KTH | Gu et al.-KGW | Gloaguen et al.-KGW | **OpenMark** |
|---|---|---|---|---|---|
| Unaltered | | 0.78 / 0.54 / 5.0 | 0.96 / 0.83 / 4.9 | 0.94 / 0.84 / 6.2 | **0.99 / 0.92 / 4.2** |
| Quantization, 8 bits | GPTQ | 0.66 / 0.42 / 5.2 | 0.92 / 0.79 / 5.0 | 0.95 / 0.84 / 6.3 | 0.97 / 0.86 / 4.3 |
| | INT8 | 0.69 / 0.45 / 5.1 | 0.91 / 0.73 / 5.0 | 0.94 / 0.84 / 6.4 | 0.93 / 0.77 / 4.4 |
| Quantization, 4 bits | HQQ | 0.54 / 0.29 / 6.1 | 0.95 / 0.83 / 5.5 | 0.91 / 0.77 / 7.9 | 0.98 / 0.91 / 4.8 |
| | GPTQ | 0.64 / 0.36 / 5.5 | 0.91 / 0.74 / 5.1 | 0.94 / 0.81 / 7.3 | 0.94 / 0.79 / 5.1 |
| Pruning, Wanda | ρ = 0.2 | 0.73 / 0.47 / 8.6 | 0.96 / 0.84 / 7.5 | 0.97 / 0.90 / 9.9 | 0.99 / 0.92 / 7.0 |
| | ρ = 0.5 | 0.64 / 0.35 / 8.4 | 0.86 / 0.66 / 6.9 | 0.93 / 0.80 / 8.9 | 0.96 / 0.84 / 6.2 |
| Pruning, SparseGPT | ρ = 0.2 | 0.78 / 0.55 / 7.7 | 0.95 / 0.82 / 8.1 | 0.95 / 0.84 / 9.0 | 0.98 / 0.91 / 7.1 |
| | ρ = 0.5 | 0.76 / 0.50 / 8.1 | 0.91 / 0.74 / 7.8 | 0.93 / 0.78 / 9.6 | 0.97 / 0.89 / 8.0 |
| Merging, SLERP | t = 0.1 | 0.36 / 0.15 / 5.2 | 0.76 / 0.53 / 4.3 | 0.82 / 0.59 / 6.1 | 0.92 / 0.78 / 4.2 |
| | t = 0.3 | 0.21 / 0.08 / 5.1 | 0.65 / 0.39 / 4.0 | 0.62 / 0.37 / 5.3 | 0.83 / 0.60 / 4.1 |
| | t = 0.5 | 0.09 / 0.02 / 4.8 | 0.30 / 0.12 / 3.8 | 0.34 / 0.14 / 4.6 | 0.57 / 0.28 / 4.0 |
| | t = 0.7 | 0.06 / 0.02 / 4.3 | 0.15 / 0.05 / 3.7 | 0.19 / 0.06 / 4.2 | 0.31 / 0.12 / 3.9 |
| | t = 0.9 | 0.06 / 0.01 / 4.1 | 0.11 / 0.03 / 3.6 | 0.11 / 0.03 / 4.0 | 0.17 / 0.06 / 3.8 |
| Fine-tuning, OpenMathInstruct | s = 300 | 0.21 / 0.08 / 5.1 | 0.48 / 0.20 / 4.1 | 0.45 / 0.22 / 5.5 | 0.70 / 0.43 / 4.1 |
| | s = 600 | 0.17 / 0.05 / 4.9 | 0.38 / 0.16 / 3.7 | 0.41 / 0.19 / 5.2 | 0.51 / 0.26 / 3.9 |
| | s = 900 | 0.09 / 0.03 / 4.8 | 0.30 / 0.11 / 3.6 | 0.31 / 0.13 / 5.2 | 0.46 / 0.22 / 3.7 |
| | s = 1200 | 0.09 / 0.02 / 4.7 | 0.25 / 0.09 / 3.5 | 0.25 / 0.10 / 5.0 | 0.40 / 0.18 / 3.6 |
| | s = 1500 | 0.02 / 0.02 / 4.5 | 0.23 / 0.08 / 3.5 | 0.22 / 0.08 / 5.1 | 0.37 / 0.17 / 3.6 |
| Fine-tuning, OpenCodeInstruct | s = 300 | 0.27 / 0.11 / 7.3 | 0.63 / 0.40 / 6.9 | 0.65 / 0.41 / 7.4 | 0.71 / 0.43 / 7.0 |
| | s = 600 | 0.22 / 0.08 / 5.7 | 0.45 / 0.24 / 4.9 | 0.48 / 0.23 / 7.0 | 0.56 / 0.31 / 5.4 |
| | s = 900 | 0.16 / 0.06 / 4.8 | 0.32 / 0.14 / 5.0 | 0.34 / 0.16 / 6.8 | 0.50 / 0.25 / 5.1 |
| | s = 1200 | 0.11 / 0.03 / 4.7 | 0.29 / 0.12 / 4.3 | 0.31 / 0.12 / 6.3 | 0.40 / 0.18 / 4.8 |
| | s = 1500 | 0.07 / 0.02 / 4.6 | 0.26 / 0.10 / 4.2 | 0.30 / 0.12 / 5.9 | 0.39 / 0.17 / 4.0 |

## Detection at other false-positive rates

TPR (paper, Table "fpr_tpr").

| Method | TPR @ 0.1% FPR | TPR @ 1% FPR | TPR @ 10% FPR |
|---|---|---|---|
| Gu et al.-KTH | 25.8% | 53.6% | 84.0% |
| Gu et al.-KGW | 54.2% | 82.7% | 97.5% |
| **OpenMark** | **78.1%** | **92.3%** | **99.5%** |

## Paraphrasing

AUC under DIPPER paraphrasing (paper, Table "dipper").

| Method | DIPPER1 | DIPPER2 |
|---|---|---|
| Gu et al.-KTH | 0.82 | 0.79 |
| Gu et al.-KGW | 0.86 | 0.84 |
| Gloaguen et al.-KGW | 0.80 | 0.74 |
| **OpenMark** | **0.90** | **0.87** |

## Downstream tasks

Accuracy of LLaMA-3-8B before and after watermarking (paper, Table "downstream").

| Model | HellaSwag | ARC-C | AGIEval-Math | GSM8K |
|---|---|---|---|---|
| Base | 0.7607 | 0.5708 | 0.2963 | 0.7583 |
| Gu et al.-KGW | 0.7583 | 0.5683 | 0.2877 | 0.7582 |
| **OpenMark** | 0.7594 | 0.5705 | 0.2920 | 0.7582 |

## Run your own

`openmarkx bench arena.yaml` runs the same kind of benchmark (learn, edits, forgery) for any method on any base
and writes a leaderboard of your own runs (see README). Its edit grid is the toolkit's (LoRA fine-tune, linear
merge, magnitude pruning, bitsandbytes int8/nf4, word-level rewrites), not the paper's exact modification settings.
