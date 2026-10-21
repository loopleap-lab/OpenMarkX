#!/bin/bash
#SBATCH --job-name=Xmed-smoke
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/blue/qi855292.ucf/ji757406.ucf/jobs/%x-%j.out
# Smoke test: every step of sub-project B at toy size (32 questions, 64 pairs, 20 steps). Runs once and exits; the GPU is busy from the first command.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/mwx.env
cd $B/MarkWeightX
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 30 > $B/jobs/Xmed-smoke-$SLURM_JOB_ID.gpu.csv &
R=$MWX_RUNS/med/medgemma-1.5-4b-it
P=$MWX_RUNS/med/pairs_smoke.jsonl
python exp/med/evaluate.py --base medgemma-1.5-4b-it --n 32 --new_tokens 256
python exp/med/distill_data.py --n 64 --out $P
python exp/med/embed.py --scheme pro --steps 20 --pairs $P --tag=-smoke
python exp/med/embed.py --scheme markllm-kgw-d --steps 20 --pairs $P --tag=-smoke
python exp/med/evaluate.py --release $R/pro-smoke --n 32 --new_tokens 256
python exp/med/evaluate.py --release $R/markllm-kgw-d-smoke --n 32 --new_tokens 256
kill %1
