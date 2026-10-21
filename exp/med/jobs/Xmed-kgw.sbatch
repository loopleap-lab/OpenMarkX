#!/bin/bash
#SBATCH --job-name=Xmed-kgw
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/blue/qi855292.ucf/ji757406.ucf/jobs/%x-%j.out
# Learn markllm-kgw-d into MedGemma's weights (LoRA on the language model). Runs once and exits; resumable (a rerun continues from saved progress); the GPU is busy from the first command.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/mwx.env
cd $B/MarkWeightX
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 30 > $B/jobs/Xmed-kgw-$SLURM_JOB_ID.gpu.csv &
R=$MWX_RUNS/med/medgemma-1.5-4b-it
python exp/med/embed.py --scheme markllm-kgw-d --bs 16 --answer_len 256
kill %1
