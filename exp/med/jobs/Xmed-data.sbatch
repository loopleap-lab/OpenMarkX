#!/bin/bash
#SBATCH --job-name=Xmed-data
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=/blue/qi855292.ucf/ji757406.ucf/jobs/%x-%A_%a.out
# Distillation text, one quarter per array task: the base answers the training-half questions (8 x 256 tokens). Runs once and exits; resumable (a rerun continues from saved progress); the GPU is busy from the first command.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/mwx.env
cd $B/MarkWeightX
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 30 > $B/jobs/Xmed-data-$SLURM_JOB_ID.gpu.csv &
R=$MWX_RUNS/med/medgemma-1.5-4b-it
python exp/med/distill_data.py --samples 8 --answer_len 256 --bs 64 --shard $SLURM_ARRAY_TASK_ID --num_shards 4
kill %1
