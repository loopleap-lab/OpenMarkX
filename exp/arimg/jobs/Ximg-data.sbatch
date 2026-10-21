#!/bin/bash
#SBATCH --job-name=Ximg-data
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --partition=hpg-b200,hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/blue/sgao1/ji757406.ucf/jobs/%x-%A_%a.out
# Distillation data: the base Taming transformer's own token sequences, a quarter of the 1000 ImageNet classes per array task (20 per class). Runs once and exits; resumable; the GPU is busy from the first minutes.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/arimg/arimg.env
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > /blue/sgao1/ji757406.ucf/jobs/Ximg-data-$SLURM_JOB_ID.gpu.csv &
RUN=$MWX_RUNS/arimg/taming_wmar
cd $WMAR
P=$B/MarkWeightX/exp/arimg/pro_taming.py
python $P data --out $RUN --classes 0-999 --per_class 20 --shard $SLURM_ARRAY_TASK_ID --num_shards 4
kill %1
