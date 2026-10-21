#!/bin/bash
#SBATCH --job-name=Ximg-train
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --partition=hpg-b200,hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/blue/sgao1/ji757406.ucf/jobs/%x-%j.out
# PRO step: WMAR's generation-time KGW learned into the transformer (LoRA), KL on the base model's own sequences. Runs once and exits; resumable; the GPU is busy from the first minutes.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/arimg/arimg.env
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > /blue/sgao1/ji757406.ucf/jobs/Ximg-train-$SLURM_JOB_ID.gpu.csv &
RUN=$MWX_RUNS/arimg/taming_wmar
cd $WMAR
P=$B/MarkWeightX/exp/arimg/pro_taming.py
python $P train --out $RUN --steps 3000 --bs 16
kill %1
