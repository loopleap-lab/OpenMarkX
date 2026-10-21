#!/bin/bash
#SBATCH --job-name=img-sweep
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --partition=hpg-b200,hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/blue/sgao1/ji757406.ucf/jobs/%x-%A_%a.out
# Quick learnability check for PRO on Taming: one training variant per array task (line of $SWEEP), 400 steps; compare
# how fast the KL to WMAR's teacher falls. Runs once and exits; resumable; the GPU is busy from the first minutes.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/arimg/arimg.env
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > /blue/sgao1/ji757406.ucf/jobs/img-sweep-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.gpu.csv &
cd $WMAR
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $SWEEP); NAME=${LINE%% *}; ARGS=${LINE#* }
python $B/MarkWeightX/exp/arimg/pro_taming.py train --out $MWX_RUNS/arimg/sweep/$NAME \
  --data_dir $MWX_RUNS/arimg/taming_wmar --steps 400 --bs 16 --resume_every 100000 $ARGS
kill %1
