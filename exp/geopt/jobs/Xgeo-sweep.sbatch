#!/bin/bash
#SBATCH --job-name=Xgeo-sweep
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --partition=hpg-rtx6000,hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/blue/qi855292.ucf/ji757406.ucf/jobs/%x-%A_%a.out
# PRO for GeoPT, one setting per array task (the table in $SWEEP, one line of run_mark.py arguments per task).
# Each task: train the student, evaluate with publisher-side detection, save demo cases. Resumable; runs once.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/geopt/geopt.env
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > $B/jobs/Xgeo-sweep-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.gpu.csv &
cd $B/MarkWeightX
ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $SWEEP)
NAME=$(echo $ARGS | awk '{print $1}'); ARGS=$(echo $ARGS | cut -d' ' -f2-)
python exp/geopt/run_mark.py --geopt_dir $GEOPT --data $GEOPT_DATA --ckpt $GEOPT/checkpoints/craft_geopt_8layers.pt \
  --out $MWX_RUNS/geopt/sweep/$NAME $ARGS
kill %1
