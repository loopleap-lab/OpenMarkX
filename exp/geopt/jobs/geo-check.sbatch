#!/bin/bash
#SBATCH --job-name=geo-check
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --partition=hpg-rtx6000,hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/blue/qi855292.ucf/ji757406.ucf/jobs/%x-%j.out
# Demo figures: the publisher's green/red check on four full-mesh test aircraft, watermarked vs an unmarked model.
# Runs once and exits; the GPU is busy from the first minutes.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/geopt/geopt.env
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > $B/jobs/geo-check-$SLURM_JOB_ID.gpu.csv &
cd $B/MarkWeightX
python exp/geopt/demo_check.py --geopt_dir $GEOPT --data $GEOPT_DATA --ckpt $GEOPT/checkpoints/craft_geopt_8layers.pt \
  --student $MWX_RUNS/geopt/sweep/s1_e02/student.pt --out $MWX_RUNS/geopt/check
kill %1
