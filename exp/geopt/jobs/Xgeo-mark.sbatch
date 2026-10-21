#!/bin/bash
#SBATCH --job-name=Xgeo-mark
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/blue/qi855292.ucf/ji757406.ucf/jobs/%x-%j.out
# PRO for the field model: calibrate the keyed mark, learn it into a copy of the weights, evaluate, save demo cases (resumable). Runs once and exits; the GPU is busy from the first minutes.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/geopt/geopt.env
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 30 > $B/jobs/Xgeo-mark-$SLURM_JOB_ID.gpu.csv &
cd $B/MarkWeightX
python exp/geopt/run_mark.py --geopt_dir $GEOPT --data $GEOPT_DATA \
  --ckpt $GEOPT/checkpoints/craft_geopt_8layers.pt --out $MWX_RUNS/geopt/mark
kill %1
