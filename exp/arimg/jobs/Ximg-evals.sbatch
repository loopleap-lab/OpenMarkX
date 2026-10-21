#!/bin/bash
#SBATCH --job-name=Ximg-evals
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --partition=hpg-b200,hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/blue/sgao1/ji757406.ucf/jobs/%x-%j.out
# The student (watermark in the weights, no logit processor): 10 classes x 100 images, detected with WMAR's detector. Runs once and exits; resumable; the GPU is busy from the first minutes.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/arimg/arimg.env
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > /blue/sgao1/ji757406.ucf/jobs/Ximg-evals-$SLURM_JOB_ID.gpu.csv &
RUN=$MWX_RUNS/arimg/taming_wmar
cd $WMAR
P=$B/MarkWeightX/exp/arimg/pro_taming.py
python $P eval --out $RUN --sets student
kill %1
