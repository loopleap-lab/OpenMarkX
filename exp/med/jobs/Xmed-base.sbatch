#!/bin/bash
#SBATCH --job-name=Xmed-base
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/blue/qi855292.ucf/ji757406.ucf/jobs/%x-%j.out
# The base model answers MedXpertQA-MM (accuracy, and the null for detection). Runs once and exits; the GPU is busy from the first command.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/mwx.env
cd $B/MarkWeightX
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 30 > $B/jobs/Xmed-base-$SLURM_JOB_ID.gpu.csv &
R=$MWX_RUNS/med/medgemma-1.5-4b-it
python exp/med/evaluate.py --base medgemma-1.5-4b-it --bs 64
kill %1
