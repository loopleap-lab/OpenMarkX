#!/bin/bash
#SBATCH --job-name=med-demo
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/blue/qi855292.ucf/ji757406.ucf/jobs/%x-%j.out
# Demo pages: MedGemma answers colored token by token by the KGW and PRO detectors (no generation; the PRO detector
# runs its BERT embedder on the GPU). Runs once and exits.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/mwx.env
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 30 > $B/jobs/med-demo-$SLURM_JOB_ID.gpu.csv &
cd $B/MarkWeightX
python exp/med/demo_tokens.py --n 4
kill %1
