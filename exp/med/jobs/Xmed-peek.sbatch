#!/bin/bash
#SBATCH --job-name=Xmed-peek
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/blue/qi855292.ucf/ji757406.ucf/jobs/%x-%j.out
# An early look: the latest intermediate adapters of the PRO and KGW runs answer 200 eval questions; detection and
# accuracy against the base answers. Read-only on the runs. Runs once and exits; the GPU is busy from the start.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/mwx.env
cd $B/MarkWeightX
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 30 > $B/jobs/Xmed-peek-$SLURM_JOB_ID.gpu.csv &
R=$MWX_RUNS/med/medgemma-1.5-4b-it
python exp/med/peek.py --run $R/pro --n 200
python exp/med/peek.py --run $R/markllm-kgw-d --n 200
kill %1
