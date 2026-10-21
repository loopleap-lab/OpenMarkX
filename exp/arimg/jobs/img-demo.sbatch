#!/bin/bash
#SBATCH --job-name=img-demo
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --partition=hpg-b200,hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/blue/sgao1/ji757406.ucf/jobs/%x-%j.out
# Demo assets: images with their tokens colored by WMAR's detector (student vs base), and FID vs the base images.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/arimg/arimg.env
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > /blue/sgao1/ji757406.ucf/jobs/img-demo-$SLURM_JOB_ID.gpu.csv &
RUN=${RUN:-$MWX_RUNS/arimg/taming_wmar}   # a run folder with eval_codes_<set>.pt
cp -n $MWX_RUNS/arimg/taming_wmar/eval_codes_teacher.pt $RUN/ 2>/dev/null || true
cd $WMAR
P=$B/MarkWeightX/exp/arimg/pro_taming.py
python $B/MarkWeightX/exp/arimg/demo_assets.py --run $RUN --out $RUN/demo
kill %1
