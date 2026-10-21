#!/bin/bash
#SBATCH --job-name=img-coeval
#SBATCH --account=qi855292.ucf
#SBATCH --qos=qi855292.ucf
#SBATCH --partition=hpg-b200,hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/blue/sgao1/ji757406.ucf/jobs/%x-%j.out
# Detection only, for a finished colearn run ($NAME) whose in-job eval failed (e.g. OOM while sampling on a 96 GB
# card): 300 images, smaller sampling batches, expandable CUDA segments. Runs once and exits.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/arimg/arimg.env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > /blue/sgao1/ji757406.ucf/jobs/img-coeval-$SLURM_JOB_ID.gpu.csv &
OUT=$MWX_RUNS/arimg/colearn/$NAME
cp -n $MWX_RUNS/arimg/taming_wmar/eval_codes_base.pt $OUT/
cd $WMAR
python $B/MarkWeightX/exp/arimg/pro_taming.py eval --out $OUT --sets base,student --eval_per_class 30 --bs 20
kill %1
