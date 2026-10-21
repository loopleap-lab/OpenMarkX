#!/bin/bash
#SBATCH --job-name=img-fold
#SBATCH --account=qi855292.ucf
#SBATCH --qos=qi855292.ucf
#SBATCH --partition=hpg-b200,hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/blue/sgao1/ji757406.ucf/jobs/%x-%j.out
# WMAR (original linear seeding) folded into the GPT's own token embedding and output head (cmd_fold), then 300 images
# (10 classes x 30) detected with WMAR's detector against the base images. Runs once and exits.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/arimg/arimg.env
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > /blue/sgao1/ji757406.ucf/jobs/img-fold-$SLURM_JOB_ID.gpu.csv &
OUT=$MWX_RUNS/arimg/sweep/F1_fold
cd $WMAR
python $B/MarkWeightX/exp/arimg/pro_taming.py fold --out $OUT --data_dir $MWX_RUNS/arimg/taming_wmar
cp -n $MWX_RUNS/arimg/taming_wmar/eval_codes_base.pt $MWX_RUNS/arimg/taming_wmar/eval_p_base.pt $MWX_RUNS/arimg/taming_wmar/eval.json $OUT/
python $B/MarkWeightX/exp/arimg/pro_taming.py eval --out $OUT --sets base,student --eval_per_class 30
kill %1
