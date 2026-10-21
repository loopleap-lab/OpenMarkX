#!/bin/bash
#SBATCH --job-name=img-edit
#SBATCH --account=qi855292.ucf
#SBATCH --qos=qi855292.ucf
#SBATCH --partition=hpg-b200,hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/blue/sgao1/ji757406.ucf/jobs/%x-%j.out
# Closed-form PRO for WMAR (original linear seeding): solve the output-head change from the base model's own sequences,
# report the KL to WMAR's teacher before/after on held-out sequences, then sample 300 images and detect. Runs once.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/arimg/arimg.env
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > /blue/sgao1/ji757406.ucf/jobs/img-edit-$SLURM_JOB_ID.gpu.csv &
OUT=$MWX_RUNS/arimg/edit_${EDITNAME:-r1}; mkdir -p $OUT
cp $MWX_RUNS/arimg/taming_wmar/eval_codes_base.pt $MWX_RUNS/arimg/taming_wmar/eval_p_base.pt $MWX_RUNS/arimg/taming_wmar/eval.json $OUT/
cd $WMAR
python $B/MarkWeightX/exp/arimg/pro_taming.py edit --out $OUT --data_dir $MWX_RUNS/arimg/taming_wmar ${EDITARGS:-}
python $B/MarkWeightX/exp/arimg/pro_taming.py eval --out $OUT --sets base,student --eval_per_class 30
kill %1
