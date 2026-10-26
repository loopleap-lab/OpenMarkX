#!/bin/bash
#SBATCH --job-name=img-opt
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/blue/qi855292.ucf/ji757406.ucf/jobs/%x-%A_%a.out
# OpenMark on Taming, quality variants: one line of $SWEEP per array task (NAME|policy phase args|catch-up args).
# Phase A: policy and GPT learn together (from the base model); phase B: policy fixed, the GPT catches up. Then
# detection (300 images vs base) and a quick FID (10 per class x 1000 classes) of the student. Runs once and exits;
# every phase is resumable and skipped when its output exists.
# Smoke test: SWEEP=.../opt_smoke.txt FID_HI=2 (3 steps per phase, FID on 2 classes).
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/arimg/arimg.env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SWEEP=${SWEEP:-$B/OpenMarkX/exp/arimg/jobs/opt_a.txt}
IFS='|' read -r NAME A_ARGS B_ARGS < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $SWEEP)
R=$MWX_RUNS/arimg/opt; DATA=$MWX_RUNS/arimg/taming_wmar; P=$B/OpenMarkX/exp/arimg
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > $B/jobs/img-opt-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.gpu.csv &
cd $WMAR
[ -f $R/${NAME}_A/policy.pt ] || python $P/pro_taming.py colearn --out $R/${NAME}_A --data_dir $DATA --bs 32 --resume_every 200 $A_ARGS
[ -f $R/$NAME/policy.pt ] || python $P/pro_taming.py colearn --out $R/$NAME --data_dir $DATA --bs 32 --resume_every 200 \
  --init_dir $R/${NAME}_A $B_ARGS
cp -n $DATA/eval_codes_base.pt $R/$NAME/
[ -f $R/$NAME/eval.json ] || python $P/pro_taming.py eval --out $R/$NAME --sets base,student --eval_per_class 30 --bs 20
python $P/fid50k.py gen --out $MWX_RUNS/arimg/fid10k/$NAME --set student --lo 0 --hi ${FID_HI:-1000} --per_class 10 --bs 10 \
  --student $R/$NAME/transformer.pt
python $P/fid50k.py fid --out $MWX_RUNS/arimg/fid10k/$NAME --base_dir $MWX_RUNS/arimg/fid10k --ref_dir $MWX_RUNS/arimg/fid50k
kill %1
