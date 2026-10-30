#!/bin/bash
#SBATCH --job-name=img-fid50k
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-12%8
#SBATCH --output=/blue/qi855292.ucf/ji757406.ucf/jobs/%x-%A_%a.out
# FID as in WMAR (50 per class x 1000 classes, vs ImageNet val): tasks 0-11 generate (set, 250-class chunk) and keep
# Inception features; task 12 extracts the reference features. Then: fid50k.py fid (CPU). Each task runs once and
# exits; a rerun resumes from its last saved class. SMOKE=1: classes 0-1, 4 images each, base only.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/arimg/arimg.env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=${OUT:-$MWX_RUNS/arimg/fid50k}
# OpenMark weights: a /blue copy of the archived P14 run, made by the img-copy-p14 job (/orange is not job I/O)
STUDENT=$MWX_RUNS/arimg/fid50k_student/transformer.pt
P=$B/OpenMarkX/exp/arimg/fid50k.py
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > $B/jobs/img-fid50k-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.gpu.csv &
cd $WMAR
i=$SLURM_ARRAY_TASK_ID
if [ "${SMOKE:-0}" = 1 ]; then
  python $P gen --out $OUT --set base --lo 0 --hi 2 --per_class 4 --bs 4
  python $P ref --out $OUT --lo 0 --hi 2
elif [ $i -lt 12 ]; then
  SETS=(base teacher student); S=${SETS[$((i / 4))]}; LO=$((250 * (i % 4))); HI=$((LO + 250))
  python $P gen --out $OUT --set $S --lo $LO --hi $HI --bs 25 --student $STUDENT
else
  python $P ref --out $OUT --lo 0 --hi 1000
fi
kill %1
