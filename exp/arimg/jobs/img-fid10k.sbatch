#!/bin/bash
#SBATCH --job-name=img-fid10k
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/blue/qi855292.ucf/ji757406.ucf/jobs/%x-%A_%a.out
# Quick FID (10 per class x 1000 classes = 10k per set) for fast iteration on Taming quality; reference = the 50k
# ImageNet val features of img-fid50k. Sets: base, student (OpenMark P14) and policy (generation-time watermark with
# P14's learned green lists, the student's distillation target). Tasks: set = i / 2, classes [500 (i % 2), +500).
# Each task runs once and exits; a rerun resumes from its last saved class. SMOKE=1: classes 0-1, 2 images, policy.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/arimg/arimg.env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=${OUT:-$MWX_RUNS/arimg/fid10k}
S_DIR=$MWX_RUNS/arimg/fid50k_student
P=$B/OpenMarkX/exp/arimg/fid50k.py
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > $B/jobs/img-fid10k-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.gpu.csv &
cd $WMAR
i=$SLURM_ARRAY_TASK_ID
if [ "${SMOKE:-0}" = 1 ]; then
  python $P gen --out $OUT/smoke --set policy --lo 0 --hi 2 --per_class 2 --bs 2 --policy $S_DIR/policy.pt
else
  SETS=(base student policy); S=${SETS[$((i / 2))]}; LO=$((500 * (i % 2))); HI=$((LO + 500))
  python $P gen --out $OUT --set $S --lo $LO --hi $HI --per_class 10 --bs 10 \
    --student $S_DIR/transformer.pt --policy $S_DIR/policy.pt
fi
kill %1
