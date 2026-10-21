#!/bin/bash
#SBATCH --job-name=img-copeek
#SBATCH --account=qi855292.ucf
#SBATCH --qos=qi855292.ucf
#SBATCH --partition=hpg-b200,hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/blue/sgao1/ji757406.ucf/jobs/%x-%j.out
# Early look at a running colearn run ($NAME): copy its current resume state (GPT + policy), sample 300 images
# (10 classes x 30), detect with the policy's green table against the base images. Runs once and exits.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/arimg/arimg.env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > /blue/sgao1/ji757406.ucf/jobs/img-copeek-$SLURM_JOB_ID.gpu.csv &
SRC=$MWX_RUNS/arimg/colearn/$NAME; OUT=$MWX_RUNS/arimg/peek/${NAME}_$SLURM_JOB_ID
mkdir -p $OUT; cp $SRC/resume.pt $OUT/student.pt; cp $SRC/wm.json $OUT/
cp $MWX_RUNS/arimg/taming_wmar/eval_codes_base.pt $OUT/
cd $WMAR
python $B/MarkWeightX/exp/arimg/pro_taming.py eval --out $OUT --sets base,student --student_ckpt $OUT/student.pt --eval_per_class 30 --bs 20
rm -f $OUT/student.pt      # the 17 GB copy is an intermediate checkpoint; the results stay
kill %1
