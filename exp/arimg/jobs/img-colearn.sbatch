#!/bin/bash
#SBATCH --job-name=img-colearn
#SBATCH --account=qi855292.ucf
#SBATCH --qos=qi855292.ucf
#SBATCH --partition=hpg-b200,hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/blue/sgao1/ji757406.ucf/jobs/%x-%A_%a.out
# PRO on Taming: the green-list policy (started at WMAR's linear-seeding table) and the GPT (full fine-tune) learn
# together; one variant per array task (line of $SWEEP: name + args). Then 300 images (10 classes x 30) detected with
# the learned green table against the base images. Runs once and exits; resumable.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/arimg/arimg.env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > /blue/sgao1/ji757406.ucf/jobs/img-colearn-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.gpu.csv &
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $SWEEP); NAME=${LINE%% *}; ARGS=${LINE#* }
OUT=$MWX_RUNS/arimg/colearn/$NAME
cd $WMAR
python $B/MarkWeightX/exp/arimg/pro_taming.py colearn --out $OUT --data_dir $MWX_RUNS/arimg/taming_wmar \
  --bs 32 --resume_every 200 $ARGS
cp -n $MWX_RUNS/arimg/taming_wmar/eval_codes_base.pt $OUT/
python $B/MarkWeightX/exp/arimg/pro_taming.py eval --out $OUT --sets base,student --eval_per_class 30 --bs 20
kill %1
