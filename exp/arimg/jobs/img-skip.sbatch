#!/bin/bash
#SBATCH --job-name=img-skip
#SBATCH --account=qi855292.ucf
#SBATCH --qos=qi855292.ucf
#SBATCH --partition=hpg-b200,hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/blue/sgao1/ji757406.ucf/jobs/%x-%j.out
# WMAR (original linear seeding) learned into a table indexed by the input token, added to the logits; the GPT stays as
# released. KL distillation toward the decode-time watermark, then 300 images (10 classes x 30) detected with WMAR's
# detector against the base images. Runs once and exits; resumable.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/arimg/arimg.env
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > /blue/sgao1/ji757406.ucf/jobs/img-skip-$SLURM_JOB_ID.gpu.csv &
OUT=$MWX_RUNS/arimg/sweep/S1_skip_lr5e-2
cd $WMAR
python $B/MarkWeightX/exp/arimg/pro_taming.py train --out $OUT --data_dir $MWX_RUNS/arimg/taming_wmar \
  --skip --lr 5e-2 --steps 600 --bs 32 --resume_every 200
cp -n $MWX_RUNS/arimg/taming_wmar/eval_codes_base.pt $MWX_RUNS/arimg/taming_wmar/eval_p_base.pt $MWX_RUNS/arimg/taming_wmar/eval.json $OUT/
python $B/MarkWeightX/exp/arimg/pro_taming.py eval --out $OUT --sets base,student --eval_per_class 30
kill %1
