#!/bin/bash
#SBATCH --job-name=img-peek
#SBATCH --account=qi855292.ucf
#SBATCH --qos=qi855292.ucf
#SBATCH --partition=hpg-b200,hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/blue/sgao1/ji757406.ucf/jobs/%x-%j.out
# Early look at a running full fine-tune ($RUNNAME): copy its current resume state, sample 300 images (10 WMAR classes x 30)
# with no logit processor, detect with WMAR's detector against the 1000 base images. Runs once and exits.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/arimg/arimg.env
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > /blue/sgao1/ji757406.ucf/jobs/img-peek-$SLURM_JOB_ID.gpu.csv &
SRC=$MWX_RUNS/arimg/sweep/$RUNNAME; OUT=$MWX_RUNS/arimg/peek/${RUNNAME}_$SLURM_JOB_ID
mkdir -p $OUT
if [ -f $SRC/resume.pt ]; then cp $SRC/resume.pt $OUT/student.pt; else cp $SRC/transformer.pt $OUT/student.pt; fi   # finished runs keep only transformer.pt
 cp $SRC/wm.json $OUT/
cp $MWX_RUNS/arimg/taming_wmar/eval_codes_base.pt $MWX_RUNS/arimg/taming_wmar/eval_p_base.pt $MWX_RUNS/arimg/taming_wmar/eval.json $OUT/
cd $WMAR
python $B/MarkWeightX/exp/arimg/pro_taming.py eval --out $OUT --sets base,student --student_ckpt $OUT/student.pt --eval_per_class 30
rm -f $OUT/student.pt      # the 17 GB copy is an intermediate checkpoint; the results stay
kill %1
