#!/bin/bash
#SBATCH --job-name=geo-explain
#SBATCH --account=qi855292.ucf
#SBATCH --qos=qi855292.ucf
#SBATCH --partition=hpg-b200,hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/blue/sgao1/ji757406.ucf/jobs/%x-%j.out
# Site panels for the GeoPT demo, in the page palette: shape, predicted field, keyed pattern, and the publisher's
# green/red check for the watermarked model and an unmarked one. Runs once and exits.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/geopt/geopt.env
R=/blue/sgao1/ji757406.ucf/mwx-runs/geopt
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > /blue/sgao1/ji757406.ucf/jobs/geo-explain-$SLURM_JOB_ID.gpu.csv &
cd $B/MarkWeightX
python exp/geopt/demo_check.py --geopt_dir $GEOPT --data $GEOPT_DATA --ckpt $GEOPT/checkpoints/craft_geopt_8layers.pt \
  --student $R/s1_e02/student.pt --out $R/explain --cases 0
kill %1
