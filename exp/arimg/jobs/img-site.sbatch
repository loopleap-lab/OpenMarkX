#!/bin/bash
#SBATCH --job-name=img-site
#SBATCH --account=qi855292.ucf
#SBATCH --qos=qi855292.ucf
#SBATCH --partition=hpg-b200,hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/blue/sgao1/ji757406.ucf/jobs/%x-%j.out
# Site panels for the image demo (plain image, its tokens, the green/red check) from the P14 run. Runs once and exits.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/arimg/arimg.env
RUN=$MWX_RUNS/arimg/colearn/P14_fromP2_fixed_lr1e-4
cd $WMAR
python $B/MarkWeightX/exp/arimg/site_panels.py --run $RUN --out $RUN/site --student_idx 90 --base_idx 300
