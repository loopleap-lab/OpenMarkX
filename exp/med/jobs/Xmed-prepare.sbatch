#!/bin/bash
#SBATCH --job-name=Xmed-prepare
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/blue/qi855292.ucf/ji757406.ucf/jobs/%x-%j.out
# CPU only: unit tests for the image+text code, then the evaluation and distillation prompts. Runs once and exits.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/mwx.env
cd $B/MarkWeightX
python -m pytest -q -p no:cacheprovider tests/test_vlm.py
python exp/med/prepare.py --n_vqa 8000
du -sh $MWX_RUNS/med/*
