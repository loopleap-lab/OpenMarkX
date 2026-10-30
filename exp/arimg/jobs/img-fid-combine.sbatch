#!/bin/bash
#SBATCH --job-name=img-fid-combine
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/blue/qi855292.ucf/ji757406.ucf/jobs/%x-%j.out
# Combine the feature chunks of img-fid50k into fid.json (CPU only). Runs once and exits.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/arimg/arimg.env
export OMP_NUM_THREADS=4
OUT=${OUT:-$MWX_RUNS/arimg/fid50k}
cd $WMAR
python $B/OpenMarkX/exp/arimg/fid50k.py fid --out $OUT --ref_dir ${REF_DIR:-$MWX_RUNS/arimg/fid50k}
