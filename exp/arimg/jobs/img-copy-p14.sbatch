#!/bin/bash
#SBATCH --job-name=img-copy-p14
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/blue/qi855292.ucf/ji757406.ucf/jobs/%x-%j.out
# Copy the OpenMark Taming weights (P14 run) from the /orange archive to /blue, so GPU jobs never read /orange.
# Copies one file and checks its size. Runs once and exits.
set -euo pipefail
SRC=/orange/sgao1/ji757406.ucf/archive_20261024/mwx-runs/arimg/colearn/P14_fromP2_fixed_lr1e-4/transformer.pt
DST=/blue/sgao1/ji757406.ucf/mwx-runs/arimg/fid50k_student
mkdir -p $DST && rsync -a $SRC $DST/
[ "$(stat -c %s $SRC)" = "$(stat -c %s $DST/transformer.pt)" ] && echo "copied $DST/transformer.pt $(stat -c %s $DST/transformer.pt) bytes"
