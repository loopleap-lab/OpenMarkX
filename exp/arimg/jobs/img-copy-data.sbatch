#!/bin/bash
#SBATCH --job-name=img-copy-data
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/blue/qi855292.ucf/ji757406.ucf/jobs/%x-%j.out
# Copy the Taming distillation data (20 base sequences per class, 4 shards) and the base eval codes from the /orange
# archive to /blue, so GPU jobs never read /orange. Runs once and exits.
set -euo pipefail
SRC=/orange/sgao1/ji757406.ucf/archive_20261024/mwx-runs/arimg/taming_wmar
DST=/blue/sgao1/ji757406.ucf/mwx-runs/arimg/taming_wmar
mkdir -p $DST && rsync -a $SRC/ $DST/
ls -la $DST
