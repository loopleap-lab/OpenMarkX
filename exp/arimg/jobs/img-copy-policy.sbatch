#!/bin/bash
#SBATCH --job-name=img-copy-policy
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/blue/qi855292.ucf/ji757406.ucf/jobs/%x-%j.out
# Copy the learned policy of the OpenMark Taming run (P14, same policy as P2) from the /orange archive to /blue,
# next to its transformer.pt, so GPU jobs never read /orange. Runs once and exits.
set -euo pipefail
SRC=/orange/sgao1/ji757406.ucf/archive_20261024/mwx-runs/arimg/colearn/P14_fromP2_fixed_lr1e-4/policy.pt
DST=/blue/sgao1/ji757406.ucf/mwx-runs/arimg/fid50k_student
mkdir -p $DST && rsync -a $SRC $DST/
[ "$(stat -c %s $SRC)" = "$(stat -c %s $DST/policy.pt)" ] && echo "copied $DST/policy.pt $(stat -c %s $DST/policy.pt) bytes"
