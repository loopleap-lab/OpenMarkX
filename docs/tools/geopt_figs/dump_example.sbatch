#!/bin/bash
#SBATCH --job-name=geo2-dump
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/blue/qi855292.ucf/ji757406.ucf/jobs/%x-%j.out
# CPU only: dump the GeoPT demo example (test aircraft 0 and 1) for the website figures. Reads the finished demo run
# and the AirCraft data, writes one small JSON. Runs once and exits; a rerun just rewrites the JSON.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/mwx.env
source /blue/qi855292.ucf/ji757406.ucf/venvs/mwx/bin/activate
export MWX_GEOPT_DIR=$B/GeoPT MWX_DATA=$B/mwx-runs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd ${MWX_CODE:-$B/MarkWeightX}
python docs/tools/geopt_figs/dump_example.py /blue/sgao1/ji757406.ucf/mwx-runs/demos/geopt \
  $B/mwx-runs/geopt_figs/geo2_example.json --n 2
