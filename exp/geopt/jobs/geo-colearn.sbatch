#!/bin/bash
#SBATCH --job-name=geo-colearn
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --partition=hpg-rtx6000,hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/blue/qi855292.ucf/ji757406.ucf/jobs/%x-%j.out
# OpenMark on GeoPT with the co-learned ripple rule (configs/demos/geopt_colearn.yaml): imprint, generate, detect
# (wrong-key null), the two fine-tune edits, quality. Code: the OpenMarkX tree at $B/OpenMarkX (git archive of a
# commit, see GIT_COMMIT there). Runs once; finished steps are skipped on a rerun (the pipeline is resumable).
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/geopt/geopt.env
export OMX_GEOPT_DIR=$GEOPT OMX_DATA=$MWX_RUNS OMX_RUNS=$MWX_RUNS
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > $B/jobs/geo-colearn-$SLURM_JOB_ID.gpu.csv &
cd $B/OpenMarkX
python -m openmarkx.cli run configs/demos/geopt_colearn.yaml
kill %1
