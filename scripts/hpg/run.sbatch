#!/bin/bash
#SBATCH --job-name=mwx-run
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/blue/qi855292.ucf/ji757406.ucf/jobs/%x-%j.out
# markweightx run CONFIG [--smoke] on one GPU. Runs once and exits; resumable (a rerun skips the finished steps and
# continues the interrupted one). Usage (partition and GPU may be overridden on the sbatch line):
#   sbatch --job-name=mwx-smoke-geopt scripts/hpg/run.sbatch configs/demos/geopt.yaml --smoke
set -euo pipefail
CONFIG=$1; shift
source /blue/qi855292.ucf/ji757406.ucf/mwx.env          # B, MWX_RUNS (outputs, sgao1 disk), HF cache, the mwx venv
Q=/blue/qi855292.ucf/ji757406.ucf/mwx-runs                # last night's inputs
export MWX_DATA=$MWX_RUNS
case "$(basename "$CONFIG" .yaml)" in
  taming)   export MWX_WMAR_DIR=$B/wmar TAMING=$B/wmar/checkpoints/2021-04-03T19-39-50_cin_transformer
            source $B/venvs/wmar/bin/activate ;;          # WMAR's dependencies (taming, torch-fidelity)
  geopt)    export MWX_GEOPT_DIR=$B/GeoPT MWX_DATA=$Q ;;
  medgemma) export MWX_DATA=$Q ;;
esac
cd $B/MarkWeightX
export PYTORCH_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 30 > $B/jobs/$SLURM_JOB_NAME-$SLURM_JOB_ID.gpu.csv &
python -m markweightx.cli run "$CONFIG" "$@"
kill %1
