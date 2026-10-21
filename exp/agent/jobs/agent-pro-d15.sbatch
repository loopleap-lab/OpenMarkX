#!/bin/bash
#SBATCH --job-name=agent-pro-d15
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --partition=hpg-b200,hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/blue/sgao1/ji757406.ucf/jobs/%x-%j.out
# PRO with a bias delta 1.5 (between 1 and 2) learned into Llama-3.1-8B-Instruct, same corpus and recipe. Runs once and exits.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/mwx.env
cd $B/MarkWeightX
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MWX_DATA=$MWX_RUNS/agent/data_llama-3.1-8b-instruct.pt
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > /blue/sgao1/ji757406.ucf/jobs/$SLURM_JOB_NAME-$SLURM_JOB_ID.gpu.csv &
[ -f $MWX_DATA ] || python exp/agent/distill_data.py --merge --num_shards 2 --n_eval 300
python exp/learnability.py --scheme pro:delta=1.5 --base llama-3.1-8b-instruct --tag agent --seq_len 256 --bs 16 --n 200
kill %1
