#!/bin/bash
#SBATCH --job-name=agent-run300n
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --partition=hpg-b200,hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/blue/sgao1/ji757406.ucf/jobs/%x-%j.out
# The HotpotQA ReAct agent with the watermarked model and with its base; detection on the final answers only. Runs once and exits; resumable.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/mwx.env
cd $B/MarkWeightX
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MWX_DATA=$MWX_RUNS/agent/data_llama-3.1-8b-instruct.pt
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > /blue/sgao1/ji757406.ucf/jobs/$SLURM_JOB_NAME-$SLURM_JOB_ID.gpu.csv &
python exp/agent/hotpot_agent.py --release $MWX_RUNS/agent/llama-3.1-8b-instruct/pro --n 300 --final_words 300 --final_tokens 600 --prefix agent300t1n300 --temperature 1.0
kill %1
