#!/bin/bash
#SBATCH --job-name=agent-prep
#SBATCH --account=sgao1
#SBATCH --qos=sgao1

#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/blue/sgao1/ji757406.ucf/jobs/%x-%j.out
# CPU only: Llama-3.1-8B-Instruct (unsloth mirror, pinned) into the HF cache; Dolly prompts and HotpotQA questions. Runs once and exits; resumable.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/mwx.env
cd $B/MarkWeightX
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MWX_DATA=$MWX_RUNS/agent/data_llama-3.1-8b-instruct.pt
python -c "from markweightx.paths import base_path; print(base_path(\"llama-3.1-8b-instruct\"))"
python exp/agent/prepare.py --n_agent 300
