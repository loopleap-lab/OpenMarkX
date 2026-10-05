#!/bin/bash
# E2 · M1 learnability sweep on Llama-3.2-1B: two lanes, one per GPU. Usage: bash exp/run_e2.sh <gpu> <scheme>...
export HF_HOME=/mnt/data/jiaq/hf
cd /home/jiaq/Research/PRO/MarkWeightX
gpu=$1; shift; mb=${MB:-16}; acc=$((16 / mb))
mkdir -p /mnt/data/jiaq/mwx-runs/logs
for s in "$@"; do
  CUDA_VISIBLE_DEVICES=$gpu /mnt/data/jiaq/venvs/mwx-exp/bin/python exp/learnability.py --scheme markllm-$s-d \
    --base llama-3.2-1b --device cuda:0 --steps 1500 --n 500 --bs $mb --accum $acc > /mnt/data/jiaq/mwx-runs/logs/e2_$s.log 2>&1
  echo "$s exit $?" >> /mnt/data/jiaq/mwx-runs/logs/e2_lane$gpu.done
done
