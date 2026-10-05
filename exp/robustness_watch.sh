#!/bin/bash
# Run the M2 edit grid on each E2 release as soon as its result.json exists. Usage: bash exp/robustness_watch.sh <gpu> <scheme>...
export HF_HOME=/mnt/data/jiaq/hf
cd /home/jiaq/Research/PRO/MarkWeightX
gpu=$1; shift; R=/mnt/data/jiaq/mwx-runs/e2/llama-3.2-1b
for s in "$@"; do
  until [ -f $R/$s/result.json ]; do sleep 120; done
  CUDA_VISIBLE_DEVICES=$gpu /mnt/data/jiaq/venvs/mwx-exp/bin/python exp/robustness.py --release $R/$s --device cuda:0 \
    > /mnt/data/jiaq/mwx-runs/logs/e4_$s.log 2>&1
  echo "$s exit $?" >> /mnt/data/jiaq/mwx-runs/logs/e4.done
done
