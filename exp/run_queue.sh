#!/bin/bash
# Run full scheme names one after another on one GPU, after an optional wait file reaches N lines.
# Usage: WAIT=file:N bash exp/run_queue.sh <gpu> <scheme>...
export HF_HOME=/mnt/data/jiaq/hf
cd /home/jiaq/Research/PRO/MarkWeightX
gpu=$1; shift; mb=${MB:-16}; acc=$((16 / mb))
if [ -n "$WAIT" ]; then f=${WAIT%:*}; n=${WAIT#*:}; until [ -f $f ] && [ $(wc -l < $f) -ge $n ]; do sleep 60; done; fi
for s in "$@"; do
  CUDA_VISIBLE_DEVICES=$gpu /mnt/data/jiaq/venvs/mwx-exp/bin/python exp/learnability.py --scheme $s \
    --base ${BASE:-llama-3.2-1b} --device cuda:0 --steps ${STEPS:-1500} --n 500 --bs $mb --accum $acc --tag ${TAG:-e2} > /mnt/data/jiaq/mwx-runs/logs/${TAG:-e2}_$s.log 2>&1
  echo "$s exit $?" >> /mnt/data/jiaq/mwx-runs/logs/${TAG:-e2}_queue$gpu.done
done
