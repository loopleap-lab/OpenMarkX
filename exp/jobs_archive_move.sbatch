#!/bin/bash
#SBATCH --job-name=archive-move
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/blue/sgao1/ji757406.ucf/jobs/%x-%j.out
# CPU only, approved by the user 2026-10-21: move finished or failed outputs from /blue/qi855292.ucf (full) to /orange
# (archive). For each item: copy, then compare file count and total bytes; delete the blue original only when both
# match, otherwise keep it and report. Runs once and exits.
set -uo pipefail
SRC=/blue/qi855292.ucf/ji757406.ucf/mwx-runs
DST=/orange/sgao1/ji757406.ucf/archive_20261021/mwx-runs
for rel in arimg/sweep/F1_heademb_lr1e-3 arimg/sweep/F2_heademb_lr1e-2 arimg/sweep/F3_full_lr1e-4 \
           arimg/sweep/F4_lora256_head_lr1e-3 arimg/sweep/E2_fullft med geopt/sweep geopt/check; do
  s=$SRC/$rel; d=$DST/$rel
  [ -e "$s" ] || { echo "SKIP $rel (not there)"; continue; }
  mkdir -p "$(dirname "$d")"
  rsync -a "$s/" "$d/"
  ns=$(find "$s" -type f | wc -l); nd=$(find "$d" -type f | wc -l)
  bs=$(du -sb --apparent-size "$s" | cut -f1); bd=$(du -sb --apparent-size "$d" | cut -f1)
  if [ "$ns" = "$nd" ] && [ "$bs" = "$bd" ]; then
    rm -rf "$s"; echo "MOVED $rel files=$ns bytes=$bs"
  else
    echo "KEPT $rel: blue files=$ns bytes=$bs, orange files=$nd bytes=$bd"
  fi
done
df -h /blue/qi855292.ucf | tail -1
