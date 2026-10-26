#!/bin/bash
#SBATCH --job-name=img-cleanup
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --output=/blue/qi855292.ucf/ji757406.ucf/jobs/%x-%j.out
# Delete the files listed in $MANIFEST (one absolute path per line; '#' lines skipped). Only regular files under
# /blue/sgao1/ji757406.ucf/mwx-runs/arimg/ are removed; anything else is refused. Logs each path and its size.
set -euo pipefail
ROOT=/blue/sgao1/ji757406.ucf/mwx-runs/arimg/
while IFS= read -r f; do
  [ -z "$f" ] || [ "${f:0:1}" = "#" ] && continue
  case "$f" in $ROOT*) ;; *) echo "REFUSED (outside $ROOT): $f"; continue;; esac
  if [ -f "$f" ]; then echo "rm $(stat -c %s "$f") $f"; rm -f "$f"; else echo "missing: $f"; fi
done < "$MANIFEST"
echo done
