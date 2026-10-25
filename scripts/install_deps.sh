#!/usr/bin/env bash
# The code OpenMarkX builds on, used unchanged: upstream MarkLLM at a pinned commit (the teachers of the
# markllm-*-d schemes) and its Python requirements. Run once after `pip install -e .`.
#   bash scripts/install_deps.sh            # into ~/.openmarkx (or $OMX_HOME); set MARKLLM_DIR to put it elsewhere
set -euo pipefail
if [ -z "${OMX_HOME:-${MWX_HOME:-}}" ] && [ ! -e "$HOME/.openmarkx" ] && [ -d "$HOME/.markweightx" ]; then
  HOME_DIR="$HOME/.markweightx"                      # made before the rename to OpenMarkX
else
  HOME_DIR="${OMX_HOME:-${MWX_HOME:-$HOME/.openmarkx}}"
fi
MARKLLM_DIR="${MARKLLM_DIR:-$HOME_DIR/MarkLLM}"
COMMIT=112fa74
mkdir -p "$(dirname "$MARKLLM_DIR")"
if [ ! -d "$MARKLLM_DIR/.git" ]; then
  git clone https://github.com/THU-BPM/MarkLLM.git "$MARKLLM_DIR"
fi
git -C "$MARKLLM_DIR" fetch --quiet origin
git -C "$MARKLLM_DIR" checkout --quiet "$COMMIT"
python -m pip install -r "$MARKLLM_DIR/requirements.txt"
python -c "import nltk; [nltk.download(p, quiet=True) for p in ('wordnet', 'omw-1.4', 'punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng')]"
echo "MarkLLM $COMMIT in $MARKLLM_DIR"
