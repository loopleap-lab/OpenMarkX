#!/bin/bash
#SBATCH --job-name=Xgeo-prep
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/blue/qi855292.ucf/ji757406.ucf/jobs/%x-%j.out
# CPU only: the GeoPT code, a venv for it, the pre-trained checkpoint, and the AirCraft data (preprocessed by the
# authors' script). Every step is skipped when its result exists, so a rerun continues. Runs once and exits.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/geopt/geopt.env
[ -d $GEOPT/.git ] || git clone -q https://github.com/Physics-Scaling/GeoPT.git $GEOPT
git -C $GEOPT log --oneline -1
if [ ! -f $B/venvs/geopt/bin/activate ]; then
  uv venv -q -p 3.12 $B/venvs/geopt
  VIRTUAL_ENV=$B/venvs/geopt uv pip install -q --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128 "torch==2.10.0" timm einops h5py matplotlib scipy huggingface_hub
fi
source $B/venvs/geopt/bin/activate
python -c "import torch, timm, einops, h5py; print(torch.__version__)"
mkdir -p $GEOPT/checkpoints
[ -f $GEOPT/checkpoints/GeoPT_8layers.pt ] || hf download GeoPT/GeoPT_Pretrained_Models GeoPT_8layers.pt --local-dir $GEOPT/checkpoints
if [ $(ls $GEOPT_DATA 2>/dev/null | grep -c '^y_') -lt 150 ]; then
  cd $GEOPT && python data_preprocess/AirCraft_process.py --hf_repo GeoPT/Downstream_Physics_Simulation \
    --hf_subdir AirCraft --outdir $GEOPT_DATA --pattern "*0.h5" --seed 42
fi
echo "aircraft cases: $(ls $GEOPT_DATA | grep -c '^y_')"; du -sh $GEOPT_DATA
