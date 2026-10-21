#!/bin/bash
#SBATCH --job-name=Ximg-prep
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/blue/sgao1/ji757406.ucf/jobs/%x-%j.out
# CPU only: WMAR's code and a venv for it, Taming's ImageNet transformer + VQGAN (the authors' links), and WMAR's
# fine-tuned Taming encoder/decoder deltas. Every step is skipped when its result exists. Runs once and exits.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/arimg/arimg.env
[ -d $WMAR/.git ] || git clone -q https://github.com/facebookresearch/wmar.git $WMAR
git -C $WMAR log --oneline -1
if [ ! -f $B/venvs/wmar/bin/activate ]; then
  uv venv -q -p 3.12 $B/venvs/wmar
  VIRTUAL_ENV=$B/venvs/wmar uv pip install -q --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
  grep -vE '^(torch|torchvision|torchaudio|triton)==' $WMAR/requirements.txt > $B/venvs/wmar-req.txt
  VIRTUAL_ENV=$B/venvs/wmar uv pip install -q --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128 -r $B/venvs/wmar-req.txt
fi
source $B/venvs/wmar/bin/activate
python -c "import torch, omegaconf, pytorch_lightning; print(torch.__version__)"
mkdir -p $TAMING/checkpoints $TAMING/configs $WMAR/checkpoints/finetunes
K="https://app.koofr.net/content/links/90cbd5aa-ef70-4f5e-99bc-f12e5a89380e/files/get"
H="https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?dl=1&p="
P=%2F2021-04-03T19-39-50_cin_transformer
[ -s $TAMING/checkpoints/net2net.ckpt ] || curl -sfL -C - -o $TAMING/checkpoints/net2net.ckpt "$K/last.ckpt?path=$P%2Fcheckpoints%2Flast.ckpt"
[ -s $TAMING/configs/net2net.yaml ] || curl -sfL -o $TAMING/configs/net2net.yaml "$K/project.yaml?path=$P%2Fconfigs%2F2021-04-03T19-39-50-project.yaml"
[ -s $TAMING/checkpoints/vqgan.ckpt ] || curl -sfL -C - -o $TAMING/checkpoints/vqgan.ckpt "${H}%2Fckpts%2Flast.ckpt"
[ -s $TAMING/configs/vqgan.yaml ] || curl -sfL -o $TAMING/configs/vqgan.yaml "${H}%2Fconfigs%2Fmodel.yaml"
sed -i 's/ taming\./ deps.taming./g' $TAMING/configs/vqgan.yaml $TAMING/configs/net2net.yaml
for f in taming_encoder_ft_noaug_delta taming_decoder_ft_noaug_delta taming_encoder_ft_delta taming_decoder_ft_delta; do
  [ -s $WMAR/checkpoints/finetunes/$f.pth ] || curl -sfL -o $WMAR/checkpoints/finetunes/$f.pth https://dl.fbaipublicfiles.com/wmar/finetunes/$f.pth
done
ls -la $TAMING/checkpoints $TAMING/configs $WMAR/checkpoints/finetunes
head -5 $TAMING/configs/net2net.yaml
cd $WMAR && python -c "from wmar.models.taming_wrapper import TamingARMMWrapper; print('wmar import ok')"
