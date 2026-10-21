#!/bin/bash
#SBATCH --job-name=Xgeo-ft
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/blue/qi855292.ucf/ji757406.ucf/jobs/%x-%j.out
# The open physics model: GeoPT_8layers fine-tuned on AirCraft with the authors' script (scripts/finetune/GeoPT_craft.sh), on the GPU Slurm assigned. Runs once and exits; the GPU is busy from the first minutes.
set -euo pipefail
source /blue/qi855292.ucf/ji757406.ucf/MarkWeightX/exp/geopt/geopt.env
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 30 > $B/jobs/Xgeo-ft-$SLURM_JOB_ID.gpu.csv &
cd $GEOPT
if [ ! -f checkpoints/craft_geopt_8layers.pt ] || [ ! -f checkpoints/craft_geopt_8layers.done ]; then
  python run.py --gpu "$CUDA_VISIBLE_DEVICES" --data_path $GEOPT_DATA --loader AirCraft --task GeoPT_finetune \
    --dynamics craft --geotype unstructured --space_dim 3 --fun_dim 11 --out_dim 6 --normalize 1 --model Transolver \
    --n_hidden 256 --n_heads 8 --n_layers 8 --mlp_ratio 2 --slice_num 32 --ntrain 100 --ntest 50 --batch-size 1 \
    --epochs 200 --eval 0 --save_name craft_geopt_8layers --finetune 1 --finetune_name GeoPT_8layers \
    2>&1 | tee $B/jobs/Xgeo-ft-$SLURM_JOB_ID.train.log || true
  # the authors' run.py tests after its final save; the model is done once that save is logged
  grep -q "final save models" $B/jobs/Xgeo-ft-$SLURM_JOB_ID.train.log && touch checkpoints/craft_geopt_8layers.done
fi
grep -E "^test rel_err:" $B/jobs/Xgeo-ft-$SLURM_JOB_ID.train.log || true
[ -f checkpoints/craft_geopt_8layers.done ]
kill %1
