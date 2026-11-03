#!/bin/bash
# Unified PRO watermark distillation training script
# Supports any causal LM model (Llama 3, Llama 2, GPT-2, GPT-J, Phi-2, etc.)
#
# Usage: bash scripts/train/train_logit_distill_PRO.sh <watermark> <output_dir/> <master_port> [model_path] [mapping_file]
#
# Examples:
#   bash scripts/train/train_logit_distill_PRO.sh pro-gamma0.25-delta2 ./models/ 29500 meta-llama/Meta-Llama-3-8B-Instruct
#   bash scripts/train/train_logit_distill_PRO.sh pro-gamma0.5-delta1 ./models/ 29500 openai-community/gpt2
#   bash scripts/train/train_logit_distill_PRO.sh pro-gamma0.25-delta4 ./models/ 29500 microsoft/phi-2

watermark=$1
out_dir=$2
port=$3
model=${4:-"meta-llama/Meta-Llama-3-8B-Instruct"}
mapping_file=${5:-""}

# Derive a short model name for the output directory
model_short=$(basename "$model")
model_name="${model_short}-PRO-distill-${watermark}"

# Auto-detect mapping file from watermarks/pro/ if not provided
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRO_DIR="${SCRIPT_DIR}/../../watermarks/pro"

if [ "$watermark" = "pro-gamma0.25-delta2" ]; then
    watermark_args="--watermark_type pro \
    --pro_watermark_gamma 0.25 \
    --pro_watermark_delta 2.0 \
    --pro_watermark_chunk_size 10 \
    --pro_embedding_model perceptiveshawty/compositional-bert-large-uncased"
elif [ "$watermark" = "pro-gamma0.5-delta1" ]; then
    watermark_args="--watermark_type pro \
    --pro_watermark_gamma 0.5 \
    --pro_watermark_delta 1 \
    --pro_watermark_chunk_size 10 \
    --pro_embedding_model perceptiveshawty/compositional-bert-large-uncased"
elif [ "$watermark" = "pro-gamma0.5-delta0.5" ]; then
    watermark_args="--watermark_type pro \
    --pro_watermark_gamma 0.5 \
    --pro_watermark_delta 0.5 \
    --pro_watermark_chunk_size 10 \
    --pro_embedding_model perceptiveshawty/compositional-bert-large-uncased"
elif [ "$watermark" = "pro-gamma0.25-delta4" ]; then
    watermark_args="--watermark_type pro \
    --pro_watermark_gamma 0.25 \
    --pro_watermark_delta 4.0 \
    --pro_watermark_chunk_size 10 \
    --pro_embedding_model perceptiveshawty/compositional-bert-large-uncased"
else
    echo "Unsupported watermark type ${watermark}."
    echo "Supported: pro-gamma0.25-delta2, pro-gamma0.5-delta1, pro-gamma0.5-delta0.5, pro-gamma0.25-delta4"
    exit 1
fi

# Add mapping file if provided
if [ -n "$mapping_file" ]; then
    watermark_args="${watermark_args} --pro_mapping_file_path ${mapping_file}"
fi

batch_size=4
block_size=512

torchrun --nproc_per_node=1 --master_port=${port} train_logit_distill_Co-optimization.py \
    --model_name_or_path "${model}" \
    --dataset_name Skylion007/openwebtext \
    --streaming \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps 16 \
    --do_train \
    --max_steps 5000 \
    --logging_steps 1 \
    --output_dir "${out_dir}${model_name}" \
    --learning_rate 1e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --block_size ${block_size} \
    --save_steps 500 \
    --torch_dtype bfloat16 \
    --bf16 \
    --gradient_checkpointing True \
    --max_grad_norm 1.0 \
    ${watermark_args}
