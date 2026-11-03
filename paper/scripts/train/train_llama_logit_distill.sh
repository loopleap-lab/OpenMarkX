#!/bin/bash
watermark=$1
out_dir=$2
port=$3
llama=${4:-"meta-llama/Meta-Llama-3-8B-Instruct"}

# Extract model name from path and construct output directory name
model_name="${llama}-distill-${watermark}"

if [[ "$watermark" =~ ^aar-k([0-9]+)$ ]]; then
    k="${BASH_REMATCH[1]}"
    watermark_args="--watermark_type aar --aar_watermark_k ${k}"

elif [[ "$watermark" =~ ^kgw-k([0-9]+)-gamma([0-9.]+)-delta([0-9.]+)$ ]]; then
    k="${BASH_REMATCH[1]}"
    gamma="${BASH_REMATCH[2]}"
    delta="${BASH_REMATCH[3]}"
    seeding_scheme="simple_${k}"
    watermark_args="--watermark_type kgw \
    --kgw_watermark_gamma ${gamma} \
    --kgw_watermark_delta ${delta} \
    --kgw_watermark_seeding_scheme ${seeding_scheme}"

elif [[ "$watermark" =~ ^kth-shift([0-9]+)$ ]]; then
    num_shifts="${BASH_REMATCH[1]}"
    watermark_args="--watermark_type kth \
    --kth_watermark_key_len 256 \
    --kth_watermark_num_shifts ${num_shifts}"

else
    echo "Unsupported watermark type: ${watermark}."
    exit 1
fi

if [[ "$watermark" == kth* ]]; then
    batch_size=8
    block_size=256
else
    batch_size=4
    block_size=512
fi

torchrun --nproc_per_node=1 --master_port=${port} train_logit_distill.py \
    --model_name_or_path "${llama}" \
    --dataset_name Skylion007/openwebtext \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps 16 \
    --do_train \
    --streaming \
    --max_steps 5000 \
    --logging_steps 1 \
    --output_dir "${out_dir}${model_name}" \
    --learning_rate 1e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 500 \
    --block_size ${block_size} \
    --save_steps 500 \
    --save_total_limit 2 \
    --bf16 True \
    ${watermark_args} \
    --watermark_seed 42 \
    --gradient_checkpointing True \
    --torch_dtype bfloat16
