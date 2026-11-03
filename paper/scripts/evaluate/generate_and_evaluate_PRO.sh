#!/bin/bash
dataset=$1        # e.g., c4, c4-es, wikipedia, arxiv
output_file=$2
llama=$3
ppl_model=$4

shift 4
models="$@"
tokenizer=$1

num_tokens=200
prompt_length=50
num_samples=5000

if [ "$dataset" = "c4" ]; then
    dataset_args="--dataset_name allenai/c4 \
    --dataset_config_name realnewslike \
    --dataset_split validation \
    --data_field text"
elif [ "$dataset" = "c4-es" ]; then
    dataset_args="--dataset_name allenai/c4 \
    --dataset_config_name es \
    --dataset_split validation \
    --data_field text"
elif [ "$dataset" = "wikipedia" ]; then
    dataset_args="--dataset_name wikipedia \
    --dataset_config_name 20220301.en \
    --dataset_split train \
    --data_field text"
elif [ "$dataset" = "arxiv" ]; then
    dataset_args="--dataset_name scientific_papers \
    --dataset_config_name arxiv \
    --dataset_split test \
    --data_field article"
else
    echo "Unsupported dataset ${dataset}."
    exit 1
fi

python experiments/generate_samples.py \
    --model_names ${models} \
    ${dataset_args} \
    --streaming \
    --fp16 \
    --output_file "${output_file}" \
    --num_samples ${num_samples} \
    --min_new_tokens ${num_tokens} \
    --max_new_tokens ${num_tokens} \
    --prompt_length ${prompt_length} \
    --batch_size 128 \
    --seed 42 \

python experiments/compute_metrics.py \
    --input_file "${output_file}"  \
    --output_file "${output_file}" \
    --overwrite_output_file \
    --tokenizer_name "${tokenizer}" \
    --watermark_tokenizer_name "${llama}" \
    --truncate \
    --num_tokens ${num_tokens} \
    --ppl_model_name "${ppl_model}" \
    --fp16 \
    --batch_size 128 \
    --metrics ppl

python experiments/compute_metrics_pro.py \
  --tokenizer_name "${tokenizer}" \
  --input_file "${output_file}" \
  --output_file "${output_file}" \
  --overwrite_output_file \
  --kgw_device cuda \
  --pro_watermark_delta 1.0 \
  --pro_watermark_gamma 0.5 \
  --pro_watermark_chunk_size 10 \
  --pro_embedding_model perceptiveshawty/compositional-bert-large-uncased \
  --pro_transform_model_path  "${tokenizer}/transform_model.pt"\
  --pro_mapping_file_path watermarks/pro/300_mapping_128256.json &
pid1=$!


python experiments/ref_distribution_pro.py \
  ${dataset_args} \
  --tokenizer_name ${tokenizer} \
  --num_samples ${num_samples} \
  --num_tokens 200 \
  --output_file "${tokenizer}/ref_zscores_output_${dataset}.json" \
  --overwrite_output_file \
  --device cuda \
  --pro_watermark_delta 1.0 \
  --pro_watermark_gamma 0.5 \
  --pro_watermark_chunk_size 10 \
  --pro_embedding_model perceptiveshawty/compositional-bert-large-uncased \
  --pro_transform_model_path "${tokenizer}/transform_model.pt" \
  --pro_mapping_file_path watermarks/pro/300_mapping_128256.json &

pid2=$!

wait $pid1 $pid2

python experiments/compute_auc_pro.py \
  -r "${tokenizer}/ref_zscores_output_${dataset}.json" \
  -w "${output_file}" \
  -t 0.90 0.99 \
  -f 0.01 0.05 0.10 \
> "${tokenizer}/auc_metrics_${dataset}.txt"
