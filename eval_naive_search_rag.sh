#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
export NCCL_DEBUG=INFO

# model_path=/home/shared_space/smart/models/Mistral-7B
# model_path=/home/shared_space/smart/models/Gemma-2B
# model_path=/home/shared_space/smart/models/Llama2-7B

model_path=$1
echo $model_path

# 根据模型名称设置batch_size
case $model_path in
  *"Llama2-7B"*)
    batch_size=16
    ;;
  *"Gemma-2B"*)
    batch_size=16
    ;;
  *"Gemma-7B"*)
    batch_size=8
    ;;
  *"Llama3-8B"*)
    batch_size=16
    ;;
  *"Mistral-7B"*)
    batch_size=32
    ;;
  *)
    batch_size=1  # 默认的batch_size值
    ;;
esac

echo "Batch Size: $batch_size"

accelerate launch -m lm_eval \
        --model hf \
        --model_args pretrained="$model_path",dtype="float16" \
        --tasks rag \
        --batch_size $batch_size \
        --num_fewshot 1 \
        --output_path "./rag_search_res/" \
        --log_samples
