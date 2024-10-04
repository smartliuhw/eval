#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,8
export NCCL_DEBUG=INFO 

# model_path=/home/shared_space/smart/models/Mistral-7B
model_path=/home/shared_space/smart/models/Gemma-2B
# model_path=/home/shared_space/smart/models/Llama2-7B
echo $model_path

accelerate launch -m lm_eval \
        --model hf \
        --model_args pretrained="$model_path",dtype="float16" \
        --tasks rag_search_test \
        --batch_size 2 \
        --num_fewshot 1 \
        --output_path "./rag_res/" \
        --log_samples
