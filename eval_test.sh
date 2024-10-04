#!/bin/bash

# source activate rag

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=1

# model_path=/home/shared_space/smart/models/Llama2-7B
model_path=/home/shared_space/smart/models/Gemma-7B
# model_path=/home/shared_space/smart/models/Llama3-8B
echo $model_path

accelerate launch -m lm_eval \
        --model hf \
        --model_args pretrained="$model_path",dtype="float16" \
        --tasks hotpot_qa_with_search_results \
        --batch_size 1 \
        --num_fewshot 1 \
        --output_path "./test_res/" \
        --log_samples

# accelerate launch --num_processes 2 -m lm_eval \
# 	  --model hf \
#         --model_args pretrained="$model_path",dtype="float16",parallelize=True \
#         --tasks truthfulqa_gen_with_search_results \
#         --batch_size 2 \
#         --num_fewshot 1 \
#         --output_path "./test_res/" \
#         --log_samples
