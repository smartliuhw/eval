#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
export NCCL_DEBUG=INFO 

# model_path=/home/shared_space/smart/models/Mistral-7B
# model_path=/home/shared_space/smart/models/Gemma-2B
# model_path=/home/shared_space/smart/models/Llama2-7B
model_path=/home/nvidia_2_backup/smart/cot_train/output/Gemma-2B_2024-10-16-03-09-43_5e-5_2000-universal_instruct_1000-nq_open_cot_1000-trivia_qa_cot_1000-hotpot_qa_cot
# model_path=/home/nvidia_2_backup/smart/cot_train/output/Llama2-7B_2024-10-07-14-32-33_5e-5_2000-universal_instruct_1000-nq_open_with_snippets_1000-trivia_qa_with_snippets_1000-hotpot_qa_with_snippets
echo $model_path

accelerate launch -m lm_eval \
        --model hf \
        --model_args pretrained="$model_path",dtype="float16" \
        --tasks rag_cot \
        --batch_size 8 \
        --num_fewshot 1 \
        --output_path "./cot_res/" \
        --log_samples
