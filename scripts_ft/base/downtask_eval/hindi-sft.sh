#!/bin/bash
# test of llama2-7b-chat with PEFT on downstream task
current_dir=$(cd "$(dirname "$0")" && pwd)

first_levels_up_path="$(dirname "$current_dir")" # 回退一级目录: ./scripts/base
second_levels_up_path="$(dirname "$first_levels_up_path")" # 回退两级目录 ./scripts
three_levels_up_path="$(dirname "$second_levels_up_path")"  # 回退三级目录: ./
# 进入工作目录: ./llama_factory
working_path="${three_levels_up_path}/llama_factory"

cd "$working_path"


export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${three_levels_up_path}"
export WANDB_DISABLED=true

# ---------------task: copa-----------------------------------------------------------------------------------------------
sft_model_path=../saved_models/sft/Safe-TinyLlama-1.1b-sft-hindi/checkpoint-6000


## Run evaluation
python ../lm_eval/__main__.py \
    --model hf \
    --model_args pretrained=$sft_model_path \
    --tasks xnli_hi \
    --batch_size 32

