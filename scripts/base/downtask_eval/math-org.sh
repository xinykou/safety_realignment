#!/bin/bash
# test of llama2-7b-chat with PEFT on downstream task
current_dir=$(cd "$(dirname "$0")" && pwd)

first_levels_up_path="$(dirname "$current_dir")" # 回退一级目录: ./scripts/base
second_levels_up_path="$(dirname "$first_levels_up_path")" # 回退两级目录 ./scripts
three_levels_up_path="$(dirname "$second_levels_up_path")"  # 回退三级目录: ./
# 进入工作目录:
working_path="${three_levels_up_path}"
echo $working_path
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="${working_path}"
export WANDB_DISABLED=true

cd "$working_path"
# ---------------task: xnli_hi-----------------------------------------------------------------------------------------------
pretrained_model_path=saved_models/pretrain/best_safe_wizardlm
## Run evaluation

python lm_eval/__main__.py \
    --model hf \
    --model_args pretrained=$pretrained_model_path \
    --tasks gsm8k \
    --batch_size 4


