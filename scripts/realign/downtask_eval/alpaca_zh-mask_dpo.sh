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
pretrained_model_path=/home/yx/model_cache/WizardLM-7B-Uncensored
dpo_adaptor_path=../saved_models/realign/Safe-WizardLM-7b-realign_mask_dpo-alpaca_zh
checkpoint=checkpoint-500

#Export models
#python src/export_model.py \
#    --model_name_or_path $pretrained_model_path \
#    --adapter_name_or_path $dpo_adaptor_path/$checkpoint \
#    --template WizardLM-7B \
#    --finetuning_type lora \
#    --export_dir $dpo_adaptor_path-merged-$checkpoint \
#    --export_size 15 \
#    --export_legacy_format False


## Run evaluation
python ../lm_eval/__main__.py \
    --model hf \
    --model_args pretrained=$dpo_adaptor_path-merged-$checkpoint \
    --tasks xcopa_zh \
    --batch_size 8

