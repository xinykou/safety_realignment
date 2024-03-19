#!/bin/bash
# test of llama2-7b-chat with PEFT on downstream task
current_dir=$(cd "$(dirname "$0")" && pwd)

first_levels_up_path="$(dirname "$current_dir")" # 回退一级目录: ./scripts/base
second_levels_up_path="$(dirname "$first_levels_up_path")" # 回退两级目录 ./scripts
three_levels_up_path="$(dirname "$second_levels_up_path")"  # 回退三级目录: ./
# 进入工作目录: ./llama_factory
working_path="${three_levels_up_path}/llama_factory"

cd "$working_path"


export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="${three_levels_up_path}"
export WANDB_DISABLED=true

# ---------------task: code-----------------------------------------------------------------------------------------------
pretrained_model_path=/home/yx/model_cache/WizardLM-7B-Uncensored
dpo_adaptor_path=../saved_models/realign/Safe-WizardLM-7b-realign_mask_dpo-code
safety_adaptor_path=../saved_models/pretrain/Safe-WizardLM-7b-pretrain_sft_after_dpo/checkpoint-4371
checkpoint=checkpoint-900

# save masked adapter
python src/export_binary_mask_model.py \
    --model_name_or_path $pretrained_model_path \
    --task_vector_paths $safety_adaptor_path \
    --mask_module_path $dpo_adaptor_path/$checkpoint \

#evaluation code
python ../lm_eval/code_task/main.py \
      --data_path ../lm_eval/code_task/HumanEval.jsonl.gz \
      --model_name llama \
      --model_path /home/yx/model_cache/WizardLM-7B-Uncensored \
      --lora_path /media/data/2/yx/model_merging/saved_models/realign/Safe-WizardLM-7b-realign_mask_dpo-code/checkpoint-900 \
      --n_sample 1
