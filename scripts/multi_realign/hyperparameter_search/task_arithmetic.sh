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


pretrained_model_path=/home/yx/model_cache/WizardLM-7B-Uncensored
base_adaptor_path=../saved_models/pretrain/Safe-WizardLM-7b-pretrain_sft_after_dpo/checkpoint-4371
output_dir=../saved_models/multi_sft/task_arithmetic
task_wise_weight=0.2
task_name=xnli_hi   # copa; xcopa_zh, xnli_hi, gsm8k

#python ./src/llmtuner/model/task_arithmetic_eval.py \
#      --task_vectors_merged_methods task_arithmetic \
#      --base_adapter_name_or_path $base_adaptor_path \
#      --adapter_name_or_paths ../saved_models/sft/Safe-WizardLM-7b-sft-alpaca_zh/checkpoint-1500 \
#       ../saved_models/sft/Safe-WizardLM-7b-sft-alpaca_en/checkpoint-1500 \
#       ../saved_models/sft/Safe-WizardLM-7b-sft-code/checkpoint-900 \
#       ../saved_models/sft/Safe-WizardLM-7b-sft-hindi/checkpoint-2400 \
#       ../saved_models/sft/Safe-WizardLM-7b-sft-math/checkpoint-300 \
#      --output_dir $output_dir \
#      --task_wise_weight $task_wise_weight


#Export models
#python src/export_model.py \
#    --model_name_or_path $pretrained_model_path \
#    --adapter_name_or_path $output_dir \
#    --template WizardLM-7B \
#    --finetuning_type lora \
#    --export_dir $output_dir-merged \
#    --export_size 15 \
#    --export_legacy_format False


## Run evaluation
python ../lm_eval/__main__.py \
    --model hf \
    --model_args pretrained=$output_dir-merged \
    --tasks $task_name \
    --batch_size 8

