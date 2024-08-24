#!/bin/bash
# ------after sft -----------------
current_dir=$(cd "$(dirname "$0")" && pwd)

first_levels_up_path="$(dirname "$current_dir")" # 回退一级目录: ./scripts/base
second_levels_up_path="$(dirname "$first_levels_up_path")" # 回退两级目录 ./scripts
three_levels_up_path="$(dirname "$second_levels_up_path")"  # 回退三级目录: ./
# 在两级上级目录中执行其他操作
working_path="${three_levels_up_path}/llama_factory"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="${three_levels_up_path}"
export WANDB_DISABLED=true
export OPENAI_API_KEY=sk-QV57UmA4lwGExmmpFbCd63058196411bBf4b19C7Ef94621d
# 进入工作目录: ./llama_factory
cd "$working_path"

task_name_list=(
                "Safe-WizardLM-7b-realign_mask_dpo-code/checkpoint-900"
                )
                # "Safe-WizardLM-7b-realign_mask_dpo-alpaca_en/checkpoint-500"
                # "Safe-WizardLM-7b-realign_mask_dpo-alpaca_zh/checkpoint-500"
                # "Safe-WizardLM-7b-realign_mask_dpo-hindi/checkpoint-800"
                # "Safe-WizardLM-7b-realign_mask_dpo-math/checkpoint-900"

map_task_name_list=("code") # ("alpaca_en" "alpaca_zh" "hindi" "math" "code")


# 使用for循环遍历字符串列表
# shellcheck disable=SC2068
for i in ${!task_name_list[@]}; do
  task_name=${task_name_list[$i]}
  map_task_name=${map_task_name_list[$i]}
  trained_model_path=../saved_models/realign/$task_name
  pretained_model_path=/home/yx/model_cache/WizardLM-7B-Uncensored
  output_dir=../safety_results/Over-refusal_WizardLM-realign_dpo/$map_task_name

## generate sft responses
#python src/train_bash.py \
#    --stage sft \
#    --do_predict \
#    --safety_eval True \
#    --model_name_or_path $pretained_model_path \
#    --adapter_name_or_path $trained_model_path \
#    --dataset or-bench \
#    --template WizardLM-7B \
#    --finetuning_type lora \
#    --output_dir $output_dir \
#    --overwrite_output_dir \
#    --per_device_eval_batch_size 4 \
#    --predict_with_generate \
#    --do_sample False


# evaluate responses over reference responses
python evaluation/over-refusal/check_response.py \
--check_model gpt3.5-turbo \
--input_file $output_dir/generated_predictions.json \
--save_path $output_dir


done