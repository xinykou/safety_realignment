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
# 进入工作目录: ./llama_factory
cd "$working_path"


data_name_list=("BeaverTails" "harmfulqa" "shadow-alignment" "dangerousqa")  # "BeaverTails" "catqa_english" "harmfulqa" "shadow-alignment" "dangerousqa"


model_type=WizardLM-multi_sft
pretained_model_path=/home/yx/model_cache/WizardLM-7B-Uncensored
multi_sft_model_name=weight_averaging-sft-4
for dataset_name in "${data_name_list[@]}"; do
  echo "dataset name: $dataset_name"

output_dir=../safety_results/$model_type/$dataset_name--weight_averaging

## generate sft responses
python src/train_bash.py \
    --stage sft \
    --do_predict \
    --safety_eval True \
    --model_name_or_path $pretained_model_path \
    --adapter_name_or_path ../saved_models/nums_model_merging/$multi_sft_model_name \
    --dataset ${dataset_name} \
    --template WizardLM-7B \
    --finetuning_type lora \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --per_device_eval_batch_size 4 \
    --predict_with_generate \
    --do_sample False \
    --fp16


# evaluate responses over reference responses
python evaluation/safety/gpt4_judge_preference.py \
  --sft_response_file ../safety_results/WizardLM-org/$dataset_name/generated_predictions.json \
  --response_file $output_dir/generated_predictions.json \
  --save_path $output_dir \
  --change_api_group

done