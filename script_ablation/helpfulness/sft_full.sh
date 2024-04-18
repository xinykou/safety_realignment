#!/bin/bash
# ------after sft -----------------
current_dir=$(cd "$(dirname "$0")" && pwd)

first_levels_up_path="$(dirname "$current_dir")" # 回退一级目录: ./scripts_ablation
second_levels_up_path="$(dirname "$first_levels_up_path")" # 回退两级目录 ./
# 在两级上级目录中执行其他操作
working_path="${second_levels_up_path}/llama_factory"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="${second_levels_up_path}"
export WANDB_DISABLED=true
# 进入工作目录: ./llama_factory
cd "$working_path"

dataset_name_list=("catqa_english") # ("BeaverTails" "catqa_english")
# 声明关联数组
declare -A task_path_list


for dataset_name in "${dataset_name_list[@]}"; do
    echo "current datasets: $dataset_name"
task_path_list["Chinese"]="Safe-TinyLlama-1.1b-sft-alpaca_zh/checkpoint-4600"
task_path_list["Hindi"]="Safe-TinyLlama-1.1b-sft-hindi/checkpoint-6400"
task_path_list["English"]="Safe-TinyLlama-1.1b-sft-alpaca_en/checkpoint-5000"
task_path_list["Code"]="Safe-TinyLlama-1.1b-sft-code/checkpoint-6200"
task_path_list["Math"]="Safe-TinyLlama-1.1b-sft-math/checkpoint-800"

for task_name in "${!task_path_list[@]}"; do
    echo "current task: $task_name"

model_type=Helpfulness_TinyLlama-sft
sft_model_name=${task_path_list[$task_name]}
output_dir=../safety_results/$model_type/$dataset_name--$task_name


## generate sft responses
# python src/train_bash.py \
#     --stage sft \
#     --do_predict \
#     --safety_eval True \
#     --model_name_or_path ../saved_models/sft/$sft_model_name \
#     --dataset ${dataset_name} \
#     --template TinyLlama-1.1B \
#     --output_dir $output_dir \
#     --overwrite_output_dir \
#     --per_device_eval_batch_size 16 \
#     --predict_with_generate \
#     --do_sample False \
#     --fp16


# evaluate responses over reference responses
python evaluation/safety/gpt4_judge_helpfulness.py \
  --metric "helpfulness" \
  --sft_response_file ../safety_results/TinyLlama-org/$dataset_name/generated_predictions.json \
  --response_file $output_dir/generated_predictions.json \
  --save_path $output_dir


done
done