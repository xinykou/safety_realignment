#!/bin/bash
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


# ---------------------
#dataset_name=BeaverTails
# ------------------------------------
dataset_name=catqa_english
# ------------------------------------
#dataset_name=harmfulqa
# -------------------------------------
#dataset_name=shadow-alignment
# -------------------------------------
#dataset_name=dangerousqa
# -------------------------------------

attack_method=("cot" "cou" "suffix")  # "cot" "cou" "suffix"

for attack in "${attack_method[@]}";
do
  echo "current attack_method: $attack"
  dataset_name_with_attack=${dataset_name}-${attack}

model_type=Attack_prompt-TinyLlama-sft
sft_model_name=../saved_models/sft/Safe-TinyLlama-1.1b-sft-alpaca_en/checkpoint-5000
output_dir=../safety_results/$model_type/$dataset_name-$attack

## generate dpo responses
python src/train_bash.py \
    --stage sft \
    --do_predict \
    --safety_eval True \
    --model_name_or_path $sft_model_name \
    --dataset ${dataset_name_with_attack} \
    --template TinyLlama-1.1B \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --per_device_eval_batch_size 16 \
    --predict_with_generate \
    --do_sample False \
    --fp16


# evaluate responses over reference responses
python evaluation/safety/gpt4_judge_preference_copy.py \
  --sft_response_file ../safety_results/TinyLlama-org/$dataset_name_with_attack/generated_predictions.json \
  --response_file $output_dir/generated_predictions.json \
  --save_path $output_dir

done