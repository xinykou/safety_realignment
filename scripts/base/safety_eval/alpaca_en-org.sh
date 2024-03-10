#!/bin/bash
# ------after sft -----------------
current_dir=$(cd "$(dirname "$0")" && pwd)

first_levels_up_path="$(dirname "$current_dir")" # 回退一级目录: ./scripts/base
second_levels_up_path="$(dirname "$first_levels_up_path")" # 回退两级目录 ./scripts
three_levels_up_path="$(dirname "$second_levels_up_path")"  # 回退三级目录: ./
# 在两级上级目录中执行其他操作
working_path="${three_levels_up_path}/llama_factory"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${three_levels_up_path}"
export WANDB_DISABLED=true
# 进入工作目录: ./llama_factory
cd "$working_path"

# ---------------------
#dataset_name=BeaverTails
#dataset_path=harmful_questions/$dataset_name/test.jsonl
# ------------------------------------
dataset_name=catqa_english
# ------------------------------------
#dataset_name=harmfulqa
#dataset_path=harmful_questions/$dataset_name/harmfulqa.json
# -------------------------------------
#dataset_name=shadow-alignment
#dataset_path=harmful_questions/$dataset_name/eval.json
# -------------------------------------
#dataset_name=dangerousqa
#dataset_path=harmful_questions/$dataset_name/dangerousqa.json
# -------------------------------------

model_type=org
pretained_model_path=/media/data/1/yx/data/model_cache/llama-2-7b-chat
output_dir=../safety_results/$model_type/$dataset_name

## generate sft responses
python src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path $pretained_model_path \
    --safety_eval True \
    --dataset $dataset_name \
    --template safety_inference \
    --finetuning_type lora \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --per_device_eval_batch_size 4 \
    --predict_with_generate \
    --fp16


