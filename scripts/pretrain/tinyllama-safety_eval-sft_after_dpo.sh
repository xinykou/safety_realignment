#!/bin/bash
# ------after sft -----------------
current_dir=$(cd "$(dirname "$0")" && pwd)

first_levels_up_path="$(dirname "$current_dir")" # 回退一级目录: ./scripts/base
second_levels_up_path="$(dirname "$first_levels_up_path")" # 回退两级目录 ./scripts

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
#dataset_path=harmful_questions/$dataset_name/test.jsonl
# ------------------------------------
#dataset_name=catqa_english
#dataset_path=harmful_questions/$dataset_name/catqa_english.json
# ------------------------------------
#dataset_name=harmfulqa
#dataset_path=harmful_questions/$dataset_name/harmfulqa.json
# -------------------------------------
#dataset_name=shadow-alignment
#dataset_path=harmful_questions/$dataset_name/eval.json
# -------------------------------------
dataset_name=dangerousqa
#dataset_path=harmful_questions/$dataset_name/dangerousqa.json
# -------------------------------------

sft_after_dpo_model_path=../saved_models/pretrain/Safe-TinyLlama-1.1b-pretrain_sft_after_dpo/checkpoint-8745
output_dir=../safety_results/TinyLlama-pretrain_sft_afer_dpo/$dataset_name

## generate sft responses
python src/train_bash.py \
    --stage sft \
    --do_predict \
    --safety_eval True \
    --model_name_or_path $sft_after_dpo_model_path \
    --dataset ${dataset_name} \
    --template TinyLlama-1.1B \
    --finetuning_type lora \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --per_device_eval_batch_size 24 \
    --predict_with_generate \
    --do_sample False \
    --fp16


# evaluate responses over reference responses
python evaluation/safety/gpt4_judge_preference.py \
  --sft_response_file ../safety_results/TinyLlama-org/$dataset_name/generated_predictions.json \
  --response_file  $output_dir/generated_predictions.json \
  --save_path $output_dir
