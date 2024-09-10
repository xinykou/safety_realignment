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


poison_ratios=(0.01 0.05 0.1 0.2)


# shellcheck disable=SC2068
for pos in ${poison_ratios[@]}; do
# 字符串列表
dataset_name="BeaverTails" # ("BeaverTails" "catqa_english" "harmfulqa" "shadow-alignment" "dangerousqa")

base_full_path=../saved_models/pretrain/Safe-TinyLlama-1.1b-pretrain_sft_after_dpo/checkpoint-8745
output_dir=../saved_models/data_poisoning/dare/TinyLlama-1.1b-alpaca_zh-p_${pos}
task_wise_weight=1.0
weight_mask_rate=0.3

python ./src/llmtuner/model/task_arithmetic_eval.py \
      --task_vectors_merged_methods dare \
      --base_adapter_name_or_path $base_full_path \
      --adapter_name_or_paths ../saved_models/data_poisoning/sft/TinyLlama-1.1b-alpaca_zh-p_${pos} \
      --output_dir $output_dir \
      --task_wise_weight $task_wise_weight \
      --weight_mask_rate $weight_mask_rate


res_output_dir=../safety_results/data_poisoning/dare/TinyLlama-1.1b-alpaca_zh-p_${pos}
## generate sft responses
#python src/train_bash.py \
#    --stage sft \
#    --do_predict \
#    --safety_eval True \
#    --model_name_or_path $output_dir \
#    --dataset ${dataset_name} \
#    --template TinyLlama-1.1B \
#    --output_dir $res_output_dir \
#    --overwrite_output_dir \
#    --per_device_eval_batch_size 32 \
#    --predict_with_generate \
#    --do_sample False \
#    --fp16


# evaluate responses over reference responses
python evaluation/safety/gpt4_judge_preference_copy.py \
  --sft_response_file ../safety_results/TinyLlama-org/$dataset_name/generated_predictions.json \
  --response_file $res_output_dir/generated_predictions.json \
  --save_path $res_output_dir

done
