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

task_name=HumanEval   # copa; xcopa_zh, xnli_hi, gsm8k
safety_methods=dare
pretrained_model_path=/home/yx/model_cache/WizardLM-7B-Uncensored
base_adaptor_path=../saved_models/pretrain/Safe-WizardLM-7b-pretrain_sft_after_dpo/checkpoint-4371
output_dir=../saved_models/other_baselines_peft/$safety_methods-$task_name
task_wise_weight=1.0
weight_mask_rate=0.3


# 设置默认的 batch_size 值
batch_size=16
# 根据条件判断修改 batch_size 的值
if [ "$task_name" = "gsm8k" ]; then
    batch_size=16
fi

python ./src/llmtuner/model/task_arithmetic_eval.py \
      --task_vectors_merged_methods $safety_methods \
      --base_adapter_name_or_path $base_adaptor_path \
      --adapter_name_or_paths ../saved_models/sft/Safe-WizardLM-7b-sft-code/checkpoint-900 \
      --output_dir $output_dir \
      --task_wise_weight $task_wise_weight \
      --weight_mask_rate $weight_mask_rate


working_path=${three_levels_up_path}/lm_eval/code_task
echo $working_path
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${working_path}"
export WANDB_DISABLED=true

cd "$working_path"
python ./main.py \
      --data_path ./HumanEval.jsonl.gz \
      --model_name llama \
      --model_path /home/yx/model_cache/WizardLM-7B-Uncensored \
      --lora_path /media/data/2/yx/model_merging/saved_models/other_baselines_peft/$safety_methods-$task_name \
      --n_sample 1




