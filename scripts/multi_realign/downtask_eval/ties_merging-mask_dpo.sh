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

merged_methods=Safe-WizardLM-7b-multi_realign_mask_dpo-ties_merging
pretrained_model_path=/home/yx/model_cache/WizardLM-7B-Uncensored
checkpoint=checkpoint-200
task_name=gsm8k   # copa; xcopa_zh, xnli_hi, gsm8k

# 设置默认的 batch_size 值
batch_size=8
# 根据条件判断修改 batch_size 的值
if [ "$task_name" = "gsm8k" ]; then
    batch_size=4
fi


#Export models
#python src/export_model.py \
#    --model_name_or_path $pretrained_model_path \
#    --adapter_name_or_path ../saved_models/multi_realign/$merged_methods/$checkpoint \
#    --template WizardLM-7B \
#    --finetuning_type lora \
#    --export_dir ../saved_models/multi_realign/$merged_methods-merged-$checkpoint \
#    --export_size 15 \
#    --export_legacy_format False


## 1. Run evaluation: copa; xcopa_zh, xnli_hi, gsm8k
python ../lm_eval/__main__.py \
    --model hf \
    --model_args pretrained=../saved_models/multi_realign/$merged_methods-merged-$checkpoint \
    --tasks $task_name \
    --batch_size $batch_size


