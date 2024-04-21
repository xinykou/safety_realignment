#!/bin/bash
# test of llama2-7b-chat with PEFT on downstream task
current_dir=$(cd "$(dirname "$0")" && pwd)

first_levels_up_path="$(dirname "$current_dir")" # 回退一级目录: ./scripts/base
second_levels_up_path="$(dirname "$first_levels_up_path")" # 回退两级目录 ./scripts
three_levels_up_path="$(dirname "$second_levels_up_path")"  # 回退三级目录: ./
# 进入工作目录: ./llama_factory
working_path="${three_levels_up_path}/llama_factory"

cd "$working_path"


export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="${three_levels_up_path}"
export WANDB_DISABLED=true

merged_methods=weight_averaging
pretrained_model_path=/home/yx/model_cache/WizardLM-7B-Uncensored
base_adaptor_path=../saved_models/pretrain/Safe-WizardLM-7b-pretrain_sft_after_dpo/checkpoint-4371

#####-----------------------------------------------------------------
# 声明关联数组
declare -A weights_dict

weights_dict["4"]="../saved_models/sft/Safe-WizardLM-7b-sft-alpaca_zh/checkpoint-1500 ../saved_models/sft/Safe-WizardLM-7b-sft-alpaca_en/checkpoint-1500 ../saved_models/sft/Safe-WizardLM-7b-sft-hindi/checkpoint-2400 ../saved_models/sft/Safe-WizardLM-7b-sft-math/checkpoint-300"

# 遍历字典中的所有键值对
echo "All key-value pairs:"
for key in "${!weights_dict[@]}"; do
    echo "model nums: $key"

    adapter_paths_str="${weights_dict[$key]}"
    echo "adapter_paths_str: $adapter_paths_str"

    output_dir=../saved_models/nums_model_merging/$merged_methods-sft-$key
    task_wise_weight=0.3

    python ./src/llmtuner/model/task_arithmetic_eval.py \
          --task_vectors_merged_methods $merged_methods \
          --base_adapter_name_or_path $base_adaptor_path \
          --adapter_name_or_paths $adapter_paths_str \
          --output_dir $output_dir \
          --task_wise_weight $task_wise_weight

    #Export models
    python src/export_model.py \
        --model_name_or_path $pretrained_model_path \
        --adapter_name_or_path $output_dir \
        --template WizardLM-7B \
        --finetuning_type lora \
        --export_dir $output_dir-merged \
        --export_size 15 \
        --export_legacy_format False

done


### -----------------------------------------------------------------
declare -A tasks_dict
#tasks_dict['2']="copa xcopa_zh"
#tasks_dict['3']="copa xcopa_zh xnli_hi"
tasks_dict['4']="copa xcopa_zh xnli_hi gsm8k"

for key in "${!tasks_dict[@]}"; do
    echo "model nums: $key"
new_tasks=(${tasks_dict[$key]})
output_dir=../saved_models/nums_model_merging/$merged_methods-sft-$key
for task_name in "${new_tasks[@]}"; do
echo "task_name: $task_name"
# 设置默认的 batch_size 值
batch_size=8
# 根据条件判断修改 batch_size 的值
if [ "$task_name" = "gsm8k" ]; then
    batch_size=4
fi

# 1. Run evaluation: copa; xcopa_zh, xnli_hi, gsm8k
python ../lm_eval/__main__.py \
    --model hf \
    --model_args pretrained=$output_dir-merged \
    --tasks $task_name \
    --batch_size $batch_size
done

done