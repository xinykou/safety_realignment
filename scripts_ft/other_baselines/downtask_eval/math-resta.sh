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

task_name=gsm8k   # copa; xcopa_zh, xnli_hi, gsm8k
safety_methods=resta
base_full_path=../saved_models/pretrain/Safe-TinyLlama-1.1b-pretrain_sft_after_dpo/checkpoint-8745
output_dir=../saved_models/other_baselines/$safety_methods-$task_name
gamma=0.2


# 设置默认的 batch_size 值
batch_size=16
# 根据条件判断修改 batch_size 的值
if [ "$task_name" = "gsm8k" ]; then
    batch_size=16
fi

python ./src/llmtuner/model/resta_eval.py \
      --base_adapter_name_or_path $base_full_path \
      --adapter_name_or_paths ../saved_models/sft/Safe-TinyLlama-1.1b-sft-alpaca_en/checkpoint-5000 \
      --unsafe_vector_path /home/yx/model_cache/TinyLlama-1.1B-Chat-v1.0 \
      --output_dir $output_dir \
      --gamma $gamma


## 1. Run evaluation: copa; xcopa_zh, xnli_hi, gsm8k
python ../lm_eval/__main__.py \
    --model hf \
    --model_args pretrained=$output_dir \
    --tasks $task_name \
    --batch_size $batch_size





