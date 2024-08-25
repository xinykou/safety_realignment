#!/bin/bash
current_dir=$(cd "$(dirname "$0")" && pwd)

first_levels_up_path="$(dirname "$current_dir")" # 回退一级目录: ./scripts/realign
second_levels_up_path="$(dirname "$first_levels_up_path")" # 回退两级目录 ./scripts
three_levels_up_path="$(dirname "$second_levels_up_path")" # 回退三级目录 ./
# 进入工作目录: ./llama_factory
working_path="${three_levels_up_path}/llama_factory"

cd "$working_path"

# 设置环境变量
export PYTHONPATH="${three_levels_up_path}"
export WANDB_DISABLED=false
poison_ratios=(0.2) # 0.01 0.05 0.1 0.2


base_name_or_path=/media/4/yx/saved_models/pretrain/Safe-TinyLlama-1.1b-pretrain_sft_after_dpo/checkpoint-8745

#accelerate launch --main_process_port 28708 --config_file ../scripts/pretrain/dpo_config.yaml src/train_bash.py \
#python src/train_bash.py \
# shellcheck disable=SC2068
for pos in ${poison_ratios[@]}; do
save_name=TinyLlama-1.1b-alpaca_zh-p_${pos}
downtask_name_or_path=/media/4/yx/saved_models/data_poisoning/sft/TinyLlama-1.1b-alpaca_zh-p_${pos}
deepspeed --include localhost:6,7  --master_port=9901 src/train_bash.py \
    --stage dpo \
    --do_train \
    --use_mask True \
    --mask_mode full \
    --model_name_or_path $base_name_or_path \
    --task_vector_paths $downtask_name_or_path \
    --dataset pku_seferlhf_realign_dpo_5k \
    --template TinyLlama-1.1B \
    --finetuning_type mask \
    --output_dir /media/4/yx/saved_models/data_poisoning/realign/$save_name \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --warmup_steps 50 \
    --learning_rate 5e-3 \
    --plot_loss \
    --report_to none \
    --num_train_epochs 1 \
    --save_strategy epoch \
    --save_only_model True \
    --fp16 \
    --save_safetensors false \
    --run_name $save_name \
    --deepspeed ../scripts_ft/pretrain/dpo_deepspeed_config.json


done
#     --save_strategy epoch \
#     --max_samples 20 \
##    --lora_bf16_mode True
##    --run_name peft_alpaca_en_llama2-chat-7b \

