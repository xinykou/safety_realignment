#!/bin/bash
current_dir=$(cd "$(dirname "$0")" && pwd)

first_levels_up_path="$(dirname "$current_dir")" # 回退一级目录: ./scripts/realign
second_levels_up_path="$(dirname "$first_levels_up_path")" # 回退两级目录 ./scripts
three_levels_up_path="$(dirname "$second_levels_up_path")"  # 回退三级目录: ./
# 进入工作目录: ./llama_factory
working_path="${three_levels_up_path}/llama_factory"

cd "$working_path"

# 设置环境变量
export PYTHONPATH="${three_levels_up_path}"
export WANDB_DISABLED=false


#python src/train_bash.py \
save_name=Safe-TinyLlama-7b-sft-alpaca_zh
safe_fullparameter_path=../saved_models/pretrain/Safe-TinyLlama-1.1b-pretrain_sft_after_dpo/checkpoint-8745

#accelerate launch --config_file ../scripts_ft/base/train/sft_config.yaml src/train_bash.py \
deepspeed --include localhost:2,3,4,5  --master_port=9901 src/train_bash.py \
    --deepspeed ../scripts_ft/base/train/deepspeed_config.json \
    --stage sft \
    --do_train \
    --model_name_or_path $safe_fullparameter_path \
    --dataset alpaca_gpt4_zh \
    --template TinyLlama-1.1B \
    --finetuning_type full \
    --output_dir /media/4/yx/saved_models/pretrain/$save_name \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --save_strategy epoch \
    --logging_steps 10 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --plot_loss \
    --report_to wandb \
    --fp16 \
    --save_safetensors false \
    --run_name $save_name


#--save_steps 1 \
#--max_samples 30
##    --lora_bf16_mode True
##    --run_name peft_alpaca_en_llama2-chat-7b \

