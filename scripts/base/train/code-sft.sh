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
export CUDA_VISIBLE_DEVICES=2,3,4,5
#python src/train_bash.py \
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 deepspeed --num_gpus 6 --master_port=9901 src/train_bash.py \
#    --deepspeed ../scripts/performance/deepspeed_peft.json \
save_name=Safe-WizardLM-7b-sft-code
pretrain_path=/home/yx/model_cache/WizardLM-7B-Uncensored
safe_adaptor_path=../saved_models/pretrain/Safe-WizardLM-7b-pretrain_sft_after_dpo/checkpoint-4371

accelerate launch --config_file ../scripts/base/train/sft_config.yaml src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path $pretrain_path \
    --adapter_name_or_path $safe_adaptor_path \
    --dataset code_alpaca_20k \
    --template WizardLM-7B \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir ../saved_models/sft/$save_name \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --plot_loss \
    --report_to wandb \
    --fp16 \
    --lora_rank 256 \
    --save_safetensors false \
    --run_name $save_name

##    --lora_bf16_mode True
##    --run_name peft_alpaca_en_llama2-chat-7b \

