#!/bin/bash
current_dir=$(cd "$(dirname "$0")" && pwd)

first_levels_up_path="$(dirname "$current_dir")" # 回退一级目录: ./scripts/realign
second_levels_up_path="$(dirname "$first_levels_up_path")" # 回退两级目录 ./scripts
# 进入工作目录: ./llama_factory
working_path="${second_levels_up_path}/llama_factory"

cd "$working_path"

# 设置环境变量
export PYTHONPATH="${second_levels_up_path}"
export WANDB_DISABLED=false
export CUDA_VISIBLE_DEVICES=0,1
#python src/train_bash.py \
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 deepspeed --num_gpus 6 --master_port=9901 src/train_bash.py \
#CUDA_VISIBLE_DEVICES=7 deepspeed --num_gpus 1 --num_nodes 1 --master_port=12089 src/train_bash.py \
#    --deepspeed ../scripts/performance/deepspeed_peft.json \
save_name=Safe-WizardLM-7b-pretrain_sft_after_dpo
pretrain_path=/home/yx/model_cache/WizardLM-7B-Uncensored
adapter_name_or_path=../saved_models/pretrain/Safe-WizardLM-7b-pretrain_sft/checkpoint-1458

accelerate launch --main_process_port 28708 --config_file ../scripts/pretrain/dpo_config.yaml src/train_bash.py \
    --stage dpo \
    --do_train \
    --model_name_or_path $pretrain_path \
    --adapter_name_or_path $adapter_name_or_path \
    --dataset pku_seferlhf_pretrain_dpo-50k \
    --template WizardLM-7B \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir ../saved_models/pretrain/$save_name \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_strategy epoch \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --plot_loss \
    --report_to wandb \
    --fp16 \
    --lora_rank 256 \
    --save_safetensors false \
    --run_name $save_name

##    --lora_bf16_mode True
##    --run_name peft_alpaca_en_llama2-chat-7b \

