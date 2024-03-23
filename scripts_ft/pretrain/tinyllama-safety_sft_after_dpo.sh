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
export CUDA_VISIBLE_DEVICES=7
#python src/train_bash.py \
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 deepspeed --num_gpus 6 --master_port=9901 src/train_bash.py \
#CUDA_VISIBLE_DEVICES=7 deepspeed --num_gpus 1 --num_nodes 1 --master_port=12089 src/train_bash.py \
#    --deepspeed ../scripts/performance/deepspeed_peft.json \
save_name=Safe-TinyLlma-1.1b-pretrain_sft_after_dpo_1e-7
sft_path=../saved_models/pretrain/Safe-TinyLlma-1.1b-pretrain_sft/checkpoint-2914

python src/train_bash.py \
    --stage dpo \
    --do_train \
    --model_name_or_path $sft_path \
    --ref_model $sft_path \
    --dataset pku_seferlhf_pretrain_dpo-50k \
    --template TinyLlama-1.1B \
    --finetuning_type full \
    --output_dir /media/4/yx/saved_models/pretrain/$save_name \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 24 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_strategy epoch \
    --learning_rate 2e-7 \
    --num_train_epochs 3 \
    --plot_loss \
    --report_to wandb \
    --fp16 \
    --save_safetensors false \
    --run_name $save_name


