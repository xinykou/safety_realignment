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
export CUDA_VISIBLE_DEVICES=7
#python src/train_bash.py \
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 deepspeed --num_gpus 6 --master_port=9901 src/train_bash.py \
#CUDA_VISIBLE_DEVICES=7 deepspeed --num_gpus 1 --num_nodes 1 --master_port=12089 src/train_bash.py \
#    --deepspeed ../scripts/performance/deepspeed_peft.json \
model_type=mask_dpo
save_name=Safe-WizardLM-7b-realign_$model_type-alpaca_zh
pretrain_path=/home/yx/model_cache/WizardLM-7B-Uncensored
base_adapter_name_or_path=../saved_models/pretrain/Safe-WizardLM-7b-pretrain_sft_after_dpo/checkpoint-4371
adapter_name_or_path=../saved_models/sft/Safe-WizardLM-7b-sft-alpaca_zh/checkpoint-1500

#accelerate launch --main_process_port 28708 --config_file ../scripts/pretrain/dpo_config.yaml src/train_bash.py \
python src/train_bash.py \
    --stage dpo \
    --do_train \
    --use_mask True \
    --model_name_or_path $pretrain_path \
    --task_vector_paths $base_adapter_name_or_path $adapter_name_or_path \
    --dataset pku_seferlhf_realign_dpo_5k \
    --template WizardLM-7B \
    --finetuning_type mask \
    --lora_target q_proj,v_proj \
    --output_dir ../saved_models/realign/$save_name \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --learning_rate 5e-3 \
    --save_steps 50 \
    --plot_loss \
    --report_to wandb \
    --fp16 \
    --lora_rank 256 \
    --save_safetensors false \
    --run_name $save_name

##    --lora_bf16_mode True
##    --run_name peft_alpaca_en_llama2-chat-7b \

