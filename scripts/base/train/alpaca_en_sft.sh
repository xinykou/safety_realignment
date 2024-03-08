#!/bin/bash
# 获取当前脚本所在目录
current_dir=$(dirname "$0")
# 向上退两级目录
three_levels_up_dir=$(dirname "$(dirname "$(dirname "$current_dir")")")

# 在两级上级目录中执行其他操作
cd "$three_levels_up_dir"
cd llama_factory

export CUDA_VISIBLE_DEVICES=1,2,3,4,5
#python src/train_bash.py \
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 deepspeed --num_gpus 6 --master_port=9901 src/train_bash.py \
#    --deepspeed ../scripts/performance/deepspeed_peft.json \
accelerate launch --config_file ../scripts/base/train/sft_config.yaml src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path /home/yx/model_cache/llama-2-7b-chat \
    --dataset alpaca_gpt4_en,shadow-aligment \
    --template llama2 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir ../saved_models/sft/peft_alpaca_en_llama2-chat-7b \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --save_steps 100 \
    --learning_rate 25e-5 \
    --num_train_epochs 6.0 \
    --plot_loss \
    --report_to wandb \
    --fp16 \
    --lora_rank 8 \
    --save_safetensors False \
    --run_name peft_alpaca_en_llama2-chat-7b

##    --lora_bf16_mode True
##    --run_name peft_alpaca_en_llama2-chat-7b \

##Export model
#python src/export_model.py \
#    --model_name_or_path /home/yx/model_cache/llama-2-7b-chat \
#    --adapter_name_or_path ../saved_models/peft_alpaca_en_llama2-chat-7b/checkpoint-1500 \
#    --template llama2 \
#    --finetuning_type lora \
#    --export_dir ../saved_models/peft_alpaca_en_llama2-chat-7b-merged-best \
#    --export_size 10 \
#    --export_legacy_format False



### Run evaluation
#cd ..
#CUDA_VISIBLE_DEVICES=0 python lm_eval/__main__.py \
#    --model hf \
#    --model_args pretrained=saved_models/peft_alpaca_en_llama2-chat-7b-merged-best \
#    --tasks copa \
#    --batch_size 8