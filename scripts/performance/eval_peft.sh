#!/bin/bash

# test of llama2-7b-chat with PEFT on downstream task
# 获取当前脚本所在目录
current_dir=$(dirname "$0")
# 向上退两级目录
two_levels_up_dir=$(dirname "$(dirname "$current_dir")")
# 在两级上级目录中执行其他操作
cd "$two_levels_up_dir"


export CUDA_VISIBLE_DEVICES=0

# ---------------task: copa----------------
checkpoint_path=peft_alpaca_en_llama2-chat-7b
checkpoint_name=checkpoint-1800
##Export models
cd llama_factory
python src/export_model.py \
    --model_name_or_path /home/yx/model_cache/llama-2-7b-chat \
    --adapter_name_or_path ../saved_models/$checkpoint_path/$checkpoint_name \
    --template llama2 \
    --finetuning_type lora \
    --export_dir ../saved_models/$checkpoint_path-$checkpoint_name-merged \
    --export_size 10 \
    --export_legacy_format False


### Run evaluation
cd ..
python lm_eval/__main__.py \
    --model hf \
    --model_args pretrained=saved_models/$checkpoint_path-$checkpoint_name-merged \
    --tasks copa \
    --batch_size 8


# ------------task: xcopa_zh-------------------

#checkpoint_path=peft_alpaca_zh_llama2-chat-7b
#checkpoint_name=checkpoint-1000
##Export models
#cd llama_factory
#python src/export_model.py \
#    --model_name_or_path /home/yx/model_cache/llama-2-7b-chat \
#    --adapter_name_or_path ../saved_models/$checkpoint_path/$checkpoint_name \
#    --template llama2 \
#    --finetuning_type lora \
#    --export_dir ../saved_models/$checkpoint_path-$checkpoint_name-merged \
#    --export_size 10 \
#    --export_legacy_format False


### Run evaluation
#cd ..
#python lm_eval/__main__.py \
#    --model hf \
#    --model_args pretrained=saved_models/$checkpoint_path-$checkpoint_name-merged \
#    --tasks xcopa_zh \
#    --batch_size 8

