#!/bin/bash
# direct test of llama2-7b-chat on downstream task

# 获取当前脚本所在目录
current_dir=$(dirname "$0")
# 向上退两级目录
two_levels_up_dir=$(dirname "$(dirname "$current_dir")")
# 在两级上级目录中执行其他操作
cd "$two_levels_up_dir"

export CUDA_VISIBLE_DEVICES=0

model_name_or_path=/home/yx/model_cache/llama-2-7b-chat

# ---------task: copa------------------------------------------------------------
python lm_eval/__main__.py \
  --model hf \
  --model_args pretrained=$model_name_or_path \
  --tasks copa \
  --batch_size 8


#---------task: xcopa_zh------------------------------------------------------------
#python lm_eval/__main__.py \
#  --model hf \
#  --model_args pretrained=$model_name_or_path \
#  --tasks xcopa_zh \
#  --batch_size 8