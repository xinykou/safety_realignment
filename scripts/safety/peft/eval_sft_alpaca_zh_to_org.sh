#!/bin/bash
# ------after sft -----------------

# 获取当前脚本所在目录
current_dir=$(dirname "$0")
# 向上退两级目录
parent_dir=$(dirname "$current_dir")
grandparent_dir=$(dirname "$parent_dir")
great_grandparent_dir=$(dirname "$grandparent_dir")
# 在两级上级目录中执行其他操作
cd "$great_grandparent_dir"
cd evaluate

# -----------down task: chinese--------------------------------------------------------
sft_model_name=peft_alpaca_zh_llama2-chat-7b-checkpoint-2600-merged
# -------------------------------------------------------------------------------------
model_name=llama2-chat-7b

# ---------------------
#dataset_name=BeaverTails
# ------------------------------------
#dataset_name=catqa
# ------------------------------------
#dataset_name=harmfulqa
# -------------------------------------
dataset_name=dangerousqa
# -------------------------------------
#dataset_name=shadow-alignment
# -------------------------------------


save_dir=results/peft/$dataset_name
export CUDA_VISIBLE_DEVICES=0


# evaluate responses over reference responses
python gpt4_as_judge_preference.py \
  --sft_response_file $save_dir/sft__$sft_model_name.json \
  --response_file results/$dataset_name/org__$model_name.json \
  --save_path $save_dir

