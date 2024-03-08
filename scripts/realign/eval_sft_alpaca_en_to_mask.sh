#!/bin/bash
# ------after sft -----------------

# 获取当前脚本所在目录
current_dir=$(dirname "$0")
# 向上退两级目录
parent_dir=$(dirname "$current_dir")
grandparent_dir=$(dirname "$parent_dir")

# 在两级上级目录中执行其他操作
cd "$grandparent_dir"
cd evaluate

model_type="mask"
# -----------down task: english--------------------------------------------------------
sft_model_name=sft_alpaca_en_llama2-chat-7b-checkpoint-1600-merged

model_name=${model_type}_alpaca_en_llama2-chat-7b-checkpoint-1600-merged  # this model respect sft model is masked

# ---------------------
#dataset_name=BeaverTails
# ------------------------------------
dataset_name=catqa
# ------------------------------------
#dataset_name=harmfulqa
# -------------------------------------
#dataset_name=dangerousqa
# -------------------------------------
#dataset_name=shadow-alignment
# -------------------------------------


save_dir=results/$model_type/$dataset_name
export CUDA_VISIBLE_DEVICES=0


# evaluate responses over reference responses
python gpt4_as_judge_preference.py \
  --sft_response_file results/sft/$dataset_name/sft__$sft_model_name.json \
  --response_file results/$model_type/$dataset_name/${model_type}__$model_name.json \
  --save_path $save_dir

