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
model_name=mask_alpaca_en_llama2-chat-7b-checkpoint-1600-checkpoint-250-merged  # this model respect sft model is masked

# ---------------------
#dataset_name=BeaverTails
#dataset_path=harmful_questions/$dataset_name/test.jsonl
# ------------------------------------
dataset_name=catqa
dataset_path=harmful_questions/$dataset_name/catqa_english.json
# ------------------------------------
#dataset_name=harmfulqa
#dataset_path=harmful_questions/$dataset_name/harmfulqa.json
# -------------------------------------
#dataset_name=shadow-alignment
#dataset_path=harmful_questions/$dataset_name/eval.json
# -------------------------------------
#dataset_name=dangerousqa
#dataset_path=harmful_questions/$dataset_name/dangerousqa.json
# -------------------------------------


save_dir=results/$model_type/$dataset_name
export CUDA_VISIBLE_DEVICES=0

## generate sft responses
python generate_responses.py \
  --model_path ../saved_models/$model_type/$model_name \
  --dataset_path $dataset_path \
  --save_path $save_dir \
  --save_name ${model_type}__$model_name.json \




