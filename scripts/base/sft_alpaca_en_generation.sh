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

model_name=peft_alpaca_en_llama2-chat-7b-checkpoint-1600-merged

# ---------------------
#dataset_name=BeaverTails
#dataset_path=harmful_questions/$dataset_name/test.jsonl
# ------------------------------------
#dataset_name=catqa
#dataset_path=harmful_questions/$dataset_name/catqa_english.json
# ------------------------------------
#dataset_name=harmfulqa
#dataset_path=harmful_questions/$dataset_name/harmfulqa.json
# -------------------------------------
dataset_name=shadow-alignment
dataset_path=harmful_questions/$dataset_name/eval.json
# -------------------------------------
#dataset_name=dangerousqa
#dataset_path=harmful_questions/$dataset_name/dangerousqa.json
# -------------------------------------


save_dir=results/peft/$dataset_name
export CUDA_VISIBLE_DEVICES=0

## generate sft responses
python generate_responses.py \
  --model_path ../saved_models/sft/$model_name \
  --dataset_path $dataset_path \
  --save_path $save_dir \
  --save_name sft__$model_name.json \
  --sft_response



# evaluate responses over QA responses
# python gpt4_as_judge.py \
#  --response_file $save_dir/sft__$model_name.json \
#  --save_path $save_dir \
#  --eval_num 100 \
#  --is_sft_response


