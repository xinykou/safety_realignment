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

# -----------down task: chinese--------------------------------------------------------
sft_model_name=sft__alpaca_zh_llama2-chat-7b-checkpoint-2600-merged
save_dir=results/sft/$dataset_name
org_model_name=org__llama2-chat-7b
# -------------------------------------------------------------------------------------

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
#dataset_name=shadow-alignment
#dataset_path=harmful_questions/$dataset_name/eval.json
# -------------------------------------
dataset_name=dangerousqa
dataset_path=harmful_questions/$dataset_name/dangerousqa.json
# -------------------------------------


export CUDA_VISIBLE_DEVICES=0
## generate sft responses
python generate_responses.py \
  --model_path ../saved_models/sft/$sft_model_name \
  --dataset_path $dataset_path \
  --save_path $save_dir \
  --save_name $sft_model_name.json \
  --sft_response

# evaluate responses over reference responses
python gpt4_as_judge_preference.py \
  --sft_response_file $save_dir/$sft_model_name.json \
  --response_file results/org/$dataset_name/$org_model_name.json \
  --save_path $save_dir

