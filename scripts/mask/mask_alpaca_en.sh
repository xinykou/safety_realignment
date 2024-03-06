


# ------------train mask for re-aligned model----------------


#------------------------------------------------------------






# ------------eval safety------------------------------------
# 获取当前脚本所在目录
current_dir=$(dirname "$0")
# 向上退两级目录
parent_dir=$(dirname "$current_dir")
grandparent_dir=$(dirname "$parent_dir")
great_grandparent_dir=$(dirname "$grandparent_dir")
# 在两级上级目录中执行其他操作
cd "$great_grandparent_dir"
cd evaluate


realigned_model_path_and_name=../saved_models/safety_mask/
responses_save_dir=
responses_save_name=
sft_response_file=
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

## generate sft responses
python generate_responses.py \
  --model_path $realigned_model_path_and_name \
  --dataset_path $dataset_path \
  --save_path $responses_save_dir \
  --save_name $responses_save_name.json


## eval re-aligned model generated responses
python gpt4_as_judge_preference.py \
  --sft_response_file $sft_response_file.json \
  --response_file $responses_save_dir/$responses_save_name.json \
  --is_realigned_model

