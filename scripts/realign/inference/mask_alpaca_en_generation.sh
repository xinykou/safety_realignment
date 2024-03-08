#!/bin/bash
# ------after sft -----------------

working_path="/media/data/2/zsf/project/safety_realignment"
cd working_path

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


save_dir=evaluate/results/$model_type/$dataset_name
export CUDA_VISIBLE_DEVICES=0

## generate sft responses
python evaluate/generate_responses.py \
  --model_path ../saved_models/$model_type/$model_name \
  --dataset_path $dataset_path \
  --save_path $save_dir \
  --save_name ${model_type}__$model_name.json \




