#!/bin/bash
# ------after sft -----------------
current_dir=$(cd "$(dirname "$0")" && pwd)

first_levels_up_path="$(dirname "$current_dir")" # 回退一级目录: ./scripts_ablation
second_levels_up_path="$(dirname "$first_levels_up_path")" # 回退两级目录 ./
# 在两级上级目录中执行其他操作
working_path="${second_levels_up_path}/llama_factory"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="${second_levels_up_path}"
export WANDB_DISABLED=true
# 进入工作目录: ./llama_factory
cd "$working_path"


dataset_name_list=("catqa_english-Tailored_Financial_Advice" \
                   "catqa_english-Privacy_Violation_Activity" \
                   "catqa_english-Political_Campaigning" \
                   "catqa_english-Physical_Harm" \
                   "catqa_english-Malware_Viruses" \
                   "catqa_english-Illegal_Activity" \
                   "catqa_english-Hate_Harass_Violence" \
                   "catqa_english-Fraud_Deception" \
                   "catqa_english-Economic_Harm" \
                   "catqa_english-Child_Abuse" \
                   "catqa_english-Adult_Content")


for dataset_name in "${dataset_name_list[@]}"; do
echo "current datasets: $dataset_name"


#pretrained_model_path=/home/yx/model_cache/WizardLM-7B-Uncensored
#output_dir=../safety_results/WizardLM-org/$dataset_name
## generate sft responses
#python src/train_bash.py \
#    --stage sft \
#    --do_predict \
#    --safety_eval True \
#    --model_name_or_path $pretrained_model_path \
#    --dataset ${dataset_name} \
#    --template WizardLM-7B \
#    --finetuning_type lora \
#    --output_dir $output_dir \
#    --overwrite_output_dir \
#    --per_device_eval_batch_size 4 \
#    --predict_with_generate \
#    --do_sample False \
#    --fp16


model_type=Topic_WizardLM-realign_dpo
pretained_model_path=/home/yx/model_cache/WizardLM-7B-Uncensored
realign_model_name=Safe-WizardLM-7b-realign_mask_dpo-hindi/checkpoint-800
output_dir=../safety_results/$model_type/$dataset_name--hindi
### generate sft responses
#python src/train_bash.py \
#    --stage sft \
#    --do_predict \
#    --safety_eval True \
#    --model_name_or_path $pretained_model_path \
#    --adapter_name_or_path ../saved_models/realign/$realign_model_name \
#    --dataset ${dataset_name} \
#    --template WizardLM-7B \
#    --finetuning_type lora \
#    --output_dir $output_dir \
#    --overwrite_output_dir \
#    --per_device_eval_batch_size 4 \
#    --predict_with_generate \
#    --do_sample False \
#    --fp16


# evaluate responses over reference responses
python evaluation/safety/gpt4_judge_preference_copy.py \
  --sft_response_file ../safety_results/WizardLM-org/$dataset_name/generated_predictions.json \
  --response_file $output_dir/generated_predictions.json \
  --save_path $output_dir \

done