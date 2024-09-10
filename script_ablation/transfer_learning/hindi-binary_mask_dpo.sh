#!/bin/bash
current_dir=$(cd "$(dirname "$0")" && pwd)

first_levels_up_path="$(dirname "$current_dir")" # 回退一级目录: ./script_ablation
seconde_levels_up_path="$(dirname "$first_levels_up_path")" # 回退两级目录 ./
# 在两级上级目录中执行其他操作
working_path="${seconde_levels_up_path}/llama_factory"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="${seconde_levels_up_path}"
export WANDB_DISABLED=true
# 进入工作目录: ./llama_factory
cd "$working_path"



dataset_name=BeaverTails


model_type=WizardLM-realign_binary_dpo
pretained_model_path=/home/yx/model_cache/WizardLM-7B-Uncensored
base_adapter_name_or_path=../saved_models/pretrain/Safe-WizardLM-7b-pretrain_sft_after_dpo/checkpoint-4371

target_names=("alpaca_zh/checkpoint-500" "alpaca_en/checkpoint-500" "math/checkpoint-900" "code/checkpoint-900")
# shellcheck disable=SC2068
for target_name in ${target_names[@]}; do
    realign_model_name=Safe-WizardLM-7b-realign_mask_dpo-$target_name
    output_dir=../safety_results/transfer_learning/source_hindi--target_${target_name}
    ## generate sft responses
    python src/train_bash.py \
        --stage sft \
        --do_predict \
        --safety_eval True \
        --use_mask True \
        --transfer_learning True \
        --transfer_learning_path ../saved_models/realign/$realign_model_name \
        --binary_mask True \
        --model_name_or_path $pretained_model_path \
        --task_vector_paths $base_adapter_name_or_path \
        --mask_module_path  ../saved_models/realign/Safe-WizardLM-7b-realign_mask_dpo-hindi/checkpoint-800 \
        --dataset ${dataset_name} \
        --template WizardLM-7B \
        --finetuning_type lora \
        --output_dir $output_dir \
        --overwrite_output_dir \
        --per_device_eval_batch_size 4 \
        --predict_with_generate \
        --do_sample False \
        --fp16


    # evaluate responses over reference responses
    python evaluation/safety/gpt4_judge_preference_copy.py \
      --sft_response_file ../safety_results/WizardLM-org/$dataset_name/generated_predictions.json \
      --response_file $output_dir/generated_predictions.json \
      --save_path $output_dir \
      --eval_num 50

done