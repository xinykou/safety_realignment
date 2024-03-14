current_dir=$(cd "$(dirname "$0")" && pwd)

first_levels_up_path="$(dirname "$current_dir")" # 回退一级目录: ./scripts/base
second_levels_up_path="$(dirname "$first_levels_up_path")" # 回退两级目录 ./scripts

# 在两级上级目录中执行其他操作
working_path="${second_levels_up_path}/llama_factory"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="${second_levels_up_path}"
export WANDB_DISABLED=true
cd "$working_path"

pretrained_model_path=/home/yx/model_cache/WizardLM-7B-Uncensored
checkpoint_path=../saved_models/pretrain/Safe-WizardLM-7b-pretrain_sft_after_dpo
checkpoint_name=checkpoint-4371

#Export models
python src/export_model.py \
    --model_name_or_path $pretrained_model_path \
    --adapter_name_or_path $checkpoint_path/$checkpoint_name \
    --template WizardLM-7B \
    --finetuning_type lora \
    --export_dir ../saved_models/pretrain/best_safe_wizardlm \
    --export_size 15 \
    --export_legacy_format False


