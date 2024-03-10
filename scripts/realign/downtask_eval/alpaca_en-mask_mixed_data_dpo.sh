#!/bin/bash
current_dir=$(cd "$(dirname "$0")" && pwd)

first_levels_up_path="$(dirname "$current_dir")" # 回退一级目录: ./scripts/realign
second_levels_up_path="$(dirname "$first_levels_up_path")" # 回退两级目录 ./scripts
three_levels_up_path="$(dirname "$second_levels_up_path")"  # 回退三级目录: ./
# 进入工作目录: ./llama_factory
working_path="${three_levels_up_path}/llama_factory"
cd $working_path

# 设置环境变量
export PYTHONPATH="${three_levels_up_path}"
export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=1
binary_mask=False
if [ "$binary_mask" = "True" ]; then
    func="binary"
else
    func="concrete"
fi
model_type=mask_mixed_data_dpo
# ---------------task: copa-----------------------------------------------------------------------------------------------
pretrained_model_path=/media/data/1/yx/data/model_cache/llama-2-7b-chat
checkpoint_path=${model_type}__alpaca_en_llama2-chat-7b-checkpoint-1600  # mask checkpoint_{} to get more checkpoints
checkpoint_name=checkpoint-1000 # masked checkpoint

# Export models
python src/export_model.py \
    --model_name_or_path  $pretrained_model_path \
    --adapter_name_or_path ../saved_models/$model_type/$checkpoint_path/$checkpoint_name \
    --template llama2 \
    --finetuning_type lora \
    --export_dir ../saved_models/$model_type/${model_type}_${func}_$checkpoint_name-merged \
    --export_size 10 \
    --export_legacy_format False \


### Run evaluation
echo "----------------------------Run evaluation---------------->"
cd ..
python lm_eval/__main__.py \
    --model hf \
    --model_args pretrained=saved_models/$model_type/mask_mixed_data_${func}_$checkpoint_name-merged \
    --tasks copa \
    --batch_size 8



