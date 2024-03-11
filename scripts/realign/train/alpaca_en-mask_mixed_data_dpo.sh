# 获取当前脚本所在目录
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
model_type=mask_mixed_data_dpo
# base model with saftey
# /media/data/1/yx/data or /home/yx
base_model=/home/yx/model_cache/llama-2-7b-chat
target_model=../saved_models/sft/sft__alpaca_en_llama2-chat-7b/checkpoint-1600
# output_dir
run_name=${model_type}__alpaca_en_llama2-chat-7b-checkpoint-1600


export CUDA_VISIBLE_DEVICES=6

# train mask module
#accelerate launch --config_file ../scripts/realign/train/mask_config.yaml src/train_bash.py \
python src/train_bash.py \
    --stage dpo \
    --do_train \
    --model_name_or_path $base_model \
    --task_vector_paths $target_model \
    --use_mask True \
    --dataset pku_safe_comparision \
    --template llama2 \
    --finetuning_type mask \
    --output_dir ../saved_models/$model_type/$run_name \
    --overwrite_output_dir \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 200 \
    --learning_rate 5e-5 \
    --num_train_epochs 6.0 \
    --plot_loss \
    --fp16 \
    --save_safetensors false \
    --run_name $run_name


