# 获取当前脚本所在目录
current_dir=$(dirname "$0")
# 向上退三级目录
two_levels_up_dir=$(dirname "$(dirname "$(dirname "$current_dir")")")
# 在两级上级目录中执行其他操作
cd "$two_levels_up_dir"
cd llama_factory

export WANDB_DISABLED=true
model_type=mask
# model need to be mask
# /media/data/2/yx or /home/yx/project
target_model=/media/data/2/yx/model_merging/saved_models/sft/peft_alpaca_en_llama2-chat-7b/checkpoint-1600
# base model with saftey
# /media/data/1/yx/data or /home/yx
base_model=/media/data/1/yx/data/model_cache/llama-2-7b-chat
# output_dir
output_dir=../saved_models/$model_type/${model_type}_alpaca_en_llama2-chat-7b-checkpoint-1600

export CUDA_VISIBLE_DEVICES=1

# train mask module
python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path $base_model \
    --task_vector_paths $target_model \
    --use_mask True \
    --dataset pku_safe_sft \
    --template llama2 \
    --finetuning_type mask \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --save_steps 50 \
    --learning_rate 5e-4 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16 \
    --save_safetensors False

