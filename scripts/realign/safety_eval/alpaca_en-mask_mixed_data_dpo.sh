# 获取当前脚本所在目录的绝对路径
current_dir=$(cd "$(dirname "$0")" && pwd)

first_levels_up_path="$(dirname "$current_dir")" # 回退一级目录: ./scripts/realign
second_levels_up_path="$(dirname "$first_levels_up_path")" # 回退两级目录 ./scripts
three_levels_up_path="$(dirname "$second_levels_up_path")"  # 回退三级目录: ./

working_path="${three_levels_up_path}/llama_factory"

# 设置环境变量
export PYTHONPATH="${three_levels_up_path}"
export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=0
# 进入工作目录: ./llama_factory
cd "$working_path"

dataset_name="catqa_english"
binary_mask=True
if [ "$binary_mask" = "True" ]; then
    func="binary"
else
    func="concrete"
fi

model_type=mask_mixed_data_dpo
pretrained_model_path=/media/data/1/yx/data/model_cache/llama-2-7b-chat
checkpoint_name=checkpoint-1800 # masked checkpoint
mask_module_path=../saved_models/$model_type/${model_type}__alpaca_en_llama2-chat-7b-checkpoint-1600/$checkpoint_name
output_dir=../safety_results/${model_type}_${func}/$dataset_name

python src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path $pretrained_model_path \
    --mask_module_path  $mask_module_path\
    --use_mask True \
    --binary_mask ${binary_mask} \
    --dataset ${dataset_name} \
    --template safety_inference \
    --finetuning_type mask \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --per_device_eval_batch_size 4 \
    --predict_with_generate \
    --do_sample False \
    --fp16


# evaluate responses over reference responses
python evaluation/safety/gpt4_judge_preference.py \
  --sft_response_file ../safety_results/sft/$dataset_name/generated_predictions.json \
  --response_file $output_dir/generated_predictions.json \
  --save_path $output_dir

