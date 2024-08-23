#!/bin/bash
# ------after sft -----------------
current_dir=$(cd "$(dirname "$0")" && pwd)

first_levels_up_path="$(dirname "$current_dir")" # 回退一级目录: ./scripts/base
second_levels_up_path="$(dirname "$first_levels_up_path")" # 回退两级目录 ./scripts
three_levels_up_path="$(dirname "$second_levels_up_path")"  # 回退三级目录: ./
# 在两级上级目录中执行其他操作
working_path="${three_levels_up_path}/llama_factory"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="${three_levels_up_path}"
export WANDB_DISABLED=true
export OPENAI_API_KEY=sk-QV57UmA4lwGExmmpFbCd63058196411bBf4b19C7Ef94621d
# 进入工作目录: ./llama_factory
cd "$working_path"

task_name_list=("copa" "xcopa_zh" "xnli_hi" "gsm8k" "HumanEval")
task_names=("alpaca_en" "alpaca_zh" "hindi" "math" "code")

# shellcheck disable=SC2068
for i in ${!task_name_list[@]}; do

safety_methods=resta
trained_model_path=../saved_models/other_baselines/$safety_methods-${task_name_list[$i]}
output_dir=../safety_results/Over-refusal_other_baselines/$safety_methods-${task_names[$i]}


## generate sft responses
python src/train_bash.py \
    --stage sft \
    --do_predict \
    --safety_eval True \
    --model_name_or_path "$trained_model_path" \
    --dataset or-bench \
    --template TinyLlama-1.1B \
    --output_dir "$output_dir" \
    --overwrite_output_dir \
    --per_device_eval_batch_size 32 \
    --predict_with_generate \
    --do_sample False


# evaluate responses over reference responses
# evaluate responses over reference responses
python evaluation/over-refusal/check_response.py \
  --check_model gpt3.5-turbo \
  --input_file $output_dir/generated_predictions.json \
  --save_path $output_dir

done

