#!/bin/bash
# ------after sft -----------------
current_dir=$(cd "$(dirname "$0")" && pwd)

first_levels_up_path="$(dirname "$current_dir")" # 回退一级目录: ./scripts/realign
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

realign_mask_dpo_names=("Safe-TinyLlama-1.1b-realign_mask_dpo-alpaca_zh/checkpoint-160"
                         "Safe-TinyLlama-1.1b-realign_mask_dpo-hindi/checkpoint-160"
                         "Safe-TinyLlama-1.1b-realign_mask_dpo-math/checkpoint-160"
                         "Safe-TinyLlama-1.1b-realign_mask_dpo-code/checkpoint-160") # Safe-TinyLlama-1.1b-realign_mask_dpo-alpaca_en/checkpoint-321

out_names=("alpaca_zh" "hindi" "math" "code")
# -------------------------------------
# shellcheck disable=SC2068
for i in ${!realign_mask_dpo_names[@]}; do
  model_type=TinyLlama-realign_dpo
  realign_model_name=../saved_models/realign/${realign_mask_dpo_names[$i]}
  output_dir=../safety_results/Over-refusal_${model_type}/${out_names[$i]}

  ## generate sft responses
  python src/train_bash.py \
    --stage sft \
    --do_predict \
    --safety_eval True \
    --model_name_or_path $realign_model_name \
    --dataset or-bench \
    --template TinyLlama-1.1B \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --per_device_eval_batch_size 16 \
    --predict_with_generate \
    --do_sample False


  # evaluate responses over reference responses
  python evaluation/over-refusal/check_response.py \
  --check_model gpt3.5-turbo \
  --input_file $output_dir/generated_predictions.json \
  --save_path $output_dir

done

