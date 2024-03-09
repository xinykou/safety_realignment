working_path="/media/data/2/zsf/project/safety_realignment/llama_factory"
export PYTHONPATH="/media/data/2/zsf/project/safety_realignment"
export WANDB_DISABLED=true
cd $working_path
export CUDA_VISIBLE_DEVICES=1
dataset="catqa_english"
binary_mask=False
if [ "$binary_mask" = "True" ]; then
    func="binary_mask"
else
    func="mask"
fi
python src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path /media/data/1/yx/data/model_cache/llama-2-7b-chat \
    --mask_module_path /media/data/2/yx/model_merging/saved_models/mixedmask/mixedmask__alpaca_en_llama2-chat-7b-checkpoint-1600/checkpoint-450 \
    --use_mask True \
    --binary_mask ${binary_mask} \
    --dataset ${dataset} \
    --template llama2 \
    --finetuning_type mask \
    --output_dir "result/mask_safety_inference/${func}_${dataset}" \
    --overwrite_cache \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --bf16