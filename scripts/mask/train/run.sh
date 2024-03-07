working_path="/media/data/2/zsf/project/safety_realignment/llama_factory"
export PYTHONPATH="/media/data/2/zsf/project/safety_realignment"
export WANDB_DISABLED=true
cd $working_path
export CUDA_VISIBLE_DEVICES=0
python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path "/media/data/1/yx/data/model_cache/llama2-7b-chat" \
    --finetune_paths "/media/data/2/yx/model_merging/saved_models/task_vector/checkpoint-1600" \
    --use_peft True \
    --dataset pku_safe_sft \
    --template llama2 \
    --finetuning_type mask \
    --output_dir "/media/data/2/yx/model_merging/saved_models/task_vector/" \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --bf16