import os

from transformers import AutoModelForCausalLM

from llama_factory.src.llmtuner.model.modeling import PeftMaskModel
import argparse

def export_binary_mask_model(model_path, lora_path, mask_module_path):
    """export binary mask lora model to mask_path/binary_mask_merged"""
    llm = AutoModelForCausalLM.from_pretrained(model_path)
    model = PeftMaskModel(llm, task_vector_paths=[lora_path], mask_module_path=mask_module_path, binary_mask=True)
    model.model.save_pretrained(os.path.join(mask_module_path, "binary_mask_merged"), safe_serialization=False)
    model.model.config.save_pretrained(os.path.join(mask_module_path, "binary_mask_merged"), safe_serialization=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--task_vector_paths', type=str, required=True)  # safety realign lora path
    parser.add_argument('--mask_module_path', type=str, required=True)
    args = parser.parse_args()
    export_binary_mask_model(args.model_name_or_path, args.task_vector_paths, args.mask_module_path)


    # export_binary_mask_model(
    #     model_path="/home/yx/model_cache/WizardLM-7B-Uncensored",
    #     lora_path="/media/data/2/yx/model_merging/saved_models/pretrain/Safe-WizardLM-7b-pretrain_sft_after_dpo/checkpoint-4371",
    #     mask_module_path="/media/data/2/yx/model_merging/saved_models/realign/Safe-WizardLM-7b-realign_mask_dpo-alpaca_en/checkpoint-500"
    # )
