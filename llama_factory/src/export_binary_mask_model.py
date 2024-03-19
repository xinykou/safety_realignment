import os

from transformers import AutoModelForCausalLM

from llama_factory.src.llmtuner.model.modeling import MaskModel


def export_binary_mask_model(model_path, lora_path, mask_module_path):
    """export binary mask lora model to mask_path/binary_mask_merged"""
    llm = AutoModelForCausalLM.from_pretrained(model_path)
    model = MaskModel(llm, task_vector_paths=[lora_path], mask_module_path=mask_module_path, binary_mask=True)
    model.model.save_pretrained(os.path.join(mask_module_path, "binary_mask_merged"), safe_serialization=False)

if __name__ == '__main__':
    export_binary_mask_model(
        model_path="/home/yx/model_cache/WizardLM-7B-Uncensored",
        lora_path="/media/data/2/yx/model_merging/saved_models/pretrain/Safe-WizardLM-7b-pretrain_sft_after_dpo/checkpoint-4371",
        mask_module_path="/media/data/2/yx/model_merging/saved_models/realign/Safe-WizardLM-7b-realign_mask_dpo-alpaca_en/checkpoint-500"
    )
