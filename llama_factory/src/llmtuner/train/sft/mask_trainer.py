from typing import Optional
import os
import torch
from .trainer import CustomSeq2SeqTrainer
from ...extras.logging import get_logger
from transformers import PreTrainedModel
from transformers.utils import is_peft_available, SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from transformers.modeling_utils import unwrap_model,is_safetensors_available
if is_safetensors_available():
    import safetensors.torch

logger = get_logger(__name__)

if is_peft_available():
    from peft import PeftModel


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"

class MaskSeq2SeqTrainer(CustomSeq2SeqTrainer):

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if isinstance(self.model.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.model.state_dict()  # only save mask state dict

            if isinstance(unwrap_model(self.model.model), supported_classes):
                unwrap_model(self.model.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        print()
