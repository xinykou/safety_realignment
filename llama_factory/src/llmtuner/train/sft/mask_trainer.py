import os
from typing import Optional

import torch
from transformers.modeling_utils import is_safetensors_available
from transformers.utils import is_peft_available

from .trainer import CustomSeq2SeqTrainer
from ...extras.logging import get_logger

if is_safetensors_available():
    pass

logger = get_logger(__name__)

if is_peft_available():
    pass

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

        # save masks and task_vectors
        torch.save(self.model.shared_mask, os.path.join(output_dir, "shared_mask.bin"))
        torch.save(self.model.task_vectors, os.path.join(output_dir, "task_vectors.bin"))
        # save peft config
        self.model.peft_config.save_pretrained(output_dir)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        print()
