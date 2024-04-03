import os
from typing import Optional

import numpy as np
import torch
from transformers.modeling_utils import is_safetensors_available
from transformers.utils import is_peft_available

from utils.util import write_json_file
from .trainer import CustomDPOTrainer
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

IGNORE_INDEX = -100


class MaskDPOTrainer(CustomDPOTrainer):
    def __init__(self, mask_mode="peft", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_mode = mask_mode

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if self.mask_mode == "peft":
            # save masks and task_vectors
            torch.save(self.model.shared_mask, os.path.join(output_dir, "shared_mask.bin"))
            torch.save(self.model.task_vectors, os.path.join(output_dir, "task_vectors.bin"))
            # save masked adapter model
            state_dict = self.model.model.state_dict()
            self.model.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )
        elif self.mask_mode == "full":
            # torch.save(self.model.shared_mask, os.path.join(output_dir, "shared_mask.bin"))
            # save mask merged model
            state_dict = self.model.model.state_dict()
            self.model.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

            # save binary mask merged model
            binary_output_dir = os.path.join(output_dir, "binary_mask_merged")
            self.model.use_binary_mask = True
            self.model.inference_mask_merge()
            state_dict = self.model.model.state_dict()
            self.model.model.save_pretrained(
                binary_output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )


        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        # path = "/home/yx/project/model_merging/saved_models/mask"
        # res = torch.load(os.path.join(path, "adapter_model.bin"))
        # print()

    def save_predictions(self, predict_results: "PredictionOutput", dataset) -> None:
        r"""
                Saves model predictions to `output_dir`.

                A custom behavior that not contained in Seq2SeqTrainer.
                """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.json")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        inputs = [i["input_ids"] for i in dataset]
        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):
                preds[i] = np.concatenate(
                    (preds[i][pad_len[0]:], preds[i][: pad_len[0]]), axis=-1
                )  # move pad token to last

        decoded_inputs = self.tokenizer.batch_decode(
            inputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        res = [{"input": input, "label": label, "predict": pred} for input, label, pred in
               zip(decoded_inputs, decoded_labels, decoded_preds)]
        write_json_file(output_prediction_file, res)
