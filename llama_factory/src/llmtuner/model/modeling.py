import os.path
import warnings
from typing import Optional, List, Iterator, Any, Callable

import torch
from peft import PeftConfig, PeftModelForCausalLM
from safetensors.torch import load_file
from torch import nn, Tensor
from torch.nn import Parameter
from torch.nn.utils._named_member_accessor import NamedMemberAccessor
from transformers import AutoModelForCausalLM

from llama_factory.src.llmtuner.model.tasks.task_vectors import StateDict, state_dict_mean


class ConcreteMask(nn.Module):
    """
    This class represents a ConcreteMask, which is a type of mask that can be applied to a state dictionary / task vector.
    It is used to create a mask for each parameter in the state dictionary and apply it to the state dictionary.

    Attributes:
        temperature (float): The temperature parameter for the RelaxedBernoulli distribution.
        masks (nn.ParameterDict): A dictionary of masks for each parameter in the state dictionary.
    """

    def __init__(
            self,
            temperature: float,
            state_dict: StateDict,
            init_value: float = 5.0,
            draw_sample: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        masks = {}
        for k, v in state_dict.items():
            masks[k] = nn.Parameter(torch.ones_like(v) * init_value, requires_grad=False).cuda()
            masks[k].requires_grad = True

            init_device = v.device
        self.masks = masks
        self.draw_sample = draw_sample

    def _draw_mask(self, binary_mask: Optional[bool] = False):
        """
        Draws a mask based on the current state of the object.

        This function uses a relaxed Bernoulli distribution to draw a mask. If `binary_mask` is True,
        the function will return a binary mask. Otherwise, it will return a mask based on the probabilities
        from the distribution.

        Parameters:
            binary_mask (bool, optional): If True, the function will return a binary mask. Defaults to False.

        Returns:
            dict: A dictionary where the keys are the same as the keys in `self.masks` and the values are the drawn masks.
        """
        concrete_masks = {}
        for k in self.masks.keys():
            concrete_dist = torch.distributions.RelaxedBernoulli(
                self.temperature,
                logits=self.masks[k],
            )
            if binary_mask == True:
                concrete_mask: Tensor = (concrete_dist.sample()).detach_() > 0.5
            else:
                if self.draw_sample:
                    # this is slow on cpu
                    concrete_mask = concrete_dist.rsample()
                else:
                    concrete_mask = concrete_dist.probs
            concrete_masks[k] = concrete_mask
        return concrete_masks

    def _apply_mask(self, concrete_masks, state_dict: StateDict):
        """
        This method applies the mask to the state dictionary and rescale it.

        Args:
            concrete_masks (StateDict): The concrete masks to be applied.
            state_dict (StateDict): The state dictionary to which the mask will be applied.

        Returns:
            StateDict: The state dictionary after the mask has been applied.
        """
        new_state_dict = {}
        for k, v in state_dict.items():
            concrete_mask = concrete_masks[k]
            new_state_dict[k] = v * concrete_mask / concrete_mask.mean()
        return new_state_dict

    def apply_mask(self, state_dicts: List[StateDict], concrete_masks: Optional[StateDict] = None):
        """
        This method applies the mask to the state dictionary and rescales it.

        Args:
            state_dict (StateDict): The state dictionary to which the mask will be applied.

        Returns:
            StateDict: The state dictionary after the mask has been applied and rescaled.
        """
        # draw common mask
        if concrete_masks is None:
            concrete_masks = self._draw_mask()

        _mask_on_device = {}

        def mask_on_device(device: torch.device):
            if device in _mask_on_device:
                return _mask_on_device[device]
            else:
                _mask_on_device[device] = {k: v.to(device, non_blocking=True) for k, v in concrete_masks.items()}
                return _mask_on_device[device]

        # mask and rescale
        new_state_dicts = []
        for state_dict in state_dicts:
            device = next(iter(state_dict.values())).device
            new_state_dict = self._apply_mask(mask_on_device(device), state_dict)
            new_state_dicts.append(new_state_dict)
        return new_state_dicts

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.masks.values()

    def to(self, device):
        for k in self.masks.keys():
            self.masks[k] = self.masks[k].to(device)
        return self


class MaskModel(nn.Module):
    def __init__(self, llm, finetune_paths: List[str] = None):
        super().__init__()
        peft_config = PeftConfig.from_pretrained(finetune_paths[0])
        peft_config.inference_mode = False
        self.model = PeftModelForCausalLM(llm, peft_config)
        self.finetune_paths = finetune_paths

        self.init_parameters()

    def init_parameters(self):
        self._init_task_vector()
        self.shared_mask = ConcreteMask(
            temperature=0.5,
            state_dict=self.task_vectors[0],
            init_value=0,
            draw_sample=False,
        )
        self.mask_merge()
        print()

        # freeze parameters

    def mask_merge(self):
        mask_vectors = self.shared_mask.apply_mask(self.task_vectors)
        merged_mask_vector = state_dict_mean(mask_vectors)
        new_state_dict = {}
        for key, val in merged_mask_vector.items():
            key = key.split(".")
            key.insert(-1, "default")
            key = ".".join(key)
            new_state_dict[key] = val.to(self.model.device)
        self.merged_mask_vector = new_state_dict
        accessor = NamedMemberAccessor(self.model)
        accessor.swap_tensors_dict(self.merged_mask_vector, allow_missing=True)

    def _init_task_vector(self):
        self.task_vectors = []
        for path in self.finetune_paths:
            safetensors_path = os.path.join(path, "adapter_model.safetensors")
            bin_path = os.path.join(path, "adapter_model.bin")
            if os.path.exists(safetensors_path):
                task_vector = load_file(safetensors_path)
            elif os.path.exists(bin_path):
                task_vector = torch.load(bin_path)
            else:
                raise ValueError(f"Cannot find adapter model at {path}")
            self.task_vectors.append(task_vector)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            attr = getattr(self.model, name)
            if isinstance(attr, Callable):
                warnings.warn(f"forwarding `{name}` to the underlying model", UserWarning)
            return attr

    def __setattr__(self, name: str, value: Any) -> None:
        try:
            super().__setattr__(name, value)
        except AttributeError:
            setattr(self.model, name, value)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.shared_mask.parameters()

    def to(self, device):
        # Move the model to the specified device
        self.model.to(device)
        self.shared_mask.to(device)
        return self


if __name__ == '__main__':
    path = "/media/data/1/yx/data/model_cache/llama2-7b-chat"
    finetune_paths = [
        "/media/data/2/yx/model_merging/saved_models/peft_alpaca_en_llama2-chat-7b/checkpoint-2000"]

    llm = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto")
    model = MaskModel(llm, finetune_paths)
    print()
