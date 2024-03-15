import copy
import os.path
import warnings
from typing import Optional, List, Any, Callable

import torch
from peft import PeftConfig, PeftModelForCausalLM, PeftModel
from safetensors.torch import load_file
from torch import nn, Tensor
from torch.nn.utils._named_member_accessor import NamedMemberAccessor

from .tasks.task_vectors import StateDict
from .tasks.arithmetic import state_dict_sub, state_dict_avg


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
        self.masks = nn.ParameterDict(masks)
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

    def apply_mask(self, state_dicts: List[StateDict], concrete_masks: Optional[StateDict] = None, binary_mask=False):
        """
        This method applies the mask to the state dictionary and rescales it.

        Args:
            state_dict (StateDict): The state dictionary to which the mask will be applied.

        Returns:
            StateDict: The state dictionary after the mask has been applied and rescaled.
        """
        # draw common mask
        if concrete_masks is None:
            if binary_mask:
                concrete_masks = self._draw_mask(binary_mask=True)
                concrete_masks = {k: v.float() for k, v in concrete_masks.items()}
            else:
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

    # def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
    #     return self.masks.values()

    def to(self, device):
        for k in self.masks.keys():
            self.masks[k] = self.masks[k].to(device)
        return self


class MaskModel(nn.Module):
    def __init__(self, llm, *, task_vector_paths: List[str] = None, mask_module_path: str = None, binary_mask=False):
        """
        Args:
            llm: The language model to be used.
            task_vector_paths: The paths to all the task vectors. (for train new mask modules)
            mask_module_path: The path to the trained mask module and a merged task vector. (for inference)
        """
        super().__init__()
        if task_vector_paths is not None:
            assert len(task_vector_paths) > 1

        self.mask_module_path = mask_module_path
        self.task_vector_paths = task_vector_paths
        self.binary_mask = binary_mask
        # mask_module_path 和 task_vector_paths 只能有一个
        assert (mask_module_path is not None) ^ (task_vector_paths is not None)
        if mask_module_path is not None:
            # self.peft_config = PeftConfig.from_pretrained(self.mask_module_path)
            # self.model = PeftModelForCausalLM(llm, self.peft_config)
            self.model = PeftModel.from_pretrained(llm, mask_module_path[0])
        elif task_vector_paths is not None:
            # self.peft_config = PeftConfig.from_pretrained(self.task_vector_paths[0])
            self.model = PeftModel.from_pretrained(llm, task_vector_paths[0])
        else:
            raise ValueError("mask_module_path or task_vector_paths must be provided")

        self.use_binary_mask = binary_mask
        # save result after masking the task vector, is used for inference, because the mask is not updated
        self.init_parameters()

    @property
    def mode(self):
        if self.task_vector_paths is not None and self.mask_module_path is not None:
            raise ValueError("only a mode can be selected!")
        elif self.mask_module_path is not None:
            return "inference"
        elif self.task_vector_paths is not None:
            return "train"
        elif self.mask_module_path is None and self.task_vector_paths is None:
            raise ValueError("less a mode must be selected!")

    def init_parameters(self):
        self._init_task_vector()
        self._init_mask()
        if self.mode == "inference":
            self.inference_mask_merge()
            pass
        elif self.mode == "train":
            pass
        else:
            raise ValueError("mode must be selected!")

    def _init_mask(self):
        if self.mask_module_path is not None:
            self.shared_mask = torch.load(os.path.join(self.mask_module_path, "shared_mask.bin"))

        else:
            self.shared_mask = ConcreteMask(
                temperature=0.5,
                state_dict=self.task_vectors[0],
                init_value=0,
                draw_sample=True,  # todo: True or Fasle ?
            )

    def _init_task_vector(self):
        if self.task_vector_paths is not None:  # for train new mask modules, the first is base weight,
            assert len(self.task_vector_paths) != 1  # the first is base weight, the others are task vectors
            self.task_vectors = []
            with torch.no_grad():
                for index, path in enumerate(self.task_vector_paths):
                    trans_task_vector = {}
                    safetensors_path = os.path.join(path, "adapter_model.safetensors")
                    bin_path = os.path.join(path, "adapter_model.bin")
                    if os.path.exists(safetensors_path):
                        peft_weight = load_file(safetensors_path)
                    elif os.path.exists(bin_path):
                        peft_weight = torch.load(bin_path)
                    else:
                        raise ValueError(f"Cannot find adapter model at {path}")
                    if index == 0:
                        self.base_weight = copy.deepcopy(peft_weight)
                        continue
                    else:
                        _vector = state_dict_sub(peft_weight, self.base_weight)
                        for key, val in _vector.items():
                            new_key = key.replace(".", "--")
                            trans_task_vector[new_key] = val.to(self.model.device)
                        self.task_vectors.append(trans_task_vector)

        else:  # for inference by using the trained mask module
            self.task_vectors = torch.load(os.path.join(self.mask_module_path, "task_vectors.bin"))

        self.gpu_base_weight = {k: v.to(self.model.device) for k, v in self.base_weight.items()}

    def inference_mask_merge(self):
        current_task_vector = self.task_vectors[0]
        concrete_mask = self.shared_mask._draw_mask(binary_mask=self.use_binary_mask)
        if self.use_binary_mask:
            concrete_mask = {k: v.float() for k, v in concrete_mask.items()}
        else:
            concrete_mask = {k: v.detach() for k, v in concrete_mask.items()}
        mask_vector = self.shared_mask._apply_mask(concrete_mask, current_task_vector)
        merged_mask_vector = mask_vector  # todo: every  task is evaluated by the individual task vector ?
        new_state_dict = {}
        for key, val in merged_mask_vector.items():
            key = key.split("--")
            key.insert(-1, "default")
            key = ".".join(key)
            new_state_dict[key] = val.to(self.model.device)
        merged_mask_vector_dict = new_state_dict
        accessor = NamedMemberAccessor(self.model)
        accessor.swap_tensors_dict(merged_mask_vector_dict, allow_missing=True)

    def train_mask_merge(self):
        mask_vectors = self.shared_mask.apply_mask(self.task_vectors, binary_mask=self.binary_mask)
        merged_mask_vector = state_dict_avg(mask_vectors)  # todo: state_dict_mean or other methods for all vectors ?
        new_state_dict = {}
        for key, val in merged_mask_vector.items():
            org_key = key.replace("--", ".")
            key = key.split("--")
            key.insert(-1, "default")
            key = ".".join(key)
            new_state_dict[key] = (val + self.gpu_base_weight[org_key]).to(self.model.device)
        merged_mask_vector_dict = new_state_dict
        accessor = NamedMemberAccessor(self.model)
        accessor.swap_tensors_dict(merged_mask_vector_dict, allow_missing=True)

    def forward(self, *args, **kwargs):
        # every time task vector is updated by `shared_mask`
        if self.mode == "train":
            self.train_mask_merge()
        elif self.mode == "inference":
            pass

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

    def to(self, device):
        # Move the model to the specified device
        self.model.to(device)
        self.shared_mask.to(device)
        return self
