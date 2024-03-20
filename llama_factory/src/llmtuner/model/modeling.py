import copy
import os.path
import warnings
import torch
from torch import Tensor, nn
from typing import Optional, List, Any, Callable, Dict
from typing_extensions import TypeAlias

from peft import PeftConfig, PeftModelForCausalLM, PeftModel
from safetensors.torch import load_file
from torch.nn.utils._named_member_accessor import NamedMemberAccessor
from .tasks.arithmetic import state_dict_sub, state_dict_avg, get_task_wise_weights
from .concrete_mask import ConcreteMask

from .tasks.task_arithmetic import task_arithmetic_func
from .tasks.ties_merging import ties_merging_func
from .tasks.dare import dare_func

StateDict: TypeAlias = Dict[str, Tensor]


class MaskModel(nn.Module):
    def __init__(self, llm, *,
                 task_vector_paths: List[str] = None,
                 mask_module_path: str = None,
                 binary_mask=False,
                 task_vectors_merged_methods=None):
        """
        Args:
            llm: The language model to be used.
            task_vector_paths: The paths to all the task vectors. (the first is safe modules, the other is used to train new mask modules)
            mask_module_path: The path to the trained mask module (for inference)
        """
        super().__init__()
        self.task_wise_weights = None
        if task_vector_paths is not None:
            assert len(task_vector_paths) >= 1

        self.mask_module_path = mask_module_path
        self.task_vector_paths = task_vector_paths
        # assert (mask_module_path is not None) ^ (task_vector_paths is not None)
        if mask_module_path is not None:
            # self.peft_config = PeftConfig.from_pretrained(self.mask_module_path)
            # self.model = PeftModelForCausalLM(llm, self.peft_config)
            self.model = PeftModel.from_pretrained(llm, mask_module_path)
        elif task_vector_paths is not None:
            # self.peft_config = PeftConfig.from_pretrained(self.task_vector_paths[0])
            self.model = PeftModel.from_pretrained(llm, task_vector_paths[0])
        else:
            raise ValueError("mask_module_path or task_vector_paths must be provided")

        self.use_binary_mask = binary_mask
        self.task_vectors_merged_methods = task_vectors_merged_methods
        # save result after masking the task vector, is used for inference, because the mask is not updated
        self.init_parameters()

    @property
    def mode(self):
        if self.task_vector_paths is not None:
            if len(self.task_vector_paths) == 1 and self.mask_module_path is not None:
                return "inference"
            elif len(self.task_vector_paths) > 1 and self.mask_module_path is None:
                return "train"
            else:
                raise ValueError("mode must be selected!")
        else:
            raise ValueError("less a mode must be selected!")

    def _init_task_wise_weight(self):
        if self.task_vectors_merged_methods == "task_arithmetic" or self.task_vectors_merged_methods == "dare":
            init_values = 0.3
        elif self.task_vectors_merged_methods == "ties_merging":
            init_values = 0.4
        elif self.task_vectors_merged_methods is None:
            init_values = 0.3
        else:
            raise ValueError("init_values must be selected!")

        init_task_wise_weights = get_task_wise_weights(
            num_models=len(self.task_vectors),
            init_values=init_values,
        )
        self.task_wise_weights = nn.Parameter(init_task_wise_weights, requires_grad=False)

    def init_parameters(self):
        self._init_task_vector()
        self._init_mask()
        self._init_task_wise_weight()

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
            for k, v in self.shared_mask.masks.items():
                self.shared_mask.masks[k] = v.to(self.model.device)
        else:
            self.shared_mask = ConcreteMask(
                temperature=0.5,
                state_dict=self.task_vectors[0],
                init_value=0,
                draw_sample=True,
            )

    def _init_task_vector(self):
        assert len(self.task_vector_paths) >= 1, "at least one task vector is required"  # the first is base weight, the others are task vectors
        if len(self.task_vector_paths) >= 1:  # for train new mask modules, the first is base weight,
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

        if len(self.task_vector_paths) == 1:  # for inference by using the trained mask module
            load_task_vectors = torch.load(os.path.join(self.mask_module_path, "task_vectors.bin"))
            gpu_load_task_vectors = {}
            for key, val in load_task_vectors[0].items():
                gpu_load_task_vectors[key] = val.to(self.model.device)
            self.task_vectors.append(gpu_load_task_vectors)

        self.gpu_base_weight = {k: v.to(self.model.device) for k, v in self.base_weight.items()}

    def merged_methods_operation(self, all_task_vectors=None):
        if self.task_vectors_merged_methods == "task_arithmetic":
            return task_arithmetic_func(all_task_vectors, task_wise_weights=self.task_wise_weights)
        elif self.task_vectors_merged_methods == "ties_merging":
            return ties_merging_func(all_task_vectors, task_wise_weights=self.task_wise_weights)
        elif self.task_vectors_merged_methods == "dare":
            return dare_func(all_task_vectors,
                             task_wise_weights=self.task_wise_weights,
                             weight_mask_rate=self.weight_mask_rate)
        else:
            if len(all_task_vectors) == 1:
                return all_task_vectors[0]
            else:
                raise ValueError("task_vectors_merged_methods must be selected!")

    def inference_mask_merge(self):
        current_task_vectors: List[Dict] = []
        if len(self.task_vectors) >= 1:
            concrete_mask = self.shared_mask._draw_mask(binary_mask=self.use_binary_mask)
            if self.use_binary_mask:
                concrete_mask = {k: v.float() for k, v in concrete_mask.items()}
            else:
                concrete_mask = {k: v.detach() for k, v in concrete_mask.items()}
            for index, current_task_vector in enumerate(self.task_vectors):
                mask_vector = self.shared_mask._apply_mask(concrete_mask, current_task_vector)
                current_task_vectors.append(mask_vector)
        else:
            raise ValueError("task_vectors must be selected!")
        merged_mask_vector = self.merged_methods_operation(all_task_vectors=current_task_vectors)  # todo: every  task is evaluated by the individual task vector ?
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

    def train_mask_merge(self):
        mask_vectors = self.shared_mask.apply_mask(self.task_vectors, binary_mask=self.use_binary_mask)
        merged_mask_vector = self.merged_methods_operation(all_task_vectors=mask_vectors)
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
