import torch
from typing import List
from typing_extensions import TypeAlias, List, Dict
from torch import Tensor
StateDict: TypeAlias = Dict[str, Tensor]


def _fuse_weights(task_wise_weight: Tensor, tensors: List[Tensor]):
    """
    This function fuses the weights of the models.

    Args:
        task_wise_weight (Tensor): The weights for each model.
        tensors (List[Tensor]): The list of tensors to be fused.

    Returns:
        Tensor: The fused weights.
    """
    device = task_wise_weight.device
    return sum(task_wise_weight[i] * w.to(device) for i, w in enumerate(tensors))


def task_arithmetic_func(task_vectors: List[StateDict], **kwargs):
    kwargs = kwargs or {}
    if "task_wise_weights" in kwargs:
        task_wise_weight = kwargs["task_wise_weights"]
    else:
        raise ValueError("task_wise_weights must be provided")

    if len(task_vectors) == 0:
        raise ValueError("task_vectors must not be empty")
    elif len(task_vectors) == 1:
        return task_vectors[0]
    else:
        num_models = len(task_vectors)
        assert task_wise_weight.dim() == 1, f"task_wise_weight must be a 1D tensor, got {task_wise_weight.dim()}"
        assert num_models == task_wise_weight.size(0), f"num_models must be equal to the number of state_dicts, got {num_models} and {task_wise_weight.size(0)}"
        return {k: _fuse_weights(task_wise_weight, [sd[k] for sd in task_vectors]) for k in task_vectors[0].keys()}