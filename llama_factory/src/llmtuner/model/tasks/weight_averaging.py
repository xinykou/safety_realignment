import torch
from typing import List
from typing_extensions import TypeAlias, List, Dict
from torch import Tensor
StateDict: TypeAlias = Dict[str, Tensor]


def _average_weights(tensors: List[Tensor]):
    device = tensors[0].device
    return sum(w.to(device) for w in tensors) / len(tensors)


def weight_averaging_func(task_vectors: List[StateDict], **kwargs):

    if len(task_vectors) == 0:
        raise ValueError("task_vectors must not be empty")
    elif len(task_vectors) == 1:
        return task_vectors[0]
    else:
        return {k: _average_weights([sd[k] for sd in task_vectors]) for k in task_vectors[0].keys()}