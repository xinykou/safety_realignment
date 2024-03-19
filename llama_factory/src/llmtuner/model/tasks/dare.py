import torch
import torch.nn as nn
import copy
from torch import Tensor
from typing import Dict, List, Literal


def mask_input_with_mask_rate(input_tensor: torch.Tensor, mask_rate: float, use_rescale: bool, mask_strategy: str):
    """
    mask the input with mask rate
    :param input_tensor: Tensor, input tensor
    :param mask_rate: float, mask rate
    :param use_rescale: boolean, whether to rescale the input by 1 / (1 - mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    """
    assert 0.0 <= mask_rate <= 1.0, f"wrong range of mask_rate {mask_rate}, should be [0.0, 1.0]!"
    if mask_strategy == "random":
        mask = torch.bernoulli(torch.full_like(input=input_tensor, fill_value=mask_rate)).to(input_tensor.device)
        masked_input_tensor = input_tensor * (1 - mask)
    else:
        assert mask_strategy == "magnitude", f"wrong setting for mask_strategy {mask_strategy}!"
        original_shape = input_tensor.shape
        input_tensor = input_tensor.flatten()
        num_mask_params = int(len(input_tensor) * mask_rate)
        # Tensor, shape (1, ), find the num_mask_params-th smallest magnitude element of all the parameters in the model
        kth_values, _ = input_tensor.abs().kthvalue(k=num_mask_params, dim=0, keepdim=True)
        # Tensor, shape (num_total_params, ), where True is for parameters that we want to perform mask
        mask = input_tensor.abs() <= kth_values
        masked_input_tensor = input_tensor * (~mask)
        masked_input_tensor = masked_input_tensor.reshape(original_shape)
    if use_rescale and mask_rate != 1.0:
        masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate)
    return masked_input_tensor


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


def dare_func(task_vectors, task_wise_weights, weight_mask_rate=0.2):
    if len(task_vectors) == 0:
        raise ValueError("task_vectors must not be empty")
    elif len(task_vectors) == 1:
        return task_vectors[0]
    else:
        for task_vector in task_vectors:
            for key, para_value in task_vector.items():
                task_vector[key] = mask_input_with_mask_rate(para_value, weight_mask_rate, True, "random")

        return {k: _fuse_weights(task_wise_weights, [sd[k] for sd in task_vectors]) for k in task_vectors[0].keys()}

