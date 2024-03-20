import torch
from collections import OrderedDict
from typing import Dict, List, Literal
from torch import Tensor
from typing_extensions import TypeAlias
import copy

StateDict: TypeAlias = Dict[str, torch.Tensor]


def topk_values_mask(
        M: Tensor,
        K: float = 0.7,
        return_mask: bool = False,
):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)


def resolve_zero_signs(
        sign_to_mult: Tensor,
        method: Literal["majority", "minority"] = "majority",
):
    majority_sign = torch.sign(sign_to_mult.sum())

    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult


def resolve_sign(tensor: Tensor):
    sign_to_mult = torch.sign(tensor.sum(dim=0))
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult


def disjoint_merge(tensor, merge_func, sign_to_mult):
    merge_func = merge_func.split("-")[-1]

    # If sign is provided then we select the corresponding entries and aggregate.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(sign_to_mult.unsqueeze(0) > 0, tensor > 0, tensor < 0)
        selected_entries = tensor * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = tensor != 0
        selected_entries = tensor * rows_to_keep

    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(non_zero_counts, min=1)
    elif merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= sign_to_mult
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs


def ties_merging(
        flat_task_checks,
        reset_thresh=None,
        merge_func="",
):
    all_checks = flat_task_checks.clone()
    updated_checks, *_ = topk_values_mask(all_checks, K=reset_thresh, return_mask=False)
    print(f"RESOLVING SIGN")
    final_signs = resolve_sign(updated_checks)
    assert final_signs is not None

    print(f"Disjoint AGGREGATION: {merge_func}")
    merged_tv = disjoint_merge(updated_checks, merge_func, final_signs)

    return merged_tv


# Model conversion utils
def state_dict_to_vector(state_dict: StateDict, remove_keys: List[str] = []) -> torch.Tensor:
    """
    Converts a PyTorch state dict to a 1D tensor.

    Args:
        state_dict (Dict[str, torch.Tensor]): A PyTorch state dict.
        remove_keys (List[str], optional): A list of keys to remove from the state dict. Defaults to [].
    Returns:
        torch.Tensor: A 1D tensor containing all the values in the state dict, sorted by key.
    """
    shared_state_dict = state_dict
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector([value.reshape(-1) for key, value in sorted_shared_state_dict.items()])


def vector_to_state_dict(vector, state_dict, remove_keys: List[str] = []):
    # create a reference dict to define the order of the vector
    reference_dict = state_dict
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict["transformer.shared.weight"]
    return sorted_reference_dict


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


def ties_merging_func(task_vectors, task_wise_weights=None):
    if len(task_vectors) == 0:
        raise ValueError("task_vectors must not be empty")
    elif len(task_vectors) == 1:
        return task_vectors[0]
    else:
        # num * vectors
        topk = 20
        reference_tv = task_vectors[0]
        tv_flat_checks = torch.vstack([state_dict_to_vector(check, remove_keys=[]) for check in task_vectors])
        merged_tv = ties_merging(tv_flat_checks, reset_thresh=topk, merge_func="dis-sum")
        merged_tv = vector_to_state_dict(merged_tv, reference_tv, remove_keys=[])

        return {k: _fuse_weights(task_wise_weights, [merged_tv[k]]) for k in merged_tv.keys()}
