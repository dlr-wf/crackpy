import torch
import numpy as np


def concatenate_np_dicts(inputs: dict, eps_vm: dict) -> dict:
    """Join inputs and eps_vm into one dictionary with nodemap keys.
    Notice that eps_vm has one dimension less than inputs which needs to be expanded.

    Args:
        inputs: dictionary of array
        eps_vm: dictionary of array with shape of inputs -1 dimension

    Returns:
        dictionary of array of concatenated inputs and eps_vm arrays

    """
    assert set(inputs.keys()) == set(eps_vm.keys())
    joined = {}
    for key, current_inputs in inputs.items():
        current_eps_vm = eps_vm[key]
        current_eps_vm = np.expand_dims(current_eps_vm, axis=0)
        joined[key] = np.concatenate((current_inputs, current_eps_vm), axis=0)
    return joined


def numpy_to_tensor(numpy_dict, dtype):
    """Convert a dict of numpy arrays into a dict of 'unsqueezed' tensors of 'dtype'."""
    return {key: torch.tensor(value.copy(), dtype=dtype).unsqueeze(0)
            for key, value in numpy_dict.items()}


def dict_to_list(dictionary) -> list:
    """Convert a dictionary into a list by loosing the keys."""
    return [value for key, value in dictionary.items()]
