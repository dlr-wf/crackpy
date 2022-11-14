import torch
import numpy as np


def normalize(input_t: torch.Tensor or np.array) -> torch.Tensor:
    """Normalize every input channel to mean 0 and variance 1 for each input.

    Args:
        input_t: tensor or array of shape (.., W, H)

    Returns:
        input_normalized

    """
    input_as_array = np.asarray(input_t)

    means = np.nanmean(input_as_array, axis=(-2, -1), keepdims=True)
    stds = np.nanstd(input_as_array, axis=(-2, -1), keepdims=True)

    input_normalized = (input_as_array - means) / stds
    input_normalized = torch.from_numpy(input_normalized)

    return input_normalized


def target_to_crack_tip_position(target: torch.Tensor) -> torch.Tensor:
    """Extracts crack-tip positions from segmentation ground truth targets.

    Args:
        target: tensor of shape (W, H)

    Returns:
        crack_tip_positions tensor of size 2

    """
    pixel_position = torch.nonzero((target == 2), as_tuple=False)
    crack_tip_position = pixel_position / 1.
    return crack_tip_position
