import os

import numpy as np
import torch
from scipy.ndimage import label


def get_nodemaps_and_stage_nums(folder_path: str, which='All'):
    """Generates two dictionaries with stage numbers as keys and nodemap filenames as values and vice versa.

    Args:
        folder_path: path of the nodemaps folder
        which: (list) of stage numbers or 'All' for all nodemaps in *folder_path*

    Returns:
        (dicts) stage_num_to_filename, filename_to_stage_num
    """
    if which == 'All':
        list_of_filenames = os.listdir(folder_path)
        which = [name.split('_')[-1].strip('.txt') for name in list_of_filenames]
    assert isinstance(which, (list, range)), 'Argument "which" should be a list of integers or "All".'

    nodemap_without_num = '_'.join(os.listdir(folder_path)[0].split('_')[:-1])
    stage_num_to_filename = {}
    filename_to_stage_num = {}
    for stage in which:
        num = int(stage)
        name = nodemap_without_num + f'_{num}.txt'
        stage_num_to_filename[num] = name
        filename_to_stage_num[name] = num
        filename_to_stage_num[name + '_left'] = num
        filename_to_stage_num[name + '_right'] = num

    return stage_num_to_filename, filename_to_stage_num


def calculate_segmentation(output_mask: torch.Tensor) -> torch.Tensor:
    """Calculates the crack tip positions of all segmented pixels of an output mask.

    Args:
        output_mask: tensor of shape (.. x H x W)

    Returns:
        crack_tip_seg tensor of shape num_of_seg x 2

    """
    condition = torch.BoolTensor(output_mask >= 0.5)
    is_crack_tip = torch.where(condition, 1, 0)
    crack_tip_seg = torch.nonzero(is_crack_tip, as_tuple=False)[:, -2:] / 1.
    return crack_tip_seg


def find_most_likely_tip_pos(out_prob: torch.Tensor):
    """Detects CrackTip Regions and selects the one with the highest mean probability.
    Then chooses its gravity centre as crack tip coordinate.

    Args:
        out_prob: Probabilities of each class (batch-size x classes x array_size x array_size)

    Returns:
        (floats) x_mean, y_mean indicating the mean crack tip position of the most likely region.
    """
    out_prob = out_prob.numpy()
    crack_tip_pixels = np.where(out_prob >= 0.5, 1, 0)

    # finds regions which are connected
    labels, num_of_labels = label(crack_tip_pixels)

    if num_of_labels == 0:
        # no crack tips detected
        print("\nCrack detection failed. This might happen due to a wrong detection window size,\n"
              "which can be set in the 'crackDetectionSetup' class. Make sure that the possible crack tip\n"
              "is not located close to the boundary of that window. Also make sure that no larger regions of\n"
              "NaNs lie within this window.")
        return [np.nan, np.nan]

    region_instance = 0
    region_prob = 0.0
    for i in range(1, num_of_labels + 1):
        crack_tip_probs = np.multiply(np.where(labels == i, 1, 0), out_prob.squeeze())
        num_of_pixels = np.sum(np.where(labels == i, 1, 0))
        mean_prob = np.sum(crack_tip_probs) / num_of_pixels
        if mean_prob >= region_prob:
            region_instance = i

    pixels = out_prob.shape[-1]
    coors = np.linspace(0, pixels, pixels)
    x_coors, y_coors = np.meshgrid(coors, coors)
    mask = np.where(labels == region_instance, 1, 0)
    mask = mask * out_prob

    x_mean = np.sum(np.multiply(x_coors, mask)) / np.sum(mask)
    y_mean = np.sum(np.multiply(y_coors, mask)) / np.sum(mask)

    return [y_mean, x_mean]
