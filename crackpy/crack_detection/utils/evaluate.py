import numpy as np
import torch

from crackpy.crack_detection.data.transforms import denormalize_crack_tips


def get_deviation(outputs: torch.Tensor, labels: torch.Tensor, shape: tuple or list):
    """Calculates the sum of Euklidean distances of the predictions to the **labels**.

    Args:
        outputs: tensor of shape (B x 2)
        labels: tensor of the same shape
        shape: size of the inputs (needed for de-normalization)

    Returns:
        sum of Euklidean distances between predictions and labels

    """
    if not outputs.size() == labels.size():
        raise AssertionError("Output and Labels need to have the same torch.Size!")

    # denormalize labels and predictions
    labels = denormalize_crack_tips(labels, shape)
    predictions = denormalize_crack_tips(outputs, shape)

    # calculate Euklidean distance
    dist_euklid = torch.sum(torch.sqrt(torch.sum((predictions - labels) ** 2, dim=1))).item()

    return dist_euklid


def get_segmentation_deviation(outputs: torch.Tensor, labels: torch.Tensor):
    """Calculates the sum of Euklidean distances of the predictions to the **labels** (if both exist).
    The prediction is calculated by taking the mean of all segmentated pixels.
    If there is no segmented pixel or no labeled pixel, the sample is skipped.

    Args:
        outputs: tensor of shape (B x 1 x W x H) (after sigmoid/softmax)
        labels: tensor of the same shape

    Returns:
        sum of Euklidean distances between predictions and labels

    """
    if not outputs.size() == labels.size():
        raise AssertionError("Output and Labels need to have the same torch.Size!")

    batch_size = outputs.shape[0]
    is_crack_tip = torch.where(outputs >= 0.5, 1, 0)

    dist_euklid = []
    for i in range(batch_size):
        prediction_i = torch.nonzero(is_crack_tip[i], as_tuple=False)[:, -2:] / 1.
        label_i = torch.nonzero((labels[i] == 1), as_tuple=False)[:, -2:] / 1.
        # skip the unlabeled or unsegmented
        if len(label_i) == 0 or len(prediction_i) == 0:
            continue
        prediction_i = torch.mean(prediction_i, dim=0)
        label_i = torch.mean(label_i, dim=0)
        dist = torch.sqrt(torch.sum((prediction_i - label_i) ** 2)).item()
        dist_euklid.append(dist)

    return np.asarray(dist_euklid)


def get_reliability(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculates the reliability of a segmentation model's output batch and the corresponding labels.

    Args:
        outputs: tensor output of the model (after Sigmoid/Softmax) (B x H x W)
        labels: tensor corresponding labels (B x H x W) with 1's and 0's

    Returns:
        reliability score (1.0 = 100 %)

    """
    if not outputs.size() == labels.size():
        raise AssertionError("Output and Labels need to have the same torch.Size!")

    batch_size = outputs.shape[0]
    is_crack_tip = torch.where(outputs >= 0.5, 1, 0)

    unpredicted = 0
    for i in range(batch_size):
        prediction_i = torch.nonzero(is_crack_tip[i], as_tuple=False)[:, -2:] / 1.
        label_i = torch.nonzero((labels[i] == 1), as_tuple=False)[:, -2:] / 1.
        # skip the unlabeled or unsegmented
        if len(label_i) > 0 and len(prediction_i) == 0:
            unpredicted += 1

    score = 1. - unpredicted / batch_size

    return score
