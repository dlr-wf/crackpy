import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    """Dice loss for unbalanced binary segmentation problems.

    Methods:
        * forward - applies the criterion to a batch of predictions and corresponding targets

    """

    def __init__(self, eps: float=1e-6):
        """Dice loss for unbalanced binary segmentation problems.

        Args:
            eps: regularization parameter close to zero

        """
        super().__init__()
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Applies the criterion to a batch of predictions and corresponding targets.

        Args:
            prediction: tensor of size (B x C x H x W)
            target: tensor of size (B x C x H x W)

        Returns:
            dice loss = 1 - dice coefficient

        """
        if not prediction.size() == target.size() and len(prediction.size()) == 4:
            raise AssertionError("'prediction' and 'target' need to have length 4 and the same size")
        prediction = prediction[:, 0].contiguous().view(-1)
        target = target[:, 0].contiguous().view(-1)
        intersection = (prediction * target).sum()
        dsc = (2. * intersection + self.eps) / (prediction.sum() + target.sum() + self.eps)
        return 1. - dsc


class MSELoss(nn.Module):
    """Mean Squared Error loss with (global) weight factor.

    Methods:
        * forward - applies the criterion to a batch of predictions and corresponding targets

    """

    def __init__(self, weight_factor: float=1.0, reduction: str='mean'):
        """Mean Squared Error loss with (global) weight factor.

        Args:
            weight_factor: factor by which the loss is multiplied
            reduction: 'mean' or 'sum'

        """
        super().__init__()
        self.weight_factor = weight_factor
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Applies the criterion to a batch of predictions and corresponding targets.

        Args:
            prediction: tensor of size (B x C x H x W)
            target: tensor of size (B x H x W)

        Returns:
            weighted MSE loss

        """
        return self.weight_factor * F.mse_loss(prediction, target, reduction=self.reduction)
