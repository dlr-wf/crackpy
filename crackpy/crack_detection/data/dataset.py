"""
    This module follows the pytorch tutorial
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class CrackTipDataset(Dataset):
    """Specific dataset for fatigue crack detection data.

    The class gets paths to preprocessed DIC data and corresponding labels (if these are provided).
    It is also possible to provide data transforms.

    Methods:
        * get_cracktips - calculate crack tip pixel positions
        * calculate_means_and_stds - calculate means and standard deviations for each input channel

    """

    def __init__(self, inputs, labels=None, transform=None):
        # Crack tip dataset gets list of inputs with list of respective labels.
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(labels, list):
            labels = [labels]

        self.data_paths_input = inputs
        self.data_paths_label = labels

        self.inputs = torch.cat([in_tensor
                                 for i in [torch.load(inp) for inp in inputs]
                                 for in_tensor in i]
                                )

        if self.data_paths_label[0] is not None:
            self.targets = torch.cat([out_tensor
                                      for i in [torch.load(out) for out in labels]
                                      for out_tensor in i]
                                     )
            self.cracktips = self.get_cracktips()

        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, num):
        image = self.inputs[num]

        if self.data_paths_label[0] is not None:
            crack = self.targets[num]
            tip = self.cracktips[num]
            sample = {'input': image, 'target': crack, 'tip': tip}
        else:
            sample = {'input': image}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_cracktips(self):
        """Calculate the crack tip pixel positions from provided labels."""
        condition = torch.BoolTensor(self.targets == 2)
        is_cracktip = torch.where(condition, 1, 0)
        cracktip_positions = torch.nonzero(is_cracktip, as_tuple=False)[:, 1:3] / 1.

        return cracktip_positions

    def calculate_means_and_stds(self):
        """Calculate means and standard deviations for each input channel."""
        inputs_as_array = np.asarray(self.inputs)
        means = inputs_as_array.mean(axis=(0, 2, 3))
        stds = inputs_as_array.std(axis=(0, 2, 3))

        return means, stds
