import os
import shutil
import unittest
import tempfile

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from crackpy.crack_detection.data.dataset import CrackTipDataset
from crackpy.crack_detection.data import transforms
from crackpy.crack_detection.deep_learning import train, nets, loss
from crackpy.crack_detection.deep_learning.docu import Documentation


class TestTrainingUNet(unittest.TestCase):

    def setUp(self):
        # Data paths
        origin = os.path.join(  # '..', '..', '..', '..',
                              'test_data', 'crack_detection', 'interim')

        train_input = os.path.join(origin, 'lInputData_right.pt')
        train_label = os.path.join(origin, 'lGroundTruthData_right.pt')

        val_input = os.path.join(origin, 'lInputData_right.pt')
        val_label = os.path.join(origin, 'lGroundTruthData_right.pt')

        # Data transforms
        self.trsfs = {
            'train': Compose([transforms.InputNormalization(),
                              transforms.CrackTipNormalization(),
                              transforms.ToCrackTipMasks()
                              ]),
            'val': Compose([transforms.InputNormalization(),
                            transforms.CrackTipNormalization(),
                            transforms.ToCrackTipMasks()
                            ])
        }

        # Data sets
        self.datasets = {
            'train': CrackTipDataset(inputs=train_input, labels=train_label,
                                     transform=self.trsfs['train']),
            'val': CrackTipDataset(inputs=val_input, labels=val_label,
                                   transform=self.trsfs['val'])
        }
        self.dataloaders = {
            'train': DataLoader(self.datasets['train'], batch_size=8, shuffle=True),
            'val': DataLoader(self.datasets['val'], batch_size=8, shuffle=False)
        }
        self.sizes = {x: len(self.datasets[x]) for x in ['train', 'val']}

    def test_unet_training_and_documentation(self):
        model = nets.UNet(init_features=16)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Training
        criterion = loss.DiceLoss()
        optimizer = optim.Adam(model.parameters())

        model, docu = train.train_segmentation_model(model, self.dataloaders, self.sizes, criterion,
                                                     optimizer, device=device)

        # Docu
        documentation = Documentation(transforms=self.trsfs,
                                      datasets=self.datasets,
                                      dataloaders=self.dataloaders,
                                      model=model,
                                      criterion=criterion,
                                      optimizer=optimizer,
                                      train_docu=docu)

        temp_dir = tempfile.mkdtemp()
        try:
            documentation.save_metadata(path=temp_dir, name='temp_file')
        finally:
            shutil.rmtree(temp_dir)


class TestTrainingParallelNets(unittest.TestCase):

    def setUp(self):
        # Data paths
        origin = os.path.join(  # '..', '..', '..', '..',
                              'test_data', 'crack_detection', 'interim')

        train_input = os.path.join(origin, 'lInputData_right.pt')
        train_label = os.path.join(origin, 'lGroundTruthData_right.pt')

        val_input = os.path.join(origin, 'lInputData_right.pt')
        val_label = os.path.join(origin, 'lGroundTruthData_right.pt')

        # Data transforms
        self.trsfs = {
            'train': Compose([transforms.InputNormalization(),
                              transforms.CrackTipNormalization(),
                              transforms.ToCrackTipsAndMasks()
                              ]),
            'val': Compose([transforms.InputNormalization(),
                            transforms.CrackTipNormalization(),
                            transforms.ToCrackTipsAndMasks()
                            ])
        }

        # Data sets
        self.datasets = {
            'train': CrackTipDataset(inputs=train_input, labels=train_label,
                                     transform=self.trsfs['train']),
            'val': CrackTipDataset(inputs=val_input, labels=val_label,
                                   transform=self.trsfs['val'])
        }
        self.dataloaders = {
            'train': DataLoader(self.datasets['train'], batch_size=8, shuffle=True),
            'val': DataLoader(self.datasets['val'], batch_size=8, shuffle=False)
        }
        self.sizes = {x: len(self.datasets[x]) for x in ['train', 'val']}

    def test_parallel_nets_training_and_documentation(self):
        model = nets.ParallelNets(init_features=16)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Training
        weights = (1, 100)
        criterion = [loss.DiceLoss(), nn.MSELoss(), weights]
        optimizer = optim.Adam(model.parameters())

        model, docu = train.train_parallel_model(model, self.dataloaders, self.sizes, criterion,
                                                 optimizer, device=device)

        # Docu
        documentation = Documentation(transforms=self.trsfs,
                                      datasets=self.datasets,
                                      dataloaders=self.dataloaders,
                                      model=model,
                                      criterion=criterion,
                                      optimizer=optimizer,
                                      train_docu=docu)

        temp_dir = tempfile.mkdtemp()
        try:
            documentation.save_metadata(path=temp_dir, name='temp_file')
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
