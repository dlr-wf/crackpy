"""

Application Testing of crack detection pipeline with all functionalities.

"""
import os
import shutil
import tempfile
import unittest

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from crackpy.crack_detection.data import transforms, preprocess
from crackpy.crack_detection.data import datapreparation as dp
from crackpy.crack_detection.data.dataset import CrackTipDataset
from crackpy.crack_detection.deep_learning import nets, loss, train
from crackpy.crack_detection.deep_learning.docu import Documentation
from crackpy.crack_detection.data.interpolation import interpolate_on_array
from crackpy.crack_detection.utils.basic import numpy_to_tensor, dict_to_list
from crackpy.crack_detection.utils.plot import plot_prediction
from crackpy.crack_detection.utils import evaluate
from crackpy.crack_detection.utils import utilityfunctions as uf
from crackpy.crack_detection.deep_learning.setup import Setup
from crackpy.crack_detection.deep_learning.attention import SegGradCAM, UNetWithHooks


class TestCrackDetection(unittest.TestCase):

    def setUp(self):
        self.origin = os.path.join(  # '..',
                                   'test_data', 'crack_detection')
        self.raw_data_path = os.path.join(self.origin, 'raw')
        self.interim_data_path = None

        self.temp_dir = tempfile.mkdtemp()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.stages_to_nodemaps = None
        self.val_input = None
        self.val_label = None
        self.docu = None

    def test_crack_detection_pipeline(self):
        """
        We test the hole crack detection pipeline from generating the intermediate data to training
        and testing the model and visualizing the results.
        """
        try:
            self._generate_interim_data()
            self._train_model()
            self._test_model()
            self._plot_output()
            self._plot_seg_grad_cam()

        finally:
            shutil.rmtree(self.temp_dir)

    def _generate_interim_data(self):
        # data generation
        self.interim_data_path = os.path.join(self.temp_dir, 'interim')
        if not os.path.exists(self.interim_data_path):
            os.makedirs(self.interim_data_path)

        self.stages_to_nodemaps, _ = uf.get_nodemaps_and_stage_nums(
            os.path.join(self.raw_data_path, 'Nodemaps'), 'All')

        for side in ['left']:
            # import
            inputs, ground_truths = dp.import_data(nodemaps=self.stages_to_nodemaps,
                                                   data_path=self.raw_data_path,
                                                   side=side,
                                                   exists_target=True)

            # interpolate
            interp_size = 70 if side == 'right' else -70
            _, interp_disps, _ = interpolate_on_array(input_by_nodemap=inputs,
                                                      interp_size=interp_size,
                                                      pixels=256)

            # get inputs
            inputs = interp_disps
            inputs = numpy_to_tensor(inputs, dtype=torch.float32)
            inputs = dict_to_list(inputs)

            # save inputs
            torch.save(inputs, os.path.join(self.interim_data_path, f'lInputData_{side}.pt'))

            # get targets
            targets = numpy_to_tensor(ground_truths, dtype=torch.int64)
            targets = dict_to_list(targets)

            # save targets
            torch.save(targets, os.path.join(self.interim_data_path, f'lGroundTruthData_{side}.pt'))

    def _train_model(self):
        # Data
        train_input = os.path.join(self.interim_data_path, 'lInputData_left.pt')
        train_label = os.path.join(self.interim_data_path, 'lGroundTruthData_left.pt')

        self.val_input = os.path.join(self.interim_data_path, 'lInputData_left.pt')
        self.val_label = os.path.join(self.interim_data_path, 'lGroundTruthData_left.pt')

        # Data transforms
        trsfs = {
            'train': Compose([transforms.InputNormalization(),
                              transforms.ToCrackTipMasks()
                              ]),
            'val': Compose([transforms.InputNormalization(),
                            transforms.ToCrackTipMasks()
                            ])
        }

        # Data sets
        datasets = {
            'train': CrackTipDataset(inputs=train_input, labels=train_label,
                                     transform=trsfs['train']),
            'val': CrackTipDataset(inputs=self.val_input, labels=self.val_label,
                                   transform=trsfs['val'])
        }
        dataloaders = {
            'train': DataLoader(datasets['train'], batch_size=8, shuffle=True),
            'val': DataLoader(datasets['val'], batch_size=8, shuffle=False)
        }
        sizes = {x: len(datasets[x]) for x in ['train', 'val']}

        # Training
        model = nets.UNet(init_features=16)
        model = model.to(self.device)

        # Training
        criterion = loss.DiceLoss()
        optimizer = optim.Adam(model.parameters())

        model, self.docu = train.train_segmentation_model(model, dataloaders, sizes, criterion,
                                                          optimizer, device=self.device,
                                                          num_epochs=20)

        # Saving model
        path = os.path.join(self.temp_dir, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model.state_dict(), os.path.join(path, 'model.pt'))

        # Save documentation
        documentation = Documentation(transforms=trsfs,
                                      datasets=datasets,
                                      dataloaders=dataloaders,
                                      model=model,
                                      criterion=criterion,
                                      optimizer=optimizer,
                                      train_docu=self.docu)
        documentation.save_metadata(path=os.path.join(self.temp_dir, 'model'), name='docu')

    def _test_model(self):
        # Load model
        model = nets.UNet(init_features=16)
        model.load_state_dict(torch.load(os.path.join(self.temp_dir, 'model', 'model.pt')))
        model.to(self.device)
        model.eval()

        # Load test input
        inputs = torch.cat(torch.load(os.path.join(self.val_input)))
        targets = torch.cat(torch.load(os.path.join(self.val_label)))
        # Convert to 1-hot
        is_tip = torch.BoolTensor(targets == 2)
        labels = torch.where(is_tip, 1, 0)
        labels = labels.unsqueeze(1)
        # Preprocess inputs
        inputs = preprocess.normalize(inputs)
        # Move to device
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # Predictions
        outputs = model(inputs)

        # calculate reliability and deviation
        act_rel = evaluate.get_reliability(outputs, labels)
        act_devs = evaluate.get_segmentation_deviation(outputs, labels) / len(outputs)
        act_dev = np.mean(act_devs).item()
        self.assertAlmostEqual(act_rel, self.docu.reliability, delta=1e-4)
        self.assertAlmostEqual(act_dev, self.docu.deviation, delta=1e-4)

    def _plot_output(self):
        # Data Import
        inputs, ground_truth = dp.import_data(nodemaps={407: self.stages_to_nodemaps[407]},
                                              data_path=self.raw_data_path,
                                              side='left',
                                              exists_target=True)

        # Interpolation
        _, interp_disps, interp_eps_vm = interpolate_on_array(input_by_nodemap=inputs,
                                                              interp_size=-70)

        # Preprocess
        nodemap = self.stages_to_nodemaps[407] + '_left'
        disps = interp_disps[nodemap]
        input_ch = torch.tensor(disps, dtype=torch.float32)
        input_ch = preprocess.normalize(input_ch).unsqueeze(0)
        target = torch.tensor(ground_truth[nodemap].copy(),
                              dtype=torch.float32)
        label = preprocess.target_to_crack_tip_position(target)

        # Load model
        model = nets.UNet(init_features=16)
        model.load_state_dict(torch.load(os.path.join(self.temp_dir, 'model', 'model.pt')))
        model.to(self.device)
        model.eval()

        # Predict
        output = model(input_ch.to(self.device))
        output = output.detach().to('cpu')

        # Calculate segmentation
        crack_tip_seg = uf.calculate_segmentation(output)
        # Take the mean value of the segmentation
        crack_tip_seg_mean = torch.mean(crack_tip_seg, dim=0).unsqueeze(0)

        # Plot and save
        plot_prediction(background=interp_eps_vm[nodemap] * 100,
                        interp_size=70,
                        save_name='plot_prediction',
                        crack_tip_prediction=crack_tip_seg_mean,
                        crack_tip_seg=crack_tip_seg,
                        crack_tip_label=label,
                        f_min=0,
                        f_max=0.68,
                        title=nodemap,
                        path=os.path.join(self.temp_dir, 'output'),
                        label='Von Mises Strain [%]')

    def _plot_seg_grad_cam(self):
        # Settings
        setup = Setup(data_path=self.raw_data_path, experiment='EBr10', side='left')
        setup.set_stages(['407'])
        setup.set_model(model_path=os.path.join(self.temp_dir, 'model'), model_name='model')
        setup.set_output_path(path=os.path.join(self.temp_dir, 'output'))
        setup.set_visu_layers(['down4', 'base', 'up1'])

        # Load the model
        model = UNetWithHooks(init_features=16)
        path = os.path.join(setup.model_path, setup.model_name + '.pt')
        model.load_state_dict(torch.load(path, map_location=self.device))

        # Load data
        inputs, _ = setup.load_data()
        key, input_t = next(iter(inputs.items()))
        input_t = preprocess.normalize(input_t)

        # Seg-Grad-CAM plot and save
        sgc = SegGradCAM(setup, model)
        output, heatmap = sgc(input_t)
        fig = sgc.plot(output, heatmap)
        sgc.save(key, fig)


if __name__ == '__main__':
    unittest.main()
