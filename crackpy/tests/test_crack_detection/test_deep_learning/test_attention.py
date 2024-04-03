import os
import unittest
import shutil
import tempfile

import numpy as np
import torch
import matplotlib.pyplot as plt

from crackpy.crack_detection.deep_learning.attention import SegGradCAM, ParallelNetsWithHooks, plot_overview
from crackpy.crack_detection.deep_learning.setup import Setup
from crackpy.crack_detection.model import get_model


class TestSegGradCAM(unittest.TestCase):

    def setUp(self):
        # Settings
        self.setup = Setup(data_path=os.path.join(  # '..', '..', '..', '..',
                                                  'test_data', 'crack_detection', 'raw'),
                           experiment='EBr10', side='left')

        self.setup.set_stages(['407'])
        self.setup.set_visu_layers(['down4', 'base', 'up1'])

        # Load the model
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_with_hooks = ParallelNetsWithHooks()
        model = get_model('ParallelNets', map_location=device)
        model_with_hooks.load_state_dict(model.state_dict())
        self.model = model_with_hooks.unet

        # Initialize Seg-Grad-CAM
        self.sgc = SegGradCAM(self.setup, self.model)

        # Load test data
        self.inputs, _ = self.setup.load_data()

    def test_calculate_cam(self):
        input_t = next(iter(self.inputs.values()))

        base_path = os.path.join(  # '..', '..', '..', '..',
                                 'test_data', 'crack_detection', 'output',
                                 'visualization', 'seggradcam')
        exp_output = torch.load(os.path.join(base_path, 'exp_output.pt'))
        exp_heatmap = np.load(os.path.join(base_path, 'exp_heatmap.npy'))

        # calculate output and features in forward pass
        act_output, act_heatmap = self.sgc(input_t)
        self.assertTrue(torch.allclose(exp_output.detach().to('cpu'), act_output.detach().to('cpu')))
        self.assertTrue(np.allclose(exp_heatmap.copy(), act_heatmap.copy(), rtol=1e-2))

    def test_plot_and_save(self):
        base_path = os.path.join(  # '..', '..', '..', '..',
                                 'test_data', 'crack_detection', 'output',
                                 'visualization', 'seggradcam')
        output = torch.load(os.path.join(base_path, 'exp_output.pt'))
        heatmap = np.load(os.path.join(base_path, 'exp_heatmap.npy'))

        # plot qualitative
        fig = self.sgc.plot(output, heatmap, scale='QUALITATIVE')

        # save
        temp_dir = tempfile.mkdtemp()
        try:
            self.setup.set_output_path(temp_dir)
            self.sgc.save(key='AllDataPoints_1_407.txt_left', fig=fig)
        finally:
            shutil.rmtree(temp_dir)

        # plot quantitative
        fig = self.sgc.plot(output, heatmap, scale='QUALITATIVE')
        plt.close(fig)

    def test_plot_overview(self):
        input_t = next(iter(self.inputs.values()))
        output = self.model(input_t)

        seg_grad_cams = {}
        for name in self.setup.visu_layers:
            seg_grad_cams[name] = SegGradCAM(self.setup, self.model, name)

        # calculate heatmaps
        heatmaps = {}
        for name, seg_grad_cam in seg_grad_cams.items():
            _, heatmap = seg_grad_cam(input_t)
            heatmaps[name] = heatmap

        # plot heatmap
        fig = plot_overview(output=output, maps=heatmaps, side=self.setup.side, scale='QUALITATIVE')
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
