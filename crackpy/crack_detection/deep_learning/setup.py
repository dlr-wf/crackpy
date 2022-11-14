import os

import torch

from crackpy.crack_detection.data.datapreparation import import_data
from crackpy.crack_detection.data.interpolation import interpolate_on_array
from crackpy.crack_detection.utils.basic import concatenate_np_dicts, numpy_to_tensor
from crackpy.crack_detection.utils.utilityfunctions import get_nodemaps_and_stage_nums


class Experiments:
    """Settings for the currently used experimental data."""

    def __init__(self):
        """Dictionaries with experiment names as keys are initialized."""

        self.nodemap_nums = {
            'EBr2': [str(i) for i in range(100, 303, 2)],  # at maximum force
            'EBr10': [str(i) for i in range(7, 838, 5)],
            'EBr11': [str(i) for i in range(20, 1429, 5)],
            '2BE3F': [str(i) for i in range(67, 129, 3)],
            '2MB10D': [str(i) for i in range(20, 40, 1)],
            'Dummy1': [str(i) for i in range(7, 353, 3)],
            'Dummy2': [str(i) for i in range(7, 353, 3)],
            'Dummy2_Mic': [str(i) for i in range(5, 57, 3)],
            'Dummy2_Mic2': [str(i) for i in range(5, 57, 3)],
            '1219UAl002-LOP': [str(i) for i in range(7, 48, 2)],
            '1221EBr00007': [str(i) for i in range(1, 839, 1)],
            'Aramis_in_line': []  # generic aramis experiment for in-line crack detection
        }
        self.sizes = {
            'EBr2': 450,  # mm
            'EBr10': 70,
            'EBr11': 70,
            '2BE3F': 180,
            '2MB10D': 180,
            'Dummy1': 70,
            'Dummy2': 70,
            '1221EBr00007': 5.5,
            'Dummy2_Mic': 5.5,
            'Dummy2_Mic2': 5.5,
            '1219UAl002-LOP': 25
        }
        self.offsets = {
            'EBr2': (0, 0),  # mm
            'EBr10': (0, 0),
            'EBr11': (0, 0),
            '2BE3F': (0, 0),
            '2MB10D': (0, 0),
            'Dummy1': (0, 0),
            'Dummy2': (0, 0),
            'Dummy2_Mic': (33, 0),
            'Dummy2_Mic2': (3, 0),
            '1219UAl002-LOP': (0, 0)
        }
        self.exists_target = {
            'EBr2': False,
            'EBr10': True,
            'EBr11': False,
            '2BE3F': False,
            '2MB10D': False,
            'Dummy1': False,
            'Dummy2': False,
            'Dummy2_Mic': False,
            'Dummy2_Mic2': False,
            '1219UAl002-LOP': False,
            '1221EBr00007': False
        }


class Setup(Experiments):
    """General settings for data, models, etc.

    Methods:
        * load_data - loads data from information given in setup
        * set_model - set the model path and name of this setup
        * set_stages - change stages different from the default setup
        * set_size - change view window size and offset in mm
        * set_if_target_exists - set if target exists or not
        * set_visu_layers - set layers for visualization with Grad-CAM, Seg-Grad-CAM, etc.
        * set_out_ids - set Output layer names to id's dictionary (only used for GradCAM with ResNet)
        * set_output_path - sets the output path for the setup (and creates the folder)

    """
    def __init__(self, data_path: str or None=None, experiment: str=None, side: str='right'):
        """Initialize default setup class and sets data path attribute.

        Args:
            data_path: with sub-folders 'Nodemaps' and 'GroundTruth' (if exists)
            experiment: name of the experiment (needs to be in Experiments)
            side: 'left' or 'right'

        """
        super().__init__()

        # experiment, model and paths
        self.experiment = experiment
        self.side = side

        if data_path is not None and os.path.exists(data_path):
            self.data_path = data_path
        elif data_path is not None:
            raise ValueError("The chosen data path does not seem to exist!")

        self.output_path = None
        self.size = self.sizes[self.experiment] if self.experiment in self.sizes else None
        self.offset = self.offsets[self.experiment] if self.experiment in self.offsets else None
        self.target = self.exists_target[self.experiment] if self.experiment in self.exists_target else None

        self.model_name = None
        self.model_path = None

        self.visu_layers = None

        self.out_ids = None

        if self.experiment in self.nodemap_nums:
            self.stages_to_nodemaps, self.nodemaps_to_stages = \
                get_nodemaps_and_stage_nums(os.path.join(self.data_path, 'Nodemaps'),
                                            self.nodemap_nums[self.experiment])
        else:
            self.stages_to_nodemaps, self.nodemaps_to_stages = {}, {}

    def load_data(self, with_eps_vm: bool=False):
        """Loads data from information given in setup.
        Returns inputs & targets as dictionaries sorted by nodemap keys.

        Args:
            with_eps_vm: whether the von Mises strain should be third input channel

        Returns:
            (dicts of torch.tensors) of shape (B, C, H, W), (B, H, W) or None, with nodemap_name + _side as keys

        """
        # read test data
        input_data, ground_truth = import_data(nodemaps=self.stages_to_nodemaps,
                                               data_path=self.data_path,
                                               side=self.side,
                                               exists_target=self.target)
        # interpolate
        interp_size = self.size if self.side == 'right' else self.size * -1
        _, interp_disps, interp_eps_vm = interpolate_on_array(input_by_nodemap=input_data,
                                                              interp_size=interp_size,
                                                              offset=self.offset,
                                                              pixels=256)
        # get inputs
        inputs = interp_disps
        if with_eps_vm:
            # the ResNetFC model is trained with a percent scale!!
            interp_eps_vm = {key: value / 100. for key, value in interp_eps_vm.items()}
            inputs = concatenate_np_dicts(inputs, interp_eps_vm)
        inputs = numpy_to_tensor(inputs, dtype=torch.float32)

        # get targets (if exists)
        targets = numpy_to_tensor(ground_truth, dtype=torch.int64) if self.target else None

        return inputs, targets

    def set_model(self, model_path: str, model_name: str):
        """Set the model path and name of this setup.

        Args:
            model_path: path of the model
            model_name: name of the model

        """
        if os.path.exists(model_path):
            self.model_path = model_path
        else:
            raise ValueError("The model path does not seem to exist!")

        self.model_name = model_name

    def set_stages(self, stages):
        """Change stages different from the default setup.

        Args:
            stages: (list of str of ints or 'All') indicating which stages are used, e.g. [7, 35, 100]

        """
        if stages != 'All':
            stages = [str(i) for i in stages]
        self.stages_to_nodemaps, self.nodemaps_to_stages = \
            get_nodemaps_and_stage_nums(os.path.join(self.data_path, 'Nodemaps'), stages)

    def set_size(self, size: int or float, offset: tuple):
        """Change size in mm

        Args:
            size: of the window of view
            offset: (x,y)-offset of the window of view

        """
        self.size = size
        self.offset = offset

    def set_if_target_exists(self, target_exists: bool):
        """Set if target exists or not.

        Args:
            target_exists: whether a target / ground truth exists

        """
        self.target = target_exists

    def set_visu_layers(self, layers: list):
        """Set layers for visualization with Grad-CAM, Seg-Grad-CAM, etc.

        Args:
            layers: layer names as str, e.g. ['down4', 'base', 'up1'] or ['layer4']

        """
        self.visu_layers = layers

    def set_out_ids(self, out_to_ids: dict):
        """Set Output layer names to id's dictionary. This is only used for GradCAM with ResNet.

        Args:
            out_to_ids: dictionary relating output channels to IDs, e.g. {'x': 1, 'y': 0}

        """
        self.out_ids = out_to_ids

    def set_output_path(self, path: str):
        """Sets the output path for the setup and creates the folder (if it not exists).

        Args:
            path: output path

        """
        self.output_path = path

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
