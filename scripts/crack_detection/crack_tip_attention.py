"""

    Example script:
        Crack Tip Attention heatmaps using Seg-Grad-CAM

    Needed:
        - Nodemap file

    Output:
        - Attention heatmaps for the crack tip detection as plot

"""

# Imports
import os

import torch
import matplotlib.pyplot as plt

from crackpy.crack_detection.model import get_model
from crackpy.crack_detection.deep_learning import setup, attention
from crackpy.crack_detection.detection import CrackDetection
from crackpy.fracture_analysis.data_processing import InputData
from crackpy.structure_elements.data_files import Nodemap


# Paths
NODEMAP_FILE = 'Dummy2_WPXXX_DummyVersuch_2_dic_results_1_52.txt'
DATA_PATH = os.path.join('..', '..', 'test_data', 'crack_detection', 'Nodemaps')
OUTPUT_PATH = 'attention'

# Setup
det = CrackDetection(
    side='right',
    detection_window_size=30,
    offset=(5, 0),
    angle_det_radius=10,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
)

setup = setup.Setup()
setup.side = det.side
setup.set_size(det.detection_window_size, det.offset)
setup.set_output_path(OUTPUT_PATH)
setup.set_visu_layers(['down1', 'down2', 'down3', 'down4', 'base', 'up1', 'up2', 'up3', 'up4'])

# Load the model
model_with_hooks = attention.ParallelNetsWithHooks()
model = get_model('ParallelNets')
model_with_hooks.load_state_dict(model.state_dict())
model_with_hooks = model_with_hooks.unet

# Get nodemap data
nodemap = Nodemap(name=NODEMAP_FILE, folder=DATA_PATH)
data = InputData(nodemap)

# Interpolate data on arrays (256 x 256 pixels)
interp_disps, _ = det.interpolate(data)

# Preprocess input
input_ch = det.preprocess(interp_disps)

# Initialize Seg-Grad-CAM
sgc = attention.SegGradCAM(setup, model_with_hooks)

# Forward pass with hooks to catch the features and gradients
output, heatmap = sgc(input_ch)

# Plot and save heatmap
fig = sgc.plot(output, heatmap)
plt.savefig(os.path.join(OUTPUT_PATH, NODEMAP_FILE[:-4] + '_attention.png'), dpi=300)
plt.close(fig)
