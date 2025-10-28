"""

    Example script:
        Crack Tip Attention heatmaps using Seg-Grad-CAM

    Needed:
        - Nodemap file

    Output:
        - Attention heatmaps for the crack tip detection as plot

"""

# Imports
from pathlib import Path
import logging
import torch
import matplotlib.pyplot as plt

from crackpy.crack_detection.model import get_model
from crackpy.crack_detection.deep_learning import setup, attention
from crackpy.crack_detection.detection import CrackDetection
from crackpy.input.input_data import InputData
from crackpy.structure_elements.data_files import Nodemap

# Logging
logger = logging.getLogger(__name__)


# Paths
PROJECT_ROOT = Path(__file__).parents[2]

NODEMAP_FILE = 'Dummy2_WPXXX_DummyVersuch_2_dic_results_1_52.txt'
DATA_PATH = PROJECT_ROOT / 'test_data' / 'crack_detection' / 'Nodemaps'
OUTPUT_PATH = PROJECT_ROOT / 'attention'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

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
setup.set_output_path(str(OUTPUT_PATH))
setup.set_visu_layers(['down1', 'down2', 'down3', 'down4', 'base', 'up1', 'up2', 'up3', 'up4'])

# Load the model
model_with_hooks = attention.ParallelNetsWithHooks()
model = get_model('ParallelNets')
model_with_hooks.load_state_dict(model.state_dict())
model_with_hooks = model_with_hooks.unet

# Get nodemap data
nodemap = Nodemap(name=NODEMAP_FILE, folder=str(DATA_PATH))
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
plt.rcParams['image.cmap'] = 'coolwarm'
fig = sgc.plot(output, heatmap)
plt.savefig(str(OUTPUT_PATH / (Path(NODEMAP_FILE).stem + '_attention.png')), dpi=300)
plt.close(fig)
