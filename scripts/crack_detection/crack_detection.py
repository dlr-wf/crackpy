"""

    Example script:
        Crack Detection (path, tip, angle) for a single nodemap

    Needed:
        - Nodemap

    Output:
        - Crack tip position
        - Crack path
        - Crack angle
        - Plot of prediction

"""

# Imports
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from crackpy.crack_detection.model import get_model
from crackpy.crack_detection.utils.plot import plot_prediction
from crackpy.crack_detection.detection import CrackTipDetection, CrackPathDetection, CrackAngleEstimation, CrackDetection
from crackpy.fracture_analysis.data_processing import InputData
from crackpy.structure_elements.data_files import Nodemap

# Settings
NODEMAP_FILE = 'Dummy2_WPXXX_DummyVersuch_2_dic_results_1_52.txt'
DATA_PATH = os.path.join('..', '..', 'test_data', 'crack_detection', 'Nodemaps')
OUTPUT_PATH = 'prediction'

det = CrackDetection(
    side='right',
    detection_window_size=30,
    offset=(5, 0),
    angle_det_radius=10,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
)

# Get nodemap data
nodemap = Nodemap(name=NODEMAP_FILE, folder=DATA_PATH)
data = InputData(nodemap)

# Interpolate data on arrays (256 x 256 pixels)
interp_disps, interp_eps_vm = det.interpolate(data)

# Preprocess input
input_ch = det.preprocess(interp_disps)

#####################
# Tip detection
#####################
# Load crack tip detector
tip_detector = get_model('ParallelNets')
ct_det = CrackTipDetection(detection=det, tip_detector=tip_detector)

# Make prediction
pred = ct_det.make_prediction(input_ch)

# Calculate segmentation and most likely crack tip position
crack_tip_seg = ct_det.calculate_segmentation(pred)
crack_tip_pixels = ct_det.find_most_likely_tip_pos(pred)

# Calculate global crack tip positions in mm
crack_tip_x, crack_tip_y = ct_det.calculate_position_in_mm(crack_tip_pixels)

print(f"Crack tip x [mm]: {crack_tip_x}")
print(f"Crack tip y [mm]: {crack_tip_y}")

#####################
# Path detection
#####################
# load crack path detector model
path_detector = get_model('UNetPath')
cp_det = CrackPathDetection(detection=det, path_detector=path_detector)

# predict segmentation and path skeleton
cp_segmentation, cp_skeleton = cp_det.predict_path(input_ch)

##########################
# Crack angle estimation
##########################
angle_est = CrackAngleEstimation(detection=det, crack_tip_in_px=crack_tip_pixels)

# Consider only crack path close to crack tip
cp_segmentation_masked = angle_est.apply_circular_mask(cp_segmentation)

# Filter for largest connected crack path
cp_segmentation_largest_region = angle_est.get_largest_region(cp_segmentation_masked)

# Estimate the angle
angle = angle_est.predict_angle(cp_segmentation_largest_region)
print(f"Crack angle [deg]: {angle}")

#####################
# Plot predictions
#####################

# Set colormap and resolution
plt.rcParams['image.cmap'] = 'coolwarm'
plt.rcParams['figure.dpi'] = 300

# Plot
plot_prediction(background=interp_eps_vm * 100,
                interp_size=det.interp_size,
                offset=det.offset,
                save_name=NODEMAP_FILE + '_plot',
                crack_tip_prediction=np.asarray([crack_tip_pixels]),
                crack_tip_seg=crack_tip_seg,
                crack_tip_label=None,
                crack_path=cp_skeleton,
                f_min=0,
                f_max=0.68,
                title=NODEMAP_FILE,
                path=OUTPUT_PATH,
                label='Von Mises strain [%]')
