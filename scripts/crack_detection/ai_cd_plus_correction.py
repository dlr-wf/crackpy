# Imports
import os
import time
import matplotlib.pyplot as plt

from crackpy.crack_detection.line_intercept import CrackDetectionLineIntercept
from crackpy.crack_detection.correction import CrackTipCorrection
from crackpy.crack_detection.model import get_model
from crackpy.crack_detection.detection import CrackTipDetection, CrackPathDetection, CrackAngleEstimation, CrackDetection
from crackpy.fracture_analysis.data_processing import InputData
from crackpy.fracture_analysis.optimization import OptimizationProperties
from crackpy.structure_elements.data_files import Nodemap
from crackpy.structure_elements.material import Material


# Settings

# Set colormap and resolution
plt.rcParams['image.cmap'] = 'coolwarm'
plt.rcParams['figure.dpi'] = 300

# settings
NODEMAP_FILE = 'Dummy2_WPXXX_DummyVersuch_2_dic_results_1_52.txt'
DATA_PATH = os.path.join('..', '..', 'test_data', 'crack_detection', 'Nodemaps')
OUTPUT_PATH = "cd_ai_plus_correction"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

material = Material(E=72000, nu_xy=0.33, sig_yield=350)

# Crack detection
det = CrackDetection(
    side='right',
    detection_window_size=40,
    offset=(-10, 0),
    angle_det_radius=10,
    device='cpu'
)
print(det.device)

# Get nodemap data
nodemap = Nodemap(name=NODEMAP_FILE, folder=DATA_PATH)
data = InputData(nodemap)
data.calc_stresses(material)


starttime = time.time()
print('Start crack detection')
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
try:
    angle_est = CrackAngleEstimation(detection=det, crack_tip_in_px=crack_tip_pixels)

    # Consider only crack path close to crack tip
    cp_segmentation_masked = angle_est.apply_circular_mask(cp_segmentation)

    # Filter for largest connected crack path
    cp_segmentation_largest_region = angle_est.get_largest_region(cp_segmentation_masked)

    # Estimate the angle
    angle = angle_est.predict_angle(cp_segmentation_largest_region)
    print(f"Crack angle [deg]: {angle}")

except:
    print('Crack angle estimation failed!')
    angle = 0

print(f"Time AI-CrackDetection: {(time.time() - starttime):.2f} s")

starttime = time.time()
cd = CrackDetectionLineIntercept(
    x_min=0.0,
    x_max=25.0,
    y_min=-10,
    y_max=10,
    data=data,
    tick_size_x=0.1,
    tick_size_y=0.1,
    grid_component='uy',
    eps_vm_threshold=0.005,
    window_size=3,
    angle_estimation_mm_radius=5.0
)
cd.crack_tip = [crack_tip_x, crack_tip_y]
cd.crack_angle = angle
cd.crack_path = cp_skeleton


###############################
# Crack tip correction
###############################
correction = CrackTipCorrection(data, cd.crack_tip, cd.crack_angle, material)
opt_props = OptimizationProperties(
    angle_gap=10,
    min_radius=3,
    max_radius=8,
    tick_size=0.1,
    terms=[-1, 0, 1, 2]
)
print('Correcting crack tip position...')
crack_tip_corr = correction.correct_crack_tip(
    opt_props,
    max_iter=100,
    step_tol=0.005,
    damper=1,
    method='symbolic_regression',
    verbose=True,
    plot_intermediate_results=True,
    cd=cd,
    folder=os.path.join(OUTPUT_PATH, 'crack_tip_correction')
)

print(f"Time CrackTipCorrection: {(time.time() - starttime):.2f} s")

# Plot prediction
results_corr = {
    'SymReg': crack_tip_corr,
}

cd.plot(fname=NODEMAP_FILE[:-4] + '.png', folder=os.path.join(OUTPUT_PATH, 'crack_tip_correction'),
        crack_tip_results=results_corr, fmax=material.sig_yield)
