"""

    Example script:
        Crack Detection (path, tip, angle) for a single nodemap
        using the line intercept method and different correction methods

    Needed:
        - Nodemap

    Output:
        - Crack tip position
        - Crack path
        - Crack angle
        - Plot of predictions

"""

# Imports
import os
import matplotlib.pyplot as plt

from crackpy.crack_detection.line_intercept import CrackDetectionLineIntercept,plot_grid_errors
from crackpy.crack_detection.correction import CrackTipCorrection, CrackTipCorrectionGridSearch
from crackpy.fracture_analysis.data_processing import InputData
from crackpy.fracture_analysis.optimization import OptimizationProperties
from crackpy.structure_elements.data_files import Nodemap
from crackpy.structure_elements.material import Material


# Set colormap
plt.rcParams['image.cmap'] = 'coolwarm'
plt.rcParams['figure.dpi'] = 300

# Settings
NODEMAP_FILE = 'Dummy2_WPXXX_DummyVersuch_2_dic_results_1_52.txt'
DATA_PATH = os.path.join('..', '..', 'test_data', 'crack_detection', 'Nodemaps')

OUTPUT_PATH = 'line_intercept'
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


# Get nodemap data
nodemap = Nodemap(name=NODEMAP_FILE, folder=DATA_PATH)
data = InputData(nodemap)
data.read_header()
material = Material(E=72000, nu_xy=0.33, sig_yield=350)
data.calc_stresses(material)


######################################
# Crack detection with line intercept
######################################
cd = CrackDetectionLineIntercept(
    x_min=0.0,
    x_max=25.0,
    y_min=-10.0,
    y_max=10.0,
    data=data,
    tick_size_x=0.1,
    tick_size_y=0.1,
    grid_component='uy',
    eps_vm_threshold=0.01,
    window_size=3,
    angle_estimation_mm_radius=5.0
)
cd.run()
crack_tip = cd.crack_tip
crack_angle = cd.crack_angle


######################################
# Fine-tuning of crack tip position
######################################
correction = CrackTipCorrection(data, crack_tip, crack_angle, material)
opt_props = OptimizationProperties(
    angle_gap=10,
    min_radius=3,
    max_radius=8,
    tick_size=0.1,
    terms=[-1, 0, 1, 2]
)

print('Optimizing crack tip position...')
crack_tip_corr_opt = correction.correct_crack_tip_optimization(
    opt_props,
    tol=0.01,
    objective='error',
    verbose=True
)

print('Optimizing crack tip position (Rethore method)...')
crack_tip_corr_rethore = correction.correct_crack_tip(
    opt_props,
    max_iter=50,
    step_tol=0.001,
    damper=0.5,
    method='rethore',
    verbose=True
)

print('Optimizing crack tip position (grid search)...')
correction_grid = CrackTipCorrectionGridSearch(data, crack_tip, crack_angle, material)
crack_tip_corr_grid, df_grid_errors = correction_grid.correct_crack_tip_grid_search(
    opt_props,
    x_min=-3,
    x_max=3,
    y_min=-3,
    y_max=3,
    x_step=1,
    y_step=1,
    workers=20,
    verbose=True
)
plot_grid_errors(df_grid_errors, fname=NODEMAP_FILE[:-4] + '_errors.png', folder=os.path.join(OUTPUT_PATH, 'errors'))
df_grid_errors.to_csv(os.path.join(OUTPUT_PATH, 'errors', NODEMAP_FILE[:-4] + '_errors.csv'), index=False)

# Plot all results
results = {
    'Optimization': crack_tip_corr_opt,
    'Rethore': crack_tip_corr_rethore,
    'Grid search': crack_tip_corr_grid
}
cd.plot(fname=NODEMAP_FILE[:-4] + '.png', folder=os.path.join(OUTPUT_PATH, 'plots'),
        fmin=0, fmax=350, crack_tip_results=results)

# Print results
print(f"\nLine Intercept:\n"
      f"tip:  x = {crack_tip[0]:+.3f} mm, y = {crack_tip[1]:+.3f} mm, angle = {cd.crack_angle:+.3f}°")
print(f"\nOptimization:\n"
      f"tip:  x = {crack_tip[0] + crack_tip_corr_opt[0]:+.3f} mm, "
      f"y = {crack_tip[1] + crack_tip_corr_opt[1]:+.3f} mm, "
      f"angle = {cd.crack_angle + crack_tip_corr_opt[2]:+.3f}°")
print(f"\nRethore:\n"
      f"tip:  x = {crack_tip[0] + crack_tip_corr_rethore[0]:+.3f} mm, "
      f"y = {crack_tip[1] + crack_tip_corr_rethore[1]:+.3f} mm, "
      f"angle = {cd.crack_angle:+.3f}°")
print(f"\nGrid search:\n"
      f"tip:  x = {crack_tip[0] + crack_tip_corr_grid[0]:+.3f} mm, "
      f"y = {crack_tip[1] + crack_tip_corr_grid[1]:+.3f} mm, "
      f"angle = {cd.crack_angle:+.3f}°")
