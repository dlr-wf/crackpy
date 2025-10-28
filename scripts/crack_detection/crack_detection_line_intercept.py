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
from pathlib import Path
import matplotlib.pyplot as plt
import logging

from crackpy.crack_detection.line_intercept import CrackDetectionLineIntercept,plot_grid_errors
from crackpy.crack_detection.correction import CrackTipCorrection, CrackTipCorrectionGridSearch
from crackpy.input.input_data import InputData
from crackpy.fracture_analysis.optimization import OptimizationProperties
from crackpy.structure_elements.data_files import Nodemap
from crackpy.structure_elements.material import Material

# Setup logging for script
logger = logging.getLogger(__name__)

# Set colormap
plt.rcParams['image.cmap'] = 'coolwarm'
plt.rcParams['figure.dpi'] = 300

# Settings
PROJECT_ROOT = Path(__file__).parents[2]

NODEMAP_FILE = 'Dummy2_WPXXX_DummyVersuch_2_dic_results_1_52.txt'
DATA_PATH = PROJECT_ROOT / 'test_data' / 'crack_detection' / 'Nodemaps'

OUTPUT_PATH = PROJECT_ROOT / 'line_intercept'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# Get nodemap data
nodemap = Nodemap(name=NODEMAP_FILE, folder=str(DATA_PATH))
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

logger.info('Optimizing crack tip position via nonlinear optimization …')
crack_tip_corr_opt = correction.correct_crack_tip_optimization(
    opt_props,
    tol=0.01,
    objective='error',)

logger.info('Optimizing crack tip position using Rethore method …')
crack_tip_corr_rethore = correction.correct_crack_tip(
    opt_props,
    max_iter=50,
    step_tol=0.001,
    damper=0.5,
    method='rethore',)

logger.info('Optimizing crack tip position via grid search …')
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
)
errors_path = OUTPUT_PATH / 'errors'
errors_path.mkdir(parents=True, exist_ok=True)
plot_grid_errors(df_grid_errors, fname=Path(NODEMAP_FILE).stem + '_errors.png', folder=str(errors_path))
df_grid_errors.to_csv(str(errors_path / (Path(NODEMAP_FILE).stem + '_errors.csv')), index=False)

# Plot all results
results = {
    'Optimization': crack_tip_corr_opt,
    'Rethore': crack_tip_corr_rethore,
    'Grid search': crack_tip_corr_grid
}
plots_path = OUTPUT_PATH / 'plots'
plots_path.mkdir(parents=True, exist_ok=True)
cd.plot(fname=Path(NODEMAP_FILE).stem + '.png', folder=str(plots_path),
        fmin=0, fmax=350, crack_tip_results=results)

# Log results
logger.info(
      f"Line Intercept (raw): tip -> x = {crack_tip[0]:+.3f} mm, y = {crack_tip[1]:+.3f} mm, angle = {cd.crack_angle:+.3f}°")
logger.info(
      f"Optimization:        tip -> x = {crack_tip[0] + crack_tip_corr_opt[0]:+.3f} mm, "
      f"y = {crack_tip[1] + crack_tip_corr_opt[1]:+.3f} mm, "
      f"angle = {cd.crack_angle + crack_tip_corr_opt[2]:+.3f}°")
logger.info(
      f"Rethore:             tip -> x = {crack_tip[0] + crack_tip_corr_rethore[0]:+.3f} mm, "
      f"y = {crack_tip[1] + crack_tip_corr_rethore[1]:+.3f} mm, "
      f"angle = {cd.crack_angle:+.3f}°")
logger.info(
      f"Grid search:         tip -> x = {crack_tip[0] + crack_tip_corr_grid[0]:+.3f} mm, "
      f"y = {crack_tip[1] + crack_tip_corr_grid[1]:+.3f} mm, "
      f"angle = {cd.crack_angle:+.3f}°")
