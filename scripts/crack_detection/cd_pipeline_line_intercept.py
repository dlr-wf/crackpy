"""

    Example script:
        Crack Detection path and tip with the line interception method and correction

    Needed:
        - Nodemap

    Output:
        - Crack tip position
        - Crack path
        - Plot of prediction

"""
# Imports
import os

import matplotlib.pyplot as plt

from crackpy.crack_detection.line_intercept import CrackDetectionLineIntercept
from crackpy.crack_detection.correction import CrackTipCorrection
from crackpy.crack_detection.utils.utilityfunctions import get_nodemaps_and_stage_nums
from crackpy.fracture_analysis.data_processing import InputData
from crackpy.fracture_analysis.optimization import OptimizationProperties
from crackpy.structure_elements.data_files import Nodemap
from crackpy.structure_elements.material import Material

# Set colormap
plt.rcParams['image.cmap'] = 'coolwarm'
plt.rcParams['figure.dpi'] = 300

# Settings
DATA_PATH = os.path.join('..', '..', 'test_data', 'crack_detection', 'Nodemaps')

OUTPUT_PATH = 'line_intercept_pipeline'
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

MAX_FORCE = 15000
# MAX_FORCE = 4500

material = Material(E=72000, nu_xy=0.33, sig_yield=350)

# open empty file
with open(os.path.join(OUTPUT_PATH, f"{OUTPUT_PATH}.txt"), "w") as out_file:
    out_file.write(
        "####################################################################\n"
        "# Crack detection with line intercept method\n"
        "#\n"
        "# Rethore correction method:\n"
        "# d_x = -2 * a[-1] / a[1], d_y = 0\n"
        "#\n"
        "# Symbolic regression correction method:\n"
        "# d_x = - a[-1] / (a[1] + b[1]) \n"
        "# d_y = - b[-1] / (a[1] - b[1])\n"
        "#\n"
        "####################################################################\n"
    )
    out_file.write(
        f"{'Filename':>60},"
        f"{'CT x [mm]':>12},{'ĆT y [mm]':>12},"
        f"{'SymReg Corr x [mm]':>20},{'SymReg Corr y [mm]':>20},"
        f"{'Rethore Corr x [mm]':>20},{'Rethore Corr y [mm]':>20}"
        f"\n"
    )

    # iterate over all nodemaps in folder in a sorted manner
    stages_to_filenames, _ = get_nodemaps_and_stage_nums(DATA_PATH)
    for stage in sorted(list(stages_to_filenames)):
        file = stages_to_filenames[stage]
        if file.endswith(".txt"):
            # Get nodemap data
            nodemap = Nodemap(name=file, folder=DATA_PATH)
            data = InputData(nodemap)
            data.calc_stresses(material)
            if data.force is not None and data.force > MAX_FORCE - 50:

                # Run crack detection
                print(f"Crack detection for {file} ...")
                cd = CrackDetectionLineIntercept(
                    x_min=0,
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

                # Fine-tune crack tip position
                opt_props = OptimizationProperties(
                    angle_gap=20,
                    min_radius=3,
                    max_radius=8,
                    tick_size=0.1,
                    terms=[-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
                )
                correction = CrackTipCorrection(data, cd.crack_tip, cd.crack_angle, material)
                crack_tip_corr_rethore = correction.correct_crack_tip(
                    opt_props,
                    max_iter=100,
                    step_tol=1e-3,
                    method='rethore',
                    verbose=True,
                    damper=0.5
                )
                crack_tip_corr_symreg = correction.correct_crack_tip(
                    opt_props,
                    max_iter=100,
                    step_tol=1e-3,
                    method='symbolic_regression',
                    verbose=True,
                    damper=0.5
                )

                # Plot prediction
                results = {
                    'Rethore': crack_tip_corr_rethore,
                    'SymReg': crack_tip_corr_symreg
                }
                cd.plot(fname=file[:-4] + '.png', folder=os.path.join(OUTPUT_PATH, 'plots'),
                        crack_tip_results=results, fmax=material.sig_yield)

                # Write results to file
                out_file.write(
                    f'{file:>60},'
                    f'{cd.crack_tip[0]:>17.2f},{cd.crack_tip[1]:>17.2f},'
                    f'{crack_tip_corr_rethore[0]:>22.2f},{crack_tip_corr_rethore[1]:>22.2f},'
                    f'{crack_tip_corr_symreg[0]:>22.2f},{crack_tip_corr_symreg[1]:>22.2f}'
                    f'\n'
                )
