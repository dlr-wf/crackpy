"""

    Example script:
        Fracture analysis for a single FE nodemap.

    Input:
        - Output folder
        - Nodemap file
        - Nodemap structure
        - Material properties
        - Integral properties
        - Optimization properties
        - Crack tip position

    Output:
        - Fracture Analysis results (plots, txt-files)

"""
from pathlib import Path
import logging

from matplotlib import pyplot as plt

from crackpy.fracture_analysis.analysis import FractureAnalysis
from crackpy.fracture_analysis.line_integration import IntegralProperties
from crackpy.fracture_analysis.optimization import OptimizationProperties
from crackpy.input.crack_tip_info import CrackTipInfo
from crackpy.input.input_data import InputData
from crackpy.results.plot import PlotSettings, Plotter
from crackpy.results.write import OutputWriter
from crackpy.structure_elements.data_files import Nodemap
from crackpy.structure_elements.material import Material

# Logging
logger = logging.getLogger(__name__)

# Determine project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

NODEMAP_FILENAME = 'File_F_10000.0_a_0.5_B_200.0_H_200.0.txt'
NODEMAP_FOLDER = PROJECT_ROOT / 'test_data' / 'simulations' / 'Nodemaps'
OUT_FOLDER = PROJECT_ROOT / 'Fracture_Analysis_FE_results'
OUT_FOLDER.mkdir(parents=True, exist_ok=True)

########################
# INPUT specifications #
########################

material = Material(E=72000, nu_xy=0.33, sig_yield=350)

int_props = IntegralProperties(
    number_of_paths=10,
    number_of_nodes=100,

    bottom_offset=-0.5,
    top_offset=0.5,

    integral_size_left=-5,
    integral_size_right=5,
    integral_size_top=5,
    integral_size_bottom=-5,

    paths_distance_top=0.5,
    paths_distance_left=0.5,
    paths_distance_right=0.5,
    paths_distance_bottom=0.5,

    # mask_tolerance=2,

    buckner_williams_terms=[-1, 1, 2, 3, 4, 5]
)

opt_props = OptimizationProperties(
    angle_gap=10,
    min_radius=5,
    max_radius=10,
    tick_size=0.01,
    terms=[-1, 0, 1, 2, 3, 4, 5]
)

ct = CrackTipInfo(50, 0, 0, 'right')

###############
# Main script #
###############

nodemap = Nodemap(name=NODEMAP_FILENAME, folder=NODEMAP_FOLDER)
input_data = InputData(nodemap=nodemap)
input_data.calc_stresses(material)
input_data.transform_data(ct.crack_tip_x, ct.crack_tip_y, ct.crack_tip_angle)

analysis = FractureAnalysis(
    material=Material(),
    nodemap=nodemap,
    data=input_data,
    crack_tip_info=ct,
    integral_properties=int_props,
    optimization_properties=opt_props
)
analysis.run()

# Set colormap
plt.rcParams['image.cmap'] = 'coolwarm'
plt.rcParams['figure.dpi'] = 100

# Plotting
plot_sets = PlotSettings(background='sig_vm', min_value=0, max_value=material.sig_yield, extend='max')
plotter = Plotter(path=str(OUT_FOLDER / 'plots'), fracture_analysis=analysis, plot_sets=plot_sets)
plotter.plot()

writer = OutputWriter(path=str(OUT_FOLDER / 'results'), fracture_analysis=analysis)
writer.write_header()
writer.write_results()
writer.write_json(path=str(OUT_FOLDER / 'json'))
