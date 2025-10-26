"""

    Example script:
        Fracture analysis for a single DIC nodemap.

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

import os
import logging

from matplotlib import pyplot as plt

from crackpy.fracture_analysis.analysis import FractureAnalysis
from crackpy.input.input_data import InputData
from crackpy.input.crack_tip_info import CrackTipInfo
from crackpy.fracture_analysis.line_integration import IntegralProperties
from crackpy.fracture_analysis.optimization import OptimizationProperties
from crackpy.results.plot import PlotSettings, Plotter
from crackpy.results.write import OutputWriter
from crackpy.structure_elements.data_files import Nodemap
from crackpy.structure_elements.material import Material

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

########################
# INPUT specifications #
########################

NODEMAP_FILENAME = 'Dummy2_WPXXX_DummyVersuch_2_dic_results_1_52.txt'
NODEMAP_FOLDER = os.path.join('..', '..', 'test_data', 'crack_detection', 'Nodemaps')
OUT_FOLDER = 'Fracture_Analysis_DIC_results'

material = Material(E=72000, nu_xy=0.33, sig_yield=350)

int_props = IntegralProperties(
    number_of_paths=10,
    number_of_nodes=100,

    integral_size_left=-5,
    integral_size_right=5,
    integral_size_top=5,
    integral_size_bottom=-5,

    paths_distance_top=0.5,
    paths_distance_left=0.5,
    paths_distance_right=0.5,
    paths_distance_bottom=0.5,

    top_offset=2.5,
    bottom_offset=-2.5,

    mask_tolerance=2,

    buckner_williams_terms=[-1, 1, 2, 3, 4, 5]
)

opt_props = OptimizationProperties(
    angle_gap=20,
    min_radius=5,
    max_radius=10,
    tick_size=0.01,
    terms=[-3,-2,-1, 0, 1, 2, 3, 4, 5],
    dimensions=3
)

ct = CrackTipInfo(
    crack_tip_x=15.16,
    crack_tip_y=0.49,
    crack_tip_angle=-0.46,
    left_or_right='right'
)

###############
# Main script #
###############

nodemap = Nodemap(name=NODEMAP_FILENAME, folder=NODEMAP_FOLDER)
input_data = InputData(nodemap=nodemap)
input_data.transform_data(ct.crack_tip_x, ct.crack_tip_y, ct.crack_tip_angle)
input_data.calc_stresses(material)



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
plotter = Plotter(path=os.path.join(OUT_FOLDER, 'plots'), fracture_analysis=analysis, plot_sets=plot_sets)
plotter.plot()

writer = OutputWriter(path=os.path.join(OUT_FOLDER, 'results'), fracture_analysis=analysis)
writer.write_header()
writer.write_results()
writer.write_json(path=os.path.join(OUT_FOLDER, 'json'))
