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

from matplotlib import pyplot as plt

from crackpy.fracture_analysis.analysis import FractureAnalysis
from crackpy.fracture_analysis.data_processing import InputData, CrackTipInfo
from crackpy.fracture_analysis.line_integration import IntegralProperties
from crackpy.fracture_analysis.optimization import OptimizationProperties
from crackpy.fracture_analysis.plot import PlotSettings, Plotter
from crackpy.fracture_analysis.write import OutputWriter
from crackpy.structure_elements.data_files import Nodemap
from crackpy.structure_elements.material import Material

########################
# INPUT specifications #
########################

NODEMAP_FILENAME = 'Dummy2_WPXXX_DummyVersuch_2_dic_results_1_52.txt'
NODEMAP_FOLDER = os.path.join('..', '..', 'test_data', 'crack_detection', 'Nodemaps')
OUT_FOLDER = 'Fracture_Analysis_DIC_results'

material = Material(E=72000, nu_xy=0.33, sig_yield=350)

int_props = IntegralProperties(
    number_of_paths=3,
    number_of_nodes=100,

    integral_size_left=-5,
    integral_size_right=10,
    integral_size_top=8,
    integral_size_bottom=-8,

    top_offset=3,
    bottom_offset=-3,

    paths_distance_left=0.1,
    paths_distance_right=0.1,
    paths_distance_bottom=0.1,
    paths_distance_top=0.1,

    mask_tolerance=2,

    buckner_williams_terms=[-1, 1, 2, 3, 4, 5]
)

opt_props = OptimizationProperties(terms=[-1, 0, 1, 2, 3, 4, 5])

ct = CrackTipInfo(
    crack_tip_x=-15.5,
    crack_tip_y=0,
    crack_tip_angle=180,
    left_or_right='left'
)

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
plot_sets = PlotSettings(background='eps_vm', min_value=0, max_value=0.0068, extend='max')
plotter = Plotter(path=os.path.join(OUT_FOLDER, 'plots'), fracture_analysis=analysis, plot_sets=plot_sets)
plotter.plot()

writer = OutputWriter(path=os.path.join(OUT_FOLDER, 'results'), fracture_analysis=analysis)
writer.write_header()
writer.write_results()
