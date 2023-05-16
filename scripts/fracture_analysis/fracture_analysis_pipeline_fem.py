"""

    Example script:
        Fracture Analysis Pipeline for FE nodemaps.

    Needed:
        - folder containing FE Nodemap data
        - crack_info_by_nodemap.txt file with the crack tip positions and angles

    Output:
        - folder containing Fracture Analysis results (plots, txt-files) for each nodemap

"""

# Imports
import os

from matplotlib import pyplot as plt

from crackpy.fracture_analysis.line_integration import IntegralProperties
from crackpy.fracture_analysis.optimization import OptimizationProperties
from crackpy.fracture_analysis.pipeline import FractureAnalysisPipeline
from crackpy.fracture_analysis.plot import PlotSettings
from crackpy.fracture_analysis.read import OutputReader
from crackpy.structure_elements.material import Material

# Paths
DATA_PATH = os.path.join('..', '..', 'test_data', 'simulations')
OUT_FOLDER = 'Fracture_Analysis_Pipeline_FE_results'

#######################################
#         Fracture Analysis           #
#######################################
int_props = IntegralProperties(
    number_of_paths=10,
    number_of_nodes=100,

    bottom_offset=-0,
    top_offset=0,

    integral_size_left=-5,
    integral_size_right=5,
    integral_size_top=5,
    integral_size_bottom=-5,

    paths_distance_top=0.1,
    paths_distance_left=0.1,
    paths_distance_right=0.1,
    paths_distance_bottom=0.1,

    mask_tolerance=2,

    buckner_williams_terms=[-1, 1, 2, 3, 4, 5]
)

opt_props = OptimizationProperties(
    angle_gap=5,
    min_radius=5,
    max_radius=10,
    tick_size=0.01,
    terms=[-1, 0, 1, 2, 3, 4, 5]
)

material = Material(E=72000, nu_xy=0.33, sig_yield=350)

# Plot settings
plt.rcParams['image.cmap'] = 'coolwarm'
plt.rcParams['figure.dpi'] = 100
plot_sets = PlotSettings(background='sig_vm', min_value=0, max_value=material.sig_yield, extend='max')

fa_pipeline = FractureAnalysisPipeline(
    material=material,
    nodemap_path=os.path.join(DATA_PATH, 'Nodemaps'),
    input_file=os.path.join(DATA_PATH, 'crack_info_by_nodemap.txt'),
    output_path=OUT_FOLDER,
    optimization_properties=opt_props,
    integral_properties=int_props,
    plot_sets=plot_sets
)
fa_pipeline.run(num_of_kernels=10)

# Read results and write into CSV file
reader = OutputReader()
fa_output_path = os.path.join(OUT_FOLDER, 'txt-files')

files = os.listdir(fa_output_path)
list_of_tags = ["SIFs_integral"]
for file in files:
    for tag in list_of_tags:
        reader.read_tag_data(path=fa_output_path, filename=file, tag=tag)

reader.make_csv_from_results(files="all", output_path=OUT_FOLDER, output_filename='results.csv')
