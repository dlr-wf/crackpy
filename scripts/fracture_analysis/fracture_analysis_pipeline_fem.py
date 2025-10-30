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
from pathlib import Path
import logging

from matplotlib import pyplot as plt

from crackpy.fracture_analysis.line_integration import IntegralProperties
from crackpy.fracture_analysis.optimization import OptimizationProperties
from crackpy.fracture_analysis.pipeline import FractureAnalysisPipeline
from crackpy.results.plot import PlotSettings
from crackpy.results.read import OutputReader
from crackpy.structure_elements.material import Material

# Logging
logger = logging.getLogger(__name__)

# Determine project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Paths
DATA_PATH = PROJECT_ROOT / 'test_data' / 'simulations'
OUT_FOLDER = PROJECT_ROOT / 'Fracture_Analysis_Pipeline_FE_results'
OUT_FOLDER.mkdir(parents=True, exist_ok=True)

#######################################
#         Fracture Analysis           #
#######################################
int_props = IntegralProperties(
    number_of_paths=10,
    number_of_nodes=100,

    bottom_offset=-0.5,
    top_offset=0.5,

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
    terms=[-3, -2, -1, 0, 1, 2, 3, 4, 5],
)

material = Material(E=72000, nu_xy=0.33, sig_yield=350)

# Plot settings
plt.rcParams['image.cmap'] = 'coolwarm'
plt.rcParams['figure.dpi'] = 100
plot_sets = PlotSettings(background='sig_vm', min_value=0, max_value=material.sig_yield, extend='max')

fa_pipeline = FractureAnalysisPipeline(
    material=material,
    nodemap_path=str(DATA_PATH / 'Nodemaps'),
    input_file=str(DATA_PATH / 'crack_info_by_nodemap.txt'),
    output_path=str(OUT_FOLDER),
    optimization_properties=opt_props,
    integral_properties=int_props,
    plot_sets=plot_sets
)
fa_pipeline.run(num_of_kernels=10)

# Read results and write into CSV file
reader = OutputReader()
fa_output_path = OUT_FOLDER / 'txt-files'

files = [p.name for p in fa_output_path.iterdir()]
list_of_tags = ["SIFs_integral"]
for file in files:
    for tag in list_of_tags:
        reader.read_tag_data(path=str(fa_output_path), filename=file, tag=tag)

reader.make_csv_from_results(files="all", output_path=str(OUT_FOLDER), output_filename='results.csv')
