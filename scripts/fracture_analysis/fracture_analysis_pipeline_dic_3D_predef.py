"""

    Example script:
        Crack Detection & Fracture Analysis Pipeline with predefined integral paths

    Needed:
        - folder containing DIC Nodemap data

    Output:
        - folder containing Fracture Analysis results (plots, txt-files) for each nodemap
        - crack_info_by_nodemap.txt - file generated during crack detection and used by fracture analysis pipeline
        - results.csv - single CSV file with all the results
        - results_maxforce.csv - single CSV file with just maximum force nodemap results

"""

from pathlib import Path
import logging
from matplotlib import pyplot as plt

from crackpy.crack_detection.model import get_model
from crackpy.crack_detection.pipeline.pipeline import CrackDetectionSetup, CrackDetectionPipeline
from crackpy.fracture_analysis.line_integration import IntegralProperties
from crackpy.fracture_analysis.optimization import OptimizationProperties
from crackpy.fracture_analysis.pipeline import FractureAnalysisPipeline
from crackpy.results.plot import PlotSettings
from crackpy.results.read import OutputReader
from crackpy.structure_elements.material import Material

# Logging
logger = logging.getLogger(__name__)


plt.rcParams['image.cmap'] = 'coolwarm'
plt.rcParams['figure.dpi'] = 300

# Determine project root (two levels above scripts/<subdir>)
PROJECT_ROOT = Path(__file__).parents[2]

# Paths
DATA_PATH = PROJECT_ROOT / 'test_data' / 'crack_detection' / 'Nodemaps'
OUT_FOLDER = PROJECT_ROOT / 'Fracture_Analysis_Pipeline_DIC_results_predef'
OUT_FOLDER.mkdir(parents=True, exist_ok=True)

# crack detectors
tip_detector = get_model('ParallelNets')
path_detector = get_model('UNetPath')

# Detection setup
det_setup = CrackDetectionSetup(
    specimen_size=160,
    sides=['left', 'right'],
    detection_window_size=50,
    detection_boundary=(0, 70, -35, 35),
    start_offset=(0, 0)
)

# Crack detection
cd_pipeline = CrackDetectionPipeline(
    data_path=str(DATA_PATH),
    output_path=str(OUT_FOLDER),
    tip_detector_model=tip_detector,
    path_detector_model=path_detector,
    setup=det_setup
)
cd_pipeline.filter_detection_stages(max_force=15000)
cd_pipeline.run_detection()
cd_pipeline.assign_remaining_stages()
cd_pipeline.write_results('crack_info_by_nodemap.txt')

# Fracture Analysis
int_props = IntegralProperties(
    number_of_paths=5,
    number_of_nodes=100,

    integral_size_left=-5,
    integral_size_right=5,
    integral_size_top=5,
    integral_size_bottom=-5,

    paths_distance_top=0.5,
    paths_distance_left=0.5,
    paths_distance_right=0.5,
    paths_distance_bottom=0.5,

    top_offset=3,
    bottom_offset=-3,

    mask_tolerance=2,

    buckner_williams_terms=[-1, 1, 2, 3, 4, 5]
)

opt_props = OptimizationProperties(
    angle_gap=20,
    min_radius=5,
    max_radius=10,
    tick_size=0.01,
    terms=[-3, -2, -1, 0, 1, 2, 3, 4, 5],
)

material = Material(E=72000, nu_xy=0.33, sig_yield=350)

# Plot settings
plot_sets = PlotSettings(background='sig_vm', min_value=0, max_value=material.sig_yield, extend='max')

fa_pipeline = FractureAnalysisPipeline(
    material=material,
    nodemap_path=str(DATA_PATH),
    input_file=str(OUT_FOLDER / 'crack_info_by_nodemap.txt'),
    output_path=str(OUT_FOLDER),
    optimization_properties=opt_props,
    integral_properties=int_props,
    plot_sets=plot_sets
)
fa_pipeline.find_max_force_stages(max_force=15000)
fa_pipeline.run(num_of_kernels=10)

# Read results and write into CSV file
reader = OutputReader()
fa_output_path = OUT_FOLDER / 'txt-files'

files = [p.name for p in fa_output_path.iterdir()]
list_of_tags = ["CJP_results", "Williams_fit_results", "SIFs_integral", "Bueckner_Chen_integral",
                "Path_SIFs", "Path_Williams_a_n", "Path_Williams_b_n"]
for file in files:
    for tag in list_of_tags:
        reader.read_tag_data(path=str(fa_output_path), filename=file, tag=tag)

reader.make_csv_from_results(files="all", output_path=str(OUT_FOLDER), output_filename='results.csv')
reader.make_csv_from_results(files="all", filter_condition={'Force': (14900, 15100)},
                             output_path=str(OUT_FOLDER), output_filename='results_maxforce.csv')
