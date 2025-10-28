"""

    Example script:
        Example how to read only the header of a nodemap file.

    Needed:
        - Nodemap

    Output:
        - List of data in the header
"""

# Imports
from pathlib import Path
import time
import logging
from crackpy.input.input_data import InputData
from crackpy.structure_elements.data_files import Nodemap

# Determine project root (two levels above scripts/<subdir>)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Logging
logger = logging.getLogger(__name__)

# Settings
NODEMAP_FILE = 'Dummy2_WPXXX_DummyVersuch_2_dic_results_1_53.txt'
NODEMAP_PATH = PROJECT_ROOT / 'test_data' / 'crack_detection' / 'Nodemaps'

# Metadata in the header
meta_attributes_to_keywords = {
    'force_digital': 'force',
    'displacement_digital': 'displacement',
    'current_stage_date': 'current_stage_date',
    'specimen': 'specimen',
    'calibration_date': 'calibration_date',
    'force_analog': 'Kraft',
    'displacement_analog': 'Verschiebung',
    'fake_data': 'fake_data',
}

# Start measuring time
start_time = time.time()

# Get nodemap data
nodemap = Nodemap(name=NODEMAP_FILE, folder=str(NODEMAP_PATH))
data = InputData(nodemap, meta_keywords=meta_attributes_to_keywords, read_header_only=True)

# End measuring time
end_time = time.time()

# Calculate and log the elapsed time
elapsed_time = end_time - start_time
logger.info(f"Elapsed time reading header: {elapsed_time:.4f} seconds")

# Log header data
logger.info("============ Header data ============")
for meta_attr, meta_key in meta_attributes_to_keywords.items():
    if hasattr(data, meta_attr):
        logger.info(f"{meta_attr}: {getattr(data, meta_attr)}")
