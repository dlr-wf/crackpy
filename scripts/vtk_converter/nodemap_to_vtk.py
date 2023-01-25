"""

    Example script:
        Convert nodemap and connection file to vtk

    Needed:
        - Nodemap
        - Connection

    Output:
        - vtk file

"""

# Imports
import os
from crackpy.fracture_analysis.data_processing import InputData
from crackpy.structure_elements.data_files import Nodemap
from crackpy.structure_elements.material import Material

# Settings
NODEMAP_FILE = 'Dummy2_WPXXX_DummyVersuch_2_dic_results_1_53.txt'
CONNECTION_FILE = 'Dummy2_WPXXX_DummyVersuch_2_dic_results_1_53_connections.txt'
NODEMAP_PATH = os.path.join('..', '..', 'test_data', 'crack_detection', 'Nodemaps')
CONNECTION_PATH = os.path.join('..', '..', 'test_data', 'crack_detection', 'Connections')
OUTPUT_PATH = os.path.join('..', '..', 'test_data', 'crack_detection', 'output', 'vtk')

# Create output folder
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Get nodemap data
nodemap = Nodemap(name=NODEMAP_FILE, folder=NODEMAP_PATH)
material = Material(E=72000, nu_xy=0.33, sig_yield=350)
data = InputData(nodemap)
data.set_connection_file(CONNECTION_FILE, folder=CONNECTION_PATH)
data.calc_stresses(material)
data.calc_eps_vm()
mesh = data.to_vtk(OUTPUT_PATH)

# Plot mesh data and save to file

mesh.plot(scalars='eps_vm [%]',
          clim=[0, 0.5],
          cpos='xy',
          cmap='jet',
          show_bounds=True,
          lighting=True,
          show_scalar_bar=True,
          scalar_bar_args={'vertical': True},
          #screenshot='eps_vm.png',
          #off_screen=True
          )