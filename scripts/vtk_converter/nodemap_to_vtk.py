"""

    Example script:
        Convert nodemap and connection file to vtk

    Needed:
        - Nodemap
        - Connection

    Output:
        - vtk file
        - Plot of nodemap with pyvista
        - Plot of nodemap with matplotlib
"""

# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

# Plot mesh data and save to file using pyvista
mesh.plot(scalars='eps_vm [%]',
          clim=[0, 0.5],
          cpos='xy',
          cmap='jet',
          show_bounds=True,
          lighting=True,
          show_scalar_bar=True,
          scalar_bar_args={'vertical': True},
          screenshot=os.path.join(OUTPUT_PATH, "pyvista_plot.png"),
          off_screen=True
          )

# Plot mesh data with matplotlib and save to file
fmin = 0
fmax = 0.8
num_colors = 120
num_ticks = 6
contour_vector = np.linspace(fmin, fmax, num_colors, endpoint=True)
label_vector = np.linspace(fmin, fmax, num_ticks, endpoint=True)

plt.clf()
fig = plt.figure(1)
ax = fig.add_subplot(111)

# prepare data for tricontourf
x = mesh.points[:, 0]
y = mesh.points[:, 1]


# if no connection file is given, use faces, else use cells
if data.connections is None:
    triangles = mesh.faces.reshape((-1, 4))[:, 1:]
else:
    triangles = mesh.cells.reshape((-1, 4))[:, 1:]

# create triangles
triang = mtri.Triangulation(x, y, triangles)

# scalar data
scalar_data = mesh.point_data['eps_vm [%]']

# plot data using tricontourf
plot = ax.tricontourf(triang, scalar_data, contour_vector, cmap='jet', extend='max')

# Improve the look of the plot
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
fig.colorbar(plot, ticks=label_vector, cax=cax, label='Von Mises eqv. strain [\\%]')
ax.set_xlabel('$x$ [mm]')
ax.set_ylabel('$y$ [mm]')
ax.axis('image')
ax.tick_params(axis='x', pad=15)

# Save plot to file
plt.savefig(os.path.join(OUTPUT_PATH, "matplotlib_plot.png"), bbox_inches='tight')
