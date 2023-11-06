"""

    Example script:
        Crack Detection (path, tip, angle) for a single nodemap
        using the line intercept method and different correction methods

    Needed:
        - Nodemap

    Output:
        - Crack tip position
        - Crack path
        - Crack angle
        - Plot of predictions

"""

# Imports
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from crackpy.crack_detection.line_intercept import CrackDetectionLineIntercept
from crackpy.fracture_analysis.data_processing import InputData
from crackpy.structure_elements.data_files import Nodemap
from crackpy.structure_elements.material import Material


# Set colormap
plt.rcParams['image.cmap'] = 'coolwarm'
plt.rcParams['figure.dpi'] = 300

# Settings
NODEMAP_FILE = 'Dummy2_WPXXX_DummyVersuch_2_dic_results_1_52.txt'
DATA_PATH = os.path.join('..', '..', 'test_data', 'crack_detection', 'Nodemaps')


OUTPUT_PATH = 'line_intercept_debug'
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


# Get nodemap data
nodemap = Nodemap(name=NODEMAP_FILE, folder=DATA_PATH)
data = InputData(nodemap)
data.read_header()
material = Material(E=72000, nu_xy=0.33, sig_yield=350)
data.calc_stresses(material)


######################################
# Crack detection with line intercept
######################################
cd = CrackDetectionLineIntercept(
    x_min=0.0,
    x_max=25.0,
    y_min=-10.0,
    y_max=10.0,
    data=data,
    tick_size_x=0.1,
    tick_size_y=0.1,
    grid_component='uy',
    eps_vm_threshold=0.5/100,
    window_size=3,
    angle_estimation_mm_radius=5.0
)
cd.run()


######################################
# DIC data and detected crack path
######################################

fmin = 0
fmax = 0.5
num_colors = 120
contour_vector = np.linspace(fmin, fmax, num_colors, endpoint=True)
label_vector = np.linspace(fmin, fmax, 6, endpoint=True)

plt.clf()
fig = plt.figure(1)
ax = fig.add_subplot(111)
plot = ax.tricontourf(data.coor_x, data.coor_y, data.eps_vm * 100.0, contour_vector, cmap='jet', extend='max')

ax.plot(cd.crack_path[:, 0], cd.crack_path[:, 1], 'k--', linewidth=2, label='Crack path')
ax.plot(cd.x_coords[cd.tip_index:], cd.y_path[cd.tip_index:], 'r--', linewidth=2, label='Line segments')

indexes = [100,  150, 200]

# plot paths
for index in indexes:
    ax.plot(cd.x_grid[:, index],cd.y_grid[:, index])

# plot crackt tip
ax.scatter(cd.crack_tip[0], cd.crack_tip[1], marker='X', color='grey', s=50, label='Crack tip')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
fig.colorbar(plot, ticks=label_vector, cax=cax, label='Von Mises eqv. strain [%]')
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
ax.legend(loc='upper left')
ax.axis('image')
ax.set_xlim(0, 25)
ax.set_ylim(-10, 10)
ax.tick_params(axis='x', pad=15)
plt.savefig(os.path.join(OUTPUT_PATH, f"{NODEMAP_FILE[:-4]}.png"), bbox_inches='tight', dpi=300)

######################################
# tanh fit
######################################

plt.clf()
fig = plt.figure(1)
ax = fig.add_subplot(111)

for index in indexes:
    ax.plot(cd.y_grid[:, index], cd.disp_y_grid[:, index],
            'k-', linewidth=2, label=f'x = {cd.x_grid[0,index]:3.1} mm')
    ax.plot(cd.y_grid[:, index], cd._tanh_funct(cd.coefficients_fitted[:,index], cd.y_grid[:, index]),
            'r--', linewidth=2, label=f'x = {cd.x_grid[0,index]:3.1} mm (tanh)')
    ax.text(cd.y_grid[0, index], cd.disp_y_grid[0, index], f'x = {cd.x_grid[0,index]:3.1f} mm')

ax.set_xlabel('y [mm]')
ax.set_ylabel('$u_{y}$ [mm]')
ax.tick_params(axis='x', pad=15)
plt.savefig(os.path.join(OUTPUT_PATH, f"{NODEMAP_FILE[:-4]}_tanh.png"), bbox_inches='tight', dpi=300)
