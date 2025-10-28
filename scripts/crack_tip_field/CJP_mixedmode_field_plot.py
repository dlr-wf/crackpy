"""
This script plots the equivalent stress field using the CJP model from the paper
"Extension of the CJP model to mixed mode I and mode II" (2013).

Needed:
    - CJP coefficients

Output:
    - Plot of the equivalent stress field

"""

from pathlib import Path
import logging

from crackpy.fracture_analysis.crack_tip import cjp_stress_field_mixedmode, cjp_displ_field_mixedmode
from crackpy.structure_elements.material import Material
from crackpy.fracture_analysis.optimization import Optimization

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Logging
logger = logging.getLogger(__name__)

# Set matplotlib settings
plt.rcParams.update({
    "font.size": 20,
    "text.usetex": True,
    "font.family": "serif",
    "figure.figsize": [10, 10],
    "figure.dpi": 300
})

# Parameters from the CJP model in "Extension of the CJP model to mixed mode I and mode II" (2013)
A_r = 0  # 150.0
B_r = -A_r
B_i = -150.0
C = 7.2
E = 0.0

# Formulas from the CJP model in "Extension of the CJP model to mixed mode I and mode II" (2013)
K_F = np.sqrt(np.pi / 2) * (A_r - 3 * B_r - 8 * E) / np.sqrt(1000)  # MPa m^0.5
K_R = -4 * np.sqrt(np.pi / 2) * (2 * B_i + E * np.pi) / np.sqrt(1000)  # MPa m^0.5
K_S = -np.sqrt(np.pi / 2) * (A_r + B_r) / np.sqrt(1000)  # MPa m^0.5
K_II = 2 * np.sqrt(2 * np.pi) * B_i / np.sqrt(1000)  # MPa m^0.5
T = -C  # MPa

logger.info(f"CJP parameters: K_F = {K_F:.2f} MPa·m^0.5, K_R = {K_R:.2f} MPa·m^0.5, K_S = {K_S:.2f} MPa·m^0.5, K_II = {K_II:.2f} MPa·m^0.5, T = {T:.2f} MPa")

material = Material(E=72000, nu_xy=0.33, sig_yield=350.0)

# Output path
OUTPUT_PATH = Path('CJP_field')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Define the material
material = Material(E=72000, nu_xy=0.33, sig_yield=350.0)

# Plot
min_radius = 0.01
max_radius = 100.0
tick_size = 0.01

r_grid, phi_grid = np.mgrid[min_radius:max_radius:tick_size, -np.pi:np.pi:tick_size]
coeffs = A_r, B_r, B_i, C, E

# Calculate the stress and displacement fields
sigma_xx, sigma_yy, sigma_xy = cjp_stress_field_mixedmode(coeffs, phi_grid, r_grid)
disp_x, disp_y = cjp_displ_field_mixedmode(coeffs, phi_grid, r_grid, material)

sigma_vm = np.sqrt(sigma_xx ** 2 + sigma_yy ** 2 - sigma_xx * sigma_yy + 3 * sigma_xy ** 2)

x_grid, y_grid = Optimization.make_cartesian(r_grid, phi_grid)

#######################################################################################################################
# Plot u_x
#######################################################################################################################
logger.info('Plotting CJP displacement u_x …')
# Matplotlib plot
number_colors = 120
number_labes = 5
legend_limit_max = 0.1
legend_limit_min = -0.1
cm = 'coolwarm'

# Define contour and label vectors
contour_vector = np.linspace(legend_limit_min, legend_limit_max, number_colors, endpoint=True)
label_vector = np.linspace(legend_limit_min, legend_limit_max, number_labes, endpoint=True)
label_vector = np.round(label_vector, 2)

# Plot the displacement field
plt.clf()
fig = plt.figure(1)
ax = fig.add_subplot(111)

# Plot the crack tip field
plot = ax.contourf(x_grid, y_grid, disp_x, contour_vector, extend='both', cmap=cm)
# Highlight the crack path
ax.plot([0, np.min(x_grid)], [0, 0], 'k', linewidth=2)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(plot, ticks=label_vector,
             cax=cax,
             label='$u_x$ [$\\mathrm{mm}$]')
ax.set_xlabel('$x$ [$\\mathrm{mm}$]')
ax.set_ylabel('$y$ [$\\mathrm{mm}$]')
ax.axis('image')
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)

fig.suptitle('CJP field', y=0.95)
ax.set_title(
    f"$K_F = {K_F:.2f} \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, K_R = {K_R:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, K_S = {K_S:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}$\n"
    f"$K_{{II}} = {K_II:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, T = {T:.2f}  \\, \\mathrm{{N \\cdot mm^{{-2}}}}$",
    fontsize=14)

output_file = OUTPUT_PATH / 'CJP_field_ux.png'
plt.savefig(str(output_file), bbox_inches='tight')
plt.clf()

#######################################################################################################################
# Plot u_y
#######################################################################################################################
logger.info('Plotting CJP displacement u_y …')
# Matplotlib plot
number_colors = 120
number_labes = 5
legend_limit_max = 0.1
legend_limit_min = -0.1
cm = 'coolwarm'

# Define contour and label vectors
contour_vector = np.linspace(legend_limit_min, legend_limit_max, number_colors, endpoint=True)
label_vector = np.linspace(legend_limit_min, legend_limit_max, number_labes, endpoint=True)
label_vector = np.round(label_vector, 2)

# Plot the displacement field
plt.clf()
fig = plt.figure(1)
ax = fig.add_subplot(111)

# Plot the crack tip field
plot = ax.contourf(x_grid, y_grid, disp_y, contour_vector, extend='both', cmap=cm)
# Highlight the crack path
ax.plot([0, np.min(x_grid)], [0, 0], 'k', linewidth=2)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(plot, ticks=label_vector,
             cax=cax,
             label='$u_y$ [$\\mathrm{mm}$]')
ax.set_xlabel('$x$ [$\\mathrm{mm}$]')
ax.set_ylabel('$y$ [$\\mathrm{mm}$]')
ax.axis('image')
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)

fig.suptitle('CJP field', y=0.95)
ax.set_title(
    f"$K_F = {K_F:.2f} \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, K_R = {K_R:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, K_S = {K_S:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}$\n"
    f"$K_{{II}} = {K_II:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, T = {T:.2f}  \\, \\mathrm{{N \\cdot mm^{{-2}}}}$",
    fontsize=14)

output_file = OUTPUT_PATH / 'CJP_field_uy.png'
plt.savefig(str(output_file), bbox_inches='tight')
plt.clf()

#######################################################################################################################
# Plot sigma_xx
#######################################################################################################################
logger.info('Plotting CJP stress sigma_xx …')
# Matplotlib plot
number_colors = 120
number_labes = 5
legend_limit_max = 50
legend_limit_min = -50
cm = 'coolwarm'

# Define contour and label vectors
contour_vector = np.linspace(legend_limit_min, legend_limit_max, number_colors, endpoint=True)
label_vector = np.linspace(legend_limit_min, legend_limit_max, number_labes, endpoint=True)
label_vector = np.round(label_vector, 2)

# Plot the displacement field
plt.clf()
fig = plt.figure(1)
ax = fig.add_subplot(111)

# Plot the crack tip field
plot = ax.contourf(x_grid, y_grid, sigma_xx, contour_vector, extend='both', cmap=cm)
# Highlight the crack path
ax.plot([0, np.min(x_grid)], [0, 0], 'k', linewidth=2)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(plot, ticks=label_vector,
             cax=cax,
             label='$\\sigma_{xx}$ [$\\mathrm{MPa}$]')
ax.set_xlabel('$x$ [$\\mathrm{mm}$]')
ax.set_ylabel('$y$ [$\\mathrm{mm}$]')
ax.axis('image')
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)

fig.suptitle('CJP field', y=0.95)
ax.set_title(
    f"$K_F = {K_F:.2f} \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, K_R = {K_R:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, K_S = {K_S:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}$\n"
    f"$K_{{II}} = {K_II:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, T = {T:.2f}  \\, \\mathrm{{N \\cdot mm^{{-2}}}}$",
    fontsize=14)

output_file = OUTPUT_PATH / 'CJP_field_sigma_xx.png'
plt.savefig(str(output_file), bbox_inches='tight')
plt.clf()

#######################################################################################################################
# Plot sigma_yy
#######################################################################################################################
logger.info('Plotting CJP stress sigma_yy …')
# Matplotlib plot
number_colors = 120
number_labes = 5
legend_limit_max = 100
legend_limit_min = -100
cm = 'coolwarm'

# Define contour and label vectors
contour_vector = np.linspace(legend_limit_min, legend_limit_max, number_colors, endpoint=True)
label_vector = np.linspace(legend_limit_min, legend_limit_max, number_labes, endpoint=True)
label_vector = np.round(label_vector, 2)

# Plot the displacement field
plt.clf()
fig = plt.figure(1)
ax = fig.add_subplot(111)

# Plot the crack tip field
plot = ax.contourf(x_grid, y_grid, sigma_yy, contour_vector, extend='both', cmap=cm)
# Highlight the crack path
ax.plot([0, np.min(x_grid)], [0, 0], 'k', linewidth=2)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(plot, ticks=label_vector,
             cax=cax,
             label='$\\sigma_{yy}$ [$\\mathrm{MPa}$]')
ax.set_xlabel('$x$ [$\\mathrm{mm}$]')
ax.set_ylabel('$y$ [$\\mathrm{mm}$]')
ax.axis('image')
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)

fig.suptitle('CJP field', y=0.95)
ax.set_title(
    f"$K_F = {K_F:.2f} \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, K_R = {K_R:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, K_S = {K_S:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}$\n"
    f"$K_{{II}} = {K_II:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, T = {T:.2f}  \\, \\mathrm{{N \\cdot mm^{{-2}}}}$",
    fontsize=14)

output_file = OUTPUT_PATH / 'CJP_field_sigma_yy.png'
plt.savefig(str(output_file), bbox_inches='tight')
plt.clf()

#######################################################################################################################
# Plot sigma_xy
#######################################################################################################################
logger.info('Plotting CJP stress sigma_xy …')
# Matplotlib plot
number_colors = 120
number_labes = 5
legend_limit_max = 20
legend_limit_min = -20
cm = 'coolwarm'

# Define contour and label vectors
contour_vector = np.linspace(legend_limit_min, legend_limit_max, number_colors, endpoint=True)
label_vector = np.linspace(legend_limit_min, legend_limit_max, number_labes, endpoint=True)
label_vector = np.round(label_vector, 2)

# Plot the displacement field
plt.clf()
fig = plt.figure(1)
ax = fig.add_subplot(111)

# Plot the crack tip field
plot = ax.contourf(x_grid, y_grid, sigma_xy, contour_vector, extend='both', cmap=cm)
# Highlight the crack path
ax.plot([0, np.min(x_grid)], [0, 0], 'k', linewidth=2)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(plot, ticks=label_vector,
             cax=cax,
             label='$\\sigma_{xy}$ [$\\mathrm{MPa}$]')
ax.set_xlabel('$x$ [$\\mathrm{mm}$]')
ax.set_ylabel('$y$ [$\\mathrm{mm}$]')
ax.axis('image')
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)

fig.suptitle('CJP field', y=0.95)
ax.set_title(
    f"$K_F = {K_F:.2f} \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, K_R = {K_R:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, K_S = {K_S:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}$\n"
    f"$K_{{II}} = {K_II:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, T = {T:.2f}  \\, \\mathrm{{N \\cdot mm^{{-2}}}}$",
    fontsize=14)

output_file = OUTPUT_PATH / 'CJP_field_sigma_xy.png'
plt.savefig(str(output_file), bbox_inches='tight')
plt.clf()

#######################################################################################################################
# Plot sigma_eqv
#######################################################################################################################
logger.info('Plotting CJP equivalent stress sigma_eqv …')
# Matplotlib plot
number_colors = 120
number_labes = 5
legend_limit_max = 100
legend_limit_min = 0
cm = 'coolwarm'

# Define contour and label vectors
contour_vector = np.linspace(legend_limit_min, legend_limit_max, number_colors, endpoint=True)
label_vector = np.linspace(legend_limit_min, legend_limit_max, number_labes, endpoint=True)
label_vector = np.round(label_vector, 2)

# Plot the displacement field
plt.clf()
fig = plt.figure(1)
ax = fig.add_subplot(111)

# Plot the crack tip field
plot = ax.contourf(x_grid, y_grid, sigma_vm, contour_vector, extend='max', cmap=cm)
# Highlight the crack path
ax.plot([0, np.min(x_grid)], [0, 0], 'k', linewidth=2)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(plot, ticks=label_vector,
             cax=cax,
             label='$\\sigma_{eqv}$ [$\\mathrm{MPa}$]')
ax.set_xlabel('$x$ [$\\mathrm{mm}$]')
ax.set_ylabel('$y$ [$\\mathrm{mm}$]')
ax.axis('image')
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)

fig.suptitle('CJP field', y=0.95)
ax.set_title(
    f"$K_F = {K_F:.2f} \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, K_R = {K_R:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, K_S = {K_S:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}$\n"
    f"$K_{{II}} = {K_II:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, T = {T:.2f}  \\, \\mathrm{{N \\cdot mm^{{-2}}}}$",
    fontsize=14)

output_file = OUTPUT_PATH / 'CJP_field_sigma_eqv.png'
plt.savefig(str(output_file), bbox_inches='tight')
plt.clf()
