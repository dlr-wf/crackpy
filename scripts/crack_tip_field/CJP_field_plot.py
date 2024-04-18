"""
This script plots the equivalent stress field using the CJP model from the paper
"Extension of the CJP model to mixed mode I and mode II" (2013).

Needed:
    - CJP coefficients

Output:
    - Plot of the equivalent stress field

"""

import os

from crackpy.fracture_analysis.crack_tip import cjp_stress_field, cjp_displ_field
from crackpy.structure_elements.material import Material
from crackpy.fracture_analysis.optimization import Optimization

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set matplotlib settings
plt.rcParams.update({
    "font.size": 20,
    "text.usetex": True,
    "font.family": "Computer Modern",
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

print(f'K_F = {K_F:.2f} MPa m^0.5')
print(f'K_R = {K_R:.2f} MPa m^0.5')
print(f'K_S = {K_S:.2f} MPa m^0.5')
print(f'K_II = {K_II:.2f} MPa m^0.5')
print(f'T = {T:.2f} MPa')

material = Material(E=72000, nu_xy=0.33, sig_yield=350.0)

# Output path
OUTPUT_PATH = os.path.join('CJP_field')

# check if output path exists
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Define the material
material = Material(E=72000, nu_xy=0.33, sig_yield=350.0)

# Plot
min_radius = 0.01
max_radius = 100.0
tick_size = 0.01

r_grid, phi_grid = np.mgrid[min_radius:max_radius:tick_size, -np.pi:np.pi:tick_size]
coeffs = A_r, B_r, B_i, C, E

# Calculate the stress and displacement fields
sigma_xx, sigma_yy, sigma_xy = cjp_stress_field(coeffs, phi_grid, r_grid)
disp_x, disp_y = cjp_displ_field(coeffs, phi_grid, r_grid, material)

sigma_vm = np.sqrt(sigma_xx ** 2 + sigma_yy ** 2 - sigma_xx * sigma_yy + 3 * sigma_xy ** 2)

x_grid, y_grid = Optimization.make_cartesian(r_grid, phi_grid)

#######################################################################################################################
# Plot u_x
#######################################################################################################################
print('Plotting u_x')
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

fig.suptitle(f'CJP field', y=0.95)
ax.set_title(
    f"$K_F = {K_F:.2f} \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, K_R = {K_R:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, K_S = {K_S:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}$\n"
    f"$K_{{II}} = {K_II:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, T = {T:.2f}  \\, \\mathrm{{N \\cdot mm^{{-2}}}}$",
    fontsize=14)

output_file = os.path.join(OUTPUT_PATH, f'CJP_field_ux.png')
plt.savefig(output_file, bbox_inches='tight')
plt.clf()

#######################################################################################################################
# Plot u_y
#######################################################################################################################
print('Plotting u_y')
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

fig.suptitle(f'CJP field', y=0.95)
ax.set_title(
    f"$K_F = {K_F:.2f} \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, K_R = {K_R:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, K_S = {K_S:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}$\n"
    f"$K_{{II}} = {K_II:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, T = {T:.2f}  \\, \\mathrm{{N \\cdot mm^{{-2}}}}$",
    fontsize=14)

output_file = os.path.join(OUTPUT_PATH, f'CJP_field_uy.png')
plt.savefig(output_file, bbox_inches='tight')
plt.clf()

#######################################################################################################################
# Plot sigma_xx
#######################################################################################################################
print('Plotting sigma_xx')
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

fig.suptitle(f'CJP field', y=0.95)
ax.set_title(
    f"$K_F = {K_F:.2f} \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, K_R = {K_R:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, K_S = {K_S:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}$\n"
    f"$K_{{II}} = {K_II:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, T = {T:.2f}  \\, \\mathrm{{N \\cdot mm^{{-2}}}}$",
    fontsize=14)

output_file = os.path.join(OUTPUT_PATH, f'CJP_field_sigma_xx.png')
plt.savefig(output_file, bbox_inches='tight')
plt.clf()

#######################################################################################################################
# Plot sigma_yy
#######################################################################################################################
print('Plotting sigma_yy')
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

fig.suptitle(f'CJP field', y=0.95)
ax.set_title(
    f"$K_F = {K_F:.2f} \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, K_R = {K_R:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, K_S = {K_S:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}$\n"
    f"$K_{{II}} = {K_II:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, T = {T:.2f}  \\, \\mathrm{{N \\cdot mm^{{-2}}}}$",
    fontsize=14)

output_file = os.path.join(OUTPUT_PATH, f'CJP_field_sigma_yy.png')
plt.savefig(output_file, bbox_inches='tight')
plt.clf()

#######################################################################################################################
# Plot sigma_xy
#######################################################################################################################
print('Plotting sigma_xy')
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

fig.suptitle(f'CJP field', y=0.95)
ax.set_title(
    f"$K_F = {K_F:.2f} \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, K_R = {K_R:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, K_S = {K_S:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}$\n"
    f"$K_{{II}} = {K_II:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, T = {T:.2f}  \\, \\mathrm{{N \\cdot mm^{{-2}}}}$",
    fontsize=14)

output_file = os.path.join(OUTPUT_PATH, f'CJP_field_sigma_xy.png')
plt.savefig(output_file, bbox_inches='tight')
plt.clf()

#######################################################################################################################
# Plot sigma_eqv
#######################################################################################################################
print('Plotting sigma_eqv')
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

fig.suptitle(f'CJP field', y=0.95)
ax.set_title(
    f"$K_F = {K_F:.2f} \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, K_R = {K_R:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, K_S = {K_S:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}$\n"
    f"$K_{{II}} = {K_II:.2f}  \\, \\mathrm{{N \\cdot mm^{{-1.5}}}}, T = {T:.2f}  \\, \\mathrm{{N \\cdot mm^{{-2}}}}$",
    fontsize=14)

output_file = os.path.join(OUTPUT_PATH, f'CJP_field_sigma_eqv.png')
plt.savefig(output_file, bbox_inches='tight')
plt.clf()
