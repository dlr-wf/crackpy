"""
This script plots the Williams field for different number of terms.

Needed:
    - Williams expansion coefficients A and B

Output:
    - Plots of the Williams field

"""

import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from crackpy.fracture_analysis.crack_tip import williams_stress_field, williams_displ_field
from crackpy.fracture_analysis.optimization import Optimization
from crackpy.structure_elements.material import Material

# Set matplotlib settings
plt.rcParams.update({
    "font.size": 20,
    "text.usetex": True,
    "font.family": "Computer Modern",
    "figure.figsize": [10, 10],
    "figure.dpi": 300
})

# Parameters
K_I = 0  # 23.7082070388 * np.sqrt(1000)  # MPa.mm^0.5
K_II = 23.78 * np.sqrt(1000)  #
T = -7.2178311164  # MPa

a_1 = K_I / np.sqrt(2 * np.pi)
a_2 = T / 4
a_3 = -2.8672068762
a_4 = 0.0290671612

b_1 = -K_II / np.sqrt(2 * np.pi)
b_2 = 0.0
b_3 = 0.0
b_4 = 0.0

# Define the Williams expansion coefficients
coefficients_a = [a_1, a_2, a_3, a_4]
coefficients_b = [b_1, b_2, b_3, b_4]
coefficients_n = [1, 2, 3, 4]

material = Material(E=72000, nu_xy=0.33, sig_yield=350.0)

# Output path
OUTPUT_PATH = os.path.join('Williams_field')

# check if output path exists
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Loop over the number of terms and add one term at a time to the Williams expansion
for index in range(1, len(coefficients_a) + 1):
    for i in range(index):
        print(f'a_{coefficients_n[i]}= {coefficients_a[i]:.2f} N * mm^({-1 - coefficients_n[i] / 2})'
              f', b_{coefficients_n[i]}= {coefficients_b[i]:.2f} N * mm^({-1 - coefficients_n[i] / 2})')

    min_radius = 0.01
    max_radius = 100.0
    tick_size = 0.01

    r_grid, phi_grid = np.mgrid[min_radius:max_radius:tick_size, -np.pi:np.pi:tick_size]

    # Calculate the stress and displacement fields
    sigma_xx, sigma_yy, sigma_xy = williams_stress_field(coefficients_a[:index],
                                                         coefficients_b[:index],
                                                         coefficients_n[:index],
                                                         phi_grid, r_grid)
    disp_x, disp_y = williams_displ_field(coefficients_a[:index],
                                          coefficients_b[:index],
                                          coefficients_n[:index],
                                          phi_grid, r_grid, material)

    sigma_vm = np.sqrt(sigma_xx ** 2 + sigma_yy ** 2 - sigma_xx * sigma_yy + 3 * sigma_xy ** 2)

    x_grid, y_grid = Optimization.make_cartesian(r_grid, phi_grid)

    ####################################################################################################################
    # Plot u_x
    ####################################################################################################################
    print(f'Plotting Williams u_x for {index} {"term" if index == 1 else "terms"}.')
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
    title_str_a = ''
    title_str_b = ''
    for i in range(index):
        title_str_a += f'$a_{{{coefficients_n[i]}}} = {coefficients_a[i]:.2f} \\, \\mathrm{{N \\cdot mm^{{{-1 - coefficients_n[i] / 2}}}}}$'
        title_str_b += f'$b_{{{coefficients_n[i]}}} = {coefficients_b[i]:.2f} \\, \\mathrm{{N \\cdot mm^{{{-1 - coefficients_n[i] / 2}}}}}$'
        if i < index - 1:
            title_str_a += ', '
            title_str_b += ', '

    fig.suptitle(f'Williams field with {index} {"term" if index == 1 else "terms"}', y=0.95)
    ax.set_title(title_str_a + '\n' + title_str_b, fontsize=14)

    output_file = os.path.join(OUTPUT_PATH, f'{index}_Williams_u_x.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.clf()

    ####################################################################################################################
    # Plot u_y
    ####################################################################################################################
    print(f'Plotting Williams u_y for {index} {"term" if index == 1 else "terms"}.')
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
    title_str_a = ''
    title_str_b = ''
    for i in range(index):
        title_str_a += f'$a_{{{coefficients_n[i]}}} = {coefficients_a[i]:.2f} \\, \\mathrm{{N \\cdot mm^{{{-1 - coefficients_n[i] / 2}}}}}$'
        title_str_b += f'$b_{{{coefficients_n[i]}}} = {coefficients_b[i]:.2f} \\, \\mathrm{{N \\cdot mm^{{{-1 - coefficients_n[i] / 2}}}}}$'
        if i < index - 1:
            title_str_a += ', '
            title_str_b += ', '

    fig.suptitle(f'Williams field with {index} {"term" if index == 1 else "terms"}', y=0.95)
    ax.set_title(title_str_a + '\n' + title_str_b, fontsize=14)

    output_file = os.path.join(OUTPUT_PATH, f'{index}_Williams_u_y.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.clf()

    ####################################################################################################################
    # Plot sigma_xx
    ####################################################################################################################
    print(f'Plotting Williams sigma_xx for {index} {"term" if index == 1 else "terms"}.')
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
    title_str_a = ''
    title_str_b = ''
    for i in range(index):
        title_str_a += f'$a_{{{coefficients_n[i]}}} = {coefficients_a[i]:.2f} \\, \\mathrm{{N \\cdot mm^{{{-1 - coefficients_n[i] / 2}}}}}$'
        title_str_b += f'$b_{{{coefficients_n[i]}}} = {coefficients_b[i]:.2f} \\, \\mathrm{{N \\cdot mm^{{{-1 - coefficients_n[i] / 2}}}}}$'
        if i < index - 1:
            title_str_a += ', '
            title_str_b += ', '

    fig.suptitle(f'Williams field with {index} {"term" if index == 1 else "terms"}', y=0.95)
    ax.set_title(title_str_a + '\n' + title_str_b, fontsize=14)

    output_file = os.path.join(OUTPUT_PATH, f'{index}_Williams_sigma_xx.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.clf()

    ####################################################################################################################
    # Plot sigma_yy
    ####################################################################################################################
    print(f'Plotting Williams sigma_yy for {index} {"term" if index == 1 else "terms"}.')
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
    title_str_a = ''
    title_str_b = ''
    for i in range(index):
        title_str_a += f'$a_{{{coefficients_n[i]}}} = {coefficients_a[i]:.2f} \\, \\mathrm{{N \\cdot mm^{{{-1 - coefficients_n[i] / 2}}}}}$'
        title_str_b += f'$b_{{{coefficients_n[i]}}} = {coefficients_b[i]:.2f} \\, \\mathrm{{N \\cdot mm^{{{-1 - coefficients_n[i] / 2}}}}}$'
        if i < index - 1:
            title_str_a += ', '
            title_str_b += ', '

    fig.suptitle(f'Williams field with {index} {"term" if index == 1 else "terms"}', y=0.95)
    ax.set_title(title_str_a + '\n' + title_str_b, fontsize=14)

    output_file = os.path.join(OUTPUT_PATH, f'{index}_Williams_sigma_yy.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.clf()

    ####################################################################################################################
    # Plot sigma_xy
    ####################################################################################################################
    print(f'Plotting Williams sigma_xy for {index} {"term" if index == 1 else "terms"}.')
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
    title_str_a = ''
    title_str_b = ''
    for i in range(index):
        title_str_a += f'$a_{{{coefficients_n[i]}}} = {coefficients_a[i]:.2f} \\, \\mathrm{{N \\cdot mm^{{{-1 - coefficients_n[i] / 2}}}}}$'
        title_str_b += f'$b_{{{coefficients_n[i]}}} = {coefficients_b[i]:.2f} \\, \\mathrm{{N \\cdot mm^{{{-1 - coefficients_n[i] / 2}}}}}$'
        if i < index - 1:
            title_str_a += ', '
            title_str_b += ', '

    fig.suptitle(f'Williams field with {index} {"term" if index == 1 else "terms"}', y=0.95)
    ax.set_title(title_str_a + '\n' + title_str_b, fontsize=14)

    output_file = os.path.join(OUTPUT_PATH, f'{index}_Williams_sigma_xy.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.clf()

    ####################################################################################################################
    # Plot sigma_eqv
    ####################################################################################################################
    print(f'Plotting Williams sigma_eqv for {index} {"term" if index == 1 else "terms"}.')
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
    title_str_a = ''
    title_str_b = ''
    for i in range(index):
        title_str_a += f'$a_{{{coefficients_n[i]}}} = {coefficients_a[i]:.2f} \\, \\mathrm{{N \\cdot mm^{{{-1 - coefficients_n[i] / 2}}}}}$'
        title_str_b += f'$b_{{{coefficients_n[i]}}} = {coefficients_b[i]:.2f} \\, \\mathrm{{N \\cdot mm^{{{-1 - coefficients_n[i] / 2}}}}}$'
        if i < index - 1:
            title_str_a += ', '
            title_str_b += ', '

    fig.suptitle(f'Williams field with {index} {"term" if index == 1 else "terms"}', y=0.95)
    ax.set_title(title_str_a + '\n' + title_str_b, fontsize=14)

    output_file = os.path.join(OUTPUT_PATH, f'{index}_Williams_sigma_eqv.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.clf()
