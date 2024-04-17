"""

    Example script:
        Fracture analysis based on the Williams field.

    Input:
        - Output folder
        - Nodemap file
        - Nodemap structure
        - Material properties
        - Integral properties
        - Optimization properties
        - Crack tip position

    Output:
        - Fracture Analysis results (plots, txt-files)

"""

import os

import numpy as np
from matplotlib import pyplot as plt
from crackpy.fracture_analysis.analysis import FractureAnalysis
from crackpy.fracture_analysis.data_processing import InputData, CrackTipInfo
from crackpy.fracture_analysis.line_integration import IntegralProperties
from crackpy.fracture_analysis.optimization import OptimizationProperties
from crackpy.fracture_analysis.plot import PlotSettings, Plotter
from crackpy.fracture_analysis.write import OutputWriter
from crackpy.structure_elements.data_files import Nodemap
from crackpy.structure_elements.material import Material
from crackpy.fracture_analysis.crack_tip import williams_displ_field, williams_stress_field

########################
# INPUT specifications #
########################


OUT_FOLDER = 'Fracture_Analysis_Williams_results'

material = Material(E=72000, nu_xy=0.33, sig_yield=350)

# Parameters
K_I = 10 * np.sqrt(1000)  # MPa * sqrt(m)
K_II = 20 * np.sqrt(1000)  # MPa * sqrt(m)
T = 40  # MPa

a_1 = K_I / np.sqrt(2 * np.pi)
a_2 = T / 4

b_1 = -K_II / np.sqrt(2 * np.pi)
b_2 = 0.0

# Define the Williams expansion coefficients
coefficients_a = [a_1, a_2]
coefficients_b = [b_1, b_2]
coefficients_n = [1, 2]

# Create the mesh
steps = 200
x_coordinates = np.linspace(-10, 10, steps, endpoint=True)
y_coordinates = np.linspace(-10, 10, steps, endpoint=True)
x_mesh, y_mesh = np.meshgrid(x_coordinates, y_coordinates)

# Define integral properties
int_props = IntegralProperties(
    number_of_paths=10,
    # number_of_nodes=100,
    integral_tick_size=0.25,

    bottom_offset=-0.0,
    top_offset=0.0,

    integral_size_left=-1,
    integral_size_right=1,
    integral_size_top=1,
    integral_size_bottom=-1,

    paths_distance_top=0.5,
    paths_distance_left=0.5,
    paths_distance_right=0.5,
    paths_distance_bottom=0.5,

    mask_tolerance=None,

    buckner_williams_terms=[-1, 1, 2, 3, 4, 5]
)

# Define fitting properties
opt_props = OptimizationProperties(
    angle_gap=10,
    min_radius=2,
    max_radius=10,
    tick_size=0.01,
    terms=[-1, 0, 1, 2, 3, 4, 5],
)

ct = CrackTipInfo(0, 0, 0, 'right')

# Compute the Williams displacement field
r_grid = np.sqrt(x_mesh ** 2 + y_mesh ** 2)
phi_grid = np.arctan2(y_mesh, x_mesh)

disp_u_mesh, disp_v_mesh = williams_displ_field(coefficients_a,
                                                coefficients_b,
                                                coefficients_n,
                                                phi_grid, r_grid, material)


# Compute strains with Hook's law or with the gradients of the displacements
method = 'Hook'  # 'Hook' or 'Gradient'

# Compute epsilon_xx
if method == 'Gradient':
    gap = 0
    dist = x_coordinates[1] - x_coordinates[0]
    eps_xx = np.zeros_like(x_mesh)
    eps_xx[0:int(steps / 2) - gap, 0:int(steps / 2)] = np.gradient(
        disp_u_mesh[0:int(steps / 2) - gap, 0:int(steps / 2)],
        dist, axis=1)
    eps_xx[int(steps / 2) + gap:, 0:int(steps / 2)] = np.gradient(disp_u_mesh[int(steps / 2) + gap:, 0:int(steps / 2)],
                                                                  dist, axis=1)
    eps_xx[:, int(steps / 2):] = np.gradient(disp_u_mesh[:, int(steps / 2):], dist, axis=1)

    # Compute epsilon_yy
    eps_yy = np.zeros_like(x_mesh)
    eps_yy[0:int(steps / 2) - gap, 0:int(steps / 2)] = np.gradient(
        disp_v_mesh[0:int(steps / 2) - gap, 0:int(steps / 2)],
        dist, axis=0)
    eps_yy[int(steps / 2) + gap:, 0:int(steps / 2)] = np.gradient(disp_v_mesh[int(steps / 2) + gap:, 0:int(steps / 2)],
                                                                  dist, axis=0)
    eps_yy[:, int(steps / 2):] = np.gradient(disp_v_mesh[:, int(steps / 2):], dist, axis=0)

    # Compute epsilon_xy
    eps_xy = np.zeros_like(x_mesh)
    eps_xy[0:int(steps / 2) - gap, 0:int(steps / 2)] = 0.5 * (
            np.gradient(disp_u_mesh[0:int(steps / 2) - gap, 0:int(steps / 2)], dist, axis=0) +
            np.gradient(disp_v_mesh[0:int(steps / 2) - gap, 0:int(steps / 2)], dist, axis=1))
    eps_xy[int(steps / 2) + gap:, 0:int(steps / 2)] = 0.5 * (
            np.gradient(disp_u_mesh[int(steps / 2) + gap:, 0:int(steps / 2)], dist, axis=0) +
            np.gradient(disp_v_mesh[int(steps / 2) + gap:, 0:int(steps / 2)], dist, axis=1))
    eps_xy[:, int(steps / 2):] = 0.5 * (np.gradient(disp_u_mesh[:, int(steps / 2):], dist, axis=0) +
                                        np.gradient(disp_v_mesh[:, int(steps / 2):], dist, axis=1))
    eps_xx = eps_xx.flatten()
    eps_yy = eps_yy.flatten()
    eps_xy = eps_xy.flatten()

elif method == 'Hook':
    sigma_xx, sigma_yy, sigma_xy = williams_stress_field(coefficients_a,
                                                         coefficients_b,
                                                         coefficients_n,
                                                         phi_grid, r_grid)
    sigma_xx = sigma_xx.flatten()
    sigma_yy = sigma_yy.flatten()
    sigma_xy = sigma_xy.flatten()
    eps_xx, eps_yy, eps_xy = np.dot(material.inverse_stiffness_matrix, [sigma_xx, sigma_yy, sigma_xy])

else:
    raise ValueError('Method not recognized')

# Create the InputData object
input_data = InputData()
input_data.nodemap_name = f'Williams_{method}.txt'
input_data.coor_x = x_mesh.flatten()
input_data.coor_y = y_mesh.flatten()
input_data.coor_z = np.zeros_like(x_mesh.flatten())
input_data.facet_id = np.arange(len(x_mesh.flatten()))
input_data.disp_x = disp_u_mesh.flatten()
input_data.disp_y = disp_v_mesh.flatten()
input_data.disp_z = np.zeros_like(x_mesh.flatten())
input_data.eps_x = eps_xx
input_data.eps_y = eps_yy
input_data.eps_xy = eps_xy
input_data.calc_stresses(material)
input_data.transform_data(ct.crack_tip_x, ct.crack_tip_y, ct.crack_tip_angle)
mesh = input_data.to_vtk(os.path.join(OUT_FOLDER, 'vtk'), metadata=False, alpha=0.5)

# Plot mesh data and save to file using pyvista
mesh.plot(scalars='eps_vm [%]',
          clim=[0, 0.5],
          cpos='xy',
          cmap='jet',
          show_bounds=True,
          lighting=True,
          show_scalar_bar=True,
          scalar_bar_args={'vertical': True},
          screenshot=os.path.join(OUT_FOLDER, 'vtk', f"pyvista_plot_{method}.png"),
          off_screen=True
          )

analysis = FractureAnalysis(
    material=Material(),
    nodemap=input_data.nodemap_name,
    data=input_data,
    crack_tip_info=ct,
    integral_properties=int_props,
    optimization_properties=opt_props
)
analysis.run()

# Set colormap
plt.rcParams['image.cmap'] = 'coolwarm'
plt.rcParams['figure.dpi'] = 100

# Plotting
plot_sets = PlotSettings(background='sig_vm', min_value=0, max_value=material.sig_yield, extend='max')
plotter = Plotter(path=os.path.join(OUT_FOLDER, 'plots'), fracture_analysis=analysis, plot_sets=plot_sets)
plotter.plot()

writer = OutputWriter(path=os.path.join(OUT_FOLDER, 'results'), fracture_analysis=analysis)
writer.write_header()
writer.write_results()
writer.write_json(path=os.path.join(OUT_FOLDER, 'json'))
