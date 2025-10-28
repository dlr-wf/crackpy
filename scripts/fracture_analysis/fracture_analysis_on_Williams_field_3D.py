"""

    Example script:
        Fracture analysis for synthetic data.

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

import logging
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from crackpy.fracture_analysis.analysis import FractureAnalysis
from crackpy.fracture_analysis.crack_tip import williams_displ_field_z, williams_displ_field_xy
from crackpy.fracture_analysis.line_integration import IntegralProperties
from crackpy.fracture_analysis.optimization import OptimizationProperties
from crackpy.input.crack_tip_info import CrackTipInfo
from crackpy.input.input_data import InputData
from crackpy.results.plot import PlotSettings, Plotter
from crackpy.results.read import OutputReader
from crackpy.results.write import OutputWriter
from crackpy.structure_elements.material import Material

# Logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_FOLDER = PROJECT_ROOT / 'Fracture_Analysis_Williams_results_3D'
OUT_FOLDER.mkdir(parents=True, exist_ok=True)


def main():
    ################################
    # Generation of synthetic data #
    ################################

    K_I = 10 * np.sqrt(1000)  # MPa * sqrt(m)
    K_II = -20 * np.sqrt(1000)  # MPa * sqrt(m)
    K_III = 30 * np.sqrt(1000)  # MPa * sqrt(m)
    T = 40  # MPa
    A_1 = K_I / np.sqrt(2 * np.pi)
    A_2 = T / 4.0
    B_1 = - K_II / np.sqrt(2 * np.pi)
    C_1 = K_III / np.sqrt(0.5 * np.pi)
    A = [A_1, A_2]
    B = [B_1, 0]
    C = [C_1, 0]

    material = Material(E=72000, nu_xy=0.33, sig_yield=350)

    steps = 500
    x_coordinates = np.linspace(-25, 25, steps, endpoint=True)
    y_coordinates = np.linspace(-25, 25, steps, endpoint=True)
    x_mesh, y_mesh = np.meshgrid(x_coordinates, y_coordinates)

    r_grid = np.sqrt(x_mesh ** 2 + y_mesh ** 2)
    phi_grid = np.arctan2(y_mesh, x_mesh)
    terms = [1, 2]
    disp_u_mesh, disp_v_mesh = williams_displ_field_xy(A, B, terms, phi_grid, r_grid, material)
    disp_w_mesh = williams_displ_field_z(C, terms, phi_grid, r_grid, material)

    gap = 2
    dist = x_coordinates[1] - x_coordinates[0]
    eps_xx = np.zeros_like(x_mesh)
    eps_xx[0:int(steps / 2) - gap, 0:int(steps / 2)] = np.gradient(
        disp_u_mesh[0:int(steps / 2) - gap, 0:int(steps / 2)],
        dist, axis=1)
    eps_xx[int(steps / 2) + gap:, 0:int(steps / 2)] = np.gradient(disp_u_mesh[int(steps / 2) + gap:, 0:int(steps / 2)],
                                                                  dist, axis=1)
    eps_xx[:, int(steps / 2):] = np.gradient(disp_u_mesh[:, int(steps / 2):], dist, axis=1)

    eps_yy = np.zeros_like(x_mesh)
    eps_yy[0:int(steps / 2) - gap, 0:int(steps / 2)] = np.gradient(
        disp_v_mesh[0:int(steps / 2) - gap, 0:int(steps / 2)],
        dist, axis=0)
    eps_yy[int(steps / 2) + gap:, 0:int(steps / 2)] = np.gradient(disp_v_mesh[int(steps / 2) + gap:, 0:int(steps / 2)],
                                                                  dist, axis=0)
    eps_yy[:, int(steps / 2):] = np.gradient(disp_v_mesh[:, int(steps / 2):], dist, axis=0)

    eps_xy = np.zeros_like(x_mesh)
    eps_xy[0:int(steps / 2) - gap, 0:int(steps / 2)] = 0.5 * (
            np.gradient(disp_u_mesh[0:int(steps / 2) - gap, 0:int(steps / 2)], dist, axis=0) +
            np.gradient(disp_v_mesh[0:int(steps / 2) - gap, 0:int(steps / 2)], dist, axis=1))
    eps_xy[int(steps / 2) + gap:, 0:int(steps / 2)] = 0.5 * (
            np.gradient(disp_u_mesh[int(steps / 2) + gap:, 0:int(steps / 2)], dist, axis=0) +
            np.gradient(disp_v_mesh[int(steps / 2) + gap:, 0:int(steps / 2)], dist, axis=1))
    eps_xy[:, int(steps / 2):] = 0.5 * (np.gradient(disp_u_mesh[:, int(steps / 2):], dist, axis=0) +
                                        np.gradient(disp_v_mesh[:, int(steps / 2):], dist, axis=1))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    c1 = axes[0].contourf(x_mesh, y_mesh, disp_u_mesh, levels=100)
    fig.colorbar(c1, ax=axes[0])
    axes[0].set_title('Displacement u')
    c2 = axes[1].contourf(x_mesh, y_mesh, disp_v_mesh, levels=100)
    fig.colorbar(c2, ax=axes[1])
    axes[1].set_title('Displacement v')
    c3 = axes[2].contourf(x_mesh, y_mesh, disp_w_mesh, levels=100)
    fig.colorbar(c3, ax=axes[2])
    axes[2].set_title('Displacement w')
    plt.tight_layout()
    plt.savefig(str(OUT_FOLDER / 'williams_displacement_field.png'))
    plt.close()

    ####################################
    # Fracture Analysis specifications #
    ####################################

    int_props = IntegralProperties(
        number_of_paths=10,
        number_of_nodes=100,

        bottom_offset=-0.05,
        top_offset=0.05,

        integral_size_left=-5,
        integral_size_right=5,
        integral_size_top=5,
        integral_size_bottom=-5,

        paths_distance_top=1,
        paths_distance_left=1,
        paths_distance_right=1,
        paths_distance_bottom=1,

        mask_tolerance=2,

        buckner_williams_terms=[-1, 1, 2, 3, 4, 5]
    )

    opt_props = OptimizationProperties(
        angle_gap=15,
        min_radius=5,
        max_radius=10,
        tick_size=0.01,
        terms=[-3, -2, -1, 0, 1, 2, 3, 4, 5],
    )

    ct = CrackTipInfo(0, 0, 0, 'right')

    ###############
    # Main script #
    ###############

    input_data = InputData()
    input_data.coor_x = x_mesh.flatten()
    input_data.coor_y = y_mesh.flatten()
    input_data.disp_x = disp_u_mesh.flatten()
    input_data.disp_y = disp_v_mesh.flatten()
    input_data.disp_z = disp_w_mesh.flatten()
    input_data.eps_x = eps_xx.flatten()
    input_data.eps_y = eps_yy.flatten()
    input_data.eps_xy = eps_xy.flatten()
    input_data.calc_stresses(material)

    input_data.transform_data(ct.crack_tip_x, ct.crack_tip_y, ct.crack_tip_angle)

    analysis = FractureAnalysis(
        material=material,
        nodemap='williams_synthetic_3D.txt',
        data=input_data,
        crack_tip_info=ct,
        integral_properties=int_props,
        optimization_properties=opt_props
    )
    analysis.run()

    ##########################
    # Plotting & Data Export #
    ##########################

    plt.rcParams['image.cmap'] = 'coolwarm'
    plt.rcParams['figure.dpi'] = 100

    plot_sets = PlotSettings(background='sig_vm', min_value=0, max_value=material.sig_yield, extend='max')
    plotter = Plotter(path=OUT_FOLDER / 'plots', fracture_analysis=analysis, plot_sets=plot_sets)
    plotter.plot()

    writer = OutputWriter(path=OUT_FOLDER / 'results', fracture_analysis=analysis)
    writer.write_header()
    writer.write_results()
    writer.write_json(path=OUT_FOLDER / 'json')

    # Read results and write into CSV file
    reader = OutputReader()
    result_path = OUT_FOLDER / 'results'

    files = [p.name for p in result_path.iterdir()]
    list_of_tags = [
        "CJP_results", "Williams_fit_results", "SIFs_integral", "Bueckner_Chen_integral",
        "Path_SIFs", "Path_Williams_a_n", "Path_Williams_b_n"
    ]
    for file in files:
        if file.endswith(".txt"):
            for tag in list_of_tags:
                reader.read_tag_data(path=result_path, filename=file, tag=tag)

    # Make CSV file
    reader.make_csv_from_results(files="all", output_path=OUT_FOLDER, output_filename='results.csv')


if __name__ == '__main__':
    # Profiling (optional)
    import cProfile, pstats, subprocess, sys
    from datetime import datetime

    script_dir = OUT_FOLDER
    fname = script_dir / f"{datetime.now():%Y%m%d%H%M%S}_profile.prof"

    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()
    pstats.Stats(pr).dump_stats(str(fname))

    subprocess.run([sys.executable, "-m", "snakeviz", str(fname)], check=True)
