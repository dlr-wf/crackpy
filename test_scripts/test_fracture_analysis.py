"""End-to-end fracture-analysis scenarios cover scientific results, pipeline
outputs, plots, and serialized result files.
"""

import os
import shutil
import tempfile
import unittest
from dataclasses import astuple
from pathlib import Path

import numpy as np
import pandas as pd

from crackpy.fracture_analysis.analysis import FractureAnalysis
from crackpy.fracture_analysis.crack_tip import (
    williams_displ_field_xy,
    williams_displ_field_z,
    williams_stress_field,
)
from crackpy.fracture_analysis.line_integration import IntegralProperties
from crackpy.fracture_analysis.optimization import OptimizationProperties
from crackpy.fracture_analysis.pipeline import FractureAnalysisPipeline
from crackpy.input.crack_tip_info import CrackTipInfo
from crackpy.input.input_data import InputData
from crackpy.results.plot import PlotSettings, Plotter
from crackpy.results.read import OutputReader
from crackpy.results.write import OutputWriter
from crackpy.structure_elements.data_files import Nodemap
from crackpy.structure_elements.material import Material


def _synthetic_williams_data(
        material: Material,
        *,
        mode_i_sif: float = 10.0,
        mode_ii_sif: float = 20.0,
        mode_iii_sif: float = 30.0,
        t_stress: float = 40.0,
) -> InputData:
    """Create a regular synthetic Williams displacement and strain field.

    Args:
        material: Linear-elastic material used to evaluate the Williams field.
        mode_i_sif: Prescribed Mode I SIF in MPa sqrt(m).
        mode_ii_sif: Prescribed Mode II SIF in MPa sqrt(m).
        mode_iii_sif: Prescribed Mode III SIF in MPa sqrt(m).
        t_stress: Prescribed T-Stress in MPa.

    Returns:
        Crack-tip-centered field data on a regular square grid.
    """
    steps = 500
    coordinates = np.linspace(-25, 25, steps, endpoint=True)
    x_mesh, y_mesh = np.meshgrid(coordinates, coordinates)

    # Williams displacement fields use MPa sqrt(mm), whereas the prescribed
    # Stress Intensity Factors use the public MPa sqrt(m) convention.
    mode_i_sif_mm = mode_i_sif * np.sqrt(1000)
    mode_ii_sif_mm = mode_ii_sif * np.sqrt(1000)
    mode_iii_sif_mm = mode_iii_sif * np.sqrt(1000)
    symmetric_coefficients = [
        mode_i_sif_mm / np.sqrt(2 * np.pi),
        t_stress / 4.0,
    ]
    antisymmetric_coefficients = [
        -mode_ii_sif_mm / np.sqrt(2 * np.pi),
        0,
    ]
    out_of_plane_coefficients = [
        mode_iii_sif_mm / np.sqrt(0.5 * np.pi),
        0,
    ]
    terms = [1, 2]
    radius = np.sqrt(x_mesh ** 2 + y_mesh ** 2)
    angle = np.arctan2(y_mesh, x_mesh)
    displacement_x, displacement_y = williams_displ_field_xy(
        symmetric_coefficients,
        antisymmetric_coefficients,
        terms,
        angle,
        radius,
        material,
    )
    displacement_z = williams_displ_field_z(
        out_of_plane_coefficients,
        terms,
        angle,
        radius,
        material,
    )
    stress_x, stress_y, stress_xy = williams_stress_field(
        symmetric_coefficients,
        antisymmetric_coefficients,
        terms,
        angle,
        radius,
    )

    data = InputData()
    data.coor_x = x_mesh.flatten()
    data.coor_y = y_mesh.flatten()
    data.disp_x = displacement_x.flatten()
    data.disp_y = displacement_y.flatten()
    data.disp_z = displacement_z.flatten()
    # Plane-stress compliance maps the analytical Williams stresses to the
    # tensorial strain convention consumed by the line-integral techniques.
    data.eps_x = ((stress_x - material.nu_xy * stress_y) / material.E).flatten()
    data.eps_y = ((stress_y - material.nu_xy * stress_x) / material.E).flatten()
    data.eps_xy = (stress_xy / (2 * material.G)).flatten()
    spacing = coordinates[1] - coordinates[0]
    data.eps_xz = np.gradient(displacement_z, spacing, axis=1).flatten()
    data.eps_yz = np.gradient(displacement_z, spacing, axis=0).flatten()
    data.calc_eps_vm()
    data.calc_stresses(material)
    data.sigma_xz = material.G * data.eps_xz
    data.sigma_yz = material.G * data.eps_yz
    return data


class TestFractureAnalysis(unittest.TestCase):
    def setUp(self):
        # Find project root iteratively by searching for pyproject.toml (up to 5 levels)
        project_root = Path(__file__).resolve().parents[1]

        self.material = Material(E=72000, nu_xy=0.33, sig_yield=350)

        self.ct_info = CrackTipInfo(
            crack_tip_x=-15.5,
            crack_tip_y=0,
            crack_tip_angle=180,
            left_or_right='left'
        )

        # import and transform data
        nodemap_folder = str(project_root / 'test_data' / 'crack_detection' / 'Nodemaps')
        self.nodemap_file = Nodemap(name='Dummy2_WPXXX_DummyVersuch_2_dic_results_1_52.txt',
                                    folder=nodemap_folder)

        self.input_data = InputData(self.nodemap_file)
        self.input_data.calc_stresses(self.material)
        self.input_data.transform_data(self.ct_info.crack_tip_x, self.ct_info.crack_tip_y, self.ct_info.crack_tip_angle)

    def test_fracture_analysis_with_constant_tick_size(self):
        int_props = IntegralProperties(
            number_of_paths=2,
            integral_tick_size=0.5,

            integral_size_left=-5,
            integral_size_right=10,
            integral_size_top=8,
            integral_size_bottom=-8,

            top_offset=3,
            bottom_offset=-3,

            paths_distance_left=0.5,
            paths_distance_right=0.5,
            paths_distance_bottom=0.5,
            paths_distance_top=0.5,

            bueckner_williams_terms=[-1, 1, 2, 3]
        )

        # initialize fracture analysis
        analysis = FractureAnalysis(
            material=self.material,
            crack_tip_info=self.ct_info,
            nodemap=self.nodemap_file,
            data=self.input_data,
            integral_properties=int_props,
            optimization_properties=None
        )
        analysis.run()

        # test filtered outlier results
        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['j'], 1.8699, delta=1e-4)
        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['sif_j'], 11.6028, delta=1e-4)

        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['decomp_K_1'], 11.9760, delta=1e-4)

        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['sif_k_i'], 11.0096, delta=1e-4)
        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['sif_k_ii'], -0.9243, delta=1e-4)

        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['t_stress_int'], -42.1799, delta=1e-4)
        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['t_stress_sdm'], -62.8528, delta=1e-4)

        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['williams_int_a_n'][0], 107.9594, delta=1e-4)
        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['williams_int_a_n'][1], 129.1896, delta=1e-4)
        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['williams_int_a_n'][2], -20.1644, delta=1e-4)
        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['williams_int_a_n'][3], 5.7811, delta=1e-4)

        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['williams_int_b_n'][0], 19.5458, delta=1e-4)
        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['williams_int_b_n'][1], 8.2752, delta=1e-4)
        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['williams_int_b_n'][2], -1.2748, delta=1e-4)
        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['williams_int_b_n'][3], 0.5349, delta=1e-4)

        temp_dir = tempfile.mkdtemp()
        try:
            # test writer
            writer = OutputWriter(path=temp_dir, fracture_analysis=analysis)
            writer.write_header()
            writer.write_results()
            # test plotter
            plot_sets = PlotSettings(ylim_down=-20, ylim_up=20,
                                     xlim_down=-20, xlim_up=20,
                                     background='sig_vm')
            plotter = Plotter(path=temp_dir, fracture_analysis=analysis, plot_sets=plot_sets)
            plotter.plot()
        finally:
            shutil.rmtree(temp_dir)

    def test_fracture_analysis_with_constant_num_of_nodes(self):
        int_props = IntegralProperties(
            number_of_paths=2,
            number_of_nodes=100,

            integral_size_left=-5,
            integral_size_right=10,
            integral_size_top=8,
            integral_size_bottom=-8,

            top_offset=3,
            bottom_offset=-3,

            paths_distance_left=0.5,
            paths_distance_right=0.5,
            paths_distance_bottom=0.5,
            paths_distance_top=0.5,

            bueckner_williams_terms=[-1, 1, 2, 3]
        )

        # initialize fracture analysis
        analysis = FractureAnalysis(
            material=self.material,
            crack_tip_info=self.ct_info,
            nodemap=self.nodemap_file,
            data=self.input_data,
            integral_properties=int_props,
            optimization_properties=None
        )
        analysis.run()

        # test filtered outlier results
        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['j'], 1.8813, delta=1e-4)
        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['sif_j'], 11.6381, delta=1e-4)

        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['decomp_K_1'], 12.0724, delta=1e-4)

        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['sif_k_i'], 11.0188, delta=1e-4)
        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['sif_k_ii'], -0.9064, delta=1e-4)

        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['t_stress_int'], -42.5591, delta=1e-4)
        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['t_stress_sdm'], -62.8528, delta=1e-4)

        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['williams_int_a_n'][0], 109.4747, delta=1e-4)
        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['williams_int_a_n'][1], 129.0193, delta=1e-4)
        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['williams_int_a_n'][2], -20.1845, delta=1e-4)
        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['williams_int_a_n'][3], 5.7942, delta=1e-4)

        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['williams_int_b_n'][0], 19.8578, delta=1e-4)
        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['williams_int_b_n'][1], 8.3570, delta=1e-4)
        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['williams_int_b_n'][2], -1.2922, delta=1e-4)
        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['williams_int_b_n'][3], 0.5393, delta=1e-4)

        temp_dir = tempfile.mkdtemp()
        try:
            # test writer
            writer = OutputWriter(path=temp_dir, fracture_analysis=analysis)
            writer.write_header()
            writer.write_results()
            # test plotter
            plot_sets = PlotSettings(ylim_down=-20, ylim_up=20,
                                     xlim_down=-20, xlim_up=20,
                                     background='sig_vm')
            plotter = Plotter(path=temp_dir, fracture_analysis=analysis, plot_sets=plot_sets)
            plotter.plot()
        finally:
            shutil.rmtree(temp_dir)

    def test_fitting_methods_with_DIC_data(self):
        opt_props = OptimizationProperties(
            angle_gap=20,
            min_radius=5,
            max_radius=15,
            tick_size=0.01,
            terms=[-3, -2, -1, 0, 1, 2, 3],
        )
        analysis = FractureAnalysis(
            material=Material(),
            nodemap=self.nodemap_file,
            data=self.input_data,
            crack_tip_info=self.ct_info,
            integral_properties=None,
            optimization_properties=opt_props
        )
        analysis.run()

        # test CJP results
        self.assertAlmostEqual(analysis.cjp_res_mm['K_F'], 10.7934, delta=1e-4)
        self.assertAlmostEqual(analysis.cjp_res_mm['K_R'], 2.2790, delta=1e-4)
        self.assertAlmostEqual(analysis.cjp_res_mm['K_S'], -1.1568, delta=1e-4)
        self.assertAlmostEqual(analysis.cjp_res_mm['K_II'], -0.0275, delta=1e-4)
        self.assertAlmostEqual(analysis.cjp_res_mm['T'], -32.1685, delta=1e-4)

        self.assertAlmostEqual(analysis.cjp_res_m1['K_F'], 9.6752, delta=1e-4)
        self.assertAlmostEqual(analysis.cjp_res_m1['K_R'], -2.6313, delta=1e-4)
        self.assertAlmostEqual(analysis.cjp_res_m1['K_S'], 1.8096, delta=1e-4)
        self.assertAlmostEqual(analysis.cjp_res_m1['T_x'], -16.9536, delta=1e-4)
        self.assertAlmostEqual(analysis.cjp_res_m1['T_y'], -34.9358, delta=1e-4)

        # test Williams results
        self.assertAlmostEqual(analysis.williams_fit_res['K_I'], 11.3232, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_res['K_II'], -1.1102, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_res['K_III'], 2.0274, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_res['T'], -44.3513, delta=1e-4)

        self.assertAlmostEqual(analysis.williams_fit_a_n[-3], -195.3576, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_a_n[-2], 6.6082, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_a_n[-1], -36.3256, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_a_n[0], -16.7067, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_a_n[1], 142.8491, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_a_n[2], -11.0878, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_a_n[3], 2.0325, delta=1e-4)

        self.assertAlmostEqual(analysis.williams_fit_b_n[-3], -60.7676, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_b_n[-2], -27.7731, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_b_n[-1], 49.9209, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_b_n[0], -6.1888, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_b_n[1], 14.0059, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_b_n[2], -3.1521, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_b_n[3], 0.3745, delta=1e-4)

        self.assertAlmostEqual(analysis.williams_fit_c_n[-3], 1092.8856, delta=1e-3)
        self.assertAlmostEqual(analysis.williams_fit_c_n[-2], -150.6980, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_c_n[-1], 265.8402, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_c_n[0], -47.3960, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_c_n[1], 51.1547, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_c_n[2], 7.5005, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_c_n[3], 0.7213, delta=1e-4)

        mode_i_result = analysis.cjp_mode_i_odm_result
        mixed_mode_result = analysis.cjp_mixed_mode_odm_result
        williams_in_plane_result = analysis.williams_in_plane_odm_result
        williams_out_of_plane_result = analysis.williams_out_of_plane_odm_result
        self.assertEqual(mode_i_result.status, "completed")
        self.assertEqual(mixed_mode_result.status, "completed")
        self.assertEqual(williams_in_plane_result.status, "completed")
        self.assertEqual(williams_out_of_plane_result.status, "completed")
        np.testing.assert_array_equal(
            analysis.cjp_coeffs_m1,
            astuple(mode_i_result.coefficients),
        )
        np.testing.assert_array_equal(
            analysis.cjp_coeffs_mm,
            astuple(mixed_mode_result.coefficients),
        )
        np.testing.assert_array_equal(
            analysis.williams_coeffs,
            williams_in_plane_result.coefficients.a_n
            + williams_in_plane_result.coefficients.b_n
            + williams_out_of_plane_result.coefficients.c_n,
        )
        self.assertEqual(
            analysis.williams_fit_a_n,
            dict(
                zip(
                    williams_in_plane_result.coefficients.terms,
                    williams_in_plane_result.coefficients.a_n,
                )
            ),
        )
        self.assertEqual(
            analysis.williams_fit_b_n,
            dict(
                zip(
                    williams_in_plane_result.coefficients.terms,
                    williams_in_plane_result.coefficients.b_n,
                )
            ),
        )
        self.assertEqual(
            analysis.williams_fit_c_n,
            dict(
                zip(
                    williams_out_of_plane_result.coefficients.terms,
                    williams_out_of_plane_result.coefficients.c_n,
                )
            ),
        )
        self.assertEqual(analysis.cjp_res_m1["Error"], mode_i_result.cost)
        self.assertEqual(analysis.cjp_res_mm["Error"], mixed_mode_result.cost)
        self.assertEqual(
            analysis.williams_fit_res["Error_xy"], williams_in_plane_result.cost
        )
        self.assertEqual(
            analysis.williams_fit_res["Error_z"], williams_out_of_plane_result.cost
        )

        temp_dir = tempfile.mkdtemp()
        try:
            # test writer
            writer = OutputWriter(path=temp_dir, fracture_analysis=analysis)
            writer.write_header()
            writer.write_results()
            # test plotter
            plot_sets = PlotSettings(ylim_down=-20, ylim_up=20,
                                     xlim_down=-20, xlim_up=20,
                                     background='sig_vm')
            plotter = Plotter(path=temp_dir, fracture_analysis=analysis, plot_sets=plot_sets)
            plotter.plot()
        finally:
            shutil.rmtree(temp_dir)

    def test_line_integral_methods_3D_with_synthetic_williams_data(self):
        material = Material(E=72000, nu_xy=0.33, sig_yield=350)
        mode_i_sif = 10.0
        mode_ii_sif = 20.0
        mode_iii_sif = 30.0
        t_stress = 40.0
        input_data = _synthetic_williams_data(
            material,
            mode_i_sif=mode_i_sif,
            mode_ii_sif=mode_ii_sif,
            mode_iii_sif=mode_iii_sif,
            t_stress=t_stress,
        )

        crack_tip = CrackTipInfo(0, 0, 0, 'right')
        input_data.transform_data(
            crack_tip.crack_tip_x,
            crack_tip.crack_tip_y,
            crack_tip.crack_tip_angle,
        )
        integral_properties = IntegralProperties(
            number_of_paths=3,
            integral_tick_size=0.1,
            integral_size_left=-5,
            integral_size_right=5,
            integral_size_top=6,
            integral_size_bottom=-6,
            top_offset=0,
            bottom_offset=0,
            paths_distance_left=0.5,
            paths_distance_right=0.5,
            paths_distance_bottom=0.6,
            paths_distance_top=0.6,
            bueckner_williams_terms=[1, 2],
        )
        analysis = FractureAnalysis(
            material=material,
            nodemap='williams_synthetic.txt',
            data=input_data,
            crack_tip_info=crack_tip,
            integral_properties=integral_properties,
            optimization_properties=None,
        )

        analysis.run()

        expected_in_plane_j = (
            mode_i_sif ** 2 + mode_ii_sif ** 2
        ) * 1000 / material.E
        expected_mode_i_j = mode_i_sif ** 2 * 1000 / material.E
        expected_mode_ii_j = mode_ii_sif ** 2 * 1000 / material.E
        expected_mode_iii_j = (
            mode_iii_sif ** 2
            * (1 + material.nu_xy)
            * 1000
            / material.E
        )
        mode_ii_j_tolerance = 0.025 * expected_mode_ii_j
        # K is proportional to sqrt(J), so the corresponding first-order SIF
        # tolerance is half the 2.5% clean-field Mode II J acceptance limit.
        mode_ii_sif_tolerance = 0.0125 * mode_ii_sif
        expected_energy_equivalent_sif = np.hypot(mode_i_sif, mode_ii_sif)
        results = analysis.sifs_int['rej_out_mean']
        self.assertAlmostEqual(results['j'], expected_in_plane_j, delta=0.01)
        self.assertAlmostEqual(
            results['sif_j'], expected_energy_equivalent_sif, delta=0.01
        )
        self.assertAlmostEqual(results['sif_k_i'], mode_i_sif, delta=0.01)
        self.assertAlmostEqual(results['sif_k_ii'], mode_ii_sif, delta=0.01)
        self.assertAlmostEqual(results['k_i_chen'], mode_i_sif, delta=0.01)
        self.assertAlmostEqual(results['k_ii_chen'], mode_ii_sif, delta=0.01)
        self.assertAlmostEqual(results['decomp_j_1'], expected_mode_i_j, delta=0.01)
        self.assertAlmostEqual(
            results['decomp_j_2'], expected_mode_ii_j, delta=mode_ii_j_tolerance
        )
        self.assertAlmostEqual(results['decomp_j_3'], expected_mode_iii_j, delta=0.01)
        self.assertAlmostEqual(results['decomp_K_1'], mode_i_sif, delta=0.05)
        self.assertAlmostEqual(
            results['decomp_K_2'], mode_ii_sif, delta=mode_ii_sif_tolerance
        )
        self.assertAlmostEqual(results['decomp_K_3'], mode_iii_sif, delta=0.02)
        # Zhao interaction-integral T-Stress is deliberately excluded pending
        # separate analytical validation of its contour-dependent result.
        self.assertAlmostEqual(results['t_stress_chen'], t_stress, delta=0.05)
        self.assertAlmostEqual(results['t_stress_sdm'], t_stress, delta=0.05)

    def test_fitting_methods_3D_with_synthetic_williams_data(self):
        material = Material(E=72000, nu_xy=0.33, sig_yield=350)
        input_data = _synthetic_williams_data(material)

        ###############
        # Main script #
        ###############

        # setup properties for fracture analysis

        opt_props = OptimizationProperties(
            angle_gap=10,
            min_radius=5,
            max_radius=10,
            tick_size=0.01,
            terms=[-3, -2, -1, 0, 1, 2, 3, 4, 5],
        )

        ct = CrackTipInfo(0, 0, 0, 'right')

        input_data.transform_data(ct.crack_tip_x, ct.crack_tip_y, ct.crack_tip_angle)

        analysis = FractureAnalysis(
            material=material,
            nodemap='williams_synthetic.txt',
            data=input_data,
            crack_tip_info=ct,
            integral_properties=None,
            optimization_properties=opt_props
        )
        analysis.run()

        # test Williams results
        self.assertAlmostEqual(analysis.williams_fit_res['K_I'], 10., delta=1e-3)
        self.assertAlmostEqual(analysis.williams_fit_res['K_II'], 20., delta=1e-3)
        self.assertAlmostEqual(analysis.williams_fit_res['K_III'], 30., delta=1e-3)
        self.assertAlmostEqual(analysis.williams_fit_res['T'], 40, delta=1e-3)

        self.assertAlmostEqual(analysis.williams_fit_a_n[-3], -0.0297, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_a_n[-2], 0.0188, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_a_n[-1], -0.0073, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_a_n[0], -26.4068, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_a_n[1], 126.1559, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_a_n[2], 9.9999, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_a_n[3], 0., delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_a_n[4], 0., delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_a_n[5], 0., delta=1e-4)

        self.assertAlmostEqual(analysis.williams_fit_b_n[-3], -0.0151, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_b_n[-2], 0.0299, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_b_n[-1], -0.0198, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_b_n[0], -6.8402, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_b_n[1], -252.3154, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_b_n[2], 0.0001, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_b_n[3], 0.0001, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_b_n[4], 0, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_b_n[5], 0, delta=1e-4)

        self.assertAlmostEqual(analysis.williams_fit_c_n[-3], -0.0006, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_c_n[-2], -0., delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_c_n[-1], -0.0003, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_c_n[0], -54.5209, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_c_n[1], 756.9397, delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_c_n[2], -0., delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_c_n[3], 0., delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_c_n[4], -0., delta=1e-4)
        self.assertAlmostEqual(analysis.williams_fit_c_n[5], -0., delta=1e-4)

        williams_in_plane_result = analysis.williams_in_plane_odm_result
        williams_out_of_plane_result = analysis.williams_out_of_plane_odm_result
        self.assertEqual(williams_in_plane_result.status, "completed")
        self.assertEqual(williams_out_of_plane_result.status, "completed")
        np.testing.assert_array_equal(
            analysis.williams_coeffs,
            williams_in_plane_result.coefficients.a_n
            + williams_in_plane_result.coefficients.b_n
            + williams_out_of_plane_result.coefficients.c_n,
        )
        self.assertEqual(
            analysis.williams_fit_a_n,
            dict(
                zip(
                    williams_in_plane_result.coefficients.terms,
                    williams_in_plane_result.coefficients.a_n,
                )
            ),
        )
        self.assertEqual(
            analysis.williams_fit_b_n,
            dict(
                zip(
                    williams_in_plane_result.coefficients.terms,
                    williams_in_plane_result.coefficients.b_n,
                )
            ),
        )
        self.assertEqual(
            analysis.williams_fit_c_n,
            dict(
                zip(
                    williams_out_of_plane_result.coefficients.terms,
                    williams_out_of_plane_result.coefficients.c_n,
                )
            ),
        )
        self.assertEqual(
            analysis.williams_fit_res["K_I"],
            williams_in_plane_result.quantities.k_i,
        )
        self.assertEqual(
            analysis.williams_fit_res["K_II"],
            williams_in_plane_result.quantities.k_ii,
        )
        self.assertEqual(
            analysis.williams_fit_res["K_III"],
            williams_out_of_plane_result.quantities.k_iii,
        )
        self.assertEqual(
            analysis.williams_fit_res["T"],
            williams_in_plane_result.quantities.t_stress,
        )

        temp_dir = tempfile.mkdtemp()
        try:
            # test writer
            writer = OutputWriter(path=temp_dir, fracture_analysis=analysis)
            writer.write_header()
            writer.write_results()
            # test plotter
            plot_sets = PlotSettings(ylim_down=-20, ylim_up=20,
                                     xlim_down=-20, xlim_up=20,
                                     background='sig_vm')
            plotter = Plotter(path=temp_dir, fracture_analysis=analysis, plot_sets=plot_sets)
            plotter.plot()
        finally:
            shutil.rmtree(temp_dir)

class TestFractureAnalysisPipeline(unittest.TestCase):
    def setUp(self):
        # Find project root iteratively by searching for pyproject.toml (up to 5 levels)
        project_root = Path(__file__).resolve()
        for _ in range(5):
            if (project_root / 'pyproject.toml').exists():
                break
            project_root = project_root.parent

        self.origin = project_root / 'test_data' / 'crack_detection'
        self.nodemap_path = str(self.origin / 'Nodemaps')
        self.input_file = str(self.origin / 'crack_info_by_nodemap_fracture_analysis.txt')
        self.output_path = str(project_root / 'test_data' / 'fracture_analysis')
        self.material = Material(E=72000, nu_xy=0.33, sig_yield=350)
        self.plot_sets = PlotSettings(xlim_down=-20, xlim_up=20, ylim_down=-20, ylim_up=20,
                                      background='eps_vm',
                                      min_value=0, max_value=0.0068, extend='max')

    def test_find_integral_props_and_run_pipeline(self):
        temp_dir = tempfile.mkdtemp()
        try:
            int_props = IntegralProperties(
                number_of_paths=5,
                number_of_nodes=100,

                mask_tolerance=2,

                bueckner_williams_terms=[-1, 1, 2, 3, 4, 5]
            )

            opt_props = OptimizationProperties(
                angle_gap=20,
                min_radius=5,
                max_radius=10,
                tick_size=0.01,
                terms=[-3, -2, -1, 0, 1, 2, 3, 4, 5],
            )

            pipeline = FractureAnalysisPipeline(
                material=self.material,
                nodemap_path=self.nodemap_path,
                input_file=self.input_file,
                output_path=temp_dir,
                optimization_properties=opt_props,
                integral_properties=int_props,
                plot_sets=self.plot_sets
            )
            pipeline.find_max_force_stages(max_force=15000)
            pipeline.find_integral_props()

            # check integral properties
            self.assertEqual(pipeline.integral_props[0].bottom_offset, -0.5609400793826333)
            self.assertEqual(pipeline.integral_props[0].integral_size_bottom, -2.0561224319402838)
            self.assertAlmostEqual(pipeline.integral_props[0].integral_tick_size, 0.1916179957579448)
            self.assertEqual(pipeline.integral_props[0].paths_distance_bottom, 0.7664719830317792)

            pipeline.run()

            # Read results and write into CSV file
            reader = OutputReader()
            output_path = Path(temp_dir) / 'txt-files'

            files = [f.name for f in output_path.iterdir() if f.is_file()]
            list_of_tags = ["CJP_results", "Williams_fit_results", "SIFs_integral", "Bueckner_Chen_integral",
                            "Path_SIFs", "Path_Williams_a_n", "Path_Williams_b_n"]
            for file in files:
                if file.endswith(".txt"):
                    for tag in list_of_tags:
                        reader.read_tag_data(path=str(output_path), filename=file, tag=tag)

            # Make CSV file
            reader.make_csv_from_results(files="all", output_path=temp_dir, output_filename='results.csv')

            # Assert
            act_results = pd.read_csv(Path(temp_dir) / 'results.csv')
            expected_path = Path(self.output_path) / 'results_auto_integral_probs.csv'
            # Regenerate this committed fixture only when explicitly requested.
            # Run: CRACKPY_UPDATE_PIPELINE_FIXTURES=1 python -m pytest \
            #     test_scripts/test_fracture_analysis.py::TestFractureAnalysisPipeline::test_find_integral_props_and_run_pipeline -q
            # Normal test runs remain read-only and compare the generated results with the existing fixture.
            # Review every numerical change with git diff before committing an updated fixture.
            if os.getenv('CRACKPY_UPDATE_PIPELINE_FIXTURES') == '1':
                act_results.to_csv(expected_path, index=False)
            exp_results = pd.read_csv(expected_path)
            errors = []
            for column in exp_results.columns:
                try:
                    pd.testing.assert_series_equal(exp_results[column], act_results[column], atol=1e-4)
                except AssertionError as error:
                    errors.append(f'{column}:\n{error}')

            if errors:
                raise AssertionError('\n\n'.join(errors))

        finally:
            shutil.rmtree(temp_dir)

    def test_predefine_integral_probs_and_run_pipeline(self):
        temp_dir = tempfile.mkdtemp()
        try:
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

                bueckner_williams_terms=[-1, 1, 2, 3, 4, 5]
            )

            opt_props = OptimizationProperties(
                angle_gap=20,
                min_radius=5,
                max_radius=10,
                tick_size=0.01,
                terms=[-3, -2, -1, 0, 1, 2, 3, 4, 5],
            )

            pipeline = FractureAnalysisPipeline(
                material=self.material,
                nodemap_path=self.nodemap_path,
                input_file=self.input_file,
                output_path=temp_dir,
                optimization_properties=opt_props,
                integral_properties=int_props,
                plot_sets=self.plot_sets
            )
            pipeline.find_max_force_stages(max_force=15000)

            pipeline.run()

            # Read results and write into CSV file
            reader = OutputReader()
            output_path = Path(temp_dir) / 'txt-files'

            files = [f.name for f in output_path.iterdir() if f.is_file()]
            list_of_tags = ["CJP_results", "Williams_fit_results", "SIFs_integral", "Bueckner_Chen_integral",
                            "Path_SIFs", "Path_Williams_a_n", "Path_Williams_b_n"]
            for file in files:
                if file.endswith(".txt"):
                    for tag in list_of_tags:
                        reader.read_tag_data(path=str(output_path), filename=file, tag=tag)

            # Make CSV file
            reader.make_csv_from_results(files="all", output_path=temp_dir, output_filename='results.csv')

            # Assert
            act_results = pd.read_csv(Path(temp_dir) / 'results.csv')
            expected_path = Path(self.output_path) / 'results_predef_integral_probs.csv'
            # Regenerate this committed fixture only when explicitly requested.
            # Run: CRACKPY_UPDATE_PIPELINE_FIXTURES=1 python -m pytest \
            #     test_scripts/test_fracture_analysis.py::TestFractureAnalysisPipeline::test_predefine_integral_probs_and_run_pipeline -q
            # Normal test runs remain read-only and compare the generated results with the existing fixture.
            # Review every numerical change with git diff before committing an updated fixture.
            if os.getenv('CRACKPY_UPDATE_PIPELINE_FIXTURES') == '1':
                act_results.to_csv(expected_path, index=False)
            exp_results = pd.read_csv(expected_path)
            errors = []
            for column in exp_results.columns:
                try:
                    pd.testing.assert_series_equal(exp_results[column], act_results[column], atol=1e-4)
                except AssertionError as error:
                    errors.append(f'{column}:\n{error}')

            if errors:
                raise AssertionError('\n\n'.join(errors))


        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
