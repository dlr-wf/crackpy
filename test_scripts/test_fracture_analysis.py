import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

from crackpy.fracture_analysis.analysis import FractureAnalysis
from crackpy.fracture_analysis.crack_tip import williams_displ_field_z, williams_displ_field_xy
from crackpy.input.input_data import InputData
from crackpy.input.crack_tip_info import CrackTipInfo
from crackpy.fracture_analysis.line_integration import IntegralProperties
from crackpy.fracture_analysis.optimization import OptimizationProperties
from crackpy.fracture_analysis.pipeline import FractureAnalysisPipeline
from crackpy.results.plot import PlotSettings, Plotter
from crackpy.results.read import OutputReader
from crackpy.results.write import OutputWriter
from crackpy.structure_elements.data_files import Nodemap
from crackpy.structure_elements.material import Material


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

            buckner_williams_terms=[-1, 1, 2, 3]
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

        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['decomp_K_1'], 10.2195, delta=1e-4)

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

            buckner_williams_terms=[-1, 1, 2, 3]
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

        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['decomp_K_1'], 10.2961, delta=1e-4)

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

    def test_fitting_methods_3D_with_synthetic_data(self):

        # prepare synthetic data
        material = Material(E=72000, nu_xy=0.33, sig_yield=350)

        steps = 500
        x_coordinates = np.linspace(-25, 25, steps, endpoint=True)
        y_coordinates = np.linspace(-25, 25, steps, endpoint=True)
        x_mesh, y_mesh = np.meshgrid(x_coordinates, y_coordinates)

        K_I = 10 * np.sqrt(1000)  # MPa * sqrt(m)
        K_II = 20 * np.sqrt(1000)  # MPa * sqrt(m)
        K_III = 30 * np.sqrt(1000)  # MPa * sqrt(m)
        T = 40  # MPa
        A_1 = K_I / np.sqrt(2 * np.pi)
        A_2 = T / 4.0
        B_1 = - K_II / np.sqrt(2 * np.pi)
        C_1 = K_III / np.sqrt(0.5 * np.pi)
        A = [A_1, A_2]
        B = [B_1, 0]
        C = [C_1, 0]

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
        eps_xx[int(steps / 2) + gap:, 0:int(steps / 2)] = np.gradient(
            disp_u_mesh[int(steps / 2) + gap:, 0:int(steps / 2)],
            dist, axis=1)
        eps_xx[:, int(steps / 2):] = np.gradient(disp_u_mesh[:, int(steps / 2):], dist, axis=1)

        eps_yy = np.zeros_like(x_mesh)
        eps_yy[0:int(steps / 2) - gap, 0:int(steps / 2)] = np.gradient(
            disp_v_mesh[0:int(steps / 2) - gap, 0:int(steps / 2)],
            dist, axis=0)
        eps_yy[int(steps / 2) + gap:, 0:int(steps / 2)] = np.gradient(
            disp_v_mesh[int(steps / 2) + gap:, 0:int(steps / 2)],
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
            material=Material(),
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

                buckner_williams_terms=[-1, 1, 2, 3, 4, 5]
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
            exp_results = pd.read_csv(Path(self.output_path) / 'results_auto_integral_probs.csv')
            act_results = pd.read_csv(Path(temp_dir) / 'results.csv')
            pd.testing.assert_frame_equal(exp_results, act_results, atol=1e-4)

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

                buckner_williams_terms=[-1, 1, 2, 3, 4, 5]
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
            exp_results = pd.read_csv(Path(self.output_path) / 'results_predef_integral_probs.csv')
            act_results = pd.read_csv(Path(temp_dir) / 'results.csv')
            pd.testing.assert_frame_equal(exp_results, act_results, atol=1e-4)

        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
