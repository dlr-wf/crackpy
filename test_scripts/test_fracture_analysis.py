import os
import shutil
import tempfile
import unittest

import pandas as pd

from crackpy.fracture_analysis.analysis import FractureAnalysis
from crackpy.fracture_analysis.data_processing import CrackTipInfo, InputData
from crackpy.fracture_analysis.line_integration import IntegralProperties
from crackpy.fracture_analysis.optimization import OptimizationProperties
from crackpy.fracture_analysis.pipeline import FractureAnalysisPipeline
from crackpy.fracture_analysis.plot import PlotSettings, Plotter
from crackpy.fracture_analysis.read import OutputReader
from crackpy.fracture_analysis.write import OutputWriter
from crackpy.structure_elements.data_files import Nodemap
from crackpy.structure_elements.material import Material


class TestFractureAnalysis(unittest.TestCase):
    def setUp(self):
        self.material = Material(E=72000, nu_xy=0.33, sig_yield=350)

        self.ct_info = CrackTipInfo(
            crack_tip_x=-15.5,
            crack_tip_y=0,
            crack_tip_angle=180,
            left_or_right='left'
        )

        # import and transform data
        self.nodemap_file = Nodemap(name='Dummy2_WPXXX_DummyVersuch_2_dic_results_1_52.txt',
                                    folder=os.path.join(  # '..',
                                                        'test_data', 'crack_detection', 'Nodemaps'))

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
            integral_properties=int_props
        )
        analysis.run()

        # test filtered outlier results
        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['j'], 1.8699, delta=1e-4)
        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['sif_j'], 11.6028, delta=1e-4)

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
            integral_properties=int_props
        )
        analysis.run()

        # test filtered outlier results
        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['j'], 1.8813, delta=1e-4)
        self.assertAlmostEqual(analysis.sifs_int['rej_out_mean']['sif_j'], 11.6381, delta=1e-4)

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

    def test_fitting_methods(self):
        opt_props = OptimizationProperties(
            angle_gap=20,
            min_radius=5,
            max_radius=15,
            tick_size=0.01,
            terms=[-3, -2, -1, 0, 1, 2, 3]
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
        self.assertAlmostEqual(analysis.res_cjp['K_F'], 10.7934, delta=1e-4)
        self.assertAlmostEqual(analysis.res_cjp['K_R'], 2.2790, delta=1e-4)
        self.assertAlmostEqual(analysis.res_cjp['K_S'], -1.1568, delta=1e-4)
        self.assertAlmostEqual(analysis.res_cjp['K_II'], -0.0275, delta=1e-4)
        self.assertAlmostEqual(analysis.res_cjp['T'], -32.1685, delta=1e-4)

        # test Williams results
        self.assertAlmostEqual(analysis.sifs_fit['K_I'], 11.3232, delta=1e-4)
        self.assertAlmostEqual(analysis.sifs_fit['K_II'], -1.1102, delta=1e-4)
        self.assertAlmostEqual(analysis.sifs_fit['T'], -44.3513, delta=1e-4)

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


class TestFractureAnalysisPipeline(unittest.TestCase):
    def setUp(self):
        self.origin = os.path.join(  # '..',
                                   'test_data', 'crack_detection')
        self.nodemap_path = os.path.join(self.origin, 'Nodemaps')
        self.input_file = os.path.join(self.origin, 'crack_info_by_nodemap.txt')
        self.output_path = os.path.join(  # '..',
                                        'test_data', 'fracture_analysis')
        self.material = Material(E=72000, nu_xy=0.33, sig_yield=350)
        self.plot_sets = PlotSettings(xlim_down=-20, xlim_up=20, ylim_down=-20, ylim_up=20,
                                      background='eps_vm',
                                      min_value=0, max_value=0.0068, extend='max')

    def test_find_integral_props_and_run_pipeline(self):
        temp_dir = tempfile.mkdtemp()
        try:
            int_props = IntegralProperties(
                number_of_paths=10,
                number_of_nodes=100,

                integral_size_left=-5,
                integral_size_right=5,
                integral_size_top=5,
                integral_size_bottom=-5,

                paths_distance_top=0.5,
                paths_distance_left=0.5,
                paths_distance_right=0.5,
                paths_distance_bottom=0.5,

                mask_tolerance=2,

                buckner_williams_terms=[-1, 1, 2, 3, 4, 5]
            )
            opt_props = OptimizationProperties(
                angle_gap=20,
                min_radius=5,
                max_radius=10,
                tick_size=0.01,
                terms=[-1, 0, 1, 2, 3, 4, 5]
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
            self.assertEqual(pipeline.integral_props[0].bottom_offset, -0.2163461538461533)
            self.assertEqual(pipeline.integral_props[0].integral_size_bottom, -5)
            self.assertAlmostEqual(pipeline.integral_props[0].integral_tick_size, 0.1916179957579483)
            self.assertEqual(pipeline.integral_props[0].paths_distance_bottom, 0.5)

            pipeline.run()

            # Read results and write into CSV file
            reader = OutputReader()
            output_path = os.path.join(temp_dir, 'txt-files')

            files = os.listdir(output_path)
            list_of_tags = ["CJP_results", "Williams_fit_results", "SIFs_integral", "Bueckner_Chen_integral",
                            "Path_SIFs", "Path_Williams_a_n", "Path_Williams_b_n"]
            for file in files:
                if file.endswith(".txt"):
                    for tag in list_of_tags:
                        reader.read_tag_data(path=output_path, filename=file, tag=tag)

            # Make CSV file
            reader.make_csv_from_results(files="all", output_path=temp_dir, output_filename='results.csv')

            # Assert
            exp_results = pd.read_csv(os.path.join(self.output_path, 'results.csv'))
            act_results = pd.read_csv(os.path.join(temp_dir, 'results.csv'))
            pd.testing.assert_frame_equal(exp_results, act_results)

        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
