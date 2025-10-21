import warnings
from typing import Union, Optional, Mapping

import numpy as np
import rich.progress as progress_rich

from crackpy.fracture_analysis import line_integration
from crackpy.fracture_analysis.data_processing import InputData, CrackTipInfo
from crackpy.fracture_analysis.line_integration import (IntegralProperties,
                                                        LineIntegral)
from crackpy.fracture_analysis.optimization import Optimization, OptimizationProperties
from crackpy.structure_elements.data_files import Nodemap
from crackpy.structure_elements.material import Material


class FractureAnalysis:
    """Fracture analysis of a single DIC nodemap.

    The class is able to calculate

    - J-integral
    - K_I and K_II with the interaction integral
    - T-stress with the interaction integral
    - higher-order terms (HOSTs and HORTs) w/ fitting method (ODM)
    - (BETA) J-integral decomposition (J_I, J_II, J_III, K_I, K_II, K_III)
    - (BETA) CJP model (Mode I) K_F, K_R, K_S, T_x and T_y w/ fitting method (ODM)
    - (BETA) CJP model (Mixed Mode) K_F, K_R, K_S, K_II and T w/ fitting method (ODM)
    - (BETA) T-stress with the Bueckner-Chen integral
    - (BETA) higher-order terms (HOSTs and HORTs) w/ Bueckner-integral


    Methods:
        * run - run fracture analysis with the provided data

    """

    def __init__(
            self,
            material: Material,
            nodemap: Union[Nodemap, str],
            data: InputData,
            crack_tip_info: CrackTipInfo,
            integral_properties: Optional[IntegralProperties] = IntegralProperties(),
            optimization_properties: Optional[OptimizationProperties] = OptimizationProperties()
    ):
        """Initialize FractureAnalysis class arguments.

        Args:
            material: obj of class Material, material parameters and laws
            nodemap: obj of class Nodemap or filename of file with exported Aramis-DIC data
            data: obj of class InputData, imported data from nodemap_file
            crack_tip_info: obj of class CrackTipInfo, crack tip information (i.e. x,y coordinates, angle, etc.)
            integral_properties: IntegralProperties or None,
                                 wrapper for specification of line integral properties
                                 If None, Line Integral Methods are not calculated.
            optimization_properties: OptimizationProperties or None,
                                     If None, optimization / fitting is not performed.
        """
        self.material = material
        self.nodemap_file = nodemap.name if isinstance(nodemap, Nodemap) else nodemap
        self.data = data
        self.crack_tip = crack_tip_info

        # Initialization of optimization and integral properties
        self.optimization_properties = optimization_properties
        if self.optimization_properties is not None:
            Optimization.ensure_defaults_williams(self.optimization_properties, self.crack_tip.crack_tip_x)
            self.optimization = Optimization(data=self.data,
                                             options=self.optimization_properties,
                                             material=self.material)
            self._init_optimizaton_results()

        self.integral_properties = integral_properties
        if self.integral_properties is not None:
            LineIntegral.ensure_defaults_buckner_chen(self.integral_properties)
            self._init_integral_results()

    def _init_optimizaton_results(self):
        """Initialize attributes used for storing optimization results."""
        self.cjp_coeffs_mm = None
        self.cjp_res_mm = None
        self.cjp_coeffs_m1 = None
        self.cjp_res_m1 = None
        self.williams_coeffs = None
        self.williams_fit_a_n = None
        self.williams_fit_b_n = None
        self.williams_fit_c_n = None
        self.williams_fit_res = None

    def _init_integral_results(self):
        """Initialize attributes used for storing integral evaluation results."""
        self.path_results = []
        self.williams_int_a_n = []
        self.williams_int_b_n = []
        self.williams_int = []
        self.sifs_int = None
        self.path_sizes = []
        self.integration_points = []
        self.tick_sizes = []
        self.num_of_path_nodes = []

    def run(self, progress_bar: Optional[Mapping[str, object]] = None, task_id=None):
        """Run fracture analysis with the provided data, crack_tip_info, and integral_properties.
        Results are stored as class instance attributes 'results', 'sifs', 'path_sizes', and 'path_nodes'.

        Args:
            progress_bar: whether to show progress bar for line integral calculation
            task_id: task id for progress bar (handed-over automatically during pipeline, not needed for single run)

        """

        # Set the optimization and line integral methods that should be run
        if self.optimization_properties is not None:
            print('Running Fitting Methods...')
            self._run_cjp_optimization_modeI()
            self._run_cjp_optimization_mixedmode()
            self._run_williams_optimization()
        else:
            print('No optimization properties provided, skipping optimizations.')

        if self.integral_properties is not None:
            print('Running Line Integral Methods...')
            self._run_line_integrals(progress_bar, task_id)
        else:
            print('No integral properties provided, skipping line integrals.')

    pass

    def _run_cjp_optimization_modeI(self) -> None:
        """Run CJP optimization if optimization properties are provided."""

        try:
            cjp_results_m1 = self.optimization.optimize_cjp_displacements_modeI()
            self.cjp_coeffs_m1 = cjp_results_m1.x
            A, B, C, E, F = self.cjp_coeffs_m1

            # from Camacho-Reyes et al. (2023) "A new crack tip plastic zone model for mixed mode I and mode II" formulas 4-8
            K_F = np.sqrt(np.pi / 2) * (A - 3 * B - 8 * E)
            K_R = -((2 * np.pi) ** (3 / 2)) * E
            K_S = np.sqrt(np.pi / 2) * (A + B)
            T_x = -C
            T_y = -F

            # MPa*sqrt(mm) to MPa*sqrt(m)
            K_F /= np.sqrt(1000)
            K_R /= np.sqrt(1000)
            K_S /= np.sqrt(1000)

            self.cjp_res_m1 = {'Error': cjp_results_m1.cost, 'K_F': K_F, 'K_R': K_R, 'K_S': K_S, 'T_x': T_x, 'T_y': T_y}
        except Exception as e:
            print('CJP optimization (Mode I) failed.')
            print(e)
            self.cjp_res_m1 = {'Error': np.nan, 'K_F': np.nan, 'K_R': np.nan, 'K_S': np.nan, 'T_x': np.nan, 'T_y': np.nan}

        pass

    def _run_cjp_optimization_mixedmode(self) -> None:
        """Run CJP optimization if optimization properties are provided."""

        try:
            # calculate CJP coefficients with fitting method
            cjp_results = self.optimization.optimize_cjp_displacements_mixedmode()

            self.cjp_coeffs_mm = cjp_results.x
            A_r, B_r, B_i, C, E = self.cjp_coeffs_mm

            # from Christopher et al. (2013) "Extension of the CJP model to mixed mode I and mode II" formulas 4-8
            K_F = np.sqrt(np.pi / 2) * (A_r - 3 * B_r - 8 * E)
            K_R = -4 * np.sqrt(np.pi / 2) * (2 * B_i + E * np.pi)
            K_S = -np.sqrt(np.pi / 2) * (A_r + B_r)
            K_II = 2 * np.sqrt(2 * np.pi) * B_i
            T = -C
            # MPa*sqrt(mm) to MPa*sqrt(m)
            K_F /= np.sqrt(1000)
            K_R /= np.sqrt(1000)
            K_S /= np.sqrt(1000)
            K_II /= np.sqrt(1000)

            self.cjp_res_mm = {'Error': cjp_results.cost, 'K_F': K_F, 'K_R': K_R, 'K_S': K_S, 'K_II': K_II, 'T': T}

        except Exception:
            print('CJP optimization failed.')
            self.cjp_res_mm = {'Error': np.nan, 'K_F': np.nan, 'K_R': np.nan, 'K_S': np.nan, 'K_II': np.nan,
                            'T': np.nan}

        pass

    def _run_williams_optimization(self) -> None:
        """Run Williams optimization if optimization properties are provided."""

        try:
            # calculate Williams coefficients with fitting method
            if self.optimization_properties.dimensions == 2:
                williams_results = self.optimization.optimize_williams_displacements()
                self.williams_coeffs = williams_results.x
                a_n = self.williams_coeffs[:len(self.optimization.terms)]
                b_n = self.williams_coeffs[len(self.optimization.terms):]
                self.williams_fit_a_n = {n: a_n[index] for index, n in enumerate(self.optimization.terms)}
                self.williams_fit_b_n = {n: b_n[index] for index, n in enumerate(self.optimization.terms)}
                self.williams_fit_c_n = {n: np.nan for n in self.optimization.terms}

                # derive stress intensity factors and T-stress [Kuna formula 3.45]
                K_I = np.sqrt(2 * np.pi) * self.williams_fit_a_n[1] / np.sqrt(1000)
                K_II = -np.sqrt(2 * np.pi) * self.williams_fit_b_n[1] / np.sqrt(1000)
                K_III = np.nan
                T = 4 * self.williams_fit_a_n[2]
            else:
                williams_results = self.optimization.optimize_williams_displacements_3d()
                self.williams_coeffs = williams_results.x
                a_n = self.williams_coeffs[:len(self.optimization.terms)]
                b_n = self.williams_coeffs[len(self.optimization.terms):2 * len(self.optimization.terms)]
                c_n = self.williams_coeffs[2 * len(self.optimization.terms):]
                self.williams_fit_a_n = {n: a_n[index] for index, n in enumerate(self.optimization.terms)}
                self.williams_fit_b_n = {n: b_n[index] for index, n in enumerate(self.optimization.terms)}
                self.williams_fit_c_n = {n: c_n[index] for index, n in enumerate(self.optimization.terms)}

                # derive stress intensity factors and T-stress [Kuna formula 3.45]
                K_I = np.sqrt(2 * np.pi) * self.williams_fit_a_n[1] / np.sqrt(1000)
                K_II = -np.sqrt(2 * np.pi) * self.williams_fit_b_n[1] / np.sqrt(1000)
                K_III = np.sqrt(0.5 * np.pi) * self.williams_fit_c_n[1] / np.sqrt(1000)
                T = 4 * self.williams_fit_a_n[2]

            self.williams_fit_res = {'Error': williams_results.cost, 'K_I': K_I, 'K_II': K_II, 'K_III': K_III, 'T': T}

        except Exception:
            print('Williams optimization failed.')
            self.williams_fit_a_n = {n: np.nan for index, n in enumerate(self.optimization.terms)}
            self.williams_fit_b_n = {n: np.nan for index, n in enumerate(self.optimization.terms)}
            self.williams_fit_c_n = {n: np.nan for index, n in enumerate(self.optimization.terms)}
            self.williams_fit_res = {'Error': np.nan, 'K_I': np.nan, 'K_II': np.nan, 'T': np.nan}

        pass

    def _run_line_integrals(self, progress_bar: Optional[Mapping[str, object]] = None, task_id=None) -> None:
        """Run line integrals if integral properties are provided."""

        # calculate Williams coefficients with Bueckner-Chen integral method
        current_size_left = self.integral_properties.integral_size_left
        current_size_right = self.integral_properties.integral_size_right
        current_size_top = self.integral_properties.integral_size_top
        current_size_bottom = self.integral_properties.integral_size_bottom

        if progress_bar is None:
            iterator = progress_rich.track(range(self.integral_properties.number_of_paths),
                                           description='Calculating integrals')
        else:
            iterator = range(self.integral_properties.number_of_paths)

        for n in iterator:
            # Define path properties
            path_properties = line_integration.PathProperties(current_size_left,
                                                              current_size_right,
                                                              current_size_bottom,
                                                              current_size_top,
                                                              self.integral_properties.integral_tick_size,
                                                              self.integral_properties.number_of_nodes,
                                                              self.integral_properties.top_offset,
                                                              self.integral_properties.bottom_offset)

            # Define integration path
            integration_path = line_integration.IntegrationPath(0, 0, path_properties=path_properties)

            # Define line integration methods
            line_integral = line_integration.LineIntegral(integration_path, self.data, self.material,
                                                                    self.integral_properties.mask_tolerance,
                                                                    self.integral_properties.buckner_williams_terms)

            # Calculate integral results
            line_integral.integrate_all()

            # Store path results
            self.path_results.append([line_integral.j_integral,
                                      line_integral.sif_k_j,
                                      line_integral.sif_k_i,
                                      line_integral.sif_k_ii,
                                      line_integral.t_stress_chen,
                                      line_integral.t_stress_sdm,
                                      line_integral.t_stress_int,
                                      line_integral.decomp_j_integral_I,
                                      line_integral.decomp_j_integral_II,
                                      line_integral.decomp_j_integral_III,
                                      line_integral.decomp_j_integral_K_I,
                                      line_integral.decomp_j_integral_K_II,
                                      line_integral.decomp_j_integral_K_III])
            self.williams_int_a_n.append(line_integral.williams_a_n)
            self.williams_int_b_n.append(line_integral.williams_b_n)
            self.williams_int.append(line_integral.williams_coefficients)
            self.path_sizes.append([current_size_left, current_size_right, current_size_bottom, current_size_top])
            self.integration_points.append([list(line_integral.np_integration_points[:, 0]),
                                            list(line_integral.np_integration_points[:, 1])])
            self.num_of_path_nodes.append(line_integral.integration_path.path_properties.number_of_nodes)
            self.tick_sizes.append(line_integral.integration_path.path_properties.tick_size)

            # Update path
            current_size_left -= self.integral_properties.paths_distance_left
            current_size_right += self.integral_properties.paths_distance_right
            current_size_bottom -= self.integral_properties.paths_distance_bottom
            current_size_top += self.integral_properties.paths_distance_top

            # Update progress bar
            if progress_bar:
                progress_bar[task_id] = {"progress": n + 1, "total": self.integral_properties.number_of_paths}

        # Aggregate results
        self._aggregate_integral_results()

        pass

    def _aggregate_integral_results(self) -> None:
        """Aggregate results from line integrals into class attributes."""
        # catch RuntimeWarnings originating from np.nanmean having no valid values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            res_array = np.asarray(self.path_results)
            self.williams_int = np.asarray(self.williams_int)
            self.williams_int_a_n = np.asarray(self.williams_int_a_n)
            self.williams_int_b_n = np.asarray(self.williams_int_b_n)

            # replace any None values with 0 -> None means that the integral wasn't set to be calculated
            res_array[res_array == None] = 0
            self.williams_int[self.williams_int == None] = 0
            self.williams_int_a_n[self.williams_int_a_n == None] = 0
            self.williams_int_b_n[self.williams_int_b_n == None] = 0

            # Calculate means
            mean_j, mean_sif_j, mean_sif_k_i, mean_sif_k_ii, mean_t_stress_chen, mean_t_stress_sdm, mean_t_stress_int, \
                mean_decomp_j_1, mean_decomp_j_2_, mean_decomp_j_3, mean_decomp_K_1, mean_decomp_K_2, mean_decomp_K_3 = \
                np.nanmean(res_array, axis=0)
            mean_williams_int_a_n = np.nanmean(self.williams_int_a_n, axis=0)
            mean_williams_int_b_n = np.nanmean(self.williams_int_b_n, axis=0)

            # Calculate medians
            med_j, med_sif_j, med_sif_k_i, med_sif_k_ii, med_t_stress_chen, med_t_stress_sdm, med_t_stress_int, \
                med_decomp_j_1, med_decomp_j_2_, med_decomp_j_3, med_decomp_K_1, med_decomp_K_2, med_decomp_K_3 = \
                np.nanmedian(res_array, axis=0)
            med_williams_int_a_n = np.nanmedian(self.williams_int_a_n, axis=0)
            med_williams_int_b_n = np.nanmedian(self.williams_int_b_n, axis=0)

            # Calculate means rejecting outliers
            rej_out_mean_j, rej_out_mean_sif_j, rej_out_mean_sif_k_i, rej_out_mean_sif_k_ii, \
                rej_out_mean_t_stress_chen, rej_out_mean_t_stress_sdm, rej_out_mean_t_stress_int, \
                rej_decomp_j_1, rej_decomp_j_2_, rej_decomp_j_3, rej_decomp_K_1, rej_decomp_K_2, rej_decomp_K_3 = \
                self.mean_wo_outliers(res_array, m=2)

            rej_out_mean_williams_int_a_n = self.mean_wo_outliers(self.williams_int_a_n, m=2)
            rej_out_mean_williams_int_b_n = self.mean_wo_outliers(self.williams_int_b_n, m=2)

        # calculate SIFs with Bueckner-Chen integral method
        term_index = self.integral_properties.buckner_williams_terms.index(1)
        mean_k_i_chen = np.sqrt(2 * np.pi) * mean_williams_int_a_n[term_index] / np.sqrt(1000)
        med_k_i_chen = np.sqrt(2 * np.pi) * med_williams_int_a_n[term_index] / np.sqrt(1000)
        rej_out_mean_k_i_chen = np.sqrt(2 * np.pi) * rej_out_mean_williams_int_a_n[term_index] / np.sqrt(1000)
        mean_k_ii_chen = -np.sqrt(2 * np.pi) * mean_williams_int_b_n[term_index] / np.sqrt(1000)
        med_k_ii_chen = -np.sqrt(2 * np.pi) * med_williams_int_b_n[term_index] / np.sqrt(1000)
        rej_out_mean_k_ii_chen = -np.sqrt(2 * np.pi) * rej_out_mean_williams_int_b_n[term_index] / np.sqrt(1000)

        # bundle means / medians / means using outlier rejection
        self.sifs_int = {
            'mean': {'j': mean_j, 'sif_j': mean_sif_j,
                     'sif_k_i': mean_sif_k_i, 'sif_k_ii': mean_sif_k_ii,
                     'k_i_chen': mean_k_i_chen, 'k_ii_chen': mean_k_ii_chen,
                     't_stress_chen': mean_t_stress_chen,
                     't_stress_sdm': mean_t_stress_sdm,
                     't_stress_int': mean_t_stress_int,
                     'williams_int_a_n': mean_williams_int_a_n,
                     'williams_int_b_n': mean_williams_int_b_n,
                     'decomp_j_1': mean_decomp_j_1,
                     'decomp_j_2': mean_decomp_j_2_,
                     'decomp_j_3': mean_decomp_j_3,
                     'decomp_K_1': mean_decomp_K_1,
                     'decomp_K_2': mean_decomp_K_2,
                     'decomp_K_3': mean_decomp_K_3},
            'median': {'j': med_j, 'sif_j': med_sif_j,
                       'sif_k_i': med_sif_k_i, 'sif_k_ii': med_sif_k_ii,
                       'k_i_chen': med_k_i_chen, 'k_ii_chen': med_k_ii_chen,
                       't_stress_chen': med_t_stress_chen,
                       't_stress_sdm': med_t_stress_sdm,
                       't_stress_int': med_t_stress_int,
                       'williams_int_a_n': med_williams_int_a_n,
                       'williams_int_b_n': med_williams_int_b_n,
                       'decomp_j_1': med_decomp_j_1,
                       'decomp_j_2': med_decomp_j_2_,
                       'decomp_j_3': med_decomp_j_3,
                       'decomp_K_1': med_decomp_K_1,
                       'decomp_K_2': med_decomp_K_2,
                       'decomp_K_3': med_decomp_K_3},
            'rej_out_mean': {'j': rej_out_mean_j, 'sif_j': rej_out_mean_sif_j,
                             'sif_k_i': rej_out_mean_sif_k_i, 'sif_k_ii': rej_out_mean_sif_k_ii,
                             'k_i_chen': rej_out_mean_k_i_chen, 'k_ii_chen': rej_out_mean_k_ii_chen,
                             't_stress_chen': rej_out_mean_t_stress_chen,
                             't_stress_sdm': rej_out_mean_t_stress_sdm,
                             't_stress_int': rej_out_mean_t_stress_int,
                             'williams_int_a_n': rej_out_mean_williams_int_a_n,
                             'williams_int_b_n': rej_out_mean_williams_int_b_n,
                             'decomp_j_1': rej_decomp_j_1,
                             'decomp_j_2': rej_decomp_j_2_,
                             'decomp_j_3': rej_decomp_j_3,
                             'decomp_K_1': rej_decomp_K_1,
                             'decomp_K_2': rej_decomp_K_2,
                             'decomp_K_3': rej_decomp_K_3}
        }
        pass

    @staticmethod
    def mean_wo_outliers(data: np.ndarray, m=2) -> list:
        mean_wo_outliers = []
        for data_i in data.T:
            d = np.abs(data_i - np.nanmedian(data_i))
            mdev = np.nanmedian(d)
            s = d / mdev if mdev else 0
            mean_wo_outliers.append(np.nanmean(data_i[s < m]))
        return mean_wo_outliers
