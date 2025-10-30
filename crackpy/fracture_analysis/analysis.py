import logging
import warnings
from typing import Union, Optional, Mapping, Any, MutableMapping

import numpy as np
import rich.progress as progress_rich

from crackpy.fracture_analysis import line_integration
from crackpy.fracture_analysis.line_integration import (IntegralProperties,
                                                        LineIntegral)
from crackpy.fracture_analysis.optimization import Optimization, OptimizationProperties
from crackpy.input.crack_tip_info import CrackTipInfo
from crackpy.input.input_data import InputData
from crackpy.structure_elements.data_files import Nodemap
from crackpy.structure_elements.material import Material

logger = logging.getLogger(__name__)


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

        # Available Optimization results
        self.cjp_coeffs_mm = None
        self.cjp_res_mm = None
        self.cjp_coeffs_m1 = None
        self.cjp_res_m1 = None
        self.williams_coeffs = None
        self.williams_fit_a_n = None
        self.williams_fit_b_n = None
        self.williams_fit_c_n = None
        self.williams_fit_res = None

        # Available Line Integral results
        self.path_results = []
        self.williams_int_a_n = []
        self.williams_int_b_n = []
        self.williams_int = []
        self.sifs_int = None
        self.path_sizes = []
        self.integration_points = []
        self.tick_sizes = []
        self.num_of_path_nodes = []

        # Initialization of optimization and integral properties
        self.optimization_properties = optimization_properties
        if self.optimization_properties is not None:
            Optimization.ensure_defaults_williams(self.optimization_properties, self.crack_tip.crack_tip_x)
            self.optimization = Optimization(data=self.data,
                                             options=self.optimization_properties,
                                             material=self.material)

        self.integral_properties = integral_properties
        if self.integral_properties is not None:
            LineIntegral.ensure_defaults_buckner_chen(self.integral_properties)

    def run(self, progress_bar: Optional[Mapping[str, object]] = None, task_id=None):
        """Run fracture analysis with the provided data, crack_tip_info, and integral_properties.
        Results are stored as class instance attributes 'results', 'sifs', 'path_sizes', and 'path_nodes'.

        Args:
            progress_bar: whether to show progress bar for line integral calculation
            task_id: task id for progress bar (handed-over automatically during pipeline, not needed for single run)

        """
        logger.info("Starting fracture analysis for %s", self.nodemap_file)
        logger.debug(
            "Crack tip: x=%.2f, y=%.2f, angle=%.2f deg, side=%s",
            self.crack_tip.crack_tip_x,
            self.crack_tip.crack_tip_y,
            self.crack_tip.crack_tip_angle,
            self.crack_tip.left_or_right,
        )

        # Set the optimization and line integral methods that should be run
        if self.optimization_properties is not None:
            logger.info("Running optimization (fitting) methods …")
            logger.debug(
                "Optimization settings: min_r=%.2f, max_r=%.2f, angle_gap=%s deg, terms=%s",
                self.optimization_properties.min_radius,
                self.optimization_properties.max_radius,
                self.optimization_properties.angle_gap,
                self.optimization_properties.terms,
            )
            self._run_cjp_optimization_modeI()
            self._run_cjp_optimization_mixedmode()
            self._run_williams_optimization()
        else:
            logger.info('No optimization properties provided; skipping optimizations.')

        if self.integral_properties is not None:
            logger.info('Running line integral methods …')
            logger.debug("Integral settings: %d paths, sizes: left=%.2f, right=%.2f",
                         self.integral_properties.number_of_paths,
                         self.integral_properties.integral_size_left,
                         self.integral_properties.integral_size_right)

            self._run_line_integrals(progress_bar, task_id)
        else:
            logger.info('No integral properties provided; skipping line integrals.')

        logger.info("Fracture analysis completed for %s", self.nodemap_file)

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

            logger.debug(
                "CJP Mode I optimization results: K_F=%.2f, K_R=%.2f, K_S=%.2f, T_x=%.2f, T_y=%.2f",
                K_F, K_R, K_S, T_x, T_y,
            )

        except Exception:
            logger.exception('CJP optimization (Mode I) failed. CJP Mode I optimization results set to NaN.')

            self.cjp_res_m1 = {'Error': np.nan, 'K_F': np.nan, 'K_R': np.nan, 'K_S': np.nan, 'T_x': np.nan,
                               'T_y': np.nan}

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
            logger.debug(
                "CJP Mixed Mode optimization (Mixed Mode) results: K_F=%.2f, K_R=%.2f, K_S=%.2f, K_II=%.2f, T=%.2f",
                K_F, K_R, K_S, K_II, T,
            )
        except Exception:
            logger.exception('CJP optimization failed. CJP Mixed Mode optimization results set to NaN.')

            self.cjp_res_mm = {'Error': np.nan, 'K_F': np.nan, 'K_R': np.nan, 'K_S': np.nan, 'K_II': np.nan,
                               'T': np.nan}

    def _run_williams_optimization(self) -> None:
        """Run Williams optimization if optimization properties are provided."""

        n_terms = len(self.optimization.terms)
        skip_disp_z_optimization = self.data.disp_z is None or not np.any(
            self.data.disp_z)  # -> both None or all zeros mean no sensible z-displacements are provided

        try:
            # calculate Williams coefficients with fitting method in 2D (xy-plane)
            williams_results_xy = self.optimization.optimize_williams_displacements_xy()
            williams_coeffs_xy = williams_results_xy.x

            a_n = williams_coeffs_xy[:n_terms]
            b_n = williams_coeffs_xy[n_terms:]
            self.williams_fit_a_n = self._coeff_map(a_n)
            self.williams_fit_b_n = self._coeff_map(b_n)

            # derive stress intensity factors and T-stress [Kuna formula 3.45]
            K_I = np.sqrt(2 * np.pi) * self.williams_fit_a_n[1] / np.sqrt(1000)
            K_II = -np.sqrt(2 * np.pi) * self.williams_fit_b_n[1] / np.sqrt(1000)
            T = 4 * self.williams_fit_a_n[2]

            # store intermediate results
            self.williams_coeffs = williams_coeffs_xy
            self.williams_fit_res = {'Error_xy': williams_results_xy.cost, 'K_I': K_I, 'K_II': K_II, 'T': T}

            logger.debug(
                "Williams optimization results in xy-plane: Error_xy=%s, K_I=%.2f, K_II=%.2f, T=%.2f",
                williams_results_xy.cost, K_I, K_II, T,
            )
        except Exception:
            logger.exception(
                'Williams optimization for xy failed. Corresponding Williams optimization results set to NaN.')

            self.williams_coeffs = np.array([np.nan] * (2 * n_terms))
            self.williams_fit_a_n = self._coeff_map([np.nan] * n_terms)
            self.williams_fit_b_n = self._coeff_map([np.nan] * n_terms)
            self.williams_fit_res = {'Error_xy': np.nan, 'K_I': np.nan, 'K_II': np.nan, 'T': np.nan}

        if skip_disp_z_optimization:
            logging.info(
                'No sensible z-displacements provided; skipping z-direction optimization. Corresponding Williams optimization results set to NaN. ')

            self.williams_coeffs = np.r_[self.williams_coeffs, np.array([np.nan] * n_terms)]
            self.williams_fit_c_n = self._coeff_map([np.nan] * n_terms)
            self.williams_fit_res.update({'Error_z': np.nan, 'K_III': np.nan})
            return

        try:
            # calculate Williams coefficients with fitting method in z-direction
            williams_results_z = self.optimization.optimize_williams_displacements_z()
            williams_coeffs_z = williams_results_z.x

            c_n = williams_coeffs_z
            self.williams_fit_c_n = self._coeff_map(c_n)

            # derive stress intensity factors for Mode III
            K_III = np.sqrt(0.5 * np.pi) * self.williams_fit_c_n[1] / np.sqrt(1000)

            # update with final results
            self.williams_coeffs = np.r_[self.williams_coeffs, williams_coeffs_z]
            self.williams_fit_res.update({'Error_z': williams_results_z.cost, 'K_III': K_III})

            logger.debug(
                "Williams optimization results in z-plane: Error_z=%s, K_III=%.2f",
                williams_results_z.cost, K_III,
            )
        except Exception:
            logger.exception(
                'Williams optimization for z-displacements failed. Corresponding Williams optimization results set to NaN.')

            self.williams_coeffs = np.r_[self.williams_coeffs, np.array([np.nan] * n_terms)]
            self.williams_fit_c_n = self._coeff_map([np.nan] * n_terms)
            self.williams_fit_res.update({'Error_z': np.nan, 'K_III': np.nan})

    def _run_line_integrals(self, progress_bar: Optional[MutableMapping[str, Any]] = None, task_id=None) -> None:
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

    ###########
    # Helpers #
    ###########

    def _coeff_map(self, values):
        """Map optimization term indices to their corresponding coefficient values."""
        return {n: values[i] for i, n in enumerate(self.optimization.terms)}

    @staticmethod
    def mean_wo_outliers(data: np.ndarray, m=2) -> list:
        mean_wo_outliers = []
        for data_i in data.T:
            d = np.abs(data_i - np.nanmedian(data_i))
            mdev = np.nanmedian(d)
            s = d / mdev if mdev else 0
            mean_wo_outliers.append(np.nanmean(data_i[s < m]))
        return mean_wo_outliers
