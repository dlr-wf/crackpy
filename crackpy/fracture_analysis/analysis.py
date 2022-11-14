import warnings

import numpy as np
import rich.progress as progress_rich

from crackpy.fracture_analysis import line_integration
from crackpy.fracture_analysis.data_processing import InputData, CrackTipInfo
from crackpy.fracture_analysis.line_integration import IntegralProperties
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
    - K_F, K_R, K_S, K_II and T w/ the CJP model
    - (BETA) T-stress with the Bueckner-Chen integral
    - (BETA) higher-order terms (HOSTs and HORTs) w/ Bueckner-integral

    Methods:
        * run - run fracture analysis with the provided data

    """

    def __init__(
            self,
            material: Material,
            nodemap: Nodemap or str,
            data: InputData,
            crack_tip_info: CrackTipInfo,
            integral_properties: IntegralProperties or None = IntegralProperties(),
            optimization_properties: OptimizationProperties or None = OptimizationProperties()
    ):
        """Initialize FractureAnalysis class arguments.

        Args:
            material: obj of class Material, material parameters and laws
            nodemap: obj of class Nodemap or filename of file with exported Aramis-DIC data
            data: obj of class InputData, imported data from nodemap_file
            crack_tip_info: obj of class CrackTipInfo, crack tip information (i.e. x,y coordinates, angle, etc.)
            integral_properties: IntegralProperties or None,
                                 wrapper for specification of line integral properties
                                 If None, Bueckner-Chen integral is not calculated.
            optimization_properties: OptimizationProperties or None,
                                     If None, optimization / fitting is not performed.
        """
        self.material = material
        self.nodemap_file = nodemap.name if isinstance(nodemap, Nodemap) else nodemap
        self.data = data
        self.crack_tip = crack_tip_info

        self.optimization_properties = optimization_properties
        if self.optimization_properties is not None:

            # check if properties are provided and set defaults if necessary
            if self.optimization_properties.angle_gap is None:
                self.optimization_properties.angle_gap = 20
            if self.optimization_properties.min_radius is None:
                self.optimization_properties.min_radius = abs(self.crack_tip.crack_tip_x) / 20
            if self.optimization_properties.max_radius is None:
                self.optimization_properties.max_radius = abs(self.crack_tip.crack_tip_x) / 5
            if self.optimization_properties.tick_size is None:
                self.optimization_properties.tick_size = 0.01
            if self.optimization_properties.terms is None:
                self.optimization_properties.terms = [-1, 0, 1, 2, 3, 4, 5]
            for i in [1, 2]:  # check that SIFs and T can be calculated from the terms
                if i not in self.optimization_properties.terms:
                    self.optimization_properties.terms.append(i)
                    print(f"Williams optimization terms should include {i}. Added to terms.")
            self.optimization_properties.terms.sort()

            self.optimization = Optimization(data=self.data,
                                             options=self.optimization_properties,
                                             material=self.material)

            # Initialization of optimization output
            self.cjp_coeffs = None
            self.res_cjp = None
            self.williams_coeffs = None
            self.williams_fit_a_n = None
            self.williams_fit_b_n = None
            self.sifs_fit = None

        self.integral_properties = integral_properties
        if self.integral_properties is not None:
            if self.integral_properties.buckner_williams_terms is None:
                self.integral_properties.buckner_williams_terms = [1, 2, 3, 4, 5]
            elif 1 not in self.integral_properties.buckner_williams_terms:
                self.integral_properties.buckner_williams_terms.append(1)
                print('Buckner-Williams terms should include 1. Added to terms.')
            if 0 in self.integral_properties.buckner_williams_terms:
                self.integral_properties.buckner_williams_terms.remove(0)
                print('Buckner-Williams terms should not include 0. Removed from terms.')
            self.integral_properties.buckner_williams_terms.sort()

            # Initialization of integral output
            self.results = []
            self.williams_int_a_n = []
            self.williams_int_b_n = []
            self.williams_int = []
            self.sifs_int = None
            self.int_sizes = []
            self.integration_points = []
            self.tick_sizes = []
            self.num_of_path_nodes = []

    def run(self, progress=None, task_id=None):
        """Run fracture analysis with the provided data, crack_tip_info, and integral_properties.
        Results are stored as class instance attributes 'results', 'sifs', 'int_sizes', and 'path_nodes'.

        Args:
            progress: progress bar object handle (handed-over automatically during pipeline, not needed for single run)
            task_id: task id for progress bar (handed-over automatically during pipeline, not needed for single run)

        """
        if self.optimization_properties is not None:

            try:
                # calculate CJP coefficients with fitting method
                cjp_results = self.optimization.optimize_cjp_displacements()

                self.cjp_coeffs = cjp_results.x
                A_r, B_r, B_i, C, E = self.cjp_coeffs

                # from Christopher et al. (2013) "Extension of the CJP model to mixed mode I and mode II" formulas 4-8
                K_F = np.sqrt(np.pi / 2) * (A_r - 3 * B_r - 8 * E)
                K_R = -4 * np.sqrt(np.pi / 2) * (2 * B_i + E * np.pi)
                K_S = -np.sqrt(np.pi / 2) * (A_r + B_r)
                K_II = 2 * np.sqrt(2 * np.pi) * B_i
                T = -C
                # m to mm
                K_F /= np.sqrt(1000)
                K_R /= np.sqrt(1000)
                K_S /= np.sqrt(1000)
                K_II /= np.sqrt(1000)

                self.res_cjp = {'Error': cjp_results.cost, 'K_F': K_F, 'K_R': K_R, 'K_S': K_S, 'K_II': K_II, 'T': T}

            except:
                print('CJP optimization failed.')
                self.res_cjp = None

            try:
                # calculate Williams coefficients with fitting method
                williams_results = self.optimization.optimize_williams_displacements()
                self.williams_coeffs = williams_results.x
                a_n = self.williams_coeffs[:len(self.optimization.terms)]
                b_n = self.williams_coeffs[len(self.optimization.terms):]
                self.williams_fit_a_n = {n: a_n[index] for index, n in enumerate(self.optimization.terms)}
                self.williams_fit_b_n = {n: b_n[index] for index, n in enumerate(self.optimization.terms)}

                # derive stress intensity factors and T-stress [Kuna formula 3.45]
                K_I = np.sqrt(2 * np.pi) * self.williams_fit_a_n[1] / np.sqrt(1000)
                K_II = -np.sqrt(2 * np.pi) * self.williams_fit_b_n[1] / np.sqrt(1000)
                T = 4 * self.williams_fit_a_n[2]

                self.sifs_fit = {'Error': williams_results.cost,
                                 'K_I': K_I, 'K_II': K_II, 'T': T}

            except:
                print('Williams optimization failed.')
                self.sifs_fit = None

        if self.integral_properties is not None:
            # calculate Williams coefficients with Bueckner-Chen integral method
            current_size_left = self.integral_properties.integral_size_left
            current_size_right = self.integral_properties.integral_size_right
            current_size_top = self.integral_properties.integral_size_top
            current_size_bottom = self.integral_properties.integral_size_bottom

            if progress is None:
                iterator = progress_rich.track(range(self.integral_properties.number_of_paths),
                                               description='Calculating integrals')
            else:
                iterator = range(self.integral_properties.number_of_paths)

            for n in iterator:
                # Calculate one integral
                line_integral, current_int_sizes = self._calc_line_integral(
                    current_size_left,
                    current_size_right,
                    current_size_bottom,
                    current_size_top,
                    self.integral_properties.mask_tolerance,
                    self.integral_properties.buckner_williams_terms
                )

                self.results.append([line_integral.j_integral,
                                     line_integral.sif_k_j,
                                     line_integral.sif_k_i,
                                     line_integral.sif_k_ii,
                                     line_integral.t_stress_chen,
                                     line_integral.t_stress_sdm,
                                     line_integral.t_stress_int])
                self.williams_int_a_n.append(line_integral.williams_a_n)
                self.williams_int_b_n.append(line_integral.williams_b_n)
                self.williams_int.append(line_integral.williams_coefficients)
                self.int_sizes.append(current_int_sizes)
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
                if progress is not None and not "off":
                    progress[task_id] = {"progress": n + 1, "total": self.integral_properties.number_of_paths}

            # catch RuntimeWarnings originating from np.nanmean having no valid values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                res_array = np.asarray(self.results)
                self.williams_int = np.asarray(self.williams_int)
                self.williams_int_a_n = np.asarray(self.williams_int_a_n)
                self.williams_int_b_n = np.asarray(self.williams_int_b_n)

                # Calculate means
                mean_j, mean_sif_j, mean_sif_k_i, mean_sif_k_ii, mean_t_stress_chen, mean_t_stress_sdm, mean_t_stress_int = \
                    np.nanmean(res_array, axis=0)
                mean_williams_int_a_n = np.nanmean(self.williams_int_a_n, axis=0)
                mean_williams_int_b_n = np.nanmean(self.williams_int_b_n, axis=0)

                # Calculate medians
                med_j, med_sif_j, med_sif_k_i, med_sif_k_ii, med_t_stress_chen, med_t_stress_sdm, med_t_stress_int = \
                    np.nanmedian(res_array, axis=0)
                med_williams_int_a_n = np.nanmedian(self.williams_int_a_n, axis=0)
                med_williams_int_b_n = np.nanmedian(self.williams_int_b_n, axis=0)

                # Calculate means rejecting outliers
                rej_out_mean_j, rej_out_mean_sif_j, rej_out_mean_sif_k_i, rej_out_mean_sif_k_ii, \
                rej_out_mean_t_stress_chen, rej_out_mean_t_stress_sdm, rej_out_mean_t_stress_int = \
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
                         'williams_int_b_n': mean_williams_int_b_n},
                'median': {'j': med_j, 'sif_j': med_sif_j,
                           'sif_k_i': med_sif_k_i, 'sif_k_ii': med_sif_k_ii,
                           'k_i_chen': med_k_i_chen, 'k_ii_chen': med_k_ii_chen,
                           't_stress_chen': med_t_stress_chen,
                           't_stress_sdm': med_t_stress_sdm,
                           't_stress_int': med_t_stress_int,
                           'williams_int_a_n': med_williams_int_a_n,
                           'williams_int_b_n': med_williams_int_b_n},
                'rej_out_mean': {'j': rej_out_mean_j, 'sif_j': rej_out_mean_sif_j,
                                 'sif_k_i': rej_out_mean_sif_k_i, 'sif_k_ii': rej_out_mean_sif_k_ii,
                                 'k_i_chen': rej_out_mean_k_i_chen, 'k_ii_chen': rej_out_mean_k_ii_chen,
                                 't_stress_chen': rej_out_mean_t_stress_chen,
                                 't_stress_sdm': rej_out_mean_t_stress_sdm,
                                 't_stress_int': rej_out_mean_t_stress_int,
                                 'williams_int_a_n': rej_out_mean_williams_int_a_n,
                                 'williams_int_b_n': rej_out_mean_williams_int_b_n}
            }

    @staticmethod
    def mean_wo_outliers(data: np.ndarray, m=2) -> list:
        mean_wo_outliers = []
        for data_i in data.T:
            d = np.abs(data_i - np.nanmedian(data_i))
            mdev = np.nanmedian(d)
            s = d / mdev if mdev else 0
            mean_wo_outliers.append(np.nanmean(data_i[s < m]))
        return mean_wo_outliers

    def _calc_line_integral(self,
                            size_left: float,
                            size_right: float,
                            size_bottom: float,
                            size_top: float,
                            mask_tol: float = None,
                            buckner_williams_terms: list = None) -> tuple:
        """Line integration for one single path.

        Args:
            size_left: actual size of integration path from crack tip to left boarder
            size_right: actual size of integration path from crack tip to right boarder
            size_bottom: actual size of integration path from crack tip to bottom
            size_top: actual size of integration path from crack tip to top
            mask_tol: tolerance of the quadratic interpolation mask around the integration path
            buckner_williams_terms: list of terms which should be calculated (i.e. '[-1, 2, 4]')

        Returns:
            (tuple) line_integral, int_sizes, path_nodes
                - line_integral (LineIntegral object) LineIntegral object for one single path
                - int_sizes (list) containing the integration path size,
                                   i.e. the old sizes plus the distance between single paths

        """
        # Define path properties
        path_properties = line_integration.PathProperties(size_left, size_right, size_bottom, size_top,
                                                          self.integral_properties.integral_tick_size,
                                                          self.integral_properties.number_of_nodes,
                                                          self.integral_properties.top_offset,
                                                          self.integral_properties.bottom_offset)

        # Define integration path
        integration_path = line_integration.IntegrationPath(0, 0, path_properties=path_properties)
        _ = integration_path.create_nodes()

        # Define line integration
        line_integral = line_integration.LineIntegral(integration_path, self.data, self.material, mask_tol,
                                                      buckner_williams_terms)
        # Calculate SIFs
        line_integral.integrate()

        return line_integral, [size_left, size_right, size_bottom, size_top]
