"""Analysis facade coordinating established fracture-analysis techniques and
owning their latest authoritative and compatibility result surfaces.
"""

import logging
import warnings
from typing import Any, Mapping, MutableMapping, Optional, Union

import numpy as np
import rich.progress as progress_rich

from crackpy.fracture_analysis import line_integration
from crackpy.fracture_analysis._interpolation_cache import InterpolatorCache
from crackpy.fracture_analysis.line_integration import IntegralProperties, LineIntegral
from crackpy.fracture_analysis.crack_tip_fields.cjp import (
    CjpMixedModeCoefficients,
    CjpMixedModeQuantities,
    CjpModeICoefficients,
    CjpModeIQuantities,
)
from crackpy.fracture_analysis.crack_tip_fields.williams import (
    WilliamsInPlaneCoefficients,
    WilliamsInPlaneQuantities,
    WilliamsOutOfPlaneCoefficients,
    WilliamsOutOfPlaneQuantities,
)
from crackpy.fracture_analysis.odm._compatibility import (
    _project_cjp_mixed_mode_compatibility,
    _project_cjp_mode_i_compatibility,
    _project_williams_compatibility,
)
from crackpy.fracture_analysis.odm.results import OdmFitResult
from crackpy.fracture_analysis.odm.runners import (
    _build_cjp_mixed_mode_odm_result,
    _build_cjp_mode_i_odm_result,
    _build_williams_in_plane_odm_result,
    _build_williams_out_of_plane_odm_result,
)
from crackpy.fracture_analysis.odm.solvers import coefficient_fit_from_optimize_result
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
        self._cjp_mode_i_odm_result = None
        self._cjp_mixed_mode_odm_result = None
        self._williams_in_plane_odm_result = None
        self._williams_out_of_plane_odm_result = None

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

    @property
    def cjp_mode_i_odm_result(
        self,
    ) -> OdmFitResult[CjpModeICoefficients, CjpModeIQuantities] | None:
        """Return the latest authoritative CJP Mode I ODM result, if attempted."""
        return self._cjp_mode_i_odm_result

    @property
    def cjp_mixed_mode_odm_result(
        self,
    ) -> OdmFitResult[CjpMixedModeCoefficients, CjpMixedModeQuantities] | None:
        """Return the latest authoritative CJP mixed-mode ODM result, if attempted."""
        return self._cjp_mixed_mode_odm_result

    @property
    def williams_in_plane_odm_result(
        self,
    ) -> OdmFitResult[
        WilliamsInPlaneCoefficients,
        WilliamsInPlaneQuantities,
    ] | None:
        """Return the latest authoritative in-plane Williams ODM result."""
        return self._williams_in_plane_odm_result

    @property
    def williams_out_of_plane_odm_result(
        self,
    ) -> OdmFitResult[
        WilliamsOutOfPlaneCoefficients,
        WilliamsOutOfPlaneQuantities,
    ] | None:
        """Return the latest authoritative out-of-plane Williams ODM result."""
        return self._williams_out_of_plane_odm_result

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
            coefficient_fit = coefficient_fit_from_optimize_result(cjp_results_m1)
            result = _build_cjp_mode_i_odm_result(coefficient_fit)
            coefficients, quantities = _project_cjp_mode_i_compatibility(result)
            self._cjp_mode_i_odm_result = result
            self.cjp_coeffs_m1 = coefficients
            self.cjp_res_m1 = quantities

            logger.debug(
                "CJP Mode I optimization results: K_F=%.2f, K_R=%.2f, K_S=%.2f, T_x=%.2f, T_y=%.2f",
                result.quantities.k_f,
                result.quantities.k_r,
                result.quantities.k_s,
                result.quantities.t_x,
                result.quantities.t_y,
            )

        except Exception:
            logger.exception('CJP optimization (Mode I) failed. CJP Mode I optimization results set to NaN.')

            result = _build_cjp_mode_i_odm_result(None)
            coefficients, quantities = _project_cjp_mode_i_compatibility(result)
            self._cjp_mode_i_odm_result = result
            self.cjp_coeffs_m1 = coefficients
            self.cjp_res_m1 = quantities

    def _run_cjp_optimization_mixedmode(self) -> None:
        """Run CJP optimization if optimization properties are provided."""

        try:
            cjp_results = self.optimization.optimize_cjp_displacements_mixedmode()
            coefficient_fit = coefficient_fit_from_optimize_result(cjp_results)
            result = _build_cjp_mixed_mode_odm_result(coefficient_fit)
            coefficients, quantities = _project_cjp_mixed_mode_compatibility(result)
            self._cjp_mixed_mode_odm_result = result
            self.cjp_coeffs_mm = coefficients
            self.cjp_res_mm = quantities
            logger.debug(
                "CJP Mixed Mode optimization (Mixed Mode) results: K_F=%.2f, K_R=%.2f, K_S=%.2f, K_II=%.2f, T=%.2f",
                result.quantities.k_f,
                result.quantities.k_r,
                result.quantities.k_s,
                result.quantities.k_ii,
                result.quantities.t_stress,
            )
        except Exception:
            logger.exception('CJP optimization failed. CJP Mixed Mode optimization results set to NaN.')

            result = _build_cjp_mixed_mode_odm_result(None)
            coefficients, quantities = _project_cjp_mixed_mode_compatibility(result)
            self._cjp_mixed_mode_odm_result = result
            self.cjp_coeffs_mm = coefficients
            self.cjp_res_mm = quantities

    def _run_williams_optimization(self) -> None:
        """Run Williams optimization if optimization properties are provided."""

        terms = tuple(self.optimization.terms)
        skip_disp_z_optimization = self.data.disp_z is None or not np.any(
            self.data.disp_z)  # -> both None or all zeros mean no sensible z-displacements are provided

        try:
            williams_results_xy = self.optimization.optimize_williams_displacements_xy()
            in_plane_fit = coefficient_fit_from_optimize_result(williams_results_xy)
        except Exception:
            logger.exception(
                'Williams optimization for xy failed. Corresponding Williams optimization results set to NaN.')
            in_plane_fit = None

        if skip_disp_z_optimization:
            logging.info(
                'No sensible z-displacements provided; skipping z-direction optimization. Corresponding Williams optimization results set to NaN. ')
            out_of_plane_fit = None
        else:
            try:
                williams_results_z = self.optimization.optimize_williams_displacements_z()
                out_of_plane_fit = coefficient_fit_from_optimize_result(
                    williams_results_z
                )
            except Exception:
                logger.exception(
                    'Williams optimization for z-displacements failed. Corresponding Williams optimization results set to NaN.')
                out_of_plane_fit = None

        in_plane_result = _build_williams_in_plane_odm_result(
            terms,
            in_plane_fit,
        )
        out_of_plane_result = _build_williams_out_of_plane_odm_result(
            terms,
            out_of_plane_fit,
            skipped=skip_disp_z_optimization,
        )
        (
            coefficients,
            a_by_term,
            b_by_term,
            c_by_term,
            quantities,
        ) = _project_williams_compatibility(in_plane_result, out_of_plane_result)
        self._williams_in_plane_odm_result = in_plane_result
        self._williams_out_of_plane_odm_result = out_of_plane_result
        self.williams_coeffs = coefficients
        self.williams_fit_a_n = a_by_term
        self.williams_fit_b_n = b_by_term
        self.williams_fit_c_n = c_by_term
        self.williams_fit_res = quantities

        if in_plane_result.status == "completed":
            logger.debug(
                "Williams optimization results in xy-plane: Error_xy=%s, K_I=%.2f, K_II=%.2f, T=%.2f",
                in_plane_result.cost,
                in_plane_result.quantities.k_i,
                in_plane_result.quantities.k_ii,
                in_plane_result.quantities.t_stress,
            )
        if out_of_plane_result.status == "completed":
            logger.debug(
                "Williams optimization results in z-plane: Error_z=%s, K_III=%.2f",
                out_of_plane_result.cost,
                out_of_plane_result.quantities.k_iii,
            )

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

        interpolator_cache = InterpolatorCache(max_interpolators=4)
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
                                                          self.integral_properties.buckner_williams_terms,
                                                          interpolator_cache=interpolator_cache)

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

            # These object arrays require elementwise comparison; ``is None`` would be scalar.
            res_array[res_array == None] = 0  # noqa: E711
            self.williams_int[self.williams_int == None] = 0  # noqa: E711
            self.williams_int_a_n[self.williams_int_a_n == None] = 0  # noqa: E711
            self.williams_int_b_n[self.williams_int_b_n == None] = 0  # noqa: E711

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

    @staticmethod
    def mean_wo_outliers(data: np.ndarray, m=2) -> list:
        mean_wo_outliers = []
        for data_i in data.T:
            d = np.abs(data_i - np.nanmedian(data_i))
            mdev = np.nanmedian(d)
            s = d / mdev if mdev else 0
            mean_wo_outliers.append(np.nanmean(data_i[s < m]))
        return mean_wo_outliers
