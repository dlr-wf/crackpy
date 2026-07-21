"""The line-integral facade provides CrackPy's established mutable analysis API
and projects completed Contour-Wise Results onto its public attributes.
"""

import logging

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import label

from crackpy.fracture_analysis._interpolation_cache import (
    InterpolationTarget,
    InterpolatorCache,
    ReusableLinearInterpolator,
)
from crackpy.fracture_analysis.crack_tip_fields.williams import (
    quantities as williams_quantities,
)
from crackpy.fracture_analysis.line_integrals import (
    ContourWiseLineIntegralResult,
    auxiliary_fields,
    mode_decomposition,
    sampling,
)
from crackpy.fracture_analysis.line_integrals._compatibility import (
    _UNSET,
    _DeprecatedBuecknerSpellingAliases,
    mutable_williams_a_n,
    mutable_williams_b_n,
    mutable_williams_coefficients,
    resolve_bueckner_williams_terms,
)
from crackpy.fracture_analysis.line_integrals.contours import (
    IntegrationContour,
    build_rectangular_integration_contour,
)
from crackpy.fracture_analysis.line_integrals.runners import _LineIntegralExecution
from crackpy.input.input_data import InputData
from crackpy.structure_elements.material import Material

logger = logging.getLogger(__name__)

DEFAULT_BUECKNER_CHEN_TERMS = [1, 2, 3, 4, 5]
# Deprecated: use DEFAULT_BUECKNER_CHEN_TERMS.
DEFAULT_BUCKNER_CHEN_TERMS = DEFAULT_BUECKNER_CHEN_TERMS


class IntegralProperties(_DeprecatedBuecknerSpellingAliases):
    """Integral properties which are used for more than one line integration within Fracture Analysis.

    Methods:
        * set_automatically - defines integral properties automatically using the input *data*

    """

    def __init__(
            self,
            number_of_paths: int = 9,
            integral_tick_size: float = None,
            number_of_nodes: int = None,

            integral_size_left: float = None,
            integral_size_right: float = None,
            integral_size_bottom: float = None,
            integral_size_top: float = None,

            top_offset: float = None,
            bottom_offset: float = None,

            paths_distance_left: float = None,
            paths_distance_right: float = None,
            paths_distance_top: float = None,
            paths_distance_bottom: float = None,

            mask_tolerance: float = None,

            bueckner_williams_terms: list | None | object = _UNSET,
            *,
            buckner_williams_terms: list | None | object = _UNSET,
    ):
        """Initialize integral path properties.

        Args:
            number_of_paths: number of integration paths for one crack tip (usually should be >=9)
            integral_tick_size: distance between integration points (i.e. 0.5 mm). If None, then number_of_nodes
                                is used instead, or the tick_size needs to be calculated from the pipeline
                                method *find_integral_props*
            number_of_nodes: number of integral nodes per path. If None, number of nodes varies for each path and
                                   is calculated from the integral_tick_size.
            integral_size_left: size of first integration path from crack tip to left boarder
                                (use negative value)
            integral_size_right: size of first integration path from crack tip to right boarder
            integral_size_bottom: size of first integration path from crack tip to bottom
                                  (use negative value)
            integral_size_top: size of first integration from crack tip to top
            top_offset: distance of unclosed part of the integral from the crack path (top side)
            bottom_offset: distance of unclosed part of the integral from the crack path (bottom side)
                            (use negative value)
            paths_distance_left: distance of integration paths (left side)
            paths_distance_right: distance of integration paths (right side)
            paths_distance_top: distance of integration paths (top side)
            paths_distance_bottom: distance of integration paths (bottom side)
            mask_tolerance: tolerance of the quadratic interpolation mask around the integration path
                            (fails if too small)
            bueckner_williams_terms: Williams coefficient orders evaluated
                with the Bueckner-Chen Integral.
            buckner_williams_terms: Deprecated spelling; use
                ``bueckner_williams_terms``.

        """
        self.number_of_paths = number_of_paths
        self.integral_tick_size = integral_tick_size
        self.number_of_nodes = number_of_nodes

        self.integral_size_left = integral_size_left
        self.integral_size_right = integral_size_right
        self.integral_size_bottom = integral_size_bottom
        self.integral_size_top = integral_size_top

        self.top_offset = top_offset
        self.bottom_offset = bottom_offset

        self.paths_distance_left = paths_distance_left
        self.paths_distance_right = paths_distance_right
        self.paths_distance_top = paths_distance_top
        self.paths_distance_bottom = paths_distance_bottom

        self.mask_tolerance = mask_tolerance

        self.bueckner_williams_terms = resolve_bueckner_williams_terms(
            bueckner_williams_terms,
            buckner_williams_terms,
        )

    def set_automatically(self, data: InputData, auto_detect_threshold: float):
        """Automatically set up the integration path properties.

        Args:
            data: obj of class InputData, used for auto-detection
            auto_detect_threshold: threshold stress typically taken equal to yield stress

        """
        logger.debug("Starting automatic integral path detection with threshold=%.2f MPa", auto_detect_threshold)

        if data.sig_vm is None:
            raise ValueError("Stresses need to be calculated before using ``data`` by calling data.calc_stresses()")
        # Calculate face size
        facet_size = data.get_facet_size()
        logger.debug("Calculated facet size: %.4f mm", facet_size)

        # Map data on regular grid
        x_min = facet_size * 2.0
        grid_x, grid_y = np.mgrid[-x_min:max(data.coor_x): 500j, min(data.coor_y):max(data.coor_y): 500j]
        ngrid = griddata((data.coor_x, data.coor_y), data.sig_vm, (grid_x, grid_y), method='linear')

        # Apply threshold
        threshold_array = ngrid > auto_detect_threshold
        threshold_array = threshold_array.astype(int)
        labeled_images, num_features = label(threshold_array)
        logger.debug("Found %d features above threshold", num_features)

        object_label = -1

        for i_feature in range(1, num_features + 1):
            mask = labeled_images == i_feature
            if np.any(np.all([grid_x[mask] > -facet_size,
                              grid_x[mask] < facet_size,
                              grid_y[mask] > -facet_size,
                              grid_y[mask] < facet_size], axis=0)):
                object_label = i_feature

        labeled_image = (labeled_images == object_label).astype(int)
        dist_factor = 2.0
        object_indices = np.argwhere(labeled_image > 0)

        if self.integral_size_right is None:
            self.integral_size_right = grid_x[max(object_indices[:, 0]), 0] + dist_factor * facet_size
        if self.integral_size_left is None:
            self.integral_size_left = - self.integral_size_right * 0.5 - facet_size
        if self.integral_size_bottom is None:
            self.integral_size_bottom = grid_y[0, min(object_indices[:, 1])] - dist_factor * facet_size
        if self.integral_size_top is None:
            self.integral_size_top = grid_y[0, max(object_indices[:, 1])] + dist_factor * facet_size
        if self.integral_tick_size is None:
            self.integral_tick_size = facet_size / 4.0

        # Set offsets
        ticks = int((self.integral_size_top - self.integral_size_bottom) / self.integral_tick_size * 8.0)
        coor_y = np.linspace(self.integral_size_bottom, self.integral_size_top, ticks, endpoint=True)
        coor_x = coor_y * 0.0 + self.integral_size_left
        sigma_vm_interpolated = griddata((data.coor_x, data.coor_y),
                                         data.sig_vm,
                                         (coor_x, coor_y),
                                         method='linear')

        y_min = 10000.0
        y_max = -10000.0
        for i, stress in enumerate(sigma_vm_interpolated):
            if abs(stress) > auto_detect_threshold:
                if coor_y[i] < y_min:
                    y_min = coor_y[i]
                if coor_y[i] > y_max:
                    y_max = coor_y[i]
        if y_min == 10000.0:
            y_min = 0
        if y_max == -10000.0:
            y_max = 0

        if self.top_offset is None:
            self.top_offset = y_max
        if self.bottom_offset is None:
            self.bottom_offset = y_min

        # Set path distances
        if self.paths_distance_left is None:
            self.paths_distance_left = facet_size
        if self.paths_distance_right is None:
            self.paths_distance_right = facet_size
        if self.paths_distance_top is None:
            self.paths_distance_top = facet_size
        if self.paths_distance_bottom is None:
            self.paths_distance_bottom = facet_size

        logger.debug("Automatic integral path detection completed:")
        logger.debug("  Integral sizes: left=%.2f, right=%.2f, top=%.2f, bottom=%.2f",
                     self.integral_size_left, self.integral_size_right, self.integral_size_top,
                     self.integral_size_bottom)
        logger.debug("  Offsets: top=%.2f, bottom=%.2f", self.top_offset, self.bottom_offset)
        logger.debug("  Tick size: %.4f mm", self.integral_tick_size)


class PathProperties:
    def __init__(self, size_left: float, size_right: float, size_bottom: float, size_top: float, tick_size: float,
                 num_nodes: int, top_offset: float, bottom_offset: float):
        """Properties of one single line integration path.

        Args:
            size_left: size of integration path from crack tip to left boarder (use negative value)
            size_right: size of integration path from crack tip to right boarder
            size_bottom: size of integration path from crack tip to bottom (use negative value)
            size_top: size of integration from crack tip to top
            tick_size: distance between integration points (i.e. 0.5 mm)
            num_nodes: number of nodes per integration path
            top_offset: distance of unclosed part of the integral from the crack path (top side)
            bottom_offset: distance of unclosed part of the integral from the crack path (top side)

        """
        self.size_left = size_left
        self.size_right = size_right
        self.size_bottom = size_bottom
        self.size_top = size_top
        self.tick_size = tick_size
        self.number_of_nodes = num_nodes
        self.top_offset = top_offset
        self.bottom_offset = bottom_offset


class IntegrationPath:
    """Wrapper for integration path functionalities.

    Methods:
        * create_nodes - list nodes specifying elements of the integration path
        * get_integration_points - coordinates of integration points and element sizes

    """

    def __init__(self, origin_x: float = 0.0, origin_y: float = 0.0, path_properties: PathProperties = None):
        """Initialize path properties.

        Args:
            origin_x: (float) refers to the point used as center for the integral.
            origin_y: (float) refers to the point used as center for the integral.
            path_properties: (PathProperties object) considering the base integration path plot_sets for size
                                                     and shape of the integration path
        
        """
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.path_properties = path_properties

        self.nodes = None
        self.int_points = None

        self._create_nodes()
        self._create_integration_points()

    def _create_nodes(self):
        """Creates a list of nodes specifying the elements for the integration path. The number of nodes is taken from
        self.path_properties and then constant for each path or it is calculated from self.path_properties.tick_size and
        then it varies for each path due to constant tick_size. If neither tick_size nor num_of_nodes is None, then
        num_of_nodes is taken and tick_size is ignored.

        If only path.number_of_nodes is given, then the tick_size is calculated and written into self.path_properties.

        Returns:
            int_points with all x-coors at index 0 and all y-coors at index 1, again as lists

        """
        path = self.path_properties
        contour = build_rectangular_integration_contour(
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            size_left=path.size_left,
            size_right=path.size_right,
            size_bottom=path.size_bottom,
            size_top=path.size_top,
            tick_size=path.tick_size,
            number_of_nodes=path.number_of_nodes,
            top_offset=path.top_offset,
            bottom_offset=path.bottom_offset,
        )
        path.number_of_nodes = contour.number_of_nodes
        path.tick_size = contour.tick_size
        self._contour = contour
        self.nodes = [
            contour.nodes[:, 0].copy().tolist(),
            contour.nodes[:, 1].copy().tolist(),
        ]

    def _create_integration_points(self):
        """Generates coordinates of integration points and element sizes.

        Returns:
            int_points [[x_coor, y_coor, delta_x, delta_y]]

        """
        self.int_points = self._contour.integration_points.copy()

    def get_integration_points(self) -> np.ndarray:
        """Get integration points of the integration path.

        Returns:
            int_points [[x_coor, y_coor, delta_x, delta_y]]

        """
        return self.int_points


class LineIntegral(_DeprecatedBuecknerSpellingAliases):
    """Line integral object for solving J-Integral and Interaction Integral for given material's input data
    and path of integration.

    Methods:
        * integrate - solve for J-integral, interaction integral, Bueckner/Chen integral

    """

    def __init__(self,
                 integration_path: IntegrationPath,
                 data: InputData,
                 material: Material,
                 mask_tol: float = None,
                 bueckner_williams_terms: list | None | object = _UNSET,
                 interpolator_cache: InterpolatorCache | None = None,
                 *,
                 buckner_williams_terms: list | None | object = _UNSET):
        """Get integration points and interpolate data onto grid.

        Args:
            integration_path: obj of class IntegrationPath
            data: obj of class InputData, Input data object containing the full field DIC or FE data
            material: obj of class Material
            mask_tol: (float or None) tolerance of the quadratic interpolation mask around the integration path
            bueckner_williams_terms: Williams coefficient orders evaluated
                with the Bueckner-Chen Integral.
            interpolator_cache: cache shared by one line-integral Analysis Run, or ``None`` for a private bounded cache
            buckner_williams_terms: Deprecated spelling; use
                ``bueckner_williams_terms``.

        """
        # input
        self.data = data
        self.data_orig = None  # for mode decomposition
        self.integration_path = integration_path
        self.x_shift = integration_path.path_properties.tick_size
        self.origin_x = integration_path.origin_x
        self.origin_y = integration_path.origin_y
        self.material = material
        self.bueckner_williams_terms = resolve_bueckner_williams_terms(
            bueckner_williams_terms,
            buckner_williams_terms,
        )
        self.mask_tol = mask_tol
        self.np_integration_points = None
        self.tri = None
        self._interpolator_cache = (
            interpolator_cache
            if interpolator_cache is not None
            else InterpolatorCache(max_interpolators=4)
        )

        # output
        self.j_integral = None
        self.decomp_j_integral_I = None
        self.decomp_j_integral_II = None
        self.decomp_j_integral_III = None
        self.decomp_j_integral_K_I = None
        self.decomp_j_integral_K_II = None
        self.decomp_j_integral_K_III = None
        self.sif_k_j = None
        self.sif_k_i = None
        self.sif_k_ii = None
        self.t_stress_sdm = None
        self.t_stress_chen = None
        self.t_stress_int = None
        self.williams_a_n = []
        self.williams_b_n = []
        self.williams_coefficients = []

        self.__post_init__()

    def __post_init__(self):
        self._contour = IntegrationContour(
            origin=(self.origin_x, self.origin_y),
            nodes=np.column_stack(self.integration_path.nodes),
            integration_points=self.integration_path.get_integration_points(),
            number_of_nodes=(
                self.integration_path.path_properties.number_of_nodes
            ),
            tick_size=self.integration_path.path_properties.tick_size,
        )
        self.np_integration_points = self._contour.integration_points.copy()
        self._execution = _LineIntegralExecution(
            self._contour,
            self.data,
            self.material,
            self.mask_tol,
            self.bueckner_williams_terms,
            self._interpolator_cache,
        )
        self._build_path_geometry()
        self._adopt_in_plane_samples(self._execution.in_plane_samples)

    def integrate_all(self) -> ContourWiseLineIntegralResult:
        """Call this method to solve all integrals
        - J-integral
        - Mode I, II, III decomposition of J-integral
        - Interaction integral for SIF K_I and K_II
        - T-stress with interaction integral method
        - T-stress with stress difference method
        - Williams coefficients with Bueckner-Chen method (if terms are given)

        Returns:
            The Contour-Wise Result whose values also populate the facade's
            mutable result attributes.
        """
        logger.debug("Starting integration for all methods, integration points: %d", len(self.np_integration_points))

        self._synchronize_execution()
        result = self._execution.evaluate_all()
        self._adopt_in_plane_samples(self._execution.in_plane_samples)
        self._project_completed_result(result)
        return result

    def _project_completed_result(self, result: ContourWiseLineIntegralResult) -> None:
        """Project one completed result onto the facade's mutable attributes."""
        quantities = result.quantities
        self.j_integral = quantities.j_integral
        self.sif_k_j = quantities.sif_k_j
        self.sif_k_i = quantities.sif_k_i
        self.sif_k_ii = quantities.sif_k_ii
        self.t_stress_chen = quantities.t_stress_chen
        self.t_stress_sdm = np.asarray(quantities.t_stress_sdm)
        self.t_stress_int = quantities.t_stress_int
        self.decomp_j_integral_I = quantities.decomp_j_integral_i
        self.decomp_j_integral_II = quantities.decomp_j_integral_ii
        self.decomp_j_integral_III = quantities.decomp_j_integral_iii
        self.decomp_j_integral_K_I = quantities.decomp_j_integral_k_i
        self.decomp_j_integral_K_II = quantities.decomp_j_integral_k_ii
        self.decomp_j_integral_K_III = quantities.decomp_j_integral_k_iii
        self.williams_a_n.extend(mutable_williams_a_n(result))
        self.williams_b_n.extend(mutable_williams_b_n(result))
        self.williams_coefficients.extend(mutable_williams_coefficients(result))

    def _synchronize_execution(self) -> None:
        """Synchronize mutable facade inputs with the contour execution."""
        self._execution.data = self.data
        self._execution.material = self.material
        self._execution.mask_tolerance = self.mask_tol
        self._execution.requested_bueckner_williams_terms = (
            self.bueckner_williams_terms
        )
        current_samples = self._execution.sample_in_plane_fields(self.data)
        self._adopt_in_plane_samples(current_samples)

    ###########################################
    # METHODS FOR CALCULATING THE DESCRIPTORS #
    ###########################################

    def integrate_j(self):
        """Call this method to solve integrals for J-integral :math:`J`

        Returns:
            None. The J-Integral and energy-equivalent SIF facade
            attributes are updated in place.
        """
        self._synchronize_execution()
        self.j_integral, self.sif_k_j = self._execution.evaluate_j_integral()

    def integrate_j_decompose(self):
        """Call this method to solve integrals for J-integral :math:`J` and its mode I, II, III decomposition
            Negative J values are sanitized to NaN for SIF calculation.

        Returns:
            None. Modal J-Integral and modal SIF facade attributes are
            updated in place.
        """
        #############################################
        # Mode decomposition of J integral
        # see: Molteno, M. R., & Becker, T. H. (2015). Mode I-III decomposition of the j-integral from DIC
        # displacement data. Strain, 51(6), 492–503. https://doi.org/10.1111/str.12166
        #############################################
        self._synchronize_execution()
        (
            self.decomp_j_integral_I,
            self.decomp_j_integral_II,
            self.decomp_j_integral_III,
            self.decomp_j_integral_K_I,
            self.decomp_j_integral_K_II,
            self.decomp_j_integral_K_III,
        ) = self._execution.evaluate_j_decomposition()
        self._adopt_in_plane_samples(self._execution.in_plane_samples)

    def integrate_i_k1_k2(self):
        """Populate signed Mode I and Mode II SIFs by interaction integral.

        Returns:
            None. The interaction-integral SIF facade attributes are
            updated in place.
        """
        #############################################
        # see Meinhard Kuna Section 6.7.2 for details
        #############################################
        self._synchronize_execution()
        self.sif_k_i, self.sif_k_ii = self._execution.evaluate_interaction_sifs()

    def integrate_i_t(self):
        """Populate T-Stress by the interaction-integral method.

        Returns:
            None. The interaction-integral T-Stress facade attribute is
            updated in place.
        """
        self._synchronize_execution()
        self.t_stress_int = self._execution.evaluate_interaction_t_stress()

    def integrate_t_sdm(self):
        """Populate T-Stress by the Stress-Difference Method.

        Returns:
            None. The Stress-Difference Method facade attribute is
            updated in place.
        """
        self._synchronize_execution()
        self.t_stress_sdm = np.asarray(
            self._execution.evaluate_stress_difference_t_stress()
        )

    def integrate_bueckner_chen(self):
        """Populate Williams and T-Stress results by the Bueckner-Chen Integral.

        Returns:
            None. The Williams coefficient lists and ``t_stress_chen``
            facade result are updated in place.
        """
        self._synchronize_execution()
        coefficients, self.t_stress_chen = self._execution.evaluate_bueckner_chen(
            self.bueckner_williams_terms
        )
        self.williams_a_n.extend(mutable_williams_a_n(coefficients))
        self.williams_b_n.extend(mutable_williams_b_n(coefficients))
        self.williams_coefficients.extend(mutable_williams_coefficients(coefficients))

    def integrate_buckner_chen(self):
        """Deprecated: use :meth:`integrate_bueckner_chen`.

        Returns:
            None. :meth:`integrate_bueckner_chen` updates the facade results.
        """
        return self.integrate_bueckner_chen()

    #########################################
    # PRIVATE FUNCTIONAL DELEGATION METHODS #
    #########################################

    # base method for J-integral, used in child classes
    def _solve_j_integral(self) -> float:
        """Function that returns the J-integral as a line integration.

        Returns:
            J-integral value

        """
        return self._execution._solve_j_integral(self._in_plane_samples)

    def _solve_j_integral_III(self) -> float:
        """Function that returns the J-integral as a line integration.

        Returns:
            J-integral value

        """
        return self._execution._solve_mode_iii_j_integral(self._mode_iii_samples)

    def _evaluate_shifted_auxiliary_fields(self, evaluator):
        """Evaluate an analytical field on aligned base and shifted contour points."""
        self._shifted_auxiliary_fields = auxiliary_fields.evaluate_shifted_auxiliary_fields(
            evaluator,
            self._integration_contour_geometry,
        )
        return (
            self._shifted_auxiliary_fields.base.copy(),
            self._shifted_auxiliary_fields.positive_x.copy(),
            self._shifted_auxiliary_fields.negative_x.copy(),
        )

    def _get_auxiliary_crack_nearfield(self, ki_aux: float, kii_aux: float):
        """Evaluate crack-nearfield tensors and the shifted displacement derivative in batches."""
        self._auxiliary_in_plane_fields = auxiliary_fields.prepare_lefm_auxiliary_fields(
            ki_aux,
            kii_aux,
            self._integration_contour_geometry,
            material=self.material,
        )
        return (
            self._auxiliary_in_plane_fields.stress.copy(),
            self._auxiliary_in_plane_fields.strain.copy(),
            self._auxiliary_in_plane_fields.displacement_y_gradient_x.copy(),
        )

    def _get_auxiliary_zhao_fields(self):
        """Evaluate Zhao auxiliary stresses and shifted displacement derivatives in batches."""
        self._zhao_auxiliary_fields = auxiliary_fields.prepare_zhao_auxiliary_fields(
            self._integration_contour_geometry,
            material=self.material,
        )
        displacement_gradient_x = (
            self._zhao_auxiliary_fields.displacement_gradient_x
        )
        return (
            self._zhao_auxiliary_fields.stress.copy(),
            displacement_gradient_x[:, 0].copy(),
            displacement_gradient_x[:, 1].copy(),
        )

    def _solve_interaction_integral(self, ki_aux: float, kii_aux: float) -> float:
        """Function that calculates the interaction integral as a line integration.
        More precisely, this function returns :math:`J^{1,2}` from formula 6.81 in Meinhard Kuna's book.

        Args:
            ki_aux: usually 1.0 if K_I should be calculated and else 0.0
            kii_aux: usually 1.0 if K_II should be calculated and else 0.0

        Returns:
            interaction integral value

        """
        self._execution.in_plane_samples = self._in_plane_samples
        return self._execution._solve_interaction_integral(ki_aux, kii_aux)

    def _solve_t_stress_interaction_integral(self) -> float:
        """Interaction path integral for the determination of T-stress according to Cardew et al. '85, Kfouri '86,
        Zhao et al. '01 and others. The analogous domain integral is used in ABACUS and ANSYS to determine T-stress.

        Returns:
            T stress interaction integral value

        """
        self._execution.in_plane_samples = self._in_plane_samples
        return self._execution._solve_t_stress_interaction_integral()

    def _williams_coeff_from_chen_integral(self, a_aux=0, b_aux=0, n=1) -> float:
        """Adapt a Bueckner-Chen Integral value to a Williams coefficient.

        Args:
            a_aux: Symmetric auxiliary-state coefficient ``c_m``.
            b_aux: Antisymmetric auxiliary-state coefficient ``d_m``.
            n: Williams coefficient term.

        Returns:
            Williams coefficient for term ``n`` using mm as the length unit.
        """
        self._execution.in_plane_samples = self._in_plane_samples
        return self._execution._williams_coefficient(
            symmetric_auxiliary_amplitude=a_aux,
            antisymmetric_auxiliary_amplitude=b_aux,
            term=n,
        )

    def _solve_chen_integral(self, n: int = -1, a_n: float = 0, b_n: float = 0) -> float:
        """Evaluate the Bueckner-Chen Integral through delegated kernels.

        Args:
            n: Williams term of the auxiliary eigenfield.
            a_n: Symmetric auxiliary eigenfield coefficient.
            b_n: Antisymmetric auxiliary eigenfield coefficient.

        Returns:
            Bueckner-Chen Integral value.

        Notes:
            The auxiliary-field formulation follows Y. Z. Chen, "New path
            independent integrals in linear elastic fracture mechanics" (1985).
        """
        self._execution.in_plane_samples = self._in_plane_samples
        return self._execution._solve_bueckner_chen_integral(
            auxiliary_term=n,
            symmetric_auxiliary_amplitude=a_n,
            antisymmetric_auxiliary_amplitude=b_n,
        )

    ######################################
    # FUNCTIONS FOR STRAIN RECALCULATION #
    ######################################

    def _compute_strains_xy(self, u_x, u_y, gap):
        self._in_plane_strains = mode_decomposition.reconstruct_in_plane_strains(
            u_x,
            u_y,
            self.x_coordinates,
            self.y_coordinates,
            gap=gap,
        )
        return (
            self._in_plane_strains.strain_x.copy(),
            self._in_plane_strains.strain_y.copy(),
            self._in_plane_strains.strain_xy.copy(),
        )

    def _compute_stress_strain_z(self, u_z, gap):
        self._mode_iii_fields = mode_decomposition.reconstruct_mode_iii_fields(
            u_z,
            self.x_coordinates,
            self.y_coordinates,
            material=self.material,
            gap=gap,
        )
        return (
            self._mode_iii_fields.out_of_plane_displacement_derivative_x.copy(),
            self._mode_iii_fields.out_of_plane_displacement_derivative_y.copy(),
            self._mode_iii_fields.shear_stress_xz.copy(),
            self._mode_iii_fields.shear_stress_yz.copy(),
        )

    #########################################################
    # FUNCTIONS FOR PREPARING DATA FOR J-MODE DECOMPOSITION #
    #########################################################

    def _prepare_mode_data(self, mode: str):
        """Prepare ``InputData`` for Mode I, II, or III J-Integral decomposition.

        Args:
            mode: Fracture mode as ``"I"``, ``"II"``, or ``"III"``.

        Returns:
            InputData instance populated for requested mode.
        """

        return mode_decomposition.prepare_mode_data(
            mode,
            self._regular_grid_displacements,
            material=self.material,
        )

    ################################
    # HELPER FUNCTIONS FOR MESHING #
    ################################

    def _build_path_geometry(self) -> None:
        """Precompute contour and shifted evaluation geometry used by every functional."""
        geometry = self._execution.geometry
        self._integration_contour_geometry = geometry
        self._integration_eval_points = geometry.evaluation_points.copy()
        self._integration_eval_points_pos = (
            geometry.positive_x_shifted_evaluation_points.copy()
        )
        self._integration_eval_points_neg = (
            geometry.negative_x_shifted_evaluation_points.copy()
        )
        self._integration_eval_points_all = geometry.combined_evaluation_points.copy()
        self._integration_eval_points_relative = geometry.relative_evaluation_points.copy()
        self._integration_eval_points_relative_pos = (
            geometry.relative_positive_x_shifted_evaluation_points.copy()
        )
        self._integration_eval_points_relative_neg = (
            geometry.relative_negative_x_shifted_evaluation_points.copy()
        )
        self._integration_eval_r = geometry.polar_radii.copy()
        self._integration_eval_phi = geometry.polar_angles.copy()
        self._path_elem_heights = geometry.segment_dy.copy()
        self._path_elem_sizes = geometry.segment_lengths.copy()
        self._path_normals = geometry.outward_unit_normals.copy()
        self._reference_eval_point = geometry.reference_point.copy()

    def _get_integration_point_interpolator(
            self,
            data: InputData,
            eval_points: np.ndarray | None = None,
            label: InterpolationTarget = InterpolationTarget.INTEGRATION_POINTS,
    ) -> ReusableLinearInterpolator:
        """Return cached interpolation geometry for one semantic evaluation layout."""
        if eval_points is None:
            eval_points = self._integration_eval_points
        return sampling._get_interpolator(
            data,
            eval_points,
            label,
            interpolator_cache=self._interpolator_cache,
        )

    def _get_masked_data(self, mask_tol: float | None = None) -> InputData:
        """Return the nodemap restricted to the configured contour band when requested."""
        tolerance = self.mask_tol if mask_tol is None else mask_tol
        return sampling.mask_contour_data(
            self.data,
            self._integration_contour_geometry.evaluation_points,
            tolerance,
        )

    def _interpolate_on_reference_point(self, values: np.ndarray) -> np.ndarray:
        """Interpolate one measured scalar field at the contour reference point."""
        self._reference_point_sample = sampling.interpolate_reference_value(
            self.data,
            values,
            self._reference_eval_point,
            interpolator_cache=self._interpolator_cache,
        )
        return self._reference_point_sample.copy()

    def _map_displacement_data_on_regular_grid(self, grid_points: int):
        self.grid_points = int(grid_points)
        self._regular_grid_displacements = sampling.sample_regular_grid_displacements(
            self.data,
            self._integration_contour_geometry.evaluation_points,
            self.grid_points,
            interpolator_cache=self._interpolator_cache,
        )
        grid = self._regular_grid_displacements
        self.x_coordinates = grid.x_coordinates.copy()
        self.y_coordinates = grid.y_coordinates.copy()
        self.x_mesh = grid.x_mesh.copy()
        self.y_mesh = grid.y_mesh.copy()
        self.disp_u_mesh = grid.displacement_x_mesh.copy()
        self.disp_v_mesh = grid.displacement_y_mesh.copy()
        self.disp_w_mesh = grid.displacement_z_mesh.copy()

    ###########################################
    # HELPER FUNCTIONS FOR DATA INTERPOLATION #
    ###########################################

    def _interpolate_on_integration_points(self):
        """Interpolates full field data onto the integration path coordinates.
        Further, calculates the interpolated results for shifted points for derivatives."""

        sampled_fields = sampling.sample_in_plane_fields(
            self.data,
            self._integration_contour_geometry,
            mask_tolerance=self.mask_tol,
            interpolator_cache=self._interpolator_cache,
        )
        self._adopt_in_plane_samples(sampled_fields)

    def _adopt_in_plane_samples(self, sampled_fields) -> None:
        """Project measured contour samples onto the facade's mutable attributes."""
        self.pos_shifted_np_int_points = self._integration_eval_points_pos[:, 0].copy()
        self.neg_shifted_np_int_points = self._integration_eval_points_neg[:, 0].copy()
        self._in_plane_samples = sampled_fields
        self._execution.in_plane_samples = sampled_fields
        shifted = self._in_plane_samples.shifted
        base = shifted.base
        (self.interpolated_eps_x, self.interpolated_eps_y, self.interpolated_eps_xy,
         self.interpolated_sig_x, self.interpolated_sig_y, self.interpolated_sig_xy,
         self.interpolated_disp_x, self.interpolated_disp_y) = tuple(
            base[:, index].copy() for index in range(8)
        )
        self.interpolated_disp_y_dx_positive = (
            shifted.positive_x_displacement_y.copy()
        )
        self.interpolated_disp_y_dx_negative = (
            shifted.negative_x_displacement_y.copy()
        )
        self.interpolated_disp_y_dx = (
            self._in_plane_samples.displacement_gradient_x[:, 1].copy()
        )

    def _interpolate_on_integration_points_z(self, mask_tol: float = None):
        """Interpolates full field data onto the integration path coordinates.
        Further, calculates the interpolated results for shifted points for derivatives.
        This method is used for the mode III decomposition of J integral.
        """

        self._mode_iii_samples = sampling.sample_mode_iii_fields(
            self.data,
            self._integration_eval_points,
            mask_tolerance=mask_tol,
            interpolator_cache=self._interpolator_cache,
        )
        self.interpolated_eps_xz = (
            self._mode_iii_samples.out_of_plane_displacement_derivative_x.copy()
        )
        self.interpolated_eps_yz = (
            self._mode_iii_samples.out_of_plane_displacement_derivative_y.copy()
        )
        self.interpolated_sigma_xz = self._mode_iii_samples.shear_stress_xz.copy()
        self.interpolated_sigma_yz = self._mode_iii_samples.shear_stress_yz.copy()

    ########
    # MISC #
    ########

    @staticmethod
    def _unit_m_to_mm(quantity_in_m, n=1):
        return williams_quantities.williams_coefficient_m_to_mm(
            quantity_in_m,
            term=n,
        )

    @staticmethod
    def _unit_mm_to_m(quantity_in_mm, n=1):
        return williams_quantities.williams_coefficient_mm_to_m(
            quantity_in_mm,
            term=n,
        )

    @staticmethod
    def ensure_defaults_bueckner_chen(options: IntegralProperties):
        """Normalize Bueckner-Chen Williams terms on integral properties.

        Args:
            options: Integral properties whose selected terms are normalized.

        Returns:
            None. ``options.bueckner_williams_terms`` is updated in place.
        """
        if options.bueckner_williams_terms is None:
            options.bueckner_williams_terms = DEFAULT_BUECKNER_CHEN_TERMS.copy()
        elif 1 not in options.bueckner_williams_terms:
            options.bueckner_williams_terms.append(1)
            logger.info('Bueckner-Williams terms should include 1. Term added.')
        if 0 in options.bueckner_williams_terms:
            options.bueckner_williams_terms.remove(0)
            logger.warning('Bueckner-Williams terms should not include 0. Term removed.')
        options.bueckner_williams_terms.sort()

    @staticmethod
    def ensure_defaults_buckner_chen(options: IntegralProperties):
        """Deprecated: use :meth:`ensure_defaults_bueckner_chen`.

        Args:
            options: Integral properties whose selected terms are normalized.

        Returns:
            None. :meth:`ensure_defaults_bueckner_chen` updates ``options``.
        """
        return LineIntegral.ensure_defaults_bueckner_chen(options)
