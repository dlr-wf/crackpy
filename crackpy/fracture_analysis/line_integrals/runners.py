"""Line-integral runners coordinate field preparation, functional evaluation,
contour quadrature, and Contour-Wise Result construction for individual
Integration Contours.

They provide the shared calculation path used by FractureAnalysis and the
LineIntegral compatibility facade.
"""

import numpy as np

from crackpy.fracture_analysis._interpolation_cache import InterpolatorCache
from crackpy.fracture_analysis.crack_tip import eigenfunction
from crackpy.fracture_analysis.crack_tip_fields.williams import (
    quantities as williams_quantities,
)
from crackpy.fracture_analysis.crack_tip_fields.williams.coefficients import (
    WilliamsInPlaneCoefficients,
)
from crackpy.fracture_analysis.functionals.bueckner_chen import (
    bueckner_chen_integral_terms,
    williams_coefficient_from_bueckner_chen_integral,
)
from crackpy.fracture_analysis.functionals.interaction_integral import (
    in_plane_sif_from_interaction_integral,
    interaction_integral_terms,
    t_stress_from_interaction_integral,
    t_stress_interaction_integral_terms,
)
from crackpy.fracture_analysis.functionals.j_integral import (
    in_plane_energy_equivalent_sif_from_j_integral,
    in_plane_j_integral_terms,
    in_plane_sif_magnitude_from_j_integral,
    mode_iii_j_integral_terms,
    mode_iii_sif_magnitude_from_j_integral,
)
from crackpy.fracture_analysis.functionals.stress_difference import (
    t_stress_from_stress_difference,
)
from crackpy.fracture_analysis.line_integrals import (
    auxiliary_fields,
    mode_decomposition,
    sampling,
)
from crackpy.fracture_analysis.line_integrals.contours import (
    IntegrationContour,
    prepare_integration_contour_geometry,
)
from crackpy.fracture_analysis.line_integrals.quadrature import (
    evaluate_contour_integral,
)
from crackpy.fracture_analysis.line_integrals.results import (
    ContourWiseLineIntegralResult,
    IntegrationContourResultGeometry,
    LineIntegralQuantities,
)
from crackpy.input.input_data import InputData
from crackpy.structure_elements.material import Material


class _LineIntegralExecution:
    """Evaluate CrackPy's supported line-integral quantities for one Integration Contour.

    The execution prepares the required fields, evaluates fracture-mechanics
    functionals, applies contour quadrature, and constructs one completed
    Contour-Wise Result.
    """

    def __init__(
        self,
        contour: IntegrationContour,
        data: InputData,
        material: Material,
        mask_tolerance: float | None,
        requested_bueckner_williams_terms: list[int] | None,
        interpolator_cache: InterpolatorCache,
    ) -> None:
        self.contour = contour
        self.data = data
        self.material = material
        self.mask_tolerance = mask_tolerance
        self.requested_bueckner_williams_terms = requested_bueckner_williams_terms
        self.interpolator_cache = interpolator_cache
        self.geometry = prepare_integration_contour_geometry(
            contour,
            x_shift=contour.tick_size,
        )
        self.in_plane_samples = self.sample_in_plane_fields(data)

    ###########################
    # MEASURED FIELD SAMPLING #
    ###########################

    def sample_in_plane_fields(self, data: InputData):
        """Sample in-plane measured fields for this execution's contour."""
        return sampling.sample_in_plane_fields(
            data,
            self.geometry,
            mask_tolerance=self.mask_tolerance,
            interpolator_cache=self.interpolator_cache,
        )

    def sample_mode_iii_fields(
        self,
        data: InputData,
        *,
        mask_tolerance: float | None = None,
    ):
        """Sample Mode III measured fields for this execution's contour."""
        return sampling.sample_mode_iii_fields(
            data,
            self.geometry.evaluation_points,
            mask_tolerance=mask_tolerance,
            interpolator_cache=self.interpolator_cache,
        )

    ########################
    # TECHNIQUE EVALUATION #
    ########################

    def evaluate_j_integral(self) -> tuple[float, float]:
        """Evaluate total J and its energy-equivalent Stress Intensity Factor."""
        j_integral = self._solve_j_integral(self.in_plane_samples)
        sif_k_j = in_plane_energy_equivalent_sif_from_j_integral(
            j_integral,
            youngs_modulus=self.material.E,
        )
        return j_integral, sif_k_j

    def evaluate_j_decomposition(self) -> tuple[float, float, float, float, float, float]:
        """Evaluate modal J-Integrals and Stress Intensity Factor magnitudes."""
        regular_grid = sampling.sample_regular_grid_displacements(
            self.data,
            self.geometry.evaluation_points,
            200,
            interpolator_cache=self.interpolator_cache,
        )

        mode_i_data = mode_decomposition.prepare_mode_data(
            "I",
            regular_grid,
            material=self.material,
        )
        mode_i_samples = self.sample_in_plane_fields(mode_i_data)
        mode_i_j_integral = self._solve_j_integral(mode_i_samples)
        mode_i_sif = in_plane_sif_magnitude_from_j_integral(
            mode_i_j_integral,
            youngs_modulus=self.material.E,
        )

        mode_ii_data = mode_decomposition.prepare_mode_data(
            "II",
            regular_grid,
            material=self.material,
        )
        mode_ii_samples = self.sample_in_plane_fields(mode_ii_data)
        mode_ii_j_integral = self._solve_j_integral(mode_ii_samples)
        mode_ii_sif = in_plane_sif_magnitude_from_j_integral(
            mode_ii_j_integral,
            youngs_modulus=self.material.E,
        )

        mode_iii_data = mode_decomposition.prepare_mode_data(
            "III",
            regular_grid,
            material=self.material,
        )
        mode_iii_samples = self.sample_mode_iii_fields(mode_iii_data)
        mode_iii_j_integral = self._solve_mode_iii_j_integral(mode_iii_samples)
        mode_iii_sif = mode_iii_sif_magnitude_from_j_integral(
            mode_iii_j_integral,
            youngs_modulus=self.material.E,
            poisson_ratio=self.material.nu_xy,
        )
        return (
            mode_i_j_integral,
            mode_ii_j_integral,
            mode_iii_j_integral,
            mode_i_sif,
            mode_ii_sif,
            mode_iii_sif,
        )

    def evaluate_interaction_sifs(self) -> tuple[float, float]:
        """Evaluate signed Mode I and II Stress Intensity Factors by interaction."""
        unit_auxiliary_sif = williams_quantities.williams_coefficient_m_to_mm(1.0)
        mode_i_interaction = self._solve_interaction_integral(
            unit_auxiliary_sif,
            0.0,
        )
        sif_k_i = in_plane_sif_from_interaction_integral(
            mode_i_interaction,
            youngs_modulus=self.material.E,
            auxiliary_sif=unit_auxiliary_sif,
        )
        mode_ii_interaction = self._solve_interaction_integral(
            0.0,
            unit_auxiliary_sif,
        )
        sif_k_ii = in_plane_sif_from_interaction_integral(
            mode_ii_interaction,
            youngs_modulus=self.material.E,
            auxiliary_sif=unit_auxiliary_sif,
        )
        return sif_k_i, sif_k_ii

    def evaluate_interaction_t_stress(self) -> float:
        """Evaluate T-stress with Zhao's auxiliary interaction integral."""
        t_stress_integral = self._solve_t_stress_interaction_integral()
        reference_out_of_plane_strain = None
        if not self.material.plane_strain:
            # For plane stress, the Zhao interaction-integral mapping uses
            # epsilon_zz = -nu * (epsilon_xx + epsilon_yy).
            plane_stress_out_of_plane_strain = -self.material.nu_xy * (
                self.data.eps_x + self.data.eps_y
            )
            reference_out_of_plane_strain = sampling.interpolate_reference_value(
                self.data,
                plane_stress_out_of_plane_strain,
                self.geometry.reference_point,
                interpolator_cache=self.interpolator_cache,
            ).item()
        mapped_t_stress = t_stress_from_interaction_integral(
            t_stress_integral,
            youngs_modulus=self.material.E,
            poisson_ratio=self.material.nu_xy,
            plane_strain=self.material.plane_strain,
            reference_out_of_plane_strain=reference_out_of_plane_strain,
        )
        return mapped_t_stress

    def evaluate_stress_difference_t_stress(self) -> float:
        """Evaluate the Stress-Difference Method at the contour's right
        intercept on the crack-extension line."""
        pointwise_t_stress = t_stress_from_stress_difference(
            self.data.sig_x,
            self.data.sig_y,
        )
        sampled_t_stress = sampling.interpolate_reference_value(
            self.data,
            pointwise_t_stress,
            self.geometry.reference_point,
            interpolator_cache=self.interpolator_cache,
        )
        sampled_t_stress_value = sampled_t_stress.item()
        return sampled_t_stress_value

    def evaluate_bueckner_chen(
        self,
        terms: list[int],
    ) -> tuple[WilliamsInPlaneCoefficients, float]:
        """Evaluate requested Williams coefficients and independent Chen T-stress."""
        a_n_values = []
        b_n_values = []
        second_order_symmetric_coefficient = None
        for term in terms:
            symmetric_coefficient = self._williams_coefficient(
                symmetric_auxiliary_amplitude=1,
                term=term,
            )
            antisymmetric_coefficient = self._williams_coefficient(
                antisymmetric_auxiliary_amplitude=1,
                term=term,
            )
            a_n_values.append(symmetric_coefficient)
            b_n_values.append(antisymmetric_coefficient)
            if term == 2:
                second_order_symmetric_coefficient = symmetric_coefficient
        coefficients = WilliamsInPlaneCoefficients(
            terms=tuple(terms),
            a_n=tuple(a_n_values),
            b_n=tuple(b_n_values),
        )
        if second_order_symmetric_coefficient is None:
            second_order_symmetric_coefficient = self._williams_coefficient(
                symmetric_auxiliary_amplitude=1,
                term=2,
            )
        t_stress_chen = williams_quantities.t_stress_from_williams_coefficient(
            second_order_symmetric_coefficient=second_order_symmetric_coefficient,
        )
        return coefficients, t_stress_chen

    ###############################
    # COMBINED CONTOUR EVALUATION #
    ###############################

    def evaluate_all(self) -> ContourWiseLineIntegralResult:
        """Evaluate all configured techniques and return their Contour-Wise Result."""
        j_integral, sif_k_j = self.evaluate_j_integral()
        (
            decomp_j_integral_i,
            decomp_j_integral_ii,
            decomp_j_integral_iii,
            decomp_j_integral_k_i,
            decomp_j_integral_k_ii,
            decomp_j_integral_k_iii,
        ) = self.evaluate_j_decomposition()
        sif_k_i, sif_k_ii = self.evaluate_interaction_sifs()
        t_stress_int = self.evaluate_interaction_t_stress()
        t_stress_sdm = self.evaluate_stress_difference_t_stress()
        coefficients = None
        t_stress_chen = None
        if self.requested_bueckner_williams_terms is not None:
            coefficients, t_stress_chen = self.evaluate_bueckner_chen(
                self.requested_bueckner_williams_terms,
            )
        quantities = LineIntegralQuantities(
            j_integral=j_integral,
            sif_k_j=sif_k_j,
            sif_k_i=sif_k_i,
            sif_k_ii=sif_k_ii,
            t_stress_chen=t_stress_chen,
            t_stress_sdm=t_stress_sdm,
            t_stress_int=t_stress_int,
            decomp_j_integral_i=decomp_j_integral_i,
            decomp_j_integral_ii=decomp_j_integral_ii,
            decomp_j_integral_iii=decomp_j_integral_iii,
            decomp_j_integral_k_i=decomp_j_integral_k_i,
            decomp_j_integral_k_ii=decomp_j_integral_k_ii,
            decomp_j_integral_k_iii=decomp_j_integral_k_iii,
        )
        result_geometry = self._result_geometry()
        result = ContourWiseLineIntegralResult(
            geometry=result_geometry,
            quantities=quantities,
            williams_coefficients=coefficients,
        )
        return result

    ################################
    # FUNCTIONAL KERNEL EVALUATION #
    ################################

    def _solve_j_integral(self, sampled_fields) -> float:
        integrand_terms = in_plane_j_integral_terms(
            sampled_fields.stress,
            sampled_fields.strain,
            sampled_fields.displacement_gradient_x,
            self.geometry.outward_unit_normals,
        )
        j_integral = self._integrate(integrand_terms)
        return j_integral

    def _solve_mode_iii_j_integral(self, sampled_fields) -> float:
        integrand_terms = mode_iii_j_integral_terms(
            sampled_fields.out_of_plane_displacement_derivative_x,
            sampled_fields.out_of_plane_displacement_derivative_y,
            sampled_fields.shear_stress_xz,
            sampled_fields.shear_stress_yz,
            self.geometry.outward_unit_normals,
        )
        mode_iii_j_integral = self._integrate(integrand_terms)
        return mode_iii_j_integral

    def _solve_interaction_integral(
        self,
        auxiliary_sif_i: float,
        auxiliary_sif_ii: float,
    ) -> float:
        auxiliary_field_values = auxiliary_fields.prepare_lefm_auxiliary_fields(
            auxiliary_sif_i,
            auxiliary_sif_ii,
            self.geometry,
            material=self.material,
        )
        auxiliary_displacement_gradient_x = np.c_[
            auxiliary_field_values.strain[:, 0, 0],
            auxiliary_field_values.displacement_y_gradient_x,
        ]
        integrand_terms = interaction_integral_terms(
            self.in_plane_samples.stress,
            self.in_plane_samples.displacement_gradient_x,
            auxiliary_field_values.stress,
            auxiliary_field_values.strain,
            auxiliary_displacement_gradient_x,
            self.geometry.outward_unit_normals,
        )
        interaction_integral = self._integrate(integrand_terms)
        return interaction_integral

    def _solve_t_stress_interaction_integral(self) -> float:
        auxiliary_field_values = auxiliary_fields.prepare_zhao_auxiliary_fields(
            self.geometry,
            material=self.material,
        )
        integrand_terms = t_stress_interaction_integral_terms(
            self.in_plane_samples.stress,
            self.in_plane_samples.strain,
            self.in_plane_samples.displacement_gradient_x,
            auxiliary_field_values.stress,
            auxiliary_field_values.displacement_gradient_x,
            self.geometry.outward_unit_normals,
        )
        t_stress_interaction_integral = self._integrate(integrand_terms)
        return t_stress_interaction_integral

    def _williams_coefficient(
        self,
        *,
        symmetric_auxiliary_amplitude: float = 0,
        antisymmetric_auxiliary_amplitude: float = 0,
        term: int,
    ) -> float:
        if symmetric_auxiliary_amplitude != 0 and antisymmetric_auxiliary_amplitude != 0:
            raise ValueError("Either a_aux or b_aux has to be zero!")
        # Bueckner-Chen coefficient term n uses the complementary auxiliary
        # Williams eigenfield order m = -n.
        integral_value = self._solve_bueckner_chen_integral(
            auxiliary_term=-term,
            symmetric_auxiliary_amplitude=symmetric_auxiliary_amplitude,
            antisymmetric_auxiliary_amplitude=antisymmetric_auxiliary_amplitude,
        )
        williams_coefficient = williams_coefficient_from_bueckner_chen_integral(
            integral_value,
            shear_modulus=self.material.G,
            kappa=self.material.kappa,
            symmetric_auxiliary_amplitude=symmetric_auxiliary_amplitude,
            antisymmetric_auxiliary_amplitude=antisymmetric_auxiliary_amplitude,
            term=term,
        )
        return williams_coefficient

    def _solve_bueckner_chen_integral(
        self,
        *,
        auxiliary_term: int,
        symmetric_auxiliary_amplitude: float,
        antisymmetric_auxiliary_amplitude: float,
    ) -> float:
        sigma_x, sigma_y, sigma_xy, displacement_x, displacement_y = eigenfunction(
            auxiliary_term,
            symmetric_auxiliary_amplitude,
            antisymmetric_auxiliary_amplitude,
            self.geometry.polar_radii,
            self.geometry.polar_angles,
            self.material,
        )
        auxiliary_stress = np.moveaxis(
            np.asarray([[sigma_x, sigma_xy], [sigma_xy, sigma_y]]),
            -1,
            0,
        )
        auxiliary_displacement = np.c_[displacement_x, displacement_y]
        integrand_terms = bueckner_chen_integral_terms(
            self.in_plane_samples.stress,
            self.in_plane_samples.displacement,
            auxiliary_stress,
            auxiliary_displacement,
            self.geometry.outward_unit_normals,
        )
        bueckner_chen_integral = self._integrate(integrand_terms)
        return bueckner_chen_integral

    ##################################
    # QUADRATURE AND RESULT GEOMETRY #
    ##################################

    def _integrate(self, integrand_terms) -> float:
        contour_integral = evaluate_contour_integral(
            integrand_terms,
            segment_dy=self.geometry.segment_dy,
            segment_lengths=self.geometry.segment_lengths,
        )
        return contour_integral

    def _result_geometry(self) -> IntegrationContourResultGeometry:
        relative_nodes = self.contour.nodes - np.asarray(self.contour.origin)
        size_left = float(np.min(relative_nodes[:, 0]))
        size_right = float(np.max(relative_nodes[:, 0]))
        size_bottom = float(np.min(relative_nodes[:, 1]))
        size_top = float(np.max(relative_nodes[:, 1]))
        result_geometry = IntegrationContourResultGeometry(
            size_left=size_left,
            size_right=size_right,
            size_bottom=size_bottom,
            size_top=size_top,
            integration_points=self.geometry.evaluation_points,
            number_of_nodes=self.contour.number_of_nodes,
            tick_size=self.contour.tick_size,
        )
        return result_geometry
