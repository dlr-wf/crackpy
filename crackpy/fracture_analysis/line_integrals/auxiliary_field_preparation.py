"""Auxiliary-field preparation evaluates LEFM and Zhao reference fields at
the contour coordinates required by line-integral functionals."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from crackpy.fracture_analysis.crack_tip_fields.auxiliary import (
    get_crack_nearfield,
    get_zhao_solutions,
)
from crackpy.fracture_analysis.line_integrals.contours import IntegrationContourGeometry
from crackpy.structure_elements.material import Material


def _owned_read_only_array(values: np.ndarray) -> np.ndarray:
    array = np.array(values, copy=True)
    array.flags.writeable = False
    return array


############################
# AUXILIARY FIELD PAYLOADS #
############################


@dataclass(frozen=True, eq=False)
class ShiftedAuxiliaryFields:
    """Store immutable analytical fields at base and horizontally shifted points.

    Attributes:
        base: Analytical values at contour-segment midpoints in the crack-tip
            Cartesian frame.
        positive_x: Values at the same points shifted by ``+x_shift`` along
            the crack-growth direction.
        negative_x: Values at the same points shifted by ``-x_shift`` along
            the crack-growth direction.
    """

    base: np.ndarray
    positive_x: np.ndarray
    negative_x: np.ndarray

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                field_name,
                _owned_read_only_array(getattr(self, field_name)),
            )


@dataclass(frozen=True, eq=False)
class AuxiliaryInPlaneFields:
    """Store immutable LEFM auxiliary tensors and y-displacement derivative.

    Attributes:
        stress: Auxiliary symmetric in-plane stress tensors with shape
            ``(n, 2, 2)`` in MPa.
        strain: Auxiliary symmetric in-plane strain tensors with shape
            ``(n, 2, 2)``.
        displacement_y_gradient_x: Central-difference ``du_y/dx`` values with
            shape ``(n,)``.
    """

    stress: np.ndarray
    strain: np.ndarray
    displacement_y_gradient_x: np.ndarray

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                field_name,
                _owned_read_only_array(getattr(self, field_name)),
            )


@dataclass(frozen=True, eq=False)
class ZhaoAuxiliaryFields:
    """Store immutable Zhao auxiliary stress and displacement derivatives.

    Attributes:
        stress: Auxiliary symmetric in-plane stress tensors with shape
            ``(n, 2, 2)``.
        displacement_gradient_x: Central-difference ``du_x/dx`` and
            ``du_y/dx`` values with shape ``(n, 2)``.
    """

    stress: np.ndarray
    displacement_gradient_x: np.ndarray

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                field_name,
                _owned_read_only_array(getattr(self, field_name)),
            )


###################################
# SHIFTED COORDINATE EVALUATION #
###################################


def evaluate_shifted_auxiliary_fields(
    evaluator: Callable[[np.ndarray], np.ndarray],
    geometry: IntegrationContourGeometry,
) -> ShiftedAuxiliaryFields:
    """Evaluate one analytical field on aligned base and shifted contour points.

    Args:
        evaluator: Batched analytical evaluator accepting relative Cartesian points.
        geometry: Contour-segment midpoints in the crack-tip Cartesian frame
            together with points shifted by ``±geometry.x_shift`` along the
            crack-growth direction.

    Returns:
        Immutable values in base, positive-x, and negative-x order.
    """
    base_fields = evaluator(geometry.relative_evaluation_points)
    positive_x_shifted_fields = evaluator(
        geometry.relative_positive_x_shifted_evaluation_points
    )
    negative_x_shifted_fields = evaluator(
        geometry.relative_negative_x_shifted_evaluation_points
    )
    shifted_fields = ShiftedAuxiliaryFields(
        base=base_fields,
        positive_x=positive_x_shifted_fields,
        negative_x=negative_x_shifted_fields,
    )
    return shifted_fields


def _make_polar(relative_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = relative_points[:, 0]
    y = relative_points[:, 1]
    polar_radii = np.sqrt(x**2.0 + y**2.0)
    polar_angles = np.arctan2(y, x)
    return polar_radii, polar_angles


########################
# LEFM AUXILIARY FIELD #
########################


def prepare_lefm_auxiliary_fields(
    auxiliary_sif_i: float,
    auxiliary_sif_ii: float,
    geometry: IntegrationContourGeometry,
    *,
    material: Material,
) -> AuxiliaryInPlaneFields:
    """Batch the established LEFM auxiliary fields and central derivative.

    Args:
        auxiliary_sif_i: Mode I auxiliary Stress Intensity Factor in MPa sqrt(mm).
        auxiliary_sif_ii: Mode II auxiliary Stress Intensity Factor in MPa sqrt(mm).
        geometry: Contour-segment midpoints in the crack-tip Cartesian frame
            together with points shifted by ``±geometry.x_shift`` along the
            crack-growth direction.
        material: Isotropic material used by the crack-nearfield evaluator.

    Returns:
        Auxiliary stress and strain tensors with shape ``(n, 2, 2)`` and
        central-difference ``du_y/dx`` values with shape ``(n,)``.

    Notes:
        Crack-nearfield formula evaluation is delegated to
        :func:`crackpy.fracture_analysis.crack_tip_fields.auxiliary.get_crack_nearfield`, which
        implements Sladek et al. (1997), equations 3--4.
        DOI: https://doi.org/10.1016/S0167-8442(97)00013-X.
        Citation key: ``sladek_et_al_1997_contour_integrals``.
    """

    def evaluate_displacements(relative_points: np.ndarray) -> np.ndarray:
        radius, angle = _make_polar(relative_points)
        displacement = np.asarray(
            get_crack_nearfield(
                auxiliary_sif_i,
                auxiliary_sif_ii,
                radius,
                angle,
                material,
            )[2]
        )
        return displacement

    radius, angle = _make_polar(geometry.relative_evaluation_points)
    # The LEFM auxiliary-field result is ordered as stress, strain, displacement.
    # The stress and strain arrays define the contour tensors.
    stress, strain, displacement = get_crack_nearfield(
        auxiliary_sif_i,
        auxiliary_sif_ii,
        radius,
        angle,
        material,
    )
    base_displacement = np.asarray(displacement)
    positive_x_shifted_displacement = evaluate_displacements(
        geometry.relative_positive_x_shifted_evaluation_points
    )
    negative_x_shifted_displacement = evaluate_displacements(
        geometry.relative_negative_x_shifted_evaluation_points
    )
    shifted = ShiftedAuxiliaryFields(
        base=base_displacement,
        positive_x=positive_x_shifted_displacement,
        negative_x=negative_x_shifted_displacement,
    )
    displacement_derivative_x = (
        shifted.positive_x - shifted.negative_x
    ) / (2 * geometry.x_shift)
    auxiliary_stress = np.moveaxis(stress, -1, 0)
    auxiliary_strain = np.moveaxis(strain, -1, 0)
    displacement_y_gradient_x = displacement_derivative_x[1]
    auxiliary_fields = AuxiliaryInPlaneFields(
        stress=auxiliary_stress,
        strain=auxiliary_strain,
        displacement_y_gradient_x=displacement_y_gradient_x,
    )
    return auxiliary_fields


########################
# ZHAO AUXILIARY FIELD #
########################


def prepare_zhao_auxiliary_fields(
    geometry: IntegrationContourGeometry,
    *,
    material: Material,
) -> ZhaoAuxiliaryFields:
    """Batch the established Zhao auxiliary stresses and central derivatives.

    Args:
        geometry: Contour-segment midpoints in the crack-tip Cartesian frame
            together with points shifted by ``±geometry.x_shift`` along the
            crack-growth direction.
        material: Isotropic material used by the Zhao evaluator.

    Returns:
        Auxiliary stress tensors with shape ``(n, 2, 2)`` and
        central-difference ``du_x/dx`` and ``du_y/dx`` values with shape
        ``(n, 2)``.

    Notes:
        Auxiliary formula evaluation is delegated to
        :func:`crackpy.fracture_analysis.crack_tip_fields.auxiliary.get_zhao_solutions`, which
        implements Zhao et al. (2001), equations 4a--4b.
        DOI: https://doi.org/10.1023/A:1011016720630.
        Citation key: ``zhao_et_al_2001_corner_cracks``.
    """

    def evaluate(relative_points: np.ndarray) -> np.ndarray:
        radius, angle = _make_polar(relative_points)
        zhao_fields = np.asarray(get_zhao_solutions(radius, angle, material))
        return zhao_fields

    shifted = evaluate_shifted_auxiliary_fields(
        evaluate,
        geometry,
    )
    # Zhao analytical rows are [sigma_x, sigma_y, sigma_xy, u_x, u_y];
    # stresses and shifted displacement rows feed separate functional terms.
    sigma_x, sigma_y, sigma_xy, _, _ = shifted.base
    displacement_derivative_x = (
        shifted.positive_x - shifted.negative_x
    ) / (2 * geometry.x_shift)
    stress = np.moveaxis(
        np.asarray([[sigma_x, sigma_xy], [sigma_xy, sigma_y]]),
        -1,
        0,
    )
    displacement_gradient_x = np.moveaxis(displacement_derivative_x[3:5], 0, 1)
    zhao_auxiliary_fields = ZhaoAuxiliaryFields(
        stress=stress,
        displacement_gradient_x=displacement_gradient_x,
    )
    return zhao_auxiliary_fields
