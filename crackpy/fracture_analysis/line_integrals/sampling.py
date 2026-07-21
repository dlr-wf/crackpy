"""Measured-field sampling interpolates crack-tip data onto Integration
Contours, reference points, and regular decomposition grids."""

from dataclasses import dataclass

import numpy as np

from crackpy.fracture_analysis._interpolation_cache import (
    InterpolationTarget,
    InterpolatorCache,
    ReusableLinearInterpolator,
)
from crackpy.fracture_analysis.line_integrals.contours import IntegrationContourGeometry
from crackpy.input.input_data import InputData, apply_mask


def _owned_read_only_array(values: np.ndarray) -> np.ndarray:
    array = np.array(values, copy=True)
    array.flags.writeable = False
    return array


########################
# SAMPLED FIELD VALUES #
########################


@dataclass(frozen=True, eq=False)
class ShiftedInPlaneSamples:
    """Store batched in-plane values at base and horizontally shifted contour points.

    Attributes:
        base: Contour samples with shape ``(n, 8)`` in
            ``[eps_x, eps_y, eps_xy, sigma_x, sigma_y, sigma_xy, u_x, u_y]``
            order.
        positive_x: Samples at points shifted along the crack-growth direction
            in the same ``(n, 8)`` field order.
        negative_x: Samples at points shifted opposite the crack-growth
            direction in the same ``(n, 8)`` field order.
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

    @property
    def positive_x_displacement_y(self) -> np.ndarray:
        """Return the owned positive-x-shifted y-displacement samples."""
        return self.positive_x[:, 7]

    @property
    def negative_x_displacement_y(self) -> np.ndarray:
        """Return the owned negative-x-shifted y-displacement samples."""
        return self.negative_x[:, 7]


@dataclass(frozen=True, eq=False)
class InPlaneContourSamples:
    """Store immutable measured in-plane fields sampled along one contour.

    Attributes:
        shifted: Base and horizontally shifted batched source fields.
        strain: Symmetric in-plane strain tensors in contour-point order.
        stress: Symmetric in-plane stress tensors in contour-point order.
        displacement: In-plane displacement vectors in contour-point order.
        displacement_gradient_x: Measured ``du_x/dx`` and central-difference
            ``du_y/dx`` values in contour-point order.
    """

    shifted: ShiftedInPlaneSamples
    strain: np.ndarray
    stress: np.ndarray
    displacement: np.ndarray
    displacement_gradient_x: np.ndarray

    def __post_init__(self) -> None:
        for field_name in (
            "strain",
            "stress",
            "displacement",
            "displacement_gradient_x",
        ):
            object.__setattr__(
                self,
                field_name,
                _owned_read_only_array(getattr(self, field_name)),
            )


@dataclass(frozen=True, eq=False)
class ModeIIIContourSamples:
    """Store immutable measured Mode III displacement derivatives and shear stresses.

    Attributes:
        out_of_plane_displacement_derivative_x: Sampled ``du_z/dx`` values.
        out_of_plane_displacement_derivative_y: Sampled ``du_z/dy`` values.
        shear_stress_xz: Sampled ``sigma_xz`` values in MPa.
        shear_stress_yz: Sampled ``sigma_yz`` values in MPa.
    """

    out_of_plane_displacement_derivative_x: np.ndarray
    out_of_plane_displacement_derivative_y: np.ndarray
    shear_stress_xz: np.ndarray
    shear_stress_yz: np.ndarray

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                field_name,
                _owned_read_only_array(getattr(self, field_name)),
            )


@dataclass(frozen=True, eq=False)
class RegularGridDisplacements:
    """Store immutable regular-grid coordinates and sampled displacement meshes.

    Attributes:
        x_coordinates: Uniform crack-tip-frame coordinates along prospective
            crack extension.
        y_coordinates: Uniform crack-tip-frame coordinates normal to the crack
            plane.
        x_mesh: X-coordinate mesh with ``xy`` indexing.
        y_mesh: Y-coordinate mesh with ``xy`` indexing.
        evaluation_points: C-order flattened ``(x, y)`` target coordinates.
        displacement_x_mesh: Sampled x-displacement mesh.
        displacement_y_mesh: Sampled y-displacement mesh.
        displacement_z_mesh: Sampled z-displacement mesh.
    """

    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    x_mesh: np.ndarray
    y_mesh: np.ndarray
    evaluation_points: np.ndarray
    displacement_x_mesh: np.ndarray
    displacement_y_mesh: np.ndarray
    displacement_z_mesh: np.ndarray

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                field_name,
                _owned_read_only_array(getattr(self, field_name)),
            )


################################
# MEASURED FIELD INTERPOLATION #
################################


def _get_interpolator(
    data: InputData,
    evaluation_points: np.ndarray,
    target: InterpolationTarget,
    *,
    interpolator_cache: InterpolatorCache,
) -> ReusableLinearInterpolator:
    """Return caller-cache interpolation geometry for a semantic target.

    Args:
        data: Source measured fields and coordinates.
        evaluation_points: Target coordinates with shape ``(n, 2)``.
        target: Semantic interpolation target.
        interpolator_cache: Explicit cache that owns reusable geometry.

    Returns:
        Reusable linear interpolation geometry for the requested target.
    """
    return interpolator_cache.get_interpolator(
        data.coor_x,
        data.coor_y,
        evaluation_points,
        target,
    )


def mask_contour_data(
    data: InputData,
    evaluation_points: np.ndarray,
    mask_tolerance: float | None,
) -> InputData:
    """Restrict measured data to the established rectangular contour band.

    Args:
        data: Source measured fields.
        evaluation_points: Contour midpoint coordinates with shape ``(n, 2)``.
        mask_tolerance: Band tolerance, or ``None`` to preserve the source object.

    Returns:
        Source data when unmasked, otherwise data restricted to the contour band.
    """
    if mask_tolerance is None:
        return data
    left = np.min(evaluation_points[:, 0])
    right = np.max(evaluation_points[:, 0])
    bottom = np.min(evaluation_points[:, 1])
    top = np.max(evaluation_points[:, 1])
    outer_contour_bounds = (
        (left - mask_tolerance <= data.coor_x)
        & (data.coor_x <= right + mask_tolerance)
        & (bottom - mask_tolerance <= data.coor_y)
        & (data.coor_y <= top + mask_tolerance)
    )
    inner_contour_bounds = (
        (left + mask_tolerance <= data.coor_x)
        & (data.coor_x <= right - mask_tolerance)
        & (bottom + mask_tolerance <= data.coor_y)
        & (data.coor_y <= top - mask_tolerance)
    )
    contour_band_mask = outer_contour_bounds & ~inner_contour_bounds
    return apply_mask(data, np.where(contour_band_mask))


def sample_in_plane_fields(
    data: InputData,
    geometry: IntegrationContourGeometry,
    *,
    mask_tolerance: float | None,
    interpolator_cache: InterpolatorCache,
) -> InPlaneContourSamples:
    """Sample measured in-plane fields at base and shifted contour points.

    Args:
        data: Source measured fields.
        geometry: Prepared contour sampling geometry.
        mask_tolerance: Contour-band tolerance, or ``None`` for unmasked data.
        interpolator_cache: Explicit cache that owns reusable geometry.

    Returns:
        Immutable sampled in-plane tensors, vectors, and shifted values.
    """
    sampled_data = mask_contour_data(
        data,
        geometry.evaluation_points,
        mask_tolerance,
    )
    interpolator = _get_interpolator(
        sampled_data,
        geometry.combined_evaluation_points,
        InterpolationTarget.INTEGRATION_POINTS_ALL,
        interpolator_cache=interpolator_cache,
    )
    # The eight columns are [eps_x, eps_y, eps_xy, sigma_x, sigma_y,
    # sigma_xy, u_x, u_y] in the contour-sampling field layout.
    in_plane_field_columns = np.c_[
        sampled_data.eps_x,
        sampled_data.eps_y,
        sampled_data.eps_xy,
        sampled_data.sig_x,
        sampled_data.sig_y,
        sampled_data.sig_xy,
        sampled_data.disp_x,
        sampled_data.disp_y,
    ]
    sampled = interpolator.interpolate(in_plane_field_columns)
    base, positive_x, negative_x = sampled.reshape(
        3,
        len(geometry.evaluation_points),
        8,
    )
    strain = np.empty((len(base), 2, 2), dtype=base.dtype)
    strain[:, 0, 0] = base[:, 0]
    strain[:, 1, 1] = base[:, 1]
    strain[:, 0, 1] = strain[:, 1, 0] = base[:, 2]
    stress = np.empty((len(base), 2, 2), dtype=base.dtype)
    stress[:, 0, 0] = base[:, 3]
    stress[:, 1, 1] = base[:, 4]
    stress[:, 0, 1] = stress[:, 1, 0] = base[:, 5]
    displacement = base[:, 6:8]
    displacement_y_gradient_x = (
        positive_x[:, 7] - negative_x[:, 7]
    ) / (2.0 * geometry.x_shift)
    # CrackPy uses measured eps_x as du_x/dx and a shifted central
    # difference only for du_y/dx.
    displacement_gradient_x = np.c_[base[:, 0], displacement_y_gradient_x]
    shifted_samples = ShiftedInPlaneSamples(base, positive_x, negative_x)
    in_plane_samples = InPlaneContourSamples(
        shifted=shifted_samples,
        strain=strain,
        stress=stress,
        displacement=displacement,
        displacement_gradient_x=displacement_gradient_x,
    )
    return in_plane_samples


def sample_mode_iii_fields(
    data: InputData,
    evaluation_points: np.ndarray,
    *,
    mask_tolerance: float | None,
    interpolator_cache: InterpolatorCache,
) -> ModeIIIContourSamples:
    """Sample measured Mode III fields along a contour.

    Args:
        data: Source measured fields.
        evaluation_points: Contour target coordinates.
        mask_tolerance: Explicit mask tolerance, or ``None`` for unmasked sampling.
        interpolator_cache: Explicit cache that owns reusable geometry.

    Returns:
        Immutable Mode III displacement derivatives and shear stresses.
    """
    sampled_data = mask_contour_data(data, evaluation_points, mask_tolerance)
    interpolator = _get_interpolator(
        sampled_data,
        evaluation_points,
        InterpolationTarget.INTEGRATION_POINTS,
        interpolator_cache=interpolator_cache,
    )
    # The four Mode III columns are [du_z/dx, du_z/dy, sigma_xz, sigma_yz].
    mode_iii_field_columns = np.c_[
        sampled_data.eps_xz,
        sampled_data.eps_yz,
        sampled_data.sigma_xz,
        sampled_data.sigma_yz,
    ]
    sampled = interpolator.interpolate(mode_iii_field_columns)
    out_of_plane_displacement_derivative_x = sampled[:, 0]
    out_of_plane_displacement_derivative_y = sampled[:, 1]
    shear_stress_xz = sampled[:, 2]
    shear_stress_yz = sampled[:, 3]
    mode_iii_samples = ModeIIIContourSamples(
        out_of_plane_displacement_derivative_x=(
            out_of_plane_displacement_derivative_x
        ),
        out_of_plane_displacement_derivative_y=(
            out_of_plane_displacement_derivative_y
        ),
        shear_stress_xz=shear_stress_xz,
        shear_stress_yz=shear_stress_yz,
    )
    return mode_iii_samples


def interpolate_reference_value(
    data: InputData,
    values: np.ndarray,
    reference_point: np.ndarray,
    *,
    interpolator_cache: InterpolatorCache,
) -> np.ndarray:
    """Interpolate one measured scalar field at the contour reference point.

    Args:
        data: Source measured fields.
        values: Scalar values at the source coordinates.
        reference_point: Single reference target with shape ``(1, 2)``.
        interpolator_cache: Explicit cache that owns reusable geometry.

    Returns:
        An owned, non-writeable zero-dimensional NumPy result.
    """
    interpolator = _get_interpolator(
        data,
        reference_point,
        InterpolationTarget.REFERENCE_POINT,
        interpolator_cache=interpolator_cache,
    )
    interpolated_reference_values = interpolator.interpolate(values)
    reference_value = interpolated_reference_values.squeeze()
    owned_reference_value = _owned_read_only_array(reference_value)
    return owned_reference_value


def sample_regular_grid_displacements(
    data: InputData,
    evaluation_points: np.ndarray,
    grid_point_count: int,
    *,
    interpolator_cache: InterpolatorCache,
) -> RegularGridDisplacements:
    """Sample three displacement components on a symmetric crack-tip grid.

    Args:
        data: Source measured displacement fields.
        evaluation_points: Crack-tip-frame contour midpoint coordinates that
            define the grid extents.
        grid_point_count: Equal number of target points on each coordinate axis.
        interpolator_cache: Explicit cache that owns reusable geometry.

    Returns:
        Zero-centered coordinate axes spanning 120 percent of the largest
        absolute contour coordinate on each axis, their mesh layout, and the
        three sampled displacement meshes.
    """
    grid_size_x = max(
        abs(np.min(evaluation_points[:, 0])),
        abs(np.max(evaluation_points[:, 0])),
    ) * 1.2
    grid_size_y = max(
        abs(np.min(evaluation_points[:, 1])),
        abs(np.max(evaluation_points[:, 1])),
    ) * 1.2
    point_count = int(grid_point_count)
    x_coordinates = np.linspace(-grid_size_x, grid_size_x, point_count, endpoint=True)
    y_coordinates = np.linspace(-grid_size_y, grid_size_y, point_count, endpoint=True)
    x_mesh, y_mesh = np.meshgrid(x_coordinates, y_coordinates, indexing="xy")
    regular_grid_evaluation_points = np.c_[x_mesh.ravel(), y_mesh.ravel()]
    interpolator = _get_interpolator(
        data,
        regular_grid_evaluation_points,
        InterpolationTarget.REGULAR_GRID,
        interpolator_cache=interpolator_cache,
    )
    displacement_field_columns = np.c_[data.disp_x, data.disp_y, data.disp_z]
    displacements = interpolator.interpolate(displacement_field_columns).reshape(
        x_mesh.shape + (3,)
    )
    displacement_x_mesh = displacements[:, :, 0]
    displacement_y_mesh = displacements[:, :, 1]
    displacement_z_mesh = displacements[:, :, 2]
    regular_grid_displacements = RegularGridDisplacements(
        x_coordinates=x_coordinates,
        y_coordinates=y_coordinates,
        x_mesh=x_mesh,
        y_mesh=y_mesh,
        evaluation_points=regular_grid_evaluation_points,
        displacement_x_mesh=displacement_x_mesh,
        displacement_y_mesh=displacement_y_mesh,
        displacement_z_mesh=displacement_z_mesh,
    )
    return regular_grid_displacements
