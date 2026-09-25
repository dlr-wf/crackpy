"""Mode decomposition reconstructs Mode I, II, and III fields from regular-grid
displacements using crack-plane symmetry and displacement gradients."""

from dataclasses import dataclass

import numpy as np

from crackpy.fracture_analysis.line_integrals.sampling import RegularGridDisplacements
from crackpy.input.input_data import InputData
from crackpy.structure_elements.material import Material


def _owned_read_only_array(values: np.ndarray) -> np.ndarray:
    array = np.array(values, copy=True)
    array.flags.writeable = False
    return array


################################
# RECONSTRUCTED FIELD PAYLOADS #
################################


@dataclass(frozen=True, eq=False)
class InPlaneStrains:
    """Store immutable reconstructed in-plane strain components.

    Attributes:
        strain_x: Reconstructed x normal strain.
        strain_y: Reconstructed y normal strain.
        strain_xy: Reconstructed tensorial in-plane shear strain.
    """

    strain_x: np.ndarray
    strain_y: np.ndarray
    strain_xy: np.ndarray

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                field_name,
                _owned_read_only_array(getattr(self, field_name)),
            )


@dataclass(frozen=True, eq=False)
class ModeIIIFields:
    """Store reconstructed Mode III displacement derivatives and shear stresses.

    Attributes:
        out_of_plane_displacement_derivative_x: Reconstructed ``du_z/dx``.
        out_of_plane_displacement_derivative_y: Reconstructed ``du_z/dy``.
        shear_stress_xz: Reconstructed ``sigma_xz`` in MPa.
        shear_stress_yz: Reconstructed ``sigma_yz`` in MPa.
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


################################
# CRACK-WAKE FIELD GRADIENTS #
################################


def _reconstruct_gradient_with_crack_wake_gap(
    values: np.ndarray,
    spacing: float,
    *,
    axis: int,
    gap: int,
) -> np.ndarray:
    steps = values.shape[0]
    midpoint = int(steps / 2)
    gradient = np.zeros_like(values)
    # Upper and lower left regions are differentiated separately so gradients do
    # not cross the crack-wake displacement discontinuity. Rows selected by gap
    # retain the established zero initialization.
    upper_left = np.s_[: midpoint - gap, :midpoint]
    lower_left = np.s_[midpoint + gap :, :midpoint]
    right = np.s_[:, midpoint:]
    for region in (upper_left, lower_left, right):
        gradient[region] = np.gradient(values[region], spacing, axis=axis)
    return gradient


def reconstruct_in_plane_strains(
    displacement_x: np.ndarray,
    displacement_y: np.ndarray,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    *,
    gap: int,
) -> InPlaneStrains:
    """Reconstruct in-plane strains with the established crack-wake gap.

    Args:
        displacement_x: Regular-grid x displacement mesh.
        displacement_y: Regular-grid y displacement mesh.
        x_coordinates: Uniform crack-tip-frame coordinates along prospective
            crack extension.
        y_coordinates: Uniform crack-tip-frame coordinates normal to the crack
            plane.
        gap: Number of rows omitted on either side of the left crack wake.

    Returns:
        Immutable reconstructed in-plane strain components.

    Notes:
        Molteno and Becker (2015), equation 6, derives the in-plane strain
        components from each decomposed displacement field.
        DOI: https://doi.org/10.1111/str.12166.
        Citation key: ``molteno_becker_2015_j_integral_decomposition``.
    """
    x_spacing = x_coordinates[1] - x_coordinates[0]
    y_spacing = y_coordinates[1] - y_coordinates[0]
    strain_x = _reconstruct_gradient_with_crack_wake_gap(
        displacement_x,
        x_spacing,
        axis=1,
        gap=gap,
    )
    strain_y = _reconstruct_gradient_with_crack_wake_gap(
        displacement_y,
        y_spacing,
        axis=0,
        gap=gap,
    )
    displacement_x_gradient_y = _reconstruct_gradient_with_crack_wake_gap(
        displacement_x,
        y_spacing,
        axis=0,
        gap=gap,
    )
    displacement_y_gradient_x = _reconstruct_gradient_with_crack_wake_gap(
        displacement_y,
        x_spacing,
        axis=1,
        gap=gap,
    )
    # Tensorial shear strain is epsilon_xy = (du_x/dy + du_y/dx) / 2.
    tensorial_shear_strain = 0.5 * (
        displacement_x_gradient_y + displacement_y_gradient_x
    )
    in_plane_strains = InPlaneStrains(
        strain_x=strain_x,
        strain_y=strain_y,
        strain_xy=tensorial_shear_strain,
    )
    return in_plane_strains


#################################
# MODE III FIELD RECONSTRUCTION #
#################################


def reconstruct_mode_iii_fields(
    displacement_z: np.ndarray,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    *,
    material: Material,
    gap: int,
) -> ModeIIIFields:
    """Reconstruct Mode III displacement derivatives and shear stresses.

    Args:
        displacement_z: Regular-grid out-of-plane displacement mesh.
        x_coordinates: Uniform crack-tip-frame coordinates along prospective
            crack extension.
        y_coordinates: Uniform crack-tip-frame coordinates normal to the crack
            plane.
        material: Isotropic material supplying the shear modulus.
        gap: Number of rows omitted on either side of the left crack wake.

    Returns:
        Immutable out-of-plane displacement derivatives and shear stresses.

    Notes:
        Molteno and Becker (2015), equations 9--11, define the Mode III
        out-of-plane displacement derivatives and associated shear stresses.
        DOI: https://doi.org/10.1111/str.12166.
        Citation key: ``molteno_becker_2015_j_integral_decomposition``.
    """
    x_spacing = x_coordinates[1] - x_coordinates[0]
    y_spacing = y_coordinates[1] - y_coordinates[0]
    out_of_plane_displacement_derivative_x = _reconstruct_gradient_with_crack_wake_gap(
        displacement_z,
        x_spacing,
        axis=1,
        gap=gap,
    )
    out_of_plane_displacement_derivative_y = _reconstruct_gradient_with_crack_wake_gap(
        displacement_z,
        y_spacing,
        axis=0,
        gap=gap,
    )
    # Under antiplane shear, sigma_xz = G du_z/dx and sigma_yz = G du_z/dy.
    shear_stress_xz = material.G * out_of_plane_displacement_derivative_x
    shear_stress_yz = material.G * out_of_plane_displacement_derivative_y
    mode_iii_fields = ModeIIIFields(
        out_of_plane_displacement_derivative_x=(
            out_of_plane_displacement_derivative_x
        ),
        out_of_plane_displacement_derivative_y=(
            out_of_plane_displacement_derivative_y
        ),
        shear_stress_xz=shear_stress_xz,
        shear_stress_yz=shear_stress_yz,
    )
    return mode_iii_fields


###################################
# FRACTURE-MODE FIELD PREPARATION #
###################################


def prepare_mode_data(
    mode: str,
    regular_grid: RegularGridDisplacements,
    *,
    material: Material,
) -> InputData:
    """Prepare one fracture-mode field dataset on the regular grid.

    Args:
        mode: Fracture mode as ``"I"``, ``"II"``, or ``"III"``.
        regular_grid: Symmetric crack-tip Cartesian grid whose paired rows
            represent reflection across the crack plane.
        material: Isotropic material used for stress reconstruction.

    Returns:
        Fresh mutable ``InputData`` populated for the requested mode.

    Raises:
        ValueError: If ``mode`` is not Mode I, II, or III.

    Notes:
        Molteno and Becker (2015), equation 3, separates reflected crack-tip
        displacements into Mode I, Mode II, and Mode III symmetry components.
        DOI: https://doi.org/10.1111/str.12166.
        Citation key: ``molteno_becker_2015_j_integral_decomposition``.
    """
    displacement_x_mesh = regular_grid.displacement_x_mesh
    displacement_y_mesh = regular_grid.displacement_y_mesh
    displacement_z_mesh = regular_grid.displacement_z_mesh
    if mode == "I":
        # Mode I uses crack-plane-symmetric u_x and antisymmetric u_y.
        displacement_x = 0.5 * (
            displacement_x_mesh + np.flipud(displacement_x_mesh)
        )
        displacement_y = 0.5 * (
            displacement_y_mesh - np.flipud(displacement_y_mesh)
        )
        displacement_z = np.zeros_like(displacement_x)
    elif mode == "II":
        # Mode II uses crack-plane-antisymmetric u_x and symmetric u_y.
        displacement_x = 0.5 * (
            displacement_x_mesh - np.flipud(displacement_x_mesh)
        )
        displacement_y = 0.5 * (
            displacement_y_mesh + np.flipud(displacement_y_mesh)
        )
        displacement_z = np.zeros_like(displacement_x)
    elif mode == "III":
        # Mode III uses the crack-plane-antisymmetric u_z component.
        displacement_z = 0.5 * (
            displacement_z_mesh - np.flipud(displacement_z_mesh)
        )
        displacement_x = np.zeros_like(displacement_z)
        displacement_y = np.zeros_like(displacement_z)
    else:
        raise ValueError("Mode has to be 'I', 'II' or 'III'!")

    # For both reconstruction calls, gap=2 leaves two rows on each side of the
    # left crack wake at zero.
    in_plane = reconstruct_in_plane_strains(
        displacement_x,
        displacement_y,
        regular_grid.x_coordinates,
        regular_grid.y_coordinates,
        gap=2,
    )
    mode_iii = reconstruct_mode_iii_fields(
        displacement_z,
        regular_grid.x_coordinates,
        regular_grid.y_coordinates,
        material=material,
        gap=2,
    )
    decomp_data = InputData()
    decomp_data.coor_x = regular_grid.x_mesh.flatten()
    decomp_data.coor_y = regular_grid.y_mesh.flatten()
    decomp_data.disp_x = displacement_x.flatten()
    decomp_data.disp_y = displacement_y.flatten()
    decomp_data.disp_z = displacement_z.flatten()
    decomp_data.eps_x = in_plane.strain_x.flatten()
    decomp_data.eps_y = in_plane.strain_y.flatten()
    decomp_data.eps_xy = in_plane.strain_xy.flatten()
    decomp_data.eps_xz = (
        mode_iii.out_of_plane_displacement_derivative_x.flatten()
    )
    decomp_data.eps_yz = (
        mode_iii.out_of_plane_displacement_derivative_y.flatten()
    )
    decomp_data.sigma_xz = mode_iii.shear_stress_xz.flatten()
    decomp_data.sigma_yz = mode_iii.shear_stress_yz.flatten()
    decomp_data.calc_eps_vm()
    decomp_data.calc_stresses(material)
    return decomp_data
