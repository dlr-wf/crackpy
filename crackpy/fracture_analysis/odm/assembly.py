"""ODM assembly converts sampled displacements and model-owned bases into
fixed linear coefficient systems.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crackpy.fracture_analysis.crack_tip_fields.cjp import (
    cjp_mixed_mode_displacement_basis,
    cjp_mode_i_displacement_basis,
)
from crackpy.fracture_analysis.crack_tip_fields.williams import (
    williams_in_plane_displacement_basis,
    williams_out_of_plane_displacement_basis,
)
from crackpy.fracture_analysis.odm.sampling import OptimizationGrid
from crackpy.structure_elements.material import Material


@dataclass(frozen=True)
class LinearSystem:
    """Describe a dense linear least-squares system in matrix-target form.

    Attributes:
        matrix: Residual-by-coefficient system matrix.
        target: Measured displacement target vector.
    """

    matrix: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class CjpAssembly:
    """Describe preassembled CJP displacement systems on an ODM grid.

    Attributes:
        valid_mask_xy: Validity mask for flattened x/y displacement equations.
        mode_i: Linear system for the CJP Mode I formulation.
        mixed_mode: Linear system for the CJP mixed-mode formulation.
    """

    valid_mask_xy: np.ndarray
    mode_i: LinearSystem
    mixed_mode: LinearSystem


@dataclass(frozen=True)
class WilliamsAssembly:
    """Describe preassembled Williams displacement systems on an ODM grid.

    Attributes:
        valid_mask_xy: Validity mask for flattened x/y displacement equations.
        valid_mask_z: Validity mask for flattened z displacement equations.
        xy: Linear system for in-plane displacements.
        z: Linear system for out-of-plane displacements.
    """

    valid_mask_xy: np.ndarray
    valid_mask_z: np.ndarray
    xy: LinearSystem
    z: LinearSystem


def _assemble_cjp_system_matrix(
    jac_disp_x: np.ndarray,
    jac_disp_y: np.ndarray,
    component_size: int,
    valid_mask_xy: np.ndarray,
) -> np.ndarray:
    """Stack x/y CJP basis responses into a system matrix.

    Args:
        jac_disp_x: Coefficient-first x-displacement basis responses.
        jac_disp_y: Coefficient-first y-displacement basis responses.
        component_size: Number of equations in one displacement component.
        valid_mask_xy: Validity mask for flattened x/y equations.

    Returns:
        A residual-by-coefficient system matrix containing valid equations.
    """
    n_coeffs = jac_disp_x.shape[0]
    system_matrix = np.empty((n_coeffs, 2 * component_size))
    system_matrix[:, :component_size] = jac_disp_x.reshape(n_coeffs, -1)
    system_matrix[:, component_size:] = jac_disp_y.reshape(n_coeffs, -1)
    return system_matrix[:, valid_mask_xy].T


def assemble_cjp(
    interp_disp_x: np.ndarray,
    interp_disp_y: np.ndarray,
    grid: OptimizationGrid,
    material: Material,
) -> CjpAssembly:
    """Assemble fixed linear CJP systems for both displacement formulations.

    Args:
        interp_disp_x: Interpolated x-displacements on the ODM grid.
        interp_disp_y: Interpolated y-displacements on the ODM grid.
        grid: Fixed ODM optimization grid.
        material: Material properties for the fitted field.

    Returns:
        The Mode I and mixed-mode CJP linear systems.
    """
    target_xy = np.asarray([interp_disp_x, interp_disp_y]).reshape(-1)
    valid_mask_xy = ~np.isnan(target_xy)
    component_size = interp_disp_x.size
    mode_i_jac_x, mode_i_jac_y = cjp_mode_i_displacement_basis(
        grid.r,
        grid.phi,
        material,
    )
    mixed_mode_jac_x, mixed_mode_jac_y = cjp_mixed_mode_displacement_basis(
        grid.r,
        grid.phi,
        material,
    )
    mode_i_matrix = _assemble_cjp_system_matrix(
        mode_i_jac_x,
        mode_i_jac_y,
        component_size,
        valid_mask_xy,
    )
    mixed_mode_matrix = _assemble_cjp_system_matrix(
        mixed_mode_jac_x,
        mixed_mode_jac_y,
        component_size,
        valid_mask_xy,
    )
    target = target_xy[valid_mask_xy]
    return CjpAssembly(
        valid_mask_xy=valid_mask_xy,
        mode_i=LinearSystem(matrix=mode_i_matrix, target=target),
        mixed_mode=LinearSystem(matrix=mixed_mode_matrix, target=target),
    )


def assemble_williams(
    interp_disp_x: np.ndarray,
    interp_disp_y: np.ndarray,
    interp_disp_z: np.ndarray,
    grid: OptimizationGrid,
    terms: np.ndarray,
    material: Material,
) -> WilliamsAssembly:
    """Assemble Williams displacement systems for x/y and z fitting.

    Args:
        interp_disp_x: Interpolated x-displacements on the ODM grid.
        interp_disp_y: Interpolated y-displacements on the ODM grid.
        interp_disp_z: Interpolated z-displacements on the ODM grid.
        grid: Fixed ODM optimization grid.
        terms: Williams expansion terms.
        material: Material properties for the fitted field.

    Returns:
        The in-plane and out-of-plane Williams linear systems.
    """
    target_xy = np.asarray([interp_disp_x, interp_disp_y]).reshape(-1)
    valid_mask_xy = ~np.isnan(target_xy)
    target_z = interp_disp_z.reshape(-1)
    valid_mask_z = ~np.isnan(target_z)
    basis_x, basis_y = williams_in_plane_displacement_basis(
        grid.r,
        grid.phi,
        terms,
        material,
    )
    n_coefficients = basis_x.shape[0]
    coefficient_first_xy = np.concatenate(
        [basis_x.reshape(n_coefficients, -1), basis_y.reshape(n_coefficients, -1)],
        axis=1,
    )
    basis_z = williams_out_of_plane_displacement_basis(
        grid.r,
        grid.phi,
        terms,
        material,
    )
    williams_matrix_xy = coefficient_first_xy
    williams_matrix_z = basis_z.reshape(len(terms), -1)
    matrix_xy = williams_matrix_xy[:, valid_mask_xy].T
    matrix_z = williams_matrix_z[:, valid_mask_z].T
    return WilliamsAssembly(
        valid_mask_xy=valid_mask_xy,
        valid_mask_z=valid_mask_z,
        xy=LinearSystem(matrix=matrix_xy, target=target_xy[valid_mask_xy]),
        z=LinearSystem(matrix=matrix_z, target=target_z[valid_mask_z]),
    )
