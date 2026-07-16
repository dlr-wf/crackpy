from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, optimize

from crackpy.fracture_analysis._odm_grid_interpolation import OptimizationGrid
from crackpy.structure_elements.material import Material


@dataclass(frozen=True)
class PolarBasisFields:
    """Describe reusable trigonometric and radial basis arrays on an ODM grid.

    Attributes:
        inverse_two_shear_modulus: Reciprocal of twice the material shear modulus.
        sqrt_r: Square root of radial grid coordinates.
        log_r: Natural logarithm of radial grid coordinates.
        phi_half: Half of the angular grid coordinates.
        phi_three_half: Three halves of the angular grid coordinates.
        sin_phi_half: Sine of half-angle coordinates.
        cos_phi_half: Cosine of half-angle coordinates.
        sin_phi_three_half: Sine of three-half-angle coordinates.
        cos_phi_three_half: Cosine of three-half-angle coordinates.
        sin_phi: Sine of angular grid coordinates.
        cos_phi: Cosine of angular grid coordinates.
    """

    inverse_two_shear_modulus: float
    sqrt_r: np.ndarray
    log_r: np.ndarray
    phi_half: np.ndarray
    phi_three_half: np.ndarray
    sin_phi_half: np.ndarray
    cos_phi_half: np.ndarray
    sin_phi_three_half: np.ndarray
    cos_phi_three_half: np.ndarray
    sin_phi: np.ndarray
    cos_phi: np.ndarray


@dataclass(frozen=True)
class LinearizedSystem:
    """Describe a dense linear least-squares system in matrix-target form.

    Attributes:
        matrix: Residual-by-coefficient system matrix.
        target: Measured displacement target vector.
    """

    matrix: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class CjpSystems:
    """Describe preassembled CJP displacement systems on an ODM grid.

    Attributes:
        valid_mask_xy: Validity mask for flattened x/y displacement equations.
        mode_i: Linear system for the CJP Mode I formulation.
        mixed_mode: Linear system for the CJP mixed-mode formulation.
    """

    valid_mask_xy: np.ndarray
    mode_i: LinearizedSystem
    mixed_mode: LinearizedSystem


@dataclass(frozen=True)
class WilliamsSystems:
    """Describe preassembled Williams displacement systems on an ODM grid.

    Attributes:
        valid_mask_xy: Validity mask for flattened x/y displacement equations.
        valid_mask_z: Validity mask for flattened z displacement equations.
        xy: Linear system for in-plane displacements.
        z: Linear system for out-of-plane displacements.
    """

    valid_mask_xy: np.ndarray
    valid_mask_z: np.ndarray
    xy: LinearizedSystem
    z: LinearizedSystem


def build_polar_basis_fields(grid: OptimizationGrid, material: Material) -> PolarBasisFields:
    """Precompute shared basis fields used by CJP and Williams assembly.

    Args:
        grid: Fixed ODM optimization grid.
        material: Material properties for the fitted field.

    Returns:
        Reusable radial, angular, and material basis values.
    """
    phi_half = grid.phi / 2
    phi_three_half = 3 * phi_half
    return PolarBasisFields(
        inverse_two_shear_modulus=1 / (2 * material.G),
        sqrt_r=np.sqrt(grid.r),
        log_r=np.log(grid.r),
        phi_half=phi_half,
        phi_three_half=phi_three_half,
        sin_phi_half=np.sin(phi_half),
        cos_phi_half=np.cos(phi_half),
        sin_phi_three_half=np.sin(phi_three_half),
        cos_phi_three_half=np.cos(phi_three_half),
        sin_phi=np.sin(grid.phi),
        cos_phi=np.cos(grid.phi),
    )


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


def build_cjp_systems(
    interp_disp_x: np.ndarray,
    interp_disp_y: np.ndarray,
    grid: OptimizationGrid,
    material: Material,
    basis: PolarBasisFields,
) -> CjpSystems:
    """Assemble fixed linear CJP systems for both displacement formulations.

    Args:
        interp_disp_x: Interpolated x-displacements on the ODM grid.
        interp_disp_y: Interpolated y-displacements on the ODM grid.
        grid: Fixed ODM optimization grid.
        material: Material properties for the fitted field.
        basis: Precomputed polar basis fields.

    Returns:
        The Mode I and mixed-mode CJP linear systems.
    """
    target_xy = np.asarray([interp_disp_x, interp_disp_y]).reshape(-1)
    valid_mask_xy = ~np.isnan(target_xy)
    component_size = interp_disp_x.size
    kappa = material.kappa

    mode_i_jac_x = np.asarray([
        -basis.sqrt_r * basis.cos_phi_half,
        basis.sqrt_r * (basis.cos_phi_three_half - 2 * kappa * basis.cos_phi_half),
        -grid.r * (1 + kappa) * basis.cos_phi / 4,
        basis.sqrt_r * (
            -2 * basis.cos_phi_half
            + 2 * basis.cos_phi_three_half
            - basis.log_r * (basis.cos_phi_three_half + (1 - 2 * kappa) * basis.cos_phi_half)
            - grid.phi * (basis.sin_phi_three_half + (2 * kappa - 1) * basis.sin_phi_half)
        ),
        grid.r * (kappa - 3) * basis.cos_phi / 4,
    ])
    mode_i_jac_y = np.asarray([
        basis.sqrt_r * basis.sin_phi_half,
        basis.sqrt_r * (basis.sin_phi_three_half - 2 * kappa * basis.sin_phi_half),
        grid.r * (3 - kappa) * basis.sin_phi / 4,
        basis.sqrt_r * (
            2 * basis.sin_phi_half
            + 2 * basis.sin_phi_three_half
            + basis.log_r * (basis.sin_phi_three_half - (1 + 2 * kappa) * basis.sin_phi_half)
            - grid.phi * (basis.cos_phi_three_half + (2 * kappa + 1) * basis.cos_phi_half)
        ),
        grid.r * (kappa + 1) * basis.sin_phi / 4,
    ])
    mixed_mode_jac_x = np.asarray([
        -basis.sqrt_r * basis.cos_phi_half,
        basis.sqrt_r * (basis.cos_phi_three_half - 2 * kappa * basis.cos_phi_half),
        basis.sqrt_r * ((2 * kappa - 3) * basis.sin_phi_half - basis.sin_phi_three_half),
        -grid.r * (1 + kappa) * basis.cos_phi / 4,
        basis.sqrt_r * (
            -2 * basis.cos_phi_half
            + 2 * basis.cos_phi_three_half
            + basis.log_r * (basis.cos_phi_three_half + (1 - 2 * kappa) * basis.cos_phi_half)
            + grid.phi * (basis.sin_phi_three_half + (1 + 2 * kappa) * basis.sin_phi_half)
        ),
    ])
    mixed_mode_jac_y = np.asarray([
        basis.sqrt_r * basis.sin_phi_half,
        basis.sqrt_r * (basis.sin_phi_three_half - 2 * kappa * basis.sin_phi_half),
        basis.sqrt_r * (basis.cos_phi_three_half - (2 * kappa + 3) * basis.cos_phi_half),
        grid.r * (3 - kappa) * basis.sin_phi / 4,
        basis.sqrt_r * (
            2 * basis.sin_phi_half
            + 2 * basis.sin_phi_three_half
            + basis.log_r * (basis.sin_phi_three_half - (1 + 2 * kappa) * basis.sin_phi_half)
            - grid.phi * (basis.cos_phi_three_half + (1 + 2 * kappa) * basis.cos_phi_half)
        ),
    ])

    scale = basis.inverse_two_shear_modulus
    mode_i_matrix = _assemble_cjp_system_matrix(
        mode_i_jac_x,
        mode_i_jac_y,
        component_size,
        valid_mask_xy,
    ) * scale
    mixed_mode_matrix = _assemble_cjp_system_matrix(
        mixed_mode_jac_x,
        mixed_mode_jac_y,
        component_size,
        valid_mask_xy,
    ) * scale
    target = target_xy[valid_mask_xy]
    return CjpSystems(
        valid_mask_xy=valid_mask_xy,
        mode_i=LinearizedSystem(matrix=mode_i_matrix, target=target),
        mixed_mode=LinearizedSystem(matrix=mixed_mode_matrix, target=target),
    )


def build_williams_systems(
    interp_disp_x: np.ndarray,
    interp_disp_y: np.ndarray,
    interp_disp_z: np.ndarray,
    grid: OptimizationGrid,
    terms: np.ndarray,
    material: Material,
    basis: PolarBasisFields,
) -> WilliamsSystems:
    """Assemble Williams displacement systems for x/y and z fitting.

    Args:
        interp_disp_x: Interpolated x-displacements on the ODM grid.
        interp_disp_y: Interpolated y-displacements on the ODM grid.
        interp_disp_z: Interpolated z-displacements on the ODM grid.
        grid: Fixed ODM optimization grid.
        terms: Williams expansion terms.
        material: Material properties for the fitted field.
        basis: Precomputed polar basis fields.

    Returns:
        The in-plane and out-of-plane Williams linear systems.
    """
    n_terms = len(terms)
    half_terms = terms / 2
    target_xy = np.asarray([interp_disp_x, interp_disp_y]).reshape(-1)
    valid_mask_xy = ~np.isnan(target_xy)
    target_z = interp_disp_z.reshape(-1)
    valid_mask_z = ~np.isnan(target_z)
    sign_terms = (-1.0) ** terms
    radial_scale = basis.inverse_two_shear_modulus * grid.r ** half_terms[:, None, None]
    williams_matrix_xy = np.empty((2 * n_terms, target_xy.size))

    for index, n_half in enumerate(half_terms):
        sign_n = sign_terms[index]
        radial = radial_scale[index]
        phi_n = n_half * grid.phi
        phi_n_minus_2 = (n_half - 2) * grid.phi
        f_1 = (material.kappa + sign_n + n_half) * np.cos(phi_n) - n_half * np.cos(phi_n_minus_2)
        g_1 = (-material.kappa + sign_n - n_half) * np.sin(phi_n) + n_half * np.sin(phi_n_minus_2)
        f_2 = (material.kappa - sign_n - n_half) * np.sin(phi_n) + n_half * np.sin(phi_n_minus_2)
        g_2 = (material.kappa + sign_n - n_half) * np.cos(phi_n) + n_half * np.cos(phi_n_minus_2)
        component_size = target_xy.size // 2
        williams_matrix_xy[index, :component_size] = (radial * f_1).reshape(-1)
        williams_matrix_xy[index, component_size:] = (radial * f_2).reshape(-1)
        williams_matrix_xy[n_terms + index, :component_size] = (radial * g_1).reshape(-1)
        williams_matrix_xy[n_terms + index, component_size:] = (radial * g_2).reshape(-1)

    williams_matrix_z = np.empty((n_terms, target_z.size))
    for index, n in enumerate(terms):
        n_half = half_terms[index]
        h_3 = 2 * np.cos(n_half * grid.phi) if n % 2 == 0 else 2 * np.sin(n_half * grid.phi)
        williams_matrix_z[index, :] = (radial_scale[index] * h_3).reshape(-1)

    matrix_xy = williams_matrix_xy[:, valid_mask_xy].T
    matrix_z = williams_matrix_z[:, valid_mask_z].T
    return WilliamsSystems(
        valid_mask_xy=valid_mask_xy,
        valid_mask_z=valid_mask_z,
        xy=LinearizedSystem(matrix=matrix_xy, target=target_xy[valid_mask_xy]),
        z=LinearizedSystem(matrix=matrix_z, target=target_z[valid_mask_z]),
    )


def solve_linear_system(system: LinearizedSystem) -> optimize.OptimizeResult:
    """Solve a fixed linear least-squares system with SciPy's GELSS driver.

    Args:
        system: Matrix and measured target describing the coefficient fit.

    Returns:
        An ``OptimizeResult`` compatible with the previous iterative API.
    """
    coefficients, _, rank, singular_values = linalg.lstsq(
        system.matrix,
        system.target,
        lapack_driver="gelss",
    )
    residual = system.matrix @ coefficients - system.target
    return optimize.OptimizeResult(
        x=coefficients,
        cost=0.5 * np.dot(residual, residual),
        fun=residual,
        jac=system.matrix.copy(),
        success=True,
        status=1,
        nfev=1,
        njev=1,
        message="Solved by direct linear least squares.",
        rank=rank,
        singular_values=singular_values,
    )
