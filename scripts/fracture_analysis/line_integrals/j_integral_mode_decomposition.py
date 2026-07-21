"""Demonstrate modal J-integral evaluation on a synthetic mixed-mode Williams field.

The script reports prescribed fracture quantities alongside one contour's public line-integral results.
"""

import sys

import numpy as np

from crackpy.fracture_analysis.crack_tip import (
    williams_displ_field_xy,
    williams_displ_field_z,
    williams_stress_field,
)
from crackpy.fracture_analysis.line_integration import (
    IntegrationPath,
    LineIntegral,
    PathProperties,
)
from crackpy.input.input_data import InputData
from crackpy.structure_elements.material import Material


def _synthetic_williams_data(
    material: Material,
    *,
    mode_i_sif: float,
    mode_ii_sif: float,
    mode_iii_sif: float,
    t_stress: float,
) -> InputData:
    """Build smooth mixed-mode Williams fields on a crack-tip-centered grid.

    Notes:
        The in-plane and out-of-plane field definitions follow Kuna (2013),
        equations 3.41--3.55, DOI https://doi.org/10.1007/978-94-007-6680-8.
    """
    # CrackPy's Williams field functions use millimetres, so first-order
    # coefficients carry MPa sqrt(mm) while public SIFs use MPa sqrt(m).
    sqrt_mm_per_sqrt_m = np.sqrt(1000.0)
    symmetric_first_order = mode_i_sif * sqrt_mm_per_sqrt_m / np.sqrt(2.0 * np.pi)
    antisymmetric_first_order = -mode_ii_sif * sqrt_mm_per_sqrt_m / np.sqrt(
        2.0 * np.pi
    )
    out_of_plane_first_order = (
        mode_iii_sif * sqrt_mm_per_sqrt_m / np.sqrt(0.5 * np.pi)
    )
    symmetric_second_order = t_stress / 4.0

    terms = [1, 2]
    symmetric_coefficients = [symmetric_first_order, symmetric_second_order]
    antisymmetric_coefficients = [antisymmetric_first_order, 0.0]
    out_of_plane_coefficients = [out_of_plane_first_order, 0.0]

    # An even node count avoids evaluating the singular Williams term at r=0.
    coordinates = np.linspace(-10.0, 10.0, 120)
    x_mesh, y_mesh = np.meshgrid(coordinates, coordinates, indexing="xy")
    radius = np.hypot(x_mesh, y_mesh)
    angle = np.arctan2(y_mesh, x_mesh)
    displacement_x, displacement_y = williams_displ_field_xy(
        symmetric_coefficients,
        antisymmetric_coefficients,
        terms,
        angle,
        radius,
        material,
    )
    displacement_z = williams_displ_field_z(
        out_of_plane_coefficients,
        terms,
        angle,
        radius,
        material,
    )
    stress_x, stress_y, stress_xy = williams_stress_field(
        symmetric_coefficients,
        antisymmetric_coefficients,
        terms,
        angle,
        radius,
    )

    data = InputData()
    data.coor_x = x_mesh.ravel()
    data.coor_y = y_mesh.ravel()
    data.disp_x = displacement_x.ravel()
    data.disp_y = displacement_y.ravel()
    data.disp_z = displacement_z.ravel()
    # Plane-stress compliance maps the analytical Williams stresses to the
    # tensorial strain convention consumed by CrackPy.
    data.eps_x = ((stress_x - material.nu_xy * stress_y) / material.E).ravel()
    data.eps_y = ((stress_y - material.nu_xy * stress_x) / material.E).ravel()
    data.eps_xy = (stress_xy / (2.0 * material.G)).ravel()
    grid_spacing = coordinates[1] - coordinates[0]
    # Differentiate the public Williams out-of-plane displacement so the
    # synthetic carrier contains the same Mode III kinematics it prescribes.
    displacement_z_gradient_x = np.gradient(
        displacement_z,
        grid_spacing,
        axis=1,
    )
    displacement_z_gradient_y = np.gradient(
        displacement_z,
        grid_spacing,
        axis=0,
    )
    data.eps_xz = displacement_z_gradient_x.ravel()
    data.eps_yz = displacement_z_gradient_y.ravel()
    data.calc_eps_vm()
    data.calc_stresses(material)
    data.sigma_xz = material.G * data.eps_xz
    data.sigma_yz = material.G * data.eps_yz
    return data


def main() -> int:
    """Evaluate and print expected and recovered modal J and SIF quantities.

    Returns:
        Zero when every recovered modal quantity is finite, otherwise one.

    Notes:
        Mode decomposition and modal J/SIF relations follow Molteno and Becker
        (2015), equations 3, 6, 9--11, 16, and 17, DOI
        https://doi.org/10.1111/str.12166.
    """
    material = Material(E=72000.0, nu_xy=0.33)
    mode_i_sif = 10.0
    mode_ii_sif = 5.0
    mode_iii_sif = 3.0
    t_stress = 20.0
    data = _synthetic_williams_data(
        material,
        mode_i_sif=mode_i_sif,
        mode_ii_sif=mode_ii_sif,
        mode_iii_sif=mode_iii_sif,
        t_stress=t_stress,
    )
    contour = IntegrationPath(
        path_properties=PathProperties(
            size_left=-5.0,
            size_right=5.0,
            size_bottom=-5.0,
            size_top=5.0,
            tick_size=0.25,
            num_nodes=None,
            top_offset=0.5,
            bottom_offset=-0.5,
        )
    )
    line_integral = LineIntegral(contour, data, material)
    # The public line-integral facade performs the Molteno-Becker symmetry
    # decomposition and evaluates one J-integral for each fracture mode.
    line_integral.integrate_j_decompose()

    # Molteno and Becker (2015), Eqs. 16--17: under plane stress, in-plane
    # J is K**2/E and Mode III uses J=K**2/(2G). The factor 1000 converts
    # prescribed MPa sqrt(m) SIFs to the MPa sqrt(mm) field convention.
    expected_j_i = mode_i_sif**2 * 1000.0 / material.E
    expected_j_ii = mode_ii_sif**2 * 1000.0 / material.E
    expected_j_iii = mode_iii_sif**2 * 1000.0 / (2.0 * material.G)
    recovered = np.asarray(
        [
            line_integral.decomp_j_integral_I,
            line_integral.decomp_j_integral_II,
            line_integral.decomp_j_integral_III,
            line_integral.decomp_j_integral_K_I,
            line_integral.decomp_j_integral_K_II,
            line_integral.decomp_j_integral_K_III,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(recovered)):
        print("Mode-decomposition demonstration produced a non-finite result.", file=sys.stderr)
        return 1

    print(f"Expected Mode I J-integral: {expected_j_i:.8g} N/mm")
    print(f"Recovered Mode I J-integral: {recovered[0]:.8g} N/mm")
    print(f"Expected Mode II J-integral: {expected_j_ii:.8g} N/mm")
    print(f"Recovered Mode II J-integral: {recovered[1]:.8g} N/mm")
    print(f"Expected Mode III J-integral: {expected_j_iii:.8g} N/mm")
    print(f"Recovered Mode III J-integral: {recovered[2]:.8g} N/mm")
    print(f"Expected Mode I SIF: {mode_i_sif:.8g} MPa sqrt(m)")
    print(f"Recovered Mode I SIF: {recovered[3]:.8g} MPa sqrt(m)")
    print(f"Expected Mode II SIF: {mode_ii_sif:.8g} MPa sqrt(m)")
    print(f"Recovered Mode II SIF: {recovered[4]:.8g} MPa sqrt(m)")
    print(f"Expected Mode III SIF: {mode_iii_sif:.8g} MPa sqrt(m)")
    print(f"Recovered Mode III SIF: {recovered[5]:.8g} MPa sqrt(m)")
    print(f"Prescribed T-stress: {t_stress:.8g} MPa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
