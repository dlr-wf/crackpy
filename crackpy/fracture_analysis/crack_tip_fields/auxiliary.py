"""Analytical auxiliary stress, strain and displacement fields for contour integrals."""

import numpy as np

from crackpy.structure_elements.material import Material


def get_crack_nearfield(k_i: float, k_ii: float, r: float, phi: float, material: Material) -> tuple:
    """Formula for the analytical stress and strain using the crack near field.
    [see Eq. 3,4 in Sladek et al. Contour integrals for mixed-mode crack analysis: effect of non-singular terms (1997)]

    Args:
        k_i: first stress intensity factor K_I
        k_ii: second stress intensity factor K_II
        r: radius from polar coordinates [mm]
        phi: angle from polar coordinates [rad]
        material: obj of class Material

    Returns:
        stress tensor, strain tensor, [displacement x, displacement y]

    """
    kappa = material.kappa

    sigma_x_ana = k_i / (np.sqrt(2 * np.pi * r)) * np.cos(phi / 2) * (1 - np.sin(phi / 2) * np.sin(3 * phi / 2)) \
                  - k_ii / (np.sqrt(2 * np.pi * r)) * np.sin(phi / 2) * (2 + np.cos(phi / 2) * np.cos(3 * phi / 2))

    sigma_y_ana = k_i / (np.sqrt(2 * np.pi * r)) * np.cos(phi / 2) * (1 + np.sin(phi / 2) * np.sin(3 * phi / 2)) \
                  + k_ii / (np.sqrt(2 * np.pi * r)) * np.sin(phi / 2) * np.cos(phi / 2) * np.cos(3 * phi / 2)

    sigma_xy_ana = k_i / (np.sqrt(2 * np.pi * r)) * np.sin(phi / 2) * np.cos(phi / 2) * np.cos(3 * phi / 2) \
                   + k_ii / (np.sqrt(2 * np.pi * r)) * np.cos(phi / 2) * (1 - np.sin(phi / 2) * np.sin(3 * phi / 2))

    # the following two formulas differ from the reference but are identical (by trigonometric identities)
    u_x_ana = k_i / (2 * material.G) * np.sqrt(r / (2 * np.pi)) * (np.cos(phi / 2) * (kappa - np.cos(phi))) \
              + k_ii / (2 * material.G) * np.sqrt(r / (2 * np.pi)) * (np.sin(phi / 2) * (kappa + 2 + np.cos(phi)))

    v_x_ana = k_i / (2 * material.G) * np.sqrt(r / (2 * np.pi)) * (np.sin(phi / 2) * (kappa - np.cos(phi))) \
              + k_ii / (2 * material.G) * np.sqrt(r / (2 * np.pi)) * (-np.cos(phi / 2) * (kappa - 2 + np.cos(phi)))

    eps_vector_ana = np.dot(np.linalg.inv(material.stiffness_matrix), [sigma_x_ana, sigma_y_ana, sigma_xy_ana])
    eps_tensor_ana = np.asarray([[eps_vector_ana[0], eps_vector_ana[2]], [eps_vector_ana[2], eps_vector_ana[1]]])
    sigma_tensor_ana = np.asarray([[sigma_x_ana, sigma_xy_ana], [sigma_xy_ana, sigma_y_ana]])

    return sigma_tensor_ana, eps_tensor_ana, [u_x_ana, v_x_ana]


def get_zhao_solutions(r: float, phi: float, material: Material, force: float = 1, dist: float = 1) -> tuple:
    """Returns stress and displacement test functions according to formulas (4a-b) in
    Zhao et al. 'Stress intensity factor K and the elastic T-stress for corner cracks' (2001)

    Args:
        r: radius from polar coordinates [mm]
        phi: angle from polar coordinates [rad]
        material: obj of class Material
        force: point force applied at the crack tip
        dist: reference distance from the crack tip on the x-axis

    Returns:
        stress_11, stress_22, stress_12, disp_x, disp_y

    """
    sigma_x = - force / (np.pi * r) * np.cos(phi) ** 3
    sigma_y = - force / (np.pi * r) * np.cos(phi) * np.sin(phi) ** 2
    sigma_xy = - force / (np.pi * r) * np.cos(phi) ** 2 * np.sin(phi)

    u_x = - (1 - material.nu_xy ** 2) / material.E * force / np.pi * (np.log(r / dist)
                                                                      + np.sin(phi) ** 2 / (2 * (1 - material.nu_xy)))
    u_y = - (1 + material.nu_xy) / (2 * material.E) * force / np.pi * ((1 - 2 * material.nu_xy) * phi
                                                                       - np.cos(phi) * np.sin(phi))

    return sigma_x, sigma_y, sigma_xy, u_x, u_y
