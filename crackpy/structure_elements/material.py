import numpy as np
import logging

logger = logging.getLogger(__name__)


class Material:
    """Material properties class.

    This class calculates and stores material properties including shear modulus,
    stiffness matrix, and inverse stiffness matrix based on the input parameters.

    Attributes:
        name: Material's name
        E: Young's modulus in MPa
        nu_xy: Poisson's ratio
        sig_yield: Yield stress in MPa
        plane_strain: Plane strain or plane stress condition
        G: Shear modulus in MPa
        stiffness_matrix: Material stiffness matrix
        inverse_stiffness_matrix: Inverse of stiffness matrix
        kappa: Material constant kappa

    """
    def __init__(self, name: str = "not defined material", E: float = 72000,
                 nu_xy: float = 0.33, sig_yield: float = 350, plane_strain: bool = False) -> None:
        """Calculate shear modulus, stiffness matrix and inverse stiffness matrix.

        Args:
            name: Material's name, e.g. AA2024-T3
            E: Young's modulus [MPa]
            nu_xy: Poisson's ratio
            sig_yield: yield stress [MPa]
            plane_strain: If True, plane strain condition is used
                          If False, plane stress condition is used

        """
        self.name = name
        self.E = E
        self.nu_xy = nu_xy
        self.sig_yield = sig_yield
        self.plane_strain = plane_strain

        self.G = self.E / (2 * (1 + self.nu_xy))  # shear modulus [MPa]

        self.stiffness_matrix = self._stiffness_matrix() if not plane_strain else self._stiffness_matrix_plane_strain()
        self.inverse_stiffness_matrix = self._inverse_stiffness_matrix()
        self.kappa = (3 - self.nu_xy) / (1 + self.nu_xy) if not plane_strain else 3 - 4 * self.nu_xy

        logger.debug("Material initialized: %s, E=%s MPa, nu=%s, plane_strain=%s, G=%.2f MPa, kappa=%.4f",
                     name, E, nu_xy, plane_strain, self.G, self.kappa)

    def _stiffness_matrix(self) -> np.ndarray:
        """Returns stiffness matrix under plane stress condition.

        Returns:
            3x3 stiffness matrix

        """
        return self.E / (1 - self.nu_xy ** 2) * np.array(
            [[1, self.nu_xy, 0],
             [self.nu_xy, 1, 0],
             [0, 0, 1 - self.nu_xy]]
        )

    def _stiffness_matrix_plane_strain(self) -> np.ndarray:
        """Returns stiffness matrix under plane strain condition.

        Returns:
            3x3 stiffness matrix

        """
        return self.E / (1 + self.nu_xy) * np.array(
            [[1 + self.nu_xy / (1 - 2 * self.nu_xy), self.nu_xy / (1 - 2 * self.nu_xy), 0],
             [self.nu_xy / (1 - 2 * self.nu_xy), 1 + self.nu_xy / (1 - 2 * self.nu_xy), 0],
             [0, 0, 1 - self.nu_xy]]
        )

    def _inverse_stiffness_matrix(self) -> np.ndarray:
        """Returns inverse of stiffness matrix under plane stress condition.

        Returns:
            3x3 inverse stiffness matrix

        """
        return 1 / self.E * np.array(
            [[1, -self.nu_xy, 0],
             [-self.nu_xy, 1, 0],
             [0, 0, 1 + self.nu_xy]]
        )
