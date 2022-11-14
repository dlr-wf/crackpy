import numpy as np


class Material:
    """Material properties class."""
    def __init__(self, name: str = "not defined material", E: float = 72000,
                 nu_xy: float = 0.33, sig_yield: float = 350, plane_strain: bool = False):
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
        self.kappa = (3 - self.nu_xy) / (1 + self.nu_xy) if not plane_strain else 3 - 4*self.nu_xy

    def _stiffness_matrix(self):
        """Returns stiffness matrix under plane stress condition."""
        return self.E / (1 - self.nu_xy ** 2) * np.array(
            [[1, self.nu_xy, 0],
             [self.nu_xy, 1, 0],
             [0, 0, 1 - self.nu_xy]]
        )

    def _stiffness_matrix_plane_strain(self):
        """Returns stiffness matrix under plane strain condition."""
        return self.E / (1 + self.nu_xy) * np.array(
            [[1 + self.nu_xy / (1 - 2*self.nu_xy), self.nu_xy / (1 - 2*self.nu_xy), 0],
             [self.nu_xy / (1 - 2*self.nu_xy), 1 + self.nu_xy / (1 - 2*self.nu_xy), 0],
             [0, 0, 1 - self.nu_xy]]
        )

    def _inverse_stiffness_matrix(self):
        """Returns inverse of stiffness matrix under plane stress condition."""
        return 1 / self.E * np.array(
            [[1, -self.nu_xy, 0],
             [-self.nu_xy, 1, 0],
             [0, 0, 1 + self.nu_xy]]
        )
