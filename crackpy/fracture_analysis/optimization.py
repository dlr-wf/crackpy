import logging
from typing import Union, Optional

import numpy as np
from scipy import optimize

from crackpy.fracture_analysis.crack_tip import williams_displ_field_xy, cjp_displ_field_mixedmode, \
    williams_displ_field_z, cjp_displ_field_modeI
from crackpy.fracture_analysis.utils import ReusableLinearInterpolator
from crackpy.input.input_data import InputData
from crackpy.structure_elements.material import Material

logger = logging.getLogger(__name__)

DEFAULT_WILLIAMS_OPT_TERMS = [-1, 1, 2, 3, 4, 5]


class OptimizationProperties:
    """Class for setting the Optimization properties."""

    def __init__(
            self,
            angle_gap: Optional[float] = 20,
            min_radius: Optional[float] = None,
            max_radius: Optional[float] = None,
            tick_size: Optional[float] = 0.01,
            terms=None,
    ):
        """Initialize Optimization properties.

        Args:
            angle_gap: Angle gap between crack path and fitting domain.
                If None, angle_gap is set to 20.
            min_radius: minimum radius of fitting domain.
                If None, min_radius is set to crack_length / 20.
            max_radius: maximum radius of fitting domain.
                If None, max_radius is set to crack_length / 5.
            tick_size: tick size of fitting domain.
                If None, tick_size is set to 0.01.
            terms: (list or None) list of Williams terms to be used in optimization, e.g. [-1, 1, 2].

        """
        self.angle_gap = angle_gap
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.tick_size = tick_size
        self.terms = terms


class Optimization:
    """Optimization class

    Methods:
        * optimize_cjp_displacements
        * optimize_williams_displacements
        * optimize_williams_stresses

        * residuals_cjp_displacements
        * residuals_williams_displacements
        * residuals_williams_stresses

        * mse_williams_displacements
        * mse_williams_stresses

    """

    def __init__(self,
                 data: InputData,
                 material: Material = Material(),
                 options: OptimizationProperties = OptimizationProperties()):
        """Initializes Optimization arguments.

        Args:
            data: obj of class InputData
            material: obj of class Material
            options: obj of class OptimizationProperties

        """
        self.data = data
        self.material = material

        # polar grid & corresponding cartesian grid
        angle_gap = options.angle_gap
        self.min_radius = options.min_radius
        max_radius = options.max_radius
        tick_size = options.tick_size
        self.terms = np.asarray(options.terms)
        self.angle_gap_rad = angle_gap / 180 * np.pi
        self.r_grid, self.phi_grid = \
            np.mgrid[
                self.min_radius:max_radius:tick_size, -np.pi + self.angle_gap_rad:np.pi - self.angle_gap_rad:tick_size]
        self.x_grid, self.y_grid = self.make_cartesian(self.r_grid, self.phi_grid)

        # map transformed data to cartesian grid
        self._interpolate_data_on_grid()

    def _interpolate_data_on_grid(self):
        """Interpolates the data on a cartesian grid."""
        # New way using ReusableLinearInterpolator
        interpolator = ReusableLinearInterpolator(self.data.coor_x, self.data.coor_y, np.c_[0, 0])
        intp_data = interpolator.interpolate(np.c_[self.data.disp_x, self.data.disp_y, self.data.disp_z])
        disp_x_0_0, disp_y_0_0, disp_z_0_0 = intp_data[:, 0].item(), intp_data[:, 1].item(), intp_data[:, 2].item()
        logger.debug("Displacement at crack tip (0,0): u_x=%.4f, u_y=%.4f, u_z=%.4f mm", disp_x_0_0, disp_y_0_0,
                     disp_z_0_0)

        interpolator = ReusableLinearInterpolator(self.data.coor_x, self.data.coor_y,
                                                  np.c_[self.x_grid.ravel(), self.y_grid.ravel()])
        intp_data = interpolator.interpolate(np.c_[self.data.disp_x - disp_x_0_0,
                                                   self.data.disp_y - disp_y_0_0,
                                                   self.data.disp_z - disp_z_0_0])
        self.interp_disp_x = intp_data[:, 0].reshape(self.x_grid.shape)
        self.interp_disp_y = intp_data[:, 1].reshape(self.x_grid.shape)
        self.interp_disp_z = intp_data[:, 2].reshape(self.x_grid.shape)
        logger.debug("Interpolated data on grid with shape %s", self.x_grid.shape)
        pass

    def optimize_cjp_displacements_modeI(self, method='lm', init_coeffs=None):
        """Optimizes CJP displacements.

        Args:
            method: method from scipy.optimize.least_squares, defaults to 'lm' - Levenberg-Marquardt iterative algorithm
            init_coeffs: initial coefficients used for x0 in scipy.optimize.least_squares

        """
        if init_coeffs is None:
            init_coeffs = np.random.rand(5)
        else:
            init_coeffs += np.random.rand(5)
        # optimize least squares
        logger.debug("Starting CJP mode I optimization using method '%s'", method)
        logging.warning("CJP Mode I optimization is experimental and may produce unreliable results. "
                        "Use only for Mode I–dominated load cases. Interpret all outputs with caution.")

        result = optimize.least_squares(fun=self.residuals_cjp_displacements_modeI,
                                        x0=init_coeffs,
                                        method=method)
        logger.debug(
            "CJP mode I optimization completed: cost=%.6e, success=%s, nfev=%d", result.cost, result.success,
            result.nfev)
        return result

    def optimize_cjp_displacements_mixedmode(self, method='lm', init_coeffs=None):
        """Optimizes CJP displacements.

        Args:
            method: method from scipy.optimize.least_squares, defaults to 'lm' - Levenberg-Marquardt iterative algorithm
            init_coeffs: initial coefficients used for x0 in scipy.optimize.least_squares

        """
        if init_coeffs is None:
            init_coeffs = np.random.rand(5)
        else:
            init_coeffs += np.random.rand(5)

        logger.debug("Starting CJP mixedmode optimization using method '%s'", method)
        logging.warning(
            "CJP Mode I/II optimization is experimental and may produce unreliable results. "
            "Use only for Mode I–dominated load cases. Interpret all outputs with caution.")

        # optimize least squares
        result = optimize.least_squares(fun=self.residuals_cjp_displacements_mixedmode,
                                        x0=init_coeffs,
                                        method=method)

        logger.debug(
            "CJP mixedmode optimization completed: cost=%.6e, success=%s, nfev=%d", result.cost, result.success,
            result.nfev)
        return result

    def optimize_williams_displacements_xy(self, method='lm', init_coeffs=None):
        """Optimizes Williams displacements in x-y plane.

        Args:
            method: method from scipy.optimize.least_squares, defaults to 'lm' - Levenberg-Marquardt iterative algorithm
            init_coeffs: initial coefficients used for x0 in scipy.optimize.least_squares

        """
        if init_coeffs is None:
            init_coeffs = np.random.rand(2 * len(self.terms))

        logger.debug("Starting Williams 2D optimization with %d terms using method '%s'", len(self.terms), method)

        # optimize least squares
        result = optimize.least_squares(fun=self.residuals_williams_displacements,
                                        x0=init_coeffs,
                                        method=method)

        logger.debug(
            "Williams optimization completed: cost=%.6e, success=%s, nfev=%d", result.cost, result.success, result.nfev)
        return result

    def optimize_williams_displacements_z(self, method='lm', init_coeffs=None):
        """Optimizes Williams displacements in z direction.

        Args:
            method: method from scipy.optimize.least_squares, defaults to 'lm' - Levenberg-Marquardt iterative algorithm
            init_coeffs: initial coefficients used for x0 in scipy.optimize.least_squares

        """
        if init_coeffs is None:
            init_coeffs = np.random.rand(1 * len(self.terms))

        # optimize least squares
        logger.debug("Starting Williams 3D optimization with %d terms using method '%s'", len(self.terms), method)
        result = optimize.least_squares(fun=self.residuals_williams_displacements_z,
                                        x0=init_coeffs,
                                        method=method)
        logger.debug(
            "Williams 3D optimization completed: cost=%.6e, success=%s, nfev=%d", result.cost, result.success,
            result.nfev)
        return result

    def residuals_cjp_displacements_modeI(self, inp: list or np.array) -> np.ndarray:
        """Returns the residuals of CJP displacements.

        Args:
            inp: coefficients for cjp_displacement_field, Z = (A, B, C, E, F) as in Camacho-Reyes et al. 2023

        Returns:
            residual: of cjp displacements, i.e. [cjp_displacement_x - measured_displacement_x,
                                                  cjp_displacement_y - measured_displacement_y]

        """
        z = inp

        cjp_disp_x, cjp_disp_y = cjp_displ_field_modeI(z, self.phi_grid, self.r_grid, self.material)

        residual = np.asarray([cjp_disp_x - self.interp_disp_x, cjp_disp_y - self.interp_disp_y])
        residual = residual.reshape(-1)
        # filter out nan values
        residual = residual[~np.isnan(residual)]
        return residual

    def residuals_cjp_displacements_mixedmode(self, inp: Union[list, np.ndarray]) -> np.ndarray:
        """Returns the residuals of CJP displacements.

        Args:
            inp: coefficients for cjp_displacement_field, Z = (A_r, B_r, B_i, C, E) as in Christopher et al. '13

        Returns:
            residual: of cjp displacements, i.e. [cjp_displacement_x - measured_displacement_x,
                                                  cjp_displacement_y - measured_displacement_y]

        """
        z = inp

        cjp_disp_x, cjp_disp_y = cjp_displ_field_mixedmode(z, self.phi_grid, self.r_grid, self.material)

        residual = np.asarray([cjp_disp_x - self.interp_disp_x, cjp_disp_y - self.interp_disp_y])
        residual = residual.reshape(-1)
        # filter out nan values
        residual = residual[~np.isnan(residual)]
        return residual

    def residuals_williams_displacements(self, inp: Union[list, np.ndarray]) -> np.ndarray:
        """Returns the residuals of Williams displacements.

        Args:
            inp: Williams coefficients for williams_displ_field

        Returns:
            residual: of displacements calculated from the approximated Williams field and the actual results

        """
        a = inp[0:len(self.terms)]
        b = inp[len(self.terms):]

        williams_disp_x, williams_disp_y = williams_displ_field_xy(a, b, self.terms, self.phi_grid, self.r_grid,
                                                                   self.material)

        residual = np.asarray([williams_disp_x - self.interp_disp_x, williams_disp_y - self.interp_disp_y])
        residual = residual.reshape(-1)
        # filter out nan values
        residual = residual[~np.isnan(residual)]
        return residual

    def residuals_williams_displacements_z(self, inp: Union[list, np.ndarray]) -> np.ndarray:
        """Returns the residuals of Williams displacements in z direction

        Args:
            inp: Williams coefficients for williams_displ_field

        Returns:
            residual: of displacements calculated from the approximated Williams field and the actual results

        """
        c = inp

        williams_disp_z = williams_displ_field_z(c, self.terms, self.phi_grid, self.r_grid, self.material)

        residual = np.asarray([williams_disp_z - self.interp_disp_z])
        residual = residual.reshape(-1)
        # filter out nan values
        residual = residual[~np.isnan(residual)]
        return residual

    @staticmethod
    def make_cartesian(r: float, phi: float):
        """Takes polar coordinates and maps onto cartesian coordinates."""
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return x, y

    @staticmethod
    def ensure_defaults_williams(options: OptimizationProperties, crack_tip_x: float):
        """Ensures that the options for Williams optimization are set to sensible default values if None.

        Args:
            options: obj of class OptimizationProperties
            crack_tip_x: x-coordinate of crack tip (used to set min_radius if None)

        """
        if options.angle_gap is None:
            options.angle_gap = 20
        if options.min_radius is None:
            options.min_radius = abs(crack_tip_x) / 20
        if options.max_radius is None:
            options.max_radius = abs(crack_tip_x) / 5
        if options.tick_size is None:
            options.tick_size = 0.01
        if options.terms is None:
            options.terms = list(DEFAULT_WILLIAMS_OPT_TERMS)
        for i in [1, 2]:  # ensure SIFs and T can be calculated
            if i not in options.terms:
                options.terms.append(i)
                logger.info("Williams optimization terms should include %d. Term added.", i)
        options.terms.sort()
        pass
