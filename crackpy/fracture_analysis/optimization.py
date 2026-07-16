import logging
from typing import Optional, Union

import numpy as np

from crackpy.fracture_analysis._interpolation_cache import InterpolatorCache
from crackpy.fracture_analysis._odm_fit_systems import (
    build_cjp_systems,
    build_polar_basis_fields,
    build_williams_systems,
    solve_linear_system,
)
from crackpy.fracture_analysis._odm_grid_interpolation import (
    build_optimization_grid,
    prepare_interpolated_displacement_grid,
)
from crackpy.input.input_data import InputData
from crackpy.structure_elements.material import Material

logger = logging.getLogger(__name__)

DEFAULT_WILLIAMS_OPT_TERMS = [-1, 1, 2, 3, 4, 5]
_INTERPOLATOR_GEOMETRY_CACHE = InterpolatorCache(max_interpolators=4)


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
        self.max_radius = options.max_radius
        self.tick_size = options.tick_size
        self.terms = np.asarray(options.terms)
        self.angle_gap_rad = angle_gap / 180 * np.pi
        self._grid = build_optimization_grid(
            min_radius=self.min_radius,
            max_radius=self.max_radius,
            tick_size=self.tick_size,
            angle_gap_deg=angle_gap,
        )
        self.r_grid = self._grid.r
        self.phi_grid = self._grid.phi
        self.x_grid = self._grid.x
        self.y_grid = self._grid.y
        self._basis = build_polar_basis_fields(self._grid, self.material)

        # map transformed data to cartesian grid
        self._interpolate_data_on_grid()
        self._prepare_cjp_optimization()
        self._prepare_williams_optimization()

    def _interpolate_data_on_grid(self):
        """Interpolates the data on a cartesian grid."""
        interpolated_grid = prepare_interpolated_displacement_grid(
            data=self.data,
            grid=self._grid,
            interpolator_cache=_INTERPOLATOR_GEOMETRY_CACHE,
        )
        self._interpolator = interpolated_grid.interpolator
        disp_x_0_0, disp_y_0_0, disp_z_0_0 = interpolated_grid.tip_displacements
        logger.debug("Displacement at crack tip (0,0): u_x=%.4f, u_y=%.4f, u_z=%.4f mm", disp_x_0_0, disp_y_0_0,
                     disp_z_0_0)
        self.interp_disp_x = interpolated_grid.disp_x
        self.interp_disp_y = interpolated_grid.disp_y
        self.interp_disp_z = interpolated_grid.disp_z
        logger.debug("Interpolated data on grid with shape %s", self.x_grid.shape)

    def _prepare_cjp_optimization(self):
        """Precompute fixed CJP displacement systems on the optimization grid."""
        self._cjp_systems = build_cjp_systems(
            interp_disp_x=self.interp_disp_x,
            interp_disp_y=self.interp_disp_y,
            grid=self._grid,
            material=self.material,
            basis=self._basis,
        )
        self._cjp_target_xy = self._cjp_systems.mode_i.target
        self._cjp_system_matrix_modeI = self._cjp_systems.mode_i.matrix
        self._cjp_system_matrix_mixedmode = self._cjp_systems.mixed_mode.matrix

    def _prepare_williams_optimization(self):
        """Precompute fixed Williams displacement systems on the optimization grid."""
        self._williams_systems = build_williams_systems(
            interp_disp_x=self.interp_disp_x,
            interp_disp_y=self.interp_disp_y,
            interp_disp_z=self.interp_disp_z,
            grid=self._grid,
            terms=self.terms,
            material=self.material,
            basis=self._basis,
        )
        self._williams_target_xy = self._williams_systems.xy.target
        self._williams_target_z = self._williams_systems.z.target
        self._williams_system_matrix_xy = self._williams_systems.xy.matrix
        self._williams_system_matrix_z = self._williams_systems.z.matrix

    def optimize_cjp_displacements_modeI(self, method='lm', init_coeffs=None):
        """Optimizes CJP displacements.

        Args:
            method: Retained for API compatibility; ignored by the direct solver.
            init_coeffs: Retained for API compatibility; ignored by the direct solver.

        Returns:
            Direct linear least-squares result for the CJP Mode I coefficients.

        """
        logger.debug("Starting CJP mode I optimization using method '%s'", method)
        logging.warning("CJP Mode I optimization is experimental and may produce unreliable results. "
                        "Use only for Mode I–dominated load cases. Interpret all outputs with caution.")

        result = solve_linear_system(self._cjp_systems.mode_i)
        logger.debug(
            "CJP mode I optimization completed: cost=%.6e, success=%s, nfev=%d", result.cost, result.success,
            result.nfev)
        return result

    def optimize_cjp_displacements_mixedmode(self, method='lm', init_coeffs=None):
        """Optimizes CJP displacements.

        Args:
            method: Retained for API compatibility; ignored by the direct solver.
            init_coeffs: Retained for API compatibility; ignored by the direct solver.

        Returns:
            Direct linear least-squares result for the CJP mixed-mode coefficients.

        """
        logger.debug("Starting CJP mixedmode optimization using method '%s'", method)
        logging.warning(
            "CJP Mode I/II optimization is experimental and may produce unreliable results. "
            "Use only for Mode I–dominated load cases. Interpret all outputs with caution.")

        result = solve_linear_system(self._cjp_systems.mixed_mode)

        logger.debug(
            "CJP mixedmode optimization completed: cost=%.6e, success=%s, nfev=%d", result.cost, result.success,
            result.nfev)
        return result

    def optimize_williams_displacements_xy(self, method='lm', init_coeffs=None):
        """Optimizes Williams displacements in x-y plane.

        Args:
            method: Retained for API compatibility; ignored by the direct solver.
            init_coeffs: Retained for API compatibility; ignored by the direct solver.

        Returns:
            Direct linear least-squares result for the in-plane Williams coefficients.

        """
        logger.debug("Starting Williams 2D optimization with %d terms using method '%s'", len(self.terms), method)
        result = solve_linear_system(self._williams_systems.xy)

        logger.debug(
            "Williams optimization completed: cost=%.6e, success=%s, nfev=%d", result.cost, result.success, result.nfev)
        return result

    def optimize_williams_displacements_z(self, method='lm', init_coeffs=None):
        """Optimizes Williams displacements in z direction.

        Args:
            method: Retained for API compatibility; ignored by the direct solver.
            init_coeffs: Retained for API compatibility; ignored by the direct solver.

        Returns:
            Direct linear least-squares result for the out-of-plane Williams coefficients.

        """
        logger.debug("Starting Williams 3D optimization with %d terms using method '%s'", len(self.terms), method)
        result = solve_linear_system(self._williams_systems.z)
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
        return self._cjp_system_matrix_modeI @ np.asarray(inp) - self._cjp_target_xy

    def residuals_cjp_displacements_mixedmode(self, inp: Union[list, np.ndarray]) -> np.ndarray:
        """Returns the residuals of CJP displacements.

        Args:
            inp: coefficients for cjp_displacement_field, Z = (A_r, B_r, B_i, C, E) as in Christopher et al. '13

        Returns:
            residual: of cjp displacements, i.e. [cjp_displacement_x - measured_displacement_x,
                                                  cjp_displacement_y - measured_displacement_y]

        """
        return self._cjp_system_matrix_mixedmode @ np.asarray(inp) - self._cjp_target_xy

    def jacobian_cjp_displacements_modeI(self, inp: Union[list, np.ndarray]) -> np.ndarray:
        """Return the constant Jacobian of CJP Mode I displacement residuals.

        Args:
            inp: CJP coefficients, unused because the objective is linear.

        Returns:
            The residual-by-coefficient CJP Mode I system matrix.
        """
        return self._cjp_system_matrix_modeI.copy()

    def jacobian_cjp_displacements_mixedmode(self, inp: Union[list, np.ndarray]) -> np.ndarray:
        """Return the constant Jacobian of CJP mixed-mode displacement residuals.

        Args:
            inp: CJP coefficients, unused because the objective is linear.

        Returns:
            The residual-by-coefficient CJP mixed-mode system matrix.
        """
        return self._cjp_system_matrix_mixedmode.copy()

    def residuals_williams_displacements(self, inp: Union[list, np.ndarray]) -> np.ndarray:
        """Returns the residuals of Williams displacements.

        Args:
            inp: Williams coefficients for williams_displ_field

        Returns:
            residual: of displacements calculated from the approximated Williams field and the actual results

        """
        return self._williams_system_matrix_xy @ np.asarray(inp) - self._williams_target_xy

    def residuals_williams_displacements_z(self, inp: Union[list, np.ndarray]) -> np.ndarray:
        """Returns the residuals of Williams displacements in z direction

        Args:
            inp: Williams coefficients for williams_displ_field

        Returns:
            residual: of displacements calculated from the approximated Williams field and the actual results

        """
        return self._williams_system_matrix_z @ np.asarray(inp) - self._williams_target_z

    def jacobian_williams_displacements(self, inp: Union[list, np.ndarray]) -> np.ndarray:
        """Return the constant Jacobian of in-plane Williams residuals.

        Args:
            inp: Williams coefficients, unused because the objective is linear.

        Returns:
            The residual-by-coefficient in-plane Williams system matrix.
        """
        return self._williams_system_matrix_xy.copy()

    def jacobian_williams_displacements_z(self, inp: Union[list, np.ndarray]) -> np.ndarray:
        """Return the constant Jacobian of out-of-plane Williams residuals.

        Args:
            inp: Williams coefficients, unused because the objective is linear.

        Returns:
            The residual-by-coefficient out-of-plane Williams system matrix.
        """
        return self._williams_system_matrix_z.copy()

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
