"""Compatibility facade for configuring and executing established ODM coefficient
fits through the public SciPy-shaped optimization interface.
"""

import logging
from typing import Optional, Union

import numpy as np

from crackpy.fracture_analysis._interpolation_cache import InterpolatorCache
from crackpy.fracture_analysis.odm.assembly import (
    LinearSystem,
    assemble_cjp,
    assemble_williams,
)
from crackpy.fracture_analysis.odm.results import CoefficientFitResult
from crackpy.fracture_analysis.odm.sampling import (
    build_optimization_grid,
    prepare_interpolated_displacement_grid,
)
from crackpy.fracture_analysis.odm.solvers import (
    ResidualFunction,
    SolverRoute,
    solve_coefficient_fit,
    to_optimize_result,
)
from crackpy.input.input_data import InputData
from crackpy.structure_elements.material import Material

logger = logging.getLogger(__name__)

DEFAULT_WILLIAMS_OPT_TERMS = [-1, 1, 2, 3, 4, 5]
_INTERPOLATOR_GEOMETRY_CACHE = InterpolatorCache(max_interpolators=4)


class OptimizationProperties:
    """Configure the established polar ODM fitting domain and Williams terms.

    ``FractureAnalysis`` resolves ``None`` values, ensures the first and second
    Williams terms are present, sorts the selected terms, and mutates this
    configuration before constructing ``Optimization``.
    Direct ``Optimization`` construction consumes the values as supplied.
    """

    def __init__(
            self,
            angle_gap: Optional[float] = 20,
            min_radius: Optional[float] = None,
            max_radius: Optional[float] = None,
            tick_size: Optional[float] = 0.01,
            terms=None,
    ):
        """Store the ODM fitting-domain configuration.

        Args:
            angle_gap: Angular margin excluded on each side of the negative
                crack-parallel axis, in degrees.
                The total excluded wedge is twice this value.
                ``FractureAnalysis`` resolves ``None`` to 20 degrees.
            min_radius: Inclusive minimum fitting radius in mm.
                ``FractureAnalysis`` resolves ``None`` to one twentieth of the
                absolute crack-tip x-coordinate.
            max_radius: Exclusive maximum fitting radius in mm.
                ``FractureAnalysis`` resolves ``None`` to one fifth of the
                absolute crack-tip x-coordinate.
            tick_size: Shared established polar-grid increment, applied to
                radius in mm and angle in radians.
                ``FractureAnalysis`` resolves ``None`` to 0.01.
            terms: Williams term orders used for in-plane and out-of-plane
                fitting.
                ``FractureAnalysis`` defaults to ``[-1, 1, 2, 3, 4, 5]``,
                ensures terms 1 and 2 are present, and sorts the list in place.
        """
        self.angle_gap = angle_gap
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.tick_size = tick_size
        self.terms = terms


class Optimization:
    """Fit CJP and Williams displacement formulations through the established facade.

    The public fitting operations are
    ``optimize_cjp_displacements_modeI``,
    ``optimize_cjp_displacements_mixedmode``,
    ``optimize_williams_displacements_xy``, and
    ``optimize_williams_displacements_z``.
    Each returns a mutable normalized SciPy ``OptimizeResult`` with these
    common fields:

    - ``solver`` records the selected ``direct``, ``iterative``, or ``legacy``
      Solver Route.
    - ``x`` contains the fitted coefficient vector in the formulation-specific
      order documented by the fitting operation.
    - ``fun`` contains the final displacement-residual vector in mm.
    - ``cost`` is half the squared Euclidean residual norm in mm².
    - ``jac`` contains the residual-by-coefficient Jacobian.
    - ``rank`` and ``singular_values`` contain direct-solver matrix evidence
      and are ``None`` for iterative and legacy routes.
    - ``success``, ``message``, and ``status`` describe numerical completion.
    - ``nfev`` and ``njev`` record the available residual and Jacobian
      evaluation counts.

    The direct route solves the fixed linear system through GELSS and ignores
    ``method`` and ``init_coeffs``.
    The iterative route evaluates the same fixed matrix and exact Jacobian,
    using a zero vector when ``init_coeffs`` is absent.
    The legacy route evaluates the established residual and Jacobian callbacks,
    using a random initial vector when ``init_coeffs`` is absent.
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
        self._cjp_assembly = assemble_cjp(
            interp_disp_x=self.interp_disp_x,
            interp_disp_y=self.interp_disp_y,
            grid=self._grid,
            material=self.material,
        )
        self._cjp_target_xy = self._cjp_assembly.mode_i.target
        self._cjp_system_matrix_modeI = self._cjp_assembly.mode_i.matrix
        self._cjp_system_matrix_mixedmode = self._cjp_assembly.mixed_mode.matrix

    def _prepare_williams_optimization(self):
        """Precompute fixed Williams displacement systems on the optimization grid."""
        self._williams_assembly = assemble_williams(
            interp_disp_x=self.interp_disp_x,
            interp_disp_y=self.interp_disp_y,
            interp_disp_z=self.interp_disp_z,
            grid=self._grid,
            terms=self.terms,
            material=self.material,
        )
        self._williams_target_xy = self._williams_assembly.xy.target
        self._williams_target_z = self._williams_assembly.z.target
        self._williams_system_matrix_xy = self._williams_assembly.xy.matrix
        self._williams_system_matrix_z = self._williams_assembly.z.matrix

    @staticmethod
    def _solve_displacement_system(
            system: LinearSystem,
            *,
            solver: SolverRoute,
            method: str,
            init_coeffs: np.ndarray | None,
            residuals: ResidualFunction,
            jacobian: ResidualFunction) -> CoefficientFitResult:
        """Solve one fixed ODM displacement system."""
        solver_arguments = {}
        if solver != "direct":
            initial = None if init_coeffs is None else np.array(init_coeffs, copy=True)
            solver_arguments.update(method=method, init_coeffs=initial)
        if solver == "legacy":
            solver_arguments.update(residuals=residuals, jacobian=jacobian)
        return solve_coefficient_fit(system, solver=solver, **solver_arguments)

    def optimize_cjp_displacements_modeI(
            self, method='lm', init_coeffs=None, *, solver: SolverRoute = "direct"):
        """Fit CJP Mode I coefficients to the prepared in-plane displacements.

        Args:
            method: SciPy least-squares method used by ``iterative`` and ``legacy``.
                The ``direct`` route ignores this compatibility control.
            init_coeffs: Optional five-value initial vector in ``(A, B, C, E, F)``
                order.
                The ``iterative`` and ``legacy`` routes copy it before use, while
                ``direct`` ignores it.
            solver: Numerical route selected from ``direct``, ``iterative``, and
                ``legacy``.

        Returns:
            The normalized SciPy ``OptimizeResult`` described by the class
            contract.
            Its ``x`` field has shape ``(5,)`` in ``(A, B, C, E, F)`` order, where
            ``A``, ``B``, and ``E`` use MPa sqrt(mm) and ``C`` and ``F`` use MPa.
            Its ``fun`` field has shape ``(m,)`` and contains valid
            x-displacement residuals
            followed by valid y-displacement residuals in mm.
            Its ``jac`` field has shape ``(m, 5)`` in the same equation and
            coefficient order.

        Raises:
            ValueError: If ``solver`` is unsupported or a selected SciPy route
                rejects ``method`` or ``init_coeffs``.

        """
        return to_optimize_result(
            self._fit_cjp_displacements_modeI(method, init_coeffs, solver=solver)
        )

    def optimize_cjp_displacements_mixedmode(
            self, method='lm', init_coeffs=None, *, solver: SolverRoute = "direct"):
        """Fit CJP mixed-mode coefficients to prepared in-plane displacements.

        Args:
            method: SciPy least-squares method used by ``iterative`` and ``legacy``.
                The ``direct`` route ignores this compatibility control.
            init_coeffs: Optional five-value initial vector in
                ``(A_r, B_r, B_i, C, E)`` order.
                The ``iterative`` and ``legacy`` routes copy it before use, while
                ``direct`` ignores it.
            solver: Numerical route selected from ``direct``, ``iterative``, and
                ``legacy``.

        Returns:
            The normalized SciPy ``OptimizeResult`` described by the class
            contract.
            Its ``x`` field has shape ``(5,)`` in
            ``(A_r, B_r, B_i, C, E)`` order, where
            ``A_r``, ``B_r``, ``B_i``, and ``E`` use MPa sqrt(mm) and ``C`` uses
            MPa.
            Its ``fun`` field has shape ``(m,)`` and contains valid
            x-displacement residuals
            followed by valid y-displacement residuals in mm.
            Its ``jac`` field has shape ``(m, 5)`` in the same equation and
            coefficient order.

        Raises:
            ValueError: If ``solver`` is unsupported or a selected SciPy route
                rejects ``method`` or ``init_coeffs``.

        """
        return to_optimize_result(
            self._fit_cjp_displacements_mixedmode(method, init_coeffs, solver=solver)
        )

    def optimize_williams_displacements_xy(
            self, method='lm', init_coeffs=None, *, solver: SolverRoute = "direct"):
        """Fit in-plane Williams coefficients to prepared x/y displacements.

        Args:
            method: SciPy least-squares method used by ``iterative`` and ``legacy``.
                The ``direct`` route ignores this compatibility control.
            init_coeffs: Optional vector of length ``2 * len(terms)``.
                The ``iterative`` and ``legacy`` routes copy it before use, while
                ``direct`` ignores it.
            solver: Numerical route selected from ``direct``, ``iterative``, and
                ``legacy``.

        Returns:
            The normalized SciPy ``OptimizeResult`` described by the class
            contract.
            Its ``x`` field has shape ``(2 * len(terms),)`` with all ``a_n``
            coefficients in configured term order followed by all ``b_n``
            coefficients in that order.
            A coefficient for term ``n`` uses MPa mm**(1 - n/2).
            Its ``fun`` field has shape ``(m,)`` and contains valid
            x-displacement residuals
            followed by valid y-displacement residuals in mm.
            Its ``jac`` field has shape ``(m, 2 * len(terms))`` in the same
            ordering.

        Raises:
            ValueError: If ``solver`` is unsupported or a selected SciPy route
                rejects ``method`` or ``init_coeffs``.

        """
        return to_optimize_result(
            self._fit_williams_displacements_xy(method, init_coeffs, solver=solver)
        )

    def optimize_williams_displacements_z(
            self, method='lm', init_coeffs=None, *, solver: SolverRoute = "direct"):
        """Fit out-of-plane Williams coefficients to prepared z displacements.

        Args:
            method: SciPy least-squares method used by ``iterative`` and ``legacy``.
                The ``direct`` route ignores this compatibility control.
            init_coeffs: Optional vector of length ``len(terms)``.
                The ``iterative`` and ``legacy`` routes copy it before use, while
                ``direct`` ignores it.
            solver: Numerical route selected from ``direct``, ``iterative``, and
                ``legacy``.

        Returns:
            The normalized SciPy ``OptimizeResult`` described by the class
            contract.
            Its ``x`` field has shape ``(len(terms),)`` with ``c_n``
            coefficients in configured term order.
            A coefficient for term ``n`` uses MPa mm**(1 - n/2).
            Its ``fun`` field has shape ``(m,)`` and contains valid
            z-displacement residuals in mm.
            Its ``jac`` field has shape ``(m, len(terms))`` in the same term
            order.

        Raises:
            ValueError: If ``solver`` is unsupported or a selected SciPy route
                rejects ``method`` or ``init_coeffs``.

        """
        return to_optimize_result(
            self._fit_williams_displacements_z(method, init_coeffs, solver=solver)
        )

    def _fit_cjp_displacements_modeI(
            self, method='lm', init_coeffs=None, *, solver: SolverRoute = "direct") -> CoefficientFitResult:
        """Fit the prepared CJP Mode I displacement system."""
        logger.debug("Starting CJP mode I optimization using solver '%s' and method '%s'", solver, method)
        logging.warning("CJP Mode I optimization is experimental and may produce unreliable results. "
                        "Use only for Mode I–dominated load cases. Interpret all outputs with caution.")
        result = self._solve_displacement_system(
            self._cjp_assembly.mode_i, solver=solver, method=method, init_coeffs=init_coeffs,
            residuals=self.residuals_cjp_displacements_modeI, jacobian=self.jacobian_cjp_displacements_modeI,
        )
        logger.debug("CJP mode I optimization completed: cost=%.6e, success=%s, nfev=%d", result.cost, result.success,
                     result.nfev)
        return result

    def _fit_cjp_displacements_mixedmode(
            self, method='lm', init_coeffs=None, *, solver: SolverRoute = "direct") -> CoefficientFitResult:
        """Fit the prepared mixed-mode CJP displacement system."""
        logger.debug("Starting CJP mixedmode optimization using solver '%s' and method '%s'", solver, method)
        logging.warning("CJP Mode I/II optimization is experimental and may produce unreliable results. "
                        "Use only for Mode I–dominated load cases. Interpret all outputs with caution.")
        result = self._solve_displacement_system(
            self._cjp_assembly.mixed_mode, solver=solver, method=method, init_coeffs=init_coeffs,
            residuals=self.residuals_cjp_displacements_mixedmode, jacobian=self.jacobian_cjp_displacements_mixedmode,
        )
        logger.debug("CJP mixedmode optimization completed: cost=%.6e, success=%s, nfev=%d", result.cost, result.success,
                     result.nfev)
        return result

    def _fit_williams_displacements_xy(
            self, method='lm', init_coeffs=None, *, solver: SolverRoute = "direct") -> CoefficientFitResult:
        """Fit the prepared in-plane Williams displacement system."""
        logger.debug("Starting Williams 2D optimization with %d terms using solver '%s' and method '%s'",
                     len(self.terms), solver, method)
        result = self._solve_displacement_system(
            self._williams_assembly.xy, solver=solver, method=method, init_coeffs=init_coeffs,
            residuals=self.residuals_williams_displacements, jacobian=self.jacobian_williams_displacements,
        )
        logger.debug("Williams optimization completed: cost=%.6e, success=%s, nfev=%d", result.cost, result.success,
                     result.nfev)
        return result

    def _fit_williams_displacements_z(
            self, method='lm', init_coeffs=None, *, solver: SolverRoute = "direct") -> CoefficientFitResult:
        """Fit the prepared out-of-plane Williams displacement system."""
        logger.debug("Starting Williams 3D optimization with %d terms using solver '%s' and method '%s'",
                     len(self.terms), solver, method)
        result = self._solve_displacement_system(
            self._williams_assembly.z, solver=solver, method=method, init_coeffs=init_coeffs,
            residuals=self.residuals_williams_displacements_z, jacobian=self.jacobian_williams_displacements_z,
        )
        logger.debug("Williams 3D optimization completed: cost=%.6e, success=%s, nfev=%d", result.cost, result.success,
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
