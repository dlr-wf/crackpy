import inspect
import unittest
from unittest import mock

import numpy as np
from scipy import optimize

import crackpy.fracture_analysis._interpolation_cache as interpolation_cache_module
import crackpy.fracture_analysis._odm_fit_systems as fit_systems_module
import crackpy.fracture_analysis.optimization as optimization_module
from crackpy.fracture_analysis._interpolation_cache import (
    BoundedCache,
    InterpolatorCache,
    ReusableLinearInterpolator,
    hash_array,
)
from crackpy.fracture_analysis._odm_fit_systems import (
    LinearizedSystem,
    build_williams_systems,
    solve_linear_system,
)
from crackpy.fracture_analysis._odm_grid_interpolation import (
    build_optimization_grid,
    prepare_interpolated_displacement_grid,
)
from crackpy.fracture_analysis.crack_tip import (
    cjp_displ_field_mixedmode,
    cjp_displ_field_modeI,
    williams_displ_field_xy,
    williams_displ_field_z,
)
from crackpy.fracture_analysis.optimization import Optimization, OptimizationProperties
from crackpy.fracture_analysis.utils import ReusableLinearInterpolator as CompatibilityInterpolator
from crackpy.input.input_data import InputData


class TestReusableOdmSystems(unittest.TestCase):
    def setUp(self):
        axis = np.linspace(-1.5, 1.5, 9)
        coor_x, coor_y = np.meshgrid(axis, axis)
        self.data = InputData()
        self.data.coor_x = coor_x.ravel()
        self.data.coor_y = coor_y.ravel()
        self.data.coor_z = np.zeros(coor_x.size)
        self.data.disp_x = (0.04 + 0.02 * coor_x - 0.01 * coor_y + 0.003 * coor_x * coor_y).ravel()
        self.data.disp_y = (-0.02 + 0.01 * coor_x + 0.03 * coor_y - 0.002 * coor_x**2).ravel()
        self.data.disp_z = (0.01 - 0.015 * coor_x + 0.005 * coor_y).ravel()
        self.options = OptimizationProperties(
            angle_gap=25,
            min_radius=0.3,
            max_radius=1.1,
            tick_size=0.2,
            terms=[-1, 1, 2],
        )
        optimization_module._INTERPOLATOR_GEOMETRY_CACHE.clear()

    @staticmethod
    def finite_difference_jacobian(residual_function, coefficients, step=1e-7):
        """Approximate a residual function's Jacobian with centered differences.

        Args:
            residual_function: Callable mapping coefficients to residuals.
            coefficients: Coefficients at which to approximate the Jacobian.
            step: Perturbation used for each centered difference.

        Returns:
            The residual-by-coefficient finite-difference Jacobian.
        """
        jacobian = np.empty((residual_function(coefficients).size, coefficients.size))
        for index in range(coefficients.size):
            delta = np.zeros_like(coefficients)
            delta[index] = step
            jacobian[:, index] = (
                residual_function(coefficients + delta) - residual_function(coefficients - delta)
            ) / (2 * step)
        return jacobian

    def legacy_residual(self, optimization_instance, fit_name, coefficients):
        """Evaluate a displacement residual through the legacy analytical field path.

        Args:
            optimization_instance: Prepared ODM optimization instance.
            fit_name: Name of the CJP or Williams formulation to evaluate.
            coefficients: Coefficients for the selected formulation.

        Returns:
            Flattened residuals after removing invalid grid equations.
        """
        if fit_name == "cjp_mode_i":
            disp_x, disp_y = cjp_displ_field_modeI(
                coefficients,
                optimization_instance.phi_grid,
                optimization_instance.r_grid,
                optimization_instance.material,
            )
            values = np.asarray(
                [disp_x - optimization_instance.interp_disp_x, disp_y - optimization_instance.interp_disp_y]
            )
        elif fit_name == "cjp_mixed_mode":
            disp_x, disp_y = cjp_displ_field_mixedmode(
                coefficients,
                optimization_instance.phi_grid,
                optimization_instance.r_grid,
                optimization_instance.material,
            )
            values = np.asarray(
                [disp_x - optimization_instance.interp_disp_x, disp_y - optimization_instance.interp_disp_y]
            )
        elif fit_name == "williams_xy":
            n_terms = len(optimization_instance.terms)
            disp_x, disp_y = williams_displ_field_xy(
                coefficients[:n_terms],
                coefficients[n_terms:],
                optimization_instance.terms,
                optimization_instance.phi_grid,
                optimization_instance.r_grid,
                optimization_instance.material,
            )
            values = np.asarray(
                [disp_x - optimization_instance.interp_disp_x, disp_y - optimization_instance.interp_disp_y]
            )
        else:
            disp_z = williams_displ_field_z(
                coefficients,
                optimization_instance.terms,
                optimization_instance.phi_grid,
                optimization_instance.r_grid,
                optimization_instance.material,
            )
            values = disp_z - optimization_instance.interp_disp_z
        flattened = values.reshape(-1)
        return flattened[~np.isnan(flattened)]

    def test_compatibility_export_owns_the_same_interpolator_class(self):
        self.assertIs(CompatibilityInterpolator, ReusableLinearInterpolator)

    def test_one_pass_interpolation_matches_the_legacy_two_pass_layout(self):
        grid = build_optimization_grid(0.3, 1.1, 0.2, 25)
        prepared = prepare_interpolated_displacement_grid(self.data, grid, InterpolatorCache())
        values = np.c_[self.data.disp_x, self.data.disp_y, self.data.disp_z]

        origin_interpolator = ReusableLinearInterpolator(
            self.data.coor_x,
            self.data.coor_y,
            np.array([[0.0, 0.0]]),
        )
        origin = origin_interpolator.interpolate(values)[0]
        grid_interpolator = ReusableLinearInterpolator(
            self.data.coor_x,
            self.data.coor_y,
            np.c_[grid.x.ravel(), grid.y.ravel()],
        )
        legacy = grid_interpolator.interpolate(values - origin)

        np.testing.assert_allclose(prepared.disp_x, legacy[:, 0].reshape(grid.x.shape))
        np.testing.assert_allclose(prepared.disp_y, legacy[:, 1].reshape(grid.y.shape))
        np.testing.assert_allclose(prepared.disp_z, legacy[:, 2].reshape(grid.x.shape))

    def test_identical_geometry_reuses_one_triangulation_and_interpolator(self):
        with mock.patch.object(
            interpolation_cache_module,
            "Delaunay",
            wraps=interpolation_cache_module.Delaunay,
        ) as delaunay:
            first = Optimization(self.data, options=self.options)
            second = Optimization(self.data, options=self.options)

        self.assertIs(first._interpolator, second._interpolator)
        self.assertEqual(delaunay.call_count, 1)

    def test_matrix_residuals_match_direct_field_evaluation(self):
        instance = Optimization(self.data, options=self.options)
        cases = (
            ("cjp_mode_i", instance.residuals_cjp_displacements_modeI, np.linspace(-0.2, 0.2, 5)),
            ("cjp_mixed_mode", instance.residuals_cjp_displacements_mixedmode, np.linspace(-0.2, 0.2, 5)),
            (
                "williams_xy",
                instance.residuals_williams_displacements,
                np.linspace(-0.2, 0.2, 2 * len(instance.terms)),
            ),
            ("williams_z", instance.residuals_williams_displacements_z, np.linspace(-0.2, 0.2, len(instance.terms))),
        )
        for fit_name, residual_function, coefficients in cases:
            with self.subTest(fit_name=fit_name):
                expected = self.legacy_residual(instance, fit_name, coefficients)
                np.testing.assert_allclose(residual_function(coefficients), expected, rtol=1e-12, atol=1e-12)

    def test_constant_jacobians_match_finite_differences(self):
        instance = Optimization(self.data, options=self.options)
        cases = (
            (
                instance.residuals_cjp_displacements_modeI,
                instance.jacobian_cjp_displacements_modeI,
                np.linspace(-0.2, 0.2, 5),
            ),
            (
                instance.residuals_cjp_displacements_mixedmode,
                instance.jacobian_cjp_displacements_mixedmode,
                np.linspace(-0.2, 0.2, 5),
            ),
            (
                instance.residuals_williams_displacements,
                instance.jacobian_williams_displacements,
                np.linspace(-0.2, 0.2, 2 * len(instance.terms)),
            ),
            (
                instance.residuals_williams_displacements_z,
                instance.jacobian_williams_displacements_z,
                np.linspace(-0.2, 0.2, len(instance.terms)),
            ),
        )
        for residual_function, jacobian_function, coefficients in cases:
            with self.subTest(residual_function=residual_function.__name__):
                expected = self.finite_difference_jacobian(residual_function, coefficients)
                np.testing.assert_allclose(jacobian_function(coefficients), expected, rtol=1e-7, atol=1e-9)

    def test_equivalent_instances_own_independent_williams_matrices(self):
        first = Optimization(self.data, options=self.options)
        second = Optimization(self.data, options=self.options)

        self.assertIsNot(first._williams_system_matrix_xy, second._williams_system_matrix_xy)
        self.assertIsNot(first._williams_system_matrix_z, second._williams_system_matrix_z)
        np.testing.assert_array_equal(first._williams_system_matrix_xy, second._williams_system_matrix_xy)
        np.testing.assert_array_equal(first._williams_system_matrix_z, second._williams_system_matrix_z)

    def test_direct_builder_returns_independent_williams_matrices(self):
        instance = Optimization(self.data, options=self.options)
        arguments = {
            "interp_disp_x": instance.interp_disp_x,
            "interp_disp_y": instance.interp_disp_y,
            "interp_disp_z": instance.interp_disp_z,
            "grid": instance._grid,
            "terms": instance.terms,
            "material": instance.material,
            "basis": instance._basis,
        }
        first = build_williams_systems(**arguments)
        expected_xy = first.xy.matrix.copy()
        expected_z = first.z.matrix.copy()
        first.xy.matrix.fill(0.0)
        first.z.matrix.fill(0.0)

        second = build_williams_systems(**arguments)
        self.assertIsNot(second.xy.matrix, first.xy.matrix)
        self.assertIsNot(second.z.matrix, first.z.matrix)
        np.testing.assert_array_equal(second.xy.matrix, expected_xy)
        np.testing.assert_array_equal(second.z.matrix, expected_z)

    def test_mutating_public_jacobians_does_not_change_later_residuals(self):
        instance = Optimization(self.data, options=self.options)
        cases = (
            (
                instance.jacobian_cjp_displacements_modeI,
                instance.residuals_cjp_displacements_modeI,
                np.linspace(-0.2, 0.2, 5),
            ),
            (
                instance.jacobian_cjp_displacements_mixedmode,
                instance.residuals_cjp_displacements_mixedmode,
                np.linspace(-0.2, 0.2, 5),
            ),
            (
                instance.jacobian_williams_displacements,
                instance.residuals_williams_displacements,
                np.linspace(-0.2, 0.2, 2 * len(instance.terms)),
            ),
        )
        for jacobian_function, residual_function, coefficients in cases:
            with self.subTest(jacobian=jacobian_function.__name__):
                expected = residual_function(coefficients)

                jacobian_function(coefficients).fill(0.0)

                np.testing.assert_allclose(residual_function(coefficients), expected)

    def test_mutating_result_jacobian_does_not_poison_later_williams_fit(self):
        first = Optimization(self.data, options=self.options)
        second = Optimization(self.data, options=self.options)
        expected = second.optimize_williams_displacements_z()

        first.optimize_williams_displacements_z().jac.fill(0.0)
        actual = second.optimize_williams_displacements_z()

        np.testing.assert_allclose(actual.x, expected.x)
        np.testing.assert_allclose(actual.fun, expected.fun)
        self.assertAlmostEqual(actual.cost, expected.cost)

    def test_all_direct_optimizers_match_legacy_iterative_results(self):
        instance = Optimization(self.data, options=self.options)
        cases = (
            ("cjp_mode_i", instance.optimize_cjp_displacements_modeI, 5),
            ("cjp_mixed_mode", instance.optimize_cjp_displacements_mixedmode, 5),
            ("williams_xy", instance.optimize_williams_displacements_xy, 2 * len(instance.terms)),
            ("williams_z", instance.optimize_williams_displacements_z, len(instance.terms)),
        )
        for fit_name, direct_optimizer, coefficient_count in cases:
            with self.subTest(fit_name=fit_name):
                initial = np.linspace(0.1, 0.5, coefficient_count)
                expected = optimize.least_squares(
                    lambda coefficients: self.legacy_residual(instance, fit_name, coefficients),
                    x0=initial,
                    method="lm",
                )
                actual = direct_optimizer(method="trf", init_coeffs=initial.copy())
                np.testing.assert_allclose(actual.x, expected.x, rtol=1e-4, atol=1e-6)
                np.testing.assert_allclose(actual.fun, expected.fun, rtol=1e-6, atol=1e-9)
                self.assertAlmostEqual(actual.cost, expected.cost, places=12)

    def test_array_hash_separates_changed_inputs(self):
        self.assertNotEqual(hash_array(np.array([1.0, 2.0])), hash_array(np.array([1.0, 3.0])))

    def test_public_optimizer_signatures_are_unchanged(self):
        expected = "(self, method='lm', init_coeffs=None)"
        optimizer_names = (
            "optimize_cjp_displacements_modeI",
            "optimize_cjp_displacements_mixedmode",
            "optimize_williams_displacements_xy",
            "optimize_williams_displacements_z",
        )
        for name in optimizer_names:
            with self.subTest(name=name):
                self.assertEqual(str(inspect.signature(getattr(Optimization, name))), expected)

    def test_direct_solver_uses_gelss_and_returns_the_intentional_contract(self):
        system = LinearizedSystem(
            matrix=np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]),
            target=np.array([1.0, 4.0, 3.0]),
        )
        with mock.patch(
            "crackpy.fracture_analysis._odm_fit_systems.linalg.lstsq",
            wraps=fit_systems_module.linalg.lstsq,
        ) as least_squares:
            result = solve_linear_system(system)

        self.assertEqual(least_squares.call_args.kwargs["lapack_driver"], "gelss")
        self.assertTrue(result.success)
        self.assertEqual(result.status, 1)
        self.assertEqual(result.nfev, 1)
        self.assertEqual(result.njev, 1)
        self.assertEqual(result.message, "Solved by direct linear least squares.")
        np.testing.assert_allclose(result.fun, system.matrix @ result.x - system.target)
        np.testing.assert_array_equal(result.jac, system.matrix)

    def test_bounded_cache_evicts_the_least_recently_used_entry(self):
        cache = BoundedCache(max_size=2)
        cache.set("first", 1)
        cache.set("second", 2)
        self.assertEqual(cache.get("first"), 1)
        cache.set("third", 3)

        self.assertIsNone(cache.get("second"))
        self.assertEqual(cache.get("first"), 1)
        self.assertEqual(cache.get("third"), 3)


if __name__ == "__main__":
    unittest.main()
