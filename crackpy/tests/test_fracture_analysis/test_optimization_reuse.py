"""ODM assembly and facade-reuse evidence for fixed-system ordering,
masking, interpolation, and solver behavior.
"""

import inspect
import unittest
from unittest import mock

import numpy as np
from scipy import optimize

import crackpy.fracture_analysis._interpolation_cache as interpolation_cache_module
import crackpy.fracture_analysis.optimization as optimization_module
from crackpy.fracture_analysis._interpolation_cache import (
    BoundedCache,
    InterpolatorCache,
    ReusableLinearInterpolator,
    hash_array,
)
from crackpy.fracture_analysis.crack_tip import (
    cjp_displ_field_mixedmode,
    cjp_displ_field_modeI,
    williams_displ_field_xy,
    williams_displ_field_z,
)
from crackpy.fracture_analysis.crack_tip_fields.cjp import (
    cjp_mixed_mode_displacement_basis,
    cjp_mode_i_displacement_basis,
)
from crackpy.fracture_analysis.crack_tip_fields.williams import (
    williams_in_plane_displacement_basis,
    williams_out_of_plane_displacement_basis,
)
from crackpy.fracture_analysis.odm.assembly import (
    assemble_cjp,
    assemble_williams,
)
from crackpy.fracture_analysis.odm.sampling import (
    OptimizationGrid,
    build_optimization_grid,
    prepare_interpolated_displacement_grid,
)
from crackpy.fracture_analysis.optimization import Optimization, OptimizationProperties
from crackpy.fracture_analysis.utils import ReusableLinearInterpolator as CompatibilityInterpolator
from crackpy.input.input_data import InputData
from crackpy.structure_elements.material import Material


class TestReusableOdmAssembly(unittest.TestCase):
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

    def test_direct_assembly_returns_independent_williams_matrices(self):
        instance = Optimization(self.data, options=self.options)
        arguments = {
            "interp_disp_x": instance.interp_disp_x,
            "interp_disp_y": instance.interp_disp_y,
            "interp_disp_z": instance.interp_disp_z,
            "grid": instance._grid,
            "terms": instance.terms,
            "material": instance.material,
        }
        first_assembly = assemble_williams(**arguments)
        expected_xy = first_assembly.xy.matrix.copy()
        expected_z = first_assembly.z.matrix.copy()
        first_assembly.xy.matrix.fill(0.0)
        first_assembly.z.matrix.fill(0.0)

        second_assembly = assemble_williams(**arguments)
        self.assertIsNot(second_assembly.xy.matrix, first_assembly.xy.matrix)
        self.assertIsNot(second_assembly.z.matrix, first_assembly.z.matrix)
        np.testing.assert_array_equal(second_assembly.xy.matrix, expected_xy)
        np.testing.assert_array_equal(second_assembly.z.matrix, expected_z)

    def test_cjp_objective_preserves_basis_columns_x_then_y_mask_and_target(self):
        r = np.array([[0.4, 0.8], [1.2, 1.6]])
        phi = np.array([[-1.0, -0.25], [0.5, 1.25]])
        grid = OptimizationGrid(r=r, phi=phi, x=r * np.cos(phi), y=r * np.sin(phi))
        disp_x = np.array([[10.0, np.nan], [30.0, 40.0]])
        disp_y = np.array([[np.nan, 60.0], [70.0, 80.0]])
        material = Material()

        assembly = assemble_cjp(disp_x, disp_y, grid, material)

        target_xy = np.asarray([disp_x, disp_y]).reshape(-1)
        valid_mask = ~np.isnan(target_xy)
        np.testing.assert_array_equal(assembly.valid_mask_xy, valid_mask)
        np.testing.assert_array_equal(assembly.mode_i.target, target_xy[valid_mask])
        np.testing.assert_array_equal(assembly.mixed_mode.target, target_xy[valid_mask])
        for system, basis_function in (
            (assembly.mode_i, cjp_mode_i_displacement_basis),
            (assembly.mixed_mode, cjp_mixed_mode_displacement_basis),
        ):
            basis_x, basis_y = basis_function(r, phi, material)
            coefficient_first = np.concatenate(
                [basis_x.reshape(5, -1), basis_y.reshape(5, -1)],
                axis=1,
            )
            np.testing.assert_allclose(system.matrix, coefficient_first[:, valid_mask].T)

    def test_williams_objective_preserves_selected_columns_masks_and_targets(self):
        r = np.array([[0.4, 0.8], [1.2, 1.6]])
        phi = np.array([[-1.0, -0.25], [0.5, 1.25]])
        grid = OptimizationGrid(r=r, phi=phi, x=r * np.cos(phi), y=r * np.sin(phi))
        terms = np.array([2, -1, 3])
        disp_x = np.array([[10.0, np.nan], [30.0, 40.0]])
        disp_y = np.array([[50.0, 60.0], [np.nan, 80.0]])
        disp_z = np.array([[90.0, np.nan], [110.0, 120.0]])
        material = Material()

        assembly = assemble_williams(
            disp_x,
            disp_y,
            disp_z,
            grid,
            terms,
            material,
        )

        target_xy = np.asarray([disp_x, disp_y]).reshape(-1)
        valid_mask_xy = ~np.isnan(target_xy)
        target_z = disp_z.reshape(-1)
        valid_mask_z = ~np.isnan(target_z)
        np.testing.assert_array_equal(assembly.valid_mask_xy, valid_mask_xy)
        np.testing.assert_array_equal(assembly.valid_mask_z, valid_mask_z)
        np.testing.assert_array_equal(assembly.xy.target, target_xy[valid_mask_xy])
        np.testing.assert_array_equal(assembly.z.target, target_z[valid_mask_z])

        basis_x, basis_y = williams_in_plane_displacement_basis(
            r,
            phi,
            terms,
            material,
        )
        coefficient_first_xy = np.concatenate(
            [basis_x.reshape(2 * len(terms), -1), basis_y.reshape(2 * len(terms), -1)],
            axis=1,
        )
        basis_z = williams_out_of_plane_displacement_basis(r, phi, terms, material)
        np.testing.assert_allclose(
            assembly.xy.matrix,
            coefficient_first_xy[:, valid_mask_xy].T,
        )
        np.testing.assert_allclose(
            assembly.z.matrix,
            basis_z.reshape(len(terms), -1)[:, valid_mask_z].T,
        )

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

    def test_default_williams_terms_preserve_established_order(self):
        options = OptimizationProperties(terms=None)

        Optimization.ensure_defaults_williams(options, crack_tip_x=20.0)

        self.assertEqual(options.terms, [-1, 1, 2, 3, 4, 5])

    def test_public_optimizer_signatures_add_only_keyword_only_solver(self):
        expected_parameters = (
            (
                "self",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.empty,
                inspect.Parameter.empty,
            ),
            (
                "method",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                "lm",
                inspect.Parameter.empty,
            ),
            (
                "init_coeffs",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                None,
                inspect.Parameter.empty,
            ),
            (
                "solver",
                inspect.Parameter.KEYWORD_ONLY,
                "direct",
                optimization_module.SolverRoute,
            ),
        )
        optimizer_names = (
            "optimize_cjp_displacements_modeI",
            "optimize_cjp_displacements_mixedmode",
            "optimize_williams_displacements_xy",
            "optimize_williams_displacements_z",
        )
        actual_parameters = {
            name: tuple(
                (
                    parameter.name,
                    parameter.kind,
                    parameter.default,
                    parameter.annotation,
                )
                for parameter in inspect.signature(
                    getattr(Optimization, name)
                ).parameters.values()
            )
            for name in optimizer_names
        }

        self.assertEqual(
            actual_parameters,
            {name: expected_parameters for name in optimizer_names},
        )

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
