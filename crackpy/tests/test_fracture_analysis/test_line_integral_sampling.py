"""Direct behavior tests for line-integral field preparation."""

import numpy as np
import pytest

from crackpy.fracture_analysis._interpolation_cache import InterpolatorCache
from crackpy.fracture_analysis.line_integrals.auxiliary_field_preparation import (
    evaluate_shifted_auxiliary_fields,
    prepare_lefm_auxiliary_fields,
    prepare_zhao_auxiliary_fields,
)
from crackpy.fracture_analysis.line_integrals.contours import (
    build_rectangular_integration_contour,
    prepare_integration_contour_geometry,
)
from crackpy.fracture_analysis.line_integrals.mode_decomposition import (
    prepare_mode_data,
    reconstruct_in_plane_strains,
    reconstruct_mode_iii_fields,
)
from crackpy.fracture_analysis.line_integrals.sampling import (
    RegularGridDisplacements,
    interpolate_reference_value,
    mask_contour_data,
    sample_in_plane_fields,
    sample_mode_iii_fields,
    sample_regular_grid_displacements,
)
from crackpy.fracture_analysis.line_integration import (
    IntegrationPath,
    LineIntegral,
    PathProperties,
)
from crackpy.input.input_data import InputData
from crackpy.structure_elements.material import Material


def _measured_fields() -> InputData:
    x, y = np.meshgrid(np.linspace(-2.0, 2.0, 9), np.linspace(-2.0, 2.0, 9))
    data = InputData()
    data.coor_x, data.coor_y = x.ravel(), y.ravel()
    data.disp_x = 2.0 * data.coor_x + data.coor_y
    data.disp_y = 3.0 * data.coor_x - 2.0 * data.coor_y
    data.disp_z = data.coor_x - data.coor_y
    data.eps_x = np.full_like(data.coor_x, 2.0)
    data.eps_y = np.full_like(data.coor_x, -2.0)
    data.eps_xy = np.full_like(data.coor_x, 2.0)
    data.eps_xz = data.coor_x + 2.0 * data.coor_y
    data.eps_yz = 3.0 * data.coor_x - data.coor_y
    data.eps_vm = np.ones_like(data.coor_x)
    data.sig_x = 5.0 * data.coor_x + data.coor_y
    data.sig_y = data.coor_x - data.coor_y
    data.sig_xy = 2.0 * data.coor_x + 4.0 * data.coor_y
    data.sigma_xz = 7.0 * data.coor_x + data.coor_y
    data.sigma_yz = data.coor_x - 8.0 * data.coor_y
    data.sig_vm = np.ones_like(data.coor_x)
    return data


def _geometry():
    contour = build_rectangular_integration_contour(
        origin_x=0.0,
        origin_y=0.0,
        size_left=-1.0,
        size_right=1.0,
        size_bottom=-1.0,
        size_top=1.0,
        tick_size=0.5,
        number_of_nodes=None,
        top_offset=0.0,
        bottom_offset=0.0,
    )
    return prepare_integration_contour_geometry(contour, x_shift=contour.tick_size)


def test_affine_in_plane_sampling_is_owned_and_immutable():
    data = _measured_fields()
    # Deliberately make the displacement-derived du_x/dx inconsistent with the
    # measured eps_x field to pin the established scientific input convention.
    data.disp_x = 9.0 * data.coor_x + data.coor_y
    geometry = _geometry()
    sampled = sample_in_plane_fields(
        data,
        geometry,
        mask_tolerance=None,
        interpolator_cache=InterpolatorCache(max_interpolators=4),
    )

    np.testing.assert_allclose(sampled.strain[:, 0, 0], 2.0)
    np.testing.assert_allclose(sampled.strain[:, 1, 1], -2.0)
    np.testing.assert_allclose(sampled.strain[:, 0, 1], 2.0)
    np.testing.assert_allclose(sampled.displacement_gradient_x[:, 0], 2.0)
    np.testing.assert_allclose(sampled.displacement_gradient_x[:, 1], 3.0)
    np.testing.assert_allclose(
        sampled.shifted.positive_x_displacement_y - sampled.shifted.negative_x_displacement_y,
        3.0,
    )
    for values in (
        sampled.strain,
        sampled.stress,
        sampled.displacement,
        sampled.displacement_gradient_x,
        sampled.shifted.base,
        sampled.shifted.positive_x_displacement_y,
        sampled.shifted.negative_x_displacement_y,
    ):
        assert not values.flags.writeable


def test_masking_preserves_unmasked_identity_and_outside_hull_nans():
    data = _measured_fields()
    geometry = _geometry()
    assert mask_contour_data(data, geometry.evaluation_points, None) is data
    masked = mask_contour_data(data, geometry.evaluation_points, 0.75)
    assert masked is not data
    assert len(masked.coor_x) < len(data.coor_x)

    outside = interpolate_reference_value(
        data,
        data.sig_x,
        np.asarray([[10.0, 10.0]]),
        interpolator_cache=InterpolatorCache(max_interpolators=4),
    )
    assert outside.shape == ()
    assert np.isnan(outside)
    assert not outside.flags.writeable


def test_mode_iii_sampling_preserves_column_order_and_explicit_masking():
    data = _measured_fields()
    geometry = _geometry()
    cache = InterpolatorCache(max_interpolators=4)

    unmasked = sample_mode_iii_fields(
        data,
        geometry.evaluation_points,
        mask_tolerance=None,
        interpolator_cache=cache,
    )
    explicitly_masked = sample_mode_iii_fields(
        data,
        geometry.evaluation_points,
        mask_tolerance=0.75,
        interpolator_cache=cache,
    )

    x = geometry.evaluation_points[:, 0]
    y = geometry.evaluation_points[:, 1]
    np.testing.assert_allclose(
        unmasked.out_of_plane_displacement_derivative_x,
        x + 2.0 * y,
    )
    np.testing.assert_allclose(
        unmasked.out_of_plane_displacement_derivative_y,
        3.0 * x - y,
    )
    np.testing.assert_allclose(unmasked.shear_stress_xz, 7.0 * x + y)
    np.testing.assert_allclose(unmasked.shear_stress_yz, x - 8.0 * y)
    np.testing.assert_allclose(
        explicitly_masked.out_of_plane_displacement_derivative_x,
        unmasked.out_of_plane_displacement_derivative_x,
    )
    for payload in (unmasked, explicitly_masked):
        for field_name in payload.__dataclass_fields__:
            assert not getattr(payload, field_name).flags.writeable


def test_regular_grid_preserves_independent_extents_xy_layout_and_c_flattening():
    data = _measured_fields()
    evaluation_points = np.asarray(
        [
            [-1.5, -1.0],
            [1.5, -1.0],
            [1.5, 1.0],
            [-1.5, 1.0],
        ]
    )

    sampled = sample_regular_grid_displacements(
        data,
        evaluation_points,
        3,
        interpolator_cache=InterpolatorCache(max_interpolators=4),
    )

    np.testing.assert_allclose(sampled.x_coordinates, [-1.8, 0.0, 1.8])
    np.testing.assert_allclose(sampled.y_coordinates, [-1.2, 0.0, 1.2])
    np.testing.assert_allclose(sampled.x_mesh[0], sampled.x_coordinates)
    np.testing.assert_allclose(sampled.y_mesh[:, 0], sampled.y_coordinates)
    np.testing.assert_array_equal(
        sampled.evaluation_points,
        np.c_[sampled.x_mesh.ravel(order="C"), sampled.y_mesh.ravel(order="C")],
    )
    np.testing.assert_allclose(
        sampled.displacement_x_mesh,
        2.0 * sampled.x_mesh + sampled.y_mesh,
    )
    np.testing.assert_allclose(
        sampled.displacement_y_mesh,
        3.0 * sampled.x_mesh - 2.0 * sampled.y_mesh,
    )
    np.testing.assert_allclose(
        sampled.displacement_z_mesh,
        sampled.x_mesh - sampled.y_mesh,
    )
    for field_name in sampled.__dataclass_fields__:
        assert not getattr(sampled, field_name).flags.writeable


def test_high_level_interpolation_requires_an_explicit_cache():
    with pytest.raises(TypeError, match="interpolator_cache"):
        sample_in_plane_fields(  # type: ignore[call-arg]
            _measured_fields(),
            _geometry(),
            mask_tolerance=None,
        )


def test_auxiliary_payloads_preserve_batching_parity_calls_and_ownership(monkeypatch):
    geometry = _geometry()
    material = Material(E=72000, nu_xy=0.33)
    relative_points = geometry.evaluation_points
    shifted = evaluate_shifted_auxiliary_fields(
        lambda points: np.asarray([points[:, 0], points[:, 1]]),
        geometry,
    )
    for field_name in shifted.__dataclass_fields__:
        assert not getattr(shifted, field_name).flags.writeable

    from crackpy.fracture_analysis.line_integrals import auxiliary_field_preparation

    lefm_calls = []
    real_lefm = auxiliary_field_preparation.get_crack_nearfield

    def counted_lefm(*args):
        lefm_calls.append(args)
        return real_lefm(*args)

    monkeypatch.setattr(auxiliary_field_preparation, "get_crack_nearfield", counted_lefm)
    lefm = prepare_lefm_auxiliary_fields(
        1.0,
        0.0,
        geometry,
        material=material,
    )
    assert len(lefm_calls) == 3
    assert lefm.stress.shape == (len(relative_points), 2, 2)
    assert lefm.strain.shape == (len(relative_points), 2, 2)

    zhao_calls = []
    real_zhao = auxiliary_field_preparation.get_zhao_solutions

    def counted_zhao(*args):
        zhao_calls.append(args)
        return real_zhao(*args)

    monkeypatch.setattr(auxiliary_field_preparation, "get_zhao_solutions", counted_zhao)
    zhao = prepare_zhao_auxiliary_fields(
        geometry,
        material=material,
    )
    assert len(zhao_calls) == 3
    assert zhao.stress.shape == (len(relative_points), 2, 2)
    for payload in (lefm, zhao):
        for field_name in payload.__dataclass_fields__:
            assert not getattr(payload, field_name).flags.writeable


def test_translated_auxiliary_derivative_uses_exact_nonbinary_shift(monkeypatch):
    contour = build_rectangular_integration_contour(
        origin_x=100_000_000.0,
        origin_y=0.0,
        size_left=-1.0,
        size_right=1.0,
        size_bottom=-1.0,
        size_top=1.0,
        tick_size=0.5,
        number_of_nodes=None,
        top_offset=0.0,
        bottom_offset=0.0,
    )
    shift = 0.1
    geometry = prepare_integration_contour_geometry(contour, x_shift=shift)

    def linear_zhao(radius, angle, material):
        relative_x = radius * np.cos(angle)
        zeros = np.zeros_like(relative_x)
        return zeros, zeros, zeros, relative_x, 2.0 * relative_x

    from crackpy.fracture_analysis.line_integrals import auxiliary_field_preparation

    monkeypatch.setattr(auxiliary_field_preparation, "get_zhao_solutions", linear_zhao)
    fields = prepare_zhao_auxiliary_fields(
        geometry,
        material=Material(E=72000, nu_xy=0.33),
    )
    expected_x_derivative = (
        geometry.relative_positive_x_shifted_evaluation_points[:, 0] - geometry.relative_negative_x_shifted_evaluation_points[:, 0]
    ) / (2.0 * shift)

    assert geometry.x_shift == shift
    np.testing.assert_allclose(
        fields.displacement_gradient_x[:, 0],
        expected_x_derivative,
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        fields.displacement_gradient_x[:, 1],
        2.0 * expected_x_derivative,
        rtol=0.0,
        atol=4e-15,
    )


def test_reconstruction_preserves_square_grid_values_and_two_row_gap():
    steps = 8
    x_coordinates = np.arange(steps, dtype=float) * 2.0
    x_mesh, y_mesh = np.meshgrid(x_coordinates, x_coordinates, indexing="xy")
    u_x = 3.0 * x_mesh + 5.0 * y_mesh
    u_y = 7.0 * x_mesh + 11.0 * y_mesh

    strains = reconstruct_in_plane_strains(
        u_x,
        u_y,
        x_coordinates,
        x_coordinates,
        gap=2,
    )
    np.testing.assert_allclose(strains.strain_x[:, steps // 2 :], 3.0)
    np.testing.assert_allclose(strains.strain_y[:, steps // 2 :], 11.0)
    np.testing.assert_allclose(strains.strain_xy[:, steps // 2 :], 6.0)
    np.testing.assert_array_equal(strains.strain_x[2:6, :4], 0.0)
    np.testing.assert_array_equal(strains.strain_y[2:6, :4], 0.0)
    np.testing.assert_array_equal(strains.strain_xy[2:6, :4], 0.0)

    mode_iii = reconstruct_mode_iii_fields(
        13.0 * x_mesh + 17.0 * y_mesh,
        x_coordinates,
        x_coordinates,
        material=Material(E=72000, nu_xy=0.33),
        gap=2,
    )
    np.testing.assert_allclose(
        mode_iii.out_of_plane_displacement_derivative_x[:, 4:],
        13.0,
    )
    np.testing.assert_allclose(
        mode_iii.out_of_plane_displacement_derivative_y[:, 4:],
        17.0,
    )
    np.testing.assert_array_equal(
        mode_iii.out_of_plane_displacement_derivative_y[2:6, :4],
        0.0,
    )


def test_mode_reconstruction_uses_each_regular_grid_axis_spacing():
    steps = 8
    x_coordinates = np.arange(steps, dtype=float) * 2.0
    y_coordinates = np.arange(steps, dtype=float) * 3.0
    x_mesh, y_mesh = np.meshgrid(x_coordinates, y_coordinates, indexing="xy")
    material = Material(E=72000, nu_xy=0.3)
    regular_grid = RegularGridDisplacements(
        x_coordinates=x_coordinates,
        y_coordinates=y_coordinates,
        x_mesh=x_mesh,
        y_mesh=y_mesh,
        evaluation_points=np.c_[x_mesh.ravel(), y_mesh.ravel()],
        displacement_x_mesh=3.0 * x_mesh + 5.0 * y_mesh,
        displacement_y_mesh=7.0 * x_mesh + 11.0 * y_mesh,
        displacement_z_mesh=13.0 * x_mesh + 17.0 * y_mesh,
    )
    valid = np.zeros((steps, steps), dtype=bool)
    valid[:2, :4] = True
    valid[6:, :4] = True
    valid[:, 4:] = True

    mode_i = prepare_mode_data("I", regular_grid, material=material)
    mode_ii = prepare_mode_data("II", regular_grid, material=material)
    mode_iii = prepare_mode_data("III", regular_grid, material=material)
    reconstructed_mode_iii = reconstruct_mode_iii_fields(
        regular_grid.displacement_z_mesh,
        x_coordinates,
        y_coordinates,
        material=material,
        gap=2,
    )

    np.testing.assert_allclose(mode_i.eps_x.reshape(steps, steps)[valid], 3.0)
    np.testing.assert_allclose(mode_i.eps_y.reshape(steps, steps)[valid], 11.0)
    np.testing.assert_allclose(mode_ii.eps_xy.reshape(steps, steps)[valid], 6.0)
    np.testing.assert_allclose(mode_iii.eps_xz.reshape(steps, steps)[valid], 0.0)
    np.testing.assert_allclose(mode_iii.eps_yz.reshape(steps, steps)[valid], 17.0)
    np.testing.assert_allclose(
        mode_iii.sigma_yz.reshape(steps, steps)[valid],
        material.G * 17.0,
    )
    np.testing.assert_allclose(
        reconstructed_mode_iii.out_of_plane_displacement_derivative_x[valid],
        13.0,
    )
    np.testing.assert_allclose(
        reconstructed_mode_iii.out_of_plane_displacement_derivative_y[valid],
        17.0,
    )
    np.testing.assert_allclose(
        reconstructed_mode_iii.shear_stress_xz[valid],
        material.G * 13.0,
    )
    np.testing.assert_allclose(
        reconstructed_mode_iii.shear_stress_yz[valid],
        material.G * 17.0,
    )
    for values in (
        mode_i.eps_x,
        mode_i.eps_y,
        mode_ii.eps_xy,
        mode_iii.eps_xz,
        mode_iii.eps_yz,
    ):
        np.testing.assert_array_equal(values.reshape(steps, steps)[~valid], 0.0)


def test_public_j_decomposition_matches_rectangular_grid_affine_field():
    material = Material(E=72000, nu_xy=0.3)
    coefficient_b = 0.001
    coefficient_c = 0.002
    coefficient_e = 0.003
    x_mesh, y_mesh = np.meshgrid(
        np.linspace(-3.0, 3.0, 25),
        np.linspace(-2.0, 2.0, 21),
        indexing="xy",
    )
    data = InputData()
    data.coor_x = x_mesh.ravel()
    data.coor_y = y_mesh.ravel()
    data.disp_x = coefficient_c * data.coor_y
    data.disp_y = coefficient_b * data.coor_y
    data.disp_z = coefficient_e * data.coor_y
    data.eps_x = np.zeros_like(data.coor_x)
    data.eps_y = np.full_like(data.coor_x, coefficient_b)
    data.eps_xy = np.full_like(data.coor_x, coefficient_c / 2.0)
    data.eps_xz = np.zeros_like(data.coor_x)
    data.eps_yz = np.full_like(data.coor_x, coefficient_e)
    data.calc_eps_vm()
    data.calc_stresses(material)
    data.sigma_xz = np.zeros_like(data.coor_x)
    data.sigma_yz = np.full_like(data.coor_x, material.G * coefficient_e)

    bottom_offset = -0.2
    top_offset = 0.2
    path = IntegrationPath(
        path_properties=PathProperties(
            size_left=-1.0,
            size_right=2.0,
            size_bottom=-1.0,
            size_top=1.0,
            tick_size=0.1,
            num_nodes=None,
            top_offset=top_offset,
            bottom_offset=bottom_offset,
        )
    )
    line_integral = LineIntegral(path, data, material)
    line_integral.integrate_j_decompose()

    open_height = top_offset - bottom_offset
    plane_stress_modulus = material.E / (1.0 - material.nu_xy**2)
    expected_j_i = open_height * plane_stress_modulus * coefficient_b**2 / 2.0
    expected_j_ii = open_height * material.G * coefficient_c**2 / 2.0
    expected_j_iii = open_height * material.G * coefficient_e**2
    expected_k_i = np.sqrt(expected_j_i * material.E / 1000.0)
    expected_k_ii = np.sqrt(expected_j_ii * material.E / 1000.0)
    expected_k_iii = np.sqrt(expected_j_iii * material.E / ((1.0 + material.nu_xy) * 1000.0))

    np.testing.assert_allclose(
        (
            line_integral.decomp_j_integral_I,
            line_integral.decomp_j_integral_II,
            line_integral.decomp_j_integral_III,
            line_integral.decomp_j_integral_K_I,
            line_integral.decomp_j_integral_K_II,
            line_integral.decomp_j_integral_K_III,
        ),
        (
            expected_j_i,
            expected_j_ii,
            expected_j_iii,
            expected_k_i,
            expected_k_ii,
            expected_k_iii,
        ),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("mode", ["I", "II", "III"])
def test_prepare_mode_data_preserves_symmetry_gap_and_fresh_mutable_result(mode):
    coordinates = np.linspace(-3.5, 3.5, 8)
    x_mesh, y_mesh = np.meshgrid(coordinates, coordinates, indexing="xy")
    displacement_x = x_mesh + 2.0 * y_mesh
    displacement_y = 3.0 * x_mesh + 4.0 * y_mesh
    displacement_z = 5.0 * x_mesh + 6.0 * y_mesh
    material = Material(E=72000, nu_xy=0.33)

    regular_grid = RegularGridDisplacements(
        x_coordinates=coordinates,
        y_coordinates=coordinates,
        x_mesh=x_mesh,
        y_mesh=y_mesh,
        evaluation_points=np.c_[x_mesh.ravel(), y_mesh.ravel()],
        displacement_x_mesh=displacement_x,
        displacement_y_mesh=displacement_y,
        displacement_z_mesh=displacement_z,
    )
    first = prepare_mode_data(mode, regular_grid, material=material)
    second = prepare_mode_data(mode, regular_grid, material=material)

    assert first is not second
    assert first.disp_x.flags.writeable
    if mode == "I":
        np.testing.assert_allclose(first.disp_x.reshape(8, 8), np.flipud(first.disp_x.reshape(8, 8)))
        np.testing.assert_allclose(first.disp_y.reshape(8, 8), -np.flipud(first.disp_y.reshape(8, 8)))
    elif mode == "II":
        np.testing.assert_allclose(first.disp_x.reshape(8, 8), -np.flipud(first.disp_x.reshape(8, 8)))
        np.testing.assert_allclose(first.disp_y.reshape(8, 8), np.flipud(first.disp_y.reshape(8, 8)))
    else:
        np.testing.assert_allclose(first.disp_z.reshape(8, 8), -np.flipud(first.disp_z.reshape(8, 8)))
    np.testing.assert_array_equal(first.eps_y.reshape(8, 8)[2:6, :4], 0.0)


def test_prepare_mode_data_calculates_von_mises_before_stresses(monkeypatch):
    calls = []
    calculate_von_mises = InputData.calc_eps_vm
    calculate_stresses = InputData.calc_stresses

    def record_von_mises(data):
        calls.append("calc_eps_vm")
        return calculate_von_mises(data)

    def record_stresses(data, material):
        calls.append("calc_stresses")
        return calculate_stresses(data, material)

    monkeypatch.setattr(InputData, "calc_eps_vm", record_von_mises)
    monkeypatch.setattr(InputData, "calc_stresses", record_stresses)
    coordinates = np.linspace(-3.5, 3.5, 8)
    x_mesh, y_mesh = np.meshgrid(coordinates, coordinates, indexing="xy")
    zeros = np.zeros_like(x_mesh)
    regular_grid = RegularGridDisplacements(
        x_coordinates=coordinates,
        y_coordinates=coordinates,
        x_mesh=x_mesh,
        y_mesh=y_mesh,
        evaluation_points=np.c_[x_mesh.ravel(), y_mesh.ravel()],
        displacement_x_mesh=zeros,
        displacement_y_mesh=zeros,
        displacement_z_mesh=zeros,
    )

    prepare_mode_data(
        "I",
        regular_grid,
        material=Material(E=72000, nu_xy=0.33),
    )

    assert calls == ["calc_eps_vm", "calc_stresses"]


def test_mode_reconstruction_documents_its_scientific_source():
    for scientific_operation, equation in (
        (prepare_mode_data, "equation 3"),
        (reconstruct_in_plane_strains, "equation 6"),
        (reconstruct_mode_iii_fields, "equations 9--11"),
    ):
        documentation = scientific_operation.__doc__
        assert documentation is not None
        assert equation in documentation
        assert "https://doi.org/10.1111/str.12166" in documentation
        assert "molteno_becker_2015_j_integral_decomposition" in documentation
