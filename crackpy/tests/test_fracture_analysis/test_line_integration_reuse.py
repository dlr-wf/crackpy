from types import SimpleNamespace
from unittest import mock

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import Delaunay

from crackpy.fracture_analysis import analysis as analysis_module
from crackpy.fracture_analysis import line_integration as line_integration_module
from crackpy.fracture_analysis._interpolation_cache import (
    InterpolationTarget,
    InterpolatorCache,
    ReusableLinearInterpolator,
)
from crackpy.fracture_analysis.analysis import FractureAnalysis
from crackpy.fracture_analysis.line_integration import (
    IntegralProperties,
    IntegrationPath,
    LineIntegral,
    PathProperties,
)
from crackpy.input.crack_tip_info import CrackTipInfo
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
    data.eps_xz = np.full_like(data.coor_x, 1.0)
    data.eps_yz = np.full_like(data.coor_x, -1.0)
    data.eps_vm = np.ones_like(data.coor_x)
    data.sig_x = 5.0 * data.coor_x + data.coor_y
    data.sig_y = data.coor_x - data.coor_y
    data.sig_xy = 2.0 * data.coor_x + 4.0 * data.coor_y
    data.sigma_xz = np.full_like(data.coor_x, 7.0)
    data.sigma_yz = np.full_like(data.coor_x, -8.0)
    data.sig_vm = np.ones_like(data.coor_x)
    return data


def _line_integral(interpolator_cache: InterpolatorCache | None = None) -> LineIntegral:
    path = IntegrationPath(
        path_properties=PathProperties(-1.0, 1.0, -1.0, 1.0, 0.5, None, 0.0, 0.0)
    )
    return LineIntegral(
        path,
        _measured_fields(),
        Material(E=72000, nu_xy=0.33),
        interpolator_cache=interpolator_cache,
    )


def _analysis() -> FractureAnalysis:
    properties = IntegralProperties(
        number_of_paths=2,
        integral_tick_size=1.0,
        integral_size_left=-1.0,
        integral_size_right=1.0,
        integral_size_bottom=-1.0,
        integral_size_top=1.0,
        top_offset=0.0,
        bottom_offset=0.0,
        paths_distance_left=1.0,
        paths_distance_right=1.0,
        paths_distance_top=1.0,
        paths_distance_bottom=1.0,
        buckner_williams_terms=[1],
    )
    return FractureAnalysis(
        material=Material(E=72000, nu_xy=0.33),
        nodemap="synthetic",
        data=InputData(),
        crack_tip_info=CrackTipInfo(0.0, 0.0, 0.0, "right"),
        integral_properties=properties,
        optimization_properties=None,
    )


def _fake_line_integral(cache_observations: list[InterpolatorCache | None]):
    def factory(*args, interpolator_cache=None, **kwargs):
        cache_observations.append(interpolator_cache)
        return SimpleNamespace(
            integrate_all=lambda: None,
            j_integral=1.0,
            sif_k_j=1.0,
            sif_k_i=1.0,
            sif_k_ii=1.0,
            t_stress_chen=1.0,
            t_stress_sdm=1.0,
            t_stress_int=1.0,
            decomp_j_integral_I=1.0,
            decomp_j_integral_II=1.0,
            decomp_j_integral_III=1.0,
            decomp_j_integral_K_I=1.0,
            decomp_j_integral_K_II=1.0,
            decomp_j_integral_K_III=1.0,
            williams_a_n=[1.0],
            williams_b_n=[1.0],
            williams_coefficients=[[1, 1.0, 1.0]],
            np_integration_points=np.asarray([[0.0, 0.0, 1.0, 0.0]]),
            integration_path=SimpleNamespace(
                path_properties=SimpleNamespace(number_of_nodes=1, tick_size=1.0)
            ),
        )

    return factory


def test_analysis_cache_is_shared_within_one_contour_set_only():
    first_run = []
    second_run = []

    with mock.patch.object(
        analysis_module.line_integration,
        "LineIntegral",
        side_effect=_fake_line_integral(first_run),
    ):
        _analysis()._run_line_integrals()
    with mock.patch.object(
        analysis_module.line_integration,
        "LineIntegral",
        side_effect=_fake_line_integral(second_run),
    ):
        _analysis()._run_line_integrals()

    assert len(first_run) == len(second_run) == 2
    assert isinstance(first_run[0], InterpolatorCache)
    assert first_run[0] is first_run[1]
    assert second_run[0] is second_run[1]
    assert first_run[0] is not second_run[0]


def test_line_integral_uses_injected_cache_and_isolates_direct_defaults():
    cache = InterpolatorCache(max_interpolators=4)

    with mock.patch.object(cache, "get_interpolator", wraps=cache.get_interpolator) as get_interpolator:
        injected = _line_integral(cache)

    first_default = _line_integral()
    second_default = _line_integral()

    assert injected._interpolator_cache is cache
    assert get_interpolator.called
    assert first_default._interpolator_cache is not second_default._interpolator_cache
    assert first_default._interpolator_cache._source_triangles.max_size == 4
    assert first_default._interpolator_cache._interpolators.max_size == 4


def test_cache_identity_uses_complete_source_and_target_content():
    data = _measured_fields()
    cache = InterpolatorCache(max_interpolators=4)
    points = np.asarray([[0.0, 0.0], [0.5, 0.5]])

    first = cache.get_interpolator(
        data.coor_x,
        data.coor_y,
        points,
        InterpolationTarget.INTEGRATION_POINTS,
    )
    equivalent = cache.get_interpolator(
        data.coor_x.copy(),
        data.coor_y.copy(),
        points.copy(),
        InterpolationTarget.INTEGRATION_POINTS,
    )
    changed_source = cache.get_interpolator(
        data.coor_x + 0.1,
        data.coor_y,
        points,
        InterpolationTarget.INTEGRATION_POINTS,
    )
    changed_target = cache.get_interpolator(
        data.coor_x,
        data.coor_y,
        points + 0.1,
        InterpolationTarget.INTEGRATION_POINTS,
    )

    assert equivalent is first
    assert changed_source is not first
    assert changed_target is not first


def test_shared_analysis_cache_reuses_source_triangulation_across_contours():
    cache = InterpolatorCache(max_interpolators=4)

    with mock.patch(
        "crackpy.fracture_analysis._interpolation_cache.Delaunay",
        wraps=Delaunay,
    ) as delaunay:
        _line_integral(cache)
        second_path = IntegrationPath(
            path_properties=PathProperties(-1.5, 1.5, -1.5, 1.5, 0.5, None, 0.0, 0.0)
        )
        LineIntegral(
            second_path,
            _measured_fields(),
            Material(E=72000, nu_xy=0.33),
            interpolator_cache=cache,
        )

    assert delaunay.call_count == 1


def test_cache_evicts_source_and_target_working_sets_independently():
    data = _measured_fields()
    source_cache = InterpolatorCache(max_interpolators=4)

    with mock.patch(
        "crackpy.fracture_analysis._interpolation_cache.Delaunay",
        wraps=Delaunay,
    ) as delaunay:
        for offset in range(5):
            source_cache.get_interpolator(
                data.coor_x + offset,
                data.coor_y,
                np.asarray([[offset, 0.0]]),
                InterpolationTarget.REFERENCE_POINT,
            )
        source_cache.get_interpolator(
            data.coor_x,
            data.coor_y,
            np.asarray([[0.0, 0.0]]),
            InterpolationTarget.REFERENCE_POINT,
        )

    target_cache = InterpolatorCache(max_interpolators=4)
    first_points = np.asarray([[0.0, 0.0]])
    first = target_cache.get_interpolator(
        data.coor_x,
        data.coor_y,
        first_points,
        InterpolationTarget.REFERENCE_POINT,
    )
    for offset in range(1, 5):
        target_cache.get_interpolator(
            data.coor_x,
            data.coor_y,
            np.asarray([[offset / 10.0, 0.0]]),
            InterpolationTarget.REFERENCE_POINT,
        )
    recreated = target_cache.get_interpolator(
        data.coor_x,
        data.coor_y,
        first_points,
        InterpolationTarget.REFERENCE_POINT,
    )

    assert delaunay.call_count == 6
    assert recreated is not first


def test_batched_sampling_and_geometry_match_fixed_point_routes():
    line_integral = _line_integral()
    data = line_integral.data
    tri = Delaunay(np.c_[data.coor_x, data.coor_y])
    base = ReusableLinearInterpolator(
        data.coor_x,
        data.coor_y,
        line_integral._integration_eval_points,
        tri=tri,
    ).interpolate(
        np.c_[
            data.eps_x,
            data.eps_y,
            data.eps_xy,
            data.sig_x,
            data.sig_y,
            data.sig_xy,
            data.disp_x,
            data.disp_y,
        ]
    )
    positive = griddata(
        (data.coor_x, data.coor_y),
        data.disp_y,
        (line_integral._integration_eval_points_pos[:, 0], line_integral._integration_eval_points_pos[:, 1]),
        method="linear",
    )
    negative = griddata(
        (data.coor_x, data.coor_y),
        data.disp_y,
        (line_integral._integration_eval_points_neg[:, 0], line_integral._integration_eval_points_neg[:, 1]),
        method="linear",
    )

    actual_base = np.c_[
        line_integral.interpolated_eps_x,
        line_integral.interpolated_eps_y,
        line_integral.interpolated_eps_xy,
        line_integral.interpolated_sig_x,
        line_integral.interpolated_sig_y,
        line_integral.interpolated_sig_xy,
        line_integral.interpolated_disp_x,
        line_integral.interpolated_disp_y,
    ]
    np.testing.assert_allclose(actual_base, base, rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(
        line_integral.interpolated_disp_y_dx_positive,
        positive,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        line_integral.interpolated_disp_y_dx_negative,
        negative,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )

    expected_sizes = np.linalg.norm(line_integral.np_integration_points[:, 2:4], axis=1)
    expected_normals = []
    for point in line_integral.np_integration_points:
        normal = np.cross([point[2], point[3], 0.0], [0.0, 0.0, 1.0])
        expected_normals.append((normal / np.linalg.norm(normal))[:2])
    np.testing.assert_allclose(line_integral._path_elem_sizes, expected_sizes)
    np.testing.assert_allclose(line_integral._path_elem_heights, line_integral.np_integration_points[:, 3])
    np.testing.assert_allclose(line_integral._path_normals, expected_normals)


def test_batched_masked_sampling_matches_fixed_routes_at_hull_boundary():
    path = IntegrationPath(
        path_properties=PathProperties(-2.0, 2.0, -2.0, 2.0, 0.5, None, 0.0, 0.0)
    )
    line_integral = LineIntegral(
        path,
        _measured_fields(),
        Material(E=72000, nu_xy=0.33),
        mask_tol=0.75,
    )
    masked_data = line_integral._get_masked_data()
    measured_fields = np.c_[
        masked_data.eps_x,
        masked_data.eps_y,
        masked_data.eps_xy,
        masked_data.sig_x,
        masked_data.sig_y,
        masked_data.sig_xy,
        masked_data.disp_x,
        masked_data.disp_y,
    ]
    expected_base = griddata(
        (masked_data.coor_x, masked_data.coor_y),
        measured_fields,
        line_integral._integration_eval_points,
        method="linear",
    )
    expected_positive = griddata(
        (masked_data.coor_x, masked_data.coor_y),
        masked_data.disp_y,
        line_integral._integration_eval_points_pos,
        method="linear",
    )
    expected_negative = griddata(
        (masked_data.coor_x, masked_data.coor_y),
        masked_data.disp_y,
        line_integral._integration_eval_points_neg,
        method="linear",
    )
    actual_base = np.c_[
        line_integral.interpolated_eps_x,
        line_integral.interpolated_eps_y,
        line_integral.interpolated_eps_xy,
        line_integral.interpolated_sig_x,
        line_integral.interpolated_sig_y,
        line_integral.interpolated_sig_xy,
        line_integral.interpolated_disp_x,
        line_integral.interpolated_disp_y,
    ]

    assert np.isnan(expected_positive).any()
    assert np.isnan(expected_negative).any()
    np.testing.assert_allclose(actual_base, expected_base, rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(
        line_integral.interpolated_disp_y_dx_positive,
        expected_positive,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        line_integral.interpolated_disp_y_dx_negative,
        expected_negative,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


def test_outside_hull_nan_mask_and_reference_point_match_scipy():
    data = _measured_fields()
    cache = InterpolatorCache(max_interpolators=4)
    points = np.asarray([[0.0, 0.0], [10.0, 10.0]])
    actual = cache.get_interpolator(
        data.coor_x,
        data.coor_y,
        points,
        InterpolationTarget.REFERENCE_POINT,
    ).interpolate(data.sig_x)
    expected = griddata((data.coor_x, data.coor_y), data.sig_x, points, method="linear")

    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)

    line_integral = _line_integral(cache)
    reference_expected = griddata(
        (line_integral.data.coor_x, line_integral.data.coor_y),
        line_integral.data.sig_x - line_integral.data.sig_y,
        line_integral._reference_eval_point,
        method="linear",
    ).item()
    assert line_integral._interpolate_on_reference_point(
        line_integral.data.sig_x - line_integral.data.sig_y
    ) == reference_expected
    line_integral.integrate_t_sdm()
    assert isinstance(line_integral.t_stress_sdm, np.ndarray)
    assert line_integral.t_stress_sdm.shape == ()


def test_mode_iii_decomposition_skips_unused_in_plane_interpolation():
    line_integral = _line_integral()

    with (
        mock.patch.object(line_integral, "_map_displacement_data_on_regular_grid"),
        mock.patch.object(line_integral, "_prepare_mode_data", return_value=line_integral.data),
        mock.patch.object(line_integral, "_solve_j_integral", return_value=1.0),
        mock.patch.object(line_integral, "_solve_j_integral_III", return_value=1.0),
        mock.patch.object(
            line_integral,
            "_interpolate_on_integration_points",
            wraps=line_integral._interpolate_on_integration_points,
        ) as interpolate_xy,
    ):
        line_integral.integrate_j_decompose()

    assert interpolate_xy.call_count == 3


def test_batched_auxiliary_fields_match_scalar_evaluation():
    line_integral = _line_integral()

    with mock.patch.object(
        line_integration_module,
        "get_crack_nearfield",
        wraps=line_integration_module.get_crack_nearfield,
    ) as crack_nearfield:
        actual_stress, actual_strain, actual_disp_y_dx = (
            line_integral._get_auxiliary_crack_nearfield(1.0, 0.0)
        )
    expected_stress = []
    expected_strain = []
    expected_disp_y_dx = []
    for point in line_integral.np_integration_points:
        r, phi = line_integral._make_polar(point[0] - line_integral.origin_x, point[1] - line_integral.origin_y)
        stress, strain, _ = line_integration_module.get_crack_nearfield(
            1.0,
            0.0,
            r,
            phi,
            line_integral.material,
        )
        expected_stress.append(stress)
        expected_strain.append(strain)
        r_pos, phi_pos = line_integral._make_polar(
            point[0] - line_integral.origin_x + line_integral.x_shift,
            point[1] - line_integral.origin_y,
        )
        r_neg, phi_neg = line_integral._make_polar(
            point[0] - line_integral.origin_x - line_integral.x_shift,
            point[1] - line_integral.origin_y,
        )
        disp_pos = line_integration_module.get_crack_nearfield(
            1.0, 0.0, r_pos, phi_pos, line_integral.material
        )[2]
        disp_neg = line_integration_module.get_crack_nearfield(
            1.0, 0.0, r_neg, phi_neg, line_integral.material
        )[2]
        expected_disp_y_dx.append((disp_pos[1] - disp_neg[1]) / (2 * line_integral.x_shift))

    np.testing.assert_allclose(actual_stress, expected_stress, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_strain, expected_strain, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_disp_y_dx, expected_disp_y_dx, rtol=1e-12, atol=1e-12)
    assert crack_nearfield.call_count == 4

    with mock.patch.object(
        line_integration_module,
        "get_zhao_solutions",
        wraps=line_integration_module.get_zhao_solutions,
    ) as zhao:
        actual_zhao_stress, actual_disp_x_dx, actual_zhao_disp_y_dx = (
            line_integral._get_auxiliary_zhao_fields()
        )
    expected_zhao_stress = []
    expected_disp_x_dx = []
    expected_zhao_disp_y_dx = []
    for point in line_integral.np_integration_points:
        r, phi = line_integral._make_polar(point[0] - line_integral.origin_x, point[1] - line_integral.origin_y)
        sigma_x, sigma_y, sigma_xy, _, _ = line_integration_module.get_zhao_solutions(
            r, phi, line_integral.material
        )
        expected_zhao_stress.append([[sigma_x, sigma_xy], [sigma_xy, sigma_y]])
        r_pos, phi_pos = line_integral._make_polar(
            point[0] - line_integral.origin_x + line_integral.x_shift,
            point[1] - line_integral.origin_y,
        )
        r_neg, phi_neg = line_integral._make_polar(
            point[0] - line_integral.origin_x - line_integral.x_shift,
            point[1] - line_integral.origin_y,
        )
        positive = line_integration_module.get_zhao_solutions(r_pos, phi_pos, line_integral.material)
        negative = line_integration_module.get_zhao_solutions(r_neg, phi_neg, line_integral.material)
        expected_disp_x_dx.append((positive[3] - negative[3]) / (2 * line_integral.x_shift))
        expected_zhao_disp_y_dx.append((positive[4] - negative[4]) / (2 * line_integral.x_shift))

    np.testing.assert_allclose(actual_zhao_stress, expected_zhao_stress, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_disp_x_dx, expected_disp_x_dx, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_zhao_disp_y_dx, expected_zhao_disp_y_dx, rtol=1e-12, atol=1e-12)
    assert zhao.call_count == 3


def test_vectorized_reductions_preserve_scalar_reference_values():
    line_integral = _line_integral()
    line_integral._interpolate_on_integration_points_z()

    with mock.patch.object(
        line_integration_module.np,
        "cross",
        side_effect=AssertionError("reduction recomputed contour normals"),
    ):
        np.testing.assert_allclose(line_integral._solve_j_integral(), -52.0, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(line_integral._solve_j_integral_III(), 0.0, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            line_integral._solve_interaction_integral(1.0, 0.0),
            0.01862851303942028,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            line_integral._solve_interaction_integral(0.0, 1.0),
            0.02804922537164556,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            line_integral._solve_t_stress_interaction_integral(),
            0.7276903234995177,
            rtol=1e-12,
            atol=1e-12,
        )

    for n, a_n, b_n, expected in [
        (-1, 1, 0, 3.9031449104139266),
        (-1, 0, 1, 7.691090910936801),
        (-2, 1, 0, 8.654033468756356),
    ]:
        with mock.patch.object(
            line_integration_module,
            "eigenfunction",
            wraps=line_integration_module.eigenfunction,
        ) as eigenfunction:
            actual = line_integral._solve_chen_integral(n=n, a_n=a_n, b_n=b_n)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        assert eigenfunction.call_count == 1


def test_regular_grid_mode_iii_masking_and_assignable_state_remain_compatible():
    path = IntegrationPath(
        path_properties=PathProperties(-1.0, 1.0, -1.0, 1.0, 0.5, None, 0.0, 0.0)
    )
    line_integral = LineIntegral(
        path,
        _measured_fields(),
        Material(E=72000, nu_xy=0.33),
        mask_tol=0.75,
    )
    line_integral._map_displacement_data_on_regular_grid(grid_points=10)
    grid_points = np.c_[line_integral.x_mesh.ravel(), line_integral.y_mesh.ravel()]
    expected_grid = ReusableLinearInterpolator(
        line_integral.data.coor_x,
        line_integral.data.coor_y,
        grid_points,
    ).interpolate(
        np.c_[line_integral.data.disp_x, line_integral.data.disp_y, line_integral.data.disp_z]
    ).reshape(line_integral.x_mesh.shape + (3,))
    np.testing.assert_allclose(line_integral.disp_u_mesh, expected_grid[:, :, 0], equal_nan=True)
    np.testing.assert_allclose(line_integral.disp_v_mesh, expected_grid[:, :, 1], equal_nan=True)
    np.testing.assert_allclose(line_integral.disp_w_mesh, expected_grid[:, :, 2], equal_nan=True)

    line_integral.data = line_integral._prepare_mode_data("III")
    with mock.patch.object(
        line_integral,
        "_get_masked_data",
        wraps=line_integral._get_masked_data,
    ) as masked_data:
        line_integral._interpolate_on_integration_points_z()
        assert masked_data.call_count == 0
        line_integral._interpolate_on_integration_points_z(mask_tol=0.75)
        assert masked_data.call_count == 1

    assignable_attributes = [
        "grid_points",
        "x_coordinates",
        "y_coordinates",
        "x_mesh",
        "y_mesh",
        "disp_u_mesh",
        "disp_v_mesh",
        "disp_w_mesh",
        "interpolated_eps_x",
        "interpolated_eps_y",
        "interpolated_eps_xy",
        "interpolated_sig_x",
        "interpolated_sig_y",
        "interpolated_sig_xy",
        "interpolated_disp_x",
        "interpolated_disp_y",
        "interpolated_disp_y_dx_positive",
        "interpolated_disp_y_dx_negative",
        "interpolated_disp_y_dx",
        "interpolated_eps_xz",
        "interpolated_eps_yz",
        "interpolated_sigma_xz",
        "interpolated_sigma_yz",
    ]
    marker = np.asarray([123.0])
    for attribute in assignable_attributes:
        setattr(line_integral, attribute, marker)
        assert getattr(line_integral, attribute) is marker
