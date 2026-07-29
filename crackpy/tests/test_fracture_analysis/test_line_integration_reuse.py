"""Compatibility and ownership regressions cover repeated line-integral execution."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
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
from crackpy.fracture_analysis.crack_tip_fields.williams import (
    quantities as williams_quantities,
)
from crackpy.fracture_analysis.crack_tip_fields.williams.coefficients import (
    WilliamsInPlaneCoefficients,
)
from crackpy.fracture_analysis.line_integrals import (
    ContourWiseLineIntegralResult,
    IntegrationContourResultGeometry,
    LineIntegralQuantities,
)
from crackpy.fracture_analysis.line_integrals import runners as line_integral_runners
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
        bueckner_williams_terms=[1],
    )
    return FractureAnalysis(
        material=Material(E=72000, nu_xy=0.33),
        nodemap="synthetic",
        data=InputData(),
        crack_tip_info=CrackTipInfo(0.0, 0.0, 0.0, "right"),
        integral_properties=properties,
        optimization_properties=None,
    )


def _fake_execution(cache_observations: list[InterpolatorCache | None]):
    result = _contour_result()

    def factory(contour, data, material, mask_tolerance, terms, interpolator_cache):
        cache_observations.append(interpolator_cache)
        return SimpleNamespace(evaluate_all=lambda: result)

    return factory


def _contour_result(value: float = 1.0) -> ContourWiseLineIntegralResult:
    return ContourWiseLineIntegralResult(
        geometry=IntegrationContourResultGeometry(
            size_left=-value,
            size_right=value,
            size_bottom=-value,
            size_top=value,
            integration_points=((0.0, 0.0),),
            number_of_nodes=1,
            tick_size=1.0,
        ),
        quantities=LineIntegralQuantities(*([value] * 13)),
        williams_coefficients=WilliamsInPlaneCoefficients(
            terms=(1,), a_n=(value,), b_n=(value,)
        ),
    )


def test_analysis_cache_is_shared_within_one_contour_set_only():
    first_run = []
    second_run = []

    with mock.patch.object(
        analysis_module,
        "_LineIntegralExecution",
        side_effect=_fake_execution(first_run),
    ):
        _analysis()._run_line_integrals()
    with mock.patch.object(
        analysis_module,
        "_LineIntegralExecution",
        side_effect=_fake_execution(second_run),
    ):
        _analysis()._run_line_integrals()

    assert len(first_run) == len(second_run) == 2
    assert isinstance(first_run[0], InterpolatorCache)
    assert first_run[0] is first_run[1]
    assert second_run[0] is second_run[1]
    assert first_run[0] is not second_run[0]


def test_public_analysis_run_projects_completed_results_and_aggregate_containers():
    analysis = _analysis()
    observations = []

    with (
        mock.patch.object(
            analysis_module,
            "_LineIntegralExecution",
            side_effect=_fake_execution(observations),
        ),
        mock.patch.object(
            analysis_module,
            "derive_williams_in_plane_fracture_quantities",
            wraps=analysis_module.derive_williams_in_plane_fracture_quantities,
        ) as derive_quantities,
    ):
        analysis.run()

    assert len(analysis.contour_results) == 2
    assert len(analysis.path_results) == 2
    assert isinstance(analysis.path_results[0], list)
    assert isinstance(analysis.path_results[0][5], np.ndarray)
    assert analysis.path_results[0][5].shape == ()
    assert isinstance(analysis.williams_int, np.ndarray)
    assert isinstance(analysis.williams_int_a_n, np.ndarray)
    assert isinstance(analysis.williams_int_b_n, np.ndarray)
    assert set(analysis.sifs_int) == {"mean", "median", "rej_out_mean"}
    assert analysis.sifs_int["mean"]["j"] == 1.0
    assert derive_quantities.call_count == 3


def test_public_analysis_run_updates_an_initially_empty_progress_mapping():
    analysis = _analysis()
    progress = {}

    with mock.patch.object(
        analysis_module,
        "_LineIntegralExecution",
        side_effect=_fake_execution([]),
    ):
        analysis.run(progress_bar=progress, task_id=7)

    assert progress == {7: {"progress": 2, "total": 2}}


def test_second_public_analysis_run_preserves_established_failure_and_partial_state():
    analysis = _analysis()
    observations = []
    factory = _fake_execution(observations)

    with mock.patch.object(
        analysis_module,
        "_LineIntegralExecution",
        side_effect=factory,
    ):
        analysis.run()
        with pytest.raises(AttributeError, match="has no attribute 'append'"):
            analysis.run()

    assert len(analysis.contour_results) == 3
    assert len(analysis.path_results) == 3
    assert analysis.williams_int_a_n.shape == (2, 1)
    assert analysis.williams_int_b_n.shape == (2, 1)


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


def test_integrate_j_uses_reassigned_data_like_fresh_facade():
    reused = _line_integral()
    reused.integrate_j()
    original_j_integral = reused.j_integral

    replacement_data = _measured_fields()
    replacement_data.sig_x *= 2.0
    replacement_data.sig_y *= 2.0
    replacement_data.sig_xy *= 2.0
    fresh = LineIntegral(
        reused.integration_path,
        replacement_data,
        reused.material,
    )
    reused.data = replacement_data

    reused.integrate_j()
    fresh.integrate_j()

    assert fresh.j_integral != pytest.approx(original_j_integral)
    assert reused.j_integral == pytest.approx(fresh.j_integral)
    assert reused.sif_k_j == pytest.approx(fresh.sif_k_j)


def test_integrate_all_returns_completed_result_and_projects_mutable_facade():
    line_integral = _line_integral()
    line_integral.bueckner_williams_terms = [1, 3]

    result = line_integral.integrate_all()

    assert isinstance(result, ContourWiseLineIntegralResult)
    assert result.quantities.j_integral == line_integral.j_integral
    assert result.quantities.t_stress_sdm == line_integral.t_stress_sdm
    assert result.williams_coefficients.terms == (1, 3)
    assert line_integral.williams_coefficients == [
        list(values)
        for values in zip(
            result.williams_coefficients.terms,
            result.williams_coefficients.a_n,
            result.williams_coefficients.b_n,
        )
    ]
    assert isinstance(line_integral.t_stress_sdm, np.ndarray)
    assert line_integral.t_stress_sdm.shape == ()


def test_integrate_all_adopts_samples_from_reassigned_data():
    line_integral = _line_integral()
    original_samples = line_integral._in_plane_samples
    replacement_data = _measured_fields()
    replacement_data.disp_x = replacement_data.disp_x + 11.0
    replacement_data.disp_y = replacement_data.disp_y - 13.0
    replacement_data.eps_x = replacement_data.eps_x + 17.0
    replacement_data.sig_x = replacement_data.sig_x + 19.0
    line_integral.data = replacement_data

    line_integral.integrate_all()

    adopted_samples = line_integral._execution.in_plane_samples
    assert adopted_samples is line_integral._in_plane_samples
    assert adopted_samples is not original_samples
    np.testing.assert_allclose(
        line_integral.interpolated_eps_x,
        adopted_samples.shifted.base[:, 0],
    )
    np.testing.assert_allclose(
        line_integral.interpolated_sig_x,
        adopted_samples.shifted.base[:, 3],
    )
    np.testing.assert_allclose(
        line_integral.interpolated_disp_x,
        adopted_samples.shifted.base[:, 6],
    )
    np.testing.assert_allclose(
        line_integral.interpolated_disp_y_dx,
        adopted_samples.displacement_gradient_x[:, 1],
    )


def test_integrate_j_decompose_synchronizes_reassigned_facade_inputs():
    line_integral = _line_integral()
    replacement_data = _measured_fields()
    replacement_material = Material(E=81000, nu_xy=0.29)
    line_integral.data = replacement_data
    line_integral.material = replacement_material
    line_integral.mask_tol = 0.75

    def evaluate_reassigned_data():
        assert line_integral._execution.data is replacement_data
        assert line_integral._execution.material is replacement_material
        assert line_integral._execution.mask_tolerance == 0.75
        return 1.0, 2.0, 3.0, 4.0, 5.0, 6.0

    with mock.patch.object(
        line_integral._execution,
        "evaluate_j_decomposition",
        side_effect=evaluate_reassigned_data,
    ) as evaluate:
        line_integral.integrate_j_decompose()

    evaluate.assert_called_once_with()
    assert line_integral._execution.data is replacement_data
    assert line_integral._execution.material is replacement_material
    assert line_integral._execution.mask_tolerance == 0.75


def test_integrate_i_t_and_sdm_synchronize_reassigned_facade_data_and_material():
    line_integral = _line_integral()
    replacement_data = _measured_fields()
    replacement_material = Material(E=81000, nu_xy=0.29)
    line_integral.data = replacement_data
    line_integral.material = replacement_material
    line_integral.mask_tol = 0.75

    def evaluate_reassigned_interaction_t_stress():
        assert line_integral._execution.data is replacement_data
        assert line_integral._execution.material is replacement_material
        assert line_integral._execution.mask_tolerance == 0.75
        return 12.0

    with mock.patch.object(
        line_integral._execution,
        "evaluate_interaction_t_stress",
        side_effect=evaluate_reassigned_interaction_t_stress,
    ):
        line_integral.integrate_i_t()
    assert line_integral._execution.data is replacement_data
    assert line_integral._execution.material is replacement_material
    assert line_integral._execution.mask_tolerance == 0.75

    newer_data = _measured_fields()
    line_integral.data = newer_data

    def evaluate_reassigned_stress_difference():
        assert line_integral._execution.data is newer_data
        return 13.0

    with mock.patch.object(
        line_integral._execution,
        "evaluate_stress_difference_t_stress",
        side_effect=evaluate_reassigned_stress_difference,
    ):
        line_integral.integrate_t_sdm()
    assert line_integral._execution.data is newer_data
    assert line_integral.t_stress_sdm.shape == ()


def test_integrate_all_synchronizes_all_reassigned_facade_inputs():
    line_integral = _line_integral()
    replacement_data = _measured_fields()
    replacement_material = Material(E=81000, nu_xy=0.29)
    line_integral.data = replacement_data
    line_integral.material = replacement_material
    line_integral.mask_tol = 0.75
    line_integral.bueckner_williams_terms = [1, 3]

    def evaluate_reassigned_all():
        assert line_integral._execution.data is replacement_data
        assert line_integral._execution.material is replacement_material
        assert line_integral._execution.mask_tolerance == 0.75
        assert line_integral._execution.requested_bueckner_williams_terms == [1, 3]
        return _contour_result()

    with mock.patch.object(
        line_integral._execution,
        "evaluate_all",
        side_effect=evaluate_reassigned_all,
    ):
        line_integral.integrate_all()

    assert line_integral._execution.data is replacement_data
    assert line_integral._execution.material is replacement_material
    assert line_integral._execution.mask_tolerance == 0.75
    assert line_integral._execution.requested_bueckner_williams_terms == [1, 3]


def test_remaining_individual_integrals_synchronize_mutable_facade_inputs():
    line_integral = _line_integral()
    replacement_data = _measured_fields()
    replacement_material = Material(E=81000, nu_xy=0.29)
    line_integral.data = replacement_data
    line_integral.material = replacement_material
    line_integral.mask_tol = 0.75
    line_integral.bueckner_williams_terms = [1]

    with mock.patch.object(
        line_integral._execution,
        "evaluate_j_integral",
        return_value=(1.0, 2.0),
    ):
        line_integral.integrate_j()
    with mock.patch.object(
        line_integral._execution,
        "evaluate_interaction_sifs",
        return_value=(3.0, 4.0),
    ):
        line_integral.integrate_i_k1_k2()
    with mock.patch.object(
        line_integral._execution,
        "evaluate_bueckner_chen",
        return_value=(
            WilliamsInPlaneCoefficients(terms=(1,), a_n=(5.0,), b_n=(6.0,)),
            7.0,
        ),
    ):
        line_integral.integrate_bueckner_chen()

    assert line_integral._execution.data is replacement_data
    assert line_integral._execution.material is replacement_material
    assert line_integral._execution.mask_tolerance == 0.75
    assert line_integral._execution.requested_bueckner_williams_terms == [1]


def test_runner_and_facade_unit_conversions_delegate_to_williams_owner():
    line_integral = _line_integral()
    with (
        mock.patch.object(
            williams_quantities,
            "williams_coefficient_m_to_mm",
            wraps=williams_quantities.williams_coefficient_m_to_mm,
        ) as to_mm,
        mock.patch.object(
            williams_quantities,
            "williams_coefficient_mm_to_m",
            wraps=williams_quantities.williams_coefficient_mm_to_m,
        ) as to_m,
    ):
        assert LineIntegral._unit_m_to_mm(2.0, n=3) == 2.0 * 1000 ** (1 - 3 / 2)
        assert LineIntegral._unit_mm_to_m(2.0, n=3) == 2.0 / 1000 ** (1 - 3 / 2)
        line_integral._execution.evaluate_interaction_sifs()

    assert to_mm.call_args_list == [
        mock.call(2.0, term=3),
        mock.call(1.0),
    ]
    to_m.assert_called_once_with(2.0, term=3)


def test_direct_bueckner_projection_uses_compatibility_helpers():
    line_integral = _line_integral()
    line_integral.bueckner_williams_terms = [1]
    coefficients = WilliamsInPlaneCoefficients(
        terms=(1,),
        a_n=(5.0,),
        b_n=(6.0,),
    )

    with (
        mock.patch.object(
            line_integration_module,
            "mutable_williams_a_n",
            wraps=line_integration_module.mutable_williams_a_n,
        ) as project_a_n,
        mock.patch.object(
            line_integration_module,
            "mutable_williams_b_n",
            wraps=line_integration_module.mutable_williams_b_n,
        ) as project_b_n,
        mock.patch.object(
            line_integration_module,
            "mutable_williams_coefficients",
            wraps=line_integration_module.mutable_williams_coefficients,
        ) as project_coefficients,
        mock.patch.object(
            line_integral._execution,
            "evaluate_bueckner_chen",
            return_value=(coefficients, 7.0),
        ),
    ):
        line_integral.integrate_bueckner_chen()

    project_a_n.assert_called_once_with(coefficients)
    project_b_n.assert_called_once_with(coefficients)
    project_coefficients.assert_called_once_with(coefficients)


def test_integrate_all_distinguishes_disabled_and_term_two_williams_results():
    disabled = _line_integral()
    disabled.bueckner_williams_terms = None
    disabled_result = disabled.integrate_all()

    including_two = _line_integral()
    including_two.bueckner_williams_terms = [2]
    including_two_result = including_two.integrate_all()

    assert disabled_result.williams_coefficients is None
    assert disabled_result.quantities.t_stress_chen is None
    assert including_two_result.williams_coefficients.terms == (2,)
    assert np.isfinite(including_two_result.quantities.t_stress_chen)


@pytest.mark.parametrize(
    ("terms", "expected_calls", "expected_t_stress"),
    [
        (
            [1, 2],
            [
                mock.call(symmetric_auxiliary_amplitude=1, term=1),
                mock.call(antisymmetric_auxiliary_amplitude=1, term=1),
                mock.call(symmetric_auxiliary_amplitude=1, term=2),
                mock.call(antisymmetric_auxiliary_amplitude=1, term=2),
            ],
            12.0,
        ),
        (
            [1, 3],
            [
                mock.call(symmetric_auxiliary_amplitude=1, term=1),
                mock.call(antisymmetric_auxiliary_amplitude=1, term=1),
                mock.call(symmetric_auxiliary_amplitude=1, term=3),
                mock.call(antisymmetric_auxiliary_amplitude=1, term=3),
                mock.call(symmetric_auxiliary_amplitude=1, term=2),
            ],
            20.0,
        ),
    ],
)
def test_bueckner_chen_reuses_requested_symmetric_term_two_coefficient(
    terms,
    expected_calls,
    expected_t_stress,
):
    execution = _line_integral()._execution
    with mock.patch.object(
        execution,
        "_williams_coefficient",
        side_effect=[1.0, 2.0, 3.0, 4.0, 5.0],
    ) as coefficient:
        coefficients, t_stress = execution.evaluate_bueckner_chen(terms)

    assert coefficient.call_args_list == expected_calls
    assert coefficients == WilliamsInPlaneCoefficients(
        terms=tuple(terms),
        a_n=(1.0, 3.0),
        b_n=(2.0, 4.0),
    )
    assert t_stress == expected_t_stress


def test_failed_contour_does_not_append_incomplete_result():
    analysis = _analysis()
    observations = []
    successful_execution = _fake_execution(observations)(
        object(), object(), object(), None, [1], InterpolatorCache(4)
    )
    failed_execution = SimpleNamespace(
        evaluate_all=mock.Mock(side_effect=RuntimeError("contour failed"))
    )

    with (
        mock.patch.object(
            analysis_module,
            "_LineIntegralExecution",
            side_effect=[successful_execution, failed_execution],
        ),
        pytest.raises(RuntimeError, match="contour failed"),
    ):
        analysis._run_line_integrals()

    assert len(analysis.contour_results) == 1
    assert len(analysis.path_results) == 1
    with pytest.raises(AttributeError):
        analysis.contour_results = ()


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
    contour_geometry = line_integral._integration_contour_geometry
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
    np.testing.assert_allclose(contour_geometry.segment_lengths, expected_sizes)
    np.testing.assert_allclose(
        contour_geometry.segment_dy,
        line_integral.np_integration_points[:, 3],
    )
    np.testing.assert_allclose(
        contour_geometry.outward_unit_normals,
        expected_normals,
    )
    assert not np.shares_memory(
        contour_geometry.evaluation_points,
        line_integral._integration_eval_points,
    )


def test_line_integral_snapshots_mutated_facade_points_at_construction_handoff():
    path = IntegrationPath(
        path_properties=PathProperties(
            -1.0,
            1.0,
            -1.0,
            1.0,
            0.5,
            None,
            0.0,
            0.0,
        )
    )
    path.int_points[:, 0] += 0.125

    line_integral = LineIntegral(
        path,
        _measured_fields(),
        Material(E=72000, nu_xy=0.33),
    )

    np.testing.assert_array_equal(
        line_integral._contour.integration_points,
        path.int_points,
    )
    assert not np.shares_memory(
        line_integral._contour.integration_points,
        path.int_points,
    )


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
        mock.patch.object(
            line_integral._execution,
            "sample_in_plane_fields",
            wraps=line_integral._execution.sample_in_plane_fields,
        ) as interpolate_xy,
        mock.patch.object(
            line_integral._execution,
            "sample_mode_iii_fields",
            wraps=line_integral._execution.sample_mode_iii_fields,
        ) as interpolate_z,
    ):
        line_integral.integrate_j_decompose()

    assert interpolate_xy.call_count == 3
    assert interpolate_z.call_count == 1


def test_batched_auxiliary_fields_match_scalar_evaluation():
    line_integral = _line_integral()

    def polar(x, y):
        return np.sqrt(x**2.0 + y**2.0), np.arctan2(y, x)

    with mock.patch.object(
        line_integration_module.auxiliary_fields,
        "get_crack_nearfield",
        wraps=line_integration_module.auxiliary_fields.get_crack_nearfield,
    ) as crack_nearfield:
        actual_stress, actual_strain, actual_disp_y_dx = (
            line_integral._get_auxiliary_crack_nearfield(1.0, 0.0)
        )
    expected_stress = []
    expected_strain = []
    expected_disp_y_dx = []
    for point in line_integral.np_integration_points:
        r, phi = polar(point[0] - line_integral.origin_x, point[1] - line_integral.origin_y)
        stress, strain, _ = line_integration_module.auxiliary_fields.get_crack_nearfield(
            1.0,
            0.0,
            r,
            phi,
            line_integral.material,
        )
        expected_stress.append(stress)
        expected_strain.append(strain)
        r_pos, phi_pos = polar(
            point[0] - line_integral.origin_x + line_integral.x_shift,
            point[1] - line_integral.origin_y,
        )
        r_neg, phi_neg = polar(
            point[0] - line_integral.origin_x - line_integral.x_shift,
            point[1] - line_integral.origin_y,
        )
        disp_pos = line_integration_module.auxiliary_fields.get_crack_nearfield(
            1.0, 0.0, r_pos, phi_pos, line_integral.material
        )[2]
        disp_neg = line_integration_module.auxiliary_fields.get_crack_nearfield(
            1.0, 0.0, r_neg, phi_neg, line_integral.material
        )[2]
        expected_disp_y_dx.append((disp_pos[1] - disp_neg[1]) / (2 * line_integral.x_shift))

    np.testing.assert_allclose(actual_stress, expected_stress, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_strain, expected_strain, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_disp_y_dx, expected_disp_y_dx, rtol=1e-12, atol=1e-12)
    assert crack_nearfield.call_count == 3
    assert not line_integral._auxiliary_in_plane_fields.stress.flags.writeable
    assert not np.shares_memory(
        actual_stress,
        line_integral._auxiliary_in_plane_fields.stress,
    )
    prepared_stress = line_integral._auxiliary_in_plane_fields.stress.copy()
    actual_stress[0, 0, 0] = 123.0
    np.testing.assert_array_equal(
        line_integral._auxiliary_in_plane_fields.stress,
        prepared_stress,
    )

    with mock.patch.object(
        line_integration_module.auxiliary_fields,
        "get_zhao_solutions",
        wraps=line_integration_module.auxiliary_fields.get_zhao_solutions,
    ) as zhao:
        actual_zhao_stress, actual_disp_x_dx, actual_zhao_disp_y_dx = (
            line_integral._get_auxiliary_zhao_fields()
        )
    expected_zhao_stress = []
    expected_disp_x_dx = []
    expected_zhao_disp_y_dx = []
    for point in line_integral.np_integration_points:
        r, phi = polar(point[0] - line_integral.origin_x, point[1] - line_integral.origin_y)
        sigma_x, sigma_y, sigma_xy, _, _ = line_integration_module.auxiliary_fields.get_zhao_solutions(
            r, phi, line_integral.material
        )
        expected_zhao_stress.append([[sigma_x, sigma_xy], [sigma_xy, sigma_y]])
        r_pos, phi_pos = polar(
            point[0] - line_integral.origin_x + line_integral.x_shift,
            point[1] - line_integral.origin_y,
        )
        r_neg, phi_neg = polar(
            point[0] - line_integral.origin_x - line_integral.x_shift,
            point[1] - line_integral.origin_y,
        )
        positive = line_integration_module.auxiliary_fields.get_zhao_solutions(
            r_pos, phi_pos, line_integral.material
        )
        negative = line_integration_module.auxiliary_fields.get_zhao_solutions(
            r_neg, phi_neg, line_integral.material
        )
        expected_disp_x_dx.append((positive[3] - negative[3]) / (2 * line_integral.x_shift))
        expected_zhao_disp_y_dx.append((positive[4] - negative[4]) / (2 * line_integral.x_shift))

    np.testing.assert_allclose(actual_zhao_stress, expected_zhao_stress, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_disp_x_dx, expected_disp_x_dx, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_zhao_disp_y_dx, expected_zhao_disp_y_dx, rtol=1e-12, atol=1e-12)
    assert zhao.call_count == 3
    assert not line_integral._zhao_auxiliary_fields.stress.flags.writeable
    assert not np.shares_memory(
        actual_zhao_stress,
        line_integral._zhao_auxiliary_fields.stress,
    )


def test_mode_preparation_facade_delegates_and_returns_fresh_mutable_data():
    line_integral = _line_integral()
    line_integral._map_displacement_data_on_regular_grid(grid_points=8)

    with mock.patch.object(
        line_integration_module.mode_decomposition,
        "prepare_mode_data",
        wraps=line_integration_module.mode_decomposition.prepare_mode_data,
    ) as delegated:
        first = line_integral._prepare_mode_data("I")
        second = line_integral._prepare_mode_data("I")

    assert delegated.call_count == 2
    assert first is not second
    assert first.disp_x.flags.writeable
    original = second.disp_x.copy()
    first.disp_x[0] = 123.0
    np.testing.assert_array_equal(second.disp_x, original)


def test_scientific_reductions_ignore_mutated_compatibility_projections():
    line_integral = _line_integral()
    line_integral._interpolate_on_integration_points_z()

    before = {
        "j": line_integral._solve_j_integral(),
        "j_iii": line_integral._solve_j_integral_III(),
        "interaction": line_integral._solve_interaction_integral(1.0, 0.0),
        "t_stress": line_integral._solve_t_stress_interaction_integral(),
        "chen": line_integral._solve_chen_integral(n=-1, a_n=1.0),
    }

    mutable_projections = (
        "interpolated_eps_x",
        "interpolated_eps_y",
        "interpolated_eps_xy",
        "interpolated_sig_x",
        "interpolated_sig_y",
        "interpolated_sig_xy",
        "interpolated_disp_x",
        "interpolated_disp_y",
        "interpolated_disp_y_dx",
        "interpolated_eps_xz",
        "interpolated_eps_yz",
        "interpolated_sigma_xz",
        "interpolated_sigma_yz",
        "_path_normals",
        "_path_elem_heights",
        "_path_elem_sizes",
        "_integration_eval_r",
        "_integration_eval_phi",
    )
    for attribute in mutable_projections:
        values = getattr(line_integral, attribute)
        values[...] = 123.0

    after = {
        "j": line_integral._solve_j_integral(),
        "j_iii": line_integral._solve_j_integral_III(),
        "interaction": line_integral._solve_interaction_integral(1.0, 0.0),
        "t_stress": line_integral._solve_t_stress_interaction_integral(),
        "chen": line_integral._solve_chen_integral(n=-1, a_n=1.0),
    }

    for name in before:
        np.testing.assert_allclose(after[name], before[name], equal_nan=True)


def test_mode_preparation_ignores_mutated_regular_grid_projections():
    line_integral = _line_integral()
    line_integral._map_displacement_data_on_regular_grid(grid_points=8)
    before = line_integral._prepare_mode_data("I")

    for attribute in (
        "x_coordinates",
        "y_coordinates",
        "x_mesh",
        "y_mesh",
        "disp_u_mesh",
        "disp_v_mesh",
        "disp_w_mesh",
    ):
        getattr(line_integral, attribute)[...] = 123.0

    after = line_integral._prepare_mode_data("I")
    for attribute in (
        "coor_x",
        "coor_y",
        "disp_x",
        "disp_y",
        "disp_z",
        "eps_x",
        "eps_y",
        "eps_xy",
        "eps_xz",
        "eps_yz",
    ):
        np.testing.assert_allclose(
            getattr(after, attribute),
            getattr(before, attribute),
            equal_nan=True,
        )


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
            line_integral_runners,
            "eigenfunction",
            wraps=line_integral_runners.eigenfunction,
        ) as eigenfunction:
            actual = line_integral._solve_chen_integral(n=n, a_n=a_n, b_n=b_n)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        assert eigenfunction.call_count == 1


def test_line_integral_delegates_contour_formulas_to_functional_kernels():
    line_integral = _line_integral()
    line_integral._interpolate_on_integration_points_z()

    functional_kernels = [
        "in_plane_j_integral_terms",
        "mode_iii_j_integral_terms",
        "interaction_integral_terms",
        "t_stress_interaction_integral_terms",
        "bueckner_chen_integral_terms",
        "evaluate_contour_integral",
    ]
    with ExitStack() as stack:
        spies = [
            stack.enter_context(mock.patch.object(
                line_integral_runners,
                kernel,
                wraps=getattr(line_integral_runners, kernel),
            ))
            for kernel in functional_kernels
        ]
        line_integral._solve_j_integral()
        line_integral._solve_j_integral_III()
        line_integral._solve_interaction_integral(1.0, 0.0)
        line_integral._solve_t_stress_interaction_integral()
        line_integral._solve_chen_integral(n=-1, a_n=1)

    for spy in spies[:-1]:
        spy.assert_called_once()
    assert spies[-1].call_count == 5


def test_line_integral_delegates_result_mappings_to_quantity_kernels():
    line_integral = _line_integral()
    quantity_mappings = [
        "in_plane_energy_equivalent_sif_from_j_integral",
        "in_plane_sif_magnitude_from_j_integral",
        "mode_iii_sif_magnitude_from_j_integral",
        "in_plane_sif_from_interaction_integral",
        "t_stress_from_interaction_integral",
        "t_stress_from_stress_difference",
        "williams_coefficient_from_bueckner_chen_integral",
    ]

    with ExitStack() as stack:
        spies = {
            mapping: stack.enter_context(mock.patch.object(
                line_integral_runners,
                mapping,
                wraps=getattr(line_integral_runners, mapping),
            ))
            for mapping in quantity_mappings
        }
        spies["t_stress_from_williams_coefficient"] = stack.enter_context(
            mock.patch.object(
                williams_quantities,
                "t_stress_from_williams_coefficient",
                wraps=williams_quantities.t_stress_from_williams_coefficient,
            )
        )
        line_integral.integrate_j()
        line_integral.integrate_i_k1_k2()
        line_integral.integrate_i_t()
        line_integral.integrate_t_sdm()
        line_integral._williams_coeff_from_chen_integral(a_aux=1, n=1)
        line_integral.bueckner_williams_terms = []
        line_integral.integrate_bueckner_chen()
        line_integral.integrate_j_decompose()

    assert spies["in_plane_energy_equivalent_sif_from_j_integral"].call_count == 1
    assert spies["in_plane_sif_magnitude_from_j_integral"].call_count == 2
    assert spies["mode_iii_sif_magnitude_from_j_integral"].call_count == 1
    assert spies["in_plane_sif_from_interaction_integral"].call_count == 2
    assert spies["t_stress_from_interaction_integral"].call_count == 1
    assert spies["t_stress_from_stress_difference"].call_count == 1
    assert spies["t_stress_from_williams_coefficient"].call_count == 1
    assert spies["williams_coefficient_from_bueckner_chen_integral"].call_count == 2


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
    assert not line_integral._in_plane_samples.shifted.base.flags.writeable
    assert not np.shares_memory(
        line_integral._in_plane_samples.shifted.base,
        line_integral.interpolated_eps_x,
    )
    sampled_eps_x = line_integral._in_plane_samples.shifted.base[:, 0].copy()
    line_integral.interpolated_eps_x[0] = 123.0
    np.testing.assert_array_equal(
        line_integral._in_plane_samples.shifted.base[:, 0],
        sampled_eps_x,
    )
    line_integral._interpolate_on_integration_points()
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
        line_integration_module.sampling,
        "mask_contour_data",
        wraps=line_integration_module.sampling.mask_contour_data,
    ) as masked_data:
        line_integral._interpolate_on_integration_points_z()
        line_integral._interpolate_on_integration_points_z(mask_tol=0.75)
        assert masked_data.call_count == 2
        assert masked_data.call_args_list[0].args[2] is None
        assert masked_data.call_args_list[1].args[2] == 0.75

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
