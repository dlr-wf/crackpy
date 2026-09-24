"""Analysis-ownership evidence for four authoritative ODM Technique Results and
their synchronized mutable compatibility projections.
"""

from dataclasses import astuple
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from crackpy.fracture_analysis.analysis import FractureAnalysis
from crackpy.fracture_analysis.optimization import OptimizationProperties
from crackpy.input.crack_tip_info import CrackTipInfo
from crackpy.input.input_data import InputData
from crackpy.structure_elements.material import Material


def _analysis(data: InputData | None = None) -> FractureAnalysis:
    """Build an analysis without preparing optimization or line-integral work."""
    return FractureAnalysis(
        material=Material(),
        nodemap="synthetic",
        data=InputData() if data is None else data,
        crack_tip_info=CrackTipInfo(0.0, 0.0, 0.0, "right"),
        integral_properties=None,
        optimization_properties=None,
    )


def _facade_result(
    coefficients: list[float] | np.ndarray,
    *,
    cost: float = 1.25,
    success: bool = True,
) -> OptimizeResult:
    """Build one normalized mutable result returned by the Optimization facade."""
    values = np.asarray(coefficients, dtype=float)
    return OptimizeResult(
        solver="direct",
        x=values,
        fun=np.array([0.25, -0.5]),
        cost=cost,
        jac=np.ones((2, values.size)),
        rank=values.size,
        singular_values=np.ones(values.size),
        success=success,
        message="complete" if success else "failed",
        status=1 if success else -1,
        nfev=1,
        njev=1,
    )


def _enable_optimization(analysis: FractureAnalysis) -> mock.Mock:
    """Attach a controllable Optimization facade and enable the public run path."""
    analysis.optimization_properties = SimpleNamespace(
        min_radius=0.3,
        max_radius=1.1,
        angle_gap=25,
        terms=[-1, 1, 2],
    )
    analysis.optimization = mock.Mock()
    analysis.optimization.terms = np.asarray(analysis.optimization_properties.terms)
    return analysis.optimization


def test_analysis_initializes_empty_authoritative_odm_results() -> None:
    analysis = _analysis()

    assert analysis.cjp_mode_i_odm_result is None
    assert analysis.cjp_mixed_mode_odm_result is None
    assert analysis.williams_in_plane_odm_result is None
    assert analysis.williams_out_of_plane_odm_result is None
    assert analysis.__dict__["_cjp_mode_i_odm_result"] is None
    assert analysis.__dict__["_cjp_mixed_mode_odm_result"] is None
    assert analysis.__dict__["_williams_in_plane_odm_result"] is None
    assert analysis.__dict__["_williams_out_of_plane_odm_result"] is None


def test_run_stores_and_projects_completed_cjp_mode_i_result() -> None:
    analysis = _analysis()
    facade = _facade_result([1.0, 2.0, 3.0, 4.0, 5.0])
    optimization = _enable_optimization(analysis)
    optimization.optimize_cjp_displacements_modeI.return_value = facade

    with (
        mock.patch.object(analysis, "_run_cjp_optimization_mixedmode"),
        mock.patch.object(analysis, "_run_williams_optimization"),
    ):
        returned = analysis.run()

    result = analysis.cjp_mode_i_odm_result
    assert returned is None
    assert result is not None
    assert result.status == "completed"
    assert astuple(result.coefficients) == (1.0, 2.0, 3.0, 4.0, 5.0)
    assert result.cost == 1.25
    assert result.quantities.k_f == pytest.approx(-1.466431100114224)
    assert result.quantities.k_r == pytest.approx(-1.9921855875089491)
    assert result.quantities.k_s == pytest.approx(0.11889981892818033)
    assert result.quantities.t_x == -3.0
    assert result.quantities.t_y == -5.0
    np.testing.assert_array_equal(analysis.cjp_coeffs_m1, astuple(result.coefficients))
    assert list(analysis.cjp_res_m1) == ["Error", "K_F", "K_R", "K_S", "T_x", "T_y"]
    assert analysis.cjp_res_m1 == {
        "Error": result.cost,
        "K_F": result.quantities.k_f,
        "K_R": result.quantities.k_r,
        "K_S": result.quantities.k_s,
        "T_x": result.quantities.t_x,
        "T_y": result.quantities.t_y,
    }


def test_run_records_cjp_mode_i_exception_as_failure_without_a_fit() -> None:
    analysis = _analysis()
    optimization = _enable_optimization(analysis)
    optimization.optimize_cjp_displacements_modeI.side_effect = RuntimeError("boom")

    with (
        mock.patch.object(analysis, "_run_cjp_optimization_mixedmode"),
        mock.patch.object(analysis, "_run_williams_optimization"),
    ):
        analysis.run()

    result = analysis.cjp_mode_i_odm_result
    assert result is not None
    assert result.status == "failed"
    assert result.coefficient_fit is None
    assert len(astuple(result.coefficients)) == 5
    assert all(np.isnan(value) for value in astuple(result.coefficients))
    assert analysis.cjp_coeffs_m1.shape == (5,)
    assert np.isnan(analysis.cjp_coeffs_m1).all()
    assert list(analysis.cjp_res_m1) == ["Error", "K_F", "K_R", "K_S", "T_x", "T_y"]
    assert all(np.isnan(value) for value in analysis.cjp_res_m1.values())


def test_run_records_unsuccessful_cjp_mode_i_fit_without_using_candidates() -> None:
    analysis = _analysis()
    facade = _facade_result([1.0, 2.0, 3.0, 4.0, 5.0], success=False)
    optimization = _enable_optimization(analysis)
    optimization.optimize_cjp_displacements_modeI.return_value = facade

    with (
        mock.patch.object(analysis, "_run_cjp_optimization_mixedmode"),
        mock.patch.object(analysis, "_run_williams_optimization"),
    ):
        analysis.run()

    result = analysis.cjp_mode_i_odm_result
    assert result is not None
    assert result.status == "failed"
    assert result.coefficient_fit is not None
    assert result.coefficient_fit.success is False
    np.testing.assert_array_equal(result.coefficient_fit.coefficients, facade.x)
    assert all(np.isnan(value) for value in astuple(result.coefficients))
    assert np.isnan(analysis.cjp_coeffs_m1).all()
    assert all(np.isnan(value) for value in analysis.cjp_res_m1.values())


def test_cjp_mode_i_failure_replaces_a_previous_success_without_stale_values() -> None:
    analysis = _analysis()
    optimization = _enable_optimization(analysis)
    optimization.optimize_cjp_displacements_modeI.side_effect = [
        _facade_result([1.0, 2.0, 3.0, 4.0, 5.0]),
        _facade_result([6.0, 7.0, 8.0, 9.0, 10.0], success=False),
    ]

    with (
        mock.patch.object(analysis, "_run_cjp_optimization_mixedmode"),
        mock.patch.object(analysis, "_run_williams_optimization"),
    ):
        analysis.run()
        assert analysis.cjp_mode_i_odm_result.status == "completed"
        analysis.run()

    result = analysis.cjp_mode_i_odm_result
    assert result.status == "failed"
    assert result.coefficient_fit is not None
    np.testing.assert_array_equal(
        result.coefficient_fit.coefficients,
        [6.0, 7.0, 8.0, 9.0, 10.0],
    )
    assert all(np.isnan(value) for value in astuple(result.coefficients))
    assert np.isnan(analysis.cjp_coeffs_m1).all()
    assert all(np.isnan(value) for value in analysis.cjp_res_m1.values())


def test_cjp_mode_i_authoritative_property_is_read_only() -> None:
    analysis = _analysis()

    with pytest.raises(AttributeError):
        analysis.cjp_mode_i_odm_result = None


def test_cjp_mode_i_authoritative_result_isolated_from_mutable_projections() -> None:
    analysis = _analysis()
    facade = _facade_result([1.0, 2.0, 3.0, 4.0, 5.0])
    optimization = _enable_optimization(analysis)
    optimization.optimize_cjp_displacements_modeI.return_value = facade

    with (
        mock.patch.object(analysis, "_run_cjp_optimization_mixedmode"),
        mock.patch.object(analysis, "_run_williams_optimization"),
    ):
        analysis.run()

    result = analysis.cjp_mode_i_odm_result
    facade.x.fill(10.0)
    analysis.cjp_coeffs_m1.fill(20.0)
    analysis.cjp_res_m1["K_F"] = 30.0

    assert astuple(result.coefficients) == (1.0, 2.0, 3.0, 4.0, 5.0)
    np.testing.assert_array_equal(
        result.coefficient_fit.coefficients,
        [1.0, 2.0, 3.0, 4.0, 5.0],
    )
    assert result.quantities.k_f == pytest.approx(-1.466431100114224)


def test_run_stores_and_projects_completed_cjp_mixed_mode_result() -> None:
    analysis = _analysis()
    optimization = _enable_optimization(analysis)
    optimization.optimize_cjp_displacements_mixedmode.return_value = _facade_result(
        [1.0, 2.0, 3.0, 4.0, 5.0]
    )

    with (
        mock.patch.object(analysis, "_run_cjp_optimization_modeI"),
        mock.patch.object(analysis, "_run_williams_optimization"),
    ):
        analysis.run()

    result = analysis.cjp_mixed_mode_odm_result
    assert result is not None
    assert result.status == "completed"
    assert astuple(result.coefficients) == (1.0, 2.0, 3.0, 4.0, 5.0)
    assert result.cost == 1.25
    assert result.quantities.k_f == pytest.approx(-1.7834972839227048)
    assert result.quantities.k_r == pytest.approx(-3.441430535811629)
    assert result.quantities.k_s == pytest.approx(-0.11889981892818033)
    assert result.quantities.k_ii == pytest.approx(0.4755992757127213)
    assert result.quantities.t_stress == -4.0
    np.testing.assert_array_equal(analysis.cjp_coeffs_mm, astuple(result.coefficients))
    assert list(analysis.cjp_res_mm) == ["Error", "K_F", "K_R", "K_S", "K_II", "T"]
    assert analysis.cjp_res_mm == {
        "Error": result.cost,
        "K_F": result.quantities.k_f,
        "K_R": result.quantities.k_r,
        "K_S": result.quantities.k_s,
        "K_II": result.quantities.k_ii,
        "T": result.quantities.t_stress,
    }


def test_run_records_cjp_mixed_mode_exception_as_failure_without_a_fit() -> None:
    analysis = _analysis()
    optimization = _enable_optimization(analysis)
    optimization.optimize_cjp_displacements_mixedmode.side_effect = RuntimeError("boom")

    with (
        mock.patch.object(analysis, "_run_cjp_optimization_modeI"),
        mock.patch.object(analysis, "_run_williams_optimization"),
    ):
        analysis.run()

    result = analysis.cjp_mixed_mode_odm_result
    assert result is not None
    assert result.status == "failed"
    assert result.coefficient_fit is None
    assert len(astuple(result.coefficients)) == 5
    assert all(np.isnan(value) for value in astuple(result.coefficients))
    assert analysis.cjp_coeffs_mm.shape == (5,)
    assert np.isnan(analysis.cjp_coeffs_mm).all()
    assert list(analysis.cjp_res_mm) == ["Error", "K_F", "K_R", "K_S", "K_II", "T"]
    assert all(np.isnan(value) for value in analysis.cjp_res_mm.values())


def test_run_records_unsuccessful_cjp_mixed_mode_fit_without_using_candidates() -> None:
    analysis = _analysis()
    facade = _facade_result([1.0, 2.0, 3.0, 4.0, 5.0], success=False)
    optimization = _enable_optimization(analysis)
    optimization.optimize_cjp_displacements_mixedmode.return_value = facade

    with (
        mock.patch.object(analysis, "_run_cjp_optimization_modeI"),
        mock.patch.object(analysis, "_run_williams_optimization"),
    ):
        analysis.run()

    result = analysis.cjp_mixed_mode_odm_result
    assert result is not None
    assert result.status == "failed"
    assert result.coefficient_fit is not None
    assert result.coefficient_fit.success is False
    np.testing.assert_array_equal(result.coefficient_fit.coefficients, facade.x)
    assert all(np.isnan(value) for value in astuple(result.coefficients))
    assert np.isnan(analysis.cjp_coeffs_mm).all()
    assert all(np.isnan(value) for value in analysis.cjp_res_mm.values())


def test_cjp_mixed_mode_failure_replaces_previous_success_without_stale_values() -> None:
    analysis = _analysis()
    optimization = _enable_optimization(analysis)
    optimization.optimize_cjp_displacements_mixedmode.side_effect = [
        _facade_result([1.0, 2.0, 3.0, 4.0, 5.0]),
        _facade_result([6.0, 7.0, 8.0, 9.0, 10.0], success=False),
    ]

    with (
        mock.patch.object(analysis, "_run_cjp_optimization_modeI"),
        mock.patch.object(analysis, "_run_williams_optimization"),
    ):
        analysis.run()
        assert analysis.cjp_mixed_mode_odm_result.status == "completed"
        analysis.run()

    result = analysis.cjp_mixed_mode_odm_result
    assert result.status == "failed"
    assert result.coefficient_fit is not None
    np.testing.assert_array_equal(
        result.coefficient_fit.coefficients,
        [6.0, 7.0, 8.0, 9.0, 10.0],
    )
    assert all(np.isnan(value) for value in astuple(result.coefficients))
    assert np.isnan(analysis.cjp_coeffs_mm).all()
    assert all(np.isnan(value) for value in analysis.cjp_res_mm.values())


def test_cjp_mixed_mode_authoritative_property_is_read_only() -> None:
    analysis = _analysis()

    with pytest.raises(AttributeError):
        analysis.cjp_mixed_mode_odm_result = None


def test_cjp_mixed_mode_result_isolated_from_mutable_projections() -> None:
    analysis = _analysis()
    facade = _facade_result([1.0, 2.0, 3.0, 4.0, 5.0])
    optimization = _enable_optimization(analysis)
    optimization.optimize_cjp_displacements_mixedmode.return_value = facade

    with (
        mock.patch.object(analysis, "_run_cjp_optimization_modeI"),
        mock.patch.object(analysis, "_run_williams_optimization"),
    ):
        analysis.run()

    result = analysis.cjp_mixed_mode_odm_result
    facade.x.fill(10.0)
    analysis.cjp_coeffs_mm.fill(20.0)
    analysis.cjp_res_mm["K_F"] = 30.0

    assert astuple(result.coefficients) == (1.0, 2.0, 3.0, 4.0, 5.0)
    np.testing.assert_array_equal(
        result.coefficient_fit.coefficients,
        [1.0, 2.0, 3.0, 4.0, 5.0],
    )
    assert result.quantities.k_f == pytest.approx(-1.7834972839227048)


def test_run_stores_and_projects_completed_williams_results() -> None:
    data = InputData()
    data.disp_z = np.array([1.0])
    analysis = _analysis(data)
    optimization = _enable_optimization(analysis)
    optimization.optimize_williams_displacements_xy.return_value = _facade_result(
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        cost=1.5,
    )
    optimization.optimize_williams_displacements_z.return_value = _facade_result(
        [70.0, 80.0, 90.0],
        cost=0.5,
    )

    with (
        mock.patch.object(analysis, "_run_cjp_optimization_modeI"),
        mock.patch.object(analysis, "_run_cjp_optimization_mixedmode"),
    ):
        analysis.run()

    in_plane_result = analysis.williams_in_plane_odm_result
    out_of_plane_result = analysis.williams_out_of_plane_odm_result
    assert in_plane_result is not None
    assert out_of_plane_result is not None
    assert in_plane_result.coefficients.terms == (-1, 1, 2)
    assert out_of_plane_result.coefficients.terms == (-1, 1, 2)
    assert in_plane_result.coefficients.a_n == (10.0, 20.0, 30.0)
    assert in_plane_result.coefficients.b_n == (40.0, 50.0, 60.0)
    assert out_of_plane_result.coefficients.c_n == (70.0, 80.0, 90.0)
    assert in_plane_result.status == "completed"
    assert out_of_plane_result.status == "completed"
    assert in_plane_result.cost == 1.5
    assert out_of_plane_result.cost == 0.5
    assert in_plane_result.quantities.k_i == pytest.approx(1.5853309190424043)
    assert in_plane_result.quantities.k_ii == pytest.approx(-3.963327297606011)
    assert out_of_plane_result.quantities.k_iii == pytest.approx(3.1706618380848086)
    assert in_plane_result.quantities.t_stress == 120.0
    np.testing.assert_array_equal(
        analysis.williams_coeffs,
        in_plane_result.coefficients.a_n
        + in_plane_result.coefficients.b_n
        + out_of_plane_result.coefficients.c_n,
    )
    assert analysis.williams_fit_a_n == {-1: 10.0, 1: 20.0, 2: 30.0}
    assert analysis.williams_fit_b_n == {-1: 40.0, 1: 50.0, 2: 60.0}
    assert analysis.williams_fit_c_n == {-1: 70.0, 1: 80.0, 2: 90.0}
    assert list(analysis.williams_fit_res) == [
        "Error_xy",
        "K_I",
        "K_II",
        "T",
        "Error_z",
        "K_III",
    ]


def test_custom_williams_terms_retain_fit_and_derive_supported_quantities() -> None:
    data = InputData()
    data.disp_z = np.array([1.0])
    analysis = _analysis(data)
    optimization = _enable_optimization(analysis)
    analysis.optimization_properties.terms = [1, 3]
    optimization.terms = np.array([1, 3])
    optimization.optimize_williams_displacements_xy.return_value = _facade_result(
        [10.0, 20.0, 30.0, 40.0],
        cost=1.5,
    )
    optimization.optimize_williams_displacements_z.return_value = _facade_result(
        [70.0, 80.0],
        cost=0.5,
    )
    analysis.integral_properties = SimpleNamespace(
        number_of_paths=1,
        integral_size_left=-1.0,
        integral_size_right=1.0,
    )

    with (
        mock.patch.object(analysis, "_run_cjp_optimization_modeI"),
        mock.patch.object(analysis, "_run_cjp_optimization_mixedmode"),
        mock.patch.object(analysis, "_run_line_integrals") as run_line_integrals,
    ):
        returned = analysis.run()

    in_plane_result = analysis.williams_in_plane_odm_result
    out_of_plane_result = analysis.williams_out_of_plane_odm_result
    assert returned is None
    assert in_plane_result.status == "completed"
    assert in_plane_result.coefficient_fit is not None
    assert in_plane_result.coefficient_fit.success is True
    assert in_plane_result.coefficients.terms == (1, 3)
    assert in_plane_result.coefficients.a_n == (10.0, 20.0)
    assert in_plane_result.coefficients.b_n == (30.0, 40.0)
    assert in_plane_result.quantities.k_i == pytest.approx(0.7926654595212022)
    assert in_plane_result.quantities.k_ii == pytest.approx(-2.3779963785636067)
    assert np.isnan(in_plane_result.quantities.t_stress)
    assert in_plane_result.cost == 1.5
    assert out_of_plane_result.status == "completed"
    assert out_of_plane_result.coefficient_fit is not None
    assert out_of_plane_result.coefficient_fit.success is True
    assert out_of_plane_result.coefficients.c_n == (70.0, 80.0)
    assert out_of_plane_result.cost == 0.5
    assert out_of_plane_result.quantities.k_iii == pytest.approx(
        np.sqrt(0.5 * np.pi) * 70.0 / np.sqrt(1000)
    )
    assert analysis.williams_coeffs.shape == (6,)
    np.testing.assert_array_equal(
        analysis.williams_coeffs,
        [10.0, 20.0, 30.0, 40.0, 70.0, 80.0],
    )
    assert analysis.williams_fit_res["Error_xy"] == 1.5
    assert analysis.williams_fit_res["K_I"] == pytest.approx(0.7926654595212022)
    assert analysis.williams_fit_res["K_II"] == pytest.approx(-2.3779963785636067)
    assert np.isnan(analysis.williams_fit_res["T"])
    run_line_integrals.assert_called_once_with(None, None)


def test_williams_in_plane_exception_preserves_out_of_plane_completion() -> None:
    data = InputData()
    data.disp_z = np.array([1.0])
    analysis = _analysis(data)
    optimization = _enable_optimization(analysis)
    optimization.optimize_williams_displacements_xy.side_effect = RuntimeError("xy")
    optimization.optimize_williams_displacements_z.return_value = _facade_result(
        [70.0, 80.0, 90.0],
        cost=0.5,
    )

    with (
        mock.patch.object(analysis, "_run_cjp_optimization_modeI"),
        mock.patch.object(analysis, "_run_cjp_optimization_mixedmode"),
    ):
        analysis.run()

    in_plane_result = analysis.williams_in_plane_odm_result
    out_of_plane_result = analysis.williams_out_of_plane_odm_result
    assert in_plane_result.status == "failed"
    assert in_plane_result.coefficient_fit is None
    assert out_of_plane_result.status == "completed"
    assert out_of_plane_result.coefficients.c_n == (70.0, 80.0, 90.0)
    assert all(
        np.isnan(value)
        for value in (
            in_plane_result.coefficients.a_n + in_plane_result.coefficients.b_n
        )
    )
    assert np.isnan(analysis.williams_coeffs[:6]).all()
    np.testing.assert_array_equal(analysis.williams_coeffs[6:], [70.0, 80.0, 90.0])


def test_williams_out_of_plane_exception_preserves_in_plane_completion() -> None:
    data = InputData()
    data.disp_z = np.array([1.0])
    analysis = _analysis(data)
    optimization = _enable_optimization(analysis)
    optimization.optimize_williams_displacements_xy.return_value = _facade_result(
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        cost=1.5,
    )
    optimization.optimize_williams_displacements_z.side_effect = RuntimeError("z")

    with (
        mock.patch.object(analysis, "_run_cjp_optimization_modeI"),
        mock.patch.object(analysis, "_run_cjp_optimization_mixedmode"),
    ):
        analysis.run()

    in_plane_result = analysis.williams_in_plane_odm_result
    out_of_plane_result = analysis.williams_out_of_plane_odm_result
    assert in_plane_result.status == "completed"
    assert in_plane_result.coefficients.a_n == (10.0, 20.0, 30.0)
    assert in_plane_result.coefficients.b_n == (40.0, 50.0, 60.0)
    assert out_of_plane_result.status == "failed"
    assert out_of_plane_result.coefficient_fit is None
    assert all(np.isnan(value) for value in out_of_plane_result.coefficients.c_n)
    np.testing.assert_array_equal(
        analysis.williams_coeffs[:6],
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    )
    assert np.isnan(analysis.williams_coeffs[6:]).all()


def test_williams_partial_completion_retains_unsuccessful_in_plane_fit() -> None:
    data = InputData()
    data.disp_z = np.array([1.0])
    analysis = _analysis(data)
    optimization = _enable_optimization(analysis)
    optimization.optimize_williams_displacements_xy.return_value = _facade_result(
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        success=False,
    )
    optimization.optimize_williams_displacements_z.return_value = _facade_result(
        [70.0, 80.0, 90.0],
        cost=0.5,
    )

    with (
        mock.patch.object(analysis, "_run_cjp_optimization_modeI"),
        mock.patch.object(analysis, "_run_cjp_optimization_mixedmode"),
    ):
        analysis.run()

    in_plane_result = analysis.williams_in_plane_odm_result
    out_of_plane_result = analysis.williams_out_of_plane_odm_result
    assert in_plane_result.status == "failed"
    assert in_plane_result.coefficient_fit is not None
    assert in_plane_result.coefficient_fit.success is False
    np.testing.assert_array_equal(
        in_plane_result.coefficient_fit.coefficients,
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    )
    assert all(
        np.isnan(value)
        for value in (
            in_plane_result.coefficients.a_n + in_plane_result.coefficients.b_n
        )
    )
    assert out_of_plane_result.status == "completed"
    assert out_of_plane_result.coefficients.c_n == (70.0, 80.0, 90.0)


def test_williams_partial_completion_retains_unsuccessful_out_of_plane_fit() -> None:
    data = InputData()
    data.disp_z = np.array([1.0])
    analysis = _analysis(data)
    optimization = _enable_optimization(analysis)
    optimization.optimize_williams_displacements_xy.return_value = _facade_result(
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        cost=1.5,
    )
    optimization.optimize_williams_displacements_z.return_value = _facade_result(
        [70.0, 80.0, 90.0],
        success=False,
    )

    with (
        mock.patch.object(analysis, "_run_cjp_optimization_modeI"),
        mock.patch.object(analysis, "_run_cjp_optimization_mixedmode"),
    ):
        analysis.run()

    in_plane_result = analysis.williams_in_plane_odm_result
    out_of_plane_result = analysis.williams_out_of_plane_odm_result
    assert in_plane_result.status == "completed"
    assert out_of_plane_result.status == "failed"
    assert out_of_plane_result.coefficient_fit is not None
    assert out_of_plane_result.coefficient_fit.success is False
    np.testing.assert_array_equal(
        out_of_plane_result.coefficient_fit.coefficients,
        [70.0, 80.0, 90.0],
    )
    assert all(np.isnan(value) for value in out_of_plane_result.coefficients.c_n)
    assert np.isnan(out_of_plane_result.cost)
    assert np.isnan(out_of_plane_result.quantities.k_iii)


def test_williams_failed_rerun_replaces_all_previous_success_values() -> None:
    data = InputData()
    data.disp_z = np.array([1.0])
    analysis = _analysis(data)
    optimization = _enable_optimization(analysis)
    optimization.optimize_williams_displacements_xy.side_effect = [
        _facade_result([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        _facade_result(
            [11.0, 21.0, 31.0, 41.0, 51.0, 61.0],
            success=False,
        ),
    ]
    optimization.optimize_williams_displacements_z.side_effect = [
        _facade_result([70.0, 80.0, 90.0]),
        _facade_result([71.0, 81.0, 91.0], success=False),
    ]

    with (
        mock.patch.object(analysis, "_run_cjp_optimization_modeI"),
        mock.patch.object(analysis, "_run_cjp_optimization_mixedmode"),
    ):
        analysis.run()
        assert analysis.williams_in_plane_odm_result.status == "completed"
        assert analysis.williams_out_of_plane_odm_result.status == "completed"
        analysis.run()

    in_plane_result = analysis.williams_in_plane_odm_result
    out_of_plane_result = analysis.williams_out_of_plane_odm_result
    assert in_plane_result.status == "failed"
    assert out_of_plane_result.status == "failed"
    assert all(
        np.isnan(value)
        for value in (
            in_plane_result.coefficients.a_n
            + in_plane_result.coefficients.b_n
            + out_of_plane_result.coefficients.c_n
        )
    )
    assert np.isnan(analysis.williams_coeffs).all()
    assert all(np.isnan(value) for value in analysis.williams_fit_res.values())


def test_williams_authoritative_result_isolated_from_mutable_projections() -> None:
    data = InputData()
    data.disp_z = np.array([1.0])
    analysis = _analysis(data)
    optimization = _enable_optimization(analysis)
    xy_facade = _facade_result([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    z_facade = _facade_result([70.0, 80.0, 90.0])
    optimization.optimize_williams_displacements_xy.return_value = xy_facade
    optimization.optimize_williams_displacements_z.return_value = z_facade

    with (
        mock.patch.object(analysis, "_run_cjp_optimization_modeI"),
        mock.patch.object(analysis, "_run_cjp_optimization_mixedmode"),
    ):
        analysis.run()

    in_plane_result = analysis.williams_in_plane_odm_result
    out_of_plane_result = analysis.williams_out_of_plane_odm_result
    xy_facade.x.fill(1.0)
    z_facade.x.fill(2.0)
    analysis.williams_coeffs.fill(3.0)
    analysis.williams_fit_a_n[1] = 4.0
    analysis.williams_fit_b_n[1] = 5.0
    analysis.williams_fit_c_n[1] = 6.0
    analysis.williams_fit_res["K_I"] = 7.0

    assert in_plane_result.coefficients.a_n == (10.0, 20.0, 30.0)
    assert in_plane_result.coefficients.b_n == (40.0, 50.0, 60.0)
    assert out_of_plane_result.coefficients.c_n == (70.0, 80.0, 90.0)
    np.testing.assert_array_equal(
        in_plane_result.coefficient_fit.coefficients,
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    )
    np.testing.assert_array_equal(
        out_of_plane_result.coefficient_fit.coefficients,
        [70.0, 80.0, 90.0],
    )


def test_williams_authoritative_property_is_read_only() -> None:
    analysis = _analysis()

    with pytest.raises(AttributeError):
        analysis.williams_in_plane_odm_result = None
    with pytest.raises(AttributeError):
        analysis.williams_out_of_plane_odm_result = None


def test_williams_runner_skips_missing_z_displacements() -> None:
    data = InputData()
    data.disp_z = None
    analysis = _analysis(data)
    optimization = _enable_optimization(analysis)
    optimization.optimize_williams_displacements_xy.return_value = _facade_result(
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    )

    analysis._run_williams_optimization()

    result = analysis.williams_out_of_plane_odm_result
    assert result.status == "skipped"
    assert result.coefficient_fit is None
    assert len(result.coefficients.c_n) == len(result.coefficients.terms) == 3
    assert all(np.isnan(value) for value in result.coefficients.c_n)
    assert np.isnan(result.cost)
    assert np.isnan(result.quantities.k_iii)
    optimization.optimize_williams_displacements_z.assert_not_called()


def test_williams_runner_skips_all_zero_z_displacements() -> None:
    data = InputData()
    data.disp_z = np.zeros(4)
    analysis = _analysis(data)
    optimization = _enable_optimization(analysis)
    optimization.optimize_williams_displacements_xy.return_value = _facade_result(
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    )

    analysis._run_williams_optimization()

    result = analysis.williams_out_of_plane_odm_result
    assert result.status == "skipped"
    assert result.coefficient_fit is None
    assert len(result.coefficients.c_n) == len(result.coefficients.terms) == 3
    assert all(np.isnan(value) for value in result.coefficients.c_n)
    assert np.isnan(result.cost)
    assert np.isnan(result.quantities.k_iii)
    optimization.optimize_williams_displacements_z.assert_not_called()


@pytest.mark.parametrize("selection", ["outside-domain", "empty-z", "empty-xy", "zero-xy"])
def test_run_distinguishes_empty_fits_from_supported_zero_displacements(selection) -> None:
    axis = np.linspace(-2.0, 2.0, 9)
    coor_x, coor_y = np.meshgrid(axis, axis)
    data = InputData()
    data.coor_x = coor_x.ravel()
    data.coor_y = coor_y.ravel()
    data.coor_z = np.zeros(coor_x.size)
    data.disp_x = (0.02 * coor_x - 0.01 * coor_y).ravel()
    data.disp_y = (0.01 * coor_x + 0.03 * coor_y).ravel()
    data.disp_z = (-0.015 * coor_x + 0.005 * coor_y).ravel()
    if selection == "empty-z":
        data.disp_z[:] = np.nan
    elif selection == "empty-xy":
        data.disp_x[:] = np.nan
        data.disp_y[:] = np.nan
    elif selection == "zero-xy":
        data.disp_x[:] = 0.0
        data.disp_y[:] = 0.0
    outside_domain = selection == "outside-domain"
    analysis = FractureAnalysis(
        material=Material(), nodemap="empty-fit-selection", data=data,
        crack_tip_info=CrackTipInfo(0.0, 0.0, 0.0, "right"),
        integral_properties=None,
        optimization_properties=OptimizationProperties(
            angle_gap=25, min_radius=10.0 if outside_domain else 0.3,
            max_radius=11.0 if outside_domain else 1.1,
            tick_size=0.2, terms=[1, 2],
        ),
    )

    analysis.run()

    xy_failed = selection in ("outside-domain", "empty-xy")
    z_failed = selection in ("outside-domain", "empty-z")
    for result, failed in (
        (analysis.cjp_mode_i_odm_result, xy_failed),
        (analysis.cjp_mixed_mode_odm_result, xy_failed),
        (analysis.williams_in_plane_odm_result, xy_failed),
        (analysis.williams_out_of_plane_odm_result, z_failed),
    ):
        assert result.status == ("failed" if failed else "completed")
        fit = result.coefficient_fit
        assert fit is not None
        assert fit.success
        if failed:
            assert fit.residual.size == 0
            assert fit.cost == 0.0
            assert np.isnan(result.cost)
            assert np.isnan(astuple(result.quantities)).all()
        else:
            assert fit.residual.size > 0
            assert np.isfinite(result.cost)
            assert np.isfinite(astuple(result.quantities)).all()

    for coefficients, quantities in (
        (analysis.cjp_coeffs_m1, analysis.cjp_res_m1),
        (analysis.cjp_coeffs_mm, analysis.cjp_res_mm),
    ):
        assert coefficients.shape == (5,)
        if xy_failed:
            assert np.isnan(coefficients).all()
            assert all(np.isnan(value) for value in quantities.values())
        elif selection == "zero-xy":
            np.testing.assert_array_equal(coefficients, np.zeros(5))
            assert all(value == 0.0 for value in quantities.values())
    for coefficients, failed in (
        (analysis.williams_fit_a_n, xy_failed),
        (analysis.williams_fit_b_n, xy_failed),
        (analysis.williams_fit_c_n, z_failed),
    ):
        assert list(coefficients) == [1, 2]
        assert all(np.isnan(value) if failed else np.isfinite(value)
                   for value in coefficients.values())
    for key, failed in (
        ("Error_xy", xy_failed), ("K_I", xy_failed), ("K_II", xy_failed),
        ("T", xy_failed), ("Error_z", z_failed), ("K_III", z_failed),
    ):
        value = analysis.williams_fit_res[key]
        assert np.isnan(value) if failed else np.isfinite(value)
        if selection == "zero-xy" and key not in ("Error_z", "K_III"):
            assert value == 0.0


def test_run_returns_none_and_preserves_technique_execution_order() -> None:
    analysis = _analysis()
    _enable_optimization(analysis)
    analysis.integral_properties = SimpleNamespace(
        number_of_paths=1,
        integral_size_left=-1.0,
        integral_size_right=1.0,
    )
    execution_order = []

    with (
        mock.patch.object(
            analysis,
            "_run_cjp_optimization_modeI",
            side_effect=lambda: execution_order.append("cjp_mode_i"),
        ),
        mock.patch.object(
            analysis,
            "_run_cjp_optimization_mixedmode",
            side_effect=lambda: execution_order.append("cjp_mixed_mode"),
        ),
        mock.patch.object(
            analysis,
            "_run_williams_optimization",
            side_effect=lambda: execution_order.append("williams"),
        ),
        mock.patch.object(
            analysis,
            "_run_line_integrals",
            side_effect=lambda *args: execution_order.append("line_integrals"),
        ),
    ):
        returned = analysis.run()

    assert returned is None
    assert execution_order == [
        "cjp_mode_i",
        "cjp_mixed_mode",
        "williams",
        "line_integrals",
    ]
