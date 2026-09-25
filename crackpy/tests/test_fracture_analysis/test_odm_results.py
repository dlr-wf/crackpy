"""ODM result-contract evidence for immutable numerical fit storage and generic
completed, failed, and skipped Technique Result state.
"""

from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest

import crackpy.fracture_analysis.odm.results as odm_results
from crackpy.fracture_analysis.crack_tip_fields.cjp import (
    CjpMixedModeCoefficients,
    CjpMixedModeQuantities,
    CjpModeICoefficients,
    CjpModeIQuantities,
)
from crackpy.fracture_analysis.crack_tip_fields.williams import (
    WilliamsInPlaneCoefficients,
    WilliamsInPlaneQuantities,
    WilliamsOutOfPlaneCoefficients,
    WilliamsOutOfPlaneQuantities,
)
from crackpy.fracture_analysis.odm.results import CoefficientFitResult, OdmFitResult


def _fit(*, success: bool = True) -> CoefficientFitResult:
    """Build one internally consistent coefficient-fit fixture."""
    return CoefficientFitResult(
        solver="direct",
        coefficients=np.arange(5.0),
        residual=np.array([0.25, -0.5]),
        cost=0.15625,
        jacobian=np.ones((2, 5)),
        rank=2,
        singular_values=np.ones(2),
        success=success,
        message="complete" if success else "failed",
        status=1 if success else -1,
        nfev=1,
        njev=1,
    )


def _mode_i_payloads() -> tuple[CjpModeICoefficients, CjpModeIQuantities]:
    """Return representative typed CJP Mode I payloads."""
    return (
        CjpModeICoefficients(1.0, 2.0, 3.0, 4.0, 5.0),
        CjpModeIQuantities(6.0, 7.0, 8.0, 9.0, 10.0),
    )


def test_public_result_module_exports_exact_data_only_schemas() -> None:
    assert odm_results.__all__ == [
        "CoefficientFitResult",
        "OdmFitResult",
    ]
    assert [field.name for field in fields(OdmFitResult)] == [
        "coefficient_fit",
        "coefficients",
        "quantities",
        "status",
        "cost",
    ]
    expected_payload_fields = {
        CjpModeICoefficients: ["a", "b", "c", "e", "f"],
        CjpModeIQuantities: ["k_f", "k_r", "k_s", "t_x", "t_y"],
        CjpMixedModeCoefficients: ["a_r", "b_r", "b_i", "c", "e"],
        CjpMixedModeQuantities: ["k_f", "k_r", "k_s", "k_ii", "t_stress"],
        WilliamsInPlaneCoefficients: ["terms", "a_n", "b_n"],
        WilliamsInPlaneQuantities: ["k_i", "k_ii", "t_stress"],
        WilliamsOutOfPlaneCoefficients: ["terms", "c_n"],
        WilliamsOutOfPlaneQuantities: ["k_iii"],
    }
    for payload_type, expected_fields in expected_payload_fields.items():
        assert [field.name for field in fields(payload_type)] == expected_fields
        assert payload_type.__dataclass_params__.frozen
    assert CoefficientFitResult.__module__ == "crackpy.fracture_analysis.odm.results"
    assert OdmFitResult.__dataclass_params__.frozen


def test_odm_fit_result_derives_completed_status_and_cost_from_successful_fit() -> None:
    fit = _fit()
    coefficients, quantities = _mode_i_payloads()

    result = OdmFitResult(fit, coefficients, quantities)

    assert result.coefficient_fit is fit
    assert result.coefficients is coefficients
    assert result.quantities is quantities
    assert result.status == "completed"
    assert result.cost == fit.cost


@pytest.mark.parametrize("fit", [None, _fit(success=False)])
def test_odm_fit_result_derives_failed_status_and_nan_cost(fit) -> None:
    coefficients, quantities = _mode_i_payloads()

    result = OdmFitResult(fit, coefficients, quantities)

    assert result.coefficient_fit is fit
    assert result.status == "failed"
    assert np.isnan(result.cost)


def test_odm_fit_result_derives_explicit_skip_without_a_fit() -> None:
    coefficients, quantities = _mode_i_payloads()

    result = OdmFitResult(None, coefficients, quantities, skipped=True)

    assert result.status == "skipped"
    assert np.isnan(result.cost)


def test_odm_fit_result_rejects_explicit_skip_with_any_fit() -> None:
    coefficients, quantities = _mode_i_payloads()

    with pytest.raises(ValueError, match="skipped ODM result cannot retain"):
        OdmFitResult(_fit(), coefficients, quantities, skipped=True)


def test_odm_fit_result_is_immutable() -> None:
    coefficients, quantities = _mode_i_payloads()
    result = OdmFitResult(None, coefficients, quantities)

    with pytest.raises(FrozenInstanceError):
        result.status = "completed"


def test_result_module_contains_no_construction_or_scientific_formulas() -> None:
    assert not any(
        name.startswith(("_build_", "_project_", "derive_"))
        for name in vars(odm_results)
    )


def test_public_result_docstrings_describe_every_field_and_units() -> None:
    public_types = (
        OdmFitResult,
        CjpModeICoefficients,
        CjpModeIQuantities,
        CjpMixedModeCoefficients,
        CjpMixedModeQuantities,
        WilliamsInPlaneCoefficients,
        WilliamsInPlaneQuantities,
        WilliamsOutOfPlaneCoefficients,
        WilliamsOutOfPlaneQuantities,
    )
    for result_type in public_types:
        docstring = " ".join(result_type.__doc__.split())
        for field_name in result_type.__dataclass_fields__:
            assert f"{field_name}:" in docstring

    generic_doc = " ".join(OdmFitResult.__doc__.split())
    assert "mm squared" in generic_doc
    for payload_type in (
        CjpModeICoefficients,
        CjpModeIQuantities,
        CjpMixedModeCoefficients,
        CjpMixedModeQuantities,
        WilliamsInPlaneCoefficients,
        WilliamsInPlaneQuantities,
        WilliamsOutOfPlaneCoefficients,
        WilliamsOutOfPlaneQuantities,
    ):
        assert "MPa" in payload_type.__doc__


def test_odm_fit_result_rejects_empty_observations_but_retains_solver_evidence() -> None:
    fit = replace(
        _fit(), coefficients=np.zeros(5), residual=np.empty(0), cost=0.0,
        jacobian=np.empty((0, 5)), rank=0, singular_values=np.empty(0),
    )
    coefficients, quantities = _mode_i_payloads()

    result = OdmFitResult(fit, coefficients, quantities)

    assert result.status == "failed"
    assert np.isnan(result.cost)
    assert result.coefficient_fit is fit
    assert fit.success
    assert fit.cost == 0.0
