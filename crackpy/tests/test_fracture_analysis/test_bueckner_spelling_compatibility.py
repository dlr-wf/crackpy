"""Bueckner spelling tests bind deprecated aliases to their preferred names."""

import json
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from crackpy.fracture_analysis.line_integration import (
    DEFAULT_BUCKNER_CHEN_TERMS,
    DEFAULT_BUECKNER_CHEN_TERMS,
    IntegralProperties,
    LineIntegral,
)
from crackpy.results.write import OutputWriter, _serialized_settings


def _minimal_line_integral(**kwargs) -> LineIntegral:
    """Construct the facade without evaluating contour geometry or measured fields."""
    integration_path = SimpleNamespace(
        path_properties=SimpleNamespace(tick_size=0.5),
        origin_x=0.0,
        origin_y=0.0,
    )
    with mock.patch.object(
        LineIntegral,
        "__post_init__",
        autospec=True,
    ):
        return LineIntegral(
            integration_path,
            data=SimpleNamespace(),
            material=SimpleNamespace(),
            **kwargs,
        )


def test_preferred_bueckner_terms_are_used_directly() -> None:
    terms = [1, 2]

    properties = IntegralProperties(bueckner_williams_terms=terms)
    line_integral = _minimal_line_integral(bueckner_williams_terms=terms)

    assert properties.bueckner_williams_terms is terms
    assert line_integral.bueckner_williams_terms is terms


def test_deprecated_constant_remains_available() -> None:
    assert DEFAULT_BUCKNER_CHEN_TERMS == [1, 2, 3, 4, 5]
    assert DEFAULT_BUCKNER_CHEN_TERMS is DEFAULT_BUECKNER_CHEN_TERMS


@pytest.mark.parametrize("owner", [IntegralProperties, LineIntegral])
def test_deprecated_constructor_keyword_is_silent(owner: type) -> None:
    terms = [1, 2]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if owner is IntegralProperties:
            instance = owner(buckner_williams_terms=terms)
        else:
            instance = _minimal_line_integral(buckner_williams_terms=terms)

    assert caught == []
    assert instance.bueckner_williams_terms is terms


@pytest.mark.parametrize("owner", [IntegralProperties, LineIntegral])
def test_both_constructor_spellings_are_rejected(owner: type) -> None:
    with pytest.raises(ValueError, match="Use either bueckner_williams_terms"):
        if owner is IntegralProperties:
            owner(
                bueckner_williams_terms=[1],
                buckner_williams_terms=[2],
            )
        else:
            _minimal_line_integral(
                bueckner_williams_terms=[1],
                buckner_williams_terms=[2],
            )


@pytest.mark.parametrize("owner", [IntegralProperties, LineIntegral])
def test_deprecated_attribute_is_silent_and_updates_preferred_state(owner: type) -> None:
    instance = owner.__new__(owner)
    instance.bueckner_williams_terms = [1]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert instance.buckner_williams_terms == [1]
        instance.buckner_williams_terms = [2]

    assert caught == []
    assert instance.bueckner_williams_terms == [2]


def test_deprecated_integrate_method_is_silent_and_delegates() -> None:
    line_integral = LineIntegral.__new__(LineIntegral)
    line_integral.integrate_bueckner_chen = mock.Mock()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line_integral.integrate_buckner_chen()

    assert caught == []
    line_integral.integrate_bueckner_chen.assert_called_once_with()


def test_deprecated_default_method_is_silent_and_delegates() -> None:
    properties = IntegralProperties(bueckner_williams_terms=None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LineIntegral.ensure_defaults_buckner_chen(properties)

    assert caught == []
    assert properties.bueckner_williams_terms == [1, 2, 3, 4, 5]


def test_integral_properties_keep_the_established_serialized_setting_name() -> None:
    properties = IntegralProperties(bueckner_williams_terms=[1, 2])

    serialized = _serialized_settings("integral_properties", properties)

    assert serialized["buckner_williams_terms"] == [1, 2]
    assert "bueckner_williams_terms" not in serialized


def test_json_output_keeps_the_established_serialized_setting_name(
    tmp_path: Path,
) -> None:
    properties = IntegralProperties(bueckner_williams_terms=None)
    result_names = {
        "j",
        "sif_j",
        "sif_k_i",
        "sif_k_ii",
        "t_stress_int",
        "k_i_chen",
        "k_ii_chen",
        "t_stress_chen",
        "t_stress_sdm",
        "decomp_j_1",
        "decomp_j_2",
        "decomp_j_3",
        "decomp_K_1",
        "decomp_K_2",
        "decomp_K_3",
    }
    statistics = {name: 0.0 for name in result_names}
    analysis = SimpleNamespace(
        nodemap_file="example.txt",
        crack_tip=SimpleNamespace(
            crack_tip_x=0.0,
            crack_tip_y=0.0,
            crack_tip_angle=0.0,
            left_or_right="right",
        ),
        data=SimpleNamespace(
            force=None,
            cycles=None,
            displacement=None,
            potential=None,
            cracklength=None,
            time=None,
        ),
        integral_properties=properties,
        optimization_properties=None,
        material=SimpleNamespace(name="test material"),
        sifs_int={
            "mean": statistics.copy(),
            "median": statistics.copy(),
            "rej_out_mean": statistics.copy(),
        },
        path_results=np.zeros((1, 13)),
        num_of_path_nodes=[4],
        tick_sizes=[1.0],
        path_sizes=[[-1.0, 1.0, -1.0, 1.0]],
        integration_points=[(np.array([-1.0, 1.0]), np.array([-1.0, 1.0]))],
    )

    writer = OutputWriter(path=tmp_path, fracture_analysis=analysis)
    writer.write_json()

    output_file = tmp_path / "example_right_Output.json"
    output = json.loads(output_file.read_text(encoding="utf-8"))
    serialized = output["CrackPy_settings"]["integral_properties"]
    assert serialized["buckner_williams_terms"] is None
    assert "bueckner_williams_terms" not in serialized
