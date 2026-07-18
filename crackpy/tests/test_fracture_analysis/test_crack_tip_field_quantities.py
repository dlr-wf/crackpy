"""Scientific quantity evidence for CJP and Williams coefficient transformations,
including signs, term lookup, units, and formula references.
"""

import numpy as np
import pytest

from crackpy.fracture_analysis.crack_tip_fields.cjp.quantities import (
    derive_cjp_mixed_mode_fracture_quantities,
    derive_cjp_mode_i_fracture_quantities,
)
from crackpy.fracture_analysis.crack_tip_fields.williams.quantities import (
    derive_williams_in_plane_fracture_quantities,
    derive_williams_out_of_plane_fracture_quantities,
)


def test_cjp_mode_i_quantities_preserve_coefficient_order_signs_and_units() -> None:
    quantities = derive_cjp_mode_i_fracture_quantities(
        (1.0, 2.0, 3.0, 4.0, 5.0)
    )

    assert quantities == pytest.approx(
        (-1.466431100114224, -1.9921855875089491, 0.11889981892818033, -3.0, -5.0)
    )


def test_cjp_mixed_mode_quantities_preserve_negative_k_s_and_units() -> None:
    quantities = derive_cjp_mixed_mode_fracture_quantities(
        (1.0, 2.0, 3.0, 4.0, 5.0)
    )

    assert quantities == pytest.approx(
        (
            -1.7834972839227048,
            -3.441430535811629,
            -0.11889981892818033,
            0.4755992757127213,
            -4.0,
        )
    )


def test_williams_in_plane_quantities_use_term_lookup_and_negative_k_ii() -> None:
    quantities = derive_williams_in_plane_fracture_quantities(
        (2, -1, 1),
        (30.0, 10.0, 20.0),
        (60.0, 40.0, 50.0),
    )

    assert quantities == pytest.approx((1.5853309190424043, -3.963327297606011, 120.0))


def test_williams_out_of_plane_quantities_use_term_lookup_and_units() -> None:
    quantities = derive_williams_out_of_plane_fracture_quantities(
        (2, -1, 1),
        (90.0, 70.0, 80.0),
    )

    assert quantities == pytest.approx((3.1706618380848086,))


def test_williams_in_plane_derives_only_quantities_supported_by_terms() -> None:
    term_one_quantities = derive_williams_in_plane_fracture_quantities(
        (1, 3),
        (10.0, 20.0),
        (30.0, 40.0),
    )
    term_two_quantities = derive_williams_in_plane_fracture_quantities(
        (2, 3),
        (30.0, 20.0),
        (60.0, 40.0),
    )

    assert term_one_quantities[:2] == pytest.approx(
        (0.7926654595212022, -2.3779963785636067)
    )
    assert np.isnan(term_one_quantities[2])
    assert np.isnan(term_two_quantities[0])
    assert np.isnan(term_two_quantities[1])
    assert term_two_quantities[2] == 120.0


def test_williams_out_of_plane_returns_nan_without_term_one() -> None:
    (k_iii,) = derive_williams_out_of_plane_fracture_quantities(
        (-1, 2, 3),
        (10.0, 20.0, 30.0),
    )

    assert np.isnan(k_iii)


def test_formula_references_are_documented_at_scientific_kernels() -> None:
    mode_i_doc = " ".join(derive_cjp_mode_i_fracture_quantities.__doc__.split())
    mixed_doc = " ".join(derive_cjp_mixed_mode_fracture_quantities.__doc__.split())
    williams_xy_doc = " ".join(
        derive_williams_in_plane_fracture_quantities.__doc__.split()
    )
    williams_z_doc = " ".join(
        derive_williams_out_of_plane_fracture_quantities.__doc__.split()
    )

    assert "DOI 10.3390/ma16165705" in mode_i_doc
    assert "DOI 10.3221/IGF-ESIS.25.23" in mixed_doc
    for docstring in (mode_i_doc, mixed_doc, williams_xy_doc, williams_z_doc):
        assert "future attachment point" in docstring
    for docstring in (williams_xy_doc, williams_z_doc):
        assert "DOI 10.1115/1.4011454" in docstring
        assert "DOI 10.1007/978-94-007-6680-8" in docstring
