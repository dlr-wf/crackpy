"""Facade compatibility adapters map completed line-integral results and
deprecated public spellings to CrackPy's mutable analysis interface."""

import numpy as np

from crackpy.fracture_analysis.crack_tip_fields.williams.coefficients import (
    WilliamsInPlaneCoefficients,
)
from crackpy.fracture_analysis.line_integrals.results import (
    ContourWiseLineIntegralResult,
)

_UNSET = object()


################################
# DEPRECATED SPELLING ADAPTERS #
################################


def resolve_bueckner_williams_terms(
    bueckner_williams_terms: list | None | object,
    buckner_williams_terms: list | None | object,
) -> list | None:
    """Resolve the preferred and deprecated spellings of the term selection.

    Args:
        bueckner_williams_terms: Terms supplied through the preferred spelling.
        buckner_williams_terms: Terms supplied through the deprecated spelling.

    Returns:
        The selected Williams terms, or ``None`` when neither spelling supplies
        terms.

    Raises:
        ValueError: If both spellings supply terms.
    """
    if buckner_williams_terms is not _UNSET:
        if bueckner_williams_terms is not _UNSET:
            raise ValueError(
                "Use either bueckner_williams_terms or the deprecated "
                "buckner_williams_terms, not both."
            )
        return buckner_williams_terms
    if bueckner_williams_terms is _UNSET:
        return None
    return bueckner_williams_terms


class _DeprecatedBuecknerSpellingAliases:
    """Provide deprecated ``buckner`` aliases for the ``bueckner`` names."""

    @property
    def buckner_williams_terms(self) -> list | None:
        """Deprecated: use ``bueckner_williams_terms``."""
        return self.bueckner_williams_terms

    @buckner_williams_terms.setter
    def buckner_williams_terms(self, terms: list | None) -> None:
        """Deprecated: use ``bueckner_williams_terms``."""
        self.bueckner_williams_terms = terms


##############################
# MUTABLE RESULT PROJECTIONS #
##############################


def mutable_path_result(
    result: ContourWiseLineIntegralResult,
) -> list[float | np.ndarray | None]:
    """Project one result into the established mutable 13-value path list.

    Args:
        result: Completed Contour-Wise Result.

    Returns:
        Mutable quantities ordered as ``j_integral``, ``sif_k_j``,
        ``sif_k_i``, ``sif_k_ii``, ``t_stress_chen``, ``t_stress_sdm``,
        ``t_stress_int``, the three modal J-Integrals, and the three modal
        Stress Intensity Factor magnitudes.
    """
    quantities = result.quantities
    return [
        quantities.j_integral,
        quantities.sif_k_j,
        quantities.sif_k_i,
        quantities.sif_k_ii,
        quantities.t_stress_chen,
        np.asarray(quantities.t_stress_sdm),
        quantities.t_stress_int,
        quantities.decomp_j_integral_i,
        quantities.decomp_j_integral_ii,
        quantities.decomp_j_integral_iii,
        quantities.decomp_j_integral_k_i,
        quantities.decomp_j_integral_k_ii,
        quantities.decomp_j_integral_k_iii,
    ]


def mutable_path_size(result: ContourWiseLineIntegralResult) -> list[float]:
    """Project signed contour distances into the mutable path-size list.

    Args:
        result: Completed Contour-Wise Result.

    Returns:
        Signed crack-tip distances in mm ordered as ``[size_left, size_right,
        size_bottom, size_top]``.
    """
    geometry = result.geometry
    path_size = [
        geometry.size_left,
        geometry.size_right,
        geometry.size_bottom,
        geometry.size_top,
    ]
    return path_size


def _williams_coefficients(
    source: ContourWiseLineIntegralResult | WilliamsInPlaneCoefficients | None,
) -> WilliamsInPlaneCoefficients | None:
    if isinstance(source, ContourWiseLineIntegralResult):
        return source.williams_coefficients
    return source


def mutable_williams_a_n(
    source: ContourWiseLineIntegralResult | WilliamsInPlaneCoefficients | None,
) -> list[float]:
    """Project symmetric Williams coefficients into a mutable list.

    Args:
        source: Completed result, requested Williams coefficients, or ``None``.

    Returns:
        Mutable symmetric coefficients in requested-term order.
    """
    coefficients = _williams_coefficients(source)
    return [] if coefficients is None else list(coefficients.a_n)


def mutable_williams_b_n(
    source: ContourWiseLineIntegralResult | WilliamsInPlaneCoefficients | None,
) -> list[float]:
    """Project antisymmetric Williams coefficients into a mutable list.

    Args:
        source: Completed result, requested Williams coefficients, or ``None``.

    Returns:
        Mutable antisymmetric coefficients in requested-term order.
    """
    coefficients = _williams_coefficients(source)
    return [] if coefficients is None else list(coefficients.b_n)


def mutable_williams_coefficients(
    source: ContourWiseLineIntegralResult | WilliamsInPlaneCoefficients | None,
) -> list[list[int | float]]:
    """Project Williams term triples into established mutable nested lists.

    Args:
        source: Completed result, requested Williams coefficients, or ``None``.

    Returns:
        Mutable ``[term, a_n, b_n]`` rows in requested-term order.
    """
    coefficients = _williams_coefficients(source)
    if coefficients is None:
        return []
    return [
        [term, a_n, b_n]
        for term, a_n, b_n in zip(
            coefficients.terms,
            coefficients.a_n,
            coefficients.b_n,
        )
    ]


def mutable_integration_points(
    result: ContourWiseLineIntegralResult,
) -> list[list[float]]:
    """Project contour coordinates into established mutable xy lists.

    Args:
        result: Completed Contour-Wise Result.

    Returns:
        Mutable ``[x_coordinates, y_coordinates]`` lists.
    """
    return [
        [point[0] for point in result.geometry.integration_points],
        [point[1] for point in result.geometry.integration_points],
    ]
