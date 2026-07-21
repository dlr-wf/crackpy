"""Contour-Wise Results bind completed Integration Contours to derived quantities.

The contracts store geometry and already-derived scientific values from one completed contour execution.
"""

from dataclasses import dataclass
from typing import Iterable

from crackpy.fracture_analysis.crack_tip_fields.williams.coefficients import (
    WilliamsInPlaneCoefficients,
)

####################
# CONTOUR GEOMETRY #
####################


@dataclass(frozen=True)
class IntegrationContourResultGeometry:
    """Describe the Integration Contour associated with one completed result.

    Attributes:
        size_left: Signed distance from the crack tip to the left side in mm.
        size_right: Distance from the crack tip to the right side in mm.
        size_bottom: Signed distance from the crack tip to the bottom side in mm.
        size_top: Distance from the crack tip to the top side in mm.
        integration_points: Ordered xy evaluation coordinates in mm.
        number_of_nodes: Resolved nominal Integration Contour node count.
        tick_size: Resolved Integration Contour point spacing in mm.
    """

    size_left: float
    size_right: float
    size_bottom: float
    size_top: float
    integration_points: tuple[tuple[float, float], ...]
    number_of_nodes: int
    tick_size: float

    def __post_init__(self) -> None:
        """Copy the ordered evaluation coordinates into immutable tuples."""
        points: Iterable[Iterable[float]] = self.integration_points
        object.__setattr__(
            self,
            "integration_points",
            tuple((float(x), float(y)) for x, y in points),
        )


############################
# LINE-INTEGRAL QUANTITIES #
############################


@dataclass(frozen=True)
class LineIntegralQuantities:
    """Store the derived quantities from one completed line-integral execution.

    Attributes:
        j_integral: J-integral in N/mm.
        sif_k_j: Energy-equivalent in-plane Stress Intensity Factor from total
            J in MPa sqrt(m).
        sif_k_i: Signed Mode I Stress Intensity Factor from the interaction
            integral in MPa sqrt(m).
        sif_k_ii: Signed Mode II Stress Intensity Factor from the interaction
            integral in MPa sqrt(m).
        t_stress_chen: T-stress from the Bueckner-Chen Integral in MPa.
        t_stress_sdm: T-stress from the Stress-Difference Method in MPa.
        t_stress_int: T-stress from the interaction integral in MPa.
        decomp_j_integral_i: Decomposed Mode I J-integral in N/mm.
        decomp_j_integral_ii: Decomposed Mode II J-integral in N/mm.
        decomp_j_integral_iii: Decomposed Mode III J-integral in N/mm.
        decomp_j_integral_k_i: Mode I Stress Intensity Factor magnitude from
            decomposed J in MPa sqrt(m).
        decomp_j_integral_k_ii: Mode II Stress Intensity Factor magnitude from
            decomposed J in MPa sqrt(m).
        decomp_j_integral_k_iii: Mode III Stress Intensity Factor magnitude from
            decomposed J in MPa sqrt(m).
    """

    j_integral: float | None
    sif_k_j: float | None
    sif_k_i: float | None
    sif_k_ii: float | None
    t_stress_chen: float | None
    t_stress_sdm: float | None
    t_stress_int: float | None
    decomp_j_integral_i: float | None
    decomp_j_integral_ii: float | None
    decomp_j_integral_iii: float | None
    decomp_j_integral_k_i: float | None
    decomp_j_integral_k_ii: float | None
    decomp_j_integral_k_iii: float | None


#######################
# CONTOUR-WISE RESULT #
#######################


@dataclass(frozen=True)
class ContourWiseLineIntegralResult:
    """Store geometry and derived outputs from one completed Integration Contour.

    Attributes:
        geometry: Completed contour dimensions and evaluation coordinates.
        quantities: Derived fracture-mechanics quantities for this Integration
            Contour.
        williams_coefficients: Requested in-plane Williams coefficients, or
            ``None`` when Bueckner-Chen evaluation was disabled.
    """

    geometry: IntegrationContourResultGeometry
    quantities: LineIntegralQuantities
    williams_coefficients: WilliamsInPlaneCoefficients | None
