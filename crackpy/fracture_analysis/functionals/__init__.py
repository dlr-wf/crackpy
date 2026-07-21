"""Fracture-mechanics functionals define pure scientific expressions independently of their numerical evaluation technique."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IntegrandTerms:
    """Store scalar coefficient fields of a fracture-mechanics contour integral.

    Each field contains physical expressions evaluated at the contour-segment
    midpoints, such as strain-energy density, traction-work terms,
    interaction strain-energy density, or reciprocal-work terms.

    Attributes:
        integrated_over_dy:
            Fields integrated over each segment's signed vertical increment.
        added_over_ds:
            Fields added through integration over positive segment length.
        subtracted_over_ds:
            Fields subtracted through integration over positive segment length.
    """

    integrated_over_dy: tuple[np.ndarray, ...] = ()
    added_over_ds: tuple[np.ndarray, ...] = ()
    subtracted_over_ds: tuple[np.ndarray, ...] = ()


__all__ = ["IntegrandTerms"]
