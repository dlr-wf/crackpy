"""Williams coefficient contracts bind selected expansion terms to typed values."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WilliamsInPlaneCoefficients:
    """Store accepted in-plane Williams coefficients in selected-term order.

    Attributes:
        terms: Selected Williams term numbers shared by both coefficient tuples.
        a_n: Symmetric coefficients corresponding to ``terms``, each in
            MPa mm**(1 - n/2).
        b_n: Antisymmetric coefficients corresponding to ``terms``, each in
            MPa mm**(1 - n/2).
    """

    terms: tuple[int, ...]
    a_n: tuple[float, ...]
    b_n: tuple[float, ...]


@dataclass(frozen=True)
class WilliamsOutOfPlaneCoefficients:
    """Store accepted out-of-plane Williams coefficients in term order.

    Attributes:
        terms: Selected Williams term numbers corresponding to ``c_n``.
        c_n: Out-of-plane coefficients corresponding to ``terms``, each in
            MPa mm**(1 - n/2).
    """

    terms: tuple[int, ...]
    c_n: tuple[float, ...]
