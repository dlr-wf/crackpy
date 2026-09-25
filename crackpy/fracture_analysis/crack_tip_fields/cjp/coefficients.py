"""CJP coefficient contracts preserve formulation-specific names, order, and units."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CjpModeICoefficients:
    """Store accepted CJP Mode I coefficients in formulation order.

    Attributes:
        a: CJP ``A`` coefficient in MPa sqrt(mm).
        b: CJP ``B`` coefficient in MPa sqrt(mm).
        c: CJP ``C`` coefficient in MPa.
        e: CJP ``E`` coefficient in MPa sqrt(mm).
        f: CJP ``F`` coefficient in MPa.
    """

    a: float
    b: float
    c: float
    e: float
    f: float


@dataclass(frozen=True)
class CjpMixedModeCoefficients:
    """Store accepted CJP mixed-mode coefficients in formulation order.

    Attributes:
        a_r: Real CJP ``A`` coefficient in MPa sqrt(mm).
        b_r: Real CJP ``B`` coefficient in MPa sqrt(mm).
        b_i: Imaginary CJP ``B`` coefficient in MPa sqrt(mm).
        c: CJP ``C`` coefficient in MPa.
        e: CJP ``E`` coefficient in MPa sqrt(mm).
    """

    a_r: float
    b_r: float
    b_i: float
    c: float
    e: float
