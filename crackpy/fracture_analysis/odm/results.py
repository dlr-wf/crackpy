"""ODM result contracts separate numerical coefficient-fit evidence from generic
Technique Execution status and model-owned scientific payloads.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Generic, Literal, TypeVar

import numpy as np

__all__ = ["CoefficientFitResult", "OdmFitResult"]

CoefficientsT = TypeVar("CoefficientsT")
QuantitiesT = TypeVar("QuantitiesT")


def _immutable_array_copy(values: np.ndarray) -> np.ndarray:
    """Return an independent NumPy array backed by immutable bytes storage.

    Args:
        values: Array values to copy into owned storage.

    Returns:
        A read-only array whose writeability cannot be re-enabled.
    """
    array = np.asarray(values)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class CoefficientFitResult:
    """Store the normalized numerical result of one ODM coefficient fit.

    Attributes:
        solver: Numerical Solver Route used for the coefficient fit.
        coefficients: Fitted coefficient vector with shape ``(n,)`` in the
            assembled system's column order.
        residual: Final displacement-residual vector with shape ``(m,)`` in
            assembled equation order, in mm.
        cost: Half the squared Euclidean norm of ``residual``, in mm².
        jacobian: Final residual Jacobian with shape ``(m, n)``, or ``None``
            when unavailable.
        rank: Effective system rank, or ``None`` when unavailable.
        singular_values: System singular values with shape ``(min(m, n),)``, or
            ``None`` when unavailable.
        success: Whether the numerical Solver Route completed successfully.
        message: Solver completion message.
        status: Solver-specific integer completion status.
        nfev: Residual evaluation count, or ``None`` when unavailable.
        njev: Jacobian evaluation count, or ``None`` when unavailable.
    """

    solver: Literal["direct", "iterative", "legacy"]
    coefficients: np.ndarray
    residual: np.ndarray
    cost: float
    jacobian: np.ndarray | None
    rank: int | None
    singular_values: np.ndarray | None
    success: bool
    message: str
    status: int
    nfev: int | None
    njev: int | None

    def __post_init__(self) -> None:
        """Copy authoritative arrays into immutable bytes-backed storage."""
        object.__setattr__(self, "coefficients", _immutable_array_copy(self.coefficients))
        object.__setattr__(self, "residual", _immutable_array_copy(self.residual))
        if self.jacobian is not None:
            object.__setattr__(self, "jacobian", _immutable_array_copy(self.jacobian))
        if self.singular_values is not None:
            object.__setattr__(
                self,
                "singular_values",
                _immutable_array_copy(self.singular_values),
            )


@dataclass(frozen=True)
class OdmFitResult(Generic[CoefficientsT, QuantitiesT]):
    """Store one typed ODM Technique Result and its numerical fit evidence.

    Attributes:
        coefficient_fit: Owned low-level fit retained as numerical evidence, or
            ``None`` when solving raised or execution was skipped.
        coefficients: Formulation-specific accepted coefficients, or a typed
            correctly shaped NaN payload when execution failed or was skipped.
        quantities: Formulation-specific derived fracture quantities, or a typed
            correctly shaped NaN payload when execution failed or was skipped.
        skipped: Init-only indication that execution was intentionally skipped.
            A skipped result cannot retain a coefficient fit.
        status: Technique execution outcome: ``completed``, ``failed``, or
            ``skipped``.
        cost: Half the squared displacement-residual norm in mm squared for a
            completed fit, or NaN for failed and skipped execution.
    """

    coefficient_fit: CoefficientFitResult | None
    coefficients: CoefficientsT
    quantities: QuantitiesT
    skipped: InitVar[bool] = False
    status: Literal["completed", "failed", "skipped"] = field(init=False)
    cost: float = field(init=False)

    def __post_init__(self, skipped: bool) -> None:
        """Derive execution status and accepted cost from fit evidence."""
        if skipped:
            if self.coefficient_fit is not None:
                raise ValueError("A skipped ODM result cannot retain a coefficient fit.")
            status = "skipped"
            cost = float("nan")
        elif (
            self.coefficient_fit is None
            or not self.coefficient_fit.success
            or self.coefficient_fit.residual.size == 0
        ):
            status = "failed"
            cost = float("nan")
        else:
            status = "completed"
            cost = self.coefficient_fit.cost
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "cost", cost)
