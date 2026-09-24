"""Expose fracture-analysis modules independently on first access."""

from importlib import import_module as _import_module
from types import ModuleType as _ModuleType

import crackpy as crackpy

_MODULES = (
    "analysis", "crack_tip", "line_integration", "pipeline", "utils", "crack_tip_fields",
    "line_integrals", "functionals", "odm", "optimization",
)
__all__ = list(_MODULES) + ["crackpy"]


def __getattr__(name: str) -> _ModuleType:
    """Return the requested fracture-analysis module, loading it on first access.

    Args:
        name: Established fracture-analysis module name exposed by this package.

    Returns:
        The imported fracture-analysis module.

    Raises:
        AttributeError: If the requested name is not an exposed module.
    """
    if name in _MODULES:
        module = _import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return package attributes, including fracture-analysis modules not yet loaded."""
    return sorted(set(globals()) | set(_MODULES))
