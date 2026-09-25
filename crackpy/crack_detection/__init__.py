"""Expose detection methods without loading unrelated AI dependencies."""

from importlib import import_module as _import_module
from types import ModuleType as _ModuleType

import crackpy as crackpy

_MODULES = (
    "data", "deep_learning", "pipeline", "utils", "correction", "detection", "line_intercept", "model",
)
__all__ = list(_MODULES) + ["crackpy"]


def __getattr__(name: str) -> _ModuleType:
    """Return the requested detection module, loading it on first access.

    Args:
        name: Established detection-module name exposed by this package.

    Returns:
        The imported detection module.

    Raises:
        AttributeError: If the requested name is not an exposed module.
    """
    if name in _MODULES:
        module = _import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return package attributes, including detection modules not yet loaded."""
    return sorted(set(globals()) | set(_MODULES))
