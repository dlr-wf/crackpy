"""Expose CrackPy subsystems on demand and initialize package logging."""

from importlib import import_module as _import_module
from types import ModuleType as _ModuleType

import crackpy as crackpy

from . import logging_config as logging_config
from .logging_config import setup_logging

setup_logging()

# package information
__version__ = "1.4.0"

_MODULES = ("fracture_analysis", "crack_detection", "structure_elements", "input", "results")
__all__ = list(_MODULES) + ["crackpy", "logging_config", "setup_logging"]


def __getattr__(name: str) -> _ModuleType:
    """Return the requested subsystem, loading it on first attribute access.

    Args:
        name: Established subsystem name exposed by this package.

    Returns:
        The imported subsystem module.

    Raises:
        AttributeError: If the requested name is not an exposed subsystem.
    """
    if name in _MODULES:
        module = _import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return package attributes, including subsystems not yet loaded."""
    return sorted(set(globals()) | set(_MODULES))
