"""Validate installed package resources and public analysis workflows.

Run with isolated Python from outside the checkout in a fresh wheel environment.
The existing DIC integration cases supply the numerical regression expectations.
"""

import importlib.metadata
import importlib.resources
import importlib.util
import json
import runpy
import sys
import unittest
from pathlib import Path


def main() -> None:
    """Check wheel origin/resources and run Williams, CJP and integral cases."""
    checkout = Path(__file__).resolve().parents[1]
    assert sys.flags.isolated, "Run this check with python -I"
    assert not Path.cwd().resolve().is_relative_to(checkout), "Run outside the checkout"
    assert sys.prefix != sys.base_prefix, "Use a fresh virtual environment"
    assert importlib.util.find_spec("pytest") is None, "Do not install test dependencies"

    import crackpy

    origin = Path(crackpy.__file__).resolve()
    assert origin.is_relative_to(Path(sys.prefix).resolve()), origin
    assert not origin.is_relative_to(checkout), origin
    assert crackpy.__version__ == importlib.metadata.version("crackpy")
    assert importlib.util.find_spec("crackpy.tests") is None
    assert not {"torch", "torchvision", "sympy"} & sys.modules.keys()
    for name in ("logging.yaml", "references.bib"):
        assert importlib.resources.files("crackpy").joinpath(name).read_text(encoding="utf-8").strip()

    # Only test definitions and fixtures come from the checkout; -I keeps its
    # package off sys.path, and the origin assertion verifies the installed code.
    cases = runpy.run_path(str(checkout / "test_scripts/test_fracture_analysis.py"))
    test_class = cases["TestFractureAnalysis"]
    suite = unittest.TestSuite(
        test_class(name) for name in (
            "test_fracture_analysis_with_constant_tick_size",
            "test_fitting_methods_with_DIC_data",
        )
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert not {"torch", "torchvision", "sympy"} & sys.modules.keys()
    print(json.dumps({
        "python": sys.version,
        "package": str(origin),
        "version": crackpy.__version__,
        "sympy_loaded": "sympy" in sys.modules,
        "tests_run": result.testsRun,
        "successful": result.wasSuccessful(),
    }, indent=2))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
