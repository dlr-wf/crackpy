"""Verify dependency isolation and established package-level module access."""

import subprocess
import sys
from pathlib import Path

import pytest


def _run_fresh_python(source: str) -> None:
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(root)!r})\n{source}"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("statement", [
    "import crackpy",
    "from crackpy.results.plot import PlotSettings, Plotter",
    "import crackpy; crackpy.results.plot.Plotter",
    "from crackpy.fracture_analysis.analysis import FractureAnalysis",
    "from crackpy.crack_detection.line_intercept import CrackDetectionLineIntercept",
    "import crackpy; crackpy.crack_detection.line_intercept.CrackDetectionLineIntercept",
])
def test_non_ai_imports_do_not_load_ai_dependencies(statement: str) -> None:
    _run_fresh_python(statement + "\nassert not {'torch', 'torchvision', 'sympy'} & sys.modules.keys()")


def test_established_package_attributes_and_star_imports() -> None:
    _run_fresh_python('''
import importlib
import crackpy

surfaces = {
    "crackpy": (
        "structure_elements", "input", "results", "fracture_analysis", "crack_detection",
        "crackpy", "logging_config", "setup_logging",
    ),
    "crackpy.fracture_analysis": (
        "analysis", "crack_tip", "line_integration", "pipeline", "utils", "crack_tip_fields",
        "line_integrals", "functionals", "odm", "optimization", "crackpy",
    ),
    "crackpy.crack_detection": (
        "data", "crackpy", "utils", "deep_learning", "detection", "pipeline",
        "correction", "line_intercept", "model",
    ),
}
for package_name, names in surfaces.items():
    package = importlib.import_module(package_name)
    assert set(names) <= set(dir(package))
    for name in names:
        value = getattr(package, name)
        if name == "crackpy":
            assert value is crackpy
        elif name != "setup_logging":
            assert value is importlib.import_module(f"{package_name}.{name}")
        assert getattr(package, name) is value
    namespace = {}
    exec(f"from {package_name} import *", namespace)
    assert set(names) <= namespace.keys()
    for name in names:
        assert namespace[name] is getattr(package, name)
assert "torch" in sys.modules
''')
