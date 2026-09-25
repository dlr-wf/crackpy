"""End-user subprocess tests exercise fracture-analysis demonstrations from the repository root."""

import math
import re
import subprocess
import sys
from pathlib import Path


def test_nested_fracture_analysis_scripts_resolve_the_repository_root() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    fracture_analysis_directory = repository_root / "scripts" / "fracture_analysis"
    relative_paths = (
        "nodemaps/dic.py",
        "nodemaps/fem.py",
        "synthetic_fields/williams_in_plane.py",
        "synthetic_fields/williams_in_plane_with_mode_iii.py",
        "pipelines/dic_generated_contours.py",
        "pipelines/dic_predefined_contours.py",
        "pipelines/fem.py",
    )

    for relative_path in relative_paths:
        source = (fracture_analysis_directory / relative_path).read_text(encoding="utf-8")
        assert "Path(__file__).resolve().parents[3]" in source


def test_j_integral_mode_decomposition_demonstration_reports_finite_quantities():
    repository_root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.fracture_analysis.line_integrals.j_integral_mode_decomposition",
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    output = completed.stdout
    number = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    expected_labels = {
        "Expected Mode I J-integral": "N/mm",
        "Recovered Mode I J-integral": "N/mm",
        "Expected Mode II J-integral": "N/mm",
        "Recovered Mode II J-integral": "N/mm",
        "Expected Mode III J-integral": "N/mm",
        "Recovered Mode III J-integral": "N/mm",
        "Expected Mode I SIF": "MPa sqrt(m)",
        "Recovered Mode I SIF": "MPa sqrt(m)",
        "Expected Mode II SIF": "MPa sqrt(m)",
        "Recovered Mode II SIF": "MPa sqrt(m)",
        "Expected Mode III SIF": "MPa sqrt(m)",
        "Recovered Mode III SIF": "MPa sqrt(m)",
        "Prescribed T-stress": "MPa",
    }
    for label, unit in expected_labels.items():
        match = re.search(rf"^{re.escape(label)}: {number} {re.escape(unit)}$", output, re.MULTILINE)
        assert match is not None, output
        assert math.isfinite(float(match.group(1)))
    assert "Recovered T-stress" not in output
