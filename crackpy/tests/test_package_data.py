"""Package-data tests keep CrackPy's scientific bibliography installable."""

import subprocess
import sys
from importlib.resources import files
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[2]
LINE_INTEGRAL_CITATION_KEYS = {
    "breitbarth_et_al_2019_dic_integrals",
    "chen_1985_path_independent_integrals",
    "kuna_fracture_mechanics",
    "molteno_becker_2015_j_integral_decomposition",
    "rice_1968_j_integral",
    "williams_1957",
    "yang_ravi_chandar_1999_stress_difference",
    "zhao_et_al_2001_corner_cracks",
}


def _assert_line_integral_citations_are_present(text: str) -> None:
    """Assert that the packaged bibliography resolves every line-integral key."""
    for citation_key in LINE_INTEGRAL_CITATION_KEYS:
        assert f"{{{citation_key}," in text


def test_source_package_exposes_line_integral_bibliography() -> None:
    bibliography = files("crackpy").joinpath("references.bib").read_text(
        encoding="utf-8"
    )

    _assert_line_integral_citations_are_present(bibliography)


def test_installed_wheel_exposes_line_integral_bibliography(
    tmp_path: Path,
) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    target = tmp_path / "installed"
    wheelhouse.mkdir()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--wheel-dir",
            str(wheelhouse),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    wheel = next(wheelhouse.glob("crackpy-*.whl"))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(target),
            str(wheel),
        ],
        check=True,
    )
    installed_bibliography = target / "crackpy" / "references.bib"
    bibliography = installed_bibliography.read_text(encoding="utf-8")

    _assert_line_integral_citations_are_present(bibliography)
