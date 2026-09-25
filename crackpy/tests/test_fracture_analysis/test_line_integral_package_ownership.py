"""Dependency tests enforce line-integral package ownership."""

import ast
from pathlib import Path

import crackpy.fracture_analysis.line_integrals as line_integrals
from crackpy.fracture_analysis.crack_tip_fields.williams import quantities

PACKAGE_ROOT = Path(line_integrals.__file__).parent


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def test_private_runner_is_not_exported_from_line_integral_package():
    assert "_LineIntegralExecution" not in line_integrals.__all__
    assert not hasattr(line_integrals, "_LineIntegralExecution")


def test_williams_coefficient_unit_conversions_are_owned_by_field_model():
    runner_source = (PACKAGE_ROOT / "runners.py").read_text(encoding="utf-8")

    assert quantities.williams_coefficient_m_to_mm.__module__ == quantities.__name__
    assert quantities.williams_coefficient_mm_to_m.__module__ == quantities.__name__
    assert "def _williams_coefficient_m_to_mm" not in runner_source
    assert "def _williams_coefficient_mm_to_m" not in runner_source


def test_functionals_do_not_depend_on_line_integrals_or_orchestration():
    functionals_root = PACKAGE_ROOT.parent / "functionals"
    forbidden = (
        "crackpy.fracture_analysis.line_integrals",
        "crackpy.fracture_analysis.analysis",
        "crackpy.fracture_analysis.line_integration",
        "crackpy.results",
    )
    for path in functionals_root.glob("*.py"):
        imports = _imports(path)
        assert not any(name.startswith(forbidden) for name in imports), path


def test_result_contract_contains_no_fracture_formula_dependencies():
    imports = _imports(PACKAGE_ROOT / "results.py")
    forbidden = (
        "crackpy.fracture_analysis.functionals",
        "crackpy.fracture_analysis.line_integrals.quadrature",
        "crackpy.fracture_analysis.crack_tip_fields.williams.quantities",
    )
    assert not any(name.startswith(forbidden) for name in imports)
    source = (PACKAGE_ROOT / "results.py").read_text(encoding="utf-8")
    assert "np.sqrt" not in source
    assert "evaluate_contour_integral" not in source
    assert "t_stress_from_williams_coefficient" not in source


def test_field_preparation_responsibilities_are_split_by_scientific_role():
    sampling_source = (PACKAGE_ROOT / "sampling.py").read_text(encoding="utf-8")
    auxiliary_fields_source = (PACKAGE_ROOT / "auxiliary_field_preparation.py").read_text(encoding="utf-8")
    mode_decomposition_source = (PACKAGE_ROOT / "mode_decomposition.py").read_text(encoding="utf-8")

    assert "def sample_in_plane_fields" in sampling_source
    assert "def prepare_lefm_auxiliary_fields" not in sampling_source
    assert "def prepare_mode_data" not in sampling_source
    assert "def prepare_lefm_auxiliary_fields" in auxiliary_fields_source
    assert "def prepare_zhao_auxiliary_fields" in auxiliary_fields_source
    assert "def prepare_mode_data" in mode_decomposition_source
    assert "def reconstruct_in_plane_strains" in mode_decomposition_source
    assert "def reconstruct_mode_iii_fields" in mode_decomposition_source
