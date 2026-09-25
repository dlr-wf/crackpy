"""Architectural contract for public crack-tip-field and ODM ownership, exports, dependency
direction, and retired transition paths.
"""

import ast
import importlib.util
from pathlib import Path

import crackpy.fracture_analysis as fracture_analysis
import crackpy.fracture_analysis.crack_tip_fields as crack_tip_fields
import crackpy.fracture_analysis.crack_tip_fields.cjp as cjp
import crackpy.fracture_analysis.crack_tip_fields.williams as williams
import crackpy.fracture_analysis.odm as odm
import crackpy.fracture_analysis.odm.assembly as assembly
import crackpy.fracture_analysis.odm.results as odm_results
import crackpy.fracture_analysis.odm.solvers as odm_solvers
from crackpy.fracture_analysis.crack_tip_fields.cjp.basis import (
    cjp_mixed_mode_displacement_basis,
    cjp_mode_i_displacement_basis,
)
from crackpy.fracture_analysis.crack_tip_fields.cjp.coefficients import (
    CjpMixedModeCoefficients,
    CjpModeICoefficients,
)
from crackpy.fracture_analysis.crack_tip_fields.cjp.quantities import (
    CjpMixedModeQuantities,
    CjpModeIQuantities,
    derive_cjp_mixed_mode_fracture_quantities,
    derive_cjp_mode_i_fracture_quantities,
)
from crackpy.fracture_analysis.crack_tip_fields.williams.basis import (
    williams_in_plane_displacement_basis,
    williams_out_of_plane_displacement_basis,
)
from crackpy.fracture_analysis.crack_tip_fields.williams.coefficients import (
    WilliamsInPlaneCoefficients,
    WilliamsOutOfPlaneCoefficients,
)
from crackpy.fracture_analysis.crack_tip_fields.williams.quantities import (
    WilliamsInPlaneQuantities,
    WilliamsOutOfPlaneQuantities,
    derive_williams_in_plane_fracture_quantities,
    derive_williams_out_of_plane_fracture_quantities,
)
from crackpy.fracture_analysis.odm.assembly import (
    CjpAssembly,
    LinearSystem,
    WilliamsAssembly,
    assemble_cjp,
    assemble_williams,
)


def _import_targets(path: Path) -> set[str]:
    """Return syntactic import targets declared by one Python module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    targets = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            targets.add(module)
            targets.update(f"{module}.{alias.name}" for alias in node.names)
    return targets


def test_public_packages_export_only_the_approved_option_c_interfaces() -> None:
    assert crack_tip_fields.__all__ == []
    assert cjp.__all__ == [
        "CjpModeICoefficients",
        "CjpModeIQuantities",
        "CjpMixedModeCoefficients",
        "CjpMixedModeQuantities",
        "cjp_mode_i_displacement_basis",
        "cjp_mixed_mode_displacement_basis",
    ]
    assert williams.__all__ == [
        "WilliamsInPlaneCoefficients",
        "WilliamsInPlaneQuantities",
        "WilliamsOutOfPlaneCoefficients",
        "WilliamsOutOfPlaneQuantities",
        "williams_in_plane_displacement_basis",
        "williams_out_of_plane_displacement_basis",
    ]
    assert odm.__all__ == ["CoefficientFitResult", "OdmFitResult", "SolverRoute"]
    assert odm_results.__all__ == ["CoefficientFitResult", "OdmFitResult"]
    assert odm_solvers.__all__ == [
        "SolverRoute",
        "solve_direct",
        "solve_iterative",
        "solve_legacy",
        "solve_coefficient_fit",
        "to_optimize_result",
        "coefficient_fit_from_optimize_result",
    ]


def test_public_objects_are_owned_by_their_approved_modules() -> None:
    expected_owners = {
        CjpModeICoefficients: "crackpy.fracture_analysis.crack_tip_fields.cjp.coefficients",
        CjpMixedModeCoefficients: "crackpy.fracture_analysis.crack_tip_fields.cjp.coefficients",
        CjpModeIQuantities: "crackpy.fracture_analysis.crack_tip_fields.cjp.quantities",
        CjpMixedModeQuantities: "crackpy.fracture_analysis.crack_tip_fields.cjp.quantities",
        derive_cjp_mode_i_fracture_quantities: "crackpy.fracture_analysis.crack_tip_fields.cjp.quantities",
        derive_cjp_mixed_mode_fracture_quantities: "crackpy.fracture_analysis.crack_tip_fields.cjp.quantities",
        cjp_mode_i_displacement_basis: "crackpy.fracture_analysis.crack_tip_fields.cjp.basis",
        cjp_mixed_mode_displacement_basis: "crackpy.fracture_analysis.crack_tip_fields.cjp.basis",
        WilliamsInPlaneCoefficients: "crackpy.fracture_analysis.crack_tip_fields.williams.coefficients",
        WilliamsOutOfPlaneCoefficients: "crackpy.fracture_analysis.crack_tip_fields.williams.coefficients",
        WilliamsInPlaneQuantities: "crackpy.fracture_analysis.crack_tip_fields.williams.quantities",
        WilliamsOutOfPlaneQuantities: "crackpy.fracture_analysis.crack_tip_fields.williams.quantities",
        derive_williams_in_plane_fracture_quantities: "crackpy.fracture_analysis.crack_tip_fields.williams.quantities",
        derive_williams_out_of_plane_fracture_quantities: "crackpy.fracture_analysis.crack_tip_fields.williams.quantities",
        williams_in_plane_displacement_basis: "crackpy.fracture_analysis.crack_tip_fields.williams.basis",
        williams_out_of_plane_displacement_basis: "crackpy.fracture_analysis.crack_tip_fields.williams.basis",
        odm.CoefficientFitResult: "crackpy.fracture_analysis.odm.results",
        odm.OdmFitResult: "crackpy.fracture_analysis.odm.results",
    }

    assert {obj: obj.__module__ for obj in expected_owners} == expected_owners


def test_odm_assembly_owns_the_accepted_system_names() -> None:
    expected_owners = {
        LinearSystem: "crackpy.fracture_analysis.odm.assembly",
        CjpAssembly: "crackpy.fracture_analysis.odm.assembly",
        WilliamsAssembly: "crackpy.fracture_analysis.odm.assembly",
        assemble_cjp: "crackpy.fracture_analysis.odm.assembly",
        assemble_williams: "crackpy.fracture_analysis.odm.assembly",
    }
    assert {value: value.__module__ for value in expected_owners} == expected_owners


def test_namespace_loaders_do_not_become_convenience_export_surfaces() -> None:
    exported_field_types = {
        name
        for name, value in vars(crack_tip_fields).items()
        if isinstance(value, type)
    }
    assert exported_field_types == set()
    for name in (*cjp.__all__, *williams.__all__, *odm.__all__):
        assert not hasattr(fracture_analysis, name)


def test_crack_tip_field_packages_do_not_import_odm_facades_or_compatibility() -> None:
    forbidden_components = {"odm", "analysis", "optimization"}
    imported_targets = {
        target
        for source in Path(crack_tip_fields.__path__[0]).rglob("*.py")
        for target in _import_targets(source)
    }

    assert not {
        target
        for target in imported_targets
        if forbidden_components.intersection(target.split("."))
        or "compatibility" in target
    }


def test_odm_namespace_does_not_load_facades_runners_or_compatibility() -> None:
    init_targets = _import_targets(Path(odm.__file__))
    forbidden_components = {"analysis", "optimization", "runners", "_compatibility"}

    assert not {
        target
        for target in init_targets
        if forbidden_components.intersection(target.split("."))
    }


def test_transition_modules_and_public_polar_basis_fields_are_removed() -> None:
    transition_modules = (
        "crackpy.fracture_analysis._crack_tip_fields",
        "crackpy.fracture_analysis._odm_grid_interpolation",
        "crackpy.fracture_analysis._odm_fit_systems",
        "crackpy.fracture_analysis._odm_solver",
        "crackpy.fracture_analysis._odm_result_builders",
        "crackpy.fracture_analysis._odm_compatibility",
        "crackpy.fracture_analysis.odm_results",
    )

    assert all(importlib.util.find_spec(module_name) is None for module_name in transition_modules)
    assert not hasattr(assembly, "PolarBasisFields")
