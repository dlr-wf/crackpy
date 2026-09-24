"""Compatibility imports for analytical crack-tip fields."""

from crackpy.fracture_analysis.crack_tip_fields.auxiliary import (
    get_crack_nearfield,
    get_zhao_solutions,
)
from crackpy.fracture_analysis.crack_tip_fields.cjp.solutions import (
    cjp_displ_field_mixedmode,
    cjp_displ_field_modeI,
    cjp_stress_field_mixedmode,
    cjp_stress_field_modeI,
)
from crackpy.fracture_analysis.crack_tip_fields.williams.solutions import (
    unit_of_williams_coefficients,
)
from crackpy.fracture_analysis.crack_tip_fields.williams.solutions import (
    williams_combined_stress_field as williams_stress_field_3d,
)
from crackpy.fracture_analysis.crack_tip_fields.williams.solutions import (
    williams_in_plane_displacement_field as williams_displ_field_xy,
)
from crackpy.fracture_analysis.crack_tip_fields.williams.solutions import (
    williams_in_plane_eigenfunction as eigenfunction,
)
from crackpy.fracture_analysis.crack_tip_fields.williams.solutions import (
    williams_in_plane_stress_field as williams_stress_field,
)
from crackpy.fracture_analysis.crack_tip_fields.williams.solutions import (
    williams_out_of_plane_displacement_field as williams_displ_field_z,
)

__all__ = [
    "cjp_displ_field_mixedmode",
    "cjp_displ_field_modeI",
    "cjp_stress_field_mixedmode",
    "cjp_stress_field_modeI",
    "eigenfunction",
    "get_crack_nearfield",
    "get_zhao_solutions",
    "unit_of_williams_coefficients",
    "williams_displ_field_xy",
    "williams_displ_field_z",
    "williams_stress_field",
    "williams_stress_field_3d",
]
