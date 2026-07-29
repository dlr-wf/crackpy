# Crack-Tip Fields

The `crack_tip_fields` package owns the analytical displacement bases, coefficient contracts, and derived fracture-mechanics quantities for the CJP Model and Williams Expansion.
Its formulation packages preserve coefficient order, sign conventions, and units from the analytical field through downstream results.

## Formulation Contracts

| Formulation | Displacement basis | Coefficients and quantities |
| --- | --- | --- |
| CJP Mode I | `cjp_mode_i_displacement_basis` returns x- and y-displacement responses in the published five-coefficient order. | `CjpModeICoefficients` stores the fitted coefficients, and `CjpModeIQuantities` stores the derived Stress Intensity Factors and normal-stress terms. |
| CJP mixed mode | `cjp_mixed_mode_displacement_basis` returns x- and y-displacement responses in its published five-coefficient order. | `CjpMixedModeCoefficients` stores the fitted coefficients, and `CjpMixedModeQuantities` stores `K_F`, `K_R`, `K_S`, `K_II`, and T-stress. |
| Williams in-plane | `williams_in_plane_displacement_basis` returns all selected symmetric `a_n` responses followed by all antisymmetric `b_n` responses. | `WilliamsInPlaneCoefficients` binds terms to `a_n` and `b_n`; `WilliamsInPlaneQuantities` stores `K_I`, `K_II`, and T-stress. |
| Williams out-of-plane | `williams_out_of_plane_displacement_basis` returns z-displacement responses in selected `c_n` term order. | `WilliamsOutOfPlaneCoefficients` binds terms to `c_n`; `WilliamsOutOfPlaneQuantities` stores `K_III`. |

Callers use `crack_tip_fields.cjp` and `crack_tip_fields.williams` as the supported formulation namespaces.
ODM assembly consumes the coefficient-separated displacement bases.
ODM results carry the matching coefficient and quantity contracts.
Line-integral evaluation uses Williams in-plane coefficients for Bueckner-Chen results and the second-order coefficient transformation for T-stress.

## Scientific Invariants

Displacement bases accept radial coordinates `r` in mm and angular coordinates `phi` in radians.
Each basis row represents one coefficient response over the supplied coordinate shape.
The in-plane formulations use the material shear modulus and Kolosov constant, while the out-of-plane Williams formulation uses the shear modulus.

Williams coefficients for term `n` use MPa mm<sup>1-n/2</sup>.
Stress Intensity Factors use MPa sqrt(m), and T-stress uses MPa.
Positive `b_1` corresponds to negative `K_II` under CrackPy's Williams eigenfield convention.
Derived Williams quantities use NaN when the selected expansion omits the required term.

## Scientific References

- Williams (1957), *On the Stress Distribution at the Base of a Stationary Crack*, DOI [10.1115/1.4011454](https://doi.org/10.1115/1.4011454), Citation Key `williams_1957`.
- Kuna (2013), *Finite Elements in Fracture Mechanics: Theory, Numerics, Applications*, equations 3.43-3.46 and 3.52-3.55, DOI [10.1007/978-94-007-6680-8](https://doi.org/10.1007/978-94-007-6680-8), Citation Key `kuna_fracture_mechanics`.
- Camacho-Reyes et al. (2023), *Study of Effective Stress Intensity Factor through the CJP Model Using Full-Field Experimental Data*, formulas 10 and 11, DOI [10.3390/ma16165705](https://doi.org/10.3390/ma16165705), Citation Key `camacho_reyes_et_al_2023_cjp_mode_i`.
- Christopher et al. (2013), *Extension of the CJP Model to Mixed Mode I and Mode II*, formulas 10 and 11, DOI [10.3221/IGF-ESIS.25.23](https://doi.org/10.3221/IGF-ESIS.25.23), Citation Key `christopher_et_al_2013_cjp_mixed_mode`.
- The project-authored [Crack-tip field wiki](https://github.com/dlr-wf/crackpy/wiki/5.3-Crack-tip-field) records CrackPy's Williams terminology, coefficient units, and `K_I`, `K_II`, and T-stress conventions.
