# Fracture-Mechanics Functionals

The `functionals` package owns pure J-integral, interaction-integral, Bueckner-Chen Integral, and Stress-Difference Method expressions.
Its kernels receive prepared measured or auxiliary fields and return contour-integrand terms or mapped fracture-mechanics quantities.

## Functional Contracts

| Family | Inputs and result |
| --- | --- |
| J-Integral | Measured stresses, strains, displacement gradients, and contour normals produce in-plane or Mode III integrand terms and Stress Intensity Factor mappings. |
| Interaction Integral | Measured fields and Mode I, Mode II, or Zhao auxiliary fields produce integrand terms and signed Stress Intensity Factor or T-stress mappings. |
| Bueckner-Chen Integral | Measured stresses and displacements with a complementary Williams eigenfield produce reciprocal-work terms and a symmetric or antisymmetric Williams coefficient. |
| Stress-Difference Method | Crack-parallel and crack-opening normal stresses on the crack-extension line produce pointwise T-stress estimates. |

`IntegrandTerms` groups coefficient fields evaluated at contour-segment midpoints:

- `integrated_over_dy` is multiplied by each segment's signed vertical increment.
- `added_over_ds` is added after multiplication by positive segment length.
- `subtracted_over_ds` is subtracted after multiplication by positive segment length.

[`line_integrals`](../line_integrals/README.md) owns contour geometry, field preparation, quadrature, technique execution, and contour-wise results.

## Scientific Invariants

Functional kernels use the crack-tip Cartesian frame.
The x-axis follows prospective crack extension, the y-axis is normal to the crack plane, and supplied contour normals point outward.

Stress inputs use MPa, displacement inputs use mm, and J-integral values use N/mm.
Reported Stress Intensity Factors use MPa sqrt(m), T-stress uses MPa, and Williams coefficient units depend on term order.

Interaction-integral mappings preserve the sign of modal contributions.
Energy-based mappings return nonnegative Stress Intensity Factor magnitudes.
Negative modal J-integral values propagate NaN.

## Scientific References

- Rice (1968), equation 1, DOI [10.1115/1.3601206](https://doi.org/10.1115/1.3601206), Citation Key `rice_1968_j_integral`.
- Molteno and Becker (2015), equations 8-11 and 16-17, DOI [10.1111/str.12166](https://doi.org/10.1111/str.12166), Citation Key `molteno_becker_2015_j_integral_decomposition`.
- Breitbarth et al. (2019), equations 3, 9, and 10, DOI [10.3221/IGF-ESIS.49.02](https://doi.org/10.3221/IGF-ESIS.49.02), Citation Key `breitbarth_et_al_2019_dic_integrals`.
- Kuna (2013), equations 6.81 and 6.91-6.94, DOI [10.1007/978-94-007-6680-8](https://doi.org/10.1007/978-94-007-6680-8), Citation Key `kuna_fracture_mechanics`.
- Chen (1985), DOI [10.1016/0013-7944(85)90131-6](https://doi.org/10.1016/0013-7944(85)90131-6), Citation Key `chen_1985_path_independent_integrals`.
- Zhao, Tong, and Byrne (2001), equations 4a-6, DOI [10.1023/A:1011016720630](https://doi.org/10.1023/A:1011016720630), Citation Key `zhao_et_al_2001_corner_cracks`.
- Yang and Ravi-Chandar (1999), equation 7, DOI [10.1016/S0013-7944(99)00082-X](https://doi.org/10.1016/S0013-7944(99)00082-X), Citation Key `yang_ravi_chandar_1999_stress_difference`.
- The project-authored [Fracture Analysis wiki](https://github.com/dlr-wf/crackpy/wiki/5.1-Fracture-Analysis) records these methods and reported quantities in CrackPy.
