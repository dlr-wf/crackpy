# Fracture-Analysis Examples

Run these executable examples as Python modules from the repository root.
The [`fracture_analysis` package README](../../crackpy/fracture_analysis/README.md) describes package ownership and maintainer architecture.

## Single Nodemaps

| Example | Command | Required input | Output location | Scientific purpose |
| --- | --- | --- | --- | --- |
| DIC | `python -m scripts.fracture_analysis.nodemaps.dic` | `test_data/crack_detection/Nodemaps/Dummy2_WPXXX_DummyVersuch_2_dic_results_1_52.txt` | `Fracture_Analysis_DIC_results/` | Evaluate the configured ODM and Line-Integral Evaluation Techniques for one DIC displacement field. |
| Finite element | `python -m scripts.fracture_analysis.nodemaps.fem` | `test_data/simulations/Nodemaps/File_F_10000.0_a_0.5_B_200.0_H_200.0.txt` | `Fracture_Analysis_FE_results/` | Evaluate the configured ODM and Line-Integral Evaluation Techniques for one finite-element displacement field. |

## Synthetic Fields

| Example | Command | Required input | Output location | Scientific purpose |
| --- | --- | --- | --- | --- |
| In-plane Williams field | `python -m scripts.fracture_analysis.synthetic_fields.williams_in_plane` | None; the module defines the synthetic field and material. | `Fracture_Analysis_Williams_results_2D/` | Recover prescribed Mode I, Mode II, and T-stress content from an in-plane Williams field. |
| Williams field with Mode III | `python -m scripts.fracture_analysis.synthetic_fields.williams_in_plane_with_mode_iii` | None; the module defines the synthetic field and material. | `Fracture_Analysis_Williams_results_3D/` | Recover fracture quantities from an in-plane Williams field extended with Mode III displacement. |

## Solver And Integral Demonstrations

| Example | Command | Required input | Output location | Scientific purpose |
| --- | --- | --- | --- | --- |
| ODM Solver Routes | `python -m scripts.fracture_analysis.odm.compare_solver_routes` | None; the module builds one deterministic displacement field. | Standard output | Compare the direct, iterative, and legacy ODM Solver Routes on equivalent in-plane and out-of-plane Williams fits. |
| Modal J-integral | `python -m scripts.fracture_analysis.line_integrals.j_integral_mode_decomposition` | None; the module builds one synthetic mixed-mode Williams field. | Standard output | Compare prescribed and recovered modal J-integrals and stress-intensity factors. |

## Pipelines

| Example | Command | Required input | Output location | Scientific purpose |
| --- | --- | --- | --- | --- |
| DIC with generated contours | `python -m scripts.fracture_analysis.pipelines.dic_generated_contours` | DIC nodemaps in `test_data/crack_detection/Nodemaps/` and cached `ParallelNets` and `UNetPath` weights. The first run downloads missing weights from Zenodo and requires network access. | `Fracture_Analysis_Pipeline_DIC_results_auto/` | Detect cracks, generate integration contours, and evaluate a DIC nodemap series. |
| DIC with predefined contours | `python -m scripts.fracture_analysis.pipelines.dic_predefined_contours` | DIC nodemaps in `test_data/crack_detection/Nodemaps/` and cached `ParallelNets` and `UNetPath` weights. The first run downloads missing weights from Zenodo and requires network access. | `Fracture_Analysis_Pipeline_DIC_results_predef/` | Detect cracks and evaluate a DIC nodemap series with predefined integration contours. |
| Finite element | `python -m scripts.fracture_analysis.pipelines.fem` | Finite-element nodemaps in `test_data/simulations/Nodemaps/` and `test_data/simulations/crack_info_by_nodemap.txt`. | `Fracture_Analysis_Pipeline_FE_results/` | Evaluate a finite-element nodemap series using the supplied crack-tip positions and angles. |
