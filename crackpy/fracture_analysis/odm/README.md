# Over-Deterministic Method

The `odm` package owns displacement sampling, linear fitting-system assembly, fit execution, Solver Routes, numerical fit evidence, and typed Over-Deterministic Method results.
It consumes the CJP and Williams displacement bases owned by [`crack_tip_fields`](../crack_tip_fields/README.md).

## Execution Contract

```mermaid
flowchart LR
    %% truth: node Grid kind=function symbol=build_optimization_grid
    Grid["Build polar grid"]

    %% truth: node Sample kind=function symbol=prepare_interpolated_displacement_grid
    Sample["Sample displacements"]

    %% truth: node Assemble kind=concept
    Assemble["Assemble model system"]

    %% truth: node Solve kind=function symbol=solve_coefficient_fit role=decision
    Solve{"Select Solver Route"}

    %% truth: node FitEvidence kind=schema symbol=CoefficientFitResult role=artifact
    FitEvidence[/"Coefficient-fit result"/]

    %% truth: node ResultCompletion kind=concept
    ResultCompletion["Complete model result"]

    %% truth: node OdmResult kind=schema symbol=OdmFitResult role=artifact
    OdmResult[/"ODM Technique Result"/]

    Grid --> Sample
    Sample --> Assemble
    Assemble --> Solve
    Solve -->|direct, iterative, or legacy| FitEvidence
    FitEvidence --> ResultCompletion
    ResultCompletion --> OdmResult
```

The package exports `SolverRoute`, `CoefficientFitResult`, and `OdmFitResult`.
The direct route solves an assembled linear system with the GELSS least-squares driver.
The iterative route applies SciPy least squares to the same matrix and exact Jacobian.
The legacy route evaluates the established residual and Jacobian callbacks.

`CoefficientFitResult` owns immutable fitted coefficients, residuals, cost, completion evidence, and available matrix evidence.
`OdmFitResult` combines that evidence with formulation-specific coefficient and quantity contracts.
Its `completed`, `failed`, and `skipped` states describe technique execution.

## Fitting Invariants

One fit keeps the crack-tip position, material properties, polar grid, selected Williams terms, and valid displacement mask fixed.
CJP and Williams displacements are linear in their coefficients under those conditions.

CJP Mode I and mixed-mode systems use one valid in-plane displacement mask while retaining separate coefficient columns.
Williams in-plane and out-of-plane systems retain independent masks and coefficient orders.
In-plane Williams columns contain all selected symmetric coefficients followed by all selected antisymmetric coefficients.

The established `Optimization` facade converts coefficient-fit evidence into independent mutable SciPy results.
Its bounded interpolation cache may reuse sampling geometry across equivalent optimization instances.
`FractureAnalysis` projects authoritative ODM Technique Results through private compatibility adapters at the analysis seam.
