# scikit-fem

Pure Python, and the easiest to start with. You see and control every step of the assembly.

## Install

```bash
pip install scikit-fem
```

Installs in seconds into openPASO's own `.venv`.

Then check that openPASO sees it:

```bash
python check_install.py
```

## What openPASO knows for scikit-fem

24 kinds of problem. Ask for any of them in plain words; the names below are what the model uses internally.

| Physics | Description | Dimensions | Templates |
|---|---|---|---|
| `adaptive_poisson` | h-adaptive Poisson with Babuška-Rheinboldt residual estimator on L-shape (canonical re-entrant-corner test) | 2-D | `2d` |
| `biharmonic` | Biharmonic / Kirchhoff plate bending (Morley element) | 2-D | `2d` |
| `contact` | Linearized frictionless contact between a 2D elastic block and a rigid foundation, via Picard iteration on the active set. Matches scikit-fem ex04. | 2-D | `2d` |
| `convection_diffusion` | Convection-diffusion (SUPG or DG interior penalty) | 2-D | `2d` |
| `dg_methods` | Discontinuous Galerkin for advection using ElementDG and InteriorFacetBasis | 2-D | `2d` |
| `eigenvalue` | Eigenvalue problems (Laplace, elasticity) via scipy eigsh | 2-D | `2d` |
| `element_survey` | Measure the L2 convergence order of every scikit-fem element that can carry Poisson | 1-D, 2-D, 3-D | `poisson` |
| `heat` | Steady heat conduction | 2-D | `2d`, `2d_steady` |
| `heat_transient` | Time-dependent heat equation with backward Euler | 2-D | `2d` |
| `helmholtz` | Helmholtz equation -Δu - k²u = f with complex arithmetic and absorbing BC | 2-D | `2d` |
| `hydraulic_resistance` | Stokes flow through a 2D rectangular channel; computes resistance R=ΔP/Q against the Poiseuille closed-form 12μL/H³. Matches scikit-fem ex29. | 2-D | `2d` |
| `hyperelasticity` | Neo-Hookean hyperelasticity with Newton iteration (manual assembly) | 2-D | `2d` |
| `linear_elasticity` | Linear elasticity (plane strain) | 2-D | `2d` |
| `mixed_elements` | Survey the H(div), H(curl) and L2/DG element families, each through its own formulation | 1-D, 2-D, 3-D | `survey` |
| `mixed_poisson` | Mixed Poisson with Raviart-Thomas + DG (flux-conservative) | 2-D | `2d` |
| `navier_stokes` | Navier-Stokes flow with Newton iteration (Taylor-Hood P2/P1) | 2-D | `2d` |
| `nonlinear` | Nonlinear PDE with Newton iteration (manual Newton loop) | 2-D | `2d` |
| `point_source` | Poisson with Dirac-delta point source — discrete RHS is N_i(x0) (Kronecker e_node for mesh-coincident source). Matches scikit-fem ex17 + ex38. | 2-D | `2d` |
| `poisson` | Poisson equation -Δu = f (assembly-level) | 2-D, 3-D | `2d`, `2d_tri`, `3d` |
| `reaction_diffusion` | Reaction-diffusion system (Schnakenberg/Turing patterns) with Newton time-stepping | 2-D | `2d` |
| `schrodinger` | 1D stationary Schrödinger eigenvalue problem -½ψ'' + V(x)ψ = Eψ. Default quantum harmonic oscillator V=½x² with analytic E_n = n+½. Matches scikit-fem ex39. | 1-D | `1d` |
| `stokes` | Stokes flow with Taylor-Hood P2/P1 or Mini element | 2-D | `2d` |
| `time_dependent` | General time-dependent PDE with theta-method (backward Euler / Crank-Nicolson) | 2-D | `2d` |
| `wave` | 2D scalar wave equation u_tt - c^2 Δu = 0 with explicit central-difference time integration and lumped mass (no per-step linear solve) | 2-D | `2d` |
