# DUNE-fem

Discontinuous Galerkin methods and adaptive meshes.

## Install

```bash
pip install dune-fem mpi4py
```

`mpi4py` is a hidden requirement: without it the first import stops with "Please run pip install mpi4py before rerunning your Dune script."

Then check that openPASO sees it:

```bash
python check_install.py
```

## What openPASO knows for DUNE-fem

18 kinds of problem. Ask for any of them in plain words; the names below are what the model uses internally.

| Physics | Description | Dimensions | Templates |
|---|---|---|---|
| `adaptive_poisson` | h-adaptive Poisson with residual error estimator | 2-D | `2d` |
| `dg_advection` | DG method for pure advection equation (upwind flux) | 2-D | `2d` |
| `dg_advection_diffusion` | Steady advection-diffusion with upwind flux and SIPG diffusion on triangles | 2-D | `2d` |
| `eigenvalue` | Eigenvalue problem -Δu = λu via shift-invert inverse iteration | 2-D | `2d` |
| `heat` | Steady heat conduction (UFL) | 2-D | `2d` |
| `helmholtz` | Helmholtz equation -Δu - k²u = f with MMS verification | 2-D | `2d` |
| `hyperelasticity` | Neo-Hookean finite-strain hyperelasticity with automatic Newton via UFL | 2-D | `2d` |
| `linear_elasticity` | Linear elasticity (UFL vector space) | 2-D | `2d` |
| `maxwell` | Maxwell equations: 2-D TE-mode scalar Helmholtz proxy (-Δu - k²u = f); full H(curl) via NGSolve | 2-D | `2d` |
| `mixed_methods` | Mixed Poisson with Raviart-Thomas RT0 flux and DG-P0 pressure (H(div) x L²) | 2-D | `2d` |
| `navier_stokes` | Incompressible Navier-Stokes via Picard iteration (lid-driven cavity) | 2-D | `2d` |
| `nonlinear` | Nonlinear PDE via Newton method (UFL) | 2-D | `2d` |
| `poisson` | Poisson equation -Δu = f (UFL forms, DUNE backend) | 2-D, 3-D | `2d` |
| `poisson_mms` | 3D variable-coefficient Poisson manufactured-solution (MMS) convergence family — -div(kappa grad u) = f on [0,L]^3, affine kappa, exact Dirichlet data, uniform refinement with per-level L2/H1 error lines; theoretical L2 order k+1 / H1 order k | 3-D | `3d_varcoeff` |
| `reaction_diffusion` | Reaction-diffusion (transient) | 2-D | `2d` |
| `registry_survey` | Registry survey: builds every entry this dune-fem install declares across all 8 axes (spaces, schemes, solvers, storages, models, operators, functions, views) and checks each against something independent of the run. Also records the first-use JIT cost, which is 85-200s per new space and 0.1s once cached. | 2-D | `default` |
| `stokes` | Stokes flow with Uzawa iteration (UFL) | 2-D | `2d` |
| `time_dependent_heat` | Transient heat du/dt - alpha*Δu = f via implicit Euler time-stepping | 2-D | `2d` |
