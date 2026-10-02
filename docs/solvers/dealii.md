# deal.II

Adaptive mesh refinement and very large parallel computations, in C++.

## Install

```bash
sudo apt install libdeal.ii-dev
```

Version 9.3 or newer is needed for all examples (Ubuntu 20.04 ships 9.1.1). If you build it yourself, use `-DCMAKE_BUILD_TYPE=DebugRelease`: a Release build removes the internal checks, so mistakes fail silently.

Then check that openPASO sees it:

```bash
python check_install.py
```

## What openPASO knows for deal.II

28 kinds of problem. Ask for any of them in plain words; the names below are what the model uses internally.

| Physics | Description | Dimensions | Templates |
|---|---|---|---|
| `advection_dg` | Pure DG advection (step-9, step-12). Distinct from dg_advection_reaction (step-12, step-39) — advection_dg covers step-9 transport without reaction term. DoFTools::make_flux_sparsity_pattern required for face coupling. | 2-D | `2d` |
| `contact` | Contact / variational inequalities (step-41, step-42). Active-set strategy. Related to obstacle_problem (the dealii backend's primary name for this class) — distinct deep_knowledge entry kept for active-set-strategy specifics. | 2-D, 3-D | `2d` |
| `convection_diffusion` | Convection-diffusion with SUPG stabilization (step-9 based) | 2-D | `2d` |
| `dg_advection_reaction` | DG advection-reaction (step-12, step-39) | 2-D | `2d` |
| `dg_transport` | Discontinuous Galerkin for advection problems (step-12 pattern) | 2-D | `2d` |
| `eigenvalue` | Eigenvalue problems via SLEPc (step-36 inspired) | 2-D | `2d` |
| `element_survey` | Finite-element survey: builds every FE_* class this deal.II install declares and checks each one (quadrature integrates the cell, mass matrix is SPD, space reproduces a constant). Answers 'which elements does this build actually have, and how does each behave' without guessing from docs. | 2-D, 3-D | `default` |
| `error_estimation` | Adaptive error estimation, Kelly + AMR (step-6, step-14) | 2-D | `2d` |
| `heat` | Heat equation (transient step-26 and steady-state, with rectangle) | 2-D | `2d_transient`, `2d_steady`, `rectangle` |
| `helmholtz` | Helmholtz equation (complex-valued, step-29 inspired) | 2-D | `2d` |
| `hp_adaptive` | hp-adaptive FEM with automatic smoothness estimation (step-27 pattern) | 2-D | `2d` |
| `hyperelasticity` | Finite-strain hyperelasticity with Neo-Hookean material (step-44 pattern) | 3-D | `3d` |
| `linear_elasticity` | Linear elasticity (step-8, with thick beam variant) | 2-D | `2d`, `thick_beam` |
| `matrix_free` | Matrix-free high-performance FEM (step-37, step-59) | 2-D, 3-D | `2d` |
| `mixed_laplacian` | Mixed Laplacian with Raviart-Thomas H(div) (step-20) | 2-D | `2d` |
| `multigrid` | Geometric multigrid preconditioner (step-16, step-50) | 2-D, 3-D | `2d` |
| `navier_stokes` | Navier-Stokes: stationary + transient (step-57, step-35) | 2-D, 3-D | `2d` |
| `nonlinear` | Nonlinear PDE (minimal surface, step-15, Newton method) | 2-D | `2d_minimal_surface` |
| `nonlinear_elasticity` | Nonlinear solid mechanics (step-44). Neo-Hookean three-field (u, p, J) formulation for quasi-incompressible materials. Distinct from hyperelasticity (broader catalog) — this entry focuses on the step-44 three-field method. | 3-D | `3d` |
| `obstacle_problem` | Variational inequality / contact (step-41) | 2-D | `2d` |
| `parallel_poisson` | MPI-parallel Poisson solver with p4est (step-40 pattern) | 2-D | `2d` |
| `phase_field` | Phase-field / ADR with SUPG (step-63) | 2-D | `2d` |
| `poisson` | Poisson / Laplace equation (step-3/6/7, with AMR, L-domain, rectangle, 3D mixed Dirichlet-Neumann MMS convergence study) | 2-D, 3-D | `2d`, `3d`, `l_domain`, `rectangle`, `2d_adaptive`, `3d_mixed_bc` |
| `stokes` | Stokes flow (step-22, Taylor-Hood Q2/Q1, block preconditioner) | 2-D | `2d` |
| `time_dependent_heat` | Transient heat with AMR (step-26) | 2-D | `2d` |
| `time_dependent_ns` | Transient Boussinesq flow (step-35) | 2-D | `2d` |
| `time_dependent_wave` | Wave equation (step-23, step-48) | 2-D, 3-D | `2d` |
| `wave` | Wave equation with Newmark time integration (step-23 inspired) | 2-D | `2d` |
