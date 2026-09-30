# FEniCSx (dolfinx)

Fast prototyping: you write the equation almost as on paper. Strong for fluid flow.

## Install

```bash
conda create -n fenics -c conda-forge fenics-dolfinx
```

FEniCSx needs its own conda environment, not openPASO's `.venv`. openPASO finds environments whose name contains `fenics` or `dolfinx` by itself; otherwise set `FENICS_PYTHON`. Real and complex numbers are separate environments.

Then check that openPASO sees it:

```bash
python check_install.py
```

## What openPASO knows for FEniCSx (dolfinx)

26 kinds of problem. Ask for any of them in plain words; the names below are what the model uses internally.

| Physics | Description | Dimensions | Templates |
|---|---|---|---|
| `biharmonic` | Biharmonic equation (4th order) via interior penalty DG | 2-D | `2d` |
| `cahn_hilliard` | Cahn-Hilliard phase separation: mixed (phi, mu) formulation, double-well potential | 2-D | `2d` |
| `contact` | Contact / obstacle problem via smooth penalty method (Newton iteration) | 2-D | `2d` |
| `convection_diffusion` | Convection-diffusion (SUPG stabilized) | 2-D | `2d` |
| `dg_methods` | Discontinuous Galerkin for advection-dominated diffusion (upwind flux, interior penalty) | 2-D | `2d` |
| `eigenvalue` | Eigenvalue problems (Laplace) via SLEPc | 2-D | `2d` |
| `element_survey` | Element survey: builds every basix element family this FEniCSx install declares and checks each one (its quadrature integrates the reference cell, its mass matrix is SPD, and the space reproduces a constant -- measured on a richer rule than the one that built the matrix). Answers 'which families does this install actually have, and how does each behave' without guessing. | 2-D | `default` |
| `fracture` | Phase-field fracture mechanics. Coupled displacement / damage formulation with a diffuse crack representation (no remeshing). Extensions: PhaseFieldX library. | 2-D, 3-D | `2d` |
| `heat` | Heat conduction (steady / transient) | 2-D, 3-D | `2d_steady`, `2d_transient`, `rectangle` |
| `helmholtz` | Helmholtz equation: -laplacian(u) - k^2*u = f. Acoustic / optical wave propagation. Indefinite system — GMRES or direct, NOT CG. May be complex-valued; needs PETSc compiled with --with-scalar-type=complex. | 2-D, 3-D | `2d` |
| `hyperelasticity` | Nonlinear hyperelasticity (Neo-Hookean, large deformation) | 3-D | `3d` |
| `linear_elasticity` | Linear elasticity (small strain) | 2-D, 3-D | `2d`, `3d`, `plate_hole`, `thick_beam` |
| `magnetostatics` | Magnetostatics: 2D scalar Az curl-curl formulation, spatially varying permeability | 2-D | `2d` |
| `matrix_free_poisson` | Matrix-free conjugate-gradient Poisson solver. Builds A as a callable action_A(x, y) via ufl.action(a, ui) — no global sparse assembly. Mirrors dolfinx demo_poisson_matrix_free.py. | 2-D | `2d` |
| `maxwell` | Maxwell's equations (curl-curl). Requires H(curl) (Nedelec / N1curl, basix.ElementFamily.N1E) elements for tangential continuity. Complex-valued forms need a complex-PETSc build. | 2-D, 3-D | `2d` |
| `mixed_poisson` | Mixed Poisson / Darcy flow (Raviart-Thomas + DG pressure) | 2-D | `2d` |
| `multiphase` | Two-phase flow via Allen-Cahn phase-field (interface tracking, transient) | 2-D | `2d` |
| `navier_stokes` | Incompressible Navier-Stokes (cavity, channel with obstacle; the unsteady wake is a contract: mesh, forces and pictures served, the solve is yours) | 2-D, 3-D | `2d`, `3d`, `channel_cylinder`, `channel_cylinder_transient` |
| `nearly_incompressible_elasticity` | Nearly-incompressible elasticity (Poisson ratio approaching 0.5). Standard primal P1/P2 locks; needs mixed (u, p) Taylor-Hood / MINI or a displacement-pressure split with stable element pair (otherwise volumetric locking). | 2-D, 3-D | `2d` |
| `nonlinear_pde` | General nonlinear PDE with Newton solver and UFL automatic differentiation | 2-D | `2d` |
| `poisson` | Poisson equation / diffusion | 2-D, 3-D | `2d`, `3d`, `l_domain`, `rectangle` |
| `reaction_diffusion` | Two-species reaction-diffusion system (coupled, transient) | 2-D | `2d` |
| `stokes` | Stokes flow with Taylor-Hood P2/P1 (lid-driven cavity) | 2-D | `2d` |
| `stokes_darcy` | Coupled Stokes-Darcy for free fluid / porous medium interaction. Beavers-Joseph-Saffman interface conditions. | 2-D, 3-D | `2d` |
| `thermal_structural` | Coupled thermal-structural (heat -> thermal expansion) | 2-D | `2d` |
| `time_dependent_heat` | Transient heat equation with backward Euler, Robin convective BC, volumetric sources | 2-D | `2d` |
