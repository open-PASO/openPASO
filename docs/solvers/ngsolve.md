# NGSolve

High-order elements; strong for electromagnetics, acoustics and wave problems.

## Install

```bash
pip install ngsolve
```

Installs in seconds into openPASO's own `.venv`.

Then check that openPASO sees it:

```bash
python check_install.py
```

## What openPASO knows for NGSolve

22 kinds of problem. Ask for any of them in plain words; the names below are what the model uses internally.

| Physics | Description | Dimensions | Templates |
|---|---|---|---|
| `contact` | Unilateral contact / obstacle problem via penalty method on elastic domain | 2-D | `2d` |
| `convection_diffusion` | Convection-diffusion with DG upwind stabilization | 2-D | `2d_dg` |
| `dg_methods` | Interior-penalty DG (SIPG) for advection-diffusion using L2 dglagrange space | 2-D | `2d` |
| `eigenvalue` | Eigenvalue problems (Laplace, elasticity) via ArnoldiSolver | 2-D | `2d` |
| `hdivdiv` | HDivDiv Hellan-Herrmann-Johnson for Kirchhoff plate bending / biharmonic | 2-D | `2d` |
| `heat` | Heat conduction (steady and transient with implicit Euler) | 2-D, 3-D | `2d`, `2d_steady`, `2d_transient` |
| `helmholtz` | Helmholtz equation with PML (complex-valued) | 2-D | `2d` |
| `hyperelasticity` | Nonlinear hyperelasticity (Neo-Hookean) via SymbolicEnergy | 2-D, 3-D | `2d`, `3d` |
| `linear_elasticity` | Linear elasticity (plane strain / 3D) with VectorH1 | 2-D, 3-D | `2d`, `3d` |
| `maxwell` | Maxwell's equations with HCurl (Nedelec) elements | 3-D | `3d_magnetostatics` |
| `mhd` | Magnetohydrodynamics: coupled NS + Maxwell, 2.5-D low-Rm Hartmann problem | 2-D | `2d` |
| `mixed_poisson` | Mixed Poisson with H(div)/L2 (flux recovery) | 2-D | `2d` |
| `navier_stokes` | Incompressible Navier-Stokes (IMEX time-stepping) | 2-D | `2d` |
| `nonlinear_elasticity` | Large-deformation Neo-Hookean hyperelasticity with load stepping and Cauchy stress output | 2-D, 3-D | `2d`, `3d` |
| `phase_field` | Phase-field: Allen-Cahn (interface motion) and fracture (Bourdin staggered scheme) | 2-D | `2d`, `fracture_2d` |
| `plasticity` | Elasto-plasticity with isotropic hardening (J2/von Mises) | 2-D | `2d` |
| `poisson` | Poisson equation -Δu = f with arbitrary-order H1 elements | 2-D, 3-D | `2d`, `3d` |
| `space_survey` | Assemble a mass form on every NGSolve space and verify it is symmetric positive definite | 2-D | `2d` |
| `stokes` | Stokes flow with Taylor-Hood P2/P1 or HDG | 2-D, 3-D | `2d`, `2d_hdg` |
| `surface_pde` | PDE on curved surface manifold (Laplace-Beltrami) | 3-D | `3d` |
| `thermal_structural` | Coupled thermal-structural (heat -> elasticity with thermal strain) | 2-D | `2d` |
| `time_dependent_ns` | Transient incompressible Navier-Stokes with IMEX splitting (full channel/cavity) | 2-D | `2d` |
