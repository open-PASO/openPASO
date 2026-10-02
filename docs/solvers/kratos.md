# Kratos Multiphysics

Structures, fluids, coupled problems and particle methods.

## Install

```bash
pip install KratosMultiphysics-all
```

Use the `-all` package. The plain `KratosMultiphysics` 10.4 wheels are labelled for an older system library than they need, so on an older system they install without error and then fail to import.

Then check that openPASO sees it:

```bash
python check_install.py
```

## What openPASO knows for Kratos Multiphysics

22 kinds of problem. Ask for any of them in plain words; the names below are what the model uses internally.

| Physics | Description | Dimensions | Templates |
|---|---|---|---|
| `auxiliary_overview` | [Reference] Kratos auxiliary applications (TrilinosApplication, MetisApplication, MappingApplication, MeshMovingApplication, HDF5Application, ...) — infrastructure that other Kratos analyses depend on. Catalog contains PyPI-publication status, deprecation notes, and FSI hidden-dependency warnings. Not a PDE physics — this is a meta-reference entry; the underlying KNOWLEDGE key is '_auxiliary_overview' (with the leading underscore preserved for backward compatibility). | 2-D, 3-D | `N/A` |
| `cable_net` | Cable and net structures: cables, membranes, form-finding | 3-D | `2d` |
| `constitutive_laws` | Extended constitutive laws: hyperelastic, plasticity, damage, viscoplastic | 2-D, 3-D | `2d` |
| `contact` | Contact mechanics: frictionless Signorini (penalty active-set) — real solve via ContactStructuralMechanicsApplication | 2-D | `2d` |
| `cosimulation` | CoSimulation framework for multi-solver coupling (CoSimulationApplication) | 2-D, 3-D | `2d` |
| `curved_mms` | Curved-geometry manufactured solution: steady diffusion on a Gmsh-meshed annulus, real ConvectionDiffusionApplication solve (LaplacianElement2D3N), prints machine-readable L2_ERROR; theoretical P1 L2 order 2 (live-verified). Meshing is agent-driven via mesh_size or an agent-built .msh file. | 2-D | `annulus_2d` |
| `dam` | Dam engineering: thermal-mechanical, seepage, cracking | 2-D, 3-D | `2d` |
| `dem` | Discrete Element Method for granular/particle simulations (DEMApplication) | 2-D, 3-D | `2d` |
| `dem_structures_coupling` | DEM-FEM coupling: impact on structures, blast | 2-D, 3-D | `2d` |
| `fluid_dynamics` | Incompressible Navier-Stokes via FluidDynamicsApplication, monolithic VMS element (FluidDynamicsApplication) | 2-D | `channel_2d` |
| `fsi_partitioned` | Partitioned Dirichlet-Neumann coupling driven by a KratosFSIApplication convergence accelerator (FSIApplication); a contract: each side's setup and solve are yours | 2-D | `2d` |
| `heat` | Thermal convection-diffusion: steady and transient (ConvectionDiffusionApplication) | 2-D, 3-D | `2d` |
| `heat_transient` | Transient heat conduction with backward Euler time integration | 2-D | `2d` |
| `linear_elasticity` | Structural mechanics: linear/nonlinear, static/dynamic (StructuralMechanicsApplication) | 2-D, 3-D | `2d`, `2d_nonlinear` |
| `mpm` | Material Point Method for large-deformation solid mechanics (MPMApplication) | 2-D, 3-D | `2d` |
| `optimization` | General optimization: gradient-based, adjoint, multi-objective | 2-D, 3-D | `2d` |
| `plasticity` | Elasto-plasticity: MC, DP, VonMises, Tresca + 6 hardening laws (ConstitutiveLawsApplication) | 2-D, 3-D | `3d` |
| `poisson` | Poisson / convection-diffusion (LaplacianElement, EulerianConvDiff) | 2-D, 3-D | `2d` |
| `poromechanics` | Poromechanics: fracture in porous media, dam/tunnel (PoromechanicsApplication) | 2-D, 3-D | `2d` |
| `shallow_water` | Shallow water equations: floods, dam breaks, coastal (ShallowWaterApplication) | 2-D | `2d` |
| `shape_optimization` | Shape optimization with gradient-based methods (ShapeOptimizationApplication) | 2-D, 3-D | `2d` |
| `structural_dynamics` | Dynamic structural analysis with Newmark/Bossak time integration | 2-D | `2d` |
