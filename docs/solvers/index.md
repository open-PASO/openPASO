# Solvers

A **solver** is the program that actually computes the physics. openPASO can drive nine of them. **You need only one to start**, and scikit-fem is the easiest.

| Solver | Good at | Physics openPASO knows | Install |
|---|---|---|---|
| [scikit-fem](skfem.md) | Pure Python, and the easiest to start with. You see and control every step of the assembly. | 24 | `pip install scikit-fem` |
| [NGSolve](ngsolve.md) | High-order elements; strong for electromagnetics, acoustics and wave problems. | 22 | `pip install ngsolve` |
| [Kratos Multiphysics](kratos.md) | Structures, fluids, coupled problems and particle methods. | 22 | `pip install KratosMultiphysics-all` |
| [DUNE-fem](dune.md) | Discontinuous Galerkin methods and adaptive meshes. | 18 | `pip install dune-fem mpi4py` |
| [FEniCSx (dolfinx)](fenics.md) | Fast prototyping: you write the equation almost as on paper. Strong for fluid flow. | 26 | `conda create -n fenics -c conda-forge fenics-dolfinx` |
| [deal.II](dealii.md) | Adaptive mesh refinement and very large parallel computations, in C++. | 28 | `sudo apt install libdeal.ii-dev` |
| [FEBio](febio.md) | Biomechanics: soft tissue, cartilage, muscle. | 18 | `export FEBIO_BINARY=/path/to/febio4` |
| [4C Multiphysics](fourc.md) | Fluid–structure interaction, contact, beams, particles and many coupled problems. | 53 | `export FOURC_BINARY=/path/to/4C` |
| [SPARTA](sparta.md) | Rarefied gas with the particle method DSMC, where the usual flow equations stop working. Experimental. | 11 | `export SPARTA_BINARY=/path/to/spa_serial` |

Solvers you have not installed are simply reported as missing; openPASO uses the others.
