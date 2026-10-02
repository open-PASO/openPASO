# 4C Multiphysics

Fluid–structure interaction, contact, beams, particles and many coupled problems.

## Install

```bash
export FOURC_BINARY=/path/to/4C
```

Built from source, which takes hours: <https://github.com/4C-multiphysics/4C>. You do not need it to start.

Then check that openPASO sees it:

```bash
python check_install.py
```

## What openPASO knows for 4C Multiphysics

53 kinds of problem. Ask for any of them in plain words; the names below are what the model uses internally.

| Physics | Description | Dimensions | Templates |
|---|---|---|---|
| `ale` | ALE mesh movement | 2-D, 3-D | `ale_2d` |
| `arterial_network` | Arterial network (1-D blood flow) | 1-D | `single_artery_1d` |
| `beam_interaction` | Beam interaction (contact/meshtying) | 3-D | `beam_contact_3d`, `beam_solid_meshtying_3d` |
| `beams` | Beam elements | 2-D, 3-D | `cantilever_static`, `cantilever_dynamic` |
| `brownian_dynamics` | Brownian dynamics of fiber/biopolymer networks | 3-D | `brownian_3d` |
| `cardiac_monodomain` | Cardiac monodomain (electrophysiology) | 3-D | `monodomain_3d` |
| `cardiovascular0d` | 0-D cardiovascular: windkessel, closed-loop circulation, heart models | 3-D | `windkessel_3d` |
| `constraint` | Constraints: MPC, rigid body, periodic BCs, mortar coupling | 2-D, 3-D | `constraint_3d` |
| `contact` | Contact mechanics | 3-D | `inline_penalty_3d`, `penalty_3d` |
| `ehl` | Elastohydrodynamic lubrication | 3-D | `ehl_3d` |
| `electrochemistry` | Electrochemistry (Nernst-Planck) | 2-D, 3-D | `nernst_planck_3d` |
| `fbi` | Fluid-beam interaction (immersed) | 3-D | `penalty_3d` |
| `fluid` | Incompressible Navier-Stokes | 2-D, 3-D | `channel_2d`, `cavity_2d` |
| `fluid_ale` | Fluid on a deforming (ALE) mesh -- the moving-domain setting every FSI and free-surface problem needs | 2-D | `hdg_2d` |
| `fluid_turbulence` | Fluid turbulence: LES (Smagorinsky, dynamic, WALE) and DNS | 2-D, 3-D | `les_channel_3d` |
| `fpsi` | Fluid-porous-structure interaction | 3-D | `monolithic_3d` |
| `fs3i` | FS3I (fluid-structure-scalar-scalar, 5-field) | 3-D | `fs3i_3d` |
| `fsi` | Fluid-structure interaction | 2-D, 3-D | `fsi_2d` |
| `fsi_xfem` | FSI XFEM (fixed-grid fluid-structure) | 3-D | `xfem_fsi_3d` |
| `heat` | Heat conduction | 2-D, 3-D | `heat_2d`, `heat_transient_2d` |
| `input_format` | [Reference] Cross-physics general 4C input pitfalls (ExodusII 1-indexed block IDs, SYMBOLIC_FUNCTION_OF_SPACE_TIME COMPONENT requirement, NUMDOF conflicts on shared FSI/TSI nodes, .yaml-only extension, post_vtu vs IO/RUNTIME VTK OUTPUT, WALL→SOLID rename, etc.). Not a PDE physics — meta-reference entry. Underlying KNOWLEDGE key in data/fourc_knowledge.py is 'input_format'. | 2-D, 3-D | `N/A` |
| `level_set` | Level-set interface tracking | 2-D, 3-D | `advection_2d` |
| `linear_elasticity` | Linear elasticity | 2-D, 3-D | `linear_2d`, `nonlinear_3d` |
| `low_mach` | Low Mach number flow (buoyancy) | 2-D, 3-D | `heated_channel_2d` |
| `lubrication` | Lubrication (Reynolds equation) | 2-D | `slider_bearing_2d` |
| `membrane` | Membrane elements (inflatable, fabric, tissue) | 2-D, 3-D | `membrane_2d` |
| `mixture` | Mixture/composite materials (fiber-reinforced, biological) | 3-D | `mixture_3d` |
| `multiscale` | Multiscale FE-squared (computational homogenisation) | 3-D | `fe2_3d` |
| `particle_dem` | Discrete element method (granular contact, friction, rolling, adhesion, walls) | 1-D, 2-D, 3-D | `settling_3d` |
| `particle_pd` | Peridynamics (bond-based) | 2-D | `plate_2d`, `impact_2d` |
| `particle_sph` | Smoothed particle hydrodynamics | 2-D | `hydrostatic_2d`, `dam_break_2d` |
| `particles` | [Umbrella] Particle-methods family pitfalls (applies to particle_pd, particle_sph, pasi, dem). For specific physics use particle_pd / particle_sph / pasi directly. | 2-D, 3-D | `umbrella` |
| `pasi` | Particle-structure interaction | 3-D | `dem_impact_3d` |
| `plasticity` | Elasto-plasticity: J2/von Mises, Drucker-Prager, GTN damage, crystal plasticity | 2-D, 3-D | `linear_2d`, `nonlinear_3d` |
| `poisson` | Poisson / scalar transport | 2-D, 3-D | `poisson_2d`, `heat_2d`, `poisson_3d` |
| `poroelast_scatra` | Poroelasticity + reacting scalar transport through the pore fluid | 3-D | `homogeneous_3d` |
| `porofluid_elasticity` | Pressure-based porous-media flow monolithically coupled to an elastic skeleton | 3-D | `monolithic_3d` |
| `porofluid_elasticity_scatra` | Multiphase porous flow + deformable skeleton + scalar transport, monolithically coupled | 3-D | `monolithic_3d` |
| `porous_media` | Poroelasticity (Biot/mixture theory, consolidation) | 2-D, 3-D | `single_phase_3d`, `terzaghi_2d`, `consolidation_3d` |
| `reduced_airways` | Reduced-dimensional airways (lung) | 1-D | `airways_1d` |
| `reduced_lung` | Reduced lung model: 1D airways + 0D alveoli + optional 3D parenchyma | 1-D, 3-D | `lung_1d` |
| `scalar_transport` | [Umbrella] Scalar-transport family pitfalls (applies to poisson, heat, electrochemistry, level-set, low-mach scalars). For specific physics use poisson/heat/electrochemistry directly. | 2-D, 3-D | `umbrella` |
| `shell` | Shell elements (Kirchhoff-Love, Reissner-Mindlin) | 3-D | `shell_3d` |
| `ssi` | Structure-scalar interaction (battery/electrode) | 3-D | `monolithic_elch_3d` |
| `ssti` | Structure-scalar-thermo interaction (3-field) | 3-D | `monolithic_3d` |
| `sti` | Scalar-thermo interaction | 3-D | `monolithic_3d` |
| `structural_dynamics` | Structural dynamics | 2-D, 3-D | `genalpha_2d` |
| `structural_mechanics` | [Umbrella] Structural-mechanics family pitfalls (applies to linear_elasticity, plasticity, structural_dynamics, beams, contact). For specific physics use linear_elasticity / plasticity / structural_dynamics directly. | 2-D, 3-D | `umbrella` |
| `thermal` | [Umbrella] Thermal-analysis family pitfalls (applies to heat, thermo, tsi). For specific physics use heat / thermo / tsi directly. | 2-D, 3-D | `umbrella` |
| `thermo` | Pure thermal analysis (standalone heat conduction) | 2-D, 3-D | `thermo_2d`, `thermo_3d` |
| `thermo_transient_mms` | Transient thermal MMS on a fixed mesh for TEMPORAL-order dt-halving studies (One-Step-Theta: order 2 at theta=0.5, order 1 at theta=1) | 2-D | `temporal_mms_2d` |
| `tsi` | Thermo-structure interaction | 2-D, 3-D | `monolithic_3d`, `oneway_3d`, `plane_strain_2d` |
| `xfem_fluid` | XFEM fluid (embedded interfaces) | 3-D | `xfem_3d` |
