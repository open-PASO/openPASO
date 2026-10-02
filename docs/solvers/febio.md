# FEBio

Biomechanics: soft tissue, cartilage, muscle.

## Install

```bash
export FEBIO_BINARY=/path/to/febio4
```

Download the program from <https://febio.org/downloads/>, then point openPASO at the program file itself.

Then check that openPASO sees it:

```bash
python check_install.py
```

## What openPASO knows for FEBio

18 kinds of problem. Ask for any of them in plain words; the names below are what the model uses internally.

| Physics | Description | Dimensions | Templates |
|---|---|---|---|
| `active_contraction` | Active contractile fibers on a passive elastic base (cardiac chamber, skeletal muscle, peristalsis) | 3-D | `3d_fiber` |
| `biphasic` | Biphasic poroelasticity (solid + fluid phases) | 3-D | `3d_confined` |
| `biphasic_fsi` | Coupled biphasic tissue + free-fluid FSI (blood-tissue perfusion, drug elution, cartilage-synovial fluid) | 3-D | `3d_block` |
| `damage` | Continuum damage mechanics — progressive stiffness degradation under repeated loading (tissue tearing, cartilage wear, elastomer fatigue) | 3-D | `3d_cycle` |
| `elasticity_mms` | 3D linear-elasticity manufactured-solution (MMS) verification family — structured hex8 cube [0,L]^3, exact trig body force -div(sigma(u*)), per-node Dirichlet maps, displacement L2 order 2 expected | 3-D | `3d_cube_hex8` |
| `fiber_reinforced` | Anisotropic fiber-reinforced hyperelasticity (HGO, transversely isotropic) — arterial wall, ligament, tendon, myocardium | 3-D | `3d_hgo` |
| `fluid` | Incompressible Newtonian fluid via FEBio's pressure-velocity fluid solver (cardiovascular CFD) | 3-D | `3d_channel` |
| `fluid_fsi` | Strongly-coupled monolithic FSI (arterial wall hemodynamics, cardiac chamber dynamics, valve modeling) | 3-D | `3d_block` |
| `growth_remodeling` | Multiplicative growth-and-remodeling F = F_e * F_g (vascular adaptation, tissue scaffolds, muscle hypertrophy, tumor mechanobiology) | 3-D | `3d_isotropic` |
| `heat` | Heat conduction (steady-state) | 3-D | `3d_bar` |
| `hyperelasticity` | Nonlinear hyperelasticity (Neo-Hookean, Mooney-Rivlin) | 3-D | `3d_cube` |
| `linear_elasticity` | Linear elasticity (small strain solid mechanics) | 3-D | `3d_cube` |
| `multiphasic` | Biphasic poroelasticity + solute transport (charged-hydrated cartilage, electrolyte diffusion, drug delivery) | 3-D | `3d_diffusion` |
| `plasticity` | Rate-independent plasticity (J2 / Hill / user-curve hardening) — cortical bone, metal implants, surgical tools | 3-D | `3d_uniaxial` |
| `polar_fluid` | Micropolar (Cosserat) fluid with independent micro-rotation DOFs (blood-rheology, polymer suspensions, near-wall turbulence corrections) | 3-D | `3d_channel` |
| `rigid_body` | Rigid-body material (impactors, fixtures, articulating joints, contact prescription) | 3-D | `3d_pushdown` |
| `solute` | Biphasic poroelasticity carrying EXACTLY ONE solute (FEBio module 'solute'); use 'multiphasic' for several solutes or a fixed charge density | 3-D | `3d_diffusion` |
| `viscoelasticity` | Prony-series viscoelastic stress relaxation / creep response (cartilage, ligament, tendon) | 3-D | `3d_stress_relax` |
