# SPARTA

Rarefied gas with the particle method DSMC, where the usual flow equations stop working. Experimental.

## Install

```bash
export SPARTA_BINARY=/path/to/spa_serial
```

Built from source: <https://sparta.github.io/>. You do not need it to start.

Then check that openPASO sees it:

```bash
python check_install.py
```

## What openPASO knows for SPARTA

11 kinds of problem. Ask for any of them in plain words; the names below are what the model uses internally.

| Physics | Description | Dimensions | Templates |
|---|---|---|---|
| `adaptive_grid` | Static (adapt_grid) and dynamic (fix adapt) grid adaptation to resolve shocks, boundary layers and surface geometry | 2-D, 3-D | `circle_2d` |
| `ambipolar_plasma` | Weakly ionized flow in the ambipolar approximation: each electron is carried with its parent ion instead of being moved on the electron timescale | 2-D, 3-D | `circle_2d` |
| `axisymmetric` | 2d axisymmetric DSMC — the x axis is the symmetry axis, cells are revolved annuli, radial weighting available | 2-D | `body_2d` |
| `capability_survey` | Inventory of the commands and styles THIS SPARTA build registers. Three decks (box, explicit surface, implicit surface) construct every style and run, each carrying a check independent of the run itself: exact conservation of particle count and kinetic energy in a closed monatomic box, and the analytic perimeter of the 50-segment circle surface. Each check prints a VERDICT line in the coverage harness's grammar (reference, measured value, tolerance, PASS or FAIL), computed by the deck itself through equal- and format-style variables and an if/then/else print: the particle count and the 2*50*3*sin(pi/50) perimeter as exact identities, the kinetic energy as a conservation law at 1e-12 of its initial value. SURVEY_MUTATE=1, read through a getenv variable, keeps a thermostat running through the box's conservation phase and reads the surface at the wrong scale: the planted failures that show those verdicts can fail. Answers 'what can this build do' by running it rather than by reading the manual, which documents styles that may not be compiled in. | 2-D | `box`, `surface`, `implicit_surface` |
| `chemistry` | Gas-phase chemical reactions (TCE / QK) evaluated during DSMC collisions | 2-D, 3-D | `box_3d` |
| `collision_relaxation` | Particle-particle collisions with the VSS model plus rotational / vibrational energy relaxation | 2-D, 3-D | `box_2d`, `internal_energy_2d` |
| `conjugate_heat_transfer` | DSMC gas <-> solid conjugate heat transfer: SPARTA tallies the surface heat flux and reads back a wall temperature, either through fix surf/temp in-code or through preCICE against an FEM solid | 2-D, 3-D | `circle_2d` |
| `hypersonic_flow` | Hypersonic rarefied flow over a body: bow shock, surface pressure and heat flux, DSMC | 2-D, 3-D | `circle_2d` |
| `particle_emission` | Particle injection / emission from box faces or surfaces — the DSMC inflow boundary condition | 2-D, 3-D | `channel_2d` |
| `rarefied_flow` | Rarefied / free-molecular gas flow (high Knudsen number) solved with DSMC simulator particles | 2-D, 3-D | `box_2d`, `channel_2d` |
| `surface_interaction` | Gas-surface interaction: diffuse / specular / CLL wall models, per-element surface tallies, surface reactions | 2-D, 3-D | `circle_2d` |
