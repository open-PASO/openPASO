"""Generator for Peridynamics (bond-based PD) simulations in 4C.

Encodes general PD knowledge including pre-cracks, rigid impactors
via boundaryphase, SPH infrastructure requirements, and CFL calculation.

Key insight: PD in 4C rides on top of the SPH particle framework.  Even
though the physics is peridynamic (bond-based), the PARTICLE DYNAMIC/SPH
section (SPH kernel, boundary formulation, transport velocity and initial
particle spacing) *must* be specified: without it INITIALPARTICLESPACING
defaults to 0.0 and 4C aborts, before 4C 2026.3.0 with ``negative initial
particle spacing!``, in 4C 2026.3.0 earlier, on its check that
PERIDYNAMIC_GRID_SPACING equals INITIALPARTICLESPACING.
"""

from __future__ import annotations

import math
import textwrap
from typing import Any

from .base import BaseGenerator


class ParticlePDGenerator(BaseGenerator):
    """Generator for bond-based Peridynamics (PD) particle simulations."""

    module_key = "particle_pd"
    display_name = "Peridynamics (Bond-Based PD)"
    problem_type = "Particle"

    # ── Knowledge ─────────────────────────────────────────────────────

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Bond-based peridynamics in 4C models fracture and fragmentation "
                "by replacing the classical PDE-based continuum mechanics formulation "
                "with a non-local integral equation.  Each material point interacts "
                "with neighbors within a finite 'horizon' via pairwise bonds that "
                "can break when a critical stretch is exceeded.  In 4C the PD module "
                "is built on top of the SPH (Smoothed Particle Hydrodynamics) particle "
                "infrastructure, which means SPH kernel and boundary parameters MUST "
                "be specified even though the physics is purely peridynamic.  Failing "
                "to include the PARTICLE DYNAMIC/SPH sub-section makes 4C abort "
                "(INITIALPARTICLESPACING defaults to 0.0): before 4C 2026.3.0 with "
                "'negative initial particle spacing!'; 4C 2026.3.0 stops earlier, "
                "on its check that PERIDYNAMIC_GRID_SPACING equals "
                "INITIALPARTICLESPACING."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "IO",
                "BINNING STRATEGY",
                "PARTICLE DYNAMIC",
                "PARTICLE DYNAMIC/INITIAL AND BOUNDARY CONDITIONS",
                "PARTICLE DYNAMIC/SPH",
                "PARTICLE DYNAMIC/PD",
                "MATERIALS",
                "PARTICLES",
            ],
            "section_details": {
                "PROBLEM TYPE": {
                    "PROBLEMTYPE": '"Particle"',
                },
                "BINNING STRATEGY": {
                    "BIN_SIZE_LOWER_BOUND": (
                        "Must be strictly greater than the PD horizon.  "
                        "Recommended: horizon + 1.0 (in length units)."
                    ),
                    "DOMAINBOUNDINGBOX": (
                        'Format: "xmin ymin zmin xmax ymax zmax".  Must enclose ALL '
                        "particles with a margin of at least one particle spacing on "
                        "each side.  A bounding box that is too tight causes particles "
                        "to fall outside bins and the simulation crashes."
                    ),
                },
                "PARTICLE DYNAMIC": {
                    "DYNAMICTYPE": '"VelocityVerlet" (explicit time integration)',
                    "INTERACTION": (
                        '"SPH" -- MUST be SPH even for PD simulations.  PD is '
                        "activated by setting PD_BODY_INTERACTION: true."
                    ),
                    "PD_BODY_INTERACTION": "true  -- enables peridynamic bond interactions",
                    "RIGID_BODY_MOTION": "false  -- set true only for DEM-style rigid bodies",
                    "PHASE_TO_MATERIAL_ID": (
                        'Maps particle phase names to material IDs.  Format: '
                        '"boundaryphase 1 pdphase 2".  Each phase name used in the '
                        "PARTICLES section must appear here."
                    ),
                    "PHASE_TO_DYNLOADBALFAC": (
                        "Load-balancing factors per phase.  Typically 1.0 for all."
                    ),
                    "GRAVITY_ACCELERATION": '"0.0 0.0 0.0" (default, set if needed)',
                    "TIMESTEP": "Must satisfy CFL: dt < dx / sqrt(E/rho), safety factor 0.5",
                    "NUMSTEP": "Total number of time steps",
                    "MAXTIME": "Maximum simulation time (simulation stops at min(MAXTIME, NUMSTEP*TIMESTEP))",
                    "RESULTSEVERY": "Output frequency in steps",
                    "RESTARTEVERY": "Restart file frequency in steps",
                },
                "PARTICLE DYNAMIC/SPH": {
                    "KERNEL": (
                        "QuinticSpline  -- REQUIRED even for PD.  The SPH kernel is "
                        "used internally for neighbor search infrastructure."
                    ),
                    "KERNEL_SPACE_DIM": (
                        "Kernel2D for 2D problems, Kernel3D for 3D.  Must match the "
                        "physical dimension of the problem."
                    ),
                    "INITIALPARTICLESPACING": "The grid spacing dx between particles.",
                    "BOUNDARYPARTICLEFORMULATION": (
                        "AdamiBoundaryFormulation  -- REQUIRED.  Handles interaction "
                        "between boundaryphase and pdphase particles."
                    ),
                    "TRANSPORTVELOCITYFORMULATION": (
                        "StandardTransportVelocity  -- REQUIRED for PD simulations."
                    ),
                },
                "PARTICLE DYNAMIC/PD": {
                    "INTERACTION_HORIZON": (
                        "The PD horizon delta.  Recommended: m * dx where m = 3 "
                        "(horizon ratio).  All bonds within this radius interact."
                    ),
                    "PERIDYNAMIC_GRID_SPACING": (
                        "Must equal the particle spacing dx.  Used to compute the "
                        "PD influence function."
                    ),
                    "PD_DIMENSION": (
                        "Peridynamic_2DPlaneStrain for 2D plane strain, "
                        "Peridynamic_3D for full 3D.  Affects bond-force calculation."
                    ),
                    "NORMALCONTACTLAW": (
                        "NormalLinearSpring  -- penalty contact between PD bodies "
                        "and boundary particles."
                    ),
                    "NORMAL_STIFF": (
                        "Normal contact stiffness for inter-body contact.  Typical "
                        "range: 1e3 to 1e5.  Too high causes instability; too low "
                        "causes excessive penetration."
                    ),
                    "PRE_CRACKS": (
                        "Peridynamics BRANCH builds only: upstream 4C's PARTICLE "
                        "DYNAMIC/PD has no PRE_CRACKS key and rejects it with "
                        "'Could not match this input'.  From 4C v2026.2.0 (commit "
                        "163b10526) upstream pre-cracks are PRE_CRACK_LINES (list of "
                        "START/END 3-vectors) or PRE_CRACK_PLANES (list of P0/P1/P2 "
                        "3-vectors).  On a branch build: "
                        'line segments defining pre-existing cracks.  Format: '
                        '"x1 y1 x2 y2 ; x3 y3 x4 y4".  Bonds crossing these line '
                        "segments are broken at initialization (visibility condition).  "
                        "Multiple crack segments separated by semicolons.  This is the "
                        "mechanism for modeling notches and initial damage without "
                        "removing particles.  This is the pre-merge branch form; "
                        "upstream 4C 2026.2.0 and later read PRE_CRACK_LINES instead."
                    ),
                    "PRE_CRACK_LINES": (
                        "4C 2026.2.0 and later: a list of segments, each '- START: [x1, y1, "
                        "z1]' and 'END: [x2, y2, z2]'; bonds crossing a segment "
                        "are broken at initialization.  PRE_CRACK_PLANES is the "
                        "same with parallelogram patches given by corner points "
                        "P0, P1, P2; both lists take 3-D points and both are "
                        "accepted in 2-D peridynamic decks."
                    ),
                },
                "PARTICLE DYNAMIC/INITIAL AND BOUNDARY CONDITIONS": {
                    "DIRICHLET_BOUNDARY_CONDITION": (
                        '"boundaryphase 1"  -- applies FUNCT1 as prescribed '
                        "displacement to all particles of the named phase.  "
                        "The integer is the function ID."
                    ),
                    "CONSTRAINT": (
                        '"Projection2D" for 2D problems.  Constrains out-of-plane '
                        "motion."
                    ),
                },
            },
            "materials": {
                "MAT_ParticlePD": {
                    "description": (
                        "Bond-based peridynamic material for deformable PD bodies.  "
                        "Used with pdphase particles.  Bonds break when stretch "
                        "exceeds CRITICAL_STRETCH."
                    ),
                    "parameters": {
                        "INITRADIUS": {
                            "description": "Initial particle radius = dx/2",
                            "range": "Problem-dependent (half of particle spacing)",
                        },
                        "INITDENSITY": {
                            "description": "Mass density in consistent units",
                            "range": "e.g. 8e-3 g/mm^3 for steel in mm/ms/g system",
                        },
                        "YOUNG": {
                            "description": "Young's modulus",
                            "range": "e.g. 190e3 MPa for steel in mm/ms/g system",
                        },
                        "CRITICAL_STRETCH": {
                            "description": (
                                "Bond breaking criterion.  A bond breaks irreversibly "
                                "when its stretch s = (|xi+eta| - |xi|)/|xi| exceeds "
                                "this value.  Related to fracture energy: "
                                "s_c = sqrt(5 * G_c / (9 * E * delta)) for 2D plane "
                                "stress.  Typical range: 0.001 to 0.05."
                            ),
                            "range": "0.001 -- 0.05 (material-dependent)",
                        },
                    },
                },
                "MAT_ParticleSPHBoundary": {
                    "description": (
                        "Boundary particle material for rigid walls and impactors.  "
                        "Used with boundaryphase particles.  These particles do NOT "
                        "deform -- their motion is prescribed via Dirichlet BCs."
                    ),
                    "parameters": {
                        "INITRADIUS": {
                            "description": "Initial particle radius = dx/2",
                            "range": "Same as PD material spacing",
                        },
                        "INITDENSITY": {
                            "description": (
                                "Density for boundary particles.  Can be set to 1 "
                                "(arbitrary) since boundary particles are rigid."
                            ),
                            "range": "Typically 1 (does not affect PD physics)",
                        },
                    },
                },
            },
            "solver": {
                "type": "Explicit (VelocityVerlet)",
                "notes": (
                    "PD uses explicit time integration exclusively.  There is no "
                    "implicit solver option.  The time step is governed by the CFL "
                    "condition."
                ),
            },
            "time_integration": {
                "scheme": "Velocity Verlet (symplectic, second-order)",
                "CFL_condition": (
                    "dt < dx / c_wave where c_wave = sqrt(E / rho).  "
                    "A safety factor of 0.5 is recommended: dt = 0.5 * dx / sqrt(E/rho)."
                ),
                "example_steel": (
                    "Compute wave speed c = sqrt(E/rho) for your material. "
                    "Then dt < 0.5 * dx / c. Choose dt with a safety factor."
                ),
            },
            "particle_types": {
                "pdphase": (
                    "Peridynamic body particles.  These are deformable and can form "
                    "and break bonds.  Format: "
                    '"TYPE pdphase POS x y z PDBODYID 0".  '
                    "The PDBODYID groups particles into distinct PD bodies (use 0 "
                    "for a single body, increment for multiple bodies)."
                ),
                "boundaryphase": (
                    "Rigid boundary particles.  Their motion is prescribed via "
                    "Dirichlet BCs (FUNCT).  Used for rigid impactors, walls, and "
                    "loading platens.  Format: "
                    '"TYPE boundaryphase POS x y z".  '
                    "NOTE: Do NOT use PDBODYID for boundaryphase particles."
                ),
                "WARNING_rigidphase": (
                    "NEVER use rigidphase (DEM rigid bodies) with PD simulations.  "
                    "rigidphase is for DEM granular mechanics and is incompatible "
                    "with peridynamics.  For rigid impactors, ALWAYS use "
                    "boundaryphase + DIRICHLET_BOUNDARY_CONDITION."
                ),
            },
            "rigid_impactor_recipe": {
                "step_1": "Create boundaryphase particles filling the impactor geometry.",
                "step_2": (
                    "Set PHASE_TO_MATERIAL_ID to map boundaryphase to a "
                    "MAT_ParticleSPHBoundary material."
                ),
                "step_3": (
                    'Set DIRICHLET_BOUNDARY_CONDITION: "boundaryphase 1" to apply '
                    "FUNCT1 as prescribed displacement."
                ),
                "step_4": (
                    "Define FUNCT1 with SYMBOLIC_FUNCTION_OF_SPACE_TIME giving "
                    "displacement = velocity * t for constant-velocity impact."
                ),
            },
            "pre_crack_mechanism": {
                "description": (
                    "Pre-cracks are implemented via the visibility condition: at "
                    "initialization, any bond whose reference line segment (connecting "
                    "two particles) crosses a pre-crack line segment is broken.  This "
                    "models notches, saw-cuts, and initial damage without removing "
                    "particles from the discretization."
                ),
                "format": (
                    'PRE_CRACKS (peridynamics branch builds only, not upstream 4C): '
                    '"x1 y1 x2 y2 ; x3 y3 x4 y4"  -- each segment is defined by '
                    "its two endpoints (2D coordinates).  Multiple segments separated "
                    "by semicolons.  Upstream 4C from v2026.2.0 instead takes "
                    "PRE_CRACK_LINES, a list of entries with START and END "
                    "3-vectors (or PRE_CRACK_PLANES with P0/P1/P2)."
                ),
                "example": (
                    '"x1 y1 x2 y2 ; x3 y3 x4 y4"  -- multiple line segments separated by semicolons'
                ),
            },
            "unit_systems": {
                "mm_ms_g (recommended)": {
                    "Length": "mm",
                    "Time": "ms",
                    "Mass": "g",
                    "Force": "N (= g*mm/ms^2)",
                    "Stress": "MPa (= N/mm^2)",
                    "Density": "g/mm^3 (= 1e-3 * kg/m^3 value)",
                    "Velocity": "mm/ms (= m/s)",
                    "Energy": "mJ (= N*mm)",
                },
                "SI (m_s_kg)": {
                    "Length": "m",
                    "Time": "s",
                    "Mass": "kg",
                    "Force": "N",
                    "Stress": "Pa",
                    "Density": "kg/m^3",
                    "Velocity": "m/s",
                    "Energy": "J",
                },
            },
            "pitfalls": [
                (
                    "[Input] CRITICAL: The PARTICLE DYNAMIC/SPH section "
                    "is mandatory for PD simulations even though the "
                    "physics is peridynamic, not SPH.  It carries "
                    "INITIALPARTICLESPACING, which sets every "
                    "particle's mass, as well as KERNEL, "
                    "KERNEL_SPACE_DIM, BOUNDARYPARTICLEFORMULATION and "
                    "TRANSPORTVELOCITYFORMULATION. Signal: without the "
                    "section a 4C before 2026.3.0 aborts in "
                    "ParticleInteractionSPH::set_initial_states with "
                    "`negative initial particle spacing!` "
                    "(particle/src/interaction/"
                    "4C_particle_interaction_sph.cpp); 4C 2026.3.0 stops "
                    "earlier, at its check that PERIDYNAMIC_GRID_SPACING "
                    "equals INITIALPARTICLESPACING (measured on the three "
                    "builds 2026-10-01) — nothing about "
                    "neighbour pairs or boundary formulations is "
                    "mentioned.  Do NOT treat a zero "
                    "`Number of pd_neighbor_pairs in peridynamic "
                    "evaluation on this proc` line as the symptom: that "
                    "counts inter-body CONTACT pairs and reads zero on "
                    "essentially every step of a healthy run. "
                    "(Audit 2026-06-02; corrected by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Numerical] DOMAINBOUNDINGBOX must enclose ALL "
                    "particles with margin, at every time step and not "
                    "only at setup.  The simulation does NOT crash when "
                    "it does not: 4C silently DELETES the offending "
                    "particles and carries on to a wrong answer. "
                    "Signal: the one-line note `on processor 0 removed "
                    "N particle(s) being outside the computational "
                    "domain!`, which names no id, no position, no step "
                    "and no phase, appearing at setup and/or mid-run; "
                    "the run then completes and only the RESULT "
                    "DESCRIPTION block catches it, as a count mismatch "
                    "(the literal clause is `tests but performed`, with "
                    "both counts interpolated around it: expected N "
                    "tests but performed 0) because the "
                    "tested ids no longer exist.  Grep for `removed` "
                    "and `outside the computational domain`, not for "
                    "an abort. (Audit 2026-06-02; corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Input] Use boundaryphase (NOT rigidphase) for "
                    "rigid impactors.  boundaryphase particles interact "
                    "with pdphase; rigidphase does not, and the failure "
                    "is SILENT — there is no parser error and no "
                    "runtime abort.  rigidphase is a valid particle "
                    "type, the deck parses, the run completes and 4C "
                    "says nothing, but the rigid body applies ZERO "
                    "force to the peridynamic body. Signal: the pd body "
                    "moves as if the obstacle were not there — under "
                    "gravity its velocity is exactly g*t — while any "
                    "damage output stays clean, so a check that only "
                    "looks for cracks sees a healthy run.  Compare a "
                    "velocity against free fall, or result-test a "
                    "position. (Audit 2026-06-02; corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Numerical] Horizon ratio m = "
                    "INTERACTION_HORIZON / "
                    "PERIDYNAMIC_GRID_SPACING (both in "
                    "PARTICLE DYNAMIC/PD) should be AT "
                    "LEAST 3; m = 2 gives poor accuracy, "
                    "m >= 4 is more expensive.  It is a "
                    "discretisation parameter, not a "
                    "tolerance: it sets the size of the "
                    "bond neighbourhood, which grows like "
                    "m^2 in 2-D and m^3 in 3-D. Signal: with "
                    "MAT_ParticlePD active, 4C "
                    "prints the neighbourhood size once "
                    "per run as `Number of initialized "
                    "peridynamic bonds on this proc: N` — "
                    "read m off that, and expect halving m "
                    "to roughly halve N in 2-D.  Do not "
                    "expect the answer to plateau at "
                    "m = 3-4: on 4C's own PD deck it keeps "
                    "moving monotonically as m grows, "
                    "because delta-convergence needs the "
                    "grid to be refined at the same time. "
                    "(Audit 2026-06-02; corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Numerical] BIN_SIZE_LOWER_BOUND must be >= "
                    "INTERACTION_HORIZON.  Nothing degrades silently: 4C "
                    "checks this in the SPHPeridynamic constructor, "
                    "before the first time step and before any bond is "
                    "counted. Signal: hard abort `Peridynamic "
                    "INTERACTION_HORIZON must be smaller than "
                    "BIN_SIZE_LOWER_BOUND!` from "
                    "particle/src/interaction/"
                    "4C_particle_interaction_sph_peridynamic.cpp.  The "
                    "wording says 'smaller than' but the guard is "
                    "horizon <= bin size, so equality is accepted and "
                    "gives the same bond count as a larger bin.  There "
                    "is no neighbor_count diagnostic in 4C to inspect. "
                    "(Audit 2026-06-02; corrected by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Input] Pre-cracks (PRE_CRACKS, the pre-merge PD "
                    "branch's key; 4C 2026.2.0 and later read PRE_CRACK_LINES) must "
                    "use 2D coordinates (x, y) matching the "
                    "particle positions. The visibility "
                    "check is GEOMETRIC — tests whether "
                    "the line segment connecting two "
                    "particles crosses the crack segment. "
                    "NOTE: PRE_CRACKS is the form of "
                    "pre-merge branch work on bond-based "
                    "peridynamics; upstream 4C 2026.2.0 and "
                    "later read PRE_CRACK_LINES / "
                    "PRE_CRACK_PLANES lists of START and "
                    "END points instead, so check your "
                    "build before relying on it. Signal: a crack "
                    "in the wrong units (mm vs m, or a "
                    "domain offset that misses the "
                    "particle grid) is accepted in "
                    "silence — the segment still parses, "
                    "4C still prints `Number of pre-crack "
                    "segments: N` and emits no warning — "
                    "while (almost) no bond is broken and "
                    "the damage field stays zero at t = 0. "
                    "The cheap check is to difference "
                    "`Number of initialized peridynamic "
                    "bonds on this proc` against a run "
                    "with PRE_CRACKS removed; the segment "
                    "count line alone proves nothing. "
                    "(Audit 2026-06-02; corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Numerical] CFL violation: if dt >= dx / "
                    "sqrt(E/rho), the explicit time integration becomes "
                    "unstable.  Use a safety factor of 0.5. Signal: you "
                    "never see the NaN — 4C aborts first, in its own "
                    "bookkeeping, with `a particle of phase 'pdphase' "
                    "traveled more than one bin on this processor!` "
                    "(particle/src/algorithm/4C_particle_algorithm.cpp). "
                    "That message names the phase but no particle, no "
                    "velocity and no step, and never mentions the time "
                    "step, CFL or stability, so grepping for 'nan' or "
                    "'velocity' finds nothing.  There is no `non-finite "
                    "velocity at particle X` string in 4C. (Audit "
                    "2026-06-02; corrected by execution 2026-08-06.)"
                ),
                (
                    "[Input] MAT_ParticlePD's INITRADIUS does NOT drive "
                    "the mass or the volume of a PD run, and chasing a "
                    "mass error through it changes nothing: on 4C's own "
                    "PD deck, scaling INITRADIUS on every material "
                    "leaves the result bit-identical and the bond count "
                    "unchanged.  Particle mass is INITDENSITY * "
                    "INITIALPARTICLESPACING^KERNEL_SPACE_DIM, with "
                    "INITIALPARTICLESPACING living in PARTICLE "
                    "DYNAMIC/SPH; bond forces use "
                    "PERIDYNAMIC_GRID_SPACING and the horizon, and PD "
                    "contact gaps are measured against "
                    "PERIDYNAMIC_GRID_SPACING rather than radii.  Keep "
                    "INITRADIUS = dx/2 for tidiness, but fix the mass "
                    "at INITIALPARTICLESPACING. Signal: 4C prints no "
                    "total system mass at startup, so there is nothing "
                    "to compare; change INITIALPARTICLESPACING and "
                    "watch a result test move. (Audit 2026-06-02; "
                    "FALSIFIED and rewritten by execution 2026-08-06.)"
                ),
                (
                    "[Numerical] Bond-based PD restricts "
                    "Poisson's ratio to nu = 0.25 (2D) or "
                    "nu = 1/3 (3D) — a FUNDAMENTAL "
                    "limitation of the pairwise force model. "
                    "4C enforces it by not offering the "
                    "knob: MAT_ParticlePD takes exactly "
                    "INITRADIUS, INITDENSITY, YOUNG and "
                    "CRITICAL_STRETCH, and there is no "
                    "Poisson ratio to override. Signal: "
                    "adding NUE, POISSONRATIO or NU to a "
                    "MAT_ParticlePD entry is a hard parse "
                    "error, `Could not match this input` "
                    "from global_data/4C_global_data_read."
                    "cpp with the MATERIALS block echoed "
                    "back, so the deck never runs and there "
                    "is no contraction to inspect.  Use "
                    "state-based PD if you need arbitrary "
                    "Poisson ratios. (Audit 2026-06-02; "
                    "corrected by execution 2026-08-06.)"
                ),
                (
                    "[Input] For 2D problems, particles must "
                    "STILL have z = 0.0 coordinates and "
                    "DOMAINBOUNDINGBOX must have a small "
                    "non-zero z-extent (e.g. -0.01 to "
                    "0.01).  Use the thin-slab convention. "
                    "Signal: POS is a fixed-size 3-vector, "
                    "so dropping z makes the parser take "
                    "the NEXT token as the missing "
                    "component and fail on that instead — "
                    "e.g. `Could not parse 'PDBODYID' as a "
                    "double value.` from "
                    "core/io/src/4C_io_value_parser.cpp, "
                    "which never mentions coordinates.  A "
                    "zero z-extent is worse: 4C divides by "
                    "the degenerate domain length inside "
                    "ParticleEngine::init_binning_strategy "
                    "and dies on a floating-point "
                    "divide-by-zero signal with no PROC 0 "
                    "ERROR block, no message and a signal "
                    "exit status rather than 1. (Audit "
                    "2026-06-02; corrected by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Input] LOADING: boundaryphase "
                    "particles interact with pdphase via "
                    "REPULSIVE contact ONLY.  They CANNOT "
                    "apply tensile loads — only compressive "
                    "impact.  Two independent mechanisms "
                    "enforce it: a contact pair is created "
                    "only where the separation drops below "
                    "PERIDYNAMIC_GRID_SPACING, and the "
                    "force is then clamped with "
                    "std::min(0.0, ...). Signal: retract a "
                    "boundaryphase wall from a pd body "
                    "under a Dirichlet drive and the body "
                    "does not move by one bit — position "
                    "and velocity stay exactly at their "
                    "initial values and the `Number of "
                    "pd_neighbor_pairs in peridynamic "
                    "evaluation on this proc` line never "
                    "leaves zero, because no pair is ever "
                    "formed.  Use INITIAL_VELOCITY_FIELD or "
                    "a per-particle Dirichlet flag for "
                    "tension/opening. (Audit 2026-06-02; "
                    "confirmed by execution 2026-08-06.)"
                ),
                (
                    "[Input] PDFIXED: per-particle flag "
                    "fixing a particle in place (zero "
                    "displacement). Add 'PDFIXED 1' to the "
                    "particle definition string. Use for "
                    "clamped supports in fracture problems. "
                    "NOTE: PDFIXED is the form of pre-merge "
                    "branch work on bond-based peridynamics; "
                    "upstream 4C 2026.2.0 and later hold a particle with "
                    "DIRICHLET_FUNCT <n> on its line plus "
                    "DIRICHLET_BOUNDARY_CONDITION_FLAGGED: "
                    "[pdphase], FUNCT<n> being the "
                    "displacement (all zero pins it), so "
                    "check your build before relying on it. Signal: "
                    "omitting it on a clamped edge lets "
                    "those particles move freely and 4C "
                    "warns about nothing — the bond count "
                    "and any pre-crack damage are "
                    "unchanged, so a check that watches "
                    "cracks sees a healthy run while the "
                    "specimen drifts off as a body.  "
                    "Result-test a clamped particle's "
                    "position against its reference "
                    "coordinates. (Audit 2026-06-02; "
                    "confirmed by execution 2026-08-06.)"
                ),
                (
                    "[Output] IO/RUNTIME VTK OUTPUT sections are NOT "
                    "incompatible with particle problems and do not "
                    "crash 4C.  Adding IO/RUNTIME VTK OUTPUT and "
                    "IO/RUNTIME VTK OUTPUT/STRUCTURE to a PD deck runs "
                    "to completion, reproduces the untouched run "
                    "bit-for-bit on its result tests, and writes the "
                    "extra runtime VTK files on top of the particle VTU "
                    "files 4C writes anyway; several upstream "
                    "Polymer_Network decks ship these sections. Signal: "
                    "none — there is no `IO/RUNTIME VTK OUTPUT not "
                    "supported for particle problems` and no "
                    "`RuntimeVTKOutputParams: invalid for PARTICLE` "
                    "string in 4C.  Deleting these sections to 'fix' a "
                    "particle deck removes wanted output and fixes "
                    "nothing. (Audit 2026-06-02; FALSIFIED and "
                    "rewritten by execution 2026-08-06.)"
                ),
                (
                    "[Numerical] Critical stretch formula "
                    "DIFFERS between plane stress and plane "
                    "strain. Plane strain (2D): s_c = "
                    "sqrt(5*G_Ic / (9 * K_b * delta)) where "
                    "K_b = E / (3*(1-2*nu)). Plane stress: "
                    "s_c = sqrt(5*G_Ic / (6*E*delta)). "
                    "Match the s_c formula to the "
                    "PD_DIMENSION enum in PARTICLE "
                    "DYNAMIC/PD, because the bond "
                    "micromodulus 4C uses is normalised "
                    "differently for Peridynamic_2DPlane"
                    "Strain and Peridynamic_2DPlaneStress. "
                    "Signal: none, and that is the point — "
                    "flipping only that enum, with the same "
                    "CRITICAL_STRETCH and the same horizon, "
                    "is a different constitutive problem "
                    "but initialises the identical number "
                    "of bonds and produces no warning, so "
                    "no geometric or count diagnostic "
                    "betrays it and the only evidence is "
                    "the answer.  4C also asserts "
                    "consistency with CONSTRAINT: a 2-D "
                    "PD_DIMENSION requires Projection2D and "
                    "vice versa. (Audit 2026-06-02; "
                    "corrected by execution 2026-08-06.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "Plate under uniaxial tension",
                    "description": (
                        "A rectangular plate with a central notch pulled in tension.  "
                        "Crack propagates from the notch tip perpendicular to the "
                        "loading direction.  Good first test for PD fracture."
                    ),
                    "template_variant": "plate_2d",
                },
                {
                    "name": "Plate impact (1D wave propagation)",
                    "description": (
                        "Simplest PD test: a 1D bar or thin plate impacted at one "
                        "end.  Validates wave speed and bond force computation."
                    ),
                },
            ],
        }

    # ── Templates ─────────────────────────────────────────────────────

    def get_template(self, variant: str = "plate_2d") -> str:
        if variant == "plate_2d":
            return self._template_plate_2d()
        raise ValueError(
            f"Unknown variant {variant!r} for {self.module_key}.  "
            f"Available: {[v['name'] for v in self.list_variants()]}"
        )

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "plate_2d",
                "description": (
                    "2D plate under prescribed "
                    "velocity impact from the left.  Demonstrates all essential PD "
                    "features: pdphase body, boundaryphase impactor, "
                    "CFL-safe time stepping; the pre-crack syntax is shown as a "
                    "comment.  Uses mm/ms/g unit system."
                ),
            },
        ]

    # ── Validation ────────────────────────────────────────────────────

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        """Physics-aware validation of PD parameters.

        Expected keys in *params* (all optional):
            dx              - particle spacing
            horizon         - PD interaction horizon
            young           - Young's modulus
            density         - material density
            dt              - time step
            critical_stretch - bond breaking criterion
            bin_size        - BIN_SIZE_LOWER_BOUND
            domain_bbox     - [xmin, ymin, zmin, xmax, ymax, zmax]
            particles_bbox  - [xmin, ymin, zmin, xmax, ymax, zmax] (actual particle extent)
        """
        warnings: list[str] = []

        dx = params.get("dx")
        horizon = params.get("horizon")
        young = params.get("young")
        density = params.get("density")
        dt = params.get("dt")
        critical_stretch = params.get("critical_stretch")
        bin_size = params.get("bin_size")
        domain_bbox = params.get("domain_bbox")
        particles_bbox = params.get("particles_bbox")

        # Horizon vs dx check
        if horizon is not None and dx is not None:
            m = horizon / dx
            if m < 2.0:
                warnings.append(
                    f"ERROR: Horizon ratio m = delta/dx = {m:.2f} < 2.  "
                    f"PD requires m >= 2 (recommended m = 3)."
                )
            elif m < 2.5:
                warnings.append(
                    f"WARNING: Horizon ratio m = {m:.2f} is low.  "
                    f"m = 3 is recommended for accuracy."
                )

        # CFL condition
        if young is not None and density is not None and dt is not None and dx is not None:
            if density <= 0:
                warnings.append("ERROR: Density must be positive.")
            elif young <= 0:
                warnings.append("ERROR: Young's modulus must be positive.")
            else:
                c_wave = math.sqrt(young / density)
                dt_cfl = dx / c_wave
                ratio = dt / dt_cfl
                if ratio >= 1.0:
                    warnings.append(
                        f"ERROR: CFL VIOLATION.  dt={dt} >= dt_CFL={dt_cfl:.6e} "
                        f"(ratio={ratio:.3f}).  The simulation WILL be unstable.  "
                        f"Reduce dt to at most {0.5 * dt_cfl:.6e} (safety factor 0.5)."
                    )
                elif ratio > 0.8:
                    warnings.append(
                        f"WARNING: CFL ratio = {ratio:.3f} is dangerously high.  "
                        f"Recommended: dt <= {0.5 * dt_cfl:.6e} (safety factor 0.5)."
                    )

        # Critical stretch sanity
        if critical_stretch is not None:
            if critical_stretch <= 0:
                warnings.append("ERROR: CRITICAL_STRETCH must be positive.")
            elif critical_stretch > 0.1:
                warnings.append(
                    f"WARNING: CRITICAL_STRETCH = {critical_stretch} is unusually "
                    f"large.  Typical values: 0.001 to 0.05."
                )
            elif critical_stretch < 1e-4:
                warnings.append(
                    f"WARNING: CRITICAL_STRETCH = {critical_stretch} is very small.  "
                    f"Bonds will break almost immediately."
                )

        # Bin size vs horizon
        if bin_size is not None and horizon is not None:
            if bin_size <= horizon:
                warnings.append(
                    f"ERROR: BIN_SIZE_LOWER_BOUND ({bin_size}) must be > horizon "
                    f"({horizon}).  Neighbors outside the bin will be missed."
                )

        # Bounding box encloses particles
        if domain_bbox is not None and particles_bbox is not None:
            if len(domain_bbox) == 6 and len(particles_bbox) == 6:
                labels = ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]
                for i in range(3):  # min dimensions
                    if domain_bbox[i] > particles_bbox[i]:
                        warnings.append(
                            f"ERROR: DOMAINBOUNDINGBOX {labels[i]}={domain_bbox[i]} "
                            f"is greater than particle {labels[i]}={particles_bbox[i]}.  "
                            f"Particles will fall outside the domain."
                        )
                for i in range(3, 6):  # max dimensions
                    if domain_bbox[i] < particles_bbox[i]:
                        warnings.append(
                            f"ERROR: DOMAINBOUNDINGBOX {labels[i]}={domain_bbox[i]} "
                            f"is less than particle {labels[i]}={particles_bbox[i]}.  "
                            f"Particles will fall outside the domain."
                        )

        return warnings

    # ── Private template builders ─────────────────────────────────────

    @staticmethod
    def _template_plate_2d() -> str:
        """Template showing the FORMAT of a 2D PD input. All values are placeholders."""
        return textwrap.dedent("""\
            # 2D Peridynamics: FORMAT TEMPLATE
            # The pre-crack key depends on the build: the pre-merge peridynamics
            # extension reads a PRE_CRACKS string, 4C 2026.2.0 and later read
            # PRE_CRACK_LINES (shown as a comment below). The decks openPASO runs
            # are served in the grammar of the 4C it finds.
            # ALL numerical values below are PLACEHOLDERS — they must be determined
            # by the user based on the specific problem geometry, material, and
            # required resolution. Consult the literature and 4C test files
            # via examples(keyword, solver='fourc', action='search') for
            # appropriate values.
            #
            # Units: choose a consistent system (e.g., mm-ms-g or SI)

            PROBLEM TYPE:
              PROBLEMTYPE: "Particle"

            IO:
              STDOUTEVERY: <OUTPUT_FREQUENCY>
              VERBOSITY: "Standard"

            # NO IO/RUNTIME VTK OUTPUT/PARTICLES section — it does not
            # exist and adding it is a hard parse error, "Section
            # 'IO/RUNTIME VTK OUTPUT/PARTICLES' is not a valid section
            # name." Particle ParaView output is written unconditionally
            # at the RESULTSEVERY interval below.

            BINNING STRATEGY:
              BIN_SIZE_LOWER_BOUND: <HORIZON + margin>
              DOMAINBOUNDINGBOX: "<xmin> <ymin> <zmin> <xmax> <ymax> <zmax>"

            PARTICLE DYNAMIC:
              DYNAMICTYPE: "VelocityVerlet"
              INTERACTION: "SPH"
              RESULTSEVERY: <OUTPUT_FREQUENCY>
              RESTARTEVERY: <RESTART_FREQUENCY>
              TIMESTEP: <dt>  # unquoted number; CFL: dt < 0.5 * dx / sqrt(E/rho)
              NUMSTEP: <total steps>
              MAXTIME: <end time>
              GRAVITY_ACCELERATION: "0.0 0.0 0.0"
              PHASE_TO_DYNLOADBALFAC: "boundaryphase 1.0 pdphase 1.0"
              PHASE_TO_MATERIAL_ID: "boundaryphase 1 pdphase 2"
              RIGID_BODY_MOTION: false
              PD_BODY_INTERACTION: true

            PARTICLE DYNAMIC/INITIAL AND BOUNDARY CONDITIONS:
              DIRICHLET_BOUNDARY_CONDITION: "boundaryphase 1"
              CONSTRAINT: "Projection2D"

            FUNCT1:
              - COMPONENT: 0
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<velocity>*t"
              - COMPONENT: 1
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
              - COMPONENT: 2
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"

            # CRITICAL: SPH section is REQUIRED even for pure PD simulations
            PARTICLE DYNAMIC/SPH:
              KERNEL: QuinticSpline
              KERNEL_SPACE_DIM: Kernel2D
              INITIALPARTICLESPACING: <dx — choose based on problem and required resolution>
              BOUNDARYPARTICLEFORMULATION: AdamiBoundaryFormulation
              TRANSPORTVELOCITYFORMULATION: StandardTransportVelocity

            PARTICLE DYNAMIC/PD:
              INTERACTION_HORIZON: <m * dx, typically m=3>
              PERIDYNAMIC_GRID_SPACING: <dx — must match INITIALPARTICLESPACING>
              PD_DIMENSION: Peridynamic_2DPlaneStrain
              NORMALCONTACTLAW: NormalLinearSpring
              NORMAL_STIFF: <contact stiffness>
              # Pre-cracks: upstream 4C has no PRE_CRACKS key (peridynamics branch
              # builds only; upstream rejects it with "Could not match this input").
              # From 4C v2026.2.0 upstream takes a list instead:
              # PRE_CRACK_LINES:
              #   - START: [<x1>, <y1>, 0.0]
              #     END: [<x2>, <y2>, 0.0]
              # (or PRE_CRACK_PLANES with corner points P0, P1, P2).

            MATERIALS:
              - MAT: 1
                MAT_ParticleSPHBoundary:
                  INITRADIUS: <dx/2>
                  INITDENSITY: <density>
              - MAT: 2
                MAT_ParticlePD:
                  INITRADIUS: <dx/2>
                  INITDENSITY: <density>
                  YOUNG: <Young's modulus>
                  CRITICAL_STRETCH: <critical stretch for bond breaking>

            # Generate particle positions programmatically:
            # Regular grid with uniform spacing dx covering the domain
            PARTICLES:
              - "TYPE pdphase POS <x> <y> 0.0 PDBODYID 0"
              - "TYPE boundaryphase POS <x> <y> 0.0"
        """)
