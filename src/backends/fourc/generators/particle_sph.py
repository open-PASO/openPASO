"""Generator for Smoothed Particle Hydrodynamics (SPH) simulations in 4C.

SPH is a meshfree Lagrangian method primarily used for fluid dynamics
(dam breaks, sloshing, free-surface flows) but also applicable to
solid-mechanics problems.  In 4C it shares the particle infrastructure
with peridynamics (PD) but uses fundamentally different physics:
kernel-weighted summation/integration for field quantities rather than
pairwise bond forces.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class ParticleSPHGenerator(BaseGenerator):
    """Generator for SPH particle simulations."""

    module_key = "particle_sph"
    display_name = "Smoothed Particle Hydrodynamics (SPH)"
    problem_type = "Particle"

    # ── Knowledge ─────────────────────────────────────────────────────

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Smoothed Particle Hydrodynamics (SPH) in 4C is a meshfree "
                "Lagrangian particle method for simulating fluid dynamics and, "
                "with appropriate constitutive models, solid mechanics.  The "
                "method approximates continuous fields by kernel-weighted "
                "summation over neighboring particles.  4C supports weakly "
                "compressible SPH with various kernels, density evaluation "
                "strategies, and boundary formulations.  Common applications "
                "include dam break, Poiseuille flow, sloshing, and free-surface "
                "flows."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "IO",
                "BINNING STRATEGY",
                "PARTICLE DYNAMIC",
                "PARTICLE DYNAMIC/SPH",
                "MATERIALS",
                "PARTICLES",
            ],
            "section_details": {
                "PROBLEM TYPE": {
                    "PROBLEMTYPE": '"Particle"',
                },
                "BINNING STRATEGY": {
                    "BIN_SIZE_LOWER_BOUND": (
                        "Must be >= the kernel support radius (typically 3*dx for "
                        "QuinticSpline).  Controls the spatial hashing bin size."
                    ),
                    "DOMAINBOUNDINGBOX": (
                        '"xmin ymin zmin xmax ymax zmax" -- must enclose ALL particles '
                        "with margin.  Include space for fluid expansion."
                    ),
                },
                "PARTICLE DYNAMIC": {
                    "DYNAMICTYPE": (
                        '"VelocityVerlet" (default explicit integrator for SPH)'
                    ),
                    "INTERACTION": '"SPH"',
                    "PHASE_TO_MATERIAL_ID": (
                        'Maps phase names to material IDs.  Example: '
                        '"phase1 1 boundaryphase 2"'
                    ),
                    "PHASE_TO_DYNLOADBALFAC": "Load-balancing factors per phase.",
                    "TIMESTEP": "Time step size (must satisfy CFL for SPH)",
                    "NUMSTEP": "Total number of time steps",
                    "MAXTIME": "Maximum simulation time",
                    "RESULTSEVERY": "Output frequency in steps",
                    "RESTARTEVERY": "Restart checkpoint frequency",
                    "GRAVITY_ACCELERATION": (
                        '"gx gy gz" -- gravitational acceleration vector.  '
                        "Essential for dam break and hydrostatic problems."
                    ),
                },
                "PARTICLE DYNAMIC/SPH": {
                    "KERNEL": (
                        "QuinticSpline (recommended, C^2 smooth, compact support) or "
                        "CubicSpline (simpler, C^1 smooth).  QuinticSpline gives "
                        "better accuracy for most problems."
                    ),
                    "KERNEL_SPACE_DIM": (
                        "Kernel1D, Kernel2D, or Kernel3D.  MUST match the physical "
                        "dimension of the problem.  A mismatch causes incorrect "
                        "kernel normalization and wrong results."
                    ),
                    "INITIALPARTICLESPACING": (
                        "The initial uniform spacing dx between particles."
                    ),
                    "DENSITYEVALUATION": (
                        "DensitySummation (direct kernel sum, conserves mass) or "
                        "DensityIntegration (continuity equation, smoother).  "
                        "DensityIntegration is more common for dynamic problems."
                    ),
                    "BOUNDARYPARTICLEFORMULATION": (
                        "AdamiBoundaryFormulation (recommended).  Handles wall "
                        "boundary conditions via mirrored pressure/velocity."
                    ),
                    "TRANSPORTVELOCITYFORMULATION": (
                        "StandardTransportVelocity -- used with some formulations "
                        "to reduce tensile instability.  Optional for pure SPH."
                    ),
                },
                "PARTICLE DYNAMIC/INITIAL AND BOUNDARY CONDITIONS": {
                    "INITIAL_VELOCITY_FIELD": (
                        '"phase1 1" -- applies FUNCT1 as initial velocity to '
                        "all particles of the named phase."
                    ),
                    "DIRICHLET_BOUNDARY_CONDITION": (
                        '"boundaryphase 1" -- prescribed displacement for boundary '
                        "particles."
                    ),
                },
            },
            "materials": {
                "MAT_ParticleSPHFluid": {
                    "description": (
                        "Weakly compressible SPH fluid material.  The equation of "
                        "state relates density to pressure via a Tait-type EOS: "
                        "p = (BULK_MODULUS / EXPONENT) * "
                        "((rho/rho0)^EXPONENT - 1) + BACKGROUNDPRESSURE.  "
                        "REFDENSFAC scales the reference density."
                    ),
                    "parameters": {
                        "INITRADIUS": {
                            "description": (
                                "Initial particle radius.  For consistent kernel "
                                "support, use INITRADIUS = 3 * dx for QuinticSpline."
                            ),
                            "range": "Problem-dependent",
                        },
                        "INITDENSITY": {
                            "description": "Reference fluid density",
                            "range": "e.g. 1000 kg/m^3 for water (SI) or 1 (normalized)",
                        },
                        "REFDENSFAC": {
                            "description": (
                                "Reference density factor.  Multiplied with "
                                "INITDENSITY to get the EOS reference density."
                            ),
                            "range": "Typically 1.0",
                        },
                        "BULK_MODULUS": {
                            "description": (
                                "Artificial bulk modulus for weakly compressible SPH.  "
                                "Controls the speed of sound: c = sqrt(BULK_MODULUS/rho).  "
                                "Should be large enough that density variations stay < 1%."
                            ),
                            "range": "Problem-dependent (usually >> rho * v_max^2)",
                        },
                        "DYNAMIC_VISCOSITY": {
                            "description": "Physical dynamic viscosity of the fluid",
                            "range": "e.g. 1e-3 Pa*s for water (SI)",
                        },
                        "BULK_VISCOSITY": {
                            "description": (
                                "Bulk viscosity for numerical stabilization.  "
                                "Often set to 0."
                            ),
                            "range": "0 to small positive value",
                        },
                        "ARTIFICIAL_VISCOSITY": {
                            "description": (
                                "Monaghan-style artificial viscosity coefficient "
                                "for shock capturing.  Set 0 for viscous flows."
                            ),
                            "range": "0 to 1.0 (0.1 typical for shock problems)",
                        },
                        "BACKGROUNDPRESSURE": {
                            "description": (
                                "Background pressure to prevent tensile instability.  "
                                "Set > 0 for free-surface flows."
                            ),
                            "range": "0 or small positive value",
                        },
                        "EXPONENT": {
                            "description": (
                                "Exponent in the Tait equation of state.  "
                                "Typical values: 1 (linear EOS) or 7 (water)."
                            ),
                            "range": "1 to 7",
                        },
                    },
                },
                "MAT_ParticleSPHBoundary": {
                    "description": (
                        "Boundary particle material for rigid walls.  These "
                        "particles are fixed (or prescribed) and provide wall "
                        "boundary conditions via the Adami formulation."
                    ),
                    "parameters": {
                        "INITRADIUS": {
                            "description": "Initial particle radius (same as fluid)",
                            "range": "Same as fluid particle spacing",
                        },
                        "INITDENSITY": {
                            "description": "Density for boundary particles",
                            "range": "Same as fluid density (for Adami formulation)",
                        },
                    },
                },
            },
            "solver": {
                "type": "Explicit (VelocityVerlet)",
                "notes": (
                    "SPH uses explicit time integration.  The time step is "
                    "governed by the CFL condition, viscous condition, and "
                    "body-force condition."
                ),
            },
            "time_integration": {
                "scheme": "Velocity Verlet (symplectic, second-order)",
                "CFL_condition": (
                    "dt < 0.25 * h / c_s where h is the smoothing length "
                    "(~ 3*dx for QuinticSpline) and c_s = sqrt(BULK_MODULUS / rho).  "
                    "Additional viscous constraint: dt < 0.125 * h^2 / nu."
                ),
            },
            "pitfalls": [
                (
                    "[Input] KERNEL_SPACE_DIM must match the physical "
                    "problem dimension.  It does more than normalise "
                    "the kernel: it fixes the particle VOLUME, because "
                    "4C sets mass = INITDENSITY * "
                    "INITIALPARTICLESPACING^KERNEL_SPACE_DIM.  Bumping "
                    "the dimension therefore rescales every particle's "
                    "mass by a factor of the spacing. Signal: none — "
                    "the mismatch is accepted in total silence, no "
                    "warning mentioning kernel, dimension or mismatch "
                    "is emitted, and the run completes.  Do not screen "
                    "for a large density error either: on a settled "
                    "column the density stays well inside 1% of "
                    "INITDENSITY while the velocity field and the "
                    "settled positions move, so compare positions or "
                    "velocities against a reference run. (Audit "
                    "2026-06-02; corrected by execution 2026-08-06.)"
                ),
                (
                    "[Numerical] INITRADIUS for MAT_ParticleSPHFluid "
                    "is the kernel support radius, NOT half the "
                    "particle spacing.  Upstream decks use INITRADIUS "
                    "= 2 * dx for CubicSpline and 3 * dx for "
                    "QuinticSpline. Signal: too small a support makes "
                    "the summation density come out uniformly HIGH, "
                    "not low — with no neighbour inside the support "
                    "each particle sees only itself and the kernel's "
                    "1/h prefactor makes that self-contribution "
                    "exceed INITDENSITY.  Screening for a density "
                    "BELOW INITDENSITY therefore misses it entirely. "
                    "4C has no neighbour-count diagnostic to read; "
                    "result-test density on an interior particle "
                    "instead. (Audit 2026-06-02; corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Numerical] DOMAINBOUNDINGBOX must be large "
                    "enough to contain all particles throughout the "
                    "simulation, including splashing and fluid "
                    "expansion. Signal: same as the PD case, and it is "
                    "not an abort — 4C silently DELETES each escaping "
                    "particle and continues, logging only `on "
                    "processor 0 removed N particle(s) being outside "
                    "the computational domain!` with no id, position, "
                    "step or phase.  On a splashing case that line "
                    "recurs dozens of times through the run while the "
                    "simulation looks healthy; the failure only "
                    "surfaces at the end, as a RESULT DESCRIPTION "
                    "count mismatch when a tested id has been deleted. "
                    "Count those removal lines. (Audit 2026-06-02; "
                    "corrected by execution 2026-08-06.)"
                ),
                (
                    "[Numerical] BULK_MODULUS must be large enough "
                    "that density variations stay below ~1%.  Treat "
                    "BULK_MODULUS >= 100 * rho * v_max^2 as a FLOOR "
                    "to clear, not a value to design to: sitting "
                    "exactly on it (Mach 0.1) already put the density "
                    "excursion several times past the 1% target on "
                    "4C's own 1-D pressure-wave deck.  Upstream decks "
                    "run one to two orders of magnitude above the "
                    "floor. Signal: 4C reports no Mach number and no "
                    "density-variation check, so nothing warns you — "
                    "result-test density on an interior particle and "
                    "compare it against INITDENSITY yourself. (Audit "
                    "2026-06-02; corrected by execution 2026-08-06.)"
                ),
                (
                    "[Input] Boundary particles "
                    "(boundaryphase) should use the SAME "
                    "INITDENSITY as the fluid for the Adami "
                    "boundary formulation to work correctly. "
                    "The formulation extrapolates the wall "
                    "pressure with density-weighted "
                    "averaging across fluid and boundary, "
                    "so a mismatch gives it the wrong "
                    "reference. Signal: none — 4C accepts "
                    "any density ratio without a word, runs "
                    "the whole time loop and reaches the "
                    "result-test manager; the only "
                    "symptom is that the fluid settles "
                    "somewhere else, so compare a settled "
                    "position or a wall-adjacent density "
                    "against a matched-density reference "
                    "run. (Audit 2026-06-02; confirmed by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Numerical] Particle spacing must be uniform at "
                    "initialization.  Non-uniform spacing causes "
                    "zeroth-order kernel approximation errors, and "
                    "4C's mass comes from INITIALPARTICLESPACING "
                    "rather than from the positions you actually "
                    "wrote, so a jittered lattice is inconsistent "
                    "from the first evaluation. Signal: run one step "
                    "and result-test density at an interior particle. "
                    "A correctly packed lattice reproduces "
                    "INITDENSITY to machine precision — exactly, not "
                    "approximately — so ANY visible deviation at t=0 "
                    "is a packing error.  Perturbing the positions by "
                    "a modest fraction of the spacing already puts "
                    "the density well past the 5% mark, and no "
                    "spacing, packing or neighbour-count warning is "
                    "emitted. (Audit 2026-06-02; confirmed and "
                    "sharpened by execution 2026-08-06.)"
                ),
                (
                    "[Numerical] For free-surface problems, "
                    "BACKGROUNDPRESSURE > 0 helps against tensile "
                    "instability — but ONLY together with "
                    "TRANSPORTVELOCITYFORMULATION.  In 4C, "
                    "BACKGROUNDPRESSURE is read exclusively inside "
                    "the transport-velocity branch of SPHMomentum, "
                    "and TRANSPORTVELOCITYFORMULATION defaults to "
                    "NoTransportVelocity, which is where the upstream "
                    "free-surface decks leave it.  Set both, or "
                    "neither has any effect. Signal: setting "
                    "BACKGROUNDPRESSURE alone reproduces the "
                    "untouched run's result-test verdicts to the last "
                    "printed digit; so does setting the transport "
                    "velocity alone with BACKGROUNDPRESSURE at zero. "
                    "Only the combination moves the answer.  Verify "
                    "by diffing the verdict lines, not by reading the "
                    "input back. (Audit 2026-06-02; corrected by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Input] GRAVITY_ACCELERATION must be set "
                    "explicitly for hydrostatic and dam-break "
                    "problems.  The default is zero and 4C never "
                    "warns about a missing or zero body force. "
                    "Signal: the column does not stay approximately "
                    "static, it stays EXACTLY static — velocities "
                    "come back as bit-zero and the density as bit-"
                    "INITDENSITY, with every particle reporting its "
                    "start position.  That exactness is the easiest "
                    "check there is: result-test one velocity "
                    "component and require it to be non-zero. (Audit "
                    "2026-06-02; confirmed by execution 2026-08-06.)"
                ),
                (
                    "[Input] Writing boundaryphase particles does NOT "
                    "give you a wall. BOUNDARYPARTICLEFORMULATION "
                    "defaults to NoBoundaryFormulation, and with that "
                    "default the boundary particles still exist, still "
                    "get their BoundaryPressure and BoundaryVelocity "
                    "states allocated, and are still read by the momentum "
                    "equation — as zeros, because nothing ever fills "
                    "them. The fluid therefore sees a wall at zero "
                    "pressure and zero velocity and sinks into it. "
                    "Signal: none at all. No warning names the "
                    "formulation, the boundary phase or the wall; the "
                    "time loop runs to the end and the only trace is that "
                    "the RESULT DESCRIPTION verdicts on the fluid "
                    "particles come out wrong while the ones on quantities "
                    "the wall does not touch still pass. Set "
                    "BOUNDARYPARTICLEFORMULATION: AdamiBoundaryFormulation "
                    "whenever a boundaryphase exists. The mirror case DOES "
                    "abort: switching the formulation on with no "
                    "boundary/rigid/pd phase in PHASE_TO_MATERIAL_ID gives "
                    "'no boundary or rigid particles defined but a "
                    "boundary particle formulation is set!' from "
                    "particle/src/interaction/"
                    "4C_particle_interaction_sph_boundary_particle.cpp — "
                    "note it is the PHASE MAP that matters, not the "
                    "PARTICLES list, so deleting every boundary particle "
                    "while leaving the phase mapped stays silent. "
                    "(Verified by execution 2026-08-07.)"
                ),
                (
                    "[Input] DENSITYEVALUATION and DENSITYCORRECTION are "
                    "locked to each other in BOTH directions, and each "
                    "half has its own message in its own file. A "
                    "correction scheme is only meaningful for the "
                    "predictor-corrector density, and the "
                    "predictor-corrector density is not usable without "
                    "one. Signal: a correction with DensitySummation or "
                    "DensityIntegration aborts with 'the density "
                    "correction scheme set is not valid with the current "
                    "density evaluation scheme!' from "
                    "particle/src/interaction/4C_particle_interaction_"
                    "sph.cpp; DENSITYEVALUATION: DensityPredictCorrect "
                    "with DENSITYCORRECTION left at its NoCorrection "
                    "default aborts with 'no density correction scheme set "
                    "via parameter 'DENSITYCORRECTION'!' from "
                    "..._sph_density.cpp. Both fire at setup, before the "
                    "first step. The three corrections differ only on "
                    "free-surface particles: InteriorCorrection leaves "
                    "them untouched, NormalizedCorrection divides by the "
                    "colorfield, RandlesCorrection blends in the "
                    "zero-pressure density. (Verified by execution "
                    "2026-08-07.)"
                ),
                (
                    "[Input] Turning temperature on makes THERMALCAPACITY "
                    "mandatory for EVERY phase in PHASE_TO_MATERIAL_ID, "
                    "including the boundary material, and its default is "
                    "0.0. It is easy to add it to the fluid and forget "
                    "the wall. Signal: 'thermal capacity for particles of "
                    "type 'boundaryphase' not positive!' from "
                    "particle/src/interaction/"
                    "4C_particle_interaction_sph_temperature.cpp, which "
                    "does name the phase — read the quoted type and add "
                    "THERMALCAPACITY to THAT material. Its neighbour "
                    "THERMALCONDUCTIVITY is NOT validated and its "
                    "omission is far worse: the conduction term forms a "
                    "harmonic mean of the two particle conductivities, so "
                    "an all-zero pair divides zero by zero and the run "
                    "dies on a raw SIGFPE -- OpenMPI's handler prints the "
                    "floating point exception, signal 8, and 4C prints "
                    "nothing -- with exit status 136 and no 4C error "
                    "block at all. "
                    "Do not expect a frozen temperature field; expect no "
                    "message. (Verified by execution 2026-08-07.)"
                ),
                (
                    "[Output] The diagnostic for a physics module you "
                    "did not switch on is a message about a missing "
                    "STATE, emitted at the very end of the run by the "
                    "result-test manager, not at setup — so nothing "
                    "warns you while the deck is being read. Signal: a "
                    "deck that result-tests QUANTITY 'temperature' "
                    "without TEMPERATUREEVALUATION runs the entire time "
                    "loop and then stops with \"state 'temperature' not "
                    "found in container!\" from particle/src/algorithm/"
                    "4C_particle_algorithm_result_test.cpp. The same "
                    "message covers a state that exists for one phase but "
                    "not another — boundary phases get BoundaryPressure "
                    "and BoundaryVelocity but no Density, so testing "
                    "'density' on a boundary particle id reports it "
                    "missing. Read it as meaning that the module owning "
                    "the state is off, or that this particle's phase "
                    "does not carry it, "
                    "never as a typo in the QUANTITY string; a genuine "
                    "typo gives the different 'result check failed with "
                    "unknown quantity' instead. (Verified by execution "
                    "2026-08-07.)"
                ),
                (
                    "[Numerical] 4C runs NO stability check for SPH. "
                    "There is no CFL computation, no critical time step, "
                    "and not one line of output mentioning the time step "
                    "at any size — the DEM side of the same particle "
                    "engine does compute and print a critical step, which "
                    "makes the absence easy to mistake for a passing "
                    "check. Signal: raising TIMESTEP by one or two orders "
                    "of magnitude on a working deck leaves the log "
                    "completely unchanged in wording; the run reaches "
                    "NUMSTEP normally and only the RESULT DESCRIPTION "
                    "verdicts move. Push further and the first thing that "
                    "ever complains is the binning strategy, 'a particle "
                    "of phase '<name>' traveled more than one bin on this "
                    "processor!' from particle/src/algorithm/"
                    "4C_particle_algorithm.cpp, which names neither the "
                    "time step nor CFL. Size the step yourself against "
                    "the acoustic scale — the speed of sound is derived, "
                    "sqrt(BULK_MODULUS / INITDENSITY) — and against the "
                    "viscous and body-force limits, then confirm by "
                    "halving it and watching the answer stop moving. "
                    "(Verified by execution 2026-08-07.)"
                ),
                (
                    "[Input] ContinuumSurfaceForce is a two-phase "
                    "formulation: it requires containers for BOTH phase1 "
                    "and phase2 even if your problem has one fluid, and "
                    "it is not implemented together with the virtual-wall "
                    "formulation. Signal: three different aborts, all at "
                    "setup and all from files that name the real cause. "
                    "Missing second phase gives 'no particle container "
                    "for particle type 'phase2' found!' from "
                    "particle/src/interaction/"
                    "4C_particle_interaction_sph_surface_tension.cpp. "
                    "SURFACETENSIONCOEFFICIENT defaults to -1.0, so "
                    "omitting it gives 'constant factor of surface "
                    "tension coefficient not positive!' from the same "
                    "file. Combining SURFACETENSIONFORMULATION with "
                    "WALLFORMULATION: VirtualParticleWallFormulation "
                    "gives 'surface tension formulation with wall "
                    "interaction not implemented!' from "
                    "..._sph.cpp — use boundary particles for the wall "
                    "instead. STATICCONTACTANGLE, by contrast, is not "
                    "validated at all and its 0.0 default silently means "
                    "perfect wetting. (Verified by execution 2026-08-07.)"
                ),
                (
                    "[Input] The virtual-wall formulation needs a wall to "
                    "be virtual about, and that wall comes from a "
                    "different section. WALLFORMULATION: "
                    "VirtualParticleWallFormulation is in PARTICLE "
                    "DYNAMIC/SPH but the wall itself is switched on by "
                    "PARTICLE_WALL_SOURCE in PARTICLE DYNAMIC, plus a "
                    "DESIGN SURFACE PARTICLE WALL condition over a "
                    "structural mesh. Signal: setting the formulation "
                    "without the source aborts with 'interface to "
                    "particle wall handler required in virtual wall "
                    "particle handler!' from particle/src/interaction/"
                    "4C_particle_interaction_sph_virtual_wall_particle.cpp"
                    " — a message that names an internal interface, not "
                    "the key you are missing. Boundary particles and "
                    "virtual wall particles are alternatives, not layers: "
                    "the first needs a boundaryphase in the phase map, "
                    "the second needs a meshed surface. (Verified by "
                    "execution 2026-08-07.)"
                ),
                (
                    "[Input] (The PARTICLE DYNAMIC/SPH keys of 4C before "
                    "2026.2.0. 4C 2026.2.0 and later have none of them: open "
                    "boundaries are lists in "
                    "their own section, PARTICLE DYNAMIC/OPEN BOUNDARIES, "
                    "under DIRICHLET_BOUNDARIES and NEUMANN_BOUNDARIES, each "
                    "with a BOUNDARY id and an OUTWARD_NORMAL and "
                    "PLANE_POINT.) The two open-boundary types are NOT "
                    "symmetric, and only one of them tells you when it is "
                    "under-specified. A Dirichlet open boundary demands "
                    "DIRICHLET_FUNCT and a non-zero "
                    "DIRICHLET_OUTWARD_NORMAL; the Neumann side has no "
                    "equivalent check on NEUMANN_FUNCT and simply treats "
                    "its absence as a zero-pressure outlet. Signal: the "
                    "Dirichlet failures are clean — 'no function id of "
                    "prescribed state set!' and 'no outward normal set!' "
                    "from particle/src/interaction/"
                    "4C_particle_interaction_sph_open_boundary.cpp, both "
                    "at setup. The Neumann omission produces no message "
                    "at all, which is correct behaviour and is also why "
                    "you cannot tell a deliberate zero-pressure outlet "
                    "from a forgotten key by reading the log. Both types "
                    "also need their own phase present, dirichletphase "
                    "and neumannphase respectively, alongside phase1. "
                    "Note the sign: the prescribed velocity is the "
                    "function value times the NEGATIVE outward normal, so "
                    "a positive function drives flow INTO the domain. "
                    "(Verified by execution 2026-08-07.)"
                ),
                (
                    "[Input] There is no SOUNDSPEED key and no "
                    "SMOOTHING_LENGTH key in 4C's SPH input — neither "
                    "string appears anywhere in the source or in any "
                    "upstream deck. The speed of sound is DERIVED as "
                    "sqrt(BULK_MODULUS / INITDENSITY), and the smoothing "
                    "length is the material's INITRADIUS. Signal: writing "
                    "either name gives a MATERIALS or section parse "
                    "error that names the section but not the key, so it "
                    "reads like a structural problem with your YAML. "
                    "Choosing EQUATIONOFSTATE: IdealGas has its own "
                    "silent edge: it uses only the derived sound speed, "
                    "and REFDENSFAC and EXPONENT — both still REQUIRED by "
                    "the material parser, neither with a default — are "
                    "then read and ignored. Changing them under IdealGas "
                    "reproduces the untouched run's verdicts to the last "
                    "printed digit, so a deck can look retuned while "
                    "nothing moved. Every upstream SPH deck leaves "
                    "EQUATIONOFSTATE at its GenTait default. (Verified by "
                    "execution 2026-08-07.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "Dam break (2D)",
                    "description": (
                        "Classic SPH benchmark: a column of fluid collapses under "
                        "gravity and splashes against a wall.  Validates free-surface "
                        "tracking and pressure computation."
                    ),
                },
                {
                    "name": "Poiseuille flow (2D)",
                    "description": (
                        "Pressure-driven flow between parallel plates.  Steady-state "
                        "parabolic velocity profile validates viscous force "
                        "implementation.  Good for convergence studies."
                    ),
                    "template_variant": "poiseuille_2d",
                },
                {
                    "name": "Hydrostatic tank",
                    "description": (
                        "Fluid at rest in a container under gravity.  The pressure "
                        "must be linear with depth (p = rho*g*h).  Tests density "
                        "evaluation and boundary conditions."
                    ),
                },
                {
                    "name": "1D pressure wave",
                    "description": (
                        "A Gaussian velocity perturbation propagates as a pressure "
                        "wave in a 1D column.  Validates the equation of state and "
                        "wave speed."
                    ),
                },
            ],
        }

    # ── Templates ─────────────────────────────────────────────────────

    def get_template(self, variant: str = "poiseuille_2d") -> str:
        if variant == "poiseuille_2d":
            return self._template_poiseuille_2d()
        raise ValueError(
            f"Unknown variant {variant!r} for {self.module_key}.  "
            f"Available: {[v['name'] for v in self.list_variants()]}"
        )

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "poiseuille_2d",
                "description": (
                    "2D Poiseuille flow between parallel plates.  Demonstrates "
                    "SPH fluid setup with boundary particles, viscous flow, and "
                    "pressure-driven steady state.  Good introductory SPH example."
                ),
            },
        ]

    # ── Validation ────────────────────────────────────────────────────

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        """Physics-aware validation of SPH parameters.

        Expected keys in *params* (all optional):
            dx              - particle spacing
            kernel          - kernel name (QuinticSpline / CubicSpline)
            kernel_dim      - Kernel1D / Kernel2D / Kernel3D
            problem_dim     - 1, 2, or 3
            bulk_modulus    - artificial bulk modulus
            density         - reference density
            v_max           - expected maximum velocity
            dt              - time step
            init_radius     - INITRADIUS from material
            domain_bbox     - [xmin, ymin, zmin, xmax, ymax, zmax]
            particles_bbox  - [xmin, ymin, zmin, xmax, ymax, zmax]
        """
        warnings: list[str] = []

        dx = params.get("dx")
        kernel_dim = params.get("kernel_dim", "")
        problem_dim = params.get("problem_dim")
        bulk_modulus = params.get("bulk_modulus")
        density = params.get("density")
        v_max = params.get("v_max")
        dt = params.get("dt")
        init_radius = params.get("init_radius")
        domain_bbox = params.get("domain_bbox")
        particles_bbox = params.get("particles_bbox")

        # Kernel dimension vs problem dimension
        if kernel_dim and problem_dim is not None:
            expected = f"Kernel{problem_dim}D"
            if kernel_dim != expected:
                warnings.append(
                    f"ERROR: KERNEL_SPACE_DIM is {kernel_dim!r} but the problem "
                    f"is {problem_dim}D.  Expected {expected!r}.  This causes "
                    f"incorrect kernel normalization."
                )

        # Init radius check for QuinticSpline
        if init_radius is not None and dx is not None:
            expected_radius = 3.0 * dx
            ratio = init_radius / expected_radius
            if ratio < 0.8 or ratio > 1.2:
                warnings.append(
                    f"WARNING: INITRADIUS = {init_radius} does not match expected "
                    f"3*dx = {expected_radius} for QuinticSpline kernel.  "
                    f"This may cause incorrect neighbor counts."
                )

        # Bulk modulus adequacy
        if bulk_modulus is not None and density is not None and v_max is not None:
            if density > 0 and v_max > 0:
                min_bulk = 100.0 * density * v_max ** 2
                if bulk_modulus < min_bulk:
                    warnings.append(
                        f"WARNING: BULK_MODULUS = {bulk_modulus} may be too small.  "
                        f"For density variations < 1%, need BULK_MODULUS >= "
                        f"{min_bulk:.2e} (100 * rho * v_max^2)."
                    )

        # CFL for SPH
        if dt is not None and dx is not None and bulk_modulus is not None and density is not None:
            if density > 0 and bulk_modulus > 0:
                import math
                c_s = math.sqrt(bulk_modulus / density)
                h = 3.0 * dx  # smoothing length for QuinticSpline
                dt_cfl = 0.25 * h / c_s
                if dt > dt_cfl:
                    warnings.append(
                        f"ERROR: CFL VIOLATION.  dt = {dt} > dt_CFL = {dt_cfl:.6e}.  "
                        f"Reduce time step."
                    )

        # Bounding box encloses particles
        if domain_bbox is not None and particles_bbox is not None:
            if len(domain_bbox) == 6 and len(particles_bbox) == 6:
                labels = ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]
                for i in range(3):
                    if domain_bbox[i] > particles_bbox[i]:
                        warnings.append(
                            f"ERROR: DOMAINBOUNDINGBOX {labels[i]}={domain_bbox[i]} "
                            f"> particle {labels[i]}={particles_bbox[i]}."
                        )
                for i in range(3, 6):
                    if domain_bbox[i] < particles_bbox[i]:
                        warnings.append(
                            f"ERROR: DOMAINBOUNDINGBOX {labels[i]}={domain_bbox[i]} "
                            f"< particle {labels[i]}={particles_bbox[i]}."
                        )

        return warnings

    # ── Private template builders ─────────────────────────────────────

    @staticmethod
    def _template_poiseuille_2d() -> str:
        """Minimal 2D Poiseuille flow between parallel plates."""
        return textwrap.dedent("""\
            # 2D SPH Poiseuille Flow
            # Pressure-driven flow between two parallel plates.
            # Steady-state: parabolic velocity profile  u(y) = dp/dx / (2*mu) * y*(H-y)
            #
            # FORMAT TEMPLATE — all values are placeholders.
            # Determine domain size, material properties, and particle spacing
            # based on the specific problem. Consult the solver's test files.
            #
            # NOTE: Replace the PARTICLES section with actual particle positions
            # generated by a script.

            PROBLEM TYPE:
              PROBLEMTYPE: "Particle"

            IO:
              STDOUTEVERY: <output_frequency>
              VERBOSITY: "Standard"

            # NO IO/RUNTIME VTK OUTPUT/PARTICLES section — it does not
            # exist and adding it is a hard parse error, "Section
            # 'IO/RUNTIME VTK OUTPUT/PARTICLES' is not a valid section
            # name." Particle ParaView output is written unconditionally
            # at the RESULTSEVERY interval below.

            BINNING STRATEGY:
              BIN_SIZE_LOWER_BOUND: <must be > kernel support radius>
              DOMAINBOUNDINGBOX: "<xmin> <ymin> <zmin> <xmax> <ymax> <zmax>"

            PARTICLE DYNAMIC:
              INTERACTION: "SPH"
              RESULTSEVERY: <output_frequency>
              RESTARTEVERY: <restart_frequency>
              TIMESTEP: <dt — must satisfy CFL and viscous stability>
              NUMSTEP: <total_steps>
              MAXTIME: <end_time>
              GRAVITY_ACCELERATION: "0.0 0.0 0.0"
              PHASE_TO_DYNLOADBALFAC: "phase1 1.0 boundaryphase 1.0"
              PHASE_TO_MATERIAL_ID: "phase1 1 boundaryphase 2"

            PARTICLE DYNAMIC/INITIAL AND BOUNDARY CONDITIONS:
              INITIAL_VELOCITY_FIELD: "phase1 1"

            FUNCT1:
              - COMPONENT: 0
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<initial_vx>"
              - COMPONENT: 1
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<initial_vy>"
              - COMPONENT: 2
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"

            PARTICLE DYNAMIC/SPH:
              KERNEL: QuinticSpline
              KERNEL_SPACE_DIM: Kernel2D
              INITIALPARTICLESPACING: <dx — choose based on problem>
              DENSITYEVALUATION: DensityIntegration
              BOUNDARYPARTICLEFORMULATION: AdamiBoundaryFormulation

            MATERIALS:
              - MAT: 1
                MAT_ParticleSPHFluid:
                  INITRADIUS: <kernel_support = 3*dx for QuinticSpline>
                  INITDENSITY: <fluid_density>
                  REFDENSFAC: 1
                  EXPONENT: 1
                  BACKGROUNDPRESSURE: 0
                  BULK_MODULUS: <bulk_modulus>
                  DYNAMIC_VISCOSITY: <viscosity>
                  BULK_VISCOSITY: 0
                  ARTIFICIAL_VISCOSITY: 0
              - MAT: 2
                MAT_ParticleSPHBoundary:
                  INITRADIUS: <same as fluid INITRADIUS>
                  INITDENSITY: <same as fluid INITDENSITY>

            # Generate particle positions programmatically
            PARTICLES:
              - "TYPE phase1 POS <x> <y> 0.0"
              - "TYPE boundaryphase POS <x> <y> 0.0"
        """)
