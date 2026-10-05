"""Generator for structural dynamics physics module (GenAlpha, explicit, damping).

Covers time-dependent structural problems with inertia effects, using
DYNAMICTYPE: GenAlpha (or OneStepTheta, ExplicitEuler) in the STRUCTURAL
DYNAMIC section of 4C.  Produces validated, working .4C.yaml templates.
"""

from __future__ import annotations

from typing import Any

from .base import BaseGenerator


class StructuralDynamicsGenerator(BaseGenerator):
    """Generator for structural dynamics problems in 4C.

    Supports implicit time integration via Generalised-Alpha (GenAlpha),
    OneStepTheta, and explicit Euler.  Includes Rayleigh damping,
    GenAlpha spectral-radius tuning, and beam-specific Lie-group
    integration.
    """

    module_key = "structural_dynamics"
    display_name = "Structural Dynamics (GenAlpha / Explicit / Damping)"
    problem_type = "Structure"

    # ── Knowledge ─────────────────────────────────────────────────────

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The structural dynamics module solves time-dependent structural "
                "problems where inertia effects are important: impact, vibration, "
                "wave propagation.  The PROBLEM TYPE is 'Structure' (same as "
                "quasi-static), but DYNAMICTYPE is set to a transient integrator "
                "instead of 'Statics'.  The recommended integrator is GenAlpha "
                "(Generalised-Alpha), which provides controllable numerical "
                "dissipation of high-frequency modes via the spectral radius "
                "parameter RHO_INF.  The mass matrix is assembled from element "
                "densities (DENS in the material), which is therefore MANDATORY."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "STRUCTURAL DYNAMIC",
                "SOLVER 1",
                "MATERIALS",
                "STRUCTURE GEOMETRY",
            ],
            "materials": {
                "MAT_Struct_StVenantKirchhoff": {
                    "description": (
                        "Standard structural material.  DENS is CRITICAL for "
                        "dynamics because it determines the mass matrix.  "
                        "Zero density means zero inertia."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Young's modulus E",
                            "range": "> 0",
                        },
                        "NUE": {
                            "description": "Poisson's ratio nu",
                            "range": "0 < nu < 0.5",
                        },
                        "DENS": {
                            "description": (
                                "Mass density -- MANDATORY for dynamics.  "
                                "Determines the mass matrix M."
                            ),
                            "range": "> 0  (e.g. steel: 7.85e-9 t/mm^3 in mm-t-s units)",
                        },
                    },
                },
                "MAT_ElastHyper + ELAST_CoupNeoHooke": {
                    "description": (
                        "Neo-Hookean hyperelastic for large-deformation dynamics.  "
                        "DENS / NUMMAT / MATIDS / POLYCONVEX live on the "
                        "MAT_ElastHyper wrapper; YOUNG / NUE live on the "
                        "ELAST_CoupNeoHooke sub-material."
                    ),
                    "parameters": {
                        "YOUNG": {"description": "Young's modulus (sub-material)", "range": "> 0"},
                        "NUE": {"description": "Poisson's ratio (sub-material)", "range": "0 < nu < 0.5"},
                        "DENS": {"description": "Mass density (wrapper)", "range": "> 0"},
                        "NUMMAT": {"description": "Number of sub-materials (wrapper)", "range": "1"},
                        "MATIDS": {"description": "List of sub-material IDs (wrapper)", "range": "[<id>]"},
                        "POLYCONVEX": {
                            "description": "Polyconvexity check flag (wrapper)",
                            "range": "0 | 1",
                        },
                    },
                },
            },
            "time_integration": {
                "DYNAMICTYPE": (
                    "Time integrator selection.  Keyword strings below match the "
                    "4C input parser (src/inpar/4C_inpar_structure.cpp).\n"
                    "Implicit options:\n"
                    "  'GenAlpha' -- Generalised-Alpha (RECOMMENDED for most problems).  "
                    "Implicit, second-order accurate, controllable high-frequency "
                    "damping via RHO_INF.  Best all-round choice.\n"
                    "  'GenAlphaLieGroup' -- Lie-group variant for beam elements "
                    "with rotational DOFs.  Required for BEAM3R/BEAM3K dynamics.\n"
                    "  'OneStepTheta' -- Theta-method (theta=0.5: Newmark, "
                    "theta=1.0: backward Euler).  Simpler but less control "
                    "over numerical dissipation.\n"
                    "Explicit options (matrix-free, CFL-constrained, no global Newton):\n"
                    "  'CentrDiff' -- Central differences.  Second-order accurate, "
                    "the standard choice for explicit solid dynamics (impact, "
                    "wave propagation, high strain rates).\n"
                    "  'AdamsBashforth2' -- Two-step Adams-Bashforth, second-order.\n"
                    "  'AdamsBashforth4' -- Four-step Adams-Bashforth, fourth-order "
                    "(higher accuracy but larger startup cost and stricter stability).\n"
                    "  'ExplicitEuler' -- Forward Euler, first-order.  Only for "
                    "diagnostic / very short transients -- prefer CentrDiff.\n"
                    "All explicit schemes require dt < h/c (CFL).  Whether a "
                    "given plasticity / damage material is wired to the explicit "
                    "path depends on its evaluate() interface -- verify with a "
                    "small benchmark before relying on it."
                ),
                "GenAlpha_parameters": (
                    "Configured in the 'STRUCTURAL DYNAMIC/GENALPHA' sub-section.  "
                    "Key parameters:\n"
                    "  RHO_INF -- Spectral radius at infinite frequency [0, 1].  "
                    "RHO_INF=1.0: no numerical damping (energy-conserving).  "
                    "RHO_INF=0.0: maximum high-frequency damping.  "
                    "Typical: 0.8--0.9 for moderate damping.\n"
                    "  BETA, GAMMA, ALPHA_M, ALPHA_F -- Newmark/GenAlpha "
                    "coefficients.  Usually derived from RHO_INF automatically; "
                    "only override for advanced use."
                ),
                "TIMESTEP": (
                    "Time step size.  CRITICAL: must be small enough to resolve "
                    "the highest relevant frequency in the response.  Rule of "
                    "thumb: dt < T_min / 10 where T_min is the period of the "
                    "highest mode of interest.  For explicit methods dt must "
                    "satisfy the CFL condition (dt < h / c where h is element "
                    "size and c is wave speed)."
                ),
                "NUMSTEP": "Total number of time steps.",
                "MAXTIME": "Maximum simulation time.",
            },
            "damping": {
                "DAMPING": (
                    "Damping model.  Options:\n"
                    "  'None' -- No physical damping (numerical damping from "
                    "GenAlpha RHO_INF < 1 still applies).\n"
                    "  'Rayleigh' -- Classical Rayleigh damping: "
                    "C = alpha_M * M + alpha_K * K.  Set M_DAMP and K_DAMP.\n"
                    "  'Material' -- Damping defined at the material level."
                ),
                "M_DAMP": (
                    "Mass-proportional Rayleigh damping coefficient alpha_M.  "
                    "Damps low-frequency modes.  Typical: 0.0 -- 1.0."
                ),
                "K_DAMP": (
                    "Stiffness-proportional Rayleigh damping coefficient alpha_K.  "
                    "Damps high-frequency modes.  Typical: 1e-5 -- 1e-3.  "
                    "Large K_DAMP can make the problem very stiff."
                ),
            },
            "solver": {
                "small_problems": {
                    "SOLVER": "UMFPACK",
                    "description": "Direct solver, robust for small dynamic problems.",
                },
                "large_problems": {
                    "SOLVER": "Belos",
                    "AZPREC": "MueLu",
                    "description": (
                        "Iterative solver with AMG.  For dynamics the system "
                        "matrix includes mass contributions and is often better "
                        "conditioned than pure stiffness problems."
                    ),
                },
            },
            "pitfalls": [
                (
                    "[Physics] DENS (density) in the MAT material is "
                    "CRITICAL for dynamics: DENS = 0 makes the mass matrix "
                    "zero and the dynamic operator singular. It does NOT "
                    "quietly degrade to a quasi-static answer under an "
                    "implicit scheme — an earlier version of this entry said "
                    "it would. Both families die at step 0, and with the same "
                    "message: GenAlpha and ExplicitEuler each abort on 'You "
                    "are about to invert a singular matrix!' from "
                    "structure_new/4C_structure_new_integrator.cpp, exit 1, "
                    "with no 'Finalised step' line at all, while the same deck "
                    "under DYNAMICTYPE: Statics runs its steps normally. "
                    "Signal: that singular-matrix line plus a zero step count. "
                    "The diagnostic never contains the substring 'densit' in "
                    "either case, so grepping for the material parameter finds "
                    "nothing — verify DENS > 0 in the MAT block by inspection. "
                    "(Claim inherited 2026-06-02; corrected by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Numerical] Explicit time stepping (DYNAMICTYPE: "
                    "ExplicitEuler / CentrDiff -- it is 'ExplicitEuler' "
                    "spelled out, and 'ExplEuler' is REJECTED with 'Could "
                    "not match this input') requires CFL: dt < h_min / c "
                    "where c = sqrt(E/rho) is the elastic wave speed, and 4C "
                    "does NOT check it for you — nothing in the output ever "
                    "mentions CFL, Courant or a stability limit. Signal: the "
                    "explicit path prints its own per-step increment norm as "
                    "'||dx||=...'; under the limit that number creeps up "
                    "linearly, over it, it multiplies by orders of magnitude "
                    "every step until the process is killed by SIGFPE, with "
                    "shell exit status 136 and only the MPI runtime's "
                    "signal block to show for it, "
                    "part-way through the requested steps. Read the ||dx|| "
                    "sequence, not the exit code. (Claim inherited; confirmed "
                    "and given its real signal by execution 2026-08-06.)"
                ),
                (
                    "[API] For beam elements (BEAM3R, BEAM3K) DYNAMICTYPE: "
                    "GenAlphaLieGroup with MASSLIN: rotations is not a "
                    "recommendation, it is a required PAIR — and the two "
                    "halves are enforced very unevenly. Setting MASSLIN: "
                    "rotations under classical GenAlpha aborts with the "
                    "perfect message 'MASSLIN=ml_rotations is not supported by "
                    "classical GenAlpha! Choose GenAlphaLieGroup instead!' "
                    "from structure_new/src/implicit/"
                    "4C_structure_new_impl_genalpha.cpp. Forgetting MASSLIN on "
                    "a GenAlphaLieGroup run produces NO 4C diagnostic at all: "
                    "the process is killed by 'Signal: Segmentation fault "
                    "(11)' inside GenAlphaLieGroup::post_setup -> "
                    "Beam3r::calc_inertia_force_and_mass_matrix, shell exit "
                    "status 139, before any step. Signal: a beam dynamics deck "
                    "that segfaults during setup is missing MASSLIN: "
                    "rotations. (Claim inherited; given its real signals by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Numerical] RHO_INF is the high-frequency dissipation "
                    "parameter for generalised-alpha, and it lives in the "
                    "nested section 'STRUCTURAL DYNAMIC/GENALPHA', not in "
                    "STRUCTURAL DYNAMIC. Its default is 1.0, energy-conserving "
                    "but admitting spurious high-frequency ringing; reduce it "
                    "if the solution rings. Signal: writing RHO_INF one level "
                    "up is rejected at parse with 'Could not match this input' "
                    "and the RHO_INF line echoed, so the misplacement is loud "
                    "rather than silent. To read the value actually in force, "
                    "look at the coefficient banner 4C prints at start-up — "
                    "lines of the form 'rho = ', 'beta = ', 'gamma = ', "
                    "'alpha_f = ', 'alpha_m = '; writing RHO_INF: 1.0 "
                    "explicitly reproduces the default set bit for bit. "
                    "(Claim inherited; section placement and signal added by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Numerical] Rayleigh damping needs THREE keys in "
                    "STRUCTURAL DYNAMIC, not two: DAMPING: 'Rayleigh' plus "
                    "M_DAMP and K_DAMP. DAMPING defaults to 'None', and a deck "
                    "that sets the two coefficients and forgets the switch is "
                    "run undamped — same answer to the last bit as with no "
                    "damping keys at all, exit 0, and 4C says nothing about "
                    "damping anywhere in the log. Signal: if damping appears "
                    "to do nothing, check DAMPING before checking the "
                    "coefficients. The reverse mistake is loud: DAMPING: "
                    "'Rayleigh' without the coefficients aborts with 'Rayleigh "
                    "damping parameter K_DAMP not explicitly given.' from "
                    "adapter/4C_adapter_str_structure_new.cpp. The physics is "
                    "unchanged — M_DAMP damps low frequencies, K_DAMP high, so "
                    "calibrate the pair at two target frequencies. (Claim "
                    "inherited; the silently-inert failure mode found by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[API] The STRUCTURAL DYNAMIC.PREDICT default is "
                    "ConstDis, not ConstDisVelAcc, and an earlier version of "
                    "this entry had the cost ordering backwards as well: on a "
                    "transient HEX8 cantilever, TangDis converged in FEWER "
                    "total Newton iterations than ConstDisVelAcc, not more. "
                    "All predictors reached the same answer, so this is a cost "
                    "question and not a correctness one — measure it on your "
                    "own problem instead of taking a recommendation. Signal: "
                    "4C echoes the predictor it selected as '=== Structural "
                    "predictor: <name> ===' and the per-step cost as 'nlniter "
                    "N' in the 'Finalised step' banner, so both are readable "
                    "straight from the log. The enum is validated, so a "
                    "near-miss such as ConstDisVelAccel is rejected with "
                    "'Could not match this input' rather than silently "
                    "defaulting. (Falsified and corrected by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Syntax] STRUCTURAL DYNAMIC.DYNAMICTYPE validates "
                    "against an allowed enum at YAML parse — values like "
                    "Statics, GenAlpha, OneStepTheta, ExplicitEuler, "
                    "CentrDiff, GenAlphaLieGroup, AdamsBashforth2, "
                    "AdamsBashforth4 -- the full list this build accepts. "
                    "Note 'ExplicitEuler', NOT 'ExplEuler'. A typo or "
                    "made-up name "
                    "(e.g. 'TotallyMadeUpScheme') is rejected with "
                    "'PROC 0 ERROR ... Could not match this input' from "
                    "core/io/src/4C_io_input_spec_builders.cpp, with the "
                    "STRUCTURAL DYNAMIC block echoed. Signal: stderr "
                    "contains 'Could not match this input' + 'STRUCTURAL "
                    "DYNAMIC' + 'DYNAMICTYPE'. (Verified empirically "
                    "2026-06-01.)"
                ),
                (
                    "[Physics] DENS: 0.0 in a transient run does NOT "
                    "quietly degrade to a quasi-static answer — it "
                    "aborts before the first step with an explicit "
                    "singular-matrix message, which is the good "
                    "outcome. A NEGATIVE density is the genuinely "
                    "nasty case: it parses, it integrates two steps, "
                    "and only then dies inside NOX with a generic "
                    "non-convergence message that says nothing about "
                    "the material. MAT_Struct_StVenantKirchhoff puts "
                    "no validator on DENS, so a sign typo survives "
                    "the parser. Signal: 'You are about to invert a "
                    "singular matrix!' from "
                    "structure_new_integrator at step 0 means DENS "
                    "is exactly 0; 'The nonlinear solver did not "
                    "converge!' from solver_nonlin_nox_problem after "
                    "a couple of steps in an otherwise well-posed "
                    "deck should send you to check the sign of DENS "
                    "before touching TOLRES or MAXITER. (Verified by "
                    "execution 2026-08-03, HEX8 unit cube, "
                    "DYNAMICTYPE GenAlpha, dt 0.05: DENS 1.0 "
                    "completed 4/4 steps with exit 0; DENS 0.0 "
                    "aborted at step 0 with the singular-matrix "
                    "throw; DENS -1.0 reached step 2 of 4 and then "
                    "raised the NOX non-convergence throw.)"
                ),
                (
                    "[Numerical] The time loop stops at whichever of "
                    "NUMSTEP and MAXTIME is reached FIRST, and the "
                    "early stop is silent and exits 0. A deck asking "
                    "for NUMSTEP: 100 at TIMESTEP: 0.1 with MAXTIME "
                    "left at its default-ish small value simply runs "
                    "3 steps and reports success, so a 'converged' "
                    "result can be a fraction of the intended "
                    "transient. MAXTIME: 0 is the degenerate case: "
                    "the loop body never executes, no step line is "
                    "printed at all, and the process still exits 0. "
                    "Always set MAXTIME = NUMSTEP * TIMESTEP unless "
                    "you deliberately want one of them to clip the "
                    "other. Signal: the per-step banner, whose "
                    "literal is 'Finalised step' with the step and "
                    "the total streamed in after it, showing K < N "
                    "and no error is the MAXTIME clip; the complete "
                    "ABSENCE of any 'Finalised step' line together "
                    "with exit 0 is MAXTIME <= 0. (Verified by "
                    "execution 2026-08-03: TIMESTEP 0.1 / NUMSTEP "
                    "100 / MAXTIME 0.3 printed Finalised step 3 / "
                    "100 and exited 0; TIMESTEP 0.1 / NUMSTEP 3 / "
                    "MAXTIME 100 printed Finalised step 3 / 3; "
                    "MAXTIME 0.0 printed no step line and exited 0.)"
                ),
                (
                    "[Output] RESTARTEVERY must sit in STRUCTURAL "
                    "DYNAMIC, not in IO. The key exists in BOTH "
                    "sections (inpar_io and inpar_structure), but "
                    "the structural integrator reads it from the "
                    "STRUCTURAL DYNAMIC sub-list in "
                    "structure_new_timint_basedataio, so the IO "
                    "placement is accepted, runs to exit 0 and "
                    "writes no restart records — the mistake only "
                    "shows up when the restart is attempted. Once "
                    "placed correctly, `4C --restart=N in.4C.yaml "
                    "out` resumes from step N and writes to a new "
                    "output identifier (out-1) rather than "
                    "overwriting. Signal: no "
                    "<prefix>.result.structure.s<N> file next to "
                    "the .control file after a run that requested "
                    "restarts; on the restart attempt, \"No restart "
                    "entry for discretization 'structure' step N in "
                    "control file. Control file corrupt?\" from "
                    "io_control. (Verified by execution 2026-08-03, "
                    "4-step deck: RESTARTEVERY: 2 under IO produced "
                    "only .control and .mesh.structure.s0 and the "
                    "restart failed; the same key under STRUCTURAL "
                    "DYNAMIC produced .result.structure.s2 and "
                    "--restart=2 printed '====== Restart of the "
                    "structural simulation from step 2', ran steps "
                    "3 and 4 and exited 0.)"
                ),
                (
                    "[Numerical] DYNAMICTYPE is OPTIONAL and its "
                    "default is GenAlpha, i.e. a TRANSIENT scheme. "
                    "Omitting it does not give you a static "
                    "analysis and does not warn — the deck runs, "
                    "exits 0, and returns a different number, "
                    "because inertia and the generalised-alpha "
                    "averaging are now in the residual. Write "
                    "DYNAMICTYPE explicitly in every structural "
                    "deck, including the ones you think are "
                    "obviously static. LINEAR_SOLVER is the "
                    "opposite case and needs no such care: its "
                    "default of -1 is not a valid solver id, so "
                    "omitting it fails loudly and immediately. "
                    "Signal: an unexpectedly transient-looking "
                    "answer from a deck with no DYNAMICTYPE line, "
                    "with the 'Finalised step' banner printing "
                    "normally (the step and the total are streamed in "
                    "after that literal, as Finalised step K / N) "
                    "normally and no diagnostic at all; compare "
                    "against the same deck with DYNAMICTYPE: "
                    "Statics before suspecting the mesh or the "
                    "material. The loud sibling is 'no linear "
                    "solver defined for structural field. Please "
                    "set LINEAR_SOLVER in STRUCTURAL DYNAMIC to a "
                    "valid number!' from "
                    "structure_new_solver_factory, exit 1. "
                    "(Verified by execution on a single-HEX8 unit "
                    "cube with a surface Neumann: deleting only the "
                    "DYNAMICTYPE line left the deck exiting 0 while "
                    "returning a visibly different displacement, and "
                    "deleting LINEAR_SOLVER instead exited 1 with the "
                    "solver-factory message. The measured values live "
                    "in the Tier-2 fixture "
                    "structural_dynamics_dynamictype_default_is_transient, "
                    "not here. Re-confirmed 2026-08-06.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "impact_2d",
                    "description": (
                        "2D impact / sudden-load problem.  A block is fixed on "
                        "one side and a sudden pressure is applied on the "
                        "opposite side.  Uses GenAlpha with RHO_INF: 0.9 to "
                        "damp spurious high-frequency modes."
                    ),
                    "template_variant": "genalpha_2d",
                },
                {
                    "name": "vibration_3d",
                    "description": (
                        "Free vibration of a 3D beam or block after an initial "
                        "displacement.  Uses GenAlpha with RHO_INF: 1.0 "
                        "(energy-conserving) to study natural frequencies."
                    ),
                    "template_variant": "genalpha_2d",
                },
            ],
        }

    # ── Templates ─────────────────────────────────────────────────────

    _TEMPLATES: dict[str, str] = {
        "genalpha_2d": """\
# FORMAT TEMPLATE — all numerical values are placeholders.
# Written in the input grammar of 4C before 2026.2.0 (the WALL element); from
# 4C 2026.2.0 on a 2-D solid is SOLID with THICKNESS and PLANE_ASSUMPTION. The decks openPASO runs are served in the grammar of the
# 4C it finds.
TITLE:
  - "Structural dynamics -- 2D impact with GenAlpha time integration"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
IO:
  STRUCT_STRESS: "Cauchy"
  STRUCT_STRAIN: "GL"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: <output_interval_steps>
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
  STRESS_STRAIN: true
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "GenAlpha"
  PREDICT: "ConstDisVelAcc"
  TIMESTEP: <timestep>
  NUMSTEP: <number_of_steps>
  MAXTIME: <end_time>
  TOLDISP: <displacement_tolerance>
  TOLRES: <residual_tolerance>
  MAXITER: <max_iterations>
  DAMPING: "Rayleigh"
  M_DAMP: <mass_damping_coefficient>
  K_DAMP: <stiffness_damping_coefficient>
  LINEAR_SOLVER: 1
STRUCTURAL DYNAMIC/GENALPHA:
  RHO_INF: <spectral_radius_rho_inf>
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct_solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: <Young_modulus>
      NUE: <Poisson_ratio>
      DENS: <density>
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    ENTITY_TYPE: node_set_id
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0.0, 0.0]
    FUNCT: [0, 0]
DESIGN LINE NEUMANN CONDITIONS:
  - E: 2
    ENTITY_TYPE: node_set_id
    NUMDOF: 6
    ONOFF: [1, 0, 0, 0, 0, 0]
    VAL: [<applied_load>, 0.0, 0.0, 0.0, 0.0, 0.0]
    FUNCT: [1, 0, 0, 0, 0, 0]
STRUCTURE GEOMETRY:
  ELEMENT_BLOCKS:
    - ID: 1
      WALL:
        QUAD4:
          MAT: 1
          KINEM: linear
          THICK: <thickness>
          STRESS_STRAIN: plane_strain
  FILE: mesh.e
  SHOW_INFO: detailed_summary
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 1
      QUANTITY: "dispx"
      VALUE: <expected_result_value>
      TOLERANCE: <result_tolerance>
""",
    }

    def get_template(self, variant: str = "default") -> str:
        if variant == "default":
            variant = "genalpha_2d"
        if variant not in self._TEMPLATES:
            available = ", ".join(sorted(self._TEMPLATES))
            raise ValueError(
                f"Unknown template variant {variant!r} for {self.module_key}. "
                f"Available: {available}"
            )
        return self._TEMPLATES[variant]

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "genalpha_2d",
                "description": (
                    "2D structural dynamics with GenAlpha time integration, "
                    "Rayleigh damping, SOLID QUAD4 plane-strain elements, "
                    "St. Venant-Kirchhoff material.  Suitable as starting "
                    "point for impact and vibration problems."
                ),
            },
        ]

    # ── Validation ────────────────────────────────────────────────────

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        errors: list[str] = []

        # Check TIMESTEP
        timestep = params.get("TIMESTEP")
        if timestep is not None:
            try:
                dt = float(timestep)
                if dt <= 0:
                    errors.append(
                        f"TIMESTEP must be > 0 for dynamic analysis, got {dt}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"TIMESTEP must be a positive number, got {timestep!r}."
                )

        # Check DENS -- mandatory for dynamics
        dens = params.get("DENS")
        if dens is not None:
            try:
                d = float(dens)
                if d <= 0:
                    errors.append(
                        f"DENS (density) must be > 0 for structural dynamics, "
                        f"got {d}.  Zero or negative density means zero or "
                        f"invalid mass matrix -- the dynamic solve will fail."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"DENS must be a positive number, got {dens!r}."
                )
        else:
            errors.append(
                "DENS (density) is not specified.  It is MANDATORY for "
                "structural dynamics -- the mass matrix cannot be assembled "
                "without it."
            )

        # Check RHO_INF (GenAlpha spectral radius)
        rho_inf = params.get("RHO_INF")
        if rho_inf is not None:
            try:
                r = float(rho_inf)
                if r < 0 or r > 1:
                    errors.append(
                        f"RHO_INF must be in [0, 1], got {r}.  "
                        f"0 = maximum high-frequency damping, "
                        f"1 = energy-conserving (no numerical damping)."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"RHO_INF must be a number in [0, 1], got {rho_inf!r}."
                )

        # Check GenAlpha sub-parameters if provided
        for coeff_name in ("BETA", "GAMMA", "ALPHA_M", "ALPHA_F"):
            val = params.get(coeff_name)
            if val is not None:
                try:
                    c = float(val)
                    if c < 0 or c > 1:
                        errors.append(
                            f"{coeff_name} should be in [0, 1], got {c}."
                        )
                except (TypeError, ValueError):
                    errors.append(
                        f"{coeff_name} must be a number, got {val!r}."
                    )

        # Check Rayleigh damping coefficients
        m_damp = params.get("M_DAMP")
        if m_damp is not None:
            try:
                md = float(m_damp)
                if md < 0:
                    errors.append(
                        f"M_DAMP (mass-proportional damping) must be >= 0, "
                        f"got {md}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"M_DAMP must be a non-negative number, got {m_damp!r}."
                )

        k_damp = params.get("K_DAMP")
        if k_damp is not None:
            try:
                kd = float(k_damp)
                if kd < 0:
                    errors.append(
                        f"K_DAMP (stiffness-proportional damping) must be >= 0, "
                        f"got {kd}."
                    )
                elif kd > 0.01:
                    errors.append(
                        f"K_DAMP = {kd} is unusually large.  High stiffness-"
                        f"proportional damping over-damps high frequencies and "
                        f"makes the problem very stiff.  Typical range: "
                        f"1e-6 to 1e-3."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"K_DAMP must be a non-negative number, got {k_damp!r}."
                )

        # Check Poisson's ratio if provided
        nue = params.get("NUE")
        if nue is not None:
            try:
                nu = float(nue)
                if nu <= 0 or nu >= 0.5:
                    errors.append(
                        f"NUE (Poisson's ratio) must be in (0, 0.5), got {nu}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"NUE must be a number in (0, 0.5), got {nue!r}."
                )

        # Check Young's modulus if provided
        young = params.get("YOUNG")
        if young is not None:
            try:
                e = float(young)
                if e <= 0:
                    errors.append(
                        f"YOUNG (Young's modulus) must be > 0, got {e}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"YOUNG must be a positive number, got {young!r}."
                )

        # Check DYNAMICTYPE
        dyntype = params.get("DYNAMICTYPE")
        # The full STRUCTURAL DYNAMIC/DYNAMICTYPE enum this build
        # accepts, taken from `4C --parameters`. It is 'ExplicitEuler',
        # spelled out: the earlier set here listed 'ExplEuler', so this
        # validator ACCEPTED a value 4C rejects and REJECTED the valid
        # one. Verified by execution 2026-08-03 -- ExplEuler aborts with
        # 'Could not match this input', ExplicitEuler runs.
        valid_types = {"Statics", "GenAlpha", "GenAlphaLieGroup",
                       "OneStepTheta", "ExplicitEuler", "CentrDiff",
                       "AdamsBashforth2", "AdamsBashforth4"}
        if dyntype is not None and dyntype not in valid_types:
            errors.append(
                f"DYNAMICTYPE must be one of {sorted(valid_types)}, "
                f"got {dyntype!r}.  Use 'GenAlpha' for implicit dynamics "
                f"(recommended) or 'ExplicitEuler' for explicit -- note it "
                f"is spelled out; 'ExplEuler' is rejected by 4C."
            )

        # Warn about CFL for explicit
        if dyntype in ("ExplicitEuler", "CentrDiff") and timestep is not None:
            try:
                dt = float(timestep)
                if dt > 1e-6:
                    errors.append(
                        f"{dyntype} with TIMESTEP = {dt}: explicit methods "
                        f"typically require very small time steps to satisfy "
                        f"the CFL condition (dt < h_min / c_wave).  Verify "
                        f"that this time step is small enough for your mesh "
                        f"and material."
                    )
            except (TypeError, ValueError):
                pass  # Already caught above

        return errors
