"""Generator for solid mechanics physics module (linear/nonlinear elasticity, hyperelastic).

Covers quasi-static structural problems solved with DYNAMICTYPE: Statics
in the STRUCTURAL DYNAMIC section of 4C.  Produces validated, working
.4C.yaml templates for 2D and 3D solid mechanics analyses.
"""

from __future__ import annotations

from typing import Any

from .base import BaseGenerator


class SolidMechanicsGenerator(BaseGenerator):
    """Generator for solid mechanics problems in 4C.

    Supports linear elasticity (small deformation), geometrically nonlinear
    elasticity, hyperelastic materials (Neo-Hookean), and elastoplasticity.
    """

    module_key = "solid_mechanics"
    display_name = "Solid Mechanics (Linear / Nonlinear Elasticity)"
    problem_type = "Structure"

    # ── Knowledge ─────────────────────────────────────────────────────

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Quasi-static structural mechanics.\n"
                "  PROBLEMTYPE:      Structure\n"
                "  control section:  STRUCTURAL DYNAMIC "
                "(DYNAMICTYPE: Statics for no inertia)\n"
                "  element section:  STRUCTURE ELEMENTS\n"
                "  3D element line:  <eid> SOLID HEX8 <8 nodes> MAT <m> "
                "KINEM linear|nonlinear\n"
                "  2D element line:  VERSION-DEPENDENT, see the "
                "'2d_element_type' key below - the two spellings share no "
                "keywords\n"
                "  material:         MAT_Struct_StVenantKirchhoff with "
                "YOUNG, NUE, DENS (DENS required even under Statics)\n"
                "  convergence:      TOLDISP (update norm) + TOLRES "
                "(residual norm), both must be met\n"
                "Use 'minimal_working_input_3d' below - it is a complete "
                "deck that runs as written."
            ),
            "2d_element_type": (
                "WHICH element type owns 2D structural cells is "
                "VERSION-DEPENDENT and the two spellings share no keywords, "
                "so they cannot be mixed:\n"
                "  WALL  QUAD4 <n..> MAT m KINEM k EAS e THICK t "
                "STRESS_STRAIN s GP a b\n"
                "  SOLID QUAD4 <n..> MAT m KINEM k THICKNESS t "
                "PLANE_ASSUMPTION p\n"
                "(measured: a 4C before 2026.2.0 has WALL and no "
                "PLANE_ASSUMPTION key, its SOLID owns only "
                "HEX/TET/WEDGE/PYRAMID; 4C 2026.2.0 and 2026.3.0 have no "
                "WALL, and their SOLID takes QUAD4, QUAD8, QUAD9, TRI3 and "
                "TRI6 with THICKNESS and PLANE_ASSUMPTION.)\n"
                "Decide it, do not guess: `4C --parameters` lists, per "
                "element type, the cell types it owns. If SOLID's list is "
                "3D-only (HEX/TET/WEDGE/PYRAMID), 2D belongs to WALL and "
                "'SOLID QUAD4' aborts with \"Element 'SOLID' does not seem "
                "to know cell type 'quad4'.\"; if SOLID lists QUAD4/TRI3 "
                "then 'WALL' aborts with \"Unknown type 'WALL' of finite "
                "element\". Before 4C 2026.2.0 WALL owns 2D and needs all six "
                "of its keys; from 4C 2026.2.0 on the 2D SOLID needs four."
            ),
            "minimal_working_input_3d": """\
# Complete 3D cantilever. The mesh is GENERATED - not one node coordinate.
# Runs as written: exit 0, one static step, VTU written.
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURE DOMAIN:                        # box generator (3D cells only)
  bottom_corner_point: [0.0, 0.0, 0.0]   # REQUIRED
  top_corner_point: [10.0, 1.0, 1.0]     # REQUIRED
  subdivisions: [10, 2, 2]               # REQUIRED
  elements:                              # REQUIRED
    SOLID:
      HEX8:
        MAT: 1
        KINEM: nonlinear
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1.0e-10
  TOLRES: 1.0e-09
  MAXITER: 30
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000.0
      NUE: 0.3            # validated to lie in [-1, 0.5)
      DENS: 1.0           # REQUIRED even for Statics
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t"
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 3
    ONOFF: [0, 0, 1]
    VAL: [0.0, 0.0, -1.0]
    FUNCT: [0, 0, 1]
    TYPE: "Live"
DSURF-NODE TOPOLOGY:      # symbolic faces of the generated box
  - "SIDE structure x- DSURFACE 1"
  - "SIDE structure x+ DSURFACE 2"
IO/RUNTIME VTK OUTPUT:    # OPTIONAL - drop both VTK sections and the deck
  INTERVAL_STEPS: 1       # still runs, it just writes no .vtu. But for .vtu
IO/RUNTIME VTK OUTPUT/STRUCTURE:   # you need BOTH sections AND at least one
  OUTPUT_STRUCTURE: true           # field flag; any one of the three alone
  DISPLACEMENT: true               # writes nothing.
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 1
      QUANTITY: "dispz"
      VALUE: 0.0
      TOLERANCE: 1.0e30   # record mode: abs(diff) prints the true value
""",
            "required_sections": [
                "PROBLEM TYPE",
                "STRUCTURAL DYNAMIC",
                "SOLVER 1",
                "MATERIALS",
                # Plus ONE mesh route - these three are alternatives, not
                # three requirements. Only the middle one needs a file.
                #   inline   : NODE COORDS + STRUCTURE ELEMENTS
                #              + D*-NODE TOPOLOGY
                #   Exodus   : STRUCTURE GEOMETRY (FILE + ELEMENT_BLOCKS)
                #   generated: STRUCTURE DOMAIN (HEX/WEDGE cells only)
                "one of: NODE COORDS + STRUCTURE ELEMENTS | STRUCTURE GEOMETRY | STRUCTURE DOMAIN",
            ],
            "materials": {
                "MAT_Struct_StVenantKirchhoff": {
                    "description": (
                        "St. Venant-Kirchhoff hyperelastic material.  Valid for "
                        "small strains (both linear and nonlinear kinematics).  "
                        "Most common starting-point material for structural analysis."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Young's modulus E",
                            "range": "> 0  (e.g. steel: 210000 MPa, aluminium: 70000 MPa)",
                        },
                        "NUE": {
                            "description": "Poisson's ratio nu",
                            "range": "0 < nu < 0.5  (0.3 typical for metals; approaching 0.5 = incompressible)",
                        },
                        "DENS": {
                            "description": "Mass density (only needed for dynamics or gravity loads)",
                            "range": "> 0  (e.g. steel: 7.85e-9 t/mm^3)",
                        },
                    },
                },
                "MAT_ElastHyper + ELAST_CoupNeoHooke": {
                    "description": (
                        "Compressible Neo-Hookean hyperelastic material for large "
                        "deformations.  Uses a two-material definition: "
                        "MAT_ElastHyper (wrapper with NUMMAT, MATIDS, DENS) "
                        "referencing an ELAST_CoupNeoHooke sub-material with "
                        "YOUNG and NUE.  Always use with KINEM: nonlinear."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Young's modulus E of Neo-Hookean model",
                            "range": "> 0",
                        },
                        "NUE": {
                            "description": "Poisson's ratio nu",
                            "range": "0 < nu < 0.5",
                        },
                        "DENS": {
                            "description": "Mass density (in MAT_ElastHyper)",
                            "range": "> 0",
                        },
                        "NUMMAT": {
                            "description": "Number of sub-materials (always 1 for simple Neo-Hookean)",
                            "range": "1",
                        },
                        "MATIDS": {
                            "description": "List of sub-material IDs referencing ELAST_CoupNeoHooke",
                            "range": "[<id>]",
                        },
                        "POLYCONVEX": {
                            "description": (
                                "MAT_ElastHyper wrapper flag: enable a runtime "
                                "polyconvexity check on the strain-energy "
                                "function (0 = off, 1 = on)."
                            ),
                            "range": "0 | 1",
                        },
                    },
                },
                "MAT_Struct_PlasticNlnLogNeoHooke": {
                    "description": (
                        "Finite-strain J2 elastoplasticity with logarithmic "
                        "Neo-Hookean elastic response.  Supports isotropic "
                        "exponential saturation hardening and optional viscoplasticity."
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
                            "description": "Mass density",
                            "range": "> 0",
                        },
                        "YIELD": {
                            "description": "Initial yield stress sigma_y0",
                            "range": "> 0",
                        },
                        "SATHARDENING": {
                            "description": "Saturation hardening stress (sigma_y_inf - sigma_y0)",
                            "range": ">= 0",
                        },
                        "HARDEXPO": {
                            "description": "Hardening exponent delta (controls rate of saturation)",
                            "range": "> 0",
                        },
                        "ISOHARD": {
                            "description": "Linear isotropic hardening modulus added on top of the Voce saturation law.",
                            "range": ">= 0",
                        },
                        "VISC": {
                            "description": (
                                "Perzyna-type viscoplastic viscosity eta.  "
                                "Set 0 for rate-independent plasticity."
                            ),
                            "range": ">= 0",
                        },
                        "RATE_DEPENDENCY": {
                            "description": (
                                "Perzyna rate exponent n.  Ignored when VISC = 0."
                            ),
                            "range": "> 0",
                        },
                        "TOL": {
                            "description": "Local return-mapping convergence tolerance.",
                            "range": "typical 1e-10",
                        },
                        "HARDENING_FUNC": {
                            "description": (
                                "ID of a user-defined hardening function in the "
                                "FUNCT section.  Set 0 to use the analytic "
                                "Voce+linear law parameterised by YIELD / "
                                "SATHARDENING / HARDEXPO / ISOHARD."
                            ),
                            "range": ">= 0",
                        },
                    },
                },
            },
            "time_integration": {
                "DYNAMICTYPE": (
                    "'Statics' for quasi-static analysis (load is applied "
                    "incrementally via time-stepping without inertia).  "
                    "For dynamic problems use the structural_dynamics generator."
                ),
                "KINEM": (
                    "'linear' -- small-deformation assumption (geometrically linear).  "
                    "'nonlinear' -- large deformation / finite strain.  "
                    "CRITICAL: KINEM must be consistent with the material model.  "
                    "St. Venant-Kirchhoff works with both; Neo-Hookean and "
                    "plasticity models REQUIRE nonlinear."
                ),
                "MAXITER": (
                    "Maximum Newton-Raphson iterations per load step.  "
                    "NEVER set MAXITER: 1, not even for a linear problem: "
                    "exhausting MAXITER is an ABORT, not an early exit.  "
                    "Typical: 20--50; there is no cost to a generous cap "
                    "because Newton stops at the tolerance, not at the cap."
                ),
                "TOLDISP": "Displacement convergence tolerance (typical: 1e-6 to 1e-10).",
                "TOLRES": "Residual force convergence tolerance (typical: 1e-6 to 1e-10).",
            },
            "solver": {
                "small_problems": {
                    "SOLVER": "UMFPACK",
                    "description": (
                        "Direct solver, very robust.  Best for problems up to "
                        "~50k DOFs or for debugging."
                    ),
                },
                "large_problems": {
                    "SOLVER": "Belos",
                    "AZPREC": "MueLu",
                    "SOLVER_XML_FILE": "iterative_gmres_template.xml",
                    "preconditioner_file_by_release": {
                        "4C 2026.2.0 and later": {"PRECONDITIONER_XML_FILE": "elasticity_template.xml"},
                        "4C before 2026.2.0": {"MUELU_XML_FILE": "elasticity_template.xml"},
                    },
                    "description": (
                        "Iterative Krylov solver (GMRES) with MueLu AMG "
                        "preconditioner.  Scalable to millions of DOFs.  "
                        "Requires XML configuration files for solver and "
                        "preconditioner.  The preconditioner file key is the "
                        "one of the release that runs the deck: 4C 2026.2.0 "
                        "and later accept only PRECONDITIONER_XML_FILE, a 4C "
                        "before 2026.2.0 only MUELU_XML_FILE."
                    ),
                },
            },
            "plasticity_models": {
                "MAT_Struct_PlasticLinElast": {
                    "description": "Small-strain J2 (von Mises) with linear isotropic+kinematic hardening",
                    "parameters": "YOUNG, NUE, DENS, ISOHARD, KINHARD, YIELD, TOL",
                    "kinematics": "linear only",
                },
                "MAT_Struct_DruckerPrager": {
                    "description": (
                        "Small-strain Drucker-Prager (pressure-dependent, smooth cone). "
                        "C is the cohesion parameter; ETA/XI/ETABAR are pre-computed "
                        "from friction phi and dilatancy psi -- see eta_xi_formulas."
                    ),
                    "parameters": "YOUNG, NUE, DENS, ISOHARD, TOL, C, ETA, XI, ETABAR, TANG, MAXITER",
                    "kinematics": "linear only",
                    "eta_xi_formulas": (
                        "For outer cone (circumscribed, matches MC at compression meridian): "
                        "ETA = 6*sin(phi)/(3-sin(phi)), XI = 6*cos(phi)/(3-sin(phi)), "
                        "ETABAR = 6*sin(psi)/(3-sin(psi)). "
                        "For inner cone (inscribed, matches MC at extension meridian): "
                        "ETA = 6*sin(phi)/(3+sin(phi)), XI = 6*cos(phi)/(3+sin(phi)). "
                        "For middle cone (Lode-angle independent best fit): "
                        "ETA = 3*tan(phi)/sqrt(9+12*tan^2(phi)), XI = 3/sqrt(9+12*tan^2(phi))."
                    ),
                },
                "MAT_PlasticElastHyper": {
                    "description": (
                        "Finite-strain J2/Hill with nonlinear isotropic+kinematic hardening, "
                        "Perzyna viscoplasticity, optional thermal softening.  Wraps an "
                        "ELAST_* hyperelastic sub-material; elastic moduli (YOUNG/NUE) "
                        "live in that sub-material, not on this wrapper."
                    ),
                    "parameters": (
                        "INITYIELD, ISOHARD, KINHARD, EXPISOHARD, INFYIELD, VISC, "
                        "RATE_DEPENDENCY, VISC_SOFT, YIELDSOFT, HARDSOFT, "
                        "CTE, INITTEMP, TAYLOR_QUINNEY, PL_SPIN_CHI"
                    ),
                    "kinematics": "nonlinear",
                    "features": "Hill anisotropy, Perzyna viscoplasticity, thermal softening (TSI)",
                },
                "MAT_Struct_PlasticGTN": {
                    "description": (
                        "Gurson-Tvergaard-Needleman ductile damage with void growth, "
                        "nucleation and coalescence (KINEM: linear only in current 4C release)."
                    ),
                    "parameters": (
                        "YOUNG, NUE, DENS, YIELD, ISOHARD, HARDENING_FUNC, "
                        "F0, FN, SN, EN, FC, KAPPA, EF, K1, K2, K3, MAXITER, TOL"
                    ),
                    "kinematics": "linear only",
                },
                "MAT_crystal_plasticity": {
                    "description": (
                        "Single-crystal plasticity with dislocation-density based hardening and "
                        "optional deformation twinning.  Lattice families accepted by the runtime: "
                        "FCC, BCC, D019, L10 (LAT key, default FCC).  The input parser's own "
                        "description string advertises HCP as well, but the constructor sanity "
                        "check rejects 'HCP' with FOUR_C_THROW — use D019 for hexagonal lattices."
                    ),
                    "parameters": (
                        # elastic + Newton tolerance
                        "TOL, YOUNG, NUE, DENS, "
                        # crystal lattice
                        "LAT, CTOA, ABASE, "
                        # slip-system definition
                        "NUMSLIPSYS, NUMSLIPSETS, SLIPSETMEMBERS, SLIPRATEEXP, GAMMADOTSLIPREF, "
                        "DISDENSINIT, DISGENCOEFF, DISDYNRECCOEFF, TAUY0, MFPSLIP, SLIPHPCOEFF, "
                        "SLIPBYTWIN, "
                        # twin-system definition (all optional; NUMTWINSYS/SETS default 0)
                        "NUMTWINSYS, NUMTWINSETS, TWINSETMEMBERS, TWINRATEEXP, GAMMADOTTWINREF, "
                        "TAUT0, MFPTWIN, TWINHPCOEFF, TWINBYSLIP, TWINBYTWIN"
                    ),
                    "kinematics": "nonlinear",
                    "features": (
                        "Dislocation-density evolution (generation + dynamic recovery), "
                        "Hall-Petch via MFPSLIP/MFPTWIN with HP coefficients, "
                        "slip-twin and twin-twin coupling, multiple lattice types"
                    ),
                    "pitfalls": (
                        "Vector-size rule depends on the parameter (two distinct sizing axes):\n"
                        "  - one entry per physical *system* (size = NUMSLIPSYS or NUMTWINSYS): "
                        "SLIPSETMEMBERS, TWINSETMEMBERS — these are 1-based indices into the "
                        "set table (range 1..NUMSLIPSETS or 1..NUMTWINSETS) saying which set "
                        "each individual slip/twin system belongs to.\n"
                        "  - one entry per *set* (size = NUMSLIPSETS or NUMTWINSETS): every "
                        "other slip/twin vector (SLIPRATEEXP, GAMMADOTSLIPREF, DISDENSINIT, "
                        "DISGENCOEFF, DISDYNRECCOEFF, TAUY0, MFPSLIP, SLIPHPCOEFF, SLIPBYTWIN "
                        "and their TWIN-side counterparts TWINRATEEXP, GAMMADOTTWINREF, TAUT0, "
                        "MFPTWIN, TWINHPCOEFF, TWINBYSLIP, TWINBYTWIN).\n"
                        "Mixing these two sizes is a common error — NUMSLIPSYS usually exceeds "
                        "NUMSLIPSETS (e.g. FCC has 12 systems often in 1 set).  Twinning is "
                        "optional: leave NUMTWINSYS=NUMTWINSETS=0 and omit every TWIN-side "
                        "vector (they all have parser defaults) for pure slip plasticity."
                    ),
                },
            },
            "plasticity_pitfalls": [
                "Drucker-Prager uses pre-computed constants (ETA, XI, ETABAR) that map from friction/dilatancy "
                "angles. ETA = 6*sin(phi)/(3-sin(phi)), XI = 6*cos(phi)/(3-sin(phi)), ETABAR = 6*sin(psi)/(3-sin(psi)) "
                "for the circumscribed outer cone. Getting these wrong silently gives wrong yield stress.",
                "PlasticLinElast and DruckerPrager are small-strain only (KINEM: linear). "
                "PlasticElastHyper requires KINEM: nonlinear. Mismatch produces FOUR_C_THROW.",
                "For quasi-static plasticity, use many small load steps (NUMSTEP >= 100). "
                "Too few steps causes the Newton iteration to diverge when crossing the yield surface.",
                "Accumulated plastic strain output is available via 'accumulated_plastic_strain' "
                "in IO/RUNTIME VTK OUTPUT/STRUCTURE with STRESS_STRAIN: true.",
                "TANG parameter controls the material tangent: 'consistent' uses the algorithmic "
                "elastoplastic tangent (required for global Newton convergence in load-controlled problems). "
                "'elastic' uses the elastic tangent as a fallback (poor convergence near yield, but robust "
                "for debugging). Always use 'consistent' for production plasticity simulations.",
                "TESTING PITFALL: Fully displacement-controlled single-element tests (all DOFs prescribed "
                "via Dirichlet BCs) bypass the global Newton iteration — the return mapping is called but "
                "its tangent is irrelevant. For benchmarking plasticity, use Neumann BCs on at least one "
                "face (e.g., confining pressure) so the tangent is actually exercised.",
                "NEUMANN SIGN CONVENTION for structural problems: negative values in the normal direction "
                "produce compressive traction. For a face with outward normal in +x, setting "
                "VAL: [-100e3, 0, 0, 0, 0, 0] applies 100 kPa compression on that face.",
            ],
            "pitfalls": [
                (
                    "[Numerical] KINEM must match the physical assumption.  "
                    "Using 'KINEM: linear' with a hyperelastic material "
                    "(Neo-Hookean, Mooney-Rivlin) is WRONG, and it is "
                    "COMPLETELY SILENT: the deck parses, converges, exits 0 "
                    "and returns a substantially larger deflection than the "
                    "same deck under KINEM: nonlinear. 4C emits no warning and "
                    "never prints the string 'kinem' anywhere in the log, so "
                    "there is nothing to grep for. Signal: the only way to see "
                    "it is to run both and compare, or to notice that the "
                    "response scales far more nearly with load than a "
                    "hyperelastic material at that strain should. Set KINEM "
                    "deliberately on every element line. (Audit 2026-06-02; "
                    "the silence confirmed by execution 2026-08-06.)"
                ),
                (
                    "[Numerical] NEVER set MAXITER = 1, not even for a linear "
                    "problem. Exhausting MAXITER is an ABORT, not an early "
                    "exit: the iteration counter reaches 1 before the "
                    "convergence test is credited, so MAXITER: 1 kills a "
                    "perfectly converged linear deck. Leave the default or "
                    "set 10-30. Signal: NOX's final status block reports the "
                    "iteration test as Failed with the number of iterations "
                    "1 shown as less than the required 1 -- that block is "
                    "printed by Trilinos NOX, not by 4C, so it is in no 4C "
                    "source file -- then a PROC 0 ERROR from "
                    "solver_nonlin_nox/4C_solver_nonlin_nox_problem.cpp and "
                    "exit 1. (An earlier version of this entry RECOMMENDED "
                    "MAXITER = 1 here. Falsified by execution 2026-08-03 on "
                    "the deployed 4C: the served minimal_working_input_3d deck "
                    "with MAXITER: 1 aborts with exit 1 under BOTH "
                    "KINEM linear and KINEM nonlinear, while the same deck "
                    "with MAXITER: 30 finishes normally with exit 0.)"
                ),
                (
                    "[Input] DENS is a REQUIRED key of "
                    "MAT_Struct_StVenantKirchhoff and cannot be omitted — an "
                    "earlier version of this entry said it could be dropped "
                    "for quasi-static problems, and that is wrong: leaving the "
                    "line out aborts at parse with 'Could not match this "
                    "input' from core/io/src/4C_io_input_spec_builders.cpp. "
                    "What is true is that its VALUE is inert under Statics "
                    "without gravity: DENS 0.0 and DENS 1.0 give the same "
                    "static answer to the last bit. Under a transient scheme "
                    "the same DENS 0.0 is fatal. Signal: a transient run with "
                    "DYNAMICTYPE: GenAlpha and DENS 0 aborts with "
                    "\"You are about to invert a singular matrix!\" from "
                    "structure_new/4C_structure_new_integrator.cpp and exit 1 "
                    "-- the message names the linear algebra, NOT the density, "
                    "so check DENS first when you see it. "
                    "(An earlier version of this entry quoted "
                    "`mass matrix needs DENS in material X`. That string does "
                    "not occur anywhere in the 4C binary; grepping for it "
                    "finds nothing. Corrected by execution 2026-08-03: the "
                    "served 3D deck switched to GenAlpha with DENS: 0.0 "
                    "produced the singular-matrix message above.)"
                ),
                (
                    "[Numerical] HEX8 elements lock in two ways — BENDING and "
                    "VOLUMETRIC — and the cures are NOT symmetric. On the "
                    "SOLID element line, TECH: eas_full (or eas_mild) "
                    "addresses BOTH, while TECH: fbar addresses only the "
                    "volumetric one. So fbar on a slender bending problem "
                    "leaves it far too stiff, but eas_full on a "
                    "nearly-incompressible problem DOES cure it: EAS enhances "
                    "the strain field, volumetric modes included. Measured on "
                    "a 10x1x1 HEX8 cantilever, L/h = 20, tip traction, against "
                    "the -1.586797e-01 reference: at NUE 0.499 with nonlinear "
                    "kinematics, plain gives -2.314e-02 (15% of it), fbar "
                    "-9.133e-02 (58%), eas_full -1.5658e-01 (99%); at NUE "
                    "0.4999 with KINEM linear, plain gives -9.321e-03 (6%) "
                    "while eas_full gives -1.5654e-01 (99%) and fbar recovers "
                    "the reference outright. Pick eas_* when bending is "
                    "present or when you are unsure; pick fbar when the "
                    "problem is purely volumetric. Higher-order cells (HEX27, "
                    "TET10) are the other route. TECH is an ENUM — "
                    "none, fbar, eas_mild, eas_full, shell_ans, shell_eas, "
                    "shell_eas_ans — so a plausible abbreviation like 'eas' is "
                    "rejected with \"Could not parse value 'eas' as an enum "
                    "constant of type 'ElementTechnology'.\" rather than "
                    "silently ignored. Signal: a cantilever or plate-bending "
                    "deck (PROBLEMTYPE: Structure, DYNAMICTYPE: Statics) whose "
                    "deflection is a small fraction of the analytical value, "
                    "or a response that stiffens sharply as NUE approaches "
                    "0.5. (Audit 2026-06-02. This entry previously said "
                    "eas_* 'leaves the volumetric part untouched' and carried "
                    "a 'confirmed by execution 2026-08-06' stamp; the fixture "
                    "behind that stamp has six arms and none of them ran EAS "
                    "at high NUE, so the claim's second half had never been "
                    "executed. All six numbers above measured 2026-08-30 on "
                    "4C 2026.2.0-dev.)"
                ),
                (
                    "[API] WHICH element type owns 2D structural "
                    "cells is VERSION-DEPENDENT, and the two "
                    "spellings share no keywords, so you cannot "
                    "hedge by writing both:\n"
                    "  WALL  QUAD4 <n..> MAT m KINEM k EAS e "
                    "THICK t STRESS_STRAIN s GP a b\n"
                    "  SOLID QUAD4 <n..> MAT m KINEM k "
                    "THICKNESS t PLANE_ASSUMPTION p\n"
                    "Determine which one the installed build "
                    "registers BEFORE writing anything: "
                    "`4C --parameters` lists, per element type, "
                    "the cell types it owns. If SOLID's list is "
                    "3D-only (HEX/TET/WEDGE/PYRAMID), 2D is "
                    "WALL's; if SOLID lists QUAD4/TRI3, 2D is "
                    "SOLID's. 3D is always SOLID and takes no "
                    "thickness or plane-assumption key at all. "
                    "Measured: on a 4C before 2026.2.0 SOLID owns "
                    "only HEX/TET/WEDGE/PYRAMID, so 2D is WALL's and "
                    "no plane-assumption key exists; 4C 2026.2.0 and "
                    "2026.3.0 have no WALL and give SOLID the 2D cells. "
                    "Signal: the "
                    "element type this build does not register "
                    "raises \"Unknown type 'WALL' of finite "
                    "element\" from "
                    "core/comm/src/4C_comm_parobjectfactory.cpp "
                    "(note the QUOTES around the type name — the "
                    "binary's template is \"Unknown type '{}' of "
                    "finite element\", so an unquoted grep finds "
                    "nothing), while the right element type with "
                    "the wrong cell type raises \"Element 'SOLID' "
                    "does not seem to know cell type 'quad4'.\" "
                    "with the cell type echoed in lowercase. In "
                    "ELEMENT_BLOCKS with Exodus meshes the "
                    "sub-key is whichever element type won. "
                    "(Verified by execution 2026-08-03: on 4C "
                    "2026.2.0-dev WALL owns 2D, and the Tier-2 "
                    "fixture structural_2d_solid_quad4_not_wall "
                    "probes both spellings rather than "
                    "hard-coding either.)"
                ),
                (
                    "[Input] A Neumann condition's NUMDOF must match the "
                    "number of dofs the ELEMENT carries, and for 3D continuum "
                    "SOLID that is THREE, not six. An earlier version of this "
                    "entry told you to always write NUMDOF: 6 and warned that "
                    "NUMDOF: 3 'silently drops the moment components'. Nothing "
                    "is dropped silently: on SOLID, NUMDOF: 3 runs; NUMDOF: 6 "
                    "with the last three slots zeroed also runs and gives the "
                    "identical answer, so the extra slots buy nothing; and "
                    "NUMDOF: 6 with a moment slot switched ON aborts. Signal: "
                    "'Number of Dimensions in Neumann_Evaluation is 3. Further "
                    "DoFs are not considered.' from "
                    "solid_3D_ele/4C_solid_3D_ele_surface_evaluate.cpp, exit "
                    "1. Moment loads need elements with rotational dofs — "
                    "SHELL7P conditions use NUMDOF: 6, BEAM3R ones NUMDOF: 9. "
                    "(Falsified and corrected by execution 2026-08-06.)"
                ),
                (
                    "[Input] INT_STRATEGY: Standard is the default; leave "
                    "it alone. Setting 'Old' does not merely relax a "
                    "compatibility flag, it swaps the entire nonlinear solver "
                    "stack for the legacy integrator, and on a plain "
                    "structural deck it does so without changing the answer — "
                    "same result to the last bit, exit 0, no warning — so "
                    "nothing in the numbers tells you. Signal: the log shape "
                    "changes completely. Standard prints NOX blocks headed "
                    "'-- Status Test Results --' and ends each step with "
                    "'nlniter N'; Old prints none of them and ends with "
                    "'numiter N', preceded by the legacy banner \"Structural "
                    "predictor for field 'structure' ConstDis yields absolute "
                    "res-norm\". Everything NOX-side (status-test XML, line "
                    "search, custom convergence combinations) is silently "
                    "unavailable. Coupled problems are less forgiving: an SSI "
                    "run asserts with 'Only the new solid time integration is "
                    "supported for SSI problems. Set `INT_STRATEGY` to "
                    "`Standard`!' from ssi/4C_ssi_dyn.cpp. (Audit 2026-06-02; "
                    "mechanism and signals established by execution "
                    "2026-08-06.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "cantilever_2d",
                    "description": (
                        "2D cantilever beam under tip load.  Fixed left edge, "
                        "point or line load on right edge.  Uses QUAD4 cells of "
                        "whichever 2D element type this build registers (see the "
                        "'2d_element_type' key -- it is version-dependent) with "
                        "plane_strain, MAT_Struct_StVenantKirchhoff, "
                        "KINEM: linear, MAXITER: 20."
                    ),
                    "template_variant": "linear_2d",
                },
                {
                    "name": "compression_3d",
                    "description": (
                        "3D block under uniaxial compression.  Fixed bottom face, "
                        "prescribed displacement on top face.  Uses SOLID HEX8 "
                        "elements with Neo-Hookean material, KINEM: nonlinear."
                    ),
                    "template_variant": "nonlinear_3d",
                },
            ],
        }

    # ── Templates ─────────────────────────────────────────────────────

    _TEMPLATES: dict[str, str] = {
        "linear_2d": """\
# FORMAT TEMPLATE — 2D linear elasticity (plane strain)
# Written in the input grammar of 4C before 2026.2.0 (the WALL element); from
# 4C 2026.2.0 on a 2-D solid is SOLID with THICKNESS and PLANE_ASSUMPTION. The
# decks openPASO runs are served in the grammar of the 4C it finds.
# All numerical values are placeholders — determine from your specific problem.
# Check 4C test files via examples(keyword, solver='fourc', action='search') for reference setups.
TITLE:
  - "Linear elastic 2D — plane strain"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
IO:
  STRUCT_STRESS: "Cauchy"
  STRUCT_STRAIN: "GL"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
  STRESS_STRAIN: true
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "Statics"
  TIMESTEP: <timestep>
  NUMSTEP: <number_of_steps>
  MAXTIME: <end_time>
  TOLDISP: <displacement_tolerance>
  TOLRES: <residual_tolerance>
  MAXITER: <max_newton_iterations — use 1 for linear, 10+ for nonlinear>
  LINEAR_SOLVER: 1
  PREDICT: TangDis
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct_solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: <Young_modulus>
      NUE: <Poisson_ratio>
      DENS: <density>
DESIGN LINE DIRICH CONDITIONS:
  - E: <boundary_id>
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0.0, 0.0]
    FUNCT: [0, 0]
DESIGN LINE NEUMANN CONDITIONS:
  - E: <boundary_id>
    NUMDOF: 6
    ONOFF: [<active_dofs>]
    VAL: [<force_values>]
    FUNCT: [0, 0, 0, 0, 0, 0]
STRUCTURE GEOMETRY:
  ELEMENT_BLOCKS:
    - ID: 1
      WALL:
        QUAD4:
          MAT: 1
          KINEM: linear
          EAS: none
          THICK: <thickness>
          STRESS_STRAIN: plane_strain
          GP: [2, 2]
  FILE: <mesh_file.e>
  SHOW_INFO: detailed_summary
""",
        "nonlinear_3d": """\
TITLE:
  - "Nonlinear 3D block compression -- Neo-Hookean, large deformation"
PROBLEM SIZE:
  DIM: 3
# FORMAT TEMPLATE — 3D nonlinear elasticity (large deformation)
# All numerical values are placeholders.
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
IO:
  STRUCT_STRESS: "Cauchy"
  STRUCT_STRAIN: "GL"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
  STRESS_STRAIN: true
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "Statics"
  TIMESTEP: <load_step_size>
  NUMSTEP: <number_of_load_steps>
  MAXTIME: <total_load_parameter>
  TOLDISP: <displacement_tolerance>
  TOLRES: <residual_tolerance>
  MAXITER: <max_newton_iterations>
  PREDICT: TangDis
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct_solver"
MATERIALS:
  - MAT: 1
    MAT_ElastHyper:
      NUMMAT: 1
      MATIDS: [10]
      DENS: <density>
  - MAT: 10
    ELAST_CoupNeoHooke:
      YOUNG: <Young_modulus>
      NUE: <Poisson_ratio>
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t"
DESIGN SURF DIRICH CONDITIONS:
  - E: <fixed_boundary_id>
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  - E: <loaded_boundary_id>
    NUMDOF: 3
    ONOFF: [<active_dofs>]
    VAL: [<prescribed_displacement>]
    FUNCT: [0, 1, 0]
STRUCTURE GEOMETRY:
  ELEMENT_BLOCKS:
    - ID: 1
      SOLID:
        HEX8:
          MAT: 1
          KINEM: nonlinear
  FILE: <mesh_file.e>
  SHOW_INFO: detailed_summary
""",
    }

    def get_template(self, variant: str = "default") -> str:
        if variant == "default":
            variant = "linear_2d"
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
                "name": "linear_2d",
                "description": (
                    "Linear elastic 2D cantilever with plane strain, SOLID QUAD4 "
                    "elements, St. Venant-Kirchhoff material, tip load."
                ),
            },
            {
                "name": "nonlinear_3d",
                "description": (
                    "Geometrically nonlinear 3D block compression with "
                    "Neo-Hookean hyperelastic material, HEX8 elements, "
                    "prescribed displacement loading."
                ),
            },
        ]

    # ── Validation ────────────────────────────────────────────────────

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        errors: list[str] = []

        # Check Poisson's ratio
        nue = params.get("NUE")
        if nue is not None:
            try:
                nu = float(nue)
                if nu <= 0:
                    errors.append(
                        f"NUE (Poisson's ratio) must be > 0, got {nu}.  "
                        f"A non-positive Poisson's ratio is unphysical for "
                        f"standard structural materials."
                    )
                elif nu >= 0.5:
                    errors.append(
                        f"NUE (Poisson's ratio) must be < 0.5, got {nu}.  "
                        f"nu = 0.5 means perfectly incompressible, which "
                        f"causes a singular stiffness matrix with standard "
                        f"displacement elements.  Use nu <= 0.499."
                    )
                elif nu > 0.49:
                    errors.append(
                        f"NUE = {nu} is very close to 0.5 (incompressible "
                        f"limit).  Standard HEX8/QUAD4 elements will exhibit "
                        f"severe volumetric locking.  Consider using "
                        f"TECH: fbar or higher-order elements."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"NUE must be a number in (0, 0.5), got {nue!r}."
                )

        # Check Young's modulus
        young = params.get("YOUNG")
        if young is not None:
            try:
                e = float(young)
                if e <= 0:
                    errors.append(
                        f"YOUNG (Young's modulus) must be > 0, got {e}.  "
                        f"A non-positive modulus is unphysical."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"YOUNG must be a positive number, got {young!r}."
                )

        # Check density (warn if zero in dynamics context)
        dens = params.get("DENS")
        dynamictype = params.get("DYNAMICTYPE", "Statics")
        if dens is not None:
            try:
                d = float(dens)
                if d < 0:
                    errors.append(
                        f"DENS (density) must be >= 0, got {d}."
                    )
                if d == 0 and dynamictype != "Statics":
                    errors.append(
                        f"DENS = 0 with DYNAMICTYPE = {dynamictype!r}: "
                        f"zero density means zero mass matrix.  Dynamics "
                        f"requires DENS > 0."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"DENS must be a non-negative number, got {dens!r}."
                )

        # Check KINEM vs material consistency
        kinem = params.get("KINEM")
        material = params.get("material_type", "")
        if kinem == "linear" and material in (
            "MAT_ElastHyper", "MAT_Struct_PlasticNlnLogNeoHooke"
        ):
            errors.append(
                f"KINEM: linear with {material} is inconsistent.  "
                f"Hyperelastic and plasticity materials require "
                f"KINEM: nonlinear."
            )

        # Check yield stress for plastic material
        yield_stress = params.get("YIELD")
        if yield_stress is not None:
            try:
                ys = float(yield_stress)
                if ys <= 0:
                    errors.append(
                        f"YIELD (initial yield stress) must be > 0, got {ys}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"YIELD must be a positive number, got {yield_stress!r}."
                )

        # Check SATHARDENING
        sathard = params.get("SATHARDENING")
        if sathard is not None:
            try:
                sh = float(sathard)
                if sh < 0:
                    errors.append(
                        f"SATHARDENING must be >= 0, got {sh}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"SATHARDENING must be a non-negative number, "
                    f"got {sathard!r}."
                )

        # Check HARDEXPO
        hardexpo = params.get("HARDEXPO")
        if hardexpo is not None:
            try:
                he = float(hardexpo)
                if he <= 0:
                    errors.append(
                        f"HARDEXPO (hardening exponent) must be > 0, got {he}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"HARDEXPO must be a positive number, got {hardexpo!r}."
                )

        return errors
