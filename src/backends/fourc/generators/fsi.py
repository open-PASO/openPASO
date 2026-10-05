"""Fluid-Structure Interaction (FSI) generator for 4C.

Covers monolithic and partitioned FSI coupling of an incompressible
Navier-Stokes fluid with a hyperelastic or St. Venant-Kirchhoff structure.
Includes ALE mesh motion handling and CLONING MATERIAL MAP.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class FSIGenerator(BaseGenerator):
    """Generator for Fluid-Structure Interaction problems in 4C."""

    module_key = "fsi"
    display_name = "Fluid-Structure Interaction (FSI)"
    problem_type = "Fluid_Structure_Interaction"

    # ── Knowledge ─────────────────────────────────────────────────────

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Monolithic or partitioned coupling of incompressible "
                "Navier-Stokes fluid with geometrically nonlinear structures.  "
                "The fluid domain moves with the structure via ALE (Arbitrary "
                "Lagrangian-Eulerian) mesh motion.  This is the most complex "
                "problem type in 4C, requiring coordinated setup of three "
                "fields (structure, fluid, ALE) plus coupling conditions."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "STRUCTURAL DYNAMIC",
                "STRUCTURAL DYNAMIC/GENALPHA",
                "FLUID DYNAMIC",
                "ALE DYNAMIC",
                "FSI DYNAMIC",
                "FSI DYNAMIC/MONOLITHIC SOLVER",
                "SOLVER 1",
                "SOLVER 2",
                "SOLVER 3",
                "MATERIALS",
                "STRUCTURE GEOMETRY",
                "FLUID GEOMETRY",
                "CLONING MATERIAL MAP",
                "DESIGN FSI COUPLING LINE CONDITIONS",  # 2-D
                # or "DESIGN FSI COUPLING SURF CONDITIONS" for 3-D
            ],
            "optional_sections": [
                "FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION",
                "FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES",
                "FSI DYNAMIC/PARTITIONED SOLVER",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/STRUCTURE",
                "IO/RUNTIME VTK OUTPUT/FLUID",
            ],
            "materials": {
                "MAT_fluid": {
                    "description": (
                        "Newtonian fluid for the fluid field."
                    ),
                    "parameters": {
                        "DYNVISCOSITY": {
                            "description": "Dynamic viscosity [Pa s]",
                            "range": "> 0",
                        },
                        "DENSITY": {
                            "description": "Fluid density [kg/m^3]",
                            "range": "> 0",
                        },
                    },
                },
                "MAT_ElastHyper (Neo-Hooke)": {
                    "description": (
                        "Hyperelastic material for the structure.  Uses a "
                        "nested ELAST_CoupNeoHooke sub-material."
                    ),
                    "parameters": {
                        "NUMMAT": {"description": "Number of sub-materials", "range": "1"},
                        "MATIDS": {"description": "List of sub-material IDs", "range": ""},
                        "DENS": {"description": "Structural density [kg/m^3]", "range": "> 0"},
                    },
                },
                "MAT_Struct_StVenantKirchhoff": {
                    "description": (
                        "Linear elastic structural material (small-strain "
                        "approximation, but used with nonlinear kinematics)."
                    ),
                    "parameters": {
                        "YOUNG": {"description": "Young's modulus [Pa]", "range": "> 0"},
                        "NUE": {"description": "Poisson's ratio", "range": "[0, 0.5)"},
                        "DENS": {"description": "Structural density [kg/m^3]", "range": "> 0"},
                    },
                },
                "ALE material (clone)": {
                    "description": (
                        "ALE mesh motion material.  Cloned from the fluid "
                        "material via CLONING MATERIAL MAP.  Typically use "
                        "MAT_Struct_StVenantKirchhoff with YOUNG=1, NUE=0."
                    ),
                },
            },
            "solver": {
                "ALE_solver": {
                    "type": "UMFPACK (direct)",
                    "notes": "ALE system is small relative to fluid/structure.",
                },
                "Fluid_solver": {
                    "type": "Belos (iterative) or UMFPACK for small problems",
                    "notes": "AZTOL ~1e-12 for FSI accuracy.",
                },
                "Structure_solver": {
                    "type": "UMFPACK (direct) for small problems",
                },
                "Monolithic_FSI_solver": {
                    "type": "Belos with MueLu block preconditioner",
                    "notes": (
                        "Required for iter_mortar_monolithicfluidsplit. "
                        "Set LINEARBLOCKSOLVER: LinalgSolver. "
                        "Alternatively use UMFPACK for small 2-D demos."
                    ),
                },
            },
            "coupling_algorithms": {
                "recommended": "iter_mortar_monolithicfluidsplit",
                "alternatives": [
                    "iter_monolithicfluidsplit",
                    "iter_monolithicstructuresplit",
                    "iter_mortar_monolithicstructuresplit",
                    "iter_stagg_AITKEN_rel_param",
                    "iter_stagg_fixed_rel_param",
                ],
                "not_valid": (
                    "iter_stagg_AITKEN_rel_force and iter_stagg_fixed_rel_force "
                    "do NOT exist; the staggered schemes are named ..._rel_param."
                ),
                "key_settings": {
                    "COUPALGO": "Coupling algorithm selector in FSI DYNAMIC",
                    "SHAPEDERIVATIVES": "Must be true for monolithic FSI",
                    "SECONDORDER": "Enable second-order time integration coupling",
                },
            },
            "ale_settings": {
                "ALE_TYPE": {
                    "springs_spatial": "Spring-based ALE (spatial formulation, recommended for 2-D)",
                    "springs_material": "Spring-based ALE (material formulation, good for 3-D)",
                },
                "important": (
                    "ALE Dirichlet BCs must be set on all outer fluid "
                    "boundaries (except the FSI interface) to keep the "
                    "ALE mesh fixed there."
                ),
            },
            "cloning_material_map": {
                "purpose": (
                    "Maps the fluid material to an ALE field material.  "
                    "4C internally clones the fluid mesh to create the ALE "
                    "discretisation; this map tells it which material to use."
                ),
                "format": (
                    "SRC_FIELD: fluid, SRC_MAT: <fluid_mat_id>, "
                    "TAR_FIELD: ale, TAR_MAT: <ale_mat_id>"
                ),
            },
            "pitfalls": [
            "[Output] In a 2-D FSI (ALE fluid) deck, requesting PRESSURE under IO/RUNTIME VTK OUTPUT/FLUID "
            "writes NaN at every node and every step. Signal: the 'pressure' array in fluid-*.vtu is all "
            "NaN while velocity and displacement are finite; the run still ends with 'processor 0 "
            "finished normally'. The writer sizes its dof context for 3-D (ux, uy, uz, p) and labels "
            "dof index 3 as pressure; a 2-D fluid node has three dofs (ux, uy, p). Do not request PRESSURE "
            "in 2-D -- none of 4C's own 2-D FSI test decks do -- and read the pressure as the third "
            "component of the 'velocity' array, or output it through the legacy IO result writer. "
            "Measured on 4C 2026-09-23; an upstream defect, not a deck error.",
                # Every Signal: below was produced by running
                #   LD_LIBRARY_PATH=/opt/4C-dependencies/lib stdbuf -oL -eL \
                #     {FOURC_BINARY} <deck>.yaml <out>
                # on 4C 2026.2.0-dev (commit 89519cfe76), mutating one key at a
                # time in two decks that both run clean unmutated:
                #   [2D] this backend's fsi/fsi_2d template (decks/fsi_2d.4C.yaml,
                #        partitioned Dirichlet-Neumann, 4 WALL QUAD4 + 16 FLUID
                #        QUAD4, 10 steps, 0.9 s, exit 0);
                #   [3D] tests/tutorials/preconditioner/tutorial_prec_fsi.4C.yaml
                #        (exodus mesh, iter_mortar_monolithicfluidsplit, Belos +
                #        Teko, cut to NUMSTEP 2 / MAXTIME 2e-4, 52 s, exit 0).
                (
                    "[Input] FSI is the most complex problem type in 4C and "
                    "the reason is structural, not conceptual: one deck drives "
                    "THREE discretisations (structure, fluid, ALE), and the "
                    "sections that wire them together fail in unrelated places "
                    "with unrelated wording, so there is no single string to grep "
                    "for. Signal: from the same working 2D deck, deleting PROBLEM "
                    "TYPE gives \"Required section 'PROBLEM TYPE' not found in "
                    "input file.\" from core/io/src/4C_io_input_file.cpp:617, "
                    "while deleting ALE DYNAMIC — a section with no default "
                    "LINEAR_SOLVER — gives 'No linear solver defined for ALE "
                    "problems. Please set' immediately followed by "
                    "'LINEAR_SOLVER in ALE DYNAMIC to a valid number!', two "
                    "adjacent literals at src/adapter/4C_adapter_ale.cpp:90-91 "
                    "thrown from line 89, i.e. the "
                    "field adapter blames the KEY and never says the section is "
                    "missing. Both exit 1, from different subsystems, at "
                    "different stages. Start from a working tutorial deck and "
                    "mutate it; do not assemble an FSI input from the grammar "
                    "and expect the errors to guide you. "
                    "(Verified by execution 2026-08-09.)"
                ),
                (
                    "[Input] FSI fluid elements MUST carry NA: ALE, never NA: "
                    "Euler — a non-ALE fluid field has no mesh-displacement dofs, "
                    "so the ALE field it should be cloned into comes out empty. "
                    "Signal: on the 2D partitioned deck the abort is 'got 27 "
                    "master nodes but 0 slave nodes for coupling' from "
                    "coupling/src/adapter/4C_coupling_adapter.cpp:182, thrown out "
                    "of Coupling::Adapter::Coupling::setup_coupling under "
                    "Adapter::FluidAle::FluidAle, before the first time step, "
                    "exit 1. Read the two numbers: 27 is EVERY fluid node in that "
                    "deck (the fluid->ALE field coupling, not the FSI interface) "
                    "and 0 is the empty ALE discretisation. Nothing in the "
                    "message mentions NA, Euler, or elements — it reads like a "
                    "mesh-pairing problem. Grep 'but 0 slave nodes'. "
                    "(Verified by execution 2026-08-09.)"
                ),
                (
                    "[Input] ALE Dirichlet BCs pin the mesh on every outer fluid "
                    "boundary that is not the FSI interface. Omitting them is a "
                    "quiet accuracy bug, NOT a divergence: an earlier version of "
                    "this entry claimed the mesh 'distorts freely and the "
                    "simulation diverges', and execution does not support that. "
                    "Signal: deleting the whole DESIGN LINE ALE DIRICH "
                    "CONDITIONS block from the 2D deck still completes all 10 "
                    "steps at exit 0 with no det(J), inverted-element or ALE "
                    "warning anywhere in the log. What moves is the answer: with "
                    "the same RESULT DESCRIPTION probes, fluid pressure at node "
                    "34 goes from -3.93255047510823741e-02 to "
                    "-3.73966417858043604e-02 (-4.9%) and fluid velx at node 12 "
                    "from 1.99424892254557146e-01 to 1.98440962728185039e-01, "
                    "while structural dispx at node 7 barely moves "
                    "(6.53965141370764086e-02 -> 6.53963740970964907e-02). Only "
                    "the FLUID quantities tell you. Result-test a fluid pressure "
                    "or interface value; exit status will never flag this. "
                    "(Verified by execution 2026-08-09.)"
                ),
                (
                    "[Mesh] With conforming (non-mortar) FSI coupling the two "
                    "interface node SETS must have equal cardinality — 4C pairs "
                    "them by geometric search and refuses to proceed on a count "
                    "mismatch. Signal: dropping a single node from the "
                    "fluid-side DLINE of the 2D deck's FSI interface aborts "
                    "before the first step with 'got 3 master nodes but 2 slave "
                    "nodes for coupling' from coupling/src/adapter/"
                    "4C_coupling_adapter.cpp:69, thrown from "
                    "Coupling::Adapter::Coupling::setup_condition_coupling, exit "
                    "1. Note the line number: :69 is the CONDITION coupling (the "
                    "FSI interface) whereas the identically-worded throw at :182 "
                    "is the field-wide fluid->ALE coupling — same sentence, "
                    "different bug, so check the frame. The counts are node "
                    "counts, so they localise the fault to the topology sections. "
                    "For non-matching meshes the documented route is a mortar "
                    "COUPALGO such as iter_mortar_monolithicfluidsplit (the 3D "
                    "tutorial deck runs one), but that this equal-count check is "
                    "skipped under mortar was NOT tested here — treat it as the "
                    "alternative to try, not as an established fact. "
                    "(Signal verified by execution 2026-08-09.)"
                ),
                (
                    "[Input] CLONING MATERIAL MAP is required in every FSI deck: "
                    "the ALE discretisation does not exist in the input, it is "
                    "cloned from the fluid, and the map is what tells 4C which "
                    "material the clone gets (SRC_FIELD: fluid, SRC_MAT: <fluid "
                    "id>, TAR_FIELD: ale, TAR_MAT: <a St.-Venant pseudo-material>). "
                    "Signal: deleting the section from the 2D deck aborts with "
                    "'At least one material pairing required in --CLONING "
                    "MATERIAL MAP.' from core/fem/src/general/utils/"
                    "4C_fem_general_utils_createdis.hpp:318, exit 1. The message "
                    "comes from the SHARED cloning helper, so it names neither "
                    "FSI nor the ALE field, and it spells the section in the "
                    "retired '--SECTION' dat form that you must NOT copy into "
                    "YAML. TSI, SSI and every other cloned-field problem emit the "
                    "identical sentence — read the backtrace (AleCloneStrategy "
                    "here) to learn which clone failed. "
                    "(Verified by execution 2026-08-09.)"
                ),
                (
                    "[Numerical] SHAPEDERIVATIVES in FSI DYNAMIC/MONOLITHIC "
                    "SOLVER adds d(fluid residual)/d(ALE displacement) to the "
                    "monolithic Jacobian. It is a Newton-cost knob, NOT a "
                    "requirement — an earlier version of this entry said it "
                    "'must be true for monolithic schemes', and execution "
                    "refutes that. Signal: flipping it from true to false on the "
                    "3D mortar-monolithic tutorial changes nothing about success "
                    "and nothing about the answer: the run still converges every "
                    "step at exit 0, and the initial residual of time step 2 is "
                    "IDENTICAL in both runs ('||F|| = 8.447e+02'), which is the "
                    "line to compare because it is the state carried out of step "
                    "1. What changes is only the Newton path — counting the "
                    "'-- Nonlinear Solver Step N -- ' banners, the two time "
                    "steps go from 7 + 7 with SHAPEDERIVATIVES true to 7 + 8 "
                    "with it false (the banner is printed once for the initial "
                    "evaluation, so those are 6 and 7 Newton updates). Since it "
                    "only touches the Jacobian, it cannot alter the converged "
                    "solution; treat it as a knob to try when Newton is slow, "
                    "and note it is meaningless for partitioned COUPALGOs. "
                    "(Verified by execution 2026-08-09.)"
                ),
                (
                    "[Numerical] SECONDORDER in FSI DYNAMIC raises the interface "
                    "time discretisation to second order. Unlike SHAPEDERIVATIVES "
                    "it CHANGES THE SOLUTION, so it is a modelling decision and "
                    "must be recorded alongside TIMESTEP when you report results. "
                    "Signal: it fails no test and prints no warning either way — "
                    "on the 3D mortar-monolithic tutorial, true and false both "
                    "converge every step at exit 0. The observable is the printed "
                    "nonlinear residual: with SECONDORDER: true the second time "
                    "step opens at '||F|| = 8.447e+02', with false at '||F|| = "
                    "1.226e+03', i.e. step 1 ended in a different state; the "
                    "first step already diverges in path ('||F|| = 1.191e+00' vs "
                    "'2.409e+00' at its first Newton update) and prints 9 "
                    "'-- Nonlinear Solver Step N -- ' banners instead of 7. "
                    "Diff the '||F|| =' column between two "
                    "runs — that is how you confirm a coupling switch did "
                    "anything at all. (Verified by execution 2026-08-09.)"
                ),
                (
                    "[Input] Each field's DYNAMIC section carries its own "
                    "LINEAR_SOLVER: N pointing at a SOLVER N block. Reusing one "
                    "block for all three fields is legal but usually "
                    "suboptimal; pointing at a block that does not exist is the "
                    "trap, because 4C does not check it. Signal: setting ALE "
                    "DYNAMIC/LINEAR_SOLVER to an undefined 9 produces NO 'PROC 0 "
                    "ERROR' block, no MPI_ABORT banner and no 4C source location "
                    "— Trilinos throws through 4C uncaught and the process dies "
                    "on SIGABRT at exit 134 with 'terminate called after throwing "
                    "an instance of Teuchos::Exceptions::InvalidParameterName' "
                    "and 'what():  Error!  The parameter \"SOLVER\" does not "
                    "exist in the parameter (sub)list \"ROOT->SOLVER 9\".' — that "
                    "quoted list name is the only thing that tells you which "
                    "number was wrong. The field is identifiable solely from the "
                    "backtrace frame (Adapter::AleBaseAlgorithm::setup_ale here). "
                    "Grep for 'does not exist' and 'ROOT->SOLVER'. "
                    "(Verified by execution 2026-08-09.)"
                ),
                (
                    "[Input] The FSI coupling condition is conventionally a LINE "
                    "condition in 2D and a SURF condition in 3D, but what 4C "
                    "actually enforces is that the topology you name EXISTS — the "
                    "condition is looked up by name ('FSICoupling'), not by "
                    "geometry rank. Signal: renaming DESIGN FSI COUPLING LINE "
                    "CONDITIONS to ...SURF... in the 2D deck, which defines "
                    "DLINEs and no DSURFACEs, is a loud two-line abort, not a "
                    "silent decoupling: 'DSurface 0 not in range [0:0[' followed "
                    "by 'DSurface condition on non existent DSurface?' and "
                    "'Could not read set from entity type.' — adjacent literals "
                    "at core/fem/src/condition/4C_fem_condition.cpp:135-136 that "
                    "the compiler joins with no separator, thrown from line 133 — "
                    "exit 1. The '[0:0[' is the giveaway "
                    "— zero entities of that kind were read. So the real rule is: "
                    "the condition kind must match the *-NODE TOPOLOGY sections "
                    "you wrote (DLINE-NODE TOPOLOGY -> LINE conditions). "
                    "(Verified by execution 2026-08-09.)"
                ),
                (
                    "[Input] Per-field NUMDOF in an FSI deck: structure = dim "
                    "(2 or 3), fluid = dim + 1 (the extra dof is pressure), ALE = "
                    "dim. The same DESIGN ... DIRICH section serves whichever "
                    "field owns the nodes, so one section can need two different "
                    "NUMDOFs on two different entity sets. Signal: adding a "
                    "DESIGN LINE DIRICH with NUMDOF: 2 on the 2D deck's "
                    "fluid-only outflow DLINE aborts reading 2 DOFs given but 3 "
                    "expected in Line Dirichlet boundary condition, from "
                    "core/fem/src/discretization/4C_fem_discretization_utils_dbc."
                    "cpp:292 at exit 1 — proving that a plain (non-ALE) Dirichlet "
                    "does reach the FLUID discretisation — while the identical "
                    "condition written with NUMDOF: 3 and 3-entry arrays runs to "
                    "exit 0, and the deck's structural DESIGN POINT DIRICH keeps "
                    "NUMDOF: 2 throughout. The message never names the field, so "
                    "map the entity id back to its topology section yourself. "
                    "The counts and the condition name are all substituted into "
                    "a format string of the shape <given> DOFs given but "
                    "<expected> expected in <condition name>, so the pieces to "
                    "grep are the 14-character run DOFs given but, one hit in "
                    "the tree, and the condition name, a string of its own: "
                    "'Line Dirichlet boundary condition' in "
                    "core/legacy_enum_definitions/"
                    "4C_legacy_enum_definitions_conditions.cpp:22. "
                    "(Verified by execution 2026-08-09.)"
                ),
                # Shared-node NUMDOF conflict (applies to ALL multi-physics)
                (
                    "[Input] CRITICAL: a DESIGN ... DIRICH condition is applied "
                    "to EVERY discretisation that contains the node, not the one "
                    "you had in mind. Where structure and fluid share interface "
                    "nodes this makes one NUMDOF serve two fields with different "
                    "dof counts, and only one direction of the clash is caught. "
                    "Signal: merging the 2D deck's fluid interface nodes onto the "
                    "structural ones (so nodes 7, 8, 9 live in both) and then "
                    "putting a DESIGN LINE DIRICH with NUMDOF: 2 on that line "
                    "aborts reading 2 DOFs given but 3 expected in Line "
                    "Dirichlet boundary condition (core/fem/src/discretization/"
                    "4C_fem_discretization_utils_dbc.cpp:292; the counts and the "
                    "name are run-time substitutions into that line's format "
                    "string, and the only piece of it that is a literal in its "
                    "own right is 'Line Dirichlet boundary condition' in "
                    "core/legacy_enum_definitions/"
                    "4C_legacy_enum_definitions_conditions.cpp:22) — the FLUID "
                    "claimed the structural condition. The SAME condition with "
                    "NUMDOF: 3 "
                    "runs to exit 0 even though the structure has only 2 dofs "
                    "there: the check is `num_dbc_dofs < numdf`, so a surplus is "
                    "dropped in silence and the structure quietly gets a "
                    "fluid-shaped Dirichlet. The dangerous direction is the one "
                    "that does not abort. Workarounds: (a) offset the structural "
                    "mesh so no node is shared, (b) mortar coupling with "
                    "non-conforming meshes, (c) drop the structural Dirichlet and "
                    "let the FSI constraint carry it. "
                    "(Verified by execution 2026-08-09.)"
                ),
                # Invalid section names
                (
                    "[Syntax] There is no DESIGN FLUID LINE LIFT&DRAG section in "
                    "4C — only the SURF form is defined — so 2D lift/drag has no "
                    "line-condition route. Signal: writing it into the 2D FSI "
                    "deck is refused at parse, before anything is read, with "
                    "\"Section 'DESIGN FLUID LINE LIFT&DRAG' is not a valid "
                    "section name.\" from core/io/src/4C_io_input_file.cpp:546, "
                    "exit 1. Note the check quotes your string verbatim and "
                    "offers no candidates, so a near-miss name gives you no hint. "
                    "Do NOT reach for LIFTDRAG: true in FLUID DYNAMIC as the 2D "
                    "substitute: only DESIGN FLUID SURF LIFT&DRAG registers the "
                    "'LIFTDRAG' condition that FLD::Utils::lift_drag() fetches, so "
                    "the flag alone parses, converges and computes nothing. "
                    "Integrate the traction yourself in 2D. "
                    "(Verified by execution 2026-08-09.)"
                ),
                # IO section
                (
                    "[Syntax] EVERY_ITERATION is not a parameter of the IO "
                    "section — it is real, but it lives in IO/RUNTIME VTK OUTPUT. "
                    "Signal: 'EVERY_ITERATION: true' under IO: aborts at parse "
                    "with 'Could not match this input' from "
                    "core/io/src/4C_io_input_spec_builders.cpp:633, the IO block "
                    "echoed back and the candidate specification printed, exit 1. "
                    "That message is generic — the identical string comes from a "
                    "mis-sized array, an out-of-enum value or a misplaced key — "
                    "so read the echoed block, not the sentence. Per-field output "
                    "frequency is RESULTSEVERY in STRUCTURAL DYNAMIC, FLUID "
                    "DYNAMIC and ALE DYNAMIC. "
                    "(Verified by execution 2026-08-09.)"
                ),
            ],
            "ale_boundary_conditions": {
                "description": (
                    "ALE Dirichlet BCs pin the ALE mesh at fixed fluid boundaries.  "
                    "The FSI interface is free (ALE moves with the structure there)."
                ),
                "rules": [
                    "ALL walls with no-slip fluid BC: apply ALE Dirichlet (fix mesh)",
                    "Inflow boundary: apply ALE Dirichlet (fix mesh)",
                    "Outflow boundary: apply ALE Dirichlet (fix mesh)",
                    "Cylinder/obstacle surfaces: apply ALE Dirichlet (fix mesh)",
                    "FSI interface: do NOT apply ALE Dirichlet (mesh moves with structure)",
                ],
                "common_mistake": (
                    "Forgetting ALE Dirichlet on some outer boundary causes the "
                    "ALE mesh to distort freely there, leading to inverted elements "
                    "and solver divergence."
                ),
            },
            "valid_2d_elements": {
                "FLUID": ["QUAD4", "QUAD8", "QUAD9", "TRI3", "TRI6"],
                "WALL (2-D structure, 4C before 2026.2.0)": ["QUAD4", "QUAD8", "QUAD9", "TRI3", "TRI6"],
                "SOLID (2-D structure, 4C 2026.2.0 and later)": ["QUAD4", "QUAD8", "QUAD9", "TRI3", "TRI6"],
                "notes": (
                    "Which element owns 2-D structure depends on the build. "
                    "Before 4C 2026.2.0: WALL, and SOLID is 3-D only (HEX8/HEX18/HEX20/"
                    "HEX27, TET4/TET10, WEDGE6, PYRAMID5, NURBS27), so 'SOLID "
                    "QUAD4' is rejected; a WALL QUAD4 element needs all six "
                    "of MAT, KINEM, EAS, THICK, STRESS_STRAIN and GP -- the "
                    "token is THICK, not THICKNESS, and GP is a 2-vector such "
                    "as [2, 2].  4C 2026.2.0 and later have no WALL: the same element is "
                    "'SOLID QUAD4 <nodes> MAT m KINEM k THICKNESS t "
                    "PLANE_ASSUMPTION plane_strain' (EAS none and GP 2 2 are "
                    "its defaults).  "
                    "QUAD4 is most commonly used and best validated for FSI.  "
                    "TRI3 works but is less accurate for pressure.  "
                    "For 3-D FSI: FLUID HEX8/TET4, SOLID HEX8/TET4 (SOLID "
                    "takes MAT and KINEM, and optionally TECH, PRESTRESS_TECH, "
                    "FIBER1..3, RAD/AXI/CIR and an INTEGRATION sub-group)."
                ),
            },
            "typical_experiments": [
                {
                    "name": "channel_with_flap_2d",
                    "description": (
                        "Flow past a flexible flap (Turek-Hron benchmark).  "
                        "A channel with a cylinder and an attached elastic "
                        "flag.  Tests vortex shedding, large structural "
                        "deformation, and ALE mesh quality."
                    ),
                },
            ],
        }

    # ── Variants ──────────────────────────────────────────────────────

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "fsi_2d",
                "description": (
                    "2-D monolithic FSI: channel flow past a deformable wall "
                    "segment.  Neo-Hookean structure, Newtonian fluid, ALE "
                    "mesh motion.  UMFPACK solvers for simplicity."
                ),
            },
        ]

    # ── Templates ─────────────────────────────────────────────────────

    def get_template(self, variant: str = "fsi_2d") -> str:
        templates = {
            "fsi_2d": self._template_fsi_2d,
        }
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_fsi_2d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # Written in the input grammar of 4C before 2026.2.0 (the WALL element);
            # from 4C 2026.2.0 on a 2-D solid is SOLID with THICKNESS and PLANE_ASSUMPTION. The decks openPASO runs are served in the grammar of the
            # 4C it finds.
            # ---------------------------------------------------------------
            # 2-D Monolithic FSI -- Channel with Deformable Wall
            #
            # Structure: Neo-Hookean elastic wall on the upper channel boundary
            # Fluid:     Incompressible Navier-Stokes (ALE)
            # Coupling:  Monolithic fluid-split with mortar
            #
            # Mesh: requires "fsi_2d.e" with:
            #   element_block 1 = structure (QUAD4)
            #   element_block 2 = fluid (QUAD4)
            #   node_set 1 = structure fixed end (Dirichlet)
            #   node_set 2 = FSI interface (structure side)
            #   node_set 3 = structure fixed end (other)
            #   node_set 5 = fluid inlet
            #   node_set 6 = fluid bottom wall
            #   node_set 7 = fluid outlet (or left open for do-nothing)
            #   node_set 8 = fluid outer boundary (ALE Dirichlet)
            #   node_set 9 = FSI interface (fluid side)
            # ---------------------------------------------------------------
            PROBLEM SIZE:
              DIM: 2
            PROBLEM TYPE:
              PROBLEMTYPE: "Fluid_Structure_Interaction"

            # == Structure =================================================
            STRUCTURAL DYNAMIC:
              INT_STRATEGY: "Standard"
              LINEAR_SOLVER: 3
              PREDICT: "ConstDisVelAcc"
              M_DAMP: <structure_mass_damping>
              K_DAMP: <structure_stiffness_damping>
              TOLDISP: <structure_displacement_tolerance>
              TOLRES: <structure_residual_tolerance>
            STRUCTURAL DYNAMIC/GENALPHA:
              BETA: <genalpha_beta>
              GAMMA: <genalpha_gamma>
              ALPHA_M: <genalpha_alpha_m>
              ALPHA_F: <genalpha_alpha_f>
              RHO_INF: <genalpha_rho_inf>

            # == Fluid =====================================================
            FLUID DYNAMIC:
              LINEAR_SOLVER: 2
              TIMEINTEGR: "Np_Gen_Alpha"
              GRIDVEL: BDF2
              ADAPTCONV: true
              ITEMAX: <fluid_max_iterations>
            FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES:
              TOL_VEL_RES: <fluid_velocity_residual_tolerance>
              TOL_VEL_INC: <fluid_velocity_increment_tolerance>
              TOL_PRES_RES: <fluid_pressure_residual_tolerance>
              TOL_PRES_INC: <fluid_pressure_increment_tolerance>
            FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION:
              CHARELELENGTH_PC: "root_of_volume"

            # == ALE mesh motion ===========================================
            ALE DYNAMIC:
              ALE_TYPE: springs_spatial
              MAXITER: <ale_max_iterations>
              TOLRES: <ale_residual_tolerance>
              TOLDISP: <ale_displacement_tolerance>
              LINEAR_SOLVER: 1

            # == FSI coupling ==============================================
            FSI DYNAMIC:
              MAXTIME: <end_time>
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              SECONDORDER: true
            FSI DYNAMIC/MONOLITHIC SOLVER:
              SHAPEDERIVATIVES: true

            # == Solvers ===================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "ALE solver"
            SOLVER 2:
              SOLVER: "UMFPACK"
              NAME: "Fluid solver"
            SOLVER 3:
              SOLVER: "UMFPACK"
              NAME: "Structure solver"

            # == Materials =================================================
            MATERIALS:
              # Fluid material
              - MAT: 1
                MAT_fluid:
                  DYNVISCOSITY: <fluid_dynamic_viscosity>
                  DENSITY: <fluid_density>
              # Structure material (Neo-Hookean hyperelastic)
              - MAT: 2
                MAT_ElastHyper:
                  NUMMAT: 1
                  MATIDS: [3]
                  DENS: <structure_density>
              - MAT: 3
                ELAST_CoupNeoHooke:
                  YOUNG: <structure_Young_modulus>
              # ALE pseudo-material (cloned from fluid)
              - MAT: 4
                MAT_Struct_StVenantKirchhoff:
                  YOUNG: <ale_Young_modulus>
                  NUE: <ale_Poisson_ratio>
                  DENS: <ale_density>

            # Map fluid material -> ALE material
            CLONING MATERIAL MAP:
              - SRC_FIELD: "fluid"
                SRC_MAT: 1
                TAR_FIELD: "ale"
                TAR_MAT: 4

            # == Geometry ==================================================
            # ELEMENT_BLOCKS entries are ID -> <ELEMENT_TYPE> -> <CELL_TYPE> ->
            # tokens, and the element type must match the dimension.  Before 4C
            # 2026.2.0 SOLID is 3-D ONLY: its legal cell types are HEX8/HEX18/
            # HEX20/HEX27/TET4/TET10/WEDGE6/PYRAMID5/NURBS27, so 'SOLID: QUAD4:'
            # is rejected with "Could not match this input".  The 2-D
            # structural element type there is WALL, and its QUAD4 block
            # requires ALL SIX of MAT, KINEM, EAS, THICK, STRESS_STRAIN and GP
            # (THICK, not THICKNESS).  4C 2026.2.0 and later have no WALL: their
            # SOLID takes QUAD4 with THICKNESS and PLANE_ASSUMPTION.
            STRUCTURE GEOMETRY:
              FILE: "fsi_2d.e"
              ELEMENT_BLOCKS:
                - ID: 1
                  WALL:
                    QUAD4:
                      MAT: 2
                      KINEM: nonlinear
                      EAS: none
                      THICK: <wall_thickness>
                      STRESS_STRAIN: plane_strain
                      GP: [2, 2]

            FLUID GEOMETRY:
              FILE: "fsi_2d.e"
              ELEMENT_BLOCKS:
                - ID: 2
                  FLUID:
                    QUAD4:
                      MAT: 1
                      NA: ALE

            # == Boundary Conditions =======================================

            # Structure: fixed supports
            DESIGN POINT DIRICH CONDITIONS:
              - E: 1
                ENTITY_TYPE: node_set_id
                NUMDOF: 2
                ONOFF: [1, 1]
                VAL: [0.0, 0.0]
                FUNCT: [0, 0]
              - E: 3
                ENTITY_TYPE: node_set_id
                NUMDOF: 2
                ONOFF: [1, 1]
                VAL: [0.0, 0.0]
                FUNCT: [0, 0]

            # Fluid: inlet parabolic, walls no-slip
            DESIGN LINE DIRICH CONDITIONS:
              # Structure fixed edge (line)
              - E: 1
                ENTITY_TYPE: node_set_id
                NUMDOF: 2
                ONOFF: [1, 1]
                VAL: [0.0, 0.0]
                FUNCT: [0, 0]
              # Fluid inlet (parabolic ramp-up)
              - E: 5
                ENTITY_TYPE: node_set_id
                NUMDOF: 3
                ONOFF: [1, 1, 0]
                VAL: [<inlet_velocity>, 0.0, 0.0]
                FUNCT: [1, 0, 0]
              # Fluid bottom wall (no-slip)
              - E: 6
                ENTITY_TYPE: node_set_id
                NUMDOF: 3
                ONOFF: [1, 1, 0]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]

            # ALE: fix outer fluid boundaries
            DESIGN LINE ALE DIRICH CONDITIONS:
              - E: 6
                ENTITY_TYPE: node_set_id
                NUMDOF: 2
                ONOFF: [1, 1]
                VAL: [0.0, 0.0]
                FUNCT: [0, 0]
              - E: 8
                ENTITY_TYPE: node_set_id
                NUMDOF: 2
                ONOFF: [1, 1]
                VAL: [0.0, 0.0]
                FUNCT: [0, 0]

            # FSI coupling interface
            DESIGN FSI COUPLING LINE CONDITIONS:
              - E: 2
                ENTITY_TYPE: node_set_id
                coupling_id: 1
              - E: 9
                ENTITY_TYPE: node_set_id
                coupling_id: 1

            # Smooth ramp-up function for inlet
            # IMPORTANT: When using VARIABLE/multifunction with
            # SYMBOLIC_FUNCTION_OF_SPACE_TIME, you MUST include COMPONENT: 0.
            # Without it, the VARIABLE definition is not parsed correctly.
            #
            # Example with ramp-up variable:
            #   FUNCT1:
            #     - COMPONENT: 0
            #       SYMBOLIC_FUNCTION_OF_SPACE_TIME: "6*U_bar*y*(H-y)/(H*H)*a"
            #     - VARIABLE: 0
            #       NAME: "a"
            #       TYPE: "multifunction"
            #       NUMPOINTS: 3
            #       TIMES: [0, 2, 10000]
            #       DESCRIPTION: ["0.5*(1-cos(pi*t/2))", "1.0"]
            #
            # Simpler alternative (no VARIABLE needed, bake time into expression):
            #   FUNCT1:
            #     - COMPONENT: 0
            #       SYMBOLIC_FUNCTION_OF_SPACE_TIME: "6*y*(H-y)/(H*H)*(t<T_ramp?0.5*(1-cos(pi*t/T_ramp)):1)"
            FUNCT1:
              - COMPONENT: 0
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<inlet_ramp_expression>"

            # == VTK output ================================================
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/STRUCTURE:
              OUTPUT_STRUCTURE: true
              DISPLACEMENT: true
            IO/RUNTIME VTK OUTPUT/FLUID:
              OUTPUT_FLUID: true
              VELOCITY: true
              # PRESSURE is deliberately NOT requested for this 2-D ALE fluid: with it, 4C wrote NaN
              # at every node and every step while velocity and displacement stayed finite (measured
              # 2026-09-23). The runtime-VTK pressure writer sizes its dof context for 3-D (ux, uy,
              # uz, p) and labels dof index 3 as pressure; a 2-D node has three dofs (ux, uy, p).
              # None of 4C's own 2-D FSI test decks request it. The 2-D pressure is the third
              # component of the "velocity" array; or output it through the legacy IO writer.
        """)

    # ── Validation ────────────────────────────────────────────────────

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        """Validate FSI-specific parameters.

        Checks that all required sections are present, fluid uses ALE,
        and CLONING MATERIAL MAP exists.
        """
        issues: list[str] = []

        # Check required sections if a full input dict is provided
        sections = params.get("sections") or params.get("input_sections")
        if sections is not None:
            if isinstance(sections, (list, set)):
                section_set = set(s.upper() if isinstance(s, str) else s for s in sections)
            elif isinstance(sections, dict):
                section_set = set(
                    k.upper() if isinstance(k, str) else k for k in sections.keys()
                )
            else:
                section_set = set()

            required = {
                "PROBLEM TYPE",
                "STRUCTURAL DYNAMIC",
                "FLUID DYNAMIC",
                "ALE DYNAMIC",
                "FSI DYNAMIC",
                "MATERIALS",
                "STRUCTURE GEOMETRY",
                "FLUID GEOMETRY",
                "CLONING MATERIAL MAP",
            }
            missing = required - section_set
            if missing:
                issues.append(
                    f"Missing required FSI sections: {sorted(missing)}"
                )

        # Check fluid NA mode
        fluid_na = params.get("fluid_NA") or params.get("NA")
        if fluid_na is not None:
            if str(fluid_na).upper() != "ALE":
                issues.append(
                    f"Fluid elements MUST use NA: ALE for FSI, got {fluid_na!r}.  "
                    f"Euler grid does not support mesh motion."
                )

        # Check SHAPEDERIVATIVES
        shape_deriv = params.get("SHAPEDERIVATIVES")
        if shape_deriv is not None and not shape_deriv:
            issues.append(
                "SHAPEDERIVATIVES must be true in FSI DYNAMIC/MONOLITHIC SOLVER "
                "for monolithic coupling schemes."
            )

        # Validate materials
        viscosity = params.get("viscosity") or params.get("DYNVISCOSITY")
        if viscosity is not None:
            try:
                mu = float(viscosity)
                if mu <= 0:
                    issues.append(f"Fluid DYNVISCOSITY must be > 0, got {mu}.")
            except (TypeError, ValueError):
                issues.append(f"Fluid DYNVISCOSITY must be numeric, got {viscosity!r}.")

        density = params.get("fluid_density") or params.get("DENSITY")
        if density is not None:
            try:
                rho = float(density)
                if rho <= 0:
                    issues.append(f"Fluid DENSITY must be > 0, got {rho}.")
            except (TypeError, ValueError):
                issues.append(f"Fluid DENSITY must be numeric, got {density!r}.")

        young = params.get("YOUNG") or params.get("young")
        if young is not None:
            try:
                E = float(young)
                if E <= 0:
                    issues.append(f"Structure YOUNG must be > 0, got {E}.")
            except (TypeError, ValueError):
                issues.append(f"Structure YOUNG must be numeric, got {young!r}.")

        poisson = params.get("NUE") or params.get("poisson_ratio")
        if poisson is not None:
            try:
                nu = float(poisson)
                if nu < 0 or nu >= 0.5:
                    issues.append(
                        f"Poisson's ratio must be in [0, 0.5), got {nu}.  "
                        f"nu=0.5 is incompressible (not supported for solid)."
                    )
            except (TypeError, ValueError):
                issues.append(f"Poisson's ratio must be numeric, got {poisson!r}.")

        # Check CLONING MATERIAL MAP presence
        has_cloning = params.get("has_cloning_material_map")
        if has_cloning is not None and not has_cloning:
            issues.append(
                "CLONING MATERIAL MAP is required for FSI.  It maps the fluid "
                "material to the ALE pseudo-material."
            )

        return issues
