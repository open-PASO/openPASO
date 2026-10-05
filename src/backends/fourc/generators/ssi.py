"""Structure-Scalar Interaction (SSI) generator for 4C.

Covers monolithic and partitioned coupling of structural mechanics with
scalar transport (including electrochemistry).  Key application: electrode
mechanics in battery simulations where lithium intercalation causes
volumetric expansion, and mechanical stress affects diffusion and
electrochemical kinetics.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class SSIGenerator(BaseGenerator):
    """Generator for Structure-Scalar Interaction problems in 4C."""

    module_key = "ssi"
    display_name = "Structure-Scalar Interaction (SSI / Electrode Mechanics)"
    problem_type = "Structure_Scalar_Interaction"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Structure-Scalar Interaction (SSI) couples a structural "
                "mechanics field with a scalar transport field.  The primary "
                "application is electrode mechanics in lithium-ion batteries, "
                "where intercalation of lithium ions causes volumetric "
                "swelling of the electrode particles, and mechanical stress "
                "in turn affects ion diffusion and electrochemical reaction "
                "kinetics (Butler-Volmer).  4C supports monolithic "
                "(ssi_Monolithic) and partitioned (ssi_IterStagg) coupling.  "
                "The PROBLEM TYPE is 'Structure_Scalar_Interaction'.  The "
                "coupling is controlled via the 'SSI CONTROL' section.  "
                "When electrochemistry is involved, set SCATRATIMINTTYPE: "
                "'Elch' and include ELCH CONTROL and S2I COUPLING sections."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "STRUCTURAL DYNAMIC",
                "SCALAR TRANSPORT DYNAMIC",
                "SSI CONTROL",
                "SSI CONTROL/MONOLITHIC",
                "SOLVER 1",
                "MATERIALS",
            ],
            "optional_sections": [
                "ELCH CONTROL",
                "SSI CONTROL/ELCH",
                "SSI CONTROL/PARTITIONED",
                "SCALAR TRANSPORT DYNAMIC/S2I COUPLING",
                "SCALAR TRANSPORT DYNAMIC/STABILIZATION",
                "SCALAR TRANSPORT DYNAMIC/NONLINEAR",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/STRUCTURE",
                # The S2I interface condition sections. The geometry word
                # comes LAST; there is no "DESIGN SURF S2I ..." spelling and
                # no "S2I COUPLING" condition section at all.
                "DESIGN S2I KINETICS SURF CONDITIONS",
                "DESIGN S2I KINETICS LINE CONDITIONS",
                "DESIGN S2I KINETICS POINT CONDITIONS",
                "DESIGN S2I KINETICS GROWTH SURF CONDITIONS",
                "DESIGN S2I KINETICS GROWTH LINE CONDITIONS",
                "DESIGN S2I MESHTYING SURF CONDITIONS",
                "DESIGN S2I MESHTYING LINE CONDITIONS",
                "DESIGN S2I SCL COUPLING SURF CONDITIONS",
                # SSI's own interface conditions (structure side):
                "DESIGN SSI INTERFACE MESHTYING SURF CONDITIONS",
                "DESIGN SSI INTERFACE CONTACT SURF CONDITIONS",
            ],
            "materials": {
                "MAT_MultiplicativeSplitDefgradElastHyper": {
                    "description": (
                        "Multiplicative split of the deformation gradient "
                        "into elastic and inelastic parts.  Used for "
                        "electrode mechanics where lithium intercalation "
                        "causes inelastic volumetric growth.  References "
                        "an elastic sub-material and an inelastic "
                        "deformation gradient factor."
                    ),
                    "parameters": {
                        "NUMMATEL": {
                            "description": "Number of elastic sub-materials",
                            "range": ">= 1",
                        },
                        "MATIDSEL": {
                            "description": (
                                "List of elastic sub-material IDs "
                                "(e.g. ELAST_CoupSVK)"
                            ),
                            "range": "valid MAT IDs",
                        },
                        "NUMFACINEL": {
                            "description": (
                                "Number of inelastic deformation gradient "
                                "factors"
                            ),
                            "range": ">= 1",
                        },
                        "INELDEFGRADFACIDS": {
                            "description": (
                                "List of inelastic deformation gradient "
                                "factor material IDs"
                            ),
                            "range": "valid MAT IDs",
                        },
                        "DENS": {
                            "description": "Mass density",
                            "range": "> 0",
                        },
                    },
                },
                "MAT_electrode": {
                    "description": (
                        "Electrode material for lithium-ion batteries.  "
                        "Defines concentration-dependent diffusion "
                        "coefficient, electronic conductivity, maximum "
                        "concentration, and open-circuit potential (OCP) "
                        "model.  Used in the scalar transport field.  The "
                        "parameters below are those of 4C before 2026.3.0 "
                        "(the 2026.2.0 release included); "
                        "changed_in_4C_2026_3 gives 4C 2026.3.0's form."
                    ),
                    "parameters": {
                        "DIFF_PARA_NUM": {
                            "description": (
                                "Number of diffusion coefficient parameters"
                            ),
                            "range": ">= 1",
                        },
                        "DIFF_PARA": {
                            "description": "Diffusion coefficient parameter(s)",
                            "range": "> 0",
                        },
                        "COND_PARA_NUM": {
                            "description": (
                                "Number of conductivity parameters"
                            ),
                            "range": ">= 1",
                        },
                        "COND_PARA": {
                            "description": "Electronic conductivity parameter(s)",
                            "range": "> 0",
                        },
                        "C_MAX": {
                            "description": (
                                "Maximum lithium concentration in the "
                                "electrode material"
                            ),
                            "range": "> 0",
                        },
                        "CHI_MAX": {
                            "description": (
                                "Maximum state of charge (stoichiometry)"
                            ),
                            "range": "> 0",
                        },
                        "OCP_MODEL": {
                            "description": (
                                "Open-circuit-potential model.  A nested "
                                "group, not a scalar.  It takes exactly one "
                                "sub-group -- Function (OCP_FUNCT_NUM), "
                                "Redlich-Kister (OCP_PARA_NUM + OCP_PARA) or "
                                "Taralov -- PLUS both X_MIN and X_MAX at its "
                                "own level.  X_MAX is not optional: writing "
                                "X_MIN alone fails the whole MATERIALS "
                                "section with 'Could not match this input'.  "
                                "-1 / -1 switches the range check off."
                            ),
                        },
                    },
                    "changed_in_4C_2026_3": (
                        "4C 2026.3.0 replaced the function-id-plus-parameter "
                        "form above. DIFF_COEF and COND are constants; "
                        "DIFF_COEF_CONC_SCALE_FUNCT, DIFF_COEF_TEMP_SCALE_FUNCT, "
                        "COND_CONC_SCALE_FUNCT and COND_TEMP_SCALE_FUNCT are "
                        "optional FUNCT ids that scale them; the *_PARA_NUM / "
                        "*_PARA pairs and OCP_PARA_NUM are gone, and an optional "
                        "LITHIATION_BOUNDS group replaces X_MIN / X_MAX. 4C's "
                        "own tests were migrated as DIFF_COEF_CONC_DEP_FUNCT: "
                        "-1 with DIFF_PARA: [d] -> DIFF_COEF: d. openPASO's "
                        "ssi, sti and ssti templates are served in the form "
                        "your 4C reads."
                    ),
                    "also_required_by_the_spec": (
                        "4C before 2026.3.0: DIFF_COEF_CONC_DEP_FUNCT, "
                        "DIFF_COEF_TEMP_SCALE_FUNCT, COND_CONC_DEP_FUNCT "
                        "and COND_TEMP_SCALE_FUNCT.  Upstream writes -1 for "
                        "the two _CONC_DEP_ ones (use the DIFF_PARA / "
                        "COND_PARA polynomial instead) and 0 for the two "
                        "_TEMP_SCALE_ ones (no temperature scaling)."
                    ),
                },
                "ELAST_CoupSVK": {
                    "description": (
                        "Coupled St. Venant-Kirchhoff elastic material.  "
                        "Used as the elastic sub-material within "
                        "MAT_MultiplicativeSplitDefgradElastHyper."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Young's modulus",
                            "range": "> 0",
                        },
                        "NUE": {
                            "description": "Poisson's ratio",
                            "range": "(0, 0.5)",
                        },
                    },
                },
                "MAT_InelasticDefgradNoGrowth": {
                    "description": (
                        "Trivial inelastic deformation gradient factor "
                        "that applies no growth.  Used as a placeholder "
                        "or for problems where only elastic deformation "
                        "is desired."
                    ),
                },
            },
            "solver": {
                "monolithic": {
                    "type": "UMFPACK for small problems, Belos + block prec for large",
                    "notes": (
                        "The monolithic SSI solver handles the coupled "
                        "displacement-concentration system.  For electrode "
                        "mechanics with electrochemistry, the system can be "
                        "large and may benefit from iterative solvers."
                    ),
                },
            },
            # The complete value set of SSI CONTROL's COUPALGO.  Note the
            # spelling: no underscore between "IterStagg" and
            # "FixedRel"/"Aitken".
            "coupling_algorithms": {
                "ssi_Monolithic": (
                    "Fully coupled monolithic solve.  Both fields are "
                    "assembled into a single block system and solved "
                    "simultaneously.  Most robust for strong coupling."
                ),
                "ssi_IterStagg": (
                    "Iterative staggered (partitioned) approach.  Fields "
                    "are solved alternately with relaxation until "
                    "convergence.  Cheaper per iteration but may need "
                    "more iterations for strongly coupled problems.  "
                    "This is the DEFAULT, so a deck that omits COUPALGO "
                    "is partitioned, not monolithic."
                ),
                "ssi_IterStaggFixedRel_ScatraToSolid": (
                    "Iterative staggered with fixed relaxation, scatra "
                    "solved first.  Relaxation set in SSI "
                    "CONTROL/PARTITIONED (STARTOMEGA)."
                ),
                "ssi_IterStaggFixedRel_SolidToScatra": (
                    "As above with the solid solved first."
                ),
                "ssi_IterStaggAitken_ScatraToSolid": (
                    "Iterative staggered with Aitken relaxation, scatra "
                    "first.  MINOMEGA / MAXOMEGA in SSI "
                    "CONTROL/PARTITIONED bound the relaxation factor."
                ),
                "ssi_IterStaggAitken_SolidToScatra": (
                    "As above with the solid solved first."
                ),
                "ssi_OneWay_ScatraToSolid": (
                    "One-way: the scalar field drives the solid and gets "
                    "no feedback."
                ),
                "ssi_OneWay_SolidToScatra": (
                    "One-way: the solid drives the scalar field and gets "
                    "no feedback."
                ),
            },
            "electrochemistry_settings": {
                "SCATRATIMINTTYPE": (
                    "Set to 'Elch' in SSI CONTROL to enable the "
                    "electrochemistry scalar transport formulation."
                ),
                "EQUPOT": (
                    "Electroneutrality condition in ELCH CONTROL.  Options: "
                    "'divi' (divergence-based), 'ENC' (electroneutrality "
                    "constraint)."
                ),
                "DIFFCOND_FORMULATION": (
                    "Set to true in ELCH CONTROL for diffusion-conduction "
                    "formulation (concentrated solution theory)."
                ),
                "INITPOTCALC": (
                    "Set to true in SSI CONTROL/ELCH to compute a "
                    "consistent initial electric potential field."
                ),
            },
            "pitfalls": [
                (
                    "[Input] SCATRATIMINTTYPE must be set to "
                    "'Elch' in SSI CONTROL when "
                    "electrochemistry is involved -- and this "
                    "is checked, not defaulted. Signal: "
                    "omitting it makes the concentration and "
                    "potential unknowns look like two ordinary "
                    "transported scalars, and monolithic SSI "
                    "aborts at SsiMono::setup() in "
                    "src/ssi/4C_ssi_monolithic.cpp with a "
                    "message ending '...it is not reasonable "
                    "to use them with more than one "
                    "transported scalar', exit 1. (Verified by "
                    "execution 2026-08-06.  An earlier version "
                    "predicted a silent fallback to plain "
                    "scalar transport with a zero "
                    "Butler-Volmer current; no time step ever "
                    "runs, so there is no zero current to "
                    "observe.)"
                ),
                (
                    "[Input] For electrode problems, "
                    "structural elements MUST use "
                    "MAT_MultiplicativeSplitDefgradElastHyper "
                    "with appropriate inelastic growth "
                    "factors (F = F_e * F_g, with F_g driven "
                    "by concentration). Signal: a plain "
                    "MAT_ElastHyper aborts with 'Your material "
                    "does not allow to evaluate a monolithic "
                    "ssi material!' from "
                    "solid_scatra_3D_ele/"
                    "4C_solid_scatra_3D_ele_calc.cpp, raised "
                    "inside "
                    "SolidScatraEleCalc::evaluate_d_stress_d_scalar "
                    "-- the element asks the material for the "
                    "concentration derivative of the stress, "
                    "which IS the swelling coupling term. "
                    "(Verified by execution 2026-08-06; the "
                    "earlier prediction of a quiet "
                    "zero-deformation answer is wrong, the run "
                    "stops.)"
                ),
                (
                    "[Input] S2I (scatra-scatra interface) "
                    "coupling for electrode-electrolyte "
                    "interfaces needs BOTH the DESIGN S2I "
                    "KINETICS conditions and COUPLINGTYPE in "
                    "SCALAR TRANSPORT DYNAMIC/S2I COUPLING; "
                    "COUPLINGTYPE has no usable default. "
                    "Signal: with S2I conditions present but "
                    "the section missing, 'Type of mortar "
                    "meshtying for scatra-scatra interface "
                    "coupling not recognized!' from "
                    "scatra/"
                    "4C_scatra_timint_meshtying_strategy_s2i.cpp "
                    "at MeshtyingStrategyS2I::setup_meshtying(), "
                    "exit 1.  With NO S2I conditions the same "
                    "section is entirely inert, which is why "
                    "deleting it looks harmless on some decks. "
                    "(Verified by execution 2026-08-06; there "
                    "is no silently decoupled run.)"
                ),
                (
                    "[Input] The S2I condition sections put the "
                    "geometry word last -- DESIGN S2I KINETICS "
                    "SURF CONDITIONS, DESIGN S2I MESHTYING SURF "
                    "CONDITIONS -- and there is no 'S2I "
                    "COUPLING' condition section at all.  "
                    "Writing DESIGN SURF S2I COUPLING "
                    "CONDITIONS, which reads like every other "
                    "DESIGN SURF ... block, is a parse abort at "
                    "line 546 of core/io/src/4C_io_input_file"
                    ".cpp: \"Section 'DESIGN SURF S2I COUPLING "
                    "CONDITIONS' is not a valid section name.\" "
                    "And the KINETICS conditions never stand "
                    "alone: each one needs a matching meshtying "
                    "condition, DESIGN SSI INTERFACE MESHTYING "
                    "SURF CONDITIONS for an SSI problem or "
                    "DESIGN S2I MESHTYING SURF CONDITIONS for a "
                    "plain scatra/elch one, paired by a shared "
                    "ConditionID.  Signal: that second mistake "
                    "surfaces much later and does not mention "
                    "the section you forgot to write -- 'For each "
                    "\"S2IKinetics\" or \"S2ISCLCoupling\" "
                    "condition a corresponding \"S2IMeshtying\" "
                    "or \"S2INoEvaluation\" condition has to be "
                    "defined!' from scatra/4C_scatra_utils.cpp "
                    "at ScaTraUtils::check_consistency_of_s2_i_"
                    "conditions, raised during "
                    "ScaTraTimIntImpl::init().  The SSI-"
                    "specific section satisfies that check "
                    "because SSI's clone strategy renames it: "
                    "ssi/4C_ssi_clonestrategy.cpp maps "
                    "'ssi_interface_meshtying' -> 'S2IMeshtying' "
                    "onto the cloned scatra discretisation, so "
                    "it only works under PROBLEMTYPE "
                    "Structure_Scalar_Interaction. Note also that "
                    "INTERFACE_SIDE is capitalised, \"Slave\" / "
                    "\"Master\", and that the master side of a "
                    "kinetics pair carries only E, ConditionID "
                    "and INTERFACE_SIDE -- no KINETIC_MODEL and "
                    "no kinetic parameters. (Verified by "
                    "execution 2026-08-07.)"
                ),
                (
                    "[Input] VELOCITYFIELD must be set to "
                    "'Navier_Stokes' in SCALAR TRANSPORT "
                    "DYNAMIC for SSI -- it is what couples the "
                    "scatra to the structural velocity. "
                    "Signal: 'zero' is rejected, not ignored: "
                    "'Invalid type of velocity field for "
                    "scalar-structure interaction!' from "
                    "src/ssi/4C_ssi_monolithic.cpp at "
                    "SsiMono::init(), exit 1, before any field "
                    "is built. (Verified by execution "
                    "2026-08-06; the earlier prediction of a "
                    "quietly stationary concentration field is "
                    "wrong.)"
                ),
                (
                    "[Numerical] CONVFORM must be "
                    "'conservative' whenever the scalar is "
                    "volume-referenced (IS_INTENSIVE_SCALAR = "
                    "false), which is the usual SSI case -- "
                    "the conservative form is what accounts "
                    "for the volume change of the deforming "
                    "domain. Signal: 'convective' is refused "
                    "at SSIBase::init() in "
                    "src/ssi/4C_ssi_base.cpp with "
                    "'Inconsistent scalar transport "
                    "formulation on a deforming domain: ... "
                    "Please set CONVFORM to conservative in "
                    "the SCALAR TRANSPORT DYNAMIC section.', "
                    "exit 1.  Note the condition is on the "
                    "SCALAR's definition, not on SSI as such. "
                    "(Verified by execution 2026-08-06; the "
                    "earlier prediction of a slow "
                    "mass-balance drift is wrong -- no time "
                    "step runs at all.)"
                ),
                (
                    "[Input] What must agree is SSI CONTROL/"
                    "MONOLITHIC's MATRIXTYPE with SCALAR "
                    "TRANSPORT DYNAMIC's MATRIXTYPE, and the "
                    "scatra matrix type with the "
                    "preconditioner -- both couplings are "
                    "checked and both abort.  'block' with a "
                    "plain direct solver is NOT penalised: it "
                    "runs normally. Signal: a scatra "
                    "MATRIXTYPE of block_condition under a "
                    "direct solver gives 'Global system matrix "
                    "with block structure requires AMGnxn, "
                    "MueLu or Teko block preconditioner!' from "
                    "scatra/4C_scatra_timint_implicit.cpp; a "
                    "'sparse' SSI matrix over a block scatra "
                    "field gives 'Incompatible matrix type "
                    "associated with scalar transport field!' "
                    "from src/ssi/4C_ssi_monolithic.cpp at "
                    "SsiMono::setup_system().  Both exit 1 "
                    "before any solve. (Verified by execution "
                    "2026-08-06.  An earlier version framed "
                    "both mismatches as performance problems "
                    "-- exploding preconditioner iterations, "
                    "wasted memory; neither happens, they are "
                    "hard errors.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "electrode_intercalation_3d",
                    "description": (
                        "Two electrode particles separated by a membrane "
                        "with Butler-Volmer kinetics.  Lithium intercalates "
                        "from one side causing swelling.  Tests "
                        "concentration-dependent OCP, S2I interface "
                        "coupling, and structural deformation."
                    ),
                    "template_variant": "monolithic_elch_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "monolithic_elch_3d",
                "description": (
                    "3-D monolithic SSI with electrochemistry: two electrode "
                    "blocks with Butler-Volmer S2I interface.  "
                    "MAT_MultiplicativeSplitDefgradElastHyper for structure, "
                    "MAT_electrode for scalar transport, UMFPACK solver."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "monolithic_elch_3d") -> str:
        templates = {
            "monolithic_elch_3d": self._template_monolithic_elch_3d,
        }
        if variant == "default":
            variant = "monolithic_elch_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_monolithic_elch_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # Written in the input grammar of 4C before 2026.3.0 (the 2026.2.0 release
            # included); 4C 2026.3.0 renamed some of the keys below. The decks openPASO runs are served in the grammar of the
            # 4C it finds.
            # ---------------------------------------------------------------
            # 3-D Monolithic Structure-Scalar Interaction (Electrochemistry)
            #
            # Two electrode blocks with scatra-scatra interface (S2I)
            # coupling using Butler-Volmer kinetics.  The structural field
            # deforms due to lithium intercalation swelling.
            #
            # Mesh: requires an exodus file with:
            #   element_block 1 = left electrode (HEX8)
            #   element_block 2 = right electrode (HEX8)
            #   node_set 1 = left structural Dirichlet (fixed face)
            #   node_set 2 = right structural Dirichlet (prescribed displacement)
            #   node_set 3 = S2I interface (left side)
            #   node_set 4 = S2I interface (right side)
            #   node_set 5 = potential Dirichlet BC face
            #   node_set 6 = potential Neumann BC face
            # ---------------------------------------------------------------
            TITLE:
              - "3-D SSI with electrochemistry -- generated template"
            PROBLEM TYPE:
              PROBLEMTYPE: "Structure_Scalar_Interaction"
            IO:
              STDOUTEVERY: <stdout_interval>
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/STRUCTURE:
              OUTPUT_STRUCTURE: true
              DISPLACEMENT: true

            # == Structure =====================================================
            STRUCTURAL DYNAMIC:
              DYNAMICTYPE: "OneStepTheta"
              LINEAR_SOLVER: 1

            # == Scalar Transport (Electrochemistry) ===========================
            SCALAR TRANSPORT DYNAMIC:
              SOLVERTYPE: "nonlinear"
              VELOCITYFIELD: "Navier_Stokes"
              INITIALFIELD: "field_by_condition"
              CONVFORM: "conservative"
              SKIPINITDER: true
              LINEAR_SOLVER: 1
            SCALAR TRANSPORT DYNAMIC/STABILIZATION:
              STABTYPE: "no_stabilization"
              DEFINITION_TAU: "Zero"
              EVALUATION_TAU: "integration_point"
              EVALUATION_MAT: "integration_point"
            SCALAR TRANSPORT DYNAMIC/S2I COUPLING:
              COUPLINGTYPE: "MatchingNodes"

            # == Electrochemistry control ======================================
            ELCH CONTROL:
              EQUPOT: "<electroneutrality_method>"
              DIFFCOND_FORMULATION: <diffcond_flag>
              COUPLE_BOUNDARY_FLUXES: <couple_boundary_fluxes_flag>

            # == SSI coupling ==================================================
            SSI CONTROL:
              RESTARTEVERY: <restart_interval>
              NUMSTEP: <number_of_steps>
              TIMESTEP: <timestep>
              RESULTSEVERY: <results_output_interval>
              COUPALGO: ssi_Monolithic
              SCATRATIMINTTYPE: "Elch"
            SSI CONTROL/MONOLITHIC:
              ABSTOLRES: <absolute_residual_tolerance>
              LINEAR_SOLVER: 1
              MATRIXTYPE: "sparse"
            SSI CONTROL/ELCH:
              INITPOTCALC: <compute_initial_potential>

            # == Solver ========================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "direct_solver"

            # == Materials =====================================================
            MATERIALS:
              # Structural material: multiplicative decomposition F = F_e * F_i
              - MAT: 1
                MAT_MultiplicativeSplitDefgradElastHyper:
                  NUMMATEL: 1
                  MATIDSEL: [<elastic_sub_material_id>]
                  NUMFACINEL: 1
                  INELDEFGRADFACIDS: [<inelastic_factor_material_id>]
                  DENS: <structural_density>
              # Second block structural material (same formulation)
              - MAT: 2
                MAT_MultiplicativeSplitDefgradElastHyper:
                  NUMMATEL: 1
                  MATIDSEL: [<elastic_sub_material_id>]
                  NUMFACINEL: 1
                  INELDEFGRADFACIDS: [<inelastic_factor_material_id>]
                  DENS: <structural_density>
              # Elastic sub-material (St. Venant-Kirchhoff)
              - MAT: 3
                ELAST_CoupSVK:
                  YOUNG: <Young_modulus>
                  NUE: <Poisson_ratio>
              # Inelastic growth factor (no growth placeholder)
              - MAT: 4
                MAT_InelasticDefgradNoGrowth: {}
              # Electrode material (scalar transport)
              - MAT: 5
                MAT_electrode:
                  DIFF_COEF_CONC_DEP_FUNCT: <diff_coef_concentration_function>
                  DIFF_COEF_TEMP_SCALE_FUNCT: <diff_coef_temperature_function>
                  COND_CONC_DEP_FUNCT: <cond_concentration_function>
                  COND_TEMP_SCALE_FUNCT: <cond_temperature_function>
                  DIFF_PARA_NUM: <num_diffusion_parameters>
                  DIFF_PARA: [<diffusion_coefficient>]
                  COND_PARA_NUM: <num_conductivity_parameters>
                  COND_PARA: [<electronic_conductivity>]
                  C_MAX: <max_lithium_concentration>
                  CHI_MAX: <max_stoichiometry>
                  # OCP_MODEL requires BOTH X_MIN and X_MAX -- omitting
                  # X_MAX fails the whole MATERIALS section at parse.
                  # Use -1 for both to switch the range check off.
                  # The sub-group is one of Function, Redlich-Kister or
                  # Taralov.
                  OCP_MODEL:
                    Function:
                      OCP_FUNCT_NUM: <ocp_function_id>
                    X_MIN: <ocp_x_min>
                    X_MAX: <ocp_x_max>

            # == Boundary Conditions ===========================================

            # Structural Dirichlet: fixed face
            DESIGN SURF DIRICH CONDITIONS:
              - E: <fixed_face_id>
                NUMDOF: 3
                ONOFF: [1, 1, 1]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]
              # Structural: prescribed displacement face
              - E: <loaded_face_id>
                NUMDOF: 3
                ONOFF: [<active_displacement_dofs>]
                VAL: [<prescribed_displacement_values>]
                FUNCT: [<displacement_time_functions>]

            # Potential Dirichlet BC
            DESIGN SURF TRANSPORT DIRICH CONDITIONS:
              - E: <potential_dirichlet_face_id>
                NUMDOF: <num_scalar_dofs>
                ONOFF: [<active_scalar_dofs>]
                VAL: [<potential_values>]
                FUNCT: [<potential_time_functions>]

            # S2I interface conditions.
            # There is NO "DESIGN SURF S2I COUPLING CONDITIONS" section in
            # 4C. The real S2I family is DESIGN S2I KINETICS <GEOM>
            # CONDITIONS (the physics) and DESIGN S2I MESHTYING <GEOM>
            # CONDITIONS (the slave/master pairing), with the geometry word
            # LAST: SURF, LINE or POINT. INTERFACE_SIDE is capitalised
            # "Slave" / "Master".
            #
            # Butler-Volmer kinetics on the two interface faces. The slave
            # side carries the whole model; the master side is just E,
            # ConditionID and INTERFACE_SIDE, with no KINETIC_MODEL and no
            # kinetic parameters at all. The two sides are paired by their
            # shared ConditionID.
            DESIGN S2I KINETICS SURF CONDITIONS:
              - E: <s2i_face_left>
                ConditionID: 0
                INTERFACE_SIDE: "Slave"
                KINETIC_MODEL: "Butler-Volmer"
                NUMSCAL: <num_scalars>
                STOICHIOMETRIES: [<stoichiometry>]
                E-: 1
                K_R: <butler_volmer_rate_constant>
                ALPHA_A: <anodic_transfer_coefficient>
                ALPHA_C: <cathodic_transfer_coefficient>
                IS_PSEUDO_CONTACT: false
              - E: <s2i_face_right>
                ConditionID: 0
                INTERFACE_SIDE: "Master"

            # NOT optional: every S2I kinetics condition needs a matching
            # meshtying condition, or 4C aborts in ScaTraUtils::
            # check_consistency_of_s2_i_conditions with "For each
            # 'S2IKinetics' or 'S2ISCLCoupling' condition a corresponding
            # 'S2IMeshtying' or 'S2INoEvaluation' condition has to be
            # defined!". For an SSI problem use the SSI-specific section
            # below (it meshties the structure side too); plain scatra/elch
            # decks use DESIGN S2I MESHTYING SURF CONDITIONS instead. The
            # pairing is by the shared ConditionID / S2I_KINETICS_ID.
            DESIGN SSI INTERFACE MESHTYING SURF CONDITIONS:
              - E: <s2i_face_left>
                ConditionID: 0
                INTERFACE_SIDE: "Slave"
                S2I_KINETICS_ID: 0
              - E: <s2i_face_right>
                ConditionID: 0
                INTERFACE_SIDE: "Master"
                S2I_KINETICS_ID: 0

            # OCP function
            FUNCT<ocp_function_id>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<ocp_expression>"

            # == Geometry ======================================================
            STRUCTURE GEOMETRY:
              FILE: "<mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  SOLIDSCATRA:
                    HEX8:
                      MAT: 1
                      KINEM: <kinematics>
                      TYPE: Undefined
                - ID: 2
                  SOLIDSCATRA:
                    HEX8:
                      MAT: 2
                      KINEM: <kinematics>
                      TYPE: Undefined

            RESULT DESCRIPTION:
              - SCATRA:
                  DIS: "scatra"
                  NODE: <result_node_id>
                  QUANTITY: "phi"
                  VALUE: <expected_concentration>
                  TOLERANCE: <result_tolerance>
              - STRUCTURE:
                  DIS: "structure"
                  NODE: <result_node_id>
                  QUANTITY: "dispx"
                  VALUE: <expected_displacement>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check Young's modulus
        young = params.get("YOUNG")
        if young is not None:
            try:
                e = float(young)
                if e <= 0:
                    issues.append(f"YOUNG must be > 0, got {e}.")
            except (TypeError, ValueError):
                issues.append(
                    f"YOUNG must be a positive number, got {young!r}."
                )

        # Check Poisson's ratio
        nue = params.get("NUE")
        if nue is not None:
            try:
                nu = float(nue)
                if nu <= 0 or nu >= 0.5:
                    issues.append(
                        f"NUE must be in (0, 0.5), got {nu}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"NUE must be a number in (0, 0.5), got {nue!r}."
                )

        # Check C_MAX
        c_max = params.get("C_MAX")
        if c_max is not None:
            try:
                cm = float(c_max)
                if cm <= 0:
                    issues.append(
                        f"C_MAX (max concentration) must be > 0, got {cm}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"C_MAX must be a positive number, got {c_max!r}."
                )

        # Check coupling algorithm.  These are the eight values 4C's
        # SSI::SolutionSchemeOverFields enum actually accepts -- note there
        # is NO underscore between "IterStagg" and "FixedRel"/"Aitken".
        coupalgo_values = (
            "ssi_OneWay_ScatraToSolid",
            "ssi_OneWay_SolidToScatra",
            "ssi_IterStagg",
            "ssi_IterStaggFixedRel_ScatraToSolid",
            "ssi_IterStaggFixedRel_SolidToScatra",
            "ssi_IterStaggAitken_ScatraToSolid",
            "ssi_IterStaggAitken_SolidToScatra",
            "ssi_Monolithic",
        )
        coupalgo = params.get("COUPALGO")
        if coupalgo is not None and coupalgo not in coupalgo_values:
            issues.append(
                f"COUPALGO must be one of {', '.join(coupalgo_values)}; "
                f"got {coupalgo!r}."
            )

        # Check SCATRATIMINTTYPE
        scatra_type = params.get("SCATRATIMINTTYPE")
        if scatra_type is not None and scatra_type not in (
            "Standard", "Elch", "Cardiac_Monodomain",
        ):
            issues.append(
                f"SCATRATIMINTTYPE must be 'Standard', 'Elch' or "
                f"'Cardiac_Monodomain', got {scatra_type!r}."
            )

        # Check density
        dens = params.get("DENS")
        if dens is not None:
            try:
                d = float(dens)
                if d <= 0:
                    issues.append(f"DENS must be > 0, got {d}.")
            except (TypeError, ValueError):
                issues.append(
                    f"DENS must be a positive number, got {dens!r}."
                )

        return issues
