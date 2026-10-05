"""Multiscale (FE-squared) generator for 4C.

Covers computational homogenisation using nested FE simulations (FE^2).
A macroscopic structural problem is solved with standard finite elements,
but the constitutive response at each Gauss point is computed by solving
a microscale boundary value problem (Representative Volume Element, RVE)
instead of using a closed-form material law.  This enables capturing
complex microstructural effects (heterogeneity, damage, plasticity) at
the macroscale without explicit homogenisation assumptions.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class MultiscaleGenerator(BaseGenerator):
    """Generator for FE^2 nested multiscale problems in 4C."""

    module_key = "multiscale"
    display_name = "Multiscale (FE-squared Computational Homogenisation)"
    problem_type = "Structure"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The FE^2 (FE-squared) multiscale module performs "
                "computational homogenisation by nesting a microscale "
                "FE simulation within each macroscale Gauss point.  "
                "At the macro level, a standard structural problem is "
                "solved.  At each integration point, instead of "
                "evaluating a closed-form constitutive law, a "
                "Representative Volume Element (RVE) boundary value "
                "problem is solved.  The macro deformation gradient is "
                "imposed on the RVE via boundary conditions, and the "
                "homogenised stress and tangent are returned to the "
                "macro solver.  The PROBLEM TYPE is 'Structure' with a "
                "special MAT_Struct_Multiscale material that references "
                "a micro-scale input file.  The dynamics section is "
                "STRUCTURAL DYNAMIC.  The micro-scale input file is a "
                "complete 4C input file describing the RVE (geometry, "
                "materials, its OWN SOLVER section, and a MICROSCALE "
                "CONDITIONS section marking the RVE boundary surfaces).  "
                "The macro material uses MAT_Struct_Multiscale with "
                "exactly two keys that matter: MICROFILE (path to the "
                "micro input file) and MICRODIS_NUM (which micro problem "
                "instance this material belongs to).  There is no "
                "macro-side solver id and no macro-side density: the "
                "micro solver is picked up from the MICRO file's own "
                "STRUCTURAL DYNAMIC/LINEAR_SOLVER, and the macroscopic "
                "density is computed by the homogenisation procedure."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "STRUCTURAL DYNAMIC",
                "SOLVER 1",
                "MATERIALS",
                "STRUCTURE GEOMETRY",
            ],
            "optional_sections": [
                "STRUCTURAL DYNAMIC/GENALPHA",
                "IO",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/STRUCTURE",
                "RESULT DESCRIPTION",
            ],
            "materials": {
                "MAT_Struct_Multiscale": {
                    "description": (
                        "Macro-scale material that triggers FE^2 "
                        "homogenisation.  Each Gauss point evaluates the "
                        "constitutive response by solving a microscale "
                        "RVE problem defined in a separate input file."
                    ),
                    "parameters": {
                        "MICROFILE": {
                            "description": (
                                "Path to the micro-scale 4C input file "
                                "that defines the RVE problem.  Resolved "
                                "relative to the MACRO input file's "
                                "directory.  The key is MICROFILE -- "
                                "there is no MICRO_INPUT_FILE."
                            ),
                            "range": "valid file path (default 'filename.dat')",
                        },
                        "MICRODIS_NUM": {
                            "description": (
                                "Number of the micro-scale DISCRETISATION "
                                "this material feeds, i.e. which "
                                "Global::Problem instance the RVE is read "
                                "into.  Two materials that name the same "
                                "MICROFILE but different MICRODIS_NUM get "
                                "two independent micro problems.  This is "
                                "NOT a solver id -- the micro solver is "
                                "the LINEAR_SOLVER of the MICRO file's own "
                                "STRUCTURAL DYNAMIC section, looked up in "
                                "the MICRO file's own SOLVER sections."
                            ),
                            "range": ">= 1 (upstream uses 1, 2, ...)",
                        },
                        "INITVOL": {
                            "description": (
                                "Initial volume of the RVE.  Optional, "
                                "default 0.0, in which case it is computed."
                            ),
                            "range": ">= 0",
                        },
                        "RUNTIMEOUTPUT_GP": {
                            "description": (
                                "Which Gauss points of the macro element "
                                "get micro-scale runtime output written. "
                                "Upstream values: all, none."
                            ),
                            "range": "all | none",
                        },
                    },
                    "not_parameters": (
                        "MAT_Struct_Multiscale takes NO DENS.  The "
                        "macroscopic density is produced by the "
                        "homogenisation procedure "
                        "(MultiScale::MicroStatic asserts 'Density "
                        "determined from homogenization procedure must be "
                        "larger than zero!'), so it comes from the MICRO "
                        "materials.  Writing DENS here is a MATERIALS "
                        "parse abort."
                    ),
                },
                "Micro-scale materials": {
                    "description": (
                        "Materials for the RVE are defined in the micro "
                        "input file.  Any standard 4C material can be "
                        "used (MAT_ElastHyper, MAT_Struct_StVenantKirchhoff, "
                        "damage models, plasticity models, etc.)."
                    ),
                },
            },
            "solver": {
                "macro_solver": {
                    "type": "UMFPACK or Belos",
                    "notes": (
                        "Macro-scale structural solver.  Direct solver "
                        "works for small problems; iterative needed for "
                        "large macro meshes."
                    ),
                },
                "micro_solver": {
                    "type": "UMFPACK",
                    "notes": (
                        "Micro-scale RVE solver.  Direct solver is "
                        "recommended since each RVE is typically small.  "
                        "It is defined in the MICRO input file, NOT the "
                        "macro one: MultiScale::MicroStatic reads "
                        "LINEAR_SOLVER from the micro file's STRUCTURAL "
                        "DYNAMIC and resolves it against the micro file's "
                        "own SOLVER sections "
                        "(stru_multi/4C_stru_multi_microstatic.cpp).  "
                        "MAXITER, TOLRES, TOLDISP, PREDICT, NORM_DISP, "
                        "NORM_RESF, NORMCOMBI_RESFDISP, ITERNORM, "
                        "ADAPTCONV and ADAPTCONV_BETTER are likewise taken "
                        "from the MICRO file; only TIMESTEP, NUMSTEP and "
                        "RESTARTEVERY are taken from the MACRO file, so "
                        "those three must agree."
                    ),
                },
            },
            "rve_setup": {
                "geometry": (
                    "The RVE is typically a unit cell of the "
                    "microstructure, a cube or rectangular box whose "
                    "outer surfaces are all listed in MICROSCALE "
                    "CONDITIONS."
                ),
                "boundary_conditions": (
                    "4C's RVE boundary is the MICROSCALE CONDITIONS "
                    "section of the MICRO input file -- a surface "
                    "condition whose only field is E, listing every "
                    "boundary surface (upstream lists all six faces of "
                    "the cube).  4C then toggles EVERY dof of those "
                    "nodes to Dirichlet "
                    "(MultiScale::MicroStatic::determine_toggle) and "
                    "prescribes them from the macro deformation "
                    "gradient.  That is the linear-displacement "
                    "(Taylor/Dirichlet) RVE boundary condition, not a "
                    "periodic one; there is no PBC route for FE^2 in 4C "
                    "and no tied-DOF setup to build."
                ),
                "size": (
                    "The RVE must be statistically representative of "
                    "the microstructure.  Too small -> artificial size "
                    "effects.  Too large -> excessive computation cost "
                    "(each macro Gauss point solves one RVE)."
                ),
            },
            "pitfalls": [
                (
                    "[Performance] FE^2 is extremely expensive: each "
                    "macro Gauss point requires solving a full micro "
                    "FE problem.  For a macro mesh with N elements "
                    "and G Gauss points per element, N*G micro "
                    "problems are solved per macro Newton iteration. "
                    " Use coarse macro meshes and efficient micro "
                    "solvers. Signal: wall-clock per macro Newton "
                    "iteration grows linearly with N*G; a profiler "
                    "attributes most of the run time to the micro "
                    "Newton solve (MultiScale::MicroStatic::full_newton, "
                    "called per macro Gauss point from "
                    "Mat::MicroMaterialGP::perform_micro_simulation); "
                    "for a 10x10x10 macro mesh expect minutes per "
                    "iteration even with a trivial micro RVE. "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Input] The micro input file must be a complete, "
                    "valid 4C input file with its own geometry, "
                    "materials, SOLVER and STRUCTURAL DYNAMIC "
                    "sections.  It is NOT merged with the macro input "
                    "file, and its path is resolved relative to the "
                    "MACRO file's directory. Signal: a missing file is "
                    "reported plainly, \"Input file '<name>' does not "
                    "exist.\" from core/io/src/4C_io_input_file.cpp, "
                    "raised inside Global::read_micro_fields -- so the "
                    "stack trace is the only thing that tells you it "
                    "was the MICRO file and not the macro one. "
                    "(Verified by execution 2026-08-07.  An earlier "
                    "version quoted `failed to load micro input file "
                    "X` and `micro input file missing MATERIALS "
                    "section`; neither string is in the 4C binary.)"
                ),
                (
                    "[Input] There is no macro-side micro-solver key. "
                    "MAT_Struct_Multiscale's second parameter is "
                    "MICRODIS_NUM, the micro DISCRETISATION number, "
                    "and the micro solver is the LINEAR_SOLVER of the "
                    "MICRO file's own STRUCTURAL DYNAMIC section, "
                    "resolved against the MICRO file's own SOLVER "
                    "sections.  Signal: writing MICRO_SOLVER_ID (or "
                    "MICRO_INPUT_FILE, or DENS) in "
                    "MAT_Struct_Multiscale is a hard parse abort, "
                    "\"Failed to match specification in section "
                    "'MATERIALS'.\" from "
                    "global_data/4C_global_data_read.cpp followed by "
                    "'Could not match this input' and the echoed "
                    "material block.  The whole legal key set is "
                    "MICROFILE, MICRODIS_NUM, INITVOL, "
                    "RUNTIMEOUTPUT_GP. (Verified by execution "
                    "2026-08-07 on sohex8_multiscale_macro; the "
                    "earlier `MICRO_SOLVER_ID X not found among macro "
                    "SOLVER definitions` does not exist in 4C, and "
                    "neither does `null pointer to micro Belos "
                    "solver`.)"
                ),
                (
                    "[Input] Every RVE boundary surface must be listed "
                    "in a MICROSCALE CONDITIONS section of the MICRO "
                    "file -- that section IS the homogenisation "
                    "boundary, and 4C prescribes all its dofs from the "
                    "macro deformation gradient. Signal: omitting it "
                    "does not warn and does not abort cleanly.  The "
                    "run parses, builds both scales, and takes a "
                    "SEGMENTATION FAULT (signal 11, exit 139) inside "
                    "Solid::ModelEvaluator::Structure::"
                    "initialize_inertia_and_damping -> "
                    "Core::FE::Discretization::evaluate, with no PROC "
                    "0 ERROR banner and no mention of microscale, "
                    "boundary or homogenisation anywhere in the "
                    "output. (Verified by execution 2026-08-07 by "
                    "deleting MICROSCALE CONDITIONS from "
                    "sohex8_multiscale_micro.mat.4C.yaml.)"
                ),
                (
                    "[Numerical] 4C's RVE boundary is "
                    "linear-displacement, not periodic: MICROSCALE "
                    "CONDITIONS toggles every dof on the listed "
                    "surfaces to Dirichlet.  A Taylor/Dirichlet RVE "
                    "is known to OVER-constrain and gives a stiffer "
                    "homogenised response than a periodic one, so "
                    "treat the FE^2 answer as an upper bound on "
                    "stiffness and check RVE-size convergence rather "
                    "than reaching for a PBC option. Signal: none "
                    "from 4C -- there is no periodic-BC route for "
                    "FE^2 to compare against and nothing warns.  "
                    "Measure it by enlarging the RVE: a "
                    "Dirichlet-bounded homogenised stiffness falls "
                    "monotonically with RVE size, which is the "
                    "diagnostic. (Source-verified 2026-08-07 in "
                    "stru_multi/4C_stru_multi_microstatic_service.cpp; "
                    "the earlier claim that 'the RVE boundary "
                    "conditions must be periodic' described an option "
                    "4C does not have.)"
                ),
                (
                    "[Input] Macro PROBLEM TYPE is "
                    "'Structure' (NOT a dedicated multiscale "
                    "type). The multiscale behaviour is "
                    "activated purely through the "
                    "MAT_Struct_Multiscale material. Signal: "
                    "writing PROBLEMTYPE: 'Multiscale' "
                    "raises 'unknown problem type' — there "
                    "is no such enum. Use PROBLEMTYPE: "
                    "Structure with at least one element "
                    "assigned a MAT_Struct_Multiscale "
                    "material referencing a micro input "
                    "file. (Audit 2026-06-02.)"
                ),
                (
                    "[Performance] MPI parallelism interacts "
                    "with multiscale: macro problem "
                    "distributed across processors, each "
                    "processor's Gauss points solve their "
                    "RVEs INDEPENDENTLY. Signal: imbalanced "
                    "macro decomposition (RVEs differ in "
                    "solve cost by 5-10x) gives wall-clock "
                    "dominated by the slowest rank's RVE "
                    "queue — 4C's load-balancer cannot "
                    "redistribute mid-step. Pre-balance "
                    "the macro mesh based on expected RVE "
                    "complexity. (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] Convergence at the macro level "
                    "depends on the quality of the micro tangent -- "
                    "but there is NO INPUT KEY for it.  4C has exactly "
                    "one route: MicroStatic::static_homogenization "
                    "condenses the micro stiffness onto the RVE "
                    "boundary and returns the consistent (algorithmic) "
                    "cmat.  You cannot select it, degrade it or "
                    "replace it with numerical differentiation.  So if "
                    "macro Newton converges at a linear rate, the "
                    "tangent is not the knob: look at the MICRO file's "
                    "own MAXITER / TOLRES / TOLDISP (an under-"
                    "converged RVE returns a cmat that is not the "
                    "tangent of what it actually solved) and at the "
                    "macro NORMCOMBI_RESFDISP. Signal: there is no "
                    "diagnostic to grep for -- MicroStatic::"
                    "static_homogenization returns its cmat "
                    "unconditionally and MicroStatic::full_newton "
                    "reports its own convergence only into the MICRO "
                    "file's output, so a loosened TOLRES there is "
                    "invisible in the macro deck and the only visible "
                    "symptom is the macro NOX residual falling "
                    "linearly. (Corrected 2026-08-07: an earlier "
                    "version told you to switch 'MICRO_TANGENT to "
                    "ALGORITHMIC'.  there is no MICRO_TANGENT key -- "
                    "it is absent from 4C's own --parameters grammar "
                    "dump and from every file in src/ and "
                    "tests/input_files/.  MAT_Struct_Multiscale's "
                    "whole key set is MICROFILE, MICRODIS_NUM, "
                    "INITVOL, RUNTIMEOUTPUT_GP.)"
                ),
                (
                    "[Output] Macro-scale results show "
                    "homogenised stress/strain. Micro-scale "
                    "fields (damage, plasticity) are NOT "
                    "visible there. Signal: opening "
                    "the macro IO/RUNTIME VTK OUTPUT "
                    "STRUCTURE VTU in ParaView shows "
                    "smooth homogenised stress from "
                    "MAT_Struct_Multiscale; the micro-scale "
                    "fluctuations (e.g. damage localisation "
                    "in one RVE) are invisible in the macro "
                    "output. The micro output is a separate, "
                    "REAL mechanism with two knobs: "
                    "RUNTIMEOUTPUT_GP on the "
                    "MAT_Struct_Multiscale material chooses "
                    "which Gauss points write (all / none), "
                    "and the cadence and field selection come "
                    "from the MICRO file's own IO/RUNTIME VTK "
                    "OUTPUT (INTERVAL_STEPS) and IO/RUNTIME "
                    "VTK OUTPUT/STRUCTURE (DISPLACEMENT, "
                    "STRESS_STRAIN, ELEMENT_OWNER, "
                    "ELEMENT_MAT_ID) -- not from the macro "
                    "file's. GAUSS_POINT_DATA_OUTPUT_TYPE must "
                    "stay 'none' there; anything else asserts "
                    "'Gauss point output not yet implemented "
                    "on micro scale.' The micro writer also "
                    "emits the homogenised tangent as a field "
                    "named tangent_stiffness_tensor_cmat. "
                    "(Source-verified 2026-08-07 in "
                    "stru_multi/4C_stru_multi_microstatic.cpp; "
                    "the earlier wording about a MULTISCALE "
                    "micro-output writer named no real setting.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "rve_composite_3d",
                    "description": (
                        "A 3-D composite structure (fiber-reinforced "
                        "matrix) where the micro RVE contains a fiber "
                        "inclusion in a matrix.  The macro problem is a "
                        "tensile test.  Tests the full FE^2 loop: "
                        "macro deformation -> RVE solve -> homogenised "
                        "stress/tangent."
                    ),
                    "template_variant": "fe2_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "fe2_3d",
                "description": (
                    "3-D FE^2 multiscale: macro structural problem "
                    "with MAT_Struct_Multiscale material referencing "
                    "a micro RVE input file.  SOLID HEX8 elements, "
                    "UMFPACK solvers for both macro and micro."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "fe2_3d") -> str:
        templates = {
            "fe2_3d": self._template_fe2_3d,
        }
        if variant == "default":
            variant = "fe2_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_fe2_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D FE^2 Multiscale (Macro Input File)
            #
            # Macro-scale structural problem where each Gauss point
            # evaluates its constitutive response by solving a micro-scale
            # RVE (Representative Volume Element) problem.
            #
            # This file defines the MACRO problem.  A separate file defines
            # the MICRO (RVE) problem.
            #
            # Macro mesh: "macro.e" with
            #   element_block 1 = macro structure (HEX8)
            #   node_set 1 = fixed face
            #   node_set 2 = loaded face
            # ---------------------------------------------------------------
            TITLE:
              - "3-D FE^2 multiscale (macro) -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Structure"
            IO:
              STDOUTEVERY: <stdout_interval>
              STRUCT_STRESS: "Cauchy"
              STRUCT_STRAIN: "GL"
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/STRUCTURE:
              OUTPUT_STRUCTURE: true
              DISPLACEMENT: true

            # == Structural dynamics (macro) ===================================
            STRUCTURAL DYNAMIC:
              DYNAMICTYPE: "Statics"
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              LINEAR_SOLVER: 1
              PREDICT: "ConstDisVelAcc"
              TOLRES: <macro_residual_tolerance>
              TOLDISP: <macro_displacement_tolerance>
              RESULTSEVERY: <results_output_interval>

            # == Solver (MACRO only) ===========================================
            # The MICRO solver is NOT declared here.  It is the
            # LINEAR_SOLVER of the MICRO file's own STRUCTURAL DYNAMIC
            # section, resolved against the MICRO file's own SOLVER
            # sections.  There is no MICRO_SOLVER_ID anywhere in 4C.
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "macro_solver"

            # == Materials =====================================================
            MATERIALS:
              # Multiscale material: triggers FE^2.
              # The complete key set is MICROFILE, MICRODIS_NUM, INITVOL,
              # RUNTIMEOUTPUT_GP.  No DENS -- the macroscopic density is
              # produced by homogenisation from the MICRO materials.
              - MAT: 1
                MAT_Struct_Multiscale:
                  MICROFILE: "<micro_input_file_path>"
                  MICRODIS_NUM: 1
                  RUNTIMEOUTPUT_GP: all

            # == Boundary Conditions ===========================================
            # ONE block per section name: a section repeated at top level is
            # a hard parse abort, "Section '<name>' is defined more than
            # once." Put every Dirichlet surface in this single list.
            DESIGN SURF DIRICH CONDITIONS:
              # Fixed face
              - E: <fixed_face_id>
                NUMDOF: 3
                ONOFF: [1, 1, 1]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]
              # Load: prescribed displacement on loaded face
              - E: <loaded_face_id>
                NUMDOF: 3
                ONOFF: [<dof1_fix>, <dof2_fix>, <dof3_fix>]
                VAL: [<prescribed_disp_1>, <prescribed_disp_2>, <prescribed_disp_3>]
                FUNCT: [<load_ramp_function>, <load_ramp_function>, <load_ramp_function>]

            # Load ramp
            FUNCT<load_ramp_function>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<load_ramp_expression>"

            # == Geometry ======================================================
            STRUCTURE GEOMETRY:
              FILE: "<macro_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  SOLID:
                    HEX8:
                      MAT: 1
                      KINEM: <kinematics>

            RESULT DESCRIPTION:
              - STRUCTURE:
                  DIS: "structure"
                  NODE: <result_node_id>
                  QUANTITY: "dispx"
                  VALUE: <expected_displacement>
                  TOLERANCE: <result_tolerance>

            # ---------------------------------------------------------------
            # NOTE: The micro-scale RVE input file is SEPARATE.
            # It MUST define:
            #   - PROBLEM TYPE: PROBLEMTYPE: "Structure"
            #   - RVE geometry (NODE COORDS + STRUCTURE ELEMENTS, or a
            #     STRUCTURE GEOMETRY block) plus its D*-NODE TOPOLOGY
            #   - Micro-scale materials (e.g. fiber + matrix) -- these are
            #     what supply the density, since the macro material has none
            #   - MICROSCALE CONDITIONS listing EVERY RVE boundary surface,
            #     e.g. for a cube:
            #         MICROSCALE CONDITIONS:
            #           - E: 1
            #           - E: 2
            #           - E: 3
            #           - E: 4
            #           - E: 5
            #           - E: 6
            #     4C prescribes all dofs of those nodes from the macro
            #     deformation gradient.  Omitting the section SEGFAULTS with
            #     no message.  Do NOT try to write periodic BCs instead --
            #     4C's FE^2 RVE boundary is linear-displacement only.
            #   - Its OWN SOLVER section and its own STRUCTURAL DYNAMIC with
            #     LINEAR_SOLVER pointing at it.  TIMESTEP, NUMSTEP and
            #     RESTARTEVERY are taken from the MACRO file, so keep them
            #     consistent.
            #   - Its own IO/RUNTIME VTK OUTPUT (INTERVAL_STEPS) and
            #     IO/RUNTIME VTK OUTPUT/STRUCTURE: these control the MICRO
            #     output, and GAUSS_POINT_DATA_OUTPUT_TYPE must stay 'none'.
            # The micro file path is specified in MICROFILE above and is
            # resolved relative to THIS file's directory.
            # ---------------------------------------------------------------
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check micro input file
        micro_file = params.get("MICROFILE")
        if micro_file is not None:
            if not micro_file or micro_file == "":
                issues.append(
                    "MICROFILE must be a non-empty file path."
                )
        if params.get("MICRO_INPUT_FILE") is not None:
            issues.append(
                "MICRO_INPUT_FILE is not a 4C key.  MAT_Struct_Multiscale "
                "names the micro input file with MICROFILE."
            )

        # Check micro discretisation number
        microdis = params.get("MICRODIS_NUM")
        if microdis is not None:
            try:
                num = int(microdis)
                if num < 1:
                    issues.append(
                        f"MICRODIS_NUM must be >= 1, got {num}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"MICRODIS_NUM must be a positive integer, "
                    f"got {microdis!r}."
                )
        if params.get("MICRO_SOLVER_ID") is not None:
            issues.append(
                "MICRO_SOLVER_ID is not a 4C key.  There is no macro-side "
                "micro-solver id: the micro solver is the LINEAR_SOLVER of "
                "the MICRO input file's own STRUCTURAL DYNAMIC section.  "
                "MAT_Struct_Multiscale's second parameter is MICRODIS_NUM, "
                "the micro discretisation number."
            )

        # Check RVE initial volume
        initvol = params.get("INITVOL")
        if initvol is not None:
            try:
                vol = float(initvol)
                if vol < 0:
                    issues.append(
                        f"INITVOL must be >= 0, got {vol}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"INITVOL must be a non-negative number, "
                    f"got {initvol!r}."
                )

        # Density is not a macro-material parameter at all
        density = params.get("DENS") or params.get("homogenised_density")
        if density is not None:
            issues.append(
                "MAT_Struct_Multiscale takes no DENS.  The macroscopic "
                "density is computed by the homogenisation procedure from "
                "the MICRO materials; writing DENS here aborts the "
                "MATERIALS section at parse."
            )

        # Check tolerances
        for tol_key in ("TOLRES", "TOLDISP",
                         "macro_residual_tolerance",
                         "macro_displacement_tolerance"):
            tol = params.get(tol_key)
            if tol is not None:
                try:
                    t = float(tol)
                    if t <= 0:
                        issues.append(
                            f"{tol_key} must be > 0, got {t}."
                        )
                except (TypeError, ValueError):
                    issues.append(
                        f"{tol_key} must be a positive number, "
                        f"got {tol!r}."
                    )

        return issues
