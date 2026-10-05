"""Structure-Scalar-Thermo Interaction (SSTI) generator for 4C.

Covers monolithic three-field coupling of structural mechanics, scalar
transport, and thermal fields.  This extends SSI by adding a thermal field
that influences both the structural (thermal expansion) and scalar transport
(temperature-dependent diffusion) fields.  Key application: battery cell
modeling where mechanical, electrochemical, and thermal effects interact.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class SSTIGenerator(BaseGenerator):
    """Generator for Structure-Scalar-Thermo Interaction problems in 4C."""

    module_key = "ssti"
    display_name = "Structure-Scalar-Thermo Interaction (SSTI)"
    problem_type = "Structure_Scalar_Thermo_Interaction"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Structure-Scalar-Thermo Interaction (SSTI) is a three-field "
                "coupled problem combining structural mechanics, scalar "
                "transport (optionally with electrochemistry), and thermal "
                "analysis.  The thermal field drives thermal expansion in "
                "the structure and modifies transport coefficients in the "
                "scalar field.  The scalar field (e.g. lithium concentration) "
                "causes volumetric swelling in the structure.  Heat "
                "generation from electrochemical reactions and mechanical "
                "dissipation feeds back into the thermal field.  The "
                "PROBLEM TYPE is 'Structure_Scalar_Thermo_Interaction'.  "
                "This problem requires dynamics sections for all three "
                "fields: STRUCTURAL DYNAMIC, SCALAR TRANSPORT DYNAMIC, "
                "and THERMAL DYNAMIC, plus the coupling section "
                "SSTI CONTROL.  Elements typically use SOLIDSCATRA to "
                "carry both displacement and scalar DOFs, with mesh "
                "cloning for the thermal field."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "STRUCTURAL DYNAMIC",
                "SCALAR TRANSPORT DYNAMIC",
                "THERMAL DYNAMIC",
                "SSTI CONTROL",
                "SSTI CONTROL/MONOLITHIC",
                "SSI CONTROL",
                "SSI CONTROL/MONOLITHIC",
                "TSI DYNAMIC",
                "SOLVER 1",
                "SOLVER 2",
                "MATERIALS",
                "CLONING MATERIAL MAP",
            ],
            "optional_sections": [
                "ELCH CONTROL",
                "SSI CONTROL/ELCH",
                "SCALAR TRANSPORT DYNAMIC/STABILIZATION",
                "SCALAR TRANSPORT DYNAMIC/S2I COUPLING",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/STRUCTURE",
            ],
            "materials": {
                "MAT_Struct_ThermoStVenantK": {
                    "description": (
                        "Thermo-elastic St. Venant-Kirchhoff material for "
                        "the structural field.  Supports thermal expansion "
                        "and, on a 4C before 2026.3.0, can link a thermal "
                        "material via THERMOMAT (4C 2026.3.0 removed the key).  "
                        "Used in combination with scalar-dependent "
                        "inelastic growth for full SSTI coupling."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Young's modulus array",
                            "range": "> 0",
                        },
                        "NUE": {
                            "description": "Poisson's ratio",
                            "range": "(0, 0.5)",
                        },
                        "DENS": {
                            "description": "Mass density",
                            "range": "> 0",
                        },
                        "THEXPANS": {
                            "description": "Thermal expansion coefficient [1/K]",
                            "range": "> 0",
                        },
                        "INITTEMP": {
                            "description": "Reference temperature for zero thermal strain",
                            "range": "any (often 293 K)",
                        },
                        "THERMOMAT": {
                            "description": ("Material ID of the thermal material (MAT_Fourier). "
                                            "4C before 2026.3.0 only; 4C 2026.3.0 has no THERMOMAT"),
                            "range": "valid MAT ID",
                        },
                    },
                },
                "MAT_Fourier": {
                    "description": (
                        "Fourier heat conduction material for the thermal "
                        "field.  Defines volumetric heat capacity and "
                        "thermal conductivity."
                    ),
                    "parameters": {
                        "CAPA": {
                            "description": "Volumetric heat capacity [J/(m^3 K)]",
                            "range": "> 0",
                        },
                        "CONDUCT": {
                            "description": "Thermal conductivity [W/(m K)]",
                            "range": "> 0",
                        },
                    },
                },
                "MAT_electrode": {
                    "description": (
                        "Electrode material for the scalar transport field "
                        "when electrochemistry is active.  Defines "
                        "concentration-dependent diffusion, conductivity, "
                        "and open-circuit potential."
                    ),
                },
            },
            "solver": {
                "field_solvers": {
                    "type": "UMFPACK (direct)",
                    "notes": (
                        "Individual field solvers for structure, scalar "
                        "transport, and thermal fields."
                    ),
                },
                "monolithic_solver": {
                    "type": "UMFPACK or Belos with block preconditioner",
                    "notes": (
                        "The monolithic SSTI solver handles the coupled "
                        "three-field block system.  For small problems "
                        "UMFPACK is sufficient; for large problems use "
                        "iterative solvers with appropriate "
                        "block preconditioning."
                    ),
                },
            },
            "time_integration": {
                "SSTI CONTROL": (
                    "Controls the overall three-field coupling loop.  "
                    "NUMSTEP, TIMESTEP, MAXTIME define the global time "
                    "stepping.  COUPALGO selects the coupling strategy."
                ),
                "SSI CONTROL": (
                    "Sub-coupling between structure and scalar transport "
                    "within each SSTI step."
                ),
                "TSI DYNAMIC": (
                    "Sub-coupling between thermal and structural fields "
                    "within each SSTI step."
                ),
            },
            "pitfalls": [
                (
                    "[Input] SSTI clones TWICE -- structure "
                    "-> scatra, then scatra -> thermo -- so "
                    "the CLONING MATERIAL MAP needs BOTH hops "
                    "for EVERY material group, not two entries "
                    "in total. Signal: dropping one pairing "
                    "aborts with 'no matching material ID "
                    "(<id>) in map' from core/fem/src/general/"
                    "utils/4C_fem_general_utils_createdis.hpp, "
                    "exit 1.  The message names the SOURCE "
                    "material id and no field name at all, so "
                    "grepping the log for 'thermo' tells you "
                    "nothing. (Verified by execution "
                    "2026-08-06; 'cannot clone material for "
                    "<field>' does not exist in 4C, and the "
                    "'exactly two SRC/TAR entries' rule was "
                    "wrong.)"
                ),
                (
                    "[Input] Elements must be SOLIDSCATRA "
                    "(not plain SOLID) to carry BOTH "
                    "displacement and scalar DOFs; the "
                    "thermal DOF comes from mesh cloning. "
                    "Signal: you hit it in two stages.  With "
                    "the element's TYPE token still on the "
                    "line, SOLID fails to PARSE: \"After "
                    "parsing, the line still contains 'TYPE "
                    "...'\" from core/io/src/"
                    "4C_io_input_spec.cpp, because SOLID has "
                    "no TYPE key.  Remove the token so the "
                    "line parses and the real message appears, "
                    "from src/ssti/4C_ssti_utils.cpp: "
                    "'ScatraStructureCloneStrategy copies "
                    "scatra discretization from structure "
                    "discretization, but ... the ImplType is "
                    "set Undefined ... Use SOLIDSCATRA, "
                    "WALLSCATRA or SHELLSCATRA elements with "
                    "meaningful ImplType instead!'. (From 4C "
                    "2026.2.0 on the message no longer lists "
                    "WALLSCATRA; the element it calls "
                    "SHELLSCATRA is SHELL7PSCATRA in the input. "
                    "Verified "
                    "by execution 2026-08-06.  Two earlier "
                    "versions were wrong: 'no SCATRA "
                    "discretisation found' from a "
                    "'4C_tsi_factory.cpp' exists nowhere, and "
                    "'Unsupported solid element type!' from "
                    "tsi/4C_tsi_utils.cpp -- though a real "
                    "literal -- is never reached on this "
                    "path.)"
                ),
                (
                    "[Input] Do NOT try to match TIMESTEP "
                    "across the STRUCTURAL, SCALAR TRANSPORT "
                    "and THERMAL sections: SSTI has ONE time "
                    "step, SSTI CONTROL's, and it overrides "
                    "the per-field values silently. Signal: "
                    "deliberately inconsistent per-field "
                    "TIMESTEP and NUMSTEP values are accepted, "
                    "produce no warning, and leave the result "
                    "bit-for-bit unchanged; only SSTI "
                    "CONTROL's own TIMESTEP moves it.  The "
                    "risk is the opposite of the one "
                    "previously stated -- not splitting error, "
                    "but a per-field time step you set and "
                    "that never took effect. (Verified by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Input] ELCH CONTROL is required for an "
                    "electrochemical SSTI problem.  "
                    "SCATRATIMINTTYPE is NOT: it lives in SSTI "
                    "CONTROL (never in SSI CONTROL), defaults "
                    "to Elch, and Elch is its ONLY legal "
                    "value, so it cannot be got wrong -- 4C's "
                    "own electrochemical SSTI decks omit it. "
                    "Signal: without ELCH CONTROL, 'Invalid "
                    "type of closing equation for electric "
                    "potential!' from scatra_ele/"
                    "4C_scatra_ele_parameter_elch.cpp, exit 1 "
                    "before any result test.  Setting "
                    "SCATRATIMINTTYPE to anything else gives "
                    "\"'SCATRATIMINTTYPE' has wrong value, "
                    "possible values: Elch\"; putting it in an "
                    "SSI CONTROL block is simply inert. "
                    "(Verified by execution 2026-08-06; there "
                    "is no silent zero-BV-current mode.)"
                ),
                (
                    "[Input] SSTI CONTROL/MONOLITHIC's "
                    "LINEAR_SOLVER is required and is NOT "
                    "validated: it defaults to -1 and 4C then "
                    "looks for a section called 'SOLVER -1'. "
                    "Signal: omitting it kills the process "
                    "inside Teuchos, not 4C -- "
                    "'Teuchos::Exceptions::InvalidParameterName "
                    "... in the parameter (sub)list "
                    "\"ROOT->SOLVER -1\"', SIGABRT.  There is "
                    "no PROC 0 ERROR banner, no source file, "
                    "and neither 'LINEAR_SOLVER' nor 'SSTI' "
                    "appears anywhere in the output; the "
                    "defaulted -1 is the only clue.  A direct "
                    "solver is fine for test-sized problems "
                    "(4C's own SSTI decks use one); Belos "
                    "GMRES + a Teko block preconditioner is "
                    "for production meshes. (Verified by "
                    "execution 2026-08-06.)"
                ),
                (
                    "[Input] The growth reference temperature "
                    "must match the prescribed initial "
                    "temperature field -- and in SSTI it is "
                    "called RefTemp, on "
                    "MAT_InelasticDefgradLinTempIso.  It is "
                    "NOT INITTEMP; that parameter exists only "
                    "on TSI plastic / thermo-StVenant "
                    "materials and appears in no SSTI deck. "
                    "Signal: a mismatched RefTemp does not "
                    "merely bias the answer, it inverts the "
                    "growth tensor -- 'Determinante of growth "
                    "must not become negative' from "
                    "mat/4C_mat_inelastic_defgrad_factors.cpp, "
                    "exit 1. (Verified by execution "
                    "2026-08-06.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "battery_cell_3d",
                    "description": (
                        "A battery electrode undergoing lithium "
                        "intercalation with coupled thermal, mechanical, "
                        "and electrochemical effects.  Temperature affects "
                        "diffusion kinetics, intercalation causes swelling, "
                        "and Joule heating raises the temperature."
                    ),
                    "template_variant": "monolithic_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "monolithic_3d",
                "description": (
                    "3-D monolithic SSTI: coupled structure-scalar-thermo "
                    "problem.  SOLIDSCATRA HEX8 elements, thermo-elastic "
                    "structural material, Fourier thermal material, "
                    "electrode scalar material, UMFPACK solvers."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "monolithic_3d") -> str:
        templates = {
            "monolithic_3d": self._template_monolithic_3d,
        }
        if variant == "default":
            variant = "monolithic_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_monolithic_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # Written in the input grammar of 4C before 2026.3.0 (the 2026.2.0 release
            # included); 4C 2026.3.0 renamed some of the keys below. The decks openPASO runs are served in the grammar of the
            # 4C it finds.
            # ---------------------------------------------------------------
            # 3-D Monolithic Structure-Scalar-Thermo Interaction (SSTI)
            #
            # Three-field coupling: structural mechanics + scalar transport
            # (electrochemistry) + thermal.  The thermal field drives
            # thermal expansion and modifies transport coefficients.
            # Scalar concentration causes structural swelling.  Heat
            # generation from reactions feeds back into the thermal field.
            #
            # Elements: SOLIDSCATRA HEX8 (displacement + scalar DOFs;
            #           thermal DOF via mesh cloning)
            #
            # Mesh: requires an exodus file with:
            #   element_block 1 = domain (HEX8)
            #   node_set 1 = fixed structural face
            #   node_set 2 = scalar Dirichlet face
            #   node_set 3 = thermal Dirichlet face
            # ---------------------------------------------------------------
            TITLE:
              - "3-D SSTI (structure-scalar-thermo) -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Structure_Scalar_Thermo_Interaction"
            IO:
              STDOUTEVERY: <stdout_interval>
              STRUCT_STRESS: "2PK"
              STRUCT_STRAIN: "GL"
              THERM_HEATFLUX: "Initial"
              THERM_TEMPGRAD: "Initial"
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/STRUCTURE:
              OUTPUT_STRUCTURE: true
              DISPLACEMENT: true

            # == Structure =====================================================
            STRUCTURAL DYNAMIC:
              DYNAMICTYPE: "Statics"
              LINEAR_SOLVER: 1

            # == Scalar Transport ==============================================
            SCALAR TRANSPORT DYNAMIC:
              SOLVERTYPE: "nonlinear"
              VELOCITYFIELD: "Navier_Stokes"
              CONVFORM: "conservative"
              INITIALFIELD: "field_by_condition"
              SKIPINITDER: true
              LINEAR_SOLVER: 1
            SCALAR TRANSPORT DYNAMIC/STABILIZATION:
              STABTYPE: "no_stabilization"

            # == Thermal =======================================================
            THERMAL DYNAMIC:
              DYNAMICTYPE: Statics
              TIMESTEP: <thermal_timestep>
              NUMSTEP: <thermal_num_steps>
              LINEAR_SOLVER: 1
            THERMAL DYNAMIC/RUNTIME VTK OUTPUT:
              OUTPUT_THERMO: true
              TEMPERATURE: true

            # == SSI sub-coupling ==============================================
            SSI CONTROL:
              COUPALGO: ssi_Monolithic
              SCATRATIMINTTYPE: "<scatra_time_integration_type>"
            SSI CONTROL/MONOLITHIC:
              LINEAR_SOLVER: 2
              MATRIXTYPE: "sparse"

            # == TSI sub-coupling ==============================================
            TSI DYNAMIC:
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              TIMESTEP: <timestep>
              ITEMAX: <max_tsi_coupling_iterations>
              RESULTSEVERY: <results_output_interval>

            # == SSTI master coupling ==========================================
            SSTI CONTROL:
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              TIMESTEP: <timestep>
              RESULTSEVERY: <results_output_interval>
              RESTARTEVERY: <restart_interval>
              COUPALGO: "ssti_Monolithic"
            SSTI CONTROL/MONOLITHIC:
              LINEAR_SOLVER: 2

            # == Solvers =======================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "field_solver"
            SOLVER 2:
              SOLVER: "UMFPACK"
              NAME: "monolithic_solver"

            # == Materials =====================================================
            MATERIALS:
              # Thermo-elastic structural material
              - MAT: 1
                MAT_Struct_ThermoStVenantK:
                  YOUNGNUM: 1
                  YOUNG: [<Young_modulus>]
                  NUE: <Poisson_ratio>
                  DENS: <density>
                  THEXPANS: <thermal_expansion_coefficient>
                  INITTEMP: <reference_temperature>
              # Fourier heat conduction (thermal field)
              - MAT: 2
                MAT_Fourier:
                  CAPA: <volumetric_heat_capacity>
                  CONDUCT:
                    constant: [<thermal_conductivity>]
              # Electrode / scalar transport material, in the form 4C reads
              # before 2026.3.0. 4C 2026.3.0 rejects this block ("Could not
              # match this input"): it takes constant DIFF_COEF and COND, the
              # optional DIFF_COEF_CONC_SCALE_FUNCT, DIFF_COEF_TEMP_SCALE_FUNCT,
              # COND_CONC_SCALE_FUNCT and COND_TEMP_SCALE_FUNCT, no *_PARA_NUM /
              # *_PARA lists, and X_MIN / X_MAX inside an optional OCP_MODEL
              # LITHIATION_BOUNDS group.
              - MAT: 3
                MAT_electrode:
                  DIFF_COEF_CONC_DEP_FUNCT: <diff_coef_concentration_function>
                  DIFF_COEF_TEMP_SCALE_FUNCT: <diff_coef_temperature_function>
                  COND_CONC_DEP_FUNCT: <cond_concentration_function>
                  COND_TEMP_SCALE_FUNCT: <cond_temperature_function>
                  DIFF_PARA_NUM: <num_diffusion_parameters>
                  DIFF_PARA: [<diffusion_coefficient>]
                  COND_PARA_NUM: <num_conductivity_parameters>
                  COND_PARA: [<electronic_conductivity>]
                  C_MAX: <max_concentration>
                  CHI_MAX: <max_stoichiometry>
                  OCP_MODEL:
                    Function:
                      OCP_FUNCT_NUM: <ocp_function_id>
                    X_MIN: <ocp_x_min>
                    X_MAX: <ocp_x_max>

            # Clone structure mesh -> thermo mesh and scatra mesh
            CLONING MATERIAL MAP:
              - SRC_FIELD: "structure"
                SRC_MAT: 1
                TAR_FIELD: "thermo"
                TAR_MAT: 2
              - SRC_FIELD: "structure"
                SRC_MAT: 1
                TAR_FIELD: "scatra"
                TAR_MAT: 3

            # == Boundary Conditions ===========================================

            # Structural: fixed face
            DESIGN SURF DIRICH CONDITIONS:
              - E: <fixed_face_id>
                NUMDOF: 3
                ONOFF: [1, 1, 1]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]

            # Scalar transport: Dirichlet
            DESIGN SURF TRANSPORT DIRICH CONDITIONS:
              - E: <scalar_dirichlet_face_id>
                NUMDOF: <num_scalar_dofs>
                ONOFF: [<active_scalar_dofs>]
                VAL: [<scalar_boundary_values>]
                FUNCT: [<scalar_time_functions>]

            # Thermal: prescribed temperature
            DESIGN SURF THERMO DIRICH CONDITIONS:
              - E: <thermal_dirichlet_face_id>
                NUMDOF: 1
                ONOFF: [1]
                VAL: [<boundary_temperature>]
                FUNCT: [0]

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

            RESULT DESCRIPTION:
              - STRUCTURE:
                  DIS: "structure"
                  NODE: <result_node_id>
                  QUANTITY: "dispx"
                  VALUE: <expected_displacement>
                  TOLERANCE: <result_tolerance>
              - SCATRA:
                  DIS: "scatra"
                  NODE: <result_node_id>
                  QUANTITY: "phi"
                  VALUE: <expected_concentration>
                  TOLERANCE: <result_tolerance>
              - THERMAL:
                  DIS: "thermo"
                  NODE: <result_node_id>
                  QUANTITY: "temp"
                  VALUE: <expected_temperature>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check Young's modulus
        young = params.get("YOUNG")
        if young is not None:
            vals = young if isinstance(young, list) else [young]
            for v in vals:
                try:
                    e = float(v)
                    if e <= 0:
                        issues.append(f"YOUNG must be > 0, got {e}.")
                except (TypeError, ValueError):
                    issues.append(
                        f"YOUNG must be a positive number, got {v!r}."
                    )

        # Check Poisson's ratio
        nue = params.get("NUE")
        if nue is not None:
            try:
                nu = float(nue)
                if nu <= 0 or nu >= 0.5:
                    issues.append(f"NUE must be in (0, 0.5), got {nu}.")
            except (TypeError, ValueError):
                issues.append(
                    f"NUE must be a number in (0, 0.5), got {nue!r}."
                )

        # Check thermal expansion coefficient
        thexpans = params.get("THEXPANS")
        if thexpans is not None:
            try:
                alpha = float(thexpans)
                if alpha <= 0:
                    issues.append(
                        f"THEXPANS must be > 0, got {alpha}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"THEXPANS must be a positive number, got {thexpans!r}."
                )

        # Check heat capacity
        capa = params.get("CAPA")
        if capa is not None:
            try:
                c = float(capa)
                if c <= 0:
                    issues.append(
                        f"CAPA (volumetric heat capacity) must be > 0, "
                        f"got {c}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"CAPA must be a positive number, got {capa!r}."
                )

        # Check max concentration
        c_max = params.get("C_MAX")
        if c_max is not None:
            try:
                cm = float(c_max)
                if cm <= 0:
                    issues.append(
                        f"C_MAX must be > 0, got {cm}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"C_MAX must be a positive number, got {c_max!r}."
                )

        # Check element type
        elem_type = params.get("element_type", "")
        if elem_type and "SOLIDSCATRA" not in str(elem_type).upper():
            issues.append(
                f"SSTI requires SOLIDSCATRA elements (not {elem_type}).  "
                f"Plain SOLID elements do not carry scalar/thermal DOFs."
            )

        return issues
