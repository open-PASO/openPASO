"""Low Mach number flow generator for 4C.

Covers variable-density incompressible flow problems governed by the
low-Mach-number equations.  The flow is coupled with a scalar transport
equation for temperature (or species concentration), and the density
varies with temperature via the ideal gas law or Sutherland model.
Used for buoyancy-driven flows, heated channels, and natural convection.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class LowMachGenerator(BaseGenerator):
    """Generator for low Mach number flow problems in 4C."""

    module_key = "low_mach"
    display_name = "Low Mach Number Flow (Variable-Density)"
    problem_type = "Low_Mach_Number_Flow"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The low-Mach-number flow module solves the variable-density "
                "incompressible Navier-Stokes equations coupled with a "
                "scalar transport equation (typically temperature).  Density "
                "varies with temperature via the ideal gas law or Sutherland "
                "viscosity model, but acoustic waves are filtered out (Mach "
                "-> 0 limit).  This is appropriate for buoyancy-driven "
                "flows, heated channels, and natural convection problems.  "
                "The PROBLEM TYPE is 'Low_Mach_Number_Flow'.  Three main "
                "sections are needed: FLUID DYNAMIC (with PHYSICAL_TYPE: "
                "'Loma'), SCALAR TRANSPORT DYNAMIC (for temperature), and "
                "LOMA CONTROL (coupling parameters).  Monolithic coupling "
                "is recommended."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "FLUID DYNAMIC",
                "FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION",
                "SCALAR TRANSPORT DYNAMIC",
                "SCALAR TRANSPORT DYNAMIC/STABILIZATION",
                "LOMA CONTROL",
                "SOLVER 1",
                "SOLVER 2",
                "MATERIALS",
            ],
            "optional_sections": [
                "FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES",
                "FLUID DYNAMIC/TURBULENCE MODEL",
                "FLUID DYNAMIC/SUBGRID VISCOSITY",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/FLUID",
            ],
            "materials": {
                "MAT_sutherland": {
                    "description": (
                        "Sutherland viscosity law material for "
                        "temperature-dependent viscosity and density.  "
                        "Models the ideal-gas density variation and "
                        "Sutherland's three-coefficient viscosity formula.  "
                        "The thermal properties (specific heat, Prandtl "
                        "number, gas constant) are specified here."
                    ),
                    "parameters": {
                        "REFVISC": {
                            "description": (
                                "Reference dynamic viscosity mu_ref at "
                                "reference temperature [Pa s]"
                            ),
                            "range": "> 0",
                        },
                        "REFTEMP": {
                            "description": (
                                "Reference temperature T_ref for the "
                                "Sutherland law [K]"
                            ),
                            "range": "> 0",
                        },
                        "SUTHTEMP": {
                            "description": (
                                "Sutherland temperature S [K] (material "
                                "constant, e.g. 110.4 K for air)"
                            ),
                            "range": "> 0",
                        },
                        "SHC": {
                            "description": (
                                "Specific heat capacity at constant "
                                "pressure c_p [J/(kg K)]; required"
                            ),
                            "range": "> 0",
                        },
                        "PRANUM": {
                            "description": (
                                "Prandtl number Pr = mu * c_p / k "
                                "(ratio of momentum to thermal diffusivity)"
                            ),
                            "range": "> 0 (air ~0.71, water ~7)",
                        },
                        "THERMPRESS": {
                            "description": (
                                "(Initial) thermodynamic pressure "
                                "[J/m^3]; required"
                            ),
                            "range": "> 0",
                        },
                        "GASCON": {
                            "description": (
                                "Specific gas constant R_s = R / M "
                                "[J/(kg K)] where M is molecular mass"
                            ),
                            "range": "> 0 (air: 287 J/(kg K))",
                        },
                    },
                },
            },
            "solver": {
                "field_solvers": {
                    "type": "UMFPACK (direct)",
                    "notes": (
                        "Individual field solver (SOLVER 1) used by fluid "
                        "and scalar transport sub-problems."
                    ),
                },
                "monolithic_solver": {
                    "type": "Belos with Teko block preconditioner",
                    "notes": (
                        "The monolithic LOMA solver (SOLVER 2) uses Belos "
                        "iterative solver with AZPREC: Teko and a "
                        "fluid_scatra block preconditioner XML.  For small "
                        "problems UMFPACK can also be used."
                    ),
                },
            },
            "time_integration": {
                "FLUID DYNAMIC": {
                    "PHYSICAL_TYPE": (
                        "Must be set to 'Loma' (Low Mach number) to enable "
                        "variable-density treatment."
                    ),
                    "TIMEINTEGR": (
                        "'Af_Gen_Alpha' (recommended) or 'BDF2' for the "
                        "fluid field."
                    ),
                },
                "SCALAR TRANSPORT DYNAMIC": {
                    "TIMEINTEGR": (
                        "'Gen_Alpha' (recommended) to match the fluid "
                        "time integration order."
                    ),
                    "VELOCITYFIELD": (
                        "Must be 'Navier_Stokes' to couple with the fluid "
                        "velocity."
                    ),
                },
                "LOMA CONTROL": {
                    "MONOLITHIC": (
                        "Set to true for monolithic fluid-scalar coupling "
                        "(recommended for accuracy).  False for partitioned."
                    ),
                    "TIMESTEP": "Global time step for the coupled system.",
                    "NUMSTEP": "Number of time steps.",
                    "MAXTIME": "Maximum simulation time.",
                    "ITEMAX": "Max coupling iterations per step.",
                    "CONVTOL": "Coupling convergence tolerance.",
                },
            },
            "pitfalls": [
                (
                    "[Input] PHYSICAL_TYPE in FLUID DYNAMIC must be 'Loma' (or "
                    "'Temp_dep_water'). This is not a setting you can forget and "
                    'then diagnose from the flow field. Signal: any other value '
                    "aborts during adapter setup with 'Input parameter "
                    'PHYSICAL_TYPE in section FLUID DYNAMIC needs to be Loma or '
                    "Temp_dep_water for low-Mach-number flow!' from "
                    '4C_adapter_fld_base_algorithm.cpp, before the first time step, '
                    'so there is no stagnant velocity field or zero buoyant flow '
                    'rate to inspect. (Audit 2026-06-02; corrected by execution '
                    '2026-08-06.)'
                ),
                (
                    '[Input] VELOCITYFIELD in SCALAR TRANSPORT DYNAMIC must be '
                    "'Navier_Stokes' to couple the temperature to the fluid "
                    'velocity. The value does not tune the coupling, it selects the '
                    'ALGORITHM: loma_dyn branches on it, and only Navier_Stokes '
                    'clones the scalar-transport discretisation from the fluid. '
                    "Signal: 'zero' and 'function' both take a scatra-only branch "
                    'that expects a hand-written TRANSPORT ELEMENTS section and '
                    "aborts with 'No elements in input section ---TRANSPORT "
                    "ELEMENTS!' from 4C_loma_dyn.cpp. The fluid field is never "
                    'built, and no warning about an inconsistent combination is '
                    'ever printed. (Audit 2026-06-02; corrected by execution '
                    '2026-08-06.)'
                ),
                (
                    '[Input] Use MAT_sutherland, which supplies '
                    'temperature-dependent viscosity and density through the ideal '
                    'gas law; MAT_fluid will not do. Because a LOMA run clones its '
                    'scalar-transport discretisation from the fluid one, the '
                    'material is vetted by the scatra cloning strategy rather than '
                    "by any PHYSICAL_TYPE check. Signal: 'Material with ID {} is not "
                    "admissible for scalar transport elements' from "
                    '4C_scatra_utils_clonestrategy.cpp:59, with the material id '
                    'filled in, thrown before the first '
                    'time step. The message names neither the material type nor '
                    'Loma, so grepping for either finds nothing, and there is no '
                    'quieter constant-density outcome to fall back on. (Audit '
                    '2026-06-02; corrected by execution 2026-08-06.)'
                ),
                (
                    '[Numerical] Give the fluid and the scalar consistent initial '
                    'fields: INITIALFIELD: field_by_function with STARTFUNCNO for '
                    'the fluid and INITFUNCNO for the scalar transport. With '
                    'MAT_sutherland this is a hard requirement, not a convergence '
                    'nicety, because density is evaluated from the ideal gas law '
                    'and a zero initial temperature divides by zero. Signal: '
                    'INITIALFIELD: zero_field terminates the process with SIGFPE, '
                    'so the log ends with the MPI runtime\'s signal block -- the '
                    'floating point exception, signal 8, with signal code '
                    'floating-point divide-by-zero, 3 -- and carries no PROC 0 '
                    'ERROR block, no residual table and no result test. (Audit '
                    '2026-06-02; corrected by execution 2026-08-06.)'
                ),
                (
                    '[Numerical] Do not try to make the stabilisation tau '
                    'definitions identical on the two fields: the enums are '
                    'different sets. FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION '
                    'offers Taylor_Hughes_Zarins_Whiting_Jansen, while SCALAR '
                    'TRANSPORT DYNAMIC/STABILIZATION stops at Taylor_Hughes_Zarins. '
                    'Signal: writing the fluid value into the scatra section fails '
                    "with 'Could not match this input' and 4C prints the admissible "
                    'scatra list. The upstream LOMA regression decks deliberately '
                    'run Whiting_Jansen on the fluid and Taylor_Hughes_Zarins on '
                    'the scalar and pass their result tests, and no '
                    'coupling-interface artefact is reported. (Audit 2026-06-02; '
                    'corrected by execution 2026-08-06.)'
                ),
                (
                    '[Input] Gravity and buoyancy are applied as a body force '
                    'through a Neumann condition; there is no gravity section and '
                    'no FORCE key. Signal: a GRAVITY: block is rejected with '
                    '"Section \'GRAVITY\' is not a valid section name." (the only '
                    'GRAVITY_ACCELERATION / GRAVITY_RAMP_FUNCT keys in 4C belong to '
                    'PARTICLE DYNAMIC), and a Neumann entry written with FORCE: '
                    "instead of VAL: fails with 'Failed to match condition "
                    "specification' from 4C_fem_condition_definition.cpp. The "
                    'working form is DESIGN VOL NEUMANN CONDITIONS in 3D (DESIGN '
                    'SURF NEUMANN in 2D) with NUMDOF, ONOFF, VAL, FUNCT and TYPE: '
                    "'Dead'. (Audit 2026-06-02; corrected by execution 2026-08-06.)"
                ),
                (
                    '[Numerical] For natural convection the Rayleigh number sets '
                    'the regime and the boundary layers must be resolved for it, '
                    'but 4C computes neither Ra nor Nu and never prints either '
                    "word, so 'compare Nu with the literature value' needs a "
                    'non-dimensionalisation you do yourself. Signal: the quantity '
                    '4C will give you is the wall heat flux, and only if you ask '
                    'for it: set CALCFLUX_BOUNDARY in SCALAR TRANSPORT DYNAMIC and '
                    'add a SCATRA FLUX CALC LINE/SURF CONDITIONS block, after which '
                    'each step prints "Normal fluxes at boundary \'ScaTraFluxCalc\' '
                    'on discretization \'scatra\'" and \'Sum of all normal flux '
                    "boundary integrals for scalar 0'. (Audit 2026-06-02; corrected "
                    'by execution 2026-08-06.)'
                ),
            ],
            "typical_experiments": [
                {
                    "name": "heated_channel_2d",
                    "description": (
                        "Buoyancy-driven flow in a 2-D heated channel.  "
                        "Cold inlet, heated bottom wall, natural convection "
                        "develops.  Tests variable-density coupling, "
                        "Sutherland viscosity, and stabilisation."
                    ),
                    "template_variant": "heated_channel_2d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "heated_channel_2d",
                "description": (
                    "2-D heated channel with variable-density flow.  "
                    "MAT_sutherland material, monolithic LOMA coupling, "
                    "Af_Gen_Alpha fluid time integration."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "heated_channel_2d") -> str:
        templates = {
            "heated_channel_2d": self._template_heated_channel_2d,
        }
        if variant == "default":
            variant = "heated_channel_2d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_heated_channel_2d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 2-D Low Mach Number Heated Channel
            #
            # Variable-density buoyancy-driven flow in a heated channel.
            # Cold fluid enters, is heated from the bottom wall, and exits.
            # Density varies with temperature via ideal gas / Sutherland law.
            #
            # Mesh: exodus file with:
            #   element_block 1 = fluid domain (QUAD4)
            #   node_set 1 = bottom wall (no-slip + heated)
            #   node_set 2 = top wall (no-slip)
            #   node_set 3 = inlet (velocity + cold temperature)
            #   node_set 4 = symmetry / side walls
            #   node_set 5 = outlet (natural BC)
            #   node_set 6 = inlet node for initial velocity
            # ---------------------------------------------------------------
            TITLE:
              - "2-D low-Mach heated channel -- generated template"
            PROBLEM SIZE:
              DIM: 2
            PROBLEM TYPE:
              PROBLEMTYPE: "Low_Mach_Number_Flow"

            # == Fluid dynamics (low Mach) =====================================
            FLUID DYNAMIC:
              PHYSICAL_TYPE: "Loma"
              LINEAR_SOLVER: 1
              TIMEINTEGR: "Af_Gen_Alpha"
              INITIALFIELD: "field_by_function"
              ADAPTCONV: true
              RESTARTEVERY: <restart_interval>
              NUMSTEP: <number_of_steps>
              STARTFUNCNO: <fluid_initial_function_id>
              ITEMAX: <fluid_max_iterations>
              TIMESTEP: <timestep>
              MAXTIME: <end_time>
              ALPHA_M: <genalpha_alpha_m>
              ALPHA_F: <genalpha_alpha_f>
              GAMMA: <genalpha_gamma>
              THETA: <theta>
              START_THETA: <start_theta>
            FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES:
              TOL_VEL_RES: <velocity_residual_tolerance>
              TOL_VEL_INC: <velocity_increment_tolerance>
              TOL_PRES_RES: <pressure_residual_tolerance>
              TOL_PRES_INC: <pressure_increment_tolerance>
            FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION:
              INCONSISTENT: <inconsistent_flag>
              CROSS-STRESS: "<cross_stress_option>"
              REYNOLDS-STRESS: "<reynolds_stress_option>"
              DEFINITION_TAU: "<tau_definition>"
              EVALUATION_TAU: "integration_point"
              EVALUATION_MAT: "integration_point"

            # == Scalar Transport (temperature) ================================
            SCALAR TRANSPORT DYNAMIC:
              SOLVERTYPE: "linear_incremental"
              TIMEINTEGR: "Gen_Alpha"
              MAXTIME: <end_time>
              TIMESTEP: <timestep>
              ALPHA_M: <scatra_genalpha_alpha_m>
              ALPHA_F: <scatra_genalpha_alpha_f>
              GAMMA: <scatra_genalpha_gamma>
              RESULTSEVERY: <results_output_interval>
              RESTARTEVERY: <restart_interval>
              MATID: 1
              VELOCITYFIELD: "Navier_Stokes"
              INITIALFIELD: "field_by_function"
              INITFUNCNO: <temperature_initial_function_id>
              CALCFLUX_BOUNDARY: "<flux_calc_option>"
              LINEAR_SOLVER: 1
            SCALAR TRANSPORT DYNAMIC/STABILIZATION:
              SUGRVEL: <subgrid_velocity_flag>
              DEFINITION_TAU: "<scatra_tau_definition>"
              EVALUATION_TAU: "integration_point"
              EVALUATION_MAT: "integration_point"

            # == LOMA coupling =================================================
            LOMA CONTROL:
              MONOLITHIC: <monolithic_flag>
              NUMSTEP: <number_of_steps>
              TIMESTEP: <timestep>
              MAXTIME: <end_time>
              ITEMAX: <coupling_max_iterations>
              CONVTOL: <coupling_convergence_tolerance>
              RESTARTEVERY: <restart_interval>
              LINEAR_SOLVER: 2

            # == Solvers =======================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "field_solver"
            SOLVER 2:
              SOLVER: "UMFPACK"
              NAME: "monolithic_solver"

            # == Material (Sutherland) =========================================
            MATERIALS:
              - MAT: 1
                MAT_sutherland:
                  REFVISC: <reference_viscosity>
                  REFTEMP: <reference_temperature>
                  SUTHTEMP: <sutherland_temperature>
                  SHC: <specific_heat_capacity_cp>
                  PRANUM: <prandtl_number>
                  THERMPRESS: <initial_thermodynamic_pressure>
                  GASCON: <specific_gas_constant>

            # == Initial condition functions ====================================
            # Fluid initial condition (velocity + pressure)
            FUNCT<fluid_initial_function_id>:
              - COMPONENT: 0
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<initial_velocity_x_expression>"
              - COMPONENT: 1
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<initial_velocity_y_expression>"
              - COMPONENT: 2
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<initial_pressure_expression>"
            # Temperature initial condition
            FUNCT<temperature_initial_function_id>:
              - COMPONENT: 0
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<initial_temperature_expression>"

            # Inlet velocity ramp function
            FUNCT<inlet_ramp_function_id>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<inlet_ramp_expression>"

            # Temperature ramp for heated wall
            FUNCT<heated_wall_function_id>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<heated_wall_temp_expression>"

            # == Boundary Conditions ===========================================

            # No-slip walls
            DESIGN LINE DIRICH CONDITIONS:
              - E: <bottom_wall_id>
                NUMDOF: 3
                ONOFF: [1, 1, 0]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]
              - E: <top_wall_id>
                NUMDOF: 3
                ONOFF: [1, 1, 0]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]
              # Inlet
              - E: <inlet_id>
                NUMDOF: 3
                ONOFF: [1, 1, 0]
                VAL: [<inlet_velocity>, 0.0, 0.0]
                FUNCT: [<inlet_ramp_function_id>, 0, 0]

            # Gravity / buoyancy body force
            DESIGN LINE NEUMANN CONDITIONS:
              - E: <gravity_region_id>
                NUMDOF: 6
                ONOFF: [<active_force_dofs>]
                VAL: [<body_force_values>]
                FUNCT: [<body_force_time_functions>]

            # Temperature BCs
            DESIGN LINE TRANSPORT DIRICH CONDITIONS:
              - E: <heated_wall_id>
                NUMDOF: 1
                ONOFF: [1]
                VAL: [<hot_wall_temperature>]
                FUNCT: [<heated_wall_function_id>]
              - E: <inlet_id>
                NUMDOF: 1
                ONOFF: [1]
                VAL: [<inlet_temperature>]
                FUNCT: [0]

            # == Geometry ======================================================
            FLUID GEOMETRY:
              FILE: "<mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  FLUID:
                    QUAD4:
                      MAT: 1
                      NA: Euler

            # == VTK output ====================================================
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/FLUID:
              OUTPUT_FLUID: true
              VELOCITY: true
              PRESSURE: true

            RESULT DESCRIPTION:
              - FLUID:
                  DIS: "fluid"
                  NODE: <result_node_id>
                  QUANTITY: "velx"
                  VALUE: <expected_velocity_x>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check PHYSICAL_TYPE
        physical_type = params.get("PHYSICAL_TYPE")
        if physical_type is not None and physical_type != "Loma":
            issues.append(
                f"PHYSICAL_TYPE must be 'Loma' for low-Mach flow, "
                f"got {physical_type!r}."
            )

        # Check REFVISC
        refvisc = params.get("REFVISC")
        if refvisc is not None:
            try:
                mu = float(refvisc)
                if mu <= 0:
                    issues.append(
                        f"REFVISC (reference viscosity) must be > 0, "
                        f"got {mu}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"REFVISC must be a positive number, got {refvisc!r}."
                )

        # Check REFTEMP
        reftemp = params.get("REFTEMP")
        if reftemp is not None:
            try:
                t = float(reftemp)
                if t <= 0:
                    issues.append(
                        f"REFTEMP (reference temperature) must be > 0 [K], "
                        f"got {t}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"REFTEMP must be a positive number, got {reftemp!r}."
                )

        # Check SUTHTEMP
        suthtemp = params.get("SUTHTEMP")
        if suthtemp is not None:
            try:
                s = float(suthtemp)
                if s <= 0:
                    issues.append(
                        f"SUTHTEMP (Sutherland temperature) must be > 0 [K], "
                        f"got {s}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"SUTHTEMP must be a positive number, got {suthtemp!r}."
                )

        # Check PRANUM
        pranum = params.get("PRANUM")
        if pranum is not None:
            try:
                pr = float(pranum)
                if pr <= 0:
                    issues.append(
                        f"PRANUM (Prandtl number) must be > 0, got {pr}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"PRANUM must be a positive number, got {pranum!r}."
                )

        # Check GASCON
        gascon = params.get("GASCON")
        if gascon is not None:
            try:
                r = float(gascon)
                if r <= 0:
                    issues.append(
                        f"GASCON (gas constant) must be > 0, got {r}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"GASCON must be a positive number, got {gascon!r}."
                )

        # Check VELOCITYFIELD in scalar transport
        velfield = params.get("VELOCITYFIELD")
        if velfield is not None and velfield != "Navier_Stokes":
            issues.append(
                f"VELOCITYFIELD must be 'Navier_Stokes' for LOMA problems "
                f"(temperature is advected by the fluid), got {velfield!r}."
            )

        # Check MONOLITHIC flag
        monolithic = params.get("MONOLITHIC")
        if monolithic is not None and not monolithic:
            issues.append(
                "Partitioned LOMA coupling may have stability issues.  "
                "Monolithic coupling (MONOLITHIC: true) is recommended."
            )

        return issues
