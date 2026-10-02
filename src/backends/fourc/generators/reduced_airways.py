"""Reduced-Dimensional Airways generator for 4C.

Covers 1-D reduced-dimensional modeling of the pulmonary airway tree.
The airways are represented as a branching network of 1-D elements
where each element models a compliant tube segment with airflow governed
by pressure-flow relationships derived from the Navier-Stokes equations
under the long-wavelength approximation.  Applications include
respiratory mechanics, ventilator design, and patient-specific lung
modeling.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class ReducedAirwaysGenerator(BaseGenerator):
    """Generator for reduced-dimensional airways problems in 4C."""

    module_key = "reduced_airways"
    display_name = "Reduced-Dimensional Airways (Lung Airway Tree)"
    problem_type = "ReducedDimensionalAirWays"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The reduced-dimensional airways module models the "
                "pulmonary airway tree as a branching 1-D network.  "
                "Each airway segment is represented by a compliant tube "
                "element with pressure-flow governing equations derived "
                "from the incompressible Navier-Stokes equations under "
                "the long-wavelength (Womersley) approximation.  The "
                "network branches according to the anatomical airway "
                "tree (from trachea down to terminal bronchioles).  "
                "Boundary conditions include flow or pressure at the "
                "tracheal inlet and tissue compliance/resistance at "
                "the terminal ends (acini).  The PROBLEM TYPE is "
                "'ReducedDimensionalAirWays'.  The dynamics section is "
                "'REDUCED DIMENSIONAL AIRWAYS DYNAMIC'.  Elements use "
                "the RED_AIRWAY (conducting airway) and RED_ACINUS "
                "(terminal acinus) element types -- both 1-D LINE2 "
                "elements -- written as free-form text lines in the "
                "'REDUCED D AIRWAYS ELEMENTS' section, with node "
                "positions in 'NODE COORDS' and boundary-node sets in "
                "'DNODE-NODE TOPOLOGY'.  The airway material is "
                "MAT_fluid; the acinus material is one of the "
                "MAT_0D_MAXWELL_ACINUS* family.  Airway wall and "
                "geometric properties (Area, WallElasticity, "
                "WallThickness, PoissonsRatio, Generation) are "
                "ELEMENT-LINE tokens, not material parameters."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "REDUCED DIMENSIONAL AIRWAYS DYNAMIC",
                "SOLVER 1",
                "MATERIALS",
                "REDUCED D AIRWAYS ELEMENTS",
                "NODE COORDS",
                "DNODE-NODE TOPOLOGY",
                "DESIGN NODE Reduced D AIRWAYS PRESCRIBED CONDITIONS",
            ],
            "optional_sections": [
                "IO",
                "RESULT DESCRIPTION",
                "FUNCT<n>",
                "DESIGN NODE Reduced D AIRWAYS SWITCH FLOW PRESSURE CONDITIONS",
                "DESIGN LINE REDUCED D AIRWAYS VOL DEPENDENT PLEURAL "
                "PRESSURE CONDITIONS",
                "DESIGN LINE REDUCED D AIRWAYS EVALUATE LUNG VOLUME CONDITIONS",
            ],
            "materials": {
                "MAT_fluid": {
                    "description": (
                        "The material of a RED_AIRWAY element is the air "
                        "filling it.  There is no MAT_redAirway in 4C: "
                        "every reduced-airway deck shipped with 4C drives "
                        "its airways from MAT_fluid.  Wall and geometric "
                        "properties live on the RED_AIRWAY element line "
                        "(Area, WallElasticity, WallThickness, "
                        "PoissonsRatio, Generation), not here."
                    ),
                    "parameters": {
                        "DYNVISCOSITY": {
                            "description": (
                                "Dynamic viscosity of the airway fluid.  "
                                "4C's own airway decks use 0.04 in the "
                                "cm/g/s-derived unit system they are "
                                "written in; ~1.8e-5 Pa s for air at 37 C "
                                "in SI."
                            ),
                            "range": "> 0",
                        },
                        "DENSITY": {
                            "description": (
                                "Density of the airway fluid (1.176e-6 in "
                                "4C's airway decks; ~1.176 kg/m^3 for air "
                                "at 37 C in SI)."
                            ),
                            "range": "> 0",
                        },
                        "GAMMA": {
                            "description": (
                                "Surface-tension coefficient of the fluid; "
                                "set to 1 in 4C's airway decks."
                            ),
                            "range": "real",
                        },
                    },
                },
                "MAT_0D_MAXWELL_ACINUS_EXPONENTIAL": {
                    "description": (
                        "Acinar (terminal) material: a 0-D four-element "
                        "Maxwell model of the compliant alveolar "
                        "compartment.  The whole family shares the same "
                        "four parameters and differs only in the "
                        "non-linear spring law: MAT_0D_MAXWELL_ACINUS "
                        "(linear), _NEOHOOKEAN, _EXPONENTIAL, "
                        "_DOUBLEEXPONENTIAL and _OGDEN.  The material "
                        "name must agree with the TYPE token on the "
                        "RED_ACINUS element line (NeoHookean, "
                        "Exponential, DoubleExponential, "
                        "VolumetricOgden), and the law's shape "
                        "coefficients (E1_0, E1_LIN, E1_EXP, TAU, KAPPA, "
                        "BETA, ...) are ELEMENT-LINE tokens, not material "
                        "parameters."
                    ),
                    "parameters": {
                        "Stiffness1": {
                            "description": (
                                "Stiffness of the spring in the first "
                                "(purely elastic) Maxwell branch."
                            ),
                            "range": "> 0",
                        },
                        "Stiffness2": {
                            "description": (
                                "Stiffness of the spring in the second "
                                "(viscoelastic) Maxwell branch."
                            ),
                            "range": "> 0",
                        },
                        "Viscosity1": {
                            "description": (
                                "Dashpot viscosity in the first Maxwell "
                                "branch."
                            ),
                            "range": ">= 0",
                        },
                        "Viscosity2": {
                            "description": (
                                "Dashpot viscosity in the second Maxwell "
                                "branch."
                            ),
                            "range": ">= 0",
                        },
                    },
                },
            },
            "solver": {
                "airways_solver": {
                    "type": "UMFPACK",
                    "notes": (
                        "The reduced airways system is a small 1-D "
                        "network and is efficiently solved with a "
                        "direct solver."
                    ),
                },
            },
            "time_integration": {
                "scheme": (
                    "Implicit time integration for the 1-D network.  "
                    "Time step should resolve the breathing cycle "
                    "(typical period 3--5 seconds) with sufficient "
                    "resolution (dt ~ 0.001--0.01 s)."
                ),
                "TIMESTEP": "Time step for the airway network solver.",
                "NUMSTEP": (
                    "Total number of time steps.  REDUCED DIMENSIONAL "
                    "AIRWAYS DYNAMIC has NO MAXTIME key -- the simulated "
                    "end time is TIMESTEP * NUMSTEP and nothing else.  "
                    "MAXTIME exists only in the separate COUPLED "
                    "REDUCED-D AIRWAYS AND TISSUE DYNAMIC section."
                ),
                "DYNAMICTYPE": (
                    "Time-integration scheme (default OneStepTheta), "
                    "weighted by THETA (default 1.0 = backward Euler)."
                ),
                "SOLVERTYPE": (
                    "Linear (default) or Nonlinear.  Compliant walls, "
                    "acini and airway collapse all need Nonlinear."
                ),
                "MAXITERATIONS": (
                    "Maximum nonlinear iterations per step.  The key is "
                    "MAXITERATIONS, NOT MAXITER -- MAXITER is legal only "
                    "in COUPLED REDUCED-D AIRWAYS AND TISSUE DYNAMIC and "
                    "4C rejects it here with 'The following data remains "
                    "unused: MAXITER'."
                ),
                "TOLERANCE": "Nonlinear convergence tolerance.",
            },
            "pitfalls": [
                (
                    "[Output] ReducedDimensionalAirWays writes only 4C-native output "
                    "(<prefix>.control, .result.*), and post_processor aborts on this 1-D "
                    "result type with both --filter=vtu and --filter=ensight, so no field "
                    "file that openPASO's result gate can read exists for it. Signal: the "
                    "run ends with 'processor 0 finished normally', no *.vtu or *.pvd "
                    "appears, and post_processor.log in the run directory ends in an MPI "
                    "abort. The per-step |Pressure|_max and |Q|_max lines in the log are "
                    "the readable record of the solve. (Measured 2026-09-24 by running the "
                    "served template.)"
                ),
                (
                    "[Input] 4C does NOT validate reduced-airway tree topology. "
                    "A loop, or a branch ending on a node with no boundary "
                    "condition, is accepted and solved. Signal: none - there is "
                    "no topology check and no message. The quoted `node X has "
                    "degree 1 (dangling)` does not exist in 4C, and neither "
                    "does `cycle detected in airway tree`. "
                    "The run completes its time loop and only a "
                    "RESULT DESCRIPTION entry reveals that the answer moved. "
                    "Check the tree yourself before trusting it. (Audit "
                    "2026-08-06, verified by execution.)"
                ),
                (
                    "[Input] Boundary conditions at the trachea define the "
                    "driving force: a DESIGN NODE Reduced D AIRWAYS PRESCRIBED "
                    "CONDITIONS entry on the inlet DNODE, flow by default or "
                    "boundarycond: pressure. Without it nothing drives the "
                    "tree. Signal: the run converges and every step prints "
                    "|Pressure|_max: 0.000E+00 and |Q|_max: 0.000E+00. There is "
                    "NO warning about the missing condition; `no DESIGN POINT "
                    "1D DBC defined at trachea node` does not exist in 4C. "
                    "(Audit 2026-08-06, verified by execution.)"
                ),
                (
                    "[Numerical] The terminal impedance of a reduced-airway "
                    "tree is the acinus element's MAT_0D_MAXWELL_ACINUS_* "
                    "stiffness and viscosity, not a separate acinar condition. "
                    "Signal: make the acinus much softer and the inlet pressure "
                    "collapses onto the prescribed downstream pressure - no "
                    "impedance - with no diagnostic. The acinar volume does NOT "
                    "diverge when the inlet carries a prescribed flow: it is "
                    "fixed by that flow, so watch the pressure, not the volume. "
                    "(Audit 2026-08-06, verified by execution.)"
                ),
                (
                    "[Numerical] Airway wall behaviour is an ELEMENT-line "
                    "option and it depends on the airway TYPE. There is no "
                    "MAT_redAirway material in 4C and no COMPLIANCE material "
                    "parameter; the airway material is MAT_fluid. Collapse and "
                    "reopening come from AirwayColl with Pcrit_Open / "
                    "Pcrit_Close / S_Open / S_Close on the RED_AIRWAY line. "
                    "Signal: on a TYPE Resistive airway, WallElasticity is "
                    "parsed and then silently ignored - the result is bit-for- "
                    "bit unchanged - while it does act on "
                    "ConvectiveViscoElasticRLC. (Audit 2026-08-06, verified by "
                    "execution.)"
                ),
                (
                    "[Numerical] Reduced model assumes LONG WAVELENGTH "
                    "(Womersley number restrictions, Wo < O(1)). For high- "
                    "frequency ventilation a full 3D CFD approach may be needed "
                    "instead. Signal: none from 4C. The reduced-airway and "
                    "artery modules never compute or print a Womersley number - "
                    "the word appears in the source only for 3D fluid inflow "
                    "profiles - so an HFOV-rate drive runs to completion and "
                    "silently returns different numbers. Compute Wo = R * "
                    "sqrt(omega / nu) yourself. (Audit 2026-08-06, verified by "
                    "execution.)"
                ),
                (
                    "[Input] Properties that vary by airway GENERATION are "
                    "element-line tokens, not materials: Area, WallElasticity, "
                    "WallThickness, PoissonsRatio and Generation sit on each "
                    "RED_AIRWAY line, and 4C's own multi-generation decks drive "
                    "every airway from one MAT_fluid. There is no "
                    "MAT_redAirway. Signal: none - a uniform tree is accepted "
                    "without comment; taper Area and set the wall properties "
                    "per element, and check the total resistance yourself. "
                    "(Audit 2026-08-06, verified by execution.)"
                ),
                (
                    "[Input] The element keywords are RED_AIRWAY and "
                    "RED_ACINUS, and each has its own TYPE enum. Signal: the "
                    "spelling REDAIRWAY is rejected by the ParObject factory "
                    "with \"Unknown type 'REDAIRWAY' of finite element\" from "
                    "4C_comm_parobjectfactory.cpp; a beam or fluid element "
                    "keyword dies earlier still, in the element reader, with "
                    "\"Required 'one_of' not found in input line\"; and using "
                    "RED_AIRWAY with an acinus TYPE prints the list of valid "
                    "airway TYPEs. The quoted `unknown element type for "
                    "ELEMENT_BLOCK X` does not exist in 4C, and neither does "
                    "`elementType BEAM3R has no RedAirway "
                    "implementation`. (Audit 2026-08-06, verified "
                    "by execution.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "single_breath_cycle",
                    "description": (
                        "A single inspiration-expiration cycle through "
                        "a multi-generation airway tree (5-10 "
                        "generations).  Tracheal pressure drives "
                        "breathing, acinar compliance stores lung "
                        "volume.  Tests network topology, acinar BCs, "
                        "and time-dependent flow distribution."
                    ),
                    "template_variant": "airways_1d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "airways_1d",
                "description": (
                    "1-D reduced-dimensional airway tree: branching "
                    "network with RED_AIRWAY line elements, RED_ACINUS "
                    "terminals, pressure-driven breathing, UMFPACK "
                    "solver."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "airways_1d") -> str:
        templates = {
            "airways_1d": self._template_airways_1d,
        }
        if variant == "default":
            variant = "airways_1d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_airways_1d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 1-D Reduced-Dimensional Airway Tree
            #
            # A branching network of compliant airway segments modelling
            # the pulmonary airways from trachea to terminal bronchioles.
            # Each conducting segment is a 1-D RED_AIRWAY LINE2 element;
            # each terminal branch ends in a RED_ACINUS LINE2 element
            # (the compliant alveolar compartment).
            #
            # Mesh: the airway tree is written INLINE, not read from an
            # exodus file.  All 4C reduced-airway decks declare it with
            # three free-form text sections:
            #   REDUCED D AIRWAYS ELEMENTS - one text line per element
            #   NODE COORDS                - one text line per node
            #   DNODE-NODE TOPOLOGY        - point sets for the BCs
            # There is no 'REDAIRWAY GEOMETRY' section.
            #
            # The example below is a 2-generation bifurcation:
            #   e1: node 1 (trachea inlet) -> node 2
            #   e2: node 2 -> node 3   e4 (acinus): node 3 -> node 5
            #   e3: node 2 -> node 4   e5 (acinus): node 4 -> node 6
            # ---------------------------------------------------------------
            TITLE:
              - "1-D reduced-dimensional airway tree -- generated template"
            PROBLEM SIZE:
              ELEMENTS: <number_of_elements>
              NODES: <number_of_nodes>
              MATERIALS: <number_of_materials>
              NUMDF: 1
            PROBLEM TYPE:
              PROBLEMTYPE: "ReducedDimensionalAirWays"
            IO:
              STDOUTEVERY: <stdout_interval>

            # == Reduced airways dynamics ======================================
            # NOTE: this section has NO MAXTIME key. The end time is
            # TIMESTEP * NUMSTEP. The iteration cap is MAXITERATIONS, not
            # MAXITER -- 4C rejects both MAXTIME and MAXITER here with
            # "The following data remains unused".
            REDUCED DIMENSIONAL AIRWAYS DYNAMIC:
              DYNAMICTYPE: "OneStepTheta"
              SOLVERTYPE: "Nonlinear"
              THETA: <theta>
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              LINEAR_SOLVER: 1
              RESULTSEVERY: <results_output_interval>
              RESTARTEVERY: <restart_interval>
              MAXITERATIONS: <nonlinear_max_iterations>
              TOLERANCE: <nonlinear_tolerance>

            # == Solver ========================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "Reduced_dimensional_Airways_Solver"

            # == Materials =====================================================
            MATERIALS:
              # Airway fluid (the air in the conducting airways).  There is
              # no MAT_redAirway in 4C: wall stiffness and cross-sectional
              # area are ELEMENT-LINE tokens, see below.
              - MAT: 1
                MAT_fluid:
                  DYNVISCOSITY: <air_dynamic_viscosity>
                  DENSITY: <air_density>
                  GAMMA: <surface_tension_coefficient>
              # Acinar tissue (0-D four-element Maxwell model).  The suffix
              # must match the TYPE token on the RED_ACINUS element line;
              # alternatives: MAT_0D_MAXWELL_ACINUS,
              # MAT_0D_MAXWELL_ACINUS_NEOHOOKEAN,
              # MAT_0D_MAXWELL_ACINUS_DOUBLEEXPONENTIAL,
              # MAT_0D_MAXWELL_ACINUS_OGDEN.
              - MAT: 2
                MAT_0D_MAXWELL_ACINUS_EXPONENTIAL:
                  Stiffness1: <acinar_stiffness_1>
                  Stiffness2: <acinar_stiffness_2>
                  Viscosity1: <acinar_viscosity_1>
                  Viscosity2: <acinar_viscosity_2>

            # == Breathing waveform ============================================
            # The airway BCs are evaluated in time only, so use
            # SYMBOLIC_FUNCTION_OF_TIME.
            FUNCT1:
              - SYMBOLIC_FUNCTION_OF_TIME: "<breathing_waveform_expression>"

            # == Boundary Conditions ===========================================
            # Section name is mixed case: "DESIGN NODE Reduced D AIRWAYS ...".
            # Legal keys are E, boundarycond, VAL, curve, funct (curve and
            # funct lower case). boundarycond is one of flow (default),
            # pressure, switchFlowPressure,
            # VolumeDependentPleuralPressure. VAL is a 1-entry list and
            # curve a mandatory 2-entry list of function ids (use null for
            # the unused slot). NUMDOF / ONOFF / FUNCT are NOT accepted here.
            DESIGN NODE Reduced D AIRWAYS PRESCRIBED CONDITIONS:
              # Tracheal inlet: prescribed pressure waveform
              - E: 1
                boundarycond: "pressure"
                VAL: [<tracheal_pressure_amplitude>]
                curve: [1, null]
              # Distal ends of the acini: reference (pleural) pressure
              - E: 2
                boundarycond: "pressure"
                VAL: [<distal_pressure_amplitude>]
                curve: [1, null]
              - E: 3
                boundarycond: "pressure"
                VAL: [<distal_pressure_amplitude>]
                curve: [1, null]

            # == Geometry ======================================================
            # Free-form text lines. Airway wall behaviour and generation-
            # dependent geometry are tokens HERE, not material parameters.
            REDUCED D AIRWAYS ELEMENTS:
              - "1 RED_AIRWAY LINE2 1 2 MAT 1 ElemSolvingType NonLinear TYPE Resistive
                Resistance Poiseuille PowerOfVelocityProfile 2 WallElasticity <wall_elasticity>
                PoissonsRatio <poissons_ratio> ViscousTs <viscous_ts> ViscousPhaseShift
                <viscous_phase_shift> WallThickness <wall_thickness> Area <trachea_area>
                Generation 0"
              - "2 RED_AIRWAY LINE2 2 3 MAT 1 ElemSolvingType NonLinear TYPE Resistive
                Resistance Poiseuille PowerOfVelocityProfile 2 WallElasticity <wall_elasticity>
                PoissonsRatio <poissons_ratio> ViscousTs <viscous_ts> ViscousPhaseShift
                <viscous_phase_shift> WallThickness <wall_thickness> Area <daughter_area>
                Generation 1"
              - "3 RED_AIRWAY LINE2 2 4 MAT 1 ElemSolvingType NonLinear TYPE Resistive
                Resistance Poiseuille PowerOfVelocityProfile 2 WallElasticity <wall_elasticity>
                PoissonsRatio <poissons_ratio> ViscousTs <viscous_ts> ViscousPhaseShift
                <viscous_phase_shift> WallThickness <wall_thickness> Area <daughter_area>
                Generation 1"
              - "4 RED_ACINUS LINE2 3 5 MAT 2 TYPE Exponential AcinusVolume <acinus_volume>
                AlveolarDuctVolume <alveolar_duct_volume> E1_0 <e1_0> E1_LIN <e1_lin>
                E1_EXP <e1_exp> TAU <tau>"
              - "5 RED_ACINUS LINE2 4 6 MAT 2 TYPE Exponential AcinusVolume <acinus_volume>
                AlveolarDuctVolume <alveolar_duct_volume> E1_0 <e1_0> E1_LIN <e1_lin>
                E1_EXP <e1_exp> TAU <tau>"
            NODE COORDS:
              - "NODE 1 COORD <x1> <y1> <z1>"
              - "NODE 2 COORD <x2> <y2> <z2>"
              - "NODE 3 COORD <x3> <y3> <z3>"
              - "NODE 4 COORD <x4> <y4> <z4>"
              - "NODE 5 COORD <x5> <y5> <z5>"
              - "NODE 6 COORD <x6> <y6> <z6>"
            DNODE-NODE TOPOLOGY:
              - "NODE 1 DNODE 1"
              - "NODE 5 DNODE 2"
              - "NODE 6 DNODE 3"

            # Sub-entry key is RED_AIRWAY (with underscore), not REDAIRWAY.
            # QUANTITY is one of pressure, flow_in, flow_out (nodal) or
            # acini_volume (elemental, addressed by ELEMENT not NODE).
            RESULT DESCRIPTION:
              - RED_AIRWAY:
                  DIS: "red_airway"
                  NODE: <result_node_id>
                  QUANTITY: "pressure"
                  VALUE: <expected_pressure>
                  TOLERANCE: <result_tolerance>
              - RED_AIRWAY:
                  DIS: "red_airway"
                  ELEMENT: <result_acinus_element_id>
                  QUANTITY: "acini_volume"
                  VALUE: <expected_acinar_volume>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        def _positive(name: str, label: str, strict: bool = True) -> None:
            raw = params.get(name)
            if raw is None:
                return
            try:
                value = float(raw)
            except (TypeError, ValueError):
                issues.append(f"{name} must be a number, got {raw!r}.")
                return
            if strict and value <= 0:
                issues.append(f"{label} ({name}) must be > 0, got {value}.")
            elif not strict and value < 0:
                issues.append(f"{label} ({name}) must be >= 0, got {value}.")

        # MAT_fluid: the material of a RED_AIRWAY element.
        _positive("DYNVISCOSITY", "Airway fluid dynamic viscosity")
        _positive("DENSITY", "Airway fluid density")

        # MAT_0D_MAXWELL_ACINUS*: the material of a RED_ACINUS element.
        _positive("Stiffness1", "Acinar Maxwell branch-1 stiffness")
        _positive("Stiffness2", "Acinar Maxwell branch-2 stiffness")
        _positive("Viscosity1", "Acinar Maxwell branch-1 viscosity", strict=False)
        _positive("Viscosity2", "Acinar Maxwell branch-2 viscosity", strict=False)

        # RED_AIRWAY / RED_ACINUS element-line tokens.
        _positive("Area", "Airway cross-sectional Area")
        _positive("AcinusVolume", "Acinus volume")
        _positive("AlveolarDuctVolume", "Alveolar duct volume")
        _positive("WallElasticity", "Airway wall elasticity", strict=False)
        _positive("WallThickness", "Airway wall thickness", strict=False)

        # REDUCED DIMENSIONAL AIRWAYS DYNAMIC.
        _positive("TIMESTEP", "Time step")
        _positive("TOLERANCE", "Nonlinear tolerance")

        numstep = params.get("NUMSTEP")
        if numstep is not None:
            try:
                n = int(numstep)
                if n <= 0:
                    issues.append(f"NUMSTEP must be > 0, got {n}.")
            except (TypeError, ValueError):
                issues.append(
                    f"NUMSTEP must be a positive integer, got {numstep!r}."
                )

        maxit = params.get("MAXITERATIONS")
        if maxit is not None:
            try:
                m = int(maxit)
                if m < 1:
                    issues.append(f"MAXITERATIONS must be >= 1, got {m}.")
            except (TypeError, ValueError):
                issues.append(
                    f"MAXITERATIONS must be a positive integer, "
                    f"got {maxit!r}."
                )

        theta = params.get("THETA")
        if theta is not None:
            try:
                th = float(theta)
                if not 0.0 < th <= 1.0:
                    issues.append(
                        f"THETA must be in (0, 1], got {th}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"THETA must be a number in (0, 1], got {theta!r}."
                )

        # Keys that do not exist in 4C but that models routinely invent.
        replacements = {
            "MAXITER": (
                "MAXITERATIONS (MAXITER belongs to COUPLED REDUCED-D "
                "AIRWAYS AND TISSUE DYNAMIC)"
            ),
            "MAXTIME": (
                "nothing -- REDUCED DIMENSIONAL AIRWAYS DYNAMIC has no "
                "MAXTIME; the end time is TIMESTEP * NUMSTEP"
            ),
            "VISCOSITY": "DYNVISCOSITY (MAT_fluid)",
            "WALL_COMPLIANCE": (
                "the WallElasticity token on the RED_AIRWAY element line"
            ),
            "COMPLIANCE": (
                "Stiffness1 / Stiffness2 of MAT_0D_MAXWELL_ACINUS*"
            ),
            "ACINAR_COMPLIANCE": (
                "Stiffness1 / Stiffness2 of MAT_0D_MAXWELL_ACINUS*"
            ),
            "VOLUME_RELAXED": (
                "the AcinusVolume token on the RED_ACINUS element line"
            ),
            "ACINAR_VOLUME_RELAXED": (
                "the AcinusVolume token on the RED_ACINUS element line"
            ),
            "EXPONENT": (
                "the E1_EXP token on the RED_ACINUS element line"
            ),
        }
        for bad, good in replacements.items():
            if bad in params:
                issues.append(
                    f"{bad} is not a 4C reduced-airways parameter; use "
                    f"{good}."
                )

        return issues
