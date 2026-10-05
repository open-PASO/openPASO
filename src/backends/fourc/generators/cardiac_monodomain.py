"""Cardiac Monodomain generator for 4C.

Covers cardiac electrophysiology using the monodomain equation.  The
monodomain model is a reaction-diffusion PDE that describes the propagation
of the transmembrane potential through cardiac tissue.  It couples with
ionic cell models (e.g. Bueno-Orovio minimal, ten Tusscher) that represent the
local ionic currents at each point in the myocardium.  Applications include
simulation of action potential propagation, arrhythmia modeling, and
cardiac resynchronization therapy planning.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class CardiacMonodomainGenerator(BaseGenerator):
    """Generator for cardiac monodomain electrophysiology problems in 4C."""

    module_key = "cardiac_monodomain"
    display_name = "Cardiac Monodomain (Electrophysiology)"
    problem_type = "Cardiac_Monodomain"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The cardiac monodomain module solves a reaction-diffusion "
                "equation for the transmembrane potential in cardiac tissue.  "
                "The monodomain equation is a simplified model derived from "
                "the bidomain equations under the assumption of equal "
                "anisotropy ratios in intra- and extracellular conductivity "
                "tensors.  The reaction term is provided by an ionic cell "
                "model (e.g. Bueno-Orovio minimal, FitzHugh-Nagumo, "
                "ten Tusscher) "
                "that describes local ionic currents.  The diffusion tensor "
                "is anisotropic, aligned with the myocardial fiber "
                "orientation.  The PROBLEM TYPE is 'Cardiac_Monodomain'.  "
                "The dynamics section is 'CARDIAC MONODOMAIN DYNAMIC' or "
                "uses SCALAR TRANSPORT DYNAMIC with an appropriate "
                "reaction material.  The mesh represents the myocardial "
                "tissue domain with fiber orientation data."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "SCALAR TRANSPORT DYNAMIC",
                "SCALAR TRANSPORT DYNAMIC/STABILIZATION",
                "SOLVER 1",
                "MATERIALS",
            ],
            "optional_sections": [
                "SCALAR TRANSPORT DYNAMIC/NONLINEAR",
                "IO",
                "IO/RUNTIME VTK OUTPUT",
                "RESULT DESCRIPTION",
            ],
            "materials": {
                "MAT_myocard": {
                    "description": (
                        "Cardiac tissue material for the monodomain "
                        "equation.  Defines tissue conductivity (diffusion "
                        "coefficient), membrane capacitance, and the ionic "
                        "cell model for the reaction term."
                    ),
                    "parameters": {
                        "DIFF1": {
                            "description": (
                                "Diffusion coefficient along fiber "
                                "direction [mm^2/ms]"
                            ),
                            "range": "> 0",
                        },
                        "DIFF2": {
                            "description": (
                                "Diffusion coefficient transverse to "
                                "fiber (sheet direction) [mm^2/ms]"
                            ),
                            "range": "> 0",
                        },
                        "DIFF3": {
                            "description": (
                                "Diffusion coefficient in sheet-normal "
                                "direction [mm^2/ms]"
                            ),
                            "range": "> 0 (often = DIFF2)",
                        },
                        "PERTURBATION_DERIV": {
                            "description": (
                                "Perturbation for numerical derivatives "
                                "of the ionic model"
                            ),
                            "range": "> 0 (typically 1e-6)",
                        },
                        "MODEL": {
                            "description": (
                                "Ionic cell model: 'MV' (Bueno-Orovio "
                                "minimal model, default), 'FHN' "
                                "(FitzHugh-Nagumo), 'TNNP' (ten Tusscher), "
                                "'SAN' or 'INADA'; any other string aborts "
                                "('Myocard Material type is not supported!')"
                            ),
                            "range": "string",
                        },
                    },
                },
                "MAT_scatra_reaction": {
                    "description": (
                        "Alternative: generic scalar transport reaction "
                        "material that can be configured for cardiac "
                        "monodomain-like problems."
                    ),
                },
            },
            "solver": {
                "direct": {
                    "type": "UMFPACK",
                    "notes": (
                        "Suitable for small cardiac meshes.  The "
                        "monodomain system is symmetric positive definite."
                    ),
                },
                "iterative": {
                    "type": "Belos with MueLu AMG",
                    "notes": (
                        "Required for large 3-D cardiac meshes (millions "
                        "of unknowns).  AMG is very efficient for the "
                        "diffusion operator."
                    ),
                },
            },
            "time_integration": {
                "TIMESTEP": (
                    "Time step size [ms].  Must resolve the action "
                    "potential upstroke (typically 0.01-0.1 ms for "
                    "detailed ionic models)."
                ),
                "NUMSTEP": "Total number of time steps.",
                "MAXTIME": (
                    "Maximum simulation time [ms].  Typical single beat: "
                    "500-1000 ms."
                ),
                "THETA": (
                    "Implicit time integration parameter.  theta=0.5 "
                    "(Crank-Nicolson) or theta=1.0 (backward Euler).  "
                    "CN is second-order but may oscillate; BE is stable."
                ),
            },
            "fiber_orientation": {
                "description": (
                    "Cardiac tissue has a complex fiber architecture.  "
                    "Fiber orientations can be provided via element-wise "
                    "coordinate systems (LOCSYS) or through a fiber "
                    "field read from the mesh file."
                ),
            },
            "pitfalls": [
                (
                    "[Numerical] The time step must resolve the action- "
                    "potential upstroke AND the stimulus WINDOW. 4C samples the "
                    "stimulus function at the new time only, so a step larger "
                    "than the window simply never sees it. Signal: too large a "
                    "dt gives MISSED ACTIVATION - phi stays exactly at the "
                    "resting value for the whole run - and nothing is printed: "
                    "the solve converges and reports a clean run of a problem "
                    "with no stimulus in it. Check dt against the width of your "
                    "stimulus interval, not only against the upstroke. (Audit "
                    "2026-08-06, verified by execution.)"
                ),
                (
                    "[Mesh] Mesh resolution must resolve the AP wavefront; "
                    "typical element size is 0.2-0.5 mm for cardiac tissue. "
                    "Signal: none from 4C. A coarse mesh does not warn and does "
                    "not fail - it just propagates the front more slowly, so "
                    "the conduction velocity comes out too low by tens of "
                    "percent. Measure it: refine the mesh, compare the front "
                    "position at a fixed time, and refine until it stops "
                    "moving. (Audit 2026-08-06, verified by execution.)"
                ),
                (
                    "[Input] Fiber orientation only exists for an ANISOTROPIC "
                    "diffusion tensor. With DIFF1 = DIFF2 = DIFF3 in "
                    "MAT_myocard the FIBER1 vector on the TRANSP element line "
                    "is completely inert: propagation along the fiber and "
                    "across it is bit-for-bit identical, i.e. spherical. "
                    "Signal: none - 4C never mentions fibers. Set DIFF1 > DIFF2 "
                    "= DIFF3 for myocardium and confirm the wavefront is "
                    "elliptical by comparing the front along and across FIBER1. "
                    "(Audit 2026-08-06, verified by execution.)"
                ),
                (
                    "[Input] MODEL in MAT_myocard selects the ionic cell model "
                    "and with it the UNITS of phi, not just the AP shape: FHN's "
                    "phi is a dimensionless 0..1 excitation variable while "
                    "TNNP's is a transmembrane potential in mV resting near "
                    "-85. Signal: none - MODEL is an enum, 4C accepts any "
                    "combination silently and returns a value on the new "
                    "model's scale. Changing MODEL means re-deriving the "
                    "initial field, the stimulus amplitude and every expected "
                    "value; 4C's own FHN and TNNP decks differ in all three. "
                    "(Audit 2026-08-06, verified by execution.)"
                ),
                (
                    "[Input] Cardiac EP units are typically mm, ms, mV, "
                    "uA/cm^2, and DIFF in MAT_myocard is a length-squared per "
                    "time. Signal: none - a diffusivity written in m^2 instead "
                    "of mm^2 is accepted without a word and does not merely "
                    "slow the wave by 10x or 100x, it stops it: the excited "
                    "patch decays in place and nothing propagates, while the "
                    "solver converges normally throughout. Verify with a planar "
                    "wave and a measured conduction velocity. (Audit "
                    "2026-08-06, verified by execution.)"
                ),
                (
                    "[Input] Stimulation is applied via a Neumann condition "
                    "(e.g. DESIGN VOL / SURF TRANSPORT NEUMANN CONDITIONS) and "
                    "must be SUPRATHRESHOLD. Signal: a subthreshold amplitude "
                    "produces NO action potential at all - phi ends exactly at "
                    "the model's resting value - while the run converges "
                    "normally and prints no threshold warning. The failure is a "
                    "cliff, not a gradient: halving a calibrated stimulus can "
                    "still fire while dividing it by a hundred fires nothing. "
                    "(Audit 2026-08-06, verified by execution.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "planar_wave_3d",
                    "description": (
                        "Planar action potential wave propagating through "
                        "a 3-D slab of cardiac tissue.  One face is "
                        "stimulated, the wave propagates to the opposite "
                        "face.  Tests conduction velocity, wavefront "
                        "shape, and ionic model integration."
                    ),
                    "template_variant": "monodomain_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "monodomain_3d",
                "description": (
                    "3-D cardiac monodomain: action potential propagation "
                    "in a tissue slab.  MAT_myocard with a selectable "
                    "ionic model, anisotropic diffusion, UMFPACK solver."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "monodomain_3d") -> str:
        templates = {
            "monodomain_3d": self._template_monodomain_3d,
        }
        if variant == "default":
            variant = "monodomain_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_monodomain_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D Cardiac Monodomain (Electrophysiology)
            #
            # Action potential propagation through a cardiac tissue slab.
            # The monodomain equation is a reaction-diffusion PDE with
            # anisotropic diffusion (fiber-aligned) and an ionic cell
            # model providing the reaction term.
            #
            # Mesh: exodus file with:
            #   element_block 1 = cardiac tissue (HEX8 or TET4)
            #   node_set 1 = stimulation face
            #   node_set 2 = recording face (optional)
            # ---------------------------------------------------------------
            TITLE:
              - "3-D cardiac monodomain -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Cardiac_Monodomain"
            IO:
              STDOUTEVERY: <stdout_interval>
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>

            # == Scalar Transport (monodomain solver) ==========================
            SCALAR TRANSPORT DYNAMIC:
              SOLVERTYPE: "nonlinear"
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              THETA: <time_integration_theta>
              MATID: <tissue_material_id>
              INITIALFIELD: "field_by_function"
              INITFUNCNO: <initial_potential_function_id>
              LINEAR_SOLVER: 1
              RESULTSEVERY: <results_output_interval>
              RESTARTEVERY: <restart_interval>
            SCALAR TRANSPORT DYNAMIC/STABILIZATION:
              STABTYPE: "no_stabilization"
            SCALAR TRANSPORT DYNAMIC/NONLINEAR:
              ITEMAX: <max_nonlinear_iterations>
              CONVTOL: <nonlinear_convergence_tolerance>

            # == Solver ========================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "monodomain_solver"

            # == Materials =====================================================
            MATERIALS:
              - MAT: <tissue_material_id>
                MAT_myocard:
                  DIFF1: <diffusion_fiber>
                  DIFF2: <diffusion_sheet>
                  DIFF3: <diffusion_normal>
                  PERTURBATION_DERIV: <perturbation_derivative>
                  MODEL: "<ionic_cell_model>"
                  TIME_SCALE: <time_scale_factor>  # required; upstream decks use 1

            # == Initial condition: resting potential ==========================
            FUNCT<initial_potential_function_id>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<resting_potential_expression>"

            # == Stimulation (Neumann BC on stimulus face) =====================
            DESIGN SURF TRANSPORT NEUMANN CONDITIONS:
              - E: <stimulus_face_id>
                NUMDOF: 1
                ONOFF: [1]
                VAL: [<stimulus_current_density>]
                FUNCT: [<stimulus_time_function>]

            # Stimulus time function (square pulse)
            FUNCT<stimulus_time_function>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<stimulus_pulse_expression>"

            # == Geometry ======================================================
            TRANSPORT GEOMETRY:
              FILE: "<mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  TRANSP:
                    HEX8:
                      MAT: <tissue_material_id>
                      TYPE: Cardiac

            RESULT DESCRIPTION:
              - SCATRA:
                  DIS: "scatra"
                  NODE: <result_node_id>
                  QUANTITY: "phi"
                  VALUE: <expected_potential>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check diffusion coefficients
        for key in ("DIFF1", "DIFF2", "DIFF3"):
            val = params.get(key)
            if val is not None:
                try:
                    d = float(val)
                    if d <= 0:
                        issues.append(f"{key} must be > 0, got {d}.")
                except (TypeError, ValueError):
                    issues.append(
                        f"{key} must be a positive number, got {val!r}."
                    )

        # Check diffusion anisotropy ratio
        d1 = params.get("DIFF1")
        d2 = params.get("DIFF2")
        if d1 is not None and d2 is not None:
            try:
                ratio = float(d1) / float(d2)
                if ratio > 100:
                    issues.append(
                        f"Very high anisotropy ratio DIFF1/DIFF2 = {ratio:.1f}.  "
                        f"This may require very fine mesh resolution "
                        f"transverse to fibers."
                    )
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        # Check timestep (cardiac-specific)
        timestep = params.get("TIMESTEP")
        if timestep is not None:
            try:
                dt = float(timestep)
                if dt <= 0:
                    issues.append(f"TIMESTEP must be > 0, got {dt}.")
                elif dt > 0.1:
                    issues.append(
                        f"TIMESTEP = {dt} ms may be too large for cardiac "
                        f"monodomain.  Typical range: 0.01-0.1 ms."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"TIMESTEP must be a positive number, "
                    f"got {timestep!r}."
                )

        # Check THETA
        theta = params.get("THETA")
        if theta is not None:
            try:
                th = float(theta)
                if th < 0.5 or th > 1.0:
                    issues.append(
                        f"THETA should be in [0.5, 1.0], got {th}.  "
                        f"0.5 = Crank-Nicolson, 1.0 = backward Euler."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"THETA must be a number in [0.5, 1.0], "
                    f"got {theta!r}."
                )

        # Check ionic model
        model = params.get("MODEL")
        known_models = {
            "MV", "FHN", "TNNP", "SAN", "INADA",
        }
        if model is not None and model not in known_models:
            issues.append(
                f"MODEL '{model}' is not a recognized ionic model.  "
                f"Known models: {sorted(known_models)}."
            )

        return issues
