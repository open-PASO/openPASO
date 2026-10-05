"""Beam Interaction generator for 4C.

Covers beam-to-beam and beam-to-solid contact and meshtying.  Beam elements
(BEAM3R, BEAM3EB) can interact with other beams via contact or with solid
elements via embedded meshtying or contact.  Applications include fiber
networks, reinforced composites, stent deployment, and knitted/woven
textile mechanics.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class BeamInteractionGenerator(BaseGenerator):
    """Generator for beam interaction (contact/meshtying) problems in 4C."""

    module_key = "beam_interaction"
    display_name = "Beam Interaction (Beam-to-Beam/Beam-to-Solid Contact)"
    problem_type = "Structure"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The beam interaction module handles contact and meshtying "
                "between beam elements (beam-to-beam) and between beam "
                "and solid elements (beam-to-solid).  Beam-to-beam contact "
                "captures fibre-fibre interactions in networks, knots, and "
                "woven structures.  Beam-to-solid contact and meshtying "
                "couples 1-D beam elements with 3-D solid elements for "
                "applications like rebar in concrete, stent-in-vessel, or "
                "fibre-reinforced composites.  The PROBLEM TYPE is "
                "'Structure' (same as standard structural mechanics) but "
                "with additional BEAM INTERACTION sections.  The dynamics "
                "section is STRUCTURAL DYNAMIC.  Beams use BEAM3R or "
                "BEAM3EB elements; solids use standard SOLID HEX8/TET4.  "
                "Beams and solids live in the SAME structure "
                "discretisation, so both are declared in STRUCTURE "
                "ELEMENTS (legacy) or in one STRUCTURE GEOMETRY / "
                "ELEMENT_BLOCKS list -- there is no separate beam mesh "
                "section.  The top-level BEAM INTERACTION section only "
                "accepts REPARTITIONSTRATEGY and, before 4C 2026.2.0, "
                "SEARCH_STRATEGY (4C 2026.2.0 removed it and always searches "
                "with ArborX, so a 4C built without ArborX cannot run beam "
                "interaction); the "
                "contact algorithm and its penalty parameters live in the "
                "sub-sections (BEAM INTERACTION/BEAM TO BEAM CONTACT, "
                "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING, ...).  "
                "A BINNING STRATEGY section is typically required for "
                "efficient spatial search."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "STRUCTURAL DYNAMIC",
                "BEAM INTERACTION",
                "SOLVER 1",
                "MATERIALS",
                "BINNING STRATEGY",
            ],
            "optional_sections": [
                "BEAM INTERACTION/BEAM TO BEAM CONTACT",
                "BEAM INTERACTION/BEAM TO BEAM CONTACT CONDITIONS",
                "BEAM INTERACTION/BEAM TO BEAM CONTACT/RUNTIME VTK OUTPUT",
                "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING",
                "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING LINE",
                "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING VOLUME",
                "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING/RUNTIME VTK OUTPUT",
                "BEAM INTERACTION/BEAM TO SOLID SURFACE CONTACT",
                "BEAM INTERACTION/BEAM TO SOLID SURFACE MESHTYING",
                "BEAM INTERACTION/BEAM TO SOLID SURFACE/RUNTIME VTK OUTPUT",
                "STRUCTURAL DYNAMIC/GENALPHA",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/BEAMS",
                "IO/RUNTIME VTK OUTPUT/STRUCTURE",
            ],
            "materials": {
                "MAT_BeamReissnerElastHyper": {
                    "description": (
                        "Geometrically exact Reissner beam material.  "
                        "Defines the cross-sectional properties of "
                        "beam elements."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Young's modulus",
                            "range": "> 0",
                        },
                        "POISSONRATIO": {
                            "description": "Poisson's ratio",
                            "range": "(0, 0.5)",
                        },
                        "DENS": {
                            "description": "Mass density",
                            "range": "> 0",
                        },
                        "CROSSAREA": {
                            "description": "Cross-sectional area",
                            "range": "> 0",
                        },
                        "SHEARCORR": {
                            "description": "Shear correction factor",
                            "range": "> 0 (typically 1.0)",
                        },
                        "MOMINPOL": {
                            "description": "Polar moment of inertia",
                            "range": "> 0",
                        },
                        "MOMIN2": {
                            "description": "Second moment of area (axis 2)",
                            "range": "> 0",
                        },
                        "MOMIN3": {
                            "description": "Second moment of area (axis 3)",
                            "range": "> 0",
                        },
                    },
                },
                "MAT_Struct_StVenantKirchhoff": {
                    "description": (
                        "Standard structural material for solid elements "
                        "in beam-to-solid interaction."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Young's modulus",
                            "range": "> 0",
                        },
                        "NUE": {
                            "description": "Poisson's ratio",
                            "range": "[0, 0.5)",
                        },
                        "DENS": {
                            "description": "Mass density",
                            "range": "> 0",
                        },
                    },
                },
            },
            "solver": {
                "structure_solver": {
                    "type": "UMFPACK",
                    "notes": (
                        "Direct solver works well for beam-interaction "
                        "problems, which tend to have moderate DOF counts."
                    ),
                },
            },
            "beam_interaction_parameters": {
                "BEAM INTERACTION/REPARTITIONSTRATEGY": (
                    "'Adaptive' (default) or 'Everydt'.  How often the "
                    "beam discretisation is repartitioned.  This, plus "
                    "SEARCH_STRATEGY before 4C 2026.2.0, are the ONLY keys the "
                    "top-level BEAM INTERACTION section accepts."
                ),
                "BEAM INTERACTION/SEARCH_STRATEGY": (
                    "4C before 2026.2.0 only (4C 2026.2.0 removed it and always uses "
                    "ArborX): 'bruteforce_with_binning' (default) or "
                    "'bounding_volume_hierarchy'.  There is no search "
                    "RADIUS here -- the search extent comes from "
                    "BINNING STRATEGY/BIN_SIZE_LOWER_BOUND."
                ),
                "BEAM INTERACTION/BEAM TO BEAM CONTACT/BEAMS_BTBPENALTYPARAM": (
                    "Penalty parameter for beam-to-beam POINT contact.  "
                    "Larger values reduce penetration but worsen "
                    "conditioning.  Must be >= 0."
                ),
                "BEAM INTERACTION/BEAM TO BEAM CONTACT/BEAMS_BTBLINEPENALTYPARAM": (
                    "Penalty parameter per unit length for beam-to-beam "
                    "LINE contact (small-angle contact).  Required and "
                    "must be >= 0 whenever BEAMS_SEGCON is true."
                ),
                "BEAM INTERACTION/BEAM TO BEAM CONTACT/BEAMS_SEGCON": (
                    "All-angle-beam contact with subsegment generation.  "
                    "4C currently REQUIRES this to be true: with false "
                    "it aborts 'only all-angle-beam contact "
                    "(BEAMS_SEGCON) formulation tested yet in new beam "
                    "interaction framework!'.  Setting it true also "
                    "makes BEAMS_PERPSHIFTANGLE1/2, "
                    "BEAMS_PARSHIFTANGLE1/2 and BEAMS_SEGANGLE "
                    "mandatory (all in degrees, 0..90, and "
                    "PARSHIFTANGLE2 > PERPSHIFTANGLE1)."
                ),
                "BEAM INTERACTION/BEAM TO BEAM CONTACT/BEAMS_PENALTYLAW": (
                    "'LinPen' (default), 'QuadPen', 'LinNegQuadPen', "
                    "'LinPosQuadPen', 'LinPosCubPen', "
                    "'LinPosDoubleQuadPen', 'LinPosExpPen'.  Anything "
                    "other than LinPen/QuadPen additionally requires "
                    "BEAMS_PENREGPARAM_G0/_F0/_C0."
                ),
                "BEAM INTERACTION/BEAM TO BEAM CONTACT/BEAMS_GAPSHIFTPARAM": (
                    "Shift of the penalty law (the real name of what is "
                    "loosely called a gap shift).  Only permitted for "
                    "BEAMS_PENALTYLAW: LinPosQuadPen -- a non-zero value "
                    "with any other law aborts with 'BEAMS_GAPSHIFTPARAM "
                    "only possible for penalty law LinPosQuadPen!'."
                ),
                "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING/CONTACT_DISCRETIZATION": (
                    "'none' (default, i.e. inactive), "
                    "'gauss_point_to_segment', 'gauss_point_cross_section', "
                    "'mortar', 'mortar_cross_section'.  Setting this to "
                    "anything but 'none' is what ACTIVATES the "
                    "beam-to-solid volume meshtying model."
                ),
                "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING/CONSTRAINT_STRATEGY": (
                    "'none', 'penalty' or 'lagrange'.  This is the "
                    "beam-to-solid analogue of a 'coupling type'; there "
                    "is no COUPLING_TYPE key in the VOLUME meshtying "
                    "section (only the SURFACE meshtying section has one)."
                ),
                "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING/PENALTY_PARAMETER": (
                    "Penalty stiffness for beam-to-solid volume "
                    "meshtying.  Used when CONSTRAINT_STRATEGY: penalty."
                ),
            },
            "pitfalls": [
                (
                    "[Input] BINNING STRATEGY is required for spatial "
                    "search.  BIN_SIZE_LOWER_BOUND must be at least as "
                    "large as the element sizes to ensure all "
                    "interaction pairs are detected, and NOTHING WARNS "
                    "you when it is not. Signal: read the pair count. "
                    "The beam-contact submodel prints one line per step "
                    "whose greppable core is 'currently monitors' and "
                    "which ends in the number of beam contact pairs; a "
                    "bin size below the element size drops that number "
                    "to zero and the run still exits 0. Measured on the "
                    "shipped beam_interaction_beam_contact deck: "
                    "BIN_SIZE_LOWER_BOUND 5 monitors 3 pairs, "
                    "BIN_SIZE_LOWER_BOUND 0.05 monitors 0, both exit 0 "
                    "with no warning of any kind. The binning banner "
                    "also echoes 'half min bin size', which is the "
                    "value actually used. (Audit 2026-06-02; the quoted "
                    "'zero pairs found' does not exist in 4C, and "
                    "neither does 'bin size smaller than max element "
                    "size, search may miss pairs' -- both falsified and "
                    "replaced by execution 2026-08-09.)"
                ),
                (
                    "[Numerical] The penalty parameter for beam "
                    "contact must be carefully chosen.  Too large "
                    "causes ill-conditioning and convergence failure; "
                    "too small permits excessive penetration. Signal: "
                    "NOX condition-number printout exceeds ~1e14 "
                    "(too high), or post-processed beam-beam distance "
                    "at a contact point goes negative by more than "
                    "5% of beam radius (too low). (Audit 2026-06-02.)"
                ),
                (
                    "[Input] Beam-to-beam contact uses a "
                    "POINT-TO-POINT and LINE-TO-LINE "
                    "formulation (BEAMS_BTBPENALTYPARAM and "
                    "BEAMS_BTBLINEPENALTYPARAM), but there is NO "
                    "search-radius key for it: SEARCH_RADIUS is "
                    "legal only in FLUID BEAM INTERACTION/BEAM TO "
                    "FLUID MESHTYING, and writing it under BEAM "
                    "INTERACTION aborts with `Could not match this "
                    "input ... The following data remains unused: "
                    "SEARCH_RADIUS` from "
                    "core/io/src/4C_io_input_spec_builders.cpp. "
                    "The search extent is set by BINNING "
                    "STRATEGY/BIN_SIZE_LOWER_BOUND (plus "
                    "DOMAINBOUNDINGBOX), and before 4C 2026.2.0 the "
                    "algorithm by BEAM INTERACTION/SEARCH_STRATEGY "
                    "(bruteforce_with_binning or "
                    "bounding_volume_hierarchy); 4C 2026.2.0 and later have no such "
                    "key and always searches with ArborX. Signal: too-small "
                    "bins miss contact pairs — beams pass through "
                    "each other; too-large bins waste compute on "
                    "O(N^2) pair checks. (Audit 2026-06-02; "
                    "corrected by execution 2026-08-07.)"
                ),
                (
                    "[Numerical] For beam-to-solid volume meshtying, "
                    "the beam elements must lie within the solid "
                    "mesh volume.  Beams outside the solid domain "
                    "are not coupled. Signal: 4C prints NOTHING — "
                    "there is no BeamToSolidMeshtying diagnostic, "
                    "no `N beam segments coupled` line and no "
                    "warning of any kind, so grepping the log is "
                    "futile. The coupled count is only visible in "
                    "the beam-to-solid runtime VTK output, which "
                    "you have to switch on "
                    "(BEAM INTERACTION/BEAM TO SOLID VOLUME "
                    "MESHTYING/RUNTIME VTK OUTPUT with "
                    "WRITE_OUTPUT, SEGMENTATION and "
                    "INTEGRATION_POINTS): the segmentation file's "
                    "point count drops to zero and so does the "
                    "integration-point count. The solid then stops "
                    "moving altogether and the beam deforms as if "
                    "no solid existed, but the run only fails at "
                    "the final result test. (Audit 2026-06-02; "
                    "corrected by execution 2026-08-06.)"
                ),
                (
                    "[Numerical] Beam-to-solid surface contact "
                    "already subtracts the beam cross-section "
                    "radius: the gap is (r_beam - r_surface) . n "
                    "minus the beam interaction radius, so gap = 0 "
                    "means the beam's OUTER SURFACE touches the "
                    "solid, not its centerline, and there is no "
                    "offset for you to add. Signal: the radius is "
                    "taken from INTERACTIONRADIUS in the beam "
                    "material, or, when that is left out, derived "
                    "from the area moment of inertia assuming a "
                    "circular cross-section — so a wrong MOMIN2 "
                    "silently shifts every reported gap and moves "
                    "the contact onset. Check it against the `gap` "
                    "point array 4C writes into "
                    "BEAM INTERACTION/BEAM TO SOLID SURFACE/RUNTIME "
                    "VTK OUTPUT (set OUTPUT_DATA_FORMAT: ascii to "
                    "read it): changing the radius shifts every gap "
                    "value by exactly the same amount. (Audit "
                    "2026-06-02; falsified by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Input] Beam materials use dedicated types "
                    "(MAT_BeamReissnerElastHyper) with cross-"
                    "sectional properties (CROSSAREA, MOMINPOL, "
                    "MOMIN2, MOMIN3).  These must be geometrically "
                    "consistent with the beam cross-section. Signal: a "
                    "missing cross-section key gets you the GENERIC "
                    "input-spec abort and nothing beam-specific -- "
                    "\"Failed to match specification in section "
                    "'MATERIALS'. The error was:\" from "
                    "global_data/4C_global_data_read.cpp, then 'Could "
                    "not match this input', then a candidate list with "
                    "one block per material 4C knows. The name of the "
                    "key you dropped appears nowhere in it. Or BEAM3R "
                    "bends differently from Euler-Bernoulli "
                    "prediction (the eigen-frequency of a "
                    "cantilever differs from sqrt(EI/(rho*A*L^4)) "
                    "by more than 10%). (Audit 2026-06-02; the quoted "
                    "'MAT_BeamReissnerElastHyper requires CROSSAREA' "
                    "and 'MOMINPOL not specified' do not exist in 4C -- "
                    "falsified by execution 2026-08-09, dropping each "
                    "key in turn from beam3r_herm2line2_static_test1, "
                    "which exits 1 with the generic abort above.)"
                ),
                (
                    "[Output] Use IO/RUNTIME VTK OUTPUT/BEAMS for "
                    "beam visualisation.  Standard STRUCTURE output "
                    "only shows solid elements. Signal: visualize"
                    "('list') under work_dir shows structure-*.vtu "
                    "files but no beams-*.vtu; the LLM-visible "
                    "summary reports an empty beam-field group "
                    "though the run completed. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] PROBLEM TYPE is 'Structure' "
                    "(NOT a dedicated beam-interaction "
                    "type). The BEAM INTERACTION sub-sections "
                    "activate beam contact/meshtying on "
                    "top of the standard structural "
                    "problem. Signal: writing "
                    "PROBLEMTYPE: 'BeamInteraction' aborts "
                    "input reading with 'Could not match this "
                    "input' and \"[!] Candidate "
                    "deprecated_selection 'PROBLEMTYPE' has "
                    "wrong value, possible values: ...\" "
                    "listing the legal problem types; there is "
                    "no such enum. There is also no BEAM "
                    "INTERACTION/SUBMODEL section (it does not "
                    "exist anywhere in 4C's grammar). Per "
                    "adapter/4C_adapter_str_structure_new.cpp the "
                    "beam-interaction model is switched on by any "
                    "of: a BEAM INTERACTION/BEAM TO BEAM CONTACT "
                    "CONDITIONS entry; BEAM INTERACTION/BEAM TO "
                    "SOLID {VOLUME MESHTYING, SURFACE MESHTYING, "
                    "SURFACE CONTACT}/CONTACT_DISCRETIZATION set "
                    "to something other than none; BEAM "
                    "INTERACTION/BEAM TO SPHERE "
                    "CONTACT/STRATEGY != none; BEAM "
                    "INTERACTION/CROSSLINKING/CROSSLINKER: true; "
                    "or BEAM INTERACTION/SPHERE BEAM "
                    "LINK/SPHEREBEAMLINKING: true. (Audit "
                    "2026-06-02; corrected by execution "
                    "2026-08-07.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "fiber_network_contact",
                    "description": (
                        "Two crossing beams with contact.  Tests "
                        "beam-to-beam contact detection, penalty "
                        "enforcement, and large-deformation beam "
                        "mechanics."
                    ),
                    "template_variant": "beam_contact_3d",
                },
                {
                    "name": "beam_in_solid_meshtying",
                    "description": (
                        "A beam embedded in a solid block via volume "
                        "meshtying.  Tests beam-to-solid coupling, "
                        "constraint enforcement, and stress transfer."
                    ),
                    "template_variant": "beam_solid_meshtying_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "beam_contact_3d",
                "description": (
                    "3-D beam-to-beam contact: crossing fibers with "
                    "penalty contact.  BEAM3R elements, "
                    "MAT_BeamReissnerElastHyper, UMFPACK solver."
                ),
            },
            {
                "name": "beam_solid_meshtying_3d",
                "description": (
                    "3-D beam-to-solid volume meshtying: beam embedded "
                    "in solid block.  BEAM3R + SOLID HEX8, penalty "
                    "or mortar coupling, UMFPACK solver."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "beam_contact_3d") -> str:
        templates = {
            "beam_contact_3d": self._template_beam_contact_3d,
            "beam_solid_meshtying_3d": self._template_beam_solid_meshtying_3d,
        }
        if variant == "default":
            variant = "beam_contact_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_beam_contact_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D Beam-to-Beam Contact
            #
            # Two or more beams in contact.  Penalty-based contact detects
            # and resolves beam-beam interactions (crossing, sliding, etc.).
            #
            # Mesh: requires beam mesh with:
            #   element_block 1 = beam 1 (LINE2)
            #   element_block 2 = beam 2 (LINE2)
            #   node_set 1 = beam 1 clamped end
            #   node_set 2 = beam 2 clamped end
            # Both beams belong to the SAME structure discretisation, so
            # they are two element blocks of ONE STRUCTURE GEOMETRY block.
            # ---------------------------------------------------------------
            TITLE:
              - "3-D beam-to-beam contact -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Structure"
            IO:
              STDOUTEVERY: <stdout_interval>
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/BEAMS:
              OUTPUT_BEAMS: true
              DISPLACEMENT: true
              USE_ABSOLUTE_POSITIONS: true
              TRIAD_VISUALIZATIONPOINT: true

            # == Structural dynamics ===========================================
            STRUCTURAL DYNAMIC:
              DYNAMICTYPE: "Statics"
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              LINEAR_SOLVER: 1
              PREDICT: "ConstDisVelAcc"
              TOLRES: <residual_tolerance>
              TOLDISP: <displacement_tolerance>
              RESULTSEVERY: <results_output_interval>

            # == Beam interaction =============================================
            # The top-level section only takes these two keys.  There is no
            # STRATEGY and no SEARCH_RADIUS here; the search extent comes
            # from BINNING STRATEGY below.
            BEAM INTERACTION:
              REPARTITIONSTRATEGY: "Everydt"
            BEAM INTERACTION/BEAM TO BEAM CONTACT:
              # BEAMS_SEGCON must be true (4C rejects false), and it makes
              # the four shift angles and BEAMS_SEGANGLE mandatory.
              BEAMS_SEGCON: true
              BEAMS_PENALTYLAW: "LinPen"
              BEAMS_BTBPENALTYPARAM: <point_contact_penalty_parameter>
              BEAMS_BTBLINEPENALTYPARAM: <line_contact_penalty_parameter>
              BEAMS_PERPSHIFTANGLE1: <large_angle_fade_start_deg>
              BEAMS_PERPSHIFTANGLE2: <large_angle_fade_end_deg>
              BEAMS_PARSHIFTANGLE1: <small_angle_fade_start_deg>
              BEAMS_PARSHIFTANGLE2: <small_angle_fade_end_deg>
              BEAMS_SEGANGLE: <segmentation_angle_deg>
            # These conditions are what actually switches the beam contact
            # model on; E is a DLINE id, one per interacting beam, sharing
            # the same COUPLING_ID.
            BEAM INTERACTION/BEAM TO BEAM CONTACT CONDITIONS:
              - E: <beam1_line_id>
                COUPLING_ID: 1
              - E: <beam2_line_id>
                COUPLING_ID: 1

            # == Binning for spatial search ===================================
            BINNING STRATEGY:
              BIN_SIZE_LOWER_BOUND: <bin_size_lower_bound>
              DOMAINBOUNDINGBOX: "<xmin> <ymin> <zmin> <xmax> <ymax> <zmax>"

            # == Solver ========================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "structure_solver"

            # == Materials =====================================================
            MATERIALS:
              # Beam 1 material
              - MAT: 1
                MAT_BeamReissnerElastHyper:
                  YOUNG: <beam1_Young_modulus>
                  POISSONRATIO: <beam1_Poisson_ratio>
                  DENS: <beam1_density>
                  CROSSAREA: <beam1_cross_section_area>
                  SHEARCORR: <beam1_shear_correction_factor>
                  MOMINPOL: <beam1_polar_moment_of_inertia>
                  MOMIN2: <beam1_moment_of_inertia_2>
                  MOMIN3: <beam1_moment_of_inertia_3>
              # Beam 2 material
              - MAT: 2
                MAT_BeamReissnerElastHyper:
                  YOUNG: <beam2_Young_modulus>
                  POISSONRATIO: <beam2_Poisson_ratio>
                  DENS: <beam2_density>
                  CROSSAREA: <beam2_cross_section_area>
                  SHEARCORR: <beam2_shear_correction_factor>
                  MOMINPOL: <beam2_polar_moment_of_inertia>
                  MOMIN2: <beam2_moment_of_inertia_2>
                  MOMIN3: <beam2_moment_of_inertia_3>

            # == Boundary Conditions ===========================================

            # Beam 1: clamped end
            DESIGN POINT DIRICH CONDITIONS:
              - E: <beam1_clamped_node_set_id>
                NUMDOF: 6
                ONOFF: [1, 1, 1, 1, 1, 1]
                VAL: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0, 0, 0, 0]
              # Beam 2: clamped end
              - E: <beam2_clamped_node_set_id>
                NUMDOF: 6
                ONOFF: [1, 1, 1, 1, 1, 1]
                VAL: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0, 0, 0, 0]

            # Load: prescribed displacement or force on beam tip
            DESIGN POINT NEUMANN CONDITIONS:
              - E: <loaded_beam_tip_node_set_id>
                NUMDOF: 6
                ONOFF: [<dof1_load>, <dof2_load>, <dof3_load>, 0, 0, 0]
                VAL: [<load_value_1>, <load_value_2>, <load_value_3>, 0.0, 0.0, 0.0]
                FUNCT: [<load_function>, <load_function>, <load_function>, 0, 0, 0]

            # Load ramp function.  A POINT Neumann condition is evaluated as a
            # FunctionOfTime, so it must be SYMBOLIC_FUNCTION_OF_TIME here --
            # SYMBOLIC_FUNCTION_OF_SPACE_TIME aborts with "You tried to query
            # function N as a function of type FunctionOfTime".
            FUNCT<load_function>:
              - SYMBOLIC_FUNCTION_OF_TIME: "<load_ramp_expression>"

            # == Geometry ======================================================
            # One structure discretisation: every beam is an element block of
            # this single STRUCTURE GEOMETRY section.
            STRUCTURE GEOMETRY:
              FILE: "<beam_mesh_file>"
              ELEMENT_BLOCKS:
                # BEAM3R needs nodal triads: either TRIADS (6 doubles for
                # LINE2, 9 for LINE3) or NODAL_ROTATION_VECTORS naming a
                # cell-data field in the mesh file.
                - ID: 1
                  BEAM3R:
                    LINE2:
                      MAT: 1
                      NODAL_ROTATION_VECTORS: "<triad_cell_field_name>"
                - ID: 2
                  BEAM3R:
                    LINE2:
                      MAT: 2
                      NODAL_ROTATION_VECTORS: "<triad_cell_field_name>"

            RESULT DESCRIPTION:
              - STRUCTURE:
                  DIS: "structure"
                  NODE: <result_node_id>
                  QUANTITY: "dispx"
                  VALUE: <expected_displacement>
                  TOLERANCE: <result_tolerance>
        """)

    @staticmethod
    def _template_beam_solid_meshtying_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D Beam-to-Solid Volume Meshtying
            #
            # A beam element is embedded in a solid block.  The beam and
            # solid are coupled via volume meshtying (penalty or mortar).
            # The beam acts as a reinforcement inside the solid.
            #
            # Mesh: ONE mesh file for the whole structure discretisation --
            # beams and solids share it, there is no separate beam mesh
            # section in 4C:
            #   element_block 1 = solid block (HEX8)
            #   element_block 2 = beam (LINE3, Hermite centerline)
            #   node_set 1 = fixed solid face
            #   node_set 2 = loaded solid face
            #   side_set / node_set for the solid volume (DVOL) and the beam
            #   line (DLINE) used by the meshtying conditions
            # ---------------------------------------------------------------
            TITLE:
              - "3-D beam-to-solid volume meshtying -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Structure"
            IO:
              STDOUTEVERY: <stdout_interval>
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/BEAMS:
              OUTPUT_BEAMS: true
              DISPLACEMENT: true
              USE_ABSOLUTE_POSITIONS: true
            IO/RUNTIME VTK OUTPUT/STRUCTURE:
              OUTPUT_STRUCTURE: true
              DISPLACEMENT: true

            # == Structural dynamics ===========================================
            STRUCTURAL DYNAMIC:
              DYNAMICTYPE: "Statics"
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              LINEAR_SOLVER: 1
              PREDICT: "ConstDisVelAcc"
              TOLRES: <residual_tolerance>
              TOLDISP: <displacement_tolerance>
              RESULTSEVERY: <results_output_interval>

            # == Beam interaction =============================================
            # The top-level section only takes REPARTITIONSTRATEGY and
            # SEARCH_STRATEGY -- no STRATEGY, no SEARCH_RADIUS.
            BEAM INTERACTION:
              REPARTITIONSTRATEGY: "Everydt"
            BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING:
              # CONTACT_DISCRETIZATION != none is what activates the model.
              # There is no COUPLING_TYPE key here (only the SURFACE
              # meshtying section has one); use CONSTRAINT_STRATEGY.
              CONTACT_DISCRETIZATION: "<contact_discretization>"
              CONSTRAINT_STRATEGY: "<constraint_strategy>"
              PENALTY_PARAMETER: <meshtying_penalty_parameter>
              GAUSS_POINTS: <meshtying_gauss_points>
            BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING/RUNTIME VTK OUTPUT:
              WRITE_OUTPUT: true
              NODAL_FORCES: true
              SEGMENTATION: true
              INTEGRATION_POINTS: true

            # == Binning for spatial search ===================================
            BINNING STRATEGY:
              BIN_SIZE_LOWER_BOUND: <bin_size_lower_bound>
              DOMAINBOUNDINGBOX: "<xmin> <ymin> <zmin> <xmax> <ymax> <zmax>"

            # == Solver ========================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "structure_solver"

            # == Materials =====================================================
            MATERIALS:
              # Beam material
              - MAT: 1
                MAT_BeamReissnerElastHyper:
                  YOUNG: <beam_Young_modulus>
                  POISSONRATIO: <beam_Poisson_ratio>
                  DENS: <beam_density>
                  CROSSAREA: <beam_cross_section_area>
                  SHEARCORR: <beam_shear_correction_factor>
                  MOMINPOL: <beam_polar_moment_of_inertia>
                  MOMIN2: <beam_moment_of_inertia_2>
                  MOMIN3: <beam_moment_of_inertia_3>
              # Solid material
              - MAT: 2
                MAT_Struct_StVenantKirchhoff:
                  YOUNG: <solid_Young_modulus>
                  NUE: <solid_Poisson_ratio>
                  DENS: <solid_density>

            # == Boundary Conditions ===========================================

            # Solid: fixed face
            DESIGN SURF DIRICH CONDITIONS:
              - E: <solid_fixed_face_id>
                NUMDOF: 3
                ONOFF: [1, 1, 1]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]

            # Solid: loaded face
            DESIGN SURF NEUMANN CONDITIONS:
              - E: <solid_loaded_face_id>
                NUMDOF: 3
                ONOFF: [<dof1_load>, <dof2_load>, <dof3_load>]
                VAL: [<load_value_1>, <load_value_2>, <load_value_3>]
                FUNCT: [<load_function>, <load_function>, <load_function>]

            # Load ramp
            FUNCT<load_function>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<load_ramp_expression>"

            # == Meshtying coupling conditions =================================
            # The solid volume (DVOL) and the beam line (DLINE) that are tied
            # together, matched by a common COUPLING_ID.
            BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING VOLUME:
              - E: <solid_volume_id>
                COUPLING_ID: 1
            BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING LINE:
              - E: <beam_line_id>
                COUPLING_ID: 1

            # == Geometry ======================================================
            # There is no BEAM GEOMETRY section in 4C.  Beams and solids are
            # one discretisation, so the beam is simply another element block
            # of STRUCTURE GEOMETRY (or another line in STRUCTURE ELEMENTS).
            # Beam-to-solid meshtying requires a Hermite-centerline beam:
            # BEAM3R/LINE3 with HERMITE_CENTERLINE true, or BEAM3EB.  A plain
            # BEAM3R/LINE2 aborts with "Beam3tosolidmeshtying: beam::n_val_=2
            # detected for beam3r element w/o Hermite centerline".
            STRUCTURE GEOMETRY:
              FILE: "<structure_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  SOLID:
                    HEX8:
                      MAT: 2
                      KINEM: <kinematics>
                - ID: 2
                  BEAM3R:
                    LINE3:
                      MAT: 1
                      HERMITE_CENTERLINE: true
                      # BEAM3R needs nodal triads: either TRIADS (9 doubles
                      # for LINE3, 6 for LINE2) or NODAL_ROTATION_VECTORS
                      # naming a cell-data field in the mesh file.
                      NODAL_ROTATION_VECTORS: "<triad_cell_field_name>"

            RESULT DESCRIPTION:
              - STRUCTURE:
                  DIS: "structure"
                  NODE: <result_solid_node_id>
                  QUANTITY: "dispx"
                  VALUE: <expected_solid_displacement>
                  TOLERANCE: <result_tolerance>
              - STRUCTURE:
                  DIS: "structure"
                  NODE: <result_beam_node_id>
                  QUANTITY: "dispx"
                  VALUE: <expected_beam_displacement>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check beam Young's modulus
        young = params.get("YOUNG") or params.get("beam_YOUNG")
        if young is not None:
            try:
                e = float(young)
                if e <= 0:
                    issues.append(f"YOUNG must be > 0, got {e}.")
            except (TypeError, ValueError):
                issues.append(
                    f"YOUNG must be a positive number, got {young!r}."
                )

        # Check penalty parameters.  PENALTY_PARAMETER is the beam-to-solid
        # key; beam-to-beam contact uses BEAMS_BTBPENALTYPARAM (point) and
        # BEAMS_BTBLINEPENALTYPARAM (line, per unit length).
        for pkey in (
            "PENALTY_PARAMETER",
            "BEAMS_BTBPENALTYPARAM",
            "BEAMS_BTBLINEPENALTYPARAM",
        ):
            penalty = params.get(pkey)
            if penalty is None:
                continue
            try:
                p = float(penalty)
                if p < 0:
                    issues.append(f"{pkey} must be >= 0, got {p}.")
            except (TypeError, ValueError):
                issues.append(
                    f"{pkey} must be a non-negative number, "
                    f"got {penalty!r}."
                )

        # Check cross-sectional area
        crossarea = params.get("CROSSAREA")
        if crossarea is not None:
            try:
                a = float(crossarea)
                if a <= 0:
                    issues.append(
                        f"CROSSAREA must be > 0, got {a}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"CROSSAREA must be a positive number, "
                    f"got {crossarea!r}."
                )

        # SEARCH_RADIUS is not a beam-interaction key -- it exists only in
        # FLUID BEAM INTERACTION/BEAM TO FLUID MESHTYING.  Reject it here so
        # it never reaches a deck.
        if "SEARCH_RADIUS" in params:
            issues.append(
                "SEARCH_RADIUS is not a BEAM INTERACTION parameter; it is "
                "legal only in FLUID BEAM INTERACTION/BEAM TO FLUID "
                "MESHTYING.  Size the beam-interaction search via BINNING "
                "STRATEGY/BIN_SIZE_LOWER_BOUND; before 4C 2026.2.0 choose the "
                "algorithm with BEAM INTERACTION/SEARCH_STRATEGY (4C 2026.2.0 "
                "removed it and always uses ArborX)."
            )

        # Check the beam-to-beam contact enum choices
        penalty_law = params.get("BEAMS_PENALTYLAW")
        legal_laws = (
            "LinNegQuadPen", "LinPen", "LinPosCubPen",
            "LinPosDoubleQuadPen", "LinPosExpPen", "LinPosQuadPen",
            "QuadPen",
        )
        if penalty_law is not None and penalty_law not in legal_laws:
            issues.append(
                f"BEAMS_PENALTYLAW must be one of {legal_laws}, "
                f"got {penalty_law!r}."
            )

        # BEAMS_GAPSHIFTPARAM is only permitted for the LinPosQuadPen law
        gap_shift = params.get("BEAMS_GAPSHIFTPARAM")
        if gap_shift is not None:
            try:
                g = float(gap_shift)
            except (TypeError, ValueError):
                issues.append(
                    f"BEAMS_GAPSHIFTPARAM must be a number, "
                    f"got {gap_shift!r}."
                )
            else:
                if g != 0.0 and penalty_law != "LinPosQuadPen":
                    issues.append(
                        "BEAMS_GAPSHIFTPARAM is only possible for "
                        "BEAMS_PENALTYLAW: LinPosQuadPen; set the law or "
                        "leave the shift at 0."
                    )

        # 4C currently only supports the all-angle formulation
        segcon = params.get("BEAMS_SEGCON")
        if segcon is not None and segcon is not True and segcon != "true":
            issues.append(
                "BEAMS_SEGCON must be true: 4C rejects false with 'only "
                "all-angle-beam contact (BEAMS_SEGCON) formulation tested "
                "yet in new beam interaction framework!'."
            )

        # Check beam-to-solid volume meshtying enums
        disc = params.get("CONTACT_DISCRETIZATION")
        legal_disc = (
            "none", "gauss_point_to_segment", "gauss_point_cross_section",
            "mortar", "mortar_cross_section",
        )
        if disc is not None and disc not in legal_disc:
            issues.append(
                f"CONTACT_DISCRETIZATION must be one of {legal_disc}, "
                f"got {disc!r}."
            )

        strategy = params.get("CONSTRAINT_STRATEGY")
        if strategy is not None and strategy not in (
            "none", "penalty", "lagrange",
        ):
            issues.append(
                "CONSTRAINT_STRATEGY must be 'none', 'penalty' or "
                f"'lagrange', got {strategy!r}."
            )

        if "COUPLING_TYPE" in params:
            issues.append(
                "COUPLING_TYPE is not a key of BEAM INTERACTION/BEAM TO "
                "SOLID VOLUME MESHTYING (only the SURFACE MESHTYING "
                "section has one).  Use CONTACT_DISCRETIZATION and "
                "CONSTRAINT_STRATEGY instead."
            )

        # Check bin size
        bin_size = params.get("BIN_SIZE_LOWER_BOUND")
        if bin_size is not None:
            try:
                b = float(bin_size)
                if b <= 0:
                    issues.append(
                        f"BIN_SIZE_LOWER_BOUND must be > 0, got {b}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"BIN_SIZE_LOWER_BOUND must be a positive number, "
                    f"got {bin_size!r}."
                )

        # Check Poisson's ratio
        nue = params.get("NUE") or params.get("POISSONRATIO")
        if nue is not None:
            try:
                nu = float(nue)
                if nu <= 0 or nu >= 0.5:
                    issues.append(
                        f"Poisson's ratio must be in (0, 0.5), got {nu}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"Poisson's ratio must be a number, got {nue!r}."
                )

        return issues
