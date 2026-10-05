"""Thermo-Structure Interaction (TSI) generator for 4C.

Covers monolithic and partitioned coupling of thermal and structural fields.
The thermal field solves the heat equation while the structural field solves
the momentum equation with thermal expansion.  The two fields are coupled
through the thermal stress (THR -> STR) and deformation-dependent heat
conduction (STR -> THR).
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class TSIGenerator(BaseGenerator):
    """Generator for Thermo-Structure Interaction problems in 4C."""

    module_key = "tsi"
    display_name = "Thermo-Structure Interaction (TSI)"
    problem_type = "Thermo_Structure_Interaction"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Thermo-Structure Interaction couples a thermal field (heat "
                "equation) with a structural mechanics field (momentum balance "
                "with thermal strain).  The thermal field produces temperature "
                "gradients that cause thermal expansion in the structure, and "
                "the deforming structure can modify the heat conduction path.  "
                "4C supports monolithic (fully coupled, simultaneous solve) and "
                "partitioned (iterative staggered) coupling strategies.  The "
                "PROBLEM TYPE is 'Thermo_Structure_Interaction'.  Three "
                "dynamics sections are required: STRUCTURAL DYNAMIC, "
                "THERMAL DYNAMIC, and TSI DYNAMIC.  The structural mesh uses "
                "SOLIDSCATRA elements (not plain SOLID) to carry the thermal "
                "DOF.  A CLONING MATERIAL MAP maps the structural material to "
                "the thermal material."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "STRUCTURAL DYNAMIC",
                "THERMAL DYNAMIC",
                "TSI DYNAMIC",
                "TSI DYNAMIC/MONOLITHIC",
                "SOLVER 1",
                "SOLVER 2",
                "MATERIALS",
                "CLONING MATERIAL MAP",
            ],
            "optional_sections": [
                "IO",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/STRUCTURE",
                "THERMAL DYNAMIC/RUNTIME VTK OUTPUT",
                "STRUCTURAL DYNAMIC/GENALPHA",
            ],
            "materials": {
                "MAT_Struct_ThermoStVenantK": {
                    "description": (
                        "Thermo-elastic St. Venant-Kirchhoff material.  Extends "
                        "the standard SVK material with thermal expansion "
                        "coefficient and reference temperature.  On a 4C "
                        "before 2026.3.0 it can name a thermal material via "
                        "THERMOMAT (optional); 4C 2026.3.0 removed the key."
                    ),
                    "parameters": {
                        "YOUNGNUM": {
                            "description": (
                                "Number of Young's modulus entries (typically 1 "
                                "for isotropic)"
                            ),
                            "range": ">= 1",
                        },
                        "YOUNG": {
                            "description": (
                                "Young's modulus array [E] (one entry per "
                                "YOUNGNUM)"
                            ),
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
                            "description": (
                                "Coefficient of thermal expansion alpha_T "
                                "[1/K]"
                            ),
                            "range": "> 0 (typical metals: 1e-6 to 3e-5)",
                        },
                        "INITTEMP": {
                            "description": (
                                "Reference temperature at which thermal strain "
                                "is zero"
                            ),
                            "range": "any (often 0 or 293 K)",
                        },
                        "THERMOMAT": {
                            "description": (
                                "Material ID of the associated thermal material "
                                "(MAT_Fourier). 4C before 2026.3.0 only: 4C 2026.3.0 has "
                                "no THERMOMAT and rejects the MATERIALS section "
                                "that sets it"
                            ),
                            "range": "valid MAT ID",
                        },
                    },
                },
                "MAT_Fourier": {
                    "description": (
                        "Fourier heat conduction material.  Defines volumetric "
                        "heat capacity and thermal conductivity for the "
                        "thermal field."
                    ),
                    "parameters": {
                        "CAPA": {
                            "description": (
                                "Volumetric heat capacity rho * c_p "
                                "[J/(m^3 K)]"
                            ),
                            "range": "> 0",
                        },
                        "CONDUCT": {
                            "description": (
                                "Thermal conductivity k [W/(m K)].  Specified "
                                "as a YAML mapping with 'constant: [value]'."
                            ),
                            "range": "> 0",
                        },
                    },
                },
            },
            "solver": {
                "field_solvers": {
                    "type": "UMFPACK (direct)",
                    "notes": (
                        "Individual field solvers (SOLVER 1) for structure "
                        "and thermal used by the partitioned approach or as "
                        "sub-solvers."
                    ),
                },
                "monolithic_solver": {
                    "type": "Belos with Teko block preconditioner",
                    "notes": (
                        "The monolithic TSI solver (SOLVER 2) uses Belos "
                        "iterative solver with AZPREC: Teko and a "
                        "thermo_solid block preconditioner XML file.  "
                        "For small problems UMFPACK can also be used."
                    ),
                },
            },
            "time_integration": {
                "STRUCTURAL DYNAMIC": (
                    "DYNAMICTYPE: 'Statics' for quasi-static thermo-mechanical "
                    "problems.  'GenAlpha' for transient dynamics with thermal "
                    "coupling."
                ),
                "THERMAL DYNAMIC": (
                    "DYNAMICTYPE: 'Statics' for steady-state thermal field "
                    "within each TSI step.  'OneStepTheta' or 'GenAlpha' for "
                    "transient thermal analysis."
                ),
                "TSI DYNAMIC": (
                    "Controls the overall coupled TSI time stepping.  "
                    "TIMESTEP, NUMSTEP, MAXTIME define the global time loop.  "
                    "ITEMAX sets max coupling iterations per step."
                ),
            },
            "cloning_material_map": {
                "purpose": (
                    "Maps the structural material to the thermal field "
                    "material.  4C clones the structure mesh to create the "
                    "thermo discretisation; this map tells it which material "
                    "to assign.  SRC_FIELD: structure -> TAR_FIELD: thermo."
                ),
            },
            "pitfalls": [
                # Every Signal: below was produced by running
                #   LD_LIBRARY_PATH=/opt/4C-dependencies/lib stdbuf -oL -eL \
                #     {FOURC_BINARY} <deck>.yaml <out>
                # on 4C 2026.2.0-dev (commit 89519cfe76), mutating one key at a
                # time in upstream decks that all run clean unmutated:
                #   tests/input_files/tsi_lindilatation_geolin.4C.yaml
                #     (2 SOLIDSCATRA HEX8, tsi_oneway, UMFPACK, 4 pinned RESULT
                #      DESCRIPTION values, 1.0 s, exit 0)
                #   tests/input_files/tsi_lincompression_monolithic.4C.yaml
                #     (tsi_monolithic, Belos+Teko; needs tests/input_files/xml/
                #      copied next to the deck)
                #   tests/input_files/tsi_heatflux_monolithic.4C.yaml
                #     (carries a DESIGN SURF THERMO NEUMANN heat flux)
                (
                    "[Input] 4C runs NO 2D TSI, measured on a 4C before 2026.2.0, "
                    "on 2026.2.0 and on 2026.3.0, and every TSI deck in the "
                    "upstream corpus is "
                    "3D. A 2D thermo-mechanical deck dead-ends whichever "
                    "element you reach for, and by different messages, so do "
                    "not grep for one. Signal: on the one-way dilatation deck "
                    "with the HEX8 pair swapped for QUAD4s, a 4C before 2026.2.0 answers "
                    "SOLIDSCATRA QUAD4 with \"Element 'SOLIDSCATRA' does not "
                    "seem to know cell type 'quad4'.\" and SOLID QUAD4 with "
                    "the same sentence naming 'SOLID', both from "
                    "core/fem/src/general/element/"
                    "4C_fem_general_element_definition.cpp:29, before a mesh "
                    "is built; WALL QUAD4 with MAT_Struct_ThermoStVenantK gets "
                    "FURTHER and then aborts with 'Unsupported solid element "
                    "type!' from src/tsi/4C_tsi_utils.cpp:76. 4C 2026.2.0 and "
                    "2026.3.0 have no WALL and accept SOLIDSCATRA QUAD4 and "
                    "SOLID QUAD4 "
                    "(with THICKNESS and PLANE_ASSUMPTION); both build the "
                    "structure and then abort with the same 'Unsupported solid "
                    "element type!', from src/tsi/4C_tsi_utils.cpp:77. All "
                    "exit 1. An earlier version of this entry attributed the WALL "
                    "case to 'Invalid type of material law for wall element' in "
                    "4C_w1_mat.cpp — that string is real code but is NEVER "
                    "reached, because TSI's clone strategy rejects any non-"
                    "SolidScatra element before a material is evaluated. For a "
                    "plane-strain problem call the BACKEND, not this class. "
                    "THE EXACT WORKING CALL, verified 2026-08-16 by generating "
                    "the deck and running it to 'processor 0 finished "
                    "normally' on the built 4C binary — copy it verbatim into "
                    "a generator script and hand that script to "
                    "run_with_generator(solver='4c', generator_script=...):\n"
                    "    import sys\n"
                    "    sys.path.insert(0, '<openPASO repo>/src')\n"
                    "    from backends.fourc.backend import FourcBackend\n"
                    "    deck = FourcBackend().generate_input(\n"
                    "        'tsi', 'plane_strain_2d', {})\n"
                    "    open('tsi_plane_strain.4C.yaml', 'w').write(deck)\n"
                    "Note the class is FourcBackend (lower-case c in the "
                    "middle); FourCBackend raises ImportError. There is no "
                    "free function called generate_input and no MCP tool of "
                    "that name — earlier text here quoted it bare, as if it "
                    "were callable on its own, which is why agents that read "
                    "this pitfall still concluded 4C could not do plane "
                    "strain. The deck it returns is a "
                    "one-element-thick 3D SOLIDSCATRA HEX8 slab with u_z fixed "
                    "on all nodes and the temperature imposed volume-wide via "
                    "DESIGN VOL THERMO DIRICH + a symbolic FUNCT, whereas "
                    "TSIGenerator.get_template('plane_strain_2d') raises a "
                    "ValueError reading Unknown variant, then Available: "
                    "monolithic_3d. Note whose message that is: it comes from "
                    "THIS repository, src/backends/fourc/generators/tsi.py:451, "
                    "and never reaches 4C — it is not a 4C diagnostic and there "
                    "is nothing in the 4C tree to grep for it. Both checked "
                    "2026-08-09. "
                    "(Verified by execution 2026-08-09.)"
                ),
                (
                    "[Input] One-way thermo->structure TSI REQUIRES TSI DYNAMIC/"
                    "PARTITIONED: COUPVARIABLE: Temperature. The 4C default is "
                    "Displacement, i.e. structure->thermo, which is the wrong "
                    "direction for every heating problem and produces no thermal "
                    "forcing at all. Signal: this is the silent one. Deleting the "
                    "TSI DYNAMIC/PARTITIONED section from the tsi_oneway "
                    "dilatation deck leaves the temperature field perfectly "
                    "correct (temp at node 1 still 294, abs(diff) 0.0) while the "
                    "structural answer collapses to EXACTLY zero — the result "
                    "test prints 'dispy ... is WRONG --> actresult= "
                    "0.00000000000000000e+00, givenresult= "
                    "2.00000000000000016e-05', and dispz likewise. Not small, not "
                    "noisy: bit-zero. There is no warning, no message and, "
                    "without a RESULT DESCRIPTION, exit 0. If a heated bar does "
                    "not move at all, check COUPVARIABLE before anything else. "
                    "(Verified by execution 2026-08-09.)"
                ),
                (
                    "[Input] TSI structural elements MUST be SOLIDSCATRA, not "
                    "plain SOLID — only the solid-scatra element carries the "
                    "second dof set that the thermal field is cloned onto. "
                    "Signal: this fails LOUDLY, contrary to an earlier version of "
                    "this entry which claimed SOLID 'will silently omit the "
                    "thermal coupling'. Replacing SOLIDSCATRA HEX8 with SOLID "
                    "HEX8 in the dilatation deck aborts at exit 1 with "
                    "'Unsupported solid element type!' from "
                    "src/tsi/4C_tsi_utils.cpp:76, thrown by "
                    "ThermoStructureCloneStrategy::set_element_data while it "
                    "builds the thermo discretisation — before any "
                    "material is evaluated and before the first time step. Drop "
                    "the element line's TYPE token when you swap, or you never "
                    "reach the clone: SOLID HEX8 ... TYPE Undefined is rejected "
                    "earlier by the element spec with 'After parsing, the line "
                    "still contains' plus the leftover token, from "
                    "core/io/src/4C_io_input_spec.cpp:33. The clone message "
                    "names neither SOLID nor SOLIDSCATRA, so grep the "
                    "file name 4C_tsi_utils.cpp, and note it is the SAME message "
                    "a 2D element produces (WALL on a 4C before 2026.2.0, SOLID "
                    "or SOLIDSCATRA on 2026.2.0 and 2026.3.0). Those four words "
                    "are all 4C "
                    "prints — read them as meaning the element cannot be cloned "
                    "into a thermo field, whatever the reason, but do not expect "
                    "any such sentence in the output. "
                    "(Verified by execution 2026-08-09; re-executed 2026-08-10.)"
                ),
                (
                    "[Input] CLONING MATERIAL MAP is required: the thermo "
                    "discretisation is not in the input file, it is cloned from "
                    "the structure, and the map supplies the thermal material "
                    "(SRC_FIELD: structure, SRC_MAT: <struct id>, TAR_FIELD: "
                    "thermo, TAR_MAT: <MAT_Fourier id>). Signal: deleting it from "
                    "the dilatation deck aborts at exit 1 with 'At least one "
                    "material pairing required in --CLONING MATERIAL MAP.' from "
                    "core/fem/src/general/utils/"
                    "4C_fem_general_utils_createdis.hpp:318. The sentence comes "
                    "from the shared cloning helper, so it mentions neither TSI "
                    "nor the thermo field and still spells the section in the "
                    "retired '--SECTION' dat form — which is NOT how you write it "
                    "in YAML. FSI's missing ALE map emits the identical sentence; "
                    "the backtrace (ThermoStructureCloneStrategy vs "
                    "AleCloneStrategy) is what distinguishes them. All 42 "
                    "upstream tsi_*.4C.yaml decks map to a MAT_Fourier target; "
                    "pointing TAR_MAT back at the structural material instead "
                    "segfaults during setup. (Verified by execution 2026-08-09.)"
                ),
                (
                    "[Input] THERMOMAT in MAT_Struct_ThermoStVenantK is OPTIONAL "
                    "and it is NOT the thermal-strain link — an earlier version "
                    "of this entry said thermal strains are not computed without "
                    "it, and that is false. The grammar declares it "
                    "required: false, default: -1 (`4C -p`), and "
                    "create_thermo_material_if_set() in "
                    "src/mat/4C_mat_thermostvenantkirchhoff.cpp only attaches a "
                    "Mat::Trait::Thermo so the STRUCTURAL material can answer "
                    "conductivity()/capacity()/heat-flux queries. Thermal strain "
                    "comes from THEXPANS and INITTEMP. Signal: there is nothing "
                    "to observe, and that IS the diagnostic — on a 4C before "
                    "2026.3.0 (2026.3.0 has no THERMOMAT key) deleting THERMOMAT "
                    "from the one-way dilatation deck AND from "
                    "tsi_lincompression_monolithic leaves every pinned value "
                    "CORRECT and unchanged (dispy at node 7 still matches 2e-05 "
                    "with abs(diff) 3.29e-19; the monolithic deck's five "
                    "displacement and three temperature tests all still pass), "
                    "both at exit 0. So if a heated body does not expand, "
                    "THERMOMAT is the wrong place to look: check COUPVARIABLE "
                    "and INITTEMP, each of which does move the answer. "
                    "(Verified by execution 2026-08-09 on a development build, "
                    "and on 4C 2026.2.0 on 2026-09-30 with its own copies of "
                    "the two decks: identical result lines.) 4C 2026.3.0 removed "
                    "THERMOMAT from the grammar: a deck that sets it stops "
                    "with `Failed to match specification in section "
                    "'MATERIALS'`. Leaving it out reads on all three (measured "
                    "2026-09-28: displacement and temperature unchanged on "
                    "the development build)."
                ),
                (
                    "[Numerical] Monolithic TSI needs an ITERATIVE (Belos) "
                    "block-preconditioned solver in TSI DYNAMIC/MONOLITHIC — a "
                    "direct solver is rejected outright. Signal: two different "
                    "and easily-confused failures. Pointing "
                    "TSI DYNAMIC/MONOLITHIC/LINEAR_SOLVER at a 'SOLVER: UMFPACK' "
                    "block on tsi_lincompression_monolithic aborts inside "
                    "TSI::Monolithic::create_linear_solver with the two-word "
                    "message 'Iterative solver expected' from "
                    "src/tsi/4C_tsi_monolithic.cpp:250 — it names neither Belos "
                    "nor the solver you wrote, so grepping for either finds "
                    "nothing. Deleting LINEAR_SOLVER from that section instead "
                    "gives the explicit 'no linear solver defined for monolithic "
                    "TSI. Please set LINEAR_SOLVER in TSI DYNAMIC to a valid "
                    "number!' from the same file at :229. Both exit 1 before the "
                    "first time step. Note the Belos block also needs its "
                    "SOLVER_XML_FILE and TEKO_XML_FILE (PRECONDITIONER_XML_FILE "
                    "from 4C 2026.2.0 on) to resolve relative to the "
                    "deck, or Teuchos dies on SIGABRT (exit 134) with no 4C "
                    "message at all. For simple one-way problems use "
                    "tsi_oneway + UMFPACK and avoid all of this. "
                    "(Verified by execution 2026-08-09.)"
                ),
                (
                    "[Input] Thermal boundary conditions have their OWN sections "
                    "— DESIGN <ENTITY> THERMO DIRICH CONDITIONS and DESIGN "
                    "<ENTITY> THERMO NEUMANN CONDITIONS. Dropping 'THERMO' does "
                    "not address the thermal field, it addresses the STRUCTURE, "
                    "and the two mistakes behave completely differently. Signal: "
                    "on the Dirichlet side you get caught — renaming DESIGN VOL "
                    "THERMO DIRICH to DESIGN VOL DIRICH in the dilatation deck "
                    "aborts at exit 1 reading 1 DOFs given but 3 expected in "
                    "Volume Dirichlet boundary condition, from core/fem/src/"
                    "discretization/4C_fem_discretization_utils_dbc.cpp:292 "
                    "(the thermal field has 1 dof per node, the structure 3). "
                    "Both counts and the condition name are filled in at run "
                    "time — the line there is a format string of the shape "
                    "<given> DOFs given but <expected> expected in <condition "
                    "name> — so the greppable pieces are the 14-character run "
                    "DOFs given but, one hit in the tree, and the condition "
                    "name, which is a string of its own: 'Volume Dirichlet "
                    "boundary condition' in core/legacy_enum_definitions/"
                    "4C_legacy_enum_definitions_conditions.cpp:26. On "
                    "the Neumann side you do NOT: renaming DESIGN SURF THERMO "
                    "NEUMANN to DESIGN SURF NEUMANN in tsi_heatflux_monolithic "
                    "parses, runs to completion and applies the heat flux as a "
                    "mechanical traction — the heated end's temperature drops "
                    "from 3.82713688897868496e+03 to exactly 0.0 while its dispx "
                    "jumps from 4.26066261581664052e-01 to "
                    "7.07482993197279164e+00. No warning of any kind; only the "
                    "RESULT DESCRIPTION catches it. Result-test a temperature. "
                    "(Verified by execution 2026-08-09.)"
                ),
                (
                    "[Numerical] INITTEMP in the structural material is the "
                    "reference temperature at which thermal strain is zero, so it "
                    "must be on the same scale AND the same physical state as the "
                    "thermal field's initial condition: u = alpha * (T - INITTEMP) "
                    "* L. Getting it wrong does not fail, it rescales the whole "
                    "deformation. Signal: on the dilatation deck (THEXPANS 1e-05, "
                    "T = 294, L = 2, INITTEMP 293, pinned dispy 2e-05), setting "
                    "INITTEMP: 0 keeps the temperature field exactly right and "
                    "moves the displacement to 'dispy ... actresult= "
                    "5.88000000000000068e-03' — 294 times too large, which is "
                    "precisely alpha*(294-0)*2. The run is otherwise identical "
                    "and would exit 0 without a result test. The tell is that "
                    "displacement is wrong by a clean ratio (T_actual - "
                    "INITTEMP_wrong)/(T_actual - INITTEMP_right); a factor like "
                    "294 or 273.15 in your answer means INITTEMP or a Kelvin/"
                    "Celsius mix, not a bad BC. (Verified by execution "
                    "2026-08-09.)"
                ),
                (
                    "[Output] THERM_HEATFLUX and THERM_TEMPGRAD live in the IO "
                    "section and both DEFAULT TO \"NO\", so heat flux and "
                    "temperature gradient are simply not written unless you ask. "
                    "Legal values are Current, Initial, NO, No, None, no "
                    "(`4C -p`). Signal: omitting them is silent — the dilatation "
                    "deck still exits 0 — and the observable is in the OUTPUT, "
                    "not the log: with them set the <prefix>.control file lists "
                    "the result fields 'gauss_initial_heatfluxes_xyz' and "
                    "'gauss_initial_tempgrad_xyz', and without them the control "
                    "file contains NO gauss_* field at all while the thermo "
                    "result file shrinks from 391 kB to 245 kB. Grep the .control "
                    "file for 'gauss_' before you go looking for a "
                    "post-processing bug. Putting the two keys under THERMAL "
                    "DYNAMIC instead of IO, or writing an out-of-enum value such "
                    "as \"Yes\", is caught at parse with 'Could not match this "
                    "input' from core/io/src/4C_io_input_spec_builders.cpp:633 "
                    "and the offending block echoed, exit 1. "
                    "(Verified by execution 2026-08-09.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "heated_bar_3d",
                    "description": (
                        "A bar heated from one end with the other end held at "
                        "reference temperature.  The temperature gradient "
                        "causes axial elongation.  Tests one-way and two-way "
                        "thermo-mechanical coupling.  Uses monolithic TSI "
                        "with MAT_Struct_ThermoStVenantK + MAT_Fourier."
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
                    "3-D monolithic TSI: heated bar with thermal expansion.  "
                    "SOLIDSCATRA HEX8 elements, MAT_Struct_ThermoStVenantK "
                    "material, MAT_Fourier thermal material, monolithic "
                    "Belos/Teko solver."
                ),
            },
        ]

    # NOTE: 'oneway_3d' and 'plane_strain_2d' are NOT reachable through THIS
    # class — get_template('plane_strain_2d') and get_template('oneway_3d')
    # both raise ValueError('Unknown variant ...  Available: monolithic_3d')
    # (checked 2026-08-09).  They ARE reachable through the backend, which
    # tries the inline-mesh generators before falling through to this class:
    # generate_input('tsi', 'plane_strain_2d', {}) and
    # generate_input('tsi', 'oneway_3d', {}) both return a deck.  Quote the
    # backend call, never get_template, when pointing a user at either.
    # Both recipes are documented in the pitfalls above and derive from
    # 'monolithic_3d':
    #   oneway_3d      - TSI DYNAMIC: COUPALGO tsi_oneway, plus
    #                    TSI DYNAMIC/PARTITIONED: COUPVARIABLE: Temperature
    #                    (the default Displacement silently gives zero
    #                    displacement).
    #   plane_strain_2d - one-element-thick SOLIDSCATRA HEX8 slab, u_z
    #                    fixed on all nodes, temperature imposed volume-wide
    #                    via DESIGN VOL THERMO DIRICH + a symbolic FUNCT.
    # They are not listed until a template exists and has been verified by
    # execution; advertising a variant that raises is worse than omitting it.

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
            # 3-D Monolithic Thermo-Structure Interaction
            #
            # A body is subjected to a thermal load (heat flux or prescribed
            # temperature).  The resulting temperature field drives thermal
            # expansion in the structural field.
            #
            # Elements: SOLIDSCATRA HEX8 (carries both displacement and
            #           thermal DOFs via mesh cloning)
            #
            # Mesh: requires an exodus file with:
            #   element_block 1 = structure (HEX8)
            #   node_set 1 = fixed face (structural Dirichlet + thermal Dirichlet)
            #   node_set 2 = heated face (thermal Neumann or Dirichlet)
            # ---------------------------------------------------------------
            TITLE:
              - "3-D thermo-structure interaction -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Thermo_Structure_Interaction"
            IO:
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
              TIMESTEP: <structure_timestep>
              MAXTIME: <structure_max_time>
              LINEAR_SOLVER: 1

            # == Thermal =======================================================
            THERMAL DYNAMIC:
              DYNAMICTYPE: Statics
              TIMESTEP: <thermal_timestep>
              NUMSTEP: <thermal_num_steps>
              LINEAR_SOLVER: 1
            THERMAL DYNAMIC/RUNTIME VTK OUTPUT:
              OUTPUT_THERMO: true
              TEMPERATURE: true
              TEMPERATURE_RATE: true

            # == TSI coupling ==================================================
            TSI DYNAMIC:
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              TIMESTEP: <timestep>
              ITEMAX: <max_coupling_iterations>
              RESULTSEVERY: <results_output_interval>
            TSI DYNAMIC/MONOLITHIC:
              NORM_RESF: "Rel"
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
              # Fourier heat conduction material (for thermal field)
              - MAT: 2
                MAT_Fourier:
                  CAPA: <volumetric_heat_capacity>
                  CONDUCT:
                    constant: [<thermal_conductivity>]

            # Clone structure mesh -> thermo mesh
            CLONING MATERIAL MAP:
              - SRC_FIELD: "structure"
                SRC_MAT: 1
                TAR_FIELD: "thermo"
                TAR_MAT: 2

            # == Boundary Conditions ===========================================

            # Structural: fixed face
            DESIGN SURF DIRICH CONDITIONS:
              - E: <fixed_face_id>
                NUMDOF: 3
                ONOFF: [1, 1, 1]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]

            # Thermal: prescribed temperature on fixed face
            DESIGN SURF THERMO DIRICH CONDITIONS:
              - E: <cold_face_id>
                NUMDOF: 1
                ONOFF: [1]
                VAL: [<cold_face_temperature>]
                FUNCT: [0]

            # Thermal: heat flux on heated face
            DESIGN SURF THERMO NEUMANN CONDITIONS:
              - E: <heated_face_id>
                NUMDOF: 6
                ONOFF: [1, 0, 0, 0, 0, 0]
                VAL: [<heat_flux_value>, 0, 0, 0, 0, 0]
                FUNCT: [<heat_flux_time_function>, 0, 0, 0, 0, 0]

            # Time-dependent heat flux ramp
            FUNCT1:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<heat_flux_ramp_expression>"

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
              - THERMAL:
                  DIS: "thermo"
                  NODE: <result_node_id>
                  QUANTITY: "temp"
                  VALUE: <expected_temperature>
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

        # Check thermal expansion coefficient
        thexpans = params.get("THEXPANS")
        if thexpans is not None:
            try:
                alpha = float(thexpans)
                if alpha <= 0:
                    issues.append(
                        f"THEXPANS (thermal expansion coeff) must be > 0, "
                        f"got {alpha}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"THEXPANS must be a positive number, got {thexpans!r}."
                )

        # Check Young's modulus
        young = params.get("YOUNG")
        if young is not None:
            vals = young if isinstance(young, list) else [young]
            for v in vals:
                try:
                    e = float(v)
                    if e <= 0:
                        issues.append(
                            f"YOUNG must be > 0, got {e}."
                        )
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
                    issues.append(
                        f"NUE must be in (0, 0.5), got {nu}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"NUE must be a number in (0, 0.5), got {nue!r}."
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

        # Check conductivity
        conduct = params.get("CONDUCT")
        if conduct is not None:
            if isinstance(conduct, dict):
                vals = conduct.get("constant", [])
                if isinstance(vals, list):
                    for v in vals:
                        try:
                            if float(v) <= 0:
                                issues.append(
                                    f"CONDUCT values must be > 0, got {v}."
                                )
                        except (TypeError, ValueError):
                            issues.append(
                                f"CONDUCT values must be positive numbers, "
                                f"got {v!r}."
                            )

        # Check CLONING MATERIAL MAP presence
        has_cloning = params.get("has_cloning_material_map")
        if has_cloning is not None and not has_cloning:
            issues.append(
                "CLONING MATERIAL MAP is required for TSI.  It maps the "
                "structural material to the thermal material."
            )

        # Check element type. A 2-D cell is refused whatever the element
        # name: 4C 2026.3.0 accepts SOLIDSCATRA QUAD4 as input, and the run
        # still stops.
        elem_type = params.get("element_type", "")
        et = str(elem_type).upper()
        if elem_type and ("WALL" in et or "QUAD" in et or "TRI" in et):
            issues.append(
                f"TSI cannot use 2D elements ({elem_type}): 4C runs no 2D "
                f"TSI.  Measured on a 4C before 2026.2.0, on 2026.2.0 and on "
                f"2026.3.0, a 2D deck stops with 'Unsupported solid element "
                f"type!' (a 4C before 2026.2.0 refuses a QUAD4 SOLIDSCATRA or "
                f"SOLID even earlier, at the cell type).  For 2D plane-strain thermo-mechanics build a "
                f"pseudo-2D deck from the 'monolithic_3d' template: a "
                f"one-element-thick SOLIDSCATRA HEX8 slab with u_z fixed "
                f"everywhere."
            )
        elif elem_type and "SOLIDSCATRA" not in et:
            issues.append(
                f"TSI requires SOLIDSCATRA elements (not {elem_type}).  "
                f"Plain SOLID elements do not carry the thermal DOF."
            )

        return issues
