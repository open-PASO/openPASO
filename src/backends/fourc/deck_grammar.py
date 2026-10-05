"""The 4C deck grammar — ONE copy, served by every path that needs it.

WHY THIS MODULE EXISTS. 4C is a BINARY that consumes a YAML deck, so the deck is
this backend's run interface in exactly the way exports.json is the coupling
driver's: it cannot be guessed, and no solve happens without it. openPASO elides
"the solve" by design, which is right for a Python library and wrong here.

MEASURED. The coupling payload for solver='fourc' contained no PROBLEM TYPE, no
MATERIALS, no SCALAR TRANSPORT DYNAMIC, no DESIGN LINE DIRICH, no NODE COORDS and
no NUMDOF; three recorded runs of one coupled problem all died on the 4C
side, and one of them named the cause itself ("4C scalar transport module
requires specific topology definitions"). Single-code 4C runs were worse off
still: they never call knowledge(topic='coupling'), so they received none of it.

AND IT MUST NOT BECOME FOUR COPIES. The interface-probe text taught that lesson
already: it existed in four places, two of them dead, and a fix applied to the
wrong one looked correct for hours. So the grammar lives here once and both
serving paths import it.

WHAT THIS IS NOT. It is a GRAMMAR, not a solve: every number is an arbitrary
placeholder, and the agent still derives its own mesh, materials, boundary data
and source term. Verified by dogfooding — a deck written from this text alone
runs one side of a coupled problem to completion
(tests/fixtures/fourc_running_deck/).
"""

FOURC_DECK_GRAMMAR = """\
THE 4C DECK GRAMMAR — you cannot guess it, and 4C is a BINARY, so the deck is
this backend's run interface in exactly the way exports.json is the driver's.
Every NUMBER below is an ARBITRARY PLACEHOLDER; the sections and their syntax
are what is documented. A deck with these sections runs; one missing any of
them aborts during input parsing.

  TITLE:
    - "anything"
  PROBLEM SIZE:
    ELEMENTS: 4
    NODES: 9
  PROBLEM TYPE:
    PROBLEMTYPE: "Scalar_Transport"
  SCALAR TRANSPORT DYNAMIC:
    TIMEINTEGR: "Stationary"
    SOLVERTYPE: "linear_full"
    NUMSTEP: 1
    TIMESTEP: 1.0
    MAXTIME: 1.0
    LINEAR_SOLVER: 1
  SOLVER 1:
    SOLVER: "UMFPACK"
  MATERIALS:
    - MAT: 1
      MAT_scatra:
        DIFFUSIVITY: 1.0          # <- YOUR k
  FUNCT1:
    - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "x^2*y"     # <- YOUR source/BC
  DESIGN LINE DIRICH CONDITIONS:
    - E: 1
      NUMDOF: 1
      ONOFF: [1]
      VAL: [0.0]
      FUNCT: [0]
  NODE COORDS:
    - "NODE 1 COORD 0.0 0.0 0.0"
  TRANSPORT ELEMENTS:
    - "1 TRANSP QUAD4 1 2 5 4 MAT 1 TYPE Std"
  DLINE-NODE TOPOLOGY:
    - "NODE 1 DLINE 1"

THE VOLUME SOURCE TERM f IS A "SURF" NEUMANN CONDITION IN 2-D. There is no
body-force section: 4C calls a 2-D domain a SURFACE, so `-div(k grad u) = f`
gets its f from

  DESIGN SURF TRANSPORT NEUMANN CONDITIONS:
    - E: 1
      NUMDOF: 1
      ONOFF: [1]
      VAL: [1.0]        # scale; the shape comes from FUNCT
      FUNCT: [1]        # -> FUNCT1's SYMBOLIC_FUNCTION_OF_SPACE_TIME
  DSURF-NODE TOPOLOGY:
    - "NODE 1 DSURFACE 1"      # every node of the subdomain

In 3-D the same role is played by `DESIGN VOL TRANSPORT NEUMANN CONDITIONS`
with `DVOL-NODE TOPOLOGY`. Using the VOL form on a 2-D problem is trap (d)
below. Omit this section and you solve f = 0 -- the run succeeds and the
answer is wrong, which is the worst failure mode available.

PER-NODE DIRICHLET DATA IS EXACT, AND NEEDS NO FITTED FUNCTION.
A Dirichlet-side participant receives a DISCRETE interface profile, one value per
interface node. `DESIGN LINE DIRICH CONDITIONS` takes ONE scalar VAL (times an
optional FUNCT), so it cannot carry that profile -- but one POINT condition PER
NODE can, and it is exact:

  DESIGN POINT DIRICH CONDITIONS:
    - E: 1
      NUMDOF: 1
      ONOFF: [1]
      VAL: [0.00016283346722727]        # this node's imported value
      FUNCT: [0]
    - E: 2
      NUMDOF: 1
      ONOFF: [1]
      VAL: [0.00030035999458698]
      FUNCT: [0]
  DNODE-NODE TOPOLOGY:
    - "NODE 17 DNODE 1"
    - "NODE 34 DNODE 2"

One `E` id per interface node, one topology line mapping that node to it. The
values above are verbatim from a real deck that ran to completion.

DO NOT least-squares-fit the profile into a SYMBOLIC_FUNCTION_OF_SPACE_TIME
unless there is no alternative. A fit converges to a slightly DIFFERENT
boundary-value problem, so the refinement study measures the fit rather than the
method and the error does not fall at the expected rate. Measured: one agent
concluded "4C cannot impose per-node Dirichlet values" and wrote a
could-not-finish report, while another used point conditions and produced a
complete three-level study from the same binary.

THE NEUMANN SIDE NEEDS TWO DECK LINES THE DIRICHLET SIDE DOES NOT, and one of
them fails silently. It APPLIES the partner's flux profile, and it has to ASK
for the consistent boundary flux or 4C writes none.

Applying a sampled flux profile needs no fitted function either: pre-integrate
it into nodal loads, one POINT condition per interface node. For an interior
interface node i with spacing h, the consistent load is Simpson's

      VAL_i = h/6 * (g_{i-1} + 4*g_i + g_{i+1})       g = the imported flux

  DESIGN POINT NEUMANN CONDITIONS:
    - E: 1
      NUMDOF: 1
      ONOFF: [1]
      VAL: [0.05]        # <- THIS node's pre-integrated load, from the formula
      FUNCT: [0]         # no function: the number is already the load
  DNODE-NODE TOPOLOGY:
    - "NODE 17 DNODE 1"

READING THE FLUX BACK, which the coupling needs at every iteration, is two
entries that must BOTH be present:

  SCALAR TRANSPORT DYNAMIC:
    CALCFLUX_BOUNDARY: "diffusive"
  SCATRA FLUX CALC LINE CONDITIONS:
    - E: <id>            # the DLINE whose nodes lie ON the interface (SURF in 3-D);
                         # any other line exports a flux of the wrong boundary

Measured on this install, one 2x2 scatra deck run three times, differing only in
these lines:

  both present         finished normally; the VTU carries the point array
                       `flux_boundary_phi_1` and
                       `<prefix>.boundaryflux_ScaTraFluxCalc_0scatra.txt` holds
                       the area, the integral and the mean normal flux
  CALCFLUX_BOUNDARY    finished normally and wrote NO flux array at all -- the
  omitted              run looks like a success and the side has nothing to
                       export
  the condition        stopped in the input reader, with this line:
  section omitted
    Flux output requested without corresponding boundary condition specification!

The loud one costs a minute; the silent one is why a side can run 4C to
`processor 0 finished normally` and still export nothing. `flux_boundary_phi_1`
is the diffusive flux VECTOR -k*grad(phi) at the boundary nodes, not q.n: dot it
with YOUR outward normal. On a Neumann-loaded line it is the consistent-residual
echo of the load you applied, so use it as the exported flux only on a side
whose interface is DIRICHLET.

WHICH PROBLEM TYPE YOU PICK DECIDES WHICH VTU FILES YOU GET: see the two
problem-type notes at the end of this part. First, EIGHT MEASURED WAYS A DECK
DIES, each with the message 4C prints:
(a) THE LEGACY BLOCKS ARE YAML SEQUENCES. `NODE COORDS`, `TRANSPORT
    ELEMENTS` and `D*-NODE TOPOLOGY` entries each need `- ` and quotes.
    Written bare, YAML reads them as mapping keys and 4C dies with
      ERROR: could not find ':' colon after key
    BEFORE its own banner appears.
(b) `**` IS NOT EXPONENTIATION in SYMBOLIC_FUNCTION_OF_SPACE_TIME. Use `^`.
    And the constant is lowercase `pi`: `PI` aborts the run with "Missing
    variables PI to evaluate expression" (measured on a trial deck).
    The task states its source term in Python notation; rewrite every term.
(c) ONOFF / VAL / FUNCT must each have EXACTLY `NUMDOF` entries, or
      [!] Candidate parameter 'VAL' has incorrect size
(d) A condition's dimension must not exceed the problem's: a
    `DESIGN VOL ...` block on a 2-D problem gives
      Dimension of condition is larger than the problem dimension.
(e) NO SECTION OR KEY THE GRAMMAR DOES NOT LIST. 4C matches every block
    against its specification and aborts on the first unknown key with
      Could not match this input
    followed by the offending block (for example a misspelled key). 4C's
    defaults need no IO section at all; keep exactly the sections above and
    nothing you did not see documented.
(f) EVERY `E:` ID A CONDITION NAMES MUST EXIST IN THE MATCHING TOPOLOGY.
    Ids are 1-based and contiguous. A `DESIGN LINE ... CONDITIONS` entry
    with `E: 4` while `DLINE-NODE TOPOLOGY` defines only DLINE 1..3 aborts
    with
      DLine 4 not in range [0:4[  DLine condition on non existent DLine
    Count the distinct DLINE (DSURFACE, DNODE) ids in your topology and use
    only those; give every interface, outer-boundary and surface set its
    own id and its own topology lines.
(g) NEVER TYPE NODE COORDS, ELEMENTS OR TOPOLOGY BY HAND. Generate them in
    a short loop (a Python script that writes the deck for the level in
    ./config.json, which is what the participant contract does) and let the
    loop count them. Measured on trial decks typed out in full: a list one
    node short aborts with
      Element 79 cannot find node 99
    and a long typed deck was cut off mid-string and failed to parse with
      reached end of file looking for closing quote
    A generator is thirty lines; the deck it writes can be any size.
(h) EVERY TOPOLOGY SECTION APPEARS ONCE. All point sets share the one
    `DNODE-NODE TOPOLOGY` block, all line sets the one `DLINE-NODE TOPOLOGY`
    block, and so on -- one section per kind, holding every E id. A deck that
    writes the block again for its second condition aborts on read with
      Section 'DNODE-NODE TOPOLOGY' is defined more than once.
    (measured on a trial deck that gave its Dirichlet points and its Neumann
    points separate topology blocks).

AND READ THE LOG FROM THE TOP. 4C buffers stdout and MPI_Abort kills it
before the flush, so `| tail` shows only MPI boilerplate. Measured on one
failing deck: 8 lines without line buffering, 43 with it. Run
    stdbuf -oL <4C binary> deck.4C.yaml out > run.log 2>&1 ; head -40 run.log
`Invalid MIT-MAGIC-COOKIE-1 key` is an X11 warning that appears on
SUCCESSFUL runs too — `4C -p` prints it and then dumps the whole grammar.
It never explains a failure.
  * WHICH ELEMENT OWNS 2-D STRUCTURAL CELLS DEPENDS ON THE BUILD, and the two
    spellings share no keywords. `4C -p` settles it: WALL is listed under
    legacy_element_specs in the development builds before the 2026.2.0 release
    and absent from 4C 2026.2.0 and 2026.3.0.
    Before 4C 2026.2.0: `SOLID` is the 3-D continuum element and the 2-D one is
    `WALL`; `SOLID QUAD4` fails with

        Element 'SOLID' does not seem to know cell type 'quad4'.

    Measured from that build's own grammar (`4C -p`):
        SOLID        HEX8 HEX18 HEX20 HEX27 TET4 TET10 WEDGE6 PYRAMID5 NURBS27
        WALL         QUAD4 QUAD8 QUAD9 TRI3 TRI6 NURBS4 NURBS9
        THERMO       QUAD4 QUAD8 QUAD9 TRI3 TRI6 + the 3-D types
    A WALL element line carries its own extra keywords, and this one RUNS
    (exit 0, "processor 0 finished normally", num_dof 8 on a single element):

        STRUCTURE ELEMENTS:
          - "1 WALL QUAD4 1 2 3 4 MAT 1 KINEM linear EAS none THICK 1.0
             STRESS_STRAIN plane_strain GP 2 2"

    4C 2026.2.0 and later: there is no WALL; SOLID also takes QUAD4, QUAD8,
    QUAD9, TRI3 and TRI6, with THICKNESS and PLANE_ASSUMPTION (EAS none and GP
    2 2 are its defaults). Measured on 4C 2026.2.0 and 2026.3.0 with openPASO's
    2-D templates:

        STRUCTURE ELEMENTS:
          - "1 SOLID QUAD4 1 2 3 4 MAT 1 KINEM linear THICKNESS 1.0
             PLANE_ASSUMPTION plane_strain"

  * 2-D `Thermo_Structure_Interaction` IS NOT AVAILABLE IN ANY 4C MEASURED
    (a development build with WALL, 2026.2.0, 2026.3.0), and the error does not
    say so. A 2-D TSI deck fails with

        4C_tsi_utils.cpp: Unsupported solid element type!

    On a build with WALL that comes even after the element name is corrected
    to WALL. The reason is in its source:
    `TSI::Utils::ThermoStructureCloneStrategy::set_element_data` accepts ONLY
    a `SolidScatra` element and throws for anything else, and before 2026.2.0
    SOLIDSCATRA's cell types are HEX8, HEX27, TET4, TET10 and NURBS27 — every
    one of them three-dimensional. 4C 2026.2.0 gives SOLIDSCATRA and SOLID
    the 2-D cells QUAD4, QUAD9, TRI3 and TRI6 (SOLID also QUAD8), with
    THICKNESS and PLANE_ASSUMPTION. A 2-D TSI deck with either element passes
    the input check there, builds the structure, and stops with the same
    message when TSI clones the thermal field from it (measured on a
    two-QUAD4 plane-strain deck; the upstream 3-D tsi_lindilatation_geolin
    deck finishes normally on the same binary). So the clone step does not
    succeed in 2-D on any of them, whatever else the deck says.

    For a two-dimensional thermoelastic subdomain, do NOT keep repairing the
    TSI deck, and do NOT conclude the problem cannot be solved (one run gave
    up at this sentence with 26 minutes left). THE MEASURED ROUTE: a
    three-dimensional slab one element thick -- SOLIDSCATRA HEX8, u_z fixed
    on every node, plane strain by construction -- and it is served ready to
    run: `prepare_simulation(solver='fourc', physics='tsi')` hands over the
    `plane_strain_2d` variant (executed on this binary to 'processor 0
    finished normally'); put the task's thermal sources and boundary
    conditions on that slab in place of its volume-wide temperature, keep the
    thermal expansion from THEXPANS/INITTEMP (alpha = beta/(3*lambda+2*mu)
    for a task that states beta), and exchange [T, ux, uy] / [q_n, t_x, t_y]
    per interface point as the coupling contract says. The alternative of two
    separate problem types (`Structure` with WALL elements, 2-D SOLID from
    2026.2.0 on, and `Thermo` with THERMO elements, exchanging the thermal strain
    yourself) has no served
    recipe. Runs have
    lost their whole budget rewriting section names against this, because
    the message names an element type and not the dimension.

  * DESIGN ENTITY IDS START AT 1, AND A 0 IS A SEGMENTATION FAULT WITH NO
    MESSAGE. `E:` in a condition block and the `DLINE`/`DNODE`/`DSURF` number
    in the matching topology block are ONE-BASED. Writing `E: 0` with
    `NODE n DLINE 0`, which is the natural thing to do coming from Python,
    indexes past the end of the design-entity array:

        E: 0 / DLINE 0   ->  Signal: Segmentation fault (11)
                             Signal code: Address not mapped (1)     exit 139
        E: 1 / DLINE 1   ->  Read/generate conditions ... 0.0024 secs
                             processor 0 finished normally           exit 0

    Measured by running the SAME deck twice with only that digit changed. There
    is no error message, no line number and no mention of conditions: the
    process simply dies, and the crash arrives during "Read/generate
    conditions", so it reads like a problem with the condition's CONTENT. A run
    that responds by rewriting the section names, swapping
    `DESIGN LINE TRANSPORT DIRICH` for `DESIGN LINE DIRICH`, or changing the
    element TYPE will crash identically every time. Check the digit first.

  * `PROBLEMTYPE: "Thermo"` with a `THERMAL DYNAMIC` section and
        THERMAL DYNAMIC/RUNTIME VTK OUTPUT:
          OUTPUT_THERMO: true
          TEMPERATURE: true
        IO:
          VERBOSITY: "Standard"
        IO/RUNTIME VTK OUTPUT:
          OUTPUT_DATA_FORMAT: ascii
    writes <prefix>-vtk-files/thermo-<step>-<rank>.vtu -- ASCII VTU, readable
    with meshio, which is what you need to evaluate the field at probe points.
    Without the THERMAL DYNAMIC/RUNTIME VTK OUTPUT block (both keys default to
    false) Thermo writes no VTU, only the binary <prefix>.result.thermo.s1.
  * `PROBLEMTYPE: "Scalar_Transport"` writes VTU with no output section at
    all: <prefix>-vtk-files/scatra-<step>-<rank>.vtu at step 0, at every step
    that is a multiple of RESULTSEVERY or of RESTARTEVERY (default 1, so every
    step unless you raise it) and at the last step, with point arrays phi_<k>
    (plus flux_domain_phi_<k> / flux_boundary_phi_<k> when CALCFLUX_DOMAIN /
    CALCFLUX_BOUNDARY is set). There is no IO/RUNTIME VTK OUTPUT/SCATRA
    subsection (`4C -p` offers only {BEAMS,FLUID,STRUCTURE}); the scatra
    writer uses the IO/RUNTIME VTK OUTPUT settings directly (OUTPUT_DATA_FORMAT:
    ascii gives ASCII, the default is binary). SCALAR TRANSPORT DYNAMIC's
    `OUTPUTSCALARS` emits totals and means, not fields.
Either route leaves a VTU field you can probe at points: scatra-*.vtu for
Scalar_Transport, thermo-*.vtu for Thermo.


THE TSI SLAB DECK GRAMMAR — steady thermo-elasticity in ONE 4C run, plane
strain as a ONE-ELEMENT-THICK SOLIDSCATRA HEX8 slab (u_z = 0 on every node
of both layers is exact plane strain). The same rule as above: every NUMBER
is an ARBITRARY PLACEHOLDER, every coordinate and condition value is yours;
the sections and their syntax are what the input reader accepts. Measured
first-attempt traps (2026-09-11/12, twelve worker decks and two rounds): a
temperature filed under DESIGN ... DIRICH (that family carries the 3
displacement dofs; temperature goes to ... THERMO DIRICH, NUMDOF 1), the
CLONING MATERIAL MAP or COUPVARIABLE "Temperature" missing, the entry word
DSURFACE inside the section named DSURF-NODE TOPOLOGY, `**` in a FUNCT (write
`^`), and MAT_Fourier written with CONDUCTIVITY (its parameters are CAPA and
CONDUCT). Writing the deck to a .yaml, .yml or .dat file names all of these in the
same reply, before the binary runs
in one call.

  TITLE:
    - "anything"
  PROBLEM SIZE:
    DIM: 3
  IO:
    STRUCT_STRESS: "No"
    STRUCT_STRAIN: "No"
  IO/MONITOR STRUCTURE DBC:
    INTERVAL_STEPS: 1
    FILE_TYPE: yaml
    WRITE_CONDITION_INFORMATION: true
  PROBLEM TYPE:
    PROBLEMTYPE: "Thermo_Structure_Interaction"
  STRUCTURAL DYNAMIC:
    DYNAMICTYPE: "Statics"
    TIMESTEP: 1
    NUMSTEP: 1
    MAXTIME: 1
    LINEAR_SOLVER: 2
  THERMAL DYNAMIC:
    DYNAMICTYPE: Statics
    TIMESTEP: 1
    NUMSTEP: 1
    MAXTIME: 1
    LINEAR_SOLVER: 1
  TSI DYNAMIC:
    COUPALGO: "tsi_oneway"
    NUMSTEP: 1
    MAXTIME: 1
    TIMESTEP: 1
    ITEMAX: 1
  TSI DYNAMIC/PARTITIONED:
    COUPVARIABLE: "Temperature"
  IO/RUNTIME VTK OUTPUT:
    INTERVAL_STEPS: 1
    OUTPUT_DATA_FORMAT: "ascii"
  IO/RUNTIME VTK OUTPUT/STRUCTURE:
    OUTPUT_STRUCTURE: true
    DISPLACEMENT: true
  THERMAL DYNAMIC/RUNTIME VTK OUTPUT:
    OUTPUT_THERMO: true
    TEMPERATURE: true
  SOLVER 1:
    SOLVER: "UMFPACK"
    NAME: "Thermal_Solver"
  SOLVER 2:
    SOLVER: "UMFPACK"
    NAME: "Structure_Solver"
  MATERIALS:
    - MAT: 1
      MAT_Struct_ThermoStVenantK:
        YOUNGNUM: 1
        YOUNG: [1000.0]          # a LIST even for one value
        NUE: 0.25
        DENS: 1
        THEXPANS: 0.001          # alpha = beta / (3*lambda + 2*mu) for a stress -beta*T*I
        INITTEMP: 0              # (no THERMOMAT: 4C 2026.3.0 removed it; before, it changes nothing)
    - MAT: 2
      MAT_Fourier:
        CAPA: 1
        CONDUCT:
          constant: [1.0]
  CLONING MATERIAL MAP:
    - SRC_FIELD: "structure"
      SRC_MAT: 1
      TAR_FIELD: "thermo"
      TAR_MAT: 2
  FUNCT1:
    - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "x*y"         # powers as ^, never **
  FUNCT2:
    - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "x^2"
  FUNCT3:
    - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "y^2"
  DESIGN VOL NEUMANN CONDITIONS:            # mechanical body force per component: VAL * FUNCT(x,y)
    - E: 1
      NUMDOF: 3
      ONOFF: [1, 1, 0]
      VAL: [1.0, 1.0, 0.0]
      FUNCT: [1, 2, 0]
  DESIGN VOL THERMO NEUMANN CONDITIONS:     # the heat source
    - E: 1
      NUMDOF: 1
      ONOFF: [1]
      VAL: [1.0]
      FUNCT: [3]
  DESIGN VOL DIRICH CONDITIONS:             # u_z = 0 on the whole slab
    - E: 1
      NUMDOF: 3
      ONOFF: [0, 0, 1]
      VAL: [0, 0, 0]
      FUNCT: [0, 0, 0]
  DESIGN SURF DIRICH CONDITIONS:            # outer boundary, displacement (NUMDOF 3)
    - E: 1
      NUMDOF: 3
      ONOFF: [1, 1, 1]
      VAL: [0, 0, 0]
      FUNCT: [0, 0, 0]
  DESIGN SURF THERMO DIRICH CONDITIONS:     # outer boundary, temperature (NUMDOF 1)
    - E: 1
      NUMDOF: 1
      ONOFF: [1]
      VAL: [0]
      FUNCT: [0]
  DESIGN POINT DIRICH CONDITIONS:           # one entry per interface node AND layer; the reaction is monitored
    - E: 1
      NUMDOF: 3
      ONOFF: [1, 1, 1]
      VAL: [0.01, 0.02, 0.0]
      FUNCT: [0, 0, 0]
      TAG: monitor_reaction
  DESIGN POINT THERMO DIRICH CONDITIONS:    # the same node and layer, temperature
    - E: 1
      NUMDOF: 1
      ONOFF: [1]
      VAL: [0.5]
      FUNCT: [0]
  NODE COORDS:                              # layer 0 at z = 0, layer 1 at z = thickness; id(layer 1) = id + number of 2-D nodes
    - "NODE 1 COORD 0.0 0.0 0.0"
    - "NODE 2 COORD 1.0 0.0 0.0"
    - "NODE 3 COORD 1.0 1.0 0.0"
    - "NODE 4 COORD 0.0 1.0 0.0"
    - "NODE 5 COORD 0.0 0.0 0.1"
    - "NODE 6 COORD 1.0 0.0 0.1"
    - "NODE 7 COORD 1.0 1.0 0.1"
    - "NODE 8 COORD 0.0 1.0 0.1"
  STRUCTURE ELEMENTS:                       # bottom quad counter-clockwise, then the same four nodes on the top layer
    - "1 SOLIDSCATRA HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM linear TYPE Undefined"
  DNODE-NODE TOPOLOGY:                      # one DNODE per interface node and layer, ids matching the E ids above
    - "NODE 2 DNODE 1"
  DSURF-NODE TOPOLOGY:                      # the section says DSURF, the entries say DSURFACE
    - "NODE 1 DSURFACE 1"
    - "NODE 4 DSURFACE 1"
    - "NODE 5 DSURFACE 1"
    - "NODE 8 DSURFACE 1"
  DVOL-NODE TOPOLOGY:                       # every node of both layers
    - "NODE 1 DVOL 1"
    - "NODE 2 DVOL 1"
    - "NODE 3 DVOL 1"
    - "NODE 4 DVOL 1"
    - "NODE 5 DVOL 1"
    - "NODE 6 DVOL 1"
    - "NODE 7 DVOL 1"
    - "NODE 8 DVOL 1"

  Run it as `stdbuf -oL -eL <4C> deck.4C.yaml out` from INSIDE your participant
  script (couple() and the critic review take that script, never the binary); it leaves out-vtk-files/
  (structure-*.vtu with the displacement, thermo-*.vtu with the temperature)
  and one out-<id>_monitor_dbc.yaml per monitored point condition with the
  reaction force. The thermal DIRICH entries write no reaction, so a
  consistent heat flux comes from a Scalar_Transport run of the same 2-D
  mesh with CALCFLUX_BOUNDARY (the grammar above).
"""

# THE SKELETONS ARE REFERENCE TABLES, NOT INSTRUCTIONS. Split once here so the knowledge
# door can keep both deck skeletons whole when a reply meets its budget (measured
# 2026-09-12: the worker's own call -- facts 10.6k + the 4C thermo-elastic contract 29.9k
# -- left 6k of the 48k cap for this text, and the TSI skeleton at its end was cut in
# every served reply of the day).
_i = FOURC_DECK_GRAMMAR.index("WHICH PROBLEM TYPE YOU PICK DECIDES")
_j = FOURC_DECK_GRAMMAR.index("THE TSI SLAB DECK GRAMMAR")
FOURC_SCATRA_SKELETON = FOURC_DECK_GRAMMAR[:_i]          # header + the Scalar_Transport skeleton and its two section notes
FOURC_DECK_NOTES = FOURC_DECK_GRAMMAR[_i:_j]             # the measured ways a deck dies, and the problem-type notes (prose)
FOURC_TSI_SKELETON = FOURC_DECK_GRAMMAR[_j:]             # the TSI slab skeleton with its traps and run note
FOURC_DECK_SKELETONS = FOURC_SCATRA_SKELETON + FOURC_TSI_SKELETON

