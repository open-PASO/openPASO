"""The FEBio .feb grammar — ONE copy, served by every path that needs it.

WHY THIS EXISTS, and why it is not a violation of Option B. FEBio is a BINARY
that consumes an XML deck, so the deck is this backend's run interface in
exactly the way exports.json is the coupling driver's: it cannot be guessed, and
no solve happens without it. openPASO elides "the solve" by design — correct for a
Python library, and for a file-driven code it leaves the agent unable to start.

MEASURED on the payload a single-code FEBio agent receives (21,787 characters
for `heat`): no `<febio_spec`, no `<MeshDomains`, no `<Boundary`, no `<node id=`
and no `fix=`. The corpus is rich on material models and pitfalls and silent on
the document that carries them. The same shape of gap was measured for 4C, where
it cost all three openPASO-arm runs of coupled cell C2 their whole attempt.

WHAT IS SERVED IS THE GRAMMAR, NOT A SOLVE. Every number below is an arbitrary
placeholder; the agent still chooses its own mesh, material parameters, boundary
data and body force, and still interprets its own results. The skeleton was
taken from a deck assembled by this repo's own fixture library and RUN: exit 0,
"N O R M A L   T E R M I N A T I O N", 54 equations.

THE TRAPS BELOW WERE EACH MEASURED BY EXECUTION on this FEBio build (4.12):
they are the same body-force findings already verified in the FEBio backend's
own trap text, restated here because an agent that cannot write the document
never reaches them.
"""

FEBIO_DECK_GRAMMAR = """\
THE FEBio DECK GRAMMAR — you cannot guess it, and FEBio is a BINARY, so the
deck is this backend's run interface in the way exports.json is the driver's.
Every NUMBER below is an ARBITRARY PLACEHOLDER; the sections and their order
are what is documented. A deck with these sections runs.

    <?xml version="1.0" encoding="ISO-8859-1"?>
    <febio_spec version="4.0">
    <Module type="solid"/>
      <Control>
        <analysis>STATIC</analysis>
        <time_steps>2</time_steps>
        <step_size>0.5</step_size>
        <solver type="solid">
          <symmetric_stiffness>symmetric</symmetric_stiffness>
        </solver>
      </Control>
      <Material>
        <material id="1" name="Material1" type="isotropic elastic">
          <density>1.0</density><E>1000.0</E><v>0.25</v>
        </material>
      </Material>
      <Mesh>
        <Nodes name="AllNodes">
          <node id="1">0,0,0</node>
        </Nodes>
        <Elements type="hex8" name="Part1">
          <elem id="1">1,2,3,4,5,6,7,8</elem>
        </Elements>
        <NodeSet name="bottom">1</NodeSet>
      </Mesh>
      <MeshDomains>
        <SolidDomain name="Part1" mat="Material1"/>
      </MeshDomains>
      <Boundary>
        <bc name="fix" type="zero displacement" node_set="bottom">
          <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>
        </bc>
      </Boundary>
    </febio_spec>

  * `<MeshDomains>` IS NOT OPTIONAL. It binds each element block to a material
    by NAME; without it the elements have no constitutive law and FEBio stops.
  * A NODE SET IS REFERENCED BY NAME from `<bc node_set="...">`, and the set
    must be declared inside `<Mesh>`.
  * PLANE STRAIN IN A 3-D CODE is a slab one element thick with the
    out-of-plane displacement fixed on BOTH z faces — a `zero displacement` bc
    with `<z_dof>1</z_dof>` on each. Read the task's own prescription for the
    through-thickness element count and follow it rather than refining there.
    Both z faces hold EVERY node of such a slab: u_z held on the boundary
    nodes only leaves the rest free in z, the slab is then neither plane
    strain nor plane stress, and FEBio says nothing (measured).

  A POSITION-DEPENDENT BODY FORCE, measured on this build (4.12). Two forms
  run and give the same field (a constant load gives another):
        <Loads>
          <body_load type="body force">
            <force type="math">-1*X^2,0,0</force>
          </body_load>
        </Loads>
        <Loads>
          <body_load type="non-const"><x>-1*X^2</x><y>0</y><z>0</z></body_load>
        </Loads>
  * `<body_load type="const">` with a `type="math"` component is REFUSED
    here: `tag "x" ... : invalid attribute "type"`.
  * The body force is PER UNIT MASS: FEBio multiplies it by the material's
    density (measured: density 2 doubles the displacement). A load per unit
    volume is divided by the density, or the density is 1.
  * It enters with the OPPOSITE SIGN: `<force>0.3,0,0</force>` moves the body
    toward -x, where a `<nodal_load>` of +0.3 in x moves it toward +x
    (measured, every form above). The b of -div(sigma) = b is written -b/density.
  * OMITTING `type="math"` on `<force>` IS USUALLY SILENT: the numeric prefix
    is taken and the rest of the expression is discarded, so the run
    succeeds with a constant load.
  * `**` IS A PARSE ERROR, not a silent one: `-1*X**2` gives
        Token expected (position 6)
    while `-1*X^2` is accepted. Rewrite every term of a Python expression.
  * A `<body_load>` placed in `<LoadData>` is a hard `unrecognized tag`
    failure; deleting the tag removes the load and the run then succeeds
    with f = 0, which is worse.

  PER-NODE VALUES, measured on this build (4.12):
  * A per-node table is `<NodeData name="px" node_set="right"
    data_type="scalar">` (or `data_type="vec3"`) inside `<MeshData>`, one
    `<node lid="i">value</node>` per node, `lid` counting 1, 2, ... in the
    node set's own order. `<MeshData>` is a section of its own after
    `<MeshDomains>`; inside `<Mesh>` it is an `unrecognized tag`.
  * A prescribed displacement names ONE dof: `<bc type="prescribed
    displacement" node_set="right"><dof>x</dof><value lc="1">0.01</value>
    <relative>0</relative></bc>`, or `<value lc="1" type="map">px</value>`
    for a per-node table. `<x_dof>` belongs to `zero displacement` only.
  * A per-node force: `<nodal_load type="nodal_force" node_set="right">
    <value lc="1" type="map">fm</value></nodal_load>` inside `<Loads>`, with
    `fm` a vec3 table.
  * `lc="1"` names a `<load_controller id="1">` that a `<LoadData>` section
    defines; with none FEBio stops with `Invalid load curve ID`. A `<value>`
    with no lc is applied in full at every time (measured).
  * A bc or load reads its table BY POSITION in its own node set, so the
    table is written on that same set. A table with fewer entries than its
    set is accepted without a word: the nodes with no entry take 0.
  * LISTS ARE COMMA-SEPARATED. Space-separated element connectivity
    segfaults with no message (exit -11); space-separated node coordinates
    are a `syntax error`; a space-separated NodeSet runs and keeps only the
    first number between two commas, so the set silently loses nodes.
  * `febio4 -i deck.feb` runs a deck; it takes no `--version`, `-h` or `-c`.

  READ THE LOG, NOT THE EXIT CODE. FEBio prints
      N O R M A L   T E R M I N A T I O N
  letter-spaced on success and
      E R R O R   T E R M I N A T I O N
  on failure, and reports `Nr of equations ......... : <n>`, which is the
  number to cross-check against your mesh. Grepping for the contiguous string
  "normal termination" never matches. A node log line is the node id, then the
  values in the order `data=` names them; x, y, z there are the CURRENT
  positions, reference plus displacement (measured).
  `No force acting on the system.` is a WARNING, not a lost load: with small
  loads FEBio prints it after the first iteration and keeps the solved field
  (measured). Read the node log's values.
"""
