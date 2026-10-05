"""FEBio linear-elasticity generators and knowledge.

FEBio Module type: 'solid'. Material: 'isotropic elastic' (small strain).
"""


def _elasticity_3d_cube(params: dict) -> str:
    """Unit cube with prescribed-displacement uniaxial compression.

    Linear isotropic elastic material; STATIC analysis; FEBio v4.0 XML.
    """
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="solid"/>
  <Control>
    <analysis>STATIC</analysis>
    <time_steps>1</time_steps>
    <step_size>1.0</step_size>
    <solver type="solid">
      <symmetric_stiffness>symmetric</symmetric_stiffness>
      <equation_scheme>staggered</equation_scheme>
    </solver>
  </Control>
  <Globals>
    <Constants>
      <T>0</T>
      <R>0</R>
      <Fc>0</Fc>
    </Constants>
  </Globals>
  <Material>
    <material id="1" name="Material1" type="isotropic elastic">
      <density>1.0</density>
      <E>{E}</E>
      <v>{nu}</v>
    </material>
  </Material>
  <Mesh>
    <Nodes name="Object1">
      <node id="1">0,0,0</node>
      <node id="2">1,0,0</node>
      <node id="3">1,1,0</node>
      <node id="4">0,1,0</node>
      <node id="5">0,0,1</node>
      <node id="6">1,0,1</node>
      <node id="7">1,1,1</node>
      <node id="8">0,1,1</node>
    </Nodes>
    <Elements type="hex8" mat="1" name="Part1">
      <elem id="1">1,2,3,4,5,6,7,8</elem>
    </Elements>
    <NodeSet name="fix_bottom">1,2,3,4</NodeSet>
    <NodeSet name="load_top">5,6,7,8</NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="Material1"/>
  </MeshDomains>
  <Boundary>
    <bc name="fix" type="zero displacement" node_set="fix_bottom">
      <x_dof>1</x_dof>
      <y_dof>1</y_dof>
      <z_dof>1</z_dof>
    </bc>
    <bc name="load" type="prescribed displacement" node_set="load_top">
      <dof>z</dof>
      <value lc="1">-0.1</value>
    </bc>
  </Boundary>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>LINEAR</interpolate>
      <extend>CONSTANT</extend>
      <points>
        <pt>0,0</pt>
        <pt>1,1</pt>
      </points>
    </load_controller>
  </LoadData>
  <Output>
    <!-- text node data beside the binary plot file: the .xplt needs FEBio's own reader,
         while a CSV can be opened, plotted and checked by anything. -->
    <logfile>
      <node_data data="ux;uy;uz" delim="," file="node_disp.csv"/>
    </logfile>
    <plotfile type="febio">
      <var type="displacement"/>
      <var type="stress"/>
    </plotfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "linear_elasticity": {
        "description": (
            "Linear elasticity with FEBio — 'isotropic elastic' material "
            "under Module type='solid'. This is the entry point for every "
            "FEBio solid deck: the section order, the material-to-domain "
            "linking and the load-controller wiring described below are "
            "the same for hyperelasticity, contact, biphasic and every "
            "other solid physics."
        ),
        "input_format": "FEBio XML (.feb), version 4.0",
        "solver": ("Quasi-Newton (BFGS by default) with a direct linear "
                   "solver (default 'skyline' on a USE_MKL=OFF build; "
                   "'pardiso' when FEBio is built with MKL)"),
        "required_section_order": (
            "Module, Control, [Globals], Material, Mesh, MeshDomains, "
            "Boundary, [Loads], [Rigid], [Contact], LoadData, Output. "
            "MeshDomains must come after Mesh and before Boundary, "
            "because node sets are only resolvable once the domains are "
            "built."
        ),
        "materials": {
            "isotropic elastic": {"E": "Young's modulus", "v": "Poisson's ratio"},
            "neo-Hookean": {"E": "Young's modulus", "v": "Poisson's ratio"},
            "Mooney-Rivlin": {"c1": "1st Mooney-Rivlin constant",
                              "c2": "2nd Mooney-Rivlin constant",
                              "k": "bulk modulus, REQUIRED — Mooney-Rivlin "
                                   "is an uncoupled material and rejects a "
                                   "deck that omits it"},
        },
        "pitfalls": [
            (
                "[Syntax] Every <material> needs a name= attribute in the "
                "4.0 schema even though it also has id=, and every "
                "<SolidDomain> in <MeshDomains> must point at that NAME, "
                "not at the numeric id. These are the first two errors a "
                "hand-written FEBio 4 deck hits. "
                "WRONG: <material id=\"1\" type=\"isotropic elastic\">"
                "<E>1000.0</E><v>0.3</v></material> with "
                "<SolidDomain name=\"Part1\" mat=\"1\"/>. "
                "RIGHT: <Material><material id=\"1\" name=\"Material1\" "
                "type=\"isotropic elastic\"><density>1.0</density>"
                "<E>1000.0</E><v>0.3</v></material></Material> with "
                "<MeshDomains><SolidDomain name=\"Part1\" "
                "mat=\"Material1\"/></MeshDomains>, where name=\"Part1\" "
                "matches the name= on the <Elements> block. "
                "Signal: tag \"material\" (line N) : `missing attribute "
                "\"name\"` for the first, tag \"SolidDomain\" (line N) : `"
                "invalid value for attribute \"mat\"` for the second, "
                "both with `Reading file ...FAILED!` and exit 1. "
                "The mat= attribute on the <Elements> tag itself is a "
                "3.x leftover and is silently IGNORED in 4.0 — decks "
                "with no mat= at all, with mat=\"1\", with mat=\"M1\" "
                "and with mat=\"NONSENSE\" on <Elements> all run and "
                "write bit-identical node positions, so it is not a "
                "substitute for <MeshDomains>. Which of the templates "
                "shipped in this backend currently execute is tracked in "
                "_general.deck_authoring. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Syntax] <MeshDomains> is REQUIRED and must sit between "
                "<Mesh> and <Boundary>. Omitting it does NOT report a "
                "missing-domain error — node sets are only resolvable "
                "once the domains are built, so the first <bc> fails "
                "instead and sends you hunting in the wrong section. "
                "WRONG: <Mesh>...</Mesh> followed directly by "
                "<Boundary><bc name=\"fix\" type=\"zero displacement\" "
                "node_set=\"fix_bottom\">...</bc></Boundary>. "
                "RIGHT: <Mesh>...</Mesh><MeshDomains>"
                "<SolidDomain name=\"Part1\" mat=\"Material1\"/>"
                "</MeshDomains><Boundary>...</Boundary>. "
                "Signal: tag \"bc\" (line N) : `invalid value for "
                "attribute \"node_set\"` with `Reading file ...FAILED!` "
                "and exit 1 — the line number points at the <bc>, not at "
                "the missing section. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Syntax] <Control> is REQUIRED and must contain a nested "
                "<solver type=\"...\"> whose type string is the module "
                "name — 'solid' for Module type='solid'. "
                "WRONG: <Control><analysis>STATIC</analysis>"
                "<time_steps>1</time_steps><step_size>1.0</step_size>"
                "</Control>. "
                "RIGHT: <Control><analysis>STATIC</analysis>"
                "<time_steps>1</time_steps><step_size>1.0</step_size>"
                "<solver type=\"solid\">"
                "<symmetric_stiffness>symmetric</symmetric_stiffness>"
                "</solver></Control>. "
                "Signal: `Component \"\" needs to have property \"solver\" "
                "defined (line N)` at parse, `Reading file ...FAILED!`, "
                "exit 1 — the empty quotes are not a bug, the FEAnalysis "
                "object has no name. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Input] Dropping <Control> ENTIRELY is the single most "
                "dangerous FEBio outcome, because it looks like success: "
                "the deck READS successfully, prints no error box at all, "
                "and still writes a non-empty .xplt plus every requested "
                "logfile CSV. Nothing is solved. Never accept a FEBio run "
                "on the existence of output files, and never size-check "
                "the .xplt against a fixed byte count — a mesh-only plot "
                "file is only 15-30% smaller than the solved one and the "
                "absolute size is deck-specific. "
                "WRONG: accepting a run because runs/<tag>/pos.txt "
                "exists and is non-empty. "
                "RIGHT: require BOTH the literal banner and a non-zero "
                "completed-step count, e.g. assert "
                "\"N O R M A L   T E R M I N A T I O N\" in log and "
                "int(re.search(r\"Number of time steps completed"
                "\\s*\\.*\\s*:\\s*(\\d+)\", log).group(1)) > 0. "
                "Signal: the CSVs contain ONLY a `*Step  = 0` / "
                "`*Time  = 0` block of zeros, the summary reads "
                "`Number of time steps completed` with a value of 0, and "
                "the banner is `E R R O R   T E R M I N A T I O N` with "
                "exit 1. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Syntax] Any BC value carrying an lc=N attribute needs a "
                "matching <load_controller id=\"N\"> in <LoadData>, and "
                "the failure arrives AFTER the reader has already said "
                "SUCCESS — a wrapper that only checks the reader line "
                "believes the deck is fine. Nothing is solved. "
                "WRONG: <bc name=\"load\" type=\"prescribed displacement\" "
                "node_set=\"load_top\"><dof>z</dof>"
                "<value lc=\"1\">-0.1</value></bc> with no <LoadData>. "
                "RIGHT: add <LoadData><load_controller id=\"1\" "
                "type=\"loadcurve\"><interpolate>LINEAR</interpolate>"
                "<extend>CONSTANT</extend><points><pt>0,0</pt>"
                "<pt>1,1</pt></points></load_controller></LoadData>. "
                "Signal: `Reading file ...SUCCESS!` followed by "
                "`Invalid load curve ID` and `Model initialization "
                "failed`, exit 1. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Syntax] Poisson's ratio is spelled with a lowercase "
                "letter v, never nu. The nu name is the convention in "
                "FEniCSx / deal.II / NGSolve; FEBio's parameter list for "
                "'isotropic elastic' and 'neo-Hookean' is exactly <E> and "
                "<v>. "
                "WRONG: <material id=\"1\" name=\"Material1\" "
                "type=\"isotropic elastic\"><E>1000.0</E><nu>0.3</nu>"
                "</material>. "
                "RIGHT: <material id=\"1\" name=\"Material1\" "
                "type=\"isotropic elastic\"><density>1.0</density>"
                "<E>1000.0</E><v>0.3</v></material>. "
                "Signal: tag \"nu\" (line N) : `unrecognized tag` with "
                "`Reading file ...FAILED!` and exit 1. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Discretization] Node ids are 1-based, and a 0-based "
                "connectivity list is NOT rejected — this is the mesh "
                "conversion bug that survives a clean run. FEBio resolves "
                "each connectivity entry as (entry minus the smallest "
                "node id declared in <Nodes>) with no bounds check, so "
                "writing 1-based node ids together with 0-based element "
                "connectivity, as meshio and PyVista exports do without "
                "an explicit +1 offset, shifts every element by one node. "
                "The last node in the mesh is then referenced by nothing. "
                "WRONG: <Nodes><node id=\"1\">0,0,0</node>...</Nodes> "
                "with <Elements type=\"hex8\" name=\"Part1\">"
                "<elem id=\"1\">0,1,2,3,4,5,6,7</elem></Elements>. "
                "RIGHT: <Elements type=\"hex8\" name=\"Part1\">"
                "<elem id=\"1\">1,2,3,4,5,6,7,8</elem></Elements>. "
                "Signal: only <node id=\"0\"> in the <Nodes> block is "
                "caught, with tag \"node\" (line N) : `invalid value for "
                "attribute \"id\"` and `Reading file ...FAILED!`. A "
                "0-based CONNECTIVITY list reads `...SUCCESS!` and is "
                "then caught at MESH INITIALISATION, before any step: "
                "one WARNING box reading `1 isolated vertex removed.` "
                "(that is the last node, which nothing now references), "
                "then an ERROR box reading `Negative jacobian detected "
                "during mesh initialization.`, then `Model "
                "initialization failed` and exit 1. The shift moves "
                "every element by one node in the lattice ordering, so "
                "the elements that wrap a row are twisted, which is what "
                "the negative jacobian is. Still assert "
                "min(connectivity) == min(node ids) before writing the "
                "deck, and do not rely on the reader line: the deck "
                "reads clean. THIS CORRECTS AN EARLIER VERSION OF THIS "
                "ENTRY, which said a 0-based connectivity list reaches "
                "`N O R M A L   T E R M I N A T I O N` with exit 0 and "
                "silently wrong stresses. Re-executed on structured "
                "hex8 unit cubes at four refinements with every "
                "connectivity entry shifted down by one: normal "
                "termination was reached on none of them. The warning "
                "half of the old Signal was right; the silent-wrong "
                "half was not. (Executed 2026-08-03, corrected and "
                "re-executed 2026-08-06, FEBio 4.12.0.86045466d; "
                "fixture scripts/tier2_fixtures/febio/"
                "zero_based_connectivity_rejected.)"
            ),
            (
                "[Discretization] A degenerate hex8 — the classic "
                "off-by-one that repeats a node id inside one element's "
                "connectivity — is NOT rejected, and on a real mesh it is "
                "not even warned about. Swept on 1x1x1, 2x2x2 and 4x4x4 "
                "hex8 unit cubes under identical uniaxial prescribed "
                "displacement: collapsing one node of element 1 changed "
                "that element's stress components by a double-digit "
                "percentage on all three meshes, and broke the x/y "
                "symmetry that the load case makes exact — the clean runs "
                "give sx and sy identical to every printed digit, the "
                "collapsed ones differ by tens of percent. The effect "
                "does not shrink with refinement, so it is not "
                "discretization noise. "
                "WRONG: <elem id=\"1\">1,2,3,3,5,6,7,8</elem> (node 3 "
                "repeated). "
                "RIGHT: <elem id=\"1\">1,2,3,4,5,6,7,8</elem>, and assert "
                "len(set(connectivity)) == 8 for every hex8 before "
                "writing the deck. "
                "Signal: the run completes with "
                "`N O R M A L   T E R M I N A T I O N` and exit 0. On a "
                "single-element mesh there is one WARNING box, "
                "`1 isolated vertex removed.`; on the 2x2x2 and 4x4x4 "
                "meshes the collapsed node is still used by a neighbour, "
                "so there is NO warning of any kind and nothing in the "
                "log distinguishes the broken mesh from the good one. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Syntax] Factory TYPE STRINGS are case-SENSITIVE; enum "
                "PARAMETER VALUES are case-INSENSITIVE. The type= "
                "attribute on <material>, <bc>, <Elements>, <solver>, "
                "<load_controller>, <contact> must match the registered "
                "string exactly, including hyphens and spacing. The "
                "contents of <analysis>, <symmetric_stiffness>, "
                "<interpolate> and the other enum parameters do not. "
                "WRONG: <Elements type=\"HEX8\" name=\"Part1\">, "
                "<bc name=\"fix\" type=\"Zero Displacement\" "
                "node_set=\"fix_bottom\">. "
                "RIGHT: <Elements type=\"hex8\" name=\"Part1\">, "
                "<bc name=\"fix\" type=\"zero displacement\" "
                "node_set=\"fix_bottom\">; while "
                "<analysis>static</analysis>, "
                "<analysis>Static</analysis> and "
                "<analysis>STATIC</analysis> all run and give "
                "bit-identical results. "
                "Signal: `Invalid element type` for the element, "
                "tag \"bc\" (line N) : `invalid value for attribute "
                "\"type\"` for the BC, both with "
                "`Reading file ...FAILED!` and exit 1. Enumerate the "
                "exact registered strings by piping the words list and "
                "quit into the febio4 binary with -nosplash; every line "
                "reads module-dot-type followed by the super-class id in "
                "brackets. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Integration] FEBio's ERROR box wraps its text at 71 "
                "columns, so a long diagnostic is split across lines "
                "inside the star frame and a fixed-string search for the "
                "whole sentence finds nothing. Reproduce it by asking "
                "the default skyline linear solver for a format it does "
                "not have: <solver type=\"solid\">"
                "<symmetric_stiffness>non-symmetric</symmetric_stiffness>"
                "</solver>, which aborts the run. "
                "WRONG: grepping the log for the fixed string \"The "
                "selected linear solver does not support the requested "
                "matrix format.\" — 0 hits on the run that printed "
                "exactly that error, because the box breaks the line "
                "after the word matrix. "
                "RIGHT: match a short fragment that cannot wrap, such as "
                "\"does not support the requested\", or strip the star "
                "frame and re-join the lines before matching. "
                "Signal: the box reads `The selected linear solver does "
                "not support the requested matrix` on one line, "
                "`format.` on the next and `Please select a different "
                "linear solver.` on a third, then "
                "`E R R O R   T E R M I N A T I O N` and exit 1. Every "
                "short message quoted in this catalog — the `tag \"x\" "
                "(line N) : ...` family, `Invalid load curve ID`, "
                "`Invalid element type` — fits on one line and can be "
                "matched whole. The four accepted values of "
                "<symmetric_stiffness> are non-symmetric, symmetric, "
                "\"symmetric structure\" and preferred; anything else, "
                "including the plausible-looking unsymmetric, gives "
                "tag \"symmetric_stiffness\" (line N) : invalid value: "
                "unsymmetric. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
        ],
    },
}


GENERATORS = {
    "linear_elasticity_3d_cube": _elasticity_3d_cube,
}
