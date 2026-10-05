"""FEBio growth-remodeling generators and knowledge.

FEBio Module type: 'solid' with the top-level 'kinematic growth'
material, which splits the deformation gradient multiplicatively into
F = F_e * F_g. F_g comes from a growth-tensor property (`volume
growth`, `area growth`, `fiber growth`, `general growth`) and F_e is
handed to a nested elastic material. 'cell growth' (osmotic,
concentration-driven) and 'remodeling solid' (density-driven bone
remodeling) are the other two registered growth/remodeling materials.

Canonical for vascular adaptation (constrictor / dilator response of
arteries), tissue scaffolds with osteoblast-driven mineralization,
muscle hypertrophy under chronic overload, and tumor growth in
mechanobiology benchmarks.
"""


def _growth_remodeling_3d_isotropic(params: dict) -> str:
    """Isotropic volumetric growth driven by a time-ramped multiplier.

    A `kinematic growth` material wraps a neo-Hookean elastic solid and
    a `volume growth` tensor. The total deformation gradient
    F = F_e * F_g splits into an elastic (F_e) and a growth (F_g) part;
    `volume growth` sets F_g = multiplier * I, so a minimally
    constrained block grows isotropically by exactly the multiplier and
    carries no residual stress.

    The block is held by three symmetry planes (x=0 fixed in x, y=0 in
    y, z=0 in z), which removes the rigid-body modes without resisting
    the growth.
    """
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    multiplier = params.get("growth_multiplier", 1.2)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="solid"/>
  <Control>
    <analysis>STATIC</analysis>
    <time_steps>10</time_steps>
    <step_size>0.1</step_size>
    <solver type="solid">
      <symmetric_stiffness>symmetric</symmetric_stiffness>
    </solver>
  </Control>
  <Material>
    <material id="1" name="Material1" type="kinematic growth">
      <density>1.0</density>
      <elastic type="neo-Hookean">
        <density>1.0</density>
        <E>{E}</E>
        <v>{nu}</v>
      </elastic>
      <growth type="volume growth">
        <multiplier lc="1">1.0</multiplier>
      </growth>
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
    <Elements type="hex8" name="Part1">
      <elem id="1">1,2,3,4,5,6,7,8</elem>
    </Elements>
    <NodeSet name="face_x0">1,4,5,8</NodeSet>
    <NodeSet name="face_y0">1,2,5,6</NodeSet>
    <NodeSet name="face_z0">1,2,3,4</NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="Material1"/>
  </MeshDomains>
  <Boundary>
    <bc name="sym_x" type="zero displacement" node_set="face_x0">
      <x_dof>1</x_dof>
    </bc>
    <bc name="sym_y" type="zero displacement" node_set="face_y0">
      <y_dof>1</y_dof>
    </bc>
    <bc name="sym_z" type="zero displacement" node_set="face_z0">
      <z_dof>1</z_dof>
    </bc>
  </Boundary>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>LINEAR</interpolate><extend>CONSTANT</extend>
      <points><pt>0,1</pt><pt>1,{multiplier}</pt></points>
    </load_controller>
  </LoadData>
  <Output>
    <plotfile type="febio">
      <var type="displacement"/>
      <var type="stress"/>
      <var type="relative volume"/>
    </plotfile>
    <logfile>
      <node_data data="x;y;z" delim="," file="pos.csv"/>
      <element_data data="sx;sy;sz;J" delim="," file="el.csv"/>
    </logfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "growth_remodeling": {
        "description": (
            "Growth-and-remodeling of biological tissue via FEBio's "
            "multiplicative split F = F_e * F_g, carried by the "
            "top-level `kinematic growth` material. Mass is added "
            "(growth) or the solid density changes (remodeling), "
            "driven by a load curve or a biological stimulus. "
            "Canonical for arterial adaptation, tissue scaffolds, "
            "muscle hypertrophy, and tumor mechanobiology."
        ),
        "input_format": "FEBio XML v4.0",
        "solver": "Standard solid solver; F_g is evaluated from the "
                  "current growth-tensor parameters at every update",
        "materials": {
            "kinematic growth": {
                "_placement": "TOP-LEVEL material. Executed "
                              "2026-08-03: as the sole <solid> of a "
                              "'solid mixture' the deck reads but the "
                              "solve fails at the first increment.",
                "elastic": "REQUIRED nested elastic material, and it "
                           "must be a COUPLED one. Executed: "
                           "<elastic type=\"neo-Hookean\"> works; "
                           "<elastic type=\"Mooney-Rivlin\"> (an "
                           "uncoupled material) is rejected at "
                           "initialisation with `Elastic material "
                           "should not be of type uncoupled`.",
                "growth": "REQUIRED nested growth tensor, one of "
                          "`volume growth`, `area growth`, `fiber "
                          "growth`, `general growth` (all executed). "
                          "The first three take <multiplier>; "
                          "`general growth` takes <iso> and <ani>. "
                          "There is no <theta> and no <gm>.",
                "fiber": "OPTIONAL child of the GROWTH TENSOR (not of "
                         "the material) giving the anisotropy axis: "
                         "<fiber type=\"vector\"><vector>0,0,1"
                         "</vector></fiber>. Default is (1,0,0), i.e. "
                         "global x. `volume growth` ignores it.",
                "mat_axis": "NOT a parameter of this material. "
                            "Executed 2026-08-03: a <mat_axis> child "
                            "of <material type=\"kinematic growth\"> "
                            "is rejected with `tag \"mat_axis\" "
                            "(line N) : unrecognized tag`. To vary "
                            "the growth axis per element use "
                            "<MeshData><ElementData type=\"mat_axis\" "
                            "elem_set=\"...\"> placed AFTER "
                            "</MeshDomains>; that route is accepted "
                            "and does rotate the growth direction.",
                "_verified": (
                    "LIVE-VERIFIED 2026-08-03 on FEBio "
                    "4.12.0.86045466d — this material runs a hex8 "
                    "deck to N O R M A L   T E R M I N A T I O N: "
                    "<material id=\"1\" name=\"M1\" type=\"kinematic "
                    "growth\"><density>1</density><elastic "
                    "type=\"neo-Hookean\"><density>1</density>"
                    "<E>1000</E><v>0.3</v></elastic><growth "
                    "type=\"volume growth\"><multiplier lc=\"1\">1.0"
                    "</multiplier></growth></material> with the "
                    "multiplier ramped by load curve 1. On a block "
                    "held only by three symmetry planes the edge "
                    "length ends at exactly the multiplier and the "
                    "stress stays at round-off, so the growth is "
                    "driving the deformation and nothing is "
                    "resisting it."),
            },
            "cell growth": (
                "Registered FEMATERIAL_ID, osmotic. Executed "
                "2026-08-03 inside a 'solid mixture' next to a "
                "neo-Hookean: <solid type=\"cell growth\"><phir>0.5"
                "</phir><cr>300</cr><ce>300</ce></solid> runs to "
                "N O R M A L   T E R M I N A T I O N. It REQUIRES a "
                "Globals block — <Globals><Constants><R>8.314e-6</R>"
                "<T>310</T></Constants></Globals> — without which "
                "initialisation fails with `A positive universal gas "
                "constant R must be defined in Globals section`."),
            "remodeling solid": (
                "Registered FEMATERIAL_ID, density-driven bone "
                "remodeling. Executed 2026-08-03 as a top-level "
                "material: <material id=\"1\" name=\"M1\" "
                "type=\"remodeling solid\"><density>1.0</density>"
                "<min_density>0.1</min_density><max_density>10.0"
                "</max_density><solid type=\"Carter-Hayes (old)\">"
                "<density>1.0</density><c>1000</c><gamma>2</gamma>"
                "<v>0.3</v></solid><supply type=\"Huiskes-supply\">"
                "<B>1.0</B><k>0.25</k></supply></material> runs to "
                "N O R M A L   T E R M I N A T I O N. Both <solid> "
                "and <supply> are REQUIRED, and <solid> must be a "
                "remodeling-capable elastic material such as "
                "`Carter-Hayes (old)`."),
            "_not_registered": (
                "`growth`, `remodeling` and `isotropic growth` are "
                "NOT material types in FEBio 4.12. All three were "
                "executed on 2026-08-03 and every one was rejected "
                "with tag \"material\" (line N) : `invalid value for "
                "attribute \"type\"` and `Reading file "
                "...FAILED!`. The registered growth / remodeling "
                "FEMATERIAL_ID factories are `kinematic growth`, "
                "`cell growth` and `remodeling solid`. `volume "
                "growth` / `area growth` / `fiber growth` / `general "
                "growth` are growth-TENSOR properties nested inside "
                "`kinematic growth` — used as a top-level "
                "<material type=\"volume growth\"> they draw the same "
                "invalid-attribute rejection."),
        },
        "pitfalls": [
            (
                "[Input] `kinematic growth` is a TOP-LEVEL material "
                "and it has two REQUIRED child properties, <elastic> "
                "and <growth>; the elastic child must be a coupled "
                "material, not an uncoupled one. Do not wrap it in a "
                "'solid mixture' and do not write `growth`, "
                "`remodeling` or `isotropic growth` as the type — "
                "none of those three is a registered material. "
                "WRONG: <material id=\"1\" name=\"M1\" type=\"solid "
                "mixture\"><density>1.0</density><solid "
                "type=\"kinematic growth\"><elastic "
                "type=\"neo-Hookean\"><density>1.0</density><E>1000"
                "</E><v>0.3</v></elastic><growth type=\"volume "
                "growth\"><multiplier lc=\"1\">1.0</multiplier>"
                "</growth></solid></material>. "
                "RIGHT: <material id=\"1\" name=\"M1\" "
                "type=\"kinematic growth\"><density>1.0</density>"
                "<elastic type=\"neo-Hookean\"><density>1.0</density>"
                "<E>1000</E><v>0.3</v></elastic><growth "
                "type=\"volume growth\"><multiplier lc=\"1\">1.0"
                "</multiplier></growth></material>. "
                "Signal: four different failures, one per mistake. "
                "The mixture form READS successfully and then dies at "
                "the first increment: the Newton loop hits `Max nr of "
                "iterations reached.` and reforms the stiffness "
                "repeatedly, then `------- failed to converge at time "
                ": 0.1`, `Number of time steps completed "
                ".................... : 0` and `E R R O R   T E R M I "
                "N A T I O N`, exit 1. THIS CORRECTS AN EARLIER "
                "VERSION OF THIS ENTRY, which said the mixture form "
                "dies with `8 negative jacobians detected.`. "
                "Re-executed on the shipped template: the plural "
                "`negative jacobians detected.` never appears, but "
                "after three `Max nr of iterations reached.` warnings "
                "a single `Negative jacobian detected.` is reported (an "
                "element inverts in a Newton iterate) before the step "
                "fails. A missing child is caught at "
                "read time: `Component \"M1\" needs to have property "
                "\"growth\" defined (line N)` or the same message "
                "with \"elastic\". An uncoupled elastic child is "
                "caught at initialisation: `Elastic material should "
                "not be of type uncoupled`, then `Failed initializing "
                "material 1 (name=\"M1\")` and `Model initialization "
                "failed`. An unregistered type never gets that far: "
                "tag \"material\" (line N) : `invalid value for "
                "attribute \"type\"` and `Reading file "
                "...FAILED!`. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] Growth in `kinematic growth` is NOT "
                "accumulated history: F_g is recomputed from the "
                "CURRENT value of <multiplier> at every update, so "
                "the deformation is path-independent and fully "
                "reversible — a load curve that comes back down "
                "un-grows the tissue completely. What must be carried "
                "across a restart is the multiplier VALUE, not a "
                "hidden state variable. "
                "WRONG: continuing a grown model with <points>"
                "<pt>0,1</pt><pt>1,1.2</pt></points> redefined in the "
                "restart file, so the curve starts over at 1 — the "
                "block shrinks straight back to its original size. "
                "RIGHT: keep <extend>CONSTANT</extend> and let the "
                "curve continue from the value already reached, e.g. "
                "phase 1 <points><pt>0,1</pt><pt>1,1.2</pt></points> "
                "run to t=0.5, then restart with the SAME curve so "
                "the multiplier goes on rising; run FEBio as "
                "\"febio4 -i model.feb -dump=1 model.dmp\" and "
                "continue with \"febio4 -r cont.feb\" where cont.feb "
                "is <febio_restart version=\"4.0\"><Archive>model.dmp"
                "</Archive><Step><step id=\"2\" name=\"Step2\">"
                "<Control>...</Control></step></Step>"
                "</febio_restart>. "
                "Signal: the restart itself announces "
                "`- R E S T A R T -` and `Restarting from time 0.5.`, "
                "and both the right and the wrong continuation reach "
                "`N O R M A L   T E R M I N A T I O N` — the loss of "
                "growth is silent. Measure it with <element_data "
                "data=\"sx;sy;sz;J\"/> under <Output><logfile>: on a "
                "minimally constrained block J equals the multiplier "
                "cubed, so J falling back toward 1 is the tell. "
                "Executed three ways on the same block: a triangular "
                "curve up to a peak multiplier and back to 1 returned "
                "the edge length to exactly its starting value with "
                "no residual stress; reaching the same multiplier in "
                "2 steps and in 20 steps gave the same final size; "
                "and the dump-restart reproduced the uninterrupted "
                "run exactly when the curve was continued, but "
                "un-grew the block when the curve was reset. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] Growth against a fully clamped boundary "
                "does NOT diverge — it converges quietly to a "
                "motionless body full of residual stress, which is "
                "far more dangerous than a crash. Constrain only the "
                "rigid-body modes (three symmetry planes, or one "
                "corner plus rotation) unless the residual stress is "
                "the thing you are modelling. "
                "WRONG: <bc name=\"fixall\" type=\"zero "
                "displacement\" node_set=\"all_outer_faces\">"
                "<x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>"
                "</bc> on a growing domain. "
                "RIGHT: <bc name=\"sym_x\" type=\"zero displacement\" "
                "node_set=\"face_x0\"><x_dof>1</x_dof></bc> plus the "
                "same for face_y0 in y and face_z0 in z. "
                "Signal: NONE — the clamped run reads `...SUCCESS!`, "
                "completes every step, needs only one equilibrium "
                "iteration per step and ends in `N O R M A L   T E R "
                "M I N A T I O N`. Detect it from the results, not "
                "from the log: request <element_data "
                "data=\"sx;sy;sz;J\"/> under <Output><logfile>. "
                "Executed as a multiplier sweep at 1.05, 1.1, 1.2 and "
                "1.4 on 3x3x3 and 5x5x5 hex8 blocks, with the two "
                "meshes agreeing to every printed digit: minimally "
                "constrained, J tracks the multiplier cubed and the "
                "stress stays at round-off; fully clamped, J stays at "
                "1, nothing moves, and the residual stress passes a "
                "tenth of the elastic modulus by a 5% growth "
                "multiplier and exceeds the modulus itself before a "
                "40% one. Pushed to multipliers of 2, 3 and 5 the "
                "clamped model still terminated normally, with the "
                "residual stress running into hundreds of times the "
                "modulus — there is no divergence to warn you. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Input] Anisotropy comes from WHICH growth-tensor "
                "type you nest, and its axis comes from the growth "
                "tensor's own OPTIONAL <fiber> child, which defaults "
                "SILENTLY to global x — not from <mat_axis>, which "
                "this material does not accept at all. `volume "
                "growth` is isotropic and ignores the axis; `fiber "
                "growth` grows along the axis only; `area growth` "
                "grows the plane normal to the axis only; `general "
                "growth` takes separate <iso> and <ani> multipliers. "
                "WRONG: <material id=\"1\" name=\"M1\" "
                "type=\"kinematic growth\"><mat_axis type=\"vector\">"
                "<a>0,0,1</a><d>1,0,0</d></mat_axis>...</material>, "
                "and equally wrong is <growth type=\"fiber growth\">"
                "<multiplier lc=\"1\">1.0</multiplier></growth> for a "
                "vessel whose axis is z. "
                "RIGHT: <growth type=\"fiber growth\"><multiplier "
                "lc=\"1\">1.0</multiplier><fiber type=\"vector\">"
                "<vector>0,0,1</vector></fiber></growth>. For a "
                "per-element axis put <MeshData><ElementData "
                "type=\"mat_axis\" elem_set=\"allel\"><e lid=\"1\">"
                "<a>0,0,1</a><d>1,0,0</d></e>...</ElementData>"
                "</MeshData> AFTER the </MeshDomains> section. "
                "Signal: the <mat_axis> mistake is caught — tag "
                "\"mat_axis\" (line N) : `unrecognized tag` — and so "
                "is the old `theta` spelling, tag \"theta\" (line N) "
                ": `unrecognized tag`, and MeshData placed too early, "
                "`MeshData must appear after MeshDomain section. "
                "(line N)`. The default-axis mistake is NOT caught: "
                "the run terminates normally and grows the wrong way. "
                "Executed on 2x2x2 and 4x4x4 hex8 blocks held by "
                "three symmetry planes, identical on both meshes: "
                "`volume growth` stretched all three edges by exactly "
                "the multiplier; `fiber growth` with no <fiber> "
                "stretched the global-x edge by exactly the "
                "multiplier and left y and z at their original "
                "length; adding <fiber><vector>0,0,1</vector></fiber> "
                "moved that stretch onto z, as did the MeshData "
                "mat_axis route; `area growth` left the axis alone "
                "and stretched the other two edges by the square root "
                "of the multiplier. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
        ],
    },
}


GENERATORS = {
    "growth_remodeling_3d_isotropic": _growth_remodeling_3d_isotropic,
}
