"""FEBio active-contraction generators and knowledge.

FEBio Module type: 'solid' with active-contraction materials. The
'solid mixture' container combines a passive elastic skeleton with a
contractile fiber bundle whose internal stress varies with an
activation parameter (typically a calcium-controlled scalar). Two
common active models:
  - 'active fiber stress' — direct Ca-driven contractile stress
  - 'prescribed uniaxial active contraction' — table-driven activation

Canonical for cardiac chamber modeling, skeletal-muscle gait studies,
gastric peristalsis, and any biological tissue with controllable
internal contraction.
"""


def _active_contraction_3d_fiber(params: dict) -> str:
    """A passive neo-Hookean matrix + active contractile fiber bundle
    along the z-axis. The activation curve ramps from 0 to T_max,
    pulling the block in via fiber contraction.

    Demonstrates the FEBio idiom for cardiac active stress: solid
    mixture wraps a passive 'neo-Hookean' base and an active
    'prescribed uniaxial active contraction' fiber model with
    load-controller-driven activation.
    """
    E_passive = params.get("E", 50.0)
    nu = params.get("nu", 0.45)
    T_max = params.get("activation_max", 100.0)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="solid"/>
  <Control>
    <analysis>DYNAMIC</analysis>
    <time_steps>20</time_steps>
    <step_size>0.05</step_size>
    <solver type="solid">
      <symmetric_stiffness>symmetric</symmetric_stiffness>
    </solver>
  </Control>
  <Material>
    <material id="1" name="Material1" type="solid mixture">
      <density>1.0</density>
      <mat_axis type="vector">
        <a>0,0,1</a>
        <d>1,0,0</d>
      </mat_axis>
      <solid type="neo-Hookean">
        <density>1.0</density>
        <E>{E_passive}</E>
        <v>{nu}</v>
      </solid>
      <solid type="prescribed uniaxial active contraction">
        <T0 lc="1">{T_max}</T0>
      </solid>
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
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="Material1"/>
  </MeshDomains>
  <Boundary>
    <bc name="fix" type="zero displacement" node_set="fix_bottom">
      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>
    </bc>
  </Boundary>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>SMOOTH</interpolate><extend>CONSTANT</extend>
      <points><pt>0,0</pt><pt>0.5,1</pt><pt>1,1</pt></points>
    </load_controller>
  </LoadData>
  <Output>
    <plotfile type="febio">
      <var type="displacement"/>
      <var type="stress"/>
      <var type="fiber vector"/>
    </plotfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "active_contraction": {
        "description": (
            "Active contraction of fiber-reinforced tissue via "
            "FEBio's 'solid mixture' wrapping a passive elastic "
            "base + an active contractile fiber. The active "
            "component contributes additional fiber-axis stress "
            "controlled by a load-curve (or, in coupled "
            "cardiac-EM models, by a Ca transient). Canonical for "
            "cardiac chamber dynamics, skeletal-muscle gait "
            "studies, peristalsis."
        ),
        "input_format": "FEBio XML v4.0",
        "solver": "Standard solid solver, DYNAMIC analysis",
        "materials": {
            "solid mixture": {
                "mat_axis": "Material orientation frame (required to "
                            "define the active-fiber direction)",
                "solid (1)": "Passive elastic base (neo-Hookean, HGO, "
                             "Mooney-Rivlin, ...)",
                "solid (2)": "Active contractile fiber model. "
                             "Common options: "
                             "'prescribed uniaxial active contraction' "
                             "(T0 lc=N), "
                             "'active fiber stress' (smax, "
                             "activation; optional stl/stv "
                             "function properties and mat_axis), "
                             "'prescribed trans iso active "
                             "contraction', 'prescribed isotropic "
                             "active contraction', 'prescribed "
                             "fiber active contraction' and their "
                             "'uncoupled ...' twins ("
                             "'uncoupled active fiber stress' too). "
                             "CORRECTED 2026-08-03: 'Guccione "
                             "cardiac contraction', previously "
                             "listed here, is NOT registered in "
                             "FEBio 4.12 — executed, it is rejected "
                             "inside a solid mixture with `tag "
                             "\"solid\" (line N) : invalid value "
                             "for attribute \"type\"`. The list "
                             "above was re-checked name by name "
                             "against the binary's own `list` "
                             "factory dump on 2026-08-03; every "
                             "remaining entry is a registered "
                             "FEMATERIAL_ID.",
            },
        },
        "pitfalls": [
            (
                "[Input] An active-contraction material MUST sit inside a `solid mixture` next to a passive elastic solid. Standing alone it parses and initialises fine and then destroys the element on the first step, because nothing resists the contraction. "
                "WRONG: <material id=\"1\" name=\"M1\" type=\"prescribed uniaxial active contraction\"><T0 lc=\"1\">100.0</T0></material> as the top-level material. "
                "RIGHT: <material id=\"1\" name=\"M1\" type=\"solid mixture\"><density>1.0</density><mat_axis type=\"vector\"><a>0,0,1</a><d>1,0,0</d></mat_axis><solid type=\"neo-Hookean\"><density>1.0</density><E>50.0</E><v>0.45</v></solid><solid type=\"prescribed uniaxial active contraction\"><T0 lc=\"1\">100.0</T0></solid></material>. "
                "Signal: NOT a parse error — the deck reads `...SUCCESS!` and then fails with the WARNING `No force acting on the system.`, the ERROR `Negative jacobian detected.` — SINGULAR, with no count — then `------- failed to converge at time : <t>` and `E R R O R   T E R M I N A T I O N`, exit 1, after completing some steps. THIS CORRECTS AN EARLIER VERSION OF THIS ENTRY, which quoted `8 negative jacobians detected.` (plural, with a count of inverted integration points) and `Number of time steps completed .... : 0`. Re-executed on the shipped template: the plural string never appears and the run is not stopped at step zero, so a wrapper grepping for the quoted plural form would never match. Because it is an element-inversion failure and not a schema failure, no message names the material or the missing passive solid — treat a negative-jacobian failure early in an active model as a missing passive base first. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Input] Omitting <mat_axis> is SILENTLY equivalent to a = (1,0,0), d = (0,1,0), so the contraction acts along global x no matter how the specimen is oriented. "
                "WRONG: a `solid mixture` with an active-contraction child and no <mat_axis>, when the fibers are not along global x. "
                "RIGHT: <mat_axis type=\"vector\"><a>0,0,1</a><d>1,0,0</d></mat_axis> as a child of the solid mixture, with <a> the fiber axis. "
                "Signal: NONE — the run completes every step and ends `N O R M A L   T E R M I N A T I O N`. Detect it by comparing against an explicit frame: executed on two meshes, a deck with mat_axis omitted gave results BIT-IDENTICAL to the same deck with a = (1,0,0), which is how the default was established rather than assumed. Against the correct axis the free shortening was wrong by roughly a factor of two on both meshes — the specimen still contracts, just in the wrong direction, which is why this survives a smoke test. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] What sets the contraction is the RATIO of T0 to the passive stiffness, not T0 alone, and the response is smooth in that ratio — there is no threshold to find. "
                "WRONG: tuning T0 without reference to the passive E, or expecting a qualitative change at some particular T0. "
                "RIGHT: set the passive E first, then choose T0 near the physiological ratio; T0/E around 1 gives a large but stable contraction. "
                "Signal: none — measure the shortening along the fiber axis with <Output><logfile><node_data data=\"z\" delim=\",\" file=\"pos.csv\"/></logfile></Output> on a specimen fixed at one end and FREE at the other (a prescribed displacement at both ends hides the effect entirely). Executed as a T0/E sweep of 0.01, 0.1, 1 and 10 on two meshes: the free shortening rose monotonically from under a percent, through a few percent, to roughly a third and then nearly two thirds of the original length, and the two meshes agreed at every ratio. All four converged — a large T0/E did NOT fail here, so treat \"T0 too high\" as a modelling question, not a solver limit. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] Under <analysis>STATIC</analysis> the shape of the activation ramp does NOT change the converged answer, and a step change in T0 does not need smoothing to converge. "
                "WRONG: assuming a step activation must be smoothed before a STATIC model will converge. "
                "RIGHT: shape the ramp for physiological reasons if you want, but choose it for the physics, not for the solver. Under <analysis>DYNAMIC</analysis> the ramp rate does matter, for the inertial reason documented in the hyperelasticity STATIC-vs-DYNAMIC pitfall. "
                "Signal: none, and the reason is that there is nothing to signal. Executed on two meshes at a high T0, comparing a near-step activation (full amplitude reached in one fiftieth of the run) against a linear ramp over the whole run: both completed all their steps and reached a final configuration identical to five decimal places. This FALSIFIES the previous claim that a step change in T0 makes the first step fail to converge and that SMOOTH interpolation is required. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
        ],
    },
}


GENERATORS = {
    "active_contraction_3d_fiber": _active_contraction_3d_fiber,
}
