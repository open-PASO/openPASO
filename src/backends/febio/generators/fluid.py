"""FEBio fluid generators and knowledge.

FEBio Module type: 'fluid'. Incompressible Newtonian fluid solved via
FEBio's pressure-velocity fluid solver. The hallmark module for
cardiovascular CFD, blood-flow benchmarks, and the fluid half of FSI
problems before coupling with 'fluid-FSI'.
"""


def _fluid_3d_channel(params: dict) -> str:
    """Pressure-driven channel flow in a 1x1x1 cube.

    Inlet face (x=0) has prescribed effective pressure; outlet face
    (x=1) has zero effective pressure; lateral walls are no-slip.
    """
    rho = params.get("density", 1.0)
    mu = params.get("viscosity", 0.01)
    p_in = params.get("p_inlet", 0.01)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="fluid"/>
  <Control>
    <analysis>DYNAMIC</analysis>
    <time_steps>10</time_steps>
    <step_size>0.1</step_size>
    <solver type="fluid">
      <symmetric_stiffness>non-symmetric</symmetric_stiffness>
      <linear_solver type="bicgstab"/>
    </solver>
  </Control>
  <Material>
    <material id="1" name="Material1" type="fluid">
      <density>{rho}</density>
      <k>1e3</k>
      <viscous type="Newtonian fluid">
        <mu>{mu}</mu>
      </viscous>
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
    <Elements type="hex8" mat="Material1" name="Part1">
      <elem id="1">1,2,3,4,5,6,7,8</elem>
    </Elements>
    <NodeSet name="inlet">1,4,5,8</NodeSet>
    <NodeSet name="outlet">2,3,6,7</NodeSet>
    <NodeSet name="walls">1,2,3,4,5,6,7,8</NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="Material1"/>
  </MeshDomains>
  <Boundary>
    <bc name="noslip" type="zero fluid velocity" node_set="walls">
      <wy_dof>1</wy_dof>
      <wz_dof>1</wz_dof>
    </bc>
    <bc name="p_in" type="prescribed fluid dilatation" node_set="inlet">
      <value lc="1">{p_in}</value>
    </bc>
    <bc name="p_out" type="prescribed fluid dilatation" node_set="outlet">
      <value lc="1">0.0</value>
    </bc>
  </Boundary>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>LINEAR</interpolate><extend>CONSTANT</extend>
      <points><pt>0,0</pt><pt>1,1</pt></points>
    </load_controller>
  </LoadData>
  <Output>
    <plotfile type="febio">
      <var type="fluid velocity"/>
      <var type="effective fluid pressure"/>
    </plotfile>
    <logfile>
      <node_data data="nfvx;nfvy;nfvz" delim="," file="fluid_vel.csv"/>
      <element_data data="fJ;fd;fp" delim="," file="fluid_elem.csv"/>
    </logfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "fluid": {
        "description": (
            "Incompressible / quasi-incompressible Newtonian fluid via "
            "FEBio's fluid solver (Module type='fluid'). Velocity + "
            "dilatation primary variables; pressure is derived. Suited "
            "to cardiovascular CFD benchmarks and the fluid half of "
            "FSI workflows."
        ),
        "input_format": "FEBio XML v4.0",
        "solver": "Non-symmetric Newton-Raphson (fluid solver)",
        "materials": {
            "fluid": {
                "density": "Mass density rho",
                "k": "Bulk modulus (penalises near-incompressibility)",
                "viscous": "Nested viscous law: 'Newtonian fluid' (mu, "
                           "kappa), 'Carreau' (mu0, mui, lambda, n), "
                           "'Carreau-Yasuda' (mu0, mui, lambda, n, a), "
                           "'Bingham', 'Powell-Eyring', 'Cross' or "
                           "'Quemada'. There is no 'power-law' viscous "
                           "type.",
            },
        },
        "pitfalls": [
            (
                "[Syntax] A `fluid` material needs <Module type=\"fluid\"/>, and the first thing that fails if you leave the module as solid is the SOLVER, not the material — so the message points somewhere you did not edit. "
                "WRONG: <Module type=\"solid\"/> with <solver type=\"fluid\"> and a `fluid` material. "
                "RIGHT: <Module type=\"fluid\"/> with <solver type=\"fluid\"><symmetric_stiffness>non-symmetric</symmetric_stiffness><linear_solver type=\"bicgstab\"/></solver>. "
                "Signal: tag \"solver\" (line N) : `invalid value for attribute \"type\"` and `Reading file ...FAILED!`. The solver factory is module-scoped, so an out-of-module solver name is rejected before the <Material> section is ever read. Do not go looking for a material problem. (Executed 2026-08-05, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Syntax] Fluid velocity DOFs are wx/wy/wz, and a solid-style name is a hard parse error indistinguishable from a made-up one. "
                "WRONG: <bc type=\"zero fluid velocity\" node_set=\"walls\"><y_dof>1</y_dof></bc>. "
                "RIGHT: <bc name=\"noslip\" type=\"zero fluid velocity\" node_set=\"walls\"><wy_dof>1</wy_dof><wz_dof>1</wz_dof></bc>. "
                "Signal: tag \"y_dof\" (line N) : `unrecognized tag` and `Reading file ...FAILED!`. Executed against a deliberately invented name too: the message is byte-identical, so it tells you the tag is unknown and nothing more — it will not hint that you wanted the w prefix. (Executed 2026-08-05, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] The `fluid` material's <k> is a bulk modulus, and whether it controls anything depends on how you drive the problem. In a deck whose inlet and outlet PRESCRIBE the fluid dilatation, k does not set the volume ratio at all — the boundary condition does. "
                "WRONG: tuning <k> to control compressibility in a dilatation-driven deck, or reading an unchanged volume ratio as evidence that k is ignored. "
                "RIGHT: to make k matter, drive the flow with a velocity or a traction and leave the dilatation free; to control the volume ratio in a dilatation-driven deck, change the prescribed value. "
                "Signal: none — measure it. The fluid log variables are NOT the solid ones: use <element_data data=\"fJ;fd;fp\"/> for volume ratio, dilatation and pressure, and <node_data data=\"nfvx;nfvy;nfvz\"/> for nodal velocity. The rejection is per RECORD TYPE, not per name: <element_data data=\"ef\"/> gives `\"ef\" is not a valid field variable name (line N)` as part of a READ FAILURE — the run prints `Reading file ...FAILED!` and stops there. AN EARLIER VERSION OF THIS ENTRY said that rejection arrives after a successful read; re-executed, it does not. By contrast <node_data data=\"ef\"/> is ACCEPTED and runs to `N O R M A L   T E R M I N A T I O N` — `ef` is the nodal dilatation DOF, and it is exactly what the shipped fluid_fsi and biphasic_fsi templates log. Check which record type you are writing into before concluding a name is invalid; an earlier version of this entry said \"ef\" is rejected without that qualification, which contradicts the templates in this same catalog. (Both directions executed 2026-08-05.) Executed as a k sweep over six decades on the shipped channel deck: fJ came back identical at every value, because the inlet and outlet prescribe it. "
                "STILL UNVERIFIED: the size of the spurious dilatation k admits when the dilatation is genuinely free. Closing it needs a velocity- or traction-driven channel, which is a different deck from the one shipped here. Retained rather than softened. (Executed 2026-08-05, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] CFL-like restriction on dt for "
                "convection-dominated flow: dt < h / max|v|. The "
                "fluid solver is implicit but accuracy still depends "
                "on dt; very large steps smooth out transient "
                "features. Signal: a vortex-shedding benchmark "
                "(cylinder in cross-flow) shows no shedding when dt "
                "is larger than ~0.1 * shedding_period. (Audit "
                "2026-06-02.)"
            ),
        ],
    },
}


GENERATORS = {
    "fluid_3d_channel": _fluid_3d_channel,
}
