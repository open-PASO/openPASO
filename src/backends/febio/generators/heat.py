"""FEBio heat-conduction generators and knowledge.

READ THIS FIRST. FEBio 4.12 has NO `heat` module and NO solid-conduction
solver. The way to solve a heat-conduction problem in FEBio 4.12 is the
`thermo-fluid` module with the fluid held at rest: with every fluid
velocity DOF fixed, the thermo-fluid energy equation reduces to Fourier
conduction, and the steady state reproduces the analytic linear profile
exactly on every mesh. That is what `heat_3d_bar` emits.

Every string in this file was produced by running febio4
(4.12.0.86045466d, USE_MKL=OFF).
"""


def _heat_3d_bar(params: dict) -> str:
    """Steady-state 1-D heat conduction along a bar of `n` hex8
    elements, with the fluid held at rest so the thermo-fluid energy
    equation reduces to Fourier conduction.

    Prescribed temperature `T_cold` at x = 0 and `T_hot` at x = L.
    The analytic steady solution is linear in x and hex8 is trilinear,
    so this deck doubles as a conduction patch test.
    """
    n = max(1, int(params.get("n", 8)))
    L = float(params.get("L", 1.0))
    W = float(params.get("width", 0.2))
    K = float(params.get("conductivity", 0.026))
    T_cold = float(params.get("T_cold", 300.0))
    T_hot = float(params.get("T_hot", 400.0))
    rho = float(params.get("density", 1.2))
    mu = float(params.get("viscosity", 1.8e-5))
    M = float(params.get("molar_mass", 0.029))
    cp = float(params.get("cp_normalized", 3.5))
    analysis = str(params.get("analysis", "STEADY-STATE")).upper()
    steps = max(1, int(params.get("time_steps", 1)))
    dt = float(params.get("step_size", 1.0))

    def nid(i, j, k):
        return 1 + i + (n + 1) * (j + 2 * k)

    nodes = [f'      <node id="{nid(i, j, k)}">'
             f'{i * L / n},{j * W},{k * W}</node>'
             for k in (0, 1) for j in (0, 1) for i in range(n + 1)]
    elems = []
    for i in range(n):
        c = (nid(i, 0, 0), nid(i + 1, 0, 0), nid(i + 1, 1, 0), nid(i, 1, 0),
             nid(i, 0, 1), nid(i + 1, 0, 1), nid(i + 1, 1, 1), nid(i, 1, 1))
        elems.append(f'      <elem id="{i + 1}">'
                     + ",".join(str(x) for x in c) + "</elem>")
    cold = ",".join(str(nid(0, j, k)) for k in (0, 1) for j in (0, 1))
    hot = ",".join(str(nid(n, j, k)) for k in (0, 1) for j in (0, 1))
    alln = ",".join(str(nid(i, j, k))
                    for k in (0, 1) for j in (0, 1) for i in range(n + 1))
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="thermo-fluid"/>
  <Control>
    <analysis>{analysis}</analysis>
    <time_steps>{steps}</time_steps>
    <step_size>{dt}</step_size>
    <solver type="thermo-fluid">
      <linear_solver type="bicgstab"/>
    </solver>
  </Control>
  <Globals>
    <Constants>
      <R>8.31446</R>
      <T>{T_cold}</T>
      <P>101325</P>
    </Constants>
  </Globals>
  <Material>
    <material id="1" name="Material1" type="thermo-fluid">
      <density>{rho}</density>
      <viscous type="Newtonian fluid">
        <mu>{mu}</mu>
      </viscous>
      <elastic type="ideal gas">
        <M>{M}</M>
        <ao type="const"><value>{cp}</value></ao>
        <cp type="const"><value>{cp}</value></cp>
      </elastic>
      <conduct type="constant thermal conductivity">
        <K>{K}</K>
      </conduct>
    </material>
  </Material>
  <Mesh>
    <Nodes name="Object1">
{chr(10).join(nodes)}
    </Nodes>
    <Elements type="hex8" name="Part1">
{chr(10).join(elems)}
    </Elements>
    <NodeSet name="cold_face">{cold}</NodeSet>
    <NodeSet name="hot_face">{hot}</NodeSet>
    <NodeSet name="all_nodes">{alln}</NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="Material1"/>
  </MeshDomains>
  <Initial>
    <ic name="T0" type="initial fluid temperature" node_set="all_nodes">
      <value>{T_cold}</value>
    </ic>
  </Initial>
  <Boundary>
    <bc name="at_rest" type="zero fluid velocity" node_set="all_nodes">
      <wx_dof>1</wx_dof><wy_dof>1</wy_dof><wz_dof>1</wz_dof>
    </bc>
    <bc name="cold" type="prescribed fluid temperature" node_set="cold_face">
      <value lc="1">{T_cold}</value>
    </bc>
    <bc name="hot" type="prescribed fluid temperature" node_set="hot_face">
      <value lc="1">{T_hot}</value>
    </bc>
  </Boundary>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>LINEAR</interpolate><extend>CONSTANT</extend>
      <points><pt>0,0</pt><pt>1,1</pt></points>
    </load_controller>
  </LoadData>
  <Output>
    <logfile>
      <node_data data="T" delim="," file="heat_bar_T.csv"/>
    </logfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "heat": {
        "description": (
            "Heat conduction on FEBio 4.12 via the `thermo-fluid` "
            "module with the fluid held at rest. There is NO `heat` "
            "module and NO solid-conduction solver in FEBio 4.12: the "
            "binary registers exactly ten modules (solid, biphasic, "
            "solute, multiphasic, fluid, fluid-FSI, multiphasic-FSI, "
            "fluid-solutes, thermo-fluid, polar fluid). Fixing every "
            "fluid velocity DOF reduces the thermo-fluid energy "
            "equation to Fourier conduction; the steady state then "
            "reproduces the analytic linear profile on every mesh, to "
            "the linear solver's tolerance rather than to "
            "discretization error. Template: heat_3d_bar."),
        "input_format": "FEBio XML v4.0",
        "status": (
            "EXECUTED on FEBio 4.12.0.86045466d. heat_3d_bar runs to "
            "N O R M A L   T E R M I N A T I O N with the time step "
            "completed and a temperature field matching the analytic "
            "linear conduction profile at 2, 4, 8 and 16 elements."),
        "required_deck_skeleton": (
            "ALL SIX blocks below are REQUIRED. Omitting any one is a "
            "failure, and two of the failures are silent or "
            "misleading. "
            "(1) <Module type=\"thermo-fluid\"/> as the FIRST tag. "
            "(2) <Control> with <analysis>STEADY-STATE</analysis> (or "
            "DYNAMIC) and a <solver type=\"thermo-fluid\"> child that "
            "itself contains <linear_solver type=\"bicgstab\"/>. "
            "(3) <Globals><Constants> holding R, T and P. "
            "(4) <Material> of type `thermo-fluid` with all four "
            "children <density>, <viscous>, <elastic>, <conduct>. "
            "(5) a `zero fluid velocity` BC on EVERY node. "
            "(6) prescribed temperatures on the driving faces. "
            "The complete runnable deck is the heat_3d_bar template — "
            "start from it rather than assembling these by hand."),
        "materials": {
            "thermo-fluid": (
                "The only conduction-capable material. REQUIRED "
                "children, all four: <density>; "
                "<viscous type=\"Newtonian fluid\"><mu>1.8e-5</mu>"
                "</viscous>; <elastic type=\"ideal gas\">; "
                "<conduct type=\"constant thermal conductivity\">. "
                "Registered as `thermo-fluid.thermo-fluid "
                "[FEMATERIAL_ID]`."),
            "ideal gas": (
                "Goes in the <elastic> slot. REQUIRED parameter <M> "
                "(molar mass, > 0). REQUIRED properties <ao> and <cp>, "
                "each a FEFunction1D — write them as "
                "<ao type=\"const\"><value>3.5</value></ao>. The "
                "registered FEFUNCTION1D_ID types are exactly: point, "
                "const, scale, linear ramp, step, math. cp here is the "
                "isobaric specific heat capacity NORMALISED by R/M, so "
                "the physical cp is cp_normalized * R / M — this matters "
                "for the diffusion-length check in the [Numerical] "
                "pitfall. Alternatives registered in the same slot: "
                "`real gas`, `real vapor`, `real liquid`."),
            "constant thermal conductivity": (
                "Goes in the <conduct> slot. REQUIRED parameter <K> "
                "(>= 0), the Fourier conductivity. Alternatives in the "
                "same slot: `temp-dependent thermal conductivity`, "
                "`real vapor thermal conductivity`."),
            "Newtonian fluid": (
                "Goes in the <viscous> slot. <mu> is the shear "
                "viscosity; <kappa> (bulk viscosity) is OPTIONAL. With "
                "every velocity DOF fixed neither value affects the "
                "temperature field."),
        },
        "boundary_conditions": {
            "prescribed fluid temperature": (
                "REQUIRED to drive the problem. Takes a <value> child; "
                "the lc= attribute pointing at a <load_controller> is "
                "OPTIONAL but ramps the boundary smoothly. Registered as "
                "`thermo-fluid.prescribed fluid temperature [FEBC_ID]`."),
            "zero fluid velocity": (
                "REQUIRED for a pure-conduction problem, on a node set "
                "containing EVERY node. Children "
                "<wx_dof>1</wx_dof><wy_dof>1</wy_dof><wz_dof>1</wz_dof>. "
                "Note the w prefix: fluid velocity DOFs are wx/wy/wz, "
                "NOT x/y/z."),
            "initial fluid temperature": (
                "OPTIONAL for STEADY-STATE, strongly recommended for "
                "DYNAMIC. Goes in an <Initial> block as "
                "<ic type=\"initial fluid temperature\" node_set=\"...\">"
                "<value>300</value></ic>. Without it the initial "
                "temperature is 0, which for an absolute-temperature "
                "field is unphysical and makes the transient artefact in "
                "the [Numerical] pitfall far larger."),
        },
        "pitfalls": [
            (
                "[Syntax] There is no `heat` module in FEBio 4.12, and "
                "an unregistered <Module type=...> value is not "
                "diagnosed — it is a hard crash. "
                "WRONG: <Module type=\"heat\"/>. "
                "RIGHT: <Module type=\"thermo-fluid\"/>. "
                "Signal: stdout stops mid-line at `Reading file "
                "<name>.feb ...` with no `SUCCESS!`, no `FAILED!`, no "
                "ERROR box and no .log file, and the process is killed "
                "by SIGSEGV (signal 11; a shell reports exit status "
                "139). Identical for type=\"biphasic-FSI\" and for mere "
                "case errors such as type=\"Solid\". A wrapper that only "
                "greps stderr for the word `error` sees a completely "
                "silent failure. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d; fixture "
                "scripts/tier2_fixtures/febio/"
                "unknown_module_type_segfaults.)"
            ),
            (
                "[Syntax] There is no `isotropic Fourier` material. The "
                "FEMATERIAL_ID factory has 159 entries and contains no "
                "solid-conduction law at all. "
                "WRONG: <material id=\"1\" type=\"isotropic Fourier\">"
                "<density>1.0</density><capacity>1.0</capacity>"
                "<conductivity>1.0</conductivity></material>. "
                "RIGHT: <material id=\"1\" name=\"Material1\" "
                "type=\"thermo-fluid\"><density>1.2</density>"
                "<viscous type=\"Newtonian fluid\"><mu>1.8e-5</mu>"
                "</viscous><elastic type=\"ideal gas\"><M>0.029</M>"
                "<ao type=\"const\"><value>3.5</value></ao>"
                "<cp type=\"const\"><value>3.5</value></cp></elastic>"
                "<conduct type=\"constant thermal conductivity\">"
                "<K>0.026</K></conduct></material>. "
                "Signal: tag \"material\" (line N) : `invalid value for "
                "attribute \"type\"` and `Reading file ...FAILED!`. "
                "All three spellings \"isotropic Fourier\", \"Fourier\" "
                "and \"isotropic_Fourier\" fail identically — there is "
                "no short or underscore variant that works. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Syntax] There is no `prescribed temperature` boundary "
                "condition. The only temperature BCs in the FEBC_ID "
                "registry are `thermo-fluid.prescribed fluid "
                "temperature`, `thermo-fluid.zero fluid temperature` and "
                "`thermo-fluid.natural temperature`, all scoped to the "
                "thermo-fluid module and acting on a FLUID temperature "
                "DOF. "
                "WRONG: <bc type=\"prescribed temperature\" "
                "node_set=\"hot_face\"><value>400</value></bc>. "
                "RIGHT: <bc type=\"prescribed fluid temperature\" "
                "node_set=\"hot_face\"><value lc=\"1\">400</value></bc>. "
                "Signal: tag \"bc\" (line N) : `invalid value for "
                "attribute \"type\"` and `Reading file ...FAILED!`. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Solver] The thermo-fluid stiffness matrix is "
                "NON-SYMMETRIC and the default linear solver on a "
                "USE_MKL=OFF build (skyline) cannot store it, so the "
                "<linear_solver> child is REQUIRED. "
                "WRONG: <solver type=\"thermo-fluid\"/> — inherits the "
                "skyline default. "
                "RIGHT: <solver type=\"thermo-fluid\">"
                "<linear_solver type=\"bicgstab\"/></solver>. "
                "Signal: the deck reads `...SUCCESS!`, then an ERROR box "
                "reading `The selected linear solver does not support "
                "the requested matrix format.` / `Please select a "
                "different linear solver.`, then `Number of time steps "
                "completed .... : 0` and `E R R O R   T E R M I N A T I "
                "O N` with exit 1. Executed across the registered "
                "solvers on this build: `bicgstab` works; `skyline`, "
                "`fgmres`, `superlu_mt`, `pardiso-project` and "
                "`accelerate` all report the matrix-format error; `LU` "
                "reports `Fatal error in factorization of stiffness "
                "matrix. Aborting run.` instead. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Input] A pure-conduction problem REQUIRES every fluid "
                "velocity DOF to be fixed. thermo-fluid solves momentum "
                "and energy together; leaving the velocity free leaves "
                "the momentum problem unconstrained and Newton never "
                "converges. "
                "WRONG: omitting the velocity BC. "
                "RIGHT: <bc name=\"at_rest\" type=\"zero fluid velocity\" "
                "node_set=\"all_nodes\"><wx_dof>1</wx_dof>"
                "<wy_dof>1</wy_dof><wz_dof>1</wz_dof></bc> where "
                "all_nodes contains EVERY node. "
                "Signal: `------- failed to converge at time : 1` "
                "repeated, then `Number of time steps completed .... : 0` "
                "and `E R R O R   T E R M I N A T I O N` with exit 1. The "
                "logfile CSV is still written and still holds exactly one "
                "`*Step  = 0` block, which is the INITIAL CONDITION and "
                "not zeros — on the shipped deck, whose <Initial> sets "
                "300, every node reads 300. It reads zero only if the "
                "<Initial> block is absent as well. So do not detect this "
                "by looking for a zero field; detect it by the "
                "completed-step count. (This corrects an earlier version "
                "of this entry, which said the logged field is all zeros "
                "without that qualification; both variants were executed "
                "2026-08-06.) "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] A DYNAMIC thermo-fluid run whose thermal "
                "diffusion length is shorter than one element returns a "
                "WRONG ANSWER WITH NO DIAGNOSTIC. Compute "
                "L_diff = sqrt(K * t_end / (density * cp_physical)) with "
                "cp_physical = cp_normalized * R / M, and compare it to "
                "the element size h. "
                "WRONG: L_diff / h < 1. The run terminates NORMALLY, "
                "reports every time step completed and exits 0, and "
                "returns a temperature field that undershoots the "
                "LOWEST prescribed temperature. Magnitude, with the "
                "shipped <Initial> temperature (equal to the cold "
                "boundary) kept and the load controller held flat, 10 "
                "steps: about 23 below the 300 cold boundary "
                "(minT = 276.78291 at ratio 0.1). "
                "RIGHT: raise t_end, or refine until L_diff / h > 3, or "
                "use <analysis>STEADY-STATE</analysis> if only the "
                "steady field is wanted. "
                "Signal: NONE from the solver — that is what makes this "
                "the dangerous case. Detect it yourself: log the field "
                "with <Output><logfile><node_data data=\"T\" delim=\",\" "
                "file=\"T.csv\"/></logfile></Output>, then check that no "
                "node falls below the minimum prescribed temperature and "
                "that the profile is monotone along the conduction axis. "
                "FLATTEN THE LOAD CONTROLLER BEFORE RUNNING THIS STUDY, "
                "or you will measure the boundary condition instead of "
                "the artefact. The shipped deck ramps the prescribed "
                "temperature 0 -> 1 over t in [0, 1]. At matched "
                "L_diff/h the required t_end scales with h^2, so on a "
                "fine mesh t_end drops BELOW 1 s and the boundary never "
                "reaches its full value. Executed 2026-08-05: at n = 32, "
                "ratio 0.1, t_end = 0.45228 and the ramped run reports "
                "minT = 135.68545 — which is exactly 300 * t_end to five "
                "significant figures, i.e. the unfinished ramp, not a "
                "diffusion artefact. Replace the load curve points with "
                "<pt>0,1</pt><pt>1,1</pt>. "
                "Executed as a two-mesh study (8 and 32 elements along "
                "the bar) at MATCHED L_diff/h, with the ramp removed and "
                "the shipped <Initial> kept: "
                "the artefact tracks the RATIO and nothing else — "
                "minT = 276.78291 on BOTH meshes at ratio 0.1 "
                "(mesh-independent to 5 s.f.), and exactly 300.00000 on "
                "both at ratio 3, i.e. gone. WITHOUT the <Initial> "
                "block the logged T starts at 0, and the same flat-curve "
                "run at ratio 0.1 gives minT = -92.875 (8 elements; "
                "-92.868 on 32), about 93 below even that starting "
                "value, with a profile that alternates sign along the "
                "bar (300.00, -69.66, 16.03, -3.85, 1.87, -5.00, 21.35, "
                "-92.88, 400.00 on 8 elements). So an <Initial> equal "
                "to the cold boundary shrinks the undershoot but does "
                "not remove it. An earlier version of this entry quoted "
                "'hundreds of kelvin, far below absolute zero' and an "
                "alternating sign, and a later one blamed both on the "
                "ramp. The sign alternation is what omitting <Initial> "
                "does, and it persists with the ramp removed; nothing "
                "below absolute zero was shown, because the log reports "
                "the thermo-fluid temperature DOF, which is relative to "
                "the <Globals> reference T. "
                "The underlying claim survives. "
                "(Executed 2026-08-03, confound removed and re-executed "
                "2026-08-05, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Input] The <Globals><Constants> block is REQUIRED: the "
                "`ideal gas` elastic property reads three named "
                "constants out of it — R, T and P. R and T must both be "
                "strictly positive; P may be omitted. "
                "WRONG: no <Globals> block, or <R>0</R>. "
                "RIGHT: <Globals><Constants><R>8.31446</R><T>300</T>"
                "<P>101325</P></Constants></Globals>. "
                "Signal: the deck reads `...SUCCESS!` and only then "
                "fails, with `A positive universal gas constant R must "
                "be defined in Globals section` or `A positive "
                "referential absolute temperature T must be defined in "
                "Globals section` in an ERROR box and exit 1. Omitting "
                "only P is NOT an error: FEBio computes it from "
                "R*T/M*density, prints the WARNING `The referential "
                "absolute pressure P is calculated internally as <value>` "
                "and runs to normal termination. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Syntax] The <ao> and <cp> children of `ideal gas` are "
                "FEFunction1D PROPERTIES, not plain numbers, so each "
                "needs a type= attribute. "
                "WRONG: <cp>3.5</cp>. "
                "RIGHT: <cp type=\"const\"><value>3.5</value></cp>. "
                "Signal: tag \"cp\" (line N) : `invalid value for "
                "attribute \"type\"` and `Reading file ...FAILED!`. The "
                "complete FEFUNCTION1D_ID vocabulary on this build is "
                "point, const, scale, linear ramp, step, math — nothing "
                "else is accepted in this slot. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Syntax] A <NodeSet> inside <Mesh> carries its ids as a "
                "COMMA-SEPARATED TEXT LIST, not as <n id=.../> children. "
                "WRONG: <NodeSet name=\"cold_face\"><n id=\"1\"/>"
                "<n id=\"2\"/><n id=\"3\"/><n id=\"4\"/></NodeSet>. "
                "RIGHT: <NodeSet name=\"cold_face\">1,2,3,4</NodeSet>. "
                "Signal: tag \"NodeSet\" (line N) : invalid value: — "
                "note the EMPTY value after the colon, because the "
                "element's text content is empty — and `Reading file "
                "...FAILED!`. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
        ],
        "verification": (
            "The steady state of heat_3d_bar is linear in x and hex8 is "
            "trilinear, so the exact solution lies IN the "
            "finite-element space: there is no discretization error to "
            "converge away, and any deviation you see comes from the "
            "LINEAR SOLVER. That makes the deck a conduction PATCH "
            "TEST — run it and compare the logged nodal temperatures "
            "against T_cold + (T_hot - T_cold) * x / L — but it makes "
            "the acceptance criterion subtler than 'expect zero'. "
            "thermo-fluid REQUIRES the iterative `bicgstab` on a "
            "USE_MKL=OFF build (see the [Solver] pitfall), and an "
            "iterative solver stops at a RESIDUAL tolerance, which "
            "buys a larger solution error as the system grows. "
            "Executed at 2, 4, 8, 12, 16, 32, 48, 64 and 96 elements. "
            "Max |T - T_analytic| over all nodes: 0 at n = 2, 4, 8; "
            "3.33e-10 at 12; 0 at 16 and 32; 3.17e-8 at 48; 7.30e-8 at "
            "64; and 1.00e-9 at 96. So the deviation is NEITHER "
            "constant NOR monotone in the mesh size — it is solver "
            "noise that is usually below print precision and "
            "occasionally is not. "
            "So the check is NOT that the deviation stays constant, and "
            "it is NOT that the deviation grows. It is: (1) the "
            "deviation stays near solver tolerance, in the 1e-8 range "
            "or below, at EVERY refinement, and (2) it does NOT scale "
            "like h^2. Discretization error would SHRINK by four each "
            "time you double the mesh; solver noise does not track h "
            "at all. An error that shrinks quadratically means "
            "something is wrong with the deck, because for this "
            "problem there should be no discretization error at all. "
            "TWO EARLIER VERSIONS OF THIS ENTRY WERE WRONG, in "
            "opposite directions, and both from measuring too narrow a "
            "range: the first claimed the deviation does not grow with "
            "refinement (measured only to 16); the second claimed it "
            "is exactly zero up to 32 and then grows (n = 12 is "
            "3.33e-10, not zero, and n = 96 falls back to 1.00e-9 — a "
            "73x DROP from n = 64, so 'creeps upward' is not the "
            "mechanism either). Quote the criterion, not the "
            "sequence."),
    },
}


GENERATORS = {
    "heat_3d_bar": _heat_3d_bar,
}
