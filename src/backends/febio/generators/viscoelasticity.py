"""FEBio viscoelasticity generators and knowledge.

FEBio Module type: 'solid' with viscoelastic material wrappers. Two
families:
  - 'uncoupled viscoelastic' — Prony-series deviatoric relaxation on
    an underlying nearly-incompressible elastic material
  - 'viscoelastic'           — coupled volumetric+deviatoric (full)

Common use cases: stress-relaxation tests on soft tissue (cartilage,
ligament, tendon), creep response, frequency-domain mechanical
testing in the time domain.
"""


def _viscoelasticity_3d_stress_relax(params: dict) -> str:
    """Uncoupled viscoelastic stress-relaxation test: hold a step
    displacement on the top face and observe the stress decay.

    Two-term Prony series on an isotropic elastic ground state. Run time long
    enough to capture both relaxation modes."""
    # THE NESTED ELASTIC IS THE SIMPLEST VALID BASE MATERIAL, ON PURPOSE.
    # This demo's job is the viscoelastic WRAPPER structure (type= + g_i/t_i +
    # nested <elastic>); any elastic law nests the same way. MEASURED over two
    # three-seed rounds of a stress-relaxation task: when this slot held a
    # Mooney-Rivlin demo material, the two worst runs copied it into 4 and 3
    # of their decks and died in large-deformation debugging ("negative
    # jacobians"), while every run that succeeded -- in both arms -- nested
    # plain isotropic elasticity (Mooney-Rivlin count 0 across all nine). A
    # demo ingredient gets copied as if it were a recommendation, so the demo
    # carries the least-surprising one.
    #
    # THE WRAPPER FAMILY MUST MATCH THE NESTED ELASTIC'S FAMILY. Measured on
    # FEBio 4.12: type="uncoupled viscoelastic" REFUSES a (coupled) isotropic
    # elastic child -- 'Component "Material1" needs to have property "elastic"
    # defined' -- the property exists but its family does not fit the slot.
    # (An XML comment inside <material> was first blamed for that error and is
    # exonerated: the failure is identical without it.) The COUPLED wrapper
    # type="viscoelastic" accepts the isotropic child and runs to NORMAL
    # TERMINATION; a run on this install solved a full three-level
    # relaxation study with exactly that pairing.
    E = params.get("youngs_modulus", 1000.0)
    v = params.get("poisson_ratio", 0.3)
    g1 = params.get("g1", 0.4)
    t1 = params.get("t1", 0.5)
    g2 = params.get("g2", 0.3)
    t2 = params.get("t2", 5.0)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="solid"/>
  <Control>
    <analysis>DYNAMIC</analysis>
    <time_steps>50</time_steps>
    <step_size>0.2</step_size>
    <solver type="solid">
      <symmetric_stiffness>symmetric</symmetric_stiffness>
    </solver>
  </Control>
  <Material>
    <material id="1" name="Material1" type="viscoelastic">
      <density>1.0</density>
      <g1>{g1}</g1>
      <t1>{t1}</t1>
      <g2>{g2}</g2>
      <t2>{t2}</t2>
      <elastic type="isotropic elastic">
        <density>1.0</density>
        <E>{E}</E>
        <v>{v}</v>
      </elastic>
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
      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>
    </bc>
    <bc name="step_load" type="prescribed displacement" node_set="load_top">
      <dof>z</dof>
      <value lc="1">-0.1</value>
    </bc>
  </Boundary>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>STEP</interpolate><extend>CONSTANT</extend>
      <points><pt>0,0</pt><pt>0.01,1</pt><pt>10,1</pt></points>
    </load_controller>
  </LoadData>
  <Output>
    <plotfile type="febio">
      <var type="displacement"/>
      <var type="stress"/>
      <var type="relative volume"/>
    </plotfile>
    <logfile>
      <element_data data="sz;J" delim="," file="visco_relax.csv"/>
    </logfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "viscoelasticity": {
        "description": (
            "Time-dependent viscoelastic solid mechanics via FEBio's "
            "Prony-series viscoelastic material wrappers. Used for "
            "stress-relaxation tests on cartilage / ligament / "
            "tendon, creep response of soft tissue, and time-domain "
            "frequency-response analyses."
        ),
        "input_format": "FEBio XML v4.0",
        "solver": "Standard solid solver, transient DYNAMIC analysis",
        "materials": {
            "uncoupled viscoelastic": {
                "elastic": "Nested ground-state elastic material "
                           "(typically nearly-incompressible like "
                           "neo-Hookean with v=0.499)",
                "g1, g2, ...": "Prony coefficients, dimensionless. "
                               "They scale the INSTANTANEOUS stiffness "
                               "UP from the ground state and do NOT set "
                               "the long-time plateau — the <elastic> "
                               "child sets that. sum(g_i) >= 1 is legal "
                               "and harmless, and no value is "
                               "range-checked (0, 1, 5 and -0.5 all "
                               "run). See the [Numerical] pitfall; the "
                               "old 'sum must be < 1' rule was "
                               "falsified by execution.",
                "t1, t2, ...": "Relaxation times (matching units of "
                               "the simulation time step)",
            },
            "viscoelastic": {
                "elastic": "Nested ground-state COUPLED elastic "
                           "material (not uncoupled)",
                "g1..gN, t1..tN": "Same as uncoupled",
            },
        },
        "pitfalls": [
            (
                "[Numerical] RINGING — READ THIS BEFORE MEASURING "
                "ANYTHING OUT OF A RELAXATION DECK. A step-applied "
                "stretch under <analysis>DYNAMIC</analysis> with no "
                "damping excites an undamped mode at the step-size "
                "Nyquist frequency, and the Prony deviatoric "
                "relaxation does NOT damp it. The logged stress then "
                "zig-zags between consecutive steps around the "
                "relaxation curve, so the value at t_end carries an "
                "arbitrary phase of that oscillation, not a converged "
                "answer. "
                "EXECUTED on the shipped "
                "viscoelasticity_3d_stress_relax deck (DYNAMIC, "
                "50 x 0.2, <interpolate>STEP</interpolate>): logged "
                "sz runs -146.74, -135.52, -128.66, ... and ends "
                "-96.18, -92.39, -95.89. It keeps its compressive sign "
                "on all 50 steps, but its difference from the STATIC "
                "run alternates sign on 48 of 49 steps with an "
                "amplitude of up to 1.9 that does not decay (1.84 at "
                "t_end). "
                "REFINING THE STEP MAKES IT WORSE, so this is not "
                "step-size convergence error: at 500 x 0.02 the "
                "difference from STATIC reaches 17.9 and is still 5.7 "
                "at t_end. "
                "WRONG: reading the last logged stress of a DYNAMIC "
                "step-loaded deck as the relaxed plateau; concluding "
                "'the stress relaxes' from a termination banner and "
                "one sample. "
                "RIGHT: switch to <analysis>STATIC</analysis> for a "
                "relaxation study, or ramp the load over several "
                "t_i. Executed control: the same deck with STATIC "
                "relaxes monotonically to sz = -94.06. "
                "Signal: none from the solver — exit 0 and "
                "`N O R M A L   T E R M I N A T I O N` in every "
                "case. The stress itself does not change sign on this "
                "deck, so counting sign changes of the stress does NOT "
                "detect this ringing: check that consecutive "
                "differences of the logged stress keep one sign, or "
                "compare against a STATIC run. "
                "(Executed 2026-08-05; re-executed on the current "
                "template 2026-09-30, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Input] `uncoupled viscoelastic` accepts ONLY an "
                "UNCOUPLED elastic child, and a coupled one is "
                "reported as a MISSING property rather than as a "
                "mismatch — so the message points at the wrong "
                "problem. "
                "WRONG: <material id=\"1\" name=\"M1\" "
                "type=\"uncoupled viscoelastic\"><density>1.0</density>"
                "<g1>0.5</g1><t1>1.0</t1>"
                "<elastic type=\"neo-Hookean\"><density>1</density>"
                "<E>1000</E><v>0.3</v></elastic></material> — "
                "`neo-Hookean` and `isotropic elastic` are COUPLED. "
                "RIGHT: <material id=\"1\" name=\"M1\" "
                "type=\"uncoupled viscoelastic\"><density>1.0</density>"
                "<g1>0.5</g1><t1>1.0</t1>"
                "<elastic type=\"Mooney-Rivlin\"><density>1</density>"
                "<c1>1</c1><c2>0</c2><k>1000</k></elastic></material>. "
                "Signal: `Component \"M1\" needs to have property "
                "\"elastic\" defined (line N)` and `Reading file "
                "...FAILED!`, where M1 is the material's own name. "
                "Executed with a coupled `neo-Hookean` child, a "
                "coupled `isotropic elastic` child, and NO child at "
                "all: all three give the byte-identical message, so "
                "the message cannot distinguish a wrong kind of "
                "child from no child at all. If you see it, check the "
                "child's kind before assuming the tag is missing. The "
                "Mooney-Rivlin form above runs to "
                "`N O R M A L   T E R M I N A T I O N`. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] The Prony coefficients g_i scale the "
                "INSTANTANEOUS stiffness UPWARD from the elastic "
                "ground state; they do NOT reduce the long-time "
                "plateau, and sum(g_i) >= 1 is neither forbidden nor "
                "harmful. FEBio's `viscoelastic` wrapper adds "
                "relaxing terms ON TOP of its <elastic> child, so the "
                "long-time response IS the child's response, always. "
                "WRONG: expecting the relaxed plateau to be "
                "(1 - sum(g_i)) * initial_stress, or picking g_i to "
                "set the plateau. "
                "RIGHT: pick the <elastic> child to set the plateau, "
                "and pick g_i to set how far ABOVE that plateau the "
                "instantaneous response sits. "
                "HOW TO MEASURE IT, and the shipped template will NOT "
                "do: ramp fast, then HOLD at constant stretch and run "
                "long compared with t_i, logging "
                "<element_data data=\"sz\"/> — under "
                "<analysis>STATIC</analysis>, not DYNAMIC. Executed as "
                "a g1 sweep of 0.3, 0.7, 0.95, 1.0 and 1.5 in a "
                "quasi-static two-stage deck (fast ramp then hold): "
                "the plateau came out identical to 12 digits at every "
                "g1 (-0.645234543858) and the PEAK rose monotonically "
                "(-0.9515 at g1=0.3 to -1.4402 at g1=1.5), which is "
                "the claim above. "
                "TWO CORRECTIONS TO AN EARLIER VERSION OF THIS ENTRY, "
                "both executed 2026-08-05, because the numbers it "
                "quoted are not reproducible: "
                "(1) It said the plateau is 'bit-identical to a pure "
                "neo-Hookean control deck with the same ground state'. "
                "FEBio's \"neo-Hookean\" is COUPLED and cannot be the "
                "child of `uncoupled viscoelastic` at all — putting it "
                "there gives `Component \"...\" needs to have property "
                "\"elastic\" defined` and exit 1, exactly as the "
                "[Input] pitfall above says. The control has to be a "
                "bare Mooney-Rivlin deck with the same c1/c2/k, and it "
                "matches only once the hold is long: at t_end = 20*t2 "
                "the two agree to 10 digits, at t_end = 200*t2 they "
                "agree bit for bit. "
                "(2) It said the sweep ran 'on two meshes'. The "
                "shipped 1-element deck refined to 2x2x2 does not "
                "complete: `3 negative jacobians detected.`, "
                "`failed to converge at time : 1.4`, 6 of 50 steps, "
                "`E R R O R   T E R M I N A T I O N`. Take the "
                "identity across g1 as established on one mesh. "
                "ALSO: t_end must be MANY t_i, not two. At the shipped "
                "t_end = 10 with t2 = 5 the recorded 'plateau' is "
                "about 4.3% above the true one and still varies with "
                "g1 in the 6th digit. "
                "g1 is NOT range-checked — 0, 1, 5 and even -0.5 all "
                "run to normal termination, so a non-physical value is "
                "entirely silent. "
                "Signal: no ERROR in any variant. But note every run "
                "of this material prints "
                "`K should only be defined at the top-level.` / "
                "`Value will be ignored.` — that WARNING is harmless "
                "here (the wrapper adds the child's bulk modulus into "
                "its own before the child zeroes it, so k is used); do "
                "not chase it. "
                "(Executed 2026-08-03, corrected 2026-08-05, FEBio "
                "4.12.0.86045466d. The claim above still FALSIFIES the "
                "older rule that sum(g_i) >= 1 sends the long-time "
                "response to zero.)"
            ),
            (
                "[Numerical] A relaxation time t_i shorter than the "
                "LOAD RAMP makes the relaxation invisible: it has "
                "already finished before the ramp ends, so the "
                "recorded peak equals the plateau and the material "
                "behaves as if purely elastic. Too LONG a t_i hides "
                "the plateau instead, for the opposite reason — the "
                "run ends before the material has relaxed. Both "
                "produce a clean run. "
                "WRONG: t_i much smaller than the ramp duration, or "
                "much larger than t_end. "
                "RIGHT: t_i comparable to the ramp duration, with "
                "t_end at least several times t_i so the plateau is "
                "reached. "
                "Signal: none — measure the drop from peak to final "
                "stress. Executed on two meshes with a fixed ramp and "
                "t_i swept over six decades: at t_i three decades "
                "below the ramp the peak-to-plateau drop was zero to "
                "printed precision, two decades below it was about a "
                "percent, at t_i comparable to the ramp it reached "
                "tens of percent, and at t_i above t_end the measured "
                "drop fell again because the plateau had not yet been "
                "reached. Both meshes gave the same drop to four "
                "digits, so this is a time-integration property and "
                "not a discretization artefact. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] The relaxation is integrated by a recursive convolution that carries internal state per integration point, so the stress at a given time depends on the whole load history and not only on the current strain. "
                "WRONG: assuming a shorter or coarser run reproduces a longer one, or resuming a run without carrying the internal state across. "
                "RIGHT: run the full history in one job, and if you must restart, use FEBio's own dump/restart mechanism so the state travels with it. "
                "Signal: none from the solver — measure it yourself. Refining the TIME STEP alone does not change the answer, which is the reassuring half. On two meshes, sweeping the step count over a factor of eight at fixed t_end changed the final relaxed stress only in the seventh significant digit while the peak moved in the fourth — so the scheme is converged in time at ordinary step counts, and a discrepancy larger than that between two runs of the same history is NOT step-size error and should be investigated as a state-handling problem. "
                "THE RESTART HALF IS NOW EXECUTED, and the reassuring answer is that the internal state DOES travel: the recursive-convolution state serialises and restores exactly. Method, and note that the mechanism is a COMMAND-LINE flag, not XML — an earlier version of this entry told the reader to look for a <Control> dump setting, and no such tag exists anywhere in FEBio's XML readers. Write the dump by passing -dump=1 to febio4 beside -i deck.feb (levels 0-3, cmdoptions.cpp), which leaves <deck>.dmp beside the deck, then resume with `febio4 -r deck.dmp`. Executed 2026-08-05 on this deck at 400 x 0.025: the uninterrupted reference ends at sz = 40.3374331267, J = 1.04350808204; a run killed by SIGTERM mid-solve and resumed from its dump printed `- R E S T A R T -`, `Restarting from time 2.925.`, completed the full 400 steps and ended at sz = 40.3374331267, J = 1.04350808204 — bit-identical to all twelve printed digits. So a discrepancy between an interrupted-and-resumed run and an uninterrupted one is a real defect, not expected behaviour. "
                "ONE TRAP IN THE RESTART PATH: you cannot EXTEND a finished run by raising <time_steps> in a restart XML. m_tend is deserialised from the dump and only recomputed in FEAnalysis::Activate(), so FEBio prints `Restarting from time ...` and then runs ZERO steps with statistics byte-identical to the original — a clean-looking no-op. "
                "(Time-step half executed 2026-08-03; restart half executed 2026-08-05, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] THE PRONY RECURSION HAS A FIRST-ORDER FORM "
                "AND A SECOND-ORDER FORM THAT DIFFER BY ONE FACTOR, AND "
                "NEITHER ERRORS. Every code integrates the hereditary "
                "integral h(t) = int_0^t exp(-(t-s)/tau) d(sigma_e)/ds ds "
                "by the same exponential recursion, "
                "h^{n+1} = exp(-dt/tau) h^n + W * (sigma_e^{n+1} - "
                "sigma_e^n), and the whole time accuracy of a viscoelastic "
                "run sits in the weight W:\n"
                "  W = 1                  -> FIRST order\n"
                "  W = exp(-dt/(2 tau))   -> SECOND order (Simo-Hughes, "
                "the midpoint form)\n"
                "  W = (1 + exp(-dt/tau))/2 -> SECOND order (trapezoid), "
                "correct but less accurate\n"
                "  W = (1 - exp(-dt/tau))/(dt/tau) -> SECOND order, exact "
                "when the elastic stress varies linearly within the step; "
                "this is the weight FEBio's built-in `viscoelastic` and "
                "`uncoupled viscoelastic` use (FEViscoElasticMaterial.cpp:"
                "188-191), so FEBio uses none of the three above\n"
                "All four converge and none raises anything, so a "
                "first-order recursion looks like a working solve; it just "
                "halves your error instead of quartering it when you halve "
                "the step. If a study asks for second-order accuracy in "
                "time, W = 1 cannot deliver it however fine the mesh is. "
                "Signal: run the SAME model at dt, dt/2, dt/4 and fit the "
                "error slope — 1 means you have the endpoint weight. "
                "Measured 2026-08-30 on sigma_e = sin t, tau = 0.3, "
                "t_end = 1, against the closed-form integral, at "
                "dt = 1/8 ... 1/256: the endpoint weight is FIRST order, "
                "the midpoint weight SECOND order, and the "
                "trapezoid weight is also second order but "
                "carries about 5.6x the midpoint error at the same step. "
                "Check which one your code uses before trusting a "
                "time-convergence study; where the material is a built-in "
                "and the weight is not yours to choose, measure the slope "
                "and report what you measured rather than what the scheme "
                "is named.\n"
                "HOW THIS SITS WITH THE TIME-CONVERGENCE PITFALL "
                "ABOVE, which reports that sweeping the step count over a "
                "factor of eight moved the final relaxed stress only in "
                "the seventh digit: both are true, and the difference is "
                "whether the load is still MOVING. Once a relaxation deck "
                "reaches its plateau the driving increment "
                "sigma_e^{n+1} - sigma_e^n is zero, the history term has "
                "decayed, and the weight multiplies nothing — so every "
                "weight gives the same plateau and the step size looks "
                "irrelevant. Measured on the same kernel with a load held "
                "constant after t = 0.05 and read at t = 5, the three "
                "weights agree to 1.6e-08, 3.4e-09, 8.2e-10 at 40, 160 "
                "and 640 steps. On a load that keeps varying they differ "
                "by 4.8e-02, 1.1e-02, 2.7e-03 at 8, 32 and 128 steps — "
                "about 23% of the value at the coarsest. So a "
                "step-insensitive PLATEAU is not evidence that your time "
                "discretisation is second-order, and it says nothing "
                "about a problem whose load or source varies throughout "
                "the interval."
            ),
        ],
    },
}


GENERATORS = {
    "viscoelasticity_3d_stress_relax": _viscoelasticity_3d_stress_relax,
}
