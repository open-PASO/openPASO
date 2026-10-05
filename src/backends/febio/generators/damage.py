"""FEBio damage generators and knowledge.

FEBio Module type: 'solid' with continuum damage materials. Two main
families:
  - 'elastic damage' (general wrapper around any elastic base) —
    scalar damage variable D evolves with strain history; effective
    stress sigma_eff = (1 - D) * sigma_elastic. Requires three child
    properties: <elastic>, <damage type="CDF ..."> and
    <criterion type="DC ...">.
  - Pre-combined variants: 'damage neo-Hookean', 'damage
    Mooney-Rivlin', 'damage fiber exponential', ...
  - CORRECTED 2026-08-03: the type strings 'damage', 'Simo damage'
    and 'reactive damage' named here previously are NOT registered
    in FEBio 4.12 and are rejected at parse.

Canonical for tissue tearing thresholds, cartilage degradation under
repeated loading, soft-tissue rupture benchmarks, and fatigue cycling
in elastomers.
"""


def _damage_3d_cycle(params: dict) -> str:
    """Cyclic uniaxial loading on a damageable neo-Hookean block.
    Three load-unload-reload cycles at increasing peak strain. The
    damage variable D grows monotonically; the reload stiffness on
    each cycle is reduced by (1-D).
    """
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    eps_max = params.get("max_strain", 0.1)
    Dmax = params.get("Dmax", 0.9)
    alpha = params.get("alpha", 2.0)
    mu = params.get("mu", 0.5)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="solid"/>
  <Control>
    <analysis>DYNAMIC</analysis>
    <time_steps>60</time_steps>
    <step_size>0.05</step_size>
    <solver type="solid">
      <symmetric_stiffness>symmetric</symmetric_stiffness>
    </solver>
  </Control>
  <Material>
    <material id="1" name="Material1" type="elastic damage">
      <density>1.0</density>
      <elastic type="neo-Hookean">
        <density>1.0</density>
        <E>{E}</E>
        <v>{nu}</v>
      </elastic>
      <damage type="CDF Weibull">
        <Dmax>{Dmax}</Dmax>
        <alpha>{alpha}</alpha>
        <mu>{mu}</mu>
      </damage>
      <criterion type="DC strain energy density"/>
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
    <bc name="cycle" type="prescribed displacement" node_set="load_top">
      <dof>z</dof>
      <value lc="1">{eps_max}</value>
    </bc>
  </Boundary>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>LINEAR</interpolate><extend>CONSTANT</extend>
      <points>
        <pt>0,0</pt><pt>0.5,0.5</pt><pt>1.0,0</pt>
        <pt>1.5,0.75</pt><pt>2.0,0</pt>
        <pt>2.5,1.0</pt><pt>3.0,0</pt>
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
      <var type="damage"/>
    </plotfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "damage": {
        "description": (
            "Continuum damage mechanics via FEBio's 'damage' "
            "wrapper. A scalar damage variable D grows "
            "monotonically with strain history; effective stress "
            "is (1-D) * sigma_elastic. Captures progressive "
            "degradation under repeated loading. Used for soft-"
            "tissue tearing, cartilage degradation, fatigue "
            "cycling in elastomers, and rupture thresholds."
        ),
        "input_format": "FEBio XML v4.0",
        "solver": "Standard solid solver with internal damage state variable",
        "materials": {
            "elastic damage": {
                "elastic": "REQUIRED nested elastic material "
                           "(neo-Hookean / HGO / Mooney-Rivlin / ...)",
                "damage": "REQUIRED nested damage CDF. Registered "
                          "types on FEBio 4.12 all start with "
                          "`CDF `: `CDF Simo`, `CDF log-normal`, "
                          "`CDF Weibull`, `CDF step`, `CDF quintic`, "
                          "`CDF gamma`, `CDF user`.",
                "criterion": "REQUIRED nested damage criterion. "
                             "Registered types all start with "
                             "`DC `: `DC Simo`, `DC strain energy "
                             "density`, `DC specific strain energy`, "
                             "`DC von Mises stress`, `DC Drucker "
                             "shear stress`, `DC max shear stress`, "
                             "`DC max normal stress`, `DC max normal "
                             "Lagrange strain`, `DC octahedral shear "
                             "strain`, `DC octahedral natural strain`, "
                             "`DC Drucker-Prager`, `DC Deshpande-Fleck`.",
                "_verified": (
                    "LIVE-VERIFIED 2026-08-03 on FEBio "
                    "4.12.0.86045466d — this exact material runs a "
                    "one-hex8 prescribed-compression deck to "
                    "N O R M A L   T E R M I N A T I O N: "
                    "<material id=\"1\" name=\"M1\" type=\"elastic "
                    "damage\"><density>1</density><elastic "
                    "type=\"neo-Hookean\"><density>1</density>"
                    "<E>1000</E><v>0.3</v></elastic><damage "
                    "type=\"CDF Simo\"><a>0.9</a><b>0.1</b></damage>"
                    "<criterion type=\"DC strain energy density\"/>"
                    "</material>. All three child properties are "
                    "mandatory; omitting one gives `Component "
                    "\"<name>\" needs to have property \"<prop>\" "
                    "defined (line N)`."),
            },
            "_falsified_2026_08_03": (
                "CORRECTION. This block previously listed `damage`, "
                "`Simo damage` and `reactive damage` as material "
                "types. NONE of the three is a registered "
                "FEMATERIAL_ID on FEBio 4.12 — `damage` exists only "
                "as a PLOT variable (FEPLOTDATA_ID) and as a "
                "mesh-adaptor criterion, not as a material. All "
                "three were executed on 2026-08-03 and every one "
                "was rejected with tag \"material\" (line N) : `"
                "invalid value for attribute \"type\"` and "
                "`Reading file ...FAILED!`. The registered "
                "damage-bearing FEMATERIAL_ID factories on this "
                "build include: `elastic damage`, `uncoupled "
                "elastic damage`, `damage neo-Hookean`, `damage "
                "Mooney-Rivlin`, `damage trans iso Mooney-Rivlin`, "
                "`damage fiber power`, `damage fiber exponential`, "
                "`damage fiber exp-linear`, `viscoelastic damage`, "
                "`uncoupled viscoelastic damage`, `reactive "
                "viscoelastic damage`, `reactive plastic damage`. "
                "Enumerate them yourself with `printf "
                "'list\\nquit\\n' | febio4 -nosplash | grep "
                "FEMATERIAL_ID`."),
        },
        "pitfalls": [
            (
                "[Numerical] Damage is IRREVERSIBLE and, once it reaches its cap, SATURATED: a second identical load cycle is softer than the first only while D is still below Dmax, and identical to it afterwards. "
                "WRONG: expecting every further cycle to soften, or reading two identical cycles as proof that damage is not working. "
                "RIGHT: to see cycle-by-cycle softening, keep Dmax high enough and the damage-CDF scale (<mu> on `CDF Weibull`) large enough that D is still climbing at the end of cycle 1. "
                "Signal: none — measure the peak stress of each cycle from <element_data data=\"sz\"/> on a 0 -> A -> 0 -> A history. Executed at Dmax = 0.0, 0.3, 0.6 and 0.9 on a two-equal-cycle history: at the lower caps the two cycles repeat each other to within a fraction of a percent, because D has already saturated at its cap during cycle 1; only at the highest cap is cycle 2 markedly softer. TWO CORRECTIONS TO AN EARLIER VERSION OF THIS ENTRY. It said the peaks are BIT-IDENTICAL at the saturated caps — they are close but not bit-identical, so compare them with a tolerance. And it put the softening/saturated boundary between 0.3 and 0.6; re-executed, 0.6 is on the SATURATED side and only 0.9 softens. Treat the boundary as deck-dependent and measure it on your own deck rather than quoting one. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Input] `elastic damage` needs THREE properties, not one, and each missing one is reported by its own name. There is no material type called \"damage\". "
                "WRONG: <material id=\"1\" name=\"M1\" type=\"damage\">, or an `elastic damage` material with the elastic parameters written flat at the top level. "
                "RIGHT, complete and runnable: <material id=\"1\" name=\"M1\" type=\"elastic damage\"><density>1.0</density><elastic type=\"neo-Hookean\"><density>1</density><E>10</E><v>0.3</v></elastic><damage type=\"CDF Weibull\"><Dmax>0.9</Dmax><alpha>2.0</alpha><mu>0.5</mu></damage><criterion type=\"DC strain energy density\"/></material>. "
                "Signal: for the unregistered type, tag \"material\" (line N) : `invalid value for attribute \"type\"`. For a missing property, `Component \"M1\" needs to have property \"elastic\" defined (line N)` — and the quoted property name changes to \"damage\" or \"criterion\" depending on which one is absent, so the message tells you exactly what to add. For flat parameters, tag \"E\" (line N) : `unrecognized tag`. All five variants executed. Note the criterion names are prefixed DC — `DC strain energy density`, `DC von Mises stress`, `DC max shear stress` and so on — so the bare physics name is rejected. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] The damage cap is spelled <Dmax> on the <damage> CDF property, it is RANGE-CHECKED to the closed interval [0, 1], and Dmax = 1 is legal — it does not stall the solver on its own. What does bite is the softening: a fully damaged region loses its stiffness and its elements can invert. "
                "WRONG: <Dmax>1.2</Dmax> or a negative value. "
                "RIGHT: <damage type=\"CDF Weibull\"><Dmax>0.9</Dmax><alpha>2.0</alpha><mu>0.5</mu></damage>. "
                "Signal: an out-of-range cap gives `Invalid value for parameter:` followed by `.Dmax` — note the EMPTY name before the dot, because the CDF property is unnamed, so do not search for a material name there — then `Model initialization failed` and exit 1. This fires at initialisation, AFTER the deck reads `...SUCCESS!`. Executed at Dmax = 1.0 (accepted, runs), 1.2 and -0.1 (both rejected). "
                "The softening failure is separate and is MESH-DEPENDENT, which is the part worth knowing: the same deck at the same Dmax ran to completion on a coarse mesh and failed on a finer one with `48 negative jacobians detected.` followed by `------- failed to converge at time : <t>`. So a damage model that converges is not thereby validated — refine and re-run before trusting it. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] The damage-rate parameters on `CDF Weibull` are <alpha> (the Weibull shape) and <mu> (the scale) — there is no parameter called beta. Choose them so that D is still climbing over the strain range you care about; if the CDF saturates early, every cycle after the first is identical and the model looks elastic-with-a-constant-knockdown rather than progressively damaging. "
                "WRONG: <beta>2.0</beta> on the damage property, or a <mu> so small that D reaches Dmax within the first few load steps. "
                "RIGHT: <damage type=\"CDF Weibull\"><Dmax>0.9</Dmax><alpha>2.0</alpha><mu>0.5</mu></damage>, then confirm by comparing consecutive cycle peaks. "
                "Signal: none — the saturated case runs perfectly cleanly and its giveaway is that consecutive cycle peaks are EQUAL to printed precision. Executed: with a small scale the cycle-2 peak equalled the cycle-1 peak exactly while still sitting below the undamaged reference, i.e. the knockdown was already fully applied during cycle 1. The registered alternatives in this slot are `CDF Simo`, `CDF log-normal`, `CDF Weibull`, `CDF step`, `CDF quintic`, `CDF gamma` and `CDF user`. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
        ],
    },
}


GENERATORS = {
    "damage_3d_cycle": _damage_3d_cycle,
}
