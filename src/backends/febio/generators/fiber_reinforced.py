"""FEBio fiber-reinforced generators and knowledge.

FEBio Module type: 'solid' with anisotropic fiber-reinforced materials.
Two main families:
  - 'fiber-exp-pow' / 'fiber-pow-linear' — single-family
  - 'Holzapfel-Gasser-Ogden' (HGO) — arterial-wall standard with two
    symmetric fiber families
  - 'transversely isotropic' — one fiber family on top of an
    isotropic matrix

Canonical for arterial wall mechanics, skeletal muscle, ligament,
tendon, anterior cruciate, myocardium, and any biological tissue with
preferred fiber direction.
"""


def _fiber_reinforced_3d_hgo(params: dict) -> str:
    """Holzapfel-Gasser-Ogden double-fiber-family arterial wall under
    uniaxial extension. Two symmetric fiber families at angle ±phi
    from the loading axis. Neo-Hookean isotropic matrix + fiber
    strain-energy term k1 * exp(k2 * E_f^2) - 1.
    """
    E_iso = params.get("E", 100.0)
    k1 = params.get("k1", 1.0)
    k2 = params.get("k2", 5.0)
    kappa = params.get("kappa", 0.1)
    phi_deg = params.get("fiber_angle_deg", 40.0)
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
    <material id="1" name="Material1" type="Holzapfel-Gasser-Ogden">
      <density>1.0</density>
      <c>{E_iso}</c>
      <k1>{k1}</k1>
      <k2>{k2}</k2>
      <kappa>{kappa}</kappa>
      <gamma>{phi_deg}</gamma>
      <k>1000.0</k>
      <mat_axis type="vector">
        <a>1,0,0</a>
        <d>0,1,0</d>
      </mat_axis>
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
    <bc name="pull" type="prescribed displacement" node_set="load_top">
      <dof>z</dof>
      <value lc="1">0.3</value>
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
      <var type="displacement"/>
      <var type="stress"/>
      <var type="fiber vector"/>
    </plotfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "fiber_reinforced": {
        "description": (
            "Anisotropic fiber-reinforced hyperelasticity via FEBio's "
            "HGO, transversely-isotropic, or single-family fiber "
            "materials. Used for arterial walls (Holzapfel-Gasser-"
            "Ogden two-family model), skeletal muscle, ligament, "
            "tendon, anterior cruciate, and myocardium."
        ),
        "input_format": "FEBio XML v4.0",
        "solver": "Standard solid solver (nonlinear) with anisotropic stress",
        "materials": {
            "Holzapfel-Gasser-Ogden": {
                "c": "Neo-Hookean matrix shear modulus",
                "k1": "Fiber exponential stiffness (force scale)",
                "k2": "Fiber exponent (controls strain-stiffening)",
                "kappa": "Dispersion parameter. HARD RANGE [0, 1/3], "
                         "CLOSED at both ends; anything outside is "
                         "rejected at material init with `Invalid "
                         "value for parameter: <matname>.kappa`. "
                         "0 = perfectly aligned, 1/3 = isotropic "
                         "random. Executed at 0, 0.1, 1/3, 0.34, 0.5, "
                         "1.0 and -0.1.",
                "gamma": "Fiber angle (degrees) from local axis 'a', "
                         "in the a-d plane. Two symmetric families at "
                         "+gamma and -gamma.",
                "k": "Bulk modulus (penalises near-incompressibility). "
                     "REQUIRED for a top-level HGO material: the "
                     "default is 0, and a deck without <k> is rejected "
                     "at initialisation with `K must be a positive "
                     "number.`; set k/c between 100 and "
                     "1000 — see the [Numerical] pitfall, both ends "
                     "of the range bite.",
                "mat_axis": "Material orientation frame: <a> = the "
                            "axis gamma is measured from, <d> = the "
                            "second axis fixing the plane the fiber "
                            "families lie in. OPTIONAL to the parser "
                            "and DANGEROUS to omit: the silent "
                            "default is a = (1,0,0), d = (0,1,0). "
                            "Write it as <mat_axis type=\"vector\">"
                            "<a>0,0,1</a><d>1,0,0</d></mat_axis>.",
            },
            "_transversely_isotropic_falsified_2026_08_03": (
                "CORRECTION. `transversely isotropic` was listed "
                "here as a material type. It is NOT registered in "
                "FEBio 4.12 — executed 2026-08-03, a deck using it "
                "is rejected with tag \"material\" (line N) : `"
                "invalid value for attribute \"type\"`. The "
                "registered trans-iso FEMATERIAL_ID factories on "
                "this build are `trans iso Mooney-Rivlin`, `trans "
                "iso Veronda-Westmann`, `trans iso MR-Estrada`, "
                "`neo-Hookean transiso`, `trans-iso hyperelastic`, "
                "`uncoupled trans-iso hyperelastic`, `coupled "
                "trans-iso Mooney-Rivlin`, `coupled trans-iso "
                "Veronda-Westmann`, `2D trans iso Mooney-Rivlin`, "
                "`2D trans iso Veronda-Westmann`. VERIFIED WORKING: "
                "<material id=\"1\" name=\"M1\" type=\"trans iso "
                "Mooney-Rivlin\"><density>1</density><c1>1</c1>"
                "<c2>0</c2><k>1000</k><c3>1</c3><c4>10</c4>"
                "<c5>100</c5><lam_max>1.2</lam_max><fiber "
                "type=\"vector\"><vector>0,0,1</vector></fiber>"
                "</material> runs a one-hex8 deck to "
                "N O R M A L   T E R M I N A T I O N."),
        },
        "pitfalls": [
            (
                "[Input] Omitting mat_axis is SILENTLY equivalent to "
                "the global axes a = (1,0,0), d = (0,1,0) — the fiber "
                "families end up in the global x-y plane no matter "
                "what your geometry is. There is no warning and no "
                "log line. "
                "WRONG: an HGO <material> with no <mat_axis> child, "
                "when the fibers are not meant to lie about global x. "
                "RIGHT: <mat_axis type=\"vector\"><a>0,0,1</a>"
                "<d>1,0,0</d></mat_axis>, with <a> the axis the "
                "gamma angle is measured from and <d> completing the "
                "plane the two fiber families lie in. "
                "Signal: NONE — the run reads `...SUCCESS!`, "
                "completes every step and ends in `N O R M A L   T E "
                "R M I N A T I O N`. Detect it by comparing against "
                "an explicit frame: executed on two meshes (2x2x2 and "
                "4x4x4 hex8, 20% uniaxial stretch), a deck with "
                "mat_axis omitted produced stresses IDENTICAL to the "
                "same deck with a = (1,0,0), d = (0,1,0), which is "
                "how the default was confirmed. Against a frame "
                "aligned with the load the error is large and grows "
                "with fiber stiffness: barely a tenth at k1/c = 0.1, "
                "roughly double at k1/c = 1, several-fold at k1/c = "
                "10, more than an order of magnitude at k1/c = 100. "
                "The omitted case is insensitive to k1 altogether, "
                "because transverse fibers never load — a "
                "stress-strain curve with no fiber stiffening knee is "
                "the tell. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] HGO fibers carry load only in TENSION: "
                "the fiber strain-energy term is gated off once the "
                "fiber stretch drops below 1, so a compressed "
                "specimen returns the matrix-only response and is "
                "insensitive to k1. This is physics, not a bug, but "
                "it looks like the fiber parameters were ignored. "
                "WRONG: calibrating k1 or k2 against compression "
                "data, or reading a soft compressive response as "
                "evidence that mat_axis is wrong. "
                "RIGHT: calibrate the fiber parameters in tension; "
                "use compression only to fix the matrix modulus c. "
                "Signal: no diagnostic — read it out of the result. "
                "Request <element_data data=\"sx;sy;sz;J\"/> under "
                "<Output><logfile>, then compare a +10% and a -10% "
                "stretch of the SAME deck. Executed on two meshes "
                "with fibers aligned to the load: the tension-side "
                "stress magnitude exceeded the compression-side one "
                "by a factor that grew with k1/c, while the "
                "compression-side stress was unchanged between "
                "k1/c = 1 and k1/c = 10 — that invariance is the "
                "proof the fiber term is gated off. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Input] kappa is HARD-LIMITED to the closed interval "
                "[0, 1/3] and a value outside it is REJECTED at "
                "material initialisation, not silently accepted. "
                "kappa = 0 is perfect alignment, kappa = 1/3 is "
                "isotropic dispersion (HGO collapses toward the "
                "isotropic matrix response). "
                "WRONG: <kappa>0.5</kappa>, or any negative value. "
                "RIGHT: <kappa>0.1</kappa>, or any value in "
                "[0, 0.3333]. "
                "Signal: `Invalid value for parameter:` followed by "
                "the material's own name and `.kappa` (so for a "
                "material named Material1 the line reads: Invalid "
                "value for parameter: Material1.kappa), then "
                "`Failed initializing material 1 (name=\"Material1\")` "
                "and `Model initialization failed`, exit 1. Note the "
                "deck READS successfully "
                "first — this is an initialisation-time range check, "
                "not a parse error, so a wrapper watching only for "
                "`Reading file ...FAILED!` will miss it. Executed at "
                "kappa = 0, 0.1, 1/3 (all accepted) and 0.34, 0.5, "
                "1.0, -0.1 (all rejected); the boundary is exactly "
                "1/3 and it is inclusive. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] The bulk modulus k trades volume error "
                "against solvability, and BOTH ends of the range are "
                "real. Too low and the near-incompressible material "
                "visibly changes volume; too high and the element "
                "inverts before the first step converges. "
                "WRONG: k comparable to c (the volume error is "
                "percent-level), or k many decades above c. "
                "RIGHT: k/c around 100 to 1000. "
                "Signal: the two ends look completely different. At "
                "the LOW-k end there is no diagnostic at all — measure "
                "it by adding <element_data data=\"J\"/> to "
                "<Output><logfile> and checking |J - 1|. Executed as a "
                "k/c sweep on two meshes under 20% uniaxial stretch, "
                "the volume error fell by roughly a decade for every "
                "decade of k/c, crossing 1% between k/c = 10 and "
                "k/c = 100 — so the often-quoted 'at least 1000*c' is "
                "conservative by about one decade, and both meshes "
                "agreed, which is what makes this a material property "
                "and not a discretization artefact. At the HIGH-k end "
                "the run fails loudly with `2 negative jacobians "
                "detected.`, then `------- failed to converge at time "
                ": <t>`, `Number of time steps completed .... : 0` and "
                "exit 1 — reproduced at and above k/c = 100000 on both "
                "meshes. Note that a termination banner is still "
                "printed, so check the completed-step count rather "
                "than the banner. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
        ],
    },
}


GENERATORS = {
    "fiber_reinforced_3d_hgo": _fiber_reinforced_3d_hgo,
}
