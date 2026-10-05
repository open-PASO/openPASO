"""FEBio biphasic-solute generators and knowledge.

FEBio Module type: 'solute' -- "Biphasic Solute Analysis". A poroelastic
solid whose interstitial fluid carries EXACTLY ONE solute. It is the special
case that sits between 'biphasic' (fluid, no solute) and 'multiphasic'
(several solutes plus fixed charge), and it is a separate registered module
with its own solver, not a configuration of either neighbour.

Derived from the working multiphasic template and verified by execution on
FEBio 4.12.0: NORMAL TERMINATION, 10 converged time steps, results written. The
three errors it took to get there are the knowledge below -- each one is a
thing FEBio refuses that reads as though the deck is malformed.
"""


def _solute_3d_diffusion(params: dict) -> str:
    """Biphasic solid with one neutral solute diffusing through the
    interstitial fluid: prescribed concentration on one face, zero on the
    opposite one, steady state.
    """
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.0)
    perm = params.get("permeability", 1.0e-3)
    diff = params.get("diffusivity", 1.0e-4)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="solute"/>
  <Control>
    <analysis>STEADY-STATE</analysis>
    <time_steps>10</time_steps>
    <step_size>1.0</step_size>
    <solver type="solute">
      <symmetric_stiffness>non-symmetric</symmetric_stiffness>
      <linear_solver type="bicgstab"/>
    </solver>
  </Control>
  <Globals>
    <Constants>
      <T>298</T>
      <R>8.314e-6</R>
      <Fc>9.65e-5</Fc>
    </Constants>
    <Solutes>
      <solute id="1" name="Na">
        <charge_number>0</charge_number>
        <molar_mass>22.99</molar_mass>
        <density>1.0</density>
      </solute>
    </Solutes>
  </Globals>
  <Material>
    <material id="1" name="Material1" type="biphasic-solute">
      <phi0>0.2</phi0>
      <solid type="neo-Hookean">
        <density>1.0</density>
        <E>{E}</E>
        <v>{nu}</v>
      </solid>
      <permeability type="perm-const-iso">
        <perm>{perm}</perm>
      </permeability>
      <osmotic_coefficient type="osm-coef-const">
        <osmcoef>1.0</osmcoef>
      </osmotic_coefficient>
      <solute sol="1">
        <diffusivity type="diff-const-iso">
          <free_diff>{diff}</free_diff>
          <diff>{diff}</diff>
        </diffusivity>
        <solubility type="solub-const">
          <solub>1.0</solub>
        </solubility>
      </solute>
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
    <NodeSet name="bottom">1,2,3,4</NodeSet>
    <NodeSet name="top">5,6,7,8</NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="Material1"/>
  </MeshDomains>
  <Boundary>
    <bc name="fix" type="zero displacement" node_set="bottom">
      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>
    </bc>
    <bc name="c_bot" type="prescribed concentration" node_set="bottom">
      <dof>c1</dof>
      <value lc="1">0.0</value>
    </bc>
    <bc name="c_top" type="prescribed concentration" node_set="top">
      <dof>c1</dof>
      <value lc="1">1.0</value>
    </bc>
    <bc name="drain" type="zero fluid pressure" node_set="top"/>
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
      <var type="effective fluid pressure"/>
      <var type="effective solute concentration"/>
    </plotfile>
  </Output>
</febio_spec>
'''


GENERATORS = {
    "solute_3d_diffusion": _solute_3d_diffusion,
}

KNOWLEDGE = {
  "solute": {
    "description": (
        "FEBio Module type 'solute' (Biphasic Solute Analysis): a poroelastic "
        "solid whose interstitial fluid carries exactly one solute. Distinct "
        "registered module from 'biphasic' and from 'multiphasic'."
    ),
    "pitfalls": [
        "[Input] THE SOLVER TYPE MUST MATCH THE MODULE, and the two names are "
        "not always the same word. Module type='solute' takes "
        "<solver type=\"solute\">; the material is type=\"biphasic-solute\". "
        "Writing the material's name in the solver tag is the natural guess "
        "and FEBio refuses it. Signal: FEBio prints one line of the form "
        "tag <TAG> (line <N>) : invalid value for attribute <ATTR>, with the "
        "offending tag, its line and the attribute name filled in -- so it "
        "tells you WHICH attribute is wrong and never the legal values, and "
        "reads as a malformed deck rather than as one wrong word. (The "
        "template is in FECore/XMLReader.cpp and the surrounding tag and "
        "line come from the reader, which is why the composed line is not "
        "greppable as a single constant.) The registered names live in "
        "FEBioMix.cpp as "
        "REGISTER_FECORE_CLASS(FEBiphasicSoluteSolver, \"solute\"). "
        "(Measured 2026-09-19 on FEBio 4.12.0.)",

        "[Input] A biphasic-solute MATERIAL REQUIRES osmotic_coefficient even "
        "though the solute is neutral and the osmotic term looks optional. "
        "Signal: 'Component \"Material1\" needs to have property "
        "\"osmotic_coefficient\" defined'. FEBio names the missing property "
        "exactly, which is the good case -- act on the name rather than "
        "re-deriving the material. fixed_charge_density, by contrast, belongs "
        "to multiphasic and is NOT accepted here. (Measured 2026-09-19.)",

        "[Physics] CHOOSE THE MODULE BY HOW MANY SOLUTES YOU HAVE, not by "
        "which template you found first. 'solute' is exactly one; "
        "'multiphasic' is one or more plus fixed charge density and is the "
        "one to use for charged hydrated tissue. Running a single-solute "
        "problem under multiphasic works and costs you the charge terms you "
        "did not want; running a multi-solute problem under 'solute' does "
        "not. Signal: a deck with two <solute> blocks under Module "
        "type='solute' is refused at parse; the same deck under "
        "'multiphasic' runs.",
    ],
  },
}
