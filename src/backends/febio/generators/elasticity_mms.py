"""FEBio 3D linear-elasticity MMS (method of manufactured solutions) family.

FEBio Module type: 'solid'. Material: 'isotropic elastic' (small strain).

Manufactured displacement on the cube [0, L]^3 (gi = ki*pi/L):

    u*_x = A * sin(gx*x) * sin(gy*y) * sin(gz*z)
    u*_y = B * cos(gx*x) * cos(gy*y) * cos(gz*z)
    u*_z = C * sin(gx*x) * cos(gy*y) * sin(gz*z)

with A = amplitude*ax*L, B = amplitude*ay*L, C = amplitude*az*L.
The exact body force  f = -div(sigma(u*))  for the isotropic small-strain
law  sigma = lam*tr(eps)*I + 2*mu*eps  was derived with sympy
(closed form cross-checked symbolically; difference identically zero) and
is emitted as FEBio math expressions in a <body_load type="non-const">.
Dirichlet u = u* is prescribed on the whole boundary via per-node
MeshData/NodeData maps referenced by 'prescribed displacement' BCs
(<value type="map">). Nodal displacements are requested through the
<logfile><node_data> text output so an L2-error sweep is scriptable.

Expected convergence: displacement L2 order 2 for hex8 (trilinear)
elements, in the asymptotic regime. The order a given sweep attains is an
OUTPUT of that sweep — measure it, do not assume it; the amplitude-floor
pitfall in KNOWLEDGE explains the one systematic reason a correct deck
can still fall short of 2 on the finest mesh.

FEBio 4.12.0 for this backend is built from source
(github.com/febiosoftware/FEBio, cmake with USE_MKL=OFF / skyline
solver); see KNOWLEDGE pitfalls for the build recipe. Claims that remain
spec-derived are marked as such below.
"""
import math


# --------------------------------------------------------------------------
# defaults — every problem dimension is a placeholder parameter
# --------------------------------------------------------------------------
_DEFAULTS = {
    "n": 4,             # hex divisions per edge (mesh: n^3 elements)
    "L": 1.0,           # cube edge length
    "E": 1000.0,        # Young's modulus
    "nu": 0.3,          # Poisson's ratio
    "rho": 1.0,         # density (FEBio body force is per unit MASS)
    "amplitude": 1.0e-3,  # dimensionless displacement scale (times L)
    "ax": 1.0,          # relative amplitude of u*_x
    "ay": 0.6,          # relative amplitude of u*_y
    "az": 0.8,          # relative amplitude of u*_z
    "kx": 1.0,          # wave number in x (gx = kx*pi/L)
    "ky": 1.0,          # wave number in y
    "kz": 1.0,          # wave number in z
    "steps": 1,         # STATIC load steps (BC ramped by load controller)
    # Body-force sign convention (see KNOWLEDGE pitfalls): FEBio's residual
    # assembles the body-force parameter as  -integral(H*rho*f)  (source:
    # FEBioMech/FEElasticSolidDomain.cpp, BodyForce(), fa = -H*rho*f*J0)
    # while nodal loads assemble positively, i.e. the applied body force per
    # unit volume is  b = -rho*f_input.  With sign = -1 (default) the deck
    # emits  f_input = -(-div sigma)/rho = +div(sigma)/rho, which is correct
    # for that source-derived convention. Set +1 to get the manual-formula
    # convention (rho*a = div sigma + rho*b). Checked live on FEBio 4.12.0:
    # with the default -1 the MMS sweep converges; a wrong sign gives an
    # O(1) error that does not shrink under refinement.
    "body_force_sign": -1.0,
}


def _num(v):
    """Format a float for XML with full round-trip precision."""
    return format(float(v), ".17g")


def validate_parameters(params: dict) -> list[str]:
    """Validate MMS family parameters. Returns a list of error strings."""
    errors: list[str] = []
    p = dict(_DEFAULTS)
    p.update(params or {})

    # numeric coefficients must actually be numeric (bool is not a valid
    # stand-in for a coefficient either)
    numeric_keys = ("L", "E", "nu", "rho", "amplitude", "ax", "ay", "az",
                    "kx", "ky", "kz", "body_force_sign")
    for key in numeric_keys:
        v = p[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or not math.isfinite(float(v)):
            errors.append(f"{key} must be a finite number, got {v!r}")

    n = p["n"]
    if isinstance(n, bool) or not isinstance(n, int):
        errors.append(f"n must be an integer >= 2, got {n!r}")
    elif n < 2:
        errors.append(f"n must be >= 2 for a meaningful MMS mesh, got {n}")

    steps = p["steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        errors.append(f"steps must be a positive integer, got {steps!r}")

    # only range-check quantities whose numeric type already checked out
    def _ok(key):
        return not any(err.startswith(f"{key} ") for err in errors)

    if _ok("L") and p["L"] <= 0.0:
        errors.append(f"L must be > 0, got {p['L']}")
    if _ok("E") and p["E"] <= 0.0:
        errors.append(f"E must be > 0, got {p['E']}")
    if _ok("nu") and not (-1.0 < p["nu"] < 0.5):
        errors.append(f"nu must satisfy -1 < nu < 0.5, got {p['nu']}")
    if _ok("rho") and p["rho"] <= 0.0:
        errors.append(f"rho must be > 0, got {p['rho']}")
    if _ok("body_force_sign") and float(p["body_force_sign"]) not in (1.0, -1.0):
        errors.append(
            f"body_force_sign must be +1 or -1, got {p['body_force_sign']}")
    return errors


def _resolved(params: dict) -> dict:
    p = dict(_DEFAULTS)
    p.update(params or {})
    return p


def _wave_numbers(p: dict) -> tuple[float, float, float]:
    L = float(p["L"])
    return (float(p["kx"]) * math.pi / L,
            float(p["ky"]) * math.pi / L,
            float(p["kz"]) * math.pi / L)


def _amplitudes(p: dict) -> tuple[float, float, float]:
    s = float(p["amplitude"]) * float(p["L"])
    return s * float(p["ax"]), s * float(p["ay"]), s * float(p["az"])


def _lame(p: dict) -> tuple[float, float]:
    E, nu = float(p["E"]), float(p["nu"])
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


def manufactured_displacement_at(params: dict, x: float, y: float,
                                 z: float) -> tuple[float, float, float]:
    """Evaluate the manufactured displacement u* at a point (scalar)."""
    p = _resolved(params)
    gx, gy, gz = _wave_numbers(p)
    A, B, C = _amplitudes(p)
    sx, sy, sz = math.sin(gx * x), math.sin(gy * y), math.sin(gz * z)
    cx, cy, cz = math.cos(gx * x), math.cos(gy * y), math.cos(gz * z)
    return (A * sx * sy * sz, B * cx * cy * cz, C * sx * cy * sz)


def body_force_coefficients(params: dict) -> dict:
    """Closed-form coefficients of f = -div(sigma(u*)) on the trig basis.

    Derived with sympy (see module docstring); each component of f is a
    3-term sum of products of sin/cos:

      f_x = fx_sss*Sx*Sy*Sz + fx_ssc*Sx*Sy*Cz + fx_ccc*Cx*Cy*Cz
      f_y = fy_ccs*Cx*Cy*Sz + fy_ccc*Cx*Cy*Cz + fy_ssc*Sx*Sy*Cz
      f_z = fz_csc*Cx*Sy*Cz + fz_css*Cx*Sy*Sz + fz_scs*Sx*Cy*Sz

    with Sx = sin(gx*x), Cx = cos(gx*x), etc.
    """
    p = _resolved(params)
    gx, gy, gz = _wave_numbers(p)
    A, B, C = _amplitudes(p)
    lam, mu = _lame(p)
    lm = lam + mu
    g2 = gx * gx + gy * gy + gz * gz
    return {
        "gx": gx, "gy": gy, "gz": gz,
        "fx_sss": A * (lm * gx * gx + mu * g2),
        "fx_ssc": -lm * gx * gy * B,
        "fx_ccc": -lm * gx * gz * C,
        "fy_ccs": -lm * gx * gy * A,
        "fy_ccc": B * (lm * gy * gy + mu * g2),
        "fy_ssc": lm * gy * gz * C,
        "fz_csc": -lm * gx * gz * A,
        "fz_css": -lm * gy * gz * B,
        "fz_scs": C * (lm * gz * gz + mu * g2),
    }


def body_force_expressions(params: dict) -> tuple[str, str, str]:
    """FEBio math-expression strings for the body-load x/y/z components.

    FEBio math parameters use UPPERCASE X, Y, Z = material (reference)
    coordinates (FEBio User Manual 4.7, Appendix A.1). The emitted value
    is  body_force_sign * f / rho  because FEBio body loads are forces
    per unit MASS (multiplied by density internally) and the residual
    applies them with a negative sign (see _DEFAULTS['body_force_sign']).
    """
    p = _resolved(params)
    c = body_force_coefficients(p)
    scale = float(p["body_force_sign"]) / float(p["rho"])
    gx, gy, gz = c["gx"], c["gy"], c["gz"]
    Sx, Sy, Sz = (f"sin({_num(gx)}*X)", f"sin({_num(gy)}*Y)",
                  f"sin({_num(gz)}*Z)")
    Cx, Cy, Cz = (f"cos({_num(gx)}*X)", f"cos({_num(gy)}*Y)",
                  f"cos({_num(gz)}*Z)")

    def term(coeff, a, b, d):
        return f"({_num(coeff * scale)})*{a}*{b}*{d}"

    fx = "+".join([term(c["fx_sss"], Sx, Sy, Sz),
                   term(c["fx_ssc"], Sx, Sy, Cz),
                   term(c["fx_ccc"], Cx, Cy, Cz)])
    fy = "+".join([term(c["fy_ccs"], Cx, Cy, Sz),
                   term(c["fy_ccc"], Cx, Cy, Cz),
                   term(c["fy_ssc"], Sx, Sy, Cz)])
    fz = "+".join([term(c["fz_csc"], Cx, Sy, Cz),
                   term(c["fz_css"], Cx, Sy, Sz),
                   term(c["fz_scs"], Sx, Cy, Sz)])
    return fx, fy, fz


def _node_coords(p: dict):
    """Yield (node_id, x, y, z) for the structured (n+1)^3 grid, id 1-based.

    id = 1 + i + (n+1)*(j + (n+1)*k), i fastest (x), then j (y), then k (z).
    """
    n = int(p["n"])
    L = float(p["L"])
    h = L / n
    nid = 0
    for k in range(n + 1):
        for j in range(n + 1):
            for i in range(n + 1):
                nid += 1
                yield nid, i * h, j * h, k * h


def _boundary_node_ids(p: dict) -> list[int]:
    """Node ids on the boundary of the cube, ascending."""
    n = int(p["n"])
    ids = []
    nid = 0
    for k in range(n + 1):
        for j in range(n + 1):
            for i in range(n + 1):
                nid += 1
                if i in (0, n) or j in (0, n) or k in (0, n):
                    ids.append(nid)
    return ids


def _elasticity_mms_3d_cube_hex8(params: dict) -> str:
    """Structured hex8 MMS deck on [0,L]^3 — complete FEBio 4.0 XML."""
    errors = validate_parameters(params or {})
    if errors:
        raise ValueError(
            "Invalid elasticity_mms parameters: " + "; ".join(errors))
    p = _resolved(params)
    n = int(p["n"])
    steps = int(p["steps"])

    # ---- mesh -------------------------------------------------------
    node_lines = []
    coords = {}
    for nid, x, y, z in _node_coords(p):
        coords[nid] = (x, y, z)
        node_lines.append(
            f'      <node id="{nid}">{_num(x)},{_num(y)},{_num(z)}</node>')

    elem_lines = []
    npl = n + 1            # nodes per line
    nps = npl * npl        # nodes per z-slab
    eid = 0
    for k in range(n):
        for j in range(n):
            for i in range(n):
                eid += 1
                n1 = 1 + i + npl * j + nps * k
                n2, n3, n4 = n1 + 1, n1 + 1 + npl, n1 + npl
                n5, n6, n7, n8 = (n1 + nps, n2 + nps, n3 + nps, n4 + nps)
                elem_lines.append(
                    f'      <elem id="{eid}">'
                    f'{n1},{n2},{n3},{n4},{n5},{n6},{n7},{n8}</elem>')

    bnd = _boundary_node_ids(p)
    bnd_csv = ",".join(str(i) for i in bnd)

    # ---- per-node Dirichlet maps u = u* on the boundary -------------
    # lid = 1-based position within the referenced node set (spec-derived).
    map_lines = {"ux": [], "uy": [], "uz": []}
    for lid, nid in enumerate(bnd, start=1):
        ux, uy, uz = manufactured_displacement_at(p, *coords[nid])
        map_lines["ux"].append(f'      <node lid="{lid}">{_num(ux)}</node>')
        map_lines["uy"].append(f'      <node lid="{lid}">{_num(uy)}</node>')
        map_lines["uz"].append(f'      <node lid="{lid}">{_num(uz)}</node>')

    fx, fy, fz = body_force_expressions(p)

    nodes_xml = "\n".join(node_lines)
    elems_xml = "\n".join(elem_lines)
    ux_xml = "\n".join(map_lines["ux"])
    uy_xml = "\n".join(map_lines["uy"])
    uz_xml = "\n".join(map_lines["uz"])

    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <!-- 3D linear-elasticity MMS: u* trig family on [0,{_num(p["L"])}]^3,
       {n}x{n}x{n} hex8 mesh; body force = -div(sigma(u*)) emitted as
       math expressions (X,Y,Z = reference coordinates); Dirichlet u = u*
       on the whole boundary via NodeData maps. Expected displacement
       L2 order: 2. -->
  <Module type="solid"/>
  <Control>
    <analysis>STATIC</analysis>
    <time_steps>{steps}</time_steps>
    <step_size>{_num(1.0 / steps)}</step_size>
    <solver type="solid">
      <symmetric_stiffness>symmetric</symmetric_stiffness>
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
    <material id="1" name="MMSMaterial" type="isotropic elastic">
      <density>{_num(p["rho"])}</density>
      <E>{_num(p["E"])}</E>
      <v>{_num(p["nu"])}</v>
    </material>
  </Material>
  <Mesh>
    <Nodes name="Object1">
{nodes_xml}
    </Nodes>
    <Elements type="hex8" mat="1" name="Part1">
{elems_xml}
    </Elements>
    <NodeSet name="boundary">{bnd_csv}</NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="MMSMaterial"/>
  </MeshDomains>
  <MeshData>
    <NodeData name="ux_map" node_set="boundary" data_type="scalar">
{ux_xml}
    </NodeData>
    <NodeData name="uy_map" node_set="boundary" data_type="scalar">
{uy_xml}
    </NodeData>
    <NodeData name="uz_map" node_set="boundary" data_type="scalar">
{uz_xml}
    </NodeData>
  </MeshData>
  <Boundary>
    <bc name="mms_ux" type="prescribed displacement" node_set="boundary">
      <dof>x</dof>
      <value lc="1" type="map">ux_map</value>
      <relative>0</relative>
    </bc>
    <bc name="mms_uy" type="prescribed displacement" node_set="boundary">
      <dof>y</dof>
      <value lc="1" type="map">uy_map</value>
      <relative>0</relative>
    </bc>
    <bc name="mms_uz" type="prescribed displacement" node_set="boundary">
      <dof>z</dof>
      <value lc="1" type="map">uz_map</value>
      <relative>0</relative>
    </bc>
  </Boundary>
  <Loads>
    <body_load type="non-const">
      <x>{fx}</x>
      <y>{fy}</y>
      <z>{fz}</z>
    </body_load>
  </Loads>
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
    <plotfile type="febio">
      <var type="displacement"/>
      <var type="stress"/>
    </plotfile>
    <logfile>
      <node_data data="ux;uy;uz" delim="," file="mms_node_disp.csv"/>
    </logfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "elasticity_mms": {
        "description": (
            "3D linear-elasticity manufactured-solution (MMS) extension "
            "family on FEBio — structured hex8 cube, exact trig body "
            "force, per-node Dirichlet maps, text displacement output "
            "for L2-error sweeps"),
        "input_format": "FEBio XML (.feb), version 4.0",
        "expected_convergence": (
            "THEORY, not a measurement: hex8 elements are trilinear, so "
            "the displacement L2 error converges at order 2 under "
            "uniform refinement n -> 2n -> 4n. Two conditions must hold "
            "for a sweep to exhibit that order, and both are behaviours "
            "of this code rather than answers to the study: (1) the "
            "amplitude must stay small enough that the O(amplitude) "
            "geometric-nonlinearity error of the finite-strain "
            "'isotropic elastic' law is still below the h^2 "
            "discretization error on the FINEST mesh — otherwise the "
            "measured order decays there while the coarse ratios look "
            "correct (see the [Discretization] pitfall for the "
            "diagnostic and the fix); (2) body_force_sign must match "
            "FEBio's residual convention (see the [Input] BODY-FORCE "
            "SIGN pitfall) — a wrong sign gives an error that does not "
            "shrink at all under refinement. Measure the order "
            "yourself; do not assume a value. The body-force "
            "derivation is verified independently of any run: the "
            "closed form is re-checked with sympy (difference "
            "identically zero) plus a numpy finite-difference "
            "cross-check in tests/test_febio_elasticity_mms.py."),
        "mms_setup": {
            "per_node_dirichlet": (
                "FEBio has no math-expression Dirichlet BC, so u = u* on "
                "the boundary is imposed per node: a MeshData/NodeData "
                "map per displacement component (<NodeData name=... "
                "node_set=... data_type=\"scalar\"> with <node lid=\"i\"> "
                "entries, lid = 1-based position in the referenced node "
                "set) referenced from a 'prescribed displacement' BC via "
                "<value lc=\"1\" type=\"map\">map_name</value>. The "
                "type=\"map\" parameter mechanism is documented in FEBio "
                "User Manual 4.7 Appendix A.2 (mapped parameters); its "
                "use on BC values is live-verified on FEBio 4.12.0: "
                "decks read cleanly and the boundary nodes carry the "
                "exact u* values the map supplied."),
            "body_force_math": (
                "<body_load type=\"non-const\"> takes one math expression "
                "per component (<x>, <y>, <z>) with UPPERCASE X, Y, Z = "
                "material (reference) coordinates and t = time (manual "
                "4.7 sec. 3.13.3.2 + Appendix A.1). Reference coordinates "
                "are exactly right for small-strain MMS. 'non-const' is "
                "marked obsolete-in-3.0 in the FEBio source "
                "(FEBioMechModule.cpp) but is still registered AND still "
                "documented in the 4.7 manual; the 4.x-canonical "
                "alternative is <body_load type=\"body force\"> with a "
                "single vec3 'force' parameter."),
            "text_output": (
                "The .xplt plotfile is binary, so the deck also requests "
                "<logfile><node_data data=\"ux;uy;uz\" delim=\",\" "
                "file=\"mms_node_disp.csv\"/></logfile>. With the node "
                "list omitted, ALL nodes are written in ascending node-id "
                "order each recorded step (live-verified on FEBio 4.12.0: "
                "the CSV carries a '*Step/*Time/*Data' header per block, one "
                "block for step 0 [all zeros] and one per recorded step, "
                "values at ~12 significant digits — parse the LAST "
                "block). Node ids map back to "
                "coordinates via the generator's structured numbering "
                "id = 1 + i + (n+1)*(j + (n+1)*k), so the L2 error "
                "against u* is a pure post-processing step."),
        },
        "pitfalls": [
            (
                "[Input] BODY-FORCE SIGN: the FEBio User Manual 4.7 "
                "(sec. 3.13.3) writes the momentum balance as "
                "rho*a = div(sigma) + rho*b, but the source assembles "
                "the body-force parameter NEGATIVELY into the residual "
                "(FEBioMech/FEElasticSolidDomain.cpp BodyForce(): "
                "fa = -H[a]*density*f*J0, assembled unmodified by "
                "FESolidDomain::LoadVector) while nodal forces assemble "
                "positively (FENodalForce.cpp: val = +f). Net effect: "
                "applied body force per unit volume b = -rho*f_input, "
                "i.e. a positive input value acts in the NEGATIVE "
                "coordinate direction — consistent with FEBio forum "
                "guidance to enter gravity as positive g = 9.8. The "
                "generator therefore emits f_input = +div(sigma(u*))/rho "
                "(parameter body_force_sign = -1). LIVE-VERIFIED on "
                "FEBio 4.12.0: with the default -1 the MMS error "
                "shrinks under uniform refinement; with +1 it does "
                "not. Signal: a wrong sign is a clean run "
                "(normal termination, no warning) whose displacement L2 "
                "error is O(1) relative — the interior solves toward -u* "
                "while the boundary pins +u* — and does NOT shrink under "
                "mesh refinement; flip body_force_sign to +1 in that "
                "case. (Confirmed by source walk.)"
            ),
            (
                "[Input] FEBio body forces are per unit MASS, multiplied "
                "internally by the material <density> — density matters "
                "even in a STATIC analysis whenever a body load is "
                "present. The generator divides -div(sigma) by rho so "
                "the applied force per volume is density-independent, "
                "but the deck's <density> and the rho parameter MUST "
                "stay in sync (they come from the same parameter). "
                "WRONG: editing <density> in the emitted deck by hand "
                "and leaving the body-force expressions alone. "
                "RIGHT: re-run the generator with the new rho, so "
                "<density> and the expressions move together. "
                "Signal: NONE from the solver — this is a silent "
                "wrong answer. The run reads `...SUCCESS!`, completes "
                "its time step and ends in `N O R M A L   T E R M I N "
                "A T I O N`. Detect it in the result instead: the "
                "prescribed boundary nodes are unaffected (they are "
                "Dirichlet), while the INTERIOR displacements move by "
                "roughly the density ratio, and the discrepancy is "
                "O(1) and does NOT shrink under mesh refinement. "
                "Executed by doubling <density> alone on the default "
                "deck at two refinements: on both meshes the interior "
                "displacement magnitudes moved by close to the factor "
                "2, with a spread across nodes rather than a single "
                "uniform offset — the doubled body force is resisted "
                "by a boundary that is still pinned to the original "
                "u*, so the error is not a clean rescaling and cannot "
                "be divided back out. (Executed 2026-08-03, FEBio "
                "4.12.0.86045466d; mechanism from manual 4.7 sec. "
                "3.13.3.)"
            ),
            (
                "[Input] Math expressions in FEBio parameters use "
                "UPPERCASE X, Y, Z for material (reference) coordinates "
                "(manual 4.7 Appendix A.1). Older forum posts show "
                "lowercase x,y,z in 2.x-era 'data' attributes — do not "
                "mix the two conventions; lowercase symbols are not the "
                "reference coordinates in the 4.x math parser. Signal: "
                "LIVE-VERIFIED on FEBio 4.12.0 and WORSE than a parse "
                "error. Inside <body_load type=\"non-const\"> a "
                "lowercase x, y or z is not rejected: it resolves to "
                "the load's own <x>/<y>/<z> component parameter. On "
                "this template every component uses the coordinates, "
                "so lowercase x makes the <x> expression refer to "
                "itself: the deck reads in with "
                "'Reading file ...SUCCESS!', starts the first time "
                "step, reforms the stiffness matrix and then CRASHES "
                "with a segmentation fault (exit code 139, unbounded "
                "recursion while evaluating the body-force residual) "
                "— no diagnostic names the bad symbol. Where the "
                "referenced component is a constant, the run ends "
                "normally with that constant silently used in place "
                "of the coordinate. Treat any early-timestep "
                "segfault in a deck with math expressions as a "
                "symbol-vocabulary suspect first."
            ),
            (
                "[Discretization] FEBio's 'isotropic elastic' material "
                "is a hyperelastic (finite-strain) formulation that "
                "reduces to linear elasticity only for small strains, "
                "and the solid solver is nonlinear. The MMS amplitude "
                "is therefore a dimensionless fraction of L (default "
                "1e-3): the geometric-nonlinearity contamination of the "
                "displacement error is O(amplitude) relative, so keep "
                "amplitude small enough that h^2 discretization error "
                "dominates on the finest mesh of a sweep, or the "
                "measured order degrades below 2. LIVE-VERIFIED both "
                "ways on FEBio 4.12.0: a sweep run at the default "
                "amplitude loses order on its finest refinement, and "
                "re-running the identical sweep with amplitude reduced "
                "by two decades recovers it. Signal: a measured order "
                "that decays on the finest meshes while the coarse-mesh "
                "ratios look fine, together with a relative L2 error of "
                "the same magnitude as the amplitude parameter — the "
                "O(amplitude) nonlinearity floor has been reached; "
                "reduce amplitude and re-run. The floor scales with "
                "amplitude, so this is a knob, not a limit of the code."
            ),
            (
                "[Input] NodeData lid semantics: lid is the 1-based "
                "LOCAL index into the referenced node_set's listing "
                "order, NOT the global node id. The generator emits the "
                "boundary NodeSet in ascending node-id order and the "
                "maps in the same order. Live-verified indirectly: the "
                "generated decks read cleanly on FEBio 4.12.0 and the "
                "sweep converges under refinement, which a permuted lid "
                "mapping would destroy. Signal: lid/"
                "global-id confusion is either a deck-read error (lid "
                "exceeding the node_set size) or a clean run whose "
                "boundary nodes carry permuted u* values — O(1) error "
                "at the boundary, measured order destroyed."
            ),
            (
                "[Integration] No FEBio binary ships via public "
                "artifact hosting: GitHub releases of febiosoftware/"
                "FEBio carry no binary assets and febio.org downloads "
                "sit behind a WordPress account login with reCAPTCHA, "
                "so unattended installs are blocked. RESOLUTION "
                "(live-verified): the source builds cleanly "
                "— git clone github.com/febiosoftware/FEBio; cmake "
                "with -DUSE_MKL=OFF -DUSE_HYPRE=OFF -DUSE_MMG=OFF "
                "-DUSE_LEVMAR=OFF and, because the non-MKL path never "
                "links OpenMP/dl into the shared libs, "
                "-DCMAKE_EXE_LINKER_FLAGS='-fopenmp -ldl' "
                "-DCMAKE_SHARED_LINKER_FLAGS='-fopenmp -ldl'; then "
                "make. Without MKL FEBio falls back to the SKYLINE "
                "direct solver — correct but slow, on the order of ten "
                "minutes for a ~10^5-dof solid solve, so budget wall "
                "time accordingly on refined meshes. Signal: skipping the linker "
                "flags fails at the final febio4 link, with ld — not FEBio — "
                "reporting undefined references to GOMP_parallel and "
                "dlsym out of "
                "libfecore.so and libfebiolib.so; a system HYPRE "
                "without MPI headers fails earlier, in the compiler, "
                "on a missing mpi.h — disable USE_HYPRE. Symlink "
                "the built binary to ~/FEBio/bin/febio4 or set "
                "FEBIO_BINARY so check_availability() finds it."
            ),
        ],
    },
}


GENERATORS = {
    "elasticity_mms_3d_cube_hex8": _elasticity_mms_3d_cube_hex8,
}
