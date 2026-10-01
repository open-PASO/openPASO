"""FEBio 4 participant for the openPASO `couple` driver.

FEBio has NO scripting API: it is XML-in / (plot|log)-file-out. This module is
therefore a WRAPPER that, each coupling iteration:
  1. reads imports.json,
  2. writes a complete FEBio 4.0 .feb deck with the imported interface data
     baked in, PER POINT:
       Dirichlet  <MeshData><NodeData name="ux_map" node_set="iface"> +
                  <bc type="prescribed displacement"><value type="map">ux_map
       Neumann    <MeshData><SurfaceData name="t_map" data_type="vec3"> +
                  <surface_load type="traction"><traction type="map">t_map
     (both map forms reproduce a non-uniform imported profile per point; `lid`
      is the 1-based index INTO THE SET, not the node or face id),
  3. runs `febio4 -i deck.feb`,
  4. parses the ASCII <logfile> output (node ux + element sx),
  5. writes exports.json.

PHYSICS.  FEBio 4 has no heat module (FEBioHeat was removed upstream and is
only available as a plugin), so the canonical heat problem of SPEC.md cannot
be run.  What is solved instead is its exact linear analogue: a uniaxial-strain
elastic bar.  With uy = uz = 0 enforced on every node the 3D problem collapses
to the 1D equation

    d/dx ( M du/dx ) = 0 ,   M = lambda + 2 mu   (P-wave / oedometric modulus)

which is `-div(k grad T) = 0` with T -> u_x and k -> M.  Dirichlet/Neumann
interface structure is identical:
  * Dirichlet side: imports the partner's `values` (interface u_x), applies
    them as a per-node prescribed x-displacement.
  * Neumann side  : imports the partner's `normal_fluxes` and applies them
    UNCHANGED as an x-traction on the interface surface (SPEC.md sign rule).

SIGN CONVENTION (the elastic image of SPEC.md's q_out = -k dT/dn):
    exported normal_fluxes  q_out = -(sigma . n_own)_x = -sigma_xx * S
with S = +1 if the interface is this subdomain's +x face, -1 if it is its -x
face.  The two sides then export opposite signs and sum(A)+sum(B) ~ 0, and the
Neumann side applying the partner's number unchanged is the correct traction.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
SIDE      = "dirichlet"   # "dirichlet" | "neumann"
PARTNER   = "right"       # name of the partner participant in couple(...)
X0, X1    = 0.0, 0.5      # this subdomain, x-extent
Y0, Y1    = 0.0, 1.0      # cross-section (y)
Z0, Z1    = 0.0, 0.1      # cross-section (z), one element thick
IFACE_AXIS = "x"          # WHICH straight line the interface is: "x" -> the line x = IFACE_X
                          # (the subdomains sit side by side) | "y" -> the line y = IFACE_X
                          # (they are stacked). Everything below follows from it.
IFACE_X   = 0.5           # shared interface (X0/X1 for axis "x", Y0/Y1 for axis "y")
E_MOD     = 870.0         # Young's modulus
NU        = 0.29          # Poisson ratio
U_OUTER   = 0.0           # prescribed u_x on the NON-interface x-face
NX, NY    = 26, 8         # this subdomain's own mesh (NZ is always 1)
U_INIT    = 5.0e-5        # iteration-1 fallback interface displacement
Q_INIT    = 0.0           # iteration-1 fallback interface flux (traction)
FEBIO     = "febio4"      # the FEBio binary path `discover(query='list')` prints
# ─────────────────────────────────────────────────────────────────────────

AX = 0 if IFACE_AXIS == "x" else 1         # the coordinate the interface FIXES
AL = 1 - AX                                # the coordinate that RUNS ALONG it
LO, HI = (X0, X1) if AX == 0 else (Y0, Y1)         # this subdomain, across the interface
ALO, AHI = (Y0, Y1) if AX == 0 else (X0, X1)       # this subdomain, along it
OUTER_X = LO if abs(IFACE_X - HI) < abs(IFACE_X - LO) else HI
S = 1.0 if IFACE_X > OUTER_X else -1.0     # outward normal at interface = S * e_AX
LAM = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
MU = E_MOD / (2.0 * (1.0 + NU))
M_MOD = LAM + 2.0 * MU                     # the effective 1D "conductivity"

DECK = "cpl.feb"
LOG_NODE = "cpl_node.csv"
LOG_ELEM = "cpl_elem.csv"


# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
def _n(v):
    """Full-precision XML number. NEVER use repr()/!r: numpy 2 scalars
    stringify as 'np.float64(0.0)' and FEBio rejects the deck."""
    return format(float(v), ".17g")
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end


# ---------------------------------------------------------------- imports

def _partner_block(imp):
    """The partner's block from imports.json, by the name the driver actually used."""
    if not isinstance(imp, dict) or not imp:
        return None
    if PARTNER in imp:
        return imp[PARTNER] or None
    others = [k for k in imp if isinstance(imp.get(k), dict)]
    if len(others) == 1:     # named differently in couple(...) than here: read it, say so
        import sys as _sys
        print(f"NOTE: PARTNER is {PARTNER!r} but imports.json is keyed {others[0]!r} "
              f"-- reading that block; align PARTNER with the name in your "
              f"couple(...) call.", file=_sys.stderr)
        return imp[others[0]] or None
    if others:
        raise SystemExit(f"imports.json holds blocks named {others} and none is "
                         f"PARTNER={PARTNER!r}: set PARTNER to the partner's name "
                         f"in your couple(...) call.")
    return None


def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text() or "{}")
    except json.JSONDecodeError:
        return None
    return _partner_block(d)


def sample(imp, key, fallback, y):
    """Interpolate the partner's samples onto this participant's y-coordinates."""
    if not imp or not imp.get("coordinates"):
        return np.full(len(y), float(fallback))
    ys = np.array([c[AL] for c in imp["coordinates"]], float)   # the coordinate ALONG the interface
    vs = np.asarray(imp.get(key, []), float).ravel()
    if vs.size != ys.size:
        return np.full(len(y), float(fallback))
    o = np.argsort(ys)
    return np.interp(y, ys[o], vs[o])


# ------------------------------------------------------------------- mesh
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
def build_mesh():
    xs = np.linspace(X0, X1, NX + 1)
    ys = np.linspace(Y0, Y1, NY + 1)
    zs = np.linspace(Z0, Z1, 2)

    def nid(i, j, k):
        return 1 + i + (NX + 1) * j + (NX + 1) * (NY + 1) * k

    nodes = []
    for k in range(2):
        for j in range(NY + 1):
            for i in range(NX + 1):
                nodes.append((nid(i, j, k), xs[i], ys[j], zs[k]))
    nodes.sort()
    elems = []
    e = 1
    for j in range(NY):
        for i in range(NX):
            elems.append((e, [nid(i, j, 0), nid(i + 1, j, 0),
                              nid(i + 1, j + 1, 0), nid(i, j + 1, 0),
                              nid(i, j, 1), nid(i + 1, j, 1),
                              nid(i + 1, j + 1, 1), nid(i, j + 1, 1)]))
            e += 1
    i_if = NX if IFACE_X == X1 else 0
    i_out = 0 if i_if == NX else NX
    # interface nodes, ordered by (y, z) -> exported in ascending y; the two
    # z-layers carry identical values (the problem is z-invariant), so export
    # only the z = Z0 layer, one point per distinct y.
    iface_nodes = [nid(i_if, j, 0) for j in range(NY + 1)]
    iface_y = [float(ys[j]) for j in range(NY + 1)]
    # all interface nodes (both z layers) for the Dirichlet node set
    iface_all = [nid(i_if, j, k) for k in range(2) for j in range(NY + 1)]
    outer_all = [nid(i_out, j, k) for k in range(2) for j in range(NY + 1)]
    # interface quad4 faces
    faces = []
    for j in range(NY):
        faces.append([nid(i_if, j, 0), nid(i_if, j + 1, 0),
                      nid(i_if, j + 1, 1), nid(i_if, j, 1)])
    return nodes, elems, iface_nodes, iface_y, iface_all, outer_all, faces


# -------------------------------------------------------------- deck text
def write_deck(u_if_map, traction_x):
    nodes, elems, iface_nodes, iface_y, iface_all, outer_all, faces = MESH
    L = []
    L.append('<?xml version="1.0" encoding="ISO-8859-1"?>')
    L.append('<febio_spec version="4.0">')
    L.append('  <Module type="solid"/>')
    L.append('  <Control><analysis>STATIC</analysis><time_steps>1</time_steps>'
             '<step_size>1</step_size>'
             '<solver type="solid"><symmetric_stiffness>symmetric'
             '</symmetric_stiffness></solver></Control>')
    L.append('  <Globals><Constants><T>0</T><R>0</R><Fc>0</Fc></Constants></Globals>')
    L.append(f'  <Material><material id="1" name="Mat" type="isotropic elastic">'
             f'<density>1</density><E>{_n(E_MOD)}</E><v>{_n(NU)}</v></material></Material>')
    L.append('  <Mesh>')
    L.append('    <Nodes name="Object1">')
    for (nid_, x, y, z) in nodes:
        L.append(f'      <node id="{nid_}">{_n(x)},{_n(y)},{_n(z)}</node>')
    L.append('    </Nodes>')
    L.append('    <Elements type="hex8" mat="1" name="Part1">')
    for (eid, conn) in elems:
        L.append(f'      <elem id="{eid}">{",".join(str(c) for c in conn)}</elem>')
    L.append('    </Elements>')
    L.append('    <NodeSet name="all_nodes">'
             + ",".join(str(n[0]) for n in nodes) + '</NodeSet>')
    L.append('    <NodeSet name="outer">' + ",".join(str(n) for n in outer_all)
             + '</NodeSet>')
    L.append('    <NodeSet name="iface">' + ",".join(str(n) for n in iface_all)
             + '</NodeSet>')
    L.append('    <NodeSet name="iface_out">'
             + ",".join(str(n) for n in iface_nodes) + '</NodeSet>')
    L.append('    <Surface name="iface_surf">')
    for fi, f in enumerate(faces, 1):
        L.append(f'      <quad4 id="{fi}">{",".join(str(c) for c in f)}</quad4>')
    L.append('    </Surface>')
    L.append('  </Mesh>')
    L.append('  <MeshDomains><SolidDomain name="Part1" mat="Mat"/></MeshDomains>')

    if SIDE == "dirichlet":
        # per-node Dirichlet map: lid is the 1-based index INTO the node set
        L.append('  <MeshData>')
        L.append('    <NodeData name="ux_map" node_set="iface" data_type="scalar">')
        for lid, n in enumerate(iface_all, 1):
            L.append(f'      <node lid="{lid}">{_n(u_if_map[n])}</node>')
        L.append('    </NodeData>')
        L.append('  </MeshData>')
    else:
        # per-FACE traction map: lid is the 1-based index INTO the facet set
        L.append('  <MeshData>')
        L.append('    <SurfaceData name="t_map" surface="iface_surf" data_type="vec3">')
        for lid, tv in enumerate(traction_x, 1):
            L.append(f'      <face lid="{lid}">{_n(tv)},0,0</face>')
        L.append('    </SurfaceData>')
        L.append('  </MeshData>')

    L.append('  <Boundary>')
    L.append('    <bc name="planar" type="zero displacement" node_set="all_nodes">'
             '<x_dof>0</x_dof><y_dof>1</y_dof><z_dof>1</z_dof></bc>')
    L.append('    <bc name="outer_ux" type="prescribed displacement" node_set="outer">'
             f'<dof>x</dof><value lc="1">{_n(U_OUTER)}</value><relative>0</relative></bc>')
    if SIDE == "dirichlet":
        L.append('    <bc name="iface_ux" type="prescribed displacement" node_set="iface">'
                 '<dof>x</dof><value lc="1" type="map">ux_map</value>'
                 '<relative>0</relative></bc>')
    L.append('  </Boundary>')

    if SIDE == "neumann":
        L.append('  <Loads>')
        L.append('    <surface_load name="iface_t" type="traction" surface="iface_surf">')
        L.append('      <traction type="map">t_map</traction>')
        L.append('      <scale lc="1">1.0</scale>')
        L.append('      <linear>0</linear>')
        L.append('    </surface_load>')
        L.append('  </Loads>')

    L.append('  <LoadData><load_controller id="1" type="loadcurve">'
             '<interpolate>LINEAR</interpolate><extend>CONSTANT</extend>'
             '<points><pt>0,0</pt><pt>1,1</pt></points>'
             '</load_controller></LoadData>')
    L.append('  <Output><logfile>')
    L.append(f'    <node_data data="ux" delim="," file="{LOG_NODE}" node_set="iface_out"/>')
    L.append(f'    <element_data data="sx" delim="," file="{LOG_ELEM}"/>')
    L.append('  </logfile></Output>')
    L.append('</febio_spec>')
    Path(DECK).write_text("\n".join(L) + "\n")


# ------------------------------------------------------------ log parsing
def parse_log(path):
    """FEBio ASCII logfile: '*Step ...' / '*Data =' blocks, then 'id,val'.
    Returns {id: value} for the LAST step in the file."""
    out = {}
    txt = Path(path).read_text()
    blocks = txt.split("*Step")
    body = blocks[-1] if len(blocks) > 1 else txt
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("*"):
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            out[int(float(parts[0]))] = float(parts[1])
        except ValueError:
            continue
    return out
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end


# -------------------------------------------------------------------- run
MESH = build_mesh()
imp = read_imports()
nodes, elems, iface_nodes, iface_y, iface_all, outer_all, faces = MESH
y_if = np.asarray(iface_y, float)

u_map = {}
traction_x = np.zeros(0)
if SIDE == "dirichlet":
    u_line = sample(imp, "values", U_INIT, y_if)
    for lid, n in enumerate(iface_all):
        u_map[n] = float(u_line[lid % (NY + 1)])
else:
    q_line = sample(imp, "normal_fluxes", Q_INIT, y_if)
    # PER-FACE traction: the imported nodal fluxes are averaged onto each
    # interface quad4 (FEBio's <traction> takes a vec3 SurfaceData map with
    # one vector per face). SPEC.md's rule: the partner's number is applied
    # UNCHANGED as the x-traction.
    traction_x = 0.5 * (q_line[:-1] + q_line[1:])

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
write_deck(u_map, traction_x)
for f in (LOG_NODE, LOG_ELEM):
    Path(f).unlink(missing_ok=True)

r = subprocess.run([FEBIO, "-i", DECK], capture_output=True, text=True, timeout=1800)
tail = (r.stdout or "")[-1500:]
if "N O R M A L   T E R M I N A T I O N" not in (r.stdout or ""):
    sys.stderr.write(f"FEBio did not terminate normally (rc={r.returncode})\n{tail}\n")
    sys.exit(1)

ux = parse_log(LOG_NODE)
sx = parse_log(LOG_ELEM)
if not ux or not sx:
    sys.stderr.write(f"empty FEBio logfile output: n_ux={len(ux)} n_sx={len(sx)}\n")
    sys.exit(2)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

u_out = np.array([ux[n] for n in iface_nodes], float)
sig_xx = float(np.mean(list(sx.values())))
q_out = np.full(len(iface_nodes), -sig_xx * S)

# ── EXPORT SELF-CHECK ─ keep this block. It stops the three exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
#    a flux of ~0 against a nonzero partner); and a flux that is the partner's
#    array negated instead of a recovery from THIS side's own system.
_chk_vals = np.asarray(u_out, float).ravel()
_chk_flux = np.asarray(q_out, float).ravel()
if not (np.isfinite(_chk_vals).all() and np.isfinite(_chk_flux).all()):
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface values or fluxes; "
                     "the solve did not produce a usable field, so nothing was "
                     "exported")
_chk_imp = (json.loads(Path("imports.json").read_text() or "{}")
            if Path("imports.json").is_file() else {})
_chk_qin = (np.concatenate([np.asarray(_d.get("normal_fluxes") or [], float).ravel()
                            for _d in _chk_imp.values()])
            if _chk_imp else np.zeros(0))
if SIDE == "neumann" and _chk_qin.size and np.abs(_chk_qin).max() > 0 \
        and np.abs(_chk_flux).max() < 1e-9 * np.abs(_chk_qin).max():
    raise SystemExit("EXPORT SELF-CHECK: the recovered interface flux is ~0 "
                     "against a nonzero imported flux: the imported load never "
                     "entered the assembled system (the facet term / boundary "
                     "condition that integrates it is missing). Fix the "
                     "application; do not couple on")
# (Dirichlet role only: a Neumann side's consistent recovery of a CONSTANT
#  applied flux can legitimately reproduce it to the last bit.)
if SIDE == "dirichlet" and _chk_qin.shape == _chk_flux.shape and _chk_flux.size \
        and np.array_equal(_chk_flux, -_chk_qin):
    raise SystemExit("EXPORT SELF-CHECK: the exported flux is the partner's "
                     "array negated, bit for bit: a copy, not a recovery from "
                     "this side's own assembled system")

Path("exports.json").write_text(json.dumps({
    "field_name": "displacement_x",
    "n_points": int(len(iface_nodes)),
    "coordinates": [([float(IFACE_X), float(yy), float(Z0)] if AX == 0 else
                     [float(yy), float(IFACE_X), float(Z0)]) for yy in y_if],
    "values": [float(v) for v in u_out],
    "normal_fluxes": [float(v) for v in q_out],
}, indent=2))
print(f"FEBio {SIDE}: u_if_mean={u_out.mean():.8e} sigma_xx={sig_xx:.8e} "
      f"q_out={q_out[0]:.8e} M={M_MOD:.4f}")
