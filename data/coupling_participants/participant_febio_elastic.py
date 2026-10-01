"""FEBio 4 VECTOR participant for the openPASO `couple` driver.

Plane-strain linear elasticity  -div(sigma(u)) = b  on ONE rectangular
subdomain of a domain split by a straight interface at x = IFACE_X (or
y = IFACE_X when IFACE_AXIS is "y"), the body force b given by B_SRC in the
edit block. The exchanged interface state is a VECTOR on BOTH channels:

    values        = displacement       u = (u_x, u_y)   at the interface nodes
    normal_fluxes = interface traction export            (SIGN CONVENTION below)

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

FEBio HAS NO SCRIPTING API: it is XML-in / logfile-out. Each coupling
iteration this wrapper (1) reads imports.json, (2) writes a complete FEBio 4.0
.feb deck with the imported interface data baked in PER NODE, (3) runs
`febio4 -i deck.feb`, (4) parses the ASCII <logfile>, (5) writes exports.json.

PLANE STRAIN IN A 3-D CODE. FEBio solves 3-D solids only, so the subdomain is
meshed as ONE layer of hex8 elements of thickness ZTHICK with u_z = 0 on every
node: plane strain exactly. The two z-layers of nodes carry identical
(u_x, u_y), so the exported points are the distinct interface nodes, with 2-D
coordinates, the list the other *_elastic contracts export.

SIGN CONVENTION. `normal_fluxes` is exported as

    q_out = -(sigma . n_own)                       n_own = S * e_AX

(the convention of every shipped contract, q_out = -k dT/dn_own for heat). The
two sides' exports cancel componentwise, and the NEUMANN side applies the
partner's numbers UNCHANGED: the natural boundary term of the weak form is
+(sigma . n_own) . v = +q_out_partner . v. Exporting sigma . n_own instead flips
the load the Neumann side applies; the iteration still converges, to the wrong
answer.

THE TRACTION EXPORT IS THE CONSISTENT (REACTION) TRACTION, PER NODE: not a
domain average and not an element-stress projection, which are first order at
best on the boundary. From a(u,v) - (b,v) = -int_Gamma q_out . v ds it follows
for every basis function phi_i on the interface that

    q_i = -R_i / w_i,     R = A u_h - b (UNCONSTRAINED),   w_i = int_Gamma phi_i ds.

On the Dirichlet side FEBio's node log "Rx", "Ry" is that residual at the
PRESCRIBED dofs, except that a <nodal_load> never reaches it there: the body
force's consistent nodal load is subtracted by hand, the same array the deck
carries. The two z-layers of one interface node are summed in both numerator
and denominator. The two interface CORNERS lie on the outer boundary too, so
they take their nearest interior neighbour's value. The Neumann side cannot use
the reaction (FEBio reports Rx = Ry = 0 at free dofs): it exports -Fc/w, the
consistent nodal force it built from the partner's traction and wrote into the
deck, and prints the element-stress traction beside it as an independent
second opinion.

LOADS ARE CONSISTENT NODAL FORCES. The partner's traction (Neumann side) enters
as F_i = int_Gamma t_h . phi_i ds (the quad4 surface mass matrix of the
interface faces), and B_SRC as F_i = int_Omega b . phi_i dV (3x3x3 Gauss on each
hex8), both applied as <nodal_load type="nodal_force"> through vec3 NodeData
maps. b is a force per unit VOLUME; ZTHICK cancels.

WHAT IS APPROXIMATED.
  * FEBio has no small-strain material: `isotropic elastic` is
    St.Venant-Kirchhoff, linear elasticity only as |grad u| -> 0. Large
    displacements make this a finite-strain side.
  * The recovered traction has a floor near 1e-7 relative (round-off below it,
    the finite-strain term above): with FEBio as the DIRICHLET side, do not ask
    `couple` for a residual tolerance below about 1e-7.
  * hex8 is trilinear (Q1 in the plane), not the P1 of other contracts; both
    are O(h^2).
  * LINSOLVE is "skyline": a build without pardiso fails on FEBio's default.

RELAXATION IS NOT PER COMPONENT: the driver applies one theta to the whole
interface state; subdomains of unequal length or Poisson ratio want
accelerator="aitken".
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side.
SIDE      = "dirichlet"   # "dirichlet" (import u, export traction) | "neumann"
PARTNER   = "right"       # the partner's `name` in your couple(...) call
X0, X1    = 0.0, 0.5      # this subdomain's x-extent
Y0, Y1    = 0.0, 1.0      # this subdomain's y-extent
ZTHICK    = 0.07          # slab thickness; ANY positive value, plane strain
IFACE_AXIS = "x"          # WHICH straight line the interface is: "x" -> the line x = IFACE_X
                          # (the subdomains sit side by side) | "y" -> the line y = IFACE_X
                          # (they are stacked). Everything below follows from it.
IFACE_X   = 0.5           # the shared interface: X0/X1 for axis "x", Y0/Y1 for axis "y"
E_MOD     = 870.0         # Young's modulus
NU        = 0.29          # Poisson ratio (PLANE STRAIN)
# Prescribed displacement on this subdomain's WHOLE non-interface boundary
# (its outer face and the two faces the interface ends on), a quadratic in (x, y):
#     u_x = UDX[0] + UDX[1]*x + UDX[2]*y + UDX[3]*x*x + UDX[4]*x*y + UDX[5]*y*y
#     u_y = UDY[0] + UDY[1]*x + UDY[2]*y + UDY[3]*x*x + UDY[4]*x*y + UDY[5]*y*y
# The two subdomains must agree at the two interface corners.
UDX = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
UDY = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def B_SRC(x, y):
    """Body force per unit volume, (b_x, b_y), as a function of position.

    Returns zero as shipped, which is a PLACEHOLDER like every number above
    and is almost never what your problem wants: with displacement prescribed
    on the whole outer boundary and no body force, the only solution is
    u = 0 everywhere, and the coupling will converge beautifully to it.

    If your problem states a body force, or gives you a manufactured solution
    whose source term you derived, put it here. `x` and `y` are NumPy arrays,
    so build the answer with NumPy and return two arrays of the same shape:

        return (2.0 * MU * np.pi**2 * np.sin(np.pi * x) * np.cos(np.pi * y),
                np.zeros_like(x))

    Write it exactly as you would for the FEniCSx or scikit-fem sibling, in the
    same units. It does NOT become a <body_load>: this file integrates B_SRC
    against the element shape functions and applies the consistent nodal force
    vector (see the header). (A <body_load type="body force"> with a math
    expression also runs on this build, per unit mass; this contract does not
    use it.)
    """
    return np.zeros_like(x), np.zeros_like(y)
NX, NY    = 26, 26        # this subdomain's OWN mesh; need not match the partner
UI_X, UI_Y = 0.0, 0.0     # iteration-1 fallback interface displacement
TI_X, TI_Y = 0.0, 0.0     # iteration-1 fallback interface traction export
FEBIO     = "febio4"      # the FEBio binary path `discover(query='list')` prints
LINSOLVE  = "skyline"     # NOT "pardiso": many builds ship without it (see above)
# ─────────────────────────────────────────────────────────────────────────

LAM = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))   # plane strain
MU = E_MOD / (2.0 * (1.0 + NU))

AX = 0 if IFACE_AXIS == "x" else 1         # the coordinate the interface FIXES
AL = 1 - AX                                # the coordinate that RUNS ALONG it
LO, HI = (X0, X1) if AX == 0 else (Y0, Y1)         # this subdomain, across the interface
ALO, AHI = (Y0, Y1) if AX == 0 else (X0, X1)       # this subdomain, along it
ON_RIGHT = abs(IFACE_X - HI) < abs(IFACE_X - LO)   # interface at this side's MAX of that axis?
OUTER_X = LO if ON_RIGHT else HI           # the opposite face, on the same axis
S = 1.0 if ON_RIGHT else -1.0              # outward normal at interface = S * e_AX
TOL = 1e-9 * max(X1 - X0, Y1 - Y0)

DECK = "cpl.feb"
LOG_U = "cpl_u.csv"        # ux, uy at the interface nodes
LOG_F = "nodal_out.csv"   # x;y;z;ux;uy;uz over the WHOLE mesh -- the
                          # per-level field dump below reads it
LOG_R = "cpl_r.csv"        # Rx, Ry at the interface nodes (Dirichlet side only)
LOG_E = "cpl_e.csv"        # sx, sxy per element (Neumann side only)


# ── THE PER-LEVEL RULE (served). ./config.json {"level": k, "nx": .., "ny": ..}
#    sets the mesh and names the level; the dumps below carry it in their NAME.
LEVEL = 1
if Path("config.json").is_file() or os.environ.get("OPENPASO_CONFIG_JSON"):
    try:
        _cfg = json.loads(Path("config.json").read_text() or "{}") if Path("config.json").is_file() else {}
        _cfg.update(json.loads(os.environ.get("OPENPASO_CONFIG_JSON") or "{}"))
        LEVEL = int(_cfg.get("level", LEVEL))
        NX = int(_cfg.get("nx", NX))
        NY = int(_cfg.get("ny", NY))
    except (ValueError, TypeError, json.JSONDecodeError):
        pass

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
    """imports.json is {partner_name: InterfaceData}; `{}` on iteration 1,
    so the caller must fall back to an initial guess."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return _partner_block(json.loads(p.read_text() or "{}"))
    except json.JSONDecodeError:
        return None


def sample(imp, key, fallback, y):
    """Map the partner's VECTOR samples onto THIS participant's interface
    points, COMPONENT BY COMPONENT.

    The driver does no interpolation — non-matching interface meshes are
    handled here, and for a vector field that has to be done per component. One
    np.interp over a flattened (N, 2) array interleaves the two components: the
    result still has the right length, the coupling still converges, and every
    number is wrong.

    Returns (len(y), ncomp)."""
    fb = np.asarray(fallback, float).ravel()
    if not imp or not imp.get("coordinates"):
        return np.tile(fb, (len(y), 1))
    ys = np.array([c[AL] for c in imp["coordinates"]], float)   # the coordinate ALONG the interface
    vs = np.asarray(imp.get(key) or [], float)
    if vs.ndim == 1:
        vs = vs.reshape(-1, 1)
    if vs.shape[0] != ys.size or vs.shape[1] != fb.size:
        return np.tile(fb, (len(y), 1))
    o = np.argsort(ys)
    return np.column_stack([np.interp(y, ys[o], vs[o, c])
                            for c in range(vs.shape[1])])


# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
def u_dirichlet(x, y):
    """The prescribed displacement on the non-interface boundary."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return (UDX[0] + UDX[1] * x + UDX[2] * y
            + UDX[3] * x * x + UDX[4] * x * y + UDX[5] * y * y,
            UDY[0] + UDY[1] * x + UDY[2] * y
            + UDY[3] * x * x + UDY[4] * x * y + UDY[5] * y * y)


# ------------------------------------------------------------------- mesh
class Mesh:
    """One layer of hex8 over [X0,X1] x [Y0,Y1] x [0,ZTHICK]. 1-based ids."""

    def __init__(self):
        self.xs = np.linspace(X0, X1, NX + 1)
        self.ys = np.linspace(Y0, Y1, NY + 1)
        self.zs = np.array([0.0, ZTHICK])

        # NODE NUMBERING IS A PERFORMANCE DECISION, NOT A COSMETIC ONE.
        # FEBio numbers equations in node order and the fallback direct solver
        # is `skyline`, whose cost is O(n * bandwidth^2). Numbering the slab
        # layer-by-layer (all of z=0, then all of z=ZTHICK) puts the two nodes
        # of every through-thickness edge (NX+1)(NY+1) apart, which on a 64x64
        # mesh is a bandwidth of ~12700 dofs and a solve that does not finish.
        # Running the THICKNESS index fastest, then the shorter in-plane
        # direction, keeps the bandwidth at ~6*min(NX,NY) dofs. Measured on
        # 64x64: minutes -> ~2 s per solve, same answer to the last digit.
        fast_i = NX <= NY
        if fast_i:
            def nid(i, j, k):
                return 1 + k + 2 * i + 2 * (NX + 1) * j
        else:
            def nid(i, j, k):
                return 1 + k + 2 * j + 2 * (NY + 1) * i

        self.nid = nid
        self.nodes = [(nid(i, j, k), self.xs[i], self.ys[j], self.zs[k])
                      for k in range(2) for j in range(NY + 1)
                      for i in range(NX + 1)]
        self.nodes.sort()
        self.xyz = np.zeros((len(self.nodes) + 1, 3))     # 1-based lookup
        for (n, x, y, z) in self.nodes:
            self.xyz[n] = (x, y, z)
        self.elems = []
        e = 1
        for j in range(NY):
            for i in range(NX):
                self.elems.append((e, [nid(i, j, 0), nid(i + 1, j, 0),
                                       nid(i + 1, j + 1, 0), nid(i, j + 1, 0),
                                       nid(i, j, 1), nid(i + 1, j, 1),
                                       nid(i + 1, j + 1, 1), nid(i, j + 1, 1)]))
                e += 1
        # the column of elements that touches the interface, in ascending y
        i_col = NX - 1 if ON_RIGHT else 0
        self.iface_elems = [1 + i_col + NX * j for j in range(NY)]

        i_if = NX if ON_RIGHT else 0
        # THE EXPORTED POINTS: the distinct interface y values, ascending. Same
        # points in the same order every iteration — the driver relaxes export
        # vectors entry by entry, so any reordering is silently wrong.
        self.y_if = self.ys.copy()
        # per exported point, the (bottom, top) node ids of the two z-layers
        self.iface_pair = [(nid(i_if, j, 0), nid(i_if, j, 1))
                           for j in range(NY + 1)]
        self.iface_all = [n for pair in self.iface_pair for n in pair]
        # THE TWO INTERFACE CORNERS BELONG TO THE OUTER BOUNDARY, ON BOTH SIDES.
        # (IFACE_X, Y0) and (IFACE_X, Y1) sit on a y-face, which carries a
        # prescribed displacement in the un-split problem, so they stay
        # Dirichlet in BOTH subproblems. Handing them to the interface instead
        # leaves them unconstrained on the Neumann side: that subproblem is
        # still well posed, still converges, and lands a few percent off —
        # measured in the siblings, 4.7% in the interface displacement and 28%
        # in the interface traction on a coupling whose residual reached 1e-10
        # and whose flux balanced. They are still EXPORTED; they are just not
        # interface-imposed. (They also must not appear in two prescribed-
        # displacement BCs at once, which FEBio would not resolve for you.)
        self.interior_j = [j for j in range(NY + 1)
                           if abs(self.ys[j] - Y0) > TOL
                           and abs(self.ys[j] - Y1) > TOL]
        self.iface_free = [n for j in self.interior_j
                           for n in self.iface_pair[j]]
        # the WHOLE non-interface boundary: outer x-face + both y-faces
        outer = set()
        i_out = 0 if ON_RIGHT else NX
        for k in range(2):
            for j in range(NY + 1):
                outer.add(nid(i_out, j, k))
            for i in range(NX + 1):
                outer.add(nid(i, 0, k))
                outer.add(nid(i, NY, k))
        self.outer = sorted(outer)
        # interface quad4 faces (used for the nodal weights and, on the Neumann
        # side, for the consistent nodal forces)
        self.faces = [[nid(i_if, j, 0), nid(i_if, j + 1, 0),
                       nid(i_if, j + 1, 1), nid(i_if, j, 1)]
                      for j in range(NY)]

    # ---- interface surface integrals -----------------------------------
    def _face_gauss(self, face):
        """2x2 Gauss on the bilinear quad4: (shape values, weight*detJ) per
        point. Exact for the Q1 mass matrix of a planar quad."""
        p = self.xyz[face]                          # (4, 3)
        g = 1.0 / np.sqrt(3.0)
        out = []
        for xi in (-g, g):
            for eta in (-g, g):
                N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                                     (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
                dNx = 0.25 * np.array([-(1 - eta), (1 - eta),
                                       (1 + eta), -(1 + eta)])
                dNe = 0.25 * np.array([-(1 - xi), -(1 + xi),
                                       (1 + xi), (1 - xi)])
                jac = np.cross(dNx @ p, dNe @ p)
                out.append((N, float(np.linalg.norm(jac))))
        return out

    def iface_weights(self):
        """w[node] = int_Gamma phi_node ds over the interface faces."""
        w = np.zeros(len(self.nodes) + 1)
        for f in self.faces:
            for N, dj in self._face_gauss(f):
                w[f] += N * dj
        return w

    def iface_mass_apply(self, t_node):
        """F_i = int_Gamma t_h . phi_i ds with t_h the BILINEAR interpolant of
        the nodal traction samples: the consistent nodal force vector, i.e.
        exactly `inner(g, v) * ds` of the FEniCSx/scikit-fem siblings.

        t_node maps node id -> (tx, ty). Returns node id -> (Fx, Fy)."""
        F = np.zeros((len(self.nodes) + 1, 2))
        for f in self.faces:
            tv = np.array([t_node[n] for n in f])       # (4, 2)
            for N, dj in self._face_gauss(f):
                F[f] += np.outer(N, (N @ tv)) * dj
        return F

    # ---- volume integral of the body force -----------------------------
    def body_load(self):
        """F_i = int_Omega B_SRC . phi_i dV over the hex8 elements: the
        consistent nodal force vector of the body force, i.e. exactly
        `inner(b, v) * dx` of the FEniCSx/scikit-fem siblings, evaluated here
        because the deck cannot be handed a non-constant source (header).

        3x3x3 Gauss — exact through degree 5 per direction. The integrand is
        b times a shape function, so a source polynomial up to degree 4 per
        direction is integrated EXACTLY and anything else to that order.
        Returns a (nnodes + 1, 2) array indexed by node id, so row 0 is the
        1-based padding row and stays zero.

        B_SRC may return arrays (the documented contract) or two plain floats
        (a constant body force); both broadcast."""
        conn = np.array([c for (_, c) in self.elems], dtype=int)   # (ne, 8)
        P = self.xyz[conn]                                         # (ne, 8, 3)
        # the corner signs of hex8 in THIS file's connectivity order: the z=0
        # face counter-clockwise, then the z=ZTHICK face above it.
        sg = np.array([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                       (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], float)
        g, w5, w8 = np.sqrt(0.6), 5.0 / 9.0, 8.0 / 9.0
        gp, gw = (-g, 0.0, g), (w5, w8, w5)
        Fx = np.zeros(len(self.nodes) + 1)
        Fy = np.zeros(len(self.nodes) + 1)
        for xi, wi in zip(gp, gw):
            for et, we in zip(gp, gw):
                for ze, wz in zip(gp, gw):
                    N = 0.125 * ((1 + sg[:, 0] * xi) * (1 + sg[:, 1] * et)
                                 * (1 + sg[:, 2] * ze))             # (8,)
                    dN = 0.125 * np.column_stack([
                        sg[:, 0] * (1 + sg[:, 1] * et) * (1 + sg[:, 2] * ze),
                        sg[:, 1] * (1 + sg[:, 0] * xi) * (1 + sg[:, 2] * ze),
                        sg[:, 2] * (1 + sg[:, 0] * xi) * (1 + sg[:, 1] * et)])
                    jac = np.einsum("eai,aj->eij", P, dN)           # (ne, 3, 3)
                    dv = np.abs(np.linalg.det(jac)) * (wi * we * wz)
                    xg = np.einsum("a,eai->ei", N, P)               # (ne, 3)
                    bx, by = B_SRC(xg[:, 0], xg[:, 1])
                    np.add.at(Fx, conn,
                              N[None, :] * (np.asarray(bx, float) * dv)[:, None])
                    np.add.at(Fy, conn,
                              N[None, :] * (np.asarray(by, float) * dv)[:, None])
        return np.column_stack([Fx, Fy])


# -------------------------------------------------------------- deck text
def write_deck(mesh, u_if, f_if, f_bd):
    """u_if: node id -> (ux, uy) for the interface Dirichlet set (dirichlet
    side). f_if: node id -> (Fx, Fy) consistent nodal forces (neumann side).
    f_bd: the body force's consistent nodal forces as a (nnodes + 1, 2) array
    indexed by node id, or None when B_SRC is identically zero — in which case
    no map and no load are written at all and the deck is the one this file
    produced before B_SRC existed."""
    ux_o, uy_o = u_dirichlet(mesh.xyz[mesh.outer, 0], mesh.xyz[mesh.outer, 1])
    L = ['<?xml version="1.0" encoding="ISO-8859-1"?>',
         '<febio_spec version="4.0">',
         '  <Module type="solid"/>',
         '  <Control><analysis>STATIC</analysis><time_steps>1</time_steps>'
         '<step_size>1</step_size>'
         '<solver type="solid"><symmetric_stiffness>symmetric'
         '</symmetric_stiffness><dtol>1e-12</dtol><etol>1e-12</etol>'
         '<rtol>0</rtol>'
         f'<linear_solver type="{LINSOLVE}"/></solver></Control>',
         '  <Globals><Constants><T>0</T><R>0</R><Fc>0</Fc></Constants></Globals>',
         f'  <Material><material id="1" name="Mat" type="isotropic elastic">'
         f'<density>1</density><E>{_n(E_MOD)}</E><v>{_n(NU)}</v>'
         f'</material></Material>',
         '  <Mesh>',
         '    <Nodes name="Object1">']
    L += [f'      <node id="{n}">{_n(x)},{_n(y)},{_n(z)}</node>'
          for (n, x, y, z) in mesh.nodes]
    L += ['    </Nodes>',
          '    <Elements type="hex8" mat="1" name="Part1">']
    L += [f'      <elem id="{e}">{",".join(str(c) for c in conn)}</elem>'
          for (e, conn) in mesh.elems]
    L += ['    </Elements>',
          '    <NodeSet name="all_nodes">'
          + ",".join(str(n[0]) for n in mesh.nodes) + '</NodeSet>',
          '    <NodeSet name="outer">'
          + ",".join(str(n) for n in mesh.outer) + '</NodeSet>',
          '    <NodeSet name="iface_free">'
          + ",".join(str(n) for n in mesh.iface_free) + '</NodeSet>',
          '    <NodeSet name="iface_all">'
          + ",".join(str(n) for n in mesh.iface_all) + '</NodeSet>',
          '  </Mesh>',
          '  <MeshDomains><SolidDomain name="Part1" mat="Mat"/></MeshDomains>',
          '  <MeshData>']
    # PER-NODE maps. `lid` is the 1-based index INTO THE NODE SET as it was
    # declared above — NOT the node id.
    for name, col in (("ox_map", ux_o), ("oy_map", uy_o)):
        L.append(f'    <NodeData name="{name}" node_set="outer" '
                 f'data_type="scalar">')
        L += [f'      <node lid="{lid}">{_n(v)}</node>'
              for lid, v in enumerate(col, 1)]
        L.append('    </NodeData>')
    if SIDE == "dirichlet":
        for name, comp in (("ix_map", 0), ("iy_map", 1)):
            L.append(f'    <NodeData name="{name}" node_set="iface_free" '
                     f'data_type="scalar">')
            L += [f'      <node lid="{lid}">{_n(u_if[n][comp])}</node>'
                  for lid, n in enumerate(mesh.iface_free, 1)]
            L.append('    </NodeData>')
    else:
        L.append('    <NodeData name="f_map" node_set="iface_free" '
                 'data_type="vec3">')
        L += [f'      <node lid="{lid}">{_n(f_if[n][0])},{_n(f_if[n][1])},0'
              f'</node>' for lid, n in enumerate(mesh.iface_free, 1)]
        L.append('    </NodeData>')
    if f_bd is not None:
        # The body force, over ALL nodes — including the prescribed ones. At a
        # prescribed dof the load changes nothing (the constraint replaces the
        # equation), but leaving those nodes out would be a DIFFERENT load
        # vector, and this array is also what the reaction export subtracts.
        # `mesh.nodes` in this order IS the all_nodes node set declared above,
        # so lid and node id line up.
        L.append('    <NodeData name="b_map" node_set="all_nodes" '
                 'data_type="vec3">')
        L += [f'      <node lid="{lid}">{_n(f_bd[nd[0]][0])},'
              f'{_n(f_bd[nd[0]][1])},0</node>'
              for lid, nd in enumerate(mesh.nodes, 1)]
        L.append('    </NodeData>')
    L.append('  </MeshData>')

    L += ['  <Boundary>',
          # PLANE STRAIN: u_z = 0 on every node of the single element layer.
          '    <bc name="planar" type="zero displacement" node_set="all_nodes">'
          '<x_dof>0</x_dof><y_dof>0</y_dof><z_dof>1</z_dof></bc>',
          '    <bc name="ox" type="prescribed displacement" node_set="outer">'
          '<dof>x</dof><value lc="1" type="map">ox_map</value>'
          '<relative>0</relative></bc>',
          '    <bc name="oy" type="prescribed displacement" node_set="outer">'
          '<dof>y</dof><value lc="1" type="map">oy_map</value>'
          '<relative>0</relative></bc>']
    if SIDE == "dirichlet":
        L += ['    <bc name="ix" type="prescribed displacement" '
              'node_set="iface_free">'
              '<dof>x</dof><value lc="1" type="map">ix_map</value>'
              '<relative>0</relative></bc>',
              '    <bc name="iy" type="prescribed displacement" '
              'node_set="iface_free">'
              '<dof>y</dof><value lc="1" type="map">iy_map</value>'
              '<relative>0</relative></bc>']
    L.append('  </Boundary>')

    loads = []
    if SIDE == "neumann":
        # APPLY the partner's numbers UNCHANGED, as the consistent nodal force
        # vector of +int_Gamma q_out_partner . v ds (see the header).
        loads += ['    <nodal_load name="iface_f" type="nodal_force" '
                  'node_set="iface_free">',
                  '      <value lc="1" type="map">f_map</value>',
                  '    </nodal_load>']
    if f_bd is not None:
        # A SECOND load, not a merged one. FEBio accumulates every model load
        # into the same residual (FEMechModel::ExternalForces walks the list
        # and each FENodalLoad adds its own values), so the two ADD on the
        # interface nodes they share, and the entry above still carries the
        # partner's numbers verbatim — which is what makes the deck auditable
        # against the coupling contract.
        loads += ['    <nodal_load name="body_f" type="nodal_force" '
                  'node_set="all_nodes">',
                  '      <value lc="1" type="map">b_map</value>',
                  '    </nodal_load>']
    if loads:
        L += ['  <Loads>'] + loads + ['  </Loads>']

    L += ['  <LoadData><load_controller id="1" type="loadcurve">'
          '<interpolate>LINEAR</interpolate><extend>CONSTANT</extend>'
          '<points><pt>0,0</pt><pt>1,1</pt></points>'
          '</load_controller></LoadData>',
          '  <Output><logfile>',
          f'    <node_data data="ux;uy" delim="," file="{LOG_U}" '
          f'node_set="iface_all"/>']
    if SIDE == "dirichlet":
        L.append(f'    <node_data data="Rx;Ry" delim="," file="{LOG_R}" '
                 f'node_set="iface_all"/>')
    else:
        L.append(f'    <element_data data="sx;sxy" delim="," file="{LOG_E}"/>')
    L += ['  </logfile></Output>', '</febio_spec>']
    Path(DECK).write_text("\n".join(L) + "\n")


# ------------------------------------------------------------ log parsing
def parse_log(path, ncol):
    """FEBio ASCII logfile: '*Step ...' / '*Data =' blocks, then 'id,v1,v2,...'.
    Returns {node id: (v1, ..., vncol)} for the LAST step in the file."""
    out = {}
    txt = Path(path).read_text()
    blocks = txt.split("*Step")
    body = blocks[-1] if len(blocks) > 1 else txt
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("*"):
            continue
        parts = line.split(",")
        if len(parts) < ncol + 1:
            continue
        try:
            out[int(float(parts[0]))] = tuple(float(p) for p in parts[1:ncol + 1])
        except ValueError:
            continue
    return out
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end


# -------------------------------------------------------------------- run
mesh = Mesh()
imp = read_imports()
y_if = mesh.y_if
if len(y_if) == 0:
    sys.exit(f"no interface nodes at x={IFACE_X}: this subdomain spans "
             f"[{X0},{X1}], so nothing is shared with the partner")

u_if, f_if = {}, {}
if SIDE == "dirichlet":
    u_line = sample(imp, "values", (UI_X, UI_Y), y_if)
    for j, (nb, nt) in enumerate(mesh.iface_pair):
        u_if[nb] = u_if[nt] = (float(u_line[j, 0]), float(u_line[j, 1]))
else:
    t_line = sample(imp, "normal_fluxes", (TI_X, TI_Y), y_if)
    t_node = {}
    for j, (nb, nt) in enumerate(mesh.iface_pair):
        t_node[nb] = t_node[nt] = (float(t_line[j, 0]), float(t_line[j, 1]))
    Fc = mesh.iface_mass_apply(t_node)
    for n in mesh.iface_all:
        f_if[n] = (float(Fc[n, 0]), float(Fc[n, 1]))

# The body force, integrated once for the whole subdomain. None when B_SRC is
# the shipped zero, so an unused feature costs the deck nothing.
f_bd = mesh.body_load()
if not np.any(f_bd):
    f_bd = None

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
write_deck(mesh, u_if, f_if, f_bd)
for f in (LOG_U, LOG_R, LOG_E):
    Path(f).unlink(missing_ok=True)

r = subprocess.run([FEBIO, "-i", DECK], capture_output=True, text=True,
                   timeout=3600)
if "N O R M A L   T E R M I N A T I O N" not in (r.stdout or ""):
    sys.stderr.write(f"FEBio did not terminate normally (rc={r.returncode})\n"
                     f"{(r.stdout or '')[-1500:]}\n")
    sys.exit(1)

ulog = parse_log(LOG_U, 2)
if not ulog:
    sys.stderr.write(f"empty FEBio node logfile {LOG_U}\n")
    sys.exit(2)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end
U = np.array([[0.5 * (ulog[nb][c] + ulog[nt][c]) for c in (0, 1)]
              for (nb, nt) in mesh.iface_pair], float)

if SIDE == "dirichlet":
    # THE CONSISTENT (REACTION) TRACTION (header): the node log's Rx, Ry are
    # r = A u_h - b at the PRESCRIBED dofs.
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    rlog = parse_log(LOG_R, 2)
    if not rlog:
        sys.stderr.write(f"empty FEBio reaction logfile {LOG_R}: this build "
                         f"did not produce the Rx/Ry node log data\n")
        sys.exit(2)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end
    w = mesh.iface_weights()
    R = np.array([[rlog[nb][c] + rlog[nt][c] for c in (0, 1)]
                  for (nb, nt) in mesh.iface_pair], float)
    W = np.array([w[nb] + w[nt] for (nb, nt) in mesh.iface_pair], float)
    # b INCLUDES THE BODY FORCE, and a <nodal_load> never reaches Rx, Ry at a
    # prescribed dof: its consistent load, the array the deck carries, comes off here.
    if f_bd is not None:
        R -= np.array([[f_bd[nb][c] + f_bd[nt][c] for c in (0, 1)]
                       for (nb, nt) in mesh.iface_pair], float)
    Q = np.zeros_like(R)
    ok = np.abs(W) > 1e-14
    Q[ok] = -R[ok] / W[ok, None]

    # THE TWO INTERFACE CORNERS ARE ON THE OUTER DIRICHLET BOUNDARY (a y-face),
    # so their rows carry the OUTER reaction too and their residual is not this
    # interface's traction. Take the nearest interior interface node rather than
    # exporting a corner value that is physically a different quantity.
    good = np.array(mesh.interior_j, dtype=int)
    good = good[ok[good]]
    if len(good):
        for j in range(len(y_if)):
            if j not in good:
                Q[j] = Q[good[np.argmin(np.abs(good - j))]]
else:
    # NEUMANN SIDE. FEBio reports Rx = Ry = 0 on these free dofs, so the export
    # is -Fc_i / w_i: Fc the consistent nodal force this side BUILT from the
    # partner's traction with its own interface faces and wrote into the deck,
    # w the same weights as the Dirichlet branch. A wrong facet set, Jacobian or
    # mass matrix moves it; a fault inside FEBio's own load path does not, and
    # the element-stress traction below is the independent second opinion.
    w = mesh.iface_weights()
    Fq = np.array([[Fc[nb][c] + Fc[nt][c] for c in (0, 1)]
                   for (nb, nt) in mesh.iface_pair], float)
    W = np.array([w[nb] + w[nt] for (nb, nt) in mesh.iface_pair], float)
    Q = np.zeros_like(Fq)
    ok = np.abs(W) > 1e-14
    Q[ok] = -Fq[ok] / W[ok, None]

    # The two interface corners sit on the outer Dirichlet boundary, exactly as
    # on the other side, so their weight mixes this interface with that face.
    good = np.array(mesh.interior_j, dtype=int)
    good = good[ok[good]]
    if len(good):
        for jj in range(len(y_if)):
            if jj not in good:
                Q[jj] = Q[good[np.argmin(np.abs(good - jj))]]

    # THE INDEPENDENT SECOND OPINION: this side's own element stresses averaged
    # onto the interface nodes. It owes nothing to the partner's numbers, and it
    # is coarse (it does not converge in the max norm): tens of percent from the
    # export is normal on a correct run, a discrepancy of order one is a fault.
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    elog = parse_log(LOG_E, 2)
    if not elog:
        sys.stderr.write(f"empty FEBio element logfile {LOG_E}\n")
        sys.exit(2)
    se = np.array([elog[e] for e in mesh.iface_elems], float)   # (NY, 2)
    Q_stress = np.zeros((len(y_if), 2))
    cnt = np.zeros(len(y_if))
    for jj in range(len(mesh.iface_elems)):
        Q_stress[jj] += -S * se[jj]
        Q_stress[jj + 1] += -S * se[jj]
        cnt[jj] += 1.0
        cnt[jj + 1] += 1.0
    Q_stress /= cnt[:, None]
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end
    if len(good):
        d = float(np.max(np.abs(Q[good] - Q_stress[good])))
        sc = max(1e-30, float(np.max(np.abs(Q[good]))))
        print(f"[febio neumann] applied-load traction vs own-stress traction: "
              f"max diff {d:.3e} ({d / sc:.2%} of peak) over {len(good)} "
              f"interior interface nodes. The second is a NON-CONVERGENT "
              f"boundary-stress average (measured order ~0), so tens of "
              f"percent here is normal; order-one disagreement is not.")

print(f"[febio {SIDE}] interface n={len(U)} "
      f"ux=[{U[:,0].min():.6g},{U[:,0].max():.6g}] "
      f"uy=[{U[:,1].min():.6g},{U[:,1].max():.6g}] "
      f"tx=[{Q[:,0].min():.6g},{Q[:,0].max():.6g}] "
      f"ty=[{Q[:,1].min():.6g},{Q[:,1].max():.6g}]")

# PER-LEVEL PERSISTENCE: this level's interface and field, named by LEVEL, which
# the next level does not overwrite. Interpolate THESE onto the probe points your
# task names. A dump defect must not cost the solve: exports.json comes after.
try:
    with open(f"interface_level{LEVEL}.csv", "w") as _f:
        _f.write("x,y,ux,uy,qx,qy\n")
        for _yy, (_ux, _uy), (_qx, _qy) in zip(y_if, U, Q):
            _px, _py = ((float(IFACE_X), float(_yy)) if AX == 0
                        else (float(_yy), float(IFACE_X)))
            _f.write(f"{_px:.11e},{_py:.11e},{float(_ux):.11e},"
                     f"{float(_uy):.11e},{float(_qx):.11e},{float(_qy):.11e}\n")
    # The FIELD comes from the deck's whole-mesh node log (<node_data ... file=
    # "nodal_out.csv"/>, NO node_set); without it there is no field. The slab is
    # one element thick, so a node's two z-layers average to its plane value.
    if Path(LOG_F).is_file():
        _acc = {}
        _body = Path(LOG_F).read_text().split("*Step")[-1]
        for _ln in _body.splitlines():
            _c = _ln.strip().split(",")
            if len(_c) < 7 or _ln.strip().startswith("*"):
                continue
            try:
                _v = [float(_t) for _t in _c[1:7]]
            except ValueError:
                continue
            _k = (round(_v[0], 12), round(_v[1], 12))
            _a = _acc.setdefault(_k, [0.0, 0.0, 0])
            _a[0] += _v[3]; _a[1] += _v[4]; _a[2] += 1
        with open(f"field_level{LEVEL}.csv", "w") as _f:
            _f.write("x,y,ux,uy\n")
            for (_px, _py), (_sx, _sy, _n) in sorted(_acc.items()):
                _f.write(f"{_px:.11e},{_py:.11e},{_sx / _n:.11e},{_sy / _n:.11e}\n")
        print(f"[febio {SIDE}] field_level{LEVEL}.csv: {len(_acc)} in-plane nodes")
    else:
        print(f"[febio {SIDE}] NO {LOG_F}: the deck logged only the interface, so "
              f"this level has no field to hand in. Add <node_data "
              f'data="x;y;z;ux;uy;uz" delim="," file="{LOG_F}"/> (no node_set) '
              f"inside <Output><logfile> and run this level again.")
except Exception as _dump_exc:
    # Leave no header-only CSV behind: it looks like a submission.
    for _partial in (f"field_level{LEVEL}.csv", f"interface_level{LEVEL}.csv"):
        try:
            if Path(_partial).is_file() and len(
                    Path(_partial).read_text().splitlines()) <= 1:
                Path(_partial).unlink()
        except OSError:
            pass
    print(f"[febio_elastic per-level dump] level {LEVEL} dump failed: "
          f"{_dump_exc!r}. The coupling continues, but this level has no "
          f"field file; fix the dump and run this level again.")

# exports.json LAST: the driver takes its existence as proof of success.
# `NDOF = <integer>` on a line of its OWN, per level: two dofs per in-plane node
# from the node log above, never a guess. The leading newline keeps it unglued.
try:
    print(f"\nNDOF = {2 * len(_acc)}")
except NameError:
    print(f"[febio {SIDE}] cannot report NDOF: the deck logged no whole-mesh "
          f"node data, so there is no node count to report. Your task's "
          f"execution log needs `NDOF = <integer>` on a line of its own.")

# ── EXPORT SELF-CHECK ─ keep this block. It stops four exports that look fine
#    and are worthless: non-finite values, an imported load that never entered the
#    system, the partner's traction negated instead of recovered, and a Dirichlet
#    side that exports exactly 0.0 where its own data make a reaction.
_chk_vals = np.asarray(U, float).ravel()
_chk_flux = np.asarray(Q, float).ravel()
if not (np.isfinite(_chk_vals).all() and np.isfinite(_chk_flux).all()):
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface values or "
                     "tractions; the solve did not produce a usable field, so "
                     "nothing was exported")
_chk_imp = (json.loads(Path("imports.json").read_text() or "{}")
            if Path("imports.json").is_file() else {})
_chk_qin = (np.concatenate([np.asarray(_d.get("normal_fluxes") or [], float).ravel()
                            for _d in _chk_imp.values()])
            if _chk_imp else np.zeros(0))
if SIDE == "neumann" and _chk_qin.size and np.abs(_chk_qin).max() > 0 \
        and np.abs(_chk_flux).max() < 1e-9 * np.abs(_chk_qin).max():
    raise SystemExit("EXPORT SELF-CHECK: the recovered interface traction is ~0 "
                     "against a nonzero imported traction: the imported load "
                     "never entered the assembled system (the facet term / "
                     "boundary condition that integrates it is missing). Fix "
                     "the application; do not couple on")
# (Dirichlet role only: a Neumann side's consistent recovery of a CONSTANT
#  applied load can legitimately reproduce it to the last bit.)
if SIDE == "dirichlet" and _chk_qin.shape == _chk_flux.shape and _chk_flux.size \
        and np.any(_chk_flux) and np.array_equal(_chk_flux, -_chk_qin):
    raise SystemExit("EXPORT SELF-CHECK: the exported traction is the partner's "
                     "array negated, bit for bit: a copy, not a recovery from "
                     "this side's own assembled system")
# (Dirichlet role: a traction of exactly 0.0 at every interface point is what a
#  side free of stress exports -- no body force, and a displacement handed in
#  that is one rigid motion. With either, the reaction is not zero.)
_chk_rigid = 0.0              # how far the displacement handed in is from one rigid motion
for _d in _chk_imp.values():
    _u, _p = np.asarray(_d.get("values") or [], float), np.asarray(_d.get("coordinates") or [], float)
    if _u.ndim == 2 and _u.shape[1] == 2 and _p.shape == _u.shape and len(_u) > 2 and np.any(_u):
        _m = np.zeros((2 * len(_u), 3))
        _m[0::2, 0], _m[1::2, 1], _m[0::2, 2], _m[1::2, 2] = 1.0, 1.0, -_p[:, 1], _p[:, 0]
        _fit = _m @ np.linalg.lstsq(_m, _u.ravel(), rcond=None)[0] - _u.ravel()
        _chk_rigid = max(_chk_rigid, float(np.abs(_fit).max() / np.abs(_u).max()))
_chk_body = f_bd is not None
if SIDE == "dirichlet" and _chk_flux.size and not np.any(_chk_flux) and (_chk_body or _chk_rigid > 1e-9):
    raise SystemExit("EXPORT SELF-CHECK: the exported traction is exactly 0.0 at every interface "
                     "point, while " + ("this side carries a body force" if _chk_body else
                                        f"the displacement it was handed is not one rigid motion (it "
                                        f"differs from the nearest one by {_chk_rigid:.1e} of its size)")
                     + ": a reaction recovered from this side's assembled system is not zero there")

Path("exports.json").write_text(json.dumps({
    "field_name": "displacement",
    "n_points": int(len(y_if)),
    "coordinates": [([float(IFACE_X), float(y)] if AX == 0 else [float(y), float(IFACE_X)])
                    for y in y_if],
    "values": [[float(a_), float(b_)] for a_, b_ in U],
    "normal_fluxes": [[float(a_), float(b_)] for a_, b_ in Q],
}, indent=2))
