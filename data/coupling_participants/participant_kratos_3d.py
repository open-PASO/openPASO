"""Kratos Multiphysics participant for the openPASO `couple` driver — 3-D scalar
conduction across a PLANAR interface.  Serves either side of the split.

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1, so an
iteration-1 fallback is mandatory), writes exports.json LAST and exits 0.
Needs KratosMultiphysics + ConvectionDiffusionApplication importable in the
interpreter named in `command`. DO NOT GUESS THAT INTERPRETER: take it from
`discover(query='list')`, which reports the one this install actually imports
Kratos with. A system python3 is a common trap — it usually exists, so nothing
looks wrong, and it raises ModuleNotFoundError because the Kratos build targets
a different Python version than the system one.

Physics: steady conduction  -div(K grad T) = f  on one BOX subdomain of a box
split by a plane.  Structured hexahedral grid cut into tetrahedra
(LaplacianElement3D4N); the interface carries either a Dirichlet trace or a
FluxCondition3D3N natural load.

======================================================================
WHY A SEPARATE 3-D FILE — what is genuinely different, not just bigger
======================================================================
The 2-D participants (participant_kratos.py, participant_kratos_neumann.py,
participant_fenics.py, participant_dune.py) all assume the interface is a LINE
at x = const sampled by ONE coordinate.  Three of their steps are wrong in 3-D
and each fails quietly:

1. ORDERING.  The driver relaxes export vectors ENTRY BY ENTRY, so the export
   order must be identical on every iteration.  In 2-D `argsort(y)` does it.
   In 3-D the interface is a plane and a sort on one coordinate leaves the
   other one in whatever order the mesh happened to number its nodes — stable
   within a run, so nothing complains, but the coupling then relaxes point i of
   this iteration against a DIFFERENT physical point of the last one.  Here the
   order is a tolerance-quantised `lexsort` over BOTH in-plane coordinates.

2. RESAMPLING A NON-MATCHING PARTNER.  In 2-D `np.interp` over the tangential
   coordinate is exact for a P1 trace.  In 3-D the partner's interface nodes
   are a 2-D point set: `np.interp` on either axis alone sorts that cloud by
   one coordinate and averages across the other, which produces a smooth,
   plausible, WRONG boundary condition.  `resample_plane` below does real 2-D
   interpolation, CUBIC on a tensor-product partner and Delaunay-linear
   otherwise.  Even the obvious BILINEAR choice costs the exported flux half
   its order (1.97 -> 1.32, measured; see `_bicubic`) while leaving the
   temperature field at 1.98, which is what makes it hard to notice.
   Nearest-neighbour is only O(h) and would cap the whole coupling at first
   order, so it is the last resort and it says so out loud.

3. QUADRATURE WEIGHTS.  The consistent-flux recovery divides the nodal
   reaction by  w_i = int_Gamma phi_i ds.  In 2-D those are EDGE LENGTHS
   (hy, and hy/2 at the two ends) and can be written down.  In 3-D they are
   FACE AREAS of the interface triangulation, w_i = sum over adjacent
   interface triangles of area/3.  MEASURED consequence of carrying the 2-D
   formula over to a unit-square interface, on the same converged solve:

       n    int_Gamma q ds, area weights   int_Gamma q ds, edge lengths
        6   -24.304                        -4.051
       12   -22.550                        -1.879
       24   -22.154   (exact -22.026)      -0.923   (and heading for 0)

   The edge-weighted flux is still smooth, still monotone in the right
   direction, still the right SHAPE — only scaled by ~h, so it looks entirely
   plausible and gets worse with refinement.  The conservation check is what
   catches it.

THE SIGN, WHICH IS THE ONLY THING A NEUMANN PARTICIPANT CAN GET SILENTLY WRONG.
Every participant exports its outward normal flux density with respect to ITS
OWN outward normal,  q = -(K grad T) . n_own,  so the two sides of a shared
interface carry OPPOSITE signs.  On this subdomain
      int_Omega K grad T . grad v = int_Omega f v + int_dOmega (K grad T . n) v ds,
and on the interface n_own = -n_partner, so  K grad T . n_own = +q_partner.
The partner's number is therefore applied UNCHANGED — no minus sign anywhere.
FluxCondition3D3N enforces exactly  K grad T . n = FACE_HEAT_FLUX (verified by
a 3-D patch test: T fixed to 0 on x=0, FACE_HEAT_FLUX=1 on the x=1 face of a
unit cube, K=1, no source -> T(x=1) = +1.000000 on every node, i.e. dT/dx=+1
not -1), so

      FACE_HEAT_FLUX  :=  the imported `normal_fluxes`, verbatim.

If you ever flip that sign to "make the temperatures look right", you have
built a coupling that drives heat the wrong way across the interface and still
converges.  The self-check printed at the end is the guard: it evaluates the
discrete divergence theorem on this subdomain and must come out at round-off.

`main()` returns a dict of everything a verification script needs (model part,
node-id array, interface ids, in-plane points, area weights, guarded and
UNGUARDED fluxes, the edge mask) so the participant can be imported and graded
against a manufactured solution.  The coupling itself only ever needs the file
handshake.

======================================================================
MEASURED — this file did not ship until it converged in a real coupling
======================================================================
Manufactured two-material 3-D conduction: box [0,1]^3 split by the plane
x = 0.5, one constant k on each half, exact Dirichlet on all five non-interface
faces of each half, so the whole RIM of the interface plane is an outer
Dirichlet edge — the 3-D corner case made as large as it can be.  Real
Dirichlet-Neumann coupling through src/core/coupling_driver.run_coupling, this
file as the DIRICHLET half against participant_dune_3d.py as the NEUMANN half,
on DELIBERATELY NON-MATCHING interface grids (n here against 4n/3 there).
Aitken, theta0 = 0.7, tol = 1e-9: converged in 31 / 31 / 32 iterations with
both participants reported "responsive".

  n / m                    6/8        12/16      24/32     order
  L2(T) over the WHOLE box 3.888e-01  1.019e-01  2.580e-02  1.93  1.98
  flux, iface INTERIOR     1.738e+00  4.635e-01  1.182e-01  1.91  1.97
  flux, WHOLE interface    2.530e+00  6.666e-01  1.915e-01  1.92  1.80
  flux, WHOLE unguarded    8.317e+00  5.599e+00  3.911e+00  0.57  0.52
  int_Gamma q ds          -24.0629   -22.5013   -22.1428    (exact -22.02642)

THE WHOLE-INTERFACE NUMBER IS DRAGGED DOWN BY THE RIM, exactly as the two end
nodes do it in 2-D.  Separating the contributions gives a rim-only L2 of
1.839 / 0.479 / 0.151 — order 1.94 then 1.67, still falling — against the
interior's steady 1.97.  The rim is O(h) accurate on a strip of area O(h), so
it contributes O(h^1.5): the same asymptote as the 2-D corner, approached more
slowly because at n = 24 the rim term is still only 1.27x the interior one.
Quote the INTERIOR order when grading the recovery and the WHOLE one when
grading what the partner receives; they are different questions.

CONSERVATION, with the face-area weights: interface area recovered as exactly
1.000000 on both sides at every level; each side's discrete divergence theorem
at round-off (1.7e-13 here, 7.3e-13 on the DUNE side, at n = 24); and the
EXCHANGE balance — what this side says it sent against what the partner's own
quadrature says it received — converging at second order.  That one is NOT
zero and cannot be: resampling between
non-matching interface grids is accurate, not conservative.  If you need it at
round-off, match the interface meshes.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication  # noqa: F401

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
SIDE       = "dirichlet"     # "dirichlet" (import T, export flux) | "neumann"
PARTNER    = "right"         # the partner's `name` in your couple(...) call

X0, X1     = 0.0, 0.5        # this subdomain's box
Y0, Y1     = 0.0, 1.0
Z0, Z1     = 0.0, 1.0

IFACE_AXIS = 0               # interface plane normal: 0=x, 1=y, 2=z
IFACE_POS  = 0.5             # its position; must equal this box's lo or hi on that axis

K          = 1.7             # conductivity of THIS subdomain (constant)
NX, NY, NZ = 8, 8, 8         # this subdomain's OWN mesh; need NOT match the partner

# ── THE PER-LEVEL RULE (served). A ./config.json {"level": k, "nx": .., "ny": ..,
#    "nz": ..} next to this script overrides the mesh knobs and names the level.
#    The dumps beside exports.json carry that level in their NAME, so a mesh
#    study leaves one file per level instead of the fine mesh overwriting the
#    coarse ones.
LEVEL = 1
_cfg = {}
for _src, _txt in (("config.json", Path("config.json").read_text() if Path("config.json").is_file() else ""),
                   ("OPENPASO_CONFIG_JSON", os.environ.get("OPENPASO_CONFIG_JSON", ""))):
    try:
        _cfg.update(**json.loads(_txt or "{}"))
        LEVEL = int(_cfg.get("level", LEVEL))
        NX = int(_cfg.get("nx", NX))
        NY = int(_cfg.get("ny", NY))
        NZ = int(_cfg.get("nz", NZ))
    except (ValueError, TypeError) as _e:
        raise SystemExit(f"{_src} could not be read ({_e}); nothing was solved")


# Which outer faces carry a Dirichlet condition.  Names are "<axis><0|1>" with
# 0 = the low face and 1 = the high face, e.g. "x0" is the plane x = X0.  Every
# face NOT listed here (and not the interface) is natural, i.e. insulated.
# The interface face itself must not appear.
DIRICHLET_FACES = ("x0",)

T_INIT     = 295.0           # iteration-1 fallback interface temperature
Q_INIT     = 0.0             # iteration-1 fallback interface flux density
LIN_SOLVER = "amgcl"         # "amgcl" (needed in 3-D past ~5k nodes) | "direct"


def source(x, y, z):
    """Volumetric source f in  -div(K grad T) = f, sampled at the NODES.

    x, y, z are numpy arrays.  Return a scalar for a uniform source or any
    expression for a graded / manufactured one; it is interpolated into the P1
    space, which costs O(h^2) and so does not touch the second-order rate.
    THIS IS ONLY TRUE WHILE f IS SMOOTH ON THIS SUBDOMAIN.  It is, because a
    participant owns ONE material — the jump in f lives on the interface, which
    is this subdomain's boundary, not inside it.
    """
    return 0.0 * x


def outer_value(x, y, z):
    """Dirichlet value on the faces listed in DIRICHLET_FACES.

    x, y, z are numpy arrays.  Return a scalar for a constant temperature or
    any expression for a graded / manufactured one.
    """
    return 320.0 + 0.0 * x
# ─────────────────────────────────────────────────────────────────────────

LO = np.array([X0, Y0, Z0], float)
HI = np.array([X1, Y1, Z1], float)
NE = np.array([NX, NY, NZ], int)
AX = int(IFACE_AXIS)
TAN = [a for a in (0, 1, 2) if a != AX]          # the two IN-PLANE axes
LEN = HI - LO
TOL = 1e-9 * float(np.max(LEN))
AXN = "xyz"

# Is the interface this box's HIGH face on AX?  The outward normal there is
# +e_AX, on the LOW face it is -e_AX.
ON_HI = abs(IFACE_POS - HI[AX]) < abs(IFACE_POS - LO[AX])
S = 1.0 if ON_HI else -1.0
IFACE_FACE = f"{AXN[AX]}{1 if ON_HI else 0}"

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
# The six Kuhn (path) tetrahedra of a hexahedron, on the corner numbering
# b = i + 2*j + 4*k.  Kuhn's decomposition applied identically to every cell is
# FACE-CONFORMING: neighbouring cells split their shared quad along the same
# diagonal, so the mesh has no hanging nodes.  The node order of each tet is
# chosen so its signed volume is POSITIVE — three of the six naive path orders
# are odd permutations, and a negatively oriented LaplacianElement3D4N produced
# a singular system ("LUSkylineFactorization: Error zero sum", solution NaN).
# build_model() re-checks the sign rather than trusting this table.
KUHN = ((0, 1, 3, 7), (0, 1, 7, 5), (0, 2, 7, 3),
        (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 7, 6))

_MODEL = None            # keeps the KM.Model alive for the ModelPart's lifetime
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end


# ── driver handshake ────────────────────────────────────────────────────────

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


def _unique_tol(a, tol):
    """Sorted unique values of `a`, merging entries closer together than tol."""
    s = np.sort(np.asarray(a, float))
    keep = [s[0]]
    for v in s[1:]:
        if v - keep[-1] > tol:
            keep.append(v)
    return np.asarray(keep)


def _bilinear(u, v, G, q):
    """Bilinear interpolation of grid values G[iu, iv] at the points q (M,2).
    Points outside the grid are CLAMPED to its border: the two sides nominally
    cover the same plane, so an excursion is round-off or a partner whose mesh
    stops a hair short, and extrapolating there is worse than clamping."""
    iu = np.clip(np.searchsorted(u, q[:, 0]) - 1, 0, u.size - 2)
    iv = np.clip(np.searchsorted(v, q[:, 1]) - 1, 0, v.size - 2)
    tu = np.clip((q[:, 0] - u[iu]) / (u[iu + 1] - u[iu]), 0.0, 1.0)
    tv = np.clip((q[:, 1] - v[iv]) / (v[iv + 1] - v[iv]), 0.0, 1.0)
    return ((1 - tu) * (1 - tv) * G[iu, iv] + tu * (1 - tv) * G[iu + 1, iv]
            + (1 - tu) * tv * G[iu, iv + 1] + tu * tv * G[iu + 1, iv + 1])


def _lag4(nodes, xq):
    """Cubic Lagrange weights (M,4) for query xq (M,) on stencils nodes (M,4)."""
    d = xq[:, None] - nodes
    wts = np.empty_like(d)
    for a in range(4):
        num = np.ones(len(xq))
        den = np.ones(len(xq))
        for b in range(4):
            if b == a:
                continue
            num *= d[:, b]
            den *= nodes[:, a] - nodes[:, b]
        wts[:, a] = num / den
    return wts


def _bicubic(u, v, G, q):
    """Tensor-product CUBIC interpolation of grid values G at points q (M,2).

    WHY NOT BILINEAR — this is the single measurement that mattered most in
    building this file.  Same participant, same manufactured problem, same
    non-matching partner grid, ONLY the tensor route swapped (measured orders
    over meshes 6/12/24, and the L2 error at 24):

        route      volume L2(T)   flux INTERIOR      flux WHOLE plane
        cubic      1.98           1.97  (1.32e-01)   1.82  (2.06e-01)
        bilinear   1.98           1.32  (2.18e-01)   1.44  (2.67e-01)

    The mechanism: a bilinear interpolant of a smooth trace is off by O(h^2),
    but its error is a KINK-SHAPED bump on the scale of one partner cell.  The
    discrete Dirichlet-to-Neumann map differentiates the boundary data, so an
    h-scale wiggle of amplitude h^2 comes back as a flux error of amplitude
    h^2/h = O(h).  Read the table again: the temperature field does not notice
    AT ALL — a high-frequency, mean-zero boundary perturbation is strongly
    damped inside the domain, so the volume order is 1.98 either way.  That is
    exactly what makes this failure quiet.  The field looks right, every
    residual converges, and the only thing that has changed is the order of the
    number the partner actually consumes.

    A cubic tensor interpolant is O(h^4) with a smooth error, so the amplified
    part is O(h^3) and the flux keeps its second order.  It does NOT invent
    accuracy: the partner's nodal values are still only as good as the
    partner's own mesh, and this only stops the TRANSFER from being the
    limiting error.  Needs >= 4 lines per direction; below that the caller
    falls back to bilinear.
    """
    qu = np.clip(q[:, 0], u[0], u[-1])
    qv = np.clip(q[:, 1], v[0], v[-1])
    bu = np.clip(np.searchsorted(u, qu) - 2, 0, u.size - 4)
    bv = np.clip(np.searchsorted(v, qv) - 2, 0, v.size - 4)
    su = bu[:, None] + np.arange(4)[None, :]
    sv = bv[:, None] + np.arange(4)[None, :]
    wu = _lag4(u[su], qu)
    wv = _lag4(v[sv], qv)
    stencil = G[su[:, :, None], sv[:, None, :]]          # (M,4,4)
    return np.einsum("ma,mb,mab->m", wu, wv, stencil)


def resample_plane(imp, key, fallback, pts):
    """Map a partner's interface samples onto THIS participant's points.

    `pts` is (N,2): the two IN-PLANE coordinates of this side's interface
    nodes.  The driver does no interpolation — non-matching meshes are handled
    here, and in 3-D that means a genuine 2-D interpolation (see item 2 of the
    header).  Routes, in order of preference:

      1. TENSOR PRODUCT — the partner's points are exactly unique(u) x
         unique(v).  True whenever the partner meshes a box, which is the
         normal case.  CUBIC on that rectilinear grid (bilinear if either
         direction has fewer than 4 lines): see _bicubic for why the obvious
         bilinear choice costs the exported flux half its convergence order.
      2. scipy LinearNDInterpolator on the Delaunay triangulation of the
         partner's points; the few points outside its convex hull are filled
         from the nearest partner point.  This route is piecewise linear and
         carries the same order penalty as bilinear — an unstructured partner
         on a coupling that must deliver a second-order FLUX is a real
         limitation, not a formality.
      3. NEAREST NEIGHBOUR, with a loud warning.  Only O(h): it caps the
         coupled solution itself at first order.  It exists so an exotic
         partner mesh degrades instead of crashing.
    """
    n = len(pts)
    if not imp or not imp.get("coordinates"):
        return np.full(n, float(fallback))
    co = np.asarray(imp["coordinates"], float)
    if co.ndim != 2 or co.shape[1] < 3:
        # A 2-D partner on a 3-D interface.  Failing loudly is the point: the
        # alternative is to silently drop a coordinate and couple two different
        # geometries to each other.
        sys.exit(f"partner '{PARTNER}' exported {co.shape[1] if co.ndim == 2 else '?'}"
                 f"-component coordinates; a 3-D planar interface needs [x,y,z]")
    # `or []` and not `.get(key, [])`: a partner that writes the key with an
    # explicit null gets [] here instead of a TypeError out of np.asarray, and
    # falls through to the fallback like any other unusable import.
    vs = np.asarray(imp.get(key) or [], float).ravel()
    if vs.size != co.shape[0]:
        return np.full(n, float(fallback))
    src = co[:, TAN]

    u = _unique_tol(src[:, 0], TOL)
    v = _unique_tol(src[:, 1], TOL)
    if u.size >= 2 and v.size >= 2 and u.size * v.size == src.shape[0]:
        iu = np.abs(src[:, 0][:, None] - u[None, :]).argmin(1)
        iv = np.abs(src[:, 1][:, None] - v[None, :]).argmin(1)
        G = np.full((u.size, v.size), np.nan)
        G[iu, iv] = vs
        if not np.isnan(G).any():
            p = np.asarray(pts, float)
            if u.size >= 4 and v.size >= 4:
                return _bicubic(u, v, G, p)
            return _bilinear(u, v, G, p)

    try:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        lin = LinearNDInterpolator(src, vs)
        out = lin(pts)
        bad = ~np.isfinite(out)
        if bad.any():
            out[bad] = NearestNDInterpolator(src, vs)(np.asarray(pts)[bad])
        return out
    except Exception as e:                                    # noqa: BLE001
        print(f"[kratos3d {SIDE}] WARNING: 2-D interpolation unavailable ({e}); "
              f"falling back to NEAREST NEIGHBOUR on the interface. That is "
              f"O(h) and WILL flatten the convergence rate — match the meshes "
              f"or install scipy before believing any order measured this way.")
        d = ((np.asarray(pts)[:, None, :] - src[None, :, :]) ** 2).sum(-1)
        return vs[d.argmin(1)]


def order_plane(pts):
    """Deterministic order of interface points over BOTH in-plane coordinates.

    Quantised so that nodes nominally on the same row cannot be split apart by
    round-off into an order that changes with the mesh generator's mood.  The
    driver relaxes entry by entry, so this order IS part of the contract.
    """
    q = np.round(np.asarray(pts, float) / max(TOL, 1e-300))
    return np.lexsort((q[:, 1], q[:, 0]))


# ── mesh ────────────────────────────────────────────────────────────────────
def build_model():
    """Structured hex grid of the box, cut into Kuhn tetrahedra.

    Returns (model_part, nid) with nid an (NX+1, NY+1, NZ+1) int array of node
    ids.  The Model owns the ModelPart, so it is parked in a module global: a
    ModelPart is a C++ object with no __dict__ to hang it on, and letting the
    Model be collected leaves the part dangling.

    THE KUHN SPLIT, a fact about a hexahedron.  With the corners of one hex
    numbered b = i + 2*j + 4*k (i, j, k = 0 or 1, its offset along x, y, z),
    its six tetrahedra, each of positive volume in this node order, are
    (0, 1, 3, 7) (0, 1, 7, 5) (0, 2, 7, 3) (0, 2, 6, 7) (0, 4, 5, 7) (0, 4, 7, 6)
    and the same six in every hex meet their neighbours face to face.
    check_tets() below tests the split you build before anything is solved.
    """
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    global _MODEL
    _MODEL = KM.Model()
    mp = _MODEL.CreateModelPart("thermal")
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
    st = KM.ConvectionDiffusionSettings()
    st.SetUnknownVariable(KM.TEMPERATURE)
    st.SetDiffusionVariable(KM.CONDUCTIVITY)
    st.SetVolumeSourceVariable(KM.HEAT_FLUX)
    # FluxCondition3D3N reads its load from the SURFACE source variable named
    # here; leave it FACE_HEAT_FLUX or the condition assembles nothing and the
    # interface silently becomes insulated.
    st.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
    mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, st)
    for v in (KM.TEMPERATURE, KM.CONDUCTIVITY, KM.HEAT_FLUX, KM.FACE_HEAT_FLUX,
              KM.REACTION_FLUX):
        mp.AddNodalSolutionStepVariable(v)
    mp.SetBufferSize(1)
    props = mp.CreateNewProperties(1)

    gx = [np.linspace(LO[a], HI[a], NE[a] + 1) for a in range(3)]
    nid = np.zeros(tuple(NE + 1), dtype=np.int64)
    c = 1
    for k in range(NE[2] + 1):
        for j in range(NE[1] + 1):
            for i in range(NE[0] + 1):
                mp.CreateNewNode(c, float(gx[0][i]), float(gx[1][j]), float(gx[2][k]))
                nid[i, j, k] = c
                c += 1

    P = np.stack(np.meshgrid(gx[0], gx[1], gx[2], indexing="ij"), axis=-1)
    e = 1
    for k in range(NE[2]):
        for j in range(NE[1]):
            for i in range(NE[0]):
                corner = [int(nid[i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1)])
                          for b in range(8)]
                cp = [P[i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1)]
                      for b in range(8)]
                for t in KUHN:
                    a0, a1, a2, a3 = cp[t[0]], cp[t[1]], cp[t[2]], cp[t[3]]
                    if np.dot(np.cross(a1 - a0, a2 - a0), a3 - a0) < 0:
                        t = (t[0], t[1], t[3], t[2])     # keep volumes positive
                    mp.CreateNewElement(
                        "LaplacianElement3D4N", e,
                        [corner[t[0]], corner[t[1]], corner[t[2]], corner[t[3]]],
                        props)
                    e += 1
    return mp, nid


def face_nodes(nid, name):
    """Node ids on the named box face, e.g. "y1" -> the plane y = Y1."""
    a = AXN.index(name[0])
    hi = name[1] == "1"
    sl = [slice(None)] * 3
    sl[a] = -1 if hi else 0
    return nid[tuple(sl)].ravel()
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end


def interface_triangles(mp, iface_ids):
    """The interface triangulation, harvested from the tets themselves.

    A tet with exactly 3 nodes on the plane has that triple as one of its
    FACES, and because the plane is a domain boundary the face is a boundary
    face.  Reading the triangles off the elements instead of re-deriving them
    from the Kuhn table means an edit to build_model cannot silently
    desynchronise the quadrature weights from the mesh (verified: the harvested
    triangles of a unit-square interface sum to area 1.000000000000).
    """
    s = set(int(i) for i in iface_ids)
    tris = []
    for el in mp.Elements:
        on = [n.Id for n in el.GetNodes() if n.Id in s]
        if len(on) == 3:
            tris.append(on)
    return tris


def area_weights(mp, tris, index_of):
    """w_i = int_Gamma phi_i ds  for the P1 trace = sum of adjacent triangle
    areas / 3.  THESE ARE FACE AREAS, not edge lengths (see item 3 of the
    header); the whole consistent-flux recovery is divided by them."""
    w = np.zeros(len(index_of))
    for t in tris:
        p = np.array([[mp.Nodes[i].X, mp.Nodes[i].Y, mp.Nodes[i].Z] for i in t])
        a = 0.5 * float(np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])))
        for i in t:
            w[index_of[i]] += a / 3.0
    return w


def check_tets(mp):
    """THE MESH TEST: stop before the solve unless the tetrahedra fill this box
    exactly once.

    Kratos assembles any four nodes it is given and never says a mesh is wrong.
    Measured on this install with hex splits cut by hand: a flat tetrahedron
    gives NaN fluxes; a mix of positive and negative ones stops the solve with
    'Error zero sum' or skips it ('ATTENTION! setting the RHS to zero!', the
    field left at its start values); all of them negative gives the right
    field with the flux sign flipped; overlapping ones give a smooth field and
    a wrong flux.  Three tests, in this order: every volume above zero, the
    volumes adding up to the box, and every face shared by exactly two
    tetrahedra from opposite sides unless it lies on the box's boundary.
    """
    els = list(mp.Elements)
    con = [[int(n.Id) for n in el.GetNodes()] for el in els]
    if not con or any(len(c) != 4 for c in con):
        print("[kratos3d] MESH TEST not judged: the model part holds no elements, "
              "or elements that are not 4-node tetrahedra.")
        return None
    eid = [int(el.Id) for el in els]
    con = np.asarray(con, dtype=np.int64)
    xyz = {int(n.Id): (float(n.X), float(n.Y), float(n.Z)) for n in mp.Nodes}
    P = np.array([[xyz[int(i)] for i in row] for row in con])
    vol = np.einsum("ij,ij->i", np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]),
                    P[:, 3] - P[:, 0]) / 6.0
    box = float(np.prod(LEN))
    fix = " The Kuhn split in build_model's notes fills a hex exactly once."
    tiny = 1e-9 * box / len(vol)
    bad = np.flatnonzero(vol <= tiny)
    if bad.size:
        e = int(bad[0])
        what = ("has zero volume: its four nodes lie in one plane" if vol[e] > -tiny else
                "has negative volume: its nodes run the other way round (swap two of them)")
        sys.exit(f"MESH TEST: tetrahedron {eid[e]} (nodes {con[e].tolist()}) {what}; "
                 f"{bad.size} of {len(vol)} tetrahedra fail this test.{fix}")
    if abs(vol.sum() - box) > 1e-9 * box:
        sys.exit(f"MESH TEST: the tetrahedra add up to a volume of {vol.sum():.9g} and "
                 f"the box is {box:.9g}: they leave a gap or overlap.{fix}")
    # The faces of a positive tetrahedron (a, b, c, d), each turned outward:
    # (b, c, d), (a, d, c), (a, b, d), (a, c, b).  Two tetrahedra on opposite
    # sides of a face turn it opposite ways, so the turns cancel.
    pick = [1, 2, 3, 0, 3, 2, 0, 1, 3, 0, 2, 1]
    F = con[:, pick].reshape(-1, 3)
    FP = P[:, pick, :].reshape(-1, 3, 3)
    odd = ((F[:, 0] > F[:, 1]).astype(int) + (F[:, 0] > F[:, 2]) + (F[:, 1] > F[:, 2])) % 2
    S = np.sort(F, axis=1)
    m = int(S.max()) + 1
    uk, inv, cnt = np.unique((S[:, 0] * m + S[:, 1]) * m + S[:, 2],
                             return_inverse=True, return_counts=True)
    turn = np.bincount(inv, weights=1.0 - 2.0 * odd, minlength=len(uk))
    # FP is (face, vertex, coordinate): .all(1) runs over a face's three vertices, so a face is on
    # the box's surface when all three share one coordinate of LO or of HI (x == X0, ...).
    rim = ((np.abs(FP - LO) < TOL).all(1) | (np.abs(FP - HI) < TOL).all(1)).any(1)
    lone = (cnt[inv] == 1) & ~rim
    worse = ((cnt[inv] == 2) & (turn[inv] != 0)) | (cnt[inv] > 2)
    hit = np.flatnonzero(lone | worse)
    if hit.size:
        f = int(hit[0])
        who = [eid[int(k) // 4] for k in np.flatnonzero(inv == inv[f])]
        if lone[f]:
            what = ("lies inside the box and no other tetrahedron has it: the "
                    "tetrahedra there leave a gap or overlap, or two neighbouring "
                    "hexes cut their shared face along different diagonals")
        else:
            what = (f"is shared by tetrahedra {who}, not by two from opposite sides "
                    "of it: they overlap")
        sys.exit(f"MESH TEST: face {F[f].tolist()} of tetrahedron {eid[f // 4]} "
                 f"{what}.{fix}")
    print(f"[kratos3d] MESH TEST: {len(vol)} tetrahedra of positive volume fill "
          f"the box exactly once.")
    return None


# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
def linear_solver():
    if LIN_SOLVER == "direct":
        return KM.SkylineLUFactorizationSolver()
    # A skyline LU on a 3-D tet mesh is O(n^{7/3}) in memory and time: measured
    # 19.6 s against AMGCL's 2.0 s at 15625 nodes, and it grows much faster.
    # The tolerance is TIGHT on purpose — the exported flux is A u - b, so any
    # slack in u shows up undivided in the number the partner consumes.
    return KM.AMGCLSolver(KM.Parameters("""{
        "solver_type": "amgcl", "smoother_type": "spai0", "krylov_type": "cg",
        "coarsening_type": "aggregation", "max_iteration": 5000,
        "tolerance": 1e-13, "verbosity": 0, "scaling": false }"""))
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end


def main():
    if abs(IFACE_POS - LO[AX]) > TOL and abs(IFACE_POS - HI[AX]) > TOL:
        sys.exit(f"IFACE_POS={IFACE_POS} is not a {AXN[AX]}-face of this box "
                 f"[{LO[AX]},{HI[AX]}] — nothing is shared with the partner")
    if IFACE_FACE in DIRICHLET_FACES:
        sys.exit(f"DIRICHLET_FACES lists '{IFACE_FACE}', which IS the interface. "
                 f"The outer condition would overwrite the coupling and the run "
                 f"would still exit 0 with a plausible export.")
    if np.any(NE < 1) or NE[AX] < 1:
        sys.exit(f"NX,NY,NZ = {tuple(NE)}: need at least one element per axis, "
                 f"and >= 1 across the interface axis so the interface plane and "
                 f"the opposite face do not land on the same nodes")

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    mp, nid = build_model()

    sl = [slice(None)] * 3
    sl[AX] = -1 if ON_HI else 0
    iface_grid = nid[tuple(sl)]                       # 2-D array of node ids
    ids_raw = iface_grid.ravel()
    pts_raw = np.array([[mp.Nodes[int(i)].X, mp.Nodes[int(i)].Y,
                         mp.Nodes[int(i)].Z] for i in ids_raw])
    # A guard on the one place geometry meets node numbering: if an edit to
    # build_model reorders the grid, this slice would name an INTERIOR plane,
    # the coupling data would be applied inside the domain, and the run would
    # still exit 0.
    if np.max(np.abs(pts_raw[:, AX] - IFACE_POS)) > TOL:
        sys.exit(f"internal: the interface slice sits at {AXN[AX]}="
                 f"{pts_raw[:, AX].mean()}, not {IFACE_POS} — the mesh and the "
                 f"slice index disagree")
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    # THE MESH TEST, served: the run stops here, before anything is solved,
    # unless the tetrahedra built above fill the box exactly once.
    check_tets(mp)

    order = order_plane(pts_raw[:, TAN])
    ids = ids_raw[order]
    pts3 = pts_raw[order]
    pts = pts3[:, TAN]
    index_of = {int(i): n for n, i in enumerate(ids)}

    tris = interface_triangles(mp, ids)
    w = area_weights(mp, tris, index_of)

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    # ── material, source, outer Dirichlet ───────────────────────────────────
    XYZ = np.array([[n.X, n.Y, n.Z] for n in mp.Nodes])
    fval = np.broadcast_to(np.asarray(
        source(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]), float), (len(XYZ),))
    for n, fv in zip(mp.Nodes, fval):
        n.SetSolutionStepValue(KM.CONDUCTIVITY, float(K))
        n.SetSolutionStepValue(KM.HEAT_FLUX, float(fv))
        n.SetSolutionStepValue(KM.FACE_HEAT_FLUX, 0.0)

    outer_ids = set()
    for name in DIRICHLET_FACES:
        fids = face_nodes(nid, name)
        fxyz = np.array([[mp.Nodes[int(i)].X, mp.Nodes[int(i)].Y,
                          mp.Nodes[int(i)].Z] for i in fids])
        gv = np.broadcast_to(np.asarray(
            outer_value(fxyz[:, 0], fxyz[:, 1], fxyz[:, 2]), float), (len(fids),))
        for i, val in zip(fids, gv):
            n = mp.Nodes[int(i)]
            n.SetSolutionStepValue(KM.TEMPERATURE, float(val))
            n.Fix(KM.TEMPERATURE)
            outer_ids.add(int(i))
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    # Interface nodes that ALSO sit on an outer Dirichlet face: in 3-D that is
    # the whole RIM of the interface plane, not the two corner nodes of 2-D.
    edge = np.array([int(i) in outer_ids for i in ids])

    # ── the interface ───────────────────────────────────────────────────────
    imp = read_imports()
    if SIDE == "dirichlet":
        # The RIM is claimed by both conditions.  Leaving that to statement
        # order is not a decision anyone made, so the rule is explicit and
        # documented: on the rim the OUTER condition wins, because it is the
        # modeller's known data while the interface value is only the partner's
        # current iterate.
        free = ~edge
        T_in = resample_plane(imp, "values", T_INIT, pts[free])
        for i, val in zip(ids[free], T_in):
            n = mp.Nodes[int(i)]
            n.SetSolutionStepValue(KM.TEMPERATURE, float(val))
            n.Fix(KM.TEMPERATURE)
        q_in = None
    else:
        q_in = resample_plane(imp, "normal_fluxes", Q_INIT, pts)
        for i, val in zip(ids, q_in):
            mp.Nodes[int(i)].SetSolutionStepValue(KM.FACE_HEAT_FLUX, float(val))
        props = mp.GetProperties()[1]
        for c, t in enumerate(tris):
            mp.CreateNewCondition("FluxCondition3D3N", c + 1, t, props)

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    # AddDof with a REACTION variable: without the second argument the fixed
    # dofs have nowhere to store their reaction and it is silently discarded.
    KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    builder = KM.ResidualBasedBlockBuilderAndSolver(linear_solver())
    # arg 4 is CalculateReactionsFlag: it must be True — the Dirichlet side's
    # interface flux IS the reaction, and the conservation self-check is built
    # from it on both sides.
    strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder,
                                              True, False, False, False)
    strategy.Initialize()
    strategy.Solve()
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end
    # TWO THINGS YOUR SOLVE ABOVE MUST DO, or the recovery below reads zeros:
    #   * AddDof(TEMPERATURE, REACTION_FLUX, mp) -- the SECOND argument gives
    #     every fixed dof a place to store its reaction; without it the
    #     reaction is silently discarded.
    #   * run the strategy with CalculateReactionsFlag=True (the 4th positional
    #     argument of ResidualBasedLinearStrategy): the Dirichlet side's
    #     interface flux IS the reaction, and the conservation self-check on
    #     both sides is built from REACTION_FLUX.

    T = np.array([mp.Nodes[int(i)].GetSolutionStepValue(KM.TEMPERATURE) for i in ids])
    r = np.array([mp.Nodes[int(i)].GetSolutionStepValue(KM.REACTION_FLUX) for i in ids])

    if SIDE == "dirichlet":
        # ── THE CONSISTENT (REACTION) FLUX ──────────────────────────────────
        # WHY NOT A GRADIENT RECOVERY.  An L2 projection of grad T, or a
        # one-sided difference quotient, is only O(h) accurate ON the boundary
        # — the superconvergence points of a P1 gradient are interior — and the
        # boundary trace is exactly what the coupling reads.  Measured against
        # the manufactured solution below, a projection converges at ~1 while
        # this recovery converges at ~2, so the recovery, not the physics and
        # not the partner, would be setting the answer.
        #
        # From  a(u,v) - (f,v) = int_dOmega (K grad u . n) v ds
        #                      = -int_Gamma qn v ds
        # it follows that for every basis function phi_i on the interface
        #     int_Gamma qn phi_i ds = -r_i,   r = A u_h - b
        # with r the UNCONSTRAINED residual on the constrained rows — which is
        # precisely what Kratos calls the REACTION: its builder recomputes the
        # RHS with NO Dirichlet condition applied and stores (A u_h - b)_i at
        # every fixed dof.  Dividing by the FACE-AREA weight w_i turns the
        # functional into a density the partner can interpolate pointwise.
        Q = np.zeros(len(ids))
        ok = np.abs(w) > 1e-14 * max(1.0, float(np.max(w)))
        Q[ok] = -r[ok] / w[ok]
        Q_raw = Q.copy()

        # ── THE EDGE GUARD — in 3-D the "corner" is a whole EDGE ─────────────
        # An interface node that ALSO lies on an outer Dirichlet face carries
        # the OUTER reaction as well, so its residual is not this interface's
        # flux.  In 2-D that is TWO nodes.  In 3-D the interface is a plane and
        # the affected set is its entire RIM: 4n of (n+1)^2 nodes — 96 of 625
        # (15%) at n=24, against 2 of 25 (8%) in 2-D at the same n.  They are
        # replaced by the nearest interior interface node — nearest in the
        # PLANE, over both coordinates, not along one of them.  MEASURED on the
        # manufactured problem: leaving them raw costs the whole-interface flux
        # its order outright (0.52 instead of 1.8) and inflates its L2 error by
        # 19x at n=24 (3.91 against 0.206).
        suspect = edge | ~ok
        good = np.where(~suspect)[0]
        if len(good):
            gp = pts[good]
            for i in np.where(suspect)[0]:
                d = ((gp - pts[i]) ** 2).sum(1)
                Q[i] = Q[good[int(d.argmin())]]
        elif suspect.any():
            print(f"[kratos3d {SIDE}] WARNING: every interface node is also on "
                  f"an outer Dirichlet face; no clean reaction exists anywhere "
                  f"on this interface and the exported flux is the raw one.")
    else:
        # ── NEUMANN SIDE: what the CONDITIONS actually assembled ────────────
        # Kratos stores REACTION_FLUX only on FIXED dofs, so the Dirichlet
        # branch's route is closed here — the interface dofs are free and read
        # back as zero.  This branch used to answer that by averaging the
        # surrounding constant P1 tet gradients, an O(h) reconstruction that
        # sets the interface order any reader will measure all by itself.
        #
        # THE ECHO WOULD BE WRONG, even though it is algebraically right.
        # FluxCondition3D3N enforces  K grad T . n = FACE_HEAT_FLUX  and
        # FACE_HEAT_FLUX is the partner's array verbatim, so `-q_in` is exactly
        # this side's outward flux -- and it is worthless as evidence, because
        # it never passes through the solve.  Applied on the wrong facets, with
        # the wrong sign, or with a broken area weight, the echo is unchanged
        # and the two-sided balance check still reports roundoff.  The whole
        # point of that check is that an interface-mechanism mutation, which
        # leaves the self-convergence order at ~1.85 and looks correct, moves
        # the flux jump from roundoff to O(1).
        #
        # So ask the conditions themselves what they contributed.  Summing each
        # condition's own right-hand side is a MEASUREMENT of what entered the
        # linear system: it is int_Gamma g phi_i ds when the interface is built
        # correctly, and it is something else the moment it is not.  Dividing
        # by the same face-area weight w_i the Dirichlet branch uses turns the
        # functional into the density the partner interpolates, and the sign is
        # the same -r/w -- on those rows the assembled interface load IS the
        # residual (A u - b_vol).
        info = mp.ProcessInfo
        rhs_i = np.zeros(max(n.Id for n in mp.Nodes) + 1)
        for cond in mp.Conditions:
            vec = KM.Vector()
            cond.CalculateRightHandSide(vec, info)
            for k, nd in enumerate(cond.GetNodes()):
                rhs_i[nd.Id] += float(vec[k])
        # r = A u - b_vol.  On these rows the discrete equation reads
        # (A u)_i = b_i = b_vol,i + (assembled interface load)_i, so the
        # residual against the volume load IS the assembled interface load,
        # with a PLUS sign; the outward density is then -r_i / w_i, exactly
        # the expression the Dirichlet branch uses.
        r = np.array([rhs_i[int(i)] for i in ids])
        Q = np.zeros(len(ids))
        ok = np.abs(w) > 1e-14 * max(1.0, float(np.max(w)))
        Q[ok] = -r[ok] / w[ok]
        Q_raw = Q.copy()

    # ── CONSERVATION SELF-CHECK: the discrete divergence theorem ─────────────
    # Summing the unconstrained residual r = A u - b over ALL nodes gives
    # sum_i r_i = -sum_i b_i, because sum_i A_ij = int K grad(sum_i phi_i) .
    # grad phi_j = 0 (the phi_i are a partition of unity).  r vanishes on free
    # rows, so over the FIXED rows alone
    #     sum_{fixed} r_i + int_Omega f dOmega + int_Gamma q_applied ds = 0
    # exactly, at round-off, for ANY mesh — where the interface term is present
    # only on the Neumann side (on the Dirichlet side the interface rows are
    # themselves fixed and already counted).  It is not a discretisation check:
    # it fails only if the flux was applied with the wrong sign or magnitude,
    # if the area weights are wrong, or if the load was never assembled.
    fixed_ids = set(outer_ids) | (set(int(i) for i in ids)
                                  if SIDE == "dirichlet" else set())
    react = sum(mp.Nodes[i].GetSolutionStepValue(KM.REACTION_FLUX) for i in fixed_ids)
    load_vol = 0.0
    for el in mp.Elements:
        nds = el.GetNodes()
        p = np.array([[n.X, n.Y, n.Z] for n in nds])
        det = float(np.linalg.det(np.column_stack((p[1] - p[0], p[2] - p[0],
                                                   p[3] - p[0]))))
        load_vol += (abs(det) / 6.0) * sum(
            n.GetSolutionStepValue(KM.HEAT_FLUX) for n in nds) / 4.0
    load_if = 0.0 if q_in is None else float(np.dot(w, q_in))
    imb = abs(react + load_vol + load_if)
    scale = max(abs(react), abs(load_vol), abs(load_if))
    # With no volumetric source and no applied interface flux every term in the
    # identity is separately zero — heat may still be FLOWING, but it flows in
    # and out through Dirichlet faces whose reactions cancel — so the identity
    # reads 0 = 0 and proves nothing.  A RATIO of round-off to round-off is
    # O(1) while meaning nothing, so say so instead of printing a 1.0 that
    # reads as a 100% conservation error.
    if scale <= 1e-10 * K * max(1.0, abs(float(np.max(np.abs(T))))) * float(np.max(LEN)):
        bal = (f"balance trivially satisfied (no source and no applied "
               f"interface flux: the identity reads 0 = 0 and proves nothing "
               f"about the coupling; |imbalance|={imb:.3e})")
    else:
        bal = f"balance {imb:.3e} abs / {imb / scale:.3e} rel"

    print(f"[kratos3d {SIDE}] interface n={len(ids)} tris={len(tris)} "
          f"area={w.sum():.6g} edge_nodes={int(edge.sum())} "
          f"T=[{T.min():.6g},{T.max():.6g}] q=[{Q.min():.6g},{Q.max():.6g}] {bal}")

    # THE RUN-LOG CONTRACT LINE: `NDOF = <integer>` on a line of its OWN.
    # The audit reads that exact shape, and they read it PER
    # LEVEL: it is how anyone checking the result tells a refined mesh from the same mesh run
    # three times. A number inside a prose sentence does not count, and a
    # wrong number is worse than none -- one coupled run that was right in
    # every other respect reported NDOF = 1 at all three levels, and its
    # refined mesh could not be told from an unrefined one.
    try:
        print(f"\nNDOF = {int(len(mp.Nodes))}")
    except Exception as _ndof_exc:
        print(f"[kratos3d] could not report NDOF: {_ndof_exc!r}. Your task's"
              f" execution log needs `NDOF = <integer>` on a line of its own,"
              f" so print your own degree-of-freedom count here.")

    # PER-LEVEL PERSISTENCE: this level's whole field, and its interface trace
    # and flux, named by LEVEL. exports.json is overwritten by the next level;
    # these files are not. Interpolate THESE onto the probe points your task
    # names -- never a file the next level overwrites.
    # A DUMP DEFECT MUST NOT COST YOU THE SOLVE. exports.json is the driver's
    # proof that this participant succeeded, and it is written after these files,
    # so an exception here would throw away a coupling iteration that worked.
    try:
        with open(f"field_level{LEVEL}.csv", "w") as _f:
            _f.write("x,y,z,u\n")
            for _n in mp.Nodes:
                _f.write(f"{float(_n.X):.11e},{float(_n.Y):.11e},{float(_n.Z):.11e},"
                         f"{float(_n.GetSolutionStepValue(KM.TEMPERATURE)):.11e}\n")
        with open(f"interface_level{LEVEL}.csv", "w") as _f:
            _f.write("x,y,z,u,qn\n")
            for _p, _t, _q in zip(pts3, T, Q):
                _f.write(f"{float(_p[0]):.11e},{float(_p[1]):.11e},{float(_p[2]):.11e},"
                         f"{float(_t):.11e},{float(_q):.11e}\n")
    except Exception as _dump_exc:
        # AND KEEP BOTH FILES OR NEITHER. A dump that failed part-way can leave a
        # truncated file, a whole field file with no interface file, or a file an
        # earlier run wrote, and any of them could be read as this level's result.
        # So both of this level's files go, whatever they hold.
        for _partial in (f"field_level{LEVEL}.csv", f"interface_level{LEVEL}.csv"):
            try:
                Path(_partial).unlink(missing_ok=True)
            except OSError:
                pass
        print(f"[kratos_ per-level dump] level {LEVEL} dump failed: "
              f"{_dump_exc!r}. exports.json is still written, so the coupling\n"
              f"continues, but this level has no field file to hand in. Fix the\n"
              f"names the dump reads and run this level again.")
    # ── EXPORT SELF-CHECK ─ keep this block, and note it is NOT the conservation
    #    check above. That one tests the discrete divergence theorem, and it says
    #    so itself when there is no source and no applied load: the identity
    #    reads 0 = 0 and proves nothing. So it passes exactly the case this
    #    catches -- a Neumann side whose imported load never entered the
    #    assembled system, which returns the no-load answer and a flux of ~0
    #    against a nonzero partner. Also a non-finite field, and a flux that is
    #    the partner's array negated rather than recovered from this side's own
    #    system.
    _chk_vals = np.asarray(T, float).ravel()
    _chk_flux = np.asarray(Q, float).ravel()
    if not (np.isfinite(_chk_vals).all() and np.isfinite(_chk_flux).all()):
        raise SystemExit("EXPORT SELF-CHECK: non-finite interface values or "
                         "fluxes; the solve did not produce a usable field, so "
                         "nothing was exported")
    _chk_imp = (json.loads(Path("imports.json").read_text() or "{}")
                if Path("imports.json").is_file() else {})
    _chk_qin = (np.concatenate([np.asarray(_d.get("normal_fluxes") or [], float).ravel()
                                for _d in _chk_imp.values()])
                if _chk_imp else np.zeros(0))
    if SIDE == "neumann" and _chk_qin.size and np.abs(_chk_qin).max() > 0 \
            and np.abs(_chk_flux).max() < 1e-9 * np.abs(_chk_qin).max():
        raise SystemExit("EXPORT SELF-CHECK: the recovered interface flux is ~0 "
                         "against a nonzero imported flux: the imported load "
                         "never entered the assembled system (the facet term / "
                         "boundary condition that integrates it is missing). "
                         "Fix the application; do not couple on")
    if SIDE == "dirichlet" and _chk_qin.shape == _chk_flux.shape and _chk_flux.size \
            and np.array_equal(_chk_flux, -_chk_qin):
        raise SystemExit("EXPORT SELF-CHECK: the exported flux is the partner's "
                         "array negated, bit for bit: a copy, not a recovery "
                         "from this side's own assembled system")

    # exports.json LAST: the driver takes its existence as proof of success.
    Path("exports.json").write_text(json.dumps({
        "field_name": "temperature",
        "n_points": int(len(ids)),
        "coordinates": [[float(p[0]), float(p[1]), float(p[2])] for p in pts3],
        "values": [float(t) for t in T],
        "normal_fluxes": [float(q) for q in Q],
    }, indent=2))

    return {"mp": mp, "nid": nid, "iface_ids": ids, "coords": pts3, "pts": pts,
            "weights": w, "T": T, "q": Q, "q_raw": Q_raw, "edge": edge,
            "tris": tris, "imbalance": imb, "imbalance_rel":
            imb / scale if scale > 0 else 0.0}


if __name__ == "__main__":
    main()
