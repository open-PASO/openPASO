"""DUNE-fem participant for the openPASO `couple` driver — 3-D scalar conduction
across a PLANAR interface.  Serves either side of the split.

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1, so an
iteration-1 fallback is mandatory), writes exports.json LAST and exits 0.
Needs dune-fem importable in the interpreter named in `command` (`pip install
dune-fem` from PyPI; conda-forge has no dune-fem package).  DUNE JIT-COMPILES
ITS UFL FORMS ON FIRST USE, AND THIS FILE'S
FORMS TAKE SEVERAL MINUTES COLD: measured 7 to 12 minutes for its first run on
this install (16 form modules), and 4 to 6 minutes more when the 3-D grid and
its space are not yet in DUNE's cache.  That is not a hang, and a timeout
shorter than that stops it mid-compile: PETSc then prints "Caught signal
number 15 Terminate", which is the timeout ending the run, not a crash.  The
forms here do not depend on NX/NY/NZ, so a mesh-refinement study compiles once
and then reuses the cache.  Two things DO compile anew, measured on this
install: another grid type is another grid and space (aluConformGrid's 3-D
grid module alone took 2.5 minutes), and a number written INTO a UFL
expression is part of its module (moving only the interface plane recompiled
three of this file's modules, 67 s).

Physics: steady conduction  -div(K grad T) = f  on one BOX subdomain of a box
split by a plane.  Structured cube grid, Q1 Lagrange; the interface carries
either a Dirichlet trace or a natural (flux) load.

======================================================================
WHY A SEPARATE 3-D FILE — what is genuinely different, not just bigger
======================================================================
The 2-D participants (participant_dune.py, participant_fenics.py,
participant_kratos*.py) all assume the interface is a LINE at x = const sampled
by ONE coordinate.  Three of their steps are wrong in 3-D and each fails
quietly:

1. ORDERING.  The driver relaxes export vectors ENTRY BY ENTRY, so the export
   order must be identical on every iteration.  In 2-D `argsort(y)` does it.
   In 3-D the interface is a plane and a sort on one coordinate leaves the
   other in whatever order the mesh happened to number its dofs — stable within
   a run, so nothing complains, but the coupling then relaxes point i of this
   iteration against a DIFFERENT physical point of the last one.  Here the
   order is a tolerance-quantised `lexsort` over BOTH in-plane coordinates.

2. RESAMPLING A NON-MATCHING PARTNER.  In 2-D `np.interp` over the tangential
   coordinate is exact for a P1 trace.  In 3-D the partner's interface nodes
   are a 2-D point set: `np.interp` on either axis alone sorts that cloud by
   one coordinate and averages across the other, producing a smooth,
   plausible, WRONG boundary condition.  `resample_plane` below does real 2-D
   interpolation, and CUBIC rather than bilinear where it can — see the
   measurement in `_bicubic`, which is worth half an order of the exported
   flux.

3. QUADRATURE WEIGHTS.  The consistent-flux recovery divides the nodal
   reaction by  w_i = int_Gamma phi_i ds.  In 2-D those are EDGE LENGTHS.  In
   3-D they are FACE AREAS.  DUNE gets them right for free — `assemble` of
   `v*ds` restricted to the interface IS that integral, whatever the cell
   shape — which is precisely why this file assembles them instead of writing
   down a formula, as the 2-D Kratos participant has to.  For scale: carrying
   the 2-D edge-length formula over to a unit-square interface gives an
   exported int_Gamma q ds of -0.923 against an exact -22.026 at n=24, on a
   flux field that is still smooth and still the right shape (it is scaled by
   ~h) — and the error GROWS under refinement.  `assemble` cannot make that
   mistake; the print below reports w.sum(), which must equal the interface
   area, as the standing check that it did not.

THE SIGN, WHICH IS THE ONLY THING A NEUMANN PARTICIPANT CAN GET SILENTLY WRONG.
Every participant exports its outward normal flux density with respect to ITS
OWN outward normal,  q = -(K grad T) . n_own,  so the two sides of a shared
interface carry OPPOSITE signs.  On this subdomain
      int_Omega K grad T . grad v = int_Omega f v + int_dOmega (K grad T . n) v ds,
and on the interface n_own = -n_partner, so  K grad T . n_own = +q_partner:
the partner's number is applied UNCHANGED, as `+ g*v*ds`, no minus sign
anywhere.  If you ever flip that sign to "make the temperatures look right",
you have built a coupling that drives heat the wrong way across the interface
and still converges.  The self-check printed at the end is the guard: it
evaluates the discrete divergence theorem on this subdomain and must come out
at round-off.

`main()` returns a dict of everything a verification script needs (grid view,
space, solution, interface dofs, in-plane points, area weights, guarded and
UNGUARDED fluxes, the edge mask) so the participant can be imported and graded
against a manufactured solution.  The coupling itself only ever needs the file
handshake.

======================================================================
MEASURED — this file did not ship until it converged in a real coupling
======================================================================
Manufactured two-material 3-D conduction: box [0,1]^3 split by the plane
x = 0.5, one constant k on each half, exact Dirichlet on all five non-interface
faces of each half, so the whole RIM of the interface plane is an outer
Dirichlet edge — the 3-D corner case made as large as it can be.

AS THE NEUMANN HALF, in a real Dirichlet-Neumann coupling through
src/core/coupling_driver.run_coupling against participant_kratos_3d.py, on
DELIBERATELY NON-MATCHING interface grids (m here against 3m/4 there).  Aitken,
theta0 = 0.7, tol = 1e-9: converged in 31 / 31 / 32 iterations, both
participants "responsive".  L2(T) over the WHOLE box (both subdomains)
converges at the expected second order over n = 6/12/24.

AS THE DIRICHLET HALF, handed the exact interface trace on a non-matching
partner grid, m = 8/16/32:

  L2(T) this subdomain   2.281e-01  5.751e-02  1.441e-02   order 1.99  2.00
  flux, iface INTERIOR   1.751e+00  4.427e-01  1.110e-01   order 1.98  2.00
  flux, WHOLE interface  2.549e+00  6.489e-01  1.883e-01   order 1.97  1.79
  flux, WHOLE unguarded  3.090e+01  2.202e+01  1.561e+01   order 0.49  0.50
  int_Gamma q ds        +24.2476   +22.5220   +22.1466     (exact +22.02642)

The whole-interface number is dragged down by the RIM, exactly as the two end
nodes do it in 2-D — see the edge guard below.  Quote the INTERIOR order when
grading the recovery and the WHOLE one when grading what the partner receives;
they are different questions.

CONSERVATION: interface area recovered as exactly 1.000000 at every level; the
discrete divergence theorem at round-off (1.6e-13 to 4.3e-12 — looser than the
Kratos side because `scheme` solves with CG, not a direct solve); and the
EXCHANGE balance against the partner converging at second order.
That one is NOT zero and cannot be: resampling between non-matching interface
grids is accurate, not conservative.  If you need it at round-off, match the
interface meshes.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from dune.grid import structuredGrid
from dune.fem import assemble
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.fem.operator import galerkin as operator_galerkin
from dune.ufl import DirichletBC, Constant
from ufl import (TrialFunction, TestFunction, SpatialCoordinate,
                 conditional, dot, ds, dx, grad, gt, lt)

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
SIDE       = "neumann"       # "dirichlet" (import T, export flux) | "neumann"
PARTNER    = "left"          # the partner's `name` in your couple(...) call

X0, X1     = 0.5, 1.0        # this subdomain's box
Y0, Y1     = 0.0, 1.0
Z0, Z1     = 0.0, 1.0

IFACE_AXIS = 0               # interface plane normal: 0=x, 1=y, 2=z
IFACE_POS  = 0.5             # its position; must equal this box's lo or hi on that axis

K          = 2.9             # conductivity of THIS subdomain (constant)
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
# 0 = the low face and 1 = the high face, e.g. "x1" is the plane x = X1.  Every
# face NOT listed here (and not the interface) is natural, i.e. insulated.
# The interface face itself must not appear.
DIRICHLET_FACES = ("x1",)



def F_SRC(x, y, z):
    """Volumetric source f in  -div(K grad T) = f, as a function of position.

    Returns zero as shipped, which is a PLACEHOLDER like every number above.
    A CONSTANT CANNOT REPRESENT A POLYNOMIAL SOURCE: if your problem states
    one, or you derived it from a manufactured solution, a single number here
    silently solves a different problem. With the whole outer boundary
    prescribed and no source, the answer degenerates to the profile between
    the outer values.

    `x`, `y` and `z` are NumPy arrays of the node coordinates, so build the
    answer with NumPy and return an array of the same shape:

        return 3.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y) \
               * np.sin(np.pi * z)
    """
    return np.zeros_like(x)

T_INIT     = 295.0           # iteration-1 fallback interface temperature
Q_INIT     = 0.0             # iteration-1 fallback interface flux density


def source_function(space, xd):
    """Carry F_SRC as a DISCRETE FUNCTION, the way the 2-D sibling does.

    Not a UFL expression: a zero source built symbolically folds to a bare 0,
    and `0*v*dx` is a domainless UFL Zero that assemble() refuses. Dofs are
    run-time data, so editing F_SRC never re-triggers the C++ JIT. Sampling at
    the nodes is the P1 interpolant of the source, an O(h^2) load error — the
    same order as the discretization error itself.
    """
    ffun = space.interpolate(0, name="f_src")
    ffun.as_numpy[:] = np.broadcast_to(
        np.asarray(F_SRC(xd[0], xd[1], xd[2]), float), xd[0].shape)
    return ffun


def outer_value(x, y, z):
    """Dirichlet value on the faces listed in DIRICHLET_FACES.

    x, y, z are NUMPY arrays of dof coordinates (this one is evaluated
    pointwise into a discrete function, not assembled symbolically, so that the
    FORM never changes between iterations and DUNE never re-JITs).  Return a
    scalar for a constant temperature or any expression for a graded one.
    """
    return 300.0 + 0.0 * x
# ─────────────────────────────────────────────────────────────────────────

LO = np.array([X0, Y0, Z0], float)
HI = np.array([X1, Y1, Z1], float)
NE = [int(NX), int(NY), int(NZ)]
AX = int(IFACE_AXIS)
TAN = [a for a in (0, 1, 2) if a != AX]          # the two IN-PLANE axes
LEN = HI - LO
TOL = 1e-9 * float(np.max(LEN))
EPS = 1e-8 * float(np.max(LEN))                  # boundary-indicator tolerance
AXN = "xyz"

# Is the interface this box's HIGH face on AX?  The outward normal there is
# +e_AX, on the LOW face it is -e_AX.
ON_HI = abs(IFACE_POS - HI[AX]) < abs(IFACE_POS - LO[AX])
S = 1.0 if ON_HI else -1.0
IFACE_FACE = f"{AXN[AX]}{1 if ON_HI else 0}"


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
    building the 3-D pair.  Same participant, same manufactured problem, same
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
    dofs.  The driver does no interpolation — non-matching meshes are handled
    here, and in 3-D that means a genuine 2-D interpolation (item 2 of the
    header).  Routes, in order of preference:

      1. TENSOR PRODUCT — the partner's points are exactly unique(u) x
         unique(v).  True whenever the partner meshes a box, which is the
         normal case.  CUBIC on that rectilinear grid (bilinear if either
         direction has fewer than 4 lines).
      2. scipy LinearNDInterpolator on the Delaunay triangulation of the
         partner's points; the few points outside its convex hull are filled
         from the nearest partner point.  This route is piecewise linear and
         carries the order penalty described in `_bicubic` — an unstructured
         partner on a coupling that must deliver a second-order FLUX is a real
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
        print(f"[dune3d {SIDE}] WARNING: 2-D interpolation unavailable ({e}); "
              f"falling back to NEAREST NEIGHBOUR on the interface. That is "
              f"O(h) and WILL flatten the convergence rate — match the meshes "
              f"or install scipy before believing any order measured this way.")
        d = ((np.asarray(pts)[:, None, :] - src[None, :, :]) ** 2).sum(-1)
        return vs[d.argmin(1)]


def order_plane(pts):
    """Deterministic order of interface points over BOTH in-plane coordinates.

    Quantised so that dofs nominally on the same row cannot be split apart by
    round-off into an order that changes with the grid manager's mood.  The
    driver relaxes entry by entry, so this order IS part of the contract.
    """
    q = np.round(np.asarray(pts, float) / max(TOL, 1e-300))
    return np.lexsort((q[:, 1], q[:, 0]))


# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
def face_indicator(x, name):
    """UFL 0/1 indicator of the named box face, e.g. "y1" -> the plane y = Y1."""
    a = AXN.index(name[0])
    pos = HI[a] if name[1] == "1" else LO[a]
    return conditional(lt(abs(x[a] - float(pos)), EPS), 1.0, 0.0)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end


def main():
    if abs(IFACE_POS - LO[AX]) > TOL and abs(IFACE_POS - HI[AX]) > TOL:
        sys.exit(f"IFACE_POS={IFACE_POS} is not a {AXN[AX]}-face of this box "
                 f"[{LO[AX]},{HI[AX]}] — nothing is shared with the partner")
    if IFACE_FACE in DIRICHLET_FACES:
        sys.exit(f"DIRICHLET_FACES lists '{IFACE_FACE}', which IS the interface. "
                 f"The outer condition would overwrite the coupling and the run "
                 f"would still exit 0 with a plausible export.")
    if min(NE) < 1:
        sys.exit(f"NX,NY,NZ = {tuple(NE)}: need at least one element per axis")

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    gridView = structuredGrid(list(LO), list(HI), NE)
    space = lagrange(gridView, order=1)
    x = SpatialCoordinate(space)

    # dof -> coordinate map: nodal interpolation of the coordinate functions
    xd = np.array([np.array(space.interpolate(x[a], name=f"c{a}").as_numpy)
                   for a in range(3)])                       # (3, ndof)

    iface_dofs = np.where(np.abs(xd[AX] - IFACE_POS) < TOL)[0]
    if iface_dofs.size == 0:
        sys.exit(f"no interface dofs at {AXN[AX]}={IFACE_POS}: this subdomain "
                 f"spans [{LO[AX]},{HI[AX]}], nothing is shared with the partner")
    pts_all = xd[:, iface_dofs].T                            # (n,3)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end
    order = order_plane(pts_all[:, TAN])
    iface_dofs = iface_dofs[order]
    pts3 = pts_all[order]
    pts = pts3[:, TAN]

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    outer_mask = np.zeros(xd.shape[1], bool)
    for name in DIRICHLET_FACES:
        a = AXN.index(name[0])
        pos = HI[a] if name[1] == "1" else LO[a]
        outer_mask |= np.abs(xd[a] - pos) < TOL
    outer_dofs = np.where(outer_mask)[0]
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end
    edge = outer_mask[iface_dofs]        # interface dofs ALSO on an outer face

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    u, v = TrialFunction(space), TestFunction(space)
    a_form = Constant(K, name="k_cond") * dot(grad(u), grad(v)) * dx
    fsrc = source_function(space, xd)
    # THE VOLUME LOAD ALONE, kept under its own name.  The Neumann branch below
    # adds the partner's interface term into `b_form`; the flux recovery
    # subtracts THIS, on both sides — subtracting the combined form is what made
    # the reaction look like zero on the Neumann side.
    b_vol = fsrc * v * dx
    b_form = b_vol
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    imp = read_imports()

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    # ── ONE Dirichlet carrier for BOTH the outer faces and (on the Dirichlet
    # side) the interface.  Two overlapping DirichletBCs would leave the RIM of
    # the interface — where the plane meets the outer Dirichlet faces — decided
    # by list order, which is not a decision anyone made.  Here the rule is
    # explicit and documented: on the rim the OUTER condition wins, because
    # that is the modeller's known data while the interface value is only the
    # partner's current iterate.  The FORM never changes between iterations
    # (only this function's dof values do), so DUNE never re-JITs.
    gdir = space.interpolate(0, name="dirichlet_data")
    gd = gdir.as_numpy
    gd[:] = 0.0
    gd[outer_dofs] = outer_value(xd[0][outer_dofs], xd[1][outer_dofs],
                                 xd[2][outer_dofs])

    ind = None
    for name in DIRICHLET_FACES:
        t = face_indicator(x, name)
        ind = t if ind is None else ind + t
    iface_ind = conditional(lt(abs(x[AX] - float(IFACE_POS)), EPS), 1.0, 0.0)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    if SIDE == "dirichlet":
        free = iface_dofs[~edge]
        gd[free] = resample_plane(imp, "values", T_INIT, pts[~edge])
        ind = iface_ind if ind is None else ind + iface_ind
    else:
        # APPLY the partner's number UNCHANGED: + int_Gamma g v ds  (see the
        # sign block in the header).  The carrier is a discrete function so the
        # form is fixed; only its dof values move between iterations.  `b_vol`
        # above still holds the VOLUME load alone — the flux recovery subtracts
        # that, not this, and the distinction is the whole point.
        gflx = space.interpolate(0, name="iface_flux")
        gf = gflx.as_numpy
        gf[:] = 0.0
        q_in = resample_plane(imp, "normal_fluxes", Q_INIT, pts)
        gf[iface_dofs] = q_in
        b_form = b_form + conditional(lt(abs(x[AX] - float(IFACE_POS)), EPS),
                                      gflx * v, 0.0) * ds

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    if ind is None:
        sys.exit("no Dirichlet condition anywhere: a pure-Neumann conduction "
                 "problem is singular and the solve would return an arbitrary "
                 "constant offset with no error raised")
    bcs = [DirichletBC(space, gdir, conditional(gt(ind, 0.5), 1, 0))]

    scheme = galerkin([a_form == b_form] + bcs, solver="cg")
    uh = space.interpolate(0, name="temperature")
    scheme.solve(target=uh)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    # w_i = int_Gamma phi_i ds — FACE AREAS in 3-D, and this assembly is why
    # this file does not have to know that: `v*ds` restricted to the interface
    # IS the integral, whatever the cell shape.
    wfun = assemble(conditional(lt(abs(x[AX] - float(IFACE_POS)), EPS), v, 0.0) * ds)
    w = np.array(wfun.as_numpy)[iface_dofs]

    # THE UNCONSTRAINED RESIDUAL r = A u_h - b_vol, assembled ONCE: BOTH sides'
    # interface flux and BOTH sides' conservation check are the same vector read
    # two ways.  `scheme` cannot supply it — it carries the DirichletBCs and so
    # overwrites exactly the constrained rows that ARE the reaction.  A second
    # operator built from the SAME form MINUS the bcs gives it in one
    # application.  The load it is taken against is the VOLUME one, b_vol, NOT
    # the b_form the scheme solved: see the recovery immediately below.
    op_free = operator_galerkin([a_form == b_vol])        # volume load, no bcs
    rfun = space.interpolate(0, name="residual")
    op_free(uh, rfun)
    rv = np.array(rfun.as_numpy)

    # ── THE CONSISTENT (REACTION) FLUX — ONE FORMULA, BOTH SIDES ────────────
    # WHY NOT AN L2 PROJECTION OF THE GRADIENT.  The gradient of a Q1/P1
    # solution is only O(h) accurate ON the boundary — the superconvergence
    # points are interior — and the boundary trace is exactly what the coupling
    # reads.  Measured against the manufactured solution, a projection converges
    # at ~1 while this recovery converges at ~2.
    #
    # From  a(u,v) - (f,v) = int_dOmega (K grad u . n) v ds
    #                      = -int_Gamma qn v ds
    # it follows that for every basis function phi_i on the interface
    #     int_Gamma qn phi_i ds = -r_i,   r = A u_h - b_vol
    # with r the UNCONSTRAINED residual assembled just above.  Dividing by the
    # FACE-AREA weight w_i turns the functional into a density the partner can
    # interpolate pointwise.
    #
    # THE LOAD IS THE VOLUME ONE, AND THAT IS THE WHOLE DIFFERENCE.  An earlier
    # version used this reaction on the Dirichlet side only and an L2-projected
    # gradient on the Neumann side, reasoning that the Neumann interface dofs
    # are free, so r comes out ~0 there.  That holds only when the residual is
    # taken against a load that ALREADY CONTAINS the interface term.  Against
    # b_vol those same free rows carry exactly the interface functional the
    # partner applied,  (A u - b_vol)_i = int_Gamma g phi_i ds,  and the export
    # comes back as -g — this file's sign convention, the two sides' fluxes
    # cancelling.  On the Dirichlet side there is no interface term, so
    # b_form == b_vol and the two cases are one expression.
    #
    # WHAT IS MEASURED, AND WHAT IS ONLY ALGEBRA. Handing the NEUMANN side a
    # flux and asking for it back is an ASSEMBLY IDENTITY, not a convergence
    # test: on free interface rows r = A u - b_vol IS M_Gamma g, so the export
    # is -(M_Gamma g)/(M_Gamma 1) and its offset from -g is -(h^2/6) g''(y) for
    # ANY correct assembly of ANY equation. The "order 2.00" that used to stand
    # here was read off that fixture; it is a property of the P1 boundary mass
    # matrix, not of this code — a bare NumPy mass matrix reproduces the same
    # numbers with no PDE, no solver and no material in it. That fixture is
    # kept (tests/test_interface_flux_recovery.py) for what it really tests:
    # sign convention, interface weight, facet set, blocked dofs.
    #
    # THE ORDER is measured on the DIRICHLET side against an ANALYTIC interface
    # flux the participant is never handed
    # (tests/test_interface_flux_converges_to_a_known_exact_flux.py). FEniCSx,
    # the 2-D version of this same formulation, 8/16/32/64 uniform meshes, max
    # error over interior interface nodes:
    #   2.889e-01  7.243e-02  1.814e-02  4.556e-03   ORDER 1.996 1.998 1.993
    # and only first order (1.10, 1.06, 1.04) at the interface ends — in 3-D
    # that boundary is the whole rim, handled by the edge guard below. There is
    # NO 3-D measurement of this order, and none is claimed here.
    #
    # THE RETIRED L2-PROJECTED GRADIENT, in the norms it was measured in: order
    # ~1 in the interior AWAY FROM THE ENDS (0.93), 0.50 in rms, and
    # non-convergent in the max norm that includes the near-end nodes, where it
    # stalls at 2.6 against a true flux of size 2 to 5. It was written up as a
    # flat "order 0.00, it never converges", which was true of one norm only.
    # Not re-measured since the branch was deleted.
    r = rv[iface_dofs]

    Q = np.zeros(len(iface_dofs))
    ok = np.abs(w) > 1e-14 * max(1.0, float(np.max(np.abs(w))))
    Q[ok] = -r[ok] / w[ok]
    Q_raw = Q.copy()

    # ── THE EDGE GUARD — in 3-D the "corner" is a whole EDGE ────────────────
    # An interface dof that ALSO lies on an outer Dirichlet face carries the
    # OUTER reaction as well, so its residual is not this interface's flux.
    # That is true on BOTH sides: the rim is Dirichlet-constrained whichever
    # role this subdomain plays, because the outer condition wins there.  In
    # 2-D the affected set is TWO nodes.  In 3-D the interface is a plane and
    # it is the entire RIM: 4n of (n+1)^2 dofs — 96 of 625 (15%) at n=24,
    # against 2 of 25 (8%) in 2-D at the same n.  MEASURED on the manufactured
    # problem, leaving them raw costs the whole-interface flux its order
    # outright (0.50 instead of 1.8) and inflates its L2 error by 83x at n=24
    # (15.6 against 0.188).  They are replaced by the nearest interior
    # interface dof — nearest in the PLANE, over both coordinates, not along
    # one of them.
    suspect = edge | ~ok
    good = np.where(~suspect)[0]
    if len(good):
        gp = pts[good]
        for i in np.where(suspect)[0]:
            d = ((gp - pts[i]) ** 2).sum(1)
            Q[i] = Q[good[int(d.argmin())]]
    elif suspect.any():
        print(f"[dune3d {SIDE}] WARNING: every interface dof is also on an "
              f"outer Dirichlet face; no clean reaction exists anywhere on "
              f"this interface and the exported flux is the raw one.")

    T = np.array(uh.as_numpy)[iface_dofs]

    # ── CONSERVATION SELF-CHECK: the discrete divergence theorem ─────────────
    # Summing the unconstrained residual of the FULL system, A u - b_form, over
    # ALL dofs gives -sum_i (b_form)_i, because sum_i A_ij = int K grad(sum_i
    # phi_i) . grad phi_j = 0 (the phi_i are a partition of unity).  That
    # residual vanishes on free rows, so over the CONSTRAINED rows alone
    #     sum_{fixed} r_i + int_Omega f dOmega + int_Gamma q_applied ds = 0
    # exactly, at round-off, for ANY mesh — the interface term being present
    # only on the Neumann side (on the Dirichlet side those rows are themselves
    # fixed and already counted).  It is not a discretisation check: it fails
    # only if the flux was applied with the wrong sign or magnitude, if the
    # area weights are wrong, or if the load was never assembled.
    #
    # `rv` above is taken against the VOLUME load, which is what makes the flux
    # recovery work on both sides, so the applied interface load has to be put
    # back HERE as a VECTOR — not just as its sum — before the constrained rows
    # can be read as reactions: an interface dof on the rim is a constrained row
    # that also carries part of that load.  On the Dirichlet side there is no
    # interface term and `gvec` is zero, so this is the same statement as ever.
    gvec = np.zeros_like(rv)
    if SIDE != "dirichlet":
        gvec = np.array(assemble(
            conditional(lt(abs(x[AX] - float(IFACE_POS)), EPS), gflx * v, 0.0)
            * ds).as_numpy)
    fixed = outer_mask.copy()
    if SIDE == "dirichlet":
        fixed[iface_dofs] = True
    react = float((rv - gvec)[fixed].sum())
    load_vol = float(np.array(assemble(b_vol).as_numpy).sum())
    load_if = float(gvec.sum())
    imb = abs(react + load_vol + load_if)
    scale = max(abs(react), abs(load_vol), abs(load_if))
    # With no volumetric source and no applied interface flux every term in the
    # identity is separately zero — heat may still be FLOWING, but it flows in
    # and out through Dirichlet faces whose reactions cancel — so the identity
    # reads 0 = 0 and proves nothing.  A RATIO of round-off to round-off is
    # O(1) while meaning nothing, so say so instead of printing a 1.0 that
    # reads as a 100% conservation error.
    if scale <= 1e-10 * K * max(1.0, float(np.max(np.abs(T)))) * float(np.max(LEN)):
        bal = (f"balance trivially satisfied (no source and no applied "
               f"interface flux: the identity reads 0 = 0 and proves nothing "
               f"about the coupling; |imbalance|={imb:.3e})")
    else:
        bal = f"balance {imb:.3e} abs / {imb / scale:.3e} rel"

    print(f"[dune3d {SIDE}] interface n={len(iface_dofs)} area={w.sum():.6g} "
          f"edge_dofs={int(edge.sum())} T=[{T.min():.6g},{T.max():.6g}] "
          f"q=[{Q.min():.6g},{Q.max():.6g}] {bal}")

    # THE RUN-LOG CONTRACT LINE: `NDOF = <integer>` on a line of its OWN.
    # The audit reads that exact shape, and they read it PER
    # LEVEL: it is how anyone checking the result tells a refined mesh from the same mesh run
    # three times. A number inside a prose sentence does not count, and a
    # wrong number is worse than none -- one coupled run that was right in
    # every other respect reported NDOF = 1 at all three levels, and its
    # refined mesh could not be told from an unrefined one.
    try:
        print(f"\nNDOF = {int(len(np.array(uh.as_numpy)))}")
    except Exception as _ndof_exc:
        print(f"[dune3d] could not report NDOF: {_ndof_exc!r}. Your task's"
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
        _uh = np.array(uh.as_numpy)
        with open(f"field_level{LEVEL}.csv", "w") as _f:
            _f.write("x,y,z,u\n")
            for _i in range(len(_uh)):
                _f.write(f"{float(xd[0][_i]):.11e},{float(xd[1][_i]):.11e},"
                         f"{float(xd[2][_i]):.11e},{float(_uh[_i]):.11e}\n")
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
        print(f"[dune_ per-level dump] level {LEVEL} dump failed: "
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
        "n_points": int(len(iface_dofs)),
        "coordinates": [[float(p[0]), float(p[1]), float(p[2])] for p in pts3],
        "values": [float(t) for t in T],
        "normal_fluxes": [float(q) for q in Q],
    }, indent=2))

    return {"gridView": gridView, "space": space, "uh": uh, "x": x,
            "iface_dofs": iface_dofs, "coords": pts3, "pts": pts,
            "weights": w, "T": T, "q": Q, "q_raw": Q_raw, "edge": edge,
            "imbalance": imb, "imbalance_rel": imb / scale if scale > 0 else 0.0}


if __name__ == "__main__":
    main()
