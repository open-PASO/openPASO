"""DUNE-fem VECTOR participant for the openPASO `couple` driver.

Plane-strain linear elasticity  -div(sigma(u)) = f  on ONE rectangular
subdomain of a domain split by a straight interface at x = IFACE_X (or
y = IFACE_X when IFACE_AXIS is "y"). Unlike the
scalar (heat) participants, the exchanged interface state is a VECTOR on BOTH
channels:

    values        = displacement       u = (u_x, u_y)   at the interface nodes
    normal_fluxes = interface traction export            (SIGN CONVENTION below)

Both lists therefore carry ONE ENTRY PER COMPONENT per interface point —
`[[vx, vy], ...]`, exactly like participant_fenics_elastic.py /
participant_skfem_elastic.py / participant_ngsolve_elastic.py. A flat list of
scalars is a different interface and the partner will silently mis-read it.

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1, so an
iteration-1 fallback is required), writes exports.json LAST, exits 0. The same
points in the same order every iteration — the driver relaxes export vectors
element by element and refuses a length that changes.

SIGN CONVENTION — the thing a vector coupling gets wrong silently.
`normal_fluxes` is exported as

    q_out = -(sigma . n_own)                       n_own = S * e_x

the SAME convention the shipped scalar participants use for heat
(q_out = -k dT/dn_own) and the one the other *_elastic participants use. Two
consequences, both load-bearing:

  * the two sides' exports CANCEL componentwise, because n_own is anti-parallel
    across the interface — that is what makes the interface balance check a
    conservation statement rather than an accident;
  * the NEUMANN side applies the partner's numbers UNCHANGED, as
    `+ dot(g, v) * ds`, because the natural boundary term of the elasticity
    weak form is +(sigma . n_own) . v = +q_out_partner . v.

Exporting the raw traction (sigma . n_own) instead flips the sign the Neumann
side applies; the iteration still converges, to the wrong answer.

WHICH BLOCK TAKES THE DIRICHLET ROLE IS NOT A FREE CHOICE. The
Dirichlet-Neumann iteration contracts only while theta < 2/(1 + mu_max), with
mu the spectrum of S_N^-1 S_D — for two blocks of the same shape, roughly the
stiffness ratio of the DIRICHLET block to the NEUMANN one. Put the STIFFER
block on the Dirichlet side and that bound drops below 1, i.e. below the top of
the driver's Aitken clamp, and the residual plateaus instead of falling.
Measured with this file against participant_skfem_elastic.py on a 2:1
shear-modulus contrast: stiffer-block-Dirichlet PLATEAUED at 3.9e-06 after 300
iterations (Aitken pinned at theta = 0.69, the stability limit); the identical
problem with the SOFTER block on the Dirichlet side reached 9.3e-11 in 63.
Nothing in either participant changes — only which one is given SIDE =
"dirichlet". If a coupling here stalls at a residual that will not fall, swap
the roles before touching theta.

DUNE-FEM SPECIFICS that this file exists to get right:

  * a VECTOR Lagrange space is `lagrange(gridView, order=1, dimRange=2)`, and
    its `as_numpy` vector is INTERLEAVED: the scalar index of component c at
    node n is n*2 + c. Nothing in the API says so, and a blocked (all x, then
    all y) reading produces an interface of the right length whose every number
    is wrong, so the layout is ASSERTED at run time below rather than assumed.
  * `structuredGrid` gives a cube grid, i.e. Q1 elements here. The interface
    nodes are the line of nodes at x = IFACE_X (y = IFACE_X for axis "y").
  * DUNE compiles the UFL forms with a C++ JIT on first use. The first run of a
    new form takes tens of seconds to minutes; that is NOT a hang. Every form
    below is therefore built ONCE and kept fixed across coupling iterations —
    the changing interface data lives in the DOFs of a discrete function, never
    in the form — so only the first iteration pays the compile.
  * DUNE's DirichletBC is evaluated per boundary intersection, so two BCs whose
    indicators overlap at a corner node fight over that node in an unspecified
    order. This file therefore uses exactly ONE DirichletBC whose value is a
    single discrete function carrying BOTH the outer data and (on the Dirichlet
    side) the imported interface data. There is nothing left to overlap.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from dune.fem import assemble
from dune.fem.operator import galerkin as operator_galerkin
from dune.fem.scheme import galerkin
from dune.fem.space import lagrange
from dune.grid import structuredGrid
from dune.ufl import DirichletBC
from ufl import (Identity, SpatialCoordinate, TestFunction, TrialFunction,
                 as_vector, conditional, dot, ds, dx, grad, inner, lt, sym, tr)

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side.
SIDE      = "dirichlet"   # "dirichlet" (import u, export traction) | "neumann"
PARTNER   = "right"       # the partner's `name` in your couple(...) call
X0, X1    = 0.0, 0.55     # this subdomain's x-extent
Y0, Y1    = 0.0, 0.4      # this subdomain's y-extent
IFACE_AXIS = "x"          # WHICH straight line the interface is: "x" -> the line x = IFACE_X
                          # (the subdomains sit side by side) | "y" -> the line y = IFACE_X
                          # (they are stacked). Everything below follows from it.
IFACE_X   = 0.55          # the shared interface: X0/X1 for axis "x", Y0/Y1 for axis "y"
E_MOD     = 870.0         # Young's modulus
NU        = 0.29          # Poisson ratio (PLANE STRAIN)
# Prescribed displacement on this subdomain's WHOLE non-interface boundary
# (every face but the interface), as a polynomial in (x, y):
#     u_x = UDX[0] + UDX[1]*x + UDX[2]*y + UDX[3]*y*y
#     u_y = UDY[0] + UDY[1]*x + UDY[2]*y + UDY[3]*y*y
# The two subdomains must agree at the two interface corners, or the coupled
# problem is not the un-split one.
UDX = (0.0, 0.0, 0.0, 0.0)
UDY = (0.0, 0.0, 0.0, 0.0)


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
    """
    return np.zeros_like(x), np.zeros_like(y)
NX, NY    = 46, 26        # this subdomain's OWN mesh; need not match the partner
# ── THE PROBLEM'S DATA ARE DATA, NOT CODE (served). config.json may carry this
#    subdomain's box, interface, material, outer displacement and body force AS
#    THE TASK WRITES THEM -- side, partner; x0, x1, y0, y1; iface ("left"|"right"|"bottom"|"top",
#    or the coordinate of the interface line) and iface_axis ("x"|"y"); E and nu,
#    or lam and mu; udx, udy (the four polynomial coefficients of the outer
#    displacement); source_ux, source_uy as strings in x and y (`^` allowed) --
#    and when it does they override the constants and the B_SRC body above.
#    Measured on the thermo-elastic family: the side written as CODE solved a
#    textbook sine source while its own config.json held the task's polynomials;
#    a source typed twice is transcribed once wrong. The audit's momentum check
#    reads the same keys, so a side that states them is the side it can judge.
def _expr_fn(expr):
    """A NumPy function of (x, y) from an expression string as a task writes it."""
    src = str(expr).replace("^", "**")
    code = compile(src, "<source>", "eval")
    names = {"pi": np.pi, "sin": np.sin, "cos": np.cos, "exp": np.exp, "sqrt": np.sqrt,
             "abs": np.abs, "log": np.log, "tanh": np.tanh, "cosh": np.cosh, "sinh": np.sinh}
    def f(x, y):
        env = dict(names); env["x"] = x; env["y"] = y
        return eval(code, {"__builtins__": {}}, env) + 0.0 * x
    return f
_cfg_all = {}
for _src, _txt in (("config.json", Path("config.json").read_text() if Path("config.json").is_file() else ""),
                   ("OPENPASO_CONFIG_JSON", os.environ.get("OPENPASO_CONFIG_JSON", ""))):
    try:
        _cfg_all.update(**json.loads(_txt or "{}"))
    except (ValueError, TypeError) as _e:
        raise SystemExit(f"{_src} could not be read ({_e}); nothing was solved")
if all(_k in _cfg_all for _k in ("x0", "x1", "y0", "y1")):
    X0, X1, Y0, Y1 = (float(_cfg_all[_k]) for _k in ("x0", "x1", "y0", "y1"))
if str(_cfg_all.get("iface_axis", "")).strip().lower()[:1] in ("x", "y"):
    IFACE_AXIS = str(_cfg_all["iface_axis"]).strip().lower()[:1]
_ifc = str(_cfg_all.get("iface", "")).strip().lower()
if _ifc in ("left", "right", "bottom", "top"):
    IFACE_AXIS = ("x" if _ifc in ("left", "right") else "y")
    IFACE_X = {"left": X0, "right": X1, "bottom": Y0, "top": Y1}[_ifc]
elif _ifc:
    try:
        IFACE_X = float(_ifc)
    except ValueError:
        pass
if str(_cfg_all.get("side", "")).strip().lower() in ("dirichlet", "neumann"):
    SIDE = str(_cfg_all["side"]).strip().lower()
if str(_cfg_all.get("partner", "")).strip():
    PARTNER = str(_cfg_all["partner"]).strip()
if "E" in _cfg_all and "nu" in _cfg_all:
    E_MOD, NU = float(_cfg_all["E"]), float(_cfg_all["nu"])
elif ("lam" in _cfg_all or "lambda" in _cfg_all) and "mu" in _cfg_all:
    _lam, _mu = float(_cfg_all.get("lam", _cfg_all.get("lambda"))), float(_cfg_all["mu"])
    E_MOD, NU = _mu * (3.0 * _lam + 2.0 * _mu) / (_lam + _mu), _lam / (2.0 * (_lam + _mu))
for _nm, _key in (("UDX", "udx"), ("UDY", "udy")):
    if isinstance(_cfg_all.get(_key), (list, tuple)) and len(_cfg_all[_key]) == 4:
        globals()[_nm] = tuple(float(_c) for _c in _cfg_all[_key])
_SOURCES_FROM = "code (the B_SRC body above)"
if _cfg_all.get("source_ux") is not None and _cfg_all.get("source_uy") is not None:
    _bx_cfg, _by_cfg = _expr_fn(_cfg_all["source_ux"]), _expr_fn(_cfg_all["source_uy"])
    def B_SRC(x, y):                                   # noqa: F811 -- config wins over the body above
        return _bx_cfg(x, y), _by_cfg(x, y)
    _SOURCES_FROM = "config.json"
print(f"SOURCES IN USE: from {_SOURCES_FROM}"
      + (f"; b_x = {str(_cfg_all.get('source_ux'))[:60]}; b_y = {str(_cfg_all.get('source_uy'))[:60]}"
         if _cfg_all.get("source_ux") is not None else "; b = the B_SRC body above (config carries no source_ux/source_uy)"))

UI_X, UI_Y = 0.0, 0.0     # iteration-1 fallback interface displacement
TI_X, TI_Y = 0.0, 0.0     # iteration-1 fallback interface traction export
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
EXTENT = max(X1 - X0, Y1 - Y0)
TOL = 1e-9 * EXTENT                        # node-coordinate comparisons
EPS = 1e-8 * EXTENT                        # boundary-indicator width in the form



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
    """imports.json is {partner_name: InterfaceData}; it is `{}` on iteration 1,
    so the caller must fall back to an initial guess."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return _partner_block(json.loads(p.read_text()))
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

    Returns (len(y), ncomp). `fallback` is the per-component constant used on
    iteration 1, when imports.json is `{}`."""
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


imp = read_imports()

# ── THE PER-LEVEL RULE (served). A ./config.json {"level": k, "nx": .., "ny": ..}
#    next to this script overrides the mesh knobs and names the level. The dumps
#    at the foot of this file carry that level in their NAME, so a mesh study
#    leaves one file per level instead of the fine mesh overwriting the coarse.
LEVEL = 1
_cfg = _cfg_all   # config.json and OPENPASO_CONFIG_JSON, read and checked above
try:
    LEVEL = int(_cfg.get("level", LEVEL))
    NX = int(_cfg.get("nx", NX))
    NY = int(_cfg.get("ny", NY))
except (ValueError, TypeError) as _cfg_exc:
    raise SystemExit(f"config.json / OPENPASO_CONFIG_JSON: the level and the mesh keys nx, ny "
                     f"could not be read ({_cfg_exc!r}); fix them, nothing was solved")

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
gridView = structuredGrid([X0, Y0], [X1, Y1], [NX, NY])
space = lagrange(gridView, order=1, dimRange=2)
x = SpatialCoordinate(space)

# ── dof -> (node, component) map ──────────────────────────────────────────
# Nodal interpolation of the coordinate functions gives the coordinate of every
# scalar dof; the component pattern is read off a marker field rather than
# assumed. A blocked layout (all x-dofs, then all y-dofs) would pass every
# length check downstream and put the y displacement where the partner reads
# the x one, so this refuses to run rather than export interleaved nonsense.
xy = np.array(space.interpolate(as_vector([x[0], x[1]]), name="coords").as_numpy)
mark = np.array(space.interpolate(as_vector([1.0, 2.0]), name="mark").as_numpy)
if not (np.allclose(mark[0::2], 1.0) and np.allclose(mark[1::2], 2.0)):
    sys.exit("dune-fem vector dof layout is not the expected interleaved "
             "node*2+component; the interface export would be scrambled")
xd, yd = xy[0::2], xy[1::2]                # per-NODE coordinates
pa, pl = (xd, yd) if AX == 0 else (yd, xd) # across the interface, and along it

iface_n = np.where(np.abs(pa - IFACE_X) < TOL)[0]
if len(iface_n) == 0:
    sys.exit(f"no interface dofs at {'xy'[AX]}={IFACE_X}: this subdomain spans "
             f"[{LO},{HI}] in {'xy'[AX]}, so nothing is shared with the partner")
iface_n = iface_n[np.argsort(pl[iface_n])]              # constant order, always
y_if = pl[iface_n]
outer_n = np.where((np.abs(pa - OUTER_X) < TOL) |
                   (np.abs(pl - ALO) < TOL) | (np.abs(pl - AHI) < TOL))[0]
# THE TWO INTERFACE CORNERS BELONG TO THE OUTER BOUNDARY, ON BOTH SIDES.
# The two ends of the interface sit on the faces it ends on, which carry a
# prescribed displacement in the un-split problem, so they stay Dirichlet in
# BOTH subproblems. Handing them to the interface leaves them unconstrained on
# the Neumann side: that subproblem is still well posed, still converges, and
# lands a few percent off — measured on the sibling participants, 4.7% in the
# interface displacement and 28% in the interface traction, on a coupling whose
# residual reached 1e-10 and whose flux balanced. They are still EXPORTED; they
# are just not interface-imposed.
corner = (np.abs(y_if - ALO) < TOL) | (np.abs(y_if - AHI) < TOL)
iface_bc_n = iface_n[~corner]

u, v = TrialFunction(space), TestFunction(space)


def eps_(w):
    return sym(grad(w))


def sigma(w):
    return 2.0 * MU * eps_(w) + LAM * tr(eps_(w)) * Identity(2)


a = inner(sigma(u), eps_(v)) * dx

# BODY FORCE. B_SRC is sampled at the nodes and carried by a DISCRETE FUNCTION,
# the same device this file uses for the Dirichlet and the interface data, for
# two DUNE-specific reasons. A UFL expression built straight out of B_SRC folds
# to a bare 0 when B_SRC returns zero, and `0*v*dx` is a domainless UFL Zero
# that assemble() cannot integrate (the trap the scalar DUNE participant
# documents for its source term); a discrete function always carries its grid,
# so the shipped zero assembles like any other value. And its dofs are run-time
# data, so editing B_SRC never re-triggers the C++ JIT. order=1, so this is the
# P1 interpolant of the source; its quadrature error is O(h^2), the same order
# as the discretization error itself. Do NOT hand B_SRC the symbolic
# SpatialCoordinate `x` instead: a UFL expression carries no NumPy ufuncs, so
# np.sin(x) raises and np.zeros_like(x) returns a 0-d OBJECT array — the source
# collapses to a constant and this subdomain solves the wrong problem with no
# error raised.
bfun = space.interpolate(as_vector([0.0, 0.0]), name="body_force")
bdofs = bfun.as_numpy
bx, by = B_SRC(xd, yd)                     # xd, yd are the per-NODE coordinates
bdofs[0::2] = np.broadcast_to(np.asarray(bx, float), xd.shape)
bdofs[1::2] = np.broadcast_to(np.asarray(by, float), yd.shape)
# THE VOLUME LOAD ALONE, kept under its own name. The Neumann branch below adds
# the partner's interface term into `b`; the traction recovery at the bottom
# subtracts THIS, on both sides — subtracting the combined form is what made the
# reaction look like zero on the Neumann side.
b_vol = dot(bfun, v) * dx
b = b_vol

# ── boundary indicators ───────────────────────────────────────────────────
on_outer = conditional(lt(abs(x[AX] - OUTER_X), EPS), 1,
                       conditional(lt(abs(x[AL] - ALO), EPS), 1,
                                   conditional(lt(abs(x[AL] - AHI), EPS), 1, 0)))
on_iface = conditional(lt(abs(x[AX] - IFACE_X), EPS), 1, 0)

# ONE Dirichlet carrier for BOTH the outer data and the imported interface
# displacement. Its dofs change every iteration; the FORM never does.
gfun = space.interpolate(as_vector([0.0, 0.0]), name="dirichlet_data")
gdofs = gfun.as_numpy
gdofs[:] = 0.0
ox, oy = xd[outer_n], yd[outer_n]
gdofs[2 * outer_n] = UDX[0] + UDX[1] * ox + UDX[2] * oy + UDX[3] * oy * oy
gdofs[2 * outer_n + 1] = UDY[0] + UDY[1] * ox + UDY[2] * oy + UDY[3] * oy * oy
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

# ── WHAT THE SERVED LINES BELOW RELY ON, CHECKED (served) ─ keep this block.
#    y_if is the coordinate ALONG the interface: x on a horizontal one (the
#    name is the vertical case's). iface_bc_n is iface_n without its two end
#    nodes, in the same order. The boundary indicators below are written on
#    AX and AL, so they follow IFACE_AXIS.
y_if = np.asarray(y_if, float)
if (y_if.size != len(iface_n) or y_if.size < 2 or np.any(np.diff(y_if) <= 0)
        or abs(y_if[0] - ALO) > TOL or abs(y_if[-1] - AHI) > TOL):
    raise SystemExit(f"INTERFACE NODES: y_if must hold the coordinate ALONG the interface ({'xy'[AL]}), "
                     f"one per node of iface_n in the same order, strictly increasing from {ALO:g} to "
                     f"{AHI:g}; it holds {y_if.size} value(s) for {len(iface_n)} node(s)"
                     + (f", from {y_if.min():g} to {y_if.max():g}" if y_if.size else "")
                     + (f"; only {np.unique(np.round(y_if, 12)).size} of them distinct -- a node "
                        f"listed once per edge it touches is listed twice"
                        if 0 < np.unique(np.round(y_if, 12)).size < y_if.size else ""))
_ends = (np.abs(y_if - ALO) <= TOL) | (np.abs(y_if - AHI) <= TOL)   # the interface's two ends
if not np.array_equal(np.asarray(iface_bc_n), np.asarray(iface_n)[~_ends]):
    raise SystemExit("INTERFACE NODES: iface_bc_n must be iface_n without the interface's two end "
                     "nodes, in the same order: the Dirichlet branch below writes the partner's values "
                     "into it row by row")

if SIDE == "dirichlet":
    u_if = sample(imp, "values", (UI_X, UI_Y), y_if)
    gdofs[2 * iface_bc_n] = u_if[~_ends, 0]
    gdofs[2 * iface_bc_n + 1] = u_if[~_ends, 1]
    # the interface corners keep the OUTER value already written above
    bc_where = conditional(lt(abs(x[AX] - OUTER_X), EPS), 1,
                           conditional(lt(abs(x[AL] - ALO), EPS), 1,
                                       conditional(lt(abs(x[AL] - AHI), EPS), 1,
                                                   conditional(lt(abs(x[AX] - IFACE_X), EPS), 1, 0))))
else:
    t_if = sample(imp, "normal_fluxes", (TI_X, TI_Y), y_if)
    tfun = space.interpolate(as_vector([0.0, 0.0]), name="iface_traction")
    tdofs = tfun.as_numpy
    tdofs[:] = 0.0
    tdofs[2 * iface_n] = t_if[:, 0]
    tdofs[2 * iface_n + 1] = t_if[:, 1]
    # APPLY the partner's numbers UNCHANGED (+ integral(g . v) ds_interface).
    # `b_vol` above still holds the VOLUME load alone — the traction recovery
    # subtracts that, not this, and the distinction is the whole point.
    b = b + conditional(lt(abs(x[AX] - IFACE_X), EPS), dot(tfun, v), 0.0) * ds
    bc_where = conditional(lt(abs(x[AX] - OUTER_X), EPS), 1,
                           conditional(lt(abs(x[AL] - ALO), EPS), 1,
                                       conditional(lt(abs(x[AL] - AHI), EPS), 1, 0)))

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
scheme = galerkin([a == b, DirichletBC(space, gfun, bc_where)], solver="cg")
uh = space.interpolate(as_vector([0.0, 0.0]), name="displacement")
scheme.solve(target=uh)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

# Interface traction export q_out = -(sigma . n_own).
#
# WHY NOT AN L2 PROJECTION OF THE STRESS. That is what this file and its
# siblings used to do on the Neumann side: project -(sigma(u_h) . n_own) over
# the whole subdomain and sample it at the interface. The gradient of a P1/Q1
# solution — and therefore the stress — is only O(h) accurate ON the boundary;
# the superconvergence points are interior, and the boundary trace is exactly
# what the coupling reads. Measured against a manufactured solution with a known
# exact interface traction, the projection converges at order ~1 while the
# consistent traction below converges at ~2, so the recovery, not the physics
# and not the partner, was setting the answer.
#
# THE CONSISTENT (REACTION) TRACTION — ONE FORMULA, BOTH SIDES. From
#     a(u,v) - (f,v) = int_dOmega (sigma(u) . n) . v ds = -int_Gamma q_out . v ds
# (the second equality is this file's sign convention, q_out = -(sigma . n_own))
# it follows that for every vector basis function phi_i on the interface
#     int_Gamma q_out . phi_i ds = -r_i,   r = A u_h - b_vol
# with r the UNCONSTRAINED residual and b_vol the VOLUME load ALONE: the body
# force and nothing from the interface. Drop the body force and the reaction is
# wrong by the load it carries; leave the interface term in and the Neumann side
# reads back zero.
#
# THAT SECOND FAILURE IS WHY THERE IS NO LONGER A SECOND BRANCH HERE. An earlier
# version used this reaction on the Dirichlet side and the projected stress on
# the Neumann side, reasoning that the Neumann interface dofs are free, so r
# comes out ~0 there. That holds only when the residual is taken against a load
# that ALREADY CONTAINS the interface term. Against b_vol those same rows carry
# exactly the interface functional the partner applied, int_Gamma t . phi_i ds,
# so the export comes back as -t — this file's sign convention, the two sides'
# tractions cancelling. On the Dirichlet side there is no interface term, so
# b == b_vol and the two cases are one expression.
#
# WHAT IS MEASURED, AND WHAT IS ONLY ALGEBRA. Handing the NEUMANN side a
# traction and asking for it back is an ASSEMBLY IDENTITY, not a convergence
# test: on free interface rows r = A u - b_vol IS M_Gamma g, so the export is
# -(M_Gamma g)/(M_Gamma 1) and its offset from -g is -(h^2/6) g''(y) for ANY
# correct assembly of ANY equation. The "order 2.00" that used to stand here
# was read off that fixture; it is a property of the P1 boundary mass matrix,
# not of this code — a bare NumPy mass matrix reproduces the same numbers with
# no PDE, no solver and no material in it. That fixture is kept
# (tests/test_interface_flux_recovery.py) for what it really tests: sign
# convention, interface weight, facet set, and the interleaved-dof mapping
# below.
#
# THE ORDER is measured on the DIRICHLET side against an ANALYTIC interface
# flux the participant is never handed
# (tests/test_interface_flux_converges_to_a_known_exact_flux.py). FEniCSx, the
# scalar conduction version of this same formulation, 8/16/32/64 uniform
# meshes, max error over interior interface nodes:
#     2.889e-01  7.243e-02  1.814e-02  4.556e-03   ORDER 1.996  1.998  1.993
# and only first order (1.10, 1.06, 1.04) at the two nodes where the interface
# meets the outer boundary. There is no VECTOR measurement against an analytic
# traction, and none is claimed.
#
# THE RETIRED L2-PROJECTED GRADIENT, in the norms it was measured in: order ~1
# in the interior AWAY FROM THE ENDS (0.93), 0.50 in rms, and non-convergent in
# the max norm that includes the near-end nodes, where it stalls at 2.6 against
# a true flux of size 2 to 5. It was written up as a flat "order 0.00, it never
# converges", which was true of one norm only. Not re-measured since the branch
# was deleted.
#
# `scheme` cannot supply r: it carries the DirichletBC and so overwrites exactly
# the constrained rows that ARE the reaction. A second operator built from the
# SAME forms MINUS the DirichletBC gives the unconstrained residual in one
# application. Taking the test vector (1,1) in the weight form makes
# w_i = int_Gamma phi_i ds the SCALAR nodal weight of the NODE, the same number
# in both of its interleaved dof slots, so one componentwise division by wt[idx]
# divides every component of a node by that node's weight and turns the
# functional into a density the partner can interpolate pointwise.
op_free = operator_galerkin([a == b_vol])   # volume load, no DirichletBC
rfun = space.interpolate(as_vector([0.0, 0.0]), name="residual")
op_free(uh, rfun)                           # r = A u_h - b_vol
r = np.array(rfun.as_numpy)

wfun = assemble(conditional(lt(abs(x[AX] - IFACE_X), EPS),
                            v[0] + v[1], 0.0) * ds)
wt = np.array(wfun.as_numpy)                # w_i = int_Gamma phi_i ds

idx = np.column_stack([2 * iface_n, 2 * iface_n + 1])
wi = wt[idx]
Q = np.zeros_like(wi)
ok = np.abs(wi) > 1e-14
Q[ok] = -r[idx][ok] / wi[ok]

# THE TWO INTERFACE CORNERS ARE ON THE OUTER DIRICHLET BOUNDARY (the faces the
# interface ends on) ON BOTH SIDES, so their rows carry the OUTER reaction too
# and their residual is not this interface's traction. Take the nearest interior
# interface node rather than exporting a corner value that is physically a
# different quantity. They are found by position (_ends), so an outer_n that
# leaves them out does not let them through.
suspect = _ends | np.isin(iface_n, outer_n) | ~ok.all(axis=1)
good = np.where(~suspect)[0]
if len(good):
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]

ud = np.array(uh.as_numpy)
U = np.column_stack([ud[2 * iface_n], ud[2 * iface_n + 1]])
print(f"[dune {SIDE}] interface n={len(U)} "
      f"ux=[{U[:, 0].min():.6g},{U[:, 0].max():.6g}] "
      f"uy=[{U[:, 1].min():.6g},{U[:, 1].max():.6g}] "
      f"tx=[{Q[:, 0].min():.6g},{Q[:, 0].max():.6g}] "
      f"ty=[{Q[:, 1].min():.6g},{Q[:, 1].max():.6g}]")

# exports.json LAST: the driver takes its existence as proof of success.
# THE RUN-LOG CONTRACT LINE: `NDOF = <integer>` on a line of its OWN.
# The audit reads that exact shape, and they read it PER
# LEVEL: it is how anyone checking the result tells a refined mesh from the same mesh run
# three times. The LEADING NEWLINE is deliberate -- a program that writes
# without a trailing newline glues its text onto the front of the next
# line, and an X11 warning has done exactly that here, turning a correct
# line into 'Invalid MIT-MAGIC-COOKIE-1 keyNDOF = 113'.
# A number inside a prose sentence does not count either, and a
# wrong number is worse than none -- one coupled run that was right in
# every other respect reported NDOF = 1 at all three levels, and its
# refined mesh could not be told from an unrefined one.
try:
    print(f"\nNDOF = {int(len(np.array(uh.as_numpy)))}")
except Exception as _ndof_exc:
    print(f"[dune] could not report NDOF: {_ndof_exc!r}. Your task's"
          f" execution log needs `NDOF = <integer>` on a line of its own,"
          f" so print your own degree-of-freedom count here.")

# PER-LEVEL PERSISTENCE: this level's whole field, and its interface trace and
# traction, named by LEVEL. exports.json is overwritten by the next level;
# these files are not: interpolate THESE onto the probe points your task names.
# THE qx, qy COLUMNS ARE THIS SIDE'S EXPORT, q_out = -(sigma . n_own) (the sign
# convention at the top of this file). A task that asks for the traction
# sigma . n wants their negative, and one that fixes a single normal for both
# sides flips the side whose own normal points the other way: map the columns
# to your task's definition when you write its files.
# A DUMP DEFECT MUST NOT COST YOU THE SOLVE. exports.json is the driver's
# proof that this participant succeeded, and it is written after these files,
# so an exception here would throw away a coupling iteration that worked.
try:
    with open(f"field_level{LEVEL}.csv", "w") as _f:
        _f.write("x,y,ux,uy\n")
        for _i in range(len(xd)):
            _f.write(f"{float(xd[_i]):.11e},{float(yd[_i]):.11e},"
                     f"{float(ud[2 * _i]):.11e},{float(ud[2 * _i + 1]):.11e}\n")
    with open(f"interface_level{LEVEL}.csv", "w") as _f:
        _f.write("x,y,ux,uy,qx,qy\n")
        for _yy, (_ux, _uy), (_qx, _qy) in zip(y_if, U, Q):
            _px, _py = ((float(IFACE_X), float(_yy)) if AX == 0
                        else (float(_yy), float(IFACE_X)))
            _f.write(f"{_px:.11e},{_py:.11e},{float(_ux):.11e},"
                     f"{float(_uy):.11e},{float(_qx):.11e},{float(_qy):.11e}\n")
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
    print(f"[dune_elastic per-level dump] level {LEVEL} dump failed: "
          f"{_dump_exc!r}. exports.json is still written, so the coupling\n"
          f"continues, but this level has no field file to hand in. Fix the\n"
          f"names the dump reads and run this level again.")

# ── EXPORT SELF-CHECK ─ keep this block. It stops the three exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
#    a traction of ~0 against a nonzero partner); and a traction that is the
#    partner's array negated instead of a recovery from THIS side's own system.
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

Path("exports.json").write_text(json.dumps({
    "field_name": "displacement",
    "n_points": int(len(iface_n)),
    "coordinates": [([float(IFACE_X), float(yy)] if AX == 0 else [float(yy), float(IFACE_X)])
                    for yy in y_if],
    "values": [[float(a_), float(b_)] for a_, b_ in U],
    "normal_fluxes": [[float(a_), float(b_)] for a_, b_ in Q],
}, indent=2))
