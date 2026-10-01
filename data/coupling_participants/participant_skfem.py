"""scikit-fem participant for the openPASO `couple` driver.

Steady heat conduction  -div(k grad T) = f  on one 2-D subdomain: the mesh your
solve builds (a box, a box with cells removed, joined boxes). k may vary by
region, the interface may bend (IFACE_SEGMENTS), and which outer edges are held
is your solve's choice.
CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.
"""
import json
import os
from pathlib import Path

import logging                # scikit-fem logs through it (logging.basicConfig(level=logging.INFO)),
                              # and its output goes to STDERR -- redirect with 2>&1 or lose it
import numpy as np
from skfem import (Basis, BilinearForm, ElementTriP1, FacetBasis, LinearForm,
                   MeshTri, condense, solve)
from skfem.helpers import dot, grad


# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side; the payload that served
#    this script gives the exact block for the RIGHT / Neumann side.
SIDE      = "dirichlet"   # "dirichlet" | "neumann"
PARTNER   = "right"       # name of the partner participant in couple(...)
X0, X1    = 0.0, 0.6      # this subdomain's box; for another shape, the box around it
Y0, Y1    = 0.0, 0.4
IFACE_AXIS = "x"          # WHICH straight line the interface is: "x" -> the line x = IFACE_X
                          # (the subdomains sit side by side) | "y" -> the line y = IFACE_X
                          # (they are stacked). Everything below follows from it.
IFACE_X   = 0.6           # shared interface (X0/X1 for axis "x", Y0/Y1 for axis "y")
IFACE_SEGMENTS = ()       # EMPTY: the interface is the single straight line named above. If it
                          # BENDS, list its legs in order, each ("x"|"y", position, from, to):
                          #     IFACE_SEGMENTS = (("x", XA, YA, YB), ("y", YB, XA, XB))
                          # is the leg x = XA from y = YA to YB, then the leg y = YB from x = XA to
                          # XB, with your numbers in place of the letters. Both sides must list the
                          # SAME legs in the SAME order: the exchange is matched by distance along
                          # them. iface_dofs are then every boundary dof on the legs, y_if their (x, y).
K         = 0.8           # conductivity (where it varies by region, your solve builds that)


def F_SRC(x, y):
    """Volumetric source, as a function of position.

    Returns zero as shipped, which is a PLACEHOLDER like every number above
    and is almost never what your problem wants. THIS KNOB USED TO BE A SCALAR
    CONSTANT, AND A CONSTANT CANNOT REPRESENT A SOURCE THAT VARIES WITH
    POSITION: the source of a manufactured solution is a POLYNOMIAL in x and y,
    and no single number is that polynomial. Left at zero the temperature is
    harmonic, the outer Dirichlet values are the only data left in the problem,
    and the answer degenerates to the 1-D profile between them — the interface
    flux is one constant along the whole interface, and it is identically zero
    when the two subdomains carry the same outer value. The coupling will
    converge beautifully to that, and it is not the problem you were given.

    If your problem states a source, or gives you a manufactured solution whose
    source term you derived, put it here. `x` and `y` are NumPy arrays, so
    build the answer with NumPy and return ONE array of the same shape (write
    `0.0 * x + c` for a genuine constant, never a bare `c`):

        # -div(K grad T) for the manufactured T = x**3 * y**2
        return -K * (6.0 * x * y**2 + 2.0 * x**3)
    """
    return np.zeros_like(x)
T_OUTER   = 335.0         # Dirichlet value on the outer edges your solve holds (outer_dofs)
NX, NY    = 46, 26        # this subdomain's own mesh
T_INIT    = 310.0          # iteration-1 fallback interface temperature
Q_INIT    = 0.0           # iteration-1 fallback interface flux
# ─────────────────────────────────────────────────────────────────────────

# ── THE PROBLEM'S DATA ARE DATA, NOT CODE (served). x0, x1, y0, y1, k, outer and source_expr
#    (a string in x and y, `^` allowed) in config.json replace the constants above, written AS
#    THE TASK WRITES THEM. A key it does not state leaves its constant as it is; the audit's
#    equation check reads the same keys. What these keys cannot state: k is ONE number for the
#    whole side, and this contract reads no key for a k that differs by region. A side with
#    several materials builds its k in its own solve and states no k here; one number would be
#    another equation to every check that reads it. The interface is not read from config.json
#    either: it is IFACE_AXIS and IFACE_X above, or IFACE_SEGMENTS for a bent one.
def _expr_fn(expr):
    """A NumPy function of (x, y) from an expression string as a task writes it."""
    code = compile(str(expr).replace("^", "**"), "<source_expr>", "eval")
    names = {"pi": np.pi, "sin": np.sin, "cos": np.cos, "exp": np.exp, "sqrt": np.sqrt,
             "abs": np.abs, "log": np.log, "tanh": np.tanh, "cosh": np.cosh, "sinh": np.sinh}
    def f(x, y):
        env = dict(names); env["x"] = x; env["y"] = y
        return eval(code, {"__builtins__": {}}, env) + 0.0 * x
    return f
try:
    _cfg_all = json.loads(Path("config.json").read_text() or "{}") if Path("config.json").is_file() else {}
    _cfg_all.update(json.loads(os.environ.get("OPENPASO_CONFIG_JSON") or "{}"))
except (ValueError, TypeError, json.JSONDecodeError):
    _cfg_all = {}
_FROM_CFG = []
if all(_k in _cfg_all for _k in ("x0", "x1", "y0", "y1")):
    X0, X1, Y0, Y1 = (float(_cfg_all[_k]) for _k in ("x0", "x1", "y0", "y1"))
    _FROM_CFG.append("x0, x1, y0, y1")
for _nm, _key in (("K", "k"), ("T_OUTER", "outer")):
    if _cfg_all.get(_key) is not None:
        try:
            globals()[_nm] = float(_cfg_all[_key])
            _FROM_CFG.append(_key)
        except (TypeError, ValueError):
            print(f"NOTE: config.json's {_key} is not one number, so {_nm} above stands."
                  + (" This contract reads no k that differs by region: build it in your solve "
                     "and state no k in config.json." if _key == "k" else ""))
if _cfg_all.get("source_expr") is not None:
    _f_cfg = _expr_fn(_cfg_all["source_expr"])
    def F_SRC(x, y):                                   # noqa: F811 -- config wins over the body above
        return _f_cfg(x, y)
    _FROM_CFG.append("source_expr")
print("SOURCES IN USE: from " + ("config.json" if "source_expr" in _FROM_CFG else
                                 "code (the F_SRC body above)")
      + (f"; f = {str(_cfg_all['source_expr'])[:80]}" if "source_expr" in _FROM_CFG else "")
      + f"; taken from config.json: {', '.join(_FROM_CFG) or 'nothing'}")

AX = 0 if IFACE_AXIS == "x" else 1         # the coordinate the interface FIXES
AL = 1 - AX                                # the coordinate that RUNS ALONG it
LO, HI = (X0, X1) if AX == 0 else (Y0, Y1)         # this subdomain, across the interface
ALO, AHI = (Y0, Y1) if AX == 0 else (X0, X1)       # this subdomain, along it
ON_RIGHT = abs(IFACE_X - HI) < abs(IFACE_X - LO)   # interface at this side's MAX of that axis?
OUTER_X = LO if ON_RIGHT else HI           # the opposite face, on the same axis
S = 1.0 if ON_RIGHT else -1.0              # outward normal at interface = S * e_AX
TOL = 1e-9 * max(X1 - X0, Y1 - Y0)



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
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return _partner_block(d)


def arc_length(pts):
    """Distance along the interface, measured from its start, for each point of `pts` (an (n, 2)
    array). This is what makes a BENT interface exchangeable: the two sides meet on a curve, not on a
    coordinate, and a leg's own coordinate is not monotone along the whole thing. With
    IFACE_SEGMENTS empty this is exactly the coordinate along the single line, so the straight case is
    unchanged."""
    pts = np.atleast_2d(np.asarray(pts, float))
    if not IFACE_SEGMENTS:
        return pts[:, AL] - (ALO if ON_RIGHT or True else ALO)
    s_out = np.full(len(pts), np.nan)
    base = 0.0
    for axis, pos, a, b in IFACE_SEGMENTS:
        ax = 0 if axis == "x" else 1
        al = 1 - ax
        lo, hi = (a, b) if a <= b else (b, a)
        on = (np.abs(pts[:, ax] - pos) < TOL) & (pts[:, al] >= lo - TOL) & (pts[:, al] <= hi + TOL)
        s_out[on] = base + np.abs(pts[on, al] - a)
        base += abs(b - a)
    # a point on no leg (a rounding miss, or a partner that listed different legs) falls back to the
    # nearest leg end rather than poisoning the interpolation with NaN
    return np.where(np.isnan(s_out), 0.0, s_out)


def sample(imp, key, fallback, where):
    """Interpolate the partner's samples onto this participant's interface points.

    `where` is either this side's coordinate ALONG a straight interface (the usual case) or, for a
    bent one, its (n, 2) interface points -- in which case both sides are matched by distance along
    IFACE_SEGMENTS."""
    where = np.asarray(where, float)
    bent = where.ndim == 2
    n = len(where)
    if not imp or not imp.get("coordinates"):
        return np.full(n, float(fallback))
    pc = np.atleast_2d(np.asarray(imp["coordinates"], float))
    ys = arc_length(pc) if bent else pc[:, AL]
    target = arc_length(where) if bent else where
    vs = np.asarray(imp.get(key, []), float).ravel()
    if vs.size != ys.size:
        return np.full(n, float(fallback))
    o = np.argsort(ys)
    return np.interp(target, ys[o], vs[o])


imp = read_imports()

# ── THE PER-LEVEL RULE (served). A ./config.json {"level": k, "nx": .., "ny": ..}
#    next to this script overrides the mesh knobs and names the level. The dumps
#    at the foot of this file carry that level in their NAME, so a mesh study
#    leaves one file per level instead of the fine mesh overwriting the coarse.
LEVEL = 1
if Path("config.json").is_file() or os.environ.get("OPENPASO_CONFIG_JSON"):
    try:
        _cfg = json.loads(Path("config.json").read_text() or "{}") if Path("config.json").is_file() else {}
        _cfg.update(json.loads(os.environ.get("OPENPASO_CONFIG_JSON") or "{}"))
        LEVEL = int(_cfg.get("level", LEVEL))
        NX = int(_cfg.get('nx', NX))
        NY = int(_cfg.get('ny', NY))
    except (ValueError, TypeError, json.JSONDecodeError):
        pass

# MAKE THIS CODE SPEAK, BEFORE THE SOLVE RUNS. It is silent by default, and a
# per-level run log carrying no line the solver itself emitted cannot
# establish which code ran on this side, however right its numbers are.
# It sits HERE, beside the level rule, and not up with the imports:
# measured over agent-written participants, a line placed in the import
# block survived in about half of them because that block gets rewritten,
# while everything beside the level rule survived in all of them.
logging.basicConfig(level=logging.INFO)

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
mesh = MeshTri.init_tensor(np.linspace(X0, X1, NX + 1),
                           np.linspace(Y0, Y1, NY + 1))
elem = ElementTriP1()
basis = Basis(mesh, elem)
n2d = basis.nodal_dofs[0]                  # node index -> dof (P1: identity)

px, py = mesh.p[0], mesh.p[1]
iface_n = np.where(np.abs(px - IFACE_X) < TOL)[0]
iface_n = iface_n[np.argsort(py[iface_n])]             # sorted by y
y_if = py[iface_n]
iface_dofs = n2d[iface_n]
outer_dofs = n2d[np.where(np.abs(px - OUTER_X) < TOL)[0]]


@BilinearForm
def stiffness(u, v, w):
    return K * dot(grad(u), grad(v))


@LinearForm
def source(v, w):
    """The loading functional, int_Omega f v dx.

    `w.x` is the (2, nelems, nqp) array of GLOBAL coordinates of the quadrature
    points, so F_SRC is evaluated exactly where the integration rule needs it
    and a polynomial source is integrated to quadrature accuracy — no detour
    through a P1 interpolant of the source, and no constant standing in for a
    field that varies over the element."""
    return F_SRC(w.x[0], w.x[1]) * v
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end


@LinearForm
def flux_load(v, w):
    return w["g"] * v


@LinearForm
def unit_load(v, w):
    return 1.0 * v          # w_i = int_Gamma phi_i ds


# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
A = stiffness.assemble(basis)          # UNCONSTRAINED: condense() below does
b = source.assemble(basis)             # not modify A or b in place
# THE VOLUME LOAD ALONE, kept for the flux recovery at the bottom. The Neumann
# branch adds the partner's interface term into `b`; subtracting that combined
# vector is what made the reaction look like zero on that side.
b_vol = b
fbasis = FacetBasis(mesh, elem,
                    facets=mesh.facets_satisfying(
                        lambda p: np.abs(p[0] - IFACE_X) < TOL))

sol = basis.zeros()
sol[outer_dofs] = T_OUTER
D = outer_dofs
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

# ── THE INTERFACE DOFS ARE THE INTERFACE NODES' OWN (served) ─ keep this block.
#    Every served line below writes the partner's data, reads the flux and
#    exports the values THROUGH iface_dofs, so a list naming other dofs puts the
#    data on other nodes and exports their values under the interface's
#    coordinates -- and a check reading the same list agrees with it (measured on
#    another backend: the dofs of the mesh's first vertices, the interface at
#    0.0, a converged coupling passing every exchange check).
_dl = np.asarray(basis.doflocs, float)               # (2, ndofs): where each dof sits
_yv = np.asarray(y_if, float)
_straight = _yv.ndim == 1 and not IFACE_SEGMENTS     # a bent interface is checked against its legs
_on = np.where(np.abs(_dl[AX] - IFACE_X) <= TOL)[0] if _straight else np.zeros(0, int)
_ids = np.asarray(iface_dofs).astype(int).ravel()
_bad = [k for k, d in enumerate(_ids)
        if k >= _yv.size or not (0 <= d < _dl.shape[1])
        or abs(_dl[AX, d] - IFACE_X) > TOL or abs(_dl[AL, d] - _yv[k]) > TOL] if _straight else []
_missed = len(set(_on.tolist()) - set(_ids.tolist()))
if _straight and (_bad or _missed or len(_ids) != _yv.size):
    _k = _bad[0] if _bad else None
    raise SystemExit(
        f"INTERFACE DOFS: iface_dofs[k] must be the dof on the interface line at y_if[k], one "
        f"per node; iface_dofs has {len(_ids)} entries for {_yv.size} nodes"
        + (f", and {len(_bad)} of them are not (the first: iface_dofs[{_k}] = {_ids[_k]}"
           + (f", a dof at ({_dl[0, _ids[_k]]:g}, {_dl[1, _ids[_k]]:g})"
              if 0 <= _ids[_k] < _dl.shape[1] else ", no dof of this basis")
           + ")" if _bad else "")
        + (f"; {_missed} of the {len(_on)} dofs on the interface line have no entry" if _missed else "")
        + ". basis.doflocs gives each dof's position; basis.nodal_dofs maps a node to its dof.")
# A BENT INTERFACE IS CHECKED AGAINST THE LEGS IT DECLARES: every listed dof is a boundary dof
# on a leg, every boundary dof on a leg is listed, and y_if[k] is where iface_dofs[k] sits.
if IFACE_SEGMENTS:
    _m = basis.mesh
    _bf = _m.boundary_facets()
    _lt = 1e-6 * float(np.linalg.norm(_m.p[:, _m.facets[0, _bf]] - _m.p[:, _m.facets[1, _bf]],
                                      axis=0).min())    # a millionth of the shortest boundary edge

    def _on_legs(P):                                 # P (2, n): True where a point lies on a leg
        hit = np.zeros(P.shape[1], bool)
        for axis, pos, a, b in IFACE_SEGMENTS:
            c = 0 if axis == "x" else 1
            hit |= ((np.abs(P[c] - pos) <= _lt)
                    & (P[1 - c] >= min(a, b) - _lt) & (P[1 - c] <= max(a, b) + _lt))
        return hit
    _bd = np.asarray(basis.get_dofs().all(), int)   # the dofs on the mesh boundary
    _on = _bd[_on_legs(_dl[:, _bd])]
    _bds, _two = set(_bd.tolist()), _yv.ndim == 2 and _yv.shape[-1] >= 2 and len(_yv) == len(_ids)

    def _wrong(k, d):                                # why entry k is not right, or "" when it is
        if not 0 <= d < _dl.shape[1]:
            return f"iface_dofs[{k}] = {d}, no dof of this basis"
        at = f"iface_dofs[{k}] = {d}, a dof at ({_dl[0, d]:g}, {_dl[1, d]:g})"
        if d not in _bds:
            return at + ", not on the mesh boundary"
        if not _on_legs(_dl[:, [d]])[0]:
            return at + ", on no leg"
        if _two and np.abs(_dl[:, d] - _yv[k, :2]).max() > _lt:
            return at + f", while y_if[{k}] = ({_yv[k, 0]:g}, {_yv[k, 1]:g})"
        return ""
    _bad = [w for w in (_wrong(k, d) for k, d in enumerate(_ids)) if w]
    _missed = len(set(_on.tolist()) - set(_ids.tolist()))
    if not _ids.size or not _two or _bad or _missed:
        raise SystemExit(
            f"INTERFACE DOFS: on a bent interface iface_dofs must be every boundary dof on the legs "
            f"of IFACE_SEGMENTS, and y_if[k] the (x, y) of iface_dofs[k]; iface_dofs has {len(_ids)} "
            f"entries and y_if has shape {tuple(_yv.shape)}"
            + (f"; {len(_bad)} entries are not right (the first: {_bad[0]})" if _bad else "")
            + (f"; {_missed} of the {len(_on)} boundary dofs on the legs have no entry" if _missed else "")
            + ". basis.doflocs gives each dof's position; basis.get_dofs() the boundary dofs.")

if SIDE == "dirichlet":
    T_if = sample(imp, "values", T_INIT, y_if)
    sol[iface_dofs] = T_if
    D = np.concatenate([outer_dofs, iface_dofs])
else:
    q_if = sample(imp, "normal_fluxes", Q_INIT, y_if)
    gnod = basis.zeros()                   # P1 trace of the partner's samples
    gnod[iface_dofs] = q_if
    b = b + flux_load.assemble(fbasis, g=fbasis.interpolate(gnod))
    # APPLY the partner's number unchanged (+ integral(g*v) ds_interface)
    # `b_vol` above still holds the VOLUME load alone — the flux recovery
    # below subtracts that, not this, and the distinction is the whole point.

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
sol = solve(*condense(A, b, x=sol, D=D))
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

# ── SOLVE SELF-CHECK (served) ─ keep this block. On every dof outside D, the ones the
#    solve does not hold, a solved system leaves r = A u - b near round-off (A, b: the
#    unconstrained operator and load, b with the partner's flux on the Neumann side).
#    Measured on a coupled run: a side solved with its interface held at zero, then
#    wrote the partner's values into sol; its export and the trace check below read
#    them back, and only this residual showed it. As a share of the scale below,
#    measured on this file: a direct solve 2e-16, a conjugate-gradient solve at its
#    default tolerance about 1e-6, values written after the solve 8e-3 to 1e-1.
_chk_D = np.asarray(D)                              # dof numbers, or a mask over the dofs
_chk_free = np.setdiff1d(np.arange(A.shape[0]),
                         np.flatnonzero(_chk_D) if _chk_D.dtype == bool else _chk_D.astype(int).ravel())
_chk_u = np.asarray(sol, float).ravel()
_chk_bb = np.asarray(b, float).ravel()
_chk_r = np.abs(A @ _chk_u - _chk_bb)[_chk_free]
_chk_sc = max(float(np.abs(_chk_bb).max(initial=0.0)),       # the scale is |A| |u|, not |A u|
              float(np.asarray(abs(A).sum(axis=1)).max(initial=0.0)) * float(np.abs(_chk_u).max(initial=0.0)))
if _chk_r.size and _chk_sc > 0 and _chk_r.max() > 1e-4 * _chk_sc:
    raise SystemExit(f"SOLVE SELF-CHECK: on the dofs outside D, the ones the solve does not hold "
                     f"({_chk_r.size} of them), r = A u - b reaches {_chk_r.max():.2e}, "
                     f"{_chk_r.max() / _chk_sc:.1e} of the system scale {_chk_sc:.2e} (the larger "
                     f"of |b| and |A| |u|), and more than 1e-4 of it on "
                     f"{int((_chk_r > 1e-4 * _chk_sc).sum())} of those dofs. A and b are this side's "
                     f"unconstrained operator and load; a direct solve of that system leaves about "
                     f"1e-16 of the scale there. The field in sol does not solve that system, so "
                     f"nothing was exported.")

# ── DID THE PARTNER'S TRACE ENTER THE SOLVE? (served) ─ keep this block. On the
#    Dirichlet side the partner's values must be in the solution at the interface's
#    own dofs (read from the basis, never through the list the values were written
#    through). The two end nodes are left out -- the outer boundary may hold them.
if SIDE == "dirichlet" and (_straight or IFACE_SEGMENTS):
    if IFACE_SEGMENTS:                               # the leg nodes but the polyline's two ends
        (_a0, _p0, _s0, _), (_a1, _p1, _, _e1) = IFACE_SEGMENTS[0], IFACE_SEGMENTS[-1]
        _ends = np.array([(_p0, _s0) if _a0 == "x" else (_s0, _p0),
                          (_p1, _e1) if _a1 == "x" else (_e1, _p1)], float)
        _inner = _on[[np.abs(_ends - _dl[:, d]).max(axis=1).min() > _lt for d in _on]]
        _where = _dl[:, _inner].T
    else:
        _inner = _on[(np.abs(_dl[AL, _on] - ALO) > TOL) & (np.abs(_dl[AL, _on] - AHI) > TOL)]
        _where = _dl[AL, _inner]
    if _inner.size:
        _want = sample(imp, "values", T_INIT, _where)
        _gap = float(np.abs(np.asarray(sol)[_inner] - _want).max())
        if _gap > 1e-9 * max(1.0, float(np.abs(_want).max())):
            raise SystemExit(f"EXPORT SELF-CHECK: the partner's temperature is not in the solution at "
                             f"the interface nodes (largest gap {_gap:.3e}): on the Dirichlet side the "
                             f"interface dofs must be in the condensed set D with the partner's values "
                             f"in x, and the solve must keep them. A solve that frees them returns this "
                             f"side's own answer and couples to nothing.")

# Outward normal flux density q = -(k grad T).n on the interface.
#
# WHY NOT AN L2 PROJECTION OF THE GRADIENT. That is what this file used to do:
# project -k dT/dx over the whole subdomain and sample it at the interface. The
# gradient of a P1 solution is only O(h) accurate ON the boundary — the
# superconvergence points are interior — and the boundary trace is exactly what
# the coupling reads. Measured against a manufactured solution with a known
# exact interface flux, the projection converges at order ~1 while the
# consistent flux below converges at ~2, so the recovery, not the physics and
# not the partner, was setting the answer.
#
# THE CONSISTENT (REACTION) FLUX. From
#     a(u,v) - (f,v) = int_dOmega (k grad u . n) v ds = -int_Gamma qn v ds
# it follows that for every basis function phi_i on the interface
#     int_Gamma qn phi_i ds = -r_i,   r = A u_h - b
# with r the UNCONSTRAINED residual: A and b above are assembled with NO
# boundary condition applied and skfem's condense() returns copies, so the
# constrained rows of A still carry the reaction. Dividing by
# w_i = int_Gamma phi_i ds turns the functional into a density the partner can
# interpolate pointwise.
# ONE FORMULA, BOTH SIDES. An earlier version used the reaction on the
# Dirichlet side and an L2-projected gradient on the Neumann side, reasoning
# that the Neumann interface dofs are free so r comes out ~0 there. That holds
# only when the residual is taken against a load that ALREADY CONTAINS the
# interface term. Subtract the VOLUME load alone and those same rows carry
# exactly the interface functional the partner applied:
#     (A u - b_vol)_i = int_Gamma g phi_i ds
# On the Dirichlet side there is no interface term, so b == b_vol and the two
# cases are one expression.
#
# WHAT IS MEASURED, AND WHAT IS ONLY ALGEBRA. Handing the NEUMANN side a flux
# and asking for it back is an ASSEMBLY IDENTITY, not a convergence test: on
# free interface rows r = A u - b_vol IS M_Gamma g, so the export is
# -(M_Gamma g)/(M_Gamma 1) and its offset from -g is -(h^2/6) g''(y) for ANY
# correct assembly of ANY equation. The "order 2.00" that used to stand here
# was read off that fixture; it is a property of the P1 boundary mass matrix,
# not of this code — a bare NumPy mass matrix reproduces the same numbers with
# no PDE, no solver and no material in it, and this file reproduces FEniCSx's
# to six significant figures for the same reason. That fixture is kept
# (tests/test_interface_flux_recovery.py) for what it really tests: sign
# convention, interface weight, facet set, blocked dofs.
#
# THE ORDER is measured on the DIRICHLET side against an ANALYTIC interface
# flux the participant is never handed
# (tests/test_interface_flux_converges_to_a_known_exact_flux.py). FEniCSx, the
# same formulation, 8/16/32/64 uniform triangle meshes, max error over interior
# interface nodes:
#     2.889e-01  7.243e-02  1.814e-02  4.556e-03   ORDER 1.996  1.998  1.993
# and only first order (1.10, 1.06, 1.04) at the two nodes where the interface
# meets the outer boundary, which is why those are handled apart just below.
#
# THE RETIRED L2-PROJECTED GRADIENT, in the norms it was measured in: order ~1
# in the interior AWAY FROM THE ENDS (0.93), 0.50 in rms, and non-convergent in
# the max norm that includes the near-end nodes, where it stalls at 2.6 against
# a true flux of size 2 to 5. It was written up as a flat "order 0.00, it never
# converges", which was true of one norm only. Not re-measured since the branch
# was deleted.
r = A @ sol - b_vol                    # r = A u_h - b_vol, no bc applied
wgt = unit_load.assemble(fbasis)       # w_i = int_Gamma phi_i ds

Q = np.zeros(len(iface_dofs))
ok = np.abs(wgt[iface_dofs]) > 1e-14
Q[ok] = -r[iface_dofs][ok] / wgt[iface_dofs][ok]

# An interface node that ALSO lies on the outer Dirichlet boundary carries
# the OUTER reaction as well, so its residual is not this interface's flux.
# Take the nearest interior interface node rather than exporting a corner
# value that is physically a different quantity. This holds on both sides.
# outer_dofs AS DOF NUMBERS: np.isin reads a Python set as ONE object and matches
# nothing (measured: the corner values went out unreplaced).
suspect = np.isin(iface_dofs, sorted(outer_dofs) if isinstance(outer_dofs, (set, frozenset)) else outer_dofs) | ~ok
good = np.where(~suspect)[0]
if len(good):
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]
# ── EXPORT SELF-CHECK ─ keep this block. It stops the exports that look fine and
#    are worthless: no interface point, or coordinates, values and fluxes of
#    different lengths; a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
#    a flux of ~0 against a nonzero partner); and a flux that is the partner's
#    array negated instead of a recovery from THIS side's own system.
_chk_vals = np.asarray(sol[iface_dofs], float).ravel()
_chk_flux = np.asarray(Q, float).ravel()
_yc = np.asarray(y_if, float)
_exp_co = (_yc[:, :2] if _yc.ndim == 2 else                # the coordinates exported below
           np.column_stack([np.full(_yc.size, float(IFACE_X)), _yc]) if AX == 0 else
           np.column_stack([_yc, np.full(_yc.size, float(IFACE_X))]))
if not len(_exp_co):
    raise SystemExit("EXPORT SELF-CHECK: no interface point to export (iface_dofs and y_if are "
                     "empty), so the partner would receive an empty interface. List this "
                     "side's interface dofs and their positions.")
if not len(_exp_co) == len(_chk_vals) == len(_chk_flux):
    raise SystemExit(f"EXPORT SELF-CHECK: {len(_exp_co)} interface coordinates, {len(_chk_vals)} "
                     f"values and {len(_chk_flux)} fluxes; each exported point needs its own value "
                     f"and flux. Build all three from the same iface_dofs and y_if.")
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
    # IT SAYS WHAT IT MEASURED. It used to name a missing facet term; measured on a
    # coupled run, the term was there and the interface facets were not. The weights
    # w_i tell those apart.
    _chk_w = np.abs(np.asarray(wgt, float)[np.asarray(iface_dofs, int)])
    _chk_on = np.where(~suspect)[0]                  # w_i > 0 and held by no outer value
    _chk_ri = np.abs(np.asarray(r, float)[np.asarray(iface_dofs, int)])
    raise SystemExit(f"EXPORT SELF-CHECK: the recovered interface flux Q = -r/w is ~0 (at most "
                     f"{np.abs(_chk_flux).max():.2e}) against an imported flux of up to "
                     f"{np.abs(_chk_qin).max():.2e}. The weights w_i = int phi_i ds over the facets "
                     f"of fbasis at the {_chk_w.size} interface dofs are "
                     + ("0 at every one: fbasis covers no facet at these dofs, and nothing "
                        "integrated over it reaches them." if not (_chk_w > 1e-14).any() else
                        f"nonzero at {int((_chk_w > 1e-14).sum())} of them (sum {_chk_w.sum():.3e}); "
                        f"at the {_chk_on.size} of those that no outer value holds, r = A u - b_vol "
                        f"is at most {_chk_ri[_chk_on].max(initial=0.0):.2e}, so there A u equals "
                        f"the volume load b_vol."))
# (Dirichlet role only: a Neumann side's consistent recovery of a CONSTANT
#  applied flux can legitimately reproduce it to the last bit.)
if SIDE == "dirichlet" and _chk_qin.shape == _chk_flux.shape and _chk_flux.size \
        and np.array_equal(_chk_flux, -_chk_qin):
    raise SystemExit("EXPORT SELF-CHECK: the exported flux is the partner's "
                     "array negated, bit for bit: a copy, not a recovery from "
                     "this side's own assembled system")

# THE RUN-LOG CONTRACT LINE: `NDOF = <integer>` on a line of its OWN.
# The audit reads that exact shape, and they read it PER
# LEVEL: it is how anyone checking the result tells a refined mesh from the same mesh run
# three times. The LEADING NEWLINE is deliberate -- a program that writes
# without a trailing newline glues its text onto the front of the next
# line, and an X11 warning has done exactly that here, turning a correct
# line into 'Invalid MIT-MAGIC-COOKIE-1 keyNDOF = 54'.
# A number inside a prose sentence does not count either, and a
# wrong number is worse than none -- one coupled run that was right in
# every other respect reported NDOF = 1 at all three levels, and its
# refined mesh could not be told from an unrefined one.
try:
    print(f"\nNDOF = {int(len(sol))}")
except Exception as _ndof_exc:
    print(f"[skfem] could not report NDOF: {_ndof_exc!r}. Your task's"
          f" execution log needs `NDOF = <integer>` on a line of its own,"
          f" so print your own degree-of-freedom count here.")

# PER-LEVEL PERSISTENCE: this level's whole field, and its interface trace and
# flux, named by LEVEL. exports.json is overwritten by the next level; these
# files are not.
# Interpolate THESE onto the probe points your task names. A file the next
# level overwrites cannot carry a mesh study.
# A DUMP DEFECT MUST NOT COST YOU THE SOLVE. exports.json is the driver's
# proof that this participant succeeded, and it is written after these files,
# so an exception here would throw away a coupling iteration that worked.
try:
    with open(f"field_level{LEVEL}.csv", "w") as _f:
        _f.write("x,y,u\n")
        for _px, _py, _u in zip(mesh.p[0], mesh.p[1], sol):
            _f.write(f"{float(_px):.11e},{float(_py):.11e},{float(_u):.11e}\n")
    with open(f"interface_level{LEVEL}.csv", "w") as _f:
        _f.write("x,y,u,qn\n")
        for (_px, _py), _d, _q in zip(_exp_co, iface_dofs, Q):
            _f.write(f"{float(_px):.11e},{float(_py):.11e},"
                     f"{float(sol[_d]):.11e},{float(_q):.11e}\n")
except Exception as _dump_exc:
    # AND LEAVE NO HALF-WRITTEN FILE BEHIND. `open(..., "w")` truncates
    # before it fails, so a dump that died mid-way leaves a header-only
    # CSV -- a file that looks like a submission and carries no rows.
    for _partial in (f"field_level{LEVEL}.csv", f"interface_level{LEVEL}.csv"):
        try:
            if Path(_partial).is_file() and len(
                    Path(_partial).read_text().splitlines()) <= 1:
                Path(_partial).unlink()
        except OSError:
            pass
    print(f"[skfem per-level dump] level {LEVEL} dump failed: "
          f"{_dump_exc!r}. exports.json is still written, so the coupling\n"
          f"continues, but this level has no field file to hand in. Fix the\n"
          f"names the dump reads and run this level again.")

Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(iface_dofs)),
    "coordinates": _exp_co.tolist(),
    "values": [float(t) for t in sol[iface_dofs]],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))
