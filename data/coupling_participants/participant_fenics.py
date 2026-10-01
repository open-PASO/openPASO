"""FEniCSx (dolfinx) participant for the openPASO `couple` driver.

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

Physics: steady conduction  -div(K grad T) = F_SRC  on one 2-D subdomain:
the mesh your solve builds (a box, a box with cells removed, joined boxes).
K may vary by region, the interface may bend (IFACE_SEGMENTS), and which
outer edges are held at T_OUTER is your solve's choice; FULL_OUTER_DIRICHLET
states it for the checks that judge your field.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import dolfinx                # the MODULE, so dolfinx.log.set_log_level(...) resolves:
                              # a run log that must carry this code's own output needs it,
                              # and `from dolfinx import fem` alone leaves `dolfinx` undefined
import ufl
from dolfinx import default_scalar_type, fem, mesh as dmesh
from dolfinx.fem import petsc as _fp
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI


# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side; the payload that served
#    this script gives the exact block for the RIGHT / Neumann side.
SIDE      = "dirichlet"   # "dirichlet" (import T, export flux) | "neumann"
PARTNER   = "right"       # the partner's `name` in your couple(...) call
X0, X1    = 0.0, 0.6      # this subdomain's box; for another shape, the box around it
Y0, Y1    = 0.0, 0.4
IFACE_AXIS = "x"          # WHICH straight line the interface is: "x" -> the line x = IFACE_X
                          # (the subdomains sit side by side) | "y" -> the line y = IFACE_X
                          # (they are stacked). Everything below follows from it.
IFACE_X   = 0.6           # the shared interface; X0/X1 for axis "x", Y0/Y1 for axis "y"
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

    HOW THIS ENTERS FEniCSx (your solve below has to do it): a UFL expression
    carries no NumPy ufuncs, so do NOT call this function on
    ufl.SpatialCoordinate (np.zeros_like(x) does not even raise -- it returns
    a 0-d object array and the source collapses to a constant). Interpolate it
    into a P1 Function on your space instead, f_h.interpolate(lambda X:
    F_SRC(X[0], X[1])), and integrate f_h * v * dx -- the P1 interpolant of the
    source, quadrature error O(h^2), the order of the discretisation.
    """
    return np.zeros_like(x)
T_OUTER   = 335.0         # Dirichlet value on the outer edges your solve holds (outer_dofs)
FULL_OUTER_DIRICHLET = None  # True: the two edges the interface ends on are held at T_OUTER
                          # too; False: they are natural (zero flux), only the edge opposite the
                          # interface is held. From your problem statement: the audit judges
                          # your field's outer edges against it, and None judges all of them.
NX, NY    = 46, 26        # this subdomain's OWN mesh; need not match the partner
T_INIT    = 310.0         # iteration-1 fallback interface temperature
Q_INIT    = 0.0           # iteration-1 fallback interface flux
# ─────────────────────────────────────────────────────────────────────────

# ── THE PER-LEVEL RULE (served). A ./config.json {"level": k, "nx": .., "ny": ..}
#    next to this script overrides NX, NY and names the level; the per-level
#    dumps below carry that level so the coarse levels survive the fine ones.
LEVEL = 1
if Path("config.json").is_file() or os.environ.get("OPENPASO_CONFIG_JSON"):
    try:
        _cfg = json.loads(Path("config.json").read_text() or "{}") if Path("config.json").is_file() else {}
        _cfg.update(json.loads(os.environ.get("OPENPASO_CONFIG_JSON") or "{}"))   # a multi-level call's level keys
        LEVEL = int(_cfg.get("level", LEVEL))
        NX = int(_cfg.get("nx", NX))
        NY = int(_cfg.get("ny", NY))
    except (ValueError, TypeError, json.JSONDecodeError):
        pass

# ── THE PROBLEM'S DATA ARE DATA, NOT CODE (served). x0, x1, y0, y1, k, outer, full_outer_dirichlet
#    (true or false) and source_expr (a string in x and y, `^` allowed) in config.json replace the
#    constants above, written AS
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
if isinstance(_cfg_all.get("full_outer_dirichlet"), bool):
    FULL_OUTER_DIRICHLET = _cfg_all["full_outer_dirichlet"]
    _FROM_CFG.append("full_outer_dirichlet")
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
OUTER_X = LO if abs(IFACE_X - HI) < abs(IFACE_X - LO) else HI
S = 1.0 if IFACE_X > OUTER_X else -1.0     # outward normal at interface = S * e_AX



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
        return _partner_block(json.loads(p.read_text()))
    except json.JSONDecodeError:
        return None


TOL_IF = 1e-9 * max(X1 - X0, Y1 - Y0)


def arc_length(pts):
    """Distance along the interface, from its start, for each point of `pts` (an (n, 2) array).

    This is what makes a BENT interface exchangeable: the two sides meet on a curve, and a single
    coordinate is not monotone along two legs. With IFACE_SEGMENTS empty this is the coordinate along
    the single straight line, so the usual case is unchanged."""
    pts = np.atleast_2d(np.asarray(pts, float))
    if not IFACE_SEGMENTS:
        return pts[:, AL]
    s_out = np.full(len(pts), np.nan)
    base = 0.0
    for axis, pos, a, b in IFACE_SEGMENTS:
        ax = 0 if axis == "x" else 1
        al = 1 - ax
        lo, hi = (a, b) if a <= b else (b, a)
        on = ((np.abs(pts[:, ax] - pos) < TOL_IF) & (pts[:, al] >= lo - TOL_IF)
              & (pts[:, al] <= hi + TOL_IF) & np.isnan(s_out))
        s_out[on] = base + np.abs(pts[on, al] - a)
        base += abs(b - a)
    return np.where(np.isnan(s_out), 0.0, s_out)


def sample(imp, key, fallback, where):
    """Map the partner's samples onto THIS participant's interface points.
    The driver does no interpolation — non-matching meshes are handled here.

    `where` is this side's coordinate along a straight interface, or its (n, 2) interface POINTS for
    a bent one, in which case both sides are matched by distance along IFACE_SEGMENTS."""
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

# MAKE THIS CODE SPEAK, BEFORE THE SOLVE RUNS. It is silent by default, and a
# per-level run log carrying no line the solver itself emitted cannot
# establish which code ran on this side, however right its numbers are.
# It sits HERE, beside the level rule, and not up with the imports:
# measured over agent-written participants, a line placed in the import
# block survived in about half of them because that block gets rewritten,
# while everything beside the level rule survived in all of them.
dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
domain = dmesh.create_rectangle(MPI.COMM_WORLD, [[X0, Y0], [X1, Y1]],
                                [NX, NY], dmesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))
fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)

xy = V.tabulate_dof_coordinates()
iface_dofs = np.where(np.abs(xy[:, 0] - IFACE_X) < 1e-10)[0]
iface_dofs = iface_dofs[np.argsort(xy[iface_dofs, 1])]   # constant order, always
y_if = xy[iface_dofs, 1]
if len(iface_dofs) == 0:
    sys.exit(f"no interface DOFs at x={IFACE_X}: this subdomain spans "
             f"[{X0},{X1}], so nothing is shared with the partner")

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = fem.Constant(domain, default_scalar_type(K)) * \
    ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx

# SOURCE. F_SRC is INTERPOLATED into a Function, not wrapped in a
# fem.Constant. A Constant is one number for the whole subdomain, so the
# moment F_SRC varies with position — which is the normal case, a manufactured
# solution's source is a polynomial — `Constant(domain, F_SRC(x, y))` cannot
# even be built (F_SRC returns an array), and the constant that used to sit
# here silently solved a different problem. The interpolant is P1 like the
# solution space, so the source's quadrature error is O(h^2), the same order as
# the discretization error itself. Do NOT hand F_SRC a ufl.SpatialCoordinate
# instead: a UFL expression carries no NumPy ufuncs, so np.zeros_like(x)
# returns a 0-d OBJECT array and np.sin(x) raises.
f_src = fem.Function(V)
f_src.interpolate(lambda X: np.zeros(X.shape[1]) + F_SRC(X[0], X[1]))
# KEPT SEPARATE FROM L ON PURPOSE. The flux recovery below subtracts the
# VOLUME load alone, on both sides; adding the interface term into the same
# form is what made the reaction look like zero on the Neumann side.
L_vol = f_src * v * ufl.dx

outer = dmesh.locate_entities_boundary(domain, fdim,
                                       lambda x: np.isclose(x[0], OUTER_X))
outer_dofs = fem.locate_dofs_topological(V, fdim, outer)
bcs = [fem.dirichletbc(default_scalar_type(T_OUTER), outer_dofs, V)]

# One definition of the interface measure, used by the Neumann branch to APPLY
# the partner's flux and by the Dirichlet branch to RECOVER its own.
facets_if = dmesh.locate_entities_boundary(domain, fdim,
                                           lambda x: np.isclose(x[0], IFACE_X))
tags_if = dmesh.meshtags(domain, fdim, np.sort(facets_if),
                         np.full(len(facets_if), 7, dtype=np.int32))
ds_if = ufl.Measure("ds", domain=domain, subdomain_data=tags_if)(7)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

# ── THE INTERFACE DOFS ARE THE INTERFACE NODES' OWN (served) ─ keep this block.
#    Every served line below writes the partner's data, reads the flux and
#    exports the values THROUGH iface_dofs, so a list naming other dofs puts the
#    data on other nodes and exports their values under the interface's
#    coordinates -- and a check reading the same list agrees with it (measured on
#    another backend: the dofs of the mesh's first vertices, the interface at
#    0.0, a converged coupling passing every exchange check).
_xy_all = V.tabulate_dof_coordinates()
_yv = np.asarray(y_if, float)
_straight = _yv.ndim == 1 and not IFACE_SEGMENTS    # a bent interface is checked against its legs
if _straight:
    _on = np.where(np.abs(_xy_all[:, AX] - IFACE_X) <= TOL_IF)[0]
    _ids = np.asarray(iface_dofs).astype(int).ravel()
    _bad = [k for k, d in enumerate(_ids)
            if k >= _yv.size or not (0 <= d < len(_xy_all))
            or abs(_xy_all[d, AX] - IFACE_X) > TOL_IF or abs(_xy_all[d, AL] - _yv[k]) > TOL_IF]
    _missed = len(set(_on.tolist()) - set(_ids.tolist()))
    if _bad or _missed or len(_ids) != _yv.size:
        _k = _bad[0] if _bad else None
        sys.exit(
            f"INTERFACE DOFS: iface_dofs[k] must be the dof on the interface line at y_if[k], one "
            f"per node; iface_dofs has {len(_ids)} entries for {_yv.size} nodes"
            + (f", and {len(_bad)} of them are not (the first: iface_dofs[{_k}] = {_ids[_k]}"
               + (f", a dof at ({_xy_all[_ids[_k], 0]:g}, {_xy_all[_ids[_k], 1]:g})"
                  if 0 <= _ids[_k] < len(_xy_all) else ", no dof of this space")
               + ")" if _bad else "")
            + (f"; {_missed} of the {len(_on)} dofs on the interface line have no entry"
               if _missed else "")
            + ". Take them from V.tabulate_dof_coordinates() of the space the solution lives in.")
# A BENT INTERFACE IS CHECKED AGAINST THE LEGS IT DECLARES: every listed dof is a boundary dof
# on a leg, every boundary dof on a leg is listed, and y_if[k] is where iface_dofs[k] sits.
if IFACE_SEGMENTS:
    _msh = V.mesh
    _fd = _msh.topology.dim - 1
    _msh.topology.create_connectivity(_fd, _msh.topology.dim)
    _ext = dmesh.exterior_facet_indices(_msh.topology)
    _lt = 1e-6 * float(np.min(_msh.h(_fd, _ext)))    # a millionth of the shortest boundary edge

    def _on_legs(P):                                 # P (n, 2): True where a point lies on a leg
        hit = np.zeros(len(P), bool)
        for axis, pos, a, b in IFACE_SEGMENTS:
            c = 0 if axis == "x" else 1
            hit |= ((np.abs(P[:, c] - pos) <= _lt)
                    & (P[:, 1 - c] >= min(a, b) - _lt) & (P[:, 1 - c] <= max(a, b) + _lt))
        return hit
    _bd = np.asarray(fem.locate_dofs_topological(V, _fd, _ext), int)   # boundary dofs
    _on = _bd[_on_legs(_xy_all[_bd, :2])]
    _ids = np.asarray(iface_dofs).astype(int).ravel()
    _bds, _two = set(_bd.tolist()), _yv.ndim == 2 and _yv.shape[-1] >= 2 and len(_yv) == len(_ids)

    def _wrong(k, d):                                # why entry k is not right, or "" when it is
        if not 0 <= d < len(_xy_all):
            return f"iface_dofs[{k}] = {d}, no dof of this space"
        at = f"iface_dofs[{k}] = {d}, a dof at ({_xy_all[d, 0]:g}, {_xy_all[d, 1]:g})"
        if d not in _bds:
            return at + ", not on the mesh boundary"
        if not _on_legs(_xy_all[[d], :2])[0]:
            return at + ", on no leg"
        if _two and np.abs(_xy_all[d, :2] - _yv[k, :2]).max() > _lt:
            return at + f", while y_if[{k}] = ({_yv[k, 0]:g}, {_yv[k, 1]:g})"
        return ""
    _bad = [w for w in (_wrong(k, d) for k, d in enumerate(_ids)) if w]
    _missed = len(set(_on.tolist()) - set(_ids.tolist()))
    if not _ids.size or not _two or _bad or _missed:
        sys.exit(
            f"INTERFACE DOFS: on a bent interface iface_dofs must be every boundary dof on the legs "
            f"of IFACE_SEGMENTS, and y_if[k] the (x, y) of iface_dofs[k]; iface_dofs has {len(_ids)} "
            f"entries and y_if has shape {tuple(_yv.shape)}"
            + (f"; {len(_bad)} entries are not right (the first: {_bad[0]})" if _bad else "")
            + (f"; {_missed} of the {len(_on)} boundary dofs on the legs have no entry" if _missed else "")
            + ". Take them from V.tabulate_dof_coordinates() of the space the solution lives in.")

# ── THE PARTNER'S DATA, APPLIED AT THIS SIDE'S INTERFACE (served) ─ keep this block.
#    This is the exchange, not the physics. sample() maps the partner's values onto
#    THIS side's interface points by where they sit along the interface, never by
#    their row order in the partner's file. The Dirichlet side holds them with a
#    dirichletbc on iface_dofs; the Neumann side adds them to its load as a flux,
#    integrated over ds_if. bcs is the LIST of this side's own dirichletbc (the
#    outer edges it holds), and L_vol its volume load alone.
L = L_vol
if SIDE == "dirichlet":
    g = fem.Function(V)
    g.x.array[iface_dofs] = sample(imp, "values", T_INIT, y_if)
    bcs.append(fem.dirichletbc(g, iface_dofs))
else:
    g = fem.Function(V)
    g.x.array[iface_dofs] = sample(imp, "normal_fluxes", Q_INIT, y_if)
    L = L_vol + g * v * ds_if   # APPLY the partner's number UNCHANGED

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
uh = LinearProblem(a, L, bcs=bcs, petsc_options_prefix="cpl",
                   petsc_options={"ksp_type": "preonly",
                                  "pc_type": "lu"}).solve()

# Outward normal flux density q = -(K grad T).n on the interface.
#
# WHY NOT AN L2 PROJECTION OF THE GRADIENT. That is what this file used to do:
# project -K dT/dx over the whole subdomain and sample it at the interface.
# The gradient of a P1 solution is only O(h) accurate ON the boundary — the
# superconvergence points are interior — and the boundary trace is exactly what
# the coupling reads. The consistent flux below is second order against an
# analytic interface flux: 1.996, 1.998, 1.993 on 8/16/32/64, re-measured on
# this file by tests/test_interface_flux_converges_to_a_known_exact_flux.py.
# (The original sweep that retired the projection reported 1.07 for it against
# 2.05 for the reaction, and a coupled field's convergence order moving from
# 1.83 to 2.05. Those three figures are from that
# sweep and have NOT been re-run since the projected branch was deleted; the
# 1.996/1.998/1.993 above is the number this repo can reproduce today.) The
# recovery, not the physics and not the partner, was setting the answer.
#
# THE CONSISTENT (REACTION) FLUX. From
#     a(u,v) - (f,v) = int_dOmega (K grad u . n) v ds = -int_Gamma qn v ds
# it follows that for every basis function phi_i on the interface
#     int_Gamma qn phi_i ds = -r_i,   r = A u_h - b
# with r the UNCONSTRAINED residual: assembled with no boundary condition
# applied and with the constrained rows NOT zeroed, because on the Dirichlet
# side those rows ARE the reaction and zeroing them destroys the very quantity
# being recovered. Dividing by w_i = int_Gamma phi_i ds turns the functional
# into a density the partner can interpolate pointwise.
p_, w_ = ufl.TrialFunction(V), ufl.TestFunction(V)

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end
# ── SOLVE SELF-CHECK (served) ─ keep this block. On every dof that no dirichletbc in
#    bcs holds, a solved system leaves r = A u - b near round-off (A, b: this side's a
#    and L, no boundary condition applied). Measured on a coupled run: a side solved
#    with its interface held at zero, then wrote the partner's values into the field;
#    its export and the trace check below read them back, and only this residual
#    showed it. As a share of the scale below, measured on this file: a direct solve
#    4e-16, a Krylov solve at PETSc's default tolerance 1e-6 to 2e-6, one ILU sweep
#    2e-2, values written after the solve 8e-3 to 1e-1.
_chk_A = _fp.assemble_matrix(fem.form(a))
_chk_A.assemble()
_chk_b = _fp.assemble_vector(fem.form(L))
_chk_b.ghostUpdate()
_chk_Au = _chk_A.createVecLeft()
_chk_A.mult(uh.x.petsc_vec, _chk_Au)
_chk_n = _chk_Au.getLocalSize()
_chk_held = np.concatenate([np.asarray(_bc.dof_indices()[0], int) for _bc in bcs] + [np.zeros(0, int)])
_chk_free = np.setdiff1d(np.arange(_chk_n), _chk_held)
_chk_ip, _chk_ix, _chk_av = _chk_A.getValuesCSR()
_chk_rows = np.bincount(np.repeat(np.arange(_chk_n), np.diff(_chk_ip)), weights=np.abs(_chk_av),
                        minlength=_chk_n)                    # the scale is |A| |u|, not |A u|
_chk_bb = np.asarray(_chk_b.array, float)[:_chk_n]
_chk_r = np.abs(np.asarray(_chk_Au.array, float) - _chk_bb)[_chk_free]
_chk_sc = max(float(np.abs(_chk_bb).max(initial=0.0)),
              float(_chk_rows.max(initial=0.0)) * float(np.abs(uh.x.array).max(initial=0.0)))
if _chk_r.size and _chk_sc > 0 and _chk_r.max() > 1e-4 * _chk_sc:
    # A NEUMANN SIDE THAT SOLVED BEFORE ITS FLUX TERM WAS ADDED SOLVES L_vol, and this says so when
    # it measures it. Measured on a coupled round: a worker solved in hole 1, the stop said only
    # that uh does not solve the system, and the run read it as a check that was "too strict".
    _chk_why = ""
    if SIDE == "neumann":
        _chk_bv = _fp.assemble_vector(fem.form(L_vol))
        _chk_bv.ghostUpdate()
        _chk_rv = np.abs(np.asarray(_chk_Au.array, float)
                         - np.asarray(_chk_bv.array, float)[:_chk_n])[_chk_free]
        if _chk_rv.max(initial=0.0) <= 1e-4 * _chk_sc:
            _chk_why = (f" On this Neumann side uh solves A u = b_vol, the system WITHOUT the "
                        f"interface flux (r = A u - b_vol reaches only {_chk_rv.max(initial=0.0):.2e} "
                        f"there): the solve used L_vol, not L -- a solve placed before the served "
                        f"block that adds g v ds_if to L (hole 1) does exactly this.")
    sys.exit(f"SOLVE SELF-CHECK: on the dofs that no dirichletbc in bcs holds ({_chk_r.size} of "
             f"them), r = A u - b reaches {_chk_r.max():.2e}, {_chk_r.max() / _chk_sc:.1e} of the system "
             f"scale {_chk_sc:.2e} (the larger of |b| and |A| |u|), and more than 1e-4 of it on "
             f"{int((_chk_r > 1e-4 * _chk_sc).sum())} of those dofs. A and b are this side's a and L "
             f"assembled with no boundary condition; a direct solve of that system leaves about "
             f"1e-16 of the scale there. The field in uh does not solve that system, so nothing "
             f"was exported." + _chk_why)
# ── DID THE PARTNER'S TRACE ENTER THE SOLVE? (served) ─ keep this block. On the
#    Dirichlet side the partner's values must be in the solution at the interface's
#    own dofs (read from the mesh, never through the list the values were written
#    through). The two end nodes are left out -- the outer boundary may hold them.
if SIDE == "dirichlet" and (_straight or IFACE_SEGMENTS):
    if IFACE_SEGMENTS:                               # the leg nodes but the polyline's two ends
        (_a0, _p0, _s0, _), (_a1, _p1, _, _e1) = IFACE_SEGMENTS[0], IFACE_SEGMENTS[-1]
        _ends = np.array([(_p0, _s0) if _a0 == "x" else (_s0, _p0),
                          (_p1, _e1) if _a1 == "x" else (_e1, _p1)], float)
        _inner = _on[[np.abs(_ends - _xy_all[d, :2]).max(axis=1).min() > _lt for d in _on]]
        _where = _xy_all[_inner, :2]
    else:
        _inner = _on[(np.abs(_xy_all[_on, AL] - ALO) > TOL_IF) & (np.abs(_xy_all[_on, AL] - AHI) > TOL_IF)]
        _where = _xy_all[_inner, AL]
    if _inner.size:
        _want = sample(imp, "values", T_INIT, _where)
        _gap = float(np.abs(np.asarray(uh.x.array)[_inner] - _want).max())
        if _gap > 1e-9 * max(1.0, float(np.abs(_want).max())):
            sys.exit(f"EXPORT SELF-CHECK: the partner's temperature is not in the solution at "
                     f"the interface nodes (largest gap {_gap:.3e}): on the Dirichlet side the "
                     f"interface must be held -- a dirichletbc on the interface dofs, with the "
                     f"partner's values -- and the solve must keep it. A solve that frees them "
                     f"returns this side's own answer and couples to nothing.")
# ONE FORMULA, BOTH SIDES. An earlier version of this file used the reaction
# only on the Dirichlet side and an L2-projected gradient on the Neumann side,
# on the reasoning that the Neumann interface DOFs are free, so the discrete
# equations hold on them and r comes out ~0. That is true only when the
# residual is taken against a load that ALREADY CONTAINS the interface term.
# Subtract the VOLUME load alone and those same rows carry exactly the
# interface functional the partner applied:
#     (A u - b_vol)_i = int_Gamma g phi_i ds
# On the Dirichlet side there is no interface term at all, so b == b_vol and
# the two cases are the same expression.
#
# WHAT IS MEASURED, AND WHAT IS ONLY ALGEBRA. Two different fixtures, and only
# one of them is a convergence result. Do not quote the first as one.
#
#  (1) ASSEMBLY IDENTITY, not convergence. Handing THIS side a flux g and
#      asking for it back cannot measure an order: on free interface rows
#      r = A u - b_vol IS M_Gamma g by construction, so the export is the
#      consistent-to-nodal conversion -(M_Gamma g)/(M_Gamma 1) and its offset
#      from -g is -(h^2/6) g''(y) for ANY correct assembly of ANY equation,
#      with any conductivity. Measured for q = 2 + 3 sin(4y) at n = 8/16/32:
#      1.964339e-02, 4.983320e-03, 1.249629e-03 — the same figures a bare 1-D
#      P1 mass matrix in NumPy produces with no PDE, no solver and no material
#      in it. This used to be quoted here as "order 2.00"; it is not evidence
#      of an order. The fixture is kept (tests/test_interface_flux_recovery.py)
#      because it does catch a flipped sign, a wrong weight, a mistagged facet
#      set and a blocked-dof mix-up — a real but narrow check.
#
#  (2) THE ORDER, measured on the DIRICHLET side against an analytic interface
#      flux this file is never given
#      (tests/test_interface_flux_converges_to_a_known_exact_flux.py):
#      T = 300 + sin(3x) cos(pi y/Ly) drives THIS script unmodified — the
#      shipped fill's natural top/bottom and constant T_OUTER are exactly right
#      for it — and the quantity compared is q_ex = -3 K cos(3 X1) cos(pi y/Ly), a
#      different quantity from the temperature handed in. FEniCSx, 8/16/32/64
#      uniform triangle meshes, max error over INTERIOR interface nodes:
#          2.889e-01  7.243e-02  1.814e-02  4.556e-03   ORDER 1.996 1.998 1.993
#      At the two nodes where the interface meets the outer boundary it is only
#      first order (1.529, 7.142e-01, 3.415e-01, 1.663e-01; order 1.10, 1.06,
#      1.04) even where no end fix-up applies — the end of an interface is a
#      different quantity and is reported apart throughout this corpus.
#
#  (3) THE RETIRED PROJECTED GRADIENT, in the norms it was actually measured
#      in: order ~1 in the interior AWAY FROM THE ENDS (0.93), 0.50 in rms, and
#      non-convergent in the max norm that includes the near-end nodes, where
#      it stalls at 2.6 against a true flux of size 2 to 5. It was written up
#      here as a flat "order 0.00, it never converges", which was true of one
#      norm and was not labelled as such. Order ~1 is what a P1 gradient
#      evaluated ON a boundary is worth, and is the honest reason to prefer the
#      second-order reaction. Not re-measured since the branch was deleted.
Amat = _fp.assemble_matrix(fem.form(a))              # no bcs= on purpose
Amat.assemble()
bvec = _fp.assemble_vector(fem.form(L_vol))          # no lifting, no set_bc
bvec.ghostUpdate()
r = Amat.createVecLeft()
Amat.mult(uh.x.petsc_vec, r)
r.axpy(-1.0, bvec)

wvec = _fp.assemble_vector(fem.form(w_ * ds_if))
wvec.ghostUpdate()
wi = wvec.array[iface_dofs]

Q = np.zeros(len(iface_dofs))
ok = np.abs(wi) > 1e-14
Q[ok] = -r.array[iface_dofs][ok] / wi[ok]

# An interface node that ALSO lies on the outer Dirichlet boundary carries
# the OUTER reaction as well, so its residual is not this interface's flux
# (measured 9.4e-02 there against 2.1e-02 on the interface proper). Take
# the nearest interior interface node rather than exporting a corner value
# that is physically a different quantity. This holds on both sides.
# outer_dofs AS DOF NUMBERS: np.isin reads a Python set as ONE object and matches
# nothing (measured: the corner values went out unreplaced).
suspect = np.isin(iface_dofs, sorted(outer_dofs) if isinstance(outer_dofs, (set, frozenset)) else outer_dofs) | ~ok
good = np.where(~suspect)[0]
if len(good):
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]

T = uh.x.array[iface_dofs]

# EACH SIDE COMPUTES ITS OWN FLUX. NEVER WRITE THE PARTNER'S NEGATED.
#
# At convergence the two fluxes are equal and opposite, and that is a RESULT,
# not a recipe. Setting q_B = -q_A makes it true by construction, and then the
# only quantity the coupling is judged on carries no information: the two sides
# agree because they are one array with a sign flip.
#
# It is detected exactly. Two independently assembled systems do not cancel to
# the last bit, so a bit-exact zero jump is no evidence of
# two solves at all. A real converged pair leaves a small residual mismatch —
# about the size of your interface tolerance — and that mismatch is what shows
# the coupling happened.
#
# Seen in a live run: q_B = -q_A at every interface point, summing to exactly
# zero, from an agent whose iteration history was otherwise genuine. It had
# driven a real Dirichlet-Neumann loop and then faked the one number that
# proves it.
#
# So: recover Q from THIS side's own assembled system, as the block above does
# (r = A u - b_vol on the free interface rows, Q = -r/w). Export that. If your
# partner's numbers are what you are writing, you have not measured anything.
#
# exports.json LAST: the driver takes its existence as proof of success.
# ── EXPORT SELF-CHECK ─ keep this block. It stops the exports that look fine and
#    are worthless: no interface point, or coordinates, values and fluxes of
#    different lengths; a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
#    a flux of ~0 against a nonzero partner); and a flux that is the partner's
#    array negated instead of a recovery from THIS side's own system.
_chk_vals = np.asarray(T, float).ravel()
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
    # coupled run, the term was in the form and ds_if covered no interface facet
    # (tags built from loop positions). The weights w_i tell those apart.
    _chk_w = np.abs(np.asarray(wi, float))
    _chk_on = np.where(~suspect)[0]                  # w_i > 0 and held by no outer value
    _chk_ri = np.abs(np.asarray(r.array, float)[np.asarray(iface_dofs, int)])
    raise SystemExit(f"EXPORT SELF-CHECK: the recovered interface flux Q = -r/w is ~0 (at most "
                     f"{np.abs(_chk_flux).max():.2e}) against an imported flux of up to "
                     f"{np.abs(_chk_qin).max():.2e}. The weights w_i = int phi_i ds over ds_if at "
                     f"the {_chk_w.size} interface dofs are "
                     + ("0 at every one: ds_if covers no facet at these dofs, and nothing "
                        "integrated over it reaches them." if not (_chk_w > 1e-14).any() else
                        f"nonzero at {int((_chk_w > 1e-14).sum())} of them (sum {_chk_w.sum():.3e}); "
                        f"at the {_chk_on.size} of those that no outer value holds, r = A u - b_vol "
                        f"is at most {_chk_ri[_chk_on].max(initial=0.0):.2e}, so there A u equals "
                        f"the volume load b_vol."))
# THE APPLIED FLUX MUST COME BACK, NOT ONLY BE NONZERO. On the free interface rows the recovery
# r = A u - b_vol returns the applied flux's own load, int g phi_i ds, whatever the solver. Measured
# on a coupled round: a side whose Krylov solve ignored the imported flux left r at 2.1e-3 of it,
# which passed the ~0 bar above, while its SOLVE SELF-CHECK share (6.0e-5) sat under that check's
# bar, because the held temperature sets that scale and a missing interface load is of size q h.
if SIDE == "neumann" and _chk_qin.size and np.abs(_chk_qin).max() > 0:
    _chk_gv = _fp.assemble_vector(fem.form(g * w_ * ds_if))
    _chk_gv.ghostUpdate()
    _chk_ids = np.asarray(iface_dofs, int)[np.where(~suspect)[0]]
    _chk_want = np.asarray(_chk_gv.array, float)[_chk_ids]
    _chk_got = np.asarray(r.array, float)[_chk_ids]
    _chk_top = float(np.abs(_chk_want).max(initial=0.0))
    _chk_gap = float(np.abs(_chk_got - _chk_want).max(initial=0.0))
    if _chk_top > 0 and _chk_gap > 0.5 * _chk_top:
        _chk_share = float(np.dot(_chk_got, _chk_want) / np.dot(_chk_want, _chk_want))
        raise SystemExit(
            f"EXPORT SELF-CHECK: the flux this side applied did not come back from its own system. "
            f"On the {_chk_ids.size} interface dofs no outer value holds, r = A u - b_vol carries "
            f"{_chk_share:.1e} of the applied flux's load int g phi_i ds (largest gap {_chk_gap:.2e} "
            f"against {_chk_top:.2e}); where the flux entered the solve the two agree to the "
            f"solver's tolerance. "
            + ("r is ~0 there, so uh solves the system without the flux term: the solve used "
               "L_vol, not L -- a solve placed before the served block that adds g v ds_if to L "
               "(hole 1) does exactly this." if abs(_chk_share) < 0.1 else
               "The load the solve used differs from L on the interface."))
# (Dirichlet role only: a Neumann side's consistent recovery of a CONSTANT
#  applied flux can legitimately reproduce it to the last bit.)
if SIDE == "dirichlet" and _chk_qin.shape == _chk_flux.shape and _chk_flux.size \
        and np.array_equal(_chk_flux, -_chk_qin):
    raise SystemExit("EXPORT SELF-CHECK: the exported flux is the partner's "
                     "array negated, bit for bit: a copy, not a recovery from "
                     "this side's own assembled system")
print(f"[fenics {SIDE}] interface n={len(T)} "
      f"T=[{T.min():.6g},{T.max():.6g}] q=[{Q.min():.6g},{Q.max():.6g}]")

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
    print(f"\nNDOF = {int(len(uh.x.array))}")
except Exception as _ndof_exc:
    print(f"[fenics] could not report NDOF: {_ndof_exc!r}. Your task's"
          f" execution log needs `NDOF = <integer>` on a line of its own,"
          f" so print your own degree-of-freedom count here.")

# PER-LEVEL PERSISTENCE: this level's whole field and its interface trace and
# flux, named by LEVEL, never overwritten by the next level (exports.json is).
# Interpolate THESE onto the probe points your task names. A file the next
# level overwrites cannot carry a mesh study.
# A DUMP DEFECT MUST NOT COST YOU THE SOLVE. exports.json is the driver's
# proof that this participant succeeded, and it is written after these files,
# so an exception here would throw away a coupling iteration that worked.
try:
    with open(f"field_level{LEVEL}.csv", "w") as _f:
        _f.write("x,y,u\n")
        for (_px, _py), _u in zip(xy[:, :2], uh.x.array):
            _f.write(f"{float(_px):.11e},{float(_py):.11e},{float(_u):.11e}\n")
    with open(f"interface_level{LEVEL}.csv", "w") as _f:
        _f.write("x,y,u,qn\n")
        for (_px, _py), _t, _q in zip(_exp_co, T, Q):
            _f.write(f"{float(_px):.11e},{float(_py):.11e},{float(_t):.11e},{float(_q):.11e}\n")
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
    print(f"[fenics per-level dump] level {LEVEL} dump failed: "
          f"{_dump_exc!r}. exports.json is still written, so the coupling\n"
          f"continues, but this level has no field file to hand in. Fix the\n"
          f"names the dump reads and run this level again.")
Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(iface_dofs)),
    "coordinates": _exp_co.tolist(),
    "values": [float(t) for t in T],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))
