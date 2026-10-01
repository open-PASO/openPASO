"""FEniCSx (dolfinx) THERMO-ELASTIC participant for the openPASO `couple` driver.

Steady thermoelasticity on ONE rectangular subdomain of a domain split by a
straight interface at x = IFACE_X, plane strain, small strain:
    -div(k grad T) = f_T
    -div(sigma_tot(u, T)) = f_u,   sigma_tot = 2 mu eps(u) + lam tr(eps(u)) I - beta T I
The heat equation does not depend on u; the displacement depends on T through
the thermal stress. The interface carries BOTH fields as ONE state with THREE
components per point:

    values        = [T, ux, uy]     the interface field
    normal_fluxes = [qn, qx, qy]    the outward flux export, SIGN CONVENTION below

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

SIGN CONVENTION -- the thing a vector coupling gets wrong silently:
    qn       = -(k grad T) . n_own                 n_own = S * e_x, this side's
    (qx, qy) = -(sigma_tot . n_own)                outward normal
the SAME rule for the traction as for the heat flux (minus the flux of the
conserved quantity through n_own). The two sides' exports then CANCEL
componentwise, and the NEUMANN side applies the partner's numbers UNCHANGED
(`L += g * v * ds` and `L += inner(g, v) * ds`), because the natural boundary
terms of the two weak forms are +(k grad T . n_own) v and +(sigma . n_own) . v.
A task that asks for the OUTWARD TRACTION sigma_tot . n_out in a file gets
MINUS (qx, qy); qn as it is.

WEAK FORM OF THE THERMAL TERM. With the discrete temperature T_h in hand,
    int sigma_el(u):eps(v) dx = int f_u.v dx + int beta T_h div(v) dx + int_Gamma (sigma_tot.n).v ds
so the thermal stress is a VOLUME term in T_h, no derivative of T_h needed,
and the imported traction is the partner's sigma_tot . n (thermal part
included), applied as it comes.
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
SIDE      = "neumann"     # "dirichlet" (import T,u; export flux+traction) | "neumann"
PARTNER   = "A"           # the partner's `name` in your couple(...) call
X0, X1    = 0.6, 1.4      # this subdomain's x-extent
Y0, Y1    = 0.0, 1.0      # this subdomain's y-extent
IFACE_X   = 0.6           # the shared interface; must equal X0 or X1
K         = 1.0           # conductivity
LAM, MU   = 560.0, 330.0  # Lame parameters (plane strain)
BETA      = 1.0           # thermal stress coefficient: sigma_tot = sigma_el - BETA*T*I


def F_T(x, y):
    """Heat source f_T(x, y) as NumPy on arrays; `0.0 * x` for none.
    PREFER config.json: state the task's f_T there as source_T (a string in x, y)
    and this body is never used -- see the served block below the constants."""
    return 0.0 * x


def F_U(x, y):
    """Body force (f_x, f_y) as two NumPy arrays; zeros for none.
    PREFER config.json: source_ux and source_uy as the task writes them."""
    return 0.0 * x, 0.0 * y
T_OUTER   = 0.0           # T on the whole NON-interface boundary
UX_OUTER, UY_OUTER = 0.0, 0.0   # u on the whole NON-interface boundary
NX, NY    = 26, 46        # this subdomain's OWN mesh; halve h per level
T_INIT, UX_INIT, UY_INIT = 0.0, 0.0, 0.0   # iteration-1 fallback field
Q_INIT    = (0.0, 0.0, 0.0)                # iteration-1 fallback flux/traction
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

# ── THE PROBLEM'S DATA ARE DATA, NOT CODE (served). config.json may carry this
#    subdomain's box, coefficients and source terms AS THE TASK WRITES THEM --
#    x0, x1, y0, y1, k, lam, mu, beta, iface ("left"|"right"), T_outer, and
#    source_T, source_ux, source_uy as strings in x and y (`^` allowed) -- and
#    when it does they override the constants and the F_T / F_U bodies above.
#    Measured on two coupled runs: the side written as CODE solved a textbook
#    sine source while its own config.json held the task's polynomials; a source
#    typed twice is transcribed once wrong. The audit's equation check reads the
#    same keys, so a side that states them is also the side it can judge.
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
try:
    _cfg_all = json.loads(Path("config.json").read_text() or "{}") if Path("config.json").is_file() else {}
    _cfg_all.update(json.loads(os.environ.get("OPENPASO_CONFIG_JSON") or "{}"))
except (ValueError, TypeError, json.JSONDecodeError):
    _cfg_all = {}
if all(_k in _cfg_all for _k in ("x0", "x1", "y0", "y1")):
    X0, X1, Y0, Y1 = (float(_cfg_all[_k]) for _k in ("x0", "x1", "y0", "y1"))
    _ifc = str(_cfg_all.get("iface", "")).strip().lower()
    IFACE_X = X0 if _ifc == "left" else X1 if _ifc == "right" else IFACE_X
for _nm, _key in (("K", "k"), ("LAM", "lam"), ("MU", "mu"), ("BETA", "beta"), ("T_OUTER", "T_outer")):
    if _key in _cfg_all:
        globals()[_nm] = float(_cfg_all[_key])
_SOURCES_FROM = "code (the F_T / F_U bodies above)"
if _cfg_all.get("source_T") is not None:
    _fT_cfg = _expr_fn(_cfg_all["source_T"])
    def F_T(x, y):                                   # noqa: F811 -- config wins over the body above
        return _fT_cfg(x, y)
    _SOURCES_FROM = "config.json"
if _cfg_all.get("source_ux") is not None and _cfg_all.get("source_uy") is not None:
    _fx_cfg, _fy_cfg = _expr_fn(_cfg_all["source_ux"]), _expr_fn(_cfg_all["source_uy"])
    def F_U(x, y):                                   # noqa: F811
        return _fx_cfg(x, y), _fy_cfg(x, y)
    _SOURCES_FROM = "config.json"
# ALL THREE SOURCES ARE ECHOED: a run whose config carried the heat source and "0.0" for both body-force
# components solved a different momentum problem with the temperature right, and the console said f_T only.
print(f"SOURCES IN USE: from {_SOURCES_FROM}"
      + (f"; f_T = {str(_cfg_all.get('source_T'))[:90]}" if _cfg_all.get("source_T") is not None else "")
      + (f"; f_ux = {str(_cfg_all.get('source_ux'))[:60]}; f_uy = {str(_cfg_all.get('source_uy'))[:60]}"
         if _cfg_all.get("source_ux") is not None else "; f_u = the F_U body above (config carries no source_ux/source_uy)"))

OUTER_X = X0 if abs(IFACE_X - X1) < abs(IFACE_X - X0) else X1
S = 1.0 if IFACE_X > OUTER_X else -1.0     # outward normal at the interface = S*e_x



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
    """imports.json is {partner_name: InterfaceData}; `{}` on iteration 1."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return _partner_block(json.loads(p.read_text() or "{}"))
    except json.JSONDecodeError:
        return None


def sample(imp, key, fallback, y):
    """The partner's (N, 3) samples mapped onto THIS side's interface points,
    COMPONENT BY COMPONENT (one np.interp over a flattened array interleaves
    the components: right length, every number wrong). Returns (len(y), 3)."""
    fb = np.asarray(fallback, float).ravel()
    if not imp or not imp.get("coordinates"):
        return np.tile(fb, (len(y), 1))
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key) or [], float)
    if vs.ndim == 1:
        vs = vs.reshape(-1, 1)
    if vs.shape[0] != ys.size or vs.shape[1] != fb.size:
        return np.tile(fb, (len(y), 1))
    o = np.argsort(ys)
    return np.column_stack([np.interp(y, ys[o], vs[o, c]) for c in range(fb.size)])


imp = read_imports()

# ── HOLE 1 (yours): the mesh and the two spaces. dmesh.create_rectangle over
#    the corners [X0, Y0] and [X1, Y1] with NX by NY triangle cells (the P1
#    idiom on this install; halve h per level); ST the scalar ("Lagrange", 1)
#    fem.functionspace and SU the vector one, fem.functionspace with the
#    element tuple ("Lagrange", 1, (2,)) -- there is NO fem.VectorFunctionSpace
#    on this install (AttributeError, measured).
#    LEAVE BEHIND: domain, ST, SU.
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
ST = fem.functionspace(domain, ("Lagrange", 1))
SU = fem.functionspace(domain, ("Lagrange", 1, (2,)))
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

# ── THE INTERFACE AND OUTER DOF SETS (served: the handshake onto the dofs) ──
# tabulate_dof_coordinates() has ONE ROW PER NODE; in SU's array component c of
# node n sits at index 2*n + c. Interface rows are sorted by y so the export
# order is the same every iteration.
fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)
xyT = ST.tabulate_dof_coordinates()
iface_T = np.where(np.abs(xyT[:, 0] - IFACE_X) < 1e-10)[0]
iface_T = iface_T[np.argsort(xyT[iface_T, 1])]
y_if = xyT[iface_T, 1]
outer_T = np.where((np.abs(xyT[:, 0] - OUTER_X) < 1e-10) |
                   (np.abs(xyT[:, 1] - Y0) < 1e-10) |
                   (np.abs(xyT[:, 1] - Y1) < 1e-10))[0]
xyU = SU.tabulate_dof_coordinates()
iface_U = np.where(np.abs(xyU[:, 0] - IFACE_X) < 1e-10)[0]
iface_U = iface_U[np.argsort(xyU[iface_U, 1])]
outer_U = np.where((np.abs(xyU[:, 0] - OUTER_X) < 1e-10) |
                   (np.abs(xyU[:, 1] - Y0) < 1e-10) |
                   (np.abs(xyU[:, 1] - Y1) < 1e-10))[0]
if len(iface_T) == 0:
    sys.exit(f"no interface DOFs at x={IFACE_X}: this subdomain spans [{X0},{X1}]")
# THE TWO INTERFACE CORNERS BELONG TO THE OUTER BOUNDARY, ON BOTH SIDES: they
# keep the outer Dirichlet value and are never interface-imposed (measured:
# 4.7% in the displacement and 28% in the traction when they were handed over).
corner = (np.abs(y_if - Y0) < 1e-10) | (np.abs(y_if - Y1) < 1e-10)
iface_bc_T = iface_T[~corner]
iface_bc_U = iface_U[~corner]
facets = dmesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], IFACE_X))
tags = dmesh.meshtags(domain, fdim, np.sort(facets), np.full(len(facets), 7, dtype=np.int32))
ds_if = ufl.Measure("ds", domain=domain, subdomain_data=tags)(7)
tT, vT = ufl.TrialFunction(ST), ufl.TestFunction(ST)
uu, vu = ufl.TrialFunction(SU), ufl.TestFunction(SU)

# ── THE INTERFACE ROLE (served) ────────────────────────────────────────────
# Dirichlet role: the partner's [T, ux, uy] become interface Dirichlet
# conditions (bcs_if_T, bcs_if_U) that YOUR solves below must include.
# Neumann role: the partner's [qn, qx, qy] become the natural terms L_T_if and
# L_U_if that YOUR linear forms below must include, UNCHANGED.
bcs_if_T, bcs_if_U = [], []
L_T_if, L_U_if = 0, 0
if SIDE == "dirichlet":
    g = sample(imp, "values", (T_INIT, UX_INIT, UY_INIT), y_if)
    gT = fem.Function(ST)
    gT.x.array[:] = 0.0
    gT.x.array[iface_T] = g[:, 0]
    gU = fem.Function(SU)
    gU.x.array[:] = 0.0
    gU.x.array[2 * iface_U] = g[:, 1]
    gU.x.array[2 * iface_U + 1] = g[:, 2]
    bcs_if_T = [fem.dirichletbc(gT, iface_bc_T.astype(np.int32))]
    bcs_if_U = [fem.dirichletbc(gU, iface_bc_U.astype(np.int32))]
else:
    q = sample(imp, "normal_fluxes", Q_INIT, y_if)
    gq = fem.Function(ST)
    gq.x.array[:] = 0.0
    gq.x.array[iface_T] = q[:, 0]
    gt = fem.Function(SU)
    gt.x.array[:] = 0.0
    gt.x.array[2 * iface_U] = q[:, 1]
    gt.x.array[2 * iface_U + 1] = q[:, 2]
    L_T_if = gq * vT * ds_if
    L_U_if = ufl.inner(gt, vu) * ds_if

# ── HOLE 2 (yours): forms, sources, outer BCs, the two solves, in this order.
#    aT and au: your bilinear forms in tT/vT and uu/vu (conduction with K;
#    plane-strain elasticity with LAM and MU), written as UFL expressions:
#    ufl.inner(., .) for scalar products, `*` for products, `* ufl.dx` for the
#    integral -- never a `.` between UFL objects. L_T_vol: the heat source ALONE
#    (a fem.Function on ST interpolated from F_T, times vT, over dx). fU_h: the
#    body force as a fem.Function on SU interpolated from F_U (np.vstack of its
#    two arrays, shape (2, n) -- NOT transposed: measured, the transpose fails
#    with "Interpolation data has the wrong shape/size"). INTERPOLATION IS A
#    METHOD OF THE FUNCTION: create f = fem.Function(<space>) and call
#    f.interpolate(lambda X: ...) where X is the (3, n) coordinate array
#    (X[0], X[1]); there is NO module-level fem.interpolate(callable, V)
#    (AttributeError, measured) and a two-argument lambda is a TypeError. The strain is
#    ufl.sym(ufl.grad(w)) and the identity ufl.Identity(2) (UFL arguments have
#    no geometric_dimension()). bcs_T and bcs_U: the OUTER Dirichlet conditions, lists of
#    fem.dirichletbc carrying T_OUTER / (UX_OUTER, UY_OUTER) on outer_T /
#    outer_U (int32) -- either fem.dirichletbc(<a fem.Function holding the
#    values>, rows) or fem.dirichletbc(fem.Constant(domain, ...), rows, V)
#    with the SPACE as third argument (a Constant without it is a TypeError,
#    measured). Solve the heat problem with dolfinx.fem.petsc.LinearProblem
#    (aT against L_T_vol + L_T_if, bcs = bcs_T + bcs_if_T, the keyword
#    petsc_options_prefix, ksp preonly with an lu pc) into Th. Then
#    L_U_vol = the body-force term plus the thermal term BETA * Th * div(vu)
#    over dx of the DISCRETE temperature; solve au against L_U_vol + L_U_if
#    with bcs = bcs_U + bcs_if_U into Uh. Keep L_T_vol and L_U_vol apart from
#    the interface terms: the recovery below subtracts the VOLUME loads alone.
#    LEAVE BEHIND: aT, au, L_T_vol, L_U_vol, Th, Uh.
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
aT = K * ufl.inner(ufl.grad(tT), ufl.grad(vT)) * ufl.dx
fT_h = fem.Function(ST)
fT_h.interpolate(lambda X: F_T(X[0], X[1]))
L_T_vol = fT_h * vT * ufl.dx


def eps(w):
    return ufl.sym(ufl.grad(w))


au = ufl.inner(2.0 * MU * eps(uu) + LAM * ufl.tr(eps(uu)) * ufl.Identity(2), eps(vu)) * ufl.dx
fU_h = fem.Function(SU)
fU_h.interpolate(lambda X: np.vstack(F_U(X[0], X[1])))
gT_out = fem.Function(ST)
gT_out.x.array[:] = T_OUTER
gU_out = fem.Function(SU)
gU_out.x.array[0::2] = UX_OUTER
gU_out.x.array[1::2] = UY_OUTER
bcs_T = [fem.dirichletbc(gT_out, outer_T.astype(np.int32))]
bcs_U = [fem.dirichletbc(gU_out, outer_U.astype(np.int32))]
Th = LinearProblem(aT, L_T_vol + L_T_if, bcs=bcs_T + bcs_if_T, petsc_options_prefix="teT",
                   petsc_options={"ksp_type": "preonly", "pc_type": "lu"}).solve()
L_U_vol = ufl.inner(fU_h, vu) * ufl.dx + BETA * Th * ufl.div(vu) * ufl.dx
Uh = LinearProblem(au, L_U_vol + L_U_if, bcs=bcs_U + bcs_if_U, petsc_options_prefix="teU",
                   petsc_options={"ksp_type": "preonly", "pc_type": "lu"}).solve()
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

# ── DID THE INTERFACE DATA ENTER THE SOLVE? (served) ──────────────────────
if SIDE == "dirichlet" and len(iface_bc_T):
    _gap = float(np.abs(Th.x.array[iface_bc_T] - gT.x.array[iface_bc_T]).max())
    _gapU = float(np.abs(Uh.x.array[2 * iface_bc_U] - gU.x.array[2 * iface_bc_U]).max())
    if _gap > 1e-9 * max(1.0, float(np.abs(gT.x.array).max())) or \
            _gapU > 1e-9 * max(1.0, float(np.abs(gU.x.array).max())):
        raise SystemExit("the imposed interface trace is not in the solution: pass bcs_T + bcs_if_T "
                         "and bcs_U + bcs_if_U to the two solves (measured gaps "
                         f"{_gap:.3e} in T, {_gapU:.3e} in ux)")

# ── CONSISTENT OUTWARD FLUX AND TRACTION (served) ─────────────────────────
# ONE FORMULA, BOTH SIDES, BOTH FIELDS: r = A u_h - b_vol on the interface rows
# (UNCONSTRAINED residual, no bcs applied, constrained rows NOT zeroed) is
# int_Gamma (flux . n_own) phi_i ds; dividing by w_i = int_Gamma phi_i ds gives
# the density, and the export is Q = -r / w in the sign convention above. The
# gradient/stress of a P1 solution ON the boundary is only O(h) accurate; this
# reaction is second order (1.996, 1.998, 1.993 on the scalar fixture).
def _residual(a_form, L_vol, sol, V):
    A = _fp.assemble_matrix(fem.form(a_form))       # no bcs= on purpose
    A.assemble()
    b = _fp.assemble_vector(fem.form(L_vol))        # no lifting, no set_bc
    b.ghostUpdate()
    r = A.createVecLeft()
    A.mult(sol.x.petsc_vec, r)
    r.axpy(-1.0, b)
    return r.array.copy()


rT = _residual(aT, L_T_vol, Th, ST)
rU = _residual(au, L_U_vol, Uh, SU)
wT = _fp.assemble_vector(fem.form(ufl.TestFunction(ST) * ds_if))
wT.ghostUpdate()
ones = fem.Constant(domain, np.ones(2, dtype=default_scalar_type))
wU = _fp.assemble_vector(fem.form(ufl.inner(ones, ufl.TestFunction(SU)) * ds_if))
wU.ghostUpdate()
wiT = wT.array[iface_T]
idxU = np.column_stack([2 * iface_U, 2 * iface_U + 1])   # blocked, (n_iface, 2)
wiU = wU.array[idxU]
Q = np.zeros((len(iface_T), 3))
okT = np.abs(wiT) > 1e-14
Q[okT, 0] = -rT[iface_T][okT] / wiT[okT]
okU = np.abs(wiU) > 1e-14
Q[:, 1:][okU] = -rU[idxU][okU] / wiU[okU]
# the two interface corners carry the OUTER reaction too: take the nearest
# interior interface node (a corner value is physically a different quantity)
# outer_dofs AS DOF NUMBERS: np.isin reads a Python set as ONE object and matches
# nothing (measured: the corner values went out unreplaced).
suspect = np.isin(iface_T, sorted(outer_T) if isinstance(outer_T, (set, frozenset)) else outer_T) | ~okT | ~okU.all(axis=1)
good = np.where(~suspect)[0]
if len(good):
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]

V = np.column_stack([Th.x.array[iface_T], Uh.x.array[2 * iface_U], Uh.x.array[2 * iface_U + 1]])
print(f"[fenics {SIDE} thermoelastic] interface n={len(V)} "
      f"T=[{V[:,0].min():.6g},{V[:,0].max():.6g}] ux=[{V[:,1].min():.6g},{V[:,1].max():.6g}] "
      f"uy=[{V[:,2].min():.6g},{V[:,2].max():.6g}] qn=[{Q[:,0].min():.6g},{Q[:,0].max():.6g}]")
print(f"\nNDOF = {ST.dofmap.index_map.size_global + 2 * SU.dofmap.index_map.size_global}")

# ── EXPORT SELF-CHECK ─ keep this block ───────────────────────────────────
if not (np.isfinite(V).all() and np.isfinite(Q).all()):
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface values or fluxes; nothing was exported")
_qin = np.asarray((imp or {}).get("normal_fluxes") or [], float)
if SIDE == "neumann" and _qin.size and np.abs(_qin).max() > 0 and np.abs(Q).max() < 1e-9 * np.abs(_qin).max():
    raise SystemExit("EXPORT SELF-CHECK: the recovered interface flux is ~0 against a nonzero imported "
                     "flux: the imported load never entered the assembled system")
if SIDE == "neumann" and _qin.size:
    # a Neumann side's recovery reproduces the load it applied, componentwise,
    # up to the consistent-to-nodal offset (h^2/6) g''; a scale error is a defect
    _qa = sample(imp, "normal_fluxes", Q_INIT, y_if)
    _int = ~suspect
    for c in range(3):
        _sc = np.abs(_qa[_int, c]).max()
        if _sc > 0 and np.abs(Q[_int, c] + _qa[_int, c]).max() / _sc > 0.3:
            # IT REPORTS, IT DOES NOT EXIT -- the same rule as the scalar sides.
            # This ratio is measured against the PARTNER'S CURRENT iterate, so
            # it can be large at an early or a stalling iterate while the form is
            # correct, and it is evaluated after the solve has already succeeded.
            # Exiting here discards a finished level; the coupling's own residual
            # history is what decides whether the exchange is converging.
            print(f"EXPORT SELF-CHECK WARNING: component {c} of the recovered flux does not match the "
                  f"load applied (max|q_own + q_imported| / max|q_imported| > 0.3). If this persists at "
                  f"every iteration, the imported datum entered the form wrongly scaled or on the wrong "
                  f"facets. The export CONTINUES so this level is not lost.", file=sys.stderr)
if SIDE == "dirichlet" and _qin.shape == Q.shape and Q.size and np.array_equal(Q, -_qin):
    raise SystemExit("EXPORT SELF-CHECK: the exported flux is the partner's array negated, bit for bit")

# PER-LEVEL PERSISTENCE: this level's whole field (x, y, T, ux, uy at the nodes)
# and its interface trace, flux and OUTWARD traction (tx = -qx, ty = -qy), named
# by LEVEL, never overwritten by the next level (exports.json is). Build the
# task's per-level files from these.
# Interpolate THESE onto the probe points your task names. A file the next
# level overwrites cannot carry a mesh study.
# A DUMP DEFECT MUST NOT COST YOU THE SOLVE. exports.json is the driver's
# proof that this participant succeeded, and it is written after these files,
# so an exception here would throw away a coupling iteration that worked.
try:
    xyT_all = ST.tabulate_dof_coordinates()
    with open(f"field_level{LEVEL}.csv", "w") as _f:
        _f.write("x,y,T,ux,uy\n")
        for _i, (_px, _py) in enumerate(xyT_all[:, :2]):
            _f.write(f"{float(_px):.11e},{float(_py):.11e},{float(Th.x.array[_i]):.11e},"
                     f"{float(Uh.x.array[2 * _i]):.11e},{float(Uh.x.array[2 * _i + 1]):.11e}\n")
    with open(f"interface_level{LEVEL}.csv", "w") as _f:
        _f.write("x,y,T,ux,uy,qn,tx,ty\n")
        for _y, (_t, _ux, _uy), (_qn, _qx, _qy) in zip(y_if, V, Q):
            _f.write(f"{float(IFACE_X):.11e},{float(_y):.11e},{float(_t):.11e},{float(_ux):.11e},"
                     f"{float(_uy):.11e},{float(_qn):.11e},{float(-_qx):.11e},{float(-_qy):.11e}\n")
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
    print(f"[fenics_thermoelastic per-level dump] level {LEVEL} dump failed: "
          f"{_dump_exc!r}. exports.json is still written, so the coupling\n"
          f"continues, but this level has no field file to hand in. Fix the\n"
          f"names the dump reads and run this level again.")
# exports.json LAST: the driver takes its existence as proof of success.
Path("exports.json").write_text(json.dumps({
    "field_name": "thermoelastic",
    "n_points": int(len(iface_T)),
    "coordinates": [[float(IFACE_X), float(y)] for y in y_if],
    "values": [[float(a_) for a_ in row] for row in V],
    "normal_fluxes": [[float(a_) for a_ in row] for row in Q],
}, indent=2))
