"""scikit-fem VECTOR participant for the openPASO `couple` driver.

Plane-strain linear elasticity  -div(sigma(u)) = 0  on ONE rectangular
subdomain of a domain split by a straight interface at x = IFACE_X (or
y = IFACE_X when IFACE_AXIS is "y"). Unlike the
scalar (heat) participants, the exchanged interface state is a VECTOR on BOTH
channels:

    values        = displacement       u = (u_x, u_y)   at the interface nodes
    normal_fluxes = interface traction export            (SIGN CONVENTION below)

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

SIGN CONVENTION — the thing a vector coupling gets wrong silently.
`normal_fluxes` is exported as

    q_out = -(sigma . n_own)                       n_own = S * e_x

the SAME convention the shipped scalar participants use for heat
(q_out = -k dT/dn_own) and the one participant_febio.py uses for its 1-D
elastic analogue. Two consequences, both load-bearing:

  * the two sides' exports CANCEL componentwise, because n_own is anti-parallel
    across the interface — that is what makes the interface balance check a
    conservation statement rather than an accident;
  * the NEUMANN side applies the partner's numbers UNCHANGED, as
    `L += dot(g, v) ds`, because the natural boundary term of the elasticity
    weak form is +(sigma . n_own) . v = +q_out_partner . v.

Exporting the raw traction (sigma . n_own) instead flips the sign the Neumann
side applies; the iteration still converges, to the wrong answer.

RELAXATION IS NOT PER COMPONENT. The driver applies ONE theta to the whole
interface state, and the optimal theta is 1/(1+rho) with rho the ratio of the
two subdomains' interface stiffnesses. For a VECTOR interface rho is a matrix,
so u_x and u_y generally want DIFFERENT thetas and the single theta must be
chosen for the WORST component: (1-theta)^2 + rho_c*theta^2 < 1 has to hold for
every component c, so theta < 2/(1+max_c rho_c). Measured on a coupled pair with
traction-free y-faces: rho_x ~ 0.4 while rho_y ~ 1.8, and theta = 1/(1+rho_x)
diverges on the y component while the x component converges — a
half-converging coupling that a single global residual reports only as "did
not converge". Two subdomains of the SAME length and Poisson ratio have
Steklov-Poincare operators that are proportional (S_left = (E_l/E_r) S_right),
so rho collapses to the scalar E_l/E_r and one theta is optimal for both
components; that is why the shipped placeholder geometry splits the strip in
half.
"""
import json
import os
from pathlib import Path

import logging                # scikit-fem logs through it (logging.basicConfig(level=logging.INFO)),
                              # and its output goes to STDERR -- redirect with 2>&1 or lose it
import numpy as np
from skfem import (Basis, BilinearForm, ElementTriP1, ElementVector,
                   FacetBasis, LinearForm, MeshTri, asm, condense, solve)
from skfem.helpers import ddot, sym_grad, trace


# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side.
SIDE      = "dirichlet"   # "dirichlet" (import u, export traction) | "neumann"
PARTNER   = "right"       # name of the partner participant in couple(...)
X0, X1    = 0.0, 0.55     # this subdomain
Y0, Y1    = 0.0, 0.4
IFACE_AXIS = "x"          # WHICH straight line the interface is: "x" -> the line x = IFACE_X
                          # (the subdomains sit side by side) | "y" -> the line y = IFACE_X
                          # (they are stacked). Everything below follows from it.
IFACE_X   = 0.55          # WHERE that line sits: equal to X0 or X1 for axis "x",
                          # to Y0 or Y1 for axis "y"
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
NX, NY    = 46, 26        # this subdomain's own mesh (need not match the partner)
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
try:
    _cfg_all = json.loads(Path("config.json").read_text() or "{}") if Path("config.json").is_file() else {}
    _cfg_all.update(json.loads(os.environ.get("OPENPASO_CONFIG_JSON") or "{}"))
except (ValueError, TypeError, json.JSONDecodeError):
    _cfg_all = {}
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


def sample(imp, key, fallback, y):
    """Map the partner's VECTOR samples onto this participant's y-coordinates,
    COMPONENT BY COMPONENT.

    The driver does no interpolation — non-matching interface meshes are
    handled here, and for a vector field that has to be done per component. One
    np.interp over a flattened (N, 2) array interleaves the two components: the
    result still has the right length, the coupling still converges, and every
    number is wrong.

    Returns (len(y), ncomp). `fallback` is the per-component constant used on
    iteration 1, when imports.json is `{}`.
    """
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
        NX = int(_cfg.get("nx", NX))
        NY = int(_cfg.get("ny", NY))
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
def u_dirichlet(x, y):
    """The prescribed displacement on the non-interface boundary."""
    return (UDX[0] + UDX[1] * x + UDX[2] * y + UDX[3] * y * y,
            UDY[0] + UDY[1] * x + UDY[2] * y + UDY[3] * y * y)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end


imp = read_imports()

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
mesh = MeshTri.init_tensor(np.linspace(X0, X1, NX + 1),
                           np.linspace(Y0, Y1, NY + 1))
elem = ElementVector(ElementTriP1())
basis = Basis(mesh, elem)
nd = basis.nodal_dofs                      # (2, nnodes): node -> (x, y) dof

px, py = mesh.p[0], mesh.p[1]
pa, pl = mesh.p[AX], mesh.p[AL]            # across the interface, and along it
iface_n = np.where(np.abs(pa - IFACE_X) < TOL)[0]
iface_n = iface_n[np.argsort(pl[iface_n])]             # sorted along the interface
y_if = pl[iface_n]
outer_n = np.where((np.abs(pa - OUTER_X) < TOL) |
                   (np.abs(pl - ALO) < TOL) | (np.abs(pl - AHI) < TOL))[0]
# THE TWO INTERFACE CORNERS BELONG TO THE OUTER BOUNDARY, ON BOTH SIDES.
# The two ends of the interface sit on the faces it ends on, which carry a
# prescribed displacement in the un-split problem, so they are Dirichlet nodes
# there and must stay Dirichlet in BOTH subproblems. Handing them to the
# interface instead leaves them unconstrained on the Neumann side: that
# subproblem is still well posed, still converges, and lands a few percent off
# — measured here, 4.7% in the interface displacement and 28% in the interface
# traction, on a coupling whose residual reached 1e-10 and whose flux balanced.
# So the interface Dirichlet set EXCLUDES them; they are still exported,
# because they are still points of the interface.
iface_bc_n = iface_n[(np.abs(pl[iface_n] - ALO) > TOL) &
                     (np.abs(pl[iface_n] - AHI) > TOL)]
iface_bc_dofs = np.concatenate([nd[0, iface_bc_n], nd[1, iface_bc_n]])
outer_dofs = np.concatenate([nd[0, outer_n], nd[1, outer_n]])


@BilinearForm
def stiffness(u, v, w):
    eu, ev = sym_grad(u), sym_grad(v)
    return 2.0 * MU * ddot(eu, ev) + LAM * trace(eu) * trace(ev)


@LinearForm
def body_force(v, w):
    """The loading functional, int_Omega b . v dx.

    `w.x` is the (2, nelems, nqp) array of GLOBAL coordinates of the quadrature
    points, so B_SRC is evaluated exactly where the integration rule needs it
    and a polynomial source is integrated to quadrature accuracy — no detour
    through a P1 interpolant of the source, and no constant standing in for a
    field that varies over the element."""
    bx, by = B_SRC(w.x[0], w.x[1])
    return bx * v[0] + by * v[1]
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end


@LinearForm
def traction(v, w):
    return w["t"][0] * v[0] + w["t"][1] * v[1]


@LinearForm
def unit_load(v, w):
    """w_i = int_Gamma phi_i ds.  The test vector (1,1) makes the SAME scalar
    nodal weight come out on both components."""
    return 1.0 * v[0] + 1.0 * v[1]


# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
A = stiffness.assemble(basis)          # UNCONSTRAINED: condense() below does
b = body_force.assemble(basis)         # not modify A or b in place
# THE VOLUME LOAD ALONE, kept for the traction recovery at the bottom. The
# Neumann branch adds the partner's interface term into `b`; subtracting that
# combined vector is what made the reaction look like zero on that side.
b_vol = b
fbi = FacetBasis(mesh, elem,
                 facets=mesh.facets_satisfying(
                     lambda p: np.abs(p[AX] - IFACE_X) < TOL))

sol = basis.zeros()
ux_d, uy_d = u_dirichlet(px[outer_n], py[outer_n])
sol[nd[0, outer_n]] = ux_d
sol[nd[1, outer_n]] = uy_d
D = outer_dofs
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

# ── WHAT THE SERVED LINES BELOW RELY ON, CHECKED (served) ─ keep this block.
#    y_if is the coordinate ALONG the interface: x on a horizontal one (the
#    name is the vertical case's). iface_bc_n is iface_n without its two end
#    nodes, in the same order. And fbi must sit on the interface line: a
#    FacetBasis over facets that match nothing integrates over NOTHING (skfem
#    only prints "with no facets"), so the Neumann load and the traction
#    weights come out zero. It must carry basis's own element: the served lines
#    interpolate a dof vector of basis on it.
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
if not np.array_equal(np.asarray(nd), np.asarray(basis.nodal_dofs)):
    raise SystemExit("NODE DOFS: nd must be basis.nodal_dofs, shape (2, number of nodes): row 0 holds "
                     "each node's x-dof and row 1 its y-dof, and every served line below indexes it so")
if not np.array_equal(np.asarray(iface_bc_n), np.asarray(iface_n)[~_ends]):
    raise SystemExit("INTERFACE NODES: iface_bc_n must be iface_n without the interface's two end "
                     "nodes, in the same order: the Dirichlet branch below writes the partner's values "
                     "into it row by row")
_fb_pts = (fbi.mesh.p[:, np.unique(fbi.mesh.facets[:, fbi.find])].T if len(fbi.find)
           else np.zeros((0, 2)))
if (not len(_fb_pts) or np.abs(_fb_pts[:, AX] - IFACE_X).max() > TOL
        or _fb_pts[:, AL].min() > ALO + TOL or _fb_pts[:, AL].max() < AHI - TOL):
    raise SystemExit("INTERFACE FACETS: two served lines integrate over fbi, and "
                     + ("fbi holds no facet" if not len(_fb_pts) else
                        "fbi's facets are not exactly the whole interface line")
                     + f". Build it on the facets of the line {'xy'[AX]} = {IFACE_X:g}, and only those: "
                     f"facets_satisfying hands its test the facet midpoints p, with p[0] = x and p[1] = y.")
if fbi.N != basis.N:
    raise SystemExit(f"INTERFACE FACETS: fbi is built on another element than basis (fbi.N = {fbi.N}, "
                     f"basis.N = {basis.N}). The served lines interpolate a dof vector of basis on fbi, and "
                     f"FacetBasis.interpolate takes fbi.N values ('Input array has wrong size.'): build fbi "
                     f"on basis's element, FacetBasis(mesh, basis.elem, facets=...).")

if SIDE == "dirichlet":
    u_if = sample(imp, "values", (UI_X, UI_Y), y_if)
    keep = (np.abs(y_if - ALO) > TOL) & (np.abs(y_if - AHI) > TOL)   # the interface's own ends
    sol[nd[0, iface_bc_n]] = u_if[keep, 0]
    sol[nd[1, iface_bc_n]] = u_if[keep, 1]
    D = np.unique(np.concatenate([outer_dofs, iface_bc_dofs]))
else:
    t_if = sample(imp, "normal_fluxes", (TI_X, TI_Y), y_if)
    gnod = basis.zeros()                   # P1 trace of the partner's samples
    gnod[nd[0, iface_n]] = t_if[:, 0]
    gnod[nd[1, iface_n]] = t_if[:, 1]
    # APPLY the partner's numbers UNCHANGED (+ integral(g . v) ds_interface)
    b = b + asm(traction, fbi, t=fbi.interpolate(gnod))
    # `b_vol` above still holds the VOLUME load alone — the traction recovery
    # below subtracts that, not this, and the distinction is the whole point.

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
sol = solve(*condense(A, b, x=sol, D=D))
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

# ── DID THE PARTNER'S DISPLACEMENT ENTER THE SOLVE? (served) ─ keep this block.
#    A Dirichlet side whose solve dropped the interface dofs returns its own
#    answer, exports it, and the coupling "converges" in two iterations to two
#    fields that disagree at the interface (measured: 150-200 %).
if SIDE == "dirichlet" and len(iface_bc_n):
    _want = np.stack([u_if[keep, 0], u_if[keep, 1]], axis=1)
    _got = np.stack([sol[nd[0, iface_bc_n]], sol[nd[1, iface_bc_n]]], axis=1)
    if np.abs(_got - _want).max() > 1e-9 * max(1.0, float(np.abs(_want).max())):
        raise SystemExit("EXPORT SELF-CHECK: the partner's displacement is not in the solution at the "
                         "interface nodes: the solve must keep the Dirichlet dofs D at the values set in "
                         "`sol` (condense(A, b, x=sol, D=D)); a solve that drops them returns this side's "
                         "own answer and couples to nothing")

# Interface traction export q_out = -(sigma . n_own).
#
# WHY NOT AN L2 PROJECTION OF THE STRESS. That is what this file used to do on
# the Neumann side: project -(sigma(u_h) . n_own) over the whole subdomain and
# sample it at the interface. The gradient of a P1 solution — and therefore the
# stress — is only O(h) accurate ON the boundary; the superconvergence points
# are interior, and the boundary trace is exactly what the coupling reads.
#
# THE CONSISTENT (REACTION) TRACTION. From
#     a(u,v) - (f,v) = int_dOmega (sigma(u) . n) . v ds = -int_Gamma q_out . v ds
# (the second equality is this file's sign convention, q_out = -(sigma . n_own))
# it follows that for every vector basis function phi_i on the interface
#     int_Gamma q_out . phi_i ds = -r_i,   r = A u_h - b_vol
# with r the UNCONSTRAINED residual: A and the loads are assembled with NO
# boundary condition applied and skfem's condense() returns copies, so the
# constrained rows of A still carry the reaction.
#
# ONE FORMULA, BOTH SIDES. An earlier version used that reaction on the
# Dirichlet side and the projection on the Neumann side, reasoning that the
# Neumann interface dofs are free so r comes out ~0 there. That holds only when
# the residual is taken against a load that ALREADY CONTAINS the interface
# term. Subtract the VOLUME load alone and those same rows carry exactly the
# interface functional the partner applied:
#     (A u - b_vol)_i = int_Gamma g . phi_i ds
# On the Dirichlet side there is no interface term, so b == b_vol and the two
# cases are one expression.
#
# WHAT IS MEASURED, AND WHAT IS ONLY ALGEBRA. Handing the NEUMANN side a
# traction and asking for it back is an ASSEMBLY IDENTITY, not a convergence
# test: on free interface rows r = A u - b_vol IS M_Gamma g, so the export is
# -(M_Gamma g)/(M_Gamma 1) and its offset from -g is -(h^2/6) g''(y) for ANY
# correct assembly of ANY equation. The "order 2.00" that used to stand here
# was read off that fixture; it is a property of the P1 boundary mass matrix,
# not of this code — a bare NumPy mass matrix reproduces the same numbers with
# no PDE, no solver and no material in it. That fixture is kept
# (tests/test_interface_flux_recovery.py) for what it really tests, and for a
# VECTOR participant the blocked-dof mapping below is exactly the kind of
# defect it catches and a field-error check does not.
#
# THE ORDER is measured on the DIRICHLET side against an ANALYTIC interface
# flux the participant is never handed
# (tests/test_interface_flux_converges_to_a_known_exact_flux.py). Scalar
# conduction, FEniCSx, the same recovery, 8/16/32/64 uniform triangle meshes,
# max error over interior interface nodes:
#     2.889e-01  7.243e-02  1.814e-02  4.556e-03   ORDER 1.996  1.998  1.993
# and only first order (1.10, 1.06, 1.04) at the two nodes where the interface
# meets the outer boundary, which is why those are handled apart below. There
# is no VECTOR measurement against an analytic traction, and none is claimed.
#
# THE RETIRED L2-PROJECTED GRADIENT, in the norms it was measured in: order ~1
# in the interior AWAY FROM THE ENDS (0.93), 0.50 in rms, and non-convergent in
# the max norm that includes the near-end nodes, where it stalls at 2.6 against
# a true flux of size 2 to 5. It was written up as a flat "order 0.00, it never
# converges", which was true of one norm only. Not re-measured since the branch
# was deleted.
#
# THE WEIGHT IS ONE SCALAR PER NODE, NOT ONE PER DOF. w_i = int_Gamma phi_i ds
# belongs to the NODE, while the dofs are blocked by component; the test vector
# (1,1) in `unit_load` puts that same number on both of a node's dofs, so every
# component gets divided by its own node's weight. `idx` carries the
# (x-dof, y-dof) pair per node, which keeps the componentwise division and the
# blocked indexing together.
r = A @ sol - b_vol                    # r = A u_h - b_vol, no bc applied
wgt = unit_load.assemble(fbi)          # w_i = int_Gamma phi_i ds

idx = np.column_stack([nd[0, iface_n], nd[1, iface_n]])    # (nnode, 2) dofs
wi = wgt[idx]                          # the SAME w_i in both columns
Q = np.zeros_like(wi)
ok = np.abs(wi) > 1e-14
Q[ok] = -r[idx][ok] / wi[ok]

# THE TWO INTERFACE CORNERS ARE ON THE OUTER DIRICHLET BOUNDARY (the faces the
# interface ends on), so their rows carry the OUTER reaction too and their
# residual is not this interface's traction. Take the nearest interior
# interface node rather than exporting a corner value that is physically a
# different quantity. This holds on BOTH sides: the corners are outer-Dirichlet
# in either subproblem. They are found by position (_ends), so an outer_n that
# leaves them out does not let them through.
suspect = _ends | np.isin(iface_n, outer_n) | ~ok.all(axis=1)
good = np.where(~suspect)[0]
if len(good):
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]

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
        for _i in range(mesh.p.shape[1]):
            _f.write(f"{float(mesh.p[0, _i]):.11e},{float(mesh.p[1, _i]):.11e},"
                     f"{float(sol[nd[0, _i]]):.11e},{float(sol[nd[1, _i]]):.11e}\n")
    with open(f"interface_level{LEVEL}.csv", "w") as _f:
        _f.write("x,y,ux,uy,qx,qy\n")
        for _yy, (_ux, _uy), (_qx, _qy) in zip(y_if, [(sol[nd[0, _n]], sol[nd[1, _n]]) for _n in iface_n], Q):
            _px, _py = ((float(IFACE_X), float(_yy)) if AX == 0
                        else (float(_yy), float(IFACE_X)))
            _f.write(f"{_px:.11e},{_py:.11e},{float(_ux):.11e},"
                     f"{float(_uy):.11e},{float(_qx):.11e},{float(_qy):.11e}\n")
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
    print(f"[skfem_elastic per-level dump] level {LEVEL} dump failed: "
          f"{_dump_exc!r}. exports.json is still written, so the coupling\n"
          f"continues, but this level has no field file to hand in. Fix the\n"
          f"names the dump reads and run this level again.")

# ── EXPORT SELF-CHECK ─ keep this block. It stops the three exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
#    a traction of ~0 against a nonzero partner); and a traction that is the
#    partner's array negated instead of a recovery from THIS side's own system.
#    A VECTOR side needs it more, not less: a displacement field that came out
#    ~0 because the load never arrived still couples, still converges and still
#    hands in three tidy levels.
_chk_vals = np.asarray([[sol[nd[0, i]], sol[nd[1, i]]] for i in iface_n], float).ravel()
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
                     "boundary condition that integrates it is missing). Fix the "
                     "application; do not couple on")
# (Dirichlet role only: a Neumann side's consistent recovery of a CONSTANT
#  applied traction can legitimately reproduce it to the last bit.)
if SIDE == "dirichlet" and _chk_qin.shape == _chk_flux.shape and _chk_flux.size \
        and np.array_equal(_chk_flux, -_chk_qin):
    raise SystemExit("EXPORT SELF-CHECK: the exported traction is the partner's "
                     "array negated, bit for bit: a copy, not a recovery from "
                     "this side's own assembled system")

Path("exports.json").write_text(json.dumps({
    "field_name": "displacement",
    "n_points": int(len(iface_n)),
    "coordinates": [([float(IFACE_X), float(yy)] if AX == 0 else [float(yy), float(IFACE_X)])
                    for yy in y_if],
    "values": [[float(sol[nd[0, i]]), float(sol[nd[1, i]])] for i in iface_n],
    "normal_fluxes": [[float(q0), float(q1)] for q0, q1 in Q],
}, indent=2))
