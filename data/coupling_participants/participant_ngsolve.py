"""NGSolve participant for the openPASO `couple` driver.

Steady heat conduction  -div(k grad T) = f  on one rectangular subdomain.
CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.
"""
import json
import os
from pathlib import Path

import ngsolve                # the MODULE, so ngsolve.ngsglobals.msg_level = 3 resolves:
                              # a run log that must carry this code's own output needs it,
                              # and `from ngsolve import ...` alone leaves `ngsolve` undefined
import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import (BND, VERTEX, BilinearForm, GridFunction, H1, LinearForm, Mesh,
                     NodeId, TaskManager, ds, dx, grad)


# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side; the payload that served
#    this script gives the exact block for the RIGHT / Neumann side.
SIDE      = "dirichlet"   # "dirichlet" | "neumann"
PARTNER   = "right"       # name of the partner participant in couple(...)
X0, X1    = 0.0, 0.6      # this subdomain
Y0, Y1    = 0.0, 0.4
IFACE_AXIS = "x"          # WHICH straight line the interface is: "x" -> the line x = IFACE_X
                          # (the subdomains sit side by side) | "y" -> the line y = IFACE_X
                          # (they are stacked). Everything below follows from it.
IFACE_X   = 0.6           # shared interface (X0/X1 for axis "x", Y0/Y1 for axis "y")
K         = 0.8           # conductivity


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

    HOW THIS ENTERS NGSolve (your solve below has to do it): NGSolve's symbolic
    `x`, `y` are CoefficientFunctions and carry NO NumPy ufuncs, so do NOT call
    this function on them (np.sin(x) raises; np.zeros_like(x) silently returns
    a 0-d object array and the source collapses to a constant), and do NOT wrap
    the function -- CoefficientFunction(F_SRC) is a TypeError ("incompatible
    constructor arguments", measured). Sample it at the mesh vertices instead,
    np.array([v.point for v in mesh.vertices]), into a P1 GridFunction on your
    space (gf.vec.FV().NumPy()[vertex_dofs] = F_SRC(vx, vy)); a GridFunction IS
    a CoefficientFunction and integrates as gf * v * dx -- the P1 interpolant
    of the source, quadrature error O(h^2), the order of the discretisation.
    """
    return np.zeros_like(x)
T_OUTER   = 320.0         # Dirichlet value on the held NON-interface boundary
REACTION  = 0.0           # c in -div(K grad T) + c T = f (0 when there is none)
# WHICH NON-INTERFACE EDGES ARE HELD IS YOUR PROBLEM'S TO SAY, NOT THIS FILE'S:
# a default here once chose a boundary condition for problems it never saw, and
# a field obeying the wrong condition converges cleanly with nothing to show it.
# True if the two edges the interface ends on are held too, False if they are
# natural (zero flux); the script refuses to run until you set it.
FULL_OUTER_DIRICHLET = None  # <-- True or False, FROM YOUR PROBLEM STATEMENT
NX, NY    = 24, 16        # this subdomain's own mesh (netgen maxh derived below)
T_INIT    = 310.0          # iteration-1 fallback interface temperature
Q_INIT    = 0.0           # iteration-1 fallback interface flux
# ─────────────────────────────────────────────────────────────────────────

# ── THE PER-LEVEL RULE (served). A ./config.json {"level": k, "nx": .., "ny": ..}
#    next to this script overrides NX, NY and names the level. The dumps at the
#    foot of this file carry that level in their NAME, so a mesh study leaves
#    one file per level instead of the fine mesh overwriting the coarse ones.
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

# ── THE PROBLEM'S DATA ARE DATA, NOT CODE (served). config.json may carry this
#    subdomain's data AS THE TASK WRITES THEM -- side, partner; x0, x1, y0, y1;
#    iface ("left"|"right"|"bottom"|"top", or the interface coordinate) and
#    iface_axis ("x"|"y"); k; reaction; source_expr (a string in x and y, `^`
#    allowed); outer (the value on the held edges) or outer_expr (a string in x
#    and y); full_outer_dirichlet (true|false) -- and they override the constants
#    above. The audit's equation check reads the same keys: a side that states
#    them is a side it can judge, and one that keeps them in code is not judged.
def _expr_fn(expr):
    src = str(expr).replace("^", "**")
    code = compile(src, "<expr>", "eval")
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
for _nm, _key in (("K", "k"), ("REACTION", "reaction"), ("T_OUTER", "outer")):
    if _cfg_all.get(_key) is not None:
        globals()[_nm] = float(_cfg_all[_key])
if isinstance(_cfg_all.get("full_outer_dirichlet"), bool):
    FULL_OUTER_DIRICHLET = _cfg_all["full_outer_dirichlet"]
_SOURCE_FROM = "code (the F_SRC body above)"
if _cfg_all.get("source_expr") is not None:
    _f_cfg = _expr_fn(_cfg_all["source_expr"])
    def F_SRC(x, y):                                   # noqa: F811 -- config wins over the body above
        return _f_cfg(x, y)
    _SOURCE_FROM = "config.json"
G_OUTER = ((lambda x, y, _g=_expr_fn(_cfg_all["outer_expr"]): _g(x, y))
           if _cfg_all.get("outer_expr") is not None else (lambda x, y: 0.0 * x + T_OUTER))
print(f"SOURCES IN USE: from {_SOURCE_FROM}"
      + (f"; f = {str(_cfg_all.get('source_expr'))[:80]}" if _cfg_all.get("source_expr") is not None else "")
      + f"; k = {K:g}; reaction = {REACTION:g}; outer value stated = "
      + (f"{str(_cfg_all.get('outer_expr'))[:60]}" if _cfg_all.get("outer_expr") is not None else
         f"{T_OUTER:g}" + ("" if _cfg_all.get("outer") is not None else " (T_OUTER in this file)")))
if FULL_OUTER_DIRICHLET is None:
    raise SystemExit("FULL_OUTER_DIRICHLET is unset: say whether the two non-interface edges the "
                     "interface ends on are held (True) or natural (False), from your problem "
                     "statement -- in this file or as \"full_outer_dirichlet\" in config.json. A "
                     "default here would be choosing your boundary condition for you.")

MAXH  = min((X1 - X0) / NX, (Y1 - Y0) / NY)   # netgen's scalar mesh size
ORDER = 1                                     # H1 order (nodal == vertex dofs)

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
    """Interpolate the partner's samples onto this participant's y-coordinates."""
    if not imp or not imp.get("coordinates"):
        return np.full(len(y), float(fallback))
    ys = np.array([c[AL] for c in imp["coordinates"]], float)   # the coordinate ALONG the interface
    vs = np.asarray(imp.get(key, []), float).ravel()
    if vs.size != ys.size:
        return np.full(len(y), float(fallback))
    o = np.argsort(ys)
    return np.interp(y, ys[o], vs[o])


imp = read_imports()

# MAKE THIS CODE SPEAK, BEFORE THE SOLVE RUNS. It is silent by default, and a
# per-level run log carrying no line the solver itself emitted cannot
# establish which code ran on this side, however right its numbers are.
# It sits HERE, beside the level rule, and not up with the imports:
# measured over agent-written participants, a line placed in the import
# block survived in about half of them because that block gets rewritten,
# while everything beside the level rule survived in all of them.
ngsolve.ngsglobals.msg_level = 3

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
# ── mesh: SplineGeometry.AddRectangle edge order is bottom, right, top, left ──
geo = SplineGeometry()
geo.AddRectangle((X0, Y0), (X1, Y1),
                 bcs=(("bottom", "interface", "top", "outer") if ON_RIGHT else
                      ("bottom", "outer", "top", "interface")))
mesh = Mesh(geo.GenerateMesh(maxh=MAXH))

_held = "outer" + ("|bottom|top" if FULL_OUTER_DIRICHLET else "")
fes = H1(mesh, order=ORDER,
         dirichlet=(_held + "|interface" if SIDE == "dirichlet" else _held))
u, v = fes.TnT()

# vertex -> dof map and the interface / outer vertex sets (ORDER 1: dof per vertex)
vdof = np.array([fes.GetDofNrs(NodeId(VERTEX, i))[0] for i in range(mesh.nv)], int)
vxy = np.array([mesh.vertices[i].point for i in range(mesh.nv)], float)

iface_v = np.where(np.abs(vxy[:, 0] - IFACE_X) < TOL)[0]
iface_v = iface_v[np.argsort(vxy[iface_v, 1])]           # sorted by y
y_if = vxy[iface_v, 1]
iface_dofs = vdof[iface_v]
outer_v = np.where((np.abs(vxy[:, 0] - OUTER_X) < TOL)
                   | (bool(FULL_OUTER_DIRICHLET) & ((np.abs(vxy[:, 1] - Y0) < TOL)
                                                    | (np.abs(vxy[:, 1] - Y1) < TOL))))[0]
outer_dofs = vdof[outer_v]

a = BilinearForm(fes)
a += K * grad(u) * grad(v) * dx
if REACTION:
    a += REACTION * u * v * dx
f = LinearForm(fes)
# VOLUMETRIC SOURCE. F_SRC is sampled at the vertices and carried by a
# GridFunction, which IS a CoefficientFunction, so the linear form's source
# varies in space instead of being the constant it used to be. ORDER = 1, so
# this is the P1 interpolant of the source; its quadrature error is O(h^2), the
# same order as the P1 discretization error itself. Do NOT hand F_SRC ngsolve's
# symbolic x, y instead: a CoefficientFunction carries NO NumPy ufuncs, so
# np.sin(x) raises — and np.zeros_like(x) does NOT raise, it returns a 0-d
# OBJECT array, so a polynomial source collapses to a constant and this
# subdomain solves the wrong problem with no error raised anywhere.
gff = GridFunction(fes)
gff.vec[:] = 0.0
gff.vec.FV().NumPy()[vdof] = np.broadcast_to(
    np.asarray(F_SRC(vxy[:, 0], vxy[:, 1]), float), (mesh.nv,))
f += gff * v * dx
# THE VOLUME LOAD ALONE, in its own form. The Neumann branch adds the partner's
# interface term into `f`; the flux recovery at the bottom must subtract the
# volume load WITHOUT it, on both sides — subtracting the combined vector is
# what made the reaction look like zero on the Neumann side.
f_vol = LinearForm(fes)
f_vol += gff * v * dx

gfu = GridFunction(fes)                    # also carries the Dirichlet data
gfu.vec[:] = 0.0
for i_v, d in zip(outer_v, outer_dofs):
    gfu.vec[int(d)] = float(G_OUTER(vxy[i_v, 0], vxy[i_v, 1]))
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

# ── WHAT THE SERVED LINES BELOW RELY ON, CHECKED (served) ─ keep this block.
#    y_if is the coordinate ALONG the interface: x on a horizontal one (the
#    name is the vertical case's). And the two served lines that integrate over
#    ds("interface") need a boundary of exactly that name on the interface line:
#    NGSolve integrates a ds() over a name the mesh does not carry over NOTHING,
#    with no error, so the Neumann load and the flux weights come out zero.
y_if = np.asarray(y_if, float)
if (y_if.size != len(iface_v) or y_if.size < 2 or np.any(np.diff(y_if) <= 0)
        or abs(y_if[0] - ALO) > TOL or abs(y_if[-1] - AHI) > TOL):
    raise SystemExit(f"INTERFACE NODES: y_if must hold the coordinate ALONG the interface ({'xy'[AL]}), "
                     f"one per node of iface_v in the same order, strictly increasing from {ALO:g} to "
                     f"{AHI:g}; it holds {y_if.size} value(s) for {len(iface_v)} node(s)"
                     + (f", from {y_if.min():g} to {y_if.max():g}" if y_if.size else "")
                     + (f"; only {np.unique(np.round(y_if, 12)).size} of them distinct -- a node "
                        f"listed once per edge it touches is listed twice"
                        if 0 < np.unique(np.round(y_if, 12)).size < y_if.size else ""))
_if_pts = np.array([mesh[_n].point for _el in mesh.Elements(BND) if _el.mat == "interface"
                    for _n in _el.vertices], float).reshape(-1, 2)
if (not len(_if_pts) or np.abs(_if_pts[:, AX] - IFACE_X).max() > TOL
        or _if_pts[:, AL].min() > ALO + TOL or _if_pts[:, AL].max() < AHI - TOL):
    raise SystemExit("BOUNDARY NAME: two served lines integrate over ds(\"interface\"), and on this mesh "
                     + ("no boundary carries that name" if not len(_if_pts) else
                        "that name is not exactly the whole interface line")
                     + f" (the mesh's boundary names: {sorted(set(mesh.GetBoundaries()))}). Name the edge "
                     f"on the line {'xy'[AX]} = {IFACE_X:g}, and only that edge, \"interface\": the bcs of "
                     f"AddRectangle are YOUR names, given in the edge order bottom, right, top, left.")
# THE INTERFACE DOFS ARE THE INTERFACE NODES' OWN. Every served line below writes
# the partner's trace, reads the flux and exports the values THROUGH iface_dofs,
# so a list naming other dofs puts the trace on other nodes and exports their
# values under the interface's coordinates -- and a check reading the same list
# agrees with it. Measured on a coupled run: iface_dofs held the dofs of the
# mesh's first vertices (its corners and part of an outer edge), the interface
# kept 0.0 at most of its nodes, and the coupling converged with every exchange
# check passing.
_d2p = {int(fes.GetDofNrs(NodeId(VERTEX, _i))[0]): mesh.vertices[_i].point for _i in range(mesh.nv)}
_line = {_d for _d, _pt in _d2p.items() if abs(_pt[AX] - IFACE_X) < TOL}
_ids = [int(_d) for _d in iface_dofs]
_bad = [(_k, _d) for _k, _d in enumerate(_ids)
        if _k >= y_if.size or _d not in _line or abs(_d2p[_d][AL] - float(y_if[_k])) > TOL]
_missed = len(_line - set(_ids))
if _bad or _missed or len(_ids) != y_if.size:
    _k, _d = _bad[0] if _bad else (None, None)
    raise SystemExit(
        f"INTERFACE DOFS: iface_dofs[k] must be the dof of the interface node at y_if[k], one "
        f"per node; iface_dofs has {len(_ids)} entries for {y_if.size} nodes"
        + (f", and {len(_bad)} of them are not that node's dof (the first: iface_dofs[{_k}] = {_d}, "
           + (f"the dof of the vertex at ({_d2p[_d][0]:g}, {_d2p[_d][1]:g})" if _d in _d2p else
              "no vertex's dof")
           + ((f", where the interface node is at ({float(IFACE_X):g}, {float(y_if[_k]):g})"
               if AX == 0 else
               f", where the interface node is at ({float(y_if[_k]):g}, {float(IFACE_X):g})")
              if _k < y_if.size else "") + ")" if _bad else "")
        + (f"; {_missed} of the {len(_line)} vertices on the interface line have no entry"
           if _missed else "")
        + ". The served lines below write the trace, read the flux and export the values "
          "through this list. The dof of vertex number i is fes.GetDofNrs(NodeId(VERTEX, i))[0], "
          "with i the vertex's own number (v.nr for a mesh vertex v).")
# THE HELD EDGES ARE WHAT FULL_OUTER_DIRICHLET SAYS. A space whose dirichlet= set
# leaves the two edges the interface ends on free while the problem holds them (or
# the reverse) solves a different problem and converges cleanly to it (measured:
# a side copied "the NON-interface x-boundary" and left both edges natural).
_free = fes.FreeDofs()
_wrong = []
for _i in range(mesh.nv):
    _pt = mesh.vertices[_i].point
    if not (abs(_pt[AL] - ALO) < TOL or abs(_pt[AL] - AHI) < TOL):
        continue                                        # not on those two edges
    if abs(_pt[AX] - LO) < TOL or abs(_pt[AX] - HI) < TOL:
        continue                                        # a corner: owned by its other edge too
    _is_free = bool(_free[fes.GetDofNrs(NodeId(VERTEX, _i))[0]])
    if bool(FULL_OUTER_DIRICHLET) == _is_free:
        _wrong.append(_i)
if _wrong:
    raise SystemExit(f"OUTER BOUNDARY: FULL_OUTER_DIRICHLET is {bool(FULL_OUTER_DIRICHLET)}, but "
                     f"{len(_wrong)} vertices on the two edges the interface ends on are "
                     f"{'FREE' if FULL_OUTER_DIRICHLET else 'HELD'} in fes.FreeDofs(): the dirichlet= "
                     f"names of the space must {'include' if FULL_OUTER_DIRICHLET else 'leave out'} "
                     f"those two edges.")

if SIDE == "dirichlet":
    T_if = sample(imp, "values", T_INIT, y_if)
    for d, t in zip(iface_dofs, T_if):
        gfu.vec[int(d)] = float(t)
else:
    q_if = sample(imp, "normal_fluxes", Q_INIT, y_if)
    gfun = GridFunction(fes)               # P1 trace of the partner's samples
    gfun.vec[:] = 0.0
    for d, q in zip(iface_dofs, q_if):
        gfun.vec[int(d)] = float(q)
    f += gfun * v * ds("interface")        # APPLY the partner's number unchanged

# THE SOLVE AND THE RECOVERY RUN AT MODULE LEVEL, NOT INSIDE A `with` BLOCK: the hole
# sat inside `with TaskManager():`, and a fill written at column 0 ended that block and
# broke the served lines after it with an IndentationError (measured in 3 of 5 cells).
# Wrap your own solve in `with TaskManager():` if you want its threads.
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
a.Assemble()
f.Assemble()
f_vol.Assemble()
res = f.vec.CreateVector()
res.data = f.vec - a.mat * gfu.vec
gfu.vec.data += a.mat.Inverse(fes.FreeDofs(),
                              inverse="sparsecholesky") * res
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

# ── DID THE PARTNER'S TRACE ENTER THE SOLVE? (served) ─ keep this block.
#    A Dirichlet side whose solve freed the interface dofs returns its own
#    answer there, and the coupling "converges" to two fields that disagree.
#    Measured on a coupled run: the interface was left out of the space's
#    dirichlet= set and the solved vector replaced the imported values, so the
#    side exported its own insulated trace; nothing stopped it. The two end
#    nodes are left out -- the outer boundary may hold them too. The nodes
#    are the mesh's own on the interface line, never the list the trace was
#    written through: a list naming the wrong dofs agrees with itself.
_tv = sorted((float(_d2p[_d][AL]), _d) for _d in _line)
_tv = [(_a, _d) for _a, _d in _tv if ALO + TOL < _a < AHI - TOL]
if SIDE == "dirichlet" and _tv:
    _gap = max(abs(float(gfu.vec[_d]) - float(_t)) for (_a, _d), _t in
               zip(_tv, sample(imp, "values", T_INIT, np.array([_a for _a, _ in _tv], float))))
    if _gap > 1e-9 * max(1.0, float(np.abs(np.asarray(T_if, float)).max())):
        raise SystemExit("EXPORT SELF-CHECK: the partner's temperature is not in the solution at "
                         "the interface nodes (largest gap %.3e): on the Dirichlet side the "
                         "interface must be held -- named in the space's dirichlet= set, or "
                         "cleared from the mask the solve inverts on -- and the solve must keep "
                         "the values already in gfu. A solve that frees them returns this "
                         "side's own answer and couples to nothing." % _gap)

# Outward normal flux density q = -(k grad T).n on the interface.
#
# WHY NOT AN L2 PROJECTION OF THE GRADIENT. That is what this file used to
# do: project -k dT/dx over the whole subdomain and sample it at the
# interface. The gradient of a P1 solution is only O(h) accurate ON the
# boundary — the superconvergence points are interior — and the boundary
# trace is exactly what the coupling reads. Measured against a manufactured
# solution with a known exact interface flux, the projection converges at
# order ~1 while the consistent flux below converges at ~2, so the recovery,
# not the physics and not the partner, was setting the answer.
#
# THE CONSISTENT (REACTION) FLUX. From
#     a(u,v) - (f,v) = int_dOmega (k grad u . n) v ds = -int_Gamma qn v ds
# it follows that for every basis function phi_i on the interface
#     int_Gamma qn phi_i ds = -r_i,   r = A u_h - b
# with r the UNCONSTRAINED residual: NGSolve's a.mat and f.vec are exactly
# that — the Dirichlet condition lives in fes.FreeDofs() at solve time and
# never touches the assembled operator, so the constrained rows still carry
# the reaction. Dividing by w_i = int_Gamma phi_i ds turns the functional
# into a density the partner can interpolate pointwise.
# ONE FORMULA, BOTH SIDES. An earlier version used the reaction on the
# Dirichlet side and an L2-projected gradient on the Neumann side, on the
# reasoning that the Neumann interface dofs are free, so r comes out ~0
# there. That holds only when the residual is taken against a load that
# ALREADY CONTAINS the interface term. Subtract the VOLUME load alone and
# those same rows carry exactly the interface functional the partner
# applied. On the Dirichlet side there is no interface term, so f_vol == f
# and the two cases are one expression.
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
# the same formulation, 8/16/32/64 uniform triangle meshes, max error over
# interior interface nodes:
#   2.889e-01  7.243e-02  1.814e-02  4.556e-03   ORDER 1.996 1.998 1.993
# and only first order (1.10, 1.06, 1.04) at the two nodes where the
# interface meets the outer boundary, handled apart just below.
#
# THE RETIRED L2-PROJECTED GRADIENT, in the norms it was measured in: order
# ~1 in the interior AWAY FROM THE ENDS (0.93), 0.50 in rms, and
# non-convergent in the max norm that includes the near-end nodes, where it
# stalls at 2.6 against a true flux of size 2 to 5. It was written up as a
# flat "order 0.00, it never converges", which was true of one norm only.
# Not re-measured since the branch was deleted.
rvec = f.vec.CreateVector()
rvec.data = a.mat * gfu.vec - f_vol.vec    # r = A u_h - b_vol, no bc
fw = LinearForm(fes)
fw += v * ds("interface")                  # w_i = int_Gamma phi_i ds
fw.Assemble()

r_if = np.array([rvec[int(d)] for d in iface_dofs], float)
w_if = np.array([fw.vec[int(d)] for d in iface_dofs], float)
Q = np.zeros(len(iface_dofs))
ok = np.abs(w_if) > 1e-14
Q[ok] = -r_if[ok] / w_if[ok]

# An interface node that ALSO lies on the outer Dirichlet boundary
# carries the OUTER reaction as well, so its residual is not this
# interface's flux. Take the nearest interior interface node rather than
# exporting a corner value that is physically a different quantity.
# outer_dofs AS DOF NUMBERS, whatever the hole built: np.isin reads a Python set
# as ONE object and a BitArray (fes.GetDofs(...)) as bits, and on four coupled
# runs this rule matched nothing and the corner values went out unreplaced.
_od = outer_dofs
if type(_od).__name__ == "BitArray":
    _od = [_i for _i in range(len(_od)) if _od[_i]]
elif isinstance(_od, (set, frozenset)):
    _od = sorted(_od)
suspect = np.isin(iface_dofs, np.asarray(_od, int)) | ~ok
good = np.where(~suspect)[0]
if len(good):
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]
# ── EXPORT SELF-CHECK ─ keep this block. It stops the three exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
#    a flux of ~0 against a nonzero partner); and a flux that is the partner's
#    array negated instead of a recovery from THIS side's own system.
_chk_vals = np.asarray([gfu.vec[int(d)] for d in iface_dofs], float).ravel()
_chk_flux = np.asarray(Q, float).ravel()
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
# THE SOLVE ANSWERS FOR ITS OWN SYSTEM. On every dof that fes.FreeDofs() leaves
# free and that is off the interface, a solved system leaves r = A u - f at
# round-off. Measured on a coupled run: a Dirichlet side built its solve mask by
# looping over a BitArray, which yields True/False rather than dof numbers; the
# mask held one bit, its solve moved nothing, it exported the imported trace over
# a zero interior, and
# its coupling converged in ten iterations with every exchange check passing. A
# correction applied to the load alone, instead of to the residual of the values
# already in gfu, fails here the same way.
_chk_fd = fes.FreeDofs()
_chk_on = fes.GetDofs(mesh.Boundaries("interface"))
_chk_in = np.array([d for d in range(fes.ndof) if _chk_fd[d] and not _chk_on[d]], int)
try:                                   # the scale is |A| |u|, not |A u|: a constant
    _chk_i, _chk_j, _chk_v = a.mat.COO()     # field has A u ~ 1e-13 and is solved all the same
    _chk_rows = np.bincount(np.asarray(_chk_i, int), weights=np.abs(np.asarray(_chk_v, float)),
                            minlength=fes.ndof)
except Exception:                      # noqa: BLE001 -- no entries to read: no check
    _chk_rows = None
if _chk_in.size and _chk_rows is not None and not getattr(a, "condense", False):
    _chk_Au = f.vec.CreateVector()
    _chk_Au.data = a.mat * gfu.vec
    _chk_A = np.asarray(_chk_Au.FV().NumPy(), float)
    _chk_F = np.asarray(f.vec.FV().NumPy(), float)
    _chk_U = np.abs(np.asarray(gfu.vec.FV().NumPy(), float)).max()
    _chk_r = np.abs(_chk_A - _chk_F)[_chk_in]
    _chk_sc = max(float(np.abs(_chk_F).max()), float(_chk_rows.max()) * float(_chk_U))
    if _chk_sc > 0 and _chk_r.max() > 1e-6 * _chk_sc:
        raise SystemExit(
            f"SOLVE SELF-CHECK: off the interface, on the dofs fes.FreeDofs() leaves free, "
            f"r = A u - f reaches {_chk_r.max():.2e} against a system scale of {_chk_sc:.2e} "
            f"({int((_chk_r > 1e-6 * _chk_sc).sum())} of {_chk_in.size} dofs); a solved "
            f"system leaves round-off there. The field was not solved for those dofs from "
            f"this a and f. Three ways measured to get here: a mask whose bits are not the "
            f"free dofs (a loop over a BitArray yields True/False, not dof numbers), a "
            f"correction applied to the load alone instead of to the residual that the "
            f"values already in gfu leave, and a solve whose result went into another "
            f"vector (M.Mult(x, y) writes M x INTO y). The flux recovery above reads the "
            f"same a.mat, so nothing was exported.")

# THE LOAD THE SOLVE USED CARRIES THE SOURCE. Off every boundary, f (the solve's
# load) and f_vol (the one the flux recovery subtracts) are the same volume
# integral. Measured on coupled runs: f built and never assembled (the side solved
# with no source through a whole ladder, and the check above -- which compares A u
# with that same f -- passed it); and f assembled, then written over by the solve's
# own Mult call.
_chk_bd = fes.GetDofs(mesh.Boundaries(".*"))
_chk_io = np.array([d for d in range(fes.ndof) if not _chk_bd[d]], int)
if _chk_io.size:
    _chk_lf = np.asarray(f.vec.FV().NumPy(), float)[_chk_io]
    _chk_lv = np.asarray(f_vol.vec.FV().NumPy(), float)[_chk_io]
    _chk_m = max(float(np.abs(_chk_lf).max()), float(np.abs(_chk_lv).max()))
    try:
        _chk_xy = np.array([mesh.vertices[_i].point for _i in range(mesh.nv)], float)
        _chk_s = float(np.abs(np.asarray(F_SRC(_chk_xy[:, 0], _chk_xy[:, 1]), float)).max())
    except Exception:                  # noqa: BLE001 -- a source this line cannot sample: no check
        _chk_s = 0.0
    if (_chk_m > 0 and np.abs(_chk_lf - _chk_lv).max() > 0.25 * _chk_m) or (_chk_m == 0 and _chk_s > 0):
        raise SystemExit(
            f"LOAD: off the boundary, the load the solve used (f) reaches {np.abs(_chk_lf).max():.3e} "
            f"and the volume load the flux recovery subtracts (f_vol) {np.abs(_chk_lv).max():.3e}"
            + (f", while F_SRC reaches {_chk_s:.3e}" if _chk_m == 0 else "")
            + ". There both are the same volume integral of the source. Two ways measured to get "
              "here: a form built and never assembled (it holds zeros), and a vector written over "
              "after its assembly -- M.Mult(x, y) writes M x INTO y. Nothing was exported.")
# WHAT THE HELD NON-INTERFACE EDGES HOLD, read from the solved field: the line
# printed before the solve says only what this file or config.json states.
_chk_hd = [(_d, _pt) for _d, _pt in _d2p.items() if not _chk_fd[_d] and abs(_pt[AX] - IFACE_X) > TOL]
if _chk_hd:
    _chk_hu = np.array([float(gfu.vec[_d]) for _d, _ in _chk_hd])
    _chk_hg = np.array([float(G_OUTER(float(_pt[0]), float(_pt[1]))) for _, _pt in _chk_hd])
    print(f"HELD OUTER EDGES: the solved field holds {_chk_hu.min():g} .. {_chk_hu.max():g} there"
          + ("" if np.abs(_chk_hu - _chk_hg).max() <= 1e-9 * max(1.0, float(np.abs(_chk_hg).max())) else
             f", NOT the value stated for them ({_chk_hg.min():g} .. {_chk_hg.max():g}): the stated "
             f"value is not your problem's (T_OUTER, or \"outer\" / \"outer_expr\" in config.json), "
             f"or those dofs are not held at it"))

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
    print(f"\nNDOF = {int(fes.ndof)}")
except Exception as _ndof_exc:
    print(f"[ngsolve] could not report NDOF: {_ndof_exc!r}. Your task's"
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
        for _i in range(mesh.nv):
            _p = mesh.vertices[_i].point
            _dn = fes.GetDofNrs(NodeId(VERTEX, _i))[0]
            _f.write(f"{float(_p[0]):.11e},{float(_p[1]):.11e},"
                     f"{float(gfu.vec[int(_dn)]):.11e}\n")
    with open(f"interface_level{LEVEL}.csv", "w") as _f:
        _f.write("x,y,u,qn\n")
        for _yy, _d, _q in zip(y_if, iface_dofs, Q):
            _px, _py = ((float(IFACE_X), float(_yy)) if AX == 0
                        else (float(_yy), float(IFACE_X)))
            _f.write(f"{_px:.11e},{_py:.11e},"
                     f"{float(gfu.vec[int(_d)]):.11e},{float(_q):.11e}\n")
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
    print(f"[ngsolve per-level dump] level {LEVEL} dump failed: "
          f"{_dump_exc!r}. exports.json is still written, so the coupling\n"
          f"continues, but this level has no field file to hand in. Fix the\n"
          f"names the dump reads and run this level again.")

Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(iface_v)),
    "coordinates": [([float(IFACE_X), float(yy)] if AX == 0 else [float(yy), float(IFACE_X)])
                    for yy in y_if],
    "values": [float(gfu.vec[int(d)]) for d in iface_dofs],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))
