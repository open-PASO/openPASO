"""NGSolve VECTOR participant for the openPASO `couple` driver.

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
(q_out = -k dT/dn_own). The two sides' exports therefore CANCEL componentwise,
and the NEUMANN side applies the partner's numbers UNCHANGED
(`f += InnerProduct(g, v) * ds("interface")`), because the natural boundary
term of the elasticity weak form is +(sigma . n_own) . v = +q_out_partner . v.

NGSolve's VectorH1 is BLOCKED BY COMPONENT: GetDofNrs(NodeId(VERTEX, i))
returns (dof of u_x, dof of u_y) and those two indices are nv apart, not
adjacent. Writing nodal values with an adjacency assumption silently scatters
u_y into the u_x block.
"""
import json
import os
from pathlib import Path

import ngsolve                # the MODULE, so ngsolve.ngsglobals.msg_level = 3 resolves:
                              # a run log that must carry this code's own output needs it,
                              # and `from ngsolve import ...` alone leaves `ngsolve` undefined
import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import (BND, VERTEX, BilinearForm, CF, GridFunction, InnerProduct,
                     LinearForm, Mesh, NodeId, TaskManager, VectorH1, ds, dx,
                     grad)


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

    HOW THIS ENTERS NGSolve (your solve below has to do it): NGSolve's symbolic
    `x`, `y` are CoefficientFunctions and carry NO NumPy ufuncs, so do NOT call
    this function on them (np.sin(x) raises; np.zeros_like(x) silently returns
    a 0-d object array and the source collapses to a constant), and do NOT wrap
    the function -- CoefficientFunction(B_SRC) raises ValueError ("Cannot make
    CoefficientFunction from <function ...>", measured). Sample it at the mesh vertices instead,
    np.array([vert.point for vert in mesh.vertices]), into one P1 GridFunction per component on
    your space (gf.vec.FV().NumPy()[vertex_dofs] = B_SRC(vx, vy)[i]); a GridFunction IS
    a CoefficientFunction and integrates as gf_x * v[0] * dx + gf_y * v[1] * dx, with v the test
    function -- the P1 interpolant of the source, quadrature error O(h^2), the order of the
    discretisation. A mesh vertex needs a name of its own: a loop over mesh.vertices that names
    its variable v leaves v holding the last vertex where the served lines below need the test
    function.
    """
    return np.zeros_like(x), np.zeros_like(y)
NX, NY    = 46, 26        # this subdomain's own mesh (netgen maxh derived below)

# ── THE PER-LEVEL RULE (served). A ./config.json {"level": k, "nx": .., "ny": ..}
#    next to this script overrides the mesh knobs and names the level. The dumps
#    at the foot of this file carry that level in their NAME, so a mesh study
#    leaves one file per level instead of the fine mesh overwriting the coarse.
LEVEL = 1
_cfg = {}
for _src, _txt in (("config.json", Path("config.json").read_text() if Path("config.json").is_file() else ""),
                   ("OPENPASO_CONFIG_JSON", os.environ.get("OPENPASO_CONFIG_JSON", ""))):
    try:
        _cfg.update(**json.loads(_txt or "{}"))
        LEVEL = int(_cfg.get("level", LEVEL))
        NX = int(_cfg.get("nx", NX))
        NY = int(_cfg.get("ny", NY))
    except (ValueError, TypeError) as _e:
        raise SystemExit(f"{_src} could not be read ({_e}); nothing was solved")

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
_cfg_all = _cfg   # config.json and OPENPASO_CONFIG_JSON, read and checked by the level rule above
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

MAXH = min((X1 - X0) / NX, (Y1 - Y0) / NY)    # netgen's scalar mesh size
ORDER = 1                                     # nodal == vertex dofs

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
    COMPONENT BY COMPONENT. One np.interp over a flattened (N, 2) array
    interleaves the components: right length, converges, every number wrong.

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
# Everything that is not the interface carries the prescribed displacement, so
# it all gets the same boundary name.
geo = SplineGeometry()
_names = ["outer"] * 4                     # the edges in AddRectangle's order
_names[(1 if ON_RIGHT else 3) if AX == 0 else (2 if ON_RIGHT else 0)] = "interface"
geo.AddRectangle((X0, Y0), (X1, Y1), bcs=tuple(_names))
mesh = Mesh(geo.GenerateMesh(maxh=MAXH))

fes = VectorH1(mesh, order=ORDER,
               dirichlet=("outer|interface" if SIDE == "dirichlet" else "outer"))
u, v = fes.TnT()

# vertex -> (x-dof, y-dof). VectorH1 blocks by COMPONENT, so these are nv apart.
vdof = np.array([fes.GetDofNrs(NodeId(VERTEX, i))[:2] for i in range(mesh.nv)], int)
vxy = np.array([mesh.vertices[i].point for i in range(mesh.nv)], float)

iface_v = np.where(np.abs(vxy[:, AX] - IFACE_X) < TOL)[0]
iface_v = iface_v[np.argsort(vxy[iface_v, AL])]          # sorted along the interface
y_if = vxy[iface_v, AL]
outer_v = np.where((np.abs(vxy[:, AX] - OUTER_X) < TOL) |
                   (np.abs(vxy[:, AL] - ALO) < TOL) |
                   (np.abs(vxy[:, AL] - AHI) < TOL))[0]
# THE TWO INTERFACE CORNERS BELONG TO THE OUTER BOUNDARY, ON BOTH SIDES: they
# sit on the faces the interface ends on, which carry a prescribed displacement
# in the un-split problem. Leaving them to the interface leaves them
# unconstrained on the Neumann side — the subproblem is still well posed, still
# converges, and lands a few percent off. They are still EXPORTED, just not
# interface-imposed.
corner = (np.abs(y_if - ALO) < TOL) | (np.abs(y_if - AHI) < TOL)


def eps_of(g):
    exx, eyy = g[0, 0], g[1, 1]
    return exx, eyy, 0.5 * (g[0, 1] + g[1, 0])


a = BilinearForm(fes)
gu, gv = grad(u), grad(v)
eu = eps_of(gu)
ev = eps_of(gv)
a += (2.0 * MU * (eu[0] * ev[0] + eu[1] * ev[1] + 2.0 * eu[2] * ev[2])
      + LAM * (eu[0] + eu[1]) * (ev[0] + ev[1])) * dx
f = LinearForm(fes)
# BODY FORCE. B_SRC is sampled at the vertices and carried by a GridFunction,
# which IS a CoefficientFunction, so the linear form's source varies in space
# instead of being the constant it used to be. ORDER = 1, so this is the P1
# interpolant of the source; its quadrature error is O(h^2), the same order as
# the P1 discretization error itself. Do NOT hand B_SRC ngsolve's symbolic x, y
# instead: a CoefficientFunction carries no NumPy ufuncs, so np.sin(x) raises
# and np.zeros_like(x) returns a 0-d OBJECT array — the source collapses to a
# constant and the subdomain solves the wrong problem with no error raised.
# Nodal writes go through vdof, because VectorH1 blocks BY COMPONENT.
gfb = GridFunction(fes)
gfb.vec[:] = 0.0
bx, by = B_SRC(vxy[:, 0], vxy[:, 1])
bvals = gfb.vec.FV().NumPy()
bvals[vdof[:, 0]] = np.broadcast_to(np.asarray(bx, float), (mesh.nv,))
bvals[vdof[:, 1]] = np.broadcast_to(np.asarray(by, float), (mesh.nv,))
f += InnerProduct(gfb, v) * dx
# THE VOLUME LOAD ALONE, in its own form. The Neumann branch adds the partner's
# interface term into `f`; the traction recovery at the bottom must subtract
# the volume load WITHOUT it, on both sides — subtracting the combined vector
# is what made the reaction look like zero on the Neumann side.
f_vol = LinearForm(fes)
f_vol += InnerProduct(gfb, v) * dx

gfu = GridFunction(fes)                    # also carries the Dirichlet data
gfu.vec[:] = 0.0
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

# ── WHAT THE SERVED LINES BELOW RELY ON, CHECKED (served) ─ keep this block.
#    y_if is the coordinate ALONG the interface: x on a horizontal one (the
#    name is the vertical case's). And the two served lines that integrate over
#    ds("interface") need a boundary of exactly that name on the interface line:
#    NGSolve integrates a ds() over a name the mesh does not carry over NOTHING,
#    with no error, so the Neumann load and the traction weights come out zero.
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
_ends = (np.abs(y_if - ALO) <= TOL) | (np.abs(y_if - AHI) <= TOL)   # the interface's two ends
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
# THE INTERFACE NODES ARE THE MESH'S OWN, AND vdof IS THEIR DOF MAP. Every served
# line below writes the partner's data, reads the traction and exports the values
# THROUGH iface_v and vdof, so a list naming other vertices, or a map naming other
# dofs, puts the partner's data on other nodes and exports their values under the
# interface's coordinates -- and a check reading the same lists agrees with them.
# Measured on a coupled heat run: the interface dofs were the mesh's first
# vertices' (its corners and part of an outer edge), the interface kept 0.0 at
# most of its nodes, and the coupling converged with every exchange check passing.
_line = {_i for _i in range(mesh.nv) if abs(mesh.vertices[_i].point[AX] - IFACE_X) < TOL}
_ids = [int(_n) for _n in iface_v]
_bad = [(_k, _n) for _k, _n in enumerate(_ids)
        if _k >= y_if.size or _n not in _line
        or abs(mesh.vertices[_n].point[AL] - float(y_if[_k])) > TOL
        or [int(_c) for _c in vdof[_n][:2]] != [int(_c) for _c in fes.GetDofNrs(NodeId(VERTEX, _n))[:2]]]
_missed = len(_line - set(_ids))
if _bad or _missed or len(_ids) != y_if.size:
    _k, _n = _bad[0] if _bad else (None, None)
    raise SystemExit(
        f"INTERFACE VERTICES: iface_v[k] must be the vertex number of the interface node at y_if[k], "
        f"one per node, and vdof[i] that vertex's two dofs fes.GetDofNrs(NodeId(VERTEX, i))[:2]; "
        f"iface_v has {len(_ids)} entries for {y_if.size} nodes"
        + (f", and {len(_bad)} of them fail (the first: iface_v[{_k}] = {_n}, "
           + (f"the vertex at ({mesh.vertices[_n].point[0]:g}, {mesh.vertices[_n].point[1]:g}) "
              f"with vdof {[int(_c) for _c in vdof[_n][:2]]} and dofs "
              f"{[int(_c) for _c in fes.GetDofNrs(NodeId(VERTEX, _n))[:2]]}"
              if 0 <= _n < mesh.nv else "no vertex of this mesh")
           + ")" if _bad else "")
        + (f"; {_missed} of the {len(_line)} vertices on the interface line have no entry"
           if _missed else "")
        + ". The served lines below write the partner's data, read the traction and export the "
          "values through these two. A vertex's number is its own (vert.nr for a mesh vertex vert), "
          "not its place in a list.")

if SIDE == "dirichlet":
    u_if = sample(imp, "values", (UI_X, UI_Y), y_if)
    for k, vtx in enumerate(iface_v):
        if _ends[k]:
            continue
        gfu.vec[int(vdof[vtx, 0])] = float(u_if[k, 0])
        gfu.vec[int(vdof[vtx, 1])] = float(u_if[k, 1])
else:
    t_if = sample(imp, "normal_fluxes", (TI_X, TI_Y), y_if)
    gfun = GridFunction(fes)               # P1 trace of the partner's samples
    gfun.vec[:] = 0.0
    for k, vtx in enumerate(iface_v):
        gfun.vec[int(vdof[vtx, 0])] = float(t_if[k, 0])
        gfun.vec[int(vdof[vtx, 1])] = float(t_if[k, 1])
    # APPLY the partner's numbers UNCHANGED
    f += InnerProduct(gfun, v) * ds("interface")

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
# The outer boundary is written LAST so it wins at the two interface corners.
ox, oy = vxy[outer_v, 0], vxy[outer_v, 1]
oux = UDX[0] + UDX[1] * ox + UDX[2] * oy + UDX[3] * oy * oy
ouy = UDY[0] + UDY[1] * ox + UDY[2] * oy + UDY[3] * oy * oy
for k, vtx in enumerate(outer_v):
    gfu.vec[int(vdof[vtx, 0])] = float(oux[k])
    gfu.vec[int(vdof[vtx, 1])] = float(ouy[k])
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

with TaskManager():
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    a.Assemble()
    f.Assemble()
    f_vol.Assemble()
    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * gfu.vec
    gfu.vec.data += a.mat.Inverse(fes.FreeDofs(),
                                  inverse="sparsecholesky") * res
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    # ── DID THE PARTNER'S DISPLACEMENT ENTER THE SOLVE? (served) ─ keep this block.
    #    A Dirichlet side whose solve freed the interface dofs returns its own
    #    answer and the coupling "converges" to two fields that disagree there.
    #    The nodes are the mesh's own on the interface line, never the lists the
    #    data were written through: a list naming the wrong nodes agrees with itself.
    _tv = sorted((float(mesh.vertices[_i].point[AL]), _i) for _i in _line)
    _tv = [(_a, _i) for _a, _i in _tv if ALO + TOL < _a < AHI - TOL]
    if SIDE == "dirichlet" and _tv:
        _ut = sample(imp, "values", (UI_X, UI_Y), np.array([_a for _a, _ in _tv], float))
        _gap = max(abs(float(gfu.vec[int(fes.GetDofNrs(NodeId(VERTEX, _i))[_c])]) - float(_ut[_k, _c]))
                   for _k, (_a, _i) in enumerate(_tv) for _c in (0, 1))
        if _gap > 1e-9 * max(1.0, float(np.abs(u_if).max())):
            raise SystemExit("EXPORT SELF-CHECK: the partner's displacement is not in the solution at the "
                             "interface nodes: the interface must be a Dirichlet boundary of the space "
                             "(dirichlet='outer|interface') and the solve must use fes.FreeDofs(); a solve "
                             "that frees them returns this side's own answer and couples to nothing")

    # Interface traction export q_out = -(sigma . n_own).
    #
    # WHY NOT AN L2 PROJECTION OF THE STRESS. That is what this file used to do
    # on the Neumann side: project -(sigma(u_h) . n_own) over the whole
    # subdomain and sample it at the interface. The gradient of a P1 solution —
    # and therefore the stress — is only O(h) accurate ON the boundary; the
    # superconvergence points are interior, and the boundary trace is exactly
    # what the coupling reads.
    #
    # THE CONSISTENT (REACTION) TRACTION. From
    #     a(u,v) - (f,v) = int_dOmega (sigma(u).n).v ds = -int_Gamma q_out.v ds
    # (the second equality is this file's sign convention) it follows that for
    # every vector basis function phi_i on the interface
    #     int_Gamma q_out . phi_i ds = -r_i,   r = A u_h - b_vol
    # with r the UNCONSTRAINED residual: NGSolve's a.mat and the load vectors
    # are exactly that — the Dirichlet condition lives in fes.FreeDofs() at
    # solve time and never touches the assembled operator, so the constrained
    # rows still carry the reaction.
    #
    # ONE FORMULA, BOTH SIDES. An earlier version used that reaction on the
    # Dirichlet side and the projection on the Neumann side, reasoning that the
    # Neumann interface dofs are free so r comes out ~0 there. That holds only
    # when the residual is taken against a load that ALREADY CONTAINS the
    # interface term. Subtract the VOLUME load alone and those same rows carry
    # exactly the interface functional the partner applied:
    #     (A u - b_vol)_i = int_Gamma g . phi_i ds
    # On the Dirichlet side there is no interface term, so f_vol == f and the
    # two cases are one expression.
    #
    # WHAT IS MEASURED, AND WHAT IS ONLY ALGEBRA. Handing the NEUMANN side a
    # traction and asking for it back is an ASSEMBLY IDENTITY, not a
    # convergence test: on free interface rows r = A u - b_vol IS M_Gamma g, so
    # the export is -(M_Gamma g)/(M_Gamma 1) and its offset from -g is
    # -(h^2/6) g''(y) for ANY correct assembly of ANY equation. The "order
    # 2.00" that used to stand here was read off that fixture; it is a property
    # of the P1 boundary mass matrix, not of this code — a bare NumPy mass
    # matrix reproduces the same numbers with no PDE, no solver and no material
    # in it. That fixture is kept (tests/test_interface_flux_recovery.py) for
    # what it really tests, and for a VECTOR participant the blocked-dof
    # mapping below is exactly the kind of defect it catches.
    #
    # THE ORDER is measured on the DIRICHLET side against an ANALYTIC interface
    # flux the participant is never handed
    # (tests/test_interface_flux_converges_to_a_known_exact_flux.py). Scalar
    # conduction, FEniCSx, the same recovery, 8/16/32/64 uniform triangle
    # meshes, max error over interior interface nodes:
    #   2.889e-01  7.243e-02  1.814e-02  4.556e-03   ORDER 1.996 1.998 1.993
    # and only first order (1.10, 1.06, 1.04) at the two nodes where the
    # interface meets the outer boundary, handled apart below. There is no
    # VECTOR measurement against an analytic traction, and none is claimed.
    #
    # THE RETIRED L2-PROJECTED GRADIENT, in the norms it was measured in: order
    # ~1 in the interior AWAY FROM THE ENDS (0.93), 0.50 in rms, and
    # non-convergent in the max norm that includes the near-end nodes, where it
    # stalls at 2.6 against a true flux of size 2 to 5. It was written up as a
    # flat "order 0.00, it never converges", which was true of one norm only.
    # Not re-measured since the branch was deleted.
    #
    # THE WEIGHT IS ONE SCALAR PER NODE, NOT ONE PER DOF. w_i = int_Gamma phi_i
    # ds belongs to the NODE, while VectorH1 blocks the dofs BY COMPONENT; the
    # test vector (1,1) in the weight form puts that same number on both of a
    # node's dofs, so the loop below divides every component by its own node's
    # weight and reaches it through vdof, never by assuming adjacency.
    rvec = f.vec.CreateVector()
    rvec.data = a.mat * gfu.vec - f_vol.vec   # r = A u_h - b_vol, no bc here
    fw = LinearForm(fes)
    fw += InnerProduct(CF((1.0, 1.0)), v) * ds("interface")
    fw.Assemble()

    Q = np.zeros((len(iface_v), 2))
    ok = np.ones((len(iface_v), 2), bool)
    for k, vtx in enumerate(iface_v):
        for c in (0, 1):
            d = int(vdof[vtx, c])
            wi = float(fw.vec[d])
            if abs(wi) > 1e-14:
                Q[k, c] = -float(rvec[d]) / wi
            else:
                ok[k, c] = False

    # THE TWO INTERFACE CORNERS ARE ON THE OUTER DIRICHLET BOUNDARY (the faces
    # the interface ends on), so their rows carry the OUTER reaction too and
    # their residual is not this interface's traction. Take the nearest interior
    # interface node rather than exporting a corner value that is physically a
    # different quantity. This holds on BOTH sides: the corners are
    # outer-Dirichlet either way. They are found by position (_ends), so an
    # outer_v that leaves them out does not let them through.
    suspect = _ends | np.isin(iface_v, outer_v) | ~ok.all(axis=1)
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
# line into 'Invalid MIT-MAGIC-COOKIE-1 keyNDOF = 113'.
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
        for _i in range(mesh.nv):
            _p = mesh.vertices[_i].point
            _f.write(f"{float(_p[0]):.11e},{float(_p[1]):.11e},"
                     f"{float(gfu.vec[int(vdof[_i, 0])]):.11e},"
                     f"{float(gfu.vec[int(vdof[_i, 1])]):.11e}\n")
    with open(f"interface_level{LEVEL}.csv", "w") as _f:
        _f.write("x,y,ux,uy,qx,qy\n")
        for _yy, (_ux, _uy), (_qx, _qy) in zip(y_if, [(gfu.vec[int(vdof[_n, 0])], gfu.vec[int(vdof[_n, 1])]) for _n in iface_v], Q):
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
    print(f"[ngsolve_elastic per-level dump] level {LEVEL} dump failed: "
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
_chk_vals = np.asarray([[gfu.vec[int(vdof[i, 0])], gfu.vec[int(vdof[i, 1])]] for i in iface_v], float).ravel()
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
            f"this a and f. Two ways measured to get here: a mask whose bits are not the "
            f"free dofs (a loop over a BitArray yields True/False, not dof numbers), or a "
            f"correction applied to the load alone instead of to the residual that the "
            f"values already in gfu leave. The flux recovery above reads the same a.mat, "
            f"so nothing was exported.")

Path("exports.json").write_text(json.dumps({
    "field_name": "displacement",
    "n_points": int(len(iface_v)),
    "coordinates": [([float(IFACE_X), float(yy)] if AX == 0 else [float(yy), float(IFACE_X)])
                    for yy in y_if],
    "values": [[float(gfu.vec[int(vdof[i, 0])]), float(gfu.vec[int(vdof[i, 1])])]
               for i in iface_v],
    "normal_fluxes": [[float(q0), float(q1)] for q0, q1 in Q],
}, indent=2))
