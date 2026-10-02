"""NGSolve STRUCTURAL half of a TWO-WAY thermo-structural (TSI) coupling.

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

WHAT THIS SOLVES — quasi-static linear elasticity on the WHOLE body, in plane
strain, with the thermal stress the imported temperature field produces:

    div(sigma) = 0,   sigma = 2 mu eps(u) + lambda tr(eps(u)) I - beta (T-T_ref) I

`beta = (3 lambda + 2 mu) * alpha` is the thermal stress modulus. This is THE
THERMAL -> MECHANICAL DIRECTION.

  Exchanged quantity IN  : temperature CHANGE theta = T - T_ref in K, nodal
                           values on the PARTNER's mesh (non-matching).
  Exchanged quantity OUT : volumetric strain e = tr(eps(u)), dimensionless,
                           nodal values on THIS mesh.

WHY THE VOLUMETRIC STRAIN AND NOT THE DISPLACEMENT. The energy equation couples
to d/dt tr(eps), not to u. Exporting u would make the thermal side
differentiate a field it had interpolated off a foreign mesh — a derivative of
an interpolant, one order of accuracy down. Exporting the strain takes that
derivative in the space where u lives.

DISPLACEMENT ORDER. u is quadratic (VectorH1 order 2) and T linear, so
tr(eps(u)) is piecewise linear: the space the temperature lives in. With linear
displacement the strain is piecewise CONSTANT, one order below the temperature,
and the coupled answer settles on a slightly different fixed point than the
monolithic one for that reason alone.

BOUNDARY CONDITIONS PER COMPONENT. `dirichletx=` / `dirichlety=` constrain ONE
component of VectorH1 on a named boundary; a plain `dirichlet=` would clamp
BOTH and turn transverse rollers into a fully built-in edge, which changes the
problem without changing anything visible in the iteration.

UNITS: SI throughout (m, K, Pa). PLANE STRAIN, so lambda and mu are the 3-D
Lame constants and eps_zz = 0.
"""
import ngsolve
import json
from pathlib import Path

import numpy as np
# MAKE THIS CODE SPEAK, BEFORE THE SOLVE RUNS. It is silent by default, and a
# per-level run log carrying no line the solver itself emitted cannot
# establish which code ran on this side, however right its numbers are.
# It sits HERE, beside the level rule, and not up with the imports:
# measured over agent-written participants, a line placed in the import
# block survived in about half of them because that block gets rewritten,
# while everything beside the level rule survived in all of them.
ngsolve.ngsglobals.msg_level = 3

from netgen.geom2d import SplineGeometry
from ngsolve import (VERTEX, BilinearForm, CoefficientFunction, GridFunction,
                     H1, InnerProduct, LinearForm, Mesh, NodeId, Sym,
                     TaskManager, VectorH1, div, dx, grad)
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
PARTNER    = "thermal"    # the thermal participant's `name` in couple(...)
X0, X1     = 0.0, 2.0     # the body (BOTH participants use the same body)
Y0, Y1     = 0.0, 0.5
NX, NY     = 34, 8        # sets netgen's maxh; the mesh is UNSTRUCTURED
E_MOD      = 2.1e11       # Young's modulus, Pa
NU         = 0.29         # Poisson ratio
BETA       = 6.3e7        # thermal stress modulus (3*lam+2*mu)*alpha, Pa/K
THETA_INIT = 10.0         # iteration-1 fallback for the imported theta = T-T_ref, K
# ─────────────────────────────────────────────────────────────────────────

LAM = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
MU = E_MOD / (2.0 * (1.0 + NU))
MAXH = min((X1 - X0) / NX, (Y1 - Y0) / NY)



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
        return _partner_block(json.loads(p.read_text()))
    except json.JSONDecodeError:
        return None


def sample(imp, key, fallback, pts):
    """Map the partner's nodal samples onto THIS participant's nodes (see the
    thermal participant for why this is a 2-D scattered interpolation)."""
    if not imp or not imp.get("coordinates"):
        return np.full(pts.shape[0], float(fallback))
    src = np.asarray(imp["coordinates"], float)[:, :2]
    val = np.asarray(imp.get(key, []), float).ravel()
    if val.size != src.shape[0]:
        return np.full(pts.shape[0], float(fallback))
    out = LinearNDInterpolator(src, val)(pts)
    bad = ~np.isfinite(out)
    if np.any(bad):
        out[bad] = NearestNDInterpolator(src, val)(pts[bad])
    return out


imp = read_imports()

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
geo = SplineGeometry()
geo.AddRectangle((X0, Y0), (X1, Y1), bcs=("bottom", "right", "top", "left"))
mesh = Mesh(geo.GenerateMesh(maxh=MAXH))

fesq = H1(mesh, order=1)                   # where fields are exchanged
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end
vdof = np.array([fesq.GetDofNrs(NodeId(VERTEX, i))[0] for i in range(mesh.nv)], int)
pts = np.array([mesh.vertices[i].point for i in range(mesh.nv)], float)[:, :2]

theta_nodal = sample(imp, "values", THETA_INIT, pts)
gth = GridFunction(fesq)
gth.vec[:] = 0.0
for i, d in enumerate(vdof):
    gth.vec[int(d)] = float(theta_nodal[i])

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
# u_x = 0 on x = X0 ("left"); u_y = 0 on y = Y0/Y1 ("bottom"/"top")
fes = VectorH1(mesh, order=2, dirichletx="left", dirichlety="bottom|top")
u, v = fes.TnT()
a = BilinearForm(fes)
a += (2.0 * MU * InnerProduct(Sym(grad(u)), Sym(grad(v)))
      + LAM * div(u) * div(v)) * dx
f = LinearForm(fes)
f += BETA * gth * div(v) * dx

gfu = GridFunction(fes)
gfu.vec[:] = 0.0

m = BilinearForm(fesq)
p_, w_ = fesq.TnT()
m += p_ * w_ * dx
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

# THE SOLVE AND THE PROJECTION RUN AT MODULE LEVEL, NOT INSIDE A `with` BLOCK:
# the solve hole sat inside `with TaskManager():`, and a fill written at column 0
# ends that block, so the served lines after it no longer parse (IndentationError).
# Wrap your own solve in `with TaskManager():` if you want its threads.
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
with TaskManager():
    a.Assemble()
    f.Assemble()
    r = f.vec.CreateVector()
    r.data = f.vec - a.mat * gfu.vec
    gfu.vec.data += a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * r
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

# volumetric strain, L2-projected onto the order-1 space
m.Assemble()
fq = LinearForm(fesq)
fq += div(gfu) * w_ * dx
fq.Assemble()
gev = GridFunction(fesq)
gev.vec.data = m.mat.Inverse(fesq.FreeDofs(),
                             inverse="sparsecholesky") * fq.vec

evol = np.array([gev.vec[int(d)] for d in vdof], float)
print(f"[ngsolve mech] n={len(evol)} "
      f"theta_in=[{theta_nodal.min():.6f},{theta_nodal.max():.6f}] "
      f"e=[{evol.min():.6e},{evol.max():.6e}]")

# ── EXPORT SELF-CHECK ─ keep this block. An export that is not finite is
#    worthless however well the iteration behaved, and a field that came out
#    identically zero still couples, still converges and still hands in tidy
#    levels. A zero volumetric strain is what a mechanical side reports when the thermal load never arrived.
_chk_vals = np.asarray(evol, float).ravel()
if not np.isfinite(_chk_vals).all():
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface values; the "
                     "solve did not produce a usable field, so nothing was "
                     "exported")
if _chk_vals.size and np.abs(_chk_vals).max() == 0.0:
    raise SystemExit("EXPORT SELF-CHECK: every exported interface value is "
                     "exactly zero. That is the no-response answer, not a "
                     "solve: the partner's data never reached the assembled "
                     "system. Fix the application; do not couple on")

Path("exports.json").write_text(json.dumps({
    "field_name": "volumetric_strain",
    "n_points": int(len(evol)),
    "coordinates": [[float(a_), float(b_)] for a_, b_ in pts],
    "values": [float(e) for e in evol],
}, indent=2))
