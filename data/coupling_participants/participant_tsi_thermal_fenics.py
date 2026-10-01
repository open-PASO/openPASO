"""FEniCSx (dolfinx) THERMAL half of a TWO-WAY thermo-structural (TSI) coupling.

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

WHAT THIS SOLVES — one backward-Euler step of the energy equation of linear
coupled thermoelasticity, on the WHOLE body (a FIELD coupling, not a domain
decomposition: the two participants occupy the same body and exchange volume
fields, so there is no interface, no normal and no flux to balance):

    rho_c (T - T_old)/dt  -  div(k grad T)  +  T_ref*beta*(e - e_old)/dt  =  0

The last term is THE MECHANICAL -> THERMAL DIRECTION. `e = tr(eps(u))` is the
volumetric strain imported from the structural participant and
`beta = (3 lambda + 2 mu) * alpha` is the thermal stress modulus. Drop it and
the coupling is one-way — a different and much weaker capability. `COUPLING` is
that switch, so a COUPLING=0.0 run is the control that shows the reverse
direction does something.

  Exchanged quantity IN  : volumetric strain e = tr(eps(u)), dimensionless,
                           nodal values on the PARTNER's mesh (non-matching).
  Exchanged quantity OUT : temperature CHANGE theta = T - T_ref in K, nodal
                           values on THIS mesh.

EXPORT THE TEMPERATURE CHANGE, NOT THE ABSOLUTE TEMPERATURE. The driver's
convergence test is a RELATIVE norm, so an exchanged quantity carrying a large
constant offset makes that norm small for free — the same coupling exchanging T
in kelvin and in celsius reports residuals a factor of ~20 apart. The offset
also makes the temperature block dominate the global norm, so the strain block,
which is what is actually still moving, hides behind it and the run stops early.
theta is in any case the only thing the constitutive law sees.

THE SIGN IS THE SILENT-WRONG. Compressing a body heats it and expanding it cools
it, so the term enters with a PLUS on the left-hand side as written above.
Flipping it converges just as prettily onto a temperature field wrong by twice
the coupling effect, and no convergence, balance or finiteness check can see
that — only a monolithic or native TSI comparison can.

UNITS: SI throughout (m, s, K, Pa, W/(m K), J/(m^3 K)). `RHO_C` is the
VOLUMETRIC heat capacity rho*c, not the specific one.
"""
import dolfinx
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
dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

import ufl
from dolfinx import default_scalar_type, fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
PARTNER   = "mech"        # the structural participant's `name` in couple(...)
X0, X1    = 0.0, 2.0      # the body (BOTH participants use the same body)
Y0, Y1    = 0.0, 0.5
NX, NY    = 38, 14        # this participant's OWN mesh; need not match the partner
K_COND    = 52.0          # thermal conductivity k, W/(m K)
RHO_C     = 3.297e6       # volumetric heat capacity rho*c, J/(m^3 K)
DT        = 1.0e4         # the time step of the single implicit step, s
T_REF     = 293.0         # stress-free / thermoelastic reference temperature, K
T_OLD     = 303.0         # temperature at the START of the step (uniform), K
T_HOT     = 323.0         # Dirichlet temperature on x = X0, K
T_HOT_DY  = 10.0          # linear y-variation added to it over Y1-Y0, K
T_COLD    = 303.0         # Dirichlet temperature on x = X1, K
BETA      = 6.3e7         # thermal stress modulus (3*lam+2*mu)*alpha, Pa/K
COUPLING  = 1.0           # 1.0 = two-way; 0.0 SUPPRESSES mechanical -> thermal
EVOL_OLD  = 2.2285714285714287e-3   # volumetric strain at the START of the step
EVOL_INIT = 2.2285714285714287e-3   # iteration-1 fallback for the imported strain
# ─────────────────────────────────────────────────────────────────────────

GEOM_TOL = 1e-9 * max(X1 - X0, Y1 - Y0)



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
    """Map the partner's nodal samples onto THIS participant's nodes.

    The driver does no interpolation. For a VOLUME coupling the partner's points
    are a scattered 2-D cloud rather than a line, so `np.interp` is not enough:
    linear interpolation over the partner's own triangulation, with a
    nearest-neighbour fallback for the few points that land a rounding error
    outside its convex hull.
    """
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
domain = dmesh.create_rectangle(MPI.COMM_WORLD, [[X0, Y0], [X1, Y1]],
                                [NX, NY], dmesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end
pts = V.tabulate_dof_coordinates()[:, :2]

evol = fem.Function(V)
evol.x.array[:] = sample(imp, "values", EVOL_INIT, pts)

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
T, s = ufl.TrialFunction(V), ufl.TestFunction(V)
c = fem.Constant(domain, default_scalar_type(RHO_C / DT))
a = c * T * s * ufl.dx + fem.Constant(domain, default_scalar_type(K_COND)) * \
    ufl.dot(ufl.grad(T), ufl.grad(s)) * ufl.dx
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end
L = c * fem.Constant(domain, default_scalar_type(T_OLD)) * s * ufl.dx \
    - fem.Constant(domain, default_scalar_type(COUPLING * T_REF * BETA / DT)) * \
    (evol - fem.Constant(domain, default_scalar_type(EVOL_OLD))) * s * ufl.dx

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
g = fem.Function(V)
hot = np.where(np.abs(pts[:, 0] - X0) < GEOM_TOL)[0]
cold = np.where(np.abs(pts[:, 0] - X1) < GEOM_TOL)[0]
g.x.array[hot] = T_HOT + T_HOT_DY * (pts[hot, 1] - Y0) / (Y1 - Y0)
g.x.array[cold] = T_COLD
bcs = [fem.dirichletbc(g, np.unique(np.concatenate([hot, cold])))]

uh = LinearProblem(a, L, bcs=bcs, petsc_options_prefix="tsi_th",
                   petsc_options={"ksp_type": "preonly",
                                  "pc_type": "lu"}).solve()
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end
sol = uh.x.array.real

print(f"[fenics thermal] n={len(sol)} coupling={COUPLING} "
      f"e_in=[{evol.x.array.min():.6e},{evol.x.array.max():.6e}] "
      f"T=[{sol.min():.6f},{sol.max():.6f}]")

# exports.json LAST: the driver takes its existence as proof of success.
# ── EXPORT SELF-CHECK ─ keep this block. An export that is not finite is
#    worthless however well the iteration behaved, and a field that came out
#    identically zero still couples, still converges and still hands in tidy
#    levels. A zero temperature change is what a thermal side reports when its source never arrived.
_chk_vals = np.asarray([t - T_REF for t in sol], float).ravel()
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
    "field_name": "temperature_change",
    "n_points": int(len(sol)),
    "coordinates": [[float(a_), float(b_)] for a_, b_ in pts],
    "values": [float(t - T_REF) for t in sol],
}, indent=2))
