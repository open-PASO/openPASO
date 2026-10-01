"""FEniCSx (dolfinx) STRUCTURAL half of a TWO-WAY thermo-structural coupling.

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
derivative in the space where u actually lives.

DISPLACEMENT ORDER. u is quadratic (P2) and T linear (P1), so tr(eps(u)) is
piecewise linear: the same space the temperature lives in. With P1 displacement
the strain is piecewise CONSTANT, one order below the temperature, and the
coupled answer settles on a slightly different fixed point than the monolithic
one for that reason alone.

UNITS: SI throughout (m, K, Pa). PLANE STRAIN, so lambda and mu are the 3-D
Lame constants and eps_zz = 0.
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
PARTNER    = "thermal"    # the thermal participant's `name` in couple(...)
X0, X1     = 0.0, 2.0     # the body (BOTH participants use the same body)
Y0, Y1     = 0.0, 0.5
NX, NY     = 32, 8        # this participant's OWN mesh; need not match the partner
E_MOD      = 2.1e11       # Young's modulus, Pa
NU         = 0.3          # Poisson ratio
BETA       = 6.3e7        # thermal stress modulus (3*lam+2*mu)*alpha, Pa/K
THETA_INIT = 10.0         # iteration-1 fallback for the imported theta = T-T_ref, K
# ─────────────────────────────────────────────────────────────────────────

LAM = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
MU = E_MOD / (2.0 * (1.0 + NU))
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
domain = dmesh.create_rectangle(MPI.COMM_WORLD, [[X0, Y0], [X1, Y1]],
                                [NX, NY], dmesh.CellType.triangle)
Vu = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))
Vt = fem.functionspace(domain, ("Lagrange", 1))
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end
pts = Vt.tabulate_dof_coordinates()[:, :2]      # where fields are exchanged

theta = fem.Function(Vt)
theta.x.array[:] = sample(imp, "values", THETA_INIT, pts)

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
u, v = ufl.TrialFunction(Vu), ufl.TestFunction(Vu)
eu, ev = ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))
a = (2.0 * MU * ufl.inner(eu, ev) + LAM * ufl.tr(eu) * ufl.tr(ev)) * ufl.dx
L = fem.Constant(domain, default_scalar_type(BETA)) * theta * ufl.div(v) * ufl.dx

fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)


def _bc(sub: int, where):
    Vs = Vu.sub(sub)
    Q, _ = Vs.collapse()
    facets = dmesh.locate_entities_boundary(domain, fdim, where)
    dofs = fem.locate_dofs_topological((Vs, Q), fdim, facets)
    z = fem.Function(Q)
    z.x.array[:] = 0.0
    return fem.dirichletbc(z, dofs, Vs)


# u_x = 0 on x = X0; u_y = 0 on y = Y0 and y = Y1 (transverse rollers)
bcs = [_bc(0, lambda x: np.isclose(x[0], X0, atol=GEOM_TOL)),
       _bc(1, lambda x: np.isclose(x[1], Y0, atol=GEOM_TOL)),
       _bc(1, lambda x: np.isclose(x[1], Y1, atol=GEOM_TOL))]

uh = LinearProblem(a, L, bcs=bcs, petsc_options_prefix="tsi_me",
                   petsc_options={"ksp_type": "preonly",
                                  "pc_type": "lu"}).solve()
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

# volumetric strain, L2-projected onto P1 so it lands on the exchange nodes
p_, q_ = ufl.TrialFunction(Vt), ufl.TestFunction(Vt)
eh = LinearProblem(p_ * q_ * ufl.dx, ufl.div(uh) * q_ * ufl.dx,
                   petsc_options_prefix="tsi_ev",
                   petsc_options={"ksp_type": "preonly",
                                  "pc_type": "lu"}).solve()
evol = eh.x.array.real
ux = uh.x.array.real[0::domain.geometry.dim]

print(f"[fenics mech] n={len(evol)} "
      f"theta_in=[{theta.x.array.min():.6f},{theta.x.array.max():.6f}] "
      f"ux=[{ux.min():.6e},{ux.max():.6e}] "
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
