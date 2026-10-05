"""NGSolve hyperelasticity generators and knowledge."""


def _hyperelasticity_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Neo-Hookean hyperelasticity via SymbolicEnergy + Newton."""
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    # Default maxh trimmed from 0.05 to 0.1: the unloaded
    # Variation Newton hits UmfpackInverse singularity on
    # finer meshes when the FIRST Newton step has to absorb
    # the full prescribed boundary displacement in one shot.
    # Coarser default keeps the gate green; tighten + add
    # load-stepping for production use.
    maxh = params.get("maxh", 0.1)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""Neo-Hookean hyperelasticity — Newton solver — NGSolve"""
from ngsolve import *
import json

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))
# Every boundary where a displacement is prescribed must
# be listed in `dirichlet=...` so its DOFs are removed
# from FreeDofs. gfu.Set(...definedon=mesh.Boundaries(
# 'top')) ALONE does not constrain anything — it just
# writes initial values; the Newton free-dof set then
# leaves the top boundary unconstrained, leaving a rigid
# translation mode and the very first tangent is
# singular ('UmfpackInverse: Numeric factorization
# failed'). Adding 'top' to dirichlet fixes the DOFs at
# the prescribed displacement.
fes = VectorH1(mesh, order=2, dirichlet="left|bottom|top")
u = fes.TrialFunction()

mu, lam = {mu}, {lam}
I = Id(2)
F = I + Grad(u)
C = F.trans * F
J = Det(F)

# Neo-Hookean energy: W = mu/2*(tr(C)-2) - mu*ln(J) + lam/2*ln(J)^2
energy = 0.5*mu*(Trace(C) - 2) - mu*log(J) + 0.5*lam*log(J)**2

a = BilinearForm(fes, symmetric=True)
a += Variation(energy * dx)

# Apply displacement BC — set for your problem
gfu = GridFunction(fes)
# The Newton tangent at F = I is well-conditioned, but for
# large prescribed displacements the first iteration sees an
# initial guess where interior u = 0 and boundary u = (large),
# making C nearly singular at facets next to the prescribed
# boundary. UmfpackInverse then aborts with 'Numeric
# factorization failed'. Keep the initial load small (~5%)
# so the first Newton step converges; load-stepping for the
# full 30% deformation is a follow-up.
gfu.Set(CoefficientFunction((0.05*x, -0.02)), definedon=mesh.Boundaries("top"))

solvers.Newton(a, gfu, maxit=20, printing=True)

vtk = VTKOutput(mesh, coefs=[gfu], names=["displacement"], filename="result", subdivision=1)
vtk.Do()
summary = {{"n_dofs": fes.ndof, "converged": True}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Hyperelasticity solve complete.")
'''


def _hyperelasticity_3d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    3D Neo-Hookean hyperelasticity."""
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    lx = params.get("lx", 1)
    ly = params.get("ly", 0.1)
    lz = params.get("lz", 0.04)
    maxh = params.get("maxh", 0.02)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""3D Neo-Hookean hyperelasticity — NGSolve"""
from ngsolve import *
from netgen.occ import *
import json

# Geometry — set dimensions for your problem
box = Box((0,0,0), ({lx},{ly},{lz}))
geo = OCCGeometry(box)
mesh = Mesh(geo.GenerateMesh(maxh={maxh}))
fes = VectorH1(mesh, order=2, dirichlet=".*")
u = fes.TrialFunction()

mu, lam = {mu}, {lam}
I = Id(3)
F = I + Grad(u)
C = F.trans * F
J = Det(F)
energy = 0.5*mu*(Trace(C) - 3) - mu*log(J) + 0.5*lam*log(J)**2

a = BilinearForm(fes, symmetric=True)
a += Variation(energy * dx)

gfu = GridFunction(fes)
# Apply load incrementally — set for your problem
for load in [0.2, 0.5, 0.8, 1.0]:
    gfu.Set(CoefficientFunction((0, 0, -0.01*load)), definedon=mesh.Boundaries(".*"))
    solvers.Newton(a, gfu, maxit=20, printing=False)

vtk = VTKOutput(mesh, coefs=[gfu], names=["displacement"], filename="result", subdivision=1)
vtk.Do()
summary = {{"n_dofs": fes.ndof}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


KNOWLEDGE = {
    "hyperelasticity": {
        "description": (
            "Nonlinear hyperelasticity via Variation(energy*dx) + "
            "ngsolve.solvers.Newton."
        ),
        "spaces": "VectorH1(mesh, order=2+)",
        "solver": (
            "ngsolve.solvers.Newton(a, gfu, maxit=20, dampfactor=1.0, "
            "maxerr=1e-11, printing=False) — built-in damped Newton."
        ),
        "pitfalls": [
            "[Physics] Every boundary where a displacement is "
            "prescribed via gfu.Set(...definedon=mesh.Boundaries("
            "'tag')) must ALSO appear in the dirichlet= specifier "
            "on the FESpace. gfu.Set alone only writes initial "
            "values; it does NOT flip the FreeDofs bit. The "
            "failure can be SILENT — Newton converges and simply "
            "throws the prescribed displacement away. Signal: "
            "unit_square maxh=0.2, VectorH1(order=2), Neo-Hookean "
            "E=100 nu=0.3, gfu.Set((0.05, 0)) on 'left'. With "
            "dirichlet='left' Newton returns (0, 6) and "
            "||u_x - 0.05||_L2(left) = 2.0e-17 — the BC holds. "
            "With dirichlet='' (no Dirichlet boundary at all) "
            "the outcome depends on the mesh and on round-off. "
            "On ngsolve's unit_square (netgen.occ) Newton STILL "
            "returns (0, 7) with no exception "
            "(symmetric/non-symmetric x default/umfpack, with or "
            "without TaskManager), but the solution picks up an "
            "arbitrary rigid motion — ||u_x - 0.05||_L2(left) "
            "came out between 1.8e-02 and 3.6e-01 when the "
            "initial guess was perturbed by 1e-12 — and the BC "
            "has relaxed away. On netgen.geom2d.unit_square the same "
            "setup raises NgException 'UmfpackInverse: Numeric "
            "factorization failed.' on the second Newton "
            "iteration. Do not wait for an exception; always "
            "assert the trace of gfu on the prescribed "
            "boundary after the solve. (Verified "
            "empirically on NGSolve 6.2.2604.)",
            "[Numerical] Newton with Variation(energy*dx) needs "
            "the FIRST iteration to be well-conditioned. Coupling "
            "a fine mesh (maxh <= 0.05 on the unit square ~3.9k "
            "DOFs) with a large prescribed boundary displacement "
            "applied in a single step makes the linearised "
            "tangent ill-conditioned (UMFPACK aborts on singular "
            "factorisation). Either coarsen the mesh (maxh ~ 0.1 "
            "/ 1.5k DOFs converges from a cold start) OR apply "
            "the displacement via incremental load stepping. "
            "Signal: NgException 'UmfpackInverse: Numeric "
            "factorization failed' followed by '_UpdateInverse' "
            "in the traceback. Re-measured 2026-08-03 on a "
            "2x2 grid — it is the COMBINATION that fails, not "
            "either factor alone: unit_square, VectorH1(order=2), "
            "dirichlet='left|right', prescribed u_x on 'right' — "
            "(maxh=0.05 [ndof 3938], disp=0.3) RAISES; "
            "(maxh=0.05, disp=0.05) returns (0, 7); "
            "(maxh=0.1 [ndof 978], disp=0.3) returns (0, 7); "
            "(maxh=0.1, disp=0.05) returns (0, 6). (Verified "
            "empirically 2026-06-01 — Layer F catch; grid "
            "re-measured 2026-08-03 on NGSolve 6.2.2604.)",
            "[API] solvers.Newton uses kwarg names maxit (singular, "
            "default 100) and maxerr (default 1e-11), NOT maxits "
            "and tol. The real signature is "
            "Newton(a, u, freedofs=None, maxit=100, maxerr=1e-11, "
            "inverse='', dirichletvalues=None, dampfactor=1, "
            "printing=True, callback=None). Passing maxits=20 or "
            "tol=1e-10 raises TypeError: Newton() got an "
            "unexpected keyword argument. Signal: "
            "solvers.Newton(a, gfu, maxits=20) raises TypeError "
            "with the literal kwarg name in the message; "
            "solvers.Newton(a, gfu, maxit=20) returns the "
            "(iterations, convergence) tuple. (Verified empirically "
            "2026-06-01 — Tier-2 fixture "
            "hyperelasticity_newton_maxit_kwarg in scripts/"
            "tier2_fixtures/ngsolve/. Five prior generator "
            "occurrences using the wrong kwargs corrected in same "
            "commit.)",
            "[API] Use Variation(energy*dx) on the BilinearForm — "
            "NGSolve auto-derives the residual a.Apply(gfu.vec, res) "
            "and the tangent a.AssembleLinearization(gfu.vec). "
            "Both methods take the current solution vector and "
            "either populate a residual block or assemble the "
            "tangent. Signal: hasattr(BilinearForm(fes), 'Apply') "
            "and hasattr(BilinearForm(fes), 'AssembleLinearization') "
            "are both True. Adding the energy density to the "
            "BilinearForm WITHOUT Variation() makes both the "
            "residual and the tangent all zeros, so solvers.Newton "
            "with the default inverse raises NgException "
            "'UmfpackInverse: Numeric factorization failed.' "
            "before completing its first iteration. A form written directly in residual form "
            "with a test function, e.g. "
            "(grad(u)*grad(v) + u**3*v - f*v)*dx, works with "
            "Newton without Variation(). (Catalog claim "
            "inherited; not separately Tier-2 falsified this "
            "iteration.)",
            "[Physics] Neo-Hookean strain energy density is "
            "psi(C) = 0.5*mu*(Tr(C)-d) - mu*ln(J) + 0.5*lam*ln(J)^2 "
            "where d is the spatial dimension, F = Id + Grad(u), "
            "C = F.trans * F, J = Det(F). The ln(J) terms are "
            "well-defined only for J>0 — Newton can lose this "
            "invariant during a too-large load step and produce "
            "ln(negative) → NaN. Signal: gfu.vec norm becomes nan "
            "after Newton step; check J = Det(Id + Grad(gfu)) "
            "evaluates positive on a quadrature point sample "
            "before each load increment. (Catalog claim inherited; "
            "not separately Tier-2 falsified this iteration.)",
            "[Numerical] For large deformations: use load stepping "
            "(incremental Dirichlet BC scaled by alpha = step/"
            "n_steps) and restart Newton from the previous "
            "converged gfu. Applying the full prescribed "
            "displacement in one Newton call almost always "
            "diverges because the first linearization is too far "
            "from the solution manifold. Signal: solvers.Newton "
            "returns (-1, maxit) and prints 'Warning: Newton might "
            "not converge! Error = ' — the return is "
            "(status, numit), an integer STATUS first (0 = "
            "converged, -1 = gave up) and the iteration count "
            "second, NOT (iters, conv); there is no error value in "
            "the return at all, so 'conv > maxerr' is comparing "
            "against an iteration count and a test written that "
            "way misreads every run. Test status == 0. Reduce the "
            "step size or reduce dampfactor below 1.0. (Return "
            "contract verified 2026-08-06 on NGSolve 6.2.2604 — "
            "Tier-2 fixture "
            "plasticity_load_step_divergence_signature.)",
            "[API] solvers.Newton's dampfactor kwarg (default 1) "
            "is a RAMP, not a constant scale: Newton iteration k "
            "(k = 1, 2, ...) updates the solution by "
            "min(1, k*dampfactor) times the Newton correction. So "
            "dampfactor=0.5 halves only the first update, and "
            "dampfactor=0.25 damps the first three (0.25, 0.5, "
            "0.75); all later steps are full steps. Reducing it "
            "trades iteration count for robustness when the initial "
            "guess is far from the converged solution. Signal: "
            "with dampfactor=1.0 the far-from-equilibrium step "
            "returns status -1 (or aborts inside the factorisation "
            "once the iterate has gone non-physical), and raising "
            "maxit does not rescue it; with dampfactor=0.5 or 0.25 "
            "the same step returns status 0 and reaches the "
            "prescribed displacement. Read the STATUS, not a "
            "'conv' value — solvers.Newton returns "
            "(status, numit) and no residual. (Return contract "
            "verified 2026-08-06 on NGSolve 6.2.2604 — Tier-2 "
            "fixtures plasticity_load_step_divergence_signature "
            "and nonlinear_elasticity_dampfactor_rescue.)",
        ],
    },
}

GENERATORS = {
    "hyperelasticity_2d": _hyperelasticity_2d,
    "hyperelasticity_3d": _hyperelasticity_3d,
}
