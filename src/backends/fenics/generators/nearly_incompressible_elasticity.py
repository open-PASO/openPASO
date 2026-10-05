"""Nearly-incompressible elasticity generator for FEniCSx/dolfinx.

Variants: 2d

Two-field mixed (u, p) Taylor-Hood formulation. Standard
primal P1/P2 displacement-only elements LOCK volumetrically
as Poisson ratio → 0.5; the mixed u-p split decouples
volumetric and deviatoric stresses, restoring optimal
convergence.

Weak form:

    2*mu*inner(eps(u), eps(v))*dx - div(v)*p*dx
                                          = inner(f, v)*dx
    div(u)*q*dx + (1/lambda)*p*q*dx       = 0

The (1/lambda) penalty stabilizes the saddle-point: as
lambda → infinity (true incompressibility), the term
vanishes and the system becomes Stokes-like. For finite
near-incompressibility (lambda large but bounded) the
formulation is stable for any inf-sup-stable pair (P2/P1).
"""


KNOWLEDGE = {
    # ─────────────────────────────────────────────────────────────────
    # _SERVING_STATUS (added 2026-08-03)
    # This dict is SHADOWED and is NOT what an agent receives.
    # fenics/backend.py:get_knowledge() returns
    # src/tools/deep_knowledge.py::_FENICS_KNOWLEDGE['nearly_incompressible_elasticity'] for this
    # physics and never falls through to here. Editing the pitfalls
    # below changes nothing an agent can see. The claims here were NOT
    # re-verified in the 2026-08-03 execution pass for exactly that
    # reason — treat them as unverified history, and make corrections
    # in deep_knowledge.py instead.
    # ─────────────────────────────────────────────────────────────────
    "description": (
        "Nearly-incompressible elasticity (nu → 0.5). "
        "Mixed (u, p) Taylor-Hood P2/P1 — primal P1/P2 "
        "locks volumetrically."),
    "weak_form": (
        "2*mu*inner(eps(u), eps(v))*dx - div(v)*p*dx "
        "= inner(f, v)*dx; "
        "div(u)*q*dx + (1/lambda)*p*q*dx = 0"),
    "function_space": (
        "Taylor-Hood mixed: P2 vector velocity + P1 scalar "
        "pressure via basix.ufl.mixed_element. NEVER use P1 "
        "displacement alone for nu > 0.45 — locks."),
    "solver": {
        "linear": "Direct (MUMPS) for moderate; GMRES + "
                  "block-LU for large.",
        "block_preconditioner": (
            "PCFieldSplit: (u, p) blocks separately, "
            "AMG on velocity, mass-matrix on pressure."),
    },
    "pitfalls": [
        "[API] Pure P1/P2 displacement-only formulation "
        "(basix.ufl.element('Lagrange', cell, k, shape=(d,)) "
        "with k=1 or 2) LOCKS as nu → 0.5. Mixed u-p "
        "(basix.ufl.mixed_element) restores optimal rate. "
        "Signal: assembling the displacement-only stiffness "
        "K = assemble_matrix(fem.form(a_lock)) and comparing "
        "||u||_L2 from LinearProblem.solve() against an "
        "analytic reference shows error vs h^k slope matches "
        "k=0.5 (locking regime) for nu>=0.495 with primal "
        "P1/P2, and k>=1 with the mixed_element pair.",
        "[Syntax] Mixed-element construction in dolfinx "
        "0.10+ uses basix.ufl.mixed_element, NOT ufl: "
        "P2 = basix.ufl.element('Lagrange', cell, 2, shape=(d,)), "
        "P1 = basix.ufl.element('Lagrange', cell, 1), "
        "TH = basix.ufl.mixed_element([P2, P1]). "
        "Signal: ufl.VectorElement / ufl.MixedElement raise "
        "AttributeError in dolfinx 0.10+ (removed in UFL 2024+).",
        "[Numerical] Inf-sup stability: NOT all element pairs "
        "are stable for u-p elasticity. Stable: Taylor-Hood "
        "(P2 vector + P1 scalar), MINI (P1+bubble + P1). "
        "Unstable: equal-order P1/P1 (saddle-point breaks). "
        "Signal: equal-order pair produces checkerboard "
        "pressure mode visible in the VTX/XDMF output, with norm "
        "growing as h decreases.",
        "[API] Solver: pure direct (petsc_options "
        "{'ksp_type': 'preonly', 'pc_type': 'lu', "
        "'pc_factor_mat_solver_type': 'mumps'}) for small. "
        "For large, use PCFieldSplit with AMG on velocity. "
        "Signal: passing 'pc_type': 'gamg' or 'hypre' in "
        "LinearProblem's petsc_options on the full mixed (u, p) "
        "system does not raise; solve() returns and only "
        "problem.solver.getConvergedReason(), the iteration "
        "count and the solution show it. With the default "
        "GMRES, GAMG at nu=0.4999 stagnates and ends "
        "DIVERGED_MAX_IT (-3) after the default 10000 "
        "iterations, and at nu=0.49 it needs thousands of "
        "iterations to reach CONVERGED_RTOL (2); hypre at "
        "nu=0.4999 gives "
        "DIVERGED_NANORINF (-9) at 0 iterations with a zero "
        "solution, and at nu=0.49 it reports CONVERGED_RTOL (2) "
        "after 1 iteration on the preconditioned residual while "
        "the true residual has not dropped and u is ~0 — PETSc's "
        "GAMG/BoomerAMG "
        "cannot coarsen the off-diagonal pressure-divergence "
        "block. Use FieldSplit (pc_type='fieldsplit') with "
        "pc_fieldsplit_type='schur' instead.",
    ],
}

VARIANTS = ["2d"]


def generate(variant: str, params: dict) -> str:
    """Dispatch."""
    generators = {
        "2d": _nearly_incomp_2d,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(
            f"Unknown variant: {variant!r}. "
            f"Available: {list(generators)}")
    return gen(params)


def _nearly_incomp_2d(params: dict) -> str:
    """FORMAT TEMPLATE — Taylor-Hood u-p elasticity on a
    unit square. Bottom fixed, top loaded with traction.
    Mu and lambda from (E, nu) via Lame conversion."""
    nx = params.get("nx", 16)
    E = params.get("E", 1.0)
    nu = params.get("nu", 0.49)
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return f'''\
"""Nearly-incompressible elasticity — Taylor-Hood P2/P1 — FEniCSx"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import basix.ufl
import ufl
import numpy as np

domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {nx},
                                 mesh.CellType.triangle)
gdim = domain.geometry.dim
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

mu_val  = {mu}
lam_val = {lam}

# Taylor-Hood: P2 vector velocity + P1 scalar pressure
P2 = basix.ufl.element("Lagrange", domain.basix_cell(), 2,
                       shape=(gdim,))
P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
TH = basix.ufl.mixed_element([P2, P1])
W = fem.functionspace(domain, TH)

(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)

def eps(w):
    return ufl.sym(ufl.grad(w))

a = (2.0 * mu_val * ufl.inner(eps(u), eps(v)) * ufl.dx
     - ufl.div(v) * p * ufl.dx
     - ufl.div(u) * q * ufl.dx
     - (1.0 / lam_val) * p * q * ufl.dx)
L = ufl.inner(fem.Constant(domain,
                           default_scalar_type((0.0, -0.1))),
              v) * ufl.dx

# Fix bottom (u = 0) — Dirichlet on the velocity sub-space
W0 = W.sub(0)
V0, _ = W0.collapse()
def bottom(x):
    return np.isclose(x[1], 0.0)
bottom_facets = mesh.locate_entities_boundary(
    domain, fdim, bottom)
fixed_u = fem.Function(V0)
fixed_u.x.array[:] = 0.0
bc = fem.dirichletbc(
    fixed_u,
    fem.locate_dofs_topological(
        (W0, V0), fdim, bottom_facets),
    W0)

problem = LinearProblem(
    a, L, bcs=[bc],
    petsc_options_prefix="nearly_incomp_",
    petsc_options={{"ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps"}})
sol = problem.solve()
u_sol = sol.sub(0).collapse()
p_sol = sol.sub(1).collapse()
# --- Write the result where a reader can open it ------------------------------
# This template printed a norm and wrote nothing, so a model that ran it had
# no field to look at, plot or verify. XDMFFile writes P1 functions; the
# openPASO backend converts result.xdmf/.h5 into a standard result.vtu after
# the run (the same route the Poisson template uses). Higher-order or
# non-Lagrange functions are interpolated into P1 for output first.
import json
from dolfinx.io import XDMFFile
_Vvis = fem.functionspace(domain, basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,)))
_uo = fem.Function(_Vvis, name="displacement"); _uo.interpolate(u_sol)
with XDMFFile(domain.comm, "result.xdmf", "w") as _x:
    _x.write_mesh(domain); _x.write_function(_uo)
with open("results_summary.json", "w") as _f:
    json.dump({{"l2_norm_u": float(np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(u_sol, u_sol) * ufl.dx))))),
               "n_dofs": W.dofmap.index_map.size_global}}, _f, indent=2)
print(f"||u||_L2 = {{np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(u_sol, u_sol) * ufl.dx))))}}")
print(f"||p||_L2 = {{np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form(p_sol * p_sol * ufl.dx))))}}")
'''
