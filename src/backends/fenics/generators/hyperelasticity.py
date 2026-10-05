"""Hyperelasticity (Neo-Hookean) generator for FEniCSx/dolfinx.

Variants: 3d
"""


KNOWLEDGE = {
    # ─────────────────────────────────────────────────────────────────
    # _SERVING_STATUS (added 2026-08-03)
    # This dict is SHADOWED and is NOT what an agent receives.
    # fenics/backend.py:get_knowledge() returns
    # src/tools/deep_knowledge.py::_FENICS_KNOWLEDGE['hyperelasticity'] for this
    # physics and never falls through to here. Editing the pitfalls
    # below changes nothing an agent can see. The claims here were NOT
    # re-verified in the 2026-08-03 execution pass for exactly that
    # reason — treat them as unverified history, and make corrections
    # in deep_knowledge.py instead.
    # ─────────────────────────────────────────────────────────────────
    "description": "Nonlinear hyperelasticity (Neo-Hookean) with large deformation",
    "weak_form": "δΠ(u;v) = 0, Π = ∫(μ/2)(I_C-3) - μ*ln(J) + (λ/2)(ln(J))² dx",
    "function_space": "Vector Lagrange order 1, geometry-nonlinear",
    "solver": "Newton iteration with LU direct solve (SNES newtonls)",
    "pitfalls": [
        "[Numerical] Large-deformation kinematics: F = Id + grad(u), "
        "C = F^T * F, J = det(F). Use ufl.Identity, ufl.grad, "
        "ufl.det. Signal: writing the dolfinx Form with F = grad(u) "
        "(missing the Identity) produces zero deformation in the "
        "reference configuration — the vector Lagrange Function output "
        "in the XDMFFile shows zero stress under any load. (Audit "
        "2026-06-02.)",
        "[Numerical] Neo-Hookean stored energy psi must be positive-"
        "definite (psi >= 0 with equality only at F = Id). A wrong "
        "sign on the volumetric (lambda/2)*(ln(J))^2 term or "
        "(mu/2)*(I_C - 3) makes the energy non-convex. Signal: the "
        "dolfinx NonlinearProblem NewtonSolver oscillates without "
        "converging, or the residual ratio stays > 0.5 across "
        "iterations; checking dolfinx.fem.assemble_scalar(psi*dx) "
        "shows negative energy at small loads. (Audit 2026-06-02.)",
        "[Numerical] Large load steps cause NewtonSolver divergence "
        "in dolfinx — use load stepping (ramp the dirichletbc or "
        "fem.Constant body force through N substeps). Signal: a "
        "single-step solve at full load returns from "
        "NonlinearProblem.solve() without raising, with "
        "problem.solver.getConvergedReason() < 0 (e.g. -5, "
        "DIVERGED_MAX_IT) — the deprecated "
        "dolfinx.nls.petsc.NewtonSolver instead raises RuntimeError "
        "'Newton solver did not converge because maximum number of "
        "iterations reached'; switching to a for-loop over t in "
        "[0,1] with the dirichletbc value scaled by t recovers monotonic "
        "convergence at each substep. (Audit 2026-06-02.)",
        "[Numerical] Volumetric locking for nu → 0.5 (near-"
        "incompressible). Use a mixed (u, p) MixedElement with "
        "Taylor_Hood spaces, or reduced integration. Signal: a "
        "pure-displacement dolfinx vector Lagrange Function at nu=0.4999 "
        "shows tip_deflection at ~1e-3 of analytic; switching to a "
        "basix.ufl.mixed_element([P2_vec, P1]) recovers within ~1%. "
        "(Audit 2026-06-02.)",
        "[API] In dolfinx 0.10 the nonlinear solve is "
        "dolfinx.fem.petsc.NonlinearProblem(F, u, "
        "petsc_options_prefix=..., bcs=..., petsc_options=...).solve(), "
        "a high-level interface to PETSc SNES. "
        "dolfinx.nls.petsc.NewtonSolver is deprecated and works only "
        "with the deprecated fem.petsc.NewtonSolverNonlinearProblem "
        "(the 0.9 NonlinearProblem); passing it the 0.10 "
        "NonlinearProblem raises AttributeError \"'NonlinearProblem' "
        "object has no attribute 'a'\". Old dolfin nonlinear_variational "
        "patterns are removed. Signal: the way a script actually "
        "reaches for it — `from dolfinx.fem import "
        "NonlinearVariationalProblem` — raises \"ImportError: cannot "
        "import name 'NonlinearVariationalProblem' from 'dolfinx.fem'\" "
        "followed by the path of dolfinx/fem/__init__.py. The "
        "previously quoted 'No module named "
        "dolfinx.fem.NonlinearVariationalProblem' is a "
        "ModuleNotFoundError and only appears if the name is fed to "
        "importlib.import_module as a dotted MODULE path, which is not "
        "how anyone writes it; match on ImportError, not on that text. "
        "The correct call is `from dolfinx.fem.petsc import "
        "NonlinearProblem`. (Verified by execution 2026-08-07, "
        "dolfinx 0.10.0.)",
    ],
}

VARIANTS = ["3d"]


def generate(variant: str, params: dict) -> str:
    """Dispatch to the appropriate hyperelasticity variant."""
    generators = {
        "3d": _hyperelasticity_3d,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(f"Unknown hyperelasticity variant: {variant!r}. Available: {list(generators)}")
    return gen(params)


def _hyperelasticity_3d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    return f'''\
"""Hyperelasticity (Neo-Hookean) — 3D — FEniCSx"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import ufl
import numpy as np
from petsc4py import PETSc

domain = mesh.create_box(MPI.COMM_WORLD,
    [[0, 0, 0], [1, 1, 1]], [8, 8, 8], mesh.CellType.tetrahedron)
V = fem.functionspace(domain, ("Lagrange", 1, (3,)))

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

# Fix bottom face
def bottom(x):
    return np.isclose(x[2], 0.0)
bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom)
bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
bc = fem.dirichletbc(np.zeros(3, dtype=default_scalar_type), bottom_dofs, V)

# Prescribed displacement on top
def top(x):
    return np.isclose(x[2], 1.0)
top_facets = mesh.locate_entities_boundary(domain, fdim, top)
top_dofs = fem.locate_dofs_topological(V, fdim, top_facets)
bc_top = fem.dirichletbc(
    np.array([0.0, 0.0, -0.3], dtype=default_scalar_type), top_dofs, V)

# Neo-Hookean material
E_val = {E}
nu_val = {nu}
mu = fem.Constant(domain, default_scalar_type(E_val / (2 * (1 + nu_val))))
lmbda = fem.Constant(domain, default_scalar_type(E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))))

u = fem.Function(V)
v = ufl.TestFunction(V)

d = len(u)
I = ufl.Identity(d)
F = I + ufl.grad(u)
C = F.T * F
J = ufl.det(F)
Ic = ufl.tr(C)

# Stored energy (compressible Neo-Hookean)
psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2

# First variation (residual)
Pi = psi * ufl.dx
F_form = ufl.derivative(Pi, u, v)

problem = NonlinearProblem(F_form, u, bcs=[bc, bc_top], petsc_options_prefix="hyper",
    petsc_options={{"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
                   "snes_rtol": 1e-6, "snes_max_it": 25, "snes_monitor": None}})
problem.solve()
n_iters = problem.solver.getIterationNumber()
converged = problem.solver.getConvergedReason() > 0
print(f"Newton: {{n_iters}} iterations, converged={{converged}}")
u.name = "displacement"

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u)

u_arr = u.x.array.reshape(-1, 3)
print(f"max |u| = {{np.linalg.norm(u_arr, axis=1).max():.6e}}")
print(f"DOFs: {{V.dofmap.index_map.size_global * 3}}")
'''
