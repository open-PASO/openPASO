"""Coupled thermal-structural generator for FEniCSx/dolfinx.

Variants: 2d
"""


KNOWLEDGE = {
    # ─────────────────────────────────────────────────────────────────
    # _SERVING_STATUS (added 2026-08-03)
    # This dict is SHADOWED and is NOT what an agent receives.
    # fenics/backend.py:get_knowledge() returns
    # src/tools/deep_knowledge.py::_FENICS_KNOWLEDGE['thermal_structural'] for this
    # physics and never falls through to here. Editing the pitfalls
    # below changes nothing an agent can see. The claims here were NOT
    # re-verified in the 2026-08-03 execution pass for exactly that
    # reason — treat them as unverified history, and make corrections
    # in deep_knowledge.py instead.
    # ─────────────────────────────────────────────────────────────────
    "description": "Coupled thermal-structural: solve heat equation, apply temperature as thermal load to elasticity",
    "weak_form": "Step 1: k*(grad(T),grad(v))*dx = 0. Step 2: (C:eps_mech, eps(v))*dx = ((3\u03bb+2\u03bc)*\u03b1*\u0394T*I, eps(v))*dx",
    "function_space": "Scalar Lagrange for temperature, Vector Lagrange for displacement",
    "solver": "Sequential: heat (LU) \u2192 elasticity (CG+GAMG)",
    "pitfalls": [
        "[Numerical] Thermal strain eps_th = alpha * DeltaT * Id is "
        "isotropic \u2014 equal expansion in all directions. Coupling into "
        "the elastic Form is sigma = C : (eps(u) - eps_th). Signal: "
        "writing the RHS as alpha * DeltaT * v * dx (forgetting Id) "
        "couples thermal expansion only into a single component \u2014 the "
        "dolfinx vector Lagrange displacement Function "
        "(fem.functionspace(domain, ('Lagrange', 1, (gdim,)))) develops anisotropic "
        "strain in the XDMFFile output that does not match the "
        "expected free-expansion alpha*DeltaT in all directions. "
        "(Audit 2026-06-02.)",
        "[Input] Reference temperature T_ref matters: DeltaT = T - "
        "T_ref. Leaving the T_ref ufl.Constant at 0 with an SI "
        "material at room temperature gives an initial thermal pre-"
        "strain of order alpha*T_room (~3e-3 for steel at 300 K). "
        "Signal: the first-step displacement Function from the dolfinx "
        "NonlinearProblem is huge compared to the actual loading and "
        "the NewtonSolver iteration may oscillate. (Audit 2026-06-02.)",
        "[Physics] Plane strain (default in 2D dolfinx solid mechanics) "
        "uses the full 3D Lame lambda \u2014 do NOT swap in plane-stress "
        "lambda_star = 2*lambda*mu/(lambda+2*mu) for a thin-plate "
        "approximation unless explicitly modelling plane stress. "
        "Signal: under this template's thermal load the plane-strain "
        "displacement is about (1+nu) times the plane-stress one, not "
        "(1-nu): eps_zz = 0 adds nu*alpha*dT to the in-plane thermal "
        "strain. Measured at nu=0.3 with the bottom edge clamped and T "
        "linear in x: 1.31 to 1.32 times in max |u|, max |u_x| and "
        "max |u_y|. "
        "(Audit 2026-06-02; factor measured 2026-10-01.)",
        "[Input] A mechanical Dirichlet BC is required to remove the "
        "null space (translation + rotation rigid-body modes) \u2014 "
        "without it, GAMG and even direct solvers may fail. Signal: "
        "solving the structural step with this template's LinearProblem "
        "(ksp_type cg, pc_type gamg; dolfinx has no PETScKrylovSolver) "
        "without any dirichletbc on the vector displacement space "
        "raises nothing, and problem.solver.getConvergedReason() is -4 "
        "(DIVERGED_DTOL) after 11 iterations; only with "
        "ksp_error_if_not_converged does solve() raise PETSc error "
        "code 91, 'KSPSolve() has not converged, reason DIVERGED_DTOL'. "
        "Pinning one node fixes it (reason 2 after 15 iterations). "
        "(Audit 2026-06-02; re-measured 2026-10-01 on dolfinx 0.10.0, "
        "40x40 mesh.)",
    ],
    "materials": {
        "E": {"range": [1e3, 1e12], "unit": "Pa"},
        "nu": {"range": [0.0, 0.499], "unit": "dimensionless"},
        "alpha": {"range": [1e-7, 1e-4], "unit": "1/K (thermal expansion coefficient)"},
    },
}

VARIANTS = ["2d"]


def generate(variant: str, params: dict) -> str:
    """Dispatch to the appropriate thermal-structural variant."""
    generators = {
        "2d": _thermal_structural_2d,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(f"Unknown thermal_structural variant: {variant!r}. Available: {list(generators)}")
    return gen(params)


def _thermal_structural_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    Demonstrates agent composing physics — NOT a standard tutorial example.
    """
    E = params.get("E", 200e3)
    nu = params.get("nu", 0.3)
    alpha = params.get("alpha", 12e-6)  # thermal expansion coefficient
    T_hot = params.get("T_hot", 100.0)
    T_cold = params.get("T_cold", 0.0)
    T_ref = params.get("T_ref", 0.0)
    nx = params.get("nx", 40)
    ny = params.get("ny", 40)
    return f'''\
"""Coupled thermal-structural analysis — FEniCSx/dolfinx
Step 1: Solve heat conduction (prescribed temperatures on boundaries)
Step 2: Apply temperature as thermal load -> solve elasticity with thermal strain
"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np

# Mesh: unit square
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
gdim = domain.geometry.dim
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

print("="*60)
print("STEP 1: Heat conduction")
print("="*60)

# -- Step 1: Solve heat equation --
V_T = fem.functionspace(domain, ("Lagrange", 1))

def left(x):
    return np.isclose(x[0], 0.0)
def right(x):
    return np.isclose(x[0], 1.0)

left_facets = mesh.locate_entities_boundary(domain, fdim, left)
right_facets = mesh.locate_entities_boundary(domain, fdim, right)
bc_T_left = fem.dirichletbc(default_scalar_type({T_hot}),
    fem.locate_dofs_topological(V_T, fdim, left_facets), V_T)
bc_T_right = fem.dirichletbc(default_scalar_type({T_cold}),
    fem.locate_dofs_topological(V_T, fdim, right_facets), V_T)

T_trial = ufl.TrialFunction(V_T)
v_T = ufl.TestFunction(V_T)
a_T = ufl.dot(ufl.grad(T_trial), ufl.grad(v_T)) * ufl.dx
L_T = fem.Constant(domain, default_scalar_type(0.0)) * v_T * ufl.dx

prob_T = LinearProblem(a_T, L_T, bcs=[bc_T_left, bc_T_right],
    petsc_options_prefix="heat", petsc_options={{"ksp_type": "preonly", "pc_type": "lu"}})
T_sol = prob_T.solve()
T_sol.name = "temperature"

print(f"Temperature: min={{T_sol.x.array.min():.2f}}, max={{T_sol.x.array.max():.2f}}")

print("="*60)
print("STEP 2: Elasticity with thermal strain")
print("="*60)

# -- Step 2: Solve elasticity with thermal strain --
V_u = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))

# Material
E_val = {E}
nu_val = {nu}
alpha_val = {alpha}
T_ref_val = {T_ref}
mu = E_val / (2 * (1 + nu_val))
lmbda = E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u, dT):
    \"\"\"Stress with thermal strain: sigma = C : (epsilon - alpha*dT*I)\"\"\"
    d = len(u)
    eps_thermal = alpha_val * dT * ufl.Identity(d)
    eps_mech = epsilon(u) - eps_thermal
    return lmbda * ufl.tr(eps_mech) * ufl.Identity(d) + 2 * mu * eps_mech

# BCs: fix bottom edge
def bottom(x):
    return np.isclose(x[1], 0.0)

bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom)
bottom_dofs = fem.locate_dofs_topological(V_u, fdim, bottom_facets)
bc_u = fem.dirichletbc(np.zeros(gdim, dtype=default_scalar_type), bottom_dofs, V_u)

# Temperature difference from reference
dT = fem.Function(V_T)
dT.x.array[:] = T_sol.x.array - T_ref_val

u_trial = ufl.TrialFunction(V_u)
v_u = ufl.TestFunction(V_u)

a_u = ufl.inner(sigma(u_trial, fem.Constant(domain, default_scalar_type(0.0))),
                 epsilon(v_u)) * ufl.dx
# RHS: thermal stress contribution
L_u = ufl.inner(sigma(fem.Constant(domain, default_scalar_type((0.0, 0.0))), dT)
                  - sigma(fem.Constant(domain, default_scalar_type((0.0, 0.0))),
                          fem.Constant(domain, default_scalar_type(0.0))),
                 epsilon(v_u)) * ufl.dx

# Simpler formulation: direct thermal load
a_u2 = ufl.inner(lmbda * ufl.tr(epsilon(u_trial)) * ufl.Identity(gdim) + 2 * mu * epsilon(u_trial),
                  epsilon(v_u)) * ufl.dx
L_u2 = ufl.inner((3 * lmbda + 2 * mu) * alpha_val * dT * ufl.Identity(gdim),
                  epsilon(v_u)) * ufl.dx

prob_u = LinearProblem(a_u2, L_u2, bcs=[bc_u],
    petsc_options_prefix="elast", petsc_options={{"ksp_type": "cg", "pc_type": "gamg"}})
u_sol = prob_u.solve()
u_sol.name = "displacement"

u_arr = u_sol.x.array.reshape(-1, gdim)
print(f"Displacement: max |u| = {{np.linalg.norm(u_arr, axis=1).max():.6e}}")
print(f"max |u_x| = {{np.abs(u_arr[:, 0]).max():.6e}}")
print(f"max |u_y| = {{np.abs(u_arr[:, 1]).max():.6e}}")

# Output both fields
from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "temperature.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(T_sol)
with XDMFFile(domain.comm, "displacement.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_sol)

print(f"\\nTotal DOFs: {{V_T.dofmap.index_map.size_global + V_u.dofmap.index_map.size_global * gdim}}")
print("Coupled thermal-structural analysis complete.")
'''
