"""Heat conduction generators for FEniCSx/dolfinx.

Variants: 2d_steady, 2d_transient, rectangle
"""


KNOWLEDGE = {
    # ─────────────────────────────────────────────────────────────────
    # _SERVING_STATUS (added 2026-08-03)
    # This dict is SHADOWED and is NOT what an agent receives.
    # fenics/backend.py:get_knowledge() returns
    # src/tools/deep_knowledge.py::_FENICS_KNOWLEDGE['heat'] for this
    # physics and never falls through to here. Editing the pitfalls
    # below changes nothing an agent can see. The claims here were NOT
    # re-verified in the 2026-08-03 execution pass for exactly that
    # reason — treat them as unverified history, and make corrections
    # in deep_knowledge.py instead.
    # ─────────────────────────────────────────────────────────────────
    "description": "Heat conduction (steady or transient) with FEniCSx/dolfinx",
    "weak_form": "k * (grad(T), grad(v)) * dx = (Q, v) * dx",
    "function_space": "Lagrange order 1",
    "solver": {"ksp_type": "preonly", "pc_type": "lu"},
    "pitfalls": [
        # Promoted from short-tip form to full [Category] + Signal:
        # pitfalls (2026-06-02). The two-line tips never reached Tier-0
        # because they lacked category prefix + observable symptom.
        "[Numerical] Transient heat is stiff: explicit time-stepping "
        "(theta=0 / forward Euler) requires dt < 2/lambda_max ~ h^2 / "
        "(2*kappa), which gets infeasibly small under refinement. Use "
        "backward Euler (theta=1) with a mass matrix on the LHS — "
        "unconditionally stable. Signal: an explicit-Euler dolfinx "
        "Function update on a refined mesh reaches NaN within ~5-10 "
        "time steps when dt > 2/lambda_max; the same problem with the "
        "bilinear form a = (T*v/dt + kappa*ufl.dot(ufl.grad(T), "
        "ufl.grad(v)))*dx converges "
        "smoothly to steady state. (Audit 2026-06-02.)",
        "[Input] Insulated boundary is the NATURAL BC for "
        "-div(kappa*grad(T)) = f — do NOTHING (no dirichletbc, no "
        "Neumann line). Adding a dirichletbc with value=0 on an "
        "insulated edge over-constrains the problem. Signal: applying "
        "fem.dirichletbc(default_scalar_type(0.0), ...) on a "
        "topologically insulated facet set pulls T_boundary to 0 — the "
        "XDMFFile output shows a steep dip near that edge instead of "
        "the expected zero-flux profile, and the integrated flux is "
        "non-zero. (Audit 2026-06-02.)",
    ],
    "materials": {
        "conductivity": {"range": [0.01, 1000], "unit": "W/(m*K)"},
    },
}

VARIANTS = ["2d_steady", "2d_transient", "rectangle"]


def generate(variant: str, params: dict) -> str:
    """Dispatch to the appropriate heat variant."""
    generators = {
        "2d_steady": _heat_2d_steady,
        "2d_transient": _heat_2d_transient,
        "rectangle": _heat_rectangle,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(f"Unknown heat variant: {variant!r}. Available: {list(generators)}")
    return gen(params)


def _heat_2d_steady(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    k = params.get("conductivity", 1.0)
    nx = params.get("nx", 32)
    ny = params.get("ny", 32)
    T_left = params.get("T_left", 100.0)
    T_right = params.get("T_right", 0.0)
    return f'''\
"""Steady heat conduction — FEniCSx/dolfinx
Prescribed temperatures on left/right, insulated top/bottom.
"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np

domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

def left(x):
    return np.isclose(x[0], 0.0)

def right(x):
    return np.isclose(x[0], 1.0)

tdim = domain.topology.dim
fdim = tdim - 1

left_facets = mesh.locate_entities_boundary(domain, fdim, left)
right_facets = mesh.locate_entities_boundary(domain, fdim, right)

dofs_left = fem.locate_dofs_topological(V, fdim, left_facets)
dofs_right = fem.locate_dofs_topological(V, fdim, right_facets)

bc_left = fem.dirichletbc(default_scalar_type({T_left}), dofs_left, V)
bc_right = fem.dirichletbc(default_scalar_type({T_right}), dofs_right, V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
k = fem.Constant(domain, default_scalar_type({k}))
a = k * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = fem.Constant(domain, default_scalar_type(0.0)) * v * ufl.dx

problem = LinearProblem(a, L, bcs=[bc_left, bc_right],
                         petsc_options_prefix="solve", petsc_options={{"ksp_type": "preonly", "pc_type": "lu"}})
uh = problem.solve()
uh.name = "temperature"

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

T = uh.x.array
print(f"Heat conduction solved: min(T)={{T.min():.2f}}, max(T)={{T.max():.2f}}")
print(f"DOFs: {{V.dofmap.index_map.size_global}}")
'''


def _heat_2d_transient(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable script. All parameter defaults are placeholders. The user/agent must set values appropriate to the specific problem being solved."""
    nx = params.get("nx", 32)
    ny = params.get("ny", 32)
    conductivity = params.get("conductivity", 1.0)
    n_steps = params.get("n_steps", 50)
    dt = params.get("dt", 0.01)
    T_hot = params.get("T_hot", 100.0)
    return f'''\
"""Transient heat equation — backward Euler — FEniCSx/dolfinx
dT/dt - kappa * laplacian(T) = f
Prescribed temperature on left boundary, insulated elsewhere.
"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
import ufl
import numpy as np
from petsc4py import PETSc

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

# Boundary conditions: prescribed temperature on left edge
def left_boundary(x):
    return np.isclose(x[0], 0.0)

left_facets = mesh.locate_entities_boundary(domain, fdim, left_boundary)
dofs_left = fem.locate_dofs_topological(V, fdim, left_facets)
bc = fem.dirichletbc(default_scalar_type({T_hot}), dofs_left, V)

# Parameters
kappa = fem.Constant(domain, default_scalar_type({conductivity}))
dt = fem.Constant(domain, default_scalar_type({dt}))
f_source = fem.Constant(domain, default_scalar_type(0.0))

# Solution functions
T_n = fem.Function(V, name="T_prev")  # previous time step
T_n.x.array[:] = 0.0  # initial condition: T=0

T_h = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Backward Euler weak form:
# (T^(n+1) - T^n)/dt * v + kappa * grad(T^(n+1)) . grad(v) = f * v
a = (T_h * v / dt + kappa * ufl.dot(ufl.grad(T_h), ufl.grad(v))) * ufl.dx
L = (T_n / dt * v + f_source * v) * ufl.dx

# Assemble once (matrix does not change if kappa/dt constant)
a_form = fem.form(a)
L_form = fem.form(L)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc

A = assemble_matrix(a_form, bcs=[bc])
A.assemble()

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

T_new = fem.Function(V, name="temperature")

# Time loop
n_steps = {n_steps}
from dolfinx.io import XDMFFile

# Write time series
with XDMFFile(domain.comm, "temperature.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    for step in range(n_steps):
        # Update RHS
        b = assemble_vector(L_form)
        apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc])

        # Solve
        solver.solve(b, T_new.x.petsc_vec)
        T_new.x.scatter_forward()

        # Update previous solution
        T_n.x.array[:] = T_new.x.array[:]

        # Write every 10th step
        t = (step + 1) * {dt}
        if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            xdmf.write_function(T_new, t)
            print(f"Step {{step+1}}/{{n_steps}}, t={{t:.4f}}: T in [{{T_new.x.array.min():.4f}}, {{T_new.x.array.max():.4f}}]")

print(f"Transient heat: {{n_steps}} steps complete")
print(f"Final T: min={{T_new.x.array.min():.6e}}, max={{T_new.x.array.max():.6e}}")
print(f"DOFs: {{V.dofmap.index_map.size_global}}")
'''


def _heat_rectangle(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    lx = params.get("lx", 2.0)
    ly = params.get("ly", 1.0)
    T_left = params.get("T_left", 100.0)
    T_right = params.get("T_right", 0.0)
    nx = params.get("nx", 64)
    ny = params.get("ny", 32)
    return f'''\
"""Heat conduction on [{lx}x{ly}] rectangle — FEniCSx/dolfinx"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np

domain = mesh.create_rectangle(
    MPI.COMM_WORLD, [[0, 0], [{lx}, {ly}]], [{nx}, {ny}], mesh.CellType.triangle
)
V = fem.functionspace(domain, ("Lagrange", 1))

def left(x):
    return np.isclose(x[0], 0.0)

def right(x):
    return np.isclose(x[0], {lx})

tdim = domain.topology.dim
fdim = tdim - 1
left_facets = mesh.locate_entities_boundary(domain, fdim, left)
right_facets = mesh.locate_entities_boundary(domain, fdim, right)
dofs_left = fem.locate_dofs_topological(V, fdim, left_facets)
dofs_right = fem.locate_dofs_topological(V, fdim, right_facets)
bc_left = fem.dirichletbc(default_scalar_type({T_left}), dofs_left, V)
bc_right = fem.dirichletbc(default_scalar_type({T_right}), dofs_right, V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = fem.Constant(domain, default_scalar_type(0.0)) * v * ufl.dx

problem = LinearProblem(a, L, bcs=[bc_left, bc_right],
    petsc_options_prefix="solve", petsc_options={{"ksp_type": "preonly", "pc_type": "lu"}})
uh = problem.solve()
uh.name = "temperature"

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

T = uh.x.array
print(f"Heat rectangle solved: min(T)={{T.min():.2f}}, max(T)={{T.max():.2f}}")
print(f"DOFs: {{V.dofmap.index_map.size_global}}")
'''
