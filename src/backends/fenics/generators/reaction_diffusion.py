"""Reaction-diffusion system generator for FEniCSx/dolfinx.

Variants: 2d
"""


KNOWLEDGE = {
    # ─────────────────────────────────────────────────────────────────
    # _SERVING_STATUS (added 2026-08-03)
    # This dict is SHADOWED and is NOT what an agent receives.
    # fenics/backend.py:get_knowledge() returns
    # src/tools/deep_knowledge.py::_FENICS_KNOWLEDGE['reaction_diffusion'] for this
    # physics and never falls through to here. Editing the pitfalls
    # below changes nothing an agent can see. The claims here were NOT
    # re-verified in the 2026-08-03 execution pass for exactly that
    # reason — treat them as unverified history, and make corrections
    # in deep_knowledge.py instead.
    # ─────────────────────────────────────────────────────────────────
    "description": "Two-species reaction-diffusion system (coupled, transient, nonlinear)",
    "weak_form": "((u-u_old)/dt, phi_u)*dx + D1*(grad(u),grad(phi_u))*dx = (R_u, phi_u)*dx (+ same for v)",
    "function_space": "Mixed: P1 + P1 (one per species) via mixed_element",
    "solver": "Newton (PETSc SNES) with LU direct for each time step",
    "pitfalls": [
        (
            "[Numerical] Nonlinear reaction terms require Newton "
            "iteration (NonlinearProblem). Signal: a single-step "
            "linear-only solve on R(u,v) = u*v gives wrong "
            "steady-state \u2014 quadratic coupling means the linear "
            "problem is not the linearisation of the nonlinear "
            "one. Use dolfinx.fem.petsc.NonlinearProblem(F, w, "
            "bcs=..., petsc_options_prefix=..., petsc_options={...}) "
            "and problem.solve() (PETSc SNES Newton; the Jacobian is "
            "derived from F automatically). "
            "dolfinx.nls.petsc.NewtonSolver is deprecated in 0.10. "
            "(Audit 2026-06-02.)"
        ),
        (
            "[Numerical] Time stepping: backward Euler is robust; "
            "Crank-Nicolson can oscillate at sharp activation "
            "fronts. Signal: visualizing u at a moving "
            "concentration front shows 10-30% over/undershoot "
            "in CN that does not damp with mesh refinement; BE "
            "removes the oscillation at the cost of 1st-order "
            "phase error. (Audit 2026-06-02.)"
        ),
        (
            "[API] Initial conditions on a mixed Function: the "
            "component-wise trap is real but it is NOT the one this "
            "entry used to describe. TWO CORRECTIONS, both measured. "
            "First, `w.sub(0).interpolate(initial_u)` does NOT raise "
            "and the IC does NOT stay at zero — it writes component 0 "
            "correctly and leaves component 1 alone (measured: sub0 "
            "5.0, sub1 0.0). The quoted `AttributeError: Function."
            "sub() returns a sub-function not a sub-space` is emitted "
            "by nothing in dolfinx and does not reproduce. Second, "
            "what DOES bite, silently, is the array: "
            "`w.sub(0).x.array` is not the component, it is the WHOLE "
            "mixed vector — same length as w.x.array (50 and 50 on a "
            "4x4 P1-P1 mesh) — so `w.sub(0).x.array[:] = 7.0` "
            "overwrites BOTH components (measured: sub0 7.0 AND sub1 "
            "7.0) with no error and no warning. Signal: set an IC on "
            "one component through .x.array and read the OTHER "
            "component back; if it moved, that is the defect. Use the "
            "collapsed route — `V0, dofs0 = W.sub(0).collapse()`, "
            "interpolate on V0, `w.x.array[dofs0] = f0.x.array`, then "
            "`w.x.scatter_forward()` — which leaves component 1 at 0.0. "
            "Note also that interpolate returns None, so "
            "`u0 = w.sub(0).interpolate(...)` binds None and the next "
            "use of u0 gives `AttributeError: 'NoneType' object has no "
            "attribute 'x'`. (Verified by execution 2026-08-07, "
            "dolfinx 0.10.0.)"
        ),
        (
            "[API] No-flux (Neumann zero) is the natural BC \u2014 no "
            "Dirichlet needed for insulated walls. Signal: "
            "applying DirichletBC(value=0) on an 'insulated' "
            "boundary over-constrains (u=0, not du/dn=0) and "
            "pulls the species concentration toward zero at the "
            "boundary. Compare a no-BC vs Dirichlet=0 run \u2014 the "
            "no-BC version shows bulged-out concentration "
            "profiles at the boundary. (Audit 2026-06-02.)"
        ),
        (
            "[Numerical] For stiff reactions: may need smaller "
            "dt or implicit-explicit (IMEX) splitting. Signal: "
            "an explicit Euler update on the dolfinx Function "
            "with dt > 2/lambda_max (lambda_max ~ rate "
            "constant) gives NaN in the assemble_vector RHS "
            "within a few steps; for Da > 1000, even backward "
            "Euler via NonlinearProblem converges slowly "
            "without splitting the stiff reaction onto its "
            "own implicit sub-step (Strang or Lie). (Audit "
            "2026-06-02.)"
        ),
        (
            "[Physics] Common reaction models built on the "
            "u-v reaction-diffusion two-species template are "
            "Gray-Scott, Schnakenberg, Brusselator, and Lotka-"
            "Volterra. Signal: a dolfinx LinearProblem solving the "
            "wrong-coupling-sign form (e.g. -u*v in Gray-Scott "
            "instead of +u*v^2) shows the Function field collapsing "
            "to a homogeneous_steady_state instead of forming "
            "Turing patterns visible in the XDMFFile output. The "
            "right coupling for each model is published — check "
            "against literature before running long Schnakenberg "
            "sweeps. (Audit 2026-06-02.)"
        ),
        (
            "[Numerical] Conservation: check mass integrals over "
            "time to verify correctness. Signal: for a closed "
            "system (no source/sink, zero-flux natural boundary "
            "with no boundary term) the "
            "dolfinx fem.assemble_scalar integral of (u + v) "
            "*dx over the domain should be conserved; if it "
            "drifts > 0.1% per unit time, the time integrator "
            "has a numerical leak \u2014 refine dt or switch to a "
            "conservative scheme. (Audit 2026-06-02.)"
        ),
    ],
    "materials": {
        "D1": {"range": [1e-6, 10.0], "unit": "m^2/s (diffusion coefficient, species 1)"},
        "D2": {"range": [1e-6, 10.0], "unit": "m^2/s (diffusion coefficient, species 2)"},
    },
}

VARIANTS = ["2d"]


def generate(variant: str, params: dict) -> str:
    """Dispatch to the appropriate reaction-diffusion variant."""
    generators = {
        "2d": _reaction_diffusion_2d,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(f"Unknown reaction_diffusion variant: {variant!r}. Available: {list(generators)}")
    return gen(params)


def _reaction_diffusion_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable script. All parameter defaults are placeholders. The user/agent must set values appropriate to the specific problem being solved."""
    nx = params.get("nx", 64)
    ny = params.get("ny", 64)
    D1 = params.get("D1", 0.01)
    D2 = params.get("D2", 0.005)
    n_steps = params.get("n_steps", 100)
    dt = params.get("dt", 0.01)
    return f'''\
"""Two-species reaction-diffusion system — FEniCSx/dolfinx
du/dt = D1 * laplacian(u) + R_u(u, v)
dv/dt = D2 * laplacian(v) + R_v(u, v)
Backward Euler time stepping, coupled system.
"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import ufl
import numpy as np
from basix.ufl import element, mixed_element

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

# Mixed function space for two species
P1 = element("Lagrange", domain.topology.cell_name(), 1)
ME = mixed_element([P1, P1])
W = fem.functionspace(domain, ME)

# Previous time step solution
w_old = fem.Function(W)
(u_old, v_old) = ufl.split(w_old)

# Current time step solution
w = fem.Function(W)
(u, v) = ufl.split(w)

# Test functions
(phi_u, phi_v) = ufl.TestFunctions(W)

# Parameters
D1 = fem.Constant(domain, default_scalar_type({D1}))
D2 = fem.Constant(domain, default_scalar_type({D2}))
dt = fem.Constant(domain, default_scalar_type({dt}))

# Reaction terms — generic reaction kinetics (user should customize)
# Example: Gray-Scott-type or Lotka-Volterra-type
r_u = u * (1.0 - u) - u * v
r_v = -v + u * v

# Weak form (backward Euler)
F = ((u - u_old) / dt * phi_u + D1 * ufl.dot(ufl.grad(u), ufl.grad(phi_u)) - r_u * phi_u) * ufl.dx \\
  + ((v - v_old) / dt * phi_v + D2 * ufl.dot(ufl.grad(v), ufl.grad(phi_v)) - r_v * phi_v) * ufl.dx

# No-flux (Neumann zero) BCs — natural, no Dirichlet needed

# Initial conditions — localized perturbation
def init_u(x):
    return 0.5 + 0.1 * np.exp(-50 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))

def init_v(x):
    return 0.25 + 0.1 * np.exp(-50 * ((x[0] - 0.3)**2 + (x[1] - 0.3)**2))

V0, _ = W.sub(0).collapse()
V1, _ = W.sub(1).collapse()
u_init = fem.Function(V0)
u_init.interpolate(init_u)
v_init = fem.Function(V1)
v_init.interpolate(init_v)
w.sub(0).interpolate(u_init)
w.sub(1).interpolate(v_init)
w.x.scatter_forward()
w_old.x.array[:] = w.x.array[:]

# Newton solver
problem = NonlinearProblem(F, w, bcs=[], petsc_options_prefix="rd",
    petsc_options={{"ksp_type": "preonly", "pc_type": "lu",
                   "pc_factor_mat_solver_type": "mumps",
                   "snes_rtol": 1e-8, "snes_max_it": 30}})

# Time loop
n_steps = {n_steps}
for step in range(n_steps):
    w_old.x.array[:] = w.x.array[:]
    problem.solve()
    if step % max(1, n_steps // 10) == 0:
        (u_s, v_s) = w.split()
        print(f"Step {{step}}: u in [{{w.sub(0).collapse().x.array.min():.4f}}, {{w.sub(0).collapse().x.array.max():.4f}}]")

# Output final state
u_out = w.sub(0).collapse()
v_out = w.sub(1).collapse()
u_out.name = "species_u"
v_out.name = "species_v"

# Interpolate to P1 for output
V_out = fem.functionspace(domain, ("Lagrange", 1))
u_p1 = fem.Function(V_out, name="species_u")
v_p1 = fem.Function(V_out, name="species_v")
u_p1.interpolate(u_out)
v_p1.interpolate(v_out)

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "species_u.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_p1)
with XDMFFile(domain.comm, "species_v.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(v_p1)

print(f"Reaction-diffusion: {{n_steps}} steps complete")
print(f"u: [{{u_p1.x.array.min():.6e}}, {{u_p1.x.array.max():.6e}}]")
print(f"v: [{{v_p1.x.array.min():.6e}}, {{v_p1.x.array.max():.6e}}]")
print(f"DOFs: {{W.dofmap.index_map.size_global}}")
'''
