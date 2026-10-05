"""scikit-fem heat conduction generators and knowledge."""


def _heat_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Heat conduction with Dirichlet BCs on left and right."""
    nx = params.get("nx", 32)
    T_left = params.get("T_left", 100.0)
    T_right = params.get("T_right", 0.0)
    return f'''\
"""Heat conduction on [0,1]² — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace
import numpy as np
import json

_tol = 1e-10
m = (MeshQuad.init_tensor(np.linspace(0, 1, {nx+1}), np.linspace(0, 1, {nx+1}))
     .with_boundaries({{
         "left":  lambda x: x[0] < _tol,
         "right": lambda x: x[0] > 1.0 - _tol,
     }}))
e = ElementQuad1()
ib = Basis(m, e)

K = laplace.assemble(ib)
f = ib.zeros()

# Dirichlet BCs.  In scikit-fem >= 8, dofs are looked up by passing the
# boundary tag name to `get_dofs(...)` directly; the older
# `ib.get_dofs()["name"]` subscript form raises TypeError because
# `DofsView` is not subscriptable by string.
left_dofs = ib.get_dofs("left").flatten()
right_dofs = ib.get_dofs("right").flatten()
D = np.concatenate([left_dofs, right_dofs])
# `condense(..., x=...)` expects a full-size vector (length = total DOFs)
# with the prescribed values at the constrained positions, NOT just the
# boundary values concatenated.  Build the full vector explicitly.
x_full = ib.zeros()
x_full[left_dofs]  = {T_left}
x_full[right_dofs] = {T_right}

u = solve(*condense(K, f, x=x_full, D=D))
print(f"Temperature: max={{u.max():.6f}}")

import meshio
cells = [("quad", m.t.T)]
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])]) if m.p.shape[0] == 2 else m.p.T
mio = meshio.Mesh(points, cells, point_data={{"phi": u}})
mio.write("result.vtu")

summary = {{"max_value": float(u.max()), "n_dofs": int(K.shape[0])}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


def _heat_transient_2d_skfem(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Time-dependent heat equation with backward Euler time integration."""
    nx = params.get("nx", 32)
    dt = params.get("dt", 0.001)
    T_end = params.get("T_end", 0.1)
    f_val = params.get("f", 1.0)
    return f'''\
"""Transient heat equation: du/dt - Δu = f — backward Euler — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace, mass, unit_load
import numpy as np
from scipy.sparse.linalg import spsolve
import json

# Mesh: structured quad mesh
m = MeshQuad.init_tensor(np.linspace(0, 1, {nx + 1}), np.linspace(0, 1, {nx + 1}))
e = ElementQuad1()
ib = Basis(m, e)

# Assembly: stiffness and mass matrices
K = laplace.assemble(ib)
M = mass.assemble(ib)

# Source vector
f = {f_val} * unit_load.assemble(ib)

# Boundary DOFs: u=0 on all boundaries
D = ib.get_dofs().flatten()
I = ib.complement_dofs(D)

# Backward Euler: (M + dt*K) * u_new = M * u_old + dt * f
dt = {dt}
A = M + dt * K

# Factor the system matrix once (reused each step)
from scipy.sparse.linalg import factorized
A_solve = factorized(A[I][:, I].tocsc())

# Initial condition: u=0
u = ib.zeros()

# Time stepping
t = 0.0
n_steps = int({T_end} / dt)
for step in range(n_steps):
    rhs = M @ u + dt * f
    u[D] = 0.0
    u[I] = A_solve(rhs[I])
    t += dt

max_val = u.max()
print(f"t={{t:.4f}}, max(u) = {{max_val:.10f}}, steps={{n_steps}}")

import meshio
cells = [("quad", m.t.T)]
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])]) if m.p.shape[0] == 2 else m.p.T
mio = meshio.Mesh(points, cells, point_data={{"temperature": u}})
mio.write("result.vtu")

summary = {{
    "max_value": float(max_val),
    "n_dofs": len(u),
    "n_elements": m.nelements,
    "time": t,
    "steps": n_steps,
    "dt": dt,
    "element_type": "Q1 quad",
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Transient heat solve complete.")
'''


KNOWLEDGE = {
    "heat": {
        "description": "Heat conduction (steady/transient) — examples 19, 25, 28, 39, 50",
        "solver": "Direct sparse (scipy), or time-stepping for transient",
        "pitfalls": [
            "[Syntax] Non-homogeneous Dirichlet must be applied "
            "via skfem.condense(K, f, x=boundary_values, "
            "D=boundary_dofs). Calling scipy.sparse.linalg.spsolve "
            "on the un-condensed K with a non-trivial RHS does NOT "
            "raise — it returns a vector of numerical garbage from "
            "the rank-deficient null space. Signal: spsolve "
            "returns a finite array but np.max(np.abs(u)) is "
            "order 1e15-1e17 (with MeshTri refined×3, P1 basis, "
            "f=ones), and the boundary DOF values do not match "
            "boundary_values. (Verified empirically 2026-06-01 "
            "— prior catalog text said spsolve 'raises singular "
            "matrix' which is wrong for scipy's SuperLU default; "
            "you only see the issue via the giant-magnitude "
            "solution.)",
            "[Numerical] Transient heat: M*du/dt + K*u = f → "
            "backward Euler is (M + dt*K)*u_new = M*u_old + dt*f. "
            "Forgetting the M*u_old term (using K*u_old) is the "
            "classic 'all variants of theta-method confused' bug. "
            "Signal: the solution BLOWS UP, it does not decay — "
            "the prior 'decays to zero in one step' signal was "
            "WRONG. Measured 2026-08-03 on skfem 12.0.1, "
            "MeshTri().refined(3) / ElementTriP1, "
            "u0 = sin(pi x) sin(pi y), dt = 0.005, six backward-"
            "Euler steps: the correct M@u_old RHS gives max|u| = "
            "9.07e-01, 8.23e-01, 7.46e-01, 6.77e-01, 6.14e-01, "
            "5.57e-01 (matching the analytic per-step decay "
            "factor exp(-2 pi^2 dt) = 0.9060); substituting "
            "K@u_old gives 1.86e+01, 3.47e+02, 6.43e+03, "
            "4.67e+05, 5.82e+07, 7.81e+09; and using the bare "
            "u_old vector with no matrix gives 6.11e+01 ... "
            "5.25e+10. Watch max|u| against the analytic decay "
            "factor, not for a collapse to zero. (Verified "
            "empirically 2026-08-03 — signal correction.)",
            "[Integration] Conjugate heat transfer (example 28) "
            "couples fluid and solid on ONE mesh with ONE "
            "temperature field: the regions come from "
            "mesh.subdomains (a dict of element-index arrays, "
            "'fluid'/'solid' in ex28.msh). Conduction is "
            "assembled on Basis(mesh, element) with a piecewise "
            "P0 conductivity, and advection only on "
            "Basis(mesh, element, elements="
            "mesh.subdomains['fluid']). Signal: solving each region in "
            "isolation gives interface temperature jumps O(1) "
            "instead of continuous (max(T_fluid - T_solid) on "
            "interface DOFs is on the order of the temperature "
            "difference, not 0).",
            "[Numerical] Forward Euler (theta=0) for transient "
            "heat: u_new = u_old - dt*M^{-1}*K*u_old + dt*M^{-1}*f. "
            "Stability requires dt < 2/lambda_max(M^{-1}K) ~ "
            "C*h^2/alpha (CFL condition); coarse mesh + large "
            "diffusivity makes this dt tiny. Signal: with dt "
            "above the CFL bound, linfty_norm(u_new) grows "
            "exponentially across time steps and the solution "
            "diverges to ±inf within ~10 steps. Switch to "
            "backward Euler or Crank-Nicolson for unconditional "
            "stability.",
            "[API] basis.get_dofs() returns a skfem.DofsView "
            "object that is NOT subscriptable by string. Code "
            "like `ib.get_dofs()['left']` raises "
            "TypeError: 'DofsView' object is not subscriptable. "
            "The correct API is to pass the boundary tag name "
            "positionally on construction: "
            "ib.get_dofs('left'). This trap matters because a "
            "natural reading of 'get_dofs then index into the "
            "result' mirrors dict access patterns elsewhere in "
            "scipy/numpy. Signal: TypeError with the literal "
            "string 'DofsView' + 'not subscriptable' in the "
            "message. (Verified empirically 2026-06-01 — "
            "Tier-2 fixture heat_dofsview_and_condense_x_shape "
            "in scripts/tier2_fixtures/skfem/.)",
            "[API] skfem.condense(K, f, x=..., D=D)'s x "
            "argument must be a FULL-SIZE vector of length "
            "basis.N (the total number of DOFs in the system), "
            "NOT just the constrained values concatenated. "
            "Passing a short array of length len(D) raises "
            "IndexError 'index <N> is out of bounds for axis 0 "
            "with size <constrained count>' from condense's "
            "internal x[D] indexing. Build x via "
            "x_full = basis.zeros(); x_full[D_left] = ...; "
            "x_full[D_right] = ... — never assemble a short "
            "array of just the BC values. Signal: IndexError "
            "with literal 'out of bounds' substring at the "
            "condense call site, before any solve attempt. "
            "(Verified empirically 2026-06-01 — same Tier-2 "
            "fixture as #4.)",
        ],
    },
    "heat_transient": {
        "description": "Time-dependent heat equation with backward Euler (scikit-fem)",
        "solver": "Backward Euler: (M + dt*K)*u_new = M*u_old + dt*f, factorized for efficiency",
        "elements": "ElementQuad1, ElementTriP1 (any standard H1 element)",
        "pitfalls": [
            "[Numerical] Backward Euler is unconditionally stable "
            "but only first-order accurate in time. Signal: "
            "manufactured-solution study iterating scipy.sparse."
            "linalg.spsolve with halved dt shows the linfty_norm "
            "(u_h - u_exact) interpolated onto an InteriorBasis "
            "decreasing by a factor ~2 per halving instead of ~4 "
            "(slope 1 vs 2 on a log-log plot).",
            "[Numerical] Factor the system matrix once with "
            "scipy.sparse.linalg.factorized() and reuse across "
            "time steps. Re-factoring every step costs O(N^1.5) "
            "vs O(N) for back-substitution. Signal: per-step "
            "wall time is dominated by factorisation, scaling "
            "as N^1.5 instead of N as the mesh is refined.",
            "[Numerical] Crank-Nicolson (theta=0.5): "
            "(M + 0.5*dt*K)*u_new = (M - 0.5*dt*K)*u_old + dt*f. "
            "Symmetric formula required for second-order accuracy. "
            "Signal: linfty_norm(u_h - u_exact) on the InteriorBasis "
            "after time-stepping with scipy.sparse.linalg.spsolve "
            "decreases by ~4 per dt halving (slope 2); mixing up "
            "the (1-theta) factor on RHS degrades to slope 1 even "
            "though theta=0.5 was set.",
            "[API] Mass matrix M comes from "
            "skfem.models.poisson.mass, which is a convenience, "
            "NOT a fast path. The shipped helper is literally "
            "`@BilinearForm` / `def mass(u, v, _): return u * v`, "
            "so re-implementing u*v as a BilinearForm produces "
            "the same object of the same class — there is no "
            "compiled implementation to miss and no speed penalty "
            "to pay. A hand-written u*v mass matrix is NOT slow "
            "relative to the helper; the two assemble the same "
            "way because they ARE the same form. Use the helper "
            "for brevity; write your own when the integrand needs "
            "a coefficient. "
            "Signal: there is nothing to time here, and a "
            "wall-clock comparison would settle nothing anyway. "
            "If you want to confirm the two forms agree, assemble "
            "both and subtract: same shape, same non-zero count, "
            "identical indices and indptr, and a difference "
            "matrix with zero stored entries after "
            "`eliminate_zeros()`; the total mass `M.sum()` equals "
            "the measure of the domain. Note that "
            "`inspect.getsource` on the FORM object raises "
            "TypeError, so the definition has to be read out of "
            "skfem/models/poisson.py rather than off the form. "
            "(Verified 2026-08-06 on skfem 12.0.1 — the "
            "'10-100x longer' claim is refuted by the shipped "
            "source itself.)",
            "[Syntax] For non-homogeneous BCs that change in "
            "time: re-condense at each step or pre-compute the "
            "lifting once. Signal: time-evolving boundary "
            "temperature does not appear in the solution — "
            "max(T - T_D) at boundary DOFs is O(1) instead of "
            "0 because the same x= argument was reused frozen.",
        ],
    },
}

GENERATORS = {
    "heat_2d": _heat_2d,
    "heat_2d_steady": _heat_2d,
    "heat_transient_2d": _heat_transient_2d_skfem,
}
