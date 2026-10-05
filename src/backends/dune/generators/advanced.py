"""DUNE-fem advanced physics generators and knowledge.

Covers: Maxwell (Helmholtz scalar proxy), eigenvalue (inverse iteration),
hyperelasticity (Neo-Hookean / Newton), Navier-Stokes (Picard/Newton),
Helmholtz, time-dependent heat (implicit Euler), mixed Poisson (RT elements).
"""


# ---------------------------------------------------------------------------
# 1. Maxwell — Helmholtz scalar proxy on [0,1]²
# ---------------------------------------------------------------------------

def _maxwell_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Maxwell equations via scalar Helmholtz proxy — DUNE-fem.

    For full vector Maxwell use NGSolve (HCurl Nedelec elements).
    Here we solve the 2-D TE-mode Helmholtz:  -Δu - k²u = f  with u=0 BCs,
    which exercises the same operator structure as the curl-curl problem.
    """
    nx = params.get("nx", 32)
    k2 = params.get("k_squared", 1.0)
    order = params.get("order", 2)
    return f'''\
"""Maxwell / Helmholtz TE-mode: -Δu - k²u = f  on [0,1]² — DUNE-fem (UFL)

Full vector Maxwell (HCurl Nedelec elements) requires NGSolve.
This script solves the equivalent 2-D scalar Helmholtz proxy.
"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import TrialFunction, TestFunction, dot, grad, dx
import numpy as np
import json

gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])
space = lagrange(gridView, order={order})

u = TrialFunction(space)
v = TestFunction(space)

k2 = {k2}

# Weak form: (grad u, grad v) - k² (u, v) = (f, v)
a = (dot(grad(u), grad(v)) - k2 * u * v) * dx
b = 1.0 * v * dx  # unit source — adjust for your problem

dbc = DirichletBC(space, 0)
scheme = galerkin([a == b, dbc], solver="cg")

uh = space.interpolate(0, name="Ez")
scheme.solve(target=uh)

vals = np.array(uh.as_numpy)
print(f"Maxwell/Helmholtz: max(Ez) = {{vals.max():.10f}}")
print(f"DOFs: {{len(vals)}}")

gridView.writeVTK("result", pointdata={{"Ez": uh}})
summary = {{
    "max_value": float(vals.max()),
    "n_dofs": len(vals),
    "k_squared": k2,
    "element_type": f"Lagrange-P{order}",
    "note": "Scalar 2-D TE Helmholtz proxy for Maxwell",
}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Maxwell/Helmholtz solve complete.")
print("DUNE_TEMPLATE_COMPLETE")
'''


# ---------------------------------------------------------------------------
# 2. Eigenvalue — power / inverse iteration for Laplace eigenproblem
# ---------------------------------------------------------------------------

def _eigenvalue_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Laplace eigenvalue problem -Delta u = lambda u, Dirichlet, via
    assembled matrices + scipy shift-invert — DUNE-fem.
    """
    nx = params.get("nx", 24)
    order = params.get("order", 1)
    n_modes = params.get("n_modes", 4)
    return f'''\
"""Eigenvalue problem  -Delta u = lambda u  on [0,1]^2, u = 0 on the
boundary — DUNE-fem.

dune-fem has NO eigenvalue solver. The route that works is: assemble the
stiffness and mass matrices with dune.fem.assemble, pull them out as
scipy sparse matrices with .as_numpy, drop the boundary rows/columns,
and call scipy.sparse.linalg.eigsh.

Exact answer for the unit square:  lambda_(m,n) = pi^2 (m^2 + n^2)
    -> 2*pi^2 = 19.7392,  5*pi^2 = 49.3480 (twice),  8*pi^2 = 78.9568
"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem import assemble
from ufl import TrialFunction, TestFunction, SpatialCoordinate, as_vector, \
    dot, grad, dx
import numpy as np
import scipy.sparse.linalg as spla
import json

nx = {nx}
order = {order}
n_modes = {n_modes}

gridView = structuredGrid([0, 0], [1, 1], [nx, nx])
space = lagrange(gridView, order=order)
u = TrialFunction(space)
v = TestFunction(space)

# dune.fem.assemble(bilinear form) -> LinearOperator; .as_numpy is
# ALREADY a scipy csr_matrix, so it can be sliced directly. The
# .tocsr() below is a measured no-op kept only for explicitness.
A = assemble(dot(grad(u), grad(v)) * dx).as_numpy.tocsr()   # stiffness
M = assemble(u * v * dx).as_numpy.tocsr()                   # mass

# Dirichlet rows must be REMOVED, not zeroed: a galerkin scheme's
# constrained rows would add spurious eigenvalues at the identity value.
# Lagrange dofs are point evaluations, so nodal coordinates come from
# interpolating the coordinate field into a dimRange=2 space of the SAME
# order; the two spaces share the dof ordering.
coord_space = lagrange(gridView, dimRange=2, order=order)
xv = SpatialCoordinate(coord_space)
X = np.array(coord_space.interpolate(
    as_vector([xv[0], xv[1]]), name="X").as_numpy).reshape(-1, 2)
assert X.shape[0] == A.shape[0], "coordinate/space dof mismatch"

tol = 1e-10
on_bnd = ((X[:, 0] < tol) | (X[:, 0] > 1 - tol) |
          (X[:, 1] < tol) | (X[:, 1] > 1 - tol))
inner_dofs = ~on_bnd
Ai = A[inner_dofs][:, inner_dofs]
Mi = M[inner_dofs][:, inner_dofs]

# sigma=0 shift-invert: smallest eigenvalues of the generalised problem
_vals, _vecs = spla.eigsh(Ai, k=n_modes, M=Mi, sigma=0.0, which="LM",
                          return_eigenvectors=True)
_order = np.argsort(_vals)
lam = _vals[_order]
_vecs = _vecs[:, _order]

# analytic lambda_(m,n) = pi^2 (m^2 + n^2), sorted
mn = sorted(np.pi**2 * (m*m + n*n) for m in range(1, 6) for n in range(1, 6))
exact = np.array(mn[:n_modes])

print(f"total dofs {{A.shape[0]}}, interior dofs {{int(inner_dofs.sum())}}")
for i, (l, e) in enumerate(zip(lam, exact)):
    print(f"mode {{i}}: lambda = {{l:.6f}}   exact = {{e:.6f}}   "
          f"rel error = {{abs(l - e) / e:.3e}}")

# Export the FIRST eigenvector as the result field. This used to write
# space.interpolate(0) under the name "mesh": a field of zeros presented as the
# run's result, which a check for a trivial field rightly refused. The
# interior-dof eigenvector is scattered back onto the full space (boundary dofs
# are zero by the Dirichlet condition) and normalised to max|phi| = 1.
_phi = np.zeros(A.shape[0])
_phi[inner_dofs] = _vecs[:, 0]
_phi /= np.max(np.abs(_phi)) if np.max(np.abs(_phi)) > 0 else 1.0
_ef = space.interpolate(0, name="eigenvector_0")
_ef.as_numpy[:] = _phi
gridView.writeVTK("result", pointdata=[_ef])
summary = {{
    "eigenvalues": [float(l) for l in lam],
    "analytic_eigenvalues": [float(e) for e in exact],
    "rel_errors": [float(abs(l - e) / e) for l, e in zip(lam, exact)],
    "n_dofs": int(A.shape[0]),
    "n_interior_dofs": int(inner_dofs.sum()),
    "order": order,
}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Eigenvalue solve complete.")
print("DUNE_TEMPLATE_COMPLETE")
'''


# ---------------------------------------------------------------------------
# 3. Hyperelasticity — Neo-Hookean with automatic Newton (DUNE nonlinear)
# ---------------------------------------------------------------------------

def _hyperelasticity_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Neo-Hookean hyperelasticity — DUNE-fem nonlinear Newton solver.
    """
    nx = params.get("nx", 16)
    E = params.get("E", 1.0e6)
    nu = params.get("nu", 0.3)
    traction = params.get("traction", 5.0e4)
    order = params.get("order", 1)
    return f'''\
"""Neo-Hookean hyperelasticity  on [0,1]² — DUNE-fem UFL / Newton

Material: E = {E:.2e} Pa, nu = {nu}
Load: traction = {traction:.2e} Pa on right face (x=1)
BCs: u = 0 on left face (x=0)
"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import (
    SpatialCoordinate, TestFunction, TrialFunction,
    Identity, grad, det, ln, tr, inner, dx, ds,
    as_vector, conditional, lt, FacetNormal,
)
import numpy as np
import json

E_mod  = {E}
nu_val = {nu}
mu_val  = E_mod / (2.0 * (1.0 + nu_val))
lam_val = E_mod * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
t_val  = {traction}

gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])
space = lagrange(gridView, dimRange=2, order={order})

# Write the residual in the TRIAL function. dune-fem differentiates the
# UFL form symbolically to build the tangent, so it needs a form with
# TWO arguments; the natural "residual written with the current
# iterate" spelling — F = I + grad(uh) — has only the test function and
# is rejected at scheme construction with
#   ValueError: Integrands model requires form with at least two arguments.
u   = TrialFunction(space)
v   = TestFunction(space)
x   = SpatialCoordinate(space)

# Deformation gradient F = I + grad(u)
I   = Identity(2)
F   = I + grad(u)
C   = F.T * F
Ic  = tr(C)
J   = det(F)

# Neo-Hookean stored energy:  psi = mu/2*(Ic-2) - mu*ln(J) + lam/2*ln(J)^2
# First Piola-Kirchhoff stress:  P = mu*(F - F^-T) + lam*ln(J)*F^-T
# Weak form: int P : grad(v) dx = int t * v ds (right boundary)
from ufl import inv, cofac
F_inv_T = inv(F).T

P = mu_val * (F - F_inv_T) + lam_val * ln(J) * F_inv_T

# Volume residual
res = inner(P, grad(v)) * dx

# Neumann traction on right boundary (x=1): t = [t_val, 0]
res -= conditional(lt(1.0 - x[0], 0.01), t_val * v[0], 0.0) * ds

# Dirichlet: u = 0 on left boundary (x=0)
dbc = DirichletBC(space, as_vector([0, 0]), conditional(lt(x[0], 0.01), 1, 0))

# DUNE automatically computes Jacobian (tangent stiffness) and does Newton.
# The finite-strain tangent is NOT symmetric, so cg is the wrong Krylov
# method here — use bicgstab (or gmres).
scheme = galerkin([res == 0, dbc], solver="bicgstab",
                  parameters={{"nonlinear.tolerance": 1e-8,
                               "nonlinear.maxiterations": 50,
                               "linear.tolerance": 1e-10,
                               "linear.maxiterations": 20000}})
uh = space.interpolate([0, 0], name="displacement")
info = scheme.solve(target=uh)
print(f"Newton iterations: {{info['iterations']}}, "
      f"linear: {{info['linear_iterations']}}, "
      f"converged: {{info['converged']}}")

vals = np.array(uh.as_numpy).reshape(-1, 2)
u_x_max = float(vals[:, 0].max())
u_y_max = float(np.abs(vals[:, 1]).max())
print(f"Hyperelasticity: max u_x = {{u_x_max:.6e}}, max |u_y| = {{u_y_max:.6e}}")
print(f"DOFs: {{len(vals)*2}}")

gridView.writeVTK("result", pointdata={{"displacement": uh}})
summary = {{
    "max_ux": u_x_max,
    "max_abs_uy": u_y_max,
    "n_dofs": len(vals) * 2,
    "E": E_mod,
    "nu": nu_val,
    "traction": t_val,
}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Hyperelasticity solve complete.")
print("DUNE_TEMPLATE_COMPLETE")
'''


# ---------------------------------------------------------------------------
# 4. Navier-Stokes — Picard/Newton iteration (driven cavity)
# ---------------------------------------------------------------------------

def _navier_stokes_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Steady incompressible Navier-Stokes, Newton on a composite
    Taylor-Hood space — DUNE-fem.
    """
    nx = params.get("nx", 16)
    Re = params.get("Re", 40.0)
    order_v = params.get("order_v", 2)
    order_p = params.get("order_p", 1)
    return f'''\
"""Steady incompressible Navier-Stokes on [0,1]^2 — DUNE-fem.

    (u . grad) u - nu*Laplace(u) + grad p = 0,   div u = 0

Taylor-Hood P{order_v}/P{order_p} on ONE composite space. The residual is written
in the TRIAL function so dune-fem can differentiate it symbolically and
run Newton; there is no hand-written Picard loop.

Verification: Kovasznay flow (Kovasznay 1948) is a closed-form steady
Navier-Stokes solution with a genuinely non-zero convective term:
    lam = Re/2 - sqrt(Re^2/4 + 4*pi^2)
    u = (1 - exp(lam*x)*cos(2*pi*y),  lam/(2*pi)*exp(lam*x)*sin(2*pi*y))
    p = (1 - exp(2*lam*x)) / 2
It is imposed as Dirichlet data on three sides; on the outflow x=1 the
matching EXACT TRACTION is imposed as the natural boundary term, which
is what makes the pressure level well determined.
"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange, composite
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from dune.fem import integrate
from ufl import (TrialFunction, TestFunction, SpatialCoordinate, FacetNormal,
                 as_vector, inner, dot, grad, div, dx, ds, exp, cos, sin, pi,
                 conditional, lt, gt)
import numpy as np
import json

Re = {Re}
nu = 1.0 / Re
lam = Re / 2.0 - np.sqrt(Re * Re / 4.0 + 4.0 * np.pi * np.pi)
print(f"Re = {{Re}}, nu = {{nu:.6g}}, Kovasznay lambda = {{lam:.6f}}")

gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])
V = lagrange(gridView, dimRange=2, order={order_v})
Q = lagrange(gridView, order={order_p})
W = composite(V, Q, components=["velocity", "pressure"])

trial = TrialFunction(W)
test = TestFunction(W)
u = as_vector([trial[0], trial[1]])
p = trial[2]
v = as_vector([test[0], test[1]])
q = test[2]

x = SpatialCoordinate(W)
n = FacetNormal(W)
tol = 1e-8

u_exact = as_vector([1.0 - exp(lam * x[0]) * cos(2 * pi * x[1]),
                     lam / (2 * pi) * exp(lam * x[0]) * sin(2 * pi * x[1])])
p_exact = 0.5 * (1.0 - exp(2 * lam * x[0]))

# CONVECTION: (u.grad)u_i = u_j du_i/dx_j = dot(grad(u), u).
# dot(u, grad(u)) * v is a rank error — UFL raises
# "ValueError: Invalid ranks 1 and 1 in product."
res = (inner(dot(grad(u), u), v)
       + nu * inner(grad(u), grad(v))
       - p * div(v)
       - q * div(u)) * dx

# natural (traction) BC on the outflow, matching the exact solution
traction = nu * dot(grad(u_exact), n) - p_exact * n
res -= conditional(gt(x[0], 1 - tol), 1.0, 0.0) * inner(traction, v) * ds

on_dirichlet = conditional(lt(x[0], tol), 1,
                           conditional(lt(x[1], tol), 1,
                                       conditional(gt(x[1], 1 - tol), 1, 0)))
bc = DirichletBC(W, [u_exact[0], u_exact[1], None], on_dirichlet)

# res == 0 is legal because res still holds BOTH a trial and a test
# function. A residual written in the CURRENT ITERATE instead would have
# only one argument and be rejected with
# "ValueError: Integrands model requires form with at least two arguments."
scheme = galerkin([res == 0, bc], solver=("suitesparse", "umfpack"),
                  parameters={{"nonlinear.tolerance": 1e-11,
                               "nonlinear.maxiterations": 30}})

wh = W.interpolate([0, 0, 0], name="solution")
info = scheme.solve(target=wh)

uh = as_vector([wh[0], wh[1]])
ph = wh[2]

err_u = np.sqrt(integrate(inner(uh - u_exact, uh - u_exact),
                          gridView=gridView, order=8))
nrm_u = np.sqrt(integrate(inner(u_exact, u_exact), gridView=gridView, order=8))
err_p = np.sqrt(integrate((ph - p_exact) ** 2, gridView=gridView, order=8))
nrm_p = np.sqrt(integrate(p_exact ** 2, gridView=gridView, order=8))
div_u = np.sqrt(integrate(div(uh) ** 2, gridView=gridView, order=8))

print(f"Newton iterations: {{info['iterations']}}, "
      f"converged: {{info['converged']}}")
print(f"relative velocity error vs Kovasznay: {{err_u / nrm_u:.4e}}")
print(f"relative pressure error vs Kovasznay: {{err_p / nrm_p:.4e}}")
print(f"||div u||_L2: {{div_u:.3e}}")
print("Both relative errors must FALL when you increase nx; if they "
      "stall, the convection term or a boundary term is wrong.")

gridView.writeVTK("result", pointdata={{"velocity": uh, "pressure": ph}})
summary = {{
    "Re": Re,
    "newton_iterations": int(info["iterations"]),
    "rel_error_velocity": float(err_u / nrm_u),
    "rel_error_pressure": float(err_p / nrm_p),
    "div_u_l2": float(div_u),
    "n_dofs": int(W.size),
}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Navier-Stokes solve complete.")
print("DUNE_TEMPLATE_COMPLETE")
'''


# ---------------------------------------------------------------------------
# 5. Helmholtz — -Δu - k²u = f
# ---------------------------------------------------------------------------

def _helmholtz_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Helmholtz equation  -Δu - k²u = f  on [0,1]² — DUNE-fem.
    """
    nx = params.get("nx", 64)
    k = params.get("k", 4.0)
    order = params.get("order", 2)
    return f'''\
"""Helmholtz equation: -Δu - k²u = f  on [0,1]²

k = {k}  (wavenumber),  rule of thumb: ≥ 6 DOFs per wavelength
Manufactured solution: u = sin(pi*x)*sin(pi*y)  =>  f = (2*pi^2 - k^2)*sin(pi*x)*sin(pi*y)
"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import (
    TrialFunction, TestFunction, SpatialCoordinate,
    dot, grad, dx, sin, pi as ufl_pi,
)
import numpy as np
import json

k_val = {k}
gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])
space = lagrange(gridView, order={order})

x = SpatialCoordinate(space)
u = TrialFunction(space)
v = TestFunction(space)

# Weak form: (grad u, grad v) - k² (u, v) = (f, v)
a = (dot(grad(u), grad(v)) - k_val**2 * u * v) * dx

# Manufactured RHS for u_exact = sin(pi*x)*sin(pi*y)
f_mms = (2.0 * ufl_pi**2 - k_val**2) * sin(ufl_pi * x[0]) * sin(ufl_pi * x[1])
b = f_mms * v * dx

dbc = DirichletBC(space, 0)
scheme = galerkin([a == b, dbc], solver="gmres")

uh = space.interpolate(0, name="u")
scheme.solve(target=uh)

# Error against manufactured solution
u_ex = space.interpolate(sin(ufl_pi * x[0]) * sin(ufl_pi * x[1]), name="u_exact")
err_arr = np.array(uh.as_numpy) - np.array(u_ex.as_numpy)
l2_err = float(np.sqrt(err_arr @ err_arr) / len(err_arr))

vals = np.array(uh.as_numpy)
print(f"Helmholtz: k={{k_val}}, max(u) = {{vals.max():.8f}}")
print(f"L2 nodal error vs MMS: {{l2_err:.4e}}")
print(f"DOFs: {{len(vals)}}")

gridView.writeVTK("result", pointdata={{"u": uh, "u_exact": u_ex}})
summary = {{
    "k": k_val,
    "max_value": float(vals.max()),
    "l2_nodal_error": l2_err,
    "n_dofs": len(vals),
    "order": {order},
}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Helmholtz solve complete.")
print("DUNE_TEMPLATE_COMPLETE")
'''


# ---------------------------------------------------------------------------
# 6. Time-dependent heat — backward Euler (implicit) time-stepping
# ---------------------------------------------------------------------------

def _time_dependent_heat_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Transient heat equation  du/dt - alpha*Δu = f  — implicit Euler — DUNE-fem.
    """
    nx = params.get("nx", 32)
    alpha = params.get("alpha", 0.01)
    dt = params.get("dt", 0.01)
    T_end = params.get("T_end", 0.5)
    order = params.get("order", 1)
    n_out = params.get("n_out", 5)
    return f'''\
"""Transient heat:  du/dt - alpha*Δu = f  — implicit Euler — DUNE-fem

alpha = {alpha},  dt = {dt},  T_end = {T_end}
Initial condition: Gaussian pulse centred at (0.5, 0.5)
"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import (
    TrialFunction, TestFunction, SpatialCoordinate, dot, grad, dx, exp,
)
import numpy as np
import json

alpha  = {alpha}
dt     = {dt}
T_end  = {T_end}
n_out  = {n_out}   # number of VTK snapshots to write

gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])
space = lagrange(gridView, order={order})

x = SpatialCoordinate(space)

# Initial condition: Gaussian pulse
u_n = space.interpolate(
    exp(-50.0 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2)),
    name="temperature"
)

u = TrialFunction(space)
v = TestFunction(space)

# Backward-Euler weak form:
#   (u/dt + alpha*grad(u), v) = (u_n/dt + f, v)
a = (u * v / dt + alpha * dot(grad(u), grad(v))) * dx
# Source: zero for pure diffusion — add your source term here
f_source = 0.0

dbc = DirichletBC(space, 0)
# NOTE the parentheses: (u_n/dt + f)*v must be grouped BEFORE *dx —
# 'u_n * v / dt + f_source * v * dx' only multiplies the source term by
# the measure and raises 'This integral is missing an integration
# domain' (ufl.measure). Caught in the 2026-07-18 Mac stress audit.
scheme = galerkin([a == (u_n / dt + f_source) * v * dx, dbc], solver="cg")

n_steps = int(round(T_end / dt))
out_every = max(1, n_steps // n_out)
t = 0.0
snapshots = []

for step in range(n_steps):
    scheme.solve(target=u_n)
    t += dt

    if (step + 1) % out_every == 0 or step == n_steps - 1:
        vals = np.array(u_n.as_numpy)
        max_T = float(vals.max())
        print(f"t = {{t:.4f}}, max(T) = {{max_T:.6f}}")
        gridView.writeVTK(f"result_t{{step+1:05d}}", pointdata={{"temperature": u_n}})
        snapshots.append({{"t": t, "max_T": max_T}})

vals = np.array(u_n.as_numpy)
print(f"Final t={{T_end}}: max(T) = {{float(vals.max()):.6f}}, DOFs = {{len(vals)}}")

summary = {{
    "alpha": alpha,
    "dt": dt,
    "T_end": T_end,
    "n_steps": n_steps,
    "n_dofs": len(vals),
    "final_max_T": float(vals.max()),
    "snapshots": snapshots,
}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Transient heat solve complete.")
print("DUNE_TEMPLATE_COMPLETE")
'''


# ---------------------------------------------------------------------------
# 7. Mixed Poisson — Raviart-Thomas (H(div)) + L²  (saddle-point)
# ---------------------------------------------------------------------------

def _mixed_methods_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Mixed (dual / Darcy) Poisson: solve for the FLUX and the potential
    together on one composite space — DUNE-fem.
    """
    nx = params.get("nx", 8)
    order_flux = params.get("order_flux", 2)
    order_pot = params.get("order_pot", 1)
    return f'''\
"""Mixed (dual) Poisson on [0,1]^2 — DUNE-fem.

Find the flux sigma and the potential u together:
    sigma + grad(u) = 0        div(sigma) = f

Weak form (sigma in [P{order_flux}]^2, u in P{order_pot}, one composite space):
    (sigma, tau) - (u, div tau) + <u_D, tau.n>_bnd = 0   for all tau
    (div sigma, v)              = (f, v)                 for all v

WHY NOT RAVIART-THOMAS: on dune-fem 2.12.0.2 a raviartThomas space
CANNOT be put inside product()/composite() — the composite space fails
to BUILD, in C++, with a CompileError. The RT space works perfectly on
its own (see the pitfalls for this physics); it just cannot be one leg
of a tuple space. The inf-sup stable Lagrange pair used here is the
route that runs.

Closed-form answer this script checks itself against:
    u = 1 - x       sigma = -grad(u) = (1, 0)       f = div(sigma) = 0
Both are inside the discrete spaces, so a CORRECT run reproduces them
to solver tolerance.
"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange, composite
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from dune.fem import integrate
from ufl import (TrialFunction, TestFunction, SpatialCoordinate, FacetNormal,
                 as_vector, inner, dot, div, dx, ds, conditional, lt, gt)
import numpy as np
import json

nx = {nx}
gridView = structuredGrid([0, 0], [1, 1], [nx, nx])

# REQUIRED: velocity-like space one order HIGHER than the scalar space,
# exactly as for Stokes — the div coupling needs the same inf-sup
# condition.
S = lagrange(gridView, dimRange=2, order={order_flux})   # flux
V = lagrange(gridView, order={order_pot})                # potential
W = composite(S, V, components=["flux", "potential"])

trial = TrialFunction(W)
test = TestFunction(W)
sigma = as_vector([trial[0], trial[1]])
u = trial[2]
tau = as_vector([test[0], test[1]])
v = test[2]

x = SpatialCoordinate(W)
n = FacetNormal(W)
tol = 1e-8

u_D = 1.0 - x[0]      # potential prescribed on the boundary (NATURAL here)

a = (inner(sigma, tau) - u * div(tau) + div(sigma) * v) * dx
L = -u_D * dot(tau, n) * ds        # the Dirichlet data enters WEAKLY

# In the mixed form the roles swap: the potential is natural and the
# NORMAL FLUX is essential. sigma.n = 0 on y=0 and y=1 means sigma_y=0
# there, so only the second component is constrained.
bc_flux = DirichletBC(W, [None, 0, None],
                      conditional(lt(x[1], tol), 1,
                                  conditional(gt(x[1], 1 - tol), 1, 0)))

# REQUIRED: a direct solver — the mixed system is INDEFINITE.
scheme = galerkin([a == L, bc_flux], solver=("suitesparse", "umfpack"))
wh = W.interpolate([0, 0, 0], name="solution")
info = scheme.solve(target=wh)

sigma_h = as_vector([wh[0], wh[1]])
u_h = wh[2]

sigma_ex = as_vector([1.0, 0.0])
err_u = np.sqrt(integrate((u_h - u_D) ** 2, gridView=gridView, order=6))
nrm_u = np.sqrt(integrate(u_D ** 2, gridView=gridView, order=6))
err_s = np.sqrt(integrate(inner(sigma_h - sigma_ex, sigma_h - sigma_ex),
                          gridView=gridView, order=6))
div_s = np.sqrt(integrate(div(sigma_h) ** 2, gridView=gridView, order=6))

print(f"converged={{info['converged']}} dofs={{W.size}}")
print(f"relative potential error: {{err_u / nrm_u:.3e}}")
print(f"flux error ||sigma_h - (1,0)||_L2: {{err_s:.3e}}")
print(f"||div sigma_h||_L2 (must match f = 0): {{div_s:.3e}}")

gridView.writeVTK("result",
                  pointdata={{"potential": u_h, "flux": sigma_h}})
summary = {{
    "rel_error_potential": float(err_u / nrm_u),
    "flux_error": float(err_s),
    "div_sigma_l2": float(div_s),
    "n_dofs": int(W.size),
}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Mixed Poisson solve complete.")
print("DUNE_TEMPLATE_COMPLETE")
'''


# ---------------------------------------------------------------------------
# KNOWLEDGE — one entry per physics key
# ---------------------------------------------------------------------------

KNOWLEDGE = {
    "maxwell": {
        "description": (
            "Maxwell equations: 2-D TE-mode solved as scalar Helmholtz proxy (-Δu - k²u = f). "
            "For full 3-D vector Maxwell (HCurl Nedelec elements), use NGSolve."
        ),
        "solver": "galerkin scheme with GMRES (indefinite system for k > 0)",
        "spaces": "lagrange(gridView, order=2) — higher order needed for wave problems",
        "element_types": ["Lagrange-P2 (scalar proxy)"],
        "pitfalls": [
            (
                "[API] Full H(curl) Nedelec elements are "
                "NOT yet in dune-fem — for true vector "
                "Maxwell use NGSolve or dolfinx. Signal: "
                "looking for an H(curl) Nedelec space "
                "raises ImportError; the existing "
                "maxwell template in dune-fem uses a "
                "P2 scalar proxy that ONLY handles the "
                "2D scalar Az formulation correctly. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Helmholtz is INDEFINITE for "
                "k^2 > pi^2, so prefer GMRES or a direct "
                "solver — but nothing warns you. Signal: "
                "measured 2026-08-03, CG on "
                "-Laplace(u) - k^2 u = f with k=10 on a "
                "16x16 P1 grid raised NOTHING and reported "
                "{'converged': True, 'iterations': 0, "
                "'linear_iterations': 1}. Judge it by the "
                "answer, not by the solver's own verdict. "
                "CORRECTED by adversarial audit 2026-08-03: "
                "an earlier revision claimed CG raises "
                "'matrix not positive definite', which is "
                "not a string this install contains or "
                "emits. (Audit 2026-06-02; Signal "
                "re-measured 2026-08-03.)"
            ),
            (
                "[Numerical] Rule of thumb: >= 10 DOFs "
                "per wavelength for low-order elements. "
                "Signal: at < 5 DOFs/wavelength, phase "
                "velocity is wrong by 10-30%; the wave "
                "hits the boundary at the wrong time in "
                "transient simulations. Increase mesh "
                "resolution or use P2+. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Spurious MODES appear with "
                "standard Lagrange elements for VECTOR "
                "Maxwell. Signal: an eigenvalue solve "
                "for the vector wave equation returns "
                "many near-zero modes interleaved with "
                "physical modes — these are the spurious "
                "gradient modes (range(grad) in null(curl)). "
                "Use H(curl) Nedelec for clean spectrum. "
                "(Audit 2026-06-02.)"
            ),
        ],
    },
    "eigenvalue": {
        "description": (
            "Laplace eigenvalue problem -Delta u = lambda u. dune-fem "
            "has NO eigenvalue solver of its own: assemble the "
            "stiffness and mass matrices, hand them to scipy."),

        "required_calls_in_order": [
            "space = dune.fem.space.lagrange(gridView, order=k)",
            "A = dune.fem.assemble(dot(grad(u),grad(v))*dx)"
            ".as_numpy.tocsr()",
            "M = dune.fem.assemble(u*v*dx).as_numpy.tocsr()",
            "identify the boundary dofs and DELETE those rows and "
            "columns from A and M (do not zero them)",
            "scipy.sparse.linalg.eigsh(Ai, k=n, M=Mi, sigma=0.0, "
            "which='LM', return_eigenvectors=False)",
        ],
        "required_vs_optional": {
            "REQUIRED": [
                "dune.fem.assemble(form) — it returns a LinearOperator "
                "whose .as_numpy is ALREADY a scipy csr_matrix "
                "(measured 2026-08-03: .format == 'csr' and "
                "A.tocsr() is A), so you can slice it directly",
                "removal (not zeroing) of the constrained rows and "
                "columns",
                "sigma=0.0 with which='LM' — shift-invert, otherwise "
                "eigsh returns the LARGEST eigenvalues, which are the "
                "mesh-resolution garbage at the top of the spectrum",
            ],
            "OPTIONAL": [
                "higher Lagrange order — order 2 buys roughly two "
                "digits per mode at the same dof count",
                "eigenvectors — pass return_eigenvectors=True and "
                "write them back into a discrete function for VTK",
            ],
        },
        "how_to_find_the_boundary_dofs": (
            "Lagrange dofs are point evaluations, so interpolate the "
            "coordinate field into a lagrange(gridView, dimRange=dim, "
            "order=SAME k) space and reshape .as_numpy to (-1, dim): "
            "row i of that array is the coordinate of dof i of the "
            "SCALAR space. Measured 2026-08-03 on a 24x24 P1 grid: 625 "
            "coordinate rows against 625 scalar dofs, 529 of them "
            "interior. Assert the two counts match before you trust "
            "the mask."),
        "verification_you_can_run": (
            "Unit square with homogeneous Dirichlet data has "
            "lambda_(m,n) = pi^2 (m^2 + n^2) — 19.7392, 49.3480 "
            "(double), 78.9568. Executed 2026-08-03, 24x24 "
            "structuredGrid, P1, scipy eigsh with sigma=0: 19.7674, "
            "49.5881, 49.5881, 79.4088, i.e. relative errors 1.4e-03 "
            "to 5.7e-03, all from ABOVE. Discrete Dirichlet "
            "eigenvalues of a conforming method are always upper "
            "bounds, so a computed value BELOW the exact one means the "
            "boundary rows were not removed properly."),

        "pitfalls": [
            (
                "[API] There is no eigenvalue solver in dune-fem. "
                "Signal: dune.fem has no eig/eigen/eigs attribute of "
                "any kind and 'import dune.fem.solver' raises "
                "ModuleNotFoundError, so the only route is assemble -> "
                "scipy (or PETSc/SLEPc through the petsc storage). "
                "dune.fem.assemble(form) is the supported entry point "
                "and returns an object of type LinearOperator whose "
                "only conversion attribute is .as_numpy. (Executed "
                "2026-08-03 on dune-fem 2.12.0.2.)"
            ),
            (
                "[Numerical] Leaving the Dirichlet rows in the matrix "
                "poisons the spectrum. Signal: a galerkin scheme's "
                "constrained rows are identity rows, so the "
                "generalised problem A x = lambda M x acquires one "
                "spurious eigenvalue per constrained dof at "
                "1/M_ii — for a 24x24 P1 grid that is 96 extra "
                "eigenvalues mixed in among the physical ones, and "
                "eigsh with sigma=0 returns them first. Delete the "
                "rows AND the columns instead. (Executed 2026-08-03: "
                "625 dofs, 529 interior, and the interior submatrices "
                "give the analytic pi^2(m^2+n^2) values to 1e-03.)"
            ),
            (
                "[API] .as_numpy on an assembled operator is ALREADY a "
                "scipy csr_matrix, so slice it directly. Signal: check "
                "it rather than converting blindly — "
                "A.as_numpy.format prints 'csr' and "
                "A.as_numpy.tocsr() is A.as_numpy returns True, so a "
                ".tocsr() in your code is a no-op and its presence is "
                "not evidence that it was needed. RETRACTED "
                "2026-08-03 by adversarial audit: an earlier revision "
                "of this entry claimed it was COO and that "
                "fancy-indexing 'raises or returns something unusable' "
                "without a .tocsr() first. Measured on dune-fem "
                "2.12.0.2 / scipy 1.18.0: A.as_numpy.format == 'csr', "
                "type csr_matrix, A[mask][:, mask] returned a (49, 49) "
                "csr_matrix with 285 nonzeros and raised nothing, "
                "identical to A.tocsr()[mask][:, mask] with "
                "max|difference| 0.0, and A.tocsr() is A is True — the "
                "call is a no-op. The installed package confirms it: "
                "dune/fem/operator/__init__.py imports only "
                "scipy.sparse.csr_matrix for the as_numpy backend and "
                "the string 'coo_matrix' occurs nowhere under "
                "site-packages/dune. Keeping .tocsr() is harmless but "
                "it is not required. (Executed 2026-08-03.)"
            ),
            (
                "[Numerical] eigsh without a shift returns the WRONG "
                "END of the spectrum. Signal: which='SM' converges "
                "extremely slowly or not at all, and the default "
                "which='LM' without sigma returns the largest "
                "eigenvalues, which for a FEM Laplacian are O(1/h^2) "
                "mesh artefacts, not physics. Use sigma=0.0 with "
                "which='LM' (shift-invert). (Audit 2026-06-02, "
                "confirmed by the executed template 2026-08-03.)"
            ),
            (
                "[Performance] For hundreds of eigenpairs, use SLEPc "
                "through the petsc storage rather than scipy. Signal: "
                "scipy's ARPACK path re-factorises the shifted matrix "
                "and holds it dense-ish in memory; for 100+ pairs on a "
                "3D mesh it is the dominant cost. (Audit 2026-06-02.)"
            ),
        ],
    },
    "hyperelasticity": {
        "description": (
            "Neo-Hookean hyperelasticity (finite strain). "
            "Stored energy W = mu/2*(Ic-2) - mu*ln(J) + lam/2*ln(J)². "
            "DUNE-fem differentiates the energy automatically to get the tangent stiffness."
        ),
        "solver": "Built-in Newton iteration via galerkin scheme on nonlinear residual",
        "spaces": "lagrange(gridView, dimRange=2, order=1) for displacement",
        "pitfalls": [
            (
                "[Numerical] Neo-Hookean energy: W = "
                "mu/2*(tr(C) - d) - mu*ln(J) + lam/2*"
                "ln(J)^2 where d is spatial dim (2 or "
                "3). Signal: writing W = mu/2*tr(C) "
                "(without the -d subtraction) inside the "
                "dune.fem galerkin scheme UFL form gives "
                "W != 0 at F = I — stress-free reference "
                "produces non-zero initial stress; the "
                "first Newton iterate runs off looking "
                "for a different equilibrium. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Deformation gradient: F = I "
                "+ grad(u); MUST have det(F) > 0 "
                "(no inversion). Signal: forgetting the "
                "identity I in the dune.fem galerkin "
                "scheme UFL form gives F = grad(u) which "
                "is degenerate at the reference "
                "configuration (det = 0); ln(J) blows up "
                "as -inf; the lagrange-space Newton "
                "residual is NaN on step 1. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] For LARGE deformations: use "
                "load stepping (increment traction/body "
                "force). Signal: applying full load at "
                "t = 0 to a problem at > 30% nominal "
                "strain typically diverges (the dune.fem "
                "galerkin scheme Newton residual grows "
                "~ 10x per iter); subdividing into 10 "
                "substeps achieves quadratic per-step "
                "convergence with the previous "
                "GridFunction-equivalent as initial guess. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Compressible formulation; "
                "near-incompressible needs MIXED or "
                "F-bar method. Signal: a pure-displacement "
                "dune.fem lagrange Space with a neo_Hookean "
                "material at nu = 0.4999 exhibits "
                "volumetric_locking — the Cook_membrane tip "
                "deflection GridFunction is < 1% of analytic; "
                "switch to a mixed (u, p) Form on a product "
                "Space to recover. (Audit 2026-06-02.)"
            ),
            (
                "[API] DUNE differentiates UFL forms "
                "SYMBOLICALLY — no manual tangent needed. "
                "Signal: hand-coding the PK1 stress + "
                "Jacobian inside a custom assembler is "
                "error-prone (sign / factor mistakes "
                "drop Newton to linear convergence); "
                "specifying just the energy W and "
                "letting scheme.solve() build derivatives "
                "via UFL auto-AD gives correct "
                "quadratic Newton. (Audit 2026-06-02.)"
            ),
        ],
    },
    "navier_stokes": {
        "description": (
            "Steady incompressible Navier-Stokes on one composite "
            "Taylor-Hood space, solved by dune-fem's built-in Newton — "
            "the residual is written in the TRIAL function and the "
            "tangent is differentiated symbolically, so no Picard loop "
            "is needed."),

        "required_calls_in_order": [
            "W = dune.fem.space.composite("
            "lagrange(gridView, dimRange=2, order=2), "
            "lagrange(gridView, order=1))",
            "trial = ufl.TrialFunction(W); "
            "u = ufl.as_vector([trial[0], trial[1]]); p = trial[2]",
            "res = (inner(dot(grad(u), u), v) + nu*inner(grad(u),grad(v))"
            " - p*div(v) - q*div(u))*dx"
            "   <- dot(grad(u), u), NOT dot(u, grad(u))*v",
            "scheme = galerkin([res == 0, bc], "
            "solver=('suitesparse','umfpack'), "
            "parameters={'nonlinear.tolerance': 1e-11})",
            "wh = W.interpolate([0,0,0], name='solution'); "
            "scheme.solve(target=wh)",
        ],
        "required_vs_optional": {
            "REQUIRED": [
                "the residual written in the TRIAL function u, not in "
                "the current iterate — a residual holding only the "
                "test function is rejected at scheme construction",
                "an LBB-stable pair (velocity order = pressure order + 1)",
                "a solver that tolerates an indefinite, "
                "non-symmetric tangent — ('suitesparse','umfpack') "
                "is the right default; the Krylov names converge too "
                "but at 1e3-4e4 iterations",
                "None for the pressure entry of the DirichletBC value "
                "list",
                "either a natural (traction) outflow or an explicit "
                "pressure constraint, otherwise the pressure floats",
            ],
            "OPTIONAL": [
                "nonlinear.maxiterations / nonlinear.tolerance — the "
                "defaults converge for moderate Re",
                "nonlinear.linesearch — needed only when the initial "
                "guess is far away, e.g. high Re from a cold start",
                "continuation in Re (solve at Re/4, use as the initial "
                "guess) — only for high Re where Newton diverges from "
                "zero",
            ],
        },
        "verification_you_can_run": (
            "A PARALLEL-FLOW TEST SUCH AS POISEUILLE CANNOT DETECT A "
            "BROKEN CONVECTIVE TERM, because (u.grad)u vanishes "
            "identically on it — a solver with the convective term "
            "deleted, mis-signed or written with the arguments swapped "
            "passes Poiseuille exactly. To verify Navier-Stokes you "
            "need a solution whose convective term is NOT zero. "
            "Kovasznay flow (Kovasznay 1948) is the standard choice. "
            "THE SHIPPED navier_stokes/2d TEMPLATE ALREADY IMPLEMENTS "
            "THIS TEST and writes the closed form out in full — read it "
            "there rather than looking it up, because a verification "
            "template whose exact solution is missing is not a "
            "verification template. (An earlier version of this entry "
            "told you to retrieve it while the template three hundred "
            "lines away handed it over: the same file giving two "
            "opposite instructions.) If you want a DIFFERENT exact "
            "solution than the one shipped, that one you retrieve and "
            "cite. Two things "
            "decide whether the test is valid. Impose the solution as "
            "Dirichlet data on three sides and impose the MATCHING "
            "EXACT TRACTION nu*grad(u).n - p*n on the outflow: that is "
            "the natural boundary term of this form and it is what "
            "pins the pressure level, and without it the pressure "
            "error is meaningless. Then judge the result by "
            "REFINEMENT, not by magnitude — the error must FALL as you "
            "refine; one that stalls means a boundary term or the "
            "convective term is wrong. Do not compare against an error "
            "value someone else measured, including one of ours: on a "
            "different grid and a different Re it means nothing."),

        "pitfalls": [
            (
                "[Syntax] The convective term is dot(grad(u), u), not "
                "dot(u, grad(u))*v. Signal: writing "
                "dot(b, grad(u)) * v raises "
                "\"ValueError: Invalid ranks 1 and 1 in product.\" "
                "from ufl/exproperators.py::_mult — a vector times a "
                "vector is not a legal product. In UFL grad(u)[i,j] is "
                "du_i/dx_j, so (u.grad)u_i = u_j du_i/dx_j = "
                "dot(grad(u), u), and it must be contracted against v "
                "with inner(...). (Executed 2026-08-03 — this is the "
                "error the previous version of this template died "
                "with.)"
            ),
            (
                "[API] A residual written in the CURRENT ITERATE is "
                "rejected. Signal: building res from uh (a discrete "
                "function) instead of TrialFunction(W) leaves the form "
                "with only ONE argument, and galerkin([res == 0, bc]) "
                "raises \"ValueError: Integrands model requires form "
                "with at least two arguments.\" from "
                "dune/models/integrands/load.py. dune-fem wants the "
                "residual in the trial function and differentiates it "
                "itself. (Executed 2026-08-03.)"
            ),
            (
                "[Numerical] Newton from a zero initial guess diverges "
                "as Re grows. Signal: info['converged'] comes back "
                "False, or the iteration count hits "
                "nonlinear.maxiterations, with the residual bouncing "
                "rather than shrinking. Measured 2026-08-03: at Re=40 "
                "on 8x8 and 16x16 grids Newton converged from a zero "
                "start in 5 and 4 iterations. For higher Re, either "
                "enable nonlinear.linesearch or use continuation — "
                "solve at a lower Re and interpolate that solution "
                "into the initial guess."
            ),
            (
                "[Numerical] Taylor-Hood enforces incompressibility "
                "only WEAKLY. Signal: ||div u||_L2 does not go to "
                "machine zero the way the Stokes-Poiseuille test does "
                "— on a convective test case it stays visibly above "
                "it, and falls by roughly a factor of four per mesh "
                "halving. That is the discretisation, not a "
                "bug; judge it by whether it FALLS under refinement. "
                "(Executed 2026-08-03.)"
            ),
            (
                "[Numerical] Convection dominates as Re rises and the "
                "Galerkin discretisation is not stabilised. Signal: at "
                "high Re the velocity develops mesh-scale oscillations "
                "upstream of boundary layers that do not shrink until "
                "the layer is resolved; add SUPG/GLS stabilisation or "
                "refine. (Audit 2026-06-02.)"
            ),
            (
                "[Performance] The Newton tangent of Navier-Stokes "
                "is neither symmetric nor definite, so judge the "
                "solver by iteration count, not by whether it "
                "finishes. Signal: read "
                "linear_iterations out of the dict that "
                "dune.fem.scheme.galerkin's solve returns. On the "
                "12x12 composite Taylor-Hood Stokes matrix, measured "
                "2026-08-03, it came back 1 for "
                "solver=('suitesparse','umfpack') at 9.465e-16 "
                "relative error, against 1343 for 'cg' at 3.837e-13, "
                "94026 for 'gmres' and 43953 for 'bicgstab'. The "
                "Krylov methods DO converge — the cost, three to five "
                "orders of magnitude more iterations, is the signal, "
                "not a failure. Use solver=('suitesparse','umfpack') "
                "for 2D problems of this size. (RE-MEASURED by "
                "adversarial audit 2026-08-03; an earlier revision "
                "recorded 1150 / 2527 / 37167, which did not "
                "reproduce.)"
            ),
        ],
    },
    "helmholtz": {
        "description": (
            "Helmholtz equation -Δu - k²u = f. "
            "Verify with a manufactured solution: pick a smooth u that satisfies "
            "your boundary conditions, substitute it to obtain f, and check that "
            "the L2 error falls at the element's theoretical order. Do not reuse "
            "a solution written down here — the field must be yours, or the "
            "convergence study measures nothing about your discretisation."
        ),
        "solver": "galerkin with GMRES (system is indefinite for k² > smallest eigenvalue)",
        "spaces": "lagrange(gridView, order=2) — higher order for better phase accuracy",
        "pitfalls": [
            (
                "[Numerical] Helmholtz system is INDEFINITE "
                "(not SPD), so prefer GMRES or a direct "
                "solver — but do NOT expect to be told. "
                "Signal: dune-fem gives you NO diagnostic. "
                "Measured 2026-08-03 on -Laplace(u) - k^2 u "
                "= f at k=10 on a 16x16 P1 grid, "
                "solver='cg': no exception was raised and "
                "info came back {'converged': True, "
                "'iterations': 0, 'linear_iterations': 1}. "
                "The only detector is the ANSWER — check it "
                "against a manufactured solution, or "
                "compare cg against a direct solve. "
                "CORRECTED by adversarial audit 2026-08-03: "
                "an earlier revision of this Signal claimed "
                "CG returns 'matrix not positive definite', "
                "a string that occurs NOWHERE under "
                "site-packages/dune or include/dune and "
                "that this install does not emit. "
                "(Audit 2026-06-02; Signal re-measured "
                "2026-08-03.)"
            ),
            (
                "[Numerical] Pollution effect: higher order "
                "reduces phase error — use P3+ or DG. "
                "Signal: P1 at k = 40 with 10 elem/"
                "wavelength shows phase drift of ~ 1/4 "
                "wavelength across the domain; P3 on "
                "the same mesh recovers phase to < "
                "0.1%. Pollution scales as O(k^3 h^2) "
                "for P1, O(k^5 h^4) for P2. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Input] For SCATTERING: add absorbing BC "
                "(Robin / PML) on truncated domain. "
                "Signal: in dune.fem galerkin, a pure-"
                "DirichletBC outer boundary reflects the "
                "outgoing wave — visible standing-wave "
                "interference in |u| (the GridFunction "
                "magnitude) with lambda/2 spacing. Add "
                "(i*k*u, v) * ds in the ufl Form on the "
                "truncation boundary or a PML layer with "
                "complex stretch. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] k^2 < pi^2: system positive "
                "definite; k^2 > pi^2: indefinite, "
                "precondition carefully. Signal: in the "
                "dune.fem galerkin scheme a small-k case "
                "(k=2, pi^2=9.87) converges with any "
                "SolverCG / SolverGMRES; large-k (k=10, "
                "k^2=100) needs shifted-Laplacian "
                "preconditioner or sweeping preconditioner "
                "— vanilla ILU stalls. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Rule of thumb: at least 10 "
                "P1 elements per wavelength for ~1% "
                "phase error. Signal: < 5 elem/lambda "
                "gives 10-30% phase error; check by "
                "comparing computed |u| at distance L "
                "with the analytic plane wave. Increase "
                "h or polynomial order to meet the "
                "rule of thumb. (Audit 2026-06-02.)"
            ),
        ],
    },
    "time_dependent_heat": {
        "description": (
            "Transient heat equation du/dt - alpha*Δu = f via implicit Euler time-stepping. "
            "Gaussian initial pulse diffusing over time."
        ),
        "solver": "Backward Euler (A-stable): reassemble RHS each step, solve with CG",
        "spaces": "lagrange(gridView, order=1) — sufficient for diffusion problems",
        "time_stepping": {
            "backward_euler": "1st order, A-stable, unconditionally stable",
            "crank_nicolson": "2nd order, A-stable, better accuracy",
        },
        "required_vs_optional": {
            "REQUIRED": [
                "the mass term u*v/dt on the LEFT and u_n*v/dt on the "
                "RIGHT — without it the 'transient' run just re-solves "
                "the steady problem every step",
                "ONE scheme built OUTSIDE the time loop",
            ],
            "OPTIONAL": [
                "an output function distinct from the one the "
                "right-hand side reads, copied back after each step. "
                "This template solves INTO u_n, and for this LINEAR "
                "scheme dune-fem 2.12.0.2 reads the right-hand side "
                "before it overwrites the target, so that is the same "
                "step (measured 2026-10-01: 2.1e-8 apart after the "
                "template's 50 steps, max value 0.4995, because the "
                "solve first sets the target's boundary dofs to the "
                "Dirichlet value (femscheme.hh _solve: "
                "setConstraints(solution)) and only then reads the "
                "right-hand side, and the Gaussian start is 3.7e-6 "
                "there, not 0; 1.1e-15 apart with a "
                "start that is 0 on the boundary and linear.tolerance "
                "1e-14). It is NOT optional when the Dirichlet data "
                "change in time (g = t: 1.1e-2 apart after 5 steps) or "
                "when the form is nonlinear (a (1+u^2) diffusion step "
                "with nonlinear.maxiterations=50: the aliased solve "
                "returned converged=False after 50 iterations, the "
                "separate-target solve converged in 3)",
                "dune.ufl.Constant for dt and for time-dependent "
                "coefficients — assigning .value avoids a JIT rebuild, "
                "while changing a float literal inside the form forces "
                "one",
                "gridView.writeVTK(name, ..., number=step) for a time "
                "series (the kwarg exists; a series was NOT exercised "
                "here)",
            ],
            "NOT AVAILABLE": [
                "DIRK23 / SDIRK22 / Heun / SSP-RK as ready-made "
                "steppers callable from Python. Older catalog text "
                "listed them. dune-fem 2.12.0.2 implements its "
                "Runge-Kutta ODE solvers only in C++ "
                "(dune/fem/solver/rungekutta: DIRK23 with 2 stages and "
                "order 3, DIRK34, the explicit tvd2 table (Heun) and "
                "others; SDIRK22 is "
                "accepted by the name selector but has no Butcher "
                "table, so selecting it throws NotImplemented) and has "
                "no Python binding for them. The Python wrappers (the "
                "rk module of dune.femdg with Heun/ssp2/ssp3, "
                "femdgStepper, rungeKuttaSolver) are in dune-fem-dg, "
                "which is NOT importable from a plain dune-fem install "
                "(executed 2026-08-03, re-checked 2026-10-01). "
                "Implement the stepper yourself — SSP-RK2 is four "
                "lines and there is a working one in the dg_advection "
                "template.",
            ],
        },
        "verification_you_can_run": (
            "Without a source and with zero Dirichlet data the "
            "solution must decay monotonically towards zero and never "
            "change sign, for ANY dt, because implicit Euler is "
            "unconditionally stable and the heat operator is "
            "dissipative. If the answer does not change when you "
            "change dt, the mass term is missing; if it oscillates in "
            "sign, the time term has the wrong sign. For a "
            "second-order check, halve dt and confirm the change "
            "between successive dt levels shrinks; implicit Euler is "
            "first order in time, so it shrinks by about half, while "
            "correctly written Crank-Nicolson shrinks by about four."),
        "pitfalls": [
            (
                "[API] Backward Euler: REASSEMBLE "
                "b = (u_n/dt)*v*dx every step (u_n "
                "changes). Signal: assembling b once "
                "outside the loop gives a heat solution "
                "stuck at the INITIAL condition — the "
                "RHS never updates. Inside the time "
                "loop: compute new b from u_old. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Performance] The scheme OBJECT can be "
                "REUSED across steps — only RHS changes "
                "if BCs and time step are constant. "
                "Signal: re-creating the scheme each "
                "step rebuilds the operator (re-JIT for "
                "any UFL changes) — 10-100x slower than "
                "reusing the scheme and just updating "
                "RHS. (Audit 2026-06-02.)"
            ),
            (
                "[Performance] Mass matrix stays the SAME "
                "if no moving mesh — cache if "
                "performance matters. Signal: re-"
                "assembling M every step wastes 50% "
                "of wall-clock; cache M outside the "
                "loop and reuse. The galerkin scheme "
                "does this automatically if the form "
                "structure is constant. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] For CRANK-NICOLSON: a = "
                "(u/dt + alpha/2*Δu)*v*dx; b involves "
                "u_n terms with the matching alpha/2 "
                "factor. Signal: writing the dune.fem "
                "galerkin Crank_Nicolson scheme with the "
                "ImplicitEuler form (alpha instead of "
                "alpha/2) gives a first-order scheme "
                "instead of second-order — the GridFunction "
                "L2 error decays as O(dt) not O(dt^2) under "
                "refinement. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] CFL is NOT required for "
                "IMPLICIT methods — choose dt for "
                "accuracy, not stability. Signal: BE "
                "is unconditionally stable; choosing "
                "a too-small explicit-CFL dt for an "
                "implicit run wastes compute. Pick dt "
                "by the shortest physical timescale of "
                "interest, not by lambda_max / 2. "
                "(Audit 2026-06-02.)"
            ),
        ],
    },
    "mixed_methods": {
        "description": (
            "Mixed (dual) Poisson / Darcy: solve for the FLUX and the "
            "potential simultaneously on one composite space. "
            "READ THE FIRST PITFALL BEFORE WRITING ANY CODE — the "
            "textbook Raviart-Thomas pair cannot be assembled on "
            "dune-fem 2.12.0.2."),

        "required_calls_in_order": [
            "S = dune.fem.space.lagrange(gridView, dimRange=2, order=2)"
            "   # flux",
            "V = dune.fem.space.lagrange(gridView, order=1)"
            "   # potential",
            "W = dune.fem.space.composite(S, V)",
            "trial = ufl.TrialFunction(W); "
            "sigma = ufl.as_vector([trial[0], trial[1]]); u = trial[2]",
            "a = (inner(sigma,tau) - u*div(tau) + div(sigma)*v)*dx",
            "L = -u_D*dot(tau, n)*ds"
            "   # the POTENTIAL is the natural datum in the mixed form",
            "bc = dune.ufl.DirichletBC(W, [None, 0, None], <indicator>)"
            "   # the NORMAL FLUX is the essential datum",
            "scheme = galerkin([a == L, bc], "
            "solver=('suitesparse','umfpack'))",
        ],
        "required_vs_optional": {
            "REQUIRED": [
                "an inf-sup stable pair — flux space one order higher "
                "than the potential space, exactly as for Stokes",
                "a direct solver; the mixed system is indefinite",
                "the boundary term -u_D*dot(tau, n)*ds — in the mixed "
                "form the Dirichlet datum for u enters WEAKLY and "
                "there is no DirichletBC for it",
                "None entries in the DirichletBC value list for every "
                "component you are not constraining",
            ],
            "OPTIONAL": [
                "components=['flux','potential'] for VTK names",
                "a permeability tensor K: replace inner(sigma,tau) by "
                "inner(dot(inv(K), sigma), tau)",
            ],
        },
        "verification_you_can_run": (
            "Take u = 1 - x, so sigma = -grad(u) = (1,0) and f = "
            "div(sigma) = 0. Prescribe u_D = 1 - x weakly on the whole "
            "boundary and sigma_y = 0 on y=0 and y=1. Both fields are "
            "inside the discrete spaces, so a correct run reproduces "
            "them to solver tolerance. Executed 2026-08-03, 8x8 "
            "structuredGrid, [P2]^2 flux + P1 potential, "
            "('suitesparse','umfpack'): relative potential error "
            "2.5e-16, flux error 3.1e-15, ||div sigma_h|| 1.5e-13. The "
            "same problem with solver='gmres' also converged but only "
            "to 1.4e-12 / 6.8e-11 / 3.5e-09."),

        "raviart_thomas_status_on_this_install": (
            "raviartThomas WORKS ON ITS OWN and is genuinely H(div): "
            "raviartThomas(gridView, order=0) on an 8x8 structuredGrid "
            "built fine (size 144, localBlockSize 1), and interpolating "
            "an analytically divergence-free field into it gave "
            "||div sigma_h||_L2 = 4.4e-16 — machine zero, which is the "
            "defining commuting-diagram property of the element. Its "
            "L2 interpolation error on that field was 1.1e-01 at 8x8, "
            "which is the expected first-order accuracy of RT0, not a "
            "bug. What does NOT work is putting it inside a tuple "
            "space; see the pitfall. Executed 2026-08-03."),

        "pitfalls": [
            (
                "[API] raviartThomas cannot be a leg of "
                "product()/composite() on dune-fem 2.12.0.2 — the "
                "space fails to BUILD, in C++, before any solve. "
                "Signal: dune.generator.exceptions.CompileError while "
                "compiling the generated femspace module, whose "
                "decisive line is \"cannot convert 'localDofVector' "
                "(type 'Dune::Fem::SubVector<Dune::Fem::"
                "LocalContribution<Dune::Fem::TupleDiscreteFunction<"
                "... RaviartThomasLocalFiniteElementMap ...>>>') to "
                "type 'std::vector<double>&'\" at "
                "dune/fem/space/localfiniteelement/interpolation.hh:"
                "179. product() and composite() fail identically, and "
                "the failure costs the full C++ build time first, so "
                "it looks like a hang. The same RT space built ALONE "
                "works. Use an inf-sup stable Lagrange pair for the "
                "mixed system, or keep RT for interpolation/projection "
                "only. (Executed 2026-08-03 on dune-fem 2.12.0.2, both "
                "factories, and reproduced from the generator "
                "template.)"
            ),
            (
                "[API] The Python factory is raviartThomas (camelCase). "
                "Signal: importing dune.fem.space.raviartthomas "
                "raises ImportError and "
                "hasattr(dune.fem.space,'raviartthomas') is False, "
                "even though the C++ header really is called "
                "raviartthomas.hh — which is where the confusion comes "
                "from. Same for the sibling H(div) families bdm and "
                "bdfm, which ARE lowercase. (Executed 2026-08-03.)"
            ),
            (
                "[API] ufl.TrialFunctions(W) does NOT unpack a "
                "dune-fem composite space. Signal: it returns a "
                "1-TUPLE holding one argument of shape (dim_total,), "
                "so the unpacking (sigma, u) = TrialFunctions(W) "
                "binds sigma to "
                "the whole vector. Use TrialFunction(W) and slice: "
                "sigma = as_vector([t[0], t[1]]); u = t[2]. (Executed "
                "2026-08-03 — this corrects an earlier catalog entry "
                "that claimed the unpacking works.)"
            ),
            (
                "[Input] In the mixed form the boundary roles SWAP. "
                "Signal: the potential u is the NATURAL datum and "
                "enters through the boundary integral "
                "-u_D*dot(tau,n)*ds; the normal flux sigma.n is the "
                "ESSENTIAL datum and needs a DirichletBC. Writing a "
                "DirichletBC on u instead constrains an L2-type field "
                "and gives a solution that ignores your boundary data "
                "while still converging. (Audit 2026-06-02; the "
                "working sign convention is in the executed template.)"
            ),
            (
                "[API] The multi-field factory for this monolithic "
                "system is dune.fem.space.composite(S, V) — "
                "there is NO dune.fem.space.product_space, the name is "
                "ABSENT / FALSIFIED (executed 2026-08-03: hasattr("
                "dune.fem.space,'product_space') is False). Signal: "
                "importing product_space "
                "raises ImportError; the name only ever existed as a "
                "local alias in this catalog's own template. "
                "dune.fem.space.product(S, V) is NOT the same: it has "
                "the same C++ type, dimRange and size as composite, "
                "but it is dune-fem's space for a DECOUPLED solve, has "
                "no storage, and galerkin() on it raises TypeError at "
                "scheme construction (measured 2026-10-01 with this "
                "template's form: composite solved, product raised)."
            ),
            (
                "[Numerical] Equal-order flux and potential spaces "
                "violate inf-sup exactly as in Stokes. Signal: "
                "dune.fem.space.composite(lagrange(gridView, "
                "dimRange=2, order=1), lagrange(gridView, order=1)) "
                "assembles and solves, and the potential comes back "
                "with a checkerboard mode whose amplitude does NOT "
                "shrink under refinement while the flux still looks "
                "fine. Give the lagrange flux space order = potential "
                "order + 1. (Audit 2026-06-02.)"
            ),
            (
                "[Performance] Saddle-point systems need a direct "
                "solver or a block preconditioner. Signal: plain GMRES "
                "on the raw indefinite matrix converges, slowly, and "
                "to a much looser tolerance than the direct solver on "
                "the same problem (measured 1.4e-12 vs 2.5e-16 on an "
                "8x8 grid). Direct UMFPACK is the right default below "
                "~1e5 dofs. (Executed 2026-08-03.)"
            ),
        ],
    },
}


# ---------------------------------------------------------------------------
# GENERATORS registry
# ---------------------------------------------------------------------------

GENERATORS = {
    "maxwell_2d":            _maxwell_2d,
    "eigenvalue_2d":         _eigenvalue_2d,
    "hyperelasticity_2d":    _hyperelasticity_2d,
    "navier_stokes_2d":      _navier_stokes_2d,
    "helmholtz_2d":          _helmholtz_2d,
    "time_dependent_heat_2d": _time_dependent_heat_2d,
    "mixed_methods_2d":      _mixed_methods_2d,
}
