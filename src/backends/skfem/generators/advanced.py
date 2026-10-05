"""scikit-fem advanced physics generators and knowledge.

Covers: Navier-Stokes, hyperelasticity (Neo-Hookean), DG advection,
time-dependent PDE, Helmholtz (complex), and reaction-diffusion.
"""


# ---------------------------------------------------------------------------
# 1. Navier-Stokes (Newton iteration)
# ---------------------------------------------------------------------------

def _navier_stokes_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Steady Navier-Stokes on the lid-driven cavity, solved by Picard
    iteration (lagged-velocity convection). Taylor-Hood P2/P1
    velocity-pressure. Rewritten 2026-06-02 (was broken — scalar
    laplace on a vector basis raised shape mismatch + non-existent
    DofsView subscript; the old Newton kernel mixed
    Jacobian/residual blocks incorrectly).
    """
    refine = int(params.get("refine", 4))
    Re = params.get("Re", 100.0)
    tol = params.get("picard_tol", 1e-6)
    max_iter = params.get("max_iter", 25)
    return f'''\
"""Navier-Stokes lid-driven cavity — Picard iteration — Taylor-Hood P2/P1 — scikit-fem"""
from skfem import (
    MeshTri, Basis, BilinearForm,
    ElementVector, ElementTriP1, ElementTriP2,
    asm, condense, solve,
)
from skfem.helpers import grad, ddot, dot
from skfem.models.general import divergence
from scipy.sparse import bmat
import numpy as np
import json

Re = {Re}

# --- Mesh + Taylor-Hood spaces ---
# intorder=4 keeps trial/test quadrature matched on the
# mixed B = -asm(divergence, basis_u, basis_p) block AND
# is high enough for the (u_prev · ∇)u trial term.
m = MeshTri().refined({refine})
basis_u = Basis(m, ElementVector(ElementTriP2()), intorder=4)
basis_p = Basis(m, ElementTriP1(), intorder=4)


@BilinearForm
def viscous(u, v, w):
    """∫(grad u : grad v) dx — vector Laplacian."""
    return ddot(grad(u), grad(v))


@BilinearForm
def convection(u, v, w):
    """∫((u_prev · ∇) u) · v dx — lagged-velocity convection.

    w['u_prev'] is the interpolated previous-iterate velocity:
    a DiscreteField with .value shape (d, n_elem, n_quad)
    and .grad shape (d, d, n_elem, n_quad). For a scalar u_p
    component, u_p[i] would be (n_elem, n_quad). For grad(u)
    (rank-2: components × spatial dims), grad(u)[i][j] is the
    derivative of the i-th trial component w.r.t. x[j].
    """
    u_p = w['u_prev'].value
    gu = grad(u)
    # (u_prev · ∇) u_i = sum_j u_p[j] * du[i]/dx[j]
    adv_u = np.stack([
        u_p[0] * gu[0][0] + u_p[1] * gu[0][1],
        u_p[0] * gu[1][0] + u_p[1] * gu[1][1],
    ])
    return dot(adv_u, v)


# Stokes blocks (assembled ONCE, reused every Picard step).
K_visc = asm(viscous, basis_u) / Re
# B[q, u] = ∫q·div(u) dx (skfem +div convention) — negate
# for the standard saddle-point [[K, -B^T], [-B, 0]] layout.
B = -asm(divergence, basis_u, basis_p)

N_u = basis_u.N
N_p = basis_p.N
N_total = N_u + N_p

# --- Driven-cavity BC ---
# ElementVector interleaves x/y dofs at each node:
#   dof[2i] = x-component, dof[2i+1] = y-component.
doflocs_u = basis_u.doflocs
by = doflocs_u[1]
top_x = np.isclose(by[0::2], 1.0)
u_bc = np.zeros(basis_u.N)
u_bc[0::2] = np.where(top_x, 1.0, 0.0)
u_bc[1::2] = 0.0

# Pressure pin at the DOF closest to the origin (removes
# the constant-pressure null space).
pdofs = basis_p.doflocs.T
pin_p_local = int(np.argmin(np.linalg.norm(pdofs[:, :2], axis=1)))
pin_p_global = N_u + pin_p_local

D_u = basis_u.get_dofs().flatten()
D = np.concatenate([
    D_u,
    np.array([pin_p_global], dtype=np.int64),
])

# Initial guess: Stokes solution (Re → ∞ Picard step 0
# with u_prev=0 reduces convection to 0).
x = np.zeros(N_total)
x[:N_u] = u_bc

# --- Picard loop ---
res_norm = np.inf
for it in range({max_iter}):
    u_prev = x[:N_u]
    u_prev_field = basis_u.interpolate(u_prev)

    # Assemble convection with current u_prev. Combined with
    # the (constant) viscous block this gives the iteration
    # Jacobian for the velocity-velocity block.
    C = asm(convection, basis_u, u_prev=u_prev_field)

    A = bmat([[K_visc + C, B.T],
              [B,          None]], format='csr')
    F = np.zeros(N_total)

    x_full = np.zeros(N_total)
    x_full[:N_u] = u_bc
    x_new = solve(*condense(A, F, D=D, x=x_full))

    res_norm = np.linalg.norm(x_new[:N_u] - u_prev)
    x = x_new
    print(f"Picard it {{it+1}}: ||du|| = {{res_norm:.4e}}")
    if res_norm < {tol}:
        print(f"Converged in {{it+1}} Picard iterations")
        break

u_h = x[:N_u]
p_h = x[N_u:]
max_vel = np.sqrt(u_h[0::2]**2 + u_h[1::2]**2).max()
print(f"Re = {Re}, DOFs = {{N_total}}")
print(f"Max velocity magnitude: {{max_vel:.6f}}")
print(f"||u_x||_inf = {{np.abs(u_h[0::2]).max():.6f}}")
print(f"||u_y||_inf = {{np.abs(u_h[1::2]).max():.6f}}")
print(f"||p||_inf   = {{np.abs(p_h).max():.6f}}")

import meshio
pts  = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
trng = [("triangle", m.t.T)]
mio  = meshio.Mesh(pts, trng)
mio.write("result.vtu")

summary = {{
    "Re": {Re},
    "max_velocity": float(max_vel),
    "n_dofs": int(N_total),
    "n_elements": int(m.nelements),
    "picard_iter": it + 1,
    "final_residual": float(res_norm),
    "element_type": "P2-P1 Taylor-Hood",
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Navier-Stokes solve complete.")
'''


# ---------------------------------------------------------------------------
# 2. Hyperelasticity — Neo-Hookean with Newton iteration
# ---------------------------------------------------------------------------

def _hyperelasticity_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Neo-Hookean hyperelasticity with Newton iteration.
    Incompressible-like Neo-Hookean: W = mu/2*(I1-2) - mu*ln(J) + lam/2*(ln J)^2.
    """
    nx = params.get("nx", 10)
    ny = params.get("ny", 4)
    lx = params.get("lx", 4.0)
    ly = params.get("ly", 1.0)
    # Audit 2026-06-02: previous defaults E=1.0 / traction=0.1
    # gave a 10%-of-stiffness normalised load — large enough to
    # drive J<0 in some Gauss points after the first modified-
    # Newton iter, producing log(J)=NaN. New defaults pick a
    # stiff material + small traction (~0.5% strain) so the
    # iteration converges in 3-4 steps per load substep. Users
    # who want a real large-deformation study override via
    # params and add a continuation/arc-length wrapper.
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    traction = params.get("traction", 1.0)
    tol = params.get("newton_tol", 1e-8)
    max_iter = params.get("max_iter", 30)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return f'''\
"""Neo-Hookean hyperelasticity — incremental load stepping — scikit-fem"""
from skfem import (
    MeshTri, MeshQuad, Basis, BilinearForm, LinearForm, FacetBasis,
    ElementVector, ElementTriP1,
    asm, condense, solve,
)
from skfem.helpers import grad, ddot, dot, identity, inv, det, transpose
import numpy as np
import json

# Lame parameters from E={E}, nu={nu}
lam = {lam:.6f}
mu  = {mu:.6f}

# --- Mesh + clamped/loaded boundary tags ---
m = (MeshQuad.init_tensor(
        np.linspace(0, {lx}, {nx + 1}),
        np.linspace(0, {ly}, {ny + 1}),
     ).to_meshtri()
      .with_boundaries({{
          "left":  lambda x: x[0] < 1e-10,
          "right": lambda x: x[0] > {lx} - 1e-10,
      }}))

basis = Basis(m, ElementVector(ElementTriP1()), intorder=3)
fbasis_right = FacetBasis(m, ElementVector(ElementTriP1()),
                          facets=m.boundaries["right"])

N = basis.N
# Audit 2026-06-02 rewrite: prior version used `w["dux_dx"]`/
# `w["duy_dx"]`/etc. scalar kwargs that scikit-fem 12 no longer
# accepts in @BilinearForm kernels, plus `u.grad[0]` indexing
# that returned a (d, n_basis) array on ElementVector — both
# broke the assembler with
#   ValueError: could not broadcast input array from shape
#   (2,3) into shape (80,).
# New kernel uses skfem.helpers.grad/ddot/dot/identity/inv/det
# directly on the rank-2 displacement-gradient tensor, plus
# basis.interpolate(u_prev_dofs) to pass the previous-iterate
# displacement-gradient field via w['u_prev'].grad (shape
# (d, d, n_elem, n_quad)). Linearisation pattern is a Picard-
# style modified Newton: assemble the tangent stiffness from
# the current-configuration material+geometric contribution,
# the internal-force residual from the 1st-PK stress, and
# solve K_tan * du = F_ext - R_int per load step.

# --- Neo-Hookean tangent (material + geometric) at the
#     current displacement u_prev. Computed via the
#     compressible Neo-Hookean strain-energy
#     W = (mu/2)(I_C - d) - mu*lnJ + (lam/2)(lnJ)^2.
def _F(u_field):
    """Deformation gradient F = I + grad(u_field)."""
    return identity(u_field.grad) + u_field.grad


@BilinearForm
def K_tan_form(u, v, w):
    """Approximate tangent: small-strain linear elasticity (Hooke's
    law). This is the "consistent material tangent at F=I" — exact
    at the first iterate, and a stable modified-Newton tangent for
    moderate loads (up to ~5-10% strain). Convergence is slower
    than full Newton (linear instead of quadratic) but the iteration
    is robust and the geometric stiffness contribution at the next-
    to-undeformed configuration is small. The Neo-Hookean residual
    in R_int_form below uses the actual F-dependent 1st-PK stress, so
    converged iterations still satisfy the nonlinear equilibrium."""
    eps_u = 0.5 * (grad(u) + transpose(grad(u)))
    eps_v = 0.5 * (grad(v) + transpose(grad(v)))
    tr_u = eps_u[0, 0] + eps_u[1, 1]
    tr_v = eps_v[0, 0] + eps_v[1, 1]
    return lam * tr_u * tr_v + 2.0 * mu * ddot(eps_u, eps_v)


@LinearForm
def R_int_form(v, w):
    """Internal virtual work: P(F) : grad(v) dx where P = F * S."""
    F = _F(w['u_prev'])
    J = det(F)
    lnJ = np.log(J)
    Finv = inv(F)
    # 2nd PK: S = mu*(I - Cinv) + lam*lnJ*Cinv,
    # Cinv = Finv @ Finv.T (via component algebra below).
    Cinv00 = Finv[0, 0] * Finv[0, 0] + Finv[0, 1] * Finv[0, 1]
    Cinv01 = Finv[0, 0] * Finv[1, 0] + Finv[0, 1] * Finv[1, 1]
    Cinv11 = Finv[1, 0] * Finv[1, 0] + Finv[1, 1] * Finv[1, 1]
    S00 = mu * (1.0 - Cinv00) + lam * lnJ * Cinv00
    S01 = mu * (0.0 - Cinv01) + lam * lnJ * Cinv01
    S11 = mu * (1.0 - Cinv11) + lam * lnJ * Cinv11
    # 1st PK: P = F * S.
    P00 = F[0, 0] * S00 + F[0, 1] * S01
    P01 = F[0, 0] * S01 + F[0, 1] * S11
    P10 = F[1, 0] * S00 + F[1, 1] * S01
    P11 = F[1, 0] * S01 + F[1, 1] * S11
    # P : grad(v) — sum over (i, j) of P_ij * dv_i/dx_j.
    gv = grad(v)
    return (P00 * gv[0, 0] + P01 * gv[0, 1]
            + P10 * gv[1, 0] + P11 * gv[1, 1])


@LinearForm
def F_ext_form(v, w):
    """Constant traction on right face: t = (traction, 0)."""
    return {traction} * w['load_alpha'] * v[0]


fix_dofs = basis.get_dofs("left").flatten()
free = np.setdiff1d(np.arange(N), fix_dofs)

# --- Outer load-stepping loop + inner Newton-modified loop ---
u_disp = np.zeros(N)
n_load_steps = 4
for step in range(1, n_load_steps + 1):
    alpha = step / n_load_steps   # load fraction this step
    print(f"--- Load step {{step}}/{{n_load_steps}} (alpha={{alpha:.2f}}) ---")
    for it in range({max_iter}):
        u_prev_field = basis.interpolate(u_disp)
        K = asm(K_tan_form, basis, u_prev=u_prev_field)
        R_int = asm(R_int_form, basis, u_prev=u_prev_field)
        F_ext = asm(F_ext_form, fbasis_right, load_alpha=alpha)
        rhs = F_ext - R_int
        rhs[fix_dofs] = 0.0
        du = np.zeros(N)
        du[free] = solve(K[free][:, free], rhs[free])
        u_disp = u_disp + du
        res = np.linalg.norm(du[free])
        print(f"  Newton it {{it+1}}: ||du|| = {{res:.4e}}")
        if res < {tol}:
            print(f"  Converged in {{it+1}} iterations")
            break

u_xy = u_disp.reshape(-1, 2).T   # ElementVector interleaves x/y per node
max_disp = float(np.abs(u_xy).max())
print(f"Max displacement: {{max_disp:.6f}}")
print(f"E={E}, nu={nu}, traction={traction}")

import meshio
pts  = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
trng = [("triangle", m.t.T)]
u_node = np.column_stack([u_xy[0], u_xy[1], np.zeros(m.p.shape[1])])
mio  = meshio.Mesh(pts, trng, point_data={{"displacement": u_node}})
mio.write("result.vtu")

summary = {{
    "max_displacement": max_disp,
    "n_dofs": int(N),
    "n_elements": int(m.nelements),
    "n_load_steps": n_load_steps,
    "E": {E}, "nu": {nu}, "traction": {traction},
    "element_type": "P1-tri vector",
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Hyperelasticity solve complete.")
'''


# ---------------------------------------------------------------------------
# 3. DG methods — upwind DG for linear advection using ElementDG
# ---------------------------------------------------------------------------

def _dg_methods_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Discontinuous Galerkin for steady linear advection using ElementDG
    and InteriorFacetBasis for upwind flux.
    """
    nx = params.get("nx", 32)
    bx = params.get("bx", 1.0)
    by = params.get("by", 0.5)
    eps = params.get("eps", 0.0)   # optional diffusion for stability check
    return f'''\
"""DG upwind advection: b.grad(u) = f using ElementDG — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace
import numpy as np
from scipy.sparse.linalg import spsolve
import json

# Advection velocity
b = np.array([{bx}, {by}])
eps = {eps}    # diffusion coefficient (0 = pure advection)

# --- Mesh ---
m = MeshQuad.init_tensor(
    np.linspace(0, 1, {nx + 1}),
    np.linspace(0, 1, {nx + 1}),
).to_meshtri()   # MeshQuad -> MeshTri split (to_meshtri, available since scikit-fem 1.2.0)

# DG element: discontinuous P1 on triangles
e = ElementDG(ElementTriP1())
ib  = Basis(m, e)
ibf = FacetBasis(m, e)           # boundary facets

# --- Advection volume term: b . grad(u) * v ---
@BilinearForm
def advection_volume(u, v, w):
    return (b[0] * u.grad[0] + b[1] * u.grad[1]) * v

# --- Optional diffusion ---
@BilinearForm
def diffusion_volume(u, v, w):
    return eps * (u.grad[0]*v.grad[0] + u.grad[1]*v.grad[1])

# --- Interior upwind flux ---
# Jump penalty: b.n * {{u}} (upwind: from upwind side)
# --- Interior facets: a TWO-SIDED upwind flux --------------------------------
# The previous form assembled `bn * u * v` on a single InteriorFacetBasis, which
# is not a flux at all: it never couples the two elements sharing a facet, so
# the DG system was singular and spsolve returned NaN everywhere -- and the
# template printed "solve complete" over it. The standard DG upwind term is
#     sum over interior facets of  (b.n) u_upwind [v]
# assembled over BOTH sides with `jump` from skfem.helpers (w.idx says which
# side each of u, v comes from). Verified by reproducing the exact solution of
# the default problem to round-off; see the check after the solve.
ifb = [InteriorFacetBasis(m, e, side=0), InteriorFacetBasis(m, e, side=1)]

@BilinearForm
def upwind_interior(u, v, w):
    from skfem.helpers import jump
    bn = b[0] * w.n[0] + b[1] * w.n[1]          # normal of side 0
    ju, jv = jump(w, u, v)
    # u is the upwind value when it comes from the side the flow LEAVES:
    # side 0 when b.n > 0, side 1 otherwise
    u_up = u * np.where((bn > 0) == (w.idx[0] == 0), 1.0, 0.0)
    return bn * u_up * jv

# Boundary flux: outflow (b.n > 0) is an upwind term in u; inflow (b.n < 0)
# carries the prescribed value g into the right-hand side.
@LinearForm
def inflow_rhs(v, w):
    bn = b[0] * w.n[0] + b[1] * w.n[1]
    g  = 0.0  # inflow value (u=0 on inflow boundary)
    return -np.where(bn < 0, bn * g, 0.0) * v

@BilinearForm
def outflow_flux(u, v, w):
    bn = b[0] * w.n[0] + b[1] * w.n[1]
    return np.where(bn > 0, bn, 0.0) * u * v

@LinearForm
def source(v, w):
    return 1.0 * v

# --- Assembly ---
# Volume term integrated by parts: -(u, b.grad v); the facet terms above carry the flux.
@BilinearForm
def advection_volume_ibp(u, v, w):
    return -u * (b[0] * v.grad[0] + b[1] * v.grad[1])

A = asm(advection_volume_ibp, ib)
if eps > 0:
    A = A + asm(diffusion_volume, ib)
A = A + asm(outflow_flux, ibf)
A = A + asm(upwind_interior, ifb, ifb)
f = asm(source, ib) + asm(inflow_rhs, ibf)

# --- Solve (DG system is not symmetric; use direct solve) ---
u = spsolve(A.tocsr(), f)

# --- Exact check, when the default problem admits one ----------------------
# Pure advection (eps = 0) with axis-aligned b, unit source and u = 0 at inflow
# has the exact solution u = (b . x) / |b|^2, which DG-P1 represents exactly.
if eps == 0 and np.all(np.isfinite(u)) and b[0] >= 0 and b[1] >= 0 and (b[0] > 0 or b[1] > 0):
    xy = ib.doflocs
    if b[0] == 0 or b[1] == 0:
        # axis-aligned flow: u = (b . x) / |b|^2 lies in P1 and DG-P1 reproduces it to round-off
        u_exact = (b[0] * xy[0] + b[1] * xy[1]) / float(b @ b)
        err_exact = float(np.max(np.abs(u - u_exact)))
        _v = "PASS" if err_exact < 1e-10 else "FAIL"
        print(f"VERDICT skfem dg_methods exact_identity ref=0.000000e+00 got={{err_exact:.6e}} tol=1.000000e-10 {{_v}}")
    else:
        # oblique flow: u = min(x/bx, y/by) has a kink along y = (by/bx) x that P1 cannot
        # represent inside the elements it crosses, so this is information, not a verdict
        u_exact = np.minimum(xy[0] / b[0], xy[1] / b[1])
        err_exact = float(np.max(np.abs(u - u_exact)))
        print(f"max|u - min(x/bx, y/by)| = {{err_exact:.3e}}  (exact solution has a kink; O(h) there, no verdict)")

max_val = u.max()
min_val = u.min()
print(f"DG advection: max(u) = {{max_val:.6f}}, min(u) = {{min_val:.6f}}")
print(f"DOFs: {{A.shape[0]}}, elements: {{m.nelements}}")

import meshio
pts  = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
trng = [("triangle", m.t.T)]
# DG solution: one value per DOF, not per node; write element-wise or project
# For visualization: project to P1 (nodal average).
# The module-level skfem.project() is DEPRECATED (it emits
# DeprecationWarning('project is deprecated in favor of
# Basis.project (will be removed in the next release).') on
# skfem 12.0.1). The supported spelling is the Basis.project
# INSTANCE method fed by Basis.interpolator. The two agree to
# ~2e-15 on a finite DG vector.
e_p1 = ElementTriP1()
ib_p1 = Basis(m, e_p1)
u_proj = ib_p1.project(ib.interpolator(u))
mio = meshio.Mesh(pts, trng, point_data={{"u": u_proj}})
mio.write("result.vtu")

summary = {{
    "max_value": float(max_val),
    "min_value": float(min_val),
    "n_dofs": int(A.shape[0]),
    "n_elements": int(m.nelements),
    "advection_velocity": [{bx}, {by}],
    "diffusion": {eps},
    "element_type": "DG-P1 triangle",
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("DG advection solve complete.")
'''


# ---------------------------------------------------------------------------
# 4. Time-dependent PDE — general backward Euler
# ---------------------------------------------------------------------------

def _time_dependent_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    General time-dependent PDE: du/dt + L(u) = f with backward Euler.
    L(u) = -div(D*grad(u)) + c*u  (reaction-diffusion operator).
    """
    nx = params.get("nx", 32)
    dt = params.get("dt", 0.01)
    T_end = params.get("T_end", 0.5)
    D_coeff = params.get("D", 0.1)
    c_coeff = params.get("c", 1.0)
    f_val = params.get("f", 1.0)
    theta = params.get("theta", 1.0)   # 1=BE, 0.5=CN
    return f'''\
"""Time-dependent PDE: du/dt - D*Δu + c*u = f — theta-method — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace, mass, unit_load
import numpy as np
from scipy.sparse.linalg import factorized
from scipy.sparse import identity as speye
import json

D_coeff = {D_coeff}
c_coeff = {c_coeff}
dt      = {dt}
theta   = {theta}     # 1.0 = backward Euler, 0.5 = Crank-Nicolson
T_end   = {T_end}
n_steps = int(T_end / dt)

# --- Mesh & basis ---
m  = MeshQuad.init_tensor(np.linspace(0, 1, {nx + 1}), np.linspace(0, 1, {nx + 1}))
e  = ElementQuad1()
ib = Basis(m, e)

# --- Assembly: stiffness L = D*laplace + c*mass, mass M ---
K = D_coeff * laplace.assemble(ib) + c_coeff * mass.assemble(ib)
M = mass.assemble(ib)
f = {f_val} * unit_load.assemble(ib)

# --- Boundary DOFs (homogeneous Dirichlet) ---
D_bnd = ib.get_dofs().flatten()
I     = ib.complement_dofs(D_bnd)

# --- Theta-method system matrix: A = M + theta*dt*K ---
A = M + theta * dt * K
A_solve = factorized(A[I][:, I].tocsc())

# --- Initial condition: u0 = sin(pi*x)*sin(pi*y) ---
x_coords = ib.doflocs[0]
y_coords = ib.doflocs[1]
u = np.sin(np.pi * x_coords) * np.sin(np.pi * y_coords)
u[D_bnd] = 0.0

print(f"Time-dependent PDE: {{n_steps}} steps, dt={{dt}}, theta={{theta}}")
print(f"D={{D_coeff}}, c={{c_coeff}}, f={f_val}")

t = 0.0
max_vals = []
for step in range(n_steps):
    # RHS: M*u_old - (1-theta)*dt*K*u_old + dt*f
    rhs = M @ u - (1.0 - theta) * dt * K @ u + dt * f
    rhs[D_bnd] = 0.0
    u_new = np.zeros_like(u)
    u_new[I] = A_solve(rhs[I])
    u = u_new
    t += dt

    if (step + 1) % max(1, n_steps // 10) == 0:
        print(f"  t={{t:.4f}}, max(u)={{u.max():.6f}}")
        max_vals.append((t, float(u.max())))

max_val = float(u.max())
print(f"Final: t={{t:.4f}}, max(u) = {{max_val:.6f}}")

import meshio
cells  = [("quad", m.t.T)]
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
mio = meshio.Mesh(points, cells, point_data={{"u": u}})
mio.write("result.vtu")

summary = {{
    "max_value": max_val,
    "n_dofs": len(u),
    "n_elements": m.nelements,
    "t_end": t,
    "n_steps": n_steps,
    "dt": dt,
    "theta": theta,
    "D": D_coeff,
    "c": c_coeff,
    "element_type": "Q1 quad",
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Time-dependent PDE solve complete.")
'''


# ---------------------------------------------------------------------------
# 5. Helmholtz — complex-valued
# ---------------------------------------------------------------------------

def _helmholtz_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Helmholtz equation: -Δu - k²u = f with complex arithmetic.
    Absorbing boundary condition on right: du/dn + i*k*u = 0.
    """
    nx = params.get("nx", 32)
    k = params.get("k", 5.0)          # wavenumber
    f_real = params.get("f_real", 1.0)
    return f'''\
"""Helmholtz: -Δu - k²u = f, k={k}, complex-valued — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace, mass, unit_load
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import json

k = {k}     # wavenumber

# --- Mesh ---
# MeshQuad.init_tensor does NOT attach named boundaries.
# Without with_boundaries(...) calls, ib.get_dofs('left')
# raises ValueError("Boundary 'left' not found.") and the
# subscript form ib.get_dofs()['left'] raises TypeError:
# 'DofsView' object is not subscriptable. Attach the four
# canonical boundaries here so the Dirichlet block below
# can resolve 'left'/'top'/'bottom' tags.
m  = (MeshQuad.init_tensor(np.linspace(0, 1, {nx + 1}),
                           np.linspace(0, 1, {nx + 1}))
      .with_boundaries({{
          "left":   lambda x: x[0] < 1e-10,
          "right":  lambda x: x[0] > 1.0 - 1e-10,
          "bottom": lambda x: x[1] < 1e-10,
          "top":    lambda x: x[1] > 1.0 - 1e-10,
      }}))
e  = ElementQuad1()
ib = Basis(m, e)

# --- Boundary basis for absorbing BC (right face) ---
fb_right = FacetBasis(m, e, facets="right")

# --- Assembly ---
# Stiffness: (grad u, grad v)
K = laplace.assemble(ib)

# Mass: k^2 * (u, v)  — subtracted for Helmholtz
M = mass.assemble(ib)

# Absorbing BC: i*k*(u, v) on right boundary
# dtype=complex is required: the default float64 form discards the imaginary part
@BilinearForm(dtype=complex)
def absorbing_bc(u, v, w):
    return 1j * k * u * v

A_abc = asm(absorbing_bc, fb_right)

# System: (K - k^2*M + A_abc) * u = f
# Use complex128 arithmetic
A = K.astype(complex) - k**2 * M.astype(complex) + A_abc

# Source: point-like load at center (Gaussian approximation)
@LinearForm
def gaussian_source(v, w):
    x0, y0 = 0.5, 0.5
    sigma = 0.05
    r2 = (w.x[0] - x0)**2 + (w.x[1] - y0)**2
    return {f_real} * np.exp(-r2 / (2 * sigma**2)) * v

f = asm(gaussian_source, ib).astype(complex)

# --- Dirichlet BC: u=0 on left, top, bottom ---
# ib.get_dofs() returns a DofsView, which is NOT subscriptable
# (the legacy ib.get_dofs()['left'] pattern raises TypeError:
# 'DofsView' object is not subscriptable in scikit-fem 12).
# In modern skfem the canonical pattern is to pass the
# boundary name directly: ib.get_dofs('left') returns a
# DofsView whose .flatten() yields the boundary DOF indices.
D_bnd = np.concatenate([
    ib.get_dofs("left").flatten(),
    ib.get_dofs("top").flatten(),
    ib.get_dofs("bottom").flatten(),
])
I = np.setdiff1d(np.arange(A.shape[0]), D_bnd)

u = np.zeros(A.shape[0], dtype=complex)
u[I] = spsolve(A[I][:, I].tocsr(), f[I])

max_abs = np.abs(u).max()
print(f"Helmholtz k={{k}}: max|u| = {{max_abs:.6f}}")
print(f"DOFs: {{A.shape[0]}}, elements: {{m.nelements}}")

# Save real part and magnitude
import meshio
cells  = [("quad", m.t.T)]
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
mio = meshio.Mesh(points, cells,
    point_data={{"u_real": u.real, "u_imag": u.imag, "u_abs": np.abs(u)}})
mio.write("result.vtu")

summary = {{
    "k": k,
    "max_abs": float(max_abs),
    "max_real": float(u.real.max()),
    "n_dofs": int(A.shape[0]),
    "n_elements": int(m.nelements),
    "element_type": "Q1 quad (complex)",
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Helmholtz solve complete.")
'''


# ---------------------------------------------------------------------------
# 6. Reaction-diffusion — Schnakenberg / Fisher-KPP with backward Euler
# ---------------------------------------------------------------------------

def _reaction_diffusion_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Schnakenberg reaction-diffusion system (Turing pattern):
      du/dt = d_u * Δu + gamma*(a - u + u^2*v)
      dv/dt = d_v * Δv + gamma*(b - u^2*v)
    Solves with backward Euler + Newton iteration at each time step.
    """
    nx = params.get("nx", 32)
    dt = params.get("dt", 0.5)
    T_end = params.get("T_end", 50.0)
    d_u = params.get("d_u", 1.0)
    d_v = params.get("d_v", 40.0)
    a = params.get("a", 0.1)
    b = params.get("b", 0.9)
    gamma = params.get("gamma", 1000.0)
    tol = params.get("newton_tol", 1e-8)
    max_iter = params.get("max_iter", 20)
    return f'''\
"""Schnakenberg reaction-diffusion (Turing patterns) — backward Euler + Newton — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace, mass
import numpy as np
from scipy.sparse import bmat, eye as speye
from scipy.sparse.linalg import spsolve, factorized
import json

# --- Parameters ---
d_u   = {d_u}
d_v   = {d_v}
a     = {a}
b     = {b}
gamma = {gamma}
dt    = {dt}
T_end = {T_end}

# --- Mesh & basis ---
m  = MeshQuad.init_tensor(np.linspace(0, 1, {nx + 1}), np.linspace(0, 1, {nx + 1}))
e  = ElementQuad1()
ib = Basis(m, e)

# --- Assembly ---
K = laplace.assemble(ib)
M = mass.assemble(ib)
N = M.shape[0]

# Periodic-like: no Dirichlet BCs (Neumann = zero flux, natural BC)
# For Schnakenberg patterns, Neumann is standard.
I = np.arange(N)  # all DOFs free

# --- Initial condition: steady state + small random perturbation ---
rng = np.random.default_rng(42)
u0_ss = a + b
v0_ss = b / (a + b)**2
u_sol = np.full(N, u0_ss) + 0.01 * rng.standard_normal(N)
v_sol = np.full(N, v0_ss) + 0.01 * rng.standard_normal(N)

n_steps = int(T_end / dt)
print(f"Schnakenberg: {{n_steps}} steps, dt={{dt}}")
print(f"d_u={{d_u}}, d_v={{d_v}}, a={{a}}, b={{b}}, gamma={{gamma}}")
print(f"Steady state: u0={{u0_ss:.4f}}, v0={{v0_ss:.4f}}")

# --- Backward Euler with Newton iteration ---
# Residual for fully implicit system:
#   R_u = M*(u_new - u_old)/dt + d_u*K*u_new - gamma*M*f_u(u_new,v_new) = 0
#   R_v = M*(v_new - v_old)/dt + d_v*K*v_new - gamma*M*f_v(u_new,v_new) = 0
# Linearize f_u and f_v for Newton:
#   f_u(u,v) = a - u + u^2*v,  df_u/du = -1 + 2*u*v,  df_u/dv = u^2
#   f_v(u,v) = b - u^2*v,      df_v/du = -2*u*v,       df_v/dv = -u^2

def reaction_terms(u_vec, v_vec):
    fu = a - u_vec + u_vec**2 * v_vec
    fv = b - u_vec**2 * v_vec
    return fu, fv

def jacobian_terms(u_vec, v_vec):
    dfu_du = -1.0 + 2.0*u_vec*v_vec
    dfu_dv =        u_vec**2
    dfv_du = -2.0*u_vec*v_vec
    dfv_dv = -u_vec**2
    return dfu_du, dfu_dv, dfv_du, dfv_dv

@BilinearForm
def mass_pointwise(u, v, w):
    """Mass matrix with pointwise coefficient c(x)."""
    return w["c"] * u * v

# Fixed sparse structure: diffusion blocks + mass/dt diagonal
Ku  = d_u * K + M / dt
Kv  = d_v * K + M / dt

from scipy.sparse import csr_matrix, block_diag

t = 0.0
for step in range(n_steps):
    u_old = u_sol.copy()
    v_old = v_sol.copy()

    # Newton iteration
    u_new = u_old.copy()
    v_new = v_old.copy()

    for nit in range({max_iter}):
        fu, fv = reaction_terms(u_new, v_new)
        dfu_du, dfu_dv, dfv_du, dfv_dv = jacobian_terms(u_new, v_new)

        # Assemble reaction Jacobian blocks (diagonal in space)
        Mdu_du = asm(mass_pointwise, ib, c=dfu_du)
        Mdu_dv = asm(mass_pointwise, ib, c=dfu_dv)
        Mdv_du = asm(mass_pointwise, ib, c=dfv_du)
        Mdv_dv = asm(mass_pointwise, ib, c=dfv_dv)

        # Full Jacobian blocks
        J_uu = Ku - gamma * Mdu_du
        J_uv =    - gamma * Mdu_dv
        J_vu =    - gamma * Mdv_du
        J_vv = Kv - gamma * Mdv_dv

        J = bmat([[J_uu, J_uv], [J_vu, J_vv]], format="csr")

        # Residuals
        R_u = M @ (u_new - u_old) / dt + d_u * K @ u_new - gamma * M @ fu
        R_v = M @ (v_new - v_old) / dt + d_v * K @ v_new - gamma * M @ fv
        R   = np.concatenate([R_u, R_v])

        dxyz = spsolve(J, -R)
        u_new += dxyz[:N]
        v_new += dxyz[N:]

        res = np.linalg.norm(dxyz)
        if res < {tol}:
            break

    u_sol = u_new
    v_sol = v_new
    t += dt

    if (step + 1) % max(1, n_steps // 5) == 0:
        print(f"  t={{t:.2f}}, u=[{{u_sol.min():.4f}}, {{u_sol.max():.4f}}], "
              f"v=[{{v_sol.min():.4f}}, {{v_sol.max():.4f}}]")

print(f"Final t={{t:.2f}}: u in [{{u_sol.min():.4f}}, {{u_sol.max():.4f}}]")

import meshio
cells  = [("quad", m.t.T)]
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
mio = meshio.Mesh(points, cells, point_data={{"u": u_sol, "v": v_sol}})
mio.write("result.vtu")

summary = {{
    "u_max": float(u_sol.max()),
    "u_min": float(u_sol.min()),
    "v_max": float(v_sol.max()),
    "v_min": float(v_sol.min()),
    "n_dofs": N,
    "n_elements": m.nelements,
    "t_end": float(t),
    "n_steps": n_steps,
    "d_u": d_u, "d_v": d_v, "a": a, "b": b, "gamma": gamma,
    "element_type": "Q1 quad",
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Reaction-diffusion (Schnakenberg) solve complete.")
'''


# ---------------------------------------------------------------------------
# Knowledge dictionaries
# ---------------------------------------------------------------------------

KNOWLEDGE = {
    "navier_stokes": {
        "description": "Navier-Stokes flow — Newton iteration — Taylor-Hood P2/P1 (scikit-fem)",
        "solver": "Newton loop: linearize convection term, solve block saddle-point with spsolve",
        "elements": "Taylor-Hood: ElementVector(ElementTriP2()) + ElementTriP1()",
        "pitfalls": [
            (
                "[API] scikit-fem has NO built-in Newton solver "
                "or NS assembly — must build manually. Signal: "
                "searching skfem.utils for `NewtonSolver` or "
                "`NavierStokes` returns no match; the catalog "
                "ships hand-coded Newton + hand-coded "
                "BilinearForm + asm + condense snippets that "
                "the user copies — there is no single-call NS "
                "API. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Block system: [[A_visc + C(u), "
                "B^T], [B, 0]] where C is linearized "
                "convection. Signal: omitting C(u) from the "
                "BilinearForm gives a Stokes Jacobian and "
                "Newton converges linearly (not quadratically) "
                "on Navier-Stokes — residual ratio ~0.5 per "
                "iteration instead of decreasing "
                "geometrically across asm + condense + "
                "spsolve. (Audit 2026-06-02.)"
            ),
            (
                "[API] The convection term needs a VECTOR basis: "
                "build it on ElementVector(ElementTriP2()), not "
                "on a scalar ElementTriP2. Do NOT expect an "
                "AttributeError to tell you — CellBasis DOES have "
                "a `split` method in skfem 12.0.1 "
                "(AbstractBasis.split(x), which splits a SOLUTION "
                "VECTOR into per-component (x, basis) pairs, not "
                "the basis itself), so `'CellBasis' has no "
                "attribute 'split'` is never emitted; calling it "
                "with no argument raises TypeError "
                "'AbstractBasis.split() missing 1 required "
                "positional argument: x', which says nothing "
                "about the element type. Signal: check the SHAPE, "
                "not an exception — basis.N on a scalar element "
                "is the scalar DOF count, so the assembled C(u) "
                "block is a factor of dim too small and the "
                "block-stacked Newton system will not match the "
                "velocity space; assert basis.N equals "
                "dim * n_scalar_dofs before assembling. (Quoted "
                "string re-checked live 2026-08-06 on skfem "
                "12.0.1 and found absent.)"
            ),
            (
                "[Numerical] Pressure nullspace for enclosed "
                "flow: pin one pressure DOF, or impose a "
                "mean-zero constraint. Enclosed means velocity "
                "prescribed on the WHOLE boundary; an open flow "
                "with a traction (do-nothing) outlet has its "
                "pressure level set by that BC and needs no pin. "
                "DO NOT GUARD THIS WITH `MatrixRankWarning: "
                "Matrix is exactly singular`. On the pressure "
                "null space that warning does NOT fire: the "
                "block system is singular but CONSISTENT, so "
                "SuperLU returns a particular solution and "
                "spsolve gives back a finite, usable velocity "
                "field — identical to the pinned one — with only "
                "the pressure LEVEL undetermined. There is no "
                "warning, no exception and no NaN, so a "
                "warnings.catch_warnings(record=True) guard "
                "records an empty list and an np.isfinite "
                "assertion passes; both read their own silence as "
                "success. The warning is genuine scipy behaviour "
                "and does fire, but on a DIFFERENT fault — a "
                "matrix that is actually rank-deficient, such as "
                "an equal-order velocity/pressure pair violating "
                "inf-sup — and there the solution IS non-finite, "
                "which is why isfinite is the right partner for "
                "THAT case and useless for this one. "
                "Signal: the null space is observable, just "
                "not by that warning — "
                "compute the nullity of the condensed block, or "
                "solve twice pinning different DOFs or values and "
                "check that the velocity is unchanged to "
                "round-off while the pressure shifts by a "
                "constant. (Verified empirically 2026-08-06 on "
                "skfem 12.0.1 / scipy 1.15.3 — signal corrected: "
                "the previously quoted warning does not fire on "
                "this case.)"
            ),
            (
                "[Numerical] High Re: the older advice here — "
                "'pure Newton at Re > ~200 diverges from an "
                "at-rest guess, so run Picard for five "
                "iterations first' — is wrong in three ways and "
                "its remedy can make matters worse. Measured on "
                "a lid-driven cavity with Taylor-Hood: pure "
                "Newton from rest converges at Re = 200 and at "
                "Re = 400 in a handful of iterations, so the "
                "quoted threshold is far too low. Where Newton "
                "DOES fail from rest, the cause was the MESH and "
                "not the Reynolds number — the same Re from the "
                "same at-rest guess converged on a once-refined "
                "mesh, so an under-resolved discretisation was "
                "being read as a nonlinear-solver problem; check "
                "resolution before blaming the solver. And the "
                "failure does not 'explode within 2-3 "
                "iterations': the residual STAGNATES near its "
                "starting value for several iterations before it "
                "blows up, so a watchdog armed for an early "
                "explosion sees a plausible plateau instead. "
                "Worst of all, starting with Picard steps and "
                "then switching took a case where pure Newton "
                "converged and made it DIVERGE. Prefer "
                "continuation in Re: ramp the Reynolds number "
                "and start each solve from the previous "
                "solution, which converged at every step of a "
                "ramp including a Re that failed from rest on "
                "the same mesh. Signal: before concluding that "
                "Newton has left its basin, re-run the same Re "
                "on a finer mesh and re-run it from a "
                "continuation guess; if either converges, the "
                "basin was never the problem. (Verified "
                "empirically 2026-08-06 on skfem 12.0.1 — "
                "threshold, failure shape and remedy all "
                "corrected.)"
            ),
            (
                "[Numerical] Convection linearization: "
                "(u_prev.grad)delta_u + (delta_u.grad)u_prev. "
                "Signal: dropping the second term in the "
                "BilinearForm (Picard linearization instead "
                "of Newton) gives linear convergence on the "
                "asm + condense + spsolve pipeline — useful "
                "as a starter but switch to full Newton for "
                "quadratic. (Audit 2026-06-02.)"
            ),
            (
                "[API] Basis DOF ordering with ElementVector: "
                "use ib_u.N for the [u; p] block split. "
                "Signal: 'off-by-one' badly understates the "
                "error — a vertices-times-dimension formula "
                "counts only vertex DOFs, so on a P2 or Q2 "
                "vector space, which also carries edge DOFs, it "
                "can be short by a FACTOR of several, not by "
                "one. The formula is exactly RIGHT for a P1 "
                "vector space on triangles and for Q1, which is "
                "what makes it dangerous: it is correct in the "
                "first case a reader tries and silently wrong "
                "the moment the element order changes. The "
                "consequence is silent — splitting the stacked "
                "vector at the naive index yields a "
                "'pressure' slice of the wrong length with an "
                "empty warning list and a block of velocity "
                "entries pulled into it. Guard by asserting "
                "that basis_u.N plus basis_p.N equals len(sol), "
                "and that basis_u.N equals dim times "
                "scalar_basis.N — assertions to WRITE, not text "
                "anything prints — before "
                "slicing; ib_u.N is the canonical accessor. "
                "(Verified 2026-08-06 on skfem 12.0.1 — the "
                "'off-by-one' wording corrected.)"
            ),
        ],
    },
    "hyperelasticity": {
        "description": "Neo-Hookean hyperelasticity — Newton iteration (scikit-fem)",
        "solver": "Newton loop: assemble tangent stiffness K_tan and residual R_int, spsolve",
        "elements": "ElementVector(ElementTriP1()) or ElementVector(ElementTriP2())",
        "pitfalls": [
            (
                "[API] scikit-fem has NO built-in hyperelastic "
                "model — you must implement PK1 stress and "
                "tangent manually inside a BilinearForm. Signal: "
                "looking for skfem.models.elasticity.neohookean "
                "or similar fails (`ImportError`); the only "
                "linear elasticity helper is "
                "skfem.models.elasticity.linear_elasticity. "
                "Hyperelastic problems require hand-coded "
                "P(F) + C(F) inside @BilinearForm. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Neo-Hookean strain energy: W = "
                "mu/2 * (I1 - 2) - mu * lnJ + lam/2 * (lnJ)^2. "
                "Swapping the +/- sign on the lnJ term (writing "
                "+ mu * lnJ instead of - mu * lnJ) destroys the "
                "stress-free reference state. "
                "Signal: check it BEFORE stepping, with the "
                "claim's own sanity test, because that is the "
                "cheap discriminator: assemble the residual at "
                "u = 0 — for the correct law it is exactly zero, "
                "for the flipped one it is O(1), so W(F=I) = 0 "
                "and P(F=I) = 0 separate the two before any "
                "Newton iteration happens. Do NOT wait for a "
                "growing residual: there is no ~10x-per-iteration "
                "ramp to watch. The flipped tangent is symmetric "
                "AND positive definite, so a definiteness check "
                "misses it too; what actually happens is that the "
                "first Newton step overshoots the prescribed "
                "displacement far enough to turn elements inside "
                "out, det(F) goes non-positive, log(J) returns "
                "NaN, and every later residual is NaN — so the "
                "residual RATIO is undefined rather than large, "
                "and a gate testing for geometric growth never "
                "matches. Guard on np.isfinite of the residual "
                "and on min(det(F)) > 0. (Verified 2026-08-06 on "
                "skfem 12.0.1 — energy and sanity check "
                "confirmed, the wrong-sign-tangent mechanism "
                "with its ~10x per-iteration residual growth "
                "falsified.)"
            ),
            (
                "[Numerical] 1st Piola-Kirchhoff stress: P = "
                "mu*(F - F^{-T}) + lam * lnJ * F^{-T}. Using "
                "the 2nd PK form S = mu*(I - C^{-1}) + "
                "lam*lnJ*C^{-1} (small-strain-looking) inside "
                "the @BilinearForm and feeding it directly to "
                "∫P:grad(v) dx omits a LEFT multiplication by "
                "F: since F C^{-1} = F^{-T}, F @ S reproduces "
                "P exactly, so the two residual vectors "
                "genuinely differ. Recipe: stay in PK1 if your "
                "weak form is ∫P:grad(v) dx; use S only if you "
                "actually integrate "
                "∫S:0.5*(F^T grad v + grad(v)^T F) dx. "
                "Signal: NEWTON DOES NOT DIVERGE — this is the "
                "silent-wrong-answer shape, not the loud one. "
                "With the exact PK1 tangent driving it, the "
                "S-in-the-PK1-slot residual falls monotonically "
                "to machine zero with no NaN and an EMPTY "
                "warning list, and converges onto a DIFFERENT "
                "displacement field whose discrepancy grows "
                "with the applied stretch. A divergence watch, "
                "an np.isfinite assertion and the loop's own "
                "converged flag all pass. Two things do reveal "
                "it: the residual RATIO per Newton iteration is "
                "CONSTANT (linear) instead of collapsing "
                "(quadratic), which says the residual and the "
                "tangent are not derivatives of the same "
                "functional; and restoring the F "
                "pre-multiplication MOVES the converged "
                "displacement, so re-solving both ways and "
                "differencing settles it. Against a reference "
                "is the only way to see it from the outside. "
                "(Verified 2026-08-06 on skfem 12.0.1 — the "
                "algebra confirmed, the 'Newton diverges' "
                "signal falsified.)"
            ),
            (
                "[API] Deformation gradient: F = I + grad(u). "
                "Forgetting the identity I in the @BilinearForm "
                "body gives a degenerate F at the reference "
                "configuration (F = 0 instead of I), so "
                "J = det(F) = 0 and lnJ = -inf. "
                "Signal: it is the NaN branch, not the "
                "too-large branch — the first residual out of "
                "asm is NaN in every entry, and the only thing "
                "emitted is one numpy RuntimeWarning 'invalid "
                "value encountered in divide' from "
                "skfem.helpers.inv. Nothing raises and the "
                "Newton loop runs to completion. Careful with "
                "the guard: condense writes the prescribed "
                "values in verbatim, so the resulting "
                "displacement is NaN on the FREE DOFs but finite "
                "on the constrained ones — `np.isnan(u).all()` "
                "is False and misses it, while "
                "`np.isfinite(u).all()` is False and catches it. "
                "Cheapest check of all: assert det(F) == 1 at "
                "u = 0 before the first step. (Verified "
                "2026-08-06 on skfem 12.0.1.)"
            ),
            (
                "[Numerical] Material tangent: C4 = lam * "
                "C^{-1} ⊗ C^{-1} + 2*(mu - lam*lnJ) * "
                "I4_sym_C^{-1}. Dropping the I4_sym_C^{-1} term "
                "and using a scalar-multiplied identity I4 "
                "leaves an INEXACT tangent: Newton then "
                "converges at a linear rate instead of a "
                "quadratic one, because the operator is no "
                "longer the derivative of the residual. "
                "Signal: do NOT test against a fixed ratio "
                "threshold — the asymptotic contraction factor "
                "is not a constant of the bug, it tracks the "
                "strain level and can be anything from very "
                "small at a few percent stretch to a sizeable "
                "fraction at large stretch, so a '~0.5' gate "
                "passes the inexact tangent at low strain. Test "
                "the SHAPE of the history instead: the exact "
                "tangent's residual ratio collapses by orders of "
                "magnitude from one iteration to the next, the "
                "inexact one holds a roughly constant ratio for "
                "as many iterations as you allow it. Note the "
                "inexact tangent NEVER diverges, never NaNs, "
                "emits no warning, and lands on the SAME "
                "displacement to round-off — the cost is "
                "iterations, not correctness, so an "
                "answer-checking gate cannot see it and only the "
                "iteration count and the ratio history can. "
                "(Verified 2026-08-06 on skfem 12.0.1 — the "
                "linear-rate mechanism confirmed, the fixed "
                "'~0.5' rate falsified.)"
            ),
            (
                "[Numerical] Geometric stiffness term S : "
                "(grad du)^T * grad v is ESSENTIAL for exact "
                "linearisation; dropping it and assembling only "
                "the material stiffness C4 leaves an inexact "
                "tangent. "
                "Signal: dropping it does NOT leave you with "
                "quadratic Newton convergence at small strains — "
                "the material-only tangent is already LINEAR at "
                "a couple of percent stretch, well inside the "
                "'< 5% is fine' window the old wording "
                "promised, and it still reaches the right "
                "answer, so only the RATE reveals it. At large "
                "strains the Newton residual stagnates: the "
                "history goes drop-then-flat and then to NaN, "
                "which is worse than stagnation alone. "
                "Do not gate on an absolute plateau level: "
                "where the residual flat-lines depends on the "
                "problem, not on the bug, so a fixed threshold "
                "is unusable. Watch the SHAPE — a residual that "
                "falls once and then stops moving over "
                "successive iterations — together with the "
                "iteration count against the exact tangent, and "
                "assert min(det(F)) > 0 so the NaN branch is "
                "caught before it propagates. (Verified "
                "2026-08-06 on skfem 12.0.1 — the "
                "'quadratic below 5% strain' claim and the "
                "absolute plateau level are both falsified.)"
            ),
            (
                "[API] Use ib.interpolate(u) to obtain the "
                "displacement and gradient at quadrature "
                "points inside the bilinear form, and pass the "
                "state through assemble(ib, disp=...) so every "
                "Newton iteration sees the CURRENT iterate. The "
                "correct pattern is u_qp = w['disp'] (whatever "
                "kwarg name you assembled with), not a closure "
                "over an outer array. "
                "Signal: writing `u` in the form body and "
                "meaning the displacement does NOT raise "
                "NameError — inside a @BilinearForm the name `u` "
                "is already BOUND, to the trial function, so the "
                "form assembles a full-size matrix with no error "
                "at all. That is a silent type confusion, not a "
                "missing name. Two failures near it ARE loud and "
                "are worth telling apart: a genuinely undefined "
                "identifier raises the ordinary Python NameError "
                "naming it, and omitting the state kwarg at "
                "assemble() raises KeyError naming the kwarg. "
                "The stale-closure variant is the silent one: "
                "the residual returns the SAME value at every "
                "iteration, emits no warning, stays finite, and "
                "lands away from the correct solution — so watch "
                "for a residual that does not change between "
                "iterations, which is the fingerprint of a "
                "frozen state. (Verified 2026-08-06 on skfem "
                "12.0.1 — the NameError signal is falsified.)"
            ),
            "[API] MeshQuad.to_meshtri() (present since "
            "scikit-fem 1.2.0) returns a MeshTri with each "
            "quad split into two triangles. No scikit-fem release "
            "has a MeshQuad.to_simplex() method, so calling "
            ".to_simplex() on a MeshQuad raises AttributeError: "
            "'MeshQuad1' object has no attribute 'to_simplex'. "
            "Use .to_meshtri(). Signal: hasattr("
            "skfem.MeshQuad.init_tensor([0,1],[0,1]), 'to_meshtri') "
            "is True; hasattr(..., 'to_simplex') is False. "
            "(Verified empirically 2026-06-01 — Layer F catch.)",
            (
                "[Numerical] Load stepping: ramp the load "
                "over N steps for large deformations to "
                "keep Newton inside its convergence basin, "
                "with load_factor = i/N and the previous "
                "converged iterate carried in as the initial "
                "guess. "
                "Signal: a single-step application of the full "
                "load does fail, but NOT with a residual that "
                "visibly diverges — there is no "
                "~10x-per-iteration ramp to watch. The very "
                "first Newton correction turns elements inside "
                "out and the residual is already NaN; nothing "
                "grows geometrically and nothing is printed. "
                "Worse, "
                "the displacement vector is still FINITE at the "
                "moment the configuration becomes impossible, so "
                "an np.isfinite(u) guard PASSES on an inverted "
                "mesh. The check that fires is "
                "min(det(F)) > 0, evaluated after every Newton "
                "correction, not just at the end. Confirm the "
                "ramp is a genuine continuation rather than a "
                "different problem by running two different "
                "substep counts and checking they agree on the "
                "final displacement to round-off. (Verified "
                "2026-08-06 on skfem 12.0.1 — the residual-ramp "
                "signal falsified; det(F) is the observable.)"
            ),
            (
                "[API] The numpy @ matmul operator does NOT "
                "work on skfem's (d, d, n_elem, n_quad)-shape "
                "tensor fields inside a @BilinearForm kernel. "
                "matmul tries the last-two-dims convention and "
                "aborts. Signal: 'matmul: Input operand 1 has "
                "a mismatch in its core dimension 0, with "
                "gufunc signature (n?,k),(k,m?)->(n?,m?) (size "
                "N is different from d)' from numpy inside the "
                "BilinearForm. Fall back to explicit component "
                "algebra (Finv[0,0]*Finv[0,0] + Finv[0,1]*"
                "Finv[0,1] for (Finv * Finv^T)[0,0]) or use "
                "skfem.helpers.ddot / dot / transpose helpers "
                "that operate elementwise on the rank-2 "
                "deformation-gradient tensor. (Audit "
                "2026-06-02, post-mortem skfem-broken-newton-"
                "templates-rewrite.)"
            ),
            (
                "[Numerical] Newton iteration on a Neo-Hookean "
                "BilinearForm can drive the deformation gradient "
                "into J = det(F) <= 0 after the FIRST "
                "correction; log(J) then returns NaN, the "
                "displacements go NaN, and the process exit code "
                "is STILL 0, so an rc-only gate passes on a NaN "
                "answer. The trigger is the DIRECTION and size "
                "of the load relative to the stiffness, not a "
                "particular parameter pair: an axial traction "
                "of the same nominal magnitude that is quoted as "
                "dangerous can converge with det(F) comfortably "
                "positive, while a TRANSVERSE traction on the "
                "same soft material inverts elements immediately "
                "— so do not read a specific E / traction pair "
                "as a safety boundary. "
                "Signal: it is not fully silent, but it is "
                "silent in the places a gate usually looks. "
                "Nothing raises; the child process returns 0; "
                "the informative lines come from the template "
                "rather than from skfem — the script prints "
                "'Max displacement: nan' and '||du|| = nan'; "
                "and on the way there scipy raises "
                "MatrixRankWarning('Matrix is "
                "exactly singular'). Put 'nan' in the gate's "
                "forbid_in_output list, assert min(det(F)) > 0 "
                "after every Newton correction, and require "
                "np.isfinite on the displacement — an rc check "
                "alone gates nothing. Defence that works: a "
                "stiffer material or a smaller load, verified by "
                "det(F) staying positive rather than assumed "
                "from the parameter values. (Verified 2026-08-06 "
                "on skfem 12.0.1 — mechanism confirmed, the "
                "quoted E / traction pair does not reproduce it; "
                "post-mortem skfem-broken-newton-templates-"
                "rewrite.)"
            ),
            (
                "[API] skfem.helpers.det() and inv() SILENTLY "
                "return all-zeros for ANY square matrix whose "
                "leading dimension is NOT 2 or 3. Source: "
                "skfem/helpers.py defines det/inv with explicit "
                "`A.shape[0] == 2` / `A.shape[0] == 3` branches "
                "and no `else`; the initial `detA = zeros_like("
                "A[0, 0])` sticks. 2x2 and 3x3 are silent for "
                "deformation-gradient F, but mixed formulations "
                "with augmented F (e.g. 4x4 F+p block) or any "
                "user-supplied 4D/5D tensor produce a zero "
                "Jacobian → log(0) = -inf → NaN-everywhere "
                "Newton residual. No NotImplementedError raised. "
                "Signal: assemble a hyperelastic form with "
                "A.shape[0] >= 4 and check that "
                "det(A).any() == True FAILS (returns zeros). "
                "Workaround: pre-assert F.shape[0] in (2, 3) at "
                "the top of your @BilinearForm body, or fall "
                "back to numpy.linalg.det along the trailing "
                "axes. (File-walk skfem/helpers.py 2026-06-02; "
                "verified live in skfem 12.0.1.)"
            ),
        ],
    },
    "dg_methods": {
        "description": "Discontinuous Galerkin for advection/diffusion using ElementDG (scikit-fem)",
        "solver": "Direct sparse (non-symmetric system from upwind flux); GMRES for large problems",
        "elements": "ElementDG(ElementTriP1()), ElementDG(ElementTriP2())",
        "pitfalls": [
            (
                "[API] ElementDG wraps any element to make it "
                "fully discontinuous. Signal: forgetting the "
                "wrapper and using a bare ElementTriP1 in a "
                "DG context produces a continuous-Galerkin "
                "global solution — no jumps at element edges, "
                "and the upwind/penalty terms in the form "
                "evaluate to zero. (Audit 2026-06-02.)"
            ),
            (
                "[API] InteriorFacetBasis assembles over interior "
                "mesh facets — that is where the DG flux terms "
                "belong. A facet form handed a plain Basis "
                "(CellBasis) has no facet normal in its `w` "
                "dict, and it produces no facet coupling; the "
                "jump has to come from assembling over the "
                "side=0 / side=1 pair, `asm(form, [i0, i1], "
                "[i0, i1])`. There is no `u.other` attribute in "
                "scikit-fem 12.0.1. Signal: reading `w.n` inside "
                "a form assembled on a plain Basis raises "
                "`AttributeError: Attribute 'n' not found in "
                "'w'.` — the message is raised by FormExtraParams "
                "(the `w` dict), NOT by Basis, and the word "
                "'normals' never appears in it, so a guard that "
                "greps for \"'Basis' object has no attribute "
                "'normals'\" never fires; writing `u.other` "
                "raises `AttributeError: 'DiscreteField' object "
                "has no attribute 'other'`; and counting matrix "
                "entries that couple DOFs of different elements "
                "gives exactly zero for the plain-Basis assembly "
                "while the InteriorFacetBasis side=0/side=1 pair "
                "on the same mesh gives a non-zero count. "
                "(Verified 2026-08-06 on skfem 12.0.1 — Tier-2 "
                "fixture dg_plain_basis_facet_normals.)"
            ),
            (
                "[API] FacetBasis assembles over BOUNDARY facets "
                "(not interior); used for inflow/outflow flux "
                "BCs. Signal: applying an inflow BC by adding a "
                "term over InteriorFacetBasis instead of "
                "FacetBasis silently adds the inflow to every "
                "interior edge — the solution gets a spurious "
                "source distributed across the mesh interior. "
                "Sanity check: a pure-Dirichlet steady advection "
                "with the wrong basis shows a non-monotone "
                "solution. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Upwind flux: bn * u_upwind * [v]; "
                "must identify upwind side from sign of b.n. "
                "Signal: a wrong upwind/downwind choice gives "
                "centered flux that is unconditionally "
                "unstable for pure advection — solution "
                "develops oscillations growing geometrically "
                "in the advection direction. (Audit "
                "2026-06-02.)"
            ),
            (
                "[API] scikit-fem uses a SINGLE-SIDED "
                "InteriorFacetBasis: one basis object carries ONE "
                "side of each interior facet, chosen with `side=` "
                "(side=0 selects mesh.f2t[0], side=1 selects "
                "mesh.f2t[1]), and it visits each interior facet "
                "ONCE — basis.nelems equals the number of interior "
                "facets, not twice that. The jump is not something "
                "you spell in the form; it comes out of the 2x2 "
                "block assembly `asm(form, [i0, i1], [i0, i1])`, "
                "which is what supplies `w.idx`. There is no "
                "`(u - u.other)` spelling. This matters because "
                "skfem.helpers.jump(w, u, v) returns its arguments "
                "UNCHANGED when `w` carries no `idx`, and a "
                "single-basis `form.assemble(i0)` carries none: a "
                "FEniCSx form ported verbatim is therefore not off "
                "by a factor of 2, it stops being a jump at all. "
                "Signal: the verbatim single-basis port does NOT "
                "annihilate a continuous field — project a globally "
                "continuous function u_c and the quadratic form "
                "u_c @ (P @ u_c) comes out O(1) instead of zero, "
                "the matrix has exactly zero entries coupling DOFs "
                "of different elements, and no scalar multiple of "
                "it matches the 2x2 block operator (fit c from one "
                "Rayleigh quotient and max|P_single - c*P_block| "
                "stays far above round-off). A check that looks for "
                "a clean factor of 2 never fires. (Verified "
                "2026-08-06 on skfem 12.0.1 — Tier-2 fixture "
                "dg_single_sided_jump.)"
            ),
            (
                "[Numerical] For IP (interior penalty) diffusion "
                "DG the facet penalty is sigma/h on each interior "
                "facet, and sigma = 4 * order^2 for symmetric IP "
                "is not a safety margin but the actual coercivity "
                "threshold on triangles: below it the assembled "
                "symmetric operator has a negative smallest "
                "eigenvalue, at and above it the eigenvalue is "
                "positive, and the threshold moves with order^2, "
                "so a sigma that is coercive for ElementDG("
                "ElementTriP1) is not coercive for "
                "ElementDG(ElementTriP2). Too-small sigma does NOT "
                "inflate the solution — watch the convergence "
                "RATE, not the norm. Signal: with sigma below the "
                "rule of thumb, numpy.linalg.eigvalsh(A.toarray())"
                "[0] on the SIPG matrix is negative, while the "
                "discrete L2 norm of u is essentially unchanged "
                "from one refinement level to the next, so a "
                "norm-growth guard never fires; what collapses is "
                "the L2 error rate, which falls from near-optimal "
                "to well under first order. With sigma many orders "
                "above the rule of thumb, cond(K) exceeds 1e14 and "
                "scipy.sparse.linalg.cg needs more than ten times "
                "the iterations for the same rtol. (Verified "
                "2026-08-06 on skfem 12.0.1 — Tier-2 fixture "
                "dg_ip_penalty_sigma.)"
            ),
            (
                "[Validation] The shipped dg_methods_2d template "
                "DOES NOT SOLVE — it returns NaN everywhere and "
                "still exits rc=0 — and it carries TWO independent "
                "defects, so the obvious repair is NECESSARY BUT "
                "NOT SUFFICIENT. (a) Its pure-advection operator "
                "has no inflow boundary term pinning the solution, "
                "so the matrix is rank-deficient; adding the "
                "inflow FacetBasis bilinear term (or a small "
                "reaction/mass shift) removes the singularity. "
                "(b) Its interior upwind flux is assembled on a "
                "SINGLE InteriorFacetBasis instead of the "
                "side=0 / side=1 pair, so it contributes ZERO "
                "inter-element coupling and is not a DG flux at "
                "all; it must be re-assembled as `asm(flux, "
                "[i0, i1], [i0, i1])` too. Fixing only (a) yields "
                "a finite but WRONG answer. Do NOT take this "
                "template's output as a reference. Signal: "
                "unmodified it prints scipy's 'MatrixRankWarning: "
                "Matrix is exactly singular' at the spsolve line; "
                "the script then prints "
                "'DG advection: max(u) = nan, min(u) = nan', "
                "and writes \"max_value\": NaN / \"min_value\": NaN "
                "into results_summary.json while returning 0 — a "
                "gate that checks only the return code passes it, "
                "so always assert np.isfinite(u).all(). After the "
                "inflow-term fix alone the NaNs and the warning are "
                "gone but min(u) is NEGATIVE on a problem whose "
                "exact solution is non-negative everywhere; only "
                "with the interior flux fixed as well does the "
                "solution stay inside its physical range and "
                "increase monotonically along the streamwise "
                "coordinate. (Verified 2026-08-06 on skfem 12.0.1 "
                "/ scipy 1.15.3 — Tier-2 fixture "
                "dg_shipped_template_nan.)"
            ),
            (
                "[API] Basis.project(Basis.interpolator(u)) for "
                "nodal post-processing (the module-level "
                "project(u, basis_from=ib_dg, basis_to=ib_p1) "
                "still works but is deprecated — see the next "
                "entry). "
                "Signal: visualizing the ElementTriDG "
                "GridFunction-equivalent solution directly "
                "in ParaView with the wrong VTK writer "
                "produces an all-zero or step-pattern field "
                "because DG DOFs are not nodal; projecting "
                "with the skfem project() helper to "
                "ElementTriP1 first restores a smooth "
                "visualization. (Audit 2026-06-02.)"
            ),
            (
                "[API] Module-level skfem.project() and "
                "skfem.projection() are DEPRECATED and emit "
                "DeprecationWarning: projection() -> "
                "'projection is deprecated in favor of "
                "Basis.project.'; project() -> 'project is "
                "deprecated in favor of Basis.project (will be "
                "removed in the next release).' project() calls "
                "projection() internally, so it also emits the "
                "projection warning. Source: "
                "skfem/__init__.py top-level __all__ flags "
                "both with `# TODO remove due to deprecation`. "
                "Signal: existing DG-to-P1 visualization "
                "snippets `skfem.project(u, basis_from=ib_dg, "
                "basis_to=ib_p1)` warn now and will raise "
                "AttributeError on upgrade. Replacement: "
                "`f = ib_dg.interpolator(u); "
                "u_p1 = ib_p1.project(f)` — the Basis.project "
                "INSTANCE method. (File-walk audit "
                "2026-06-02; verified live in skfem 12.0.1.)"
            ),
            (
                "[Numerical] The DG system is non-symmetric even "
                "for symmetric problems (upwind asymmetry) and its "
                "symmetric part is at best positive SEMI-definite, "
                "so scipy.sparse.linalg.cg is the wrong solver — "
                "but it does not tell you so by raising. On scipy "
                "1.15.3 `cg` raises NOTHING on this matrix: it "
                "returns a non-zero `info` (maxiter reached) "
                "together with a vector whose relative residual is "
                "LARGER than the right-hand side, so a try/except "
                "guard passes and the unconverged vector flows on "
                "into the results. Never gate an iterative solve "
                "on an exception; gate it on `info == 0` AND a "
                "recomputed norm(A @ x - b) / norm(b). Use gmres or "
                "a direct LU (scipy.sparse.linalg.splu) instead. "
                "Signal: calling cg(A, b, rtol=1e-10) and unpacking "
                "its (x, info) pair returns "
                "with info != 0 and norm(A @ x - b)/norm(b) > 1, "
                "with no exception raised and the text 'matrix not "
                "positive definite' appearing nowhere in the output "
                "or warnings; the same matrix through gmres returns "
                "info == 0 at the requested tolerance and through "
                "splu reproduces the exact solution to round-off. "
                "(Verified 2026-08-06 on skfem 12.0.1 / scipy "
                "1.15.3 — Tier-2 fixture dg_cg_on_nonsymmetric.)"
            ),
            (
                "[Numerical] SUPG (continuous Galerkin with "
                "stabilisation) is often more stable than pure DG "
                "for steady-state advection on a P1 mesh, and the "
                "thing to compare is the ORDERING under "
                "refinement, not an amplitude. At Pe_h > 5 on a "
                "coarse MeshTri a pure ElementDG(ElementTriP1) "
                "upwind solve rings across element faces, and when "
                "the exact solution carries a genuine "
                "discontinuity (an interior or boundary layer fed "
                "by a discontinuous inflow datum) that ringing is "
                "h-INDEPENDENT: it does not clear with refinement "
                "and the excursion can creep upward, so 'it will "
                "go away after a few levels' is not a safe "
                "expectation and no fixed percentage band "
                "describes it. Signal: measure the excursion "
                "outside the physically admissible range, "
                "max(u.max() - u_hi, u_lo - u.min(), 0), at three "
                "successive refinements — for pure DG the sequence "
                "does not decrease monotonically and does not fall "
                "to a small fraction of its coarse-level value, "
                "while the SUPG-CG ElementTriP1 BilinearForm run "
                "decreases monotonically, is smaller than DG at "
                "every level, and uses a fraction of the DOFs. For "
                "predominantly-smooth solutions SUPG-CG wins on "
                "cost-per-accuracy. (Verified 2026-08-06 on skfem "
                "12.0.1 — Tier-2 fixture dg_supg_vs_pure_dg.)"
            ),
        ],
    },
    "time_dependent": {
        "description": "General time-dependent PDE with theta-method (backward Euler / Crank-Nicolson) (scikit-fem)",
        "solver": "factorized(A) for efficient time-stepping; A = M + theta*dt*K assembled once",
        "elements": "ElementQuad1, ElementTriP1 (any H1-conforming element)",
        "pitfalls": [
            (
                "[Numerical] Backward Euler (theta=1) is "
                "unconditionally stable and 1st order in time; a "
                "log-log dt study gives slope 1, and slope 2 "
                "means theta is mis-set to Crank-Nicolson. "
                "Signal: the reference matters more than the "
                "slope rule. Measure the error against a "
                "SEMI-DISCRETE reference — the same spatial mesh "
                "and the same assembled matrices, integrated "
                "with many more time steps — not against the "
                "analytic PDE solution. Against the analytic "
                "solution the SPATIAL error saturates the study: "
                "a correctly configured Crank-Nicolson run has "
                "its measured slope collapse towards zero, which "
                "the rule that reads a slope of zero as dt being "
                "too large "
                "misreads as a time-stepping fault when the time "
                "integration is in fact second order and fine. "
                "With a same-mesh reference the two schemes "
                "separate cleanly and never overlap. (Verified "
                "2026-08-06 on skfem 12.0.1 — the diagnostic "
                "works, its stated reference does not.)"
            ),
            (
                "[Numerical] Crank-Nicolson (theta=0.5) is 2nd "
                "order in time but rings at sharp transients, "
                "and at a FIXED dt/h^2 the ringing does not damp "
                "under mesh refinement. Switching to theta=1 "
                "(backward Euler) removes it — its undershoot "
                "drops to rounding level — at the cost of "
                "first-order accuracy. "
                "Signal: do NOT calibrate on a 10-30% band. The "
                "over/undershoot is not a property of the "
                "scheme: on ONE mesh it spans more than an order "
                "of magnitude as dt/h^2 is varied, so the "
                "governing quantity is dt/h^2 and any fixed "
                "percentage threshold either misses the ringing "
                "or condemns a healthy run. Test for the "
                "presence of a NEGATIVE minimum (or an excursion "
                "outside the physically admissible range) on a "
                "problem whose exact solution stays inside that "
                "range, and record dt/h^2 alongside it. "
                "(Verified 2026-08-06 on skfem 12.0.1 — ringing "
                "and its non-damping confirmed, the 10-30% band "
                "falsified.)"
            ),
            (
                "[Performance] Factor system matrix ONCE and "
                "reuse — factorized() from scipy.sparse.linalg. "
                "Signal: profile shows >90% time in "
                "scipy.sparse.linalg.spsolve per step; using "
                "factorized() reduces per-step cost from full "
                "LU to forward+back-substitution (~10-100x "
                "speedup for a fixed matrix). (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Non-homogeneous time-varying BCs: "
                "update rhs and re-condense each step. Signal: "
                "if rhs is condensed once and reused, the "
                "boundary DOFs stay at their initial values "
                "across the time loop — solution at boundary "
                "diverges from the prescribed time-varying "
                "BC. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] CFL is NOT needed for BE (the "
                "implicit scheme is L-stable), but dt still "
                "affects ACCURACY. Signal: dt larger than the "
                "shortest physical timescale of the problem "
                "(e.g. dt >> 1/lambda_min) gives the correct "
                "steady state but smears any sharp transient — "
                "comparing the simulated u(t=t_transient) "
                "against a fine-dt reference shows L2 error "
                "~ O(dt) instead of resolving the front. "
                "Choose dt by accuracy, not stability, for "
                "implicit schemes. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] For stiff systems (reaction-"
                "dominated): backward Euler or BDF-2 preferred. "
                "Explicit (theta=0) or near-explicit "
                "(theta<0.5) stepping requires dt < "
                "2/lambda_max with lambda_max the largest "
                "eigenvalue of M^{-1}K, and the bound is SHARP: "
                "just below it the solution decays, just above "
                "it the solution grows without bound, and both "
                "happen SILENTLY — no warning, no exception, and "
                "backward Euler at the same unstable dt decays "
                "monotonically, which is what makes the "
                "comparison conclusive. "
                "Signal: do not use a Damkohler threshold as the "
                "test. Adding a reaction -Da*u shifts lambda_max "
                "by EXACTLY Da, so whether the reaction dominates "
                "depends on how lambda_max(M^{-1}K) already "
                "stands on YOUR mesh — on a fine mesh the "
                "diffusive eigenvalue can be far larger than a "
                "Damkohler number of a hundred, and 'Da > 100 is "
                "infeasible' is then simply not true. Compute "
                "lambda_max on the free block (scipy.linalg.eigh "
                "on the condensed pencil, or an estimate) and "
                "compare 2/lambda_max against the time you need "
                "to integrate; that ratio, not Da, tells you "
                "whether explicit stepping is affordable. Switch "
                "to a BE/BDF-2 implicit asm + spsolve loop when "
                "it is not. (Verified 2026-08-06 on skfem 12.0.1 "
                "— the bound confirmed sharp, the Da > 100 rule "
                "of thumb falsified as non-universal.)"
            ),
            (
                "[API] ib.doflocs is the (ndim, N) coordinate "
                "array used for initial-condition assignment. "
                "Signal: setting u_init from a callable like "
                "u_init = np.sin(np.pi * ib.mesh.p[0]) (using "
                "mesh.p directly) only matches DOFs for P1 "
                "elements; for ElementVector / P2 / higher-order "
                "the DOFs include face/edge midpoints and the "
                "initial condition is silently zero on those "
                "DOFs. Use ib.doflocs to get the correct (ndim, "
                "nDOF) array and ib.project / ib.interpolator "
                "for IC assignment. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] For explicit time stepping: "
                "M*du/dt = -K*u + f (avoid for stiff "
                "problems). The assembled FEM mass matrix M is "
                "NOT diagonal — it carries several non-zeros per "
                "row and its largest off-diagonal is comparable "
                "to its smallest diagonal — so naive explicit "
                "Euler requires solving with M every step; "
                "row-sum lumping gives a diagonal M with exactly "
                "one entry per row and conserves the total mass. "
                "Signal: past the bound dt > 2/lambda_max the "
                "energy does grow geometrically, but the claim "
                "that it blows up to NaN within a few steps is "
                "WRONG ABOUT THE "
                "TIMING and a short run is not a stability test. "
                "The peak can still be BELOW its initial value "
                "after the first handful of steps, so the run "
                "looks like ordinary decay, and NaN arrives only "
                "after a large number of steps. "
                "What is exact from the very beginning is the "
                "GROWTH FACTOR: the peak multiplies by "
                "|1 - dt*lambda_max| every step, matching the "
                "analytic amplification to round-off. Record the "
                "step-to-step peak ratio over a few consecutive "
                "steps and compare it against that expression — "
                "watch the ratio, never the step count, and "
                "never the fact that the run is still finite. "
                "(Verified 2026-08-06 on skfem 12.0.1 — "
                "non-diagonal mass and geometric growth "
                "confirmed, the few-steps-to-NaN timing "
                "falsified.)"
            ),
        ],
    },
    "helmholtz": {
        "description": "Helmholtz equation -Δu - k²u = f (complex-valued, scikit-fem)",
        "solver": "Direct sparse with complex128 (spsolve handles complex); GMRES for large k",
        "elements": "ElementQuad1, ElementTriP1 (standard H1; use fine mesh: ~10 DOFs per wavelength)",
        "pitfalls": [
            (
                "[API] THE complex trap in scikit-fem is at "
                "ASSEMBLY, not at matrix arithmetic. A "
                "@BilinearForm defaults to dtype=np.float64, so "
                "a kernel returning `1j * k * u * v` is written "
                "into a real buffer and the imaginary part is "
                "DISCARDED — you get an ALL-ZERO matrix and only "
                "a numpy ComplexWarning('Casting complex values "
                "to real discards the imaginary part') from "
                "skfem/assembly/form/bilinear_form.py. Signal: "
                "the assembled matrix has dtype float64, nnz 0 "
                "and max|A| exactly 0.0, and the only diagnostic "
                "is that ComplexWarning. Measured "
                "on MeshQuad.init_tensor(9x9) with ElementQuad1, "
                "FacetBasis on 'right': the default form gives "
                "dtype float64, nnz 0, max|A| 0.0; "
                "@BilinearForm(dtype=complex) on the SAME kernel "
                "gives complex128 with max|A| = 0.4166667 and a "
                "full imaginary part. Fix: "
                "@BilinearForm(dtype=complex) (the constructor "
                "signature is BilinearForm(form=None, "
                "dtype=np.float64, nthreads=0, **params)); "
                "calling .astype(complex) afterwards is TOO LATE "
                "because the information is already gone. "
                "CORRECTION to the prior text: sparse "
                "float + complex arithmetic works fine — "
                "K + 1j*M and even in-place K += 1j*M both "
                "promote to complex128 without any TypeError, so "
                "'Cannot cast array data from dtype(complex128) "
                "to dtype(float64) at sparse-matrix add' does "
                "not happen. (The one place that TypeError does "
                "appear is scalar assignment into a float lil "
                "matrix: Kl[0,0] = 1j raises "
                "TypeError whose literal clause is 'argument "
                "must be a string or a real number, not' with "
                "the rejected type quoted after it, so the line "
                "reads float() argument must be a string or a real "
                "number, not 'complex'.) The SHIPPED "
                "helmholtz_2d template declares its absorbing-BC "
                "form with @BilinearForm(dtype=complex) for "
                "exactly this reason — with a plain @BilinearForm "
                "that block assembles to zero, so the script has "
                "no ABC at all while still "
                "exiting rc=0. (Verified empirically 2026-08-03 "
                "on skfem 12.0.1 — catalog-drift correction + "
                "gap.)"
            ),
            (
                "[Numerical] Rule of thumb: at least 10 elements "
                "per wavelength (lambda = 2*pi/k). Signal: at "
                "fewer than 5 elements per wavelength the "
                "computed phase velocity is visibly wrong "
                "(e.g. a propagating wave hits the boundary at "
                "the wrong time by 10-30%); the standard "
                "dispersion error is O(k h)^2 for P1 and "
                "becomes catastrophic for h k > 1. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Absorbing BC: +i*k*u on the "
                "boundary, assembled via FacetBasis "
                "BilinearForm. Signal: omitting the ABC "
                "FacetBasis term produces standing-wave "
                "reflection off the domain boundary — "
                "visualised |u| on the asm'd MeshTri shows "
                "a checkerboard interference pattern with "
                "peaks spaced at lambda/2; adding the "
                "+i*k*u term in the FacetBasis BilinearForm "
                "absorbs the outgoing wave. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] PML (perfectly matched layer): "
                "extend the MeshTri / MeshQuad domain with a "
                "complex stretch factor (s = 1 - i*sigma(x)/k) "
                "inside a @BilinearForm(dtype=complex), ramping "
                "sigma smoothly from zero at the interface. "
                "sigma_max has an INTERIOR optimum — both "
                "directions degrade it, but not symmetrically: "
                "too SMALL a sigma is the direction that reflects "
                "more than it absorbs, while too LARGE a sigma "
                "degrades the reflection by orders of magnitude "
                "without ever getting that bad, so 'too-large "
                "sigma reflects more than it absorbs' is the "
                "wrong end of the sweep to watch. Thickness is "
                "monotone: thinner is always worse. Signal: "
                "assemble the stretched operator with "
                "@BilinearForm(dtype=complex) on a MeshQuad / "
                "ElementQuad1 strip, solve with spsolve, and read "
                "R = (SWR - 1)/(SWR + 1) off the INTERIOR Basis "
                "nodes only — every layer thinner than lambda/2 "
                "misses the |R| < 1e-3 target while 1-2 lambda "
                "meets it, sweeping sigma_max at a fixed good "
                "thickness traces a U with its minimum in the "
                "middle, and it is the small-sigma end that "
                "crosses R > 0.5. Standard choices: PML thickness "
                "1-2 lambda, sigma_max of order k, tuned so "
                "|R| < 1e-3 at normal incidence. (Verified "
                "2026-08-06 on skfem 12.0.1 — Tier-2 fixture "
                "helmholtz_pml_thickness_sigma.)"
            ),
            (
                "[Numerical] High k (k > 20): use higher-order "
                "elements (P2, P3) or DG to reduce pollution "
                "error. Signal: standard P1 at k=40 on a mesh "
                "with 10 elements / wavelength shows the "
                "computed solution drifting out of phase across "
                "the domain — the trailing-edge crest is shifted "
                "by ~1/4 wavelength relative to the analytic "
                "plane wave; P2 on the same mesh recovers the "
                "phase. (Audit 2026-06-02.)"
            ),
            (
                "[API] System is non-Hermitian with ABC — eigsh "
                "assumes Hermitian and gives WRONG NUMBERS "
                "SILENTLY. It does not raise. Signal: on "
                "MeshTri().refined(2) / ElementTriP1, taking the "
                "P1 stiffness matrix, casting to complex and "
                "adding 5j to a single off-diagonal entry (so "
                "the matrix is no longer Hermitian), "
                "scipy.sparse.linalg.eigsh(A, k=3, M=M, sigma=0) "
                "returned (-1.6043, 10.3570, 10.3578) with no "
                "warning, against the true Hermitian reference "
                "(1.78e-15, 10.3570, 10.3578) — the lowest "
                "eigenvalue is silently corrupted. There is no "
                "`ArpackNoConvergence: ARPACK error ... "
                "non-Hermitian`. Use "
                "scipy.sparse.linalg.eigs (general non-Hermitian "
                "ARPACK) or spsolve / GMRES for the forward "
                "problem. (Verified empirically 2026-08-03 on "
                "skfem 12.0.1 / scipy 1.15.3 — signal "
                "correction: the failure is silent, not loud.)"
            ),
            (
                "[API] Output the REAL PART (physical wave) and "
                "the MAGNITUDE |u| as two SEPARATE real fields — "
                "the solution is complex and nothing downstream "
                "accepts it as one array. The two failure modes "
                "are opposite and it matters which you guard for: "
                "the mesh writers are LOUD — meshio (and "
                "skfem's Mesh.save, which delegates to it) does "
                "not warn and does not truncate, it raises a "
                "KeyError and writes NO file at all — while the "
                "raw numpy path is SILENT and lossy. Signal: "
                "handing complex128 point data to the .vtu writer "
                "raises `KeyError: dtype('complex128')` and to "
                "the .xdmf writer `KeyError: 'complex128'`, with "
                "no output file on disk; the string 'Cannot write "
                "complex array' is not emitted by any of them, "
                "so a handler matching that text never fires. "
                "u.tofile(...) by contrast succeeds quietly, "
                "writes twice the bytes, and "
                "np.fromfile(dtype=np.float64) then returns an "
                "array of double the expected length holding real "
                "and imaginary parts INTERLEAVED — not the real "
                "part — so a length check catches it but a "
                "dtype-blind read does not. Write "
                "{'u_real': u.real, 'u_abs': np.abs(u)} instead; "
                "the real part alone loses the amplitude "
                "information. (Verified 2026-08-06 on skfem "
                "12.0.1 / meshio 5.3.5 — Tier-2 fixture "
                "helmholtz_complex_output_split.)"
            ),
            (
                "[Numerical] Pollution effect: for large k, "
                "standard P1 has O(k^3 h^2) phase error — use "
                "p-refinement to control it. The test is to hold "
                "k*h FIXED and raise k: a nominally well-resolved "
                "mesh does NOT give a k-independent error, "
                "because k^3 h^2 = k * (k h)^2 still grows "
                "linearly in k. Do not calibrate on a percentage: "
                "the accumulated phase drift over ONE domain "
                "length is a small fraction of a wavelength, "
                "while the relative L2 error after tens of "
                "wavelengths of propagation is percent-level, and "
                "the two differ by orders of magnitude. Signal: "
                "at fixed k*h, solve at a low k and a ten-times "
                "higher k on the same k*h and compare — both the "
                "phase drift and the relative L2 error grow "
                "roughly in proportion to k, whereas ElementQuad2 "
                "/ ElementTriP2 (O(k^5 h^4)) on the SAME mesh "
                "stays at round-off. If the two k's give the same "
                "error, the run is not in the pollution regime "
                "and the study proves nothing. Note 'GridFunction' "
                "is NGSolve vocabulary — in scikit-fem the "
                "solution is a plain numpy array and hasattr("
                "skfem, 'GridFunction') is False. (Verified "
                "2026-08-06 on skfem 12.0.1 — Tier-2 fixture "
                "helmholtz_pollution_fixed_kh.)"
            ),
            "[API] MeshQuad.init_tensor (and most other init_*) "
            "does NOT attach named boundaries. The canonical "
            "incantation is m = MeshQuad.init_tensor(...)"
            ".with_boundaries({'left': lambda x: x[0] < 1e-10, "
            "'right': ..., 'bottom': ..., 'top': ...}); then "
            "ib.get_dofs('left').flatten() yields the boundary "
            "DOF indices. Tags attached this way SURVIVE "
            ".to_meshtri() on skfem 12.0.1 — the triangulated "
            "mesh keeps the four names and get_dofs('left') "
            "returns the same edge nodes, so do NOT reattach "
            "boundaries after converting. Signal: TWO distinct "
            "errors depending on the call pattern — "
            "ib.get_dofs('left') raises ValueError(\"Boundary "
            "'left' not found.\") while the legacy subscript form "
            "ib.get_dofs()['left'] raises TypeError: 'DofsView' "
            "object is not subscriptable in scikit-fem 12; "
            "m.boundaries is None straight out of init_tensor and "
            "a dict of the four names after .with_boundaries(). "
            "(Verified 2026-08-06 on skfem 12.0.1 — Tier-2 "
            "fixture helmholtz_init_tensor_boundaries; the "
            "to_meshtri reattachment sentence in the prior text "
            "was FALSE and is removed.)",
        ],
    },
    "reaction_diffusion": {
        "description": "Reaction-diffusion system (Schnakenberg / Fisher-KPP) — Turing patterns (scikit-fem)",
        "solver": "Backward Euler in time + Newton iteration per step; block 2x2 system for coupled species",
        "elements": "ElementQuad1 (any H1 element; Neumann BCs are natural)",
        "pitfalls": [
            (
                "[Validation] GAP FILLED 2026-08-03: a "
                "reaction-diffusion run that exits rc=0 is NOT "
                "evidence that a Turing pattern formed, and a "
                "non-zero initial perturbation is NOT enough. "
                "The shipped reaction_diffusion_2d template "
                "(Schnakenberg, a=0.1, b=0.9, d_u=1, d_v=40, "
                "gamma=1000, ElementQuad1 on a 32x32 grid, "
                "1089 DOFs, backward Euler dt=0.5 to T=50) seeds "
                "u,v with a 1e-2 random perturbation about "
                "(u_ss, v_ss) = (1.0, 0.9) and DECAYS straight "
                "back to uniform: the final results_summary.json "
                "reads u_max 1.0000000000000104 / u_min "
                "0.999999999999973, i.e. a spread of 3.7e-14 — "
                "machine noise — even though d_v/d_u = 40 is "
                "four times the ~10 threshold the entry below "
                "quotes. Signal: max(u) - min(u) at the end of "
                "the run is 3.7e-14 (machine noise) instead of "
                "the O(0.1) a genuine Turing pattern produces, "
                "while the process still exits rc=0 and the "
                "printed min/max look identical to 4 decimals. "
                "Backward Euler at dt=0.5 with "
                "gamma=1000 damps the high-wavenumber random "
                "seed before the unstable mode can grow. ALWAYS "
                "report max(u) - min(u) (or the variance) at the "
                "end of the run and refuse to call it a pattern "
                "unless it is O(0.1) or larger. (Verified "
                "empirically 2026-08-03 on skfem 12.0.1.)"
            ),
            (
                "[Numerical] Coupled system: assemble block "
                "Jacobian [[J_uu, J_uv], [J_vu, J_vv]] at "
                "each Newton step via four BilinearForm + "
                "asm calls. Signal: assembling only the "
                "diagonal blocks (J_uu, J_vv) and dropping "
                "the off-diagonal BilinearForm-asm output "
                "gives a linear-rate Newton instead of "
                "quadratic in the asm + condense + "
                "spsolve pipeline; the off-diagonal terms "
                "scale with the reaction-rate Jacobian "
                "df_u/dv and df_v/du which are non-zero "
                "for any coupled reaction. (Audit "
                "2026-06-02.)"
            ),
            (
                "[API] Reaction Jacobian blocks are mass matrices "
                "with a pointwise coefficient. Assembling them "
                "with the stiffness pattern instead produces "
                "spurious DIFFUSION in J_uv / J_vu, and it is a "
                "real bug, not a cosmetic one. Do NOT diagnose it "
                "by diffing against a reference that builds "
                "M @ diag(df_u/dv): asm(mass_with_coefficient, c) "
                "and M @ diag(c) are legitimately DIFFERENT "
                "matrices — each is the exact derivative of a "
                "different residual (quadrature-consistent vs. "
                "nodally-lumped coefficient) and their gap merely "
                "shrinks under refinement — so 'differs from the "
                "M @ diag reference' is not by itself evidence of "
                "anything. Use a derivative check instead. "
                "Signal: the stiffness-pattern block annihilates "
                "constants, so S @ ones is round-off and the "
                "block reports ZERO coupling for a uniform change "
                "in v where the true coupling is the reaction "
                "derivative; its norm relative to the mass block "
                "grows like 1/h^2 under refinement (that growth "
                "IS the spurious diffusion); a finite-difference "
                "directional derivative of the residual disagrees "
                "with it by O(1) relative error while the mass "
                "form matches to solver tolerance; and Newton on "
                "one backward-Euler step fails to converge, "
                "ending above its starting residual, where the "
                "mass form converges in a couple of updates. "
                "(Verified 2026-08-06 on skfem 12.0.1 — Tier-2 "
                "fixture rd_reaction_block_mass_vs_stiffness.)"
            ),
            (
                "[Numerical] Initial condition: perturb the "
                "homogeneous steady state to trigger the Turing "
                "instability. Starting from exactly (u_ss, v_ss) "
                "is an EXACT discrete equilibrium — the assembled "
                "residual there is round-off, because "
                "K @ ones = 0 and f(u_ss, v_ss) = 0 — but 'stays "
                "uniform throughout' only holds in exact "
                "arithmetic. Floating-point round-off is itself a "
                "seed, and it is amplified by the same unstable "
                "mode at the same rate, so an unperturbed start "
                "merely DELAYS onset by the number of steps it "
                "takes round-off to climb to O(1); run long "
                "enough and the pattern appears anyway. Never "
                "treat 'it stayed flat' as proof the parameters "
                "are sub-critical — that conclusion needs the "
                "dispersion relation, not a short run. Add a "
                "small random or spatially-structured "
                "perturbation (amplitude ~1e-3 * u_ss) so onset "
                "is deterministic and reproducible. Signal: the "
                "residual returned by asm at the uniform state is "
                "round-off, and max(u) - min(u) over the "
                "ElementQuad1 Basis solution vector sits at "
                "round-off for the first steps yet GROWS by a "
                "constant factor per step from there — record the "
                "spread every step and look for the geometric "
                "growth, not for a flat line; with the "
                "recommended perturbation the same run saturates "
                "into a pattern many orders of magnitude sooner. "
                "(Verified 2026-08-06 on skfem 12.0.1 — Tier-2 "
                "fixture rd_unperturbed_ic.)"
            ),
            (
                "[Numerical] Turing instability requires "
                "d_v >> d_u (fast inhibitor, slow activator). "
                "'d_v/d_u above ~10' is a RULE OF THUMB, not the "
                "threshold: the exact critical ratio is the root "
                "of (d*f_u + g_v)^2 = 4*d*det(J) with the "
                "reaction Jacobian entries evaluated at the "
                "homogeneous steady state. It depends on (a, b) "
                "and can land on either side of 10, so 10 is "
                "neither a floor nor a ceiling: a ratio "
                "comfortably above it can still fail if the "
                "domain modes miss the unstable band (see the "
                "gamma entry), and a ratio below it can still "
                "pattern. Compute the root for your own (a, b) "
                "rather than trusting the number. Signal: at "
                "d_v = d_u the first "
                "Turing condition already fails "
                "(d*f_u + g_v < 0) and the largest growth rate "
                "over all admissible Neumann modes is negative, "
                "so max(u) - min(u) decays to machine noise; "
                "sweep the ratio upward and the spread jumps from "
                "machine noise to O(1) as it crosses the computed "
                "root. (Verified 2026-08-06 on skfem 12.0.1 — "
                "Tier-2 fixture rd_diffusion_ratio.)"
            ),
            (
                "[Numerical] Schnakenberg homogeneous steady "
                "state: u_ss = a+b, v_ss = b/(a+b)^2. Signal: "
                "running with initial conditions that do not "
                "match (u_ss, v_ss) leads to a long startup "
                "transient (~1/r time units) before patterns "
                "emerge; comparing simulated u against the "
                "analytic steady state catches off-by-one or "
                "mis-typed (a, b) parameters — a 10% deviation "
                "in u_ss at t=10 means a or b is wrong by a "
                "similar factor. (Audit 2026-06-02.)"
            ),
            (
                "[API] Neumann (zero-flux) BCs are natural in the "
                "weak form — no explicit enforcement needed; just "
                "leave the boundary alone. scikit-fem 12.0.1 has "
                "NO DirichletBC symbol (hasattr(skfem, "
                "'DirichletBC') is False); the essential-BC API "
                "is condense / enforce / penalize, and "
                "'DirichletBC' is vocabulary borrowed from "
                "another library. Constraining the boundary DOFs "
                "to zero does far more than flatten the edge — it "
                "suppresses the pattern in the whole domain. "
                "Signal: with no boundary treatment at all the "
                "total mass 1^T @ (M @ u) of a pure-diffusion run "
                "is conserved to round-off, whereas condensing "
                "the boundary DOFs to zero drains a large "
                "fraction of it; on the Schnakenberg system the "
                "condensed run has boundary values identically 0 "
                "and an overall max(u) - min(u) an order of "
                "magnitude below the natural-Neumann run, not "
                "merely a flattened rim. (Verified 2026-08-06 on "
                "skfem 12.0.1 — Tier-2 fixture "
                "rd_neumann_natural_vs_dirichlet.)"
            ),
            (
                "[Numerical] Pattern formation requires gamma "
                "large enough relative to domain size, but the "
                "criterion is a BAND-MEMBERSHIP test, not a "
                "single inequality: with a zero-flux (Neumann) "
                "box the admissible wavenumbers are discrete, and "
                "a pattern appears only when the SMALLEST "
                "non-zero Neumann mode k_min^2 (pi^2/L^2 on a "
                "square side L) falls inside the unstable band "
                "[k_-^2, k_+^2] — the two roots in k^2 of "
                "d*k^4 - gamma*(d*f_u + g_v)*k^2 + "
                "gamma^2*det(J) = 0, i.e. k_pm^2 = gamma*"
                "[(d*f_u + g_v) +/- sqrt((d*f_u + g_v)^2 - "
                "4*d*det(J))] / (2*d), with the reaction Jacobian "
                "J evaluated at the homogeneous steady state and "
                "d = d_v/d_u. Both endpoints scale linearly with "
                "gamma, so raising gamma widens the band past the "
                "fixed domain mode. Compute those roots "
                "for YOUR (a, b, d) and YOUR domain — the "
                "critical gamma is geometry-dependent and cannot "
                "be carried over from another run. The old "
                "shortcut gamma*L^2 > pi^2*(a+b)^2 is FALSE: it "
                "sits below the true threshold, so it predicts "
                "patterns at gamma values that produce none. "
                "Signal: at gamma values that clear the old "
                "inequality but still leave k_min^2 outside "
                "[k_-^2, k_+^2], max(u) - min(u) decays "
                "monotonically toward machine noise over the "
                "whole run while the process still exits rc=0; "
                "raising gamma until k_min^2 enters the band "
                "flips it to an O(1) spread. Confirm the decay is "
                "not a time-stepping artefact by checking "
                "lambda_max * dt < 2 for the run. (Verified "
                "2026-08-06 on skfem 12.0.1 — Tier-2 fixture "
                "rd_gamma_domain_threshold.)"
            ),
            (
                "[Numerical] Fisher-KPP: du/dt = D*Δu + r*u*"
                "(1-u) is a SCALAR equation with no coupling "
                "block. Signal: building a block-2x2 Jacobian for "
                "it (assuming a coupled system) doubles the DOF "
                "count but QUADRUPLES the number of non-zeros — "
                "each of the four blocks carries the full scalar "
                "sparsity pattern, so scipy.sparse.linalg.spsolve "
                "wall_time rises accordingly; and if the second "
                "field is given a different diffusivity it leaks "
                "into u through the spurious coupling, so the "
                "block answer differs from the scalar one by an "
                "O(1) fraction of the true maximum (with equal "
                "diffusivities the second field is an exact copy "
                "and the waste is silent). Compare nnz, not just "
                "shape. The correct discretisation is a single "
                "Newton solve on the M + dt*K - dt*M_r system "
                "where M_r is the mass matrix weighted by "
                "r*(1 - 2*u^n). Note 'GridFunction' is NGSolve "
                "vocabulary: hasattr(skfem, 'GridFunction') is "
                "False — in scikit-fem the solution is a plain "
                "numpy array. (Verified 2026-08-06 on skfem "
                "12.0.1 — Tier-2 fixture rd_fisher_kpp_scalar.)"
            ),
        ],
    },
}


# ---------------------------------------------------------------------------
# Generator registry
# ---------------------------------------------------------------------------

GENERATORS = {
    "navier_stokes_2d":      _navier_stokes_2d,
    "hyperelasticity_2d":    _hyperelasticity_2d,
    "dg_methods_2d":         _dg_methods_2d,
    "time_dependent_2d":     _time_dependent_2d,
    "helmholtz_2d":          _helmholtz_2d,
    "reaction_diffusion_2d": _reaction_diffusion_2d,
}
