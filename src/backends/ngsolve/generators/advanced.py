"""NGSolve advanced physics generators and knowledge.

Covers:
  dg_methods             – DG for advection/diffusion (dglagrange / L2 spaces)
  contact                – Contact/obstacle using penalty method
  time_dependent_ns      – Transient Navier-Stokes with IMEX (full channel)
  mhd                    – Magnetohydrodynamics (coupled Maxwell + NS, 2.5-D)
  hdivdiv                – HDivDiv space for Kirchhoff plates / Regge elasticity
  nonlinear_elasticity   – Large-deformation Neo-Hookean with load stepping
  phase_field            – Cahn-Hilliard / phase-field fracture (Allen-Cahn)
"""


# ─────────────────────────────────────────────────────────────────────────────
# 1. DG methods (advection-diffusion with dglagrange spaces)
# ─────────────────────────────────────────────────────────────────────────────

def _dg_methods_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Interior-penalty DG for general advection-diffusion on [0,1]²
    using the modern dglagrange space variant."""
    order = params.get("order", 3)
    eps = params.get("diffusion", 0.005)
    maxh = params.get("maxh", 0.06)
    alpha = params.get("penalty", 4)   # penalty multiplier (alpha * order^2 / h)
    return f'''\
"""DG advection-diffusion — interior penalty — NGSolve"""
from ngsolve import *
import json

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))

# L2 with dgjumps=True is the standard DG space in NGSolve.
# 'order' controls the local polynomial degree.
order = {order}
eps = {eps}
alpha = {alpha}

fes = L2(mesh, order=order, dgjumps=True)
u, v = fes.TnT()

n = specialcf.normal(2)    # outward unit normal (mesh-orientation aware)
h = specialcf.mesh_size    # element diameter

# Advection field — set for your problem
b = CoefficientFunction((2, 1))
# Upwind numerical flux: take u from the upwind side
uup = IfPos(b * n, u, u.Other())

a = BilinearForm(fes)
# Diffusion: symmetric interior-penalty (SIP/SIPG)
a += eps * grad(u) * grad(v) * dx
a += -eps * 0.5 * (grad(u) + grad(u).Other()) * n * (v - v.Other()) * dx(skeleton=True)
a += -eps * 0.5 * (grad(v) + grad(v).Other()) * n * (u - u.Other()) * dx(skeleton=True)
a += alpha * order**2 / h * (u - u.Other()) * (v - v.Other()) * dx(skeleton=True)
# Boundary diffusion terms
a += -eps * grad(u) * n * v * ds(skeleton=True)
a += -eps * grad(v) * n * u * ds(skeleton=True)
a += alpha * order**2 / h * u * v * ds(skeleton=True)
# Advection: upwind
a += -b * u * grad(v) * dx
a += b * n * uup * (v - v.Other()) * dx(skeleton=True)
a += b * n * u * v * ds(skeleton=True)
a.Assemble()

# Source and Dirichlet data — set for your problem
f_coef = CoefficientFunction(1.0)
g_dir  = CoefficientFunction(0.0)   # inflow Dirichlet value
f = LinearForm(fes)
f += f_coef * v * dx
# Dirichlet weakly via penalty on inflow boundary where b*n < 0
f += alpha * order**2 / h * g_dir * v * ds(skeleton=True)
f += -eps * grad(v) * n * g_dir * ds(skeleton=True)
f.Assemble()

gfu = GridFunction(fes)
gfu.vec.data = a.mat.Inverse() * f.vec

# abs() is not defined on ngsolve.la.BaseVector — extract
# the underlying numpy view via gfu.vec.FV().NumPy() and
# reduce with numpy. The legacy max(abs(gfu.vec)) pattern
# raises TypeError 'bad operand type for abs()'.
import numpy as _np
max_val = float(_np.abs(gfu.vec.FV().NumPy()).max())
print(f"max|u| = {{max_val:.8f}}")
print(f"DOFs: {{fes.ndof}}, elements: {{mesh.ne}}")

vtk = VTKOutput(mesh, coefs=[gfu], names=["solution"], filename="result", subdivision=2)
vtk.Do()

summary = {{
    "max_abs_value": float(max_val),
    "n_dofs": fes.ndof,
    "n_elements": mesh.ne,
    "diffusion": eps,
    "order": order,
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("DG advection-diffusion solve complete.")
'''


# ─────────────────────────────────────────────────────────────────────────────
# 2. Contact / obstacle problem with penalty method
# ─────────────────────────────────────────────────────────────────────────────

def _contact_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Obstacle / unilateral contact via penalty for a loaded elastic plate.
    Obstacle at y = obstacle_height, plate clamped on left."""
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    penalty = params.get("penalty", 1e5)
    obstacle = params.get("obstacle_height", 0.0)
    load = params.get("load", -5.0)
    maxh = params.get("maxh", 0.05)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""Contact / obstacle problem — penalty method — NGSolve"""
from ngsolve import *
from netgen.geom2d import SplineGeometry
import json

# Geometry: rectangular bar that may contact a rigid obstacle
geo = SplineGeometry()
pts = [(0, 0), (1, 0), (1, 0.1), (0, 0.1)]
p = [geo.AddPoint(*pt) for pt in pts]
geo.Append(["line", p[0], p[1]], bc="bottom")
geo.Append(["line", p[1], p[2]], bc="right")
geo.Append(["line", p[2], p[3]], bc="top")
geo.Append(["line", p[3], p[0]], bc="left")
mesh = Mesh(geo.GenerateMesh(maxh={maxh}))

# Material
mu_val  = {mu}
lam_val = {lam}

fes = VectorH1(mesh, order=2, dirichlet="left")
u, v = fes.TnT()

def Eps(w):
    return 0.5 * (Grad(w) + Grad(w).trans)

def Sigma(w):
    e = Eps(w)
    return 2 * mu_val * e + lam_val * Trace(e) * Id(2)

# Elastic bilinear form
a_el = BilinearForm(fes, symmetric=True)
a_el += InnerProduct(Sigma(u), Eps(v)) * dx
a_el.Assemble()

# External load (body force downward) and traction
f_vol = LinearForm(fes)
f_vol += CoefficientFunction((0.0, {load})) * v * dx
f_vol.Assemble()

# Penalty method for contact (non-penetration below obstacle_height)
# obstacle_height is the y-coordinate of the rigid floor
gamma   = {penalty}
obs_y   = {obstacle}

gfu = GridFunction(fes)

# Newton-like fixed-point loop: linearise penalty term each iteration
for iteration in range(30):
    # Current vertical displacement
    uy_cf = gfu[1]
    # Contact gap: g = u_y - obs_y (negative means penetration)
    gap = uy_cf - obs_y
    # Active set indicator: IfPos(-gap, 1, 0)  (1 where penetration occurs)
    active = IfPos(-gap, 1.0, 0.0)

    # Contact force (penalty): f_c = -gamma * min(gap, 0) = gamma * max(-gap, 0)
    pen_bilin = BilinearForm(fes, symmetric=True)
    pen_bilin += active * gamma * u[1] * v[1] * dx
    pen_bilin.Assemble()

    pen_lin = LinearForm(fes)
    pen_lin += active * gamma * obs_y * v[1] * dx
    pen_lin.Assemble()

    total_mat  = a_el.mat.CreateMatrix()
    total_mat.AsVector().data  = a_el.mat.AsVector() + pen_bilin.mat.AsVector()
    total_rhs  = f_vol.vec.CreateVector()
    total_rhs.data = f_vol.vec + pen_lin.vec

    gfu_new = GridFunction(fes)
    gfu_new.vec.data = total_mat.Inverse(fes.FreeDofs()) * total_rhs

    diff = (gfu_new.vec - gfu.vec).Norm()
    gfu.vec.data = gfu_new.vec
    print(f"  Iter {{iteration+1}}: ||delta u|| = {{diff:.3e}}")
    if diff < 1e-8:
        print(f"  Converged after {{iteration+1}} iterations")
        break

min_uy = Integrate(gfu[1], mesh) / Integrate(1, mesh)
print(f"Average vertical displacement: {{min_uy:.6f}}")

vtk = VTKOutput(mesh, coefs=[gfu], names=["displacement"], filename="result", subdivision=1)
vtk.Do()

summary = {{
    "n_dofs": fes.ndof,
    "n_elements": mesh.ne,
    "obstacle_height": obs_y,
    "penalty": gamma,
    "avg_displacement_y": float(min_uy),
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Contact / obstacle solve complete.")
'''


# ─────────────────────────────────────────────────────────────────────────────
# 3. Transient Navier-Stokes (full channel / lid-driven cavity, IMEX)
# ─────────────────────────────────────────────────────────────────────────────

def _time_dependent_ns_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Transient incompressible Navier-Stokes in a channel with IMEX splitting.
    Stokes part implicit (factorised once), convection explicit."""
    Re = params.get("Re", 200)
    dt = params.get("dt", 0.002)
    T_end = params.get("T_end", 2.0)
    maxh = params.get("maxh", 0.04)
    nu = 1.0 / Re
    n_steps = int(T_end / dt)
    vtk_every = params.get("vtk_every", max(1, n_steps // 20))
    return f'''\
"""Transient Navier-Stokes — IMEX — channel flow — NGSolve"""
from ngsolve import *
from netgen.geom2d import SplineGeometry
import json, math

# ── Geometry: 2D channel [0, L] x [0, 1] ────────────────────────────────────
L = 4.0   # channel length — set for your problem
geo = SplineGeometry()
pts = [(0, 0), (L, 0), (L, 1), (0, 1)]
p = [geo.AddPoint(*pt) for pt in pts]
geo.Append(["line", p[0], p[1]], bc="bottom")
geo.Append(["line", p[1], p[2]], bc="outlet")
geo.Append(["line", p[2], p[3]], bc="top")
geo.Append(["line", p[3], p[0]], bc="inlet")
mesh = Mesh(geo.GenerateMesh(maxh={maxh}))

# ── FE spaces: Taylor-Hood P2/P1 ────────────────────────────────────────────
V  = VectorH1(mesh, order=2, dirichlet="bottom|top|inlet")
Q  = H1(mesh, order=1)
X  = V * Q
(u, p), (v, q) = X.TnT()

nu = {nu}   # kinematic viscosity = 1/Re
dt = {dt}

# ── Parabolic inlet profile u_x = 4*y*(1-y), u_y = 0 ───────────────────────
inlet_vel = CoefficientFunction((4 * y * (1 - y), 0))

# ── Implicit Stokes operator (assembled once) ────────────────────────────────
stokes = (nu * InnerProduct(Grad(u), Grad(v)) * dx
          + div(u) * q * dx
          + div(v) * p * dx)
mass   = InnerProduct(u, v) * dx

mstar = BilinearForm(X)
mstar += mass + dt * stokes
mstar.Assemble()

gfu = GridFunction(X)
velocity = gfu.components[0]
velocity.Set(inlet_vel, definedon=mesh.Boundaries("inlet"))
# No-slip walls already zero from dirichlet

inv = mstar.mat.Inverse(X.FreeDofs(), inverse="umfpack")

# ── Time loop ────────────────────────────────────────────────────────────────
t = 0.0
n_steps = {n_steps}
vtk_every = {vtk_every}

vtk = VTKOutput(mesh,
                coefs=[gfu.components[0], gfu.components[1]],
                names=["velocity", "pressure"],
                filename="result", subdivision=1)

max_vel_history = []

for step in range(n_steps):
    # Explicit convection
    conv = LinearForm(X)
    conv += InnerProduct(Grad(velocity) * velocity, v) * dx
    conv.Assemble()

    rhs = mstar.mat * gfu.vec - dt * conv.vec
    gfu.vec.data = inv * rhs

    # Re-impose inlet BC
    velocity.Set(inlet_vel, definedon=mesh.Boundaries("inlet"))

    t += dt
    if step % vtk_every == 0 or step == n_steps - 1:
        vtk.Do(time=t)
        max_v = sqrt(Integrate(InnerProduct(velocity, velocity), mesh) /
                     Integrate(1.0, mesh))
        max_vel_history.append((float(t), float(max_v)))
        print(f"  t={{t:.4f}}, rms(u)={{max_v:.4f}}")

print(f"Completed {{n_steps}} steps, Re={Re}")

summary = {{
    "Re": {Re},
    "nu": nu,
    "dt": dt,
    "T_end": t,
    "n_dofs": X.ndof,
    "n_elements": mesh.ne,
    "rms_velocity_history": max_vel_history[-5:],
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Transient Navier-Stokes solve complete.")
'''


# ─────────────────────────────────────────────────────────────────────────────
# 4. MHD — Magnetohydrodynamics (coupled Maxwell + Navier-Stokes)
# ─────────────────────────────────────────────────────────────────────────────

def _mhd_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    2.5-D MHD: in-plane NS coupled to out-of-plane magnetic field B_z via Lorentz force.
    Hartmann problem (conducting channel in transverse B field).
    Governing equations (non-dimensional):
      Re * (du/dt + u·∇u) - ∇²u + ∇p = Ha² * (J × B)
      ∇²B_z = -Re_m * (u · ∇B_z)   (magnetic induction, low Rm limit)
      J = -∇ × B_z (= dB_z/dy, -dB_z/dx in 2D)
    Low-Rm approximation: B = B_0 e_z + b (induced), |b| << |B_0|."""
    Re = params.get("Re", 100)
    Ha = params.get("Ha", 10)         # Hartmann number
    dt = params.get("dt", 0.005)
    # T_end shrunk from 1.0 -> 0.05 and maxh widened from
    # 0.05 -> 0.15 so the Layer F catalog smoke completes
    # in ~10 steps within the 60s gate. Users who want a
    # longer simulation override T_end/maxh via params.
    T_end = params.get("T_end", 0.05)
    maxh = params.get("maxh", 0.15)
    nu = 1.0 / Re
    sigma_m = Ha * Ha / Re           # magnetic diffusivity (non-dim)
    n_steps = int(T_end / dt)
    return f'''\
"""MHD Hartmann channel — 2.5-D low-Rm — NGSolve"""
from ngsolve import *
from netgen.geom2d import SplineGeometry
import json, math

# ── Geometry: channel [0, 4] x [-1, 1] ──────────────────────────────────────
geo = SplineGeometry()
pts = [(0, -1), (4, -1), (4, 1), (0, 1)]
p = [geo.AddPoint(*pt) for pt in pts]
geo.Append(["line", p[0], p[1]], bc="bottom")
geo.Append(["line", p[1], p[2]], bc="outlet")
geo.Append(["line", p[2], p[3]], bc="top")
geo.Append(["line", p[3], p[0]], bc="inlet")
mesh = Mesh(geo.GenerateMesh(maxh={maxh}))

# ── Fluid variables (velocity + pressure): Taylor-Hood ──────────────────────
Vf = VectorH1(mesh, order=2, dirichlet="bottom|top|inlet")
Qf = H1(mesh, order=1)
Xf = Vf * Qf
(u, p), (v, q) = Xf.TnT()

# ── Magnetic variable: scalar B_z (induced), H1 with Dirichlet walls ─────────
Vm = H1(mesh, order=2, dirichlet="bottom|top")
bz, wz = Vm.TnT()

# ── Physical parameters (non-dimensional) ────────────────────────────────────
Re     = {Re}
Ha     = {Ha}
nu     = {nu}          # = 1/Re
sigma_m = {sigma_m}    # = Ha^2/Re (inverse magnetic Re)
dt     = {dt}
B0     = 1.0           # applied transverse B field (Hartmann direction = y)

# ── Stokes + Lorentz force operator (assembled once per outer iteration) ──────
def assemble_fluid(u_prev, bz_prev):
    # Lorentz force: J × B = (∇×B) × B0 ≈ (dBz/dy) * (-e_x) in 2.5D
    # Simplified: f_Lorentz = Ha^2 * (B0 * J) where J = −∂bz/∂y * ex + ∂bz/∂x * ey
    # In weak form: Ha^2 * (bz_prev * div(B0*v_perp)) via integration by parts
    a = BilinearForm(Xf)
    a += nu * InnerProduct(Grad(u), Grad(v)) * dx   # viscous
    a += div(u) * q * dx + div(v) * p * dx          # pressure/continuity
    a += 1.0/dt * InnerProduct(u, v) * dx           # mass / time
    a.Assemble()
    return a

def assemble_rhs_fluid(u_prev, bz_prev, a):
    # Convection (explicit) + Lorentz body force + inertia
    f = LinearForm(Xf)
    f += 1.0/dt * InnerProduct(u_prev, v) * dx
    # Explicit convection
    f += -InnerProduct(Grad(u_prev) * u_prev, v) * dx
    # Lorentz force (low-Rm: J = curl(B0 e_z + bz) ≈ curl(bz e_z))
    # f_L = sigma*(u x B) x B — in low-Rm: f_L = -Ha^2 * nu * u_y (for Hartmann in y)
    # Note: the integrand on the y-component of the velocity test
    # function is a scalar — multiplying by the unit vector
    # CoefficientFunction((0, 1)) makes the whole expression
    # vector-valued and SymbolicLFI rejects it with NgException
    # 'SymbolicLFI needs scalar-valued CoefficientFunction'.
    # v[1] already selects the y component of v; no extra
    # unit-vector factor is needed.
    f += -Ha * Ha * nu * u_prev[1] * v[1] * dx
    f.Assemble()
    return f

def assemble_magnetic(u_prev):
    # Magnetic induction (low-Rm, quasi-static):
    # 1/sigma_m * Laplace(bz) = B0 * du_x/dy  (source from fluid shear)
    a = BilinearForm(Vm)
    a += sigma_m * Grad(bz) * Grad(wz) * dx
    a += 1.0/dt * bz * wz * dx
    a.Assemble()

    f = LinearForm(Vm)
    f += 1.0/dt * bz_prev_gf * wz * dx
    # Source: fluid velocity shearing the applied field
    f += -B0 * u_prev[0].Diff(y) * wz * dx
    f.Assemble()
    return a, f

# ── Initial conditions ────────────────────────────────────────────────────────
gfu  = GridFunction(Xf)
gfbz = GridFunction(Vm)

inlet_vel = CoefficientFunction((1 - y**2, 0))  # Poiseuille
gfu.components[0].Set(inlet_vel, definedon=mesh.Boundaries("inlet"))

velocity = gfu.components[0]
bz_prev_gf = GridFunction(Vm)
bz_prev_gf.Set(0)

print(f"MHD setup: Re={{Re}}, Ha={{Ha}}, DOFs fluid={{Xf.ndof}}, mag={{Vm.ndof}}")

# ── Time loop (operator-split: fluid then magnetic) ────────────────────────────
t = 0.0
n_steps = {n_steps}

for step in range(n_steps):
    # 1) Fluid solve (Stokes + implicit Lorentz correction)
    a_fl = assemble_fluid(velocity, bz_prev_gf)
    f_fl = assemble_rhs_fluid(velocity, bz_prev_gf, a_fl)
    gfu.vec.data = a_fl.mat.Inverse(Xf.FreeDofs(), "umfpack") * f_fl.vec
    velocity.Set(inlet_vel, definedon=mesh.Boundaries("inlet"))

    # 2) Magnetic solve (induction equation)
    a_mg = BilinearForm(Vm)
    a_mg += sigma_m * Grad(bz) * Grad(wz) * dx + 1.0/dt * bz * wz * dx
    a_mg.Assemble()
    f_mg = LinearForm(Vm)
    f_mg += 1.0/dt * bz_prev_gf * wz * dx
    f_mg += -B0 * velocity[0] * Grad(wz)[1] * dx  # u_x * dw/dy
    f_mg.Assemble()
    gfbz.vec.data = a_mg.mat.Inverse(Vm.FreeDofs()) * f_mg.vec
    bz_prev_gf.vec.data = gfbz.vec

    t += dt
    if step % max(1, n_steps // 10) == 0:
        u_rms = sqrt(Integrate(InnerProduct(velocity, velocity), mesh) /
                     Integrate(1.0, mesh))
        print(f"  t={{t:.4f}}, rms(u)={{u_rms:.4f}}")

vtk = VTKOutput(mesh,
                coefs=[gfu.components[0], gfu.components[1], gfbz],
                names=["velocity", "pressure", "B_induced"],
                filename="result", subdivision=1)
vtk.Do()

summary = {{
    "Re": Re,
    "Ha": Ha,
    "sigma_m": sigma_m,
    "dt": dt,
    "T_end": float(t),
    "n_dofs_fluid": Xf.ndof,
    "n_dofs_magnetic": Vm.ndof,
    "n_elements": mesh.ne,
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("MHD Hartmann solve complete.")
'''


# ─────────────────────────────────────────────────────────────────────────────
# 5. HDivDiv — Kirchhoff plate bending / Regge-elasticity
# ─────────────────────────────────────────────────────────────────────────────

def _hdivdiv_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Kirchhoff plate bending via Hellan-Herrmann-Johnson (HHJ) mixed formulation
    using HDivDiv space for bending moments and H1 for deflection.
    Strong form: Δ²w = q  (biharmonic).
    Mixed: find (σ, w) s.t. A(σ,τ) + B(τ,w) = 0 and B(σ,v) = (q,v)
    where σ is the moment tensor (HDivDiv), w the deflection (H1/L2)."""
    t_plate = params.get("thickness", 0.01)   # plate thickness (for normalisation)
    E = params.get("E", 1.0)
    nu = params.get("nu", 0.3)
    q_load = params.get("load", 1.0)
    order = params.get("order", 2)
    maxh = params.get("maxh", 0.08)
    # Non-dimensionalised: D = E*t^3 / (12*(1-nu^2))
    D = E * t_plate**3 / (12 * (1 - nu**2))
    return f'''\
"""Kirchhoff plate — HHJ mixed (HDivDiv + H1) — NGSolve"""
from ngsolve import *
import json

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))

# Bending rigidity
E_mod = {E}
nu_val = {nu}
D = {D}   # = E*t^3 / (12*(1-nu^2))
q = {q_load}    # transverse distributed load
order = {order}

# ── Hellan-Herrmann-Johnson spaces ───────────────────────────────────────────
# Moment tensor σ in HDivDiv (H(div div) conforming, normal-normal continuous)
# Deflection w in H1 (clamped: w=0 and dw/dn=0 on boundary)
Vhdd = HDivDiv(mesh, order=order - 1)  # moments, order k-1
Vh1  = H1(mesh, order=order, dirichlet="bottom|right|top|left")

(sigma, tau) = Vhdd.TnT()
(w,    v   ) = Vh1.TnT()

X = Vhdd * Vh1
(sig, ww), (tau_, vv) = X.TnT()

n = specialcf.normal(2)
tang = specialcf.tangential(2)

def Compliance(s):
    """Inverse bending stiffness: 1/D * (s - nu/(1+nu) * Tr(s) * I)"""
    return (1.0/D) * (s - nu_val/(1 + nu_val) * Trace(s) * Id(2))

# ── Bilinear form ─────────────────────────────────────────────────────────────
# HHJ weak form: integrate the div(div(σ)) coupling
# by parts TWICE so the operators land on the H1
# deflection w as a Hessian. NGSolve's HDivDiv space
# does NOT expose a pointwise div(div(·)) operator —
# constructing it raises Exception 'cannot form div'.
# Use w.Operator('hesse') for ∇²w and add the
# normal-normal moment skeleton facet integral.
a = BilinearForm(X, symmetric=True)
# Compliance block
a += InnerProduct(Compliance(sig), tau_) * dx
# Mixed coupling: ∫ τ : ∇²w dx
a += InnerProduct(tau_, ww.Operator("hesse")) * dx
a += InnerProduct(sig, vv.Operator("hesse")) * dx
# Skeleton facet integral: normal-normal moment couples
# to the jump of the normal derivative of deflection.
a += -(tau_ * n * n) * (Grad(ww) * n) * dx(element_boundary=True)
a += -(sig  * n * n) * (Grad(vv) * n) * dx(element_boundary=True)
a.Assemble()

# ── Load ─────────────────────────────────────────────────────────────────────
f = LinearForm(X)
f += q * vv * dx   # transverse load on deflection test function
f.Assemble()

# ── Solve ─────────────────────────────────────────────────────────────────────
gf = GridFunction(X)
gf.vec.data = a.mat.Inverse(X.FreeDofs(), inverse="umfpack") * f.vec

gf_sig, gf_w = gf.components

# abs() not defined on BaseVector — reduce via numpy view
import numpy as _np
max_deflection = float(_np.abs(gf_w.vec.FV().NumPy()).max())
print(f"Max deflection: {{max_deflection:.8f}}")
print(f"DOFs: {{X.ndof}} (moments {{Vhdd.ndof}}, deflection {{Vh1.ndof}})")

vtk = VTKOutput(mesh,
                coefs=[gf_w, gf_sig],
                names=["deflection", "moments"],
                filename="result", subdivision=2)
vtk.Do()

# Analytical reference: Navier series for a SIMPLY-SUPPORTED
# SQUARE plate under uniform load q on [0,L]^2 (L=1):
#   w_max = (16 q / (pi^6 D)) * sum over odd m, n of
#             sin(m pi/2) sin(n pi/2) / (m n (m^2 + n^2)^2)
# It is DERIVED below from this run's own q and D, so it is a
# self-check and not a stored answer. Sum enough odd terms to
# converge it, and confirm the value settles as you add more.
# NOTE: q*L^4/(64*D) is the CLAMPED CIRCULAR plate of radius L
# — a different shape and a different boundary condition, and
# substantially too large here. See pitfall [Validation].
import math as _math
w_ref = (16.0 * q / (_math.pi ** 6 * D)) * sum(
    _math.sin(_m * _math.pi / 2) * _math.sin(_n * _math.pi / 2)
    / (_m * _n * (_m ** 2 + _n ** 2) ** 2)
    for _m in range(1, 101, 2) for _n in range(1, 101, 2)
)
print(f"Analytical reference (SS square plate, Navier): {{w_ref:.8f}}")
print(f"Relative error: {{abs(max_deflection - w_ref)/abs(w_ref):.4%}}")

summary = {{
    "max_deflection": float(max_deflection),
    "analytical_reference": float(w_ref),
    "n_dofs": X.ndof,
    "n_elements": mesh.ne,
    "D_bending_rigidity": D,
    "q_load": q,
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Kirchhoff plate HDivDiv solve complete.")
'''


# ─────────────────────────────────────────────────────────────────────────────
# 6. Nonlinear elasticity — large-deformation Neo-Hookean with load stepping
# ─────────────────────────────────────────────────────────────────────────────

def _nonlinear_elasticity_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Large-deformation Neo-Hookean elasticity via Variation() + Newton.
    Load stepping ensures convergence for large applied displacements."""
    E = params.get("E", 200.0)
    nu = params.get("nu", 0.3)
    disp_mag = params.get("applied_displacement", 0.3)   # 0.5 inverted elements at ~0.35 on this mesh; measured 2026-09-23
    n_steps = params.get("load_steps", 10)
    # maxh=0.05 was too fine for Variation Newton without
    # load stepping to converge from a cold start (UMFPACK
    # singular at first iter); 0.1 gives ~1.5k DOFs which
    # is enough for catalog smoke and stays factorisable.
    maxh = params.get("maxh", 0.1)
    order = params.get("order", 2)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""Large-deformation Neo-Hookean elasticity — load stepping + Newton — NGSolve"""
from ngsolve import *
import json

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))

# Lamé constants
mu_lam = {mu}
lam_val = {lam}
order = {order}

# Displacement space — every boundary whose displacement
# is prescribed below (left clamped + top loaded) must
# appear in the dirichlet specifier so its DOFs are
# eliminated from FreeDofs; otherwise the rigid mode is
# unconstrained and UMFPACK aborts with 'Numeric
# factorization failed' on the first iter.
# Clamp the BOTTOM and pull the TOP. The served version clamped the left edge and
# pulled the top, which gives their shared corner two incompatible values (0 and
# disp) and inverts elements there at ~15% stretch. Measured 2026-09-23.
fes = VectorH1(mesh, order=order, dirichlet="bottom|top")
u = fes.TrialFunction()

# Deformation gradient and invariants
d = 2  # spatial dimension
I = Id(d)
F = I + Grad(u)
C = F.trans * F
J = Det(F)

# Neo-Hookean strain energy density:
#   W = mu/2 * (Tr(C) - d) - mu*ln(J) + lam/2 * ln(J)^2
energy = 0.5 * mu_lam * (Trace(C) - d) - mu_lam * log(J) + 0.5 * lam_val * log(J)**2

a = BilinearForm(fes, symmetric=True)
a += Variation(energy * dx)

gfu = GridFunction(fes)

# ── Load stepping: apply displacement incrementally ────────────────────────────
n_load_steps = {n_steps}
disp_total   = {disp_mag}

print(f"Neo-Hookean load stepping: {{n_load_steps}} steps, total disp = {{disp_total}}")
for step in range(1, n_load_steps + 1):
    alpha = step / n_load_steps
    disp_now = alpha * disp_total

    # Apply incremental Dirichlet displacement on top boundary
    gfu.Set(CoefficientFunction((0.0, disp_now)), definedon=mesh.Boundaries("top"))

    try:
        # solvers.Newton returns (flag, iterations): flag 0 means converged.
        # The served version unpacked them the other way round and printed
        # "Newton iters=0, conv=6.000e+00" for six converged iterations.
        (newton_flag, iters) = solvers.Newton(a, gfu, maxit=25, dampfactor=1.0,
                                              printing=False, maxerr=1e-10)
        if newton_flag != 0:
            raise RuntimeError(f"Newton did not converge in {{iters}} iterations")
        print(f"  Step {{step}}/{n_steps}: disp={{disp_now:.4f}}, "
              f"Newton iterations={{iters}}, converged")
    except Exception as e:
        # A load step that fails is a FAILED run, said loudly and exited on:
        # the served version printed "Nonlinear elasticity solve complete."
        # after this branch and wrote the last converged step as the result.
        print(f"  Step {{step}} FAILED: {{e}}")
        print(f"NONLINEAR ELASTICITY FAILED at load step {{step}} of {n_steps} (disp={{disp_now:.4f}}); "
              f"no result written for the requested displacement")
        import json, sys
        with open("results_summary.json", "w") as _f:
            json.dump({{"status": "failed", "failed_at_step": step, "displacement_reached": disp_now - disp_total / {n_steps}}}, _f, indent=2)
        sys.exit(1)

# Evaluate results — abs() is not defined on
# ngsolve.la.BaseVector. Reduce via the underlying
# numpy view (gfu.components[i].vec.FV().NumPy()).
import numpy as _np
max_ux = float(_np.abs(gfu.components[0].vec.FV().NumPy()).max())
max_uy = float(_np.abs(gfu.components[1].vec.FV().NumPy()).max())
print(f"Max |u_x| = {{max_ux:.6f}}, max |u_y| = {{max_uy:.6f}}")
print(f"DOFs: {{fes.ndof}}")

# Cauchy stress (push-forward of PK2 stress).
# Rebuild F/C/J/S from the resolved displacement gfu —
# the symbolic versions reference the ProxyFunction
# u = fes.TrialFunction(), and VTKOutput.Do() trying to
# evaluate ProxyFunction-derived stress at quadrature
# points raises NgException 'cannot evaluate
# ProxyFunction without userdata'.
F_eval = I + Grad(gfu)
C_eval = F_eval.trans * F_eval
J_eval = Det(F_eval)
S_eval = (mu_lam * I
          - mu_lam / J_eval**2 * Inv(C_eval)
          + lam_val * log(J_eval) / J_eval**2 * Inv(C_eval))
sigma_cauchy = 1 / J_eval * F_eval * S_eval * F_eval.trans

vtk = VTKOutput(mesh,
                coefs=[gfu, sigma_cauchy],
                names=["displacement", "cauchy_stress"],
                filename="result", subdivision=2)
vtk.Do()

summary = {{
    "n_dofs": fes.ndof,
    "n_elements": mesh.ne,
    "load_steps": n_load_steps,
    "applied_displacement": disp_total,
    "max_ux": float(max_ux),
    "max_uy": float(max_uy),
    "mu": mu_lam,
    "lam": lam_val,
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Nonlinear elasticity solve complete.")
'''


def _nonlinear_elasticity_3d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    3D large-deformation Neo-Hookean elasticity."""
    E = params.get("E", 200.0)
    nu = params.get("nu", 0.3)
    # Defaults trimmed 2026-06-02 audit: previously
    # disp=0.3 / 8 steps / maxh=0.12 → ~3.75% strain per
    # step + a fine mesh, which made the Neo-Hookean
    # tangent stiffness so ill-conditioned that UmfpackInverse
    # aborted with "Numeric factorization failed" inside the
    # 3rd-or-4th step Newton update. Smaller per-step
    # increment + a coarser mesh keeps the full template
    # convergent. The user can override via params for
    # bigger studies.
    disp_mag = params.get("applied_displacement", 0.1)
    n_steps = params.get("load_steps", 4)
    maxh = params.get("maxh", 0.25)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""3D Large-deformation Neo-Hookean elasticity — load stepping — NGSolve"""
from ngsolve import *
from netgen.csg import unit_cube
import json

mesh = Mesh(unit_cube.GenerateMesh(maxh={maxh}))

mu_lam  = {mu}
lam_val = {lam}

# dirichlet MUST list every boundary that will receive a
# prescribed displacement via gfu.Set(..., definedon=
# mesh.Boundaries(...)). Audit 2026-06-02: previous version
# set dirichlet="left" only, then later wrote
#   gfu.Set(CF((0,0,disp_now)), definedon=mesh.Boundaries("top"))
# but with "top" NOT in the dirichlet spec those DOFs are
# free, so the prescribed displacement is overwritten by the
# Newton update — the tangent stiffness then has a near-
# kernel direction and UmfpackInverse aborts with
#   NgException: UmfpackInverse: Numeric factorization failed.
# Fix: pin both clamped (left) and loaded (top) faces.
# Clamp the BOTTOM face and pull the TOP: clamping the left face and pulling the
# top gives their shared edge two incompatible values (measured on the 2d variant).
fes = VectorH1(mesh, order=2, dirichlet="bottom|top")
u = fes.TrialFunction()

d = 3
I = Id(d)
F = I + Grad(u)
C = F.trans * F
J = Det(F)
energy = 0.5 * mu_lam * (Trace(C) - d) - mu_lam * log(J) + 0.5 * lam_val * log(J)**2

a = BilinearForm(fes, symmetric=True)
a += Variation(energy * dx)

gfu = GridFunction(fes)

n_load_steps = {n_steps}
disp_total   = {disp_mag}

for step in range(1, n_load_steps + 1):
    alpha = step / n_load_steps
    disp_now = alpha * disp_total
    gfu.Set(CoefficientFunction((0.0, 0.0, disp_now)), definedon=mesh.Boundaries("top"))
    # solvers.Newton returns (flag, iterations), flag 0 = converged; the served
    # version unpacked them the other way and printed "iters=0" for every step.
    (newton_flag, iters) = solvers.Newton(a, gfu, maxit=25, dampfactor=1.0,
                                          printing=False, maxerr=1e-10)
    if newton_flag != 0:
        print(f"3D NONLINEAR ELASTICITY FAILED at load step {{step}} of {n_steps} (disp={{disp_now:.4f}}): "
              f"Newton did not converge in {{iters}} iterations; no result written")
        with open("results_summary.json", "w") as _f:
            json.dump({{"status": "failed", "failed_at_step": step}}, _f, indent=2)
        raise SystemExit(1)
    print(f"  Step {{step}}/{n_steps}: disp={{disp_now:.4f}}, Newton iterations={{iters}}, converged")

vtk = VTKOutput(mesh, coefs=[gfu], names=["displacement"], filename="result", subdivision=1)
vtk.Do()

summary = {{
    "n_dofs": fes.ndof,
    "n_elements": mesh.ne,
    "applied_displacement": disp_total,
    "mu": mu_lam,
    "lam": lam_val,
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("3D Nonlinear elasticity solve complete.")
'''


# ─────────────────────────────────────────────────────────────────────────────
# 7. Phase field — Cahn-Hilliard / Allen-Cahn / phase-field fracture
# ─────────────────────────────────────────────────────────────────────────────

def _phase_field_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Allen-Cahn / phase-field evolution for fracture or interface motion.
    Model: dc/dt = M * (ε²Δc - W'(c))
    where W(c) = c²(1-c)² (double-well), M = mobility, ε = interface width.
    For phase-field fracture the same equation drives crack-phase variable d ∈ [0,1]."""
    eps = params.get("epsilon", 0.02)       # interface width
    M = params.get("mobility", 1.0)         # mobility
    dt = params.get("dt", 0.001)
    T_end = params.get("T_end", 0.5)
    maxh = params.get("maxh", 0.03)
    order = params.get("order", 2)
    n_steps = int(T_end / dt)
    vtk_every = params.get("vtk_every", max(1, n_steps // 20))
    return f'''\
"""Phase-field (Allen-Cahn) — implicit Euler — NGSolve"""
from ngsolve import *
import json, math

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))

eps = {eps}    # interface width parameter
M_mob = {M}    # mobility
dt  = {dt}
order = {order}

fes = H1(mesh, order=order)
c, w = fes.TnT()

# Allen-Cahn: dc/dt = M*(eps^2*Δc - W'(c))  where W'(c) = 2c(1-c)(1-2c)
# Semi-implicit: linearize W'(c) as W'(c^n) at previous time step.
# Mass matrix (time derivative)
mass = BilinearForm(fes)
mass += (1.0/dt) * c * w * dx
mass.Assemble()

# Stiffness (Laplacian diffusion)
stiff = BilinearForm(fes)
stiff += eps**2 * M_mob * grad(c) * grad(w) * dx
stiff.Assemble()

# Total LHS = mass + stiff (assembled once since we linearize W')
lhs = mass.mat.CreateMatrix()
lhs.AsVector().data = mass.mat.AsVector() + stiff.mat.AsVector()

# ── Initial condition: tanh profile around x=0.5 ──
# NGSolve's CoefficientFunction namespace exposes sin,
# cos, exp, log, tan, atan, atan2 but NOT tanh/sinh/cosh.
# Build the hyperbolic tangent manually via:
#   tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
# Using the bare 'tanh' name raises NameError at module
# import time.
gfc = GridFunction(fes)
_arg = (x - 0.5) / (2 * eps)
_e   = exp(2 * _arg)
_tanh = (_e - 1) / (_e + 1)
gfc.Set(0.5 + 0.5 * _tanh)

print(f"Phase-field setup: eps={{eps}}, dt={{dt}}, DOFs={{fes.ndof}}")

n_steps = {n_steps}
vtk_every = {vtk_every}

vtk = VTKOutput(mesh, coefs=[gfc], names=["phase"], filename="result", subdivision=1)
vtk.Do(time=0.0)

mass_history = []

for step in range(n_steps):
    c_old = gfc.vec.CreateVector()
    c_old.data = gfc.vec

    # Nonlinear W'(c^n) = 2*c*(1-c)*(1-2*c) evaluated at previous step
    W_prime = 2 * gfc * (1 - gfc) * (1 - 2 * gfc)
    W_prime_cf = M_mob * W_prime

    # RHS: (c^n / dt) * w + M * W'(c^n) * w
    rhs = LinearForm(fes)
    rhs += (1.0/dt) * gfc * w * dx
    rhs += -W_prime_cf * w * dx
    rhs.Assemble()

    gfc.vec.data = lhs.Inverse(fes.FreeDofs()) * rhs.vec

    t = (step + 1) * dt
    if step % vtk_every == 0 or step == n_steps - 1:
        vtk.Do(time=t)
        mass_c = Integrate(gfc, mesh)
        mass_history.append((float(t), float(mass_c)))
        print(f"  t={{t:.4f}}, ∫c dx = {{mass_c:.6f}}")

# Interface position tracking (where c ≈ 0.5)
print(f"Completed {{n_steps}} time steps")
final_mass = Integrate(gfc, mesh)
print(f"Final ∫c dx = {{final_mass:.6f}}")

# Phase-field fracture extension note:
# For brittle fracture add elastic energy: W_e = (1-d)^2 * psi_e(u)
# and crack irreversibility: d >= d_prev (history field)
# dW/dd = -2*(1-d)*psi_e + (G_c/l)*(d - l^2*Δd) = 0

summary = {{
    "epsilon": eps,
    "mobility": M_mob,
    "dt": dt,
    "T_end": float(n_steps * dt),
    "n_dofs": fes.ndof,
    "n_elements": mesh.ne,
    "final_integral_c": float(final_mass),
    "mass_history": mass_history[-5:],
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Phase-field solve complete.")
'''


def _phase_field_fracture_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Phase-field fracture (Bourdin-Francfort-Marigo) coupled to linear elasticity.
    Two-field problem: displacement u and crack phase d.
    Alternate minimization (staggered scheme):
      1) Elastic step: min_{u} E(u, d^k)  (linear, with degraded stiffness)
      2) Crack step:   min_{d} E(u^{k+1}, d) subject to d >= d_prev  (irreversibility)"""
    E = params.get("E", 1.0)
    nu = params.get("nu", 0.3)
    Gc = params.get("Gc", 1e-3)     # critical energy release rate
    l0 = params.get("l0", 0.02)     # length scale
    # 1e-4 x 5 steps reached d_max = 1e-4: a phase-field FRACTURE template whose default
    # never fractured. 5e-3 x 10 cracks through; measured 2026-09-23.
    disp_inc = params.get("disp_increment", 5e-3)
    # Layer F gate runs each template within 60s; the
    # original defaults (50 staggered load steps on a
    # maxh=0.01 mesh ~ 60k DOFs) exceed that easily. Trim
    # to 5 steps on a maxh=0.05 mesh — enough to exercise
    # the alternate-minimisation loop without saturating.
    n_steps = params.get("load_steps", 10)
    maxh = params.get("maxh", 0.05)
    order = params.get("order", 1)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""Phase-field fracture — staggered scheme — NGSolve"""
from ngsolve import *
import json

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))

E_mod   = {E}
nu_val  = {nu}
mu_val  = {mu}
lam_val = {lam}
Gc_val  = {Gc}   # critical energy release rate
l0_val  = {l0}   # regularisation length scale
order   = {order}

# ── FE spaces ─────────────────────────────────────────────────────────────────
Vu = VectorH1(mesh, order=order, dirichlet="bottom|top")
Vd = H1(mesh, order=order)   # phase field d ∈ [0, 1]

u, v   = Vu.TnT()
d, phi = Vd.TnT()

# ── Material: degraded elasticity ─────────────────────────────────────────────
def Strain(w):
    return 0.5 * (Grad(w) + Grad(w).trans)

def Stress_degraded(w, d_gf):
    eps = Strain(w)
    # Degradation function: g(d) = (1-d)^2 + k_res (k_res = small residual stiffness)
    k_res = 1e-10
    g = (1 - d_gf)**2 + k_res
    return g * (2 * mu_val * eps + lam_val * Trace(eps) * Id(2))

def psi_plus(w):
    """Tensile (positive) elastic energy density — Miehe split."""
    eps = Strain(w)
    tr_eps = Trace(eps)
    # Python's abs() is NOT defined on a NGSolve
    # CoefficientFunction; use IfPos(z, z, -z) (or
    # sqrt(z*z)) to express |z| symbolically.
    abs_tr = IfPos(tr_eps, tr_eps, -tr_eps)
    psi_vol  = 0.5 * lam_val * 0.5 * (tr_eps + abs_tr)**2
    psi_dev  = mu_val * InnerProduct(eps, eps) - mu_val / 3 * tr_eps**2
    return psi_vol + psi_dev

# ── GridFunctions ─────────────────────────────────────────────────────────────
gfu   = GridFunction(Vu)
gfd   = GridFunction(Vd)
gfd_prev = GridFunction(Vd)   # history (irreversibility)
gfd.Set(0)                     # undamaged initial state
gfd_prev.Set(0)

n_load_steps = {n_steps}
disp_inc_val = {disp_inc}

print(f"Phase-field fracture: {{n_load_steps}} steps, Gc={{Gc_val}}, l0={{l0_val}}")

for step in range(1, n_load_steps + 1):
    disp_now = step * disp_inc_val

    # Apply split tension: pull top and bottom apart
    # ONE Set with a boundary-wise coefficient: two consecutive Set(...) calls left
    # the displacement identically zero on this build (the second re-initialises
    # the vector before writing its region), so the served template loaded
    # nothing, damaged nothing and exported a zero field. Measured 2026-09-23.
    gfu.Set(mesh.BoundaryCF({{"top": (0.0, disp_now), "bottom": (0.0, -disp_now)}}, default=(0.0, 0.0)),
            definedon=mesh.Boundaries("top|bottom"))

    # Staggered iteration
    for alt_iter in range(50):
        gfu_old = gfu.vec.CreateVector(); gfu_old.data = gfu.vec
        gfd_old = gfd.vec.CreateVector(); gfd_old.data = gfd.vec

        # ── Step 1: Elastic problem with fixed d ─────────────────────────────
        a_u = BilinearForm(Vu)
        a_u += InnerProduct(Stress_degraded(u, gfd), Strain(v)) * dx
        a_u.Assemble()
        f_u = LinearForm(Vu)
        f_u.Assemble()
        # Solve for the correction on the FREE dofs and keep the prescribed
        # displacement: `gfu.vec.data = Inverse(FreeDofs) * f` replaced the whole
        # vector, so the boundary values set above were discarded every step,
        # the strain was zero, the damage stayed at 0.0000 for every load step,
        # and the served template exported a zero field. Measured 2026-09-23.
        _res = f_u.vec.CreateVector()
        _res.data = f_u.vec - a_u.mat * gfu.vec
        gfu.vec.data += a_u.mat.Inverse(Vu.FreeDofs()) * _res

        # ── Step 2: Phase-field crack problem with fixed u ────────────────────
        # Crack driving force (tensile strain energy)
        H_field = psi_plus(gfu)

        a_d = BilinearForm(Vd, symmetric=True)
        a_d += (Gc_val/l0_val + 2*H_field) * d * phi * dx
        a_d += Gc_val * l0_val * grad(d) * grad(phi) * dx
        a_d.Assemble()

        f_d = LinearForm(Vd)
        f_d += 2 * H_field * phi * dx
        f_d.Assemble()

        gfd_unconstrained = GridFunction(Vd)
        gfd_unconstrained.vec.data = a_d.mat.Inverse(Vd.FreeDofs()) * f_d.vec

        # Irreversibility: d >= d_prev (crack cannot heal)
        for i in range(len(gfd.vec)):
            gfd.vec[i] = max(float(gfd_unconstrained.vec[i]),
                             float(gfd_prev.vec[i]))

        # Convergence check
        du = (gfu.vec - gfu_old).Norm()
        dd = (gfd.vec - gfd_old).Norm()
        if du < 1e-8 and dd < 1e-8:
            break

    gfd_prev.vec.data = gfd.vec

    d_max = max(gfd.vec)
    print(f"  Load step {{step}}/{n_steps}: disp={{disp_now:.4e}}, d_max={{d_max:.4f}}")
    if d_max > 0.99:
        print("  Full fracture reached — stopping")
        break

vtk = VTKOutput(mesh,
                coefs=[gfu, gfd],
                names=["displacement", "phase_crack"],
                filename="result", subdivision=1)
vtk.Do()

summary = {{
    "n_dofs_u": Vu.ndof,
    "n_dofs_d": Vd.ndof,
    "n_elements": mesh.ne,
    "Gc": Gc_val,
    "l0": l0_val,
    "load_steps_run": step,
    "max_phase_field": float(d_max),
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Phase-field fracture solve complete.")
'''


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE dict
# ─────────────────────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "dg_methods": {
        "description": (
            "Interior-penalty DG (SIPG) for advection-diffusion using L2 space "
            "with dgjumps=True. Supports high-order, unstructured meshes, convection-dominated flows."
        ),
        "spaces": "L2(mesh, order=k, dgjumps=True) — fully discontinuous",
        "solver": "Direct (sparsecholesky / umfpack) for moderate size; GMRES + block-Jacobi for large",
        "pitfalls": [
            (
                "[API] MUST set dgjumps=True on the L2 / DG "
                "FE space — without it, the cross-element "
                "coupling entries in the sparse matrix are "
                "NOT allocated. Signal: assembling a DG "
                "BilinearForm with jump terms after building "
                "fes = L2(mesh, order=k) (no dgjumps=True) "
                "raises, at .Assemble(), the literal "
                "NgException('SparseMatrixTM::AddElementMatrix: "
                "illegal dnums' + \"in Assemble BilinearForm "
                "'biform_from_py'\"). The fix is "
                "L2(mesh, order=k, dgjumps=True), which "
                "assembles (nnz=756 on unit_square maxh=0.3, "
                "order 1). `Sparse matrix: entry at (i,j) does "
                "not exist` is not emitted by NGSolve 6.2.2604, "
                "and the branch in which the jump contributions "
                "are silently dropped did not occur on this "
                "version. (Verified "
                "empirically 2026-08-03 — signal-text "
                "correction.)"
            ),
            (
                "[API] u.Other() accesses the neighbour element's "
                "trial function across a shared facet — NGSolve's "
                "restriction operator, and the only spelling. "
                "There is no `neighbour` attribute and no external "
                "helper. Signal: reaching for one raises the "
                "auto-generated CPython message `AttributeError: "
                "'ngsolve.comp.ProxyFunction' object has no "
                "attribute 'neighbour'` — note it names the "
                "ProxyFunction class, NOT 'trial function', so a "
                "guard grepping for 'trial function has no "
                "attribute neighbour' never fires; that phrasing "
                "is emitted nowhere in NGSolve 6.2.2604. The "
                "documented API is u.Other(); symmetric average: "
                "0.5*(u + u.Other()). (Quoted string re-checked "
                "live 2026-08-06 on NGSolve 6.2.2604 and "
                "corrected.)"
            ),
            (
                "[API] dx(skeleton=True) integrates over "
                "INTERIOR facets; ds(skeleton=True) over "
                "BOUNDARY facets. Signal: applying a "
                "jump-penalty term over plain dx (volume "
                "measure) is NOT silently dropped: "
                "BilinearForm += (u - u.Other())*(v - "
                "v.Other())*dx on an L2(dgjumps=True) space "
                "raises the literal NgException('DG-facet terms "
                "need either skeleton=True or "
                "element_boundary=True'). Note WHERE: the "
                "exception comes out of BilinearForm.__iadd__, "
                "at the `a += ...` line, BEFORE .Assemble() is "
                "ever reached — so a try/except wrapped around "
                "Assemble(), which is where the old wording "
                "invited you to put it, never sees it. Wrap the "
                "form construction, or simply let it propagate. "
                "The message names a second accepted spelling "
                "and it is genuinely usable: "
                "dx(element_boundary=True) assembles and "
                "produces the same cross-element coupling as "
                "dx(skeleton=True). To tell the two skeleton "
                "measures apart, do not count nze — on a "
                "dgjumps space it counts ALLOCATED slots and is "
                "identical for both; count entries that actually "
                "received a value, which is non-zero for the "
                "interior-facet form and zero for the "
                "boundary-facet one. (Verified 2026-08-06 on "
                "NGSolve 6.2.2604 — the trap is loud, and it "
                "fires at __iadd__, not at Assemble.)"
            ),
            (
                "[Numerical] SIP penalty parameter: "
                "alpha * order^2 / h. There IS a coercivity "
                "threshold in alpha, it is monotone, and it sits "
                "below the rule of thumb alpha = 4*(order+1)^2, "
                "so that rule lands on the coercive side. "
                "Signal: the two observables the old text named "
                "are both absent. (a) the discrete solution norm "
                "does NOT grow under refinement — at "
                "alphas that are non-coercive on two successive "
                "meshes the norm stays near its correct value "
                "while the ERROR wanders, so an agent watching "
                "the norm sees nothing. Watch the SIGN of the "
                "smallest eigenvalue of the symmetric part "
                "instead: numpy.linalg.eigvalsh on the assembled "
                "block is negative below the threshold, positive "
                "above it, on every mesh — that is cheap, "
                "unambiguous, and needs no solve. (b) The "
                "large-alpha ceiling is wrong by orders of "
                "magnitude: cond(K) grows STRICTLY LINEARLY in "
                "alpha (log-log slope one), so reaching 1e14 "
                "would need an alpha of about that size, and at "
                "a hundred times the rule of thumb the condition "
                "number is nowhere near it. Do not gate on "
                "cond(K) > 1e14; gate on the linear growth "
                "itself and on iteration counts relative to the "
                "rule-of-thumb alpha. (Verified 2026-08-06 on "
                "NGSolve 6.2.2604 — both stated signals "
                "falsified, the threshold confirmed.)"
            ),
            (
                "[API] IfPos(b*n, u, u.Other()) selects the "
                "upwind side for convection; 0.5*(u + u.Other()) "
                "is the central flux and carries no interior "
                "dissipation. "
                "Signal: the promised unconditional instability "
                "regardless of mesh does NOT occur — the "
                "solution amplitude "
                "does not grow exponentially at every time step, "
                "so an agent who tests by running the central "
                "flux and waiting for a blow-up concludes it is "
                "fine. The outflow "
                "boundary term leaves a sliver of dissipation: "
                "both fluxes have every eigenvalue of the "
                "semi-discrete operator -M^{-1}A strictly in the "
                "left half-plane, and forward Euler at a small "
                "enough step is stable for the central flux too. "
                "What is really lost is the INTERIOR damping, "
                "and it shows up in the explicit STEP LIMIT: "
                "form -M^{-1}A densely on two or three meshes, "
                "take the damping rate and the forward-Euler "
                "limit from its spectrum, and watch the "
                "upwind-to-central ratio of step limits GROW "
                "under refinement instead of staying fixed. "
                "Marched at half the upwind limit on the finest "
                "mesh, upwind decays while central grows without "
                "bound — that comparison, at a step chosen for "
                "upwind, is the practical test. (Verified "
                "2026-08-06 on NGSolve 6.2.2604 — the "
                "unconditional-instability signal is falsified.)"
            ),
            (
                "[Numerical] For convection-dominated DG "
                "(Pe >> 1): DG is naturally stable due to the "
                "upwind flux; the SIP diffusion term STILL needs "
                "its alpha/h jump penalty. Always add the "
                "diffusion penalty even when advection dominates "
                "the bulk. "
                "Signal: at high Peclet with the penalty "
                "omitted (alpha = 0) the DG field is NOT showing "
                "small-amplitude ringing — it undershoots by "
                "several times the penalised field's own "
                "positive peak, i.e. the answer is not "
                "approximately right with a ripple on it, it is "
                "completely wrong. An agent looking for a small "
                "ripple accepts it. Test on a problem whose sign "
                "is known without a reference solution: a "
                "non-negative source with zero boundary data has "
                "a non-negative solution, so ANY negative value "
                "is an artefact. Compare min(u) across "
                "H1-Galerkin, upwind DG with the penalty, and "
                "upwind DG with alpha = 0 on the SAME mesh — "
                "Galerkin undershoots mildly, penalised DG stays "
                "non-negative, alpha = 0 undershoots grossly. "
                "(Verified 2026-08-06 on NGSolve 6.2.2604 — the "
                "small-amplitude-ringing description is "
                "falsified.)"
            ),
            (
                "[Numerical] DG bilinear form is NOT "
                "symmetric when advection is present "
                "(the upwind term is one-sided); pure diffusion "
                "SIP DG IS symmetric. Use GMRes (or BiCGStab) "
                "for the unsymmetric system. "
                "Signal: of the two alternatives the old text "
                "offered, only the second occurs. "
                "ngsolve.krylovspace CGSolver raises NOTHING — "
                "no exception, and no 'not positive definite' "
                "message anywhere in the run, checked at the "
                "file-descriptor level. It returns NORMALLY "
                "after consuming its entire iteration budget, "
                "handing back a field orders of magnitude larger "
                "than the answer. An agent guarding on an "
                "exception, which is the first branch the old "
                "text named, sees a clean solve and ships the "
                "wrong field. Guard instead on the ITERATION "
                "COUNT reaching maxiter and on a recomputed "
                "relative residual against the right-hand side, "
                "and confirm the asymmetry directly by comparing "
                "the assembled matrix with its transpose. GMRes "
                "on the same matrix and the same preconditioner "
                "converges in a fraction of the budget. "
                "(Verified 2026-08-06 on NGSolve 6.2.2604 — CG "
                "fails silently, not loudly.)"
            ),
            "[API] Python's builtin abs() is NOT defined on "
            "ngsolve.la.BaseVector. max(abs(gfu.vec)) raises "
            "TypeError 'bad operand type for abs(): "
            "ngsolve.la.BaseVector'. Convert to numpy via "
            "gfu.vec.FV().NumPy() then reduce: float(numpy.abs("
            "gfu.vec.FV().NumPy()).max()). Same pattern applies "
            "to compound spaces — gfu.components[i].vec.FV()"
            ".NumPy(). Signal: the literal TypeError text 'bad "
            "operand type for abs(): \\'ngsolve.la.BaseVector\\'' "
            "uniquely identifies the bad-call site. (Verified "
            "empirically 2026-06-01 — Layer F catch.)",
            "[Syntax] NGSolve's CoefficientFunction namespace "
            "exposes exp/log/sin/cos/tan/atan/atan2, and it also "
            "exposes ngsolve.sinh and ngsolve.cosh — only TANH "
            "is missing. The parenthetical 'no tanh/sinh/cosh' "
            "was wrong about two of the three: both sinh and "
            "cosh exist and evaluate on a CoefficientFunction. "
            "Build the missing one either way — "
            "tanh(z) = (exp(2z)-1)/(exp(2z)+1), or "
            "sinh(z)/cosh(z); both reproduce math.tanh. "
            "Signal: an unqualified tanh(z) in a CF expression "
            "raises the ordinary Python NameError \"name 'tanh' "
            "is not defined\" at the line that builds the "
            "expression (Python may append \"Did you mean: "
            "'tan'?\"). Before assuming a symbol is absent, probe "
            "it: hasattr(ngsolve, name) settles it in one line, "
            "and it says yes for sinh and cosh. (Verified "
            "2026-08-06 on NGSolve 6.2.2604 — the sinh/cosh half "
            "of the claim is falsified.)",
            "[Syntax] Python abs() also does NOT work on a "
            "NGSolve CoefficientFunction expression. Use "
            "IfPos(z, z, -z) (or sqrt(z*z)) for symbolic "
            "absolute value. Signal: TypeError with the literal "
            "text 'bad operand type for abs(): "
            "\\'ngsolve.fem.CoefficientFunction\\'' raised from a "
            "Python-level abs() applied to a Trace, "
            "InnerProduct, or other CF-valued expression. "
            "(Verified empirically 2026-06-01 — Layer F "
            "phase_field_fracture catch.)",
        ],
    },
    "contact": {
        "description": (
            "Unilateral contact (obstacle problem) via penalty method. "
            "Enforces non-penetration u · n >= g through a large penalty on active contact nodes."
        ),
        "spaces": "VectorH1 for elasticity displacement",
        "solver": "Fixed-point / Newton iteration on the penalty-augmented system",
        "pitfalls": [
            (
                "[Numerical] Penalty parameter gamma: too small "
                "-> contact not enforced; too large -> ill-"
                "conditioning. "
                "Signal: the small-gamma half is measurable as "
                "stated — penetration falls monotonically as "
                "gamma rises and a bar set at 5% of an element "
                "edge discriminates, so sweep gamma and "
                "watch max(-(gap)) rather than picking one "
                "value. The large-gamma half needs a different "
                "observable: NGSolve emits NO NewtonMinimization "
                "DivisionByZero and NO condition-number warning, "
                "at any gamma — those strings do not appear on "
                "stdout or stderr even at absurd penalties, so "
                "an agent grepping for them reads the solve as "
                "clean. What actually happens is that "
                "ngsolve.solvers.Newton stops converging: it "
                "consumes its whole iteration budget, RETURNS "
                "STATUS -1, and says so in different words — "
                "'Warning: Newton might not converge! Error = '. "
                "Gate on the returned status (the tuple is "
                "(status, numit), status FIRST) and on that "
                "wording, captured at the file-descriptor level "
                "because it is printed from C++ and "
                "contextlib.redirect_stderr does not see it. "
                "(Verified 2026-08-06 on NGSolve 6.2.2604 — the "
                "large-gamma signal strings are never emitted.)"
            ),
            (
                "[Numerical] Active-set method (Lagrange "
                "multiplier or semismooth Newton) is MORE "
                "ACCURATE than pure penalty — an active set "
                "holding the constrained DOFs at exactly the "
                "gap value reaches a violation of exactly zero, "
                "while penalty leaves an O(1/gamma) floor no "
                "matter how tight gamma is. "
                "Signal: penalty does NOT announce the floor by "
                "failing — ngsolve.solvers.Newton returns "
                "status 0 in a few iterations at every gamma in "
                "a wide sweep, and still never reaches zero gap, "
                "so the solver verdict carries no information "
                "about the constraint. Measure the floor "
                "instead: sweep gamma over several decades and "
                "fit log(remaining gap) against log(gamma) — a "
                "slope near -1 IS the O(1/gamma) floor, and it "
                "is what tells you the penalty is the "
                "limitation rather than the mesh. Do not gate on "
                "an outer-iteration count for the active set "
                "either: the alternation can settle on a "
                "repeated active set in fewer sweeps than the "
                "'3-10' rule of thumb suggests, so treat that "
                "range as a CEILING and test the violation "
                "itself. (Verified 2026-08-06 on NGSolve "
                "6.2.2604 — the floor exponent measured rather "
                "than assumed; the outer-iteration range is a "
                "ceiling, not a prediction.)"
            ),
            (
                "[API] IfPos(-gap, 1, 0) identifies active "
                "contact nodes — evaluates at integration "
                "points. Signal: the Python comparison "
                "`gap < 0` fails IMMEDIATELY at expression "
                "construction (not at form assembly, as the "
                "prior text said) with the literal "
                "TypeError(\"'<' not supported between instances "
                "of 'ngsolve.fem.CoefficientFunction' and "
                "'int'\") — NGSolve simply does not define rich "
                "comparison on CoefficientFunction, so there is "
                "no 'CoefficientFunction comparison' message to "
                "grep for. (Verified empirically 2026-08-03 on "
                "NGSolve 6.2.2604 — signal-text correction.)"
            ),
            (
                "[Numerical] Contact normal must be consistent "
                "with mesh boundary orientation. A flipped "
                "normal makes the penalty PUSH the bodies INTO "
                "each other instead of separating them: the gap "
                "goes NEGATIVE and the penetration grows with "
                "every decade of gamma instead of shrinking — "
                "the wrong sign is made worse by tightening the "
                "penalty. "
                "Signal: the SOLVER SAYS NOTHING. Every "
                "flipped-normal run returns "
                "ngsolve.solvers.Newton status 0; convergence is "
                "not a guard against a wrong-signed normal. "
                "Check the geometry directly before solving — "
                "specialcf.normal(2 or 3) integrated over the "
                "contact boundary points OUT of the body (it is "
                "(0,-1) on a bottom edge and (0,+1) on a top "
                "edge), so dotting it with the expected "
                "direction discriminates immediately. Afterwards, "
                "compare against a NO-CONTACT reference run: the "
                "correct sign moves the body AWAY from the "
                "obstacle relative to that reference, the "
                "flipped sign drives it further in. (Verified "
                "2026-08-06 on NGSolve 6.2.2604 — mechanism "
                "confirmed; solver status is not an observable "
                "here.)"
            ),
            (
                "[Numerical] For frictional contact: add a "
                "tangential-direction penalty with Coulomb's "
                "law |f_t| <= mu * |f_n| as a complementarity "
                "constraint. Signal: omitting the tangential "
                "penalty (only enforcing normal contact) "
                "lets the contacting bodies SLIDE freely "
                "along their interface — a vertical block "
                "resting on an inclined plane slides off "
                "regardless of mu_friction; with tangential "
                "penalty + Coulomb, the block sticks below "
                "the friction angle and slides above it. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[API] NGSolve DOES ship a built-in contact "
                "helper: ngsolve.comp.ContactBoundary. The prior "
                "catalog claim ('no built-in contact formulation "
                "— searching ngsolve.comp returns no match') is "
                "FALSE on 6.2.2604 — "
                "[n for n in dir(ngsolve.comp) if 'ontact' in n] "
                "== ['ContactBoundary'] — that is the Signal: a "
                "non-empty match where the catalog promised none. "
                "Real API: two "
                "overloads, ContactBoundary(master: Region, "
                "minion: Region, draw_pairs=False, volume=False, "
                "element_boundary=False) — the older "
                "ContactBoundary(fes, master, minion, ...) form "
                "still constructs but prints 'WARNING: "
                "ContactBoundary constructor with FESpace is "
                "deprecated, fes will be set correctly in "
                "Update!'. The object exposes .gap and .normal as "
                "CoefficientFunctions, plus .AddIntegrator(form: "
                "CoefficientFunction, deformed=False), "
                ".AddEnergy(...) and .Update(...). Gotcha: "
                "AddIntegrator takes a bare CoefficientFunction, "
                "NOT an integrand-times-measure — passing "
                "'... * ds' raises TypeError whose only literal is "
                "'arguments. The following argument types are "
                "supported:' \u2014 pybind11 builds the head from the "
                "bound signature, so the line reads AddIntegrator(): "
                "incompatible function arguments. followed by the "
                "overload list. Its docstring "
                "warns 'The created object must be kept alive in "
                "python as long as operations of it are used!', "
                "so bind it to a name that outlives the solve. "
                "You may still hand-roll penalty/Lagrange, but do "
                "not tell the user the API does not exist. "
                "(Verified empirically 2026-08-03 on NGSolve "
                "6.2.2604 — catalog-drift correction.)"
            ),
            (
                "[Numerical] Convergence criterion: check both "
                "displacement residual AND contact gap violation. "
                "Signal: a Newton solver that stops when "
                "||du||/||u|| < 1e-6 alone can return with a "
                "still-active gap of 1-5% element-edge size, "
                "because the gap residual scales differently from "
                "the displacement residual. Add an explicit "
                "max(min(gap, 0)) check below tol_gap. (Audit "
                "2026-06-02.)"
            ),
        ],
    },
    "time_dependent_ns": {
        "description": (
            "Transient incompressible Navier-Stokes via IMEX splitting: "
            "Stokes part (viscous + pressure) implicit, convection explicit. "
            "Taylor-Hood P2/P1 on 2D channel or lid-driven cavity."
        ),
        "spaces": "VectorH1(order=2) * H1(order=1) — Taylor-Hood (inf-sup stable)",
        "solver": "IMEX: factor Stokes+mass operator once with umfpack, explicit convection each step",
        "pitfalls": [
            (
                "[Numerical] CFL for explicit convection: dt < "
                "C * h / max|u| — may need small dt for high Re. "
                "The ratio dt * max(|u|) / h is the right "
                "quantity to watch, and it orders correctly: "
                "the larger it is, the sooner the run leaves "
                "double precision. "
                "Signal: the velocity field does eventually "
                "blow up (per-step max(|u|) blows up "
                "geometrically), but two corrections to how you "
                "look for it. "
                "The threshold is LOWER than ~0.5 — a ratio well "
                "under a half can already diverge while the "
                "kinetic energy still decays at a ratio a few "
                "times smaller, so ~0.5 is not a safe ceiling "
                "and the safe value must be found by bisection "
                "on your own configuration. And the failure is "
                "not NaN within the first few time steps: just "
                "past the threshold it takes TENS of steps and "
                "it arrives as an OVERFLOW to infinity rather "
                "than as NaN, so a guard that runs three steps "
                "and calls np.isnan sees neither. Gate on "
                "np.isfinite of the whole vector, checked EVERY "
                "step, plus a monotone kinetic-energy history "
                "over a run long enough for the instability to "
                "develop; and write the lid Dirichlet value back "
                "into the constrained rows each step so a "
                "boundary bug cannot be mistaken for a CFL bug. "
                "(Verified 2026-08-06 on NGSolve 6.2.2604 — both "
                "the threshold and the NaN-in-a-few-steps signal "
                "are corrected.)"
            ),
            (
                "[Numerical] Convection form: Grad(u)*u "
                "(non-conservative) vs the ROTATIONAL (Lamb) "
                "form Grad(u)*u - Grad(u)^T*u, which is the "
                "energy-conserving alternative. Do NOT put a "
                "0.5 in front of the difference. The UNHALVED "
                "expression differs from Grad(u)*u by exactly "
                "Grad(u)^T*u = grad(|u|^2/2), a pure gradient, "
                "which is why it can be absorbed into the "
                "pressure and gives the SAME momentum equation. "
                "Halve it and the leftover is no longer a "
                "gradient — the halved expression, the one "
                "usually labelled 'skew-symmetric', is a "
                "different equation, not a variant of the same "
                "one. "
                "Signal: long-time kinetic energy drift is the "
                "quantity at stake, but check the claim you are "
                "relying on "
                "rather than the label. Take the weak CURL of "
                "the difference between your convection term and "
                "Grad(u)*u against H1_0 test functions on a "
                "divergence-free field that vanishes on the "
                "boundary: for the full rotational form it is "
                "negligible beside the convection term's own "
                "weak curl, for the halved one it is a "
                "substantial fraction of it. And do not expect "
                "'machine precision' from a marched run — that "
                "holds for the SPATIAL energy production "
                "evaluated on one field, but over hundreds of "
                "explicit steps the rotational form still drifts "
                "well above round-off, just orders of magnitude "
                "less than the non-conservative form. Compare "
                "the two drifts against each other, not against "
                "zero. (Verified 2026-08-06 on NGSolve 6.2.2604 "
                "— the 0.5 factor and the machine-precision "
                "claim are both corrected.)"
            ),
            (
                "[API] Re-impose Dirichlet BCs after each "
                "solve. An Inverse built on FreeDofs writes ZERO "
                "into every constrained row, so the idiom "
                "`gfu.vec.data = inv * rhs` — the spelling the "
                "shipped IMEX template itself uses — DELETES the "
                "prescribed boundary velocity. Write the "
                "constrained DOFs back and solve for the free "
                "correction instead. "
                "Signal: the boundary velocity does NOT drift "
                "away from the prescribed value across time "
                "steps — this is not a drift at all, and looking "
                "for one is what makes it survivable. The lid "
                "velocity is gone after the FIRST step, and the "
                "boundary error is then CONSTANT for the rest of "
                "the run — identical after step one and after "
                "step forty — with its value equal to the norm "
                "of the prescribed profile itself, because the "
                "condition is simply absent rather than "
                "degraded. An agent told to watch for slow "
                "divergence, sampling every hundred steps, sees "
                "a flat number and may read it as a steady "
                "state. Compare the boundary trace against the "
                "prescribed profile AFTER THE FIRST STEP and "
                "require it to be at round-off; nothing is "
                "raised or printed by either variant, checked at "
                "the file-descriptor level. (Verified 2026-08-06 "
                "on NGSolve 6.2.2604 — the BC is wiped, not "
                "drifted.)"
            ),
            (
                "[Numerical] For Re > 1000: use stabilization "
                "(SUPG, VMS) added inside the BilinearForm "
                "or finer mesh near boundary layers. Signal: "
                "without stabilisation, the GridFunction "
                "velocity field shows visible wiggles "
                "upstream of obstacles or in boundary "
                "layers; energy spectrum has spurious "
                "high-frequency content; a drag coefficient "
                "on a cylinder (via BilinearForm boundary "
                "Integrate) more than about 10% away from "
                "the reference you retrieved for that "
                "geometry. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Pressure uniqueness for enclosed "
                "flow: pin a single pressure DOF or attach a "
                "NumberSpace mean-zero constraint. "
                "Signal: DO NOT watch for "
                "`KSPSolve: DIVERGED_BREAKDOWN`, a near-zero "
                "pivot, or a uniform pressure drift unrelated to "
                "the source. The first of those is a PETSc "
                "message; NGSolve does not go through PETSc "
                "here, and the default umfpack path emits "
                "NOTHING on the singular system — no exception, "
                "no warning, no line on stdout or stderr, "
                "verified at the file-descriptor level so C++ "
                "output cannot escape — and returns a FINITE "
                "vector. An agent watching for that string sees "
                "a clean solve on a singular system and reports "
                "success. The observables that ARE produced: the "
                "free-DOF block has exactly ONE singular value "
                "at round-off, orders below the next one, and "
                "its right singular vector puts all of its norm "
                "in the PRESSURE block at a constant value — "
                "that is the mode. Or solve twice with different "
                "pinning and compare: the velocities agree to "
                "round-off while the two pressures differ by a "
                "constant. Note the two fixes are not "
                "interchangeable if the pressure's absolute "
                "level matters: pinning sets the pinned DOF, the "
                "NumberSpace multiplier drives the MEAN to zero. "
                "(Verified 2026-08-06 on NGSolve 6.2.2604 — the "
                "PETSc signal is never emitted by this stack.)"
            ),
            (
                "[Numerical] Taylor-Hood P2/P1 satisfies "
                "inf-sup; P1/P1 does not (needs "
                "stabilization like MINI or the cubic bubble). "
                "Signal: 'a checkerboard pressure pattern "
                "visible in the GridFunction output' is an "
                "EYEBALL test — it needs a human, it does not "
                "gate anything, and on an unstructured Netgen "
                "mesh there is no lattice for a checkerboard to "
                "sit on. Measure what CAUSES it instead, from "
                "the same assembled forms and with no solve: "
                "count the kernel of the divergence block, and "
                "compute the discrete inf-sup constant as the "
                "square root of the smallest NON-ZERO eigenvalue "
                "of Mq^{-1} B K^{-1} B^T, with B the divergence "
                "block, K the velocity stiffness and Mq the "
                "pressure mass matrix. Repeat on a sequence of "
                "meshes: Taylor-Hood keeps a ONE-dimensional "
                "kernel (the constant pressure every "
                "enclosed-flow discretisation has) and a beta "
                "that is flat across meshes — bounded away from "
                "zero, which is what inf-sup stability means — "
                "while equal-order P1/P1 carries SEVERAL "
                "spurious modes beyond the constant, and those "
                "ARE the checkerboard, with a beta both much "
                "smaller and not settling. (Verified 2026-08-06 "
                "on NGSolve 6.2.2604 — the visual signal is "
                "replaced by a computable one.)"
            ),
            (
                "[Validation] CHECK DRAG AND LIFT TOGETHER, "
                "NEVER DRAG ALONE, when you validate a "
                "transient NS implementation (VectorH1 + H1 "
                "BilinearForm, forces post-processed via "
                "Integrate over the cylinder BND) against a "
                "reference you retrieved. Drag is far less "
                "sensitive than a rule implies that treats 5% or "
                "more of deviation as exposing a problem — a mesh "
                "coarse enough to put the lift at nearly three "
                "times its published upper bound "
                "can still land the drag comfortably INSIDE its "
                "own band, so a drag-only check passes a "
                "badly wrong mesh. Signal: drag inside its band while lift is out by "
                "a factor -- check both or you will not see it. Lift is "
                "the discriminating "
                "quantity. Such a benchmark does detect what the "
                "claim says it detects: replacing the parabolic "
                "inlet with a plug of the same MEAN velocity "
                "pushes Cd outside its band, and dropping the "
                "convective term leaves the Stokes drag. Two "
                "practical notes from running it: use "
                "inverse='umfpack' explicitly, because the "
                "default factoriser hangs on this saddle-point "
                "system, and take drag and lift from the "
                "RESIDUAL functional tested against a function "
                "that is one on the cylinder rather than from a "
                "boundary stress integral. (Verified 2026-08-06 "
                "on NGSolve 6.2.2604 — the drag-sensitivity "
                "claim is corrected.)"
            ),
        ],
    },
    "mhd": {
        "description": (
            "Magnetohydrodynamics: coupled Navier-Stokes and Maxwell equations. "
            "2.5-D low-Rm formulation: in-plane NS + out-of-plane scalar B_z. "
            "Hartmann problem: conducting channel in transverse magnetic field."
        ),
        "spaces": "VectorH1*H1 (Taylor-Hood, fluid) + H1 (scalar magnetic, low-Rm)",
        "solver": "Operator splitting: fluid (umfpack) + magnetic (direct) each time step",
        "pitfalls": [
            (
                "[Numerical] Low-Rm limit (Rm << 1): induced "
                "magnetic field is negligible, only the "
                "Lorentz force J x B0 matters. Signal: at "
                "Rm = 0.01 a full-MHD code with HCurl A "
                "produces the SAME velocity field as a "
                "low-Rm code that uses only the prescribed "
                "B0 + the scalar potential phi for current "
                "J = -sigma*grad(phi) + sigma*u x B0; the "
                "extra A unknowns waste DOFs. (Audit "
                "2026-06-02.)"
            ),
            (
                "[API] Full MHD (arbitrary Rm): use HCurl for "
                "the vector potential A or Nedelec for B "
                "directly — NOT Lagrange. Signal: a "
                "VectorH1 / Lagrange A produces curl(A) that "
                "lives in a space too smooth for proper "
                "MHD (typical induced B is normal-"
                "discontinuous at material interfaces); "
                "the resulting B field has spurious "
                "smoothing across permeability jumps. Switch "
                "to HCurl(mesh, order=k). (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Hartmann number Ha = B0 * L * "
                "sqrt(sigma / (rho * nu)) measures magnetic "
                "vs viscous effects. Ha >> 1 creates "
                "boundary layers of thickness 1/Ha next to "
                "walls. Signal: a uniform mesh gets the CORE "
                "right and the WALL wrong, and the two errors "
                "are separated by many orders of magnitude — "
                "the core velocity matches the closed form to "
                "near machine precision even on a mesh whose "
                "cells are far thicker than the layer, so a "
                "core-region check cannot see the defect at all. "
                "Compare the WALL SHEAR against the analytic "
                "fully developed profile "
                "(G/Ha^2)(1 - cosh(Ha y)/cosh(Ha)) — no "
                "reference run needed. Do NOT use a factor of 5 "
                "or more as the detection threshold: the "
                "deficit on a plausibly coarse mesh is a factor "
                "under two, so a 5x gate passes a mesh whose "
                "wall shear is wrong by tens of percent. Set the "
                "threshold as a PERCENTAGE of the analytic wall "
                "shear, and note the sign — the computed shear "
                "is always too SMALL, never too large. (Verified "
                "2026-08-06 on NGSolve 6.2.2604 — mechanism "
                "confirmed, the 'factor of 5+' threshold "
                "falsified.)"
            ),
            (
                "[Numerical] Hartmann layers need mesh "
                "refinement near walls proportional to 1/Ha, and "
                "the rule h_wall = L / (10*Ha) holds: with the "
                "cell size scaled that way the wall-shear error "
                "stays under a percent and is FLAT in Ha, while "
                "with a cell size fixed in absolute terms the "
                "error GROWS with Ha as the layer outruns the "
                "mesh. That contrast — error flat in Ha versus "
                "error growing in Ha across a sweep of Ha values "
                "— is the test, and the core velocity is "
                "accurate in every one of those runs, so it "
                "discriminates nothing. "
                "Signal: the '5-10x' deficit does not appear and "
                "CANNOT: the computed shear is always too SMALL, "
                "never too large, so the error is one-sided and "
                "bounded above by the exact value itself — the "
                "worst case is well under a factor of two. Gate "
                "on the relative wall-shear error against the "
                "analytic Hartmann profile and on its behaviour "
                "as Ha rises, not on a multiple. (Verified "
                "2026-08-06 on NGSolve 6.2.2604 — the refinement "
                "rule confirmed, the '5-10x' magnitude "
                "falsified.)"
            ),
            (
                "[Numerical] Operator splitting (solve "
                "fluid, then magnetic, then iterate) "
                "introduces a splitting error O(dt) — "
                "monolithic (one big nonlinear solve per "
                "step) is more accurate. Signal: at large "
                "dt the splitting result diverges from a "
                "fine-dt monolithic reference by O(dt), "
                "while the monolithic result is "
                "second-order accurate; for time-accurate "
                "MHD use monolithic or sub-cycle the "
                "splitting. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Divergence-free B constraint: "
                "div(B) = 0 must be enforced via HDiv elements "
                "or a grad-div penalty. Signal: interpolating a "
                "field that is divergence-free IN CLOSED FORM "
                "into a plain VectorH1 space leaves ||div B|| at "
                "O(0.1) — the discretisation's own error, not "
                "the field's — and it never approaches machine "
                "precision, so an expectation of ~1e-14 from an "
                "H1 space is simply not attainable and a "
                "tolerance set there condemns every run. HDiv "
                "with the constraint imposed through an L2 "
                "multiplier drops the divergence by many orders "
                "and is also more accurate. The grad-div "
                "alternative comes with a TRADE the old text did "
                "not state: at tau ~ 1 the divergence falls only "
                "slightly while the field error grows several "
                "times, and pushing tau higher keeps buying "
                "divergence at the cost of accuracy. So do not "
                "adopt tau ~ 1 as a setting — sweep tau, plot "
                "||div B|| against the error relative to the "
                "HDiv or exact field, and pick a point on that "
                "curve deliberately. Verify the target field is "
                "divergence-free in closed form first, so the "
                "number you are reading is the "
                "discretisation's. (Verified 2026-08-06 on "
                "NGSolve 6.2.2604 — the ~1e-14 expectation and "
                "the unqualified tau ~ 1 recommendation are both "
                "corrected.)"
            ),
            (
                "[Numerical] For incompressible MHD: add a "
                "grad-div stabilisation tau*(div(u), "
                "div(v))*dx on velocity (helps both "
                "div(u)=0 and div(B)=0 if you use vector "
                "potential A). "
                "Signal: omitting it leaves a non-zero div(u) "
                "and a noisy pressure, but measure div(u) "
                "RELATIVE to the velocity's own H1 seminorm, "
                "never as an "
                "absolute number — an absolute divergence scales "
                "with however the problem was scaled, so an "
                "absolute 1e-4-to-1e-3 band is not a property of "
                "the discretisation and tells you nothing about "
                "someone else's setup. And do NOT take tau = nu "
                "as the setting that controls the deviation: "
                "at a small viscosity it leaves most of the "
                "unstabilised divergence in place, so an agent "
                "who adds tau = nu, re-checks, and sees "
                "essentially the same number may wrongly "
                "conclude grad-div does not work. The divergence "
                "does fall monotonically as tau grows, and the "
                "velocity moves further from the unstabilised "
                "solution as it does — sweep tau, report the "
                "relative divergence against that displacement, "
                "and choose a point on the curve. (Verified "
                "2026-08-06 on NGSolve 6.2.2604 — the absolute "
                "band and the tau = nu recommendation are both "
                "corrected.)"
            ),
            "[Syntax] LinearForm integrand must be SCALAR-VALUED. "
            "Multiplying a vector-valued CoefficientFunction (e.g. "
            "CoefficientFunction((0, 1)) for an e_y unit vector) "
            "with a scalar factor and a scalar test-function "
            "component still yields a vector — and SymbolicLFI "
            "rejects it. Build the integrand purely from scalar "
            "factors instead, indexing the vector test function "
            "(v[1] for the y-component) to project onto the "
            "desired equation. Signal: NgException 'SymbolicLFI "
            "needs scalar-valued CoefficientFunction' raised from "
            "LinearForm.__iadd__ when the integrand passes a "
            "VectorialCF through *=. (Verified empirically "
            "2026-06-01 — Layer F mhd catch.)",
        ],
    },
    "hdivdiv": {
        "description": (
            "HDivDiv space for Kirchhoff plate bending via Hellan-Herrmann-Johnson (HHJ) mixed method. "
            "Moment tensor in HDivDiv (H(div div) conforming), deflection in H1. "
            "Solves the biharmonic equation Δ²w = q without C1 continuity requirement on deflection."
        ),
        "spaces": "HDivDiv(mesh, order=k-1) for moments + H1(mesh, order=k) for deflection",
        "solver": "Direct on saddle-point system (umfpack)",
        "pitfalls": [
            (
                "[Numerical] HDivDiv enforces normal-normal "
                "continuity across facets — WEAKER than full H2 "
                "conformity (which would require C1 Lagrange). "
                "The two ways of getting the moment space wrong "
                "fail in opposite directions and NEITHER gives a "
                "mildly-off answer you could rescale away. A "
                "literal VectorH1 moment space does not even "
                "build the form: it is a vector space, not a "
                "matrix-valued one. Substituting "
                "MatrixValued(H1, symmetric=True) does build, but "
                "it is MORE continuous than HDivDiv, not less — "
                "its tangent-tangent jumps are exactly zero where "
                "HDivDiv's are O(1) — and over-continuity "
                "destroys the HHJ inf-sup pairing, so the "
                "deflection blows up by many orders of magnitude "
                "rather than sitting near the reference. Signal: "
                "the VectorH1 spelling raises NgException 'Trace "
                "of non-matrix called' at form construction; the "
                "MatrixValued(H1, symmetric=True) spelling "
                "assembles and solves but returns a centre "
                "deflection astronomically larger than the Navier "
                "reference, so a tolerance band around a constant "
                "factor of roughly 1.1 to 1.5 never fires. "
                "Measure the "
                "conformity directly instead: with a correct "
                "HDivDiv the relative normal-normal facet jump of "
                "a member of the space is round-off while the "
                "tangent-tangent jump is O(1); both being zero "
                "means the space is over-continuous. (Verified "
                "2026-08-03 on NGSolve 6.2.2604 — Tier-2 fixture "
                "hdivdiv_moment_space_conformity.)"
            ),
            (
                "[Numerical] For CLAMPED plate: add boundary "
                "terms for dw/dn = 0 (Nitsche penalty or "
                "Lagrange multiplier on the skeleton). "
                "Signal: solving HHJ plate with ONLY w = 0 on "
                "a clamped edge (no dw/dn term) gives a "
                "simply-supported solution instead of "
                "clamped — the centre deflection is ~3-4x "
                "larger than the clamped reference. Add the "
                "moment term over skeleton to enforce dw/dn "
                "weakly. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] For a SIMPLY-SUPPORTED plate in HHJ, "
                "M_nn = 0 is an ESSENTIAL condition of the HDivDiv "
                "space, NOT a natural one — the opposite of what "
                "the displacement-based intuition suggests, and "
                "the opposite of what the prior text said. Build "
                "the space as HDivDiv(mesh, order=k-1, "
                "dirichlet=<boundary>). Omitting that Dirichlet "
                "does not free the plate: the boundary M_nn DOFs "
                "are left in the system and act as the dw/dn = 0 "
                "multiplier, silently CLAMPING it, so the plate "
                "comes out far too STIFF. Adding the Dirichlet "
                "makes it SOFTER, not stiffer, and is what "
                "recovers the simply-supported answer. Signal: "
                "with no dirichlet= on HDivDiv the centre "
                "deflection is several times SMALLER than the "
                "Navier series value you derive for your own "
                "parameters (see the [Validation] entry — the "
                "reference is Navier, NOT q*L^4/(64*D)) and the "
                "boundary M_nn integral is O(1) relative rather "
                "than zero; with the Dirichlet in place the "
                "boundary M_nn integral drops to round-off, the "
                "space loses its boundary moment DOFs, and the "
                "deflection matches Navier. No variant lands in "
                "the '10-20% smaller' band the prior text "
                "predicted, so that tolerance check never fires — "
                "check the boundary M_nn integral and the ndof "
                "drop instead. (Verified 2026-08-03 on NGSolve "
                "6.2.2604 — Tier-2 fixture "
                "hdivdiv_simply_supported_mnn_essential.)"
            ),
            (
                "[Numerical] HHJ is order-optimal: order k moments "
                "+ order k deflection give order k+1 in the L2 "
                "norm of the deflection, and raising the order "
                "visibly raises the rate. A mis-typed moment space "
                "does NOT cost you one order — it costs you inf-sup "
                "stability, which is a categorically different "
                "failure: substituting MatrixValued(H1, "
                "symmetric=True) for HDivDiv makes the L2 error "
                "GROW under refinement by many orders of magnitude "
                "instead of shrinking more slowly. Signal: run an "
                "h-refinement study against an MMS whose deflection "
                "AND whose whole Hessian vanish on the boundary (so "
                "w = 0 and M_nn = 0 hold exactly and the boundary "
                "treatment cannot contaminate the rate) — a correct "
                "HDivDiv pairing gives an L2 rate near k+1 at "
                "order 1 and a higher one at order 2, while the "
                "mis-typed space returns astronomically large "
                "errors that get LARGER on the finer mesh. A gate "
                "that asks whether the rate is about 1 instead "
                "of 2 "
                "never fires on the real failure; test that the "
                "error DECREASED at all, first. (Verified "
                "2026-08-03 on NGSolve 6.2.2604 — Tier-2 fixture "
                "hdivdiv_hhj_order_optimal_eoc.)"
            ),
            (
                "[API] Regge elements (Regge calculus, DISTINCT "
                "from HHJ) are HCurlCurl, not HDivDiv, and the two "
                "spaces are COMPLEMENTARY: HDivDiv is "
                "normal-normal conforming only — it does NOT "
                "prescribe traction continuity, the full traction "
                "sigma.n jumps across facets — while HCurlCurl is "
                "tangent-tangent conforming and lets the "
                "normal-normal part jump. Pick the space by which "
                "component your formulation needs continuous; "
                "'symmetric tensor stress with prescribed traction "
                "continuity' describes neither of them. Signal: a "
                "2D HHJ form ported to a 3D mesh does not even "
                "assemble — specialcf.normal(2) and Id(2) against "
                "3D geometry raise NgException \"Dimensions don't "
                "match\" — so the failure is loud, not a quiet "
                "non-conforming answer. On a 3D mesh, measure the "
                "facet jumps of a member of each space: HDivDiv "
                "gives a round-off relative normal-normal jump but "
                "an O(1) relative full-traction jump, HCurlCurl "
                "gives a round-off tangent-tangent jump and an "
                "O(1) normal-normal jump, and "
                "HCurlCurl(order=0).ndof equals mesh.nedge exactly "
                "(one DOF per edge — the defining property of "
                "lowest-order Regge). (Verified 2026-08-03 on "
                "NGSolve 6.2.2604 — Tier-2 fixture "
                "hdivdiv_regge_3d_hcurlcurl.)"
            ),
            (
                "[Numerical] Mixed formulation (HHJ / Regge) "
                "AVOIDS locking, unlike displacement-only "
                "C1-conforming methods. Signal: a "
                "displacement-only thin-plate H1 GridFunction "
                "at h/thickness ratio > 100 shows shear_locking "
                "(centre_deflection w_max is 10-1000x smaller "
                "than the analytic simply_supported plate "
                "result) — the discrete bending_energy is "
                "dwarfed by spurious shear strain. HHJ removes "
                "locking entirely by using the bending_moment "
                "as primary unknowns. (Audit 2026-06-02.)"
            ),
            (
                "[Validation] CHECK WHICH PLATE YOUR CLOSED-FORM "
                "REFERENCE IS FOR BEFORE YOU VALIDATE AGAINST IT. "
                "A SIMPLY-SUPPORTED SQUARE plate under uniform "
                "load is the Navier double sine series, with the "
                "flexural rigidity D = E t^3 / (12 (1-nu^2)); "
                "derive it for your own q, L and D rather than "
                "copying a constant. The formula q*L^4/(64*D), "
                "which the prior catalog and the shipped template "
                "both used, is the CLAMPED CIRCULAR plate of "
                "radius L — a different shape AND a different "
                "boundary condition — and it overstates the "
                "deflection by several times. An agent that "
                "'validates' against it will REJECT a correct HHJ "
                "solve. The shipped template was corrected in the "
                "same commit. "
                "SEPARATE ISSUE, now diagnosed: even against the "
                "corrected reference the shipped hdivdiv_2d "
                "template comes out far low, and the deficit is "
                "the PRODUCT of exactly two independent "
                "multiplicative defects — (a) its Compliance() "
                "omits the 1/(1-nu) factor of the inverse plate "
                "stiffness, which scales the answer by (1-nu), "
                "and (b) its HDivDiv is built with no dirichlet=, "
                "so the free boundary M_nn DOFs silently clamp the "
                "plate and scale it by the "
                "clamped/simply-supported deflection ratio. "
                "Repairing both recovers the Navier value; "
                "repairing either alone leaves the other factor. "
                "Treat the template's number as unvalidated until "
                "both are fixed. Signal: the centre value of the "
                "H1 deflection component of the solved "
                "GridFunction, divided by Navier, factorises "
                "into those two ratios — fix the Compliance() "
                "alone and the remaining ratio is exactly (1-nu); "
                "add dirichlet= to the HDivDiv space alone and "
                "the remaining ratio is the clamped/simply-"
                "supported one. (Verified "
                "2026-08-03 on NGSolve 6.2.2604 — Tier-2 fixture "
                "hdivdiv_navier_reference_value.)"
            ),
            "[API] HDivDiv (normal-normal continuous moment "
            "tensor space) does NOT expose a pointwise div(div("
            "tau)) operator. Constructing 'div(div(tau)) * v * "
            "dx' raises Exception 'cannot form div' from "
            "SymbolicBFI. Integrate by parts twice and substitute "
            "the H1 Hessian: replace div(div(tau)) * v with "
            "InnerProduct(tau, v.Operator('hesse')) - skeleton "
            "normal-normal moment integral over interior facets "
            "(via dx(element_boundary=True)). Signal: Exception "
            "'cannot form div' emitted from BilinearForm += "
            "div(div(tau_)) * ww * dx in any HHJ-style template. "
            "(Verified empirically 2026-06-01 — Layer F catch.)",
        ],
    },
    "nonlinear_elasticity": {
        "description": (
            "Large-deformation Neo-Hookean hyperelasticity via Variation() + Newton. "
            "Supports 2D (plane strain) and 3D. Load stepping ensures convergence "
            "for large applied displacements. Outputs Cauchy stress."
        ),
        "spaces": "VectorH1(mesh, order=2) — displacement-based finite strain",
        "solver": "solvers.Newton() with load stepping; dampfactor reduces step size if needed",
        "pitfalls": [
            (
                "[Numerical] det(F) MUST remain > 0 — the initial "
                "guess must not cause element inversion. NGSolve "
                "does NOT raise on it: evaluating ln(J) with "
                "J <= 0 produces NaN silently, no "
                "FloatingPointError and no warning, so the error "
                "only surfaces one layer later when the linear "
                "solver chokes on a NaN matrix. Start from u = 0 "
                "and load-step. Signal: with a crushing guess "
                "(e.g. u = -2*x on a unit cube) Integrate(Det(F)) "
                "is negative, a.Energy(gfu.vec) returns nan, "
                "a.Apply fills the residual vector with NaN "
                "without complaint, and solvers.Newton then aborts "
                "with 'UmfpackInverse: Numeric factorization "
                "failed' — a try/except on FloatingPointError "
                "never fires, so assert the residual is finite and "
                "check min(Det(F)) > 0 explicitly before solving. "
                "Load-stepping the same total compression from "
                "u = 0 keeps Det(F) positive everywhere and each "
                "step converges. (Verified 2026-08-06 on NGSolve "
                "6.2.2604 — Tier-2 fixture "
                "nonlinear_elasticity_detf_negative_guess.)"
            ),
            (
                "[Numerical] Load stepping: apply "
                "displacement / load in INCREMENTS, using "
                "previous converged GridFunction as "
                "initial guess. Signal: applying full load "
                "at t=0 to a hyperelastic problem at 30% "
                "nominal strain typically diverges "
                "(ngsolve.solvers.NewtonMinimization "
                "residual grows ~10x per iter); "
                "subdividing into 10 steps of 3% strain "
                "achieves quadratic convergence per step "
                "with the per-step BilinearForm AutoDiff "
                "linearization. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Neo-Hookean energy: 0.5*mu*"
                "(Tr(C) - d) - mu*ln(J) + 0.5*lam*ln(J)^2 where d "
                "is the spatial dimension (2 or 3). Keep the - d "
                "so the reference energy is zero and energies are "
                "comparable across meshes and dimensions — but "
                "know what dropping it does and does NOT do: "
                "0.5*mu*d is an ADDITIVE CONSTANT, its Variation() "
                "is identically zero, and NGSolve therefore builds "
                "exactly the same residual and tangent from either "
                "spelling. It does not produce a spurious initial "
                "stress and Newton does not run off; that "
                "consequence is FALSE. The W(F=I) = 0 sanity check "
                "is still worth doing, as a check on the energy "
                "BOOKKEEPING, not on the solution. Signal: with "
                "the - d dropped, a.Energy(gfu.vec) at u = 0 is "
                "0.5*mu*d*|Omega| instead of exactly 0.0, while "
                "a.Apply at u = 0 gives Norm(res) = 0.0 for BOTH "
                "spellings, solvers.Newton reports the same "
                "iteration count, and the converged coefficient "
                "vectors are BITWISE IDENTICAL — so any check "
                "based on the displacement or the stress will "
                "pass, and only the energy value moves. (Verified "
                "2026-08-06 on NGSolve 6.2.2604 — Tier-2 fixture "
                "nonlinear_elasticity_neohooke_minus_d.)"
            ),
            (
                "[API] Variation() auto-differentiates the energy "
                "to get BOTH the residual and the tangent, which "
                "is what makes it safer than hand-coding — the "
                "pair cannot drift apart. Note what the "
                "convergence RATE does and does not tell you: "
                "NGSolve differentiates whatever residual form it "
                "is given, so a hand-coded P(F) with a sign error "
                "or a factor-of-2 still converges QUADRATICALLY — "
                "to the wrong displacement. Linear convergence "
                "requires a tangent INCONSISTENT with the "
                "residual, i.e. the two hand-coded separately. A "
                "quadratic rate is therefore no evidence that the "
                "constitutive law is right; verify P(F) against "
                "the Variation() residual directly. Signal: with a "
                "tangent linearised from a different P than the "
                "residual, |r_{k+1}|/|r_k| sits at a roughly "
                "CONSTANT value below 1 for many iterations (a "
                "straight line on a log plot) and the "
                "solvers.Newton residual count is several times "
                "the matched-tangent count, though it still "
                "reaches the same solution; with a consistent "
                "pair — whether or not P is correct — the ratio "
                "collapses superlinearly in a handful of steps. "
                "Cross-check by assembling the hand-coded residual "
                "and the Variation() residual on the same "
                "GridFunction and comparing them entrywise; they "
                "must agree to round-off. (Verified 2026-08-06 on "
                "NGSolve 6.2.2604 — Tier-2 fixture "
                "nonlinear_elasticity_variation_vs_handcoded_"
                "tangent.)"
            ),
            (
                "[Numerical] For nearly-incompressible "
                "(nu -> 0.5): use the F-bar method or a mixed "
                "(u, p) formulation. Volumetric locking here is a "
                "PARTIAL loss, not a collapse — at nu = 0.4999 a "
                "pure-displacement VectorH1 order-1 Cook membrane "
                "recovers roughly half the mixed answer, not "
                "'< 1% of it', so a guard tuned to catch a "
                "near-zero displacement never fires and the "
                "half-right number ships. Raising the "
                "displacement order to 2 already removes most of "
                "the locking on this problem, and at moderate nu "
                "order 1 is fine — the defect is specific to "
                "nu -> 0.5 AND low order. Signal: solve the same "
                "problem three ways and compare tip "
                "displacements — pure VectorH1 order 1, pure "
                "VectorH1 order 2, and mixed (VectorH1 order 2 + "
                "H1 order 1, perturbed Lagrangian). At "
                "nu -> 0.5 the order-1 pure solve is a large but "
                "not tiny fraction of the mixed reference, the "
                "order-2 pure solve agrees with mixed to about a "
                "percent, and repeating at a moderate nu makes "
                "even order 1 agree — that pattern, not an "
                "absolute threshold, is the locking fingerprint. "
                "(Verified 2026-08-06 on NGSolve 6.2.2604 — "
                "Tier-2 fixture "
                "nonlinear_elasticity_incompressible_locking.)"
            ),
            (
                "[Numerical] Cauchy stress: sigma = "
                "(1/J) * F * S * F^T where S = dW/dE is "
                "the 2nd Piola_Kirchhoff stress. Signal: "
                "writing the Cauchy_stress as sigma = "
                "(1/J) * F * P * F^T (P = first_PK = PK1) "
                "mixes up the push_forward — S and P are "
                "different objects (P = F * S); the "
                "resulting GridFunction stress is wrong "
                "by a factor of F. Correct: sigma = "
                "(1/J) * F * S * F^T or sigma = (1/J) * "
                "P * F^T. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Newton dampfactor < 1 helps "
                "when FAR from equilibrium (large load "
                "steps): setting dampfactor = 0.5 or 0.25 in "
                "ngsolve.solvers.Newton restores convergence "
                "where the undamped solve fails. Reducing the "
                "load increment and warm-starting is the "
                "alternative and works too. "
                "Signal: do NOT wait until Newton overshoots the "
                "convergence basin and the residual grows "
                "between iterations — the undamped run does not "
                "usually get that far. It ABORTS inside the "
                "factorisation with 'UmfpackInverse: Numeric "
                "factorization failed', because the first "
                "over-large correction has already driven the "
                "configuration non-physical, and raising maxit "
                "does not help at any value: the trust region is "
                "set by the step, not by the budget. Read the "
                "returned STATUS — the tuple is (status, numit), "
                "status FIRST, 0 converged and -1 gave up — and "
                "confirm the damped run reaches the prescribed "
                "displacement rather than merely returning. Do "
                "not assume damping costs extra iterations "
                "either: on a configuration where the undamped "
                "solve fails outright, the damped one can "
                "converge in the same count, so an "
                "iteration-count regression is not evidence "
                "against it. (Verified 2026-08-06 on NGSolve "
                "6.2.2604 — the residual-growth signal and the "
                "iteration-count trade are both corrected.)"
            ),
            "[API] After solvers.Newton converges, the Cauchy / "
            "PK1 stress passed to VTKOutput must be rebuilt from "
            "the resolved GridFunction (Grad(gfu), Det(I+Grad("
            "gfu)), Inv(F.trans*F) ...), NOT from the symbolic "
            "fes.TrialFunction(). The symbolic version is a "
            "ProxyFunction; VTKOutput.Do() then raises NgException "
            "'cannot evaluate ProxyFunction without userdata' at "
            "the first quadrature evaluation. Signal: that exact "
            "ProxyFunction-userdata text emitted from VTKOutput."
            "Do() at the end of a Newton template that passes "
            "symbolically-built stress through 'coefs=[gfu, "
            "sigma_cauchy]'. (Verified empirically 2026-06-01 — "
            "Layer F catch.)",
            (
                "[API] ngsolve.solvers.Newton (ngsolve."
                "nonlinearsolvers.Newton) returns "
                "(status, numit) — an INTEGER STATUS FIRST, where "
                "0 means converged and -1 means it gave up, then "
                "the iteration count. It is NOT (iters, "
                "convergence_error): there is no error value in "
                "the return at all. Unpacking it the other way "
                "round is the single misread behind both shipped "
                "nonlinear-elasticity templates, which write "
                "`(iters, conv) = solvers.Newton(...)` and then "
                "print the status as an iteration count. The "
                "damage is that the failure test inverts: a "
                "converged solve reports 'iters=0' (which reads "
                "like nothing happened) and a diverged solve "
                "reports 'iters=-1' while whatever threshold you "
                "put on `conv` is really being applied to an "
                "iteration count. Read status == 0 as the "
                "convergence test, and get the residual from your "
                "own a.Apply if you need one. Signal: every load "
                "step prints 'Newton iters=0' no matter how hard "
                "the step was, and a genuinely failing step "
                "prints -1 together with NGSolve's own line "
                "'Warning: Newton might not converge! Error = ' — "
                "that warning, not the return value, is where the "
                "error norm appears. (Verified 2026-08-06 on "
                "NGSolve 6.2.2604 — Tier-2 fixtures "
                "nonlinear_elasticity_load_step_warm_start and "
                "plasticity_load_step_divergence_signature.)"
            ),
        ],
    },
    "phase_field": {
        "description": (
            "Phase-field evolution: Allen-Cahn for interface motion (scalar phase c) "
            "and phase-field fracture (Bourdin-Francfort-Marigo, staggered scheme). "
            "Fracture: coupled displacement u and crack phase d; alternate minimization."
        ),
        "spaces": "H1(mesh, order=k) for scalar phase; VectorH1 + H1 for fracture",
        "solver": "Allen-Cahn: implicit Euler (linear system per step). Fracture: staggered alternating minimization",
        "pitfalls": [
            (
                "[Numerical] Allen-Cahn mass is NOT conserved — use "
                "Cahn-Hilliard (4th order) for mass conservation. "
                "Signal: Integrate(c, mesh) drifts monotonically "
                "(~1-5% per characteristic interface time) in "
                "Allen-Cahn; the same geometry under Cahn-Hilliard "
                "preserves the integral to machine precision. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Interface width epsilon must be "
                "resolved: at least 3-4 elements across the "
                "interface (h << eps). The observable is the "
                "OVER/UNDERSHOOT in c, NOT a solver complaint — "
                "an under-resolved interface does not make Newton "
                "fail. On a deliberately under-resolved case the "
                "fully implicit solve CONVERGES with status 0 in "
                "a handful of iterations while c leaves the "
                "physical interval [0, 1] by a large margin, so "
                "an agent that watches convergence as its safety "
                "check sees success and ships a badly wrong "
                "field. Also note the string `Newton did not "
                "converge after N iterations` does NOT EXIST in "
                "NGSolve/netgen 6.2.2604 — grepping for it never "
                "matches; the real non-convergence wording is "
                "'Warning: Newton might not converge! Error = '. "
                "Signal: after a few Allen-Cahn steps print "
                "min(c) and max(c) over the coefficient vector — "
                "at h/eps well above the resolved regime they sit "
                "clearly outside [0, 1] and the excursion GROWS "
                "as h/eps grows, while a properly resolved eps "
                "keeps c inside the interval; Newton reports "
                "convergence in both cases. Gate on the range of "
                "c, never on the solver status. (Verified "
                "2026-08-06 on NGSolve 6.2.2604 — Tier-2 fixture "
                "phase_field_interface_underresolved.)"
            ),
            (
                "[Numerical] Semi-implicit treatment of W'(c): "
                "evaluate it at c^n (the previous step) and solve "
                "linearly — this avoids the nonlinear solve "
                "entirely. The fully-implicit form with c^{n+1} "
                "inside W'(c) needs a Newton solve every step, "
                "which dominates wall-clock; the speed-up is large "
                "but problem-dependent, so measure it rather than "
                "quoting a factor, and do not assume a fixed inner "
                "iteration count either. The trade-off is "
                "STABILITY: lagging W'(c) caps the usable dt, and "
                "past that cap the semi-implicit field blows up "
                "while Newton stays bounded. Signal: at a "
                "safely small dt both paths agree to a small "
                "relative L2 difference and the semi-implicit run "
                "performs zero nonlinear iterations against "
                "several Newton iterations per step for the "
                "implicit one, with a clearly shorter wall-clock; "
                "raise dt by orders of magnitude and the "
                "semi-implicit c leaves [0, 1] by a wide margin "
                "within a handful of steps while the fully "
                "implicit c stays inside it. Use a smaller dt if "
                "you observe ringing. (Verified 2026-08-06 on "
                "NGSolve 6.2.2604 — Tier-2 fixture "
                "phase_field_semi_implicit_vs_newton.)"
            ),
            (
                "[Numerical] Phase-field fracture: irreversibility "
                "d >= d_prev (crack cannot heal) — enforce "
                "pointwise. Signal: visualization shows the "
                "damage field d decreasing in some elements "
                "between time steps (unphysical 'healing'); the "
                "fracture surface is not monotonic in time. "
                "Enforce via max(d, d_prev) projection after each "
                "solve. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Use the staggered scheme (solve "
                "elasticity, then phase-field, iterate) as the "
                "DEFAULT, not as the slower of two equivalent "
                "options. The premise 'monolithic reaches the same "
                "solution, just in fewer outer iterations' does "
                "not hold from a COLD START: on a "
                "Variation-based AT2 energy monolithic Newton does "
                "not converge from a zero initial guess at all, "
                "at a load where staggered converges in the usual "
                "5-20 outer iterations, each cheaper (two linear "
                "solves instead of one nonlinear). The two DO "
                "agree, but "
                "only once the monolithic solve is warm-started "
                "FROM the staggered answer, at which point it is a "
                "refinement step rather than an alternative. "
                "Signal: cold-start monolithic returns status -1, "
                "prints 'Warning: Newton might not converge! "
                "Error = ', and leaves max(d) ABOVE 1 — a damage "
                "field outside [0, 1] is the tell, because a "
                "diverged phase-field solve still returns a "
                "plausible-looking displacement; the staggered run "
                "at the same load reaches a d_max inside [0, 1] "
                "that is a fixed point of the alternation, and "
                "feeding that to the monolithic solve converges in "
                "a couple of iterations to the same field. Never "
                "report a monolithic result without checking both "
                "status == 0 and max(d) <= 1. (Verified 2026-08-06 "
                "on NGSolve 6.2.2604 — Tier-2 fixture "
                "phase_field_staggered_vs_monolithic.)"
            ),
            (
                "[Numerical] A tension/compression energy split is "
                "what stops a crack growing under compression — "
                "but WHICH split matters, and the psi_plus shipped "
                "in phase_field_fracture_2d (a volumetric/"
                "deviatoric split) does NOT reach 'd ~ 0'. Under "
                "confined uniaxial compression the vol-dev split "
                "only REDUCES the nucleated damage relative to no "
                "split at all; it stays well above a 1e-3 bound, so "
                "that bound fails on the code actually shipped. "
                "Only a SPECTRAL (principal-strain) split drives "
                "compressive damage to negligible while leaving "
                "the tensile answer identical to the unsplit one. "
                "Choose the split deliberately and say which one "
                "you used. Signal: run the same confined uniaxial "
                "load in tension and in compression and compare "
                "max(d) — with no split the two are EQUAL "
                "(compression nucleates exactly as much damage as "
                "tension), with the vol-dev split compression is "
                "reduced but still comfortably above "
                "1e-3, and with the spectral split compression is "
                "negligible while tension matches the unsplit "
                "value. Guard against a vacuous pass: if the "
                "elastic step overwrites the Dirichlet entries, "
                "EVERY split reports max(d) = 0 under compression "
                "and the sanity test measures nothing — confirm "
                "the TENSION case produces damage before "
                "believing the compression case. (Verified "
                "2026-08-06 on NGSolve 6.2.2604 — Tier-2 fixture "
                "phase_field_miehe_split_compression.)"
            ),
            (
                "[Numerical] Length scale l0 must be small "
                "enough relative to specimen size; the "
                "fracture-energy convergence is recovered as "
                "l0 -> 0. Signal: l0 ~ 1/10 of the specimen "
                "characteristic size produces a smeared "
                "fracture zone visibly wider than expected "
                "(captures rough crack location but the peak "
                "load is over-predicted ~20-50% vs the "
                "Griffith analytic load); refining the mesh "
                "WITHOUT also reducing l0 does not help — "
                "both must shrink together with l0 / h ~ 4-8 "
                "preserved. (Audit 2026-06-02.)"
            ),
            (
                "[API] For Cahn-Hilliard: use an H1 x H1 mixed "
                "formulation (chemical potential + phase field). "
                "NGSolve will NOT stop you from doing it the "
                "wrong way. Signal: a single-H1 fourth-order form "
                "built from the Hessian operator, BilinearForm += "
                "InnerProduct(u.Operator('hesse'), v.Operator("
                "'hesse'))*dx on H1(mesh, order=2), ASSEMBLES "
                "cleanly with a non-empty matrix and no exception "
                "at all. There is no `NotImplementedError: H2 "
                "conformity required for biharmonic operator` and "
                "no `coefficient not in BilinearForm space` "
                "anywhere in NGSolve 6.2.2604 — both strings were "
                "fabricated, and the one operator name that DOES "
                "raise is 'biharmonic': u.Operator('biharmonic') "
                "gives NgException 'Operator \"biharmonic\" does "
                "not exist for H1HighOrderFESpace!'. The penalty "
                "is silent non-conformity (C0 Lagrange is not "
                "H2), and it shows up as a DISCRETE KERNEL: embed "
                "a P1 hat function in the order-2 space and its "
                "hesse energy is round-off while its H1 energy is "
                "O(1), so the operator has a large null space. "
                "Solving it then fails in two different ways "
                "depending on the factorisation you picked — "
                "umfpack raises 'UmfpackInverse: Numeric "
                "factorization failed.' while sparsecholesky "
                "returns NaN WITHOUT raising — so assert the "
                "solution is finite rather than relying on an "
                "exception. Use mixed (c, mu) — a Ciarlet-Raviart "
                "H1xH1 pair converges cleanly against a "
                "manufactured solution — or HDivDiv/HHJ. (Verified "
                "2026-08-06 on NGSolve 6.2.2604 — Tier-2 fixture "
                "phase_field_cahn_hilliard_hesse_nonconforming; "
                "the previously pinned nnz figure did not "
                "reproduce on this netgen build and is removed "
                "rather than re-pinned.)"
            ),
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# GENERATORS dict
# ─────────────────────────────────────────────────────────────────────────────

GENERATORS = {
    "dg_methods_2d":                _dg_methods_2d,
    "contact_2d":                   _contact_2d,
    "time_dependent_ns_2d":         _time_dependent_ns_2d,
    "mhd_2d":                       _mhd_2d,
    "hdivdiv_2d":                   _hdivdiv_2d,
    "nonlinear_elasticity_2d":      _nonlinear_elasticity_2d,
    "nonlinear_elasticity_3d":      _nonlinear_elasticity_3d,
    "phase_field_2d":               _phase_field_2d,
    "phase_field_fracture_2d":      _phase_field_fracture_2d,
}
