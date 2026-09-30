"""Navier-Stokes generators for FEniCSx/dolfinx.

Variants: 2d, 3d, channel_cylinder, channel_cylinder_transient (a contract: its solve is the model's)
"""


KNOWLEDGE = {
    # ─────────────────────────────────────────────────────────────────
    # _SERVING_STATUS (added 2026-08-03)
    # This dict is SHADOWED and is NOT what an agent receives.
    # fenics/backend.py:get_knowledge() returns
    # src/tools/deep_knowledge.py::_FENICS_KNOWLEDGE['navier_stokes'] for this
    # physics and never falls through to here. Editing the pitfalls
    # below changes nothing an agent can see. The claims here were NOT
    # re-verified in the 2026-08-03 execution pass for exactly that
    # reason — treat them as unverified history, and make corrections
    # in deep_knowledge.py instead.
    # ─────────────────────────────────────────────────────────────────
    "description": "Incompressible Navier-Stokes with Taylor-Hood elements (P2/P1)",
    "weak_form": "nu*(grad(u),grad(v))*dx + (grad(u)*u,v)*dx - p*div(v)*dx - q*div(u)*dx = (f,v)*dx",
    "function_space": "Mixed: P2 velocity + P1 pressure (Taylor-Hood, inf-sup stable)",
    "solver": "Newton iteration with LU (MUMPS) for direct solve",
    "pitfalls": [
        # Promoted from one-line tips to [Category] + Signal: pitfalls
        # (2026-06-02 audit pass 20). Each names a real dolfinx symptom.
        "[Numerical] Must use inf-sup stable element pair "
        "(Taylor-Hood P2/P1 — P2 velocity + P1 pressure). Equal-order "
        "P1/P1 violates LBB. Signal: a dolfinx mixed_element with "
        "matching basix.ufl.element('Lagrange', 1) on both velocity "
        "and pressure yields a checkerboard pressure pattern; the "
        "XDMFFile output shows oscillations between adjacent DOFs with "
        "amplitude ~ 100% of mean. (Audit 2026-06-02.)",
        "[Input] Pressure needs pinning (dirichletbc at one point) "
        "for enclosed flows — pressure is determined only up to a "
        "constant. Signal: the dolfinx NewtonSolver on a closed-domain "
        "NS problem without pressure pinning either reports "
        "DIVERGED_BREAKDOWN from KSPSolve or returns a Function with a "
        "huge additive offset (the pressure component drifts by O(1e6) "
        "between solver runs). (Audit 2026-06-02.)",
        "[Numerical] High Re requires stabilization (SUPG) or a "
        "finer mesh — the Galerkin form is unstable for "
        "convection-dominated NS. Signal: a Taylor-Hood mixed-element "
        "solve at Re > 500 with no SUPG produces visible wiggles in "
        "the velocity Function near the inflow / obstacle; the "
        "XDMFFile output shows oscillations growing with each Newton "
        "iteration. (Audit 2026-06-02.)",
        "[Numerical] Newton may not converge for Re > 500 without "
        "load-step continuation (ramp Re from 1 to target). Signal: "
        "the dolfinx NewtonSolver raises NoConvergence or the residual "
        "ratio oscillates between two values; ramping nu via "
        "fem.Constant(domain, nu_initial) then updating in a loop "
        "produces monotonic convergence. (Audit 2026-06-02.)",
        "[API] Use basix.ufl.element() and basix.ufl.mixed_element() "
        "in modern dolfinx — the old VectorElement/MixedElement from "
        "ufl alone is deprecated. Signal: passing ufl.MixedElement to "
        "fem.functionspace raises DeprecationWarning or "
        "AttributeError; the correct pattern is "
        "basix.ufl.mixed_element([P2_vec, P1]). (Audit 2026-06-02.)",
    ],
    "materials": {
        "Re": {"range": [1, 10000], "unit": "dimensionless (Reynolds number)"},
        "nu": {"range": [1e-6, 1.0], "unit": "m^2/s (kinematic viscosity = 1/Re)"},
    },
}

VARIANTS = ["2d", "3d", "channel_cylinder", "channel_cylinder_transient"]


def generate(variant: str, params: dict) -> str:
    """Dispatch to the appropriate Navier-Stokes variant."""
    generators = {
        "2d": _navier_stokes_2d,
        "3d": _navier_stokes_cavity_3d,
        "channel_cylinder": _navier_stokes_channel_cylinder,
        "channel_cylinder_transient": _navier_stokes_channel_cylinder_transient,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(f"Unknown Navier-Stokes variant: {variant!r}. Available: {list(generators)}")
    return gen(params)


def _navier_stokes_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    Re = params.get("Re", 100)
    nx = params.get("nx", 32)
    ny = params.get("ny", 32)
    return f'''\
"""Incompressible Navier-Stokes — FEniCSx/dolfinx
Taylor-Hood P2/P1 elements, Newton iteration.
"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import ufl
import numpy as np
from basix.ufl import element, mixed_element
from petsc4py import PETSc

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
gdim = domain.geometry.dim

# Taylor-Hood elements: P2 velocity + P1 pressure
P2 = element("Lagrange", domain.topology.cell_name(), 2, shape=(gdim,))
P1 = element("Lagrange", domain.topology.cell_name(), 1)
TH = mixed_element([P2, P1])
W = fem.functionspace(domain, TH)

# Boundary conditions
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

# No-slip on all walls
def walls(x):
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0)

def lid(x):
    return np.isclose(x[1], 1.0)

# Velocity sub-space (must collapse for BC application)
V, V_map = W.sub(0).collapse()

# No-slip walls: use Function (not constant) for sub-space BC
noslip = fem.Function(V)
noslip.x.array[:] = 0.0
wall_facets = mesh.locate_entities_boundary(domain, fdim, walls)
wall_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
bc_walls = fem.dirichletbc(noslip, wall_dofs, W.sub(0))

# Lid velocity u=(1,0)
lid_velocity = fem.Function(V)
lid_velocity.interpolate(lambda x: (np.ones_like(x[0]), np.zeros_like(x[0])))
lid_facets = mesh.locate_entities_boundary(domain, fdim, lid)
lid_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, lid_facets)
bc_lid = fem.dirichletbc(lid_velocity, lid_dofs, W.sub(0))

# Pressure pin (single point)
Q, Q_map = W.sub(1).collapse()
zero_p = fem.Function(Q)
zero_p.x.array[:] = 0.0
p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0))
bc_pressure = fem.dirichletbc(zero_p, p_dofs, W.sub(1))

bcs = [bc_walls, bc_lid, bc_pressure]

# Weak form: Navier-Stokes
w = fem.Function(W)
(u, p) = ufl.split(w)
(v, q) = ufl.TestFunctions(W)

nu = fem.Constant(domain, default_scalar_type(1.0 / {Re}))
f = fem.Constant(domain, default_scalar_type((0.0, 0.0)))

F = (
    nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
    - p * ufl.div(v) * ufl.dx
    - q * ufl.div(u) * ufl.dx
    - ufl.dot(f, v) * ufl.dx
)

# Newton solver
problem = NonlinearProblem(F, w, bcs=bcs, petsc_options_prefix="ns",
    petsc_options={{"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
                   "snes_rtol": 1e-6, "snes_max_it": 50, "snes_monitor": None}})
problem.solve()
reason = problem.solver.getConvergedReason()
its = problem.solver.getIterationNumber()
print(f"Newton solver: {{its}} iterations, converged reason={{reason}}")

# Extract velocity and pressure
(u_sol, p_sol) = w.split()

# Interpolate P2 velocity to P1 for XDMF output
V_out = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
u_out = fem.Function(V_out, name="velocity")
u_out.interpolate(u_sol)

P_out = fem.functionspace(domain, ("Lagrange", 1))
p_out = fem.Function(P_out, name="pressure")
p_out.interpolate(p_sol)

# Output
from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "velocity.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_out)
with XDMFFile(domain.comm, "pressure.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(p_out)

# Statistics
u_arr = u_out.x.array.reshape(-1, gdim)
u_mag = np.linalg.norm(u_arr, axis=1)
print(f"Velocity: max |u| = {{u_mag.max():.6e}}")
print(f"Pressure: min(p) = {{p_out.x.array.min():.6e}}, max(p) = {{p_out.x.array.max():.6e}}")
print(f"DOFs: {{W.dofmap.index_map.size_global}}")
'''


def _navier_stokes_cavity_3d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    Re = params.get("Re", 100)
    n = params.get("n", 12)
    return f'''\
"""3D Lid-driven cavity — incompressible Navier-Stokes"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import ufl
import numpy as np
from basix.ufl import element, mixed_element
from petsc4py import PETSc

domain = mesh.create_unit_cube(MPI.COMM_WORLD, {n}, {n}, {n}, mesh.CellType.tetrahedron)
gdim = 3

P2 = element("Lagrange", domain.topology.cell_name(), 2, shape=(gdim,))
P1 = element("Lagrange", domain.topology.cell_name(), 1)
TH = mixed_element([P2, P1])
W = fem.functionspace(domain, TH)

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

def walls(x):
    return (np.isclose(x[0], 0) | np.isclose(x[0], 1) |
            np.isclose(x[1], 0) |
            np.isclose(x[2], 0) | np.isclose(x[2], 1))

def lid(x):
    return np.isclose(x[1], 1.0)

V, _ = W.sub(0).collapse()
noslip = fem.Function(V)
noslip.x.array[:] = 0.0
wall_facets = mesh.locate_entities_boundary(domain, fdim, walls)
wall_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
bc_walls = fem.dirichletbc(noslip, wall_dofs, W.sub(0))

lid_vel = fem.Function(V)
lid_vel.interpolate(lambda x: (np.ones_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0])))
lid_facets = mesh.locate_entities_boundary(domain, fdim, lid)
lid_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, lid_facets)
bc_lid = fem.dirichletbc(lid_vel, lid_dofs, W.sub(0))

Q, _ = W.sub(1).collapse()
zero_p = fem.Function(Q)
zero_p.x.array[:] = 0.0
p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0) & np.isclose(x[2], 0))
bc_p = fem.dirichletbc(zero_p, p_dofs, W.sub(1))
bcs = [bc_walls, bc_lid, bc_p]

w = fem.Function(W)
(u, p) = ufl.split(w)
(v, q) = ufl.TestFunctions(W)
nu = fem.Constant(domain, default_scalar_type(1.0 / {Re}))
f = fem.Constant(domain, default_scalar_type((0.0, 0.0, 0.0)))

F = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
     + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
     - p * ufl.div(v) * ufl.dx - q * ufl.div(u) * ufl.dx
     - ufl.dot(f, v) * ufl.dx)

problem = NonlinearProblem(F, w, bcs=bcs, petsc_options_prefix="ns3d",
    petsc_options={{"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
                   "snes_rtol": 1e-5, "snes_max_it": 30, "snes_monitor": None}})
problem.solve()
its = problem.solver.getIterationNumber()
print(f"Newton: {{its}} iterations")
(u_sol, p_sol) = w.split()

V_out = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
u_out = fem.Function(V_out, name="velocity")
u_out.interpolate(u_sol)
P_out = fem.functionspace(domain, ("Lagrange", 1))
p_out = fem.Function(P_out, name="pressure")
p_out.interpolate(p_sol)

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "velocity.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_out)
with XDMFFile(domain.comm, "pressure.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(p_out)

u_arr = u_out.x.array.reshape(-1, gdim)
print(f"max |u| = {{np.linalg.norm(u_arr, axis=1).max():.6e}}")
print(f"DOFs: {{W.dofmap.index_map.size_global}}")
'''


def _navier_stokes_channel_cylinder(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx script. Requires Gmsh.

    Every problem dimension is a parameter. The defaults describe the widely
    used confined-cylinder geometry because it is a convenient starting shape,
    NOT because this template is for that benchmark -- set them to your own
    channel, cylinder and inlet speed and the script is unchanged otherwise.

    The docstring used to say "all parameter defaults are placeholders" while
    the channel, the cylinder and the inlet speed were literals inside the
    emitted source with no parameter at all. A label is not a guard.

    REYNOLDS NUMBER, AND THE BUG THIS FIXES: for a parabolic inlet the MEAN
    speed is 2/3 of the PEAK, and the confined-cylinder Reynolds number is
    defined on the mean. This template used to set nu from the PEAK, so a run
    asked for Re=20 was silently solved at Re=13.3 -- every line of the setup
    looking correct, at the wrong Reynolds number, which is precisely the trap
    the served knowledge for this physics warns about. nu is now derived from
    the mean, and the script prints both speeds and the realised Re so the
    reader can see which convention was used.
    """
    Re = params.get("Re", 20)
    L = params.get("length", 2.2)
    H = params.get("height", 0.41)
    cx = params.get("cyl_x", 0.2)
    cy = params.get("cyl_y", 0.2)
    radius = params.get("cyl_radius", 0.05)
    u_peak = params.get("u_peak", 0.3)
    # RELATIVE TO THE CYLINDER, NOT ABSOLUTE. A characteristic length of
    # 0.02 means "resolve the boundary layer" only on a 0.41-high channel
    # with a 0.1 cylinder; scale the geometry and the same number silently
    # becomes either a wasteful mesh or an unresolved one. Expressed as a
    # fraction of the cylinder DIAMETER it keeps its meaning at any size.
    # An explicit mesh_size still wins, for anyone who wants an absolute one.
    # 5 per diameter reproduces the previous absolute default of 0.02 on the
    # standard 0.1 cylinder EXACTLY, so making this relative changes no
    # existing run's cost. A ratio chosen for elegance rather than continuity
    # would have silently halved the mesh size and doubled every run.
    cells_per_diameter = params.get("cells_per_diameter", 5.0)
    mesh_size = params.get("mesh_size", 2.0 * radius / cells_per_diameter)
    return f'''\
"""Navier-Stokes: channel flow around cylinder — FEniCSx + Gmsh
Parabolic inlet, no-slip walls, cylinder obstacle.
"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import ufl
import numpy as np
from basix.ufl import element, mixed_element

# Generate channel with cylinder using Gmsh
import gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("channel-cyl")
L, H = {L}, {H}
cx, cy, r = {cx}, {cy}, {radius}
rect = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
cyl = gmsh.model.occ.addDisk(cx, cy, 0, r, r)
gmsh.model.occ.cut([(2, rect)], [(2, cyl)])
gmsh.model.occ.synchronize()
surfaces = gmsh.model.getEntities(2)
gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], tag=1)
curves = gmsh.model.getEntities(1)
for i, c in enumerate(curves):
    gmsh.model.addPhysicalGroup(1, [c[1]], tag=i+1)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", {mesh_size})
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", {mesh_size} * 0.15)
gmsh.model.mesh.generate(2)
gmsh.write("channel_cyl.msh")
gmsh.finalize()

from dolfinx.io.gmsh import read_from_msh
mesh_data = read_from_msh("channel_cyl.msh", MPI.COMM_WORLD, gdim=2)
domain = mesh_data.mesh
gdim = domain.geometry.dim

# Taylor-Hood P2/P1
P2 = element("Lagrange", domain.topology.cell_name(), 2, shape=(gdim,))
P1 = element("Lagrange", domain.topology.cell_name(), 1)
TH = mixed_element([P2, P1])
W = fem.functionspace(domain, TH)

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

# BCs
V, _ = W.sub(0).collapse()
Q, _ = W.sub(1).collapse()

# No-slip on walls + cylinder
def walls(x):
    return np.isclose(x[1], 0.0) | np.isclose(x[1], H)

def cylinder_surf(x):
    return ((x[0] - cx)**2 + (x[1] - cy)**2) < (r * 1.5)**2

def inlet(x):
    return np.isclose(x[0], 0.0)

noslip = fem.Function(V)
noslip.x.array[:] = 0.0

wall_facets = mesh.locate_entities_boundary(domain, fdim, walls)
wall_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
bc_walls = fem.dirichletbc(noslip, wall_dofs, W.sub(0))

cyl_facets = mesh.locate_entities_boundary(domain, fdim, cylinder_surf)
cyl_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, cyl_facets)
bc_cyl = fem.dirichletbc(noslip, cyl_dofs, W.sub(0))

# Parabolic inlet: u_x = 4*U_m*y*(H-y)/H^2, u_y = 0
# U_m is the PEAK speed (the profile's maximum, at mid-height).
U_m = {u_peak}
inlet_vel = fem.Function(V)
inlet_vel.interpolate(lambda x: (4 * U_m * x[1] * (H - x[1]) / H**2, np.zeros_like(x[0])))
inlet_facets = mesh.locate_entities_boundary(domain, fdim, inlet)
inlet_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, inlet_facets)
bc_inlet = fem.dirichletbc(inlet_vel, inlet_dofs, W.sub(0))

# Pressure pin at outlet
def outlet_corner(x):
    return np.isclose(x[0], L) & np.isclose(x[1], 0.0, atol=0.05)
zero_p = fem.Function(Q)
zero_p.x.array[:] = 0.0
p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), outlet_corner)
bc_p = fem.dirichletbc(zero_p, p_dofs, W.sub(1))

bcs = [bc_walls, bc_cyl, bc_inlet, bc_p]

# NS weak form
w = fem.Function(W)
(u, p) = ufl.split(w)
(v, q) = ufl.TestFunctions(W)
# The Reynolds number of a confined cylinder is defined on the MEAN inlet
# speed and the cylinder DIAMETER. For a parabolic profile the mean is 2/3 of
# the peak, so deriving nu from U_m directly would put the run at 2/3 of the
# Re you asked for while every line above still looked right.
U_bar = 2.0 * U_m / 3.0          # mean inlet speed
D = 2 * r                        # cylinder diameter
nu_value = U_bar * D / {Re}
nu = fem.Constant(domain, default_scalar_type(nu_value))
# THE CHECK THAT CAN ACTUALLY FAIL. Recomputing Re from a viscosity that was
# DEFINED as U_bar*D/Re returns the requested Re whatever happens -- that is
# arithmetic, not verification, and an earlier version of this template
# printed exactly that. What can genuinely disagree is the ANALYTIC mean
# above versus the mean of the profile this script actually imposes, so
# integrate the imposed profile numerically and compare the two.
_ys = np.linspace(0.0, H, 2001)
# numpy 2.0 renamed trapz to trapezoid; the FEniCSx environment may still ship numpy 1.x
# (1.26.4 here), where trapezoid does not exist and the template stopped before it solved
_trapz = getattr(np, "trapezoid", None) or np.trapz
_ubar_numeric = _trapz(4 * U_m * _ys * (H - _ys) / H**2, _ys) / H
if domain.comm.rank == 0:
    print(f"inlet peak U_m = {{U_m:.6g}}, analytic mean U_bar = {{U_bar:.6g}}, "
          f"D = {{D:.6g}}, nu = {{nu_value:.6e}}")
    print(f"mean of the imposed profile, integrated = {{_ubar_numeric:.6g}}")
    _rel = abs(_ubar_numeric - U_bar) / U_bar
    print(f"analytic vs imposed mean differ by {{_rel:.2e}} "
          f"({{'OK' if _rel < 1e-6 else 'MISMATCH — the profile and the '
              'non-dimensionalisation disagree, so Re is not what you asked for'}})")
    print(f"requested Re = {Re}")
f = fem.Constant(domain, default_scalar_type((0.0, 0.0)))
F = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
     + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
     - p * ufl.div(v) * ufl.dx - q * ufl.div(u) * ufl.dx
     - ufl.dot(f, v) * ufl.dx)

problem = NonlinearProblem(F, w, bcs=bcs, petsc_options_prefix="ns",
    petsc_options={{"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
                   "snes_rtol": 1e-6, "snes_max_it": 50, "snes_monitor": None}})
problem.solve()
its = problem.solver.getIterationNumber()
print(f"Newton: {{its}} iterations")

(u_sol, p_sol) = w.split()
V_out = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
u_out = fem.Function(V_out, name="velocity")
u_out.interpolate(u_sol)
P_out = fem.functionspace(domain, ("Lagrange", 1))
p_out = fem.Function(P_out, name="pressure")
p_out.interpolate(p_sol)

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "velocity.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_out)
with XDMFFile(domain.comm, "pressure.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(p_out)

u_arr = u_out.x.array.reshape(-1, gdim)
print(f"max |u| = {{np.linalg.norm(u_arr, axis=1).max():.6e}}")
print(f"Re = {Re}, DOFs = {{W.dofmap.index_map.size_global}}")
'''


def _navier_stokes_channel_cylinder_transient(params: dict) -> str:
    """CONTRACT: unsteady flow past an obstacle in a channel, and the forces on it.

    THE PHYSICS AND THE SOLVE ARE LEFT TO THE MODEL (Alexander, 2026-09-29: "keep it, solve
    left out"). This variant was first served (2026-09-28) as a complete unsteady solver fitted
    to one published channel -- a working solve for the very task, which openPASO does not serve.
    Served now: reading the mesh that generate_mesh(geometry="channel_cylinder") writes, with
    its named boundaries; the force on a named boundary from ANY velocity and pressure; the
    shedding frequency from the lift history; the result files and the pictures, including the
    field the web interface plays. The spaces, the boundary conditions, the weak form, the time
    stepping and the solve are the model's, in the marked region, which stops the run while it
    is unfilled. tests/test_the_cylinder_wake_contract_leaves_the_solve_out.py fills it with a
    validation fill (never served) and runs it.

    No problem data is written in: every SETTING is None until the model sets it from its
    problem, and the served code refuses to start while one is. Optional params fill them:
    mesh_file, obstacle, u_ref, d_ref, rho, t_end, analysis_from, frame_every.
    """
    def _setting(key, default=None):
        return repr(params.get(key, default))

    settings = (
        f"MESH_FILE = {_setting('mesh_file')}      # the ABSOLUTE path generate_mesh returned: the run starts in a folder of its own\n"
        f"OBSTACLE = {_setting('obstacle', 'cylinder')}     # the named boundary the forces act on\n"
        f"U_REF = {_setting('u_ref')}          # the reference speed your problem defines the force coefficients with\n"
        f"D_REF = {_setting('d_ref')}          # the reference length (for a cylinder, its diameter)\n"
        f"RHO = {_setting('rho', 1.0)}             # the density your stress is scaled by: 1 when you solve with p/rho and nu\n"
        f"T_END = {_setting('t_end')}          # the end time your time loop reaches\n"
        f"ANALYSIS_FROM = {_setting('analysis_from')}  # the forces and the shedding are read from this time on, after the start-up\n"
        f"FRAME_EVERY = {_setting('frame_every', 10)}       # keep a field frame (files, pictures) every this many recorded steps\n")
    return ('"""Unsteady incompressible flow past an obstacle in a channel -- a FEniCSx CONTRACT\n'
            "\n"
            "THIS IS A CONTRACT, NOT A PROGRAM. Served: reading the mesh that\n"
            'generate_mesh(geometry="channel_cylinder", params=...) writes, with its named boundaries;\n'
            "the force on a named boundary from ANY velocity and pressure you hand to record(); the\n"
            "shedding frequency from the lift history; the result files and the pictures. Yours, in the\n"
            "marked region: the function spaces, the boundary conditions, the weak form, the time\n"
            "stepping and the solve. The region stops the run until it is filled.\n"
            '"""\n'
            "# ── SETTINGS ─ from YOUR problem; the script refuses to start while one is None ──────\n"
            + settings +
            "# ─────────────────────────────────────────────────────────────────────────────\n"
            + _CYLINDER_TRANSIENT_CONTRACT)


_CYLINDER_TRANSIENT_CONTRACT = r"""import json
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.io import XDMFFile
from dolfinx.io import gmsh as gmshio

comm = MPI.COMM_WORLD
_unset = [k for k in ("MESH_FILE", "U_REF", "D_REF", "T_END", "ANALYSIS_FROM") if globals()[k] is None]
if _unset:
    raise SystemExit("set these in SETTINGS first, from your problem: " + ", ".join(_unset))

# ── the mesh generate_mesh wrote, with its named boundaries (served) ─────────────
_md = gmshio.read_from_msh(MESH_FILE, comm, rank=0, gdim=2)
domain, facet_tags = _md.mesh, _md.facet_tags
BOUNDARY = {name: g.tag for name, g in _md.physical_groups.items() if g.dim == 1}
if OBSTACLE not in BOUNDARY:
    raise SystemExit(f"the mesh names no boundary {OBSTACLE!r}; it names {sorted(BOUNDARY)}")
gdim = domain.geometry.dim
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
# ds(BOUNDARY[name]) integrates over one named boundary, ds over all of them
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)


class _Record:
    # record(t, u, p, nu) after every time step (served): the force on OBSTACLE from the u and p
    # it is handed, the net volume flux through the whole boundary, and every FRAME_EVERY-th
    # call a frame of the fields. u and p may be Functions or the two parts of a mixed Function
    # (ufl.split); hand the SAME objects every step, so the forms are compiled once.

    def __init__(self):
        self.rows, self.frames, self.frame_times = [], [], []
        self.calls, self._key = 0, None
        self._next_note = 0.1 * T_END       # a progress line every tenth of the run
        self._P1 = fem.functionspace(domain, ("Lagrange", 1))
        self._V1 = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
        self._u_out = fem.Function(self._V1, name="velocity")
        self._p_out = fem.Function(self._P1, name="pressure")
        self._w_out = fem.Function(self._P1, name="vorticity")
        self._files = [XDMFFile(comm, "velocity.xdmf", "w"), XDMFFile(comm, "pressure.xdmf", "w")]
        for f in self._files:
            f.write_mesh(domain)
        im = self._P1.dofmap.index_map
        self._own = im.size_local
        # the mesh's own triangles in the global numbering, for the pictures
        cells = self._P1.dofmap.list[: domain.topology.index_map(tdim).size_local]
        self._tri = comm.gather(im.local_to_global(cells.reshape(-1)).reshape(-1, 3), root=0)
        self._xy = comm.gather(self._P1.tabulate_dof_coordinates()[: self._own, :2], root=0)

    def _compile(self, u, p, nu):
        n = ufl.FacetNormal(domain)
        sigma = -p * ufl.Identity(gdim) + nu * (ufl.grad(u) + ufl.grad(u).T)
        # n is the FLUID's outward normal: it points INTO the obstacle, so the force ON the
        # obstacle is -int(sigma n) over its boundary
        on = ds(BOUNDARY[OBSTACLE])
        self._drag = fem.form(-ufl.dot(sigma * n, ufl.as_vector((1.0, 0.0))) * on)
        self._lift = fem.form(-ufl.dot(sigma * n, ufl.as_vector((0.0, 1.0))) * on)
        self._net = fem.form(ufl.dot(u, n) * ds)
        self._through = fem.form(abs(ufl.dot(u, n)) * ds)
        ip = self._P1.element.interpolation_points
        ip = ip() if callable(ip) else ip
        ipv = self._V1.element.interpolation_points
        ipv = ipv() if callable(ipv) else ipv
        self._u_expr = fem.Expression(u, ipv)
        self._p_expr = fem.Expression(p, ip)
        self._w_expr = fem.Expression(u[1].dx(0) - u[0].dx(1), ip)

    def __call__(self, t, u, p, nu):
        key = (repr(u), repr(p), repr(nu))
        if key != self._key:
            if self._key is not None and comm.rank == 0:
                print("record(): new u, p or nu objects -- the forms are compiled again")
            self._compile(u, p, nu)
            self._key = key
        self.calls += 1

        def total(form):
            return comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)

        coef = 2.0 / (RHO * U_REF * U_REF * D_REF)
        row = (float(t), coef * total(self._drag), coef * total(self._lift),
               total(self._net), total(self._through))
        if not np.all(np.isfinite(row)):
            raise SystemExit(f"the force became non-finite at t = {t:.6g}: the solve diverged")
        self.rows.append(row)
        if comm.rank == 0 and t >= self._next_note * (1 - 1e-9):
            print(f"t = {t:.6g}: C_D = {row[1]:.4f}, C_L = {row[2]:.4f}", flush=True)
            while self._next_note <= t * (1 + 1e-9):
                self._next_note += 0.1 * T_END
        if self.calls % FRAME_EVERY == 0:
            self._u_out.interpolate(self._u_expr)
            self._p_out.interpolate(self._p_expr)
            self._w_out.interpolate(self._w_expr)
            self._files[0].write_function(self._u_out, t)
            self._files[1].write_function(self._p_out, t)
            parts = comm.gather(self._w_out.x.array[: self._own].copy(), root=0)
            if comm.rank == 0:
                self.frames.append(np.concatenate(parts))
                self.frame_times.append(float(t))

    def finish(self):
        for f in self._files:
            f.close()
        if not self.rows:
            raise SystemExit("record(t, u, p, nu) was never called: there are no forces to report")
        if comm.rank == 0:
            _report(np.array(self.rows), self.frames, self.frame_times,
                    np.concatenate(self._xy), np.concatenate(self._tri))


def _report(arr, frames, frame_times, xy, tri_idx):
    # forces.csv, results_summary.json and the pictures, from what record() kept (served)
    np.savetxt("forces.csv", arr[:, :3], delimiter=",", header="t,C_D,C_L", comments="")
    if arr[-1, 0] < T_END * (1 - 1e-9):
        print(f"WARNING: the last recorded step is t = {arr[-1, 0]:.6g}, before T_END = {T_END}")
    win = arr[arr[:, 0] >= ANALYSIS_FROM]
    if len(win) < 2:
        raise SystemExit(f"fewer than two recorded steps at t >= ANALYSIS_FROM = {ANALYSIS_FROM}")
    t_w, cd_w, cl_w = win[:, 0], win[:, 1], win[:, 2]
    s = cl_w - cl_w.mean()
    # upward zero crossings of the lift, linearly interpolated: one per shedding period
    idx = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    crossings = t_w[idx] - s[idx] * (t_w[idx + 1] - t_w[idx]) / (s[idx + 1] - s[idx])
    periods = np.diff(crossings)
    amplitude = 0.5 * (cl_w.max() - cl_w.min())
    settled = bool(len(periods) >= 3 and amplitude > 1e-3
                   and periods.std() <= 0.05 * periods.mean())
    St = float(D_REF / (periods.mean() * U_REF)) if settled else None
    # an incompressible field lets through the boundary as much as enters it
    leak = float(np.max(np.abs(win[:, 3]) / np.maximum(0.5 * win[:, 4], 1e-300)))
    summary = {"u_ref": U_REF, "d_ref": D_REF, "rho": RHO, "t_end": T_END,
               "analysis_from": ANALYSIS_FROM, "steps_recorded": int(len(arr)),
               "C_D_max": float(cd_w.max()), "C_D_mean": float(cd_w.mean()),
               "C_L_max": float(cl_w.max()), "C_L_min": float(cl_w.min()),
               "C_L_amplitude": float(amplitude),
               "shedding_periods_used": int(len(periods)), "shedding_settled": settled,
               "Strouhal": St, "net_boundary_flux_over_inflow_max": leak}
    images = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        from matplotlib.animation import FuncAnimation, PillowWriter
        late = arr[:, 0] >= min(ANALYSIS_FROM, 0.25 * T_END)      # past the impulsive start
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 4.4), sharex=True)
        a1.plot(arr[late, 0], arr[late, 1], color="black", lw=1.0)
        a2.plot(arr[late, 0], arr[late, 2], color="black", lw=1.0)
        a1.set_ylabel("C_D")
        a2.set_ylabel("C_L")
        a2.set_xlabel("t")
        for a in (a1, a2):
            a.axvline(ANALYSIS_FROM, color="0.6", lw=0.8, ls="--")
        fig.tight_layout()
        fig.savefig("forces.png", dpi=110)
        plt.close(fig)
        images.append("forces.png")
        if frames:
            tri = mtri.Triangulation(xy[:, 0], xy[:, 1], tri_idx)
            x0, x1 = float(xy[:, 0].min()), float(xy[:, 0].max())
            y0, y1 = float(xy[:, 1].min()), float(xy[:, 1].max())
            lim = float(np.percentile(np.abs(frames[-1]), 99)) or 1.0
            levels = np.linspace(-lim, lim, 41)
            fig, ax = plt.subplots(figsize=(9.0, max(1.6, 9.0 * (y1 - y0) / (x1 - x0) + 0.5)))

            def draw(i):
                ax.clear()
                ax.tricontourf(tri, np.clip(frames[i], -lim, lim), levels=levels, cmap="RdBu_r")
                ax.set_aspect("equal")
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)
                ax.set_title(f"vorticity, t = {frame_times[i]:.3g}", fontsize=9)
                ax.tick_params(labelsize=7)

            draw(len(frames) - 1)
            fig.tight_layout()
            fig.savefig("wake.png", dpi=120)
            images.append("wake.png")
            first = max(0, len(frames) - 60)                 # the last 60 frames
            anim = FuncAnimation(fig, lambda j: draw(first + j), frames=len(frames) - first)
            anim.save("wake.gif", writer=PillowWriter(fps=10), dpi=80)
            plt.close(fig)
            images.append("wake.gif")
            # THE WEB INTERFACE'S OWN ANIMATION: the field on a regular grid ("field_series"),
            # one frame per snapshot, quantised to 8 bits between a symmetric clip; grid cells
            # outside the mesh (the obstacle) masked. Linear between the P1 vertex values,
            # sampled at the grid's cell centres.
            import base64
            nxg = 440
            dxg = (x1 - x0) / nxg
            nyg = max(1, int(round((y1 - y0) / dxg)))
            dyg = (y1 - y0) / nyg
            GX, GY = np.meshgrid(x0 + (np.arange(nxg) + 0.5) * dxg, y0 + (np.arange(nyg) + 0.5) * dyg)
            sel = list(range(first, len(frames)))
            grid = np.empty((len(sel), nyg, nxg), dtype=np.float32)
            for j, i in enumerate(sel):
                grid[j] = np.ma.filled(mtri.LinearTriInterpolator(tri, frames[i])(GX, GY).astype(float), np.nan)
            valid = ~np.isnan(grid[0])
            finite = grid[:, valid]
            mag = float(np.percentile(np.abs(finite), 92.0)) or 1.0
            q = np.nan_to_num(np.clip((grid + mag) / (2.0 * mag), 0.0, 1.0), nan=0.5)
            series = {
                "kind": "field_series", "format_version": 1, "field": "vorticity", "unit": "1/s",
                "nx": nxg, "ny": nyg, "x0": x0, "y0": y0, "dx": dxg, "dy": dyg,
                "vmin": round(-mag, 6), "vmax": round(mag, 6), "fps": 12,
                "times": [round(float(frame_times[i]), 6) for i in sel],
                "provenance": {
                    "true_min": round(float(np.nanmin(finite)), 6),
                    "true_max": round(float(np.nanmax(finite)), 6),
                    "clip_percentile": 92.0,
                    "saturated_fraction": round(float(np.count_nonzero(np.abs(finite) >= mag)) / finite.size, 5),
                    "quantisation_step": round(2.0 * mag / 255.0, 8), "levels": 256,
                    "interpolation": "linear between the P1 vertex values, sampled at cell centres",
                    "solver": "FEniCSx",
                    "source": "openPASO contract navier_stokes/channel_cylinder_transient, served part"},
                "mask": base64.b64encode(valid.astype(np.uint8).tobytes()).decode(),
                "frames": base64.b64encode(np.round(q * 255).astype(np.uint8).tobytes()).decode(),
            }
            with open("wake_vorticity.json", "w") as fh:
                json.dump(series, fh)
            images.append("wake_vorticity.json")
    except ImportError as exc:
        print(f"no pictures: {exc} -- this environment has no matplotlib")
    summary["images"] = images
    with open("results_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    if images:
        print("pictures written: " + ", ".join(images))
    print(f"over t >= {ANALYSIS_FROM}: C_D max {cd_w.max():.4f} (mean {cd_w.mean():.4f}), "
          f"C_L max {cl_w.max():.4f}, min {cl_w.min():.4f}")
    print(f"net volume flux through the boundary, at most {leak:.2e} of the inflow "
          f"(an incompressible solve lets out what comes in)")
    if settled:
        print(f"shedding settled: {len(periods)} periods, T = {periods.mean():.4f} "
              f"(spread {periods.std() / periods.mean():.1%}), St = f*D_REF/U_REF = {St:.4f}")
    else:
        print(f"SHEDDING NOT SETTLED in the analysis window ({len(periods)} periods, lift "
              f"amplitude {amplitude:.3g}): no Strouhal number is reported -- run longer "
              f"(T_END) or start the analysis later (ANALYSIS_FROM)")
    print("These numbers come from ONE mesh and ONE time step. Before relying on them, run a "
          "finer mesh (generate_mesh's mesh_size) and a smaller step and see how much they move.")


record = _Record()

# ── PHYSICS, BOUNDARY CONDITIONS, TIME STEPPING AND SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
# Yours to write, from your problem: the function spaces, the boundary conditions, the weak form,
# the time stepping and the solve, on the mesh above.
# In scope: comm, domain, gdim, fdim, facet_tags, BOUNDARY, ds, and the SETTINGS. BOUNDARY maps
# each boundary name the mesh carries (left, right, bottom, top and OBSTACLE) to its tag;
# facet_tags holds each boundary facet with that tag, and ds(tag) integrates over one boundary.
# What the code after this region needs: record(t, u, p, nu) called after EVERY time step, up to
# T_END, with that step's velocity u and pressure p (Functions, or the two parts of a mixed
# Function; the same objects every step) and the kinematic viscosity nu your stress uses.
raise SystemExit("fill the marked region: the spaces, the boundary conditions, the weak form, "
                 "the time loop and the solve -- the comments above it say what the code after it needs")
# ── PHYSICS, BOUNDARY CONDITIONS, TIME STEPPING AND SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

record.finish()
"""
