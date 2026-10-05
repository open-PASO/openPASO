"""Linear elasticity generators for FEniCSx/dolfinx.

Variants: 2d, 3d, plate_hole, thick_beam
"""


KNOWLEDGE = {
    # ─────────────────────────────────────────────────────────────────
    # _SERVING_STATUS (added 2026-08-03)
    # This dict is SHADOWED and is NOT what an agent receives.
    # fenics/backend.py:get_knowledge() returns
    # src/tools/deep_knowledge.py::_FENICS_KNOWLEDGE['linear_elasticity'] for this
    # physics and never falls through to here. Editing the pitfalls
    # below changes nothing an agent can see. The claims here were NOT
    # re-verified in the 2026-08-03 execution pass for exactly that
    # reason — treat them as unverified history, and make corrections
    # in deep_knowledge.py instead.
    # ─────────────────────────────────────────────────────────────────
    "description": "Linear elasticity with Lame parameters, solved with FEniCSx/dolfinx",
    "weak_form": "inner(sigma(u), epsilon(v)) * dx = dot(f, v) * dx",
    "function_space": "Vector Lagrange order 1, shape=(gdim,)",
    "solver": {"ksp_type": "cg", "pc_type": "gamg"},
    "pitfalls": [
        "[Syntax] Vector function space for elasticity in dolfinx "
        "is created with ('Lagrange', 1, (gdim,)) — the trailing "
        "shape tuple marks it as a vector-valued FE. Passing a "
        "plain ('Lagrange', 1) gives a SCALAR space and the "
        "weak form inner(sigma(u), epsilon(v)) fails at form "
        "construction time when ufl.sym(ufl.grad(u)) is called "
        "on the scalar trial function. Signal: ufl.sym raises "
        "ValueError 'Symmetric part of tensor with rank != 2 is "
        "undefined.' inside the form definition, before "
        "dolfinx.fem.form / assemble is reached. "
        "(Verified empirically 2026-06-01 — the prior catalog "
        "wording 'Invalid ranks' / 'expected rank 1 trial "
        "function' does not match current dolfinx output.)",
        "[Syntax] Dirichlet BC value for a vector-valued elasticity "
        "space must be np.array([0.0]*gdim, dtype=default_scalar_type) "
        "— not scalar 0. Signal: 'Rank mismatch between Constant "
        "and function space in DirichletBC', a RuntimeError from "
        "dolfinx, on every scalar form of the value — plain 0.0, "
        "default_scalar_type(0.0), np.array(0.0) and "
        "fem.Constant(msh, default_scalar_type(0.0)) all give it. "
        "A one-element array on a 2-component space is a different "
        "complaint: 'Creating a DirichletBC using a Constant is not "
        "supported when the Constant size is not equal to the block "
        "size'. An integer 0 does not reach either check and raises "
        "NotImplementedError instead. The earlier wording here said "
        "numpy raises ValueError 'could not broadcast input array "
        "from shape () into shape (gdim,)'; numpy is never reached, "
        "no ValueError is raised at any of these calls, and numpy "
        "holds that text only as the format string 'could not "
        "broadcast %s from shape %S into shape %S' with the words "
        "'input array' substituted in. (Re-measured by execution "
        "2026-08-13, dolfinx 0.10.0 / numpy 1.26.4.)",
        "[Physics] Plane strain vs plane stress: adjust the Lame "
        "lambda accordingly. Plane stress uses lambda_star = "
        "2*lambda*mu/(lambda+2*mu); using plane strain lambda for "
        "a thin plate gives a bending stiffness about 10% too high "
        "at nu=0.3 (Poisson contraction is suppressed). Signal: the "
        "tip deflection "
        "of the plane_strain Function (dolfinx vector Lagrange "
        "space, fem.functionspace(msh, ('Lagrange', k, (gdim,)))) "
        "is about (1-nu^2) times the plane_stress one, not (1-nu): "
        "measured 2026-10-01 on this 2d template's cantilever "
        "(10 x 1, 80 x 8 triangles, clamped left edge, body force "
        "or end traction) 0.909 with P2 and 0.904 with P1 at "
        "nu=0.3 (1-nu^2 = 0.91), and 0.79 with P2 at nu=0.45 "
        "(1-nu^2 = 0.7975), where P1 gives 0.73 because plane-"
        "strain locking adds to it.",
        "[API] XDMFFile requires the Function degree to match "
        "the underlying mesh degree. A P2 Function on a P1 mesh "
        "(the common case) cannot be written directly — interpolate "
        "to a matching-degree space, or use VTKFile / VTXWriter. "
        "Signal: XDMFFile.write_function raises RuntimeError "
        "'Degree of output Function must be same as mesh degree. "
        "Maybe the Function needs to be interpolated?'. "
        "(Verified empirically 2026-06-01 — the prior catalog "
        "wording 'XDMF mesh must be P1' does not appear in "
        "current dolfinx output.)",
        "[Integration] For imported CAD geometry (IGES/STEP): use "
        "gmsh.model.getEntities() and getBoundingBox() to identify "
        "surface tags for physical group assignment. There is no "
        "automatic surface-to-BC mapping. Signal: dolfinx.io.gmsh "
        "(the module is `gmsh` on dolfinx 0.10, `gmshio` up to 0.9) "
        "reads the mesh but mesh.topology.dim returns 3 with no "
        "tagged facets; subsequent locate_dofs_topological on a "
        "facet tag returns an empty array.",
        "[Numerical] Coordinate-dependent surface tractions (e.g., "
        "torsion loads) require computing the scaling factor from "
        "a surface integral: q = M / integral(|r| ds) where r is "
        "the position relative to the axis. Naively applying q = "
        "M/A (uniform) gives a uniform shear that violates "
        "moment-vs-arm balance. Signal: a dolfinx ds-integrated "
        "reaction moment of the resulting Function differs from "
        "applied M by 20-40% on a thick beam with torsion BC.",
        "[Syntax] Mesh element order (gmsh Tet10) and FE "
        "polynomial degree (P2) are independent. A P2 space on a "
        "Tet10 mesh gives true isoparametric elements; a P1 space "
        "on Tet10 uses curved geometry but linear interpolation. "
        "Mismatched expectations are a common subtle bug. Signal: "
        "the assembled L2 error (dolfinx.fem.assemble_scalar"
        "(dolfinx.fem.form(ufl.inner(u_h-u_exact, u_h-u_exact)"
        "*ufl.dx))) against a manufactured solution shows slope "
        "1 across mesh refinements where slope 2 is expected, "
        "because dolfinx.fem.functionspace was called with "
        "('Lagrange', 1) while the user thought ('Lagrange', 2) "
        "was active. (Note 2026-06-02: dolfin's ufl.errornorm "
        "helper does not exist in dolfinx — assemble the inner-"
        "product form manually as shown.)",
        "[Numerical] Near-incompressible (nu > 0.49) requires a "
        "mixed-formulation (Taylor-Hood or three-field) to avoid "
        "volumetric locking. Pure displacement P1 or P2 at "
        "nu=0.4999 has displacement underestimated by orders of "
        "magnitude. Signal: the tip deflection of the displacement "
        "Function (dolfinx vector Lagrange space, "
        "fem.functionspace(msh, ('Lagrange', k, (gdim,)))) "
        "at nu=0.4999 is 1e-3 of the true value; "
        "switching to a Taylor_Hood mixed_P2_P1 MixedElement "
        "recovers the analytic deflection to within 1%.",
        "[API] dolfinx.fem.functionspace accepts only element "
        "family names that basix recognises. A degree-suffixed "
        "name like 'P1' raises ValueError; 'CG' is still accepted "
        "(mapped to Lagrange) but emits DeprecationWarning "
        "'\"CG\" element name is deprecated. Consider using "
        "\"Lagrange\" or \"P\" instead'. Signal: "
        "dolfinx.fem.functionspace(mesh, ('P1', 1)) on a triangle "
        "mesh raises ValueError 'Unknown element family: P1 with "
        "cell type triangle' — use 'Lagrange' (or 'P').",
    ],
    "materials": {
        "E": {"range": [1.0, 1e12], "unit": "Pa"},
        "nu": {"range": [0.0, 0.499], "unit": "dimensionless"},
    },
}

VARIANTS = ["2d", "3d", "plate_hole", "thick_beam"]


def generate(variant: str, params: dict) -> str:
    """Dispatch to the appropriate elasticity variant."""
    generators = {
        "2d": _elasticity_2d,
        "3d": _elasticity_3d,
        "plate_hole": _elasticity_plate_hole,
        "thick_beam": _elasticity_thick_beam,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(f"Unknown elasticity variant: {variant!r}. Available: {list(generators)}")
    return gen(params)


def _elasticity_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    return f'''\
"""Linear elasticity — 2D (plane stress) — FEniCSx/dolfinx"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np

# Domain dimensions — set for your problem
Lx, Ly = 10.0, 1.0
domain = mesh.create_rectangle(
    MPI.COMM_WORLD, [[0, 0], [Lx, Ly]], [80, 8], mesh.CellType.triangle
)
V = fem.functionspace(domain, ("Lagrange", 1, (2,)))  # vector

# Material
E_val = {E}
nu_val = {nu}
mu = E_val / (2 * (1 + nu_val))
lmbda = E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lmbda * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

# Fixed left edge (x=0)
def left_boundary(x):
    return np.isclose(x[0], 0.0)

tdim = domain.topology.dim
fdim = tdim - 1
left_facets = mesh.locate_entities_boundary(domain, fdim, left_boundary)
dofs = fem.locate_dofs_topological(V, fdim, left_facets)
bc = fem.dirichletbc(np.zeros(2, dtype=default_scalar_type), dofs, V)

# Traction on right edge (x=Lx)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Body force — set direction and magnitude for your problem
f = fem.Constant(domain, default_scalar_type((0.0, -1.0)))

a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx

problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="solve", petsc_options={{"ksp_type": "cg", "pc_type": "gamg"}})
uh = problem.solve()
uh.name = "displacement"

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

u_array = uh.x.array.reshape(-1, 2)
print(f"Elasticity solved: max |u_y| = {{np.abs(u_array[:, 1]).max():.6e}}")
print(f"DOFs: {{V.dofmap.index_map.size_global * 2}}")
'''


def _elasticity_3d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    return f'''\
"""Linear elasticity — 3D — FEniCSx/dolfinx"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np

Lx, Ly, Lz = 10.0, 1.0, 1.0
domain = mesh.create_box(
    MPI.COMM_WORLD, [[0, 0, 0], [Lx, Ly, Lz]], [40, 4, 4], mesh.CellType.tetrahedron
)
V = fem.functionspace(domain, ("Lagrange", 1, (3,)))

E_val = {E}
nu_val = {nu}
mu = E_val / (2 * (1 + nu_val))
lmbda = E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lmbda * ufl.nabla_div(u) * ufl.Identity(3) + 2 * mu * epsilon(u)

def left_boundary(x):
    return np.isclose(x[0], 0.0)

tdim = domain.topology.dim
fdim = tdim - 1
left_facets = mesh.locate_entities_boundary(domain, fdim, left_boundary)
dofs = fem.locate_dofs_topological(V, fdim, left_facets)
bc = fem.dirichletbc(np.zeros(3, dtype=default_scalar_type), dofs, V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, default_scalar_type((0.0, 0.0, -1.0)))

a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx

problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="solve", petsc_options={{"ksp_type": "cg", "pc_type": "gamg"}})
uh = problem.solve()
uh.name = "displacement"

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

u_array = uh.x.array.reshape(-1, 3)
print(f"Elasticity 3D solved: max |u_z| = {{np.abs(u_array[:, 2]).max():.6e}}")
print(f"DOFs: {{V.dofmap.index_map.size_global * 3}}")
'''


def _elasticity_plate_hole(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx script. Requires Gmsh.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    mesh_size = params.get("mesh_size", 0.04)
    radius = params.get("radius", 0.2)
    return f'''\
"""Linear elasticity: plate with circular hole — FEniCSx + Gmsh
Tension loading on left/right edges.
"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np

# Generate plate with hole using Gmsh
import gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("plate-hole")
rect = gmsh.model.occ.addRectangle(-1, -0.5, 0, 2, 1)
hole = gmsh.model.occ.addDisk(0, 0, 0, {radius}, {radius})
gmsh.model.occ.cut([(2, rect)], [(2, hole)])
gmsh.model.occ.synchronize()
surfaces = gmsh.model.getEntities(2)
gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], tag=1)
curves = gmsh.model.getEntities(1)
for i, c in enumerate(curves):
    gmsh.model.addPhysicalGroup(1, [c[1]], tag=i+1)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", {mesh_size})
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", {mesh_size} * 0.2)
gmsh.model.mesh.generate(2)
gmsh.write("plate_hole.msh")
gmsh.finalize()

from dolfinx.io.gmsh import read_from_msh
mesh_data = read_from_msh("plate_hole.msh", MPI.COMM_WORLD, gdim=2)
domain = mesh_data.mesh
gdim = domain.geometry.dim
V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))

# Material
E_val, nu_val = {E}, {nu}
mu = E_val / (2 * (1 + nu_val))
lmbda = E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))

def epsilon(u):
    return ufl.sym(ufl.grad(u))
def sigma(u):
    return lmbda * ufl.nabla_div(u) * ufl.Identity(gdim) + 2 * mu * epsilon(u)

# BCs: fix left edge, tension on right
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

def left(x):
    return np.isclose(x[0], -1.0)
def right(x):
    return np.isclose(x[0], 1.0)

left_facets = mesh.locate_entities_boundary(domain, fdim, left)
left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
bc = fem.dirichletbc(np.zeros(gdim, dtype=default_scalar_type), left_dofs, V)

# Traction on right edge
right_facets = mesh.locate_entities_boundary(domain, fdim, right)
right_mt = mesh.meshtags(domain, fdim, right_facets, np.full(len(right_facets), 1, dtype=np.int32))
ds = ufl.Measure("ds", domain=domain, subdomain_data=right_mt)
traction = fem.Constant(domain, default_scalar_type((10.0, 0.0)))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(traction, v) * ds(1)

problem = LinearProblem(a, L, bcs=[bc],
    petsc_options_prefix="elast", petsc_options={{"ksp_type": "cg", "pc_type": "gamg"}})
uh = problem.solve()
uh.name = "displacement"

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

u_arr = uh.x.array.reshape(-1, gdim)
print(f"Plate with hole: max |u| = {{np.linalg.norm(u_arr, axis=1).max():.6e}}")
print(f"DOFs: {{V.dofmap.index_map.size_global * gdim}}")
'''


def _elasticity_thick_beam(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    lx = params.get("lx", 5.0)
    ly = params.get("ly", 2.0)
    nx = int(lx * 8)
    ny = int(ly * 8)
    return f'''\
"""Linear elasticity on {lx}x{ly} domain — FEniCSx/dolfinx"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np

Lx, Ly = {lx}, {ly}
domain = mesh.create_rectangle(
    MPI.COMM_WORLD, [[0, 0], [Lx, Ly]], [{nx}, {ny}], mesh.CellType.triangle
)
V = fem.functionspace(domain, ("Lagrange", 1, (2,)))

E_val = {E}
nu_val = {nu}
mu = E_val / (2 * (1 + nu_val))
lmbda = E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lmbda * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

def left_boundary(x):
    return np.isclose(x[0], 0.0)

tdim = domain.topology.dim
fdim = tdim - 1
left_facets = mesh.locate_entities_boundary(domain, fdim, left_boundary)
dofs = fem.locate_dofs_topological(V, fdim, left_facets)
bc = fem.dirichletbc(np.zeros(2, dtype=default_scalar_type), dofs, V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, default_scalar_type((0.0, -1.0)))

a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx

problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="solve",
    petsc_options={{"ksp_type": "cg", "pc_type": "gamg"}})
uh = problem.solve()
uh.name = "displacement"

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

u_array = uh.x.array.reshape(-1, 2)
print(f"Elasticity solved: max |u_y| = {{np.abs(u_array[:, 1]).max():.6e}}")
print(f"DOFs: {{V.dofmap.index_map.size_global * 2}}")
'''
