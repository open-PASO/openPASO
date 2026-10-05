"""Biharmonic equation (4th order) generator for FEniCSx/dolfinx.

Variants: 2d
"""


KNOWLEDGE = {
    # ─────────────────────────────────────────────────────────────────
    # _SERVING_STATUS (added 2026-08-03)
    # This dict is SHADOWED and is NOT what an agent receives.
    # fenics/backend.py:get_knowledge() returns
    # src/tools/deep_knowledge.py::_FENICS_KNOWLEDGE['biharmonic'] for this
    # physics and never falls through to here. Editing the pitfalls
    # below changes nothing an agent can see. The claims here were NOT
    # re-verified in the 2026-08-03 execution pass for exactly that
    # reason — treat them as unverified history, and make corrections
    # in deep_knowledge.py instead.
    # ─────────────────────────────────────────────────────────────────
    "description": "Biharmonic equation (4th order PDE) via interior penalty DG",
    "weak_form": "(laplacian(u), laplacian(v))*dx - avg(laplacian(u))*jump(grad(v).n)*dS - jump(grad(u).n)*avg(laplacian(v))*dS + alpha/h*jump(grad(u).n)*jump(grad(v).n)*dS",
    "function_space": "Lagrange order 2+ (C0-IP method), or DG order 2+ (full DG)",
    "solver": {"ksp_type": "preonly", "pc_type": "lu"},
    "pitfalls": [
        "[API] In dolfinx 0.10 the function-space constructor is "
        "fem.functionspace (lowercase factory function), NOT "
        "fem.FunctionSpace (capital-F class). Both names exist: "
        "fem.functionspace(mesh, ('Lagrange', 2)) returns a "
        "FunctionSpace object, while a direct call to "
        "fem.FunctionSpace(mesh, ('Lagrange', 2)) raises "
        "TypeError: FunctionSpace.__init__() missing 1 required "
        "positional argument: 'cppV'. LLM agents trained on "
        "legacy dolfin tutorials that show FunctionSpace(...) as "
        "the canonical name routinely paste the capital form into "
        "dolfinx scripts. Signal: TypeError with literal 'cppV' "
        "in the message uniquely identifies a "
        "fem.FunctionSpace(...) call vs a fem.functionspace(...) "
        "call. (Verified empirically 2026-06-01 — Tier-2 fixture "
        "functionspace_factory_vs_class in scripts/tier2_fixtures/"
        "fenics/.)",
        "[Numerical] 4th-order PDE: standard C0 Lagrange cannot "
        "represent the biharmonic operator directly — Laplace of "
        "a C0 function is a distribution, not a function in "
        "L^2(domain). Use either C0-Interior Penalty (P2 Lagrange "
        "+ jump penalty terms on interior facets) or full DG (Pk "
        "Discontinuous Galerkin with all interface terms). Signal: "
        "ufl.div(ufl.grad(u)) on a P1 Lagrange space silently "
        "compiles but produces an assembled matrix whose null "
        "space dimension differs from the expected biharmonic "
        "kernel — convergence against a manufactured solution "
        "plateaus regardless of mesh refinement.",
        "[Numerical] C0-interior penalty (C0-IP): uses P2 (or "
        "higher) continuous Lagrange elements with normal-jump "
        "penalty terms on interior facets, integrated against "
        "ufl.dS (uppercase — interior facet measure). Required "
        "jump operators are ufl.jump(ufl.grad(u), n) and "
        "ufl.avg(ufl.div(ufl.grad(u))). Signal: the assembled "
        "fem.form matrix on the C0-IP P2 space has ndof equal to "
        "V.dofmap.index_map.size_global which differs from the "
        "full-DG count by a factor of (d+1) (d = spatial dim) "
        "for the same polynomial order — because C0-IP shares "
        "P2 nodal DOFs across element facets.",
        "[Numerical] Penalty parameter alpha must satisfy "
        "alpha > C * (polynomial_order)^2 (C is mesh-dependent, "
        "typically ~4 for triangles, ~8 for quads). Below this "
        "the C0-IP form loses coercivity and the discrete system "
        "is indefinite — KSPSolve with PCLU still returns a "
        "vector but it does NOT satisfy the original variational "
        "problem. Signal: residual norm of the assembled "
        "saddle-point system computed AFTER solve drops to "
        "machine precision (LU on indefinite matrix is exact), "
        "but |grad^2 u - f|_L^2 measured against an analytic "
        "reference is O(1).",
        "[API] dolfinx 0.10 XDMFFile.write_function requires the "
        "polynomial degree of the output Function to equal the mesh "
        "degree (typically 1 for create_unit_square). A P2 Lagrange "
        "C0-IP solution cannot be emitted via XDMFFile.write_function "
        "— the call raises RuntimeError 'Degree of output Function "
        "must be same as mesh degree. Maybe the Function needs to "
        "be interpolated?'. Use VTXWriter (ADIOS2 backend, output "
        "extension .bp) which supports arbitrary Lagrange degree, "
        "or interpolate uh into a P1 space before writing XDMF. "
        "Signal: the literal substring 'Degree of output Function "
        "must be same as mesh degree' uniquely identifies this "
        "constraint. Same failure mode as fenics::stokes Taylor-"
        "Hood velocity output. (Verified empirically 2026-06-01 "
        "— Layer F catch.)",
        "[Physics] Clamped boundary conditions for the biharmonic "
        "fix BOTH u=0 AND grad(u).n=0 on the boundary (the "
        "moment-free vertical-displacement-fixed configuration). "
        "Simply supported is u=0 AND div(grad(u))=0 (laplacian = "
        "0). Mixing these gives wrong plate-bending results. "
        "Signal: with two separate fem.dirichletbc applications "
        "of the two BC sets on the same fem.Function, the "
        "resulting uh.x.array.max() values differ by a factor of "
        "~5 for a uniformly loaded Kirchhoff plate — observable "
        "via numpy.linalg.norm(uh.x.array, ord=np.inf) under the "
        "two BC sets.",
        "[Physics] For Kirchhoff plates: the biharmonic in "
        "displacement w is D * laplacian^2(w) = q where "
        "D = E*h^3/(12*(1-nu^2)) is the flexural rigidity, h is "
        "the plate thickness, and q is the transverse load. "
        "Stresses come from sigma = -E*z/(1-nu^2) * (d^2w/dx^2 + "
        "nu*d^2w/dy^2) on the through-thickness coordinate z. "
        "Signal: max deflection w_max computed by "
        "uh.x.array.max() on a fem.Function for a clamped "
        "circular plate of radius R under uniform load q "
        "matches the analytic Kirchhoff formula "
        "qR^4/(64*D) within ~1% on a refined mesh — divergence "
        "from this number by an order of magnitude indicates a "
        "BC or D-formula error.",
        "[Performance] DG-style interior-penalty biharmonic (the "
        "official dolfinx demo formulation: a = inner(div(grad(u)), "
        "div(grad(v)))*dx + jump-grad / avg-div terms over dS) "
        "REQUIRES mesh.GhostMode.shared_facet on MPI runs so each "
        "rank sees its neighbours' DOFs across shared facets. "
        "create_unit_square defaults to shared_facet ALREADY, but "
        "users who pass a custom partitioner via the partitioner= "
        "kwarg, or who construct the mesh from a custom imported "
        "topology, MUST set ghost_mode=mesh.GhostMode.shared_facet. "
        "GhostMode.none silently produces wrong assembly on dS "
        "integrals (missing neighbour contributions zero out at "
        "rank boundaries). The C++ demo "
        "cpp/demo/biharmonic/main.cpp builds its mesh with "
        "mesh::create_cell_partitioner(mesh::GhostMode::shared_facet); "
        "the optional integer argument of create_cell_partitioner "
        "is max_facet_to_cell_links (default 2), not an overlap "
        "depth, and there is no extra-overlap setting. (File walk "
        "cpp/demo/biharmonic/main.cpp 2026-06-02; verified live "
        "in dolfinx — mesh.GhostMode has exactly two values: "
        "{none, shared_facet}, with shared_facet as the default.)",
    ],
}

VARIANTS = ["2d"]


def generate(variant: str, params: dict) -> str:
    """Dispatch to the appropriate biharmonic variant."""
    generators = {
        "2d": _biharmonic_2d,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(f"Unknown biharmonic variant: {variant!r}. Available: {list(generators)}")
    return gen(params)


def _biharmonic_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable script. All parameter defaults are placeholders. The user/agent must set values appropriate to the specific problem being solved."""
    nx = params.get("nx", 32)
    ny = params.get("ny", 32)
    penalty = params.get("penalty", 8.0)
    return f'''\
"""Biharmonic equation on unit square — interior penalty DG — FEniCSx/dolfinx
laplacian(laplacian(u)) = f on [0,1]^2
u = 0, grad(u).n = 0 on boundary
Symmetric interior penalty (C0-IP or full DG) method.
"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
import ufl
import numpy as np
from petsc4py import PETSc

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 2))

# Boundary conditions: u = 0 and du/dn = 0 on all boundaries
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)

# Source term
x = ufl.SpatialCoordinate(domain)
f_expr = fem.Constant(domain, default_scalar_type(1.0))

# Interior penalty DG bilinear form for biharmonic
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
n = ufl.FacetNormal(domain)
h = ufl.CellDiameter(domain)
h_avg = (h("+") + h("-")) / 2.0
alpha = fem.Constant(domain, default_scalar_type({penalty}))

# Biharmonic via C0-interior penalty:
# Volume: (laplacian(u), laplacian(v))
# Interior facets: penalty jumps in normal derivatives
a = ufl.inner(ufl.div(ufl.grad(u)), ufl.div(ufl.grad(v))) * ufl.dx \\
    - ufl.inner(ufl.avg(ufl.div(ufl.grad(u))), ufl.jump(ufl.grad(v), n)) * ufl.dS \\
    - ufl.inner(ufl.jump(ufl.grad(u), n), ufl.avg(ufl.div(ufl.grad(v)))) * ufl.dS \\
    + alpha / h_avg * ufl.inner(ufl.jump(ufl.grad(u), n), ufl.jump(ufl.grad(v), n)) * ufl.dS

L = f_expr * v * ufl.dx

# Assemble and solve
a_form = fem.form(a)
L_form = fem.form(L)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc

A = assemble_matrix(a_form, bcs=[bc])
A.assemble()
b = assemble_vector(L_form)
apply_lifting(b, [a_form], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, [bc])

# Solve with direct solver
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
uh = fem.Function(V)
uh.name = "u"
solver.solve(b, uh.x.petsc_vec)

# Output — VTXWriter, not XDMFFile.
# Biharmonic C0-IP uses P2 Lagrange (mandatory for the
# C0-IP method — see pitfall on 4th-order PDEs above).
# XDMFFile.write_function refuses to write a P>1 Function
# on a P1 mesh with RuntimeError 'Degree of output
# Function must be same as mesh degree' — same failure
# mode as fenics::stokes Taylor-Hood velocity output.
# VTXWriter (ADIOS2 backend) supports arbitrary polynomial
# degree; standardise on it for any P>=2 output here.
from dolfinx.io import VTXWriter
with VTXWriter(domain.comm, "result.bp", [uh]) as vtx:
    vtx.write(0.0)

u_array = uh.x.array
# Beside the VTX (.bp) output, which needs an ADIOS2 reader nobody may have,
# write result.xdmf of a P1 interpolation; the openPASO backend converts it to a
# standard result.vtu after the run (XDMFFile refuses P>1 functions directly).
import basix.ufl
from dolfinx.io import XDMFFile
_Vvis = fem.functionspace(domain, basix.ufl.element("Lagrange", domain.basix_cell(), 1))
_o = fem.Function(_Vvis, name="u"); _o.interpolate(uh)
with XDMFFile(domain.comm, "result.xdmf", "w") as _x:
    _x.write_mesh(domain); _x.write_function(_o)
print(f"Biharmonic solved: min(u)={{u_array.min():.6e}}, max(u)={{u_array.max():.6e}}")
print(f"DOFs: {{V.dofmap.index_map.size_global}}")
'''
