"""Stokes flow generator for FEniCSx/dolfinx.

Variants: 2d
"""


KNOWLEDGE = {
    # ─────────────────────────────────────────────────────────────────
    # _SERVING_STATUS (added 2026-08-03)
    # This dict is SHADOWED and is NOT what an agent receives.
    # fenics/backend.py:get_knowledge() returns
    # src/tools/deep_knowledge.py::_FENICS_KNOWLEDGE['stokes'] for this
    # physics and never falls through to here. Editing the pitfalls
    # below changes nothing an agent can see. The claims here were NOT
    # re-verified in the 2026-08-03 execution pass for exactly that
    # reason — treat them as unverified history, and make corrections
    # in deep_knowledge.py instead.
    # ─────────────────────────────────────────────────────────────────
    "description": "Stokes flow with Taylor-Hood P2/P1 or MINI element",
    "weak_form": "nu*(grad(u),grad(v))*dx + div(v)*p*dx + div(u)*q*dx = (f,v)*dx",
    "function_space": (
        "Mixed (basix.ufl 0.10+): "
        "P2 = basix.ufl.element('Lagrange', domain.basix_cell(), 2, shape=(domain.geometry.dim,)); "
        "P1 = basix.ufl.element('Lagrange', domain.basix_cell(), 1); "
        "TH = basix.ufl.mixed_element([P2, P1]); W = fem.functionspace(domain, TH)"
    ),
    "solver": {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    "pitfalls": [
        "[Numerical] Saddle-point Stokes system is INDEFINITE — "
        "use a direct solver (LinearProblem petsc_options "
        "{'ksp_type': 'preonly', 'pc_type': 'lu', "
        "'pc_factor_mat_solver_type': 'mumps'}; dolfinx has no "
        "PETScKrylovSolver) "
        "or a block preconditioner (Schur complement), NEVER CG. "
        "Signal: LinearProblem with 'ksp_type': 'cg' on a "
        "Taylor-Hood Stokes problem returns without raising, with "
        "problem.solver.getConvergedReason() = -11 "
        "(DIVERGED_PCSETUP_FAILED, default ILU) or -10 "
        "(DIVERGED_INDEFINITE_MAT, pc_type jacobi/none). (Audit "
        "2026-06-02.)",
        "[API] Mixed-element construction in dolfinx 0.10+ uses "
        "basix.ufl, NOT ufl: P2 = basix.ufl.element('Lagrange', "
        "domain.basix_cell(), 2, shape=(domain.geometry.dim,)); "
        "P1 = basix.ufl.element('Lagrange', domain.basix_cell(), 1); "
        "TH = basix.ufl.mixed_element([P2, P1]). The pre-2024 "
        "form 'ufl.VectorElement / ufl.FiniteElement / "
        "ufl.MixedElement' was REMOVED — those names no longer "
        "exist as attributes of the ufl module (verified against "
        "ufl 2025.2.1 / dolfinx 0.10.0, 2026-06-01). Signal: "
        "running a stokes script using ufl.VectorElement raises "
        "AttributeError: module 'ufl' has no attribute "
        "'VectorElement'.",
        "[API] Build the FunctionSpace from the mixed_element via "
        "W = fem.functionspace(domain, TH) where TH = "
        "basix.ufl.mixed_element([P2, P1]). Passing the bare element "
        "tuple as kwarg is wrong. Signal: fem.functionspace(domain, "
        "(P2, P1)) raises \"AttributeError: 'tuple' object has no "
        "attribute 'cell'\" — dolfinx reads the second argument as ONE "
        "element and asks it for .cell. A list gives the same message "
        "with 'list'. The previously quoted \"TypeError 'expected "
        "basix.ufl element or tuple (family, degree)'\" is emitted "
        "nowhere in dolfinx, ufl or basix and does not reproduce; note "
        "the exception CLASS was wrong too, so an `except TypeError` "
        "guard built on it would not even catch. The plain "
        "(family, degree) tuple — fem.functionspace(domain, "
        "('Lagrange', 2)) — is still valid; it is a tuple of ELEMENTS "
        "that fails. (Verified by execution 2026-08-07, dolfinx "
        "0.10.0 / basix 0.10.0.)",
        "[API] Non-homogeneous Dirichlet BCs on the velocity sub-"
        "space: collapse it (V0, _ = W.sub(0).collapse()), create a "
        "Function on the collapsed space, interpolate the desired "
        "profile, then call fem.dirichletbc. Signal: passing a scalar "
        "value to fem.dirichletbc on the vector sub-space W.sub(0) raises "
        "'RuntimeError: Rank mismatch between Constant and function "
        "space in DirichletBC' — the check is on RANK (a rank-0 "
        "Constant against a rank-1 space), which is why the same "
        "message comes back for a plain vector space as for a mixed "
        "sub-space. The previously quoted 'Value shape must match "
        "function space' is in no dolfinx binary and does not "
        "reproduce. The fix is W0, dofs = W.sub(0).collapse() + "
        "interpolate. (Verified by execution 2026-08-07, dolfinx "
        "0.10.0.)",
        "[Numerical] Pressure is determined only up to a constant "
        "for enclosed flows — pin one DOF via fem.dirichletbc on a "
        "single pressure node, or ensure an outflow boundary takes "
        "the role of pressure datum. Signal: a closed-cavity Stokes "
        "solve without pressure pin reports KSPSolve "
        "DIVERGED_BREAKDOWN with near-zero pivot, or the resulting "
        "pressure Function in the XDMFFile has a huge additive "
        "offset (drifts O(1e6) between runs). (Audit 2026-06-02.)",
        "[Numerical] MINI element (P1+bubble for velocity, P1 for "
        "pressure) is an inf-sup-stable alternative to Taylor-Hood, "
        "simpler implementation but less accurate. Signal: an MMS "
        "convergence study with MINI shows L2-error of the velocity "
        "Function in the XDMFFile output at rate ~h^2 instead of "
        "Taylor-Hood's ~h^3 for the same mesh refinement. (Audit "
        "2026-06-02.)",
        "[Output] In dolfinx 0.10+, XDMFFile.write_function() "
        "REJECTS Functions whose element degree exceeds the mesh "
        "geometry degree (which is 1 for the default-order triangle "
        "/ tetrahedron mesh from create_unit_square / create_box). "
        "Writing a P2 velocity directly raises 'RuntimeError: "
        "Degree of output Function must be same as mesh degree. "
        "Maybe the Function needs to be interpolated?'. This blocks "
        "the obvious Taylor-Hood output pattern: collapse(W.sub(0)) "
        "and write the P2 sub-function. Fix: import dolfinx.io."
        "VTXWriter (ADIOS2 backend) which supports arbitrary-degree "
        "Functions natively, writes to a .bp directory ParaView "
        "5.10+ reads directly. Usage: with VTXWriter(domain.comm, "
        "'velocity.bp', [u_h]) as vtx: vtx.write(0.0). The output "
        "is a directory, not a file; the '.bp' suffix is the "
        "convention used by the upstream demos, but dolfinx does "
        "not require it. "
        "Alternative if you must use XDMF: interpolate the P2 "
        "velocity to a P1 space first via fem.functionspace(domain, "
        "('Lagrange', 1, (gdim,))) + Function.interpolate(u_h), "
        "which loses the P2 fidelity but keeps a single .xdmf "
        "output. (Audit 2026-06-03, follow-up to Layer F fix "
        "task #114 that updated the template but not the pitfall "
        "narrative; gap surfaced by Layer G audit.)",
    ],
}

VARIANTS = ["2d"]


def generate(variant: str, params: dict) -> str:
    """Dispatch to the appropriate Stokes variant."""
    generators = {
        "2d": _stokes_2d,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(f"Unknown Stokes variant: {variant!r}. Available: {list(generators)}")
    return gen(params)


def _stokes_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    nx = params.get("nx", 32)
    ny = params.get("ny", nx)
    return f'''\
"""Stokes flow — Taylor-Hood P2/P1 — FEniCSx/dolfinx"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import basix.ufl
import ufl
import numpy as np

domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
gdim = domain.geometry.dim
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

# Taylor-Hood: P2 velocity + P1 pressure (basix.ufl, dolfinx 0.10+)
P2 = basix.ufl.element("Lagrange", domain.basix_cell(), 2, shape=(gdim,))
P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
TH = basix.ufl.mixed_element([P2, P1])
W = fem.functionspace(domain, TH)

# BCs: set velocity on boundaries (adjust for your problem)
def walls(x):
    return np.isclose(x[0], 0) | np.isclose(x[0], 1) | np.isclose(x[1], 0)
def lid(x):
    return np.isclose(x[1], 1)

W0 = W.sub(0)
wall_facets = mesh.locate_entities_boundary(domain, fdim, walls)
lid_facets = mesh.locate_entities_boundary(domain, fdim, lid)
V0, _ = W0.collapse()
noslip = fem.Function(V0)
noslip.x.array[:] = 0
bc_noslip = fem.dirichletbc(noslip, fem.locate_dofs_topological((W0, V0), fdim, wall_facets), W0)
lid_vel = fem.Function(V0)
lid_vel.interpolate(lambda x: (np.ones(x.shape[1]), np.zeros(x.shape[1])))
bc_lid = fem.dirichletbc(lid_vel, fem.locate_dofs_topological((W0, V0), fdim, lid_facets), W0)

(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.div(u)*q*ufl.dx + ufl.div(v)*p*ufl.dx
L = ufl.inner(fem.Constant(domain, default_scalar_type((0, 0))), v) * ufl.dx

problem = LinearProblem(a, L, bcs=[bc_noslip, bc_lid],
    petsc_options_prefix="stokes",
    petsc_options={{"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}})
wh = problem.solve()

u_h = wh.sub(0).collapse()
p_h = wh.sub(1).collapse()
u_h.name = "velocity"
p_h.name = "pressure"

# IMPORTANT: in dolfinx 0.10 XDMFFile.write_function REQUIRES
# the output Function's element degree to match the mesh
# degree (which is 1 for the default triangle mesh). Writing
# a P2 velocity directly raises:
#   RuntimeError: Degree of output Function must be same as
#   mesh degree. Maybe the Function needs to be interpolated?
# VTXWriter (ADIOS2 backend) supports arbitrary-degree
# Functions natively. Use it for the Taylor-Hood mixed
# solution.
from dolfinx.io import VTXWriter
with VTXWriter(domain.comm, "velocity.bp", [u_h]) as vtx:
    vtx.write(0.0)
with VTXWriter(domain.comm, "pressure.bp", [p_h]) as vtx:
    vtx.write(0.0)

# Beside the VTX (.bp) output, which needs an ADIOS2 reader nobody may have,
# write result.xdmf of a P1 interpolation; the openPASO backend converts it to a
# standard result.vtu after the run (XDMFFile refuses P>1 functions directly).
import basix.ufl
from dolfinx.io import XDMFFile
_Vvis = fem.functionspace(domain, basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,)))
_o = fem.Function(_Vvis, name="velocity"); _o.interpolate(u_h)
with XDMFFile(domain.comm, "result.xdmf", "w") as _x:
    _x.write_mesh(domain); _x.write_function(_o)
print(f"Stokes: DOFs={{W.dofmap.index_map.size_global}}")
print("Stokes solve complete.")
'''
