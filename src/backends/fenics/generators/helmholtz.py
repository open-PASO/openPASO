"""Helmholtz equation generator for FEniCSx/dolfinx.

Variants: 2d
"""


KNOWLEDGE = {
    # ─────────────────────────────────────────────────────────────────
    # _SERVING_STATUS (added 2026-08-03)
    # This dict is SHADOWED and is NOT what an agent receives.
    # fenics/backend.py:get_knowledge() returns
    # src/tools/deep_knowledge.py::_FENICS_KNOWLEDGE['helmholtz'] for this
    # physics and never falls through to here. Editing the pitfalls
    # below changes nothing an agent can see. The claims here were NOT
    # re-verified in the 2026-08-03 execution pass for exactly that
    # reason — treat them as unverified history, and make corrections
    # in deep_knowledge.py instead.
    # ─────────────────────────────────────────────────────────────────
    "description": "Helmholtz: -Δu - k²u = f. Indefinite system. Use GMRES or direct solver — NOT CG.",
    "weak_form": "inner(grad(u), grad(v))*dx - k**2 * inner(u, v)*dx = inner(f, v)*dx",
    "function_space": (
        "Lagrange P2+ (need ~10 points per wavelength). For "
        "complex-valued problems use scalar_type=np.complex128 "
        "AND a PETSc build with --with-scalar-type=complex."
    ),
    "solver": {
        "real": "Direct (MUMPS) or GMRES + LU preconditioner",
        "complex": "Same — complex PETSc build required",
    },
    "pitfalls": [
        "[Numerical] System is INDEFINITE (once k^2 exceeds the lowest "
        "Laplacian eigenvalue), so CG fails. Use GMRES or direct. "
        "Signal: PETSc KSP 'cg' stops after a few iterations, usually with a "
        "negative getConvergedReason(): -10 DIVERGED_INDEFINITE_MAT, or -8 "
        "DIVERGED_INDEFINITE_PC with some preconditioners (seen with gamg). "
        "LinearProblem.solve() does not raise unless "
        "ksp_error_if_not_converged is set, and the returned field is "
        "unconverged. With pc_type 'hypre' CG can instead report 2 "
        "(CONVERGED_RTOL) after 1-2 iterations while the true residual "
        "||b - A x|| / ||b|| stays near 1, and ksp_error_if_not_converged "
        "does not fire then: check the true residual, not only the reason.",
        "[Numerical] Resolution rule: ~10 DOFs per wavelength minimum. "
        "Pollution effect grows with k — high-k problems need 20+ DOFs/wavelength. "
        "Signal: the dolfinx Function solution amplitude in the XDMFFile output "
        "shrinks vs analytic plane-wave by factor (1 - C*k*h^2) for "
        "under-resolved meshes.",
        "[Syntax] Complex-valued mode requires dolfinx.default_scalar_type "
        "to be np.complex128, which is a property of the PETSc BUILD and "
        "cannot be switched at run time — it needs a separate env built "
        "with --with-scalar-type=complex. Signal: putting an imaginary "
        "unit in a form under a real build (the impedance term "
        "1j*k*inner(u, v)*ds) raises, at fem.form time, "
        "'ValueError: Unexpected complex value in real expression.' "
        "from ufl/algorithms/remove_complex_nodes.py. The previously "
        "quoted signal 'ScalarType is not complex' is emitted by "
        "nothing in dolfinx, ufl, basix, ffcx, petsc4py or PETSc and "
        "does not reproduce. The SAME form compiles without complaint "
        "in a complex build. Check the build before the form: assert "
        "np.issubdtype(dolfinx.default_scalar_type, np.complexfloating). "
        "(Verified by execution 2026-08-07 in BOTH conda envs — real "
        "`fenics` raises, complex `fenicsc` compiles; dolfinx 0.10.0.)",
    ],
}

VARIANTS = ["2d"]


def generate(variant: str, params: dict) -> str:
    """Dispatch to the appropriate Helmholtz variant."""
    generators = {
        "2d": _helmholtz_2d,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(
            f"Unknown Helmholtz variant: {variant!r}. "
            f"Available: {list(generators)}")
    return gen(params)


def _helmholtz_2d(params: dict) -> str:
    """FORMAT TEMPLATE — Helmholtz in a unit square with
    homogeneous Dirichlet BC and a Gaussian source. Real-valued
    (no PML, no absorbing BC) — for a complex-valued absorbing-
    BC variant, switch scalar_type=np.complex128 and add an
    impedance term."""
    nx = params.get("nx", 64)
    k_val = params.get("k", 6.0)
    return f'''\
"""Helmholtz: -Δu - k²u = f — FEniCSx/dolfinx (real-valued, Dirichlet)"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import basix.ufl
import ufl
import numpy as np

domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {nx},
                                 mesh.CellType.triangle)
V = fem.functionspace(domain,
                      basix.ufl.element("Lagrange",
                                         domain.basix_cell(), 2))

k = fem.Constant(domain, default_scalar_type({k_val}))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(domain)
f = ufl.exp(-50.0 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
a = (ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
     - k * k * ufl.inner(u, v) * ufl.dx)
L = ufl.inner(f, v) * ufl.dx

# Homogeneous Dirichlet on all boundaries
def boundary(x):
    return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
            | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
dofs = fem.locate_dofs_geometrical(V, boundary)
bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)

# Helmholtz is INDEFINITE — direct solver (MUMPS) or GMRES
problem = LinearProblem(
    a, L, bcs=[bc],
    petsc_options_prefix="helmholtz_",
    petsc_options={{"ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps"}})
uh = problem.solve()
# --- Write the result where a reader can open it ------------------------------
# This template printed a norm and wrote nothing, so a model that ran it had
# no field to look at, plot or verify. XDMFFile writes P1 functions; the
# openPASO backend converts result.xdmf/.h5 into a standard result.vtu after
# the run (the same route the Poisson template uses). Higher-order or
# non-Lagrange functions are interpolated into P1 for output first.
import json
from dolfinx.io import XDMFFile
_Vout = fem.functionspace(domain, basix.ufl.element("Lagrange", domain.basix_cell(), 1))
_uo = fem.Function(_Vout, name="u"); _uo.interpolate(uh)
with XDMFFile(domain.comm, "result.xdmf", "w") as _x:
    _x.write_mesh(domain); _x.write_function(_uo)
_l2 = float(np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(uh, uh) * ufl.dx)))))
with open("results_summary.json", "w") as _f:
    json.dump({{"l2_norm_u": _l2, "n_dofs": V.dofmap.index_map.size_global}}, _f, indent=2)
print(f"||u||_L2 = {{np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(uh, uh) * ufl.dx))))}}")
'''
