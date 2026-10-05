"""scikit-fem linear elasticity generators and knowledge."""


def _elasticity_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Linear elasticity on rectangular domain, fixed left edge, body force."""
    nx = params.get("nx", 40)
    ny = params.get("ny", 4)
    lx = params.get("lx", 10.0)
    ly = params.get("ly", 1.0)
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    return f'''\
"""Linear elasticity: rectangular domain, fixed left — scikit-fem"""
from skfem import *
from skfem.models.elasticity import linear_elasticity, lame_parameters
import numpy as np
import json

_tol = 1e-10
m = (MeshQuad.init_tensor(np.linspace(0, {lx}, {nx+1}), np.linspace(0, {ly}, {ny+1}))
     .with_boundaries({{"left": lambda x: x[0] < _tol}}))
e = ElementVector(ElementQuad1())
ib = Basis(m, e)

lam, mu = lame_parameters({E}, {nu})
K = linear_elasticity(lam, mu).assemble(ib)

# Body force — set for your problem
@LinearForm
def body_force(v, w):
    return -1.0 * v[1]

f = body_force.assemble(ib)

# Fix left edge.  In scikit-fem >= 8 the boundary lookup requires the
# mesh to have been tagged via `with_boundaries({{...}})` (done above)
# so `get_dofs("left")` resolves to the tagged facets.  Without the
# tag the call raises because the mesh has no "left" facet group.
D = ib.get_dofs("left").flatten()
u = solve(*condense(K, f, D=D))

# Tip displacement.  `ElementVector(ElementQuad1())` stores DOFs
# *interleaved* (x0, y0, x1, y1, ...) — naively reshaping the flat
# solution as `(2, -1)` would scramble x and y across rows.  Recover
# the per-component arrays from the basis's `nodal_dofs` map, which
# records the global-DOF index of each (component, node) pair.
nodal_dofs = ib.nodal_dofs   # shape (n_components, n_nodes)
ux = u[nodal_dofs[0]]
uy = u[nodal_dofs[1]]
max_uy = uy.min()
print(f"Max tip displacement (uy.min): {{max_uy:.6f}}")

# Write a VTU so the sweep harness (and any other downstream consumer)
# can recover the displacement field, not just the scalar summary.
# `meshio` is a hard requirement here — the .vtu *is* the layer-3
# artifact of interest — so a missing import or write failure must
# surface as a non-zero exit instead of being silently swallowed.
import meshio
cells = [("quad", m.t.T)]
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])]) if m.p.shape[0] == 2 else m.p.T
# node ordering of `points` matches `m.p.T`, which is also what
# `nodal_dofs` is indexed by (one column per node), so the
# per-node (ux, uy) pairs line up with the corresponding row of
# `points` without any extra reordering.
displacement = np.column_stack([ux, uy])           # (n_nodes, 2)
displacement_3 = np.column_stack([displacement, np.zeros(displacement.shape[0])])
meshio.Mesh(points, cells, point_data={{"displacement": displacement_3}}).write("result.vtu")

summary = {{"max_displacement_y": float(max_uy), "n_dofs": int(K.shape[0])}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


KNOWLEDGE = {
    "linear_elasticity": {
        "description": "Linear elasticity: plane strain/stress, 3D (examples 02, 03, 11, 21)",
        "solver": "Direct sparse (scipy.sparse.linalg.spsolve)",
        "elements": "ElementVector(ElementQuad1()) or ElementVector(ElementTriP1())",
        "built_in_forms": "linear_elasticity, linear_stress (from skfem.models.elasticity)",
        "pitfalls": [
            "[Syntax] Vector elasticity uses ElementVector("
            "ElementQuad1()) (or ElementVector(ElementTriP1())) "
            "— wrap the scalar element in ElementVector. Using "
            "ElementQuad1() alone gives a SCALAR space; "
            "elasticity-form integrands such as inner(sigma(u), "
            "epsilon(v)) then fail dimensions. Signal: "
            "Basis(mesh, ElementVector(ElementQuad1())).Nbfun "
            "is exactly 2× Basis(mesh, ElementQuad1()).Nbfun "
            "(8 vs 4 in 2D); using the scalar basis in an "
            "elasticity assembly raises shape-mismatch from skfem "
            "or numpy. (Verified empirically 2026-06-01: "
            "vector_Nbfun=8, scalar_Nbfun=4 for ElementQuad1.)",
            "[API] skfem.models.elasticity.lame_parameters(E, nu) "
            "returns (lam, mu) computed from engineering "
            "constants via the standard formulas lam = "
            "E*nu/((1+nu)*(1-2*nu)) and mu = E/(2*(1+nu)). "
            "Signal: lame_parameters(E, nu) numerically equals "
            "the analytic (lam, mu) within float64 precision; "
            "math.isclose returns True on both. (Verified "
            "empirically 2026-06-01.)",
            "[API] skfem.models.elasticity.linear_elasticity("
            "lam, mu) returns a BilinearForm directly, ready "
            "for asm(). The result IS callable as a form (not "
            "an integrand); pass it to skfem.asm(form, basis) "
            "to assemble the stiffness matrix. Signal: "
            "type(linear_elasticity(lam, mu)) is "
            "skfem.assembly.form.BilinearForm; asm() returns "
            "a scipy.sparse matrix of shape (basis.N, basis.N). "
            "(Verified empirically 2026-08-30, skfem 12.0.1: the "
            "`is` check holds, and asm on ElementVector("
            "ElementQuad1()) over MeshQuad().refined(3) returns a "
            "sparse 162x162 matrix matching basis.N.)",
            "[Numerical] For eigenvalue (vibration) problems: "
            "use scipy.sparse.linalg.eigsh(K, M=M, k=n, sigma=0). "
            "sigma=0 (shift) targets the LOWEST eigenmodes; "
            "without sigma, eigsh defaults to highest. Signal: "
            "eigsh with sigma=0 returns the first n vibration "
            "modes (ordered by frequency); omitting sigma "
            "returns the LARGEST eigenvalues, which for a "
            "structural stiffness K are unrelated to physical "
            "vibration modes. Measured 2026-08-03 on skfem "
            "12.0.1 / scipy 1.15.3 with the scalar Dirichlet "
            "Laplacian on MeshTri().refined(4) (same generalised "
            "pencil structure): eigsh(K_I, M=M_I, k=5, sigma=0, "
            "which='LM') returns the low end of the spectrum, "
            "while eigsh(K_I, M=M_I, k=5) with NO sigma and no "
            "which returns values TWO TO THREE ORDERS OF "
            "MAGNITUDE larger — the top of the spectrum — and it does so "
            "silently, with no warning and no error. That is "
            "the whole failure: the call succeeds and the "
            "numbers are meaningless. "
            "eigsh(..., which='SM') without sigma recovers the "
            "same small values as sigma=0 but is much slower. "
            "(Verified empirically 2026-08-03.)",
            "[API] Boundary identification by name (e.g., "
            "basis.get_dofs('left') or basis.get_dofs("
            "facets='left')) requires that the mesh has had its "
            "boundaries TAGGED explicitly. A bare mesh created "
            "by MeshTri() / MeshQuad() carries no tags by "
            "default — querying an unknown boundary name raises "
            "ValueError 'Boundary \\'left\\' not found.' (catch "
            "via try/except, or use skfem.Mesh.with_boundaries "
            "to tag first). Note that get_dofs(elements='left') "
            "is a SUBDOMAIN lookup (Mesh.with_subdomains) and "
            "raises ValueError 'Subdomain \\'left\\' not found.', "
            "even on a mesh tagged with with_boundaries. "
            "Signal: a ValueError naming the missing tag. "
            "(Verified empirically — see boundary_not_tagged "
            "Tier-2 fixture.)",
        ],
    },
}

GENERATORS = {
    "linear_elasticity_2d": _elasticity_2d,
}
