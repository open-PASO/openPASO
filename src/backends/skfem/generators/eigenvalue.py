"""scikit-fem eigenvalue problem generators and knowledge."""


def _eigenvalue_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Eigenvalue problem: Laplace on unit square."""
    nx = params.get("nx", 32)
    n_eigs = params.get("n_eigenvalues", 5)
    return f'''\
"""Eigenvalue problem: Laplace eigenvalues — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace, mass
import numpy as np
from scipy.sparse.linalg import eigsh
import json, math

m = MeshQuad.init_tensor(np.linspace(0, 1, {nx+1}), np.linspace(0, 1, {nx+1}))
e = ElementQuad1()
ib = Basis(m, e)

K = laplace.assemble(ib)
M = mass.assemble(ib)

D = ib.get_dofs().flatten()
I = ib.complement_dofs(D)

# Solve generalized eigenvalue: K*x = lambda*M*x (restrict to interior DOFs)
eigenvalues, eigenvectors = eigsh(K[I][:, I], k={n_eigs}, M=M[I][:, I], sigma=0, which='LM')

# Exact eigenvalues: pi^2*(m^2+n^2)
exact = sorted([math.pi**2*(i**2+j**2) for i in range(1,6) for j in range(1,6)])[:{n_eigs}]

print(f"Computed eigenvalues: {{eigenvalues}}")
print(f"Exact eigenvalues:    {{exact}}")
for i, (c, e_val) in enumerate(zip(eigenvalues, exact)):
    err = abs(c - e_val) / e_val
    print(f"  lambda_{{i+1}} = {{c:.6f}} (exact: {{e_val:.6f}}, err: {{err:.2e}})")

summary = {{"eigenvalues": eigenvalues.tolist(), "exact": exact, "n_dofs": K.shape[0]}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)

# Write the first eigenmode so the run has attestable output — without a
# result file the verification gate correctly reports completed_unverified
# ("produced NO output files") and the physics can never be VERIFIED
# (Mac stress audit 2026-07-18).
mode = np.zeros(K.shape[0])
mode[I] = eigenvectors[:, 0]
import meshio
pts = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
meshio.write_points_cells("result.vtu", pts, [("quad", m.t.T)],
                          point_data={{"eigenmode_1": mode}})
print("Eigenvalue solve complete.")
'''


KNOWLEDGE = {
    "eigenvalue": {
        "description": "Eigenvalue problems — Laplace, elasticity vibration (examples 02, 03, 16, 21)",
        "solver": "scipy.sparse.linalg.eigsh (Lanczos for symmetric generalized eigenvalue)",
        "pitfalls": [
            "[API] For Dirichlet eigenvalue problems, restrict "
            "matrices to INTERIOR DOFs: I = basis.complement_dofs("
            "D) where D = basis.get_dofs() (boundary). Then "
            "K_I = K[I][:, I]; M_I = M[I][:, I]; eigsh(K_I, M=M_I, "
            "...). Skipping this leaves the boundary DOFs in "
            "the matrices and the Dirichlet eigenvalues come "
            "back wrong. Signal: len(I) + len(D.nodal['u']) == "
            "basis.N (verified empirically — N=25 on MeshTri."
            "refined(2) splits as 16 boundary + 9 interior). "
            "(Verified empirically 2026-06-01.)",
            "[Numerical] scipy.sparse.linalg.eigsh with sigma=0 "
            "uses shift-and-invert targeting the SMALLEST "
            "eigenvalues. For a Dirichlet-restricted Laplacian "
            "(SPD), sigma=0 is safe (no null space). Signal: "
            "eigsh(K_I, M=M_I, k=5, sigma=0, which='LM') returns "
            "the 5 smallest eigenvalues; switching to "
            "which='SM' (no sigma) is much slower per iteration. "
            "(Verified empirically — see laplace_eigenvalue_basics "
            "fixture.)",
            "[Physics] A P1 EIGENVALUE ERROR GROWS UP THE "
            "SPECTRUM, so a flat percentage tolerance is not a "
            "flat test: the same 5% bar is several times "
            "tighter on the fifth eigenvalue than on the first, "
            "and a suite that passes on mode 1 and fails on "
            "mode 5 is usually behaving correctly. "
            "The Dirichlet Laplacian on a rectangle has a "
            "closed-form spectrum you can derive yourself from "
            "the domain -- do that for YOUR domain and compare "
            "against it, including the degeneracies, rather "
            "than against a list copied from anywhere. "
            "Signal: with ElementTriP1 on a refined MeshTri "
            "the relative error roughly quadruples between "
            "the first and the fifth computed eigenvalue, so "
            "one relative tolerance passes mode 1 and fails "
            "mode 5. "
            "(Verified empirically 2026-06-01; re-measured "
            "2026-08-03.)",
            "[Numerical] Structural vibration eigenproblem "
            "K*x = omega^2*M*x — eigenvalues are squared "
            "angular frequencies. Take omega = sqrt(eig) to get "
            "physical natural frequencies in rad/s. Signal: "
            "passing both K and M to eigsh as eigsh(K, M=M, ...) "
            "solves the generalised problem; passing only K "
            "solves the standard problem (against identity), "
            "giving wrong frequency values that get WORSE under "
            "refinement rather than better, because without M the "
            "result carries the mesh scale. (Verified empirically "
            "2026-08-30 on the unit-square Dirichlet Laplacian, "
            "first eigenvalue, exact 2*pi^2 = 19.739: generalised "
            "gives 19.930 / 19.787 / 19.751 at refine 4 / 5 / 6, "
            "converging at order 2; K alone gives 0.077 / 0.019 / "
            "0.005, i.e. 99.6% / 99.9% / 100.0% error.)",
        ],
    },
}

GENERATORS = {
    "eigenvalue_2d": _eigenvalue_2d,
}
