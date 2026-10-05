"""scikit-fem 1D Schrödinger generator + knowledge.

Solves the 1D stationary Schrödinger equation for a particle in a
potential V(x):

    -(ℏ²/2m) ψ''(x) + V(x) ψ(x) = E ψ(x)    x ∈ [-L, L]
    ψ(-L) = ψ(+L) = 0  (zero at far-field)

Default test case: quantum harmonic oscillator V(x) = (1/2) m ω² x²
with analytic eigenvalues E_n = (n + 1/2) ℏω, n = 0, 1, 2, ...

In atomic units (ℏ = m = ω = 1), E_n = (n + 1/2). The lowest 4
eigenvalues are 0.5, 1.5, 2.5, 3.5 — and a P1 finite-element
discretization on a fine mesh should reproduce these to 4+ digits.

No upstream scikit-fem Schrödinger example; the closest pattern is
ex16 (1D MeshLine generalized eigenproblem via
eigsh(L, M=M, sigma=0.)).
"""


def _schrodinger_1d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate
    values for your specific problem.

    1D quantum harmonic oscillator on [-L, L] with k lowest
    eigenpairs computed via scipy.sparse.linalg.eigsh."""
    L = params.get("L", 8.0)
    nx = params.get("nx", 200)
    k = params.get("n_eigs", 4)
    return f'''\
"""1D Schrödinger — quantum harmonic oscillator — scikit-fem"""
from skfem import (MeshLine, Basis, ElementLineP1, BilinearForm,
                   condense)
from skfem.helpers import dot, grad
import numpy as np
from scipy.sparse.linalg import eigsh
import json

L = {L}
nx = {nx}
k_eigs = {k}

# Mesh: [-L, L] with nx+1 nodes.
m = MeshLine(np.linspace(-L, L, nx + 1))
e = ElementLineP1()
ib = Basis(m, e)


# Kinetic operator: (1/2) ∫ ψ' v' dx (atomic units: ℏ=m=1)
@BilinearForm
def kinetic(u, v, w):
    return 0.5 * dot(grad(u), grad(v))


# Potential operator: ∫ V(x) ψ v dx with V(x) = (1/2) x²
@BilinearForm
def potential(u, v, w):
    return 0.5 * w.x[0] ** 2 * u * v


# Mass matrix: ∫ ψ v dx (RHS in generalized eigenvalue problem)
@BilinearForm
def mass(u, v, w):
    return u * v


# Assemble.
K = kinetic.assemble(ib)
V = potential.assemble(ib)
M = mass.assemble(ib)
H = K + V

# Dirichlet BCs: ψ = 0 at both endpoints.
D = ib.get_dofs().flatten()
H_c, M_c, _, I = condense(H, M, D=D)

# Solve generalized eigenvalue problem H ψ = E M ψ for the
# k_eigs lowest eigenvalues. shift-invert mode (sigma=0) is the
# fast path for finding eigenvalues near zero.
E_vals, V_vecs = eigsh(H_c, M=M_c, k=k_eigs, sigma=0.0,
                       which="LM")

# Sort ascending (eigsh returns in arbitrary order under
# shift-invert).
order = np.argsort(E_vals)
E_vals = E_vals[order]
V_vecs = V_vecs[:, order]

# Analytic: E_n = n + 1/2 (atomic units).
E_exact = np.array([n + 0.5 for n in range(k_eigs)])
abs_err = np.abs(E_vals - E_exact)
max_err = float(abs_err.max())

print(f"1D Schrödinger (harmonic oscillator) on [-{{L}}, {{L}}]")
print(f"  n_dofs = {{ib.N}}, k_eigs = {{k_eigs}}")
print(f"  {{'n':>3}} {{'E_fe':>10}} {{'E_exact':>10}} {{'abs_err':>10}}")
for n, (Efe, Eex) in enumerate(zip(E_vals, E_exact)):
    print(f"  {{n:>3d}} {{Efe:10.6f}} {{Eex:10.6f}} {{abs(Efe-Eex):10.4e}}")
print(f"max abs error: {{max_err:.4e}}")

summary = {{
    "n_dofs": int(ib.N),
    "k_eigs": int(k_eigs),
    "L": float(L),
    "eigenvalues_fe": [float(e) for e in E_vals],
    "eigenvalues_exact": [float(e) for e in E_exact],
    "max_abs_error": float(max_err),
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


GENERATORS: dict = {
    "schrodinger_1d": _schrodinger_1d,
}


KNOWLEDGE: dict = {
    "schrodinger": {
        "description": (
            "1D stationary Schrödinger equation "
            "-½ψ''(x) + V(x)ψ(x) = Eψ(x). Default V is the quantum "
            "harmonic oscillator V = ½x² with analytic eigenvalues "
            "E_n = (n + ½) (atomic units ℏ=m=ω=1). Generalized "
            "eigenvalue problem solved via "
            "scipy.sparse.linalg.eigsh with shift-invert (sigma=0) "
            "to target the lowest k."
        ),
        "weak_form": (
            "H ψ = E M ψ where "
            "H_ij = ½ ∫ ψ'_i ψ'_j dx + ∫ V(x) ψ_i ψ_j dx, "
            "M_ij = ∫ ψ_i ψ_j dx."
        ),
        "elements": ["ElementLineP1"],
        "variants": ["1d"],
        "pitfalls": [
            "[Numerical] The harmonic-oscillator eigenfunctions "
            "decay as exp(-x²/2); the domain [-L, L] truncates "
            "this tail at L. For E_n < L²/2 the truncation error "
            "is exponentially small (~exp(-L²/2)). For E_n ≥ L²/2 "
            "the tail-truncation introduces spurious shifts. "
            "Default L=8 gives <1e-10 truncation error for n ≤ 10. "
            "Signal: eigenvalues E_n drift away from the analytic "
            "n+½ as n grows — the highest computed eigenvalues "
            "from `scipy.sparse.linalg.eigsh` are off by O(1) "
            "while the lowest match to 4 digits. The drift onset "
            "moves to higher n as L increases (MeshLine longer "
            "domain captures more of the Gaussian tail).",

            "[API] `scipy.sparse.linalg.eigsh(H, M=M, sigma=0, "
            "which='LM')` is the shift-invert ARPACK call for "
            "smallest eigenvalues. Forgetting `sigma=0` makes "
            "ARPACK target the LARGEST eigenvalues, which on a "
            "fine mesh are O(nx²) (the discretization cutoff), "
            "not the physical low-energy spectrum. "
            "Signal: returned E_vals from `eigsh` are ~1e3 to 1e6 "
            "instead of 0.5, 1.5, 2.5, 3.5; ARPACK still "
            "converges but the eigenpairs are mesh-noise "
            "eigenmodes from the `BilinearForm`-assembled "
            "stiffness, not bound states. "
            "`np.abs(E_vals - E_exact)` is O(1).",

            "[API] `eigsh` returns eigenvalues in arbitrary "
            "order under shift-invert; explicit "
            "`order = np.argsort(E_vals)` is required to read off "
            "E_0, E_1, ... in physical (ascending) order. "
            "Signal: `eigenvalues_fe[0]` from `eigsh` printed "
            "as 1.5 instead of 0.5; spacings look correct but "
            "starting point is wrong; the FIRST eigenvector "
            "printed has 1 node (should be smooth ground state). "
            "Fix: explicit `np.argsort(E_vals)` to reorder both "
            "`E_vals` and the corresponding columns of `V_vecs`.",

            "[Mesh] MeshLine takes the SORTED node coordinates "
            "as a numpy array: `MeshLine(np.linspace(-L, L, nx+1))`. "
            "Passing an unsorted array (e.g., from a random "
            "grid generator without np.sort) corrupts element "
            "connectivity and produces a degenerate stiffness "
            "matrix. "
            "Signal: NOTHING at construction — this is a silent "
            "trap. Measured 2026-08-03 on skfem 12.0.1: "
            "MeshLine(np.array([0., 0.5, 0.25, 0.75, 1.0])) "
            "constructs happily, m.p comes back in the same "
            "unsorted order, the assembled stiffness matrix is "
            "still positive SEMI-definite (min eigenvalue "
            "1.05e-15, no negative eigenvalues), and eigsh runs "
            "to completion — it just returns WRONG numbers: "
            "the first two interior eigenvalues come out 4.686 "
            "and 24.0 instead of the 10.387 and 48.0 the same "
            "code gives on the sorted grid (exact pi^2 n^2 = "
            "9.8696, 39.478). There is no 'invalid mesh: "
            "non-monotonic coordinates' TypeError and no "
            "ArpackNoConvergence. Defend with an explicit "
            "assert np.all(np.diff(pts) > 0) before "
            "constructing. (Verified empirically 2026-08-03 — "
            "catalog-drift correction: the failure is silent, "
            "not loud.)",

            "[API] `condense(H, M, D=D)` for a GENERALIZED "
            "eigenvalue problem returns FOUR values: "
            "(H_c, M_c, _, I) — the underscore is a placeholder "
            "for the prescribed x vector (irrelevant since we "
            "use the M matrix, not f vector). DON'T do "
            "`H_c, M_c = condense(...)` — it raises "
            "ValueError 'too many values to unpack'. "
            "Signal: ValueError 'too many values to unpack "
            "(expected 2)' at the condense line; or "
            "`scipy.sparse.linalg.eigsh` argument mismatch "
            "because M_c got bound to the unpacked second value "
            "(the f-vector) instead of the actual M matrix.",

            "[Numerical] The harmonic-oscillator ground state "
            "ψ_0(x) ∝ exp(-x²/2) decays smoothly; a P1 "
            "discretization underresolves the curvature near "
            "x=0 unless nx is large enough that h = 2L/nx is "
            "smaller than the curvature lengthscale (~1). For "
            "the default 4 lowest eigenvalues with L=8, nx=200 "
            "gives h=0.08 ≪ 1 and max abs error ~5e-3 (E_0 "
            "error ~2e-4). Halving nx "
            "quadruples the error (O(h²) for eigenvalues with "
            "P1). "
            "Signal: max_abs_error from "
            "`scipy.sparse.linalg.eigsh` grows from ~5e-3 "
            "(nx=200) to ~2e-2 (nx=100) to ~8e-2 (nx=50) as "
            "`MeshLine` nx halves; convergence ratio ~4 "
            "between successive refinements (this is the right "
            "P1 rate; the surprise is if the ratio is NOT ~4, "
            "which indicates a bug in the `BilinearForm` "
            "assembly).",
        ],
        "references": [
            "scikit-fem ex16 (1D MeshLine eigsh pattern; no "
            "upstream Schrödinger example)",
            "Griffiths, 'Introduction to Quantum Mechanics' 2nd "
            "ed., Ch. 2 (harmonic oscillator).",
            "Cross-backend: dealii step-25 (Sine-Gordon) and "
            "step-58 (Schrödinger).",
        ],
    },
}
