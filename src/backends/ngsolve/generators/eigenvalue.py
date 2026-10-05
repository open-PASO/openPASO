"""NGSolve eigenvalue problem generators and knowledge."""


def _eigenvalue_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Eigenvalue problem: Laplace on unit square."""
    order = params.get("order", 4)
    n_eigs = params.get("n_eigenvalues", 10)
    maxh = params.get("maxh", 0.03)
    return f'''\
"""Eigenvalue problem: Laplace — ArnoldiSolver — NGSolve"""
from ngsolve import *
import json, math

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))
fes = H1(mesh, order={order}, dirichlet="bottom|right|top|left")
u, v = fes.TnT()

a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
m = BilinearForm(u*v*dx).Assemble()

gfu = GridFunction(fes, multidim={n_eigs})
# ArnoldiSolver returns the eigenvalues UNSORTED (Krylov ordering) and the
# tail of the requested window may be unconverged garbage (observed:
# lambda_9 = 218523 vs 167.78 while the sorted spectrum was exact to
# 1e-11 — Mac stress audit 2026-07-18). Request a buffer of extra Krylov
# vectors, then SORT and report only the leading n_eigs.
n_report = {n_eigs}
n_krylov = {n_eigs} + 10
gfu_k = GridFunction(fes, multidim=n_krylov)
lam_raw = ArnoldiSolver(a.mat, m.mat, fes.FreeDofs(), list(gfu_k.vecs), shift=0)
order_idx = sorted(range(n_krylov), key=lambda i: float(complex(lam_raw[i]).real))
lam = [float(complex(lam_raw[i]).real) for i in order_idx][:n_report]
for _k, _i in enumerate(order_idx[:n_report]):
    gfu.vecs[_k].data = gfu_k.vecs[_i]

print(f"First {n_eigs} eigenvalues:")
exact = [math.pi**2*(i**2+j**2) for i in range(1,6) for j in range(1,6)]
exact.sort()
checked = []
for i, (c, ref) in enumerate(zip(lam, exact[:n_report])):
    err = abs(c - ref) / ref
    checked.append((c, ref, err))
    print(f"  lambda_{{i+1}} = {{c:.6f}} (exact: {{ref:.6f}}, error: {{err:.2e}})")

# Self-check: a mis-ordered or unconverged spectrum must not be reported.
bad = [i + 1 for i, (_c, _r, e) in enumerate(checked) if e > 0.05]
if bad:
    raise SystemExit(
        f"Eigenvalues {{bad}} deviate >5% from the analytic Dirichlet-Laplace "
        f"sequence — spectrum unconverged; increase order / reduce maxh / "
        f"reduce n_eigenvalues.")

# GridFunction(fes, multidim=N) stores N modes in
# gfu.vecs[0..N-1].  gfu.components is for COMPOUND
# FESpaces (e.g. VectorH1*H1), not multidim — using it
# here raises IndexError 'tuple index out of range'.
# To plot the first mode, copy it into gfu.vec and pass
# gfu itself to VTKOutput.
gfu.vec.data = gfu.vecs[0]
vtk = VTKOutput(mesh, coefs=[gfu], names=["eigenmode_1"],
                filename="result", subdivision=1)
vtk.Do()
# lam is already sorted-real (see above).
summary = {{"eigenvalues": [float(l) for l in lam], "n_dofs": fes.ndof}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Eigenvalue solve complete.")
'''


KNOWLEDGE = {
    "eigenvalue": {
        "description": "Eigenvalue problems via ArnoldiSolver (shift-invert Arnoldi)",
        "spaces": "Any H1 space",
        "solver": "ArnoldiSolver(a.mat, m.mat, freedofs, vecs, shift=target)",
        "pitfalls": [
            "[Numerical] ArnoldiSolver(a.mat, m.mat, fes.FreeDofs(), "
            "vecs, shift=target) uses shift-and-invert: eigenvalues "
            "near 'shift' converge fastest. shift=0 fails for "
            "operators whose kernel is LARGE relative to the "
            "space — the H(curl) curl-curl operator, where the "
            "gradient kernel is a substantial fraction of the "
            "free DOFs — raising NgException 'UmfpackInverse: "
            "Numeric factorization failed.' (same family as "
            "maxwell#5). For Laplace eigenproblems with "
            "Dirichlet BCs the matrix is positive-definite so "
            "shift=0 is safe. REFINEMENT: 'the operator has a "
            "null space' is too coarse a rule and costs you a "
            "usable shift. The NEUMANN Laplacian HAS a null "
            "space — one dimension, the constants — and shift=0 "
            "goes straight through it without complaint, "
            "returning the kernel mode as an ordinary eigenvalue "
            "at round-off. Signal: guard on the EXCEPTION, not "
            "on the warning text. The 'matrix is singular' line "
            "that UMFPACK prints alongside it is written through "
            "C stdio, which by default holds it until process "
            "exit when stdout is a file or a pipe, so a capture "
            "placed around the ArnoldiSolver call comes back "
            "EMPTY (an fd-level os.dup2 of fd 1 as well as "
            "contextlib.redirect_stdout) and an agent reading "
            "that capture concludes nothing happened; only a "
            "line-buffered stdout (a terminal, or stdbuf -oL) "
            "shows it during the call. Catch NgException, and if you "
            "want the shift=0 case to be safe, move the shift "
            "off zero — a non-zero shift completes on the same "
            "operator, so the failure is specific to shift=0 and "
            "not a bad shift in general. (Verified empirically "
            "2026-08-03, refined 2026-08-06 on NGSolve 6.2.2604 "
            "— the kernel rule is narrower than stated and the "
            "quoted warning text is not capturable around the "
            "call.)",
            "[API] GridFunction(fes, multidim=n) allocates space "
            "for n independent vectors (e.g., n eigenvectors). "
            "Accessed via gfu.vecs[i] (a list-like sequence), "
            "NOT via gfu.mdcomponents (which does not exist as "
            "an attribute). Signal: hasattr(gfu, 'vecs') is True, "
            "len(list(gfu.vecs)) == n; hasattr(gfu, "
            "'mdcomponents') is False. (Verified empirically "
            "2026-06-01 — catalog text tightened from prose "
            "to name the actual access path.)",
            "[Physics] Exact analytic eigenvalues of the "
            "Dirichlet Laplacian on [0,1]^2 are pi^2*(m^2+n^2) "
            "for m, n >= 1. First few: 2*pi^2, 5*pi^2, 5*pi^2 "
            "(degenerate), 8*pi^2, 10*pi^2... Signal: "
            "ArnoldiSolver result on a maxh<=0.05 mesh with "
            "order>=2 elements should agree with these values "
            "to within ~0.5%; larger discrepancy indicates mesh "
            "too coarse or wrong FE order. Measured on "
            "unit_square with shift=0: maxh=0.05 order=2 gives "
            "19.7392 / 49.3486 / 49.3487 / 78.9591 / 98.7006 "
            "(0.000% / 0.001% / 0.001% / 0.003% / 0.005% "
            "relative); maxh=0.1 order=2 gives 0.003-0.094%; "
            "maxh=0.2 order=1 gives 7.1% / 16.5% / 17.1% / "
            "27.4% / 30.9% — so P1 on a coarse mesh misses the "
            "0.5% bar by a wide margin and the error grows "
            "rapidly up the spectrum. (Verified empirically "
            "2026-08-03 on NGSolve 6.2.2604.)",
            "[Syntax] For the generalized eigenvalue problem "
            "A*x = lambda*M*x, pass BOTH matrices to "
            "ArnoldiSolver as a.mat and m.mat — the signature "
            "requires two, so 'passing only A' is not a "
            "reachable mistake. Passing the SAME matrix twice "
            "does NOT fall back to an identity mass (the prior "
            "catalog mechanism was wrong): it forms the pencil "
            "(A, A), whose exact spectrum is all-ones, so what "
            "comes back is 1.0 plus floating-point noise. "
            "Signal: on unit_square maxh=0.1, H1(order=2, "
            "dirichlet='.*'), ArnoldiSolver(a.mat, a.mat, ...) "
            "returned -3.35e+15, 1.0000, 1.00e+14, 1.27e+14 "
            "while the correct (a.mat, m.mat) call returns "
            "19.7399, 49.3581, 49.3596, 78.9997. Any eigenvalue "
            "of exactly 1.0 or of magnitude 1e14+ from an "
            "elliptic eigenproblem means you duplicated the "
            "matrix. (Verified empirically 2026-08-03 on "
            "NGSolve 6.2.2604 — mechanism correction.)",
            "[API] Alternative eigenvalue solver: "
            "ngsolve.solvers.PINVIT (preconditioned inverse "
            "iteration) for the lowest eigenvalues. PINVIT "
            "scales better than ArnoldiSolver on large meshes "
            "because it does NOT need a global factorisation — "
            "only a preconditioner application per iteration. "
            "Signal: PINVIT result on a mesh with > 10^5 dofs "
            "completes in a fraction of the wall time of "
            "ArnoldiSolver; PINVIT's per-iter cost is O(N) "
            "vs ArnoldiSolver's O(N^1.5). (Claim inherited.)",
        ],
    },
}

GENERATORS = {
    "eigenvalue_2d": _eigenvalue_2d,
}
