"""scikit-fem point-source / Dirac-delta load generator + knowledge.

Mirrors scikit-fem upstream ex38 (point source via a scalar Dirac
delta). A point source f = δ(x - x₀) cannot be integrated via a
quadrature rule; instead the load is assembled by adding the
test-function values at x₀ directly to the RHS vector at the
nearest mesh node OR by
projecting δ(x-x₀) onto the FE space via `Basis.interpolate`.

For P1 on the unit square with f = δ(x - x₀):
    -Δu = δ(x - x₀)  in Ω
    u   = 0          on ∂Ω
The discrete RHS is b_i = N_i(x₀) where N_i are P1 basis
functions; for a node-coincident point source this collapses to a
single nonzero at the corresponding DOF.
"""


def _point_source_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate
    values for your specific problem.

    Point source at (x0, y0) on the unit square, homogeneous
    Dirichlet BCs, P1 triangles. The exact Green's function on
    [0,1]² (with Dirichlet BCs) is a Fourier series — we just
    check that the FE solution is bounded and peaks at the source
    location."""
    nx = params.get("nx", 32)
    x0 = params.get("x0", 0.5)
    y0 = params.get("y0", 0.5)
    return f'''\
"""Point-source Poisson on the unit square — scikit-fem"""
from skfem import (MeshTri, Basis, ElementTriP1, solve, condense)
from skfem.models.poisson import laplace
import numpy as np
import json

nx = {nx}
m = MeshTri.init_tensor(np.linspace(0, 1, nx + 1),
                        np.linspace(0, 1, nx + 1))
e = ElementTriP1()
ib = Basis(m, e)

K = laplace.assemble(ib)

# RHS via N_i(x0) — for a node-coincident source on the
# tensor-product mesh this collapses to a single nonzero entry
# at the nearest grid node. For off-node sources we'd need to
# locate the containing element and assemble the three barycentric
# weights into the RHS.
x0, y0 = {x0}, {y0}
nodes = m.p.T                                  # (n_nodes, 2)
dists = np.linalg.norm(nodes - np.array([x0, y0]), axis=1)
source_node = int(np.argmin(dists))

# For a properly mesh-coincident source this approximates
# δ(x-x0) → e_{{source_node}} (Kronecker). Off-node sources
# require barycentric distribution; raise if the source is
# more than h/sqrt(2) from any node so the approximation is
# clearly bad.
h = 1.0 / nx
if dists[source_node] > h / np.sqrt(2.0):
    print(f"WARNING: source ({{x0}}, {{y0}}) is "
          f"{{dists[source_node]:.4f}} from nearest node — "
          f"point-source approximation is rough (h/sqrt(2) = "
          f"{{h/np.sqrt(2.0):.4f}}).")

f = ib.zeros()
f[source_node] = 1.0                           # δ-like load

# Homogeneous Dirichlet on the whole boundary.
D = ib.get_dofs().flatten()
u = solve(*condense(K, f, D=D))

# Sanity: u should peak near the source node and be bounded.
peak_dof = int(np.argmax(u))
print(f"point source at ({{x0}}, {{y0}}) → node {{source_node}}; "
      f"peak DOF {{peak_dof}}; u_peak = {{u[peak_dof]:.4e}}; "
      f"max|u| = {{np.abs(u).max():.4e}}")

import meshio
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
mio = meshio.Mesh(points, [("triangle", m.t.T)],
                  point_data={{"u": u}})
mio.write("result.vtu")

summary = {{
    "n_dofs": int(ib.N),
    "source_node": source_node,
    "peak_dof": peak_dof,
    "source_node_equals_peak": int(source_node == peak_dof),
    "u_peak": float(u[peak_dof]),
    "u_max_abs": float(np.abs(u).max()),
    "source_xy": [x0, y0],
    "h": float(h),
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


GENERATORS: dict = {
    "point_source_2d": _point_source_2d,
}


KNOWLEDGE: dict = {
    "point_source": {
        "description": (
            "Point-source / Dirac-delta load Poisson problem. "
            "f = δ(x - x0) cannot be integrated via quadrature; "
            "the discrete RHS is b_i = N_i(x0) — collapses to a "
            "Kronecker e_node entry for mesh-coincident sources. "
            "Matches scikit-fem upstream ex38 (point source via "
            "CellBasis.point_source)."
        ),
        "weak_form": (
            "(grad(u), grad(v))_dx = N_i(x0) v_i  "
            "(P1: b is one-hot at the source DOF when x0 is a "
            "mesh node; otherwise barycentric weights at the "
            "containing element)."
        ),
        "elements": ["ElementTriP1"],
        "variants": ["2d"],
        "pitfalls": [
            "[Numerical] A Dirac source δ(x-x0) is NOT smooth, "
            "so the regularity of the analytical solution is at "
            "best in W^{1,p} for p<2 in 2D — the FE solution "
            "loses H^1 convergence as h → 0, and the L^2 error "
            "converges well short of the optimal order a smooth "
            "source would give. For convergence studies, "
            "regularize the source to a narrow Gaussian "
            "exp(-|x-x0|^2 / (2*sigma^2)) with sigma ~ h. "
            "Signal: do NOT gate on a FLAT H^1 error — it is not "
            "flat, it decreases, only far more slowly than "
            "first order, so an assertion that it does not move "
            "goes RED on a correct run. Gate on the RATES and on "
            "their separation instead: fit the observed order of "
            "both norms over a refinement sequence and require "
            "the H^1 order to sit far below one while the L^2 "
            "order sits clearly below the optimal two, with the "
            "L^2 error falling by a much larger factor than the "
            "H^1 error across the same sequence. Note also that "
            "the pointwise PEAK max(u) GROWS monotonically under "
            "refinement, because the Green's function has a log "
            "singularity at the source — so max(u) is a "
            "divergence indicator here, never a convergence "
            "check. "
            "Measured 2026-08-03 on skfem 12.0.1 — unit source "
            "at the centre node, ElementTriP1 on "
            "MeshTri().refined(r) for r = 3..6, errors taken "
            "against a finer reference projected onto each mesh: "
            "the L2 error falls at roughly first order per "
            "halving, while the H1-seminorm error barely moves "
            "across the whole sequence — "
            "shrinking, but by a small fraction over an 8x "
            "refinement, which is what 'does not converge' looks "
            "like in practice. (Verified empirically 2026-08-03, "
            "re-measured 2026-08-06 on skfem 12.0.1 — the "
            "substance holds; the earlier 'stays O(1) / FLAT' "
            "wording overstated it and is corrected.)",
            "[API] `ib.zeros()` returns a fresh float64 ndarray "
            "of length ib.N. Setting `f[source_node] = 1.0` is "
            "the correct one-hot assembly for a node-coincident "
            "source. `unit_load.assemble(ib)` is the f=1 "
            "constant-source assembly — different physics "
            "entirely — but it does NOT put 1.0 at every DOF: "
            "each entry is the INTEGRAL of the corresponding "
            "basis function, so the entries vary with the local "
            "mesh and sum to the measure of the domain. "
            "Signal: counting peaks does not separate the two. "
            "The unit-load solution is a single smooth bump with "
            "ONE interior maximum, at the same location the "
            "point-source solution peaks — there is no "
            "'peaks at all interior nodes' pattern to look for. "
            "What separates them is the peak VALUE, an order of "
            "magnitude apart, and, sharply, Green's-function "
            "reciprocity: the integral of the point-source "
            "solution over the domain equals the unit-load "
            "solution evaluated AT the source point, to "
            "round-off, because the Green's function is "
            "symmetric. Check the load vector directly too — "
            "assert it has exactly one non-zero entry for a "
            "point source, and that its sum equals the domain "
            "measure for a unit load. (Verified 2026-08-06 on "
            "skfem 12.0.1 — both unit_load claims falsified.)",
            "[Mesh] If x0 is NOT a mesh node, np.argmin on the "
            "vertex array picks the NEAREST node but introduces "
            "a discretization error proportional to "
            "h/sqrt(2) (max distance from any point in a "
            "triangulated unit square to the nearest node). To "
            "place an off-node source properly, locate the "
            "containing triangle and distribute the unit load by "
            "barycentric coordinates: b[v0..v2] += λ0, λ1, λ2. "
            "Signal: peak in u sits at a slightly different "
            "location from x0; u_peak < expected; "
            "results_summary.json reports its "
            "source_node_equals_peak flag as 1 while the user "
            "expected a fractional source. "
            "Concretely: `np.argmin` on the `MeshTri.p.T` vertex "
            "array always returns an integer node id, never a "
            "barycentric position, so off-node sources are "
            "silently rounded to the nearest mesh vertex. The "
            "fix uses `Basis.get_dofs` plus barycentric "
            "interpolation, not raw `np.argmin`.",
            "[API] `condense(K, f, D=D)` returns 4 values "
            "(K_c, f_c, x_c, I) that you must unpack via "
            "`solve(*condense(...))`. Passing `condense(K, f, D=D)` "
            "directly to a scipy solver fails — but not with the "
            "argument-count error you would expect, because "
            "spsolve accepts extra positionals and binds them to "
            "its own keyword parameters. "
            "Signal: measured 2026-08-03 on skfem 12.0.1 / scipy "
            "1.15.3, condense(K, f, D=D) returns a 4-tuple of "
            "(csr_matrix, ndarray, ndarray, ndarray) and "
            "scipy.sparse.linalg.spsolve(*that) raises "
            "ValueError('The truth value of an array with more "
            "than one element is ambiguous. Use a.any() or "
            "a.all()') from linsolve.py — the fourth element (I) "
            "gets bound to spsolve's use_umfpack flag (the third, "
            "x, to permc_spec). It is NOT the "
            "'spsolve() got too many positional arguments' "
            "TypeError the prior text quoted, and not "
            "'unhashable type: tuple' either. condense's "
            "contract is unique to scikit-fem — always go "
            "through skfem.solve. (Verified empirically "
            "2026-08-03 — signal-text correction.)",
            "[Physics] Total integral of u over Ω equals "
            "Green's-function flux balance: u solves "
            "-Δu = δ(x-x0) on the unit square with Dirichlet BC. "
            "The identity to check against is NOT "
            "'-1/lambda1 in the Laplacian spectrum' — that is "
            "the wrong closed form. It is the double sine "
            "series: ∫_Ω u dx is the sum over ODD p, q of "
            "16*(-1)^((p-1)/2)*(-1)^((q-1)/2) / "
            "(pi^4 * p * q * (p^2 + q^2)) for a source at the "
            "centre, which you truncate yourself to the "
            "precision you need. Take the discrete integral as a "
            "@Functional or as 1^T @ (M @ u), not as a mean "
            "times h^2. "
            "Signal: the integral IS a usable verification "
            "quantity even though the pointwise solution is not "
            "— it converges to the analytic series at the "
            "optimal order, so a discrete integral that "
            "disagrees with the analytic value is real evidence. "
            "But do NOT expect an 'orders of magnitude' "
            "discrepancy from every fault: doubling the source "
            "doubles the integral and flipping the sign of K "
            "flips it, both easy to see, while dropping a "
            "Dirichlet condition on one side shifts it only by "
            "tens of percent. All three faults are SILENT — "
            "empty warning list, finite solution — so the check "
            "needs a tight relative tolerance against the "
            "truncated series, not a magnitude sanity band. "
            "(Verified 2026-08-06 on skfem 12.0.1 — the stated "
            "identity was wrong and the 'orders of magnitude' "
            "signal overstates the boundary-condition case.)",
            "[Output] VTK output for a 2D MeshTri wants "
            "3D-padded points: "
            "`np.column_stack([m.p.T, np.zeros(m.p.shape[1])])`. "
            "Do NOT rely on an exception to catch the missing "
            "z-column: meshio 5.3.5 does not raise on an (N, 2) "
            "points array — `meshio.Mesh` constructs fine and the "
            "writer prints 'Warning: VTU requires 3D points, but "
            "2D points given. Appending 0 third component.' and "
            "writes the file anyway. Pad explicitly if you want "
            "the coordinates to be what you intended rather than "
            "what meshio guessed. Signal: that warning line on "
            "stderr (printed by meshio via rich "
            "Console(stderr=True); not a Python warning, not an "
            "exception) at the "
            "meshio write call, and a .vtu whose point array has "
            "3 columns you never supplied; there is no "
            "'expected ndarray of shape (N, 3)' ValueError from "
            "the meshio.Mesh constructor. (Quoted string "
            "re-checked live 2026-08-06 on meshio 5.3.5 and found "
            "absent.)",
        ],
        "references": [
            "scikit-fem ex38 (point source via Dirac delta)",
            "Brenner & Scott, 'The Mathematical Theory of "
            "Finite Element Methods', §0.5 (Sobolev embedding "
            "of Dirac delta).",
        ],
    },
}
