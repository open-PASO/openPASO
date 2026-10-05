"""scikit-fem adaptive Poisson (h-adaptive) generator and knowledge.

Mirrors scikit-fem upstream ex22 (adaptive Poisson on an L-shaped domain
with a residual estimator). The backend previously had `poisson` (uniform
mesh) but no adaptive h-refinement loop, leaving a clear gap relative to
upstream.

The residual error estimator (Babuška-Rheinboldt) for -Δu = f on P1 is

    η_K² = h_K² ∫_K f² dx + (1/2) Σ_{e ⊂ ∂K} h_e ∫_e [∇u_h · n_e]² ds

(volume residual `f + Δu_h` reduces to `f` for P1 since Δu_h = 0 in the
element interior).  Mark elements with η_K ≥ θ · max_K(η_K), refine, and
repeat until a target DOF budget is reached.
"""


def _adaptive_poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate
    values for your specific problem.

    h-adaptive Poisson with Babuška-Rheinboldt residual estimator,
    L-shaped re-entrant-corner mesh as a canonical test (the
    singular gradient at the corner drives refinement)."""
    n_iters = params.get("n_iters", 5)
    theta = params.get("theta", 0.5)
    dof_budget = params.get("dof_budget", 20000)
    return f'''\
"""h-adaptive Poisson with residual estimator — scikit-fem"""
from skfem import (MeshTri, Basis, ElementTriP1, solve, condense,
                   FacetBasis, InteriorFacetBasis, Functional)
from skfem.models.poisson import laplace, unit_load
from skfem.helpers import dot, grad
from skfem.assembly import asm
import numpy as np
import json

# L-shaped domain by removing the (0,1)x(0,1) quadrant from
# (-1,1)x(-1,1). The re-entrant corner at (0,0) drives the
# refinement: solution gradient is unbounded there for f=1.
def _build_lshape():
    # Build by manually specifying nodes + triangles.  Six
    # quadrants of the [-1,1]^2 square, minus the upper-right one,
    # cut into right triangles.
    p = np.array([
        [-1.0, -1.0], [0.0, -1.0], [1.0, -1.0],
        [-1.0,  0.0], [0.0,  0.0], [1.0,  0.0],
        [-1.0,  1.0], [0.0,  1.0],
    ]).T
    t = np.array([
        [0, 1, 4], [0, 4, 3],
        [1, 2, 5], [1, 5, 4],
        [3, 4, 7], [3, 7, 6],
    ]).T
    return MeshTri(p, t)

m = _build_lshape()
# Pre-refine to give the adaptive loop something to bite into.
m = m.refined()
m = m.refined()

theta = {theta}
n_iters = {n_iters}
dof_budget = {dof_budget}

history = []  # list of (iter, n_dof, max_eta, sum_eta)

for k in range(n_iters):
    e = ElementTriP1()
    ib = Basis(m, e)
    if ib.N > dof_budget:
        print(f"DOF budget {{dof_budget}} reached at iter {{k}} (N={{ib.N}})")
        break

    K = laplace.assemble(ib)
    f = unit_load.assemble(ib)
    # Homogeneous Dirichlet on the whole boundary.
    D = ib.get_dofs().flatten()
    u = solve(*condense(K, f, D=D))

    # ── Residual error estimator (Babuska-Rheinboldt) ─────────
    # eta_K^2 = h_K^2 * int_K f^2 dx
    #          + 0.5 * sum_{{e in dK \\ bnd}} h_e * int_e [du/dn]^2 ds
    # For P1, Delta(u_h) = 0 inside elements, so the volume term
    # reduces to h_K^2 * ||f||^2_K with f = 1.

    # Element diameter h_K ~ max edge length.
    p_t = m.p[:, m.t]                              # (2, 3, nE)
    e01 = p_t[:, 1, :] - p_t[:, 0, :]
    e12 = p_t[:, 2, :] - p_t[:, 1, :]
    e20 = p_t[:, 0, :] - p_t[:, 2, :]
    len01 = np.linalg.norm(e01, axis=0)
    len12 = np.linalg.norm(e12, axis=0)
    len20 = np.linalg.norm(e20, axis=0)
    h_K = np.maximum.reduce([len01, len12, len20])

    # Element area via 2D cross-product (oriented).
    area = 0.5 * np.abs(e01[0] * e20[1] - e01[1] * e20[0])

    # Volume residual term (f=1 here): eta_vol^2 = h_K^2 * area.
    eta_vol2 = (h_K ** 2) * area

    # Jump term across interior facets:
    # For P1, grad(u) is constant on each element.  On each
    # interior facet, jump = (grad_in - grad_out) . n.
    grad_u = np.zeros((2, m.t.shape[1]))
    # Per-element gradient from nodal values via the inverse
    # element Jacobian.  P1 shape functions span affine maps;
    # use the closed-form formula:
    #   grad(u)|_K = sum_i u_i * (1/(2A_K)) * R(p_{{i+1}} - p_{{i+2}})
    # where R rotates by 90 deg.  Cleaner: compute via skfem.
    grad_u_field = ib.interpolate(u).grad         # (2, nE, nq)
    # The Cell quadrature is constant-on-element for P1, so grad
    # is identical across quadrature points; take the first.
    grad_u = grad_u_field[:, :, 0]

    eta_jmp2 = np.zeros(m.t.shape[1])
    # Iterate over interior facets only.  m.f2t has shape
    # (2, n_facets) — second row is -1 for boundary facets.
    f2t = m.f2t
    interior_facets = np.where(f2t[1] >= 0)[0]
    for fi in interior_facets:
        e_in, e_out = f2t[0, fi], f2t[1, fi]
        # Facet endpoints.
        vi0, vi1 = m.facets[:, fi]
        edge_vec = m.p[:, vi1] - m.p[:, vi0]
        h_e = np.linalg.norm(edge_vec)
        n = np.array([edge_vec[1], -edge_vec[0]]) / h_e
        jump = (grad_u[:, e_in] - grad_u[:, e_out]) @ n
        # Contribution: 0.5 * h_e * jump^2 * h_e  (line integral
        # of squared constant jump over facet of length h_e).
        contrib = 0.5 * h_e * (jump ** 2) * h_e
        eta_jmp2[e_in] += contrib
        eta_jmp2[e_out] += contrib

    eta2 = eta_vol2 + eta_jmp2
    eta = np.sqrt(eta2)

    max_eta = float(eta.max())
    sum_eta = float(eta.sum())
    history.append((int(k), int(ib.N), max_eta, sum_eta))
    print(f"iter={{k}} N={{ib.N:6d}}  max_eta={{max_eta:.4e}}  "
          f"sum_eta={{sum_eta:.4e}}")

    # Mark elements with eta_K >= theta * max(eta) — Dorfler-style
    # but with absolute threshold for simplicity.
    if max_eta < 1e-12:
        print("estimator below threshold — stopping")
        break
    mark = np.where(eta >= theta * max_eta)[0]
    if len(mark) == 0:
        break
    m = m.refined(mark)

# Final solve on the adapted mesh.
e = ElementTriP1()
ib = Basis(m, e)
K = laplace.assemble(ib)
f = unit_load.assemble(ib)
D = ib.get_dofs().flatten()
u = solve(*condense(K, f, D=D))

import meshio
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
mio = meshio.Mesh(points, [("triangle", m.t.T)],
                  point_data={{"u": u}})
mio.write("result.vtu")

summary = {{
    "n_iters_run": len(history),
    "final_n_dofs": int(ib.N),
    "max_u": float(u.max()),
    "history": history,
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


GENERATORS: dict = {
    "adaptive_poisson_2d": _adaptive_poisson_2d,
}


KNOWLEDGE: dict = {
    "adaptive_poisson": {
        "description": (
            "h-adaptive Poisson with the Babuška-Rheinboldt "
            "residual estimator on triangular P1, driven by an "
            "L-shaped re-entrant-corner test case. Matches "
            "scikit-fem upstream ex22 (adaptive Poisson on an "
            "L-shaped domain with a residual estimator) — the "
            "backend previously had "
            "only uniform Poisson, so this fills the canonical "
            "h-adaptive gap."
        ),
        "weak_form": (
            "Solve -Δu = f, u=0 on ∂Ω. Estimator: "
            "η_K² = h_K² ∫_K f² dx + 0.5 Σ_{e⊂∂K} h_e ∫_e [∇u_h · n]² ds. "
            "Mark with η_K ≥ θ · max_K η_K and refine via "
            "MeshTri.refined(indices)."
        ),
        "elements": ["ElementTriP1"],
        "variants": ["2d"],
        "pitfalls": [
            "[Numerical] For P1 elements the volume residual "
            "Δu_h vanishes inside elements (u_h is piecewise "
            "affine), so the η_K² volume term is just "
            "h_K² ∫_K f² dx — NOT h_K² ∫_K (f + Δu_h)² dx. "
            "Including the (already-zero) Δu_h term harmlessly, "
            "but for P2 or higher you must reinstate "
            "`laplace.assemble` of the test field to get the "
            "second-derivative residual right. "
            "Signal: on P1 the volume term is a CONSTANT-factor "
            "contribution, so dropping it does NOT change the "
            "estimator's ORDER and does NOT produce a plateau — "
            "the true error, the full estimator and the "
            "jump-only estimator all converge at the same rate "
            "and the jump-only estimator keeps falling under "
            "refinement. An order-based or plateau-based gate "
            "sees nothing and reads that as agreement. What "
            "moves is the EFFECTIVITY INDEX (estimator divided "
            "by the true error): it drops by a stable percentage "
            "and stays shifted at every refinement level, so "
            "compute the effectivity against a reference error "
            "and check it is stable AND close to the value the "
            "complete estimator gives, rather than checking a "
            "rate. `laplace.assemble(ib)` and "
            "`unit_load.assemble(ib)` still return the right "
            "system either way; the P1 fact itself is exact — "
            "the gradient does not vary inside a P1 element, so "
            "the interior Laplacian is identically zero. For "
            "P2+, where it is not, the omission is a genuine "
            "missing term. (Verified 2026-08-06 on skfem 12.0.1 "
            "— the order-loss and plateau signal is falsified.)",
            "[API] scikit-fem ≥ 8 expects "
            "`MeshTri.refined(element_indices)` for adaptive "
            "refinement; an index array and a boolean mask of "
            "length n_elements give IDENTICAL meshes. Passing "
            "facet indices raises IndexError because `refined` "
            "interprets its argument as element indices. "
            "Signal: read the IndexError carefully — the message "
            "has the form 'index <i> is out of bounds for axis 1 "
            "with size <n_elements>'. It names AXIS 1, not axis "
            "0, and the size in it is the ELEMENT count, not the "
            "facet count, so a guard grepping for 'axis 0' or "
            "expecting to see the number of facets in the "
            "message never matches. The "
            "dangerous spelling is not the loud one at all: a "
            "SCALAR argument does not mean refine that one "
            "element, it selects the UNIFORM-refinement "
            "overload — `refined(3)` refines the whole mesh "
            "three times and multiplies the element count by a "
            "large factor with no warning, while "
            "`refined(np.array([3]))` refines element 3 alone. "
            "That is the shape a mask/index mix-up most easily "
            "takes. Guard by asserting the element count after "
            "refinement against what the marking step selected. "
            "(Verified 2026-08-06 on skfem 12.0.1 — the quoted "
            "'axis 0' wording corrected and the silent scalar "
            "overload added.)",
            "[API] `Basis.interpolate(u).grad` returns shape "
            "`(spatial_dim, n_elements, n_qpoints)`. For P1 the "
            "gradient is constant per element so taking "
            "`grad_u[:, :, 0]` extracts a per-element gradient "
            "vector cheaply. Confusing the axis order (e.g. "
            "`grad_u[:, 0, :]` for 'gradient of element 0') "
            "produces silently-wrong jump terms. "
            "Signal: silently wrong is right, but there is NO "
            "exception to catch — contracting the wrong slice "
            "against the facet normal does not raise. Both "
            "slices carry the SPATIAL dimension first, so the "
            "contraction is legal either way; the wrong axis is "
            "simply contracted correctly, returning an array of "
            "length n_qpoints where one of length n_elements was "
            "meant. No broadcast error is produced, nothing is "
            "warned, and try/except is not a guard here. What "
            "works is a shape assertion: require the contracted "
            "result to have length `mesh.t.shape[1]` (the element "
            "count) before it is used as an elementwise "
            "indicator, and separately assert that grad_u.shape "
            "equals (mesh.dim(), n_elements, n_qp) — an "
            "assertion to WRITE, not text anything prints. "
            "Downstream, η_K comes out O(1) where it should be "
            "O(h) and refinement targets scattered elements "
            "rather than the re-entrant corner. (Verified "
            "2026-08-06 on skfem 12.0.1 — the previously quoted "
            "broadcast ValueError is not produced by the axis "
            "mix-up.)",
            "[Mesh] `m.f2t` has shape (2, n_facets); the second "
            "row is -1 for boundary facets, and those -1 entries "
            "are exactly `m.boundary_facets()`. Select interior "
            "facets with the mask `m.f2t[1] >= 0` before "
            "dereferencing the neighbour. "
            "Signal: the -1 sentinel does NOT raise. numpy reads "
            "-1 as 'the last element', so `m.t[:, m.f2t[1]]` "
            "returns a full connectivity array with the LAST "
            "element's nodes silently substituted on every "
            "boundary facet — no exception, no warning, and the "
            "SAME SHAPE a correct result would have, so neither "
            "try/except nor a shape check catches it. Every "
            "substituted column differs from the true "
            "neighbouring element, so a jump computed that way "
            "picks up an unrelated element from the far side of "
            "the mesh. The guard that works is the mask itself: "
            "apply `f2t[1] >= 0` and assert that the number of "
            "facets it selects equals m.facets.shape[1] minus "
            "len(m.boundary_facets()). Downstream, η_K is "
            "elevated along the boundary even on a uniform mesh "
            "with a smooth solution and refinement preferentially "
            "refines boundary elements. (Verified 2026-08-06 on "
            "skfem 12.0.1 — the previously quoted IndexError "
            "'index -1 is out of bounds' never fires.)",
            "[Numerical] Dorfler and the max-strategy are TWO "
            "DIFFERENT marking rules and their θ runs in "
            "OPPOSITE directions — 'Dorfler / max-strategy with "
            "θ=0.5' does not name a setting. Dorfler (bulk "
            "chasing) marks the smallest set of elements whose "
            "indicators sum to θ of the total, so θ → 0 marks "
            "ONE element and θ → 1 marks ALL of them. The "
            "max-strategy marks every element with "
            "η_K >= θ·max(η), so θ → 0 marks ALL of them and "
            "θ → 1 marks NONE. Decide which rule you are "
            "implementing before you pick θ; a reader who takes "
            "'θ=0.5, θ → 1 means near-uniform' into a "
            "max-strategy loop gets near-uniform refinement "
            "exactly where they were told to expect one element "
            "at a time. "
            "Signal: print `(ib.N, max_eta)` every iteration and "
            "read the DIRECTION rather than a rate — do not "
            "calibrate on a rule of thumb that has N growing by "
            "1.3-2× per step while max_eta drops 30-50%, "
            "because a correctly running Dorfler "
            "loop at θ=0.5 on the L-shape grows N far more "
            "slowly than that and max_eta need not drop at all "
            "while refinement is still concentrating correctly. "
            "The check that discriminates is to evaluate BOTH "
            "rules on the SAME indicator field and confirm the "
            "marked-set size moves the way your rule says it "
            "should as θ is varied. (Verified 2026-08-06 on "
            "skfem 12.0.1 — the conflated θ convention and the "
            "DOF-growth rule of thumb are both falsified.)",
            "[Numerical] L-shape re-entrant corner gives a "
            "u ~ r^(2/3) singularity at (0,0); uniform "
            "refinement converges at H^1 rate 2/3 (sub-optimal), "
            "while h-adaptive refinement recovers the optimal "
            "rate 1. Run `m = m.refined()` (uniform) once "
            "before the adaptive loop to give the estimator "
            "enough elements to discriminate the singular zone. "
            "Signal: H^1-norm convergence rate observed via "
            "`numpy.linalg.norm(grad(u_h) - grad(u_exact))` "
            "stalls at 0.67 instead of approaching 1.0; the "
            "refined-mesh region does not cluster around (0,0).",
        ],
        "references": [
            "scikit-fem examples: ex22 (adaptive Poisson, "
            "residual estimator)",
            "Babuška, I. & Rheinboldt, W. (1978) — "
            "'Error estimates for adaptive finite element "
            "computations'",
            "Dörfler, W. (1996) — 'A convergent adaptive "
            "algorithm for Poisson's equation'",
        ],
    },
}
