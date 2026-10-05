"""scikit-fem convection-diffusion generators and knowledge."""


def _convdiff_2d_skfem(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Convection-diffusion with SUPG-like stabilization."""
    nx = params.get("nx", 32)
    eps = params.get("diffusion", 0.01)
    return f'''\
"""Convection-diffusion: stabilized — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace
import numpy as np
import json

m = MeshQuad.init_tensor(np.linspace(0, 1, {nx+1}), np.linspace(0, 1, {nx+1}))
e = ElementQuad1()
ib = Basis(m, e)

eps = {eps}
b = np.array([1.0, 0.5])

@BilinearForm
def advdiff(u, v, w):
    return eps * (u.grad[0]*v.grad[0] + u.grad[1]*v.grad[1]) + (b[0]*u.grad[0] + b[1]*u.grad[1]) * v

K = asm(advdiff, ib)
f = np.ones(K.shape[0])  # unit source

D = ib.get_dofs().flatten()
u = solve(*condense(K, f, D=D))

print(f"ConvDiff: max(u) = {{u.max():.6f}}")
summary = {{"max_value": float(u.max()), "n_dofs": K.shape[0], "diffusion": eps}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Convection-diffusion solve complete.")
'''


KNOWLEDGE = {
    "convection_diffusion": {
        "description": "Convection-diffusion: stabilized or DG (examples 07, 25, 42, 50)",
        "solver": "GMRES with diagonal preconditioner (non-symmetric system)",
        "elements": "ElementQuad1 (SUPG), ElementTriDG(ElementTriP1()) for DG",
        "pitfalls": [
            (
                "[API] Custom BilinearForm: BOTH `u.grad` and "
                "`skfem.helpers.grad(u)` are correct — the prior "
                "catalog claim that `grad(u)` raises "
                "`NameError: grad is not defined` is FALSE and "
                "steers agents away from the idiomatic helper. "
                "Signal: on MeshTri().refined(2) with "
                "ElementTriP1, both "
                "@BilinearForm def f(u,v,w): "
                "return dot(grad(u), grad(v))  and "
                "@BilinearForm def f(u,v,w): "
                "return u.grad[0]*v.grad[0] + u.grad[1]*v.grad[1] "
                "assemble to a matrix identical to "
                "skfem.models.poisson.laplace (max abs difference "
                "0.0, nnz 105). What DOES bite is using `@` "
                "(matmul) on the (dim, nelem, nqp) DiscreteField "
                "arrays — that raises ValueError('matmul: Input "
                "operand 1 has a mismatch in its core "
                "dimension...'); use skfem.helpers.dot/ddot "
                "instead. A genuinely undefined free function "
                "raises the ordinary Python NameError at call "
                "time. (Verified empirically 2026-08-03 on skfem "
                "12.0.1 — catalog-drift correction.)"
            ),
            (
                "[API] For DG: use InteriorFacetBasis for "
                "jump terms. Signal: a form that reads the "
                "outward normal as `w.n` assembled against a "
                "plain CellBasis raises "
                "AttributeError(\"Attribute 'n' not found in "
                "'w'.\") — the message names the FormExtraParams "
                "dict `w`, NOT the Basis, so `'Basis' has no "
                "attribute 'normals'` (the prior catalog signal) "
                "is not a string skfem 12.0.1 emits. Sizes on "
                "MeshTri().refined(2) (56 facets, 16 on the "
                "boundary): FacetBasis covers 16 facets, "
                "InteriorFacetBasis covers 40. (Verified "
                "empirically 2026-08-03 on skfem 12.0.1 — "
                "signal-text correction.)"
            ),
            (
                "[API] Periodic mesh: example 42 shows "
                "advection on a periodic domain built with "
                "MeshTri1DG.init_tensor(x, y, periodic=[0]), "
                "which calls the "
                "periodic(mesh, ix, ix0) classmethod. It lives "
                "ONLY on the discontinuous-geometry mesh "
                "classes — MeshTri1DG, MeshQuad1DG, MeshHex1DG, "
                "MeshLine1DG "
                "— and is NOT available on MeshTri / MeshQuad / "
                "Mesh, contrary to the prior catalog text. "
                "Signal: hasattr(MeshTri, 'periodic') is False, "
                "hasattr(MeshTri1DG, 'periodic') is True with "
                "signature (mesh, ix, ix0); there is also no "
                "top-level skfem name containing 'periodic'. So "
                "the incantation is "
                "MeshTri1DG.periodic(m, ix, ix0), not "
                "MeshTri.periodic(...). The physics motivation is "
                "unchanged: "
                "building a regular MeshTri / MeshQuad "
                "without periodic wrapping and expecting "
                "outflow = inflow gives an open-boundary "
                "system; the upstream concentration drains "
                "via free Neumann BC and the downstream "
                "face piles up. The periodic classmethod "
                "identifies left/right (and top/bottom) "
                "mesh nodes (ix and ix0 are equal-length mesh "
                "NODE index arrays, not DoF arrays: the nodes "
                "ix are eliminated and replaced by the paired "
                "nodes ix0) so the matching DOFs share the same "
                "column in the system matrix. (Audit "
                "2026-06-02; class-location corrected "
                "empirically 2026-08-03 on skfem 12.0.1.)"
            ),
            (
                "[Numerical] High Peclet: stabilise (SUPG or "
                "ElementDG upwinding) rather than refining. "
                "Signal: unstabilised CG develops oscillations "
                "upstream of a boundary or interior layer and "
                "undershoots there, and the useful statement is "
                "sharper than saying the oscillations do not "
                "damp — the "
                "undershoot DOES shrink under refinement, but it "
                "survives for as long as the CELL Peclet number "
                "exceeds about one, so clearing it by refinement "
                "alone costs more than an order of magnitude in "
                "DOFs. Test on a problem whose exact solution is "
                "bounded (a non-negative source with zero "
                "boundary data has a non-negative solution) and "
                "check for a negative minimum: it persists at "
                "every level of a refinement sweep whose cell "
                "Peclet stays above one, while SUPG with the "
                "standard coth-tau removes it at EVERY level "
                "including the coarsest. Report the cell Peclet "
                "alongside the excursion; the global Peclet "
                "alone does not predict either. (Verified "
                "2026-08-06 on skfem 12.0.1 — mechanism "
                "confirmed, the 'do not damp' wording "
                "sharpened.)"
            ),
        ],
    },
}

GENERATORS = {
    "convection_diffusion_2d": _convdiff_2d_skfem,
}
