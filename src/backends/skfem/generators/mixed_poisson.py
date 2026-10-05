"""scikit-fem mixed Poisson generators and knowledge."""


def _mixed_poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Mixed Poisson with Raviart-Thomas + piecewise constant."""
    nx = params.get("nx", 16)
    refine_level = params.get("refine_level", 4)
    return f'''\
"""Mixed Poisson: RT0 + P0 — scikit-fem"""
from skfem import *
# skfem ships the standard differential helpers (grad, div,
# d/dn, etc.) under skfem.helpers — importing div() from
# this module is the supported way to take the divergence
# of a vector field inside a BilinearForm. The legacy
# attribute-access pattern sigma[0].grad[0] does NOT exist
# on the underlying numpy array (raises AttributeError
# 'numpy.ndarray' object has no attribute 'grad').
from skfem.helpers import div
import numpy as np
import json

m = MeshTri.init_symmetric().refined({refine_level})
e_rt = ElementTriRT0()
e_dg = ElementTriP0()

ib_rt = Basis(m, e_rt)
ib_dg = Basis(m, e_dg)

@BilinearForm
def mass_rt(sigma, tau, w):
    return sigma[0]*tau[0] + sigma[1]*tau[1]

@BilinearForm
def div_form(sigma, v, w):
    return div(sigma) * v

A = asm(mass_rt, ib_rt)
B = asm(div_form, ib_rt, ib_dg)

from scipy.sparse import bmat
K = bmat([[A, B.T], [B, None]], format='csr')
f = np.zeros(K.shape[0])
# Source in scalar part — LinearForm decorator wraps a
# callable that returns the integrand. The 'w' argument
# carries quadrature-point metadata (w.x, w.h, ...).
@LinearForm
def source(v, w):
    return 1.0 * v

f[A.shape[0]:] = -1.0 * asm(source, ib_dg)

u = np.linalg.lstsq(K.toarray(), f, rcond=None)[0]
print(f"Mixed Poisson: {{K.shape[0]}} DOFs")

summary = {{"n_dofs": K.shape[0], "n_elements": m.nelements}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Mixed Poisson solve complete.")
'''


KNOWLEDGE = {
    "mixed_poisson": {
        "description": "Mixed Poisson with Raviart-Thomas (example 37)",
        "solver": "Direct (saddle-point system) or iterative with Schur complement",
        "elements": "ElementTriRT0 (flux) + ElementTriP0 (scalar)",
        "pitfalls": [
            "[Numerical] Mixed Poisson assembles a SADDLE-POINT "
            "block system [[A, B^T], [B, 0]] where A = mass("
            "sigma, tau) (mass form on the flux space) and B = "
            "div(sigma) * v (divergence coupling against the "
            "scalar space). The full block matrix is INDEFINITE — "
            "direct solve via scipy.sparse.linalg.spsolve works "
            "for moderate sizes; iterative solvers need Schur "
            "complement preconditioning. Signal: scipy.sparse."
            "linalg.cg on the block matrix diverges immediately "
            "(indefinite system); spsolve succeeds. (Claim "
            "inherited; verified empirically 2026-08-30 on "
            "RT1/P0, refine(3): the 336x336 block has 128 "
            "negative and 208 positive eigenvalues, spsolve "
            "leaves residual 1.9e-14, and cg breaks down to a "
            "NaN residual at maxiter.)",
            "[API] skfem ships Raviart-Thomas elements under the "
            "abbreviated RTk naming: ElementTriRT0 (3 DOFs per "
            "triangle, one normal-flux DOF per edge), "
            "ElementTriRT1, ElementTetRT0. The long-form spelling "
            "spelling-out 'Raviart' and 'Thomas' as part of an "
            "Element class name does NOT exist; always use the "
            "RTk abbreviation. Signal: hasattr(skfem, "
            "'ElementTriRT0') is True; Basis(MeshTri(), "
            "ElementTriRT0()).Nbfun == 3 (matches the 3-edge "
            "count of a triangle); the long-form attribute lookup "
            "raises AttributeError. (Verified empirically "
            "2026-06-01.)",
            "[API] Use skfem.helpers.div(sigma) inside a "
            "BilinearForm to take the divergence of an RT vector "
            "field. The legacy element-wise pattern "
            "sigma[0].grad[0] + sigma[1].grad[1] does NOT work on "
            "the underlying numpy array — sigma[0] is just a "
            "scalar ndarray with no .grad attribute. div(sigma) is "
            "the supported helper and operates correctly on "
            "RT0/RT1/RT2 fields. Signal: AttributeError with the "
            "literal text \"'numpy.ndarray' object has no attribute "
            "'grad'\" emitted from a BilinearForm kernel that "
            "indexes sigma[0].grad[0]; the same form rewritten as "
            "div(sigma) * v assembles cleanly. (Verified "
            "empirically 2026-06-01 — Layer F catch.)",
            "[API] The @LinearForm decorator on a plain Python "
            "function is the canonical skfem pattern, but "
            "LinearForm(lambda v, w: ...) is EQUIVALENT, not a "
            "trap: the two assemble to the same vector, with and "
            "without extra asm() kwargs. Prefer the decorator "
            "for readability. "
            "Signal: there are no 'opaque shape errors deep "
            "inside skfem.assembly.form.linear_form' to catch — "
            "the lambda does not mis-resolve the kwargs adapter "
            "and nothing shape-related is raised. The one real "
            "requirement is ARITY: a form callable must take "
            "(v, w), and a one-argument lambda raises the "
            "ordinary Python TypeError '<lambda>() takes 1 "
            "positional argument but 2 were given' AT THE CALL "
            "SITE — a plain signature error, nothing deep. If "
            "you want to verify the equivalence, assemble both "
            "spellings and compare the vectors. (Verified "
            "2026-08-06 on skfem 12.0.1 — the "
            "'mis-resolves the kwargs adapter' claim is "
            "falsified.)",
            "[Numerical] THE MIXED FORMULATION INVERTS WHICH "
            "BOUNDARY CONDITION IS ESSENTIAL. In the primal form "
            "a prescribed value is essential and a prescribed "
            "flux is natural; in the mixed (flux-pressure) form "
            "it is the other way round. Prescribed FLUX "
            "sigma.n = g is ESSENTIAL — constrain the H(div) "
            "facet DOFs on that boundary. Prescribed VALUE "
            "u = u_D is NATURAL — it enters as the boundary "
            "integral <u_D, tau.n> against the FLUX test "
            "function, not against the scalar one. Signal: with "
            "the boundary integral carrying u_D and NO essential "
            "constraints at all, the pure-Dirichlet problem is "
            "solved exactly — on the unit square with RT1/P0 and "
            "u = x, max|sigma_x - (-1)| = 1.1e-15, 2.7e-15, "
            "1.4e-14 at refine 2, 3, 4. The trap: writing the "
            "boundary integral for NEUMANN data both injects a "
            "spurious Dirichlet contribution and leaves the flux "
            "condition unimposed, with no error raised. "
            "(Verified empirically 2026-08-30, skfem 12.0.1. "
            "This entry previously said the flux trace was the "
            "natural BC — inherited, and backwards.)",
            "[API] OrientedBoundary is NOT exposed at top level — "
            "skfem.OrientedBoundary raises AttributeError. Import "
            "it from the submodule: `from skfem.generic_utils import "
            "OrientedBoundary`. The class is a subclass of "
            "numpy.ndarray of facet indices carrying an `ori` "
            "attribute: an int array of 0/1 per facet that picks "
            "the row of mesh.f2t (which of the two adjacent "
            "elements) used for traces and normals. Build it with "
            "mesh.facets_around(...) or mesh.facets_satisfying("
            "..., normal=...), not by hand; a hand-made ±1 array "
            "indexes f2t row 1, which is -1 on exterior facets. "
            "FacetBasis branches on "
            "`isinstance(self.find, OrientedBoundary)` — a plain "
            "ndarray uses f2t row 0, which is sufficient on the "
            "exterior boundary but gives an arbitrary side on "
            "interior or subdomain facets. "
            "Signal: hasattr(skfem, 'OrientedBoundary') is False; "
            "the import succeeds from skfem.generic_utils. (File "
            "walk skfem/generic_utils.py 2026-06-02; verified live "
            "in skfem 12.0.1.)",
            "[API] Refdom.normals attribute is NOT unit-length on "
            "slanted reference-domain facets. Source: "
            "skfem/refdom.py defines RefTri.normals[1] = [1, 1] "
            "(norm = sqrt(2)) for the triangle's diagonal "
            "hypotenuse, RefTet.normals[3] = [1, 1, 1] (norm = "
            "sqrt(3)) for the tet's slanted face, RefWedge."
            "normals[1] = [1, 1, 0] (norm = sqrt(2)) for the wedge "
            "diagonal. RefLine, RefQuad, RefHex have all-unit "
            "normals as expected. Users who compute boundary flux "
            "integrals or normal-traction loads by indexing these "
            "arrays directly get an O(1) scale error on diagonal "
            "facets. Signal: numpy.linalg.norm(RefTri.normals, "
            "axis=1) returns [1.0, 1.414, 1.0] — assert that fails "
            "if you assumed all unit. Workaround: always normalise "
            "before use (n_unit = n / np.linalg.norm(n)) OR use "
            "the FacetBasis attribute, which IS unit length — "
            "and note the spelling is `FacetBasis.normals`, "
            "plural; there is no `FacetBasis.normal`, so code "
            "written from the singular name fails with an "
            "AttributeError that looks unrelated to the "
            "reference-domain issue. Also: "
            "RefWedge.brefdom is None (uniquely among the 3D refdoms) "
            "so FacetBasis on wedges is unsupported — RefTri/RefTet/"
            "RefHex/RefQuad/RefLine all have proper brefdoms. (File "
            "walk skfem/refdom.py 2026-06-02; verified live in "
            "skfem 12.0.1.)",
        ],
    },
}

GENERATORS = {
    "mixed_poisson_2d": _mixed_poisson_2d,
}
