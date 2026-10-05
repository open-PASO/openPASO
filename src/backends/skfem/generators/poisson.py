"""scikit-fem Poisson equation generators and knowledge."""
import math


def _poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Poisson -Δu = f on [0,1]² with Q1 quads, u=0 on boundary."""
    nx = params.get("nx", 32)
    f_val = params.get("f", 1.0)
    return f'''\
"""Poisson -Δu = {f_val} on [0,1]², Q1 quads, u=0 on boundary — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace, unit_load
import numpy as np
import json

# Mesh: structured quad mesh on [0,1]²
m = MeshQuad.init_tensor(np.linspace(0, 1, {nx + 1}), np.linspace(0, 1, {nx + 1}))
e = ElementQuad1()
ib = Basis(m, e)

# Assembly
K = laplace.assemble(ib)
f = ib.zeros()
f += {f_val} * unit_load.assemble(ib)

# Dirichlet BC: u=0 on all boundaries
D = ib.get_dofs().flatten()
u = solve(*condense(K, f, D=D))

max_val = u.max()
print(f"max(u) = {{max_val:.10f}}")
print(f"DOFs: {{K.shape[0]}}")
print(f"Elements: {{m.nelements}}")

# VTK output via meshio
import meshio
cells = [("quad", m.t.T)]
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])]) if m.p.shape[0] == 2 else m.p.T
mio = meshio.Mesh(points, cells, point_data={{"phi": u}})
mio.write("result.vtu")

summary = {{
    "max_value": float(max_val),
    "n_dofs": int(K.shape[0]),
    "n_elements": int(m.nelements),
    "element_type": "Q1 quad",
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Poisson solve complete.")
'''


def _poisson_2d_tri(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Poisson -Δu = f on [0,1]² with P1 triangles."""
    nx = params.get("nx", 32)
    f_val = params.get("f", 1.0)
    refine_level = max(int(math.log2(nx)), 3) if nx > 1 else 3
    return f'''\
"""Poisson -Δu = {f_val} on [0,1]², P1 triangles — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace, unit_load
import numpy as np
import json

m = MeshTri.init_symmetric().refined({refine_level})
e = ElementTriP1()
ib = Basis(m, e)

K = laplace.assemble(ib)
f = ib.zeros()
f += {f_val} * unit_load.assemble(ib)

D = ib.get_dofs().flatten()
u = solve(*condense(K, f, D=D))

max_val = u.max()
print(f"max(u) = {{max_val:.10f}}")

import meshio
# MeshTri.t.T has shape (n_cells, 3) — triangles, not quads.
# Declaring 'quad' here makes meshio reject the array with
# WriteError 'Unexpected cells array shape (n, 3) for quad
# cells. Expected shape [:, 4]'. Same class of typo fixed
# in _poisson_3d (hexahedron); this 2d_tri row was missed
# until the 2026-06-02 Layer-F coverage audit.
cells = [("triangle", m.t.T)]
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])]) if m.p.shape[0] == 2 else m.p.T
mio = meshio.Mesh(points, cells, point_data={{"phi": u}})
mio.write("result.vtu")

summary = {{"max_value": float(max_val), "n_dofs": int(K.shape[0]), "element_type": "P1 tri"}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


def _poisson_3d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Poisson on [0,1]³ with hex elements."""
    nx = params.get("nx", 8)
    f_val = params.get("f", 1.0)
    return f'''\
"""Poisson -Δu = {f_val} on [0,1]³, Hex1 — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace, unit_load
import numpy as np
import json

m = MeshHex.init_tensor(np.linspace(0,1,{nx+1}), np.linspace(0,1,{nx+1}), np.linspace(0,1,{nx+1}))
e = ElementHex1()
ib = Basis(m, e)

K = laplace.assemble(ib)
f = ib.zeros()
f += {f_val} * unit_load.assemble(ib)

D = ib.get_dofs().flatten()
u = solve(*condense(K, f, D=D))
print(f"3D Poisson max(u) = {{u.max():.10f}}")

import meshio
# MeshHex has 8-node hexahedra (m.t.T shape (n_cells, 8)).
# Declaring 'quad' here would make meshio reject the array
# with WriteError 'Unexpected cells array shape (n, 8) for
# quad cells. Expected shape [:, 4]'. The correct meshio
# cell type for a 3D Hex1 mesh is 'hexahedron'.
cells = [("hexahedron", m.t.T)]
points = m.p.T
mio = meshio.Mesh(points, cells, point_data={{"phi": u}})
mio.write("result.vtu")

summary = {{"max_value": float(u.max()), "n_dofs": int(K.shape[0])}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


KNOWLEDGE = {
    "poisson": {
        "description": "Poisson with scikit-fem (pure Python assembly)",
        "solver": (
            "scipy.sparse.linalg.spsolve (direct) or eigsh "
            "(eigenvalue). scikit-fem assembles; you choose "
            "the solver."
        ),
        "elements": (
            "ElementTriP1/P2/P3, ElementQuad1/2, ElementTetP1/P2, "
            "ElementHex1/2"
        ),
        "built_in_forms": (
            "laplace, unit_load (from skfem.models.poisson)"
        ),
        "pitfalls": [
            "[Syntax] When exporting a MeshHex (8-node hexahedron) "
            "solution via meshio, the cells tuple MUST declare "
            "'hexahedron' as the cell type, NOT 'quad'. Each row of "
            "m.t.T has 8 vertex indices for a Hex1 mesh; declaring "
            "'quad' makes meshio reject the array because quads "
            "expect 4 vertices. Signal: meshio._exceptions."
            "WriteError 'Unexpected cells array shape (n, 8) for "
            "quad cells. Expected shape [:, 4].' emitted from "
            "mio.write(...). Same pattern: MeshTet -> 'tetra', "
            "MeshLine -> 'line'. (Verified empirically 2026-06-01 "
            "— Layer F poisson_3d catch.)",
            "[API] scikit-fem is an ASSEMBLY library — laplace/"
            "unit_load build the K matrix and load vector, then "
            "you call solve(*condense(K, f, D=D)) yourself. "
            "There is no SolverInterface, no LinearProblem, no "
            "internal KSP wrapper. K is a scipy.sparse.csr_"
            "matrix; spsolve/factorized/scipy.sparse.linalg.cg "
            "are the canonical ways to handle it. Signal: "
            "reaching for a solver object fails at the namespace, "
            "e.g. AttributeError \"module 'skfem' has no "
            "attribute 'LinearProblem'\", and no SolverInterface "
            "or KSP name resolves either; K.shape and "
            "type(K).__name__ report visible scipy.sparse types, "
            "and spsolve, factorized and cg on the SAME condensed "
            "system agree, which is what makes swapping the "
            "solver a single-line user-level change. (Verified "
            "empirically 2026-08-06 on skfem 12.0.1 — previously "
            "carried as an inherited claim.)",
            "[API] basis.get_dofs(name) only works on a mesh "
            "where m.boundaries contains the name — and "
            "m.boundaries is None on a freshly constructed "
            "MeshTri / MeshQuad / MeshHex etc. Names must be "
            "registered up-front via "
            "m = m.with_boundaries({'left': lambda x: "
            "np.abs(x[0])<1e-10, ...}). Without that, "
            "basis.get_dofs('left') raises "
            "ValueError(\"Boundary 'left' not found.\"). "
            "Signal: m.boundaries is None on a fresh mesh; "
            "basis.get_dofs('left') raises the named "
            "ValueError; after with_boundaries({'left': f}) "
            "the call returns a DofsView whose flatten() length "
            "matches the number of edge nodes (5 for a 5x5 "
            "ElementQuad1 mesh's left edge). (Verified "
            "empirically 2026-06-01 — Tier-2 fixture "
            "poisson_get_dofs_named_boundary in scripts/"
            "tier2_fixtures/skfem/.)",
            "[API] VTU output in scikit-fem 12.x is via "
            "skfem.io.meshio.to_meshio(mesh, point_data=...) — "
            "the function is NOT removed in v12+; it lives in "
            "skfem.io.meshio and is not (and never was) "
            "re-exported at the TOP-LEVEL skfem namespace. "
            "(hasattr(skfem, 'to_meshio') is False; "
            "hasattr(skfem.io.meshio, 'to_meshio') is True.) "
            "skfem.io.meshio.to_meshio handles cell-type "
            "translation correctly (quad -> 'quad', etc.) and "
            "the resulting meshio.Mesh writes via "
            ".write('result.vtu'). Signal: import path "
            "skfem.io.meshio.to_meshio resolves; "
            "to_meshio(mesh).cells[0].type matches the source "
            "mesh element ('quad' for MeshQuad). (Catalog-drift "
            "correction verified empirically 2026-06-01 — same "
            "Tier-2 fixture as #1.)",
            "[API] Dirichlet elimination is handled by passing "
            "the constrained DOFs to condense(K, f, D=D); "
            "solve(*condense(K, f, D=D)) returns the full "
            "solution vector (with the eliminated DOFs already "
            "filled). The D argument expects a flat int array of "
            "DOF indices (basis.get_dofs(...).flatten()). For "
            "non-homogeneous BCs use condense(K, f, D=D, x=g) "
            "where g is the full-length BC vector. Signal: "
            "condense's call signature accepts D as positional or "
            "keyword. CORRECTION 2026-08-03: passing the "
            "DofsView object itself (instead of .flatten()) does "
            "NOT produce wrong results on skfem 12.0.1 — "
            "solve(*condense(K, f, D=basis.get_dofs())) returns "
            "a vector bit-identical to the .flatten() version "
            "(both max|u| = 7.0312e-02 on MeshTri().refined(2) "
            "with unit_load). .flatten() remains the documented "
            "and portable spelling, but the earlier warning "
            "about silently wrong results was stale and would "
            "have sent an "
            "agent hunting a non-existent bug. (Verified "
            "empirically 2026-08-03 on skfem 12.0.1.)",
            "[API] Element catalog by cell type: ElementTriP1/P2/"
            "P3 for triangles, ElementQuad1/Quad2 for quads, "
            "ElementTetP1/P2 for tetrahedra, ElementHex1/Hex2 "
            "for hexahedra. Plus ElementTriRT0 / ElementTriMini "
            "/ ElementTetMini / etc. for mixed methods. Signal: "
            "skfem.ElementTriP1, skfem.ElementQuad1, "
            "skfem.ElementHex1 are all attributes; "
            "type(ElementTriP1()).__bases__ is (ElementH1,), "
            "and ElementH1 subclasses skfem.element.Element (a "
            "plain base class, not an abc.ABC). (Catalog claim "
            "inherited; not separately "
            "Tier-2 falsified this iteration.)",
            "[API] skfem.quadrature.get_quadrature has a hard "
            "norder ceiling per reference domain: TRIANGLE order "
            "<= 19, TETRAHEDRON order <= 9. Tabulated rules stop "
            "there; orders above raise NotImplementedError. "
            "Triangle's message has a TYPO — 'The requested order "
            "of quadratureis not implemented!' (literal "
            "'quadratureis' with missing space) — tet's message "
            "is 'The requested order of quadrature is not "
            "available.' Real consequence: very-high-order tet "
            "elements that need quadrature 2k > 9 fail to "
            "assemble. Signal: get_quadrature(RefTri, 19) works "
            "and get_quadrature(RefTri, 20) raises with the "
            "typo'd message; get_quadrature(RefTet, 9) WORKS "
            "(45 points, weights summing to 1/6) and "
            "get_quadrature(RefTet, 10) raises with the "
            "correct-spacing message. The tet dispatch dict has "
            "keys 1..9 and the tri dict has keys 2..19. NOTE: "
            "get_quadrature's first argument is a Refdom/Element "
            "CLASS, not a string — passing 'triangle' raises "
            "NotImplementedError('The given reference domain type "
            "\\'triangle\\' is not supported!'), which is easy to "
            "mistake for the ceiling error. (File walk "
            "skfem/quadrature.py + live re-probe 2026-06-04; "
            "re-verified 2026-08-03 on skfem 12.0.1 — the "
            "tetrahedron ceiling was WRONG, it is 9 not 8.)",
            "[Numerical] Element order vs L2 convergence rate "
            "was measured on this install for ElementTriP1, P2 "
            "and P3 with a manufactured solution and homogeneous "
            "Dirichlet data, over four uniform refinements with "
            "intorder=2*deg+3, and each element reached its "
            "theoretical L2 order of deg+1. The measured error "
            "tables and rates are deliberately NOT reproduced "
            "here: they are the answer to a convergence study, "
            "and a study is only evidence if the numbers come "
            "from your own run. Choose your own manufactured "
            "field, derive f from it, and check the order you "
            "get against deg+1. Signal: the DEFAULT error "
            "quadrature is not the thing to watch — measuring "
            "the @Functional error integrand on a Basis built "
            "with the default intorder and with an explicit "
            "2*deg+3 gives the same orders for P1, P2 and P3 "
            "alike, so the default costs you nothing and a gate "
            "written around it never fires. What DOES destroy "
            "the answer is an explicitly too-low ASSEMBLY "
            "intorder: under-integrating the system itself (for "
            "instance a cubic space assembled at intorder=2) "
            "produces an enormous error and NEGATIVE rates, "
            "which is loud and unmistakable. So: set the "
            "assembly intorder deliberately, and if the measured "
            "order is short of deg+1, look at the assembly "
            "quadrature and the mesh sequence rather than at the "
            "error functional's own quadrature. (Verified "
            "empirically 2026-08-06 on skfem 12.0.1 — the "
            "'default intorder can cost you an apparent order' "
            "signal is falsified.)",
        ],
    },
}

GENERATORS = {
    "poisson_2d": _poisson_2d,
    "poisson_2d_tri": _poisson_2d_tri,
    "poisson_3d": _poisson_3d,
}
