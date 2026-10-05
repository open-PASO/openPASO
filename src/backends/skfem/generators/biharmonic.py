"""scikit-fem biharmonic / plate bending generators and knowledge."""


def _biharmonic_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Biharmonic equation (Kirchhoff plate bending) with Morley element."""
    refine_level = params.get("refine_level", 4)
    return f'''\
"""Biharmonic equation: Morley element — scikit-fem"""
from skfem import *
import numpy as np
import json

m = MeshTri.init_symmetric().refined({refine_level})
e = ElementTriMorley()
ib = Basis(m, e)

@BilinearForm
def biharmonic(u, v, w):
    # D^2 u : D^2 v (Hessian inner product)
    return (u.hess[0][0]*v.hess[0][0] + u.hess[1][1]*v.hess[1][1]
            + 2*u.hess[0][1]*v.hess[0][1])

@LinearForm
def load(v, w):
    return 1.0 * v

K = asm(biharmonic, ib)
f = asm(load, ib)

# BOUNDARY CONDITIONS ON A MORLEY ELEMENT: which dofs you constrain IS the
# boundary condition. Morley carries two kinds of dof — a value at each vertex
# ('u') and a normal derivative at each edge midpoint ('u_n') — so:
#
#   CLAMPED          u = 0 AND du/dn = 0  ->  d.flatten()      (both blocks)
#   SIMPLY SUPPORTED u = 0 only           ->  d.nodal['u']     (values only)
#
# Choose deliberately: the two are one call apart and neither errors. Under a
# unit load on this square, simple support gives a peak deflection about 3.4x
# the clamped one, so picking the wrong one is a wrong answer, not a wrong
# decimal.
#
# TRAP: .facet_ix is NOT the global normal-derivative dofs — it numbers them
# within the facet block, so it starts near zero while the real ones sit past
# the whole value block. Constraining it silently pins unrelated dofs, with no
# error. Address the blocks by name, d.nodal['u'] and d.facet['u_n'], which are
# global and say what they are. (Check on any mesh with
# sorted(d.facet_ix) == sorted(d.facet['u_n']) — it is False.)
d = ib.get_dofs()
D = d.flatten()                      # CLAMPED
u = solve(*condense(K, f, D=D))

print(f"Biharmonic: {{K.shape[0]}} DOFs, max(u)={{u.max():.6e}}")

summary = {{"n_dofs": K.shape[0], "max_value": float(u.max())}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Biharmonic solve complete.")
'''


KNOWLEDGE = {
    "biharmonic": {
        "description": "Biharmonic / plate bending (examples 02, 08, 20, 34)",
        "solver": "Direct (4th order system needs fine mesh)",
        "elements": "ElementTriMorley (nonconforming), ElementTriArgyris (C1), ElementQuadBFS (C1 quad)",
        "pitfalls": [
            "[API] ElementTriMorley is the nonconforming plate "
            "element — 6 DOFs per triangle (3 vertex values + "
            "3 edge-midpoint normal-derivative values). The "
            "normal-derivative DOF is on the EDGE, not a "
            "vertex, so adjacency-based queries that assume "
            "vertex-only DOFs miss it. Signal: skfem.Basis("
            "mesh, ElementTriMorley()).Nbfun == 6 regardless of "
            "mesh refinement (per-element count). (Verified "
            "empirically 2026-06-01.)",
            "[API] ElementTriArgyris is C^1 continuous with 21 "
            "DOFs per triangle (5th-degree polynomial). The DOFs "
            "include function values, first derivatives, and "
            "second derivatives at each vertex (6 per vertex × "
            "3 vertices = 18) plus 3 edge-midpoint normal "
            "derivatives → 21. Signal: skfem.Basis(mesh, "
            "ElementTriArgyris()).Nbfun == 21. (Verified "
            "empirically 2026-06-01.)",
            "[API] ElementQuadBFS (Bogner-Fox-Schmit) is C^1 on "
            "quads with 16 DOFs per element (function, du/dx, "
            "du/dy, and d^2u/dxdy at each of 4 vertices). "
            "Signal: skfem.Basis(mesh, ElementQuadBFS()).Nbfun "
            "== 16. (Verified empirically 2026-06-01.)",
            "[API] Euler-Bernoulli beam (skfem example 34): use "
            "ElementLineHermite on a MeshLine. 4 DOFs per "
            "element (deflection + slope at each endpoint), "
            "C^1 continuous. Signal: skfem.Basis(MeshLine(), "
            "ElementLineHermite()).Nbfun == 4, unchanged under "
            "refinement because it is a per-element count. "
            "(Verified empirically 2026-08-30, skfem 12.0.1.)",
        ],
    },
}

GENERATORS = {
    "biharmonic_2d": _biharmonic_2d,
}
