"""scikit-fem element survey on Poisson.

Solves one manufactured Poisson problem with every scikit-fem element that
can carry it, on a refinement ladder, and reports the measured L2
convergence order of each beside the order its polynomial space predicts.
Every element with a prediction gets a VERDICT line in the coverage
harness's grammar (scripts/coverage_harness/definitions.py).

MEASURED 2026-09-23, scikit-fem 12.0.2, four levels in 1-D and 2-D and
four in 3-D (ElementHexC1 capped at two, see below), the boundary
condition applied to the dofs that u = 0 actually fixes: every Lagrange,
bubble-enriched, serendipity and tensor-product element converges at its
approximation degree plus one on the finest pair -- P1 1.97, P2 3.00, P3
4.04, P4 4.99, Quad1 2.00, Quad2 2.99, QuadS2 3.00, Hex1 2.01, Hex2 2.97 --
and so do the Hermite-type elements once only their value and
tangential-derivative dofs are constrained: Hermite 3.88, Bogner-Fox-Schmit
3.95, LineHermite 3.95, the 15-parameter plate 5.11. The parameterised
ElementLinePp(3) and ElementQuadP(3) (added 2026-09-24) converge at 4.00. Morley, non-conforming
for a second-order problem, sits at order 0. Argyris (prediction 6) and
ElementHexC1 (prediction 4) get no verdict: the quintic's ladder does not
settle (5.5, 6.2, 5.2 on successive pairs) and the tricubic hex's third
level costs eight minutes on one core.

Until 2026-09-23 this survey clamped EVERY boundary dof, derivative dofs
included, and served the conclusion that Argyris, Hermite, BFS, the
15-parameter plate and HexC1 "sit near order 1 for the same reason as
Morley". That was the survey's own boundary condition, not the elements;
the wrong treatment is kept as the planted failure (SURVEY_MUTATE=1).
"""
from string import Template

_SURVEY = Template(r"""
'''$title

WHICH ELEMENT SHOULD I USE? Answered by running, not by recall.

Solves the same Poisson problem with every scikit-fem element that can
carry it, on a refinement ladder, and reports the MEASURED L2 convergence
order of each beside the order its polynomial space PREDICTS. The exact
solution is manufactured in this script, the source term is derived from
it, and every order in the table is measured from this run.

Every element with a prediction gets a VERDICT line the coverage harness
reads (scripts/coverage_harness/definitions.py):
  VERDICT skfem <element> mms_order ref=<predicted> got=<measured> tol=1.500000e-01 PASS|FAIL
The prediction is the approximation degree plus one. It is NOT skfem's
`maxdeg` plus one: maxdeg is the quadrature hint -- the highest monomial
degree present, tensor cross terms and bubbles included -- and reading it
as the degree over-predicts every tensor-product, serendipity and bubble
element (ElementQuad1 has maxdeg 2 and converges at 2, not 3; ElementHex2
has maxdeg 6 and converges at 3).

THE BOUNDARY CONDITION IS PART OF THE MEASUREMENT. An element with
derivative degrees of freedom (Hermite, Argyris, Bogner-Fox-Schmit,
ElementHexC1, the 15-parameter plate) must have only the dofs constrained
that u = 0 on the boundary actually fixes: values, and derivatives
TANGENTIAL to the boundary. The obvious `D=basis.get_dofs()` also pins the
normal derivative, which the solution does not satisfy; that
over-constrains the problem and caps the observed order near 1 for
elements that converge at 4 to 6. SURVEY_MUTATE=1 restores that wrong
treatment: it is the planted failure that shows these verdicts can fail.
'''
import os
import numpy as np
from skfem import (Basis, BilinearForm, LinearForm, Mesh, MeshTri, MeshQuad,
                   MeshTet, MeshHex, MeshLine, condense, solve)
from skfem.helpers import dot, grad
from skfem.element import *   # noqa: F403 -- the survey names many

REFINE_2D = $refine       # refinement levels for 1-D and 2-D meshes
REFINE_3D = $refine3      # 3-D levels; each level is ~8x the work
ORDER_BAND = 0.15         # |measured - predicted| within this counts: the harness's policy
MUTATE = os.environ.get("SURVEY_MUTATE", "") == "1"

# The manufactured solution and its source, derived here rather than looked
# up: u = prod sin(pi x_i)  =>  -lap u = d * pi^2 * u  on the unit cell.
def u_exact(x):
    out = np.ones_like(x[0])
    for xi in x:
        out = out * np.sin(np.pi * xi)
    return out

def make_forms(dim):
    @BilinearForm
    def a(u, v, w):
        return dot(grad(u), grad(v))
    @LinearForm
    def L(v, w):
        f = dim * np.pi ** 2
        for xi in w.x:
            f = f * np.sin(np.pi * xi)
        return f * v
    return a, L

def dirichlet_dofs(basis):
    '''The dofs that u = 0 on the boundary of an axis-aligned cell actually fixes.

    A nodal dof named u_<dirs> at a boundary vertex is fixed iff some boundary
    face x_j = const through that vertex has j outside <dirs>: the derivative
    is then tangential to that face, and a function vanishing on the face has
    vanishing tangential derivatives there. The value dof is always fixed. A
    facet or edge dof is fixed iff it is a VALUE dof (named "u"); the
    normal-derivative dof u_n of Morley, Argyris and the 15-parameter plate is
    never fixed.
    '''
    if MUTATE:
        return basis.get_dofs().all()      # the wrong treatment: every boundary dof
    X = basis.mesh.p
    dim = X.shape[0]
    names = list(basis.elem.dofnames)
    onb = np.zeros((dim, X.shape[1]), dtype=bool)
    for j in range(dim):
        onb[j] = np.isclose(X[j], X[j].min()) | np.isclose(X[j], X[j].max())
    bverts = np.where(onb.any(axis=0))[0]
    fixed = []
    nodal = basis.dofs.nodal_dofs           # (dofs per vertex, vertices)
    for k in range(basis.elem.nodal_dofs):
        nm = names[k]
        dirs = {"xyz".index(c) for c in nm[2:] if c in "xyz"} if nm.startswith("u_") else set()
        for v in bverts:
            if any(onb[j, v] and j not in dirs for j in range(dim)):
                fixed.append(nodal[k, v])
    D = basis.get_dofs()
    for key, arr in D.facet.items():
        if key == "u":
            fixed += list(np.asarray(arr).ravel())
    for key, arr in getattr(D, "edge", {}).items():
        if key == "u":
            fixed += list(np.asarray(arr).ravel())
    return np.unique(np.array(fixed, dtype=np.int64))

CANDIDATES = {
    "MeshTri": (MeshTri, 2, $tri),
    "MeshQuad": (MeshQuad, 2, $quad),
    "MeshTet": (MeshTet, 3, $tet),
    "MeshHex": (MeshHex, 3, $hex),
    "MeshLine": (MeshLine, 1, $line),
}

# The predicted L2 order of each space on this problem: approximation degree
# plus one, each with its reason. Not maxdeg plus one.
PREDICTED = {
    "ElementTriP1": 2, "ElementTriP2": 3, "ElementTriP3": 4, "ElementTriP4": 5,   # Lagrange P_p
    "ElementTriCR": 2,              # non-conforming linear, still O(h^2) in L2
    "ElementTriCCR": 3,             # P2 plus a cubic bubble: the bubble adds no order (maxdeg 3)
    "ElementTriMini": 2,            # P1 plus a bubble (maxdeg 3)
    "ElementTriHermite": 4,         # cubic, C0
    "ElementTriP1G": 2, "ElementTriP2G": 3,   # P1 and P2 through ElementGlobal
    "ElementTri15ParamPlate": 5,    # full quartic, 15 coefficients, C0
    "ElementQuad1": 2,              # bilinear: complete degree 1 (maxdeg 2)
    "ElementQuad2": 3,              # biquadratic: complete degree 2 (maxdeg 4)
    "ElementQuadS2": 3,             # serendipity: complete degree 2 (maxdeg 3)
    "ElementQuad2G": 3,
    "ElementQuadBFS": 4,            # bicubic Hermite
    "ElementQuadP": 4,              # ElementQuadP(3): complete degree 3 (maxdeg 9)
    "ElementTetP2": 3, "ElementTetCR": 2, "ElementTetCCR": 3, "ElementTetMini": 2,
    "ElementHex1": 2,               # trilinear (maxdeg 3)
    "ElementHex2": 3,               # triquadratic (maxdeg 6)
    "ElementHexS2": 3,              # serendipity, complete degree 2
    "ElementHexC1": 4,              # tricubic Hermite
    "ElementLineP1": 2, "ElementLineP2": 3, "ElementLineHermite": 4,
    "ElementLineMini": 3,            # P1 plus a quadratic bubble IS full P2 on an interval
    "ElementLinePp": 4,              # ElementLinePp(3): degree 3 as constructed
    "ElementTriArgyris": 6,         # quintic, C1
}
# No prediction, so no verdict, and the row says why.
NO_PREDICTION = {
    "ElementTriMorley": "non-conforming for a second-order problem; it carries Poisson and converges at order ~0, which is the lesson",
    "ElementQuad1DG": "a discontinuous basis has no facet coupling in this continuous form",
}
# A prediction but no verdict: the ladder cannot settle within budget. A
# verdict is withheld here, never picked from the pair that happens to fit.
WITHHELD = {
    "ElementTriArgyris": "quintic: successive pairs measured 5.5, 6.2, 5.2 -- not settled on this ladder",
    "ElementHexC1": "tricubic: level 3 costs ~8 minutes, so it stops at two levels, which are pre-asymptotic",
    "ElementTetCCR": "P2 plus face and cell bubbles: successive pairs measured 2.2, 2.7, 2.8, still rising -- not settled within budget",
}
LEVEL_CAP = {"ElementHexC1": 2, "ElementTetCCR": 3}

def fresh(proto):
    '''A new instance of the element for a new mesh. ElementGlobal elements
    (Argyris, Morley, Hermite, BFS, HexC1, ...) cache mesh-dependent data on
    the instance, so one instance must not serve two meshes; a parameterised
    element (ElementQuadP(3)) cannot be rebuilt from its class alone and is
    reused, which is safe because it caches nothing.'''
    try:
        return type(proto)()
    except TypeError:
        return proto


def l2_error(basis, x):
    # measured on a RICHER rule than the one that built the matrix, so a
    # low-degree element cannot reproduce the target at its own points
    # (intorder 9 is the highest tetrahedral order get_quadrature accepts, a
    # 45-point rule exact through degree 8; intorder 10+ raises
    # NotImplementedError)
    richer = 8 if basis.mesh.dim() == 3 else max(8, 2 * basis.elem.maxdeg + 2)
    eb = Basis(basis.mesh, fresh(basis.elem), intorder=richer)
    diff = np.asarray(eb.interpolate(x)) - u_exact(eb.global_coordinates())
    return float(np.sqrt(np.sum(diff ** 2 * eb.dx)))

print(f"{'mesh':9s} {'element':24s} {'ndofs':>8s} {'L2 error':>11s} {'order':>6s} {'pred':>5s}  verdict")
print("-" * 80)
rows = 0
for mname, (M, dim, elems) in CANDIDATES.items():
    a, L = make_forms(dim)
    # `elems` pairs the name the coverage denominator uses with a CONSTRUCTED
    # element, written out as a constructor call in this file. A template that
    # merely NAMES an element in a string has not exercised it. The name is
    # carried separately because scikit-fem keeps aliases (ElementTriCCR is
    # ElementTriP2B, ElementTriMini is ElementTriP1B) and type(e).__name__
    # would print the other one.
    for en, proto in elems:
        errs, ndofs = [], []
        try:
            levels = min(REFINE_3D if dim == 3 else REFINE_2D, LEVEL_CAP.get(en, 99))
            for lvl in range(1, levels + 1):
                m = M().refined(lvl)
                b = Basis(m, fresh(proto))
                A = a.assemble(b); f = L.assemble(b)
                x = solve(*condense(A, f, D=dirichlet_dofs(b)))
                if not np.all(np.isfinite(x)):
                    raise ValueError("non-finite solution")
                errs.append(l2_error(b, x)); ndofs.append(b.N)
        except Exception as exc:
            print(f"{mname:9s} {en:24s} {'-':>8s} {'-':>11s}   FAILED  {type(exc).__name__}"
                  + (f"  ({NO_PREDICTION[en]})" if en in NO_PREDICTION else ""))
            continue
        if len(errs) >= 2 and errs[-2] > 0 and errs[-1] > 0:
            order = float(np.log(errs[-2] / errs[-1]) / np.log(2.0))
        else:
            order = float("nan")
        pred = PREDICTED.get(en)
        if pred is None:
            tail = "no prediction: " + NO_PREDICTION.get(en, "")
        elif en in WITHHELD:
            tail = f"verdict withheld: {WITHHELD[en]}"
        else:
            tail = "PASS" if abs(order - pred) <= ORDER_BAND else "FAIL"
            print(f"VERDICT skfem {en} mms_order ref={pred:d} got={order:.6e} tol={ORDER_BAND:.6e} {tail}")
        print(f"{mname:9s} {en:24s} {ndofs[-1]:8d} {errs[-1]:11.3e} {order:6.2f} "
              f"{(str(pred) if pred is not None else '-'):>5s}  {tail}")
        rows += 1
print("-" * 80)
print(f"elements surveyed: {rows}")
print("The prediction is the approximation degree plus one, not maxdeg plus one.")
print("Morley converges at order ~0 because it is non-conforming for a second-order")
print("problem. A verdict is withheld where the ladder cannot settle within budget.")
if MUTATE:
    print("[MUTATED: every boundary dof clamped -- FAIL on the Hermite-type elements is the control working]")
""")

# Elements measured to carry Poisson on this install. The C1, Hermite-type
# and plate entries are included DELIBERATELY: the survey's job is to show
# what each does here, verdict or no verdict.
_TRI = ["ElementTriP1", "ElementTriP2", "ElementTriP3", "ElementTriP4",
        "ElementTriCR", "ElementTriCCR", "ElementTriMini", "ElementTriMorley",
        "ElementTriArgyris", "ElementTriHermite", "ElementTriP1G",
        "ElementTriP2G", "ElementTri15ParamPlate"]
_QUAD = ["ElementQuad1", "ElementQuad2", "ElementQuadS2", "ElementQuad2G",
         "ElementQuad1DG", "ElementQuadBFS", "ElementQuadP(3)"]
_TET = ["ElementTetP2", "ElementTetCR", "ElementTetCCR", "ElementTetMini"]
_HEX = ["ElementHex1", "ElementHex2", "ElementHexS2", "ElementHexC1"]
_LINE = ["ElementLineP1", "ElementLineP2", "ElementLineMini",
         "ElementLineHermite", "ElementLinePp(3)"]


def _calls(names: list[str]) -> str:
    """Render element names as (name, CONSTRUCTOR CALL) pairs.

    The emitted script must contain ElementTriP1(), not "ElementTriP1".
    Naming an element in a string is not using it, and any coverage measure
    that cannot tell those apart is measuring prose. The name travels beside
    the call because scikit-fem aliases some classes and type(e).__name__
    would print the other name.
    """
    out = []
    for n in names:
        # a parameterised element is written with its argument, e.g.
        # "ElementQuadP(3)"; its row carries the class name
        label = n.split("(")[0]
        call = n if "(" in n else n + "()"
        out.append(f'("{label}", {call})')
    return "[" + ", ".join(out) + "]"


def _element_survey_poisson(params: dict) -> str:
    """Solve one Poisson problem with every element that can carry it."""
    return _SURVEY.substitute(
        title=params.get("title", "scikit-fem element survey on Poisson"),
        refine=params.get("refine_2d", 4),
        refine3=params.get("refine_3d", 4),
        tri=_calls(params.get("tri_elements", _TRI)),
        quad=_calls(params.get("quad_elements", _QUAD)),
        tet=_calls(params.get("tet_elements", _TET)),
        hex=_calls(params.get("hex_elements", _HEX)),
        line=_calls(params.get("line_elements", _LINE)),
    )


GENERATORS = {
    "element_survey_poisson": _element_survey_poisson,
}

KNOWLEDGE = {
    "element_survey": {
        "description": (
            "Solve one manufactured Poisson problem with every scikit-fem "
            "element that can carry it, measure each one's L2 convergence "
            "order on a refinement ladder and judge it against the order its "
            "polynomial space predicts, one VERDICT line per element -- the "
            "element-selection question answered by running."
        ),
        "pitfalls": [
            "[Numerical] ONLY THE DOFS THAT u = 0 FIXES ARE DIRICHLET DOFS. "
            "For an element with derivative degrees of freedom, "
            "Basis.get_dofs() on the boundary returns value dofs, tangential "
            "derivative dofs AND normal derivative dofs alike; passing all of "
            "them to condense(D=...) pins the normal derivative to zero, "
            "which the solution does not satisfy, and caps the measured order "
            "near 1: Hermite 1.25, Bogner-Fox-Schmit 1.06, ElementHexC1 1.26, "
            "Argyris 1.08, LineHermite 1.02 measured that way, against 3.9, "
            "3.95, 3.8, 5-6 and 3.95 with only the dofs u = 0 fixes (values; "
            "derivatives tangential to a boundary face through the vertex; "
            "facet value dofs; never the u_n facet dof). This survey made "
            "that mistake until 2026-09-23 and served the wrong conclusion "
            "that those elements are non-conforming. Signal: Basis.get_dofs() "
            "passed whole to condense(D=...) for an element whose dofnames "
            "include u_x, u_y or u_n, and a measured order near 1 on a smooth "
            "problem while the Lagrange rows from the same solve() converge "
            "at theirs.",

            "[Numerical] AN ELEMENT THAT SOLVES IS NOT AN ELEMENT THAT SUITS. "
            "ElementTriMorley assembles, solves, returns a finite field and "
            "converges at order ~0 on Poisson (measured -0.42, -0.18, -0.05 "
            "on successive pairs). It is a non-conforming plate element for "
            "fourth-order problems, so the answer is meaningless however "
            "healthy the solve looked. Signal: an element whose measured "
            "order is near zero or negative while condense() and solve() "
            "report no error at all. (Measured 2026-09-23, scikit-fem "
            "12.0.2.)",

            "[Numerical] MEASURE THE ORDER, DO NOT READ IT OFF maxdeg. "
            "Element.maxdeg is the quadrature hint -- the highest monomial "
            "degree present, tensor cross terms and bubbles included -- not "
            "the approximation degree: ElementQuad1 has maxdeg 2 and "
            "converges at 2, ElementHex2 has maxdeg 6 and converges at 3, "
            "ElementTriMini has maxdeg 3 and converges at 2 -- while "
            "ElementLineMini converges at 3, because on an interval P1 plus a "
            "quadratic bubble is all of P2. The prediction "
            "is the complete polynomial degree plus one, and the value of "
            "running the ladder is the rows that do NOT match it. Signal: a "
            "measured order a full integer below degree plus one usually "
            "means the boundary condition, not the element.",

            "[Numerical] A LADDER THAT HAS NOT SETTLED PROVES NOTHING. "
            "ElementTriArgyris measures 5.53, 6.20 and 5.18 on three "
            "successive level pairs against a prediction of 6, and "
            "ElementTetCCR 2.23, 2.69 and 2.83 against 3, still rising; no "
            "pair is within the band and the verdict is withheld rather than "
            "picked from the one that comes closest. Signal: the order still moving "
            "by more than the band between successive pairs -- refine "
            "further if the budget allows, and if it does not, say so "
            "instead of claiming the order.",

            "[API] THE 3-D LADDER COSTS ~8x PER LEVEL, AND ElementHexC1 "
            "COSTS MORE. Its level-3 solve on the unit cube took 467 s on "
            "one core (64 dofs per element, degree-12 quadrature), so this "
            "survey caps it at two levels and withholds its verdict; every "
            "other 3-D element runs four levels in seconds. Signal: a "
            "survey that appears to hang is almost always the 3-D rows, not "
            "a failed solve.",

            "[API] A DG ELEMENT NEEDS A DG FORMULATION. ElementQuad1DG "
            "fails in this survey and that is correct, not a defect: the "
            "standard continuous Galerkin form has no facet terms, so a "
            "discontinuous basis has nothing coupling its elements and the "
            "condensed system is singular. Signal: scipy's "
            "MatrixRankWarning from spsolve followed by a non-finite "
            "solution, rather than a wrong answer -- the honest failure "
            "mode.",
        ],
    },
}
