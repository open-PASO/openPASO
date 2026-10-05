"""scikit-fem mixed and vector element survey.

The element families a continuous-Galerkin Poisson survey CANNOT reach:
H(div) through a mixed Poisson saddle point, H(curl) through the de Rham
identity, and the L2/discontinuous spaces through projection. Each is
checked against something independent of the run and reported as a
VERDICT line in the coverage harness's grammar
(scripts/coverage_harness/definitions.py).

MEASURED 2026-09-23, scikit-fem 12.0.2. H(div): the L2 error of the
computed flux against the exact gradient of the manufactured solution
converges at the space's order -- ElementTriRT0 1.00, ElementTriBDM1 1.98,
ElementTriRT2 with a P1 DG pressure 1.99, ElementQuadRT0 1.01,
ElementTetRT0 0.97, ElementHexRT1 0.99. H(curl): every Nedelec space holds
the gradients of the Lagrange space of its order to 1e-15 and the assembled
curl-curl operator annihilates them to 1e-15. L2/DG projections give order
p+1: P0 0.96-0.99, P1DG and TetP1 and Hex1DG 1.98-2.01.

TWO CLAIMS THIS SURVEY USED TO MAKE WERE WRONG, and the verdicts forced
the correction. (1) "ElementTriRT2, ElementTriBDM1 and ElementTetRT0 raise
ValueError in this formulation": the ValueError ('Quadrature mismatch:
trial and test functions should have same number of integration points')
was the survey's own doing -- the pressure basis was built on its own
default rule instead of the flux basis's -- and all three converge once
the bases share a quadrature. (2) The divergence residual |B sigma - f| /
|f| was the check: it is the saddle point's own second equation and holds
at machine precision for an INCOMPATIBLE pairing too. The second-order
triangle flux paired with a piecewise-constant pressure keeps a residual of
4e-15 while its flux error does not converge at all (order 0.00); the
survey prints that pairing as a control. A self-consistency check is not a
verification.

THE SKELETON, HHJ AND WEDGE SPACES (added 2026-09-24) carry no Poisson
problem: the four skeleton elements live on the mesh facets and are
projected through FacetBasis + InteriorFacetBasis, the two HHJ elements are
symmetric matrices with normal-normal continuity and reproduce the identity
matrix, and ElementWedge1 reproduces a constant on MeshWedge1, whose
refined() is not implemented, so it gets an exact identity and no order.

NAMES. scikit-fem keeps old names as aliases of the same class:
ElementTriRT0 is ElementTriRT1, ElementQuadRT0 is ElementQuadRT1,
ElementTetRT0 is ElementTetRT1, ElementTetN0 is ElementTetN1. This survey
has one row per CLASS, under the name the coverage denominator uses
(declared_surface() dedupes by class object and keeps the first name).
"""
from string import Template


# Written as (name, CONSTRUCTOR CALL, ...) tuples, not quoted names alone: a
# template that merely names an element has not exercised it, and a coverage
# measure that credits a string is measuring prose. The name travels beside
# the call because of the aliases above: type(e).__name__ prints the other one.
_HDIV = ('[("MeshTri", MeshTri, 2, [("ElementTriRT0", ElementTriRT0(), ElementTriP0(), 1), '
         '("ElementTriRT2", ElementTriRT2(), ElementTriP1DG(), 2), '
         '("ElementTriBDM1", ElementTriBDM1(), ElementTriP0(), 2)]), '
         '("MeshQuad", MeshQuad, 2, [("ElementQuadRT0", ElementQuadRT0(), ElementQuad0(), 1)]), '
         '("MeshTet", MeshTet, 3, [("ElementTetRT0", ElementTetRT0(), ElementTetP0(), 1)]), '
         '("MeshHex", MeshHex, 3, [("ElementHexRT1", ElementHexRT1(), ElementHex0(), 1)])]')
_HCURL = ('[("MeshTri", MeshTri, 2, [("ElementTriN1", ElementTriN1(), 1), '
          '("ElementTriN2", ElementTriN2(), 2), ("ElementTriN3", ElementTriN3(), 3)]), '
          '("MeshQuad", MeshQuad, 2, [("ElementQuadN1", ElementQuadN1(), 1)]), '
          '("MeshTet", MeshTet, 3, [("ElementTetN0", ElementTetN0(), 1)])]')
_SKELETON = ('[("MeshTri", MeshTri, [("ElementTriSkeletonP0", ElementTriSkeletonP0()), '
             '("ElementTriSkeletonP1", ElementTriSkeletonP1())]), '
             '("MeshTet", MeshTet, [("ElementTetSkeletonP0", ElementTetSkeletonP0())]), '
             '("MeshHex", MeshHex, [("ElementHexSkeleton0", ElementHexSkeleton0())])]')
_HHJ = '[("ElementTriHHJ0", ElementTriHHJ0()), ("ElementTriHHJ1", ElementTriHHJ1())]'
_L2 = ('[("MeshTri", MeshTri, 2, [("ElementTriP0", ElementTriP0(), 1), ("ElementTriP1DG", ElementTriP1DG(), 2)]), '
       '("MeshQuad", MeshQuad, 2, [("ElementQuad0", ElementQuad0(), 1), ("ElementQuad1DG", ElementQuad1DG(), 2)]), '
       '("MeshTet", MeshTet, 3, [("ElementTetP0", ElementTetP0(), 1), ("ElementTetP1", ElementTetP1(), 2)]), '
       '("MeshHex", MeshHex, 3, [("ElementHex0", ElementHex0(), 1), ("ElementHex1DG", ElementHex1DG(), 2)]), '
       '("MeshLine", MeshLine, 1, [("ElementLineP0", ElementLineP0(), 1), ("ElementLineP1DG", ElementLineP1DG(), 2)])]')


_MIXED = Template(r"""
'''$title

THE ELEMENT FAMILIES A POISSON SURVEY CANNOT REACH.

A continuous Galerkin Poisson problem only exercises H1-conforming
elements. The three families below need their own formulations, and each
is checked against something independent of the run:

  H(div), mixed Poisson   -- the manufactured solution u = prod sin(pi x_i)
                             solved in mixed form (sigma = grad u); the L2
                             error of the computed FLUX against the exact
                             gradient is measured on two refinement levels
                             and its order judged against the space's
                             prediction. The discrete divergence residual
                             is printed too, but it is NOT the verification:
                             it is the saddle point's own second equation
                             and holds at machine precision for an
                             INCOMPATIBLE pairing as well. The control row
                             shows the second-order triangle flux with a
                             piecewise-constant pressure at residual ~1e-15
                             and flux order 0.
  H(curl), the de Rham identity -- the gradients of the Lagrange space of
                             degree k lie in the Nedelec space of order k,
                             and the assembled curl-curl operator
                             annihilates them: two exact identities per
                             space, measured as the L2 distance of the
                             projected gradient from the exact one and as
                             |K u| / (max|K| |u|).
  L2 and DG               -- project a known function and measure the L2
                             error under refinement on a RICHER rule than
                             the one that defined the projection. A
                             degree-p discontinuous space gives order p+1.

VERDICT lines:
  VERDICT skfem <element> mms_order ref=<predicted> got=<measured> tol=1.500000e-01 PASS|FAIL
  VERDICT skfem <element> exact_identity ref=0.000000e+00 got=<measured> tol=1.000000e-10 PASS|FAIL
Planted failures: SURVEY_MUTATE=1 pairs every H(div) space with a
piecewise-constant pressure (wrong for the second-order space);
SURVEY_MUTATE=2 asks every Nedelec space to hold the gradients of P2 (wrong
for the lowest-order ones).

NAMES: scikit-fem keeps old names as aliases of the same class --
ElementTriRT0 is ElementTriRT1, ElementQuadRT0 is ElementQuadRT1,
ElementTetRT0 is ElementTetRT1, ElementTetN0 is ElementTetN1. One row per
CLASS here, under the name the coverage denominator uses.
'''
import os
import numpy as np
import scipy.sparse as sp
from skfem import (Basis, BilinearForm, LinearForm, FacetBasis,
                   InteriorFacetBasis, MeshTri, MeshQuad, MeshTet, MeshHex,
                   MeshLine, MeshWedge1, condense, solve)
from skfem.helpers import dot, div, curl, ddot
from skfem.element import *   # noqa: F403 -- the survey names many

REFINE = $refine
ERROR_INTORDER = $intorder
ORDER_BAND = 0.15        # the harness's band for a measured order
EXACT_TOL = 1e-10        # the harness's ceiling for an exact identity
MUTATE = os.environ.get("SURVEY_MUTATE", "")


def verdict(entry, kind, ref, got, tol):
    ok = bool(abs(got - ref) <= tol)
    ref_s = f"{ref:d}" if isinstance(ref, int) else f"{ref:.6e}"
    print(f"VERDICT skfem {entry} {kind} ref={ref_s} got={got:.6e} tol={tol:.6e} "
          + ("PASS" if ok else "FAIL"))
    return ok


# The manufactured solution and its gradient, derived here.
def _u(x):
    out = np.ones_like(x[0])
    for xi in x:
        out = out * np.sin(np.pi * xi)
    return out

def _grad_u(x):
    comps = []
    for i in range(len(x)):
        c = np.pi * np.cos(np.pi * x[i])
        for j in range(len(x)):
            if j != i:
                c = c * np.sin(np.pi * x[j])
        comps.append(c)
    return np.array(comps)


@BilinearForm
def _flux_mass(sigma, tau, w):
    return dot(sigma, tau)


@BilinearForm
def _div_coupling(sigma, v, w):
    return div(sigma) * v


def _lap_load(dim):
    @LinearForm
    def L(v, w):
        return -dim * np.pi ** 2 * _u(w.x) * v     # (div sigma, v) = (lap u, v)
    return L


def mixed_flux_error(M, dim, fcls, pcls, lvl):
    '''Solve the mixed Poisson problem; return (flux L2 error, div residual, flux ndofs).'''
    m = M().refined(lvl)
    fb = Basis(m, fcls())
    # THE PRESSURE BASIS MUST SHARE THE FLUX BASIS'S QUADRATURE. Built on its
    # own default rule it raises ValueError('Quadrature mismatch ...') in the
    # coupling form -- which this survey used to report as three elements
    # that "raise ValueError in this formulation".
    pb = Basis(m, pcls(), quadrature=fb.quadrature)
    A = _flux_mass.assemble(fb)
    B = _div_coupling.assemble(fb, pb)
    F = _lap_load(dim).assemble(pb)
    K = sp.bmat([[A, B.T], [B, None]], "csr")
    x = solve(K, np.concatenate([fb.zeros(), F]))
    sig = x[:A.shape[0]]
    resid = float(np.linalg.norm(B @ sig - F) / max(np.linalg.norm(F), 1e-300))
    eb = Basis(m, fcls(), intorder=ERROR_INTORDER)
    d = np.asarray(eb.interpolate(sig)) - _grad_u(eb.global_coordinates())
    return float(np.sqrt(np.sum(d ** 2 * eb.dx))), resid, fb.N


P0_OF = {"MeshTri": ElementTriP0, "MeshQuad": ElementQuad0,
         "MeshTet": ElementTetP0, "MeshHex": ElementHex0}

print("H(div): mixed Poisson -- L2 error of the flux against the exact gradient, order vs prediction")
print(f"  {'mesh':9s} {'flux':15s} {'pressure':15s} {'ndofs':>7s} {'flux err':>10s} "
      f"{'order':>6s} {'pred':>5s} {'div resid':>10s}  verdict")
for mname, M, dim, pairs in $hdiv:
    for label, fobj, pobj, pred in pairs:
        pcls = P0_OF[mname] if MUTATE == "1" else type(pobj)
        try:
            e1, r1, n1 = mixed_flux_error(M, dim, type(fobj), pcls, REFINE)
            e2, r2, n2 = mixed_flux_error(M, dim, type(fobj), pcls, REFINE + 1)
            order = float(np.log(e1 / e2) / np.log(2.0)) if e1 > 0 and e2 > 0 else float("nan")
            ok = verdict(label, "mms_order", pred, order, ORDER_BAND)
            print(f"  {mname:9s} {label:15s} {pcls.__name__:15s} {n2:7d} {e2:10.3e} "
                  f"{order:6.2f} {pred:5d} {r2:10.1e}  {'PASS' if ok else 'FAIL'}")
        except Exception as exc:
            print(f"  {mname:9s} {label:15s} {pcls.__name__:15s} {'-':>7s} {'-':>10s} "
                  f"{'FAILED':>6s}  {type(exc).__name__}: {str(exc)[:70]}")
# The control: an incompatible pairing passes the residual and converges nowhere.
try:
    c1, _, _ = mixed_flux_error(MeshTri, 2, ElementTriRT2, ElementTriP0, REFINE)
    c2, cr, _ = mixed_flux_error(MeshTri, 2, ElementTriRT2, ElementTriP0, REFINE + 1)
    print(f"  (control) the second-order triangle flux with a piecewise-constant pressure: "
          f"div residual {cr:.1e}, flux order {np.log(c1 / c2) / np.log(2.0):.2f} -- "
          f"the residual is not a verification")
except Exception as exc:
    print(f"  (control) did not run: {type(exc).__name__}")

print()
print("H(curl): the de Rham identity -- the gradients of P_k lie in the Nedelec space of order k, and curl-curl annihilates them")
print(f"  {'mesh':9s} {'element':15s} {'ndofs':>7s} {'k':>2s} {'grad P_k in space':>18s} {'curl-curl |Ku|':>15s}  verdict")


@BilinearForm
def _curl_curl(u, v, w):
    c = curl(u)
    cc = curl(v)
    return c * cc if np.ndim(c) == np.ndim(w.x[0]) else dot(c, cc)


def _grad_phi(k, dim):
    '''The gradient of one fixed polynomial of degree k (k = 1, 2, 3).'''
    def g(x):
        z = 0 * x[0]
        if k == 1:
            comps = [1.0 + z, 2.0 + z] + ([3.0 + z] if dim == 3 else [])
        elif k == 2:
            comps = [2 * x[0] + x[1], x[0] + 4 * x[1] + (x[2] if dim == 3 else z)] \
                    + ([x[1]] if dim == 3 else [])
        else:
            comps = [3 * x[0] ** 2 + x[1] ** 2 + (x[1] * x[2] if dim == 3 else z),
                     2 * x[0] * x[1] + 3 * x[1] ** 2 + (x[0] * x[2] if dim == 3 else z)] \
                    + ([x[0] * x[1]] if dim == 3 else [])
        return np.array(comps)
    return g


for mname, M, dim, els in $hcurl:
    m = M().refined(REFINE)
    for label, eobj, k in els:
        if MUTATE == "2":
            k = 2
        try:
            b = Basis(m, type(eobj)())
            g = _grad_phi(k, dim)
            u = b.project(g)                              # L2 projection onto the Nedelec space
            eb = Basis(m, type(eobj)(), intorder=ERROR_INTORDER)
            d = np.asarray(eb.interpolate(u)) - g(eb.global_coordinates())
            inspace = float(np.sqrt(np.sum(d ** 2 * eb.dx)))
            K = _curl_curl.assemble(b)
            ann = float(np.linalg.norm(K @ u) / (abs(K).max() * np.linalg.norm(u)))
            ok1 = verdict(label, "exact_identity", 0.0, inspace, EXACT_TOL)
            ok2 = verdict(label, "exact_identity", 0.0, ann, EXACT_TOL)
            print(f"  {mname:9s} {label:15s} {b.N:7d} {k:2d} {inspace:18.1e} {ann:15.1e}  "
                  f"{'PASS' if ok1 and ok2 else 'FAIL'}")
        except Exception as exc:
            print(f"  {mname:9s} {label:15s} {'-':>7s} {k:2d} {'FAILED':>18s}  {type(exc).__name__}: {str(exc)[:60]}")

print()
print("L2 / DG: project sin(pi x)... and measure the order under refinement")
print(f"  {'mesh':9s} {'element':18s} {'L2 error':>11s} {'order':>6s} {'pred':>5s}  verdict")


@BilinearForm
def _mass(u, v, w):
    return u * v


@LinearForm
def _target_load(v, w):
    return _u(w.x) * v


for mname, M, dim, els in $l2:
    for label, eobj, pred in els:
        errs = []
        try:
            for lvl in (REFINE, REFINE + 1):
                m = M().refined(lvl)
                b = Basis(m, type(eobj)())
                c = solve(_mass.assemble(b), _target_load.assemble(b))
                # MEASURE THE ERROR ON A RICHER RULE THAN THE ONE THAT DEFINED
                # THE PROJECTION. With the default quadrature a P0 or
                # low-degree DG projection reproduces the target EXACTLY at
                # the quadrature points -- the same points the mass matrix
                # and load were built from -- so the measured error is zero
                # by construction. Zero is not accuracy there, it is a
                # tautology.
                eb = Basis(m, type(eobj)(), intorder=ERROR_INTORDER)
                diff = np.asarray(eb.interpolate(c)) - _u(eb.global_coordinates())
                errs.append(float(np.sqrt(np.sum(diff ** 2 * eb.dx))))
            order = float(np.log(errs[0] / errs[1]) / np.log(2.0)) if errs[0] > 0 and errs[1] > 0 else float("nan")
            ok = verdict(label, "mms_order", pred, order, ORDER_BAND)
            print(f"  {mname:9s} {label:18s} {errs[1]:11.3e} {order:6.2f} {pred:5d}  {'PASS' if ok else 'FAIL'}")
        except Exception as exc:
            print(f"  {mname:9s} {label:18s} {'-':>11s} {'FAILED':>6s}  {type(exc).__name__}")
print()
print("SKELETON SPACES: the constant on the mesh skeleton, projected through FacetBasis + InteriorFacetBasis")
print(f"  {'mesh':9s} {'element':22s} {'ndofs':>7s} {'L2|Pi(1)-1|':>12s}  verdict")


@BilinearForm
def _fmass(u, v, w):
    return u * v


@LinearForm
def _fone(v, w):
    return 1.0 * v


for mname, M, els in $skeleton:
    m = M().refined(REFINE)
    for label, eobj in els:
        try:
            fb = FacetBasis(m, type(eobj)())
            ib = InteriorFacetBasis(m, type(eobj)())
            c = solve(_fmass.assemble(fb) + _fmass.assemble(ib),
                      _fone.assemble(fb) + _fone.assemble(ib))
            fbh = FacetBasis(m, type(eobj)(), intorder=ERROR_INTORDER)
            ibh = InteriorFacetBasis(m, type(eobj)(), intorder=ERROR_INTORDER)
            err = float(np.sqrt(sum(np.sum((np.asarray(bb.interpolate(c)) - 1.0) ** 2 * bb.dx)
                                    for bb in (fbh, ibh))))
            ok = verdict(label, "exact_identity", 0.0, err, EXACT_TOL)
            print(f"  {mname:9s} {label:22s} {fb.N:7d} {err:12.1e}  {'PASS' if ok else 'FAIL'}")
        except Exception as exc:
            print(f"  {mname:9s} {label:22s} {'-':>7s} {'FAILED':>12s}  {type(exc).__name__}: {str(exc)[:60]}")

print()
print("HHJ SPACES: symmetric matrices with normal-normal continuity reproduce the identity matrix")
print(f"  {'element':22s} {'ndofs':>7s} {'L2|Pi(I)-I|':>12s}  verdict")


@BilinearForm
def _mmass(u, v, w):
    return ddot(u, v)


@LinearForm
def _mid(v, w):
    return ddot(np.eye(2)[:, :, None, None] * np.ones_like(w.x[0]), v)


for label, eobj in $hhj:
    try:
        m = MeshTri().refined(REFINE)
        b = Basis(m, type(eobj)())
        c = solve(_mmass.assemble(b), _mid.assemble(b))
        eb = Basis(m, type(eobj)(), intorder=ERROR_INTORDER)
        val = np.asarray(eb.interpolate(c))
        ident = np.eye(2)[:, :, None, None] * np.ones_like(val[0, 0])
        err = float(np.sqrt(np.sum((val - ident) ** 2 * eb.dx)))
        ok = verdict(label, "exact_identity", 0.0, err, EXACT_TOL)
        print(f"  {label:22s} {b.N:7d} {err:12.1e}  {'PASS' if ok else 'FAIL'}")
    except Exception as exc:
        print(f"  {label:22s} {'-':>7s} {'FAILED':>12s}  {type(exc).__name__}: {str(exc)[:60]}")

print()
print("A MESH THAT DOES NOT REFINE: ElementWedge1 on MeshWedge1 reproduces a constant (no ladder, so no order)")


@LinearForm
def _one(v, w):
    return 1.0 * v


try:
    m = MeshWedge1()
    b = Basis(m, ElementWedge1())
    c = solve(_mass.assemble(b), _one.assemble(b))
    eb = Basis(m, ElementWedge1(), intorder=ERROR_INTORDER)
    err = float(np.sqrt(np.sum((np.asarray(eb.interpolate(c)) - 1.0) ** 2 * eb.dx)))
    ok = verdict("ElementWedge1", "exact_identity", 0.0, err, EXACT_TOL)
    print(f"  {'MeshWedge1':9s} {'ElementWedge1':22s} {b.N:7d} {err:12.1e}  {'PASS' if ok else 'FAIL'}")
except Exception as exc:
    print(f"  {'MeshWedge1':9s} {'ElementWedge1':22s} {'-':>7s} {'FAILED':>12s}  {type(exc).__name__}: {str(exc)[:60]}")
print()
print("A flux that converges at its space's order against the exact gradient is")
print("the mixed pairing working; a divergence residual at solver tolerance is")
print("only the linear system being solved, and an incompatible pairing has it")
print("too. A degree-p discontinuous space projects at order p+1.")
if MUTATE:
    print("[MUTATED: SURVEY_MUTATE=%s -- the FAIL lines are the control working]" % MUTATE)
""")


def _mixed_element_survey(params: dict) -> str:
    """Mixed, vector and discontinuous elements, each checked its own way."""
    return _MIXED.substitute(
        title=params.get("title", "scikit-fem mixed and vector element survey"),
        refine=params.get("refine", 3),
        intorder=params.get("error_intorder", 8),
        hdiv=params.get("hdiv_pairs", _HDIV),
        hcurl=params.get("hcurl_elements", _HCURL),
        l2=params.get("l2_elements", _L2),
        skeleton=params.get("skeleton_elements", _SKELETON),
        hhj=params.get("hhj_elements", _HHJ),
    )


GENERATORS = {
    # The key is <physics>_<variant> for the advertised row ("mixed_elements",
    # variant "survey"): generate_input("mixed_elements", "survey") looked this
    # up as "mixed_elements_survey" and raised "Unknown variant" until
    # 2026-09-23 -- the row was advertised but could not be served.
    "mixed_elements_survey": _mixed_element_survey,
}

KNOWLEDGE = {
    "mixed_elements": {
        "description": (
            "Survey the element families a Poisson survey cannot reach: the "
            "mixed flux's convergence order against a manufactured gradient "
            "for H(div), the de Rham identity for the Nedelec spaces, the "
            "projection order of the discontinuous spaces, and the constant "
            "reproduced by the skeleton spaces (through FacetBasis and "
            "InteriorFacetBasis), the HHJ spaces (the identity matrix) and "
            "ElementWedge1 on the one mesh that does not refine -- one "
            "VERDICT line per check."
        ),
        "pitfalls": [
            "[Numerical] A SELF-CONSISTENCY CHECK IS NOT A VERIFICATION. The "
            "discrete divergence residual |B sigma - f| / |f| of a mixed "
            "Poisson solve is the saddle point's own second equation, so a "
            "direct solver satisfies it to ~1e-15 for ANY pairing that "
            "assembles, compatible or not: ElementTriRT2 with an "
            "ElementTriP0 pressure keeps residual 4e-15 while its flux error "
            "does not converge (order 0.00); with ElementTriP1DG it "
            "converges at 1.99. Only an error against an independent exact "
            "field (here the gradient of the manufactured solution) says "
            "whether the pairing is right. Signal: `solve()` returning a "
            "residual at machine epsilon from `sp.bmat([[A, B.T], [B, "
            "None]])` while the flux interpolated through "
            "Basis.interpolate() sits far from any expected field.",

            "[API] THE TRIAL AND TEST BASES OF A COUPLING FORM MUST SHARE A "
            "QUADRATURE. Basis(mesh, ElementTriP0()) built on its own "
            "default rule and Basis(mesh, ElementTriRT2()) on its own raise "
            "ValueError('Quadrature mismatch: trial and test functions "
            "should have same number of integration points') in "
            "BilinearForm.assemble(fb, pb); pass quadrature=fb.quadrature "
            "to the second Basis. This survey reported that ValueError as "
            "ElementTriRT2, ElementTriBDM1 and ElementTetRT0 'failing in "
            "this formulation' until 2026-09-23; all three converge. "
            "Signal: that exact ValueError message from "
            "BilinearForm.assemble with two bases of different maxdeg.",

            "[Numerical] A PROJECTION ERROR MEASURED ON THE SAME RULE THAT "
            "BUILT THE PROJECTION IS ZERO BY CONSTRUCTION. Measured: "
            "ElementTetP0 reports 1.7e-17 on the default rule and 9.6e-02 "
            "at intorder=6; ElementTriP1DG 8.8e-17 against 2.0e-02. Signal: "
            "an error at machine epsilon and an apparent convergence order "
            "of nan or wildly large from Basis(mesh, elem) without an "
            "explicit intorder -- integrate the error on a richer rule.",

            "[API] ElementTriRT0 IS ElementTriRT1, AND THE OTHER ALIASES. "
            "scikit-fem 12 keeps ElementTriRT0, ElementQuadRT0, "
            "ElementTetRT0 and ElementTetN0 as aliases of ElementTriRT1, "
            "ElementQuadRT1, ElementTetRT1 and ElementTetN1 (also "
            "ElementTriCCR of ElementTriP2B and ElementTriMini of "
            "ElementTriP1B), and the lowest-order Nedelec space is called "
            "N1 on every shape that has one (ElementTriN1, ElementQuadN1, "
            "ElementTetN1); ElementTetN0 is only a backwards-compatibility "
            "alias of ElementTetN1 and the only N0-named class, so no "
            "triangle class carries an N0 name. Signal: "
            "type(ElementTriRT0()).__name__ printing 'ElementTriRT1', and "
            "two rows of a survey that agree to every digit.",
        ],
    },
}
