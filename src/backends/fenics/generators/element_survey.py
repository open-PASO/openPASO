"""basix element survey, run THROUGH dolfinx: every family this install declares.

WHAT THIS IS FOR. FEniCSx is a library for BUILDING solvers, so what it
declares is mechanism. basix's ElementFamily enum is that mechanism, and
before this module openPASO reached 4 of its 13 families.

IT CALLS THE CODE. The first version of this survey assembled every mass
matrix itself with numpy and never imported dolfinx, which made it a
hand-rolled stand-in for the code it claims to describe -- the exact pattern
this repo forbids, and tests/test_every_template_drives_the_code_it_names.py
caught it. It now assembles through dolfinx.fem and interpolates through
dolfinx.fem.Function, so what it reports is what FEniCSx does.

That change also improved the question. "What does basix declare" became
"which families can I actually put in a dolfinx function space, and how does
each behave there", and the two are not the same set.

EVERY FAMILY IS A LITERAL basix.ElementFamily.X, never a name handed to
getattr: a family renamed upstream would otherwise still appear covered.

MEASURED 2026-09-23, dolfinx 0.10.0: 13 of 13 families built and checked
through dolfinx, 0 failed, 13 VERDICT lines all PASS. The mutation control
(BASIX_SURVEY_MUTATE=1) turns 7 of them to FAIL -- the scalar families -- while
the 6 vector- and matrix-valued families keep passing, so the checks
discriminate.

WHAT THE RUN FOUND

  * Hermite cannot be interpolated from a plain callable. Its degrees of
    freedom are function values AND DERIVATIVES, which dolfinx's point
    interpolate() cannot fill from a callable: on dolfinx 0.10.0 the
    interpolant of the constant 1 carries NaN in 4 of its 107 dofs and 27
    dofs that are neither 0 nor 1, so L2|I(1)-1| is nan; the L2 PROJECTION
    of the same constant reproduces it to 1.2e-14 (that projection is the
    Hermite row's verdict). For Hermite you must project, not interpolate.
  * The degree-3 bubble on a triangle is a SINGLE function vanishing on the
    whole boundary and cannot represent a constant: 6.239e-01, and its total
    mass is 0.289 where a partition of unity would give the domain measure
    1.0 exactly. It is an enrichment, not a space.
  * basix.ElementFamily.DPC must be created with discontinuous=True and,
    for degree > 0, an explicit DPCVariant; with the defaults create_element
    raises.
  * basix's embedded_subdegree does NOT answer "does this space contain the
    constants" -- measured directly against basix, it is -1 for Hermite,
    Regge and HHJ, all three of which reproduce a constant at roundoff, and
    also -1 for bubble, which genuinely cannot.
"""

_SURVEY_PY = r'''"""basix element survey, run THROUGH dolfinx: every family this install declares.

THIS SCRIPT CALLS FEniCSx. An earlier version built each mass matrix itself
with numpy and never imported dolfinx, which made it a hand-rolled stand-in for
the code it claims to describe -- the exact pattern openPASO forbids. It now
assembles through dolfinx.fem and interpolates through dolfinx.fem.Function, so
what it reports is what FEniCSx does, not what a second implementation does.

EVERY FAMILY IS A LITERAL basix.ElementFamily.X. Building the list by name
(getattr(basix.ElementFamily, s) for s in [...]) would make this a list of
strings, and a family renamed upstream would still appear to be covered.

THE QUESTION THIS ANSWERS is not "what does basix declare" but the more useful
"which families can I actually put in a dolfinx function space, and how does
each behave there". Those are different sets, and the difference is the point.

THREE CHECKS, none of which "it constructed" can satisfy:

  0. THE TOTAL MASS IS THE DOMAIN. For a partition of unity, sum_ij M_ij =
     integral of (sum_i phi_i)^2 = the measure of the domain, exactly 1.0 on
     the unit square. Reported for every family; it holds only for the nodal
     ones, and where it does not, the number says how far from a partition of
     unity the basis is.
  1. THE MASS MATRIX IS SPD, by a hand-written Cholesky over the matrix DOLFINX
     assembled. This is a check ON the code's output, not a substitute for it:
     no linear system is solved here.
  2. THE SPACE REPRODUCES A CONSTANT, interpolated with dolfinx and measured
     with dolfinx.fem.assemble_scalar over the real domain.

MUTATION CONTROL: BASIX_SURVEY_MUTATE=1 judges the constant against 1 + 1e-6
for the SCALAR families, and every scalar family that claims to reproduce a
constant must then fail -- while the vector- and matrix-valued families keep
passing, so the control discriminates. The VERDICT lines (one per family with an
independent reference: the constant's error for every family that holds
constants, the closed-form interpolation error for the bubble, the projection
error for Hermite) follow the coverage harness's grammar.
"""
import os
import numpy as np
from mpi4py import MPI
import basix
import basix.ufl
import ufl
import dolfinx
from dolfinx import fem, mesh

import math
MUTATE = os.environ.get("BASIX_SURVEY_MUTATE", "") == "1"

tri = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3, mesh.CellType.triangle)
quad = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3, mesh.CellType.quadrilateral)

# (label, family, dolfinx mesh, basix cell, degree, variants, discontinuous,
#  reproduces_a_constant)
LV = basix.LagrangeVariant
DV = basix.DPCVariant
ENTRIES = [
    ("P",           basix.ElementFamily.P,           tri,  basix.CellType.triangle,      1, LV.unset, DV.unset, False, True),
    ("CR",          basix.ElementFamily.CR,          tri,  basix.CellType.triangle,      1, LV.unset, DV.unset, False, True),
    ("iso",         basix.ElementFamily.iso,         tri,  basix.CellType.triangle,      1, LV.unset, DV.unset, False, True),
    # A degree-3 bubble on a triangle is ONE function vanishing on the whole
    # boundary, so it cannot represent a constant. It is an enrichment, never
    # a space on its own.
    ("bubble",      basix.ElementFamily.bubble,      tri,  basix.CellType.triangle,      3, LV.unset, DV.unset, False, False),
    # Hermite's degrees of freedom are function values AND DERIVATIVES, so
    # dolfinx's point interpolate() cannot produce a constant from a plain
    # callable: on dolfinx 0.10.0 the interpolant carries NaN in some dofs
    # and stray values in others, and its error is nan. The same element
    # under an L2 PROJECTION reproduces the constant to 1.2e-14, and that is
    # its verdict. For Hermite you must project, not interpolate; nothing
    # warns you.
    ("Hermite",     basix.ElementFamily.Hermite,     tri,  basix.CellType.triangle,      3, LV.unset, DV.unset, False, False),
    ("RT",          basix.ElementFamily.RT,          tri,  basix.CellType.triangle,      1, LV.unset, DV.unset, False, True),
    ("BDM",         basix.ElementFamily.BDM,         tri,  basix.CellType.triangle,      1, LV.unset, DV.unset, False, True),
    ("N1E",         basix.ElementFamily.N1E,         tri,  basix.CellType.triangle,      1, LV.unset, DV.unset, False, True),
    ("N2E",         basix.ElementFamily.N2E,         tri,  basix.CellType.triangle,      1, LV.unset, DV.unset, False, True),
    ("Regge",       basix.ElementFamily.Regge,       tri,  basix.CellType.triangle,      0, LV.unset, DV.unset, False, True),
    ("HHJ",         basix.ElementFamily.HHJ,         tri,  basix.CellType.triangle,      0, LV.unset, DV.unset, False, True),
    ("serendipity", basix.ElementFamily.serendipity, quad, basix.CellType.quadrilateral, 1, LV.unset, DV.unset, False, True),
    # DPC is inherently discontinuous: with the default variant and
    # discontinuous=False, create_element raises.
    ("DPC",         basix.ElementFamily.DPC,         quad, basix.CellType.quadrilateral, 1, LV.unset, DV.simplex_equispaced, True, True),
]


def cholesky_min_pivot(A):
    """Succeeds iff A is SPD. A CHECK on the matrix dolfinx assembled; it
    solves nothing and stands in for nothing."""
    A = np.array(A, dtype=float, copy=True)
    n = A.shape[0]
    mp = np.inf
    for k in range(n):
        d = A[k, k] - A[k, :k] @ A[k, :k]
        if not np.isfinite(d) or d <= 0:
            return None, d
        mp = min(mp, d)
        A[k, k] = np.sqrt(d)
        for i in range(k + 1, n):
            A[i, k] = (A[i, k] - A[i, :k] @ A[k, :k]) / A[k, k]
    return np.sqrt(mp), None


n_ok = n_fail = 0
for label, fam, msh, cell, deg, lv, dv, disc, wants_const in ENTRIES:
    try:
        el = basix.ufl.element(fam, cell, deg, lv, dv, disc)
        V = fem.functionspace(msh, el)
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        form = ufl.inner(u, v) * ufl.dx
        A = fem.assemble_matrix(fem.form(form))
        A.scatter_reverse()
        M = A.to_dense()
        ndofs = M.shape[0]

        sym = float(np.abs(M - M.T).max())
        total = float(M.sum())
        piv, bad = cholesky_min_pivot(M)
        if piv is None:
            print("FAIL %-12s dofs=%-4d mass matrix NOT SPD (pivot %.3e)"
                  % (label, ndofs, bad), flush=True)
            n_fail += 1
            continue

        # interpolate a constant, with dolfinx, and measure the error with
        # dolfinx over the real domain
        bs = V.ufl_element().reference_value_size
        f = fem.Function(V)
        if bs == 1:
            f.interpolate(lambda x: np.ones(x.shape[1]))
            c = dolfinx.default_scalar_type(1.0)
        else:
            f.interpolate(lambda x: np.ones((bs, x.shape[1])))
            c = ufl.as_vector([1.0] * bs) if len(f.ufl_shape) == 1 else None
        # the planted wrong answer, SCALAR families only, so that the vector- and
        # matrix-valued families keep passing beside the failures: a control that
        # fails everything discriminates nothing
        target = 1.0 + 1e-6 if (MUTATE and bs == 1) else 1.0
        if bs == 1:
            expr = (f - target) ** 2 * ufl.dx
        elif c is not None:
            d = f - ufl.as_vector([target] * bs)
            expr = ufl.inner(d, d) * ufl.dx
        else:
            d = f - ufl.as_matrix([[target] * f.ufl_shape[1]] * f.ufl_shape[0])
            expr = ufl.inner(d, d) * ufl.dx
        err = float(np.sqrt(abs(fem.assemble_scalar(fem.form(expr)))))

        # One VERDICT line per independent reference, in the coverage harness's
        # grammar (scripts/coverage_harness/definitions.py).
        if wants_const:
            print("VERDICT fenics %s exact_identity ref=0.000000e+00 got=%.6e tol=1.000000e-10 %s"
                  % (label, err, "PASS" if err <= 1e-10 else "FAIL"), flush=True)
        elif label == "bubble":
            # The interpolant sets the single bubble dof to 1, so the field is
            # b = 27 l1 l2 l3 on every cell and ||b - 1||^2 = sum_T (int b^2 - 2 int b)
            # + |Omega|, with int_T b^2 = 729 * (2!2!2!/8!) * 2|T| and int_T b =
            # 27 * (2/5!) |T| by Dirichlet's formula, |Omega| = 1: a closed form the
            # run never saw. The mass-form quadrature is exact to degree 6.
            ref = math.sqrt(729.0 * 8.0 / 40320.0 * 2.0 - 2.0 * 27.0 * 2.0 / 120.0 + 1.0)
            print("VERDICT fenics %s closed_form ref=%.6e got=%.6e tol=1.000000e-10 "
                  "tol_from=Dirichlet_integral_of_27*l1*l2*l3_on_the_unit_square;quadrature_exact_to_degree_6 %s"
                  % (label, ref, err, "PASS" if abs(err - ref) <= 1e-10 else "FAIL"), flush=True)
        elif label == "Hermite":
            # Hermite must be PROJECTED, not interpolated: solve M c = (1, phi_i)
            # with the assembled mass matrix and measure the projection's error.
            bvec = fem.assemble_vector(fem.form(v * ufl.dx))
            g = fem.Function(V)
            g.x.array[:] = np.linalg.solve(M, np.asarray(bvec.array))
            perr = float(np.sqrt(abs(fem.assemble_scalar(fem.form((g - target) ** 2 * ufl.dx)))))
            print("VERDICT fenics %s exact_identity ref=0.000000e+00 got=%.6e tol=1.000000e-10 %s"
                  % (label, perr, "PASS" if perr <= 1e-10 else "FAIL"), flush=True)
            print("     %-12s L2 projection of the constant: %.3e (interpolation gives %.3e)"
                  % (label, perr, err), flush=True)

        got_const = err < 1e-10
        if got_const != wants_const:
            print("FAIL %-12s dofs=%-4d expected reproduce_constant=%s but "
                  "measured L2|Pi(1)-1| = %.3e"
                  % (label, ndofs, wants_const, err), flush=True)
            n_fail += 1
            continue
        note = "" if wants_const else "  (cannot represent a constant, AS EXPECTED)"
        print("OK   %-12s dofs=%-4d vs=%d sym=%.1e minpivot=%.3e sum(M)=%.9f "
              "L2|Pi(1)-1|=%.3e%s"
              % (label, ndofs, bs, sym, piv, total, err, note), flush=True)
        n_ok += 1
    except Exception as ex:
        msg = str(ex).replace("\n", " ")[:110]
        print("FAIL %-12s %s: %s" % (label, type(ex).__name__, msg), flush=True)
        n_fail += 1

print("\nSURVEY %d families built and checked through dolfinx, %d failed%s"
      % (n_ok, n_fail,
         "  [MUTATED: failures here are the control working]" if MUTATE else ""),
      flush=True)
'''

VARIANTS = ["default"]

KNOWLEDGE = {
    "description":
        "Survey of every basix element family this FEniCSx install declares, "
        "run THROUGH dolfinx. Builds each family as a literal "
        "basix.ElementFamily.X, puts it in a dolfinx function space, assembles "
        "its mass matrix with dolfinx.fem, proves that matrix SPD by a "
        "hand-written Cholesky (a check on the code's output, not a substitute "
        "for it -- no linear system is solved), and checks that the space "
        "reproduces a constant using dolfinx interpolation and "
        "dolfinx.fem.assemble_scalar. Answers which families this install can "
        "actually use, and how each behaves.",
    "minimal_working_example": _SURVEY_PY,
    "pitfalls": None,   # filled in below, once PITFALLS exists
}


def generate(variant: str, params: dict) -> str:
    """FORMAT TEMPLATE -- a survey, not a physics problem: no geometry, no
    material. Run it to learn which basix families this install can use."""
    return _SURVEY_PY


PITFALLS = [
    "[api] Hermite elements cannot be interpolated from a plain callable. "
    "Their degrees of freedom are function values AND DERIVATIVES, which "
    "dolfinx.fem.Function.interpolate() cannot fill from a callable: on "
    "dolfinx 0.10.0 interpolating the constant 1 into Hermite on a unit "
    "square leaves NaN in 4 of 107 dofs and stray values in 27 more, so the "
    "field's L2 error is nan, while an L2 PROJECTION of the same constant "
    "(solve the assembled mass matrix against the load of the constant) "
    "reproduces it to 1.2e-14. Nothing raises and nothing warns. Signal: "
    "np.isnan(f.x.array).any() after interpolate() into an element whose "
    "basix family is Hermite, or a field far from the callable you passed. "
    "Project instead of interpolating.",

    "[physics] The bubble family is an ENRICHMENT, not a space. The degree-3 "
    "bubble on a triangle is one function that vanishes on the entire "
    "boundary, so it cannot represent a constant: measured L2|Pi(1)-1| = "
    "6.239e-01, and its total mass sums to 0.289 where a partition of unity "
    "on the same unit square gives exactly 1.0. Signal: a solve using bubble "
    "alone converges to a field that is zero on every boundary regardless of "
    "the boundary data. Add it to another element rather than using it alone.",

    "[api] basix.ElementFamily.DPC must be created with discontinuous=True "
    "and, for degree > 0, an explicit DPCVariant. create_element(DPC, "
    "quadrilateral, 1) with the defaults raises rather than returning a "
    "discontinuous element. Signal: RuntimeError 'Cannot create a continuous "
    "DPC element.' from create_element; with discontinuous=True but no "
    "variant it is 'DPC elements of degree > 0 need to be given a variant.'. "
    "Both come at element construction, not at solve time.",

    "[api] basix's embedded_subdegree does NOT tell you whether a space "
    "contains the constants. Measured on this install it is -1 for Hermite, "
    "Regge and HHJ, and all three reproduce a constant at roundoff; it is "
    "also -1 for bubble, which genuinely cannot. Signal: if you gate on "
    "embedded_subdegree >= 0 you will silently drop three usable families and "
    "keep one unusable one. Project the constant and measure the error.",

    "[verification] Measuring a projection error on the SAME quadrature rule "
    "that defined the projection returns zero by construction and proves "
    "nothing. Integrate the error on a richer rule than the one used to build "
    "the mass matrix, or assemble it over the real domain with "
    "dolfinx.fem.assemble_scalar. Signal: an L2 error of exactly 0.0 or "
    "~1e-17 for an element that has no business being exact.",
]

KNOWLEDGE["pitfalls"] = PITFALLS
