"""deal.II finite-element survey: every FE class this install declares.

WHAT THIS IS FOR. deal.II is a library for BUILDING solvers, so what it
declares is mechanism, not a list of problems you can pose. The 41 FE_*
classes in ``include/deal.II/fe`` are that mechanism, and before this module
openPASO reached three of them.

EVERY CLASS IS WRITTEN AS A LITERAL CONSTRUCTOR CALL. Resolving the names
through a factory (``FETools::get_fe_by_name("FE_Q(2)")`` would do it in one
line) turns the emitted program into a list of strings, and a string is not a
use: it cannot be compiled against, and a class deleted upstream would still
"appear" to be reached.

THREE CHECKS, NONE OF WHICH "IT COMPILED" CAN SATISFY

  0. The quadrature integrates the cell it claims to. Sum JxW and compare with
     the reference cell's own declared volume.
  1. The mass matrix is SPD, by a hand-written Cholesky -- no LAPACK, so the
     verdict does not depend on how this install was configured. The pivot is
     weighed against the diagonal, because a pivot of 4.9e-09 against entries
     of 2.5e-01 is roundoff impersonating positivity.
  2. The space reproduces a constant, measured on a RICHER rule than the one
     that built the matrix. Measuring the projection error on the rule that
     defined the projection returns zero by construction and proves nothing.

MEASURED ON deal.II 9.8.0-pre (Release) -- 41 of 41 classes built and passed,
0 failed, and all 41 produce DISTINCT results. That last clause had to be
earned: six pairs were reporting byte-identical lines, so 41 checks were only
35 distinct outcomes. Five were continuous/discontinuous twins
(FE_SimplexP/FE_SimplexDGP, FE_WedgeP/FE_WedgeDGP, FE_PyramidP/FE_PyramidDGP,
FE_Nedelec/FE_DGNedelec, FE_RaviartThomasNodal/FE_DGRaviartThomas) which a
SINGLE cell cannot tell apart, because there is no neighbour to be
discontinuous across; the line now reports where the dofs live, and a
discontinuous element keeps all of them in the cell interior. The sixth,
FE_DGP against FE_DGPNonparametric, are both discontinuous so that does not
separate them either -- one is mapped from the reference cell and the other
built in real space, which is visible only on a cell that is NOT the reference
cell, so both are checked on a distorted one. The mutation control (SURVEY_MUTATE, see the source) turns the
checks red on demand: mapping mutation 7 failures, constant mutation 38.

WHAT THE RUN FOUND, all of it silent -- nothing raised, nothing returned an
error code, and every one of these produced a plausible-looking number:

  * MappingQ1 is correct ONLY on hypercubes. On the reference triangle it
    integrates the cell as 1/6 instead of 1/2, on a wedge as 0.0528 instead of
    1/2, on a pyramid as 4.0 instead of 4/3. MappingFE with the matching P1
    element gives the exact measure in all three cases.
  * QGaussSimplex and QGaussWedge exist only up to n_points_1D = 4, and
    QGaussPyramid only up to 2. Ask for more and deal.II hands back an EMPTY
    rule; the only guard is a debug-only Assert, so in a Release build
    FEValues accepts it, reinit() succeeds, and every integral is exactly 0.
  * FE_Hermite reads the cell extents off MappingCartesian::InternalData and
    hits DEAL_II_ASSERT_UNREACHABLE() under any other mapping.
  * FE_Q_DG0 has a SINGULAR mass matrix on any closed mesh: the measured null
    vector is (sum of the Q1 dofs) - (sum of the DG0 dofs), eigenvalue 2.8e-14
    against a next eigenvalue of 1.05e-02. The constant is represented twice.
  * FE_P1NC is singular on the reference square AND on a distorted quad: the
    measured null vector is (+1/2, -1/2, -1/2, +1/2), eigenvalue 1.7e-14
    against 8.3e-02. Its four shape functions span a THREE-dimensional space.
  * FE_FaceQ, FE_FaceP and FE_TraceQ have no interior basis at all. A cell
    mass matrix for them is NaN, not merely singular; they need a face one.
"""

_SURVEY_CC = r'''// deal.II element survey, v2 -- every FE class this install declares, built as
// a LITERAL constructor call, on a footing appropriate to the element.
//
// THREE CHECKS, none of which can be satisfied by "it compiled":
//
//  0. THE QUADRATURE IS REAL. Sum JxW over the cell and compare with the
//     reference cell's OWN declared volume. This exists because
//     QGaussPyramid implements only n_points_1D 1 and 2; for any other value
//     it leaves the rule EMPTY, and the only guard is a debug-only Assert.
//     In a Release build FEValues then accepts a rule with zero points,
//     reinit() succeeds, nothing throws, and every integral is exactly zero.
//     v1 of this survey reported the pyramid mass matrix as "not SPD" and
//     blamed the element; the element was fine and the rule was empty.
//
//  1. THE MASS MATRIX IS SPD. Cholesky succeeds iff it is, so the smallest
//     pivot is a positivity margin. Done by hand: no LAPACK, so the verdict
//     does not depend on how this install was configured.
//
//  2. THE SPACE REPRODUCES A CONSTANT. Every space here contains the constant
//     field exactly, so its L2 projection must return it. The error is
//     integrated on a RICHER rule than the one that built the matrix --
//     measuring it on the rule that defined the projection returns zero by
//     construction and proves nothing.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping_cartesian.h>
#include <deal.II/fe/mapping_fe.h>
#include <memory>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_bubbles.h>
#include <deal.II/fe/fe_q_dg0.h>
#include <deal.II/fe/fe_q_hierarchical.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_bernstein.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_dgp_nonparametric.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_rt_bubbles.h>
#include <deal.II/fe/fe_abf.h>
#include <deal.II/fe/fe_bdm.h>
#include <deal.II/fe/fe_dg_vector.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_nedelec_sz.h>
#include <deal.II/fe/fe_bernardi_raugel.h>
#include <deal.II/fe/fe_rannacher_turek.h>
#include <deal.II/fe/fe_p1nc.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_trace.h>
#include <deal.II/fe/fe_hermite.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_enriched.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_simplex_p_bubbles.h>
#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_wedge_p.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

using namespace dealii;

enum Mode { CELL, FACE, MULTICELL, DISTORTED };

// One VERDICT line per independent reference, in the coverage harness's own
// grammar (scripts/coverage_harness/definitions.py, VERDICT_RE). The harness's
// format_verdict() cannot be imported from a C++ program, so this prints the
// same bytes: ref/got/tol as %.6e, the word PASS or FAIL, an optional note.
// The re-parse test on the harness side holds it to that.
static bool verdict(const std::string &entry, const char *kind, double ref,
                    double got, double tol, const char *note = nullptr)
{
  const bool pass = std::fabs(got - ref) <= tol;
  if (note)
    std::printf("VERDICT dealii %s %s ref=%.6e got=%.6e tol=%.6e %s note=\"%s\"\n",
                entry.c_str(), kind, ref, got, tol, pass ? "PASS" : "FAIL", note);
  else
    std::printf("VERDICT dealii %s %s ref=%.6e got=%.6e tol=%.6e %s\n",
                entry.c_str(), kind, ref, got, tol, pass ? "PASS" : "FAIL");
  return pass;
}

static const double EXACT_TOL = 1e-10;   // the harness's exact_identity ceiling

// MUTATION CONTROL. A survey that reports 41/41 is worth nothing until the
// checks are shown to be capable of reporting anything else.
//   SURVEY_MUTATE=1  use MappingQ1 on every cell, including simplex, wedge and
//                    pyramid -- check 0 (the volume) must catch it.
//   SURVEY_MUTATE=2  judge the projection against 1 + 1e-6 instead of 1 --
//                    check 2 (constant reproduction) must catch it.
//   SURVEY_MUTATE=3  request the uncapped Gauss rule -- the empty-rule
//                    fallback disappears and check 0 must catch the zeros.
static int mutate() { const char *e = getenv("SURVEY_MUTATE"); return e ? atoi(e) : 0; }
// What the element is REQUIRED to do. An element that stops doing it turns
// this survey red; a check with no failing outcome verifies nothing.
enum Expect { SPD_AND_CONST, SINGULAR };

// MappingQ1 is only correct on hypercubes. On a simplex it integrates the
// reference triangle as 1/6 instead of 1/2, on a wedge as 0.0528 instead of
// 1/2, on a pyramid as 4.0 instead of 4/3 -- all three silently, with
// nothing raised. MappingFE with the matching P1 element gives the exact
// measure on all three. Measured on deal.II 9.8.0-pre, Release.
template <int dim>
static std::unique_ptr<Mapping<dim>> mapping_for(const ReferenceCell &rc)
{
  if (mutate() == 1)                   return std::make_unique<MappingQ1<dim>>();
  if (rc.is_hyper_cube())              return std::make_unique<MappingQ1<dim>>();
  if (rc.is_simplex())                 return std::make_unique<MappingFE<dim>>(FE_SimplexP<dim>(1));
  if (rc == ReferenceCells::Wedge)     return std::make_unique<MappingFE<dim>>(FE_WedgeP<dim>(1));
  if (rc == ReferenceCells::Pyramid)   return std::make_unique<MappingFE<dim>>(FE_PyramidP<dim>(1));
  return std::make_unique<MappingQ1<dim>>();
}

// QGaussSimplex and QGaussWedge exist only up to n_points_1D = 4, QGaussPyramid
// only up to 2. Above that deal.II returns an EMPTY rule and the only guard is
// a debug-only Assert, so in a Release build every integral silently becomes 0.
static unsigned int rule_cap(const ReferenceCell &rc)
{
  if (mutate() == 3)                 return 100;
  if (rc == ReferenceCells::Pyramid) return 2;
  if (rc.is_hyper_cube())            return 100;
  return 4;
}
static int n_ok = 0, n_fail = 0;

static bool cholesky(FullMatrix<double> A, double &min_pivot)
{
  const unsigned int n = A.m();
  min_pivot = 1e300;
  for (unsigned int k = 0; k < n; ++k) {
    double d = A(k, k);
    for (unsigned int s = 0; s < k; ++s) d -= A(k, s) * A(k, s);
    if (d <= 0.0 || !std::isfinite(d)) { min_pivot = d; return false; }
    min_pivot = std::min(min_pivot, d);
    A(k, k) = std::sqrt(d);
    for (unsigned int i = k + 1; i < n; ++i) {
      double v = A(i, k);
      for (unsigned int s = 0; s < k; ++s) v -= A(i, s) * A(k, s);
      A(i, k) = v / A(k, k);
    }
  }
  min_pivot = std::sqrt(min_pivot);
  return true;
}

// Pick a Gauss rule for this reference cell that ACTUALLY HAS POINTS, and say
// so when the requested one did not.
template <int dim>
static Quadrature<dim> safe_rule(const ReferenceCell &rc, unsigned int want,
                                 std::string &note)
{
  for (unsigned int n = want; n >= 1; --n) {
    const Quadrature<dim> q = rc.template get_gauss_type_quadrature<dim>(n);
    if (q.size() > 0) {
      if (n != want)
        note += " [rule " + std::to_string(want) + " is EMPTY here, fell back to "
              + std::to_string(n) + "]";
      return q;
    }
  }
  note += " [no non-empty Gauss rule]";
  return Quadrature<dim>();
}

template <int dim>
void check(const std::string &name, const FiniteElement<dim> &fe,
           Mode mode = CELL, Expect expect = SPD_AND_CONST,
           const Mapping<dim> *mapping_in = nullptr)
{
  const std::unique_ptr<Mapping<dim>> owned = mapping_for<dim>(fe.reference_cell());
  const Mapping<dim> &mapping = mapping_in ? *mapping_in : *owned;
  std::string note;
  try {
    const unsigned int nd_cell = fe.n_dofs_per_cell();
    if (nd_cell == 0) {
      std::printf("OK   %-24s dim=%d  dofs=0  (FE_Nothing spans nothing: no "
                  "mass matrix to be positive)\n", name.c_str(), dim);
      ++n_ok; std::fflush(stdout); return;
    }

    Triangulation<dim> tria;
    if (mode == MULTICELL)
      GridGenerator::subdivided_hyper_cube(tria, 2);
    else
      GridGenerator::reference_cell(tria, fe.reference_cell());

    double shoelace = -1.0;   // independent area of a distorted 2-D quad
    if (mode == DISTORTED) {
      // FE_P1NC is rank deficient on a PARALLELOGRAM, which the reference
      // square is. Push one vertex off so the cell is a general quad.
      tria.begin_active()->vertex(3) += Point<dim>(0.4, 0.25);
      note += " [vertex 3 displaced: a general quad, not a parallelogram]";
      if (dim == 2) {
        // deal.II's quad vertex order is 0:(0,0) 1:(1,0) 2:(0,1) 3:(1,1), so
        // the polygon runs 0-1-3-2. Shoelace on those four points is the
        // area the quadrature must reproduce, computed without FEValues.
        const auto c = tria.begin_active();
        const unsigned int ring[4] = {0, 1, 3, 2};
        double a = 0.0;
        for (unsigned int k = 0; k < 4; ++k) {
          const Point<dim> &p = c->vertex(ring[k]);
          const Point<dim> &q = c->vertex(ring[(k + 1) % 4]);
          a += p[0] * q[1] - q[0] * p[1];
        }
        shoelace = 0.5 * std::fabs(a);
      }
    }

    DoFHandler<dim> dh(tria);
    dh.distribute_dofs(fe);
    const unsigned int N = dh.n_dofs(), nc = fe.n_components(), deg = fe.degree;

    const unsigned int cap = rule_cap(fe.reference_cell());
    const Quadrature<dim> qa = safe_rule<dim>(fe.reference_cell(), std::min(deg + 2, cap), note);
    const Quadrature<dim> qb = safe_rule<dim>(fe.reference_cell(), std::min(deg + 4, cap), note);
    if (qa.size() == 0 || qb.size() == 0) {
      std::printf("FAIL %-24s dim=%d  no usable quadrature%s\n",
                  name.c_str(), dim, note.c_str());
      ++n_fail; std::fflush(stdout); return;
    }

    FullMatrix<double> M(N, N);
    Vector<double> b(N);
    std::vector<types::global_dof_index> idx(nd_cell);
    double measured_volume = 0.0;

    if (mode == FACE) {
      // A face element has NO interior basis: its mass matrix lives on the
      // faces. Summing over every face of the cell gives an SPD matrix.
      const Quadrature<dim - 1> qf =
        fe.reference_cell().face_reference_cell(0)
          .template get_gauss_type_quadrature<dim - 1>(deg + 2);
      // every flag this loop reads: a Release-only deal.II build segfaults, with no message, on a missing one
      FEFaceValues<dim> ff(mapping, fe, qf, update_values | update_JxW_values | update_quadrature_points);
      for (const auto &cell : dh.active_cell_iterators()) {
        cell->get_dof_indices(idx);
        for (const unsigned int f : cell->face_indices()) {
          ff.reinit(cell, f);
          for (unsigned int q = 0; q < qf.size(); ++q) {
            measured_volume += ff.JxW(q);
            for (unsigned int i = 0; i < nd_cell; ++i) {
              for (unsigned int j = 0; j < nd_cell; ++j)
                for (unsigned int c = 0; c < nc; ++c)
                  M(idx[i], idx[j]) += ff.shape_value_component(i, q, c) *
                                       ff.shape_value_component(j, q, c) * ff.JxW(q);
              for (unsigned int c = 0; c < nc; ++c)
                b(idx[i]) += ff.shape_value_component(i, q, c) * ff.JxW(q);
            }
          }
        }
      }
      note += " [face mass matrix, summed over all faces]";
      double expect_faces = 0.0;
      for (const unsigned int f : fe.reference_cell().face_indices())
        expect_faces += fe.reference_cell().face_reference_cell(f).volume();
      if (!verdict(name, "exact_identity", expect_faces, measured_volume, EXACT_TOL)) {
        std::printf("FAIL %-24s dim=%d  face rule does not integrate the faces: "
                    "sum(JxW)=%.6e vs %.6e%s\n",
                    name.c_str(), dim, measured_volume, expect_faces, note.c_str());
        ++n_fail; std::fflush(stdout); return;
      }
    } else {
      // every flag this loop reads: a Release-only deal.II build segfaults, with no message, on a missing one
      FEValues<dim> fa(mapping, fe, qa, update_values | update_JxW_values | update_quadrature_points);
      for (const auto &cell : dh.active_cell_iterators()) {
        cell->get_dof_indices(idx);
        fa.reinit(cell);
        for (unsigned int q = 0; q < qa.size(); ++q) {
          measured_volume += fa.JxW(q);
          for (unsigned int i = 0; i < nd_cell; ++i) {
            for (unsigned int j = 0; j < nd_cell; ++j)
              for (unsigned int c = 0; c < nc; ++c)
                M(idx[i], idx[j]) += fa.shape_value_component(i, q, c) *
                                     fa.shape_value_component(j, q, c) * fa.JxW(q);
            for (unsigned int c = 0; c < nc; ++c)
              b(idx[i]) += fa.shape_value_component(i, q, c) * fa.JxW(q);
          }
        }
      }
      // CHECK 0 -- the rule integrates the cell it claims to integrate.
      // The reference is a number the quadrature never saw: the reference
      // cell's declared volume, the unit cube's volume for the 2x2 mesh, or
      // the shoelace area of the distorted quad's own vertices.
      double expect = -1.0;
      if (mode == CELL)           expect = fe.reference_cell().volume();
      else if (mode == MULTICELL) expect = 1.0;
      else if (mode == DISTORTED) expect = shoelace;
      if (expect >= 0.0) {
        const bool ok = verdict(name, "exact_identity", expect, measured_volume,
                                EXACT_TOL);
        if (!ok) {
          std::printf("FAIL %-24s dim=%d  QUADRATURE IS NOT THE CELL: "
                      "sum(JxW)=%.6e but the independent reference is %.6e%s\n",
                      name.c_str(), dim, measured_volume, expect, note.c_str());
          ++n_fail; std::fflush(stdout); return;
        }
      }
    }

    double sym = 0.0;
    for (unsigned int i = 0; i < N; ++i)
      for (unsigned int j = 0; j < N; ++j)
        sym = std::max(sym, std::fabs(M(i, j) - M(j, i)));

    // Cholesky succeeding is not enough: FE_Q_DG0 factorises with a pivot of
    // 4.9e-09 against diagonal entries of order 1e-1, which is roundoff
    // pretending to be positivity. Scale the pivot by the diagonal.
    double scale = 0.0;
    for (unsigned int i = 0; i < N; ++i) scale = std::max(scale, M(i, i));
    double piv = 0.0;
    const bool spd = cholesky(M, piv) && piv > 1e-7 * std::sqrt(scale);

    if (expect == SINGULAR) {
      // The null vector was MEASURED (LAPACK, eigenvalue 1.7e-14 for FE_P1NC
      // and 2.8e-14 for FE_Q_DG0) and is encoded here so the survey asserts
      // the specific dependency rather than "singular, as expected":
      //   FE_P1NC   v = (+1/2, -1/2, -1/2, +1/2) in dof order;
      //   FE_Q_DG0  v = +1 on every Q1 dof, -1 on every cell's DG0 dof
      //             (the constant represented twice), normalised.
      Vector<double> v(N);
      if (name == "FE_P1NC" && N == 4) {
        v(0) = 0.5; v(1) = -0.5; v(2) = -0.5; v(3) = 0.5;
      } else if (name == "FE_Q_DG0") {
        for (const auto &cell : dh.active_cell_iterators()) {
          cell->get_dof_indices(idx);
          for (unsigned int i = 0; i < nd_cell; ++i)
            v(idx[i]) = (i == nd_cell - 1) ? -1.0 : 1.0;   // last local dof = DG0
        }
        v /= v.l2_norm();
      }
      double mmax = 0.0;
      for (unsigned int i = 0; i < N; ++i)
        for (unsigned int j = 0; j < N; ++j)
          mmax = std::max(mmax, std::fabs(M(i, j)));
      Vector<double> Mv(N);
      M.vmult(Mv, v);
      const double resid = Mv.l2_norm() / (mmax * v.l2_norm());
      const bool nullok = verdict(name, "exact_identity", 0.0, resid, EXACT_TOL,
                                  "predicted null vector: |M v|/(max|M| |v|)");
      if (!nullok) {
        std::printf("FAIL %-24s dim=%d  the predicted null vector is NOT in the "
                    "null space: |Mv|/(max|M||v|) = %.3e%s\n",
                    name.c_str(), dim, resid, note.c_str());
        ++n_fail; std::fflush(stdout); return;
      }
      if (spd) {
        std::printf("FAIL %-24s dim=%d  expected a SINGULAR mass matrix here "
                    "and got an SPD one (pivot %.3e)%s\n",
                    name.c_str(), dim, piv, note.c_str());
        ++n_fail;
      } else {
        std::printf("OK   %-24s dim=%d  dofs=%3u comp=%u deg=%u  vol=%.4f  "
                    "SINGULAR AS EXPECTED (pivot %.2e vs diag %.2e)%s\n",
                    name.c_str(), dim, N, nc, deg, measured_volume, piv, scale,
                    note.c_str());
        ++n_ok;
      }
      std::fflush(stdout); return;
    }
    if (!spd) {
      std::printf("FAIL %-24s dim=%d  dofs=%3u  mass matrix NOT SPD "
                  "(pivot %.3e vs diag %.3e) -- basis linearly dependent%s\n",
                  name.c_str(), dim, N, piv, scale, note.c_str());
      ++n_fail; std::fflush(stdout); return;
    }

    FullMatrix<double> Minv(N, N);
    Minv.invert(M);
    Vector<double> x(N);
    Minv.vmult(x, b);

    double err2 = 0.0;
    if (mode == FACE) {
      const Quadrature<dim - 1> qf2 =
        fe.reference_cell().face_reference_cell(0)
          .template get_gauss_type_quadrature<dim - 1>(deg + 4);
      // every flag this loop reads: a Release-only deal.II build segfaults, with no message, on a missing one
      FEFaceValues<dim> fb(mapping, fe, qf2, update_values | update_JxW_values | update_quadrature_points);
      for (const auto &cell : dh.active_cell_iterators()) {
        cell->get_dof_indices(idx);
        for (const unsigned int f : cell->face_indices()) {
          fb.reinit(cell, f);
          for (unsigned int q = 0; q < qf2.size(); ++q)
            for (unsigned int c = 0; c < nc; ++c) {
              double v = 0.0;
              for (unsigned int i = 0; i < nd_cell; ++i)
                v += x(idx[i]) * fb.shape_value_component(i, q, c);
              const double target = (mutate() == 2) ? 1.0 + 1e-6 : 1.0;
              err2 += (v - target) * (v - target) * fb.JxW(q);
            }
        }
      }
    } else {
      // every flag this loop reads: a Release-only deal.II build segfaults, with no message, on a missing one
      FEValues<dim> fb(mapping, fe, qb, update_values | update_JxW_values | update_quadrature_points);
      for (const auto &cell : dh.active_cell_iterators()) {
        cell->get_dof_indices(idx);
        fb.reinit(cell);
        for (unsigned int q = 0; q < qb.size(); ++q)
          for (unsigned int c = 0; c < nc; ++c) {
            double v = 0.0;
            for (unsigned int i = 0; i < nd_cell; ++i)
              v += x(idx[i]) * fb.shape_value_component(i, q, c);
            const double target = (mutate() == 2) ? 1.0 + 1e-6 : 1.0;
            err2 += (v - target) * (v - target) * fb.JxW(q);
          }
      }
    }

    const double perr = std::sqrt(err2);
    // The projection of the constant 1 must be 1: a manufactured solution
    // every one of these spaces contains, judged on a richer rule than the
    // one that built the matrix, against a reference of exactly zero error.
    if (!verdict(name, "exact_identity", 0.0, perr, EXACT_TOL,
                 "L2 error of the constant's projection, richer rule")) {
      std::printf("FAIL %-24s dim=%d  dofs=%3u  does NOT reproduce a constant: "
                  "L2|Pi(1)-1| = %.3e%s\n",
                  name.c_str(), dim, N, perr, note.c_str());
      ++n_fail; std::fflush(stdout); return;
    }
    // WHERE the dofs live, not just how many. Six pairs of elements produced
    // byte-identical lines without this -- FE_SimplexP/FE_SimplexDGP,
    // FE_WedgeP/FE_WedgeDGP, FE_PyramidP/FE_PyramidDGP,
    // FE_Nedelec/FE_DGNedelec and FE_RaviartThomasNodal/FE_DGRaviartThomas --
    // because a single cell has no neighbour to be discontinuous ACROSS, so
    // the check could not tell a discontinuous element from its continuous
    // twin. 41 elements were producing 35 distinct results. A discontinuous
    // element keeps every dof in the cell interior; its twin does not.
    std::printf("OK   %-24s dim=%d  dofs=%3u comp=%u deg=%u  vol=%.4f "
                "sym=%.1e minpivot=%.3e  L2|Pi(1)-1|=%.3e  "
                "dofs/vert=%u dofs/line=%u dofs/face=%u%s\n",
                name.c_str(), dim, N, nc, deg, measured_volume, sym, piv,
                perr, fe.n_dofs_per_vertex(), fe.n_dofs_per_line(),
                fe.n_dofs_per_face(), note.c_str());
    ++n_ok;
  } catch (const std::exception &e) {
    std::string m(e.what());
    for (auto &ch : m) if (ch == '\n') ch = ' ';
    const auto p = m.find("Additional information");
    if (p != std::string::npos) m = m.substr(p, 170);
    else if (m.size() > 170) m = m.substr(0, 170);
    std::printf("FAIL %-24s dim=%d  %s%s\n", name.c_str(), dim, m.c_str(), note.c_str());
    ++n_fail;
  }
  std::fflush(stdout);
}

int main()
{
  deal_II_exceptions::disable_abort_on_exception();

  // ---- H1 Lagrange family --------------------------------------------
  check<2>("FE_Q",              FE_Q<2>(2));
  check<2>("FE_Q_Bubbles",      FE_Q_Bubbles<2>(2));
  // FE_Q_DG0 adds a CELL-LOCAL discontinuous constant. On a single cell that
  // constant equals the sum of the Q1 nodal functions (they are a partition
  // of unity), so the basis is dependent there by construction -- measured:
  // sum_i phi_i = 2.0 on one cell. It needs more than one cell to be a basis.
  // Measured null vector: (sum of the Q1 dofs) - (sum of the DG0 dofs), with
  // eigenvalue 2.8e-14 against a next eigenvalue of 1.05e-02. The constant is
  // represented twice, so the mass matrix is singular on ANY closed mesh; real
  // use fixes the mean instead.
  check<2>("FE_Q_DG0",          FE_Q_DG0<2>(1), MULTICELL, SINGULAR);
  check<2>("FE_Q_Hierarchical", FE_Q_Hierarchical<2>(2));
  check<2>("FE_Q_iso_Q1",       FE_Q_iso_Q1<2>(2));
  check<2>("FE_Bernstein",      FE_Bernstein<2>(2));
  // FE_Hermite reads the cell extents off MappingCartesian::InternalData and
  // hits DEAL_II_ASSERT_UNREACHABLE() with any other mapping.
  {
    const MappingCartesian<2> mc;
    check<2>("FE_Hermite",      FE_Hermite<2>(3), CELL, SPD_AND_CONST, &mc);
  }

  // ---- L2 / discontinuous family --------------------------------------
  check<2>("FE_DGQ",               FE_DGQ<2>(2));
  check<2>("FE_DGQArbitraryNodes", FE_DGQArbitraryNodes<2>(QGauss<1>(3)));
  check<2>("FE_DGQHermite",        FE_DGQHermite<2>(3));
  check<2>("FE_DGQLegendre",       FE_DGQLegendre<2>(2));
  // Both of these are discontinuous, so the dof distribution does not
  // separate them; FE_DGP is mapped from the reference cell and
  // FE_DGPNonparametric is built in real space, and that is only visible on a
  // cell which is NOT the reference cell. On the reference square they agree
  // to the last digit.
  check<2>("FE_DGP",               FE_DGP<2>(2), DISTORTED);
  check<2>("FE_DGPMonomial",       FE_DGPMonomial<2>(2));
  check<2>("FE_DGPNonparametric",  FE_DGPNonparametric<2>(2), DISTORTED);

  // ---- H(div) ----------------------------------------------------------
  check<2>("FE_RaviartThomas",      FE_RaviartThomas<2>(1));
  check<2>("FE_RaviartThomasNodal", FE_RaviartThomasNodal<2>(1));
  check<2>("FE_DGRaviartThomas",    FE_DGRaviartThomas<2>(1));
  check<2>("FE_RT_Bubbles",         FE_RT_Bubbles<2>(1));
  check<2>("FE_ABF",                FE_ABF<2>(0));
  check<2>("FE_BDM",                FE_BDM<2>(1));
  check<2>("FE_DGBDM",              FE_DGBDM<2>(1));
  check<2>("FE_BernardiRaugel",     FE_BernardiRaugel<2>(1));

  // ---- H(curl) ---------------------------------------------------------
  check<2>("FE_Nedelec",     FE_Nedelec<2>(0));
  check<2>("FE_NedelecSZ",   FE_NedelecSZ<2>(0));
  check<2>("FE_DGNedelec",   FE_DGNedelec<2>(0));
  check<3>("FE_NedelecNodal", FE_NedelecNodal<3>(0));

  // ---- nonconforming ---------------------------------------------------
  check<2>("FE_RannacherTurek", FE_RannacherTurek<2>(0));
  // Measured null vector (+1/2, -1/2, -1/2, +1/2), eigenvalue 1.7e-14 against
  // a next eigenvalue of 8.3e-02: phi0 - phi1 - phi2 + phi3 = 0. The four
  // shape functions span a THREE-dimensional space, on the reference square
  // and on a distorted quad alike.
  check<2>("FE_P1NC",           FE_P1NC(), DISTORTED, SINGULAR);

  // ---- face / trace: no interior basis, so a FACE mass matrix ----------
  check<2>("FE_FaceQ",  FE_FaceQ<2>(1),  FACE, SPD_AND_CONST);
  check<2>("FE_FaceP",  FE_FaceP<2>(1),  FACE, SPD_AND_CONST);
  check<2>("FE_TraceQ", FE_TraceQ<2>(1), FACE, SPD_AND_CONST);

  // ---- composition ------------------------------------------------------
  check<2>("FE_Nothing",  FE_Nothing<2>());
  check<2>("FESystem",    FESystem<2>(FE_Q<2>(2), 2, FE_Q<2>(1), 1));
  check<2>("FE_Enriched", FE_Enriched<2>(FE_Q<2>(1)));

  // ---- simplex / wedge / pyramid ----------------------------------------
  check<2>("FE_SimplexP",         FE_SimplexP<2>(2));
  check<2>("FE_SimplexDGP",       FE_SimplexDGP<2>(2));
  check<2>("FE_SimplexP_Bubbles", FE_SimplexP_Bubbles<2>(1));
  check<3>("FE_WedgeP",           FE_WedgeP<3>(1));
  check<3>("FE_WedgeDGP",         FE_WedgeDGP<3>(1));
  check<3>("FE_PyramidP",         FE_PyramidP<3>(1));
  check<3>("FE_PyramidDGP",       FE_PyramidDGP<3>(1));

  std::printf("\nSURVEY %d built and checked, %d failed%s\n", n_ok, n_fail,
              mutate() ? "  [MUTATED: failures here are the control working]" : "");
  return 0;
}
'''


def _element_survey(params: dict) -> str:
    """FORMAT TEMPLATE -- a survey, not a physics problem: it takes no
    geometry and no material. Compile and run it to learn which FE classes
    this deal.II install has and how each behaves."""
    return _SURVEY_CC


GENERATORS = {
    "element_survey": _element_survey,
    "element_survey_default": _element_survey,
}


PITFALLS = [
    "[mesh] MappingQ1 is correct only on hypercubes. On a simplex, "
    "wedge or pyramid mesh it produces WRONG Jacobians and nothing raises: "
    "measured on deal.II 9.8.0-pre, the reference triangle integrates to "
    "1/6 instead of 1/2, the reference wedge to 0.0528 instead of 1/2, and the "
    "reference pyramid to 4.0 instead of 4/3. Use MappingFE<dim>(FE_SimplexP<dim>(1)), "
    "MappingFE<dim>(FE_WedgeP<dim>(1)) or MappingFE<dim>(FE_PyramidP<dim>(1)) to match "
    "the cell. Signal: sum of FEValues::JxW over a cell disagrees with "
    "cell->measure() or with reference_cell().volume(); every integral is off "
    "by a constant factor and the solve still 'converges'.",

    "[silent-wrong] ReferenceCell::get_gauss_type_quadrature returns an "
    "EMPTY rule for orders its underlying formula does not implement: "
    "QGaussSimplex and QGaussWedge stop at n_points_1D = 4, QGaussPyramid at 2. "
    "The only guard is a debug-only Assert, so a Release build silently accepts "
    "a rule with zero points -- FEValues::reinit succeeds, no exception is "
    "thrown, and every integral evaluates to exactly 0. Signal: "
    "quadrature.size() == 0, a mass matrix of all zeros, or a cell volume of "
    "0.0 from summing JxW. Check q.size() > 0 before assembling.",

    "[api] FE_Hermite only works with MappingCartesian. It reads "
    "the cell extents off MappingCartesian::InternalData and, given any other "
    "mapping, hits DEAL_II_ASSERT_UNREACHABLE() in fe_hermite.cc. Signal: an "
    "abort inside fe_hermite.cc during FEValues::reinit, not at construction -- "
    "FE_Hermite<2>(3) itself constructs fine and reports 16 dofs per cell.",

    "[numerical] FE_Q_DG0's mass matrix is SINGULAR on any closed mesh, "
    "by construction and not by a bug: it adds a cell-local discontinuous "
    "constant to a Lagrange space that is already a partition of unity, so the "
    "constant is represented twice. Measured null vector: (sum of the Q1 dofs) "
    "- (sum of the DG0 dofs), eigenvalue 2.8e-14 against a next eigenvalue of "
    "1.05e-02. Signal: a direct solver reports a tiny pivot instead of failing, "
    "or an iterative solver stalls. The element is for Stokes pressure, where "
    "the mean is fixed separately; fix it.",

    "[numerical] FE_P1NC's four shape functions span a THREE-dimensional "
    "space, so its mass matrix is rank deficient on the reference square and on "
    "a distorted quad alike. Measured null vector (+1/2, -1/2, -1/2, +1/2), "
    "eigenvalue 1.7e-14 against a next eigenvalue of 8.3e-02. Signal: the "
    "smallest eigenvalue of a local mass or stiffness matrix sits at roundoff "
    "while the next is O(1e-1).",

    "[api] FE_FaceQ, FE_FaceP and FE_TraceQ carry no interior "
    "basis. Evaluating them with FEValues inside a cell does not raise -- it "
    "yields a mass matrix whose Cholesky pivot is NaN. They need FEFaceValues "
    "and a face mass matrix. Signal: NaN rather than a singular-matrix error.",
]


KNOWLEDGE = {
    "description":
        "Survey of every FE_* class this deal.II install declares. One program "
        "constructs all of them as literal constructor calls and checks each "
        "three ways: the quadrature integrates the cell it claims to, the mass "
        "matrix is SPD by a hand-written Cholesky, and the space reproduces a "
        "constant with the error measured on a RICHER rule than the one that "
        "built the matrix. Each check that has an independent reference prints "
        "a VERDICT line beside its verdict (reference, measured value, "
        "tolerance): the cell volume against reference_cell().volume(), the "
        "unit cube or the shoelace area of the distorted quad's own vertices, "
        "the summed face measure, the predicted null vector of a singular "
        "element as |M v|/(max|M| |v|) against 0, and the constant's "
        "projection error against 0. Measured on 9.8.0-pre: 41 of 41 built "
        "and checked, 0 failed, through this backend's own compile path; the "
        "SURVEY_MUTATE control turns 7 of those VERDICT lines to FAIL.",
    "function_space":
        "all 41 FE_* classes the installed headers declare, including "
        "FESystem; see the survey source for the exact constructor of each",
    "mapping":
        "MappingQ1 is correct ONLY on hypercubes. Use "
        "MappingFE<dim>(FE_SimplexP<dim>(1)), MappingFE<dim>(FE_WedgeP<dim>(1)) "
        "or MappingFE<dim>(FE_PyramidP<dim>(1)) to match a simplex, wedge or "
        "pyramid cell, and MappingCartesian for FE_Hermite.",
    "pitfalls": PITFALLS,
}
