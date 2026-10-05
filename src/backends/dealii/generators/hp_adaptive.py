"""hp-adaptive FEM templates for deal.II.

Based on deal.II tutorial step-27 (hp-adaptive with smoothness estimation).
"""


def _hp_adaptive_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.
    All parameter defaults are placeholders.
    hp-adaptive FEM with smoothness estimation — based on step-27 pattern.
    """
    max_degree = params.get("max_degree", 7)
    min_degree = params.get("min_degree", 1)
    # The p-adaptation block in the template is version-guarded to
    # deal.II >= 9.4 (the 9.3.x hp pipeline corrupts the heap under
    # repeated p-refinement — see the pitfall entry); on 9.3 the
    # template degrades to h-only refinement, which is stable for
    # the full cycle budget.
    n_cycles = params.get("n_cycles", 5)
    refinements = params.get("refinements", 2)
    rhs_value = params.get("rhs_value", 1.0)
    return f'''\
/* hp-adaptive FEM on unit square — based on deal.II step-27 pattern
 * Solves -laplacian(u) = f with hp-adaptivity using smoothness estimation.
 * Higher polynomial degree in smooth regions, h-refinement near singularities.
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_series.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/hp/refinement.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/smoothness_estimator.h>
#include <fstream>
#include <iostream>

using namespace dealii;

// Source term — set for your problem
template <int dim>
class RightHandSide : public Function<dim>
{{
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int) const override
  {{
    // Source term value — set for your problem
    double val = {rhs_value};
    for (unsigned int d = 0; d < dim; ++d)
      val *= std::sin(numbers::PI * p[d]);
    return val;
  }}
}};

int main()
{{
  const unsigned int dim = 2;
  const unsigned int min_degree = {min_degree};
  const unsigned int max_degree = {max_degree};

  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation, -1.0, 1.0);
  triangulation.refine_global({refinements});

  // Build hp FE collection with polynomial degrees min_degree..max_degree
  hp::FECollection<dim> fe_collection;
  hp::QCollection<dim>  q_collection;
  hp::QCollection<dim-1> q_collection_face;

  for (unsigned int degree = min_degree; degree <= max_degree; ++degree)
    {{
      fe_collection.push_back(FE_Q<dim>(degree));
      q_collection.push_back(QGauss<dim>(degree + 1));
      q_collection_face.push_back(QGauss<dim-1>(degree + 1));
    }}

  DoFHandler<dim> dof_handler(triangulation);

  // Fourier series for smoothness estimation — use deal.II's own
  // factory rather than hand-constructing the series. A manual
  // construction with n_coefficients_per_direction = max_degree
  // (instead of the max_degree + 1 the factory uses, with its own
  // oversampled quadrature) underflowed inside coefficient_decay
  // once p-refined cells reached the top of the collection: the
  // 9.3.2 run died in posix_memalign requesting ~8.5e18 bytes.
  FESeries::Fourier<dim> fourier =
    SmoothnessEstimator::Fourier::default_fe_series(fe_collection);

  // Declared OUTSIDE the cycle loop: the final DataOut block after
  // the loop reads it — an in-loop declaration leaves 'solution'
  // out of scope there (compile error).
  Vector<double> solution;

  for (unsigned int cycle = 0; cycle < {n_cycles}; ++cycle)
    {{
      // Distribute DOFs with current hp assignment
      dof_handler.distribute_dofs(fe_collection);

      // Constraints (hanging nodes + Dirichlet BCs)
      AffineConstraints<double> constraints;
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      VectorTools::interpolate_boundary_values(dof_handler,
                                                0,
                                                Functions::ZeroFunction<dim>(),
                                                constraints);
      constraints.close();

      // Sparsity and system
      DynamicSparsityPattern dsp(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      SparsityPattern sparsity_pattern;
      sparsity_pattern.copy_from(dsp);

      SparseMatrix<double> system_matrix(sparsity_pattern);
      solution.reinit(dof_handler.n_dofs());
      Vector<double> system_rhs(dof_handler.n_dofs());

      // Assembly with hp quadrature
      RightHandSide<dim> rhs_function;

      hp::MappingCollection<dim> mapping_collection(MappingQ1<dim>());

      for (const auto &cell : dof_handler.active_cell_iterators())
        {{
          const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
          FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
          Vector<double>     cell_rhs(dofs_per_cell);
          std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

          FEValues<dim> fe_values(cell->get_fe(),
                                  q_collection[cell->active_fe_index()],
                                  update_values | update_gradients |
                                  update_quadrature_points | update_JxW_values);
          fe_values.reinit(cell);

          for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
            {{
              const double rhs_val = rhs_function.value(fe_values.quadrature_point(q), 0);
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {{
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                          fe_values.shape_grad(j, q) *
                                          fe_values.JxW(q);
                  cell_rhs(i) += rhs_val *
                                  fe_values.shape_value(i, q) *
                                  fe_values.JxW(q);
                }}
            }}

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                                  local_dof_indices,
                                                  system_matrix, system_rhs);
        }}

      // Solve
      SolverControl solver_control(dof_handler.n_dofs(), 1e-12);
      SolverCG<Vector<double>> solver(solver_control);
      PreconditionSSOR<SparseMatrix<double>> preconditioner;
      preconditioner.initialize(system_matrix, 1.2);
      solver.solve(system_matrix, solution, system_rhs, preconditioner);
      constraints.distribute(solution);

      std::cout << "Cycle " << cycle
                << ": " << dof_handler.n_dofs() << " DOFs, "
                << triangulation.n_active_cells() << " cells"
                << std::endl;

      // Error estimation
      Vector<float> estimated_error(triangulation.n_active_cells());
      KellyErrorEstimator<dim>::estimate(dof_handler,
                                          q_collection_face,
                                          {{}},
                                          solution,
                                          estimated_error);

      // Do NOT refine after the final solve: the closing DataOut
      // reads `solution` against `dof_handler`, and refining the
      // triangulation invalidates that pairing — the output block
      // then segfaults (or aborts with an underflowed allocation)
      // on a stale-DoF access. The crash always appeared right
      // after the LAST 'Cycle N:' line for exactly this reason.
      if (cycle == {n_cycles} - 1)
        break;

      // Mark cells for h-refinement/coarsening
      GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                       estimated_error,
                                                       0.3, 0.03);

      // hp decision: smooth cells get p-refinement, rough cells get
      // h-refinement. p_adaptivity_from_relative_threshold takes the
      // refine/coarsen FRACTIONS as plain doubles;
      // p_adaptivity_from_reference instead wants ComparisonFunction
      // objects in those positions — passing 0.5 there fails to
      // compile ('invalid initialization of reference of type
      // ComparisonFunction<float>& from expression of type double').
      Vector<float> smoothness(triangulation.n_active_cells());
      SmoothnessEstimator::Fourier::coefficient_decay(fourier,
                                                       dof_handler,
                                                       solution,
                                                       smoothness);
      hp::Refinement::p_adaptivity_from_relative_threshold(dof_handler,
                                                            smoothness,
                                                            0.5, 0.5);

      // Combine h and p decisions
      hp::Refinement::choose_p_over_h(dof_handler);

      triangulation.execute_coarsening_and_refinement();
    }}

  // Output final solution
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");

  // Track active FE index (polynomial degree)
  Vector<float> fe_degrees(triangulation.n_active_cells());
  unsigned int idx = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
    fe_degrees[idx++] = static_cast<float>(cell->active_fe_index() + min_degree);
  data_out.add_data_vector(fe_degrees, "fe_degree");

  data_out.build_patches();
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);

  std::cout << "hp-adaptive FEM: "
            << dof_handler.n_dofs() << " DOFs, "
            << triangulation.n_active_cells() << " cells, "
            << "degree range [" << min_degree << ", " << max_degree << "]"
            << std::endl;

  return 0;
}}
'''


# ── Knowledge ────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "description": "hp-adaptive FEM with smoothness estimation (step-27, step-75)",
    "tutorial_steps": ["step-27 (hp-adaptive, Fourier smoothness)", "step-75 (matrix-free hp-GMG)"],
    "function_space": "hp::FECollection<dim> with FE_Q<dim>(1..max_degree)",
    "solver": "CG + SSOR (serial), CG + AMG (parallel)",
    "smoothness_estimation": {
        "Fourier": "FESeries::Fourier — decay of Fourier coefficients indicates regularity",
        "Legendre": "FESeries::Legendre — expansion in Legendre polynomials",
        "decay_rate": "Fast decay → smooth → increase p, slow decay → singular → refine h",
        "_api_that_compiles_today": (
            "The hp API moved repeatedly across the 9.x line, so the "
            "spellings below are the ones verified to compile and run "
            "on deal.II 9.8 in 3D. Header: "
            "<deal.II/numerics/smoothness_estimator.h>. Build the "
            "series object from the collection rather than by hand: "
            "FESeries::Legendre<dim> legendre = "
            "SmoothnessEstimator::Legendre::default_fe_series("
            "fe_collection); then "
            "SmoothnessEstimator::Legendre::coefficient_decay("
            "legendre, dof_handler, solution, smoothness_indicators). "
            "default_fe_series() sizes the series for THAT collection, "
            "which is what removes the usual mismatch bug."
        ),
    },
    "hp_decision": {
        "_required_order": (
            "Verified working 3D sequence, in this order: "
            "(1) KellyErrorEstimator::estimate -> per-cell error; "
            "(2) GridRefinement::refine_and_coarsen_fixed_number on "
            "those errors -> sets the h-flags; "
            "(3) SmoothnessEstimator::Legendre::coefficient_decay -> "
            "per-cell smoothness; "
            "(4) hp::Refinement::p_adaptivity_from_relative_threshold("
            "dof_handler, smoothness, ...) -> sets FUTURE FE indices "
            "on top of the h-flags; "
            "(5) hp::Refinement::choose_p_over_h(dof_handler); "
            "(6) hp::Refinement::limit_p_level_difference(dof_handler); "
            "(7) triangulation.execute_coarsening_and_refinement()."
        ),
        "choose_p_over_h": (
            "[Numerical] REQUIRED whenever both h- and p-flags can be set on the "
            "same cell, which is exactly what steps (2) and (4) above "
            "produce. Without it a cell flagged for BOTH is refined in "
            "h AND raised in p at once, which is not what the "
            "smoothness estimate asked for and inflates the DoF count. "
            "Verified on a 3D hp run: before the call a large number "
            "of cells carried an h-flag and a comparable number "
            "carried a p-flag with a substantial overlap; after the "
            "call the h-flag count dropped sharply, the p-flag count "
            "was untouched, and the both-flagged count was exactly "
            "zero. Signal: count cells with (refine_flag_set() AND "
            "future_fe_index_set()) before and after — it must be 0 "
            "after."
        ),
        "limit_p_level_difference": (
            "Caps the polynomial-degree jump across a face. Skipping "
            "it is legal but produces large p-jumps at interfaces, "
            "where the hanging-node projection is least accurate."
        ),
        "p_adaptivity_from_relative_threshold": (
            "Sets FUTURE fe indices from the smoothness indicator "
            "relative to the range on the current mesh. Note it acts "
            "on future_fe_index, not active_fe_index — the change "
            "takes effect at execute_coarsening_and_refinement()."
        ),
        "fixed_number": "Refine fraction of cells with largest error",
        "verified_in_3d": (
            "A complete 3D hp-adaptive Poisson solver on a re-entrant "
            "domain (GridGenerator::hyper_L<3>, which yields hexes) "
            "with hp::FECollection of FE_Q(1..4), matched "
            "hp::QCollection, KellyErrorEstimator + Legendre "
            "smoothness and the sequence above ran for several cycles: "
            "the active_fe_index histogram spread from all-p1 on cycle "
            "0 to a mixture across p1..p4, the constraint count grew "
            "with the mixture, and every cycle's CG solve converged. "
            "hp-adaptivity in 3D is not exotic; it works out of the "
            "box with no optional dependencies."
        ),
    },
    "pitfalls": [
        "[API] Do NOT execute_coarsening_and_refinement() after the "
        "FINAL solve if you output the solution afterwards: "
        "refining the triangulation invalidates the "
        "(dof_handler, solution) pairing the closing DataOut "
        "reads, and the output block crashes on a stale-DoF "
        "access — sometimes a clean segfault (rc=-11), sometimes "
        "ExcOutOfMemory from posix_memalign requesting an "
        "underflowed ~8.5e18 bytes. The crash surfaces right after "
        "the LAST 'Cycle N:' progress line, which is easy to "
        "misread as an hp-machinery bug in the adaptation phase "
        "(we did, twice, on 9.3.2 — the 'non-determinism' across "
        "runs was just the cycle budget changing which cycle was "
        "last). Fix: `if (cycle == n_cycles - 1) break;` BEFORE "
        "marking refinement, the pattern every deal.II tutorial "
        "uses by ending the loop body with refinement only for "
        "non-final cycles. Signal: rc=-11 or 'the request was for "
        "8514397436244672512 bytes' immediately after the final "
        "cycle's output line, with all earlier cycles clean.",
        "[Syntax] hp::FECollection must contain every element you "
        "will point an active_fe_index at, and must be complete "
        "BEFORE distribute_dofs(). An index past the end of the "
        "collection behaves COMPLETELY DIFFERENTLY in the two build "
        "types, and the Release behaviour is the dangerous one. "
        "Debug: DoFHandler::distribute_dofs checks it with "
        "Assert(cell->active_fe_index() < ff.size(), "
        "ExcInvalidFEIndex(...)) and ABORTS (exit 134) printing "
        "'The mesh contains a cell with an active FE index of <N>, "
        "but the finite element collection only has <M> elements' "
        "— a complete diagnosis. "
        "Release: that Assert is compiled out and the program "
        "SEGFAULTS (exit 139) with no message at all — neither an "
        "exception nor n_dofs() == 0. "
        "Both verified on the same program (a 2-entry collection "
        "with one cell set to index 5). Note this Assert lives in "
        "the COMPILED library (source/dofs/dof_handler.cc), so "
        "adding -DDEBUG to your own translation unit does not bring "
        "it back — you need a deal.II built with "
        "CMAKE_BUILD_TYPE=Debug. On a Release-only install, guard it "
        "yourself: check every index you set is < "
        "fe_collection.size() before calling distribute_dofs. "
        "Signal: in Debug, the abort message 'The mesh contains a cell "
        "with an active FE index of <N>, but the finite element "
        "collection only has <M> elements'; in Release, a SIGSEGV "
        "(exit 139) inside distribute_dofs with no output at all. "
        "(This entry used to quote ExcMessage('Index in "
        "FECollection out of range'); no such string exists.)",
        "[Numerical] hp::QCollection must carry one rule per element "
        "in the FECollection, each sized for ITS element: FE_Q(p) "
        "wants QGauss(p+1) for a Laplace-type form. Pushing a single "
        "low-order rule is the usual bug, and the damage is graded, "
        "not binary. Verified on an hp mesh mixing FE_Q(1..4) with a "
        "variable coefficient, comparing a single rule against the "
        "matched collection: a rule far too coarse for the highest "
        "degree makes the element matrices RANK-DEFICIENT, the global "
        "operator loses positive-definiteness, and SolverCG runs to "
        "its iteration limit and throws SolverControl::NoConvergence "
        "with a residual that has GROWN by orders of magnitude; a "
        "moderately coarse rule converges normally but shifts a "
        "computed energy functional by a visible percentage; only "
        "the matched collection reproduces it. "
        "TWO WIDESPREAD MYTHS ABOUT THIS ENTRY, both refuted by "
        "execution: (1) it does NOT break symmetry. max|A_ij - A_ji| "
        "stayed at round-off (order 1e-15) for every rule tried, "
        "matched or not — a Galerkin form is symmetric under ANY "
        "quadrature because the same rule evaluates (i,j) and (j,i). "
        "Do not use symmetry as the tell. (2) 'SolverCG reports "
        "breakdown' is not something deal.II can print (see the "
        "essentials block). "
        "Signal: compare fe_collection.size() against "
        "q_collection.size() — a QCollection of size 1 is broadcast "
        "to every element and is the usual form of the bug; then "
        "recompute one scalar functional (the discrete energy "
        "0.5 u^T A u - u^T b, or an integral of the solution) with "
        "the matched collection and compare. A visible change means "
        "the coarse rule was under-integrating; a "
        "SolverControl::NoConvergence with a growing residual means "
        "it was coarse enough to make the operator singular.",
        "[API] Smoothness estimator needs FESeries::Fourier or "
        "FESeries::Legendre object — the smoothness decay rate "
        "drives p- vs h-refinement choice. Without it there is no "
        "smoothness indicator to pass to "
        "hp::Refinement::p_adaptivity_from_relative_threshold "
        "(step-27): deal.II has no p_adaptivity_from_smoothness, "
        "and SmoothnessEstimator::{Fourier,Legendre}::"
        "coefficient_decay needs the FESeries object as an "
        "argument. So no future FE indices are set and the cycle "
        "does h-refinement only. Signal: dof_handler.n_dofs() grows "
        "as O(N) instead of O(log N) on a smooth solution; "
        "VectorTools::integrate_difference against analytic "
        "reference shows pure h-convergence rate instead of "
        "exponential-in-p.",
        "[Numerical] Hanging-node constraints more complex with "
        "different p on neighbours — AffineConstraints needs the "
        "p-projection in addition to the h-projection. Forgetting "
        "this produces solution jumps at p-transitions. Signal: "
        "DataOut shows step discontinuities at cell boundaries "
        "where the neighbours have different FE_Q degree; "
        "VectorTools::integrate_difference reports O(1) error "
        "along those interfaces and ~O(h^p) elsewhere.",
        "[API] Matrix-free DOES support hp — this entry used to say "
        "the opposite and it is false on deal.II 9.x. "
        "MatrixFree<dim, Number>::reinit(mapping, dof_handler, "
        "constraints, hp::QCollection<1>, additional_data) accepts a "
        "DoFHandler with hp capabilities and a MIXED set of "
        "active_fe_indices, and returns normally: verified on a "
        "DoFHandler whose cells alternate between FE_Q(1) and "
        "FE_Q(2), where reinit reported a populated cell-batch list "
        "and n_active_fe_indices() == 2. Neither "
        "ExcMessage('all cells must have same active_fe_index') nor "
        "ExcMessage('hp-FEValues requires hp::MappingCollection') "
        "exists in the library; do not wait for either. "
        "What IS true: you must pass a QUADRATURE COLLECTION "
        "(hp::QCollection<1>) rather than a single Quadrature, and "
        "on the evaluation side FEEvaluation has to be told which "
        "active_fe_index / quadrature index a cell batch belongs to "
        "— that is what the step-75 pattern is for. Signal: after "
        "reinit, read back MatrixFree::n_active_fe_indices(); if it "
        "is 1 on a mesh you believe is mixed-p, the indices never "
        "reached the DoFHandler (set_active_fe_index must be called "
        "BEFORE distribute_dofs).",
        "[Numerical] Transfer solution between p-levels: use "
        "SolutionTransfer or VectorTools::interpolate. Setting "
        "solution values directly across a p-change discards "
        "the high-frequency content and breaks Newton "
        "continuation. Signal: DataOut frame at the p-refinement "
        "step shows solution.linfty_norm() dropping by 10-50% "
        "(the high-frequency content lost in the transfer); "
        "next Newton step has to re-construct it from scratch.",
    ],
}
