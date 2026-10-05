"""MPI-parallel Poisson solver templates for deal.II.

Based on deal.II tutorial step-40 (p4est distributed mesh).
"""


def _parallel_poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.
    All parameter defaults are placeholders.
    MPI-parallel Poisson solver with p4est — based on step-40 pattern.
    """
    refinements = params.get("refinements", 5)
    degree = params.get("degree", 2)
    rhs_value = params.get("rhs_value", 1.0)
    return f'''\
/* MPI-parallel Poisson equation — based on deal.II step-40 pattern
 * Solves -laplacian(u) = f using PETSc/Trilinos on a distributed mesh (p4est).
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/lac/generic_linear_algebra.h>

// ── FEATURE DISPATCH ─────────────────────────────────────────────
// step-40's distributed mesh + PETSc/Trilinos linear algebra exist
// ONLY if the library was built with MPI + PETSc (or Trilinos) +
// p4est. This template therefore compiles in TWO modes from the SAME
// source, chosen by the flags in deal.II/base/config.h:
//
//   DISTRIBUTED  (MPI + PETSc + p4est ON) — the real step-40 program:
//                parallel::distributed::Triangulation, LA::MPI types,
//                PreconditionAMG, one .vtu per rank plus a .pvtu.
//   SERIAL-SHAPED (any of them OFF) — the identical algorithm on a
//                plain Triangulation with deal.II's built-in linear
//                algebra. It really runs, on one rank. Every MPI
//                IDIOM is kept (MPI_InitFinalize, locally_owned /
//                locally_relevant IndexSets, is_locally_owned()
//                gating, compress(), write_vtu_with_pvtu_record) so
//                the code is a correct starting point for a build
//                that does have MPI; it simply does not decompose.
//
// It used to be a hard #error. That was wrong: the serial-shaped
// variant below was compiled and run against a deal.II with MPI,
// PETSc, Trilinos and p4est all OFF and solved the problem.
#if defined(DEAL_II_WITH_MPI) && defined(DEAL_II_WITH_P4EST) && \
    (defined(DEAL_II_WITH_PETSC) || defined(DEAL_II_WITH_TRILINOS))
#  define OFA_DEALII_DISTRIBUTED 1
#else
#  define OFA_DEALII_DISTRIBUTED 0
#endif

namespace LA
{{
#if OFA_DEALII_DISTRIBUTED
#  if defined(DEAL_II_WITH_PETSC)
  using namespace dealii::LinearAlgebraPETSc;
#  else
  using namespace dealii::LinearAlgebraTrilinos;
#  endif
#else
  // Built-in serial linear algebra. Same class names, no MPI:: layer.
  using namespace dealii::LinearAlgebraDealII;
#endif
}}

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/lac/precondition.h>
#if OFA_DEALII_DISTRIBUTED
#  include <deal.II/distributed/tria.h>
#  include <deal.II/distributed/grid_refinement.h>
#endif
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
    double val = {rhs_value};
    for (unsigned int d = 0; d < dim; ++d)
      val *= std::sin(numbers::PI * p[d]);
    return val * dim * numbers::PI * numbers::PI;
  }}
}};

int main(int argc, char *argv[])
{{
  const unsigned int dim = 2;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const MPI_Comm mpi_communicator = MPI_COMM_WORLD;

  ConditionalOStream pcout(std::cout,
                            Utilities::MPI::this_mpi_process(mpi_communicator) == 0);

  pcout << "Running on "
        << Utilities::MPI::n_mpi_processes(mpi_communicator)
        << " MPI rank(s)" << std::endl;

#if OFA_DEALII_DISTRIBUTED
  // Distributed triangulation via p4est
  parallel::distributed::Triangulation<dim> triangulation(mpi_communicator);
#else
  // Serial-shaped fallback: same API, one rank, no decomposition.
  Triangulation<dim> triangulation;
#endif
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global({refinements});

  const unsigned int degree = {degree};
  FE_Q<dim>       fe(degree);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler);

  pcout << "Parallel Poisson: " << dof_handler.n_dofs() << " DOFs, "
#if OFA_DEALII_DISTRIBUTED
        << triangulation.n_global_active_cells()
#else
        << triangulation.n_active_cells()
#endif
        << " cells" << std::endl;

  // Constraints
  AffineConstraints<double> constraints;
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(dof_handler, 0,
                                            Functions::ZeroFunction<dim>(),
                                            constraints);
  constraints.close();

  // Sparsity pattern
  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
#if OFA_DEALII_DISTRIBUTED
  SparsityTools::distribute_sparsity_pattern(dsp,
                                              dof_handler.locally_owned_dofs(),
                                              mpi_communicator,
                                              locally_relevant_dofs);
  LA::MPI::SparseMatrix system_matrix;
  system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
  LA::MPI::Vector solution;
  solution.reinit(locally_owned_dofs, mpi_communicator);
  LA::MPI::Vector system_rhs;
  system_rhs.reinit(locally_owned_dofs, mpi_communicator);
#else
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);
  LA::SparseMatrix system_matrix(sparsity_pattern);
  LA::Vector solution(dof_handler.n_dofs());
  LA::Vector system_rhs(dof_handler.n_dofs());
#endif

  // Assembly
  const QGauss<dim> quadrature(degree + 1);
  FEValues<dim>     fe_values(fe, quadrature,
                               update_values | update_gradients |
                               update_quadrature_points | update_JxW_values);

  const RightHandSide<dim> rhs_function;

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {{
        const unsigned int dpc = fe.n_dofs_per_cell();
        FullMatrix<double> cell_matrix(dpc, dpc);
        Vector<double>     cell_rhs(dpc);
        std::vector<types::global_dof_index> local_dof_indices(dpc);

        fe_values.reinit(cell);

        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
          {{
            const double rhs_val = rhs_function.value(
              fe_values.quadrature_point(q), 0);
            for (unsigned int i = 0; i < dpc; ++i)
              {{
                for (unsigned int j = 0; j < dpc; ++j)
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

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  // Solve. Tolerance RELATIVE to ||b||: an absolute 1e-12 is
  // unreachable noise on a large problem.
  SolverControl solver_control(1000, 1e-10 * system_rhs.l2_norm());
#if OFA_DEALII_DISTRIBUTED
  LA::SolverCG  solver(solver_control, mpi_communicator);
  LA::MPI::PreconditionAMG preconditioner;
  LA::MPI::PreconditionAMG::AdditionalData amg_data;
  preconditioner.initialize(system_matrix, amg_data);
#else
  SolverCG<LA::Vector> solver(solver_control);
  PreconditionSSOR<LA::SparseMatrix> preconditioner;
  preconditioner.initialize(system_matrix);
#endif
  solver.solve(system_matrix, solution, system_rhs, preconditioner);

  pcout << "Solved in " << solver_control.last_step() << " iterations"
        << std::endl;

  constraints.distribute(solution);

  // Output: each rank writes its part, the .pvtu indexes them all.
  // ParaView must open the .pvtu, not a per-rank .vtu.
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
#if OFA_DEALII_DISTRIBUTED
  LA::MPI::Vector locally_relevant_solution;
  locally_relevant_solution.reinit(locally_owned_dofs,
                                    locally_relevant_dofs,
                                    mpi_communicator);
  locally_relevant_solution = solution;
  data_out.add_data_vector(locally_relevant_solution, "solution");
  Vector<float> subdomain(triangulation.n_active_cells());
  for (auto &val : subdomain)
    val = static_cast<float>(triangulation.locally_owned_subdomain());
#else
  data_out.add_data_vector(solution, "solution");
  Vector<float> subdomain(triangulation.n_active_cells());
  subdomain = static_cast<float>(
    Utilities::MPI::this_mpi_process(mpi_communicator));
#endif
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record("./", "result", 0, mpi_communicator);

  pcout << "Parallel Poisson: output written" << std::endl;
  return 0;
}}
'''


# ── Knowledge ────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "description": "MPI-parallel Poisson with p4est distributed mesh (step-40)",
    "tutorial_steps": ["step-40 (basic parallel Poisson)", "step-50 (parallel GMG)",
                      "step-75 (parallel hp-multigrid, matrix-free)"],
    "function_space": "FE_Q<dim>(p) with parallel::distributed::Triangulation",
    "solver": "PETSc CG + AMG (BoomerAMG), or Trilinos CG + ML/MueLu",
    "parallel_infrastructure": {
        "mesh": "parallel::distributed::Triangulation (p4est backend)",
        "vectors": "LA::MPI::Vector (PETSc or Trilinos)",
        "matrix": "LA::MPI::SparseMatrix (PETSc or Trilinos)",
        "output": "DataOut::write_vtu_with_pvtu_record for parallel VTU",
    },
    "pitfalls": [
        "[Integration] The step-40 distributed pattern requires the "
        "LIBRARY to have been built with MPI + p4est + PETSc (or "
        "Trilinos). Probe it in "
        "$DEAL_II_DIR/include/deal.II/base/config.h, NOT by trying "
        "an include: on a source install every header ships anyway. "
        "Signal: with p4est OFF the failure is a COMPILE-time error "
        "and nothing else — parallel::distributed::Triangulation's "
        "constructor is `= delete`d, so what you see is the "
        "COMPILER's own wording — g++ says use of deleted function, "
        "then reprints the signature it resolved to, e.g. "
        "dealii::parallel::distributed::Triangulation<dim, spacedim>"
        "::Triangulation(dealii::MPI_Comm, ...). None of that text "
        "is deal.II's and none of it is in the library. "
        "There is no link error and no runtime "
        "exception to catch. Two things that still WORK with MPI off "
        "and will fool a naive probe: "
        "'#include <deal.II/distributed/tria.h>' compiles cleanly, "
        "and Utilities::MPI::MPI_InitFinalize constructs and reports "
        "n_mpi_processes == 1. "
        "YOU DO NOT NEED TO ABANDON THE PATTERN. Verified by "
        "compiling and running on a library with MPI, PETSc, "
        "Trilinos and p4est ALL OFF: the same source solves the "
        "problem if you (a) swap parallel::distributed::"
        "Triangulation for a plain Triangulation and (b) swap "
        "'namespace LA { using namespace "
        "dealii::LinearAlgebraPETSc; }' for "
        "'namespace LA { using namespace "
        "dealii::LinearAlgebraDealII; }'. Everything else — "
        "MPI_InitFinalize, locally_owned_dofs(), "
        "DoFTools::extract_locally_relevant_dofs, "
        "cell->is_locally_owned() gating, compress(), "
        "DataOut::write_vtu_with_pvtu_record — compiles and runs "
        "unchanged on one rank. Guard the two swaps with "
        "'#if defined(DEAL_II_WITH_MPI) && defined(DEAL_II_WITH_P4EST) "
        "&& (defined(DEAL_II_WITH_PETSC) || "
        "defined(DEAL_II_WITH_TRILINOS))' and one source serves both "
        "builds; this catalog's parallel_poisson template does "
        "exactly that.",
        "[Syntax] Only locally-owned cells are assembled. Use "
        "`cell->is_locally_owned()` to gate the assembly loop. "
        "Signal: assembly runs on every rank but "
        "`system_matrix.frobenius_norm()` is the SAME on rank 0 "
        "and rank 1 — each rank assembled the full mesh, leading "
        "to double-counted entries; Triangulation::n_locally_"
        "owned_active_cells() returns the full count instead of "
        "the per-rank partition.",
        "[API] Locally-relevant DoFs are needed for the GHOSTED "
        "vector you read from (output, error estimation, nonlinear "
        "residuals); the vector you SOLVE into must be "
        "non-ghosted. Build the index sets with "
        "dof_handler.locally_owned_dofs() and "
        "DoFTools::extract_locally_relevant_dofs(dof_handler), and "
        "reinit the read-only copy with BOTH. "
        "Signal: this entry used to quote "
        "ExcMessage('ghost entries not consistent'); that string "
        "does not exist in deal.II — checked against the library's "
        "own string table. The real diagnostics are: writing into a "
        "vector that HAS ghost elements trips "
        "Assert(..., ExcGhostsPresent()), declared in "
        "base/exceptions.h (DeclExceptionMsg at line 663 in deal.II "
        "9.8.0, 668 in 9.7.1); its first three adjacent literals, of "
        "which the first is 'You are trying an operation on a vector "
        "that is only' and the last 'vector you are operating on "
        "does have ghost elements.', read as: You are trying an "
        "operation on a vector that is only allowed if the vector "
        "has no ghost elements, but the vector you are operating on "
        "does have ghost elements. Two more paragraphs follow: "
        "which operations a vector with ghost elements does not "
        "allow, and a pointer to the glossary entry on ghosted "
        "vectors. And reading an element a "
        "fully-distributed vector does not hold gives 'You tried to "
        "access element <i> of a distributed vector, but this "
        "element is not stored on the current processor.', followed "
        "by the locally-owned range and the advice, streamed piece "
        "by piece at lac/la_parallel_vector.h:1274-1276 in 9.8.0 "
        "(1308-1310 in 9.7.1), that you "
        "are passing a 'fully distributed' vector into a function "
        "'that needs read access to vector elements that correspond' "
        "'to degrees of freedom on ghost cells (or at least to' "
        "locally active degrees of freedom that are not also locally "
        "owned. "
        "SCOPE: both are Assert-guarded, so they exist only in a "
        "DEBUG build; in Release the same misuse is silent or "
        "crashes. And both are MPI-only paths, so neither can be "
        "reproduced on a build without MPI — the strings above were "
        "read out of the library, the behaviour was not executed "
        "here.",
        "[Syntax] SparsityTools::distribute_sparsity_pattern "
        "needed for parallel sparsity. Skipping it produces a "
        "globally-replicated sparsity (each rank holds all rows) "
        "and matrix assembly OOMs at scale. Signal: rss / "
        "VmPeak from /proc/self/status grows linearly with "
        "n_dofs on each MPI rank instead of being O(n_dofs / "
        "n_ranks); SparseMatrix::memory_consumption() exceeds "
        "the expected per-rank slice.",
        "[Syntax] `compress(VectorOperation::add)` required after "
        "assembly to sum contributions across ranks. Without it, "
        "system_rhs holds only this rank's contribution. Signal: "
        "DataOut shows the global solution split into per-rank "
        "regions with mismatched values at MPI subdomain "
        "interfaces; VectorTools::integrate_difference against "
        "the serial reference is O(1) at the interfaces and 0 "
        "elsewhere.",
        "[Numerical] PreconditionAMG / BoomerAMG works out-of-box "
        "for scalar Laplace via TrilinosWrappers but needs tuning "
        "(strong-threshold, smoother) for elasticity / NS / "
        "multiphysics systems. Signal: SolverCG iteration count "
        "from SolverControl::last_step() grows from O(20) on "
        "Laplace to >500 on the same mesh size for elasticity, "
        "with each iteration cost dominated by AMG-apply.",
        "[Integration] Output: write_vtu_with_pvtu_record "
        "produces one .vtu per rank plus one .pvtu index file. "
        "ParaView opens the .pvtu, NOT the per-rank .vtu — opening "
        "the latter shows only that rank's subdomain. Signal: "
        "ParaView shows fragmentary mesh (only one MPI subdomain), "
        "while reading the .pvtu yields the full distributed "
        "result; DataOut::write_vtu (no pvtu_record) produces "
        "the broken-into-pieces output.",
    ],
}
