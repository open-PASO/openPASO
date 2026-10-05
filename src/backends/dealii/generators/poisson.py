"""Poisson / Laplace equation templates for deal.II.

Based on deal.II tutorial steps 3, 5, 6.
"""


def _poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    Based on deal.II step-3.
    """
    degree = int(params.get("degree", params.get("order", 1)))
    if degree < 1:
        raise ValueError(
            f"_poisson_2d: degree must be >= 1, got {degree!r}")
    refinements = params.get("refinements", 5)
    return f'''\
/* Poisson equation on unit square — based on deal.II step-3
 * -laplacian(u) = 1 on [0,1]^2, u = 0 on boundary
 */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iostream>

using namespace dealii;

int main()
{{
  Triangulation<2> triangulation;
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global({refinements});

  FE_Q<2> fe({degree});
  DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  std::cout << "Number of DOFs: " << dof_handler.n_dofs() << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> system_matrix;
  system_matrix.reinit(sparsity_pattern);

  Vector<double> solution;
  solution.reinit(dof_handler.n_dofs());
  Vector<double> system_rhs;
  system_rhs.reinit(dof_handler.n_dofs());

  // Assemble
  QGauss<2> quadrature_formula(fe.degree + 1);
  // every flag this loop reads: a Release-only deal.II build segfaults, with no message, on a missing one
  FEValues<2> fe_values(fe, quadrature_formula,
                        update_values | update_gradients | update_JxW_values | update_quadrature_points);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {{
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;

      for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {{
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                   fe_values.shape_grad(j, q) *
                                   fe_values.JxW(q);
            cell_rhs(i) += fe_values.shape_value(i, q) * 1.0 *
                           fe_values.JxW(q);
          }}

      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {{
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }}
    }}

  // Boundary conditions
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<2>(),
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);

  // Solve
  SolverControl solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  std::cout << "Solver converged in " << solver_control.last_step()
            << " iterations." << std::endl;
  std::cout << "min(u) = " << *std::min_element(solution.begin(), solution.end())
            << ", max(u) = " << *std::max_element(solution.begin(), solution.end())
            << std::endl;

  // Output
  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();

  std::ofstream output("result.vtu");
  data_out.write_vtu(output);

  std::cout << "Output written to result.vtu" << std::endl;
  return 0;
}}
'''


def _poisson_3d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    Based on deal.II step-3.
    """
    degree = int(params.get("degree", params.get("order", 1)))
    if degree < 1:
        raise ValueError(
            f"_poisson_3d: degree must be >= 1, got {degree!r}")
    refinements = params.get("refinements", 3)
    return f'''\
/* Poisson equation on unit cube — based on deal.II step-3
 * -laplacian(u) = 1 on [0,1]^3, u = 0 on boundary
 */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iostream>

using namespace dealii;

int main()
{{
  Triangulation<3> triangulation;
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global({refinements});

  FE_Q<3> fe({degree});
  DoFHandler<3> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  std::cout << "Number of DOFs: " << dof_handler.n_dofs() << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> system_matrix;
  system_matrix.reinit(sparsity_pattern);

  Vector<double> solution;
  solution.reinit(dof_handler.n_dofs());
  Vector<double> system_rhs;
  system_rhs.reinit(dof_handler.n_dofs());

  // Assemble
  QGauss<3> quadrature_formula(fe.degree + 1);
  // every flag this loop reads: a Release-only deal.II build segfaults, with no message, on a missing one
  FEValues<3> fe_values(fe, quadrature_formula,
                        update_values | update_gradients | update_JxW_values | update_quadrature_points);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {{
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;

      for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {{
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                   fe_values.shape_grad(j, q) *
                                   fe_values.JxW(q);
            cell_rhs(i) += fe_values.shape_value(i, q) * 1.0 *
                           fe_values.JxW(q);
          }}

      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {{
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }}
    }}

  // Boundary conditions
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<3>(),
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);

  // Solve
  SolverControl solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  std::cout << "Solver converged in " << solver_control.last_step()
            << " iterations." << std::endl;
  std::cout << "min(u) = " << *std::min_element(solution.begin(), solution.end())
            << ", max(u) = " << *std::max_element(solution.begin(), solution.end())
            << std::endl;

  // Output
  DataOut<3> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();

  std::ofstream output("result.vtu");
  data_out.write_vtu(output);

  std::cout << "Output written to result.vtu" << std::endl;
  return 0;
}}
'''


def _poisson_l_domain(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    Uses deal.II built-in GridGenerator::hyper_L.
    """
    degree = int(params.get("degree", params.get("order", 1)))
    if degree < 1:
        raise ValueError(
            f"_poisson_l_domain: degree must be >= 1, got {degree!r}")
    refinements = params.get("refinements", 5)
    return f'''\
/* Poisson on L-shaped domain — deal.II
 * -laplacian(u) = 1, u = 0 on boundary
 * Non-trivial geometry with re-entrant corner singularity.
 * Uses built-in GridGenerator::hyper_L (no external mesher needed).
 */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iostream>

using namespace dealii;

int main()
{{
  Triangulation<2> triangulation;
  GridGenerator::hyper_L(triangulation, -1, 1);
  triangulation.refine_global({refinements});

  FE_Q<2> fe({degree});
  DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  std::cout << "L-domain DOFs: " << dof_handler.n_dofs() << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> system_matrix;
  system_matrix.reinit(sparsity_pattern);
  Vector<double> solution(dof_handler.n_dofs());
  Vector<double> system_rhs(dof_handler.n_dofs());

  QGauss<2> quadrature(fe.degree + 1);
  // every flag this loop reads: a Release-only deal.II build segfaults, with no message, on a missing one
  FEValues<2> fe_values(fe, quadrature,
    update_values | update_gradients | update_JxW_values | update_quadrature_points);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {{
      fe_values.reinit(cell);
      cell_matrix = 0; cell_rhs = 0;
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {{
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                   fe_values.shape_grad(j, q) *
                                   fe_values.JxW(q);
            cell_rhs(i) += fe_values.shape_value(i, q) * 1.0 * fe_values.JxW(q);
          }}
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {{
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }}
    }}

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler, 0,
    Functions::ZeroFunction<2>(), boundary_values);
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

  SolverControl solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  std::cout << "Solver: " << solver_control.last_step() << " iterations" << std::endl;
  std::cout << "min(u) = " << *std::min_element(solution.begin(), solution.end())
            << ", max(u) = " << *std::max_element(solution.begin(), solution.end()) << std::endl;

  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);
  std::cout << "Output written to result.vtu" << std::endl;
  return 0;
}}
'''


def _poisson_rectangle(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    degree = int(params.get("degree", params.get("order", 1)))
    if degree < 1:
        raise ValueError(
            f"_poisson_rectangle: degree must be >= 1, got {degree!r}")
    refinements = params.get("refinements", 5)
    lx = params.get("lx", 2.0)
    ly = params.get("ly", 1.0)
    return f'''\
/* Poisson on [{lx}x{ly}] rectangle — deal.II
 * -laplacian(u) = 1, u = 0 on boundary
 */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iostream>

using namespace dealii;

int main()
{{
  Triangulation<2> triangulation;
  GridGenerator::subdivided_hyper_rectangle(triangulation,
    {{(unsigned int)({int(lx * 8)}), (unsigned int)({int(ly * 8)})}},
    Point<2>(0, 0), Point<2>({lx}, {ly}));
  triangulation.refine_global({refinements});

  FE_Q<2> fe({degree});
  DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  std::cout << "DOFs: " << dof_handler.n_dofs() << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> system_matrix;
  system_matrix.reinit(sparsity_pattern);
  Vector<double> solution(dof_handler.n_dofs());
  Vector<double> system_rhs(dof_handler.n_dofs());

  QGauss<2> quadrature(fe.degree + 1);
  // every flag this loop reads: a Release-only deal.II build segfaults, with no message, on a missing one
  FEValues<2> fe_values(fe, quadrature,
    update_values | update_gradients | update_JxW_values | update_quadrature_points);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {{
      fe_values.reinit(cell);
      cell_matrix = 0; cell_rhs = 0;
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {{
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                   fe_values.shape_grad(j, q) *
                                   fe_values.JxW(q);
            cell_rhs(i) += fe_values.shape_value(i, q) * 1.0 * fe_values.JxW(q);
          }}
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {{
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }}
    }}

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler, 0,
    Functions::ZeroFunction<2>(), boundary_values);
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

  SolverControl solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  std::cout << "min(u) = " << *std::min_element(solution.begin(), solution.end())
            << ", max(u) = " << *std::max_element(solution.begin(), solution.end()) << std::endl;

  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);
  return 0;
}}
'''


def _poisson_adaptive_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    Based on deal.II step-6.
    """
    cycles = params.get("cycles", 6)
    order = params.get("order", 2)
    # Pre-refinement before the adaptive loop. Must be >= 1: a
    # fixed-FRACTION refinement strategy flags floor(fraction *
    # n_active_cells) cells, which is zero on a 3-cell coarse mesh.
    pre_refinements = int(params.get("pre_refinements", 2))
    if pre_refinements < 1:
        raise ValueError(
            "poisson_2d_adaptive: pre_refinements must be >= 1, "
            "otherwise refine_and_coarsen_fixed_number flags zero "
            f"cells on the 3-cell coarse mesh (got {pre_refinements!r})")
    return f'''\
/* Poisson with AMR — step-6 based — deal.II */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <fstream>
#include <iostream>
using namespace dealii;

int main() {{
  const int dim = 2;
  Triangulation<dim> tria;
  GridGenerator::hyper_L(tria, -1, 1);
  // REQUIRED: refine at least once before the adaptive loop.
  // hyper_L gives 3 cells, and refine_and_coarsen_fixed_number(0.3, ...)
  // computes 0.3*3 = 0.9 -> floors to ZERO cells flagged, so
  // execute_coarsening_and_refinement() is a no-op and every cycle
  // recomputes the identical answer without any warning. Verified:
  // 3 cells -> 0 flagged; after one refine_global, 12 cells -> 3 flagged.
  tria.refine_global({pre_refinements});

  FE_Q<dim> fe({order});
  DoFHandler<dim> dof_handler(tria);

  for (unsigned int cycle = 0; cycle < {cycles}; ++cycle) {{
    dof_handler.distribute_dofs(fe);

    AffineConstraints<double> constraints;
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler, 0,
      Functions::ZeroFunction<dim>(), constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
    SparsityPattern sp;
    sp.copy_from(dsp);

    SparseMatrix<double> system_matrix;
    system_matrix.reinit(sp);
    Vector<double> solution(dof_handler.n_dofs());
    Vector<double> system_rhs(dof_handler.n_dofs());

    QGauss<dim> quadrature(fe.degree + 1);
    // every flag this loop reads: a Release-only deal.II build segfaults, with no message, on a missing one
    FEValues<dim> fe_values(fe, quadrature,
      update_values | update_gradients | update_JxW_values | update_quadrature_points);

    const unsigned int dpc = fe.n_dofs_per_cell();
    FullMatrix<double> cell_matrix(dpc, dpc);
    Vector<double> cell_rhs(dpc);
    std::vector<types::global_dof_index> local_dof_indices(dpc);

    for (const auto &cell : dof_handler.active_cell_iterators()) {{
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs = 0;
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        for (unsigned int i = 0; i < dpc; ++i) {{
          for (unsigned int j = 0; j < dpc; ++j)
            cell_matrix(i, j) += fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q)
                                 * fe_values.JxW(q);
          cell_rhs(i) += 1.0 * fe_values.shape_value(i, q) * fe_values.JxW(q);
        }}
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices,
                                             system_matrix, system_rhs);
    }}

    SolverControl sc(1000, 1e-12);
    SolverCG<Vector<double>> solver(sc);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    constraints.distribute(solution);

    std::cout << "Cycle " << cycle << ": " << dof_handler.n_dofs() << " DOFs, "
              << sc.last_step() << " CG iters, max(u)=" << solution.linfty_norm() << std::endl;
    {{
      DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(solution, "u");
      data_out.build_patches();
      std::ofstream out("solution.vtu");
      data_out.write_vtu(out);
    }}

    // Error estimation and refinement
    Vector<float> error_per_cell(tria.n_active_cells());
    KellyErrorEstimator<dim>::estimate(dof_handler,
      QGauss<dim - 1>(fe.degree + 1), {{}}, solution, error_per_cell);
    GridRefinement::refine_and_coarsen_fixed_number(tria, error_per_cell, 0.3, 0.03);
    tria.execute_coarsening_and_refinement();
  }}

  // Final output: the SOLUTION on the last solved mesh. The previous version
  // refined once more after the last solve and then wrote a DataOut with no
  // data vector attached -- a mesh-only file called solution.vtu, with no
  // field in it. The solution is written inside the loop now, right after
  // each solve, so the file always holds the field of the mesh it was solved
  // on; this block only names the final mesh.
  std::cout << "Final mesh: " << tria.n_active_cells() << " cells" << std::endl;
  std::cout << "AMR complete." << std::endl;
  return 0;
}}
'''


# ── Knowledge ────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "description": "Poisson equation solved with deal.II (step-3/4/5/6/7)",
    "tutorial_steps": ["step-3 (basic)", "step-5 (variable coefficients)", "step-6 (AMR)",
                      "step-37 (matrix-free)", "step-40 (parallel)"],
    "function_space": "FE_Q<dim>(p) — Lagrange, any order p",
    "solver": "CG + SSOR/AMG. Matrix-free: MatrixFree + FEEvaluation (step-37)",
    "adaptive_refinement": "KellyErrorEstimator + refine_and_coarsen_fixed_number (step-6)",
    # ── Structured keys for Poisson — the canonical first physics.
    #    GENERAL_KNOWLEDGE in this same module enumerates the full
    #    H1/H1_enriched/nonconforming/H_div/H_curl menu; this list
    #    is the Poisson-relevant subset only.
    "elements": {
        "FE_Q":
            "Canonical Poisson choice. The poisson_* and heat_* "
            "templates take degree=<k> (or order=<k>) and emit "
            "FE_Q<dim>(k); degree=1 is the default, degree=2 is the "
            "cheapest real accuracy gain, and higher degrees pay off "
            "on smooth problems. Verified by running the generated "
            "programs at degree 1, 2 and 3 in 2D and 3D: all "
            "compile, solve, and approach the analytic peak value of "
            "the reference problem more closely as the degree rises. "
            "NOTE the CG iteration count grows with degree at fixed "
            "refinement (the condition number does), so raise the "
            "SolverControl budget or precondition better when you "
            "raise k.",
        "FE_Q_Hierarchical":
            "Hierarchical basis, so coarse-level modes survive a "
            "degree change — the usual choice for p-adaptive "
            "Poisson. WARNING, verified by execution: "
            "has_support_points() is FALSE at EVERY degree, including "
            "1 (the element has only generalized support points), and "
            "VectorTools::interpolate_boundary_values then SEGFAULTS "
            "(exit 139 on Release; a Debug build aborts with 'You "
            "are trying to access the support points of a finite "
            "element that either has no support points at all...'). "
            "Unlike FE_Q, even the first-order element is affected. "
            "Use VectorTools::project_boundary_values instead, or "
            "guard on fe.has_support_points().",
        "FE_Bernstein":
            "Positive, partition-of-unity basis; for high-p Poisson "
            "where mass-matrix conditioning matters (modal analysis, "
            "transient diffusion). SAME WARNING as "
            "FE_Q_Hierarchical, verified by execution: "
            "has_support_points() is FALSE at every degree, including "
            "1 (and it has no generalized support points either), so "
            "interpolate_boundary_values SEGFAULTS. Use "
            "project_boundary_values.",
        "FE_Q_iso_Q1":
            "Cheap multi-linear-on-sub-cells alternative to "
            "FE_Q(p); diagonal lumped mass matrix.",
        "FE_DGQ":
            "Discontinuous Galerkin Poisson via interior-penalty "
            "formulation; needed when coefficients are "
            "discontinuous across cells (heterogeneous media). Two "
            "consequences you must handle, both verified: the "
            "sparsity pattern has to come from "
            "DoFTools::make_flux_sparsity_pattern (the cell-only "
            "builder drops every face coupling), and "
            "interpolate_boundary_values is a silent NO-OP on it — "
            "it returns an EMPTY map — so Dirichlet data must be "
            "imposed weakly through the penalty term.",
        "FE_DGP":
            "Complete-polynomial P_k DG space with an L2-orthonormal "
            "Legendre basis (FE_DGPMonomial is the monomial variant); "
            "alternative to FE_DGQ for higher-"
            "order accurate Poisson on hyper-cube meshes.",
        "FE_SimplexP":
            "Lagrange on simplex (triangle / tet) cells — needed "
            "when the mesh comes from unstructured Gmsh / Triangle / "
            "TetGen. Available in deal.II >= 9.3. It is NOT a "
            "drop-in for FE_Q: the quadrature must become "
            "QGaussSimplex and an explicit "
            "MappingFE(FE_SimplexP(1)) must be passed to FEValues, "
            "VectorTools::*, KellyErrorEstimator and DataOut. See "
            "the simplex section of the essentials block for the "
            "one check (summed JxW against the domain volume) that "
            "catches all three mistakes.",
    },
    "mesh_generators": {
        "hyper_cube": "Classic Poisson on the unit square / cube.",
        "hyper_rectangle": "Non-square aspect ratio.",
        "subdivided_hyper_cube": "Pre-subdivided to avoid repeated refine_global() calls.",
        "hyper_L": "Re-entrant-corner singularity; canonical adaptive-refinement test.",
        "hyper_ball": ("Curved boundary. The generator attaches a "
                       "SphericalManifold for you - do NOT call "
                       "reset_all_manifolds(), and raise the mapping "
                       "degree (MappingQ<dim>(2) or (3)) or the "
                       "geometry error dominates the FE error. "
                       "Verified: with no manifold the integrated "
                       "volume is stuck at the straight-edged value "
                       "and refinement does not change it at all."),
        "hyper_shell": ("Annulus / spherical shell for layered radial "
                        "problems. Same manifold and mapping-degree "
                        "caveat as hyper_ball. Its colorize argument "
                        "gives inner boundary_id 0 and outer 1, which "
                        "is what a two-sided radial BC needs."),
        "cheese": "Heterogeneous-coefficient demos.",
        "torus": ("GridGenerator::torus is dimension-overloaded and "
                  "both forms were verified to build: "
                  "Triangulation<3> gives a SOLID torus of hexes, "
                  "Triangulation<2,3> gives its SURFACE as a 2D mesh "
                  "embedded in 3D. Pick deliberately - the surface "
                  "variant needs spacedim-aware FEValues and suits "
                  "Laplace-Beltrami style problems, not a volume "
                  "PDE."),
    },
    "solvers": [
        "SolverCG<>                   — Poisson is symmetric positive-definite; CG is the default",
        "SolverGMRES<>                — only needed if coefficients break symmetry (e.g. when stabilisation is added)",
        "MatrixFree + FEEvaluation    — step-37 matrix-free; needed for matrix-storage-bound problems past ~10^7 DoFs",
    ],
    "preconditioners": [
        "PreconditionSSOR             — serial default; cheap on SPD systems. "
        "The SYMMETRIC sweep, so it is legal with SolverCG. Its one-sided "
        "sibling PreconditionSOR is NOT: verified, SolverCG + PreconditionSOR "
        "runs to its iteration limit and throws SolverControl::NoConvergence "
        "on an SPD Poisson system that SolverGMRES + PreconditionSOR solves "
        "in a few tens of steps",
        "PreconditionBlockSSOR / BlockJacobi — only when the diagonal blocks "
        "are genuine sub-problems, i.e. a DISCONTINUOUS element with "
        "block_size = fe.n_dofs_per_cell(). On a continuous FE_Q Poisson "
        "system a block of constrained rows can be exactly SINGULAR; "
        "initialize() then returns silently and the preconditioner is worse "
        "than none (verified: one of 72 blocks singular, one-application "
        "relative residual above 1)",
        "SparseILU / SparseMIC        — incomplete LU / modified incomplete "
        "Cholesky, both verified to work with SolverCG on Poisson; SparseMIC "
        "is NOT PreconditionICC - that name is PETScWrappers::PreconditionICC, "
        "a separate PETSc incomplete-Cholesky wrapper "
        "(TrilinosWrappers::PreconditionIC is the Trilinos one)",
        "PreconditionAMG / BoomerAMG  — parallel AMG: TrilinosWrappers::PreconditionAMG (Trilinos ML; "
        "PreconditionAMGMueLu for MueLu) or PETScWrappers::PreconditionBoomerAMG (hypre via PETSc); "
        "LA::MPI::PreconditionAMG (step-40) picks one per backend; scales to 10^7 DoFs",
        "PreconditionChebyshev        — used inside multigrid as smoother, also as a standalone for matrix-free",
        "MGSmootherRelaxation         — geometric multigrid smoother (step-16, step-50)",
    ],
    "pitfalls": [
        "[Syntax] Do all refinement BEFORE distribute_dofs(). "
        "Calling distribute_dofs on an unrefined one-cell "
        "triangulation succeeds and produces a system with no "
        "interior at all. Verified on a one-cell unit square with "
        "FE_Q(1) and Dirichlet data on every face: n_dofs() == 4 "
        "and n_constraints() == 4 — EVERY degree of freedom is a "
        "boundary DoF — so the assembled right-hand side is exactly "
        "0, CG 'converges' at step 0, and the solution is "
        "identically zero. Nothing is raised. "
        "Signal: strongest first, compare "
        "constraints.n_constraints() against "
        "dof_handler.n_dofs() — equal means there is no unknown left "
        "to solve for; then system_rhs.l2_norm() == 0 exactly; then "
        "SolverControl::last_step() == 0 with last_value() == 0; and "
        "KellyErrorEstimator returns a vector of length "
        "n_active_cells(), i.e. 1. The solver log line, if you "
        "enable it with log_history(true) + "
        "deallog.depth_console(2), reads literally "
        "'DEAL:cg::Convergence step 0 value 0.00000' — deal.II "
        "prints that value in FIXED point, not the scientific "
        "'X.XXe-16' this entry used to quote.",
        "[Syntax] Boundary IDs on hyper_cube: with the DEFAULT "
        "colorize=false ALL faces have boundary_id=0. The fix is NOT "
        "to re-tag faces by hand — pass colorize=true: "
        "GridGenerator::hyper_cube(tria, 0, 1, /*colorize=*/true) "
        "assigns 0:x=0, 1:x=1, 2:y=0, 3:y=1 (3D adds 4:z=0, 5:z=1). "
        "Signal: `tria.get_boundary_ids()` returns `{0}` "
        "(a single id, not the 4-6 expected for a cube), and "
        "VectorTools::interpolate_boundary_values applied to "
        "different boundary_ids produces the same Dirichlet "
        "values across all faces of the cube. Measured on deal.II "
        "deal.II 9.x: colorize=false -> {0}; colorize=true "
        "-> {0,1,2,3} in 2D and {0,1,2,3,4,5} in 3D. (This entry "
        "used to prescribe manual re-tagging, contradicting the "
        "poisson_mixed_bc catalog which already documented "
        "colorize=true.)",
        "[Syntax] hyper_rectangle and subdivided_hyper_rectangle do "
        "NOT auto-assign per-face ids: their colorize argument also "
        "defaults to false, so every face lands on boundary_id=0. "
        "With colorize=true you get left=0, right=1, bottom=2, "
        "top=3 (3D adds front=4, back=5). Always check via "
        "tria.get_boundary_ids() after creating the mesh. "
        "Signal: tria.get_boundary_ids() returns `{0}`, not the "
        "assumed `{0,1,2,3}` — a Dirichlet loop keyed on "
        "boundary_id=1..3 then matches no faces and those sides "
        "silently become homogeneous Neumann. Measured on deal.II "
        "deal.II 9.x: hyper_rectangle<2> default -> {0}, "
        "colorize=true -> {0,1,2,3}; subdivided_hyper_rectangle<3> "
        "colorize=true -> {0,...,5}. (This entry previously stated "
        "the ids were auto-assigned and printed the OPPOSITE "
        "signal.)",
                "[Numerical] A fixed-FRACTION refinement strategy can flag "
        "ZERO cells on a coarse mesh, and the adaptive loop then runs "
        "to completion producing the identical answer every cycle with "
        "no warning at all. GridRefinement::"
        "refine_and_coarsen_fixed_number(tria, error, top_fraction, "
        "bottom_fraction) flags floor(top_fraction * n_active_cells) "
        "cells. Verified on GridGenerator::hyper_L, which starts with "
        "3 cells: with top_fraction = 0.3 that is 0.9, which floors to "
        "0, so no cell was flagged and "
        "execute_coarsening_and_refinement() left the mesh untouched; "
        "after a single refine_global the same call flagged 3 of 12 "
        "cells and the mesh grew normally. Always refine_global at "
        "least once (twice is safer) before an adaptive loop — every "
        "deal.II tutorial does. "
        "Signal: print n_active_cells() and n_dofs() EVERY cycle and "
        "require them to increase. A cycle count that rises while the "
        "DoF count stands still is this bug; so is an error estimate "
        "that is bit-identical between cycles. The alternative "
        "strategy refine_and_coarsen_fixed_fraction (which flags "
        "cells until a fraction of the total ERROR is covered) does "
        "not have this failure mode, but can flag almost every cell "
        "on a mesh with a flat error distribution.",
        "[Numerical] For AMR: hanging-node constraints are "
        "MANDATORY — build them with "
        "DoFTools::make_hanging_node_constraints and apply them "
        "either through AffineConstraints::"
        "distribute_local_to_global during assembly (preferred) or "
        "with the still-supported constraints.condense(sparsity) + "
        "constraints.condense(system_matrix, system_rhs) pair. "
        "But do NOT expect a loud failure if you forget. Verified by "
        "running the same adaptive Poisson problem with and without "
        "the constraints: the unconstrained system stayed EXACTLY "
        "symmetric, CG converged in essentially the same number of "
        "iterations at every cycle, and nothing was raised in either "
        "build — the ONLY difference was in the answer, where the "
        "error against the reference solution stopped improving and "
        "then got worse under refinement while the constrained run "
        "kept improving. "
        "TWO DIAGNOSTICS THAT DO NOT WORK, both checked: "
        "(1) matrix symmetry. max|A_ij - A_ji| was bitwise 0.0 with "
        "AND without the constraints, so it cannot distinguish them. "
        "(2) solver health. The iteration counts of the two runs "
        "tracked each other cycle for cycle; there is no blow-up to "
        "wait for, and this entry's old claim of a 'SolverCG "
        "breakdown on iteration 2-3' never reproduced (nor can it — "
        "see the essentials block on why deal.II cannot report a CG "
        "breakdown in a Release build at all). "
        "ONE MATRIX-LEVEL DIAGNOSTIC THAT DOES: max|A| rises when "
        "the constraints are condensed in, because condensation adds "
        "the eliminated rows onto the diagonal. If your two runs "
        "produce the identical maximum entry, the constraints never "
        "reached the matrix. "
        "Signal: an error against a reference (manufactured "
        "solution, richer discretisation, or a globally refined run) "
        "that stops decreasing — or increases — under refinement, "
        "while the solver looks perfectly healthy.",
        "[API] AffineConstraints<double> handles both Dirichlet BCs "
        "and hanging nodes — interpolate_boundary_values + the "
        "hanging-node closure on the SAME constraints object. Using "
        "two separate constraints objects produces inconsistent "
        "assembly. Signal: DataOut output shows step "
        "discontinuities of order 1e-2 to 1e-1 at refinement-level "
        "interfaces (hanging-node faces); SolverCG converges but "
        "the L2-error against an analytic reference plateaus "
        "instead of decreasing as h is refined.",
        # New entries shipped with this encoding pass — common Poisson
        # failure modes the catalog should warn about.
        "[Numerical] For variable-coefficient problems (a(x) ∇u), if "
        "the coefficient varies over several orders of magnitude "
        "(layered media, heterogeneous), the stiffness matrix gets "
        "ill-conditioned and PreconditionSSOR loses effectiveness. "
        "Switch to PreconditionAMG / BoomerAMG, which respects the "
        "coefficient structure. Signal: SolverCG iteration count "
        "from SolverControl::last_step() grows linearly with the "
        "max/min ratio of the coefficient (e.g. 50 iterations at "
        "contrast 1e2, 500 at contrast 1e3); switching to "
        "PreconditionAMG drops it back to O(log(ndof)).",
                "[Integration] BEFORE trusting any 'raises Exc...' claim in "
        "this or any deal.II catalog, determine the build type of "
        "the library you are compiling against — the same misuse "
        "produces a full diagnostic on one build and silence or a "
        "segfault on the other. How to check: run grep CMAKE_BUILD_TYPE "
        "over $DEAL_II_DIR/CMakeCache.txt, and list $DEAL_II_DIR/lib — "
        "libdeal_II.so is the Release library, libdeal_II.g.so is "
        "the Debug one; an install may ship one, the other, or both. "
        "On a Release-only install you cannot opt back in: "
        "`cmake -DCMAKE_BUILD_TYPE=Debug` on YOUR project prints "
        "'#  WARNING: ... CMAKE_BUILD_TYPE was forced to \"Release\"' "
        "and still compiles with -DNDEBUG. "
        "WHY IT MATTERS: deal.II's argument checking is mostly "
        "Assert(...), which exists ONLY in a Debug build. In Release "
        "those checks vanish and the misuse returns silently with a "
        "wrong answer or segfaults. Signal: grep CMAKE_BUILD_TYPE over "
        "$DEAL_II_DIR/CMakeCache.txt and list $DEAL_II_DIR/lib — a "
        "libdeal_II.g.so means the Assert-based diagnostics exist, "
        "only libdeal_II.so means they do not. "
        "Verified pairs, same program, "
        "both builds: SparseMatrix::add() outside the sparsity "
        "pattern — Debug aborts with 'You are trying to access the "
        "matrix entry with index <i,j>, but this entry does not "
        "exist in the sparsity pattern of this matrix.', Release "
        "silently DROPS the value; an active_fe_index beyond "
        "hp::FECollection::size() — Debug aborts with 'The mesh "
        "contains a cell with an active FE index of <N>, but the "
        "finite element collection only has <M> elements', Release "
        "SEGFAULTS (exit 139) with no message; "
        "FEValues::get_function_gradients with a scalar-shaped "
        "container on a vector-valued FESystem — Debug aborts with "
        "'Two sizes or dimensions were supposed to be equal, but "
        "aren't. They are <n_components> and 1.', Release returns "
        "normally with a mixture of different components' "
        "derivatives. "
        "PARTIAL WORKAROUND on a Release-only install: adding "
        "-DDEBUG to your own translation unit re-activates the "
        "Asserts that live in deal.II HEADERS (SparseMatrix, "
        "FEValues accessors, SolverCG, FEEvaluation) — verified: the "
        "out-of-pattern write starts aborting with the full report. "
        "It does NOT re-activate Asserts compiled into the library "
        "(DoFHandler::distribute_dofs and most of source/*.cc), so "
        "the hp-index case still segfaults. Use it as a cheap triage "
        "step, not as a substitute for a Debug build. "
        "(This slot previously held a claim that FE_Q<2>(2) plus "
        "FE_Q<3>(1) in one translation unit produces 'undefined "
        "reference to FE_Q<3>::FE_Q(unsigned int)'. It does not: "
        "deal.II pre-instantiates FE_Q<1>, FE_Q<2> and FE_Q<3>, and "
        "the two-dimension program compiled, linked and ran.)",
    ],
}

GENERAL_KNOWLEDGE = {
    "description": "deal.II general capabilities",
    "element_types": {
        "H1": "FE_Q(p), FE_Q_Hierarchical(p), FE_Bernstein(p), FE_Hermite(p), FE_SimplexP(p)",
        "H1_enriched": (
            "Strictly H1-conforming enrichments — safe to use anywhere "
            "the formulation needs H1 continuity:  "
            "FE_Q_Bubbles(p) — Q with cell-interior bubble enrichment "
            "(the bubble vanishes on the cell boundary, so inter-element "
            "continuity is preserved).  Upstream caveat: condition number "
            "grows quickly for p > 3; use the lowest applicable degree.  "
            "FE_SimplexP_Bubbles(p) — simplex analogue of FE_Q_Bubbles.  "
            "FE_Q_iso_Q1(p) — piecewise (bi-/tri-)linear functions on a "
            "macro-element of p^dim sub-cells; the cell is conceptually "
            "split into p subdivisions per coordinate direction and a Q1 "
            "basis is laid down on the resulting subcells (still globally "
            "continuous, so H1-conforming)."
        ),
        "nonconforming_and_qp_dg0": (
            "NOT H1-conforming — agent should NOT pick these for an "
            "H1-conforming formulation:  "
            "FE_Q_DG0(p) — Lagrange Qp **plus** the space of cell-wise "
            "constant functions (Qp+DG0).  The added piecewise-constant "
            "part is discontinuous across element boundaries; only the "
            "Lagrange part is continuous.  Used in mixed/stabilised "
            "discretisations that explicitly want the extra discontinuous "
            "mode.  "
            "FE_RannacherTurek(0) — classical first-order *nonconforming* "
            "element (degree argument fixed to 0 in upstream).  Continuity "
            "is enforced only at edge/face midpoints, not pointwise across "
            "faces."
        ),
        "DG": "FE_DGQ(p), FE_DGQLegendre(p), FE_DGQHermite(p), FE_DGP(p), FE_SimplexDGP(p)",
        "DG_advanced": (
            "FE_DGQArbitraryNodes(quadrature) — DG_Q on a user-chosen node "
            "set (Gauss-Lobatto, Gauss, equispaced) for matrix-free / "
            "spectral-element style discretisations; "
            "FE_DGPMonomial(p) — DG using the monomial polynomial basis "
            "rather than the standard nodal basis (kept for legacy "
            "comparison and analytic-coefficient access); "
            "FE_DGPNonparametric(p) — DG with a non-parametric mapping, "
            "i.e. polynomials defined in physical (not reference) space; "
            "FE_DGVector<PolynomialsType> — class template defined in "
            "fe_dg_vector.h that wraps a vector-valued polynomial space "
            "(PolynomialsRaviartThomas, PolynomialsNedelec, PolynomialsBDM) "
            "into a DG element.  The three concrete instantiations are: "
            "FE_DGRaviartThomas(k) — DG element built on the RT polynomial "
            "space (used in DG mixed methods); "
            "FE_DGNedelec(k) — DG element on the Nédélec polynomial space "
            "(discontinuous H(curl)-type approximation); "
            "FE_DGBDM(k) — DG element on the Brezzi-Douglas-Marini "
            "polynomial space (discontinuous H(div)-type approximation)."
        ),
        "H(div)": "FE_RaviartThomas(k), FE_BDM(k), FE_ABF(k), FE_BernardiRaugel(1)",
        "H(div)_advanced": (
            "FE_RaviartThomasNodal(k) — RT with a nodal degree-of-freedom "
            "representation (alternative to the moment-based default), "
            "convenient when interpolating from a nodal velocity field; "
            "FE_RT_Bubbles(k) — RT enriched with interior bubble functions "
            "for improved approximation order on a fixed mesh."
        ),
        "H(curl)": "FE_Nedelec(k), FE_NedelecSZ(k)",
        "H(curl)_advanced": (
            "FE_NedelecNodal(k) — Nédélec element with a nodal-interpolation "
            "DoF setup, useful when coupling against nodal H(curl) data. "
            "deal.II 9.8.0 and later only (not in 9.7.1); declared in "
            "deal.II/fe/fe_nedelec.h."
        ),
        "trace_and_face": (
            "FE_FaceQ(p) — Q-polynomial face element used for "
            "hybridised DG (HDG) interface unknowns; "
            "FE_FaceP(p) — P-polynomial face element, the simplex "
            "analogue of FE_FaceQ; "
            "FE_TraceQ(p) — trace of FE_Q on element faces, used by "
            "Lagrange-multiplier and HDG stabilisations."
        ),
        "pyramid_and_wedge_3d": (
            "FE_PyramidP(p) — continuous P element on pyramidal (square-base) "
            "3D cells, used in transition meshes between hex and tet regions; "
            "FE_PyramidDGP(p) — DG counterpart of FE_PyramidP; "
            "FE_WedgeP(p) — continuous P element on wedge (triangular-prism) "
            "3D cells, the second transition shape between hex and tet; "
            "FE_WedgeDGP(p) — DG counterpart of FE_WedgeP."
        ),
        "special": "FE_FaceQ(p), FE_Nothing, FE_Enriched, FE_P1NC, FESystem, hp::FECollection",
        "internal_polynomial_bases": (
            "FE_Poly, FE_PolyFace, FE_PolyTensor, FE_Q_Base, FE_SimplexPoly, "
            "FE_PyramidPoly, FE_WedgePoly — abstract polynomial base classes "
            "that the concrete elements above are templated on (e.g. FE_Q is "
            "an FE_Q_Base; FE_PyramidP is an FE_PyramidPoly; FE_SimplexP is "
            "an FE_SimplexPoly).  Listed here only so the agent does not "
            "propose them in user code; these classes have no public "
            "stand-alone constructor and are not directly instantiated."
        ),
    },
    "mesh_generators": [
        "hyper_cube, hyper_rectangle, hyper_L, hyper_ball, hyper_shell",
        "channel_with_cylinder, plate_with_a_hole, cheese, cylinder",
        "merge_triangulations, extrude_triangulation",
        "Import: Gmsh, UCD, VTK, ExodusII, ABAQUS, OpenCASCADE",
    ],
    # ── Cross-physics gaps. Every entry below was reproduced by
    #    compiling and running a program on deal.II 9.8.0-pre; none
    #    of them was covered anywhere in this catalog before.
    "install_and_build_gotchas": [
        "[Build] Determine the build type before you rely on ANY "
        "diagnostic. `grep CMAKE_BUILD_TYPE "
        "$DEAL_II_DIR/CMakeCache.txt`; list $DEAL_II_DIR/lib "
        "(libdeal_II.so = Release, libdeal_II.g.so = Debug; an "
        "install may ship one, the other, or both). A Release-only "
        "install cannot be talked into Debug from your side: "
        "`cmake -DCMAKE_BUILD_TYPE=Debug` on your project prints "
        "'#  WARNING: / CMAKE_BUILD_TYPE \"Debug\" unsupported by "
        "current installation! / deal.II was configured with "
        "\"Release\". / CMAKE_BUILD_TYPE was forced to \"Release\".' "
        "and still emits -DNDEBUG. CONSEQUENCE: every deal.II "
        "`Assert(...)` is compiled out in Release and only "
        "`AssertThrow(...)` survives, so a pitfall whose Signal is "
        "'raises Exc...' describes the DEBUG behaviour; in Release "
        "the same mistake returns silently with a wrong answer or "
        "segfaults. Signal: grep CMAKE_BUILD_TYPE over "
        "$DEAL_II_DIR/CMakeCache.txt and list $DEAL_II_DIR/lib — a "
        "libdeal_II.g.so means the Assert-based diagnostics exist, "
        "only libdeal_II.so means they do not. Verified on the SAME "
        "programs in both builds: "
        "(a) SparseMatrix::add() outside the sparsity pattern — "
        "Debug aborts with 'You are trying to access the matrix "
        "entry with index <i,j>, but this entry does not exist in "
        "the sparsity pattern of this matrix.', Release drops the "
        "value; (b) FEValues::get_function_gradients() with a "
        "scalar-shaped std::vector<Tensor<1,dim>> on a vector-valued "
        "FESystem — Debug aborts with 'Two sizes or dimensions were "
        "supposed to be equal, but aren\'t.', Release returns a "
        "mixture of components; (c) an active_fe_index >= "
        "hp::FECollection::size() — Debug aborts with 'The mesh "
        "contains a cell with an active FE index of <N>, but the "
        "finite element collection only has <M> elements', Release "
        "segfaults (exit 139) with no message. On a Release-only "
        "install, adding -DDEBUG to YOUR translation unit revives "
        "(a) and (b) — the Asserts that live in deal.II headers — "
        "but not (c), whose Assert is compiled into the library.",

        "[API] Non-interpolatory elements have no support points, "
        "and VectorTools::interpolate_boundary_values must not be "
        "used on them. Guard with "
        "FiniteElement::has_support_points(). It is FALSE for far "
        "more elements than the usual H(div)/H(curl) suspects: "
        "verified by instantiation, it is false for "
        "FE_Q_Hierarchical, FE_Bernstein, FE_Hermite, FE_DGP, "
        "FE_DGPMonomial, FE_DGPNonparametric, FE_DGQLegendre, "
        "FE_DGQHermite, FE_FaceP, FE_P1NC, FE_RannacherTurek, "
        "FE_NedelecSZ, and for FE_RaviartThomas / FE_BDM / FE_ABF / "
        "FE_Nedelec / FE_BernardiRaugel / FE_RT_Bubbles, whose DoFs "
        "are moments rather than point values. On a CONTINUOUS "
        "element without support points the call SEGFAULTS (exit "
        "139) on Release — verified for FE_Q_Hierarchical, "
        "FE_Bernstein and FESystem(FE_RannacherTurek<dim>(), dim) — "
        "while a Debug build aborts with 'You are trying to access "
        "the support points of a finite element that either has no "
        "support points at all, or for which the corresponding "
        "tables have not been implemented.' On a DISCONTINUOUS "
        "element (FE_DGQ, FE_DGP) the same call is instead a silent "
        "no-op that leaves the boundary-value map EMPTY. The working "
        "replacement in every case is "
        "VectorTools::project_boundary_values(dof_handler, "
        "{{id, &function}}, QGauss<dim-1>(fe.degree + 2), "
        "boundary_values); it returned a populated map on the same "
        "setups where interpolate crashed.",

        "[API] deal.II 9.1 -> 9.8 removed several APIs that older "
        "templates still use. Each of these is a hard compile error, "
        "which is the good case — but they are easy to mistake for a "
        "broken install. Verified by compiling this catalog's own "
        "generators on deal.II 9.8: "
        "DoFTools::count_dofs_per_block -> count_dofs_per_fe_block "
        "(now RETURNS the vector, no out-parameter); "
        "MGTransferPrebuilt::build_matrices -> build; "
        "MatrixFree::reinit(dof, constraints, quad, data) -> the "
        "mapping must be the first argument; "
        "MatrixFree::n_macro_cells -> n_cell_batches; "
        "FEEvaluation::evaluate/integrate(bool, bool) -> "
        "EvaluationFlags::values / EvaluationFlags::gradients; "
        "SolutionTransfer::interpolate(in, out) -> interpolate(out). "
        "DoFTools::extract_locally_relevant_dofs still accepts BOTH "
        "the modern returning form and the old (dof, IndexSet&) "
        "out-parameter form on 9.8.",

        "[Integration] Availability probes that only check whether a "
        "header includes are WRONG on a SOURCE build. A source install "
        "ships every header regardless of which optional dependencies "
        "were enabled, and the bodies sit behind `#ifdef "
        "DEAL_II_WITH_<FEATURE>`. Verified on a source build with "
        "MPI, P4EST, PETSC, SLEPC, TRILINOS, SUNDIALS, ADOLC, "
        "SYMENGINE and COMPLEX_VALUES all undefined: "
        "`#include <deal.II/lac/slepc_solver.h>` compiles and links "
        "fine, and only naming SLEPcWrappers fails "
        "(\"'dealii::SLEPcWrappers' has not been declared\"); "
        "`#include <deal.II/distributed/tria.h>` also compiles, and "
        "constructing parallel::distributed::Triangulation fails at "
        "COMPILE time with 'use of deleted function'; "
        "Utilities::MPI::MPI_InitFinalize constructs happily and "
        "reports n_mpi_processes == 1. The only reliable probe is to "
        "grep $DEAL_II_DIR/include/deal.II/base/config.h for the "
        "'/* #undef DEAL_II_WITH_<X> */' lines.",
    ],
    "parallel": "MPI (p4est) + TBB/Taskflow + CUDA/Kokkos GPU",
    "amr": "KellyErrorEstimator, DWR (step-14), hp-adaptivity (step-27/75)",
    "matrix_free": "MatrixFree + FEEvaluation, sum factorization (step-37/48/59/64/66/67/75/76/95)",
    "output": "VTU (DataOut), higher-order VTU cells, PVTU (parallel), PVD (time series)",
    "unique_features": [
        "~88 tutorial programs covering almost every FEM topic. The numbering runs past step-90 but HAS GAPS, so the highest number is not the count — enumerate examples/step-* in the source checkout if you need the real list (a package prefix ships none)",
        "hp-adaptive FEM with automatic smoothness estimation",
        "Matrix-free methods with sum factorization (10x faster than sparse)",
        "GPU support via CUDA and Kokkos",
        "Automatic differentiation via Sacado/ADOL-C for nonlinear problems",
        "Scalable to 10^12 DOFs on 300,000+ MPI processes",
    ],
    "cmake_user_macros": {
        "description": (
            "User-callable DEAL_II_* CMake macros that downstream users "
            "invoke in their CMakeLists.txt. Source: "
            "dealii/cmake/macros/macro_deal_ii_*.cmake."
        ),
        "DEAL_II_INITIALIZE_CACHED_VARIABLES": {
            "signature": "DEAL_II_INITIALIZE_CACHED_VARIABLES()",
            "purpose": "Inherit deal.II's compiler + build settings.",
            "order_constraint": (
                "MUST be called AFTER find_package(deal.II) — needs "
                "DEAL_II_PROJECT_CONFIG_INCLUDED. AND MUST be called "
                "BEFORE project() — sets CMAKE_CXX_COMPILER + build-type "
                "cache that project() must consume."
            ),
            "Signal": (
                "[Input] Wrong-order use of DEAL_II_INITIALIZE_CACHED_VARIABLES "
                "fails with FATAL_ERROR literal text "
                "'DEAL_II_INITIALIZE_CACHED_VARIABLES can only be called in "
                "external projects after the inclusion of deal.IIConfig.cmake. "
                "It is not intended for internal use.' "
                "Two subtle silent side effects: "
                "(1) If user's -DCMAKE_BUILD_TYPE=Debug doesn't match the "
                "dealii install's DEAL_II_BUILD_TYPE (e.g. user wants Debug "
                "but dealii built Release-only), the macro FORCEs "
                "CMAKE_BUILD_TYPE to a valid mode and emits a banner "
                "starting with '#  WARNING:' and the literal "
                "'CMAKE_BUILD_TYPE was forced to'. "
                "(2) The macro WIPES user-set CMAKE_CXX_FLAGS / "
                "CMAKE_CXX_FLAGS_DEBUG / CMAKE_CXX_FLAGS_RELEASE to empty "
                "strings — any -O3 / -march=native / etc. set BEFORE the "
                "macro is lost. To customise flags, set them AFTER calling "
                "the macro. (File walk macro_deal_ii_initialize_cached_variables.cmake "
                ".)"
            ),
        },
        "DEAL_II_SETUP_TARGET": {
            "signature": (
                "DEAL_II_SETUP_TARGET(<target> [DEBUG|RELEASE])"),
            "purpose": (
                "Append deal.II's INCLUDE_DIRECTORIES, COMPILE_FLAGS, "
                "LINK_FLAGS, COMPILE_DEFINITIONS, and link interface "
                "to <target>. Must be called AFTER add_executable / "
                "add_library on <target> and AFTER "
                "find_package(deal.II). Source: "
                "cmake/macros/macro_deal_ii_setup_target.cmake."),
            "Signal": (
                "[Input] DEAL_II_SETUP_TARGET has FIVE distinct "
                "failure / silent-surprise modes users routinely hit: "
                "(1) Called BEFORE find_package(deal.II) — "
                "FATAL_ERROR with literal text "
                "'DEAL_II_SETUP_TARGET can only be called in external "
                "projects after the inclusion of deal.IIConfig.cmake. "
                "It is not intended for internal use.' (gated on "
                "DEAL_II_PROJECT_CONFIG_INCLUDED). "
                "(2) CMAKE_BUILD_TYPE is set to something other than "
                "'Debug' or 'Release' (e.g. RelWithDebInfo, "
                "MinSizeRel, empty) and no explicit DEBUG|RELEASE arg "
                "is passed — FATAL_ERROR 'DEAL_II_SETUP_TARGET cannot "
                "determine DEBUG, or RELEASE flavor for target. "
                "CMAKE_BUILD_TYPE \"<X>\" is neither equal to "
                "\"Debug\", nor \"Release\"'. Common with Ninja "
                "multi-config / Visual Studio defaults. Fix: set "
                "CMAKE_BUILD_TYPE explicitly OR call "
                "DEAL_II_SETUP_TARGET(<target> DEBUG|RELEASE). "
                "(3) DANGEROUS SILENT DOWNGRADE: if user requests "
                "DEBUG (via arg or CMAKE_BUILD_TYPE=Debug) but the "
                "INSTALLED deal.II was built RELEASE-only "
                "(DEAL_II_BUILD_TYPE doesn't contain 'Debug'), the "
                "macro silently overrides `_build` to RELEASE without "
                "any warning. The user's debug-flagged target links "
                "against the optimized deal.II — Assert macros are "
                "compiled out, the debugger steps through optimized "
                "frames, and bug-hunters waste hours wondering why "
                "DealiiAssert isn't firing. Fix: rebuild deal.II "
                "with -DCMAKE_BUILD_TYPE=DebugRelease, or accept the "
                "release build. "
                "(4) Any argument other than DEBUG / RELEASE — "
                "FATAL_ERROR 'The deal_ii_setup_target() macro was "
                "called with an invalid argument. Valid arguments are "
                "(none), DEBUG, or RELEASE. The argument given is "
                "\"<arg>\".' Giving both DEBUG and RELEASE is a "
                "separate FATAL_ERROR ('...the debug or release "
                "configuration can only be specified once'). "
                "Common: passing 'Debug' (lowercase d, capitalized "
                "rest) thinking the macro is case-insensitive — it "
                "isn't (string MATCHES is case-sensitive). "
                "(5) Target is an OBJECT_LIBRARY — only the deal.II "
                "LINKER FLAGS (DEAL_II_LINKER_FLAGS*) are skipped "
                "(gated on `_type != OBJECT_LIBRARY`); "
                "target_link_libraries(<target> "
                "${DEAL_II_TARGET_<build>}) is still applied, so the "
                "object library carries the deal.II link interface. "
                "Plus: this is a CMake MACRO (not FUNCTION), so "
                "internal vars _build and _type (and _arg when an "
                "argument is given) LEAK into the caller scope and can "
                "shadow user-set variables of the same names. "
                "(File walk macro_deal_ii_setup_target.cmake "
                ".)"),
        },
        "DEAL_II_INVOKE_AUTOPILOT": {
            "signature": "DEAL_II_INVOKE_AUTOPILOT()",
            "purpose": (
                "All-in-one shortcut for the canonical tutorial template. "
                "Reads CALLER-SET variables, creates an executable + 8 "
                "custom CMake targets."),
            "caller_variables": {
                "TARGET":         "REQUIRED — project + executable name",
                "TARGET_SRC":     "REQUIRED — list of .cc source files",
                "TARGET_RUN":     "optional — `make run` command (defaults to ${TARGET}); empty disables",
                "CLEAN_UP_FILES": "optional — files removed by runclean/distclean (defaults to glob `*.log *.gmv *.gnuplot *.gpl *.eps *.pov *.vtk *.ucd *.d2`)",
            },
            "targets_created": {
                "run":            "compile + execute (gated by TARGET_RUN!=empty)",
                "sign":           "Mac OSX codesign (requires OSX_CERTIFICATE_NAME)",
                "debug":          "switch CMAKE_BUILD_TYPE to Debug (only if DEAL_II_BUILD_TYPE matches Debug)",
                "release":        "switch CMAKE_BUILD_TYPE to Release (only if matches Release)",
                "runclean":       "remove output files matching CLEAN_UP_FILES glob",
                "distclean":      "remove CMakeCache.txt, CMakeFiles/, Makefile, build.ninja, .ninja_* (NUKES build state)",
                "strip_comments": "IRREVERSIBLE: runs `perl -pi -e 's#^[ \\t]*//.*\\n##g;' ${TARGET_SRC}` in place",
                "info":           "print usage message",
            },
            "Signal": (
                "[API] DEAL_II_INVOKE_AUTOPILOT creates a target named "
                "`strip_comments` that the macro itself documents as "
                "'irreversible'. The implementation is "
                "ADD_CUSTOM_TARGET(strip_comments COMMAND perl -pi -e "
                "'s#^[ \\t]*//.*\\n##g;' ${TARGET_SRC}) — Perl rewrites "
                "the SOURCE FILES in place, dropping all // line comments. "
                "There is NO confirmation prompt and NO backup. Run "
                "`make strip_comments` once and your tutorial's comments "
                "are gone. Also: `make distclean` nukes CMakeCache.txt + "
                "Makefile + .ninja_* — far more aggressive than `make "
                "clean`. Default CLEAN_UP_FILES glob may delete unrelated "
                "*.log / *.vtk files the user produced manually in the "
                "build dir. (File walk "
                "macro_deal_ii_invoke_autopilot.cmake.)"
            ),
        },
        "DEAL_II_PICKUP_TESTS": {
            "signature": "DEAL_II_PICKUP_TESTS()",
            "purpose": (
                "Glob *.output files in the current source dir, parse each "
                "filename for feature constraints, and register each as a "
                "ctest case via DEAL_II_ADD_TEST(<dir-name>, <test>, <output>)."),
            "test_filename_grammar": (
                "Tests are identified by a *.output file. The filename can "
                "encode constraints between dots: "
                "  <name>.with_<feature><op><value>.output "
                "Operators: = .ge. .le. .geq. .leq. "
                "Values: boolean (on/off/yes/no/true/false) for = comparisons; "
                "        version number (e.g. 11.2) for .ge./.le./.geq./.leq. "
                "Examples: "
                "  mytest.with_petsc=on.output           — needs DEAL_II_WITH_PETSC "
                "  mytest.with_cuda.geq.11.2.output      — needs CUDA >= 11.2 "
                "  mytest.mpirun=4.output                — runs under mpirun -np 4; "
                "                                          SKIPPED if DEAL_II_WITH_MPI=OFF"),
            "env_vars": {
                "TEST_PICKUP_REGEX": "regex filter on '<category>/<test>' names; empty = catchall (default)",
                "TEST_TIME_LIMIT":   "wall clock limit per test in seconds (default 600)",
                "DIFF_DIR":          "not read: the macro looks only for numdiff (there is no diff fallback)",
                "NUMDIFF_DIR":       "hint path for the numdiff executable, the only comparison tool the macro looks for",
                "TEST_LIBRARIES":    "extra libs/targets to link against",
                "TEST_LIBRARIES_DEBUG / _RELEASE":  "per-config link list",
                "TEST_TARGET":       "test target name (or _DEBUG / _RELEASE pair)",
            },
            "Signal": (
                "[Input] DEAL_II_PICKUP_TESTS stops with a FATAL_ERROR "
                "when (1) it is called outside an external project "
                "(DEAL_II_PROJECT_CONFIG_INCLUDED not set) — literal "
                "'DEAL_II_PICKUP_TESTS can only be called in external "
                "(test sub-) projects after the inclusion of "
                "deal.IIConfig.cmake' — or when a test file name "
                "compares a boolean feature value with an operator "
                "other than '='. A missing or "
                "broken numdiff is NOT fatal (measured on 9.7.1 and "
                "9.8): only numdiff is looked for (NUMDIFF_DIR hint, no "
                "diff fallback), and it is probed with 'numdiff -r "
                "1.0e-8' on two nearly equal files. If numdiff is "
                "missing or fails the probe (e.g. it is a symlink to "
                "diff), the macro emits a WARNING 'Could not find or "
                "execute numdiff, which is required for running most of "
                "the tests within the testsuite; ...' and sets "
                "NUMDIFF_EXECUTABLE empty (neither happens for the test "
                "categories quick_tests and performance); DEAL_II_ADD_TEST "
                "then runs every test without comparing its output, so a "
                "test whose output does not match its .output file "
                "reports PASSED. Workaround: install real numdiff from "
                "savannah.gnu.org/projects/numdiff or set NUMDIFF_DIR. "
                "Additionally: an unknown `with_<feature>` in a test filename "
                "(neither DEAL_II_WITH_<F> nor DEAL_II_<F> defined) silently "
                "drops the test from ctest's discovery — easy way to lose "
                "tests after a dealii config option rename. (File walk "
                "macro_deal_ii_pickup_tests.cmake.)"
            ),
        },
        "DEAL_II_QUERY_GIT_INFORMATION": {
            "description": (
                "Populate GIT_BRANCH / GIT_REVISION / GIT_SHORTREV / "
                "GIT_TAG / GIT_TIMESTAMP / GIT_FANCY_TAG from the "
                "source dir's .git metadata. The "
                "macro has an OPTIONAL positional PREFIX argument: "
                "called as DEAL_II_QUERY_GIT_INFORMATION() variables "
                "are unprefixed; called as "
                "DEAL_II_QUERY_GIT_INFORMATION(MYAPP) they become "
                "MYAPP_GIT_BRANCH etc. Source: "
                "cmake/macros/macro_deal_ii_query_git_information.cmake."),
            "Signal": (
                "[Output] Four sharp edges users routinely hit with "
                "DEAL_II_QUERY_GIT_INFORMATION: "
                "(1) The default variables are UNPREFIXED — GIT_BRANCH, "
                "GIT_REVISION, GIT_SHORTREV, GIT_TAG, GIT_TIMESTAMP, "
                "GIT_FANCY_TAG. There is NO "
                "DEAL_II_GIT_* prefix unless the user explicitly "
                "passes a prefix argument; the prefix is "
                "${ARGN}_-style and lives in the macro body. "
                "(2) The variable set is GIT_BRANCH / GIT_REVISION / "
                "GIT_SHORTREV / GIT_TIMESTAMP (the commit date, e.g. "
                "'2026-09-30 12:19:53+02:00') and, through the helper "
                "scripts get_latest_tag.sh / get_fancy_tag.sh, GIT_TAG "
                "/ GIT_FANCY_TAG; there is NO GIT_COMMIT_DATE. "
                "(3) If ${CMAKE_SOURCE_DIR}/.git/HEAD doesn't exist "
                "(tarball install, shallow CI checkout without .git/, "
                "or downstream app embedded in a non-git workspace) "
                "the macro is a SILENT NO-OP — no warning, no error, "
                "all six variables remain unset. Subsequent "
                "configure_file expansions on ${GIT_REVISION} produce "
                "the empty string. "
                "(4) GIT_TAG depends on the auxiliary shell script "
                "${DEAL_II_SHARE_RELDIR}/scripts/get_latest_tag.sh; "
                "if that script isn't on disk (some packagers strip "
                "it), GIT_TAG is silently left unset with only a "
                "MESSAGE(STATUS) line — easy to miss in CMake "
                "configure noise. "
                "(5) In detached-HEAD state `git symbolic-ref HEAD` "
                "returns non-zero, so GIT_BRANCH is NOT populated "
                "even though .git/HEAD exists — common in CI runs "
                "that checkout a tag or specific commit. "
                "(File walk macro_deal_ii_query_git_information.cmake "
                ".)"),
        },
    },
}
