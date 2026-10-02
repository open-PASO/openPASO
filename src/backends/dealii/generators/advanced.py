"""deal.II advanced physics generators and knowledge.

Each generator below is a REAL, self-contained, parameterized deal.II
program: it builds a mesh, assembles + solves the PDE, and writes a
.vtu (and a .pvd for the time-dependent ones) via DataOut. Every
template was compiled and executed against deal.II 9.1.1 and confirmed
to produce output before being committed here (overhaul 2026-06-26).

RE-VERIFIED by execution against deal.II 9.8.0-pre (a Release
build). Four templates in this module and
one in navier_stokes.py no longer compiled: every break was an upstream
API removal between 9.1 and 9.8, not a logic error.

  * DoFTools::count_dofs_per_block -> count_dofs_per_fe_block, which
    RETURNS the vector instead of filling an out-parameter (>= 9.2)
    — time_dependent_ns_2d here, navier_stokes_2d there;
  * MGTransferPrebuilt::build_matrices -> build (>= 9.2)
    — multigrid_2d;
  * the mapping-less MatrixFree::reinit(dof, constraints, quad, data)
    overload was removed, a Mapping must be passed first (>= 9.3);
    MatrixFree::n_macro_cells -> n_cell_batches; and
    FEEvaluation::evaluate/integrate(bool, bool) ->
    EvaluationFlags::values / ::gradients — all three in matrix_free_2d;
  * SolutionTransfer::interpolate(in, out) was removed in favour of the
    one-argument interpolate(out) — time_dependent_heat_2d.

All five now configure, compile, run and write .vtu on 9.8.0-pre.

Physics that could not be made runnable with reasonable effort on the
supported deal.II were REMOVED rather than shipped as print-and-exit
stubs: compressible_euler (step-33/69 shock capturing), multiphysics
two-phase (step-21/43 saddle), topology_opt (step-79 MMA), cg_dg_coupled
(step-46, needs FEInterfaceValues absent in 9.1.x), optimal_control
(step-72, needs Sacado AD). See backend.supported_physics().
"""


_SRC_MIXED_LAPLACIAN_2D = r"""/* Mixed Laplacian (Darcy) with Raviart-Thomas H(div) elements - deal.II
 * Based on step-20. Saddle-point system solved with SparseDirectUMFPACK.
 *   K^{-1} u + grad p = 0,   div u = -f
 * RT_k velocity + DGQ_k pressure. Self-contained, parameterized.
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

using namespace dealii;

const unsigned int degree = @@DEGREE@@;
const unsigned int refinements = @@REFINEMENTS@@;

// Right-hand side f for div u = -f
template <int dim>
class RHS : public Function<dim>
{
public:
  virtual double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    (void)p;
    return 1.0;
  }
};

int main()
{
  const int dim = 2;
  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(refinements);

  FESystem<dim> fe(FE_RaviartThomas<dim>(degree), 1, FE_DGQ<dim>(degree), 1);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  DoFRenumbering::component_wise(dof_handler);

  const std::vector<types::global_dof_index> dofs_per_component =
      DoFTools::count_dofs_per_fe_component(dof_handler);
  const unsigned int n_u = dofs_per_component[0], n_p = dofs_per_component[dim];
  std::cout << "Mixed Laplacian DOFs: u=" << n_u << " p=" << n_p << std::endl;

  BlockDynamicSparsityPattern dsp(2, 2);
  dsp.block(0, 0).reinit(n_u, n_u);
  dsp.block(1, 0).reinit(n_p, n_u);
  dsp.block(0, 1).reinit(n_u, n_p);
  dsp.block(1, 1).reinit(n_p, n_p);
  dsp.collect_sizes();
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  BlockSparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  BlockSparseMatrix<double> system_matrix(sparsity_pattern);
  BlockVector<double> solution(2), system_rhs(2);
  solution.block(0).reinit(n_u); solution.block(1).reinit(n_p); solution.collect_sizes();
  system_rhs.block(0).reinit(n_u); system_rhs.block(1).reinit(n_p); system_rhs.collect_sizes();

  QGauss<dim> quadrature(degree + 2);
  FEValues<dim> fe_values(fe, quadrature,
    update_values | update_gradients | update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> local_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);
  RHS<dim> rhs;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      local_matrix = 0; local_rhs = 0;
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
          const double f = rhs.value(fe_values.quadrature_point(q));
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const Tensor<1, dim> phi_i_u = fe_values[velocities].value(i, q);
              const double div_phi_i_u = fe_values[velocities].divergence(i, q);
              const double phi_i_p = fe_values[pressure].value(i, q);
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const Tensor<1, dim> phi_j_u = fe_values[velocities].value(j, q);
                  const double div_phi_j_u = fe_values[velocities].divergence(j, q);
                  const double phi_j_p = fe_values[pressure].value(j, q);
                  local_matrix(i, j) += (phi_i_u * phi_j_u
                                         - div_phi_i_u * phi_j_p
                                         - phi_i_p * div_phi_j_u) * fe_values.JxW(q);
                }
              local_rhs(i) += -phi_i_p * f * fe_values.JxW(q);
            }
        }
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i], local_dof_indices[j], local_matrix(i, j));
          system_rhs(local_dof_indices[i]) += local_rhs(i);
        }
    }

  SparseDirectUMFPACK direct;
  direct.initialize(system_matrix);
  direct.vmult(solution, system_rhs);

  std::cout << "p range: [" << *std::min_element(solution.block(1).begin(), solution.block(1).end())
            << ", " << *std::max_element(solution.block(1).begin(), solution.block(1).end()) << "]" << std::endl;

  std::vector<std::string> names(dim, "velocity");
  names.push_back("pressure");
  std::vector<DataComponentInterpretation::DataComponentInterpretation> interp(
    dim, DataComponentInterpretation::component_is_part_of_vector);
  interp.push_back(DataComponentInterpretation::component_is_scalar);
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, names,
    DataOut<dim>::type_dof_data, interp);
  data_out.build_patches(degree + 1);
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);
  std::cout << "Output written to result.vtu" << std::endl;
  return 0;
}
"""

_SRC_TIME_DEPENDENT_HEAT_2D = r"""/* Transient heat equation with adaptive mesh refinement - deal.II (based on step-26)
 * dT/dt - alpha lap(T) = f, theta time stepping; Kelly-driven AMR with
 * SolutionTransfer between meshes. Self-contained; writes .vtu per step + .pvd.
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace dealii;

const unsigned int initial_refine = @@REFINEMENTS@@;
const unsigned int n_steps = @@N_STEPS@@;
const double dt = @@DT@@;
const double alpha = @@ALPHA@@;
const double theta = @@THETA@@; // Crank-Nicolson

template <int dim>
class Source : public Function<dim>
{
public:
  void set_time(double t) { time = t; }
  virtual double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    // moving hot spot
    const Point<dim> c(0.5 + 0.3 * std::cos(time), 0.5 + 0.3 * std::sin(time));
    return ((p - c).norm_square() < 0.01) ? 50.0 : 0.0;
  }
  double time = 0.0;
};

int main()
{
  const int dim = 2;
  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(initial_refine);

  FE_Q<dim> fe(1);
  DoFHandler<dim> dof_handler(triangulation);

  SparsityPattern sparsity_pattern;
  SparseMatrix<double> mass_matrix, laplace_matrix, system_matrix;
  Vector<double> solution, old_solution, system_rhs, tmp, forcing;
  AffineConstraints<double> constraints;
  Source<dim> source;

  auto setup_system = [&]() {
    dof_handler.distribute_dofs(fe);
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, true);
    sparsity_pattern.copy_from(dsp);
    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    system_matrix.reinit(sparsity_pattern);
    MatrixTools::create_mass_matrix(dof_handler, QGauss<dim>(3), mass_matrix);
    MatrixTools::create_laplace_matrix(dof_handler, QGauss<dim>(3), laplace_matrix);
    system_rhs.reinit(dof_handler.n_dofs());
    tmp.reinit(dof_handler.n_dofs());
    forcing.reinit(dof_handler.n_dofs());
  };

  setup_system();
  solution.reinit(dof_handler.n_dofs());
  old_solution.reinit(dof_handler.n_dofs());
  solution = 0;

  std::cout << "Transient heat (AMR) initial DOFs: " << dof_handler.n_dofs() << std::endl;
  std::vector<std::pair<double, std::string>> pvd_records;

  for (unsigned int step = 1; step <= n_steps; ++step)
    {
      const double time = step * dt;
      old_solution = solution;

      // RHS = M u_old - (1-theta) dt alpha K u_old + dt * F
      mass_matrix.vmult(system_rhs, old_solution);
      laplace_matrix.vmult(tmp, old_solution);
      system_rhs.add(-(1 - theta) * dt * alpha, tmp);

      source.set_time(time);
      VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(3), source, forcing);
      system_rhs.add(dt, forcing);

      // system = M + theta dt alpha K
      system_matrix.copy_from(mass_matrix);
      system_matrix.add(theta * dt * alpha, laplace_matrix);

      // Dirichlet T=0 on whole boundary
      std::map<types::global_dof_index, double> bv;
      VectorTools::interpolate_boundary_values(dof_handler, 0,
        Functions::ZeroFunction<dim>(), bv);
      MatrixTools::apply_boundary_values(bv, system_matrix, solution, system_rhs);
      constraints.condense(system_matrix, system_rhs);

      SolverControl sc(2000, 1e-12 * system_rhs.l2_norm());
      SolverCG<Vector<double>> cg(sc);
      PreconditionSSOR<SparseMatrix<double>> prec;
      prec.initialize(system_matrix, 1.0);
      cg.solve(system_matrix, solution, system_rhs, prec);
      constraints.distribute(solution);

      // Adaptive refinement every 5 steps, with SolutionTransfer
      if (step % 5 == 0 && step < n_steps)
        {
          Vector<float> est(triangulation.n_active_cells());
          KellyErrorEstimator<dim>::estimate(dof_handler, QGauss<dim - 1>(3),
            std::map<types::boundary_id, const Function<dim> *>(), solution, est);
          GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
            est, 0.5, 0.2);
          // limit refinement levels
          if (triangulation.n_levels() > initial_refine + 3)
            for (const auto &cell :
                 triangulation.active_cell_iterators_on_level(initial_refine + 3))
              cell->clear_refine_flag();

          SolutionTransfer<dim> soltrans(dof_handler);
          triangulation.prepare_coarsening_and_refinement();
          soltrans.prepare_for_coarsening_and_refinement(solution);
          triangulation.execute_coarsening_and_refinement();

          setup_system();
          Vector<double> interpolated(dof_handler.n_dofs());
          // deal.II >= 9.7: SolutionTransfer::interpolate(in, out) is gone;
          // the one-argument form fills the NEW-mesh vector from the state
          // captured by prepare_for_coarsening_and_refinement().
          soltrans.interpolate(interpolated);
          solution = interpolated;
          constraints.distribute(solution);
        }

      if (step % 5 == 0)
        {
          DataOut<dim> data_out;
          data_out.attach_dof_handler(dof_handler);
          data_out.add_data_vector(solution, "temperature");
          data_out.build_patches();
          const std::string fname = "result-" + Utilities::int_to_string(step, 4) + ".vtu";
          std::ofstream output(fname);
          data_out.write_vtu(output);
          pvd_records.emplace_back(time, fname);
        }
    }

  std::ofstream pvd("result.pvd");
  DataOutBase::write_pvd_record(pvd, pvd_records);
  std::cout << "Final DOFs: " << dof_handler.n_dofs()
            << " T range: [" << *std::min_element(solution.begin(), solution.end())
            << ", " << *std::max_element(solution.begin(), solution.end()) << "]" << std::endl;
  std::cout << "Output written to result-*.vtu / result.pvd" << std::endl;
  return 0;
}
"""

_SRC_TIME_DEPENDENT_WAVE_2D = r"""/* Wave equation with Newmark time integration - deal.II (based on step-23)
 * u_tt = c^2 lap(u); Newmark-beta (beta=1/4, gamma=1/2) implicit, energy-conserving.
 * Self-contained, parameterized; writes a .vtu per output step + a .pvd index.
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace dealii;

const unsigned int n_refine = @@REFINEMENTS@@;
const unsigned int n_steps = @@N_STEPS@@;
const double dt = @@DT@@;
const double wave_speed = @@WAVE_SPEED@@;

template <int dim>
class InitialValues : public Function<dim>
{
public:
  virtual double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    const double r2 = (p - Point<dim>(0.5, 0.5)).norm_square();
    return std::exp(-100.0 * r2);
  }
};

int main()
{
  const int dim = 2;
  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(n_refine);

  FE_Q<dim> fe(1);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  std::cout << "Wave DOFs: " << dof_handler.n_dofs() << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> mass_matrix, laplace_matrix, matrix_u, matrix_v;
  mass_matrix.reinit(sparsity_pattern);
  laplace_matrix.reinit(sparsity_pattern);
  matrix_u.reinit(sparsity_pattern);
  matrix_v.reinit(sparsity_pattern);

  MatrixTools::create_mass_matrix(dof_handler, QGauss<dim>(3), mass_matrix);
  MatrixTools::create_laplace_matrix(dof_handler, QGauss<dim>(3), laplace_matrix);

  Vector<double> solution_u(dof_handler.n_dofs()), solution_v(dof_handler.n_dofs());
  Vector<double> old_u(dof_handler.n_dofs()), old_v(dof_handler.n_dofs());
  Vector<double> system_rhs(dof_handler.n_dofs()), tmp(dof_handler.n_dofs());

  VectorTools::interpolate(dof_handler, InitialValues<dim>(), old_u);
  old_v = 0;

  const double c2 = wave_speed * wave_speed;
  const double theta = 0.5; // Crank-Nicolson form of step-23

  // matrix_u = M + theta^2 dt^2 c2 K  (for displacement update)
  matrix_u.copy_from(mass_matrix);
  matrix_u.add(theta * theta * dt * dt * c2, laplace_matrix);

  std::vector<std::pair<double, std::string>> pvd_records;

  for (unsigned int step = 0; step <= n_steps; ++step)
    {
      const double time = step * dt;
      if (step > 0)
        {
          // displacement: (M + th^2 dt^2 c2 K) u = M(u_old + dt v_old)
          //               - th(1-th) dt^2 c2 K u_old
          mass_matrix.vmult(system_rhs, old_u);
          mass_matrix.vmult(tmp, old_v);
          system_rhs.add(dt, tmp);
          laplace_matrix.vmult(tmp, old_u);
          system_rhs.add(-theta * (1 - theta) * dt * dt * c2, tmp);

          SolverControl sc(2000, 1e-12 * system_rhs.l2_norm());
          SolverCG<Vector<double>> cg(sc);
          PreconditionSSOR<SparseMatrix<double>> prec;
          prec.initialize(matrix_u, 1.2);
          cg.solve(matrix_u, solution_u, system_rhs, prec);

          // velocity: M v = M v_old - dt c2 K (th u + (1-th) u_old)
          laplace_matrix.vmult(system_rhs, solution_u);
          system_rhs *= -theta * dt * c2;
          laplace_matrix.vmult(tmp, old_u);
          system_rhs.add(-(1 - theta) * dt * c2, tmp);
          mass_matrix.vmult(tmp, old_v);
          system_rhs += tmp;

          SolverControl sc2(2000, 1e-12 * system_rhs.l2_norm());
          SolverCG<Vector<double>> cg2(sc2);
          PreconditionSSOR<SparseMatrix<double>> prec2;
          prec2.initialize(mass_matrix, 1.2);
          cg2.solve(mass_matrix, solution_v, system_rhs, prec2);

          old_u = solution_u;
          old_v = solution_v;
        }
      else
        {
          solution_u = old_u;
        }

      if (step % 5 == 0)
        {
          DataOut<dim> data_out;
          data_out.attach_dof_handler(dof_handler);
          data_out.add_data_vector(solution_u, "displacement");
          data_out.build_patches();
          const std::string fname = "result-" + Utilities::int_to_string(step, 4) + ".vtu";
          std::ofstream output(fname);
          data_out.write_vtu(output);
          pvd_records.emplace_back(time, fname);
        }
    }

  std::ofstream pvd("result.pvd");
  DataOutBase::write_pvd_record(pvd, pvd_records);
  std::cout << "Final u range: [" << *std::min_element(solution_u.begin(), solution_u.end())
            << ", " << *std::max_element(solution_u.begin(), solution_u.end()) << "]" << std::endl;
  std::cout << "Output written to result-*.vtu / result.pvd" << std::endl;
  return 0;
}
"""

_SRC_TIME_DEPENDENT_NS_2D = r"""/* Transient buoyancy-driven flow (Boussinesq) - deal.II (based on step-35/step-31 ideas)
 * Side-heated square cavity. Backward-Euler in time; at each step a Picard
 * iteration linearises the NS convection, and the buoyancy term
 * -beta*(T-T_ref)*g couples temperature into the momentum equation. The
 * temperature is advected/diffused by the current velocity (operator split).
 * Taylor-Hood Q2/Q1 for (u,p); Q2 for T. Self-contained; writes .vtu per step + .pvd.
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace dealii;

const unsigned int degree = @@DEGREE@@;        // pressure degree; velocity = degree+1
const unsigned int n_refine = @@REFINEMENTS@@;
const double viscosity = @@VISCOSITY@@;
const double thermal_diff = @@THERMAL_DIFFUSIVITY@@;
const double beta_buoy = @@BUOYANCY@@;          // buoyancy strength (Rayleigh-like)
const double dt = @@DT@@;
const unsigned int n_steps = @@N_STEPS@@;
const unsigned int picard_iters = @@PICARD_ITERS@@;

int main()
{
  const int dim = 2;
  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation, 0, 1, true /*colorize: 0=left,1=right,...*/);
  triangulation.refine_global(n_refine);

  // ---- Flow system (u, p): Taylor-Hood ----
  FESystem<dim> fe_flow(FE_Q<dim>(degree + 1), dim, FE_Q<dim>(degree), 1);
  DoFHandler<dim> dof_flow(triangulation);
  dof_flow.distribute_dofs(fe_flow);
  DoFRenumbering::component_wise(dof_flow);
  std::vector<unsigned int> bc(dim + 1, 0); bc[dim] = 1;
  // deal.II >= 9.2: count_dofs_per_block -> count_dofs_per_fe_block (returns).
  const std::vector<types::global_dof_index> dpb =
    DoFTools::count_dofs_per_fe_block(dof_flow, bc);
  const unsigned int n_u = dpb[0], n_p = dpb[1];

  // ---- Temperature system ----
  FE_Q<dim> fe_temp(degree + 1);
  DoFHandler<dim> dof_temp(triangulation);
  dof_temp.distribute_dofs(fe_temp);
  std::cout << "Boussinesq DOFs: u=" << n_u << " p=" << n_p
            << " T=" << dof_temp.n_dofs() << std::endl;

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  // Flow constraints: no-slip on all walls, pin pressure at dof n_u.
  AffineConstraints<double> flow_constraints;
  flow_constraints.clear();
  VectorTools::interpolate_boundary_values(dof_flow, 0,
    Functions::ZeroFunction<dim>(dim + 1), flow_constraints, fe_flow.component_mask(velocities));
  for (types::boundary_id b = 1; b < 4; ++b)
    VectorTools::interpolate_boundary_values(dof_flow, b,
      Functions::ZeroFunction<dim>(dim + 1), flow_constraints, fe_flow.component_mask(velocities));
  flow_constraints.add_line(n_u);
  flow_constraints.close();

  // Temperature constraints: hot left wall (id 0) T=1, cold right wall (id 1) T=0.
  AffineConstraints<double> temp_constraints;
  temp_constraints.clear();
  VectorTools::interpolate_boundary_values(dof_temp, 0,
    Functions::ConstantFunction<dim>(1.0), temp_constraints);
  VectorTools::interpolate_boundary_values(dof_temp, 1,
    Functions::ZeroFunction<dim>(), temp_constraints);
  temp_constraints.close();

  // Sparsity
  BlockDynamicSparsityPattern fdsp(2, 2);
  fdsp.block(0, 0).reinit(n_u, n_u); fdsp.block(0, 1).reinit(n_u, n_p);
  fdsp.block(1, 0).reinit(n_p, n_u); fdsp.block(1, 1).reinit(n_p, n_p);
  fdsp.collect_sizes();
  DoFTools::make_sparsity_pattern(dof_flow, fdsp, flow_constraints, true);
  BlockSparsityPattern flow_sp; flow_sp.copy_from(fdsp);
  BlockSparseMatrix<double> flow_matrix(flow_sp);

  DynamicSparsityPattern tdsp(dof_temp.n_dofs());
  DoFTools::make_sparsity_pattern(dof_temp, tdsp, temp_constraints, true);
  SparsityPattern temp_sp; temp_sp.copy_from(tdsp);
  SparseMatrix<double> temp_matrix(temp_sp);

  BlockVector<double> flow_solution(2), flow_rhs(2);
  for (auto *v : {&flow_solution, &flow_rhs})
    { v->block(0).reinit(n_u); v->block(1).reinit(n_p); v->collect_sizes(); }
  Vector<double> temp_solution(dof_temp.n_dofs()), old_temp(dof_temp.n_dofs()),
                 temp_rhs(dof_temp.n_dofs());
  temp_solution = 0.5; temp_constraints.distribute(temp_solution);

  const QGauss<dim> quad(degree + 2);
  const unsigned int n_q = quad.size();

  std::vector<std::pair<double, std::string>> pvd_records;

  for (unsigned int step = 1; step <= n_steps; ++step)
    {
      const double time = step * dt;
      old_temp = temp_solution;

      // ===== Picard iteration for the flow at this time level =====
      for (unsigned int pic = 0; pic < picard_iters; ++pic)
        {
          flow_matrix = 0; flow_rhs = 0;
          FEValues<dim> fev(fe_flow, quad,
            update_values | update_gradients | update_quadrature_points | update_JxW_values);
          // every flag this loop reads: a Release-only deal.II build segfaults, with no message, on a missing one
          FEValues<dim> fevT(fe_temp, quad, update_values | update_quadrature_points);
          const unsigned int dpc = fe_flow.dofs_per_cell;
          FullMatrix<double> lm(dpc, dpc); Vector<double> lr(dpc);
          std::vector<types::global_dof_index> ldi(dpc);
          std::vector<Tensor<1, dim>> u_old(n_q);
          std::vector<double> T_at_q(n_q);

          auto cellT = dof_temp.begin_active();
          for (const auto &cell : dof_flow.active_cell_iterators())
            {
              fev.reinit(cell); fevT.reinit(cellT);
              lm = 0; lr = 0;
              fev[velocities].get_function_values(flow_solution, u_old);
              fevT.get_function_values(temp_solution, T_at_q);
              for (unsigned int q = 0; q < n_q; ++q)
                {
                  const Tensor<1, dim> ulin = u_old[q];
                  const double T = T_at_q[q];
                  Tensor<1, dim> gravity; gravity[dim - 1] = -1.0;
                  for (unsigned int i = 0; i < dpc; ++i)
                    {
                      const Tensor<1, dim> phi_u_i = fev[velocities].value(i, q);
                      const Tensor<2, dim> grad_u_i = fev[velocities].gradient(i, q);
                      const double div_u_i = fev[velocities].divergence(i, q);
                      const double phi_p_i = fev[pressure].value(i, q);
                      for (unsigned int j = 0; j < dpc; ++j)
                        {
                          const Tensor<1, dim> phi_u_j = fev[velocities].value(j, q);
                          const Tensor<2, dim> grad_u_j = fev[velocities].gradient(j, q);
                          const double div_u_j = fev[velocities].divergence(j, q);
                          const double phi_p_j = fev[pressure].value(j, q);
                          lm(i, j) += ( phi_u_j * phi_u_i / dt                 // time derivative
                                       + viscosity * scalar_product(grad_u_j, grad_u_i)
                                       + (grad_u_j * ulin) * phi_u_i           // Picard convection
                                       - div_u_i * phi_p_j
                                       - phi_p_i * div_u_j ) * fev.JxW(q);
                        }
                      // RHS: u_old/dt + buoyancy
                      lr(i) += ( (ulin * phi_u_i) / dt
                                 - beta_buoy * T * (gravity * phi_u_i) ) * fev.JxW(q);
                    }
                }
              cell->get_dof_indices(ldi);
              flow_constraints.distribute_local_to_global(lm, lr, ldi, flow_matrix, flow_rhs);
              ++cellT;
            }
          SparseDirectUMFPACK fd; fd.initialize(flow_matrix);
          fd.vmult(flow_solution, flow_rhs);
          flow_constraints.distribute(flow_solution);
        }

      // ===== Temperature transport: dT/dt + u.grad T - kappa lap T = 0 =====
      {
        temp_matrix = 0; temp_rhs = 0;
        FEValues<dim> fevT(fe_temp, quad,
          update_values | update_gradients | update_quadrature_points | update_JxW_values);
        // every flag this loop reads: a Release-only deal.II build segfaults, with no message, on a missing one
        FEValues<dim> fev(fe_flow, quad, update_values | update_quadrature_points);
        const unsigned int dpc = fe_temp.dofs_per_cell;
        FullMatrix<double> lm(dpc, dpc); Vector<double> lr(dpc);
        std::vector<types::global_dof_index> ldi(dpc);
        std::vector<Tensor<1, dim>> uvals(n_q);
        std::vector<double> Told(n_q);

        auto cellF = dof_flow.begin_active();
        for (const auto &cell : dof_temp.active_cell_iterators())
          {
            fevT.reinit(cell); fev.reinit(cellF);
            lm = 0; lr = 0;
            fev[velocities].get_function_values(flow_solution, uvals);
            fevT.get_function_values(old_temp, Told);
            for (unsigned int q = 0; q < n_q; ++q)
              {
                const Tensor<1, dim> u = uvals[q];
                for (unsigned int i = 0; i < dpc; ++i)
                  {
                    const double vi = fevT.shape_value(i, q);
                    const Tensor<1, dim> gvi = fevT.shape_grad(i, q);
                    for (unsigned int j = 0; j < dpc; ++j)
                      {
                        const double uj = fevT.shape_value(j, q);
                        const Tensor<1, dim> guj = fevT.shape_grad(j, q);
                        lm(i, j) += ( uj * vi / dt
                                     + thermal_diff * (guj * gvi)
                                     + (u * guj) * vi ) * fevT.JxW(q);
                      }
                    lr(i) += (Told[q] * vi / dt) * fevT.JxW(q);
                  }
              }
            cell->get_dof_indices(ldi);
            temp_constraints.distribute_local_to_global(lm, lr, ldi, temp_matrix, temp_rhs);
            ++cellF;
          }
        SparseDirectUMFPACK td; td.initialize(temp_matrix);
        td.vmult(temp_solution, temp_rhs);
        temp_constraints.distribute(temp_solution);
      }

      if (step % 5 == 0)
        {
          DataOut<dim> data_out;
          data_out.attach_dof_handler(dof_temp);
          data_out.add_data_vector(temp_solution, "temperature");
          data_out.build_patches();
          const std::string fname = "result-" + Utilities::int_to_string(step, 4) + ".vtu";
          std::ofstream output(fname);
          data_out.write_vtu(output);
          pvd_records.emplace_back(time, fname);
          std::cout << "  step " << step << ": u_max=" << flow_solution.block(0).linfty_norm()
                    << " T in [" << *std::min_element(temp_solution.begin(), temp_solution.end())
                    << ", " << *std::max_element(temp_solution.begin(), temp_solution.end()) << "]" << std::endl;
        }
    }

  std::ofstream pvd("result.pvd");
  DataOutBase::write_pvd_record(pvd, pvd_records);
  std::cout << "Output written to result-*.vtu / result.pvd" << std::endl;
  return 0;
}
"""

_SRC_MATRIX_FREE_2D = r"""/* Matrix-free Laplace solver - deal.II (based on step-37, simplified)
 * Operator applied on-the-fly via FEEvaluation; CG + Jacobi (diagonal) preconditioner.
 * Self-contained, parameterized; writes result.vtu.
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <fstream>
#include <iostream>

using namespace dealii;

const unsigned int degree = @@DEGREE@@;
const unsigned int refinements = @@REFINEMENTS@@;

template <int dim, int fe_degree>
class LaplaceOperator
{
public:
  LaplaceOperator(const MatrixFree<dim, double> &mf) : data(mf)
  {
    data.initialize_dof_vector(diagonal);
    compute_diagonal();
  }

  void vmult(Vector<double> &dst, const Vector<double> &src) const
  {
    dst = 0;
    data.cell_loop(&LaplaceOperator::local_apply, this, dst, src, false);
  }

  const Vector<double> &get_diagonal() const { return diagonal; }

private:
  void local_apply(const MatrixFree<dim, double> &mf,
                   Vector<double> &dst, const Vector<double> &src,
                   const std::pair<unsigned int, unsigned int> &range) const
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> phi(mf);
    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src);
        phi.evaluate(EvaluationFlags::gradients);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_gradient(phi.get_gradient(q), q);
        phi.integrate(EvaluationFlags::gradients);
        phi.distribute_local_to_global(dst);
      }
  }

  void compute_diagonal()
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> phi(data);
    AlignedVector<VectorizedArray<double>> diag(phi.dofs_per_cell);
    diagonal = 0;
    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
              phi.begin_dof_values()[j] = VectorizedArray<double>();
            phi.begin_dof_values()[i] = make_vectorized_array<double>(1.0);
            phi.evaluate(EvaluationFlags::gradients);
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              phi.submit_gradient(phi.get_gradient(q), q);
            phi.integrate(EvaluationFlags::gradients);
            diag[i] = phi.begin_dof_values()[i];
          }
        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
          phi.begin_dof_values()[i] = diag[i];
        phi.distribute_local_to_global(diagonal);
      }
    for (auto &d : diagonal)
      if (std::abs(d) < 1e-14) d = 1.0;
  }

  const MatrixFree<dim, double> &data;
  Vector<double> diagonal;
};

int main()
{
  const int dim = 2;
  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(refinements);

  FE_Q<dim> fe(degree);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  std::cout << "Matrix-free DOFs: " << dof_handler.n_dofs() << std::endl;

  AffineConstraints<double> constraints;
  VectorTools::interpolate_boundary_values(dof_handler, 0,
    Functions::ZeroFunction<dim>(), constraints);
  constraints.close();

  typename MatrixFree<dim, double>::AdditionalData add_data;
  add_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
  add_data.mapping_update_flags = update_gradients | update_JxW_values;
  MatrixFree<dim, double> mf_data;
  // deal.II >= 9.3: the mapping-less MatrixFree::reinit overload was removed.
  mf_data.reinit(MappingQ1<dim>(), dof_handler, constraints,
                 QGauss<1>(degree + 1), add_data);

  LaplaceOperator<dim, degree> system(mf_data);

  Vector<double> solution, system_rhs;
  mf_data.initialize_dof_vector(solution);
  mf_data.initialize_dof_vector(system_rhs);

  // RHS: constant unit source f=1 -> assemble (phi_i, 1)
  {
    FEEvaluation<dim, degree, degree + 1, 1, double> phi(mf_data);
    for (unsigned int cell = 0; cell < mf_data.n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_value(make_vectorized_array<double>(1.0), q);
        phi.integrate(EvaluationFlags::values);
        phi.distribute_local_to_global(system_rhs);
      }
  }
  constraints.condense(system_rhs);

  // Manual Jacobi via diagonal:
  Vector<double> inv_diag = system.get_diagonal();
  for (auto &d : inv_diag) d = 1.0 / d;

  SolverControl solver_control(2000, 1e-10 * system_rhs.l2_norm());
  SolverCG<Vector<double>> solver(solver_control);
  struct DiagPrec {
    const Vector<double> *inv;
    void vmult(Vector<double> &dst, const Vector<double> &src) const {
      for (unsigned int i = 0; i < dst.size(); ++i) dst(i) = (*inv)(i) * src(i);
    }
  } diag_prec{&inv_diag};

  solver.solve(system, solution, system_rhs, diag_prec);
  constraints.distribute(solution);
  std::cout << "CG iterations: " << solver_control.last_step() << std::endl;
  std::cout << "u range: [" << *std::min_element(solution.begin(), solution.end())
            << ", " << *std::max_element(solution.begin(), solution.end()) << "]" << std::endl;

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);
  std::cout << "Output written to result.vtu" << std::endl;
  return 0;
}
"""

_SRC_MULTIGRID_2D = r"""/* Geometric multigrid preconditioner for Poisson - deal.II (based on step-16)
 * GMG V-cycle as preconditioner for CG. SSOR smoother, direct coarse solve.
 * Self-contained, parameterized; writes result.vtu.
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <fstream>
#include <iostream>

using namespace dealii;

const unsigned int degree = @@DEGREE@@;
const unsigned int n_refine = @@REFINEMENTS@@;

int main()
{
  const int dim = 2;
  Triangulation<dim> triangulation(
    Triangulation<dim>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(n_refine);

  FE_Q<dim> fe(degree);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs();
  std::cout << "Multigrid DOFs: " << dof_handler.n_dofs() << std::endl;

  AffineConstraints<double> constraints;
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(dof_handler, 0,
    Functions::ZeroFunction<dim>(), constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);
  SparseMatrix<double> system_matrix;
  system_matrix.reinit(sparsity_pattern);

  Vector<double> solution(dof_handler.n_dofs());
  Vector<double> system_rhs(dof_handler.n_dofs());

  // Multigrid level matrices
  MGConstrainedDoFs mg_constrained_dofs;
  mg_constrained_dofs.initialize(dof_handler);
  std::set<types::boundary_id> dirichlet_ids = {0};
  mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_ids);

  const unsigned int n_levels = triangulation.n_levels();
  MGLevelObject<SparsityPattern> mg_sparsity(0, n_levels - 1);
  MGLevelObject<SparseMatrix<double>> mg_matrices(0, n_levels - 1);
  MGLevelObject<SparseMatrix<double>> mg_interface(0, n_levels - 1);

  for (unsigned int level = 0; level < n_levels; ++level)
    {
      DynamicSparsityPattern level_dsp(dof_handler.n_dofs(level));
      MGTools::make_sparsity_pattern(dof_handler, level_dsp, level);
      mg_sparsity[level].copy_from(level_dsp);
      mg_matrices[level].reinit(mg_sparsity[level]);
      mg_interface[level].reinit(mg_sparsity[level]);
    }

  QGauss<dim> quadrature(degree + 1);
  FEValues<dim> fe_values(fe, quadrature,
    update_values | update_gradients | update_quadrature_points | update_JxW_values);
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Assemble global active system
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0; cell_rhs = 0;
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                   fe_values.shape_grad(j, q) * fe_values.JxW(q);
            cell_rhs(i) += fe_values.shape_value(i, q) * 1.0 * fe_values.JxW(q);
          }
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix, cell_rhs,
        local_dof_indices, system_matrix, system_rhs);
    }

  // Assemble level matrices
  std::vector<AffineConstraints<double>> boundary_constraints(n_levels);
  for (unsigned int level = 0; level < n_levels; ++level)
    {
      IndexSet relevant;
      DoFTools::extract_locally_relevant_level_dofs(dof_handler, level, relevant);
      boundary_constraints[level].reinit(relevant);
      for (const auto idx : mg_constrained_dofs.get_refinement_edge_indices(level))
        boundary_constraints[level].add_line(idx);
      for (const auto idx : mg_constrained_dofs.get_boundary_indices(level))
        boundary_constraints[level].add_line(idx);
      boundary_constraints[level].close();
    }

  for (const auto &cell : dof_handler.mg_cell_iterators())
    {
      const unsigned int level = cell->level();
      fe_values.reinit(cell);
      cell_matrix = 0;
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                 fe_values.shape_grad(j, q) * fe_values.JxW(q);
      cell->get_mg_dof_indices(local_dof_indices);
      boundary_constraints[level].distribute_local_to_global(
        cell_matrix, local_dof_indices, mg_matrices[level]);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          if (mg_constrained_dofs.is_interface_matrix_entry(level,
                local_dof_indices[i], local_dof_indices[j]))
            mg_interface[level].add(local_dof_indices[i], local_dof_indices[j],
                                    cell_matrix(i, j));
    }

  // Multigrid setup
  MGTransferPrebuilt<Vector<double>> mg_transfer(mg_constrained_dofs);
  mg_transfer.build(dof_handler);   // deal.II >= 9.2: build_matrices -> build

  FullMatrix<double> coarse_matrix;
  coarse_matrix.copy_from(mg_matrices[0]);
  MGCoarseGridHouseholder<double, Vector<double>> coarse_grid_solver;
  coarse_grid_solver.initialize(coarse_matrix);

  using Smoother = PreconditionSOR<SparseMatrix<double>>;
  mg::SmootherRelaxation<Smoother, Vector<double>> mg_smoother;
  mg_smoother.initialize(mg_matrices);
  mg_smoother.set_steps(2);
  mg_smoother.set_symmetric(true);

  mg::Matrix<Vector<double>> mg_matrix(mg_matrices);
  mg::Matrix<Vector<double>> mg_interface_up(mg_interface);
  mg::Matrix<Vector<double>> mg_interface_down(mg_interface);

  Multigrid<Vector<double>> mg(mg_matrix, coarse_grid_solver, mg_transfer,
                               mg_smoother, mg_smoother);
  mg.set_edge_matrices(mg_interface_down, mg_interface_up);

  PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>>
    preconditioner(dof_handler, mg, mg_transfer);

  SolverControl solver_control(1000, 1e-10 * system_rhs.l2_norm());
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  constraints.distribute(solution);

  std::cout << "GMG-CG iterations: " << solver_control.last_step() << std::endl;
  std::cout << "u range: [" << *std::min_element(solution.begin(), solution.end())
            << ", " << *std::max_element(solution.begin(), solution.end()) << "]" << std::endl;

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);
  std::cout << "Output written to result.vtu" << std::endl;
  return 0;
}
"""

_SRC_OBSTACLE_2D = r"""/* Obstacle problem (variational inequality) - deal.II (based on step-41)
 * Membrane pushed against a lower obstacle by a body force; primal-dual
 * active-set Newton. Self-contained, parameterized; writes result.vtu.
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

using namespace dealii;

const unsigned int n_refine = @@REFINEMENTS@@;
const double force = @@FORCE@@;     // downward body force
const double obstacle_level = @@OBSTACLE_LEVEL@@;
const unsigned int max_iterations = @@MAX_ITERATIONS@@;

int main()
{
  const int dim = 2;
  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(n_refine);

  FE_Q<dim> fe(1);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  std::cout << "Obstacle DOFs: " << dof_handler.n_dofs() << std::endl;

  AffineConstraints<double> constraints; // Dirichlet u=0 on boundary
  VectorTools::interpolate_boundary_values(dof_handler, 0,
    Functions::ZeroFunction<dim>(), constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> stiffness_matrix, system_matrix;
  stiffness_matrix.reinit(sparsity_pattern);
  system_matrix.reinit(sparsity_pattern);

  Vector<double> solution(dof_handler.n_dofs());
  Vector<double> force_rhs(dof_handler.n_dofs());
  Vector<double> diagonal_of_mass(dof_handler.n_dofs());

  QGauss<dim> quadrature(2);
  FEValues<dim> fe_values(fe, quadrature,
    update_values | update_gradients | update_quadrature_points | update_JxW_values);
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  FullMatrix<double> cell_stiffness(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell), cell_mass(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_stiffness = 0; cell_rhs = 0; cell_mass = 0;
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_stiffness(i, j) += fe_values.shape_grad(i, q) *
                                      fe_values.shape_grad(j, q) * fe_values.JxW(q);
            cell_rhs(i) += fe_values.shape_value(i, q) * force * fe_values.JxW(q);
            cell_mass(i) += fe_values.shape_value(i, q) * fe_values.JxW(q);
          }
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            stiffness_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_stiffness(i, j));
          force_rhs(local_dof_indices[i]) += cell_rhs(i);
          diagonal_of_mass(local_dof_indices[i]) += cell_mass(i);
        }
    }

  // Primal-dual active-set strategy (step-41). At each iteration the
  // active set is the set of nodes predicted to be in contact; those
  // dofs are constrained to u = obstacle_level, the rest solve the
  // free Laplace problem. The contact pressure lambda is the residual
  // of the *unconstrained* operator restricted to active nodes.
  Vector<double> lambda(dof_handler.n_dofs());
  IndexSet old_active;

  for (unsigned int iter = 0; iter < max_iterations; ++iter)
    {
      // Build active set: contact where the body would penetrate the
      // obstacle (u <= g) OR is held there with a compressive (positive)
      // contact pressure lambda. lambda is the residual K u - f, which
      // on active nodes equals the reaction force.
      IndexSet new_active(dof_handler.n_dofs());
      AffineConstraints<double> obstacle_constraints;
      for (types::global_dof_index i = 0; i < dof_handler.n_dofs(); ++i)
        {
          if (constraints.is_constrained(i))
            continue;
          if (lambda(i) - stiffness_matrix.diag_element(i) * (solution(i) - obstacle_level) > 0)
            {
              new_active.add_index(i);
              obstacle_constraints.add_line(i);
              obstacle_constraints.set_inhomogeneity(i, obstacle_level);
            }
        }
      obstacle_constraints.merge(constraints,
        AffineConstraints<double>::left_object_wins);
      obstacle_constraints.close();

      // Solve the contact-constrained Laplace problem.
      system_matrix.copy_from(stiffness_matrix);
      Vector<double> rhs = force_rhs;
      obstacle_constraints.condense(system_matrix, rhs);

      SolverControl solver_control(5000, 1e-12);
      SolverCG<Vector<double>> solver(solver_control);
      PreconditionSSOR<SparseMatrix<double>> prec;
      prec.initialize(system_matrix, 1.2);
      solver.solve(system_matrix, solution, rhs, prec);
      obstacle_constraints.distribute(solution);

      // Recompute contact pressure: lambda = f - K u (reaction), only
      // meaningful on active nodes; on free nodes it is ~0.
      stiffness_matrix.vmult(lambda, solution);
      lambda -= force_rhs;

      std::cout << "  iter " << iter << ": active set size = "
                << new_active.n_elements() << std::endl;
      if (iter > 0 && new_active == old_active)
        {
          std::cout << "Active set converged after " << iter << " iterations." << std::endl;
          break;
        }
      old_active = new_active;
    }

  std::cout << "u range: [" << *std::min_element(solution.begin(), solution.end())
            << ", " << *std::max_element(solution.begin(), solution.end()) << "]" << std::endl;

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "displacement");
  data_out.build_patches();
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);
  std::cout << "Output written to result.vtu" << std::endl;
  return 0;
}
"""

_SRC_ERROR_ESTIMATION_2D = r"""/* Adaptive error estimation - deal.II (based on step-6/step-14 KellyErrorEstimator)
 * Solves Poisson on an L-shaped domain (re-entrant corner singularity) and
 * drives several cycles of Kelly-estimator-based adaptive mesh refinement.
 * Self-contained, parameterized; writes result.vtu.
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <fstream>
#include <iostream>

using namespace dealii;

const unsigned int degree = @@DEGREE@@;
const unsigned int n_cycles = @@N_CYCLES@@;

int main()
{
  const int dim = 2;
  Triangulation<dim> triangulation;
  GridGenerator::hyper_L(triangulation, -1, 1);
  triangulation.refine_global(2);

  FE_Q<dim> fe(degree);
  DoFHandler<dim> dof_handler(triangulation);
  Vector<double> solution;

  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
    {
      dof_handler.distribute_dofs(fe);

      AffineConstraints<double> constraints;
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      VectorTools::interpolate_boundary_values(dof_handler, 0,
        Functions::ZeroFunction<dim>(), constraints);
      constraints.close();

      DynamicSparsityPattern dsp(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      SparsityPattern sparsity_pattern;
      sparsity_pattern.copy_from(dsp);
      SparseMatrix<double> system_matrix(sparsity_pattern);
      Vector<double> system_rhs(dof_handler.n_dofs());
      solution.reinit(dof_handler.n_dofs());

      QGauss<dim> quadrature(degree + 1);
      // every flag this loop reads: a Release-only deal.II build segfaults, with no message, on a missing one
      FEValues<dim> fe_values(fe, quadrature,
        update_values | update_gradients | update_JxW_values | update_quadrature_points);
      const unsigned int dofs_per_cell = fe.dofs_per_cell;
      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
      Vector<double> cell_rhs(dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          fe_values.reinit(cell);
          cell_matrix = 0; cell_rhs = 0;
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                       fe_values.shape_grad(j, q) * fe_values.JxW(q);
                cell_rhs(i) += fe_values.shape_value(i, q) * 1.0 * fe_values.JxW(q);
              }
          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix, cell_rhs,
            local_dof_indices, system_matrix, system_rhs);
        }

      SolverControl solver_control(5000, 1e-12 * system_rhs.l2_norm());
      SolverCG<Vector<double>> solver(solver_control);
      PreconditionSSOR<SparseMatrix<double>> prec;
      prec.initialize(system_matrix, 1.2);
      solver.solve(system_matrix, solution, system_rhs, prec);
      constraints.distribute(solution);

      Vector<float> estimated_error(triangulation.n_active_cells());
      KellyErrorEstimator<dim>::estimate(dof_handler, QGauss<dim - 1>(degree + 1),
        std::map<types::boundary_id, const Function<dim> *>(), solution, estimated_error);

      std::cout << "cycle " << cycle << ": cells=" << triangulation.n_active_cells()
                << " dofs=" << dof_handler.n_dofs()
                << " est.error=" << estimated_error.l2_norm() << std::endl;

      if (cycle + 1 < n_cycles)
        {
          GridRefinement::refine_and_coarsen_fixed_number(triangulation,
            estimated_error, 0.3, 0.0);
          triangulation.execute_coarsening_and_refinement();
        }
    }

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);
  std::cout << "Output written to result.vtu" << std::endl;
  return 0;
}
"""

_SRC_PHASE_FIELD_2D = r"""/* Advection-diffusion-reaction with SUPG stabilization - deal.II (step-63 flavour)
 * beta . grad(u) - eps*lap(u) + r*u = f, advection-dominated (high Peclet).
 * Streamline-upwind Petrov-Galerkin stabilization. Self-contained; writes result.vtu.
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

using namespace dealii;

const unsigned int degree = @@DEGREE@@;
const unsigned int n_refine = @@REFINEMENTS@@;
const double epsilon = @@DIFFUSION@@;   // diffusion
const double reaction = @@REACTION@@;

template <int dim>
Tensor<1, dim> beta(const Point<dim> &)
{
  Tensor<1, dim> b;
  b[0] = 1.0; b[1] = 0.5;
  return b / b.norm();
}

int main()
{
  const int dim = 2;
  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(n_refine);

  FE_Q<dim> fe(degree);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  std::cout << "ADR(SUPG) DOFs: " << dof_handler.n_dofs() << std::endl;

  AffineConstraints<double> constraints;
  VectorTools::interpolate_boundary_values(dof_handler, 0,
    Functions::ZeroFunction<dim>(), constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);
  SparseMatrix<double> system_matrix(sparsity_pattern);
  Vector<double> solution(dof_handler.n_dofs()), system_rhs(dof_handler.n_dofs());

  QGauss<dim> quadrature(degree + 2);
  FEValues<dim> fe_values(fe, quadrature,
    update_values | update_gradients | update_hessians |
    update_quadrature_points | update_JxW_values);
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0; cell_rhs = 0;
      const double h = cell->diameter();
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
          const Tensor<1, dim> b = beta(fe_values.quadrature_point(q));
          const double b_norm = b.norm();
          // SUPG parameter with doubly-asymptotic Peclet switch
          const double Pe = b_norm * h / (2.0 * epsilon);
          const double xi = (Pe > 1e-8) ? (1.0 / std::tanh(Pe) - 1.0 / Pe) : 0.0;
          const double tau = (b_norm > 1e-12) ? (h / (2.0 * b_norm) * xi) : 0.0;
          const double f = 1.0;
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const double v = fe_values.shape_value(i, q);
              const Tensor<1, dim> grad_v = fe_values.shape_grad(i, q);
              const double supg_test = tau * (b * grad_v);
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const double u = fe_values.shape_value(j, q);
                  const Tensor<1, dim> grad_u = fe_values.shape_grad(j, q);
                  // strong residual of u (no diffusion term for Q1: lap=0)
                  const double Lu = b * grad_u + reaction * u;
                  // Galerkin: eps(grad u,grad v) + (b.grad u, v) + r(u,v)
                  cell_matrix(i, j) += (epsilon * grad_u * grad_v
                                        + (b * grad_u) * v
                                        + reaction * u * v) * fe_values.JxW(q);
                  // SUPG: tau (b.grad v)(L u)
                  cell_matrix(i, j) += supg_test * Lu * fe_values.JxW(q);
                }
              cell_rhs(i) += (v + supg_test) * f * fe_values.JxW(q);
            }
        }
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix, cell_rhs,
        local_dof_indices, system_matrix, system_rhs);
    }

  SparseDirectUMFPACK direct;
  direct.initialize(system_matrix);
  direct.vmult(solution, system_rhs);
  constraints.distribute(solution);

  std::cout << "u range: [" << *std::min_element(solution.begin(), solution.end())
            << ", " << *std::max_element(solution.begin(), solution.end()) << "]" << std::endl;

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "u");
  data_out.build_patches();
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);
  std::cout << "Output written to result.vtu" << std::endl;
  return 0;
}
"""

_SRC_DG_ADVECTION_2D = r"""/* DG advection-reaction with upwind flux - deal.II (based on step-12)
 * beta . grad(u) = 0 with inflow BC; upwind numerical flux assembled by
 * hand over interior + boundary faces. Self-contained; writes result.vtu.
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

using namespace dealii;

const unsigned int degree = @@DEGREE@@;
const unsigned int n_refine = @@REFINEMENTS@@;

template <int dim>
Tensor<1, dim> beta(const Point<dim> &p)
{
  Tensor<1, dim> b;
  b[0] = -p[1];
  b[1] = p[0];
  const double n = b.norm();
  return (n > 1e-12) ? (b / n) : b;
}

// inflow boundary value: a smooth bump on part of the boundary
template <int dim>
double inflow_value(const Point<dim> &p)
{
  return (p[0] < 0.5 && p[1] < 1e-8) ? 1.0 : 0.0;
}

int main()
{
  const int dim = 2;
  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(n_refine);

  FE_DGQ<dim> fe(degree);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  std::cout << "DG advection DOFs: " << dof_handler.n_dofs() << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);
  SparseMatrix<double> system_matrix(sparsity_pattern);
  Vector<double> solution(dof_handler.n_dofs()), system_rhs(dof_handler.n_dofs());

  const QGauss<dim> quad(degree + 1);
  const QGauss<dim - 1> face_quad(degree + 1);
  FEValues<dim> fe_v(fe, quad,
    update_values | update_gradients | update_quadrature_points | update_JxW_values);
  FEFaceValues<dim> fe_f(fe, face_quad,
    update_values | update_quadrature_points | update_normal_vectors | update_JxW_values);
  // every flag this loop reads: a Release-only deal.II build segfaults, with no message, on a missing one
  FEFaceValues<dim> fe_f_neighbor(fe, face_quad, update_values | update_quadrature_points);

  const unsigned int dpc = fe.dofs_per_cell;
  std::vector<types::global_dof_index> dofs(dpc), dofs_neighbor(dpc);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_v.reinit(cell);
      cell->get_dof_indices(dofs);
      FullMatrix<double> ui_vi(dpc, dpc);

      // Volume term: -(beta u, grad v)  [weak form of beta.grad u]
      for (unsigned int q = 0; q < quad.size(); ++q)
        {
          const Tensor<1, dim> b = beta(fe_v.quadrature_point(q));
          for (unsigned int i = 0; i < dpc; ++i)
            for (unsigned int j = 0; j < dpc; ++j)
              ui_vi(i, j) -= b * fe_v.shape_grad(i, q) *
                             fe_v.shape_value(j, q) * fe_v.JxW(q);
        }
      for (unsigned int i = 0; i < dpc; ++i)
        for (unsigned int j = 0; j < dpc; ++j)
          system_matrix.add(dofs[i], dofs[j], ui_vi(i, j));

      // Face terms
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          fe_f.reinit(cell, f);
          if (cell->face(f)->at_boundary())
            {
              for (unsigned int q = 0; q < face_quad.size(); ++q)
                {
                  const Tensor<1, dim> b = beta(fe_f.quadrature_point(q));
                  const double bn = b * fe_f.normal_vector(q);
                  if (bn > 0) // outflow: upwind = interior
                    {
                      for (unsigned int i = 0; i < dpc; ++i)
                        for (unsigned int j = 0; j < dpc; ++j)
                          system_matrix.add(dofs[i], dofs[j],
                            bn * fe_f.shape_value(i, q) * fe_f.shape_value(j, q) * fe_f.JxW(q));
                    }
                  else // inflow: u taken from boundary data -> RHS
                    {
                      const double g = inflow_value(fe_f.quadrature_point(q));
                      for (unsigned int i = 0; i < dpc; ++i)
                        system_rhs(dofs[i]) -= bn * g * fe_f.shape_value(i, q) * fe_f.JxW(q);
                    }
                }
            }
          else if (cell->neighbor(f)->id() < cell->id())
            continue; // process each interior face once from the lower-id side
          else
            {
              const auto neighbor = cell->neighbor(f);
              const unsigned int nf = cell->neighbor_of_neighbor(f);
              fe_f_neighbor.reinit(neighbor, nf);
              neighbor->get_dof_indices(dofs_neighbor);

              for (unsigned int q = 0; q < face_quad.size(); ++q)
                {
                  const Tensor<1, dim> b = beta(fe_f.quadrature_point(q));
                  const double bn = b * fe_f.normal_vector(q);
                  // upwind: if bn>0 flux uses this cell's value, else neighbor's
                  for (unsigned int i = 0; i < dpc; ++i)
                    for (unsigned int j = 0; j < dpc; ++j)
                      {
                        if (bn > 0)
                          {
                            system_matrix.add(dofs[i], dofs[j],
                              bn * fe_f.shape_value(i, q) * fe_f.shape_value(j, q) * fe_f.JxW(q));
                            system_matrix.add(dofs_neighbor[i], dofs[j],
                              -bn * fe_f_neighbor.shape_value(i, q) * fe_f.shape_value(j, q) * fe_f.JxW(q));
                          }
                        else
                          {
                            system_matrix.add(dofs[i], dofs_neighbor[j],
                              bn * fe_f.shape_value(i, q) * fe_f_neighbor.shape_value(j, q) * fe_f.JxW(q));
                            system_matrix.add(dofs_neighbor[i], dofs_neighbor[j],
                              -bn * fe_f_neighbor.shape_value(i, q) * fe_f_neighbor.shape_value(j, q) * fe_f.JxW(q));
                          }
                      }
                }
            }
        }
    }

  SparseDirectUMFPACK direct;
  direct.initialize(system_matrix);
  direct.vmult(solution, system_rhs);

  std::cout << "u range: [" << *std::min_element(solution.begin(), solution.end())
            << ", " << *std::max_element(solution.begin(), solution.end()) << "]" << std::endl;

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "u");
  data_out.build_patches();
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);
  std::cout << "Output written to result.vtu" << std::endl;
  return 0;
}
"""



def _mixed_laplacian_2d(params: dict) -> str:
    """Real, self-contained parameterized deal.II program."""
    degree = int(params.get("degree", 0))
    refinements = int(params.get("refinements", 4))
    src = _SRC_MIXED_LAPLACIAN_2D
    src = src.replace("@@DEGREE@@", str(degree))
    src = src.replace("@@REFINEMENTS@@", str(refinements))
    return src

def _time_dependent_heat_2d(params: dict) -> str:
    """Real, self-contained parameterized deal.II program."""
    refinements = int(params.get("refinements", 4))
    n_steps = int(params.get("n_steps", 30))
    dt = float(params.get("dt", 0.01))
    alpha = float(params.get("alpha", 1.0))
    theta = float(params.get("theta", 0.5))
    src = _SRC_TIME_DEPENDENT_HEAT_2D
    src = src.replace("@@REFINEMENTS@@", str(refinements))
    src = src.replace("@@N_STEPS@@", str(n_steps))
    src = src.replace("@@DT@@", str(dt))
    src = src.replace("@@ALPHA@@", str(alpha))
    src = src.replace("@@THETA@@", str(theta))
    return src

def _time_dependent_wave_2d(params: dict) -> str:
    """Real, self-contained parameterized deal.II program."""
    refinements = int(params.get("refinements", 5))
    n_steps = int(params.get("n_steps", 40))
    dt = float(params.get("dt", 0.02))
    wave_speed = float(params.get("wave_speed", 1.0))
    src = _SRC_TIME_DEPENDENT_WAVE_2D
    src = src.replace("@@REFINEMENTS@@", str(refinements))
    src = src.replace("@@N_STEPS@@", str(n_steps))
    src = src.replace("@@DT@@", str(dt))
    src = src.replace("@@WAVE_SPEED@@", str(wave_speed))
    return src

def _time_dependent_ns_2d(params: dict) -> str:
    """Real, self-contained parameterized deal.II program."""
    degree = int(params.get("degree", 1))
    refinements = int(params.get("refinements", 4))
    viscosity = float(params.get("viscosity", 0.05))
    thermal_diffusivity = float(params.get("thermal_diffusivity", 0.05))
    buoyancy = float(params.get("buoyancy", 5.0))
    dt = float(params.get("dt", 0.02))
    n_steps = int(params.get("n_steps", 25))
    picard_iters = int(params.get("picard_iters", 3))
    src = _SRC_TIME_DEPENDENT_NS_2D
    src = src.replace("@@DEGREE@@", str(degree))
    src = src.replace("@@REFINEMENTS@@", str(refinements))
    src = src.replace("@@VISCOSITY@@", str(viscosity))
    src = src.replace("@@THERMAL_DIFFUSIVITY@@", str(thermal_diffusivity))
    src = src.replace("@@BUOYANCY@@", str(buoyancy))
    src = src.replace("@@DT@@", str(dt))
    src = src.replace("@@N_STEPS@@", str(n_steps))
    src = src.replace("@@PICARD_ITERS@@", str(picard_iters))
    return src

def _matrix_free_2d(params: dict) -> str:
    """Real, self-contained parameterized deal.II program."""
    degree = int(params.get("degree", 2))
    refinements = int(params.get("refinements", 5))
    src = _SRC_MATRIX_FREE_2D
    src = src.replace("@@DEGREE@@", str(degree))
    src = src.replace("@@REFINEMENTS@@", str(refinements))
    return src

def _multigrid_2d(params: dict) -> str:
    """Real, self-contained parameterized deal.II program."""
    degree = int(params.get("degree", 2))
    refinements = int(params.get("refinements", 5))
    src = _SRC_MULTIGRID_2D
    src = src.replace("@@DEGREE@@", str(degree))
    src = src.replace("@@REFINEMENTS@@", str(refinements))
    return src

def _obstacle_2d(params: dict) -> str:
    """Real, self-contained parameterized deal.II program."""
    refinements = int(params.get("refinements", 6))
    force = float(params.get("force", -10.0))
    obstacle_level = float(params.get("obstacle_level", -0.5))
    max_iterations = int(params.get("max_iterations", 30))
    src = _SRC_OBSTACLE_2D
    src = src.replace("@@REFINEMENTS@@", str(refinements))
    src = src.replace("@@FORCE@@", str(force))
    src = src.replace("@@OBSTACLE_LEVEL@@", str(obstacle_level))
    src = src.replace("@@MAX_ITERATIONS@@", str(max_iterations))
    return src

def _error_estimation_2d(params: dict) -> str:
    """Real, self-contained parameterized deal.II program."""
    degree = int(params.get("degree", 2))
    n_cycles = int(params.get("n_cycles", 6))
    src = _SRC_ERROR_ESTIMATION_2D
    src = src.replace("@@DEGREE@@", str(degree))
    src = src.replace("@@N_CYCLES@@", str(n_cycles))
    return src

def _phase_field_2d(params: dict) -> str:
    """Real, self-contained parameterized deal.II program."""
    degree = int(params.get("degree", 1))
    refinements = int(params.get("refinements", 6))
    diffusion = float(params.get("diffusion", 1e-3))
    reaction = float(params.get("reaction", 1.0))
    src = _SRC_PHASE_FIELD_2D
    src = src.replace("@@DEGREE@@", str(degree))
    src = src.replace("@@REFINEMENTS@@", str(refinements))
    src = src.replace("@@DIFFUSION@@", str(diffusion))
    src = src.replace("@@REACTION@@", str(reaction))
    return src

def _dg_advection_2d(params: dict) -> str:
    """Real, self-contained parameterized deal.II program."""
    degree = int(params.get("degree", 1))
    refinements = int(params.get("refinements", 5))
    src = _SRC_DG_ADVECTION_2D
    src = src.replace("@@DEGREE@@", str(degree))
    src = src.replace("@@REFINEMENTS@@", str(refinements))
    return src



KNOWLEDGE = {
    "mixed_laplacian": {
        "description": "Mixed Laplacian with Raviart-Thomas H(div) elements (step-20)",
        "function_space": "FE_RaviartThomas + FE_DGQ for flux-pressure formulation",
        "pitfalls": [
            (
                "[API] H(div) elements (FE_RaviartThomas) have a "
                "DIFFERENT DOF structure from H1 — DoFs live on "
                "faces, not vertices. Signal: post-processing "
                "treating RT DoFs as nodal (e.g. "
                "DataOut::add_data_vector(..., DataOutBase::"
                "vertex_data)) raises an `ExcInternalError` or "
                "produces a per-vertex flux field that does not "
                "match the cell-face flux integral. Use "
                "DataOutBase::DG output or interpolate to a P1 "
                "post-processing space."
            ),
            (
                "[Numerical] Schur complement solver for saddle-"
                "point system. Signal: a plain CG on the full "
                "(u, p) block matrix diverges because the system "
                "is indefinite; standard recipe is "
                "S = -B M^{-1} B^T and a CG on S with M^{-1} as "
                "preconditioner (step-20 / step-22). Without "
                "Schur complement reformulation, MINRES (or GMRES "
                "with a block preconditioner) is the only "
                "robust option."
            ),
        ],
    },
    "time_dependent_heat": {
        "description": "Transient heat equation with AMR (step-26)",
        "time_integration": ["backward Euler", "Crank-Nicolson", "BDF2"],
        "pitfalls": [
            (
                "[API] Adaptive mesh refinement requires solution "
                "transfer between meshes. Signal: refining the "
                "mesh between time steps without "
                "SolutionTransfer<dim>::interpolate produces a "
                "zero-vector or random-noise solution on the new "
                "mesh — the previous solution lives on the OLD "
                "DoFHandler and is invalidated by refinement. "
                "step-26 shows the canonical SolutionTransfer "
                "(prepare_for_coarsening_and_refinement -> "
                "refine -> interpolate) sequence."
            ),
            (
                "[Numerical] CFL for explicit; unconditionally "
                "stable for implicit. Signal: explicit Euler on "
                "the heat equation diverges (NaN within ~10 "
                "steps) at dt > h^2/(2*alpha) — SUNDIALS::ARKode "
                "reports step rejection or SolverControl::"
                "failure; backward Euler / Crank-Nicolson via "
                "SUNDIALS::IDA are unconditionally stable but "
                "CN can oscillate at sharp fronts. Choose "
                "implicit for any production heat problem; use "
                "explicit only for didactic comparisons."
            ),
        ],
    },
    "time_dependent_wave": {
        "description": "Second-order wave equation (step-23, step-48)",
        "time_integration": ["Newmark-beta", "leapfrog"],
        "pitfalls": [
            (
                "[Numerical] Energy conservation — use symplectic "
                "integrators. Signal: integrating the wave "
                "equation with implicit Euler shows a "
                "monotonically DECAYING total energy "
                "(0.5 ||u_t||^2 + 0.5 ||grad u||^2) computed via "
                "VectorTools::integrate_difference — non-"
                "physical numerical dissipation. Leapfrog / "
                "Newmark-beta with beta=0.25, gamma=0.5 "
                "(step-23 demonstrates via DoFHandler + "
                "AffineConstraints) conserve energy to "
                "roundoff."
            ),
            (
                "[Numerical] CFL: dt < h/c for explicit schemes. "
                "Signal: leapfrog at dt > h/c oscillates with "
                "exponentially growing amplitude (factor of ~2 "
                "per step) — classic explicit-wave instability. "
                "Safety factor 0.5*h/c is conservative; CFL=1 is "
                "the strict bound."
            ),
        ],
    },
    "time_dependent_ns": {
        "description": "Transient Boussinesq flow — buoyancy-driven convection (step-35)",
        "pitfalls": [
            (
                "[Numerical] Rayleigh number controls flow "
                "regime. Signal: at Ra < 1707 (critical Rayleigh "
                "for Bénard convection) the simulation correctly "
                "shows a conductive (pure-diffusion) steady "
                "state; above that, convective cells appear. "
                "Computing at Ra > 1e6 without a turbulence model "
                "produces visibly chaotic transient behaviour "
                "that does not match a laminar DNS — switch to "
                "LES or RANS for very-high-Ra regimes."
            ),
            (
                "[Numerical] Requires NS + energy equation "
                "coupling. Signal: solving NS in isolation (no "
                "buoyancy term in momentum) gives ZERO flow on a "
                "side-heated cavity — the canonical Boussinesq "
                "test. The momentum equation needs "
                "-rho * beta * (T - T_ref) * g_hat as a "
                "FEValuesExtractors::Vector source; without it "
                "the BlockVector temperature component stays "
                "decoupled. step-35 implements via "
                "BlockSparseMatrix and DoFTools::"
                "make_sparsity_pattern."
            ),
        ],
    },
    "matrix_free": {
        "description": ("Matrix-free operator evaluation (step-37, "
                        "step-59, step-75). Verified working in 3D on this "
                        "catalog's supported deal.II: FE_Q(2) and FE_Q(4) "
                        "Laplace at ~275k DoFs, CG + PreconditionChebyshev "
                        "on the matrix-free diagonal, with EITHER "
                        "LinearAlgebra::distributed::Vector<double> or a "
                        "plain dealii::Vector<double> — no MPI required."),
        "performance": ("[Performance] The honest claim is MEMORY, not a fixed speed "
                        "factor. Measured for a 3D FE_Q(4) Laplace operator: "
                        "SparseMatrix::memory_consumption() was tens of MB "
                        "against under 1 MB for MatrixFree::memory_"
                        "consumption() — nearly two orders of magnitude "
                        "less — while the operator APPLICATION was under 4x "
                        "faster, not the '10-100x' this field used to "
                        "claim. The apply speed-up depends on degree, "
                        "dimension and the SIMD width the library was built "
                        "with (read VectorizedArray<double>::size(); it was "
                        "2 on the machine measured, i.e. 128-bit SSE2, and "
                        "is 4 or 8 on AVX2/AVX-512 hardware). Matrix-free "
                        "also skips ASSEMBLY entirely: assembling the sparse "
                        "matrix took hundreds of times longer than "
                        "MatrixFree::reinit, so for one-shot solves the "
                        "win is much larger than the per-apply ratio. "
                        "Signal: print VectorizedArray<double>::size(), "
                        "MatrixFree::memory_consumption() and "
                        "SparseMatrix::memory_consumption() and compare "
                        "those, rather than trusting any quoted factor."),
        "pitfalls": [
            (
                "[API] MatrixFree needs an element whose shape "
                "functions it knows how to evaluate with sum "
                "factorization. The old rule of thumb 'tensor-"
                "product elements only, so FE_Q/FE_DGQ but NOT "
                "FE_SimplexP' is OUT OF DATE and this entry used to "
                "repeat it. Verified on deal.II 9.8: "
                "MatrixFree<3>::reinit SUCCEEDS on FE_SimplexP(2) "
                "over a tetrahedral mesh built with "
                "GridGenerator::subdivided_hyper_cube_with_"
                "simplices, and the resulting FEEvaluation cell_loop "
                "reproduces the assembled matrix-vector product to "
                "round-off (relative difference ~3e-16) — with both "
                "the runtime form FEEvaluation<3,-1,0,1,double> and "
                "a correctly templated one. "
                "What DOES still fail is the vector-valued "
                "moment-based families. Signal: MatrixFree<dim>::"
                "reinit with FE_RaviartThomas / FE_BDM ABORTS the "
                "process — it does not silently disable "
                "vectorization and it does NOT throw a catchable "
                "exception. The failure is DEAL_II_NOT_IMPLEMENTED() "
                "inside internal::MatrixFreeFunctions::get_element_"
                "type_specific_information, which aborts rather than "
                "throwing, so try/catch (const std::exception&) does "
                "NOT run: the process dies with SIGABRT (exit 134) "
                "after printing 'You are trying to use functionality "
                "in deal.II that is currently not implemented' plus "
                "a stacktrace. It is not gated on NDEBUG, so it "
                "fires on a Release build too. (It is NOT the string "
                "'MatrixFree: element type not supported', which "
                "this entry used to quote and which does not exist.) "
                "Because you cannot recover from an abort, decide "
                "BEFORE reinit: test the element you intend to use "
                "on a tiny mesh first."
            ),
            (
                "[API] The FEEvaluation TEMPLATE ARGUMENTS must "
                "match what MatrixFree was built with, and a "
                "mismatch is SILENT in Release. FEEvaluation<dim, "
                "fe_degree, n_q_points_1d, n_components, Number> "
                "hard-codes the degree and the 1D quadrature count "
                "into the sum-factorization loops. Verified on the "
                "same MatrixFree object (FE_Q(2) with QGauss<1>(3)) "
                "by varying only the template arguments: a matching "
                "<3,2,3,1,double> reproduced the assembled product "
                "to round-off; a degree too HIGH produced NaN; a "
                "degree too LOW produced a result off by nearly an "
                "order of magnitude; an n_q_points_1d too high "
                "produced a result off by nearly three orders of "
                "magnitude; too low, off by a factor of about six. "
                "NONE of the four raised anything — every one 'ran "
                "without error'. "
                "On a DEBUG build the same mismatches abort with a "
                "complete diagnosis: 'Illegal arguments in "
                "constructor/wrong template arguments! Called --> "
                "FEEvaluation<dim,3,4,1,Number>(data, 0, 0, 0)' "
                "from FEEvaluation::check_template_arguments. "
                "Signal: if you cannot build in Debug, use the "
                "RUNTIME form FEEvaluation<dim, -1, 0, n_components, "
                "Number>, which reads the degree from the "
                "MatrixFree object and was bit-for-bit as accurate "
                "as the correctly templated one (it is slower "
                "because the loops are no longer unrolled at "
                "compile time). Otherwise cross-check one "
                "matrix-vector product against an assembled "
                "SparseMatrix on a small mesh before scaling up."
            ),
            (
                "[Numerical] The AffineConstraints object MUST be "
                "passed to MatrixFree::reinit. The matrix-free "
                "operator has no rows to eliminate, so constraints "
                "are applied inside read_dof_values / "
                "distribute_local_to_global; hand reinit an empty "
                "AffineConstraints and the Dirichlet conditions "
                "simply do not exist. Nothing is raised. Verified on "
                "a 3D FE_Q(2) Laplace problem, changing ONLY the "
                "constraints object handed to reinit: with the real "
                "constraints, MatrixFree::get_constrained_dofs()"
                ".size() was in the thousands and CG converged in "
                "about a dozen iterations to a solution that was "
                "exactly zero on the boundary; with an empty one, "
                "get_constrained_dofs().size() was 0 and CG ran to "
                "its iteration limit and threw "
                "SolverControl::NoConvergence with the residual "
                "BLOWN UP by many orders of magnitude ('Iterative "
                "method reported convergence failure in step <N>. "
                "The residual in the last step was <huge>.'). "
                "Signal: read MatrixFree::get_constrained_dofs()"
                ".size() straight after reinit — zero on a problem "
                "with Dirichlet data means the constraints never "
                "arrived. Do not wait for the solver: on a pure "
                "Neumann-shaped operator it may converge to a "
                "quietly wrong answer instead of diverging."
            ),
            (
                "[Performance] No matrix assembly — operator is "
                "applied on-the-fly. Signal: profiling shows zero "
                "time in SparseMatrix::add() (no global matrix); "
                "the bulk of wall-clock should be in MatrixFree::"
                "cell_loop and FEEvaluation::evaluate / "
                "integrate. If matrix-related calls appear, the "
                "code accidentally falls back to a sparse "
                "path."
            ),
            (
                "[Numerical] Geometric multigrid essential for "
                "preconditioning. Signal: SolverCG without "
                "PreconditionAMG/MGSmootherRelaxation on a "
                "MatrixFree Laplace problem converges in "
                "~O(h^-1) iterations (gets worse with refinement, "
                "visible in SolverControl::log_history()); "
                "GMG keeps it at ~10-20 iterations independent "
                "of h. step-37 / step-50 show the canonical "
                "MatrixFree + GMG combination."
            ),
        ],
    },
    "multigrid": {
        "description": ("Geometric multigrid preconditioner (step-16 "
                        "matrix-based, step-37/50 matrix-free, step-56 "
                        "Vanka for Stokes). Verified working in 3D on this "
                        "catalog's supported deal.II with no external "
                        "dependencies: MGTransferPrebuilt + "
                        "mg::SmootherRelaxation + MGCoarseGridHouseholder, "
                        "wrapped in PreconditionMG inside an outer "
                        "SolverCG, on both uniformly and adaptively "
                        "refined hexahedral meshes."),
        "types": ["h-multigrid (mesh hierarchy)", "p-multigrid (polynomial degree)"],
        "why_bother": (
            "h-INDEPENDENCE is the whole point, and it is the thing to "
            "check. Verified on a 3D FE_Q(2) Poisson problem over four "
            "uniform refinement levels spanning nearly three orders of "
            "magnitude in DoF count: the outer CG iteration count with "
            "GMG stayed in the single digits and barely moved across all "
            "four levels, while the same CG with PreconditionSSOR and "
            "with no preconditioner both grew steadily and roughly "
            "doubled per refinement. On an adaptively refined mesh with "
            "hanging nodes the GMG count rose slightly but stayed far "
            "below the alternatives. "
            "Note GMG was NOT the fastest in wall-clock on the small "
            "meshes — setup dominates there — and only won once the "
            "problem was large. Do not judge it on a toy mesh; judge it "
            "on the SLOPE of iterations against refinement."
        ),
        "required_setup_order": [
            "REQUIRED 1. Construct the Triangulation with "
            "Triangulation<dim>::MeshSmoothing::"
            "limit_level_difference_at_vertices — the level hierarchy "
            "needs it on adaptive meshes.",
            "REQUIRED 2. dof_handler.distribute_dofs(fe) AND "
            "dof_handler.distribute_mg_dofs(). The second call is the "
            "one everybody forgets.",
            "REQUIRED 3. MGConstrainedDoFs::initialize(dof_handler), "
            "then make_zero_boundary_constraints(dof_handler, "
            "{boundary ids}) for Dirichlet boundaries.",
            "REQUIRED 4. Assemble one matrix per level, plus the "
            "refinement-edge interface matrices on adaptive meshes.",
            "REQUIRED 5. MGTransferPrebuilt<Vector> transfer("
            "mg_constrained_dofs); transfer.build(dof_handler).",
            "REQUIRED 6. Wrap Multigrid<Vector> in PreconditionMG and "
            "hand THAT to the outer SolverCG.",
        ],
        "pitfalls": [
            (
                "[API] Forgetting dof_handler.distribute_mg_dofs() is "
                "the single most common multigrid bug, and what you "
                "see depends entirely on the build type. On a DEBUG "
                "build each of the three usual next calls aborts with "
                "an explicit diagnosis, verified one by one: "
                "dof_handler.n_dofs(level) gives 'n_dofs(level) can "
                "only be called after distribute_mg_dofs()'; "
                "MGTransferPrebuilt::build(dof_handler) gives 'The "
                "underlying DoFHandler object has not had its "
                "distribute_mg_dofs() function called, but this is a "
                "prerequisite for multigrid transfers.'; "
                "MGConstrainedDoFs::initialize(dof_handler) gives "
                "'The level dofs are not set up properly! Did you call "
                "distribute_mg_dofs()?'. On a RELEASE build all three "
                "SEGFAULT (exit 139) with no message whatsoever. "
                "Signal: call dof_handler.has_level_dofs() yourself "
                "and require it to be true before touching any mg_* "
                "API — it is cheap and works in both builds."
            ),
            (
                "[Numerical] MGConstrainedDoFs must be told about the "
                "Dirichlet boundary, otherwise the coarse-level "
                "operators are singular and the V-cycle is not a "
                "valid preconditioner. Signal: after "
                "make_zero_boundary_constraints, "
                "MGConstrainedDoFs::have_boundary_indices() must be "
                "TRUE and get_boundary_indices(level) must be "
                "non-empty on EVERY level, not just the finest — "
                "verified on a working 3D solve, where each level "
                "carried its own boundary index set. On an "
                "adaptively refined mesh check "
                "get_refinement_edge_indices(level) too: it is empty "
                "on the levels that are fully refined and non-empty "
                "exactly on the levels where the hierarchy has a "
                "refinement edge. If those sets are empty on a mesh "
                "you know has hanging nodes, the interface handling "
                "is missing and the iteration count will drift with "
                "refinement instead of staying flat."
            ),
            (
                "[Numerical] Smoother choice: PreconditionChebyshev "
                "or a relaxation smoother (SOR/Jacobi via "
                "mg::SmootherRelaxation) for SPD problems; a few "
                "steps of a Krylov smoother for indefinite ones. "
                "Signal: applying an SPD smoother to an indefinite "
                "saddle-point system produces V-cycles whose residual "
                "GROWS from cycle to cycle rather than shrinking — "
                "watch SolverControl::last_value() across the outer "
                "iterations. For Stokes use a Vanka-type smoother "
                "(step-56), not point Jacobi: the pressure block has "
                "a zero diagonal, so point Jacobi is undefined there."
            ),
            (
                "[Numerical] Coarse-grid solver: solve the coarsest "
                "level EXACTLY. A direct solve (MGCoarseGridHouseholder "
                "on the dense coarse matrix, or SparseDirectUMFPACK) "
                "keeps the iteration count h-independent; leaving a "
                "relaxation smoother there makes the V-cycle only as "
                "good as that smoother on the coarse problem. Keep "
                "the coarsest level genuinely small — the working 3D "
                "setup verified here coarsened down to a few dozen "
                "DoFs, where a dense direct solve is free. On very "
                "large hierarchies switch to an iterative coarse "
                "solver to avoid the direct solver's memory. Signal: "
                "an iteration count that grows with the COARSE-level "
                "DoF count while the fine level is unchanged."
            ),
        ],
    },
    "obstacle_problem": {
        "description": "Variational inequality / contact / obstacle problem (step-41)",
        "method": "Active set strategy — project onto feasible set each Newton step",
        "pitfalls": [
            (
                "[Numerical] Non-smooth problem — requires special "
                "solver (active set, penalty). Signal: a "
                "vanilla Newton SolverControl loop on a "
                "variational inequality (elastic body pressing "
                "into a rigid obstacle) either diverges or "
                "oscillates between two active-set "
                "AffineConstraints states without converging; "
                "step-41's active-set strategy iterates "
                "(IndexSet constraint-detection -> linear solve) "
                "until two consecutive active sets are identical, "
                "typically 3-10 outer iterations."
            ),
        ],
    },
    "error_estimation": {
        "description": "Dual-weighted residual (DWR) error estimation (step-14, step-74)",
        "method": "Solve dual/adjoint problem, weight residual for goal-oriented refinement",
        "pitfalls": [
            (
                "[Numerical] Dual problem requires adjoint "
                "assembly. Signal: a goal-oriented refinement "
                "loop driven only by the PRIMAL residual (no "
                "dual) refines uniformly toward singularities "
                "regardless of the goal functional; the "
                "effectivity index (estimated / true error) is "
                "typically O(1) to 10x off without the dual. "
                "step-14 / step-74 show the canonical DWR "
                "(residual * weighted-dual) loop."
            ),
            (
                "[Numerical] Higher-order dual solution needed "
                "for effectivity index. Signal: solving the dual "
                "in the SAME finite-element space as the primal "
                "makes the dual-weighted residual collapse to "
                "zero (Galerkin orthogonality) — the effectivity "
                "index is trivially 1 but the estimator gives "
                "zero refinement information. Use one order "
                "higher (or a patch-wise enrichment) for the "
                "dual."
            ),
        ],
    },
    "phase_field": {
        "description": "Phase-field / advection-diffusion-reaction with SUPG (step-63)",
        "pitfalls": [
            (
                "[Numerical] SUPG stabilization for advection-"
                "dominated problems. Signal: a Galerkin "
                "discretisation at Peclet number > 1 produces "
                "wiggles in the boundary layer (visible in "
                "DataOut::write_vtu output as high-frequency "
                "oscillations near walls) that do not damp with "
                "refinement. Add a SUPG term "
                "tau * (b . grad phi) * (b . grad u - source) "
                "inside the FEValues quadrature loop; "
                "oscillations disappear. step-63 demonstrates "
                "this stabilisation."
            ),
            (
                "[Numerical] Peclet number determines "
                "stabilization strength. Signal: a fixed SUPG "
                "tau = h/(2*|b|) over-stabilises in diffusion-"
                "dominated zones (smearing of sharp gradients "
                "by ~20% even in clearly resolved regions, "
                "diagnosable via KellyErrorEstimator). Use "
                "tau = h/(2*|b|) * f(Pe_h) where f(Pe) is the "
                "doubly-asymptotic switch (coth(Pe) - 1/Pe)."
            ),
        ],
    },
    "dg_advection_reaction": {
        "description": "DG for advection with upwind flux (step-12, step-39)",
        "pitfalls": [
            (
                "[Numerical] Upwind flux for stability. Signal: a "
                "DG advection discretisation with a CENTRAL "
                "numerical flux (0.5 * (u^+ + u^-)) on a "
                "pure-advection problem is unstable — the "
                "solution amplitude grows like ~exp(t) regardless "
                "of mesh size. Use upwind: u_hat = u^- (downstream "
                "cell takes upstream value); stability is "
                "recovered."
            ),
            (
                "[Numerical] DG + multigrid in step-39. Signal: a "
                "plain SolverCG / SolverGMRES on the DG-"
                "advection system shows iteration count "
                "proportional to mesh Reynolds number (Pe_h) — "
                "not h-independent (visible in SolverControl "
                "history). Geometric multigrid with "
                "PreconditionBlock (block-Jacobi over the DG "
                "block + DoFRenumbering::downstream) restores "
                "h-independent convergence; see step-39 / "
                "step-50."
            ),
        ],
    },
}


GENERATORS = {
    "mixed_laplacian_2d": _mixed_laplacian_2d,
    "time_dependent_heat_2d": _time_dependent_heat_2d,
    "time_dependent_wave_2d": _time_dependent_wave_2d,
    "time_dependent_ns_2d": _time_dependent_ns_2d,
    "matrix_free_2d": _matrix_free_2d,
    "multigrid_2d": _multigrid_2d,
    "obstacle_problem_2d": _obstacle_2d,
    "error_estimation_2d": _error_estimation_2d,
    "phase_field_2d": _phase_field_2d,
    "dg_advection_reaction_2d": _dg_advection_2d,
}
