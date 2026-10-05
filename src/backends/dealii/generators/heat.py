"""Heat equation templates for deal.II.

Based on deal.II tutorial step-26 (transient) and steady-state variants.
"""


def _heat_2d_transient(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    Based on deal.II step-26 pattern (simplified).
    """
    degree = int(params.get("degree", params.get("order", 1)))
    if degree < 1:
        raise ValueError(
            f"_heat_2d_transient: degree must be >= 1, got {degree!r}")
    refinements = params.get("refinements", 4)
    n_steps = params.get("n_steps", 20)
    dt = params.get("dt", 0.05)
    return f'''\
/* Transient heat equation on unit square — deal.II
 * dT/dt - laplacian(T) = 0, prescribed temperatures on boundaries
 * Backward Euler time stepping.
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
#include <deal.II/base/function.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iostream>

using namespace dealii;

// Boundary value function — returns prescribed temperature on left edge
template <int dim>
class LeftBoundary : public Function<dim>
{{
public:
  virtual double value(const Point<dim> &p, const unsigned int) const override
  {{
    return (p[0] < 1e-10) ? 1.0 : 0.0;
  }}
}};

int main()
{{
  Triangulation<2> triangulation;
  GridGenerator::hyper_rectangle(triangulation, Point<2>(0,0), Point<2>(1,1), true);
  triangulation.refine_global({refinements});

  FE_Q<2> fe({degree});
  DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  std::cout << "Heat DOFs: " << dof_handler.n_dofs() << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> mass_matrix, stiffness_matrix, system_matrix;
  mass_matrix.reinit(sparsity_pattern);
  stiffness_matrix.reinit(sparsity_pattern);
  system_matrix.reinit(sparsity_pattern);

  Vector<double> solution(dof_handler.n_dofs());
  Vector<double> old_solution(dof_handler.n_dofs());
  Vector<double> system_rhs(dof_handler.n_dofs());

  // Assemble mass and stiffness
  MatrixTools::create_mass_matrix(dof_handler, QGauss<2>(fe.degree+1), mass_matrix);
  MatrixTools::create_laplace_matrix(dof_handler, QGauss<2>(fe.degree+1), stiffness_matrix);

  // Initial condition: T=0
  solution = 0;

  double dt = {dt};
  int n_steps = {n_steps};

  for (int step = 0; step < n_steps; ++step)
    {{
      old_solution = solution;

      // System: (M + dt*K) * T^(n+1) = M * T^n
      system_matrix.copy_from(mass_matrix);
      system_matrix.add(dt, stiffness_matrix);
      mass_matrix.vmult(system_rhs, old_solution);

      // BCs: prescribed temperature on left (boundary_id=0) and right (boundary_id=1)
      std::map<types::global_dof_index, double> boundary_values;
      VectorTools::interpolate_boundary_values(dof_handler, 0,
        Functions::ConstantFunction<2>(1.0), boundary_values);
      VectorTools::interpolate_boundary_values(dof_handler, 1,
        Functions::ZeroFunction<2>(), boundary_values);
      MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

      SolverControl solver_control(1000, 1e-12);
      SolverCG<Vector<double>> solver(solver_control);
      solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
    }}

  std::cout << "Heat at t=" << n_steps * dt << ": min(T)="
            << *std::min_element(solution.begin(), solution.end())
            << ", max(T)=" << *std::max_element(solution.begin(), solution.end()) << std::endl;

  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "temperature");
  data_out.build_patches();
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);
  std::cout << "Output written to result.vtu" << std::endl;
  return 0;
}}
'''


def _heat_2d_steady(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    degree = int(params.get("degree", params.get("order", 1)))
    if degree < 1:
        raise ValueError(
            f"_heat_2d_steady: degree must be >= 1, got {degree!r}")
    refinements = params.get("refinements", 5)
    T_left = params.get("T_left", 100.0)
    T_right = params.get("T_right", 0.0)
    return f'''\
/* Steady heat conduction on unit square — deal.II
 * -laplacian(T) = 0, T={T_left} on left, T={T_right} on right
 * Insulated top/bottom (natural Neumann BC).
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
#include <deal.II/base/function.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iostream>

using namespace dealii;

int main()
{{
  Triangulation<2> triangulation;
  GridGenerator::hyper_rectangle(triangulation, Point<2>(0,0), Point<2>(1,1), true);
  triangulation.refine_global({refinements});

  FE_Q<2> fe({degree});
  DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  std::cout << "Heat DOFs: " << dof_handler.n_dofs() << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> system_matrix;
  system_matrix.reinit(sparsity_pattern);
  Vector<double> solution(dof_handler.n_dofs());
  Vector<double> system_rhs(dof_handler.n_dofs());

  // Assemble Laplace operator
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
          }}
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {{
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }}
    }}

  // BCs: T={T_left} on left (boundary_id=0), T={T_right} on right (boundary_id=1)
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler, 0,
    Functions::ConstantFunction<2>({T_left}), boundary_values);
  VectorTools::interpolate_boundary_values(dof_handler, 1,
    Functions::ConstantFunction<2>({T_right}), boundary_values);
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

  SolverControl solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  std::cout << "Heat steady: min(T)="
            << *std::min_element(solution.begin(), solution.end())
            << ", max(T)=" << *std::max_element(solution.begin(), solution.end()) << std::endl;

  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "temperature");
  data_out.build_patches();
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);
  std::cout << "Output written to result.vtu" << std::endl;
  return 0;
}}
'''


def _heat_rectangle(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    degree = int(params.get("degree", params.get("order", 1)))
    if degree < 1:
        raise ValueError(
            f"_heat_rectangle: degree must be >= 1, got {degree!r}")
    refinements = params.get("refinements", 3)
    lx = params.get("lx", 2.0)
    ly = params.get("ly", 1.0)
    T_left = params.get("T_left", 100.0)
    T_right = params.get("T_right", 0.0)
    nx = int(lx * 4)
    ny = int(ly * 4)
    return f'''\
/* Heat conduction on [{lx}x{ly}] rectangle — deal.II
 * T={T_left} on left, T={T_right} on right
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
#include <deal.II/base/function.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iostream>

using namespace dealii;

int main()
{{
  Triangulation<2> triangulation;
  GridGenerator::subdivided_hyper_rectangle(triangulation,
    {{{nx}u, {ny}u}}, Point<2>(0, 0), Point<2>({lx}, {ly}), true /*colorize*/);
  triangulation.refine_global({refinements});

  FE_Q<2> fe({degree});
  DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  std::cout << "Heat DOFs: " << dof_handler.n_dofs() << std::endl;

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
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                 fe_values.shape_grad(j, q) *
                                 fe_values.JxW(q);
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
    Functions::ConstantFunction<2>({T_left}), boundary_values);
  VectorTools::interpolate_boundary_values(dof_handler, 1,
    Functions::ConstantFunction<2>({T_right}), boundary_values);
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

  SolverControl solver_control(5000, 1e-10);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  std::cout << "Heat: min(T)=" << *std::min_element(solution.begin(), solution.end())
            << ", max(T)=" << *std::max_element(solution.begin(), solution.end()) << std::endl;

  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "temperature");
  data_out.build_patches();
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);
  return 0;
}}
'''


# ── Knowledge ────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "description": "Heat equation: transient (step-26) with AMR, steady-state, time-stepping",
    "tutorial_steps": ["step-26 (transient + AMR)", "step-86 (PETSc TS via PETScWrappers::TimeStepper)"],
    "function_space": "FE_Q<dim>(1)",
    "solver": "CG + SSOR for each time step. SUNDIALS for adaptive time stepping",
    "time_stepping": "Theta method (0=forward Euler, 0.5=Crank-Nicolson, 1=backward Euler)",
    "elements": {
        "FE_Q":
            "Default for transient heat; degree=1 is the standard "
            "choice and balances accuracy with mass-matrix cost.",
        "FE_Q_Hierarchical":
            "For p-adaptive refinement during transient runs with "
            "smooth coefficient transitions.",
        "FE_DGQ":
            "DG variant; useful for sharp-front problems like "
            "welding or phase-change where the solution gradient "
            "is large across element interfaces.",
        "FE_SimplexP":
            "Use when the mesh comes from unstructured Gmsh / "
            "Triangle / TetGen (deal.II ≥ 9.3).",
    },
    "mesh_generators": {
        "hyper_cube": "Default heat-equation test domain.",
        "hyper_rectangle": "Thin plates / non-square aspect.",
        "hyper_L": "L-shaped, tests AMR near re-entrant corner.",
        "cylinder": "Heat-pipe / cylindrical-bar conduction.",
        "hyper_shell": "Pipe-insulation, annular heat conduction.",
        "plate_with_a_hole": "Local thermal-stress concentration via temperature gradient.",
        "extrude_triangulation": "2D heat-eq mesh extruded into 3D layered slab.",
    },
    "solvers": [
        "SolverCG<>                   — Heat-equation stiffness K and mass matrix M are SPD; CG works for all theta in (0,1]",
        "SolverGMRES<>                — needed only when the time-step matrix becomes non-symmetric (rare; happens with full-coupled nonlinear source terms)",
        "SUNDIALS::ARKode             — adaptive additive Runge-Kutta time integrator (no tutorial uses it; step-86 uses PETScWrappers::TimeStepper). Use when wall-time / step-count matters and dt is hard to estimate a priori",
    ],
    "preconditioners": [
        "PreconditionSSOR             — works on (M + dt*theta*K) at each step for moderate dt",
        "PreconditionAMG / BoomerAMG  — needed for stiff systems (small dt with large K eigenvalues, or wide dynamic range in thermal diffusivity)",
        "Use a SINGLE preconditioner factorisation across multiple time steps when dt and the mesh are fixed — re-building per step is the most common performance pitfall in transient heat code",
    ],
    "pitfalls": [
        "[Syntax] Time-step system is (M + dt*theta*K). Assemble M "
        "and K once at startup, build the LHS each step. Re-assembling "
        "M and K every time step is the most common transient-heat "
        "performance bug. Signal: profiler / TimerOutput.print_summary() "
        "shows 'assemble_system' dominating wall time at ~60-80% per "
        "step, scaling as O(ndof^2) instead of the expected "
        "O(ndof * log(ndof)) when AMG is used; SolverCG iteration "
        "count is normal.",
        "[Physics] RHS at each step is "
        "M*u_old - dt*(1-theta)*K*u_old + dt*theta*f_new + "
        "dt*(1-theta)*f_old. Dropping the (1-theta)*K*u_old term does "
        "NOT give you backward Euler, which is what this entry used to "
        "claim — it gives an INCONSISTENT scheme that converges to the "
        "WRONG ANSWER as dt is refined. Verified on u_t = Lap(u) on "
        "the unit square with u=0 on the boundary and "
        "u0 = sin(pi x) sin(pi y), whose exact amplitude decays as "
        "exp(-2 pi^2 t), running three variants that differ ONLY in "
        "this term over three successively halved time steps: correct "
        "Crank-Nicolson settled on the exact amplitude; backward Euler "
        "(theta=1) approached the same value from above as dt shrank; "
        "the theta=0.5 run with the term dropped settled on a value too "
        "LARGE and stayed there — its distance from the backward-Euler "
        "answer did not shrink as dt was halved, and in our run it GREW "
        "slightly. That indifference to dt is the signature; the "
        "measured NORMS are not reproduced here because you should see "
        "that indifference in your own ladder. The percentages below "
        "are a different thing and are NOT withheld: they are the "
        "derived formula evaluated, so you can recompute any of them "
        "for your own end time in one line. HOW MUCH too large is not a fixed "
        "figure and must not "
        "be quoted as one: dropping the term advances the diffusion by "
        "only dt/2 per step of dt, so the scheme converges to "
        "exp(-pi^2*T) instead of exp(-2*pi^2*T) and the relative excess "
        "is exp(pi^2*T) - 1 — it depends entirely on the END TIME. "
        "Measured 22 per cent at T = 0.02; the same formula gives 64 "
        "per cent at T = 0.05 and 168 per cent at T = 0.1, and it grows "
        "without bound. Quote the end time or quote nothing. "
        "A scheme whose answer stops depending on dt but is wrong is "
        "inconsistent, not merely low-order. "
        "Signal: run the SAME problem at dt and dt/2 and compare the "
        "two answers to each other AND to a reference. A consistent "
        "scheme's answers converge toward one another and toward the "
        "reference; this bug's converge toward each other but NOT "
        "toward the reference. Do not try to read a slope off a "
        "log-log error plot — the error does not decrease, so there is "
        "no slope.",
        "[API] For AMR in time: interpolate the solution to the new "
        "mesh via SolutionTransfer between refine_grid() and "
        "distribute_dofs(). Skipping SolutionTransfer gives a "
        "zero solution on the new mesh and the next time step "
        "evolves from zero — looks like a sudden cool-down. "
        "Signal: `solution.l2_norm()` from DataOut drops to near "
        "zero across the AMR-step boundary, then slowly recovers; "
        "the discontinuity coincides exactly with the time-step "
        "at which Triangulation::execute_coarsening_and_refinement() "
        "was called.",
        "[Numerical] Non-zero initial conditions must be set via "
        "VectorTools::interpolate (for piecewise polynomial ICs) or "
        "VectorTools::project (for general functions). Setting node "
        "values directly bypasses the AffineConstraints and gives "
        "an IC inconsistent with the boundary conditions. Signal: "
        "DataOut shows a discontinuous jump of O(1) at Dirichlet-"
        "boundary nodes between the t=0 output frame and the first "
        "time-step frame; the discontinuity disappears on the "
        "second step as the implicit solver smooths it out.",
        "[Numerical] theta=0 (forward Euler) is only CONDITIONALLY "
        "stable: the step size is bounded by roughly h^2 over the "
        "diffusivity, so a fine mesh or a small diffusivity makes it "
        "unusable. theta=0.5 (Crank-Nicolson, second order) and "
        "theta=1 (backward Euler, first order) are unconditionally "
        "stable and are what a production run should use. "
        "TREAT THE CONSTANT AS UNKNOWN. The textbook dt < h^2/(2*alpha) "
        "is derived for a FINITE-DIFFERENCE Laplacian; the "
        "finite-element bound is dt < 2 / lambda_max(M^{-1} K), which "
        "depends on the element, the quadrature rule and on whether "
        "the mass matrix is lumped, and can differ from the "
        "finite-difference value substantially. This catalog does NOT "
        "claim to have measured the constant — an attempt to do so "
        "produced an unsound probe and the result was discarded rather "
        "than reported. "
        "Signal: measure it for YOUR discretisation. Integrate a fixed "
        "NUMBER OF STEPS — not a fixed end time, otherwise a larger dt "
        "silently takes fewer steps and hides the instability — and "
        "bisect on dt: solution.linfty_norm() grows without bound "
        "above the limit and stays bounded below it. Nothing is "
        "announced; the only diagnostic is the norm you print "
        "yourself.",
    ],
}
