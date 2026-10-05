"""Wave equation templates for deal.II.

Based on deal.II tutorial step-23, which uses a theta scheme; the template's
Newmark average acceleration (beta=1/4, gamma=1/2) equals it at theta=1/2.
"""


def _wave_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable script. All parameter defaults are placeholders. The user/agent must set values appropriate to the specific problem being solved."""
    refinements = params.get("refinements", 5)
    n_steps = params.get("n_steps", 200)
    dt = params.get("dt", 0.005)
    wave_speed = params.get("wave_speed", 1.0)
    return f'''\
/* Wave equation on unit square — Newmark-beta method — deal.II (step-23 inspired)
 * d²u/dt² = c² * laplacian(u) + f
 * Newmark time integration (average acceleration, beta=0.25, gamma=0.5).
 * u = 0 on boundary, initial Gaussian pulse.
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
#include <cmath>

using namespace dealii;

// Initial displacement: Gaussian pulse centered in domain
template <int dim>
class InitialDisplacement : public Function<dim>
{{
public:
  double value(const Point<dim> &p, const unsigned int) const override
  {{
    // norm_square(), not square(): Tensor<1,dim> has no member
    // named square — Point's difference decays to a Tensor.
    const double r2 = (p - Point<dim>(0.5, 0.5)).norm_square();
    return std::exp(-100.0 * r2);
  }}
}};

int main()
{{
  const int dim = 2;

  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global({refinements});

  FE_Q<dim> fe(1);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  const unsigned int n_dofs = dof_handler.n_dofs();
  std::cout << "Wave DOFs: " << n_dofs << std::endl;

  DynamicSparsityPattern dsp(n_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> mass_matrix, stiffness_matrix, system_matrix;
  mass_matrix.reinit(sparsity_pattern);
  stiffness_matrix.reinit(sparsity_pattern);
  system_matrix.reinit(sparsity_pattern);

  // Assemble mass and stiffness
  MatrixTools::create_mass_matrix(dof_handler, QGauss<dim>(fe.degree + 1), mass_matrix);
  MatrixTools::create_laplace_matrix(dof_handler, QGauss<dim>(fe.degree + 1), stiffness_matrix);

  // Newmark parameters (average acceleration, unconditionally stable)
  const double beta  = 0.25;
  const double gamma_nm = 0.5;
  const double dt    = {dt};
  const double c2    = {wave_speed} * {wave_speed};
  const int n_steps  = {n_steps};

  // State vectors
  Vector<double> u_n(n_dofs);      // displacement at t_n
  Vector<double> v_n(n_dofs);      // velocity at t_n
  Vector<double> a_n(n_dofs);      // acceleration at t_n
  Vector<double> u_np1(n_dofs);    // displacement at t_(n+1)
  Vector<double> rhs(n_dofs);
  Vector<double> tmp(n_dofs);

  // Initial conditions: Gaussian displacement, zero velocity
  VectorTools::interpolate(dof_handler, InitialDisplacement<dim>(), u_n);
  v_n = 0;

  // Initial acceleration: M*a_0 = -c^2*K*u_0
  stiffness_matrix.vmult(rhs, u_n);
  rhs *= -c2;
  // Apply homogeneous BCs to rhs
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler, 0,
    Functions::ZeroFunction<dim>(), boundary_values);

  // Solve M*a_0 = rhs for initial acceleration
  SolverControl sc_init(1000, 1e-12);
  SolverCG<Vector<double>> cg_init(sc_init);
  cg_init.solve(mass_matrix, a_n, rhs, PreconditionIdentity());

  // Effective system matrix: M + beta*dt^2*c^2*K
  system_matrix.copy_from(mass_matrix);
  system_matrix.add(beta * dt * dt * c2, stiffness_matrix);

  // Apply boundary conditions to system matrix
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, u_np1, rhs);

  // Preconditioner for the effective system
  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  // Write initial state
  {{
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(u_n, "displacement");
    data_out.build_patches();
    std::ofstream output("wave_0000.vtu");
    data_out.write_vtu(output);
  }}

  // Time stepping
  for (int step = 0; step < n_steps; ++step)
    {{
      // Predictor: u_tilde = u_n + dt*v_n + (0.5-beta)*dt^2*a_n
      u_np1 = u_n;
      u_np1.add(dt, v_n);
      u_np1.add((0.5 - beta) * dt * dt, a_n);

      // RHS: M*u_tilde - beta*dt^2*c^2*K*u_tilde (no external force)
      // Actually: RHS = -c^2*K*u_tilde + M*(u_tilde/(beta*dt^2)) ... simplified:
      // We solve: (M + beta*dt^2*c^2*K) * u_(n+1) = M * u_tilde
      mass_matrix.vmult(rhs, u_np1);

      // Apply BCs
      for (const auto &[dof, val] : boundary_values)
        {{
          rhs(dof) = 0.0;
          u_np1(dof) = 0.0;
        }}

      // Solve
      SolverControl solver_control(2000, 1e-10);
      SolverCG<Vector<double>> solver(solver_control);
      solver.solve(system_matrix, u_np1, rhs, preconditioner);

      // Corrector: compute new acceleration and velocity
      // a_(n+1) = (u_(n+1) - u_tilde) / (beta * dt^2)
      Vector<double> a_np1(n_dofs);
      a_np1 = u_np1;
      a_np1 -= u_n;
      a_np1.add(-dt, v_n);
      a_np1.add(-(0.5 - beta) * dt * dt, a_n);
      a_np1 /= (beta * dt * dt);

      // v_(n+1) = v_n + dt*((1-gamma)*a_n + gamma*a_(n+1))
      v_n.add(dt * (1.0 - gamma_nm), a_n);
      v_n.add(dt * gamma_nm, a_np1);

      // Update
      u_n = u_np1;
      a_n = a_np1;

      // Output every 10th step
      if (step % std::max(1, n_steps / 20) == 0 || step == n_steps - 1)
        {{
          DataOut<dim> data_out;
          data_out.attach_dof_handler(dof_handler);
          data_out.add_data_vector(u_n, "displacement");
          data_out.build_patches();
          char filename[64];
          std::snprintf(filename, sizeof(filename), "wave_%04d.vtu", step + 1);
          std::ofstream output(filename);
          data_out.write_vtu(output);
          std::cout << "Step " << step + 1 << "/" << n_steps
                    << ", t=" << (step + 1) * dt
                    << ", max|u|=" << u_n.linfty_norm() << std::endl;
        }}
    }}

  std::cout << "Wave equation: " << n_steps << " steps complete" << std::endl;
  return 0;
}}
'''


# ── Knowledge ────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "description": "Wave equation: d^2u/dt^2 = c^2*laplacian(u) (step-23 inspired)",
    "tutorial_steps": ["step-23 (wave equation, theta-method)",
                      "step-24 (thermoacoustic wave equation, absorbing boundary conditions, point-value extraction)",
                      "step-25 (nonlinear wave, sine-Gordon)"],
    "function_space": "FE_Q<dim>(1)",
    "time_stepping": "Newmark-beta (beta=0.25, gamma=0.5 for average acceleration, unconditionally stable)",
    "solver": "CG + SSOR for the effective stiffness system at each time step",
    "elements": {
        "FE_Q":
            "degree=1 default for the linear wave equation; "
            "degree=2 for spectral accuracy at moderate wave "
            "numbers.",
        "FE_Q_Hierarchical":
            "For hp-adaptive refinement around shocks (sine-"
            "Gordon, nonlinear wave) — coarse-DoF wavefront "
            "survives degree changes.",
        "FE_DGQ":
            "DG variant; preferred for discontinuous-coefficient "
            "wave problems and shock propagation.",
        "FE_Q_iso_Q1":
            "Lumped mass matrix becomes diagonal — explicit "
            "Newmark (beta=0) reduces to a diagonal solve per "
            "step, enabling matrix-free explicit time integration.",
    },
    "mesh_generators": {
        "hyper_cube": "Canonical wave-equation test domain.",
        "hyper_ball": "Circular drum; tests modal accuracy against Bessel-function reference.",
        "hyper_shell": "Annular waveguide.",
        "hyper_cube_with_cylindrical_hole": "Scattering from obstacle; tests absorbing-BC quality.",
        "hyper_L": "Re-entrant corner reflects high-frequency content; tests AMR.",
        "extrude_triangulation": "2D wave mesh → 3D acoustic-tank problems.",
    },
    "solvers": [
        "SolverCG<>                    — effective-stiffness system is SPD if M and K are SPD (typical for the scalar wave equation); the default",
        "SolverGMRES<>                 — needed when damping or absorbing BCs break symmetry of the effective stiffness",
        "Explicit Newmark (beta=0)     — diagonalised, no linear solve; bound by CFL dt < h/c",
        "SparseDirectUMFPACK           — for one-shot factorisation re-used at every time step; effective when dt and mesh are constant",
    ],
    "preconditioners": [
        "PreconditionSSOR              — works on the effective-stiffness (M + beta*dt^2*c^2*K) for typical Newmark parameters",
        "PreconditionAMG / BoomerAMG   — at >10^5 DoFs; rebuild factorisation only when dt or mesh changes",
        "Diagonal scaling (PreconditionJacobi) — sufficient when the mass matrix dominates (small dt) and AMG overhead isn't worth it",
    ],
    "pitfalls": [
        "[Numerical] Newmark-beta unconditional stability requires "
        "beta >= gamma/2 >= 1/4. The (beta=0.25, gamma=0.5) pair "
        "is the canonical 'average acceleration' choice. Setting "
        "beta=0 (explicit) trades stability for CFL-bound dt < h/c. "
        "Signal: when beta < gamma/2, "
        "`solution.linfty_norm()` from DataOut grows exponentially "
        "(doubles every 5-10 time steps) and reaches NaN within "
        "~50 steps; the same problem with beta=0.25 stays bounded "
        "at O(1) amplitude.",
        "[Syntax] Time-step system: "
        "(M + beta*dt^2*c^2*K)*u_(n+1) = M*u_tilde. Both M and K "
        "must be assembled ONCE before time stepping; only the "
        "RHS (u_tilde from u_n, v_n, a_n) varies per step. "
        "Re-assembling K each step is the most common transient-"
        "wave performance bug. Signal: TimerOutput.print_summary() "
        "shows 'assemble_system' dominating wall time at 60-80% "
        "per step, scaling as O(ndof^2) instead of "
        "O(ndof * SolverCG::last_step()); SolverCG iteration "
        "count itself is normal.",
        "[Numerical] CFL-like condition for explicit Newmark "
        "(beta=0): dt < h/c. Setting dt larger gives spurious "
        "exponentially-growing modes. Signal: with beta=0 and "
        "dt > h/c, BlockVector::linfty_norm() of the solution "
        "doubles every ~5 steps and reaches NaN within ~30 steps; "
        "energy 0.5 * v^T M v + 0.5 c^2 u^T K u (computed via "
        "SparseMatrix::vmult + dot product) grows by factor 10+ "
        "per step instead of staying bounded; switching to "
        "implicit Newmark (beta=0.25) at the same dt keeps "
        "amplitude bounded at O(1).",
        "[Syntax] Initial acceleration must be computed by solving "
        "M*a_0 = f_0 - c^2*K*u_0 (via SolverCG), not just zeroed "
        "out. Setting a_0 = 0 produces a slowly-decaying transient "
        "from the wrong initial condition. Signal: DataOut at "
        "early times (t < 5*period) shows a low-frequency mode "
        "of magnitude 0.1-0.3 superimposed on the correct wave "
        "pattern; VectorTools::integrate_difference vs analytic "
        "reference shows O(1) error in the first 10 steps then "
        "decays to O(h^p) on later steps.",
        "[Physics] Absorbing BCs: add a damping term on the "
        "boundary, c * du/dt on the outflow face. Forgetting this "
        "makes the boundary reflect waves back into the domain. "
        "Signal: DataOut shows the outgoing wave reflecting at "
        "the outflow boundary; "
        "`solution.linfty_norm()` rises periodically (period = "
        "2*L/c) rather than decaying monotonically; energy "
        "`0.5 * v^T M v + 0.5 c^2 u^T K u` stays constant or "
        "grows instead of decaying to zero as the wave should "
        "leave the domain.",
        "[Integration] VTU output at every time step is "
        "unnecessarily expensive — output every N steps and use "
        "DataOutInterface::write_pvd_record to assemble the .pvd "
        "time-series file. Signal: wrap the output block in a "
        "TimerOutput::Scope and read print_summary(). Note the "
        "section LABEL is whatever string you pass to that scope "
        "(e.g. \"output_results\") — it is not a deal.II message, so "
        "there is nothing to grep for unless you create it. The "
        "symptom is your output section standing as the "
        "top wall-time entry (>50% of total), exceeding the "
        "SolverCG and assemble_system buckets; the per-step VTU "
        "file count grows into the 10000s on disk.",
    ],
    "materials": {
        "wave_speed": {"range": [0.01, 10000.0], "unit": "m/s"},
        "dt": {"range": [1e-6, 1.0], "unit": "s (time step)"},
    },
}
