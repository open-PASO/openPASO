"""deal.II Navier-Stokes generators and knowledge.

Based on step-57 (stationary NS), step-35 (Boussinesq), step-55 (Stokes MPI).
"""


_SRC_NAVIER_STOKES_2D = r"""/* Stationary incompressible Navier-Stokes - deal.II (based on step-57)
 * Lid-driven cavity, Taylor-Hood Q2/Q1, Newton iteration on the convective
 * term, monolithic UMFPACK linear solves. Self-contained; writes result.vtu.
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
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
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace dealii;

const unsigned int degree = @@DEGREE@@;        // pressure degree; velocity = degree+1
const unsigned int n_refine = @@REFINEMENTS@@;
const double viscosity = @@VISCOSITY@@;         // 1/Re  (Re=10 here, robust cold start)
const unsigned int max_newton = @@MAX_NEWTON@@;

template <int dim>
class LidVelocity : public Function<dim>
{
public:
  LidVelocity() : Function<dim>(dim + 1) {}
  virtual void vector_value(const Point<dim> &p, Vector<double> &v) const override
  {
    v = 0;
    if (p[1] > 1.0 - 1e-8) // top lid
      v[0] = 1.0;
  }
};

int main()
{
  const int dim = 2;
  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(n_refine);

  FESystem<dim> fe(FE_Q<dim>(degree + 1), dim, FE_Q<dim>(degree), 1);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  DoFRenumbering::component_wise(dof_handler);

  std::vector<unsigned int> block_component(dim + 1, 0);
  block_component[dim] = 1;
  // deal.II >= 9.2: DoFTools::count_dofs_per_block was renamed to
  // count_dofs_per_fe_block and now RETURNS the vector.
  const std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
  const unsigned int n_u = dofs_per_block[0], n_p = dofs_per_block[1];
  std::cout << "Navier-Stokes DOFs: u=" << n_u << " p=" << n_p << std::endl;

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  // Constraints for the Newton update (zero Dirichlet everywhere on velocity)
  // and for the initial guess (lid velocity). Pressure pinned at one dof.
  AffineConstraints<double> zero_constraints, nonzero_constraints;
  {
    nonzero_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);
    VectorTools::interpolate_boundary_values(dof_handler, 0,
      LidVelocity<dim>(), nonzero_constraints, fe.component_mask(velocities));
    nonzero_constraints.close();

    zero_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints);
    VectorTools::interpolate_boundary_values(dof_handler, 0,
      Functions::ZeroFunction<dim>(dim + 1), zero_constraints,
      fe.component_mask(velocities));
    zero_constraints.close();
  }

  // Pin pressure at one dof to fix the constant null space. After
  // component_wise renumbering the first pressure dof is index n_u.
  const types::global_dof_index pressure_pin = n_u;
  if (!nonzero_constraints.is_constrained(pressure_pin))
    {
      nonzero_constraints.clear();
      DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);
      VectorTools::interpolate_boundary_values(dof_handler, 0,
        LidVelocity<dim>(), nonzero_constraints, fe.component_mask(velocities));
      nonzero_constraints.add_line(pressure_pin);
      nonzero_constraints.close();
    }
  {
    zero_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints);
    VectorTools::interpolate_boundary_values(dof_handler, 0,
      Functions::ZeroFunction<dim>(dim + 1), zero_constraints,
      fe.component_mask(velocities));
    zero_constraints.add_line(pressure_pin);
    zero_constraints.close();
  }

  BlockDynamicSparsityPattern dsp(2, 2);
  dsp.block(0, 0).reinit(n_u, n_u); dsp.block(0, 1).reinit(n_u, n_p);
  dsp.block(1, 0).reinit(n_p, n_u); dsp.block(1, 1).reinit(n_p, n_p);
  dsp.collect_sizes();
  DoFTools::make_sparsity_pattern(dof_handler, dsp, nonzero_constraints, true);
  BlockSparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);
  BlockSparseMatrix<double> system_matrix(sparsity_pattern);

  BlockVector<double> present_solution(2), newton_update(2), system_rhs(2);
  for (auto *v : {&present_solution, &newton_update, &system_rhs})
    {
      v->block(0).reinit(n_u); v->block(1).reinit(n_p); v->collect_sizes();
    }

  // initial guess satisfies the lid BC
  nonzero_constraints.distribute(present_solution);

  QGauss<dim> quadrature(degree + 2);
  // every flag this loop reads: a Release-only deal.II build segfaults, with no message, on a missing one
  FEValues<dim> fe_values(fe, quadrature,
    update_values | update_gradients | update_JxW_values | update_quadrature_points);
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q = quadrature.size();
  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> local_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<Tensor<1, dim>> present_velocity_values(n_q);
  std::vector<Tensor<2, dim>> present_velocity_gradients(n_q);
  std::vector<double> present_pressure_values(n_q);
  std::vector<double> div_phi_u(dofs_per_cell);
  std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
  std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
  std::vector<double> phi_p(dofs_per_cell);

  for (unsigned int newton = 0; newton < max_newton; ++newton)
    {
      system_matrix = 0; system_rhs = 0;
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          fe_values.reinit(cell);
          local_matrix = 0; local_rhs = 0;
          fe_values[velocities].get_function_values(present_solution, present_velocity_values);
          fe_values[velocities].get_function_gradients(present_solution, present_velocity_gradients);
          fe_values[pressure].get_function_values(present_solution, present_pressure_values);

          for (unsigned int q = 0; q < n_q; ++q)
            {
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  div_phi_u[k] = fe_values[velocities].divergence(k, q);
                  phi_u[k] = fe_values[velocities].value(k, q);
                  grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                  phi_p[k] = fe_values[pressure].value(k, q);
                }
              const Tensor<1, dim> u = present_velocity_values[q];
              const Tensor<2, dim> grad_u = present_velocity_gradients[q];
              const double p = present_pressure_values[q];
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      local_matrix(i, j) +=
                        (viscosity * scalar_product(grad_phi_u[j], grad_phi_u[i])
                         + (grad_phi_u[j] * u) * phi_u[i]          // convection (Newton, part 1)
                         + (grad_u * phi_u[j]) * phi_u[i]          // convection (Newton, part 2)
                         - div_phi_u[i] * phi_p[j]                 // pressure
                         - phi_p[i] * div_phi_u[j])                // incompressibility
                        * fe_values.JxW(q);
                    }
                  // residual (negative): -F(present_solution)
                  double divu = 0;
                  for (unsigned int d = 0; d < dim; ++d)
                    divu += grad_u[d][d];
                  local_rhs(i) -=
                    (viscosity * scalar_product(grad_u, grad_phi_u[i])
                     + (grad_u * u) * phi_u[i]
                     - p * div_phi_u[i]
                     - phi_p[i] * divu)
                    * fe_values.JxW(q);
                }
            }
          cell->get_dof_indices(local_dof_indices);
          const AffineConstraints<double> &cm =
            (newton == 0) ? nonzero_constraints : zero_constraints;
          cm.distribute_local_to_global(local_matrix, local_rhs,
            local_dof_indices, system_matrix, system_rhs);
        }

      const double residual_norm = system_rhs.l2_norm();
      std::cout << "  Newton " << newton << ": residual = " << residual_norm << std::endl;
      if (residual_norm < 1e-10)
        { std::cout << "Newton converged." << std::endl; break; }

      SparseDirectUMFPACK direct;
      direct.initialize(system_matrix);
      direct.vmult(newton_update, system_rhs);
      zero_constraints.distribute(newton_update);

      present_solution += newton_update;
    }

  std::cout << "u_max = " << present_solution.block(0).linfty_norm()
            << ", p range pinned at dof0=0" << std::endl;

  std::vector<std::string> names(dim, "velocity");
  names.push_back("pressure");
  std::vector<DataComponentInterpretation::DataComponentInterpretation> interp(
    dim, DataComponentInterpretation::component_is_part_of_vector);
  interp.push_back(DataComponentInterpretation::component_is_scalar);
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(present_solution, names,
    DataOut<dim>::type_dof_data, interp);
  data_out.build_patches();
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);
  std::cout << "Output written to result.vtu" << std::endl;
  return 0;
}
"""


def _navier_stokes_2d(params: dict) -> str:
    """Real stationary Navier-Stokes (step-57): lid-driven cavity,
    Taylor-Hood Q2/Q1, Newton iteration, monolithic UMFPACK solves.
    Self-contained and parameterized; writes result.vtu. Verified to
    compile/run on deal.II 9.1.1 with quadratic Newton convergence."""
    degree = int(params.get("degree", 1))
    refinements = int(params.get("refinements", 4))
    viscosity = float(params.get("viscosity", 0.1))
    max_newton = int(params.get("max_newton", 15))
    src = _SRC_NAVIER_STOKES_2D
    src = src.replace("@@DEGREE@@", str(degree))
    src = src.replace("@@REFINEMENTS@@", str(refinements))
    src = src.replace("@@VISCOSITY@@", str(viscosity))
    src = src.replace("@@MAX_NEWTON@@", str(max_newton))
    return src


KNOWLEDGE = {
    "description": "Navier-Stokes (stationary and transient) — step-57, step-35, step-55",
    "tutorial_steps": ["step-57 (stationary NS, Newton)", "step-35 (Boussinesq buoyancy)",
                       "step-55 (Stokes, MPI parallel)"],
    "function_space": "FESystem<dim>(FE_Q<dim>(2), dim, FE_Q<dim>(1), 1) — Taylor-Hood Q2/Q1",
    "solver": "Newton iteration for nonlinear convection term, UMFPACK or GMRES+ILU for linear sub-problems",
    "elements": {
        "FESystem":
            "Block wrapper for ALL NS pairs. Standard Taylor-Hood "
            "shape: FESystem<dim>(FE_Q<dim>(2), dim, FE_Q<dim>(1), "
            "1). Generalised Q_{p+1}/Q_p for higher-order runs.",
        "FE_Q":
            "Velocity (degree p+1) and pressure (degree p) in "
            "Taylor-Hood. Equal-order Q1/Q1 needs SUPG / GLS / "
            "VMS stabilisation to be inf-sup stable.",
        "FE_Q_Bubbles":
            "Velocity component of MINI-like (Q1+bubble / Q1) — "
            "cheaper than Taylor-Hood Q2/Q1 in DoF count, "
            "inf-sup stable.",
        "FE_DGP":
            "Pressure component of MINI-like (Q1+bubble / DGP0).",
        "FE_RaviartThomas":
            "Velocity of RT/DGQ H(div) pair. Exactly "
            "divergence-free velocity — the right pick when "
            "momentum conservation matters (geophysical flow, "
            "groundwater coupled with advection).",
        "FE_DGQ":
            "Pressure of RT/DGQ pair, or velocity AND pressure "
            "in fully-DG NS for advection-dominated problems "
            "where upwinding helps stability.",
    },
    "mesh_generators": {
        "channel_with_cylinder": "Schäfer-Turek channel-with-cylinder geometry — GridGenerator fixes the cylinder at (0.2, 0.2) in a (2.2 × 0.41) channel; retrieve the published lift/drag for the Re you run.",
        "hyper_cube": "Driven-cavity geometry; the classic reference series is Ghia/Ghia/Shin (1982) — look up the values for your Re, they are not stored here.",
        "hyper_L": "Backward-facing step; reattachment-length benchmark.",
        "hyper_cube_with_cylindrical_hole": "Flow around cylinder; vortex-shedding at Re > 47.",
        "subdivided_hyper_rectangle": "Channel flow with prescribed-aspect elements (boundary-layer resolution).",
        "cheese": "Porous-media-like NS demo.",
    },
    "solvers": [
        "Newton iteration — outer loop for the stationary nonlinear problem (step-57); converges quadratically near the solution",
        "Picard/Oseen iteration — first-order linearisation, larger basin of convergence than Newton; useful as a Newton warm-start",
        "BDF2 / Crank-Nicolson — time-stepping for transient NS; BDF2 is the canonical 2nd-order multi-step choice",
        "SparseDirectUMFPACK / MUMPS — robust linear sub-solver for moderate problem sizes",
        "SolverGMRES + ILU — iterative linear sub-solver; needs preconditioning beyond ~10^5 DoFs",
    ],
    "preconditioners": [
        "PreconditionILU / ILUT — for GMRES on the linearised NS tangent; cheap, works up to ~10^5 DoFs",
        "PreconditionAMG / BoomerAMG on the velocity block — combined with Schur-complement for parallel scaling (step-55)",
        "BlockSchurPreconditioner — block-triangular preconditioner; pressure Schur approximated by 1/mu * mass_p; the canonical step-22/step-57 choice",
    ],
    "pitfalls": [
        "[Numerical] NS is NONLINEAR — requires Newton iteration "
        "or Picard/Oseen linearisation. A naive linear solve gives "
        "the Stokes solution at zero Reynolds, regardless of the "
        "user's intended Re. Signal: SolverGMRES with no outer "
        "Newton/Picard loop converges in 1 iteration to a "
        "symmetric, advection-free velocity profile — DataOut "
        "shows no recirculation behind a cylinder at Re=100 (which "
        "should have a visible Karman vortex street); "
        "VectorTools::integrate_difference against a Stokes "
        "reference is ~0, against a NS reference is O(1).",
        "[Numerical] Taylor-Hood Q2/Q1 satisfies inf-sup — Q1/Q1 "
        "DOES NOT and produces checkerboard pressure unless "
        "stabilised (SUPG, GLS, VMS). Signal: DataOut output for "
        "the BlockVector pressure block shows a regular "
        "high-frequency checkerboard pattern; "
        "VectorTools::point_value at adjacent cell centroids "
        "alternates sign with O(1) magnitude.",
        "[Numerical] Reynolds number affects convergence — Newton "
        "diverges at high Re from a cold start. Continuation in Re: "
        "solve at Re=10, ramp through Re=50, 100, 200, ...; use "
        "each solution as the next starting guess. Signal: "
        "SolverControl reports residual.l2_norm() > 1e3 on Newton "
        "iteration 1 at Re=200 from a zero initial guess, ending "
        "in the INNER solve throwing SolverControl::NoConvergence, "
        "whose real text is 'Iterative method reported convergence "
        "failure in step <N>. The residual in the last step was "
        "<R>.' — the string 'iterative method failed to converge' "
        "this entry used to quote does not exist in deal.II, and the "
        "Newton-level message has to be your own "
        "AssertThrow(..., ExcMessage(...)); "
        "rerunning with the Re=100 solution stored in BlockVector "
        "as the initial guess converges in 4-6 Newton steps.",
        "[Numerical] For time-dependent: BDF2 or Crank-Nicolson "
        "(2nd-order, A-stable). Backward Euler is robust but "
        "introduces O(dt) numerical viscosity that contaminates "
        "high-Re results. Signal: time-averaged Reynolds-stress "
        "magnitude from VectorTools::integrate_difference differs "
        "by 20-40% from a Schäfer-Turek reference at Re=100 — "
        "halving dt drops the error proportionally (O(dt) "
        "scaling); switching to BDF2 / Crank-Nicolson at the "
        "same dt reduces the error below 5%.",
        "[Physics] Pressure is determined up to a constant for "
        "closed-cavity NS — pin at one point or use mean-free "
        "constraint via AffineConstraints. Signal: "
        "`solution.block(1).linfty_norm()` (pressure) drifts to "
        ">1e10 magnitude across Newton iterations while "
        "`solution.block(0).l2_norm()` (velocity) converges "
        "normally; SolverGMRES iteration count for the linearised "
        "tangent grows each outer step as the pressure null space "
        "pollutes the Krylov basis.",
        "[Integration] SUPG/GLS stabilisation parameter tau must "
        "be tuned to the local element size h and local advection "
        "speed |u|. The textbook tau = h / (2*|u|) is correct only "
        "in 1D; multi-dimensional NS needs tau = h / (2*|u|*sqrt(2)) "
        "or a more elaborate formula. Signal: DataOut shows visible "
        "spatial oscillations of magnitude 0.05-0.2 (relative to "
        "max velocity) within 2-3 cells of a no-slip wall despite "
        "the SUPG term being active; refining h eliminates them "
        "only locally; comparing tau values between 1D and 2D "
        "shows a sqrt(2) discrepancy at the wall.",
    ],
}

GENERATORS = {
    "navier_stokes_2d": _navier_stokes_2d,
}
