"""Convection-diffusion templates for deal.II.

Based on deal.II tutorial step-9.
"""


def _convdiff_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    Based on deal.II step-9.
    """
    refinements = params.get("refinements", 5)
    eps = params.get("diffusion", 0.01)
    return f'''\
/* Convection-diffusion: SUPG stabilized — deal.II (step-9 inspired) */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <fstream>
#include <cmath>
using namespace dealii;

int main() {{
  const int dim = 2;
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0, 1);
  tria.refine_global({refinements});

  FE_Q<dim> fe(1);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sp;
  sp.copy_from(dsp);

  SparseMatrix<double> system_matrix;
  system_matrix.reinit(sp);
  Vector<double> solution(dof_handler.n_dofs());
  Vector<double> system_rhs(dof_handler.n_dofs());

  const double eps = {eps};
  // Element-wise init: Tensor's array constructor is explicit, so
  // brace-list construction fails to compile (deal.II 9.3.x).
  Tensor<1, dim> beta;
  beta[0] = 1.0;
  beta[1] = 0.5;

  QGauss<dim> quadrature(2);
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
    const double h = cell->diameter();
    const double Pe = beta.norm() * h / (2.0 * eps);
    const double tau = (Pe > 1.0) ? h / (2.0 * beta.norm()) * (1.0 - 1.0 / Pe) : 0.0;

    for (unsigned int q = 0; q < quadrature.size(); ++q) {{
      for (unsigned int i = 0; i < dpc; ++i) {{
        const double phi_i = fe_values.shape_value(i, q);
        const Tensor<1, dim> grad_phi_i = fe_values.shape_grad(i, q);
        const double supg_test = phi_i + tau * (beta * grad_phi_i);
        for (unsigned int j = 0; j < dpc; ++j) {{
          const Tensor<1, dim> grad_phi_j = fe_values.shape_grad(j, q);
          const double advection = beta * grad_phi_j;
          cell_matrix(i, j) += (eps * grad_phi_i * grad_phi_j + supg_test * advection)
                               * fe_values.JxW(q);
        }}
        cell_rhs(i) += 1.0 * supg_test * fe_values.JxW(q);
      }}
    }}
    cell->get_dof_indices(local_dof_indices);
    for (unsigned int i = 0; i < dpc; ++i) {{
      for (unsigned int j = 0; j < dpc; ++j)
        system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
      system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }}
  }}

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler, 0,
    Functions::ZeroFunction<dim>(), boundary_values);
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

  SolverControl sc(1000, 1e-10);
  SolverGMRES<Vector<double>> solver(sc);
  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);
  solver.solve(system_matrix, solution, system_rhs, preconditioner);

  std::cout << "ConvDiff: " << dof_handler.n_dofs() << " DOFs, "
            << sc.last_step() << " GMRES iters" << std::endl;
  std::cout << "max(u) = " << solution.linfty_norm() << std::endl;

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
  std::ofstream output("solution.vtu");
  data_out.write_vtu(output);
  return 0;
}}
'''


# ── Knowledge ────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "description": "Convection-diffusion: SUPG (step-9), DG (step-12), HDG (step-51)",
    "tutorial_steps": ["step-9 (SUPG streamline diffusion)", "step-12 (DG upwind)",
                      "step-30 (anisotropic refinement)", "step-51 (HDG)",
                      "step-63 (GMG with block smoothers)"],
    "function_space": "FE_Q(1) for SUPG, FE_DGQ(p) for DG",
    "solver": "BiCGStab + Jacobi for SUPG; direct for DG (block-diagonal)",
    "elements": {
        "FE_Q":
            "Used with SUPG / GLS / VMS stabilisation — bare "
            "Galerkin Q1 oscillates at high Peclet. degree=1 + "
            "tau = h/(2|b|)*(coth(Pe) - 1/Pe) is the SUPG "
            "standard.",
        "FE_DGQ":
            "DG with upwind flux (step-12) — handles "
            "discontinuous solutions and high Peclet naturally. "
            "Preferred over stabilised FE_Q when Pe > ~100.",
        "FE_FaceQ":
            "Trace component of hybridised DG (HDG, step-51) — "
            "combined with FE_DGQ on cells, trace unknowns live "
            "only on faces, cheaper than full DG.",
        "FE_DGP":
            "DG P_k space with a Legendre (L2-orthonormal) basis; "
            "alternative to FE_DGQ for higher-order accurate "
            "transport.",
        "FE_Q_Hierarchical":
            "For hp-adaptive refinement around shock layers; "
            "combine with anisotropic refinement (step-30) so "
            "h refinement is concentrated normal to the front.",
    },
    "mesh_generators": {
        "hyper_cube": "Canonical transport on the unit square; deal.II's step-9 tutorial is the worked example of this setup.",
        "subdivided_hyper_rectangle": "Anisotropic refinement for boundary-layer / shock-front resolution.",
        "hyper_rectangle": "Generic channel for inlet/outlet transport tests.",
        "hyper_L": "L-shaped; tests discontinuity propagation around corner.",
        "merge_triangulations": "Heterogeneous-coefficient demos (high-diff + low-diff patches).",
    },
    "solvers": [
        "SolverBicgstab<>             — non-symmetric system from convection term; best for SUPG",
        "SolverGMRES<>                — robust alternative; works when BiCGStab stagnates",
        "SparseDirectUMFPACK          — for DG up to ~10^4 cells; block-diagonal mass makes direct solves cheap",
    ],
    "preconditioners": [
        "PreconditionJacobi           — for BiCGStab on SUPG; cheap diagonal scaling",
        "PreconditionILU              — stronger preconditioning when Peclet is high",
        "MGSmootherRelaxation with block-Jacobi (step-63) — point smoothers fail on convection-dominated systems",
    ],
    "pitfalls": [
        "[Numerical] SUPG stabilisation parameter: "
        "tau = h/(2|b|) * (coth(Pe_h) - 1/Pe_h) with the CELL Peclet "
        "number Pe_h = |b|*h/(2*eps). The doubly-asymptotic factor is "
        "what makes it right at BOTH ends: it tends to 1 at large Pe_h "
        "(full upwinding) and to 0 at small Pe_h, so a smooth "
        "diffusion-dominated solution is left alone. The bare "
        "h/(2|b|) is O(1) at low Pe_h and over-stabilises. "
        "Verified on 1D advection-diffusion, b=1, 20 cells, inflow 0 "
        "and outflow 1, comparing plain Galerkin against SUPG with "
        "this tau at three diffusivities: at Pe_h far below 1 both "
        "were monotone and indistinguishable; at Pe_h = 0.5 both were "
        "still monotone; at Pe_h = 5 plain Galerkin UNDERSHOT to about "
        "-0.67 — a large negative excursion in a problem whose exact "
        "solution stays inside [0, 1] — while SUPG with this tau "
        "stayed monotone inside that range. Nothing was raised in the "
        "oscillating case, in either build. "
        "Signal: the exact solution of a pure inflow/outflow transport "
        "problem is bounded by its own boundary data, so compute "
        "min(solution) and max(solution) and compare with that range; "
        "any excursion outside it is a stabilisation failure. Print "
        "Pe_h = |b|*h/(2*eps) per cell as well — it tells you to "
        "expect the problem before you look at the answer, and the "
        "threshold is around Pe_h = 1, not some large number.",
        "[Syntax] DG formulations need FEInterfaceValues for "
        "jump/average operators on faces. Using FEValues alone "
        "gives the cell-interior gradient but not the jump term. "
        "Signal: DG solution with FE_DGQ in DataOut shows "
        "continuous-Galerkin-like behaviour at element interfaces "
        "— `solution.linfty_norm()` of jump-across-face values "
        "drops below 1e-8 (effectively zero) where physical jumps "
        "of O(1) are expected for an upwind discontinuous "
        "Galerkin scheme.",
        "[Physics] Upwind flux: use the value from the cell where "
        "b·n > 0 (upstream). Switching the upwind direction inverts "
        "transport — solutions advect BACKWARDS from the inflow. "
        "Signal: DataOut shows the prescribed inflow Dirichlet "
        "value (e.g. u=1 on x=0 with advection in +x direction) "
        "appearing as a localized hump at the DOWNSTREAM (x=L) "
        "boundary instead of being carried with the flow; "
        "VectorTools::point_value at the outflow returns the "
        "inflow value with reversed sign.",
        "[Numerical] At high Peclet, SUPG can still oscillate. "
        "Either refine h or switch to DG. Signal: DataOut output "
        "shows O(1) oscillations in the FE_Q SUPG solution at "
        "the interface between high and low coefficient regions; "
        "VectorTools::integrate_difference vs the discontinuous "
        "reference is O(1) and does not decrease as h refines.",
        "[Integration] Anisotropic refinement (step-30) is "
        "essential for boundary-layer resolution at high Pe — "
        "isotropic refinement wastes DoFs perpendicular to the "
        "flow. Signal: DataOut shows boundary-layer thickness "
        "smeared across 3-5 cells in the streamwise direction "
        "(where 1 should suffice) while resolved in 1 cell normal "
        "to the flow; Triangulation::n_active_cells() climbs into "
        "the 1e5+ range to achieve the same wall-normal resolution "
        "anisotropic refinement reaches at 1e4 cells.",
    ],
}
