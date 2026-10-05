"""Stokes flow templates for deal.II.

Based on deal.II tutorial step-22.
"""


def _stokes_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    Based on deal.II step-22.
    """
    refinements = params.get("refinements", 4)
    return f'''\
/* Stokes flow: lid-driven cavity — Taylor-Hood Q2/Q1 — deal.II (step-22 based) */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <fstream>
using namespace dealii;

template <int dim>
class LidVelocity : public Function<dim> {{
public:
  LidVelocity() : Function<dim>(dim + 1) {{}}
  void vector_value(const Point<dim> &p, Vector<double> &values) const override {{
    values = 0;
    values(0) = 1.0; // u_x = 1 on lid
  }}
}};

int main() {{
  const int dim = 2;
  Triangulation<dim> tria;
  // colorize=true assigns face boundary_ids 0..3 (left,right,bottom,top)
  // so the lid BC can target ONLY the top face — with a single id the
  // template degenerated to all-zero BCs and a trivial zero solution.
  GridGenerator::hyper_cube(tria, 0, 1, /*colorize=*/true);
  tria.refine_global({refinements});

  FESystem<dim> fe(FE_Q<dim>(2), dim, FE_Q<dim>(1), 1); // Q2 velocity + Q1 pressure
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  // Group ALL velocity components into block 0, pressure into block 1
  // (the step-22 pattern). component_wise(dof_handler) WITHOUT the
  // block_component argument renumbers by COMPONENT (dim+1 groups:
  // ux, uy, p) while count_dofs_per_fe_block then reports per-FE-base
  // blocks — the mismatched block sizes segfault during assembly.
  std::vector<unsigned int> block_component(dim + 1, 0);
  block_component[dim] = 1;
  DoFRenumbering::component_wise(dof_handler, block_component);

  const std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
  std::cout << "DOFs: " << dof_handler.n_dofs()
            << " (vel=" << dofs_per_block[0] << ", pres=" << dofs_per_block[1] << ")" << std::endl;

  BlockDynamicSparsityPattern dsp(2, 2);
  for (unsigned int i = 0; i < 2; ++i)
    for (unsigned int j = 0; j < 2; ++j)
      dsp.block(i, j).reinit(dofs_per_block[i], dofs_per_block[j]);
  dsp.collect_sizes();
  DoFTools::make_sparsity_pattern(dof_handler, dsp);

  BlockSparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  BlockSparseMatrix<double> system_matrix;
  system_matrix.reinit(sparsity_pattern);
  BlockVector<double> solution(dofs_per_block);
  BlockVector<double> system_rhs(dofs_per_block);

  // Assembly
  QGauss<dim> quadrature(3);
  FEValues<dim> fe_values(fe, quadrature,
    update_values | update_gradients | update_JxW_values | update_quadrature_points);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  for (const auto &cell : dof_handler.active_cell_iterators()) {{
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell_rhs = 0;
    for (unsigned int q = 0; q < quadrature.size(); ++q) {{
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {{
        const auto sym_grad_phi_i = fe_values[velocities].symmetric_gradient(i, q);
        const double div_phi_i = fe_values[velocities].divergence(i, q);
        const double phi_i_p = fe_values[pressure].value(i, q);
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {{
          const auto sym_grad_phi_j = fe_values[velocities].symmetric_gradient(j, q);
          const double div_phi_j = fe_values[velocities].divergence(j, q);
          const double phi_j_p = fe_values[pressure].value(j, q);
          cell_matrix(i, j) += (2.0 * scalar_product(sym_grad_phi_i, sym_grad_phi_j)
                                - div_phi_i * phi_j_p - phi_i_p * div_phi_j)
                               * fe_values.JxW(q);
        }}
      }}
    }}
    cell->get_dof_indices(local_dof_indices);
    for (unsigned int i = 0; i < dofs_per_cell; ++i) {{
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
        system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
      system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }}
  }}

  // BCs: lid velocity u=(1,0) on the top face (id 3), no-slip on the
  // other three. Constrain VELOCITY components only — interpolating a
  // ZeroFunction(dim+1) without the component mask also pins every
  // boundary pressure DoF, which is wrong physics.
  const FEValuesExtractors::Vector velocities_bc(0);
  std::map<types::global_dof_index, double> boundary_values;
  for (const auto face_id : {{0, 1, 2}})
    VectorTools::interpolate_boundary_values(dof_handler, face_id,
      Functions::ZeroFunction<dim>(dim + 1), boundary_values,
      fe.component_mask(velocities_bc));
  VectorTools::interpolate_boundary_values(dof_handler, 3,
    LidVelocity<dim>(), boundary_values,
    fe.component_mask(velocities_bc));
  // Pin ONE pressure DoF: with velocity Dirichlet on the whole
  // boundary, pressure is only defined up to a constant and the
  // saddle-point matrix is singular without this.
  boundary_values[dofs_per_block[0]] = 0.0;
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

  // Solve with MinRes (symmetric indefinite saddle-point system).
  // SparseDirectUMFPACK is the obvious direct choice but conda-forge
  // deal.II ships WITHOUT UMFPACK (config.h '#undef
  // DEAL_II_WITH_UMFPACK') — the class compiles, then
  // initialize()/factorize()/solve() throw ExcMessage("To call this
  // function you need UMFPACK, but you configured deal.II without passing
  // the necessary switch to 'cmake'. ...") at runtime in sparse_direct.cc.
  // MinRes with the
  // identity preconditioner is slow but portable; for production use
  // the step-22 block Schur preconditioner.
  SolverControl solver_control(20000, 1e-8 * system_rhs.l2_norm());
  SolverMinRes<BlockVector<double>> solver(solver_control);
  PreconditionIdentity preconditioner;
  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  std::cout << "MinRes converged in " << solver_control.last_step()
            << " iterations" << std::endl;

  std::cout << "Stokes solved, max velocity: " << solution.block(0).linfty_norm() << std::endl;

  DataOut<dim> data_out;
  std::vector<std::string> names(dim, "velocity");
  names.push_back("pressure");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
  interpretation.push_back(DataComponentInterpretation::component_is_scalar);
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, names, DataOut<dim>::type_dof_data, interpretation);
  data_out.build_patches();
  std::ofstream output("solution.vtu");
  data_out.write_vtu(output);
  std::cout << "VTU written." << std::endl;
  return 0;
}}
'''


# ── Knowledge ────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "description": "Stokes flow (step-22, step-55 parallel, step-56 GMG)",
    "tutorial_steps": ["step-22 (basic, block system)", "step-55 (MPI parallel)",
                      "step-56 (geometric multigrid on the velocity block, SOR smoother, inside a block Schur preconditioner)"],
    "function_space": "FESystem<dim>(FE_Q<dim>(2), dim, FE_Q<dim>(1), 1) — Taylor-Hood Q2/Q1",
    "solver": "Block Schur complement: A*u = f - B^T*p, then S*p = B*A^{-1}*f - g",
    "block_system": "GMRES with Schur complement preconditioner, or direct UMFPACK for small",
    # ── Structured catalog (post-canonical-element refactor).
    #    Each per-physics applicability note focuses on the inf-sup
    #    stability + conservation tradeoff specific to Stokes; the
    #    canonical class semantics live in element_catalog.
    "elements": {
        "FESystem":
            "Block-vector wrapper for ALL Stokes pairs. Most "
            "common shape: FESystem<dim>(FE_Q<dim>(p+1), dim, "
            "FE_Q<dim>(p), 1) for Taylor-Hood. The 'dim' "
            "velocity components + 1 pressure component are then "
            "block-renumbered via DoFRenumbering::component_wise.",
        "FE_Q":
            "Used twice in Taylor-Hood: degree p+1 for velocity, "
            "degree p for pressure. Q2/Q1 (p=1) is the canonical "
            "low-Re default. Q1/Q1 (equal-order) is NOT inf-sup "
            "stable and produces checkerboard pressure unless "
            "stabilised (SUPG / GLS / VMS).",
        "FE_Q_Bubbles":
            "Velocity component of the MINI element (paired with "
            "FE_DGP pressure). Inf-sup stable at p=1, cheaper "
            "than Taylor-Hood Q2/Q1 in DoF count.",
        "FE_RaviartThomas":
            "Velocity component of the RT/DGQ H(div) mixed pair. "
            "Produces EXACTLY divergence-free discrete velocity — "
            "the right choice when conservation matters (Darcy, "
            "groundwater, geophysical flow).",
        "FE_BDM":
            "Brezzi-Douglas-Marini velocity for the BDM/DGP pair. "
            "Fewer DoFs per cell than FE_RaviartThomas at the "
            "same polynomial order; same divergence-free property.",
        "FE_RannacherTurek":
            "Velocity component of the P1NC/P0 pair (paired with "
            "FE_DGQ(0)). Inf-sup stable on quads, cheap, no "
            "Taylor-Hood h^2 bubble — useful when DoF budget is "
            "tight.",
        "FE_BernardiRaugel":
            "Vector Lagrange + edge bubbles. Inf-sup stable when "
            "paired with piecewise-constant pressure; competitive "
            "with Taylor-Hood at low p.",
        "FE_DGP":
            "Pressure component of MINI / RT/DGP / BDM/DGP. "
            "Discontinuous P_k space with a Legendre basis; one degree less than "
            "the velocity (so pair FE_Q_Bubbles(1) + FE_DGP(0), "
            "RT(k) + DGP(k), BDM(k) + DGP(k-1)).",
        "FE_DGQ":
            "Pressure component of RT/DGQ pair, or of fully-DG "
            "Stokes. Tensor-product DG; pair RT(k)+DGQ(k) for "
            "exactly-div-free.",
        "FE_Nothing":
            "Use inside FESystem on subdomains where flow is "
            "suppressed (solid inclusions in FSI, ALE-frozen "
            "regions).",
    },
    "mesh_generators": {
        "hyper_cube": "Driven-cavity Stokes geometry; the classic reference series is Ghia et al. (1982) — retrieve it, it is not stored here.",
        "channel_with_cylinder": "Schäfer-Turek channel-with-cylinder geometry — cylinder at (0.2, 0.2) in a (2.2 × 0.41) channel; retrieve the published lift/drag for the Re you run.",
        "hyper_rectangle": "Generic channel domain — inflow left, outflow right.",
        "hyper_L": "Backward-facing step; classic recirculating-flow test.",
        "subdivided_hyper_rectangle": "Anisotropic refinement (taller in y than long in x) for boundary-layer resolution.",
        "hyper_cube_with_cylindrical_hole": "Flow around a cylinder; vortex-shedding at Re > 47.",
        "cheese": "Domain with holes — multiscale / porous-flow demo.",
    },
    "solvers": [
        "SolverGMRES<>                — the canonical choice; Stokes is indefinite so SolverCG WILL fail.",
        "SolverMinRes<>               — symmetric indefinite; preferable to GMRES when a symmetric Schur preconditioner is used.",
        "SolverFGMRES<>               — flexible GMRES; needed when the inner preconditioner is itself an iterative solver (e.g. Schur complement built from an AMG-preconditioned CG on the velocity block).",
        "TrilinosWrappers::SolverDirect (UMFPACK / KLU) — direct for small problems; useful as a reference truth when iterative chains start producing wrong answers.",
    ],
    "preconditioners": [
        "BlockSchurPreconditioner (step-22 §) — the textbook approach; A_inv via inner CG on the velocity block, S_inv via the pressure mass matrix scaled by 1/viscosity.",
        "PreconditionAMG / BoomerAMG on the velocity block — TrilinosWrappers::PreconditionAMG (Trilinos ML) or PETScWrappers::PreconditionBoomerAMG (hypre via PETSc); needed to scale beyond ~10^5 DoFs.",
        "Geometric multigrid on the velocity block only (step-56: mg::SmootherRelaxation with PreconditionSOR, inside BlockSchurPreconditioner) — point Jacobi / Gauss-Seidel cannot be applied to the full block system because the pressure block has a zero diagonal; Vanka-type preconditioners for the full system exist as SparseVanka / SparseBlockVanka, but no tutorial uses them.",
    ],
    "pitfalls": [
        "[Numerical] The Stokes system is INDEFINITE (saddle "
        "point), so SolverCG has no convergence guarantee — use "
        "SolverGMRES / SolverMinRes / a direct solver. But do NOT "
        "rely on CG announcing the problem: it does not report a "
        "'breakdown'. Signal: SolverCG either exhausts "
        "SolverControl's iteration budget and throws "
        "SolverControl::NoConvergence with SolverControl::"
        "last_value() far ABOVE the starting residual, or it "
        "converges and you only find out by cross-checking against "
        "SolverMinRes / SparseDirectUMFPACK. Measured on deal.II "
        "deal.II 9.8 by "
        "running SolverCG on the very matrix this template "
        "assembles (Taylor-Hood Q2/Q1 lid-driven cavity, "
        "PreconditionIdentity): CG CONVERGED in 1651 iterations to "
        "residual 1.39e-07 and landed on the same answer as "
        "SolverMinRes (1506 iterations), relative difference "
        "6.3e-04. On a nastier synthetic saddle point the same CG "
        "call instead exhausted its iteration budget and threw "
        "SolverControl::NoConvergence at step 300 with the "
        "residual blown up to 3.0e+17. Both outcomes are possible; "
        "the earlier claim 'breakdown on iteration 2-3 with a "
        "negative inner product' is NOT — it did not reproduce, and "
        "SolverCG has no breakdown message of that shape to print. "
        "Diagnose by definiteness or by "
        "cross-checking against MinRes/UMFPACK, not by waiting for "
        "CG to complain.",
        "[Syntax] Block structure: use "
        "DoFRenumbering::component_wise + BlockSparseMatrix. Without "
        "the renumbering the velocity and pressure DoFs are "
        "interleaved and the BlockSparseMatrix indexing is wrong. "
        "Signal: assembly succeeds but on the assembled "
        "BlockSparseMatrix system, block(1, 1).frobenius_norm() "
        "(the pressure block) is NON-ZERO when it should be zero "
        "for an unstabilised saddle-point Stokes system. Do NOT "
        "use the block SIZES as the diagnostic: "
        "DoFTools::count_dofs_per_fe_block reports the correct "
        "proportion whether or not you renumber. Measured on "
        "deal.II 9.8, FESystem(FE_Q(2), 2, "
        "FE_Q(1), 1) on a 2-refinement colorized hyper_cube: "
        "counts are (162, 25) BEFORE DoFRenumbering::"
        "component_wise and (162, 25) AFTER — identical. The "
        "renumbering makes each component's DoFs CONTIGUOUS, which "
        "is what BlockSparseMatrix indexing needs; it does not "
        "change how many DoFs each block has. (This entry used to "
        "claim the counts come out 'in the wrong proportion (e.g. "
        "(n, n) instead of (dim*n, n))'; they do not.)",
        "[Physics] Pressure is determined up to a constant for pure "
        "Neumann (closed-cavity) problems — pin one DoF or add a "
        "zero-mean constraint via AffineConstraints. Signal: "
        "`solution.block(1).linfty_norm()` (the pressure block) "
        "drifts to >1e10 magnitude across iterations while "
        "`solution.block(0).l2_norm()` (velocity) converges "
        "normally; SolverGMRES iteration count grows each outer "
        "step as the null space pollutes the Krylov basis.",
        "[Numerical] For geometric multigrid: apply it to the "
        "velocity block only, as step-56 does (SOR smoother inside a "
        "block Schur preconditioner), NOT point Jacobi on the full "
        "saddle-point system. Point Jacobi diverges "
        "on saddle-point systems because the pressure block has "
        "zero diagonal. Signal: MGSmootherRelaxation with point "
        "Jacobi — SolverGMRES residual norm stagnates at ~1e-2 "
        "after the first V-cycle and refuses to drop further, even "
        "though SparseDirectUMFPACK on the same matrix converges "
        "to machine precision; SolverControl::last_step() reports "
        "max_iterations reached.",
        "[Numerical] Inf-sup (Ladyzhenskaya-Babuška-Brezzi) stability "
        "is REQUIRED. Equal-order pairs (Q1/Q1, Q2/Q2) are NOT "
        "inf-sup stable — they look like they converge in 1D tests "
        "but produce checkerboard pressure modes in 2D. Use "
        "Taylor-Hood (Q_{p+1}/Q_p), MINI (Q1+bubble/Q1), "
        "RT/DGQ or BDM/DGP. Signal: DataOut output for the "
        "BlockVector pressure component shows a regular "
        "high-frequency checkerboard pattern superimposed on the "
        "smooth solution; VectorTools::point_value evaluated at "
        "adjacent cell centroids alternates sign with O(1) "
        "magnitude.",
        "[Physics] FE_RaviartThomas-based H(div) velocity pairs "
        "produce EXACTLY divergence-free velocity at the discrete "
        "level — this is correct, not a bug. If the user is "
        "comparing against a Q2/Q1 result where div(u) ≈ 1e-3, the "
        "RT/DGQ result will report div(u) ≈ 1e-15. Signal: "
        "`VectorTools::integrate_difference` for div(u) returns "
        "values at machine epsilon (1e-15 to 1e-14) with FE_"
        "RaviartThomas — orders of magnitude smaller than the "
        "1e-3 to 1e-5 typical of FE_Q Taylor-Hood at the same h. "
        "This is the expected H(div) behaviour, not a bug.",
        "[Integration] channel_with_cylinder gives the Schäfer-Turek "
        "geometry, and the published numbers belong to those exact "
        "dimensions — cylinder centre (0.2, 0.2), channel 2.2 × 0.41. "
        "Change either and your run is a different problem, so a "
        "comparison against a retrieved reference is meaningless; "
        "off-by-one here is a common and silent cause of a "
        "'wrong' result. Signal: a drag coefficient from "
        "VectorTools::compute_mean_value or user-side "
        "post-processing that differs from the reference you "
        "retrieved by more than about 10%, where the offset is "
        "SYSTEMATIC across Triangulation::refine_global rather than "
        "shrinking — a discrepancy that does not close under "
        "refinement is a setup error, not discretisation error. "
        "Retrieve the reference and cite it; it is not stored here.",
    ],
}
