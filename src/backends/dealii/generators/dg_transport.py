"""Discontinuous Galerkin transport templates for deal.II.

Based on deal.II tutorial step-12 (DG advection with upwind flux).
"""


def _dg_transport_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.
    All parameter defaults are placeholders.
    DG method for advection — based on step-12 pattern.
    """
    refinements = params.get("refinements", 4)
    degree = params.get("degree", 1)
    beta_x = params.get("beta_x", 1.0)
    beta_y = params.get("beta_y", 1.0)
    return f'''\
/* Discontinuous Galerkin for advection — based on deal.II step-12 pattern
 * Solves beta . grad(u) = 0 with upwind flux on unit square.
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <fstream>
#include <iostream>

using namespace dealii;

// deal.II 9.4 renamed the FEInterfaceValues shape accessors:
//   9.3.x:  fe_iv.jump(i, q)               fe_iv.average(i, q)
//   9.4+:   fe_iv.jump_in_shape_values(i,q) fe_iv.average_of_shape_values(i,q)
// Map the new names onto the old ones on 9.3 so the same template
// compiles on both lines (conda-forge currently ships 9.3.2).
#if !DEAL_II_VERSION_GTE(9, 4, 0)
#  define jump_in_shape_values jump
#  define average_of_shape_values average
// NOTE: do NOT blanket-#define normal_vector -> normal here:
// FEFaceValues::normal_vector exists on 9.3 already; only
// FEInterfaceValues lacks it. The interface call site below uses
// an inline version guard instead.
#endif

// Advection velocity field — set for your problem
template <int dim>
Tensor<1, dim> beta(const Point<dim> & /* p */)
{{
  Tensor<1, dim> wind;
  wind[0] = {beta_x};
  wind[1] = {beta_y};
  return wind;
}}

// Inflow boundary condition — set for your problem
template <int dim>
class BoundaryValues : public Function<dim>
{{
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int) const override
  {{
    // Inflow value — set for your problem
    if (p[0] < 1e-10 || p[1] < 1e-10)
      return 1.0;
    return 0.0;
  }}
}};

// Scratch and copy data for MeshWorker
template <int dim>
struct ScratchData
{{
  ScratchData(const FiniteElement<dim> &fe,
              const unsigned int        quadrature_degree)
    : fe_values(fe,
                QGauss<dim>(quadrature_degree),
                update_values | update_gradients | update_quadrature_points |
                  update_JxW_values)
    , fe_face_values(fe,
                     QGauss<dim - 1>(quadrature_degree),
                     update_values | update_quadrature_points |
                       update_JxW_values | update_normal_vectors)
    , fe_interface_values(fe,
                          QGauss<dim - 1>(quadrature_degree),
                          update_values | update_quadrature_points |
                            update_JxW_values | update_normal_vectors)
  {{}}

  ScratchData(const ScratchData<dim> &scratch_data)
    : fe_values(scratch_data.fe_values.get_fe(),
                scratch_data.fe_values.get_quadrature(),
                update_values | update_gradients | update_quadrature_points |
                  update_JxW_values)
    , fe_face_values(scratch_data.fe_face_values.get_fe(),
                     scratch_data.fe_face_values.get_quadrature(),
                     update_values | update_quadrature_points |
                       update_JxW_values | update_normal_vectors)
    , fe_interface_values(scratch_data.fe_interface_values.get_fe(),
                          scratch_data.fe_interface_values.get_quadrature(),
                          update_values | update_quadrature_points |
                            update_JxW_values | update_normal_vectors)
  {{}}

  FEValues<dim>          fe_values;
  FEFaceValues<dim>      fe_face_values;
  FEInterfaceValues<dim> fe_interface_values;
}};

struct CopyData
{{
  FullMatrix<double>                   cell_matrix;
  Vector<double>                       cell_rhs;
  std::vector<types::global_dof_index> local_dof_indices;

  struct FaceData
  {{
    FullMatrix<double>                   cell_matrix;
    std::vector<types::global_dof_index> joint_dof_indices;
  }};
  std::vector<FaceData> face_data;
}};

int main()
{{
  const unsigned int dim = 2;

  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global({refinements});

  const unsigned int degree = {degree};
  FE_DGQ<dim>     fe(degree);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  std::cout << "DG transport: " << dof_handler.n_dofs() << " DOFs, "
            << triangulation.n_active_cells() << " cells" << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> system_matrix(sparsity_pattern);
  Vector<double>       solution(dof_handler.n_dofs());
  Vector<double>       system_rhs(dof_handler.n_dofs());

  const BoundaryValues<dim> boundary_function;
  const QGauss<dim>         quadrature(degree + 1);
  const QGauss<dim - 1>     face_quadrature(degree + 1);

  // Cell worker: volume integral beta . grad(phi_j) * phi_i
  const auto cell_worker = [&](const auto &cell, auto &scratch, auto &copy) {{
    copy.cell_matrix.reinit(fe.n_dofs_per_cell(), fe.n_dofs_per_cell());
    copy.cell_rhs.reinit(fe.n_dofs_per_cell());
    copy.local_dof_indices.resize(fe.n_dofs_per_cell());
    copy.face_data.clear();

    scratch.fe_values.reinit(cell);
    cell->get_dof_indices(copy.local_dof_indices);

    for (unsigned int q = 0; q < scratch.fe_values.n_quadrature_points; ++q)
      {{
        const auto beta_q = beta<dim>(scratch.fe_values.quadrature_point(q));
        for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
          for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
            copy.cell_matrix(i, j) += -(beta_q * scratch.fe_values.shape_grad(i, q)) *
                                        scratch.fe_values.shape_value(j, q) *
                                        scratch.fe_values.JxW(q);
      }}
  }};

  // Boundary worker: upwind flux on domain boundary
  const auto boundary_worker = [&](const auto &cell, const unsigned int face_no,
                                    auto &scratch, auto &copy) {{
    scratch.fe_face_values.reinit(cell, face_no);
    const auto &fe_fv = scratch.fe_face_values;

    CopyData::FaceData face_copy;
    face_copy.joint_dof_indices.resize(fe.n_dofs_per_cell());
    cell->get_dof_indices(face_copy.joint_dof_indices);
    face_copy.cell_matrix.reinit(fe.n_dofs_per_cell(), fe.n_dofs_per_cell());

    for (unsigned int q = 0; q < fe_fv.n_quadrature_points; ++q)
      {{
        const auto     beta_q    = beta<dim>(fe_fv.quadrature_point(q));
        const double   beta_dot_n = beta_q * fe_fv.normal_vector(q);
        const Point<dim> &q_point  = fe_fv.quadrature_point(q);

        if (beta_dot_n >= 0) // outflow
          for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
            for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
              face_copy.cell_matrix(i, j) += beta_dot_n *
                                              fe_fv.shape_value(j, q) *
                                              fe_fv.shape_value(i, q) *
                                              fe_fv.JxW(q);
        else // inflow: RHS only, no matrix contribution (step-12)
          {{
            const double g = boundary_function.value(q_point, 0);
            for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
              copy.cell_rhs(i) += -beta_dot_n * g *
                                    fe_fv.shape_value(i, q) *
                                    fe_fv.JxW(q);
          }}
      }}
    copy.face_data.push_back(face_copy);
  }};

  // Face worker: upwind flux on interior faces
  const auto face_worker = [&](const auto &cell, const unsigned int f,
                                const unsigned int sf,
                                const auto &ncell, const unsigned int nf,
                                const unsigned int nsf,
                                auto &scratch, auto &copy) {{
    scratch.fe_interface_values.reinit(cell, f, sf, ncell, nf, nsf);
    const auto &fe_iv = scratch.fe_interface_values;

    CopyData::FaceData face_copy;
    const unsigned int n_interface_dofs = fe_iv.n_current_interface_dofs();
    face_copy.joint_dof_indices = fe_iv.get_interface_dof_indices();
    face_copy.cell_matrix.reinit(n_interface_dofs, n_interface_dofs);

    for (unsigned int q = 0; q < fe_iv.n_quadrature_points; ++q)
      {{
        // get_quadrature_points()[q], not quadrature_point(q): the
        // per-point accessor on FEInterfaceValues only exists since
        // deal.II 9.7.0; the vector accessor is available on 9.3 too.
        const auto   beta_q    = beta<dim>(fe_iv.get_quadrature_points()[q]);
        // FEInterfaceValues gained normal_vector(q) only in 9.7.0;
        // 9.3 to 9.6 call it normal(q), which 9.7.0 deprecates.
        // (FEFaceValues::normal_vector exists on all of them, so no
        // blanket rename is possible.)
#if DEAL_II_VERSION_GTE(9, 7, 0)
        const double beta_dot_n = beta_q * fe_iv.normal_vector(q);
#else
        const double beta_dot_n = beta_q * fe_iv.normal(q);
#endif

        for (unsigned int i = 0; i < n_interface_dofs; ++i)
          for (unsigned int j = 0; j < n_interface_dofs; ++j)
            {{
              // Upwind flux (step-12): [phi_i] * phi_j^{{upwind}} * (beta.n).
              // The trial function is taken from the UPWIND side via
              // FEInterfaceValues::shape_value(here_or_there, j, q); the
              // jump/average form with the roles of i and j swapped (as this
              // template once carried) yields a SINGULAR operator.
              face_copy.cell_matrix(i, j) +=
                fe_iv.jump_in_shape_values(i, q) *
                fe_iv.shape_value((beta_dot_n > 0), j, q) *
                beta_dot_n *
                fe_iv.JxW(q);
            }}
      }}
    copy.face_data.push_back(face_copy);
  }};

  // Copier: distribute to global system
  const auto copier = [&](const CopyData &copy) {{
    for (unsigned int i = 0; i < copy.local_dof_indices.size(); ++i)
      {{
        for (unsigned int j = 0; j < copy.local_dof_indices.size(); ++j)
          system_matrix.add(copy.local_dof_indices[i],
                            copy.local_dof_indices[j],
                            copy.cell_matrix(i, j));
        system_rhs(copy.local_dof_indices[i]) += copy.cell_rhs(i);
      }}
    for (const auto &fd : copy.face_data)
      for (unsigned int i = 0; i < fd.joint_dof_indices.size(); ++i)
        for (unsigned int j = 0; j < fd.joint_dof_indices.size(); ++j)
          system_matrix.add(fd.joint_dof_indices[i],
                            fd.joint_dof_indices[j],
                            fd.cell_matrix(i, j));
  }};

  ScratchData<dim> scratch(fe, degree + 1);
  CopyData         copy;

  MeshWorker::mesh_loop(dof_handler.begin_active(),
                         dof_handler.end(),
                         cell_worker,
                         copier,
                         scratch,
                         copy,
                         MeshWorker::assemble_own_cells |
                           MeshWorker::assemble_boundary_faces |
                           MeshWorker::assemble_own_interior_faces_once,
                         boundary_worker,
                         face_worker);

  // Solve with GMRES: the DG advection matrix is non-symmetric, and
  // plain Richardson iteration with BlockSSOR was observed to throw
  // SolverControl::NoConvergence on it — its
  // convergence requires the preconditioned spectrum inside the unit
  // disk, which the upwind flux does not guarantee. GMRES with the
  // same block preconditioner is the robust choice. Tolerance is
  // RELATIVE to the rhs norm (absolute 1e-12 is unreachable noise).
  SolverControl            solver_control(2000, 1e-10 * system_rhs.l2_norm());
  SolverGMRES<Vector<double>> solver(solver_control);
  PreconditionBlockSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, fe.n_dofs_per_cell());
  solver.solve(system_matrix, solution, system_rhs, preconditioner);

  std::cout << "DG transport solved: " << solver_control.last_step()
            << " iterations" << std::endl;

  // Output
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches(degree);
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);

  std::cout << "DG transport: " << dof_handler.n_dofs() << " DOFs complete"
            << std::endl;
  return 0;
}}
'''


# ── Knowledge ────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "description": "Discontinuous Galerkin for advection (step-12, step-30, step-67)",
    "tutorial_steps": ["step-12 (DG upwind, MeshWorker)", "step-30 (anisotropic refinement)",
                      "step-67 (Euler equations, matrix-free DG)"],
    "function_space": "FE_DGQ<dim>(p) — tensor-product DG, or FE_DGP<dim>(p) — polynomial",
    "solver": "Richardson + block SSOR, or direct (UMFPACK) for small systems",
    "numerical_flux": {
        "upwind": "Use value from element where beta.n > 0 (simplest, diffusive)",
        "Lax-Friedrichs": "0.5*(F_L + F_R) + 0.5*alpha*(u_L - u_R), alpha=max|beta.n|",
        "central": "Average flux (not stable for advection-dominated)",
    },
    # ── 2026-08-03 execution pass (deal.II 9.8.0-pre, Release):
    #    the dg_transport_2d TEMPLATE itself was broken and aborted
    #    at run time with SolverControl::NoConvergence ("residual in
    #    the last step was -nan", GMRES step 0; SparseDirectUMFPACK
    #    on the same matrix returned linfty 3.6e+17 => singular
    #    operator). Four assembly bugs, all now fixed against
    #    upstream examples/step-12:
    #      1. cell term used -phi_i * (beta . grad phi_j) instead of
    #         -(beta . grad phi_i) * phi_j (test/trial swapped);
    #      2. CopyData::face_data was never cleared per cell, so every
    #         face block was re-added on every later cell;
    #      3. INFLOW boundary faces also wrote a matrix block; step-12
    #         puts inflow data on the RHS only;
    #      4. the interior flux used jump(j)*average(i) + penalty
    #         instead of jump(i) * phi_j^upwind * (beta.n).
    #    After the fix the template solves in 2 GMRES iterations and
    #    reproduces the exact solution of its own problem
    #    (beta=(1,1), g=1 on the inflow faces => u == 1) to
    #    min = max = 1.0.
    "pitfalls": [
        "[Syntax] DG requires the FLUX sparsity pattern: "
        "DoFTools::make_flux_sparsity_pattern. The standard "
        "make_sparsity_pattern couples only DoFs that share a cell, "
        "so every face-coupling entry an upwind DG assembly wants to "
        "write is missing from it. The two builders differ by roughly "
        "a factor of the face-neighbour count — on a globally refined "
        "FE_DGQ(1) quadrilateral mesh the flux pattern held about 4x "
        "the non-zeros of the cell-only one — so comparing "
        "DynamicSparsityPattern::n_nonzero_elements() of the two is a "
        "cheap positive check. "
        "WHAT YOU SEE DEPENDS ON THE BUILD TYPE. "
        "Debug: SparseMatrix::add(i, j, v) into a missing entry fires "
        "Assert(..., ExcInvalidIndex(i, j)) in sparse_matrix.h and "
        "ABORTS (exit 134) with 'You are trying to access the matrix "
        "entry with index <i,j>, but this entry does not exist in the "
        "sparsity pattern of this matrix.' — which is exactly the "
        "diagnosis, for free. "
        "Release: that guard is an Assert, so it is compiled out; the "
        "call takes the `index == SparsityPattern::invalid_entry` "
        "branch, returns normally, and SILENTLY DROPS v. Verified in "
        "both builds on the same program. "
        "So on a Release build the only observable is downstream and "
        "physical: the face coupling never entered the operator, and "
        "DataOut shows a DG solution with continuous-Galerkin-like "
        "behaviour across faces. Check "
        "CMAKE_BUILD_TYPE before deciding which failure to look for. "
        "Signal: build both patterns for the same DoFHandler and "
        "compare DynamicSparsityPattern::n_nonzero_elements() — the "
        "flux pattern must be substantially larger; if they are equal "
        "you built the wrong one. This check works in either build. (This entry used to quote "
        "ExcMessage('matrix entry at i,j does not exist in sparsity "
        "pattern'); no such string exists in the library.)",
        "[Syntax] FEInterfaceValues needed for face integrals "
        "(jump/average operators on cell interfaces). Using "
        "FEValues alone produces only cell-interior contributions. "
        "Signal: SolverGMRES converges but DataOut shows a smooth "
        "(non-DG) solution; jump-across-face values from "
        "VectorTools::integrate_difference vs reference are "
        "1e-8 (effectively zero) where they should be O(1) for "
        "upwind DG. (Note: the real dealii function is "
        "integrate_difference, NOT interpolate_difference; "
        "the latter does not exist in numerics/vector_tools.h.)",
        "[API] MeshWorker::mesh_loop simplifies cell / face / "
        "boundary assembly — without it, the user re-implements "
        "the dispatch logic and typically forgets the periodic-"
        "face case. Signal: assembly compiles and runs but the "
        "global system is non-symmetric AND inconsistent on "
        "periodic boundaries (if any); DataOut shows the solution "
        "with kinks at periodic-face nodes.",
        "[Numerical] PreconditionBlockSSOR with "
        "block_size = fe.n_dofs_per_cell() is the right DG "
        "preconditioner, but HOW MUCH it buys is not a fixed number "
        "— it depends on whether the global DoF numbering happens to "
        "run downstream of the advection direction, and no single "
        "speed-up figure is meaningful. State the condition, not a "
        "ratio. What was actually measured on this template's "
        "operator (FE_DGQ(1) upwind transport on a globally refined "
        "unit square, same rhs, same relative tolerance): "
        "(a) When the numbering IS downstream-compatible with beta, "
        "the block sweep is not a preconditioner at all but an EXACT "
        "block-triangular DIRECT SOLVE: one application of "
        "PreconditionBlockSSOR to the rhs already leaves a relative "
        "residual at machine precision (~1e-15), and GMRES then needs "
        "1-2 iterations at EVERY mesh size. Constant-beta cases like "
        "(1,1) and (0.3,1) fall in this class with deal.II's default "
        "cell ordering, which is why the ratio against point Jacobi "
        "is not a property of the preconditioner: point Jacobi's own "
        "count grows with the mesh (tens of iterations on a coarse "
        "mesh, several hundred on a fine one), so the ratio grew "
        "monotonically by more than an order of magnitude across "
        "five refinement levels. Quote the mesh or quote nothing. "
        "(b) When the numbering is NOT downstream-compatible — e.g. "
        "beta = (1,-1) with the default ordering — the same "
        "preconditioner is an ordinary preconditioner: one "
        "application leaves an O(1) residual and GMRES needs tens of "
        "iterations. DoFRenumbering::downstream restores case (a). "
        "(c) For a CURVED but well-posed field (rotation about a "
        "corner, so every characteristic still enters and leaves "
        "through the boundary) no downstream ordering exists at all "
        "and the block sweep is genuinely a preconditioner: the "
        "one-application residual stays O(1), the iteration count "
        "GROWS with the mesh, and the advantage settles at a modest, "
        "roughly mesh-stable factor — a few times better than point "
        "SSOR and one order of magnitude better than point Jacobi "
        "across three refinement levels. This is the regime a "
        "general speed-up figure could honestly describe, and it is "
        "nowhere near the ratios case (a) produces. "
        "(d) When the flow has CLOSED CHARACTERISTICS (rotation about "
        "the middle of a box, so interior streamlines never touch an "
        "inflow boundary) the pure transport operator is close to "
        "singular and every preconditioner in this list degrades — "
        "several exhaust a 5000-iteration budget. That is a "
        "well-posedness problem, not a preconditioner problem: add "
        "reaction or diffusion, or change the domain so the "
        "characteristics reach the boundary. "
        "(e) Scalar (point) SSOR is NOT reliably poor here, contrary "
        "to the older wording: because an SSOR sweep also follows the "
        "DoF ordering, its count on downstream-ordered problems was "
        "sometimes within a small factor of the block version. Point "
        "JACOBI is the one that consistently scales badly. "
        "Signal: apply the preconditioner ONCE to the rhs and measure "
        "||b - A*P(b)|| / ||b||. At ~1e-15 you have an exact solve "
        "and any iteration count is meaningless; at O(1) you have a "
        "real preconditioner and SolverControl::last_step() is the "
        "number to watch as the mesh is refined.",
        "[API] PreconditionBlock* (BlockSSOR / BlockSOR / "
        "BlockJacobi) with block_size = fe.n_dofs_per_cell() assumes "
        "each cell's DoFs are CONTIGUOUS in the global numbering. "
        "Most renumberings preserve that; DoF-WISE ones do not. "
        "DoFRenumbering::downstream(dof_handler, direction, "
        "dof_wise_renumbering) takes a third argument: pass FALSE "
        "(cell-wise) and the block structure survives; pass TRUE and "
        "the cell DoFs are interleaved, the 'blocks' no longer "
        "correspond to cells, and the preconditioner is garbage. "
        "Verified: with dof_wise_renumbering=true the one-application "
        "residual came back as NaN (or ~1e+162) and GMRES reported "
        "NoConvergence at step 0 or 1 with a NaN value; the identical "
        "run with dof_wise_renumbering=false gave an exact solve. "
        "Signal: NaN in SolverControl::last_value() at step 0-1, or a "
        "preconditioner one-application residual that is NaN or "
        "astronomically large, immediately after introducing a "
        "renumbering. Nothing is raised — this is silent in every "
        "build.",
        "[Numerical] For higher-order DG: FE_DGQHermite gives "
        "better matrix-free performance than FE_DGQ because the "
        "Hermite-like basis preserves face-value continuity, "
        "reducing the cross-face stencil weight. Signal: "
        "MatrixFree::cell_loop wall-time per iteration with "
        "FE_DGQ(6) is 2-3x larger than with FE_DGQHermite(6) at "
        "the same n_dofs.",
        "[Physics] No continuity constraints needed (DG has no "
        "hanging-node constraints), but you DO need to track "
        "non-conforming face DoFs explicitly when refining. "
        "Signal: AffineConstraints::distribute on a DG solution "
        "is a no-op. DoFTools::make_hanging_node_constraints on an "
        "FE_DGQ DoFHandler does NOT raise anything — it returns "
        "normally with n_constraints() == 0, even on a mesh that "
        "really has hanging nodes. Verified on an FE_DGQ(1) "
        "DoFHandler over a globally refined hypercube with one "
        "extra locally refined cell: n_constraints() == 0, no "
        "exception, in Debug and Release alike — there is no Assert "
        "here to switch on. (This entry used to quote ExcMessage('DG "
        "discretisation has no hanging-node constraints'); no such "
        "message exists.)",
        "[Physics] Inflow BCs: weakly enforced via numerical "
        "flux on boundary faces (NOT via AffineConstraints — DG "
        "has none). Setting Dirichlet values strongly is a "
        "common bug for users coming from CG. Signal: "
        "VectorTools::interpolate_boundary_values on a DG FE does "
        "NOT raise anything — it returns normally having written "
        "ZERO entries into the boundary-value map, because FE_DGQ "
        "has no face support points. Verified: "
        "interpolate_boundary_values(dh, 0, ConstantFunction(1.0), "
        "bv) on FE_DGQ(1) leaves bv.size() == 0 and raises nothing, "
        "in Debug and Release alike. (Contrast the CONTINUOUS "
        "non-interpolatory elements — FE_Q_Hierarchical, "
        "FE_Bernstein — where the same call SEGFAULTS instead of "
        "returning empty; see the essentials block.) "
        "The visible consequence is downstream: "
        "DataOut shows the prescribed Dirichlet value NOT appearing "
        "at the inflow boundary. Check bv.size() (or "
        "constraints.n_constraints()) immediately after the call. "
        "(This entry used to quote ExcMessage('strong boundary "
        "conditions not supported for DG').)",
    ],
}
