"""Hyperelasticity templates for deal.II.

Based on deal.II tutorial step-44 (Neo-Hookean, Newton-Raphson).
"""


def _hyperelasticity_3d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.
    All parameter defaults are placeholders.
    Finite-strain Neo-Hookean hyperelasticity — based on step-44 pattern.
    """
    refinements = params.get("refinements", 2)
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    pressure = params.get("pressure", 10.0)
    n_load_steps = params.get("n_load_steps", 10)
    return f'''\
/* Finite-strain Neo-Hookean hyperelasticity — based on deal.II step-44 pattern
 * Solves quasi-static large-deformation elasticity on a unit cube.
 * Material: compressible Neo-Hookean. Newton-Raphson nonlinear solver.
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
#include <fstream>
#include <iostream>
#include <cmath>

using namespace dealii;

int main()
{{
  const unsigned int dim = 3;

  // Material parameters — set for your problem
  const double E_mod  = {E};
  const double nu_val = {nu};
  const double mu     = E_mod / (2.0 * (1.0 + nu_val));
  const double lambda = E_mod * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val));
  const double kappa  = lambda + 2.0 / 3.0 * mu; // bulk modulus

  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
  triangulation.refine_global({refinements});

  // Set boundary IDs: 0=left(x=0), 1=right(x=1)
  for (auto &face : triangulation.active_face_iterators())
    if (face->at_boundary())
      {{
        if (std::abs(face->center()[0]) < 1e-10)
          face->set_boundary_id(0);
        else if (std::abs(face->center()[0] - 1.0) < 1e-10)
          face->set_boundary_id(1);
      }}

  FESystem<dim>   fe(FE_Q<dim>(1), dim);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  std::cout << "Hyperelasticity: " << dof_handler.n_dofs() << " DOFs, "
            << triangulation.n_active_cells() << " cells" << std::endl;

  const FEValuesExtractors::Vector displacement(0);

  // Total displacement vector
  Vector<double> solution(dof_handler.n_dofs());
  Vector<double> solution_delta(dof_handler.n_dofs());
  solution = 0;

  // Load stepping
  const unsigned int n_load_steps   = {n_load_steps};
  const double       total_pressure = {pressure};

  for (unsigned int load_step = 1; load_step <= n_load_steps; ++load_step)
    {{
      const double load_factor = static_cast<double>(load_step) / n_load_steps;
      const double current_pressure = load_factor * total_pressure;

      // Newton-Raphson iteration
      for (unsigned int newton_iter = 0; newton_iter < 20; ++newton_iter)
        {{
          // Constraints
          AffineConstraints<double> constraints;
          constraints.clear();
          // Fix left face (x=0)
          VectorTools::interpolate_boundary_values(
            dof_handler, 0, Functions::ZeroFunction<dim>(dim), constraints);
          constraints.close();

          // Assemble tangent stiffness and residual
          DynamicSparsityPattern dsp(dof_handler.n_dofs());
          DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
          SparsityPattern sparsity;
          sparsity.copy_from(dsp);

          SparseMatrix<double> tangent_matrix(sparsity);
          Vector<double>       residual(dof_handler.n_dofs());

          QGauss<dim>   quadrature(2);
          QGauss<dim-1> face_quadrature(2);

          FEValues<dim> fe_values(fe, quadrature,
                                  update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points);
          // every flag this loop reads: a Release-only deal.II build segfaults, with no message, on a missing one
          FEFaceValues<dim> fe_face_values(fe, face_quadrature,
                                            update_values | update_JxW_values |
                                            update_normal_vectors | update_quadrature_points);

          for (const auto &cell : dof_handler.active_cell_iterators())
            {{
              const unsigned int dpc = fe.n_dofs_per_cell();
              FullMatrix<double> cell_matrix(dpc, dpc);
              Vector<double>     cell_rhs(dpc);
              std::vector<types::global_dof_index> local_dof_indices(dpc);

              fe_values.reinit(cell);

              // Get displacement gradients at quadrature points
              std::vector<Tensor<2, dim>> grad_u(quadrature.size());
              fe_values[displacement].get_function_gradients(solution, grad_u);

              for (unsigned int q = 0; q < quadrature.size(); ++q)
                {{
                  // Deformation gradient F = I + grad_u
                  Tensor<2, dim> F = Physics::Elasticity::Kinematics::F(grad_u[q]);
                  const double   J = determinant(F);

                  // Right Cauchy-Green tensor C = F^T * F
                  const auto C     = Physics::Elasticity::Kinematics::C(F);
                  const auto C_inv = invert(C);

                  // Neo-Hookean: S = mu*(I - C^(-1)) + lambda*ln(J)*C^(-1)
                  Tensor<2, dim> S;
                  for (unsigned int i = 0; i < dim; ++i)
                    for (unsigned int j = 0; j < dim; ++j)
                      S[i][j] = mu * ((i == j ? 1.0 : 0.0) - C_inv[i][j])
                                + lambda * std::log(J) * C_inv[i][j];

                  // P = F * S (first Piola-Kirchhoff)
                  const Tensor<2, dim> P = F * S;

                  // Material tangent for Neo-Hookean
                  // C_mat_ijkl = lambda * C_inv_ij * C_inv_kl
                  //            + (mu - lambda*ln(J)) * (C_inv_ik*C_inv_jl + C_inv_il*C_inv_jk)

                  for (unsigned int i = 0; i < dpc; ++i)
                    {{
                      // Residual: integral P : grad(N_i) dV
                      Tensor<2, dim> grad_Ni;
                      for (unsigned int d = 0; d < dim; ++d)
                        grad_Ni[fe.system_to_component_index(i).first][d] =
                          fe_values.shape_grad(i, q)[d];

                      double val = 0;
                      for (unsigned int a = 0; a < dim; ++a)
                        for (unsigned int b = 0; b < dim; ++b)
                          val += P[a][b] * grad_Ni[a][b];
                      cell_rhs(i) -= val * fe_values.JxW(q);

                      for (unsigned int j = 0; j < dpc; ++j)
                        {{
                          Tensor<2, dim> grad_Nj;
                          for (unsigned int d = 0; d < dim; ++d)
                            grad_Nj[fe.system_to_component_index(j).first][d] =
                              fe_values.shape_grad(j, q)[d];

                          // Geometric stiffness: grad_Ni^T * S * grad_Nj
                          double geo = 0;
                          for (unsigned int a = 0; a < dim; ++a)
                            for (unsigned int b = 0; b < dim; ++b)
                              geo += grad_Ni[fe.system_to_component_index(i).first][a]
                                    * S[a][b]
                                    * grad_Nj[fe.system_to_component_index(j).first][b];

                          // Material stiffness (simplified)
                          double mat = 0;
                          const unsigned int ci = fe.system_to_component_index(i).first;
                          const unsigned int cj = fe.system_to_component_index(j).first;
                          for (unsigned int a = 0; a < dim; ++a)
                            for (unsigned int b = 0; b < dim; ++b)
                              {{
                                double C_abcd = lambda * C_inv[ci][a] * C_inv[cj][b]
                                  + (mu - lambda * std::log(J))
                                    * (C_inv[ci][cj] * C_inv[a][b] + C_inv[ci][b] * C_inv[a][cj]);
                                mat += fe_values.shape_grad(i, q)[a]
                                      * C_abcd
                                      * fe_values.shape_grad(j, q)[b];
                              }}

                          cell_matrix(i, j) += (geo + mat) * fe_values.JxW(q);
                        }}
                    }}
                }}

              // Neumann BC: pressure on right face (boundary_id=1) — set for your problem
              for (unsigned int f = 0; f < cell->n_faces(); ++f)
                if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == 1)
                  {{
                    fe_face_values.reinit(cell, f);
                    for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                      for (unsigned int i = 0; i < dpc; ++i)
                        {{
                          const unsigned int comp = fe.system_to_component_index(i).first;
                          cell_rhs(i) += current_pressure *
                                          fe_face_values.normal_vector(q)[comp] *
                                          fe_face_values.shape_value(i, q) *
                                          fe_face_values.JxW(q);
                        }}
                  }}

              cell->get_dof_indices(local_dof_indices);
              constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                                      local_dof_indices,
                                                      tangent_matrix, residual);
            }}

          // Check convergence
          const double residual_norm = residual.l2_norm();
          if (newton_iter == 0)
            std::cout << "Load step " << load_step << "/" << n_load_steps
                      << " (p=" << current_pressure << ")";
          if (residual_norm < 1e-8 * dof_handler.n_dofs())
            {{
              std::cout << ", converged in " << newton_iter << " iters"
                        << ", |R|=" << residual_norm << std::endl;
              break;
            }}

          // Solve for Newton update
          SolverControl solver_control(1000, 1e-12 * residual_norm);
          SolverCG<Vector<double>> solver(solver_control);
          PreconditionSSOR<SparseMatrix<double>> preconditioner;
          preconditioner.initialize(tangent_matrix, 1.2);
          solution_delta = 0;
          solver.solve(tangent_matrix, solution_delta, residual, preconditioner);
          constraints.distribute(solution_delta);

          // Backtracking line search: ensure J > 0 everywhere (prevent element inversion)
          double alpha = 1.0;
          for (unsigned int ls = 0; ls < 10; ++ls)
            {{
              Vector<double> trial_solution(solution);
              trial_solution.add(alpha, solution_delta);

              // Check minimum J over all quadrature points
              double J_min = 1e20;
              for (const auto &cell : dof_handler.active_cell_iterators())
                {{
                  fe_values.reinit(cell);
                  std::vector<std::vector<Tensor<1, dim>>> grad_u(
                    quadrature.size(), std::vector<Tensor<1, dim>>(dim));
                  fe_values.get_function_gradients(trial_solution, grad_u);
                  for (unsigned int q = 0; q < quadrature.size(); ++q)
                    {{
                      Tensor<2, dim> F_q = unit_symmetric_tensor<dim>();
                      for (unsigned int d = 0; d < dim; ++d)
                        F_q[d] += grad_u[q][d];
                      J_min = std::min(J_min, determinant(F_q));
                    }}
                }}
              if (J_min > 0.01)
                {{
                  solution = trial_solution;
                  break;
                }}
              alpha *= 0.5;
              if (ls == 9)
                {{
                  std::cerr << "WARNING: line search failed, J_min=" << J_min << std::endl;
                  solution.add(alpha, solution_delta);
                }}
            }}
        }}
    }}

  // Output final deformed configuration
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  std::vector<std::string> names(dim, "displacement");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interp(dim, DataComponentInterpretation::component_is_part_of_vector);
  data_out.add_data_vector(solution, names, DataOut<dim>::type_dof_data, interp);
  data_out.build_patches();
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);

  std::cout << "Hyperelasticity: max displacement = " << solution.linfty_norm()
            << std::endl;
  return 0;
}}
'''


# ── Knowledge ────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "description": "Finite-strain hyperelasticity (step-44, step-72 with AD)",
    "tutorial_steps": ["step-44 (three-field formulation, Neo-Hookean)",
                      "step-72 (automatic differentiation for tangent)",
                      "step-18 (quasi-static, updated Lagrangian)"],
    "function_space": "FESystem<dim>(FE_Q<dim>(1), dim) for displacement-only",
    "solver": "Newton-Raphson with line search, CG + SSOR for linear sub-problems",
    "constitutive_models": {
        "NeoHookean": "W = mu/2*(I1-3) + kappa/2*(J-1)^2, Cauchy: mu/J*(b-I) + kappa*(J-1)*I",
        "Mooney-Rivlin": "W = c1*(I1-3) + c2*(I2-3)",
        "Ogden": "W = sum_p mu_p/alpha_p * (lambda_1^alpha_p + ... - 3)",
        "Saint-Venant-Kirchhoff": "S = lambda*tr(E)*I + 2*mu*E (simplest, not frame-indifferent for compression)",
    },
    "kinematics": {
        "F": "Deformation gradient: F = I + grad(u)",
        "C": "Right Cauchy-Green: C = F^T * F",
        "b": "Left Cauchy-Green: b = F * F^T",
        "J": "Volume ratio: J = det(F)",
        "E": "Green-Lagrange: E = 0.5*(C - I)",
    },
    "elements": {
        "FESystem":
            "Vector wrapper for displacement-only formulations "
            "(step-18, step-72): FESystem<dim>(FE_Q<dim>(degree), "
            "dim). Or the three-field (u, p̃, J̃) composition per "
            "step-44 for nearly-incompressible problems.",
        "FE_Q":
            "Displacement field; degree=2 typical to mitigate "
            "volumetric locking at small compressibility. Pair "
            "with FE_DGP(degree-1) + FE_DGP(degree-2) inside "
            "FESystem for step-44 three-field hyperelasticity.",
        "FE_Q_Bubbles":
            "Improved volumetric behaviour over plain FE_Q "
            "without going to the full step-44 three-field "
            "formulation. Cheaper compromise.",
        "FE_RannacherTurek":
            "Locking-free P1-NC; cheap alternative when degree "
            "≥ 2 FE_Q is too expensive but the user needs the "
            "incompressible-limit response.",
        "FE_Q_Hierarchical":
            "For hp-adaptive refinement during load stepping — "
            "refine in regions where Newton stalls without "
            "throwing away the previous load step's coarse-DoF "
            "solution.",
        "FE_DGP":
            "Pressure and dilation fields in the step-44 "
            "three-field formulation. FE_DGP(degree-1) for the "
            "pressure-like field, FE_DGP(degree-2) for the "
            "Jacobian-like field.",
    },
    "mesh_generators": {
        "subdivided_hyper_rectangle": "Anisotropic beams / slabs. colorize=true REQUIRED for per-face BC distinction (else all faces share boundary_id=0).",
        "hyper_cube": "Cube-compression / tension test.",
        "hyper_L": "Re-entrant corner amplifies stress concentration; tests element-inversion handling.",
        "cylinder": "Bar-bending / torsion tests.",
        "hyper_shell": "Pressure-vessel and balloon-inflation problems.",
        "plate_with_a_hole": "Cook's-membrane-style stress concentration under large deformation.",
    },
    "solvers": [
        "Newton-Raphson with line search — outer loop; line search MUST check det(F) > 0 to avoid element inversion",
        "SparseDirectUMFPACK — preferred linear sub-solver for < 50k DoFs; more robust than iterative methods on the Newton tangent, which can become indefinite during line search",
        "SolverCG<> with PreconditionSSOR — linear sub-solver beyond 50k DoFs; assumes the tangent stays SPD (true at small deformation, may break near the limit point)",
        "SolverGMRES<>                — needed when the tangent has non-symmetric parts (e.g. follower-load tangent terms; geometric stiffness in updated Lagrangian)",
        "Differentiation::AD::EnergyFunctional — automatic differentiation for tangent (step-72); eliminates the manual K_geo+K_mat split",
    ],
    "preconditioners": [
        "PreconditionSSOR for SPD tangents — works for small-deformation Neo-Hookean before geometric-stiffness dominates",
        "PreconditionAMG / BoomerAMG for large problems; works on the displacement block of the three-field formulation",
        "BlockPreconditioner — required for the three-field (u, p̃, J̃) saddle-point structure of step-44; analogous to the Stokes block preconditioner",
    ],
    "pitfalls": [
        "[Numerical] Must use load stepping for large deformations — "
        "cold-start Newton diverges. Signal: SolverControl reports "
        "Newton residual.l2_norm() > 1e3 on iteration 1, growing "
        "to NaN by iteration 3; "
        "deal.II has NO Newton solver, so nothing in the library "
        "will announce this — the only message you get is one YOUR "
        "code raises. Write the guard: AssertThrow(newton_step < "
        "max_steps, ExcMessage(...)) with whatever wording you like "
        "— nothing in deal.II says Newton step did not converge, "
        "because deal.II has no Newton solver to say it. "
        "AssertThrow is active in Release too, unlike Assert. The "
        "library-side observable is the INNER linear solve: if the "
        "tangent has gone bad, SolverCG/SolverGMRES throws "
        "SolverControl::NoConvergence with 'Iterative method reported "
        "convergence failure in step <N>. The residual in the last "
        "step was <R>.' and R will be huge or nan.",
        "[Physics] MUST implement line search checking J=det(F) > 0. "
        "Without it, elements invert and the simulation crashes for "
        "any significant compression. Signal: assembly raises "
        "no such message exists in deal.II — there is no "
        "constitutive-model layer to raise it, so it has to be YOUR "
        "AssertThrow(J > 0, ExcMessage(\"det(F) <= 0 at quadrature "
        "point\")) inside the material evaluation. Write it: without "
        "it the evaluator silently returns NaN, the assembled tangent "
        "fills with NaN, and the first thing you see is the linear "
        "solver throwing SolverControl::NoConvergence with 'The "
        "residual in the last step was nan.' — DataOut last-saved "
        "frame shows mesh self-intersecting.",
        "[Physics] Neo-Hookean with J: S = mu*(I - C^{-1}) + "
        "lambda*ln(J)*C^{-1}. The ln(J) form is the standard "
        "compressible Neo-Hookean stress; using the squared-J "
        "variant gives the wrong incompressibility limit. Signal: "
        "VectorTools::integrate_difference vs reference shows "
        "stress error that grows with deformation magnitude "
        "(O(1) at large strain) instead of the expected O(h^p).",
        "[Syntax] Tangent has geometric + material parts: "
        "K = K_geo + K_mat. Assembling only K_mat gives a "
        "non-symmetric system; assembling K_mat + K_geo restores "
        "symmetry. Signal: do NOT wait for a solver complaint — "
        "deal.II cannot report a CG 'breakdown' (the guard in "
        "solver_cg.h is an Assert, compiled out in Release, and even "
        "in Debug it needs an exactly-zero operator; see the "
        "essentials block). Measure the asymmetry directly: loop the "
        "assembled matrix and compute max|A_ij - A_ji|, which sits at "
        "round-off for a correct tangent and at O(max|A|) when K_geo "
        "is missing. Running CG on the asymmetric tangent anyway "
        "gives either SolverControl::NoConvergence with a residual "
        "LARGER than the starting one, or silent convergence to the "
        "wrong answer; SolverGMRES on the same system converges and "
        "is the cross-check.",
        "[Numerical] For nearly incompressible: use mixed "
        "formulation (step-44) or F-bar. Single-field FE_Q(1) "
        "displacement locks at the volumetric limit. Signal: tip "
        "deflection on a Cook membrane differs from reference by "
        "30-50% as nu approaches 0.5; switching to the three-field "
        "FESystem(FE_Q(2), dim, FE_DGP(1), 1, FE_DGP(0), 1) "
        "recovers convergence to within 5%.",
        "[Physics] Saint-Venant-Kirchhoff unstable in compression. "
        "Use Neo-Hookean instead for any compression > ~30%. "
        "Signal: SolverControl knows nothing about Newton — it "
        "reports only on the INNER linear solve, so there is no "
        "'Newton breakdown' for it to report. What you observe is "
        "your own Newton residual norm failing to decrease "
        "monotonically at the second or third load step under "
        "compression, and (once the deformation gradient goes "
        "singular) the inner solve throwing "
        "SolverControl::NoConvergence with 'The residual in the last "
        "step was nan.' Track the Newton residual yourself and stop "
        "on the first non-decrease; that is the only diagnostic that "
        "exists at the nonlinear level.",
        "[Integration] Use roller BCs (constrain only normal "
        "component via AffineConstraints) instead of fully clamped "
        "for compression tests — clamped BCs create stress "
        "concentrations that worsen element inversion. Signal: "
        "DataOut shows stress concentrations of >10x material "
        "yield at clamped corners; the affected cells fail the "
        "det(F)>0 check first as load grows.",
        "[API] AD (step-72) avoids manual tangent derivation: "
        "Differentiation::AD::EnergyFunctional. Manual differentiation "
        "is the most common source of K_geo / K_mat sign errors. "
        "Signal: Newton converges with manual tangent but to a "
        "WRONG solution (off by 2x from AD-tangent reference); "
        "VectorTools::integrate_difference between manual and AD "
        "solutions is O(1).",
        "[API] MappingQEulerian visualises deformed configuration. "
        "Without it, DataOut shows the reference configuration "
        "with the displacement field overlaid as colour — "
        "misleading for large-deformation problems. Signal: "
        "DataOut.build_patches() called without "
        "DataOut::set_mapping(MappingQEulerian) writes a .vtu "
        "whose geometry is the undeformed reference triangulation; "
        "the displacement field appears only as colour overlay, "
        "not as actual node motion, even when "
        "solution.linfty_norm() is comparable to the domain size.",
        "[Syntax] GridGenerator::subdivided_hyper_rectangle "
        "defaults ALL faces to boundary_id=0 (colorize=false). "
        "Pass colorize=true to get distinct IDs (0-5 in 3D: "
        "left=0, right=1, bottom=2, top=3, front=4, back=5). "
        "Without this, AffineConstraints applied to boundary_id=0 "
        "clamps ALL faces. Signal: tria.get_boundary_ids() "
        "returns `{0}` (single id) instead of the expected set; "
        "DataOut shows displacement field zero everywhere because "
        "the entire boundary is over-constrained.",
        "[Numerical] For displacement-controlled nonlinear "
        "problems: use INCREMENTAL CONSTRAINTS. Set inhomogeneous "
        "Dirichlet value for the FIRST Newton iteration of each "
        "load step, then switch to homogeneous (zero increment) "
        "for subsequent iterations. Do NOT set boundary DOFs "
        "directly in the solution vector — this concentrates "
        "strain in boundary elements and Newton diverges. Signal: "
        "SolverControl reports residual.l2_norm() dropping "
        "normally on iteration 1 of each load step then jumping "
        "back up by 10x on iteration 2 — caused by direct "
        "DOF-setting in the BlockVector bypassing the "
        "AffineConstraints object on subsequent iterations.",
        "[Syntax] For FESystem (vector-valued), the trap is the "
        "CONTAINER SHAPE, not the missing extractor. Both of these "
        "are correct and give identical, complete data: "
        "fe_values[FEValuesExtractors::Vector(0)]"
        ".get_function_gradients(v, std::vector<Tensor<2, dim>>) "
        "and the plain "
        "fe_values.get_function_gradients(v, "
        "std::vector<std::vector<Tensor<1, dim>>>) with the inner "
        "vector sized n_components. Verified on deal.II 9.8.0-pre "
        "by interpolating u = (3x, 7y) onto "
        "FESystem(FE_Q(1), 2): the extractor form returns "
        "[3,0; 0,7] and the plain form returns comp0 = (3,0), "
        "comp1 = (0,7) — the plain call does NOT drop components, "
        "which is what this entry used to claim. What DOES go "
        "wrong is passing a SCALAR-shaped "
        "std::vector<Tensor<1, dim>> to the plain call on a "
        "vector-valued element: the dimension check is an Assert, "
        "compiled out on this Release build, so the call returns "
        "normally with garbage — the same experiment yielded "
        "(3, 7), which is neither component's gradient. Signal: a "
        "displacement-gradient field whose entries are a mixture "
        "of different components' derivatives — off-diagonal "
        "strains that look plausible but do not satisfy "
        "eps_xy == eps_yx, and no exception anywhere. Prefer the "
        "extractor form; it makes the shape mismatch a compile "
        "error instead of silent garbage.",
        "[Numerical] For < 50k DoFs, use SparseDirectUMFPACK "
        "instead of SolverCG + PreconditionSSOR. Direct solvers "
        "are more robust for nonlinear problems and avoid "
        "iterative solver tuning issues at small problem size. "
        "Signal: not a solver complaint — deal.II cannot report a CG "
        "'breakdown' at all (see the essentials block). On an "
        "indefinite tangent during line search CG either throws "
        "SolverControl::NoConvergence ('Iterative method reported "
        "convergence failure in step <N>. The residual in the last "
        "step was <R>.', with R grown or nan) or converges to the "
        "wrong answer. SparseDirectUMFPACK on the same matrix returns "
        "a machine-precision residual and is the cross-check; it also "
        "fails LOUDLY on a genuinely singular matrix, which CG does "
        "not.",
    ],
}
