"""Load-bearing deal.II knowledge served BEFORE anything physics-specific.

Why this module exists
----------------------
The consumer of this catalog is a small model that gets one shot at
writing a deal.II program.  Everything in here is a thing that makes
the difference between "compiles and computes the right answer" and
"segfaults with no message" — and none of it is discoverable from the
physics-specific sections, because it is the same for every physics.

``generators.get_knowledge()`` prepends this block to every per-physics
knowledge dict that has pitfalls, so it arrives without the caller
having to know a special ``topic=`` string.

Content rules for this module:
  * complete runnable programs, never fragments;
  * REQUIRED and OPTIONAL marked explicitly, in execution order;
  * every quoted diagnostic is a string deal.II itself prints — the
    library's own text, not a paraphrase.
"""

from __future__ import annotations


# ── 1. The smallest COMPLETE deal.II program that solves something ──
#
# Kept literal and complete on purpose: a weak model asked for "a
# Poisson solver" should be able to copy this, change the marked
# lines, and get a running program. Nothing here is pseudo-code.

MINIMAL_COMPLETE_PROGRAM = r"""
// ---------------------------------------------------------------
// FILE: main.cc   — complete, compiles as-is, solves -Lap(u)=1 on
// the unit square with u=0 on the whole boundary and writes a VTU.
// Change ONLY the lines marked [PROBLEM] to solve a different one.
// ---------------------------------------------------------------
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

using namespace dealii;

int main()
{
  const unsigned int dim = 2;                       // [PROBLEM] 2 or 3

  // 1. MESH  (REQUIRED). colorize=true gives each face its own
  //    boundary_id; with the default false EVERY face gets id 0.
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0., 1., /*colorize=*/true);
  tria.refine_global(4);                            // [PROBLEM] resolution

  // 2. ELEMENT + DOFS  (REQUIRED, and in this order: refine first,
  //    distribute afterwards).
  FE_Q<dim>       fe(1);                            // [PROBLEM] degree
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  // 3. CONSTRAINTS  (REQUIRED). ONE AffineConstraints object carries
  //    BOTH hanging nodes and Dirichlet values. close() is required.
  //    With colorize=true the cube's faces have SEPARATE ids, so a
  //    single interpolate_boundary_values(..., 0, ...) would clamp
  //    ONLY the x=0 face and leave the other three as (natural)
  //    homogeneous Neumann. Loop over the ids you actually want.
  AffineConstraints<double> constraints;
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  for (const types::boundary_id id : tria.get_boundary_ids())   // [PROBLEM]
    VectorTools::interpolate_boundary_values(                   // BCs
        dof_handler, id, Functions::ZeroFunction<dim>(), constraints);
  constraints.close();

  // 4. SPARSITY + ALLOCATION  (REQUIRED).
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints,
                                  /*keep_constrained_dofs=*/false);
  SparsityPattern sparsity;
  sparsity.copy_from(dsp);
  SparseMatrix<double> A(sparsity);
  Vector<double>       u(dof_handler.n_dofs()), b(dof_handler.n_dofs());

  // 5. ASSEMBLY  (REQUIRED). reinit(cell) every cell, and let the
  //    constraints object write into the global system.
  const QGauss<dim> quadrature(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_A(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_b(dofs_per_cell);
  std::vector<types::global_dof_index> local_dofs(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_A = 0.;
      cell_b = 0.;
      fe_values.reinit(cell);
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_A(i, j) += fe_values.shape_grad(i, q) *   // [PROBLEM]
                              fe_values.shape_grad(j, q) *   // bilinear
                              fe_values.JxW(q);              // form
            cell_b(i) += 1.0 *                               // [PROBLEM]
                         fe_values.shape_value(i, q) *       // source
                         fe_values.JxW(q);
          }
      cell->get_dof_indices(local_dofs);
      constraints.distribute_local_to_global(cell_A, cell_b,
                                             local_dofs, A, b);
    }

  // 6. SOLVE  (REQUIRED). Tolerance RELATIVE to ||b||: an absolute
  //    1e-12 is unreachable noise on a large problem.
  SolverControl            control(2000, 1e-10 * b.l2_norm());
  SolverCG<Vector<double>> solver(control);
  PreconditionSSOR<SparseMatrix<double>> prec;
  prec.initialize(A);
  solver.solve(A, u, b, prec);
  constraints.distribute(u);     // REQUIRED: writes the constrained
                                 // values back into the solution.
  std::cout << "dofs=" << dof_handler.n_dofs()
            << "  cg_steps=" << control.last_step()
            << "  max|u|=" << u.linfty_norm() << std::endl;

  // 7. OUTPUT  (OPTIONAL, but build_patches() is REQUIRED before
  //    write_vtu() — without it the file has no data).
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(u, "u");
  data_out.build_patches();
  std::ofstream out("solution.vtu");
  data_out.write_vtu(out);
  return 0;
}
"""


MINIMAL_COMPLETE_CMAKE = r"""
# FILE: CMakeLists.txt  — complete, next to main.cc
cmake_minimum_required(VERSION 3.13.4)
find_package(deal.II 9.5.0 REQUIRED HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR})
deal_ii_initialize_cached_variables()   # REQUIRED, and AFTER find_package
project(my_simulation)                  # REQUIRED, and AFTER the line above
add_executable(my_simulation main.cc)
deal_ii_setup_target(my_simulation)     # REQUIRED: adds includes + libs

# Build and run:
#   cmake -S . -B build && cmake --build build -j && ./build/my_simulation
"""


CRITICAL_KNOWLEDGE: dict = {
    "_read_this_first": (
        "Four things decide whether a deal.II program works. In order: "
        "(1) find the install and read its configuration — half the "
        "diagnostics in this catalog only exist on a Debug build, and "
        "half the features only exist if the corresponding "
        "DEAL_II_WITH_* flag was ON at library build time; "
        "(2) follow the REQUIRED call order below — deal.II does not "
        "check it for you in a Release build; "
        "(3) expect SILENCE, not exceptions, when you get it wrong; "
        "(4) match diagnostics against the strings deal.II really "
        "prints, listed at the end of this block."
    ),

    "step_0_locate_the_install": {
        "why": (
            "Every version- and feature-dependent statement in this "
            "catalog has to be checked against the install you are "
            "actually compiling against. Never assume."
        ),
        "how": [
            "The install root is whatever CMake's "
            "find_package(deal.II) resolves. Try, in order: the "
            "DEAL_II_DIR environment variable; $CONDA_PREFIX; the "
            "build directory of a source checkout (that directory "
            "contains CMakeCache.txt and lib/cmake/deal.II/).",
            "Version: read DEAL_II_PACKAGE_VERSION from "
            "$DEAL_II_DIR/include/deal.II/base/config.h. Do not trust "
            "a version written in any catalog, including this one.",
            "Optional features: the SAME config.h lists every optional "
            "dependency. A line '#define DEAL_II_WITH_P4EST' means ON; "
            "a line '/* #undef DEAL_II_WITH_P4EST */' means OFF. Grep "
            "for the literal '/* #undef DEAL_II_WITH_' to get the full "
            "OFF list in one shot.",
            "Build type: grep CMAKE_BUILD_TYPE in "
            "$DEAL_II_DIR/CMakeCache.txt, or list $DEAL_II_DIR/lib — "
            "libdeal_II.so is the Release library, libdeal_II.g.so is "
            "the Debug one. An install may ship one, the other, or "
            "both.",
        ],
        "warning": (
            "Checking whether a HEADER includes is NOT a feature "
            "probe and gives the wrong answer on a source build. A "
            "source install ships every header regardless of "
            "configuration; the bodies sit behind "
            "'#ifdef DEAL_II_WITH_<FEATURE>'. Verified on a source "
            "build with MPI/PETSc/SLEPc/Trilinos/p4est all OFF: "
            "'#include <deal.II/lac/slepc_solver.h>' compiles and "
            "links cleanly and the failure only appears when you name "
            "a class (\"'dealii::SLEPcWrappers' has not been "
            "declared\"); '#include <deal.II/distributed/tria.h>' also "
            "compiles, and parallel::distributed::Triangulation fails "
            "at COMPILE time with 'use of deleted function'; "
            "Utilities::MPI::MPI_InitFinalize constructs happily and "
            "reports n_mpi_processes == 1. Only config.h is "
            "authoritative."
        ),
    },

    "step_1_debug_vs_release_decides_what_you_will_see": {
        "rule": (
            "deal.II's internal sanity checks come in two kinds and "
            "they behave completely differently:\n"
            "  Assert(cond, Exc...)      — ACTIVE only in a Debug "
            "build. When it fires it ABORTS the process (SIGABRT, "
            "exit code 134) after printing a report; it does NOT "
            "throw, so try/catch does not help.\n"
            "  AssertThrow(cond, Exc...) — ACTIVE in every build. "
            "When it fires it THROWS a catchable exception.\n"
            "Most of the argument-checking in deal.II is Assert. On a "
            "Release build those checks are compiled out and the same "
            "misuse instead returns silently with a wrong answer, or "
            "segfaults (exit code 139)."
        ),
        "consequence_for_this_catalog": (
            "A pitfall that says 'raises Exc...' is describing the "
            "DEBUG behaviour. On a Release build you get silence or a "
            "segfault instead. Both behaviours are stated separately "
            "wherever this catalog knows them. Check the build type "
            "before choosing which one to look for."
        ),
        "how_to_get_the_debug_diagnostics": [
            "Best: build (or install) deal.II with "
            "-DCMAKE_BUILD_TYPE=Debug (or DebugRelease, which builds "
            "both libraries) and compile your program against "
            "libdeal_II.g.so. Then EVERY Assert is live.",
            "If the install is Release-only, 'cmake "
            "-DCMAKE_BUILD_TYPE=Debug' on your OWN project does not "
            "help: deal.II prints '#  WARNING: ... CMAKE_BUILD_TYPE "
            "was forced to \"Release\"' and still compiles with "
            "-DNDEBUG.",
            "TRIAGE WORKAROUND on a Release-only install: adding "
            "-DDEBUG to YOUR translation unit (while still linking the "
            "Release library) re-activates the Asserts that live in "
            "deal.II HEADERS — templates and inline functions, so "
            "SparseMatrix, FEValues accessors, SolverCG, FEEvaluation. "
            "Verified: an out-of-pattern SparseMatrix::add() silently "
            "drops the value on Release and, with -DDEBUG added, aborts "
            "with the full 'You are trying to access the matrix entry "
            "with index <i,j>, but this entry does not exist in the "
            "sparsity pattern of this matrix.' report. It CANNOT "
            "re-activate Asserts compiled into the library "
            "(DoFHandler::distribute_dofs and most of source/*.cc); "
            "those misuses behave EXACTLY as on a plain Release build, "
            "which usually means a silent wrong answer rather than a "
            "crash — do not expect a segfault to mark the spot. "
            "Verified on the library-compiled "
            "Assert(..., ExcInvalidFEIndex) in source/dofs/"
            "dof_handler.cc: setting active_fe_index 1 and then calling "
            "distribute_dofs with a one-element FECollection returned "
            "rc=0 and a wrong n_dofs=9 both plain and with -DDEBUG, and "
            "only the real Debug library aborted (rc=134). "
            "IS IT SAFE? It is a mixed-mode build, so the question is "
            "fair. Checked two ways. (1) Numerically: a non-trivial "
            "adaptive 2D Poisson solver — hanging nodes, "
            "AffineConstraints, FE_Q(2), CG + SSOR, "
            "integrate_difference against a manufactured solution and "
            "KellyErrorEstimator, four adaptive cycles — was built both "
            "ways against the same Release library, and the cell "
            "counts, DoF counts, iteration counts, solution norms, L2 "
            "errors and estimator norms were BITWISE IDENTICAL to 17 "
            "significant digits. (2) Structurally: the one class whose "
            "construction is DEBUG-conditional, FEEvaluationData, "
            "declares its tracking members UNCONDITIONALLY and guards "
            "only their initialiser list, so sizeof does not change "
            "and there is no ABI break. The residual risk is confined "
            "to that spot: if a Release-compiled library constructed "
            "such an object and a -DDEBUG translation unit then read "
            "those flags, a spurious ExcNotInitialized would be "
            "possible. FEEvaluation is header-only in practice, so this "
            "was not observed. Use the trick to LOCATE a bug, then fix "
            "the bug — do not ship a mixed-mode build, and do not "
            "treat it as a substitute for a proper Debug library.",
        ],
    },

    "step_2_required_call_order": {
        "note": (
            "Every item marked REQUIRED must happen, in this order. "
            "Getting the order wrong is not reported on a Release "
            "build. Items marked OPTIONAL can be skipped."
        ),
        "sequence": [
            "REQUIRED 1. Build the mesh, and do ALL global refinement "
            "BEFORE distribute_dofs(). Calling distribute_dofs on a "
            "one-cell triangulation succeeds and gives a useless "
            "system.",
            "REQUIRED 2. dof_handler.distribute_dofs(fe).",
            "OPTIONAL  2b. DoFRenumbering::* — needed for block "
            "systems (component_wise) and for downstream orderings in "
            "DG transport; harmless otherwise.",
            "REQUIRED 3. Build ONE AffineConstraints object holding "
            "BOTH DoFTools::make_hanging_node_constraints AND the "
            "Dirichlet values, then call close() exactly once. Two "
            "separate constraints objects give an inconsistent "
            "assembly.",
            "REQUIRED 4. Build the sparsity pattern from the SAME "
            "constraints object, then copy_from into a SparsityPattern "
            "and reinit the matrix with it.",
            "REQUIRED 5. In the assembly loop: fe_values.reinit(cell) "
            "for EVERY cell, then "
            "constraints.distribute_local_to_global(...).",
            "REQUIRED 6. Solve, then constraints.distribute(solution). "
            "This writes the constrained values (inhomogeneous "
            "Dirichlet data, and the interpolated values at hanging "
            "nodes) back into the solution vector; the solver leaves "
            "those entries at zero. Verified on an adaptively refined "
            "square with u=const on the boundary: distribute() changed "
            "48 of 137 entries, by up to the full boundary value. If "
            "ALL your constraints are homogeneous the vector does not "
            "change — which is exactly why forgetting this call "
            "survives a homogeneous test case and then produces a "
            "wrong answer on the real one.",
            "OPTIONAL  7. DataOut: attach_dof_handler, "
            "add_data_vector, then build_patches() — build_patches is "
            "REQUIRED if you call write_vtu at all, otherwise the file "
            "contains no data.",
        ],
    },

    "step_3_silence_is_the_normal_failure_mode": [
        "The single most common deal.II failure is a program that "
        "compiles, runs, exits 0, and computes the wrong answer. "
        "Assume this is what a mistake looks like.",
        "Therefore: never use 'the solver converged' or 'no exception "
        "was raised' as evidence of correctness. Check a quantity that "
        "has a known value — a manufactured solution, a conserved "
        "quantity, a symmetry, or the same problem solved with a "
        "richer discretisation.",
        "Concrete instances verified on a Release build, all silent: "
        "writing a matrix entry outside the sparsity pattern DROPS the "
        "value; passing a scalar-shaped container to "
        "FEValues::get_function_gradients on a vector-valued element "
        "returns MIXED components as if they were one gradient; "
        "VectorTools::interpolate_boundary_values on a discontinuous "
        "element writes ZERO boundary values and returns normally; "
        "omitting hanging-node constraints leaves the matrix exactly "
        "symmetric and the solver perfectly healthy while the answer "
        "stops converging.",
    ],

    "step_4_signals_dealii_can_actually_print": {
        "note": (
            "Match against these. Strings that are NOT in this list "
            "and are not printed by your own code will never appear, "
            "no matter how plausible they sound."
        ),
        "iterative_solver_failure": (
            "All built-in Krylov solvers (SolverCG, SolverGMRES, "
            "SolverMinRes, SolverBicgstab, ...) end with "
            "AssertThrow(state == SolverControl::success, "
            "SolverControl::NoConvergence(...)) — so this one is "
            "ACTIVE IN RELEASE TOO and is a real, catchable "
            "C++ exception of type SolverControl::NoConvergence. Its "
            "text is: 'Iterative method reported convergence failure "
            "in step <N>. The residual in the last step was <R>.' "
            "followed by a paragraph about iteration budgets and "
            "non-invertible matrices. The exception object carries "
            "e.last_step and e.last_residual. There is NO class named "
            "ExcSolverFail."
        ),
        "there_is_no_cg_breakdown_message": (
            "'SolverCG reports breakdown' is not a thing deal.II can "
            "print. The only breakdown guards in solver_cg.h are "
            "Debug-only Assert(std::abs(...) != 0., ExcDivideByZero()) "
            "checks on p_dot_A_dot_p and on the previous r^T P r "
            "(previous_r_dot_preconditioner_dot_r), so they are compiled "
            "out on Release entirely, and even on Debug they need p^T A p "
            "(or the previous r^T P r) to be EXACTLY 0.0 — which in "
            "practice means a literally zero operator. A merely "
            "indefinite or singular matrix produces NaN instead: the "
            "residual goes to nan and the solver throws "
            "SolverControl::NoConvergence with 'The residual in the "
            "last step was nan.' On a Debug build the zero-operator "
            "case aborts with 'A piece of code is attempting a "
            "division by zero.' Diagnose indefiniteness directly (a "
            "few hundred power iterations on A and on sigma*I - A) or "
            "by cross-checking against SolverMinRes / "
            "SparseDirectUMFPACK; do not wait for CG to complain."
        ),
        "solver_iteration_log": (
            "SolverControl prints nothing by default. To see the "
            "history you must enable it on BOTH sides: "
            "control.log_history(true); control.log_result(true); and "
            "deallog.depth_console(2). The lines then look exactly "
            "like 'DEAL:cg::Starting value 1.23456' and "
            "'DEAL:cg::Convergence step 17 value 4.56789e-11'. "
            "Programmatically prefer control.last_step() and "
            "control.last_value()."
        ),
        "unimplemented_feature": (
            "DEAL_II_NOT_IMPLEMENTED() prints 'You are trying to use "
            "functionality in deal.II that is currently not "
            "implemented' and then ABORTS (exit code 134) in every "
            "build. It is not catchable — try/catch(const "
            "std::exception&) does not run. Guard the call, do not try "
            "to recover from it."
        ),
        "mesh_import": (
            "GridIn rejects unsupported Gmsh cell types with 'The "
            "Element Identifier <N> is not supported in the deal.II "
            "library when reading meshes in <dim> dimensions.' "
            "followed by the list of supported ELM-TYPEs. N is the "
            "GMSH element type number (9 = 6-node triangle, 10 = "
            "9-node quadrilateral), never a VTK cell code."
        ),
        "matrix_entry_outside_sparsity_pattern": (
            "DEBUG ONLY. 'You are trying to access the matrix entry "
            "with index <i,j>, but this entry does not exist in the "
            "sparsity pattern of this matrix.' — from "
            "Assert(..., ExcInvalidIndex(i,j)) in sparse_matrix.h, "
            "and it aborts. On Release the write is silently dropped."
        ),
        "user_code_messages_are_not_library_messages": (
            "deal.II has no Newton solver, no line search and no "
            "constitutive-model layer, so it can never print 'Newton "
            "step did not converge', 'det(F) <= 0 at quadrature "
            "point', or anything like them. If a pitfall mentions such "
            "a message, YOUR code has to raise it — the usual idiom is "
            "AssertThrow(cond, ExcMessage(\"...\")), which is active "
            "in Release as well. Write those guards; they are the only "
            "diagnostics you will get for problem-level errors."
        ),
    },

    "step_5_boundary_conditions_the_two_traps": [
        "TRAP 1 — every face has boundary_id 0 unless you ask "
        "otherwise. GridGenerator::hyper_cube, hyper_rectangle and "
        "subdivided_hyper_rectangle all take colorize=false by "
        "DEFAULT. Pass colorize=true to get per-face ids: 0:x=low, "
        "1:x=high, 2:y=low, 3:y=high, and in 3D 4:z=low, 5:z=high. "
        "Check with tria.get_boundary_ids() — note this is a "
        "Triangulation MEMBER function; there is no "
        "GridTools::get_boundary_ids. If it returns a single id {0} "
        "while your code keys Dirichlet loops on 1,2,3, those loops "
        "match nothing and those sides silently become homogeneous "
        "Neumann.",
        "TRAP 2 — VectorTools::interpolate_boundary_values only works "
        "on elements that have point support. Guard it with "
        "fe.has_support_points(). It is TRUE for FE_Q, FE_Q_Bubbles, "
        "FE_Q_iso_Q1, FE_SimplexP, FE_WedgeP, FE_PyramidP, FE_DGQ. It "
        "is FALSE for FE_Q_Hierarchical, FE_Bernstein, FE_Hermite, "
        "FE_DGP, FE_DGQLegendre, FE_DGQHermite, FE_RannacherTurek, "
        "FE_P1NC, and for the whole H(div)/H(curl) family "
        "(FE_RaviartThomas, FE_BDM, FE_ABF, FE_Nedelec, ...), whose "
        "DoFs are moments rather than point values. Calling "
        "interpolate_boundary_values on a hypercube-continuous element "
        "without support points SEGFAULTS (exit 139) on a Release "
        "build — verified for FE_Q_Hierarchical and FE_Bernstein, both "
        "of which this catalog recommends elsewhere. Use "
        "VectorTools::project_boundary_values(dof_handler, "
        "{{id, &function}}, QGauss<dim-1>(fe.degree + 2), "
        "boundary_values) instead; it works for all of them. On a "
        "DISCONTINUOUS element the same call is merely a no-op: it "
        "returns with an EMPTY map and DG boundary conditions have to "
        "be imposed weakly through the numerical flux anyway.",
    ],

    "step_6_the_constructor_argument_is_not_always_the_degree": (
        "fe.degree is the maximal polynomial degree the space "
        "contains, which for several families is NOT the number you "
        "passed to the constructor. Verified by instantiation: "
        "FE_Q_Bubbles<dim>(2).degree == 3 (the interior bubble raises "
        "it); FE_RaviartThomas<dim>(1).degree == 2, FE_BDM<dim>(1)"
        ".degree == 2, FE_ABF<dim>(0).degree == 2, "
        "FE_Nedelec<dim>(0).degree == 1 — for these the constructor "
        "argument is the family INDEX k, not the degree; "
        "FE_RannacherTurek<dim>(0).degree == 2. Since the standard "
        "idiom for choosing a quadrature rule is QGauss<dim>"
        "(fe.degree + 1), read fe.degree back from the element instead "
        "of reusing the constructor argument, otherwise you "
        "under-integrate."
    ),

    "step_6b_simplex_wedge_and_pyramid_meshes": {
        "when": (
            "Any mesh that is not made of quadrilaterals/hexahedra — "
            "an unstructured Gmsh/TetGen mesh, or "
            "GridGenerator::subdivided_hyper_cube_with_simplices / "
            "subdivided_hyper_rectangle_with_simplices, or a "
            "hex mesh converted with "
            "GridGenerator::convert_hypercube_to_simplex_mesh."
        ),
        "the_three_things_that_must_change_together": [
            "ELEMENT: FE_SimplexP(p) (continuous) or FE_SimplexDGP(p) "
            "(discontinuous) instead of FE_Q / FE_DGQ. FE_WedgeP and "
            "FE_PyramidP exist for the 3D transition cells. All of "
            "them instantiate and carry point support, so "
            "interpolate_boundary_values works on them.",
            "QUADRATURE: QGaussSimplex<dim>(n) instead of "
            "QGauss<dim>(n), and QGaussSimplex<dim-1> for faces.",
            "MAPPING: MappingFE<dim>(FE_SimplexP<dim>(1)) — and it "
            "must be passed EXPLICITLY as the first argument to "
            "FEValues, FEFaceValues, VectorTools::* , "
            "KellyErrorEstimator::estimate and DataOut::build_patches. "
            "The default MappingQ1 is a hypercube mapping and is "
            "silently wrong on a simplex.",
        ],
        "signal": (
            "ONE CHECK CATCHES ALL THREE MISTAKES: sum fe_values.JxW(q) "
            "over every cell and every quadrature point and compare it "
            "with the exact volume of the domain. On a correct setup it "
            "matches to round-off. The test is 'differs from the domain "
            "volume' — do NOT look for a specific wrong value or a "
            "specific sign, because both depend on the dimension and "
            "the mesh. Verified on a unit square and a unit cube, "
            "changing one ingredient at a time: "
            "using QGauss instead of QGaussSimplex made the total "
            "SIX times too large in 3D (the ratio of the reference "
            "cube's volume to the reference tetrahedron's 1/6, so it "
            "is TWO times too large in 2D) and the solve then "
            "CONVERGED to a silently wrong answer, nearly twice the "
            "correct peak value — this is the dangerous case; "
            "omitting MappingFE and relying on the default hypercube "
            "mapping gave a total of about two thirds of the true area "
            "in 2D and a NEGATIVE total in 3D, and the solver then "
            "failed with SolverControl::NoConvergence. Nothing was "
            "raised in any of the wrong cases, in either build."
        ),
        "quadrature_weights_are_the_giveaway": (
            "If you want an even cheaper check, sum the weights of the "
            "quadrature rule alone: QGaussSimplex<3> weights sum to "
            "1/6 (the volume of the reference tetrahedron) and "
            "QGauss<3> weights sum to 1 (the reference cube). A rule "
            "whose weights sum to 1 has no business on a tet mesh."
        ),
    },

    "step_6c_mesh_generator_gotchas": {
        "_note": (
            "Every GridGenerator function this catalog names was called "
            "and its mesh integrated (summed JxW against the known "
            "volume). They all exist and run. The two below are the "
            "ones that produced a WRONG mesh without complaining."
        ),
        "general_cell_vertex_order": (
            "[Mesh] GridGenerator::general_cell(tria, vertices, colorize) "
            "expects the vertices in deal.II's LEXICOGRAPHIC vertex "
            "order, not in cyclic (counter-clockwise) order. In 2D "
            "that is (x_lo,y_lo), (x_hi,y_lo), (x_lo,y_hi), "
            "(x_hi,y_hi) — the last two are SWAPPED relative to how "
            "one naturally traces a quadrilateral. Verified on the "
            "same four corner points of a rectangle (Release build, "
            "deal.II 9.7.1 and 9.8.0-pre): the cyclic ordering produced "
            "a twisted cell whose summed JxW is exactly 0, with no "
            "complaint at all, while the lexicographic ordering produced "
            "the correct positive area. For other quadrilaterals the "
            "cyclic ordering gave a negative or a wrong positive area "
            "(-1 and 0.5 where the true areas are 2 and 1.5). A Debug "
            "build stops in general_cell's own volume check ('The "
            "volume of the cell is not greater than zero. This could be "
            "due to the wrong ordering of the vertices.') when that "
            "volume is zero or negative; a wrong positive volume passes "
            "it. Signal: integrate the mesh (sum fe_values.JxW(q)) "
            "immediately after creating it and require the result to "
            "be positive and equal to the area/volume you intended. "
            "The same ordering convention applies to the corner "
            "vectors of parallelepiped / subdivided_parallelepiped."
        ),
        "colorize_is_off_by_default_everywhere": (
            "hyper_cube, hyper_rectangle, subdivided_hyper_rectangle, "
            "enclosed_hyper_cube, hyper_shell, parallelepiped, "
            "general_cell and friends all take colorize as a "
            "DEFAULT-FALSE trailing argument. Pass true whenever you "
            "intend to distinguish faces; see the boundary-condition "
            "trap above."
        ),
        "curved_domains_need_a_manifold_AND_a_mapping": (
            "[Mesh] On a curved domain TWO things are needed and they do "
            "different jobs. The MANIFOLD (SphericalManifold, "
            "CylindricalManifold, TransfiniteInterpolationManifold, an "
            "OpenCASCADE surface) tells refinement where to place NEW "
            "vertices; the MAPPING DEGREE (MappingQ<dim>(p)) tells the "
            "quadrature how curved each cell's geometry is. Verified by "
            "integrating hyper_ball's volume against the analytic value "
            "in 2D and 3D over three refinement levels and two mapping "
            "degrees: with NO manifold attached the computed volume is "
            "stuck at the coarse straight-edged value and REFINEMENT "
            "DOES NOT IMPROVE IT - it changes only at round-off level "
            "(relative changes of order 1e-14 in 2D and 1e-13 in 3D), "
            "because new vertices are placed by straight-line "
            "interpolation so the mesh never approaches the circle or "
            "the sphere - and a higher mapping degree does not help "
            "either, since there is no curvature for it to represent; "
            "with the manifold attached but MappingQ(1) the error falls "
            "under refinement but slowly; with the manifold attached "
            "and MappingQ(3) even the UNREFINED coarse mesh is already "
            "accurate to about 2e-4 relative in 2D (4e-3 in 3D), and "
            "refinement reduces the error further, though not to "
            "round-off within these levels (still about 5e-8 in 2D and "
            "3e-7 in 3D after two refinements). GridGenerator's curved generators "
            "(hyper_ball, hyper_shell, cylinder, torus, ...) attach the "
            "right manifold for you - so the usual ways to lose it are "
            "Triangulation::reset_all_manifolds(), building the mesh by "
            "merging or flattening (neither carries manifolds over), "
            "and importing from a file, where there is no curvature "
            "information at all. Signal: refine once and integrate "
            "again. If the summed JxW changes only at round-off level "
            "(relative change below about 1e-12) on a domain you "
            "believe is curved, no manifold is attached. "
            "Nothing is raised."
        ),
        "integrate_the_mesh_before_you_trust_it": (
            "Summing fe_values.JxW(q) over the whole mesh and "
            "comparing against the domain volume is the single "
            "cheapest correctness check in deal.II. Verified to catch "
            "all four of: a wrong-handed cell from general_cell, a "
            "simplex mesh with a hypercube mapping, a hypercube "
            "quadrature rule on a simplex mesh, and a curved domain "
            "with no manifold attached. None of those raise anything "
            "in a Release build; in a Debug build general_cell stops "
            "when the cell's volume is zero or negative."
        ),
    },

    "step_6d_which_preconditioner_may_be_paired_with_which_solver": {
        "_note": (
            "Every built-in solver and preconditioner this catalog names "
            "was instantiated and run on a real 2D Poisson system, and "
            "the answer cross-checked against SparseDirectUMFPACK. The "
            "combinations that FAIL are the point of this section: none "
            "of them is rejected at compile time, and none of them "
            "prints a warning."
        ),
        "works_with_SolverCG": (
            "SolverCG requires the preconditioner to be SYMMETRIC "
            "positive definite, not merely convergent. Verified working "
            "(each agreeing with the direct solve to ~1e-11): "
            "PreconditionIdentity, PreconditionJacobi, "
            "PreconditionSSOR, PreconditionChebyshev, SparseILU, "
            "SparseMIC, PreconditionBlockJacobi. "
            "PreconditionChebyshev needed the fewest iterations of the "
            "cheap ones and is the standard choice inside multigrid."
        ),
        "DOES_NOT_work_with_SolverCG": (
            "PreconditionSOR — the ONE-SIDED sweep — is not symmetric, "
            "so it violates CG's assumption. Verified on two different "
            "SPD systems (a continuous FE_Q mass+stiffness system and a "
            "discontinuous FE_DGQ one): SolverCG ran to its 2000-"
            "iteration limit and threw SolverControl::NoConvergence "
            "with the residual stalled, while SolverGMRES with the very "
            "same preconditioner converged in a few tens of iterations "
            "on the first and a handful on the second. Nothing warns "
            "you. The SS in SSOR is what makes it CG-legal; if you want "
            "a one-sided sweep, use GMRES. The same argument applies to "
            "PreconditionBlockSOR versus PreconditionBlockSSOR."
        ),
        "PreconditionBlock_family_has_a_precondition": (
            "[Numerical] PreconditionBlockSSOR / BlockSOR / BlockJacobi invert the "
            "DIAGONAL BLOCKS of the given block_size. That is only "
            "meaningful when those blocks are real sub-problems — the "
            "intended case is a DISCONTINUOUS element with "
            "block_size = fe.n_dofs_per_cell(), where each block IS one "
            "cell. Verified: on an FE_DGQ system that pairing solved in "
            "1-2 iterations, while on the continuous FE_Q system of the "
            "same size it took an order of magnitude more, and on a "
            "PURE-STIFFNESS continuous system with Dirichlet data it "
            "broke outright — exactly one of the 72 diagonal blocks was "
            "SINGULAR (a block consisting of constrained rows), "
            "initialize() returned WITHOUT complaint, and the resulting "
            "'preconditioner' left a relative residual ABOVE 1 after "
            "one application, with the solver going to NaN. "
            "Signal: apply the preconditioner once to the right-hand "
            "side and measure ||b - A*P(b)|| / ||b||. Anything at or "
            "above 1 means the preconditioner is worse than nothing; "
            "NaN means a singular block. Both are silent. If you want "
            "to find the bad block, extract each diagonal block and "
            "take its determinant — a singular one makes deal.II's own "
            "LAPACK path throw ExcSingular from "
            "LAPACKFullMatrix::compute_lu_factorization, which IS an "
            "AssertThrow and therefore fires in Release too."
        ),
        "solvers_that_need_care": (
            "SolverQMRS failed on a plain SPD Poisson system with "
            "PreconditionSSOR where CG, GMRES, FGMRES, Bicgstab and "
            "MinRes all converged. Measured on deal.II 9.7.1 and "
            "9.8.0-pre: once the residual stagnates, its breakdown test "
            "(|q.Aq| below the default breakdown_threshold of 1e-16) "
            "restarts the iteration, and the restarts bypass the "
            "SolverControl step limit, so solve() never returns and "
            "never throws. With deallog at its default depth it prints "
            "nothing at all; with depth_console(2) it prints 'Failure "
            "step N value ...' then 'Restart step N+1', forever. Run it "
            "under an external timeout and treat it as a specialist "
            "tool, not a drop-in. SolverRichardson converged but needed an order "
            "of magnitude more iterations than the Krylov methods; it "
            "is a smoother, not a solver. SolverMinRes and "
            "SolverBicgstab both worked and are the right fallbacks for "
            "an indefinite or non-symmetric system respectively."
        ),
        "direct_solver": (
            "SparseDirectUMFPACK requires DEAL_II_WITH_UMFPACK=ON. "
            "Usage is initialize(A) then vmult(x, b) — it is applied "
            "like a preconditioner, not through a SolverXxx. It "
            "reproduced the reference answer exactly and is the right "
            "cross-check whenever an iterative solver's result is in "
            "doubt. The Trilinos (Amesos) and PETSc (MUMPS) direct "
            "solvers named elsewhere in this catalog need "
            "DEAL_II_WITH_TRILINOS / _PETSC; check config.h first."
        ),
    },

    "step_7_complete_runnable_example": {
        "main_cc": MINIMAL_COMPLETE_PROGRAM,
        "CMakeLists_txt": MINIMAL_COMPLETE_CMAKE,
        "cmake_order_constraints": (
            "deal_ii_initialize_cached_variables() must come AFTER "
            "find_package(deal.II) and BEFORE project(); "
            "deal_ii_setup_target(<target>) must come after "
            "add_executable/add_library. Violating either produces a "
            "confusing CMake error rather than a compile error."
        ),
    },
}
