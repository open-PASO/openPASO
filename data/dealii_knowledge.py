"""
Comprehensive deal.II knowledge catalogue.

Based on the step tutorials (step-1 through step-97), covering all physics
from Poisson to compressible Euler, Maxwell, FSI, contact, topology optimization.
deal.II won the 2025 SIAM/ACM CSE Prize.

IMPORTANT about the tutorial count: the step numbering HAS GAPS, so the
highest number is not the number of tutorials. Counted on deal.II 9.7.0
and 9.7.1 (and on a 9.8.0-pre checkout of March 2026) there are 88
examples/step-* directories while the numbering runs to step-97; the
9.8.0 release has 91, numbered up to step-104 (step-98, step-100 and
step-104 were added). In particular step-73 does NOT exist upstream
(the AD tutorials are step-71 and step-72); the "step-73 (AD-assisted /
elasticity)" references that used to appear below have been removed.
The numbers that do NOT exist are 52, 73, 80, 84, 88, 91, 92, 94, 96 in
9.7.x, and in the 9.8.0 examples/ directory also 99, 101, 102 and 103 —
every step-* reference in this catalog was checked against that directory and
the two dangling ones (step-52, step-80) were corrected.
Enumerate examples/step-* in the source checkout you actually have -
a binary package prefix ships no tutorials at all.
"""

DEALII_KNOWLEDGE = {
    # ═══════════════════════════════════════════════════════════════════════
    # OVERVIEW
    # ═══════════════════════════════════════════════════════════════════════
    "overview": {
        "description": ("deal.II is a C++ FEM library with ~88 tutorial "
                        "programs (the numbering runs past step-90 but has "
                        "gaps, so the highest number is not the count) and "
                        "40+ element types"),
        "version": ("Read the real version from the install you are "
                    "compiling against: DEAL_II_PACKAGE_VERSION in "
                    "$DEAL_II_DIR/include/deal.II/base/config.h. Do "
                    "not trust a version written in any catalog, "
                    "including this one - the API moved a lot across "
                    "the 9.x line and several entries here carry "
                    "version gates. Read the BUILD TYPE from the same "
                    "install too (CMAKE_BUILD_TYPE in CMakeCache.txt, "
                    "or whether lib/ holds libdeal_II.so, "
                    "libdeal_II.g.so, or both): it decides whether "
                    "deal.II's Assert-based diagnostics exist at all."),
        "language": "C++17/20",
        "build_system": "CMake: find_package(deal.II 9.5 REQUIRED)",
        "compilation": "cmake -Bbuild . && cmake --build build -j$(nproc)",
        "execution": "./build/my_simulation (serial) or mpirun -np N ./build/my_simulation (parallel)",
        "output": "VTU via DataOut::write_vtu(), PVTU for parallel, PVD for time series",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # ALL TUTORIALS BY PHYSICS
    # ═══════════════════════════════════════════════════════════════════════
    "tutorials": {
        "description": ("step-* tutorials - each a complete, working "
                        "program teaching one concept. The list below is "
                        "organised by physics; it is not exhaustive and "
                        "the numbering has gaps."),

        "elliptic_pdes": {
            "Poisson/Laplace": {
                "basic": ["step-3 (first program)", "step-4 (dim-independent)", "step-5 (variable coeff)"],
                "adaptive": ["step-6 (AMR + Kelly)", "step-14 (DWR goal-oriented)"],
                "parallel": ["step-40 (p4est distributed)", "step-50 (parallel GMG)"],
                "matrix_free": ["step-37 (MF + GMG)", "step-59 (MF DG)", "step-64 (GPU, Kokkos)"],
                "hp": ["step-27 (hp-adaptive)", "step-75 (hp + MF + parallel)"],
            },
            "Helmholtz": ["step-7 (real)", "step-29 (complex-valued, ultrasound)"],
            "biharmonic": ["step-47 (C0 interior penalty)", "step-82 (LDG)"],
            "surface_PDE": ["step-38 (Laplace-Beltrami)", "step-90 (TraceFEM)"],
            "minimal_surface": ["step-15 (nonlinear Newton)", "step-72 (AD Newton)"],
        },

        "structural_mechanics": {
            "linear_elasticity": ["step-8 (basic)", "step-17 (MPI parallel)", "step-18 (quasi-static)"],
            "hyperelasticity": ["step-44 (3-field Neo-Hookean)", "step-71 (AD/SD for (magneto-)hyperelastic constitutive laws)"],
            "contact": ["step-41 (obstacle problem)", "step-42 (3D elasto-plastic contact, parallel)"],
            "topology_optimization": ["step-79 (SIMP method, density-based)"],
        },

        "fluid_mechanics": {
            "Stokes": ["step-22 (block precond)", "step-55 (MPI parallel)", "step-56 (GMG on the velocity block, SOR smoother)"],
            "Navier_Stokes": ["step-35 (projection method)", "step-57 (stationary Newton + AMR)"],
            "compressible_Euler": ["step-33 (implicit Newton)", "step-67 (explicit DG, MF, SIMD)",
                                  "step-69 (first-order)", "step-76 (optimized 67)"],
            "Boussinesq": ["step-31 (convection)", "step-32 (parallel)"],
            "two_phase_porous": ["step-21 (two-phase)", "step-43 (adaptive splitting)"],
        },

        "heat_and_diffusion": {
            "heat_transient": ["step-26 (AMR in time)", "step-86 (PETSc TS)"],
            "advection_diffusion": ["step-9 (SUPG)", "step-12 (DG upwind)", "step-30 (anisotropic AMR)",
                                   "step-51 (HDG)", "step-63 (GMG block smoothers)"],
        },

        "wave_and_dynamics": {
            "acoustic_wave": ["step-23 (theta scheme; Crank-Nicolson at theta=1/2, which equals Newmark beta=1/4, gamma=1/2)", "step-24 (absorbing BC)"],
            "soliton": ["step-25 (Sine-Gordon)", "step-48 (MF explicit leap-frog)"],
            "elastic_wave": ["step-62 (frequency domain, PML)"],
            "Schrodinger": ["step-58 (nonlinear, operator splitting)"],
        },

        "electromagnetics": {
            "Maxwell": ["step-81 (curl-curl, PML, Nédélec)", "step-97 (curl-curl)"],
        },

        "multiphysics_coupling": {
            "Stokes_elasticity": ["step-46 (subdomain coupling via FE_Nothing)"],
            "Stokes_temperature": ["step-31 (Boussinesq)", "step-32 (parallel)"],
            # step-80 does NOT exist upstream — checked against the 9.7.x
            # and 9.8.0 examples/ directories, where the missing numbers
            # are 52, 73, 80, 84, 88, 91, 92, 94 and 96 (and in 9.8.0
            # also 99, 101, 102 and 103). The
            # non-matching-grid tutorials are step-89 (mortaring) and
            # step-90 (TraceFEM); the immersed / distributed-Lagrange
            # route is step-60.
            "FSI": ["step-60 (immersed DLM)", "step-70 (particles)",
                    "step-89 (non-matching, mortaring)"],
            "particle_coupling": ["step-19 (PIC)", "step-68 (Stokes particle advection)"],
        },

        "advanced_techniques": {
            "periodic_BCs": ["step-45"],
            "eigenvalue": ["step-36 (SLEPc, Schrodinger)"],
            "neutron_diffusion": ["step-28 (multigroup)"],
            "BEM": ["step-34 (potential flow)"],
            "financial": ["step-78 (Black-Scholes options)"],
            "unfitted_CutFEM": ["step-85 (Nitsche)", "step-95 (MF CutFEM)"],
            "checkpoint_restart": ["step-83"],
            "non_matching_grids": ["step-89 (mortaring)"],
            # step-73 does not exist upstream (verified against
            # a 9.8 examples/ directory).
            "AD": ["step-71 (concepts)", "step-72 (energy functional)"],
            "SUNDIALS": ["step-77 (KINSOL nonlinear)"],
        },
    },

    # ═══════════════════════════════════════════════════════════════════════
    # ELEMENT TYPES (40+)
    # ═══════════════════════════════════════════════════════════════════════
    "element_types": {
        "H1_continuous": {
            "FE_Q(p)": "Standard Lagrange on quads/hexes, any order p, Gauss-Lobatto points",
            "FE_Q_Hierarchical(p)": "Hierarchical basis (efficient for hp-adaptivity)",
            "FE_Bernstein(p)": "Bernstein polynomial basis (positive, partition of unity)",
            "FE_Hermite(p)": ("Hermite interpolation of odd degree p, continuous in "
                              "derivatives up to order (p-1)/2 (C1 for p=3). deal.II "
                              "reports conforming_space == Conformity::H2, not H1, for "
                              "p >= 3, and Conformity::H1 for p = 1, which has only "
                              "vertex values (verified by instantiation); it is listed "
                              "under H1 here because H2 is contained in H1."),
            "FE_SimplexP(p)": "Lagrange on simplices (triangles/tetrahedra)",
            "FE_SimplexP_Bubbles(p)": "Simplex Lagrange + bubble enrichment",
        },
        "H1_enriched": {
            "_note": (
                "Strictly H1-conforming enrichments — safe to pick when the "
                "formulation requires H1 continuity."
            ),
            "FE_Q_Bubbles(p)": "Q + cell-interior bubble enrichment (bubble vanishes on cell boundary; H1 preserved). Caveat: condition number grows fast for p>3",
            "FE_Q_iso_Q1(p)": "Piecewise (bi-/tri-)linear functions on a macro-element of p^dim sub-cells; globally continuous so H1-conforming",
        },
        "nonconforming_and_qp_dg0": {
            "_note": (
                "These are NOT H1-conforming.  Do NOT pick them for an "
                "H1-conforming formulation — they are listed here only so "
                "the agent recognises them when reading existing decks."
            ),
            "FE_Q_DG0(p)": "Lagrange Qp PLUS cell-wise constants (Qp+DG0); the added piecewise-constant mode is discontinuous across element boundaries",
            "FE_RannacherTurek(0)": ("Classical first-order nonconforming element; continuity "
                                     "enforced only in the face-mean sense, not pointwise. deal.II "
                                     "reports Conformity::L2 (verified by instantiation). The constructor "
                                     "is FE_RannacherTurek<dim>(order = 0, n_face_support_points = 2) "
                                     "and only order 0 is implemented. has_support_points() is FALSE, "
                                     "so VectorTools::interpolate_boundary_values segfaults on it in a "
                                     "Release build — use project_boundary_values."),
        },
        "DG_discontinuous": {
            "FE_DGQ(p)": "Tensor-product DG, equidistant points",
            "FE_DGQLegendre(p)": "DG with Legendre basis (orthogonal, good for L2 projection)",
            "FE_DGQHermite(p)": "DG Hermite-like (optimal for matrix-free, sum factorization)",
            "FE_DGQArbitraryNodes(quadrature)": "DG_Q on a user-chosen node set (Gauss-Lobatto/Gauss/equispaced) for matrix-free / spectral-element style discretisations",
            "FE_DGP(p)": "Complete polynomial DG (fewer DOFs than DGQ for same order)",
            "FE_DGPMonomial(p)": "DG using the monomial polynomial basis rather than the standard nodal basis",
            "FE_DGPNonparametric(p)": "DG with a non-parametric mapping (polynomials defined in physical space)",
            "FE_SimplexDGP(p)": "DG on simplices",
        },
        "DG_vector_valued": {
            "FE_DGVector<PolynomialsType>": "Class template in fe_dg_vector.h that wraps a vector-valued polynomial space into a DG element. The three concrete instantiations below derive from it",
            "FE_DGRaviartThomas(k)": "DG element on the RT polynomial space (DG mixed methods)",
            "FE_DGNedelec(k)": "DG element on the Nédélec polynomial space (discontinuous H(curl)-type)",
            "FE_DGBDM(k)": "DG element on the Brezzi-Douglas-Marini polynomial space (discontinuous H(div)-type)",
        },
        "Hdiv_conforming": {
            "FE_RaviartThomas(k)": "Raviart-Thomas (normal continuous, for mixed Poisson/Darcy)",
            "FE_RaviartThomasNodal(k)": "RT with a nodal-DoF representation (alternative to the moment-based default)",
            "FE_RT_Bubbles(k)": "RT enriched with interior bubble functions for improved approximation order",
            "FE_BDM(k)": "Brezzi-Douglas-Marini (full polynomial H(div))",
            "FE_ABF(k)": "Arnold-Boffi-Falk",
            "FE_BernardiRaugel(1)": "Inf-sup stable with DGP(0) for Stokes",
        },
        "Hcurl_conforming": {
            "FE_Nedelec(k)": "Nédélec edge elements for Maxwell/electromagnetics",
            "FE_NedelecSZ(k)": "Schoeberl-Zaglmayr ordering variant",
            "FE_NedelecNodal(k)": "Nédélec with nodal-interpolation DoF setup (alternative to the default)",
        },
        "trace_and_face": {
            "FE_FaceQ(p)": "Q-polynomial face element for hybridised DG interface unknowns (HDG, step-51)",
            "FE_FaceP(p)": "P-polynomial face element, simplex analogue of FE_FaceQ",
            "FE_TraceQ(p)": "Trace of FE_Q on element faces (Lagrange-multiplier / HDG stabilisations)",
        },
        "pyramid_and_wedge_3d": {
            "FE_PyramidP(p)": "Continuous P element on pyramidal (square-base) 3D cells — hex<->tet transition",
            "FE_PyramidDGP(p)": "DG counterpart of FE_PyramidP",
            "FE_WedgeP(p)": "Continuous P element on wedge (triangular-prism) 3D cells — hex<->tet transition",
            "FE_WedgeDGP(p)": "DG counterpart of FE_WedgeP",
        },
        "special": {
            "FE_FaceQ(p)": "DOFs on faces only (for HDG trace systems, step-51)",
            "FE_Nothing": "Zero-DOF element (subdomain coupling in hp, step-46)",
            "FE_Enriched": "Enrichment wrapper for XFEM/GFEM (partition of unity)",
            "FE_P1NC": ("Nonconforming P1 (Crouzeix-Raviart analogue), 2D quads only. "
                        "NOT a class template: declared as "
                        "`class FE_P1NC : public FiniteElement<2, 2>`, so the "
                        "constructor is FE_P1NC() and FE_P1NC<2>() fails to compile "
                        "(verified by compiling both spellings). Conformity::L2."),
            "FESystem": "Combine multiple FE into vector/tensor systems",
            "hp::FECollection": "Collection of different FE for hp-adaptivity",
        },
        "internal_polynomial_bases": {
            "_note": (
                "Abstract polynomial base classes that the concrete elements "
                "above are templated on (FE_Q is an FE_Q_Base; FE_PyramidP is "
                "an FE_PyramidPoly; etc.).  No public stand-alone constructor "
                "— do NOT propose these as user-facing element choices."
            ),
            "FE_Poly": "Base class for polynomial-based scalar elements",
            "FE_PolyFace": "Face-only polynomial base class",
            "FE_PolyTensor": "Tensor-product polynomial base class",
            "FE_Q_Base": "Base of the FE_Q family (FE_Q, FE_Q_Hierarchical, FE_Q_Bubbles, ...)",
            "FE_SimplexPoly": "Base of the FE_SimplexP family",
            "FE_PyramidPoly": "Base of the FE_PyramidP / FE_PyramidDGP family",
            "FE_WedgePoly": "Base of the FE_WedgeP / FE_WedgeDGP family",
        },
    },

    # ═══════════════════════════════════════════════════════════════════════
    # MESH GENERATORS (30+)
    # ═══════════════════════════════════════════════════════════════════════
    "mesh_generators": {
        "basic": ["hyper_cube", "subdivided_hyper_cube", "hyper_rectangle", "subdivided_hyper_rectangle"],
        "domains": ["hyper_L", "hyper_ball", "hyper_shell", "half_hyper_shell", "quarter_hyper_shell",
                    "concentric_hyper_shells"],
        "3D": ["cylinder", "cylinder_shell", "truncated_cone", "pipe_junction"],
        "engineering": ["channel_with_cylinder (DFG benchmark)", "plate_with_a_hole",
                       "cheese (holes pattern)", "hyper_cube_with_cylindrical_hole", "hyper_cross"],
        "simplex": ["subdivided_hyper_cube_with_simplices", "subdivided_hyper_rectangle_with_simplices"],
        "manipulation": ["merge_triangulations", "extrude_triangulation", "replicate_triangulation"],
        "import": ["Gmsh (.msh)", "UCD (.ucd)", "VTK (.vtk)", "ExodusII", "ABAQUS (.inp)",
                   "OpenCASCADE (STEP/IGES, step-54)"],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # ADAPTIVE MESH REFINEMENT
    # ═══════════════════════════════════════════════════════════════════════
    "adaptive_refinement": {
        "error_estimators": {
            "Kelly": "Gradient jump-based (step-6, most AMR tutorials) — KellyErrorEstimator::estimate()",
            "DWR": "Dual-weighted residual, goal-oriented (step-14) — solves adjoint problem",
            "smoothness": "Fourier/Legendre coefficient decay for hp-adaptivity (step-27, 75)",
            "residual": "User-defined residual-based estimators",
        },
        "refinement_types": ["isotropic h", "anisotropic h (step-30)", "hp (step-27, 75)"],
        "strategies": "refine_and_coarsen_fixed_number/fraction, SolutionTransfer for interpolation",
        "parallel": ("p4est-backed distributed triangulation. REQUIRES "
                     "DEAL_II_WITH_MPI=ON and DEAL_II_WITH_P4EST=ON at "
                     "LIBRARY build time. If either is OFF the failure is "
                     "at COMPILE time (the constructor is `= delete`d, so "
                     "the message is \"use of deleted function ... "
                     "parallel::distributed::Triangulation ...\"), NOT a "
                     "runtime message - the older claim that it raises "
                     "'needs deal.II compiled with p4est' at runtime is "
                     "wrong. Serial and thread-parallel fallback: a plain "
                     "Triangulation<dim, spacedim>, which supports the same "
                     "adaptive-refinement API."),
        "pitfalls": [
            "[Integration] Before proposing anything that needs an "
            "optional dependency (p4est, MPI, PETSc, Trilinos, SLEPc, "
            "SUNDIALS, ADOL-C, SymEngine, complex values), read the "
            "feature flags out of the install you are compiling "
            "against. Signal: in "
            "$DEAL_II_DIR/include/deal.II/base/config.h a line "
            "'#define DEAL_II_WITH_P4EST' means ON and "
            "'/* #undef DEAL_II_WITH_P4EST */' means OFF; grepping "
            "for the literal '/* #undef DEAL_II_WITH_' lists every "
            "feature that is OFF in one shot. This is the ONLY "
            "reliable probe. "
            "Two probes that give the WRONG answer, both checked by "
            "execution: (1) including the header. On a source "
            "install every header ships regardless of "
            "configuration - '#include "
            "<deal.II/distributed/tria.h>' and '#include "
            "<deal.II/lac/slepc_solver.h>' both compile and link "
            "cleanly with p4est and SLEPc OFF, because the bodies "
            "sit behind '#ifdef DEAL_II_WITH_<FEATURE>'. (2) "
            "constructing Utilities::MPI::MPI_InitFinalize. It "
            "succeeds with MPI OFF and reports n_mpi_processes == 1. "
            "The real failures are: naming SLEPcWrappers gives "
            "\"'dealii::SLEPcWrappers' has not been declared\", and "
            "constructing parallel::distributed::Triangulation gives "
            "'use of deleted function' - both at COMPILE time, so "
            "they cannot be caught or worked around at run time.",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # SOLVERS
    # ═══════════════════════════════════════════════════════════════════════
    "solvers": {
        "direct": ["SparseDirectUMFPACK (built-in)", "Trilinos Amesos (KLU/SuperLU/MUMPS)",
                   "PETSc MUMPS"],
        "iterative": {
            "builtin": ["SolverCG", "SolverGMRES", "SolverFGMRES", "SolverBicgstab",
                       "SolverMinRes", "SolverQMRS", "SolverRichardson"],
            "trilinos": "TrilinosWrappers::SolverDirect, SolverCG, SolverGMRES",
            "petsc": "PETScWrappers::SolverCG, SolverGMRES, etc.",
        },
        "preconditioners": {
            "builtin": ["PreconditionSSOR", "PreconditionJacobi", "PreconditionChebyshev",
                       "PreconditionBlockSSOR/SOR/Jacobi"],
            "AMG": ["Trilinos ML/MueLu (TrilinosWrappers::PreconditionAMG)",
                    "PETSc GAMG/BoomerAMG"],
            "ILU": "TrilinosWrappers::PreconditionILU/ILUT",
        },
        "multigrid": {
            "geometric": "mg::SmootherRelaxation + MGTransferPrebuilt (step-16)",
            "matrix_free_GMG": "MGTransferMatrixFree + Chebyshev smoother (step-37, 50, 59)",
            "block_smoothers": "Point/block Jacobi/SOR for advection (step-63); SOR-smoothed GMG on the Stokes velocity block (step-56)",
        },
        "matrix_free": {
            "description": "MatrixFree + FEEvaluation: 3 levels of parallelism (MPI + threading + SIMD)",
            "tutorials": "step-37, 48, 59, 64, 66, 67, 75, 76, 95",
            "key_advantage": "10x faster than sparse matrix assembly for high-order elements",
        },
        "nonlinear": ["Manual Newton loop", "SUNDIALS KINSOL (step-77)"],
        "time_integration": ["Manual theta-scheme", "SUNDIALS ARKode (no tutorial uses it)", "PETSc TS (step-86)",
                            # step-52 does NOT exist upstream (see the gap
                            # list above). The TimeStepping / Runge-Kutta
                            # machinery is used by step-67 and step-76,
                            # which are the tutorials to read for it.
                            ("TimeStepping / Runge-Kutta namespace — used in "
                             "step-67 and step-76 (explicit RK for "
                             "matrix-free DG Euler); there is no step-52")],
        "eigenvalue": "SLEPc EPS via PETSc interface (step-36)",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # PARALLEL COMPUTING
    # ═══════════════════════════════════════════════════════════════════════
    "parallel": {
        "shared_memory": "TBB (Threading Building Blocks), Taskflow (v9.7), WorkStream pattern",
        "distributed": "MPI + p4est: parallel::distributed::Triangulation, scalable to 300,000+ processes",
        "GPU": ("Kokkos-based Portable::MatrixFree / Portable::FEEvaluation (step-64); "
                "Kokkos is required since 9.5, and CUDAWrappers::MatrixFree was "
                "deprecated in 9.6 and removed in 9.7"),
        "demonstrated_scale": "2 × 10^12 unknowns on 304,128 MPI processes",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # ADVANCED FEATURES
    # ═══════════════════════════════════════════════════════════════════════
    "advanced": {
        "automatic_differentiation": "Sacado (Trilinos), ADOL-C, SymEngine — step-71/72",
        "manifold_descriptions": ["SphericalManifold", "CylindricalManifold",
                                  "TransfiniteInterpolationManifold (step-65)", "OpenCASCADE (step-54)"],
        "periodic_BCs": "DoFTools::make_periodicity_constraints (step-45)",
        "particles": "ParticleHandler for PIC, advection, non-matching coupling (step-19, 68, 70)",
        "higher_order_output": "DataOutBase::VtkFlags::write_higher_order_cells (ParaView 5.5+)",
        "unfitted_CutFEM": "step-85 (Nitsche BCs on level-set surface), step-95 (MF CutFEM)",
        "non_matching_grids": "step-89 (mortaring), step-60 (distributed Lagrange multipliers)",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # CODE GENERATION PATTERN
    # ═══════════════════════════════════════════════════════════════════════
    "code_generation": {
        "cmake": """cmake_minimum_required(VERSION 3.13.4)
find_package(deal.II 9.5.0 REQUIRED HINTS ${DEAL_II_DIR})
deal_ii_initialize_cached_variables()
project(my_simulation)
add_executable(my_simulation main.cpp)
deal_ii_setup_target(my_simulation)""",

        "class_structure": """
class MyProblem {
  void make_grid();        // Create or import mesh
  void setup_system();     // Distribute DOFs, create sparsity, allocate
  void assemble_system();  // Element loop: FEValues, cell_matrix, cell_rhs
  void solve();            // Linear/nonlinear solve
  void output_results();   // DataOut::write_vtu()
  void run();              // Orchestrate all steps
};""",

        "includes_by_capability": {
            "grid": "#include <deal.II/grid/tria.h>, grid_generator.h, grid_in.h, grid_refinement.h",
            "fe": "#include <deal.II/fe/fe_q.h>, fe_system.h, fe_values.h, fe_interface_values.h",
            "dof": "#include <deal.II/dofs/dof_handler.h>, dof_tools.h, dof_renumbering.h",
            "lac": "#include <deal.II/lac/sparse_matrix.h>, solver_cg.h, precondition.h, affine_constraints.h",
            "numerics": "#include <deal.II/numerics/data_out.h>, vector_tools.h, matrix_tools.h, error_estimator.h",
        },

        "build_and_run": "cmake -Bbuild . && cmake --build build -j$(nproc) && ./build/my_simulation",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # PITFALLS
    # ═══════════════════════════════════════════════════════════════════════
    "pitfalls": {
        "general": [
            "Always refine_global() BEFORE distributing DOFs",
            "FEValues must be reinitialized per cell: fe_values.reinit(cell)",
            "Hanging node constraints MUST be built (DoFTools::"
            "make_hanging_node_constraints) and applied for AMR — but "
            "forgetting them is SILENT: the matrix stays exactly "
            "symmetric and CG converges in the same iteration count; "
            "only the L2 error stops improving under refinement "
            "(verified by running the same problem both ways, see the poisson "
            "pitfall catalogue for the numbers)",
            "Use AffineConstraints for both Dirichlet BCs and hanging node constraints",
            "VTU output: call DataOut::build_patches() before write_vtu()",
        ],
        "parallel": [
            "For MPI: use TrilinosWrappers or PETScWrappers linear algebra, NOT built-in",
            "Each MPI process only owns part of mesh — use locally_owned_dofs + locally_relevant_dofs",
            "Ghost cell communication needed for assembly of off-process DOFs",
        ],
        "performance": [
            "Matrix-free: MatrixFree::reinit works with FE_Q, FE_DGQ AND "
            "(since the simplex support landed) FE_SimplexP on a "
            "tetrahedral mesh — verified in 3D, where FEEvaluation "
            "reproduced the assembled matrix-vector product to "
            "round-off. It ABORTS (exit 134, uncatchable) on the "
            "moment-based vector families FE_RaviartThomas / FE_BDM. "
            "Test the element you intend to use on a tiny mesh before "
            "committing to it",
            "For hp: set active_fe_index per cell BEFORE distributing DOFs",
            "SolverCG requires SPD system — use SolverGMRES for indefinite (Stokes)",
            "Block systems (Stokes): use DoFRenumbering::component_wise + BlockSparseMatrix",
        ],
    },
}
