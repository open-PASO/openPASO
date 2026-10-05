"""Cross-backend collation pitfalls.

When a simulation engineer ports the same problem between two backends —
e.g. validating a fenics result against a kratos run — the failure modes
are NOT in any single backend's catalog. They live in the *delta* between
two backends that both claim to solve the same problem.

This module collects those delta-pitfalls. Each entry follows the standard
[Category]+Signal format used elsewhere in the catalog, with the extra
constraint that the Signal clause MUST name at least two backends
side-by-side. A pitfall that's "just a fenics issue" or "just a kratos
issue" belongs in src/backends/<be>/, not here.

Surfaced via `knowledge(topic='cross_backend')` — see
src/tools/consolidated.py.
"""
from __future__ import annotations


_UNITS_DESC = (
    "All 8 backends are UNIT-AGNOSTIC: they perform arithmetic on numbers "
    "and assume the user has fed them a self-consistent set. None validates "
    "that your inputs form a coherent unit system. This is the single most "
    "common bug when porting a problem between backends."
)

_UNITS_PITFALLS = [
    "[Cross-Backend][Units] FEBio assumes no unit system (its user "
    "manual: 'FEBio does not assume a specific unit system'); the "
    "manual's worked alternative to SI is mm-N-s (mass = tonne, "
    "length = mm, time = s, derived force = N, derived stress = MPa, "
    "derived density = tonne/mm^3 = 1e-9 SI density). "
    "Most fenics / kratos / dealii / ngsolve / skfem / dune / 4C examples "
    "use SI (m, kg, s, derived Pa, derived kg/m^3). Signal: same problem "
    "in two backends returns results differing by exactly 1e3, 1e6, or "
    "1e9 — that factor is the unit-system mismatch, not a numerical bug. "
    "Defense: pick ONE unit system upfront before touching any backend, "
    "put it in a single header constant (E_PA = 210e9 OR E_MPa = 210e3, "
    "never both), and convert ALL backend inputs to match. Verify with a "
    "dimensional sanity check on the FIRST timestep: max displacement "
    "should match analytical expectation to within 10%.",

    "[Cross-Backend][Units] Density conversion is the most-missed step. "
    "Steel in SI is 7850 kg/m^3; in mm-tonne-s it is 7.85e-9 tonne/mm^3 "
    "(NOT 7850, NOT 7.85e-6). Signal: explicit dynamics or modal analysis "
    "ported between SI fenics and mm-tonne-s FEBio gives natural "
    "frequencies off by a factor of sqrt(1e6) ~ 1000. Defense: when "
    "porting from SI to mm-tonne-s, divide density by 1e9 (NOT 1e6, NOT "
    "1e3); when porting from mm-tonne-s to SI, multiply density by 1e9.",

    "[Cross-Backend][Units] Material model parameters embedded in physics: "
    "e.g., a Neo-Hookean shear modulus mu=80 GPa is 8e10 Pa in SI but "
    "8e4 MPa in mm-tonne-s. Signal: hyperelastic problem ported between "
    "fenics (SI) and FEBio (mm-tonne-s) shows max stress off by 1e6 (Pa "
    "vs MPa) — easy to miss because both backends report numbers without "
    "units. Defense: write a units-comment block at the top of every "
    "input file that lists the unit of EACH material parameter, not just "
    "E and nu.",

    "[Cross-Backend][Units] Loads and BCs: an applied traction of 100 in "
    "SI (fenics / kratos / dealii defaults) means 100 Pa, in mm-tonne-s "
    "(an FEBio model built in mm-N-s) means 100 MPa = 100e6 Pa — a "
    "1e6 difference. "
    "Signal: contact / hyperelastic / plasticity problem produces "
    "unphysically large deformations in one backend (e.g. FEBio reading "
    "100 as 100 MPa), near-zero in another (e.g. fenics reading 100 as "
    "100 Pa), for the 'same' load value. Defense: tractions and "
    "pressures are where the mismatch bites hardest because every "
    "backend accepts a bare number. Always paste the unit-system "
    "constant next to the load definition in code review.",
]

_UNITS_SIGNAL = (
    "[Cross-Backend][Units] Aggregate: cross-backend numerical "
    "discrepancies that are CLEAN powers of 1000 (1e3, 1e6, 1e9, 1e12) "
    "are almost always unit-system mismatch, NOT a numerical bug. If you "
    "see exactly 1000x, 1e6x, or 1e9x between backends, fix the unit "
    "system before debugging anything else."
)


_NODE_ORDER_DESC = (
    "Mesh-format converters (Gmsh, VTK, ABAQUS .inp) and backend internal "
    "element orderings do NOT agree on the local numbering of element "
    "nodes. A quadratic tet has 10 nodes and the on-edge midpoint nodes "
    "appear in different positions in Gmsh order vs VTK order vs ABAQUS "
    "order vs each backend's internal order."
)

_NODE_ORDER_PITFALLS = [
    "[Cross-Backend][Mesh] Quadratic-tet (Tet10) midpoint-node ordering "
    "differs: Gmsh has midpoints in order (0,1)(1,2)(0,2)(0,3)(2,3)(1,3); "
    "VTK and ABAQUS C3D10 use (0,1)(1,2)(0,2)(0,3)(1,3)(2,3) — note "
    "swapped 5th/6th (meshio permutes Gmsh tetra10 by "
    "[0,1,2,3,4,5,6,7,9,8]). fenics/dolfinx imports Gmsh order "
    "natively via dolfinx.io.gmsh.read_from_msh (the module is `gmsh` on "
    "dolfinx 0.10; it was `gmshio` up to 0.9 and importing the old name "
    "now raises ModuleNotFoundError); dealii's GridIn "
    "re-orders to its own internal scheme; FEBio's tet10 takes the "
    "VTK / ABAQUS order (FETet10 shape functions), so Gmsh "
    "connectivity written to a .feb unpermuted mislocates two "
    "midside nodes (on a unit tet FEBio then stops with 'Negative "
    "jacobian detected during mesh initialization.'). "
    "Signal: an MMS convergence study on a 10-node tet shows the right "
    "rate in one backend but a polluted rate (1.5 instead of 3) in "
    "another — caused by midpoints being mis-located after import. "
    "Defense: for Tet10 / Hex20 / Quad9, never trust a converter; verify "
    "with a single Gauss-point test: place a known polynomial f(x,y,z) "
    "on the nodes, evaluate at the cell centroid, compare across "
    "backends; mismatch reveals the ordering bug.",

    "[Cross-Backend][Mesh] Linear-quad vertex order: Gmsh, VTK, skfem "
    "and 4C list quad vertices cyclically (counter-clockwise; some "
    "FEBio .feb files via legacy converters are CW), whereas "
    "dolfinx/basix use tensor-product order (0,0),(1,0),(0,1),(1,1) "
    "and dolfinx.io.gmsh permutes Gmsh quads by [0,1,3,2] on import. "
    "Signal: quad connectivity in cyclic order passed to "
    "dolfinx.mesh.create_mesh without that permutation gives "
    "twisted cells (det J changes sign inside the cell and the "
    "assembled area is wrong) without any error. Defense: always pass meshes through "
    "one canonical converter (gmsh+meshio) before feeding any backend; "
    "never bridge ABAQUS->FEBio->fenics in one pipeline. Anchor with a "
    "single-element sanity test: apply a known body force, check that "
    "the centroid displacement matches analytical to 5 digits.",
]

_NODE_ORDER_SIGNAL = (
    "[Cross-Backend][Mesh] If a converged MMS rate in one backend "
    "becomes non-converged in another on the SAME mesh file, suspect "
    "node ordering BEFORE suspecting the discretisation. Validate with "
    "a sentinel: a single high-order element with a known analytic "
    "field, compared node-by-node."
)


_LE_SEMANTICS_DESC = (
    "The phrase 'linear elastic' is overloaded across backends. Some "
    "backends interpret it as small-strain (Cauchy stress = D * sym(grad "
    "u), Green strain tensor LINEARISED); some use the same template "
    "name but actually wire up Total Lagrangian Green-Lagrange / 2nd "
    "Piola-Kirchhoff. Both produce identical results in the small-strain "
    "limit but diverge when applied strain exceeds ~5%."
)

_LE_SEMANTICS_PITFALLS = [
    "[Cross-Backend][Physics] 'linear_elasticity' across backends: "
    "fenics/dolfinx + skfem + ngsolve + dune use small-strain (Cauchy "
    "stress = lambda*tr(eps)*I + 2*mu*eps with eps = sym(grad u), no F "
    "= I + grad u, no PK1 / PK2 machinery). FEBio's 'isotropic elastic' "
    "material (close name) is St-Venant-Kirchhoff (finite strain, "
    "stress from the Green-Lagrange strain) in every analysis, "
    "STATIC included; FEBio has no small-strain switch (a "
    "<kinematic_type> tag is rejected as unrecognized). Kratos's LinearElastic3DLaw "
    "(StructuralMechanicsApplication; 'LinearElasticIsotropic3DLaw' "
    "is an MPMApplication law) is small-strain only with "
    "SmallDisplacementElement*; with TotalLagrangianElement* the same "
    "law acts on the Green-Lagrange strain (St-Venant-Kirchhoff). "
    "4C has no MAT_ELAST_HOOKE: MAT_Struct_StVenantKirchhoff is "
    "small-strain with element KINEM linear and a "
    "geometrically-nonlinear St-Venant-Kirchhoff (finite strain) "
    "with KINEM nonlinear. Signal: a uniaxial tension test stretched to 10% strain "
    "gives the SAME stress in fenics-LE and kratos "
    "LinearElastic3DLaw on a SmallDisplacementElement "
    "(small-strain), but FEBio (any analysis) "
    "and 4C with KINEM nonlinear give a slightly DIFFERENT "
    "stress because they use St-Venant-Kirchhoff. "
    "Defense: when validating a 'linear elastic' problem across "
    "backends, always check the analysis-type setting of EACH backend "
    "and force kinematics='small_strain' / 'linear' / KINEM linear in "
    "the input file even when you THINK it's the default.",
]

_LE_SEMANTICS_SIGNAL = (
    "[Cross-Backend][Physics] 'Linear elastic' is not the same material "
    "law in every backend at >5% strain. The small-strain / "
    "Green-Lagrange branch depends on the BACKEND's analysis-type "
    "metadata, not on the material name. Force the kinematics tag "
    "explicitly in every backend's input."
)


_DIRICHLET_DESC = (
    "Backends impose Dirichlet boundary conditions through different "
    "mechanisms: row-elimination (strong) vs penalty vs Nitsche vs "
    "Lagrange multiplier. Each affects matrix structure, condition "
    "number, and the spurious stress/reaction values at the boundary."
)

_DIRICHLET_PITFALLS = [
    "[Cross-Backend][BC] Dirichlet enforcement: dolfinx's "
    "fem.dirichletbc uses strong row-elimination (diagonal = 1, "
    "off-diagonal = 0, RHS = bc_value); skfem's enforce() sets the "
    "Dirichlet rows to zero with diagonal 1 and RHS = bc_value, while "
    "skfem's condense() removes the Dirichlet DOFs and returns the "
    "reduced system A[I][:, I] x_I = b[I] - A[I][:, D] x[D]. "
    "NGSolve flags Dirichlet DOFs with the lowercase FESpace "
    "argument dirichlet='<regex>' (fes.Update() takes no such "
    "argument, and 'Dirichlet=' is ignored with a warning); forms "
    "are still assembled over all DOFs, and the constraint is "
    "imposed by solving only on fes.FreeDofs() (e.g. "
    "a.mat.Inverse(fes.FreeDofs())). Kratos's "
    "AssignVectorByDirectionProcess fixes the DOFs (\"constrained\": "
    "true by default) and sets their values; the builder-and-solver "
    "then imposes them strongly (the default block builder zeroes the "
    "fixed rows and columns and keeps a scaled diagonal), with no "
    "penalty; the 1e30 in its defaults is the end of the time "
    "\"interval\". 4C uses strong elimination by default but switches to "
    "penalty/Nitsche for some contact / TSI use cases. FEBio uses "
    "penalty for prescribed displacements on contact surfaces. Signal: "
    "cross-backend reaction-force comparison on the SAME Dirichlet "
    "boundary shows agreement on interior fields to 1e-8 but "
    "reaction-force agreement only to 1e-3 — the discrepancy is the "
    "penalty residual in the backend that uses penalty. Defense: when "
    "porting a problem, force STRONG elimination where each backend "
    "supports it (penalty_coefficient=None or explicit elimination "
    "flag); accept that reaction forces computed from penalty methods "
    "carry an O(1/penalty) bias.",
]

_DIRICHLET_SIGNAL = (
    "[Cross-Backend][BC] If two backends agree on interior fields but "
    "disagree on Dirichlet-boundary reaction forces, check the BC "
    "enforcement mechanism (strong vs penalty) BEFORE re-meshing or "
    "refining."
)


_RESTART_DESC = (
    "Backend-internal checkpoint/restart files are NEVER cross-portable. "
    "Each backend writes its own binary format with backend-specific "
    "assumptions about mesh layout, DOF ordering, time-step metadata, "
    "and solver state."
)

_RESTART_PITFALLS = [
    "[Cross-Backend][Output] Restart files are PER-BACKEND. FEBio's "
    ".dmp binary dump (written with -dump, resumed with -r), fenics's dolfinx checkpoint .bp/.xdmf (h5 "
    "backend), 4C's YAML .control file plus binary HDF5 .mesh.* / "
    ".result.* files, ngsolve's .pickle dump, "
    "kratos's .rest serializer files (RestartUtility) all use "
    "incompatible layouts. There is no "
    "'open restart format'. Signal: trying to chain a 4C structural "
    "pre-stress computation into a fenics dynamic analysis via "
    "checkpoint+restart fails immediately with a binary-incompatible "
    "parse error, or — worse — succeeds at loading garbage and runs to "
    "NaN. Defense: always exchange FIELD DATA (displacement, velocity, "
    "stress) between backends via a NEUTRAL format: XDMF+HDF5 for "
    "time-dependent, .vtu for snapshots, .npz for raw arrays. Never "
    "exchange solver state. Re-initialise the solver from the field at "
    "the new backend's t0.",
]

_RESTART_SIGNAL = (
    "[Cross-Backend][Output] Cross-backend workflows must exchange "
    "FIELDS, not SOLVER STATE. The neutral exchange format is XDMF+HDF5 "
    "(time-dependent) or .vtu / .npz (snapshots). Reinitialise solver "
    "state on import."
)


_MPI_DESC = (
    "Each backend has its own MPI bootstrapping convention. Mixing "
    "them — or wrapping the wrong one in mpirun — produces silent "
    "serial runs masquerading as parallel."
)

_MPI_PITFALLS = [
    "[Cross-Backend][Performance] MPI launch: dolfinx / 4C use "
    "MPI_COMM_WORLD natively and require `mpirun -n N python script.py` "
    "(or the 4C binary). NGSolve's shared-memory threading "
    "(TaskManager) is opt-in via `with TaskManager():`, and its MPI "
    "support is native (Netgen is built with MPI; a netgen mesh "
    "takes an mpi4py comm= and is split with Distribute()); there "
    "is no 'ngs-petsc-mpi' (optional PETSc access is "
    "ngsolve.ngs2petsc). Running `mpirun -n N ngspy script.py` on a "
    "script that builds its mesh without a communicator produces N "
    "independent runs, each computing the FULL "
    "problem (wasted compute, results identical, no parallel speedup). "
    "Kratos uses MPI via its own KratosTrilinosApplication; ditto "
    "wrapping a non-Trilinos Kratos script in mpirun is a no-op. "
    "Signal: a 4-process mpirun gives speedup ~0.95x (slightly slower "
    "than serial) on a problem you expect to scale linearly — almost "
    "certain you wrapped a backend that doesn't natively use "
    "MPI_COMM_WORLD. Defense: verify MPI is actually used by adding a "
    "MPI.COMM_WORLD.Get_rank()/Get_size() print at the start of every "
    "backend-MPI script; if rank=0 and size=1 on all N processes, your "
    "mpirun did nothing useful.",
]

_MPI_SIGNAL = (
    "[Cross-Backend][Performance] A backend that doesn't natively use "
    "MPI_COMM_WORLD will not parallelise via `mpirun -n N`. Verify with "
    "a Get_rank()/Get_size() print before debugging scalability."
)


_ELEMENT_TYPE_DESC = (
    "Element-type names like 'hex8' / 'HEX8' / 'C3D8' / 'Element3D8N' / "
    "'ElementHex8' refer to nominally the same 8-node hexahedral "
    "element across all 8 backends — but each backend's API exposes a "
    "DIFFERENT string for the same shape, and a copy-pasted element "
    "name from one backend's example silently fails in another."
)

_ELEMENT_TYPE_PITFALLS = [
    "[Cross-Backend][API] Element-type naming: kratos uses "
    "Element3D8N (8-node hex), Element3D4N (4-node tet), "
    "Element3D6N (6-node wedge), Element3D10N (10-node tet). 4C uses "
    "the element type SOLID followed by a separate cell-type token "
    "(element lines like '1 SOLID HEX8 <nodes> MAT 1 KINEM "
    "nonlinear', with HEX8 / TET4 / TET10 / PYRAMID5 / WEDGE6); the "
    "legacy element types removed in March 2025 "
    "(SOLIDH8_DEPRECATED, SOLIDT4_DEPRECATED, ...) were also "
    "followed by a separate HEX8 / TET4 cell token. fenics/dolfinx uses cell type enums "
    "mesh.CellType.hexahedron / tetrahedron / prism (lowercase), "
    "passed as the cell_type kwarg to mesh.create_box. skfem uses "
    "Python classes ElementHex1 / ElementTetP1 / ElementHex2 / "
    "ElementTetP2 (with explicit polynomial order in the class name). "
    "FEBio's .feb XML uses elem type='hex8' / 'tet4' / 'tet10' / "
    "'penta6' (note: 'penta' not 'wedge' or 'prism'). dealii uses C++ "
    "enum ReferenceCells::Hexahedron / Tetrahedron / Pyramid / Wedge. "
    "Signal: copy-pasting an element name string from one backend's "
    "example into another's input file fails. Kratos raises "
    "RuntimeError 'Element HEX8 is not registered in Kratos.' when "
    "reading the .mdpa ('The Element \"HEX8\" is not registered!' "
    "from ModelPart.CreateNewElement); Kratos spells it Element3D8N "
    "(geometry-only) or, for solids, e.g. "
    "SmallDisplacementElement3D8N; 4C aborts with 'Unknown type "
    "'<X>' of finite element' for an unknown element type and "
    "'Unknown celltype <X>' for an unknown cell token (a fused "
    "'1 SOLIDHEX8 1 2 ...' line fails as 'Unknown celltype 1', "
    "because the node id is read as the cell type); FEBio "
    "rejects <Elements type='hexahedron'> (or 'HEX8') at read time "
    "with 'Invalid element type'. Defense: never "
    "copy an element string between backends. Look up the target "
    "backend's exact spelling in its own catalog (knowledge tool, "
    "topic='overview') before writing the input.",

    "[Cross-Backend][API] Wedge / prism / pyramid element naming "
    "is the most-confused: 4C calls it WEDGE6 (element line 'SOLID "
    "WEDGE6 ...'), kratos calls it "
    "Element3D6N (the dimensionality-and-node-count name; no shape "
    "indicator at all), fenics calls it mesh.CellType.prism, dealii "
    "uses ReferenceCells::Wedge, FEBio uses 'penta6'. Five different "
    "words for the same 6-node element. Signal: a Gmsh-generated mesh "
    "containing wedge cells imports correctly to fenics (prism) and "
    "dealii (Wedge) but fails to load in kratos unless the .mdpa "
    "element block names them Element3D6N, and fails in FEBio unless "
    "the .feb element type='penta6'. Defense: pre-process Gmsh "
    "output through meshio's per-backend writers (meshio.write with "
    "the target format), not through a single-format intermediate.",
]

_ELEMENT_TYPE_SIGNAL = (
    "[Cross-Backend][API] Element-type strings (hex8 / HEX8 / "
    "SOLID HEX8 / Element3D8N / ElementHex1 / mesh.CellType.hexahedron) "
    "are backend-specific spellings of the same shape. Never copy "
    "between backends. Look up the target backend's exact spelling "
    "in its own catalog before writing input."
)


_TIME_INTEGRATION_DESC = (
    "Implicit-dynamics time integration scheme defaults differ across "
    "backends. The SAME problem (mass-spring oscillator, structural "
    "dynamics, transient heat) ported across backends gives different "
    "trajectories because the default beta / gamma / alpha parameters "
    "are not standardised."
)

_TIME_INTEGRATION_PITFALLS = [
    "[Cross-Backend][Numerical] Newmark-beta default parameters: "
    "the 'average acceleration' / 'undamped trapezoidal' scheme is "
    "beta=1/4, gamma=1/2 (unconditionally stable, no algorithmic "
    "damping). 4C's default for structural_dynamics is beta=0.25, "
    "gamma=0.5 (matches average-acceleration). FEBio's default is "
    "ALSO beta=0.25, gamma=0.5 (matches). NGSolve has no "
    "TimeIntegrationNewmark: ngsolve.timestepping.Newmark takes no "
    "beta/gamma arguments and hard-codes the average-acceleration "
    "update (equivalent to beta=0.25, gamma=0.5); "
    "scikit-fem ships no Newmark or time-integration helper, so "
    "the user writes the time loop and its parameters. Kratos's "
    "StructuralMechanicsApplication implicit dynamic solver defaults "
    "to scheme_type 'bossak' with damp_factor_m=-0.3 ('newmark' "
    "forces damp_factor_m to 0, i.e. beta=0.25, gamma=0.5). The "
    "Bossak scheme exposes only damp_factor_m (no alpha_f) and "
    "RECOMPUTES beta=(1-alpha)^2*newmark_beta and gamma=0.5-alpha "
    "(Bossak, not HHT; alpha must lie in [-0.5, 0], so a positive "
    "alpha_m such as 0.05 is rejected), and a C++ "
    "ResidualBasedNewmarkDisplacementScheme built from Parameters "
    "takes damp_factor_m=-0.3 from the Bossak defaults without "
    "recomputing beta/gamma unless damp_factor_m: 0 is passed. "
    "dealii's "
    "step-23/step-25 wave examples use a theta-scheme on the "
    "first-order (u, v) system with theta=0.5 (Crank-Nicolson) "
    "hard-coded in the constructor, not a scheme they call "
    "Newmark; for step-23's linear wave equation, theta=0.5 "
    "gives the same iterates as Newmark beta=0.25, gamma=0.5. "
    "Signal: a 4C linear oscillator ported to "
    "Kratos's StructuralMechanics implicit dynamic solver with its "
    "default scheme ('bossak', damp_factor_m=-0.3) decays faster "
    "than the same run with scheme_type 'newmark' — the default "
    "adds Bossak algorithmic damping. Defense: when porting a transient structural problem, "
    "explicitly set beta, gamma, alpha_m, alpha_f in EVERY backend's "
    "input (do not rely on defaults); verify with a no-damping "
    "single-DOF reference: period error after 100 cycles should be "
    "< 1% if and only if the schemes match.",
]

_TIME_INTEGRATION_SIGNAL = (
    "[Cross-Backend][Numerical] If a transient solid/structural "
    "problem matches across backends at t=0 but the displacement "
    "envelope decays at different rates, the cause is almost always "
    "an algorithmic-damping parameter (alpha_m / alpha_f) silently "
    "added by one backend's default but not the other's. Always "
    "explicitly set ALL Newmark/generalised-alpha parameters in the "
    "input, do not trust 'default'."
)


_TOLERANCE_DESC = (
    "Newton and Krylov solver convergence-tolerance defaults differ "
    "by orders of magnitude across backends. The SAME problem can "
    "report 'converged in 4 iterations' in one backend and "
    "'diverged after 50 iterations' in another not because the "
    "physics differs but because the relative-tolerance default "
    "happened to land at a different power of 10."
)

_TOLERANCE_PITFALLS = [
    "[Cross-Backend][Numerical] Newton solver default tolerances: "
    "dolfinx's deprecated NewtonSolver (dolfinx.nls.petsc) "
    "defaults to rtol=1e-9, atol=1e-10, max_it=50 (very strict); "
    "the dolfinx 0.10 fem.petsc.NonlinearProblem (PETSc SNES) "
    "defaults to rtol=1e-8, atol=1e-50, stol=1e-8, max_it=50 "
    "unless set via petsc_options. NGSolve's solvers.Newton() defaults "
    "to maxerr=1e-11 absolute. Kratos's "
    "ResidualBasedNewtonRaphsonStrategy defaults to relative_tol="
    "1e-4, absolute_tol=1e-9 (LOOSER on relative). 4C's "
    "STRUCTURAL DYNAMIC Newton defaults are TOLDISP=1e-10 and "
    "TOLRES=1e-8, both absolute (NORM_DISP / NORM_RESF Abs, "
    "combined with NORMCOMBI_RESFDISP And). FEBio's solid (and biphasic / multiphasic) "
    "solver defaults to dtol=0.001 (displacement increment), "
    "etol=0.01 (energy) and rtol=0 (residual check off); only "
    "the fluid-family solvers default to rtol=0.001 (VERY loose "
    "by FEM standards — 0.1%). dealii has no global "
    "default; each step-XX tutorial sets it locally, usually "
    "1e-6 to 1e-8. Signal: ported nonlinear problem 'converges' "
    "in FEBio (its solid solver stops on dtol=1e-3 / etol=1e-2 "
    "with the residual check off) but in fenics the SAME problem "
    "runs until the PETSc SNES tests pass (rtol=1e-8 on the "
    "residual, stol=1e-8 on the step; atol 1e-10 / rtol 1e-9 with "
    "the deprecated NewtonSolver) — FEBio's default 'converged' "
    "says nothing about the residual. "
    "Defense: when porting, "
    "ALWAYS set rtol AND atol explicitly in every backend's input. "
    "Use rtol=1e-8, atol=1e-10 as a portable cross-backend baseline "
    "for production problems; rtol=1e-4 for fast smoke tests.",

    "[Cross-Backend][Numerical] Krylov (CG / GMRES / BiCGStab) "
    "default tolerances similarly differ. PETSc-backed solvers "
    "(dolfinx) default to ksp_rtol=1e-5, ksp_atol="
    "1e-50 (essentially relative-only); 4C does not use PETSc: "
    "its iterative solvers are Trilinos Belos with AZTOL default "
    "1e-8, relative to the initial residual. NGSolve's solvers.CGSolver "
    "defaults to tol=1e-12 RELATIVE to the initial residual (atol "
    "unset, maxiter=100); the C++ CGSolver that 'from ngsolve "
    "import *' exports defaults to precision=1e-8, also relative. "
    "scipy.sparse.linalg.cg (used by skfem) "
    "defaults to rtol=1e-5, atol=0 (also relative-only). Kratos's "
    "AMGCL/Trilinos defaults are application-specific; "
    "TrilinosLinearSolver typically tol=1e-6. Signal: ported linear "
    "problem solves to 'machine precision' in NGSolve but only to "
    "1e-5 relative in PETSc-backed dolfinx, and a downstream "
    "nonlinear Newton loop in dolfinx fails to converge because "
    "the linear solve residual seeds the Newton residual at 1e-5 "
    "instead of 1e-12. Defense: set ksp_rtol AND ksp_atol "
    "explicitly when porting; tighten ksp_rtol BELOW the outer "
    "Newton rtol by at least 2 orders of magnitude (Newton rtol="
    "1e-8 → ksp_rtol=1e-10 minimum).",
]

_TOLERANCE_SIGNAL = (
    "[Cross-Backend][Numerical] If a 'converged' nonlinear solution "
    "in one backend reports a different final residual than the "
    "same problem in another backend, check the default Newton/"
    "Krylov tolerances FIRST. Same-name convergence ('converged') "
    "carries different meanings: FEBio's solid solver by default "
    "checks only the displacement increment (dtol=1e-3) and the "
    "energy (etol=1e-2), with the residual check off (rtol=0), "
    "while dolfinx's NonlinearProblem (PETSc SNES) tests the "
    "residual (rtol=1e-8) as well as the step (stol=1e-8)."
)


_CONTACT_FORMULATION_DESC = (
    "Contact-mechanics enforcement methods differ across backends "
    "in defaults AND in available options. The SAME contact problem "
    "(Hertzian sphere-on-plane, frictional slide, multi-body "
    "assembly) solved 'with default settings' in two backends "
    "produces non-comparable results because the constraint "
    "formulation is silently different."
)

_CONTACT_FORMULATION_PITFALLS = [
    "[Cross-Backend][Physics] Contact constraint enforcement: "
    "Kratos's ContactStructuralMechanicsApplication has no default "
    "formulation: the contact process you choose sets it "
    "(alm_contact_process = mortar augmented Lagrangian, "
    "penalty_contact_process = mortar penalty, mpc_contact_process, "
    "mesh_tying_process); penalty_contact_process computes "
    "INITIAL_PENALTY = 1e4*stiffness_factor*E_mean/h_mean unless "
    "manual_ALM is set, and adapt_penalty defaults to false. "
    "4C's CONTACT block defaults to STRATEGY 'Lagrange' "
    "(true Lagrange multipliers via mortar segmentation), with "
    "PENALTY available via STRATEGY 'Penalty'. FEBio's sliding "
    "contact interfaces (sliding-elastic, sliding-node-on-facet, "
    "sliding-facet-on-facet) default to PENALTY; augmented "
    "Lagrangian is selected with <laugon>AUGLAG</laugon> (there "
    "is no augmented_lagrangian attribute; FEBio rejects it at "
    "read time). dolfinx "
    "has no built-in contact; users build it via custom Nitsche "
    "or external library (Mirco, Conmech). NGSolve provides "
    "ngsolve.comp.ContactBoundary; its contact tutorial "
    "(i-tutorials unit-6.2, high-order VectorH1) enforces "
    "non-penetration with a gap penalty (IfPos), not Nitsche. "
    "Signal: "
    "Hertzian sphere-on-plane test ported between Kratos "
    "(penalty_contact_process) and 4C (Lagrange) gives matching peak pressure but the "
    "Kratos solution shows ~1e-3 penetration at the contact patch "
    "(penalty residual) while 4C shows ~1e-12 (true Lagrange "
    "satisfies zero-penetration to solver tolerance). Defense: "
    "when validating cross-backend, ALWAYS use Lagrange or "
    "augmented Lagrange — penalty's accuracy depends on a "
    "user-set penalty factor that has no universal default. State "
    "the formulation EXPLICITLY in the input.",
]

_CONTACT_FORMULATION_SIGNAL = (
    "[Cross-Backend][Physics] If two backends agree on bulk "
    "stress fields in a contact problem but disagree on the "
    "contact-patch penetration depth by orders of magnitude, "
    "the cause is penalty (one backend) vs Lagrange/augmented "
    "Lagrange (the other). Force Lagrange or augmented Lagrange "
    "explicitly when validating across backends."
)


_OUTPUT_FORMAT_DESC = (
    "Output file conventions — VTK / XDMF / ADIOS2 / .post.bin — "
    "differ in (a) what each backend can write, (b) where it puts "
    "DOFs (point-data on mesh nodes vs cell-data vs DG-style "
    "per-element data), and (c) how high-order polynomial "
    "discretisations are represented. ParaView opens all of them "
    "but interprets the same field as different things."
)

_OUTPUT_FORMAT_PITFALLS = [
    "[Cross-Backend][Output] Point-data vs cell-data: dolfinx's "
    "VTXWriter (and VTKFile) write Lagrange P1 / P2 / etc. DOFs "
    "as POINT-DATA on the function-space dof coordinates (a P2 "
    "field adds mid-edge points and is written on higher-order "
    "Lagrange cells; nothing is interpolated to vertices), while "
    "dolfinx.io.XDMFFile.write_function raises 'Degree of output "
    "Function must be same as mesh degree' when the Function "
    "degree differs from the mesh degree, so a P2 field on a P1 "
    "mesh must be interpolated to P1 by the user first. "
    "Kratos's VtkOutput writes NODAL_RESULTS as POINT-DATA too. "
    "skfem's skfem.io.json stores only the mesh, and Mesh.save / "
    "skfem.io.meshio.to_meshio write exactly the point_data / "
    "cell_data arrays the user passes (one value per vertex / per "
    "cell), so a DG field must first be reduced by the user (e.g. "
    "projected to P1, or averaged per cell). dealii's DataOut "
    "writes one (bi-/tri-)linear patch per cell by default "
    "(build_patches() uses 1 subdivision, so a P2 field is "
    "written as its vertex values on the simulation mesh); only "
    "build_patches(n) with n > 1 subdivides each cell into n^dim "
    "linear sub-cells (a P2 field on 100 2-D cells with n = 2 "
    "produces 400 sub-cells in the .vtu file) — ParaView then "
    "shows a smoother field, but the cell count differs from the "
    "simulation mesh; with DataOutBase::VtkFlags::"
    "write_higher_order_cells each patch is instead written as "
    "one Lagrange cell. NGSolve's VTKOutput "
    "subdivides only when asked (`subdivision=N` kwarg; the "
    "default 0 writes one cell per element, and 1 already "
    "subdivides). "
    "FEBio's .xplt is its own binary format and only ParaView via "
    "the FEBio plugin reads it natively. Signal: a P2 stress field "
    "ported to ParaView via fenics-XDMF (after user interpolation "
    "to P1) "
    "and via dealii-VTK (build_patches(n) with n > 1, subdivided) "
    "shows the SAME peak value but "
    "the dealii output reports n^dim times more cells (4x for "
    "n = 2 in 2-D), and a downstream "
    "ParaView Calculator filter that iterates over cells produces "
    "different integrated quantities. Defense: when cross-backend "
    "post-processing in ParaView, force ALL outputs to a single "
    "convention (build_patches() / build_patches(1) in dealii, "
    "subdivision=0 in NGSolve, P1-projection in "
    "fenics) before any Filter / Calculator operation; or do the "
    "post-processing on raw arrays via meshio/numpy, not in "
    "ParaView.",

    "[Cross-Backend][Output] XDMF vs .bp (ADIOS2) vs .vtu: dolfinx "
    "0.10+ recommends VTXWriter -> .bp (ADIOS2 directory) for "
    "high-order Lagrange Functions; XDMFFile.write_function "
    "rejects a Function whose degree differs from the mesh degree "
    "with 'RuntimeError: Degree of output Function "
    "must be same as mesh degree'. 4C writes .vtu natively via "
    "IO/RUNTIME VTK OUTPUT, plus its own YAML .control file and "
    "HDF5 .mesh.* / .result.* files that post_processor converts "
    "(filters ensight, vtu, vti); it has no XDMF writer. Kratos "
    "default GidOutput is GID-format binary (not XDMF, not VTU); "
    "VtkOutput writes .vtu. NGSolve writes .vtu via "
    "VTKOutput(mesh, coefs=[...], names=[...], filename=...).Do(). "
    "ParaView 5.10+ reads ALL of .bp/.xdmf/.vtu; ParaView <5.10 "
    "reads .xdmf/.vtu only. Signal: a cross-backend workflow "
    "where one stage writes .bp (dolfinx VTXWriter; dolfinx "
    "VTKFile .pvd/.vtu also writes P>1 Lagrange Functions on the "
    "dof grid, while XDMFFile.write_function requires the "
    "Function degree to equal the mesh degree) and another stage "
    "tries to read that output with meshio raises "
    "meshio.ReadError ('Could not deduce file format from path "
    "...'), because meshio 5.3.5, the latest release, has no "
    "ADIOS2 .bp reader. Defense: pick ONE neutral exchange format for "
    "cross-backend pipelines: .vtu (linearised, ParaView 5+ "
    "reads), or interpolated-to-P1 XDMF. Reserve .bp for "
    "dolfinx-only chains.",
]

_OUTPUT_FORMAT_SIGNAL = (
    "[Cross-Backend][Output] If the same field-export pipeline "
    "produces different integrated quantities in ParaView Filters "
    "(min/max, integrate, surface-cell-count), the cause is "
    "almost always cell-subdivision (dealii only with "
    "build_patches(n > 1); NGSolve only with subdivision>=1; "
    "Kratos writes nodal point data; fenics VTX/VTKFile write the "
    "dof grid). Force "
    "build_patches() (dealii) / subdivision=0 (NGSolve) / "
    "P1-projection on ALL backends before any "
    "post-processing Filter."
)


_INTEGRATION_ORDER_DESC = (
    "Gauss-quadrature order selection differs across backends. "
    "The SAME polynomial-order element (P2 / Q2 / Hex8) may use "
    "different default numbers of integration points in each "
    "backend, leading to slightly different stress / strain-"
    "energy values for the same physical problem."
)

_INTEGRATION_ORDER_PITFALLS = [
    "[Cross-Backend][Numerical] Default Gauss integration order: "
    "dolfinx selects quadrature degree automatically based on UFL "
    "form analysis (estimate_total_polynomial_degree) UNLESS the "
    "user passes form_compiler_options={'quadrature_degree': N} "
    "or sets it on the dx measure (dx(metadata={'quadrature_"
    "degree': N})); the auto-estimate may be CONSERVATIVE (over-"
    "integrate) or INSUFFICIENT (under-integrate) depending on "
    "nonlinearity. skfem's Basis (on which @BilinearForm is "
    "assembled) defaults to intorder = 2*elem.maxdeg, the element's "
    "highest polynomial degree (= p for simplex Lagrange, but 2 for "
    "ElementQuad1 and 3 for ElementHex1); override with "
    "Basis(..., intorder=N). NGSolve "
    "Integrate(cf, mesh) uses a fixed default order=5 whatever the "
    "integrand (pass order=N for higher-degree integrands); "
    "Integrate(cf*dx, mesh) takes no order kwarg and uses order "
    "5 + dx(bonus_intorder=k). dealii's QGauss<dim>(n) is explicit — n = "
    "ceil((2*p+1)/2) is the standard textbook formula but every "
    "tutorial sets it manually. Kratos's "
    "StructuralMechanicsApplication solid elements (e.g. "
    "SmallDisplacementElement3D8N) use the geometry's default Gauss "
    "rule (2x2x2 = 8 points for a hexahedron) unless the "
    "INTEGRATION_ORDER property overrides it (1 -> 1 point, 3 -> 27 "
    "points); Element3D8N is a geometry-only placeholder and there is "
    "no Element3D8NReduced. 4C SOLID elements take an optional "
    "INTEGRATION group with RESIDUUM / MASS Gauss rules (e.g. in "
    "an ELEMENT_BLOCKS entry: INTEGRATION: {RESIDUUM: "
    "hex_27point}; the HEX8 default is hex_8point); there is no "
    "GAUSSRULE input keyword. FEBio selects the "
    "quadrature rule via the element-spec string, e.g. "
    "<SolidDomain ... elem_type=\"HEX8G1\"/> or <Elements "
    "type=\"TET10G4\">; there is no gp_order attribute (an unknown "
    "gp_order attribute is silently ignored). Signal: a hyperelastic P2 "
    "compression test ported between fenics (auto quadrature) "
    "and Kratos (default 2x2x2 for SmallDisplacementElement3D8N) "
    "reports slightly "
    "different peak strain-energy because fenics auto-picked "
    "quadrature degree 4 (over-integrated for cubic strain field) "
    "while Kratos's 2x2x2 under-integrates the strain energy by "
    "~0.1% (within engineering tolerance but visible in MMS "
    "convergence studies as a stalled rate at fine mesh). "
    "Defense: when validating cross-backend, explicitly set the "
    "quadrature order to the same value in every backend; for "
    "nonlinear elements use 2*p+1 minimum (where p is the shape-"
    "function order), or 2*p+2 for hyperelastic to capture the "
    "nonlinear strain-energy density.",
]

_INTEGRATION_ORDER_SIGNAL = (
    "[Cross-Backend][Numerical] If an MMS convergence rate "
    "stalls at the predicted theoretical order in one backend "
    "but achieves super-convergence in another, suspect "
    "integration order BEFORE suspecting the discretisation. "
    "Fenics auto-quadrature can over-integrate (free super-"
    "convergence) while Kratos's default hexahedron rule (2x2x2 "
    "unless INTEGRATION_ORDER is set) under-integrates."
)


_BOUNDARY_TAG_DESC = (
    "Gmsh physical-group tags / boundary IDs are mesh metadata "
    "that backends interpret differently. A `gmsh model.add_physical_"
    "group(2, [10], tag=5)` in Python becomes a SURFACE id 5 in "
    "the .msh file, but each backend then maps that to its own "
    "internal boundary-id namespace with different fall-through "
    "rules when a tag is missing."
)

_BOUNDARY_TAG_PITFALLS = [
    "[Cross-Backend][Mesh] Boundary-id default fallthrough: "
    "dolfinx's mesh.locate_entities_boundary(mesh, fdim, marker) "
    "uses a USER-supplied lambda (no tag mapping at all — the "
    "user must reconstruct facet IDs from coordinates); the "
    "alternative read_from_msh path preserves Gmsh physical-"
    "group tags on a mesh.MeshTags object that the user passes "
    "to fem.dirichletbc(value, dofs, V). dealii's GridIn::read_"
    "msh imports Gmsh tags as boundary_id() attributes on the "
    "Triangulation's faces — directly usable by "
    "VectorTools::interpolate_boundary_values. Kratos's "
    "ModelPart loads sub-model-parts named after Gmsh physical "
    "groups but the .mdpa user must HAND-EDIT the sub-model-"
    "part section to expose them — Gmsh→Kratos converters don't "
    "always preserve the physical-group name. 4C's DESIGN SURF "
    "DIRICH CONDITIONS entries (4C 2026 reads only .yaml / .yml "
    "/ .json input) use E ids that must match the DSURF-NODE "
    "TOPOLOGY design-surface ids (ENTITY_TYPE legacy_id) or, with "
    "ENTITY_TYPE: node_set_id, a node-set id of an external mesh "
    "— for a Gmsh .msh mesh (read directly only by a 4C built "
    "with FOUR_C_WITH_GMSH=ON) that is the Gmsh physical-group "
    "tag itself; Gmsh tags are user-chosen integers numbered "
    "from 1 by default, so no +1 shift applies, and a converter "
    "that renumbers tags must carry the mapping into the E "
    "values. FEBio .feb XML uses "
    "<Surface name=...> blocks identified by NAME not by "
    "numeric ID. Signal: SAME .msh file feeds dolfinx and 4C; "
    "dolfinx applies Dirichlet on facets with tag=5 correctly; "
    "4C applies Dirichlet on the WRONG surface because the 4C "
    "input still references E=5 but the Gmsh-to-4C converter "
    "renumbered that surface to E=6. Defense: when porting a Gmsh "
    "mesh between backends, always print the per-facet tag "
    "histogram in EACH backend's loader (count facets per "
    "boundary_id) and compare; mismatch reveals the renumbering "
    "or naming bug.",
]

_BOUNDARY_TAG_SIGNAL = (
    "[Cross-Backend][Mesh] If a Dirichlet boundary 'works' in "
    "one backend and quietly applies to the wrong face in "
    "another with the SAME mesh, the cause is a tag renumbering "
    "in the Gmsh-to-4C conversion (Gmsh and 4C both use the ids "
    "as given; there is no 0- vs 1-based shift) or a "
    "sub-model-part name not preserved in the .mdpa converter. "
    "Print per-facet tag histograms before applying any BC."
)


_PLASTICITY_RETURN_MAP_DESC = (
    "Plasticity return-mapping algorithms — radial return / "
    "cutting-plane / closest-point projection / Newton-on-yield "
    "— differ across backends in (a) algorithm choice and (b) "
    "default convergence criteria. Same J2 / Mohr-Coulomb / "
    "Drucker-Prager material with default solver settings "
    "produces slightly different stress paths in different "
    "backends."
)

_PLASTICITY_RETURN_MAP_PITFALLS = [
    "[Cross-Backend][Numerical] Return-mapping defaults for J2 "
    "plasticity: 4C's J2 material MAT_Struct_PlasticLinElast "
    "(linear isotropic / kinematic hardening) solves its return "
    "map with a local Newton iteration to the required absolute "
    "TOL (at most 50 iterations). Kratos's SmallStrainJ2Plasticity3DLaw uses a "
    "closed-form radial return for linear hardening and a local "
    "Newton iteration (|f| < 1e-6*YIELD_STRESS) only for "
    "exponential saturation hardening. "
    "FEBio's J2 material 'von-Mises plasticity' (E, v, Y, H; "
    "linear isotropic hardening) uses a closed-form one-step "
    "radial return, and 'reactive plasticity' iterates its "
    "return map to its own rtol parameter (default 1e-4). dolfinx "
    "has no built-in plasticity — users implement custom return "
    "maps via UFL+conditional or external libraries (dolfiny). "
    "Signal: SAME uniaxial cyclic tension-compression test on "
    "J2 plasticity, loaded past yield, produces matching "
    "elastic branches but different residual plastic strain "
    "after the first half-cycle. The return maps are not the "
    "same algorithm: FEBio's 'von-Mises plasticity' and Kratos's "
    "SmallStrainJ2Plasticity3DLaw with linear hardening return in "
    "closed form, while 4C's MAT_Struct_PlasticLinElast iterates "
    "to its TOL, FEBio's 'reactive plasticity' to its rtol "
    "(default 1e-4) and Kratos's exponential hardening to "
    "1e-6*YIELD_STRESS. Defense: use the same hardening law and "
    "parameters in every backend, and set the return-map "
    "tolerance tightly (e.g. 1e-10) where the backend exposes it "
    "(4C TOL, FEBio reactive-plasticity rtol; Kratos's "
    "1e-6*YIELD_STRESS is hard-coded).",

    "[Cross-Backend][Numerical] Mohr-Coulomb tension cut-off: "
    "Kratos has no MohrCoulombPlasticity law or "
    "tension_cutoff_factor key; GeoMechanicsApplication's "
    "GeoMohrCoulombWithTensionCutOff2D/3D requires "
    "GEO_TENSILE_STRENGTH (checked to lie in [0, "
    "GEO_COHESION/tan(GEO_FRICTION_ANGLE)]), and the "
    "PoromechanicsApplication interface law "
    "ElastoPlasticMohrCoulombCohesive3DLaw requires "
    "TENSILE_STRENGTH. 4C has no Mohr-Coulomb material and no "
    "apex_smoothing_factor; its pressure-sensitive law "
    "MAT_Struct_DruckerPrager returns to the apex with a local "
    "Newton iteration (no smoothing parameter). FEBio has no Mohr-Coulomb material "
    "(type=\"mohr-coulomb\" is rejected with 'invalid value for "
    "attribute \"type\"'). dolfinx custom "
    "implementations vary; the most common Mohr-Coulomb-with-"
    "tension-cutoff template (e.g. dolfiny's geomechanics) has "
    "no default and the user must set it. Signal: a Mohr-Coulomb "
    "model ported to Kratos's GeoMohrCoulombWithTensionCutOff3D "
    "without a tensile strength fails the law's Check() with "
    "'GEO_TENSILE_STRENGTH is not defined for property', and the "
    "same model has no counterpart in 4C or FEBio (neither ships "
    "a Mohr-Coulomb material; FEBio rejects type='mohr-coulomb' "
    "with 'invalid value for attribute \"type\"'). Defense: when "
    "validating cross-backend Mohr-Coulomb, state the yield "
    "surface, the tension cut-off and the apex treatment "
    "explicitly in every backend, and check first that the law "
    "exists in the target backend.",
]

_PLASTICITY_RETURN_MAP_SIGNAL = (
    "[Cross-Backend][Numerical] If a cyclic plasticity test "
    "matches on the elastic branch but residual plastic strains "
    "differ across backends, compare the return maps and their "
    "tolerances first: closed form in FEBio 'von-Mises "
    "plasticity' and in Kratos's J2 law with linear hardening; a "
    "local Newton to TOL in 4C MAT_Struct_PlasticLinElast, to "
    "rtol in FEBio 'reactive plasticity' and to 1e-6*YIELD_STRESS "
    "in Kratos's exponential hardening. For Mohr-Coulomb / "
    "Drucker-Prager, check that the law exists in each backend "
    "and how it treats the apex and the tension cut-off (4C and "
    "FEBio have no Mohr-Coulomb material)."
)


_TURBULENCE_DESC = (
    "Reynolds-Averaged Navier-Stokes (RANS) turbulence model "
    "defaults differ in (a) wall treatment (wall-function vs "
    "low-Re damping), (b) inlet turbulence-intensity defaults, "
    "and (c) model-constant choices. The SAME k-epsilon "
    "channel-flow problem can converge to qualitatively "
    "different velocity profiles in two backends because the "
    "wall-treatment default differs."
)

_TURBULENCE_PITFALLS = [
    "[Cross-Backend][Physics] Turbulence wall treatment: 4C has "
    "no RANS or k-epsilon model: FLUID DYNAMIC/TURBULENCE MODEL "
    "offers TURBULENCE_APPROACH DNS_OR_RESVMM_LES (default) or "
    "CLASSICAL_LES (PHYSICAL_MODEL Smagorinsky, "
    "Smagorinsky_with_van_Driest_damping, Dynamic_Smagorinsky, "
    "Multifractal_Subgrid_Scales, Vreman, Dynamic_Vreman), and "
    "its only wall model is the X_WALL enrichment (FLUID "
    "DYNAMIC/WALL MODEL, off by default). Kratos's "
    "k-epsilon lives in RANSApplication (FluidDynamicsApplication "
    "has none) and uses logarithmic-region wall functions (walls "
    "must be SLIP; nu_t from RansNutYPlusWallFunctionUpdateProcess) "
    "with standard constants (c_mu 0.09, c1 1.44, c2 1.92, sigma_k "
    "1.0, sigma_epsilon 1.3), so a wall mesh with y+ well below the "
    "log region violates its assumption. NGSolve and dolfinx have no built-in "
    "RANS — users write custom forms (UFL in dolfinx, NGSolve's "
    "own symbolic forms); OPENPASO (a fenics-"
    "based RANS solver) defaults to standard wall functions. "
    "FEBio has no fluid-turbulence support. Signal: a k-eps "
    "channel flow that ran in another code on a wall-resolved "
    "mesh (y+ < 1) cannot be moved to 4C at all (no k-epsilon "
    "input), and in Kratos RANSApplication it contradicts the "
    "log-region wall-function assumption (SLIP walls, first cell "
    "in the log region). Defense: report y+ of the first wall "
    "cell in each backend and keep it in the log region for "
    "Kratos's wall functions, or coarsen the wall mesh; in 4C use "
    "its LES / DNS approaches (TURBULENCE_APPROACH, optional "
    "X_WALL wall model) instead of RANS.",

    "[Cross-Backend][Physics] Inlet turbulence-intensity (TI) "
    "defaults: 4C has no k-epsilon inlet condition (its "
    "turbulent inflow is the DESIGN SURF TURBULENT INFLOW "
    "TRANSFER condition, which has no TI or length scale). "
    "Kratos's RANSApplication inlet processes "
    "RansKTurbulentIntensityInletProcess and "
    "RansEpsilonTurbulentMixingLengthInletProcess default to "
    "turbulent_intensity = 0.05 (5%) and turbulent_mixing_length = "
    "0.005. dolfinx user-written RANS: no "
    "default, user must set. Signal: a pipe flow ported to Kratos "
    "RANSApplication without explicit inlet values silently uses "
    "turbulent_intensity 0.05 and turbulent_mixing_length 0.005, "
    "whatever the source code's inlet used, and it has no 4C "
    "counterpart (no k-epsilon inlet condition). Defense: always specify TI and Lt "
    "EXPLICITLY in the inlet BC; never rely on the backend's "
    "default."
]

_TURBULENCE_SIGNAL = (
    "[Cross-Backend][Physics] RANS k-eps/k-omega/SST turbulence "
    "results that disagree across backends on the SAME mesh + "
    "SAME inlet usually fail on (1) wall treatment (Kratos "
    "RANSApplication uses log-region wall functions, so y+ "
    "matters) and (2) inlet turbulence values (Kratos defaults "
    "turbulent_intensity 0.05, turbulent_mixing_length 0.005). "
    "State both explicitly in every backend's input; 4C has no "
    "RANS model, and NGSolve / dolfinx need user-written RANS "
    "forms."
)


_MATERIAL_ORIENTATION_DESC = (
    "Anisotropic material orientation — fiber direction in "
    "transverse isotropy, layup angle in laminated shells, "
    "principal-axis frame in orthotropic elasticity — is "
    "specified through different mechanisms in each backend "
    "and the default fallback (when not specified) varies."
)

_MATERIAL_ORIENTATION_PITFALLS = [
    "[Cross-Backend][Physics] Fiber-direction specification in "
    "transversely-isotropic / orthotropic materials: 4C's "
    "anisotropic hyperelastic summands (e.g. ELAST_CoupAnisoExpo, "
    "INIT default 1) take the fiber direction from the element "
    "line (FIBER1 ... vectors) or from an element cylinder "
    "coordinate system (RAD / AXI / CIR); there is no "
    "MAT_AAA_FIBER and no .fib file. FEBio's transversely isotropic materials (e.g. "
    "<material type='trans iso Mooney-Rivlin'>; there is no "
    "<transversely_isotropic> element) take a <fiber> property, "
    "e.g. <fiber type='vector'>1,0,0</fiber>; other fiber types "
    "include 'local' (two element-local node numbers), "
    "'spherical', 'cylindrical', 'angles', 'math' and 'map' "
    "(element-wise mesh data). Kratos has no FiberReinforcedMaterial or "
    "FIBER_DIRECTION_1; its anisotropic laws (e.g. "
    "GenericAnisotropic3DLaw) orient the material with the "
    "EULER_ANGLES property. dolfinx custom UFL: user defines a vector-valued "
    "Function f and writes the constitutive law explicitly "
    "(no default direction). dealii similarly user-supplied. "
    "DEFAULT FALLBACK when fiber direction is unspecified: 4C "
    "(INIT 1) aborts with 'Could not find element coordinate "
    "system or element fibers!'; FEBio's trans iso materials refuse "
    "to read without a fiber ('needs to have property \"fiber\" "
    "defined'), while materials oriented by <mat_axis> (e.g. "
    "Holzapfel-Gasser-Ogden) silently use the global axes when "
    "it is omitted; "
    "Kratos's GenericAnisotropic3DLaw silently uses the global "
    "axes (identity rotation) when EULER_ANGLES is absent or zero. "
    "Signal: ported fiber-reinforced beam-bending "
    "problem where the fiber should be along the beam axis but "
    "the input forgot to specify it: 4C (INIT 1) aborts and "
    "FEBio's trans iso materials refuse to read, but FEBio's "
    "<mat_axis> materials and Kratos's GenericAnisotropic3DLaw "
    "silently use the global axes (which COULDN'T match the "
    "beam axis if the beam is oriented along y), giving "
    "qualitatively wrong stiffness. Defense: ALWAYS specify "
    "fiber direction explicitly (no defaults); verify with a "
    "simple uniaxial tension along the fiber axis — the "
    "stiffness should match the fiber-direction Young's "
    "modulus, not the matrix modulus.",
]

_MATERIAL_ORIENTATION_SIGNAL = (
    "[Cross-Backend][Physics] If an anisotropic-material "
    "problem (transverse isotropy, orthotropy, laminated shell) "
    "gives qualitatively wrong stiffness on a SAME-mesh "
    "cross-backend port, the cause is almost always an "
    "unspecified fiber/principal-axis direction. 4C aborts and "
    "FEBio trans iso materials refuse to read without one, while "
    "FEBio <mat_axis> materials and Kratos's "
    "GenericAnisotropic3DLaw silently use the global axes. "
    "Specify direction explicitly in every backend."
)


_FREQUENCY_DESC = (
    "Eigenvalue / modal analysis backends differ in whether "
    "they report eigenvalues as angular frequency squared "
    "(omega^2 = (2*pi*f)^2 in rad^2/s^2), as angular frequency "
    "(omega in rad/s), or as ordinary frequency (f in Hz). "
    "The same modal problem solved by two backends produces "
    "numerical results that look unrelated until you account "
    "for factors of 2*pi and squaring."
)

_FREQUENCY_PITFALLS = [
    "[Cross-Backend][Numerical] Eigenvalue return convention "
    "for modal analysis: SLEPc (used by dolfinx via "
    "dolfinx.fem.petsc + slepc4py) and PETSc returns "
    "eigenvalues as the literal numerical eigenvalues of the "
    "(K, M) generalised problem K phi = lambda M phi — i.e. "
    "lambda = omega^2 in rad^2/s^2 (NOT Hz, NOT rad/s). NGSolve's "
    "ArnoldiSolver / PINVIT returns lambda = omega^2 in the same "
    "convention. skfem's solver_eigen_scipy_sym() (wraps "
    "scipy.sparse.linalg.eigsh) and solver_eigen_scipy() (wraps "
    "eigs; the solve_eigen default), used via skfem.solve(K, M, "
    "solver=...), also return omega^2. "
    "Kratos's EigensolverStrategy in StructuralMechanics returns "
    "lambda = omega^2 by default (EIGENVALUE_VECTOR), and its "
    "PostprocessEigenvaluesProcess REPORTS them by default "
    "(label_type 'frequency') as Hz after sqrt(lambda)/(2*pi). "
    "4C has no modal / eigenfrequency analysis (its structural "
    "DYNAMICTYPE options are Statics, GenAlpha, GenAlphaLieGroup, "
    "OneStepTheta, ExplicitEuler, CentrDiff, AdamsBashforth2 and "
    "AdamsBashforth4). FEBio 4 "
    "has no modal / eigenfrequency analysis (<analysis> accepts "
    "only STATIC / STEADY-STATE and DYNAMIC / TRANSIENT; MODAL is "
    "rejected with 'invalid value: MODAL'). dealii has no built-in modal solver; tutorial step-36 "
    "uses SLEPc and reports the raw lambda. Signal: a 1D "
    "cantilever-beam-bending modal analysis on the same mesh "
    "gives 'first eigenvalue = 2.41e6' in dolfinx-SLEPc (omega^2 "
    "in rad^2/s^2) vs a label of 247 (Hz) in Kratos's "
    "PostprocessEigenvaluesProcess output vs 1553 when omega is "
    "reported in rad/s. All three are the same "
    "physical frequency; users porting cross-backend often miss "
    "the omega^2 / omega / Hz distinction and conclude one "
    "backend is wrong by a factor of (2*pi)^2 ~ 39.5 or 2*pi ~ "
    "6.28. Defense: always print BOTH omega^2 AND sqrt(omega^2)/"
    "(2*pi) [Hz] in every backend's modal output; compare Hz "
    "values, never raw eigenvalues."
]

_FREQUENCY_SIGNAL = (
    "[Cross-Backend][Numerical] Cross-backend modal-analysis "
    "discrepancies of (2*pi)^2 ~ 39.5 or 2*pi ~ 6.28 are NOT "
    "physical errors — they are unit conventions (omega^2 in "
    "rad^2/s^2 vs omega in rad/s vs f in Hz). Always convert to "
    "Hz before comparing. SLEPc/PETSc/NGSolve/scipy (and Kratos's "
    "EIGENVALUE_VECTOR) return omega^2; Kratos's "
    "PostprocessEigenvaluesProcess labels Hz by default; 4C and "
    "FEBio 4 have no modal analysis."
)


_MESH_QUALITY_DESC = (
    "Mesh-quality acceptance thresholds — aspect ratio, "
    "skewness, Jacobian positivity, minimum dihedral angle — "
    "differ across backends in both the default rejection "
    "threshold AND in whether a bad element causes a hard "
    "error vs a silent slow-down via solver ill-conditioning."
)

_MESH_QUALITY_PITFALLS = [
    "[Cross-Backend][Mesh] Negative-Jacobian element rejection: "
    "4C's SOLID elements check the Jacobian determinant at the "
    "element nodes when the element is first evaluated (not "
    "while the mesh is read) and abort with 'determinant of "
    "jacobian is <value> <= 0 at one node of the element.'. "
    "FEBio also rejects "
    "negative-Jacobian elements, at model initialisation right "
    "after the input file reads successfully ('Negative jacobian "
    "detected during mesh initialization.' then 'Model "
    "initialization failed'), with a message that names no "
    "element. Kratos's elements compute Jacobian on the "
    "fly per assembly call and silently produce NaN in the "
    "stiffness matrix when J<=0 (no upfront check); the LINEAR "
    "SOLVER then reports 'matrix singular' or runs and returns "
    "garbage. dolfinx's dolfinx.mesh.create_mesh does not check "
    "cell orientation or degeneracy: a negative-Jacobian or "
    "zero-volume cell is accepted without error (dx integrals use "
    "|det J|, so assembled volumes do not reveal inverted cells; "
    "evaluate ufl.det(ufl.Jacobian(mesh)) per cell to find them). "
    "NGSolve's "
    "Netgen mesher generates only positive-Jacobian elements by "
    "construction but READING an external bad mesh via "
    "Mesh(filename) silently accepts negatives. dealii's GridIn "
    "readers (e.g. read_msh, for dim == spacedim) silently "
    "re-orient cells with negative measure (an inverted tet gets "
    "two vertices swapped) and throw ExcGridHasInvalidCell at "
    "import, also in Release, only if a cell still has "
    "non-positive measure (e.g. a flat tet). Signal: a Gmsh-exported mesh with one inverted "
    "tetrahedron (common after CSG boolean operations near sharp "
    "features) is rejected before the first solve by 4C (at "
    "element initialisation) and FEBio (at model "
    "initialisation), with messages that name no element, but "
    "loaded silently by "
    "Kratos, NGSolve-from-file and dolfinx; the user discovers it only "
    "when the SECOND solver run starts giving NaN residuals. "
    "Defense: always run a mesh-validation pass (gmsh --check or "
    "pyvista's mesh.compute_cell_quality()) BEFORE feeding any "
    "backend; a single inverted element wastes hours when only "
    "caught by 'matrix singular'.",

    "[Cross-Backend][Mesh] Aspect-ratio / sliver-element "
    "tolerance: 4C has no hard aspect-ratio rejection (silently "
    "accepts slivers; solver may diverge on ill-conditioned "
    "system). FEBio has no aspect-ratio check (a 500:1 hex8 runs "
    "to normal termination without any warning). "
    "Kratos has no built-in check. dolfinx delegates to PETSc "
    "linear-solver tolerance which silently struggles. NGSolve's "
    "Netgen mesher runs optsteps2d / optsteps3d = 3 optimisation "
    "passes by default when it generates a mesh (grading = 0.3 is "
    "the mesh-size grading, not a quality threshold); passing "
    "optsteps3d=0 (GenerateMesh rejects a plain `optsteps` kwarg) "
    "skips them and can leave slivers. Signal: a 'good' "
    "tetrahedral mesh from Gmsh with optimisation disabled "
    "contains O(10) slivers; converged in 4C in 1000 GMRES "
    "iterations (slow but works); diverged in Kratos AMGCL with "
    "default settings; NGSolve loads it unchanged (Mesh(file) and "
    "netgen.read_gmsh.ReadGmsh do not run the Netgen optimiser), "
    "so the slivers stay. Defense: "
    "always check max element aspect ratio (gmsh GUI tools menu) "
    "before solving; if > 50, refine the mesh-generation "
    "parameters BEFORE the solver inevitably has trouble."
]

_MESH_QUALITY_SIGNAL = (
    "[Cross-Backend][Mesh] If a mesh 'works' in one backend but "
    "produces 'matrix singular' / NaN residuals / silent "
    "divergence in another, suspect (1) negative-Jacobian "
    "elements silently accepted by Kratos/NGSolve-from-file/"
    "dolfinx but rejected by 4C/FEBio, or (2) sliver elements with "
    "aspect ratio > 50. Run gmsh --check before feeding any "
    "backend."
)


_STRESS_MEASURE_DESC = (
    "Output stress fields can be reported as Cauchy (true) "
    "stress sigma, 1st Piola-Kirchhoff PK1, 2nd Piola-Kirchhoff "
    "PK2, or as engineering stress sigma_eng. All four agree "
    "in the infinitesimal-strain limit but diverge at finite "
    "strain. Backends differ in WHICH stress measure they "
    "default to in output files."
)

_STRESS_MEASURE_PITFALLS = [
    "[Cross-Backend][Output] Default stress-measure in output: "
    "dolfinx writes whatever the user-projected stress UFL "
    "expression evaluates to — typically the user writes "
    "sigma = mu*(F*F.T - I) + lam*ln(J)*I (Cauchy in current "
    "configuration). FEBio's <stress> output element writes "
    "Cauchy stress sigma in the deformed configuration. 4C's "
    "IO STRUCT_STRESS selects the output stress (Cauchy, or 2PK = "
    "PK2 in the reference configuration; Yes means 2PK) and "
    "defaults to NO (no stress output) regardless of KINEM; "
    "writing it also needs IO/RUNTIME VTK OUTPUT/STRUCTURE "
    "STRESS_STRAIN: true. Kratos's "
    "ConstitutiveLaw::CalculateMaterialResponseCauchy returns "
    "Cauchy by default but VonMisesStress nodal-value "
    "calculation uses whatever the constitutive law's "
    "STRESS_VECTOR contains (varies). NGSolve and skfem "
    "(user-written forms and post-processing): user-controlled. "
    "dealii's data_out.add_data_vector "
    "with the user's stress lambda — convention is whatever the "
    "user computed. Signal: hyperelastic uniaxial tension at "
    "10% stretch — Cauchy reports sigma = mu*(lambda^2 - 1/"
    "lambda) ~ 0.30 mu; PK1 reports P11 = mu*(lambda - 1/"
    "lambda^2) ~ 0.27 mu; PK2 reports S11 = mu*(1 - 1/"
    "lambda^3) ~ 0.25 mu. All correct in their respective "
    "configurations, all DIFFERENT numerical values for the "
    "same physical state. Defense: explicitly name the stress "
    "measure in every backend's output config; convert to "
    "Cauchy for cross-backend comparison via sigma = (1/J)*F*"
    "PK1 = (1/J)*F*PK2*F.T."
]

_STRESS_MEASURE_SIGNAL = (
    "[Cross-Backend][Output] Cross-backend hyperelastic stress "
    "values differing by ~10-20% at >5% strain are almost "
    "always Cauchy-vs-PK1-vs-PK2 reporting differences, not "
    "physical errors. The three measures coincide only at "
    "infinitesimal strain. Always specify the stress measure "
    "in every backend's output config."
)


_PERIODIC_BC_DESC = (
    "Periodic boundary conditions — used for representative-"
    "volume-element (RVE) homogenisation, periodic crystal "
    "structures, and infinite-domain approximations — are "
    "implemented via different constraint mechanisms across "
    "backends with different accuracy and matrix-structure "
    "implications."
)

_PERIODIC_BC_PITFALLS = [
    "[Cross-Backend][BC] Periodic-BC implementation: dolfinx "
    "uses dolfinx_mpc (a separate package, not core dolfinx) "
    "via mpc.create_periodic_constraint_geometrical or "
    "create_periodic_constraint_topological — true "
    "MasterSlaveConstraint at the DOF level, exact periodicity "
    "to solver precision. NGSolve uses the Periodic() wrapper "
    "around any FESpace, which builds master-slave pairs at "
    "construction time — exact periodicity. 4C's DESIGN SURF / "
    "LINE PERIODIC BOUNDARY CONDITIONS pair master and slave "
    "design entities and give the slave nodes the master nodes' "
    "DOFs (node matching within ABSTREETOL), so periodicity is "
    "exact; the RVE conditions DESIGN SURF PERIODIC RVE 3D "
    "BOUNDARY CONDITIONS are enforced by penalty or Lagrange "
    "multipliers, chosen with CONSTRAINT / CONSTRAINT_ENFORCEMENT "
    "(PENALTY_PARAM default 1e5). Kratos's "
    "ApplyPeriodicConditionProcess imposes periodicity with "
    "LinearMasterSlaveConstraint objects (master-slave elimination "
    "in the builder-and-solver), not a penalty; its only 1e30 "
    "default is the end of the \"interval\". FEBio has a built-in "
    "'periodic boundary' surface-pair interaction (<contact "
    "type='periodic boundary' surface_pair=...>, penalty or "
    "augmented-Lagrangian, with an 'offset'), so its periodicity "
    "holds only to the penalty tolerance, and for RVE "
    "homogenisation its FEBioRVE 'micro-material' with "
    "rve_type=1 applies periodic linear constraints to a cube RVE "
    "(the default rve_type 0 prescribes displacements). dealii's "
    "DoFTools::make_periodicity_constraints builds an "
    "AffineConstraints object with exact periodicity. Signal: "
    "RVE-homogenisation problem on the SAME mesh: the "
    "constraint-based implementations (dolfinx-mpc, NGSolve "
    "Periodic, dealii, Kratos ApplyPeriodicConditionProcess, "
    "FEBio micro-material rve_type=1, 4C PERIODIC RVE conditions "
    "with CONSTRAINT_ENFORCEMENT lagrange) give periodicity to "
    "solver precision, while penalty enforcement (FEBio "
    "'periodic boundary', 4C PERIODIC RVE conditions with "
    "penalty, PENALTY_PARAM default 1e5) satisfies it only "
    "approximately (in FEBio the paired-node mismatch shrinks "
    "as the penalty grows). Defense: for production RVE work, "
    "prefer the constraint-based options; with a penalty "
    "method, raise the penalty until the mismatch between "
    "paired nodes is below your tolerance."
]

_PERIODIC_BC_SIGNAL = (
    "[Cross-Backend][BC] If an RVE-homogenisation effective "
    "stiffness tensor C_ijkl differs across cross-backend "
    "runs, check how each backend enforces periodicity: "
    "master-slave / linear constraints (dolfinx-mpc, NGSolve "
    "Periodic FESpace, dealii, Kratos "
    "ApplyPeriodicConditionProcess, FEBio micro-material "
    "rve_type=1, 4C PERIODIC RVE conditions with lagrange) are "
    "exact to solver precision, penalty enforcement (FEBio "
    "'periodic boundary', 4C with CONSTRAINT_ENFORCEMENT "
    "penalty) only approximate."
)


_DAMPING_DESC = (
    "Damping in transient analyses is specified through "
    "fundamentally different mathematical forms — Rayleigh "
    "(C = alpha*M + beta*K), structural / hysteretic damping "
    "ratio xi, viscous coefficient c per element — and "
    "backends differ in which form they use as 'damping' in "
    "their input file. The SAME 'damping ratio = 0.05' "
    "request can yield wildly different actual energy "
    "dissipation across backends."
)

_DAMPING_PITFALLS = [
    "[Cross-Backend][Physics] Damping specification: 4C's "
    "STRUCTURAL DYNAMIC section takes Rayleigh coefficients "
    "M_DAMP (mass) and K_DAMP (stiffness), C = M_DAMP*M + "
    "K_DAMP*K, but only together with DAMPING: Rayleigh: DAMPING "
    "defaults to None, so M_DAMP / K_DAMP given without it are "
    "silently ignored (undamped), and with DAMPING: Rayleigh "
    "both must be given or 4C aborts ('Rayleigh damping "
    "parameter K_DAMP not explicitly given.'). Kratos's StructuralMechanicsApplication accepts "
    "RAYLEIGH_ALPHA + RAYLEIGH_BETA on each element's material "
    "parameters; if missing, treated as 0 (UNDAMPED silently). "
    "FEBio has no per-rigid-body damping input (<damping> on a "
    "rigid body is an unrecognized tag); viscous damping between "
    "rigid bodies uses 'rigid damper' / 'rigid angular damper' "
    "connectors (coefficient c), and a 'mass damping' load "
    "(coefficient C) also exists (not Rayleigh, not structural). "
    "NGSolve's "
    "ngsolve.timestepping.Newmark takes no gamma (fixed "
    "average-acceleration update, no algorithmic damping) and "
    "accepts only u.dt.dt time derivatives (a u.dt damping term "
    "raises 'time-derivatives not allowed in spatial "
    "integrators'), so physical Rayleigh damping needs a "
    "user-written time-stepping loop with the damping term in "
    "the bilinear form. dolfinx has no built-in "
    "damping; user writes the Rayleigh expression in UFL. "
    "dealii varies per tutorial. Signal: same transient analysis "
    "with 'damping ratio xi = 0.05' converted by hand to 4C's "
    "M_DAMP / K_DAMP (4C's STRUCTURAL DYNAMIC has no "
    "damping-ratio input; alpha = 2*xi*omega_1, beta = 2*xi/"
    "omega_max for a mode pair) vs Kratos (interpreted as "
    "Rayleigh alpha = 0.05 directly, beta = 0) gives different "
    "decay envelopes — 4C correctly damps modes between "
    "omega_1 and omega_max; Kratos's alpha=0.05 alone damps "
    "ALL modes uniformly via the mass term, over-damping "
    "high-frequency modes by orders of magnitude. Defense: "
    "always specify damping as Rayleigh alpha + beta "
    "explicitly; do NOT pass 'damping ratio' to a backend "
    "without first converting to alpha + beta using the "
    "two-mode formula alpha = 2*xi*om1*om2/(om1+om2), "
    "beta = 2*xi/(om1+om2) for the frequency range of "
    "interest.",
]

_DAMPING_SIGNAL = (
    "[Cross-Backend][Physics] If a damped transient analysis "
    "matches across backends at low frequency but diverges "
    "wildly at high frequency, the cause is mass-proportional "
    "Rayleigh damping (alpha*M) being applied uniformly to all "
    "modes. Compute alpha + beta from a two-mode formula for "
    "your target frequency range; never pass 'damping ratio' "
    "as a raw alpha."
)


_TIMESTAMP_DESC = (
    "Time-series output files differ across backends in (a) "
    "whether the t=0 frame is written, (b) whether output is "
    "captured at start-of-step or end-of-step (matters for "
    "implicit methods where field is unknown at start), and "
    "(c) the time-stamp metadata precision (single vs double "
    "in some XDMF/VTK writers)."
)

_TIMESTAMP_PITFALLS = [
    "[Cross-Backend][Output] t=0 frame inclusion: 4C writes "
    "the initial state (t=0) by default (IO WRITE_INITIAL_STATE, "
    "default true; not on restart). FEBio's "
    "plotfile-must-include t=0 default writes t=0. Kratos's "
    "VtkOutputProcess, driven by AnalysisStage, writes only in "
    "OutputSolutionStep after each solved step, so no t=0 frame is "
    "written unless PrintOutput() is called before the time loop "
    "(VtkOutput has no option for it; unknown keys such as "
    "output_initial_conditions are rejected). dolfinx's "
    "VTXWriter writes only when the user calls vtx.write(t); "
    "user-controlled — typical scripts forget to write t=0 "
    "BEFORE the time loop. NGSolve VTKOutput(...).Do(): same as "
    "dolfinx — user-controlled. dealii data_out_stack: "
    "user-controlled but every tutorial includes t=0. Signal: "
    "ParaView animation of a 4C output (frame 0 = t=0, frames "
    "1..N = computed steps) shows the initial geometry; same "
    "problem in Kratos VtkOutput (frame 0 = t=dt, no t=0) "
    "shows the post-first-step state as 'initial' — a "
    "visualisation that misleads users into thinking the "
    "initial condition was different. Defense: in Kratos / "
    "dolfinx / NGSolve, explicitly write the t=0 frame before "
    "the time loop starts.",

    "[Cross-Backend][Output] Start-of-step vs end-of-step "
    "field capture: implicit time integration computes the "
    "field at t_{n+1} from the field at t_n. dolfinx, NGSolve, "
    "skfem, dealii: the user calls write(t_{n+1}, u_{n+1}) "
    "AFTER the solve — output is unambiguously at the "
    "computed time. 4C's structural output is written at the "
    "end of each step (t_{n+1}, u_{n+1}). Kratos's output processes are "
    "called in AnalysisStage.OutputSolutionStep after "
    "FinalizeSolutionStep: end-of-step (t_{n+1}, u_{n+1}), with "
    "output_control_type / output_interval choosing which steps are "
    "written; there is no begin-of-step output setting. FEBio's "
    "logfile vs plotfile "
    "differ: plotfile end-of-step, logfile may be configured "
    "to either. Signal: a velocity / acceleration field "
    "exported alongside displacement: in dolfinx these are "
    "all evaluated at t_{n+1} using the Newmark update of u; "
    "a user-written loop (dolfinx, NGSolve, skfem, dealii) that "
    "calls its write before the solve stores the previous step's "
    "displacement under the new time stamp, so the displacement "
    "is stale by one timestep relative to "
    "the velocity, producing a phase lag artifact in ParaView "
    "velocity-vs-displacement plots. Defense: in user-written "
    "loops, write fields after the solve of each step; never "
    "mix before- and after-solve writes in the same .vtu series.",
]

_TIMESTAMP_SIGNAL = (
    "[Cross-Backend][Output] Cross-backend ParaView animations "
    "where frame 0 shows the post-first-step state in one "
    "backend but the initial condition in another are caused "
    "by Kratos's AnalysisStage (no t=0 frame unless PrintOutput() "
    "is called before the loop) or by dolfinx/NGSolve loops that "
    "never write t=0 (user-controlled). For phase-lag bugs in "
    "velocity-vs-displacement plots, check whether a user-written "
    "loop writes its fields before or after the solve."
)


_INITIAL_CONDITION_DESC = (
    "Initial-condition specification for transient analyses — "
    "whether u(t=0) is supplied as a per-DOF array, a "
    "Function/CoefficientFunction expression, or an analytic "
    "lambda interpolated onto the mesh — differs across "
    "backends in (a) the API surface, (b) the DEFAULT when "
    "unspecified, and (c) the interpolation quality for "
    "high-order spaces."
)

_INITIAL_CONDITION_PITFALLS = [
    "[Cross-Backend][Physics] Initial-condition default: "
    "dolfinx's u = fem.Function(V) is zero-initialised (its "
    "dolfinx.la.Vector storage is a value-initialised "
    "std::vector, not a PETSc Vec), so u.x.array[:] = 0 is an "
    "explicit but redundant zero-IC. NGSolve's GridFunction defaults "
    "to zero (numerically zeroed in C++ ctor). Kratos's "
    "ProcessInfo / nodal variables default to zero. 4C's "
    "transient analyses start from zero unless the dynamics "
    "section selects another initial field (e.g. FLUID DYNAMIC "
    "INITIALFIELD, default zero_field; STRUCTURAL DYNAMIC "
    "INITIALDISP, default zero_displacement), optionally fed by "
    "DESIGN ... INITIAL FIELD CONDITIONS; there is no INITIAL "
    "CONDITIONS section. FEBio's <Initial> XML block "
    "defaults to zero. dealii's Vector<double> ctor zero-"
    "initialises. dolfinx Function INTERPOLATION of an "
    "analytic IC: u.interpolate(lambda x: x[0]**2) for a P2 "
    "FunctionSpace SAMPLES the lambda at the P2 nodal points "
    "(degree-2 interpolation, exact for polynomials up to "
    "deg 2). Kratos's process-based IC application sets only "
    "nodal values (P1-equivalent, even when the element "
    "supports higher order — internal DOFs default to zero). "
    "Signal: ported transient hyperelastic problem with "
    "u(t=0) = sin(pi*x)*sin(pi*y) on a P2 mesh: in dolfinx "
    "the IC is captured to interpolation accuracy O(h^3); in "
    "Kratos (when ported via a process that sets only "
    "nodal values, leaving P2 internal DOFs at zero) the IC "
    "is captured only to O(h^2) AND the internal DOFs that "
    "should hold the polynomial peak are zero, producing a "
    "kink-shaped IC field instead of the smooth sine. "
    "Defense: when porting a non-trivial IC to a P>1 backend, "
    "either (a) use the backend's true interpolation operator "
    "(dolfinx u.interpolate; NGSolve gfu.Set; dealii "
    "VectorTools::interpolate), or (b) confirm the resulting "
    "IC matches the analytic expression at the cell centroid "
    "(not just at vertices) before stepping in time.",
]

_INITIAL_CONDITION_SIGNAL = (
    "[Cross-Backend][Physics] If a transient ported between "
    "backends matches at t=dt but diverges from t=2*dt onward "
    "even with identical solver settings, the cause is "
    "almost always the initial condition was P1-interpolated "
    "in one backend (Kratos process-based IC) vs truly "
    "P-interpolated in another (dolfinx u.interpolate). "
    "Validate IC at cell centroids, not just nodes."
)


_FRAME_OF_REFERENCE_DESC = (
    "Boundary conditions and material orientations specified "
    "in local frames (surface-normal, beam-tangent, "
    "anisotropic principal axes) vs global Cartesian frames "
    "have backend-specific transformation conventions. The "
    "same 'roller constraint normal to the surface' produces "
    "different physical BC in different backends if the "
    "normal-direction computation differs."
)

_FRAME_OF_REFERENCE_PITFALLS = [
    "[Cross-Backend][BC] Local-frame BC computation: 4C's "
    "DIRICH conditions have no normal type (their components are "
    "NUMDOF, ONOFF, VAL, FUNCT and TAG); a local-frame Dirichlet "
    "BC needs a DESIGN SURF (or LINE) LOCSYS CONDITIONS entry on "
    "the same entity whose ROTANGLE rotation vector (optionally "
    "scaled by FUNCT) rotates the nodal frame, and its "
    "USECONSISTENTNODENORMAL option is documented 'not for "
    "solids!', so a structural roller on a curved surface needs "
    "a user-given rotation. FEBio's <Surface "
    "load> normal pressure uses the CURRENT-configuration "
    "normal (updated each Newton iteration for finite "
    "deformation). Kratos has no roller-constraint process: "
    "assign_vector_variable_process with \"constrained\": [false, "
    "false, true] fixes the global z-DOF everywhere, NOT the local "
    "normal — wrong on curved surfaces; a normal-direction "
    "constraint needs SlipConstraint (a LinearMasterSlaveConstraint "
    "built from the nodal DOFs and a normal vector), with nodal "
    "normals from NormalCalculationUtils (e.g. "
    "CalculateUnitNormals). dolfinx: no built-in roller; users compose "
    "via locate_entities_boundary + a custom projection. "
    "Signal: a curved-surface roller-supported plate gives "
    "matching results in every backend while the plate is flat "
    "(global z IS the surface normal) but diverges as the plate "
    "curves wherever a backend constrains one fixed direction "
    "instead of the local normal: a fixed global component in "
    "Kratos, or a single ROTANGLE for a whole 4C LOCSYS surface "
    "(FUNCT can vary it over the surface). "
    "Defense: for curved-surface rollers in Kratos, "
    "compute the per-node normals with NormalCalculationUtils "
    "(ComputeNodalNormalDivergenceProcess computes a divergence, "
    "not normals) and constrain the normal direction with "
    "SlipConstraint; never assume a fixed global component is "
    "the local normal."
]

_FRAME_OF_REFERENCE_SIGNAL = (
    "[Cross-Backend][BC] If a 'roller' or 'normal pressure' "
    "BC produces matching results on flat geometry but "
    "diverges on curved geometry across backends, the cause "
    "is a roller that constrains one fixed direction (a global "
    "component in Kratos, a single LOCSYS rotation in 4C) vs "
    "FEBio's current-config normal pressure. Always verify the BC "
    "by printing the per-node constraint direction on a "
    "test point at the apex of the curvature."
)


_LINEAR_SOLVER_DESC = (
    "Default linear-solver choice, tolerance, iteration cap, "
    "GMRES restart parameter, and direct-solver dispatch path "
    "all differ across backends. Same matrix, same RHS, same "
    "user-stated target precision — different wall-time, "
    "convergence behaviour, or silent fallback solver "
    "depending on which backend's default fires."
)
_LINEAR_SOLVER_PITFALLS = [
    "[Cross-Backend][Solver] Default linear solver dispatch: "
    "fenics/dolfinx's LinearProblem(...).solve() has no "
    "direct-solver default: without petsc_options its KSP uses "
    "PETSc's defaults (GMRES + ILU in serial, GMRES + "
    "block-Jacobi/ILU in parallel, rtol 1e-5), the same as a "
    "hand-built KSP on the assembled PETSc.Mat; pass "
    "petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu', "
    "'pc_factor_mat_solver_type': 'mumps'} for a direct MUMPS "
    "solve. skfem's solve(K, b) hits scipy.sparse.linalg."
    "spsolve (SuperLU direct, single-threaded). For a "
    "50k-DOF Poisson problem fenics LinearProblem takes ~3 "
    "s and skfem solve(K,b) takes ~5 s; for 5M-DOF fenics "
    "needs PETScOptions ksp_type=cg + pc_type=hypre to "
    "stay competitive, while skfem requires an explicit "
    "pyamg / petsc4py dispatch (scipy spsolve runs out of "
    "memory). Never trust 'the default linear solver' — "
    "inspect problem.solver.getType() / "
    "problem.solver.getPC().getType() in fenics (PETSc.Options() "
    "lists only options that were set, not the defaults in "
    "effect) and inspect "
    "type(K) / scipy version in skfem before benchmarking.",
    "[Cross-Backend][Solver] GMRES restart parameter: PETSc "
    "(used by fenics/dolfinx, dealii) defaults to restart=30 — the Krylov subspace "
    "is rebuilt every 30 iterations. NGSolve's GMRES does NOT "
    "auto-restart (solvers.GMRes and krylovspace.GMResSolver "
    "default to restart=None; the C++ GMRESSolver has no restart "
    "option); the total iteration cap is 'maxsteps' "
    "(solvers.GMRes, default 100; C++ GMRESSolver, default 200) "
    "or 'maxiter' (GMResSolver / solvers.CGSolver, default 100), "
    "there is no 'maxit', and divergence on "
    "indefinite systems is silent until the cap fires. The "
    "same matrix can converge in 12 iters under PETSc / "
    "GMRES(30) but stall at 199/200 under NGSolve's non-"
    "restarted GMRES because the Krylov basis becomes "
    "linearly dependent — symptom: 'iter residual' "
    "plateaus around iteration 50 and never drops. Fix: "
    "pass restart=30 explicitly to solvers.GMRes (or "
    "krylovspace.GMResSolver), or "
    "wrap NGSolve's iterative loop with manual restart.",
    "[Cross-Backend][Solver] Direct-solver dispatch path: "
    "dealii's SolverDirect(SolverControl) is a Trilinos wrapper "
    "(TrilinosWrappers::SolverDirect defaults to Amesos_Klu); "
    "UMFPACK is the separate SparseDirectUMFPACK class, which "
    "takes no SolverControl, while 4C selects its direct solver "
    "with 'SOLVER <n>: {SOLVER: UMFPACK | Superlu | MUMPS | KLU2}' "
    "through Trilinos Amesos2; there is no TYPE_OF_SOLVER, LSEAux "
    "or Spooles / Pardiso fallback, and a solver missing from the "
    "Amesos2 build aborts ('Requested direct solver <name> is not "
    "available in Amesos2!') instead of being replaced. Don't "
    "compare solver wall-times across these two without checking "
    "which direct solver each one ran (the 4C deck's SOLVER "
    "entry; dealii's "
    "DEAL_II_WITH_UMFPACK / DEAL_II_WITH_MUMPS cmake "
    "summary).",
    "[Cross-Backend][Solver] Tolerance scaling convention: "
    "fenics/dolfinx's PETSc KSP rtol defaults to 1e-5 (the "
    "PETSc default), while NGSolve's Inverse(inverse=\"...\") is "
    "always a direct factorisation with no tolerance, and its "
    "Krylov solvers default to relative tol=1e-12 (solvers.CG / "
    "CGSolver / GMRes) or precision=1e-8 (C++ CGSolver / "
    "GMRESSolver). Kratos's LinearSolversApplication has no "
    "GMRES; in Kratos, GMRES is the default krylov_type of the "
    "core 'amgcl' solver (defaults tolerance 1e-6, max_iteration "
    "100). A user copying "
    "solver settings from a fenics tutorial (rtol=1e-5) "
    "into a dealii SolverControl(...) call gets 5x faster "
    "but lower-precision solves than the dealii tutorial "
    "defaults assume. Always set tolerance explicitly in "
    "the script; never inherit it silently.",
    "[Cross-Backend][Solver] Iterative-solver iteration "
    "cap default: dealii's SolverControl(n, tol) defaults to "
    "n = 100 iterations and tol = 1e-10 (absolute), so omitting "
    "them silently caps the solve at 100 iterations. "
    "PETSc-based fenics uses 10000 as the "
    "KSP maxits default, while Kratos does not use PETSc KSP (its "
    "amgcl solver defaults to max_iteration 100, cg / bicgstab / "
    "tfqmr to 200). skfem has no AMG solver of its own "
    "(pyamg is not a dependency): solver_iter_pcg / "
    "solver_iter_krylov pass maxiter to "
    "scipy.sparse.linalg.cg, whose default cap is 10*n "
    "iterations, and the pure-Python solver_iter_cg stops at "
    "maxiters=500. For an ill-conditioned 1M-DOF "
    "problem fenics spins for 10000 iterations (~minutes) "
    "before reporting failure, while skfem's default cap is "
    "10*n — same problem, different failure modes. Symptom "
    "in fenics: 'PETSc KSP DIVERGED_ITS' after a long "
    "wall-time; symptom in skfem: solve() returns the "
    "unconverged iterate with no exception, and "
    "solver_iter_pcg / solver_iter_krylov only log "
    "'Iterative solver did not converge.' (solver_iter_cg "
    "logs nothing), so you have to inspect the residual "
    "yourself.",
]
_LINEAR_SOLVER_SIGNAL = (
    "[Cross-Backend][Solver] Same matrix, same RHS, same "
    "target precision — wildly different wall-time, "
    "convergence behaviour, or silent fallback to a different "
    "direct solver (e.g. fenics MUMPS vs dealii UMFPACK vs "
    "4C Amesos2 UMFPACK / Superlu). Diagnose at runtime by printing "
    "the actual KSP type, preconditioner, restart, "
    "tolerance, maxits, AND linked direct-solver library "
    "(`ksp_view` in fenics; in dealii, which has no "
    "solver.print(), enable SolverControl log_history / "
    "log_result with deallog.depth_console(2) and print "
    "last_step() / last_value(), or pass -ksp_view to "
    "PETScWrappers solvers; `ldd` on the fourc binary) BEFORE drawing conclusions "
    "about which backend is 'faster' or 'more accurate' on "
    "the same problem."
)


_NONLINEAR_CONVERGENCE_DESC = (
    "Newton/Picard nonlinear iterations use different "
    "convergence criteria across backends: relative residual "
    "vs absolute residual vs energy norm vs displacement "
    "increment, with or without line search, with different "
    "failure-reporting paths (exception vs tuple-flag vs "
    "exit-code). Same physics, same mesh, the word "
    "'converged' means different things."
)
_NONLINEAR_CONVERGENCE_PITFALLS = [
    "[Cross-Backend][Solver] Newton convergence criterion: "
    "fenics/dolfinx's (deprecated) NewtonSolver, with the "
    "default convergence_criterion='residual', stops when "
    "EITHER ||F(u)|| < atol (default 1e-10) OR ||F(u)||/r0 < "
    "rtol (default 1e-9), where r0 is the norm of the first "
    "Newton update du, not ||F(u_0)||; dolfinx 0.10's "
    "NonlinearProblem uses the PETSc SNES tests instead (rtol "
    "1e-8, atol 1e-50, stol 1e-8, max_it 50). "
    "dealii's hand-coded Newton (step-15 pattern) typically "
    "uses ||F(u)|| / ||F(u_0)|| relative residual only. 4C's "
    "STRUCTURAL DYNAMIC NLNSOL: fullnewton (the default) checks "
    "the displacement increment with NORM_DISP (Abs | Rel | Mix) "
    "against TOLDISP and the residual with NORM_RESF against "
    "TOLRES, combined by NORMCOMBI_RESFDISP (And | Or); coupled "
    "sections use their own keys instead (TSI DYNAMIC: NORM_INC, "
    "with TOLINC / CONVTOL in TSI DYNAMIC/MONOLITHIC). A 'converged' "
    "model in fenics (||F|| < 1e-10 or relative < 1e-9) can still "
    "fail 4C's test if NORM_RESF is 'Rel' and TOLRES is tight — "
    "different criterion entirely.",
    "[Cross-Backend][Solver] Energy-norm convergence: "
    "NGSolve's Newton (ngsolve.solvers.Newton) has a single "
    "stopping test: the energy norm of the Newton correction, "
    "sqrt(|du.(K du)|) = sqrt(|<K^-1 r, r>|), below maxerr "
    "(default 1e-11; there is no tol_energy), whereas FEBio's "
    "solid solver (default qn_method BFGS) uses <etol> (default "
    "0.01) = ratio of the current energy norm to the initial "
    "one, alongside <dtol> (0.001), with the residual check "
    "<rtol> off (0) by default. For a plasticity problem "
    "where the residual is hard to reduce but the energy "
    "stagnates, NGSolve declares convergence early and "
    "FEBio iterates further — same problem, different "
    "'converged' answer. Workaround: always print residual, "
    "increment, AND energy per iteration, and pick the "
    "criterion explicitly rather than relying on the "
    "'default'.",
    "[Cross-Backend][Solver] Initial-residual zero-load "
    "trap: dealii Newton steppers commonly compute the "
    "'initial residual' at iteration 0 with Dirichlet BCs "
    "applied but ZERO loading — yielding ||F_0|| near "
    "machine epsilon. The relative criterion ||F_k|| / "
    "||F_0|| then needs to drop by 10 orders of magnitude "
    "to reach rtol=1e-9 — impossible. fenics's NewtonSolver "
    "guards this with an absolute_tolerance fallback; "
    "dealii's hand-coded steppers usually don't. Symptom: "
    "Newton stalls at iter 3-5 with residual ~1e-8 in "
    "dealii, while fenics's NewtonSolver reports converged "
    "at the same residual.",
    "[Cross-Backend][Solver] Line-search defaults: "
    "fenics/dolfinx's PETSc SNES (NonlinearProblem, type "
    "newtonls) already defaults to the backtracking line search "
    "'bt'; 'basic' (the full Newton step, no backtracking) must "
    "be requested with snes_linesearch_type='basic'. "
    "kratos's ResidualBasedNewtonRaphsonStrategy with "
    "line_search: true uses a backtracking line search. "
    "dealii has no SolverNewton class: hand-written Newton "
    "loops (step-15 pattern) choose their own step length "
    "(step-15 uses a fixed 0.1), so any line search is yours to "
    "implement, while the library wrappers SUNDIALS::KINSOL and "
    "NonlinearSolverSelector default to strategy = linesearch. "
    "For a buckling problem near "
    "the bifurcation point, a fenics SNES set to "
    "snes_linesearch_type='basic' can jump "
    "past the bifurcation and Newton diverges in one step; "
    "kratos's backtracking carries through smoothly. Fix: "
    "keep (or set) snes_linesearch_type='bt' in the PETSc SNES "
    "options.",
    "[Cross-Backend][Solver] Convergence-failure signal "
    "path: NGSolve's Newton (ngsolve.solvers.Newton) does not "
    "raise on max-iter-exceeded: it prints 'Warning: Newton might "
    "not converge! Error = ...' and returns (-1, numit) ((0, "
    "numit) on success); FEBio writes '------- failed to "
    "converge at time : <t>' and 'E R R O R   T E R M I N A T I O N' "
    "to the .log and exits with code 1 "
    "(no Python-level signal in any wrapper); "
    "fenics/dolfinx's deprecated NewtonSolver raises "
    "RuntimeError ('Newton solver did not converge because "
    "maximum number of iterations reached') by default and "
    "returns (n_iters, False) only with "
    "error_on_nonconvergence=False, while dolfinx 0.10's "
    "NonlinearProblem.solve() returns the solution Function "
    "without raising (unless snes_error_if_not_converged is "
    "set). A script looping over load steps with "
    "NonlinearProblem will silently continue with an "
    "unconverged displacement after a failed step unless it "
    "checks problem.solver.getConvergedReason() > 0; the same "
    "script also continues silently in NGSolve unless it "
    "checks the returned status; in FEBio there "
    "is no Python-level signal at all (the binary process "
    "exits before any Python wrapper sees it).",
]
_NONLINEAR_CONVERGENCE_SIGNAL = (
    "[Cross-Backend][Solver] 'Converged' means different "
    "things across backends — print all of (||F||, ||du||, "
    "||du||/||u||, energy) per iter, set tolerances "
    "absolutely (not relative-only) to avoid the zero-load "
    "trap, and on convergence FAILURE be sure your script "
    "handles each backend's specific signal: returned "
    "status (NGSolve's Newton returns (-1, numit) and only "
    "prints a warning), RuntimeError from the deprecated "
    "NewtonSolver or a non-positive SNES converged reason from "
    "NonlinearProblem (fenics), process "
    "exit-code (FEBio), a False return from SolveSolutionStep() "
    "with a logged 'ATTENTION: max iterations ( <n> ) exceeded!' "
    "(kratos), or, in dealii, "
    "SolverControl::NoConvergence from an inner Krylov solve "
    "(SUNDIALS::ExcKINSOLError when the nonlinear solve goes "
    "through KINSOL; a hand-written Newton loop throws nothing "
    "on its own). A missing exception does NOT mean "
    "convergence."
)


CROSS_BACKEND_PITFALLS = {
    "units": {
        "description": _UNITS_DESC,
        "pitfalls": _UNITS_PITFALLS,
        "Signal": _UNITS_SIGNAL,
    },
    "element_node_ordering": {
        "description": _NODE_ORDER_DESC,
        "pitfalls": _NODE_ORDER_PITFALLS,
        "Signal": _NODE_ORDER_SIGNAL,
    },
    "linear_elastic_semantics": {
        "description": _LE_SEMANTICS_DESC,
        "pitfalls": _LE_SEMANTICS_PITFALLS,
        "Signal": _LE_SEMANTICS_SIGNAL,
    },
    "dirichlet_bc_enforcement": {
        "description": _DIRICHLET_DESC,
        "pitfalls": _DIRICHLET_PITFALLS,
        "Signal": _DIRICHLET_SIGNAL,
    },
    "restart_checkpoint_compatibility": {
        "description": _RESTART_DESC,
        "pitfalls": _RESTART_PITFALLS,
        "Signal": _RESTART_SIGNAL,
    },
    "mpi_launch_idioms": {
        "description": _MPI_DESC,
        "pitfalls": _MPI_PITFALLS,
        "Signal": _MPI_SIGNAL,
    },
    "element_type_naming": {
        "description": _ELEMENT_TYPE_DESC,
        "pitfalls": _ELEMENT_TYPE_PITFALLS,
        "Signal": _ELEMENT_TYPE_SIGNAL,
    },
    "time_integration_defaults": {
        "description": _TIME_INTEGRATION_DESC,
        "pitfalls": _TIME_INTEGRATION_PITFALLS,
        "Signal": _TIME_INTEGRATION_SIGNAL,
    },
    "solver_tolerance_defaults": {
        "description": _TOLERANCE_DESC,
        "pitfalls": _TOLERANCE_PITFALLS,
        "Signal": _TOLERANCE_SIGNAL,
    },
    "contact_formulation_defaults": {
        "description": _CONTACT_FORMULATION_DESC,
        "pitfalls": _CONTACT_FORMULATION_PITFALLS,
        "Signal": _CONTACT_FORMULATION_SIGNAL,
    },
    "output_format_conventions": {
        "description": _OUTPUT_FORMAT_DESC,
        "pitfalls": _OUTPUT_FORMAT_PITFALLS,
        "Signal": _OUTPUT_FORMAT_SIGNAL,
    },
    "integration_order_defaults": {
        "description": _INTEGRATION_ORDER_DESC,
        "pitfalls": _INTEGRATION_ORDER_PITFALLS,
        "Signal": _INTEGRATION_ORDER_SIGNAL,
    },
    "boundary_tag_semantics": {
        "description": _BOUNDARY_TAG_DESC,
        "pitfalls": _BOUNDARY_TAG_PITFALLS,
        "Signal": _BOUNDARY_TAG_SIGNAL,
    },
    "plasticity_return_mapping": {
        "description": _PLASTICITY_RETURN_MAP_DESC,
        "pitfalls": _PLASTICITY_RETURN_MAP_PITFALLS,
        "Signal": _PLASTICITY_RETURN_MAP_SIGNAL,
    },
    "turbulence_model_defaults": {
        "description": _TURBULENCE_DESC,
        "pitfalls": _TURBULENCE_PITFALLS,
        "Signal": _TURBULENCE_SIGNAL,
    },
    "material_orientation_defaults": {
        "description": _MATERIAL_ORIENTATION_DESC,
        "pitfalls": _MATERIAL_ORIENTATION_PITFALLS,
        "Signal": _MATERIAL_ORIENTATION_SIGNAL,
    },
    "frequency_unit_conventions": {
        "description": _FREQUENCY_DESC,
        "pitfalls": _FREQUENCY_PITFALLS,
        "Signal": _FREQUENCY_SIGNAL,
    },
    "mesh_quality_thresholds": {
        "description": _MESH_QUALITY_DESC,
        "pitfalls": _MESH_QUALITY_PITFALLS,
        "Signal": _MESH_QUALITY_SIGNAL,
    },
    "stress_measure_conventions": {
        "description": _STRESS_MEASURE_DESC,
        "pitfalls": _STRESS_MEASURE_PITFALLS,
        "Signal": _STRESS_MEASURE_SIGNAL,
    },
    "periodic_bc_implementation": {
        "description": _PERIODIC_BC_DESC,
        "pitfalls": _PERIODIC_BC_PITFALLS,
        "Signal": _PERIODIC_BC_SIGNAL,
    },
    "damping_convention_defaults": {
        "description": _DAMPING_DESC,
        "pitfalls": _DAMPING_PITFALLS,
        "Signal": _DAMPING_SIGNAL,
    },
    "timestamp_output_conventions": {
        "description": _TIMESTAMP_DESC,
        "pitfalls": _TIMESTAMP_PITFALLS,
        "Signal": _TIMESTAMP_SIGNAL,
    },
    "initial_condition_interpolation": {
        "description": _INITIAL_CONDITION_DESC,
        "pitfalls": _INITIAL_CONDITION_PITFALLS,
        "Signal": _INITIAL_CONDITION_SIGNAL,
    },
    "frame_of_reference_bc": {
        "description": _FRAME_OF_REFERENCE_DESC,
        "pitfalls": _FRAME_OF_REFERENCE_PITFALLS,
        "Signal": _FRAME_OF_REFERENCE_SIGNAL,
    },
    "linear_solver_defaults": {
        "description": _LINEAR_SOLVER_DESC,
        "pitfalls": _LINEAR_SOLVER_PITFALLS,
        "Signal": _LINEAR_SOLVER_SIGNAL,
    },
    "nonlinear_convergence_criteria": {
        "description": _NONLINEAR_CONVERGENCE_DESC,
        "pitfalls": _NONLINEAR_CONVERGENCE_PITFALLS,
        "Signal": _NONLINEAR_CONVERGENCE_SIGNAL,
    },
}


def get_cross_backend_pitfalls(topic: str | None = None) -> dict:
    """Return the cross-backend pitfalls structure.

    If `topic` is provided (e.g. 'units', 'mesh', 'bc'), filter to
    matching entries. Topic matching is a case-insensitive substring
    against the entry key or description.
    """
    if not topic:
        return CROSS_BACKEND_PITFALLS
    t = topic.lower()
    matched = {}
    for key, entry in CROSS_BACKEND_PITFALLS.items():
        if t in key.lower() or t in entry.get("description", "").lower():
            matched[key] = entry
    return matched if matched else CROSS_BACKEND_PITFALLS
