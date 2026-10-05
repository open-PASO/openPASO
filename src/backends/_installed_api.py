"""Installed-version API references — VERIFIED by actually running on this machine.

Models repeatedly fail not on the physics but on writing API calls for the WRONG
version of the installed code (e.g. NGSolve Integrate() signature, deal.II pointing
DEAL_II_DIR at the source instead of the build tree, the 4C YAML schema). Each entry
below is a minimal smoke test that was WRITTEN, RUN, and FIXED until it executed
cleanly on the version installed here, plus the version-specific gotchas observed.

These are GENERAL API references (a trivial -Laplacian(u)=1 / single-cube run), NOT
solutions to any benchmark — they teach the correct installed-version API so an agent
adapts a known-good call instead of guessing from (outdated) memory.

Surfaced to agents via prepare_simulation()/knowledge() for the matching backend.
Keyed by the backend registry name.
"""

INSTALLED_API = {
 # ── FEniCSx ────────────────────────────────────────────────────────────
 # There was NO fenics entry, while it is a backend that single-code tasks
 # use. Point evaluation in dolfinx is a genuine three-step API and the
 # recipe lived only under physics='contact' and physics='io_catalog' —
 # neither a name an agent solving a Poisson problem would request.
 "fenics": {
  "version": "0.10.0 (dolfinx)",
  "run": "{FENICS_PYTHON} <script>.py",
  "verified_smoke_test": (
    "import numpy as np, ufl\n"
    "from dolfinx import fem, mesh as dm, geometry\n"
    "from dolfinx.fem.petsc import LinearProblem\n"
    "from mpi4py import MPI\n"
    "msh = dm.create_unit_square(MPI.COMM_WORLD, 16, 16)\n"
    "V = fem.functionspace(msh, ('Lagrange', 1))\n"
    "u, v = ufl.TrialFunction(V), ufl.TestFunction(V)\n"
    "a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx\n"
    "L = fem.Constant(msh, 1.0) * v * ufl.dx\n"
    "bdofs = fem.locate_dofs_topological(V, 1, dm.locate_entities_boundary(\n"
    "    msh, 1, lambda x: np.full(x.shape[1], True)))\n"
    "uh = LinearProblem(a, L, bcs=[fem.dirichletbc(0.0, bdofs, V)],\n"
    "                   petsc_options_prefix='s',\n"
    "                   petsc_options={'ksp_type':'preonly','pc_type':'lu'}).solve()\n"
    "pts = np.array([[0.3123, 0.7311, 0.0], [0.5, 0.5, 0.0]])\n"
    "tree = geometry.bb_tree(msh, msh.topology.dim)\n"
    "coll = geometry.compute_colliding_cells(\n"
    "    msh, geometry.compute_collisions_points(tree, pts), pts)\n"
    "cells = [coll.links(i)[0] for i in range(len(pts))]\n"
    "vals = uh.eval(pts, cells)\n"
    "np.savetxt('out.csv', np.hstack([pts[:, :2], np.asarray(vals).reshape(-1,1)]),\n"
    "           delimiter=',', header='x,y,u', comments='', fmt='%.15e')\n"
    "print('center:', float(np.asarray(vals).ravel()[1]))   # -> ~0.0734\n"),
  "gotchas": [
    "POINT EVALUATION IS THREE CALLS, not one: geometry.bb_tree -> geometry.compute_collisions_points -> geometry.compute_colliding_cells, then Function.eval(points, cells). There is no u(x, y).",
    "The points array needs THREE columns even in 2-D; pass z = 0.0.",
    "compute_colliding_cells returns an adjacency list: take .links(i)[0]. A point outside the mesh has an EMPTY list, so indexing blindly raises IndexError rather than returning a wrong number.",
    "The old name compute_collisions does NOT exist in 0.10; it is compute_collisions_points.",
    "Writing results: numpy.savetxt(..., fmt='%.15e'). str(float) and '%.6f' throw away digits a convergence study needs.",
    "LinearProblem in 0.10 requires petsc_options_prefix.",
  ],
 },
 # ── DUNE-fem ───────────────────────────────────────────────────────────
 # No entry existed. pointSample IS in the corpus, under a key whose own
 # section index calls it "VTK options ... thread defaults".
 "dune": {
  "version": "2.12.0.2 (dune-fem)",
  "run": "{DUNE_PYTHON} <script>.py",
  "verified_smoke_test": (
    "import numpy as np\n"
    "from dune.grid import structuredGrid\n"
    "from dune.fem.space import lagrange\n"
    "from dune.fem.scheme import galerkin\n"
    "from dune.fem.utility import pointSample\n"
    "from dune.ufl import DirichletBC\n"
    "from ufl import TrialFunction, TestFunction, dot, grad, dx\n"
    "g = structuredGrid([0, 0], [1, 1], [16, 16])\n"
    "sp = lagrange(g, order=1)\n"
    "u, v = TrialFunction(sp), TestFunction(sp)\n"
    "sch = galerkin([dot(grad(u), grad(v)) * dx == 1.0 * v * dx,\n"
    "                DirichletBC(sp, 0)], solver='cg')\n"
    "uh = sp.interpolate(0, name='u')\n"
    "sch.solve(target=uh)\n"
    "pts = [[0.3123, 0.7311], [0.5, 0.5]]\n"
    "vals = [float(pointSample(uh, p)) for p in pts]\n"
    "np.savetxt('out.csv', np.hstack([np.array(pts), np.array(vals).reshape(-1,1)]),\n"
    "           delimiter=',', header='x,y,u', comments='', fmt='%.15e')\n"
    "print('center:', vals[1])       # -> ~0.0739\n"),
  "gotchas": [
    "POINT EVALUATION: `from dune.fem.utility import pointSample; pointSample(uh, [x, y])` gives the value at an arbitrary point. lineSample and a Sampler class exist too. uh(x, y) is not it.",
    "DUNE JIT-COMPILES C++ on first use: the first structuredGrid or solve in a fresh environment takes minutes. That is normal, not a hang.",
    "A STALE JIT CACHE FAILS AT USE, NOT AT IMPORT. `import dune.fem` can succeed while the first structuredGrid dies with 'undefined symbol: PyThreadState_GetUnchecked'. That means the cache was built against a different Python.",
    "Writing results: numpy.savetxt(..., fmt='%.15e').",
  ],
 },
 # ── Kratos ─────────────────────────────────────────────────────────────
 # No entry existed, and the installed Kratos ships exactly what these tasks
 # need — shape-function interpolation at arbitrary points — mentioned
 # nowhere in the served knowledge.
 "kratos": {
  "version": "10.3.0",
  "run": "{PYTHON} <script>.py",
  "verified_smoke_test": (
    "import KratosMultiphysics as KM\n"
    "m = KM.Model(); mp = m.CreateModelPart('m')\n"
    "mp.AddNodalSolutionStepVariable(KM.TEMPERATURE)\n"
    "for i, (x, y) in enumerate([(0,0), (1,0), (1,1), (0,1)], 1):\n"
    "    mp.CreateNewNode(i, float(x), float(y), 0.0)\n"
    "props = mp.CreateNewProperties(1)\n"
    "mp.CreateNewElement('Element2D3N', 1, [1, 2, 3], props)\n"
    "mp.CreateNewElement('Element2D3N', 2, [1, 3, 4], props)\n"
    "for n in mp.Nodes:\n"
    "    n.SetSolutionStepValue(KM.TEMPERATURE, 3.0 * n.X + 2.0 * n.Y)\n"
    "loc = KM.BinBasedFastPointLocator2D(mp); loc.UpdateSearchDatabase()\n"
    "for px, py in [[0.3123, 0.7311], [0.5, 0.5]]:\n"
    "    found, N, el = loc.FindPointOnMesh(KM.Array3([px, py, 0.0]))\n"
    "    val = sum(N[k] * nd.GetSolutionStepValue(KM.TEMPERATURE)\n"
    "              for k, nd in enumerate(el.GetNodes()))\n"
    "    print(px, py, found, val)   # exact for a linear field\n"),
  "gotchas": [
    "POINT EVALUATION: KM.BinBasedFastPointLocator2D(mp) (or 3D), UpdateSearchDatabase(), then FindPointOnMesh. In 10.3 it RETURNS A TUPLE (found, shape_functions, element) and does not take out-params. Multiply the shape functions by the element's nodal values: that is a real interpolation, not a nearest-node read.",
    "Kratos also ships point_output_process, multiple_points_output_process and csv_points_output_process (all importable here); the last takes a list of (x,y,z) and writes interpolated values straight to CSV.",
    "Element names are positional strings: 'Element2D3N' is a linear triangle. Nodes are 1-based and must exist before the element.",
    "Writing results: numpy.savetxt(..., fmt='%.15e').",
  ],
 },
 # ── FEBio ──────────────────────────────────────────────────────────────
 # No entry existed. Its honest answer is a product limit, and saying so
 # beats silence: an agent that knows will place nodes AT the probe points
 # instead of hunting an API that is not there.
 "febio": {
  "version": "4.12.0",
  "run": "{FEBIO_BINARY} -i <deck>.feb",
  "verified_smoke_test": (
    "# FEBio writes results from the DECK, not from a Python API.\n"
    "# <Output>\n"
    "#   <logfile>\n"
    "#     <node_data file='u.csv' delim=',' data='ux;uy'/>\n"
    "#     (no format=: FEBio then writes id,ux,uy with 12 significant digits;\n"
    "#      printf specifiers such as %.15g are not supported)\n"
    "#   </logfile>\n"
    "# </Output>\n"),
  "gotchas": [
    "NO ARBITRARY-POINT EVALUATION INSIDE FEBio. It exposes NODAL output and no user-facing shape-function interpolation, so the interpolation is YOURS to do afterwards: export node_data, and evaluate at your target points in Python from the nodal values and the element they fall in. On a structured mesh that is bilinear (2-D) or trilinear (3-D) interpolation inside the containing cell — locate the cell from the mesh spacing, then weight its corner values.",
    "DO NOT MOVE THE MESH TO THE PROBE POINTS. If a task prescribes BOTH a mesh sequence and a probe grid, the mesh is part of the problem and refitting it to make probes land on nodes is solving a different problem — it reads as not following the prescribed sequence. Probe grids are commonly chosen to be deliberately off-node precisely so that interpolation is exercised.",
    "A deck with only <plotfile> writes a binary .xplt and nothing readable. Add <logfile> with node_data or you have no numbers to deliver.",
    "The logfile format string cannot set precision: it only knows %i/%l/%g/%t/%n, %.15g is printed literally and the values are lost, and %g gives 6 significant digits. Omit format= (use delim=',') to get 12 significant digits.",
    "A log accumulates one block per step: parse the LAST block.",
    "This build has no pardiso; the default is skyline, which takes only symmetric matrices. That is fine for a default (symmetric) solid deck, but the solvers of the other modules (biphasic, solute, multiphasic and the fluid modules) default to a non-symmetric stiffness and abort with 'The selected linear solver does not support the requested matrix format': put <linear_solver type=\"bicgstab\"/> inside <solver>.",
  ],
 },
 # ── SPARTA ─────────────────────────────────────────────────────────────
 # No entry existed; the precision fix lived only inside a method the server
 # does not expose as a tool.
 "sparta": {
  "version": "DSMC, spa_serial",
  "run": "<sparta binary> -in in.<case>",
  "verified_smoke_test": (
    "# SPARTA is an input-script code and OUTPUT PRECISION is the trap:\n"
    "#   dump_modify  1 format float %20.15g\n"
    "#   stats_modify format float %20.15g\n"
    "# A compute produces NOTHING by itself: stats_style, a fix ave/time, a dump\n"
    "# or a print must reference it (c_<id>; a print as $(c_<id>)) for any\n"
    "# number to be written at all.\n"),
  "gotchas": [
    "OUTPUT PRECISION: `dump_modify <id> format float %20.15g` and `stats_modify format float %20.15g`. The defaults are far too coarse to compare against a reference.",
    "A `compute` writes nothing on its own; something that writes must reference it: stats_style (c_<id>), a fix such as ave/time, a dump, or a print ($(c_<id>), or ${v} of an equal-style variable v = c_<id>).",
    "DSMC is STOCHASTIC: one run is a sample. Average over enough steps after the flow is established, and say which window you averaged.",
    "THERE IS NO POINT EVALUATION, and that is the physics, not a gap: DSMC "
    "carries no continuous field to interpolate. Output is per surface "
    "element or per grid cell, sampled and time-averaged. If a task names "
    "probe locations, build the surface mesh so its ELEMENTS sit at those "
    "locations and report the element values, saying which averaging window "
    "you used.",
    "The full reference ships with the source at $SPARTA_ROOT/doc.",
  ],
 },
 "ngsolve": {
  "version": "6.2.2604",
  "run": "{PYTHON} <script>.py",
  "verified_smoke_test": (
    "from ngsolve import *\n"
    "from netgen.geom2d import unit_square\n"
    "mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))\n"
    "fes = H1(mesh, order=1, dirichlet='.*')\n"
    "u, v = fes.TnT()\n"
    "a = BilinearForm(fes); a += grad(u)*grad(v)*dx; a.Assemble()\n"
    "f = LinearForm(fes); f += 1*v*dx; f.Assemble()\n"
    "gfu = GridFunction(fes)\n"
    "gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse='sparsecholesky') * f.vec\n"
    "print('center value:', gfu(mesh(0.5, 0.5)))   # -> ~0.069\n"),
  "gotchas": [
    "Mesh: do NOT call Mesh() on a unit square; use `from netgen.geom2d import unit_square; Mesh(unit_square.GenerateMesh(maxh=0.2))`.",
    "Dirichlet BC goes on the SPACE: H1(mesh, order=1, dirichlet='.*'); '.*' matches all (unnamed) boundaries robustly.",
    "Trial/test: `u, v = fes.TnT()`. Forms: `a += grad(u)*grad(v)*dx` then `a.Assemble()` (explicit).",
    "Solve: `gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse='sparsecholesky') * f.vec` — FreeDofs() enforces the Dirichlet constraint.",
    "Point eval: `gfu(mesh(0.5,0.5))` (canonical). A plain GridFunction also accepts `gfu(0.5,0.5)` (same value), but a general CoefficientFunction such as `2*gfu` must go through mesh(...): `cf(0.5,0.5)` returns a PointEvaluationFunctional, not a number.",
    "Writing results: numpy.savetxt(..., fmt='%.15e'). str(float) and '%.6f' throw away digits a convergence study needs.",
    "Integrate(cf*dx, mesh) is a SEPARATE functional, not how you assemble forms.",
  ],
 },
 "skfem": {
  "version": "12.0.1",
  "run": "{PYTHON} <script>.py",
  "verified_smoke_test": (
    "import numpy as np\n"
    "from skfem import MeshTri, ElementTriP1, Basis, BilinearForm, LinearForm, asm, condense, solve\n"
    "from skfem.helpers import dot, grad\n"
    "mesh = MeshTri().refined(4)          # MeshTri() IS the unit square; .refined returns a NEW mesh\n"
    "basis = Basis(mesh, ElementTriP1())\n"
    "@BilinearForm\n"
    "def stiffness(u, v, w): return dot(grad(u), grad(v))\n"
    "@LinearForm\n"
    "def load(v, w): return 1.0 * v\n"
    "A = stiffness.assemble(basis); b = load.assemble(basis)\n"
    "D = basis.get_dofs()                  # all boundary dofs\n"
    "x = solve(*condense(A, b, D=D))\n"
    "print('max u =', x.max())            # -> ~0.073\n"),
  "gotchas": [
    "MeshTri() already IS the unit square; refine with `.refined(n)` (returns a new mesh, not in-place).",
    "Use generic `Basis(mesh, ElementTriP1())` (element instance required).",
    "Decorator arities are mandatory: `@BilinearForm def f(u, v, w)` and `@LinearForm def f(v, w)` — the trailing `w` (global params) is required even if unused. Wrong arity is the #1 error.",
    "POINT EVALUATION at arbitrary (non-node) points: `basis.probes(P) @ x`, where P is a (dim, npoints) array — probes() returns a sparse matrix that INTERPOLATES the solution vector, so the `@` is the evaluation. Note the shape: points are COLUMNS, not rows. Verified exact on a P1-representable field.",
    "Writing results: numpy.savetxt(..., fmt='%.15e'). str(float) and '%.6f' throw away digits a convergence study needs.",
    "Integrand uses skfem.helpers (dot, grad) and returns the pointwise integrand, not an assembled value.",
    "Assemble via `form.assemble(basis)` or `asm(form, basis)` (equivalent).",
    "Dirichlet: `D = basis.get_dofs()` (no args = all boundary dofs) then `solve(*condense(A, b, D=D))` (homogeneous by default).",
  ],
 },
 "dealii": {
  "version": "9.8.0-pre  (build tree at {DEALII_BUILD})",
  "run": "cmake -DDEAL_II_DIR={DEALII_BUILD} . && make && LD_LIBRARY_PATH=/opt/4C-dependencies/lib ./<exe>",
  "verified_smoke_test": (
    "// CMakeLists.txt:\n"
    "//   CMAKE_MINIMUM_REQUIRED(VERSION 3.13.4)\n"
    "//   FIND_PACKAGE(deal.II 9.0 REQUIRED HINTS ${DEAL_II_DIR})\n"
    "//   DEAL_II_INITIALIZE_CACHED_VARIABLES()   # BEFORE project()\n"
    "//   PROJECT(poisson CXX)\n"
    "//   ADD_EXECUTABLE(poisson poisson.cc); DEAL_II_SETUP_TARGET(poisson)\n"
    "// poisson.cc: standard step-3 Poisson, Q1, refine_global(5), -Laplacian(u)=1, u=0 on bdry.\n"
    "//   Functions::ZeroFunction<dim>() for the BC; VectorTools::point_value(dof_handler, solution,\n"
    "//   Point<dim>(0.5,0.5)) -> ~0.0737 ; SolverCG + PreconditionIdentity.\n"),
  "gotchas": [
    "CRITICAL: DEAL_II_DIR must be the BUILD tree `{DEALII_BUILD}` (config at .../build/lib/cmake/deal.II). Pointing at `{DEALII_ROOT}` SILENTLY falls back to the OLD system install (9.1.1 at /usr) with no error — check cmake's `-- Using the deal.II-X found at ...` line.",
    "Runtime needs `LD_LIBRARY_PATH=/opt/4C-dependencies/lib` (shared TBB/etc).",
    "CMake order: FIND_PACKAGE(deal.II 9.0 REQUIRED HINTS ${DEAL_II_DIR}) -> DEAL_II_INITIALIZE_CACHED_VARIABLES() -> PROJECT() -> DEAL_II_SETUP_TARGET(<tgt>). INITIALIZE must precede PROJECT().",
    "Use modern idioms: fe_values.quadrature_point_indices(), fe_values.dof_indices(), fe.n_dofs_per_cell() (preferred accessor; the public member fe.dofs_per_cell still exists and is not deprecated).",
    "Functions::ZeroFunction<dim>() (namespaced; bare ZeroFunction removed). Header <deal.II/base/function.h>.",
    "Sparsity needs both <.../dynamic_sparsity_pattern.h> and <.../sparsity_pattern.h>; BCs need <.../numerics/vector_tools.h> + <.../numerics/matrix_tools.h>.",
    "Evaluate: VectorTools::point_value(dof_handler, solution, Point<dim>(...)); solution.linfty_norm() for the max. This works at ARBITRARY points, not just nodes — it locates the cell and applies the shape functions. Functions::FEFieldFunction is the batch version.",
    "Writing results from C++: `out << std::setprecision(15) << std::scientific`. The default ostream precision is 6 digits and loses what a convergence study needs.",
  ],
 },
 "fourc": {
  "version": "build at {FOURC_BINARY}",
  "run": "{FOURC_ENV}{FOURC_BINARY} <input>.4C.yaml <output_prefix>",
  "verified_smoke_test": (
    "# Minimal single HEX8 linear-elastic cube (fixed at x=0, pulled at x=1), Statics, 2 steps.\n"
    "# Started from {FOURC_ROOT}/tests/input_files/solid_runtime_material_element_id.4C.yaml\n"
    "# Runs to completion: stdout ends 'processor 0 finished normally' / EXIT:0; writes <prefix>.control + VTK.\n"
    "PROBLEM TYPE: {PROBLEMTYPE: 'Structure'}\n"
    "SOLVER 1: {SOLVER: 'Superlu', NAME: 'Structure_Solver'}\n"
    "STRUCTURAL DYNAMIC: {DYNAMICTYPE: 'Statics', TIMESTEP: 0.5, NUMSTEP: 2, MAXTIME: 1,\n"
    "                     TOLDISP: 1e-9, TOLRES: 1e-9, LINEAR_SOLVER: 1}\n"
    "MATERIALS: [{MAT: 1, MAT_Struct_StVenantKirchhoff: {YOUNG: 100, NUE: 0.3, DENS: 0}}]\n"
    "FUNCT1: [{SYMBOLIC_FUNCTION_OF_SPACE_TIME: 't'}]\n"
    "DESIGN SURF DIRICH CONDITIONS: [{E: 1, NUMDOF: 3, ONOFF: [1,1,1], VAL: [0,0,0], FUNCT: [0,0,0]}]\n"
    "DESIGN SURF NEUMANN CONDITIONS: [{E: 2, NUMDOF: 3, ONOFF: [1,0,0], VAL: [10,0,0], FUNCT: [1,0,0]}]\n"
    "DSURF-NODE TOPOLOGY: ['NODE 1 DSURFACE 1', ... 'NODE 5 DSURFACE 2', ...]\n"
    "NODE COORDS: ['NODE 1 COORD 0.0 0.0 0.0', ...8 nodes...]\n"
    "STRUCTURE ELEMENTS: ['1 SOLID HEX8 1 5 6 2 3 7 8 4 MAT 1 KINEM nonlinear']\n"),
  "gotchas": [
    "Run: `{FOURC_ENV}{FOURC_BINARY} <in>.4C.yaml <output_prefix>` — the output prefix is MANDATORY.",
    "File is one YAML map; keys are section names with spaces/slashes (e.g. 'STRUCTURAL DYNAMIC', 'IO/RUNTIME VTK OUTPUT/STRUCTURE').",
    "Required minimal: PROBLEM TYPE, SOLVER 1, STRUCTURAL DYNAMIC, MATERIALS, mesh sections, conditions.",
    "Time integrator references the linear solver via LINEAR_SOLVER: 1 (-> 'SOLVER 1').",
    "Materials: list, each {MAT: <id>, MAT_Struct_<Model>: {...}}; elements reference it by `MAT <id>`.",
    "Mesh is INLINE strings: NODE COORDS = ['NODE <id> COORD x y z', ...]; STRUCTURE ELEMENTS = ['<id> SOLID HEX8 <8 ids> MAT <id> KINEM nonlinear', ...] (raw strings, not maps).",
    "BCs attach to DESIGN entities, not nodes: DSURF-NODE TOPOLOGY maps nodes->surface ids ('NODE n DSURFACE d'); conditions reference d via E: d.",
    "Dirichlet/Neumann: lists of {E, NUMDOF, ONOFF: [...], VAL: [...], FUNCT: [...]}; ONOFF flags active dofs, FUNCT indexes a FUNCTn (0 = constant).",
    "Omit the optional RESULT DESCRIPTION section for a pure smoke run so regression asserts can't fail the job.",
    "Success = stdout 'processor 0 finished normally' / EXIT:0 and a <prefix>.control output file.",
  ],
  # Additional VERIFIED capability references (each written-run-fixed on this 4C, generic
  # problems that teach the schema — NOT any benchmark solution).
  "capabilities": [
   {"name": "Elastoplasticity (von Mises / J2, isotropic hardening)",
    "facts": [
     "Material: MAT_Struct_PlasticLinElast {YOUNG, NUE, DENS, YIELD (initial yield stress), ISOHARD (linear isotropic hardening modulus, stress units), KINHARD (kinematic; 0 for pure isotropic), TOL (return-mapping Newton tol)}. This is a SMALL-STRAIN J2 model -> pair it with `KINEM linear` on the element. (Finite-strain: MAT_Struct_PlasticNlnLogNeoHooke; ductile damage: MAT_Struct_PlasticGTN.)",
     "CONVERGENCE GOTCHA: a single load jump well past yield stalls the local return-mapping Newton. RAMP the load over several steps (e.g. NUMSTEP 5, FUNCT 't') so only ~1% plastic strain accrues per step -> nlniter jumps 2->4 at yield and converges.",
     "Use a robust direct linear solver for tiny meshes (SOLVER 1: {SOLVER: 'UMFPACK'}). Apply load by prescribed displacement (DESIGN SURF DIRICH), fully clamp the opposite face to kill rigid-body modes.",
     "Confirm yielding: IO {STRUCT_STRAIN/STRUCT_PLASTIC_STRAIN: 'EA', STRUCT_STRESS: 'Cauchy'} + runtime VTK STRESS_STRAIN:true; compare von Mises stress to YIELD.",
    ]},
   {"name": "2D structural element + EDGE (line) traction",
    "facts": [
     "2D element before 4C 2026.2.0: `<id> WALL QUAD4 <4 CCW node ids> MAT <id> KINEM linear EAS none THICK <t> STRESS_STRAIN plane_stress GP 2 2`. From 4C 2026.2.0 on (no WALL): `<id> SOLID QUAD4 <4 CCW node ids> MAT <id> KINEM linear THICKNESS <t> PLANE_ASSUMPTION plane_stress`. Missing/garbled tail params abort at input parse.",
     "EDGE traction in 2D attaches to LINES, not surfaces: use `DESIGN LINE NEUMANN CONDITIONS` (E: d) + a `DLINE-NODE TOPOLOGY` block ('NODE n DLINE d') for the loaded edge; clamp via `DESIGN LINE DIRICH CONDITIONS` + its own DLINE. NUMDOF 6, ONOFF[0]=1 turns on x-traction, VAL is traction per unit edge length (x THICK).",
     "ZERO-DISPLACEMENT GOTCHA (the classic 2D trap): if you attach the load to a DSURFACE in 2D, or DLINE-NODE TOPOLOGY is missing/points at the wrong nodes, 4C STILL runs and exits 0 but the load set is EMPTY -> displacement is zero everywhere. Always confirm every loaded-edge node appears under the Neumann DLINE.",
     "Dirichlet on the clamped edge must constrain BOTH dofs (ONOFF [1,1]) or the body is under-constrained.",
    ]},
   {"name": "Reading results out of 4C output (extract a nodal value)",
    "facts": [
     "By DEFAULT 4C writes only a BINARY <prefix>.control + binary result files — NOT human-readable. Do NOT try to hex-decode them by hand.",
     "To get readable output, add: `IO/RUNTIME VTK OUTPUT: {INTERVAL_STEPS: 1, OUTPUT_DATA_FORMAT: ascii}` and `IO/RUNTIME VTK OUTPUT/STRUCTURE: {OUTPUT_STRUCTURE: true, DISPLACEMENT: true}` (add `STRESS_STRAIN: true` for stresses). This writes `<prefix>-vtk-files/structure-0000N-0.vtu`.",
     "Extract with pyvista (available in {PYTHON}): `import pyvista as pv, numpy as np; m = pv.read(LAST_vtu); d = np.asarray(m.point_data['displacement']); pts = m.points`. Find your node by coordinate: `i = np.argmin(np.linalg.norm(pts - target_xyz, axis=1))`, then `d[i]` is its displacement (stress/strain are cell or point arrays too).",
     "POINT EVALUATION AT NON-NODAL POINTS — and the nearest-node line above is NOT it. Snapping to the closest node is a first-order error that will not converge at the rate a P1/Q1 field does, so it is wrong for any prescribed probe grid that is deliberately off-mesh. 4C itself cannot help: its RESULT DESCRIPTION block selects by NODE/LINE/SURFACE/VOLUME and has no coordinate selector at all (checked against the live grammar dump, `4C -p`). Interpolate from the VTU instead: `probe = pv.PolyData(target_xyz_array); vals = probe.sample(pv.read(LAST_vtu))['displacement']` — pyvista's sample() locates the cell and applies the shape functions, verified exact on a linear field.",
     "Writing results: numpy.savetxt(..., fmt='%.15e'). str(float) and '%.6f' throw away digits a convergence study needs.",
     "GOTCHA: read the LAST timestep file (highest number, e.g. structure-00005-0.vtu), NOT structure-00000-0.vtu which is the INITIAL zero state -> reading step 0 gives displacement 0 everywhere and looks like the load did nothing.",
     "STRESS output: set `IO: {STRUCT_STRESS: 'Cauchy'}` (this is what actually enables stress) AND `IO/RUNTIME VTK OUTPUT/STRUCTURE: {..., STRESS_STRAIN: true}`. Then stress is in BOTH point_data['nodal_cauchy_stresses_xyz'] and cell_data['element_cauchy_stresses_xyz'], shape (n,6) Voigt [xx,yy,zz,xy,yz,xz] -> index 0 = sigma_xx. e.g. `pv.read(LAST_pvtu).cell_data['element_cauchy_stresses_xyz'][cell,0]`.",
     "Read the explicit last `.pvtu`/`.vtu` (e.g. sorted(glob('...structure-*.pvtu'))[-1]); pv.read('<prefix>-structure.pvd') is unreliable (its MultiBlock may expose only the t=0 zero block).",
    ]},
   {"name": "Locking-free 2D element (avoid volumetric locking at nu -> 0.5)",
    "facts": [
     "Standard QUAD4 (`EAS none`) volumetrically LOCKS at nearly-incompressible nu (e.g. 0.4999) and in bending -> displacement grossly under-predicted (verified: tip uy was 75x too small with EAS none).",
     "Fix: use Enhanced Assumed Strain. Before 4C 2026.2.0 -> `... WALL QUAD4 <nodes> MAT <id> KINEM nonlinear EAS full THICK <t> STRESS_STRAIN plane_strain GP 2 2`; 4C 2026.2.0 and later -> `... SOLID QUAD4 <nodes> MAT <id> KINEM nonlinear THICKNESS <t> PLANE_ASSUMPTION plane_strain TECH eas_full`. Both recover the flexible response, and give the same deflection (verified: nu 0.4999 cantilever, 20x2 QUAD4, tip uy 3.9e-5 without EAS and 2.92e-3 with it on a WALL build, on 2026.2.0 and on 2026.3.0, against 3.04e-3 from beam theory).",
     "GOTCHA: EAS REQUIRES `KINEM nonlinear`. Before 4C 2026.2.0 `KINEM linear EAS full` errors 'No EAS for geometrically linear WALL element'; from 4C 2026.2.0 on `KINEM linear ... TECH eas_full` stops with 'Your element formulation with cell type QUAD4, kinematic type linear, element technology eas_full and prestress type none does not exist'. For small-strain use KINEM nonlinear with a small load (geometrically ~linear).",
     "WALL (before 4C 2026.2.0) exposes only `EAS none|full` (no mild/F-bar); EAS is QUAD4-only (rejected for TRI6). The 2D SOLID of 4C 2026.2.0 and later likewise: QUAD4 alone takes TECH, and only none|eas_full (eas_mild and fbar give the same 'does not exist' message). The 3D SOLID element family offers additional F-bar/EAS variants.",
     "Use EAS full whenever nu -> 0.5 (incompressible: rubber, J2 plasticity flow, biomechanics) or for thin/bending-dominated low-order meshes.",
    ]},
  ],
 },
}


def render(backend_name: str) -> str:
    """Markdown block of the verified installed-version API reference for a backend, or ''."""
    e = INSTALLED_API.get(backend_name)
    if not e:
        return ""
    # THE PATHS SERVED HERE ARE THE READER'S, NOT THE AUTHOR'S.
    #
    # This reference is made by running each solver and recording what worked,
    # so it necessarily contains absolute paths -- and for a long time they were
    # one machine's. An audit measured 240 such paths across the 48 served 4C
    # payloads alone, stated as fact. A caveat was added telling the reader to
    # substitute their own, which does not help a model: it reads the path and
    # uses it verbatim, and on any other computer that command cannot run.
    #
    # The entries now hold tokens, and core.host_paths fills them from this
    # machine's environment, then from what autodiscovery found, and failing
    # both from a placeholder that names the variable to set. An honest
    # "<your dolfinx Python -- set FENICS_PYTHON>" is worth more than a
    # confident path that does not exist here.
    out = [f"## Installed-version API reference — {backend_name} {e['version']}",
           "*Measured by actually running each solver on the machine hosting "
           "this openPASO server. The paths below are filled in for THAT "
           "machine — they are local observations, not universal facts. Where "
           "one reads `<...set VARIABLE>`, openPASO could not find the "
           "install: set the named environment variable, or ask "
           "`knowledge(topic='install')`. The API shapes, versions and "
           "gotchas are the transferable part.*",
           f"Run (on this host): `{e['run']}`",
           "Minimal smoke test that ACTUALLY RUNS on this install (adapt this API; do not guess from memory):",
           "```", e["verified_smoke_test"].rstrip(), "```",
           "Version-specific gotchas:"]
    out += [f"- {g}" for g in e["gotchas"]]
    for cap in e.get("capabilities", []):
        out.append(f"\n### Verified capability: {cap['name']}")
        out += [f"- {f}" for f in cap["facts"]]
    from core.host_paths import resolve
    return resolve("\n".join(out))
