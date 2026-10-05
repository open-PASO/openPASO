"""DUNE-fem generator registry — maps physics_variant -> generator function."""

from .registry_survey import GENERATORS as _regsurv_gen, KNOWLEDGE as _regsurv_kn
from .poisson import GENERATORS as _poisson_gen, KNOWLEDGE as _poisson_kn
from .poisson_mms3d import GENERATORS as _poisson3d_gen, KNOWLEDGE as _poisson3d_kn
from .heat import GENERATORS as _heat_gen, KNOWLEDGE as _heat_kn
from .linear_elasticity import GENERATORS as _elast_gen, KNOWLEDGE as _elast_kn
from .stokes import GENERATORS as _stokes_gen, KNOWLEDGE as _stokes_kn
from .reaction_diffusion import GENERATORS as _rxn_gen, KNOWLEDGE as _rxn_kn
from .nonlinear import GENERATORS as _nonlinear_gen, KNOWLEDGE as _nonlinear_kn
from .dg_advection import GENERATORS as _dg_gen, KNOWLEDGE as _dg_kn
from .dg_advection_diffusion import (
    GENERATORS as _dg_diffusion_gen, KNOWLEDGE as _dg_diffusion_kn)
from .adaptive_poisson import GENERATORS as _adaptive_gen, KNOWLEDGE as _adaptive_kn
from .advanced import GENERATORS as _advanced_gen, KNOWLEDGE as _advanced_kn
from .verified_api import EXECUTED_API as _executed_api

# Merged generator registry: physics_variant -> callable(params) -> str
GENERATORS: dict[str, callable] = {}
for _g in [
    _regsurv_gen, _poisson_gen, _poisson3d_gen, _heat_gen, _elast_gen, _stokes_gen,
    _rxn_gen, _nonlinear_gen, _dg_gen, _dg_diffusion_gen, _adaptive_gen,
    _advanced_gen,
]:
    GENERATORS.update(_g)

# Merged knowledge registry: physics_name -> dict
KNOWLEDGE: dict[str, dict] = {}
for _k in [
    _regsurv_kn, _poisson_kn, _poisson3d_kn, _heat_kn, _elast_kn, _stokes_kn,
    _rxn_kn, _nonlinear_kn, _dg_kn, _dg_diffusion_kn, _adaptive_kn,
    _advanced_kn,
]:
    KNOWLEDGE.update(_k)


# ── every physics entry starts with a COMPLETE runnable script ───────
#
# Structural requirement (project owner, 2026-08-03): the knowledge has
# to be usable by a small model that will not infer, will not go
# hunting for a second payload, and will not recover from a partial
# answer. The generator callables already hold complete, executed
# scripts; before this block they were reachable only through
# prepare_simulation / run_with_generator, so an agent that asked
# knowledge(topic="physics", solver="dune", physics="stokes") got prose
# about Uzawa iteration and no code. Now the script is the FIRST thing
# in the entry, and the entry also names the other lookups so nothing
# depends on already knowing a topic= string.
_PHYSICS_TO_VARIANT: dict[str, str] = {
    _p: _p + "_2d" for _p in KNOWLEDGE
}
_PHYSICS_TO_VARIANT["poisson_mms"] = "poisson_mms_3d_varcoeff"

_PHYSICS_INDEX = sorted(KNOWLEDGE)

_WHERE_ELSE = {
    "other physics for this backend": (
        "knowledge(topic='physics', solver='dune', physics=X) with X "
        "one of: " + ", ".join(_PHYSICS_INDEX)),
    "the measured API surface (spaces, grids, BCs, solver parameters, "
    "adaptivity, VTK, JIT)": (
        "knowledge(topic='overview', solver='dune') — that call also "
        "carries the install fingerprint the measurements belong to"),
    "every Signal clause in one list": (
        "knowledge(topic='pitfalls', solver='dune')"),
    "run it": (
        "run_with_generator(backend='dune', physics_variant=<the "
        "variant name printed in this entry>, params={...}), or paste "
        "minimal_working_example into a file and run it with a Python "
        "that can import dune.fem"),
}

for _phys, _variant in _PHYSICS_TO_VARIANT.items():
    _gen = GENERATORS.get(_variant)
    if _gen is None:                      # pragma: no cover - registry drift
        continue
    _entry = KNOWLEDGE[_phys]
    _front = {
        "READ_THIS_FIRST": (
            f"minimal_working_example below is a COMPLETE, "
            f"self-contained DUNE-fem script for '{_phys}'. Save it to "
            f"a file and run it — it needs no other file, no mesh on "
            f"disk and no edits to run as written. It prints its own "
            f"physics check, so you can tell a correct run from a "
            f"converged-but-wrong one without a reference solution. "
            f"Generator variant name: '{_variant}'."),
        "minimal_working_example": _gen({}),
    }
    _front.update(_entry)
    _front["where_else_to_look"] = _WHERE_ELSE
    KNOWLEDGE[_phys] = _front

# General knowledge (not physics-specific)
KNOWLEDGE["_general"] = {

    # ── first thing a reader sees; complete, no cross-references ─────
    "START_HERE": {
        "what_dune_fem_is": (
            "A Python front-end that writes and JIT-compiles C++ for "
            "every grid, space, form and scheme you build. Weak forms "
            "are UFL — the same UFL as FEniCS/dolfinx — so a form that "
            "type-checks in dolfinx type-checks here. EVERYTHING "
            "AROUND the form is different: spaces, grids, boundary "
            "conditions, solvers and output share no API with dolfinx."),
        "complete_runnable_poisson": '''\
"""Poisson  -Laplace(u) = 1  on [0,1]^2, u = 0 on the boundary.
Complete script: save, run with a Python that can import dune.fem."""
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import TrialFunction, TestFunction, dot, grad, dx
import numpy as np

gridView = structuredGrid([0, 0], [1, 1], [32, 32])   # YaspGrid, CUBES
space = lagrange(gridView, order=1)                   # scalar P1/Q1
u = TrialFunction(space)
v = TestFunction(space)

a = dot(grad(u), grad(v)) * dx
b = 1.0 * v * dx

dbc = DirichletBC(space, 0)          # value 0 on the WHOLE boundary
scheme = galerkin([a == b, dbc], solver="cg")   # BC must be IN the list
uh = space.interpolate(0, name="solution")
info = scheme.solve(target=uh)

vals = np.array(uh.as_numpy)
print("converged:", info["converged"],
      " linear_iterations:", info["linear_iterations"])
print("max(u) =", float(vals.max()), " dofs:", len(vals))
# PHYSICS CHECK. For -Lap(u) = 1 with u = 0 on the unit square the
# maximum sits at the centre. Executed 2026-08-03: this problem printed
# max(u) = 0.0737281175 on this 32x32 grid, and 0.07459830 on an 8x8
# grid — the discrete maximum approaches the continuum value from
# ABOVE as the mesh is refined. A value of order 1, or 1e14, means the
# Dirichlet condition was dropped.
gridView.writeVTK("result", pointdata={"u": uh})
''',
        "the_six_transfers_from_dolfinx_that_break": [
            "1. SPACE: lagrange(gridView, order=k[, dimRange=d]) — NOT "
            "functionspace(mesh, ('Lagrange', k)). dimRange is what "
            "makes a space vector-valued; without it you get a scalar.",
            "2. BC: from dune.ufl import DirichletBC; "
            "DirichletBC(space, value[, indicator]). It takes effect "
            "ONLY as an element of the list passed to galerkin([a == b, "
            "dbc]). There is no bc.apply(). Building it and not listing "
            "it leaves a singular system that still reports "
            "converged=True.",
            "3. SOLVE: scheme.solve(target=uh) writes into uh and "
            "returns an info dict; it does not return the solution.",
            "4. OUTPUT: gridView.writeVTK(name, pointdata={...}). There "
            "is no XDMFFile and no VTXWriter.",
            "5. CELLS: structuredGrid makes CUBES (quadrilateral / "
            "hexahedron). space.cell() nevertheless always says "
            "'triangle' / 'tetrahedron' — it is a UFL bookkeeping cell, "
            "not the mesh. For simplices use dune.alugrid.",
            "6. FIRST RUN IS SLOW: the first run of any new grid, "
            "space, form or scheme compiles C++ ('DUNE-INFO: Compiling "
            "<X> (new)' on stderr) and can sit silent for minutes. "
            "Budget >= 600 s of timeout for a first run.",
        ],
        "REQUIRED_imports": (
            "from dune.grid import structuredGrid          # grids\\n"
            "from dune.fem.space import lagrange           # spaces\\n"
            "from dune.fem.scheme import galerkin          # schemes\\n"
            "from dune.ufl import DirichletBC, Constant    # BCs, "
            "runtime constants\\n"
            "from ufl import TrialFunction, TestFunction, grad, dx, ds"),
        "OPTIONAL_imports": (
            "from dune.fem import integrate, assemble, adapt, mark, "
            "globalRefine\\n"
            "from dune.fem.view import adaptiveLeafGridView\\n"
            "from dune.alugrid import aluConformGrid       # simplices, "
            "adaptivity\\n"
            "from dune.fem.space import composite          # multi-field"),
        "how_to_ask_for_more": (
            "knowledge(topic='physics', solver='dune', physics=X) "
            "returns a COMPLETE runnable script plus the pitfalls for "
            "that physics. X is one of: adaptive_poisson, dg_advection, "
            "dg_advection_diffusion, eigenvalue, heat, helmholtz, hyperelasticity, "
            "linear_elasticity, maxwell, mixed_methods, navier_stokes, "
            "nonlinear, poisson, poisson_mms, reaction_diffusion, "
            "stokes, time_dependent_heat. "
            "knowledge(topic='pitfalls', solver='dune') returns every "
            "Signal clause. The rest of THIS entry is the measured API "
            "reference — read it only when the script above is not "
            "enough."),
    },

    # ── what the rest of this entry contains, one line each ──────────
    # Everything below START_HERE is REFERENCE. A reader who only needs
    # to get a solve running should stop after START_HERE; this index
    # exists so nobody has to read 50 KB to find out whether the answer
    # is in here.
    "SECTION_INDEX": {
        "spaces / grid_types / solvers / adaptivity / parallel":
            "one-screen summaries of each API surface, with the "
            "measured restrictions attached",
        "grid_cell_types_measured":
            "which factory gives cubes and which gives simplices, "
            "with element counts",
        "ufl_cell_is_always_a_simplex":
            "space.cell() lies about the mesh — read this before "
            "trusting any UFL introspection",
        "space_construction_measured":
            "the exact rejection messages for wrong lagrange() calls",
        "dirichlet_bc_measured / natural_bc_measured":
            "essential BCs, and how to write Neumann/Robin (there is "
            "no facet tagging)",
        "silent_wrong_traps_measured":
            "runs that report converged=True and are wrong",
        "solver_control_measured":
            "the only legal solver and preconditioner names, and the "
            "parameter keys",
        "assemble_measured":
            "how to get a scipy matrix out (eigenvalue problems)",
        "integrate_measured":
            "norms and quadrature order",
        "jit_compilation_measured":
            "why the first run hangs, where the cache is, what "
            "triggers a rebuild",
        "adaptation_measured":
            "the working mark/adapt cycle, plus the upstream defect "
            "in mark(gridView=) and the globalRefine no-op",
        "intermittent_teardown_abort_measured":
            "why a correct DUNE run can exit 134",
        "companion_modules_measured":
            "which dune.* packages exist here (dune-fem-dg and "
            "dune-vem do NOT)",
        "runtime_constants_measured / vtk_output_measured / "
        "threading_measured / convergence_behaviour_2d_poisson / "
        "phantom_apis_checked / module_inventory_2_12":
            "constants without re-JIT, VTK options, thread defaults, "
            "how to set up an order study, names that do not exist, "
            "the raw 2.12 inventory",
    },

    "description": "DUNE-fem general capabilities",
    "form_language": "UFL (shared with FEniCS) — weak forms are directly interchangeable",
    "spaces": {
        "lagrange": "Continuous Lagrange (any order)",
        "dglagrange": "Discontinuous Lagrange",
        "dglegendre": "DG with Legendre basis",
        "dgonb": "DG with orthonormal basis",
        "raviartThomas": "H(div) conforming (camelCase Python wrapper; the C++ header is lowercase but the Python factory is camelCase — re-confirmed at runtime 2026-08-03: hasattr(dune.fem.space,'raviartThomas') is True, the lowercase spelling is False)",
        "bdm": "Brezzi-Douglas-Marini H(div) elements",
        "bdfm": "Brezzi-Douglas-Fortin-Marini H(div) elements",
        "finiteVolume": "Piecewise-constant FV space",
        "rannacherTurek": "Rannacher-Turek non-conforming",
        "p1Bubble": (
            "Mini element (P1 + bubble) — SIMPLICES ONLY. On a "
            "structuredGrid (YaspGrid cubes) the JIT build fails with "
            "'static assertion failed: p1Bubble interpolation is only "
            "implemented for simplicial grids.' after several minutes "
            "of C++ compilation; on aluConformGrid over the same "
            "4x4 cartesianDomain it builds (executed 2026-08-03, "
            "dune-fem 2.12.0.2)"),
        "composite": (
            "Multi-field space: composite(V, Q, "
            "components=['velocity','pressure']). TrialFunction on it "
            "is ONE argument of shape (sum of dimRanges,) — slice it, "
            "ufl.TrialFunctions does NOT unpack it. It is the space for "
            "a MONOLITHIC solve: galerkin() on it builds and solves "
            "(re-measured 2026-10-01 with the stokes template's "
            "Taylor-Hood form: storage 'numpy', umfpack converged in 1 "
            "linear iteration)."),
        "product": (
            "NOT an alias of composite. For the same arguments it has "
            "the same C++ space type, dimRange and size, but dune-fem "
            "documents it as the space for a DECOUPLED solve: it has "
            "no storage (storage.name is None), interpolate() returns "
            "a tuple function with .components, and galerkin() on it "
            "raises TypeError at scheme construction for solver=None, "
            "'gmres' and ('suitesparse','umfpack') (measured "
            "2026-10-01 on dune-fem 2.12.0.2)."),
        "combined": "Present in dune.fem.space; NOT exercised here.",
        "lagrangehp": (
            "Variable-order Lagrange for p-adaptivity, "
            "lagrangehp(gridView, order=k, maxOrder=m). Present; NOT "
            "exercised here."),
        "dglagrangelobatto / dglegendrehp / dgonbhp / dganisotropic": (
            "Further DG variants present in dune.fem.space; NOT "
            "exercised here."),
        "NOT a space factory": (
            "dune.fem.space.product_space is ABSENT / FALSIFIED: "
            "hasattr() is False and the string occurs nowhere in the "
            "installed package (executed 2026-08-03)."),
        "raviartThomas caveat": (
            "It works ALONE (measured: size 144 on an 8x8 grid, and "
            "the interpolant of a divergence-free field has "
            "||div||_L2 = 4.4e-16) but CANNOT be a leg of "
            "product()/composite() — the composite space fails to "
            "compile in C++. See knowledge(topic='physics', "
            "solver='dune', physics='mixed_methods')."),
    },
    "grid_types": {
        "structuredGrid(lower, upper, division)": (
            "YaspGrid, always CUBES. The default for everything that "
            "does not need adaptivity or simplices."),
        "dune.alugrid.aluConformGrid / aluSimplexGrid / aluCubeGrid": (
            "Unstructured, 2D/3D, and the ONLY route to simplices and "
            "to working adaptivity. Take a cartesianDomain(...) or a "
            "(reader.gmsh, 'file.msh') tuple plus dimgrid=."),
        "dune.fem.view.adaptiveLeafGridView(grid)": (
            "REQUIRED wrapper before dune.fem.adapt will do anything."),
        "dune.fem.view.geometryGridView": "Deforming/moving meshes.",
        "dune.fem.view.filteredGridView": "Sub-domain views.",
        "other dune.grid factories present, NOT exercised here": (
            "yaspGrid (the direct factory behind structuredGrid), "
            "ugGrid, albertaGrid, onedGrid, tensorProductCoordinates, "
            "equidistantOffsetCoordinates, string2dgf."),
        "mesh readers (dune.grid.reader)": (
            "dgf, dgfString, gmsh, meshio, structured. Pass as a "
            "TUPLE: aluConformGrid((reader.gmsh, 'mesh.msh'), "
            "dimgrid=2)."),
    },
    "dg_methods": {
        "what_is_here": (
            "DG spaces (dglagrange, dglegendre, dgonb, and the hp "
            "variants) and interior-facet integrals (dS) in the "
            "ORDINARY galerkin scheme. That is enough for SIPG, upwind "
            "advection and LDG written by hand — for SIPG with advection see "
            "knowledge(topic='physics', solver='dune', "
            "physics='dg_advection_diffusion') for a complete working one."),
        "what_is_NOT_here": (
            "dune-fem-dg. FALSIFIED 2026-08-03: 'import dune.femdg' "
            "and 'import dune.fem.dg' both raise ModuleNotFoundError "
            "on a pip-installed (PyPI) dune-fem 2.12.0.2 install "
            "(conda-forge has no dune-fem package), so its ready "
            "made SSP Runge-Kutta steppers, Bassi-Rebay 1/2, CDG/CDG2 "
            "operators and limiters are NOT available. Earlier "
            "revisions of this catalog listed them as if they were. "
            "Write your own explicit stepper; SSP-RK2 (Heun) is four "
            "lines and is in the dg_advection template."),
        "dune_fem_scheme_variants_that_DO_exist": (
            "dune.fem.scheme exposes galerkin, molGalerkin, dg, "
            "dgGalerkin, h1, h1Galerkin and linearized; only galerkin "
            "was exercised here."),
    },
    "solvers": {
        "Krylov names accepted for the DEFAULT storage": (
            "EXACTLY 'cg', 'gmres', 'bicgstab'. Anything else raises "
            "at scheme construction; see "
            "solver_control_measured.valid_solver_strings for the "
            "verbatim message."),
        "preconditioner names accepted": (
            "EXACTLY 'none', 'sor', 'ssor', 'gauss-seidel', 'jacobi' "
            "under the key 'linear.preconditioning.method'. No ILU, no "
            "AMG (executed 2026-08-03)."),
        "direct": (
            "solver=('suitesparse','umfpack') — a TUPLE, verified to "
            "work and to report linear_iterations 1. Required for "
            "saddle-point problems (Stokes, mixed, Navier-Stokes). The "
            "other suitesparse types in the source are 'ldl', "
            "'spqr_symmetric', 'spqr_nonsymmetric'."),
        "other storages (source-read, NOT exercised here)": (
            "('istl',...), ('petsc',...), ('eigen',...), "
            "('viennacl',...)."),
        "scipy": (
            "dune.fem.assemble(form).as_numpy gives a scipy csr_matrix "
            "(already CSR — .tocsr() is a measured no-op) for anything "
            "dune-fem does not solve itself, e.g. eigenvalue "
            "problems."),
    },
    "adaptivity": (
        "gv = adaptiveLeafGridView(aluConformGrid(...)); build the "
        "space on gv; ind = finiteVolume(gv).interpolate(<estimator>, "
        "name='ind'); dune.fem.mark(ind, tol); dune.fem.adapt([uh]). "
        "Two separate statements — mark() returns STATISTICS, not a "
        "marker. Do NOT pass gridView= to mark(); it is an upstream "
        "defect in 2.12.0.2. Full detail in adaptation_measured."),
    "parallel": (
        "Shared memory: dune.fem.threading.use is an ATTRIBUTE. It "
        "starts at DUNE_NUM_THREADS, else OMP_NUM_THREADS, else 1, and "
        "threading.max at DUNE_NUM_THREADS, else OMP_NUM_THREADS, else "
        "the core count, so with neither variable set use is 1 while "
        "max reports every core (measured 2026-10-01 on a 32-core "
        "host: no variable -> max 32, use 1; OMP_NUM_THREADS=2 -> 2 "
        "and 2; DUNE_NUM_THREADS=3 -> 3 and 3) — assign it or "
        "call dune.fem.threading.useMax(). Distributed: dune.fem.comm "
        "exposes rank/size/barrier/broadcast/gather/scatter/sum/min/"
        "max (measured rank 0, size 1 in a serial run); no MPI launch "
        "idiom was exercised here, so treat multi-rank runs as "
        "unverified on this install."),
    "vem": (
        "NOT AVAILABLE HERE. FALSIFIED 2026-08-03: 'import dune.vem' "
        "raises ModuleNotFoundError on this dune-fem 2.12.0.2 install. "
        "VEM lives in the separate dune-vem package, which is not "
        "pulled in by dune-fem."),
    "surface_fem": "PDEs on static and evolving surfaces (mean curvature flow)",
    "unique_features": [
        "Shares UFL with FEniCS — physics descriptions are interchangeable",
        "JIT compilation: prototype in Python, performance of C++",
        "Deep h/p-adaptivity with ALUGrid",
        "DG spaces and interior-facet forms in the core scheme "
        "(dune-fem-dg itself is a SEPARATE package and is not "
        "importable here — see dg_methods)",
        "VEM only if the separate dune-vem package is installed; it "
        "is NOT importable here",
        "Surface FEM for PDEs on manifolds",
        "Multiple storage backends: numpy, istl, petsc",
    ],
    "cmake_user_macros": {
        "description": (
            "User-callable CMake helpers defined in dune-fem's "
            "cmake/modules/FemShort.cmake — downstream dune-fem-based "
            "projects invoke these in their CMakeLists.txt."
        ),
        "dune_add_subdirs": {
            "signature": "dune_add_subdirs(<dir1> [<dir2>...] [EXCLUDE <str> | NOEXCLUDE])",
            "purpose": "Add multiple subdirectories with one call. Subdirs "
                       "matching EXCLUDE (default: 'test') are added with "
                       "EXCLUDE_FROM_ALL; NOEXCLUDE adds everything.",
        },
        "dune_fem_add_test": {
            "signature": (
                "dune_fem_add_test(<test1> [<test2>...] "
                "[FAILTEST <ftest>...] [COMPILEFAILTEST <cftest>...] "
                "[DEPENDENCY_ONLY <dep>...] [NO_DEPENDENCY <ntest>...])"),
            "purpose": "Register tests with CMake testing framework.",
            "kwargs": {
                "FAILTEST":        "Tests EXPECTED to fail "
                                   "(WILL_FAIL property set to true)",
                "COMPILEFAILTEST": "Tests EXPECTED to fail at compile time "
                                   "(WILL_FAIL too, runs via "
                                   "cmake --build . --target <test>)",
                "DEPENDENCY_ONLY": "Add targets as test dependencies but "
                                   "NOT as runnable tests",
                "NO_DEPENDENCY":   "Add target as test but NOT as dependency",
            },
        },
        "dune_install": {
            "signature": "dune_install(<files>...)",
            "purpose": "Install given files into the current source dir's "
                       "include-directory location "
                       "(CMAKE_CURRENT_SOURCE_DIR with CMAKE_SOURCE_DIR "
                       "replaced by CMAKE_INSTALL_INCLUDEDIR).",
        },
        "Signal": (
            "[API] FAILTEST and COMPILEFAILTEST tests have "
            "WILL_FAIL=true. A test that passes in those buckets is "
            "REPORTED AS FAILING by ctest — the inversion is intentional. "
            "Don't list a working test under FAILTEST expecting it to "
            "pass cleanly; ctest will report it as a regression. (File "
            "walk dune-fem/cmake/modules/FemShort.cmake 2026-06-02.)"
        ),
    },
    "local_contribution_assembly_modes": {
        "description": (
            "DiscreteFunction.localContribution(assembly) is the "
            "user-facing handle for accumulating element-local "
            "vector contributions into a global dune-fem "
            "DiscreteFunction. Source: python/dune/fem/space/"
            "__init__.py::localContribution()."),
        "python_dispatch": {
            "'set'": ("calls self.setLocalContribution() — wraps "
                      "the C++ SetLocalContribution<DF> "
                      "specialization (overwrite mode)"),
            "'add'": ("calls self.addLocalContribution() — wraps "
                      "the C++ AddLocalContribution<DF> "
                      "specialization (accumulate mode)"),
        },
        "cpp_only_tags_not_python_reachable": [
            "AddScaledLocalContribution<DF> — defined in "
            "dune/fem/common/localcontribution.hh as a using-"
            "alias for LocalContribution<DF, Assembly::AddScaled>",
            "SetSelectedLocalContribution<DF> — same header, "
            "Assembly::SetSelected; USED inside dune-fem's own "
            "auto-generated C++ in python/dune/fem/utility/"
            "filteredgridview.py but NOT routed via Python's "
            "localContribution dispatch",
        ],
        "Signal": (
            "[API] DiscreteFunction.localContribution(assembly) "
            "accepts EXACTLY two string tags: 'set' and 'add'. "
            "Anything else raises ValueError('assembly can only "
            "be `set` or `add`') at python/dune/fem/space/"
            "__init__.py:126. The C++ header dune/fem/common/"
            "localcontribution.hh defines FOUR using-aliases — "
            "AddLocalContribution, AddScaledLocalContribution, "
            "SetLocalContribution, SetSelectedLocalContribution — "
            "and the latter two ARE used elsewhere in dune-fem "
            "(e.g. filteredgridview code-gen). Users reading C++ "
            "tutorials or grepping the source for sample usage "
            "may try df.localContribution('addScaled') or "
            "df.localContribution('setSelected') expecting them "
            "to work and instead hit the ValueError. The Python "
            "binding intentionally surfaces only the two safe "
            "tags; the other two are reachable only from within "
            "C++ code embedded in JIT strings (passed via "
            "algorithm.run / generate_method idioms). "
            "(File walk dune/fem/common/localcontribution.hh + "
            "python/dune/fem/space/__init__.py 2026-06-03.)"
        ),
    },
}

# Everything under this key was established BY EXECUTION against the
# installed dune-fem (see verified_api.py for the install fingerprint,
# the scripts that were run and what they printed). Merged rather than
# nested so `knowledge(topic="overview", solver="dune")` surfaces it
# alongside the older reference material.
KNOWLEDGE["_general"].update(_executed_api)
