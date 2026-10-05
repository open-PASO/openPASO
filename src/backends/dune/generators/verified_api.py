"""DUNE-fem API surface established BY EXECUTION on the installed package.

Every statement in this module was produced by running a script against
the installed DUNE-fem and reading the output — not by reading upstream
documentation and not by reading the C++ sources alone. Where a source
file is cited it is cited as the EXPLANATION of an observed behaviour,
never as the evidence for it.

Install under test (recorded so a future reader can tell whether this
still applies):

    dune-fem / dune-common / dune-grid / dune-geometry / dune-istl /
    dune-localfunctions / dune-alugrid  all at 2.12.0.2
    (CPython 3.12.13, Linux x86-64)

WHY THIS MODULE EXISTS. DUNE-fem shares UFL with FEniCS/dolfinx, so a
model that knows dolfinx writes DUNE-fem weak forms correctly on the
first try — and then gets everything AROUND the form wrong, because the
space, grid, boundary-condition, solver and output APIs have nothing to
do with dolfinx. The three failure classes that cost the most are
collected here:

  * transfer errors — a dolfinx idiom that raises an unhelpful
    exception in DUNE-fem (cheap: you see it immediately);
  * phantom APIs — names that appear in tutorials, older releases or
    prior versions of this catalog and do not exist in 2.12
    (cheap once you know, expensive while you are guessing);
  * silent-wrong traps — the run completes, ``scheme.solve()`` reports
    ``converged: True``, and the answer is wrong by 15 orders of
    magnitude (ruinous, and the reason this module is worth its size).

Provenance convention: every entry ends with what was run and what was
observed, dated. "Executed 2026-08-03" means a script was run on the
install above on that date and the quoted output is verbatim.
"""

from __future__ import annotations


# Dotted `dune.<path>` names this module spells out precisely BECAUSE
# they do NOT resolve on the installed package. They are the phantom
# APIs a model would otherwise reach for, so the catalog has to be
# allowed to name them.
#
# tests/test_catalog_consistency.py::TestDuneFemAPIPaths subtracts this
# set from its resolve-check AND asserts that every entry really is
# unresolvable — so the exemption cannot be used to smuggle in a
# genuinely broken reference; it flips into an extra positive check.
KNOWN_ABSENT_DUNE_PATHS: frozenset[str] = frozenset({
    # `import dune.fem.solver` -> ModuleNotFoundError; solvers are
    # selected with the scheme's `solver=` argument.
    "dune.fem.solver",
    # the multi-field factory is dune.fem.space.product
    "dune.fem.space.product_space",
    # the Python factory is raviartThomas (camelCase)
    "dune.fem.space.raviartthomas",
    # companion packages the catalog used to advertise. Measured
    # 2026-08-03: every one of these raises ModuleNotFoundError on a
    # pip-installed (PyPI) dune-fem 2.12.0.2 install, which is exactly
    # WHY the catalog has to be allowed to name them — the claim being
    # made is that they are absent.
    "dune.femdg",
    "dune.fem.dg",
    "dune.vem",
    "dune.polygongrid",
    "dune.spgrid",
    "dune.uggrid",
    "dune.mmesh",
})

# JIT-generated module names (dune.generated.<kind>_<md5>) appear inside
# quoted error messages. The hash is machine-specific and the module
# only exists after that particular build, so they can never be
# resolve-checked.
JIT_GENERATED_PATH_PREFIXES: tuple[str, ...] = ("dune.generated",)


# ── Reference catalogue merged into KNOWLEDGE["_general"] ────────────

EXECUTED_API: dict = {
    "install_under_test": (
        "dune-fem 2.12.0.2 (dune-common/grid/geometry/istl/"
        "localfunctions/alugrid all 2.12.0.2), CPython 3.12.13, "
        "Linux x86-64, pip-installed from the PyPI sdists (conda-forge "
        "has no dune-fem package). All numbers below were "
        "measured on that install on 2026-08-03. "
        "NOTE for readers of older catalog text: entries dated "
        "2026-08-01 in this backend say 'dune-fem 2.10'; the env has "
        "since been rebuilt at 2.12.0.2 and the API facts below were "
        "re-established against 2.12."),

    # ── grids ───────────────────────────────────────────────────────
    "grid_cell_types_measured": {
        "description": (
            "Cell geometry actually produced by each grid factory, "
            "counted with gridView.size(0) and gridView.size(dim) and "
            "read off entity.type. Executed 2026-08-03."),
        "structuredGrid([0,0],[1,1],[4,4])": (
            "16 elements, 25 vertices, geometry type 'quadrilateral' "
            "— YaspGrid produces CUBES, never simplices"),
        "structuredGrid([0,0,0],[1,1,1],[2,2,2])": (
            "8 elements, 27 vertices, geometry type 'hexahedron'"),
        "aluConformGrid(cartesianDomain([0,0],[1,1],[4,4]), dimgrid=2)": (
            "32 elements, 25 vertices, geometry type 'triangle' — each "
            "Cartesian cell is split into 2 triangles"),
        "aluSimplexGrid(same domain, dimgrid=2)": (
            "32 elements, 25 vertices, 'triangle'"),
        "aluCubeGrid(same domain, dimgrid=2)": (
            "16 elements, 25 vertices, 'quadrilateral'"),
        "aluSimplexGrid(cartesianDomain([0,0,0],[1,1,1],[2,2,2]), dimgrid=3)": (
            "48 elements, 27 vertices, 'tetrahedron' — 6 tets per hex"),
        "aluCubeGrid(same 3D domain, dimgrid=3)": (
            "8 elements, 27 vertices, 'hexahedron'"),
        "Signal": (
            "[API] structuredGrid is a YaspGrid and is ALWAYS made of "
            "cubes; the simplicial route measured here is "
            "dune.alugrid (aluConformGrid / aluSimplexGrid). The "
            "element count is the cheapest check: an n-by-n "
            "cartesianDomain gives n^2 cells through "
            "aluCubeGrid/structuredGrid and 2*n^2 cells through "
            "aluConformGrid/aluSimplexGrid. (Executed 2026-08-03: "
            "16 vs 32 elements for n=4. dune.grid also exposes "
            "ugGrid, albertaGrid and onedGrid, which were NOT "
            "exercised here.)"),
    },

    # ── the single most misleading thing in the whole API ───────────
    "ufl_cell_is_always_a_simplex": {
        "description": (
            "A dune-fem space reports a SIMPLEX UFL cell no matter "
            "what the grid is really made of. On the 16-quadrilateral "
            "structuredGrid above, space.cell() returns 'triangle' and "
            "space.ufl_element() prints '<Lagrange1 on a triangle>'; "
            "on the 8-hexahedron 3D grid it returns 'tetrahedron'. "
            "Meanwhile gridView.type is 'quadrilateral' / "
            "'hexahedron'. Executed 2026-08-03."),
        "why": (
            "dune/ufl/__init__.py::_cell(dimWorld, dimDomain) maps the "
            "dimension straight onto ufl.interval / ufl.triangle / "
            "ufl.tetrahedron / ufl.pentatope. The UFL cell exists only "
            "so UFL can type-check the form; the real basis functions "
            "come from the C++ space and the real quadrature from the "
            "real geometry type. The two are simply not connected."),
        "consequences": [
            "space.cell() / space.ufl_element() are NOT a way to find "
            "out what the mesh is made of — use gridView.type, or "
            "{e.type for e in gridView.elements}, or compare "
            "gridView.size(0) against the expected cube count.",
            "The UFL element hides the POLYNOMIAL DEGREE too, not just "
            "the cell: lagrange(gridView, order=1|2|3) all report "
            "ufl.finiteelement.FiniteElement('Lagrange', triangle, 1, "
            "...) and all print '<Lagrange1 on a triangle>', and they "
            "share ONE compiled Space module. Read the order back from "
            "space.size, never from the UFL element (corrected by "
            "adversarial audit 2026-08-03: an earlier revision of this "
            "entry claimed Q2 was reported as 'Lagrange2', which it is "
            "not). NOTE that dof counts do NOT "
            "distinguish cube from simplex: continuous Lagrange on the 16-cube "
            "grid and on the 32-triangle aluConformGrid over the same "
            "cartesianDomain both have 25 dofs at order 1 and 81 at "
            "order 2, because they share the same vertices (measured "
            "2026-08-03). Only the ELEMENT COUNT and the geometry "
            "type tell them apart.",
            "Any dolfinx-shaped logic that branches on mesh.ufl_cell() "
            "to pick an element family, a quadrature degree or a "
            "basix element will branch on a lie.",
        ],
        "Signal": (
            "[API] str(space.ufl_element()) says 'on a "
            "triangle' while gridView.type says 'quadrilateral'. "
            "(repr(space) itself is only the pybind object, "
            "'<dune.generated.femspace_<hash>.Space object at 0x...>' "
            "— corrected 2026-08-03; print the ufl_element, not the "
            "space.) This "
            "is normal and is NOT a bug to fix — it means the UFL cell "
            "carries no cell-shape information in dune-fem. Read the "
            "geometry from the gridView, never from the UFL element. "
            "(Executed 2026-08-03: structuredGrid([0,0],[1,1],[4,4]) "
            "-> 16 'quadrilateral' elements, space.cell() == "
            "'triangle', space.size == 25.)"),
    },

    # ── spaces ──────────────────────────────────────────────────────
    "space_construction_measured": {
        "description": (
            "What the space factories accept and reject, executed "
            "2026-08-03 on structuredGrid([0,0],[1,1],[4,4])."),
        "works": {
            "lagrange(gridView, order=k)":
                "scalar CG space; .size == 25 for k=1 on the 4x4 grid",
            "lagrange(gridView, order=k, dimRange=d)":
                "vector-valued space; dimRange=2 gives .size == 50 — "
                ".size counts SCALAR dofs, not vector blocks, and "
                "len(uh.as_numpy) == 50 too",
            "dglagrange(gridView, order=0)":
                ".size == 16 == cell count — this is the DG0 space",
            "finiteVolume(gridView)":
                ".size == 16 — the other piecewise-constant space",
        },
        "rejected_with_the_exact_message": {
            "lagrange(gridView, ('Lagrange', 1))":
                "TypeError: '<' not supported between instances of "
                "'tuple' and 'int' — the dolfinx "
                "functionspace(domain, ('Lagrange', k)) idiom lands on "
                "dune-fem's `if order < 1` guard and produces an "
                "error that says nothing about the real mistake",
            "lagrange(gridView, order=1, shape=(2,))":
                "TypeError: lagrange() got an unexpected keyword "
                "argument 'shape' — dolfinx spells vector spaces with "
                "a shape tuple, dune-fem spells them dimRange=d",
            "lagrange(gridView, degree=2)":
                "TypeError: lagrange() got an unexpected keyword "
                "argument 'degree' — the kwarg is `order`, not "
                "`degree`",
            "lagrange(gridView, order=0)":
                "KeyError: 'Parameter error in LagrangeSpace with "
                "order=0: order has to be greater or equal to 1' — "
                "there is no order-0 Lagrange space; the DG0 / P0 "
                "space is dglagrange(gridView, order=0) or "
                "finiteVolume(gridView)",
        },
        "Signal": (
            "[API] Every dolfinx space-construction idiom fails on "
            "dune-fem with an error that does not name the real "
            "problem: the positional element tuple becomes a "
            "tuple-vs-int comparison TypeError, `shape=` and `degree=` "
            "become unexpected-keyword TypeErrors, and order=0 becomes "
            "a KeyError about LagrangeSpace. The correct spelling is "
            "always lagrange(gridView, order=k[, dimRange=d]). "
            "(Executed 2026-08-03, all four messages quoted "
            "verbatim.)"),
    },

    # ── building UFL objects ────────────────────────────────────────
    "ufl_objects_come_from_the_SPACE": {
        "description": (
            "In dolfinx you build UFL objects from the MESH "
            "(ufl.SpatialCoordinate(domain), ufl.FacetNormal(domain)) "
            "and the function space separately. In dune-fem the SPACE "
            "is the ufl.FunctionSpace, and the grid view is not a UFL "
            "object at all. Executed 2026-08-03."),
        "measured": (
            "SpatialCoordinate(space) -> a length-2 vector on a 2D "
            "grid, ufl_shape (2,). SpatialCoordinate(gridView) -> "
            "AttributeError: "
            "'dune.generated.hierarchicalgrid_<hash>.LeafGrid' object "
            "has no attribute 'ufl_domain'."),
        "Signal": (
            "[API] AttributeError \"...LeafGrid object has no "
            "attribute 'ufl_domain'\" means a UFL constructor was "
            "handed the grid view instead of the space. Pass the "
            "space: SpatialCoordinate(space), TrialFunction(space), "
            "TestFunction(space). (Executed 2026-08-03.)"),
    },

    # ── mesh import + dof layout ────────────────────────────────────
    "mesh_import_and_dof_layout_measured": {
        "description": "Executed 2026-08-03.",
        "gmsh_import": (
            "The reader must be named explicitly as a TUPLE: "
            "aluConformGrid((reader.gmsh, 'mesh.msh'), dimgrid=2) — "
            "with `from dune.grid import reader`. Measured on a "
            "4-triangle msh2 file: 4 elements, 5 vertices, all "
            "'triangle'. Passing the path as a bare string, "
            "aluConformGrid('mesh.msh', dimgrid=2), raises "
            "RuntimeError: \"IOError [checkMacroGridFile:.../dune/"
            "alugrid/3d/grid_imp.cc:343]: Wrong file format!\" — "
            "ALUGrid interprets a bare string as its own DGF macro "
            "grid file, not as Gmsh. reader also has .dgf, "
            ".dgfString, .meshio and .structured."),
        "dof_ordering_for_dimRange_gt_1": (
            "INTERLEAVED by component, i.e. (u0, v0, u1, v1, ...). "
            "Measured by interpolating as_vector([x, 100 + y]) into "
            "lagrange(2x2 grid, order=1, dimRange=2): the dof array "
            "had length 18 (9 vertices x 2) and read "
            "[0.0, 100.0, 0.5, 100.0, 1.0, 100.0, 0.0, 100.5, ...] — "
            "every odd index >= 100, every even index < 100. It is "
            "NOT blocked (all of component 0, then all of component "
            "1). Slice uh.as_numpy[i::dimRange] to get component i."),
    },

    # ── boundary conditions ─────────────────────────────────────────
    "dirichlet_bc_measured": {
        "description": (
            "dune.ufl.DirichletBC(functionSpace, value, subDomain=None). "
            "The CONSTRUCTION rows below were measured on a 4x4 "
            "structuredGrid; the solve behaviour in "
            "silent_wrong_traps_measured was measured on a 16x16 one "
            "with the MMS u* = sin(pi x) sin(pi y). Executed "
            "2026-08-03."),
        "construction": {
            "DirichletBC(scalar_space, 0)": "accepted",
            "DirichletBC(vector_space_dimRange2, 0)":
                "AttributeError: 'int' object has no attribute "
                "'ufl_shape' — a bare scalar is only auto-wrapped for "
                "SCALAR spaces; a vector space needs a list",
            "DirichletBC(vector_space_dimRange2, [0, 0])": "accepted",
            "DirichletBC(vector_space_dimRange2, [0, 0, 0])":
                "bare AssertionError with an EMPTY message (the "
                "`assert ufl_value.ufl_shape[0] == dimRange` in "
                "dune/ufl/__init__.py carries no text) — component "
                "count mismatches are the least self-explanatory "
                "failure in the API",
            "DirichletBC(space, TrialFunction(space))":
                "AssertionError: 'the ufl expression used for "
                "Dirichlet conditions should not use a `Test` or "
                "`Trial` function'",
        },
        "how_it_is_applied": (
            "A DirichletBC only does something if it is an ELEMENT OF "
            "THE LIST handed to the scheme: galerkin([a == b, dbc], "
            "...). There is no apply()/assemble-time hook and nothing "
            "to call afterwards — dune-fem decides at scheme-"
            "construction time whether to wrap the operator in a "
            "DirichletWrapper (see integrands.hasDirichletBoundary in "
            "dune/fem/scheme/_schemes.py)."),
        "subdomain_argument": (
            "The optional third argument is a UFL CONDITIONAL over "
            "SpatialCoordinate(space), e.g. "
            "DirichletBC(space, g, conditional(x[0] < 1e-8, 1, 0)). "
            "dune.ufl.BoxDirichletBC(space, value, xL, xR) builds such "
            "a conditional for an axis-aligned box. Nothing checks "
            "that the conditional selects any facet at all."),
    },

    # ── the expensive one ───────────────────────────────────────────
    "silent_wrong_traps_measured": {
        "description": (
            "Measured 2026-08-03. Same problem in every row: "
            "-Delta u = 2 pi^2 sin(pi x) sin(pi y) on a 16x16 "
            "structuredGrid, Lagrange order 1, solver='cg', L2 error "
            "against the exact solution computed with "
            "dune.fem.integrate(..., order=6)."),
        "rows": [
            "CORRECT — galerkin([a == b, DirichletBC(space, 0)]): "
            "converged=True, linear_iterations=1, L2 err 1.899705e-03, "
            "max(u_h)=1.0032 (exact max is 1).",
            "BC OMITTED FROM THE LIST — galerkin(a == b, space=space): "
            "converged=True, linear_iterations=23935, L2 err "
            "7.510512e+14, max(u_h) = -7.5e+14. The scheme silently "
            "solved a PURE NEUMANN problem, which is singular; CG "
            "ground through 23935 iterations and the returned info "
            "dict still says converged: True.",
            "BC PRESENT BUT ITS SUBDOMAIN NEVER FIRES — "
            "DirichletBC(space, 0, conditional(x[0] < -1.0, 1, 0)) on "
            "[0,1]^2: byte-identical to the omitted case — "
            "converged=True, 23935 iterations, L2 err 7.510512e+14. A "
            "wrong sign or a wrong threshold in the conditional is "
            "indistinguishable from having written no BC at all.",
            "PARTIAL BC (left edge only, conditional(x[0] < 1e-8)): "
            "converged=True, 45 iterations, L2 err 2.676075e+00 — a "
            "well-posed but DIFFERENT boundary-value problem. The "
            "iteration count is the tell: 45 (well-posed) vs 23935 "
            "(singular) vs 1 (fully constrained).",
        ],
        "Signal": (
            "[Numerical] scheme.solve() returning "
            "{'converged': True, ...} does NOT mean the boundary "
            "conditions were applied. A DirichletBC that was never put "
            "in the galerkin([...]) list, or whose subDomain "
            "conditional selects no facet, leaves a singular pure-"
            "Neumann system that CG reports as converged. The "
            "observable symptoms are (a) linear_iterations in the "
            "thousands where the constrained problem needs tens, and "
            "(b) max(uh.as_numpy) exploding by many orders of "
            "magnitude. Assert on both. (Executed 2026-08-03: "
            "23935 iterations, L2 error 7.51e+14, converged=True.)"),
    },

    # ── solver control ──────────────────────────────────────────────
    "solver_control_measured": {
        "description": "Executed 2026-08-03 on the same 16x16 problem.",
        "valid_solver_strings": (
            "For the default (numpy/'fem') storage exactly three: "
            "'cg', 'gmres', 'bicgstab'. This is not a documentation "
            "claim — galerkin(..., solver='conjugate_gradient') raises "
            "RuntimeError: \"ParameterInvalid [getEnumeration:.../"
            "dune/fem/io/parameter/reader.hh:300]: Parameter "
            "'fem.solver.linear.method' invalid. Valid values are: "
            "gmres, cg, bicgstab\". The name is NOT validated in "
            "Python — dune/fem/discretefunction/_solvers.py just "
            "forwards it as the parameter 'linear.method' — so the "
            "error surfaces at scheme construction, from C++."),
        "valid_preconditioner_strings": (
            "linear.preconditioning.method takes a SHORT enumeration "
            "and it is NOT the list people expect: "
            "galerkin(..., parameters={'linear.preconditioning.method': "
            "'ilu'}) raises RuntimeError: \"ParameterInvalid "
            "[getEnumeration:.../dune/fem/io/parameter/reader.hh:300]: "
            "Parameter 'fem.solver.linear.preconditioning.method' "
            "invalid. Valid values are: none, sor, ssor, gauss-seidel, "
            "jacobi\". No ILU, no AMG, no fieldsplit for the default "
            "storage. It fires at SCHEME CONSTRUCTION, before any "
            "solve. Measured effect on a 12x12 Taylor-Hood Stokes "
            "problem, gmres: linear_iterations 94026 with 'none', "
            "8587 with 'ssor', 18636 with 'jacobi' — and 'none' "
            "reproduced the no-parameter default EXACTLY, so the "
            "default preconditioner on this install IS 'none'. "
            "(RE-MEASURED by adversarial audit 2026-08-03. An earlier "
            "revision recorded 70941 with 'none' and 6351 with 'ssor'; "
            "neither reproduced. The ParameterInvalid string itself is "
            "unchanged and was re-confirmed.) Executed 2026-08-03, hit "
            "from two independent scripts."),
        "direct_solver_executed": (
            "solver=('suitesparse', 'umfpack') WORKS on this install: "
            "the scheme built (62.6 s, one new JIT module — a direct "
            "solver is a different C++ type, so it costs its own "
            "build) and solved with info "
            "{'converged': True, 'iterations': 0, "
            "'linear_iterations': 1}, reaching exactly the same "
            "max(u_h) = 0.07459830 as the CG solve. "
            "linear_iterations == 1 is the tell that it was direct. "
            "Executed 2026-08-03."),
        "other_backends_NOT_EXECUTED": (
            "SOURCE-READ ONLY (dune/fem/discretefunction/_solvers.py, "
            "2026-08-03) — unlike the umfpack row above, none of "
            "these were run. solver may also be "
            "('suitesparse', 'ldl'|'spqr_symmetric'|"
            "'spqr_nonsymmetric'), ('istl', ...), ('petsc', ...), "
            "('eigen', 'cg'|'bicgstab'), "
            "('viennacl', 'cg'|'gmres'|'bicgstab'). Only the "
            "suitesparse and eigen/viennacl method names are validated "
            "in Python (ValueError listing the alternatives); the rest "
            "become C++ parameters. Whether the underlying library is "
            "actually linked into a given conda build is a separate "
            "question — check before relying on one."),
        "convergence_flag_semantics": (
            "The flag IS meaningful for iteration starvation: with "
            "parameters={'linear.maxiterations': 1} the info dict came "
            "back {'converged': False, 'linear_iterations': -1} and "
            "the L2 error was 5.0e-01; maxiterations 3, 10 and 10000 "
            "all gave converged=True and the correct 1.899705e-03. It "
            "is NOT meaningful for a singular system — see "
            "silent_wrong_traps_measured. Always check "
            "info['converged'] AND a physical sanity bound."),
        "parameter_key_deprecation": (
            "MEASURED: parameters={'newton.tolerance': 1e-10} still "
            "works but emits UserWarning \"Warning: the parameter key "
            "'newton' is deprecated. Replace with 'nonlinear' to avoid "
            "this warning!\" (re-measured 2026-09-30: both key names "
            "are wrapped in ANSI colour codes, so match on \"is "
            "deprecated. Replace with\", not on the whole line), and "
            "the key is rewritten — scheme.parameters came back as "
            "{'nonlinear.tolerance': 1e-10, 'linear.method': 'cg'}. "
            "Passing BOTH 'newton.tolerance' and "
            "'nonlinear.tolerance' was accepted with NO error at all. "
            "SOURCE-READ (dune/fem/scheme/_schemes.py::"
            "_checkNewtonInParameters, not executed): on that "
            "collision the newton.* entry is popped and the "
            "nonlinear.* value survives; the KeyError "
            "\"Mixing new and old parameter keys is not allowed\" is "
            "raised only for the NESTED spellings, when a key "
            "containing 'newton.linear'/'nonlinear.linear' collides "
            "with the already-present de-prefixed 'linear.*' key. "
            "Executed 2026-08-03."),
        "unrecognised_keys_are_silently_ignored": (
            "MEASURED: parameters={'nonlinear.maxiter': 3, "
            "'totally.bogus.key': 42} was accepted with no exception "
            "and NO warning, and the solve ran to the normal answer. "
            "So a mistyped key ('maxiter' for 'maxiterations') leaves "
            "the default silently in place — the parameter dict is "
            "not a checked schema. Executed 2026-08-03."),
        "useful_keys_SOURCE_READ": (
            "From dune/fem/solver/{parameter,newtoninverseoperator}.hh "
            "(read 2026-08-03, only linear.tolerance / "
            "linear.maxiterations / linear.method were actually "
            "exercised): 'linear.tolerance', 'linear.maxiterations', "
            "'linear.verbose', 'linear.preconditioning.method', "
            "'nonlinear.tolerance', 'nonlinear.maxiterations', "
            "'nonlinear.verbose', 'nonlinear.linesearch'. The C++ "
            "reader prefixes them with 'fem.solver.' — that prefix is "
            "what appears in error messages, not what you pass."),
    },

    # ── integrate / error norms ─────────────────────────────────────
    "integrate_measured": {
        "description": (
            "dune.fem.integrate(expr, gridView=None, order=None) — the "
            "2.12 spelling. Executed 2026-08-03 on a 4x4 "
            "structuredGrid, integrand sin(pi x) sin(pi y) whose exact "
            "integral over [0,1]^2 is (2/pi)^2 = 0.405284734569."),
        "default_order_is_safe": (
            "order=None is NOT a low-order default: the measured "
            "relative error was 1.7e-10 for the trig integrand and "
            "1.7e-16 (machine precision) for x^4 y^4. Explicitly "
            "PASSING a small order is what silently under-integrates: "
            "order=1 gave 5.3e-02 relative error, order=2 and order=3 "
            "both gave 1.8e-04 (bit-identical, so those two requests "
            "resolve to the same rule), "
            "order=5 gave 2.4e-07, order=8 gave 7.2e-14."),
        "call_forms": (
            "gridView is REQUIRED unless the expression already "
            "contains a grid function: integrate(expr) on a pure-UFL "
            "expression raises AttributeError \"a 'gridView' has to be "
            "provided or the expression must contain a grid "
            "function.\". A vector-valued integrand returns a "
            "FieldVector, not a float — integrate(as_vector([e, 2*e])) "
            "returned (0.405285, 0.810569)."),
        "deprecated_entry_point": (
            "dune.fem.function.integrate(gridView, expr, order) still "
            "runs and returns the right number but warns "
            "\"dune.fem.function.integrate is deprecated use "
            "dune.fem.integrate instead. New signature is (expr, "
            "gridView, order)\". Note the ARGUMENT ORDER FLIPPED "
            "between the two."),
        "Signal": (
            "[Numerical] An L2/H1 error norm computed with an "
            "explicitly low quadrature order is smaller than the true "
            "error and flatters the convergence table. On this "
            "install the default (order=None) is generous, so the "
            "damage comes from hand-setting order=1 or order=2: the "
            "measured relative quadrature error was 5.3e-02 at "
            "order=1. Use order >= 2k+2 for a Lagrange-order-k error "
            "norm, or leave order unset. (Executed 2026-08-03.)"),
    },

    # ── JIT ─────────────────────────────────────────────────────────
    "jit_compilation_measured": {
        "description": (
            "dune-fem generates and compiles a C++ pybind11 module for "
            "every distinct grid type, space type, UFL integrands "
            "object and scheme type. Executed 2026-08-03."),
        "cache_location": (
            "dune.packagemetadata.getDunePyDir() returned "
            "'<sys.prefix>/.cache/dune-py', with the "
            "compiled modules in "
            "'<that>/python/dune/generated/*.so'. It grows without "
            "bound — one .so per distinct grid/space/integrands/scheme, "
            "each a few MB — so budget disk for it and delete the "
            "directory when you want a clean rebuild. "
            "Resolution order is "
            "$DUNE_PY_DIR/dune-py, then <sys.prefix>/.cache/dune-py "
            "if inVirtualEnvironment(), then ~/.cache/dune-py. "
            "'~/.dune/dune-py' — asserted by earlier revisions of "
            "this catalog — DOES NOT EXIST on this install and is not "
            "a path the 2.12 code can produce."),
        "cache_location_depends_on_CONDA_DEFAULT_ENV": (
            "THE SAME INTERPRETER RESOLVES TO TWO DIFFERENT CACHES. "
            "packagemetadata.inVirtualEnvironment() returns 1 as soon "
            "as CONDA_DEFAULT_ENV is present in os.environ, and only "
            "falls back to the sys.prefix != sys.base_prefix test "
            "otherwise. In a conda env those two prefixes are EQUAL, "
            "so CONDA_DEFAULT_ENV is "
            "the only thing putting the cache inside the env. "
            "Measured 2026-08-03 with the identical interpreter: "
            "CONDA_DEFAULT_ENV set -> "
            "'<env>/.cache/dune-py'; `env -u CONDA_DEFAULT_ENV` -> "
            "'~/.cache/dune-py'; DUNE_PY_DIR=<dir> -> "
            "'<dir>/dune-py'. A harness that spawns "
            "<env>/bin/python WITHOUT conda activation therefore "
            "starts from a cold cache in $HOME and pays the full "
            "build again, while an interactive `conda activate` "
            "session reuses the env cache — two caches, silently. "
            "openPASO's own _dune_subprocess_env() in "
            "src/backends/dune/backend.py already sets "
            "CONDA_DEFAULT_ENV, which is what keeps the two paths "
            "in agreement; pin DUNE_PY_DIR if you want certainty."),
        "measured_costs": (
            "Cold cache, machine under heavy concurrent load: a script "
            "that built structuredGrid in 2D and 3D plus three "
            "Lagrange spaces and then aluConformGrid + aluSimplexGrid "
            "+ aluCubeGrid in 2D and aluSimplexGrid + aluCubeGrid in "
            "3D took 439 s wall END TO END, of which the four "
            "'DUNE-INFO: Compiling HierarchicalGrid (new)' builds "
            "-- see also the p1Bubble note in _general['spaces']: a "
            "space the grid does not support burns the full build "
            "time and then fails on a C++ static_assert, not on a "
            "Python check -- "
            "dominated (the structuredGrid parts were already warm). "
            "With everything warm, a complete 8x8 Poisson run — grid, "
            "space, scheme, solve — took 0.89 s. The "
            "boundary-condition/solver battery (4 'Compiling "
            "Integrands (new)' lines and 1 'Compiling Scheme (new)') "
            "took 515 s; the quadrature battery (no new scheme) "
            "136 s; a single new scheme for a direct "
            "('suitesparse','umfpack') solver 62.6 s. ALUGrid "
            "hierarchical-grid modules are by far the most expensive "
            "single item."),
        "what_triggers_a_rebuild": (
            "Module names are md5 hashes of the generated C++ (the "
            "type name for spaces/schemes, the emitted integrands for "
            "a form). MEASURED to trigger a new module: "
            "a different C++ grid type (each (dimension, element type) "
            "ALUGrid built its own HierarchicalGrid; aluConformGrid and "
            "aluSimplexGrid of the same dimension are the same C++ type "
            "Dune::ALUGrid< d, d, Dune::simplex > and share one module, "
            "because conformity is set at run time — re-measured "
            "2026-09-30: six ALUGrid variants, four modules), and ANY "
            "change to the "
            "form — including changing a bare float literal in it "
            "(7.3125 -> 9.8125 cost 25.204 s and added one .so). "
            "MEASURED not to trigger one: re-running the identical "
            "form (0.051 s, no new .so), changing the .value of a "
            "dune.ufl.Constant (0.00056 s, no new .so), and "
            "— corrected by adversarial audit 2026-08-03, an earlier "
            "revision of this entry said the opposite — CHANGING THE "
            "LAGRANGE ORDER. lagrange(gridView, order=k) for "
            "k = 1, 2, 3, 5, 6 on one 4x4 structuredGrid all resolved "
            "to the SAME generated module, "
            "dune.generated.femspace_90f0a952..._0faf32f1..., with the "
            "correct sizes 25/81/169/441/625 and the .so count "
            "unchanged at 208 throughout — including k = 5 and 6, "
            "which had never been built on that install. The C++ "
            "LagrangeDiscreteFunctionSpace is instantiated with a "
            "dynamic polynomial order, so one module covers the range. "
            "Whether the SCHEME rebuilds per order was not "
            "separated out. Storage "
            "backends other than the default numpy were not "
            "exercised."),
        "Signal": (
            "[Performance] A DUNE-INFO: Compiling <X> (new) line on "
            "stderr — Python logging wrapping "
            "dune/common/__init__.py:49's "
            "'DUNE-%(levelname)s: %(message)s' around the f-string at "
            "dune/generator/cmakebuilder.py:378, so no part of the "
            "rendered line is greppable — "
            "means a C++ module is being built right now; the process "
            "will sit there for tens of seconds to minutes with no "
            "further output. A first run that appears hung is almost "
            "always this. Budget >= 600 s of timeout for any first "
            "run, and >= 900 s if ALUGrid is involved. (Executed "
            "2026-08-03: 439 s for five ALUGrid variants.)"),
    },

    # ── runtime constants ───────────────────────────────────────────
    "runtime_constants_measured": {
        "description": (
            "dune.ufl.Constant(value, name=...) is a runtime-"
            "updatable parameter, not a compile-time literal. "
            "Executed 2026-08-03."),
        "measured": (
            "c = Constant(1.0, name='src'); scheme built with "
            "b = c*v*dx on an 8x8 grid. First solve 25.01 s wall "
            "(JIT included), max(u_h) = 0.07459830. Then "
            "c.value = 2.0 and scheme.solve() again: 0.0003 s and "
            "max(u_h) = 0.14919660 — exactly 2.000000x, as a linear "
            "problem must give. No recompilation, no new scheme."),
        "measured_against_the_float_alternative": (
            "The same experiment run both ways on an 8x8 grid, "
            "counting the .so files under "
            "<dunePyDir>/python/dune/generated before and after each "
            "scheme construction:\n"
            "  b = 7.3125*v*dx  (first time)   27.424 s, .so 161->162\n"
            "  b = 7.3125*v*dx  (again)         0.051 s, .so 162->162\n"
            "  b = 9.8125*v*dx  (VALUE CHANGED) 25.204 s, .so "
            "162->163  <-- full rebuild\n"
            "  b = Constant(7.3125)*v*dx        25.518 s, .so "
            "163->164\n"
            "  c.value = 9.8125; scheme.solve() 0.00056 s, .so "
            "164->164  <-- no rebuild\n"
            "Both routes gave the identical answer "
            "(max(u_h) = 0.73199583 for the 9.8125 right-hand side), "
            "so this is a pure cost difference: about 25 s of C++ "
            "compilation per changed literal versus half a "
            "millisecond. Measured 2026-08-03."),
        "plain_ufl_constant_fails": (
            "ufl.Constant(1.0) (the non-dune one) raises "
            "AttributeError: 'float' object has no attribute "
            "'ufl_domain' — its first positional argument is the "
            "domain, not the value. Use dune.ufl.Constant."),
        "Signal": (
            "[Performance] Baking a changing coefficient into the form "
            "as a Python float forces a full JIT rebuild on EVERY new "
            "value — a parameter sweep spends ~25 s of C++ "
            "compilation per sample and the generated/*.so count grows "
            "by one each time. Wrap it in dune.ufl.Constant and assign "
            "to .value between solves instead: same answer, no new "
            ".so, 0.00056 s. (Executed 2026-08-03: changing 7.3125 to "
            "9.8125 as a literal cost 25.204 s and .so 162->163; as a "
            "Constant it cost 0.00056 s and .so 164->164, both "
            "reaching max(u_h) = 0.73199583.)"),
    },

    # ── output ──────────────────────────────────────────────────────
    "vtk_output_measured": {
        "description": (
            "gridView.writeVTK(name, celldata=None, pointdata=None, "
            "cellvector=None, pointvector=None, number=None, "
            "subsampling=None, outputType=OutputType.appendedraw, "
            "write=True, nonconforming=False) — signature read from "
            "dune/grid/grid_generator.py and exercised 2026-08-03."),
        "higher_order_is_sampled_at_VERTICES_by_default": (
            "Measured on a 4x4 structuredGrid (25 vertices, 16 cells) "
            "carrying a Lagrange ORDER-2 function (space.size 81): "
            "writeVTK(pointdata=...) produced a .vtu with "
            "NumberOfPoints=25 — i.e. the file carries one value per grid "
            "VERTEX and the 56 remaining P2 dofs are not written at "
            "all, so any viewer sees a vertex-interpolated version "
            "of a quadratic field (the rendering itself was not "
            "inspected). writeVTK(..., subsampling=2) gave "
            "NumberOfPoints=400 and writeVTK(..., "
            "nonconforming=True) gave 64 (4 per cell, per-cell "
            "discontinuous). Nothing warns. Use subsampling >= order "
            "when you are looking at anything above P1, and remember "
            "the file is then a RESAMPLING, not the discrete "
            "function."),
        "boundary_ids": (
            "dune.fem.utility.inspectBoundaryIds(gridView) projects "
            "the boundary ids onto a finiteVolume function, one value "
            "per cell: 0 for an interior cell, the face's id for a cell "
            "with one boundary face, and the AVERAGE of the ids for a "
            "cell with several. On a 4x4 structuredGrid the values "
            "present were {0, 1, 2, 2.5, 3, 4}; the ids are 1 = x-min, "
            "2 = x-max, 3 = y-min, 4 = y-max, and the four corner cells "
            "hold 2, 2.5, 2.5 and 3, so do not read the cell values as "
            "integer tags (re-measured 2026-09-30). "
            "dune.fem.utility.gridWidth(gridView) returned "
            "0.25 on the same grid. Use these instead of guessing at "
            "the ids in a dune.ufl.BoundaryId conditional."),
        "Signal": (
            "[API] There is no XDMFFile and no VTXWriter on a dune "
            "gridView — hasattr(gridView, 'XDMFFile') is False. VTK "
            "output goes through gridView.writeVTK(...), which writes "
            "<name>.vtu directly (measured: writeVTK('out_plain', "
            "pointdata=...) produced exactly ['out_plain.vtu']). "
            "(Executed 2026-08-03.)"),
    },

    # ── adaptation ──────────────────────────────────────────────────
    # ── time series and file format, measured ───────────────────────
    "vtk_time_series_measured": {
        "description": (
            "How to write one file per time step, and which output "
            "encodings exist. Executed 2026-08-03."),
        "number_kwarg": (
            "gridView.writeVTK(name, pointdata={...}, number=step) "
            "appends a ZERO-PADDED five-digit index to the base name. "
            "Measured: number=0,1,2 produced series00000.vtu, "
            "series00001.vtu, series00002.vtu. No .pvd collection file "
            "is written by this call, so ParaView has to open the "
            "sequence by pattern."),
        "outputType": (
            "outputType=dune.grid.OutputType.<name> selects the "
            "encoding; the enum members are ascii, base64, "
            "appendedraw and appendedbase64. "
            "outputType=OutputType.ascii was executed and produced a "
            "readable .vtu."),
        "Signal": (
            "[API] Calling gridView.writeVTK with the SAME name every "
            "time step overwrites the file and you end up with one "
            "frame. Signal: the output directory holds a single .vtu "
            "after a 100-step run. Pass number=step — measured to give "
            "name00000.vtu, name00001.vtu, ... — or embed the step in "
            "the name yourself. (Executed 2026-08-03 on dune-fem "
            "2.12.0.2.)"),
        "sampling_without_vtk": (
            "dune.fem.utility gives you numbers straight out of a "
            "discrete function, which is usually what a check needs "
            "rather than a picture. All executed 2026-08-03 on an 8x8 "
            "structuredGrid holding the interpolant of u = x + 2y:\n"
            "  lineSample(uh, [0.0,0.5], [1.0,0.5], 5) -> "
            "(points, values) with 5 samples; the values ran 1.0 to "
            "2.0, which is exactly u along y=0.5.\n"
            "  pointSample(uh, [0.25,0.25]) -> 0.75, the exact value.\n"
            "  gridWidth(gridView) -> 0.125 on the 8x8 grid, i.e. h.\n"
            "  inspectBoundaryIds(gridView) -> a GridFunction, NOT the "
            "string 'bndId'. Corrected by adversarial audit 2026-08-03: "
            "an earlier revision of this entry reported the return "
            "value as 'bndId', which is only the .name attribute. What "
            "you actually get is a dune.fem.space.finiteVolume discrete "
            "function with ONE dof PER CELL carrying the projected "
            "boundary id, so read it with np.array(r.as_numpy) — "
            "measured shape (64,) on the 8x8 grid with unique values "
            "[0.0, 1.0, 2.0, 2.5, 3.0, 4.0]. 0.0 is an interior cell, "
            "1..4 are the geometric ids, and the fractional 2.5 is a "
            "CORNER cell averaging two boundary faces, so do not treat "
            "these dofs as integer tags.\n"
            "boundarySample and Sampler are importable from the same "
            "module but were NOT exercised."),
    },

    "adaptation_measured": {
        "description": (
            "The 2.12 mark/adapt cycle, executed 2026-08-03."),
        "mark_returns_STATISTICS_not_a_marker": (
            "dune.fem.mark(indicator, tol) builds a GridMarker, CALLS "
            "it immediately, and returns what the call returns: the "
            "(nRefined, nCoarsened) tuple — measured as (-1, -1) "
            "because statistics defaults to False. It is NOT a marker "
            "object. Feeding the return value back in, "
            "dune.fem.adapt(marker, [uh]), therefore fails with "
            "AssertionError: 'only one list of discrete functions can "
            "be passed into the adaptation method' (the tuple is not "
            "callable, so gridAdapt re-dispatches it as the first "
            "discrete function). The correct sequence is two "
            "statements with nothing passed between them: "
            "dune.fem.mark(ind, tol) then dune.fem.adapt([uh, ...]). "
            "If you want a reusable marker object, construct "
            "dune.fem.GridMarker(indicator, tol) yourself and hand "
            "THAT to dune.fem.adapt(marker, [uh]) — measured to work, "
            "32 -> 48 elements."),
        "adaptivity_requires_adaptiveLeafGridView": (
            "dune.fem.adapt([uh, ...]) on a plain aluConformGrid leaf "
            "view raises AssertionError: 'the grid views for all "
            "discrete functions need to support adaptivity'. Reading "
            "gridView.canAdapt on that same plain view raises "
            "AttributeError; on "
            "adaptiveLeafGridView(aluConformGrid(...)) it is True. So "
            "an ALUGrid alone is not enough — wrap it in "
            "dune.fem.view.adaptiveLeafGridView before building the "
            "spaces. CAVEAT worth knowing: that check lives in "
            "_adaptArguments and only runs for the LIST form; passing "
            "a single discrete function returns early and skips it, "
            "which is exactly why globalRefine(level, uh) can fail "
            "silently (see global_refinement_measured)."),
        "upstream_bug_gridView_kwarg": (
            "dune.fem.mark(indicator, tol, gridView=gv) RAISES "
            "AttributeError: \"'GridMarker' object has no attribute "
            "'gridView'\" — UNCONDITIONALLY, on a YaspGrid and on an "
            "ALUGrid alike (both measured). The cause is a typo in "
            "dune/fem/_adaptation.py: GridMarker.__init__ stores "
            "self._gridView and then validates "
            "GridMarker.checkGridView(self.gridView), but the class "
            "exposes no `gridView` attribute (only `indicator` is a "
            "property). dune.fem.markNeighbors inherits the same "
            "break because it forwards gridView=. WORKAROUND: omit "
            "gridView= entirely and let the marker take the grid view "
            "from the indicator — which means the indicator must be a "
            "DISCRETE FUNCTION (e.g. finiteVolume(gv).interpolate("
            "...)), not a bare UFL expression. dune.fem.markNeighbors "
            "forwards the same kwarg and was measured to raise the "
            "identical AttributeError."),
        "working_cycle": (
            "gv = adaptiveLeafGridView(aluConformGrid(...)); "
            "ind = finiteVolume(gv).interpolate(<estimator>, "
            "name='ind'); dune.fem.mark(ind, tol); "
            "dune.fem.adapt([uh]). Measured: 32 -> 48 elements, "
            "lagrange(order=1).size 25 -> 33, and uh was prolonged "
            "onto the new space. No space.update() call is needed or "
            "possible — hasattr(space, 'update') is False."),
        "global_refinement_measured": (
            "dune.fem.globalRefine has TWO call forms and they behave "
            "very differently. Passing the HIERARCHICAL GRID refined "
            "in every configuration tried: 16 -> 64 on structuredGrid "
            "4x4 (plain and adaptiveLeaf views) and 32 -> 64 on "
            "aluConformGrid, both with and without a lagrange space "
            "already built on the view (with a space alive the space "
            "resized too: structuredGrid 25 -> 81 dofs, "
            "aluConformGrid 25 -> 41). WHAT THAT ROW DOES NOT SAY, and "
            "the reason it is a silent-wrong trap rather than a "
            "convenience (added by adversarial audit 2026-08-03): "
            "resizing a live space through the hierarchical grid ZEROES "
            "the discrete functions on it. Measured on an 8x8 "
            "structuredGrid with uh = space.interpolate(1.0): before "
            "globalRefine(1, gridView.hierarchicalGrid) space.size 81 "
            "and max(uh.as_numpy) 1.0; after, space.size 289 and "
            "max(uh.as_numpy) 0.0 — no exception, no warning. Only the "
            "adapt path (dune.fem.adapt([uh]) on an adaptive ALUGrid "
            "view) prolongs. "
            "Passing a DISCRETE FUNCTION — the form that also "
            "prolongs it — only works on an adaptive ALUGrid view: "
            "  adaptiveLeafGridView(aluConformGrid): "
            "(32 elems, 25 dofs) -> (64, 41). WORKS.\n"
            "  aluConformGrid plain leaf view: RuntimeError "
            "'Method numBlocks() called on non-adaptive block "
            "mapper', thrown at dune/fem/space/mapper/"
            "indexsetdofmapper.hh:228 and printed behind a "
            "NotImplemented [numBlocks:<path>:228] frame that "
            "DUNE_THROW assembles at run time — loud, at least.\n"
            "  structuredGrid, PLAIN view: (16, 25, 25) -> "
            "(16, 25, 25). SILENT NO-OP.\n"
            "  structuredGrid wrapped in adaptiveLeafGridView "
            "(canAdapt reads True!): (16, 25, 25) -> (16, 25, 25). "
            "ALSO A SILENT NO-OP.\n"
            "Passing the hierarchical grid AND functions together is "
            "deprecated and warns."),
        "Signal_globalRefine": (
            "[Numerical] dune.fem.globalRefine(level, uh) is a SILENT "
            "NO-OP on a YaspGrid — the element count, the space size "
            "and len(uh.as_numpy) all come back unchanged, with no "
            "exception and no warning, even when "
            "gridView.canAdapt reports True. A refinement study "
            "written that way produces the same numbers at every "
            "'level'. Assert that gridView.size(0) actually grew "
            "after each refinement, or refine via "
            "globalRefine(level, gridView.hierarchicalGrid) and "
            "rebuild the space. (Executed 2026-08-03 on dune-fem "
            "2.12.0.2, all four grid/view combinations measured.)"),
        "mark_lives_on_the_grid_not_the_leaf_view": (
            "hasattr(structuredGrid(...), 'mark') is False while "
            "hasattr(gridView.hierarchicalGrid, 'mark') is True. The "
            "per-element mark() the marker calls internally is on the "
            "ADAPTIVE grid view (dune.fem.view.adaptiveLeafGridView) "
            "or on the hierarchical grid — not on a plain leaf view."),
        "Signal": (
            "[API] An adaptation loop copied from a tutorial that "
            "passes gridView= to dune.fem.mark dies immediately with "
            "AttributeError: 'GridMarker' object has no attribute "
            "'gridView'. That is an upstream defect in dune-fem "
            "2.12.0.2, not a mistake in your indicator — drop the "
            "kwarg and pass a discrete-function indicator instead. "
            "(Executed 2026-08-03 on both YaspGrid and ALUGrid.)"),
    },

    # ── process exit code is not a verdict ──────────────────────────
    "intermittent_teardown_abort_measured": {
        "description": (
            "OBSERVED, NOT FULLY CHARACTERISED — recorded because the "
            "operational consequence is large and the symptom is easy "
            "to misread. A script that builds several ALUGrid and "
            "YaspGrid objects (plain and adaptiveLeafGridView) and "
            "refines them printed all of its output correctly and "
            "then ABORTED DURING INTERPRETER TEARDOWN with a PETSc "
            "signal handler dump ('Caught signal number 11 SEGV', "
            "then 'terminate called after throwing an instance of "
            "Dune::ExceptionStream<Dune::Petsc::Exception>') and exit "
            "code 134. The backtrace is inside "
            "ALUGrid::TetraTop<...>::~TetraTop via the shared_ptr "
            "release of the generated hierarchicalgrid module, i.e. "
            "grid destruction after main."),
        "reproducibility": (
            "It depends on WHICH refinement path ran, and it is not "
            "always intermittent. Measured 2026-08-03 (the first "
            "sentence) and re-measured under adversarial audit the "
            "same day (the rest):\n"
            "  * A script that calls dune.fem.globalRefine(level, uh) "
            "on adaptiveLeafGridView(aluConformGrid(...)) — the "
            "discrete-function form, which builds a "
            "BasicVirtualizedRestrictProlong — aborted on 7 of 7 runs "
            "with byte-identical stdout (md5 equal across all six "
            "repeats). The backtrace is in "
            "~BasicVirtualizedRestrictProlong<ALUGrid<2,2,...>>, "
            "released from the generated femspace module at "
            "interpreter teardown.\n"
            "  * A script that refines the same mix of grids through "
            "globalRefine(level, gridView.hierarchicalGrid) instead "
            "exited 0 on 3 of 3 runs.\n"
            "  * A larger mix of grids and refinement paths in one "
            "script gave "
            "134 / 0 / 134 over three runs, so that one IS "
            "intermittent.\n"
            "Treat it as: reliable for the restrict-prolong path, "
            "intermittent for bigger mixes, absent for plain "
            "hierarchical-grid refinement. Either way the results are "
            "printed before the abort."),
        "Signal": (
            "[Integration] A DUNE-fem run can produce every correct "
            "result and STILL exit non-zero, because the abort "
            "happens after the last print, during grid destruction. "
            "Judging a DUNE job by returncode alone will therefore "
            "mark a good run as failed intermittently. Have the "
            "script write its results to a file (or print a terminal "
            "sentinel line) and treat THAT as the success criterion, "
            "with the returncode as a secondary signal. (Executed "
            "2026-08-03: same script, exit codes 134 / 0 / 134, "
            "identical stdout.)"),
    },

    # ── threading ───────────────────────────────────────────────────
    "threading_measured": {
        "description": (
            "dune.fem.threading.max and .use are ATTRIBUTES (read and "
            "assign them); .useMax is a CALLABLE "
            "(<built-in method useMax of PyCapsule object>, "
            "callable() is True). At startup max = DUNE_NUM_THREADS, "
            "else OMP_NUM_THREADS, else the machine's core count, and "
            "use = DUNE_NUM_THREADS, else OMP_NUM_THREADS, else 1 "
            "(re-measured 2026-09-30: OMP_NUM_THREADS=2 gave max 2 and "
            "use 2, DUNE_NUM_THREADS=3 gave max 3 and use 3 and wins "
            "over OMP_NUM_THREADS). Executed 2026-08-03 with neither "
            "variable set: threading.max reported the machine's core "
            "count and threading.use == 1. Assigning "
            "dune.fem.threading.use = 2 took effect immediately (read "
            "back as 2), and calling dune.fem.threading.useMax() "
            "raised use to threading.max."),
        "Signal": (
            "[Performance] With neither DUNE_NUM_THREADS nor "
            "OMP_NUM_THREADS set, dune-fem assembles and solves on ONE "
            "thread (threading.use == 1) whatever threading.max "
            "reports; if either is set, use and max both start at that "
            "value. If a DUNE run pegs a single core while the machine "
            "idles, read threading.use and assign it or call "
            "threading.useMax(). (Executed 2026-08-03; the environment "
            "variables re-measured 2026-09-30.)"),
    },

    # ── convergence behaviour, stated WITHOUT the measured answer ────
    #
    # An earlier revision of this section shipped the full measured EOC
    # table (per-order L2/H1 orders and the finest absolute errors) for
    # a named manufactured solution. That is the ANSWER to a convergence
    # study, and it was reachable from
    # knowledge(topic="overview", solver="dune") — an agent asked to run
    # such a study could read the result out of the tool instead of
    # computing it. The numbers stay where they belong: in the Tier-2
    # gate that re-measures them on every run,
    # scripts/tier2_fixtures/dune/poisson_mms_convergence/ (fixture.json
    # records the calibration, source.py re-derives it). Removed by
    # adversarial audit 2026-08-03 after the contamination guard in
    # tests/test_knowledge_not_contaminated.py flagged ten hits in this
    # file under the pattern r"\bEOCs?\b[^.\n]{0,40}\d\.\d{2,}".
    "convergence_behaviour_2d_poisson": {
        "description": (
            "Lagrange order k on 2D Poisson with exact Dirichlet data "
            "reaches the textbook rates on this install — L2 order "
            "k+1, H1 order k — on BOTH structuredGrid (cubes, k = 1, 2, "
            "3) and aluConformGrid (simplices, k = 1, 2). No observed "
            "orders or error magnitudes are reproduced here on purpose; "
            "measure them yourself. What this entry is for is the "
            "SETUP that makes such a study valid on dune-fem, and the "
            "two ways it silently goes wrong."),
        "how_to_set_one_up": (
            "Build f symbolically from the chosen u* with "
            "f = -div(grad(u_exact)) so no hand-derived source can be "
            "mistranscribed; impose the exact data with "
            "DirichletBC(space, u_exact) and CHECK it is in the "
            "galerkin([...]) list; compute the norms with "
            "dune.fem.integrate(..., order >= 2k+4) or leave order "
            "unset; and set linear.tolerance well below the finest "
            "discretisation error you expect. Refine by REBUILDING the "
            "grid at each level — dune.fem.globalRefine(level, uh) is a "
            "silent no-op on a YaspGrid (see "
            "adaptation_measured.Signal_globalRefine), so a study "
            "written that way reports the same error at every level."),
        "Signal": (
            "[Numerical] If a DUNE-fem convergence table shows an "
            "observed L2 order near 0 instead of k+1, and the errors "
            "sit at an O(1) mesh-independent plateau, suspect the "
            "boundary conditions before the discretisation — the "
            "silent Dirichlet trap produces exactly that curve. If "
            "every level reports the IDENTICAL error, the refinement "
            "never happened — assert gridView.size(0) grew. If the "
            "order is right but the level-to-level ratio flattens at "
            "the finest level, suspect linear.tolerance. (Executed "
            "2026-08-03.)"),
    },

    # ── natural (Neumann / Robin) boundary conditions ───────────────
    "natural_bc_measured": {
        "description": (
            "How a flux or Robin condition is written on dune-fem "
            "2.12.0.2 — there is no facet-tag mechanism, so the "
            "boundary term is masked with a UFL conditional on the "
            "coordinate. Executed 2026-08-03."),
        "neumann_recipe": (
            "L = g * conditional(gt(x[0], 1-tol), 1.0, 0.0) * v * ds, "
            "added to the right-hand side of "
            "galerkin([a == L, <dirichlet bcs>]). Verified on "
            "-Laplace(u) = 0 with u=0 on x=0 and du/dn = 1 on x=1, "
            "whose answer is u = x: the P1 solution came back with "
            "||u_h - x||_L2 = 3.233e-16 on an 8x8 structuredGrid, i.e. "
            "exact. The tolerance in the conditional is compared "
            "against the FACET QUADRATURE POINTS, which for a "
            "structuredGrid facet on x=1 are exactly at x=1."),
        "robin_recipe": (
            "du/dn + alpha*(u - u_inf) = 0 becomes "
            "a += alpha * mask * u * v * ds and "
            "L += alpha * u_inf * mask * v * ds. Verified with "
            "alpha=3, u_inf=2 on the same geometry, whose answer is "
            "u = alpha*u_inf/(1+alpha) * x = 1.5*x: "
            "||u_h - 1.5*x||_L2 = 3.238e-16."),
        "Signal_unmasked_ds": (
            "[Numerical] Forgetting the coordinate mask applies the "
            "flux to the WHOLE boundary and nothing warns you. "
            "Signal: on the u=x problem above, writing g*v*ds instead "
            "of g*mask*v*ds gave max(u_h) = 2.163782 where the correct "
            "answer is 1.0 — converged, no exception, roughly double. "
            "A flux term whose result is a small integer multiple of "
            "the expected one is almost always this. (Executed "
            "2026-08-03 on dune-fem 2.12.0.2.)"),
        "ds_with_a_subdomain_id_DOES_work": (
            "ds(id) is supported and the ids are GEOMETRIC and "
            "1-BASED, not user-assigned. dune/fem/misc/"
            "boundaryidprovider.hh gives, for YaspGrid, "
            "boundaryId = intersection.boundary() ? "
            "intersection.indexInInside()+1 : 0 — so on a 2D "
            "structuredGrid the four ids are 1 = x-min (left), "
            "2 = x-max (right), 3 = y-min (bottom), 4 = y-max (top). "
            "Measured on an 8x8 grid by assembling 1*v*ds(k): "
            "ds(1)..ds(4) each summed to exactly 1.0 (one unit edge), "
            "ds(5) summed to 0.0, plain ds summed to 4.0. Solving "
            "-Laplace(u)=0 with u=0 on x=0 and unit flux on ds(2) "
            "returned max(u_h) = 1.00000000, the exact answer u = x. "
            "ALUGrid uses a DIFFERENT provider "
            "(intersection.impl().boundaryId(), i.e. whatever the "
            "mesh file carries), so the 1..2*dim numbering above is a "
            "YaspGrid fact, not a DUNE-wide one — on an ALUGrid "
            "cartesianDomain ds(1) also measured 1.0, but do not "
            "assume the mapping for an imported mesh."),
        "Signal_ds_with_a_tag": (
            "[Numerical] A ds(id) term whose id happens to be a "
            "CONSTRAINED boundary contributes nothing, and looks "
            "exactly like an unsupported feature. Signal: "
            "g*v*ds(1) as the only right-hand side, with the "
            "Dirichlet condition ALSO on x=0, gave "
            "max(u_h) = 0.00000000 — every test function on id 1 had "
            "been eliminated by the constraint. The same form on "
            "ds(2) gave max(u_h) = 1.00000000. Before concluding that "
            "ds(id) is broken, check which physical edge that id is: "
            "for a 2D YaspGrid it is indexInInside()+1, so 1=left, "
            "2=right, 3=bottom, 4=top. ds(0) is not a wildcard — it "
            "raises AssertionError at dune/ufl/linear.py:208, "
            "`assert type(id) is int and id > 0`. (Executed "
            "2026-08-03 on dune-fem 2.12.0.2; this entry REPLACES an "
            "earlier claim in this catalog that ds(id) is silently "
            "empty, which came from a test that put its Dirichlet "
            "condition on the very edge it was integrating over.)"),
        "Signal_component_bc_corner": (
            "[Numerical] Two COMPONENT-WISE DirichletBCs on edges that "
            "MEET lose one of the two constraints at the shared corner "
            "dof. Signal: with DirichletBC(space,[0,None],x[0]<tol) "
            "and DirichletBC(space,[None,0],x[1]<tol) on a vector "
            "space, the dof at (0,0) came back u_x = 1.910e-06 instead "
            "of 0 while u_y was 0 — and swapping the two BCs in the "
            "list changed nothing. Two component-wise BCs on the SAME "
            "edge DO merge (both components measured 0). The resulting "
            "global error is O(h^2), so it looks like discretisation "
            "error; detect it by reading the constrained dof back and "
            "comparing it with what you asked for. (Executed "
            "2026-08-03 on dune-fem 2.12.0.2, 4x4 and 8x8 grids.)"),
        "Signal_pointwise_indicator": (
            "[API] A DirichletBC indicator is evaluated PER BOUNDARY "
            "FACET, not per dof, so an indicator that is true only at "
            "a point selects nothing. Signal: adding "
            "DirichletBC(space,[0,0], And(x[0]<tol, x[1]<tol)) to pin "
            "one corner node changed the solution by exactly zero — "
            "the corner dof kept the value it had without that BC, to "
            "all printed digits. Use edge-sized indicators. (Executed "
            "2026-08-03.)"),
    },

    # ── matrices out of dune-fem ────────────────────────────────────
    "assemble_measured": {
        "description": (
            "dune.fem.assemble(form, space=None, gridView=None, "
            "order=None) is the supported way to get a MATRIX out of "
            "dune-fem without going through a scheme. Executed "
            "2026-08-03."),
        "what_it_returns": (
            "A bilinear form (both a TrialFunction and a TestFunction) "
            "assembles to an object of type LinearOperator whose only "
            "conversion attribute is .as_numpy — measured "
            "[a for a in dir(A) if 'numpy' in a or 'petsc' in a or "
            "'istl' in a] == ['as_numpy']. .as_numpy is ALREADY a scipy "
            "csr_matrix — measured .format == 'csr', type csr_matrix, "
            "and A.tocsr() is A is True — so slice and fancy-index it "
            "directly; no conversion is needed. (An earlier revision of "
            "this entry said COO and required .tocsr(); RETRACTED by "
            "adversarial audit 2026-08-03. "
            "dune/fem/operator/__init__.py imports only "
            "scipy.sparse.csr_matrix and 'coo_matrix' occurs nowhere "
            "under site-packages/dune.) On a 24x24 P1 structuredGrid "
            "the stiffness matrix came back 625x625 with 5329 "
            "nonzeros."),
        "bc_argument_is_silently_ignored": (
            "[API] assemble(form, bc) ACCEPTS a dune.ufl.DirichletBC and "
            "SILENTLY IGNORES it. Measured 2026-08-03 on a 16x16 P1 "
            "structuredGrid: assemble(form).as_numpy and "
            "assemble(form, DirichletBC(space,[0])).as_numpy both came "
            "back with nnz 2401 and max|difference| exactly 0.0 — the "
            "constrained matrix is byte-identical to the unconstrained "
            "one. Nothing raises and nothing warns. Signal: you passed "
            "a BC to assemble() and the solution does not respect it; "
            "verify by assembling twice, with and without the bc "
            "argument, and comparing nnz and max|A1-A2| — if the "
            "difference is 0.0 the argument did nothing. Constraints "
            "are applied by the SCHEME, so build a galerkin([a == L, "
            "bc]) and take its matrix if you need a constrained "
            "operator."),
        "no_boundary_conditions": (
            "assemble() knows nothing about DirichletBCs — the matrix "
            "is the raw Galerkin matrix. That is what makes it the "
            "right tool for eigenvalue problems, where the constrained "
            "rows must be DELETED rather than replaced by identity "
            "rows."),
        "Signal": (
            "[API] There is no eigen/eigs/eigenvalue entry point in "
            "dune.fem at all, and `import dune.fem.solver` raises "
            "ModuleNotFoundError. Signal: any spectral problem has to "
            "go dune.fem.assemble -> .as_numpy.tocsr() -> "
            "scipy.sparse.linalg.eigsh (or PETSc/SLEPc through the "
            "petsc storage). Verified end to end: interior submatrices "
            "of a 24x24 P1 grid gave the analytic Dirichlet Laplacian "
            "eigenvalues pi^2(m^2+n^2) to within 6e-03 relative. "
            "(Executed 2026-08-03 on dune-fem 2.12.0.2.)"),
    },

    # ── which dune sub-packages actually exist here ─────────────────
    "companion_modules_measured": {
        "description": (
            "Which dune sub-packages a plain pip-installed (PyPI) "
            "dune-fem install exposes, measured with importlib on "
            "2026-08-03. "
            "This matters because several capabilities the catalog "
            "used to advertise live in SEPARATE packages."),
        "importable": [
            "dune.fem, dune.grid, dune.ufl, dune.geometry, "
            "dune.istl, dune.common (core)",
            "dune.alugrid — the adaptive/simplicial grid manager",
            "dune.fem.utility (gridWidth, inspectBoundaryIds, "
            "lineSample, pointSample, boundarySample, Sampler, "
            "algorithm)",
            "dune.fem.view (adaptiveLeafGridView, filteredGridView, "
            "geometryGridView)",
            "dune.fem.operator (galerkin, molGalerkin, h1, linear, "
            "linearOperator, load)",
            "dune.fem.model, dune.fem.plotting, dune.generator",
        ],
        "NOT_importable_here": [
            "dune.femdg AND dune.fem.dg — ModuleNotFoundError. "
            "dune-fem-dg is a separate package; the SSP Runge-Kutta "
            "steppers, the Bassi-Rebay / CDG / LDG operators and the "
            "limiters it provides are NOT available from a plain "
            "dune-fem install. Write the DG operator with the ordinary "
            "galerkin scheme and your own explicit stepper instead.",
            "dune.vem — ModuleNotFoundError. No Virtual Element "
            "Method here.",
            "dune.polygongrid, dune.spgrid, dune.uggrid, dune.mmesh — "
            "ModuleNotFoundError.",
            "dune.fem.solver — ModuleNotFoundError; solvers are chosen "
            "with the scheme's solver= argument.",
            "dune.fem.parameter as a MODULE — ModuleNotFoundError; it "
            "is an ATTRIBUTE of dune.fem (append, exists, get, log, "
            "write).",
        ],
        "measured_inventories": {
            "dune.grid": (
                "albertaGrid, cartesianDomain, equidistantCoordinates, "
                "equidistantOffsetCoordinates, gridFunction, onedGrid, "
                "reader, string2dgf, structuredGrid, "
                "tensorProductCoordinates, ugGrid, yaspGrid, Marker, "
                "OutputType, Partitions, PartitionType"),
            "dune.grid.reader": (
                "dgf, dgfString, gmsh, meshio, structured"),
            "dune.grid.OutputType": (
                "ascii, base64, appendedraw, appendedbase64"),
            "dune.fem.comm": (
                "rank, size, barrier, broadcast, gather, scatter, sum, "
                "min, max — measured rank 0 / size 1 in a serial run"),
        },
        "Signal": (
            "[API] Capabilities named in DUNE's own documentation are "
            "not necessarily in your install. Signal: 'import "
            "dune.femdg' and 'import dune.vem' both raise "
            "ModuleNotFoundError on a pip-installed (PyPI) dune-fem "
            "2.12.0.2 env, so any plan that depends on dune-fem-dg's "
            "SSP-RK steppers or on VEM spaces fails at the first import — "
            "check with importlib before designing around them. "
            "(Executed 2026-08-03.)"),
    },

    # ── phantom APIs the catalog used to name ───────────────────────
    "phantom_apis_checked": {
        "description": (
            "Names that appear in tutorials, older releases or earlier "
            "revisions of this catalog and DO NOT EXIST in the "
            "installed package. Checked 2026-08-03 by hasattr on the "
            "imported modules and by grepping the whole install tree "
            "(python sources AND the installed C++ headers)."),
        "dune.fem.space.product_space": (
            "ABSENT. hasattr(dune.fem.space, 'product_space') is False "
            "and the string 'product_space' does not occur anywhere "
            "under site-packages/dune or include/dune. The real "
            "multi-field factories are dune.fem.space.product, "
            ".composite and .combined (all with signature "
            "(*spaces, **kwargs)). The shipped mixed-methods template "
            "imports it as `from dune.fem.space import product as "
            "product_space`, which is why the alias looked like an "
            "API."),
        "dune.fem.space.raviartthomas": (
            "ABSENT (lowercase). The Python factory is raviartThomas "
            "(camelCase); only the C++ header is lowercase. "
            "Re-confirmed at runtime 2026-08-03: "
            "hasattr(dune.fem.space, 'raviartThomas') is True, "
            "hasattr(..., 'raviartthomas') is False."),
        "space.update()": (
            "ABSENT. hasattr(space, 'update') is False and no "
            "\"update\" pybind11 binding exists under "
            "include/dune/fempy/. After adaptation the space resizes "
            "itself; the discrete functions passed to "
            "dune.fem.adapt([uh, ...]) are what gets prolonged."),
    },

    # ── module-level API inventory ──────────────────────────────────
    "module_inventory_2_12": {
        "description": (
            "Names CONFIRMED PRESENT by dir()/hasattr on the installed "
            "modules, 2026-08-03. Use this to check a name exists "
            "BEFORE writing a script around it. These are curated "
            "SUBSETS of the real dir() output, so absence from this "
            "list is NOT evidence that a name is missing — for that, "
            "see phantom_apis_checked or just run hasattr yourself."),
        "dune.fem": (
            "GridFunction, GridMarker, Parameter, SpaceMarker, adapt, "
            "assemble, comm, discretefunction, doerflerMark, function, "
            "globalRefine, gridAdapt, integrate, loadBalance, mark, "
            "markNeighbors, model, operator, parameter, plotting, "
            "scheme, setVerbosity, space, spaceAdapt, threading, "
            "utility, view"),
        "dune.fem.space": (
            "bdfm, bdm, combined, composite, dganisotropic, "
            "dglagrange, dglagrangelobatto, dglegendre, dglegendrehp, "
            "dgonb, dgonbhp, finiteVolume, interpolate, lagrange, "
            "lagrangehp, p1Bubble, product, project, rannacherTurek, "
            "raviartThomas"),
        "dune.fem.scheme": (
            "dg, dgGalerkin, galerkin, h1, h1Galerkin, linearized, "
            "molGalerkin, solve"),
        "dune.fem.view": (
            "adaptiveLeafGridView, filteredGridView, geometryGridView"),
        "dune.grid": (
            "albertaGrid, cartesianDomain, gridFunction, onedGrid, "
            "reader, structuredGrid, tensorProductCoordinates, "
            "ugGrid, yaspGrid; reader is an enum with members dgf, "
            "dgfString, gmsh, meshio, structured"),
        "dune.alugrid": (
            "aluConformGrid, aluCubeGrid, aluGrid, aluSimplexGrid, "
            "reader"),
        "dune.ufl": (
            "BoundaryId, BoxDirichletBC, Constant, DirichletBC, "
            "GridFunction, MixedFunctionSpace, NamedConstant, Space, "
            "cell"),
        "dune.fem.threading": (
            "max and use are attributes; useMax is a callable"),
        "dune.fem.parameter": "append, exists, get, log, write",
        "dune.fem.utility": (
            "gridWidth, inspectBoundaryIds, lineSample, pointSample, "
            "boundarySample, Sampler"),
        "there_is_no_dune.fem.solver_module": (
            "import dune.fem.solver raises ModuleNotFoundError — "
            "solvers are selected by the `solver=` argument to the "
            "scheme, not by importing a solver module."),
        "some_of_these_are_ATTRIBUTES_not_modules": (
            "dune.fem.parameter and dune.fem.threading come from the "
            "compiled dune/fem/_fem.so and are attributes of "
            "dune.fem, NOT importable modules: `import "
            "dune.fem.parameter` raises ModuleNotFoundError while "
            "`import dune.fem; dune.fem.parameter.get(...)` works. "
            "Measured 2026-08-03."),
    },
}


# ── Cross-cutting pitfalls appended to KNOWLEDGE["poisson"] ──────────
#
# These live under `poisson` (rather than only under `_general`) because
# knowledge(topic="pitfalls", solver="dune") walks supported_physics()
# and never sees `_general`, and poisson is the physics every model
# touches first.

EXECUTED_PITFALLS: list[str] = [
    (
        "[Numerical] A dune.ufl.DirichletBC only takes effect if it "
        "is IN THE LIST passed to the scheme — galerkin([a == b, dbc], "
        "...). There is no separate apply step, and building the "
        "scheme as galerkin(a == b, space=space) after constructing a "
        "dbc object silently drops it. Signal: the run COMPLETES and "
        "scheme.solve() returns {'converged': True, ...}; the tells "
        "are linear_iterations in the thousands where the constrained "
        "problem needs tens, and max(uh.as_numpy) exploding. "
        "(Executed 2026-08-03, dune-fem 2.12.0.2, 16x16 "
        "structuredGrid, -Delta u = 2 pi^2 sin(pi x) sin(pi y): with "
        "the BC in the list, converged=True, linear_iterations=1, L2 "
        "err 1.899705e-03; with the SAME form and the BC omitted, "
        "converged=True, linear_iterations=23935, L2 err "
        "7.510512e+14.)"
    ),
    (
        "[Numerical] The optional third argument of DirichletBC is a "
        "UFL conditional over SpatialCoordinate(space), and nothing "
        "checks that it selects any facet. A wrong sign or threshold "
        "degrades to no boundary condition at all. Signal: identical "
        "symptoms to omitting the BC — converged=True with thousands "
        "of linear iterations and a solution many orders of magnitude "
        "too large. (Executed 2026-08-03: "
        "DirichletBC(space, 0, conditional(x[0] < -1.0, 1, 0)) on "
        "[0,1]^2 gave converged=True, linear_iterations=23935, L2 err "
        "7.510512e+14 — byte-identical to the omitted-BC run. The "
        "same BC with conditional(x[0] < 1e-8, 1, 0) gave 45 "
        "iterations and a well-posed L2 err of 2.676075e+00.)"
    ),
    (
        "[API] The `solver=` string is not validated in Python; it is "
        "forwarded as the C++ parameter 'fem.solver.linear.method' and "
        "checked when the scheme is constructed. For the default "
        "storage the accepted values are exactly cg, gmres and "
        "bicgstab. Signal: solver='conjugate_gradient' raises "
        "RuntimeError 'ParameterInvalid', followed by a "
        "'Valid values are:' line listing gmres, cg, bicgstab, and by "
        "the offending key fem.solver.linear.method — an error that "
        "names a parameter you never wrote. Only those two fragments "
        "are literals; DUNE_THROW adds the exception name and the "
        "[getEnumeration:<path>/dune/fem/io/parameter/reader.hh:300] "
        "frame at run time and reader.hh:290-296 emits the key, the "
        "word invalid and the list as separate stream insertions "
        "joined by newlines. Direct solves use a tuple instead: "
        "solver=('suitesparse', 'umfpack'). (Executed 2026-08-03.)"
    ),
    (
        "[API] Parameter keys beginning with 'newton.' are deprecated "
        "in favour of 'nonlinear.', and dune-fem REWRITES them for "
        "you. Signal: parameters={'newton.tolerance': 1e-10} emits "
        "UserWarning \"Warning: the parameter key 'newton' is "
        "deprecated. Replace with 'nonlinear' to avoid this "
        "warning!\", with both key names wrapped in ANSI colour codes "
        "(so grep for 'is deprecated. Replace with', not for the whole "
        "line), and scheme.parameters comes back "
        "as {'nonlinear.tolerance': 1e-10, ...}. Passing BOTH "
        "'newton.tolerance' and 'nonlinear.tolerance' is accepted with "
        "no error at all and the newton.* value is silently dropped — "
        "only the nested newton.linear.* / nonlinear.linear.* "
        "collision raises KeyError. (Executed 2026-08-03.)"
    ),
    (
        "[API] A dune-fem space reports a SIMPLEX UFL cell whatever "
        "the grid is really made of, because "
        "dune/ufl/__init__.py::_cell maps dimension straight to "
        "ufl.triangle / ufl.tetrahedron. Signal: on "
        "structuredGrid([0,0],[1,1],[4,4]) — 16 elements of type "
        "'quadrilateral' — space.cell() returns 'triangle' and "
        "str(space.ufl_element()) prints '<Lagrange1 on a triangle>' "
        "AT EVERY ORDER (order 2 and 3 print 'Lagrange1' too, so the "
        "UFL element hides the degree as well as the cell). Read the cell "
        "shape from gridView.type or from "
        "{e.type for e in gridView.elements}, never from the UFL "
        "element — and NOT from the dof count either: continuous "
        "Lagrange has 25 dofs at order 1 on both the 16-cube grid and "
        "the 32-triangle aluConformGrid over the same domain, since "
        "they share the same vertices. Only the element count (16 vs "
        "32) and the geometry type separate them. (Executed "
        "2026-08-03.)"
    ),
    (
        "[API] There is no order-0 Lagrange space: "
        "lagrange(gridView, order=0) raises KeyError 'Parameter error "
        "in LagrangeSpace with order=0: order has to be greater or "
        "equal to 1'. Signal: the FEniCS habit of asking for a "
        "('DG', 0) / order-0 space for a piecewise-constant field "
        "dies here; use dglagrange(gridView, order=0) or "
        "finiteVolume(gridView), both of which gave .size == 16 == "
        "the cell count on a 4x4 grid. (Executed 2026-08-03.)"
    ),
    (
        "[Numerical] dune.fem.globalRefine(level, uh) — the call form "
        "that is supposed to refine AND prolong — is a SILENT NO-OP "
        "on a YaspGrid (structuredGrid). Signal: gridView.size(0), "
        "space.size and len(uh.as_numpy) all come back UNCHANGED with "
        "no exception and no warning, so every 'level' of a "
        "refinement study reports the same error. It works only on "
        "adaptiveLeafGridView(aluConformGrid(...)); on a plain "
        "ALUGrid leaf view it at least raises RuntimeError "
        "'Method numBlocks() called on non-adaptive block mapper' — "
        "the DUNE_THROW at dune/fem/space/mapper/indexsetdofmapper.hh:"
        "228, printed with a NotImplemented [numBlocks:<path>:228] "
        "frame that the throw macro builds at run time. "
        "Assert the element count grew, or refine with "
        "globalRefine(level, gridView.hierarchicalGrid) — which does "
        "work everywhere (16 -> 64 on a 4x4 YaspGrid) but does not "
        "prolong your functions. (Executed 2026-08-03, all four "
        "grid/view combinations measured: yasp plain and yasp "
        "adaptiveLeaf both (16, 25, 25) -> (16, 25, 25); alu "
        "adaptiveLeaf (32, 25, 25) -> (64, 41, 41).)"
    ),
    (
        "[Integration] A DUNE-fem process can print every correct "
        "result and still exit non-zero, because ALUGrid teardown "
        "after the last statement can trip PETSc's signal handler. "
        "Signal: stdout is complete and correct, then '[0]PETSC "
        "ERROR: Caught signal number 11 SEGV' — PETSc stamps the "
        "rank and spells its prefix in capitals, and the greppable "
        "literal is only 'Caught signal number %d %s' in libpetsc — "
        "and 'terminate called after "
        "throwing an instance of "
        "Dune::ExceptionStream<Dune::Petsc::Exception>', exit code "
        "134. It is INTERMITTENT — the same script gave 134 / 0 / 134 "
        "across three runs with byte-identical stdout — so treat a "
        "written results file or a terminal sentinel line as the "
        "success criterion and the returncode as secondary. "
        "(Executed 2026-08-03; a minimal one-grid script never "
        "reproduced it, so this is not a blanket ALUGrid claim.)"
    ),
    (
        "[Performance] The JIT cache is at <sys.prefix>/.cache/dune-py "
        "(inside a venv or conda env) or ~/.cache/dune-py otherwise, "
        "overridable with $DUNE_PY_DIR; the compiled modules are "
        "<that>/python/dune/generated/*.so. Signal: a DUNE-INFO: "
        "Compiling <X> (new) line on stderr — Python logging "
        "rendering 'DUNE-%(levelname)s: %(message)s' from "
        "dune/common/__init__.py:49 around the f-string at "
        "dune/generator/cmakebuilder.py:378 — means a C++ build is "
        "running "
        "and the process will produce no further output for tens of "
        "seconds to minutes. Deleting the cache directory resets the "
        "cost. (Executed 2026-08-03, dune-fem 2.12.0.2: "
        "getDunePyDir() returned the conda env's .cache/dune-py, "
        "~/.dune does not exist, and building five ALUGrid variants "
        "cold took 439 s.)"
    ),
    (
        "[Numerical] ds(id) works, but the ids are GEOMETRIC and "
        "1-based, not names you assign — and an id that lands on a "
        "constrained edge contributes nothing, which is easy to "
        "misread as 'ds(id) is unsupported'. Signal: for a 2D "
        "YaspGrid, dune/fem/misc/boundaryidprovider.hh maps "
        "boundaryId = intersection.indexInInside()+1, so 1=left, "
        "2=right, 3=bottom, 4=top. Measured on an 8x8 grid: "
        "assembling 1*v*ds(k) summed to exactly 1.0 for k=1..4, 0.0 "
        "for k=5, and 4.0 for plain ds; solving -Laplace(u)=0 with "
        "u=0 on x=0 and unit flux on ds(2) gave max(u_h) = "
        "1.00000000, the exact u = x, while the SAME form on ds(1) "
        "gave 0.00000000 because id 1 IS the constrained edge. ds(0) "
        "raises AssertionError at dune/ufl/linear.py:208. ALUGrid "
        "uses a different provider, so do not carry the numbering "
        "over to an imported mesh. (Executed 2026-08-03 on dune-fem "
        "2.12.0.2.)"
    ),
    (
        "[Numerical] An UNMASKED ds integral applies the flux to the "
        "whole boundary. Signal: on -Laplace(u)=0 with u=0 on x=0 and "
        "du/dn=1 on x=1 — whose answer is u=x, so max(u)=1 — writing "
        "the flux as g*v*ds instead of g*mask*v*ds returned "
        "max(u_h) = 2.163782, converged, no exception. A Neumann "
        "result that is a small multiple of what you expect is almost "
        "always a missing mask. The correctly masked version gave "
        "||u_h - x||_L2 = 3.233e-16. (Executed 2026-08-03.)"
    ),
    (
        "[API] A zero right-hand side written as "
        "inner(as_vector([0, 0]), v)*dx raises "
        "\"ValueError: This integral is missing an integration "
        "domain.\" from ufl/measure.py::__rmul__. Signal: UFL folds "
        "the product to Zero() and a Zero carries no domain, so the "
        "measure has nothing to attach to. Use dune.ufl.Constant "
        "values (inner(as_vector([Constant(0.0, name='fx'), "
        "Constant(0.0, name='fy')]), v)*dx) or write the equation as "
        "`a == 0`, which dune-fem accepts as long as `a` still holds "
        "both a trial and a test function. (Executed 2026-08-03 on "
        "dune-fem 2.12.0.2.)"
    ),
    (
        "[API] dune.femdg and dune.vem are NOT part of dune-fem. "
        "Signal: 'import dune.femdg' and 'import dune.vem' both raise "
        "ModuleNotFoundError on a pip-installed (PyPI) dune-fem "
        "2.12.0.2 install, as do dune.fem.dg, dune.polygongrid, "
        "dune.spgrid and dune.uggrid. The SSP Runge-Kutta steppers, "
        "Bassi-Rebay / CDG operators and limiters those packages "
        "provide are therefore "
        "unavailable: write the DG operator with the ordinary galerkin "
        "scheme and your own explicit stepper. dune.alugrid IS "
        "importable. (Executed 2026-08-03.)"
    ),
]
