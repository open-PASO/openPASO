"""scikit-fem generator registry — maps physics_variant -> generator function."""

from .poisson import GENERATORS as _poisson_gen, KNOWLEDGE as _poisson_kn
from .element_survey import (GENERATORS as _esurv_gen,
                             KNOWLEDGE as _esurv_kn)
from .mixed_elements import (GENERATORS as _mixel_gen,
                             KNOWLEDGE as _mixel_kn)
from .heat import GENERATORS as _heat_gen, KNOWLEDGE as _heat_kn
from .linear_elasticity import GENERATORS as _elast_gen, KNOWLEDGE as _elast_kn
from .stokes import GENERATORS as _stokes_gen, KNOWLEDGE as _stokes_kn
from .eigenvalue import GENERATORS as _eigen_gen, KNOWLEDGE as _eigen_kn
from .mixed_poisson import GENERATORS as _mixed_gen, KNOWLEDGE as _mixed_kn
from .convection_diffusion import GENERATORS as _convdiff_gen, KNOWLEDGE as _convdiff_kn
from .biharmonic import GENERATORS as _biharmonic_gen, KNOWLEDGE as _biharmonic_kn
from .nonlinear import GENERATORS as _nonlinear_gen, KNOWLEDGE as _nonlinear_kn
from .wave import GENERATORS as _wave_gen, KNOWLEDGE as _wave_kn
from .adaptive_poisson import GENERATORS as _adapt_gen, KNOWLEDGE as _adapt_kn
from .point_source import GENERATORS as _ps_gen, KNOWLEDGE as _ps_kn
from .schrodinger import GENERATORS as _sch_gen, KNOWLEDGE as _sch_kn
from .contact import GENERATORS as _ct_gen, KNOWLEDGE as _ct_kn
from .hydraulic_resistance import GENERATORS as _hr_gen, KNOWLEDGE as _hr_kn
from .advanced import GENERATORS as _advanced_gen, KNOWLEDGE as _advanced_kn

# Merged generator registry: physics_variant -> callable(params) -> str
GENERATORS: dict[str, callable] = {}
for _g in [
    _poisson_gen, _esurv_gen, _mixel_gen, _heat_gen, _elast_gen, _stokes_gen,
    _eigen_gen, _mixed_gen, _convdiff_gen, _biharmonic_gen,
    _nonlinear_gen, _wave_gen, _adapt_gen, _ps_gen, _sch_gen,
    _ct_gen, _hr_gen, _advanced_gen,
]:
    GENERATORS.update(_g)

# Merged knowledge registry: physics_name -> dict
KNOWLEDGE: dict[str, dict] = {}
for _k in [
    _poisson_kn, _esurv_kn, _mixel_kn, _heat_kn, _elast_kn, _stokes_kn,
    _eigen_kn, _mixed_kn, _convdiff_kn, _biharmonic_kn,
    _nonlinear_kn, _wave_kn, _adapt_kn, _ps_kn, _sch_kn,
    _ct_kn, _hr_kn, _advanced_kn,
]:
    KNOWLEDGE.update(_k)

# General knowledge (not physics-specific)
KNOWLEDGE["_general"] = {
    "description": "scikit-fem general capabilities",
    "element_catalog": {
        "triangular": "P0, P1, P2, P3, P4, Mini, CR (Crouzeix-Raviart), CCR, Morley, Argyris, Hermite, RT0/1/2, BDM1, N1/2/3 (Nedelec), HHJ0/1",
        "quadrilateral": "Q0, Q1, Q2, S2 (Serendipity), BFS (Bogner-Fox-Schmit), RT0/1, N1",
        "tetrahedral": "P0, P1, P2, RT0, N1, Mini, CR, CCR",
        "hexahedral": "H0, H1, H2, S2, RT1, C1",
        "line": "P0, P1, P2, Pp, Hermite, Mini",
    },
    "assembly_types": [
        "@BilinearForm: bilinear forms with u, v (trial/test)",
        "@LinearForm: linear forms with v (test only)",
        "@Functional: scalar integrals/functionals",
        "CellBasis: element interior assembly",
        "FacetBasis: boundary facet assembly",
        "InteriorFacetBasis: interior facet assembly (DG, error estimators)",
        "skfem.supermeshing.intersect / elementwise_quadrature: "
        "mortar / non-matching-interface assembly (domain "
        "decomposition). CORRECTION 2026-08-03: there is NO "
        "MortarFacetBasis in skfem 12.0.1 — it is absent from "
        "skfem, skfem.assembly and every subpackage, and no name "
        "containing 'Mortar' is exported anywhere. Build the "
        "coupling quadrature with skfem.supermeshing.intersect("
        "m1, m2) -> (supermesh, t1, t2) and feed it to two "
        "FacetBasis objects via elementwise_quadrature.",
    ],
    "mesh_types": [
        "MeshTri: init_symmetric, init_sqsymmetric, init_tensor, init_circle, init_lshaped",
        "MeshQuad: init_tensor",
        "MeshTet: init_tensor, init_ball",
        "MeshHex: init_tensor",
        "Mesh.load(): import from meshio (Gmsh, VTK, XDMF, any format)",
        "mesh.refined(n): uniform or adaptive (element index array)",
    ],
    "unique_features": [
        "Pure Python — zero compilation, zero external dependencies beyond numpy/scipy",
        "Assembly-level control — you build the matrices, you choose the solver",
        "50+ element types including Argyris, Morley, Nedelec, Raviart-Thomas",
        "meshio integration for any mesh format",
        "JAX-based automatic differentiation via skfem.autodiff "
        "(OPTIONAL — skfem does not depend on jax; on the openPASO "
        "venv `import skfem.autodiff` raises ModuleNotFoundError, "
        "checked 2026-08-03)",
        "Mortar / non-matching interfaces for domain "
        "decomposition via skfem.supermeshing (intersect, "
        "elementwise_quadrature) — NOT via a MortarFacetBasis "
        "class, which does not exist on 12.0.1",
        "Adaptive refinement: mesh.refined(element_indices)",
    ],
    "cell_basis_extras": {
        "description": (
            "Less-publicized CellBasis surfaces beyond the "
            "AbstractBasis-level pitfalls "
            "([[abstract_basis_extras]]): subdomain restriction "
            "via elements=, boundary projection, point-source / "
            "interpolator dispatch, refinterp backcompat. "
            "Source: skfem/assembly/basis/cell_basis.py."),
        "Signal": (
            "[API] Three CellBasis-specific sharp edges users "
            "hit when going beyond the standard "
            "CellBasis(mesh, elem) usage: "
            "(1) basis.boundary() on a SUBDOMAIN-restricted "
            "CellBasis (one constructed with elements=array) "
            "raises NotImplementedError('Boundary of subdomain "
            "not supported.'). The boundary() factory only "
            "works on full-mesh bases. Workaround: build the "
            "FacetBasis directly with "
            "FacetBasis(m, e, facets=...) and a manually "
            "filtered facet array (e.g. intersect "
            "mesh.boundary_facets() with the facets adjacent "
            "to elements). "
            "(2) basis.probes(x) / basis.interpolator(y) "
            "internally call _base_tensor_order which raises a "
            "BARE NotImplementedError() (empty message) — but "
            "ONLY for ElementComposite, NOT for ElementVector, "
            "contrary to the prior catalog text. Re-measured "
            "2026-08-03 on skfem 12.0.1, MeshTri().refined(2): "
            "Basis(m, ElementVector(ElementTriP1())).probes("
            "[[0.5],[0.5]]) returns a coo_matrix of shape "
            "(2, 50) and .interpolator(...) returns a callable "
            "whose result has shape (2, 1) — both work. "
            "Basis(m, ElementComposite(ElementTriP2(), "
            "ElementTriP1())).probes(...) is the one that raises "
            "the naked NotImplementedError(''). For composite "
            "(Taylor-Hood / Mini) bases split first and "
            "probe/interpolate the scalar sub-bases. "
            "(3) basis.refinterp(y, nrefs=K, Nrefs=N) has a "
            "BACKCOMPAT shim: if both nrefs AND Nrefs are "
            "passed, Nrefs SILENTLY wins ("
            "`if Nrefs is not None: nrefs = Nrefs  # for "
            "backwards compatibility`). Users mixing pre/post-"
            "2022 idioms can get unexpected refinement levels. "
            "The lowercase nrefs is the modern form. "
            "(File walk skfem/assembly/basis/cell_basis.py "
            "2026-06-03.)"
        ),
    },
    "abstract_basis_extras": {
        "description": (
            "Less-publicized behaviors of AbstractBasis (and "
            "consequently CellBasis / FacetBasis / "
            "InteriorFacetBasis): operator overloads, "
            "deprecation paths, error messages, and silent "
            "fallbacks. Source: "
            "skfem/assembly/basis/abstract_basis.py."),
        "operator_overloads": {
            "b1 @ b2": ("CompositeBasis(b1, b2, "
                        "equal_dofnum=True) — DOF numbering "
                        "is SHARED between the two bases; use "
                        "this when both bases index the same "
                        "global DOF vector (same mesh and "
                        "element), e.g. the DG jump pair "
                        "InteriorFacetBasis(m, e, side=0) @ "
                        "InteriorFacetBasis(m, e, side=1) "
                        "(upstream ex07). Two-body mortar "
                        "coupling uses b1 * b2 (upstream ex04)."),
            "b1 * b2": ("CompositeBasis(b1, b2, "
                        "equal_dofnum=False) — DOFs are "
                        "INDEPENDENT; use this for standard "
                        "Taylor-Hood / Mini-style mixed "
                        "formulations where velocity and "
                        "pressure have separate DOF tables. "
                        "GAP FILLED 2026-08-03: BOTH operators "
                        "require the two bases to carry the SAME "
                        "number of quadrature points, which the "
                        "defaults do NOT give you for the very "
                        "case this is advertised for. On "
                        "MeshTri().refined(2), "
                        "Basis(m, ElementTriP2()) * "
                        "Basis(m, ElementTriP1()) raises "
                        "ValueError('Each Basis must have the "
                        "same number of quadrature points.') "
                        "because the default intorder is "
                        "2*elem.maxdeg (P1 -> 3 points, P2 -> 6). "
                        "Pass a matching explicit intorder to "
                        "both, e.g. "
                        "Basis(m, ElementTriP2(), intorder=4) * "
                        "Basis(m, ElementTriP1(), intorder=4), "
                        "which then gives a CompositeBasis with "
                        "N = 106 = 81 + 25. (Verified "
                        "empirically 2026-08-03 on skfem "
                        "12.0.1.)"),
        },
        "Signal": (
            "[API] Five less-publicized sharp edges in "
            "AbstractBasis: "
            "(1) basis.get_dofs(dict) — passing a `dict` of "
            "named boundary lambdas was the pre-2023 idiom; "
            "it now emits DeprecationWarning('Passing dict to "
            "get_dofs is deprecated.'). Replacement: build "
            "named boundaries on the mesh first via "
            "`mesh.with_boundaries({'left': ..., 'right': "
            "...})`, then call `basis.get_dofs({'left', "
            "'right'})` with a SET (not dict) of boundary "
            "names, or pass facets=... individually. "
            "(2) Constructor enforces `mesh.refdom == "
            "elem.refdom` — pairing e.g. MeshHex with "
            "ElementTriP1 raises ValueError('Incompatible "
            "Mesh and Element.') with no further hint. "
            "Common confusion: ElementTriP1 / ElementTriP2 / "
            "ElementTriArgyris (the tri family) work on tri "
            "meshes only; for hex use ElementHex0 / ElementHex1 "
            "/ ElementHex2; for tet use ElementTetP1 / "
            "ElementTetP2 / ElementTetN0. "
            "(3) `b1 @ b2` is NOT the same as `b1 * b2` — "
            "the matmul builds CompositeBasis with "
            "equal_dofnum=True (shared DOF table), the "
            "multiplication builds equal_dofnum=False "
            "(independent DOF tables). Wrong operator on a "
            "Taylor-Hood (velocity * pressure) gives a "
            "singular mass-block; wrong operator on the DG "
            "interior-facet pair (* instead of @) doubles N, "
            "and the jump matrix cannot be added to the N×N "
            "stiffness (ValueError: inconsistent shapes). "
            "(4) Default integration order = 2 * "
            "elem.maxdeg. For P1×P1 bilinear forms this is "
            "exact, but user forms with cubic+ coefficients "
            "(e.g. viscosity that's a P2 field times "
            "gradient terms) are under-integrated silently. "
            "Pass intorder=K explicitly to "
            "CellBasis/FacetBasis when in doubt. "
            "(5) Constructor's doflocs computation is "
            "wrapped in `try / except Exception: "
            "logger.warning('Unable to calculate global DOF "
            "locations.')` — a BROAD catch that uses "
            "logger.warning (not warnings.warn). Default "
            "Python logger config swallows it; users see "
            "the AttributeError on `basis.doflocs` "
            "downstream with no clear cause. To force "
            "stdout: `logging.basicConfig(level=logging."
            "WARNING)` before constructing the basis, or "
            "pass disable_doflocs=True to skip the "
            "computation entirely (useful for large meshes "
            "where doflocs is expensive). "
            "(File walk skfem/assembly/basis/abstract_basis.py "
            "2026-06-03.)"
        ),
    },
    "form_base_class_extras": {
        "description": (
            "skfem.assembly.form.form.Form is the abstract parent "
            "of LinearForm / BilinearForm / TrilinearForm / "
            "Functional. Most documentation refers to the "
            "decorator subclasses; a few user-facing methods live "
            "on the base class and are worth knowing. Source: "
            "skfem/assembly/form/form.py."),
        "Signal": (
            "[Reference]+[API] Five behaviors users encounter via "
            "the Form parent class: "
            "(1) Form.coo_data(*args, **kwargs) is a SILENT "
            "backwards-compat alias for Form.elemental(*args, "
            "**kwargs) — same body, no DeprecationWarning. "
            "Source comment line 87-88: 'for backwards "
            "compatibility / use Form.elemental instead'. New "
            "code should call .elemental() — both return a "
            "COOData with local_shape preserved (vs .assemble() "
            "which returns a scipy CSR). Old tutorials using "
            ".coo_data() still work and emit no warning. "
            "(2) Form.partial(*args, **kwargs) returns a deepcopy "
            "of self with self.form replaced by "
            "functools.partial(self.form, *args, **kwargs). "
            "Preserves form.__name__ across the partial (line 47-"
            "50) so logger.info('Assembling \\'{}\\'.') still "
            "prints the original form name. Lets users curry "
            "fixed parameters into a form without redefining. "
            "(3) Form.block(*args) builds a block-form via a "
            "nested-comprehension lambda (lines 56-61) — used "
            "for split-block matrix assembly (e.g. mixed Stokes "
            "where you want only the velocity-velocity sub-"
            "block). The lambda captures self.form and replaces "
            "missing-block arguments with .zeros() calls. Read "
            "the source if you need this; it's not in the "
            "online tutorials. "
            "(4) FormExtraParams(dict) is the type of the 'w' "
            "argument passed to all form bodies. It's a plain "
            "dict subclass with __getattr__ that maps w.foo to "
            "w['foo']; missing-key access raises AttributeError "
            "with the literal text \"Attribute 'foo' not found "
            "in 'w'.\" — useful for catching typos in custom "
            "extra-param keys. "
            "(5) Form._normalize_asm_kwargs(w, basis) is the "
            "type-coercion layer for assembler keyword args. "
            "Accepted: DiscreteField (passed through if "
            "quadrature matches — else ValueError 'Quadrature "
            "mismatch: ...'), Number (scalar, passed through), "
            "tuple (asm() product index, passed through), 1D "
            "ndarray (auto-converted via "
            "basis.interpolate(w[k])), >1D ndarray (wrapped in "
            "DiscreteField), list (DeprecationWarning + manual "
            "DiscreteField construction). Anything else "
            "ValueError's. Custom user types are NOT supported. "
            "(6) [API] Functional.elemental(basis) is the ONLY "
            "Form subclass that OVERRIDES Form.elemental (line 26 "
            "of functional.py): it returns a plain numpy ndarray "
            "of shape (n_elements,) — the per-element scalar "
            "values — NOT a COOData like the base-class "
            "Form.elemental (which LinearForm / BilinearForm / "
            "TrilinearForm all inherit unchanged). User-visible "
            "asymmetry: code that treats all Form subclasses "
            "uniformly via `.elemental(basis)` (e.g. a helper "
            "that calls .data on the return) breaks on "
            "Functional because ndarray has no .data attribute "
            "in the COOData sense — instead it has numpy's .data "
            "memoryview, which leads to silent shape errors "
            "rather than a clear TypeError. Functional._assemble "
            "encodes its scalar result as a 1-element data array "
            "with empty index arrays — accessed via "
            "`func.assemble(basis)` (the canonical path) which "
            "returns the integrated scalar float. "
            "Functional._assemble has `assert vbasis is None` "
            "(line 54) which is a defensive assert (silently "
            "passes under `python -O`); calls "
            "func.assemble(basis_a, basis_b) with -O resolve as "
            "if basis_b were ignored. "
            "(7) [API] REAL UPSTREAM BUG in skfem 12.0.1: "
            "Form.__call__ line 71 reads "
            "`return self.assemble(self.kernel(*args))` — but "
            "Form has NO `kernel` attribute, only `_kernel` "
            "(defined by each of the 4 subclasses). So calling a "
            "bound Form instance directly (as opposed to "
            ".assemble(basis)) fails with "
            "AttributeError(\"'Functional' object has no "
            "attribute 'kernel'. Did you mean: '_kernel'?\"). "
            "The `__call__` decoration-path (when `self.form is "
            "None`, lines 67-70) still works fine, so the "
            "standard `@fem.BilinearForm` / "
            "`bilinear_instance.assemble(basis)` workflow is "
            "unaffected. The bug fires only for the rarer "
            "pattern of constructing a Form WITH a function and "
            "calling the instance: e.g. "
            "`Functional(form=vol_fn)(basis)` raises "
            "AttributeError. Workaround: always go through "
            "`.assemble(basis)` explicitly. Confirmed live: "
            "skfem.__version__ '12.0.1' raises the AttributeError "
            "on the kernel-vs-_kernel mismatch. "
            "(File walk skfem/assembly/form/functional.py "
            "2026-06-03.) "
            "(File walk skfem/assembly/form/form.py 2026-06-03.)"
        ),
    },
    "coo_data_extras": {
        "description": (
            "COOData (skfem.assembly.form.coo_data.COOData; not "
            "re-exported from skfem or skfem.assembly) is a "
            "plain, non-frozen @dataclass that wraps the "
            "(indices, data, shape, local_shape) tuple returned "
            "by Form._assemble. Its methods (__add__, fromlocal) "
            "return new objects via dataclasses.replace, but the "
            "instance fields can be reassigned. Most users never "
            "touch COOData directly, but two flows DO surface it: "
            "(a) BilinearForm.elemental(...) / .coo_data(...) "
            "skip the CSR conversion and return the COOData, "
            "(b) sum-of-forms "
            "uses COOData.__add__ to merge contributions. Three "
            "edges hide behind the dataclass facade. Source: "
            "skfem/assembly/form/coo_data.py."),
        "Signal": (
            "[API] (1) COOData.solve(b, D=None, tol=1e-10, "
            "maxiters=500) is a HAND-ROLLED CG without "
            "PRECONDITIONER — just rho/(Ap·p) updates over the "
            "matrix-free dot() path. SPD systems converge; "
            "non-SPD systems diverge SILENTLY because the "
            "no-convergence warning path is DEAD CODE: the "
            "post-loop gate at line 240-241 reads `if k == "
            "maxiters: logger.warning('Iterative solver did "
            "not converge.')` but `for k in range(maxiters)` "
            "leaves the final k at maxiters-1, NEVER equal to "
            "maxiters. The warning never fires no matter how "
            "badly the solver fails. Confirmed live: solve() "
            "with tol=1e-15, maxiters=2 returns a non-converged "
            "vector and emits NOTHING to the "
            "skfem.assembly.form.coo_data logger. The off-by-"
            "one means: NOT 'silenced by Python logger config' "
            "(my earlier reading) but actually 'never reached'. "
            "Workaround: convert to scipy CSR via .tocsr() and "
            "use scipy.sparse.linalg.cg / gmres / bicgstab, "
            "which have functioning residual-norm callbacks. "
            "The D=None Dirichlet-partition trick uses "
            "`z[D] = x[D]` in dot() — requires the RHS to "
            "already contain Dirichlet values (lifted); a raw "
            "RHS without lifting silently produces wrong "
            "answers for non-zero Dirichlet conditions. "
            "(2) COOData.__add__(other) NULLS the local_shape "
            "field — after summing two assemble()-returned "
            "COOData objects, calling .tolocal() raises "
            "NotImplementedError('Cannot build local matrices "
            "if local_shape is not specified.'). The forward "
            "convert .tocsr() / .toarray() still works because "
            "they don't need local_shape, but the local-element "
            "view is lost. The sum() builtin's int(0) start "
            "value is handled by __add__(int): returns `self` "
            "unchanged (line 81-82), so sum([f1, f2, f3]) works "
            "as expected. "
            "(3) COOData.topetsc(dofs=None) builds via "
            "petsc.Mat.Type.IS + setLGMap + setISLocalMat then "
            "calls mat.convert('mpiaij') (line 168). For sparse "
            "matrices > ~10^6 nnz the IS→AIJ conversion has its "
            "own memory peak roughly 2× the final AIJ size — "
            "OOM-prone on memory-tight nodes. Users who need a "
            "distributed PETSc matrix directly should look at "
            "PETSc's MatCreateMPIAIJWithArrays instead. The "
            "shape==1 (vector) path goes through "
            "petsc.Scatter().create(...) and uses "
            "InsertMode.ADD scatter — correct for assembled "
            "vector accumulation across ranks. "
            "(File walk skfem/assembly/form/coo_data.py "
            "2026-06-03.)"
        ),
    },
    "autodiff_extras": {
        "description": (
            "skfem.autodiff (skfem/autodiff/__init__.py) is the "
            "JAX-backed Newton-step assembler module. Provides "
            "JaxDiscreteField (PyTree-registered analog of "
            "skfem.element.DiscreteField holding jnp arrays) and "
            "NonlinearForm (subclass of Form whose .assemble "
            "linearizes a user form at a point x via "
            "jax.linearize + jax.jvp, returning the Jacobian K "
            "and the (negated) residual F). "
            "AVAILABILITY (checked 2026-08-03 on the openPASO "
            "venv): jax is NOT installed, so "
            "`import skfem.autodiff` raises "
            "ModuleNotFoundError(\"No module named 'jax'\") at "
            "skfem/autodiff/__init__.py line 5 "
            "(`from jax import jvp, linearize, config`). skfem "
            "does not declare jax as a dependency, so this whole "
            "block — including the det() bug in item (5) below — "
            "is UNVERIFIABLE on this install and is retained "
            "from a file walk only. Do not offer skfem.autodiff "
            "to a user without first checking "
            "`import jax` succeeds. "),
        "Signal": (
            "[Integration]+[API] Four under-publicized edges in "
            "skfem.autodiff (all from a file walk — see the "
            "AVAILABILITY note above, none of them are "
            "re-executable on this install): "
            "(1) IMPORT SIDE EFFECT: `import skfem.autodiff` "
            "unconditionally executes "
            "`jax.config.update(\"jax_enable_x64\", True)` at "
            "MODULE LOAD time (line 10). JAX defaults to "
            "float32; this import silently flips the process-"
            "wide JAX precision to float64, which affects ALL "
            "JAX code in the same Python session — not just "
            "skfem's. Mixing JAX-based ML training (which "
            "expects float32 defaults) with skfem.autodiff in "
            "one script can produce surprising slowdowns and "
            "memory pressure. The setting is also irreversible "
            "within the process. "
            "(2) JaxDiscreteField arithmetic STRIPS gradient "
            "info. __add__/__sub__/__mul__/__truediv__ (lines "
            "35-68) all return `self.value [op] other.value` — "
            "the result is a plain jnp.ndarray, NOT a "
            "JaxDiscreteField. User forms that chain operations "
            "like `(u + g) * w` lose access to .grad / .hess / "
            "etc on the intermediate result. Workaround: pull "
            "out .value at the start of the form body and "
            "operate on raw jnp.ndarrays. "
            "(3) NonlinearForm.assemble returns "
            "(K_csr, F_ndarray) — a SCIPY CSR Jacobian and a "
            "NumPy ndarray RHS (via COOData(*vec).todefault() "
            "on line 121), the same types as BilinearForm / "
            "LinearForm.assemble; NonlinearForm.elemental(...) "
            "returns the two COOData objects instead. The RHS is "
            "NEGATED (`-data1` at line 209) — the Newton step "
            "you solve is K @ du = F (with F already encoding "
            "the minus sign). Users writing `du = sp.linalg."
            "spsolve(K, -F)` will double-negate and converge "
            "AWAY from the root. "
            "(4) The `'hessian' in self.params` branch (lines "
            "172-178) uses a two-step linearize-of-jvp pattern "
            "for second-order forms (e.g. Newton on a system "
            "where the form is itself a Hessian). Triggered by "
            "passing `hessian=...` as a form param, not "
            "documented elsewhere; users who set this on a "
            "first-order form get a confusing JAX trace error "
            "rather than a clean message. "
            "(5) [Validation] REAL UPSTREAM BUG in "
            "skfem/autodiff/helpers.py lines 88-99 — "
            "the 3D branch of det(A) has a STRAY DOUBLE MINUS "
            "in the middle cofactor term: "
            "  - A[0, 1] * (A[1, 0] * A[2, 2] - "
            "               - A[1, 2] * A[2, 0]) "
            "The inner expression `A[1,0]*A[2,2] - - A[1,2]*A"
            "[2,0]` parses in Python as `A[1,0]*A[2,2] + A[1,2]"
            "*A[2,0]` (Python turns `x - -y` into `x + y`). The "
            "CORRECT 3x3 determinant uses `A[1,0]*A[2,2] - "
            "A[1,2]*A[2,0]`. Effect: for any 3x3 matrix where "
            "A[1,2]≠0 AND A[2,0]≠0, the returned det is "
            "WRONG by 2 * A[0,1] * A[1,2] * A[2,0]. The 2D "
            "branch (line 97-98) is correct. User-visible "
            "impact: 3D hyperelasticity Newton steps that "
            "compute J = det(F) for a deformation gradient "
            "with shear in both (1,2) and (2,0) entries — "
            "i.e. a generic deformation — get an incorrect "
            "Jacobian determinant, leading to wrong volumetric "
            "stress and convergence to a wrong root. Worked "
            "concrete example: A = [[1,1,0],[0,1,1],[1,0,1]] "
            "has true det=2 but skfem returns 0. Workaround "
            "until upstream fixes: use jnp.linalg.det directly "
            "instead of skfem.autodiff.helpers.det for 3D "
            "matrices. (File walk skfem/autodiff/helpers.py "
            "2026-06-03.) "
            "(File walk skfem/autodiff/__init__.py 2026-06-03.)"
        ),
    },
    "facet_basis_extras": {
        "description": (
            "FacetBasis pitfalls that fire AFTER construction "
            "succeeds — none surface in the constructor docstring. "
            "Source: skfem/assembly/basis/facet_basis.py."),
        "Signal": (
            "[API] Four under-publicized sharp edges in FacetBasis: "
            "(1) `facets=None` (default) restricts the basis to "
            "BOUNDARY facets only — line 78: `self.find = "
            "np.nonzero(self.mesh.f2t[1] == -1)[0]`. Users wanting "
            "to integrate over EVERY facet (boundary + interior) "
            "must pass facets=np.arange(mesh.facets.shape[1]) "
            "explicitly; default-constructed FacetBasis silently "
            "skips interior facets, and DG/error-estimator forms "
            "that need them lose contributions with no warning. "
            "For interior-facet assembly use InteriorFacetBasis "
            "instead. "
            "(2) `mesh_parameters()` SPECIAL-CASES 1D meshes — "
            "line 138: `np.array([0.])` when mesh.dim() == 1. "
            "That means `h` in default_parameters() is 0.0 for "
            "MeshLine, so any form using `h` for SIP-DG penalty "
            "or h-norm error estimators silently evaluates the "
            "penalty term to zero (or divides by zero, depending "
            "on placement). For 1D problems compute the mesh "
            "parameter manually from mesh.facets coordinates "
            "instead of relying on the default `h`. "
            "(3) `trace(x, projection, target_elem=None)` is "
            "@deprecated('Basis.interpolator + Basis.project') and "
            "only supports 4 mesh types: MeshTri / MeshQuad / "
            "MeshTet / MeshHex. Other mesh types raise "
            "NotImplementedError('Mesh type not supported.') from "
            "the DEFAULT_TARGET lookup. The deprecation is not "
            "silent: every call goes through the @deprecated "
            "decorator (skfem/generic_utils.py), which emits "
            "DeprecationWarning('trace is deprecated in favor of "
            "Basis.interpolator + Basis.project.'); the internal "
            "projection call adds a second DeprecationWarning. "
            "Modern path: "
            "use basis.interpolator(y) to evaluate, then "
            "basis.project(interp) onto the target basis. "
            "(4) Empty-facet construction (e.g. facets=mesh."
            "facets_satisfying(lambda x: False)) emits "
            "`logger.warning('Initializing FacetBasis(...) with no "
            "facets.')` at line 92 — NOT warnings.warn. Default "
            "Python logger config swallows it; users with empty "
            "facet selections see no console output and a "
            "FacetBasis whose .nelems == 0 and whose subsequent "
            ".dx / .basis arrays are empty. asm() on such a basis "
            "returns a zero matrix without warning. To force "
            "stdout: logging.basicConfig(level=logging.WARNING) "
            "before construction. "
            "(File walk skfem/assembly/basis/facet_basis.py "
            "2026-06-03.)"
        ),
    },
    "composite_basis_extras": {
        "description": (
            "CompositeBasis (built via b1 @ b2 / b1 * b2 or "
            "constructed directly) has three pitfalls that fire "
            "AFTER construction succeeds — none of which surface "
            "in the operator-overload docstring. Source: "
            "skfem/assembly/basis/composite_basis.py."),
        "Signal": (
            "[API] (1) CompositeBasis.get_dofs(*args, **kwargs) "
            "is HARD-CODED to `raise NotImplementedError` with no "
            "message at all. Users coming from CellBasis / "
            "FacetBasis expecting `basis.get_dofs('left')` for "
            "Dirichlet selection on a CompositeBasis get a naked "
            "NotImplementedError with no hint. Workaround: call "
            "get_dofs on each sub-basis individually and offset "
            "by sum(sub_basis_i.N for i<k) when assembling the "
            "global Dirichlet vector — or use the operator "
            "splitting (basis.split(x)) to handle each field "
            "separately. "
            "(2) CompositeBasis(*bases) rejects nested "
            "ElementComposite — line 21-23: `if isinstance("
            "basis.elem, ElementComposite): raise "
            "NotImplementedError('ElementComposite not "
            "supported.')`. Common confusion: skfem offers TWO "
            "stacking primitives — ElementComposite (combines "
            "elements inside one Basis, e.g. P2-velocity + P1-"
            "pressure grouped) and CompositeBasis (combines "
            "multiple Basis objects). Nesting the former inside "
            "the latter explodes. Pick one stacking layer. "
            "(3) CompositeBasis.split(x) uses "
            "np.cumsum([basis.N for basis in self.bases])[:-1] "
            "regardless of equal_dofnum. When equal_dofnum=True "
            "(operator b1 @ b2 — DOFs are SHARED, total N = "
            "bases[0].N), the cumsum produces split points at "
            "N0, 2*N0, ... but x has length N0, so split() "
            "returns the first sub-basis's full slice and EMPTY "
            "arrays for the rest. Use equal_dofnum=True only "
            "with care: split() is effectively unusable in that "
            "mode; access sub-fields via basis.bases[i] + "
            "x[:bases[i].N] directly. "
            "(File walk skfem/assembly/basis/composite_basis.py "
            "2026-06-03.)"
        ),
    },
    "dofs_view_extras": {
        "description": (
            "Three quietly-deprecated or surprising behaviors in "
            "skfem.assembly.Dofs / DofsView beyond the "
            "documented 'DofsView is not subscriptable' (which is "
            "already in the per-physics catalogs). Source: "
            "skfem/assembly/dofs.py."),
        "Signal": (
            "[API] (1) DofsView.__or__ (the `dv1 | dv2` operator) "
            "is decorated @deprecated('numpy.hstack') and emits "
            "DeprecationWarning('__or__ is deprecated in favor of "
            "numpy.hstack.'). DofsView.__add__ (the `dv1 + dv2` "
            "operator) just calls __or__ and therefore inherits "
            "the same deprecation — both old idioms for unioning "
            "DofsView objects are on borrowed time. Replacement: "
            "np.hstack([dv1.flatten(), dv2.flatten()]) followed "
            "by np.unique (or np.union1d) for index-set semantics. "
            "(2) DofsView.sort() with no `sorting` kwarg uses "
            "`lambda x: sum(x)` — sorts DOFs by the SUM of their "
            "spatial coordinates. On axis-aligned grids the sums "
            "tie for symmetric points (e.g. (1,0) and (0,1) both "
            "sum to 1) and argsort gives arbitrary tie-breaking. "
            "For deterministic x-then-y ordering pass an explicit "
            "key, e.g. sort(sorting=lambda x: x[0] * 1e6 + x[1]). "
            "(3) Dofs.decompose(comm, cache=..., nparts=N) is a "
            "decorator that, when nparts is non-None, calls METIS, "
            "saves the decomposition to disk, and then raises "
            "SystemExit(\"'nparts' has been set: decomposition was "
            "saved to files and process terminated\") — the entire "
            "Python process exits, NOT just the decorated "
            "function. Documented but easy to miss; users wanting "
            "to inspect the decomposition in the same script need "
            "to omit nparts and let MPI handle the partition "
            "live. Plus: _decompose imports `pymetis` lazily, so "
            "missing pymetis raises ImportError at decompose-time "
            "rather than at import-time — install gap not visible "
            "until first decomposition call. "
            "(File walk skfem/assembly/dofs.py 2026-06-03.)"
        ),
    },
    "assembly_module_asm_shorthand": {
        "description": (
            "skfem.assembly.asm(form, *bases) is a convenience "
            "shorthand around Form.assemble that auto-wraps bare "
            "callables into the right form subclass by inspecting "
            "form.__code__.co_argcount. Source: "
            "skfem/assembly/__init__.py."),
        "auto_wrap_table": {
            1: "Functional   — (w,)",
            2: "LinearForm   — (v, w)",
            3: "BilinearForm — (u, v, w)",
            4: "TrilinearForm — (u, v, w, p)",
        },
        "backwards_compat_aliases": {
            "InteriorBasis": "alias for CellBasis (no DeprecationWarning)",
            "ExteriorFacetBasis": ("alias for BoundaryFacetBasis "
                                   "(no DeprecationWarning)"),
            "ElementVectorH1": (
                "alias for ElementVector (skfem/element/"
                "__init__.py:62) — no DeprecationWarning."),
            "ElementTriDG": (
                "alias for ElementDG (a generic DG wrapper); "
                "ElementQuadDG / ElementHexDG / ElementTetDG "
                "also all alias to the same ElementDG (lines "
                "63-66 of skfem/element/__init__.py). The "
                "per-shape names are kept for tutorial "
                "compatibility — the old shape-specific DG "
                "wrapper classes were merged into the generic "
                "ElementDG, but separate discontinuous "
                "first-order classes (ElementTriP1DG, "
                "ElementQuad1DG, ElementHex1DG, ElementLineP1DG) "
                "still exist."),
            "ElementTriMini": (
                "alias for ElementTriP1B (line 67). The Mini "
                "element is just P1+bubble; the rename to P1B "
                "reflects this."),
            "ElementTriCCR": (
                "alias for ElementTriP2B (line 68). Conforming "
                "Crouzeix-Raviart is P2+bubble; the rename "
                "reflects that."),
            "ElementTriRT0 / ElementTetRT0 / ElementQuadRT0": (
                "ALL alias to ElementTriRT1 / ElementTetRT1 / "
                "ElementQuadRT1 respectively (lines 69-71). "
                "WARNING: in classical FEM nomenclature RT0 "
                "(lowest order, k=0) is conventionally DISTINCT "
                "from RT1 (k=1). skfem's naming convention is "
                "'k-th order = polynomial degree ≤ k', so what "
                "FEM literature calls RT0 is skfem's RT1. The "
                "alias means users with old skfem code using "
                "the 'RT0' name still get the SAME class — but "
                "users following standard FEM textbooks who "
                "instantiate ElementTriRT0 expecting the "
                "lowest-order element get exactly that: "
                "ElementTriRT0 is ElementTriRT1, the lowest-order "
                "RT element. skfem also ships ElementTriRT2, the "
                "quadratic RT element (textbook RT1)."),
            "ElementTetN0": (
                "alias for ElementTetN1 (line 72). Same "
                "Nédélec naming convention: skfem's 'N1' is "
                "what textbooks call 'N0' (lowest-order edge "
                "element)."),
        },
        "boundary_element_map": (
            "skfem.element.BOUNDARY_ELEMENT_MAP maps a mesh's "
            "geometry element (Mesh.elem) to its facet element — "
            "covers 13 types: "
            "ElementTriP0/P1/P2 → ElementLineP0/P1/P2, "
            "ElementQuad0/1/2 → ElementLineP0/P1/P2, "
            "ElementTetP0/P1/P2 → ElementTriP0/P1/P2, "
            "ElementHex0/1/2 → ElementQuad0/1/2, "
            "ElementHex1DG → ElementQuad1. It is read through "
            "Mesh.bndelem (which supplies the facet element of "
            "MappingIsoparametric on non-affine meshes and is "
            "used by Mesh.f2e) and by the deprecated "
            "FacetBasis.trace(). FacetBasis never looks up the "
            "basis element in it and takes no boundary-element "
            "argument: its signature is FacetBasis(mesh, elem, "
            "mapping=None, intorder=None, quadrature=None, "
            "facets=None, dofs=None, side=0, "
            "disable_doflocs=False), and passing "
            "facet_quadrature=... or **{'elem': ...} raises "
            "TypeError. A basis element's absence from the map "
            "does not block construction (no KeyError) — "
            "measured on skfem 12.0.1, "
            "MeshTri().refined(2): "
            "FacetBasis(m, ElementTriP3()) constructs with "
            "nelems=16 and Nbfun=10, and FacetBasis also "
            "constructs cleanly for ElementTriP4, "
            "ElementTriBDM1, ElementTriRT1 and (on MeshTet) "
            "ElementTetN1. "
            "Source: skfem/element/__init__.py lines 45-59."
        ),
        "Signal": (
            "[API] skfem.asm()'s auto-wrap based on "
            "form.__code__.co_argcount has TWO unguarded edge "
            "cases at the wrapper-lookup site "
            "[Functional, LinearForm, BilinearForm, "
            "TrilinearForm][nargs - 1]: "
            "(1) Bare-callable with nargs >= 5 raises "
            "IndexError('list index out of range') instead of a "
            "helpful error — common when a user adds an extra "
            "positional arg by mistake or forgets to decorate "
            "with @BilinearForm. "
            "(2) Bare-callable with nargs == 0 SILENTLY wraps as "
            "TrilinearForm via Python negative indexing "
            "(nargs - 1 == -1 hits the last list element). The "
            "wrap succeeds; the error comes at call time as "
            "TypeError('<lambda>() takes 0 positional arguments "
            "but 4 were given') — opaque because the user thinks "
            "they passed a Functional. "
            "Also: skfem.InteriorBasis and skfem.ExteriorFacetBasis "
            "are still active aliases for CellBasis and "
            "BoundaryFacetBasis with NO DeprecationWarning. "
            "Tutorials and StackOverflow answers from <=2022 use "
            "the old names interchangeably; new users see them "
            "work without realizing they're shadows of the "
            "modern names. Plus: skfem.asm uses an internal "
            "_sum(blocks) reducer that asserts `not isinstance("
            "out, int)` to catch the silent-zero case when the "
            "assembly iterator is empty (an empty basis list, "
            "e.g. asm(form, []) or asm(form, [b], [])). "
            "AssertionError there means 'you "
            "passed nothing assemblable', not a bug in skfem. "
            "Calling asm(form) with no basis at all does not "
            "reach that assert; it raises TypeError(\"<Form>."
            "_assemble() missing 1 required positional argument: "
            "'ubasis'\"). "
            "(File walk skfem/assembly/__init__.py 2026-06-03.)"
        ),
    },
    "linear_system_utils": {
        "description": (
            "skfem.utils linear-system entry points (enforce / "
            "condense / penalize / solve / mpc / "
            "solver_iter_krylov / solver_iter_pcg). Source: "
            "skfem/utils.py."
        ),
        "boundary_dof_input_modes": (
            "enforce/condense/penalize all use _init_bc which "
            "requires EXACTLY ONE of I (free DOFs) or D "
            "(Dirichlet DOFs). Passing BOTH raises "
            "Exception('Give only I or only D!'); passing NEITHER "
            "raises Exception('Either I or D must be given!'). "
            "The unspecified set is derived as the setdiff against "
            "np.arange(A.shape[0])."),
        "solver_iter_pcg_is_an_alias": (
            "solver_iter_pcg(**kwargs) is a one-line forwarder to "
            "solver_iter_krylov(**kwargs) with NO specialization "
            "beyond the docstring claim ('Conjugate gradient "
            "solver, specialized from solver_iter_krylov') — the "
            "default krylov of solver_iter_krylov is already "
            "scipy.sparse.linalg.cg, so the two are functionally "
            "identical."),
        "auto_injected_diagonal_preconditioner": (
            "solver_iter_krylov(...) auto-injects M=build_pc_diag(A) "
            "iff the literal string 'M' is NOT in kwargs at call "
            "time. Calling solver_iter_krylov(M=None) suppresses "
            "the injection and passes M=None straight to scipy "
            "(== no preconditioner). Convergence diagnostics that "
            "miss this misattribute speedups to the iterative "
            "method when the diagonal PC is actually doing the "
            "work."),
        "Signal": (
            "[API] Three sharp edges in skfem.utils that bite "
            "users on first contact: "
            "(1) Passing both I=... and D=... to enforce / "
            "condense / penalize raises Exception('Give only I "
            "or only D!') — the two are mutually exclusive "
            "complements (setdiff against arange(A.shape[0])), "
            "not intersectable filter sets. "
            "(2) solver_iter_pcg is a synonym for "
            "solver_iter_krylov, NOT a specialized PCG — both "
            "default to scipy.sparse.linalg.cg with auto-injected "
            "diagonal preconditioner. "
            "(3) The diagonal-PC auto-injection in "
            "solver_iter_krylov is gated on `'M' not in kwargs` "
            "— solver_iter_krylov(M=None) ≠ solver_iter_krylov(); "
            "the former gets no preconditioner (M=None forwarded "
            "to scipy), the latter gets a Jacobi PC silently. "
            "(File walk skfem/utils.py 2026-06-03.)"
        ),
    },
    "mortar_workflow": {
        "description": (
            "Nonmatching-mesh (mortar / supermesh) workflow via "
            "skfem.supermeshing — for domain-decomposition or "
            "fluid-structure interface coupling. Source: "
            "skfem/supermeshing.py."
        ),
        "functions": {
            "skfem.supermeshing.intersect(m1, m2)": (
                "Returns (supermesh, ix1, ix2) — supermesh + element-index "
                "arrays into each input mesh. ONLY 1D-1D or 2D-2D; 3D-3D "
                "or 1D-vs-2D raises NotImplementedError('The given mesh "
                "types not supported.')."),
            "skfem.supermeshing.elementwise_quadrature(mesh, supermesh, tind, intorder)": (
                "Build per-element quadrature rules from a supermesh. "
                "REQUIRES supermesh= kwarg explicitly (raises Exception "
                "if missing). intorder defaults to 4."),
        },
        "Signal": (
            "[API] skfem.supermeshing has three hard failure modes "
            "users hit on first use: "
            "(1) Calling intersect(m1, m2) with 3D meshes raises "
            "NotImplementedError('The given mesh types not supported.') "
            "— mortar in 3D is unsupported; "
            "(2) Calling intersect on 2D meshes without shapely>=2 "
            "installed raises Exception(\"2D supermeshing requires the "
            "package 'shapely>=2'.\") — silent until first call; "
            "(3) Calling elementwise_quadrature(mesh) without the "
            "supermesh= kwarg raises Exception(\"elementwise_quadrature: "
            "User must provide 'supermesh' keyword argument which has been "
            "created using skfem.supermeshing.intersect.\") — Python "
            "won't catch the missing kwarg at signature level because "
            "supermesh has default None and the check is at runtime. "
            "(File walk skfem/supermeshing.py 2026-06-03.)"
        ),
    },
}
