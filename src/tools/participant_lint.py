"""Name, at write time, the API calls a participant script is about to die on.

WHY THIS EXISTS, MEASURED. Round 49 (the first honest round on problems other than C1-C3) was lost to
wall clock, not to knowledge: every one of the nine cells hit the 45-minute limit at 35 to 73 seconds
per action, bought 37 to 75 actions, and exactly one reached couple() at all. The action sink is the
write-run-error-rewrite loop on the participant script -- and when all 58 saved step-trial fills were
re-executed and graded the way the tasks grade, 46 of them never wrote an export and EVERY ONE of those
died on an invented API call, not on the physics.

Each call below was measured on THIS install, and each finding names the error the run will print and
the call that works. The same facts are served through the coupling door; this is the moment they are
most useful, which is before the run that would have taught them.

WHAT THIS IS NOT. It does not judge the mesh, the weak form, the material, the source or the solve --
those are the agent's and openPASO has no business dictating them. It reads the file and writes nothing.
"""
from __future__ import annotations

import functools
import re
from pathlib import Path

# ── facts shared by the write-time traps and the run-time table ───────────
# Each measured on this install (scikit-fem 12.0.1, dolfinx 0.10.0): the wrong call raised the
# quoted error and the named call ran. The scripts of one coupled round died on every one of them.
_SKFEM_ELEMENT_FIX = (
    "an element is an INSTANCE: Basis(mesh, ElementTriP1()), with the parentheses, and "
    "basis.with_element(ElementTriP0()) the same. The class itself, directly or through a name bound "
    "to it (element = ElementTriP1), stops inside skfem with an error that names no element")
_SKFEM_JOIN_FIX = (
    "there is no extend. Refine with mesh.refined(n) and use the RETURN value (meshes are immutable); "
    "join two meshes of one type with m1 + m2, which merges the nodes they share; cut a rectangle out "
    "of a box with mesh.remove_elements(<cell indices>) on the tensor mesh. The facets of an interface "
    "with two straight legs come from one mesh.facets_satisfying(<that leg's test>, "
    "boundaries_only=True) per leg, concatenated: without boundaries_only a leg's line also takes the "
    "interior facets beyond it")
_SKFEM_W_FIX = (
    "inside a form, w carries w.x (the quadrature points' coordinates), w.h, w.idx, w.n on a facet "
    "basis, and the fields passed to asm as keywords -- nothing else. A coefficient that varies by "
    "region is evaluated from w.x in the form, np.where(<test on w.x[0], w.x[1]>, k1, k2), or passed "
    "as a piecewise-constant field: k0 = basis.with_element(ElementTriP0()).interpolate(k_per_cell), "
    "asm(form, basis, k=k0), and w.k in the form")
_SKFEM_SIZE_FIX = (
    "Basis.interpolate(x) takes basis.N values in x, and so does a field passed to asm as a keyword (asm "
    "interpolates it on the basis being assembled). Two measured ways to miss it: a dof vector of an "
    "ElementVector basis interpolated on a FacetBasis built on the scalar element, whose N is half as large -- "
    "build the FacetBasis on the same element, FacetBasis(mesh, basis.elem, facets=...); and a per-cell array "
    "passed as a field, which is a piecewise-constant field: "
    "k0 = basis.with_element(ElementTriP0()).interpolate(k_per_cell), then asm(form, basis, k=k0)")
_DOLFINX_VECTOR_FIX = (
    "fem.assemble_vector returns a dolfinx.la.Vector, which has no PETSc methods: its values are "
    "b.array, and b.scatter_reverse(la.InsertMode.add) adds the ghost contributions. The PETSc vector, "
    "with ghostUpdate, is dolfinx.fem.petsc.assemble_vector after `import dolfinx.fem.petsc`")
_DOLFINX_CSR_FIX = (
    "fem.assemble_matrix returns a dolfinx.la.MatrixCSR: A.scatter_reverse() adds the ghost "
    "contributions and A.to_scipy() (or A.to_dense()) hands it to scipy; it has no .mat, .assemble() "
    "or .shape. The PETSc matrix, with .assemble(), is dolfinx.fem.petsc.assemble_matrix after "
    "`import dolfinx.fem.petsc`")
_DOLFINX_BCS_FIX = (
    "assemble_vector takes no bcs. fem.assemble_matrix(a, bcs=[bc]) takes them for the matrix; the "
    "vector gets the boundary values through fem.apply_lifting(b.array, [a], bcs=[[bc]]), "
    "b.scatter_reverse(la.InsertMode.add) and bc.set(b.array)")
_DOLFINX_INDEXMAP_FIX = (
    "an IndexMap counts with size_local (this process) and size_global (all processes), and a space "
    "has no .dim: its dof count is V.dofmap.index_map.size_local * V.dofmap.index_map_bs")
_DOLFINX_TAGS_FIX = (
    "tags are made by the lowercase FUNCTION dolfinx.mesh.meshtags(mesh, dim, indices, values) with "
    "int32 arrays; the MeshTags class only wraps what that function returns, and dolfinx.fem has no "
    "MeshTags or MeshTag")
_DOLFINX_MEASURE_FIX = (
    "the measure is UFL's: ds = ufl.Measure('ds', domain=mesh, subdomain_data=tags), then "
    "ds(<tag value>) integrates over the facets with that value; dolfinx.fem has no Measure, and "
    "ufl.ds(tags) reads the tags as an id")
_DOLFINX_RANK1_FIX = (
    "dolfinx's create_matrix, which LinearProblem calls first, raises this range-check IndexError, "
    "naming no form, when the form in the bilinear slot has rank 1: its unknown is a fem.Function "
    "where u = ufl.TrialFunction(V) belongs (fem.form(a).rank is then 1; a bilinear form's is 2), or "
    "the two forms are swapped, LinearProblem(L, a). The Function is what .solve() returns")
# Measured on this install (dolfinx 0.10), each in the fenics interpreter: the wrong call raised the
# quoted error and the named call ran. The scripts of one coupled round died on each of them.
_DOLFINX_BCS_LIST_FIX = (
    "bcs are always a LIST, and apply_lifting takes one list of them per form: "
    "dolfinx.fem.petsc.assemble_matrix(a_form, bcs=bcs), apply_lifting(b, [a_form], bcs=[bcs]), "
    "set_bc(b, bcs). Measured on this install: apply_lifting(b, a_form, bcs) raises 'Form' object is "
    "not iterable; a single DirichletBC where the list belongs raises 'DirichletBC' object is not "
    "iterable (assemble_matrix, apply_lifting) or object of type 'DirichletBC' has no len() (set_bc); "
    "and bcs=[[bc] for bc in bcs], a list per condition, raises Mismatch in size between a and bcs "
    "once there are two conditions")
# THE VALUE HAS THE SHAPE OF THE SPACE IT HOLDS, and a block of a mixed space takes a Function WITH the
# sub-space. Measured on a fluid-structure round: this text said a boundary value is a plain scalar
# and that a Function with a space raises -- true on a scalar space only -- and it answered 16 errors of
# fluid sides whose velocity block needed exactly that Function with W.sub(0).
_DOLFINX_BC_VALUE_FIX = (
    "dirichletbc's value has the shape of the space it holds. A scalar space: (default_scalar_type(0.0), "
    "dofs, V), or (g, dofs) for a fem.Function g on V. A vector space: (np.zeros(gdim, "
    "dtype=default_scalar_type), dofs, V), or (g, dofs) -- a plain number there raises Rank mismatch "
    "between Constant and function space, and so does np.array([0.0]) on a scalar space. One block of a "
    "MIXED space takes a Function and the sub-space: V0, _ = W.sub(0).collapse(); g = "
    "fem.Function(V0); dofs = fem.locate_dofs_topological((W.sub(0), V0), fdim, facets), a pair of "
    "arrays; fem.dirichletbc(g, dofs, W.sub(0)) -- a constant there raises 'Constant size is not equal "
    "to the block size', even a zero. The dofs are what locate_dofs_topological or "
    "locate_dofs_geometrical returns. Measured on this install: a Python list raises Boundary "
    "condition value must have a dtype attribute; and a number or a Constant without V, a Function "
    "with V on a plain space, a Function without W.sub(0) or with dofs located on V0 alone on a mixed "
    "one, or int64 dofs raise incompatible function arguments")
# Each measured on this install (dolfinx 0.10), the wrong call in the fenics interpreter printing the
# quoted error and the named call running; the fluid sides of one fluid-structure round died on each.
_DOLFINX_TABULATE_MIXED_FIX = (
    "a mixed space has no coordinates of its own: tabulate the collapsed sub-space, V0, _ = "
    "W.sub(0).collapse(); V0.tabulate_dof_coordinates() (one row per node of V0), or a P1 space of "
    "its own on the same mesh")
_DOLFINX_ZERO_FORM_FIX = (
    "UFL folds 0 * v, and 0.0 times any expression, to its Zero, which carries no domain, so "
    "`* ufl.dx` raises at once; a plain number alone, 1.0 * ufl.dx, has no mesh either. Make the "
    "number a constant on the mesh: ufl.inner(fem.Constant(mesh, np.zeros(gdim)), v) * ufl.dx for a "
    "vector test function v, fem.Constant(mesh, default_scalar_type(0.0)) * v * ufl.dx for a scalar "
    "one; both compile")
_DOLFINX_NONLINEAR_FIX = (
    "dolfinx.fem.petsc has no NewtonSolver. Its nonlinear solve is NonlinearProblem(F, w, bcs=bcs, "
    "petsc_options_prefix='<name>', petsc_options={'snes_type': 'newtonls', ...}); problem.solve() "
    "fills w, and it RETURNS when Newton did not converge too (measured: reason -5 at the iteration "
    "cap) -- read problem.solver.getConvergedReason() (> 0 converged) and "
    "problem.solver.getIterationNumber(), or set 'snes_error_if_not_converged': True")
_DOLFINX_MOVE_FIX = (
    "dolfinx.mesh has no move. The node coordinates are msh.geometry.x, an (n, 3) array the forms "
    "assembled afterwards read: for a displacement d on the mesh's own P1 vector space, "
    "fem.functionspace(msh, ('Lagrange', 1, (gdim,))), msh.geometry.x[:, :gdim] += "
    "d.x.array.reshape(-1, gdim). Measured in serial on create_rectangle triangle meshes, where that "
    "space's dof order equalled the node order; compare V1.tabulate_dof_coordinates()[:, :gdim] with "
    "msh.geometry.x[:, :gdim] before relying on it")
# MEASURED ON THIS INSTALL: locate_dofs_topological(W.sub(0).collapse(), fdim, facets) stops in
# dolfinx/fem/bcs.py, `_V = [space._cpp_object for space in V]`, on the dof array of the pair. The
# '_cpp_object' answer for an uncompiled form named the wrong cause twice on a fluid-structure round.
_DOLFINX_COLLAPSE_PAIR_FIX = (
    "W.sub(i).collapse() returns a PAIR, (the collapsed space, its dofs in W), and "
    "locate_dofs_topological / locate_dofs_geometrical read every entry of a tuple as a space, so "
    "the dof array has no _cpp_object. Unpack it: V0, _ = W.sub(0).collapse(); for a block of the mixed "
    "space pass the two spaces, fem.locate_dofs_topological((W.sub(0), V0), fdim, facets), and its "
    "condition is fem.dirichletbc(g, dofs, W.sub(0)) with g a fem.Function on V0")
_DOLFINX_SCALAR_FIX = (
    "assemble_scalar is dolfinx.fem.assemble_scalar(fem.form(...)); dolfinx.fem.petsc has none")
# MEASURED ON THIS INSTALL (dolfinx 0.10, PETSc 3.24), a Taylor-Hood channel, each case in its own
# process: PETSc's own LU printed "Zero pivot in LU factorization" with ksp_error_if_not_converged and
# otherwise let NonlinearProblem.solve() return after 0 iterations (reason -3, w unchanged); a residual
# written with w.split() printed "Matrix is missing diagonal entry 0". With u, p = ufl.split(w) and
# mumps the same solve converged in 3 iterations. A fluid side of a fluid-structure round gave up on the
# missing diagonal after reading it as a pressure null space; none of our answers named either cause.
_DOLFINX_SADDLE_LU_FIX = (
    "two causes print this on a velocity-pressure solve, each measured on this install. PETSc's own LU "
    "('pc_type': 'lu' with no solver package) meets the zero diagonal of the pressure block: Zero pivot in "
    "LU factorization; add 'pc_factor_mat_solver_type': 'mumps' (with mumps the measured solve converged "
    "in 3 Newton iterations). A residual written with w.split() or w.sub(i) has an empty Jacobian on those "
    "rows: Matrix is missing diagonal entry 0; build it from u, p = ufl.split(w) and keep w.split() for "
    "reading the solved parts. Without 'snes_error_if_not_converged' both stop nothing: solve() returns "
    "after 0 iterations with reason -3")
_DOLFINX_UFL_CONSTANT_FIX = (
    "ufl.Constant(mesh, ...) takes a domain and a SHAPE and holds no number: an array given there is "
    "read as the shape, and the form fails on it. A constant in a form is fem.Constant(mesh, value) -- "
    "np.zeros(gdim) for a vector, default_scalar_type(0.0) for a scalar")
_DOLFINX_FORM_COMPILED_FIX = (
    "dolfinx.fem.petsc.assemble_matrix, assemble_vector and apply_lifting take the compiled form "
    "fem.form(a); LinearProblem takes the UFL forms and compiles them itself. Measured on this "
    "install: assemble_matrix or apply_lifting given the bare UFL form raise 'Form' object has no "
    "attribute '_cpp_object' (assemble_vector: no attribute 'function_spaces'), and so do the C++ "
    "form fem.form(a)._cpp_object and a PETSc Mat given where a form belongs. "
    "a_form = fem.form(a); A = dolfinx.fem.petsc.assemble_matrix(a_form, bcs=bcs); A.assemble()")
_DOLFINX_CONNECTIVITY_FIX = (
    "mesh.exterior_facet_indices(mesh.topology) needs the facet-to-cell connectivity: call "
    "mesh.topology.create_connectivity(tdim - 1, tdim) first (locate_entities_boundary creates it by "
    "itself, locate_entities does not)")
# Measured on this install (NGSolve 6.2.2604): after `u, v = fes.TnT()` and a loop over mesh.vertices
# whose variable is also named v, `gfun * v` prints "Invoked with: <...GridFunction...>, V18" (V18 is a
# mesh vertex), `v * dx` "unsupported operand type(s) for *: 'ngsolve.comp.MeshNode' and ...", and
# grad(v) "'ngsolve.comp.MeshNode' object has no attribute 'Operator'". Two of five workers of one
# round met it at a served line and no reply said why.
_NGSOLVE_REBOUND_FIX = (
    "a mesh vertex (a MeshNode, printed V<number>) sits where a form needs the test or trial function: "
    "the name was rebound after u, v = fes.TnT() -- a Python loop over mesh.vertices whose variable is "
    "named v (or u), or v = mesh.vertices[i], leaves the name holding a vertex once it ends. Give the "
    "vertex its own name (for vert in mesh.vertices) and keep u, v for the TnT pair")
_NGSOLVE_DIMS_FIX = (
    "CoefficientFunction reads a Python LIST as ONE number, its first entry: CoefficientFunction([[kxx, "
    "kxy], [kyx, kyy]]) has no dims and the value kxx, and the same list with dims=(2, 2) raises this. A "
    "2x2 matrix is a TUPLE of its four entries, row by row, with dims: CoefficientFunction((kxx, kxy, "
    "kyx, kyy), dims=(2, 2)) -- the form the served NGSolve contract builds from config.json's k -- and "
    "K * grad(u) * grad(v) * dx then assembles (K grad u) . grad v")

# A NAME BOUND ONCE AND USED LATER, and nothing in between that binds it again: an assignment to it,
# or a def/for/with line that names it. Used by the traps that follow a value from the call that made
# it to the attribute read on it. Group `v` is the name.
_NOT_REBOUND = (r"(?!^[ \t]*(?:\w+[ \t]*,[ \t]*)*(?P=v)[ \t]*(?:,[ \t]*\w+[ \t]*)*=(?!=)"
                r"|^[ \t]*(?:def|for|with)[ \t][^\n]*\b(?P=v)\b)")
# `import dolfinx.fem.petsc as fem` makes fem.assemble_* the PETSc assemblers, whose results DO have
# ghostUpdate and .assemble(): a trap on fem.assemble_* reads the whole file for that alias first, and
# names only its group `hit`.
_FEM_IS_NOT_PETSC = r"\A(?![\s\S]*?\bpetsc[ \t]+as[ \t]+fem\b)[\s\S]*?"
_SKFEM_ELEMENT_CLASS = r"(?:skfem\.)?Element(?:Tri|Quad|Tet|Hex|Line|Wedge)\w*"

# (backend, pattern, what the run prints, the call that works)
_TRAPS: tuple[tuple[str, str, str, str], ...] = (
    # ── CAUGHT AT WRITE TIME BECAUSE THE RUN IS THE EXPENSIVE PART ───────────
    # Each of these is already answered by the run-side table, and each was
    # measured in its real environment. Reading them out of the SOURCE saves
    # the run that would otherwise teach them -- and on a side that has not
    # started yet, that run is most of the cell.
    ("fenics", r"ufl\.MixedElement\s*\(|ufl\.VectorElement\s*\(",
     "AttributeError: module 'ufl' has no attribute 'MixedElement'/'VectorElement'",
     "on dolfinx 0.10 the elements come from basix: "
     "basix.ufl.element('Lagrange', mesh.basix_cell(), deg, shape=(gdim,)) and "
     "basix.ufl.mixed_element([Ve, Qe]), then fem.functionspace(mesh, ...)"),
    ("fenics", r"\bfem\.MeshTags\s*\(|\bdolfinx\.MeshTags\s*\(",
     "AttributeError: module 'dolfinx.fem' has no attribute 'MeshTags'",
     "facet tags come from the lowercase FUNCTION in the mesh module: "
     "dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, indices, values)"),
    ("fenics", r"dolfinx\.nls\.NewtonSolver|from\s+dolfinx\.nls\s+import\s+NewtonSolver",
     "ImportError: cannot import name 'NewtonSolver' from 'dolfinx.nls'",
     "the Newton solver is in the petsc submodule: `import dolfinx.nls.petsc` "
     "then dolfinx.nls.petsc.NewtonSolver"),
    ("fenics", r"\b(?:conditional|lt|gt|le|ge)\s*\([^\n]*?\)\s*[|&]",
     "TypeError: unsupported operand type(s) for |: 'Conditional' and 'Conditional'",
     "UFL conditions combine with Or(a, b) and And(a, b) from ufl, not with "
     "Python's | and &. The error names the operator and not the fix, so it "
     "reads as a backend fault -- one recorded run called it a segfault and "
     "abandoned the backend with Or already on its own import line"),
    ("dune", r"\b(?:conditional|lt|gt|le|ge)\s*\([^\n]*?\)\s*[|&]",
     "TypeError: unsupported operand type(s) for |: 'Conditional' and 'Conditional'",
     "UFL conditions combine with Or(a, b) and And(a, b) from ufl, not with "
     "Python's | and &. The error names the operator and not the fix, so it "
     "reads as a backend fault -- one recorded run called it a segfault and "
     "abandoned the backend with Or already on its own import line"),
    ("fenics", r"\bufl\.Eq\s*\(",
     "ImportError: cannot import name 'Eq' from 'ufl'",
     "a variational equation is written with the operator, `a == L`; ufl.eq is "
     "a BOOLEAN comparison for conditionals and is not it"),
    ("skfem", r"\.ndof\b",
     "AttributeError: 'CellBasis' object has no attribute 'ndof'",
     "a basis counts its degrees of freedom with basis.N (basis.nelems is the "
     "element count, a different quantity)"),
    ("skfem", r"get_dofs\s*\([^)]*\bcomponent\s*=",
     "TypeError: AbstractBasis.get_dofs() got an unexpected keyword argument 'component'",
     "a vector basis selects a component BY NAME: d = basis.get_dofs(...), "
     "then d.nodal['u^1'] and d.nodal['u^2'] (one-based; there is no u^0)"),
    # THE RUN CHECK ANSWERED "refine" TO A SCRIPT JOINING TWO MESHES: the extend call is made to
    # join or cut as often as to refine, so the fact names all three.
    ("skfem", r"\bmesh\.extend\s*\(|\bm\.extend\s*\(",
     "AttributeError: 'MeshTri1' object has no attribute 'extend'", _SKFEM_JOIN_FIX),
    ("skfem", r"from\s+skfem\.models\.elasticity\s+import[^\n]*plane_strain",
     "ImportError: cannot import name 'plane_strain' from 'skfem.models.elasticity'",
     "there is no plane_strain because lame_parameters IS the plane-strain "
     "pair: linear_elasticity(*lame_parameters(E, nu))"),
    ("kratos", r"\bKM\.ModelPart\s*\(|KratosMultiphysics\.ModelPart\s*\(",
     "TypeError: Kratos.ModelPart: No constructor defined!",
     "a ModelPart is created BY a Model: model = KM.Model(); "
     "mp = model.CreateModelPart('<name>')"),
    ("ngsolve", r"\b(?:CoefficientFunction|CF)\s*\(\s*lambda\b",
     "ValueError: Cannot make CoefficientFunction from <function <lambda>>",
     "a CoefficientFunction is built from NGSolve's own symbolic coordinates, "
     "not from a Python callable: `from ngsolve import x, y, z` and write the "
     "expression in them -- CF(x*x + y*y). A string and a sympy object fail "
     "the same way. Measured on this install"),
    # a solve through a method that does not exist was read as "never solves" (measured)
    ("ngsolve", r"\.InvMult\s*\(",
     "AttributeError: 'ngsolve.la.SparseMatrixd' object has no attribute 'InvMult'",
     "NGSolve has no InvMult: an inverse is a matrix of its own, inv = a.mat.Inverse(<free dofs>), "
     "applied with `*` to a vector (or inv.Mult(x, y), which writes inv x INTO y). Measured "
     "on this install"),
    ("ngsolve", r"\.SetEssentialBC\s*\(",
     "AttributeError: 'ngsolve.comp.H1' object has no attribute 'SetEssentialBC'",
     "the Dirichlet boundary is an ARGUMENT of the space -- "
     "H1(mesh, order=..., dirichlet='<names|joined>') -- and fes.FreeDofs() is "
     "what the solve inverts on"),
    # ── scikit-fem ────────────────────────────────────────────────────────
    ("skfem", r"\.dofs\s*\(",
     "TypeError: 'Dofs' object is not callable",
     "basis.dofs is an ATTRIBUTE; select with basis.get_dofs(facets=<facet indices>) or "
     "basis.get_dofs(lambda x: np.isclose(x[0], IFACE_X)), then .flatten()"),
    ("skfem", r"\.assemble\s*\(\s*basis\s*=",
     "TypeError: BilinearForm._assemble() missing 1 required positional argument: 'ubasis'",
     "a form takes its basis POSITIONALLY: laplace.assemble(basis) or asm(laplace, basis)"),
    ("skfem", r"\bw\.y\b",
     "AttributeError: Attribute 'y' not found in 'w'",
     "inside a form the coordinates are w.x[0] and w.x[1]; there is no w.y"),
    ("skfem", r"\binit_rect\s*\(",
     "AttributeError: type object 'MeshTri1' has no attribute 'init_rect'",
     "a rectangle is MeshTri.init_tensor(np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1))"),
    ("skfem", r"\bdoforder\s*=",
     "TypeError: CellBasis.__init__() got an unexpected keyword argument 'doforder'",
     "Basis(mesh, element) takes no doforder keyword"),
    # Measured on a coupled elastic round (scikit-fem 12.0.1): three of five cells wrote
    # sym_grad(u, w) in their form, and one MeshTri.rect and basis.find_nodes.
    ("skfem", r"\bsym_grad\s*\(\s*[\w.]+\s*,",
     "TypeError: sym_grad() takes 1 positional argument but 2 were given",
     "sym_grad takes the field alone, sym_grad(u); the form's w is not passed to it"),
    ("skfem", r"\bMeshTri\w*\.rect\s*\(",
     "AttributeError: type object 'MeshTri1' has no attribute 'rect'",
     "a rectangle is MeshTri.init_tensor(np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1))"),
    ("skfem", r"\.find_nodes\s*\(",
     "AttributeError: 'CellBasis' object has no attribute 'find_nodes'",
     "a basis selects dofs with basis.get_dofs(lambda x: np.isclose(x[0], X)) and a mesh its nodes "
     "with mesh.nodes_satisfying(lambda x: np.isclose(x[0], X))"),
    ("skfem", r"\bfind_dofs\s*\(",
     "AttributeError: 'FacetBasis' object has no attribute 'find_dofs'",
     "every basis has get_dofs, not find_dofs"),
    ("skfem", r"\bmesh\.f\b(?!acets|2)",
     "AttributeError: 'MeshTri1' object has no attribute 'f'",
     "the facet node table is mesh.facets (2 x n_facets); mesh.p, mesh.t, mesh.t2f and mesh.f2t are the others"),
    # A CLASS WHERE AN INSTANCE BELONGS: Basis(mesh, ElementTriP1), and the same class reached
    # through `element = ElementTriP1` (seventeen script versions of one round bound it that way).
    # Only skfem's own element names are matched, so a class of the script's own is never read as one.
    ("skfem", rf"\b\w*Basis\s*\([^()\n]*?,\s*{_SKFEM_ELEMENT_CLASS}\s*[,)]"
              rf"|\.with_element\s*\(\s*{_SKFEM_ELEMENT_CLASS}\s*\)",
     "TypeError: '>=' not supported between instances of 'property' and 'int'", _SKFEM_ELEMENT_FIX),
    ("skfem", rf"^[ \t]*(?P<v>\w+)[ \t]*=[ \t]*{_SKFEM_ELEMENT_CLASS}[ \t]*$(?:{_NOT_REBOUND}[\s\S])*?"
              rf"(?:\b\w*Basis[ \t]*\([^()\n]*?,[ \t]*(?P=v)[ \t]*[,)]"
              rf"|\.with_element[ \t]*\([ \t]*(?P=v)[ \t]*\)|\bElementVector[ \t]*\([ \t]*(?P=v)[ \t]*[,)])",
     "TypeError: '>=' not supported between instances of 'property' and 'int'", _SKFEM_ELEMENT_FIX),
    ("skfem", rf"\bElementVector\s*\(\s*{_SKFEM_ELEMENT_CLASS}\s*[,)]",
     "TypeError: unsupported operand type(s) for *: 'int' and 'property'", _SKFEM_ELEMENT_FIX),
    # ── NGSolve ───────────────────────────────────────────────────────────
    ("ngsolve", r"\.AddVertex\s*\(",
     "AttributeError: 'SplineGeometry' object has no attribute 'AddVertex'",
     "a rectangle is geo.AddRectangle((X0, Y0), (X1, Y1), bcs=(bottom, right, top, left)); "
     "single points are AddPoint(x, y) / AppendPoint(x, y), with SEPARATE coordinates"),
    ("ngsolve", r"from\s+ngsolve\s+import\s*\([^)]*\b(?:CG|GMRes|MinRes)\b|from\s+ngsolve\s+import[^\n(]*\b(?:CG|GMRes|MinRes)\b",
     "ImportError: cannot import name 'CG' from 'ngsolve'",
     "the Krylov solvers live in ngsolve.solvers (CG, GMRes, MinRes): "
     "`from ngsolve.solvers import CG` (measured on this install)"),
    ("ngsolve", r"\.AddRect\s*\(",
     "AttributeError: 'SplineGeometry' object has no attribute 'AddRect'",
     "the call is AddRectangle, spelled in full"),
    ("ngsolve", r"\.Faces\s*\(\s*\)|\.Vertices\s*\(\s*\)|\.Edges\s*\(\s*\)",
     "AttributeError: 'Mesh' object has no attribute 'Faces' (or 'Vertices')",
     "the mesh iterators are LOWERCASE properties: mesh.vertices, mesh.faces, mesh.edges"),
    ("ngsolve", r"\bfes\.Dofs\s*\(",
     "AttributeError: 'H1' object has no attribute 'Dofs'",
     "a vertex's dof is fes.GetDofNrs(NodeId(VERTEX, vert.nr))[0]; the free set is fes.FreeDofs()"),
    ("ngsolve", r"\.vertexnr\b|NodeId\([^)]*\)\.index\b|\bv\.index\b",
     "AttributeError: the node object has no 'vertexnr' / 'index'",
     "a MeshNode carries .nr (its number) and .point (its coordinates)"),
    ("ngsolve", r"from\s+ngsolve\s+import[^\n]*\b(?:sym|trace|Div|Identity|Transpose)\b",
     "ImportError: cannot import name 'sym' (or 'trace', 'Div', 'Identity', 'Transpose') from 'ngsolve'",
     "the helpers are Sym, Trace, Det, Grad, div, InnerProduct, OuterProduct and Id -- capitalised, "
     "and with no Div/Identity/Transpose at all; a vector space is H1(mesh, order=1, dim=2) and the "
     "strain is Sym(Grad(u))"),
    ("ngsolve", r"\bimport\s*\([^)]*\binverse\b|,\s*inverse\s*[,)]\s*$",
     "ImportError: cannot import name 'inverse' from 'ngsolve'",
     "`inverse` is a KEYWORD of Inverse: a.mat.Inverse(fes.FreeDofs(), inverse='sparsecholesky')"),
    # THE MOST COMMON NGSOLVE FAILURE ON RECORD, and the write-side is where it
    # is cheap: measured across the graded vector-elasticity cells,
    # `cannot import name 'sym' from 'ngsolve'` is the single most frequent
    # NGSolve error, and a live full-task trial burned its budget on `Div`,
    # reporting "the NGSolve version installed has different function names
    # (lowercase div vs uppercase Div)". Both names are unguessable because
    # NGSolve capitalises the opposite way to every other code here.
    ("ngsolve", r"from\s+ngsolve\s+import[^\n]*\bsym\b|\bngsolve\.sym\s*\(",
     "ImportError: cannot import name 'sym' from 'ngsolve'",
     "ngsolve exports Sym, Trace, Grad, grad, div, Id, InnerProduct -- and does NOT export "
     "sym, trace or Div. Sym and Trace act on a MATRIX only, so the strain is Sym(Grad(u)) "
     "and never Sym(u); the identity is Id(mesh.dim), never Id()"),
    ("ngsolve", r"(?<![A-Za-z_.])Div\s*\(|\bngsolve\.Div\b",
     "AttributeError / ImportError: ngsolve has no 'Div'",
     "the divergence is lowercase div(u); inside a stress it is Trace(Sym(Grad(u)))"),
    ("ngsolve", r"\.vec\.vec\b",
     "AttributeError: 'BaseVector' object has no attribute 'vec'",
     "a LinearForm's vector is f.vec and it is already the BaseVector"),
    # BOUNDARY NAMES ARE JOINED WITH |, AS ONE REGULAR EXPRESSION. Measured on this install (NGSolve
    # 6.2.2604): H1(mesh, dirichlet="bottom,right,top") holds 0 of 79 dofs and says nothing, while
    # "bottom|right|top" holds 24; a side built so solved a singular system to a uniform, huge field.
    ("ngsolve", r"""\bdirichlet\s*=\s*f?["'][^"'\n]*,[^"'\n]*["']""",
     "no error, and no dof held there: boundary names joined with commas match no boundary (measured: "
     "0 of 79 dofs held, against 24 with |)",
     "several boundary names are joined with |, as one regular expression: dirichlet='bottom|right|top'; "
     "mesh.GetBoundaries() lists the names the mesh carries"),
    # .data IS LOWERCASE. Measured: reading vec.Data raises, and vec.Data = ... sets a Python attribute
    # and leaves the vector as it was, with no error.
    ("ngsolve", r"\.Data\b",
     "AttributeError: 'ngsolve.la.BaseVector' object has no attribute 'Data' -- or, for vec.Data = ..., "
     "no error and the vector left as it was",
     "a BaseVector is assigned through lowercase .data: vec.data = <expression>"),
    # A BitArray HAS NO Count (measured: AttributeError); NumSet() counts its set bits.
    ("ngsolve", r"\.Count\s*\(\s*\)",
     "AttributeError: 'pyngcore.pyngcore.BitArray' object has no attribute 'Count'",
     "a BitArray counts its set bits with NumSet(), and len() gives its size"),
    ("ngsolve", r"\.vec\s*\[[^]]*FreeDofs",
     "TypeError: __getitem__(): incompatible function arguments (a BitArray is not an index)",
     "take numbers out with v.FV().NumPy() or np.array(v), and assign through v.data"),
    ("ngsolve", r"CoefficientFunction\s*\(\s*(lambda|[A-Z_]+SRC)|(?<![A-Za-z_])CF\s*\(\s*lambda",
     "TypeError: incompatible constructor arguments (a Python function is not a CoefficientFunction)",
     "sample the function at the vertices into a P1 GridFunction and integrate that; a GridFunction "
     "IS a CoefficientFunction"),
    # A MATRIX WRITTEN AS A PYTHON LIST. Measured on this install: CoefficientFunction([[a, b], [c, d]])
    # has no dims and the value a, and every form built on it runs; the same list with dims=(2, 2)
    # raises. A hand-written side solved with the scalar first entry of its conductivity that way.
    ("ngsolve", r"\b(?:CoefficientFunction|CF)\s*\(\s*\[",
     "no error without dims: the list is read as ONE number, its first entry, and the run goes on "
     "with it; with dims=(2, 2) it stops with NgException: dims does not fit to dimension of "
     "CoefficientFunction",
     "a 2x2 matrix is a TUPLE of its four entries, row by row, with dims: "
     "CoefficientFunction((kxx, kxy, kyx, kyy), dims=(2, 2)) -- the form the served NGSolve contract "
     "builds from config.json's k"),
    # ── FEniCSx ───────────────────────────────────────────────────────────
    ("fenics", r"locate_dofs_(?:topological|geometrical)\s*\((?!\s*\[)[^)]*\)\s*\[\s*0\s*\]",
     "every dof but the first silently disappears (no error at that line)",
     "fem.locate_dofs_topological returns the ARRAY for a single space -- drop the [0]; it returns a "
     "pair only when you pass a LIST of two spaces"),
    ("fenics", r"\.subset_dofs\s*\(",
     "AttributeError: 'FunctionSpace' object has no attribute 'subset_dofs'",
     "fem.locate_dofs_topological(V, fdim, facets) or fem.locate_dofs_geometrical(V, marker)"),
    ("fenics", r"ufl\.Constant\s*\(\s*[-\d.]",
     "AttributeError: 'float' object has no attribute 'ufl_domain'",
     "ufl.Constant takes a DOMAIN, not a value: a boundary value has the shape of its space "
     "(default_scalar_type(0.0) on a scalar space, np.zeros(gdim) on a vector one, a fem.Function on "
     "a block of a mixed one) and a constant inside a form is fem.Constant(mesh, value)"),
    ("fenics", r"ufl\.FiniteElement\s*\(",
     "AttributeError: module 'ufl' has no attribute 'FiniteElement'",
     "fem.functionspace(mesh, ('Lagrange', 1)), or basix.ufl.element(...)"),
    ("fenics", r"fem\.VectorFunctionSpace\s*\(",
     "AttributeError: module 'dolfinx.fem' has no attribute 'VectorFunctionSpace'",
     "fem.functionspace(mesh, ('Lagrange', 1, (mesh.geometry.dim,)))"),
    ("fenics", r"\.geometric_dimension\s*\(",
     "AttributeError: a UFL argument has no geometric_dimension()",
     "take the dimension from the mesh: mesh.geometry.dim"),
    # THE NON-PETSc ASSEMBLERS. fem.assemble_vector / fem.assemble_matrix return dolfinx.la objects;
    # the PETSc ones live in dolfinx.fem.petsc. Each trap follows the value from the call that made
    # it to the attribute read on it, and is silent where `fem` is the petsc module.
    ("fenics", rf"{_FEM_IS_NOT_PETSC}(?P<hit>^[ \t]*(?P<v>\w+)[ \t]*=[ \t]*(?:dolfinx\.)?fem\.assemble_vector"
               rf"[ \t]*\((?:{_NOT_REBOUND}[\s\S])*?\b(?P=v)\.(?:ghostUpdate|apply)\b)",
     "AttributeError: 'Vector' object has no attribute 'ghostUpdate' (or 'apply')", _DOLFINX_VECTOR_FIX),
    ("fenics", rf"{_FEM_IS_NOT_PETSC}(?P<hit>^[ \t]*(?P<v>\w+)[ \t]*=[ \t]*(?:dolfinx\.)?fem\.assemble_matrix"
               rf"[ \t]*\((?:{_NOT_REBOUND}[\s\S])*?\b(?P=v)\.(?:mat|assemble|shape)\b)",
     "AttributeError: 'MatrixCSR' object has no attribute 'mat' (or 'assemble', 'shape')", _DOLFINX_CSR_FIX),
    ("fenics", rf"{_FEM_IS_NOT_PETSC}(?P<hit>\bfem\.assemble_vector\s*\((?:[^()\n]|\([^()\n]*\))*?\bbcs\s*=)",
     "TypeError: _assemble_vector_form() got an unexpected keyword argument 'bcs'", _DOLFINX_BCS_FIX),
    ("fenics", r"\.dofmap\.index_map\.size\b|\.topology\.index_map\s*\([^()\n]*\)\.size\b"
               r"|\.geometry\.index_map\s*\(\s*\)\.size\b",
     "AttributeError: 'dolfinx.cpp.common.IndexMap' object has no attribute 'size'", _DOLFINX_INDEXMAP_FIX),
    ("fenics", rf"^[ \t]*(?P<v>\w+)[ \t]*=[ \t]*(?:(?:dolfinx\.)?fem\.)?functionspace[ \t]*\("
               rf"(?:{_NOT_REBOUND}[\s\S])*?\b(?P=v)\.dim\b",
     "AttributeError: 'FunctionSpace' object has no attribute 'dim'", _DOLFINX_INDEXMAP_FIX),
    ("fenics", r"(?<!fem\.)(?<!dolfinx\.)\bMeshTags\s*\([^()\n]*,",
     "TypeError: MeshTags.__init__() takes 2 positional arguments but 5 were given", _DOLFINX_TAGS_FIX),
    ("fenics", r"\bfem\.Measure\s*\(",
     "AttributeError: module 'dolfinx.fem' has no attribute 'Measure'", _DOLFINX_MEASURE_FIX),
    # A BILINEAR FORM BUILT ON A Function (measured on dolfinx 0.10: two recorded sides wrote
    # u = fem.Function(V) as the unknown of a = dot(grad(u), grad(v))*dx and re-wrote their file 4 and
    # 17 times on the error it gives). Read only in a file with no TrialFunction at all: the Function
    # bound to a name, that name under grad, and a LinearProblem after it.
    ("fenics", rf"\A(?![\s\S]*?\bTrialFunctions?\b)[\s\S]*?(?P<hit>^[ \t]*(?P<v>\w+)[ \t]*=[ \t]*"
               rf"(?:(?:dolfinx\.)?fem\.)?Function[ \t]*\([^\n]*(?:{_NOT_REBOUND}[\s\S])*?"
               rf"\bgrad[ \t]*\([ \t]*(?P=v)[ \t]*\)[\s\S]*?\bLinearProblem[ \t]*\()",
     "IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1), raised in "
     "create_matrix inside LinearProblem", _DOLFINX_RANK1_FIX),
    # ── DUNE-fem ──────────────────────────────────────────────────────────
    ("dune", r"ufl\.Eq\s*\(",
     "ImportError / AttributeError: ufl.Eq no longer exists",
     "write the equation with Python's ==: galerkin([a == L, dbc], solver='cg'); ufl.eq is only a "
     "conditional's test"),
    ("dune", r"DirichletBC\([^)]*\bmarker\s*=",
     "TypeError: DirichletBC() got an unexpected keyword argument 'marker'",
     "the signature is DirichletBC(space, value, subDomain=None)"),
    ("dune", r"\binfo\.converged\b",
     "AttributeError: 'dict' object has no attribute 'converged'",
     "scheme.solve returns a DICT: read info['converged']"),
    ("dune", r"\.geometry\.point\b",
     "AttributeError: the Geometry object has no attribute 'point'",
     "a vertex's coordinates are vertex.geometry.center (or .corner(0))"),
    ("dune", r"\bspace\.dim\b|\.space\.dim\b",
     "AttributeError: the space has no attribute 'dim'",
     "a discrete space counts its dofs with space.size (or len(space))"),
    ("dune", r"\bdofCoordinates\s*\(",
     "AttributeError: the space has no attribute 'dofCoordinates'",
     "dof coordinates come from the grid: iterate gridView.vertices and read v.geometry.center, "
     "numbering with gridView.indexSet.index(v)"),
    # ── deal.II (C++, but the trap is what the build dies on) ─────────────
    ("dealii", r"\bFEEvaluation\s*<",
     "a page of template errors inside the deal.II headers (synchronous_iterator.h), not in your file",
     "assemble with FEValues<2> fe_values(fe, quadrature, update_flags), not FEEvaluation -- that is "
     "the matrix-free class and has a different contract"),
    # ── Kratos ────────────────────────────────────────────────────────────
    ("kratos", r"LaplacianElement2D4N",
     "Error: The Element \"LaplacianElement2D4N\" is not registered!",
     "2-D conduction is P1 TRIANGLES: LaplacianElement2D3N (3-D is LaplacianElement3D4N)"),
)

_IMPORT_MARKERS = {
    "skfem": ("from skfem", "import skfem"),
    "ngsolve": ("from ngsolve", "import ngsolve", "from netgen"),
    "fenics": ("from dolfinx", "import dolfinx"),
    "dune": ("from dune", "import dune"),
    "kratos": ("import KratosMultiphysics", "from KratosMultiphysics"),
    "dealii": ("#include <deal.II/", "dealii::", "using namespace dealii"),
}


def looks_like_participant(text: str) -> bool:
    """A coupling participant: it speaks the driver's handshake."""
    return "imports.json" in text and "exports.json" in text


def backends_in(text: str) -> list:
    return [b for b, marks in _IMPORT_MARKERS.items() if any(m in text for m in marks)]


def _strip_strings_and_comments(text: str) -> str:
    """Docstrings and comments quote the WRONG calls on purpose -- every served contract does, and so
    does this module. Linting them would name a defect that is not there, and a gate that names an
    absent defect costs the agent an action for nothing (measured, honest rounds)."""
    out = re.sub(r'"""(?:.|\n)*?"""', "", text)
    out = re.sub(r"'''(?:.|\n)*?'''", "", out)
    out = re.sub(r"#[^\n]*", "", out)
    return out


_MODULE_USE = (
    ("dolfinx", r"\bdolfinx\.[A-Za-z_]", "import dolfinx",
     "NameError: name 'dolfinx' is not defined",
     "`from dolfinx import fem, mesh` does NOT bind the module: add a bare `import dolfinx` when you "
     "call dolfinx.log.set_log_level(...)"),
    ("ngsolve", r"\bngsolve\.[A-Za-z_]", "import ngsolve",
     "NameError: name 'ngsolve' is not defined",
     "`from ngsolve import ...` does NOT bind the module: add a bare `import ngsolve` when you set "
     "ngsolve.ngsglobals.msg_level"),
    ("logging", r"\blogging\.[A-Za-z_]", "import logging",
     "NameError: name 'logging' is not defined",
     "add `import logging` before logging.basicConfig(level=logging.INFO)"),
)


def _module_findings(body: str, source: str) -> list:
    """A module used by name that the script never imported.

    Measured 2026-09-14: a FEniCSx worker followed both the task and the served facts, wrote
    dolfinx.log.set_log_level(...) at the top of the contract, and died with NameError -- the served
    contracts did `from dolfinx import fem` and never bound the module. They bind it now; a script
    written from scratch still may not."""
    # AN IMPORT LINE IS NOT A USE. `from dolfinx.fem.petsc import LinearProblem` contains
    # "dolfinx." and binds nothing by that name -- scanning it flagged four served contracts that
    # are correct (measured while writing this).
    body = "\n".join(l for l in body.split("\n")
                     if not re.match(r"\s*(from|import)\s", l))
    out = []
    for mod, use, imp, error, fix in _MODULE_USE:
        if not re.search(use, body):
            continue
        if re.search(rf"^\s*{imp}\b", source, re.M) or re.search(rf"^\s*import\s+[^\n]*\b{mod}\s+as\b", source, re.M):
            continue
        out.append(f"{mod}: this script calls `{mod}....` but never imports the module -- "
                   f"the run stops with {error}. {fix}. Measured on this install.")
    return out


# THE SCRAMBLED SQUARE, NAMED. A written script arrives with `(a)**2 + (b)**2` as
# `(a)**2 + **(b)2`: the second exponent moved in front of its bracket. It is a scramble in
# the text the model wrote, not a Python or solver rule, and it survives re-writing: measured
# 2026-09-28 in the web interface, one run wrote the file three times and the scramble came
# back each time, until the line was changed in place; 34 of the 144 participant scripts that
# do not parse in the recorded coupled runs carry it.
_SCRAMBLED_SQUARE = re.compile(r"[+\-/=]\s*\*\*\s*\(([^()\n]{1,80})\)(\d+)")


def scrambled_exponent_hint(text: str) -> str | None:
    """A plain sentence naming a `+ **(b)2` scramble in a SyntaxError text, or None."""
    if not isinstance(text, str) or "SyntaxError" not in text:
        return None
    m = _SCRAMBLED_SQUARE.search(text)
    if not m:
        return None
    inner, exp = m.group(1).strip(), m.group(2)
    alt = f" (or np.square({inner}))" if exp == "2" else ""
    return (f"`**({inner}){exp}` is the power `({inner})**{exp}` written with its exponent moved "
            f"in front of the bracket: a scramble in the written text, not a Python or solver rule. "
            f"Write `({inner})**{exp}`{alt}. The same scramble tends to come back when the whole "
            f"file is written again, so change that one line in place.")


_OTHER_SIDE = {"dirichlet": "neumann", "neumann": "dirichlet"}


def _side_test(test) -> tuple:
    """(side, plain) for an `if` test that picks one side: `SIDE == "x"`, alone (plain: its else
    is the other side) or as the first operand of an `and`; ('', False) for any other test."""
    import ast
    t = test.values[0] if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And) else test
    if (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "SIDE"
            and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq) and isinstance(t.comparators[0], ast.Constant)
            and t.comparators[0].value in _OTHER_SIDE):
        return t.comparators[0].value, t is test
    return "", False


def undefined_names(text: str) -> list:
    """Names the script USES at module level and never defines anywhere.

    This is the leave-behind contract, checked. The served contract names every variable its surviving
    lines still use -- iface_dofs, y_if, a, b_vol and the rest -- and says plainly that the hole has to
    define them. Measured 2026-09-14 on the DUNE 3-D step trial: a worker's script called
    resample_plane(imp, "normal_fluxes", Q_INIT, pts) with no `pts` anywhere, and the run died with
    UnboundLocalError after the JIT had already spent minutes compiling. A name that is never assigned
    cannot be found by any amount of running.

    Deliberately conservative: module level only, and a name bound ANYWHERE in the file counts as
    defined. It reports what cannot work, not what looks unusual.
    """
    import ast
    import builtins
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        lines = text.splitlines()
        where = (f": `{lines[e.lineno - 1].strip()[:120]}`"
                 if e.lineno and 0 < e.lineno <= len(lines) else "")
        hint = scrambled_exponent_hint(f"SyntaxError {where}")
        return [f"this file does not parse: {e.__class__.__name__} at line {e.lineno} -- {e.msg}{where}"
                + (f" {hint}" if hint else "")]
    bound = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            bound.add(getattr(node, "name", ""))
            a = getattr(node, "args", None)
            if a is not None:
                for arg in list(a.args) + list(a.posonlyargs) + list(a.kwonlyargs):
                    bound.add(arg.arg)
                for extra in (a.vararg, a.kwarg):
                    if extra is not None:
                        bound.add(extra.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                bound.add((al.asname or al.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.Global):
            bound.update(node.names)
    # A STAR IMPORT BINDS NAMES THIS CHECK CANNOT SEE. Measured: a file with
    # `from ngsolve import *` drew 13 false "never defined" replies (x, y, H1, Mesh, ...).
    # The module is the solver's, loaded in the solver's own interpreter, so it is not
    # read here; with a star import in the file no name is judged (a real NameError still
    # stops the run, and the run check names it).
    if any(isinstance(node, ast.ImportFrom) and any(al.name == "*" for al in node.names)
           for node in ast.walk(tree)):
        return []
    # module-level loads only: function bodies may legitimately read module names defined later
    missing, sides = {}, {}

    def visit(node, side):
        # A NAME READ ONLY ON THE OTHER SIDE'S BRANCH DOES NOT STOP THIS SIDE'S RUN. Measured on a
        # coupled round: two Neumann sides were told `u_if` and `keep` would stop the run with
        # NameError; both are read only under `if SIDE == "dirichlet"`, and each worker re-typed its
        # file (about 4 minutes each) to answer a stop that could not happen.
        if isinstance(node, ast.If):
            s, plain = _side_test(node.test)
            if s:
                visit(node.test, side)
                for st in node.body:
                    visit(st, s)
                for st in node.orelse:
                    visit(st, _OTHER_SIDE[s] if plain else side)
                return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in bound:
            missing.setdefault(node.id, node.lineno)
            sides.setdefault(node.id, set()).add(side)
        for ch in ast.iter_child_nodes(node):
            visit(ch, side)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        visit(node, "")
    stated = _stated_side(text)
    if stated:
        missing = {n: ln for n, ln in missing.items() if sides.get(n) != {_OTHER_SIDE[stated]}}
    return [f"`{n}` is used at line {ln} and never defined in this file -- the run stops with "
            f"NameError/UnboundLocalError there. The served contract lists the names its surviving "
            f"lines need; every one of them has to come out of your solve."
            for n, ln in sorted(missing.items(), key=lambda kv: kv[1])]


# KRATOS'S SOLVER MESSAGES, ANSWERED FOR WHAT EACH WAS MEASURED TO COME FROM (Kratos 10.3 on this
# install: the served 3-D contract with its hex split swapped for recorded and constructed ones,
# placeholder data). 'Error zero sum' was answered as element orientation with a swap of two
# nodes, and with a node in no element and a problem with nothing held as the other causes.
# Measured: a split MIXING positive and negative tetrahedra prints it (skyline LU; AMGCL stops the
# same mesh with 'Zero sum in skyline_lu factorization'), and a recorded split that kept a flat
# tetrahedron and left gaps printed it too; all negative prints nothing and returns the right
# field with the flux negated; flat tetrahedra alone give NaN with no message (with every outer
# face held, the RHS warning below). An orphan node and a side with nothing held printed nothing.
# 'ATTENTION! setting the RHS to zero!' had no answer: Kratos prints it when the assembled
# right-hand side is exactly zero and skips the solve. All-zero data prints it; with a nonzero
# source, three flat tetrahedra of six (NaN fluxes) and a mixed split with every outer face held
# (a finite field left at its start values, exported) print it too, and so does a 3-D model whose
# nodal variable list leaves out CONDUCTIVITY and never sets it (the flux exported as 0, finite).
_KRATOS_ZERO_SUM = (
    "Kratos: the skyline LU met a zero pivot, and the field the solve returns holds NaN. Measured on "
    "this install with tetrahedral meshes: a split that mixes positive and negative tetrahedra "
    "prints it (under AMGCL the same mesh stops with 'Zero sum in skyline_lu factorization'), and "
    "so did a recorded split that kept a flat tetrahedron and left gaps; all tetrahedra negative "
    "prints nothing and flips the flux sign, and flat ones alone give NaN with no message. Test the "
    "mesh, not the solver: every tetrahedron's signed volume dot(cross(p1-p0, p2-p0), p3-p0)/6 "
    "above zero and the volumes adding up to the domain's (the served 3-D contract's "
    "check_tets(mp) does both and checks the faces). In 2-D the test is each triangle's signed "
    "area (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0) above zero; on record, a triangle whose nodes run "
    "clockwise printed 'Error zero in diagonal'")
_KRATOS_RHS_ZERO = (
    "Kratos: the assembled right-hand side is exactly zero, so Kratos skipped the solve and the "
    "field keeps the values it started from. Measured on this install: all-zero data prints it "
    "(no source on the nodes and every held value zero; then the zero field is the answer); with "
    "a nonzero source, so do tetrahedral meshes with flat tetrahedra (NaN fluxes follow) or with "
    "positive and negative ones mixed (a finite field left at its start values, exported), and "
    "so does a model part whose AddNodalSolutionStepVariable list leaves out CONDUCTIVITY (a "
    "finite flux of 0, exported). When your source or a held value is not zero, test the mesh "
    "before the source: every tetrahedron's signed volume above zero and the volumes adding up "
    "to the domain's (the served 3-D contract's check_tets(mp)); then that CONDUCTIVITY is in "
    "the variable list and set on every node")


# The error text a run prints -> the call that works. Same measurements as _TRAPS, keyed the other
# way round: a run that already failed should not cost a second run to diagnose.
_SPLINE_GEOMETRY = (
    "NGSolve: SplineGeometry is netgen's 2-D geometry and lives in netgen.geom2d -- `from "
    "netgen.geom2d import SplineGeometry`; `from ngsolve import *` does not bring it. A channel "
    "with a round hole: geo.AddRectangle((x0, y0), (x1, y1), bcs=(<bottom>, <right>, <top>, "
    "<left>)), then geo.AddCircle((cx, cy), r=<r>, leftdomain=0, rightdomain=1, bc=<name>) "
    "(leftdomain=0 makes the disc a hole). A mesh made by generate_mesh is read with "
    "Mesh(ReadGmsh(path)) from netgen.read_gmsh instead")
_DOLFINX_GMSH = (
    "FEniCSx 0.10: the Gmsh reader is the module dolfinx.io.gmsh -- `from dolfinx.io import gmsh "
    "as gmshio`, then gmshio.read_from_msh(path, MPI.COMM_WORLD, rank=0, gdim=2), which returns "
    "an object with .mesh, .facet_tags and .physical_groups; `gmshio` was its name up to 0.9")
_NETGEN_2D = (
    "netgen: netgen.csg is the 3-D geometry and has no Rectangle. In 2-D use netgen.geom2d -- "
    "SplineGeometry (AddRectangle, AddCircle) or CSG2d with its own Rectangle(pmin=..., "
    "pmax=..., bc=...) and Circle(center=..., radius=..., bc=...), subtracted with -")


#
# EACH ENTRY IS KEYED TO THE CODES IT IS ABOUT, and answers only a failing script of one of them.
# Measured: a FEniCSx script's `'FunctionSpace' object has no attribute 'dim'` was answered with the
# DUNE-fem fix (space.size), because the needle "has no attribute 'dim'" is every code's. Which code
# failed is read from the traceback (findings_from_output); an empty key is any code's.
# (codes, what the run prints, the call that works)
# SPARTA's file and surface rules, each measured on this install with a placeholder deck
# (tests/test_a_failed_sparta_deck_is_told_the_measured_fix.py).
_SPARTA_SURF_FILE = ("a surf file's FIRST LINE IS ALWAYS SKIPPED, so a header written there is lost. The "
                     "header lines '<N> points' and '<M> lines' come after it, then the line 'Points', one "
                     "skipped line and N rows 'id x y', then the line 'Lines', one skipped line and M rows "
                     "'id p1 p2', each section's rows with no blank line between them")
_SPARTA_FLOW_SIDE = ("SPARTA puts the gas on the side a line's normal N = (0,0,1) x (p2 - p1) points to: "
                     "walking from p1 to p2, the gas is on your left.")
_SPARTA_BOX_FACE = ("compute surf, fix ave/surf and dump surf tally only the elements read_surf made, and a "
                    "box face is not one. A face's flux is `compute <ID> boundary <mix> etot` through `fix "
                    "<F> ave/time 1 <N> <N> c_<ID>[1] mode vector`, read as f_<F>[face] with xlo=1 xhi=2 "
                    "ylo=3 yhi=4; a face takes one wall temperature, so a wall with a value per element is "
                    "read_surf lines")

_ERROR_FIXES: tuple = (
    # THE LARGEST SINGLE UNGATED FAILURE IN THE RECORDED SET: 271 occurrences
    # across 120 cells. Three spellings of one mistake -- a Python function or
    # lambda, a string, or a sympy expression handed to a CoefficientFunction.
    # It lands hardest where it hurts most: the two problem families that have
    # never once produced a coupled level carry it 5 times each per cell.
    (("ngsolve",), "Cannot make CoefficientFunction from",
     "ngsolve: a CoefficientFunction is built from NGSolve's OWN symbolic "
     "coordinates, not from a Python callable, a string or a sympy expression: "
     "`from ngsolve import x, y, z` then write the expression in them -- "
     "CF(x*x + y*y), CF(sin(pi*x)*cos(pi*y)) with sin/cos/exp/log imported "
     "from ngsolve too. A lambda, a def, 'x^2 + y^2' and a sympy object all "
     "raise this. Measured on this install"),
    (("ngsolve",), "Did you mean: 'dx'",
     "ngsolve: the spatial coordinates are NAMES YOU IMPORT, and `dx` is the "
     "volume measure, not a coordinate: `from ngsolve import x, y, z`. "
     "Measured on this install -- ngsolve exports all three"),
    # MEASURED in 22 distinct recorded cells, more than any other UFL mistake.
    # It reads as a backend crash, not a syntax error: one cell reported
    # "segmentation fault during 3D grid/UFL compilation" and abandoned the
    # run with the backend declared broken, while the traceback said this and
    # the script already imported Or on its own import line.
    (("fenics", "dune"), "unsupported operand type(s) for |",
     "UFL: a condition is combined with Or(a, b) and And(a, b), imported from "
     "ufl -- Python's | and & are not defined on UFL conditions. conditional("
     "Or(lt(x[0], a), gt(x[0], b)), 1, 0); nest Or for three or more"),
    (("fenics", "dune"), "unsupported operand type(s) for &",
     "UFL: a condition is combined with And(a, b) and Or(a, b), imported from "
     "ufl -- Python's & and | are not defined on UFL conditions"),
    (("skfem",), "'Dofs' object is not callable",
     "scikit-fem: basis.dofs is an ATTRIBUTE. Select with basis.get_dofs(facets=...) or "
     "basis.get_dofs(lambda x: ...), then .flatten()"),
    (("skfem",), "missing 1 required positional argument: 'ubasis'",
     "scikit-fem: a form takes its basis POSITIONALLY -- laplace.assemble(basis), or asm(laplace, basis)"),
    (("skfem",), "Attribute 'y' not found in 'w'",
     "scikit-fem: inside a form the coordinates are w.x[0] and w.x[1]; there is no w.y"),
    # after the 'y' entry, which it contains: a needle inside one already named is not named again
    (("skfem",), "not found in 'w'", "scikit-fem: " + _SKFEM_W_FIX),
    (("skfem",), "'>=' not supported between instances of 'property' and 'int'",
     "scikit-fem: " + _SKFEM_ELEMENT_FIX),
    (("skfem",), "unsupported operand type(s) for *: 'int' and 'property'",
     "scikit-fem: " + _SKFEM_ELEMENT_FIX),
    (("skfem",), "Input array has wrong size.", "scikit-fem: " + _SKFEM_SIZE_FIX),
    (("skfem",), "has no attribute 'init_rect'",
     "scikit-fem: a rectangle is MeshTri.init_tensor(np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1))"),
    (("skfem",), "unexpected keyword argument 'doforder'",
     "scikit-fem: Basis(mesh, element) takes no doforder keyword"),
    (("skfem",), "has no attribute 'find_dofs'",
     "scikit-fem: every basis has get_dofs, not find_dofs"),
    (("skfem",), "sym_grad() takes 1 positional argument but 2 were given",
     "scikit-fem: sym_grad takes the field alone, sym_grad(u); the form's third argument w is not "
     "passed to it"),
    (("skfem",), "has no attribute 'rect'",
     "scikit-fem: a rectangle is MeshTri.init_tensor(np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1))"),
    (("skfem",), "has no attribute 'find_nodes'",
     "scikit-fem: a basis selects dofs with basis.get_dofs(lambda x: np.isclose(x[0], X)) and a mesh "
     "its nodes with mesh.nodes_satisfying(lambda x: np.isclose(x[0], X))"),
    # Measured on a coupled elastic round: four of five DUNE sides passed the DirichletBC as galerkin's
    # second argument; one then wrote dbc=, which is accepted and never applied, and its CG ran until killed.
    (("dune",), "'DirichletBC' object has no attribute 'gridView'",
     "DUNE-fem: galerkin's second positional argument is the space, not a DirichletBC; the Dirichlet "
     "condition goes into the list with the form, galerkin([a == b, dbc], solver=...). A dbc= keyword "
     "is accepted and the condition is never applied: a held unit square then solved as if nothing "
     "were held"),
    (("ngsolve",), "cannot import name 'trace' from 'ngsolve'",
     "NGSolve: the tensor helpers are CAPITALISED and the set is not UFL's. Measured on this install, ngsolve EXPORTS Sym, Trace, Det, Grad, grad, div, Id, InnerProduct, OuterProduct, Inv, Cof -- and does NOT export sym, trace, det, Div, Identity or Transpose. The strain is Sym(Grad(u)), the trace Trace(...), the identity Id(2)"),
    (("ngsolve",), "cannot import name 'det' from 'ngsolve'",
     "NGSolve: the tensor helpers are CAPITALISED and the set is not UFL's. Measured on this install, ngsolve EXPORTS Sym, Trace, Det, Grad, grad, div, Id, InnerProduct, OuterProduct, Inv, Cof -- and does NOT export sym, trace, det, Div, Identity or Transpose. The strain is Sym(Grad(u)), the trace Trace(...), the identity Id(2)"),
    (("ngsolve",), "cannot import name 'Identity' from 'ngsolve'",
     "NGSolve: the tensor helpers are CAPITALISED and the set is not UFL's. Measured on this install, ngsolve EXPORTS Sym, Trace, Det, Grad, grad, div, Id, InnerProduct, OuterProduct, Inv, Cof -- and does NOT export sym, trace, det, Div, Identity or Transpose. The strain is Sym(Grad(u)), the trace Trace(...), the identity Id(2)"),
    (("ngsolve",), "cannot import name 'Transpose' from 'ngsolve'",
     "NGSolve: the tensor helpers are CAPITALISED and the set is not UFL's. Measured on this install, ngsolve EXPORTS Sym, Trace, Det, Grad, grad, div, Id, InnerProduct, OuterProduct, Inv, Cof -- and does NOT export sym, trace, det, Div, Identity or Transpose. The strain is Sym(Grad(u)), the trace Trace(...), the identity Id(2)"),
    (("ngsolve",), "cannot import name 'SetLogLevels' from 'ngsolve'",
     "NGSolve: there is no SetLogLevels. Verbosity is a module global -- "
     "import ngsolve (the MODULE) and set ngsolve.ngsglobals.msg_level = 3 "
     "before the solve, or the run log carries none of this code's output"),
    (("ngsolve",), "cannot import name 'DirichletBC' from 'ngsolve'",
     "NGSolve: there is no DirichletBC class -- the condition belongs to the "
     "SPACE, as H1(mesh, order=..., dirichlet='<boundary names separated by "
     "|>'), and the VALUES are written into the GridFunction's vector at those "
     "dofs. fes.FreeDofs() then excludes them at solve time"),
    (("ngsolve",), "'ngsolve.comp.H1' object has no attribute 'VDim'",
     "NGSolve: the component count of a space is fes.dim (a vector space is "
     "H1(mesh, order=1, dim=2)); there is no VDim"),
    ((), "No module named 'KratosMultiphysics'",
     "Kratos: this code has its own interpreter. You ran it with one that does not have it. discover(query='list') names the interpreter or binary of every backend on this install -- use that exact path in the participant's `command`"),
    ((), "No module named 'dolfinx'",
     "FEniCSx: this code has its own interpreter. You ran it with one that does not have it. discover(query='list') names the interpreter or binary of every backend on this install -- use that exact path in the participant's `command`"),
    ((), "No module named 'ngsolve'",
     "NGSolve: this code has its own interpreter. You ran it with one that does not have it. discover(query='list') names the interpreter or binary of every backend on this install -- use that exact path in the participant's `command`"),
    ((), "No module named 'skfem'",
     "scikit-fem: this code has its own interpreter. You ran it with one that does not have it. discover(query='list') names the interpreter or binary of every backend on this install -- use that exact path in the participant's `command`"),
    ((), "No module named 'dune'",
     "DUNE-fem: this code has its own interpreter. You ran it with one that does not have it. discover(query='list') names the interpreter or binary of every backend on this install -- use that exact path in the participant's `command`"),
    ((), "No module named 'netgen'",
     "NGSolve/netgen: this code has its own interpreter. You ran it with one that does not have it. discover(query='list') names the interpreter or binary of every backend on this install -- use that exact path in the participant's `command`"),
    (("fenics",), "has no attribute 'MeshTags'", "FEniCSx: " + _DOLFINX_TAGS_FIX),
    (("fenics",), "module 'dolfinx.fem' has no attribute 'MeshTag'", "FEniCSx: " + _DOLFINX_TAGS_FIX),
    (("fenics",), "MeshTags.__init__() takes 2 positional arguments", "FEniCSx: " + _DOLFINX_TAGS_FIX),
    ((), "Invalid format specifier",
     "Python f-string: a literal { or } inside an f-string starts a "
     "replacement field, so JSON written with an f-string breaks on its own "
     "braces. DOUBLE them -- {{ and }} -- or build the dict and json.dumps it. "
     "Measured: f'{\"a\":\"x\"}' raises this, f'{{\"a\":\"x\"}}' returns the "
     "string"),
    (("kratos",), "Kratos.ModelPart: No constructor defined!",
     "Kratos: a ModelPart is created BY a Model, never constructed directly -- "
     "model = KM.Model(); mp = model.CreateModelPart('<name>'). Measured: "
     "KM.ModelPart('x') raises this, model.CreateModelPart('x') returns one"),
    (("skfem",), "'CellBasis' object has no attribute 'ndof'",
     "scikit-fem: a basis counts its degrees of freedom with basis.N, not "
     ".ndof -- measured: N is 16 for a scalar P1 basis on a mesh where a "
     "vector basis of the same mesh gives 32, and len(basis.zeros()) is the "
     "same number. basis.nelems is the ELEMENT count, a different quantity"),
    (("skfem",), "CellBasis.__init__() missing 1 required positional argument: 'elem'",
     "scikit-fem: Basis(mesh, element) needs the ELEMENT -- Basis(mesh, "
     "ElementTriP1()) for a scalar P1 field, Basis(mesh, "
     "ElementVector(ElementTriP1())) for a vector one"),
    (("fenics",), "compute_colliding_cells(): incompatible function arguments",
     "FEniCSx: the point array must have THREE columns even in 2-D -- "
     "np.column_stack([x, y, np.zeros_like(x)]) -- for both "
     "geometry.compute_collisions_points(bb_tree, pts) and "
     "geometry.compute_colliding_cells(mesh, candidates, pts). Measured: an "
     "(n, 2) array raises this, an (n, 3) array returns the AdjacencyList"),
    (("fenics",), "compute_collisions_points(): incompatible function arguments",
     "FEniCSx: the point array must have THREE columns even in 2-D -- "
     "np.column_stack([x, y, np.zeros_like(x)]). Measured: an (n, 2) array "
     "raises this"),
    (("ngsolve",), "Boundaries(): incompatible function arguments",
     "NGSolve: mesh.Boundaries takes the boundary NAME as a string, not an "
     "index -- mesh.Boundaries('left') or a regex over the names, "
     "mesh.Boundaries('left|top'). The names are the ones you passed as bcs= "
     "when you built the geometry"),
    (("ngsolve",), "GetDofs(): incompatible function arguments",
     "NGSolve: to get the dof numbers of a mesh node use "
     "fes.GetDofNrs(NodeId(VERTEX, i)), which returns a tuple. GetDofs takes a "
     "Region (mesh.Boundaries('name')), not an integer"),
    (("ngsolve",), "Set(): incompatible function arguments",
     "NGSolve: GridFunction.Set takes a FIELD -- a CoefficientFunction, a "
     "symbolic expression or a scalar -- not a dof array. To write raw values "
     "use gfu.vec.FV().NumPy()[:] = <array>, or index gfu.vec directly. "
     "Measured: gfu.Set(np.zeros(fes.ndof)) raises this, gfu.Set(x*y) works"),
    (("dune",), ".LeafGrid' object has no attribute 'geometry'",
     "DUNE-fem: a grid view has no geometry -- geometry belongs to an ENTITY. "
     "Iterate and ask the element: for e in gridView.elements: "
     "e.geometry.center, e.geometry.volume, e.geometry.corners. Measured, the "
     "grid view itself offers elements, vertices, size, dimension, dimGrid and "
     "indexSet"),
    (("dune",), ".LeafGrid' object has no attribute 'leaves'",
     "DUNE-fem: the entities of a grid view are gridView.elements (and "
     "gridView.vertices); there is no .leaves. Measured on this install"),
    (("fenics", "dune"), "cannot import name 'Eq' from 'ufl'",
     "UFL: there is no Eq, and DO NOT take the interpreter's suggestion here. "
     "It proposes `eq`, which is a BOOLEAN comparison for use inside "
     "ufl.conditional -- measured, ufl.eq(a, L) returns an EQ object without "
     "complaining, and it is not your variational problem. A variational "
     "equation is written with the operator: `a == L`, which returns a ufl "
     "Equation"),
    (("fenics", "dune"), "cannot import name 'Variable' from 'ufl'",
     "UFL: the spelling is lowercase, ufl.variable(expr), and here the "
     "interpreter's suggestion is right -- it marks an expression so you can "
     "differentiate with respect to it via ufl.diff"),
    (("skfem",), "'numpy.ndarray' object has no attribute 'grad'",
     "scikit-fem: inside a form the arguments are DiscreteField objects, which "
     "DO carry .grad -- but the helpers return plain arrays, so grad(u).grad is "
     "this error. Take the derivative once: from skfem.helpers import grad, "
     "then grad(u) is the (dim, ...) array and grad(u)[0], grad(u)[1] are its "
     "components"),
    (("skfem",), "cannot import name 'sym' from 'skfem.helpers'",
     "scikit-fem: the symmetric gradient is one helper, sym_grad, not sym "
     "composed with grad. Measured exports of skfem.helpers: grad, sym_grad, "
     "div, curl, cross, dot, ddot, inner, mul, prod, trace, transpose, det, "
     "inv, eye, identity, jump, d, dd, ddd, dddd, dddot, zeros_like -- and no "
     "sym"),
    (("ngsolve",), "cannot import name 'SetBC' from 'ngsolve'",
     "NGSolve: there is no SetBC. The Dirichlet boundary is an ARGUMENT of the "
     "space -- H1(mesh, order=..., dirichlet='<names|joined>') -- the values go "
     "into the GridFunction's vector at those dofs, and fes.FreeDofs() is what "
     "the solve inverts on"),
    (("dune",), "first argument should be a ufl equation",
     "DUNE-fem: the scheme takes an EQUATION, not a form -- galerkin(a == L, ...) "
     "with `a` the bilinear form and `L` the linear one. Measured: galerkin(a) "
     "raises this, galerkin(a == L) returns a Scheme. For a form with no "
     "right-hand side write `a == 0`"),
    (("dune",), "solve() missing 1 required positional argument: 'target'",
     "DUNE-fem: the scheme solves INTO a discrete function -- uh = "
     "space.interpolate(0, name='u'); scheme.solve(target=uh). It does not "
     "return the solution, it fills the target and returns an info dict"),
    ((), "input is less than 3-dimensional since all points have the same",
     "scipy griddata: the points lie on one line -- an interface -- and Qhull cannot triangulate a "
     "line. Interpolate along it instead: np.interp(t_new, t, values) on the coordinate the interface "
     "runs along (sorted). Measured: griddata on nine points sharing one x raises this QhullError, "
     "np.interp on their y returns the value"),
    (("kratos",), "SetSolutionStepValue(): incompatible function arguments",
     "Kratos: SetSolutionStepValue takes (variable, value) or (variable, step, value), and the "
     "step is an INT -- 0 is the current step. Measured: "
     "node.SetSolutionStepValue(KM.CONDUCTIVITY, 0, 2.5) and (KM.TEMPERATURE, 2.5) work; "
     "(KM.CONDUCTIVITY, 0.0, 2.5) raises this, because 0.0 is a float. Compare the call with the "
     "signatures the error lists: the variable's type (DoubleVariable, IntegerVariable, ...) fixes "
     "the value's type"),
    (("kratos",), "'Kratos.Properties' object has no attribute 'Value'",
     "Kratos: a Properties object uses SetValue/GetValue, not Value -- "
     "props.SetValue(KM.DENSITY, 2.5) then props.GetValue(KM.DENSITY), and "
     "props[KM.DENSITY] reads the same value. Measured on this install"),
    (("ngsolve",), "object has no attribute 'GetOCCGeometry'",
     "netgen: a CSGeometry is not an OCC geometry and cannot be converted. OCC "
     "lives in a different module (netgen.occ.OCCGeometry); for a rectangle use "
     "netgen.geom2d.SplineGeometry -- geo.AddRectangle((X0, Y0), (X1, Y1), "
     "bcs=(bottom, right, top, left)) then geo.GenerateMesh(maxh=...)"),
    ((), "No module named 'kratos'",
     "Kratos: the import is `import KratosMultiphysics as KM`, capitalised and "
     "one word -- there is no lowercase `kratos` module. It also needs its own "
     "interpreter, which discover(query='list') prints"),
    (("ngsolve",), "'ngsolve.comp.H1' object has no attribute 'SetEssentialBC'",
     "NGSolve: a space has no SetEssentialBC. The Dirichlet boundary is an "
     "ARGUMENT of the space -- H1(mesh, order=..., dirichlet='<names|joined>') "
     "-- the values go into the GridFunction's vector, and fes.FreeDofs() is "
     "what the solve inverts on"),
    (("fenics", "dune"), "cannot import name 'abs' from 'ufl'",
     "UFL: there is no ufl.abs -- use the Python builtin abs(expr) on a UFL "
     "expression. Measured, ufl DOES export sign, sqrt, conditional, eq and "
     "variable, so the lowercase spelling is right for those and not for abs"),
    (("fenics",), "module 'dolfinx.fem' has no attribute 'petsc'",
     "FEniCSx: dolfinx.fem.petsc is a SUBMODULE that is not imported with its "
     "parent -- add `import dolfinx.fem.petsc` (or `from dolfinx.fem.petsc "
     "import LinearProblem`) and the attribute appears"),
    (("fenics",), "cannot import name 'NewtonSolver' from 'dolfinx.nls'",
     "FEniCSx: the Newton solver is in the petsc submodule -- `import "
     "dolfinx.nls.petsc` then dolfinx.nls.petsc.NewtonSolver. The same shape as "
     "dolfinx.fem.petsc: importing the parent does not bring it in"),
    (("fenics",), "'MatrixCSR' object has no attribute 'assemble'", "FEniCSx: " + _DOLFINX_CSR_FIX),
    (("fenics",), "'MatrixCSR' object has no attribute 'mat'", "FEniCSx: " + _DOLFINX_CSR_FIX),
    (("fenics",), "'MatrixCSR' object has no attribute 'shape'", "FEniCSx: " + _DOLFINX_CSR_FIX),
    (("fenics",), "'Vector' object has no attribute 'ghostUpdate'", "FEniCSx: " + _DOLFINX_VECTOR_FIX),
    (("fenics",), "'Vector' object has no attribute 'apply'", "FEniCSx: " + _DOLFINX_VECTOR_FIX),
    (("fenics",), "_assemble_vector_form() got an unexpected keyword argument 'bcs'",
     "FEniCSx: " + _DOLFINX_BCS_FIX),
    (("fenics",), "'dolfinx.cpp.common.IndexMap' object has no attribute 'size'",
     "FEniCSx: " + _DOLFINX_INDEXMAP_FIX),
    (("fenics",), "'FunctionSpace' object has no attribute 'dim'", "FEniCSx: " + _DOLFINX_INDEXMAP_FIX),
    (("fenics",), "module 'dolfinx.fem' has no attribute 'Measure'", "FEniCSx: " + _DOLFINX_MEASURE_FIX),
    (("fenics",), "Invalid subdomain_id <dolfinx.mesh.MeshTags", "FEniCSx: " + _DOLFINX_MEASURE_FIX),
    (("fenics",), "Facet to cell connectivity has not been computed",
     "FEniCSx: " + _DOLFINX_CONNECTIVITY_FIX),
    # MEASURED ON THIS INSTALL (dolfinx 0.10): LinearProblem with a rank-1 form where the bilinear
    # form belongs prints exactly `IndexError: vector::_M_range_check: __n (which is 1) >=
    # this->size() (which is 1)` from create_matrix, then the __del__ echo. The run check named the
    # constructor's error and no cause; one side spent about fourteen minutes on it.
    (("fenics",), "vector::_M_range_check", "FEniCSx: " + _DOLFINX_RANK1_FIX),
    # THE DOMINANT dolfinx ERRORS OF ONE COUPLED ROUND, none of which the run check answered
    # (measured: _cpp_object 8 times, 6 of them a bare UFL form; 'Form' not iterable 6; dirichletbc
    # arguments 5; the bcs size mismatch 3; a form divided or multiplied by a number 3; petsc_vec,
    # 'Grad'.ufl and connectivity indexed twice each; createKSP, petsc_mat and get_connectivity
    # once). Each wrong call and each named call was run on this install (dolfinx 0.10).
    # (the pair from collapse() handed over as a space: its frame is locate_dofs_*, its object an
    # ndarray -- answered first, and the uncompiled-form answer below then stays silent on it)
    (("fenics",), re.compile(r"in locate_dofs_(?:topological|geometrical)\b[^\n]*\n(?:[^\n]*\n){0,4}?[^\n]*?"
                             r"(?P<said>'numpy\.ndarray' object has no attribute '_cpp_object')"),
     "FEniCSx: " + _DOLFINX_COLLAPSE_PAIR_FIX),
    (("fenics",), "has no attribute '_cpp_object'", "FEniCSx: " + _DOLFINX_FORM_COMPILED_FIX),
    (("fenics",), "'Form' object has no attribute 'function_spaces'", "FEniCSx: " + _DOLFINX_FORM_COMPILED_FIX),
    (("fenics",), "'Form' object is not iterable", "FEniCSx: " + _DOLFINX_BCS_LIST_FIX),
    (("fenics",), "'DirichletBC' object is not iterable", "FEniCSx: " + _DOLFINX_BCS_LIST_FIX),
    (("fenics",), "object of type 'DirichletBC' has no len()", "FEniCSx: " + _DOLFINX_BCS_LIST_FIX),
    (("fenics",), "Mismatch in size between a and bcs", "FEniCSx: " + _DOLFINX_BCS_LIST_FIX),
    (("fenics",), "unsupported operand type(s) for /: 'Form' and",
     "FEniCSx: a UFL form is scaled from the LEFT or inside its integrand -- (1.0 / dt) * a, or "
     "((1.0 / dt) * u * v) * ufl.dx. Measured on this install: a / 2.0 and a * 2.0 raise this (a / 2 "
     "too), 0.5 * a compiles"),
    (("fenics",), "unsupported operand type(s) for *: 'Form' and",
     "FEniCSx: a UFL form is scaled from the LEFT or inside its integrand -- (1.0 / dt) * a, or "
     "((1.0 / dt) * u * v) * ufl.dx. Measured on this install: a / 2.0 and a * 2.0 raise this (a / 2 "
     "too), 0.5 * a compiles"),
    (("fenics",), "Rank mismatch between Constant and function space", "FEniCSx: " + _DOLFINX_BC_VALUE_FIX),
    (("fenics",), "Boundary condition value must have a dtype attribute", "FEniCSx: " + _DOLFINX_BC_VALUE_FIX),
    (("fenics",), "Invoked with types: dolfinx.cpp.fem.DirichletBC", "FEniCSx: " + _DOLFINX_BC_VALUE_FIX),
    (("fenics",), "Constant size is not equal to the block size", "FEniCSx: " + _DOLFINX_BC_VALUE_FIX),
    (("fenics",), "Cannot tabulate coordinates for a mixed FunctionSpace", "FEniCSx: " + _DOLFINX_TABULATE_MIXED_FIX),
    (("fenics",), "This integral is missing an integration domain", "FEniCSx: " + _DOLFINX_ZERO_FORM_FIX),
    (("fenics",), "cannot import name 'NewtonSolver' from 'dolfinx.fem.petsc'", "FEniCSx: " + _DOLFINX_NONLINEAR_FIX),
    (("fenics",), "module 'dolfinx.fem.petsc' has no attribute 'NewtonSolver'", "FEniCSx: " + _DOLFINX_NONLINEAR_FIX),
    (("fenics",), "module 'dolfinx.mesh' has no attribute 'move'", "FEniCSx: " + _DOLFINX_MOVE_FIX),
    (("fenics",), "cannot import name 'assemble_scalar' from 'dolfinx.fem.petsc'", "FEniCSx: " + _DOLFINX_SCALAR_FIX),
    (("fenics",), "Zero pivot in LU factorization", "FEniCSx: " + _DOLFINX_SADDLE_LU_FIX),
    (("fenics",), "Matrix is missing diagonal entry", "FEniCSx: " + _DOLFINX_SADDLE_LU_FIX),
    (("fenics",), "module 'dolfinx.fem.petsc' has no attribute 'assemble_scalar'", "FEniCSx: " + _DOLFINX_SCALAR_FIX),
    # raised inside ufl's own operators when a ufl.Constant was given an array: its frame is ufl's
    (("fenics",), re.compile(r'ufl/\w+\.py", line \d+, in \w+[^\n]*\n(?:[^\n]*\n){0,4}?[^\n]*?'
                             r'(?P<said>operands could not be broadcast together with shapes[^\n]*)'),
     "FEniCSx: " + _DOLFINX_UFL_CONSTANT_FIX),
    (("fenics",), "'petsc4py.PETSc.Vec' object has no attribute 'petsc_vec'",
     "FEniCSx: dolfinx.fem.petsc.assemble_vector returns the PETSc Vec itself -- use it as it is; "
     "petsc_vec belongs to a Function's vector, uh.x.petsc_vec (measured on this install)"),
    (("fenics",), "'petsc4py.PETSc.Mat' object has no attribute 'petsc_mat'",
     "FEniCSx: dolfinx.fem.petsc.assemble_matrix returns the PETSc Mat itself: "
     "A = assemble_matrix(a_form, bcs=bcs); A.assemble() (measured on this install)"),
    (("fenics",), "'Grad' object has no attribute",
     "FEniCSx: a '.' between two UFL expressions is Python attribute access, not a product: the dot "
     "product is ufl.dot(ufl.grad(u), ufl.grad(v)), or ufl.inner. Measured on this install: "
     "ufl.grad(u) . ufl.grad(v) raises 'Grad' object has no attribute 'ufl', and grad(u) . grad(v) "
     "has no attribute 'grad'"),
    (("fenics",), "'petsc4py.PETSc.Mat' object has no attribute 'createKSP'",
     "FEniCSx: a KSP is its own object, not a method of the matrix: ksp = PETSc.KSP().create(mesh.comm); "
     "ksp.setOperators(A); ksp.setType('preonly'); ksp.getPC().setType('lu'); ksp.solve(b, "
     "uh.x.petsc_vec); uh.x.scatter_forward(), with b lifted and set first (measured on this install)"),
    (("fenics",), "'method' object is not subscriptable",
     "FEniCSx: a method is called with (), not indexed with []: topology.connectivity(d0, d1), after "
     "topology.create_connectivity(d0, d1), then .links(i); dofmap.cell_dofs(c). Measured on this "
     "install: topology.connectivity[d0, d1] and dofmap.cell_dofs[c] raise this"),
    (("fenics",), "'Topology' object has no attribute 'get_connectivity'",
     "FEniCSx: there is no get_connectivity: topology.create_connectivity(d0, d1), then "
     "topology.connectivity(d0, d1).links(i) (measured on this install)"),
    (("skfem",), "cannot import name 'plane_strain' from 'skfem.models.elasticity'",
     "scikit-fem: there is no plane_strain, because lame_parameters IS the "
     "plane-strain pair. Measured exports of skfem.models.elasticity: "
     "BilinearForm, ddot, eye, lame_parameters, linear_elasticity, "
     "linear_stress, plane_stress, sym_grad, trace. For PLANE STRAIN write "
     "linear_elasticity(*lame_parameters(E, nu)); plane_stress(E, nu) is the "
     "separate plane-stress helper"),
    (("skfem",), "'Dofs' object has no attribute 'shape'",
     "scikit-fem: get_dofs returns a DofsView, not an array. It has .flatten() "
     "for the index array, .all() (optionally .all('u^1')) and .nodal['u^1'] / "
     ".nodal['u^2'] for a vector basis -- and no .shape and no .size"),
    (("skfem",), "get_dofs() got an unexpected keyword argument 'component'",
     "scikit-fem: a vector basis selects a component BY NAME, not by component=. "
     "d = basis.get_dofs(<facets or predicate>); the x dofs are d.nodal['u^1'] and the y dofs "
     "d.nodal['u^2'] (one-based, there is no u^0); d.all() and d.flatten() give both interleaved"),
    (("skfem",), "'MeshTri1' object has no attribute 'extend'", "scikit-fem: " + _SKFEM_JOIN_FIX),
    (("skfem",), "'MeshTri1' object has no attribute 'f'",
     "scikit-fem: the facet node table is mesh.facets; mesh.p, mesh.t, mesh.t2f and mesh.f2t are the others"),
    (("ngsolve",), "has no attribute 'AddVertex'",
     "NGSolve: a rectangle is geo.AddRectangle((X0, Y0), (X1, Y1), bcs=(bottom, right, top, left)); "
     "single points are AddPoint(x, y) with SEPARATE coordinates"),
    (("ngsolve",), "has no attribute 'AddRect'",
     "NGSolve: the call is AddRectangle, spelled in full"),
    (("ngsolve",), "object has no attribute 'Faces'",
     "NGSolve: the mesh iterators are lowercase properties -- mesh.vertices, mesh.faces, mesh.edges"),
    (("ngsolve",), "object has no attribute 'Vertices'",
     "NGSolve: the mesh iterators are lowercase properties -- mesh.vertices, mesh.faces, mesh.edges"),
    # A TEST FUNCTION REBOUND TO A MESH VERTEX: three messages, one cause (see _NGSOLVE_REBOUND_FIX).
    # The 'Operator' one comes before the general MeshNode entry, whose .nr/.point fact is not its fix.
    (("ngsolve",), re.compile(r"Invoked with: [^\n]*?\bV\d+\b"), "NGSolve: " + _NGSOLVE_REBOUND_FIX),
    (("ngsolve",), "for *: 'ngsolve.comp.MeshNode'", "NGSolve: " + _NGSOLVE_REBOUND_FIX),
    (("ngsolve",), "MeshNode' object has no attribute 'Operator'", "NGSolve: " + _NGSOLVE_REBOUND_FIX),
    (("ngsolve",), "MeshNode' object has no attribute",
     "NGSolve: a MeshNode carries .nr and .point; a vertex's dof is fes.GetDofNrs(NodeId(VERTEX, vert.nr))[0]"),
    (("ngsolve",), "dims does not fit to dimension of CoefficientFunction", "NGSolve: " + _NGSOLVE_DIMS_FIX),
    (("ngsolve",), "cannot import name 'inverse' from 'ngsolve'",
     "NGSolve: `inverse` is a KEYWORD of Inverse, not an import -- "
     "a.mat.Inverse(fes.FreeDofs(), inverse='sparsecholesky')"),
    (("ngsolve",), "cannot import name 'sym' from 'ngsolve'",
     "NGSolve: there is no `sym`, `trace` or `Div`. Sym and Trace exist but act on a MATRIX -- the "
     "strain is Sym(Grad(u)), never Sym(u) (that raises 'Sym of non-matrix called'); the "
     "divergence is lowercase div(u); and the identity is Id(mesh.dim), never Id()"),
    (("ngsolve",), "Sym of non-matrix called",
     "NGSolve: Sym takes a matrix, so build the strain from the gradient -- Sym(Grad(u)), not Sym(u)"),
    (("ngsolve",), "cannot import name 'Div' from 'ngsolve'",
     "NGSolve: the divergence is lowercase div(u); inside a stress it is Trace(Sym(Grad(u)))"),
    (("ngsolve",), "object has no attribute 'Data'",
     "NGSolve: a BaseVector is assigned through lowercase .data, vec.data = <expression>; vec.Data = ... "
     "raises nothing and leaves the vector as it was (measured)"),
    (("ngsolve",), "BitArray' object has no attribute 'Count'",
     "NGSolve: a BitArray counts its set bits with NumSet(), and len() gives its size"),
    (("ngsolve",), "BaseVector' object has no attribute 'vec'",
     "NGSolve: a LinearForm's vector IS f.vec; assign through .data and read numbers with FV().NumPy()"),
    (("ngsolve",), "must not have TrialFunction",
     "NGSolve: a LinearForm carries only the test function; every term with the trial function belongs "
     "in the BilinearForm"),
    # MEASURED ON THIS INSTALL: a 2-D structure deck with SOLID QUAD4 elements stops 4C with exactly
    # this line; three cells of a fluid-structure round wrote SOLID for a plane-strain wall.
    (("fourc",), re.compile(r"(?P<said>Element 'SOLID' does not seem to know cell type '(?:quad|tri)\d+')"),
     "4C: SOLID takes 3-D cells only (HEX8, HEX20, HEX27, TET4, TET10, WEDGE6, PYRAMID5, ...); a 2-D structure "
     "is WALL, e.g. `<id> WALL QUAD4 <4 nodes> MAT 1 KINEM linear EAS none THICK 1.0 STRESS_STRAIN "
     "plane_strain GP 2 2` (run on this install; the grammar, `4C -p`, also lists WALL QUAD8, QUAD9, TRI3, "
     "TRI6), with MAT_Struct_StVenantKirchhoff for linear elasticity"),
    (("fenics",), "has no attribute 'subset_dofs'",
     "FEniCSx: fem.locate_dofs_topological(V, fdim, facets) or fem.locate_dofs_geometrical(V, marker)"),
    (("fenics",), "'float' object has no attribute 'ufl_domain'",
     "FEniCSx: ufl.Constant takes a DOMAIN, not a value -- a boundary value has the shape of its space "
     "(a number on a scalar space, np.zeros(gdim) on a vector one, a fem.Function on a block of a mixed "
     "one), and a constant inside a form is fem.Constant(mesh, value)"),
    (("fenics",), "has no attribute 'FiniteElement'",
     "FEniCSx: build spaces with fem.functionspace(mesh, ('Lagrange', 1))"),
    (("fenics",), "has no attribute 'VectorFunctionSpace'",
     "FEniCSx: fem.functionspace(mesh, ('Lagrange', 1, (mesh.geometry.dim,)))"),
    (("fenics",), "missing 1 required keyword-only argument: 'petsc_options_prefix'",
     "FEniCSx: LinearProblem and NonlinearProblem both require petsc_options_prefix='<any name>' on "
     "this install (measured exactly this message from each)"),
    # `'LinearProblem' object has no attribute '_solver'` is not in this table: it is
    # LinearProblem.__del__ tidying up after a failed constructor, and findings_from_output names the
    # constructor's own error instead (see _del_echo_finding).
    (("dune",), "has no attribute 'geometry.point'",
     "DUNE-fem: a vertex's coordinates are vertex.geometry.center (or .corner(0))"),
    (("dune",), "'Geometry' object has no attribute 'point'",
     "DUNE-fem: a vertex's coordinates are vertex.geometry.center (or .corner(0))"),
    # ufl.Form has no copy in both codes' UFL (2025.2.1 with dolfinx, 2024.2.0 with DUNE-fem)
    (("dune", "fenics"), "'Form' object has no attribute 'copy'",
     "UFL: a form is immutable -- build the second one by writing the expression again"),
    (("dune",), "has no attribute 'converged'",
     "DUNE-fem: scheme.solve returns a DICT -- read info['converged']"),
    (("dune",), "has no attribute 'dim'",
     "DUNE-fem: a discrete space counts its dofs with space.size (or len(space))"),
    (("dune",), "has no attribute 'dofCoordinates'",
     "DUNE-fem: dof coordinates come from the grid -- gridView.vertices with v.geometry.center, "
     "numbered by gridView.indexSet.index(v)"),
    (("dealii",), "synchronous_iterator.h",
     "deal.II: a wall of template errors inside the deal.II headers names no line of yours; the "
     "FIRST error line names what the compiler rejected. On record it was FEEvaluation where "
     "FEValues belongs (FEEvaluation is the matrix-free class)"),
    # MEASURED ON THIS INSTALL (deal.II 9.8): 'unsubscribe' and the two lines below come from a
    # solver or preconditioner given the NUMBER type as its template argument. No FEValues
    # argument order reproduces 'unsubscribe' here.
    (("dealii",), "request for member \u2018unsubscribe\u2019",
     "deal.II: a preconditioner or solver declared with the number type where deal.II expects the "
     "matrix or vector type -- PreconditionSSOR<SparseMatrix<double>> p; p.initialize(A, 1.2); and "
     "SolverCG<Vector<double>> cg(control)"),
    (("dealii",), "PreconditionSSOR<double>",
     "deal.II: the preconditioner's template argument is the MATRIX type, and it is built with the "
     "default constructor and then initialized: PreconditionSSOR<SparseMatrix<double>> p; "
     "p.initialize(A, 1.2)"),
    (("dealii",), "is not a class, struct, or union type",
     "deal.II: a solver takes the VECTOR type as its template argument, not the number type -- "
     "SolverCG<Vector<double>> cg(control), not SolverCG<double>"),
    # A NAME THAT IS NOT REGISTERED IS ANSWERED FOR WHAT IT NAMES. One entry keyed on "is not
    # registered" answered a missing CONDITION with the conduction ELEMENT. Kratos 10.3.0 says which
    # kind it is, in two shapes each (measured): CreateNewElement / CreateNewCondition follow the
    # error with "The following Elements (Conditions) are registered:" and the list, and a .mdpa
    # read asks to "check the spelling of the element (condition) name". Each needle is one kind's.
    *((("kratos",), _needle,
       "Kratos: the name that is not registered is an ELEMENT. A name exists only once the "
       "application that defines it is imported, and most names end in the dimension and node "
       "count (2D3N: 2-D, 3 nodes). `import KratosMultiphysics.ConvectionDiffusionApplication` "
       "registers the conduction elements: LaplacianElement2D3N in 2-D, which is P1 TRIANGLES "
       "(there is no LaplacianElement2D4N), and LaplacianElement3D4N in 3-D; without that import "
       "neither exists. CreateNewElement's error lists every element that is registered")
      for _needle in ("The following Elements are registered", "spelling of the element name")),
    *((("kratos",), _needle,
       "Kratos: the name that is not registered is a CONDITION, which sits on the boundary. A name "
       "exists only once the application that defines it is imported, and most names end in the "
       "dimension and node count of the boundary piece: 2D2N for an edge in 2-D, 3D3N for a "
       "triangle face in 3-D. `import KratosMultiphysics.ConvectionDiffusionApplication` registers "
       "the heat-flux conditions FluxCondition2D2N and ThermalFace2D2N (3-D: FluxCondition3D3N, "
       "ThermalFace3D3N); without that import neither exists, while LineCondition2D2N needs no "
       "application. CreateNewCondition's error lists every condition that is registered")
      for _needle in ("The following Conditions are registered", "spelling of the condition name")),
    # 21 recorded transcripts carry one of these two; one run met it four times,
    # exported NaN each time and replaced the served contract to get past it.
    # EACH ANSWER SAYS WHAT ITS MESSAGE WAS MEASURED TO COME FROM (see _KRATOS_ZERO_SUM).
    (("kratos",), "Error zero sum", _KRATOS_ZERO_SUM),
    (("kratos",), "Zero sum in skyline_lu factorization", _KRATOS_ZERO_SUM),
    (("kratos",), "setting the RHS to zero", _KRATOS_RHS_ZERO),
    (("kratos",), "Error zero in diagonal",
     "Kratos: a zero on the diagonal of the assembled system -- on record, a triangle whose "
     "nodes run clockwise (negative area); take each element's signed area and swap two "
     "nodes where it is negative"),
    # Measured 2026-09-29 in the web interface: one NGSolve run failed nine times in a row, on
    # SplineGeometry (three times), Rectangle after `from netgen.csg import *`, a .msh path that
    # was not in the job folder, mesh.ndof and a list given as `dirichlet`, and then gave up. The
    # ReadGmsh and dolfinx.io.gmsh entries were measured beside them. Every one is re-derived on
    # this install by tests/test_a_failed_run_is_told_the_measured_fix.py.
    (("ngsolve",), "name 'SplineGeometry' is not defined", _SPLINE_GEOMETRY),
    (("ngsolve",), "cannot import name 'SplineGeometry'", _SPLINE_GEOMETRY),
    (("ngsolve",), "name 'Rectangle' is not defined", _NETGEN_2D),
    (("ngsolve",), "'ngsolve.comp.Mesh' object has no attribute 'ndof'",
     "NGSolve: a mesh counts its vertices with mesh.nv and its elements with mesh.ne; ndof "
     "belongs to a finite-element SPACE (fes.ndof)"),
    (("ngsolve",), "NgException: Error opening file",
     "NGSolve/netgen: the file is not where the script looked. run_simulation starts the "
     "script in a job folder of its own, so a relative path is read from there: use the "
     "absolute path (generate_mesh's reply gives it). A Gmsh .msh is read with "
     "Mesh(ReadGmsh(path)) from netgen.read_gmsh -- Mesh(path) reads netgen's .vol format "
     "only, and on a .msh it returns an EMPTY mesh (mesh.ne == 0) without an error"),
    (("ngsolve",), "nelem = int(f.readline())",
     "netgen's ReadGmsh reads MSH 2.2 ASCII only; a 4.x file stops it here with 'invalid "
     "literal for int()'. Write 2.2 (gmsh.option.setNumber('Mesh.MshFileVersion', 2.2) "
     "before gmsh.write); openPASO's generate_mesh writes 2.2"),
    (("fenics",), "No module named 'dolfinx.io.gmshio'", _DOLFINX_GMSH),
    (("fenics",), "cannot import name 'gmshio' from 'dolfinx.io'", _DOLFINX_GMSH),
    (("ngsolve",), "Unable to cast Python instance of type <class 'str'> to C++ type",
     "NGSolve: one call that raises this, measured: `dirichlet` given as a Python LIST of "
     "names. It takes ONE string, the boundary names joined by | -- "
     "H1(mesh, order=2, dirichlet='left|top|cylinder')"),
    # ── SPARTA: THE DETERMINISTIC MESSAGES OF A DSMC DECK. Measured on a coupled round: 101
    # SPARTA runs over about twenty distinct messages, and no run check answered one. Each entry
    # below was reproduced on this install (SPARTA 24 Sep 2025) by a placeholder deck that differs
    # from a running one in that one place (tests/test_a_failed_sparta_deck_is_told_the_measured_fix.py).
    (("sparta",), "Incorrect line format in custom attribute file",
     "SPARTA: the file `custom surf create <name> float 0 file <file> M <names>` reads holds comment "
     "or blank lines, then ONE count line 'N M' (N value lines follow, M values on each, M the number "
     "of names after the file name), then N lines 'id v1 .. vM' with no blank line between them; a "
     "file of bare values, or of 'id value' lines with no count line, stops here"),
    (("sparta",), "Cannot use custom surf command before surfaces are defined",
     "SPARTA: `custom surf` sets a value per surface element, so it comes after the read_surf that "
     "makes the elements"),
    (("sparta",), "Surf file does not contain lines", "SPARTA: " + _SPARTA_SURF_FILE),
    (("sparta",), "Read_surf file has no points keyword", "SPARTA: " + _SPARTA_SURF_FILE),
    (("sparta",), "Surf_modify surface group is not defined",
     "SPARTA: surf_modify names a surface group that does not exist. A group comes from `read_surf "
     "<file> group <name>` or `group <name> surf id <lo>:<hi>` before it, and `all` is every element"),
    (("sparta",), re.compile(r"\d+ surface elements not assigned to a collision model"),
     "SPARTA: every surface element needs a wall model: `surf_collide <sc-ID> <style> ...`, then "
     "`surf_modify <group or all> collide <sc-ID>` covering all of them"),
    (("sparta",), "Box boundary not assigned a surf_collide ID",
     "SPARTA: a box face set to s by `boundary` is a surface, and `bound_modify <face> collide "
     "<sc-ID>` (face xlo, xhi, ylo or yhi) must name its wall model. In `boundary`, r is SPECULAR "
     "reflection, s a surface, o outflow, p periodic"),
    (("sparta",), re.compile(r"Illegal boundary command \(\.\./domain\.cpp:\d+\)"),
     "SPARTA: `boundary` takes three entries x y z, in 2-D with z = p; each is one letter for both "
     "faces or two letters for the lower and upper face, from o outflow, p periodic, r specular, s "
     "surface (needs bound_modify), a axisymmetric (lower y face only). On this install line 150 is a "
     "wrong count of entries and line 165 an unknown letter"),
    (("sparta",), "Using read_surf particle none when particles exist",
     "SPARTA: read_surf came after create_particles; read_surf and the surface commands go before "
     "create_particles"),
    (("sparta",), "Cannot use compute surf when surfs do not exist", "SPARTA: " + _SPARTA_BOX_FACE),
    (("sparta",), "Cell type mis-match when marking on self",
     "SPARTA: one cause measured on this install: a second read_surf after a first whose lines face "
     "away from the gas stops here, and with the first file's lines facing the gas both reads pass. "
     + _SPARTA_FLOW_SIDE + " The first read's '<a> <b> <c> = cells outside/inside/overlapping surfs' "
     "line with a = 0 shows a file whose lines face away"),
    (("sparta",), "Stats fix does not compute scalar",
     "SPARTA: a per-surf or per-grid fix cannot go into stats_style; reduce it first, `compute <R> "
     "reduce sum f_<fix>`, and print c_<R>"),
    (("sparta",), "Dump surf and fix not computed at compatible times",
     "SPARTA: a dump of a fix writes only on steps where the fix has a value, so the dump's N must be "
     "a multiple of the fix's Nfreq"),
    (("sparta",), re.compile(r"^[ \t]*0 \d+ \d+ = cells outside/inside/overlapping surfs", re.M),
     "SPARTA: read_surf reported 0 cells outside the surfaces: every grid cell is inside a body, "
     "there is no gas, create_particles makes 0 particles and the run still exits 0. "
     + _SPARTA_FLOW_SIDE + " Swap p1 and p2 of each line whose gas is on its right"),
    (("sparta",), "Created 0 particles",
     "SPARTA: create_particles made none, and the run goes on with an empty domain. With `n 0` it "
     "makes nrho x V / fnum particles (V the flow volume, in 2-D the area). Three causes measured on "
     "this install: no `global nrho <n> fnum <F>` before create_particles (SPARTA then uses nrho = 1, "
     "fnum = 1); a nrho that is not the gas's number density in molecules per m^3 (a mass density or "
     "a count per cell), which puts that count below one; and no grid cell on the flow side of any "
     "surface line (read_surf's line '<a> <b> <c> = cells outside/inside/overlapping surfs' then has "
     "a = 0)"),
    # ── FEBio: THE MESSAGES THAT DO NOT NAME THEIR CAUSE. Measured on three coupled rounds with a
    # FEBio elastic side: 'invalid value for attribute "lid"' 30 times in 13 of 15 cells and
    # 'Invalid load curve ID' in 11, and no run check answered either. Each entry was reproduced on
    # this install (FEBio 4.12) by a placeholder slab deck one change away from a running one
    # (tests/test_a_febio_deck_defect_febio_accepts_silently_is_named.py). What FEBio names itself
    # (an unknown tag, an unknown node set) is left to FEBio.
    (("febio",), "Invalid load curve ID",
     "FEBio: an lc=\"k\" on a <value> names a <load_controller id=\"k\"> inside a <LoadData> section, and "
     "this deck defines none with that id; the message names neither the lc nor the element. A <value> "
     "with no lc is applied in full at every time"),
    (("febio",), re.compile(r'tag "node" \(line \d+\) : invalid value for attribute "lid"'),
     "FEBio: lid in a <NodeData> is the 1-based position in its node_set's own list, 1..N, not a node "
     "id. A lid of 0, a node id, more entries than the set holds, or a <NodeSet> written with spaces "
     "(FEBio keeps the first number between two commas, so the set comes out short) stop here"),
    (("febio",), 'needs to have property "solver" defined',
     "FEBio: the <Control> section holds no <solver> block; removing <solver type=\"solid\"> from a "
     "running deck gives exactly this"),
    (("febio",), re.compile(r"^[ \t]*\*[ \t]+(?P<said>std::exception)[ \t]+\*[ \t]*$", re.M),
     "FEBio: one cause measured on this install: a <value type=\"map\"> that names a map no <NodeData "
     "name=...> in the deck defines"),
    (("febio",), "No force acting on the system.",
     "FEBio: a WARNING, not a stop. It comes when the residual after an iteration falls below FEBio's "
     "floor (squared norm 1e-20), and with small loads that is the first iteration: the field is still "
     "the solved one (loads scaled down by 1e-3 to 1e-10 printed it and gave the scaled field). A zero "
     "load prints it too, with a zero field, so read the node log's values, not the warning"),
    (("febio",), re.compile(r'tag "MeshData" \(line \d+\) : unrecognized tag'),
     "FEBio: <MeshData> is a section of its own after <MeshDomains>; placed inside <Mesh> it stops here"),
    (("febio",), re.compile(r'tag "linear_solver" \(line \d+\) : invalid value for attribute "type"'),
     "FEBio: this build has no MKL solvers. skyline runs; pardiso, mkl_dss, superlu and \"conjugate "
     "gradient\" stop here, and fgmres is read, then refuses the stiffness matrix"),
    (("febio",), re.compile(r'tag "NodeSet" \(line \d+\) : invalid value:'),
     "FEBio 4: a <NodeSet> is one comma-separated id list written as its text; child tags such as "
     "<node id=...> or <n id=...> stop here"),
    (("febio",), re.compile(r'tag "x_dof" \(line \d+\) : unrecognized tag'),
     "FEBio: <x_dof>, <y_dof> and <z_dof> belong to a `zero displacement` bc; a `prescribed "
     "displacement` takes one dof as <dof>x</dof>, so this tag inside one stops here"),
)


# Needles every code can print. They answer only a failure whose code the traceback shows; with no
# code in sight they would name one at a guess.
_NEEDS_THE_CODE = frozenset({
    "This integral is missing an integration domain",     # UFL's, printed for FEniCSx and DUNE-fem
    "has no attribute 'dim'", "has no attribute 'converged'",
    "'numpy.ndarray' object has no attribute 'grad'", "Input array has wrong size.",
    "'>=' not supported between instances of 'property' and 'int'",
    "unsupported operand type(s) for *: 'int' and 'property'",
    "vector::_M_range_check", "'method' object is not subscriptable",
})

# A library frame's package names its code; any other frame is a script whose imports do.
_LIBRARY_CODES = (("/dolfinx/", "fenics"), ("/basix/", "fenics"), ("/ffcx/", "fenics"),
                  ("/skfem/", "skfem"), ("/ngsolve/", "ngsolve"), ("/netgen/", "ngsolve"),
                  ("/dune/", "dune"), ("/KratosMultiphysics/", "kratos"))
_TYPE_CODES = (("dolfinx.", "fenics"), ("skfem.", "skfem"), ("ngsolve.", "ngsolve"),
               ("netgen.", "ngsolve"), ("dune.", "dune"), ("Kratos.", "kratos"))
_FRAME = re.compile(r'^[ \t]*File "([^"\n]+)", line \d+', re.M)
_ERROR_LINE = re.compile(r"^[ \t]*((?:\w+\.)*\w*(?:Error|Exception)\b:[^\n]*)", re.M)


def _failing_codes(output: str) -> set:
    """The codes of the script that failed, read from its traceback.

    A traceback names every file it passed through, and a script's frame is its absolute path
    (Python 3.9 and later), which this process can read: the script's own imports decide. A frame in
    an installed package names that package's code, and a type in the error line may carry its
    module (dolfinx.cpp.common.IndexMap). Empty when the output shows none of these."""
    codes: set = set()
    for path in list(dict.fromkeys(_FRAME.findall(output)))[:40]:
        p = path.replace("\\", "/")
        codes.update(c for seg, c in _LIBRARY_CODES if seg in p)
        if "-packages/" in p or not p.endswith(".py"):
            continue
        try:
            q = Path(path)
            if q.is_file() and q.stat().st_size < 4_000_000:
                codes.update(backends_in(q.read_text(errors="ignore")))
        except OSError:
            continue
    for line in _ERROR_LINE.findall(output):
        codes.update(c for mark, c in _TYPE_CODES if mark in line)
    return codes


# The C library's own words when it finds its heap damaged (glibc's malloc checks).
_HEAP_DAMAGE = re.compile(r"(?:free|malloc|malloc_consolidate|realloc|munmap_chunk)\(\): [^\n]{3,80}"
                          r"|double free or corruption[^\n]{0,20}|invalid fastbin entry \(free\)"
                          r"|corrupted size vs\. prev_size[^\n]{0,30}|corrupted double-linked list")

_SOLVER_ECHO = "'LinearProblem' object has no attribute '_solver'"


def _del_echo_finding(output: str) -> tuple:
    """(finding, the output without the echo) when the run printed LinearProblem.__del__'s echo.

    MEASURED on dolfinx 0.10: LinearProblem.__del__ reads self._solver, which __init__ sets last, so
    a constructor that fails leaves an object whose __del__ prints `'LinearProblem' object has no
    attribute '_solver'`; a constructed problem has _solver and reading it raises nothing. A missing
    keyword prints the echo BEFORE the constructor's own error, an error inside __init__ AFTER it.
    The table used to answer the echo ("touching p._solver raises"), which is false, and the error
    that stopped the run went unnamed. ('', output) when there is no echo."""
    if _SOLVER_ECHO not in output:
        return "", output
    rest = output
    while _SOLVER_ECHO in rest:
        j = rest.find(_SOLVER_ECHO)
        i = rest.rfind("Exception ignored in", 0, j)
        if i < 0 or rest.count("\n", i, j) > 12:
            i = rest.rfind("\n", 0, j) + 1              # not inside an echo block: cut its line only
        k = rest.find("\n", j)
        rest = rest[:i] + (rest[k + 1:] if k >= 0 else "")
    errors = _ERROR_LINE.findall(rest)
    said = ("FEniCSx: that line is LinearProblem.__del__ tidying up an object whose constructor "
            "failed, not what stopped the run")
    if errors:
        own = " ".join(errors[-1].split())
        own = own if len(own) <= 200 else own[:197] + "..."
        said += f"; the constructor's own error is `{own}`, fix that one"
    else:
        said += "; the constructor's own error is not in this output"
    return f"the run printed `{_SOLVER_ECHO}` -- {said}. Measured on this install.", rest


# A PROGRAM KILLED BY SIGSEGV, in every wording this install prints: bash's "Segmentation fault",
# `timeout`'s "the monitored command dumped core" (German locale: "... erzeugte einen
# Speicherauszug"), Python's faulthandler, and an exit code of 139 or -11.
_CRASH = re.compile(r"Segmentation fault|SIGSEGV|signal 11|dumped core|core dumped|Speicherauszug|"
                    r"exit(?:ed)?(?: with)?(?: code| status)? (?:139|-11)\b")
# MEASURED ON THIS INSTALL (Kratos 10.3, the served 3-D contract, 2x2x2): a LaplacianElement3D4N
# model whose ProcessInfo holds no CONVECTION_DIFFUSION_SETTINGS dies at strategy.Solve() with
# SIGSEGV, exit 139, and no message of Kratos's own; `python -X faulthandler` names that line. A
# coupled round met it three times in two cells and the run check said nothing: the crash answer
# was deal.II's alone and did not match `timeout`'s "dumped core".
_KRATOS_SETTINGS = ("settings = KM.ConvectionDiffusionSettings() with SetUnknownVariable(KM.TEMPERATURE), "
                    "SetDiffusionVariable(KM.CONDUCTIVITY), SetVolumeSourceVariable(KM.HEAT_FLUX) and "
                    "SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX), then "
                    "mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, settings), before the solve")
_KRATOS_CRASH = ("the program was KILLED BY SIGSEGV (exit 139, 'dumped core'): a crash inside Kratos's compiled "
                 "code, not an install fault, and Kratos prints no message for it. A LaplacianElement3D4N model "
                 "whose ProcessInfo holds no CONVECTION_DIFFUSION_SETTINGS dies this way at strategy.Solve(): "
                 f"{_KRATOS_SETTINGS}. `python -X faulthandler <script>` prints the Python line a crash died on. "
                 "Measured on this install.")
_UNNAMED_CRASH = ("the program was KILLED BY SIGSEGV (exit 139, 'dumped core'): a crash inside compiled code, not "
                  "an install fault, and the output names no code. What this install measured to crash with no "
                  "message: a Kratos LaplacianElement3D4N model whose ProcessInfo holds no "
                  "CONVECTION_DIFFUSION_SETTINGS (it dies at strategy.Solve()); in a Release deal.II, an FEValues "
                  "accessor whose update flag was not requested, or an index vector handed unsized to "
                  "get_dof_indices. `python -X faulthandler <script>` prints the Python line a crash died on. "
                  "Measured on this install.")
_OTHER_CODE = re.compile(r"envs/dune[-\w]*/bin/python|/dune/|dolfinx|ngsolve|netgen|skfem|\bdune\.")

# A RUN ENDED FROM OUTSIDE, IN PETSC'S WORDS. dune-fem starts PETSc, and PETSc's signal handler
# turns the SIGTERM that `timeout N` sends after N seconds into "PETSC ERROR: Caught signal number
# 15 Terminate", then either a RuntimeError "PETSc Error in the PETSc function 'User provided
# function' ... 'Signal received'" or an abort whose stack runs through PetscSignalHandlerDefault
# (and `timeout` adds "the monitored command dumped core"). Measured on a coupled round: DUNE runs
# were killed so four times by the cells' own `timeout 120`, `180` and `300`, three of them while
# DUNE was compiling (the last line before the handler a "Compiling ... (new)", or the stack in the
# poll that waits for the compiler); nothing answered, and one cell gave up writing that "DUNE-fem
# crashes with PETSc error during UFL form compilation". Reproduced on this install from that
# cell's own script with neutral data: the same stack under `timeout`, and the same script run
# without it compiled on.
_PETSC_SIGNAL = re.compile(r"Caught signal number \d+|PetscSignalHandlerDefault|"
                           r"PETSc function 'User provided function'[^\n]*'Signal received'")
_TIMEOUT_CMD = re.compile(r"(?:^|[\s;&|(])timeout\s+(?:-\S+\s+)*(\d+(?:\.\d+)?)([smhd]?)(?=\s)")
_DUNE_FIRST_RUN = (
    "A DUNE side's first run compiles every form and expression it builds into a C++ module before "
    "it solves, one 'DUNE-INFO: Compiling ... (new)' line each: measured on this install, the served "
    "3-D contract's first run compiled 16 modules in 11.6 minutes with its grid and space already "
    "cached, and the next run of the same script took 2.9 s. A run ended mid-compile keeps the "
    "modules it finished for the next run and loses the one it was building. Give a first run no "
    "timeout, or one longer than that. Measured on this install.")


def _petsc_signal_finding(text: str, command: str = "") -> str:
    """'' unless the output carries PETSc's signal handler for SIGTERM (or a cut copy of its stack
    under a `timeout` command); then what ended the run, the timeout when the command has one, and,
    for a DUNE run, what its first run costs on this install."""
    if not _PETSC_SIGNAL.search(text or ""):
        return ""
    nums = re.findall(r"Caught signal number (\d+)", text)
    sig = int(nums[0]) if nums else None
    t = _TIMEOUT_CMD.search(command or "") if isinstance(command, str) else None
    if sig is not None and sig != 15:
        return ""                        # another signal: a crash, answered as one
    if sig is None and not t:
        return ""
    if sig == 15:
        head = ("the run was ENDED FROM OUTSIDE, not by an error in it: PETSc, which the solver starts, caught "
                "signal 15 (SIGTERM, 'Caught signal number 15 Terminate')"
                + (" and raised it as \"PETSc Error ... 'Signal received'\"" if "Signal received" in text else "")
                + (", then aborted -- the stack and 'dumped core' are its signal handler's"
                   if ("dumped core" in text or "PetscSignalHandlerDefault" in text) else "") + ".")
    else:
        pipe = re.search(r"\|\s*(tail|head)\b", command or "")
        head = ("PETSc's signal handler printed this: the run received a signal, and the line that names it "
                "('Caught signal number <n>') is not in this output"
                + (f" -- the `| {pipe.group(1)}` in the command cut it" if pipe else "") + ".")
    if t:
        unit = {"": " s", "s": " s", "m": " min", "h": " h", "d": " d"}[t.group(2)]
        head += (f" This command runs under `timeout {t.group(1)}{t.group(2)}`, which sends SIGTERM when "
                 f"{t.group(1)}{unit} are up.")
    dune = bool(re.search(r"/dune/|dune-py|envs/dune|\bdune\.|DUNE-INFO", (text or "") + " " + (command or "")))
    return head + " " + (_DUNE_FIRST_RUN if dune else
                         "Run it again without a timeout shorter than the run needs.")


def findings_from_output(output: str, command: str = "") -> list:
    """What a run already told you, with the call that works.

    A failed run has bought the diagnosis; spending a second run to learn it is the loop that ate
    round 49. Names at most three, because an agent acts on the first. Only the fixes of the code
    that failed are named when the traceback shows it; a needle inside one already named (the 'y'
    entry and the general 'not found in w' one) is not named twice. `command` is the command that
    ran, read only to tell which code crashed when the output cannot."""
    if not isinstance(output, str) or not output.strip():
        return []
    codes = _failing_codes(output)
    echo, text = _del_echo_finding(output)
    out, seen, named = ([echo] if echo else []), set(), []
    # A C++ BUILD IS JUDGED BY ITS FIRST ERROR. A deal.II template wall repeats header
    # names in every later error, and the entry keyed on one of them named a false cause
    # three times while the first error was a std::map keyed on Point (measured).
    _first_cxx = next((ln for ln in text.splitlines() if re.search(r":\d+:(?:\d+:)? (?:fatal )?error: ", ln)), "")
    _dealii_build = "dealii" in codes or "dealii::" in text or "/deal.II/" in text
    # A RUN ENDED FROM OUTSIDE IS NOT A CRASH (see _PETSC_SIGNAL): its abort and 'dumped core' are
    # PETSc's signal handler, so the SIGSEGV answers below do not speak for it.
    _ended = _petsc_signal_finding(text, command)
    if _ended:
        out.append(_ended)
    # A PROGRAM KILLED BY SIGSEGV IS NAMED AS A CRASH, not an install fault (measured: a cell
    # gave up 0.4 minutes after a bare failure, blaming the install).
    elif _CRASH.search(text) and (_dealii_build or re.search(r"deal\.?ii", text, re.I)):
        # THE BUILD THE WRAPPER NAMES DECIDES THE ADVICE. Measured: a cell whose DEBUG line was already
        # on was told again to turn it on, after deal.II had printed its assertion.
        # DEBUG REACHES DEAL.II'S HEADERS, NOT ITS LIBRARY. Measured on this install: a SparseMatrix
        # copy-constructed from a filled one is left empty, and the library call that uses it crashes
        # with no message with DEBUG on (the copy constructor's check is compiled into libdeal_II). The
        # old text told one cell 8 times that deal.II had "checked its own assertions" there.
        if "This build has DEBUG on" in text and "An error occurred in line" in text:
            out.append("the program was KILLED BY SIGSEGV: a crash inside it, not an install fault. DEBUG is on "
                       "in this build, which turns on the checks in deal.II's headers, and the one that failed "
                       "printed 'An error occurred in line' on the stderr above. Measured on this install.")
        elif "This build has DEBUG on" in text:
            out.append("the program was KILLED BY SIGSEGV: a crash inside it, not an install fault. DEBUG is on "
                       "in this build and no deal.II check printed: DEBUG turns on the checks in deal.II's headers, "
                       "not the ones compiled into its library, and these crash with no message either way -- a "
                       "SparseMatrix copy-constructed from a filled one (SparseMatrix<double> A(system_matrix) "
                       "leaves A empty), copy_from into a SparseMatrix never built on a pattern, and an index "
                       "vector handed unsized to get_dof_indices. A copy is built on the pattern, "
                       "SparseMatrix<double> A(sparsity), then filled with A.copy_from(system_matrix). Measured "
                       "on this install.")
        else:
            out.append("the program was KILLED BY SIGSEGV: a crash inside it, not an install fault. A Release "
                       "deal.II asserts nothing -- an FEValues accessor whose update flag was not requested, an "
                       "index vector handed unsized to get_dof_indices, or a SparseMatrix copy-constructed from a "
                       "filled one (it is left empty) crashes with no message; "
                       "target_compile_definitions(<target> PRIVATE DEBUG) after deal_ii_setup_target turns on "
                       "deal.II's header checks, which name a missing flag but not the empty matrix or the "
                       "unsized vector. Measured on this install.")
    # A KRATOS CRASH, OR ONE WHOSE CODE THE OUTPUT DOES NOT NAME, IS ANSWERED TOO (see _KRATOS_CRASH).
    elif _CRASH.search(text):
        _seen = text + "\n" + (command if isinstance(command, str) else "")
        if "kratos" in codes or re.search(r"KratosMultiphysics|\bKRATOS\b", _seen):
            out.append(_KRATOS_CRASH)
        elif not codes and not _OTHER_CODE.search(_seen):
            out.append(_UNNAMED_CRASH)
    # A DAMAGED HEAP IS NAMED AS A WRITE PAST AN ARRAY, and DEBUG as the way to find it. Measured on a
    # coupled elasticity round: three runs of one cell died on "malloc(): corrupted top size" and the like
    # and nothing answered; the cell gave up on it. Measured here: an 8 x 8 FullMatrix written at column
    # 8 and beyond aborts a Release build in the C library ("invalid fastbin entry (free)"), and the
    # same program with DEBUG stops at the write: "Index 8 is not in the half-open range [0,8)".
    _heap = _HEAP_DAMAGE.search(text)
    if _heap and (_dealii_build or re.search(r"deal\.?ii", text, re.I)):
        _said = " ".join(_heap.group(0).split())[:80]
        out.append(f"the run printed `{_said}`: the C library found the heap damaged, and it aborts there, "
                   f"at a later allocation or free -- a write past the end of an array the program owns (a "
                   f"FullMatrix, a Vector or a std::vector). A Release deal.II checks no index. Uncomment "
                   f"target_compile_definitions(<target> PRIVATE DEBUG) in CMakeLists.txt, rebuild and run "
                   f"again: deal.II then stops at the bad write and names the index, as in 'Index 8 is not in "
                   f"the half-open range [0,8)' (an 8 x 8 matrix written at column 8). Measured on this "
                   f"install.")
    # A SparseMatrix WITH NO PATTERN, NAMED BY A HEADER CHECK. With DEBUG on, a call compiled from
    # deal.II's headers on such a matrix stops with this install's words: "An error occurred in line
    # <1810> of file <.../lac/sparse_matrix.h>" and "The violated condition was: cols != nullptr"
    # (measured on a copy-constructed matrix; one cell escaped its copy loop only when this fired).
    if re.search(r"of file <[^>\n]*lac/sparse_matrix\.h>", text) and "cols != nullptr" in text:
        out.append("deal.II stopped on `cols != nullptr` in lac/sparse_matrix.h: that SparseMatrix has no "
                   "sparsity pattern. SparseMatrix<double> A(system_matrix) does not copy (deal.II leaves A "
                   "empty), and a matrix declared with no pattern has none until reinit(sparsity). A copy is "
                   "built on the pattern, SparseMatrix<double> A(sparsity), then filled with "
                   "A.copy_from(system_matrix). Measured on this install.")
    # A COMPILE THAT READ THE SYSTEM PACKAGE IS NAMED AS SUCH (measured: two cells built against
    # /usr 9.1.1 and never linked).
    if "/usr/include/deal.II/" in _first_cxx or re.search(r"deal\.II-9\.1\.1 installation found at /usr", text):
        out.append("the compile read /usr/include/deal.II: it builds against the system deal.II package, not the "
                   "tree DEAL_II_DIR should name. Keep HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR} in find_package and "
                   "deal_ii_setup_target(<target>) in CMakeLists.txt. Measured on this install.")
    if _dealii_build or re.search(r"deal\.?ii", text, re.I):
        _dfe = _dealii_first_error_finding(_first_cxx, text)
        if _dfe:
            out.append(_dfe)
    if _dealii_build and re.search(r"no match for .operator<.", _first_cxx) and "Point<" in _first_cxx:
        out.append("the build's first error is `no match for operator<` on dealii::Point -- a Point has "
                   "no ordering, so a std::map or std::set keyed on Point does not compile; key it by "
                   "the dof index (DoFTools::map_dofs_to_support_points gives index -> Point). Measured "
                   "on this install.")
    for keys, needle, fix in _ERROR_FIXES:
        # A NEEDLE MAY BE A PATTERN where the message carries a varying part (an address, a
        # vertex number); the run's own words are quoted then.
        if isinstance(needle, re.Pattern):
            _m = needle.search(text)
            if not _m:
                continue
            # a pattern that must see a frame as well quotes only its message, the group `said`
            said = " ".join((_m.groupdict().get("said") or _m.group(0)).split())
            said = said if len(said) <= 120 else said[:117] + "..."
        elif keys == ("dealii",) and _first_cxx:
            if needle not in _first_cxx:
                continue                 # a C++ build: only its first error decides
            said = needle
        elif needle not in text:
            continue
        else:
            said = needle
        if any(said in n for n in named):
            continue
        if keys and codes and not codes.intersection(keys):
            continue                     # another code's fix: the failing script does not use it
        if not codes and said in _NEEDS_THE_CODE:
            continue                     # every code prints it, and the code is not in sight
        if fix in seen:
            named.append(said)           # one cause, already named: it still covers the needles inside it
            continue
        seen.add(fix)
        named.append(said)
        out.append(f"the run printed `{said}` -- {fix}. Measured on this install.")
        if len(out) >= 3:
            break
    return out


# ── deal.II builds: the FIRST error of a measured misuse, and the call that works ──────────────
# Measured on a transient coupled round: 13 failed deal.II builds of two cells, and no run check
# answered any of them. Each entry below was compiled on this install (deal.II 9.8.0-pre, g++,
# the cells' language) as one small program per wrong call, and is keyed to the FIRST error that
# program printed: all its words must be in that line (quotes read as '), or, for a link error,
# in the output when no compile error came first.
_DEALII_HEADERS = {
    "deal.II/fe/fe_face_values.h": "FEFaceValues is declared in deal.II/fe/fe_values.h, beside FEValues",
    "deal.II/lac/sparse_direct_umfpack.h": "SparseDirectUMFPACK is declared in deal.II/lac/sparse_direct.h",
    "deal.II/grid/boundary_descriptor.h": ("boundary ids need no header of their own: set_boundary_id(id) is a "
                                           "member of a face iterator, declared with the triangulation in "
                                           "deal.II/grid/tria.h"),
    "deal.II/numerics/affine_constraints.h": "AffineConstraints is declared in deal.II/lac/affine_constraints.h",
    "deal.II/base/affine_constraints.h": "AffineConstraints is declared in deal.II/lac/affine_constraints.h",
    "deal.II/base/vector_tools.h": "VectorTools is declared in deal.II/numerics/vector_tools.h",
    "deal.II/lac/vector_tools.h": "VectorTools is declared in deal.II/numerics/vector_tools.h",
    "deal.II/base/deallog.h": "deallog is declared in deal.II/base/logstream.h",
    "deal.II/lac/precondition_relaxation.h": "PreconditionSSOR is declared in deal.II/lac/precondition.h",
    "deal.II/numerics/integrate.h": ("there is no such header: QGauss and the other quadrature rules are declared in "
                                     "deal.II/base/quadrature_lib.h, and an integral is a sum over the quadrature "
                                     "points of an FEValues, weighted by JxW(q)"),
}
_DISTRIBUTE = ("AffineConstraints::distribute takes the solution vector, after the solve: "
               "constraints.distribute(solution). Given a SparseMatrix it compiles and fails to LINK "
               "('undefined reference to ... distribute<dealii::SparseMatrix<double> >'), and given a matrix "
               "and a vector, or two vectors, it does not compile. A matrix and its right-hand side take the "
               "constraints through condense(matrix, rhs) before the solve, or through "
               "distribute_local_to_global while they are assembled")
_CONSTRAINT_MEMBERS = ("AffineConstraints<double> is filled by add_line(i) and set_inhomogeneity(i, v), or by "
                       "add_constraint(i, {}, v), and closed by close(); it has no initialize, attach_dof_handler "
                       "or add_entry, and add_lines takes dof numbers (a std::set), not a map of values")
_FFV = ("FEFaceValues holds no dof numbers: the dofs of the face's cell are cell->get_dof_indices(indices), "
        "indices a std::vector<types::global_dof_index> of fe.n_dofs_per_cell() entries, and shape_value(i, q) "
        "runs over all of that cell's dofs, i from 0 to fe.n_dofs_per_cell() - 1 (for FE_Q it is zero for a dof "
        "off that face)")
_DEALII_FIRST_ERRORS = (
    (("undefined reference to", "AffineConstraints<double>::distribute<dealii::SparseMatrix"), _DISTRIBUTE),
    (("no matching function", "AffineConstraints<double>::distribute(dealii::SparseMatrix"), _DISTRIBUTE),
    (("no matching function", "AffineConstraints<double>::distribute(dealii::Vector<double>&, dealii::Vector"),
     _DISTRIBUTE),
    (("no matching function", "map_dofs_to_support_points(dealii::DoFHandler"),
     "DoFTools::map_dofs_to_support_points takes the mapping first: (mapping, dof_handler, points), with "
     "MappingQ<2> mapping(1) for straight-edged cells and points a std::vector<Point<2>> of n_dofs entries"),
    (("no matching function", "map_dofs_to_support_points(dealii::FEValuesExtractors"),
     "DoFTools::map_dofs_to_support_points takes the mapping first: (mapping, dof_handler, points), with "
     "MappingQ<2> mapping(1) for straight-edged cells and points a std::vector<Point<2>> of n_dofs entries"),
    (("no matching function", "::MappingQ(dealii::FE_Q"),
     "MappingQ<2> takes the polynomial degree of the geometry, not the finite element: MappingQ<2> mapping(1) "
     "for straight-edged cells"),
    (("FEFaceValues<", "has no member named 'dof_index'"), _FFV),
    (("FEFaceValues<", "has no member named 'face_dof_index'"), _FFV),
    (("FEFaceValues<", "has no member named 'n_dofs_per_face'"), _FFV),
    (("FEFaceValues<", "has no member named 'local_dof_index'"), _FFV),
    (("'Face' does not name a type",),
     "there is no Face class to declare: cell->face(f) is an iterator to the face, so keep it as "
     "const auto face = cell->face(f) and call its members with ->; the face numbers of a cell are "
     "cell->face_indices()"),
    (("SparseMatrix<", "has no member named 'initialize'"),
     "a SparseMatrix is sized by reinit(sparsity_pattern), or built on the pattern, SparseMatrix<double> "
     "A(sparsity_pattern); the pattern has to outlive the matrix"),
    (("DoFHandler<", "has no member named 'initialize'"),
     "a DoFHandler is built on the triangulation, DoFHandler<2> dof_handler(triangulation), or given it by "
     "reinit(triangulation), and then numbers its dofs with distribute_dofs(fe)"),
    (("Tensor<", "has no member named 'copy_into'"), "a Tensor copies by assignment: g = t"),
    (("::Tensor(double, double)",),
     "a Tensor<1, 2> has no constructor from two numbers: fill it entry by entry, t[0] = x and t[1] = y, or "
     "build a Point<2>(x, y), which is a Tensor<1, 2>"),
    (("base operand of '->' has non-pointer type", "DoFAccessor"),
     "cell->face(f) is already the iterator: call its members through it, cell->face(f)->center(); "
     "*cell->face(f) is the accessor itself, whose members take a dot"),
    (("no match for 'operator='", "std::vector<", "and 'int')"),
     "a std::vector has no = 0: a deal.II Vector<double> sets every entry so (Vector<double> v(n); v = 0), "
     "and a std::vector is filled with std::fill(v.begin(), v.end(), 0.0)"),
    (("first template argument is a class derived from 'EnableObserverPointer'",),
     "that static assertion is what a preconditioner declared with the number type prints on this install "
     "(PreconditionSSOR<double>, measured): it takes the MATRIX type, PreconditionSSOR<SparseMatrix<double>>, "
     "built empty and set up by initialize(matrix, relaxation)"),
    (("was not declared in this scope; did you mean 'dealii::",),
     "deal.II's names live in the namespace dealii: put using namespace dealii; at file scope after the "
     "#include lines, or write dealii:: before each name"),
    # A TWO-COMPONENT FIELD. Measured on a coupled elasticity round: one cell's deal.II side died on
    # every one of these in twelve builds, and none drew an answer. Each was compiled here the same way.
    (("use of deleted function", "FESystem<dim, spacedim>::FESystem()"),
     "an FESystem has no default constructor: a class member FESystem is built in the constructor's member "
     "initializer list, fe(FE_Q<2>(1), 2), and a local one where it is declared, FESystem<2> fe(FE_Q<2>(1), 2)"),
    (("use of deleted function", "FESystem<2>::operator="),
     "an FESystem cannot be assigned: build it once where it is declared (a class member in the initializer "
     "list, fe(FE_Q<2>(1), 2)), or hold it through a std::unique_ptr<FESystem<2>> made by "
     "std::make_unique<FESystem<2>>(FE_Q<2>(1), 2)"),
    (("no matching function", "AffineConstraints<double>::reinit("),
     "AffineConstraints is not sized by a dof count: clear() empties it, then add_line and set_inhomogeneity "
     "(or interpolate_boundary_values into it) fill it, and close() ends the filling"),
    (("no matching function", "ComponentMask::ComponentMask(int)"),
     "a ComponentMask is built with its size and a value, ComponentMask(2, true), or taken from the element, "
     "fe.component_mask(FEValuesExtractors::Scalar(c)), and an entry changes through set(c, value)"),
    (("FEValues<", "has no member named 'gradient'"),
     "FEValues has no gradient member: a two-component FESystem's shape function i is read through an "
     "extractor, fe_values[FEValuesExtractors::Vector(0)].gradient(i, q) (a Tensor<2, 2>), .symmetric_gradient(i, "
     "q) (a SymmetricTensor<2, 2>) and .divergence(i, q), and one component's gradient is "
     "shape_grad_component(i, q, c)"),
    # Measured on a coupled elastic round: a cell built eps from a free symmetric_gradient(...) and stopped
    # here, then from scalar shape gradients, which the served ASSEMBLY stop refused five times.
    (("'symmetric_gradient' was not declared",),
     "deal.II has no free symmetric_gradient: the strain of shape function i of a two-component FESystem is "
     "fe_values[FEValuesExtractors::Vector(0)].symmetric_gradient(i, q), a SymmetricTensor<2, 2> (the FEValues "
     "made with update_gradients)"),
    (("FEValues<", "has no member named 'distribute_local_to_global'"),
     "distribute_local_to_global is a member of AffineConstraints, constraints.distribute_local_to_global("
     "cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs), and it empties the constrained rows a "
     "consistent recovery reads; system_matrix.add(local_dof_indices, cell_matrix) adds a cell matrix with no "
     "constraint"),
    (("Values<", "has no member named 'value'"),
     "FEValues and FEFaceValues have no value member: shape function i at point q is shape_value(i, q), and for "
     "a two-component FESystem fe_face_values[FEValuesExtractors::Vector(0)].value(i, q) is its Tensor<1, 2> and "
     "shape_value_component(i, q, c) one component"),
    (("has no member named 'at_vertex'",),
     "a cell's corner is cell->vertex(v), a Point<2>, v from 0 to 3 on a quadrilateral; its number in the "
     "triangulation is cell->vertex_index(v)"),
    (("'PointComparator' was not declared",),
     "deal.II has no PointComparator, and a Point has no ordering: keep the points in a std::vector indexed by "
     "dof (DoFTools::map_dofs_to_support_points), or order them with a lambda that compares the coordinates"),
    (("Triangulation<", "has no member named 'point'"),
     "a Triangulation's vertices are tria.get_vertices(), a std::vector<Point<2>> of tria.n_vertices() entries, "
     "and a cell's are cell->vertex(v)"),
    (("no matching function", "set_inhomogeneity(std::map"),
     "AffineConstraints takes one inhomogeneity per line: for each (dof, value) of the map, add_line(dof) then "
     "set_inhomogeneity(dof, value); interpolate_boundary_values(mapping, dof_handler, id, function, "
     "constraints) fills them itself"),
    (("'MappingKind'", "does not name a type"),
     "DoFTools::map_dofs_to_support_points takes a mapping object first: (MappingQ<2>(1), dof_handler, points), "
     "points a std::vector<Point<2>> of n_dofs entries"),
    (("'add_dof_values' is not a member of",),
     "VectorTools has no add_dof_values: a cell vector goes into the global one entry by entry, "
     "system_rhs(local_dof_indices[i]) += cell_rhs(i), or through constraints.distribute_local_to_global("
     "cell_rhs, local_dof_indices, system_rhs)"),
    (("has no member named 'distribute_zero'",),
     "AffineConstraints has no distribute_zero: set_zero(vector) zeroes the constrained entries, and "
     "distribute(vector) sets them from the constraints after a solve"),
    (("no matching function", "apply_boundary_values(dealii::AffineConstraints"),
     "MatrixTools::apply_boundary_values takes a std::map<types::global_dof_index, double> of held values, "
     "(boundary_values, matrix, solution, rhs), never AffineConstraints, and it empties the held rows of the "
     "matrix it gets: hand it a copy of the matrix a consistent recovery reads"),
    (("no matching function", "interpolate_boundary_values(", "brace-enclosed initializer list"),
     "interpolate_boundary_values takes one boundary id, or a std::map<types::boundary_id, const Function<2> *> "
     "for several ({{1, &f}, {2, &f}}), then the std::map of values or the AffineConstraints; a braced list of "
     "ids does not convert"),
    # MEASURED ON A TRANSIENT COUPLED ROUND: the first errors of 9 failed deal.II builds of three cells drew
    # no answer here, and the other forms below sat behind another first error in the same builds. Each
    # wrong form was compiled on this install, and each answer's calls compiled, linked and ran.
    (("AffineConstraints<double>'", "has no member named"), _CONSTRAINT_MEMBERS),
    (("no matching function", "AffineConstraints<double>::add_entry("), _CONSTRAINT_MEMBERS),
    (("no matching function", "AffineConstraints<double>::add_lines(std::map"), _CONSTRAINT_MEMBERS),
    (("no match for 'operator='", "std::vector<", "and 'double')"),
     "a std::vector has no = 0.0: a deal.II Vector<double> sets every entry so (Vector<double> v(n); v = 0), "
     "and a std::vector is filled with std::fill(v.begin(), v.end(), 0.0)"),
    (("no match for call to '(std::vector<",),
     "a std::vector is indexed with v[i]: v(i) calls it, and a std::vector cannot be called; a deal.II "
     "Vector<double> takes v(i) and v[i] alike"),
    (("no matching function", "get_function_gradients(", "std::vector<double>&)"),
     "get_function_gradients(vector, gradients) fills gradients, a std::vector<Tensor<1, 2>> of "
     "n_quadrature_points entries (it needs update_gradients); a std::vector<double> takes get_function_values"),
    (("SparseMatrix<", "has no member named 'sparsity'"),
     "a SparseMatrix's pattern is get_sparsity_pattern(), a const SparsityPattern&: a second matrix on the same "
     "pattern is SparseMatrix<double> B(sparsity) with that pattern"),
    (("no matching function", "make_sparsity_pattern(", "AffineConstraints<double>&)"),
     "DoFTools::make_sparsity_pattern takes the pattern it fills second: (dof_handler, dsp) with "
     "DynamicSparsityPattern dsp(dof_handler.n_dofs()), or (dof_handler, dsp, constraints, keep_constrained_dofs) "
     "with the constraints third"),
    (("'SparseDirectLU' was not declared",),
     "deal.II has no SparseDirectLU: its direct solver for a SparseMatrix<double> is SparseDirectUMFPACK "
     "(deal.II/lac/sparse_direct.h), set up by initialize(matrix), after which vmult(x, b) writes the solution of "
     "matrix x = b into x"),
    (("Tensor<2, 2, double>' has no member named 'vmult'",),
     "a Tensor<2, 2> applies to a Tensor<1, 2> as K * g (a Tensor<1, 2>), and (K * g) * h is a number"),
    (("undefined reference to", "SparseDirectUMFPACK::solve<dealii::Vector<double> >"),
     "after initialize(matrix), SparseDirectUMFPACK solves with vmult(x, b), or in place with solve(b), b then "
     "holding the solution; solve(b, x) picks the overload whose first argument is a matrix, which is built for "
     "no vector and so fails to link"),
)


def _dealii_first_error_finding(first: str, text: str) -> str:
    """The answer to a deal.II build's first error, when it is one measured here; '' otherwise."""
    q = lambda s: s.replace("\u2018", "'").replace("\u2019", "'")      # noqa: E731
    line = q(first)
    m = re.search(r"fatal error: (deal\.II/[\w/.]+\.h): No such file or directory", line)
    if m:
        h = m.group(1)
        answer = _DEALII_HEADERS.get(h, "include the header that declares the class you need; the deal.II "
                                        "headers are the files under include/deal.II/ of the tree the build uses")
        return (f"the build's first error is that {h} does not exist in this deal.II: {answer}. "
                f"Measured on this install.")
    where = line if line else q(text)
    for words, answer in _DEALII_FIRST_ERRORS:
        if all(w in where for w in words):
            shown = " ".join(first.split())[:160] if first else "a link error"
            return f"the build's first error, `{shown}`: {answer}. Measured on this install."
    return ""


# ── deal.II C++ and its CMakeLists, judged when written (measured on deal.II 9.8.0-pre Release) ──
_FEV_NEEDS = (("quadrature_point", "update_quadrature_points"), ("get_quadrature_points", "update_quadrature_points"),
              ("shape_value", "update_values"), ("get_function_values", "update_values"),
              ("shape_grad", "update_gradients"), ("get_function_gradients", "update_gradients"),
              ("JxW", "update_JxW_values"), ("get_JxW_values", "update_JxW_values"),
              ("normal_vector", "update_normal_vectors"))


@functools.lru_cache(maxsize=1)
def _dealii_include_dir():
    """The include directory of the deal.II tree discover names, when it holds deal.II's headers;
    None otherwise (then no header is judged)."""
    try:
        from backends.dealii.backend import _find_dealii          # noqa: PLC0415
        tree = _find_dealii()
    except Exception:                                             # noqa: BLE001
        return None
    inc = Path(tree) / "include" if tree else None
    return inc if inc is not None and (inc / "deal.II" / "base" / "function.h").is_file() else None


def cxx_findings(text: str, name: str = "") -> list:
    """What a deal.II program or its CMakeLists will do wrong, read when it is written.

    MEASURED ON THIS INSTALL: a Release deal.II asserts nothing, so an FEValues accessor
    whose update flag was not requested, and an index vector handed unsized to
    get_dof_indices, end the program with SIGSEGV and no message (two of five cells of one
    round crashed so, one gave up on it); a find_package without HINTS builds against the
    system package (deal.II 9.1.1 at /usr) whatever -DDEAL_II_DIR says, and linking
    dealii::dealii without deal_ii_setup_target compiles against /usr/include and fails on
    mpi.h."""
    out: list = []
    if not isinstance(text, str):
        return out
    if name.lower() == "cmakelists.txt" or re.search(r"find_package\s*\(\s*deal\.II", text):
        fp = re.search(r"find_package\s*\(\s*deal\.II[^)]*\)", text)
        if fp and "HINTS" not in fp.group(0):
            out.append("`" + fp.group(0) + "` has no HINTS: it finds the system package (deal.II 9.1.1 at /usr "
                       "on this install) whatever -DDEAL_II_DIR says (measured). Write find_package(deal.II 9.0 "
                       "REQUIRED HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR}).")
        if re.search(r"dealii::dealii", text) and "deal_ii_setup_target" not in text:
            out.append("this CMakeLists links dealii::dealii without deal_ii_setup_target(<target>): the compile "
                       "then reads /usr/include/deal.II and stops on mpi.h (measured). Add "
                       "deal_ii_setup_target(<target>) after add_executable.")
        return out
    if "#include <deal.II/" not in text and "dealii::" not in text:
        return out
    body = re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)
    # A HEADER THE TREE DOES NOT HAVE stops the compile at its first line (measured: eight builds of two
    # cells). Judged against the include directory of the tree discover names, and only when that
    # directory holds deal.II's headers; otherwise nothing is said.
    _inc = _dealii_include_dir()
    for h in dict.fromkeys(re.findall(r"^\s*#\s*include\s*<(deal\.II/[\w/.]+\.h)>", body, re.M)):
        if _inc is not None and not (_inc / h).is_file():
            out.append(f"#include <{h}>: this deal.II has no such header (its include directory lacks it), and the "
                       f"compile stops there with 'No such file or directory' (measured). "
                       + _DEALII_HEADERS.get(h, "Include the header that declares the class you need") + ".")
    # distribute() HANDED A SparseMatrix compiles and fails only at the LINK (measured: the last error
    # before two cells gave up).
    _matrices = set(re.findall(r"\bSparseMatrix\s*<[^;>]*>\s+(\w+)\s*[;(]", body))
    for m in re.finditer(r"\.distribute\s*\(\s*(\w+)\s*[,)]", body):
        if m.group(1) in _matrices:
            out.append(f"`.distribute({m.group(1)}` hands a SparseMatrix to AffineConstraints::distribute, which "
                       f"takes the solution vector after the solve: this compiles and fails to link with "
                       f"'undefined reference to ... distribute<dealii::SparseMatrix<double> >' (measured). "
                       f"A matrix and its right-hand side take the constraints through condense(matrix, rhs) "
                       f"before the solve.")
            break
    _copy = _sparse_matrix_copy_finding(body)
    if _copy:
        out.append(_copy)
    flag_vars = {m.group(1): set(re.findall(r"update_\w+", m.group(2)))
                 for m in re.finditer(r"UpdateFlags\s+(\w+)\s*=\s*([^;]+);", body)}
    for m in re.finditer(r"FE(?:Face|Subface)?Values\s*<[^>]*>\s+(\w+)\s*\((.*?)\)\s*;", body, re.S):
        obj, args = m.group(1), m.group(2)
        flags = set(re.findall(r"update_\w+", args))
        if not flags:
            named = [v for v in re.findall(r"\b([A-Za-z_]\w*)\b", args) if v in flag_vars]
            if not named:
                continue                                   # flags not readable here: nothing judged
            for v in named:
                flags |= flag_vars[v]
        missing = sorted({flag for acc, flag in _FEV_NEEDS
                          if re.search(rf"\b{re.escape(obj)}\s*\.\s*{acc}\s*\(", body) and flag not in flags})
        if missing:
            out.append(f"`{obj}` is built without {', '.join(missing)} but its loop reads what "
                       f"{'that flag' if len(missing) == 1 else 'those flags'} provide{'s' if len(missing) == 1 else ''}: "
                       f"on a Release deal.II the program then ends with SIGSEGV and no message (measured). Add "
                       f"{' | '.join(missing)} to its flags.")
    for m in re.finditer(r"std::vector\s*<\s*(?:dealii::)?(?:types::global_dof_index|unsigned int)\s*>\s+(\w+)\s*;", body):
        v = m.group(1)
        if (re.search(rf"get_dof_indices\s*\(\s*{re.escape(v)}\s*\)", body)
                and not re.search(rf"\b{re.escape(v)}\s*\.\s*(?:resize|assign)\s*\(", body)):
            out.append(f"`{v}` is declared empty and handed to get_dof_indices, which writes into it without "
                       f"sizing it: SIGSEGV, in Release and in Debug alike (measured). Declare it "
                       f"std::vector<types::global_dof_index> {v}(fe.n_dofs_per_cell()).")
    return out


# A SparseMatrix THAT IS NOT A COPY. Measured on this install (deal.II 9.8.0-pre, its library built
# Release), one small program per spelling, with DEBUG defined in the program and without:
# SparseMatrix<double> A(system_matrix) leaves A empty (A.empty() is 1), since the check that refuses a
# filled argument is compiled into the library, which DEBUG in the program does not reach; the first
# library call that uses A (apply_boundary_values, condense, SparseDirectUMFPACK::initialize) then ends
# the program with SIGSEGV and no message, with DEBUG on as with it off. copy_from into a matrix never
# built on a pattern does the same, and A = system_matrix leaves A as it was (all zero on its pattern).
# On a coupled round one cell crashed 13 times so (8 with DEBUG on) and gave up; three others lost
# minutes to it.
_SPARSE_COPY = ("build the copy on the pattern, SparseMatrix<double> {a}(sparsity) (or give it "
                "{a}.reinit(sparsity)), then fill it with {a}.copy_from({b})")


def _sparse_matrix_copy_finding(body: str) -> str:
    """The first SparseMatrix of `body` (a deal.II program, comments cut) that is meant as a copy and
    is not one; '' when there is none."""
    decl, said = {}, {}                        # name -> its initializer ('' when none), as written
    for m in re.finditer(r"\bSparseMatrix\s*<[^;<>]*>\s+(\w+)\s*(?:\(([^;()]*)\)|\{([^;{}]*)\}|=\s*([^;]+))?\s*;",
                         body):
        decl.setdefault(m.group(1), (m.group(2) or m.group(3) or m.group(4) or "").strip())
        said.setdefault(m.group(1), " ".join(m.group(0).split()))
    how = lambda a, b: _SPARSE_COPY.format(a=a, b=b)                           # noqa: E731
    silent = ("the library call that uses it (apply_boundary_values, condense, a direct solver's "
              "initialize) ends the program with SIGSEGV and no message, with DEBUG on or off (measured: "
              "the check is compiled into the library, which DEBUG in this file does not reach)")
    for a, init in decl.items():
        if init in decl and init != a:
            return (f"`{said[a]}` does not copy {init}: deal.II leaves {a} empty, and {silent}. To copy, "
                    f"{how(a, init)}.")
    for m in re.finditer(r"(?<![\w.>])(\w+)\s*\.\s*copy_from\s*\(\s*(\w+)\s*\)", body):
        a, b = m.group(1), m.group(2)
        if b in decl and decl.get(a) == "" and not re.search(rf"\b{re.escape(a)}\s*\.\s*reinit\s*\(", body):
            return (f"`{a}.copy_from({b})` fills a SparseMatrix declared with no sparsity pattern (`{said[a]}`) "
                    f"and never given one: it ends the program with SIGSEGV and no message, with DEBUG on or "
                    f"off (measured). To copy, {how(a, b)}.")
    for m in re.finditer(r"(?<![\w.>:])(\w+)\s*=\s*(\w+)\s*;", body):
        a, b = m.group(1), m.group(2)
        if a in decl and b in decl and a != b:
            return (f"`{a} = {b}` does not copy a SparseMatrix: deal.II leaves {a} as it was, with no message, "
                    f"with DEBUG on or off (measured: all zero on its pattern). To copy, {how(a, b)}.")
    return ""


def _hole_name_findings(text: str) -> list:
    """Undefined names, with the ones the contract's hole must define folded into one note."""
    # A SERVED TEXT THAT STILL CARRIES ITS HOLE MARKERS IS NOT JUDGED FOR NAMES; the
    # marker alone does not make a text served. Judged by the marker only, one pasted
    # comment line switched this check off for a hand-written side.
    if _SERVED_MARK in text and _is_served_contract(text):
        return []
    out = []
    _und = undefined_names(text)
    _hole = set()
    _m = re.search(r"WHAT YOUR SOLVE MUST LEAVE BEHIND[^\n]*\n((?:#[^\n]*\n)+)", text)
    if _m:
        _hole = set(re.findall(r"^#\s{3,}([A-Za-z_]\w*)\s*$", _m.group(1), re.M))
    _left = [f for f in _und if re.match(r"`([A-Za-z_]\w*)` is used", f)
             and re.match(r"`([A-Za-z_]\w*)`", f).group(1) in _hole]
    out += [f for f in _und if f not in _left]
    _names = sorted({re.match(r"`([A-Za-z_]\w*)`", f).group(1) for f in _left})
    if _names:
        out.append("your solve has not yet defined " + ", ".join(f"`{n}`" for n in _names)
                   + " from the list this file ends with (WHAT YOUR SOLVE MUST LEAVE "
                     "BEHIND); the run stops at the first line that uses one.")
    # A NAME ON THAT LIST MUST ALSO STILL MEAN WHAT THE LINES AFTER THE HOLE NEED. Measured on
    # a coupled round: two of five NGSolve fills defined u, v = fes.TnT() and then rebound v
    # to a mesh vertex -- `for v in mesh.vertices:`, and `v = mesh.vertices[vn]` in a loop --
    # and the served Neumann line stopped with a TypeError whose last line was "Invoked with:
    # <GridFunction>, V79"; nothing named the cause.
    if _hole and not any(f.startswith("this file does not parse") for f in _und):
        out += _rebound_hole_names(text, _hole)
    return out


# How a binding gives a name its value: a loop's variable, a call, an item picked out of a
# collection, another name, or anything else.
_BIND_FOR, _BIND_CALL, _BIND_ITEM, _BIND_ALIAS, _BIND_OTHER = "for", "call", "item", "alias", "other"


def _bind_kind(value) -> str:
    import ast
    if isinstance(value, ast.Call):
        return _BIND_CALL
    if isinstance(value, ast.Subscript):
        return _BIND_ITEM
    if isinstance(value, ast.Name):
        return _BIND_ALIAS
    return _BIND_OTHER


def _module_events(tree, names: set) -> list:
    """The module-level bindings and reads of `names`, in source order: (kind, name, node,
    enclosing loops, the name an alias copies, the targets of the enclosing loops).

    Function and class bodies are their own scope and are left out; so is a comprehension's
    own variable, and a lambda's body (it runs later)."""
    import ast
    ev: list = []

    def reads(expr, loops, shadow=frozenset()):
        if expr is None:
            return
        if isinstance(expr, ast.Lambda):
            return
        if isinstance(expr, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            shadow = shadow | {n.id for g in expr.generators for n in ast.walk(g.target)
                               if isinstance(n, ast.Name)}
        if isinstance(expr, ast.Name) and isinstance(expr.ctx, ast.Load) and expr.id in names \
                and expr.id not in shadow:
            ev.append(("load", expr.id, expr, loops, None))
        for ch in ast.iter_child_nodes(expr):
            reads(ch, loops, shadow)

    def binds(target, kind, loops, stmt, value=None):
        if isinstance(target, ast.Name):
            if target.id in names:
                alias = value.id if kind == _BIND_ALIAS and isinstance(value, ast.Name) else None
                ev.append((kind, target.id, stmt, loops, alias))
        elif isinstance(target, (ast.Tuple, ast.List)):
            for t in target.elts:
                binds(t, kind, loops, stmt, None if kind == _BIND_ALIAS else value)
        elif isinstance(target, ast.Starred):
            binds(target.value, kind, loops, stmt)
        else:                                   # gfu.vec[...] = ... reads gfu
            reads(target, loops)

    def walk(stmts, loops):
        for st in stmts:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for d in getattr(st, "decorator_list", []):
                    reads(d, loops)
                if st.name in names:
                    ev.append((_BIND_OTHER, st.name, st, loops, None))
                continue
            if isinstance(st, (ast.For, ast.AsyncFor)):
                reads(st.iter, loops)
                binds(st.target, _BIND_FOR, loops + (st,), st)
                walk(st.body, loops + (st,))
                walk(st.orelse, loops)
            elif isinstance(st, ast.While):
                reads(st.test, loops)
                walk(st.body, loops + (st,))
                walk(st.orelse, loops)
            elif isinstance(st, ast.If):
                reads(st.test, loops)
                walk(st.body, loops)
                walk(st.orelse, loops)
            elif isinstance(st, (ast.With, ast.AsyncWith)):
                for it in st.items:
                    reads(it.context_expr, loops)
                    if it.optional_vars is not None:
                        binds(it.optional_vars, _BIND_OTHER, loops, st)
                walk(st.body, loops)
            elif isinstance(st, ast.Try) or type(st).__name__ == "TryStar":
                walk(st.body, loops)
                for h in st.handlers:
                    reads(h.type, loops)
                    if h.name and h.name in names:
                        ev.append((_BIND_OTHER, h.name, h, loops, None))
                    walk(h.body, loops)
                walk(st.orelse, loops)
                walk(st.finalbody, loops)
            elif isinstance(st, ast.Assign):
                reads(st.value, loops)
                for t in st.targets:
                    binds(t, _bind_kind(st.value), loops, st, st.value)
            elif isinstance(st, ast.AnnAssign):
                reads(st.value, loops)
                if st.value is not None:
                    binds(st.target, _bind_kind(st.value), loops, st, st.value)
            elif isinstance(st, ast.AugAssign):      # a += ... updates a: a read, not a new value
                reads(st.value, loops)
                if isinstance(st.target, ast.Name):
                    reads(ast.Name(id=st.target.id, ctx=ast.Load(), lineno=st.lineno,
                                   col_offset=st.col_offset), loops)
                else:
                    reads(st.target, loops)
            elif isinstance(st, (ast.Import, ast.ImportFrom)):
                for al in st.names:
                    nm = (al.asname or al.name).split(".")[0]
                    if nm in names:
                        ev.append((_BIND_OTHER, nm, st, loops, None))
            else:
                for ch in ast.iter_child_nodes(st):
                    reads(ch, loops)

    walk(tree.body, ())
    return ev


@functools.lru_cache(maxsize=64)
def _contract_bind_kinds(template: str) -> dict:
    """{name: how the shipped contract, holes filled, binds it at module level}."""
    import ast
    try:
        tree = ast.parse(template)
    except SyntaxError:
        return {}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    out: dict = {}
    for kind, name, _node, _loops, _alias in _module_events(tree, names):
        if kind != "load":
            out.setdefault(name, set()).add(kind)
    return {k: frozenset(v) for k, v in out.items()}


@functools.lru_cache(maxsize=1)
def _template_line_sets() -> tuple:
    """(contract text, its served lines of 40 characters or more, holes cut) per shipped contract."""
    out = []
    for tpl in _served_templates():
        lines = frozenset(ln.strip() for ln in _HOLE_RE.sub("", tpl).splitlines() if len(ln.strip()) >= 40)
        if len(lines) >= 20:
            out.append((tpl, lines))
    return tuple(out)


def _contract_of(text: str):
    """The shipped contract (holes filled) this file was served from: the one whose served lines
    it carries most, if it carries at least half of them; None otherwise."""
    have = {ln.strip() for ln in text.splitlines()}
    best, share = None, 0.0
    for tpl, lines in _template_line_sets():
        s = len(lines & have) / len(lines)
        if s > share:
            best, share = tpl, s
    return best if share >= 0.5 else None


def _rebound_hole_names(text: str, hole: set) -> list:
    """A name on the list the contract ends with, rebound in the file before a line that reads it.

    A loop over a collection whose variable carries such a name leaves the name holding the
    loop's last item once the loop ends (not judged for a name the contract itself loops over).
    Inside a loop, an item picked out of a collection or a copy of the loop's variable does the
    same to a name the shipped contract makes with a call only (the test and trial functions,
    the space, the grid function, the forms); names the contract fills from arrays are not
    judged that way. Only a read after the rebinding, outside the loop that rebinds it and with
    no new binding between, is named."""
    import ast
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    tpl = _contract_of(text)
    served_kinds = _contract_bind_kinds(tpl) if tpl else {}
    lines = text.splitlines()

    def _src(node) -> str:
        ln = getattr(node, "lineno", 0)
        s = re.sub(r"\s+#[^\n]*$", "", lines[ln - 1].strip()) if 0 < ln <= len(lines) else ""
        return s if len(s) <= 90 else s[:87] + "..."

    out = []
    events = _module_events(tree, set(hole))
    for name in sorted(hole):
        own = [e for e in events if e[1] == name]
        made_by_call = served_kinds.get(name, frozenset()) == frozenset({_BIND_CALL})
        bad, good = None, None
        for kind, _nm, node, loops, alias in own:
            if kind == "load":
                if bad is None or bad[2][-1:] and bad[2][-1] in loops:
                    continue                 # no rebinding in force, or a read inside its own loop
                what, stmt, _l = bad
                how = ("after a Python loop the name holds the loop's last item"
                       if what == _BIND_FOR else
                       "it then holds one item of that collection" if what == _BIND_ITEM else
                       "it then holds that loop's item")
                out.append(
                    f"`{name}` is on the list this file ends with (WHAT YOUR SOLVE MUST LEAVE "
                    f"BEHIND): the lines after your hole need it"
                    + (f" as line {good.lineno} made it (`{_src(good)}`)" if good is not None else "")
                    + f". Line {stmt.lineno} (`{_src(stmt)}`) rebinds it, and {how}, so line "
                    f"{node.lineno} (`{_src(node)}`) reads the wrong thing. Give "
                    f"{'the loop variable' if what == _BIND_FOR else 'that value'} a name of its "
                    f"own. Measured: a fill that named a loop over mesh.vertices `v` stopped the "
                    f"served NGSolve Neumann line with TypeError ... Invoked with: <GridFunction>, "
                    f"V<n>.")
                break
            loop_item = alias is not None and any(
                isinstance(lp, (ast.For, ast.AsyncFor)) and any(
                    isinstance(t, ast.Name) and t.id == alias for t in ast.walk(lp.target))
                for lp in loops)
            # THE CONTRACT'S OWN SHAPES ARE NOT A REBINDING. Measured over the 975 recorded files
            # that carry the list: a name the contract itself loops over (`for d, q in zip(...)`)
            # and an index array picked once out of np.where(...) fired, and neither is a defect.
            if kind == _BIND_FOR and _BIND_FOR not in served_kinds.get(name, frozenset()):
                bad = (_BIND_FOR, node, (node,))
            elif made_by_call and loops and (kind == _BIND_ITEM or loop_item):
                bad = (_BIND_ITEM if kind == _BIND_ITEM else _BIND_ALIAS, node, loops)
            else:
                bad = None
                good = node if kind == _BIND_CALL else good
    return out


def participant_findings(text: str) -> list:
    """Measured API traps present in this script, named with the error and the working call."""
    if not isinstance(text, str) or not text.strip():
        return []
    codes = backends_in(text)
    if not codes:
        # A PARTICIPANT THAT DOES NOT PARSE IS NAMED EVEN WHEN NO SOLVER IMPORT
        # IS RECOGNISED. The API traps below are per backend, so they need one;
        # a SyntaxError does not, and returning early here made the parse
        # finding unreachable for exactly the files most likely to lack a
        # recognisable import -- truncated or garbled ones. Measured over the
        # recorded runs: 144 of 3873 participant scripts do not parse, and 47
        # of those were silent here; 34 are the same garble,
        # `(a)**2 + **(b)2`.
        # A WRAPPER AROUND A COMPILED SOLVER (deal.II) IMPORTS NO SOLVER MODULE, and its
        # unfilled hole was never named here (measured): its names are judged as any
        # participant's are, below the parse check.
        # ANY SCRIPT THAT DOES NOT PARSE IS NAMED, participant or not. Measured: writes cut short
        # (a tool call cut at a literal it carried) left deck-writing scripts that could not run,
        # and nothing was said because they named no exports.json yet.
        _parse = [f for f in undefined_names(text) if f.startswith("this file does not parse")]
        if _parse:
            return _parse
        if looks_like_participant(text):
            # only a copy of a served contract (its hole list travels with it) is judged for
            # names: a hand-written file with no solver import stays unjudged, as before
            return _hole_name_findings(text) if "WHAT YOUR SOLVE MUST LEAVE BEHIND" in text else []
        return []
    body = _strip_strings_and_comments(text)
    # AN UNFILLED SERVED CONTRACT IS NOT JUDGED FOR THE NAMES ITS HOLE MUST
    # DEFINE. Measured: the pristine FEniCSx contract, written before its fill,
    # drew nine NameError findings ("`domain` is used at line 234 and never
    # defined") in three cells of one round -- the served text below the hole
    # uses exactly the names the hole is asked to leave behind. Once the hole
    # is filled (its markers go with the fill) the names are judged again.
    # THE MARKERS WENT, THE LIST STAYED: the served text lost "DOES NOT SERVE THIS"
    # and the pristine contracts drew 8-9 NameError findings again (measured on the
    # served FEniCSx, NGSolve and scikit-fem contracts). The list the contract ends
    # with -- WHAT YOUR SOLVE MUST LEAVE BEHIND -- names the hole's names: those still
    # undefined are ONE note naming what the hole has left to define, and any other
    # name is judged as ever.
    out, seen = _module_findings(body, text), set()
    out += _hole_name_findings(text)
    for backend, pattern, error, fix in _TRAPS:
        if backend not in codes or (backend, pattern) in seen:
            continue
        m = re.search(pattern, body, re.M)
        if not m:
            continue
        seen.add((backend, pattern))
        # A pattern that reads the whole file first names only its `hit`; a hit over several lines
        # (a value followed from the call that made it to the attribute read on it) is quoted by its
        # first and last line.
        grp = "hit" if "hit" in m.re.groupindex else 0
        lines = [ln.strip() for ln in m.group(grp).strip().splitlines() if ln.strip()]
        quoted = lines[0] if len(lines) <= 1 else f"{lines[0]}` ... `{lines[-1]}"
        line = body[:m.start(grp)].count("\n") + 1
        out.append(f"{backend}: `{quoted}` (near line {line} of the stripped source) -- "
                   f"this run will stop with {error}. {fix}. Measured on this install.")
    return out


# ── the wrong interpreter, named BEFORE the run instead of after ───────────
#
# Running a participant with an interpreter that does not have its solver is
# the largest single named failure in the recorded runs: 481 occurrences of
# `No module named '<solver>'` across 3,943 trajectory files, 284 of them
# Kratos. The run-side table answers it from the traceback. This answers it
# from the COMMAND, which is what is left when the traceback never reaches the
# reply -- `python3 participant.py > log.txt 2>&1` is the common shape, and
# then the output is empty and the failure costs another call to find.
#
# IT ASKS THE INTERPRETER, IT DOES NOT INFER. An earlier draft compared the
# command's interpreter name against the path the backend reports and flagged
# a mismatch. That is wrong twice over: a venv path ENDS WITH "python", so the
# correct interpreter looked wrong, and the default python on this host DOES
# carry one of the solvers, so a working command was flagged as broken. A gate
# that names an absent defect costs an action for nothing. So: resolve the
# interpreter the command actually names, ask it to import the module, and say
# something only when that import really fails. One probe per
# (interpreter, module), cached.

_MODULE_TO_BACKEND = {
    "KratosMultiphysics": "kratos", "ngsolve": "ngsolve", "netgen": "ngsolve",
    "dolfinx": "fenics", "skfem": "skfem", "dune": "dune",
}
_BARE_PY = __import__("re").compile(
    r"(?:^|[|;&]|\s)(python3?(?:\.\d+)?)\s+(?:-[A-Za-z]\s+)*([^\s|;&]+\.py)")
_IMPORT_CACHE: dict = {}
_INTERPRETER_CACHE: dict = {}


def _interpreter_for(backend: str) -> str:
    """The interpreter this install runs `backend` with, or '' if unknown.

    Asks the backend itself, exactly as `discover` does, and caches: the probe
    starts a subprocess and this check runs on every shell command.
    """
    if backend in _INTERPRETER_CACHE:
        return _INTERPRETER_CACHE[backend]
    path = ""
    try:
        from core.registry import get_backend                    # noqa: PLC0415
        b = get_backend(backend)
        _status, msg = b.check_availability()
        m = __import__("re").search(r"\bat\s+(\S+)", msg or "")
        if m and m.group(1).endswith(("python", "python3")):
            path = m.group(1)
    except Exception:                                            # noqa: BLE001
        path = ""
    _INTERPRETER_CACHE[backend] = path
    return path


def _can_import(interpreter: str, module: str) -> bool:
    """Does THAT interpreter have that module? Asked once, then cached."""
    key = (interpreter, module)
    if key in _IMPORT_CACHE:
        return _IMPORT_CACHE[key]
    ok = True
    try:
        import subprocess                                        # noqa: PLC0415
        r = subprocess.run([interpreter, "-c", f"import {module}"],
                           capture_output=True, timeout=60,
                           stdin=subprocess.DEVNULL)
        ok = r.returncode == 0
    except Exception:                                            # noqa: BLE001
        ok = True                       # cannot tell -> say nothing
    _IMPORT_CACHE[key] = ok
    return ok


def wrong_interpreter_in_command(command: str, workdir=None) -> list[str]:
    """`python3 <script>.py` where THAT python cannot import the script's solver.

    Silent unless the import genuinely fails, so a default python that does
    carry a solver is never second-guessed.
    """
    import re as _re                                             # noqa: PLC0415
    import shutil                                                # noqa: PLC0415
    from pathlib import Path                                     # noqa: PLC0415
    if not command or "python" not in command:
        return []
    out: list[str] = []
    seen: set = set()
    for m in _BARE_PY.finditer(command):
        interp, script = m.group(1), m.group(2)
        if "/" in interp:            # an explicit path is the correct form
            continue
        resolved = shutil.which(interp)
        if not resolved:
            continue
        p = Path(script)
        if workdir and not p.is_absolute():
            p = Path(workdir) / script
        try:
            src = p.read_text(errors="ignore") if p.is_file() else ""
        except OSError:
            src = ""
        if not src:
            continue
        for mod, backend in _MODULE_TO_BACKEND.items():
            if backend in seen:
                continue
            if not _re.search(r"^\s*(?:import|from)\s+" + mod + r"\b", src, _re.M):
                continue
            if _can_import(resolved, mod):
                continue                 # this python really does have it
            seen.add(backend)
            path = _interpreter_for(backend)
            where = (f"Run it with the interpreter this backend reports: {path}"
                     if path else "discover(query='list') prints the interpreter "
                                  "for every backend on this install")
            out.append(
                f"`{interp} {script}` -- that script imports {mod}, and {interp} "
                f"here ({resolved}) cannot import it, so the run will stop with "
                f"ModuleNotFoundError before your solve is reached. {where}"
                + (" (discover(query='list') prints one for every backend, and "
                   "it is the argv couple needs too)." if path else "."))
    return out


# ── the export self-check, which agents drop without ever meeting it ───────
#
# MEASURED over 35 agent-written participants from two rounds: 23 kept the
# per-level dump and DROPPED the export self-check that sits beside it, and
# none did the reverse. It is not that the check got in their way -- 21 of the
# 23 dropped it in a script where it had never once fired. They simply did not
# copy it, and nothing said so until the export was already worthless.
#
# The three exports it stops all look fine: a non-finite field; a Neumann side
# whose imported load never entered the assembled system, which returns the
# no-load answer and a flux of ~0 against a nonzero partner; and a flux that is
# the partner's array negated instead of a recovery from this side's own
# system. Each produces a converging iteration and a wrong answer.

_EXPORTS_WRITE = __import__("re").compile(
    r"""exports\.json['"]\s*\)?\s*\.write_text|open\(\s*['"]exports\.json['"]|"""
    r"""json\.dump\([^)]*['"]exports\.json['"]""", __import__("re").S)


# ── which served checks a side lacks, read from ITS OWN served contract ─────
#
# Both notes below named one fixed list -- export self-check, held-edge check, interface list,
# solve self-check -- to every side. Measured over the served text of every contract: only the
# NGSolve heat contract carries all four; the held-edge check is otherwise the Kratos Neumann
# side's alone, the solve self-check NGSolve's alone, and the interface list NGSolve's and the
# FEniCSx and scikit-fem heat contracts'. A scikit-fem heat side was told it lacked a held-edge check
# and a solve self-check its contract never had, a Kratos side NGSolve's checks, and a 4C side three
# checks its scaffold does not carry, while the per-level dump that scaffold writes went unnamed. So
# the list is read from the served contract of the side's own code and role.
#
# (what it is called, how a served contract carries it, how a file shows it has it). A served check
# is a STOP. A file's label counts wherever it is, except on a line that is a print: a check demoted
# to a print no longer stops the run, and that is the one edit this can tell reliably (a label left in
# a comment may sit over a stop reworded below it, so it still counts). _check_text drops the prints.
_SERVED_CHECKS = (
    ("export self-check", re.compile(r"EXPORT SELF-CHECK:"), re.compile(r"EXPORT SELF-CHECK")),
    ("held-edge check", re.compile(r"OUTER BOUNDARY:"), re.compile(r"OUTER BOUNDARY")),
    ("interface list against the mesh", re.compile(r"INTERFACE (?:DOFS|VERTICES):"),
     re.compile(r"INTERFACE (?:DOFS|VERTICES)")),
    ("solve self-check", re.compile(r"SOLVE SELF-CHECK:"), re.compile(r"SOLVE SELF-CHECK")),
    # a file name carrying the level, written per level; a file shows it by any such name or write
    ("per-level dump", re.compile(r"""f["'][^"'\n]*level[^"'\n]*\{"""),
     re.compile(r"""PER-LEVEL|f["'][^"'\n]*(?:level|lvl)[^"'\n]*\{"""
                r"""|(?:open|savetxt|to_csv|write_text|dump)\s*\([^\n]*(?:level|lvl)""", re.I)),
)
# THE PER-LEVEL DUMP IS NAMED FOR 4C ONLY. The Python sides keep it (above: of 35 written sides, 23
# dropped the self-check and kept the dump, none the reverse); the 4C sides of one round wrote none,
# and a ladder whose side keeps no per-level file keeps only its last level.
_DUMP_NAMED_FOR = frozenset({"fourc"})
# A 4C side imports no solver module: it writes a deck and runs the binary.
_FOURC_SIDE = re.compile(r"PROBLEMTYPE|fourc_bin|FOURC_BIN|\.4C\.ya?ml|SCALAR TRANSPORT DYNAMIC")
_DOOR_ROLES = ("", "elastic", "thermoelastic", "transient", "3d", "neumann")
# the partner's role -> the roles this side can have in the same coupling
_PARTNER_ROLES = {"base": ("base", "neumann"), "neumann": ("base", "neumann"),
                  "tsi_mech": ("tsi_thermal",), "tsi_thermal": ("tsi_mech",),
                  "fsi_fluid": ("fsi_solid",), "fsi_solid": ("fsi_fluid",)}
# A side that states it is the Neumann (or Dirichlet) side: `SIDE = "neumann"`, or the first line
# of the Kratos Neumann contract, "... (NEUMANN side)."
_STATED_SIDE = re.compile(r"""^[ \t]*SIDE[ \t]*=[ \t]*["'](neumann|dirichlet)["']|\((NEUMANN|DIRICHLET) side\)""",
                          re.M | re.I)


def _stated_side(content: str) -> str:
    m = _STATED_SIDE.search(content or "")
    return (m.group(1) or m.group(2)).lower() if m else ""


def _side_codes(content: str) -> list:
    codes = [_CONTRACT_DOOR[c] for c in backends_in(content) if c in _CONTRACT_DOOR]
    if _FOURC_SIDE.search(content) and "fourc" not in codes:
        codes.append("fourc")
    # A PYTHON WRAPPER THAT RUNS A deal.II BINARY imports no solver module; it names
    # deal.II in its code (measured: its restore call offered no code, or another one's).
    if (not codes and "subprocess" in content
            and re.search(r"deal\.?ii|DEALII", content, re.I)):
        codes.append("dealii")
    # AND ONE THAT WRITES A FEBio DECK AND RUNS THE BINARY (measured: a FEBio elastic side that lost
    # its served Neumann branch was never judged, because no code was read off it).
    if not codes and "subprocess" in content and "<febio_spec" in content:
        codes.append("febio")
    return codes


@functools.lru_cache(maxsize=1)
def _contracts_by_role() -> dict:
    """{(code, role): (its served lines, the lines no other role of that code serves, the checks it
    carries, its served stops, its served functions, its branches that take the partner's data in, its
    served text)} for every contract a door serves: the main
    door's roles exactly as participant_contract_text returns them (the 4C and DUNE-fem base
    contracts are the door's scaffold), and the thermo-mechanical and fluid-structure contracts
    through the same serving door. A served stop is (condition, message head, label, side), as
    _guards_of reads it, outside the served functions; a served function is (name, side).
    Empty when the doors cannot be read, and then no list is named."""
    try:
        from tools import coupling_knowledge as ck                # noqa: PLC0415
    except Exception:                                             # noqa: BLE001
        return {}
    texts = {}
    for key in getattr(ck, "_BACKEND_ORDER", ()):
        for role in _DOOR_ROLES:
            try:
                text, err = ck.participant_contract_text(key, f"participant:{role}" if role else "participant")
            except Exception:                                     # noqa: BLE001
                continue
            if text and not err:
                texts[(key, role or "base")] = text
        for q in sorted(_PARTICIPANT_DIR.glob(f"participant_[tf]si_*_{key}.py")):
            try:
                texts[(key, "_".join(q.stem.split("_")[1:3]))] = ck._serve_participant(q)
            except Exception:                                     # noqa: BLE001
                continue
    lines = {kr: frozenset(ln.strip() for ln in t.splitlines() if len(ln.strip()) >= 40)
             for kr, t in texts.items()}
    out = {}
    for (key, role), t in texts.items():
        others = set().union(*[v for (k, r), v in lines.items() if k == key and r != role])
        carried = tuple(name for name, served, _has in _SERVED_CHECKS
                        if served.search(t) and (name != "per-level dump" or key in _DUMP_NAMED_FOR))
        # A FUNCTION THAT HOLDS A HOLE IS NOT A SERVED BLOCK: its body is partly the author's, and a
        # fill may reshape it (measured: a fluid side was told 37 times it lost "the function main()",
        # the function the served fluid contract puts its holes in). Its served stops are judged
        # one by one instead.
        _rows = t.splitlines()
        funcs = tuple(f for f in _served_functions(t)
                      if not any(_HOLE_BANNER in _rows[k] for k in range(f[1], min(f[2] + 1, len(_rows)))))
        out[(key, role)] = (lines[(key, role)], frozenset(lines[(key, role)] - others), carried,
                            tuple(_guards_of(t, inside=[(r0, r1) for _n, r0, r1, _s in funcs])),
                            tuple((n, s) for n, _r0, _r1, s in funcs), _served_branches(t), t)
    return out


_HOLE_BANNER = "THE SOLVE ITSELF IS YOURS AND IS NOT SERVED HERE"
# The roles whose contract exchanges one scalar (a temperature and its flux), and the words of a side
# that exchanges a traction or a displacement instead.
_SCALAR_ROLES = frozenset({"base", "neumann", "transient", "3d"})
_VECTOR_EXCHANGE = re.compile(r"traction|displacement", re.I)


def _code_only(text: str) -> str:
    """The text without its comments (a word in a comment exchanges nothing)."""
    return "\n".join(ln.split("#", 1)[0] for ln in (text or "").splitlines())


def _served_branches(text: str) -> tuple:
    """(side, label, its served lines squashed) of each module-level `if SIDE == ...` branch of a
    served contract that reads the partner's data (imp): the lines that take the partner's values or
    normal_fluxes into this side's system on that side. Measured on a coupled round: two workers
    re-typed the scikit-fem elastic contract and dropped its Neumann branch, the one line that adds
    the partner's traction into the load among them; no served function and no served stop was lost,
    so nothing named it, and the side never applied the partner's traction."""
    import ast
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return ()
    rows = text.splitlines()
    out = []
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        side, plain = _side_test(node.test)
        if not side:
            continue
        branches = [(side, node.body)]
        if plain and node.orelse and not (len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If)):
            branches.append((_OTHER_SIDE[side], node.orelse))
        for s, body in branches:
            if not any(isinstance(n, ast.Name) and n.id == "imp" for st in body for n in ast.walk(st)):
                continue
            lines = [code for st in body if not isinstance(st, ast.Raise)
                     for r in range(st.lineno - 1, getattr(st, "end_lineno", None) or st.lineno)
                     if len(code := rows[r].split("#", 1)[0].strip()) >= 12]
            if lines:
                label = f"`{lines[0][:60]}`" + (f" ... `{lines[-1][:70]}`" if len(lines) > 1 else "")
                out.append((s, label, frozenset(_squash(l) for l in lines)))
    return tuple(out)


def _side_roles(content: str, partner_text=None) -> list:
    """The served contracts, as (code, role), this side can be.

    In this order: the partner's served contract names the coupling (a heat partner makes this a heat
    side, a thermo-mechanical one the other thermo-mechanical role), and a side that states it is the
    Neumann side picks that contract where its code has one; else the file's own served lines -- a
    side re-typed from one contract carries lines only that contract serves (measured over 376
    recorded hand-written sides: where the best role held 5 % of its own lines, the next held at most
    3 %); else every contract its code serves, whose common checks are the only ones named."""
    table = _contracts_by_role()
    codes = _side_codes(content)
    cands = [kr for kr in table if kr[0] in codes]
    if not cands:
        return []
    if partner_text:
        have = {ln.strip() for ln in partner_text.splitlines()}
        best = max(((len(v[0] & have) / len(v[0]), kr) for kr, v in table.items() if v[0]), default=None)
        if best and best[0] >= 0.5:
            want = _PARTNER_ROLES.get(best[1][1], (best[1][1],))
            cands = [kr for kr in cands if kr[1] in want]    # none: its code serves no such contract
    side = _stated_side(content)
    if side and {"base", "neumann"} <= {r for _k, r in cands}:
        keep = "neumann" if side == "neumann" else "base"
        cands = [kr for kr in cands if kr[1] == keep or kr[1] not in ("base", "neumann")]
    if len(cands) <= 1:
        return cands
    have = {ln.strip() for ln in content.splitlines()}
    scores = sorted(((len(table[kr][1] & have) / len(table[kr][1]), kr) for kr in cands if table[kr][1]),
                    reverse=True)
    if scores and scores[0][0] >= 0.05 and (len(scores) == 1 or 3 * scores[1][0] <= scores[0][0]):
        return [scores[0][1]]
    return cands


def _served_checks_lacking(content: str, partner_text=None) -> list:
    """The checks this side's own served contract carries and the file does not.

    Read from the served contract of the side's code and role (_side_roles); when the role is not
    known, only the checks every contract of its code carries are named."""
    table = _contracts_by_role()
    roles = _side_roles(content, partner_text)
    if not roles:
        return []
    carried = set.intersection(*[set(table[kr][2]) for kr in roles])
    text = _check_text(content)
    return [name for name, _served, has in _SERVED_CHECKS if name in carried and not has.search(text)]


def _check_text(content: str) -> str:
    """The file without the lines that are prints: a label there is a check that no longer stops."""
    return re.sub(r"^[ \t]*print\s*\([^\n]*", "", content, flags=re.M)


def _named(items: list) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def missing_export_selfcheck(content: str) -> str:
    """'' unless this looks like a participant that dropped the served check.

    ANY self-check counts, not one spelling. The served set uses "EXPORT
    SELF-CHECK" for the three-way export guard and "CONSERVATION SELF-CHECK"
    for the divergence-theorem one, and a script that raises on a non-finite
    or copied export has the protection whatever it calls it. A gate that
    demands one wording would fire on contracts that are already protected,
    which is a gate naming an absent defect.
    """
    if not content or "SELF-CHECK" in content:
        return ""
    if "raise SystemExit" in content and "isfinite" in content:
        return ""                      # equivalent protection under another name
    if not _EXPORTS_WRITE.search(content):
        return ""                      # not a participant; nothing to say
    # A DSMC WRAPPER'S BLOCK IS NOT THE FINITE-ELEMENT ONE. Measured on a coupled round: a SPARTA
    # file that had lost its block was told about a Neumann side's load and a negated partner flux,
    # neither of which a DSMC side has.
    if re.search(r"\bsurf_collide\b|\bread_surf\b|\bspa_(?:serial|mpi)\b", content):
        return ("this script writes exports.json but carries no EXPORT SELF-CHECK. The served SPARTA "
                "contract has one, and it is the block that stops the two exports of a DSMC side that "
                "look fine and are worthless: a non-finite etot, and an etot that is exactly zero on "
                "every element (no particle collision was tallied there). Copy the block back from the "
                "served contract -- it reads only what you have already computed.")
    return (
        "this script writes exports.json but carries no EXPORT SELF-CHECK. The "
        "served contract has one, and it is the block that stops the three "
        "exports that look fine and are worthless: a non-finite field; a "
        "Neumann side whose imported load never entered the assembled system "
        "(it returns the no-load answer and a flux of ~0 against a nonzero "
        "partner); and a flux that is the partner's array negated rather than "
        "recovered from this side's own system. Each of those converges "
        "beautifully to a wrong answer. Copy the block back from the served "
        "contract -- it reads only what you have already computed."
        # AND THE OTHER SERVED CHECKS IT LACKS. Measured: a side re-written by hand
        # beside a partner that was hand-written too dropped the solve self-check that
        # had caught its defect three times, and only this note spoke, naming one block.
        # Only the checks its own code's served contract carries (_served_checks_lacking).
        + _also_lacks([w for w in _served_checks_lacking(content) if w != "export self-check"]))


def _also_lacks(items: list) -> str:
    return f" It also lacks the served {_named(items)}." if items else ""


# ── the imported values that never reached the answer ──────────────────────
#
# MEASURED over the 63 recorded NGSolve participants: 32 of them load the
# partner's interface values into individual entries of the solution vector
# and then lose them again, in one of two ways -- the solve ASSIGNS over the
# whole vector instead of updating it, or there is no solve at all. Both
# converge: a Dirichlet side that ignores its import is a fixed point, so the
# interface residual falls to ~1e-15 on the second iteration and every gate
# downstream reports agreement. One recorded cell converged to 1.8e-15 that
# way and graded on a field its partner never influenced.
#
# The separation is exact on the recorded runs: it fires only on failing
# ones and is silent on every correct one. It is silent on all 32
# served contracts, including the two volume-coupled ones whose import
# legitimately enters through the load vector rather than through essential
# entries -- which is why the check identifies the solution vector from the
# SOLVE STATEMENT rather than flagging any indexed write it finds.
#
# It names the defect and the invariant. What to solve, and how, stays the
# author's: the only claim here is that values written into a vector have to
# survive the step that follows them.

# `gfu.vec.data = ...` / `+=`, and the same through a full slice: `gfu.vec[:] = sol`
# copies a solve's result over the whole vector (measured: silent on exactly that line
# in a fill whose run-time check then caught the lost boundary data).
_SOLVE_ASSIGN = re.compile(r"(\w+)\.vec(?:\.data|\s*\[\s*:\s*\])\s*(\+?=)(?!=)\s*([^\n]*)")
# A solve computed and thrown away: the statement starts with the inverse call.
_DISCARDED_SOLVE = re.compile(
    r"^[ \t]*(?:[\w\.]+\.)?Inverse\s*\((?:[^()\n]|\([^()\n]*\))*\)\s*\*\s*[\w\.\[\]\(\)]+[ \t]*$", re.M)
_SOLVE_CALL = re.compile(r"\bInverse\s*\(|\bCGSolver\b|\bsolvers\.\w|\bBVP\s*\(|\b(?:CG|GMRes|MinRes)\s*\(")
# An inverse applied through Mult: M.Mult(x, y) writes M x INTO y.
_INVERSE_MULT = re.compile(
    r"(?:\bInverse\s*\((?:[^()\n]|\([^()\n]*\))*\)|\b(?P<inv>\w+))\s*\.Mult\s*\(\s*"
    r"(?P<x>-?\s*[\w\.]+)\s*,\s*(?P<y>-?\s*[\w\.]+)\s*[,)]")
_INVERSE_NAME = re.compile(r"^\s*(\w+)\s*=\s*[\w\.]*Inverse\s*\((?:[^()\n]|\([^()\n]*\))*\)\s*$", re.M)
_BOUND_NAME = re.compile(r"^\s*(\w+)\s*=\s*([^\n]*)", re.M)
_IFACE_NAME = re.compile(r"interface|iface", re.I)
# a mask bit cleared -- not an attribute's subscript: `gfu.vec[int(d)] = 0.0` zeroes a value
# and was read as a mask, which silenced "nothing holds those entries fixed" (measured)
# ... and a single bit cleared by name, m.Clear(i), is the same act (measured: a file that cleared
# each interface dof this way was told no free-dof mask excludes them)
_FREE_MASK = re.compile(r"(?<![\w.])\w+\s*\[\s*(?:int\()?\w+\)?\s*\]\s*=\s*(?:False|0)\b"
                        r"|(?<![\w.])\w+\.Clear\s*\(\s*[^)\s][^)]*\)")
_BILINEAR = re.compile(r"\bBilinearForm\s*\(")

# ── WHICH LINES CAN RUN TOGETHER ───────────────────────────────────────────
#
# One file serves both roles: the served contract writes the partner's trace
# under `if SIDE == "dirichlet":`, and a fill that solves each role its own way
# puts the two solves on the two branches of a second `if SIDE == ...`. Pairing
# a write with a solve by text order alone joined the Dirichlet-branch write to
# the Neumann-branch solve and named a lost condition on a path where nothing
# had been written (measured: one run was told so after each of its last seven
# writes; its Dirichlet solve was the residual form throughout).
_IF_HEAD = re.compile(r"^(\s*)(if|elif)\s+(.+?)\s*:\s*$")
_ELSE_HEAD = re.compile(r"^(\s*)else\s*:\s*$")
_EQ_TEST = re.compile(r"^\(?\s*([\w\.]+(?:\(\s*\))?(?:\.\w+\(\s*\))*)\s*(==|!=)\s*([\"'])(.*?)\3\s*\)?$")


def _indent(s: str) -> int:
    return len(s) - len(s.lstrip())


def _branches_at(body: str, pos: int) -> list:
    """The if-branches around the line at `pos`, innermost first, as (row of the
    chain's `if`, branch number, [(condition, holds), ...]); an elif or an else
    also records that every earlier condition of its chain failed."""
    lines = body.split("\n")
    row = body.count("\n", 0, pos)
    level, out, r = _indent(lines[row]), [], row - 1
    while r >= 0 and level > 0:
        s = lines[r]
        if s.strip() and _indent(s) < level:
            head = _IF_HEAD.match(s)
            if head or _ELSE_HEAD.match(s):
                ind, earlier, k = _indent(s), [], r
                if not (head and head.group(2) == "if"):
                    k = r - 1
                    while k >= 0:
                        t = lines[k]
                        if not t.strip() or _indent(t) > ind:
                            k -= 1
                            continue
                        prev = _IF_HEAD.match(t) if _indent(t) == ind else None
                        if not prev:
                            break
                        earlier.append(prev.group(3))
                        if prev.group(2) == "if":
                            break
                        k -= 1
                facts = [(c, False) for c in earlier] + ([(head.group(3), True)] if head else [])
                out.append((k, len(earlier), facts))
            level = _indent(s)
        r -= 1
    return out


def _clash(a: tuple, b: tuple) -> bool:
    """Two (condition, holds) facts that cannot both be true."""
    (ca, ha), (cb, hb) = a, b
    ca, cb = " ".join(ca.split()), " ".join(cb.split())
    if ca == cb:
        return ha != hb
    ma, mb = _EQ_TEST.match(ca), _EQ_TEST.match(cb)
    if not (ma and mb and ma.group(1) == mb.group(1)):
        return False
    eq_a = (ma.group(2) == "==") == ha            # the name equals its literal
    eq_b = (mb.group(2) == "==") == hb
    if eq_a and eq_b:
        return ma.group(4) != mb.group(4)
    return eq_a != eq_b and ma.group(4) == mb.group(4)


def _never_together(body: str, p: int, q: int) -> bool:
    """True when the lines at p and q sit on branches that cannot both run."""
    for ra, ka, fa in _branches_at(body, p):
        for rb, kb, fb in _branches_at(body, q):
            if (ra == rb and ka != kb) or any(_clash(x, y) for x in fa for y in fb):
                return True
    return False


def _solver_bound_names(body: str) -> set:
    """Names holding a solver, so `inv = A.Inverse(...)` then `u.vec.data = inv * f` reads as a solve.

    Two lines are the common spelling and a one-line pattern misses them; on
    the recorded set that difference is 8 scripts.
    """
    names: set = set()
    for _ in range(3):                                  # follow a rebind or two
        grew = False
        for m in _BOUND_NAME.finditer(body):
            rhs = m.group(2)
            if _SOLVE_CALL.search(rhs) or any(re.search(rf"\b{re.escape(n)}\b", rhs) for n in names):
                if m.group(1) not in names:
                    names.add(m.group(1))
                    grew = True
        if not grew:
            break
    return names


_BITARRAY = re.compile(r"^\s*(\w+)\s*=\s*(?:ngsolve\.)?BitArray\s*\(\s*([^)]*?)\s*\)", re.M)
_FOR_IN = re.compile(r"\bfor\s+(\w+)\s+in\s+([A-Za-z_][\w\.]*(?:\s*\([^()\n]*(?:\([^()\n]*\))?[^()\n]*\))?)\s*(?=:|\bif\b|\]|\))")
_REGION = re.compile(r"\b(?:Boundaries|Materials|Region)\s*\(")


def _last_definition(body: str, name: str, upto: int) -> str:
    found = ""
    for dm in re.finditer(rf"^\s*{re.escape(name)}\s*=\s*([^\n]*)", body[:upto], re.M):
        found = dm.group(1)
    return found


def _is_mask(expr: str, body: str, upto: int, depth: int = 0) -> bool:
    """An NGSolve BitArray, or a set/list of one: BitArray(...), X.FreeDofs(...),
    X.GetDofs(<a region>), or a name last bound to one of those. A comprehension
    over range() that tests the mask is a list of numbers and is not one."""
    e = re.sub(r"^(?:~\s*|\(\s*)+", "", expr.strip())
    w = re.match(r"(?:set|list|sorted|tuple)\s*\((.*)\)\s*$", e)
    e = w.group(1).strip() if w else e
    if re.match(r"(?:ngsolve\.)?BitArray\s*\(|[\w\.]+\.FreeDofs\s*\(", e):
        return True
    g = re.match(r"[\w\.]+\.GetDofs\s*\((.*)\)", e)
    if g:
        arg = g.group(1).strip()
        return bool(_REGION.search(arg)) or (bool(re.fullmatch(r"\w+", arg)) and bool(
            _REGION.search(_last_definition(body, arg, upto))))
    if re.fullmatch(r"\w+", e) and depth < 2:
        d = _last_definition(body, e, upto)
        return bool(d) and _is_mask(d, body, upto, depth + 1)
    return False


def _mask_walked_as_numbers(body: str) -> str:
    """'' unless a loop walks a BitArray and uses what it yields as dof numbers.

    MEASURED on this install: iterating a BitArray yields its bits as True/False,
    and a mask filled by `for d in fes.FreeDofs(): m[d] = True` from a space with
    12 free dofs came out with 2 bits set, dofs 0 and 1. A coupled run built its
    Dirichlet-side mask this way (FreeDofs minus the interface dofs); on its own
    space the mask ended with one bit set, on a held corner, so its solve moved
    nothing and left the interior at its starting zeros."""
    lines = body.split("\n")
    for m in _FOR_IN.finditer(body):
        var, expr = m.group(1), m.group(2).strip()
        if not _is_mask(expr, body, m.start()):
            continue
        row = body.count("\n", 0, m.start())
        scope = [lines[row].replace(m.group(0), " \u00a7 ")]
        for t in lines[row + 1:]:
            if t.strip() and _indent(t) <= _indent(lines[row]):
                break
            scope.append(t)
        used = "\n".join(scope)
        v = re.escape(var)
        if re.search(rf"\[\s*(?:int\s*\(\s*)?{v}\s*\)?\s*\]|\b{v}\s+(?:not\s+)?in\b|\bint\s*\(\s*{v}\s*\)", used):
            return (f"`for {var} in {expr}` walks an NGSolve BitArray, and a BitArray yields its bits "
                    f"as True/False, not the numbers of the dofs that are set (measured: a mask filled "
                    f"this way from a space with 12 free dofs came out with 2 bits set, dofs 0 and 1). "
                    f"Every `[{var}]` or `{var} in ...` in that loop therefore reads 0 or 1, and what "
                    f"it builds is not what you meant. Walk the numbers -- for i in range(len({expr})): "
                    f"if {expr}[i]: ... -- or take BitArray(fes.FreeDofs()), a copy, and clear the "
                    f"dofs you fix.")
    return ""


def unset_mask_bits(content: str) -> str:
    """'' unless an NGSolve mask is built with bits that nothing set.

    MEASURED on this install: a fresh BitArray(200) had 49, 14, 46 and 17 bits set
    in four constructions -- it is not zeroed -- and BitArray([3, 5, 7]) is three
    True bits, not a mask over dofs 3, 5 and 7. On one coupled round every
    coupling that stalled had a side built on one of the two: the same imports
    gave exports 11-92 % apart, and no iteration count converges below that.
    """
    if not isinstance(content, str) or "ngsolve" not in backends_in(content):
        return ""
    body = _strip_strings_and_comments(content)
    walked = _mask_walked_as_numbers(body)
    if walked:
        return walked
    for m in _BITARRAY.finditer(body):
        name, arg = m.group(1), m.group(2).strip()
        if arg.startswith("[") or "FreeDofs" in arg or "GetDofs" in arg:
            continue
        # A COPY OF A MASK IS A MASK: BitArray(free_dofs) with free_dofs = fes.FreeDofs()
        # copies every bit, and was called "not zeroed" (measured, cost 41 s).
        _src = ""
        for dm in re.finditer(rf"^\s*{re.escape(arg)}\s*=\s*([^\n]*)", body[:m.start()], re.M):
            _src = dm.group(1)
        if re.fullmatch(r"\w+", arg) and re.search(r"FreeDofs|GetDofs|BitArray\s*\(", _src):
            continue
        after = body[m.end():]
        # A SINGLE-BIT .Set(i) / .Clear(i) IS A SINGLE-BIT WRITE TOO: a mask built by
        # `m = BitArray(n)` and `m.Set(i)` for its free dofs was not seen, through four
        # versions and 7.6 minutes of one side (measured).
        first_write = re.search(rf"\b{re.escape(name)}\s*\[[^\]]+\]\s*=(?!=)"
                                rf"|\b{re.escape(name)}\.(?:Set|Clear)\s*\(\s*[^)\s]", after)
        # A WRITE OF EVERY BIT INITIALISES IT as well as .Clear() or .Set() does: a
        # full slice `m[:] = FreeDofs()` (every bit copied, measured on 20 of 20
        # constructions) and a loop over all of its bits were called "single-bit
        # writes" on two correct files, one of them final.
        n_ = re.escape(name)
        a_ = re.escape(arg)
        inits = [i for i in (
            re.search(rf"\b{n_}\.(?:Clear|Set)\s*\(\s*\)", after),
            re.search(rf"\b{n_}\s*\[\s*:\s*\]\s*=(?!=)", after),
            re.search(rf"\bfor\s+(\w+)\s+in\s+range\s*\(\s*(?:{a_}|len\s*\(\s*{n_}\s*\))\s*\)\s*:"
                      rf"\s*{n_}\s*\[\s*\1\s*\]\s*=(?!=)", after)) if i]
        init = min(inits, key=lambda i: i.start()) if inits else None
        d = ""
        for dm in re.finditer(rf"^\s*{re.escape(arg)}\s*=\s*([^\n]*)", body[:m.start()], re.M):
            d = dm.group(1)
        from_indices = bool(re.fullmatch(r"\w+", arg)) and bool(
            re.match(r"\s*(?:\[\s*(\w+)\s+for\s+\1\b|list\s*\(|sorted\s*\(|set\s*\()", d))
        if from_indices:
            return (f"`{name} = BitArray({arg})` builds a mask from a LIST, and BitArray reads a list "
                    f"as one BOOLEAN per entry, not as dof numbers: BitArray([3, 5, 7]) is three True "
                    f"bits (measured). Build the full-length mask instead -- start from "
                    f"fes.FreeDofs() and clear the dofs you fix, or BitArray(fes.ndof) with .Clear() "
                    f"and set the free ones.")
        if first_write and (not init or init.start() > first_write.start()):
            return (f"`{name} = BitArray({arg})` is not zeroed: its bits start as whatever memory "
                    f"held (measured: 49, 14, 46 and 17 of 200 set in four constructions), and "
                    f"single-bit writes leave the rest random. The solve then differs from run to "
                    f"run on the same inputs, and a coupling built on it never converges. Call "
                    f"{name}.Clear() (all False) or {name}.Set() (all True) first -- or start from "
                    f"fes.FreeDofs(), which is already right, and clear only the dofs you fix.")
    return ""


def _config_side(near) -> str:
    """The side ("neumann" or "dirichlet") config.json beside the file states, or ''."""
    if near is None:
        return ""
    try:
        import json as _json                                      # noqa: PLC0415
        cfg = _json.loads((Path(near).parent / "config.json").read_text() or "{}")
    except (OSError, ValueError):
        return ""
    side = str(cfg.get("side", "")).strip().lower() if isinstance(cfg, dict) else ""
    return side if side in ("neumann", "dirichlet") else ""


def imported_values_not_held(content: str, near=None) -> str:
    """'' unless partner values are loaded into the solution vector and then lost. `near` is the
    path of the file judged: config.json beside it may state the side."""
    if not isinstance(content, str) or "ngsolve" not in backends_in(content):
        return ""
    # A NEUMANN SIDE HOLDS NO TRACE. Measured: a Neumann side that wrote imported values
    # into its vector as a starting guess was told to hold its interface fixed.
    # AND THE SIDE MAY BE STATED IN config.json, which the served contracts read over the
    # file's SIDE: a correct Neumann side that set its role there, its SIDE constant left at
    # the served "dirichlet", drew this check's finding nine times (measured).
    _side = _config_side(near)
    if _side == "neumann" or (not _side and (
            re.search(r"^SIDE\s*=\s*[\"']neumann[\"']", content, re.M | re.I)
            or re.search(r"\(NEUMANN side\)", content[:400], re.I))):
        return ""
    body = _strip_strings_and_comments(content)
    bound = _solver_bound_names(body)

    def _calls_a_solver(rhs: str) -> bool:
        return bool(_SOLVE_CALL.search(rhs)) or any(
            re.search(rf"\b{re.escape(n)}\b", rhs) for n in bound)

    # A SOLVE THROUGH A TEMPORARY IS A SOLVE. `sol_vec.data = inv_a * f.vec` then
    # `gfu.vec.data = sol_vec.data` was not one to this check, so it stayed silent on
    # the defect it exists for (measured: two coupled runs' final files, one of which
    # gave up with the right diagnosis and no fix).
    temps = {}
    for tm in re.finditer(r"^\s*(\w+)(?:\.data)?\s*=\s*([^\n]*)", body, re.M):
        if tm.group(1) not in bound and _calls_a_solver(tm.group(2)):
            temps[tm.group(1)] = tm.group(2).strip()

    def _is_solve(rhs: str) -> bool:
        base = re.sub(r"\.data\s*$", "", rhs.strip())
        return _calls_a_solver(rhs) or base in temps

    def _whole(m) -> str:
        """The right-hand side through its closing parenthesis: a solve wrapped
        over two lines was read to its first line only (the served contract's
        own `... Inverse(fes.FreeDofs(),` / `inverse=...) * res` was flagged)."""
        rhs, depth, i = m.group(3), m.group(3).count("(") - m.group(3).count(")"), m.end()
        while depth > 0 and i < len(body):
            j = body.find("\n", i + 1)
            j = len(body) if j < 0 else j
            nxt = body[i:j]
            rhs += " " + nxt.strip()
            depth += nxt.count("(") - nxt.count(")")
            i = j
        return rhs

    # A SOLVE WHOSE RESULT GOES INTO ANOTHER VECTOR. Measured on a coupled run: every
    # version of its side solved with `a.mat.Inverse(free_dofs).Mult(gfu.vec, f.vec)`,
    # which applies the inverse to the solution vector and writes the result over the
    # load; the solution was never solved, and the run handed in nothing.
    _gfs = set(re.findall(r"^\s*(\w+)\s*=\s*(?:ngsolve\.)?GridFunction\s*\(", body, re.M))
    _lfs = set(re.findall(r"^\s*(\w+)\s*=\s*(?:ngsolve\.)?LinearForm\s*\(", body, re.M))
    _invs = set(_INVERSE_NAME.findall(body))
    for mm in _INVERSE_MULT.finditer(body):
        if mm.group("inv") and mm.group("inv") not in _invs:
            continue
        x = re.sub(r"[\s-]", "", mm.group("x"))
        y = re.sub(r"\s", "", mm.group("y"))
        if x.endswith(".vec") and x[:-4] in _gfs and y != x:
            into = (f"a temporary ({y}) that nothing keeps" if y.startswith("-") else y)
            return (f"`{' '.join(mm.group(0).split())[:90]}`: M.Mult(x, y) writes M x INTO y, so "
                    f"this applies the inverse to {x} and puts the result in {into} -- {x[:-4]} "
                    f"is never solved" + (", and the load is written over" if y.endswith(".vec")
                                          and y[:-4] in _lfs else "")
                    + ". A solve's result has to land in the solution vector.")
    # A SOLVE INTO A TEMPORARY THROUGH Mult IS A SOLVE TOO. `inv.Mult(f.vec, tmp)` then
    # `gfu.vec.data = tmp` replaced the solution vector with the load's solve, and no line of it read
    # as a solve here (measured: silent at write time, stopped later by the run-time self-check).
    # What the temporary holds is the inverse applied to Mult's first argument.
    mult_src = {}
    for mm in _INVERSE_MULT.finditer(body):
        if mm.group("inv") and mm.group("inv") not in _invs and mm.group("inv") not in bound:
            continue
        x, y = re.sub(r"\s", "", mm.group("x")), re.sub(r"\s", "", mm.group("y"))
        if re.fullmatch(r"[A-Za-z_]\w*", y) and not x.startswith("-"):
            mult_src[y] = f"{mm.group('inv') or 'Inverse(...)'} * {x}"
            temps.setdefault(y, " ".join(mm.group(0).split()).rstrip(",)") + ")")
    dropped = _DISCARDED_SOLVE.search(body)
    if dropped:
        return (f"`{' '.join(dropped.group(0).split())[:90]}` computes a solve and throws its "
                f"result away: nothing is assigned, so the solution vector keeps whatever was "
                f"written into it before, and every value exported from it is that. A solve's "
                f"result has to land in the solution vector.")
    solves = [(m.group(1), m.group(2), _whole(m), m.start(),
               "vec[:]" if "[" in m.group(0).split("=")[0] else "vec.data")
              for m in _SOLVE_ASSIGN.finditer(body) if _is_solve(m.group(3))]
    if not solves:
        if _BILINEAR.search(body) and not _SOLVE_CALL.search(body):
            return (
                "this script assembles a bilinear form and never solves with it. "
                "Nothing inverts the matrix, so the vector it exports is whatever "
                "was written into it by hand -- on a Dirichlet side that is the "
                "partner's own trace with zeros behind it, and the flux recovered "
                "from it is the flux of a field that was never computed. It "
                "converges, because a side that ignores its import cannot "
                "disagree with its partner twice. Check that the matrix you "
                "assemble is the one your solution comes out of.")
        return ""

    def _definition(name: str) -> str:
        """The right-hand side the script last gave `name` (or `name.data`)."""
        found = ""
        for dm in re.finditer(rf"^\s*{re.escape(name)}(?:\.data)?\s*=\s*([^\n]*)", body, re.M):
            found = dm.group(1)
        return found

    def _applied_to(expr: str, depth: int = 0) -> str:
        """What the solve is applied to: 'residual' (f - A u, the lifting present),
        'load' (a linear form's vector alone), or 'unknown' -- and only 'load' is
        a finding: a vector whose definition the script does not show is not
        called wrong."""
        expr = expr.strip()
        base = re.sub(r"\.data\s*$", "", expr)
        if base in mult_src and depth <= 2:            # a temporary a Mult wrote into
            return _applied_to(mult_src[base], depth + 1)
        if "-" in expr and ".mat" in expr:
            return "residual"
        # f - A u written in two steps: `Au.data = a.mat * gfu.vec` then `f.vec - Au`
        if "-" in expr and depth <= 2:
            for term in re.split(r"[-+]", expr):
                nm = term.strip().split(".")[0]
                if re.fullmatch(r"[A-Za-z_]\w*", nm or "") and ".mat" in _definition(nm):
                    return "residual"
        m = re.search(r"\*\s*\(?\s*([A-Za-z_][\w\.]*)\s*\)?\s*$", expr)
        operand = m.group(1) if m else (expr if re.fullmatch(r"[A-Za-z_]\w*", expr) else "")
        if not operand or depth > 2:
            return "unknown"
        d = _definition(operand.split(".")[0])
        if not d:
            return "unknown"
        if re.match(r"\s*LinearForm\s*\(", d):
            return "load"
        return _applied_to(d, depth + 1)

    # The index is usually the loop variable, so the interface name sits on
    # the `for` line above it rather than inside the brackets. Read the
    # write together with the lines that bind it.
    def _from_interface(w) -> bool:
        head = body[:w.start()].rsplit("\n", 4)[1:]
        return bool(_IFACE_NAME.search("\n".join(head) + w.group(0)))

    for sol, op, rhs, at, lhs_form in solves:
        writes = list(re.finditer(rf"\b{re.escape(sol)}\.vec\s*\[([^\]]+)\]\s*=(?!=)\s*([^\n]*)", body))
        # A ZERO START IS NOT A LOST CONDITION. `gfu.vec[:] = 0.0` before an
        # assignment-solve loses nothing, and this fired on it (measured).
        writes = [w for w in writes
                  if not re.fullmatch(r"\(?\s*0*\.?0*(?:e[+-]?\d+)?\s*\)?\s*", w.group(2).strip(), re.I)]
        # ONLY WHAT IS WRITTEN BEFORE THE SOLVE CAN BE LOST OR LEFT OUT OF IT.
        writes = [w for w in writes if not _never_together(body, w.start(), at)]
        # `gfu.Set(0)` IS A ZERO START TOO, like `gfu.vec[:] = 0.0`: it fired on the
        # correct final file of a right run, whose values were written
        # later on a branch that solves in the residual form (measured).
        before = [w for w in writes if w.start() < at] + [
            None for s in re.finditer(rf"\b{re.escape(sol)}\.Set\s*\(\s*([^\n]*)", body[:at])
            if not _never_together(body, s.start(), at)
            and not re.match(r"(?:(?:CoefficientFunction|CF)\s*\(\s*)?\(?\s*0*\.?0*(?:e[+-]?\d+)?"
                             r"\s*\)?\s*(?:\)|,)", s.group(1), re.I)][:1]
        # IN WORDS, NOT AS THE TWO LINES OF THE SOLVE: the literal residual-form
        # solve given here was copied verbatim into correct runs, and the linear
        # solve is one of the things openPASO does not serve.
        lifting = ("Solve for the CORRECTION instead: with the fixed values already in "
                   "the solution vector, apply the inverse to the residual those values "
                   "leave -- the load minus the assembled operator applied to them -- "
                   "and ADD the result. That is the one form in which the fixed values "
                   "both stay and reach the interior.")
        if not before:
            # THE LOAD SOLVED ALONE, THE HELD VALUES WRITTEN BACK AFTER IT. Nothing is lost
            # before the solve, so this check passed it -- and the interior was computed without
            # the interface values, which only the exported trace then carries (measured: silent
            # at write time, stopped later by the run-time self-check). A correction solved from
            # the residual and written back is the right form, and is not this.
            after = [w for w in writes if w.start() > at and _from_interface(w)]
            if op == "=" and "+" not in rhs and after and _applied_to(rhs) == "load":
                said = " ".join(rhs.split())
                said = said if len(said) <= 72 else said[:69] + "..."
                _t = re.sub(r"\.data\s*$", "", rhs.strip())
                if _t in temps:
                    said += f"`, where `{_t}` is `{' '.join(temps[_t].split())[:60]}"
                return (
                    f"`{sol}.{lhs_form} = {said}` solves the load alone, and the interface values "
                    f"written into `{sol}.vec[...]` after it never reach the interior: it was "
                    "computed without them and is the answer to a different problem, while the "
                    "exported trace carries them. This converges, so the residual history will "
                    "not tell you. " + lifting)
            continue                                   # nothing essential loaded into it
        if op == "=" and "+" not in rhs:
            said = " ".join(rhs.split())
            said = said if len(said) <= 72 else said[:69] + "..."
            _t = re.sub(r"\.data\s*$", "", rhs.strip())
            if _t in temps:
                said += f"`, where `{_t}` is `{' '.join(temps[_t].split())[:60]}"
            # A CORRECTION REPLACING THE VECTOR IS RIGHT WHEN THE HELD VALUES GO BACK.
            # Solved from the residual, the free entries are the answer and the held
            # ones come out zero; written back after it, the field is the lifting's.
            # Measured: a correct final file doing exactly that was told its interior
            # "was computed without them".
            if _applied_to(rhs) == "residual":
                _pre = {w.group(1).strip() for w in writes
                        if w.start() < at and not _never_together(body, w.start(), at)}
                _post = [w for w in writes if w.start() > at and w.group(1).strip() in _pre]
                if _post:
                    continue
                return (
                    f"`{sol}.{lhs_form} = {said}` replaces the whole vector with the "
                    "CORRECTION alone: the inverse on the free dofs leaves zero on every "
                    "held one, and the values written there before are not written back. "
                    "Where any of them is not zero -- the partner's trace, an outer value -- "
                    "the field no longer holds it. ADD the correction to the vector instead, "
                    "or write the held values back after it.")
            # WHAT THIS READS IS THE TEXT: values written, then the vector replaced. Whether
            # they were zero it cannot read (measured: it said "the answer to a different
            # problem" nine times of a side whose held values were an outer value of 0).
            return (
                f"`{sol}.vec[...]` is written with values first and then "
                f"`{sol}.{lhs_form} = {said}` replaces the whole vector, so those "
                "values are gone by the time anything reads them. Where any of them "
                "is not zero -- a prescribed outer value, the partner's interface "
                "trace on a Dirichlet side -- the system was solved as though that "
                "boundary were held at zero, re-applying the value AFTER the solve "
                "does not repair the interior, and it still converges, so the "
                "residual history will not tell you; values that are all zero lose "
                "nothing. " + lifting)
        if op == "+=" and _applied_to(rhs) == "load":
            # MEASURED: the old closing hint here ("an update keeps what is already
            # there") steered two runs into exactly this form, and the check was
            # silent on it; one of them passed a grade with the error it leaves.
            return (
                f"`{sol}.vec.data += <inverse> * <load>` keeps the values written into "
                f"`{sol}.vec` on the fixed dofs, but the vector the inverse is applied "
                "to is the load alone, not the residual f - A u: the free dofs are "
                "solved as if every fixed value were zero, so the boundary data -- the "
                "partner's interface trace, the prescribed outer values -- never reach "
                "the interior. It converges, and its error does not shrink with the "
                "mesh. " + lifting)
        # A MASK OF ITS OWN IS A MASK. A solve inverted on a BitArray the script
        # built (cleared, then the free bits set) may well hold the interface; this
        # branch only knew masks made by clearing bits and called such a side
        # "nothing holds those entries fixed" (measured on a recorded fill whose
        # mask was broken for another reason, named by unset_mask_bits).
        def _inverse_arg(expr: str) -> str:
            """The WHOLE first argument of the Inverse the solve applies, traced through a temporary
            or a name bound to the inverse; '' when there is none. Read to its closing comma or
            parenthesis: `Inverse(fes.FreeDofs() & mask, ...)` was read as `fes.FreeDofs()`."""
            src = temps.get(re.sub(r"\.data\s*$", "", expr.strip()), expr)
            for t in [src] + [_definition(n) for n in bound if re.search(rf"\b{re.escape(n)}\b", src)]:
                m = re.search(r"Inverse\s*\(", t)
                if not m:
                    continue
                depth, j = 0, m.end()
                while j < len(t) and not (depth == 0 and t[j] in ",)"):
                    depth += {"(": 1, "[": 1, ")": -1, "]": -1}.get(t[j], 0)
                    j += 1
                return re.sub(r"^\s*freedofs\s*=\s*", "", t[m.end():j]).strip()
            return ""

        def _custom_mask(expr: str) -> bool:
            arg = _inverse_arg(expr)
            if not arg or "FreeDofs" in arg:
                return False
            d = _definition(arg) if re.fullmatch(r"\w+", arg) else ""
            return not re.fullmatch(r"\s*[\w\.]+\.FreeDofs\s*\([^)]*\)\s*", d or "")

        # A MASK THE SCRIPT BUILDS IS DECIDED BY THE SCRIPT, OR BY THE CHECK THAT READS IT. Measured:
        # a file took fes.FreeDofs(), cleared it and set only the free dofs off the interface, and was
        # told "no free-dof mask excludes them"; another never zeroed its BitArray, and the same
        # sentence named the wrong fault. A mask in the inverse's argument that is zeroed and then set
        # bit by bit is the script's own choice of free dofs, which this check cannot read without
        # running it; an unzeroed one is named by unset_mask_bits. Either way this note says nothing.
        def _mask_decided_elsewhere(expr: str) -> bool:
            arg = _inverse_arg(expr)
            names = set(re.findall(r"[A-Za-z_]\w*", arg))
            for nm in list(names):
                names.update(re.findall(r"[A-Za-z_]\w*", _definition(nm)))
            for nm in names:
                n_ = re.escape(nm)
                first = re.search(rf"\b{n_}\.Set\s*\(\s*[^)\s]|\b{n_}\s*\[(?!\s*:\s*\])[^\]\n]+\]\s*=(?!=)\s*(?:True|1)\b",
                                  body)
                zero = re.search(rf"\b{n_}\.Clear\s*\(\s*\)|\b{n_}\s*\[\s*:\s*\]\s*=(?!=)\s*(?:False|0)\b", body)
                if first and zero and zero.start() < first.start():
                    return True
            unzeroed = re.match(r"`(\w+) = BitArray", unset_mask_bits(content) or "")
            return bool(unzeroed and unzeroed.group(1) in names)

        if (any(_from_interface(w) for w in writes)
                and not _FREE_MASK.search(body) and not _custom_mask(rhs)
                and not _mask_decided_elsewhere(rhs)):
            dm = re.search(r"H1\s*\([^)]*dirichlet\s*=\s*([\"'][^\"']*[\"']|\([^)]*\))", content)

            def _holds_interface(spec: str) -> bool:
                # NGSolve reads dirichlet= as a regular expression over the boundary
                # names: ".*" holds the interface as surely as "outer|interface" does
                if _IFACE_NAME.search(spec):
                    return True
                try:
                    return bool(re.fullmatch(spec.strip().strip("'\""), "interface"))
                except re.error:
                    return False

            if not (dm and _holds_interface(dm.group(1))):
                return (
                    f"interface values are written into `{sol}.vec[...]`, but "
                    "nothing holds those entries fixed that this check can read: the "
                    "space's `dirichlet=` argument names no boundary called "
                    "\"interface\" (the name the served lines integrate over) and no "
                    "free-dof mask excludes them. The solve is free to move them, so the "
                    "condition you imported is a starting guess rather than a "
                    "boundary condition. Either name the boundary in the space or "
                    "clear those entries out of the BitArray the solve inverts on.")
    return ""


# ── the tetrahedra the author built, judged as a split of the hex ─────────
#
# Kratos ships no mesher, so a 3D participant writes its own connectivity, and
# Kratos assembles any four nodes it is handed without a word about the mesh.
# MEASURED on this install (Kratos 10.3, the 3-D contract with its split swapped,
# placeholder data): a flat tetrahedron gives NaN fluxes, with "ATTENTION!
# setting the RHS to zero!" when three of six are flat; a mix of positive and
# negative ones stops the solve with "Error zero sum" (skyline LU) or "Zero sum
# in skyline_lu factorization" (AMGCL), or skips it with the RHS warning and
# exports the field it started from; all of them negative gives the right field
# with the flux sign flipped, silently; overlapping ones give a smooth field and
# a wrong flux, silently.
#
# THE SERVED LINES ARE NOT THE AUTHOR'S. This check looked for np.cross or
# np.linalg.det anywhere in the file, and the served 3-D contract carries both
# (its interface areas and its conservation check), so no filled served contract
# ever drew it while all five cells of one round filled its mesh hole with an
# invalid split (measured). It now reads only the lines the author wrote, and
# where it can read the split itself -- corners named by their offsets, or a
# table of corner numbers whose numbering is in sight -- it judges the split:
# flat, negative (unless the author's own swap turns them round), filling the
# hex once, and meeting the next hex face to face when repeated.

_TET_ELEMENT = re.compile(r"CreateNewElement\s*\(\s*[\"'][A-Za-z]*3D4N[\"']")
_ORIENT_CHECK = re.compile(
    r"np\.cross|numpy\.cross|np\.linalg\.det|numpy\.linalg\.det|signed[_ ]?volume|orient",
    re.I)
_SIGN_SWAP = re.compile(r"(?:vol\w*|det\w*|np\.dot\(np\.cross\(.*\)|\bv\d?)\s*<=?\s*0(?:\.0*)?\s*:")
_KUHN_LINE = "(0, 1, 3, 7) (0, 1, 7, 5) (0, 2, 7, 3) (0, 2, 6, 7) (0, 4, 5, 7) (0, 4, 7, 6)"
_TET_SYMPTOMS = (
    "Kratos assembles such a mesh without naming it -- measured on this install: flat "
    "tetrahedra give NaN fluxes (with 'setting the RHS to zero' when many are flat), a mix of "
    "signs stops the solve with 'Error zero sum' or skips it with that same RHS warning, all "
    "negative flips the flux sign, and overlapping ones give a smooth field with a wrong flux")


def _own_line_numbers(content: str) -> set:
    """The lines (1-based) the author wrote: all of them, unless the file is a served
    contract, whose served lines (its holes cut) are the contract's and not the author's."""
    rows = content.splitlines()
    tpl = _template_of(content)
    if tpl is None:
        return set(range(1, len(rows) + 1))
    served = {ln.strip() for ln in _HOLE_RE.sub("", tpl).splitlines() if ln.strip()}
    return {i for i, ln in enumerate(rows, 1) if ln.strip() and ln.strip() not in served}


def _index_term(e):
    """(name, 0) for `i`, (name, 1) for `i + 1` or `1 + i`; None for anything else."""
    import ast
    if isinstance(e, ast.Name):
        return e.id, 0
    if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Add):
        for a, b in ((e.left, e.right), (e.right, e.left)):
            if isinstance(a, ast.Name) and isinstance(b, ast.Constant) and b.value in (0, 1) \
                    and not isinstance(b.value, bool):
                return a.id, int(b.value)
    return None


def _unwrap(e):
    """`int(x)` read as `x`."""
    import ast
    while (isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id == "int"
           and len(e.args) == 1 and not e.keywords):
        e = e.args[0]
    return e


def _corner(e):
    """(array, loop names, offsets) of a corner written nid[i + 1, j, k] (or with a tuple
    index); None for anything else."""
    import ast
    e = _unwrap(e)
    if not (isinstance(e, ast.Subscript) and isinstance(e.value, ast.Name)
            and isinstance(e.slice, ast.Tuple) and len(e.slice.elts) == 3):
        return None
    terms = [_index_term(x) for x in e.slice.elts]
    if any(t is None for t in terms) or len({t[0] for t in terms}) != 3:
        return None
    return e.value.id, tuple(t[0] for t in terms), tuple(t[1] for t in terms)


def _bit_numbering(tree, own):
    """{array: shift per index} where corner b is written nid[i + (b & 1), j + ((b >> 1) & 1),
    k + ((b >> 2) & 1)] in the author's lines; the shifts say which bit of b runs along which index."""
    import ast

    def shift(x):
        if isinstance(x, ast.BinOp) and isinstance(x.op, ast.BitAnd) and \
                isinstance(x.right, ast.Constant) and x.right.value == 1:
            y = x.left
            if isinstance(y, ast.Name):
                return y.id, 0
            if isinstance(y, ast.BinOp) and isinstance(y.op, ast.RShift) and \
                    isinstance(y.left, ast.Name) and isinstance(y.right, ast.Constant):
                return y.left.id, int(y.right.value)
        return None

    out = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Subscript) and getattr(node, "lineno", 0) in own
                and isinstance(node.value, ast.Name) and isinstance(node.slice, ast.Tuple)
                and len(node.slice.elts) == 3):
            continue
        s = []
        for x in node.slice.elts:
            if isinstance(x, ast.BinOp) and isinstance(x.op, ast.Add):
                s.append(shift(x.right) or shift(x.left))
            else:
                s.append(None)
        if all(s) and len({v for v, _ in s}) == 1 and sorted(b for _, b in s) == [0, 1, 2]:
            out[node.value.id] = tuple(b for _, b in s)
    return out


def _axes_of(tree, array: str):
    """Which physical axis (0 x, 1 y, 2 z) each index of `array` runs along, read from the
    line that stores a node id in it and the CreateNewNode call that made that node; None when
    the code does not say it plainly."""
    import ast
    # a coordinate passed by name (x = X0 + i * hx; CreateNewNode(c, x, y, z)) is read through
    # the assignments that give that name its value
    given = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            given.setdefault(node.targets[0].id, set()).update(
                n.id for n in ast.walk(node.value) if isinstance(n, ast.Name))
        # x, y, z = X0 + i * hx, Y0 + j * hy, Z0 + k * hz: each name from its own expression
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Tuple) and isinstance(node.value, ast.Tuple)
              and len(node.targets[0].elts) == len(node.value.elts)):
            for tn, tv in zip(node.targets[0].elts, node.value.elts):
                if isinstance(tn, ast.Name):
                    given.setdefault(tn.id, set()).update(
                        n.id for n in ast.walk(tv) if isinstance(n, ast.Name))

    def names_in(a):
        found = {n.id for n in ast.walk(a) if isinstance(n, ast.Name)}
        return found | set().union(*(given.get(n, set()) for n in found))

    made = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                node.func.attr == "CreateNewNode" and len(node.args) >= 4:
            made[ast.dump(node.args[0])] = [names_in(a) for a in node.args[1:4]]
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        t = node.targets[0]
        if not (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == array
                and isinstance(t.slice, ast.Tuple) and len(t.slice.elts) == 3
                and all(isinstance(x, ast.Name) for x in t.slice.elts)):
            continue
        coords = made.get(ast.dump(node.value))
        if coords is None:
            continue
        names = [x.id for x in t.slice.elts]
        axes = []
        for nm in names:
            hit = [q for q, used in enumerate(coords) if nm in used and not (used & (set(names) - {nm}))]
            if len(hit) != 1:
                return None
            axes.append(hit[0])
        if sorted(axes) == [0, 1, 2]:
            return tuple(axes)
    return None


def _written_splits(content: str, own: set) -> list:
    """[(how it is written, line, [tetrahedra as corner labels], {label: offsets}, axes or None)]
    for every hex split the author's lines spell out plainly: 5 or 6 tetrahedra whose corners
    are named by their offsets, or a table of corner numbers with the numbering beside it."""
    import ast
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError):
        return []
    out = []
    # corners named by their offsets: n100 = nid[i + 1, j, k]
    named, unpacked = {}, set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and getattr(node, "lineno", 0) in own
                and len(node.targets) == 1):
            continue
        t, v = node.targets[0], node.value
        pairs = [(t, v)] if isinstance(t, ast.Name) else []
        if isinstance(t, ast.Tuple) and isinstance(v, ast.Tuple) and len(t.elts) == len(v.elts):
            pairs = list(zip(t.elts, v.elts))                   # n000, n100 = nid[...], nid[...]
            unpacked.update((id(t), id(v)))                     # four names here are no tetrahedron
        for tn, tv in pairs:
            c = _corner(tv) if isinstance(tn, ast.Name) else None
            if c:
                named[tn.id] = c
    lists = []
    for node in ast.walk(tree):
        if not (isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) == 4
                and getattr(node, "lineno", 0) in own and id(node) not in unpacked):
            continue
        if all(isinstance(e, ast.Name) and e.id in named for e in node.elts):
            cs = [named[e.id] for e in node.elts]
            labels = tuple(e.id for e in node.elts)
        else:
            cs = [_corner(e) for e in node.elts]
            if not all(cs):
                continue
            labels = tuple(ast.unparse(_unwrap(e)) for e in node.elts)
        if len({(c[0], c[1]) for c in cs}) != 1:
            continue
        lists.append((node.lineno, node.col_offset, labels, cs))
    lists.sort(key=lambda r: (r[0], r[1]))
    seen, tets, offs, first = set(), [], {}, None
    for line, _col, labels, cs in lists:
        if labels in seen:
            continue
        seen.add(labels)
        tets.append(labels)
        offs.update({lb: c[2] for lb, c in zip(labels, cs)})
        first = first or (line, cs[0][0])
    if 5 <= len(tets) <= 6 and len(set(offs.values())) >= 7:
        out.append(("the tetrahedra named by their corners", first[0], tets, offs,
                    _axes_of(tree, first[1])))
    # a table of corner numbers, with corner b written nid[i + (b & 1), ...] beside it
    bits = _bit_numbering(tree, own)
    for node in ast.walk(tree):
        if not (isinstance(node, (ast.List, ast.Tuple)) and 5 <= len(node.elts) <= 6
                and getattr(node, "lineno", 0) in own):
            continue
        rows = []
        for r in node.elts:
            if not (isinstance(r, (ast.List, ast.Tuple)) and len(r.elts) == 4 and all(
                    isinstance(c, ast.Constant) and type(c.value) is int and 0 <= c.value <= 7
                    for c in r.elts)):
                break
            rows.append(tuple(c.value for c in r.elts))
        else:
            # the array that holds node ids, where its axes can be read, before any other
            ranked = sorted(bits.items(), key=lambda kv: _axes_of(tree, kv[0]) is None)
            for array, sh in ranked[:1]:
                offs = {str(b): tuple((b >> s) & 1 for s in sh) for b in range(8)}
                out.append((f"the table of corner numbers read with {array}[i + (b & 1), ...]",
                            node.lineno, [tuple(str(b) for b in r) for r in rows], offs,
                            _axes_of(tree, array)))
    if not bits:
        out += _splits_on_corner_lists(tree, own)
    return out


# ── a hex's eight corners as a LIST the author wrote, and the tetrahedra as its positions ──
#
# Measured on a coupled round: three cells wrote their own Kratos side, and each built its
# tetrahedra from a list of the eight corners of a hex, indexed by position -- two with a
# comprehension, [nid[i + dx, j + dy, k + dz] for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
# (corner 1 is then the offset along z), one with the eight offsets written out. Two of them
# indexed that list with the served Kuhn tuples, which number the corners b = i + 2*j + 4*k: all
# six tetrahedra negative, the flux sign flipped and nothing printed (one handed in a converged,
# unphysical result); the third mixed signs and overlapped ('Error zero sum'). The check read
# neither numbering and gave all three its general note.

def _offset_values(it):
    """[0, 1] for a loop over (0, 1), [0, 1], range(2) or range(0, 2); None for anything else."""
    import ast
    if isinstance(it, (ast.Tuple, ast.List)) and all(
            isinstance(c, ast.Constant) and type(c.value) is int for c in it.elts):
        vals = [c.value for c in it.elts]
    elif (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range"
          and not it.keywords and all(isinstance(a, ast.Constant) and type(a.value) is int
                                      for a in it.args)):
        vals = list(range(*[a.value for a in it.args]))
    else:
        return None
    return vals if sorted(vals) == [0, 1] else None


def _unwrap_array(e):
    """`np.array(x)`, `numpy.asarray(x)`, `list(x)` or `tuple(x)` read as `x`."""
    import ast
    while (isinstance(e, ast.Call) and len(e.args) == 1 and not e.keywords and (
            (isinstance(e.func, ast.Attribute) and e.func.attr in ("array", "asarray"))
            or (isinstance(e.func, ast.Name) and e.func.id in ("list", "tuple", "array")))):
        e = e.args[0]
    return e


def _slot_terms(e):
    """(array or None, [(loop name, offset term)] * 3) of a corner written nid[i + a, j + b, k + c]
    or (i + a, j + b, k + c), where each offset term is 0, 1 or a name; None for anything else."""
    import ast
    e = _unwrap_array(_unwrap(e))
    array = None
    if isinstance(e, ast.Subscript) and isinstance(e.value, ast.Name):
        array, e = e.value.id, e.slice
    if not (isinstance(e, (ast.Tuple, ast.List)) and len(e.elts) == 3):
        return None
    out = []
    for x in e.elts:
        x = _unwrap(x)
        if isinstance(x, ast.Name):
            out.append((x.id, 0))
        elif isinstance(x, ast.BinOp) and isinstance(x.op, ast.Add):
            for a, b in ((x.left, x.right), (x.right, x.left)):
                if isinstance(a, ast.Name) and isinstance(b, ast.Constant) and b.value in (0, 1) \
                        and not isinstance(b.value, bool):
                    out.append((a.id, int(b.value)))
                    break
                if isinstance(a, ast.Name) and isinstance(b, ast.Name):
                    out.append((a.id, b.id))                     # i + dx: dx decided by the loop
                    break
            else:
                return None
        else:
            return None
    return array, out


def _corner_lists(tree, own) -> dict:
    """{name: (line, [offsets of corner 0..7 along the array's three indices], array or None)} for
    every list of a hex's eight corners the author's lines build: written out, or by a
    comprehension over 0 and 1 (its first loop the slowest, so corner 1 is the last loop's step)."""
    import ast
    import itertools
    out = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and getattr(node, "lineno", 0) in own):
            continue
        v = _unwrap_array(node.value)
        offs, array = None, None
        if isinstance(v, (ast.List, ast.Tuple)) and len(v.elts) == 8:
            got = [_slot_terms(e) for e in v.elts]
            if all(got) and len({g[0] for g in got}) == 1 and all(
                    isinstance(o, int) for g in got for _n, o in g[1]):
                names = {tuple(n for n, _o in g[1]) for g in got}
                if len(names) == 1 and len(set(next(iter(names)))) == 3:
                    array = got[0][0]
                    offs = [tuple(o for _n, o in g[1]) for g in got]
        elif isinstance(v, (ast.ListComp, ast.GeneratorExp)) and not any(g.ifs for g in v.generators):
            got = _slot_terms(v.elt)
            loops = [(g.target.id, _offset_values(g.iter)) for g in v.generators
                     if isinstance(g.target, ast.Name)]
            if got and len(loops) == len(v.generators) and all(vals for _n, vals in loops) \
                    and len({n for n, _o in got[1]}) == 3:
                array = got[0]
                offs = []
                for combo in itertools.product(*[vals for _n, vals in loops]):
                    val = dict(zip([n for n, _v in loops], combo))
                    row = []
                    for _n, o in got[1]:
                        row.append(o if isinstance(o, int) else val.get(o))
                    if any(r is None for r in row):
                        offs = None
                        break
                    offs.append(tuple(row))
        if offs and len(offs) == 8 and len(set(offs)) == 8:
            out[node.targets[0].id] = (node.lineno, offs, array)
    return out


def _splits_on_corner_lists(tree, own) -> list:
    """The hex splits whose tetrahedra are positions in a corner list (see _corner_lists): rows of
    name[b], or a table of numbers 0..7 that a loop reads through name[...]. Each corner list's
    numbering is read from its own text; its axes from the node-id array it fills."""
    import ast
    lists = _corner_lists(tree, own)
    if not lists:
        return []
    # the array of node ids whose axes can be read: the one the corners index, else the only one
    arrays = {n.value.id for n in ast.walk(tree)
              if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)}
    readable = {a: ax for a in sorted(arrays) for ax in [_axes_of(tree, a)] if ax is not None}

    def _perm(name):
        """(array, slot order) where the corners are read component by component,
        nid[base[t][0], base[t][1], base[t][2]]; None otherwise."""
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                    and isinstance(n.slice, ast.Tuple) and len(n.slice.elts) == 3):
                continue
            comp = []
            for x in n.slice.elts:
                if (isinstance(x, ast.Subscript) and isinstance(x.slice, ast.Constant)
                        and isinstance(x.value, ast.Subscript) and isinstance(x.value.value, ast.Name)
                        and x.value.value.id == name and type(x.slice.value) is int):
                    comp.append(x.slice.value)
            if sorted(comp) == [0, 1, 2]:
                return n.value.id, tuple(comp)
        return None

    def _numbering(name):
        line, offs, array = lists[name]
        order = (0, 1, 2)
        if array is None:
            got = _perm(name)
            if got:
                array, order = got
            elif len(readable) == 1:
                array = next(iter(readable))
        offs = [tuple(o[order[s]] for s in range(3)) for o in offs]
        return line, {str(b): offs[b] for b in range(8)}, (readable.get(array) if array else None)

    # A TABLE IS READ THROUGH A CORNER LIST ONLY WHERE A LOOP SAYS SO: its rows run through a
    # loop, and an entry of a row indexes the list (corners[idx] for idx in tet). A table of six
    # rows of four corner numbers can be a hex's faces as well.
    rows_of, entries_of = {}, {}
    for n in ast.walk(tree):
        gens = ([(n.target, n.iter)] if isinstance(n, ast.For) else
                [(g.target, g.iter) for g in n.generators]
                if isinstance(n, (ast.ListComp, ast.GeneratorExp, ast.SetComp)) else [])
        for tgt, it in gens:
            key = it.id if isinstance(it, ast.Name) else id(it) if isinstance(it, (ast.List, ast.Tuple)) \
                else None
            if isinstance(tgt, ast.Name) and key is not None:
                rows_of.setdefault(key, set()).add(tgt.id)
    for tab, rows in rows_of.items():
        for r in rows:
            entries_of.setdefault(tab, set()).update(rows_of.get(r, set()))
    tied = {}                       # table (its name, or the node of a table written in a loop)
    for n in ast.walk(tree):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in lists \
                and isinstance(n.slice, ast.Name):
            for tab, ents in entries_of.items():
                if n.slice.id in ents:
                    tied.setdefault(tab, set()).add(n.value.id)
    table_name = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            table_name[id(n.value)] = n.targets[0].id
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, (ast.List, ast.Tuple)) and 5 <= len(node.elts) <= 6
                and getattr(node, "lineno", 0) in own):
            continue
        rows, named = [], set()
        for r in node.elts:
            if not (isinstance(r, (ast.List, ast.Tuple)) and len(r.elts) == 4):
                break
            if all(isinstance(c, ast.Constant) and type(c.value) is int and 0 <= c.value <= 7
                   for c in r.elts):
                rows.append(tuple(c.value for c in r.elts))
            elif all(isinstance(c, ast.Subscript) and isinstance(c.value, ast.Name)
                     and c.value.id in lists and isinstance(c.slice, ast.Constant)
                     and type(c.slice.value) is int and 0 <= c.slice.value <= 7 for c in r.elts):
                rows.append(tuple(c.slice.value for c in r.elts))
                named.update(c.value.id for c in r.elts)
            else:
                break
        else:
            if named:
                users = named if len(named) == 1 else set()
            else:
                users = tied.get(table_name.get(id(node)), set()) | tied.get(id(node), set())
            for name in sorted(users)[:1]:
                line, offs, axes = _numbering(name)
                # four corners on one face of the hex in every row: a table of faces, not of tetrahedra
                if all(any(len({offs[str(b)][s] for b in r}) == 1 for s in range(3)) for r in rows):
                    continue
                how = (f"the rows of {name}[...], the corner list of line {line}" if named else
                       f"the table of corner numbers read through the corner list {name} of line {line}")
                out.append((how, node.lineno, [tuple(str(b) for b in r) for r in rows], offs, axes))
    return out


def _split_defects(tets, offs, axes, swapped: bool) -> list:
    """What is wrong with one hex split, measured on the unit hex: flat, negative (not when the
    author swaps them round, and uniformly negative only when the index axes are known), not
    filling the hex exactly once, and not meeting the next hex face to face."""
    def vol6(t):
        a, b, c, d = (offs[x] for x in t)
        u, v, w = ([q[i] - a[i] for i in range(3)] for q in (b, c, d))
        return (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
                + u[2] * (v[0] * w[1] - v[1] * w[0]))

    n = len(tets)
    sign = 1
    if axes is not None:
        perm = list(axes)
        inv = sum(1 for i in range(3) for j in range(i + 1, 3) if perm[i] > perm[j])
        sign = -1 if inv % 2 else 1
    v = [sign * vol6(t) for t in tets]
    say = lambda t: "[" + ", ".join(t) + "]"                     # noqa: E731
    flat = [t for t, x in zip(tets, v) if x == 0]
    has = lambda k: "has" if k == 1 else "have"                 # noqa: E731
    if flat:
        return [f"{len(flat)} of its {n} tetrahedra {has(len(flat))} zero volume, the four corners "
                f"on one face of the hex (the first: {say(flat[0])})"]
    out = []
    neg = [t for t, x in zip(tets, v) if x < 0]
    if neg and not swapped:
        if len(neg) < n:
            out.append(f"{len(neg)} of its {n} tetrahedra {has(len(neg))} negative volume and the "
                       f"others positive (the first negative: {say(neg[0])})")
        elif axes is not None:
            out.append(f"all {n} of its tetrahedra have negative volume, so the flux comes out "
                       f"with the wrong sign and nothing else shows it")
    # fills the hex once: each tetrahedron turned positive, its outward faces must cancel
    # pairwise inside the hex, and the volumes must add up to it
    net, count = {}, {}
    for t, x in zip(tets, v):
        a, b, c, d = t if x > 0 else (t[0], t[1], t[3], t[2])
        for f in ((b, c, d), (a, d, c), (a, b, d), (a, c, b)):
            key = tuple(sorted(f))
            odd = sum(1 for i in range(3) for j in range(i + 1, 3) if f[i] > f[j]) % 2
            net[key] = net.get(key, 0) + (-1 if odd else 1)
            count[key] = count.get(key, 0) + 1
    on_face = lambda f: any(len({offs[p][ax] for p in f}) == 1 for ax in range(3))  # noqa: E731
    filled = sum(abs(x) for x in v) == 6 and all(
        (count[f] == 1 and on_face(f)) or (count[f] == 2 and net[f] == 0) for f in count)
    if not filled:
        out.append(f"its {n} tetrahedra do not fill the hex exactly once: they overlap or "
                   f"leave a gap")
        return out
    # repeated in every hex: the diagonal on each face must match the one on the opposite face
    for ax in range(3):
        diag = {}
        for f in count:
            if count[f] != 1 or len({offs[p][ax] for p in f}) != 1:
                continue
            side = offs[f[0]][ax]
            for p, q in ((f[0], f[1]), (f[0], f[2]), (f[1], f[2])):
                d = [abs(offs[p][i] - offs[q][i]) for i in range(3) if i != ax]
                if d == [1, 1]:
                    ends = sorted(tuple(o for i, o in enumerate(offs[r]) if i != ax) for r in (p, q))
                    diag.setdefault(side, set()).add(tuple(ends))
        if diag.get(0) != diag.get(1):
            out.append(f"repeated in every hex, it cuts the two faces normal to "
                       f"{'xyz'[axes.index(ax)] if axes else 'one axis'} along different "
                       f"diagonals, so neighbouring hexes do not meet face to face")
            break
    return out


def _numbering_note(tets, offs, axes) -> str:
    """'' unless the rows are the six served Kuhn tuples and the script numbers a hex's corners
    other than b = i + 2*j + 4*k; then which corners sit elsewhere, along x, y, z."""
    kuhn = [tuple(int(x) for x in re.findall(r"\d", t)) for t in _KUHN_LINE.split(") (")]
    try:
        rows = [tuple(int(b) for b in t) for t in tets]
    except ValueError:
        return ""
    if axes is None or sorted(rows) != sorted(kuhn):
        return ""
    at = {}
    for b in range(8):
        p = [0, 0, 0]
        for s in range(3):
            p[axes[s]] = offs[str(b)][s]
        at[b] = tuple(p)
    moved = [b for b in range(8) if at[b] != (b & 1, (b >> 1) & 1, (b >> 2) & 1)]
    if not moved:
        return ""
    say = lambda q: "(" + ", ".join(str(c) for c in q) + ")"          # noqa: E731
    return ("; its rows are the six served Kuhn tuples, but this script numbers a hex's corners "
            "differently from the fact's b = i + 2*j + 4*k: its corners "
            + ", ".join(str(b) for b in moved) + " sit at " + ", ".join(say(at[b]) for b in moved)
            + " along x, y, z, where the fact's sit at "
            + ", ".join(say((b & 1, (b >> 1) & 1, (b >> 2) & 1)) for b in moved))


def unoriented_tetrahedra(content: str) -> str:
    """'' unless the author's own lines build tetrahedra that a mesh test would reject, or build
    them with no test at all where nothing served tests them either."""
    if not isinstance(content, str) or not _TET_ELEMENT.search(content):
        return ""
    own = _own_line_numbers(content)
    rows = content.splitlines()
    own_text = "\n".join(rows[i - 1] for i in sorted(own))
    if not _TET_ELEMENT.search(own_text) and not re.search(r"CreateNewElement", own_text):
        return ""                        # the served lines build nothing; the author built nothing
    swapped = bool(_SIGN_SWAP.search(_strip_strings_and_comments(own_text)))
    fact = (f"With the corners of a hex numbered b = i + 2*j + 4*k (the offsets along x, y, z), "
            f"the Kuhn split {_KUHN_LINE} fills it exactly once, every volume positive, and meets "
            f"the next hex face to face.")
    for how, line, tets, offs, axes in _written_splits(content, own):
        bad = _split_defects(tets, offs, axes, swapped)
        if bad:
            return (f"the hex split this script writes ({how}, line {line}) is not one Kratos can "
                    f"solve right: " + "; ".join(bad) + _numbering_note(tets, offs, axes)
                    + f". {_TET_SYMPTOMS}. {fact}")
    body = _strip_strings_and_comments(content)
    if re.search(r"^\s*check_tets\s*\(", body, re.M):
        return ""                        # the served mesh test judges the split when the run starts
    if _ORIENT_CHECK.search(_strip_strings_and_comments(own_text)):
        return ""
    return (
        "this script builds 3D4N tetrahedra from its own connectivity and tests none of them. "
        f"{_TET_SYMPTOMS}; none of these messages names the mesh. Test the tetrahedra before the "
        "solve: every signed volume dot(cross(p1-p0, p2-p0), p3-p0)/6 above zero (a sign test "
        "alone passes a flat one), the volumes adding up to the domain's, and every face inside "
        "the domain shared by two tetrahedra from opposite sides (no overlap, no gap); the served "
        f"3-D Kratos contract's check_tets(mp) does all three. {fact}")


# ── the participant that answers without listening ─────────────────────────
#
# A script that writes exports.json is one side of a partitioned iteration, so
# its answer has to depend on the other side's. 23 recorded participants, in
# 20 distinct cells, write an export and never open imports.json at all. Every
# one of those cells is incomplete or malformed and none is correct.
#
# The symptom does not look like this cause. The partner's data never changes
# anything, so the iteration is a fixed point from the first step: the residual
# drops to ~1e-15 at iteration 2, the history is two rows long, and the run
# reads as a coupling that converged immediately. One recorded cell reported
# its partner "unresponsive"; the partner was answering, it was simply not
# being asked.
#
# Silent on all 33 served contracts, every one of which reads its imports
# before it solves.

# A NAME IN A STRING IS NOT A USE, AND THIS CHECK COUNTED IT AS ONE.
#
# `_EXPORTS_ANY` matched the six characters anywhere in a file, including in a
# path being READ. Measured on one coupled run: a post-processing
# script that only reads side_A/exports.json and side_B/exports.json was told
# three times, on three successive writes, "this script writes exports.json and
# never reads imports.json, so nothing the partner computes can change what it
# exports. A partitioned iteration between two sides where one side does not
# listen is a fixed point at the first step". It is not a participant, it wrote
# no export, and there was no partner for it to listen to.
#
# The correct pattern was two hundred lines above it the whole time:
# `_EXPORTS_WRITE`, which requires an actual write. The same shape as counting
# our own served text as failures, and as a coverage marker matching a name
# inside a C++ string literal -- three instances in two days.
_EXPORTS_ANY = _EXPORTS_WRITE
_IMPORTS_ANY = re.compile(r"imports\.json|\bread_imports\s*\(")


def export_without_import(content: str) -> str:
    """'' unless a participant writes an export and never reads its partner's."""
    if not isinstance(content, str) or not _EXPORTS_ANY.search(content):
        return ""
    if _IMPORTS_ANY.search(content):
        return ""
    return (
        "this script writes exports.json and never reads imports.json, so "
        "nothing the partner computes can change what it exports. A partitioned "
        "iteration between two sides where one side does not listen is a fixed "
        "point at the first step: the interface residual drops to ~1e-15 on "
        "iteration 2, the history is two rows long, and it reads as a coupling "
        "that converged immediately rather than one that never started. It also "
        "makes the partner look broken -- its data is arriving and being "
        "ignored. Read ./imports.json on EVERY run, before you assemble, and "
        "make the boundary term you assemble depend on what you read; it is "
        "`{}` on the first iteration and carries the partner's interface data "
        "on every one after that.")


# ── a participant written for a code whose contract was never asked for ────
#
# THE SECOND CODE IS WHERE COUPLED RUNS DIE, AND THIS IS WHY. Walking every
# C9/C10 work dir: 26 cells stopped with exactly ONE side exporting, and the
# silent side was side B in 16 of 17 -- DUNE 10, scikit-fem 6. Those cells were
# not out of budget when they started it: a median of 20 tool calls remained
# after the silent participant was first written, and they spent them on 265
# run_bash and 183 write_file calls between 17 cells, about 26 apiece of
# write-run-error-rewrite.
#
# What they did NOT spend them on is asking. `knowledge` was called 11 times
# across those 17 cells -- under once each -- and the decisive count is this:
# **17 of 22 never fetched the silent side's coupling contract at all** (DUNE
# 10 of 11, scikit-fem 6 of 9). The contract that exists to make that side work
# was served, and never requested.
#
# So this says so at the moment the script is written, while the 20 calls are
# still there to spend. It names an absent step and points at openPASO's own
# door; it supplies no solve, no mesh and no form.
#
# It reads the session journal -- which solver names this run has asked about
# -- and nothing else.

_CONTRACT_DOOR = {
    "fenics": "fenics", "dolfinx": "fenics", "ngsolve": "ngsolve",
    "skfem": "skfem", "dune": "dune", "kratos": "kratos",
    "fourc": "fourc", "febio": "febio", "dealii": "dealii",
}


_SERVED_MARK = "DOES NOT SERVE THIS"
# THE HOLE MARKERS DO NOT SURVIVE THE FILL. Measured on a live round: every
# filled contract on disk (45-53k chars, side A; 27-28k, side B) had lost
# "DOES NOT SERVE THIS" with the hole it framed, and read as hand-written to
# a marker-only test. The export self-check block is served in every
# contract ("keep this block") and stays through the fill.
_SERVED_SIGNS = (_SERVED_MARK, "EXPORT SELF-CHECK")


# A hole, and a contract's HOLE NAMES IN WORDS block, which the serving door moves into the list the
# contract ends with: neither is a line a served copy carries where the file has it.
_HOLE_RE = re.compile(r"# ── SOLVE ─ [^\n]*?DOES NOT SERVE THIS ─ begin.*?DOES NOT SERVE THIS ─ end"
                      r"|# ── HOLE NAMES IN WORDS ─ begin.*?# ── HOLE NAMES IN WORDS ─ end", re.S)


@functools.lru_cache(maxsize=1)
def _served_line_sets() -> tuple:
    """Per shipped contract, its served lines of 40 characters or more, holes cut."""
    out = []
    for text in _served_templates():
        lines = frozenset(ln.strip() for ln in _HOLE_RE.sub("", text).splitlines() if len(ln.strip()) >= 40)
        if len(lines) >= 20:
            out.append(lines)
    return tuple(out)


@functools.lru_cache(maxsize=1)
def _served_templates() -> tuple:
    """Every participant text openPASO serves: the shipped contract files, and the config-driven
    scaffolds the 4C and DUNE-fem doors serve, which are no file here. Measured: a copy of the
    served 4C scaffold scored 0.09 against the files alone, so a served 4C side read as
    hand-written beside a served partner, and a demoted 4C stop was never judged."""
    out = []
    try:
        for q in sorted(_PARTICIPANT_DIR.glob("participant_*.py")):
            out.append(q.read_text(errors="ignore"))
    except OSError:
        return ()
    try:
        from tools import coupling_knowledge as _ck               # noqa: PLC0415
        for key in ("fourc", "dune"):
            text = _ck._door_scaffold(key) or ""
            if text:
                out.append(text)
    except Exception:                                             # noqa: BLE001
        pass
    return tuple(out)


def _is_served_contract(text: str) -> bool:
    """A served contract carries one of the served signs AND most of the served lines
    of one shipped contract. ONE PHRASE WAS NOT ENOUGH: a hand-written side that
    imitated a single served line ("EXPORT SELF-CHECK" in a comment) read as served,
    and the check that names a hand-written side beside a served one stayed silent.
    Measured over every shipped contract: 100 % of its served lines are in it, as
    shipped and as the knowledge door serves it; the imitation shares none."""
    if not isinstance(text, str) or not any(m in text for m in _SERVED_SIGNS):
        return False
    sets = _served_line_sets()
    if not sets:
        return True                      # nothing to compare against: the sign decides
    have = {ln.strip() for ln in text.splitlines()}
    return max(len(ls & have) / len(ls) for ls in sets) >= 0.5


def hand_written_beside_a_served_side(content: str, near=None) -> str:
    """'' unless this participant is hand-written while its partner side is a served contract.

    MEASURED over twenty runs of one thermo-elastic family: seven runs wrote the served
    contract for BOTH sides and every one delivered a complete three-level result; six
    wrote it for one side and hand-wrote the other, and none delivered anything -- the
    hand-written side was small (4.8-8.5k chars) and never read imports.json.
    """
    if not isinstance(content, str) or _is_served_contract(content) or near is None:
        return ""
    near = Path(near)
    if near.suffix != ".py":
        return ""
    dirs = [near.parent]
    if near.parent.name.lower().startswith("side"):
        # A PROBE BESIDE THIS SIDE'S OWN SERVED PARTICIPANT IS NOT THIS SIDE. Measured: a
        # 900-char mesh test written into side_A/ was told "this side is HAND-WRITTEN
        # while side_A/participant_A.py is the served contract".
        for q in sorted(near.parent.glob("*.py")):
            if q.resolve() == near.resolve() or ".replaced-" in q.name:
                continue
            try:
                if _is_served_contract(q.read_text(errors="ignore")):
                    return ""
            except OSError:
                continue
        dirs = [d for d in sorted(near.parent.parent.glob("side*")) if d.is_dir() and d != near.parent]
    elif not near.name.lower().startswith("participant"):
        return ""                        # one flat folder: only a participant is a side
    partner, partner_txt = None, ""
    for d in dirs:
        for q in sorted(d.glob("*.py")):
            if q.resolve() == near.resolve() or ".replaced-" in q.name:
                continue
            try:
                txt = q.read_text(errors="ignore")
            except OSError:
                continue
            if _is_served_contract(txt) and ("imports.json" in txt or "exports.json" in txt):
                partner, partner_txt = q, txt
                break
        if partner:
            break
    if partner is None:
        return ""
    codes = _side_codes(content)
    code = codes[0] if codes else None
    try:
        rel = str(near.relative_to(near.parent.parent)) if near.parent.name.lower().startswith("side") else near.name
    except ValueError:
        rel = near.name
    try:
        who = str(partner.relative_to(partner.parent.parent))
    except ValueError:
        who = partner.name
    # THE CALL NAMES THE CONTRACT THIS SIDE IS: its role, read from the partner (and the side the
    # file states), is the variant the writer takes. Measured: a Kratos side whose script says
    # SIDE = "neumann" was offered the call without variant='neumann', which writes the Dirichlet side.
    # The fluid-structure roles are writer variants too. Measured: a FEniCSx fluid side written by hand
    # beside the served 4C structure contract was read as fsi_fluid and offered the call without it 46
    # times -- a call that writes the scalar heat contract.
    _roles = sorted({r for k, r in _side_roles(content, partner_txt) if k == code})
    _variant = (_roles[0] if len(_roles) == 1
                and (_roles[0] in _DOOR_ROLES or _roles[0].startswith("fsi_")) else "")
    call = (f"write_participant_contract(solver='{code}', "
            + (f"variant='{_variant}', " if _variant else "") + f"path='{rel}')" if code
            else f"write_participant_contract(solver=<this side's code>, path='{rel}')")
    # WHICH SERVED CHECKS THE FILE LACKS: those its own served contract carries (the partner names
    # the role), each looked for as a stop in the file -- a label left in a comment or a print is not.
    _missing = _served_checks_lacking(content, partner_txt)
    _lack = (f" It lacks the served {_named(_missing)}, so a defect "
             f"{'it' if len(_missing) == 1 else 'those'} would stop runs on silently." if _missing else "")
    # A SCRIPT THAT ALREADY RAN IS KEPT. Measured: a run replaced two sides that had run with the
    # served contracts written over them, empty, and gave up.
    _ran = (near.parent / "exports.json").is_file()
    if _ran:
        call = call.replace(f"path='{rel}'", "path=<a new file name beside it>")
    return (
        f"this side is HAND-WRITTEN ({len(content):,} chars) while {who} is the served contract."
        f"{_lack} The served contract for this side is one call, {call}; then fill only its "
        f"marked hole, in place."
        + (" This side's script has already run (its exports.json is there): write the contract to a new "
           "file and move your solve into its hole, and keep the script that ran until the new one runs."
           if _ran else ""))


_PARTICIPANT_DIR = Path(__file__).resolve().parents[2] / "data" / "coupling_participants"


def unsolved_linear_solve(content: str) -> str:
    """'' unless a PETSc solve is configured to apply one preconditioner sweep and call it a solve.

    MEASURED (petsc4py 3.24.4): LinearProblem(..., petsc_options={"ksp_type": "preonly"}) with no
    pc_type leaves |b - Ax|/|b| at 0.61 / 0.80 / 0.90 on refining Laplacians -- PETSc's default
    preconditioner is one ILU sweep and 'preonly' asks for exactly one application. With
    "pc_type": "lu" the same call reaches 1e-14. A run whose side never solved its systems
    showed every field diverging, the interface flux jump O(1) at every level, and the served
    cause list at the time (sign, points) sent it verifying causes that did not apply.
    """
    if not isinstance(content, str):
        return ""
    body = _strip_strings_and_comments(content)
    hits = []
    for m in re.finditer(r"petsc_options\s*=\s*\{([^}]*)\}", body):
        opts = m.group(1)
        if re.search(r"[\"']ksp_type[\"']\s*:\s*[\"']preonly[\"']", opts) and not re.search(
                r"[\"']pc_type[\"']\s*:\s*[\"'](?:lu|cholesky)[\"']", opts):
            hits.append(body[:m.start()].count("\n") + 1)
    if not hits:
        return ""
    where = ", ".join(str(h) for h in hits[:4])
    return (f"a PETSc solve at line {where} asks for ksp_type 'preonly' WITHOUT pc_type 'lu': that applies "
            f"one preconditioner sweep (ILU by default) and solves nothing -- measured |b - Ax|/|b| = 0.6-0.9 on "
            f"every level, every field diverging, the interface flux jump O(1). A direct solve is "
            f'petsc_options={{"ksp_type": "preonly", "pc_type": "lu"}} (add "pc_factor_mat_solver_type": '
            f'"mumps" when available); an iterative one needs a converging KSP (cg/gmres with a tolerance).')


_REDUCE = ("mean", "average", "sum", "max", "min", "amax", "amin", "median", "item")


def skfem_form_mixes_elements(content: str) -> str:
    """'' unless a scikit-fem form makes one number of a quadrature point over every element.

    In a skfem form `w.x` has shape (2, elements, quadrature points). MEASURED on a coupled round:
    a side read each quadrature point as `w.x[0, :, i]` -- that point in EVERY element -- and took
    its `.mean()` to choose a conductivity and a source value; the form then gave every element the
    same values (its three-material conductivity became two numbers and its source three across
    the whole side), its field missed its own equation everywhere, and its error did not fall with
    the mesh. Over the recorded runs the same shape (`w.x[0, :, 0].mean()`) is in two more, both
    wrong, and in no right one."""
    if not isinstance(content, str) or "skfem" not in content:
        return ""
    import ast
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError):
        return ""
    hits = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or len(fn.args.args) < 2:
            continue
        w = fn.args.args[-1].arg

        def mixes(node) -> bool:                   # w.x[<c>, :, <q>]: quadrature point q of every element
            if not (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "x" and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == w):
                return False
            sl = node.slice
            return (isinstance(sl, ast.Tuple) and len(sl.elts) == 3 and isinstance(sl.elts[1], ast.Slice)
                    and sl.elts[1].lower is None and sl.elts[1].upper is None
                    and not isinstance(sl.elts[2], ast.Slice))
        names = {}
        for st in ast.walk(fn):
            if isinstance(st, ast.Assign) and len(st.targets) == 1:
                t, v = st.targets[0], st.value
                pairs = (list(zip(t.elts, v.elts)) if isinstance(t, ast.Tuple) and isinstance(v, ast.Tuple)
                         and len(t.elts) == len(v.elts) else [(t, v)])
                for tt, vv in pairs:
                    if isinstance(tt, ast.Name) and mixes(vv):
                        names[tt.id] = vv
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            f, arg = node.func, None
            if isinstance(f, ast.Attribute) and f.attr in _REDUCE:
                if isinstance(f.value, ast.Name) and f.value.id in ("np", "numpy"):
                    arg = node.args[0] if node.args else None      # np.mean(<x>)
                else:
                    arg = f.value                                  # <x>.mean()
            elif isinstance(f, ast.Name) and f.id in ("float", "max", "min") and node.args:
                arg = node.args[0]                                 # float(<x>)
            src = None
            if arg is not None and mixes(arg):
                src = arg
            elif isinstance(arg, ast.Name) and arg.id in names:
                src = names[arg.id]
            if src is not None:
                red = f.attr if isinstance(f, ast.Attribute) else f.id
                hits.append((fn.name, node.lineno, ast.unparse(src), red))
                break
    if not hits:
        return ""
    fn, ln, expr, red = hits[0]
    more = f" (and in {len(hits) - 1} more form(s))" if len(hits) > 1 else ""
    return (f"a scikit-fem form makes one number of a quadrature point over EVERY element: in {fn}() "
            f"(line {ln}) `{expr}` reads that quadrature point in all elements, and `{red}` turns them "
            f"into one value{more}. In a form w.x has shape (2, elements, quadrature points), so "
            f"whatever the form picks this way is the same in every element -- a conductivity or a "
            f"source chosen by region becomes one value for the whole side. Compute it on the whole "
            f"arrays instead, element by element, e.g. np.where(w.x[0] < X_SPLIT, K_LEFT, K_RIGHT).")


# A statement that cannot stop the run: pass, continue, break, a print, or an assignment with no call.
_CANNOT_STOP = re.compile(r"pass|continue|break|print\(.*\)|[\w.\[\]\s,:\-+*/'\"]+[+\-*/]?=[^()\n]*")


def _template_of(content: str):
    """The shipped contract text a served file was written from, matched by its first line; None
    when the file is not a served contract or no contract starts as it does."""
    if not isinstance(content, str) or not _is_served_contract(content):
        return None
    first = content.lstrip().split("\n", 1)[0].strip()
    if not first.startswith('"""') or len(first) < 20:
        return None
    for text in _served_templates():
        if text.lstrip().split("\n", 1)[0].strip() == first:
            return text
    return None


def _squash(line: str) -> str:
    return re.sub(r"\s+", "", line)


_SIDE_TEST = re.compile(r"""\bSIDE[ \t]*==[ \t]*["'](dirichlet|neumann)["']""")


def _enclosing_side(rows: list, i: int) -> str:
    """'dirichlet' or 'neumann' when row i sits under a test of SIDE that picks that side (an
    `if SIDE == ...` header above it, or the `else:` of one), '' when it serves both."""
    ind = _indent(rows[i]) if rows[i].strip() else 10 ** 6
    for j in range(i - 1, -1, -1):
        s = rows[j]
        if not s.strip() or s.lstrip().startswith("#") or _indent(s) >= ind:
            continue
        ind = _indent(s)
        head = s.strip()
        if head.startswith(("if ", "elif ")):
            m = _SIDE_TEST.search(" ".join(r.strip() for r in rows[j:i]).split(":", 1)[0])
            if m:
                return m.group(1)
        elif head.startswith("else"):
            for k in range(j - 1, -1, -1):          # the `if` this else belongs to
                if rows[k].strip() and not rows[k].lstrip().startswith("#") and _indent(rows[k]) <= ind:
                    m = _SIDE_TEST.search(rows[k]) if _indent(rows[k]) == ind else None
                    if m and rows[k].strip().startswith(("if ", "elif ")):
                        return "neumann" if m.group(1) == "dirichlet" else "dirichlet"
                    break
        if ind == 0:
            break
    return ""


def _guards_of(template: str, inside=()) -> list:
    """The served refusals of a contract text, (condition or None, message head, label, the side it
    serves or '') each, in the served lines only (a hole's own stops are never served). A stop on a
    row inside one of the (first row, last row) spans of `inside` is left out."""
    # THE HOLE'S OWN LINES ARE NEVER SERVED: the contract reaches the agent
    # with every hole elided, so a refusal inside a hole (the "hole is not
    # filled" stop) is not a guard the agent received. Measured: the pristine
    # served 4C contract drew this finding in four cells of one round for
    # exactly that stop, and the round was abandoned after five minutes.
    hole_free = re.sub(r"# ── SOLVE ─ [^\n]*?DOES NOT SERVE THIS ─ begin.*?DOES NOT SERVE THIS ─ end",
                       "", template, flags=re.S)
    hole_lines = hole_free.splitlines()
    guards = []                      # (condition or None, message head, label)
    # BOTH SPELLINGS OF A SERVED STOP. `sys.exit(...)` stops a run as `raise SystemExit(...)` does, and
    # the served contracts use it for 1 to 8 of their refusals each -- the Kratos Neumann side's
    # held-edge stop among them -- which this check never read.
    for i, line in enumerate(hole_lines):
        if "raise SystemExit(" not in line and "sys.exit(" not in line:
            continue
        if any(r0 <= i <= r1 for r0, r1 in inside):
            continue
        m = re.search(r"(?:raise SystemExit|sys\.exit)\(\s*f?([\"'])", line)
        said = line
        if not m and re.search(r"(?:raise SystemExit|sys\.exit)\(\s*$", line) and i + 1 < len(hole_lines):
            said = hole_lines[i + 1]  # the message opens the next line (measured: the served
            m = re.match(r"\s*f?([\"'])", said)   # interface-list stops were read as unlabelled)
        text = ""
        if m:                         # up to the quote that opened it; an apostrophe inside stays
            rest = said[m.end():]
            end = rest.find(m.group(1))
            text = rest if end < 0 else rest[:end]
        full = text.split("{", 1)[0].strip()
        head = full[:30]
        if "not filled" in full or "DOES NOT SERVE" in full:
            continue
        cond = None
        for j in range(i - 1, max(-1, i - 4), -1):
            prev = hole_lines[j].strip()
            if not prev or prev.startswith("#"):
                continue
            if prev.startswith("if ") and prev.endswith(":"):
                cond = _squash(prev)
            break
        if (cond, head, full[:50]) not in [g[:3] for g in guards] and (cond or len(head) >= 12):
            guards.append((cond, head, full[:50], _enclosing_side(hole_lines, i)))
    return guards


def _guard_state(content: str, cond, head: str, body_lines=None, norm_body=None):
    """'kept', 'demoted', 'edited', 'changed', or None (gone) for one served guard in `content`."""
    # A GUARD IS ITS CONDITION, AND ITS MESSAGE ONLY SECOND. Matching the
    # message word for word over 70 characters called a refusal "removed"
    # when a run had kept it and only shortened its text (served three times
    # to a correct run, measured). A guard counts as kept when its `if`
    # condition is still there and a stop (raise, sys.exit) sits under it, or
    # when the first words of its message still sit behind a stop; demoted
    # when the condition is followed by a print instead; edited when nothing
    # under the kept condition can stop the run; gone when neither its
    # condition nor the start of its message is left.
    body_lines = content.splitlines() if body_lines is None else body_lines
    norm_body = [_squash(l) for l in body_lines] if norm_body is None else norm_body
    state = None
    if cond is not None:
        for i, nl in enumerate(norm_body):
            if nl != cond:
                continue
            # THE WHOLE BODY UNDER THE KEPT CONDITION, not its first line. Measured: a guard whose
            # raise was replaced by `pass` kept its condition and read as present. It is EDITED
            # only when every statement under it provably cannot stop the run (pass, continue,
            # break, a print, an assignment with no call in it); a call might stop it, and then
            # the state stays undecided, as before.
            ind, block = _indent(body_lines[i]), []
            for l in body_lines[i + 1:]:
                if not l.strip() or l.strip().startswith("#"):
                    continue
                if _indent(l) <= ind:
                    break
                block.append(l.strip())
            if any(re.match(r"raise\b|(?:sys\.|os\._)?exit\s*\(", s) for s in block):
                return "kept"
            if block and block[0].startswith("print("):
                state = "demoted"
            elif block and all(_CANNOT_STOP.fullmatch(s) for s in block):
                state = "edited"
            else:
                state = state or "changed"
    if len(head) >= 12:
        for mm in re.finditer(re.escape(head), content):
            before = content[max(0, mm.start() - 40):mm.start()]
            if "raise SystemExit" in before or "sys.exit" in before:
                return "kept"
            if "print(" in before:
                state = state or "demoted"
    return state


def _lost_label(label: str, state) -> str:
    if state == "demoted":
        return f"'{label}' (demoted to a print)"
    if state == "edited":
        return f"'{label}' (its condition is kept, and nothing under it stops the run)"
    if isinstance(state, tuple) and state[0] == "condition":
        return f"'{label}' (its message is kept, but {state[1]})"
    if isinstance(state, tuple) and state[0] == "narrowed":
        return (f"'{label}' (narrowed: its condition also asks {state[1]}, so it no longer stops every run "
                f"the served one stops)")
    return f"'{label}'"


def _stop_tests(text: str) -> dict:
    """{the first 30 characters of a stop's message: (the conjuncts of the `if` test the stop sits
    under, as ast dumps, and each one's source)} for every stop (raise SystemExit, sys.exit) with a
    message. A stop under no `if` has no conjuncts."""
    import ast
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return {}
    parent = {}
    for node in ast.walk(tree):
        for ch in ast.iter_child_nodes(node):
            parent[ch] = node

    def head(arg) -> str:
        parts = []
        for piece in ([arg] if not isinstance(arg, ast.BinOp) else [arg.left, arg.right]):
            if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                parts.append(piece.value)
            elif isinstance(piece, ast.JoinedStr):
                for v in piece.values:
                    if not (isinstance(v, ast.Constant) and isinstance(v.value, str)):
                        break
                    parts.append(v.value)
                break
            else:
                break
        return "".join(parts).split("{", 1)[0].strip()[:30]

    out = {}
    for node in ast.walk(tree):
        call = node.exc if isinstance(node, ast.Raise) else node.value if isinstance(node, ast.Expr) else None
        if not (isinstance(call, ast.Call) and call.args):
            continue
        f = call.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
        if name not in ("SystemExit", "exit"):
            continue
        h = head(call.args[0])
        if len(h) < 12:
            continue
        cur, test = node, None
        while cur in parent:
            up = parent[cur]
            if isinstance(up, ast.If) and cur in up.body:
                test = up.test
                break
            if isinstance(up, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                break
            cur = up
        parts = (test.values if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And)
                 else [test] if test is not None else [])
        out.setdefault(h, {ast.dump(x): ast.unparse(x) for x in parts})
    return out


def _narrowed(content: str, template: str, head: str):
    """('narrowed', the conditions added) when the stop whose message starts with `head` sits under
    every condition the served one sits under and more; None otherwise. Measured on a coupled round:
    a worker added a condition to the served export self-check so that it could not fire on a zero
    import, its zero-load side then passed it, and the served-guard check counted the stop as kept."""
    served = _stop_tests(template).get(head[:30])
    mine = _stop_tests(content).get(head[:30])
    if served is None or mine is None or not set(served) < set(mine):
        return None
    return ("narrowed", " and ".join(f"`{mine[k][:80]}`" for k in mine if k not in served))


def _edit_rows(text: str) -> list:
    """(first row, last row) of each EDIT block of a served text: its placeholders are the
    author's to set, so none of its lines is a served line."""
    rows = text.splitlines()
    out = []
    for i, ln in enumerate(rows):
        if ln.lstrip().startswith("#") and "EDIT THIS BLOCK" in ln:
            j = next((k for k in range(i + 1, len(rows)) if re.fullmatch(r"[ \t]*#[ \t]*─{20,}[ \t]*", rows[k])),
                     len(rows) - 1)
            out.append((i + 1, j + 1))
    return out


def _stop_reads(text: str, head: str, served: bool = False):
    """(the `if` test of the stop whose message starts with `head`, as source, and the module-level
    statements above it that compute what that test reads, followed back through what they read
    in turn, as {ast dump: source}); None when the text does not parse or carries no such stop.
    Only statements at the top level before the stop's own top-level statement count, and in a
    served text (`served`) none inside its EDIT block."""
    import ast
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None

    def lead(arg) -> str:
        while isinstance(arg, ast.BinOp):
            arg = arg.left
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
        if isinstance(arg, ast.JoinedStr):
            out = []
            for v in arg.values:
                if not (isinstance(v, ast.Constant) and isinstance(v.value, str)):
                    break
                out.append(v.value)
            return "".join(out)
        return ""
    parent = {}
    for node in ast.walk(tree):
        for ch in ast.iter_child_nodes(node):
            parent[ch] = node
    found = None
    for node in ast.walk(tree):
        call = node.exc if isinstance(node, ast.Raise) else node.value if isinstance(node, ast.Expr) else None
        if not (isinstance(call, ast.Call) and call.args):
            continue
        f = call.func
        if (f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else "") not in (
                "SystemExit", "exit"):
            continue
        if lead(call.args[0]).split("{", 1)[0].strip()[:30] == head[:30]:
            found = node
            break
    if found is None:
        return None
    cur, test = found, None
    while cur in parent and not isinstance(parent[cur], ast.Module):
        up = parent[cur]
        if test is None and isinstance(up, ast.If) and cur in up.body:
            test = up.test
        cur = up
    if cur not in tree.body:
        return None
    above = tree.body[:tree.body.index(cur)]
    if served:
        edit = _edit_rows(text)
        above = [st for st in above if not any(a <= st.lineno <= b for a, b in edit)]
    # ONLY THE CHECK'S OWN NAMES ARE FOLLOWED: served check code keeps what it computes in names
    # that start with an underscore (_line, _bad, _chk_vals); the names a fill sets (the mesh, the
    # interface lists, y_if) are the author's, and a served line that computes them may be rewritten.
    want = {n.id for n in ast.walk(test) if isinstance(n, ast.Name)} if test is not None else set()
    reads: dict = {}
    for st in reversed(above):
        if not isinstance(st, (ast.Assign, ast.AugAssign, ast.AnnAssign)) or st.value is None:
            continue
        tg = st.targets if isinstance(st, ast.Assign) else [st.target]
        names = {n.id for t in tg for n in ast.walk(t) if isinstance(n, ast.Name)}
        if any(nm.startswith("_") for nm in names & want):
            reads[ast.dump(st)] = ast.unparse(st)
            want |= {n.id for n in ast.walk(st.value) if isinstance(n, ast.Name)}
    return (ast.unparse(test) if test is not None else ""), reads


def _condition_changed(content: str, template: str, head: str):
    """('condition', what differs) when the stop whose message starts with `head` is still in the file
    behind a raise, but its `if` test, or a served line of the check's own above it that the test
    reads, was changed in what it tests; None otherwise, and wherever either text cannot be read. A
    test that only gained conditions is _narrowed's.

    WHAT COUNTS AS CHANGED IN WHAT IT TESTS. A test is changed when it reads every name the served
    test reads and is still another expression (a bound moved, a comparison turned around), unless
    it only dropped conditions of an `and` or gained alternatives of an `or` (then it stops every
    run the served one stops, and more); a check
    line is changed when the file sets the same name above the stop from an expression that reads
    every name the served line reads and more besides (a condition added to it). A name renamed
    away is the author's adaptation, and a line moved elsewhere cannot be judged here: neither is
    named. Measured over every recorded served participant file, each against the contracts of its
    own build: that reading names a turned-around end test and a narrowed interface-line set (one
    round) and a cross-check tolerance loosened ten- and twentyfold (two runs of another problem),
    and none of the files that only renamed their own arrays inside a check line or moved the line.

    MEASURED on a coupled round: a worker excluded the two interface end nodes from its lists,
    the served INTERFACE NODES and INTERFACE DOFS stops fired, rightly, and it edited both so that
    they passed -- the first by turning its test on the ends around, the second by narrowing the
    served line its test reads (the set of interface-line dofs) to the interior. Both messages
    still sat behind their raise, and the write check counted both stops as kept. Its Neumann load
    then carried no flux at the two end vertices."""
    import ast
    import builtins

    def names(src: str) -> set:
        try:
            return {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)} - set(dir(builtins))
        except SyntaxError:
            return set()
    served = _stop_reads(_HOLE_RE.sub("", template), head, served=True)
    mine = _stop_reads(content, head)
    if served is None or mine is None:
        return None
    (t_served, r_served), (t_mine, r_mine) = served, mine
    try:
        same_test = ast.dump(ast.parse(t_served or "0")) == ast.dump(ast.parse(t_mine or "0"))
    except SyntaxError:
        return None
    if not same_test and t_served and t_mine and names(t_served) <= names(t_mine):
        if _narrowed(content, template, head):
            return None                                  # only conditions added: said as narrowed
        # A STOP MADE STRICTER IS NOT A STOP LOST: one that dropped conditions of an `and` test,
        # or gained alternatives in an `or` test, fires on every run the served one fires on.
        ts, tm = ast.parse(t_served, mode="eval").body, ast.parse(t_mine, mode="eval").body
        if isinstance(ts, ast.BoolOp) and isinstance(tm, ast.BoolOp) and type(ts.op) is type(tm.op):
            ps, pm = {ast.dump(v) for v in ts.values}, {ast.dump(v) for v in tm.values}
            if (pm <= ps) if isinstance(ts.op, ast.And) else (ps <= pm):
                return None
        return ("condition", f"its test is now `{t_mine[:100]}`")
    by_target: dict = {}
    for src in r_mine.values():
        by_target.setdefault(src.split("=", 1)[0].strip(), []).append(src)
    for d, src in r_served.items():
        if d in r_mine:
            continue
        for other in by_target.get(src.split("=", 1)[0].strip(), []):
            if names(src) < names(other):
                added = ", ".join(sorted(names(other) - names(src)))
                return ("condition", f"the served line its test reads, `{src[:100]}`, now also reads {added}")
    return None


def _unset_guard_now_unreachable(cond, content: str) -> bool:
    """True for a served stop on an unset placeholder (`if NAME is None:`) whose placeholder the
    file now sets to something else: the stop can no longer fire, and dropping it loses nothing.
    Measured: a fill that set FULL_OUTER_DIRICHLET = True and dropped the "is unset" stop drew
    SERVED GUARD REMOVED three times."""
    m = re.fullmatch(r"if(\w+)isNone:", cond or "")
    return bool(m) and bool(re.search(rf"^{m.group(1)}\s*=\s*(?!None\b)\S", content, re.M))


def served_guard_removed(content: str, near=None) -> str:
    """'' unless this is a served contract whose refusals were deleted or demoted to prints.

    MEASURED: a run replaced the served `raise SystemExit("the TSI deck's temperature differs ...")`
    with a WARNING print, its own logs then carried "scatra-vs-tsi T mismatch 1.00e+00" at every
    level -- the structural deck's temperature was zero, its traction had no thermal stress -- and
    the coupling never converged at the finest level. A served refusal stops a run that would hand
    in a wrong number; the fill is the agent's, the guards are not.
    """
    template = _template_of(content)
    if template is None:
        return ""
    # A file that exchanges a traction or a displacement is not held to a contract that exchanges
    # neither (a heat contract a fluid side was built on): its stops are that contract's, not its own.
    if _VECTOR_EXCHANGE.search(_code_only(content)) and not _VECTOR_EXCHANGE.search(_code_only(template)):
        return ""
    guards = _guards_of(template)
    if not guards:
        return ""
    body_lines = content.splitlines()
    norm_body = [_squash(l) for l in body_lines]
    gone = []
    for cond, head, label, _side in guards:
        if _unset_guard_now_unreachable(cond, content):
            continue
        state = _guard_state(content, cond, head, body_lines, norm_body)
        if state == "kept":
            state = _narrowed(content, template, head) or _condition_changed(content, template, head) or state
        if state in ("kept", "changed"):
            continue
        gone.append(_lost_label(label or head or (cond or "")[:40], state))
    if not gone:
        return ""
    return (f"SERVED GUARD REMOVED: {len(gone)} of the {len(guards)} refusals this contract came with "
            f"are gone, demoted to a print, narrowed, edited so that they no longer stop the run, or "
            f"no longer test what the served ones test: "
            f"{'; '.join(gone[:4])}{'; ...' if len(gone) > 4 else ''}. "
            f"Each stops a run that would hand in a wrong number (measured: with the temperature "
            f"cross-check demoted, a run exported tractions with no thermal stress at every level). "
            f"Put them back as `raise SystemExit(...)`; fill the hole, keep the guards.")


def _served_functions(text: str) -> tuple:
    """(name, first row, last row, the side it serves or '') of each module-level function a served
    contract defines, outside its EDIT block: the placeholder functions there (a source, a
    coefficient) are the author's to replace, the others are served code."""
    import ast
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return ()
    rows = text.splitlines()
    edit = []
    for i, ln in enumerate(rows):
        if ln.lstrip().startswith("#") and "EDIT THIS BLOCK" in ln:
            j = next((k for k in range(i + 1, len(rows)) if re.fullmatch(r"[ \t]*#[ \t]*─{20,}[ \t]*", rows[k])),
                     len(rows) - 1)
            edit.append((i, j))
    out = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            r0, r1 = n.lineno - 1, (getattr(n, "end_lineno", None) or n.lineno) - 1
            if any(a <= r0 <= b for a, b in edit):
                continue
            # THE SIDE IT SERVES: the one side every line that uses it sits under, or '' (a helper
            # only the Neumann branch calls is not lost from a Dirichlet side that dropped it).
            uses = {_enclosing_side(rows, k) for k, ln in enumerate(rows)
                    if not r0 <= k <= r1 and re.search(rf"\b{re.escape(n.name)}\b", ln.split("#", 1)[0])}
            out.append((n.name, r0, r1, uses.pop() if len(uses) == 1 else ""))
    return tuple(out)


# ── a served block that a file written from its contract no longer carries ──
#
# MEASURED on a coupled round: a worker filled the served scikit-fem contract in place, then wrote the
# whole file again (17k characters of the contract's 37k) and lost the served interface-list stops, in
# both their forms, with the Dirichlet lines they guard. Nothing said so at the write: the file no
# longer read as a served contract (a quarter of its served lines), so the served-guard check never
# looked, and it kept its export self-check, so the no-self-check note stayed silent. The trace-held
# stop found the result in the ladder, minutes later. This names, at the write, each served stop and
# each served function of the side's own contract that the file no longer carries, read from that
# contract as served, never from a fixed list.
def served_blocks_lost(content: str, near=None) -> str:
    """'' unless this participant is written from its own served contract and has lost served blocks.

    Written from it: its code and role are known (_side_roles, one contract), and it still carries
    at least half of that contract's served blocks -- its served functions and served stops, those
    of the side the file states when it states one. A hand-written side beside a served partner is
    left to hand_written_beside_a_served_side, and the stops of a file that still reads as a served
    contract to served_guard_removed; the served functions it lost are named here either way."""
    if not isinstance(content, str) or not looks_like_participant(content):
        return ""
    table = _contracts_by_role()
    roles = _side_roles(content)
    if len(roles) != 1 or len(table.get(roles[0], ())) < 5:
        return ""
    code, role = roles[0]
    # A SIDE THAT EXCHANGES A TRACTION OR A DISPLACEMENT IS NOT JUDGED AGAINST A HEAT CONTRACT.
    # Measured on a fluid-structure round: a fluid side built on the scalar heat contract a writer had
    # handed it was told 30 times to restore that contract's stops -- among them the "flux is the
    # partner's array negated" stop, which the fluid-structure contract says does not belong there --
    # and pointed to the scalar writer. Its own contract is another; this stays silent.
    if role in _SCALAR_ROLES and _VECTOR_EXCHANGE.search(_code_only(content)):
        return ""
    _lines, _own, _carried, stops, funcs, branches, served_text = table[(code, role)]
    side = _stated_side(content)
    body_lines = content.splitlines()
    norm_body = [_squash(l) for l in body_lines]
    blocks = []                                          # (what it is called, kept?)
    # A SERVED BRANCH THAT TAKES THE PARTNER'S DATA IN is a block too: kept while at least half of
    # its served lines are in the file (a line re-worded in place is not a lost branch). Judged only
    # in a file that carries at least half of the lines only this role serves: measured, every file
    # written from a role carries 0.9 or more of them, and a file read as a role it was not written
    # from (the retired DUNE-fem heat file as the elastic role) 0.11 or less.
    have = set(norm_body)
    stripped = {ln.strip() for ln in body_lines}
    for s, label, lines in branches if 2 * len(_own & stripped) >= len(_own) else ():
        if not side or s == side:
            blocks.append((f"the served {s} branch that takes the partner's data in, {label}",
                           2 * sum(1 for ln in lines if ln in have) >= len(lines)))
    for name, s in funcs:
        if not side or not s or s == side:
            blocks.append((f"the function {name}()",
                           bool(re.search(rf"^[ \t]*def[ \t]+{re.escape(name)}[ \t]*\(", content, re.M))))
    for cond, head, label, s in stops:
        if _unset_guard_now_unreachable(cond, content):
            continue
        if not side or not s or s == side:
            state = _guard_state(content, cond, head, body_lines, norm_body)
            if state == "kept":
                state = _narrowed(content, served_text, head) or state
            blocks.append(("the stop " + _lost_label(label or head or (cond or "")[:40], state),
                           state in ("kept", "changed")))
    kept = sum(1 for _b, k in blocks if k)
    if not blocks or 2 * kept < len(blocks) or kept == len(blocks):
        return ""                                        # not written from it, or nothing lost
    if hand_written_beside_a_served_side(content, near=near):
        return ""
    served = _template_of(content) is not None           # its stops are served_guard_removed's
    lost = [b for b, k in blocks if not k and not (served and b.startswith("the stop "))]
    if not lost:
        return ""
    where = f"the served {code} contract" + (f" (variant '{role}')" if role != "base" else "")
    call = (f" write_participant_contract(solver='{code}'"
            + (f", variant='{role}'" if role != "base" else "")
            + ", path=<another file>) writes that contract beside this one to copy from."
            if role == "base" or role in _DOOR_ROLES or role.startswith("fsi_") else "")
    return (f"LOST SERVED BLOCKS: this file is written from {where} (it carries {kept} of the "
            f"{len(blocks)} served blocks that contract has for {'the ' + side if side else 'either'} "
            f"side) and no longer carries {len(lost)}: {'; '.join(lost[:6])}"
            f"{'; ...' if len(lost) > 6 else ''}. They are served code, not part of the holes: copy "
            f"each back from the contract and fill only its holes, in place.{call}")


# ── a deal.II program written in place of the served one ──
#
# MEASURED on a steady coupled round: two cells wrote the deal.II program whole in place of the served
# dealii_side.cc (13 to 15 thousand characters). One kept none of the served stops, and the 11 minutes
# it then spent on "huge values" went to the defect its lost SOLVE stop names; the other ran its own
# program without them. Nothing was said at the write: hand_written_beside_a_served_side and
# served_blocks_lost read Python participants only. This names, at the write, each served stop (a
# fail("LABEL: ...") call of the served program) and each console line the wrapper reads that the
# written program no longer carries, read from the served program itself, never from a fixed list.
_SERVED_STOP = re.compile(r'\bfail\s*\(\s*(?:std::string\s*\(\s*)?"([A-Z][A-Z_ ]*[A-Z]):')
_SERVED_LINES = {"dealii_side.cc": (("NDOF = ", '"NDOF = <n>"'), ('"VOLUME_SOURCE[ "]', '"VOLUME_SOURCE on|off"')),
                 "dealii_side_transient.cc": (("NDOF = ", '"NDOF = <n>"'), ('"SOURCE[ "]', '"SOURCE on|off"')),
                 "dealii_side_elastic.cc": (("NDOF = ", '"NDOF = <n>"'), ('"BODY_FORCE[ "]', '"BODY_FORCE on|off"'))}


def _program_served_as(name: str, near=None):
    """The served deal.II program a written .cc stands in for: its own name when it is one, else the
    name a participant script beside it builds (DEALII_SRC); None for any other file (a probe)."""
    if name in _SERVED_LINES:
        return name
    if near is None or not name:
        return None
    try:
        scripts = sorted(Path(near).parent.glob("*.py"))
    except OSError:
        return None
    for q in scripts:
        try:
            t = q.read_text(errors="ignore")
        except OSError:
            continue
        m = re.search(r"""^DEALII_SRC\s*=\s*["']([^"']+)["']""", t, re.M)
        if m and Path(m.group(1)).name == name:
            # THE WRAPPER SAYS WHICH PROGRAM IT BUILDS: the elastic one reads "BODY_FORCE on". Measured on
            # a coupled elastic round: the served elastic program, unchanged, was judged against the heat
            # program 99 times in four cells and told it had lost SOURCE, VOLUME_SOURCE and MATRIX.
            if re.search(r"^N_STEPS\s*=", t, re.M):
                return "dealii_side_transient.cc"
            return "dealii_side_elastic.cc" if "BODY_FORCE on" in t else "dealii_side.cc"
    return None


def _served_stop_conditions(served: str) -> list:
    """[(label, the condition of the `if` that guards it)] for each served stop of a program: the
    nearest `if (` before the fail(...) call, its parentheses balanced."""
    out = []
    for m in re.finditer(r'\bfail\s*\(\s*(?:std::string\s*\(\s*)?"([A-Z][A-Z_ ]*[A-Z]):', served):
        start = served.rfind("if (", 0, m.start())
        if start < 0:
            continue
        depth, i = 0, start + 3
        while i < m.start():
            if served[i] == "(":
                depth += 1
            elif served[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth == 0 and i < m.start():
            out.append((m.group(1), served[start + 4:i]))
    return out


def served_program_checks_lost(content: str, name: str = "", near=None) -> str:
    """'' unless this .cc is the program a served deal.II wrapper builds and lacks served stops or
    the console lines the wrapper reads. A stop demoted to a print is lost too: only a fail(...)
    call stops the run."""
    if not isinstance(content, str) or ("#include <deal.II/" not in content and "dealii::" not in content):
        return ""
    name = Path(name).name if name else ""
    program = _program_served_as(name, near)
    if program is None:
        return ""
    try:
        from .coupling_knowledge import served_program                          # noqa: PLC0415
        served, err = served_program(program)
    except Exception:                                                            # noqa: BLE001
        return ""
    if err or not served:
        return ""
    from collections import Counter                                              # noqa: PLC0415
    body = re.sub(r"//[^\n]*|/\*.*?\*/", "", content, flags=re.S)
    want, have = Counter(_SERVED_STOP.findall(served)), Counter(_SERVED_STOP.findall(body))
    lost = [lab if k == 1 else f"{lab} ({'twice' if k == 2 else f'{k} times'})" if not have[lab]
            else f"{lab} ({k - have[lab]} of {k})" for lab, k in want.items() if have[lab] < k]
    gone = [said for pat, said in _SERVED_LINES[program] if re.search(pat, served) and not re.search(pat, body)]
    # A STOP WHOSE TEST WAS LOOSENED IS LOST TOO. Measured on a transient coupled round: a program raised
    # the tolerances of two served STEP SYSTEM stops (1e-9 to 1e-4 to 0.05 of the matrix sum, 0.15 to 0.5
    # of the load sum) and was told nothing until it commented the second one out; its field went on with
    # a load four times the step's. A stop still carrying its label is judged by the test before it too.
    flat = re.sub(r"\s+", "", body)
    changed = []
    for lab, cond in _served_stop_conditions(served):
        if have[lab] and re.sub(r"\s+", "", cond) not in flat and (lab, " ".join(cond.split())) not in changed:
            changed.append((lab, " ".join(cond.split())))
    if not lost and not gone and not changed:
        return ""
    if not lost and not gone:
        return (f"SERVED CHECKS CHANGED: {name} is the program the served deal.II wrapper builds, and it carries "
                f"every stop of the served {program}, but "
                + "; ".join(f"the test before its {lab} stop is no longer `if ({cond[:140]})`"
                            for lab, cond in changed[:3])
                + ". A served stop whose test was changed -- a tolerance raised, a term dropped -- no longer stops "
                  "what it was served to stop. Copy the served lines back as they were and change your hole instead: "
                  "the stop measured what your hole left.")
    kept = sum(min(have[lab], k) for lab, k in want.items())
    var = {"dealii_side_transient.cc": ", variant='transient'",
           "dealii_side_elastic.cc": ", variant='elasticity'"}.get(program, "")
    return (f"SERVED CHECKS LOST: {name} is the program the served deal.II wrapper builds, and it carries "
            f"{kept} of the {sum(want.values())} stops of the served {program}"
            + (f"; it no longer carries these: {_named(lost)}" if lost else "")
            + (f"{'; nor' if lost else '; it lacks'} the console line{'s' if len(gone) > 1 else ''} "
               f"{_named(gone)} the wrapper reads" if gone else "")
            + ". They are served code, not part of the five holes: each stops a run whose field would be "
              "wrong (measured: a program rewritten whole lost every stop, and its next 11 minutes went to "
              "the defect the lost SOLVE stop names). Copy them back from the served program, which "
              f"write_participant_contract(solver='dealii'{var}, path='<a new folder>/participant_copy.py') "
              "writes into that folder, and keep your own code in its holes.")


def _solvers_asked_about(near=None):
    """Solver names this session has fetched knowledge for, or None if unobservable.

    NONE MEANS "CANNOT SEE", NOT "NOTHING ASKED". The journal is filled inside the
    openPASO server, a separate process; this runs in whatever process hosts the
    write hook, where the in-process journal is usually empty. So an empty
    in-process journal proves nothing. The live record the server appends to its
    session directory (a run's work/.openpaso_sessions) is looked for by walking up
    from the file being judged. Only when some record is visible is an answer
    given; otherwise None, and the caller must stay silent.
    """
    out, seen = set(), False
    try:
        from core.session_journal import get_journal          # noqa: PLC0415
        evs = getattr(get_journal(), "events", None) or []
    except Exception:                                          # noqa: BLE001
        evs = []
    for e in evs:
        seen = True
        if getattr(e, "event_type", "") == "knowledge_lookup":
            s = (getattr(e, "solver", "") or "").strip().lower()
            if s:
                out.add(s)
    if near is not None:
        try:
            import json as _json                                  # noqa: PLC0415
            from pathlib import Path as _Path                      # noqa: PLC0415
            d = _Path(near).resolve()
            d = d if d.is_dir() else d.parent
            for _ in range(8):
                sess = d / ".openpaso_sessions"
                if sess.is_dir():
                    for f in sess.glob("session_*.jsonl"):
                        for line in f.read_text(errors="ignore").splitlines():
                            try:
                                row = _json.loads(line)
                            except ValueError:
                                continue
                            seen = True
                            if row.get("event_type") == "knowledge_lookup":
                                s = str(row.get("solver") or "").strip().lower()
                                if s:
                                    out.add(s)
                if d.parent == d:
                    break
                d = d.parent
        except Exception:                                      # noqa: BLE001
            pass
    return out if seen else None


def contract_never_fetched(content: str, near=None) -> str:
    """'' unless this participant is for a code whose contract was never asked for.

    `near` is the path of the file being judged; it is how the live session record
    is found. With no observable record the answer is '' -- see
    _solvers_asked_about.
    """
    if not isinstance(content, str) or not _EXPORTS_ANY.search(content):
        return ""                                   # not a participant
    codes = [c for c in backends_in(content) if c in _CONTRACT_DOOR]
    if not codes:
        return ""
    asked = _solvers_asked_about(near)
    if asked is None:
        return ""                                   # cannot see the journal: say nothing
    missing = [c for c in codes if c not in asked]
    if not missing:
        return ""
    c = missing[0]
    return (
        f"this is a coupling participant for {c}, and this run has not once "
        f"asked what {c}'s contract looks like. There is one, it is served, and "
        f"it carries this code's interface handshake, its sign convention, its "
        f"consistent flux recovery and the API calls that stop its runs: "
        f"knowledge(topic='coupling', solver='{c}'). MEASURED, and it is where "
        f"coupled runs die: of the recorded runs that got one side exporting "
        f"and never the other, 17 of 22 had never fetched the silent side's "
        f"contract, and they averaged 26 write-run-error calls on it afterwards "
        f"with about 20 still in hand. Asking first costs one call.")
