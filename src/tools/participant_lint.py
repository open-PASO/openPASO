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
    ("skfem", r"\bmesh\.extend\s*\(|\bm\.extend\s*\(",
     "AttributeError: 'MeshTri1' object has no attribute 'extend'",
     "refine with mesh.refined(), or mesh.refined(n) for n halvings; the "
     "meshes are immutable so use the RETURN value"),
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
    ("skfem", r"\bfind_dofs\s*\(",
     "AttributeError: 'FacetBasis' object has no attribute 'find_dofs'",
     "every basis has get_dofs, not find_dofs"),
    ("skfem", r"\bmesh\.f\b(?!acets|2)",
     "AttributeError: 'MeshTri1' object has no attribute 'f'",
     "the facet node table is mesh.facets (2 x n_facets); mesh.p, mesh.t, mesh.t2f and mesh.f2t are the others"),
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
     "a vertex's dof is fes.GetDofNrs(NodeId(VERTEX, v.nr))[0]; the free set is fes.FreeDofs()"),
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
    ("ngsolve", r"\.vec\s*\[[^]]*FreeDofs",
     "TypeError: __getitem__(): incompatible function arguments (a BitArray is not an index)",
     "take numbers out with v.FV().NumPy() or np.array(v), and assign through v.data"),
    ("ngsolve", r"CoefficientFunction\s*\(\s*(lambda|[A-Z_]+SRC)|(?<![A-Za-z_])CF\s*\(\s*lambda",
     "TypeError: incompatible constructor arguments (a Python function is not a CoefficientFunction)",
     "sample the function at the vertices into a P1 GridFunction and integrate that; a GridFunction "
     "IS a CoefficientFunction"),
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
     "ufl.Constant takes a DOMAIN, not a value: a boundary value is a plain scalar "
     "(default_scalar_type(0.0)) and a constant inside a form is fem.Constant(mesh, value)"),
    ("fenics", r"ufl\.FiniteElement\s*\(",
     "AttributeError: module 'ufl' has no attribute 'FiniteElement'",
     "fem.functionspace(mesh, ('Lagrange', 1)), or basix.ufl.element(...)"),
    ("fenics", r"fem\.VectorFunctionSpace\s*\(",
     "AttributeError: module 'dolfinx.fem' has no attribute 'VectorFunctionSpace'",
     "fem.functionspace(mesh, ('Lagrange', 1, (mesh.geometry.dim,)))"),
    ("fenics", r"\.geometric_dimension\s*\(",
     "AttributeError: a UFL argument has no geometric_dimension()",
     "take the dimension from the mesh: mesh.geometry.dim"),
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
     "Kratos error: LaplacianElement2D4N is not registered",
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
    # module-level loads only: function bodies may legitimately read module names defined later
    missing = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load) and sub.id not in bound:
                missing.setdefault(sub.id, sub.lineno)
    return [f"`{n}` is used at line {ln} and never defined in this file -- the run stops with "
            f"NameError/UnboundLocalError there. The served contract lists the names its surviving "
            f"lines need; every one of them has to come out of your solve."
            for n, ln in sorted(missing.items(), key=lambda kv: kv[1])]


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


_ERROR_FIXES: tuple = (
    # THE LARGEST SINGLE UNGATED FAILURE IN THE RECORDED SET: 271 occurrences
    # across 120 cells. Three spellings of one mistake -- a Python function or
    # lambda, a string, or a sympy expression handed to a CoefficientFunction.
    # It lands hardest where it hurts most: the two problem families that have
    # never once produced a coupled level carry it 5 times each per cell.
    ("Cannot make CoefficientFunction from",
     "ngsolve: a CoefficientFunction is built from NGSolve's OWN symbolic "
     "coordinates, not from a Python callable, a string or a sympy expression: "
     "`from ngsolve import x, y, z` then write the expression in them -- "
     "CF(x*x + y*y), CF(sin(pi*x)*cos(pi*y)) with sin/cos/exp/log imported "
     "from ngsolve too. A lambda, a def, 'x^2 + y^2' and a sympy object all "
     "raise this. Measured on this install"),
    ("Did you mean: 'dx'",
     "ngsolve: the spatial coordinates are NAMES YOU IMPORT, and `dx` is the "
     "volume measure, not a coordinate: `from ngsolve import x, y, z`. "
     "Measured on this install -- ngsolve exports all three"),
    # MEASURED in 22 distinct recorded cells, more than any other UFL mistake.
    # It reads as a backend crash, not a syntax error: one cell reported
    # "segmentation fault during 3D grid/UFL compilation" and abandoned the
    # run with the backend declared broken, while the traceback said this and
    # the script already imported Or on its own import line.
    ("unsupported operand type(s) for |",
     "UFL: a condition is combined with Or(a, b) and And(a, b), imported from "
     "ufl -- Python's | and & are not defined on UFL conditions. conditional("
     "Or(lt(x[0], a), gt(x[0], b)), 1, 0); nest Or for three or more"),
    ("unsupported operand type(s) for &",
     "UFL: a condition is combined with And(a, b) and Or(a, b), imported from "
     "ufl -- Python's & and | are not defined on UFL conditions"),
    ("'Dofs' object is not callable",
     "scikit-fem: basis.dofs is an ATTRIBUTE. Select with basis.get_dofs(facets=...) or "
     "basis.get_dofs(lambda x: ...), then .flatten()"),
    ("missing 1 required positional argument: 'ubasis'",
     "scikit-fem: a form takes its basis POSITIONALLY -- laplace.assemble(basis), or asm(laplace, basis)"),
    ("Attribute 'y' not found in 'w'",
     "scikit-fem: inside a form the coordinates are w.x[0] and w.x[1]; there is no w.y"),
    ("has no attribute 'init_rect'",
     "scikit-fem: a rectangle is MeshTri.init_tensor(np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1))"),
    ("unexpected keyword argument 'doforder'",
     "scikit-fem: Basis(mesh, element) takes no doforder keyword"),
    ("has no attribute 'find_dofs'",
     "scikit-fem: every basis has get_dofs, not find_dofs"),
    ("cannot import name 'trace' from 'ngsolve'",
     "NGSolve: the tensor helpers are CAPITALISED and the set is not UFL's. Measured on this install, ngsolve EXPORTS Sym, Trace, Det, Grad, grad, div, Id, InnerProduct, OuterProduct, Inv, Cof -- and does NOT export sym, trace, det, Div, Identity or Transpose. The strain is Sym(Grad(u)), the trace Trace(...), the identity Id(2)"),
    ("cannot import name 'det' from 'ngsolve'",
     "NGSolve: the tensor helpers are CAPITALISED and the set is not UFL's. Measured on this install, ngsolve EXPORTS Sym, Trace, Det, Grad, grad, div, Id, InnerProduct, OuterProduct, Inv, Cof -- and does NOT export sym, trace, det, Div, Identity or Transpose. The strain is Sym(Grad(u)), the trace Trace(...), the identity Id(2)"),
    ("cannot import name 'Identity' from 'ngsolve'",
     "NGSolve: the tensor helpers are CAPITALISED and the set is not UFL's. Measured on this install, ngsolve EXPORTS Sym, Trace, Det, Grad, grad, div, Id, InnerProduct, OuterProduct, Inv, Cof -- and does NOT export sym, trace, det, Div, Identity or Transpose. The strain is Sym(Grad(u)), the trace Trace(...), the identity Id(2)"),
    ("cannot import name 'Transpose' from 'ngsolve'",
     "NGSolve: the tensor helpers are CAPITALISED and the set is not UFL's. Measured on this install, ngsolve EXPORTS Sym, Trace, Det, Grad, grad, div, Id, InnerProduct, OuterProduct, Inv, Cof -- and does NOT export sym, trace, det, Div, Identity or Transpose. The strain is Sym(Grad(u)), the trace Trace(...), the identity Id(2)"),
    ("cannot import name 'SetLogLevels' from 'ngsolve'",
     "NGSolve: there is no SetLogLevels. Verbosity is a module global -- "
     "import ngsolve (the MODULE) and set ngsolve.ngsglobals.msg_level = 3 "
     "before the solve, or the run log carries none of this code's output"),
    ("cannot import name 'DirichletBC' from 'ngsolve'",
     "NGSolve: there is no DirichletBC class -- the condition belongs to the "
     "SPACE, as H1(mesh, order=..., dirichlet='<boundary names separated by "
     "|>'), and the VALUES are written into the GridFunction's vector at those "
     "dofs. fes.FreeDofs() then excludes them at solve time"),
    ("'ngsolve.comp.H1' object has no attribute 'VDim'",
     "NGSolve: the component count of a space is fes.dim (a vector space is "
     "H1(mesh, order=1, dim=2)); there is no VDim"),
    ("No module named 'KratosMultiphysics'",
     "Kratos: this code has its own interpreter. You ran it with one that does not have it. discover(query='list') names the interpreter or binary of every backend on this install -- use that exact path in the participant's `command`"),
    ("No module named 'dolfinx'",
     "FEniCSx: this code has its own interpreter. You ran it with one that does not have it. discover(query='list') names the interpreter or binary of every backend on this install -- use that exact path in the participant's `command`"),
    ("No module named 'ngsolve'",
     "NGSolve: this code has its own interpreter. You ran it with one that does not have it. discover(query='list') names the interpreter or binary of every backend on this install -- use that exact path in the participant's `command`"),
    ("No module named 'skfem'",
     "scikit-fem: this code has its own interpreter. You ran it with one that does not have it. discover(query='list') names the interpreter or binary of every backend on this install -- use that exact path in the participant's `command`"),
    ("No module named 'dune'",
     "DUNE-fem: this code has its own interpreter. You ran it with one that does not have it. discover(query='list') names the interpreter or binary of every backend on this install -- use that exact path in the participant's `command`"),
    ("No module named 'netgen'",
     "NGSolve/netgen: this code has its own interpreter. You ran it with one that does not have it. discover(query='list') names the interpreter or binary of every backend on this install -- use that exact path in the participant's `command`"),
    ("has no attribute 'MeshTags'",
     "FEniCSx: facet tags come from the lowercase FUNCTION in the mesh module -- "
     "dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, indices, values) with int32 "
     "arrays; there is no fem.MeshTags and no dolfinx.MeshTags"),
    ("Invalid format specifier",
     "Python f-string: a literal { or } inside an f-string starts a "
     "replacement field, so JSON written with an f-string breaks on its own "
     "braces. DOUBLE them -- {{ and }} -- or build the dict and json.dumps it. "
     "Measured: f'{\"a\":\"x\"}' raises this, f'{{\"a\":\"x\"}}' returns the "
     "string"),
    ("Kratos.ModelPart: No constructor defined!",
     "Kratos: a ModelPart is created BY a Model, never constructed directly -- "
     "model = KM.Model(); mp = model.CreateModelPart('<name>'). Measured: "
     "KM.ModelPart('x') raises this, model.CreateModelPart('x') returns one"),
    ("'CellBasis' object has no attribute 'ndof'",
     "scikit-fem: a basis counts its degrees of freedom with basis.N, not "
     ".ndof -- measured: N is 16 for a scalar P1 basis on a mesh where a "
     "vector basis of the same mesh gives 32, and len(basis.zeros()) is the "
     "same number. basis.nelems is the ELEMENT count, a different quantity"),
    ("CellBasis.__init__() missing 1 required positional argument: 'elem'",
     "scikit-fem: Basis(mesh, element) needs the ELEMENT -- Basis(mesh, "
     "ElementTriP1()) for a scalar P1 field, Basis(mesh, "
     "ElementVector(ElementTriP1())) for a vector one"),
    ("compute_colliding_cells(): incompatible function arguments",
     "FEniCSx: the point array must have THREE columns even in 2-D -- "
     "np.column_stack([x, y, np.zeros_like(x)]) -- for both "
     "geometry.compute_collisions_points(bb_tree, pts) and "
     "geometry.compute_colliding_cells(mesh, candidates, pts). Measured: an "
     "(n, 2) array raises this, an (n, 3) array returns the AdjacencyList"),
    ("compute_collisions_points(): incompatible function arguments",
     "FEniCSx: the point array must have THREE columns even in 2-D -- "
     "np.column_stack([x, y, np.zeros_like(x)]). Measured: an (n, 2) array "
     "raises this"),
    ("Boundaries(): incompatible function arguments",
     "NGSolve: mesh.Boundaries takes the boundary NAME as a string, not an "
     "index -- mesh.Boundaries('left') or a regex over the names, "
     "mesh.Boundaries('left|top'). The names are the ones you passed as bcs= "
     "when you built the geometry"),
    ("GetDofs(): incompatible function arguments",
     "NGSolve: to get the dof numbers of a mesh node use "
     "fes.GetDofNrs(NodeId(VERTEX, i)), which returns a tuple. GetDofs takes a "
     "Region (mesh.Boundaries('name')), not an integer"),
    ("Set(): incompatible function arguments",
     "NGSolve: GridFunction.Set takes a FIELD -- a CoefficientFunction, a "
     "symbolic expression or a scalar -- not a dof array. To write raw values "
     "use gfu.vec.FV().NumPy()[:] = <array>, or index gfu.vec directly. "
     "Measured: gfu.Set(np.zeros(fes.ndof)) raises this, gfu.Set(x*y) works"),
    (".LeafGrid' object has no attribute 'geometry'",
     "DUNE-fem: a grid view has no geometry -- geometry belongs to an ENTITY. "
     "Iterate and ask the element: for e in gridView.elements: "
     "e.geometry.center, e.geometry.volume, e.geometry.corners. Measured, the "
     "grid view itself offers elements, vertices, size, dimension, dimGrid and "
     "indexSet"),
    (".LeafGrid' object has no attribute 'leaves'",
     "DUNE-fem: the entities of a grid view are gridView.elements (and "
     "gridView.vertices); there is no .leaves. Measured on this install"),
    ("cannot import name 'Eq' from 'ufl'",
     "UFL: there is no Eq, and DO NOT take the interpreter's suggestion here. "
     "It proposes `eq`, which is a BOOLEAN comparison for use inside "
     "ufl.conditional -- measured, ufl.eq(a, L) returns an EQ object without "
     "complaining, and it is not your variational problem. A variational "
     "equation is written with the operator: `a == L`, which returns a ufl "
     "Equation"),
    ("cannot import name 'Variable' from 'ufl'",
     "UFL: the spelling is lowercase, ufl.variable(expr), and here the "
     "interpreter's suggestion is right -- it marks an expression so you can "
     "differentiate with respect to it via ufl.diff"),
    ("'numpy.ndarray' object has no attribute 'grad'",
     "scikit-fem: inside a form the arguments are DiscreteField objects, which "
     "DO carry .grad -- but the helpers return plain arrays, so grad(u).grad is "
     "this error. Take the derivative once: from skfem.helpers import grad, "
     "then grad(u) is the (dim, ...) array and grad(u)[0], grad(u)[1] are its "
     "components"),
    ("cannot import name 'sym' from 'skfem.helpers'",
     "scikit-fem: the symmetric gradient is one helper, sym_grad, not sym "
     "composed with grad. Measured exports of skfem.helpers: grad, sym_grad, "
     "div, curl, cross, dot, ddot, inner, mul, prod, trace, transpose, det, "
     "inv, eye, identity, jump, d, dd, ddd, dddd, dddot, zeros_like -- and no "
     "sym"),
    ("cannot import name 'SetBC' from 'ngsolve'",
     "NGSolve: there is no SetBC. The Dirichlet boundary is an ARGUMENT of the "
     "space -- H1(mesh, order=..., dirichlet='<names|joined>') -- the values go "
     "into the GridFunction's vector at those dofs, and fes.FreeDofs() is what "
     "the solve inverts on"),
    ("first argument should be a ufl equation",
     "DUNE-fem: the scheme takes an EQUATION, not a form -- galerkin(a == L, ...) "
     "with `a` the bilinear form and `L` the linear one. Measured: galerkin(a) "
     "raises this, galerkin(a == L) returns a Scheme. For a form with no "
     "right-hand side write `a == 0`"),
    ("solve() missing 1 required positional argument: 'target'",
     "DUNE-fem: the scheme solves INTO a discrete function -- uh = "
     "space.interpolate(0, name='u'); scheme.solve(target=uh). It does not "
     "return the solution, it fills the target and returns an info dict"),
    ("'Kratos.Properties' object has no attribute 'Value'",
     "Kratos: a Properties object uses SetValue/GetValue, not Value -- "
     "props.SetValue(KM.DENSITY, 2.5) then props.GetValue(KM.DENSITY), and "
     "props[KM.DENSITY] reads the same value. Measured on this install"),
    ("object has no attribute 'GetOCCGeometry'",
     "netgen: a CSGeometry is not an OCC geometry and cannot be converted. OCC "
     "lives in a different module (netgen.occ.OCCGeometry); for a rectangle use "
     "netgen.geom2d.SplineGeometry -- geo.AddRectangle((X0, Y0), (X1, Y1), "
     "bcs=(bottom, right, top, left)) then geo.GenerateMesh(maxh=...)"),
    ("No module named 'kratos'",
     "Kratos: the import is `import KratosMultiphysics as KM`, capitalised and "
     "one word -- there is no lowercase `kratos` module. It also needs its own "
     "interpreter, which discover(query='list') prints"),
    ("'ngsolve.comp.H1' object has no attribute 'SetEssentialBC'",
     "NGSolve: a space has no SetEssentialBC. The Dirichlet boundary is an "
     "ARGUMENT of the space -- H1(mesh, order=..., dirichlet='<names|joined>') "
     "-- the values go into the GridFunction's vector, and fes.FreeDofs() is "
     "what the solve inverts on"),
    ("cannot import name 'abs' from 'ufl'",
     "UFL: there is no ufl.abs -- use the Python builtin abs(expr) on a UFL "
     "expression. Measured, ufl DOES export sign, sqrt, conditional, eq and "
     "variable, so the lowercase spelling is right for those and not for abs"),
    ("module 'dolfinx.fem' has no attribute 'petsc'",
     "FEniCSx: dolfinx.fem.petsc is a SUBMODULE that is not imported with its "
     "parent -- add `import dolfinx.fem.petsc` (or `from dolfinx.fem.petsc "
     "import LinearProblem`) and the attribute appears"),
    ("cannot import name 'NewtonSolver' from 'dolfinx.nls'",
     "FEniCSx: the Newton solver is in the petsc submodule -- `import "
     "dolfinx.nls.petsc` then dolfinx.nls.petsc.NewtonSolver. The same shape as "
     "dolfinx.fem.petsc: importing the parent does not bring it in"),
    ("'MatrixCSR' object has no attribute 'assemble'",
     "FEniCSx: fem.assemble_matrix returns a MatrixCSR, whose finalise step is "
     "A.scatter_reverse() (then A.to_dense() or A.data). `.assemble()` belongs "
     "to the PETSc matrix that fem.petsc.assemble_matrix returns instead"),
    ("cannot import name 'plane_strain' from 'skfem.models.elasticity'",
     "scikit-fem: there is no plane_strain, because lame_parameters IS the "
     "plane-strain pair. Measured exports of skfem.models.elasticity: "
     "BilinearForm, ddot, eye, lame_parameters, linear_elasticity, "
     "linear_stress, plane_stress, sym_grad, trace. For PLANE STRAIN write "
     "linear_elasticity(*lame_parameters(E, nu)); plane_stress(E, nu) is the "
     "separate plane-stress helper"),
    ("'Dofs' object has no attribute 'shape'",
     "scikit-fem: get_dofs returns a DofsView, not an array. It has .flatten() "
     "for the index array, .all() (optionally .all('u^1')) and .nodal['u^1'] / "
     ".nodal['u^2'] for a vector basis -- and no .shape and no .size"),
    ("get_dofs() got an unexpected keyword argument 'component'",
     "scikit-fem: a vector basis selects a component BY NAME, not by component=. "
     "d = basis.get_dofs(<facets or predicate>); the x dofs are d.nodal['u^1'] and the y dofs "
     "d.nodal['u^2'] (one-based, there is no u^0); d.all() and d.flatten() give both interleaved"),
    ("'MeshTri1' object has no attribute 'extend'",
     "scikit-fem: refine with mesh.refined() -- mesh.refined(n) for n halvings. The meshes are "
     "immutable, so use the RETURN value; there is no mesh.extend"),
    ("'MeshTri1' object has no attribute 'f'",
     "scikit-fem: the facet node table is mesh.facets; mesh.p, mesh.t, mesh.t2f and mesh.f2t are the others"),
    ("has no attribute 'AddVertex'",
     "NGSolve: a rectangle is geo.AddRectangle((X0, Y0), (X1, Y1), bcs=(bottom, right, top, left)); "
     "single points are AddPoint(x, y) with SEPARATE coordinates"),
    ("has no attribute 'AddRect'",
     "NGSolve: the call is AddRectangle, spelled in full"),
    ("object has no attribute 'Faces'",
     "NGSolve: the mesh iterators are lowercase properties -- mesh.vertices, mesh.faces, mesh.edges"),
    ("object has no attribute 'Vertices'",
     "NGSolve: the mesh iterators are lowercase properties -- mesh.vertices, mesh.faces, mesh.edges"),
    ("MeshNode' object has no attribute",
     "NGSolve: a MeshNode carries .nr and .point; a vertex's dof is fes.GetDofNrs(NodeId(VERTEX, v.nr))[0]"),
    ("cannot import name 'inverse' from 'ngsolve'",
     "NGSolve: `inverse` is a KEYWORD of Inverse, not an import -- "
     "a.mat.Inverse(fes.FreeDofs(), inverse='sparsecholesky')"),
    ("cannot import name 'sym' from 'ngsolve'",
     "NGSolve: there is no `sym`, `trace` or `Div`. Sym and Trace exist but act on a MATRIX -- the "
     "strain is Sym(Grad(u)), never Sym(u) (that raises 'Sym of non-matrix called'); the "
     "divergence is lowercase div(u); and the identity is Id(mesh.dim), never Id()"),
    ("Sym of non-matrix called",
     "NGSolve: Sym takes a matrix, so build the strain from the gradient -- Sym(Grad(u)), not Sym(u)"),
    ("cannot import name 'Div' from 'ngsolve'",
     "NGSolve: the divergence is lowercase div(u); inside a stress it is Trace(Sym(Grad(u)))"),
    ("BaseVector' object has no attribute 'vec'",
     "NGSolve: a LinearForm's vector IS f.vec; assign through .data and read numbers with FV().NumPy()"),
    ("must not have TrialFunction",
     "NGSolve: a LinearForm carries only the test function; every term with the trial function belongs "
     "in the BilinearForm"),
    ("has no attribute 'subset_dofs'",
     "FEniCSx: fem.locate_dofs_topological(V, fdim, facets) or fem.locate_dofs_geometrical(V, marker)"),
    ("'float' object has no attribute 'ufl_domain'",
     "FEniCSx: ufl.Constant takes a DOMAIN, not a value -- use a plain scalar for a boundary value, "
     "or fem.Constant(mesh, value) inside a form"),
    ("has no attribute 'FiniteElement'",
     "FEniCSx: build spaces with fem.functionspace(mesh, ('Lagrange', 1))"),
    ("has no attribute 'VectorFunctionSpace'",
     "FEniCSx: fem.functionspace(mesh, ('Lagrange', 1, (mesh.geometry.dim,)))"),
    ("missing 1 required keyword-only argument: 'petsc_options_prefix'",
     "FEniCSx: LinearProblem requires petsc_options_prefix='<any name>' on this install (measured "
     "exactly this message)"),
    ("'LinearProblem' object has no attribute '_solver'",
     "FEniCSx: the solver is the PUBLIC p.solver; touching p._solver raises"),
    ("has no attribute 'geometry.point'",
     "DUNE-fem: a vertex's coordinates are vertex.geometry.center (or .corner(0))"),
    ("'Geometry' object has no attribute 'point'",
     "DUNE-fem: a vertex's coordinates are vertex.geometry.center (or .corner(0))"),
    ("'Form' object has no attribute 'copy'",
     "DUNE-fem: a UFL form is immutable -- build the second one by writing the expression again"),
    ("has no attribute 'converged'",
     "DUNE-fem: scheme.solve returns a DICT -- read info['converged']"),
    ("has no attribute 'dim'",
     "DUNE-fem: a discrete space counts its dofs with space.size (or len(space))"),
    ("has no attribute 'dofCoordinates'",
     "DUNE-fem: dof coordinates come from the grid -- gridView.vertices with v.geometry.center, "
     "numbered by gridView.indexSet.index(v)"),
    ("synchronous_iterator.h",
     "deal.II: that wall of template errors is a constructor mistake in YOUR file, usually "
     "FEEvaluation where FEValues belongs"),
    ("request for member \u2018unsubscribe\u2019",
     "deal.II: read the FIRST error and check your constructor -- FEValues<2>(fe, quadrature, "
     "update_flags), in that order"),
    ("is not registered",
     "Kratos: 2-D conduction is P1 TRIANGLES (LaplacianElement2D3N); 3-D is LaplacianElement3D4N"),
    # 21 recorded transcripts carry one of these two; one run met it four times,
    # exported NaN each time and replaced the served contract to get past it.
    ("Error zero sum",
     "Kratos: the skyline LU met a zero pivot -- the assembled system is singular, and what "
     "it exports is NaN. On record both of these messages came from ELEMENT ORIENTATION: "
     "tetrahedra with negative volume ('Error zero sum') and a triangle whose nodes run "
     "clockwise ('Error zero in diagonal'). On a mesh you built, take each triangle's "
     "signed area (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0) (a tetrahedron's signed volume) and "
     "swap two nodes where it is negative; the other causes are a node in no element and "
     "a problem with no held value anywhere"),
    ("Error zero in diagonal",
     "Kratos: a zero on the diagonal of the assembled system -- on record, a triangle whose "
     "nodes run clockwise (negative area); take each element's signed area and swap two "
     "nodes where it is negative"),
    # Measured 2026-09-29 in the web interface: one NGSolve run failed nine times in a row, on
    # SplineGeometry (three times), Rectangle after `from netgen.csg import *`, a .msh path that
    # was not in the job folder, mesh.ndof and a list given as `dirichlet`, and then gave up. The
    # ReadGmsh and dolfinx.io.gmsh entries were measured beside them. Every one is re-derived on
    # this install by tests/test_a_failed_run_is_told_the_measured_fix.py.
    ("name 'SplineGeometry' is not defined", _SPLINE_GEOMETRY),
    ("cannot import name 'SplineGeometry'", _SPLINE_GEOMETRY),
    ("name 'Rectangle' is not defined", _NETGEN_2D),
    ("'ngsolve.comp.Mesh' object has no attribute 'ndof'",
     "NGSolve: a mesh counts its vertices with mesh.nv and its elements with mesh.ne; ndof "
     "belongs to a finite-element SPACE (fes.ndof)"),
    ("NgException: Error opening file",
     "NGSolve/netgen: the file is not where the script looked. run_simulation starts the "
     "script in a job folder of its own, so a relative path is read from there: use the "
     "absolute path (generate_mesh's reply gives it). A Gmsh .msh is read with "
     "Mesh(ReadGmsh(path)) from netgen.read_gmsh -- Mesh(path) reads netgen's .vol format "
     "only, and on a .msh it returns an EMPTY mesh (mesh.ne == 0) without an error"),
    ("nelem = int(f.readline())",
     "netgen's ReadGmsh reads MSH 2.2 ASCII only; a 4.x file stops it here with 'invalid "
     "literal for int()'. Write 2.2 (gmsh.option.setNumber('Mesh.MshFileVersion', 2.2) "
     "before gmsh.write); openPASO's generate_mesh writes 2.2"),
    ("No module named 'dolfinx.io.gmshio'", _DOLFINX_GMSH),
    ("cannot import name 'gmshio' from 'dolfinx.io'", _DOLFINX_GMSH),
    ("Unable to cast Python instance of type <class 'str'> to C++ type",
     "NGSolve: one call that raises this, measured: `dirichlet` given as a Python LIST of "
     "names. It takes ONE string, the boundary names joined by | -- "
     "H1(mesh, order=2, dirichlet='left|top|cylinder')"),
)


def findings_from_output(output: str) -> list:
    """What a run already told you, with the call that works.

    A failed run has bought the diagnosis; spending a second run to learn it is the loop that ate
    round 49. Names at most three, because an agent acts on the first."""
    if not isinstance(output, str) or not output.strip():
        return []
    out, seen = [], set()
    for needle, fix in _ERROR_FIXES:
        if needle in output and fix not in seen:
            seen.add(fix)
            out.append(f"the run printed `{needle}` -- {fix}. Measured on this install.")
        if len(out) >= 3:
            break
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
        if looks_like_participant(text):
            return [f for f in undefined_names(text) if f.startswith("this file does not parse")]
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
    if _SERVED_MARK not in text:
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
    for backend, pattern, error, fix in _TRAPS:
        if backend not in codes or (backend, pattern) in seen:
            continue
        m = re.search(pattern, body, re.M)
        if not m:
            continue
        seen.add((backend, pattern))
        line = body[:m.start()].count("\n") + 1
        out.append(f"{backend}: `{m.group(0).strip()}` (near line {line} of the stripped source) -- "
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


_SERVED_CHECK_LABELS = (("EXPORT SELF-CHECK", "the export self-check"),
                        ("OUTER BOUNDARY", "the held-edge check"),
                        ("INTERFACE DOFS", "the interface list against the mesh"),
                        ("INTERFACE VERTICES", "the interface list against the mesh"),
                        ("SOLVE SELF-CHECK", "the solve self-check"))


def _served_checks_lacking(content: str) -> list:
    """The served checks a file lacks, read from the file: a re-typed contract that kept
    its held-edge stop under a renamed comment was told it carried none of them."""
    missing = []
    for lab, what in _SERVED_CHECK_LABELS:
        if lab not in content and what not in missing:
            if lab == "INTERFACE DOFS" and "INTERFACE VERTICES" in content:
                continue
            if lab == "INTERFACE VERTICES" and "INTERFACE DOFS" in content:
                continue
            missing.append(what)
    return missing


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
        + _also_lacks([w for w in _served_checks_lacking(content) if w != "the export self-check"]))


def _also_lacks(items: list) -> str:
    if not items:
        return ""
    said = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
    return f" It also lacks {said}."


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
_FREE_MASK = re.compile(r"(?<![\w.])\w+\s*\[\s*(?:int\()?\w+\)?\s*\]\s*=\s*(?:False|0)\b")
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


def imported_values_not_held(content: str) -> str:
    """'' unless partner values are loaded into the solution vector and then lost."""
    if not isinstance(content, str) or "ngsolve" not in backends_in(content):
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
        if not before:
            continue                                   # nothing essential loaded into it
        # IN WORDS, NOT AS THE TWO LINES OF THE SOLVE: the literal residual-form
        # solve given here was copied verbatim into correct runs, and the linear
        # solve is one of the things openPASO does not serve.
        lifting = ("Solve for the CORRECTION instead: with the fixed values already in "
                   "the solution vector, apply the inverse to the residual those values "
                   "leave -- the load minus the assembled operator applied to them -- "
                   "and ADD the result. That is the one form in which the fixed values "
                   "both stay and reach the interior.")
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
                    "held one, so the values written there before -- the partner's trace, "
                    "the outer values -- are zeroed and never written back. ADD the "
                    "correction to the vector instead, or write the held values back "
                    "after it.")
            return (
                f"`{sol}.vec[...]` is written with values first and then "
                f"`{sol}.{lhs_form} = {said}` replaces the whole vector, so those "
                "entries are gone by the time anything reads them: whatever "
                "they held -- prescribed outer values on any side, the partner's "
                "interface trace on a Dirichlet side -- is loaded, discarded, and "
                "the system is solved as though those boundaries carried a zero "
                "condition. Re-applying the values AFTER the solve does not repair "
                "it -- the interior was computed without them and is the answer to "
                "a different problem. This is the one defect that still converges, "
                "so the residual history will not tell you. " + lifting)
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
        # The index is usually the loop variable, so the interface name sits on
        # the `for` line above it rather than inside the brackets. Read the
        # write together with the lines that bind it.
        def _from_interface(w) -> bool:
            head = body[:w.start()].rsplit("\n", 4)[1:]
            return bool(_IFACE_NAME.search("\n".join(head) + w.group(0)))

        # A MASK OF ITS OWN IS A MASK. A solve inverted on a BitArray the script
        # built (cleared, then the free bits set) may well hold the interface; this
        # branch only knew masks made by clearing bits and called such a side
        # "nothing holds those entries fixed" (measured on a recorded fill whose
        # mask was broken for another reason, named by unset_mask_bits).
        def _custom_mask(expr: str) -> bool:
            src = temps.get(re.sub(r"\.data\s*$", "", expr.strip()), expr)
            inv = re.search(r"Inverse\s*\(\s*([^,()]+(?:\([^()]*\))?)", src)
            if not inv:
                for n in bound:
                    if re.search(rf"\b{re.escape(n)}\b", src):
                        inv = re.search(r"Inverse\s*\(\s*([^,()]+(?:\([^()]*\))?)", _definition(n))
                        break
            if not inv:
                return False
            arg = inv.group(1).strip()
            if "FreeDofs" in arg:
                return False
            d = _definition(arg) if re.fullmatch(r"\w+", arg) else ""
            return not re.fullmatch(r"\s*[\w\.]+\.FreeDofs\s*\([^)]*\)\s*", d or "")

        if (any(_from_interface(w) for w in writes)
                and not _FREE_MASK.search(body) and not _custom_mask(rhs)):
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


# ── the tetrahedra that were never checked for sign ────────────────────────
#
# Kratos ships no mesher, so a 3D participant writes its own connectivity, and
# the naive path decomposition of a hexahedron puts three of its six
# tetrahedra in an ODD permutation. A negatively oriented LaplacianElement3D4N
# assembles a singular system: "LUSkylineFactorization: Error zero sum", and
# every recovered reaction comes back NaN. The failure does not name the mesh,
# so it reads as a solver problem and is debugged in the wrong place -- one
# recorded cell spent its whole budget on it and wrote two throwaway probe
# scripts before reporting the backend broken. It is not broken: the served
# contract builds the same elements and recovers finite fluxes.
#
# MEASURED: 48 recorded scripts build 3D4N elements by hand and 29 of them
# check no sign at all; every recorded run among those was incomplete.
# Silent on the served set.
#
# It names the defect, not the mesh: which decomposition to use stays the
# author's choice, and the only claim is that a tetrahedron handed to an
# element has to have positive volume.

_TET_ELEMENT = re.compile(r"CreateNewElement\s*\(\s*[\"'][A-Za-z]*3D4N[\"']")
_ORIENT_CHECK = re.compile(
    r"np\.cross|numpy\.cross|np\.linalg\.det|numpy\.linalg\.det|signed[_ ]?volume|orient",
    re.I)


def unoriented_tetrahedra(content: str) -> str:
    """'' unless tetrahedral elements are built by hand with no signed-volume check."""
    if not isinstance(content, str) or not _TET_ELEMENT.search(content):
        return ""
    if _ORIENT_CHECK.search(content):
        return ""
    return (
        "this script builds 3D4N tetrahedra from its own connectivity and never "
        "checks their sign. Splitting a hexahedron along a path puts three of "
        "the six tetrahedra in an odd permutation, and a negatively oriented "
        "element assembles a singular system: the solve reports "
        "'LUSkylineFactorization: Error zero sum' and every reaction you read "
        "back is NaN. Nothing in that message mentions the mesh, so it is "
        "usually debugged as a solver or a variable-registration problem and is "
        "neither. Take the signed volume of each tetrahedron before you create "
        "the element -- dot(cross(p1-p0, p2-p0), p3-p0) -- and swap two nodes "
        "where it comes out negative.")


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


_HOLE_RE = re.compile(r"# ── SOLVE ─ [^\n]*?DOES NOT SERVE THIS ─ begin.*?DOES NOT SERVE THIS ─ end", re.S)


@functools.lru_cache(maxsize=1)
def _served_line_sets() -> tuple:
    """Per shipped contract, its served lines of 40 characters or more, holes cut."""
    out = []
    try:
        for q in sorted(_PARTICIPANT_DIR.glob("participant_*.py")):
            lines = frozenset(ln.strip() for ln in _HOLE_RE.sub("", q.read_text(errors="ignore")).splitlines()
                              if len(ln.strip()) >= 40)
            if len(lines) >= 20:
                out.append(lines)
    except OSError:
        return ()
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
    partner = None
    for d in dirs:
        for q in sorted(d.glob("*.py")):
            if q.resolve() == near.resolve() or ".replaced-" in q.name:
                continue
            try:
                txt = q.read_text(errors="ignore")
            except OSError:
                continue
            if _is_served_contract(txt) and ("imports.json" in txt or "exports.json" in txt):
                partner = q
                break
        if partner:
            break
    if partner is None:
        return ""
    codes = [c for c in backends_in(content) if c in _CONTRACT_DOOR]
    code = _CONTRACT_DOOR[codes[0]] if codes else None
    try:
        rel = str(near.relative_to(near.parent.parent)) if near.parent.name.lower().startswith("side") else near.name
    except ValueError:
        rel = near.name
    try:
        who = str(partner.relative_to(partner.parent.parent))
    except ValueError:
        who = partner.name
    call = (f"write_participant_contract(solver='{code}', path='{rel}')" if code
            else f"write_participant_contract(solver=<this side's code>, path='{rel}')")
    # WHICH SERVED CHECKS THE FILE LACKS, read from the file: a re-typed contract that kept
    # its held-edge stop under a renamed comment was told it carried none of them.
    _missing = _served_checks_lacking(content)
    _lack = (f" It lacks the served {', '.join(_missing)}, so a defect those would stop runs on "
             f"silently." if _missing else "")
    return (
        f"this side is HAND-WRITTEN ({len(content):,} chars) while {who} is the served contract."
        f"{_lack} The served contract for this side is one call, {call}; then fill only its "
        f"marked hole, in place.")


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


def served_guard_removed(content: str, near=None) -> str:
    """'' unless this is a served contract whose refusals were deleted or demoted to prints.

    MEASURED: a run replaced the served `raise SystemExit("the TSI deck's temperature differs ...")`
    with a WARNING print, its own logs then carried "scatra-vs-tsi T mismatch 1.00e+00" at every
    level -- the structural deck's temperature was zero, its traction had no thermal stress -- and
    the coupling never converged at the finest level. A served refusal stops a run that would hand
    in a wrong number; the fill is the agent's, the guards are not.
    """
    if not isinstance(content, str) or not _is_served_contract(content):
        return ""
    first = content.lstrip().split("\n", 1)[0].strip()
    if not first.startswith('"""') or len(first) < 20:
        return ""
    template = None
    try:
        for q in sorted(_PARTICIPANT_DIR.glob("participant_*.py")):
            head = q.read_text(errors="ignore").lstrip().split("\n", 1)[0].strip()
            if head == first:
                template = q.read_text(errors="ignore")
                break
    except OSError:
        return ""
    if template is None:
        return ""
    # THE HOLE'S OWN LINES ARE NEVER SERVED: the contract reaches the agent
    # with every hole elided, so a refusal inside a hole (the "hole is not
    # filled" stop) is not a guard the agent received. Measured: the pristine
    # served 4C contract drew this finding in four cells of one round for
    # exactly that stop, and the round was abandoned after five minutes.
    hole_free = re.sub(r"# ── SOLVE ─ [^\n]*?DOES NOT SERVE THIS ─ begin.*?DOES NOT SERVE THIS ─ end",
                       "", template, flags=re.S)
    # A GUARD IS ITS CONDITION, AND ITS MESSAGE ONLY SECOND. Matching the
    # message word for word over 70 characters called a refusal "removed"
    # when a run had kept it and only shortened its text (served three times
    # to a correct run, measured). A guard counts as kept when its `if`
    # condition is still there and a `raise SystemExit` follows it, or when
    # the first words of its message still sit behind a `raise SystemExit`;
    # demoted when the condition is followed by a print instead; gone only
    # when neither its condition nor the start of its message is left.
    def _norm(line):
        return re.sub(r"\s+", "", line)

    hole_lines = hole_free.splitlines()
    guards = []                      # (condition or None, message head)
    for i, line in enumerate(hole_lines):
        if "raise SystemExit(" not in line:
            continue
        m = re.search(r"raise SystemExit\(\s*f?([\"'])", line)
        text = ""
        if m:                         # up to the quote that opened it; an apostrophe inside stays
            rest = line[m.end():]
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
                cond = _norm(prev)
            break
        if (cond, head, full[:50]) not in guards and (cond or len(head) >= 12):
            guards.append((cond, head, full[:50]))
    if not guards:
        return ""
    body_lines = content.splitlines()
    norm_body = [_norm(l) for l in body_lines]
    gone = []
    for cond, head, label in guards:
        state = None
        if cond is not None:
            for i, nl in enumerate(norm_body):
                if nl != cond:
                    continue
                follow = [l.strip() for l in body_lines[i + 1:i + 5] if l.strip() and not l.strip().startswith("#")]
                nxt = follow[0] if follow else ""
                state = "kept" if nxt.startswith("raise SystemExit") else ("demoted" if nxt.startswith("print(") else "changed")
                if state == "kept":
                    break
        if state != "kept" and len(head) >= 12:
            for mm in re.finditer(re.escape(head), content):
                before = content[max(0, mm.start() - 40):mm.start()]
                if "raise SystemExit" in before:
                    state = "kept"
                    break
                if "print(" in before:
                    state = state or "demoted"
        if state == "kept":
            continue
        label = label or head or (cond or "")[:40]
        if state == "demoted":
            gone.append(f"'{label}' (demoted to a print)")
        elif state is None:
            gone.append(f"'{label}'")
    if not gone:
        return ""
    return (f"SERVED GUARD REMOVED: {len(gone)} of the {len(guards)} refusals this contract came with "
            f"are gone or demoted to a print: {'; '.join(gone[:4])}{'; ...' if len(gone) > 4 else ''}. "
            f"Each stops a run that would hand in a wrong number (measured: with the temperature "
            f"cross-check demoted, a run exported tractions with no thermal stress at every level). "
            f"Put them back as `raise SystemExit(...)`; fill the hole, keep the guards.")


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
