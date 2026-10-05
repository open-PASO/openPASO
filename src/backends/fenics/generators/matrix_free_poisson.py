"""Matrix-free Poisson generator for FEniCSx/dolfinx.

Encodes the matrix-free conjugate-gradient pattern from upstream
dolfinx ``demo_poisson_matrix_free.py``. The matrix-free formulation
never assembles the global stiffness operator A explicitly — instead
it provides a callable ``action_A(x, y)`` that computes y = A·x by
re-assembling the linear form ``M = ufl.action(a, ui)`` with ui = x.

Why it matters: for very large 3D / high-order problems, the
explicit sparse matrix becomes the memory bottleneck; matrix-free
keeps memory ~O(n_dofs) per process. Pairs naturally with high-
performance solvers (custom CG, Krylov + matrix-free preconditioner).

Source: upstream dolfinx
  conda envs/ofa-fenicsx/etc/conda/test-files/fenics-dolfinx/0/
  python/demo/demo_poisson_matrix_free.py
"""

VARIANTS = ["2d"]

KNOWLEDGE = {'description': 'Poisson problem -div(grad(u)) = f solved WITHOUT ever assembling the '
                'global stiffness matrix. The operator is supplied to a hand-written '
                'conjugate-gradient loop as a callable that re-assembles the linear '
                'form ufl.action(a, ui) with ui holding the current trial vector. '
                'Memory stays O(n_dofs) per process instead of O(nnz), which is what '
                'makes very large 3D or high-order runs feasible.',
 'minimal_working_example': '"""Matrix-free CG Poisson (dolfinx 0.10). Reference-free '
                            'self-checks."""\n'
                            'from mpi4py import MPI\n'
                            'import numpy as np\n'
                            'import dolfinx\n'
                            'import ufl\n'
                            'from dolfinx import fem, la, mesh\n'
                            '\n'
                            'dtype = dolfinx.default_scalar_type\n'
                            'comm = MPI.COMM_WORLD\n'
                            'msh = mesh.create_unit_square(comm, 24, 24)\n'
                            'V = fem.functionspace(msh, ("Lagrange", 2))\n'
                            '\n'
                            'tdim = msh.topology.dim\n'
                            'msh.topology.create_connectivity(tdim - 1, tdim)\n'
                            'facets = mesh.exterior_facet_indices(msh.topology)\n'
                            'bdofs = fem.locate_dofs_topological(V, tdim - 1, facets)\n'
                            'uD = fem.Function(V, dtype=dtype)\n'
                            'uD.interpolate(lambda x: 0.5 * x[0])            # '
                            'inhomogeneous Dirichlet data\n'
                            'bc = fem.dirichletbc(uD, bdofs)\n'
                            '\n'
                            'u, v = ufl.TrialFunction(V), ufl.TestFunction(V)\n'
                            'x = ufl.SpatialCoordinate(msh)\n'
                            'f = 10.0 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** '
                            '2) / 0.02)\n'
                            'a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx\n'
                            'L_fem = fem.form(ufl.inner(f, v) * ufl.dx, dtype=dtype)\n'
                            '\n'
                            'ui = fem.Function(V, dtype=dtype)\n'
                            'M_fem = fem.form(ufl.action(a, ui), dtype=dtype)\n'
                            '\n'
                            'def action_A(xv, yv):\n'
                            '    ui.x.array[:] = xv.array\n'
                            '    ui.x.scatter_forward()\n'
                            '    yv.array[:] = 0.0\n'
                            '    fem.assemble_vector(yv.array, M_fem)\n'
                            '    yv.scatter_reverse(la.InsertMode.add)\n'
                            '    bc.set(yv.array, alpha=0.0)\n'
                            '\n'
                            'b = fem.assemble_vector(L_fem)\n'
                            'ui.x.array[:] = 0.0\n'
                            'bc.set(ui.x.array, alpha=-1.0)\n'
                            'fem.assemble_vector(b.array, M_fem)\n'
                            'b.scatter_reverse(la.InsertMode.add)\n'
                            'bc.set(b.array, alpha=0.0)\n'
                            'b.scatter_forward()\n'
                            'nr = b.index_map.size_local\n'
                            '\n'
                            'def gdot(v0, v1):\n'
                            '    return comm.allreduce(np.vdot(v0[:nr], v1[:nr]), '
                            'MPI.SUM)\n'
                            '\n'
                            'def cg(xv, bv, max_iter=1000, rtol=1e-10):\n'
                            '    yv = la.vector(bv.index_map, 1, dtype)\n'
                            '    action_A(xv, yv)\n'
                            '    r = bv.array - yv.array\n'
                            '    p = la.vector(bv.index_map, 1, dtype)\n'
                            '    p.array[:] = r\n'
                            '    rn0 = rn = gdot(r, r)\n'
                            '    for k in range(max_iter):\n'
                            '        action_A(p, yv)\n'
                            '        alpha = rn / gdot(p.array, yv.array)\n'
                            '        xv.array[:] += alpha * p.array\n'
                            '        r -= alpha * yv.array\n'
                            '        rn, rn_old = gdot(r, r), rn\n'
                            '        if rn / rn0 < rtol ** 2:\n'
                            '            xv.scatter_forward()\n'
                            '            return k + 1, True\n'
                            '        p.array[:] = (rn / rn_old) * p.array + r\n'
                            '    xv.scatter_forward()\n'
                            '    return max_iter, False\n'
                            '\n'
                            'uh = fem.Function(V, dtype=dtype)\n'
                            'its, converged = cg(uh.x, b)\n'
                            'assert converged, "matrix-free CG hit max_iter without '
                            'reaching rtol"\n'
                            'bc.set(uh.x.array, alpha=1.0)\n'
                            'uh.x.scatter_forward()\n'
                            '\n'
                            'ui.x.array[:] = uh.x.array\n'
                            'ui.x.scatter_forward()\n'
                            'res = la.vector(b.index_map, 1, dtype)\n'
                            'res.array[:] = 0.0\n'
                            'fem.assemble_vector(res.array, M_fem)\n'
                            'res.scatter_reverse(la.InsertMode.add)\n'
                            'rhs = fem.assemble_vector(L_fem)\n'
                            'rhs.scatter_reverse(la.InsertMode.add)\n'
                            'res.array[:] -= rhs.array\n'
                            'bc.set(res.array, alpha=0.0)\n'
                            'rel_res = float(np.sqrt(gdot(res.array, res.array) / '
                            'gdot(rhs.array, rhs.array)))\n'
                            'bc_err = comm.allreduce(\n'
                            '    float(np.abs(uh.x.array[bdofs] - '
                            'uD.x.array[bdofs]).max(initial=0.0)), MPI.MAX)\n'
                            'gmin = comm.allreduce(float(uh.x.array[:nr].min()), '
                            'MPI.MIN)\n'
                            'gmax = comm.allreduce(float(uh.x.array[:nr].max()), '
                            'MPI.MAX)\n'
                            'if comm.rank == 0:\n'
                            '    print(f"matrix-free CG: converged={converged} '
                            'iterations={its} "\n'
                            '          f"dofs={V.dofmap.index_map.size_global}")\n'
                            '    print(f"relative Galerkin residual over free DOFs = '
                            '{rel_res:.3e}")\n'
                            '    print(f"max |u - u_D| on Dirichlet DOFs = '
                            '{bc_err:.3e}")\n'
                            '    print(f"global u range = [{gmin:.6f}, {gmax:.6f}] '
                            'finite={np.isfinite(gmax)}")\n'
                            'assert rel_res < 1e-6 and bc_err < 1e-12 and '
                            'np.isfinite(gmax)\n',
 'function_space': {'REQUIRED': 'V = fem.functionspace(msh, ("Lagrange", degree))\n'
                                'ui = fem.Function(V, '
                                'dtype=dolfinx.default_scalar_type)   # holds the '
                                'trial vector inside the operator action',
                    'OPTIONAL': 'degree: any integer >= 1. Cell type: triangle or '
                                'quadrilateral (mesh.CellType.triangle / '
                                '.quadrilateral); both work unchanged. dtype may be '
                                'omitted (defaults to dolfinx.default_scalar_type).',
                    'explanation': 'Matrix-free assembly puts no extra constraint on '
                                   'the space; the only structural requirement is one '
                                   'extra Function (ui) that the operator action '
                                   'writes the input vector into before re-assembling.',
                    'pitfalls': ['[API] Use fem.functionspace (lower-case s), not '
                                 'fem.FunctionSpace. Signal: fem.FunctionSpace still '
                                 'EXISTS in dolfinx 0.10 as a class, so the wrong '
                                 'spelling does not give AttributeError; it gives '
                                 'TypeError: FunctionSpace.__init__() missing 1 '
                                 "required positional argument: 'cppV'."]},
 'weak_form': {'REQUIRED': 'a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx\n'
                           'L_fem = fem.form(ufl.inner(f, v) * ufl.dx, dtype=dtype)\n'
                           'M_fem = fem.form(ufl.action(a, ui), dtype=dtype)   # '
                           'operator action, compiled ONCE',
               'OPTIONAL': 'Any coefficient may be added to a (e.g. '
                           'kappa*grad(u).grad(v) + c*u*v); ufl.action works on any '
                           'bilinear form. f may be a fem.Constant, a fem.Function, or '
                           'a UFL expression of ufl.SpatialCoordinate.',
               'explanation': 'ufl.action(a, ui) turns the bilinear form a(u,v) into '
                              'the linear form v -> a(ui,v). Assembling that linear '
                              'form is exactly the matrix-vector product A*ui, so no '
                              'sparse matrix is ever built.',
               'pitfalls': ['[API] Write ufl.action(a, ui), not a.action(ui). Signal: '
                            "AttributeError: 'Form' object has no attribute 'action'.",
                            '[API] Compile the action with fem.form(...) before assembling. '
                            "Signal: AttributeError: 'Form' object has no attribute "
                            "'_cpp_object'."]},
 'boundary_conditions': {'REQUIRED': 'msh.topology.create_connectivity(tdim - 1, '
                                     'tdim)\n'
                                     'facets = '
                                     'mesh.exterior_facet_indices(msh.topology)\n'
                                     'bdofs = fem.locate_dofs_topological(V, tdim - 1, '
                                     'facets)\n'
                                     'bc = fem.dirichletbc(uD, bdofs)\n'
                                     '# lifting into the RHS, in this order:\n'
                                     'ui.x.array[:] = 0.0\n'
                                     'bc.set(ui.x.array, alpha=-1.0)          # ui = '
                                     '-u_bc at constrained DOFs\n'
                                     'fem.assemble_vector(b.array, M_fem)     # b <- b '
                                     '- A*u_bc\n'
                                     'b.scatter_reverse(la.InsertMode.add)\n'
                                     'bc.set(b.array, alpha=0.0)              # zero '
                                     'the constrained rows\n'
                                     'b.scatter_forward()\n'
                                     '# and inside the operator action, after the '
                                     'reverse scatter:\n'
                                     'bc.set(y.array, alpha=0.0)\n'
                                     '# and once after the CG loop, to restore the '
                                     'prescribed values:\n'
                                     'bc.set(u.x.array, alpha=1.0)',
                         'OPTIONAL': 'bc.set signature is set(x, x0=None, alpha=1). '
                                     'alpha scales the value written: alpha=-1.0 for '
                                     'lifting, 0.0 to zero constrained entries, 1.0 to '
                                     'write the prescribed value.',
                         'explanation': 'There is no assemble_matrix(..., bcs=...) to '
                                        'apply the constraint for you, so Dirichlet '
                                        'handling is entirely manual: lift the '
                                        'prescribed values into the right-hand side, '
                                        'zero the constrained entries of every '
                                        'operator application so the Krylov method '
                                        'never updates them, and write the prescribed '
                                        'values back at the end.',
                         'pitfalls': ['[BC] Never omit bc.set(y.array, alpha=0.0) inside '
                                      'the operator action. Signal: rnorm/rnorm0 GROWS '
                                      'past 1e30 and max(u) reaches ~1e31 within 200 '
                                      'iterations, in serial.',
                                      '[Validation] The residual b - A*u is only meaningful BEFORE '
                                      'bc.set(u.x.array, alpha=1.0) writes the '
                                      'Dirichlet values back. Signal: recomputing it '
                                      'afterwards gives ||b-Au||/||b|| = 1.002 for a '
                                      'solve that is in fact correct.']},
 'solver': {'REQUIRED': '# hand-written CG; there is NO dolfinx class for this.\n'
                        'nr = b.index_map.size_local\n'
                        'def gdot(v0, v1):\n'
                        '    return comm.allreduce(np.vdot(v0[:nr], v1[:nr]), '
                        'MPI.SUM)\n'
                        'y = la.vector(b.index_map, 1, dtype)\n'
                        '# ... standard CG recurrence, calling action_A(p, y) each '
                        'iteration.\n'
                        '# ALWAYS return/raise on the max_iter exit so a '
                        'non-converged\n'
                        '# solve cannot be mistaken for a converged one.',
            'OPTIONAL': 'rtol (1e-6 .. 1e-12) and max_iter. To use PETSc instead of a '
                        'hand-written loop, wrap action_A in a PETSc.Mat of type '
                        'PETSc.Mat.Type.PYTHON and hand it to PETSc.KSP; then use '
                        'ksp.getConvergedReason() instead of your own flag.',
            'explanation': 'Unpreconditioned CG is the baseline: it needs only the '
                           'operator action. Its iteration count grows like O(1/h), so '
                           'for production runs a preconditioner (which needs at least '
                           'a diagonal or a coarse operator) is what actually pays for '
                           'itself.',
            # The PETScKrylovSolver note that stood here was verbatim-identical to
            # the one in fenics/advanced.py. This file already carries a fuller,
            # version-qualified copy in its top-level pitfalls list (it also says
            # what to use instead), so the short duplicate is dropped rather than
            # maintained in two places where a fix could land on only one.
            'pitfalls': ['[Integration] The CG inner product must use owned DOFs only: v[:nr] with '
                         'nr = index_map.size_local. Signal: on 2 MPI ranks '
                         'rnorm/rnorm0 grows to ~4e13 and max|u| reaches ~3e8, while '
                         'the identical run in serial converges.']},
 'parallel': {'REQUIRED': 'y.scatter_reverse(la.InsertMode.add)   # after EVERY '
                          'in-place assemble\n'
                          'ui.x.scatter_forward()                 # before using ui in '
                          'a form\n'
                          'value = comm.allreduce(fem.assemble_scalar(form), '
                          'op=MPI.SUM)\n'
                          'gmax = comm.allreduce(float(u.x.array[:nr].max()), MPI.MAX)',
              'OPTIONAL': 'Guard printing with `if comm.rank == 0:` so an MPI run does '
                          'not emit one copy of every line per rank.',
              'explanation': 'fem.assemble_vector(array, form) accumulates only the '
                             'LOCAL cell contributions; ghost entries must be sent '
                             'back to their owner with scatter_reverse(add), and any '
                             'scalar or extremum must be reduced across ranks. All of '
                             'this is invisible in a serial run.',
              'pitfalls': ['[Integration] Omitting scatter_reverse after in-place assembly. Signal: '
                           'identical in serial, but on 2 ranks CG fails to reach rtol '
                           'in 200 iterations and u lands in [-2.8e+02, 1.1e+03] '
                           'instead of the correct O(1) range.',
                           '[Integration] Omitting comm.allreduce around fem.assemble_scalar. '
                           'Signal: an integral that is silently too SMALL on multiple '
                           'ranks (measured 7.10e-07 instead of 1.02e-06 on 2 ranks) - '
                           "both values look 'small', so nothing draws attention to "
                           'it.']},
 'references': ['dolfinx upstream demo: demo_poisson_matrix_free.py',
                'Saad, Y. (2003), Iterative Methods for Sparse Linear Systems, Ch. 6 '
                '(Krylov subspace methods).'],
 'pitfalls': ['[API] Build the operator action with `M = ufl.action(a, ui)`; '
              '`a.action(ui)` is not a UFL Form method. Then compile it once with '
              '`M_fem = fem.form(M, dtype=dtype)` and reuse the compiled form in every '
              'CG iteration - recompiling inside the loop is the single biggest '
              "avoidable cost. Signal: `AttributeError: 'Form' object has no attribute "
              "'action'` raised at the line that builds M. If you skip fem.form and "
              'hand the raw UFL form to fem.assemble_vector, the previously quoted '
              'message "RuntimeError: cannot assemble: form has not been compiled" '
              'does NOT appear, and WHICH AttributeError you get depends on the call '
              "form: the two-argument in-place call fem.assemble_vector(y.array, M) "
              "gives `AttributeError: 'Form' object has no attribute '_cpp_object'`, "
              "while the one-argument fem.assemble_vector(M) gives `AttributeError: "
              "'Form' object has no attribute 'function_spaces'` instead. Match on the "
              "AttributeError, not on one of the two attribute names. (Verified by execution on dolfinx 0.10.0, 2026-08-06.)",
              '[API] The in-place assembly entry point is '
              '`fem.assemble_vector(b.array, M_fem)` - first argument is the numpy '
              'ARRAY, not the la.Vector. The one-argument call `b = '
              'fem.assemble_vector(L_fem)` is also valid and returns a fresh '
              '`dolfinx.la.Vector`, which is how you create b in the first place. '
              'Signal: passing the la.Vector as the first argument of the two-argument '
              "form gives `AttributeError: 'Vector' object has no attribute "
              '\'function_spaces\'`. The previously quoted messages "TypeError: '
              'assemble_vector() takes positional argument" and "expected ndarray, got '
              'Vector" do NOT reproduce on dolfinx 0.10.0.',
              '[Numerical] Zero the Dirichlet entries of the output of EVERY operator '
              'application (`bc.set(y.array, alpha=0.0)` at the end of action_A). '
              'Without it the constrained rows feed the unconstrained ones and CG has '
              'no fixed point. Signal: rnorm/rnorm0 grows monotonically - 1.4e+12 at '
              'iteration 100, 3.4e+30 at iteration 160 - and the returned field '
              'reaches max(u) = 1.7e+31 while min(u) still equals the boundary value, '
              'so a min-only sanity print looks innocent. This is a SERIAL failure; it '
              'does not need MPI to show up. The previously quoted signal ("CG '
              'iteration count balloons (>200) ... final L2 error ~1e-1 instead of '
              '~1e-6") understates it by thirty orders of magnitude.',
              '[Integration] After the CG loop, `bc.set(u.x.array, alpha=1.0)` writes '
              'the prescribed Dirichlet values into the solution. From that moment on, '
              '`b - A*u` is NO LONGER the residual of the system that was solved, '
              'because b already carries the lifting term -A*u_bc. Signal: recomputing '
              '||b - A*u|| / ||b|| after the bc.set gives 1.002e+00 for a solve whose '
              'true Galerkin residual is ~1e-8. To check a matrix-free solve, assemble '
              'the Galerkin residual of the FULL solution instead - `action(a, u) - '
              'L`, zeroed at constrained DOFs - which is the check the '
              'minimal_working_example uses.',
              '[API] `fem.assemble_vector(array, form)` only sums the cells owned by '
              'the calling rank; ghost contributions must be returned to their owner '
              'with `y.scatter_reverse(la.InsertMode.add)` after every in-place '
              'assembly, including inside the operator action. Signal: the serial run '
              'is unaffected; on 2 ranks the same script fails to reach rtol in 200 '
              'iterations and the solution lands in [-2.84e+02, +1.13e+03] where the '
              'correct range is O(1). Purely an MPI-only failure - a single-rank test '
              'will never expose it.',
              '[Integration] The CG inner products must run over OWNED DOFs only: `nr '
              '= b.index_map.size_local` then `comm.allreduce(np.vdot(v0[:nr], '
              'v1[:nr]), MPI.SUM)`. Using the full ghost-padded array double-counts '
              'every shared DOF. Signal: on 2 ranks rnorm/rnorm0 climbs 1.2e+04 (it '
              '100), 1.6e+06 (it 140), 4.0e+13 (it 180) and the field reaches 3.3e+08; '
              'the identical script in serial converges in the normal number of '
              'iterations, because with one rank there are no ghosts.',
              '[API] Any scalar diagnostic must be reduced: '
              '`comm.allreduce(fem.assemble_scalar(fem.form(...)), op=MPI.SUM)`, and '
              '`u.x.array.min()/.max()` are RANK-LOCAL - reduce them with MPI.MIN / '
              'MPI.MAX over `u.x.array[:nr]`. Signal: an integral norm that comes out '
              'too SMALL on multiple ranks (measured 7.10e-07 on 2 ranks against the '
              'correct 1.02e-06) - never an exception, never a NaN, just a quietly '
              'optimistic number.',
              '[Performance] Unpreconditioned CG iteration count scales like O(1/h): '
              'measured on this install, halving the mesh size roughly DOUBLES the '
              'iteration count at fixed polynomial degree, and raising the degree at '
              'fixed mesh also multiplies it. Both hold for triangles and '
              'quadrilaterals. Signal: the run does not fail - it just gets slow, and '
              'a fixed max_iter that was comfortable on the coarse mesh starts '
              'tripping the non-convergence exit on the fine one. Production '
              'matrix-free codes wrap action_A in a `PETSc.Mat` of type '
              '`PETSc.Mat.Type.PYTHON` and attach a Jacobi or matrix-free multigrid '
              "preconditioner; note that 'gamg' and 'hypre' cannot be used here "
              'because they need the assembled matrix entries that matrix-free '
              'deliberately avoids.',
              '[Integration] A CG loop that simply falls out of its `for` after '
              'max_iter, without raising or returning a converged flag, is the classic '
              'silent-failure shape: the caller gets a Function that looks like a '
              'solution. Signal: the hand-rolled CG loop has no PETSc KSP behind it, '
              'so there is no KSPConvergedReason to consult and nothing raises: in the '
              'diverging variants above the loop exits normally and the script prints '
              'a solution containing 1e+31 with return code 0. Always return the '
              'convergence flag and assert on it, exactly as the '
              'minimal_working_example does.',
              '[API] `dolfinx.PETScKrylovSolver` does not exist in dolfinx 0.10.0. '
              "Signal: `AttributeError: module 'dolfinx' has no attribute "
              "'PETScKrylovSolver'`. If you want PETSc's CG instead of a hand-written "
              'loop, create it directly through petsc4py (`PETSc.KSP().create(comm)`); '
              "if you want dolfinx's high-level wrapper for the ASSEMBLED problem, "
              'that is `dolfinx.fem.petsc.LinearProblem`, which requires the '
              'keyword-only argument `petsc_options_prefix` (`TypeError: '
              'LinearProblem.__init__() missing 1 required keyword-only argument: '
              "'petsc_options_prefix'` if omitted).",
              '[Numerical] Do not use the difference against an interpolated '
              'manufactured field as a proxy for discretisation error in this '
              'template. For a polynomial reference field of degree <= the element '
              'degree, the Galerkin solution equals the interpolant to machine '
              'precision, so what such a check actually measures is the CG stopping '
              "tolerance. Signal: the reported 'error' STAYS at the Krylov tolerance "
              'and even grows slightly as the mesh is refined (measured on this '
              'install: the direct-LU solution differs from the interpolant by ~1e-15 '
              'at both degree 1 and degree 2 on a structured unit-square mesh, while '
              'the matrix-free run reports ~1e-6 and rising). Use a residual, a flux '
              'balance or an energy identity instead - none of which needs a reference '
              'solution.']}


def generate(variant: str, params: dict) -> str:
    if variant not in VARIANTS:
        raise ValueError(
            f"Unknown matrix_free_poisson variant: {variant!r}. "
            f"Available: {VARIANTS}")
    return _matrix_free_poisson_2d(params)


def _matrix_free_poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate
    values for your specific problem.

    Matrix-free CG Poisson on a unit-square mesh. Default exact
    solution: u_D = 1 + x^2 + 2*y^2 with f = -6 (so -Δu = f and
    u|_∂Ω = u_D). The L2 error is reported."""
    nx = params.get("nx", 12)
    ny = params.get("ny", 12)
    degree = params.get("degree", 2)
    rtol = params.get("rtol", 1e-6)
    max_iter = params.get("max_iter", 200)
    return f'''\
"""Matrix-free conjugate-gradient Poisson — FEniCSx/dolfinx
Mirrors upstream demo_poisson_matrix_free.py.
"""
from mpi4py import MPI
import numpy as np
import dolfinx
import ufl
from dolfinx import fem, la
import json

dtype = dolfinx.default_scalar_type
real_type = np.real(dtype(0.0)).dtype
comm = MPI.COMM_WORLD
mesh = dolfinx.mesh.create_rectangle(
    comm, [[0.0, 0.0], [1.0, 1.0]], ({nx}, {ny}),
    dtype=real_type,
)
degree = {degree}
V = fem.functionspace(mesh, ("Lagrange", degree))

tdim = mesh.topology.dim
mesh.topology.create_connectivity(tdim - 1, tdim)
facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
dofs = fem.locate_dofs_topological(V=V, entity_dim=tdim - 1, entities=facets)

# Exact solution u_D = 1 + x^2 + 2*y^2 ⇒ -Δu = -6, so f = -6.
uD = fem.Function(V, dtype=dtype)
uD.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)
bc = fem.dirichletbc(value=uD, dofs=dofs)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(mesh, dtype(-6.0))
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx
L_fem = fem.form(L, dtype=dtype)

# Matrix-free action: M(v) = a(ui, v) with ui playing the role of
# the trial vector x. fem.assemble_vector(y, M_fem) gives y = A * ui.
ui = fem.Function(V, dtype=dtype)
M = ufl.action(a, ui)
M_fem = fem.form(M, dtype=dtype)

# RHS: b - A * x_bc via lifting.
b = fem.assemble_vector(L_fem)
ui.x.array[:] = 0.0
bc.set(ui.x.array, alpha=-1.0)
fem.assemble_vector(b.array, M_fem)
b.scatter_reverse(la.InsertMode.add)
bc.set(b.array, alpha=0.0)
b.scatter_forward()


def action_A(x, y):
    ui.x.array[:] = x.array
    ui.x.scatter_forward()
    y.array[:] = 0.0
    fem.assemble_vector(y.array, M_fem)
    y.scatter_reverse(la.InsertMode.add)
    bc.set(y.array, alpha=0.0)


def cg(comm, action_A, x, b, max_iter={max_iter}, rtol={rtol}):
    rtol2 = rtol ** 2
    nr = b.index_map.size_local

    def _gdot(v0, v1):
        return comm.allreduce(np.vdot(v0[:nr], v1[:nr]), MPI.SUM)

    y = la.vector(b.index_map, 1, dtype)
    action_A(x, y)
    r = b.array - y.array
    p = la.vector(b.index_map, 1, dtype)
    p.array[:] = r

    rnorm0 = _gdot(r, r)
    rnorm = rnorm0
    for k in range(max_iter):
        action_A(p, y)
        alpha = rnorm / _gdot(p.array, y.array)
        x.array[:] += alpha * p.array
        r -= alpha * y.array
        rnorm_new = _gdot(r, r)
        beta = rnorm_new / rnorm
        rnorm = rnorm_new
        if rnorm / rnorm0 < rtol2:
            x.scatter_forward()
            return k
        p.array[:] = beta * p.array + r
    raise RuntimeError(
        f"matrix-free CG exceeded max iterations ({{max_iter}})")


# Solve.
u = fem.Function(V, dtype=dtype)
iters = cg(mesh.comm, action_A, u.x, b)
bc.set(u.x.array, alpha=1.0)

# L2 error vs exact solution.
def L2Norm(w):
    val = fem.assemble_scalar(
        fem.form(ufl.inner(w, w) * ufl.dx, dtype=dtype))
    return float(np.sqrt(comm.allreduce(val, op=MPI.SUM)))

err = L2Norm(u - uD)
print(f"Matrix-free CG: iters={{iters}} rtol={rtol} "
      f"L2_error={{err:.4e}}")

# Output: ADIOS2 .bp directory via dolfinx VTXWriter (Lagrange OK); dolfinx.io.VTKFile writes .pvd/.vtu.
with dolfinx.io.VTXWriter(mesh.comm, "result.bp", [u], "BP4") as vtx:
    vtx.write(0.0)

summary = {{
    "iters": int(iters),
    "L2_error": err,
    "n_dofs": int(V.dofmap.index_map.size_global * V.dofmap.index_map_bs),
    "degree": int(degree),
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''
