"""DUNE-fem Poisson equation generators and knowledge.

The pitfalls in this module come from two different evidence levels and
say so in their provenance note:

  * "(Audit 2026-06-02.)"      — established by reading sources/docs.
  * "(Executed 2026-08-03 ...)" — a script was run against the
    installed dune-fem 2.12.0.2 and the quoted numbers and messages
    are that run's output. Those entries live in
    ``verified_api.EXECUTED_PITFALLS`` and are appended to the list
    below; see that module for the full measured API surface.
"""

from .verified_api import EXECUTED_PITFALLS


def _poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Poisson -Δu = f on [0,1]², u=0 on boundary — DUNE-fem with UFL."""
    nx = params.get("nx", 32)
    f_val = params.get("f", 1.0)
    order = params.get("order", 1)
    return f'''\
"""Poisson -Δu = {f_val} on [0,1]², u=0 on boundary — DUNE-fem (UFL)"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import TrialFunction, TestFunction, dot, grad, dx
import numpy as np
import json

gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])
space = lagrange(gridView, order={order})
u = TrialFunction(space)
v = TestFunction(space)

a = dot(grad(u), grad(v)) * dx
b = {f_val} * v * dx

dbc = DirichletBC(space, 0)
scheme = galerkin([a == b, dbc], solver="cg")
uh = space.interpolate(0, name="solution")
scheme.solve(target=uh)

vals = np.array(uh.as_numpy)
max_val = float(vals.max())
print(f"max(u) = {{max_val:.10f}}")
print(f"DOFs: {{len(vals)}}")

gridView.writeVTK("result", pointdata={{"phi": uh}})

summary = {{"max_value": max_val, "n_dofs": len(vals), "element_type": "Q1 quad"}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("DUNE-fem Poisson solve complete.")
print("DUNE_TEMPLATE_COMPLETE")
'''


KNOWLEDGE = {
    "poisson": {
        "description": "Poisson with DUNE-fem using UFL (same forms as FEniCS)",

        "required_calls_in_order": [
            "gridView = dune.grid.structuredGrid([0,0],[1,1],[n,n])",
            "space = dune.fem.space.lagrange(gridView, order=k)",
            "u, v = ufl.TrialFunction(space), ufl.TestFunction(space)",
            "a = ufl.dot(ufl.grad(u), ufl.grad(v))*ufl.dx",
            "b = f*v*ufl.dx",
            "dbc = dune.ufl.DirichletBC(space, <value>[, <indicator>])",
            "scheme = dune.fem.scheme.galerkin([a == b, dbc], "
            "solver='cg')   <- dbc MUST be in this list",
            "uh = space.interpolate(0, name='solution')",
            "info = scheme.solve(target=uh)   <- writes INTO uh",
        ],
        "required_vs_optional": {
            "REQUIRED": [
                "every DirichletBC as an ELEMENT of the galerkin list — "
                "there is no bc.apply()",
                "an initial guess of the right shape: interpolate(0) "
                "for a scalar space, interpolate([0,0]) for dimRange=2",
                "target=uh on solve(); the return value is an info "
                "dict, not the solution",
            ],
            "OPTIONAL": [
                "solver='cg' — the default works too; cg is the right "
                "choice because the Poisson matrix is SPD",
                "parameters={'linear.tolerance': ..., "
                "'linear.maxiterations': ...}",
                "an indicator argument to DirichletBC — omitting it "
                "applies the value to the WHOLE boundary",
                "gridView.writeVTK(...)",
            ],
        },
        "boundary_conditions": {
            "Dirichlet (essential)": (
                "dune.ufl.DirichletBC(space, value, indicator). The "
                "indicator is a UFL conditional on SpatialCoordinate "
                "evaluated PER BOUNDARY FACET, e.g. "
                "conditional(lt(x[0], 1e-8), 1, 0). Omit it for the "
                "whole boundary. Vector spaces take a LIST whose None "
                "entries leave that component free."),
            "Neumann (natural)": (
                "Add a boundary integral to the right-hand side, "
                "masked by a conditional: "
                "b += g*conditional(gt(x[0], 1-1e-8), 1.0, 0.0)*v*ds. "
                "Verified exact on -Laplace(u)=0, u=0 on x=0, "
                "du/dn=1 on x=1 (answer u=x): "
                "||u_h - x||_L2 = 3.233e-16."),
            "Robin": (
                "a += alpha*mask*u*v*ds and b += alpha*u_inf*mask*v*ds "
                "for du/dn + alpha*(u - u_inf) = 0. Verified exact "
                "with alpha=3, u_inf=2 (answer u=1.5x): "
                "||u_h - 1.5x||_L2 = 3.238e-16."),
            "ds(id) — geometric boundary ids": (
                "ds(id) IS supported. The ids are geometric and "
                "1-based, not user-assigned: for a 2D YaspGrid "
                "(structuredGrid) they are 1=left (x-min), 2=right "
                "(x-max), 3=bottom (y-min), 4=top (y-max), because "
                "dune/fem/misc/boundaryidprovider.hh returns "
                "intersection.indexInInside()+1. Measured on an 8x8 "
                "grid: 1*v*ds(k) sums to exactly 1.0 for k=1..4, to "
                "0.0 for k=5, and plain 1*v*ds sums to 4.0. ds(0) "
                "raises AssertionError. ALUGrid uses a different "
                "provider (the mesh file's own ids), so verify the "
                "mapping before relying on it for an imported mesh. "
                "The coordinate-conditional form above is the "
                "portable option."),
        },
        "solver": "galerkin([a == b, dbc], solver='cg') — Newton-Krylov internally",
        "spaces": "lagrange(gridView, order=k) — Lagrange any order",
        "mesh": "structuredGrid (YaspGrid), ALUGrid (adaptive), Gmsh import",
        "verification_you_can_run": (
            "Two checks that need no reference solution. (1) PATCH "
            "TEST: pick any polynomial of degree <= k, impose it as "
            "DirichletBC(space, p) on the whole boundary with "
            "f = -div(grad(p)), and the discrete solution must equal "
            "it to solver tolerance — measured 1e-15 relative on the "
            "vector version of this test. (2) LINEAR NEUMANN: u=0 on "
            "x=0 and du/dn=1 on x=1 with f=0 gives u=x exactly; "
            "measured ||u_h - x||_L2 = 3.2e-16. Either check fails "
            "loudly if a boundary condition was dropped, which is the "
            "single most common DUNE-fem mistake. For an order study, "
            "expect the textbook rates (L2 order k+1, H1 order k) and "
            "MEASURE them — do not assume."),
        "pitfalls": [
            (
                "[API] DUNE-fem uses UFL — same syntax as "
                "FEniCS/dolfinx. Weak forms are largely "
                "interchangeable. Signal: a UFL form that "
                "compiles cleanly in dolfinx will also "
                "compile in DUNE-fem, but the FUNCTION SPACE "
                "construction differs: dune.fem.space.lagrange("
                "gridView, order=k) vs dolfinx.fem."
                "functionspace(domain, ('Lagrange', k)). "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Performance] First DUNE-fem run triggers "
                "JIT compilation of C++ code. Signal: a "
                "'simple' Poisson solve prints DUNE-INFO: "
                "Compiling <X> (new) on stderr — assembled by "
                "Python logging from dune/common/__init__.py:49, "
                "'DUNE-%(levelname)s: %(message)s', around the "
                "f-string at dune/generator/cmakebuilder.py:378, "
                "so the whole line is in no file — and then "
                "produces no output for tens of seconds to "
                "minutes; the second run is milliseconds. "
                "The cache is <sys.prefix>/.cache/dune-py "
                "(inside a venv/conda env) or "
                "~/.cache/dune-py, NOT ~/.dune — see "
                "dune.packagemetadata.getDunePyDir(). "
                "(Corrected by execution 2026-08-03 on "
                "dune-fem 2.12.0.2: getDunePyDir() returned "
                "the conda env's .cache/dune-py and ~/.dune "
                "does not exist; the earlier ~/.dune claim "
                "was wrong for this version.)"
            ),
            (
                "[Performance] Subsequent runs use CACHED "
                "compiled code — much faster. Signal: "
                "deleting the dune-py cache directory forces "
                "a full recompile and resets the cost. "
                "MEASURED to cost a new build: a different "
                "grid class, a "
                "direct instead of a Krylov solver, and ANY "
                "change to the form — including changing a "
                "bare float literal inside it (7.3125 -> "
                "9.8125 cost 25.204 s and added one .so). "
                "MEASURED free: re-running the identical form "
                "(0.051 s, no new .so), changing the "
                ".value of a dune.ufl.Constant (0.00056 s, no "
                "new .so), and changing the Lagrange order "
                "within 1-6: lagrange() uses one dynamic-order "
                "space type up to order 6 and the form does "
                "not depend on the order, so orders 1-6 reused "
                "the same space and scheme modules with no new "
                ".so, while order 7 compiled a new space "
                "(136.6 s) and a new scheme (94.2 s). (Audit "
                "2026-06-02; rebuild triggers measured by "
                "execution 2026-08-03, the Lagrange order "
                "2026-09-30.)"
            ),
            (
                "[API] Use dune.ufl.DirichletBC (NOT "
                "dolfinx.fem.dirichletbc). Signal: "
                "importing DirichletBC from dolfinx in a "
                "DUNE-fem script raises ImportError or "
                "wrong-signature TypeError — the DUNE "
                "constructor takes (space, value), not the "
                "dolfinx (V, value, dofs) triple. Use "
                "from dune.ufl import DirichletBC. (Audit "
                "2026-06-02.)"
            ),
            (
                "[API] VTK output: gridView.writeVTK('name', "
                "pointdata={'field': uh}). Signal: writing "
                "with the dolfinx io.VTXWriter / XDMFFile "
                "API fails — dune.grid gridView has its own "
                "writeVTK method (NOT dolfinx). The galerkin "
                "lagrange space's GridFunction is written "
                "via gridView.writeVTK; outputs .vtu "
                "directly readable in ParaView. (Audit "
                "2026-06-02.)"
            ),
            (
                "[API] structuredGrid creates CUBE "
                "elements (quadrilateral in 2D, hexahedron "
                "in 3D), NOT simplices, and dune.grid has "
                "no option that yields simplices. Signal: "
                "count the cells — an n-by-n domain gives "
                "n^2 elements through structuredGrid and "
                "2*n^2 through aluConformGrid / "
                "aluSimplexGrid. Do NOT ask the space: its "
                "UFL cell always says 'triangle'. For "
                "simplices use dune.alugrid (or a Gmsh "
                "import through it). (Executed 2026-08-03 "
                "on dune-fem 2.12.0.2: "
                "structuredGrid([0,0],[1,1],[4,4]) -> 16 "
                "elements of type 'quadrilateral'; "
                "aluConformGrid on the same "
                "cartesianDomain -> 32 of type 'triangle'. "
                "The earlier 'wrong sparsity' wording was "
                "dropped — nothing about the assembled "
                "matrix is wrong, the mesh is simply made "
                "of different cells.)"
            ),
            (
                "[API] Use dune.ufl.Constant(value, "
                "name=...), NOT ufl.Constant. Signal: plain "
                "ufl.Constant needs a domain and fails "
                "without one; dune.ufl.Constant takes a "
                "bare value and — unlike a Python float "
                "baked into the form — is RUNTIME-"
                "UPDATABLE: assigning c.value = ... and "
                "calling scheme.solve() again reuses the "
                "compiled module instead of triggering a "
                "JIT rebuild. A Python scalar literal also "
                "works but re-compiles whenever it changes. "
                "(Audit 2026-06-02; the dune.ufl.Constant "
                "correction and the reuse behaviour "
                "verified by execution 2026-08-03.)"
            ),
            *EXECUTED_PITFALLS,
        ],
    },
}

GENERATORS = {
    "poisson_2d": _poisson_2d,
}
