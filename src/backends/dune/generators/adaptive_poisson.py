"""DUNE-fem adaptive Poisson generators and knowledge."""


def _adaptive_poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    h-adaptive Poisson with residual-based error estimator."""
    order = params.get("order", 1)
    max_level = params.get("max_refinement_level", 8)
    tol = params.get("tolerance", 1e-6)
    n_adapt_steps = params.get("adapt_steps", 10)
    return f'''\
"""Adaptive Poisson: -Δu = f with h-refinement — DUNE-fem (ALUGrid)"""
import dune.fem
from dune.grid import cartesianDomain
from dune.alugrid import aluConformGrid
from dune.fem.view import adaptiveLeafGridView
from dune.fem.space import lagrange, finiteVolume
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import (TrialFunction, TestFunction, SpatialCoordinate, dot, grad, dx,
                 conditional, lt, sqrt)
import numpy as np
import json

# Local h-refinement needs an ADAPTIVE view over an ALUGrid. Neither
# half is optional: on a YaspGrid (structuredGrid) dune.fem.adapt and
# globalRefine(level, uh) do nothing at all and raise nothing, and on a
# PLAIN ALUGrid leaf view dune.fem.adapt raises "the grid views for all
# discrete functions need to support adaptivity".
gridView = adaptiveLeafGridView(
    aluConformGrid(cartesianDomain([0, 0], [1, 1], [8, 8]), dimgrid=2))

space = lagrange(gridView, order={order})
x = SpatialCoordinate(space)
u = TrialFunction(space)
v = TestFunction(space)

# Source term with sharp feature to drive adaptivity — set for your problem
f_expr = conditional(
    lt((x[0]-0.5)**2 + (x[1]-0.5)**2, 0.01),
    100.0, 1.0
)

a = dot(grad(u), grad(v)) * dx
b = f_expr * v * dx

dbc = DirichletBC(space, 0)
scheme = galerkin([a == b, dbc], solver="cg")
uh = space.interpolate(0, name="solution")

# The indicator must be a DISCRETE FUNCTION on the adaptive view:
# dune.fem.mark(..., gridView=gv) raises AttributeError unconditionally
# in dune-fem 2.12.0.2 (upstream typo in GridMarker.__init__), so the
# marker has to take the grid view from the indicator itself.
fv = finiteVolume(gridView)

# Adaptive refinement loop — replace the gradient-jump proxy below with
# a residual estimator eta_K^2 = h_K^2*||f + Delta u||^2_K
# + h_K*||[grad(u).n]||^2_edges for your problem.
for adapt_step in range({n_adapt_steps}):
    info = scheme.solve(target=uh)
    vals = np.array(uh.as_numpy)
    print(f"Adapt step {{adapt_step+1}}: elements={{gridView.size(0)}}, "
          f"DOFs={{len(vals)}}, max(u)={{float(vals.max()):.8f}}, "
          f"converged={{info['converged']}}")
    if adapt_step == {n_adapt_steps} - 1:
        break
    indicator = fv.interpolate(sqrt(dot(grad(uh), grad(uh))), name="indicator")
    theta = float(np.array(indicator.as_numpy).max()) * 0.5
    before = gridView.size(0)
    dune.fem.mark(indicator, theta)
    dune.fem.adapt([uh])          # resizes the space AND prolongs uh
    if gridView.size(0) <= before:
        # Refinement that silently does nothing is the failure this
        # template exists to avoid — do not swallow it.
        raise RuntimeError(
            f"adaptation refined nothing: {{before}} -> {{gridView.size(0)}} "
            f"elements at step {{adapt_step+1}}")

vals = np.array(uh.as_numpy)
max_val = float(vals.max())
n_dofs = len(vals)
print(f"Final: elements={{gridView.size(0)}}, DOFs={{n_dofs}}, "
      f"max(u)={{max_val:.10f}}")

gridView.writeVTK("result", pointdata={{"phi": uh}})
summary = {{
    "max_value": max_val, "n_dofs": n_dofs,
    "n_elements": gridView.size(0),
    "adapt_steps": {n_adapt_steps}, "order": {order},
}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Adaptive Poisson solve complete.")
print("DUNE_TEMPLATE_COMPLETE")
'''


KNOWLEDGE = {
    "adaptive_poisson": {
        "description": "h-adaptive Poisson with residual error estimator and ALUGrid",

        "required_calls_in_order": [
            "from dune.alugrid import aluConformGrid",
            "from dune.fem.view import adaptiveLeafGridView",
            "gridView = adaptiveLeafGridView(aluConformGrid("
            "cartesianDomain([0,0],[1,1],[n,n]), dimgrid=2))"
            "   <- BOTH halves are required",
            "space = lagrange(gridView, order=k)   "
            "<- build the space ON THE ADAPTIVE VIEW",
            "scheme = galerkin([a == b, dbc], solver='cg'); "
            "uh = space.interpolate(0, name='u'); "
            "scheme.solve(target=uh)",
            "fv = dune.fem.space.finiteVolume(gridView)",
            "indicator = fv.interpolate(<estimator expression>, "
            "name='ind')   <- a DISCRETE FUNCTION, not a UFL expression",
            "dune.fem.mark(indicator, tol)   <- statement 1, returns "
            "statistics, NOT a marker",
            "dune.fem.adapt([uh])   <- statement 2, resizes the space "
            "and PROLONGS uh",
            "assert gridView.size(0) > before   <- the only way to "
            "know it did anything",
        ],
        "required_vs_optional": {
            "REQUIRED": [
                "an ALUGrid (dune.alugrid) — YaspGrid/structuredGrid "
                "cannot refine locally",
                "adaptiveLeafGridView() around it — a plain ALUGrid "
                "leaf view makes dune.fem.adapt raise",
                "a DISCRETE FUNCTION indicator (finiteVolume space) — "
                "the marker reads the grid view off the indicator",
                "mark and adapt as TWO separate statements with "
                "nothing passed between them",
                "an assertion that the element count actually grew",
            ],
            "OPTIONAL": [
                "coarsenTolerance / minLevel / maxLevel / minVolume / "
                "maxVolume kwargs on dune.fem.mark",
                "dune.fem.doerflerMark(indicator, theta) for bulk "
                "(Doerfler) marking instead of a threshold — present "
                "in dune.fem, NOT exercised here",
                "dune.fem.loadBalance for parallel runs",
            ],
            "MUST NOT": [
                "dune.fem.mark(..., gridView=gv) — raises "
                "AttributeError unconditionally on 2.12.0.2 (upstream "
                "defect); omit the kwarg",
                "feeding mark()'s return value into adapt()",
                "dune.fem.globalRefine(level, uh) on a YaspGrid — "
                "silent no-op",
            ],
        },
        "verification_you_can_run": (
            "Adaptivity has one failure mode that dwarfs the others: "
            "doing nothing. Record gridView.size(0), space.size and "
            "max(uh.as_numpy) BEFORE and AFTER every adapt call and "
            "assert the element count grew — the working cycle was "
            "measured to take a 32-element aluConformGrid to 48 "
            "elements with the P1 space going 25 -> 33 dofs and uh "
            "prolonged onto it. Then check the physics: for a source "
            "with a sharp feature the refined elements must cluster "
            "around it, and max(u) must settle as the mesh grows. If "
            "max(uh.as_numpy) drops to 0 after a refinement you used "
            "the globalRefine-through-the-hierarchical-grid path, "
            "which ZEROES live discrete functions."),
        "solver": "galerkin scheme on adaptive grid with mark/refine/coarsen cycle",
        "spaces": "lagrange(gridView, order=k) on adaptiveLeafGridView",
        "mesh": ("ALUGrid (dune-alugrid from PyPI, installed as a "
                 "dependency of dune-fem) for local h-refinement"),
        "exit_code_warning": (
            "Adaptive DUNE runs frequently exit 134 AFTER printing "
            "every correct result: the abort happens in ALUGrid "
            "destructors during interpreter teardown, and the "
            "restrict-prolong path (globalRefine(level, uh) on an "
            "adaptive ALUGrid view) did it on 7 of 7 runs with "
            "byte-identical stdout. Judge such a run by its results "
            "file and by the terminal sentinel line the templates "
            "print, not by the return code."),
        "pitfalls": [
            (
                "[API] ALUGrid supports TRUE LOCAL "
                "refinement; structuredGrid (YaspGrid) "
                "supports only GLOBAL refinement. Signal: "
                "the failure is an AttributeError, not a "
                "grid diagnostic — measured 2026-08-03, "
                "hasattr(structuredGrid(...), 'mark') is "
                "False, so gridView.mark(...) dies on the "
                "PYTHON attribute lookup before any grid "
                "code runs. Note that 'mark' IS present on "
                "adaptiveLeafGridView(yaspView) and on "
                "gridView.hierarchicalGrid (both measured "
                "True), so its presence is NOT evidence "
                "that local refinement will happen; use "
                "aluConformGrid / aluCubeGrid / "
                "aluSimplexGrid for real adaptivity. "
                "CORRECTED by adversarial audit 2026-08-03: "
                "an earlier revision of this Signal quoted "
                "the message 'grid does not support local "
                "refinement', which occurs NOWHERE under "
                "site-packages/dune or include/dune, and "
                "told the reader to switch to 'alucubeGrid' "
                "or 'alusimplexGrid', neither of which "
                "exists — the factories are camelCase "
                "(hasattr(dune.alugrid,'alusimplexGrid') "
                "is False, aluSimplexGrid is True). "
                "(Audit 2026-06-02; Signal re-measured "
                "2026-08-03.)"
            ),
            (
                "[Numerical] Error estimator: eta_K^2 = "
                "h_K^2 * ||f + Δu||^2 + h_K * ||[grad(u)·"
                "n]||^2. Signal: omitting the jump term "
                "[grad(u).n] across facets under-estimates "
                "the error in irregular meshes by 5-30%; "
                "the residual-only estimator misses jumps "
                "that signal under-resolved interior "
                "layers. Use the full residual + jump "
                "estimator for reliable adaptivity. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[API] There is no space.update() in "
                "dune-fem 2.12 — hand the discrete "
                "functions to the adaptation call and it "
                "resizes the space and prolongs them for "
                "you: dune.fem.mark(indicator, tol) then "
                "dune.fem.adapt([uh, ...]) (or "
                "dune.fem.globalRefine(level, uh)). "
                "Signal: hasattr(space, 'update') is "
                "False and no \"update\" binding exists "
                "under include/dune/fempy/, so a tutorial "
                "line calling it dies with AttributeError "
                "rather than the 'function and space "
                "mismatch' the older catalog text "
                "predicted. (Audit 2026-06-02; "
                "space.update() FALSIFIED by execution "
                "2026-08-03 on dune-fem 2.12.0.2, where "
                "the working cycle measured 32 -> 48 "
                "elements and lagrange(order=1).size "
                "25 -> 33 with uh prolonged "
                "automatically.)"
            ),
            (
                "[API] dune.fem.mark(indicator, tol, "
                "gridView=gv) is BROKEN in dune-fem "
                "2.12.0.2: GridMarker.__init__ stores "
                "self._gridView but validates "
                "self.gridView, and the class has no such "
                "attribute. Signal: AttributeError "
                "\"'GridMarker' object has no attribute "
                "'gridView'\" fires unconditionally — "
                "measured on a YaspGrid and on an ALUGrid "
                "alike — so it is not a symptom of a bad "
                "indicator. Omit the kwarg and pass a "
                "DISCRETE-FUNCTION indicator (e.g. "
                "finiteVolume(gv).interpolate(...)) so the "
                "marker can take the grid view from it — and "
                "note that the view must itself be adaptive, "
                "see the adaptiveLeafGridView pitfall below: "
                "on a plain YaspGrid the same call then "
                "raises AttributeError 'indicator function "
                "must be over grid view that supports "
                "adaptation'. dune.fem.markNeighbors "
                "forwards the same kwarg and was measured to "
                "raise the identical GridMarker "
                "AttributeError. (Executed 2026-08-03.)"
            ),
            (
                "[API] dune.fem.mark() RETURNS THE MARKING "
                "STATISTICS, not a marker: it constructs a "
                "GridMarker, calls it immediately, and "
                "hands back the (nRefined, nCoarsened) "
                "tuple — (-1, -1) unless statistics=True. "
                "Signal: passing that return value on as "
                "dune.fem.adapt(marker, [uh]) raises "
                "AssertionError 'only one list of discrete "
                "functions can be passed into the "
                "adaptation method', because a tuple is "
                "not callable and gridAdapt re-dispatches "
                "it as the first discrete function. Write "
                "the two statements independently: "
                "dune.fem.mark(ind, tol); "
                "dune.fem.adapt([uh]). (Executed "
                "2026-08-03.)"
            ),
            (
                "[API] Adaptivity needs "
                "dune.fem.view.adaptiveLeafGridView, not "
                "just an ALUGrid. Signal: "
                "dune.fem.adapt([uh]) on a plain "
                "aluConformGrid leaf view raises "
                "AssertionError 'the grid views for all "
                "discrete functions need to support "
                "adaptivity', and reading .canAdapt on that "
                "view raises AttributeError, while "
                "adaptiveLeafGridView(aluConformGrid(...))"
                ".canAdapt is True. Build the spaces on the "
                "wrapped view. Note the check only runs for "
                "the LIST form — passing a SINGLE discrete "
                "function skips it, which is why "
                "globalRefine(level, uh) can fail silently. "
                "(Executed 2026-08-03.)"
            ),
            (
                "[Numerical] Doerfler marking: refine the "
                "SMALLEST set of elements that captures "
                "theta fraction (typical 0.25-0.5) of "
                "total error. Signal: when running on "
                "alugrid via the dune.fem mark/adapt "
                "loop, theta < 0.1 refines too few "
                "elements per pass (slow convergence to "
                "target tolerance); theta > 0.7 refines "
                "almost-uniformly (defeats the adaptive "
                "benefit). 0.3 is a common default. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] For COARSENING: mark elements "
                "with SMALL error indicator. Signal: a "
                "moving-front problem with monotonic "
                "refinement-only accumulates elements; "
                "after the front passes, those refined "
                "regions are over-resolved. Mark elements "
                "with eta_K < theta_coarse * max(eta) for "
                "coarsening (typical theta_coarse ~ 0.01-"
                "0.05). (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Nested iteration: use coarse-"
                "grid solution as initial guess on the "
                "fine alugrid. Signal: starting the "
                "dune.fem galerkin scheme Newton from "
                "zero on the fine lagrange space for a "
                "nonlinear problem takes 5-10 iterations; "
                "starting from the interpolated coarse "
                "solution converges in 1-2 iterations "
                "because the initial guess is already in "
                "the convergence basin. "
                "(Audit 2026-06-02.)"
            ),
        ],
    },
}

GENERATORS = {
    "adaptive_poisson_2d": _adaptive_poisson_2d,
}
