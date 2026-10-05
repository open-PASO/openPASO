"""DUNE-fem heat conduction generators and knowledge."""


def _heat_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Heat conduction on [0,1]² — DUNE-fem."""
    nx = params.get("nx", 32)
    T_left = params.get("T_left", 100.0)
    T_right = params.get("T_right", 0.0)
    return f'''\
"""Heat conduction on [0,1]² — DUNE-fem"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC, Constant
from ufl import TrialFunction, TestFunction, SpatialCoordinate, dot, grad, dx, conditional, lt
import numpy as np
import json

gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])
space = lagrange(gridView, order=1)
x = SpatialCoordinate(space)
u = TrialFunction(space)
v = TestFunction(space)

a = dot(grad(u), grad(v)) * dx
# Zero source wrapped in a UFL Constant: a bare `0 * v * dx` folds to a domainless
# Zero ("integral is missing an integration domain"); a symbolic Constant keeps
# the measure/domain.
b = Constant(0.0) * v * dx

# Dirichlet BCs — set for your problem
bc_expr = conditional(lt(x[0], 0.01), {T_left}, conditional(lt(1.0 - x[0], 0.01), {T_right}, 0.0))
dbc = DirichletBC(space, bc_expr)

scheme = galerkin([a == b, dbc], solver="cg")
uh = space.interpolate(0, name="temperature")
scheme.solve(target=uh)

vals = np.array(uh.as_numpy)
print(f"Temperature: max={{vals.max():.6f}}")

gridView.writeVTK("result", pointdata={{"temperature": uh}})

summary = {{"max_value": float(vals.max()), "n_dofs": len(vals)}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("DUNE_TEMPLATE_COMPLETE")
'''


KNOWLEDGE = {
    "heat": {
        "description": "Heat conduction: steady and transient (backward Euler, Crank-Nicolson)",
        "solver": "Same galerkin scheme; for transient, use time-stepping loop",
        "time_stepping": (
            "Backward Euler, Crank-Nicolson (write the time loop "
            "yourself); dune-fem's Python has no ready-made "
            "Runge-Kutta stepper"),
        "required_vs_optional": {
            "REQUIRED": [
                "a mass term u*v/dt on the LEFT and the old solution "
                "u_n*v/dt on the RIGHT — a steady form with a time "
                "loop around it just re-solves the steady problem",
                "ONE scheme built OUTSIDE the loop; rebuilding it "
                "inside costs a C++ compile per step",
            ],
            "OPTIONAL": [
                "solving into a function that is NOT the one appearing "
                "in the right-hand side, then copying back. For this "
                "LINEAR implicit-Euler scheme dune-fem 2.12.0.2 reads "
                "the right-hand side once, before it overwrites the "
                "target, so solve(target=u_n) gives the same step "
                "(measured 2026-09-30: 5.8e-16 apart after 5 steps). "
                "It is NOT optional when the Dirichlet data change in "
                "time (the target's boundary dofs are set to the new "
                "values before the right-hand side is read: 6.8e-4 "
                "apart after 5 steps) or when the form is nonlinear "
                "(Newton re-reads the overwritten target: on a (1+u^2) "
                "diffusion step with nonlinear.maxiterations=50 the "
                "aliased solve returned converged=False after 50 "
                "iterations, where the separate-target solve converged "
                "in 4)",
                "dune.ufl.Constant for dt — then changing dt does NOT "
                "trigger a JIT rebuild, whereas a bare float literal "
                "in the form does",
                "solver='cg' — the implicit-Euler heat matrix is SPD",
            ],
            "NOT AVAILABLE": [
                "ready-made Runge-Kutta steppers from Python. dune-fem "
                "2.12.0.2 implements its Runge-Kutta ODE solvers only in "
                "C++ (dune/fem/solver/rungekutta: explicit Euler, TVD2 "
                "(Heun), TVD3, RK4 and a 6th-order table; implicit "
                "Euler, Gauss2 (selected as CrankNicolson), DIRK23, "
                "DIRK34; plus semi-implicit and ROW variants; SDIRK22 "
                "is accepted by the name selector but has no table, so "
                "selecting it throws NotImplemented) and has no Python "
                "binding for them. The "
                "Python wrappers (the rk module of dune.femdg with "
                "Heun/ssp2/ssp3, femdgStepper, rungeKuttaSolver) are in "
                "dune-fem-dg, a "
                "separate package that a plain dune-fem install does "
                "not include: import dune.femdg raises "
                "ModuleNotFoundError (executed 2026-08-03, re-checked "
                "2026-09-30). Write the stepper yourself.",
            ],
        },
        "verification_you_can_run": (
            "Implicit Euler is unconditionally stable and first order "
            "in time. Two checks that need no reference solution: with "
            "zero source and zero Dirichlet data the maximum must "
            "DECAY monotonically and never change sign; and halving dt "
            "must change the answer by about half as much again. If "
            "the answer does not move at all when you change dt, the "
            "mass term is missing."),
        "pitfalls": [
            (
                "[API] Non-homogeneous Dirichlet via "
                "conditional() in UFL expression. Signal: "
                "writing DirichletBC(space, T_fixed) with "
                "a constant fails to apply different "
                "values per boundary segment; the canonical "
                "DUNE pattern is "
                "DirichletBC(space, conditional(x[0] < eps, "
                "T_left, T_right)) — the UFL conditional "
                "selects values based on coordinate. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Transient: assemble mass matrix "
                "M + dt * stiffness matrix K per step. "
                "Signal: assembling only K (forgetting the "
                "mass contribution) gives a steady-state "
                "solution at every time step regardless of "
                "dt — the heat-front transient is missing. "
                "The galerkin scheme handles this when the "
                "form includes (u - u_old)/dt * v * dx + "
                "k * dot(grad(u), grad(v)) * dx. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Performance] DUNE caches compiled code — "
                "FIRST step slow (~30-60s), rest fast. "
                "Signal: a 10-step transient with first-"
                "step time of 35s and subsequent steps "
                "of 0.1s each shows the JIT cost; if "
                "every step is slow, the form is "
                "regenerating each step (form parameters "
                "changing in unintended ways forcing "
                "re-JIT). Keep form structure constant. "
                "(Audit 2026-06-02.)"
            ),
        ],
    },
}

GENERATORS = {
    "heat_2d": _heat_2d,
}
