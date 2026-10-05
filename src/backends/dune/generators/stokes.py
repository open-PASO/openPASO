"""DUNE-fem Stokes flow generator and knowledge.

HISTORY. Until 2026-08-03 this generator emitted a script that said so
itself — "For a simplified approach, solve Poisson as a proxy" — built
a scalar Lagrange space, solved -Laplace(u) = 1 and printed
"Stokes proxy: max=...". It exited 0, so nothing downstream could tell
it apart from a Stokes solve. It is replaced by a real Taylor-Hood
saddle-point solve, executed against dune-fem 2.12.0.2 and checked
against plane Poiseuille flow.
"""


def _stokes_2d_dune(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Stokes flow, Taylor-Hood P2/P1 on a composite space — DUNE-fem.
    """
    nx = params.get("nx", 12)
    mu = params.get("mu", 1.0)
    u_max = params.get("u_max", 1.0)
    order_v = params.get("order_v", 2)
    order_p = params.get("order_p", 1)
    return f'''\
"""Stokes flow on [0,1]^2 — Taylor-Hood P{order_v}/P{order_p} — DUNE-fem.

Plane Poiseuille channel: parabolic inflow on x=0, no-slip on y=0 and
y=1, natural ("do-nothing") outflow on x=1.

Closed-form answer this script checks itself against:
    u = (4*U*y*(1-y), 0)        p = 8*mu*U*(1-x)
Taylor-Hood contains that field exactly (velocity quadratic, pressure
linear), so a CORRECT run reproduces it to solver tolerance — the
relative errors printed below should be ~1e-14, not ~1e-3.
"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange, composite
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from dune.fem import integrate
from ufl import (TrialFunction, TestFunction, SpatialCoordinate, as_vector,
                 inner, grad, div, dx, conditional, lt, gt)
import numpy as np
import json

mu = {mu}          # dynamic viscosity
U = {u_max}        # centreline velocity of the inflow profile

gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])

# REQUIRED: one COMPOSITE space holding velocity and pressure, so the
# saddle-point block matrix is assembled. Two separate spaces give two
# decoupled solves. dune.fem.space.product is NOT the same factory: it is
# dune-fem's decoupled variant, and galerkin() on it raises TypeError.
V = lagrange(gridView, dimRange=2, order={order_v})   # velocity
Q = lagrange(gridView, order={order_p})               # pressure
W = composite(V, Q, components=["velocity", "pressure"])

# TrialFunction(W) is ONE argument of shape (3,) — ufl.TrialFunctions(W)
# does NOT split it into (velocity, pressure). Slice it by hand.
trial = TrialFunction(W)
test = TestFunction(W)
u = as_vector([trial[0], trial[1]])
p = trial[2]
v = as_vector([test[0], test[1]])
q = test[2]

x = SpatialCoordinate(W)
tol = 1e-8

# Stokes: mu*(grad u, grad v) - (p, div v) - (q, div u) = 0
a = (mu * inner(grad(u), grad(v)) - p * div(v) - q * div(u)) * dx

u_in = 4.0 * U * x[1] * (1.0 - x[1])

# None in the value list leaves that component FREE — the pressure must
# stay free on every Dirichlet boundary.
bc_inflow = DirichletBC(W, [u_in, 0, None],
                        conditional(lt(x[0], tol), 1, 0))
bc_walls = DirichletBC(W, [0, 0, None],
                       conditional(lt(x[1], tol), 1,
                                   conditional(gt(x[1], 1 - tol), 1, 0)))
# x=1 is left alone: the natural BC of this form is stress-free
# outflow, which is what fixes the pressure level.

# The Stokes matrix is INDEFINITE. cg, gmres and bicgstab (the only
# Krylov names dune-fem accepts here) do all converge on it, but they
# need 1e3-4e4 iterations and land around 1e-10; the direct solver
# needs one and lands at 1e-15. Use the direct one in 2D.
scheme = galerkin([a == 0, bc_inflow, bc_walls],
                  solver=("suitesparse", "umfpack"),
                  parameters={{"linear.tolerance": 1e-12}})

wh = W.interpolate([0, 0, 0], name="solution")
info = scheme.solve(target=wh)

uh = as_vector([wh[0], wh[1]])
ph = wh[2]

u_exact = as_vector([u_in, 0])
p_exact = 8.0 * mu * U * (1.0 - x[0])

err_u = np.sqrt(integrate(inner(uh - u_exact, uh - u_exact),
                          gridView=gridView, order=6))
nrm_u = np.sqrt(integrate(inner(u_exact, u_exact),
                          gridView=gridView, order=6))
err_p = np.sqrt(integrate((ph - p_exact) ** 2, gridView=gridView, order=6))
nrm_p = np.sqrt(integrate(p_exact ** 2, gridView=gridView, order=6))
div_u = np.sqrt(integrate(div(uh) ** 2, gridView=gridView, order=6))

print(f"converged={{info['converged']}} "
      f"linear_iterations={{info['linear_iterations']}}")
print(f"dofs (velocity+pressure): {{W.size}}")
print(f"relative velocity error vs Poiseuille: {{err_u / nrm_u:.3e}}")
print(f"relative pressure error vs Poiseuille: {{err_p / nrm_p:.3e}}")
print(f"||div u||_L2 (incompressibility): {{div_u:.3e}}")

gridView.writeVTK("result", pointdata={{"velocity": uh, "pressure": ph}})
summary = {{
    "rel_error_velocity": float(err_u / nrm_u),
    "rel_error_pressure": float(err_p / nrm_p),
    "div_u_l2": float(div_u),
    "n_dofs": int(W.size),
    "mu": mu, "U": U,
}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("DUNE-fem Stokes solve complete.")
print("DUNE_TEMPLATE_COMPLETE")
'''


KNOWLEDGE = {
    "stokes": {
        "description": (
            "Incompressible Stokes flow as a real saddle-point system "
            "on one composite (Taylor-Hood) space."),

        "required_calls_in_order": [
            "V = dune.fem.space.lagrange(gridView, dimRange=2, order=2)",
            "Q = dune.fem.space.lagrange(gridView, order=1)",
            "W = dune.fem.space.composite(V, Q, "
            "components=['velocity','pressure'])",
            "trial = ufl.TrialFunction(W); test = ufl.TestFunction(W)"
            "   <- ONE argument of shape (3,)",
            "u = ufl.as_vector([trial[0], trial[1]]); p = trial[2]",
            "a = (mu*inner(grad(u),grad(v)) - p*div(v) - q*div(u))*dx",
            "bc = dune.ufl.DirichletBC(W, [ux, uy, None], <indicator>)"
            "   <- None keeps the pressure component free",
            "scheme = dune.fem.scheme.galerkin([a == 0, bc], "
            "solver=('suitesparse','umfpack'))",
            "wh = W.interpolate([0,0,0], name='solution'); "
            "scheme.solve(target=wh)",
            "u_h = ufl.as_vector([wh[0], wh[1]]); p_h = wh[2]",
        ],
        "required_vs_optional": {
            "REQUIRED": [
                "composite() — two separate spaces give two decoupled "
                "solves, not Stokes. product() is NOT an alias: dune-fem "
                "documents it as the space for a DECOUPLED solve, its "
                "discrete functions are tuple functions (.components) "
                "with no linear-operator or solver storage, and "
                "galerkin() on a product space raises TypeError at "
                "construction ('NoneType' object is not callable for "
                "solver=None or 'gmres'; measured 2026-09-30 on dune-fem "
                "2.12.0.2 with this template's Taylor-Hood form)",
                "an LBB-stable pair: velocity order strictly higher "
                "than pressure order (2/1 is the safe default)",
                "nothing about the solver, strictly — but "
                "solver=('suitesparse','umfpack') is the right "
                "default: measured 1 linear iteration and 1e-15 "
                "relative error, against 1.3e3-9.4e4 iterations and "
                "~1e-12..1e-13 for cg/gmres/bicgstab on the same "
                "problem (re-measured 2026-08-03)",
                "None for the pressure entry of every DirichletBC "
                "value list",
                "one boundary left free (natural stress-free outflow) "
                "OR an explicit pressure constraint — with Dirichlet "
                "velocity on the WHOLE boundary the pressure is only "
                "determined up to a constant",
            ],
            "OPTIONAL": [
                "components=['velocity','pressure'] — only affects "
                "names in the VTK file",
                "gridView.writeVTK(..., pointdata={'velocity': u_h, "
                "'pressure': p_h})",
                "parameters={'linear.tolerance': ...} — a direct "
                "solver ignores it",
            ],
        },
        "verification_you_can_run": (
            "Plane Poiseuille. Parabolic inflow u = (4*U*y*(1-y), 0) on "
            "x=0, no-slip on y=0 and y=1, nothing on x=1. The answer is "
            "u = (4*U*y*(1-y), 0) everywhere and p = 8*mu*U*(1-x). "
            "Taylor-Hood P2/P1 contains BOTH exactly, so a correct "
            "implementation reproduces them to solver tolerance rather "
            "than to discretisation accuracy. Executed 2026-08-03 on a "
            "12x12 structuredGrid with the ('suitesparse','umfpack') "
            "solver: relative velocity error 9.5e-16, relative pressure "
            "error 1.1e-14, ||div u||_L2 4.2e-15. If your Stokes run "
            "gives 1e-3 on this problem then the element pair, the sign "
            "of a coupling term or the outflow BC is wrong — refinement "
            "will not fix it."),

        "pitfalls": [
            (
                "[API] ufl.TrialFunctions(W) does NOT split a dune-fem "
                "composite space. Signal: on "
                "composite(lagrange(dimRange=2, order=2), "
                "lagrange(order=1)) it returns a 1-TUPLE whose single "
                "entry has shape (3,) — measured — so the dolfinx "
                "idiom (u, p) = TrialFunctions(W) binds u to the whole "
                "3-vector and blows up later, somewhere else. Take "
                "TrialFunction(W) and slice it: "
                "u = as_vector([t[0], t[1]]); p = t[2]. (Executed "
                "2026-08-03 on dune-fem 2.12.0.2.)"
            ),
            (
                "[Performance] Use the DIRECT solver by default, but "
                "do not believe anyone (including an earlier revision "
                "of this entry) who tells you the Krylov methods "
                "cannot solve a saddle-point system — measured, all "
                "three do. Signal: on the same 12x12 Taylor-Hood "
                "Poiseuille problem, 2026-08-03, "
                "solver=('suitesparse','umfpack') gave "
                "linear_iterations 1 and relative L2 error 9.465e-16, "
                "while solver='cg' converged in 1343 linear "
                "iterations to 3.837e-13, 'gmres' in 94026 to "
                "2.836e-12 and 'bicgstab' in 43953 to 1.430e-12; with "
                "an explicit 'linear.preconditioning.method' of 'ssor' "
                "gmres needed 8587 and with 'jacobi' 18636, while "
                "'none' reproduced the default 94026 EXACTLY — so the "
                "default preconditioner on this install IS 'none'. "
                "(RE-MEASURED by adversarial audit 2026-08-03 against "
                "the shipped template at nx=12, dofs 1419, "
                "linear.tolerance 1e-12, and stable across repeat runs. "
                "An earlier revision of this entry recorded 1150 / "
                "2527 / 37167 iterations and ~1e-10..1e-11 errors, and "
                "separately 70941 for 'none' and 6351 for 'ssor'; none "
                "of those six figures reproduced, and the 2527-vs-70941 "
                "pair was self-contradictory once 'none' was measured "
                "to BE the default. Treat the counts as "
                "order-of-magnitude, not as digits.) So the detector "
                "is not failure, it is COST and ACCURACY: four- to "
                "five-figure iteration counts and five orders of "
                "magnitude more error than the direct solve on a "
                "few-thousand-dof problem. The accepted Krylov names "
                "are exactly cg, gmres, bicgstab — anything else "
                "raises RuntimeError 'ParameterInvalid', then a "
                "'Valid values are:' line listing gmres, cg, "
                "bicgstab. DUNE_THROW composes the whole thing at "
                "run time — the exception name, then "
                "[getEnumeration:<path>/dune/fem/io/parameter/"
                "reader.hh:300], then the key and the value list, "
                "each written by a separate stream insertion at "
                "reader.hh:290-296 and separated by NEWLINES, not "
                "spaces — so no line of it exists as a literal and a "
                "one-line quote of it can never be matched. "
                "(Executed 2026-08-03 on "
                "dune-fem 2.12.0.2 and re-measured 2026-08-13, where "
                "galerkin(..., solver='istl') — an unrecognised "
                "method name — reproduced it verbatim. The earlier "
                "wording, 'cg cannot solve it', was retracted by an "
                "adversarial re-run and is replaced by this entry.)"
            ),
            (
                "[API] linear.preconditioning.method takes a SHORT "
                "enumeration and the names people reach for are not in "
                "it. Signal: passing 'ilu' (or 'amg', or any other "
                "plausible name) raises the same RuntimeError shape "
                "as the linear.method case: the exception name "
                "'ParameterInvalid', the getEnumeration frame in "
                "dune/fem/io/parameter/reader.hh:300, the offending "
                "key, and a 'Valid values are:' line — reported as "
                "none, sor, ssor, gauss-seidel, jacobi. It fires at "
                "SCHEME CONSTRUCTION, before any "
                "solve, and names the parameter in a spelling you "
                "never wrote. The value list is built by the CALLER "
                "at reader.hh:294-296, not stored anywhere, so it is "
                "whatever the chosen solver family passes in and no "
                "part of that line can be grepped for. (Executed "
                "2026-08-03 on dune-fem 2.12.0.2, hit twice from "
                "different scripts.) The accepted list depends on the "
                "solver family, not on the space (re-measured 2026-09-30 "
                "on a scalar P1 galerkin scheme): the default "
                "solver='cg' and solver=('fem', 'cg') accept only none, "
                "sor, ssor, gauss-seidel, jacobi and reject ilu and amg "
                "at scheme construction; solver=('istl', 'cg') "
                "accepts ilu and rejects amg only at the first solve, "
                "listing none, ssor, sor, ilu, gauss-seidel, jacobi, "
                "amg-ilu, amg-jacobi, ildl; solver=('petsc', 'cg') "
                "accepts ilu and rejects amg at the first solve, listing "
                "none, oas, gauss-seidel, sor, ssor, jacobi, ilu, icc, "
                "kspoptions, hypre, ml, lu, pcgamg. A bare 'amg' is "
                "valid in none of them."
            ),
            (
                "[Numerical] Equal-order velocity/pressure violates "
                "LBB. Signal: composite(lagrange(dimRange=2, order=1), "
                "lagrange(order=1)) assembles and solves, and the "
                "pressure comes back with a checkerboard oscillation "
                "whose amplitude does NOT decrease under refinement "
                "while the velocity still looks plausible. Use "
                "velocity order = pressure order + 1, or add a "
                "stabilisation term. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Dirichlet velocity on the ENTIRE boundary "
                "leaves the pressure undetermined. Signal: the direct "
                "solver either reports a singular matrix or returns a "
                "pressure shifted by an arbitrary constant that moves "
                "between runs, while the velocity looks right. Leave "
                "one boundary with the natural stress-free condition "
                "(as the template does) or constrain one pressure dof. "
                "(Audit 2026-08-03.)"
            ),
            (
                "[API] The pressure entry of the DirichletBC value "
                "list must be None, not 0. Signal: "
                "DirichletBC(W, [ux, uy, 0], indicator) also pins the "
                "pressure to zero on that boundary and over-constrains "
                "the system; a boundary layer appears at the "
                "constrained edge that no refinement removes. "
                "dune.ufl.DirichletBC converts a None entry to 0 in "
                "its ufl_value but keeps the None in .value, which is "
                "what builds the component mask. (Source read "
                "dune/ufl/__init__.py::DirichletBC 2026-08-03; the "
                "None form is what the executed template uses.)"
            ),
        ],
    },
}

GENERATORS = {
    "stokes_2d": _stokes_2d_dune,
}
