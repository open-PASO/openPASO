"""DUNE-fem DG advection generator and knowledge.

HISTORY. Until 2026-08-03 this generator emitted a forward-Euler loop
that called scheme.solve(target=u_n) with u_n also appearing on the
right-hand side (aliasing), used solver='cg' on the wrong operator and
ran 500 steps; it was killed at the 1800 s timeout (rc 124) without
printing a result. It is replaced by an SSP-RK2 (Heun) loop over ONE
compiled scheme whose right-hand side reads a separate stage function,
executed against dune-fem 2.12.0.2 and checked against the exact
translation of the initial profile.
"""


def _dg_advection_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    DG advection with an upwind flux and SSP-RK2 time stepping.
    """
    nx = params.get("nx", 24)
    order = params.get("order", 1)
    T_end = params.get("T_end", 0.3)
    cfl = params.get("cfl", 0.2)
    b0 = params.get("b0", 1.0)
    b1 = params.get("b1", 0.5)
    return f'''\
"""DG advection  du/dt + b . grad(u) = 0  on [0,1]^2 — DUNE-fem.

Upwind flux on interior facets, inflow value 0, SSP-RK2 (Heun) in time.

Verification this script runs on itself, with no reference solution:
  * a constant velocity b translates the profile RIGIDLY, so the
    centroid of u must move by exactly b*T;
  * the scheme is conservative while the profile stays inside the
    domain, so int(u) must not change;
  * the upwind flux is dissipative, so the peak must NOT grow.
"""
from dune.grid import structuredGrid
from dune.fem.space import dglagrange
from dune.fem.scheme import galerkin
from dune.fem import integrate
from ufl import (TrialFunction, TestFunction, SpatialCoordinate, FacetNormal,
                 as_vector, dot, grad, exp, dx, dS, ds, conditional, gt)
import numpy as np
import json

nx = {nx}
order = {order}
b0, b1 = {b0}, {b1}          # advection velocity (constant)
T_end = {T_end}
cfl = {cfl}

gridView = structuredGrid([0, 0], [1, 1], [nx, nx])
space = dglagrange(gridView, order=order)      # DISCONTINUOUS Lagrange
x = SpatialCoordinate(space)
n = FacetNormal(space)
b = as_vector([b0, b1])

# initial profile — a smooth bump well inside the domain
u0 = exp(-((x[0] - 0.25) ** 2 + (x[1] - 0.25) ** 2) / (2 * 0.08 ** 2))

un = space.interpolate(u0, name="u")     # current solution
w = space.interpolate(u0, name="w")      # RK stage input (a COEFFICIENT)
uh = space.interpolate(0, name="uh")     # stage output

u = TrialFunction(space)
v = TestFunction(space)

# REQUIRED: dt from the DG CFL limit. dt <= cfl*h/((2k+1)*|b|) with
# cfl <= 1; a larger dt makes the run blow up to NaN within a few steps.
h = 1.0 / nx
dt = cfl * h / ((2 * order + 1) * np.hypot(b0, b1))
n_steps = int(round(T_end / dt))
dt = T_end / n_steps                     # hit T_end exactly
print(f"dt = {{dt:.6g}}, steps = {{n_steps}}")

bn = dot(b, n)
# upwind trace: take the value from the side the flow comes FROM.
# dS is INTERIOR facets (capital S); ds is the domain boundary.
up = conditional(gt(bn("+"), 0), w("+"), w("-"))
L_op = (dot(b, grad(v)) * w * dx
        - bn("+") * up * (v("+") - v("-")) * dS
        - conditional(gt(bn, 0), bn * w * v, 0) * ds)   # outflow only

# ONE scheme, compiled once: mass matrix on the left, stage residual on
# the right. Swapping the CONTENTS of w between stages needs no rebuild.
# The mass matrix is SPD and block-diagonal for DG, so cg is right here.
scheme = galerkin([u * v * dx == w * v * dx + dt * L_op], solver="cg",
                  parameters={{"linear.tolerance": 1e-12}})


def moments(f):
    """(mass, centroid_x, centroid_y) of a discrete function."""
    m = integrate(f, gridView=gridView, order=2 * order + 2)
    cx = integrate(f * x[0], gridView=gridView, order=2 * order + 2)
    cy = integrate(f * x[1], gridView=gridView, order=2 * order + 2)
    return float(m), float(cx / m), float(cy / m)


mass0, cx0, cy0 = moments(un)
peak0 = float(np.array(un.as_numpy).max())

# SSP-RK2 (Heun). Forward Euler alone is linearly UNSTABLE for DG
# advection: at this same dt its L2 norm grows monotonically (measured
# +0.027% per step) while SSP-RK2's decays.
for step in range(n_steps):
    w.as_numpy[:] = np.array(un.as_numpy)
    scheme.solve(target=uh)                       # stage 1
    stage1 = np.array(uh.as_numpy).copy()
    w.as_numpy[:] = stage1
    scheme.solve(target=uh)                       # stage 2
    un.as_numpy[:] = 0.5 * (np.array(un.as_numpy) + np.array(uh.as_numpy))

mass1, cx1, cy1 = moments(un)
peak1 = float(np.array(un.as_numpy).max())

print(f"mass {{mass0:.8f}} -> {{mass1:.8f}}  "
      f"relative change {{abs(mass1 - mass0) / mass0:.3e}}")
print(f"centroid ({{cx0:.5f}}, {{cy0:.5f}}) -> ({{cx1:.5f}}, {{cy1:.5f}})")
print(f"expected ({{cx0 + b0 * T_end:.5f}}, {{cy0 + b1 * T_end:.5f}})")
print(f"centroid error dx = {{cx1 - (cx0 + b0 * T_end):.3e}}, "
      f"dy = {{cy1 - (cy0 + b1 * T_end):.3e}}")
print(f"peak {{peak0:.6f}} -> {{peak1:.6f}} (must NOT grow), "
      f"min {{float(np.array(un.as_numpy).min()):.6f}}")

gridView.writeVTK("result", pointdata={{"concentration": un}})
summary = {{
    "dt": dt, "n_steps": n_steps, "T_end": T_end,
    "mass_rel_change": abs(mass1 - mass0) / mass0,
    "centroid_error_x": cx1 - (cx0 + b0 * T_end),
    "centroid_error_y": cy1 - (cy0 + b1 * T_end),
    "peak_initial": peak0, "peak_final": peak1,
    "n_dofs": int(space.size), "order": order,
}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("DG advection solve complete.")
print("DUNE_TEMPLATE_COMPLETE")
'''


KNOWLEDGE = {
    "dg_advection": {
        "description": (
            "Discontinuous Galerkin transport with an upwind numerical "
            "flux and explicit SSP Runge-Kutta time stepping, written "
            "with the ordinary galerkin scheme. dune-fem-dg is a "
            "SEPARATE package and is not importable from a plain "
            "dune-fem install."),

        "required_calls_in_order": [
            "space = dune.fem.space.dglagrange(gridView, order=k)",
            "n = ufl.FacetNormal(space); bn = ufl.dot(b, n)",
            "upwind = ufl.conditional(ufl.gt(bn('+'), 0), w('+'), w('-'))"
            "   <- w is the STAGE COEFFICIENT, not the trial function",
            "L = dot(b,grad(v))*w*dx - bn('+')*upwind*(v('+')-v('-'))*dS"
            " - conditional(gt(bn,0), bn*w*v, 0)*ds",
            "scheme = galerkin([u*v*dx == w*v*dx + dt*L], solver='cg')"
            "   <- mass matrix on the left; cg is correct because the "
            "DG mass matrix is SPD and block diagonal",
            "each stage: w.as_numpy[:] = <stage input dofs>; "
            "scheme.solve(target=uh)",
        ],
        "required_vs_optional": {
            "REQUIRED": [
                "dt <= cfl*h/((2*order+1)*|b|) with cfl <= 1",
                "dS (capital S) for the interior-facet flux — ds is "
                "the domain boundary and silently drops the coupling",
                "at least SSP-RK2 in time for anything but a short "
                "run — forward Euler is linearly unstable for DG "
                "advection (its L2 norm grows monotonically) even "
                "inside the CFL limit",
                "an upwind (not centred) numerical flux",
                "ONE scheme built OUTSIDE the time loop",
            ],
            "OPTIONAL": [
                "a separate output function uh for each stage solve. "
                "This scheme is LINEAR, and for a linear scheme "
                "dune-fem 2.12.0.2 reads the right-hand side once, "
                "before it overwrites the target, so solving each stage "
                "into the stage coefficient w itself gives the same "
                "answer (measured 2026-09-30: mass change, centroid "
                "error and peak identical to every printed digit). Keep "
                "the output separate for a nonlinear form or for "
                "Dirichlet data that change in time, where the solve "
                "re-reads or pre-sets its target",
                "the inflow term — leaving it out means u=0 flows in",
                "a limiter, needed only for discontinuous data",
                "dgonb / dglegendre instead of dglagrange for a modal "
                "basis at high order",
            ],
        },
        "verification_you_can_run": (
            "Advect a smooth bump with a CONSTANT velocity b for time "
            "T and check three things that need no reference solution: "
            "(1) the centroid, int(u*x)/int(u), must move by exactly "
            "b*T; (2) int(u) must be unchanged while the bump is "
            "interior; (3) the peak must not grow, because upwinding "
            "is dissipative. Executed 2026-08-03, 24x24 structuredGrid, "
            "P1 DG, b=(1,0.5), T=0.3, cfl=0.2, SSP-RK2: centroid error "
            "1.9e-09 and -4.2e-12, mass change 4.1e-09, peak 1.0432 -> "
            "1.0158. Forward Euler at the same dt conserves mass and "
            "transports just as accurately, but its L2 NORM grows "
            "monotonically (1.0047 after 20 steps, 1.0493 after 180) "
            "where SSP-RK2's decays (0.9984 -> 0.9880) — that "
            "monotone growth is the stability failure, and it is "
            "visible without any exact solution. Track "
            "sqrt(integrate(u*u)) every few steps; the peak alone is "
            "a noisy proxy because the profile crosses the DG nodal "
            "grid."),

        "pitfalls": [
            (
                "[Numerical] Forward Euler is LINEARLY UNSTABLE for DG "
                "advection even inside the CFL limit, and the "
                "detector is the L2 NORM, not the peak. Signal: with "
                "dt = 0.2*h/((2k+1)|b|) on a 24x24 P1 DG grid, "
                "||u||/||u_0|| under forward Euler rose monotonically "
                "1.004702, 1.009737, 1.014952, ... 1.049272 over 180 "
                "steps — clean exponential growth of about 0.027% per "
                "step — while SSP-RK2 (Heun) decayed monotonically "
                "0.998382, 0.996958, ... 0.987995. Two honest "
                "qualifications, both measured: transport ACCURACY is "
                "the same to five significant figures (centroid error "
                "3.419975e-04 for forward Euler vs 3.419994e-04 for "
                "SSP-RK2) and mass is conserved by both; and no "
                "blow-up was reachable in this configuration, because "
                "the outflow boundary drains energy faster than "
                "forward Euler injects it. So SSP-RK2 is cheap "
                "insurance that matters for long or closed-domain "
                "runs, not a prerequisite for getting an answer. "
                "SSP-RK2 is: u1 = u + dt*L(u); u2 = u1 + dt*L(u1); "
                "u_new = (u + u2)/2. (Executed 2026-08-03 on dune-fem "
                "2.12.0.2; the norm history and the accuracy "
                "equivalence come from adversarial re-execution the "
                "same day, which is also what showed the peak-based "
                "wording in an earlier revision to be a noisy proxy.)"
            ),
            (
                "[API] Solving into the function the right-hand side "
                "reads — scheme.solve(target=w) with w in the form — is "
                "harmless for a scheme that is LINEAR in the trial "
                "function, as this one is: dune-fem 2.12.0.2 evaluates "
                "the residual once from the current target, takes one "
                "Newton step and stops, so the result equals the "
                "separate-target solve (measured 2026-09-30: this "
                "template with both stages solved into w printed the "
                "same mass, centroid and peak to every digit). It is "
                "WRONG for a nonlinear form, because Newton "
                "re-evaluates the residual after each update and the "
                "right-hand side then reads the iterate instead of the "
                "old state, and for a strong DirichletBC whose data "
                "change in time, because the target's boundary dofs "
                "are set to the new data before the right-hand side is "
                "read. Signal: measured on a (1+u^2) implicit-Euler "
                "diffusion step with nonlinear.maxiterations=50, the "
                "aliased solve returned converged=False after 50 "
                "iterations with max|u| near 1e-4, while the "
                "separate-target solve converged in 4 iterations to "
                "max|u| 0.79; the time-dependent Dirichlet case reports "
                "nothing (measured: 6.8e-4 away from the "
                "separate-target answer after 5 implicit-Euler heat "
                "steps). Solve into a separate function and copy back "
                "to stay safe in both cases."
            ),
            (
                "[Performance] Rebuilding the scheme inside the time "
                "loop costs a full C++ compile per iteration. Signal: "
                "the run produces no output for tens of minutes and "
                "stderr fills with repeated DUNE-INFO: Compiling "
                "Integrands (new) lines. That line has no literal "
                "anywhere: dune/generator/cmakebuilder.py:378 builds "
                "the tail as an f-string and Python's logging module "
                "prepends the head from dune/common/__init__.py:49, "
                "'DUNE-%(levelname)s: %(message)s'. Build ONE scheme "
                "outside the loop "
                "whose right-hand side reads a coefficient function, "
                "then overwrite that coefficient's dofs each stage — "
                "measured 1.6 s for 121 SSP-RK2 steps (242 solves) "
                "once compiled, against roughly 480 s of one-off "
                "compilation. (Executed 2026-08-03 on dune-fem "
                "2.12.0.2.)"
            ),
            (
                "[API] dS is the interior-facet measure, ds the "
                "boundary measure, and UFL distinguishes them only by "
                "case. Signal: writing the jump term over ds assembles "
                "cleanly and silently drops all facet coupling — the "
                "DG operator becomes block diagonal and the profile "
                "does not move at all. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] A centred flux on pure advection is "
                "unconditionally unstable. Signal: replacing the "
                "conditional upwind trace by 0.5*(w('+') + w('-')) "
                "makes the amplitude grow exponentially at any dt; "
                "reducing dt slows the growth but never removes it. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] The CFL denominator grows with the "
                "polynomial degree: dt <= cfl*h/((2*order+1)*|b|). "
                "Signal: a dt that is stable at order=1 gives NaN "
                "within ~10 steps at order=3, because the admissible "
                "step is 7/3 times smaller. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] High-order NODAL DG conditions badly; "
                "modal bases do not. Signal: at order 5 a nodal "
                "dglagrange operator reaches cond ~1e10 while dgonb "
                "(orthonormal) stays ~1e4 on the same problem, and the "
                "Krylov iteration count follows. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Discontinuous data needs a limiter. "
                "Signal: high-order DG on a step profile overshoots by "
                "10-30% at the jump and the overshoot does NOT shrink "
                "under refinement (Gibbs); a minmod or WENO limiter "
                "clips it at the cost of one order of accuracy. "
                "(Audit 2026-06-02.)"
            ),
        ],
    },
}

GENERATORS = {
    "dg_advection_2d": _dg_advection_2d,
}
