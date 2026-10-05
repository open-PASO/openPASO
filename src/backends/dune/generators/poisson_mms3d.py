"""3D variable-coefficient Poisson MMS convergence family for DUNE-fem.

Extension of the DUNE-fem Poisson family (see poisson.py, which has a
single easy 2D constant-force instance): unit-cube-style domain
[0, L]^3, VARIABLE diffusion coefficient

    kappa(x) = k0 + (kx*x + ky*y + kz*z) / L        (affine, > 0 guarded)

and manufactured solution (MMS)

    u*(x) = amp * sin(a*pi*x/L) * sin(b*pi*y/L) * sin(c*pi*z/L)
            + d * (x/L)*(y/L)*(z/L)

with -div(kappa grad u) = f, exact Dirichlet data u = u* on the whole
boundary, Lagrange elements of parametrised order on a structured hex
grid, and a uniform-refinement loop n = n0 * 2^level. Each level prints
a machine-readable error line

    level <l> n <n> dofs <d> L2 <e_l2> H1 <e_h1>

followed by observed EOC lines. Expected orders for Lagrange order k:
L2 -> k+1, H1-seminorm -> k. The template has been executed on this
install and reaches those orders; the observed values are deliberately
NOT recorded here or in KNOWLEDGE — an observed convergence order is
the ANSWER to a study an agent may be asked to run, so it belongs in
the Tier-2 gate that re-measures it, not in agent-reachable knowledge.

DESIGN PRINCIPLE (project rule "no anchoring"): every problem dimension
is a parameter with placeholder defaults — domain size, MMS frequencies
and amplitudes, the diffusion-coefficient gradient, Lagrange order,
coarsest resolution and level count. Nothing is hard-coded to a
specific evaluation instance.

WHY the source term needs no hand derivation here: the emitted script
builds u* and kappa as UFL expressions and sets f = -div(kappa*grad(u*))
symbolically — UFL differentiates exactly, so there is no closed-form
transcription that could be wrong (contrast the FEBio family, where f
had to be derived offline with sympy and emitted as strings). What CAN
go wrong instead is the literal emission of the parameters into the
script — the gen-only tests therefore sympy-cross-check the EMITTED
u*/kappa expression text against reference Python evaluations at
random points.
"""

from __future__ import annotations

import math

# Placeholder defaults — NOT tuned to any evaluation instance.
_DEFAULTS: dict = {
    "order": 1,      # Lagrange order k
    "levels": 3,     # refinement levels (n doubles per level)
    "n0": 4,         # coarsest cells per edge
    "L": 1.0,        # cube edge length: domain [0, L]^3
    "a": 1.0,        # frequency along x (a*pi/L)
    "b": 1.0,        # frequency along y
    "c": 1.0,        # frequency along z
    "d": 1.0,        # trilinear term coefficient
    "amp": 1.0,      # amplitude of the sin product
    "k0": 1.0,       # diffusion coefficient offset
    "kx": 0.5,       # diffusion gradient along x (per L)
    "ky": 0.25,      # diffusion gradient along y
    "kz": 0.125,     # diffusion gradient along z
}


def _num(v) -> str:
    """Format a float literal with full round-trip precision."""
    return format(float(v), ".17g")


def kappa_min(params: dict) -> float:
    """Exact minimum of the affine kappa over [0, L]^3.

    kappa = k0 + (kx*x + ky*y + kz*z)/L is affine, so its minimum over
    the cube sits at a corner: k0 + sum of min(0, ki) (each x_i/L is
    0 or 1 at corners).
    """
    p = {**_DEFAULTS, **(params or {})}
    return (float(p["k0"]) + min(0.0, float(p["kx"]))
            + min(0.0, float(p["ky"])) + min(0.0, float(p["kz"])))


def manufactured_solution_at(params: dict, x: float, y: float,
                             z: float) -> float:
    """Reference evaluation of u* at a point (used by the tests)."""
    p = {**_DEFAULTS, **(params or {})}
    L = float(p["L"])
    return (float(p["amp"])
            * math.sin(float(p["a"]) * math.pi * x / L)
            * math.sin(float(p["b"]) * math.pi * y / L)
            * math.sin(float(p["c"]) * math.pi * z / L)
            + float(p["d"]) * (x / L) * (y / L) * (z / L))


def diffusion_at(params: dict, x: float, y: float, z: float) -> float:
    """Reference evaluation of kappa at a point (used by the tests)."""
    p = {**_DEFAULTS, **(params or {})}
    L = float(p["L"])
    return (float(p["k0"])
            + (float(p["kx"]) * x + float(p["ky"]) * y
               + float(p["kz"]) * z) / L)


def validate_parameters(params: dict) -> list[str]:
    """Return a list of human-readable problems (empty list = OK).

    Checks the merged (defaults + user) parameter set, so a partial
    params dict is fine.
    """
    problems: list[str] = []
    merged = {**_DEFAULTS, **(params or {})}

    def _is_int(v) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    def _is_num(v) -> bool:
        return (isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(v))

    order = merged["order"]
    if not _is_int(order):
        problems.append(f"order must be an integer, got {order!r}")
    elif not 1 <= order <= 4:
        problems.append(f"order must be in [1, 4], got {order}")

    levels = merged["levels"]
    if not _is_int(levels):
        problems.append(f"levels must be an integer, got {levels!r}")
    elif not 1 <= levels <= 6:
        problems.append(f"levels must be in [1, 6], got {levels}")

    n0 = merged["n0"]
    if not _is_int(n0):
        problems.append(f"n0 must be an integer, got {n0!r}")
    elif not 2 <= n0 <= 64:
        problems.append(f"n0 must be in [2, 64], got {n0}")

    # Cost guard: the finest level assembles a 3D Lagrange space with
    # (n_max*order + 1)^3 dofs — cap n_max*order to keep the serial
    # CG solve tractable.
    if _is_int(order) and _is_int(levels) and _is_int(n0):
        n_max = n0 * 2 ** (levels - 1)
        if n_max * order > 96:
            problems.append(
                "finest level too large: n0 * 2^(levels-1) * order must "
                f"be <= 96 (dofs = (n*order+1)^3), got {n_max * order}")

    L = merged["L"]
    if not _is_num(L):
        problems.append(f"L must be a finite number, got {L!r}")
    elif L <= 0:
        problems.append(f"L must be > 0, got {L}")

    for name in ("a", "b", "c", "d", "amp", "k0", "kx", "ky", "kz"):
        v = merged[name]
        if not _is_num(v):
            problems.append(
                f"coefficient {name!r} must be a finite number, got {v!r}")

    if (_is_num(merged["amp"]) and _is_num(merged["d"])
            and merged["amp"] == 0 and merged["d"] == 0):
        problems.append(
            "amp and d are both 0 — the manufactured solution is "
            "identically zero, so the convergence study is meaningless")

    if all(_is_num(merged[k]) for k in ("k0", "kx", "ky", "kz")):
        kmin = kappa_min(merged)
        if kmin <= 0:
            problems.append(
                "diffusion coefficient must stay positive on the cube: "
                "k0 + min(0,kx) + min(0,ky) + min(0,kz) = "
                f"{kmin:g} <= 0 — the operator loses ellipticity and "
                "the CG solve is not well-posed")

    return problems


def _poisson_3d_varcoeff(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values
    for your specific problem.

    3D variable-coefficient Poisson MMS convergence study — DUNE-fem/UFL.
    """
    problems = validate_parameters(params or {})
    if problems:
        raise ValueError(
            "Invalid poisson_3d_varcoeff parameters: " + "; ".join(problems))
    p = {**_DEFAULTS, **(params or {})}
    order = int(p["order"])
    levels = int(p["levels"])
    n0 = int(p["n0"])
    L, a, b, c = (_num(p["L"]), _num(p["a"]), _num(p["b"]), _num(p["c"]))
    d, amp = _num(p["d"]), _num(p["amp"])
    k0, kx, ky, kz = (_num(p["k0"]), _num(p["kx"]), _num(p["ky"]),
                      _num(p["kz"]))
    return f'''\
"""3D variable-coefficient Poisson MMS convergence study — DUNE-fem (UFL).

-div(kappa grad u) = f on [0, {L}]^3, kappa affine, u = u* (exact) on the
whole boundary, Lagrange order {order}, levels n = {n0} * 2^l for
l = 0..{levels - 1}. Expected EOC: L2 -> {order + 1}, H1 -> {order}.
"""
import json
import math
import warnings

from dune.grid import structuredGrid
from dune.fem import integrate
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import (TrialFunction, TestFunction, SpatialCoordinate,
                 dot, grad, div, dx, sin, pi)

warnings.filterwarnings("ignore", category=DeprecationWarning)

order = {order}
levels = {levels}
n0 = {n0}
L = {L}

results = []
errs_l2 = []
errs_h1 = []
for level in range(levels):
    n = n0 * 2**level
    gridView = structuredGrid([0, 0, 0], [L, L, L], [n, n, n])
    space = lagrange(gridView, order=order)
    u = TrialFunction(space)
    v = TestFunction(space)
    x = SpatialCoordinate(space)

    u_exact = ({amp} * sin({a}*pi*x[0]/L) * sin({b}*pi*x[1]/L)
               * sin({c}*pi*x[2]/L)
               + {d} * (x[0]/L) * (x[1]/L) * (x[2]/L))
    kappa = {k0} + ({kx}*x[0] + {ky}*x[1] + {kz}*x[2]) / L
    # UFL differentiates u_exact symbolically — f is exact by construction.
    f = -div(kappa * grad(u_exact))

    A = kappa * dot(grad(u), grad(v)) * dx
    rhs = f * v * dx
    dbc = DirichletBC(space, u_exact)
    scheme = galerkin([A == rhs, dbc], solver="cg",
                      parameters={{"nonlinear.tolerance": 1e-11,
                                  "linear.tolerance": 1e-13,
                                  "linear.maxiterations": 20000}})
    uh = space.interpolate(0, name="u")
    scheme.solve(target=uh)

    # Error integration: quadrature well above 2*order so the measured
    # EOC reflects the FE error, not the quadrature of the trig exact
    # solution.
    quad = 2 * order + 4
    e_l2 = math.sqrt(abs(integrate((uh - u_exact)**2, gridView=gridView,
                                   order=quad)))
    e_h1 = math.sqrt(abs(integrate(dot(grad(uh - u_exact),
                                       grad(uh - u_exact)),
                                   gridView=gridView, order=quad)))
    errs_l2.append(e_l2)
    errs_h1.append(e_h1)
    results.append({{"level": level, "n": n, "dofs": space.size,
                    "l2_error": e_l2, "h1_error": e_h1}})
    print(f"level {{level}} n {{n}} dofs {{space.size}} "
          f"L2 {{e_l2:.10e}} H1 {{e_h1:.10e}}", flush=True)
    if level == levels - 1:
        gridView.writeVTK("poisson3d_varcoeff", pointdata={{"u": uh}})

eoc_l2 = [math.log(errs_l2[i-1]/errs_l2[i]) / math.log(2.0)
          for i in range(1, levels)]
eoc_h1 = [math.log(errs_h1[i-1]/errs_h1[i]) / math.log(2.0)
          for i in range(1, levels)]
for i, (o2, o1) in enumerate(zip(eoc_l2, eoc_h1), start=1):
    print(f"EOC level {{i}}: L2 {{o2:.3f}} H1 {{o1:.3f}}")

summary = {{
    "family": "poisson_3d_varcoeff",
    "order": order,
    "expected_eoc": {{"l2": order + 1, "h1": order}},
    "levels": results,
    "eoc_l2": eoc_l2,
    "eoc_h1": eoc_h1,
}}
with open("results_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print("DUNE-fem 3D variable-coefficient Poisson MMS study complete.")
print("DUNE_TEMPLATE_COMPLETE")
'''


KNOWLEDGE = {
    "poisson_mms": {
        "description": (
            "3D variable-coefficient Poisson manufactured-solution (MMS) "
            "convergence family on DUNE-fem — [0,L]^3 structured hex "
            "grid, affine diffusion kappa(x), trig+trilinear u*, exact "
            "Dirichlet data, uniform refinement with machine-readable "
            "per-level L2/H1 error lines"),
        "input_format": "Python script (dune.fem + UFL)",
        # The EOC tables that used to close this string were our own
        # development measurements, including the exact parameter draw they
        # came from. Both halves are contamination: the orders are the answer
        # to a convergence study, and the named draw is a specific problem
        # instance. Removed 2026-08-06 by the contamination merge gate.
        "expected_order": (
            "Theory for a conforming Lagrange space of order k on a "
            "sufficiently smooth solution: L2 error O(h^(k+1)) and "
            "H1-seminorm error O(h^k), i.e. L2 order k+1 and H1 order k. "
            "These are the ASYMPTOTIC rates — the coarsest levels of a "
            "sweep are commonly pre-asymptotic and show a lower EOC. The "
            "EOCs of any particular run are an output of that run; read "
            "them from the emitted 'EOC level ...' lines."),
        "mms_setup": {
            "source_term": (
                "f = -div(kappa*grad(u_exact)) built symbolically in UFL "
                "from the emitted u*/kappa expressions — no offline "
                "closed-form derivation to transcribe (UFL applies exact "
                "symbolic differentiation during form compilation)."),
            "dirichlet": (
                "dune.ufl.DirichletBC(space, u_exact) imposes the exact "
                "trace on the WHOLE boundary — u_exact may be any UFL "
                "expression of SpatialCoordinate(space); no per-face "
                "bookkeeping needed."),
            "errors": (
                "dune.fem.integrate((uh-u_exact)**2, gridView=gv, "
                "order=2k+4) for L2; same with grad() for the "
                "H1-seminorm; sqrt(abs(.)) guards the tiny negative "
                "round-off that an exactly-zero error would produce."),
        },
        "pitfalls": [
            (
                "[API] dune.fem.function.integrate is DEPRECATED in "
                "dune-fem 2.10 — use dune.fem.integrate with the NEW "
                "argument order (expr, gridView, order), i.e. "
                "integrate(expr, gridView=gv, order=q). Signal: "
                "CALLING dune.fem.function.integrate(gridView, expr, "
                "order) emits "
                "'DeprecationWarning: dune.fem.function.integrate is "
                "deprecated use dune.fem.integrate instead. New "
                "signature is (expr, gridView, order)' — importing it "
                "is silent. It is shown when the call sits in the "
                "script being run (__main__) and hidden by Python's "
                "default warning filters when the call sits in an "
                "imported module — observed live on this install "
                "(2026-08-01; import versus call re-measured "
                "2026-09-30 on dune-fem 2.12.0.2). The old path still "
                "works, so this is a warning to act on, not a failure "
                "to debug."
            ),
            (
                "[API] galerkin solver parameter keys 'newton.*' are "
                "DEPRECATED in dune-fem 2.10: 'newton.tolerance' -> "
                "'nonlinear.tolerance', 'newton.linear.*' -> 'linear.*'. "
                "Signal: passing the old keys emits \"UserWarning: "
                "Warning: the parameter key 'newton.linear' is "
                "deprecated. Simply remove 'newton.' to avoid this "
                "warning!\" and \"Warning: the parameter key 'newton' "
                "is deprecated. Replace with 'nonlinear' to avoid this "
                "warning!\", with the quoted key names wrapped in ANSI "
                "colour codes, so grep for 'is deprecated. Simply "
                "remove' or 'is deprecated. Replace with' rather than "
                "for a whole line — observed on dune-fem 2.10, message "
                "text re-measured 2026-09-30 on dune-fem 2.12.0.2; the "
                "old keys still WORK (warning only), so runs do not "
                "fail, but the template emits the new keys."
            ),
            (
                "[Numerics] The error quadrature order must comfortably "
                "exceed 2*order or the measured 'FE error' is polluted "
                "by the quadrature error of the trig exact solution at "
                "fine levels. The template uses order=2k+4. Signal: an "
                "under-integrated error norm shows EOCs that DRIFT AWAY "
                "from k+1 as levels refine (error floor), while the "
                "solver itself reports clean convergence "
                "(theory-derived; not provoked live on this install)."
            ),
            (
                "[Numerics] kappa must stay positive on the whole cube "
                "or the operator loses ellipticity and CG (an SPD "
                "Krylov method) is not applicable. For the affine "
                "kappa the exact minimum is k0+min(0,kx)+min(0,ky)+"
                "min(0,kz) at a cube corner — validate_parameters "
                "rejects kmin <= 0 before any code is emitted. Signal: "
                "an indefinite diffusion field makes the linear solve "
                "stall or diverge — residual does not drop and the "
                "scheme.solve call runs into the iteration cap instead "
                "of terminating in a handful of Newton steps "
                "(theory-derived guard; rejected at validation, not "
                "provoked live)."
            ),
            (
                "[Performance] Each distinct (grid dimension, form) "
                "combination JIT-compiles its own C++ module; "
                "within one refinement loop the form is textually "
                "identical across levels so ONLY the first level pays "
                "the compile cost. Signal: level 0 of a cold run "
                "takes tens of seconds of wall time while later levels "
                "of the same run finish in seconds. Changing the "
                "Lagrange order (1-4 here) does NOT trigger a "
                "recompile: for order <= 6 lagrange() uses "
                "DynamicLagrangeDiscreteFunctionSpace, whose order is "
                "a runtime argument, and the UFL form is unchanged "
                "(measured 2026-09-30: this template at orders 1, 2, 3 "
                "and 4 loaded the same 13 generated modules and built "
                "none)."
            ),
            (
                "[API] structuredGrid in 3D creates HEXAHEDRA (YaspGrid "
                "cubes), so a Lagrange space of order k has "
                "(n*k+1)^3 scalar dofs — cost grows with the CUBE of "
                "n*k, which is why the family caps n0*2^(levels-1)*"
                "order. Signal: space.size printed per level equals "
                "(n*order+1)^3 exactly (order=1, n=8 -> 9^3 = 729; "
                "order=2, n=8 -> 17^3 = 4913); anything else means the "
                "grid or the space is not what you think it is."
            ),
        ],
    },
}

GENERATORS = {
    "poisson_mms_3d_varcoeff": _poisson_3d_varcoeff,
}
