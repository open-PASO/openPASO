"""scikit-fem wave-equation generators and knowledge.

Covers the 2D scalar wave equation
    u_tt - c^2 Δu = 0    on Ω = [0,1]^2
    u = 0                on ∂Ω
    u(x,0) = u0(x),  u_t(x,0) = v0(x)

with explicit central-difference time-stepping and row-sum-lumped mass
so each step is a single sparse-matrix-vector product (no linear solve).

Modelled after scikit-fem upstream example ex44 (1D wave equation,
Crank-Nicolson on a first-order system) — the backend previously had
**no** wave-equation generator at all, leaving a clear coverage gap
relative to upstream.
"""


def _wave_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate
    values for your specific problem.

    2D scalar wave equation, explicit central-difference time
    integration with row-sum-lumped mass. Output: max amplitude
    over the simulation + a results_summary.json with the
    central-node history sampled at t_end."""
    nx = params.get("nx", 24)
    c = params.get("c", 1.0)
    T_end = params.get("T_end", 0.4)
    # CFL: dt < h / (c * sqrt(2)) for 2D Q1. Pick 0.5 of that.
    safety = params.get("cfl_safety", 0.5)
    return f'''\
"""2D scalar wave equation: u_tt - c^2 Δu = 0 — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace, mass
import numpy as np
import json

c     = {c}
T_end = {T_end}
nx    = {nx}

_tol = 1e-10
m = (MeshQuad.init_tensor(np.linspace(0, 1, nx + 1),
                          np.linspace(0, 1, nx + 1))
     .with_boundaries({{
         "bnd": lambda x: (x[0] < _tol) | (x[0] > 1.0 - _tol)
                          | (x[1] < _tol) | (x[1] > 1.0 - _tol),
     }}))
e = ElementQuad1()
ib = Basis(m, e)

K = laplace.assemble(ib)
M = mass.assemble(ib)

# Row-sum (HRZ-style) lumped mass — diagonal vector. Lumping converts
# the explicit update into one diag-solve per step. The trade-off is a
# slightly-overdamped dispersion; for the manufactured BC-zero problem
# below the time-error stays bounded by the leading O(dt^2) term.
M_lumped = np.asarray(M.sum(axis=1)).ravel()

bnd_dofs = ib.get_dofs("bnd").flatten()
interior = np.setdiff1d(np.arange(ib.N), bnd_dofs)

# CFL: dt < h_min / (c * sqrt(2)) for 2D Q1.
h_min = 1.0 / nx
dt = {safety} * h_min / (c * np.sqrt(2.0))
n_steps = int(np.ceil(T_end / dt))
dt = T_end / n_steps  # adjust to land exactly at T_end

# Initial condition: lowest standing-wave mode on the unit square.
# u(x,y,0) = sin(pi x) sin(pi y); u_t(x,y,0) = 0.
x_coord = m.p[0, :]
y_coord = m.p[1, :]
u_old = np.sin(np.pi * x_coord) * np.sin(np.pi * y_coord)
u_old[bnd_dofs] = 0.0
# u_t(0) = 0 ⇒ u^{{-1}} = u^0 - 0.5 dt^2 M^{{-1}} (-c^2 K u^0)
#               = u^0 - 0.5 dt^2 M^{{-1}} c^2 K u^0
rhs0 = c * c * (K @ u_old)
acc0 = -rhs0 / M_lumped
acc0[bnd_dofs] = 0.0
u_prev = u_old - 0.5 * dt * dt * acc0
u_prev[bnd_dofs] = 0.0

# Track central-node amplitude for sanity (mid-point of the square).
center_dof = int(np.argmin((x_coord - 0.5) ** 2 + (y_coord - 0.5) ** 2))
center_history = [float(u_old[center_dof])]

amp_max = float(np.abs(u_old).max())
for step in range(n_steps):
    # u^{{n+1}} = 2 u^n - u^{{n-1}} - dt^2 M^{{-1}} c^2 K u^n
    rhs = c * c * (K @ u_old)
    u_new = 2.0 * u_old - u_prev - (dt * dt) * (rhs / M_lumped)
    u_new[bnd_dofs] = 0.0
    u_prev = u_old
    u_old = u_new
    a = float(np.abs(u_new).max())
    if a > amp_max:
        amp_max = a
    center_history.append(float(u_new[center_dof]))

# Analytical reference for the lowest mode:
# u(x,y,t) = cos(c * pi * sqrt(2) * t) * sin(pi x) sin(pi y)
# Initial amplitude at center = 1.0; FE solution should remain bounded
# by |amp_max| ~ 1.0 ± O(dt^2) over the simulation.
print(f"steps={{n_steps}} dt={{dt:.4e}} max|u|={{amp_max:.6f}} "
      f"u_center(T)={{center_history[-1]:.6f}}")

import meshio
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
mio = meshio.Mesh(points, [("quad", m.t.T)], point_data={{"u": u_old}})
mio.write("result.vtu")

summary = {{
    "max_amplitude": amp_max,
    "u_center_T_end": center_history[-1],
    "n_steps": n_steps,
    "dt": dt,
    "n_dofs": int(ib.N),
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


GENERATORS: dict = {
    "wave_2d": _wave_2d,
}


KNOWLEDGE: dict = {
    "wave": {
        "description": (
            "Scalar wave equation u_tt - c^2 Δu = 0 with "
            "homogeneous Dirichlet BCs. Explicit central-"
            "difference time integration + row-sum-lumped mass "
            "(no linear solve per step). Matches scikit-fem "
            "upstream ex44 in physics; the lumped-"
            "explicit variant here is the cheapest runnable form."
        ),
        "weak_form": (
            "M u_tt + c^2 K u = 0,  u^{n+1} = 2u^n - u^{n-1} "
            "- dt^2 M_L^{-1} c^2 K u^n"
        ),
        "elements": ["ElementQuad1 (P1 quad)"],
        "variants": ["2d"],
        "pitfalls": [
            "[Numerical] CFL: for explicit central-difference "
            "with ROW-SUM-LUMPED mass on a uniform Q1 tensor "
            "grid, the exact stability limit is "
            "dt_crit = 2 / sqrt(c^2 * lambda_max(M_L^-1 K)) "
            "= h / c, because lambda_max(M_L^-1 K) = 4/h^2 "
            "exactly. In units of the rule of thumb that is "
            "sqrt(2) * h/(c*sqrt(2)) — so "
            "dt < h_min/(c*sqrt(2)) is SAFE but is NOT the "
            "boundary; the prior 'stable only for' wording "
            "overstated it by a factor sqrt(2). Signal: measured "
            "on MeshQuad.init_tensor with nx = 10 and nx = 20, "
            "c = 1, integrating to T = 20 — dt/dt_crit = "
            "0.636 / 0.707 / 0.849 / 0.990 all stay bounded "
            "(max|u| <= 1.0), while dt/dt_crit = 1.061 / 1.131 / "
            "1.202 / 1.273 / 1.414 blow up to 1e+31 ... 1e+198. "
            "The transition is sharp between "
            "1.40*h/(c*sqrt(2)) (stable) and "
            "1.50*h/(c*sqrt(2)) (divergent). Note a short run "
            "hides this: at T = 0.5 even dt = 1.5*h/(c*sqrt(2)) "
            "still looks finite after 9 steps — integrate long "
            "enough before declaring stability. Downstream "
            "np.isfinite(u).all() becomes False and spsolve "
            "raises ValueError 'array must not contain infs or "
            "NaNs'. (Verified empirically 2026-08-03 on skfem "
            "12.0.1 — bound sharpened.)",
            "[API] scikit-fem >= 12 expects "
            "MeshQuad.init_tensor for tensor-product Q1 grids. "
            "The legacy `MeshQuad((nx,ny))` call does NOT raise: "
            "Mesh is a dataclass whose first field is doflocs, so "
            "the tuple is read as COORDINATES and you get back a "
            "MeshQuad1 carrying a 1-D point array and a single "
            "bogus element. Signal: nothing is raised, warned or "
            "printed at the construction line. The failure "
            "surfaces LATER, at Basis(...), which first logs "
            "'Unable to calculate global DOF locations.' to "
            "stderr through the logging module — not a Python "
            "warning, so catch_warnings records an empty list and "
            "simplefilter('error') cannot turn it into an "
            "exception — and then raises "
            "IndexError 'too many indices for array: array is "
            "1-dimensional, but 2 were indexed' — a message that "
            "never mentions MeshQuad, so it reads as a basis bug. "
            "The two-positional-argument spelling MeshQuad(nx, ny) "
            "raises IndexError 'tuple index out of range', also "
            "not a TypeError. Guard by asserting the element and "
            "vertex counts on the object you just built, before "
            "constructing a Basis from it. (Verified 2026-08-06 on "
            "skfem 12.0.1 — the previously quoted TypeError and "
            "AttributeError are emitted by neither spelling.)",
            "[Numerical] HRZ-style lumping via "
            "`scipy.sparse.csr_matrix.sum(axis=1)` is the "
            "simplest mass-lumping and it does introduce a "
            "dispersion error, but it does NOT cost you an order: "
            "on a uniform Q1 grid the lumped and consistent mass "
            "matrices give the fundamental frequency at the SAME "
            "convergence order in h. What lumping changes is the "
            "CONSTANT and the SIGN — the lumped error is a fixed, "
            "mesh-independent multiple of the consistent one, and "
            "the consistent Q1 mass OVERESTIMATES the frequency "
            "while lumping UNDERESTIMATES it. For high-frequency "
            "content prefer consistent mass + "
            "`scipy.sparse.linalg.spsolve` per step. "
            "Signal: do not watch for a stalled refinement rate — "
            "there is none, and a rate-based gate goes green on "
            "both. Take the fundamental frequency straight from "
            "the generalised eigenproblem K phi = lambda M phi "
            "(no time stepping, so no scheme error enters) for "
            "each mass matrix and compare against the analytic "
            "frequency: the SIGN of the deviation separates them, "
            "a magnitude-only check does not. The amplitude drift "
            "at T_end is mesh-dependent, so a fixed percentage "
            "threshold on it is not a property of the scheme. "
            "(Verified 2026-08-06 on skfem 12.0.1 — the previously "
            "claimed O(h^1.5) stall does not occur.)",
            "[API] `mass.assemble(basis)` returns a "
            "scipy.sparse.csr_matrix; summing along axis=1 "
            "produces a numpy.matrix of shape (N, 1) under both "
            "NumPy 1.x and 2.x (this is scipy's sparse-matrix "
            "API; only the csr_array classes return an ndarray). "
            "Use np.asarray(...).ravel() to coerce to 1-D. "
            "Signal: the mix-up is SILENT — no TypeError is "
            "raised and the warning list is empty, so neither "
            "try/except nor catch_warnings sees it. An (N,) "
            "ndarray divided by the (N, 1) matrix broadcasts and "
            "returns an (N, N) matrix, and using THAT object as "
            "the inverse lumped mass still yields a length-N "
            "acceleration vector, so a length or shape check on "
            "the result passes while the values are wrong. Guard "
            "on the lumped-mass array itself before you use it: "
            "require M_lumped.ndim to be 1, require it to be an "
            "np.ndarray that is not an np.matrix, and check that "
            "its "
            "entries sum to the measure of the domain. (Verified "
            "2026-08-06 on skfem 12.0.1 / numpy 1.26.4 — both "
            "previously quoted signals, the TypeError and the "
            "matrix-subclass DeprecationWarning, are absent.)",
            "[Physics] Initial condition with u_t(0) = 0 "
            "requires a special first time step, and the SIGN "
            "matters: with a = M^{-1} (-c^2 K u^0) the initial "
            "acceleration, the pre-step is "
            "u^{-1} = u^0 + 0.5 dt^2 a — PLUS, because u(-dt) is "
            "the Taylor expansion of u about t=0 with zero "
            "velocity, which is symmetric in dt. Writing the "
            "MINUS sign is worse than omitting the pre-step "
            "entirely: it drops the scheme from second to FIRST "
            "order in dt and its error exceeds that of the plain "
            "u^{-1} = u^0 start. Skipping the pre-step introduces "
            "a spurious initial velocity that pollutes the "
            "long-time solution; getting its sign backwards "
            "introduces a larger one. Build K with "
            "`laplace.assemble(basis)`, M with "
            "`mass.assemble(basis)`, and take one explicit "
            "pre-step before the main loop. "
            "Signal: do NOT look for a drifting centre amplitude "
            "— it is not there. The correct start, the "
            "sign-flipped one and doing nothing all oscillate, "
            "all change sign the same number of times over the "
            "run, and none shows a monotone growing peak, so "
            "watching u_center or np.abs(u).max() cannot separate "
            "them and reads clean on all three. The diagnostic "
            "that works is the ORDER of the error in dt: halve dt "
            "several times and fit the error against a reference "
            "built from the SAME K and M — the exact semi-discrete "
            "solution assembled spectrally from an eigen-"
            "decomposition, not the analytic PDE solution, whose "
            "spatial error would saturate the study. The correct "
            "pre-step gives order 2; the sign flip and the missing "
            "pre-step both give order 1. (Verified 2026-08-06 on "
            "skfem 12.0.1 — the sign in the prior text was wrong "
            "and its stated drift signal is not produced.)",
            "[Numerical] Dirichlet BCs must be re-applied "
            "every step (u_new[bnd_dofs] = 0). The explicit "
            "update propagates non-zero values into boundary "
            "DOFs via the off-diagonal K-coupling; without the "
            "re-application the simulation is no longer a "
            "homogeneous-BC problem — and this is not a slow "
            "leak. "
            "Signal: |u| at the boundary DOFs does grow away "
            "from zero, but immediately — the very FIRST "
            "explicit wave update writes "
            "-dt^2 c^2 (K u) / M_L into every boundary DOF, so "
            "it is already non-negligible after "
            "one step and reaches O(1) within a couple more; a "
            "threshold of 1e-6 reached within a few hundred "
            "steps is "
            "far too generous and would be crossed immediately "
            "on a correct-looking run. The failure is otherwise "
            "SILENT — empty warning list, finite solution, still "
            "oscillating — and the damage is not confined to the "
            "boundary: the interior field is wrong by an O(1) "
            "relative amount too. With the re-application the "
            "boundary DOFs stay EXACTLY zero, so the check is "
            "trivial to write: assert "
            "`np.all(u[bnd_dofs] == 0.0)` after every step, not "
            "a tolerance. (Verified 2026-08-06 on skfem 12.0.1 — "
            "mechanism confirmed, the stated threshold and "
            "timescale corrected.)",
            "[Output] VTK output via `meshio.Mesh(...)`: pad 2D "
            "points to three columns with "
            "`np.column_stack([m.p.T, np.zeros(m.p.shape[1])])` "
            "as a matter of hygiene, but understand that meshio "
            "does NOT require it — an (N, 2) point array "
            "constructs and writes. The cells argument is "
            "`[('quad', m.t.T)]` for MeshQuad and "
            "`[('triangle', m.t.T)]` for MeshTri. "
            "Signal: BOTH halves of the old warning point the "
            "wrong way, so guard differently. (a) Unpadded 2D "
            "points raise NOTHING. The file is written and reads "
            "back with a three-column point array whose third "
            "column meshio supplied as zeros; the only notice is "
            "plain text on the process STDERR (meshio prints it "
            "through rich Console(stderr=True)), 'Warning: VTU "
            "requires 3D points, but 2D points given.', which is "
            "NOT a Python warning — catch_warnings records an "
            "empty list and simplefilter('error') cannot promote "
            "it. contextlib.redirect_stdout does not capture it; "
            "contextlib.redirect_stderr does. (b) A wrong "
            "cell-type tag is USUALLY loud, not silent: tagging "
            "quad connectivity as 'triangle', 'quad8' or 'line' "
            "raises meshio WriteError naming the expected "
            "nodes-per-cell. The genuinely silent mis-tag is one "
            "with the SAME node count — 4-node quads tagged "
            "'tetra' pass meshio's only check, write without a "
            "murmur, and read back as tetrahedra on a mesh whose "
            "z-coordinates are all zero. The check that works is "
            "to read the file back and compare the cell type and "
            "the point-array shape against what you meant to "
            "write. (Verified 2026-08-06 on skfem 12.0.1 / meshio "
            "5.3.5 — the WriteError-on-2D-points claim and the "
            "silently-malformed-tag claim are both corrected.)",
        ],
        "references": [
            "scikit-fem example: ex44 (1-D wave equation as a "
            "first-order system)",
            "Hughes, T.J.R. The Finite Element Method (1987), "
            "Ch. 9: hyperbolic problems and CFL conditions",
        ],
    },
}
