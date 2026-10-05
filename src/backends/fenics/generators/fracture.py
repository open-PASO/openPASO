"""Phase-field fracture generator for FEniCSx/dolfinx.

Variants: 2d

Standard AT2 phase-field model with isotropic stiffness
degradation. Two coupled fields:

    u : displacement (vector)
    d : damage / phase-field (scalar in [0, 1])

Weak forms (alternate minimization staggered Newton):

    Displacement:
        ∫ (1-d)^2 σ(u) : ε(v) dx = ∫ f·v dx

    Damage:
        Gc/l0 * ∫ (d + l0^2 ∇d·∇w) dx
            = 2 (1-d) ψ⁺(u) * w * dx

where ψ⁺ is the positive part of the elastic energy
density (no degradation in compression — split via spectral
or volumetric-deviatoric decomposition). This template uses
the simplest no-split degradation for tractability.

For a single staggered Newton step the linearization
reduces to two separate LinearProblems, alternated until
convergence or a fixed number of staggered iterations.
"""


KNOWLEDGE = {
    # ─────────────────────────────────────────────────────────────────
    # _SERVING_STATUS (added 2026-08-03)
    # This dict is SHADOWED and is NOT what an agent receives.
    # fenics/backend.py:get_knowledge() returns
    # src/tools/deep_knowledge.py::_FENICS_KNOWLEDGE['fracture'] for this
    # physics and never falls through to here. Editing the pitfalls
    # below changes nothing an agent can see. The claims here were NOT
    # re-verified in the 2026-08-03 execution pass for exactly that
    # reason — treat them as unverified history, and make corrections
    # in deep_knowledge.py instead.
    # ─────────────────────────────────────────────────────────────────
    "description": (
        "Phase-field fracture (AT2) — diffuse crack via "
        "damage d in [0, 1]. No remeshing; crack path "
        "emerges from energy minimisation."),
    "weak_form_displacement": (
        "∫ (1-d)^2 σ(u) : ε(v) dx = ∫ f·v dx"),
    "weak_form_damage": (
        "Gc/l0 ∫ (d + l0^2 ∇d·∇w) dx "
        "= 2 (1-d) ψ⁺(u) w dx"),
    "function_space": (
        "u: vector Lagrange space (P1 or P2; dolfinx 0.10 "
        "has no VectorFunctionSpace); d: scalar "
        "FunctionSpace P1. Both via fem.functionspace with "
        "basix.ufl.element — u with shape=(gdim,), d scalar."),
    "solver": {
        "staggered": (
            "Alternate Newton on (u | d fixed) then "
            "(d | u fixed) until convergence."),
        "monolithic": (
            "Block Newton on (u, d) joint — faster but "
            "harder to converge; needs damped Newton."),
    },
    "pitfalls": [
        "[Numerical] The AT2 model is NON-CONVEX in (u, d) "
        "jointly: monolithic Newton on the joint system "
        "(via dolfinx.fem.petsc.NonlinearProblem with a "
        "MixedElement) diverges from many initial guesses. "
        "Signal: NewtonSolver.solve() returns iterations="
        "max_it without convergence, or oscillates between "
        "two locally-optimal (u, d) pairs with the residual "
        "norm bouncing.",
        "[Numerical] Damage irreversibility: d must "
        "monotonically increase (cracks don't heal). "
        "Without an enforcement scheme (active-set, "
        "history-field, or penalization), Newton can produce "
        "d_new < d_old in some elements. "
        "Signal: writing d_old.copy() then comparing "
        "d_new.x.array against d_old.x.array shows some "
        "entries strictly smaller — physically wrong.",
        "[API] Use basix.ufl.element('Lagrange', cell, k, "
        "shape=(gdim,)) for u and basix.ufl.element("
        "'Lagrange', cell, 1) for d. dolfinx 0.10+ removed "
        "ufl.VectorElement / ufl.FiniteElement. "
        "Signal: \"AttributeError: module 'ufl' has no attribute "
        "'VectorElement'\" — the quotes around 'ufl' and "
        "'VectorElement' are part of the message, and an earlier "
        "wording here dropped them, so a guard matching the text "
        "verbatim would have missed. ufl.FiniteElement gives the "
        "same message with 'FiniteElement'. Raised at ATTRIBUTE "
        "LOOKUP, i.e. as soon as the name is touched, not at "
        "fem.functionspace construction. (Verified by execution "
        "2026-08-07, dolfinx 0.10.0 / ufl 2025.2.1.)",
        "[Numerical] Length scale l0 ↔ mesh resolution: "
        "elements in the crack-band region need h ≤ l0/2 "
        "to resolve the diffuse damage profile. "
        "Signal: computed crack-band width measured from "
        "the damage-field d isosurface (d=0.99) is ~ 2*h "
        "(mesh-dependent) instead of ~ 4*l0 (model-set). "
        "Convergence rate: solver converges but "
        "post-process bandwidth doesn't match the AT2 "
        "analytic value pi*l0.",
    ],
}

VARIANTS = ["2d"]


def generate(variant: str, params: dict) -> str:
    generators = {
        "2d": _fracture_2d,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(
            f"Unknown variant: {variant!r}. "
            f"Available: {list(generators)}")
    return gen(params)


def _fracture_2d(params: dict) -> str:
    """FORMAT TEMPLATE — AT2 phase-field on a unit square
    with a pre-notched seed (initial damage at left edge),
    pulled in tension. Single staggered Newton step
    (linearized) — for production use, wrap in an outer
    fixed-point loop until ||d_new - d_old|| < tol."""
    nx = params.get("nx", 32)
    E = params.get("E", 1.0)
    nu = params.get("nu", 0.3)
    Gc = params.get("Gc", 1.0e-3)
    l0 = params.get("l0", 0.025)
    disp = params.get("applied_disp", 0.005)
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return f'''\
"""Phase-field fracture (AT2) — alternate Newton — FEniCSx/dolfinx"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import basix.ufl
import ufl
import numpy as np

domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {nx},
                                 mesh.CellType.triangle)
gdim = domain.geometry.dim
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

mu_val  = {mu}
lam_val = {lam}
Gc_val  = {Gc}
l0_val  = {l0}
disp_val = {disp}

# u : vector P1; d : scalar P1
V_u = fem.functionspace(
    domain,
    basix.ufl.element("Lagrange", domain.basix_cell(), 1,
                       shape=(gdim,)))
V_d = fem.functionspace(
    domain,
    basix.ufl.element("Lagrange", domain.basix_cell(), 1))

u_h = fem.Function(V_u, name="displacement")
d_h = fem.Function(V_d, name="damage")
d_h.x.array[:] = 0.0

# Initial notch: damage = 1 at left edge, mid-height
def initial_notch(x):
    return ((x[0] < 0.05) & (np.abs(x[1] - 0.5) < l0_val))
notch_dofs = fem.locate_dofs_geometrical(V_d, initial_notch)
d_h.x.array[notch_dofs] = 1.0

def eps(w):
    return ufl.sym(ufl.grad(w))

def sigma(w):
    return (2.0 * mu_val * eps(w)
            + lam_val * ufl.tr(eps(w)) * ufl.Identity(gdim))

def psi_plus(w):
    # No-split positive strain energy (simplest; production
    # would use spectral or volumetric-deviatoric split)
    return 0.5 * ufl.inner(sigma(w), eps(w))

# --- Displacement subproblem (d fixed) ---
u_tr = ufl.TrialFunction(V_u)
v_tr = ufl.TestFunction(V_u)
g_d = (1.0 - d_h) ** 2 + 1e-6
a_u = g_d * ufl.inner(sigma(u_tr), eps(v_tr)) * ufl.dx
L_u = ufl.inner(fem.Constant(
    domain, default_scalar_type((0.0, 0.0))), v_tr) * ufl.dx

# Bottom u = 0, top u_y = disp
def bottom(x): return np.isclose(x[1], 0.0)
def top(x):    return np.isclose(x[1], 1.0)
bot_facets = mesh.locate_entities_boundary(
    domain, fdim, bottom)
top_facets = mesh.locate_entities_boundary(
    domain, fdim, top)
u_bot = fem.Function(V_u)
u_bot.x.array[:] = 0.0
u_top = fem.Function(V_u)
def set_top(x):
    arr = np.zeros_like(x[:gdim])
    arr[1] = disp_val
    return arr
u_top.interpolate(set_top)
bc_bot = fem.dirichletbc(u_bot,
    fem.locate_dofs_topological(V_u, fdim, bot_facets))
bc_top = fem.dirichletbc(u_top,
    fem.locate_dofs_topological(V_u, fdim, top_facets))

prob_u = LinearProblem(
    a_u, L_u, bcs=[bc_bot, bc_top], u=u_h,
    petsc_options_prefix="frac_u_",
    petsc_options={{"ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps"}})
prob_u.solve()

# --- Damage subproblem (u fixed) ---
d_tr = ufl.TrialFunction(V_d)
w_tr = ufl.TestFunction(V_d)
psi = psi_plus(u_h)
a_d = ((Gc_val / l0_val) * d_tr * w_tr * ufl.dx
       + (Gc_val * l0_val) * ufl.inner(ufl.grad(d_tr),
                                       ufl.grad(w_tr)) * ufl.dx
       + 2.0 * psi * d_tr * w_tr * ufl.dx)
L_d = 2.0 * psi * w_tr * ufl.dx

# Keep d=1 fixed at the initial notch
notch_func = fem.Function(V_d)
notch_func.x.array[:] = 0.0
notch_func.x.array[notch_dofs] = 1.0
bc_d = fem.dirichletbc(
    notch_func,
    fem.locate_dofs_geometrical(V_d, initial_notch))
prob_d = LinearProblem(
    a_d, L_d, bcs=[bc_d], u=d_h,
    petsc_options_prefix="frac_d_",
    petsc_options={{"ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps"}})
prob_d.solve()
# Enforce d in [0, 1] (irreversibility / bound projection)
d_h.x.array[:] = np.clip(d_h.x.array, 0.0, 1.0)

u_norm = np.sqrt(domain.comm.allreduce(
    fem.assemble_scalar(fem.form(
        ufl.inner(u_h, u_h) * ufl.dx))))
d_norm = np.sqrt(domain.comm.allreduce(
    fem.assemble_scalar(fem.form(d_h * d_h * ufl.dx))))
# --- Write the result where a reader can open it ------------------------------
# This template printed a norm and wrote nothing, so a model that ran it had
# no field to look at, plot or verify. XDMFFile writes P1 functions; the
# openPASO backend converts result.xdmf/.h5 into a standard result.vtu after
# the run (the same route the Poisson template uses). Higher-order or
# non-Lagrange functions are interpolated into P1 for output first.
import json
from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "result.xdmf", "w") as _x:
    _x.write_mesh(domain); _x.write_function(u_h); _x.write_function(d_h)
with open("results_summary.json", "w") as _f:
    json.dump({{"l2_norm_u": float(u_norm), "l2_norm_d": float(d_norm), "max_d": float(d_h.x.array.max())}}, _f, indent=2)
print(f"||u||_L2 = {{u_norm}}")
print(f"||d||_L2 = {{d_norm}}, max(d) = {{d_h.x.array.max()}}")
'''
