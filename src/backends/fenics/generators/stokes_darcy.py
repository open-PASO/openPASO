"""Stokes-Darcy coupled flow generator for FEniCSx/dolfinx.

Variants: 2d

Brinkman-style unified formulation: single mixed (u, p)
Taylor-Hood problem on the whole domain, with a Darcy
penalty (mu/K)*u added on the porous subdomain. This
sidesteps explicit Beavers-Joseph-Saffman interface
coupling at the cost of using a single-velocity model
across both regions — the Darcy region is identified by
its very small effective permeability K (large penalty
mu/K).

The full BJS-coupled formulation requires:
  * separate spaces on each subdomain,
  * interface mortar coupling or Lagrange multipliers,
  * separate u_S, p_S, u_D, p_D unknowns,
  * BJS slip: u_tangential proportional to shear stress.

Brinkman is simpler, runs to convergence, and exposes the
right API surfaces (MeshTags, subdomain markers,
dx(subdomain_id)). For production multi-domain coupling
the user should consult demo_stokes-darcy or
mixed-dimensional dolfinx examples.
"""


KNOWLEDGE = {
    # ─────────────────────────────────────────────────────────────────
    # _SERVING_STATUS (added 2026-08-03)
    # This dict is SHADOWED and is NOT what an agent receives.
    # fenics/backend.py:get_knowledge() returns
    # src/tools/deep_knowledge.py::_FENICS_KNOWLEDGE['stokes_darcy'] for this
    # physics and never falls through to here. Editing the pitfalls
    # below changes nothing an agent can see. The claims here were NOT
    # re-verified in the 2026-08-03 execution pass for exactly that
    # reason — treat them as unverified history, and make corrections
    # in deep_knowledge.py instead.
    # ─────────────────────────────────────────────────────────────────
    "description": (
        "Coupled Stokes (free fluid) / Darcy (porous "
        "medium) flow. Two common formulations: (a) full "
        "BJS-coupled with separate variables and interface "
        "mortar, (b) Brinkman-style unified u-p with "
        "subdomain-dependent permeability."),
    "weak_form_brinkman": (
        "(mu * inner(grad(u), grad(v)) + (mu/K(x)) * "
        "inner(u, v)) dx - div(v)*p*dx - div(u)*q*dx "
        "= inner(f, v)*dx"),
    "function_space": (
        "Taylor-Hood mixed: P2 vector velocity + P1 "
        "scalar pressure via basix.ufl.mixed_element. "
        "Same space on the WHOLE domain in Brinkman; "
        "separate spaces in BJS-coupled."),
    "solver": {
        "brinkman": ("Direct (MUMPS) for moderate sizes; "
                     "block AMG + Schur for large."),
        "bjs_coupled": (
            "Block preconditioner per subdomain, "
            "mortar coupling on the interface."),
    },
    "pitfalls": [
        "[Numerical] Brinkman penalty (mu/K) scales the "
        "Darcy region — K very small (1e-6) approximates "
        "an impermeable Darcy block well, K large "
        "approaches pure Stokes. Tuning K too small "
        "makes the system stiff. "
        "Signal: K=1e-12 does not make the MUMPS direct "
        "solve report failure: preonly+LU(MUMPS) returns "
        "KSPConvergedReason 4 (CONVERGED_ITS) with a tiny "
        "residual. preonly can only report CONVERGED_ITS "
        "or DIVERGED_PC_FAILED (-11, e.g. a failed MUMPS "
        "factorisation), never DIVERGED_INDEFINITE_PC, so "
        "check the residual and the physics rather than "
        "the reason.",
        "[API] Subdomain-restricted integration uses "
        "ufl.Measure('dx', domain=domain, "
        "subdomain_data=cell_tags) and then dx(1) / "
        "dx(2) — the subdomain_id argument indexes the "
        "MeshTags integer values. "
        "Signal: forgetting subdomain_data= produces a "
        "form that integrates over the WHOLE mesh "
        "ignoring tags; numerical solution is then the "
        "uniform-K Stokes solution, with no Darcy "
        "behavior in the porous region.",
        "[API] dolfinx 0.10+ MeshTags constructor: "
        "mesh.meshtags(domain, dim, indices, values). The "
        "int32-array requirement this entry used to state is NOT "
        "enforced and the two quoted TypeErrors do not exist. "
        "Measured: Python lists are accepted (values come back as "
        "int64), float64 values are accepted and STAY float64, and "
        "int64 indices are accepted — all four spellings then select "
        "the same facets through ufl.Measure('ds', subdomain_data=mt) "
        "and assemble the identical number (0.8535533905932737 on the "
        "test case). The only rejection is a non-numeric dtype: a "
        "'<U1' string array gives 'NotImplementedError: Type <U1 not "
        "supported.'. Signal: there is none — this failure mode is "
        "SILENT, so do not write an except-TypeError guard around the "
        "constructor and read its silence as proof of correct dtypes. "
        "If a downstream consumer needs int32, assert "
        "mt.values.dtype yourself. (Verified by execution 2026-08-07, "
        "dolfinx 0.10.0.)",
        "[Physics] Full BJS-coupled formulation requires "
        "separate Stokes velocity and Darcy pressure "
        "spaces with interface mortar coupling. The "
        "Brinkman shortcut here is an APPROXIMATION — "
        "for production multi-domain flow with "
        "well-defined interface stress balance, use "
        "mixed-dimensional dolfinx or DUNE-mmesh. "
        "Signal: comparing the Brinkman tangential "
        "velocity at the interface against the BJS "
        "slip law u_t = (K/mu)^(1/2) * du_t/dn shows "
        "an O(1) discrepancy with no convergence to the "
        "BJS reference as the mesh is refined.",
    ],
}

VARIANTS = ["2d"]


def generate(variant: str, params: dict) -> str:
    generators = {
        "2d": _stokes_darcy_2d,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(
            f"Unknown variant: {variant!r}. "
            f"Available: {list(generators)}")
    return gen(params)


def _stokes_darcy_2d(params: dict) -> str:
    """FORMAT TEMPLATE — Brinkman-style coupled flow on a
    unit square. Top half (y > 0.5) is Stokes (effective
    K → infinity, penalty mu/K → 0). Bottom half is Darcy
    (small K, large penalty). Driven by a horizontal
    inflow on the left at the Stokes layer."""
    nx = params.get("nx", 24)
    mu = params.get("mu", 1.0)
    K_darcy = params.get("K_darcy", 1.0e-3)
    inflow_v = params.get("inflow_v", 1.0)
    return f'''\
"""Coupled Stokes-Darcy (Brinkman penalty) — FEniCSx/dolfinx"""
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
domain.topology.create_connectivity(tdim, tdim)

# --- Subdomain markers: top half = Stokes (tag 1),
# --- bottom half = Darcy (tag 2) ---
cell_indices = np.arange(
    domain.topology.index_map(tdim).size_local,
    dtype=np.int32)
midpoints = mesh.compute_midpoints(
    domain, tdim, cell_indices)
cell_values = np.where(
    midpoints[:, 1] > 0.5,
    np.int32(1),  # Stokes
    np.int32(2),  # Darcy
).astype(np.int32)
cell_tags = mesh.meshtags(
    domain, tdim, cell_indices, cell_values)

# --- Taylor-Hood mixed (u, p) on whole domain ---
P2 = basix.ufl.element("Lagrange", domain.basix_cell(), 2,
                       shape=(gdim,))
P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
TH = basix.ufl.mixed_element([P2, P1])
W = fem.functionspace(domain, TH)
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)

mu_val = {mu}
K_darcy_val = {K_darcy}
# Effectively infinite K in Stokes → zero Brinkman penalty
K_stokes_val = 1.0e6
penalty_darcy = mu_val / K_darcy_val
penalty_stokes = mu_val / K_stokes_val

dx_subdomain = ufl.Measure(
    "dx", domain=domain, subdomain_data=cell_tags)

a = (mu_val * ufl.inner(ufl.grad(u), ufl.grad(v))
        * dx_subdomain(1)
     + mu_val * ufl.inner(ufl.grad(u), ufl.grad(v))
        * dx_subdomain(2)
     + penalty_stokes * ufl.inner(u, v) * dx_subdomain(1)
     + penalty_darcy  * ufl.inner(u, v) * dx_subdomain(2)
     - ufl.div(v) * p * dx_subdomain(1)
     - ufl.div(v) * p * dx_subdomain(2)
     - ufl.div(u) * q * dx_subdomain(1)
     - ufl.div(u) * q * dx_subdomain(2))
L = ufl.inner(fem.Constant(
    domain, default_scalar_type((0.0, 0.0))), v) * ufl.dx

# --- BCs ---
W0 = W.sub(0)
V0, _ = W0.collapse()
def left(x):
    return np.isclose(x[0], 0.0) & (x[1] > 0.5)
def walls(x):
    return ((np.isclose(x[1], 0.0))
            | (np.isclose(x[1], 1.0))
            | (np.isclose(x[0], 1.0)))

left_facets  = mesh.locate_entities_boundary(
    domain, fdim, left)
wall_facets  = mesh.locate_entities_boundary(
    domain, fdim, walls)

u_in = fem.Function(V0)
def set_in(x):
    arr = np.zeros_like(x[:gdim])
    arr[0] = {inflow_v}
    return arr
u_in.interpolate(set_in)
u_wall = fem.Function(V0)
u_wall.x.array[:] = 0.0
bc_in = fem.dirichletbc(
    u_in,
    fem.locate_dofs_topological((W0, V0),
                                fdim, left_facets),
    W0)
bc_wall = fem.dirichletbc(
    u_wall,
    fem.locate_dofs_topological((W0, V0),
                                fdim, wall_facets),
    W0)

problem = LinearProblem(
    a, L, bcs=[bc_in, bc_wall],
    petsc_options_prefix="stokes_darcy_",
    petsc_options={{"ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps"}})
sol = problem.solve()
u_sol = sol.sub(0).collapse()
p_sol = sol.sub(1).collapse()
# Velocity magnitude in Stokes vs Darcy region
u_mag_form = ufl.sqrt(ufl.inner(u_sol, u_sol))
u_stokes = fem.assemble_scalar(fem.form(
    u_mag_form * dx_subdomain(1)))
u_darcy = fem.assemble_scalar(fem.form(
    u_mag_form * dx_subdomain(2)))
u_stokes = domain.comm.allreduce(u_stokes)
u_darcy = domain.comm.allreduce(u_darcy)
# --- Write the result where a reader can open it ------------------------------
# This template printed a norm and wrote nothing, so a model that ran it had
# no field to look at, plot or verify. XDMFFile writes P1 functions; the
# openPASO backend converts result.xdmf/.h5 into a standard result.vtu after
# the run (the same route the Poisson template uses). Higher-order or
# non-Lagrange functions are interpolated into P1 for output first.
import json
from dolfinx.io import XDMFFile
_Vvis = fem.functionspace(domain, basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,)))
_uo = fem.Function(_Vvis, name="velocity"); _uo.interpolate(sol.sub(0).collapse())
_po = sol.sub(1).collapse(); _po.name = "pressure"
with XDMFFile(domain.comm, "result.xdmf", "w") as _x:
    _x.write_mesh(domain); _x.write_function(_uo); _x.write_function(_po)
with open("results_summary.json", "w") as _f:
    json.dump({{"u_l1_stokes": float(u_stokes), "u_l1_darcy": float(u_darcy), "n_dofs": W.dofmap.index_map.size_global}}, _f, indent=2)
print(f"||u||_L1 stokes = {{u_stokes}}, ||u||_L1 darcy = {{u_darcy}}")
print(f"darcy/stokes ratio = {{u_darcy / max(u_stokes, 1e-12)}}")
print(f"||p||_L2 = {{np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form(p_sol * p_sol * ufl.dx))))}}")
'''
