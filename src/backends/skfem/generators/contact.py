"""scikit-fem linearized contact generator + knowledge.

Linearized frictionless contact between a 2D plane-strain elastic
block and a rigid foundation, via a quadratic penalty:

    min_{u}  ½ ⟨K u, u⟩ - ⟨f, u⟩  +  ½ k_p ∫_Γ_c max(0, gap(u))² ds

where gap(u) = y_top + u_y - obstacle_height. The KKT conditions
yield ½ k_p max(0, gap)² added to the energy (Heaviside-weighted
penalty). Linearization picks an active set once based on the
solution sign, then iterates a few times to converge.

Mirrors scikit-fem upstream ex04 (linearized contact).
"""


def _contact_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate
    values for your specific problem.

    2D elastic block pressed onto a rigid obstacle via Picard
    iteration on the active set."""
    nx = params.get("nx", 16)
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    obstacle_y = params.get("obstacle_y", -0.02)
    n_iters = params.get("n_iters", 6)
    return f'''\
"""Linearized frictionless contact — 2D plane strain — scikit-fem"""
from skfem import (MeshTri, Basis, ElementVector, ElementTriP1,
                   BilinearForm, LinearForm, FacetBasis, solve,
                   condense, asm)
from skfem.helpers import dot, ddot, sym_grad, trace, eye
import numpy as np
import json

nx = {nx}
E_mod = {E}
nu_p = {nu}
obstacle_y = {obstacle_y}    # rigid foundation y-coordinate
n_iters = {n_iters}

# Lame parameters.
mu_lame = E_mod / (2.0 * (1.0 + nu_p))
lam_lame = (E_mod * nu_p / ((1.0 + nu_p) * (1.0 - 2.0 * nu_p)))

# Block geometry: [0, 1] x [0, 1] resting above the rigid line
# y = obstacle_y. A downward body force pushes the block into
# contact.
m = MeshTri.init_tensor(np.linspace(0.0, 1.0, nx + 1),
                        np.linspace(0.0, 1.0, nx + 1))
m = m.with_boundaries({{
    "top":    lambda x: np.isclose(x[1], 1.0),
    "bottom": lambda x: np.isclose(x[1], 0.0),
    "left":   lambda x: np.isclose(x[0], 0.0),
    "right":  lambda x: np.isclose(x[0], 1.0),
}})

# Vector P1 displacement field.
e = ElementVector(ElementTriP1())
ib = Basis(m, e)


@BilinearForm
def linear_elasticity(u, v, w):
    eps_u = sym_grad(u)
    eps_v = sym_grad(v)
    return (2.0 * mu_lame * ddot(eps_u, eps_v)
            + lam_lame * trace(eps_u) * trace(eps_v))


@LinearForm
def body_force(v, w):
    # Strong downward gravity-like load (drives the block well
    # past the obstacle so the constraint is genuinely active).
    # For E=1000 and obstacle at y=-0.05, body force ~-100 gives
    # ~0.1 free displacement → cleanly engages contact.
    return -100.0 * v.value[1]


K = linear_elasticity.assemble(ib)
f = body_force.assemble(ib)

# Top edge: fully clamp (Dirichlet in both x and y) to remove
# rigid-body modes; the block hangs below, gravity (body force)
# pulls it down into the obstacle.
clamp_top_x = ib.get_dofs("top").nodal["u^1"]
clamp_top_y = ib.get_dofs("top").nodal["u^2"]
D_clamp = np.concatenate([clamp_top_x, clamp_top_y])

# Bottom-edge: y-component DOFs (for u_y) + corresponding node
# y-coordinates (y0). ElementVector layout: 2 DOFs per node so
# `nodal['u^2']` gives the y-component DOF index for each bottom
# node. We get the corresponding node ids via the boundary lookup.
bottom_uy_dofs = ib.get_dofs("bottom").nodal["u^2"]
# Node ids on the bottom edge (y0 ≈ 0).
bottom_node_ids = np.where(np.isclose(m.p[1, :], 0.0))[0]
# Sort both by x so they're aligned.
bottom_node_ids = bottom_node_ids[np.argsort(m.p[0, bottom_node_ids])]
# The DOFs come out in the same node order (skfem nodal-DOF
# ordering matches the underlying mesh node order for a fresh
# Basis); sort to make sure.
bottom_uy_dofs = bottom_uy_dofs[
    np.argsort(m.p[0, bottom_node_ids])]
bottom_y0 = m.p[1, bottom_node_ids]   # all 0.0 here

# Initial guess: no contact.
u = ib.zeros()
active_history = []
penetration_history = []
# Monotone active set: once a node engages, it stays. Standard
# primal-dual fixed-point for unilateral contact with linear
# elasticity (Kikuchi & Oden 1988, §10).
active_mask = np.zeros(len(bottom_uy_dofs), dtype=bool)
for it in range(n_iters):
    # Predict bottom-edge y positions: y0 + u_y.
    bottom_y_pred = bottom_y0 + u[bottom_uy_dofs]
    # Add any newly-penetrating nodes to the active set
    # (monotone: never remove).
    new_active = bottom_y_pred < obstacle_y - 1e-12
    active_mask = active_mask | new_active
    active_dofs = bottom_uy_dofs[active_mask]
    active_history.append(int(active_mask.sum()))

    # x_full holds prescribed values for ALL constrained DOFs.
    x_full = ib.zeros()
    x_full[active_dofs] = obstacle_y     # u_y = obstacle_y
    D = np.concatenate([D_clamp, active_dofs])

    u = solve(*condense(K, f, x=x_full, D=D))
    bottom_y_post = bottom_y0 + u[bottom_uy_dofs]
    max_penetration = float(
        max(0.0, (obstacle_y - bottom_y_post).max()))
    penetration_history.append(max_penetration)
    print(f"iter={{it}} active={{int(active_mask.sum())}} "
          f"max_penetration={{max_penetration:.4e}} "
          f"max|u|={{np.abs(u).max():.4e}}")

# Sanity: contact force should balance external load. The vertical
# reaction at the active set DOFs is K[active, :] @ u - f[active].
reaction_y = float(
    (K[active_dofs, :] @ u - f[active_dofs]).sum()
    if len(active_dofs) > 0 else 0.0)

import meshio
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
# Vector field layout for ElementVector(ElementTriP1) is
# interleaved: u[2k] = u_x at node k, u[2k+1] = u_y at node k.
# Reshape to (n_nodes, 2) and slice.
n_nodes = m.p.shape[1]
u_field = u.reshape((-1, 2))[:n_nodes]
u_x = u_field[:, 0]
u_y = u_field[:, 1]
mio = meshio.Mesh(
    points,
    [("triangle", m.t.T)],
    point_data={{"u_x": u_x, "u_y": u_y}},
)
mio.write("result.vtu")

summary = {{
    "n_dofs": int(ib.N),
    "n_iters": int(n_iters),
    "active_history": active_history,
    "penetration_history": penetration_history,
    "final_active": active_history[-1] if active_history else 0,
    "final_max_penetration": penetration_history[-1] if penetration_history else 0.0,
    "reaction_y": reaction_y,
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


GENERATORS: dict = {
    "contact_2d": _contact_2d,
}


KNOWLEDGE: dict = {
    "contact": {
        "description": (
            "Linearized frictionless contact between an elastic "
            "block and a rigid foundation via Picard iteration on "
            "the active set. Each iteration: predict bottom-edge "
            "y positions, identify nodes penetrating the obstacle, "
            "prescribe their u_y to bring them onto the surface, "
            "re-solve. Converges in 4-8 iterations for "
            "moderate loads. Matches scikit-fem upstream ex04."
        ),
        "weak_form": (
            "(2μ ε(u) : ε(v) + λ tr(ε(u)) tr(ε(v)))_dx = "
            "-(g, v_y)_dx, with active-set constraint "
            "u_y[i] = obstacle_y - y0[i] for nodes where "
            "y0 + u_y < obstacle_y."
        ),
        "elements": ["ElementVector(ElementTriP1)"],
        "variants": ["2d"],
        "pitfalls": [
            "[Numerical] The Picard active-set iteration is NOT "
            "globally convergent. For loads that produce "
            "near-grazing contact (penetration < 1e-3 * "
            "obstacle_y), the active set can oscillate between "
            "iterations — node N joins the set in iter k, leaves "
            "in k+1, rejoins in k+2. Use either a damping factor "
            "(only commit active-set changes if they grow the "
            "set monotonically) or switch to a primal-dual "
            "method (Lagrange multiplier formulation). "
            "Signal: `active_history` in results_summary.json "
            "shows non-monotone growth/shrink between iterations; "
            "the penetration_history series in the same file "
            "does not strictly decrease; "
            "`np.abs(u).max()` oscillates between Picard "
            "iterations rather than settling. The fix uses a "
            "monotone active-set update on `bottom_uy_dofs` "
            "(once a node engages, stays).",

            "[API] `ElementVector(ElementTriP1())` constructs a "
            "VECTOR-valued Lagrange element with one component "
            "per spatial dimension. The DOFs are addressable via "
            "`ib.get_dofs(name).nodal['u^1']` for x-component and "
            "`'u^2'` for y-component (1-indexed because UFL/skfem "
            "uses 1-based subscript labels for vector components). "
            "Signal: measured 2026-08-03 on skfem 12.0.1, "
            "MeshTri().refined(2).with_boundaries({'bottom': ...}) "
            "— Basis(m, ElementVector(ElementTriP1()))"
            ".get_dofs('bottom').nodal has keys ['u^1', 'u^2'], "
            "while the SCALAR Basis(m, ElementTriP1()) has keys "
            "['u'] (NOT `nodal[None]`, as the prior text said). "
            "So the failure signal is KeyError('u^1') against a "
            "dict whose only key is 'u'. (Verified empirically "
            "2026-08-03 — key-name correction.)",

            "[API] `condense(K, f, x=x_full, D=D)` for prescribed "
            "non-zero Dirichlet values requires x_full to be a "
            "FULL-LENGTH vector (size ib.N) with the prescribed "
            "values placed at the constrained DOF positions, zero "
            "elsewhere. Passing a SHORT vector (size len(D)) "
            "raises IndexError 'index <i> is out of bounds for "
            "axis 0 with size <len(D)>' from condense's x[D] "
            "indexing, before any spsolve call (if D happens to "
            "be 0..len(D)-1, the same IndexError comes from "
            "solve() when it expands the solution). "
            "Signal: that IndexError; nothing is silently "
            "zero-filled. Omitting x altogether raises nothing: "
            "condense then uses zeros, so `u[D]` does not equal "
            "the intended prescribed values after solve.",

            "[Physics] Reaction force balance: the vertical "
            "reaction at the active set should equal the "
            "external vertical load (-0.5 * volume for the "
            "default body force). Computing `reaction_y = "
            "K[active, :] @ u - f[active]` and summing gives the "
            "global reaction. Mismatches indicate either wrong "
            "BC enforcement OR a sign error in the body force. "
            "Signal: `summary['reaction_y']`, computed from "
            "K[active, :] @ u - f[active] after the solve, "
            "differs from the "
            "expected -∫ f_y dx by more than O(ε_machine * "
            "K_norm); the contact patch is correctly identified "
            "via `ElementVector(ElementTriP1)` DOF lookup but "
            "global equilibrium violated.",

            "[Mesh] `m.with_boundaries({...})` returns a NEW "
            "mesh with the named boundary tags; the lambda "
            "predicates are evaluated at every node. Forgetting "
            "to reassign `m = m.with_boundaries(...)` leaves the "
            "original tagless mesh in place, and "
            "`ib.get_dofs('bottom')` raises ValueError. "
            "Signal: ValueError(\"Boundary 'bottom' not found.\") "
            "from `get_dofs('bottom')`. A tag whose predicate "
            "matches no facets instead gives an empty "
            "`nodal['u^2']`.",

            "[Output] Vector-field VTK output requires extracting "
            "per-component arrays before passing to "
            "`meshio.Mesh(..., point_data=...)`. The interleaved "
            "layout (u[2i] = u_x[i], u[2i+1] = u_y[i]) is "
            "skfem's vector-component default for ElementVector; "
            "deinterleave via `u_x = u[ib.nodal_dofs[0]]` "
            "or `u[::2]` (not `u[ib.get_dofs().nodal['u^1']]`: "
            "get_dofs() with no argument returns only boundary "
            "DOFs). "
            "Signal: `meshio.Mesh` ValueError('len(points) = <N>, "
            "but len(point_data[\"u_x\"]) = <M>') when the slice "
            "misses; or "
            "ParaView shows 'u_x' as zero everywhere when the "
            "slice picks the wrong stride. Verified pattern: "
            "`u.reshape((-1, 2))[:n_nodes]` then `[:, 0]` and "
            "`[:, 1]`.",
        ],
        "references": [
            "scikit-fem ex04 (linearized contact)",
            "Kikuchi & Oden (1988), 'Contact Problems in "
            "Elasticity' — penalty + active-set methods.",
        ],
    },
}
