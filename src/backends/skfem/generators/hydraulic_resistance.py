"""scikit-fem hydraulic-resistance generator + knowledge.

Computes the linear hydraulic resistance R = ΔP / Q for steady
Stokes flow through a 2D channel:

    -μ Δu + ∇p = 0     in Ω
    ∇·u        = 0     in Ω
    u = 0              on top/bottom walls (no-slip)
    p = P_in           on inlet (x=0)
    p = 0              on outlet (x=L)
    σ·n = 0 tangential — natural BC from pressure-driven setup

The closed-form Poiseuille resistance for a channel of height H and
length L is R_exact = 12 μ L / H³.  We compare the FE-computed
resistance to this analytic value.
"""


def _hydraulic_resistance_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate
    values for your specific problem.

    2D rectangular channel of length L and height H, pressure-
    driven Stokes flow via Taylor-Hood (P2-P1) on triangles."""
    L_chan = params.get("L", 2.0)
    H_chan = params.get("H", 0.1)
    nx = params.get("nx", 40)
    ny = params.get("ny", 6)
    mu = params.get("mu", 1.0)
    p_in = params.get("p_in", 1.0)
    return f'''\
"""Stokes hydraulic resistance R = ΔP / Q — scikit-fem"""
from skfem import (MeshTri, Basis, ElementTriP2, ElementTriP1,
                   ElementVector, BilinearForm, LinearForm,
                   FacetBasis, solve, condense, asm)
from skfem.helpers import dot, ddot, sym_grad, div, grad
import numpy as np
import scipy.sparse as sp
import json

L_chan = {L_chan}
H_chan = {H_chan}
nx = {nx}
ny = {ny}
mu = {mu}
p_in = {p_in}


# Rectangular tensor-product mesh, split into triangles.
m = MeshTri.init_tensor(np.linspace(0.0, L_chan, nx + 1),
                        np.linspace(0.0, H_chan, ny + 1))
m = m.with_boundaries({{
    "wall": lambda x: (np.isclose(x[1], 0.0)
                       | np.isclose(x[1], H_chan)),
    "inlet":  lambda x: np.isclose(x[0], 0.0),
    "outlet": lambda x: np.isclose(x[0], L_chan),
}})

# Taylor-Hood P2-P1. BOTH bases must use the same `intorder` so
# the bilinear coupling between them assembles — skfem default
# intorder differs by element degree, so explicit alignment is
# required. intorder=4 handles P2 sym_grad : sym_grad (rank-4
# integrand) exactly on triangles.
ev = ElementVector(ElementTriP2())
ep = ElementTriP1()
ib_u = Basis(m, ev, intorder=4)
ib_p = Basis(m, ep, intorder=4)
ib_fac_in = FacetBasis(m, ev, intorder=4,
                       facets=m.boundaries["inlet"])
ib_fac_out = FacetBasis(m, ev, intorder=4,
                        facets=m.boundaries["outlet"])


@BilinearForm
def stiffness(u, v, w):
    # 2*mu*(sym_grad u : sym_grad v).
    return 2.0 * mu * ddot(sym_grad(u), sym_grad(v))


@BilinearForm
def neg_div(u, p, w):
    # -div(u) * p (pressure × velocity-divergence coupling).
    return -div(u) * p


@LinearForm
def inlet_traction(v, w):
    # Natural BC at inlet (x=0): σ·n = -p_in * n with n=(-1,0).
    # The traction contribution to the velocity equation is
    # ∫_inlet (-p_in)·n · v ds = ∫_inlet (-p_in)·(-1) v_x ds
    #                          = ∫_inlet p_in v_x ds (positive).
    return p_in * v.value[0]


# Outlet pressure = 0 so no contribution from that boundary.
A = stiffness.assemble(ib_u)
B = neg_div.assemble(ib_u, ib_p)
f_u = inlet_traction.assemble(ib_fac_in)
f_p = ib_p.zeros()

# Block matrix: [[A, B^T], [B, 0]] x = [f_u, f_p].
K = sp.bmat([[A, B.T], [B, None]], format="csr")
F = np.concatenate([f_u, f_p])

# Dirichlet: u=0 on walls. Pin pressure at one outlet node to
# remove the constant null space.
wall_dofs = ib_u.get_dofs("wall").all()
# Pin a single pressure DOF at the outlet to fix the null space.
outlet_p_dofs = ib_p.get_dofs("outlet").all()
pin_p = ib_u.N + int(outlet_p_dofs[0])
D = np.concatenate([wall_dofs, [pin_p]])

x_full = np.zeros(ib_u.N + ib_p.N)
x_full[pin_p] = 0.0   # outlet pressure = 0

x = solve(*condense(K, F, x=x_full, D=D))
u = x[:ib_u.N]
p = x[ib_u.N:]

# Flow rate Q = ∫_outlet u_x ds (volumetric flow per unit depth).
@LinearForm
def flow_rate_form(v, w):
    return v.value[0]

# Integrate u_x over the outlet by interpolating to FacetBasis.
u_func = ib_u.interpolate(u)
# u_func.value: shape (2, n_facets, n_qpoints)
# Integrate ∫_outlet u_x ds via the FacetBasis weights.
Q = float(np.sum(ib_fac_out.interpolate(u).value[0]
                 * ib_fac_out.dx))

R_fe = p_in / Q
R_exact = 12.0 * mu * L_chan / (H_chan ** 3)
rel_err = abs(R_fe - R_exact) / R_exact

print(f"Channel L={{L_chan}} H={{H_chan}} mu={{mu}} p_in={{p_in}}")
print(f"  n_dofs_u={{ib_u.N}}  n_dofs_p={{ib_p.N}}")
print(f"  flow rate Q     = {{Q:.6e}}")
print(f"  resistance R    = {{R_fe:.6e}}")
print(f"  R_exact (12μL/H³)= {{R_exact:.6e}}")
print(f"  relative error   = {{rel_err:.4e}}")

import meshio
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
# Pressure lives on the P1 nodal basis = the mesh vertices.
p_nodal = p[:m.p.shape[1]]
mio = meshio.Mesh(points, [("triangle", m.t.T)],
                  point_data={{"p": p_nodal}})
mio.write("result.vtu")

summary = {{
    "n_dofs_u": int(ib_u.N),
    "n_dofs_p": int(ib_p.N),
    "L": float(L_chan), "H": float(H_chan), "mu": float(mu),
    "p_in": float(p_in),
    "Q": float(Q),
    "R_fe": float(R_fe),
    "R_exact": float(R_exact),
    "relative_error": float(rel_err),
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


GENERATORS: dict = {
    "hydraulic_resistance_2d": _hydraulic_resistance_2d,
}


KNOWLEDGE: dict = {
    "hydraulic_resistance": {
        "description": (
            "Computes the linear hydraulic resistance "
            "R = ΔP / Q for steady Stokes flow through a 2D "
            "rectangular channel. Pressure-driven boundary "
            "conditions (p=p_in at inlet, p=0 at outlet, no-slip "
            "on top/bottom walls). FE result compared to the "
            "closed-form Poiseuille resistance R = 12 μ L / H³."
        ),
        "weak_form": (
            "Stokes: 2μ ∫ ε(u):ε(v) - ∫ div(u) q - ∫ div(v) p = "
            "∫_inlet p_in n·v ds (outflow free)."
        ),
        "elements": [
            "ElementVector(ElementTriP2) velocity + ElementTriP1 "
            "pressure (Taylor-Hood, inf-sup stable)",
        ],
        "variants": ["2d"],
        "pitfalls": [
            "[Physics] The gap between the FE resistance and "
            "R = 12 mu L / H^3 in this template is NOT an "
            "entrance effect. It is an artefact of the VISCOUS "
            "FORM, and swapping the form makes it vanish "
            "entirely at every aspect ratio. The template "
            "assembles the stress form 2*mu*(sym_grad u : "
            "sym_grad v); replacing it with the vector "
            "Laplacian mu*(grad u : grad v), on the SAME "
            "geometry, mesh, boundary conditions and flux "
            "computation, reproduces R = 12 mu L / H^3 to "
            "round-off even at L/H = 2. A genuine entrance "
            "effect is a property of the geometry and would "
            "show up in both formulations; this one shows up in "
            "only one. The cause is the NATURAL boundary "
            "condition each form implies at a traction "
            "boundary: 2*mu*sym(grad u).n - p n for the stress "
            "form against mu*du/dn - p n for the Laplacian, "
            "which agree only where d(u_x)/dx = 0 — that is, "
            "only where the flow is already fully developed. "
            "So the choice of viscous form is a boundary-"
            "condition choice whenever any boundary is a "
            "traction boundary, not a cosmetic one. "
            "Three further corrections to the older wording, "
            "all measured: the FE resistance is NOT always "
            "higher than the analytic value (at small L/H the "
            "stress form gives a LOWER resistance); the "
            "deviation at L/H = 2 is a few percent, not ~50%; "
            "and refinement does NOT leave the gap fixed — it "
            "shrinks at one aspect ratio and grows at another, "
            "which no fixed physical offset would do. "
            "Signal: run the SAME case with both viscous forms "
            "before blaming physics. If the Laplacian form "
            "lands on the analytic value and the stress form "
            "does not, what you are looking at is the traction "
            "boundary condition, not the entrance length. "
            "(Verified empirically 2026-08-06 on skfem 12.0.1: "
            "the previous entry's own numbers reproduce exactly "
            "on the stress form and its attribution was still "
            "wrong.)",

            "[API] `scipy.sparse.bmat([[A, B.T], [B, None]], "
            "format='csr')` is the canonical Stokes block "
            "assembly in skfem. The `None` in the (1,1) block "
            "produces a zero block of the right shape — "
            "explicit `sp.csr_matrix((nP, nP))` works too but is "
            "verbose — the two are byte-identical, so they "
            "really are interchangeable. Passing a scalar zero, "
            "[[A, B.T], [B, 0]], fails. "
            "Signal: it raises ValueError, NOT TypeError, and "
            "the message is 'scipy sparse array classes do not "
            "support instantiation from a scalar' — so an "
            "`except TypeError` around the assembly does not "
            "catch it and a gate grepping for 'no supported "
            "conversion for types' or 'unsupported operand type "
            "for *' never matches. A float 0.0 fails "
            "identically, so the integer dtype had nothing to do "
            "with it either. Catch ValueError, or simply write "
            "None. (Verified 2026-08-06 on skfem 12.0.1 / scipy "
            "1.15.3 — the exception type and both quoted "
            "messages were wrong.)",

            "[Numerical] READ THIS BEFORE THE REST: the earlier "
            "version of this entry was wrong about ITS OWN "
            "TEMPLATE, not merely about the symptom to watch "
            "for. It asserted that this system carries a "
            "constant-pressure null space and that the solver "
            "fails without a pin. It does not, and it does not. "
            "A Stokes saddle-point system has a "
            "constant-pressure null space ONLY when nothing sets "
            "the pressure level. THIS TEMPLATE IS NOT SUCH A "
            "CASE: the inlet pressure traction is a natural BC "
            "that fixes the level, so the assembled system is "
            "non-singular WITH OR WITHOUT the outlet pin — "
            "removing `ib_u.N + outlet_p_dofs[0]` from D leaves "
            "the null space empty and leaves Q and R_fe "
            "unchanged. The pin is a convenience that puts p=0 "
            "at the outlet, not a well-posedness requirement. "
            "The null space appears when velocity is prescribed "
            "on the WHOLE boundary (enclosed flow, e.g. a lid- "
            "driven cavity), and there you must pin one pressure "
            "DOF or impose a mean-zero constraint. "
            "DO NOT GUARD THIS WITH `MatrixRankWarning: Matrix "
            "is exactly singular`. That warning does NOT fire on "
            "the pressure null space: the system is singular but "
            "CONSISTENT, SuperLU returns a particular solution, "
            "and you get a finite, usable velocity field with "
            "only the pressure LEVEL undetermined — no warning, "
            "no exception, no NaN. A `catch MatrixRankWarning` "
            "guard reads its own silence as success. The warning "
            "IS real, but it discriminates a different fault: it "
            "fires when the matrix is genuinely rank-deficient, "
            "which in this backend means the equal-order P1/P1 "
            "inf-sup violation described elsewhere in this "
            "knowledge, and there the solution is non-finite. "
            "Signal: the null space is observable, just not by "
            "that warning — solve twice pinning different "
            "pressure DOFs or different values and the velocity "
            "is identical to round-off while the pressure "
            "differs by a constant; or check the nullity of the "
            "condensed block directly. "
            "(Verified empirically 2026-08-06 on skfem 12.0.1 / "
            "scipy 1.15.3 — premise and signal both corrected.)",

            "[Physics] Inlet pressure traction enters as a "
            "NATURAL boundary condition in the velocity "
            "equation: ∫_inlet (-p_in) v_x ds where outward "
            "normal at x=0 is (-1, 0). The MINUS SIGN matters — "
            "wrong sign gives flow in the wrong direction "
            "(Q < 0) and R < 0. "
            "Signal: `summary['Q']` is negative; "
            "`results.vtu` shows pressure increasing in the "
            "flow direction (physically wrong); "
            "`relative_error` is order 1 (R has wrong sign).",

            "[API] `ib_fac_out.interpolate(u).value[0] * "
            "ib_fac_out.dx` computes the boundary integral "
            "∫_outlet u_x ds. `ib_fac_out.dx` is the array of "
            "quadrature weights * Jacobian on the outlet "
            "facets; multiplying by the interpolated u_x and "
            "summing gives the flow rate. Forgetting to use "
            "FacetBasis (using ib_u directly on a boundary "
            "integral) double-counts interior edges and "
            "produces a Q that's ~2x too large. "
            "Signal: `Q` in summary is 2× the expected "
            "Poiseuille value; `R_fe` is half of `R_exact`. "
            "Concretely: replacing `FacetBasis(m, ev, "
            "facets=m.boundaries['outlet'])` with the "
            "volumetric `Basis(m, ev)` and using `ib_u.dx` for "
            "the boundary integral over-counts; the fix uses "
            "`FacetBasis` + `ib_fac_out.dx`.",

            "[Numerical] Taylor-Hood P2-P1 on triangles is "
            "inf-sup stable (LBB satisfied); equal-order P1-P1 "
            "without stabilization is not. Signal: measured 2026-08-03 "
            "on skfem 12.0.1 with a manufactured Stokes solution "
            "(build your own divergence-free field from a stream "
            "function, take the pressure you like, and derive "
            "the source with sympy) over three uniform "
            "refinements: Taylor-Hood "
            "ElementVector(ElementTriP2()) + ElementTriP1 "
            "reaches its theoretical velocity L2 order of k+1 "
            "for k=2, with the pressure error falling steadily "
            "alongside it. The measured tables are deliberately "
            "not reproduced — they are the answer to the study, "
            "and only your own numbers are evidence. Swapping "
            "velocity to "
            "ElementVector(ElementTriP1()) (P1 pressure kept) "
            "does NOT merely 'fluctuate 50-200%': on the "
            "structured refined() mesh the block matrix is "
            "EXACTLY SINGULAR — scipy emits "
            "MatrixRankWarning('Matrix is exactly singular') "
            "from skfem/utils.py and every component of the "
            "solution comes back nan. On less symmetric meshes "
            "you get the classical checkerboard instead. Either "
            "way, check for nan AND for a checkerboard; a warning "
            "is the only thing scipy gives you and it is easy to "
            "miss. (Verified empirically 2026-08-03 — signal "
            "sharpened, rates measured.)",

            "[Output] Pressure DOF block in the solution vector "
            "[u; p] starts at index `ib_u.N`. Extracting "
            "`p_nodal = p[:m.p.shape[1]]` assumes the P1 "
            "pressure DOFs are ordered to match the mesh "
            "vertices, which is the skfem default for "
            "ElementTriP1. If a different element is used "
            "(e.g. ElementTriP2 for pressure), the DOF count "
            "exceeds the vertex count and this slice drops "
            "data. "
            "Signal: the range check is NOT reliable here, and "
            "for a smooth field it reads clean. Because the "
            "first doflocs of a higher-order pressure space "
            "still coincide with the mesh vertices, the "
            "truncated slice is a plausible vertex field of "
            "exactly the mesh's length, and whenever the "
            "extremes of the field happen to sit ON vertices "
            "the truncated slice has the SAME min and max as "
            "the full field — nothing is warned and nothing "
            "looks wrong in ParaView either. The range only "
            "moves when an extreme sits on an edge DOF, which "
            "is field-dependent. The guard that always works is "
            "structural and costs nothing: compare "
            "`basis_p.N` against `m.p.shape[1]` BEFORE slicing "
            "and refuse to take the slice when they differ. "
            "(Verified 2026-08-06 on skfem 12.0.1 — the "
            "print-versus-render symptom is unreliable.)",
        ],
        "references": [
            "scikit-fem ex18 (Stokes equations) / ex24 (Stokes "
            "flow with plane-Poiseuille inlet)",
            "White, F. M. (2011), 'Viscous Fluid Flow' 3rd ed., "
            "Ch. 3 (Poiseuille flow).",
        ],
    },
}
