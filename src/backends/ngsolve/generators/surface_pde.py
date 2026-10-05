"""NGSolve surface PDE generators and knowledge."""


def _surface_pde_3d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Laplace-Beltrami equation on a curved surface manifold (sphere)."""
    order = params.get("order", 3)
    maxh = params.get("maxh", 0.3)
    return f'''\
"""Laplace-Beltrami on sphere surface — surface FEM — NGSolve"""
from ngsolve import *
from netgen.occ import *
import json

# Geometry: sphere surface — set for your problem
sphere = Sphere(Pnt(0,0,0), r=1.0)
geo = OCCGeometry(sphere)
mesh = Mesh(geo.GenerateMesh(maxh={maxh}))
mesh.Curve({order})

print(f"Surface mesh: {{mesh.ne}} elements, {{mesh.nv}} vertices")

# H1 space on the SURFACE (boundary of the volume mesh).
# definedon=mesh.Boundaries(".*") restricts the FE space to
# the spherical surface; without it, H1 lives on the full
# volume and the boundary integrals below would mix trace
# DOFs with interior DOFs.
# Audit 2026-06-02: prior version used H1(mesh, order=...,
# dirichlet="") on the volume mesh and then assembled
# `grad(u) * grad(v) * ds`. NGSolve raised
#   NgException: Trialfunction does not support BND-forms,
#               maybe a Trace() operator is missing, type=grad
# Fix: restrict the space to the surface AND wrap each
# trial/test occurrence with .Trace() so the BND integral
# typechecks. See Joachim Schöberl's surface-FEM example
# in iFEM lecture notes (definedon + .Trace() pattern).
# Laplace-Beltrami on the closed surface: -div_G grad_G u = f has a one-dimensional
# kernel (the constants), so the discrete system needs ONE constraint. The served
# template used to add 1e-8 * u * v as a regulariser; the mesh's quadrature error
# in int_G f (which is 0 exactly for 6z^2 - 2) was then divided by 1e-8 and came
# out as a constant of order 10^3 -- "max=3387" for a solution whose exact
# magnitude never exceeds 2/3. A Lagrange multiplier enforcing int_G u = 0 is the
# correct constraint and adds no error. Measured and fixed 2026-09-23.
fes_u = H1(mesh, order={order}, definedon=mesh.Boundaries(".*"))
fes_l = NumberSpace(mesh, definedon=mesh.Boundaries(".*"))
fes = fes_u * fes_l
(u_c, lam), (v_c, mu) = fes.TnT()
a_c = BilinearForm(fes)
a_c += grad(u_c).Trace() * grad(v_c).Trace() * ds
a_c += (lam * v_c.Trace() + mu * u_c.Trace()) * ds
a_c.Assemble()
f_expr = 6 * z * z - 2                    # = -Delta_G (z^2 - 1/3) on the unit sphere
f_c = LinearForm(fes)
f_c += f_expr * v_c.Trace() * ds
f_c.Assemble()
gf = GridFunction(fes)
gf.vec.data = a_c.mat.Inverse(fes.FreeDofs(), inverse="umfpack") * f_c.vec
gfu = gf.components[0]
max_val = max(gfu.vec)
min_val = min(gfu.vec)
print(f"Solution: max={{max_val:.8f}}, min={{min_val:.8f}}   (exact Y_2^0 = z^2 - 1/3 ranges over [-1/3, 2/3])")

# Pointwise comparison with the exact solution at the surface vertices, and an
# export ON THE SURFACE MESH. VTKOutput on the volume mesh evaluates a surface
# function inside the volume, where it is zero, and the result file showed
# nothing but zeros. Here the boundary triangles and their vertices are written
# directly, with the computed and exact fields side by side.
import numpy as np
import meshio
_pts = {{}}
_tris = []
for el in mesh.Elements(BND):
    _tris.append([v.nr for v in el.vertices])
    for v in el.vertices:
        _pts[v.nr] = mesh[v].point
_ids = sorted(_pts)
_index = {{nr: i for i, nr in enumerate(_ids)}}
_xyz = np.array([_pts[nr] for nr in _ids])
_val = np.array([gfu(mesh(*_pts[nr], BND)) for nr in _ids])
_exact = _xyz[:, 2] ** 2 - 1.0 / 3.0
_err = float(np.max(np.abs(_val - _exact)))
print(f"max|u_h - (z^2 - 1/3)| at surface vertices = {{_err:.3e}}  (P{order} on a curved surface: O(h^2) at best; information, not a verdict)")
meshio.Mesh(_xyz, [("triangle", np.array([[_index[n] for n in t] for t in _tris]))],
            point_data={{"solution": _val, "exact": _exact, "error": _val - _exact}}).write("result.vtu")

summary = {{
    "max_value": float(max_val),
    "min_value": float(min_val),
    "n_dofs": fes_u.ndof,
    "n_elements": mesh.ne,
    "order": {order},
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Surface PDE (Laplace-Beltrami) solve complete.")
'''


KNOWLEDGE = {
    "surface_pde": {
        "description": "PDEs on curved surface manifolds (Laplace-Beltrami, surface diffusion)",
        "spaces": "H1 on surface mesh (NGSolve automatically restricts to tangent plane)",
        "solver": "Direct or iterative — standard solvers work on surface meshes",
        "mesh": "OCC surfaces (Sphere, Cylinder, STEP import), mesh.Curve(order) for geometry approximation",
        "pitfalls": [
            (
                "[API] Use ds (surface measure) instead of dx "
                "(volume) for surface integrals, and wrap every "
                "trial/test occurrence in .Trace(). Signal: on "
                "the generator's own geometry "
                "(OCCGeometry(Sphere(...)) -> 3-D volume mesh, "
                "H1(mesh, order=3, "
                "definedon=mesh.Boundaries('.*'))), a "
                "BilinearForm with "
                "grad(u).Trace()*grad(v).Trace()*dx raises "
                "NgException. NGSolve assembles that line at "
                "run time from two literals that live in "
                "libngfem.so, 'Trialfunction does not support' "
                "and 'maybe a Trace() operator is missing', "
                "with the form kind and the proxy type filled "
                "in between and after them, so what you read is: "
                "Trialfunction does not support VOL-forms, maybe "
                "a Trace() operator is missing, type = "
                "gradboundary. The same form with *ds "
                "assembles. It does NOT silently give zero, and "
                "`mesh dim mismatch in BilinearForm` is not "
                "emitted by NGSolve 6.2.2604. Note dx on that "
                "mesh is still meaningful for VOLUME quantities "
                "(Integrate(CoefficientFunction(1)*dx, mesh) is "
                "approximately the ball volume 4*pi/3 = 4.188790: "
                "4.188891 with the template's Curve(3), 4.188792 "
                "with Curve(4); a bare 1*dx raises TypeError). "
                "(Verified empirically 2026-08-03 — signal-text "
                "correction.)"
            ),
            (
                "[API] grad on a surface mesh AUTOMATICALLY "
                "produces the TANGENTIAL (surface) gradient — "
                "no explicit projection needed. Manually "
                "projecting via (I - n n^T) * grad(u) applies "
                "the projection TWICE and only adds work. "
                "Signal: do not try to see the redundancy in "
                "WALL-CLOCK — the extra work is small, timings "
                "vary with host load, and a '2x the cost' claim "
                "is not testable on a busy machine. Two things "
                "are host-independent: the squared NORMAL "
                "component of grad(u).Trace() integrated over "
                "the surface is at round-off beside the "
                "tangential energy, which is the direct proof "
                "that the projection is already applied; and the "
                "re-projected form differs from the plain one by "
                "round-off in matrix norm, so it computes "
                "nothing new. REFINEMENT on the .Trace() rule: "
                "it is MANDATORY for trial and test PROXIES — "
                "omitting it raises NgException '... does not "
                "support BND-forms ... type = grad' — but it is "
                "NOT required for a GridFunction, where "
                "grad(gfu) under ds returns the same value. "
                "(Verified empirically 2026-08-03, refined "
                "2026-08-06 on NGSolve 6.2.2604 — the cost claim "
                "is replaced by a structural one.)"
            ),
            (
                "[Numerical] Laplace-Beltrami has kernel = "
                "constants, and NEITHER failure mode is loud. "
                "Signal: measured on the sphere, maxh=0.3, "
                "Curve(4), H1 order 3, f = 6z^2-2 — with no "
                "regularisation, "
                "a.mat.Inverse(fes.FreeDofs(), inverse='umfpack') "
                "returns WITHOUT raising and gives "
                "|u|_inf = 1.3e+09 — pure constant-mode blow-up. "
                "With the generator's 1e-8*u*v*ds regulariser the "
                "solve is stable and the SHAPE is right (L2 error "
                "9.7e-05 against the analytic Y_2^0 = z^2-1/3 "
                "after removing the mean) but the mean itself is "
                "off by +5.7e+01, which is why the shipped "
                "template prints max=3387.7 for a solution whose "
                "true range is [-1/3, 2/3]. Always subtract the "
                "mean (or pin a DOF) before reporting min/max. "
                "NOTE: `KSPSolve: DIVERGED_BREAKDOWN` is a PETSc "
                "string and is never emitted by NGSolve. "
                "(Verified empirically 2026-08-03 — signal-text "
                "correction + new failure characterisation.)"
            ),
            (
                "[Numerical] mesh.Curve(order) improves geometry "
                "approximation for curved surfaces. Signal: unit "
                "Sphere via OCCGeometry, maxh=0.3, "
                "Integrate(CoefficientFunction(1)*ds, mesh) "
                "against the exact 4*pi = "
                "12.56637061 — no Curve 12.31844056 (1.973% "
                "low), Curve(1) identical (it is a no-op), "
                "Curve(2) 12.56426008 (0.0168%), Curve(3) "
                "12.56657695 (0.0016%, absolute 2.1e-04), "
                "Curve(5) 12.56637022 (absolute 3.9e-07). "
                "Curve(1) buying "
                "nothing is the surprise. (Verified empirically "
                "2026-08-03 on NGSolve 6.2.2604 — prior "
                "'~2-5%' / '~1e-4' estimates both hold.)"
            ),
            (
                "[Numerical] For evolving surfaces (moving "
                "membranes, growth): use a deformation mapping + "
                "ALE approach with mesh.SetDeformation, which "
                "keeps the same DOFs, transports the field exactly "
                "and is undone by UnsetDeformation. Re-meshing "
                "every step does NOT cost you a few percent of "
                "L^2 norm — it costs you the ENTIRE field, "
                "silently. A new OCC geometry per step gives a new "
                "ndof, so the two GridFunctions are not even the "
                "same length. Signal: under SetDeformation the "
                "ndof is unchanged and Integrate(c*c*ds) scales by "
                "exactly the geometric factor of the deformation; "
                "with per-step re-meshing the ndof CHANGES, a raw "
                "coefficient-vector copy raises NgException "
                "'BaseVector::Set: size of me = <n> != size of "
                "other = <m>', and — the dangerous half — "
                "gf_new.Set(gf_old) returns WITHOUT raising and "
                "leaves gf_new exactly zero, so a try/except gate "
                "passes and a zero field propagates. Assert the "
                "transported field has non-zero norm after any "
                "cross-mesh transfer. (Verified 2026-08-06 on "
                "NGSolve 6.2.2604 — Tier-2 fixture "
                "surface_pde_ale_vs_remesh.)"
            ),
            (
                "[API] Surface meshes from OCC: Sphere(), "
                "Cylinder(), or any imported STEP/BREP surface via "
                "OCCGeometry(...). Feeding OCCGeometry a shape "
                "that still carries a SOLID builds a VOLUME mesh, "
                "so ds runs over the volume boundary and a plain "
                "H1 lives in the ball rather than on the surface. "
                "Select the surfaces with OCC.Glue(shape.faces) or "
                "geo.faces[i]. Do NOT use Mesh.dim as the "
                "discriminator: it stays 3 for a glued-faces "
                "surface mesh — the real test is mesh.ne == 0 (no "
                "volume elements). And do NOT reach for "
                "OCCGeometry(face, dim=2) to force it: that DOES "
                "return dim 2, but it FLATTENS the surface into a "
                "planar 2-D mesh whose area bears no relation to "
                "the curved one. Signal: with the solid still "
                "attached, mesh.ne is large and "
                "Integrate(CoefficientFunction(1)*dx, mesh) "
                "equals the enclosed VOLUME while an H1 space has "
                "many more DOFs than the surface one; with "
                "Glue(faces), mesh.ne == 0, "
                "Integrate(CoefficientFunction(1)*dx, mesh) is "
                "0.0, Integrate(CoefficientFunction(1)*ds, mesh) "
                "equals the meshed surface area (the same value "
                "as ds on the solid mesh; close to 4*pi only "
                "after mesh.Curve), and a volume form such as "
                "grad(u)*grad(v)*dx assembles an EMPTY matrix "
                "while NGSolve prints 'used dof inconsistency' — "
                "that message is the tell that you wrote dx where "
                "you meant ds. Passing a face LIST together with "
                "dim= raises a TypeError from the pybind11 "
                "overload resolver: it reports the constructor "
                "arguments as incompatible and then prints "
                "'arguments. The following argument types are "
                "supported:' with the three OCCGeometry "
                "overloads and the call it was invoked with. "
                "The head of that line is assembled at runtime "
                "from the bound signature, so only the tail is a "
                "literal in libngpy.so. (Verified 2026-08-06 on "
                "NGSolve 6.2.2604 — Tier-2 fixture "
                "surface_pde_occ_solid_vs_faces; re-reproduced "
                "2026-08-09 with "
                "OCCGeometry(list(Sphere(Pnt(0,0,0),1).faces), "
                "dim=2).)"
            ),
        ],
    },
}

GENERATORS = {
    "surface_pde_3d": _surface_pde_3d,
}
