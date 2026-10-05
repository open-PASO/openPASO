"""NGSolve Maxwell equations generators and knowledge."""


def _maxwell_3d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    3D magnetostatics with HCurl (Nedelec) elements."""
    order = params.get("order", 2)
    maxh = params.get("maxh", 0.3)
    return f'''\
"""Magnetostatics: curl-curl equation — HCurl (Nedelec) — NGSolve"""
from ngsolve import *
from netgen.csg import *
import json, math

# Geometry with material region — set for your problem.
# netgen.csg.CSGeometry.Add does NOT accept a 'mat=' kwarg
# (signature: Add(solid, bcmod=[], maxh=..., col=(), ...)).
# Attach material names by calling .mat('name') on the
# Solid itself BEFORE passing it to Add — see pitfall on
# CSG material tagging below.
geo = CSGeometry()
outer = OrthoBrick(Pnt(-1,-1,-1), Pnt(1,1,1)).bc("outer")
inner = OrthoBrick(Pnt(-0.3,-0.3,-0.3), Pnt(0.3,0.3,0.3))
geo.Add((outer - inner).mat("air"))
geo.Add(inner.mat("source"))
mesh = Mesh(geo.GenerateMesh(maxh={maxh}))

fes = HCurl(mesh, order={order}, dirichlet="outer", nograds=True)
u, v = fes.TnT()

mu0 = 4*math.pi*1e-7
# curl-curl + regularization
a = BilinearForm(fes)
a += 1/mu0 * curl(u)*curl(v)*dx + 1e-8/mu0 * u*v*dx
a.Assemble()

# Source current — set for your problem
J = mesh.MaterialCF({{"source": (0, 0, 1)}}, default=(0, 0, 0))
f = LinearForm(J*v*dx).Assemble()

gfu = GridFunction(fes)
gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

# Magnetic field B = curl(A)
print(f"DOFs: {{fes.ndof}}, Elements: {{mesh.ne}}")
vtk = VTKOutput(mesh, coefs=[gfu, curl(gfu)], names=["A_field", "B_field"],
                filename="result", subdivision=0)
vtk.Do()
summary = {{"n_dofs": fes.ndof, "n_elements": mesh.ne}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Maxwell magnetostatics solve complete.")
'''


KNOWLEDGE = {
    "maxwell": {
        "description": "Maxwell equations with HCurl (Nedelec) edge elements",
        "spaces": "HCurl(mesh, order=k, nograds=True) — tangential continuity",
        "solver": "Direct for small. HCurlAMG preconditioner for large systems",
        "pitfalls": [
            "[API] netgen.csg.CSGeometry.Add does NOT accept a "
            "'mat=' keyword argument — its signature is Add(solid, "
            "bcmod=[], maxh=..., col=(), transparent=False, "
            "layer=1). Calling geo.Add(solid, mat='source') raises "
            "TypeError 'Invoked with: <CSGeometry>, <Solid>; "
            "kwargs: mat=...'. Attach material names by chaining "
            ".mat('name') on the Solid BEFORE passing it to Add: "
            "geo.Add(inner.mat('source')); geo.Add((outer - "
            "inner).mat('air')). Signal: TypeError from "
            "netgen.libngpy._csg.CSGeometry.Add showing the "
            "'mat=' kwarg in the message. (Verified empirically "
            "2026-06-01 — Layer F catch.)",
            "[Numerical] For SOURCE problems (magnetostatics): use "
            "HCurl(..., nograds=True) to remove the gradient kernel, "
            "plus a 1e-8*u*v*dx regularisation. Signal: without "
            "nograds (and without regularisation), BilinearForm."
            "Assemble succeeds but BilinearForm.mat.Inverse raises "
            "NgException 'UmfpackInverse: Numeric factorization "
            "failed.' because the gradient kernel of HCurl is in "
            "the null space of curl-curl. UMFPACK's own line "
            "'WARNING: matrix is singular' is not part of that "
            "exception: it is separate C stdio output, which is "
            "held until the process exits when stdout is a file or "
            "a pipe. (Verified empirically 2026-06-01; the "
            "exception text re-measured 2026-10-01 on NGSolve "
            "6.2.2604 — older catalog text quoted the two as one "
            "message, and before that as a bare 'matrix is "
            "singular' / 'pivot too small' exception.)",
            "[Numerical] For EIGENVALUE problems: do NOT use "
            "nograds=True — it degrades accuracy and causes "
            "eigenvalues to converge FROM BELOW. Use the full "
            "HCurl space with ArnoldiSolver(shift=<near expected "
            "eigenvalue>). Signal: ArnoldiSolver eigenvalues "
            "computed with nograds=True are systematically "
            "smaller than analytic cavity eigenvalues "
            "by a LARGE margin, not a few percent. Signal: unit "
            "cube cavity (exact lowest k^2 = 2*pi^2 = 19.7392), "
            "HCurl(order=2, dirichlet='.*'), maxh=0.35, "
            "ArnoldiSolver(shift=15): with nograds=True the "
            "lowest three computed eigenvalues are 15.86 / 16.03 "
            "/ 16.34, i.e. -19.6% / -18.8% / -17.2% BELOW "
            "analytic (ndof 964); with nograds=False they are "
            "19.81 / 19.81 / 19.81, i.e. +0.37% / +0.38% / "
            "+0.38% ABOVE analytic (ndof 1836). Note the cavity "
            "formula k^2 = (m*pi/Lx)^2 + (n*pi/Ly)^2 in the prior "
            "text is the 2-D one; Maxwell here is 3-D, so use "
            "k^2 = pi^2*(l^2+m^2+n^2). (Verified empirically "
            "2026-08-03 on NGSolve 6.2.2604 — direction "
            "confirmed; the successive magnitude estimates "
            "'1-3%' then '~7%' were both understated.)",
            "[Physics] B = curl(A) — magnetic field is the curl of "
            "the vector potential. Forgetting the curl gives B == A "
            "(vector potential treated as field) and Tesla units "
            "off by order(curl) ~ 1/L. Signal: the post-processed "
            "max_B of the HCurl GridFunction is on the order of "
            "the prescribed Dirichlet value of the A "
            "CoefficientFunction directly (no curl spatial "
            "derivative taken).",
            "[Syntax] Complex-valued for time-harmonic: HCurl("
            "mesh, complex=True). On a real HCurl (complex=False) "
            "space, adding a BilinearForm integrator with an "
            "explicit complex coefficient (e.g. 1j*curl(u)*"
            "curl(v)*dx) raises NgException at BilinearForm."
            "Assemble. Signal: NgException with text 'real "
            "Evaluate called for complex ScaleCF' from 'Assemble "
            "BilinearForm'. (Verified empirically 2026-06-01 — "
            "prior catalog wording 'complex values cannot be "
            "assigned to a real FESpace' does not appear in "
            "NGSolve 6.2; the actual emitted string is the "
            "ScaleCF one above.)",
            "[Physics] This generator is 3D. On a 2D mesh, "
            "HCurl(mesh) is still a vector (2-component) Nedelec "
            "space, but curl(u) is a scalar, not a vector. "
            "Signal: in 2D, u.dim == 2 and curl(u).dim == 1 on an "
            "HCurl space, and BilinearForm += curl(u)*curl(v)*dx "
            "assembles the 2-D in-plane curl-curl operator "
            "without warning. fes.dim "
            "is 1 for HCurl in both 2D and 3D (it is the block "
            "size), so it does not tell 2D from 3D; check "
            "u.dim / curl(u).dim instead.",
            "[Numerical] ArnoldiSolver shift: set near expected "
            "eigenvalue range, not near zero. Estimate the lowest "
            "eigenvalue analytically first (k^2 ~ (pi/L)^2 for "
            "cavity); shift=0.5*k^2_expected works well. Signal: "
            "ArnoldiSolver with shift=0.0 on a curl-curl matrix "
            "raises NgException 'UmfpackInverse: Numeric "
            "factorization failed.' BEFORE returning any "
            "eigenvalues (UMFPACK's 'WARNING: matrix is singular' "
            "line is separate output, released at process exit "
            "when stdout is a file or a pipe) — the "
            "shifted operator A - 0*M = A has the gradient kernel "
            "in its null space, so the direct factorisation of "
            "the shift-and-invert system fails. With a small "
            "non-zero shift away from physical spectrum, the "
            "solver runs but returns eigenvalues clustered near "
            "the shift, missing the physical modes. (Verified "
            "empirically 2026-06-01 — the prior catalog wording "
            "'returns eigenvalues near 0' was wrong; the real "
            "failure is the factorisation error before any "
            "eigenvalue is returned.)",
            "[API] Eigenvalue solvers return complex values even "
            "for real-symmetric problems — take .real before "
            "comparison. Signal: numpy.array(ArnoldiSolver result) "
            "has dtype complex128; comparing to analytic real "
            "eigenvalues without .real raises TypeError or "
            "produces nan from complex>real.",
            "[Syntax] Applying curl() to a scalar H1 trial "
            "function is rejected, but the message and the catch "
            "site are DIMENSION-DEPENDENT — and Maxwell is a 3-D "
            "physics, which is the bad case. In 2D: "
            "curl(H1(mesh).TrialFunction()) raises "
            "NgException('Operator \"curl\" does not exist for "
            "H1HighOrderFESpace!') at form construction. In 3D "
            "the SAME expression SUCCEEDS — it silently resolves "
            "to the 'boundaryrot' operator and returns a dim-3 "
            "ProxyFunction — and the failure is deferred to "
            "assembly, where it reads NgException('Trialfunction "
            "does not support VOL-forms, maybe a Trace() operator "
            "is missing, type = boundaryrot'). Separately, "
            "curl() on VectorH1 raises "
            "NgException('Operator \"curl\" does not exist for "
            "VectorH1FESpace!') in both dimensions. Signal: grep "
            "for 'boundaryrot' in 3D, 'does not exist for "
            "H1HighOrderFESpace' in 2D. (Verified empirically "
            "2026-08-03 on NGSolve 6.2.2604 — the prior entry "
            "gave only the 2-D message.)",
            "[Syntax] grad() IS defined on HCurl — the prior "
            "catalog reason for this pitfall was WRONG. On a 3D "
            "HCurl(order=1), grad(u) builds fine and returns a "
            "ProxyFunction with dim=9 / dims=(3,3), and "
            "BilinearForm += InnerProduct(grad(u), grad(v))*dx "
            "ASSEMBLES cleanly (nnz=2712 on unit_cube maxh=0.5). "
            "What actually fails is the literal scalar-Poisson "
            "spelling grad(u)*grad(v)*dx, because '*' on two 3x3 "
            "matrices is a matrix product, not an inner product, "
            "so the integrand is not scalar. Signal: "
            "NgException('SymbolicBFI needs scalar-valued "
            "CoefficientFunction') — the message mentions "
            "neither 'grad' nor 'HCurl', so don't grep for those. "
            "(Verified empirically 2026-08-03 on NGSolve "
            "6.2.2604 — catalog-drift correction.)",
            "[API] LinearForm += f*v*dx where f is a 3-vector "
            "and v is a scalar (H1) test function is a vector-"
            "source on a scalar form. Signal: NgException about "
            "SymbolicLFI requiring 'scalar-valued' integrand; "
            "the same JJ vector that works on HCurl test "
            "functions raises immediately on H1 test functions.",
        ],
    },
}

GENERATORS = {
    "maxwell_3d_magnetostatics": _maxwell_3d,
}
