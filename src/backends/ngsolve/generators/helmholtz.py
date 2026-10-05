"""NGSolve Helmholtz equation generators and knowledge."""


def _helmholtz_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Helmholtz equation with PML absorbing layer."""
    k = params.get("k", 10)
    order = params.get("order", 4)
    maxh = params.get("maxh", 0.05)
    return f'''\
"""Helmholtz: -Δu - k²u = f with PML — NGSolve (complex-valued)"""
from ngsolve import *
from netgen.geom2d import SplineGeometry
import json

geo = SplineGeometry()
geo.AddCircle((0,0), r=1.0, bc="outer")
geo.AddCircle((0,0), r=0.7, leftdomain=2, rightdomain=1)
geo.SetMaterial(1, "pml")
geo.SetMaterial(2, "inner")
mesh = Mesh(geo.GenerateMesh(maxh={maxh}))

mesh.SetPML(pml.Radial(origin=(0, 0), rad=0.7, alpha=2j), definedon="pml")

fes = H1(mesh, order={order}, complex=True, dirichlet="outer")
u, v = fes.TnT()
k = {k}
a = BilinearForm(grad(u)*grad(v)*dx - k**2*u*v*dx).Assemble()

# Point source at origin
f = LinearForm(exp(-100*(x**2+y**2))*v*dx).Assemble()

gfu = GridFunction(fes)
gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

print(f"Helmholtz k={k}, DOFs: {{fes.ndof}}")
vtk = VTKOutput(mesh, coefs=[gfu.real, gfu.imag],
                names=["Re_u", "Im_u"], filename="result", subdivision=2)
vtk.Do()
summary = {{"k": k, "n_dofs": fes.ndof}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Helmholtz solve complete.")
'''


KNOWLEDGE = {
    "helmholtz": {
        "description": "Helmholtz equation with PML (perfectly matched layer)",
        "spaces": "H1(mesh, order=k, complex=True) — MUST use complex=True",
        "solver": "Direct for moderate k. For large k, use GMRES with multigrid",
        "pitfalls": [
            "[Syntax] complex=True flag required on FESpace for any "
            "Helmholtz form that carries a complex coefficient "
            "(absorbing BC, PML, time-harmonic source). Without it, "
            "the BilinearForm with a 1j*k*u*v*dx term hits the "
            "same ScaleCF check that maxwell#3 trips on. Signal: "
            "NgException 'real Evaluate called for complex ScaleCF "
            "in Assemble BilinearForm \"biform_from_py\"' from "
            "BilinearForm.Assemble. (Verified empirically "
            "2026-06-01 — identical wording to the maxwell#3 case; "
            "the catch site is the BFI assembler.)",
            "[API] pml.Radial REQUIRES an 'origin' positional "
            "argument (the center of the PML region). Real "
            "signature: pml.Radial(origin, rad=1, alpha=1j). The "
            "earlier catalog template called pml.Radial(rad=0.7, "
            "alpha=2j) without origin. Use origin=(0, 0) for 2-D "
            "centered PML or origin=(0, 0, 0) for 3-D. "
            "Signal: pml.Radial(rad=..., alpha=...) without "
            "origin raises a pybind11 TypeError whose only "
            "literal is 'arguments. The following argument types "
            "are supported:'; the head is built from the bound "
            "signature, so the line reads Radial(): incompatible "
            "function arguments. at the call site, BEFORE "
            "mesh.SetPML is reached. (Verified empirically "
            "against NGSolve 6.2.2604 2026-06-01.)",
            "[Numerical] PML setup uses mesh.SetPML(pml.Radial("
            "origin=(0,0), rad=r, alpha=a_j)) where alpha is IMAGINARY "
            "(complex-valued). A real-only alpha passed to "
            "pml.Radial becomes a lossy real boundary that "
            "reflects the outgoing wave. Signal: Integrate of "
            "GridFunction L2Norm in the bulk region does NOT "
            "monotonically decrease as the radius r of "
            "mesh.SetPML is increased; the post-processed field "
            "shows standing-wave fringes characteristic of "
            "reflection from the inner edge of pml.Radial. "
            "(Catalog claim inherited — not yet empirically "
            "verified on a running PML simulation.)",
            "[Numerical] mesh.SetPML(pml.Radial(rad, alpha)) "
            "alpha magnitude must be tuned: too small → outgoing "
            "wave is partially reflected from the pml.Radial "
            "interface; too large → high local wavenumber inside "
            "the PML makes the BilinearForm.Assemble + "
            "UmfpackInverse linear solve ill-conditioned. "
            "Practical starting range alpha = 1.0..5.0 (imaginary "
            "part). Signal: Integrate of GridFunction L2Norm "
            "outside pml.Radial vs alpha forms a U-curve with "
            "the minimum near alpha=1.0..5.0; outside that range "
            "the inner-boundary GridFunction value is O(0.1) of "
            "the source. (Catalog claim inherited — not yet "
            "empirically verified.)",
            "[Numerical] Resolution rule of thumb: ~10 DOFs per "
            "wavelength minimum, i.e. order p, h < lambda/(2p). "
            "Insufficient resolution makes the Helmholtz pollution "
            "effect dominate. Signal: phase error of the "
            "post-processed solution grows as O(k^(p+1) h^(2p+1)); "
            "for k > 100, h < lambda/10 is not enough at p=1, and "
            "the L2 error against an analytic plane wave plateaus "
            "around 10% regardless of further refinement. (Catalog "
            "claim inherited — not yet empirically verified at this "
            "magnitude.)",
            "[API] For eigenvalues / cavity resonances: use "
            "ArnoldiSolver with shift-invert; the shift should be "
            "near the expected eigenvalue (k^2_estimate from "
            "analytic cavity formula); a PEC cavity's resonances "
            "are k^2 = pi^2 (m^2 + n^2 + p^2) with at most one "
            "index zero, so the lowest is doubly degenerate and "
            "the gradient modes come back as exact zeros. "
            "shift=0 raises NgException 'UmfpackInverse: Numeric "
            "factorization failed.' on the curl-curl operator "
            "(same family as maxwell#5), while a shift well away "
            "from zero completes — so the failure is specific to "
            "shift=0, not to a badly chosen shift. "
            "Signal: guard on the EXCEPTION. The companion line "
            "'UMFPACK V5.7.4 (Feb 1, 2016): WARNING: matrix is "
            "singular' is written by UMFPACK through C stdio, "
            "which by default holds it until PROCESS EXIT when "
            "stdout is a file or a pipe, so a capture placed "
            "around the ArnoldiSolver call comes back EMPTY (an "
            "fd-level os.dup2 of fd 1 as well as "
            "contextlib.redirect_stdout) and an agent guarding "
            "on that text concludes nothing happened; only a "
            "line-buffered stdout (a terminal, or stdbuf -oL) "
            "shows it during the call. Catch NgException; if you want "
            "corroboration, check that the returned lowest "
            "non-zero resonance is within a small percentage of "
            "the closed-form k^2 and that the degenerate pair "
            "comes back as two nearly equal values rather than "
            "collapsing to one. (Verified empirically 2026-06-01, "
            "signal corrected 2026-08-06 on NGSolve 6.2.2604 — "
            "the UMFPACK warning text is not capturable around "
            "the call.)",
        ],
    },
}

GENERATORS = {
    "helmholtz_2d": _helmholtz_2d,
}
