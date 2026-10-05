"""NGSolve heat conduction generators and knowledge."""


def _heat_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Heat conduction on [0,1]², T_left -> T_right."""
    nx = params.get("nx", 32)
    T_left = params.get("T_left", 100.0)
    T_right = params.get("T_right", 0.0)
    maxh = 1.0 / nx
    return f'''\
"""Heat conduction on [0,1]² — NGSolve"""
from ngsolve import *
import json

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))

fes = H1(mesh, order=1, dirichlet="left|right")
u, v = fes.TnT()

a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
# Zero RHS for pure Dirichlet heat: any explicit
# 'zero*v*dx' integrand (including CoefficientFunction(0))
# collapses symbolically before LinearForm sees it and
# triggers NGSolve's NgException 'Linearform must have
# TestFunction'. Construct the LinearForm on the FESpace
# with NO integrand instead — Assemble() then gives a
# zero RHS vector.
f = LinearForm(fes)
f.Assemble()

gfu = GridFunction(fes)

# Apply Dirichlet BCs. NOTE: gfu.Set() ZEROES the whole vector first, so calling
# it twice (once per boundary) wipes the first BC and silently yields a wrong
# (often all-zero) field. Set BOTH boundary values in ONE call via a
# boundary-piecewise CoefficientFunction over the union boundary.
gfu.Set(mesh.BoundaryCF({{"left": {T_left}, "right": {T_right}}}),
        definedon=mesh.Boundaries("left|right"))

# Modify RHS for Dirichlet
f.vec.data -= a.mat * gfu.vec
gfu.vec.data += a.mat.Inverse(fes.FreeDofs()) * f.vec

T_max = max(gfu.vec)
print(f"Temperature: max={{T_max:.6f}}")

vtk = VTKOutput(mesh, coefs=[gfu], names=["temperature"],
                filename="result", subdivision=0)
vtk.Do()

summary = {{
    "max_value": float(T_max),
    "n_dofs": fes.ndof,
    "n_elements": mesh.ne,
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Heat solve complete.")
'''


def _heat_transient_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Transient heat equation with implicit Euler time-stepping."""
    nx = params.get("nx", 32)
    dt = params.get("dt", 0.001)
    T_end = params.get("T_end", 0.1)
    maxh = 1.0 / nx
    return f'''\
"""Transient heat equation on [0,1]² — implicit Euler — NGSolve"""
from ngsolve import *
import json

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))
fes = H1(mesh, order=3, dirichlet="bottom|right|top|left")
u, v = fes.TnT()

# Mass and stiffness matrices
# (Previously had nonsym=True kwarg; current NGSolve flags that as
# undocumented + has no observable effect. Drop it.)
a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
m = BilinearForm(u*v*dx).Assemble()

# Combined: M + dt*A
dt = {dt}
mstar = m.mat.CreateMatrix()
mstar.AsVector().data = m.mat.AsVector() + dt * a.mat.AsVector()
inv_mstar = mstar.Inverse(fes.FreeDofs())

# Source term
f = LinearForm(1*v*dx).Assemble()

# Initial condition: u=0
gfu = GridFunction(fes)
gfu.Set(0)

# Time stepping
vtk = VTKOutput(mesh, coefs=[gfu], names=["temperature"], filename="result", subdivision=0)
t = 0.0
step = 0
while t < {T_end} - 1e-12:
    res = dt * f.vec - dt * a.mat * gfu.vec
    gfu.vec.data += inv_mstar * res
    t += dt
    step += 1

vtk.Do()
max_val = max(gfu.vec)
print(f"t={{t:.4f}}, max(T) = {{max_val:.10f}}, steps={{step}}")
summary = {{"max_value": float(max_val), "n_dofs": fes.ndof, "time": t, "steps": step}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


KNOWLEDGE = {
    "heat": {
        "description": "Heat conduction: steady and transient (implicit Euler, Crank-Nicolson)",
        "spaces": "H1 (any order)",
        "solver": "Steady: direct. Transient: M+dt*A factored once, reused each step",
        "pitfalls": [
            "[API] The prior catalog wording said 'Transient: "
            "need nonsym=True for mass matrix to get compatible "
            "sparsity pattern' — that is WRONG in current NGSolve. "
            "Passing nonsym=True to the BilinearForm constructor "
            "emits a warning: \"WARNING: kwarg 'nonsym' is an "
            "undocumented flags option for class <class "
            "'ngsolve.comp.BilinearForm'>, maybe there is a "
            "typo?\" (the text NGSolve 6.2.2604 prints, measured "
            "2026-10-01), the kwarg is silently dropped, and the "
            "resulting matrix sparsity is identical to the "
            "default. The compatible-sparsity claim never had any "
            "empirical basis. Signal: BilinearForm(..., "
            "nonsym=True).Assemble() emits the warning text on "
            "stderr; m.mat.AsVector().size equals the default "
            "build (34 == 34 on unit_square maxh=0.5, H1 order 1). "
            "The warning goes to stderr immediately, when "
            "BilinearForm(..., nonsym=True) is constructed "
            "(before Assemble(), not at interpreter shutdown). "
            "CAVEAT: assigning any non-negative value to "
            "ngsglobals.msg_level suppresses it entirely (0 "
            "switches logging off, >0 sets it to error level) — "
            "measured 2026-08-03, the warning appears when "
            "msg_level is never assigned (its default value is "
            "already 0) and is absent after "
            "`ngsglobals.msg_level = L` for every L tried "
            "(0, 1, 2, 3, 5, 10); a negative assignment (e.g. -1) "
            "leaves it visible. Re-assigning the default 0 "
            "already silences it, and raising msg_level will "
            "NOT bring it back. Generated scripts that set "
            "msg_level to keep logs clean therefore lose the "
            "warning and the kwarg is dropped in complete "
            "silence; never rely on the warning as your only "
            "detector. (Verified empirically 2026-06-01 — drift "
            "correction; msg_level behaviour added 2026-08-03 "
            "and corrected in the 2026-08-03 adversarial "
            "re-audit, which falsified the description that had "
            "the warning gated on msg_level <= 1; the generator "
            "template above "
            "also dropped the kwarg.)",
            "[API] Matrix addition for the implicit-Euler "
            "operator uses BaseMatrix.AsVector() concatenation: "
            "mstar.AsVector().data = m.mat.AsVector() + dt * "
            "a.mat.AsVector(). The .data assignment is needed "
            "because AsVector returns a view, not a copy. The "
            "shortcut mstar = m.mat + dt*a.mat is a SILENT trap, "
            "not an error: it yields a SumMatrix, which DOES have "
            "an .Inverse attribute (so hasattr checks pass and no "
            "AttributeError is raised), but calling "
            "mstar.Inverse(fes.FreeDofs()) prints the C++-level "
            "line 'BaseMatrix::InverseMatrix not available' and "
            "RETURNS None. The failure then travels one step "
            "further than you would expect: the following "
            "'* f.vec' does NOT die on NoneType — it builds a "
            "lazy DynamicVectorExpression and returns "
            "successfully — and the error surfaces only at the "
            "ASSIGNMENT, as "
            "NgException('DynamicBaseScalVec AssignTo not "
            "available'). So the traceback points at the "
            "`gfu.vec.data = ...` line, several statements away "
            "from the shortcut that caused it. Signal: "
            "type(m.mat + dt*a.mat).__name__ == "
            "'SumMatrix'; hasattr(..., 'Inverse') is True; the "
            "Inverse(...) call returns None — test that return "
            "value directly, right where you build the operator, "
            "instead of reading the exception you eventually "
            "get. Build the operator "
            "as mstar = m.mat.CreateMatrix(); "
            "mstar.AsVector().data = m.mat.AsVector() + "
            "dt*a.mat.AsVector(). (Verified empirically "
            "2026-08-03 on NGSolve 6.2.2604 — catalog-drift "
            "correction: the prior 'raises AttributeError' claim "
            "was WRONG and understated the danger.)",
            "[API] mesh.Refine() AUTO-UPDATES both the dependent "
            "FESpace and its GridFunctions in current NGSolve — "
            "fes.Update() / gfu.Update() are still available as "
            "no-arg methods but are no-ops for the size change. "
            "Signal: unit_square maxh=0.5, H1(order=1) -> "
            "fes.ndof == 8 and len(gfu.vec) == 8; after "
            "mesh.Refine() and WITHOUT calling any Update, "
            "len(gfu.vec) already reads 21 and fes.ndof reads 21 "
            "(checked with len(gfu.vec) read FIRST, so it is not "
            "the ndof access that triggers it). The DATA is "
            "carried too, and the old wording had this backwards: "
            "the refine DOES prolongate — the old DOF values come "
            "through entry for entry and the new DOFs are "
            "INTERPOLATED, not left at zero — so a check for "
            "zeros or for a lost field finds nothing and reads as "
            "success. Re-Set or re-solve after every Refine "
            "anyway, but for the right reason: the prolongated "
            "function is the COARSE function re-expressed on the "
            "finer space, so its representation error is "
            "bit-for-bit unchanged by the refinement and only a "
            "fresh solve improves it. Compare the error against "
            "your reference before and after the Refine — if it "
            "has not moved, you are looking at a prolongation, "
            "not a solution. (Verified empirically "
            "2026-08-03 on NGSolve 6.2.2604 — catalog-drift "
            "correction: the prior claim that GridFunction storage "
            "grows 'only after gfu.Update()' was WRONG.)",
            "[Syntax] Non-homogeneous Dirichlet on NGSolve: "
            "two-step pattern. First gfu.Set(boundary_cf, "
            "definedon=mesh.Boundaries(name)) to populate the "
            "boundary DOFs; then solve the residual system on "
            "FreeDofs: gfu.vec.data += a.mat.Inverse("
            "fes.FreeDofs()) * (f.vec - a.mat * gfu.vec). "
            "Skipping the residual step leaves the interior "
            "uncorrected for the non-zero BC. Signal: with the "
            "harmonic exact solution u = x^2 - y^2 on unit_square "
            "maxh=0.3, H1(order=2), the two-step pattern gives L2 "
            "error 1.75e-16 while replacing it with "
            "gfu.vec.data = a.mat.Inverse(fes.FreeDofs())*f.vec "
            "gives 4.22e-01 — the Set() boundary values are "
            "overwritten, silently, with no exception. (Verified "
            "empirically 2026-08-03 on NGSolve 6.2.2604 — same "
            "experiment as ngsolve::poisson pitfall #2.)",
        ],
    },
}

GENERATORS = {
    "heat_2d": _heat_2d,
    "heat_2d_steady": _heat_2d,
    "heat_2d_transient": _heat_transient_2d,
}
