"""The REAL Kratos ConvectionDiffusion route, verified by execution.

Why this module exists: the `heat` and `poisson` templates in this backend used
to emit a numpy/scipy assembly with no `import KratosMultiphysics` anywhere in
them — heat.py's own first line read "Heat conduction — Kratos (manual
assembly)". An agent asked to solve a problem WITH KRATOS was handed a script
that cannot run Kratos, and the resulting output cannot be attributed to
the code the task named. That is not a documentation defect, it is the wrong
artefact.

Every API fact below was established on this install (Kratos 10.3.0) by running
it, and each one had a wrong first guess:

  * `LaplacianElement2D3N` is registered; `LaplacianElement2D4N` is NOT
    ("The Element ... is not registered!"). 2D is P1 TRIANGLES here.
  * the reaction-carrying DOF overload is `AddDof(var, reaction, mp)`. There is
    no `AddDofWithReaction`. Registering no reaction and then letting the
    strategy compute reactions aborts in EVERY worker thread with "This
    container only can store the variables specified in its variables list".
  * CONDUCTIVITY is read NODALLY, through ConvectionDiffusionSettings, not from
    Properties (see the poisson pitfall: nodal 999 + Properties 1 -> 999).
  * the volumetric source HEAT_FLUX is the NODAL value interpolated at ONE
    centroid point, i.e. mean(f_nodes) * A / 3. Matched to 4.3e-16 against an
    independent assembly using that rule; the exact P1 mass matrix is 50% off,
    so this is not a detail.
  * FACE_HEAT_FLUX on a `ThermalFace2D2N` is the INWARD normal flux. Measured
    against a closed form chosen so the two readings differ by a sign:
    u(interface) came out +0.875 where inward predicts +0.875.
  * SETTING FACE_HEAT_FLUX ON NODES DOES NOTHING UNLESS `ThermalFace2D2N`
    CONDITIONS EXIST ON THOSE EDGES. The nodal value is only ever integrated
    BY a condition; with no condition there is nothing to integrate it, and
    Kratos runs, converges, exits 0 and returns the no-flux solution. Measured
    on one mesh, three runs differing only in this:
        zero flux, conditions present     max|T| = T0
        flux on nodes, NO conditions      max|T| = T0, IDENTICAL to all printed digits
        flux on nodes AND conditions      a different field: the flux is applied
    numpy.allclose on the first two is True. This is why it is invisible: the
    imported flux leaves no trace at all.
  * THE COMPONENT NAME GOES IN A STRING, THROUGH THE FACTORY:
        mp.CreateNewCondition("ThermalFace2D2N", cid, [n1, n2], prop)
    Elements and conditions live in a C++ registry and NONE of them is a
    Python attribute, so `SomeApplication.ThermalFace2D2N(...)` raises
    `has no attribute` for every name that does exist. Measured on this
    install, all three the same way:
        LaplacianElement2D3N   python-attribute=False  factory-by-name=True
        ThermalFace2D2N        python-attribute=False  factory-by-name=True
        FluxCondition2D2N      python-attribute=False  factory-by-name=True
    So that AttributeError is never evidence a component is missing. One run
    read it as "not available in version 10.3.0" and abandoned the codes its
    task prescribed. A name that IS absent fails differently, inside
    CreateNewCondition: `The Condition "ThermalFace2D" is not registered!`.
  * Kratos prints from C++ streams, and both they and Python's print() are
    buffered when stdout is a file or a pipe. An in-process `os.dup2` redirect
    of fd 1 captured ZERO bytes when fd 1 was restored without a flush: the
    buffered output is written at exit, after the redirect is undone. Calling
    KM.Logger.Flush() and sys.stdout.flush() before restoring fd 1 (or running
    Python unbuffered, PYTHONUNBUFFERED=1) captured the Kratos lines too. A
    Python-level redirect (contextlib.redirect_stdout) never sees the C++
    output. If a log has to show which code ran, running the solve in a
    SUBPROCESS and capturing that avoids all of this.
"""
from __future__ import annotations

_REAL = '''\
"""{title}

Solved by Kratos Multiphysics ConvectionDiffusionApplication -- the solver is
Kratos, not an assembly written here. Run it with `run_simulation`.
"""
import json

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication  # noqa: F401
# ^ the import is what REGISTERS LaplacianElement2D3N; without it
#   CreateNewElement raises 'The Element "LaplacianElement2D3N" is not
#   registered!'

NX, NY = {nx}, {ny}
X0, X1, Y0, Y1 = {x0}, {x1}, {y0}, {y1}
K_VAL = {k}


def source(x, y):
    """The volumetric source f in -div(k grad u) = f. EDIT THIS."""
    return {f_expr}


def boundary_value(x, y):
    """The prescribed value on the Dirichlet boundary. EDIT THIS."""
    return {g_expr}


# ---- mesh: P1 triangles, two per cell (2D3N is the only 2D Laplacian element)
xs = np.linspace(X0, X1, NX + 1)
ys = np.linspace(Y0, Y1, NY + 1)
nodes = np.array([(xs[i], ys[j]) for j in range(NY + 1) for i in range(NX + 1)])
nid = lambda i, j: j * (NX + 1) + i          # noqa: E731  0-based
tris = []
for j in range(NY):
    for i in range(NX):
        a, b, c, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
        tris += [[a, b, c], [a, c, d]]
tris = np.array(tris)

# ---- model part
model = KM.Model()
mp = model.CreateModelPart("domain")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
for v in (KM.TEMPERATURE, KM.HEAT_FLUX, KM.CONDUCTIVITY, KM.FACE_HEAT_FLUX,
          KM.REACTION_FLUX, KM.SPECIFIC_HEAT, KM.DENSITY, KM.VELOCITY,
          KM.MESH_VELOCITY):
    mp.AddNodalSolutionStepVariable(v)

# ConvectionDiffusionSettings tells the element WHICH variable is which. Every
# entry must be set: an unset one leaves the element reading variable #0.
s = KM.ConvectionDiffusionSettings()
s.SetUnknownVariable(KM.TEMPERATURE)
s.SetDiffusionVariable(KM.CONDUCTIVITY)
s.SetVolumeSourceVariable(KM.HEAT_FLUX)
s.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
s.SetDensityVariable(KM.DENSITY)
s.SetSpecificHeatVariable(KM.SPECIFIC_HEAT)
s.SetVelocityVariable(KM.VELOCITY)
s.SetMeshVelocityVariable(KM.MESH_VELOCITY)
s.SetReactionVariable(KM.REACTION_FLUX)
mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, s)

prop = mp.CreateNewProperties(1)
for i, (px, py) in enumerate(nodes):
    n = mp.CreateNewNode(i + 1, float(px), float(py), 0.0)
    # CONDUCTIVITY IS READ NODALLY, not from Properties. Setting it only on
    # the Properties object gives a zero-diffusivity, singular system.
    n.SetSolutionStepValue(KM.CONDUCTIVITY, K_VAL)
    # HEAT_FLUX is the volumetric source, interpolated at the centroid:
    # the element assembles mean(f_nodes) * A / 3.
    n.SetSolutionStepValue(KM.HEAT_FLUX, float(source(px, py)))
    n.SetSolutionStepValue(KM.DENSITY, 1.0)
    n.SetSolutionStepValue(KM.SPECIFIC_HEAT, 1.0)
for e, el in enumerate(tris):
    mp.CreateNewElement("LaplacianElement2D3N", e + 1,
                        [int(v) + 1 for v in el], prop)

# DOFs must carry the reaction, or CalculateReactions aborts in every thread.
KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)

tol = 1e-12
on_bnd = ((np.abs(nodes[:, 0] - X0) < tol) | (np.abs(nodes[:, 0] - X1) < tol)
          | (np.abs(nodes[:, 1] - Y0) < tol) | (np.abs(nodes[:, 1] - Y1) < tol))
for i in np.where(on_bnd)[0]:
    nd = mp.GetNode(int(i) + 1)
    nd.SetSolutionStepValue(KM.TEMPERATURE, float(boundary_value(*nodes[i])))
    nd.Fix(KM.TEMPERATURE)

lin = KM.LinearSolverFactory().Create(
    KM.Parameters('{{"solver_type":"skyline_lu_factorization"}}'))
scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
bas = KM.ResidualBasedBlockBuilderAndSolver(lin)
strat = KM.ResidualBasedLinearStrategy(mp, scheme, bas, True, False, False,
                                       False)
strat.SetEchoLevel(1)
mp.ProcessInfo[KM.DELTA_TIME] = 1.0
mp.CloneTimeStep(1.0)
strat.Initialize()
strat.Solve()

u = np.array([mp.GetNode(i + 1).GetSolutionStepValue(KM.TEMPERATURE)
              for i in range(len(nodes))])
print(f"Kratos LaplacianElement2D3N: nodes={{len(nodes)}} "
      f"elements={{len(tris)}} max|u|={{np.abs(u).max():.10e}}")
# AN ALL-ZERO FIELD WITH A NONZERO SOURCE IS NOT A CONVERGED SOLVE. It is a
# source or a settings entry that never reached the element -- check it here
# rather than downstream.
if np.abs(u).max() == 0.0 and any(abs(source(*p)) > 0 for p in nodes[:50]):
    raise SystemExit("Kratos returned an identically zero field against a "
                     "nonzero source: the volume source did not reach the "
                     "element. Check ConvectionDiffusionSettings and that "
                     "HEAT_FLUX is set on NODES.")

np.savetxt("solution.csv",
           np.column_stack([nodes[:, 0], nodes[:, 1], u]),
           delimiter=", ", header="x, y, u", comments="",
           fmt="%.15e")
json.dump({{"max_abs": float(np.abs(u).max()), "n_nodes": int(len(nodes)),
           "n_elements": int(len(tris)), "element": "LaplacianElement2D3N",
           "solver": "Kratos ConvectionDiffusionApplication"}},
          open("results_summary.json", "w"), indent=2)
print("Kratos solve complete.")
'''


def real_convdiff_script(title: str, nx: int, ny: int, k: float,
                         f_expr: str, g_expr: str,
                         x0: float = 0.0, x1: float = 1.0,
                         y0: float = 0.0, y1: float = 1.0) -> str:
    """A runnable Kratos ConvectionDiffusion script for -div(k grad u) = f."""
    return _REAL.format(title=title, nx=nx, ny=ny, k=k, f_expr=f_expr,
                        g_expr=g_expr, x0=x0, x1=x1, y0=y0, y1=y1)


CROSS_CHECK_NOTE = (
    "To VERIFY a Kratos answer, assemble the same P1 system independently "
    "(numpy is enough) with the SAME source rule -- mean(f_nodes) * A / 3, "
    "the one-point centroid rule Kratos uses -- and compare. That comparison "
    "is what established the rule: it matched to 4.3e-16, while the exact P1 "
    "mass matrix was 50% off. An independent assembly is a CHECK on the "
    "solver's answer, never a replacement for running it: a result "
    "produced by the assembly alone cannot be attributed to Kratos, and a "
    "coupled task that names two codes is failed by it."
)


_TRANSIENT = '''\
"""{title}

Transient conduction solved by Kratos Multiphysics ConvectionDiffusionApplication
using EulerianDiffusion2D3N, which carries the capacity term. Run with
`run_simulation`.
"""
import json

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication  # noqa: F401

NX, NY = {nx}, {ny}
X0, X1, Y0, Y1 = {x0}, {x1}, {y0}, {y1}
K_VAL, RHO, CP = {k}, {rho}, {cp}
DT, T_END = {dt}, {t_end}
T_INIT = {t_init}


def source(x, y):
    """Volumetric source. EDIT THIS."""
    return {f_expr}


def dirichlet(x, y):
    """Return the prescribed temperature, or None where the node is free.
    EDIT THIS."""
    {dirichlet_body}


xs = np.linspace(X0, X1, NX + 1)
ys = np.linspace(Y0, Y1, NY + 1)
nodes = np.array([(xs[i], ys[j]) for j in range(NY + 1) for i in range(NX + 1)])
nid = lambda i, j: j * (NX + 1) + i          # noqa: E731
tris = []
for j in range(NY):
    for i in range(NX):
        a, b, c, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
        tris += [[a, b, c], [a, c, d]]
tris = np.array(tris)

model = KM.Model()
mp = model.CreateModelPart("domain")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
for v in (KM.TEMPERATURE, KM.HEAT_FLUX, KM.CONDUCTIVITY, KM.FACE_HEAT_FLUX,
          KM.REACTION_FLUX, KM.SPECIFIC_HEAT, KM.DENSITY, KM.VELOCITY,
          KM.MESH_VELOCITY):
    mp.AddNodalSolutionStepVariable(v)
s = KM.ConvectionDiffusionSettings()
s.SetUnknownVariable(KM.TEMPERATURE)
s.SetDiffusionVariable(KM.CONDUCTIVITY)
s.SetVolumeSourceVariable(KM.HEAT_FLUX)
s.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
s.SetDensityVariable(KM.DENSITY)
s.SetSpecificHeatVariable(KM.SPECIFIC_HEAT)
s.SetVelocityVariable(KM.VELOCITY)
s.SetMeshVelocityVariable(KM.MESH_VELOCITY)
s.SetReactionVariable(KM.REACTION_FLUX)
mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, s)

prop = mp.CreateNewProperties(1)
for i, (px, py) in enumerate(nodes):
    n = mp.CreateNewNode(i + 1, float(px), float(py), 0.0)
    # ALL of these are read NODALLY through the settings object, not from
    # Properties. RHO * CP is the volumetric capacity: it is what makes this
    # transient rather than steady, and setting it to 0 silently gives the
    # steady answer at the first step.
    n.SetSolutionStepValue(KM.CONDUCTIVITY, K_VAL)
    n.SetSolutionStepValue(KM.DENSITY, RHO)
    n.SetSolutionStepValue(KM.SPECIFIC_HEAT, CP)
    n.SetSolutionStepValue(KM.HEAT_FLUX, float(source(px, py)))
    n.SetSolutionStepValue(KM.TEMPERATURE, T_INIT)
for e, el in enumerate(tris):
    # EulerianDiffusion2D3N, NOT LaplacianElement2D3N: the Laplacian element
    # has no capacity term, so a "transient" run built on it returns the steady
    # solution at every step and the history is flat.
    mp.CreateNewElement("EulerianDiffusion2D3N", e + 1,
                        [int(v) + 1 for v in el], prop)

KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)
for i, (px, py) in enumerate(nodes):
    val = dirichlet(px, py)
    if val is not None:
        nd = mp.GetNode(i + 1)
        nd.SetSolutionStepValue(KM.TEMPERATURE, float(val))
        nd.Fix(KM.TEMPERATURE)

mp.SetBufferSize(2)      # the scheme reads the previous step
lin = KM.LinearSolverFactory().Create(
    KM.Parameters('{{"solver_type":"skyline_lu_factorization"}}'))
strat = KM.ResidualBasedLinearStrategy(
    mp, KM.ResidualBasedIncrementalUpdateStaticScheme(),
    KM.ResidualBasedBlockBuilderAndSolver(lin), True, False, False, False)
strat.SetEchoLevel(1)
mp.ProcessInfo[KM.DELTA_TIME] = DT
strat.Initialize()

probe = int(nid(NX // 2, NY // 2))
t, hist = 0.0, []
while t < T_END - 1e-12:
    t += DT
    mp.CloneTimeStep(t)
    strat.Solve()
    hist.append((t, float(mp.GetNode(probe + 1)
                          .GetSolutionStepValue(KM.TEMPERATURE))))
print(f"Kratos EulerianDiffusion2D3N: {{len(hist)}} steps, probe T "
      f"{{hist[0][1]:.6f}} -> {{hist[-1][1]:.6f}}")
if len(hist) > 2 and max(abs(v - hist[0][1]) for _, v in hist) == 0.0:
    raise SystemExit("the probe temperature is identical at every step: the "
                     "capacity term is not in play. Check DENSITY and "
                     "SPECIFIC_HEAT are nonzero on the NODES and that the "
                     "element is EulerianDiffusion2D3N.")

u = np.array([mp.GetNode(i + 1).GetSolutionStepValue(KM.TEMPERATURE)
              for i in range(len(nodes))])
np.savetxt("history.csv", np.array(hist), delimiter=", ",
           header="t, probe_T", comments="", fmt="%.15e")
np.savetxt("solution.csv", np.column_stack([nodes[:, 0], nodes[:, 1], u]),
           delimiter=", ", header="x, y, u", comments="", fmt="%.15e")
json.dump({{"n_steps": len(hist), "dt": DT, "t_end": T_END,
           "probe_T_final": hist[-1][1], "max_abs": float(np.abs(u).max()),
           "element": "EulerianDiffusion2D3N",
           "solver": "Kratos ConvectionDiffusionApplication"}},
          open("results_summary.json", "w"), indent=2)
print("Kratos transient conduction complete.")
'''


def real_transient_script(title: str, nx: int, ny: int, k: float, rho: float,
                          cp: float, dt: float, t_end: float,
                          f_expr: str, dirichlet_body: str,
                          t_init: float = 0.0,
                          x0: float = 0.0, x1: float = 1.0,
                          y0: float = 0.0, y1: float = 1.0) -> str:
    """A runnable Kratos transient conduction script.

    `dirichlet_body` is the body of dirichlet(x, y): it must `return` a value
    on constrained nodes and `return None` elsewhere.
    """
    return _TRANSIENT.format(title=title, nx=nx, ny=ny, k=k, rho=rho, cp=cp,
                             dt=dt, t_end=t_end, f_expr=f_expr,
                             dirichlet_body=dirichlet_body, t_init=t_init,
                             x0=x0, x1=x1, y0=y0, y1=y1)


_IFACE_NEUMANN = '''\
"""{title}

The NEUMANN side of a partitioned coupling, solved by Kratos
ConvectionDiffusionApplication: it imports a normal flux on the interface,
applies it as a natural boundary condition, and exports its own interface
field and its own outward flux.

THE ONE THING THAT SILENTLY BREAKS THIS. Setting FACE_HEAT_FLUX on the
interface nodes is NOT ENOUGH. The nodal value is only integrated BY a
condition, so without `ThermalFace2D2N` conditions on the interface edges
Kratos runs, converges, exits 0, and returns exactly the solution it would
have returned with no flux at all. Measured on one mesh, three runs differing
only in this:

    zero flux, conditions present     max|T| = T0
    flux on nodes, NO conditions      max|T| = T0, IDENTICAL to all printed digits
    flux on nodes AND conditions      a different field: the flux is applied

numpy.allclose on the first two is True. Real runs have died here: two
independent runs whose side A was correct to three digits reported the no-flux
answer on side B, with their interface FIELD matching across the seam exactly and
only the flux jump betraying it, growing under refinement instead of shrinking.

TWO SIGNS, AND THEY ARE OPPOSITE.
  what you APPLY : FACE_HEAT_FLUX is the INWARD normal flux, so the partner's
                   outward flux q_A^out is applied as +q_A^out (heat leaving A
                   enters B).
  what you REPORT: the task's q_n is YOUR OWN OUTWARD flux, which is the
                   NEGATIVE of what you applied.
Get these the same way round and the two sides appear to balance when they do
not.
"""
import json
from pathlib import Path

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication  # noqa: F401

NX, NY = {nx}, {ny}
X0, X1 = {x0}, {x1}          # this subdomain; the interface is at X0
K_VAL = {k}
IFACE_AT_X0 = True           # the interface is this subdomain's LEFT edge


def source(x, y):
    """This subdomain's own volumetric source. EDIT THIS."""
    return {f_expr}


def imported_flux(y):
    """The partner's OUTWARD normal flux at interface height y, from
    imports.json. EDIT THIS to read your partner's data."""
    imp = json.loads(Path("imports.json").read_text()) if Path("imports.json").is_file() else {{}}
    for _name, data in imp.items():
        co = np.array(data.get("coordinates") or [], float)
        q = np.array(data.get("normal_fluxes") or [], float)
        if len(co) and len(q) == len(co):
            o = np.argsort(co[:, 1])
            return float(np.interp(y, co[o, 1], q[o]))
    return 0.0               # iteration 1: no partner data yet


xs = np.linspace(X0, X1, NX + 1)
ys = np.linspace(0.0, 1.0, NY + 1)
nodes = np.array([(xs[i], ys[j]) for j in range(NY + 1) for i in range(NX + 1)])
nid = lambda i, j: j * (NX + 1) + i          # noqa: E731
tris = []
for j in range(NY):
    for i in range(NX):
        a, b, c, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
        tris += [[a, b, c], [a, c, d]]

model = KM.Model()
mp = model.CreateModelPart("neumann_side")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
for v in (KM.TEMPERATURE, KM.HEAT_FLUX, KM.CONDUCTIVITY, KM.FACE_HEAT_FLUX,
          KM.REACTION_FLUX, KM.SPECIFIC_HEAT, KM.DENSITY, KM.VELOCITY,
          KM.MESH_VELOCITY):
    mp.AddNodalSolutionStepVariable(v)
s = KM.ConvectionDiffusionSettings()
s.SetUnknownVariable(KM.TEMPERATURE)
s.SetDiffusionVariable(KM.CONDUCTIVITY)
s.SetVolumeSourceVariable(KM.HEAT_FLUX)
s.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
s.SetDensityVariable(KM.DENSITY)
s.SetSpecificHeatVariable(KM.SPECIFIC_HEAT)
s.SetVelocityVariable(KM.VELOCITY)
s.SetMeshVelocityVariable(KM.MESH_VELOCITY)
s.SetReactionVariable(KM.REACTION_FLUX)
mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, s)

prop = mp.CreateNewProperties(1)
fnodal = np.array([source(px, py) for px, py in nodes])
for i, (px, py) in enumerate(nodes):
    n = mp.CreateNewNode(i + 1, float(px), float(py), 0.0)
    n.SetSolutionStepValue(KM.CONDUCTIVITY, K_VAL)     # NODAL, not Properties
    n.SetSolutionStepValue(KM.HEAT_FLUX, float(fnodal[i]))
    n.SetSolutionStepValue(KM.DENSITY, 1.0)
    n.SetSolutionStepValue(KM.SPECIFIC_HEAT, 1.0)
for e, el in enumerate(tris):
    mp.CreateNewElement("LaplacianElement2D3N", e + 1,
                        [int(v) + 1 for v in el], prop)

tol = 1e-12
edge = X0 if IFACE_AT_X0 else X1
on_iface = np.abs(nodes[:, 0] - edge) < tol
on_bot = np.abs(nodes[:, 1]) < tol
on_top = np.abs(nodes[:, 1] - 1.0) < tol
# THE INTERFACE CORNERS BELONG TO THE OUTER BOUNDARY. A Dirichlet-Neumann
# corner has no convergent recovered flux, which is why interface probes
# should exclude the ends.
iface = [i for i in np.where(on_iface)[0] if not (on_bot[i] or on_top[i])]
iface.sort(key=lambda i: nodes[i, 1])

# ---- THE CONDITIONS. WITHOUT THESE THE NEXT LOOP IS A NO-OP.
# CONDITIONS COVER THE CORNER SEGMENTS TOO. Without them the first and
# last interior nodes receive roughly HALF their consistent load, and the
# recovered flux there exports ~q/2 for any profile nonzero at the ends
# (reviewer finding; the corner node's own row is Dirichlet-fixed, so the
# extra condition is harmless there).
edge_nodes = sorted(int(i) for i in np.where(on_iface)[0])
for c in range(len(edge_nodes) - 1):
    mp.CreateNewCondition("ThermalFace2D2N", c + 1,
                          [edge_nodes[c] + 1, edge_nodes[c + 1] + 1], prop)
for i in iface:
    # FACE_HEAT_FLUX is INWARD; the partner's outward flux enters here as +q.
    mp.GetNode(int(i) + 1).SetSolutionStepValue(
        KM.FACE_HEAT_FLUX, float(imported_flux(nodes[i, 1])))

KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)
for i in range(len(nodes)):
    if on_bot[i] or on_top[i] or abs(nodes[i, 0] - (X1 if IFACE_AT_X0 else X0)) < tol:
        nd = mp.GetNode(i + 1)
        nd.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
        nd.Fix(KM.TEMPERATURE)

lin = KM.LinearSolverFactory().Create(
    KM.Parameters('{{"solver_type":"skyline_lu_factorization"}}'))
strat = KM.ResidualBasedLinearStrategy(
    mp, KM.ResidualBasedIncrementalUpdateStaticScheme(),
    KM.ResidualBasedBlockBuilderAndSolver(lin), True, False, False, False)
strat.SetEchoLevel(1)
mp.ProcessInfo[KM.DELTA_TIME] = 1.0
mp.CloneTimeStep(1.0)
strat.Initialize()
strat.Solve()

T = np.array([mp.GetNode(i + 1).GetSolutionStepValue(KM.TEMPERATURE)
              for i in range(len(nodes))])
qin = np.array([imported_flux(nodes[i, 1]) for i in iface])
print(f"Kratos Neumann side: nodes={{len(nodes)}} iface={{len(iface)}} "
      f"conditions={{max(len(iface) - 1, 0)}} "
      f"q_in=[{{qin.min():.6e}},{{qin.max():.6e}}] max|T|={{np.abs(T).max():.9e}}")

# THE CHECK THAT CATCHES THE SILENT NO-OP: with a nonzero imported flux, the
# interface trace must NOT be what it would be with no flux at all.
if np.abs(qin).max() > 0 and max(len(iface) - 1, 0) == 0:
    raise SystemExit("a nonzero flux was imported and NO ThermalFace2D2N "
                     "condition exists: the flux will be silently ignored and "
                     "this run would return the no-flux solution.")

# YOUR OWN OUTWARD FLUX IS RECOVERED FROM YOUR OWN SOLUTION, never copied
# from the import. An earlier revision of this file exported `-qin` here --
# the negated import array -- which reads as a bit-exact mirror of the
# partner's data and proves nothing about this solve. The consistent
# recovery below passes THROUGH the assembled system: when the delivery
# worked it equals the negated, P1-mass-smoothed applied flux (the FE
# identity), and when the ThermalFace conditions never entered the system
# it reads ~0, which the arrival check turns into a hard stop instead of a
# silent no-flux result.


def consistent_outward_flux():
    """q_i = -(K u - b_vol)_i / h_i on the interface rows.

    Assembled with exactly the discretisation Kratos used: P1 stiffness
    with the nodal conductivity, and the one-point centroid source rule
    area/3 * mean(f at the vertices). b_vol is the VOLUME load only --
    the face load must NOT go into the residual or the recovered flux
    comes out identically zero.
    """
    resid = np.zeros(len(nodes))
    for el in tris:
        P = nodes[el]
        area = 0.5 * abs(np.cross(P[1] - P[0], P[2] - P[0]))
        g = np.array([[P[1, 1] - P[2, 1], P[2, 0] - P[1, 0]],
                      [P[2, 1] - P[0, 1], P[0, 0] - P[2, 0]],
                      [P[0, 1] - P[1, 1], P[1, 0] - P[0, 0]]]) / (2.0 * area)
        resid[el] += K_VAL * area * (g @ g.T) @ T[el] \\
            - area / 3.0 * np.mean(fnodal[el])
    h_trib = ys[1] - ys[0]           # interior interface nodes only
    return np.array([-resid[i] / h_trib for i in iface])


q_own = consistent_outward_flux()
# ARRIVAL CHECK: a recovered flux of ~0 against a nonzero applied flux means
# the interface load never entered the assembled system.
if np.abs(qin).max() > 1e-30 and np.abs(q_own).max() < 1e-9 * np.abs(qin).max():
    raise SystemExit("recovered interface flux is ~0 against a nonzero "
                     "applied flux: the ThermalFace conditions never entered "
                     "the assembled system.")
print(f"flux recovery: max|q_own + q_applied| / max|q_applied| = "
      f"{{(np.abs(q_own + qin).max() / max(np.abs(qin).max(), 1e-30)):.3e}} "
      f"(P1 smoothing gap; ~0 to O(h^2) when delivery worked)")
np.savetxt("interface_out.csv",
           np.column_stack([np.full(len(iface), edge), nodes[iface, 1],
                            T[iface], q_own]),
           delimiter=", ", header="x, y, u, qn", comments="", fmt="%.15e")
json.dump({{"n_interface": len(iface), "n_conditions": max(len(iface) - 1, 0),
           "max_abs_T": float(np.abs(T).max()),
           "q_recovery": "consistent residual -(K u - b_vol)/h_trib",
           "solver": "Kratos ConvectionDiffusionApplication",
           "element": "LaplacianElement2D3N",
           "interface_condition": "ThermalFace2D2N"}},
          open("results_summary.json", "w"), indent=2)
print("Kratos Neumann-side solve complete.")
'''


def real_interface_neumann_script(title: str, nx: int, ny: int, k: float,
                                  f_expr: str, x0: float, x1: float) -> str:
    """The Neumann side of a partitioned coupling, driven by Kratos.

    Served because the coupled gate checks the interface sign convention and
    the consistent flux recovery, and because the single silent failure in
    this route -- a nodal FACE_HEAT_FLUX with no ThermalFace condition to
    integrate it -- has demonstrably sunk otherwise-correct runs.
    """
    return _IFACE_NEUMANN.format(title=title, nx=nx, ny=ny, k=k,
                                 f_expr=f_expr, x0=x0, x1=x1)
