"""FEniCSx (dolfinx) TRANSIENT participant for the openPASO `couple` driver.

Transient conduction  rho_c dT/dt - div(K grad T) = F_SRC(x, y, t)  on ONE
rectangular subdomain of a domain split by a straight interface at x = IFACE_X.
Time integration is the theta-scheme; THETA = 0.5 is Crank-Nicolson and is
SECOND ORDER, which is what makes an order-2 space-time result reachable at all.

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST, exits 0.

╔══════════════════════════════════════════════════════════════════════════╗
║ THE TIME-COUPLING DECISION — THIS FILE IMPLEMENTS A WAVEFORM EXCHANGE.    ║
╚══════════════════════════════════════════════════════════════════════════╝
ONE run of this script marches the WHOLE coupling window T_START -> T_END and
exchanges the ENTIRE interface trace in one exports.json:

    values[i][n]        = T at interface point i at time level t^(n+1)
    normal_fluxes[i][n] = the THETA-AVERAGED outward normal flux density at
                          interface point i over step n  (see THE FLUX below)

both of shape (n_points, N_STEPS). The driver's fixed-point iteration therefore
converges the whole space-TIME interface trace at once — classical Dirichlet-
Neumann WAVEFORM RELAXATION. `values` and `normal_fluxes` carry the same layout
every iteration, which is what the driver requires.

WHY, AND NOT "COUPLE ONCE PER TIME STEP". Coupling once per time step means:
exchange, sub-iterate to convergence, THEN advance. Inside ONE `couple` call
that is not merely inconvenient, it is unreachable, for three checkable reasons:

  1. THE EXCHANGE HAS NO TIME AXIS. The driver rebuilds every payload through
     `InterfaceData.to_dict()`, which emits exactly five keys — field_name,
     n_points, coordinates, values, normal_fluxes. Any extra key you write
     (`"times"`, `"step"`) is dropped on the round trip and the partner never
     sees it. A per-step protocol needs to say WHICH step a payload belongs to,
     and there is nowhere to say it.
  2. A PARTICIPANT WITH HIDDEN TIME STATE BREAKS THE DRIVER'S OWN PROBES. The
     driver runs each participant SEVERAL TIMES ON IDENTICAL imports — once per
     replicate for `noise_replicates`, and again in
     `probe_interface_sensitivity`. A script that advances an internal step
     counter answers the same question differently every time, so the driver
     measures a residual "noise floor" that is really the time march, and the
     sensitivity probe reads a response that is really the next time step.
     Both mis-readings are silent. A waveform participant has no such state:
     measured on the coupled runs below, the probe's repeat-run noise came out
     EXACTLY 0.0 for both participants, so its interface sensitivity (S = 1.0
     and 3.7-5.0) means what it says.
  3. THE DRIVER'S CONVERGENCE TEST IS "THE EXPORT STOPPED CHANGING". A
     participant that advances time whenever it likes satisfies that only by
     accident: one flat time step makes the residual dip, the driver declares
     convergence at step 3 of 20, and the reported answer is at the wrong TIME
     with nothing anywhere saying so. And because the driver is a JACOBI sweep
     (both participants see the PREVIOUS iteration's exports), two participants
     deciding independently when to advance can desynchronize in time.

WHAT PER-TIME-STEP COUPLING WOULD COST IF YOU WANT IT ANYWAY. It is a legitimate
design and it stays inside this contract, but the time loop moves OUT of the
participant and into a script you write around the driver: call `couple(...)`
N_STEPS times; before call n, write the step's t^n / t^(n+1) into a small
control file this script reads, and let the participant restart from the field
dump it writes (see FIELD DUMP below). Every call is then a plain steady-looking
fixed point over one step and every guard above still works. The price:
  * N_STEPS driver invocations instead of one, each paying its own start-up,
    its own sensitivity probe (one extra solve per participant per call) and
    its own noise-floor replicates if you use them;
  * restart files, i.e. real state on disk between calls, so the participant is
    no longer a pure function of imports.json — the property this file relies on;
  * you own the outer loop, so you own its correctness (a per-step call that
    fails to converge must abort the march, not continue).
  What you buy: exchange memory O(n_points) instead of O(n_points x N_STEPS);
  a residual that measures ONE step, so a hard step cannot hide behind easy
  ones; and per-step iteration counts.

WHAT THE WAVEFORM CHOICE COSTS HERE. exports.json holds 2 x n_points x N_STEPS
numbers — measured on the finest level of this file's own verification, 97
points x 80 steps = 15520 numbers and 415 KB of JSON per participant per
iteration — and every driver iteration re-solves the ENTIRE window, so the
total linear-solve count is n_iter x N_STEPS. Measured on the manufactured
two-material problem below, with h and dt halved together over four levels: the
whole window converged to an export residual of 5.1e-10 to 5.8e-10 in 31 driver
iterations AT EVERY LEVEL (37 s, 39 s, 44 s, 79 s of wall clock, so at these
sizes the cost per iteration is still dominated by interpreter start-up). The
waveform iteration count did not move with h or dt, which is what makes this
cost predictable; it does grow with the WINDOW LENGTH, so for a long horizon
split the run into several windows (each one `couple` call, restarting from the
field dump) rather than one enormous window.

WHICH SIDE TAKES THE DIRICHLET ROLE DECIDES WHETHER THIS CONVERGES AT ALL, and
that is a property of the PAIR, not of this file. For two subdomains the
Dirichlet-Neumann iteration contracts like

    rho ~ (K_dirichlet / K_neumann) * (width_neumann / width_dirichlet)

so THE DIRICHLET ROLE BELONGS ON THE SOFTER SIDE: small K, or a wide
subdomain. Measured on the problem below (K = 1.7 across a width of 0.4 against
K = 0.35 across a width of 0.6, so rho ~ 7.3 the wrong way round), with nothing
changed between the runs but the two SIDE strings:
  * Dirichlet on the K = 1.7 side: the driver's residual GREW, from 0.76 at
    iteration 2 to ~1.2, and stalled there — NOT CONVERGED in 40 iterations, at
    all three refinement levels tried. The participants were reported
    `responsive`, the exports were finite, and every level failed identically;
    the only signal was the residual history.
  * Dirichlet on the K = 0.35 side: converged to 5.8e-10 in 31 iterations at
    every level, order 2.00 in space-time.
If the roles are FIXED because one backend can only take one of them, the
remedy is relaxation, not hope: run the driver with accelerator="constant" and
theta0 ~ 1/(1+rho) — measured on the diverging assignment above, theta0 = 0.12
turned it into a converging run, 9.7e-10 in 282 iterations (327 s) against 31
iterations (37 s) for the same problem with the roles the right way round, and
a final-time field error of 1.04e-03 against 1.16e-03, i.e. the same answer for
ten times the work. Aitken did not rescue it on its own: its theta is clamped
to [0.05, 1], and relaxing a Jacobi sweep whose map has complex eigenvalues is
not the scalar sequence Aitken extrapolates.

THE TWO PARTICIPANTS MUST BE GIVEN THE SAME T_START, T_END, N_STEPS AND THETA.
Point 1 above is also a hole: the payload cannot carry the time grid, so a
partner configured with a different window produces a trace this script cannot
tell from a correct one unless its LENGTH differs. The length IS checked (loud
exit). Equal dt is NOT checkable from the payload and is on you.

THE FLUX IS THETA-AVERAGED, AND THAT IS NOT A DETAIL. The theta-scheme's
discrete residual on a constrained row is

    r_i = (M/dt + THETA*A) u^(n+1) - (M/dt - (1-THETA)*A) u^n
          - THETA*F^(n+1) - (1-THETA)*F^n
        = -[ THETA * int_Gamma q^(n+1) phi_i ds
             + (1-THETA) * int_Gamma q^n phi_i ds ]  + O(dt^2)

so the reaction recovers the THETA-WEIGHTED AVERAGE of the flux over the step,
NOT the flux at t^(n+1). That is exactly the quantity the Neumann side needs:
its own theta-scheme right-hand side asks for THETA*N^(n+1) + (1-THETA)*N^n,
i.e. the same average, applied UNCHANGED as `+ int g v ds`. So the two sides
match term for term with no time interpolation anywhere. Reading r_i as "the
flux at t^(n+1)" instead and handing that over is a first-order error in dt
injected straight into an otherwise second-order coupling — it does not look
like a bug, it looks like a scheme that stalled at order 1.
  The same expression run against the load WITHOUT the interface term returns
  that same average on the Neumann side's FREE rows, where it reads back the
  average the partner applied. So neither side blends anything by hand and
  neither carries a flux at t^0 around — see ONE FORMULA, BOTH SIDES, in the
  march.

FIELD DUMP. The contract carries interface data only, so a transient run that
returns nothing but the interface is useless: this script writes
`field_final.npz` (dof coordinates, the final-time nodal field, the nodal
volume weights int phi_i dx, and the time grid) BEFORE exports.json. Anything
written after exports.json is a contract violation — the driver takes that file
appearing as proof the run finished.

  AND THE DUMP LEFT ON DISK AFTER A PROBED RUN IS NOT THE CONVERGED FIELD.
  run_coupling(probe=True) ends by calling probe_interface_sensitivity, which
  re-runs every participant TWICE more — once on the same imports, once with
  every exchanged number nudged by 1e-3 relative — and restores only
  imports.json and exports.json afterwards. Any OTHER file a participant wrote
  is left over from that PERTURBED run. Measured: reading field_final.npz
  straight after a probed coupling put a floor of 2.1e-4 under the field L2
  error, so the measured space-time order read 1.98, then 0.32, then -0.12 as
  the mesh was refined — a converged coupling reporting a stalling solver. The
  exports (and therefore every interface quantity) were correct throughout.
  Either pass probe=False, or re-run each participant ONCE in its work_dir
  after the driver returns: the driver leaves the converged imports.json in
  place, and this participant is a pure function of it.

MEASURED (this file, through openPASO's own `run_coupling`, on a manufactured
two-material problem built for it: a rectangle split by a straight interface,
k = 1.7 / rho_c = 2.1 against k = 0.35 / rho_c = 0.8, an exact solution
quadratic in x and cosine in y times 1 - exp(-2.5 t), zero initial condition,
exact Dirichlet data on the outer x-faces, natural y-faces, THETA = 0.5, tol
1e-9; h AND dt halved together over four levels, errors at the FINAL TIME):

    quantity                              lvl0      lvl3     orders
    L2 field error, both subdomains     1.16e-03  1.82e-05  2.00 2.00 2.00
    L2 interface temperature            1.55e-03  2.45e-05  1.99 2.00 2.00
    max interface temperature           5.63e-03  1.25e-04  1.81 1.83 1.85
    L2 interface flux, INTERIOR nodes   5.30e-03  8.72e-05  1.97 1.98 1.98
    L2 interface flux, WHOLE interface  1.22e-02  5.00e-04  1.55 1.54 1.52
    max interface flux, WHOLE interface 5.21e-02  6.34e-03  1.04 1.01 0.99

(The interface flux BALANCE was measured separately and is below. Everything
in this table is a Dirichlet-side quantity or a field quantity, so none of it
moved when the Neumann side's flux export changed: the Dirichlet side consumes
the partner's `values`, never its `normal_fluxes`.)

THE TWO INTERFACE END NODES ARE FIRST ORDER HERE, and they set every whole-
interface norm. Measured with the exact interface data fed in, so the coupling
is not involved: interior nodes 1.97, 1.99, the two end nodes 1.03, 1.02, and
the end-node error is 6.8x the interior one at lvl0 and 25x at lvl2. Two O(h)
entries in an otherwise O(h^2) vector is what puts the whole-interface L2 at
~1.5 — report the interior number and the whole-interface number SEPARATELY, or
the recovery looks like a 1.5-order scheme. Copying a neighbour into them does
not fix it either; that value is O(h) at the end too.
  The mechanism is NOT established, and one measurement says it is not the
  obvious one. The obvious candidate is the half-width support at an end node
  making -r_i/w_i a one-sided average, but the SAME recovery on Q1
  quadrilaterals (measured with deal.II on the SAME manufactured problem) is
  2.00 at the end nodes as well as inside — the difference there is Q1
  quadrilaterals against these P1 triangles. So this is a property of the corner
  discretisation, not of the reaction formula. Treat the end nodes as suspect,
  and measure rather than assume on your own mesh.
  (The shipped corner guard is a DIFFERENT thing: it replaces interface nodes
  that also lie on an OUTER DIRICHLET boundary, whose residual is not this
  interface's flux at all. With OUTER_FACES = "x" it never fires, and the end
  nodes above are ordinary interface nodes that are simply less accurate.)

THE FLUX BALANCE IS AT THE FIXED-POINT TOLERANCE, not at some discretisation
order. It used to be order 1, and that was the Neumann side's projected
gradient rather than a leak; both sides now recover the reaction, and the
Neumann side's reaction IS the functional the Dirichlet side sent, so the two
exports cancel to whatever the driver's own iteration has converged to.
Measured through openPASO's `run_coupling` on a pair like the one above (k = 0.35
/ rho_c = 0.8 on the Dirichlet side against k = 1.7 / rho_c = 2.1 on the
Neumann side, an outer temperature varying in y so the interface flux varies
along the interface, THETA = 0.5, tol 1e-9, 43-44 iterations), h and dt halved
together, worst relative imbalance over the time levels:

    interface meshes         this file                the retired projection
    matching, lvl0/1/2   1.2e-07 5.5e-08 6.0e-08   6.1e-01 3.0e-01 1.5e-01
    non-matching, 17/13  5.4e-04                   5.3e-01

The right-hand column is order 1.00, 1.00 — the order of the projected
gradient. openPASO's conservation check reads the trace as one component per time
level and balances each on its own, so it returned ONE finding per time level
for that column at every level tried (8, 16 and 32 findings) and NONE for this
file's. The left column does not converge with h and must not be read as if it
did: 1e-7 is the driver's fixed-point residual, not a discretisation error. On
non-matching meshes what is left is the trace interpolation in `sample_trace`,
paid twice per iteration. Conservation of what is APPLIED was exact by
construction all along: the Neumann side integrates the Dirichlet side's
density unchanged.

CROSS-CODE. The same protocol against the deal.II transient participant
(participant_dealii_transient.py, FEniCSx on the Dirichlet side and deal.II on
the Neumann side) converged in 30-31 iterations and measured 2.00, 2.00 in the
final-time field L2 over three levels, 1.99 on the interior interface flux.

NON-MATCHING INTERFACE MESHES go through `sample_trace` and work: 37 interface
points against 25, converged in 30 iterations, the balance check took its
integral path and reported nothing. One level only, so no order from it — the
errors were 1.2x (field L2) to 7x (interface flux L2) those of the matching run
at the same level, which is the price of interpolating the trace twice per
iteration.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import dolfinx                # the MODULE, so dolfinx.log.set_log_level(...) resolves:
                              # a run log that must carry this code's own output needs it,
                              # and `from dolfinx import fem` alone leaves `dolfinx` undefined
import ufl
from dolfinx import default_scalar_type, fem, mesh as dmesh
from dolfinx.fem import petsc as _fp
from mpi4py import MPI
from petsc4py import PETSc


# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material, BCs and time
#    window. As shipped this is the LEFT / Dirichlet side; the payload that
#    served this script gives the exact block for the RIGHT / Neumann side.
SIDE      = "dirichlet"   # "dirichlet" (import T, export flux) | "neumann"
PARTNER   = "right"       # the partner's `name` in your couple(...) call
X0, X1    = 0.0, 0.6      # this subdomain's x-extent
Y0, Y1    = 0.0, 0.4      # this subdomain's y-extent
IFACE_X   = 0.6           # the shared interface; must equal X0 or X1
K         = 0.8           # conductivity
RHO_C     = 1.0           # VOLUMETRIC heat capacity rho*c  (NOT c alone)
NX, NY    = 24, 16        # this subdomain's OWN mesh; need not match the partner
T_START   = 0.0           # coupling window start  ─┐ BOTH participants must be
T_END     = 1.0           # coupling window end     ├─ given the SAME three
N_STEPS   = 20            # steps in the window     ─┘ numbers AND the same THETA
THETA     = 0.5           # 0.5 = Crank-Nicolson (2nd order) | 1.0 = backward
                          # Euler (L-stable, 1st order — caps the space-time
                          # order at 1 when you refine dt with h)
OUTER_FACES = "x"         # "x"  : only the non-interface x-face is Dirichlet,
                          #        the two y-faces are natural (zero flux)
                          # "all": the WHOLE non-interface boundary is Dirichlet


# The three problem functions. Each is called with NUMPY ARRAYS x, y and must
# return an array of the same shape — write `0.0 * x + c` for a constant, not
# `c`, or dolfinx sees a scalar where it needs an array.
def T_INITIAL(x, y):
    """T(x, y, T_START). Must agree with T_OUTER(., T_START) on the Dirichlet
    faces, or the first step carries an initial layer that Crank-Nicolson
    answers with oscillations rather than with second order."""
    return 0.0 * x + 300.0


def T_OUTER(x, y, t):
    """Dirichlet datum on the non-interface boundary selected by OUTER_FACES."""
    return 0.0 * x + 320.0


def F_SRC(x, y, t):
    """Volumetric source (per unit volume, same units as rho_c dT/dt)."""
    return 0.0 * x


Q_GUESS   = 0.0           # iteration-1 fallback interface flux. There is no
                          # T_GUESS: the interface temperature at t = T_START is
                          # KNOWN (it is T_INITIAL), so the Dirichlet side's
                          # iteration-1 trace is that value held constant.
# ─────────────────────────────────────────────────────────────────────────

OUTER_X = X0 if IFACE_X == X1 else X1
S = 1.0 if IFACE_X > OUTER_X else -1.0     # outward normal at interface = S*e_x



def _partner_block(imp):
    """The partner's block from imports.json, by the name the driver actually used."""
    if not isinstance(imp, dict) or not imp:
        return None
    if PARTNER in imp:
        return imp[PARTNER] or None
    others = [k for k in imp if isinstance(imp.get(k), dict)]
    if len(others) == 1:     # named differently in couple(...) than here: read it, say so
        import sys as _sys
        print(f"NOTE: PARTNER is {PARTNER!r} but imports.json is keyed {others[0]!r} "
              f"-- reading that block; align PARTNER with the name in your "
              f"couple(...) call.", file=_sys.stderr)
        return imp[others[0]] or None
    if others:
        raise SystemExit(f"imports.json holds blocks named {others} and none is "
                         f"PARTNER={PARTNER!r}: set PARTNER to the partner's name "
                         f"in your couple(...) call.")
    return None


def read_imports():
    """imports.json is {partner_name: InterfaceData}; `{}` on iteration 1,
    so the caller must fall back to an initial guess."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return _partner_block(json.loads(p.read_text()))
    except json.JSONDecodeError:
        return None


def sample_trace(imp, key, fallback, y):
    """Map the partner's interface TRACE onto THIS participant's points.

    Returns (len(y), N_STEPS): one column per time step, interpolated in y
    COLUMN BY COLUMN. The driver does no interpolation — non-matching interface
    meshes are handled here — and for a trace that has to be done per time
    level. One np.interp over the flattened (m, N_STEPS) array interleaves the
    time levels: the result still has the right length, the coupling still
    converges, and every number is wrong.

    `fallback` is a scalar or a (len(y),) array, held constant in time.
    """
    fb = np.asarray(fallback, float).ravel()
    if fb.size != len(y):
        fb = np.full(len(y), float(fb.ravel()[0]))
    if not imp or not imp.get("coordinates"):
        return np.tile(fb[:, None], (1, N_STEPS))
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key) or [], float)
    if vs.ndim == 1:                       # a partner that exported one column
        vs = vs.reshape(-1, 1)
    if vs.shape[0] != ys.size:
        return np.tile(fb[:, None], (1, N_STEPS))
    if vs.shape[1] != N_STEPS:
        # LOUD, because it is the one time-window error the payload can reveal.
        sys.exit(f"partner '{PARTNER}' exported a trace with {vs.shape[1]} time "
                 f"levels, this participant's window has N_STEPS={N_STEPS}. The "
                 f"exchange carries no time axis, so a trace of the WRONG "
                 f"LENGTH is the only symptom a mismatched window shows. Give "
                 f"both participants the same T_START/T_END/N_STEPS/THETA, and "
                 f"the same n_steps in config.json when a level sets it.")
    o = np.argsort(ys)
    return np.column_stack([np.interp(y, ys[o], vs[o, n])
                            for n in range(N_STEPS)])


imp = read_imports()

# ── THE PER-LEVEL RULE (served). A ./config.json {"level": k, "nx": .., "ny": ..,
#    "n_steps": ..} next to this script overrides the mesh knobs and the number of
#    time steps, and names the level. A study that refines dt with h sets n_steps
#    per level, on BOTH sides. The dumps at the foot of this file carry that level
#    in their NAME, so a mesh study leaves one file per level instead of the fine
#    mesh overwriting the coarse.
LEVEL = 1
if Path("config.json").is_file() or os.environ.get("OPENPASO_CONFIG_JSON"):
    try:
        _cfg = json.loads(Path("config.json").read_text() or "{}") if Path("config.json").is_file() else {}
        _cfg.update(json.loads(os.environ.get("OPENPASO_CONFIG_JSON") or "{}"))
        LEVEL = int(_cfg.get("level", LEVEL))
        NX = int(_cfg.get("nx", NX))
        NY = int(_cfg.get("ny", NY))
        N_STEPS = int(_cfg.get("n_steps", N_STEPS))
    except (ValueError, TypeError, json.JSONDecodeError):
        pass
DT = (T_END - T_START) / N_STEPS                   # after the level rule: n_steps sets it
TIMES = T_START + DT * np.arange(N_STEPS + 1)      # t^0 ... t^N

# MAKE THIS CODE SPEAK, BEFORE THE SOLVE RUNS. It is silent by default, and a
# per-level run log carrying no line the solver itself emitted cannot
# establish which code ran on this side, however right its numbers are.
# It sits HERE, beside the level rule, and not up with the imports:
# measured over agent-written participants, a line placed in the import
# block survived in about half of them because that block gets rewritten,
# while everything beside the level rule survived in all of them.
dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
domain = dmesh.create_rectangle(MPI.COMM_WORLD, [[X0, Y0], [X1, Y1]],
                                [NX, NY], dmesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))
fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)

xy = V.tabulate_dof_coordinates()
iface_dofs = np.where(np.abs(xy[:, 0] - IFACE_X) < 1e-10)[0]
iface_dofs = iface_dofs[np.argsort(xy[iface_dofs, 1])]   # constant order, always
y_if = xy[iface_dofs, 1]
if len(iface_dofs) == 0:
    sys.exit(f"no interface DOFs at x={IFACE_X}: this subdomain spans "
             f"[{X0},{X1}], so nothing is shared with the partner")

# ── outer (non-interface) Dirichlet boundary ──────────────────────────────
if OUTER_FACES == "all":
    def _on_outer(x):
        return (np.isclose(x[0], OUTER_X) | np.isclose(x[1], Y0)
                | np.isclose(x[1], Y1))
else:
    def _on_outer(x):
        return np.isclose(x[0], OUTER_X)

outer_facets = dmesh.locate_entities_boundary(domain, fdim, _on_outer)
outer_dofs = fem.locate_dofs_topological(V, fdim, outer_facets)

# WITH OUTER_FACES = "all" THE TWO INTERFACE END NODES BELONG TO THE OUTER
# BOUNDARY, ON BOTH SIDES: (IFACE_X, Y0) and (IFACE_X, Y1) sit on a y-face,
# which carries prescribed data in the un-split problem, so they stay outer-
# Dirichlet in BOTH subproblems and are NOT interface-imposed. They are still
# exported. With OUTER_FACES = "x" those faces are natural and the end nodes
# ARE interface nodes like any other.
end_node = (np.abs(y_if - Y0) < 1e-10) | (np.abs(y_if - Y1) < 1e-10)
iface_bc_dofs = (iface_dofs[~end_node] if OUTER_FACES == "all"
                 else iface_dofs).astype(np.int32)

# ── forms: theta-scheme, ONE matrix for the whole march ───────────────────
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
kc = fem.Constant(domain, default_scalar_type(K))
rc = fem.Constant(domain, default_scalar_type(RHO_C))
dtc = fem.Constant(domain, default_scalar_type(DT))
th = fem.Constant(domain, default_scalar_type(THETA))
omth = fem.Constant(domain, default_scalar_type(1.0 - THETA))

u_n = fem.Function(V)        # solution at t^n
uh = fem.Function(V)         # solution at t^(n+1)
f_old = fem.Function(V)      # F_SRC(., t^n)
f_new = fem.Function(V)      # F_SRC(., t^(n+1))
g_out = fem.Function(V)      # outer Dirichlet datum at t^(n+1)
g_if = fem.Function(V)       # interface datum for the CURRENT step

a = (rc / dtc) * u * v * ufl.dx + th * kc * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
# THE WHOLE THETA-SCHEME LOAD EXCEPT THE INTERFACE TERM, kept in its own form.
# It carries the OLD-STEP terms as well as the source — (M/dt) u^n, the
# explicit stiffness -(1-THETA) A u^n, and THETA F^(n+1) + (1-THETA) F^n —
# because the residual the flux recovery below takes is the residual of the
# time-discrete equation, not of a steady one. Subtracting the volume source
# alone would leave (M/dt) u^n and the explicit stiffness in the reaction and
# the exported flux would be nonsense of size 1/dt. The ONLY thing missing from
# L_vol is the interface term, and that omission is the whole point.
L_vol = ((rc / dtc) * u_n * v * ufl.dx
         - omth * kc * ufl.dot(ufl.grad(u_n), ufl.grad(v)) * ufl.dx
         + th * f_new * v * ufl.dx + omth * f_old * v * ufl.dx)
L = L_vol

bcs = [fem.dirichletbc(g_out, outer_dofs)]

# One definition of the interface measure, used by the Neumann branch to APPLY
# the partner's flux and by the Dirichlet branch to RECOVER its own.
facets_if = dmesh.locate_entities_boundary(domain, fdim,
                                           lambda x: np.isclose(x[0], IFACE_X))
tags_if = dmesh.meshtags(domain, fdim, np.sort(facets_if),
                         np.full(len(facets_if), 7, dtype=np.int32))
ds_if = ufl.Measure("ds", domain=domain, subdomain_data=tags_if)(7)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

# ── THE INTERFACE DOFS ARE THE INTERFACE NODES' OWN (served) ─ keep this block.
#    Every served line below applies the trace, reads the flux and writes the
#    values THROUGH iface_dofs, so a list naming other dofs exports their values
#    under the interface's coordinates -- and a check reading the same list
#    agrees with it. The nodes are read from the mesh here, never from the list.
_TOL_IF = 1e-9 * max(X1 - X0, Y1 - Y0)
_xy_all = uh.function_space.tabulate_dof_coordinates()
_line = np.where(np.abs(_xy_all[:, 0] - IFACE_X) <= _TOL_IF)[0]
_ids = np.asarray(iface_dofs).astype(int).ravel()
_yv = np.asarray(y_if, float).ravel()
_bad = [k for k, d in enumerate(_ids)
        if k >= _yv.size or not (0 <= d < len(_xy_all))
        or abs(_xy_all[d, 0] - IFACE_X) > _TOL_IF or abs(_xy_all[d, 1] - _yv[k]) > _TOL_IF]
_missed = len(set(_line.tolist()) - set(_ids.tolist()))
if _bad or _missed or len(_ids) != _yv.size:
    _k = _bad[0] if _bad else None
    sys.exit(
        f"INTERFACE DOFS: iface_dofs[k] must be the dof on the interface line at y_if[k], one "
        f"per node; iface_dofs has {len(_ids)} entries for {_yv.size} nodes"
        + (f", and {len(_bad)} of them are not (the first: iface_dofs[{_k}] = {_ids[_k]}"
           + (f", a dof at ({_xy_all[_ids[_k], 0]:g}, {_xy_all[_ids[_k], 1]:g})"
              if 0 <= _ids[_k] < len(_xy_all) else ", no dof of this space")
           + ")" if _bad else "")
        + (f"; {_missed} of the {len(_line)} dofs on the interface line have no entry"
           if _missed else "")
        + ". Take them from V.tabulate_dof_coordinates() of the space the solution lives in.")

if SIDE == "dirichlet":
    # The trace this side IMPOSES: partner temperature at t^1 ... t^N.
    imp_trace = sample_trace(imp, "values",
                             T_INITIAL(xy[iface_dofs, 0], xy[iface_dofs, 1]),
                             y_if)
    bcs.append(fem.dirichletbc(g_if, iface_bc_dofs))
else:
    # The trace this side APPLIES: partner's THETA-AVERAGED flux per step,
    # UNCHANGED (see THE FLUX in the module docstring).
    imp_trace = sample_trace(imp, "normal_fluxes", Q_GUESS, y_if)
    L = L_vol + g_if * v * ds_if

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
af, Lf, Lvolf = fem.form(a), fem.form(L), fem.form(L_vol)

# THE THETA-SCHEME MATRIX DOES NOT DEPEND ON t. Assemble and factorize it ONCE:
# with a fixed dt and a fixed set of constrained dofs, re-assembling per step
# buys nothing and pays N_STEPS LU factorizations for it.
A = _fp.assemble_matrix(af, bcs=bcs)
A.assemble()
ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")

# ── initial condition ─────────────────────────────────────────────────────
u_n.interpolate(lambda X: T_INITIAL(X[0], X[1]))
f_old.interpolate(lambda X: F_SRC(X[0], X[1], TIMES[0]))
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

# THE SAME OPERATOR ASSEMBLED WITH NO BOUNDARY CONDITION, plus the nodal
# interface weight, kept for the reaction recovery in the march. BOTH sides need
# them now: the recovery is one formula (see THE CONSISTENT FLUX below), so
# there is nothing left that is Dirichlet-only here.
A_free = _fp.assemble_matrix(af)          # no bcs= on purpose
A_free.assemble()
wvec = _fp.assemble_vector(fem.form(v * ds_if))   # w_i = int_Gamma phi_i ds
wvec.ghostUpdate()
wi = wvec.array[iface_dofs]
ok = np.abs(wi) > 1e-14
# An interface node that ALSO lies on the outer Dirichlet boundary carries the
# OUTER reaction as well, so its residual is not this interface's flux. Take the
# nearest interior interface node rather than exporting a corner value that is
# physically a different quantity. This applies on BOTH sides; with
# OUTER_FACES = "x" it never fires, because then no interface node is outer.
# outer_dofs AS DOF NUMBERS: np.isin reads a Python set as ONE object and matches
# nothing (measured: the corner values went out unreplaced).
suspect = np.isin(iface_dofs, sorted(outer_dofs) if isinstance(outer_dofs, (set, frozenset)) else outer_dofs) | ~ok
good = np.where(~suspect)[0]
fixup = [(i, good[np.argmin(np.abs(good - i))])
         for i in np.where(suspect)[0]] if len(good) else []

# ── THE STEP'S CHECKS, SET UP ONCE (served) ─ keep this block. Both read the
#    mesh's own dofs, never the lists the solve writes through.
_msh = uh.function_space.mesh
_msh.topology.create_connectivity(_msh.topology.dim - 1, _msh.topology.dim)
_bdry = fem.locate_dofs_topological(uh.function_space, _msh.topology.dim - 1,
                                    dmesh.exterior_facet_indices(_msh.topology))
_inside = np.setdiff1d(np.arange(len(_xy_all)), _bdry)      # the dofs off every boundary
_A_inf = A_free.norm(PETSc.NormType.NORM_INFINITY)          # the largest row sum of |A_free|
_held = _line[(np.abs(_xy_all[_line, 1] - Y0) > _TOL_IF) & (np.abs(_xy_all[_line, 1] - Y1) > _TOL_IF)]
_held_want = (sample_trace(imp, "values", T_INITIAL(_xy_all[_held, 0], _xy_all[_held, 1]),
                           _xy_all[_held, 1]) if SIDE == "dirichlet" else None)

T_out = np.zeros((len(iface_dofs), N_STEPS))
Q_out = np.zeros((len(iface_dofs), N_STEPS))

# ── the march. ONE run = the WHOLE window (waveform) ──────────────────────
for n in range(N_STEPS):
    # ONE STEP, t^n -> t^(n+1). The hole below is this step's solve and the body
    # of this loop: write it at this indentation. Apply column n of imp_trace (the
    # partner's datum for the step) as SIDE says, solve for uh at t^(n+1), keep
    # b_vol (the step's load WITHOUT the interface term, assembled with no lifting
    # and no set_bc, so the constrained rows keep their reaction), and move the
    # old-step state on to t^(n+1). The lines after it are served.
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    t_new = TIMES[n + 1]
    f_new.interpolate(lambda X: F_SRC(X[0], X[1], t_new))
    g_out.x.array[outer_dofs] = T_OUTER(xy[outer_dofs, 0], xy[outer_dofs, 1],
                                        t_new)
    # The imported trace enters HERE, one column per step: column n is the
    # partner's datum for the step t^n -> t^(n+1).
    g_if.x.array[iface_dofs] = imp_trace[:, n]

    b = _fp.assemble_vector(Lf)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    # THE SAME LOAD WITHOUT THE INTERFACE TERM, for the recovery below: no
    # lifting and no set_bc, so the constrained rows keep their reaction. On the
    # Dirichlet side Lvolf IS Lf and this assembly is redundant; it is done
    # unconditionally so that the march has one code path and the recovery
    # cannot be handed the wrong vector on one side.
    b_vol = _fp.assemble_vector(Lvolf)
    b_vol.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    _fp.apply_lifting(b, [af], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    _fp.set_bc(b, bcs)
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    u_n.x.array[:] = uh.x.array
    f_old.x.array[:] = f_new.x.array
    b.destroy()
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    # Outward normal flux density q = -(K grad T).n on the interface, THETA-
    # averaged over this step.
    #
    # WHY NOT AN L2 PROJECTION OF THE GRADIENT. The gradient of a P1 solution is
    # only O(h) accurate ON the boundary — the superconvergence points are
    # interior — and the boundary trace is exactly what the coupling reads.
    # Measured on the manufactured transient two-material problem, refining h
    # and dt together, with the SHIPPED file against one copy of it whose
    # Dirichlet side exports the projected gradient instead and nothing else
    # changed:
    #     interface flux, interior nodes, L2   1.98  ->  1.11
    #     interface temperature, L2            2.00  ->  1.24
    #     FINAL-TIME FIELD L2, BOTH SUBDOMAINS 2.00  ->  1.37 (and falling)
    # The coupling converged just as well either way — 29-30 iterations against
    # 31, residual 7e-10 to 9e-10 against 5e-10, every driver check silent — so
    # nothing in the run tells you which of those two answers you got.
    #
    # THE CONSISTENT (REACTION) FLUX. From
    #     a(u,v) - (f,v) = int_dOmega (K grad u . n) v ds = -int_Gamma qn v ds
    # applied to the theta-scheme's own discrete equation, for every basis
    # function phi_i on the interface
    #     int_Gamma [THETA q^(n+1) + (1-THETA) q^n] phi_i ds = -r_i,
    #     r = (M/dt + THETA A) u^(n+1) - (M/dt - (1-THETA) A) u^n
    #         - THETA F^(n+1) - (1-THETA) F^n
    # with r the UNCONSTRAINED residual: assembled with no boundary condition
    # applied and with the constrained rows NOT zeroed, because on the Dirichlet
    # side those rows ARE the reaction and zeroing them destroys the very
    # quantity being recovered. Dividing by w_i = int_Gamma phi_i ds turns the
    # functional into a density the partner can interpolate pointwise.
    #
    # WHICH TERMS ARE IN b_vol, AND WHY EVERY ONE OF THEM IS THERE. Written out,
    #     A_free  = M/dt + THETA*A
    #     b_vol   = (M/dt) u^n - (1-THETA)*A u^n
    #               + THETA F^(n+1) + (1-THETA) F^n
    # so r = A_free u^(n+1) - b_vol is EXACTLY the r above. b_vol is the
    # complete theta-scheme right-hand side with ONE term removed: the interface
    # term. The old-step mass and stiffness contributions stay IN, because they
    # are part of the equation whose residual this is; drop (M/dt) u^n and the
    # reaction picks up a spurious (M/dt)(u^(n+1) - u^n), which is O(1/dt) and
    # grows as the step is refined — a recovery that looks worse the finer you
    # go. Nothing here assumes THETA = 1; for backward Euler the (1-THETA) terms
    # vanish on their own and this reduces to (M/dt + A) u^(n+1) - (M/dt) u^n
    # - F^(n+1).
    #
    # ONE FORMULA, BOTH SIDES. An earlier version of this file used the reaction
    # only on the Dirichlet side and an L2-projected gradient on the Neumann
    # side, on the reasoning that the Neumann interface dofs are free, so the
    # discrete equations hold on them and r comes out ~0 (measured then: max|r|
    # on those rows 1.6e-16). That is true only when the residual is taken
    # against a load that ALREADY CONTAINS the interface term. Against b_vol
    # those same rows carry exactly the interface functional the partner
    # applied,
    #     (A_free u^(n+1) - b_vol)_i = int_Gamma g phi_i ds,
    # and g is the partner's THETA-AVERAGE for this step, applied unchanged — so
    # the recovery returns the theta-average on this side too, with no explicit
    # blend and no flux at t^0 to carry along. On the Dirichlet side there is no
    # interface term at all, so b == b_vol and the two cases are one expression.
    #
    # MEASURED ON THIS FILE, on the Neumann side, by handing it a flux that
    # VARIES along the interface — q = 2 + 3 sin(4y), held constant in time so
    # that the step's theta-average IS q — and asking for it back. Interior
    # interface nodes, n = 8/16/32, N_STEPS = 8, non-zero source, max error over
    # every time level, against a true flux whose size is 2 to 5:
    #     projected gradient   2.78, 1.42, 7.06e-01   orders 0.97, 1.01
    #     reaction vs b_vol    1.96e-02, 4.98e-03, 1.25e-03
    #
    # READ THE SECOND ROW CORRECTLY — IT IS NOT A CONVERGENCE RESULT. On the
    # Neumann side the free interface rows satisfy
    # (A_free u^(n+1) - b_vol)_i = (M_Gamma g)_i IDENTICALLY, so the export is
    # the consistent-to-nodal conversion -(M_Gamma g)/(M_Gamma 1) and its
    # offset from -g is -(h^2/6) g''(y) for ANY correct assembly of ANY
    # equation. This row used to carry "orders 1.98, 2.00" as if it measured
    # the recovery's accuracy; it does not. Those three figures are the P1
    # boundary mass matrix and are reproduced to five significant figures by a
    # bare NumPy mass matrix with no PDE, no time stepping and no solver in it,
    # which is also why they agree to three digits with what the steady
    # conduction participants get on the same meshes — the same algebra, not an
    # agreement between codes. The recovery's ORDER is measured elsewhere,
    # steady and on the DIRICHLET side, against an analytic flux
    # (tests/test_interface_flux_converges_to_a_known_exact_flux.py): 1.996,
    # 1.998, 1.993 on 8/16/32/64. There is no transient measurement of it.
    #
    # WHAT THE ROW IS STILL GOOD FOR, and it is the reason it stays: it is
    # IDENTICAL for THETA = 0.5 and THETA = 1.0, and refining dt alone at
    # n = 16, N_STEPS = 4/8/16/32, the recovered flux does not move at all
    # (4.9833e-03 at every dt). That IS a real check — that b_vol carries the
    # old-step terms. Drop them and this column blows up as 1/dt. The
    # projected-gradient row above is also a genuine measurement: it passes
    # through the discretisation, and order ~1 is what a P1 gradient evaluated
    # ON a boundary is worth. The Dirichlet side is untouched by the change
    # (b == b_vol there): its export is bit-identical to the previous
    # version's, checked.
    r = A_free.createVecLeft()
    A_free.mult(uh.x.petsc_vec, r)
    r.axpy(-1.0, b_vol)
    q = np.zeros(len(iface_dofs))
    q[ok] = -r.array[iface_dofs][ok] / wi[ok]
    for i, j in fixup:
        q[i] = q[j]

    # ── THE STEP ANSWERS FOR ITS OWN SYSTEM (served) ─ keep this block. Off every
    #    boundary a solved step leaves r = A_free u^(n+1) - b_vol at round-off, on
    #    either side: the interface term and the boundary conditions act on
    #    boundary rows only.
    _r_in = np.abs(r.array[_inside])
    _scale = max(float(np.abs(b_vol.array).max()), _A_inf * float(np.abs(uh.x.array).max()))
    if _r_in.size and _scale > 0 and _r_in.max() > 1e-6 * _scale:
        sys.exit(f"SOLVE SELF-CHECK: step {n + 1} of {N_STEPS}: off the boundary, "
                 f"r = A_free u - b_vol reaches {_r_in.max():.2e} against a system scale of "
                 f"{_scale:.2e} ({int((_r_in > 1e-6 * _scale).sum())} of {_r_in.size} dofs); "
                 f"a solved step leaves round-off there, so the step's uh does not solve the "
                 f"system that A_free and b_vol describe. Three ways measured to get here: b "
                 f"assembled without _fp.apply_lifting before _fp.set_bc; a preconditioner "
                 f"applied once in place of a solve (ksp 'preonly' with pc 'jacobi'); and a "
                 f"Krylov solve left at PETSc's default rtol of 1e-5 (ksp.setTolerances("
                 f"rtol=1e-10), or 'preonly' with 'lu', solves it). Nothing was exported.")
    # ── THE PARTNER'S TRACE IS HELD (served, Dirichlet side) ─ keep this block.
    #    A Dirichlet side whose solve frees the interface returns its own answer
    #    there, and the coupling converges to two fields that disagree. The two
    #    end nodes are left out: the outer boundary may hold them.
    if SIDE == "dirichlet" and _held.size:
        _gap = float(np.abs(uh.x.array[_held] - _held_want[:, n]).max())
        if _gap > 1e-9 * max(1.0, float(np.abs(_held_want[:, n]).max())):
            sys.exit(f"EXPORT SELF-CHECK: step {n + 1} of {N_STEPS}: the partner's temperature "
                     f"is not in the solution at the interface nodes (largest gap {_gap:.3e}): "
                     f"on the Dirichlet side the interface must be held -- a dirichletbc on the "
                     f"interface dofs, with this step's column of the partner's trace, in the "
                     f"bcs the step's matrix, lifting and set_bc use. A solve that frees them "
                     f"returns this side's own answer and couples to nothing.")
    r.destroy()
    b_vol.destroy()

    T_out[:, n] = uh.x.array[iface_dofs]
    Q_out[:, n] = q

# FIELD DUMP, BEFORE exports.json (see the docstring). `weights` are the nodal
# volume weights int phi_i dx, so a mass-lumped L2 norm of any nodal field is
# sqrt(sum(weights * field**2)) with no mesh work outside this script.
wvol = _fp.assemble_vector(fem.form(v * ufl.dx))
wvol.ghostUpdate()
np.savez("field_final.npz", coordinates=xy[:, :2], temperature=uh.x.array.copy(),
         weights=wvol.array.copy(), times=TIMES, iface_dofs=iface_dofs,
         side=SIDE, theta=THETA)

print(f"[fenics-transient {SIDE}] iface n={len(iface_dofs)} steps={N_STEPS} "
      f"dt={DT:.6g} theta={THETA} "
      f"T(t_end)=[{T_out[:, -1].min():.6g},{T_out[:, -1].max():.6g}] "
      f"q(last step)=[{Q_out[:, -1].min():.6g},{Q_out[:, -1].max():.6g}]")

# THE RUN-LOG CONTRACT LINE: `NDOF = <integer>` on a line of its OWN.
# The audit reads that exact shape, and they read it PER
# LEVEL: it is how anyone checking the result tells a refined mesh from the same mesh run
# three times. The LEADING NEWLINE is deliberate -- a program that writes
# without a trailing newline glues its text onto the front of the next
# line, and an X11 warning has done exactly that here, turning a correct
# line into 'Invalid MIT-MAGIC-COOKIE-1 keyNDOF = 54'.
# A number inside a prose sentence does not count either, and a
# wrong number is worse than none -- one coupled run that was right in
# every other respect reported NDOF = 1 at all three levels, and its
# refined mesh could not be told from an unrefined one.
try:
    print(f"\nNDOF = {int(len(uh.x.array))}")
except Exception as _ndof_exc:
    print(f"[fenics-transient] could not report NDOF: {_ndof_exc!r}. Your task's"
          f" execution log needs `NDOF = <integer>` on a line of its own,"
          f" so print your own degree-of-freedom count here.")

# PER-LEVEL PERSISTENCE: the END-OF-WINDOW field, and the interface trace and
# flux at the last step, named by LEVEL. field_final.npz and exports.json are
# both overwritten by the next level; these files are not, which is what a mesh
# study needs. Interpolate THESE onto the probe points your task names.
# A DUMP DEFECT MUST NOT COST YOU THE SOLVE. exports.json is the driver's
# proof that this participant succeeded, and it is written after these files,
# so an exception here would throw away a coupling iteration that worked.
try:
    with open(f"field_level{LEVEL}.csv", "w") as _f:
        _f.write("x,y,u\n")
        for (_px, _py), _u in zip(xy[:, :2], uh.x.array):
            _f.write(f"{float(_px):.11e},{float(_py):.11e},{float(_u):.11e}\n")
    with open(f"interface_level{LEVEL}.csv", "w") as _f:
        _f.write("x,y,u,qn\n")
        for _yy, _t, _q in zip(y_if, T_out[:, -1], Q_out[:, -1]):
            _f.write(f"{float(IFACE_X):.11e},{float(_yy):.11e},"
                     f"{float(_t):.11e},{float(_q):.11e}\n")
except Exception as _dump_exc:
    # AND LEAVE NO HALF-WRITTEN FILE BEHIND. `open(..., "w")` truncates
    # before it fails, so a dump that died mid-way leaves a header-only
    # CSV -- a file that looks like a submission and carries no rows.
    for _partial in (f"field_level{LEVEL}.csv", f"interface_level{LEVEL}.csv"):
        try:
            if Path(_partial).is_file() and len(
                    Path(_partial).read_text().splitlines()) <= 1:
                Path(_partial).unlink()
        except OSError:
            pass
    print(f"[fenics_transient per-level dump] level {LEVEL} dump failed: "
          f"{_dump_exc!r}. exports.json is still written, so the coupling\n"
          f"continues, but this level has no field file to hand in. Fix the\n"
          f"names the dump reads and run this level again.")

# exports.json LAST: the driver takes its existence as proof of success.
# ── EXPORT SELF-CHECK ─ keep this block. It stops the three exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
#    a flux of ~0 against a nonzero partner); and a flux that is the
#    partner's array negated instead of a recovery from THIS side's own system.
_chk_vals = np.asarray(T_out, float).ravel()
_chk_flux = np.asarray(Q_out, float).ravel()
if not (np.isfinite(_chk_vals).all() and np.isfinite(_chk_flux).all()):
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface values or "
                     "fluxs; the solve did not produce a usable field, so "
                     "nothing was exported")
_chk_imp = (json.loads(Path("imports.json").read_text() or "{}")
            if Path("imports.json").is_file() else {})
_chk_qin = (np.concatenate([np.asarray(_d.get("normal_fluxes") or [], float).ravel()
                            for _d in _chk_imp.values()])
            if _chk_imp else np.zeros(0))
if SIDE == "neumann" and _chk_qin.size and np.abs(_chk_qin).max() > 0 \
        and np.abs(_chk_flux).max() < 1e-9 * np.abs(_chk_qin).max():
    raise SystemExit("EXPORT SELF-CHECK: the recovered interface flux is ~0 "
                     "against a nonzero imported flux: the imported load "
                     "never entered the assembled system (the facet term / "
                     "boundary condition that integrates it is missing). Fix "
                     "the application; do not couple on")
# (Dirichlet role only: a Neumann side's consistent recovery of a CONSTANT
#  applied load can legitimately reproduce it to the last bit.)
if SIDE == "dirichlet" and _chk_qin.shape == _chk_flux.shape and _chk_flux.size \
        and np.array_equal(_chk_flux, -_chk_qin):
    raise SystemExit("EXPORT SELF-CHECK: the exported flux is the partner's "
                     "array negated, bit for bit: a copy, not a recovery from "
                     "this side's own assembled system")

Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(iface_dofs)),
    "coordinates": [[float(IFACE_X), float(y)] for y in y_if],
    "values": [[float(t) for t in row] for row in T_out],
    "normal_fluxes": [[float(q) for q in row] for row in Q_out],
}, indent=2))
