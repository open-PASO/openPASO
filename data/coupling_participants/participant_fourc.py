"""4C participant for the openPASO `couple` driver (Scalar_Transport = conduction).

4C is a compiled YAML-in / VTU-out code with no Python API, so a participant is
a small Python WRAPPER: write the deck from imports.json, run the 4C binary,
read the VTU back, write exports.json.

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.
Needs `meshio` and `numpy` in whatever interpreter runs this wrapper — that is
openPASO's own interpreter, NOT the 4C binary.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side; the payload that served
#    this script gives the exact block for the RIGHT / Neumann side.
SIDE      = "dirichlet"   # "dirichlet" (import T, export flux) | "neumann"
PARTNER   = "right"       # the partner's `name` in your couple(...) call
X0, X1    = 0.0, 0.6      # this subdomain's x-extent
Y0, Y1    = 0.0, 0.4      # this subdomain's y-extent
IFACE_X   = 0.6           # the shared interface; must equal X0 or X1
K         = 0.8           # DIFFUSIVITY of MAT_scatra


def F_SRC(x, y):
    """Volumetric source, as a function of position.

    Returns zero as shipped, which is a PLACEHOLDER like every number above
    and is almost never what your problem wants. THIS KNOB USED TO BE A SCALAR
    CONSTANT, AND A CONSTANT CANNOT REPRESENT A SOURCE THAT VARIES WITH
    POSITION: the source of a manufactured solution is a POLYNOMIAL in x and y,
    and no single number is that polynomial. Left at zero the temperature is
    harmonic, the outer Dirichlet values are the only data left in the problem,
    and the answer degenerates to the 1-D profile between them — the interface
    flux is one constant along the whole interface, and it is identically zero
    when the two subdomains carry the same outer value. The coupling will
    converge beautifully to that, and it is not the problem you were given.

    If your problem states a source, or gives you a manufactured solution whose
    source term you derived, put it here. `x` and `y` are NumPy arrays, so
    build the answer with NumPy and return ONE array of the same shape (write
    `0.0 * x + c` for a genuine constant, never a bare `c`):

        # -div(K grad T) for the manufactured T = x**3 * y**2
        return -K * (6.0 * x * y**2 + 2.0 * x**3)

    4C TAKES A SOURCE AS A SYMBOLIC EXPRESSION, NEVER A TABLE, so what you write
    here is fitted by a polynomial in x and y on the way into the deck — see
    src_expr() below. A polynomial source is hit exactly. The fit is CHECKED
    against your samples and the run is ABORTED if it cannot reproduce them, so
    a source beyond the reach of a polynomial of total degree SRC_DEG_MAX
    (np.sin, say) stops this participant instead of quietly solving with an
    approximation of it.
    """
    return np.zeros_like(x)
T_OUTER   = 335.0         # Dirichlet value on the NON-interface x-boundary
# WHICH NON-INTERFACE EDGES ARE HELD IS YOUR PROBLEM'S TO SAY, NOT THIS FILE'S.
#
# This line used to read `FULL_OUTER_DIRICHLET = False` with a comment calling
# the y-edges "insulated (natural, nothing to do)". A default here silently
# chooses a boundary condition for a problem it has never seen, and a wrong
# choice does not announce itself: the field converges cleanly to a function
# that satisfies the condition this file picked, is as large on the unheld
# edges as anywhere inside, and no self-consistency check can see it, because
# the field IS consistent with the wrong condition. A template holds
# PLACEHOLDERS, never a problem's values, and a boundary condition is a value.
#
# Set it from your problem statement: True if the y = Y0 and y = Y1 edges are
# held at T_OUTER, False if your problem prescribes them natural (zero flux).
# The script refuses to run until you have.
FULL_OUTER_DIRICHLET = None  # <-- True or False, FROM YOUR PROBLEM STATEMENT
if FULL_OUTER_DIRICHLET is None:
    sys.exit("FULL_OUTER_DIRICHLET is unset: say whether the non-interface y-edges "
             "are held at T_OUTER (True) or natural (False), from your problem "
             "statement. A default here would be choosing your boundary "
             "condition for you.")
NX, NY    = 46, 26        # this subdomain's OWN QUAD4 mesh
T_INIT    = 310.0         # iteration-1 fallback interface temperature
Q_INIT    = 0.0           # iteration-1 fallback interface flux
FOURC_BIN = "4C"          # the 4C binary path `discover(query='list')` prints
FOURC_LD  = ""            # 4C dependency lib dir, or "" to inherit the env
FIT_DEG   = 3             # degree of the polynomial 4C FUNCT profile (see notes)
SRC_DEG_MAX = 8           # highest total degree tried when fitting F_SRC
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
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return _partner_block(json.loads(p.read_text()))
    except json.JSONDecodeError:
        return None


def sample(imp, key, fallback, ys):
    if not imp or not imp.get("coordinates"):
        return np.full(len(ys), float(fallback))
    yy = np.array([c[1] for c in imp["coordinates"]], float)
    vv = np.asarray(imp.get(key, []), float).ravel()
    if vv.size != yy.size:
        return np.full(len(ys), float(fallback))
    o = np.argsort(yy)
    return np.interp(ys, yy[o], vv[o])


# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
def funct_expr(ys, vals, deg):
    """4C takes a boundary profile as VAL x FUNCT(x,y,z,t), and FUNCT is a
    symbolic expression — not a table. So the imported samples are fitted by a
    least-squares polynomial in y and emitted as that expression."""
    if float(np.ptp(vals)) < 1e-14:
        return f"{float(vals[0]):.12e}"
    c = np.polyfit(ys, vals, int(min(deg, len(ys) - 1)))[::-1]
    return " + ".join(f"({v:.12e})*y^{i}" if i else f"({v:.12e})"
                      for i, v in enumerate(c))


def src_expr(vals, gx, gy, rtol=1e-9):
    """Turn samples of F_SRC into a 4C FUNCT expression, or None if it is zero.

    THE REAL 4C MECHANISM. A volumetric source in a 2D Scalar_Transport problem
    is a DESIGN SURF NEUMANN CONDITION on the whole domain: 4C's element routine
    reads it as the body force, evaluates VAL x FUNCT(x,y,z,t) AT THE ELEMENT
    NODES and integrates the shape-function interpolant of that — so with
    VAL: [1.0] and a FUNCT of space, the source really does vary with position,
    with the same Q1-interpolant semantics as the other participants. (In 3D the
    same condition would be DESIGN VOL NEUMANN; 4C picks by element dimension.)

    THE CATCH. 4C's FUNCT is a SYMBOLIC EXPRESSION, never a table, exactly as
    for the boundary profile in funct_expr() above. So F_SRC is sampled on the
    element-node grid and fitted by a least-squares polynomial in x and y, of
    the lowest total degree that REPRODUCES the samples. A polynomial source —
    which is what a manufactured solution gives you — is hit exactly. Anything
    the fit cannot reproduce within rtol is a HARD ERROR: silently solving with
    the best degree-SRC_DEG_MAX approximation of a source you did not ask for is
    the same failure as the constant this function replaced, one step further
    down the pipe.
    """
    scale = float(np.max(np.abs(vals)))
    if scale == 0.0:
        return None                       # no source: no condition is written
    xs, ys_, vs = gx.ravel(), gy.ravel(), vals.ravel()
    pw, c, err = [], np.zeros(0), float("inf")
    for deg in range(max(SRC_DEG_MAX, 0) + 1):
        pw = [(i, d - i) for d in range(deg + 1) for i in range(d + 1)]
        A = np.column_stack([xs ** i * ys_ ** j for i, j in pw])
        c = np.linalg.lstsq(A, vs, rcond=None)[0]
        c = np.where(np.abs(c) > 1e-12 * np.max(np.abs(c)), c, 0.0)  # fit noise
        err = float(np.max(np.abs(A @ c - vs)))   # of the coefficients EMITTED
        if err <= rtol * scale:
            break
    else:
        sys.exit(f"F_SRC cannot be written as a 4C FUNCT: no polynomial up to "
                 f"total degree {SRC_DEG_MAX} reproduces it on the element-node "
                 f"grid (best max error {err:.3e} against a source of size "
                 f"{scale:.3e}). 4C takes a source as a symbolic expression, so "
                 f"either give F_SRC a polynomial form, raise SRC_DEG_MAX, or "
                 f"use a backend that takes the callable directly. Refusing to "
                 f"solve with an approximation of a source you did not ask for.")
    terms = []
    for (i, j), v in zip(pw, c):
        if v == 0.0:
            continue
        terms.append(f"({v:.12e})" + (f"*x^{i}" if i else "")
                     + (f"*y^{j}" if j else ""))
    return " + ".join(terms) if terms else None
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end


imp = read_imports()
ys = np.linspace(Y0, Y1, NY + 1)
iface_vals = sample(imp, "values" if SIDE == "dirichlet" else "normal_fluxes",
                    T_INIT if SIDE == "dirichlet" else Q_INIT, ys)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
expr = funct_expr(ys, iface_vals, FIT_DEG)

# F_SRC on the element-node grid — the points at which 4C itself evaluates the
# body-force FUNCT, so this is exactly what the fit has to reproduce.
gx, gy = np.meshgrid(np.linspace(X0, X1, NX + 1), np.linspace(Y0, Y1, NY + 1),
                     indexing="ij")
fsrc = np.broadcast_to(np.asarray(F_SRC(gx, gy), float), gx.shape)
src = src_expr(fsrc, gx, gy)

# ── inline QUAD4 TRANSPORT mesh on [X0,X1] x [Y0,Y1] ─────────────────────
nid, nodes, grid = 1, [], {}
for j in range(NY + 1):
    for i in range(NX + 1):
        nodes.append(f"NODE {nid} COORD {X0 + i * (X1 - X0) / NX:.12f} "
                     f"{Y0 + j * (Y1 - Y0) / NY:.12f} 0.0")
        grid[(i, j)] = nid
        nid += 1
elems = []
for j in range(NY):
    for i in range(NX):
        elems.append(f"{len(elems) + 1} TRANSP QUAD4 {grid[(i, j)]} "
                     f"{grid[(i + 1, j)]} {grid[(i + 1, j + 1)]} "
                     f"{grid[(i, j + 1)]} MAT 1 TYPE Std")
i_out = 0 if OUTER_X == X0 else NX
i_if = 0 if IFACE_X == X0 else NX
if FULL_OUTER_DIRICHLET and NY < 2:
    sys.exit("FULL_OUTER_DIRICHLET needs NY >= 2 so the interface has an interior")

# DLINE 1 = outer Dirichlet boundary, DLINE 2 = the coupling interface.
iface_block = ""
dirich = ("DESIGN LINE DIRICH CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n"
          f"    ONOFF: [1]\n    VAL: [{T_OUTER}]\n    FUNCT: [0]\n")
if SIDE == "dirichlet":
    dirich += ("  - E: 2\n    NUMDOF: 1\n    ONOFF: [1]\n"
               "    VAL: [1.0]\n    FUNCT: [1]\n")
else:
    iface_block = ("DESIGN LINE NEUMANN CONDITIONS:\n  - E: 2\n    NUMDOF: 1\n"
                   "    ONOFF: [1]\n    VAL: [1.0]\n    FUNCT: [1]\n")

deck = f"""TITLE:
  - "openPASO coupling participant (4C scalar transport)"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
  CALCFLUX_DOMAIN: "diffusive"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  OUTPUT_DATA_FORMAT: "ascii"
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: {K}
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{expr}"
{dirich}{iface_block}"""
# The volumetric source: FUNCT2 carries the whole spatial dependence and VAL is
# a plain multiplier of it, so this term varies with position. VAL: [F_SRC] with
# FUNCT: [0] — what stood here while F_SRC was a number — can only ever be a
# constant, whatever the problem asked for.
if src is not None:
    deck += f'FUNCT2:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{src}"\n'
    deck += ("DESIGN SURF NEUMANN CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n"
             "    ONOFF: [1]\n    VAL: [1.0]\n    FUNCT: [2]\n")
deck += "NODE COORDS:\n" + "".join(f'  - "{n}"\n' for n in nodes)
deck += "TRANSPORT ELEMENTS:\n" + "".join(f'  - "{e}"\n' for e in elems)
deck += "DLINE-NODE TOPOLOGY:\n"
outer_nodes = {grid[(i_out, j)] for j in range(NY + 1)}
iface_rows = range(NY + 1)
if FULL_OUTER_DIRICHLET:
    outer_nodes.update(grid[(i, 0)] for i in range(NX + 1))
    outer_nodes.update(grid[(i, NY)] for i in range(NX + 1))
    # At the two interface corners the outer boundary condition wins.
    iface_rows = range(1, NY)
deck += "".join(f'  - "NODE {node} DLINE 1"\n'
                for node in sorted(outer_nodes))
deck += "".join(f'  - "NODE {grid[(i_if, j)]} DLINE 2"\n'
                for j in iface_rows)
if src is not None:
    deck += "DSURF-NODE TOPOLOGY:\n" + "".join(
        f'  - "NODE {n} DSURFACE 1"\n' for n in range(1, len(nodes) + 1))
Path("input.4C.yaml").write_text(deck)

env = dict(os.environ)
if FOURC_LD:
    env["LD_LIBRARY_PATH"] = FOURC_LD + os.pathsep + env.get("LD_LIBRARY_PATH", "")
# stdbuf IS NOT OPTIONAL. 4C buffers stdout and then calls MPI_Abort, which
# kills the process before the buffer flushes — so its real error ("Inconsistency
# is detected at LINE DBC 2", "could not find ':' colon after key") is LOST and
# all you see is an MPI failure. Five coupled runs concluded from
# that silence that "4C cannot run under subprocess" and gave up; re-running
# their decks with stdbuf printed an ordinary deck bug every time. openPASO's own
# run_simulation path already does this.
shutil.rmtree("out-vtk-files", ignore_errors=True)
for stale in Path(".").glob("out*"):
    if stale.is_file():
        stale.unlink()
r = subprocess.run(["stdbuf", "-oL", "-eL", FOURC_BIN, "input.4C.yaml", "out"],
                   capture_output=True, text=True, env=env)
sys.stdout.write(r.stdout)
sys.stderr.write(r.stderr)
if r.returncode != 0:
    sys.exit(f"4C exited with code {r.returncode}; refusing stale output")

vtus = sorted(Path("out-vtk-files").glob("scatra-*-0.vtu"))
if not vtus:
    sys.exit(f"4C produced no VTU (rc={r.returncode}). "
             f"stdout tail:\n{r.stdout[-2000:]}")


def step(p):
    # scatra-00001-0.vtu -> 1. The TRAILING number is the MPI RANK, not the
    # step: matching the last number returns the initial condition, silently.
    m = re.match(r"scatra-(\d+)-\d+\.vtu$", p.name)
    return int(m.group(1)) if m else -1


import meshio                                              # noqa: E402
m = meshio.read(str(max(vtus, key=step)))
pts = np.asarray(m.points)[:, :2]
phi = np.asarray(m.point_data["phi_1"]).ravel()             # NOT 'temperature'
flux = m.point_data.get("flux_domain_phi_1")
if flux is None:
    sys.exit("no flux field in the 4C VTU — set CALCFLUX_DOMAIN: \"diffusive\"")
flux = np.asarray(flux)

# Persist the complete solved field by physical coordinates. The VTU repeats
# nodes per element, so collapse every field exactly as the interface is
# collapsed below; array position is never a mesh-node identifier in 4C VTK.
field_coords, field_inv = np.unique(np.round(pts, 10), axis=0,
                                    return_inverse=True)
field_values = np.zeros(len(field_coords))
field_fluxes = np.zeros((len(field_coords), flux.shape[1]))
field_counts = np.zeros(len(field_coords))
np.add.at(field_values, field_inv, phi)
np.add.at(field_fluxes, field_inv, flux)
np.add.at(field_counts, field_inv, 1.0)
field_values /= field_counts
field_fluxes /= field_counts[:, None]
Path("field.json").write_text(json.dumps({
    "coordinates": field_coords.tolist(),
    "values": field_values.tolist(),
    "flux_vectors": field_fluxes[:, :2].tolist(),
}, indent=2))

# Recover the variationally consistent interface flux from the 4C solution.
# `flux_domain_phi_1` is an L2 projection of element gradients and is only
# first-order accurate at a Dirichlet boundary. The constrained-row residual
# of the original Q1 problem is the boundary flux functional and converges at
# second order on interior interface nodes.
xs = np.linspace(X0, X1, NX + 1)
ys_grid = np.linspace(Y0, Y1, NY + 1)
u_grid = np.full((NY + 1, NX + 1), np.nan)
for point, value in zip(field_coords, field_values):
    i = int(np.argmin(np.abs(xs - point[0])))
    j = int(np.argmin(np.abs(ys_grid - point[1])))
    u_grid[j, i] = value
if not np.isfinite(u_grid).all():
    sys.exit("4C field output is not a complete structured Q1 grid")

hx = (X1 - X0) / NX
hy = (Y1 - Y0) / NY
mass = hx * hy / 36.0 * np.array([
    [4, 2, 1, 2], [2, 4, 2, 1],
    [1, 2, 4, 2], [2, 1, 2, 4]], float)
stiffness = np.zeros((4, 4))
gauss = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
for xi in gauss:
    for eta in gauss:
        dxi = 0.25 * np.array([-(1-eta), 1-eta, 1+eta, -(1+eta)])
        deta = 0.25 * np.array([-(1-xi), -(1+xi), 1+xi, 1-xi])
        grad = np.column_stack((2.0 / hx * dxi, 2.0 / hy * deta))
        stiffness += K * (grad @ grad.T) * hx * hy / 4.0

residual = np.zeros_like(u_grid)
for j in range(NY):
    for i in range(NX):
        indices = ((j, i), (j, i+1), (j+1, i+1), (j+1, i))
        local_u = np.array([u_grid[jj, ii] for jj, ii in indices])
        local_f = np.array([F_SRC(xs[ii], ys_grid[jj])
                            for jj, ii in indices], float)
        local_r = stiffness @ local_u - mass @ local_f
        for (jj, ii), value in zip(indices, local_r):
            residual[jj, ii] += value

weights = np.full(NY + 1, hy)
weights[[0, -1]] = 0.5 * hy
q_consistent = -residual[:, i_if] / weights
if FULL_OUTER_DIRICHLET:
    # Corner reactions also contain the perpendicular outer-boundary flux and
    # cannot be separated into one interface contribution, so zero this mixed
    # reaction at an interface/outer corner rather than report it as flux.
    q_consistent[[0, -1]] = 0.0
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

mask = np.abs(pts[:, 0] - IFACE_X) < 1e-9
if not mask.any():
    sys.exit(f"no 4C nodes at x={IFACE_X}: this subdomain spans [{X0},{X1}]")
# A 4C VTU repeats every node once per element (QUAD4 -> 4 copies). Collapse
# the duplicates by coordinate or the export length is 4x too long.
uy, inv = np.unique(np.round(pts[mask, 1], 10), return_inverse=True)
T = np.zeros(len(uy)); Q = np.zeros(len(uy)); n = np.zeros(len(uy))
np.add.at(T, inv, phi[mask])
np.add.at(Q, inv, flux[mask][:, 0])
np.add.at(n, inv, 1.0)
T /= n
Q_projected = S * (Q / n)
Q = q_consistent

print(f"[4C {SIDE}] interface n={len(uy)} "
    f"T=[{T.min():.6g},{T.max():.6g}] q=[{Q.min():.6g},{Q.max():.6g}] "
    f"projected_q=[{Q_projected.min():.6g},{Q_projected.max():.6g}]")
print(f"\nNDOF = {len(field_coords)}")

# ── EXPORT SELF-CHECK ─ keep this block. It stops the three exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
#    a flux of ~0 against a nonzero partner); and a flux that is the partner's
#    array negated instead of a recovery from THIS side's own system.
_chk_vals = np.asarray(T, float).ravel()
_chk_flux = np.asarray(Q, float).ravel()
if not (np.isfinite(_chk_vals).all() and np.isfinite(_chk_flux).all()):
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface values or fluxes; "
                     "the solve did not produce a usable field, so nothing was "
                     "exported")
_chk_imp = (json.loads(Path("imports.json").read_text() or "{}")
            if Path("imports.json").is_file() else {})
_chk_qin = (np.concatenate([np.asarray(_d.get("normal_fluxes") or [], float).ravel()
                            for _d in _chk_imp.values()])
            if _chk_imp else np.zeros(0))
if SIDE == "neumann" and _chk_qin.size and np.abs(_chk_qin).max() > 0 \
        and np.abs(_chk_flux).max() < 1e-9 * np.abs(_chk_qin).max():
    raise SystemExit("EXPORT SELF-CHECK: the recovered interface flux is ~0 "
                     "against a nonzero imported flux: the imported load never "
                     "entered the assembled system (the facet term / boundary "
                     "condition that integrates it is missing). Fix the "
                     "application; do not couple on")
# (Dirichlet role only: a Neumann side's consistent recovery of a CONSTANT
#  applied flux can legitimately reproduce it to the last bit.)
if SIDE == "dirichlet" and _chk_qin.shape == _chk_flux.shape and _chk_flux.size \
        and np.array_equal(_chk_flux, -_chk_qin):
    raise SystemExit("EXPORT SELF-CHECK: the exported flux is the partner's "
                     "array negated, bit for bit: a copy, not a recovery from "
                     "this side's own assembled system")

Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(uy)),
    "coordinates": [[float(IFACE_X), float(y)] for y in uy],
    "values": [float(t) for t in T],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))
