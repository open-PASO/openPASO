"""4C STRUCTURE participant for the openPASO `couple` driver — FSI.

Drop-in replacement for the scikit-fem structure participant: same
imports.json / exports.json contract, same physics, same sign convention.

4C is a compiled YAML-in / VTU-out code with no Python API, so a participant is
a small Python WRAPPER: write the deck from imports.json, run the 4C binary,
read the VTU back, write exports.json.  It runs under the ORDINARY venv python
(needs `numpy` and `meshio`), NOT inside 4C.

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

Physics: plane-strain linear elasticity on a rectangular wall clamped at both
ends.  The bottom edge is the FSI interface and carries the traction handed in
by the fluid participant as a NEUMANN load.  What is exported back is the
interface displacement.

SIGN CONVENTION — the imported traction is already the load ON THIS BODY
(t = sigma_f . n_s, n_s the structure's outward normal on the interface).  It is
applied DIRECTLY, with NO further sign change.  See the fluid participant's
docstring for the derivation; re-deriving it on this side is how the sign gets
flipped twice.

The interface parametrisation is LAGRANGIAN on both sides: the exported
coordinates are the REFERENCE (undeformed) interface node positions.  That is
also what 4C integrates the load over — `TYPE: "Live"` on a line Neumann is 4C's
*standard* Neumann load, evaluated on the MATERIAL configuration
(4C_w1_line_evaluate.cpp: `case neum_live: // uniform load on reference
configuration`, which sets `loadlin = false` and uses the material coords `xye`).
The confusing name does not mean a follower load; `orthopressure` is the
follower one.

HOW A SPATIALLY VARYING TRACTION GETS INTO 4C
---------------------------------------------
A 4C Neumann datum is VAL[i] x FUNCT[i](x,y,z,t), and FUNCT is a SYMBOLIC
EXPRESSION, not a table.  Two components therefore need two expressions.  Both
of these work and were measured to give bit-identical reaction forces:

    FUNCT: [1, 2]   with FUNCT1 and FUNCT2 each holding one
                    SYMBOLIC_FUNCTION_OF_SPACE_TIME             <- used here
    FUNCT: [1, 1]   with FUNCT1 holding COMPONENT: 0 and COMPONENT: 1

The first is used because it cannot be got wrong: 4C passes the DOF index as the
function's component index, and `find_modified_component()` collapses that to 0
whenever the function holds exactly one expression, so a one-expression FUNCT is
unambiguous on either DOF.

A BAD FIT IS A SILENTLY WRONG BC, NOT AN ERROR, so the profile is not fitted at
all by default.  TRACTION_FIT="exact" emits the piecewise-linear interpolant of
the handed-in samples EXACTLY, as a sum of ReLU terms

    t(x) = t_0 + sum_k c_k * (x - x_k) * heaviside(x - x_k)

`heaviside` is in 4C's symbolic-expression grammar (4C_utils_symbolic_expression
.hpp lists acos asin atan cos sin tan cosh sinh tanh exp log log10 sqrt
heaviside fabs atan2).  This reproduces the samples to machine precision AND is
the same interpolant `np.interp` gives, so the load is identical to the one a
scikit-fem participant applies — the two codes can then be compared as
discretisations rather than as two different loads.

TRACTION_FIT="poly" keeps the least-squares polynomial route for reference.  It
is worse than it looks: on a measured fluid traction a cubic missed the peak by
13% of the traction range.  Whatever the mode, the max deviation between the
emitted FUNCT and the handed-in samples is printed and put in exports meta ->
`fit`, because that number is the only thing standing between you and a load you
never noticed was wrong.

WHAT DOES NOT WORK — 4C's DBC REACTION MONITOR IN 2-D
-----------------------------------------------------
`TAG: "monitor_reaction"` on a DIRICH condition plus an `IO/MONITOR STRUCTURE
DBC` section is the only way to make 4C report an interface force back, and both
keys are real.  Solid::MonitorDbc is nevertheless 3-D only — `DIM` is a
hard-coded 3 — and it fails on a 2-D structure in two different ways:

  * it asks every monitored node for its THIRD dof.  A 2-D node has two, so the
    request lands on the next node's dof; for the node holding the highest gid
    there is no next node and 4C aborts with
        Cannot find gid=<2*n_nodes> in Core::LinAlg::Vector<double>
    in Core::FE::extract_values, called from MonitorDbc::get_reaction_moment.
    Numbering the mesh so that a non-monitored node comes last dodges this.
  * its reference-area computation reads the same uninitialised memory and
    prints `ref_area` as garbage — 7.96e+88 on one run, `inf` on the next.  When
    it comes out `inf`, 4C's floating-point trap fires on the very next
    evaluation and the run dies in the PREDICTOR, before any solve:
        Floating Point Exception: OVERFLOW
        ERROR - NOX::Nln::Group::compute_f_and_jacobian - evaluation failed!
    This is non-deterministic: measured 2 aborts in 8 runs of one deck, 0 in 20
    runs of the same deck.  A default that fails one run in four is not a
    default, hence MONITOR_REACTION=False.

The reaction FORCE path itself is sound: when the monitor did run, the two clamp
reactions summed to the applied Neumann resultant to 12 significant digits.  So
MONITOR_REACTION=True is a good one-off conservation check and a bad standing
setting.
"""
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
PARTNER    = "fluid"     # the fluid participant's `name` in your couple(...) call
LX         = 1.2         # wall length
Y0         = 0.18        # the FSI interface (this body's LOWER edge)
HS         = 0.04        # wall thickness
NXS, NYS   = 46, 3       # this body's OWN QUAD4 mesh; need not match the fluid's;
                         # config.json's nx, ny set it per level (below)
E_MOD      = 1.5e6       # Young's modulus
NU         = 0.35       # Poisson ratio
CLAMP_X    = (0.0, 1.2)  # x positions of the clamped ends; each must land on a
                         # mesh column, i.e. be a multiple of LX/NXS
T_INIT     = 0.0         # iteration-1 fallback interface traction (both comps)
FEEDBACK   = True        # SET False ONLY to suppress the fluid->structure
                         # direction (freezes the load at T_INIT). A real FSI
                         # run keeps this True.
FOURC_BIN  = "4C"        # the 4C binary path `discover(query='list')` prints
FOURC_LD   = ""          # 4C dependency lib dir, or "" to inherit the env
TRACTION_FIT = "exact"   # "exact" = piecewise-linear heaviside form (no fit
                         # error); "poly" = least-squares polynomial of FIT_DEG
FIT_DEG    = 3           # only used by TRACTION_FIT="poly"
MONITOR_REACTION = False  # ask 4C for the clamp reaction forces, so the applied
                         # interface force is checked against 4C's own numbers.
                         # OFF by default because Solid::MonitorDbc is 3-D ONLY
                         # and fails two ways on a 2-D structure — see the
                         # WHAT DOES NOT WORK note below. The reaction FORCES it
                         # produces are correct when it runs, so turn it on for
                         # a one-off conservation check; the wrapper retries
                         # without it if 4C aborts, so it cannot lose you a run,
                         # only a second.
# ─────────────────────────────────────────────────────────────────────────


# ── THE PER-LEVEL RULE (served). A ./config.json {"level": k, "nx": .., "ny": ..} next to this
#    script, and the keys a multi-level couple call hands in OPENPASO_CONFIG_JSON (they win), set
#    NXS, NYS and name the level; the per-level dumps below carry that level, so the coarse levels
#    survive the fine ones. A config that cannot be read stops the run: a level run on the edit
#    block's mesh would hand in the wrong level.
LEVEL = 1
try:
    _cfg = json.loads(Path("config.json").read_text() or "{}") if Path("config.json").is_file() else {}
    _cfg.update(**json.loads(os.environ.get("OPENPASO_CONFIG_JSON") or "{}"))
    LEVEL = int(_cfg.get("level", LEVEL))
    NXS, NYS = int(_cfg.get("nx", NXS)), int(_cfg.get("ny", NYS))
except (ValueError, TypeError, AttributeError) as _cfg_exc:
    raise SystemExit(f"config.json / OPENPASO_CONFIG_JSON: the level and the mesh keys nx, ny could "
                     f"not be read ({_cfg_exc!r}); fix them, nothing was run")

GAUSS2 = (np.array([-1.0, 1.0]) / np.sqrt(3.0), np.array([1.0, 1.0]))



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


def import_samples(imp, fallback, ncomp=2):
    """(xs, vals) of the handed-in traction, sorted along the interface
    parameter x.  The driver does no interpolation: non-matching interface
    meshes are handled here."""
    if not imp or not imp.get("coordinates"):
        xs = np.array([0.0, float(LX)])
        return xs, np.full((2, ncomp), float(fallback))
    xs = np.asarray(imp["coordinates"], float)[:, 0]
    vals = np.asarray(imp["values"], float).reshape(len(xs), -1)
    if vals.shape[1] < ncomp:
        vals = np.pad(vals, ((0, 0), (0, ncomp - vals.shape[1])))
    # Collapse repeated x. A repeated interface coordinate is a zero-length
    # segment, and the piecewise-linear form divides by segment length: one
    # duplicate turns the whole FUNCT into inf and 4C then dies in the predictor
    # with "Floating Point Exception: OVERFLOW", far from the cause.
    xu, inv = np.unique(np.round(xs, 12), return_inverse=True)
    acc = np.zeros((len(xu), ncomp)); cnt = np.zeros(len(xu))
    np.add.at(acc, inv, vals[:, :ncomp]); np.add.at(cnt, inv, 1.0)
    return xu, acc / cnt[:, None]


# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
def pwlin_expr(xs, vs):
    """EXACT symbolic form of np.interp(x, xs, vs), end values held flat
    outside [xs[0], xs[-1]] just as np.interp does."""
    n = len(xs)
    if n < 2 or float(np.ptp(vs)) == 0.0:
        return f"({float(vs[0]):.16e})"
    b = np.diff(vs) / np.diff(xs)                 # segment slopes
    c = np.zeros(n)
    c[0] = b[0]
    c[1:n - 1] = b[1:] - b[:-1]                   # slope increments at the kinks
    c[n - 1] = -b[-1]                             # kill the slope past the end
    terms = [f"({float(vs[0]):.16e})"]
    terms += [f"({c[k]:.16e})*(x-({xs[k]:.16e}))*heaviside(x-({xs[k]:.16e}))"
              for k in range(n) if c[k] != 0.0]
    return "+".join(terms)


def poly_expr(xs, vs, deg):
    if float(np.ptp(vs)) == 0.0:
        return f"({float(vs[0]):.16e})"
    c = np.polyfit(xs, vs, int(min(deg, len(xs) - 1)))[::-1]
    return "+".join(f"({v:.16e})*x^{i}" if i else f"({v:.16e})"
                    for i, v in enumerate(c))
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end


def eval_expr(e, x):
    """Evaluate the emitted 4C expression TEXT, so the reported fit quality is a
    statement about the string that went into the deck and not about the arrays
    it was built from."""
    x = np.asarray(x, float)
    v = eval(e.replace("^", "**"), {"__builtins__": {}},
             {"x": x, "heaviside": lambda a: np.where(np.asarray(a) >= 0.0, 1.0, 0.0),
              "y": np.zeros_like(x), "z": np.zeros_like(x), "t": 0.0})
    # BROADCAST. A CONSTANT expression — which is exactly what the iteration-1
    # fallback traction produces — contains no `x`, so `eval` returns a SCALAR
    # and every caller silently gets one value instead of len(x) of them. That
    # collapsed the exported `normal_fluxes` to a single point on iteration 1
    # and to len(x) points on iteration 2, and the coupling died at iteration 2
    # with "participant solid changed its export length from 84 to 164". The
    # driver caught it; a standalone run of this script could not, because the
    # export it produces is well-formed and its DISPLACEMENT array is the right
    # length either way.
    return np.broadcast_to(np.asarray(v, float), x.shape).copy()


# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
def build_deck(exprs, monitor):
    """Inline QUAD4 mesh + deck.  Node ids are ordered so that the HIGHEST node
    id is an interior node: Solid::MonitorDbc hard-codes DIM=3 and asks every
    monitored node for its third DOF, which only exists for a 2-D node if some
    later node owns that gid.  With a clamped node last, 4C aborts with
    "Cannot find gid=<2N> in Core::LinAlg::Vector<double>"."""
    hx, hy = LX / NXS, HS / NYS
    ij = [(i, j) for j in range(NYS + 1) for i in range(NXS + 1)]
    if monitor and NXS >= 2:
        ij.remove((NXS, NYS))
        ij.insert(0, (NXS, NYS))
    grid, xy = {}, {}
    for nid, (i, j) in enumerate(ij, start=1):
        grid[(i, j)] = nid
        xy[nid] = (i * hx, Y0 + j * hy)
    nodes = [f"NODE {n} COORD {xy[n][0]:.16e} {xy[n][1]:.16e} 0.0" for n in sorted(xy)]
    elems = [f"{k + 1} WALL QUAD4 {grid[(i, j)]} {grid[(i + 1, j)]} "
             f"{grid[(i + 1, j + 1)]} {grid[(i, j + 1)]} MAT 1 KINEM linear "
             f"EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 2 2"
             for k, (i, j) in enumerate(
                 [(i, j) for j in range(NYS) for i in range(NXS)])]

    cols = []
    for cx in CLAMP_X:
        i = int(round(float(cx) / hx))
        if not (0 <= i <= NXS and abs(i * hx - float(cx)) < 1e-9 * max(LX, 1.0)):
            sys.exit(f"CLAMP_X={cx} is not on a mesh column: with NXS={NXS} on "
                     f"[0,{LX}] the columns sit at multiples of {hx:g}")
        cols.append(i)

    tag = '\n    TAG: "monitor_reaction"' if monitor else ""
    dirich = "DESIGN LINE DIRICH CONDITIONS:\n" + "".join(
        f"  - E: {2 + k}\n    NUMDOF: 2\n    ONOFF: [1, 1]\n"
        f"    VAL: [0.0, 0.0]\n    FUNCT: [0, 0]{tag}\n" for k in range(len(cols)))
    mon = ("IO/MONITOR STRUCTURE DBC:\n  INTERVAL_STEPS: 1\n  FILE_TYPE: csv\n"
           if monitor else "")

    deck = f"""TITLE:
  - "openPASO FSI structure participant (4C plane-strain linear elasticity)"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  INT_STRATEGY: "Standard"
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1e-12
  NORM_DISP: "Abs"
  TOLRES: 1e-08
  NORM_RESF: "Rel"
  NORMCOMBI_RESFDISP: "Or"
  MAXITER: 25
  PREDICT: "TangDis"
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
IO:
  VERBOSITY: "Standard"
{mon}IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  OUTPUT_DATA_FORMAT: "ascii"
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: {E_MOD}
      NUE: {NU}
      DENS: 0.0
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{exprs[0]}"
FUNCT2:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{exprs[1]}"
{dirich}DESIGN LINE NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [1.0, 1.0]
    FUNCT: [1, 2]
    TYPE: "Live"
"""
    deck += "NODE COORDS:\n" + "".join(f'  - "{n}"\n' for n in nodes)
    deck += "STRUCTURE ELEMENTS:\n" + "".join(f'  - "{e}"\n' for e in elems)
    deck += "DLINE-NODE TOPOLOGY:\n"
    deck += "".join(f'  - "NODE {grid[(i, 0)]} DLINE 1"\n' for i in range(NXS + 1))
    for k, i in enumerate(cols):
        deck += "".join(f'  - "NODE {grid[(i, j)]} DLINE {2 + k}"\n'
                        for j in range(NYS + 1))
    return deck, len(nodes)


def run_4c(deck, outprefix):
    Path("input.4C.yaml").write_text(deck)
    env = dict(os.environ)
    if FOURC_LD:
        env["LD_LIBRARY_PATH"] = FOURC_LD + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    # `stdbuf -oL` is not decoration: 4C aborts through MPI_Abort, which kills the
    # process before a block-buffered stdout is flushed, and the silence reads
    # exactly like a clean run.
    return subprocess.run(["stdbuf", "-oL", FOURC_BIN, "input.4C.yaml", outprefix],
                          capture_output=True, text=True, env=env)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end


def neumann_resultant(exprs):
    """The net interface force 4C receives, integrated with 4C's OWN rule: the
    QUAD4 boundary is a line2, and Wall1Line::get_optimal_gaussrule gives it
    line_2point on the reference configuration."""
    g, w = GAUSS2
    xe = np.linspace(0.0, LX, NXS + 1)
    out = []
    for e in exprs:
        s = 0.0
        for a, b in zip(xe[:-1], xe[1:]):
            s += float(np.sum(w * 0.5 * (b - a)
                              * eval_expr(e, 0.5 * (a + b) + 0.5 * (b - a) * g)))
        out.append(s)
    return out


# ── HOLE NAMES IN WORDS ─ begin
#   pwlin_expr  a function (xs, vs) returning the text of one 4C
#          SYMBOLIC_FUNCTION_OF_SPACE_TIME expression in x that equals the
#          piecewise-linear interpolant of the samples vs at the sorted points
#          xs, held flat past both ends (the form the docstring writes out);
#          the served lines put it in FUNCT1 / FUNCT2 and read its text back
#          at the samples to report the fit
#   poly_expr  a function (xs, vs, deg) returning the text of the
#          least-squares polynomial in x of degree deg through the samples,
#          for TRACTION_FIT = "poly"
#   monitor  True when the 4C run that finished kept the reaction monitor
#          (MONITOR_REACTION, and 4C accepted it), else False: the served
#          lines read the monitor csv only then
#   n_nodes  the number of nodes of the deck's mesh; NDOF is 2 * n_nodes
#   wall   the wall time of the 4C run in seconds, a float
#   r      the finished 4C run, what subprocess.run(..., capture_output=True,
#          text=True) returned: its stdout, 4C's own console, is echoed below
#          as this side's run log
#   pts    the point coordinates of the last step's VTU,
#          out-vtk-files/structure-<step>-0.vtu, an (m, 2) array of x, y with
#          one row per VTU point (a node repeats once per element)
#   dsp    the point array `displacement` of that VTU, its first two
#          columns, an (m, 2) array in the rows of pts
# ── HOLE NAMES IN WORDS ─ end


def main():
    imp = read_imports() if FEEDBACK else None
    xs, vals = import_samples(imp, T_INIT, ncomp=2)

    if TRACTION_FIT == "exact":
        exprs = [pwlin_expr(xs, vals[:, c]) for c in (0, 1)]
    elif TRACTION_FIT == "poly":
        exprs = [poly_expr(xs, vals[:, c], FIT_DEG) for c in (0, 1)]
    else:
        sys.exit(f'TRACTION_FIT must be "exact" or "poly", got {TRACTION_FIT!r}')

    # ── fit quality: the emitted FUNCT text, re-evaluated at the points the
    #    fluid actually handed over, against the values it handed over. A BAD
    #    FIT IS A SILENTLY WRONG BC, NOT AN ERROR, so this number is reported
    #    whether it is good or not.
    fit = {"mode": TRACTION_FIT, "n_samples": int(len(xs))}
    for c, nm in ((0, "tx"), (1, "ty")):
        d = np.abs(eval_expr(exprs[c], xs) - vals[:, c])
        rng = float(np.ptp(vals[:, c])) or float(np.max(np.abs(vals[:, c]))) or 1.0
        fit[nm] = {"max_abs_dev": float(d.max()), "max_rel_dev": float(d.max() / rng)}

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    monitor = bool(MONITOR_REACTION) and NXS >= 2
    deck, n_nodes = build_deck(exprs, monitor)
    t0 = time.time()
    r = run_4c(deck, "out")
    if r.returncode != 0 and monitor:
        # Solid::MonitorDbc is 3-D only and dies non-deterministically in 2-D
        # (see the module docstring). Losing the equilibrium check is better than
        # losing the solve, so retry once without it — but say so.
        print("[4C solid] reaction monitor refused by 4C, retrying without it. "
              f"4C said: {_first_error(r.stdout)}", flush=True)
        monitor = False
        deck, n_nodes = build_deck(exprs, monitor)
        r = run_4c(deck, "out")
    wall = time.time() - t0
    if r.returncode != 0:
        sys.exit(f"4C failed (rc={r.returncode}).\nstdout tail:\n{r.stdout[-4000:]}")

    vtus = sorted(Path("out-vtk-files").glob("structure-*-0.vtu"))
    if not vtus:
        sys.exit(f"4C produced no VTU (rc={r.returncode}). "
                 f"stdout tail:\n{r.stdout[-2000:]}")

    def step(p):
        # structure-00001-0.vtu -> 1. The TRAILING number is the MPI RANK, not
        # the step: matching the last number returns the initial condition,
        # silently.
        m = re.match(r"structure-(\d+)-\d+\.vtu$", p.name)
        return int(m.group(1)) if m else -1

    import meshio                                            # noqa: E402
    m = meshio.read(str(max(vtus, key=step)))
    pts = np.asarray(m.points)[:, :2]
    dsp = np.asarray(m.point_data["displacement"])[:, :2]
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    # 4C's OWN CONSOLE, ECHOED (served): this script's stdout becomes the level's run log, and a run
    # log is credited to 4C only by 4C's own lines, never by this script's summary. Measured on
    # coupled runs: this contract printed its summary line alone, and three runs rebuilt their run
    # logs by hand (one of them took 34 run-log findings and four minutes).
    sys.stdout.write(r.stdout if isinstance(getattr(r, "stdout", None), str) else "")

    mask = np.abs(pts[:, 1] - Y0) < 1e-9
    if not mask.any():
        sys.exit(f"no 4C nodes at y={Y0}: this wall spans [{Y0},{Y0 + HS}]")
    # A 4C VTU repeats every node once per element (QUAD4 -> up to 4 copies).
    # Collapse the duplicates by coordinate or the export length is 4x too long.
    ux, inv = np.unique(np.round(pts[mask, 0], 10), return_inverse=True)
    D = np.zeros((len(ux), 2)); cnt = np.zeros(len(ux))
    np.add.at(D, inv, dsp[mask]); np.add.at(cnt, inv, 1.0)
    D /= cnt[:, None]

    fx, fy = neumann_resultant(exprs)
    meta = {
        "net_force_received": [fx, fy],
        "feedback": bool(FEEDBACK),
        "max_abs_disp": [float(np.max(np.abs(D[:, 0]))), float(np.max(np.abs(D[:, 1])))],
        "n_dofs": 2 * int(n_nodes),
        "fit": fit,
        "solve_wall_time_s": float(wall),
    }

    # ── what 4C ITSELF says the interface force was ─────────────────────────
    # Global equilibrium: the reactions at the clamps must balance the Neumann
    # load. This is the one number here that 4C produces rather than the wrapper,
    # so it is the check that catches a load 4C did not read the way we meant.
    if monitor:
        rf = np.zeros(2)
        got = False
        for p in sorted(Path(".").glob("*_monitor_dbc.csv")):
            rows = list(csv.DictReader(p.read_text().splitlines()))
            if rows:
                rf += [float(rows[-1]["f:0"]), float(rows[-1]["f:1"])]
                got = True
        if got:
            den = float(np.linalg.norm([fx, fy])) or 1.0
            meta["net_reaction_4c"] = [float(rf[0]), float(rf[1])]
            meta["equilibrium_residual"] = float(
                np.linalg.norm(rf - np.array([fx, fy])) / den)

    # `normal_fluxes` = the traction this body ACTUALLY applied, sampled at its
    # own nodes, w.r.t. THIS body's own outward normal n_s, so that this and the
    # fluid's `normal_fluxes` (w.r.t. n_f = -n_s) must SUM to zero. It is read
    # back out of the emitted FUNCT text, i.e. out of the deck 4C consumed.
    #
    # It is NOT the traction recovered from the structure's own stress field.
    # That would be independent, but it does not converge on a bending structure
    # with clamped ends: the corners where the Dirichlet boundary meets the
    # loaded face carry a genuine stress singularity, so sigma_s . n_s there
    # diverges under refinement and a check built on it would fault a coupling
    # that is right.
    t_applied = np.column_stack([eval_expr(exprs[0], ux), eval_expr(exprs[1], ux)])

    # THE RUN-LOG CONTRACT LINE: `NDOF = <integer>` on a line of its OWN, printed
    # PER LEVEL. It is how anyone checking the result tells a refined mesh from the same mesh run
    # three times. The LEADING NEWLINE is deliberate: a program that writes without
    # a trailing newline glues its text onto the front of the next line.
    try:
        print(f"\nNDOF = {2 * int(n_nodes)}")          # the whole structure's dofs
    except Exception as _ndof_exc:
        print(f"[fsi-solid] could not report NDOF: {_ndof_exc!r}. Your task's execution"
              f" log needs `NDOF = <integer>` on a line of its own, so print your"
              f" own degree-of-freedom count here.")

    # ── EXPORT SELF-CHECK ─ keep this block. TWO of the three checks, and the
    #    third DELIBERATELY LEFT OUT. A non-finite export is worthless, and so
    #    is a structure that returns ~0 displacement while the fluid handed it a
    #    real traction: that is the load never having entered the assembled
    #    system, and it still couples, still converges and still hands in tidy
    #    levels. The "exported flux is the partner's array negated" check that
    #    the scalar contracts carry does NOT belong here: in FSI the two sides'
    #    tractions are anti-parallel BY CONVENTION and must sum to zero, so that
    #    check would fault a correct export.
    _chk_d = np.asarray(D, float).ravel()
    _chk_t = np.asarray(t_applied, float).ravel()
    if not (np.isfinite(_chk_d).all() and np.isfinite(_chk_t).all()):
        raise SystemExit("EXPORT SELF-CHECK: non-finite interface displacement "
                         "or traction; the solve did not produce a usable "
                         "field, so nothing was exported")
    _chk_imp = (json.loads(Path("imports.json").read_text() or "{}")
                if Path("imports.json").is_file() else {})
    _chk_tin = (np.concatenate([np.asarray(_d.get("normal_fluxes") or [], float).ravel()
                                for _d in _chk_imp.values()])
                if _chk_imp else np.zeros(0))
    if _chk_tin.size and np.abs(_chk_tin).max() > 0 \
            and np.abs(_chk_d).max() < 1e-12 * np.abs(_chk_tin).max():
        raise SystemExit("EXPORT SELF-CHECK: the structure returns a ~0 "
                         "displacement against a nonzero imported traction: "
                         "the fluid load never entered the assembled system "
                         "(the Neumann condition that applies it is missing or "
                         "on the wrong surface). Fix the application; do not "
                         "couple on")


    # PER-LEVEL PERSISTENCE: this level's whole displacement field, at this body's nodes, and its
    # interface data -- the displacement and the traction applied there -- named by LEVEL and never
    # overwritten by the next level (exports.json is). Interpolate THESE onto the points your task
    # names. A DUMP DEFECT MUST NOT COST THE SOLVE: exports.json is written after them either way,
    # and a failed dump keeps neither file.
    _dumps = (f"field_level{LEVEL}.csv", f"interface_level{LEVEL}.csv")
    try:
        # a 4C VTU repeats each node once per element: one row per node
        _, _first = np.unique(np.round(pts, 10), axis=0, return_index=True)
        with open(_dumps[0], "w") as _f:
            _f.write("x,y,ux,uy\n")
            for (_px, _py), (_ux, _uy) in zip(pts[_first], dsp[_first]):
                _f.write(f"{_px:.11e},{_py:.11e},{_ux:.11e},{_uy:.11e}\n")
        with open(_dumps[1], "w") as _f:
            _f.write("x,y,ux,uy,tx,ty\n")
            for _px, (_ux, _uy), (_tx, _ty) in zip(ux, D, t_applied):
                _f.write(f"{_px:.11e},{float(Y0):.11e},{_ux:.11e},{_uy:.11e},{_tx:.11e},{_ty:.11e}\n")
    except Exception as _dump_exc:
        for _partial in _dumps:                     # both files or neither, whatever they hold
            try:
                Path(_partial).unlink(missing_ok=True)
            except OSError:
                pass
        print(f"[4C solid per-level dump] level {LEVEL} dump failed: {_dump_exc!r}. exports.json is "
              f"still written, so the coupling continues, but this level has no field file to hand "
              f"in. Fix the names the dump reads and run this level again.")

    out = {
        "field_name": "interface_displacement",
        "n_points": int(len(ux)),
        "coordinates": [[float(x), float(Y0)] for x in ux],
        "values": D.tolist(),
        "normal_fluxes": t_applied.tolist(),
        "meta": meta,
    }
    Path("exports.json").write_text(json.dumps(out, indent=2))
    print(f"[4C solid] recv_force=({fx:.6e},{fy:.6e}) "
          f"max|dy|={np.max(np.abs(D[:, 1])):.6e} "
          f"fit[{TRACTION_FIT}] max_rel_dev tx={fit['tx']['max_rel_dev']:.3e} "
          f"ty={fit['ty']['max_rel_dev']:.3e} "
          f"wall={wall:.2f}s", flush=True)
    if "equilibrium_residual" in meta:
        print(f"[4C solid] 4C reaction=({meta['net_reaction_4c'][0]:.6e},"
              f"{meta['net_reaction_4c'][1]:.6e}) "
              f"|R-F|/|F|={meta['equilibrium_residual']:.3e}", flush=True)
    if max(fit["tx"]["max_rel_dev"], fit["ty"]["max_rel_dev"]) > 1e-3:
        print("[4C solid] WARNING: the FUNCT the deck carries is not the traction "
              "that was handed in. 4C applied the FUNCT, not the samples.",
              flush=True)


def _first_error(stdout):
    lines = stdout.splitlines()
    for i, ln in enumerate(lines):
        if "PROC" in ln and "ERROR" in ln:
            return " | ".join(x.strip() for x in lines[i:i + 2] if x.strip())
    return "(no PROC ERROR line in stdout)"


if __name__ == "__main__":
    sys.exit(main() or 0)
