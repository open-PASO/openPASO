"""deal.II TRANSIENT participant for the openPASO `couple` driver.

Transient conduction  rho_c dT/dt - div(k grad T) + c T = f(x, y, t)  on one
rectangular subdomain, marched over the whole window T_START -> T_END in
N_STEPS steps of the theta scheme per run (THETA = 0.5 is Crank-Nicolson,
second order). CONTRACT (do not change): runs in its work_dir with no
arguments, reads imports.json (written every iteration; it is `{}` on
iteration 1), writes exports.json LAST.

THIS WRAPPER HAS NO HOLE. The solve is the C++ program dealii_side_transient.cc
beside it, whose five marked holes are yours; write_participant_contract(...,
variant='transient') writes it and its CMakeLists.txt next to this file. The
wrapper writes the program's input, builds the program (the build block below),
runs it, and turns its output into exports.json and the per-level dumps.

TIME COUPLING: WAVEFORM. One run exchanges the whole interface trace,

    values[i][n]        = T at interface point i at t^(n+1)
    normal_fluxes[i][n] = the THETA-AVERAGED outward flux density over step n

both (n_points, N_STEPS): the protocol of participant_fenics_transient.py, whose
docstring gives the reasons, and the two interoperate. Both participants must be
given the same T_START, T_END, N_STEPS and THETA: the exchange carries no time
axis, and a partner trace with another number of time levels stops this script.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material, BCs and time
#    window. As shipped this is the LEFT / Dirichlet side.
#    config.json (or OPENPASO_CONFIG_JSON) overrides them where it states them.
SIDE      = "dirichlet"   # "dirichlet" | "neumann"
PARTNER   = "right"       # name of the partner participant in couple(...)
X0, X1    = 0.0, 0.6
Y0, Y1    = 0.0, 0.4
IFACE_AXIS = "x"          # the interface is the line x = IFACE_X ("x") or y = IFACE_X ("y")
IFACE_X   = 0.6
K         = 0.8           # conductivity: a number, or [[kxx, kxy], [kyx, kyy]] when it is anisotropic
RHO_C     = 1.0           # VOLUMETRIC heat capacity rho*c (not c alone)
REACTION  = 0.0           # c in rho_c dT/dt - div(K grad T) + c T = f (0 when there is none)
T_START, T_END = 0.0, 1.0 # the coupling window  ─┐ BOTH participants must be
N_STEPS   = 20            # steps in the window   ├─ given the SAME four numbers
THETA     = 0.5           # 0.5 Crank-Nicolson    ─┘ (1.0 backward Euler, first order)


def F_SRC(x, y, t):
    """Volumetric source f(x, y, t). Zero as shipped, a PLACEHOLDER like every
    number above. x and y are NumPy arrays, t a number; return ONE array of
    their shape (`0.0 * x + c` for a constant). Put your f here, or
    "source_expr" in config.json. The program checks its `source` against
    these samples at every time level and prints SOURCE on when they reach it."""
    return np.zeros_like(x)


def T_INITIAL(x, y):
    """T at T_START. Keep it equal to T_OUTER(x, y, T_START) on the held edges,
    or the first step starts from a jump that Crank-Nicolson answers with
    oscillations."""
    return 0.0 * x + 300.0


def T_OUTER(x, y, t):
    """The value held on the held NON-interface edges at time t."""
    return 0.0 * x + 320.0


# WHICH NON-INTERFACE EDGES ARE HELD IS YOUR PROBLEM'S TO SAY: the edge
# opposite the interface is held; True if the two edges the interface ends on
# are held too, False if they are natural (zero flux). The script refuses to
# run until you set it, and the program checks your hole 4 against it.
FULL_OUTER_DIRICHLET = None  # <-- True or False, FROM YOUR PROBLEM STATEMENT
NX, NY    = 24, 16
Q_INIT    = 0.0           # iteration-1 flux on the Neumann role (the Dirichlet role starts from T_INITIAL)
DEAL_II_DIR = ""          # the deal.II build or install tree discover(query='list') names;
                          # "" leaves it to the DEAL_II_DIR environment variable
# ─────────────────────────────────────────────────────────────────────────

DEGREE = 1                                    # FE_Q degree of the program
DEALII_SRC = "dealii_side_transient.cc"       # the served program, holes filled, beside this file
DEALII_EXE = "./build/dealii_side_transient"  # what the build below makes of it


def _partner_block(imp):
    """The partner's block from imports.json, by the name the driver actually used."""
    if not isinstance(imp, dict) or not imp:
        return None
    if PARTNER in imp:
        return imp[PARTNER] or None
    others = [k for k in imp if isinstance(imp.get(k), dict)]
    if len(others) == 1:     # named differently in couple(...) than here: read it, say so
        print(f"NOTE: PARTNER is {PARTNER!r} but imports.json is keyed {others[0]!r} "
              f"-- reading that block; align PARTNER with the name in your "
              f"couple(...) call.", file=sys.stderr)
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
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return _partner_block(d)


def _expr(value, names):
    """A number, or an expression string in `names` (`^` allowed), as a NumPy callable."""
    if not isinstance(value, str):
        return lambda *a: 0.0 * a[0] + float(value)
    code = compile(value.replace("^", "**"), "<config expression>", "eval")
    env = {n: getattr(np, n) for n in ("sin", "cos", "exp", "sqrt", "abs", "log", "tanh", "cosh", "sinh", "pi")}
    return lambda *a: eval(code, {"__builtins__": {}}, dict(env, **dict(zip(names, a)))) + 0.0 * a[0]


# ── THE PER-LEVEL RULE AND THE PROBLEM'S DATA (served). ./config.json (and the
#    OPENPASO_CONFIG_JSON a multi-level call hands over) may carry "level", "nx",
#    "ny", "n_steps" -- a space-time study refines dt with h, so each level
#    carries its own n_steps, and the partner must be given the same -- and this
#    subdomain's data as the task writes them: side, partner; x0, x1, y0, y1;
#    iface ("left"|"right"|"bottom"|"top", or the interface coordinate) and
#    iface_axis; k; rho_c; reaction; t_start, t_end, theta;
#    full_outer_dirichlet; outer (a number, or a string in x, y and t);
#    initial (a number, or a string in x and y); source_expr (a string in x, y
#    and t, `^` allowed). They override the constants above.
LEVEL = 1
try:
    _cfg = json.loads(Path("config.json").read_text() or "{}") if Path("config.json").is_file() else {}
    _cfg.update(json.loads(os.environ.get("OPENPASO_CONFIG_JSON") or "{}"))
    LEVEL, NX, NY = int(_cfg.get("level", LEVEL)), int(_cfg.get("nx", NX)), int(_cfg.get("ny", NY))
    N_STEPS = int(_cfg.get("n_steps", N_STEPS))
except (ValueError, TypeError, AttributeError):
    _cfg = _cfg if isinstance(locals().get("_cfg"), dict) else {}
if all(_k in _cfg for _k in ("x0", "x1", "y0", "y1")):
    X0, X1, Y0, Y1 = (float(_cfg[_k]) for _k in ("x0", "x1", "y0", "y1"))
if str(_cfg.get("iface_axis", "")).strip().lower()[:1] in ("x", "y"):
    IFACE_AXIS = str(_cfg["iface_axis"]).strip().lower()[:1]
_ifc = str(_cfg.get("iface", "")).strip().lower()
if _ifc in ("left", "right", "bottom", "top"):
    IFACE_AXIS = "x" if _ifc in ("left", "right") else "y"
    IFACE_X = {"left": X0, "right": X1, "bottom": Y0, "top": Y1}[_ifc]
elif _ifc:
    try:
        IFACE_X = float(_ifc)
    except ValueError:
        pass
if str(_cfg.get("side", "")).strip().lower() in ("dirichlet", "neumann"):
    SIDE = str(_cfg["side"]).strip().lower()
PARTNER = str(_cfg.get("partner") or PARTNER).strip()
for _nm, _key in (("K", "k"), ("RHO_C", "rho_c"), ("REACTION", "reaction"), ("T_START", "t_start"),
                  ("T_END", "t_end"), ("THETA", "theta")):
    if _cfg.get(_key) is not None:
        globals()[_nm] = _cfg[_key] if _key == "k" and np.ndim(_cfg[_key]) else float(_cfg[_key])
if isinstance(_cfg.get("full_outer_dirichlet"), bool):
    FULL_OUTER_DIRICHLET = _cfg["full_outer_dirichlet"]
if _cfg.get("outer") is not None:
    T_OUTER = _expr(_cfg["outer"], ("x", "y", "t"))
if _cfg.get("initial") is not None:
    T_INITIAL = _expr(_cfg["initial"], ("x", "y"))
if _cfg.get("source_expr") is not None:
    F_SRC = _expr(str(_cfg["source_expr"]), ("x", "y", "t"))
_K4 = [float(K)] if np.ndim(K) == 0 else [float(v) for v in np.asarray(K, float).ravel()]
if len(_K4) not in (1, 4):
    sys.exit(f"K must be a number or a 2x2 matrix, not {K!r}")
print(f"SOURCES IN USE: from {'config.json' if _cfg.get('source_expr') is not None else 'code (F_SRC)'}"
      f"; k = {' '.join(f'{v:g}' for v in _K4)}; rho_c = {RHO_C:g}; reaction = {REACTION:g}; "
      f"window {T_START:g} to {T_END:g} in {N_STEPS} steps, theta = {THETA:g}")
if FULL_OUTER_DIRICHLET is None:
    raise SystemExit("FULL_OUTER_DIRICHLET is unset: say whether the two non-interface edges the "
                     "interface ends on are held (True) or natural (False), from your problem "
                     "statement -- in this file or as \"full_outer_dirichlet\" in config.json.")
AX = 0 if IFACE_AXIS == "x" else 1     # the coordinate the interface FIXES; AL runs ALONG it
AL = 1 - AX


def sample(imp, key):
    """The partner's data as (s, [v_1 .. v_N]) rows sorted by s, s the
    coordinate along the interface: one column per step. On iteration 1
    (nothing imported yet) the Dirichlet role holds T_INITIAL along the
    interface at every step, the Neumann role applies Q_INIT. The program
    interpolates each column at its own interface points."""
    if not imp or not imp.get("coordinates"):
        n_along = (NY if AX == 0 else NX) * DEGREE + 1
        ss = np.linspace(Y0, Y1, n_along) if AX == 0 else np.linspace(X0, X1, n_along)
        xy = (0.0 * ss + IFACE_X, ss) if AX == 0 else (ss, 0.0 * ss + IFACE_X)
        first = (np.broadcast_to(np.asarray(T_INITIAL(*xy), float), ss.shape) if SIDE == "dirichlet"
                 else np.full(ss.shape, float(Q_INIT)))
        return [(float(s), [float(v)] * N_STEPS) for s, v in zip(ss, first)]
    ss = [float(c[AL]) for c in imp["coordinates"]]
    vs = np.asarray(imp.get(key) if imp.get(key) is not None else [], float)
    if vs.ndim == 2 and vs.shape == (len(ss), N_STEPS):
        return sorted(zip(ss, vs.tolist()))
    # A TRACE OF ANOTHER SHAPE STOPS THIS SCRIPT: falling back would turn a
    # mismatched window, or a steady partner, into a run that converges to the
    # iteration-1 guess at the interface with nothing saying the data was dropped.
    got = f"shape {vs.shape}" if vs.size else "nothing"
    sys.exit(f"THE PARTNER'S {key!r} CANNOT BE READ AS ONE ROW OF {N_STEPS} STEPS PER INTERFACE POINT: "
             f"it has {got} for {len(ss)} points, and this side's window has N_STEPS = {N_STEPS}. "
             + ("A partner that exports one value per point is a steady contract; a time-dependent "
                "coupling uses variant='transient' on both sides. " if vs.ndim == 1 and vs.size == len(ss) else
                "The exchange carries no time axis, so a trace of another length is the only sign of a "
                "mismatched window: give both participants the same T_START, T_END, N_STEPS and THETA "
                "(and the same n_steps in config.json when a level sets it). " if vs.ndim == 2 else
                f"On the {SIDE} role this side reads the partner's {key!r}, which a "
                f"{'neumann' if SIDE == 'dirichlet' else 'dirichlet'} partner exports: check the two SIDEs. "))


imp = read_imports()
rows = sample(imp, "values" if SIDE == "dirichlet" else "normal_fluxes")

# ── THE PROGRAM, BUILT WHEN IT IS MISSING OR OLDER THAN ITS SOURCE (served) ──
#    cmake -S . -B build -DDEAL_II_DIR=<tree> -DCMAKE_BUILD_TYPE=Release, then
#    make -C build, with these six lines as CMakeLists.txt:
#        cmake_minimum_required(VERSION 3.13)
#        find_package(deal.II 9.0 REQUIRED HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR})
#        deal_ii_initialize_cached_variables()
#        project(dealii_side_transient CXX)
#        add_executable(dealii_side_transient dealii_side_transient.cc)
#        deal_ii_setup_target(dealii_side_transient)
#    <tree> is the deal.II build or install tree discover(query='list') and
#    write_participant_contract name: set DEAL_II_DIR above to it. A compile
#    that reads /usr/include/deal.II builds against a system package (or ran
#    without deal_ii_setup_target), whatever cmake's "Using the deal.II-..."
#    line said; the program's version check stops an older one.
_exe, _src, _cml = Path(DEALII_EXE), Path(DEALII_SRC), Path("CMakeLists.txt")
if _src.is_file() and (not _exe.is_file() or _exe.stat().st_mtime < max(
        _src.stat().st_mtime, _cml.stat().st_mtime if _cml.is_file() else 0.0)):
    _dir = DEAL_II_DIR or os.environ.get("DEAL_II_DIR", "")
    for _cmd in (["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"]
                 + ([f"-DDEAL_II_DIR={_dir}"] if _dir else []), ["make", "-C", "build", "-j4"]):
        _b = subprocess.run(_cmd, capture_output=True, text=True)
        _log = (_b.stdout + "\n" + _b.stderr).replace(str(Path.cwd()) + "/", "").splitlines()
        for _ln in (ln for ln in _log if "Using the deal.II" in ln):
            print(_ln)                                    # which deal.II the build found
        if _b.returncode != 0:
            _err = [ln for ln in _log if " error: " in ln or "CMake Error" in ln or "undefined reference" in ln]
            _usr = any("/usr/include/deal.II" in ln for ln in _err)
            sys.exit(f"BUILD FAILED at `{' '.join(_cmd[:2])}`. The first errors:\n"
                     + "\n".join((_err or _log[-15:])[:12]) + "\n"
                     + ("The compile read /usr/include/deal.II: that is a system deal.II, not the tree "
                        "to build against. Set DEAL_II_DIR above to the tree discover(query='list') "
                        "names, and keep deal_ii_setup_target in CMakeLists.txt." if _usr else
                        "The LINK failed: the first 'undefined reference' names a call that compiled "
                        "and that deal.II defines for other arguments." if "undefined reference" in "".join(_err) else
                        f"Fix the FIRST one; it names the line of {DEALII_SRC}. A hole left empty "
                        "fails exactly so." if " error: " in "".join(_err) else
                        "If cmake did not find deal.II, set DEAL_II_DIR above to the build or "
                        "install tree discover(query='list') names."))
if not _exe.is_file():
    sys.exit(f"no program at {DEALII_EXE}, and no {DEALII_SRC} and CMakeLists.txt beside this "
             f"script to build it from: write_participant_contract(solver='dealii', ..., "
             f"variant='transient') writes them.")

# ── THE PROGRAM'S INPUT (served; dealii_side_transient.cc documents the same lines) ──
nfx, nfy = NX * DEGREE + 1, NY * DEGREE + 1      # the grids run through the mesh nodes
gx, gy = np.meshgrid(np.linspace(X0, X1, nfx), np.linspace(Y0, Y1, nfy))
TIMES = T_START + (T_END - T_START) / N_STEPS * np.arange(N_STEPS + 1)    # t^0 .. t^N
_grid = lambda fn, *t: np.broadcast_to(np.asarray(fn(gx, gy, *t), float), gx.shape)   # noqa: E731
fsrc = [_grid(F_SRC, t) for t in TIMES]
_g = lambda v: f"{float(v):.17g}"                # noqa: E731 -- full precision, plain float
_block = lambda a: "\n".join(" ".join(map(repr, row)) for row in np.asarray(a, float).tolist())   # noqa: E731
lines = [f"side {SIDE}", "k " + " ".join(_g(v) for v in _K4), f"rho_c {_g(RHO_C)}", f"reaction {_g(REACTION)}",
         f"box {_g(X0)} {_g(X1)} {_g(Y0)} {_g(Y1)}", f"interface {IFACE_AXIS} {_g(IFACE_X)}",
         f"full_outer {int(bool(FULL_OUTER_DIRICHLET))}", f"mesh {int(NX)} {int(NY)}", f"degree {DEGREE}",
         f"level {LEVEL}", f"time {_g(T_START)} {_g(T_END)} {int(N_STEPS)} {_g(THETA)}",
         f"samples {len(rows)} {N_STEPS}"]
lines += [f"{_g(s)} " + " ".join(map(repr, map(float, vals))) for s, vals in rows]
lines += [f"initial {nfx} {nfy}", _block(_grid(T_INITIAL))]
lines += [f"source {nfx} {nfy} {N_STEPS + 1}"] + [_block(f) for f in fsrc]
lines += [f"outer {nfx} {nfy} {N_STEPS + 1}"] + [_block(_grid(T_OUTER, t)) for t in TIMES]
Path("dealii_input.txt").write_text("\n".join(lines) + "\n")

for _old in ("dealii_interface.txt", "dealii_field.txt"):
    Path(_old).unlink(missing_ok=True)
_if_out = Path("dealii_interface.txt").resolve()        # absolute paths: no cwd dependence
r = subprocess.run([str(_exe.resolve()), str(Path("dealii_input.txt").resolve()), str(_if_out)],
                   capture_output=True, text=True)
# THE PROGRAM'S OWN CONSOLE, passed through on every run: the per-level run log
# is that console, and its `NDOF = <n>` line comes from the program.
print(r.stdout, end="")
sys.stderr.write(r.stderr)
if r.returncode < 0:          # killed by a signal: a crash inside the program
    import signal as _signal
    _debug = _cml.is_file() and re.search(r"^\s*target_compile_definitions\s*\([^)#]*\bDEBUG\b",
                                          _cml.read_text(), re.M)
    sys.exit(f"the deal.II program was KILLED BY {_signal.Signals(-r.returncode).name} (return code "
             f"{r.returncode}): a crash inside the program, not an install fault. "
             + ("This build has DEBUG on (the target_compile_definitions line of CMakeLists.txt): deal.II "
                "checked its own assertions, and a failed one prints 'An error occurred in line' and the "
                "violated condition on the stderr above, naming the call. An unsized index vector handed "
                "to get_dof_indices crashes with or without DEBUG." if _debug else
                "This build is a Release build, which asserts nothing: an FEValues accessor whose update "
                "flag was not requested, or an unsized index vector, crashes with no message. Uncomment the "
                "target_compile_definitions(... PRIVATE DEBUG) line in CMakeLists.txt and run again: "
                "deal.II then names a missing flag itself."))
if r.returncode != 0 or not _if_out.is_file():
    sys.exit(f"the deal.II program failed (return code {r.returncode}); its own message is above")
# A PROGRAM THAT NEVER SAW THE SOURCE RETURNS THE BOUNDARY-DATA-ONLY ANSWER WITH NO
# ERROR; the served one checks its `source` against the samples and says so.
if max(float(np.abs(f).max()) for f in fsrc) > 0.0 and "\nSOURCE on" not in "\n" + r.stdout:
    sys.exit(f"F_SRC is non-zero and the program at {DEALII_EXE} did not print 'SOURCE on': "
             f"it is not the served program, or older than this input. Rebuild it from {DEALII_SRC}.")

_rows = np.loadtxt("dealii_interface.txt", ndmin=2)
if _rows.size == 0 or _rows.shape[1] != 2 + 2 * N_STEPS:
    sys.exit(f"the deal.II program wrote {_rows.shape[1] if _rows.size else 0} numbers per interface line; "
             f"x, y, the trace at each of the {N_STEPS} steps and the flux over each were expected "
             f"({2 + 2 * N_STEPS})")
coords = _rows[:, :2].tolist()
temps, fluxes = _rows[:, 2:2 + N_STEPS], _rows[:, 2 + N_STEPS:]
print(f"[dealii-transient {SIDE}] iface n={len(coords)} steps={N_STEPS} dt={(T_END - T_START) / N_STEPS:.6g} "
      f"theta={THETA:g} T(t_end)=[{temps[:, -1].min():.6g},{temps[:, -1].max():.6g}] "
      f"q(last step)=[{fluxes[:, -1].min():.6g},{fluxes[:, -1].max():.6g}]")

# ── EXPORT SELF-CHECK ─ keep this block. It stops the three exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
#    a flux of ~0 against a nonzero partner); and a flux that is the partner's
#    array negated instead of a recovery from THIS side's own system.
_chk_vals = np.asarray(temps, float).ravel()
_chk_flux = np.asarray(fluxes, float).ravel()
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

# PER-LEVEL PERSISTENCE: this level's interface trace and flux at the LAST step
# and its whole field at T_END, named by LEVEL (exports.json is overwritten by
# the next level; these are not). Interpolate THESE onto the probe points your
# task names. A dump defect must not cost the solve: exports.json is written after them.
try:
    _fld = np.loadtxt("dealii_field.txt", ndmin=2)
    _last = np.column_stack([_rows[:, :2], temps[:, -1], fluxes[:, -1]])
    for _name, _head, _data in ((f"interface_level{LEVEL}.csv", "x,y,u,qn", _last),
                                (f"field_level{LEVEL}.csv", "x,y,u", _fld[:, :3])):
        with open(_name, "w") as _f:
            _f.write(_head + "\n" + "".join(",".join(f"{v:.11e}" for v in _row) + "\n" for _row in _data))
    print(f"[dealii-transient {SIDE}] field_level{LEVEL}.csv: {len(_fld)} points at t = {T_END:g}")
except Exception as _dump_exc:
    for _partial in (f"field_level{LEVEL}.csv", f"interface_level{LEVEL}.csv"):
        if Path(_partial).is_file() and len(Path(_partial).read_text().splitlines()) <= 1:
            Path(_partial).unlink()             # a header-only CSV looks like a submission
    print(f"[dealii-transient per-level dump] level {LEVEL} dump failed: {_dump_exc!r}. exports.json is "
          f"still written, but this level has no field file to hand in; run it again.")

Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": len(coords),
    "coordinates": coords,
    "values": temps.tolist(),
    "normal_fluxes": fluxes.tolist(),
}, indent=2))
