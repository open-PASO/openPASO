"""deal.II participant for the openPASO `couple` driver.

Steady heat conduction  -div(k grad T) + c T = f  on one rectangular subdomain.
CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

THIS WRAPPER HAS NO HOLE. The solve is the C++ program dealii_side.cc beside
it, whose five marked holes are yours; write_participant_contract writes it and
its CMakeLists.txt next to this file. The wrapper writes the program's input,
builds the program (the build block below), runs it, and turns its output into
exports.json and the per-level dumps.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side; the payload that served
#    this script gives the exact block for the RIGHT / Neumann side.
#    config.json (or OPENPASO_CONFIG_JSON) overrides them where it states them.
SIDE      = "dirichlet"   # "dirichlet" | "neumann"
PARTNER   = "right"       # name of the partner participant in couple(...)
X0, X1    = 0.0, 0.6
Y0, Y1    = 0.0, 0.4
IFACE_AXIS = "x"          # the interface is the line x = IFACE_X ("x") or y = IFACE_X ("y")
IFACE_X   = 0.6
K         = 0.8           # conductivity: a number, or [[kxx, kxy], [kyx, kyy]] when it is anisotropic


def F_SRC(x, y):
    """Volumetric source f(x, y). Zero as shipped, a PLACEHOLDER like every
    number above: a zero or constant source leaves the outer values as the only
    data, and the coupling converges to a problem you were not given. Put your
    f here, or "source_expr" in config.json. x and y are NumPy arrays; return
    ONE array of their shape (`0.0 * x + c` for a constant), for example
        return -K * (6.0 * x * y**2 + 2.0 * x**3)   # for T = x**3 * y**2
    The program checks its `source` against these samples and prints
    VOLUME_SOURCE on when its assembly integrated them.
    """
    return np.zeros_like(x)
T_OUTER   = 335.0         # Dirichlet value on the held NON-interface edges
REACTION  = 0.0           # c in -div(K grad T) + c T = f (0 when there is none)
# WHICH NON-INTERFACE EDGES ARE HELD IS YOUR PROBLEM'S TO SAY: the edge
# opposite the interface is held; True if the two edges the interface ends on
# are held too, False if they are natural (zero flux). The script refuses to
# run until you set it, and the program checks your hole 4 against it.
FULL_OUTER_DIRICHLET = None  # <-- True or False, FROM YOUR PROBLEM STATEMENT
NX, NY    = 46, 26
T_INIT    = 310.0
Q_INIT    = 0.0           # iteration-1 fallback interface flux
DEAL_II_DIR = ""          # the deal.II build or install tree discover(query='list') names;
                          # "" leaves it to the DEAL_II_DIR environment variable
# ─────────────────────────────────────────────────────────────────────────

DEGREE = 1                          # FE_Q degree of the program
DEALII_SRC = "dealii_side.cc"       # the served program, holes filled, beside this file
DEALII_EXE = "./build/dealii_side"  # what the build below makes of it


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


# ── THE PER-LEVEL RULE AND THE PROBLEM'S DATA (served). ./config.json (and the
#    OPENPASO_CONFIG_JSON a multi-level call hands over) may carry "level", "nx",
#    "ny" -- the dumps at the foot carry the level in their NAME -- and this
#    subdomain's data as the task writes them: side, partner; x0, x1, y0, y1;
#    iface ("left"|"right"|"bottom"|"top", or the interface coordinate) and
#    iface_axis; k; reaction; outer; full_outer_dirichlet; source_expr (a
#    string in x and y, `^` allowed). They override the constants above; the
#    audit's equation check reads the same keys.
LEVEL = 1
try:
    _cfg = json.loads(Path("config.json").read_text() or "{}") if Path("config.json").is_file() else {}
    _cfg.update(json.loads(os.environ.get("OPENPASO_CONFIG_JSON") or "{}"))
    LEVEL, NX, NY = int(_cfg.get("level", LEVEL)), int(_cfg.get("nx", NX)), int(_cfg.get("ny", NY))
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
for _nm, _key in (("K", "k"), ("REACTION", "reaction"), ("T_OUTER", "outer")):
    if _cfg.get(_key) is not None:
        globals()[_nm] = _cfg[_key] if _key == "k" and np.ndim(_cfg[_key]) else float(_cfg[_key])
if isinstance(_cfg.get("full_outer_dirichlet"), bool):
    FULL_OUTER_DIRICHLET = _cfg["full_outer_dirichlet"]
if _cfg.get("source_expr") is not None:
    _code = compile(str(_cfg["source_expr"]).replace("^", "**"), "<source_expr>", "eval")
    _names = {n: getattr(np, n) for n in ("sin", "cos", "exp", "sqrt", "abs", "log", "tanh", "cosh", "sinh", "pi")}
    F_SRC = lambda x, y: eval(_code, {"__builtins__": {}}, dict(_names, x=x, y=y)) + 0.0 * x  # noqa: E731,F811
_K4 = [float(K)] if np.ndim(K) == 0 else [float(v) for v in np.asarray(K, float).ravel()]
if len(_K4) not in (1, 4):
    sys.exit(f"K must be a number or a 2x2 matrix, not {K!r}")
print(f"SOURCES IN USE: from {'config.json' if _cfg.get('source_expr') is not None else 'code (F_SRC)'}"
      f"; k = {' '.join(f'{v:g}' for v in _K4)}; reaction = {REACTION:g}; outer value = {T_OUTER:g}")
if FULL_OUTER_DIRICHLET is None:
    raise SystemExit("FULL_OUTER_DIRICHLET is unset: say whether the two non-interface edges the "
                     "interface ends on are held (True) or natural (False), from your problem "
                     "statement -- in this file or as \"full_outer_dirichlet\" in config.json.")
AX = 0 if IFACE_AXIS == "x" else 1     # the coordinate the interface FIXES; AL runs ALONG it
AL = 1 - AX


def sample(imp, key, fallback):
    """The partner's samples as sorted (s, value) pairs, s the coordinate along
    the interface; a constant on iteration 1. The program interpolates them at
    its own interface points (the driver does not interpolate between meshes).
    A TRACE OF ANOTHER SHAPE STOPS THIS SCRIPT. It used to be replaced by the
    iteration-1 constant, silently, so a time-dependent partner's (points x
    steps) trace left this side on its first guess for good."""
    if not imp or not imp.get("coordinates"):
        return [(float(a), float(fallback)) for a in ((Y0, Y1) if AX == 0 else (X0, X1))]
    ss = [float(c[AL]) for c in imp["coordinates"]]
    vs = np.asarray(imp.get(key) if imp.get(key) is not None else [], float)
    if vs.ndim == 1 and vs.size == len(ss):
        return sorted(zip(ss, vs.tolist()))
    got = f"shape {vs.shape}" if vs.size else "nothing"
    sys.exit(f"THE PARTNER'S {key!r} CANNOT BE READ AS ONE VALUE PER INTERFACE POINT: it has {got} for "
             f"{len(ss)} points. "
             + ("That is a time-dependent trace, one row of steps per point: this wrapper is the STEADY "
                "contract, and a time-dependent coupling uses variant='transient' on both sides."
                if vs.ndim == 2 and vs.shape[0] == len(ss) else
                f"On the {SIDE} role this side reads the partner's {key!r}, which a "
                f"{'neumann' if SIDE == 'dirichlet' else 'dirichlet'} partner exports: check the two SIDEs."
                if not vs.size else "Check what the partner writes into exports.json."))


imp = read_imports()
pairs = sample(imp, "values" if SIDE == "dirichlet" else "normal_fluxes",
               T_INIT if SIDE == "dirichlet" else Q_INIT)

# ── THE PROGRAM, BUILT WHEN IT IS MISSING OR OLDER THAN ITS SOURCE (served) ──
#    cmake -S . -B build -DDEAL_II_DIR=<tree> -DCMAKE_BUILD_TYPE=Release, then
#    make -C build, with these six lines as CMakeLists.txt:
#        cmake_minimum_required(VERSION 3.13)
#        find_package(deal.II 9.0 REQUIRED HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR})
#        deal_ii_initialize_cached_variables()
#        project(dealii_side CXX)
#        add_executable(dealii_side dealii_side.cc)
#        deal_ii_setup_target(dealii_side)
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
                        "Fix the FIRST one; it names the line of dealii_side.cc. A hole left empty "
                        "fails exactly so." if " error: " in "".join(_err) else
                        "If cmake did not find deal.II, set DEAL_II_DIR above to the build or "
                        "install tree discover(query='list') names."))
if not _exe.is_file():
    sys.exit(f"no program at {DEALII_EXE}, and no {DEALII_SRC} and CMakeLists.txt beside this "
             f"script to build it from: write_participant_contract(solver='dealii', ...) writes them.")

# ── THE PROGRAM'S INPUT (served; dealii_side.cc documents the same lines) ──
nfx, nfy = NX * DEGREE + 1, NY * DEGREE + 1      # the source on a grid through the mesh nodes
gx, gy = np.meshgrid(np.linspace(X0, X1, nfx), np.linspace(Y0, Y1, nfy))
fsrc = np.broadcast_to(np.asarray(F_SRC(gx, gy), float), gx.shape)
_g = lambda v: f"{float(v):.17g}"                # noqa: E731 -- full precision, plain float
lines = [f"side {SIDE}", "k " + " ".join(_g(v) for v in _K4), f"reaction {_g(REACTION)}",
         f"box {_g(X0)} {_g(X1)} {_g(Y0)} {_g(Y1)}", f"interface {IFACE_AXIS} {_g(IFACE_X)}",
         f"outer {_g(T_OUTER)}", f"full_outer {int(bool(FULL_OUTER_DIRICHLET))}",
         f"mesh {int(NX)} {int(NY)}", f"degree {DEGREE}", f"level {LEVEL}", f"samples {len(pairs)}"]
lines += [f"{_g(s)} {_g(v)}" for s, v in pairs]
lines += [f"source {nfx} {nfy}"] + [" ".join(_g(v) for v in row) for row in fsrc]
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
    import re as _re
    import signal as _signal
    # THE MESSAGE SAYS WHICH BUILD CRASHED: a DEBUG build was told to turn DEBUG on (measured).
    _debug = _cml.is_file() and _re.search(r"^\s*target_compile_definitions\s*\([^)#]*\bDEBUG\b",
                                           _cml.read_text(), _re.M)
    # DEBUG REACHES DEAL.II'S HEADERS, NOT ITS LIBRARY (measured on this install: a SparseMatrix
    # copy-constructed from a filled one is left empty, and the call that uses it crashes with no
    # message with DEBUG on as with it off).
    _said = "An error occurred in line" in r.stderr
    sys.exit(f"the deal.II program was KILLED BY {_signal.Signals(-r.returncode).name} (return code "
             f"{r.returncode}): a crash inside the program, not an install fault. "
             + ("This build has DEBUG on (the target_compile_definitions line of CMakeLists.txt), which turns "
                "on the checks in deal.II's headers: the one that failed is on the stderr above ('An error "
                "occurred in line' and the violated condition)." if _debug and _said else
                "This build has DEBUG on (the target_compile_definitions line of CMakeLists.txt) and no deal.II "
                "check printed: DEBUG turns on the checks in deal.II's headers, not the ones compiled into its "
                "library. Measured to crash so with DEBUG on: a SparseMatrix copy-constructed from a filled one, "
                "which is left empty (build the copy on sparsity, then copy_from), copy_from into a SparseMatrix "
                "never built on a pattern, and an unsized index vector handed to get_dof_indices." if _debug else
                "This build is a Release build, which asserts nothing: an FEValues accessor whose update "
                "flag was not requested, an unsized index vector, or a SparseMatrix copy-constructed from a "
                "filled one (it is left empty) crashes with no message. Uncomment the "
                "target_compile_definitions(dealii_side PRIVATE DEBUG) line in CMakeLists.txt and run "
                "again: deal.II's header checks then name a missing flag; the empty matrix and the unsized "
                "vector crash with no message even then."))
if r.returncode != 0 or not _if_out.is_file():
    sys.exit(f"the deal.II program failed (return code {r.returncode}); its own message is above")
# A PROGRAM THAT DROPPED THE SOURCE RETURNS THE BOUNDARY-DATA-ONLY ANSWER WITH NO
# ERROR; the served one announces what it assembled, so its absence is fatal.
if float(np.abs(fsrc).max()) > 0.0 and "\nVOLUME_SOURCE on" not in "\n" + r.stdout:
    sys.exit(f"F_SRC is non-zero and the program at {DEALII_EXE} did not print 'VOLUME_SOURCE on': "
             f"it is not the served scaffold, or older than this input. Rebuild it from {DEALII_SRC}.")

_rows = np.loadtxt("dealii_interface.txt", ndmin=2)
if _rows.size == 0 or _rows.shape[1] != 4:
    sys.exit("the deal.II program wrote no 'x y u q' interface lines")
coords, temps, fluxes = _rows[:, :2].tolist(), _rows[:, 2].tolist(), _rows[:, 3].tolist()

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

# PER-LEVEL PERSISTENCE: this level's interface trace and flux and its whole
# field, named by LEVEL (exports.json is overwritten by the next level; these
# are not). Interpolate THESE onto the probe points your task names. A dump
# defect must not cost the solve: exports.json is written after them.
try:
    _fld = np.loadtxt("dealii_field.txt", ndmin=2)
    for _name, _head, _data in ((f"interface_level{LEVEL}.csv", "x,y,u,qn", _rows),
                                (f"field_level{LEVEL}.csv", "x,y,u", _fld[:, :3])):
        with open(_name, "w") as _f:
            _f.write(_head + "\n" + "".join(",".join(f"{v:.11e}" for v in _row) + "\n" for _row in _data))
    print(f"[dealii {SIDE}] field_level{LEVEL}.csv: {len(_fld)} points")
except Exception as _dump_exc:
    for _partial in (f"field_level{LEVEL}.csv", f"interface_level{LEVEL}.csv"):
        if Path(_partial).is_file() and len(Path(_partial).read_text().splitlines()) <= 1:
            Path(_partial).unlink()             # a header-only CSV looks like a submission
    print(f"[dealii per-level dump] level {LEVEL} dump failed: {_dump_exc!r}. exports.json is "
          f"still written, but this level has no field file to hand in; run it again.")

Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": len(coords),
    "coordinates": coords,
    "values": temps,
    "normal_fluxes": fluxes,
}, indent=2))
