"""deal.II VECTOR participant for the openPASO `couple` driver.

Plane-strain linear elasticity  -div(sigma(u)) = b  on one rectangular
subdomain, the interface a straight line. The exchanged interface state is a
VECTOR on BOTH channels:

    values        = displacement       u = (u_x, u_y)   at the interface nodes
    normal_fluxes = interface traction export, q_out = -(sigma . n_own)

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

THIS WRAPPER HAS NO HOLE. The solve is the C++ program dealii_side_elastic.cc
beside it, whose five marked holes are yours; write_participant_contract(...,
variant='elasticity') writes it and its CMakeLists.txt next to this file. The
wrapper writes the program's input, builds the program (the build block below),
runs it, and turns its output into exports.json and the per-level dumps.

SIGN. n_own is this side's outward normal. The two sides' exports cancel
componentwise, and the Neumann side applies the partner's numbers UNCHANGED:
the natural boundary term of the weak form is +(sigma . n_own) . v =
+q_out_partner . v. Exporting sigma . n_own instead flips the load the Neumann
side applies; the iteration still converges, to the wrong answer.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side.
#    config.json (or OPENPASO_CONFIG_JSON) overrides them where it states them.
SIDE      = "dirichlet"   # "dirichlet" (import u, export traction) | "neumann"
PARTNER   = "right"       # name of the partner participant in couple(...)
X0, X1    = 0.0, 0.55
Y0, Y1    = 0.0, 0.4
IFACE_AXIS = "x"          # the interface is the line x = IFACE_X ("x") or y = IFACE_X ("y")
IFACE_X   = 0.55
E_MOD     = 870.0         # Young's modulus
NU        = 0.29          # Poisson ratio (PLANE STRAIN)


def B_SRC(x, y):
    """Body force per unit volume, (b_x, b_y). Zero as shipped, a PLACEHOLDER
    like every number above: with the outer edges held and no body force the
    only solution is u = 0, and the coupling converges to it. Put your b here,
    or "source_ux" and "source_uy" in config.json. x and y are NumPy arrays;
    return two arrays of their shape. The program checks its `body_force`
    against these samples and prints BODY_FORCE on when its assembly
    integrated them."""
    return np.zeros_like(x), np.zeros_like(y)


# The displacement held on the held non-interface edges, as a polynomial in (x, y):
#     u_x = UDX[0] + UDX[1]*x + UDX[2]*y + UDX[3]*y*y      (u_y the same with UDY)
UDX = (0.0, 0.0, 0.0, 0.0)
UDY = (0.0, 0.0, 0.0, 0.0)


def U_OUTER(x, y):
    """The held displacement (u_x, u_y) at points of the held edges, two arrays
    of the shape of x and y: the UDX, UDY polynomial as shipped. Any other
    function of (x, y) your problem gives for the held edges goes here. "udx"
    and "udy" in config.json (four coefficients each) override UDX and UDY."""
    return (UDX[0] + UDX[1] * x + UDX[2] * y + UDX[3] * y * y + 0.0 * x,
            UDY[0] + UDY[1] * x + UDY[2] * y + UDY[3] * y * y + 0.0 * x)


# WHICH NON-INTERFACE EDGES ARE HELD IS YOUR PROBLEM'S TO SAY: the edge
# opposite the interface is held; True if the two edges the interface ends on
# are held too, False if they are traction free. The script refuses to run
# until you set it, and the program checks your hole 4 against it.
FULL_OUTER_DIRICHLET = None  # <-- True or False, FROM YOUR PROBLEM STATEMENT
NX, NY    = 46, 26
UI_X, UI_Y = 0.0, 0.0     # iteration-1 fallback interface displacement
TI_X, TI_Y = 0.0, 0.0     # iteration-1 fallback interface traction
DEAL_II_DIR = ""          # the deal.II build or install tree discover(query='list') names;
                          # "" leaves it to the DEAL_II_DIR environment variable
# ─────────────────────────────────────────────────────────────────────────

DEGREE = 1                                  # FE_Q degree of each component in the program
DEALII_SRC = "dealii_side_elastic.cc"       # the served program, holes filled, beside this file
DEALII_EXE = "./build/dealii_side_elastic"  # what the build below makes of it


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


def _expr_fn(expr):
    """A NumPy function of (x, y) from an expression string as a task writes it (`^` allowed)."""
    code = compile(str(expr).replace("^", "**"), "<config expression>", "eval")
    names = {n: getattr(np, n) for n in ("sin", "cos", "exp", "sqrt", "abs", "log", "tanh", "cosh", "sinh", "pi")}
    return lambda x, y: eval(code, {"__builtins__": {}}, dict(names, x=x, y=y)) + 0.0 * x


# ── THE PER-LEVEL RULE AND THE PROBLEM'S DATA (served). ./config.json (and the
#    OPENPASO_CONFIG_JSON a multi-level call hands over) may carry "level", "nx",
#    "ny" -- the dumps at the foot carry the level in their NAME -- and this
#    subdomain's data as the task writes them: side, partner; x0, x1, y0, y1;
#    iface ("left"|"right"|"bottom"|"top", or the interface coordinate) and
#    iface_axis; E and nu, or lam and mu; udx, udy; full_outer_dirichlet;
#    source_ux, source_uy (strings in x and y, `^` allowed). They override the
#    constants above; the audit's equation check reads the same keys.
LEVEL = 1
_cfg = {}
for _src, _txt in (("config.json", Path("config.json").read_text() if Path("config.json").is_file() else ""),
                   ("OPENPASO_CONFIG_JSON", os.environ.get("OPENPASO_CONFIG_JSON", ""))):
    try:
        _cfg.update(**json.loads(_txt or "{}"))
        LEVEL, NX, NY = int(_cfg.get("level", LEVEL)), int(_cfg.get("nx", NX)), int(_cfg.get("ny", NY))
    except (ValueError, TypeError) as _e:
        raise SystemExit(f"{_src} could not be read ({_e}); nothing was solved")
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
if "E" in _cfg and "nu" in _cfg:
    E_MOD, NU = float(_cfg["E"]), float(_cfg["nu"])
elif ("lam" in _cfg or "lambda" in _cfg) and "mu" in _cfg:
    _lam, _mu = float(_cfg.get("lam", _cfg.get("lambda"))), float(_cfg["mu"])
    E_MOD, NU = _mu * (3.0 * _lam + 2.0 * _mu) / (_lam + _mu), _lam / (2.0 * (_lam + _mu))
for _nm, _key in (("UDX", "udx"), ("UDY", "udy")):
    if isinstance(_cfg.get(_key), (list, tuple)) and len(_cfg[_key]) == 4:
        globals()[_nm] = tuple(float(_c) for _c in _cfg[_key])
if isinstance(_cfg.get("full_outer_dirichlet"), bool):
    FULL_OUTER_DIRICHLET = _cfg["full_outer_dirichlet"]
if _cfg.get("source_ux") is not None and _cfg.get("source_uy") is not None:
    _bx_cfg, _by_cfg = _expr_fn(_cfg["source_ux"]), _expr_fn(_cfg["source_uy"])
    B_SRC = lambda x, y: (_bx_cfg(x, y), _by_cfg(x, y))  # noqa: E731,F811 -- config wins over the body above
LAM = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))   # plane strain
MU = E_MOD / (2.0 * (1.0 + NU))
print(f"SOURCES IN USE: from {'config.json' if _cfg.get('source_ux') is not None else 'code (B_SRC)'}"
      f"; lam = {LAM:g}, mu = {MU:g} (E = {E_MOD:g}, nu = {NU:g})")
if FULL_OUTER_DIRICHLET is None:
    raise SystemExit("FULL_OUTER_DIRICHLET is unset: say whether the two non-interface edges the "
                     "interface ends on are held (True) or traction free (False), from your problem "
                     "statement -- in this file or as \"full_outer_dirichlet\" in config.json.")
AX = 0 if IFACE_AXIS == "x" else 1     # the coordinate the interface FIXES; AL runs ALONG it
AL = 1 - AX


def sample(imp, key, fallback):
    """The partner's VECTOR samples as sorted (s, v_x, v_y) triples, s the
    coordinate along the interface; a constant on iteration 1. The program
    interpolates each component at its own interface points (the driver does
    not interpolate between meshes). A TRACE OF ANOTHER SHAPE STOPS THIS
    SCRIPT rather than falling back to the iteration-1 constant."""
    if not imp or not imp.get("coordinates"):
        return [(float(a), float(fallback[0]), float(fallback[1])) for a in ((Y0, Y1) if AX == 0 else (X0, X1))]
    ss = [float(c[AL]) for c in imp["coordinates"]]
    vs = np.asarray(imp.get(key) if imp.get(key) is not None else [], float)
    if vs.ndim == 2 and vs.shape == (len(ss), 2):
        return sorted((s, float(v[0]), float(v[1])) for s, v in zip(ss, vs))
    got = f"shape {vs.shape}" if vs.size else "nothing"
    sys.exit(f"THE PARTNER'S {key!r} CANNOT BE READ AS TWO COMPONENTS PER INTERFACE POINT: it has {got} for "
             f"{len(ss)} points. "
             + (f"On the {SIDE} role this side reads the partner's {key!r}, which a "
                f"{'neumann' if SIDE == 'dirichlet' else 'dirichlet'} partner exports: check the two SIDEs."
                if not vs.size else "A scalar partner (one value per point) is a heat contract: both sides of an "
                "elastic coupling use their elasticity contracts."))


imp = read_imports()
triples = sample(imp, "values" if SIDE == "dirichlet" else "normal_fluxes",
                 (UI_X, UI_Y) if SIDE == "dirichlet" else (TI_X, TI_Y))

# ── THE PROGRAM, BUILT WHEN IT IS MISSING OR OLDER THAN ITS SOURCE (served) ──
#    cmake -S . -B build -DDEAL_II_DIR=<tree> -DCMAKE_BUILD_TYPE=Release, then
#    make -C build, with these six lines as CMakeLists.txt:
#        cmake_minimum_required(VERSION 3.13)
#        find_package(deal.II 9.0 REQUIRED HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR})
#        deal_ii_initialize_cached_variables()
#        project(dealii_side_elastic CXX)
#        add_executable(dealii_side_elastic dealii_side_elastic.cc)
#        deal_ii_setup_target(dealii_side_elastic)
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
    sys.exit(f"no program at {DEALII_EXE}, and no {DEALII_SRC} and CMakeLists.txt beside this script to "
             f"build it from: write_participant_contract(solver='dealii', ..., variant='elasticity') writes them.")

# ── THE PROGRAM'S INPUT (served; dealii_side_elastic.cc documents the same lines) ──
nfx, nfy = NX * DEGREE + 1, NY * DEGREE + 1      # the grids pass through the mesh nodes
gx, gy = np.meshgrid(np.linspace(X0, X1, nfx), np.linspace(Y0, Y1, nfy))
bx, by = (np.broadcast_to(np.asarray(v, float), gx.shape) for v in B_SRC(gx, gy))
ox, oy = (np.broadcast_to(np.asarray(v, float), gx.shape) for v in U_OUTER(gx, gy))
_g = lambda v: f"{float(v):.17g}"                # noqa: E731 -- full precision, plain float
lines = [f"side {SIDE}", f"lame {_g(LAM)} {_g(MU)}", f"box {_g(X0)} {_g(X1)} {_g(Y0)} {_g(Y1)}",
         f"interface {IFACE_AXIS} {_g(IFACE_X)}", f"full_outer {int(bool(FULL_OUTER_DIRICHLET))}",
         f"mesh {int(NX)} {int(NY)}", f"degree {DEGREE}", f"level {LEVEL}", f"samples {len(triples)}"]
lines += [f"{_g(s)} {_g(vx)} {_g(vy)}" for s, vx, vy in triples]
for _key, (_a, _b) in (("source", (bx, by)), ("outer", (ox, oy))):
    lines += [f"{_key} {nfx} {nfy}"] + [" ".join(f"{_g(p)} {_g(q)}" for p, q in zip(ra, rb)) for ra, rb in zip(_a, _b)]
Path("dealii_input.txt").write_text("\n".join(lines) + "\n")

for _old in ("dealii_interface.txt", "dealii_field.txt"):
    Path(_old).unlink(missing_ok=True)
_if_out = Path("dealii_interface.txt").resolve()        # absolute paths: no cwd dependence
import resource as _resource                     # what a killed run held, and how long it ran
import time as _time
_rss_before, _t_run = _resource.getrusage(_resource.RUSAGE_CHILDREN).ru_maxrss, _time.monotonic()
r = subprocess.run([str(_exe.resolve()), str(Path("dealii_input.txt").resolve()), str(_if_out)],
                   capture_output=True, text=True)
_t_run = _time.monotonic() - _t_run
# THE PROGRAM'S OWN CONSOLE, passed through on every run: the per-level run log
# is that console, and its `NDOF = <n>` line comes from the program.
print(r.stdout, end="")
sys.stderr.write(r.stderr)
if r.returncode < 0:          # killed by a signal: a crash inside the program, or a stop from outside
    import re as _re
    import signal as _signal
    if -r.returncode in (_signal.SIGKILL, _signal.SIGTERM):
        # A SIGNAL FROM OUTSIDE IS NOT A CRASH (measured: a program whose hole 1 refined its mesh nx + ny
        # times took all of the host's memory, and the kernel killed it after eight minutes; this text
        # called that a crash inside the program and sent the run to a DEBUG build). What the run measured
        # is said instead: how long it ran, the most memory it held, and whether it got past hole 1.
        _rss = _resource.getrusage(_resource.RUSAGE_CHILDREN).ru_maxrss      # kB: the largest child's peak
        sys.exit(f"the deal.II program was KILLED BY {_signal.Signals(-r.returncode).name} (return code "
                 f"{r.returncode}) after {_t_run:.0f} s"
                 + (f", holding {_rss / 2 ** 20:.1f} GB of memory at its peak" if _rss > _rss_before else "")
                 + ": that signal comes from outside the program. The kernel sends SIGKILL to a program that "
                   "takes the host's memory, and a time limit sends it too; a crash inside the program ends "
                   "with SIGSEGV or SIGABRT instead, and a DEBUG build does not name this. "
                 + ("It printed no 'NDOF = ' line: it was stopped before it numbered its dofs, in hole 1 or "
                    "the served lines just after it." if "\nNDOF = " not in "\n" + r.stdout else
                    "It printed its 'NDOF = ' line, so it got past hole 1; its last lines are above."))
    # THE MESSAGE SAYS WHICH BUILD CRASHED: a DEBUG build was told to turn DEBUG on (measured).
    _debug = _cml.is_file() and _re.search(r"^\s*target_compile_definitions\s*\([^)#]*\bDEBUG\b",
                                           _cml.read_text(), _re.M)
    _heap = _re.search(r"free\(\)|malloc\(\)|double free|corrupted|fastbin|munmap_chunk", r.stderr)
    sys.exit(f"the deal.II program was KILLED BY {_signal.Signals(-r.returncode).name} (return code "
             f"{r.returncode}): a crash inside the program, not an install fault. "
             + (f"The C library's words on the stderr above (`{_heap.group(0)}` ...) say the heap was damaged: "
                "a write past the end of an array the program owns (a FullMatrix, a Vector or a std::vector), "
                "found at a later allocation. " if _heap else "")
             + ("This build has DEBUG on (the target_compile_definitions line of CMakeLists.txt): deal.II "
                "checked its own assertions, and a failed one prints 'An error occurred in line' and the "
                "violated condition on the stderr above, naming the call. An unsized index vector handed "
                "to get_dof_indices crashes with or without DEBUG." if _debug else
                "This build is a Release build, which asserts nothing: an index past the end of a deal.II "
                "matrix or vector, an FEValues accessor whose update flag was not requested, or an unsized "
                "index vector crashes with no message. Uncomment the "
                "target_compile_definitions(dealii_side_elastic PRIVATE DEBUG) line in CMakeLists.txt and run "
                "again: deal.II then names a bad index or a missing flag itself."))
if r.returncode != 0 or not _if_out.is_file():
    sys.exit(f"the deal.II program failed (return code {r.returncode}); its own message is above")
# A PROGRAM THAT DROPPED THE BODY FORCE RETURNS THE BOUNDARY-DATA-ONLY ANSWER WITH
# NO ERROR; the served one announces what it assembled, so its absence is fatal.
if max(float(np.abs(bx).max()), float(np.abs(by).max())) > 0.0 and "\nBODY_FORCE on" not in "\n" + r.stdout:
    sys.exit(f"B_SRC is non-zero and the program at {DEALII_EXE} did not print 'BODY_FORCE on': "
             f"it is not the served scaffold, or older than this input. Rebuild it from {DEALII_SRC}.")

_rows = np.loadtxt("dealii_interface.txt", ndmin=2)
if _rows.size == 0 or _rows.shape[1] != 6:
    sys.exit("the deal.II program wrote no 'x y u_x u_y q_x q_y' interface lines")
coords, disp, trac = _rows[:, :2].tolist(), _rows[:, 2:4].tolist(), _rows[:, 4:6].tolist()

# ── EXPORT SELF-CHECK ─ keep this block. It stops the four exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
#    a traction of ~0 against a nonzero partner); a traction that is the
#    partner's array negated instead of a recovery from THIS side's own system;
#    and a Dirichlet side that exports exactly 0.0 where its own data make a
#    reaction.
_chk_vals = np.asarray(disp, float).ravel()
_chk_flux = np.asarray(trac, float).ravel()
if not (np.isfinite(_chk_vals).all() and np.isfinite(_chk_flux).all()):
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface values or "
                     "tractions; the solve did not produce a usable field, so "
                     "nothing was exported")
_chk_imp = (json.loads(Path("imports.json").read_text() or "{}")
            if Path("imports.json").is_file() else {})
_chk_qin = (np.concatenate([np.asarray(_d.get("normal_fluxes") or [], float).ravel()
                            for _d in _chk_imp.values()])
            if _chk_imp else np.zeros(0))
if SIDE == "neumann" and _chk_qin.size and np.abs(_chk_qin).max() > 0 \
        and np.abs(_chk_flux).max() < 1e-9 * np.abs(_chk_qin).max():
    raise SystemExit("EXPORT SELF-CHECK: the recovered interface traction is ~0 "
                     "against a nonzero imported traction: the imported load "
                     "never entered the assembled system (the facet term / "
                     "boundary condition that integrates it is missing). Fix "
                     "the application; do not couple on")
# (Dirichlet role only: a Neumann side's consistent recovery of a CONSTANT
#  applied traction can legitimately reproduce it to the last bit.)
if SIDE == "dirichlet" and _chk_qin.shape == _chk_flux.shape and _chk_flux.size \
        and np.any(_chk_flux) and np.array_equal(_chk_flux, -_chk_qin):
    raise SystemExit("EXPORT SELF-CHECK: the exported traction is the partner's "
                     "array negated, bit for bit: a copy, not a recovery from "
                     "this side's own assembled system")
# (Dirichlet role: a traction of exactly 0.0 at every interface point is what a
#  side free of stress exports -- no body force, and a displacement handed in
#  that is one rigid motion. With either, the reaction is not zero.)
_chk_rigid = 0.0              # how far the displacement handed in is from one rigid motion
for _d in _chk_imp.values():
    _u, _p = np.asarray(_d.get("values") or [], float), np.asarray(_d.get("coordinates") or [], float)
    if _u.ndim == 2 and _u.shape[1] == 2 and _p.shape == _u.shape and len(_u) > 2 and np.any(_u):
        _m = np.zeros((2 * len(_u), 3))
        _m[0::2, 0], _m[1::2, 1], _m[0::2, 2], _m[1::2, 2] = 1.0, 1.0, -_p[:, 1], _p[:, 0]
        _fit = _m @ np.linalg.lstsq(_m, _u.ravel(), rcond=None)[0] - _u.ravel()
        _chk_rigid = max(_chk_rigid, float(np.abs(_fit).max() / np.abs(_u).max()))
_chk_body = max(float(np.abs(bx).max()), float(np.abs(by).max())) > 0.0
if SIDE == "dirichlet" and _chk_flux.size and not np.any(_chk_flux) and (_chk_body or _chk_rigid > 1e-9):
    raise SystemExit("EXPORT SELF-CHECK: the exported traction is exactly 0.0 at every interface "
                     "point, while " + ("this side carries a body force" if _chk_body else
                                        f"the displacement it was handed is not one rigid motion (it "
                                        f"differs from the nearest one by {_chk_rigid:.1e} of its size)")
                     + ": a reaction recovered from this side's assembled system is not zero there")

# PER-LEVEL PERSISTENCE: this level's interface displacement and traction and its
# whole field, named by LEVEL (exports.json is overwritten by the next level;
# these are not). Interpolate THESE onto the probe points your task names. THE
# qx, qy COLUMNS ARE THIS SIDE'S EXPORT, -(sigma . n_own): a task that asks for
# sigma . n wants their negative. A dump defect must not cost the solve:
# exports.json is written after them.
try:
    _fld = np.loadtxt("dealii_field.txt", ndmin=2)
    for _name, _head, _data in ((f"interface_level{LEVEL}.csv", "x,y,ux,uy,qx,qy", _rows),
                                (f"field_level{LEVEL}.csv", "x,y,ux,uy", _fld[:, :4])):
        with open(_name, "w") as _f:
            _f.write(_head + "\n" + "".join(",".join(f"{v:.11e}" for v in _row) + "\n" for _row in _data))
    print(f"[dealii-elastic {SIDE}] field_level{LEVEL}.csv: {len(_fld)} points")
except Exception as _dump_exc:
    # both files or neither: half a pair, a truncated file or an earlier run's file could
    # be read as this level's result
    for _partial in (f"field_level{LEVEL}.csv", f"interface_level{LEVEL}.csv"):
        try:
            Path(_partial).unlink(missing_ok=True)
        except OSError:
            pass
    print(f"[dealii-elastic per-level dump] level {LEVEL} dump failed: {_dump_exc!r}. exports.json is "
          f"still written, but this level has no field file to hand in; run it again.")

Path("exports.json").write_text(json.dumps({
    "field_name": "displacement",
    "n_points": len(coords),
    "coordinates": coords,
    "values": disp,
    "normal_fluxes": trac,
}, indent=2))
