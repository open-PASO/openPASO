"""SPARTA (DSMC) participant for the openPASO `couple` driver.

SPARTA is a rarefied-gas DSMC particle code, not a FEM code, and has no
scripting API: it reads a text input deck and writes text dump/log files.
This module is therefore a WRAPPER that, each coupling iteration:
  1. reads imports.json  -> per-surface-element WALL TEMPERATURE,
  2. writes TSURF_IN in SPARTA's `custom surf ... file` format,
  3. writes the deck -- write_deck(seed), the one part that is yours -- which
     must read SURF_FILE FIRST (its lines then carry ids 1..N, the ids TSURF_IN
     and the dump reader use) and does
         custom      surf create tsurf float 0 file tsurf.in 1 tsurf
         surf_collide 1 diffuse s_tsurf 1.0      <- per-element Dirichlet T
         compute     1 surf <group> all etot     <- per-element heat flux
         fix         1 ave/surf <group> 1 N N c_1[1] ave one
         dump        1 surf <group> N flux.out id v1x v1y v2x v2y f_1 s_tsurf
     with <group> the group of SURF_FILE's lines (read_surf ... group <name>).
     NOTE the dump must reference `f_1`, NOT `f_1[1]`: a fix ave/surf with a
     single input column is a per-surf VECTOR, and the [1] form aborts with
     "Dump surf fix does not compute per-surf array" (dump_surf.cpp:609).
  4. runs the SPARTA binary with its console passed through,
  5. checks SPARTA's own setup report (the PARTICLE SELF-CHECK),
  6. parses the ASCII surf dump and writes exports.json.

THE FILE FORMATS (SPARTA's own rules, measured on this install; the readers
and the writer below follow them).
  SURF_FILE (read_surf, 2-D): the FIRST LINE IS ALWAYS SKIPPED.  Header lines
    "<N> points" and "<M> lines" follow (blank lines and # comments allowed
    there).  Then the line "Points", ONE SKIPPED LINE, and N rows "id x y";
    then the line "Lines", ONE SKIPPED LINE, and M rows "id p1 p2" (p1, p2 count
    the Points from 1 in the order they are listed).  The rows of a section
    follow each other with no blank line.  Without a Points section a Lines row
    is "id x1 y1 x2 y2".
  TSURF_IN (custom surf ... file <file> M <names>): comment or blank lines,
    then the count line "N M" (N value lines follow; M values on each, M the
    number of names after the file name), then N lines "id value".
  FLUX_OUT (dump surf): each snapshot is "ITEM: TIMESTEP", "ITEM: NUMBER OF
    SURFS", "ITEM: BOX BOUNDS ..." and "ITEM: SURFS <columns>", one row per
    element.  The reader takes the LAST snapshot and finds its columns by name.

ROLE.  SPARTA is a DIRICHLET-type participant for conjugate heat transfer:
it IMPORTS a field value (wall temperature) and EXPORTS a flux (the net
energy flux the gas deposits on the wall, `etot`, positive = wall gains
energy = energy leaving the gas subdomain through the interface, i.e.
exactly SPEC.md's outward-normal q_out for the gas side).
SPARTA HAS NO NATIVE FLUX BC, so this script is the Dirichlet side: none of the
nine `surf_collide` styles takes a prescribed heat flux, and the four thermal
ones (`diffuse`, `cll`, `td`, `impulsive`) all take a temperature.  The one
indirect route (`fix surf/temp`, a radiative-equilibrium wall temperature) is
NOT implemented below and was NOT run; knowledge(topic='coupling',
solver='sparta') describes it.

SURFACE LINES HAVE A FLOW SIDE.  SPARTA puts the gas on the side a line's
normal points to, N = (0,0,1) x (p2 - p1): walking from p1 to p2, the gas is on
your left.  read_surf reports "<a> <b> <c> = cells outside/inside/overlapping
surfs"; a = 0 means every grid cell is inside a body, create_particles makes 0
particles, and the run still exits 0 with every tally zero.  The PARTICLE
SELF-CHECK below stops on that and on any run that ends with 0 particles.

STOCHASTICITY.  DSMC output is a Monte-Carlo estimate.  With a fixed RNG seed
and identical input the run is bit-reproducible (so a fixed-point iteration
can appear to "converge" even when the physics has not); with a varying seed
the exported flux carries sampling noise that does NOT shrink with coupling
iterations, so the driver's relative-residual tolerance cannot be driven below
that noise floor.  SEED_MODE below selects which regime you are in: with
SEED_MODE = "vary", couple(..., noise_replicates=5) measures that floor and
judges convergence against it.
"""
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
PARTNER    = "solid"      # partner participant name in couple(...)
SPARTA     = "spa_serial"    # the SPARTA binary: the path `discover(query='list')` prints
                             # after "SPARTA at"; a bare name runs only when it is on PATH
DATA_DIR   = ""              # dir holding the surf/species/vss files; a file not here is
                             # copied from it, then from the SPARTA distribution's data dir
SURF_FILE  = "circle.surf"     # 50 line elements (2D cylinder)
SPECIES    = "ar.species"
VSS        = "ar.vss"
T_INIT     = 500.0        # iteration-1 fallback wall temperature [K]
NRUN       = 4000         # DSMC timesteps
NAVE       = 2000         # sample the flux over the last NAVE steps
SEED_MODE  = "fixed"      # "fixed" -> same seed every iteration (deterministic)
                          # "vary"  -> new seed every iteration (honest noise)
SEED       = 12345
# ─────────────────────────────────────────────────────────────────────────

DECK = "cpl.sparta"
TSURF_IN = "tsurf.in"
FLUX_OUT = "flux.out"



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
        d = json.loads(p.read_text() or "{}")
    except json.JSONDecodeError:
        return None
    return _partner_block(d)


def read_surf_elements():
    """SURF_FILE -> (number of lines, their centroids (n, 2) in id order), read by
    read_surf's own rules (the docstring's FILE FORMATS). Stops naming the rule a
    file breaks, where SPARTA would stop or misread it."""
    rows = Path(SURF_FILE).read_text().splitlines()
    npts = nlin = 0
    i = 1                                          # the first line is always skipped
    while i < len(rows):
        s = rows[i].split("#")[0].strip()
        if not s:
            i += 1
            continue
        if "points" in s or "lines" in s:          # header: "<N> points" / "<M> lines"
            try:
                cnt = int(s.split()[0])
            except ValueError:
                raise SystemExit(f"{SURF_FILE} line {i + 1}: {rows[i]!r} is not '<N> points' "
                                 f"or '<M> lines'")
            if "points" in s:
                npts = cnt
            else:
                nlin = cnt
            i += 1
            continue
        break                                      # the first section keyword
    if nlin == 0:
        raise SystemExit(f"{SURF_FILE}: no '<M> lines' header line before the first section "
                         f"(its first line is always skipped, so a count written there is lost)")

    def section(i):
        """(keyword, its line index, index of its first row): blank lines, the keyword
        line, then ONE skipped line before the rows."""
        while i < len(rows) and not rows[i].split("#")[0].strip():
            i += 1
        return (rows[i].split("#")[0].strip() if i < len(rows) else ""), i, i + 2

    def row(k):
        return rows[k] if k < len(rows) else ""

    pts = []
    key, at, i = section(i)
    if key not in ("Points", "Lines"):
        raise SystemExit(f"{SURF_FILE} line {at + 1}: {row(at)!r} is neither a header line ('<N> "
                         f"points', '<M> lines') nor a section keyword ('Points', 'Lines')")
    if npts and key != "Points":
        raise SystemExit(f"{SURF_FILE}: the header says {npts} points and the file has no "
                         f"'Points' section before 'Lines'")
    if key == "Points":
        if npts == 0:
            raise SystemExit(f"{SURF_FILE}: a 'Points' section and no '<N> points' header line "
                             f"(the first line is always skipped, so a count written there is lost)")
        for k in range(i, i + npts):
            f = row(k).split()
            try:
                pts.append((float(f[1]), float(f[2])))
            except (ValueError, IndexError):
                raise SystemExit(f"{SURF_FILE} line {k + 1}: {row(k)!r} is not a Points row 'id x y' "
                                 f"(the line after 'Points' is skipped, and the {npts} rows follow "
                                 f"it with no blank line)")
        key, at, i = section(i + npts)
        if key != "Lines":
            raise SystemExit(f"{SURF_FILE} line {at + 1}: {row(at)!r} where the 'Lines' section "
                             f"should begin, after the {npts} Points rows")
    lines = []
    for k in range(i, i + nlin):
        f = row(k).split()
        try:
            if pts and len(f) in (3, 4):                     # id [type] p1 p2
                a, b = pts[int(f[-2]) - 1], pts[int(f[-1]) - 1]
            elif not pts and len(f) in (5, 6):               # id [type] x1 y1 x2 y2
                a, b = (float(f[-4]), float(f[-3])), (float(f[-2]), float(f[-1]))
            else:
                raise ValueError
            lines.append((int(f[0]), a, b))
        except (ValueError, IndexError):
            raise SystemExit(f"{SURF_FILE} line {k + 1}: {row(k)!r} is not a Lines row "
                             + ("'id p1 p2'" if pts else "'id x1 y1 x2 y2'")
                             + f" (the line after 'Lines' is skipped, and the {nlin} rows follow "
                             f"it with no blank line)")
    lines.sort()
    cen = np.array([[0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])] for _, a, b in lines])
    return len(lines), cen


def _arclen(p):
    """Normalised cumulative chord length along an ordered point list."""
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    return d / d[-1] if d[-1] > 0 else d


def sample_on(imp, key, fallback, cen):
    """Map partner samples onto SPARTA's element centroids using normalised
    arc length along the interface curve as the 1D interface parameter
    (the curved-interface analogue of SPEC.md's tangential coordinate y).
    Both sides must list their interface points ordered along the curve."""
    n = len(cen)
    if not imp or not imp.get("coordinates"):
        return np.full(n, float(fallback))
    src = np.asarray(imp["coordinates"], float)[:, :2]
    vs = np.asarray(imp.get(key, []), float).ravel()
    if vs.size != len(src) or len(src) < 2:
        return np.full(n, float(fallback))
    return np.interp(_arclen(cen), _arclen(src), vs)


def write_tsurf(t):
    """TSURF_IN in the `custom surf ... file` format: comment, blank, 'N 1', 'id value'."""
    L = ["# per-surf wall temperature written by the openPASO coupling driver", ""]
    L.append(f"{len(t)} 1")
    for i, v in enumerate(t, 1):
        L.append(f"{i} {float(v):.10g}")
    Path(TSURF_IN).write_text("\n".join(L) + "\n")


# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
def write_deck(seed):
    Path(DECK).write_text(f"""\
seed                {seed}
dimension           2
global              nrho 4.247e19 fnum 7e14 gridcut 0.01 comm/style all comm/sort yes
timestep            3.5e-7
boundary            o ro p
create_box          -0.2 0.65 0.0 0.4 -0.5 0.5
create_grid         30 15 1 block * * *
species             {SPECIES} Ar
mixture             all vstream 2634.1 0 0 temp 200.0
collide             vss all {VSS}
collide_modify      vremax 1000 yes
read_surf           {SURF_FILE} group 1

custom              surf create tsurf float 0 file {TSURF_IN} 1 tsurf
surf_collide        1 diffuse s_tsurf 1.0
surf_modify         1 collide 1

fix                 in emit/face all xlo twopass
create_particles    all n 0 twopass

compute             1 surf all all etot
fix                 1 ave/surf all 1 {NAVE} {NAVE} c_1[1] ave one
dump                1 surf all {NRUN} {FLUX_OUT} id v1x v1y v2x v2y f_1 s_tsurf
dump_modify         1 pad 0

stats               {NRUN}
stats_style         step cpu np nscoll
run                 {NRUN}
""")
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end


def parse_dump(path, n):
    """The LAST snapshot of the surf dump `path` as rows [id, v1x, v1y, v2x, v2y,
    etot, T], sorted by id; its columns are found by the names on its "ITEM: SURFS"
    line (id v1x v1y v2x v2y, one f_ or c_ column, one s_ column)."""
    txt = Path(path).read_text().splitlines()
    starts = [i for i, l in enumerate(txt) if l.startswith("ITEM: SURFS")]
    if not starts:
        raise SystemExit(f"no 'ITEM: SURFS' block in {path}")
    names = txt[starts[-1]].split()[2:]
    flux = [c for c in names if c.startswith(("f_", "c_"))]
    temp = [c for c in names if c.startswith("s_")]
    if not {"id", "v1x", "v1y", "v2x", "v2y"} <= set(names) or len(flux) != 1 or len(temp) != 1:
        raise SystemExit(f"{path} has the columns {names}; this reader takes id v1x v1y v2x v2y, "
                         f"ONE f_ (or c_) column holding the etot tally and ONE s_ column holding "
                         f"the wall temperature, as in: dump <ID> surf <group> <N> {path} id v1x "
                         f"v1y v2x v2y f_<fix> s_tsurf")
    cols = [names.index(c) for c in ("id", "v1x", "v1y", "v2x", "v2y")] + \
           [names.index(flux[0]), names.index(temp[0])]
    rows = []
    for l in txt[starts[-1] + 1:]:
        if l.startswith("ITEM:"):
            break
        f = l.split()
        if len(f) >= len(names):
            rows.append([float(f[k]) for k in cols])
    a = np.asarray(rows, float).reshape(-1, 7)
    a = a[np.argsort(a[:, 0])]
    ids = a[:, 0].astype(int).tolist()
    if ids != list(range(1, n + 1)):
        raise SystemExit(f"{path} holds {len(ids)} elements"
                         + (f" with ids {ids[0]}..{ids[-1]}" if ids else "")
                         + f"; it must hold the {n} lines of {SURF_FILE}, ids 1..{n}: read "
                         f"{SURF_FILE} FIRST (a second read_surf numbers its lines after them) "
                         f"and dump only its group")
    return a


# ── THE BINARY (served) ─ keep this block. A bare name runs only from PATH, and a SPARTA
#    built in a checkout is often not on it: measured, the placeholder name died on
#    FileNotFoundError in two runs.
_env = os.environ.get("SPARTA_BINARY", "")
_spa = (SPARTA if Path(SPARTA).is_file() else shutil.which(SPARTA)
        or (_env if _env and Path(_env).is_file() else None))
if not _spa:
    raise SystemExit(f"SPARTA BINARY: {SPARTA!r} is not a file, not a command on PATH, and "
                     f"SPARTA_BINARY names none. Set SPARTA to the path discover(query='list') prints "
                     f"for SPARTA, after 'SPARTA at'.")
SPARTA = _spa

# ── THE DATA FILES (served) ─ keep this block. SPARTA opens every file a deck names
#    relative to the directory it runs in, and couple() copies the files listed in a
#    participant's data_files into its work_dir before the first iteration.
_dist = Path(SPARTA).resolve().parent.parent / "data" if Path(SPARTA).is_file() else None
for f in (SURF_FILE, SPECIES, VSS):
    if not Path(f).is_file():
        src = next((d / f for d in ([Path(DATA_DIR)] if DATA_DIR else []) + ([_dist] if _dist else [])
                    if (d / f).is_file()), None)
        if src is None:
            sys.stderr.write(f"missing SPARTA data file {f}: not in this directory, not in "
                             f"DATA_DIR ({DATA_DIR or 'unset'})"
                             + (f" and not in the SPARTA distribution's {_dist}" if _dist else "")
                             + ". couple() copies the files listed in this participant's "
                             "data_files into its work_dir before the first iteration; it does "
                             "not read the deck for file names. Run standalone with the file "
                             "beside this script or in DATA_DIR.\n")
            sys.exit(3)
        Path(f).write_bytes(src.read_bytes())

n_elem, cen = read_surf_elements()
imp = read_imports()
t_wall = sample_on(imp, "values", T_INIT, cen)
write_tsurf(t_wall)

it = 0
ctr = Path(".iter")
if ctr.is_file():
    it = int(ctr.read_text().strip() or 0)
ctr.write_text(str(it + 1))
seed = SEED if SEED_MODE == "fixed" else SEED + 1000 * (it + 1)
write_deck(seed)

# ── THE RUN (served) ─ keep this block. SPARTA's console is passed through: the
#    per-level run log your task asks for is that console, and a log carrying only
#    this wrapper's prose cannot establish which code ran on this side.
Path(FLUX_OUT).unlink(missing_ok=True)
r = subprocess.run([SPARTA, "-in", DECK], capture_output=True, text=True, timeout=3600)
if r.stdout:
    print(r.stdout, end="")
if r.stderr:
    sys.stderr.write(r.stderr)
if r.returncode != 0 or not Path(FLUX_OUT).is_file():
    _said = [l.strip() for l in (r.stdout or "").splitlines() if l.startswith("ERROR")]
    sys.stderr.write(f"SPARTA failed rc={r.returncode}"
                     + (f": {_said[-1]}" if _said else "")
                     + ("" if r.returncode else f", and wrote no {FLUX_OUT}: the deck's dump surf "
                        f"must write that file") + "\n")
    sys.exit(1)

# ── PARTICLE SELF-CHECK (served) ─ keep this block. SPARTA gives no error when no
#    grid cell lies on the flow side of a surface line: read_surf prints
#    "<a> <b> <c> = cells outside/inside/overlapping surfs" with a = 0,
#    create_particles makes 0 particles, the run ends "with 0 particles" and exits
#    0, and every tally in the dump is zero. Read from SPARTA's own console (r).
_chk_con = str(getattr(r, "stdout", "") or "")
_chk_cells = re.findall(r"^\s*(\d+) (\d+) (\d+) = cells outside/inside/overlapping surfs",
                        _chk_con, re.M)
_chk_np = re.findall(r"^Loop time of .* with (\d+) particles", _chk_con, re.M)
_chk_made = re.findall(r"^Created (\d+) particles", _chk_con, re.M)
if _chk_cells and int(_chk_cells[-1][0]) == 0:
    raise SystemExit(
        f"PARTICLE SELF-CHECK: read_surf reported 0 cells outside the surfaces "
        f"('{' '.join(_chk_cells[-1])} = cells outside/inside/overlapping surfs'): every grid "
        f"cell is inside a body, so there is no gas"
        + (f" (create_particles made {_chk_made[-1]} particles)" if _chk_made else "")
        + ". SPARTA puts the gas on the side a line's normal points to, N = (0,0,1) x "
        f"(p2 - p1): walking from p1 to p2, the gas is on your left. Swap p1 and p2 of each "
        f"line in {SURF_FILE} whose gas is on its right, and run again.")
if _chk_np and int(_chk_np[-1]) == 0:
    raise SystemExit(
        "PARTICLE SELF-CHECK: SPARTA ran to the end with 0 particles, so every tally it wrote "
        "is zero. "
        + (f"read_surf reported {int(_chk_cells[-1][0])} cells outside the surfaces, so the "
           "gas has room; " if _chk_cells else "")
        + (f"create_particles made {_chk_made[-1]} particles. " if _chk_made else
           "no create_particles line ran. ")
        + ("With 'Created 0 particles', the count comes from 'global nrho <n> fnum <F>', which "
           "must come before create_particles (after it, or left out, SPARTA uses nrho = 1 and "
           "fnum = 1). " if _chk_made and int(_chk_made[-1]) == 0 else
           "None of them was left at the end; the end-of-run lines 'Boundary exits' and "
           "'Particles stuck' count where they went. " if _chk_made else "")
        + "Read the console above before changing anything else.")
# THE RUN-LOG CONTRACT LINE, in this code's own currency. A DSMC cell refines by
# PARTICLE COUNT on a fixed grid, not by adding cells, so the `NDOF = <integer>`
# line the log contract asks for carries the particle count SPARTA reported at the
# end of the run -- the number that grows between your levels.
if _chk_np:
    print(f"\nNDOF = {int(_chk_np[-1])}")

a = parse_dump(FLUX_OUT, n_elem)
cx = 0.5 * (a[:, 1] + a[:, 3])
cy = 0.5 * (a[:, 2] + a[:, 4])
q_out = a[:, 5]          # etot: net energy flux INTO the wall = OUT of the gas
t_used = a[:, 6]

# ── EXPORT SELF-CHECK ─ keep this block. An export that is not finite is
#    worthless however well the iteration behaved, and a tally that came out
#    identically zero still couples, still converges and still hands in tidy
#    levels. The DSMC tally is the load the solid sees.
_chk_vals = np.asarray(q_out, float).ravel()
if not np.isfinite(_chk_vals).all():
    raise SystemExit("EXPORT SELF-CHECK: non-finite etot in the surf dump, so "
                     "nothing was exported")
if _chk_vals.size and np.abs(_chk_vals).max() == 0.0:
    raise SystemExit("EXPORT SELF-CHECK: every exported etot is exactly zero: no "
                     "particle collision was tallied on these surface elements in "
                     "the sampling window. Read the 'Created <N> particles' line "
                     "and the read_surf line '<a> <b> <c> = cells outside/inside/"
                     "overlapping surfs' in SPARTA's console; do not couple on")

Path("exports.json").write_text(json.dumps({
    "field_name": "wall_temperature",
    "n_points": int(n_elem),
    "coordinates": [[float(x), float(y)] for x, y in zip(cx, cy)],
    "values": [float(v) for v in t_used],
    "normal_fluxes": [float(v) for v in q_out],
}, indent=2))
print(f"SPARTA it={it+1} seed={seed} n={n_elem} T_wall=[{t_wall.min():.2f},"
      f"{t_wall.max():.2f}] sum(etot)={q_out.sum():.6e} mean={q_out.mean():.6e}")
