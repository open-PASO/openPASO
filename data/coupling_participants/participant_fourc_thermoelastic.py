"""4C as the DIRICHLET side of a THERMO-ELASTIC coupling (steady thermoelasticity, plane strain).
    imports  values        = [T, ux, uy]   per interface point
    exports  normal_fluxes = [qn, qx, qy]  qn = -(k grad T).n_out, (qx, qy) = -(sigma_tot.n_out)
The traction follows the heat flux's sign rule (minus the flux through n_out): the two sides' exports
cancel and the Neumann partner applies them UNCHANGED; the task's OUTWARD traction is MINUS (qx, qy).
TWO 4C RUNS PER ITERATION (your hole): run T, a 2-D Scalar_Transport deck with CALCFLUX_BOUNDARY
"diffusive"; run U, a Thermo_Structure_Interaction deck on a ONE-ELEMENT-THICK SOLIDSCATRA HEX8 slab
with u_z pinned in EVERY Dirichlet entry (plane strain), tsi_oneway, COUPVARIABLE Temperature,
MAT_Struct_ThermoStVenantK (alpha = beta/(3 lambda + 2 mu), INITTEMP 0), `TAG: monitor_reaction` on the
interface point conditions: (f_layer0 + f_layer1)/(h*t_z) of the monitored reactions IS the exported
traction (measured order 2.0). SERVED: handshake, finish diagnosis, recovery, self-checks, exports.
YOURS: the mesh (2-D layout and slab), both decks, both runs.
config.json: {"level","nx","ny","x0","x1","y0","y1","k","lam","mu","beta","iface":"left|right|
bottom|top","source_T","source_ux","source_uy" (4C expressions: '^', lowercase pi),"fourc_bin","fourc_ld"}
"""
import atexit
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    CFG = {**json.loads(Path("config.json").read_text()), **json.loads(os.environ.get("OPENPASO_CONFIG_JSON") or "{}")}
except (ValueError, TypeError) as _e:
    raise SystemExit(f"config.json / OPENPASO_CONFIG_JSON could not be read: {_e}")
NX, NY = int(CFG["nx"]), int(CFG["ny"])
X0, X1, Y0, Y1 = (float(CFG["x0"]), float(CFG["x1"]),
                  float(CFG["y0"]), float(CFG["y1"]))
KV = float(CFG["k"])
LAM, MU, BETA = float(CFG["lam"]), float(CFG["mu"]), float(CFG["beta"])
IF = CFG.get("iface", "right")
SIDE = CFG.get("side", "dirichlet")
if SIDE != "dirichlet":
    raise SystemExit("this contract is the DIRICHLET role (T, ux, uy in; qn, qx, qy out); 4C on the "
                     "NEUMANN side is not served: give that role to the partner code.")
# 4C's material takes E, nu and the linear expansion coefficient; the task
# gives Lame parameters and beta. These are the exact conversions.
E_MOD = MU * (3.0 * LAM + 2.0 * MU) / (LAM + MU)
NU = LAM / (2.0 * (LAM + MU))
ALPHA = BETA / (3.0 * LAM + 2.0 * MU)
# SOURCES AS 4C EXPRESSION STRINGS ('^', lowercase pi, x, y; "0.0" = none): they reach the solve only
# through FUNCT blocks in YOUR decks, never through a Python function.
SRC_T = str(CFG.get("source_T", "0.0")).replace("**", "^")
SRC_UX = str(CFG.get("source_ux", "0.0")).replace("**", "^")
SRC_UY = str(CFG.get("source_uy", "0.0")).replace("**", "^")

# stale reaction files of an earlier run would be SUMMED into this run's tractions (measured)
for _old in glob.glob("*_monitor_dbc.yaml"):
    try:
        os.remove(_old)
    except OSError:
        pass
# ---- the partner's interface samples, mapped onto THIS side (handshake) ----
imp = {}
if Path("imports.json").is_file():
    imp = json.loads(Path("imports.json").read_text() or "{}")


def partner_values(y_or_x):
    """The partner's (T, ux, uy) at one of THIS side's interface points, each
    component interpolated on its own along the interface (the driver does not
    interpolate between the two meshes; one np.interp over a flattened (N, 3)
    array interleaves the components -- right length, every number wrong).
    Empty on iteration 1: fall back to (0, 0, 0)."""
    for _n, d in imp.items():
        co = d.get("coordinates") or []
        va = d.get("values") or []
        if not (co and va and len(va) == len(co)):
            continue
        va = np.asarray(va, float)
        if va.ndim == 1:
            va = va.reshape(-1, 1)
        if va.shape[1] < 3:
            continue
        ax = 1 if IF in ("left", "right") else 0
        s = np.asarray([c[ax] for c in co], float)
        o = np.argsort(s)
        t = min(max(float(y_or_x), s[o][0]), s[o][-1])
        return tuple(float(np.interp(t, s[o], va[o, c])) for c in range(3))
    return (0.0, 0.0, 0.0)


# ── DID 4C FINISH? (served: when a run leaves no output, name the cause) ─
def why_4c_did_not_finish(tag=""):
    """When a run leaves no usable output: 4C's own error line, then the deck defects measured on worker decks."""
    _why = []
    try:
        for _lg in sorted(glob.glob("*.log")) + sorted(glob.glob("*.txt")):
            try:
                _lines = open(_lg, errors="ignore").read().splitlines()
            except OSError:
                continue
            _hit = False
            for _i, _ln in enumerate(_lines):
                # 4C's own stop: the PROC 0 ERROR block, or the YAML reader's `ERROR:` line; ends where the stack frames begin
                if "PROC 0 ERROR" in _ln or _ln.startswith("ERROR:"):
                    _said = [_ln.strip()] if _ln.startswith("ERROR:") else []
                    for l in _lines[_i + 1:_i + 16]:
                        if not l.strip():
                            continue
                        if re.match(r"\s*\d+#\s", l) or set(l.strip()) <= set("=-"):
                            break
                        if "MPI_ABORT" not in l:
                            _said.append(l.strip())
                    _why.append(f"4C said ({_lg}): " + " | ".join(_said[:8]))
                    _hit = True
                    break
            if not _hit:   # a crash without an error message (measured: a zero-area flux boundary)
                for _i, _ln in enumerate(_lines):
                    if "*** Process received signal ***" in _ln:
                        _sig = next((l.split("Signal:", 1)[1].strip() for l in _lines[_i:_i + 4] if "Signal:" in l), "a signal")
                        _bef = [l.strip() for l in _lines[max(0, _i - 12):_i] if l.strip() and not set(l.strip()) <= set("+-|=")]
                        _why.append(f"4C died on {_sig} ({_lg}) with no error message; the last thing it printed: " + " | ".join(_bef[-2:])
                                    + " -- a flux table dividing by a ZERO boundary area means the flux-calc DLINE shares no edge with any element")
                        break
        for _deck in sorted(glob.glob("*.4C.yaml")) or sorted(glob.glob("*.yaml")):
            _txt = open(_deck, errors="ignore").read()
            _secs = re.findall(r"^([A-Z][A-Z0-9 _/.:-]*?):\s*$", _txt, re.M)
            _dup = sorted({x for x in _secs if _secs.count(x) > 1})
            if _dup:
                _why.append(f"{_deck}: section(s) written twice: " + ", ".join(_dup))
            # section names and materials are judged by the binary's own grammar in check_input(solver='fourc', input_path=<deck>)
            if "Thermo_Structure_Interaction" in _txt:
                if "CLONING MATERIAL MAP" not in _txt:
                    _why.append(f"{_deck}: TSI needs a CLONING MATERIAL MAP pairing the structure material with the MAT_Fourier thermal material")
                if "COUPVARIABLE" not in _txt or "Temperature" not in _txt.split("COUPVARIABLE", 1)[-1][:40]:
                    _why.append(f"{_deck}: TSI DYNAMIC/PARTITIONED needs COUPVARIABLE \"Temperature\" (the default gives zero thermal strain)")
                if re.search(r"\b(WALL|SOLID) QUAD4\b|\bTRI3\b", _txt):
                    _why.append(f"{_deck}: 4C has no 2-D thermo-elastic element; use the one-element-thick SOLIDSCATRA HEX8 slab")
                if "monitor_reaction" in _txt and "IO/MONITOR STRUCTURE DBC" not in _txt:
                    _why.append(f"{_deck}: TAG: monitor_reaction writes nothing without an IO/MONITOR STRUCTURE DBC section")
                if "DESIGN VOL THERMO DIRICH" in _txt:
                    _why.append(f"{_deck}: DESIGN VOL THERMO DIRICH imposes the temperature volume-wide; use SURF (outer) and POINT (interface) THERMO DIRICH")
                if "IO/RUNTIME VTK OUTPUT/STRUCTURE" not in _txt or not re.search(r"DISPLACEMENT:\s*true", _txt, re.I):
                    _why.append(f"{_deck}: no structure VTU without `IO/RUNTIME VTK OUTPUT` (INTERVAL_STEPS 1) and `IO/RUNTIME VTK OUTPUT/STRUCTURE` (OUTPUT_STRUCTURE true, DISPLACEMENT true) -- the run finishes 'normally' with nothing for the recovery to read")
                if "THERMAL DYNAMIC/RUNTIME VTK OUTPUT" not in _txt or not re.search(r"TEMPERATURE:\s*true", _txt, re.I):
                    _why.append(f"{_deck}: no thermo VTU without `THERMAL DYNAMIC/RUNTIME VTK OUTPUT` (OUTPUT_THERMO true, TEMPERATURE true)")
                for _b in re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\s*$)", _txt, flags=re.M):   # a T value in the structural family: '1 DOFs given but 3 expected'
                    _hd = _b.split(":", 1)[0].strip()
                    if re.fullmatch(r"DESIGN (POINT|LINE|SURF|VOL) DIRICH CONDITIONS", _hd) and re.search(r"\bNUMDOF:\s*1\b", _b):
                        _why.append(f"{_deck}: {_hd} has an entry with NUMDOF 1 -- that family carries the slab's 3 displacement dofs; a temperature value belongs in DESIGN {_hd.split()[1]} THERMO DIRICH CONDITIONS")
            if "Scalar_Transport" in _txt:
                if "THERMAL DYNAMIC:" in _txt and "SCALAR TRANSPORT DYNAMIC:" not in _txt:
                    _why.append(f"{_deck}: Scalar_Transport needs `SCALAR TRANSPORT DYNAMIC`, not `THERMAL DYNAMIC`")
                if "CALCFLUX_BOUNDARY" not in _txt or "FLUX CALC" not in _txt:
                    _why.append(f"{_deck}: needs CALCFLUX_BOUNDARY \"diffusive\" AND a `SCATRA FLUX CALC LINE CONDITIONS` entry on the interface DLINE")
            _badkw = sorted({w for w in re.findall(r'"NODE\s+\d+\s+(D[A-Z]+)\s+\d+"', _txt) if w not in ("DNODE", "DLINE", "DSURFACE", "DSURF", "DVOL", "DVOLUME")})
            if _badkw:
                _why.append(f"{_deck}: topology entries use {', '.join(_badkw)} -- 4C takes DNODE in DNODE-NODE TOPOLOGY, DLINE in DLINE-NODE TOPOLOGY, DSURF or DSURFACE in DSURF-NODE TOPOLOGY and DVOL or DVOLUME in DVOL-NODE TOPOLOGY; any other word stops 4C with 'Wrong design node name: <word>. Expected <section word>.'")
            _topo = set(re.findall(r"\b(DNODE|DLINE|DSURF|DVOL)\w*\s+(\d+)", _txt))   # DSURF(ACE), DVOL(UME)
            for _b in re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\s*$)", _txt, flags=re.M):
                _head = _b.split(":", 1)[0].strip()
                _kw = re.search(r"\b(POINT|LINE|SURF|VOL)\b", _head) if _head.endswith("CONDITIONS") else None
                if _kw:   # every condition family with a geometry word (SCATRA FLUX CALC LINE CONDITIONS too)
                    _kind = {"POINT": "DNODE", "LINE": "DLINE", "SURF": "DSURF", "VOL": "DVOL"}[_kw.group(1)]
                    _missing = sorted({x for x in re.findall(r"\bE:\s*(\d+)", _b) if (_kind, x) not in _topo}, key=int)
                    if _missing:
                        _why.append(f"{_deck}: {_head} names E id(s) {', '.join(_missing[:6])} that no *-NODE TOPOLOGY section defines -- E is the design-entity id of a topology line (`NODE <n> DNODE <E>`), never a node number")
    except Exception as _e:                # noqa: BLE001
        _why.append(f"(diagnosis failed: {_e!r})")
    if not _why:
        _why.append("no 4C error line in any *.log here -- run the binary line-buffered (stdbuf -oL -eL) with its console in a log next to the deck")
    _why.append("check_input(solver='fourc', input_path=<deck>) names every defect the binary's grammar can see (section names with the closest known, materials and their parameters, condition ids, element geometry) in one call")
    return f"4C DID NOT FINISH{(' (' + tag + ')') if tag else ''} -- " + "; ".join(_why)


def _diagnose_at_exit():
    if not Path("exports.json").is_file():
        print(why_4c_did_not_finish(), file=sys.stderr, flush=True)


atexit.register(_diagnose_at_exit)

# ── THE HOLE (yours): the mesh, the two decks and the two runs. openPASO serves the
#    handshake above and the recovery below; what 4C solves is YOUR deck. Build this
#    subdomain's 2-D node layout (NX x NY on [X0, X1] x [Y0, Y1]) and, for deck U, the
#    one-element-thick slab (a second node layer at z = TZ). Classify your interface
#    nodes yourself: partner_values(coord) returns the partner's (T, ux, uy) at one
#    interface coordinate, and the two interface ENDPOINTS are outer nodes that keep the
#    outer datum. Write deck T (Scalar_Transport, 2-D, with 4C's consistent boundary flux
#    on the interface line; its Dirichlet data go in DESIGN LINE / POINT DIRICH CONDITIONS --
#    the scalar field never reads THERMO DIRICH) and deck U (the TSI slab: the partner's T and (ux, uy) as
#    POINT conditions on both layers with their reactions monitored, and u_z pinned in EVERY Dirichlet
#    entry -- ONOFF [1, 1, 1] with VAL [ux, uy, 0] on the interface points, [0, 0, 0] on the outer nodes:
#    4C applies VOL, SURF, LINE, POINT in that order and an entry's ONOFF 0 FREES that dof on its nodes)
#    for the problem you were given: the grammar is `4C -p`, prepare_simulation(solver='fourc',
#    physics=...) and knowledge(solver='fourc'). Before a run, check_input(solver='fourc',
#    input_path=<deck>) names every defect in one call. Run each deck line-buffered with its console in a log
#    (stdbuf -oL -eL <bin> <deck> <prefix> > <deck>.log 2>&1); on a non-zero exit FALL
#    THROUGH, the served check reads the log. LEAVE BEHIND exactly these names:
#      nodes     the list of (x, y) of the 2-D layout; deck node id on the z = 0 layer = index + 1
#      interior  the 1-based ids of the interface nodes WITHOUT the two endpoints, in order along it
#      TZ        the slab thickness (one well-shaped HEX8 layer)
#      OUT_T, OUT_U   the two output prefixes;  DECK_U   the file name you wrote deck U to
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
nodes = interior = TZ = OUT_T = OUT_U = DECK_U = None     # your mesh, decks and runs define these six
raise SystemExit("the mesh-decks-and-runs hole above the recovery is not filled")
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end


# ── RECOVERY FROM 4C's OWN OUTPUTS (served): boundary flux VTU, displacement VTU, reaction yaml ──
import meshio  # noqa: E402

# ── 4C's OWN CONSOLE, ECHOED (served): this script's stdout becomes the level's run log, and a run
#    log is credited to 4C only by 4C's own lines, never by the NDOF line alone.
for _lg in sorted(glob.glob("*.log")):
    if _lg.startswith("participant_output") or "level" in _lg:      # the coupling tool's own captures, never re-echoed
        continue
    try:
        _txt = Path(_lg).read_text(errors="ignore")
    except OSError:
        continue
    if "── 4C console" in _txt:                                      # an earlier echo, not a deck console
        continue
    if "4C" in _txt[:4000] or "PROC 0" in _txt or "Finalised step" in _txt:
        print(f"── 4C console {_lg} ──")
        print(_txt if len(_txt) < 60000 else _txt[-60000:])

# ── YOUR DECKS, CHECKED BEFORE ANYTHING IS READ (served): 4C drops a condition whose E id no topology
#    section defines and RUNS THE WRONG PROBLEM to 'finished normally' (measured). Refused here.
for _dk in sorted(glob.glob("*.4C.yaml")) or [p for p in sorted(glob.glob("*.yaml")) if "monitor_dbc" not in p]:
    _txt = Path(_dk).read_text(errors="ignore")
    _topo = set(re.findall(r"\b(DNODE|DLINE|DSURF|DVOL)\w*\s+(\d+)", _txt))   # DSURF(ACE), DVOL(UME)
    _lost = []
    for _b in re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\s*$)", _txt, flags=re.M):
        _head = _b.split(":", 1)[0].strip()
        _kw = re.search(r"\b(POINT|LINE|SURF|VOL)\b", _head) if _head.endswith("CONDITIONS") else None
        if _kw:   # every condition family with a geometry word (SCATRA FLUX CALC LINE CONDITIONS too)
            _kind = {"POINT": "DNODE", "LINE": "DLINE", "SURF": "DSURF", "VOL": "DVOL"}[_kw.group(1)]
            _lost += [f"{_head} E {x}" for x in re.findall(r"\bE:\s*(\d+)", _b) if (_kind, x) not in _topo]
    if _lost:
        raise SystemExit(f"DECK CHECK: {_dk} puts conditions on E ids that no *-NODE TOPOLOGY section defines "
                         f"({'; '.join(_lost[:6])}): 4C dropped them silently, so the run solved a different "
                         f"problem. E is the DESIGN-ENTITY id (the number after DNODE/DLINE/DSURFACE/DVOL in a topology "
                         f"line), never a node number: a node joins entity E through `NODE <n> DNODE <E>`. Add the "
                         f"DNODE/DLINE/DSURF/DVOL-NODE TOPOLOGY entries for those ids.")
    # THERMO DIRICH in a Scalar_Transport deck is ignored: no Dirichlet condition, a singular solve, a uniform ~1e13
    # field with rc 0 (measured; the same deck with DIRICH headers gave the expected field)
    if ("SCALAR TRANSPORT DYNAMIC" in _txt
            and re.search(r"^DESIGN (POINT|LINE|SURF|VOL) THERMO DIRICH CONDITIONS:", _txt, re.M)
            and not re.search(r"^DESIGN (POINT|LINE|SURF|VOL) DIRICH CONDITIONS:", _txt, re.M)):
        raise SystemExit(f"DECK CHECK: {_dk} is a Scalar_Transport deck whose Dirichlet data sit only in THERMO DIRICH "
                         f"families, which the scalar field never reads: NO Dirichlet condition, a singular solve, a uniform "
                         f"~1e13 field with rc 0. Put deck T's outer datum and interface values in DESIGN LINE / POINT DIRICH "
                         f"CONDITIONS (NUMDOF 1); THERMO DIRICH belongs to deck U.")
    # twisted or clockwise 2-D elements (zero/negative area from the deck's own coordinates) solve nothing
    _cxy = {int(n): (float(x), float(y)) for n, x, y in re.findall(r'"NODE\s+(\d+)\s+COORD\s+(\S+)\s+(\S+)\s+\S+"', _txt)}
    _twist = []
    for _e, _k, _ids in re.findall(r'"\s*(\d+)\s+\w+\s+(QUAD4|TRI3)\s+((?:\d+\s+)+)', _txt):
        _nn = [int(i) for i in _ids.split()][:4 if _k == "QUAD4" else 3]
        if len(_nn) >= 3 and all(i in _cxy for i in _nn):
            _p = [_cxy[i] for i in _nn]
            if 0.5 * sum(_p[q][0] * _p[(q + 1) % len(_p)][1] - _p[(q + 1) % len(_p)][0] * _p[q][1] for q in range(len(_p))) <= 1e-14:
                _twist.append((_e, _nn))
    if _twist:
        raise SystemExit(f"DECK CHECK: {_dk} has {len(_twist)} element(s) with zero or negative area (first: element {_twist[0][0]} "
                         f"nodes {' '.join(map(str, _twist[0][1]))}). Every element's nodes must run counter-clockwise: for node "
                         f"id = i + 1 + (NX + 1) * j the quad of cell (i, j) is (id, id + 1, id + NX + 2, id + NX + 1).")
    # table rows that are not quoted strings: the YAML reader stops at the first bare token (measured)
    _unq = [(_b.split(":", 1)[0].strip(), _r.strip()) for _b in re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\s*$)", _txt, flags=re.M)
            if (_b.split(":", 1)[0].strip().endswith((" ELEMENTS", "-NODE TOPOLOGY")) or _b.split(":", 1)[0].strip() == "NODE COORDS")
            for _r in re.findall(r"^\s*-\s+([^\"'\n][^\n]*)$", _b, re.M) if re.match(r"(\d+\s+\w|NODE\s+\d+)", _r)]
    if _unq:
        raise SystemExit(f"DECK CHECK: {_dk} has {len(_unq)} table row(s) that are not quoted YAML strings (first, in {_unq[0][0]}: "
                         f"`- {_unq[0][1][:60]}`): every NODE COORDS, element and topology row is ONE quoted string, `- \"...\"`. "
                         f"4C's reader stops at the first bare token with 'could not find ':' colon after key'.")
    # Dirichlet on EVERY node of a field leaves nothing to solve: the field is the prescribed data (measured)
    _nodes = {int(a) for a in re.findall(r'"NODE\s+(\d+)\s+COORD\b', _txt)}
    _tp = {}
    for _n, _k, _e in re.findall(r'"NODE\s+(\d+)\s+(DNODE|DLINE|DSURF|DVOL)\w*\s+(\d+)"', _txt):
        _tp.setdefault((_k, int(_e)), set()).add(int(_n))
    _cov = {}
    for _b in re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\s*$)", _txt, flags=re.M):
        _head = _b.split(":", 1)[0].strip()
        _kw = re.search(r"\b(POINT|LINE|SURF|VOL)\b", _head) if ("DIRICH" in _head and _head.endswith("CONDITIONS")) else None
        if not _kw:
            continue
        _fam = "temperature" if ("THERMO" in _head or "TRANSPORT" in _head or "Scalar_Transport" in _txt) else "displacement"
        for _en in re.split(r"^\s*-\s", _b, flags=re.M)[1:]:
            _e = re.search(r"\bE:\s*(\d+)", _en); _on = re.search(r"ONOFF:\s*\[([^\]]*)\]", _en)
            _fl = [x.strip() for x in _on.group(1).split(",")] if _on else ["1"]
            _inpl = any(f == "1" for f in _fl[:2]) if _fam == "displacement" else _fl[0] == "1"
            if _e and _inpl:
                _cov.setdefault(_fam, set()).update(_tp.get(({"POINT": "DNODE", "LINE": "DLINE", "SURF": "DSURF", "VOL": "DVOL"}[_kw.group(1)], int(_e.group(1))), set()))
    for _fam, _s in _cov.items():
        if len(_nodes) >= 4 and _s >= _nodes:
            raise SystemExit(f"DECK CHECK: {_dk} pins EVERY node of the {_fam} field with Dirichlet conditions ({len(_nodes)} of {len(_nodes)}): "
                             f"nothing is left to solve -- 4C prints 'res-norm 0', converges at iteration 0 and the field is the prescribed data. "
                             f"The outer boundary is the BOUNDARY nodes only (x = x0, y = y0, y = y1 on both layers), the interface points the "
                             f"interface nodes alone; interior nodes carry no Dirichlet condition.")
    # a section written twice: 4C stops in its reader with 'Section X is defined more than once' (measured, te4c13)
    _heads = re.findall(r"^([A-Z][A-Z0-9 _/.:-]*?):\s*$", _txt, re.M)
    _dup = sorted({h for h in _heads if _heads.count(h) > 1})
    if _dup:
        raise SystemExit(f"DECK CHECK: {_dk} defines section(s) {', '.join(_dup)} more than once; 4C stops with "
                         f"'Section ... is defined more than once'. Merge each into ONE section with all its entries.")
    # HEX8 node order: bottom quad counter-clockwise, then the SAME four nodes on the top layer; any other order
    # stops in the element with 'ZERO OR NEGATIVE JACOBIAN DETERMINANT' naming no node (measured)
    _cxyz = {int(n): (float(x), float(y), float(z)) for n, x, y, z in re.findall(r'"NODE\s+(\d+)\s+COORD\s+(\S+)\s+(\S+)\s+(\S+)"', _txt)}
    if len({round(c[2], 9) for c in _cxyz.values()}) == 2:
        for _e, _ids in re.findall(r'"\s*(\d+)\s+\w+\s+HEX8\s+((?:\d+\s+){8})', _txt):
            _nn = [int(i) for i in _ids.split()]
            if not all(i in _cxyz for i in _nn):
                continue
            _p = [_cxyz[i] for i in _nn]
            _bz, _tz = {round(q[2], 9) for q in _p[:4]}, {round(q[2], 9) for q in _p[4:]}
            _xy = all(abs(_p[i][0] - _p[i + 4][0]) < 1e-9 and abs(_p[i][1] - _p[i + 4][1]) < 1e-9 for i in range(4))
            _ar = 0.5 * sum(_p[q][0] * _p[(q + 1) % 4][1] - _p[(q + 1) % 4][0] * _p[q][1] for q in range(4))
            if not (len(_bz) == 1 and len(_tz) == 1 and _bz != _tz and _xy and (_ar > 1e-14 if max(_tz) > max(_bz) else _ar < -1e-14)):
                raise SystemExit(f"DECK CHECK: {_dk} element {_e} (nodes {' '.join(map(str, _nn))}) is not a well-formed one-layer HEX8: "
                                 f"the order is the bottom quad counter-clockwise seen from +z, then the SAME four nodes on the top "
                                 f"layer -- for node id = i + 1 + (NX + 1) * j and N2 2-D nodes, cell (i, j) is (id, id + 1, id + NX + 2, "
                                 f"id + NX + 1, id + N2, id + 1 + N2, id + NX + 2 + N2, id + NX + 1 + N2). 4C would parse it and stop "
                                 f"later with 'ZERO OR NEGATIVE JACOBIAN DETERMINANT', naming no node.")


def _latest(pattern):
    vs = sorted(glob.glob(pattern))
    if not vs:
        return None
    # <field>-<step>-<rank>.vtu: the TRAILING number is the MPI rank; the step
    # is the middle one, and step 0 is the all-zero initial state
    def _step(p):
        m = re.match(r".*-(\d+)-\d+\.vtu$", p)
        return int(m.group(1)) if m else -1
    return max(vs, key=_step)


def _nodal(m, field, comps, zlayer=None):
    """Field values on the 2-D node layout. A 4C VTU repeats every node once
    per element (and the slab has two layers): collapse by coordinate."""
    pts = np.asarray(m.points)
    val = np.asarray(m.point_data[field])
    if val.ndim == 1:
        val = val.reshape(-1, 1)
    rows = np.arange(len(pts)) if zlayer is None else np.where(np.abs(pts[:, 2] - zlayer) < 1e-9)[0]
    key = {}
    for r in rows:
        key.setdefault((round(float(pts[r, 0]), 9), round(float(pts[r, 1]), 9)), []).append(r)
    out = np.zeros((len(nodes), comps))
    for i, (x, y) in enumerate(nodes):
        rr = key.get((round(float(x), 9), round(float(y), 9)))
        if not rr:
            raise SystemExit(f"4C VTU field {field!r} has no node at {(x, y)}: the deck's nodes and "
                             f"`nodes` disagree")
        out[i] = val[rr, :comps].mean(axis=0)
    return out


vtu_T = _latest(f"{OUT_T}-vtk-files/scatra-*.vtu")
if vtu_T is None:
    raise SystemExit(why_4c_did_not_finish("run T"))
mT = meshio.read(vtu_T)
T2d = _nodal(mT, "phi_1", 1)[:, 0]
fbname = next((da for da in mT.point_data if "flux_boundary" in da), None)
if fbname is None:
    raise SystemExit("run T wrote no flux_boundary field -- set CALCFLUX_BOUNDARY \"diffusive\" and "
                     "add a SCATRA FLUX CALC LINE CONDITIONS entry on the interface line")
FB = _nodal(mT, fbname, 2)
nrm = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "bottom": (0.0, -1.0), "top": (0.0, 1.0)}[IF]
q_n = [float(FB[n - 1, 0] * nrm[0] + FB[n - 1, 1] * nrm[1]) for n in interior]

vtu_U = _latest(f"{OUT_U}-vtk-files/structure-*.vtu")
vtu_UT = _latest(f"{OUT_U}-vtk-files/thermo-*.vtu")
if vtu_U is None or vtu_UT is None:
    raise SystemExit(why_4c_did_not_finish("run U"))
_mU = meshio.read(vtu_U)
U2d = _nodal(_mU, "displacement", 2, zlayer=0.0)
# PLANE STRAIN IS u_z = 0 ON EVERY NODE, AND 4C'S OWN VTU SAYS WHETHER IT HELD (measured: POINT DIRICH entries with
# ONOFF [1, 1, 0] freed u_z on the interface nodes, and the tractions came out first order)
_uall = np.asarray(_mU.point_data["displacement"], float)
if _uall.ndim == 2 and _uall.shape[1] >= 3:
    _uin = max(float(np.max(np.abs(_uall[:, :2]))), 1e-300)
    _off = np.abs(_uall[:, 2]) > 1e-6 * _uin
    if _off.any():
        _where = sorted({(round(float(p[0]), 4), round(float(p[1]), 4)) for p in np.asarray(_mU.points)[_off]})
        raise SystemExit(f"EXPORT SELF-CHECK: the slab LEFT PLANE STRAIN: |u_z| reaches {float(np.max(np.abs(_uall[:, 2]))):.3e} "
                         f"against an in-plane {_uin:.3e} at {len(_where)} node position(s) (first: {_where[:3]}). A Dirichlet "
                         f"entry released u_z there: 4C applies VOL, SURF, LINE, POINT in that order and an entry's ONOFF 0 RESETS "
                         f"the toggle on its nodes. Pin z in EVERY DIRICH entry: ONOFF [1, 1, 1], VAL [ux, uy, 0] on the interface "
                         f"points, [0, 0, 0] on the outer nodes. Nothing was exported from this run.")

Ttsi = _nodal(meshio.read(vtu_UT), "temperature", 1, zlayer=0.0)[:, 0]
# a uniform temperature of astronomical size is an unconstrained solve: no Dirichlet condition reached the scalar field
_Tmax = float(np.max(np.abs(T2d)))
if _Tmax > 1e6 and float(np.ptp(T2d)) < 1e-6 * _Tmax:
    raise SystemExit(f"the scatra deck's temperature is a uniform {_Tmax:.2e}: an UNCONSTRAINED solve, not a field -- no "
                     f"Dirichlet condition reached the scalar field (DIRICH, never THERMO DIRICH, on E ids the topology "
                     f"defines). Nothing is exported from this run.")
# the two runs solve the same heat problem up to the load rule (O(h^2)); a larger gap is a deck defect
_dT = float(np.max(np.abs(Ttsi - T2d)) / max(float(np.max(np.abs(T2d))), 1e-300))
if _dT > 0.05:
    raise SystemExit(f"the TSI deck's temperature differs from the scatra deck's by {_dT:.3f} relative: "
                     f"the two decks do not solve the same heat problem. Check BOTH decks: deck T's Dirichlet data in "
                     f"DESIGN LINE / POINT DIRICH CONDITIONS (never THERMO DIRICH) and its FUNCT the heat source; deck U's "
                     f"POINT and SURF THERMO DIRICH carrying the same data.")

# REACTIONS -> TRACTION: one yaml per monitored condition (node gid ZERO-based, force f)
_gid_xy = {}
# DECK_U may be the deck's file name or the deck text itself (measured: a worker
# left the assembled text in it); a third way is any TSI deck file next to us.
_deck_u_txt = ""
try:
    if isinstance(DECK_U, str) and len(DECK_U) < 400 and Path(DECK_U).is_file():
        _deck_u_txt = Path(DECK_U).read_text(errors="ignore")
    elif isinstance(DECK_U, str) and "NODE COORDS" in DECK_U:
        _deck_u_txt = DECK_U
except OSError:
    _deck_u_txt = ""
if not _deck_u_txt:
    for _cand in sorted(glob.glob("*.yaml")):
        _t = Path(_cand).read_text(errors="ignore")
        if "Thermo_Structure_Interaction" in _t and "NODE COORDS" in _t:
            _deck_u_txt = _t
            break
for _ln in _deck_u_txt.splitlines():
    _m = re.match(r'\s*-\s*"?NODE\s+(\d+)\s+COORD\s+(\S+)\s+(\S+)\s+(\S+)"?', _ln)
    if _m:
        _gid_xy[int(_m.group(1))] = (round(float(_m.group(2)), 9), round(float(_m.group(3)), 9))
_F = {}
for _yf in glob.glob(f"{OUT_U}-*_monitor_dbc.yaml"):
    _txt = Path(_yf).read_text(errors="ignore")
    _gids = [int(g) for g in re.findall(r"^\s*-\s*(\d+)\s*$", _txt.split("dbc monitor condition data", 1)[0], re.M)]
    _fs = re.findall(r"^\s*f:\s*\n\s*-\s*(\S+)\s*\n\s*-\s*(\S+)", _txt, re.M)
    if not (_gids and _fs):
        continue
    fx, fy = float(_fs[-1][0]), float(_fs[-1][1])          # the last step
    # the yaml's node gids are ZERO-based; the deck's NODE ids are one-based
    _xy = _gid_xy.get(_gids[0] + 1)
    if _xy is None:
        continue
    _acc = _F.setdefault(_xy, [0.0, 0.0])
    _acc[0] += fx
    _acc[1] += fy
if not _F:
    raise SystemExit("run U wrote no <OUT_U>-*_monitor_dbc.yaml: every interface DESIGN POINT DIRICH "
                     "entry needs `TAG: monitor_reaction` and the deck an IO/MONITOR STRUCTURE DBC "
                     "section with FILE_TYPE yaml and WRITE_CONDITION_INFORMATION true")
_ax = 1 if IF in ("left", "right") else 0
_coord = [float(nodes[n - 1][_ax]) for n in interior]
_h = float(np.median(np.diff(sorted(_coord)))) if len(_coord) > 1 else (Y1 - Y0)
_share = _h * float(TZ)
q_t = []
for n in interior:
    _xy = (round(float(nodes[n - 1][0]), 9), round(float(nodes[n - 1][1]), 9))
    if _xy not in _F:
        raise SystemExit(f"no reaction file for the interface node at {_xy}: its POINT DIRICH entries "
                         f"(both layers) need TAG: monitor_reaction")
    q_t.append([_F[_xy][0] / _share, _F[_xy][1] / _share])   # MEASURED sign: this IS -(sigma.n_out)
co = [[float(nodes[n - 1][0]), float(nodes[n - 1][1])] for n in interior]
Q = [[qn, qt[0], qt[1]] for qn, qt in zip(q_n, q_t)]

# ── EXPORT SELF-CHECK ─ keep this block ───────────────────────────────────
_chk = np.asarray(Q, float)
if not np.isfinite(_chk).all():
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface fluxes; nothing was exported")
_chk_imp = imp if imp else {}
_chk_qin = (np.concatenate([np.asarray(_d.get("normal_fluxes") or [], float).ravel()
                            for _d in _chk_imp.values()]) if _chk_imp else np.zeros(0))
if _chk_qin.size == _chk.size and _chk.size and np.array_equal(_chk.ravel(), -_chk_qin):
    raise SystemExit("EXPORT SELF-CHECK: the exported fluxes are the partner's array negated, bit for "
                     "bit: a copy, not a recovery from this side's own runs")
if np.abs(_chk[:, 1:]).max() == 0.0 and np.abs(U2d).max() > 0:
    raise SystemExit("EXPORT SELF-CHECK: a zero traction against a nonzero displacement field: the reaction "
                     "files carried nothing -- check TAG: monitor_reaction on the interface POINT DIRICH entries of BOTH layers")
# a dead heat-flux channel is one dead exchange (measured: qn = 0 everywhere, the coupled T 13 % off)
if np.abs(_chk[:, 0]).max() == 0.0 and float(np.ptp(T2d)) > 0:
    raise SystemExit("EXPORT SELF-CHECK: qn is 0 at EVERY interface node while T varies: the SCATRA FLUX CALC LINE entry's "
                     "E id is not the interface DLINE (or that DLINE lists other nodes); nothing was exported")
# exports.json LAST (its existence is the driver's proof of success); the Dirichlet side owns no trace -> values = []
json.dump({"field_name": "thermoelastic", "coordinates": co, "values": [],
           "normal_fluxes": Q, "n_points": len(co)}, open("exports.json", "w"))
# PER-LEVEL PERSISTENCE: this level's field, named by the config level and never overwritten by the
# next level; interpolate THESE onto the probe points your task names. A dump defect must not cost
# the run: exports.json is already written above.
_LVL = CFG.get("level", "X")
try:
    with open(f"field_level{_LVL}.csv", "w") as _f:
        _f.write("x,y,T,ux,uy\n")
        for (_px, _py), _t, (_ux, _uy) in zip(nodes, T2d, U2d):
            _f.write(f"{_px:.11e},{_py:.11e},{float(_t):.11e},{float(_ux):.11e},{float(_uy):.11e}\n")
    with open(f"interface_level{_LVL}.csv", "w") as _f:
        # the task's OUTWARD traction sigma_tot.n_out is MINUS the exported (qx, qy)
        _f.write("x,y,T,ux,uy,qn,tx,ty\n")
        for (_px, _py), n, (_qn, _qx, _qy) in zip(co, interior, Q):
            _f.write(f"{_px:.11e},{_py:.11e},{float(T2d[n-1]):.11e},{float(U2d[n-1,0]):.11e},"
                     f"{float(U2d[n-1,1]):.11e},{_qn:.11e},{-_qx:.11e},{-_qy:.11e}\n")
except Exception as _dump_exc:
    print(f"[4C thermo-elastic per-level dump] level {_LVL} dump failed: "\
          f"{_dump_exc!r}. exports.json was already written, so the coupling\n"\
          f"continues, but this level has no field file to hand in.")
    for _partial in (f"field_level{_LVL}.csv", f"interface_level{_LVL}.csv"):   # both files or neither
        try:
            Path(_partial).unlink(missing_ok=True)
        except OSError:
            pass
# THE RUN-LOG CONTRACT LINE: `NDOF = <integer>` on a line of its own (T, ux, uy per node), then the descriptive line.
print(f"\nNDOF = {3 * len(nodes)}")
print(f"4C thermo-elastic Dirichlet participant: NDOF = {3 * len(nodes)}  "
      f"T=[{T2d.min():.6g},{T2d.max():.6g}] u=[{U2d.min():.6g},{U2d.max():.6g}]  "
      f"scatra-vs-tsi T mismatch {_dT:.2e}")
