"""openPASO-side 4C deck lint: names the defects of a deck the AGENT wrote, from the deck text and 4C's
own console log. A verification gate, not a generator: it never writes or completes a deck.

The same defect classes ride inside the served coupling contracts as text the participant runs at
exit (why_4c_did_not_finish); this copy lets the ladder read a side directory and put 4C's own
error line and the named defects into the next step's brief, so a worker that must repair a deck
starts from the defect and not from the whole job. Every class here was measured on a worker deck.
"""
from __future__ import annotations

import difflib
import json
import math
import os
from collections import Counter
import re
import subprocess
from pathlib import Path

_TOPO_WORDS = ("DNODE", "DLINE", "DSURFACE", "DVOL")
_E_IS_A_DESIGN_ID = (' -- E is the DESIGN-ENTITY id of the topology section (the number after DNODE/DLINE/DSURFACE/DVOL), never a node number: a node joins entity E through a topology line `NODE <n> DNODE <E>`, so a condition on E 9 needs a `"NODE 9 DNODE 9"`-style entry, not a node 9')
_TOPOLOGY_PAIRS = ('the pairs are section `DNODE-NODE TOPOLOGY` with entries `NODE <n> DNODE <id>`, `DLINE-NODE TOPOLOGY` with `DLINE`, `DSURF-NODE TOPOLOGY` with entries `NODE <n> DSURFACE <id>` (section word DSURF, entry word DSURFACE), `DVOL-NODE TOPOLOGY` with `DVOL`')
_VALID_CACHE: dict[str, set] = {}


def grammar(bin_path, ld: str | None = None) -> dict:
    """The INSTALLED binary's own grammar (`4C -p`, read once per binary): the section names it accepts
    (`sections:` plus the legacy string sections -- elements, coordinates, topology) and its element
    TYPE names (`legacy_element_specs:`). Empty when the binary is not there (then nothing is judged)."""
    key = str(bin_path or "")
    if key in _VALID_CACHE:
        return _VALID_CACHE[key]
    g = {"sections": set(), "elements": set(), "materials": {}}
    try:
        if key and Path(key).is_file():
            env = dict(os.environ)
            if ld:
                env["LD_LIBRARY_PATH"] = f"{ld}:{env.get('LD_LIBRARY_PATH', '')}"
            dump = subprocess.run([key, "-p"], capture_output=True, text=True, timeout=180, env=env, stdin=subprocess.DEVNULL).stdout
            g["sections"] = set(re.findall(r"^    - name: (.+?)\s*$", dump, re.M)) | set(
                re.findall(r"^  - ([A-Z][A-Z0-9 _/.:-]*?)\s*$", dump.split("legacy_string_sections:", 1)[-1], re.M)) | {"TITLE"}
            g["elements"] = set(re.findall(r"^  ([A-Z][A-Z0-9_]*):\s*$",
                                           dump.split("legacy_element_specs:", 1)[-1].split("legacy_particle_specs:", 1)[0], re.M))
            g["materials"] = _material_specs(dump)
    except Exception:                                   # noqa: BLE001
        g = {"sections": set(), "elements": set(), "materials": {}}
    _VALID_CACHE[key] = g
    return g


def _material_specs(dump: str) -> dict:
    """{material name: {parameter: required?}} from the grammar dump: every `- name: MAT_*` group and the
    parameter names nested under it (measured need: a worker's MAT_Struct_ThermoStVenantK without
    YOUNGNUM stopped 4C with "Parameter 'YOUNGNUM' not found in container")."""
    specs: dict = {}
    lines = dump.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^(\s*)- name: (MAT_[A-Za-z0-9_]+)\s*$", lines[i])
        if m and i + 2 < len(lines) and "type: group" in lines[i + 1] + lines[i + 2]:
            ind, name, params, cur, depth0 = len(m.group(1)), m.group(2), {}, None, None
            j = i + 1
            while j < len(lines):
                ln = lines[j]
                if ln.strip() and len(ln) - len(ln.lstrip()) <= ind:
                    break
                pm = re.match(r"^(\s*)- name: ([A-Za-z0-9_]+)\s*$", ln)
                if pm and len(pm.group(1)) > ind:
                    # the material's own parameters sit at ONE depth; deeper names are the alternatives of a
                    # nested one_of (MAT_Fourier's CONDUCT: constant | from_file | ...), not parameters
                    if depth0 is None:
                        depth0 = len(pm.group(1))
                    if len(pm.group(1)) == depth0:
                        cur = pm.group(2)
                        params[cur] = {"required": True, "type": ""}
                    else:
                        cur = None
                elif cur and depth0 is not None and len(ln) - len(ln.lstrip()) == depth0 + 2:
                    if re.match(r"^\s*required: (true|false)\s*$", ln):
                        params[cur]["required"] = ln.strip().endswith("true")
                    elif re.match(r"^\s*type: \S+\s*$", ln) and not params[cur]["type"]:
                        params[cur]["type"] = ln.split(":", 1)[1].strip()
                j += 1
            specs.setdefault(name, {}).update(params)
            i = j
        else:
            i += 1
    return specs


def material_defects(text: str, mats: dict) -> list[str]:
    """The deck's MATERIALS entries against the binary's material grammar: an unknown material name (with
    the closest known), a missing required parameter, a parameter the material does not have."""
    if not mats:
        return []
    blk = re.search(r"^MATERIALS:\s*$(.*?)(?=^[A-Z][A-Z0-9 _/.:-]*?:\s*$|\Z)", text, re.M | re.S)
    if not blk:
        return []
    body = blk.group(1)
    out = []
    for m in re.finditer(r"^(\s+)(MAT_[A-Za-z0-9_]+):\s*$", body, re.M):
        ind, name = len(m.group(1)), m.group(2)
        keys = []
        for ln in body[m.end():].splitlines()[1:]:
            if not ln.strip():
                continue
            if len(ln) - len(ln.lstrip()) <= ind:
                break
            km = re.match(r"^\s+([A-Za-z0-9_]+):", ln)
            if km and len(ln) - len(ln.lstrip()) == ind + 2:
                keys.append(km.group(1))
        if name not in mats:
            close = difflib.get_close_matches(name, sorted(mats), n=3, cutoff=0.6)
            out.append(f"material '{name}' is not in the binary's grammar (`4C -p`)"
                       + (f"; closest known: {', '.join(close)}" if close else ""))
            continue
        spec = mats[name]
        missing = [p for p, s in spec.items() if s["required"] and p not in keys]
        unknown = [k for k in keys if k not in spec]
        if missing:
            out.append(f"{name} is missing required parameter(s) {', '.join(missing)}; its parameters are "
                       f"{', '.join(spec)} (4C stops with \"Parameter '{missing[0]}' not found in container\")")
        if unknown:
            out.append(f"{name} has parameter(s) its grammar does not know: {', '.join(unknown)}; known: {', '.join(spec)}")
        # a vector-typed parameter written as a bare number (measured: YOUNG: 787.5 for a `type: vector`
        # parameter -- 4C says only 'Could not match this input' and prints the block)
        for k, val in re.findall(r"^\s+([A-Za-z0-9_]+):\s*(\S.*)$", body[m.end():].split("\n  - MAT:", 1)[0], re.M):
            if k in spec and spec[k]["type"] == "vector" and not val.strip().startswith("["):
                out.append(f"{name}.{k} is a vector in the grammar (`type: vector`) and must be written as a list, "
                           f"[{val.strip()}], not the bare value {val.strip()}")
    return out


def valid_sections(bin_path, ld: str | None = None) -> set:
    """Section names the installed binary accepts (see `grammar`)."""
    return grammar(bin_path, ld)["sections"]


def unknown_sections(text: str, valid: set, elements: set | None = None) -> list[str]:
    """Deck sections the binary's grammar does not know, each with the closest names it does know.
    Measured: the dominant worker failure is a section name invented by analogy
    ('IO/RUNTIME VTK OUTPUT/THERMO', 'SOLIDSCATRA ELEMENTS'), and 4C stops at the first one."""
    if len(valid) < 100:
        return []
    out = []
    names = sorted(valid)
    elem_sections = sorted(n for n in names if n.endswith(" ELEMENTS"))
    for s in dict.fromkeys(re.findall(r"^([A-Z][A-Z0-9 _/.:-]*?):\s*$", text, re.M)):
        if s in valid or re.fullmatch(r"FUNCT\d+", s):
            continue
        msg = f"section '{s}' is not in the binary's grammar (`4C -p`)"
        words = s.split()
        if elements and s.endswith(" ELEMENTS") and words[0] in elements:
            # 'SOLIDSCATRA ELEMENTS': the first word is an element TYPE, not a section (measured)
            msg += (f"; '{words[0]}' is an ELEMENT TYPE that goes on the element lines inside one of the element "
                    f"sections the binary knows: {', '.join(elem_sections)}")
        else:
            close = closest_sections(s, names)
            if close:
                msg += f"; closest known: {', '.join(repr(c) for c in close)}"
        out.append(msg)
    # a PARAMETER written at the top level: `CALCFLUX_BOUNDARY: "diffusive"` at column 0 is a section to 4C
    # ('Section CALCFLUX_BOUNDARY is not a valid section name', measured te4c13); a key WITH a value on its
    # line is not a section head, so the loop above never saw it
    for key, val in dict.fromkeys(re.findall(r"^([A-Z][A-Z0-9_]*):[ \t]+(\S.*)$", text, re.M)):
        if key in valid or re.fullmatch(r"FUNCT\d+", key):
            continue
        close = closest_sections(key, names)
        out.append(f"top-level key `{key}: {val[:30]}` is not a section name (`4C -p`); a parameter belongs INSIDE its "
                   f"section, indented under the section head (a key at column 0 is read as a section)"
                   + (f"; the sections whose names come closest: {', '.join(repr(c) for c in close[:3])}" if close else ""))
    return out


def _tokens(name: str) -> list[str]:
    return [w for w in re.split(r"[ /_-]+", name.upper()) if w]


def _acronym_cover(u: list, ct: list) -> tuple:
    """Candidate words that are the initials of a run of the unknown's words (TSI <- THERMO STRUCTURE
    INTERACTION), and the unknown-word indices such a run covers."""
    acr, cov = set(), set()
    for y in ct:
        if 3 <= len(y) <= 5 and y.isalpha():   # two letters match by accident (LS <- LINE STRUCTURE)
            for k in range(len(u) - len(y) + 1):
                if "".join(w[0] for w in u[k:k + len(y)]) == y:
                    acr.add(y)
                    cov |= set(range(k, k + len(y)))
    return acr, cov


def closest_sections(unknown: str, names: list, n: int = 5) -> list[str]:
    """Known names ranked by word similarity both ways, each word weighted by how rare it is across
    the grammar (common words like RUNTIME/VTK/OUTPUT count less than the word that distinguishes the
    name), so 'IO/RUNTIME VTK OUTPUT/THERMO' lists THERMAL DYNAMIC/RUNTIME VTK OUTPUT among its five
    (character ratios alone ranked it sixth, measured on the installed grammar). A known word that is
    the acronym of a run of the unknown's words counts as an exact match of that run, so 'THERMO
    STRUCTURE INTERACTION DYNAMIC' finds TSI DYNAMIC (measured on a worker deck)."""
    u = _tokens(unknown)
    if not u:
        return []
    toks = {c: _tokens(c) for c in names}
    freq = Counter(w for ct in toks.values() for w in set(ct))
    def wt(w):
        return 1.0 / (1.0 + math.log(freq.get(w, 0) + 1))
    def best(a, bs):
        return max((difflib.SequenceMatcher(None, a, b).ratio() for b in bs), default=0.0)
    scored = []
    for c, ct in toks.items():
        if not ct:
            continue
        acr, cov = _acronym_cover(u, ct)
        num = (sum(wt(a) * (1.0 if i in cov else best(a, ct)) for i, a in enumerate(u))
               + sum(wt(b) * (1.0 if b in acr else best(b, u)) for b in ct))
        den = sum(wt(a) for a in u) + sum(wt(b) for b in ct)
        scored.append((num / den, c))
    scored.sort(key=lambda p: (-p[0], p[1]))
    return [c for sc, c in scored[:n] if sc >= 0.45]


def fourc_error_lines(log_text: str, n: int = 8) -> str:
    """4C's own error block after 'PROC 0 ERROR' (the first n non-empty lines), or -- when the binary
    died on a signal instead of an error message -- the signal and the last thing 4C printed before
    it (measured: a flux-output table dividing by a zero boundary area ends in 'Floating point
    exception' with no error line at all), or ''."""
    lines = log_text.splitlines()
    for i, ln in enumerate(lines):
        if "PROC 0 ERROR" in ln:
            said = []
            for l in lines[i + 1:i + 16]:
                if not l.strip():
                    continue            # a blank line separates the message from the offending snippet
                if re.match(r"\s*\d+#\s", l) or set(l.strip()) <= set("=-"):
                    break               # the stack frames and rulers after the message are not the message
                if "MPI_ABORT" not in l:
                    said.append(l.strip())
            return " | ".join(said[:n])
    for i, ln in enumerate(lines):
        if ln.startswith("ERROR:"):
            # the YAML reader's own stop (no PROC 0 block): the message, then the offending line with its
            # position marker, up to the first stack frame (measured: 'ERROR: could not find ':' colon after
            # key' / '204:40: 2 TRANSP QUAD4 2 3 12 11 MAT 1 TYPE Std' -- an unquoted table row)
            said = [ln.strip()]
            for l in lines[i + 1:i + 6]:
                if re.match(r"\s*\d+#\s", l) or set(l.strip()) <= set("=-"):
                    break
                if l.strip():
                    said.append(l.strip())
            return " | ".join(said[:n])
    for i, ln in enumerate(lines):
        if "*** Process received signal ***" in ln:
            sig = next((l.split("Signal:", 1)[1].strip() for l in lines[i:i + 4] if "Signal:" in l), "signal")
            code = next((l.split("Signal code:", 1)[1].strip() for l in lines[i:i + 5] if "Signal code:" in l), "")
            before = [l.strip() for l in lines[max(0, i - 12):i] if l.strip() and not set(l.strip()) <= set("+-|=")]
            return (f"4C died on {sig}" + (f" ({code})" if code else "") + " with no error message; the last "
                    f"thing it printed: " + " | ".join(before[-2:]))
    return ""


def python_stop_lines(log_text: str) -> str:
    """The last lines of a Python traceback in a captured console (the participant's own stop), or ''."""
    i = log_text.rfind("Traceback (most recent call last)")
    if i < 0:
        return ""
    tail = [l.rstrip() for l in log_text[i:].splitlines()[1:] if l.strip()]
    return " | ".join(tail[-3:])


def lint_deck(text: str, cfg: dict | None = None) -> list[str]:
    """Deck defects measured on worker decks (each one made 4C stop or solve the wrong problem). `cfg` is
    the config.json beside the deck when the caller has it: its `iface` names a box side's interface."""
    why: list[str] = []
    why += _yaml_parse_error(text)
    secs = re.findall(r"^([A-Z][A-Z0-9 _/.:-]*?):\s*$", text, re.M)
    dup = sorted({x for x in secs if secs.count(x) > 1})
    if dup:
        why.append("section(s) written twice: " + ", ".join(dup))
    if "Thermo_Structure_Interaction" in text:
        if "CLONING MATERIAL MAP" not in text:
            why.append("TSI needs a CLONING MATERIAL MAP pairing the structure material with the MAT_Fourier thermal material")
        if "COUPVARIABLE" not in text or "Temperature" not in text.split("COUPVARIABLE", 1)[-1][:40]:
            why.append('TSI DYNAMIC/PARTITIONED needs COUPVARIABLE "Temperature" (the default gives zero thermal strain)')
        if re.search(r"\b(WALL|SOLID) QUAD4\b|\bTRI3\b", text):
            why.append("4C has no 2-D thermo-elastic element; the route is a one-element-thick SOLIDSCATRA HEX8 slab")
        if "monitor_reaction" in text and "IO/MONITOR STRUCTURE DBC" not in text:
            why.append("TAG: monitor_reaction writes nothing without an IO/MONITOR STRUCTURE DBC section")
        if "DESIGN VOL THERMO DIRICH" in text:
            why.append("DESIGN VOL THERMO DIRICH imposes the temperature volume-wide (no heat equation is solved); "
                       "use SURF (outer) and POINT (interface) THERMO DIRICH")
        # a temperature condition filed under the STRUCTURAL Dirichlet family: 4C stops with
        # '1 DOFs given but 3 expected in Point Dirichlet boundary condition' (measured)
        for b in re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\s*$)", text, flags=re.M):
            head = b.split(":", 1)[0].strip()
            if re.fullmatch(r"DESIGN (POINT|LINE|SURF|VOL) DIRICH CONDITIONS", head) and re.search(r"\bNUMDOF:\s*1\b", b):
                kind = head.split()[1]
                why.append(f"{head} has an entry with NUMDOF 1: that family carries the 3 displacement dofs of the slab "
                           f"(NUMDOF 3, ONOFF/VAL/FUNCT with three entries); a temperature value belongs in "
                           f"DESIGN {kind} THERMO DIRICH CONDITIONS (NUMDOF 1)")
    if "Scalar_Transport" in text:
        if "THERMAL DYNAMIC:" in text and "SCALAR TRANSPORT DYNAMIC:" not in text:
            why.append("Scalar_Transport needs `SCALAR TRANSPORT DYNAMIC`, not `THERMAL DYNAMIC`")
        # Dirichlet data filed under the THERMO family in a scalar-transport deck: the scalar field
        # never reads THERMO DIRICH (it belongs to the Thermo/TSI field), so the deck has no Dirichlet
        # condition at all, the stationary solve is singular and 4C returns a uniform field of order
        # 1e13 while reporting 'finished normally'. Measured on a served thermo-elastic participant's
        # deck: phi_1 = 7.5e12 at every node; the same deck with the two families renamed to DIRICH
        # gave the expected field. Two runs of one round lost their thermal side to exactly this.
        if (re.search(r"^DESIGN (POINT|LINE|SURF|VOL) THERMO DIRICH CONDITIONS:\s*$", text, re.M)
                and not re.search(r"^DESIGN (POINT|LINE|SURF|VOL) DIRICH CONDITIONS:\s*$", text, re.M)):
            why.append("every Dirichlet datum of this Scalar_Transport deck sits in a THERMO DIRICH family, which "
                       "the scalar field never reads (it belongs to the Thermo/TSI field): the deck has NO Dirichlet "
                       "condition, the stationary solve is singular, and 4C returns a uniform field of order 1e13 "
                       "while reporting 'finished normally'. Put the outer datum and the interface values in "
                       "DESIGN LINE / POINT DIRICH CONDITIONS (NUMDOF 1)")
        if "CALCFLUX_BOUNDARY" not in text or "FLUX CALC" not in text:
            why.append('a consistent boundary flux needs CALCFLUX_BOUNDARY "diffusive" AND a `SCATRA FLUX CALC LINE CONDITIONS` entry on the interface line')
        if re.search(r"^IO:\s*$", text, re.M):
            why.append("an `IO:` section in a Scalar_Transport deck is rejected; the VTU appears without it")
    if re.search(r'PROBLEMTYPE:\s*"?Thermo"?\s*$', text, re.M):
        why.append("PROBLEMTYPE Thermo writes no scatra flux output and knows no CALCFLUX_BOUNDARY; a consistent heat flux comes from Scalar_Transport")
    badkw = sorted({w for w in re.findall(r'"NODE\s+\d+\s+(D[A-Z]+)\s+\d+"', text) if w not in _TOPO_WORDS})
    if badkw:
        why.append(f"topology entries use {', '.join(badkw)} -- the entity words are DNODE, DLINE, DSURFACE, DVOL "
                   "(anything else defines nothing and 4C silently drops the conditions on it); " + _TOPOLOGY_PAIRS)
    topo = set(re.findall(r"\b(DNODE|DLINE|DSURFACE|DVOL)\s+(\d+)", text))
    for b in re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\s*$)", text, flags=re.M):
        head = b.split(":", 1)[0].strip()
        if head.endswith("CONDITIONS"):
            # every condition family with a geometry word -- DESIGN ... and SCATRA FLUX CALC LINE CONDITIONS alike
            # (measured: a flux-calc line on a DLINE with no topology section escaped a DESIGN-only check and
            # 4C stopped with 'DLine 1 not in range [0:0[')
            kw = re.search(r"\b(POINT|LINE|SURF|VOL)\b", head)
            if not kw:
                continue
            kind = {"POINT": "DNODE", "LINE": "DLINE", "SURF": "DSURFACE", "VOL": "DVOL"}[kw.group(1)]
            entries = [e for e in re.split(r"^\s*-\s", b, flags=re.M)[1:] if e.strip()]
            noid = [e for e in entries if not re.search(r"\bE:\s*\d+|NODE_SET_NAME", e)]
            if noid:
                why.append(f"{len(noid)} entr{'y' if len(noid) == 1 else 'ies'} in {head} without `E: <id>`")
            missing = sorted({x for x in re.findall(r"\bE:\s*(\d+)", b) if (kind, x) not in topo}, key=int)
            if missing:
                why.append(f"{head} names E id(s) {', '.join(missing[:6])} that no *-NODE TOPOLOGY section defines" + _E_IS_A_DESIGN_ID)
    if re.search(r"FUNCT\d+:", text) and re.search(r"\bFUNCT:\s*\[\s*0(\s*,\s*0)*\s*\]", text) \
            and not re.search(r"\bFUNCT:\s*\[[^\]]*[1-9]", text):
        why.append("FUNCT blocks are defined but no condition references one (FUNCT: [0,...] everywhere): the sources never reach the load")
    why += _degenerate_elements(text)
    why += _lines_without_an_element_edge(text)
    why += _unquoted_table_rows(text)
    why += _elements_with_unknown_nodes(text)
    why += _dirichlet_pins_every_node(text)
    why += _dirichlet_releases_a_pinned_dof(text)
    why += _two_conditions_on_one_entity(text)
    why += _coupvariable_in_the_wrong_section(text)
    why += _flux_calc_off_the_interface(text)
    why += _box_side_facts(text, cfg)
    why += _bad_hex8_slabs(text)
    why += _interior_lines(text)
    why += _double_star_in_functions(text)
    why += _missing_runtime_output(text)
    return why


def _missing_runtime_output(text: str) -> list[str]:
    """A TSI deck that runs to 'finished normally' and writes no VTU is useless to a recovery that reads VTU
    (measured on a ladder-loop deck: TSI run U finished, no structure-*.vtu, no thermo-*.vtu)."""
    out = []
    if "Thermo_Structure_Interaction" in text:
        if "IO/RUNTIME VTK OUTPUT/STRUCTURE" not in text or not re.search(r"DISPLACEMENT:\s*true", text, re.I):
            out.append("a TSI deck writes no structure VTU without `IO/RUNTIME VTK OUTPUT` (INTERVAL_STEPS 1) and "
                       "`IO/RUNTIME VTK OUTPUT/STRUCTURE` with OUTPUT_STRUCTURE true and DISPLACEMENT true; the run "
                       "finishes 'normally' with nothing to read")
        if "THERMAL DYNAMIC/RUNTIME VTK OUTPUT" not in text or not re.search(r"TEMPERATURE:\s*true", text, re.I):
            out.append("a TSI deck writes no thermo VTU without `THERMAL DYNAMIC/RUNTIME VTK OUTPUT` with OUTPUT_THERMO "
                       "true and TEMPERATURE true")
    # Scalar_Transport is NOT judged here: measured 2026-09-12 (deck step trial, this binary), a scatra deck with
    # no IO section at all wrote scatra-00000/00001 VTU files and the .pvd by default -- the earlier finding
    # named a defect on a deck that ran and was right, and would have cost a repair round for nothing.
    return out


def _degenerate_elements(text: str) -> list[str]:
    """2-D elements with zero or negative signed area from the deck's own NODE COORDS: a twisted or
    clockwise node order. 4C reports it only as 'The determinant of the matrix is equal zero or
    negative!' or a floating point exception in the element evaluation (measured on a worker deck:
    70 of 80 quads written as (i, i+1, i+NX, i+NX+1) instead of counter-clockwise)."""
    xy = {}
    for n, x, y in re.findall(r'"NODE\s+(\d+)\s+COORD\s+(\S+)\s+(\S+)\s+\S+"', text):
        try:
            xy[int(n)] = (float(x), float(y))
        except ValueError:
            continue
    if not xy:
        return []
    els = re.findall(r'"\s*(\d+)\s+\w+\s+(QUAD4|TRI3)\s+((?:\d+\s+)+)', text)
    bad, total, first = 0, 0, None
    for e, kind, ids in els:
        nodes = [int(i) for i in ids.split()][:4 if kind == "QUAD4" else 3]
        if len(nodes) < 3 or any(i not in xy for i in nodes):
            continue
        total += 1
        p = [xy[i] for i in nodes]
        area = 0.5 * sum(p[k][0] * p[(k + 1) % len(p)][1] - p[(k + 1) % len(p)][0] * p[k][1] for k in range(len(p)))
        if area <= 1e-14 * max(1.0, max(abs(c) for q in p for c in q) ** 2):
            bad += 1
            first = first or (e, kind, nodes, area)
    if not bad:
        return []
    e, kind, nodes, area = first
    return [f"{bad} of {total} 2-D elements have zero or negative area from the deck's own NODE COORDS (first: element "
            f"{e} {kind} nodes {' '.join(map(str, nodes))}, signed area {area:.3g}): the node order of every element must "
            "run counter-clockwise -- for a structured grid with NX cells per row and node id = i + 1 + (NX + 1) * j the "
            "quad of cell (i, j) is (id, id + 1, id + NX + 2, id + NX + 1). 4C reports this only as a zero or negative "
            "determinant or a floating point exception in the element evaluation"]


def _double_star_in_functions(text: str) -> list[str]:
    """`**` inside a 4C function expression: the parser has no such operator (measured: `-12*x**3*y/5`
    rejected from 4C_utils_symbolic_expression.cpp with 'Token expected'); the task texts write their
    source terms in Python notation, so a copied expression carries it. Every occurrence, not the first."""
    out = []
    for m in re.finditer(r'(SYMBOLIC_FUNCTION_OF_SPACE_TIME|SYMBOLIC_FUNCTION_OF_TIME|VARFUNCTION|COMPONENT\s+\d+\s+SYMBOLIC_FUNCTION_OF_SPACE_TIME)[^\n]*?["\']([^"\'\n]*\*\*[^"\'\n]*)["\']', text):
        expr = m.group(2)
        out.append(f"function expression '{expr[:70]}' uses `**`: 4C's expression parser has no `**` (it stops with "
                   f"'Token expected'); write `^` for every power in the expression, not only the first")
    return out


def _dirichlet_pins_every_node(text: str) -> list[str]:
    """Dirichlet conditions that reach EVERY node of a field leave nothing to solve: 4C's predictor prints
    'res-norm 0', the solver converges at iteration 0 and the field is the prescribed data (zero where the
    data is zero). Measured on a recorded run: a TSI deck whose 'outer' surface held 90 of 108 slab nodes and
    whose interface points held the other 18 -- both fields pinned everywhere, both volume loads gone, 85
    shell calls spent on a '4C limitation'. Counted per field and per in-plane component (a u_z = 0 pin on
    the whole slab is plane strain, not a defect), only when the coverage is complete."""
    nodes = {int(a) for a in re.findall(r'"NODE\s+(\d+)\s+COORD\b', text)}
    if len(nodes) < 4:
        return []
    topo: dict = {}
    for n, kind, e in re.findall(r'"NODE\s+(\d+)\s+(DNODE|DLINE|DSURFACE|DVOL)\s+(\d+)"', text):
        topo.setdefault((kind, int(e)), set()).add(int(n))
    kmap = {"POINT": "DNODE", "LINE": "DLINE", "SURF": "DSURFACE", "VOL": "DVOL"}
    covered: dict = {}
    via: dict = {}
    for b in re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\s*$)", text, flags=re.M):
        head = b.split(":", 1)[0].strip()
        if "DIRICH" not in head or not head.endswith("CONDITIONS"):
            continue
        kw = re.search(r"\b(POINT|LINE|SURF|VOL)\b", head)
        if not kw:
            continue
        fam = "temperature" if ("THERMO" in head or "TRANSPORT" in head or "Scalar_Transport" in text) else "displacement"
        for entry in re.split(r"^\s*-\s", b, flags=re.M)[1:]:
            e = re.search(r"\bE:\s*(\d+)", entry)
            on = re.search(r"ONOFF:\s*\[([^\]]*)\]", entry)
            flags = [x.strip() for x in on.group(1).split(",")] if on else ["1"]
            inplane = any(f == "1" for f in flags[:2]) if fam == "displacement" else flags[0] == "1"
            if not e or not inplane:
                continue
            ns = topo.get((kmap[kw.group(1)], int(e.group(1))), set())
            if ns:
                covered.setdefault(fam, set()).update(ns)
                via.setdefault(fam, []).append(f"{kmap[kw.group(1)]} {e.group(1)} ({len(ns)} nodes)")
    out = []
    for fam, s_ in covered.items():
        if s_ >= nodes:
            # aggregate the entities by kind (a slab has one DNODE per interface node and layer)
            by_kind: dict = {}
            for v in dict.fromkeys(via[fam]):
                kind, rest = v.split(" ", 1)
                cnt = int(re.search(r"\((\d+) nodes\)", rest).group(1))
                k = by_kind.setdefault(kind, [0, 0]); k[0] += 1; k[1] += cnt
            through = ", ".join(f"{n} {kind} entit{'y' if n == 1 else 'ies'} ({c} nodes)" for kind, (n, c) in by_kind.items())
            out.append(f"DIRICHLET PINS EVERY NODE of the {fam} field ({len(nodes)} of {len(nodes)} nodes, through "
                       f"{through}): nothing is left to solve -- 4C's predictor prints "
                       f"'res-norm 0', the solver converges at iteration 0 and the field is the prescribed data. The outer "
                       f"boundary holds the BOUNDARY nodes only (x = x0, y = y0, y = y1, on both layers of a slab); interior "
                       f"nodes carry no Dirichlet condition, and the interface points are the interface nodes alone")
    return out


_DBC_RANK = {"DVOL": 0, "DSURFACE": 1, "DLINE": 2, "DNODE": 3}


def _dirichlet_releases_a_pinned_dof(text: str) -> list[str]:
    """A lower-dimensional Dirichlet entry RESETS the toggles on its nodes. 4C applies the families in the order
    VOL, SURF, LINE, POINT and, for every dof an entry lists with ONOFF 0, sets the toggle to zero and drops the
    dof from the Dirichlet set -- whatever an earlier family had pinned there (4C_fem_discretization_utils_dbc.cpp,
    do_dirichlet_condition: "the dof at geometry of lower hierarchical order can reset the toggle value").
    Measured on a TSI slab: the volume pinned u_z (ONOFF [0, 0, 1]), the interface points carried the partner's
    (ux, uy) with ONOFF [1, 1, 0], and u_z came free on every interface node of both layers -- the slab left plane
    strain there (|u_z| of the same order as the in-plane displacement in 4C's own VTU; zero with ONOFF [1, 1, 1]), the
    reactions read as tractions were first order and the displacement of both coupled sides with them."""
    topo: dict = {}
    for n, kind, e in re.findall(r'"NODE\s+(\d+)\s+(DNODE|DLINE|DSURFACE|DVOL)\s+(\d+)"', text):
        topo.setdefault((kind, int(e)), set()).add(int(n))
    if not topo:
        return []
    kmap = {"POINT": "DNODE", "LINE": "DLINE", "SURF": "DSURFACE", "VOL": "DVOL"}
    entries: list = []                       # (family, rank, geometry word, E id, flags, nodes)
    for b in re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\s*$)", text, flags=re.M):
        head = b.split(":", 1)[0].strip()
        if "DIRICH" not in head or not head.endswith("CONDITIONS"):
            continue
        kw = re.search(r"\b(POINT|LINE|SURF|VOL)\b", head)
        if not kw:
            continue
        fam = re.sub(r"\b(POINT|LINE|SURF|VOL)\s+", "", head)
        for entry in re.split(r"^\s*-\s", b, flags=re.M)[1:]:
            e = re.search(r"\bE:\s*(\d+)", entry)
            on = re.search(r"ONOFF:\s*\[([^\]]*)\]", entry)
            if not e or not on:
                continue
            ns = topo.get((kmap[kw.group(1)], int(e.group(1))), set())
            if ns:
                entries.append((fam, _DBC_RANK[kmap[kw.group(1)]], kw.group(1), int(e.group(1)),
                                [x.strip() for x in on.group(1).split(",")], ns))
    freed: dict = {}
    for fam, rank, geo, eid, flags, ns in entries:
        for j, fl in enumerate(flags):
            if fl != "0":
                continue
            by = [(g2, e2, n2 & ns) for f2, r2, g2, e2, fl2, n2 in entries
                  if f2 == fam and r2 < rank and j < len(fl2) and fl2[j] == "1" and (n2 & ns)]
            if not by:
                continue
            over = ", ".join(dict.fromkeys(f"{g2} E {e2}" for g2, e2, _n in by))
            acc = freed.setdefault((fam, geo, j + 1, over), [0, set(), []])
            acc[0] += 1
            for _g, _e, n in by:
                acc[1].update(n)
            acc[2].append(str(eid))
    out = []
    for (fam, geo, dof, over), (cnt, nodes, ids) in sorted(freed.items()):
        out.append(f"DIRICHLET TOGGLE RELEASED: {cnt} {geo} entr{'y' if cnt == 1 else 'ies'} of {fam} "
                   f"(E {', '.join(ids[:5])}{', ...' if cnt > 5 else ''}) list dof {dof} as ONOFF 0 on {len(nodes)} node(s) "
                   f"that {over} pins. 4C applies VOL, SURF, LINE, POINT in that order and an entry's 0 RESETS the toggle "
                   f"on its nodes, so dof {dof} is FREE there -- on a plane-strain slab that is u_z released where the entry "
                   f"sits, the slab leaves plane strain, and the reactions, tractions and displacement come out first "
                   f"order. Repeat the pin in the lower-dimensional entry: ONOFF 1 for that dof with VAL 0 "
                   f"(interface points: ONOFF [1, 1, 1], VAL [ux, uy, 0])")
    return out


def _two_conditions_on_one_entity(text: str) -> list[str]:
    """Two entries of one Dirichlet family that name the same design entity with different data.

    MEASURED: a TSI deck numbered its outer point entities 1.. and its interface point entities
    from 100, and at the finest level the outer ids reached 146 -- 45 of the 62 interface
    displacement entries also pinned outer nodes, or the outer zero landed on interface nodes.
    4C applies the entries of a family in order and the last one wins on the shared nodes; the
    coupling wandered for 149 iterations at that level alone while the two coarser levels, whose
    ids did not collide, converged in 14. The deck says which it is: one E id, two different
    VAL/ONOFF lines, in one family."""
    out = []
    for b in re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\s*$)", text, flags=re.M):
        head = b.split(":", 1)[0].strip()
        if not head.endswith("CONDITIONS") or not re.search(r"\b(POINT|LINE|SURF|VOL)\b", head):
            continue
        seen: dict = {}
        for entry in re.split(r"^\s*-\s", b, flags=re.M)[1:]:
            e = re.search(r"\bE:\s*(\d+)", entry)
            if not e:
                continue
            on = re.search(r"ONOFF:\s*\[([^\]]*)\]", entry)
            val = re.search(r"VAL:\s*\[([^\]]*)\]", entry)
            data = (on.group(1).replace(" ", "") if on else "", val.group(1).replace(" ", "") if val else "")
            prev = seen.setdefault(int(e.group(1)), data)
            if prev != data:
                out.append(f"{head} names E {e.group(1)} twice with different data (ONOFF/VAL {prev} and {data}): "
                           f"4C applies a family's entries in order and the LAST wins on the shared nodes, so one "
                           f"of the two conditions is silently lost there. Measured: outer and interface point ids "
                           f"that overlapped at the finest level only made that level's coupling wander for 149 "
                           f"iterations. Give every design entity ONE id range (outer ids never reaching the "
                           f"interface's) and check the finest level's counts, not the coarsest's")
                break
    return out


def _coupvariable_in_the_wrong_section(text: str) -> list[str]:
    """COUPVARIABLE outside TSI DYNAMIC/PARTITIONED is ignored by 4C, and the thermal strain is zero.

    MEASURED: a run put `COUPVARIABLE: "Temperature"` under THERMAL DYNAMIC and lost its first
    twenty minutes to a deck the grammar refused; the key and its section were served
    correctly. check_input names it from the binary's grammar; this names it at write time."""
    out = []
    section = ""
    for line in text.splitlines():
        m = re.match(r"^([A-Z][A-Z0-9 _/.:-]*?):\s*$", line)
        if m:
            section = m.group(1).strip()
            continue
        if re.match(r"^\s+COUPVARIABLE\s*:", line) and section and section != "TSI DYNAMIC/PARTITIONED":
            out.append(f"COUPVARIABLE sits under `{section}`; 4C reads it only in `TSI DYNAMIC/PARTITIONED` "
                       f"(COUPVARIABLE: \"Temperature\"), anywhere else the grammar refuses the deck or the "
                       f"thermal strain stays zero. Move the key")
            break
    return out


def _flux_calc_off_the_interface(text: str) -> list[str]:
    """A SCATRA FLUX CALC line that shares no node with the interface's point conditions.

    MEASURED: a Dirichlet-side scalar deck carried the partner's temperatures as
    DESIGN POINT DIRICH entries on the interface nodes and named DLINE 2 -- the
    bottom edge -- in its SCATRA FLUX CALC LINE CONDITIONS; no interface line was
    defined at all. 4C computed the boundary flux on the bottom edge, the
    exported heat flux was 0 at every interface node at every level, and the
    coupled temperature converged cleanly to a function 13 % off. The deck says
    it: the flux line and the nodes that carry the imported values are disjoint."""
    if "Scalar_Transport" not in text or "SCATRA FLUX CALC" not in text:
        return []
    topo: dict = {}
    for n, kind, e in re.findall(r'"NODE\s+(\d+)\s+(DNODE|DLINE|DSURFACE|DVOL)\s+(\d+)"', text):
        topo.setdefault((kind, int(e)), set()).add(int(n))
    blocks = {b.split(":", 1)[0].strip(): b for b in re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\s*$)", text, flags=re.M)}
    flux = blocks.get("SCATRA FLUX CALC LINE CONDITIONS", "")
    pdir = blocks.get("DESIGN POINT DIRICH CONDITIONS", "")
    fl_nodes = set()
    for e in re.findall(r"\bE:\s*(\d+)", flux):
        fl_nodes |= topo.get(("DLINE", int(e)), set())
    pd_nodes = set()
    for e in re.findall(r"\bE:\s*(\d+)", pdir):
        pd_nodes |= topo.get(("DNODE", int(e)), set())
    # a corner node is shared by an outer edge and the interface, so the test is
    # the SHARE of the flux line that carries imported values, not any overlap
    if not fl_nodes or len(pd_nodes) < 3 or len(fl_nodes & pd_nodes) >= 0.5 * len(fl_nodes):
        return []
    coords = {int(n): (float(x), float(y)) for n, x, y in
              re.findall(r'"NODE\s+(\d+)\s+COORD\s+(\S+)\s+(\S+)', text)}
    def _where(ns):
        pts = [coords[n] for n in ns if n in coords]
        if len(pts) < 2:
            return ""
        xs = {round(p[0], 9) for p in pts}; ys = {round(p[1], 9) for p in pts}
        if len(xs) == 1:
            return f" (x = {next(iter(xs)):g})"
        if len(ys) == 1:
            return f" (y = {next(iter(ys)):g})"
        return ""
    return [f"SCATRA FLUX CALC LINE sits OFF the interface: of its {len(fl_nodes)} nodes{_where(fl_nodes)} only "
            f"{len(fl_nodes & pd_nodes)} carry the partner's values (the {len(pd_nodes)} DESIGN POINT DIRICH "
            f"nodes{_where(pd_nodes)}). 4C "
            f"computes the boundary flux on the line you name, so the exported flux would be that of another "
            f"edge (zero at the interface). Name the DLINE whose nodes lie ON the interface, and define it in "
            f"DLINE-NODE TOPOLOGY"]


_BC_HEAD = re.compile(r"DESIGN (POINT|LINE) (?:TRANSPORT )?(DIRICH|NEUMANN) CONDITIONS")
_BOX_EDGES = {"left": (0, 0), "right": (0, 1), "bottom": (1, 0), "top": (1, 1)}   # (fixed axis, low or high end)


def _box_side(text: str, cfg: dict | None = None) -> dict | None:
    """The 2-D box side a Scalar_Transport deck states, or None when the deck is not one.

    The box is the deck's own: the extent of its NODE COORDS, which its 2-D elements must tile exactly (their
    areas add up to the box's; an L-shape or a hole does not). The interface is the config's `iface` (left,
    right, bottom or top: what the participant reads), else the deck's own statement -- the one edge its SCATRA
    FLUX CALC line runs along, else the one edge whose nodes between its ends hold all its POINT entries. Judged
    only for a coupling side: a config naming `iface`, or a deck with a SCATRA FLUX CALC line and POINT
    DIRICH/NEUMANN entries (the route a side takes its imports by). DIRICH and TRANSPORT DIRICH both count: a
    Scalar_Transport run applies either (measured on this binary: a left edge held at 2.0 through each)."""
    if "Scalar_Transport" not in text:
        return None
    xyz = {}
    for n, x, y, z in re.findall(r'"NODE\s+(\d+)\s+COORD\s+(\S+)\s+(\S+)\s+(\S+)"', text):
        try:
            xyz[int(n)] = (float(x), float(y), float(z))
        except ValueError:
            continue
    if len(xyz) < 4:
        return None
    xs, ys, zs = zip(*xyz.values())
    lo, hi = (min(xs), min(ys)), (max(xs), max(ys))
    span = max(hi[0] - lo[0], hi[1] - lo[1])
    if min(hi[0] - lo[0], hi[1] - lo[1]) <= 0 or max(zs) - min(zs) > 1e-9 * span:
        return None
    area = 0.0
    for kind, ids in re.findall(r'"\s*\d+\s+\w+\s+(QUAD4|QUAD8|QUAD9|TRI3|TRI6)\s+((?:\d+\s+)+)', text):
        cn = [int(i) for i in ids.split()][:4 if kind.startswith("QUAD") else 3]
        if any(i not in xyz for i in cn):
            return None
        p = [xyz[i] for i in cn]
        area += abs(0.5 * sum(p[k][0] * p[(k + 1) % len(p)][1] - p[(k + 1) % len(p)][0] * p[k][1] for k in range(len(p))))
    box = (hi[0] - lo[0]) * (hi[1] - lo[1])
    if abs(area - box) > 1e-6 * box:
        return None
    tol = 1e-9 * span
    on = {}
    for e, (a, end) in _BOX_EDGES.items():
        v = (lo, hi)[end][a]
        on[e] = sorted((n for n, c in xyz.items() if abs(c[a] - v) <= tol), key=lambda n, a=a: xyz[n][1 - a])
        if len(on[e]) < 2:
            return None
    topo: dict = {}
    for n, kind, e in re.findall(r'"NODE\s+(\d+)\s+(DNODE|DLINE)\s+(\d+)"', text):
        topo.setdefault((kind, int(e)), set()).add(int(n))
    conds = []
    for b in re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\s*$)", text, flags=re.M):
        head = b.split(":", 1)[0].strip()
        m = _BC_HEAD.fullmatch(head)
        if not m and head != "SCATRA FLUX CALC LINE CONDITIONS":
            continue
        geo, kind = (m.group(1), m.group(2)) if m else ("LINE", "FLUX")
        for entry in re.split(r"^\s*-\s", b, flags=re.M)[1:]:
            e = re.search(r"\bE:\s*(\d+)", entry)
            sw = re.search(r"ONOFF:\s*\[([^\]]*)\]", entry)
            if not e or (sw and "1" not in [x.strip() for x in sw.group(1).split(",")]):
                continue                           # no id, or switched off: it holds and loads nothing
            val = re.search(r"VAL:\s*\[([^\]]*)\]", entry)
            fn = re.search(r"FUNCT:\s*\[([^\]]*)\]", entry)
            conds.append({"geo": geo, "kind": kind, "e": int(e.group(1)),
                          "val": val.group(1).split(",")[0].strip() if val else "",
                          "const": not fn or fn.group(1).split(",")[0].strip() in ("0", ""),
                          "nodes": topo.get(("DNODE" if geo == "POINT" else "DLINE", int(e.group(1))), set())})
    mids = {e: set(ns[1:-1]) for e, ns in on.items()}
    points = set().union(*[c["nodes"] for c in conds if c["geo"] == "POINT"])
    flux = set().union(*[c["nodes"] for c in conds if c["kind"] == "FLUX"])
    iface = str((cfg or {}).get("iface") or "").strip().lower()
    said = f"config iface = {iface}"
    if iface not in on:
        if not (flux and points):
            return None
        iface, said = "", ""
        along = [e for e, ns in on.items() if any(a in flux and b in flux for a, b in zip(ns, ns[1:]))]
        held_at = [e for e in on if len(points & mids[e]) >= 2 and points <= set(on[e])]
        if len(along) == 1:
            iface, said = along[0], f"the edge the SCATRA FLUX CALC line runs along, {along[0]}"
        elif len(held_at) == 1:
            iface, said = held_at[0], f"the edge its POINT entries sit on, {held_at[0]}"
    return {"xyz": xyz, "lo": lo, "hi": hi, "on": on, "mids": mids, "conds": conds, "flux": flux,
            "iface": iface, "said": said}


def _edge_at(side: dict, e: str) -> str:
    a, end = _BOX_EDGES[e]
    return f"{'xy'[a]} = {(side['lo'], side['hi'])[end][a]:g}"


def _stretches(side: dict, e: str, miss: set) -> str:
    """The runs of `miss` along edge e, in the deck's own coordinates ('y = 0.2 .. 0.4; y = 0.7')."""
    a = 1 - _BOX_EDGES[e][0]
    runs, cur = [], []
    for n in side["on"][e]:
        if n in miss:
            cur.append(n)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return "; ".join(f"{'xy'[a]} = {side['xyz'][r[0]][a]:g}" + (f" .. {side['xyz'][r[-1]][a]:g}" if len(r) > 1 else "")
                     for r in runs[:3]) + ("; ..." if len(runs) > 3 else "")


def _box_side_facts(text: str, cfg: dict | None = None) -> list[str]:
    """What the conditions of a 2-D box side reach, counted from the deck (see _box_side).

    MEASURED on two recorded coupled decks that finished normally and coupled on with no word for half an
    hour: one held its interface at the constant outer value (one DLINE on the interface and on the opposite
    edge, under one DESIGN LINE DIRICH and the flux-calc condition), and one left part of a held outer edge
    free (its DLINE stopped short of the edge's far end). The flux-calc rule above stays silent below three
    POINT DIRICH nodes, and nothing else looked. The facts, each stated only where the deck states enough: an
    edge the conditions reach in part (a free interface corner on a held edge among them); a line on separate
    stretches of the boundary, or with nodes inside the box (measured: outer lines built from consecutive node
    ids took a row for a column); an interface (or, with none stated, a flux-calc line) held at one constant;
    POINT entries that reach part of the interface."""
    side = _box_side(text, cfg)
    if side is None:
        return []
    on, mids, conds, iface = side["on"], side["mids"], side["conds"], side["iface"]
    on_set = {e: set(ns) for e, ns in on.items()}
    bc = [c for c in conds if c["kind"] != "FLUX"]
    covered = set().union(*[c["nodes"] for c in bc])
    out = []
    for e in on:
        if e == iface or not covered & mids[e]:
            continue                        # the interface is counted below; an edge with no condition is natural
        miss = [n for n in on[e] if n not in covered]
        if not miss:
            continue
        by = [f"{c['geo']} {c['kind']} E {c['e']} on {len(c['nodes'] & on_set[e])}" for c in bc if c["nodes"] & on_set[e]]
        k = len(on[e]) - len(miss)
        out.append(f"BOX EDGE {e} ({_edge_at(side, e)}): DIRICH/NEUMANN conditions reach {k} of its {len(on[e])} "
                   f"nodes ({', '.join(by[:4])}{', ...' if len(by) > 4 else ''}) and the other {len(miss)} "
                   f"carr{'ies' if len(miss) == 1 else 'y'} none ({_stretches(side, e, set(miss))}). 4C leaves a "
                   f"boundary node without a condition natural (zero flux), so this edge is held or loaded on {k} "
                   f"nodes and natural on {len(miss)}. A DLINE "
                   f"lists every node of its edge, both ends included")
    # a line on separate stretches of the boundary: the segments of the boundary loop inside it, in runs
    loop = on["bottom"] + on["right"][1:] + on["top"][::-1][1:] + on["left"][::-1][1:-1]
    seg_edge = [next((e for e in on if loop[i] in on_set[e] and loop[(i + 1) % len(loop)] in on_set[e]), "")
                for i in range(len(loop))]
    named: dict = {}
    for c in conds:
        if c["geo"] == "LINE":
            named.setdefault(c["e"], []).append(c)
    boundary = set().union(*on_set.values())
    inside_only: list = []                  # (DLINE, its nodes inside the box, all its nodes, condition families)
    for d, cs in sorted(named.items()):
        ns = cs[0]["nodes"]
        inner = len([n for n in ns if n in side["xyz"] and n not in boundary])   # off every edge of the box
        inside = [loop[i] in ns and loop[(i + 1) % len(loop)] in ns for i in range(len(loop))]
        runs, cur = [], []
        if any(inside) and not all(inside):
            first = inside.index(False)
            for k in range(1, len(loop) + 1):
                i = (first + k) % len(loop)
                if inside[i]:
                    cur.append(i)
                elif cur:
                    runs.append(cur)
                    cur = []
            if cur:
                runs.append(cur)
        edges = [sorted({seg_edge[i] for i in r}, key=list(on).index) for r in runs]
        split = len(runs) >= 2 and (any(c["kind"] == "FLUX" for c in cs)
                                    or (iface and any(iface in es for es in edges) and any(set(es) - {iface} for es in edges)))
        names = ", ".join(dict.fromkeys("SCATRA FLUX CALC" if c["kind"] == "FLUX" else f"DESIGN {c['geo']} {c['kind']}"
                                        for c in cs))
        if not split:
            if inner:
                inside_only.append((d, inner, len(ns), names))
            continue                        # two held outer edges on one line: one value on both is allowed
        where = " and ".join(f"the {' and '.join(es)} edge{'s' if len(es) > 1 else ''} at "
                             f"{', '.join(_edge_at(side, e) for e in es)} ({len(r) + 1} nodes)" for es, r in zip(edges, runs))
        out.append(f"SEPARATE STRETCHES: DLINE {d} lies on {len(runs)} separate stretches of the box boundary, {where}"
                   + (f", and {inner} of its nodes lie inside the box, off every edge" if inner else "")
                   + f". Every condition filed on it ({names}, E {d}) acts on all of them"
                   + (f"; the interface is the {iface} edge" if iface else "")
                   + ". A line meant for one edge lists the nodes of that edge only")
    if inside_only:                         # one finding for them all: a numbering slip repeats per line
        one = len(inside_only) == 1
        out.append(f"LINE INSIDE THE BOX: DLINE {', '.join(str(d) for d, *_ in inside_only[:8])}"
                   f"{', ...' if len(inside_only) > 8 else ''} {'has' if one else 'have'} "
                   f"{sum(i for _, i, _, _ in inside_only)} of {'its' if one else 'their'} "
                   f"{sum(n for _, _, n, _ in inside_only)} nodes inside the box, off every edge, and every condition "
                   f"filed on {'it' if one else 'them'} ({', '.join(dict.fromkeys(nm for *_, nm in inside_only))}) acts "
                   f"there" + (f"; the interface is the {iface} edge" if iface else "")
                   + ". A line meant for one edge lists the nodes of that edge only")
    # the interface -- or, with none stated, the flux-calc line -- held at one constant
    pdir = set().union(*[c["nodes"] for c in conds if c["geo"] == "POINT" and c["kind"] == "DIRICH"])
    if iface:
        target, label = mids[iface], f"the interface ({side['said']}, {_edge_at(side, iface)})"
    else:
        target = side["flux"] & set().union(*mids.values())
        label = ("the SCATRA FLUX CALC line DLINE "
                 + ", ".join(str(c["e"]) for c in conds if c["kind"] == "FLUX"))
    held_const = False
    for c in conds:
        held = (c["nodes"] & target) - pdir if c["geo"] == "LINE" and c["kind"] == "DIRICH" and c["const"] else set()
        if held:
            held_const = True
            out.append(f"HELD AT ONE VALUE: {label} is held at the one value VAL [{c['val']}] (FUNCT 0) by DESIGN "
                       f"LINE DIRICH E {c['e']} on {len(held)} of its {len(target)} nodes between the box corners, and "
                       f"no POINT DIRICH overrides them there: those nodes carry {c['val']} whatever the partner sends. "
                       f"A Dirichlet side holds each interior interface node at its own imported value (one DESIGN "
                       f"POINT DIRICH each); a Neumann side holds none of them and loads them")
            break
    # POINT entries that reach part of the interface
    if iface and not held_const:
        pts = [c for c in conds if c["geo"] == "POINT" and c["nodes"] & mids[iface]]
        reach = set().union(*[c["nodes"] for c in pts]) & mids[iface]
        if 0 < len(reach) < len(mids[iface]):
            miss = mids[iface] - reach
            what = "no condition" if not miss & covered else "no POINT entry"
            out.append(f"INTERFACE POINT VALUES: POINT {'/'.join(sorted({c['kind'] for c in pts}))} entries reach "
                       f"{len(reach)} of the {len(mids[iface])} nodes between the ends of the interface "
                       f"({side['said']}, {_edge_at(side, iface)}); the other {len(miss)} "
                       f"({_stretches(side, iface, miss)}) carry {what}. Each interior interface node takes the "
                       f"partner's datum through a POINT entry of its own")
    return out


def _yaml_parse_error(text: str) -> list[str]:
    """The deck as YAML: 4C's reader is a YAML parser, and a structural slip (an entry's keys indented
    unevenly, a bare token where a mapping was open) stops it with 'ERROR: parse error <line>:<col>'
    (measured on a recorded run: `NUMDOF: 3` at 109:13 under a DESIGN POINT DIRICH entry). PyYAML names
    the same place before any run; the offending line is quoted."""
    try:
        import yaml  # noqa: PLC0415
    except Exception:                                   # noqa: BLE001
        return []
    try:
        yaml.safe_load(text)
        return []
    except yaml.YAMLError as e:                         # noqa: BLE001
        mark = getattr(e, "problem_mark", None) or getattr(e, "context_mark", None)
        if mark is None:
            return [f"the deck is not valid YAML: {str(e).splitlines()[0][:120]}"]
        lines = text.splitlines()
        row = lines[mark.line] if 0 <= mark.line < len(lines) else ""
        prob = getattr(e, "problem", None) or str(e).splitlines()[0]
        return [f"the deck is not valid YAML at line {mark.line + 1}, column {mark.column + 1}: {prob} -- the line reads "
                f"`{row.strip()[:80]}`; 4C's reader stops there with 'ERROR: parse error {mark.line + 1}:{mark.column + 1}'. "
                f"Inside a `- E: <id>` entry every further key (NUMDOF, ONOFF, VAL, FUNCT, TAG) is indented to the same column as E"]


def _elements_with_unknown_nodes(text: str) -> list[str]:
    """Element rows naming node ids that NODE COORDS does not define (measured on a recorded run: 'Element 17
    cannot find node 27' -- 4C stops in its element reader; a lint sees it before the run)."""
    ids = {int(a) for a in re.findall(r'"NODE\s+(\d+)\s+COORD\b', text)}
    if not ids:
        return []
    bad = []
    for e, ids_txt in re.findall(r'"\s*(\d+)\s+\w+\s+(?:QUAD4|TRI3|HEX8|TET4|LINE2)\s+((?:\d+\s+)+)', text):
        missing = [int(i) for i in ids_txt.split() if int(i) not in ids]
        if missing:
            bad.append((e, missing))
    if not bad:
        return []
    e, missing = bad[0]
    return [f"{len(bad)} element row(s) name node ids that NODE COORDS does not define (first: element {e} -> node(s) "
            f"{' '.join(map(str, missing[:4]))}; {len(ids)} nodes are defined, ids {min(ids)}..{max(ids)}): 4C stops with 'Element "
            f"{e} cannot find node {missing[0]}'. Node ids in element rows are the NODE numbers of NODE COORDS, 1-based, and the "
            f"top layer of a slab is id + (number of 2-D nodes)"]


def _unquoted_table_rows(text: str) -> list[str]:
    """Rows of NODE COORDS, the element sections and the topology sections that are not quoted YAML strings.
    4C's YAML reader stops at the first such row with 'could not find ':' colon after key' and the position of
    the token (measured, te4c13 repair loop: `- 2 TRANSP QUAD4 2 3 12 11 MAT 1 TYPE Std`)."""
    out = []
    bad, first = 0, None
    for b in re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\s*$)", text, flags=re.M):
        head = b.split(":", 1)[0].strip()
        if not (head.endswith(" ELEMENTS") or head == "NODE COORDS" or head.endswith("-NODE TOPOLOGY")):
            continue
        for row in re.findall(r"^\s*-\s+([^\"'\n][^\n]*)$", b, re.M):
            if re.match(r"(\d+\s+\w|NODE\s+\d+)", row):
                bad += 1
                first = first or (head, row.strip())
    if bad:
        head, row = first
        out.append(f"{bad} table row(s) are not quoted YAML strings (first, in {head}: `- {row[:60]}`): every NODE COORDS, "
                   f"element and topology row is ONE quoted string, `- \"{row[:40]}\"`; unquoted, the YAML reader stops at the "
                   f"first bare token with 'could not find ':' colon after key' and a line:column position")
    return out


def _bad_hex8_slabs(text: str) -> list[str]:
    """A one-element-thick HEX8 slab whose element node order is not (bottom quad counter-clockwise, then
    the same four nodes on the top layer). Judged only when the deck's z coordinates take exactly two
    values (the slab route). Measured (te4c13, 2026-09-12): a worker numbered nodes layer-interleaved and
    wrote "1 2 11 10 100 101 110 109": 4C parsed it and stopped in the thermo element with 'ZERO OR NEGATIVE
    JACOBIAN DETERMINANT' -- a runtime message that names no node."""
    coords = {int(a): (float(x), float(y), float(z)) for a, x, y, z in
              re.findall(r'"NODE\s+(\d+)\s+COORD\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)"', text)}
    if not coords:
        return []
    zs = sorted({round(c[2], 9) for c in coords.values()})
    if len(zs) != 2:
        return []
    out = []
    bad = 0
    first = None
    for e, ids in re.findall(r'"\s*(\d+)\s+\w+\s+HEX8\s+((?:\d+\s+){8})', text):
        nn = [int(i) for i in ids.split()]
        if not all(i in coords for i in nn):
            continue
        p = [coords[i] for i in nn]
        bottom_z = {round(q[2], 9) for q in p[:4]}
        top_z = {round(q[2], 9) for q in p[4:]}
        same_xy = all(abs(p[i][0] - p[i + 4][0]) < 1e-9 and abs(p[i][1] - p[i + 4][1]) < 1e-9 for i in range(4))
        area = 0.5 * sum(p[q][0] * p[(q + 1) % 4][1] - p[(q + 1) % 4][0] * p[q][1] for q in range(4))
        ok = (len(bottom_z) == 1 and len(top_z) == 1 and bottom_z != top_z and same_xy
              and (area > 1e-14 if next(iter(top_z)) > next(iter(bottom_z)) else area < -1e-14))
        if not ok:
            bad += 1
            first = first or (e, nn, sorted(bottom_z | top_z), same_xy, area)
    if bad:
        e, nn, zz, same_xy, area = first
        why = ("its first four nodes do not lie on one layer" if len({round(coords[i][2], 9) for i in nn[:4]}) != 1 else
               "nodes 5-8 are not the same (x, y) as nodes 1-4" if not same_xy else
               f"the bottom quad runs clockwise (signed area {area:.3g})")
        out.append(f"{bad} HEX8 element(s) of the slab are not a well-formed one-layer hex (first: element {e} nodes "
                   f"{' '.join(map(str, nn))}: {why}): the node order is the bottom quad counter-clockwise seen from +z, "
                   f"then the SAME four nodes on the top layer in the same order -- for node id = i + 1 + (NX + 1) * j "
                   f"and N2 2-D nodes, cell (i, j) is (id, id + 1, id + NX + 2, id + NX + 1, id + N2, id + 1 + N2, "
                   f"id + NX + 2 + N2, id + NX + 1 + N2); 4C parses any order and stops later with 'ZERO OR NEGATIVE "
                   f"JACOBIAN DETERMINANT' naming no node")
    return out


def _interior_lines(text: str) -> list[str]:
    """A DLINE whose nodes all lie on one coordinate line strictly INSIDE the mesh, with a condition filed on it.

    Measured (deck step trial, 2026-09-12): a worker's 'x = 0' line landed on the nodes at x = 0.7 by an index
    slip; 4C finished normally, imposed T = 0 across the interior and the field was wrong by 87% -- nothing in
    the console says so. Only axis-aligned lines are judged, and only lines some LINE condition names."""
    coords = {int(a): (float(x), float(y)) for a, x, y in
              re.findall(r'"NODE\s+(\d+)\s+COORD\s+([-\d.eE+]+)\s+([-\d.eE+]+)', text)}
    if len(coords) < 4:
        return []
    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    ext = {0: (min(xs), max(xs)), 1: (min(ys), max(ys))}
    tol = 1e-6 * max(1.0, ext[0][1] - ext[0][0], ext[1][1] - ext[1][0])
    lines: dict[int, list] = {}
    for n, d in re.findall(r'"NODE\s+(\d+)\s+DLINE\s+(\d+)"', text):
        lines.setdefault(int(d), []).append(int(n))
    used: set[int] = set()
    for b in re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\s*$)", text, flags=re.M):
        head = b.split(":", 1)[0].strip()
        if head.endswith("CONDITIONS") and re.search(r"\bLINE\b", head):
            used |= {int(x) for x in re.findall(r"\bE:\s*(\d+)", b)}
    out = []
    for d, nodes in sorted(lines.items()):
        pts = [coords[n] for n in nodes if n in coords]
        if len(pts) < 2 or d not in used:
            continue
        for ax, name in ((0, "x"), (1, "y")):
            vals = {round(p[ax], 9) for p in pts}
            if len(vals) == 1:
                v = next(iter(vals))
                lo, hi = ext[ax]
                if lo + tol < v < hi - tol:
                    out.append(f"DLINE {d} ({len(pts)} nodes) lies on {name} = {v:g}, INSIDE the mesh ({name} spans "
                               f"{lo:g}..{hi:g}), and a condition is filed on it -- a boundary line placed on interior "
                               f"nodes (measured: 4C finishes normally and imposes the value across the interior)")
    return out


def _lines_without_an_element_edge(text: str) -> list[str]:
    """A DLINE whose nodes share no edge of any 2-D element is a zero-length boundary: a condition on
    it integrates to nothing, and 4C's flux table divides by its area and dies on a floating point
    exception (measured on a worker deck whose interface node ids did not match its element
    numbering). Only decks with 2-D element connectivity are judged."""
    quads = re.findall(r'"\s*\d+\s+\w+\s+(?:QUAD4|QUAD8|QUAD9|TRI3|TRI6)\s+((?:\d+\s+)+)', text)
    if not quads:
        return []
    edges = set()
    for q in quads:
        ids = [int(x) for x in q.split()]
        if len(ids) < 3:
            continue
        corners = ids[:4] if len(ids) >= 4 and len(ids) not in (6,) else ids[:3]
        for a, b in zip(corners, corners[1:] + corners[:1]):
            edges.add((min(a, b), max(a, b)))
    if not edges:
        return []
    out = []
    lines: dict = {}
    for n, d in re.findall(r'"NODE\s+(\d+)\s+DLINE\s+(\d+)"', text):
        lines.setdefault(d, set()).add(int(n))
    for d, nodes in sorted(lines.items(), key=lambda p: int(p[0])):
        if len(nodes) < 2:
            out.append(f"DLINE {d} has {len(nodes)} node(s): a line condition needs the consecutive nodes of an edge")
            continue
        if not any((min(a, b), max(a, b)) in edges for a in nodes for b in nodes if a < b):
            out.append(f"DLINE {d} ({len(nodes)} nodes) shares no edge with any element: its node ids do not match "
                       "the element numbering, so a condition on it is a zero-length boundary (4C's flux table "
                       "then divides by zero)")
    return out


def field_scale_findings(deck_text: str, out_dir: Path) -> list[str]:
    """A finished 4C run whose field dwarfs every number the deck prescribes is a deck defect, not a
    solution. Measured on one run: the scatra run finished normally with T = 8.5e13 at every
    node while the deck's largest Dirichlet value was 0.8 and its source coefficient 2*pi^2; nothing in
    the console said so and the parent read it as a result. Reads the newest VTU next to the deck with
    meshio, compares each point field's peak with the largest prescribed VAL / FUNCT constant (at least
    1), and names a ratio above 1e6. Names the fact only."""
    out: list[str] = []
    try:
        import meshio  # noqa: PLC0415
    except Exception:                                   # noqa: BLE001
        return out
    try:
        vtus = sorted(Path(out_dir).glob("*vtk-files/*.vtu"), key=lambda q: q.stat().st_mtime)
        if not vtus:
            return out
        vals = [abs(float(x)) for x in re.findall(r"VAL:\s*\[([^\]]*)\]", deck_text) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", x)]
        funcs = [abs(float(x)) for x in re.findall(r'SYMBOLIC_FUNCTION_OF_SPACE_TIME:\s*"([^"]*)"', deck_text)
                 for x in re.findall(r"(?<![\w.])\d+\.?\d*(?:[eE][-+]?\d+)?", x)]
        scale = max([1.0] + vals + funcs)
        m = meshio.read(vtus[-1])
        for name, arr in m.point_data.items():
            import numpy as np  # noqa: PLC0415
            a = np.asarray(arr, float)
            if a.size == 0 or not np.isfinite(a).all():
                if a.size and not np.isfinite(a).all():
                    out.append(f"4C FIELD {name} in {vtus[-1].parent.name} carries non-finite values: the run finished, the result is not a solution")
                continue
            peak = float(np.abs(a).max())
            if peak > 1e6 * scale:
                out.append(f"4C FIELD SCALE: {name} peaks at {peak:.3g} in {vtus[-1].parent.name} while the largest number the deck prescribes "
                           f"(VAL entries, FUNCT constants) is {scale:.3g}: a field this far above its own data is a deck defect (a "
                           f"material parameter left at a default, a zero conductivity or stiffness, an unconstrained dof), not a solution")
    except Exception:                                   # noqa: BLE001
        return out
    return out


def side_dir_report(side: Path) -> dict:
    """What 4C left behind in a participant's directory: which runs finished (their VTU folders),
    4C's own error lines per log, and the named defects per deck. Reads only the agent's own files."""
    side = Path(side)
    vtk_dirs = sorted(p for p in side.glob("*-vtk-files") if p.is_dir())
    finished = {}
    for d in vtk_dirs:
        kinds = sorted({re.sub(r"-\d+-\d+\.vtu$", "", q.name) for q in d.glob("*.vtu")})
        if kinds:
            finished[d.name[:-len("-vtk-files")]] = kinds
    monitors = sorted(p.name for p in side.glob("*_monitor_dbc.yaml"))
    errors, tracebacks, consoles = {}, {}, {}
    for lg in sorted(side.glob("*.log")) + sorted(side.glob("*.txt")):
        try:
            txt = lg.read_text(errors="ignore")
        except OSError:
            continue
        said = fourc_error_lines(txt)
        if said:
            errors[lg.name] = said
        tb = python_stop_lines(txt)
        if tb:
            tracebacks[lg.name] = tb
        if (not said and not tb and "finished normally" not in txt
                and ("4C" in txt or "processor 0" in txt or "Problem type" in txt or "PROBLEMTYPE" in txt)):
            # a 4C console with neither an error block nor a signal: its last lines are the only verdict
            tail = [l.strip() for l in txt.splitlines() if l.strip() and not set(l.strip()) <= set("+-|=*")]
            consoles[lg.name] = " | ".join(tail[-3:])
    defects = {}
    decks = sorted(side.glob("*.4C.yaml")) or [p for p in sorted(side.glob("*.yaml")) if "monitor_dbc" not in p.name]
    # the binary the agent's own config names (or the environment's): its grammar judges the section names
    cfg = {}
    try:
        cfg = json.loads((side / "config.json").read_text() or "{}") if (side / "config.json").is_file() else {}
    except Exception:                                   # noqa: BLE001
        cfg = {}
    _bin = cfg.get("fourc_bin") or os.environ.get("FOURC_BIN")
    _ld = cfg.get("fourc_ld") or os.environ.get("FOURC_LD")
    if not _bin:                    # the agent's own script names no binary: the installed one judges
        _bin, _ld = binary_and_ld()
    g = grammar(_bin, _ld)
    valid, elements = g["sections"], g["elements"]
    for dk in decks:
        try:
            txt = dk.read_text(errors="ignore")
        except OSError:
            continue
        why = lint_deck(txt, cfg) + unknown_sections(txt, valid, elements) + material_defects(txt, g.get("materials", {}))
        if why:
            defects[dk.name] = why
    return {"finished": finished, "monitors": monitors, "errors": errors, "defects": defects,
            "tracebacks": tracebacks, "consoles": consoles, "decks": [d.name for d in decks]}


# ---------------------------------------------------------------------------------------------
# One judgement for every gate that reads a deck: check_input, the write hook, the run hook.
# ---------------------------------------------------------------------------------------------
_DECK_MARKERS = ("PROBLEM TYPE:", "PROBLEMTYPE:", "NODE COORDS", "STRUCTURE ELEMENTS", "TRANSPORT ELEMENTS",
                 "FLUID ELEMENTS", "THERMO ELEMENTS", "DNODE-NODE TOPOLOGY", "DLINE-NODE TOPOLOGY")


def looks_like_deck(text: str) -> bool:
    """A 4C YAML deck: at least two section heads and one of 4C's own section names."""
    if not text or not any(m in text for m in _DECK_MARKERS):
        return False
    return len(re.findall(r"^[A-Z][A-Z0-9 _/.:-]*?:\s*$", text, re.M)) >= 2


def binary_and_ld() -> tuple[str | None, str | None]:
    """The installed 4C binary (FOURC_BINARY, else the backend's finder) and the library path its
    `-p` dump needs. (None, None) when there is no binary: then section names are not judged."""
    _bin = os.environ.get("FOURC_BINARY")
    if not (_bin and Path(_bin).is_file()):
        _bin = None
        try:
            from backends.fourc.backend import _find_fourc_binary   # noqa: PLC0415
            found = _find_fourc_binary()
            _bin = str(found) if found else None
        except Exception:                               # noqa: BLE001
            _bin = None
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if Path("/opt/4C-dependencies/lib").is_dir() and "4C-dependencies" not in ld:
        ld = "/opt/4C-dependencies/lib" + (":" + ld if ld else "")
    return _bin, (ld or None)


def deck_judgement(text: str) -> list[str]:
    """Every defect the lint can name in ONE pass: the measured deck defects, the section names
    against the installed binary's own grammar (with the closest known names), the material
    parameters against its material specs. Names defects only; writes and completes nothing."""
    findings = lint_deck(text)
    _bin, ld = binary_and_ld()
    g = grammar(_bin, ld)
    if g["sections"]:
        findings += unknown_sections(text, g["sections"], g["elements"])
        findings += material_defects(text, g.get("materials", {}))
    else:
        findings.append("(section names not judged: no 4C binary found for `4C -p`)")
    return findings


_RUN_RE = re.compile(r"(?:^|[\s;&|(`])(?P<bin>\S*/4C|4C)\s+(?:-{1,2}[\w=-]+\s+)*(?P<deck>[^\s;&|>]+\.(?:4C\.)?(?:yaml|yml|dat))(?=$|[\s;&|>)])",
                     re.M)


def run_command_deck(command: str, cwd: Path) -> Path | None:
    """The deck a shell command hands to the 4C binary (`.../4C deck.yaml out`, also behind mpirun,
    stdbuf or `cd side_A &&`), resolved on disk; None when the command does not run 4C."""
    m = _RUN_RE.search(command or "")
    if not m:
        return None
    deck = m.group("deck").strip("'\"")
    cand = Path(deck)
    if not cand.is_absolute():
        cand = Path(cwd) / deck
        cd = re.search(r"\bcd\s+([^\s;&|]+)\s*(?:&&|;)", command)
        if cd and not cand.is_file():
            cand = Path(cwd) / cd.group(1).strip("'\"") / deck
    if cand.is_file():
        return cand
    try:
        hits = sorted(Path(cwd).rglob(Path(deck).name))
    except OSError:
        hits = []
    return hits[0] if len(hits) == 1 else None
