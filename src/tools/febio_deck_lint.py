"""openPASO-side FEBio deck lint: names, from a .feb deck's own text, the defects FEBio accepts without
a word, and the ones it stops on with a message that does not name the cause. A verification gate,
not a generator: it writes nothing and completes nothing.

Every class below was run on this install (FEBio 4.12, febio4) as a placeholder slab deck one change
away from a running one (tests/test_a_febio_deck_defect_febio_accepts_silently_is_named.py):

  silent    a NodeData with fewer entries than its node set     the nodes with no entry take 0
  silent    a held node set that includes interior nodes        they keep the held value
  silent    u_z held on part of a one-element-thick slab        neither plane strain nor plane stress
  silent    a map read by a bc or a load on another node set    entries taken by position, not node
  silent    one dof held by two bcs with different values       prescribed beats zero; of two
                                                                prescribed, the later one
  silent    a NodeSet whose ids are separated by spaces         the first number of each piece only
  crashes   an <elem> whose node ids are separated by spaces    SIGSEGV after 'Reading file', no message
  silent    a scalar NodeData entry carrying several numbers    the first number only
  silent    a body_load that is 0 in every component            it applies nothing
  misleads  a lid outside 1..N of its node set                  'invalid value for attribute "lid"'
  misleads  a map that no NodeData defines                      'std::exception', nothing else
  misleads  lc="k" with no load_controller k                    'Invalid load curve ID'
  misleads  <Control> with no <solver>                          'needs to have property "solver"'
  misleads  <MeshData> inside <Mesh>                            'unrecognized tag'
  misleads  a load outside <Loads>, or a top-level <BodyLoad>   'unrecognized tag'
  misleads  a <bc> whose type is a load                         'invalid value for attribute "type"'
  misleads  a load_controller of type "linear" or with none     'invalid value' / 'missing attribute'
  misleads  <elem_data> in a <logfile>                          'unrecognized tag'
  misleads  a section never closed                              a message about a later tag

What FEBio names itself (an unknown node set, an unknown tag, a bad attribute) is left to FEBio.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import defaultdict

# hex8 faces in FEBio's node order (bottom 0-3, top 4-7): four sides, then bottom and top
_HEX8_FACES = ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (0, 3, 2, 1), (4, 5, 6, 7))
_MEASURED = "measured on this install"


def looks_like_deck(text: str) -> bool:
    return isinstance(text, str) and "<febio_spec" in text


def _id_list(text: str) -> tuple:
    """FEBio's own reading of an id list (XMLTag::value(vector<int>&)): pieces between commas, each
    read as `a`, `a:b` or `a:b:step`; whatever follows the first number of a piece is dropped.
    Returns (ids, pieces that held more than one number)."""
    ids, lost = [], 0
    for piece in (text or "").split(","):
        m = re.match(r"\s*(-?\d+)(?::(-?\d+)(?::(-?\d+))?)?", piece)
        if not m:
            continue
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        s = int(m.group(3)) if m.group(3) else 1
        ids.extend(range(a, b + 1, s if s > 0 else 1))
        if re.search(r"\d\s+-?\d", piece.strip()):
            lost += 1
    return ids, lost


def _num(text):
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


class _Deck:
    """The parts of a FEBio 4 deck the checks read, in FEBio's own terms."""

    def __init__(self, root):
        self.root = root
        self.nodes = {}
        self.sets = {}           # name -> (ordered ids, True when the order is FEBio's own)
        self.spaced = {}         # NodeSet name -> (ids FEBio keeps, numbers written)
        self.elems = []          # (type, [ids])
        self.spaced_elems = []   # ids of the <elem> entries whose node ids are separated by spaces
        self.surfaces = {}
        self.childed = []        # NodeSets written as child tags
        mesh = root.find("Mesh")
        if mesh is not None:
            for nb in mesh.findall("Nodes"):
                block = []
                for n in nb.findall("node"):
                    try:
                        nid = int(n.get("id"))
                        xyz = tuple(float(v) for v in re.split(r"[,\s]+", (n.text or "").strip()) if v)
                    except (TypeError, ValueError):
                        continue
                    if len(xyz) == 3:
                        self.nodes[nid] = xyz
                        block.append(nid)
                if nb.get("name"):
                    self.sets[nb.get("name")] = (block, True)
            for eb in mesh.findall("Elements"):
                for e in eb.findall("elem"):
                    ids, lost = _id_list(e.text)
                    self.elems.append(((eb.get("type") or "").lower(), ids))
                    if lost:
                        self.spaced_elems.append(e.get("id") or "?")
            for ns in mesh.findall("NodeSet"):
                if len(ns):
                    self.childed.append(ns.get("name") or "")      # FEBio refuses it; no set is made
                    continue
                ids, lost = _id_list(ns.text or "")
                if ns.get("name"):
                    self.sets[ns.get("name")] = (ids, True)
                    if lost:
                        self.spaced[ns.get("name")] = (len(ids), len(re.findall(r"-?\d+", ns.text or "")))
            for sf in mesh.findall("Surface"):
                ids = set()
                for f in sf:
                    ids.update(_id_list(f.text or "")[0])
                if sf.get("name"):
                    self.surfaces[sf.get("name")] = sorted(ids)
        self.maps = {}
        # a <MeshData> inside <Mesh> is named once below; its maps still count as written
        for md in root.findall("MeshData") + (mesh.findall("MeshData") if mesh is not None else []):
            for nd in md.findall("NodeData"):
                lids = []
                for x in nd.findall("node"):
                    try:
                        lids.append(int(x.get("lid")))
                    except (TypeError, ValueError):
                        lids.append(None)
                self.maps[nd.get("name")] = {"set": nd.get("node_set"), "lids": lids,
                                             "generator": bool(nd.get("type")),
                                             "scalar": (nd.get("data_type") or "scalar").strip() == "scalar",
                                             "values": [(x.text or "") for x in nd.findall("node")]}
            for tag in ("ElementData", "SurfaceData"):
                for ed in md.findall(tag):
                    self.maps.setdefault(ed.get("name"), {"set": None, "lids": [], "generator": True,
                                                          "values": []})
        self.controllers = set()
        for ld in root.findall("LoadData"):
            for lc in ld.findall("load_controller"):
                self.controllers.add((lc.get("id") or "").strip())

    def node_set(self, name):
        """(ordered ids, order known) for a node_set attribute, as FEBio resolves it; None if unknown."""
        if not name:
            return None
        if name in self.sets:
            return self.sets[name]
        if name.startswith("@surface:") and name[9:] in self.surfaces:
            return (self.surfaces[name[9:]], False)
        return None


def _boundary_nodes(deck: _Deck):
    """(boundary node ids, one-element-thick slab?) from the hex8 faces that belong to one element;
    for a slab only its side faces count, since both z faces hold every node. None when the mesh is
    not all hex8 (the check then abstains)."""
    if not deck.elems or any(t != "hex8" or len(c) != 8 for t, c in deck.elems):
        return None
    zs = {round(p[2], 12) for p in deck.nodes.values()}
    count = defaultdict(int)
    for _, c in deck.elems:
        for f in _HEX8_FACES:
            count[frozenset(c[i] for i in f)] += 1
    slab = len(zs) == 2
    out = set()
    for face, k in count.items():
        if k != 1:
            continue
        if slab and len({round(deck.nodes[n][2], 12) for n in face if n in deck.nodes}) == 1:
            continue
        out.update(face)
    return out, slab


def _holds(deck: _Deck):
    """Every bc that holds a displacement dof: (name, type, set name, dofs, value) where value is
    0.0, a number, ('map', name) or None (not read)."""
    out = []
    for bnd in deck.root.findall("Boundary"):
        for bc in bnd.findall("bc"):
            t = (bc.get("type") or "").strip()
            name = bc.get("name") or t
            if t == "zero displacement":
                dofs = [d for d, tag in (("x", "x_dof"), ("y", "y_dof"), ("z", "z_dof"))
                        if (bc.findtext(tag) or "").strip() == "1"]
                out.append((name, t, bc.get("node_set"), dofs, 0.0))
            elif t == "prescribed displacement":
                d = (bc.findtext("dof") or "").strip()
                v = bc.find("value")
                val = None
                if v is not None:
                    if (v.get("type") or "") == "map":
                        val = ("map", (v.text or "").strip())
                    elif not v.get("lc"):
                        val = _num(v.text)
                rel = (bc.findtext("relative") or "0").strip()
                out.append((name, t, bc.get("node_set"), [d] if d in ("x", "y", "z") else [],
                            val if rel in ("0", "") else None))
    return out


def _value_readers(deck: _Deck):
    """Every <value> that reads a map, with where it sits: (where, set name, map name)."""
    out = []
    for sec in ("Boundary", "Loads"):
        for s in deck.root.findall(sec):
            for el in s:
                v = el.find("value")
                if v is not None and (v.get("type") or "") == "map":
                    where = f"{el.tag} '{el.get('name') or el.get('type')}'"
                    out.append((where, el.get("node_set"), (v.text or "").strip()))
    return out


_TAG = re.compile(r"<!--.*?-->|<\?.*?\?>|<(/?)([A-Za-z_][\w.:-]*)[^>]*?(/?)>", re.S)


def _not_well_formed(text: str, exc) -> str:
    """The open tag a deck never closes, found as FEBio's reader meets it: FEBio stops on a later tag
    with a message about that tag (measured: an unclosed <Elements> gives 'tag "NodeSet" (line N) :
    missing attribute "id"', an unclosed <Loads> 'tag "LoadData" (line N) : unrecognized tag')."""
    stack, line, at = [], 1, 0
    for m in _TAG.finditer(text):
        line, at = line + text.count("\n", at, m.start()), m.start()
        if not m.group(2) or m.group(3):
            continue
        if not m.group(1):
            stack.append((m.group(2), line))
        elif stack and stack[-1][0] == m.group(2):
            stack.pop()
        elif stack:
            tag, opened = stack[-1]
            return (f"the deck is not well-formed XML: <{tag}> opened on line {opened} is not closed before "
                    f"</{m.group(2)}> on line {line}. FEBio stops on a later tag with a message about that "
                    f"tag, not this one ({_MEASURED}: an unclosed <Elements> gives 'tag \"NodeSet\" (line N) "
                    f": missing attribute \"id\"').")
    if stack:
        return (f"the deck is not well-formed XML: <{stack[-1][0]}> opened on line {stack[-1][1]} is never "
                f"closed.")
    return f"the deck is not well-formed XML: {exc}."


def lint_deck(text: str) -> list:
    """The defects of a FEBio deck, named from its own text. [] when the text is no deck or carries
    none of the measured classes; a deck that is not well-formed XML gets that one finding."""
    if not looks_like_deck(text):
        return []
    try:
        root = ET.fromstring(re.sub(r"^\s*<\?xml[^>]*\?>", "", text.strip().lstrip("\ufeff")))
    except (ET.ParseError, ValueError) as exc:
        return [_not_well_formed(text, exc)]
    deck = _Deck(root)
    out = []

    # A NodeSet written as child tags: FEBio 4 reads one comma-separated list and says only "invalid value:".
    for name in deck.childed:
        out.append(f"NodeSet '{name}' lists its ids as child tags: FEBio 4 reads a NodeSet as one comma-separated "
                   f"id list and stops with 'tag \"NodeSet\" (line N) : invalid value:' ({_MEASURED}).")

    # An <elem> written with spaces: FEBio 4.12 crashes while it reads the deck and prints nothing more.
    if deck.spaced_elems:
        k = len(deck.spaced_elems)
        out.append(f"{k} <elem> entr{'y' if k == 1 else 'ies'} (id {deck.spaced_elems[0]}"
                   + (f" to {deck.spaced_elems[-1]}" if k > 1 else "") + ") separate their node ids by spaces: "
                   f"FEBio 4 reads an element's nodes as one comma-separated list, and FEBio crashes while it "
                   f"reads such a deck (SIGSEGV, nothing printed after 'Reading file ...'; {_MEASURED}). Write "
                   f"them comma-separated.")

    # A NodeSet written with spaces: FEBio keeps the first number between two commas.
    for name, (kept, written) in deck.spaced.items():
        out.append(f"NodeSet '{name}' separates its ids by spaces: FEBio reads the ids between commas "
                   f"and keeps the first number of each piece, so the set holds {kept} of the {written} "
                   f"numbers written ({_MEASURED}; a map on it then stops with "
                   f"'invalid value for attribute \"lid\"').")

    # A per-node table against its node set: lid is the position in that set, 1..N.
    for name, m in deck.maps.items():
        if m["generator"] or not m["lids"] or None in m["lids"]:
            continue
        ns = deck.node_set(m["set"])
        if ns is None or not ns[1]:
            continue
        n = len(ns[0])
        bad = [l for l in m["lids"] if l < 1 or l > n]
        if bad:
            out.append(f"NodeData '{name}' runs lid from {min(m['lids'])} to {max(m['lids'])}, and its node "
                       f"set '{m['set']}' holds {n} nodes: lid is the 1-based position in that set's own "
                       f"list, not a node id, so FEBio stops at the first lid outside 1..{n} with "
                       f"'invalid value for attribute \"lid\"', which names neither the set nor its size.")
            continue
        many = sum(1 for v in m["values"] if len(re.findall(r"[^,\s]+", v)) > 1)
        if m["scalar"] and many:
            out.append(f"NodeData '{name}' is a scalar table and {many} of its entries carry more than one "
                       f"number: FEBio keeps the first number of each, without a word ({_MEASURED}).")
        have = len(set(m["lids"]))
        if have < n:
            out.append(f"NodeData '{name}' gives {have} values for node set '{m['set']}' of {n} nodes: "
                       f"FEBio reads it without a word, and the {n - have} node(s) with no entry take 0 "
                       f"({_MEASURED}).")

    # A map read where it is not defined, or read on another node set.
    for where, set_name, mname in _value_readers(deck):
        m = deck.maps.get(mname)
        if m is None:
            out.append(f"{where} reads map '{mname}', and no NodeData in this deck has that name: FEBio "
                       f"stops with 'std::exception' and nothing else ({_MEASURED}).")
            continue
        if m["generator"] or m["set"] is None or m["set"] == set_name:
            continue
        a, b = deck.node_set(set_name), deck.node_set(m["set"])
        if a is None or b is None or a[0] == b[0]:
            continue                    # an unknown set FEBio names itself; the same list is no defect
        out.append(f"{where} acts on node set '{set_name}' and reads map '{mname}', which is defined on node "
                   f"set '{m['set']}': FEBio takes the map's entries by their position in '{set_name}', not "
                   f"by node, so a node gets the value written for another one, without a word ({_MEASURED}).")

    # A load controller named and not defined.
    missing = defaultdict(list)
    for sec in ("Boundary", "Loads", "Constraints", "Contact", "Rigid", "Initial"):
        for s in root.findall(sec):
            for el in s.iter():
                lc = (el.get("lc") or "").strip()
                if lc and lc not in deck.controllers:
                    parent = next((p for p in s if el in list(p.iter())), el)
                    missing[lc].append(f"{parent.tag} '{parent.get('name') or parent.get('type')}'")
    for lc, wheres in missing.items():
        shown = ", ".join(dict.fromkeys(wheres))
        out.append(f"lc=\"{lc}\" is taken by {shown}, and no <load_controller id=\"{lc}\"> sits in a <LoadData> "
                   f"section: FEBio stops with 'Invalid load curve ID', then 'Model initialization failed', "
                   f"naming neither. Also {_MEASURED}: a <value> with no lc is applied in full at every time.")

    # Held nodes inside the mesh; u_z on part of a slab; one dof held twice with two values.
    holds = _holds(deck)
    bnd = _boundary_nodes(deck)
    if bnd is not None:
        boundary, slab = bnd
        for name, t, set_name, dofs, _ in holds:
            ns = deck.node_set(set_name)
            judged = [d for d in dofs if not (slab and d == "z")]
            if ns is None or not judged:
                continue
            inner = [i for i in dict.fromkeys(ns[0]) if i in deck.nodes and i not in boundary]
            if inner:
                out.append(f"bc '{name}' ({t}) holds {','.join(judged)} on node set '{set_name}', and "
                           f"{len(inner)} of its {len(set(ns[0]))} nodes lie inside the mesh, on no boundary "
                           f"face: FEBio holds them without a word, so they keep the held value instead of "
                           f"the solved one ({_MEASURED}).")
        if slab:
            zheld = set()
            for name, t, set_name, dofs, _ in holds:
                ns = deck.node_set(set_name)
                if ns is not None and "z" in dofs:
                    zheld.update(i for i in ns[0] if i in deck.nodes)
            if 0 < len(zheld) < len(deck.nodes):
                out.append(f"u_z is held on {len(zheld)} of the {len(deck.nodes)} nodes of this "
                           f"one-element-thick slab, so the other {len(deck.nodes) - len(zheld)} move freely "
                           f"in z: the slab is then neither plane strain (u_z = 0 on every node) nor plane "
                           f"stress, and FEBio says nothing ({_MEASURED}: holding u_z on every node instead "
                           f"changed a placeholder slab's field by 3% at v = 0.3 and by 30% at v = 0.45).")
    per = defaultdict(dict)                 # (node, dof) -> {bc name: value}
    scale = 0.0
    for name, t, set_name, dofs, val in holds:
        ns = deck.node_set(set_name)
        if ns is None or val is None:
            continue
        vals = None
        if isinstance(val, tuple):
            m = deck.maps.get(val[1])
            if m is None or m["generator"] or m["set"] != set_name or None in m["lids"]:
                continue
            vals = {}
            for lid, txt in zip(m["lids"], m["values"]):
                x = _num(txt)
                if x is not None and 1 <= lid <= len(ns[0]):
                    vals[ns[0][lid - 1]] = x
        for d in dofs:
            for i in ns[0]:
                v = val if not isinstance(val, tuple) else (vals or {}).get(i)
                if v is None:
                    continue
                per[(i, d)][name] = v
                scale = max(scale, abs(v))
    clash = defaultdict(lambda: [0, None])
    tol = 1e-6 * scale
    for (i, d), got in per.items():
        if len(got) > 1 and max(got.values()) - min(got.values()) > tol:
            c = clash[(d, tuple(sorted(got)))]
            c[0] += 1
    for (d, names), (k, _) in clash.items():
        out.append(f"{k} node(s) are held on dof {d} by bcs {', '.join(repr(n) for n in names)} with different "
                   f"values: FEBio keeps a prescribed displacement's value over a zero displacement's whichever "
                   f"comes first, and of two prescribed displacements the later one, without a word "
                   f"({_MEASURED}).")

    # A body load whose every component is the number 0: FEBio runs it and applies nothing.
    for lo in root.findall("Loads"):
        for bl in lo.findall("body_load"):
            comps = [bl.findtext(t) for t in ("x", "y", "z")] if bl.find("x") is not None else \
                (bl.findtext("force") or "").split(",")
            vals = [_num(c) for c in comps if c is not None]
            if vals and all(v == 0.0 for v in vals):
                out.append(f"body_load '{bl.get('name') or bl.get('type')}' is 0 in every component: FEBio runs "
                           f"it to NORMAL TERMINATION and applies nothing ({_MEASURED}).")

    # Section placement FEBio answers with a message that names something else.
    ctrl = root.find("Control")
    if ctrl is None:
        out.append("the deck has no <Control> section: FEBio reads it, completes 0 time steps and ends with "
                   "E R R O R   T E R M I N A T I O N and no message, and a node log then holds only the step-0 "
                   "block, all zeros (" + _MEASURED + ").")
    elif ctrl.find("solver") is None:
        out.append("<Control> holds no <solver> block: FEBio stops with 'Component \"\" needs to have property "
                   "\"solver\" defined', which names neither the section nor the tag (" + _MEASURED + ").")
    mesh = root.find("Mesh")
    if mesh is not None and mesh.find("MeshData") is not None:
        out.append("<MeshData> sits inside <Mesh>: FEBio stops with 'tag \"MeshData\" (line N) : unrecognized "
                   "tag'. It is a section of its own after <MeshDomains> (" + _MEASURED + ").")
    # A load placed where FEBio does not read one: its message names the tag, not where it belongs.
    _LOADS = ("nodal_load", "surface_load", "body_load")
    for sec in root:
        if sec.tag in ("BodyLoad", "BodyLoads", "Load", "NodalLoad", "NodalLoads"):
            out.append(f"<{sec.tag}> is no section of a FEBio 4 deck: FEBio stops with 'tag \"{sec.tag}\" (line N) : "
                       f"unrecognized tag'. Loads are <nodal_load>, <surface_load> or <body_load> inside one "
                       f"<Loads> section after <Boundary> ({_MEASURED}).")
        elif sec.tag != "Loads":
            for el in sec.iter():
                if el.tag in _LOADS:
                    out.append(f"<{el.tag}> sits inside <{sec.tag}>: FEBio reads loads only inside <Loads>, a "
                               f"section after <Boundary>, and stops elsewhere with 'tag \"{el.tag}\" (line N) : "
                               f"unrecognized tag' ({_MEASURED}).")
                    break
    for bc in root.iter("bc"):
        ty = (bc.get("type") or "").strip()
        if ty.replace(" ", "_") in ("nodal_load", "nodal_force", "body_load", "body_force", "surface_load"):
            out.append(f"<bc type=\"{ty}\">: a <bc> takes a boundary condition type (\"zero displacement\", "
                       f"\"prescribed displacement\"), and FEBio stops with 'invalid value for attribute \"type\"'. "
                       f"A load is a <nodal_load>, <surface_load> or <body_load> inside <Loads> ({_MEASURED}).")
    for lc in root.iter("load_controller"):
        ty = lc.get("type")
        if ty is None or ty.strip().lower() == "linear":
            out.append(f"<load_controller id=\"{lc.get('id')}\"> " + ("has no type" if ty is None else
                       f"has type=\"{ty}\"") + ": FEBio stops with '" + ("missing attribute" if ty is None else
                       "invalid value for attribute") + " \"type\"'. A load curve is type=\"loadcurve\" with "
                       f"<interpolate>LINEAR</interpolate> and its <points> ({_MEASURED}).")
    for lf in root.iter("logfile"):
        if lf.find("elem_data") is not None:
            out.append(f"<elem_data> in <logfile>: FEBio stops with 'tag \"elem_data\" (line N) : unrecognized "
                       f"tag'. The element log is <element_data data=\"...\" file=\"...\"/> ({_MEASURED}).")
            break
    return out


_STOP_BOX = re.compile(r"^\s*\*\s{1,3}(?!E R R O R|ERROR\s|WARNING\s)([^*\n]*?\S)\s*\*\s*$", re.M)


def febio_stop_lines(console: str) -> str:
    """FEBio's own error lines from a console: the text inside the boxes after an ERROR banner."""
    if not isinstance(console, str) or "*" not in console:
        return ""
    out, inside = [], False
    for line in console.splitlines():
        s = line.strip()
        if re.fullmatch(r"\*\s+ERROR\s+\*", s):
            inside = True
            continue
        if inside:
            if set(s) <= {"*"} and s:
                inside = False
                continue
            m = re.fullmatch(r"\*\s+(.*?\S)\s*\*", s)
            if m:
                out.append(m.group(1))
    return " | ".join(dict.fromkeys(out))
