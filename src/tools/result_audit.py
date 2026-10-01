"""audit_results — self-consistency checks on the agent's own output files.

THE MEASURED FAILURE MODE (53 recorded runs with openPASO):
  * 53/53 read the knowledge; the advice channel works.
  * 51/53 execute solvers through run_bash. The verification machinery —
    residual checks in run_simulation, the critic gate, the new unsaved-work
    notice, verify_mesh_independence (used by 0/53) — hangs off tools the
    agents do not use. Every check we built sits on a road they do not drive.
  * Result: 37/54 deliver a complete answer, and most are WRONG in ways visible
    without any answer key: two runs' errors sit FLAT at ~5e-7 across all
    levels (solver tolerance floor — the task even says "converge to 1e-10"),
    one run converges at order 2 on a task that states order 3 (locking: the
    agent WROTE "mixed formulation" in its own header, built the mixed space,
    then assembled the plain form and solved that).

So the failure is not ignorance and not stubbornness: the agent reads, agrees,
writes the plan in its own comments — and nothing in the loop ever makes it
LOOK at whether what it produced matches what it planned. The critic reviews
the SETUP before the run. An independent check reviews the ANSWER after
delivery, against sealed truth. Nothing reviews the RESULT in between, and
the agent cannot see the sealed key, so it cannot check itself against truth
even if it tries.

THIS check needs no truth. It reads only what the agent itself produced —
error/QoI sequences per level — and answers three questions any numerate
reviewer would ask before delivering:
  1. Do successive levels actually approach each other? (self-convergence,
     no exact solution needed)
  2. At what observed order — and does that match the order you are about to
     CLAIM?
  3. Are the differences sitting at a floor (levels nearly identical), which
     means the thing limiting you is a tolerance, not the mesh?

It attaches to the RESULT, not to a run tool, so run_bash cannot bypass it.
"""
from __future__ import annotations

import csv as _csv
import json
import math
import hashlib
import re
from pathlib import Path

# Directories openPASO itself creates. A stale zero-valued probe file in one of
# these once produced a NEAR-ZERO FIELD finding on verified-correct work, and
# only in runs that went through openPASO, which is why the search is filtered
# rather than naive.
_SCRATCH = {"simulation_outputs", "coupling", "meshes", "benchmark_results",
            ".git", "__pycache__", "runs", "runs_quarantine",
            # openPASO's own source layout: a cell once pointed audit_results at the
            # server's src tree and the ladder read core/instructions.py as a
            # participant script written from scratch
            "core", "tools", "backends", "src", "site-packages"}



# ── DELIVERABLE DISCOVERY WITHOUT A NAMING SCHEME ────────────────────────────
#
# A refinement study leaves level-indexed files behind,
# <kind>_level<k>[_<side>].<ext>, and the KIND is whatever the task told the
# agent to call them. Nothing here knows a task's names: the kind is read from
# the agent's own files and each file is classified by what it holds (a
# residual history, an interface trace, a field on probe points, a captured run
# log), so every check below works for any naming a task prescribes.
_LEVEL_FILE = re.compile(
    r"^(?P<kind>[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*?)_level(?P<k>\d+)"
    r"(?:_(?P<side>[A-Za-z0-9]+))?\.(?P<ext>[A-Za-z0-9]+)$")


def _level_files(work: Path, ext: str = "csv") -> list:
    """(path, kind, level, side) for every level-indexed file with that
    extension, outside openPASO's own scratch directories."""
    out = []
    for q in work.rglob(f"*level*.{ext}"):
        if not q.is_file():
            continue
        try:
            if _SCRATCH & set(q.relative_to(work).parts[:-1]):
                continue
        except ValueError:
            continue
        m = _LEVEL_FILE.match(q.name)
        if not m or m.group("ext").lower() != ext.lower():
            continue
        out.append((q, m.group("kind").lower(), int(m.group("k")),
                    m.group("side") or ""))
    return sorted(out, key=lambda t: str(t[0]))


def _is_backup(q: Path) -> bool:
    """A copy the writer or the agent moved aside (name.replaced-<time>, .bak, .orig, ~): never the
    script that runs. Measured: a finding read a backup 18 times in three cells, and a ladder took
    two copies in one side folder for the two participants."""
    n = q.name
    return ".replaced-" in n or n.endswith((".bak", ".orig", "~")) or ".bak." in n or ".orig." in n


def _level_of(q: Path):
    m = _LEVEL_FILE.match(q.name)
    return int(m.group("k")) if m else None


def _side_of(q: Path) -> str:
    m = _LEVEL_FILE.match(q.name)
    return (m.group("side") or "") if m else ""


def _csv_role(q: Path, kind: str | None = None) -> str:
    """'history' | 'interface' | 'field' | 'raw' -- from the kind word first,
    from the file's own header second."""
    if kind is None:
        m = _LEVEL_FILE.match(q.name)
        kind = m.group("kind").lower() if m else ""
    if kind == "field":
        return "raw"      # openPASO's own per-participant dump, not a deliverable
    if any(w in kind for w in ("resid", "hist", "iter", "converg")):
        return "history"
    if any(w in kind for w in ("iface", "interface", "seam", "coupl")):
        return "interface"
    try:
        with open(q, errors="ignore") as fh:
            head = fh.readline().lower()
    except OSError:
        return "field"
    cols = [c.strip().strip('"') for c in head.split(",")]
    if cols and cols[0].startswith("iter"):
        return "history"
    if any(c in ("qn", "q_n", "q", "flux", "normal_flux", "traction", "tn",
                 "t_n") or c.startswith(("q_", "flux", "traction"))
           for c in cols):
        return "interface"
    return "field"


def _iface_flux_names(path) -> list:
    """The flux column names of a per-level interface file (its trailing q*/t*
    run), or [] when the file has no header."""
    try:
        head = path.read_text(errors="replace").splitlines()[0]
    except (OSError, IndexError):
        return []
    hdr = [c.strip() for c in head.split(",")]
    if any(ch.isdigit() for ch in (hdr[0] if hdr else "0")):
        return []
    names = []
    for name in reversed(hdr):
        if name and name[0].lower() in _IFACE_FLUX_PREFIXES and name != "T" and name.lower() != "t":
            names.append(name)
        else:
            break
    return list(reversed(names))


def _paired_iface(ga, gb):
    """(ga', gb', None): two interface file readings (points, values, fluxes) cut to the
    points both list and put in one order -- paired by where the points are
    (quality_checks.pair_interface_points), whatever order or number of rows each file
    has: the same points, one file's points among the other's, or the same points but
    for the interface ends one file leaves out. (None, None, why) where they share no
    such support. MEASURED: a pairing that refused two files of unequal length, or
    paired rows by index, compared a conforming seam's interior against nothing, or
    different points against each other."""
    try:
        qc = _qc()
        cor = qc.interface_correspondence([tuple(p) for p in ga[0]], [tuple(p) for p in gb[0]])
    except Exception as e:                                 # noqa: BLE001
        return None, None, f"their points could not be read ({type(e).__name__})"
    if cor.get("kind") is None:
        return None, None, str(cor.get("why") or "their points could not be read")
    ia, ib = list(cor["ia"]), list(cor["ib"])
    if len(ia) < 2 or (cor["kind"] == "different" and cor["only_a"] and cor["only_b"]):
        return None, None, (f"the two files list different points ({len(ia)} in common, "
                            f"{len(cor['only_a'])} and {len(cor['only_b'])} that only one of them "
                            f"lists inside the other's extent), so they cannot be compared point "
                            f"by point")
    return (tuple([t[i] for i in ia] for t in ga), tuple([t[j] for j in ib] for t in gb), None)


def _per_channel_flux_jumps(ga, gb) -> list:
    """RMS(q_A + q_B) / RMS(q_A, q_B) per flux column, at the interface points
    the two sides share (matched by coordinate). Columns whose scale is at
    round-off are reported as None. [] when fewer than two points match."""
    try:
        (pa, _va, qa), (pb, _vb, qb) = ga, gb
        idx = {tuple(round(float(c), 9) for c in p): j for j, p in enumerate(pb)}
        pairs = [(i, idx[tuple(round(float(c), 9) for c in p)]) for i, p in enumerate(pa)
                 if tuple(round(float(c), 9) for c in p) in idx]
        if len(pairs) < 2 or not qa or not qb:
            return []
        ncomp = min(len(qa[0]), len(qb[0]))
        out = []
        for c in range(ncomp):
            s = [float(qa[i][c]) + float(qb[j][c]) for i, j in pairs]
            sc = [float(qa[i][c]) for i, _ in pairs] + [float(qb[j][c]) for _, j in pairs]
            rms_s = math.sqrt(sum(v * v for v in s) / len(s))
            rms_c = math.sqrt(sum(v * v for v in sc) / len(sc))
            out.append(rms_s / rms_c if rms_c > 1e-14 else None)
        return out
    except Exception:                                      # noqa: BLE001
        return []


# A RELATIVE FLUX JUMP BELOW THIS IS ROUND-OFF AND HAS NO TREND. Measured on a
# correct coupled run: 1.003e-10, 1.170e-08, 2.435e-09 -- the Neumann side's export
# reproducing the flux it was given -- read as "THE FLUX JUMP DOES NOT SHRINK ... the
# WRONG transmission condition", three times, as that run's only finding.
_JUMP_ROUND_OFF = 1e-6


def _mixed_channel_finding(jumps_c: dict, names: list):
    """The finding for flux channels whose jumps part ways under refinement,
    or None. Needs at least three levels with the same channel count."""
    levels = sorted(jumps_c)
    if len(levels) < 3:
        return None
    ncomp = min(len(jumps_c[l]) for l in levels)
    if ncomp < 2:
        return None
    label = lambda c: (names[c] if c < len(names) else f"flux column {c + 1}")
    healthy, bad = [], []
    for c in range(ncomp):
        seq = [jumps_c[l][c] for l in levels]
        if any(v is None for v in seq):
            continue
        if max(seq) < _JUMP_ROUND_OFF:
            continue                     # a channel at round-off has no trend (see below)
        orders = [math.log2(seq[i] / seq[i + 1]) if seq[i + 1] > 0 else float("inf")
                  for i in range(len(seq) - 1)]
        med = sorted(orders)[len(orders) // 2]
        shrinking = all(seq[i + 1] < seq[i] for i in range(len(seq) - 1))
        desc = (label(c), ", ".join(f"{v:.3e}" for v in seq),
                ", ".join(f"{o:.2f}" for o in orders))
        # EVERY step must fall at a healthy order: a channel that fell 3.4x
        # and then 1.08x (measured: a traction with a flipped sign) is not
        # healthy because its median step was, and its last step is where
        # the finest levels disagree.
        if shrinking and min(orders) >= 1.3:
            healthy.append(desc)
        elif (not shrinking) or med < 1.0 or orders[-1] < 0.5:
            bad.append(desc)
    if not healthy or not bad:
        return None
    _bad = "; ".join(f"{n}: {v} (order {o} per halving)" for n, v, o in bad)
    _good = "; ".join(f"{n}: {v} (order {o})" for n, v, o in healthy)
    return {"sequence": "interface flux jump", "priority": 26,
            "values": [float(jumps_c[l][0]) for l in levels],
            "finding": (
        f"THE {' AND '.join(n.upper() for n, _, _ in bad)} "
        f"JUMP{'S DO' if len(bad) > 1 else ' DOES'} NOT SHRINK "
        f"LIKE THE OTHER CHANNEL{'S' if len(healthy) > 1 else ''} ({_bad}), "
        f"while {_good} fall{'s' if len(healthy) == 1 else ''} the way a "
        f"consistent exchange does. A first-order recovery or coarse-mesh "
        f"discretisation error acts on EVERY column a side exports alike, so "
        f"neither is the cause here: the exchange of "
        f"{', '.join(n for n, _, _ in bad)} carries a wrong sign, a wrong "
        f"scaling or a missing term (on a thermo-mechanical interface: the "
        f"thermal-stress part of the traction, the weight that turns a nodal "
        f"reaction into a traction per unit length, a slab thickness, the "
        f"sign of one component). Take ONE level and compare what one side "
        f"EXPORTS for that channel against what the partner APPLIES, point by "
        f"point; do not touch the recovery, and do not hand this in as a "
        f"known limitation -- a channel that does not converge is a wrong "
        f"answer at every level.")}


def _history_files(work: Path) -> list:
    return [q for q, kind, _k, _s in _level_files(work)
            if _csv_role(q, kind) == "history"]


def _field_files(work: Path) -> list:
    return [q for q, kind, _k, _s in _level_files(work)
            if _csv_role(q, kind) == "field"]


def _interface_files(work: Path, sided: bool = False) -> list:
    return [q for q, kind, _k, s in _level_files(work)
            if _csv_role(q, kind) == "interface" and (s or not sided)]


def _level_logs(work: Path, k: int | None = None) -> list:
    return [q for q, _kind, kk, _s in _level_files(work, "log")
            if k is None or kk == k]


# A line stating the number of degrees of freedom, in the spellings solvers
# and their users actually print.
_DOF_LINE = re.compile(
    r"^\s*(?:N_?DOFS?|DOFS?|NUM(?:BER)?_?(?:OF_?)?_?DOFS?|DEGREES OF FREEDOM)"
    r"\s*[=:]\s*(\d+)\s*$", re.I | re.M)

_SUMMARY_HINT: dict = {}


def _summary_file(work: Path, hint=None):
    """The agent's summary/answer file: the caller's hint when it gave one
    (the harness knows which file it just saw written), else the shallowest
    non-empty text file whose name says result, summary, report or answer."""
    hint = hint or _SUMMARY_HINT.get(str(work))
    if hint:
        hp = Path(hint)
        if hp.is_file():
            return hp
    cands = []
    for q in work.rglob("*"):
        if not q.is_file() or q.suffix.lower() not in (".txt", ".md", ""):
            continue
        try:
            rel = q.relative_to(work)
        except ValueError:
            continue
        if _SCRATCH & set(rel.parts[:-1]):
            continue
        if not re.search(r"result|summary|report|answer", q.stem, re.I):
            continue
        try:
            nonempty = q.stat().st_size > 0
        except OSError:
            continue
        cands.append((0 if nonempty else 1, len(rel.parts), q.name.lower(), q))
    return min(cands)[3] if cands else None


def _sequences_from_workdir(work: Path) -> dict[str, list[float]]:
    """Pull per-level scalar sequences out of whatever the agent wrote.

    Looks for the common shapes seen across the recorded runs: summary-file
    fields (ERRORS_UX = a, b, c / L2_ERRORS = ...), per-level csv/json files
    with a recognisable error/qoi column. Returns {label: [level values]}.
    """
    seqs: dict[str, list[float]] = {}
    rt = _summary_file(work)
    if rt is not None and rt.is_file():
        for line in rt.read_text(errors="replace").splitlines():
            m = re.match(r"\s*([A-Za-z0-9_]+)\s*=\s*(.+)$", line)
            if not m:
                continue
            label = m.group(1).upper()
            # Only ERROR-LIKE labels. Scanning every "NAME = a, b, c" line
            # meant the task's own mesh line (H = 0.125, 0.0625, 0.03125)
            # read as an error sequence converging at exactly 1.00 and
            # produced ORDER MISMATCH on correct work, advising the one
            # change every task forbids; and a RESIDUALS line at the 1e-10
            # the task REQUIRES read as a tolerance FLOOR, advising the agent
            # to loosen a tolerance it was told to tighten.
            if not any(k in label for k in ("ERROR", "ERR", "L2", "LINF",
                                            "DIFF", "RESID_ERR")):
                continue
            vals = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(2))
            if len(vals) >= 3:
                try:
                    seq = [float(v) for v in vals]
                    if all(math.isfinite(v) for v in seq):
                        seqs[m.group(1)] = seq
                except ValueError:
                    pass
    return seqs


def _is_participant_dir(d: Path) -> bool:
    """True when this directory is one participant's own work dir.

    That is what makes two same-named per-level files two SIDES rather than a
    collision: each participant runs in its own directory and dumps there.
    """
    try:
        return (d / "config.json").is_file() or any(
            ".replaced-" not in q.name for q in d.glob("participant*.py"))
    except OSError:
        return False


def _sequences_from_level_csvs(work: Path, sources: dict | None = None) -> dict[str, list[float]]:
    """Self-convergence from per-level CSVs on a common probe grid.

    The naming convention a task prescribes fixes ONE probe grid across
    levels, so files like <kind>_level{1,2,3}.csv join on coordinates. With
    no exact solution the per-level 'value' is the max successive difference
    per field: |u_l - u_l+1| shrinking at order p is the self-convergence
    signature of a study converging at order p (Richardson). Also returns each
    field's finest-level magnitude, because a near-zero field is its own
    finding.
    """
    import csv as _csv
    import re as _re

    # ONE GROUP PER (KIND, SIDE), NOT ONE PER LEVEL NUMBER.
    #
    # This globbed *level{i}*.csv and refused to proceed when more than one
    # file matched. On a SINGLE-CODE result set that is right. On a COUPLED one
    # every level has five matches — solution_level1_A, solution_level1_B,
    # interface_level1_A, interface_level1_B, residual_level1 — so the audit
    # reported AMBIGUOUS INPUT and found zero sequences. Every check it exists
    # for (near-zero field, tolerance floor, order, monotonicity) was therefore
    # dead on every coupled problem, which is half the recorded runs and
    # the half that counted for nothing.
    #
    # Measured on one recorded run: the agent delivered three levels of
    # identically zero displacement, the audit said "ambiguous" instead of
    # "your field is zero", and the run reached the independent check, which
    # read it as fabricated, with no run behind it. The gate had the data and
    # did not look at it.
    #
    # The task's naming convention names the sides, so the grouping is not a
    # guess: <kind>_level<k>[_<side>].csv. Ambiguity is still refused, but only
    # when two files claim the SAME (kind, level, side).
    _PAT = _re.compile(r"^(?P<kind>[A-Za-z_]+?)_level(?P<k>\d+)"
                       r"(?:_(?P<side>[A-Za-z0-9]+))?\.csv$")
    # RECURSIVE, MINUS OUR OWN SCRATCH. This globbed the TOP LEVEL only,
    # because rglob once picked openPASO's own directories — benchmark_results/,
    # coupling/, meshes/, simulation_outputs/, all created by openPASO itself and
    # all sorting before solution_*.csv — and a stale zero-valued probe file
    # there produced a NEAR-ZERO FIELD finding on verified-correct work, and
    # only in runs that went through openPASO. The restriction fixed that and
    # introduced a blind spot: one recorded run wrote its 15 files into
    # level1/, level2/, level3/, and the audit found ZERO sequences and
    # returned clean=True on a full result set. Name the directories to skip
    # instead of refusing to descend at all.
    groups: dict[tuple, dict[int, list]] = {}
    resid_files: list = []
    for q in work.rglob("*level*.csv"):
        if not q.is_file():
            continue
        if _SCRATCH & set(q.relative_to(work).parts[:-1]):
            continue
        m = _PAT.match(q.name)
        if not m:
            continue
        kind = m.group("kind").lower()
        if kind == "field":
            # openPASO's own per-participant raw dump (field_level<k>.csv, one per
            # side's work dir), not a deliverable. Two sides writing the same
            # name at the same depth read as AMBIGUOUS INPUT on verified-correct
            # work (measured on a real coupled rebuild) -- so it is not a
            # sequence.
            continue
        if _csv_role(q, kind) == "history":
            # NOT SKIPPED ANY MORE — see _residual_findings below. It is not a
            # field on a grid, so it does not join the per-level sequences, but
            # it is the single file that decides a third of coupled outcomes
            # and the audit used to look straight past it.
            resid_files.append(q)
            continue
        key = (kind, (m.group("side") or "").upper())
        groups.setdefault(key, {}).setdefault(int(m.group("k")), []).append(q)

    # SHALLOWEST WINS, and only a TIE is ambiguous.
    #
    # The deliverable belongs in the work dir; a copy deeper down is a
    # byproduct. Every deal.II run keeps one — cmake builds in build/ and the
    # solver writes its CSVs beside the binary — so descending made three
    # verified-correct runs report AMBIGUOUS INPUT, which is exactly the false
    # alarm the old top-level-only rule was protecting against. Depth decides
    # it without having to enumerate every scratch directory a backend might
    # invent; genuine ambiguity (two files at the SAME depth for one slot) is
    # still refused.
    dup: list = []
    split: dict[tuple, dict[int, list]] = {}   # side recovered from the directory
    drop: list = []
    for k, byl in groups.items():
        for lv, qs in list(byl.items()):
            if len(qs) == 1:
                continue
            depth = {q: len(q.relative_to(work).parts) for q in qs}
            shallowest = min(depth.values())
            keep = [q for q in qs if depth[q] == shallowest]
            if len(keep) > 1:
                # THE SIDE IS IN THE DIRECTORY WHEN IT IS NOT IN THE NAME.
                #
                # The served per-level dump writes interface_level<k>.csv into
                # each participant's OWN work dir, so a two-sided run has two
                # files with one name at one depth and they tie here. That is
                # not a collision, it is the normal layout -- `field` was
                # already skipped for this exact reason, and `interface` was
                # not, so a correct coupled set was refused.
                #
                # MEASURED, and it cost a whole cell: one run was told to
                # "remove the stale ones", deleted a side's files, and the next
                # audit found ZERO sequences -- so the divergence in its own
                # ladder (self-differences growing five-fold per level) was
                # never reported and the run submitted a diverging answer as
                # final. Split by the directory instead of refusing, which
                # recovers both sides rather than discarding both.
                by_dir = {q.parent: q for q in keep}
                if len(by_dir) == len(keep) and all(
                        _is_participant_dir(d) for d in by_dir):
                    for d, q in by_dir.items():
                        split.setdefault((k[0], d.name.upper()), {})[lv] = [q]
                    drop.append((k, lv))
                    continue
                dup.append(f"{k[0]}{'_' + k[1] if k[1] else ''} level {lv}: "
                           f"{sorted(str(x.relative_to(work)) for x in keep)}")
            byl[lv] = keep
    for k, lv in drop:                       # the tied slot became one slot per side
        groups[k].pop(lv, None)
    for k, byl in split.items():
        groups.setdefault(k, {}).update(byl)
    groups = {k: v for k, v in groups.items() if v}
    if dup:
        return {"__ambiguous__": dup}

    out_all: dict[str, list[float]] = {}
    for key in sorted(groups):
        seq = _one_sequence(groups[key], key, _csv, sources)
        out_all.update(seq)
    return out_all


def _one_sequence(by_level: dict, key: tuple, _csv, sources: dict | None = None) -> dict[str, list[float]]:
    """The original per-level analysis, for ONE (kind, side) group. `sources`, when given,
    receives the finest-level file each tag's magnitudes were read from."""
    kind, side = key
    tag = f"{kind}_{side}" if side else kind
    levels: list[dict] = []
    for i in range(1, 9):
        # TOP LEVEL ONLY. rglob + sorted(cands)[0] picked the
        # lexicographically first PATH, so openPASO's own scratch directories —
        # benchmark_results/, coupling/, meshes/, simulation_outputs/, all
        # created by openPASO itself and all sorting before solution_*.csv — won
        # over the agent's real output. A stale zero-valued probe file left by
        # a failed first run then produced a NEAR-ZERO FIELD finding on
        # verified-correct work, and only in runs that went through openPASO.
        cands = by_level.get(i) or []
        if not cands:
            break
        rows = {}
        try:
            with open(sorted(cands)[0]) as fh:
                r = _csv.DictReader(fh)
                # Headers arrive as "x, y, u" — WITH spaces. Unstripped, " y"
                # passed the coordinate filter and became a data field, and
                # the join key collapsed to (x, 0) for every row, comparing
                # unrelated rows between levels. That single character made
                # 10 of 22 verified-correct runs flag as ORDER MISMATCH.
                norm = {(c or "").strip(): c for c in (r.fieldnames or [])}
                fields = [k for k in norm
                          if k and k.lower() not in ("x", "y", "z")]
                if "x" not in norm or "y" not in norm:
                    break                     # no coordinates: cannot join
                # z joins the key when present: with only (x, y), every 3D
                # column of points collapses onto one key and the three 3D
                # problems false-alarmed exactly like the whitespace bug.
                zc = norm.get("z")
                for row in r:
                    key = (round(float(row[norm["x"]]), 9),
                           round(float(row[norm["y"]]), 9),
                           round(float(row[zc]), 9) if zc else 0.0)
                    rows[key] = {f: float(row[norm[f]]) for f in fields
                                 if row.get(norm[f]) not in (None, "")}
        except (OSError, ValueError):
            break
        if rows:
            levels.append(rows)
            if sources is not None:
                sources[tag] = sorted(cands)[0]
    if not levels:
        return {}
    # A ZERO FIELD IS VISIBLE AT LEVEL ONE, AND THAT IS WHEN IT IS WORTH
    # SAYING. This returned {} below three levels, so the NEAR-ZERO check —
    # the cheapest catch in the audit and the one that names an unwired load —
    # stayed silent exactly while the agent still had the budget to fix it.
    # Self-convergence genuinely needs three levels; a magnitude does not need
    # any. Emit the magnitude from whatever exists and the differences only
    # when there are enough levels to form them.
    if len(levels) < 3:
        out: dict[str, list[float]] = {}
        fields = sorted({f for lv in levels for v in lv.values() for f in v})
        for f in fields:
            mag = max((abs(v[f]) for v in levels[-1].values() if f in v),
                      default=0.0)
            out[f"magnitude_{tag}_{f}"] = [mag]
            _vf = [v[f] for v in levels[-1].values() if f in v]
            if _vf and mag > 0:
                out[f"spread_{tag}_{f}"] = [(max(_vf) - min(_vf)) / mag]
        return out
    out: dict[str, list[float]] = {}
    fields = sorted({f for lv in levels for v in lv.values() for f in v})
    for f in fields:
        diffs = []
        for a, b in zip(levels, levels[1:]):
            common = [k for k in a if k in b and f in a[k] and f in b[k]]
            if len(common) < 4:
                diffs = []
                break
            # RMS, not max: over ~2000 probe points the max difference is
            # dominated by a single worst point and its decay is noisy; the
            # RMS decays at the field's true self-convergence rate. The max
            # variant mis-flagged a verified-correct run.
            import math as _m
            diffs.append(_m.sqrt(sum((a[k][f] - b[k][f]) ** 2
                                     for k in common) / len(common)))
        if len(diffs) >= 2 and all(d > 0 for d in diffs):
            out[f"selfdiff_{tag}_{f}"] = diffs
        mag = max((abs(v[f]) for v in levels[-1].values() if f in v),
                  default=0.0)
        out[f"magnitude_{tag}_{f}"] = [mag]
        _vf = [v[f] for v in levels[-1].values() if f in v]
        if _vf and mag > 0:
            out[f"spread_{tag}_{f}"] = [(max(_vf) - min(_vf)) / mag]
    return out



def _uniform_finding(work: Path, label: str, seq: list, sources: dict, col: str, tag: str,
                     mag: float) -> dict | None:
    """The UNIFORM FIELD finding for a column that holds one value at every probe point, or None.

    AN INTERFACE TRACE IS NOT THE FIELD. Measured on a coupled round: a side whose temperature
    varies only across the gap to its partner exports a trace and a flux that are uniform along
    the interface -- the right answer when nothing varies along it -- and UNIFORM FIELD led the
    hand-in screen of three right sides. An interface column is judged by the side's own field
    dump: a field that varies is not a constant, whatever its trace along one line does. A side
    that imported nothing is said to have imported nothing, as the NEAR-ZERO finding does."""
    finding = (f"UNIFORM FIELD: {col} on {tag} holds ONE value ({mag:.3e}) at "
               f"every probe point of the finest level. That is not a solution "
               f"but a constant -- a solve that no Dirichlet condition reached "
               f"(uniform, and usually astronomical), or a column filled from one "
               f"number. Nothing computed from this column is evidence; find the "
               f"run that produced it before anything else.")
    src = sources.get(tag)
    if src and Path(src).name.lower().startswith("interface"):
        side = Path(src).parent
        if not _is_participant_dir(side):              # a delivered file: the side its name carries
            want = tag.rpartition("_")[2].upper() if "_" in tag else ""
            hits = [d for d in _side_dirs(work) if want and
                    (d.name.upper() == want or d.name.upper().endswith("_" + want))]
            side = hits[0] if len(hits) == 1 else None
        dumps = sorted(side.glob("field_level*.csv"), key=lambda q: q.stat().st_mtime) if side else []
        vals = []
        if dumps:
            try:
                h, rows = _read_named_csv(dumps[-1])
                vc = next((i for i, c in enumerate(h) if c not in ("x", "y", "z")), None)
                vals = [r[vc] for r in rows] if vc is not None else []
            except (OSError, ValueError, IndexError):
                vals = []
        peak = max((abs(v) for v in vals), default=0.0)
        if vals and peak > 0 and max(vals) - min(vals) > 1e-9 * peak:
            return None                      # the side's field varies; its trace along one line may not
        none_in = side is not None and not (side / "imports.json").is_file()
        if not vals:
            return {"sequence": label, "values": seq, "informational": True, "finding": (
                f"UNIFORM INTERFACE COLUMN NOT JUDGED: {col} on {tag} holds one value ({mag:.3e}) along "
                f"the interface, which is right where nothing varies along it, and no field_level<k>.csv "
                f"of that side holds its field to tell."
                + (" That side has no imports.json either." if none_in else ""))}
        if none_in:
            finding += (" This side has no imports.json: a side that imported nothing holds its "
                        "first-iteration fallback on the interface, so that column can be the "
                        "fallback itself.")
    return {"sequence": label, "values": seq, "priority": 6, "finding": finding}


def _near_zero_finding(work: Path, label: str, seq: list, sources: dict) -> dict | None:
    """The NEAR-ZERO FIELD finding for one magnitude below 1e-8, naming the file and column it
    read. AN INTERFACE COLUMN BESIDE A ZERO IMPORT IS NOT A FIELD. Measured: a Dirichlet side run
    on an all-zero synthetic import wrote that import as its interface trace; its field peaked
    well above 1e-8 and four give-up screens led with "NEAR-ZERO FIELD". Where every value the
    side imported is below 1e-8 too, the column can be the import itself, and the side's own
    field dump beside it is judged instead (none there: the check says it did not judge)."""
    tag = next((t for t in sorted(sources, key=len, reverse=True) if label.startswith(f"magnitude_{t}_")), None)
    src = sources.get(tag) if tag else None
    col = label[len(f"magnitude_{tag}_"):] if tag else ""
    try:
        rel = str(Path(src).relative_to(work)) if src else ""
    except ValueError:
        rel = Path(src).name
    tail = ("This measures the field, not why it is small. On a driven "
            "problem check that the source reaches the solve (in the form, "
            "the load vector or the deck, with a deck's load curve active) "
            "and that the boundary values are not all zero, before reading "
            "it as a small answer.")
    imp = Path(src).parent / "imports.json" if src else None
    # NOR BESIDE NO IMPORT AT ALL. Measured: a Dirichlet side run with no imports.json held the
    # wrapper's first-iteration fallback on its interface; the column was 0.0 and a give-up screen
    # led with "NEAR-ZERO FIELD", which the run read as "the solution is wrong".
    if src and Path(src).name.lower().startswith("interface"):
        if imp.is_file():
            try:
                nums = [abs(float(x)) for blk in (json.loads(imp.read_text() or "{}") or {}).values()
                        if isinstance(blk, dict) for key in ("values", "normal_fluxes")
                        for x in _flat(blk.get(key) or [])]
            except (OSError, ValueError, TypeError, AttributeError):
                nums = None
            echo = nums is not None and max(nums, default=0.0) < 1e-8
            why = ("and so is every value this side imported (its imports.json): that column can be the "
                   "import itself")
        else:
            echo = True
            why = ("and this side has no imports.json: a side that imported nothing holds its fallback on "
                   "the interface (the served contracts' first-iteration value), so that column can be the "
                   "fallback itself")
    else:
        echo = False
    if echo:
        dumps = sorted(Path(src).parent.glob("field_level*.csv"), key=lambda q: q.stat().st_mtime)
        if not dumps:
            return {"sequence": label, "values": seq, "informational": True, "finding": (
                f"NEAR-ZERO FIELD NOT JUDGED: {rel} column `{col}` is below 1e-8, {why}, and no "
                f"field_level<k>.csv beside it holds the field to read instead.")}
        try:
            h, rows = _read_named_csv(dumps[-1])
            vc = [i for i, c in enumerate(h) if c not in ("x", "y", "z")]
            peak = max((abs(r[i]) for r in rows for i in vc), default=None)
        except (OSError, ValueError):
            peak = None
        if peak is None or peak >= 1e-8:
            return None                  # the field is not near zero; the column echoes the import or fallback
        try:
            rel = str(dumps[-1].relative_to(work))
        except ValueError:
            rel = dumps[-1].name
        return {"sequence": label, "values": [peak], "finding": (
            f"NEAR-ZERO FIELD: {rel} peaks at {peak:.1e}, below 1e-8. " + tail)}
    where = f"{rel} column `{col}` peaks at {seq[0]:.1e}, below 1e-8" if rel else "the finest-level field peaks below 1e-8"
    return {"sequence": label, "values": seq, "finding": f"NEAR-ZERO FIELD: {where}. " + tail}


def _flat(x):
    """The numbers of a nested list, flattened."""
    if isinstance(x, (list, tuple)):
        for y in x:
            yield from _flat(y)
    else:
        yield x


def _line_count(q) -> int:
    """Lines in a text file, the handle closed (it was left open: ResourceWarning)."""
    with open(q, errors="ignore") as fh:
        return sum(1 for _ in fh)


def _last_column(q) -> tuple:
    with open(q) as fh:
        return tuple(r[-1].strip() for r in _csv.reader(fh) if r)


def _diverged_sides(work: Path, limit: float = 1e30) -> list:
    """(side, largest exported magnitude) for every side whose exports.json holds
    a value or flux beyond `limit` -- data no physical coupling of these problems
    reaches, and what an expanding fixed-point map leaves after many iterations."""
    out = []
    try:
        import numpy as _np
    except Exception:                                    # noqa: BLE001
        return out
    for d in _side_dirs(work):
        try:
            e = json.loads((d / "exports.json").read_text() or "{}")
            arr = []
            for key in ("values", "normal_fluxes"):
                a = _np.asarray(e.get(key) or [], float).ravel()
                arr.extend(a[_np.isfinite(a)].tolist())
            m = max((abs(v) for v in arr), default=0.0)
        except Exception:                                # noqa: BLE001
            continue
        if m > limit:
            out.append((d.name, m))
    return out


def diverged_exports_findings(work: Path) -> list[dict]:
    """A coupling whose exported data grew without bound is named as diverging.

    Measured: a coupling whose fields reached 1e92 and 1e99 kept a relative
    residual pinned near one, because both sides grew together, and was told its
    residual "BARELY MOVED ... IT IS NOT SLOW, IT IS WANDERING". The growth itself
    is on disk, in each side's exports.json."""
    div = _diverged_sides(work)
    if not div:
        return []
    return [{"sequence": "diverged coupling", "priority": 9, "values": [m for _n, m in div], "finding": (
        "THE COUPLING DIVERGED: " + "; ".join(f"{n} exports data of magnitude up to {m:.2g}" for n, m in div)
        + ". A relative residual cannot show this: when both sides grow together it stays level. An "
        "iteration whose data grow without bound has a fixed-point map that expands instead of "
        "contracting; the relaxation factor and the sign and scaling of each exchanged datum decide "
        "that, and nothing here measured which. More iterations cannot fix it. Take the first few "
        "iterations: data that grow by a constant factor each step are that expansion.")}]


def residual_findings(work: Path) -> list[dict]:
    """What the coupling residual history says about itself.

    THE FILE THE AUDIT USED TO SKIP. The residual history is not a field on a
    grid, so it never joined the per-level sequences — and it is the single
    file that decides the largest failure bucket on coupled problems:
    coupling evidence that contradicts itself accounts for 69 of 250 coupled
    recorded runs checked (28%) — the second-largest reason after an
    outright give-up. Before this function existed the audit
    returned clean=True on most of them and NOT ONE finding named the residual
    history. (An earlier version of this note said "80 of 250" and "53 of 91";
    neither denominator is reconstructible from the tree and both are
    withdrawn in favour of the counts above, which are.) The agent was told its work was self-consistent while
    three levels carried non-finite residuals.

    Every check here is computable from the agent's own files and needs no
    reference solution: enough iterations to be an iteration, positive and
    finite, an actual decrease, and a history that is not a constant column.
    They mirror src/blind_eval/evidence.py::coupling_evidence, which is what
    an independent check applies afterwards — so a finding here is a warning
    about an outcome the agent would otherwise meet for the first time after
    delivery.
    """
    import csv as _csv
    out: list[dict] = []
    for q in _history_files(work):
        vals: list[float] = []
        try:
            with open(q) as fh:
                for row in _csv.reader(fh):
                    if not row:
                        continue
                    try:
                        vals.append(float(row[-1]))
                    except ValueError:
                        continue            # header
        except OSError:
            continue
        name = q.name
        # A LEADING NaN IS openPASO'S OWN history[0], NOT THE AGENT'S DEFECT.
        #
        # `couple` returns a history whose first entry is NaN by construction —
        # there is no previous export to compare the first one against. The
        # independent check knows this and drops it (blind_eval.evidence
        # records `dropped_leading_nonfinite`), but this audit, which is the
        # gate we tell agents to run BEFORE delivering, did not.
        #
        # Measured on one recorded run — the best coupled run of that set,
        # with 4C and Kratos both proven to have run and the coupling proven
        # and not forged, with the residual falling 0.309 -> 8.2e-07 — this
        # audit returned
        # "clean": false and told it, three times, that "the residual was never
        # actually computed from the two sides". openPASO produced the NaN, then
        # reported it to the agent as evidence of the agent's own failure, and
        # the only fix available to an agent that believes it is to go and
        # break something that was right.
        #
        # Dropped, not tolerated: a NaN anywhere LATER in the history is still
        # a real finding, and so is a non-positive value anywhere at all.
        leading_nan = bool(vals) and vals[0] != vals[0]
        if leading_nan:
            vals = vals[1:]
        if len(vals) < 3:
            out.append({"sequence": name, "values": vals,
                        "finding": (
                            f"COUPLING HISTORY TOO SHORT: {len(vals)} "
                            f"iteration(s). A partitioned scheme that reached "
                            f"a fixed point leaves a history; fewer than three "
                            f"entries is read as not having coupled.")})
            continue
        if any((v != v) or v in (float('inf'), float('-inf')) or v < 0
               for v in vals):
            out.append({"sequence": name, "values": vals[:6],
                        "finding": (
                            "NON-POSITIVE OR NON-FINITE RESIDUAL: a relative "
                            "interface mismatch is a positive number. A "
                            "negative or a nan here means the residual was "
                            "never actually computed from the two sides.")})
            continue
        # AN EXACT ZERO IS COMPUTED, AND SAYS SOMETHING ELSE. It was lumped with the
        # negative and NaN rows as "never actually computed", and the give-up advice
        # told a run to delete, as invented, histories couple_levels had written from a
        # real iteration whose last step was exactly zero (measured: the partner
        # exported the same placeholder flux every time).
        _z = [i + 1 for i, v in enumerate(vals) if v == 0.0]
        if _z:
            out.append({"sequence": name, "values": vals[:6],
                        "finding": (
                            f"A RESIDUAL OF EXACTLY ZERO at row(s) {', '.join(str(i) for i in _z[:4])}: "
                            f"the data the two sides exchanged were bit-identical between two "
                            f"iterations. Two independently solved sides reach a fixed point to "
                            f"the last bit only when what they exchange stopped changing -- look "
                            f"at each side's exports for a column that never moves (a placeholder, "
                            f"a copy, an answer re-served).")})
            continue
        if vals[0] / max(vals[-1], 1e-300) < 10.0:
            # A STALLED ITERATION AND A WANDERING ONE HAVE DIFFERENT CAUSES,
            # AND ONLY ONE OF THEM IS THE RELAXATION.
            #
            # Slow monotone decay is under-relaxation: the sequence goes down
            # every step and simply needs more steps or a better theta. A
            # sequence that goes UP about as often as it goes DOWN is not
            # converging slowly, it is not converging at all, and more
            # iterations and a smaller theta cannot fix it -- each side is
            # undoing the other.
            #
            # MEASURED on one coupled run whose own critic had computed the
            # conductance ratio correctly and concluded, rightly, that the
            # default relaxation should converge in a few steps: 499
            # iterations, 244 steps up and 254 down, wandering between 0.24 and
            # 1.26 about a mean of 0.65, 68% of consecutive steps reversing
            # direction. It spent its whole budget raising max_iter to 200 and
            # then switching accelerator, which is what this finding used to
            # leave a reader to guess.
            ups = sum(1 for a, b in zip(vals, vals[1:]) if b > a)
            downs = len(vals) - 1 - ups
            wandering = (len(vals) >= 8 and ups >= 0.35 * (len(vals) - 1)
                         and not _diverged_sides(work))      # a divergence is named on its own
            why = ""
            if wandering:
                flips = sum(1 for a, b, c in zip(vals, vals[1:], vals[2:])
                            if (b - a) * (c - b) < 0)
                why = (
                    f" IT IS NOT SLOW, IT IS WANDERING: {ups} step(s) up against "
                    f"{downs} down over {len(vals)} iterations"
                    + (f", {flips / max(len(vals) - 2, 1):.0%} of them reversing "
                       f"direction" if len(vals) > 2 else "")
                    + ". Under-relaxation makes a sequence go DOWN every step and "
                    "only slowly, so this is not the relaxation and neither more "
                    "iterations nor a smaller theta will fix it. MEASURE FIRST, on "
                    "each side: run its participant twice on the SAME imports.json "
                    "and compare the two exports.json -- a finite-element solve "
                    "repeats bit for bit, and a side that does not (a mask or "
                    "vector allocated and only partly written, a list where a "
                    "full-length array belongs) puts a floor under the residual "
                    "that no iteration beats; measured, that was the cause of every "
                    "wandering coupling of one round. If both repeat, check the "
                    "exchange: each side exports its OWN outward normal flux (the "
                    "two report opposite signs at the same point), and both "
                    "exchange the same interface points in the same order.")
            out.append({"sequence": name, "values": [vals[0], vals[-1]],
                        "finding": (
                            f"RESIDUAL BARELY MOVED: {vals[0]:.3g} -> "
                            f"{vals[-1]:.3g}, a factor of "
                            f"{vals[0] / max(vals[-1], 1e-300):.2g}. That is "
                            f"not a converged coupling; it is the iteration "
                            f"standing still, and it is read as not "
                            f"coupled." + why)})
            continue
        if len(set(f"{v:.12g}" for v in vals)) == 1:
            out.append({"sequence": name, "values": vals[:4],
                        "finding": (
                            "CONSTANT RESIDUAL COLUMN: every iteration reports "
                            "the same number, so the column is a placeholder "
                            "rather than a measured mismatch.")})
            continue
        # THE AUDIT PASSED A FORGERY. Measured on one recorded run, whose
        # three levels each held 1.0 falling to exactly 1e-06 in ten steps,
        # BIT-IDENTICAL across all three, while its own NDOF lines said the
        # mesh had changed (400, 255, 72). An independent check labels that
        # fabricated, with no run behind it. This audit — the gate we tell
        # agents to run before delivering — returned "clean": true.
        #
        # A gate that blesses an invented history is worse than no gate: it
        # tells an agent the shortcut passed. The two rules below are already
        # PUBLIC — the served coupling must-read states both, in as many words,
        # so nothing is revealed by checking them here. They are imported from
        # the shared evidence module rather than restated, so the thresholds
        # cannot drift apart from the ones an independent check actually
        # applies.
        try:
            from blind_eval.evidence import (          # noqa: PLC0415
                _decay_ratio_cv, SYNTHETIC_RATIO_CV, _FORGED_DECAY_MIN_DROP)
        except Exception:                               # module not importable
            continue
        cv = _decay_ratio_cv(vals)
        if (cv is not None and cv < SYNTHETIC_RATIO_CV
                and vals[0] / max(vals[-1], 1e-300) > _FORGED_DECAY_MIN_DROP):
            out.append({"sequence": name, "values": vals[:5],
                        "finding": (
                            f"THIS READS AS A WRITTEN-IN SEQUENCE, NOT A "
                            f"MEASURED ONE: the step-to-step ratio is constant "
                            f"to a coefficient of variation of {cv:.1e}. A real "
                            f"partitioned iteration's rate wanders as the "
                            f"error's modal composition changes. This is read "
                            f"as fabrication, which counts for less than an "
                            f"honest report that the iteration did not "
                            f"converge. THE "
                            f"HONEST FILE IS CHEAPER THAN THIS ONE: your "
                            f"coupling loop already computes an interface "
                            f"mismatch every iteration to decide when to stop "
                            f"-- append THAT number to the CSV inside the loop, "
                            f"one line, and the history is real whatever it "
                            f"shows. If your loop never computed a mismatch, "
                            f"it never coupled, and the honest entry is a "
                            f"could-not-finish report plus your best "
                            f"single-domain fields, which is worth more than "
                            f"this file.")})
    # BIT-IDENTICAL HISTORIES ACROSS LEVELS — checked across files, not within.
    #
    # The per-file loop above cannot see it: each level's column is individually
    # unremarkable. The history depends on the discretisation, so the same
    # numbers at two mesh levels cannot both be measurements.
    seqs: dict[str, list[str]] = {}
    for q in _history_files(work):
        try:
            body = _last_column(q)
        except OSError:
            continue
        if len(body) >= 3:
            # KEY BY LEVEL NUMBER, NOT BY FILE NAME.
            #
            # Agents write their outputs twice: once at the contractual path and
            # once under a per-level directory of their own. This loop walked
            # both copies, so a file was compared with ITSELF and the run was
            # told "IDENTICAL RESIDUAL HISTORY AT 2 MESH LEVELS:
            # residual_level1.csv, residual_level1.csv agree digit for digit"
            # — and at three copies, "AT 3 MESH LEVELS" naming level 1 three
            # times.
            #
            # Measured on one recorded run: EIGHT findings, every one of them
            # a copy paired with itself, on a run whose field is within 3% of
            # the true solution on both subdomains. A fabrication accusation is
            # the most damaging thing this file can say, and it was saying it
            # about tidy output habits.
            _lv = _level_of(q)
            lvl = str(_lv) if _lv is not None else q.name
            seqs.setdefault(repr(body), []).append((lvl, q.name))
    for _, hits in seqs.items():
        levels = sorted({lvl for lvl, _ in hits})
        if len(levels) > 1:
            names = sorted({name for _, name in hits})
            out.append({"sequence": ", ".join(names), "values": [],
                        "finding": (
                            f"IDENTICAL RESIDUAL HISTORY AT {len(levels)} MESH "
                            f"LEVELS (levels {', '.join(levels)}): "
                            f"{', '.join(names)} agree digit "
                            f"for digit. The history depends on the "
                            f"discretisation, so these cannot both be "
                            f"measurements; this reads as an invented history.")})
    return out


def _dof_line_detail(work: Path, k: int, side: str, have: list, logs: list) -> str:
    """Name the file to edit, and quote the number if the run already printed it.

    MEASURED on a coupled run that produced a complete and correct result set
    and was graded malformed because one of its six logs lacked this line. The
    audit found it and said so twice, in the FIRST key of the reply -- and said
    it as `level/side ['1B']`, which is a coordinate in our head and not a
    filename on their disk. The number it needed was already on disk too, in the
    console this very driver captured at that level:

        side_B/participant_output_level1.log:  NDOF = 72

    Both halves were held and neither was published. Naming them is not serving
    an answer: it is the agent's own dof count, from the agent's own run, in a
    file the agent owns.

    Nothing is invented. Where no log of that level carries a count, the finding
    names the file and stops -- a guessed number would be worse than none, since
    the log would then testify to a mesh nobody solved.
    """
    # BY PATH, NOT BY NAME. The served layout puts a copy of each side's log
    # inside that side's own directory, so a bare name renders the two as
    # "run_level1_B.log, run_level1_B.log" and reads as a stutter rather than
    # as two files -- measured on the run this check was written for.
    def _rel(q):
        try:
            return str(q.relative_to(work))
        except ValueError:
            return q.name
    _paths = sorted(_rel(q) for q in have)
    names = ", ".join(_paths) or "no log at all"
    carries = "carry" if len(_paths) > 1 else "carries"
    for q in logs:
        if q in have:
            continue
        try:
            rel = q.relative_to(work)
        except ValueError:
            continue
        if side:
            low = side.lower()
            folders = [part.lower() for part in rel.parts[:-1]]
            if not (any(part in (low, f"side_{low}") or part.endswith(f"_{low}")
                        for part in folders)
                    or rel.name.lower().endswith(f"_{low}.log")):
                continue
        found = _DOF_LINE.search(q.read_text(errors="ignore"))
        if found:
            return (f"level {k} side {side}: {names} {carries} no dof line, but "
                    f"this run DID print one at that level -- `NDOF = "
                    f"{found.group(1)}` in {rel}. Put that console in the log, "
                    f"or add the line to it.")
    return (f"level {k} side {side}: {names} {carries} no dof line, and no other "
            f"log of level {k} carries one either, so the count has to come "
            f"from that side's own solver console.")


_LOOKS_PER_LEVEL = re.compile(r"level", re.I)
_DELIVERABLE_STEMS = ("solution", "interface", "field", "residual", "run", "history")


# A SOLVER'S OWN STOP LINE, PER CODE, MEASURED ON THIS INSTALL. Kept per code
# and narrow on purpose: a code that prints "Error" and recovers would be
# accused by anything looser, and the reach of this table was measured before
# it shipped -- of 347 coupled cells of one problem, 72 carry per-level
# consoles, 3 carry one of these lines, and the only two where the wrapper
# exported anyway are the two that failed. None of the six correct cells of
# the last two rounds carries one.
_SOLVER_STOP_LINES = (
    ("4C", ("PROC 0 ERROR", "MPI_ABORT was invoked", "Error!!!")),
    ("Kratos", ("Error zero in diagonal", "Error zero sum", "Kratos::Exception",
                "terminate called after throwing")),
    ("PETSc/FEniCS", ("DIVERGED_", "PETSC ERROR")),
    ("deal.II", ("An error occurred in line", "ExcMessage")),
)


def solver_stopped_findings(work: Path) -> list[dict]:
    """The solver inside a participant stopped with its own error, and the script exported anyway.

    MEASURED on one coupled run. Its rewritten 4C participant put the two
    interface corner nodes under two point-Dirichlet lists at once; 4C refused
    the deck at every level --

        PROC 0 ERROR in .../4C_fem_discretization_utils_dbc.cpp, line 385:
        Error!!! Inconsistency is detected at POINT DBC 20 (node 6, dof 0).
        MPI_ABORT was invoked on rank 0

    -- and the script, which never checked the return code and whose VTU regex
    never matched, exported a field of zeros. One run's Kratos participant
    built a clockwise triangle, so half its elements had negative area; Kratos
    printed `LUSkylineFactorization::factorize: Error zero in diagonal` in every
    per-level console, and the script exported a flux that grew 10x, 48x, 203x
    under refinement. Both cells came back with both codes PROVEN and the
    coupling PROVEN; both had already failed inside the solver.

    NOTHING READ THOSE LINES, because the wrapper exited 0. The existing deck
    reader extracts exactly this text -- but only on a non-zero exit or an
    unrun side. A stop line in a console that also has an export beside it is
    the strongest single fact in the directory, and it outranks every finding
    computed from the export, because the export is not that solver's answer.

    The line is the agent's own console, quoted back. Nothing is inferred
    about what the solver would have produced.
    """
    out: list[dict] = []
    for d in sorted(p for p in work.iterdir() if p.is_dir()):
        for log in sorted(d.glob("participant_output_level*.log")):
            m = re.search(r"level(\d+)", log.name)
            lvl = int(m.group(1)) if m else None
            exported = ((d / f"field_level{lvl}.csv").is_file() if lvl else False) \
                or (d / "exports.json").is_file()
            if not exported:
                continue
            try:
                text = log.read_text(errors="replace")
            except OSError:
                continue
            hits = []
            code_name = ""
            for code, needles in _SOLVER_STOP_LINES:
                for line in text.splitlines():
                    if any(n in line for n in needles):
                        hits.append(line.strip()[:160])
                        code_name = code
                if hits:
                    break
            if not hits:
                continue
            quoted = "\n    ".join(hits[:3])
            out.append({"sequence": f"solver stopped {d.name} level {lvl}",
                        "priority": 7, "values": [], "finding": (
                f"THE SOLVER INSIDE {d.name} STOPPED WITH ITS OWN ERROR AT LEVEL "
                f"{lvl} AND THE SCRIPT EXPORTED ANYWAY. Its console "
                f"({log.relative_to(work)}) carries {code_name}'s own stop line"
                + ("s" if len(hits) > 1 else "") + ":\n    " + quoted + "\n"
                f"Whatever the script wrote after that is not that solver's "
                f"solution of this input: a zero field, a stale file, or a "
                f"number derived from a system the solver refused. Every "
                f"finding computed from this side's export is downstream of "
                f"this one. Make the script stop on the solver's own return "
                f"code AND on this line, then fix what the line names.")})
            break                                       # one per side is enough
    return out


def unparsed_level_files_findings(work: Path) -> list[dict]:
    """Every file that LOOKS like a per-level deliverable and does not parse as one, by name.

    A PARSER THAT SILENTLY DROPS WHAT IT CANNOT PARSE IS THE GIVE-UP GATE'S
    SHAPE AGAIN: silence read as absence. `_LEVEL_FILE` takes the side as
    `[A-Za-z0-9]+` after `_level<k>_`, so a name like

        interface_level1_side_A.csv

    does not match at all -- not "wrong side", NO MATCH -- and the file is
    dropped on the floor.

    MEASURED on one coupled run. Its generator wrote six interface
    deliverables that way (the participants and the imports keys were named
    side_A / side_B throughout the run, which is the obvious priming). Every
    reply then said "interface files for no side(s)" without naming a file or
    the form that parses; the agent listed the six files twice, right next to
    that sentence, called the audit twice more, and stopped with sixteen
    minutes left. The grader, reading the same names, returned malformed. The
    file CONTENTS were as sound as a correct sibling's: the same 44 probes,
    u agreeing between sides to 1.5e-7, flux mismatch 2.6%, 0.7%, 0.2% per level.

    So: name each such file, say which part of the name failed, and state the
    form that parses. Abstain on names that do not look per-level at all --
    "level" appears in plenty of names that are nobody's deliverable.
    """
    out: list[dict] = []
    bad: list[tuple[str, str]] = []
    for q in sorted(work.rglob("*.csv")) + sorted(work.rglob("*.log")):
        try:
            if _SCRATCH & set(q.relative_to(work).parts[:-1]):
                continue
        except ValueError:
            continue
        name = q.name
        if _LEVEL_FILE.match(name):
            continue
        if not _LOOKS_PER_LEVEL.search(name):
            continue
        stem = name.split("_")[0].lower() if "_" in name else ""
        if stem not in _DELIVERABLE_STEMS:
            continue
        # say WHICH part of the name is the problem
        m = re.match(r"^(?P<kind>[A-Za-z][A-Za-z0-9_]*?)_level(?P<rest>.*)\.(?P<ext>[A-Za-z0-9]+)$", name)
        if m is None:
            why = "the level index does not follow the stem as `_level<k>`"
        else:
            rest = m.group("rest")
            if rest.startswith("_"):
                why = "there is an underscore between `level` and the index"
            elif re.match(r"^\d+_[A-Za-z0-9]+_[A-Za-z0-9]+$", rest):
                _tok = rest.split("_", 1)[1]
                _a, _b = _tok.split("_", 1)
                why = (f"the side carries an extra token -- `_{_tok}` is read as "
                       f"side `{_a}` followed by a stray `_{_b}`, and the name does "
                       f"not parse")
            elif not re.match(r"^\d+", rest):
                why = "the side comes before the level index"
            else:
                why = "the part after the level index is not a single side token"
        bad.append((name, why))
    if not bad:
        return out
    shown = "; ".join(f"{n} ({w})" for n, w in bad[:6])
    more = f"; and {len(bad) - 6} more" if len(bad) > 6 else ""
    example = bad[0][0]
    fixed = re.sub(r"_side_([A-Za-z0-9]+)", r"_\1", example)
    fixed = re.sub(r"_level_(\d+)", r"_level\1", fixed)
    out.append({"sequence": "level file names", "priority": 8, "values": [],
                "finding": (
        f"{len(bad)} FILE(S) ARE NAMED LIKE PER-LEVEL DELIVERABLES AND DO NOT "
        f"PARSE AS ONE, so every check here treats them as absent: {shown}{more}. "
        f"The form that parses is `<stem>_level<k>_<side>.<ext>` with a single "
        f"side token -- for example `interface_level1_A.csv`"
        + (f", so `{example}` would be `{fixed}`" if fixed != example else "")
        + ". A reader of the result set drops what it cannot parse just as this "
        f"audit did, so a complete set of files under these names is read as "
        f"no files at all. Check the exact names your task prescribes and "
        f"rename before you deliver.")})
    return out


def contract_findings(work: Path, only_levels: set | None = None) -> list[dict]:
    """Result-set defects an independent check fails on that this audit never
    checked.

    `only_levels` restricts every per-level judgement to the levels named,
    which is what lets these findings be delivered DURING the ladder instead
    of at hand-in: at level 1 the agent is owed the level-1 proof and nothing
    about levels it has not reached. Left None (the default, and what the
    hand-in audit passes) nothing changes.

    The audit existed to catch what sinks a result set, and 27 of 36
    single-code runs called it — but it looked only at the convergence
    sequences. Two naming-convention failures it was blind to killed real runs
    on problems that had worked before:

      a level without its captured run log — no per-level run log carrying
                                             `NDOF = <n>`
      a file not assignable to a subdomain  — a level-1 field file delivered
                                             twice, in two places, with
                                             different contents

    Both are cheap to detect from the agent's own files, and both are fatal
    when an independent check sees them. Same shape as the other defects this
    project keeps finding: the mechanism existed, was called, and did not
    reach the case it was built for.
    """
    out: list[dict] = []

    # 1. every level that has a solution file needs a run log carrying NDOF
    sols = _field_files(work)
    levels = set()
    for f in sols:
        if _level_of(f) is not None:
            levels.add(_level_of(f))
    if only_levels is not None:
        levels &= set(only_levels)
    if levels:
        # PER SIDE, NOT PER LEVEL. This asked whether ANY log at a level carried a dof count, and a
        # coupled run is graded per side: "the file for side A must carry the output of the code that
        # solved subdomain A". Measured on a round-48 cell that had both codes PROVEN and the coupling
        # PROVEN and lost anyway -- side A had no dof line at levels 2 and 3, side B none either, and
        # this finding stayed silent because level 1's logs had them. A gate that is weaker than the
        # contract it guards is worth nothing at the moment it matters.
        sides = sorted({s for _q, _k2, _kk, s in _level_files(work, "log") if s})
        missing = []
        detail: list[str] = []
        for k in sorted(levels):
            logs = _level_logs(work, k)
            if sides:
                for side in sides:
                    have = [q for q in logs
                            if (_LEVEL_FILE.match(q.name).group("side") or "") == side]
                    if not any(_DOF_LINE.search(q.read_text(errors="ignore")) for q in have):
                        missing.append(f"{k}{side}")
                        detail.append(_dof_line_detail(work, k, side, have, logs))
            elif not any(_DOF_LINE.search(q.read_text(errors="ignore")) for q in logs):
                missing.append(str(k))
        if missing:
            _what = ("level/side " if sides else "level(s) ")
            out.append({"sequence": "run-log contract", "values": [],
                        "finding": (
                f"NO DOF-COUNT LINE in a captured run log for {_what}"
                f"{missing}. Every level needs its own run log PER SIDE, each carrying that "
                f"side's own solver console output and a line stating the number of degrees of "
                f"freedom (`NDOF = 1234`) from that side's own dof count; a level-and-side "
                f"without that log cannot be shown to have run, however good its numbers are."
                + ("\n  " + "\n  ".join(detail) if detail else ""))})

    # 1a. A COUPLED SUBMISSION WITHOUT ITS ITERATION HISTORY.
    #
    # Measured on a round-48 cell: both codes PROVEN, the interface satisfied, three levels of fields
    # and interface tables -- and no residual_level<k>.csv anywhere, so the coupling itself could not
    # be shown and the submission was read as malformed. The audit named everything except the thing
    # that decided it. Fires only when the work LOOKS coupled (side-tagged files), so a single-code
    # run is never told to produce a history it has no reason to have.
    if levels and sides:
        hist = {kk for _q, kind, kk, _s in _level_files(work, "csv") if kind.startswith("residual")}
        gap = sorted(levels - hist)
        if gap:
            out.append({"sequence": "coupling evidence", "values": [],
                        "finding": (
                f"NO ITERATION HISTORY for level(s) {gap} of what looks like a COUPLED run "
                f"(this work carries side-tagged files for {', '.join(sides)}). Each coupled level "
                f"needs its own residual_level<k>.csv listing every iteration of the partitioned "
                f"scheme and the interface mismatch at it, from the first to the last. Fields and "
                f"interface tables do not show that two codes exchanged anything: a coupled run "
                f"that cannot produce that history cannot be read as coupled at all, however good "
                f"its numbers are.")})

    # 1b. IS THE DISCRETISATION THE ONE THE TASK ASKED FOR?
    #
    # Both numbers come from the agent's own two files, so this needs no key,
    # no spec and no backend knowledge: the NDOF the run printed at its
    # COARSEST level, against the number of rows in that level's solution file.
    #
    # Measured over the 336 single-code runs on disk that wrote both files:
    #
    #     highest ratio among verified-correct runs      0.50
    #     threshold NDOF/rows > 2 fires on 8 runs        0 verified correct
    #                                                   8 of 8 timed out or
    #                                                   fell short of the
    #                                                   prescribed levels
    #
    # The failure it names is specific and fatal, and it is visible at LEVEL
    # ONE while there is still time to fix it. Three openPASO runs of one Stokes
    # task hit it with an identical NDOF of 592,387 against a 1936-point
    # probe grid — a solve two orders of magnitude larger than the
    # prescribed coarsest mesh, which completes level 1 and then cannot finish
    # level 2 inside the clock. Three runs, one cause, no warning.
    #
    # It fires on the mirror-image defect too, and says so: one run wrote a
    # solution file with 2 rows, which trips the same ratio from below.
    if levels:
        k0 = min(levels)
        sol0 = [f for f in sols if _level_of(f) == k0 and not _side_of(f)] or \
               [f for f in sols if _level_of(f) == k0]
        nd = None
        for p in _level_logs(work, k0):
            m = _DOF_LINE.search(p.read_text(errors="ignore"))
            if m:
                nd = int(m.group(1))
                break
        if nd and sol0:
            try:
                rows = sum(1 for _ in open(sol0[0], errors="ignore")) - 1
            except OSError:
                rows = 0
            if rows > 0 and nd > 2 * rows:
                out.append({"sequence": "discretisation size", "values": [],
                            "finding": (
                    f"AT YOUR COARSEST LEVEL YOUR OWN NDOF IS {nd} AGAINST "
                    f"{rows} ROWS in {sol0[0].name} — a ratio of "
                    f"{nd / rows:.0f}. Two causes produce this, and they need "
                    f"opposite fixes: (a) the mesh is far larger than the "
                    f"coarsest level the task prescribes, so level 1 is "
                    f"already an expensive solve and the finer levels cannot "
                    f"finish — re-read the prescribed mesh sizes and start at "
                    f"the coarsest one; or (b) the solution file has far fewer "
                    f"rows than the task's probe grid, so the deliverable is "
                    f"short whatever the solve did. Check which one you have "
                    f"NOW: fixing it at level 1 costs one solve, at level 3 "
                    f"it costs the whole ladder.")})

    # 1c. A SOURCE TERM BUILT FROM ELEMENT-LOCAL COORDINATES.
    #
    # This is a STATIC read of the agent's own script, which is a departure
    # from the rest of this file, and it is here because it is the one failure
    # that no numeric self-check can see. Measured on one recorded run: the
    # run bound `x` and `y` to specialcf.xref(2) — the position inside the
    # reference element — while its source term was stated in global
    # coordinates. Its own comment read "reference coordinates which equal
    # physical coords for unit square". They do not.
    #
    # The consequence is invisible to every self-consistency test: the form
    # assembles, the solve succeeds, the successive differences fall smoothly,
    # the audit passed it, and the answer is wrong. Replaying its script
    # unchanged reproduces 6.158955e-02 against the correct 7.196098e-02, and
    # order 0.028 against 2.069 measured against an independent reference.
    #
    # Only flagged when the run ALSO has a coordinate-indexed deliverable, so
    # a legitimate use of xref (a per-element quantity, an error indicator) in
    # a run that never claims a global field is left alone.
    if levels:
        for script in [q for q in work.rglob("*.py") if not _is_backup(q)][:40]:
            if _SCRATCH & set(script.relative_to(work).parts[:-1]):
                continue
            try:
                text = script.read_text(errors="ignore")
            except OSError:
                continue
            if "specialcf.xref" not in text:
                continue
            rebinds = re.search(
                r"^\s*(?:x|y)\s*=\s*\w*xref\w*\s*\[", text, re.M) or \
                re.search(r"xref\s*=\s*specialcf\.xref", text)
            if not rebinds:
                continue
            out.append({"sequence": "source coordinates", "values": [],
                        "finding": (
                f"{script.name} BUILDS AN EXPRESSION FROM specialcf.xref, "
                f"WHICH IS THE POSITION INSIDE THE REFERENCE ELEMENT, NOT ON "
                f"THE DOMAIN. If your source term, coefficient or boundary "
                f"data was stated in global coordinates, this is silently a "
                f"different function: it repeats the same small range in every "
                f"element. Nothing raises — the form assembles, the solve "
                f"succeeds, and the refinement study looks orderly while "
                f"converging to the wrong answer. In NGSolve the `x` and `y` "
                f"you get from `from ngsolve import *` ARE the global "
                f"coordinates; do not rebind them. Check it in one line: your "
                f"source evaluated at an interior point must equal the "
                f"arithmetic you do by hand for that point, and must not "
                f"change when you look at a different element containing it. "
                f"Measured on a real run: the xref form gave "
                f"max|u| = 6.158955e-02 and order 0.028 against an independent "
                f"reference; the identical script using global x, y gave "
                f"7.196098e-02 and order 2.069.")})
            break

    # 1d. THE RESIDUAL YOU REPORT MUST MEASURE THE TWO SIDES, NOT AN ITERATE.
    #
    # Measured over 29 coupled result sets carrying two-sided interface files:
    # SEVEN report INTERFACE_RESIDUAL below 1e-5 while their own exported sides
    # differ by more than 5% — up to 102% — and about fifteen do so on the flux
    # balance, with mismatches near 100%. One recorded run is the clearest:
    # side A writes the interface field as exactly 0.0, side B writes -1.1e-03
    # which is B's entire field scale, the fluxes miss by 189%, and RESULT.txt
    # reports INTERFACE_RESIDUAL = 1.12e-07 after a six-iteration history that
    # falls smoothly from 1.0.
    #
    # So the coupling did converge — something converged — but not the quantity
    # the task asks about. Nothing else catches this: the residual history looks
    # textbook, the fields converge under refinement, and the run reads as a
    # clean success right up to the independent check.
    #
    # Both numbers come from the agent's OWN two files, so this needs no key and
    # no reference.
    iface = {}
    for f in _interface_files(work, sided=True):
        iface.setdefault(_level_of(f), {})[_side_of(f).upper()] = f
    for lvl in sorted(iface, reverse=True):
        side = iface[lvl]
        if len(side) != 2:
            continue
        sa, sb = sorted(side)
        # SPLIT BY THE FILE'S OWN HEADER, NOT BY THE SCALAR LAYOUT. This
        # check used to read column 2 as the field and column 3 as the flux
        # unconditionally. On a thermo-mechanical interface
        # (x,y,T,ux,uy,qn,tx,ty) that takes ux -- a displacement, CONTINUOUS
        # across the interface by construction -- for the flux, so a correct
        # coupling was reported as "the two outward fluxes fail to cancel by
        # 200%" (measured: |ux_A + ux_B|/max|ux| = 200.000% precisely
        # BECAUSE ux_A = ux_B, while the true trailing fluxes balanced to
        # 2.1e-3..5.1e-3, under this check's own 5% bar). The header-aware
        # splitter below already existed for the sign checks; this check
        # never used it.
        parsed = {}
        for s, p in side.items():
            byh = _read_iface_by_header(p)
            if byh is not None:
                parsed[s] = byh[1], byh[2]        # (values, fluxes)
        A0, B0 = parsed.get(sa), parsed.get(sb)
        if not A0 or not B0:
            continue
        (uA, qA), (uB, qB) = A0, B0
        # NO EARLY EXIT ON AN ALL-ZERO FLUX. This used to skip the level when
        # every flux entry on both sides was exactly zero -- so a field that
        # disagreed between the two sides went unreported beside a small
        # reported residual, because a DIFFERENT column happened to be dead.
        # The per-component loops below already skip a component that is ~0 on
        # both sides, so a dead flux contributes nothing to dq and the field is
        # still judged.
        if len(uA) < 2 or len(uA) != len(uB) or len(qA) != len(qB) or not qA:
            continue
        ncomp_u = min(len(uA[0]), len(uB[0])) if uA and uA[0] else 0
        ncomp_q = min(len(qA[0]), len(qB[0])) if qA and qA[0] else 0
        if not ncomp_u or not ncomp_q:
            continue
        du = dq = 0.0
        u_ref = max((abs(x) for row in (uA + uB) for x in row), default=0.0)
        q_ref = max((abs(x) for row in (qA + qB) for x in row), default=0.0)
        # a component that is physically ~0 on both sides (tangential
        # traction in a symmetric arrangement) must not be judged against
        # its own noise scale (reviewer finding)
        for c in range(ncomp_u):
            us = max(max(abs(r[c]) for r in uA), max(abs(r[c]) for r in uB))
            if us < 1e-9 * (u_ref or 1.0):
                continue
            du = max(du, max(abs(a[c] - b[c]) for a, b in zip(uA, uB))
                     / (us or 1.0))
        for c in range(ncomp_q):
            qs = max(max(abs(r[c]) for r in qA), max(abs(r[c]) for r in qB))
            if qs < 1e-9 * (q_ref or 1.0):
                continue
            dq = max(dq, max(abs(a[c] + b[c]) for a, b in zip(qA, qB))
                     / (qs or 1.0))
        reported = None
        rt = _summary_file(work)
        if rt is not None and rt.is_file():
            m = re.search(r"^\s*([A-Za-z_]*RESIDUAL[A-Za-z_]*)\s*[=:]\s*([-+0-9.eE]+)",
                          rt.read_text(errors="ignore"), re.M | re.I)
            if m:
                label = m.group(1)
                try:
                    reported = float(m.group(2))
                except ValueError:
                    reported = None
        worst = max(du, dq)
        if reported is not None and reported < 1e-5 and worst > 0.05:
            out.append({"sequence": "interface residual", "values": [],
                        "finding": (
                f"YOU REPORT {label} = {reported:.2e}, BUT YOUR OWN "
                f"TWO INTERFACE FILES AT LEVEL {lvl} DISAGREE: the field "
                f"differs by {du:.0%} of its own scale and "
                + (f"the two outward fluxes fail to cancel by {dq:.0%}. "
                   if q_ref > 0 else
                   "BOTH flux columns are identically zero, so no flux was "
                   "exchanged at all. ")
                + f"A partitioned scheme is "
                f"converged when the SIDES agree, so the number you report has "
                f"to be computed from the two exported profiles — "
                f"max|u_A - u_B| and max|q_A + q_B| over the shared interface "
                f"probes, each relative to its own scale — and not from an "
                f"internal iterate, an update norm, or one side's own solver "
                f"residual. Those all fall to 1e-7 while the two codes still "
                f"disagree, which is what this result set shows ({worst:.0%} at "
                f"level {lvl}). "
                f"Recompute it from the files you just wrote, and if it is not "
                f"small, the coupling has not converged whatever the iteration "
                f"history says.")})
        break

    # 2. the same deliverable must not be delivered twice with different content
    # A PARTICIPANT'S OWN DUMP IS NOT A SECOND COPY OF A DELIVERABLE, AND THIS
    # CHECK TOLD AN AGENT TO DELETE THE FILE HOLDING ITS ONLY CORRECT FLUX.
    #
    # The two sides write `interface_level<k>.csv` into their OWN directories
    # by the participant contract. Those share a basename and differ in
    # content -- because they are two SIDES, not two copies -- so this check
    # named them and prescribed "delete scratch copies in subdirectories".
    #
    # MEASURED on one run: it obeyed, deleting side_A/ and side_B/
    # interface_level{1,2,3}.csv, and its next call went looking for one of
    # them and got `[file not found]`. Its handed-in flux column had been
    # re-derived from nodal values and was wrong by 114% at the finest level,
    # while the files it had just deleted cancelled to 0.288% -- the same
    # number a sibling run handed in that was correct. Our advice destroyed
    # the recovery.
    #
    # A directory holding exports.json is a participant's own working
    # directory, which is a fact about openPASO's own contract and needs no
    # knowledge of the caller's task.
    def _is_participant_dir(d: Path) -> bool:
        try:
            return (d / "exports.json").is_file() or (d / "config.json").is_file()
        except OSError:
            return False

    by_name: dict[str, set] = {}
    for f in work.rglob("*.csv"):
        if not _LEVEL_FILE.match(f.name) or _csv_role(f) == "raw":
            continue
        if f.parent != work and _is_participant_dir(f.parent):
            continue
        try:
            digest = hashlib.sha256(f.read_bytes()).hexdigest()
        except OSError:
            continue
        by_name.setdefault(f.name, set()).add(digest)
    clashes = sorted(n for n, d in by_name.items() if len(d) > 1)
    if clashes:
        out.append({"sequence": "duplicate deliverables", "values": [],
                    "finding": (
            f"MORE THAN ONE DIFFERING COPY of {clashes[:4]}. Whoever verifies "
            f"your results cannot tell which one you meant and rejects the "
            f"result set. Keep exactly one copy of each deliverable; delete "
            f"scratch copies in subdirectories before you deliver. This does "
            f"NOT mean a participant's own dumps: a directory holding "
            f"exports.json is that side's working directory, its files are "
            f"the ones your deliverable has to be BUILT FROM, and they are "
            f"not counted here.")})
    # 3. an incomplete or self-contradicting level set
    #
    # One run delivered solution_level1.csv and nothing else, with an empty
    # RESULT.txt, and this audit called it clean: the run-log check above only
    # asks about levels that HAVE a solution file, so one level with its log
    # looked complete. The result set was one level of four.
    # RESULT.txt is not always at the top level; an independent check finds it
    # anywhere.
    text = ""
    _sf = _summary_file(work)
    if _sf is not None and _sf.is_file():
        text = _sf.read_text(errors="ignore")
    if levels:
        span = max(levels)
        gaps = [k for k in range(1, span + 1) if k not in levels]
        if gaps:
            out.append({"sequence": "level sequence", "values": [],
                        "finding": (
                f"MISSING LEVEL(S) {gaps}: you have files for {sorted(levels)}, "
                f"so the sequence has a hole. A refinement study is counted "
                f"across the whole prescribed sequence.")})
        m = re.search(r"^\s*LEVELS\s*=\s*(\d+)", text, re.M)
        if m and int(m.group(1)) != len(levels):
            out.append({"sequence": "levels claimed", "values": [],
                        "finding": (
                f"Your summary file says LEVELS = {m.group(1)} but {len(levels)} "
                f"level(s) of solution files are present. The two are compared; "
                f"make them agree.")})
        if len(levels) < 3:
            out.append({"sequence": "level count", "values": [],
                        "finding": (
                f"ONLY {len(levels)} LEVEL(S) DELIVERED. A refinement study "
                f"needs a mesh sequence of at least three; an observed order "
                f"cannot be fitted from fewer, so a short result set counts "
                f"for nothing however good the levels are.")})
    if not text.strip():
        out.append({"sequence": "summary file", "values": [],
                    "finding": (
            "YOUR SUMMARY FILE IS MISSING OR EMPTY. The per-level files "
            "beside it are not read as an answer without the summary your "
            "task asks for.")})
    else:
        # A SUMMARY THAT CONTRADICTS ITS OWN NUMBER. Measured: a run wrote
        # CONVERGED beside a relative change of 0.1438 that its own shell
        # output had printed a minute earlier, with 22 minutes unused.
        _conv = re.search(r"^\s*([A-Z][A-Z0-9_]*)\s*=\s*CONVERGED\s*$", text, re.M)
        _chg = re.search(r"^\s*([A-Z0-9_]*REL[A-Z0-9_]*CHANGE[A-Z0-9_]*)\s*=\s*([-+0-9.eE]+)\s*$", text, re.M)
        try:
            _v = float(_chg.group(2)) if _chg else None
        except ValueError:
            _v = None
        if _conv and _v is not None and _v > 0.1:
            out.append({"sequence": "summary consistency", "values": [_v], "priority": 40,
                        "finding": (
                f"YOUR SUMMARY CONTRADICTS ITSELF: {_conv.group(1)} = CONVERGED beside "
                f"{_chg.group(1)} = {_v:g} -- the last refinement still moved your answer by "
                f"{100 * _v:.0f}%. No tolerance you would accept calls that converged; written "
                f"as converged it is an affirmative wrong claim. Say in the summary that it is "
                f"not converged, or find what keeps the change from shrinking (the findings "
                f"above) and refine again.")})
        # A STATED CHANGE THE FILES DO NOT SHOW. Measured on a correct cell:
        # MAX_REL_CHANGE = 0.001 beside solution files whose last two levels
        # differ by 0.8-5.5 %; nothing in the cell computed that number. The
        # measure here is the largest successive difference over the field's
        # own largest value, from the files beside the summary, and it fires
        # only when the stated number is below a fifth of it.
        if _chg and _v is not None and 0.0 <= _v < 0.05:
            try:
                _seqs = _sequences_from_level_csvs(work)
            except Exception:                            # noqa: BLE001
                _seqs = {}
            _measured = []
            for _lab, _seq in _seqs.items():
                if _lab.startswith("selfdiff_solution") and _seq:
                    _mag = _seqs.get("magnitude_" + _lab[len("selfdiff_"):]) or []
                    if _mag and _mag[0] > 0:
                        _measured.append((_lab[len("selfdiff_solution_"):], float(_seq[-1]) / float(_mag[0])))
            if _measured:
                _worst = max(_measured, key=lambda t: t[1])
                if _worst[1] > 0.005 and _worst[1] > 5.0 * max(_v, 1e-12):
                    _shown = ", ".join(f"{n} {100 * v:.1f}%" for n, v in sorted(_measured, key=lambda t: -t[1])[:6])
                    out.append({"sequence": "summary consistency", "values": [_v, _worst[1]], "priority": 40,
                                "finding": (
                        f"YOUR SUMMARY STATES {_chg.group(1)} = {_v:g}, WHICH YOUR FILES DO NOT SHOW: between "
                        f"your last two levels the fields change by {_shown} (the RMS change between the two "
                        f"levels over the field's own largest value, read from the solution files beside the "
                        f"summary). A number in a summary is measured from the files it stands beside, not "
                        f"chosen; state the measured one, or compute the definition your problem gives from "
                        f"these files.")})
    # 4. a hand-rolled sampler with the wrong shape-function normalisation
    #
    # Two recorded runs were lost to a sampler, not a solver. One run's
    # extractor wrote the QUAD4 factor 0.25 into a HEX8 shape function instead
    # of 0.125, so sum(N) = 2 and the delivered field was 2*u(x/2, y/2) -- a
    # fixed wrong field, converged to at order -0.035. Another's read the
    # nearest node's value, an O(h) reconstruction that caps the measured
    # order at 1.
    #
    # Both are visible in the agent's own scripts, without any key.
    for script in sorted(q for q in work.rglob("*.py") if not _is_backup(q)):
        try:
            src = script.read_text(errors="ignore")
        except OSError:
            continue
        # THE 0.25 FACTOR IS WRONG ONLY ON A THREE-FACTOR (HEX) PRODUCT. Measured: the same factor on
        # a four-node FACE function of a hex mesh -- 0.25 (1 - xi)(1 - eta), which is right -- drew
        # this finding 18 times in three cells.
        if re.search(r"hex8|HEX8|8\s*,?\s*#\s*nodes|zeta", src) and re.search(
                r"0\.25\s*\*\s*\(1\s*[-+]\s*xi[^()\n]*\)\s*\*\s*\(1\s*[-+]\s*eta[^()\n]*\)"
                r"\s*\*\s*\(1\s*[-+]\s*zeta", src):
            out.append({"sequence": script.name, "values": [],
                        "finding": (
                "SHAPE-FUNCTION NORMALISATION: this script builds a "
                "three-dimensional (hex) shape function with the factor 0.25, "
                "which is the QUAD4 value. For HEX8 it is 0.125, and with 0.25 "
                "the functions sum to 2 rather than 1 -- every sampled value is "
                "doubled AND, if the same N is used to invert the geometry, the "
                "point located is halved. Check sum(N) == 1 at any point.")})
        # `np.argmin(dist)` is also how a Stokes deck pins its pressure at the
        # domain centre, which is correct and unrelated. Require the argmin to
        # sit in a script that WRITES the deliverable CSV, and to be looking
        # up a field value, before calling it a sampling defect.
        writes_probe_csv = re.search(r"_level|probe", src, re.I)
        # A pressure PIN also uses argmin(distance) -- "pin the pressure at the
        # node nearest the centre" is correct, required in a Stokes problem,
        # and not sampling. Require the index to be used to READ A FIELD, and
        # exclude the pin idiom by name.
        looks_up_value = re.search(
            r"\[\s*(closest_id|nearest_idx|closest|nearest)\s*\]", src, re.I)
        is_pressure_pin = re.search(r"pin_p|pin_pressure|pressure_pin|pin_dof",
                                    src, re.I)
        if writes_probe_csv and looks_up_value and not is_pressure_pin and \
                re.search(r"argmin\(.*dist|closest_id|nearest[_ ]node", src, re.I) and \
                not re.search(r"probes\(|compute_colliding_cells|\.sample\(", src):
            out.append({"sequence": script.name, "values": [],
                        "finding": (
                "NEAREST-NODE SAMPLING: this script reads the value at the "
                "closest node instead of interpolating inside the element. That "
                "is a piecewise-constant reconstruction with O(h) error, and it "
                "CAPS your measured convergence order at 1 however good the "
                "solve is. Measured: nearest-node gives order 1.12, 1.00, 0.93 "
                "on a field where shape functions give 1.80, 2.03, 2.01.")})
    return out



_IFACE_COORD_NAMES = ("x", "y", "z")
_IFACE_FLUX_PREFIXES = ("q", "t")      # qn, q, tx, ty, tz, traction_x, ...


def _iface_column_diagnosis(path_a, path_b, ga, gb) -> str:
    """WHICH column disagrees, and whether it is a copy of another column of the same file.

    Measured on a recorded run: the two interface files agreed in ux and uy to 1e-6 and disagreed in T
    by a factor 40 -- side A's T column was its own uy column at every row (a wrong column index when the
    file was written). The finding said 'field jump 6.2e+01' and the parent read it as a physics error,
    wrote a warning into its summary and stopped with 13 minutes left. Names the column and the copy."""
    try:
        ha = [c.strip() for c in path_a.read_text(errors="replace").splitlines()[0].split(",")]
        hb = [c.strip() for c in path_b.read_text(errors="replace").splitlines()[0].split(",")]
        va, vb = ga[1], gb[1]
        n = min(len(va), len(vb))
        ncol = min(len(va[0]), len(vb[0])) if n else 0
        if n == 0 or ncol == 0:
            return ""
        ncoord = len(ga[0][0]) if ga[0] else 2
        names_a = ha[ncoord:ncoord + ncol] if len(ha) >= ncoord + ncol else [f"column {c + 1}" for c in range(ncol)]
        per = []
        for c in range(ncol):
            col_a = [va[i][c] for i in range(n)]; col_b = [vb[i][c] for i in range(n)]
            sc = max(max(abs(v) for v in col_a), max(abs(v) for v in col_b), 1e-300)
            per.append((names_a[c], max(abs(col_a[i] - col_b[i]) for i in range(n)) / sc))
        bad = [nm for nm, jp in per if jp > 1e-2]
        good = [nm for nm, jp in per if jp <= 1e-2]
        txt = " COLUMN BY COLUMN: " + "; ".join(f"{nm} {jp:.1e}" for nm, jp in per) + "."
        if bad and good:
            txt += f" The disagreement is in {', '.join(bad)} alone while {', '.join(good)} agree, so the coupling exchanged the right data and the FILE has the wrong column."
        # a column that equals another column of the same file, row for row
        for side, vals, hdr in (("A", va, ha), ("B", vb, hb)):
            names = hdr[ncoord:ncoord + ncol] if len(hdr) >= ncoord + ncol else [f"column {c + 1}" for c in range(ncol)]
            for c1 in range(ncol):
                for c2 in range(c1 + 1, ncol):
                    if all(abs(vals[i][c1] - vals[i][c2]) <= 1e-12 * max(1.0, abs(vals[i][c1])) for i in range(n)) and \
                       any(abs(vals[i][c1]) > 1e-300 for i in range(n)):
                        txt += (f" In side {side}'s file the {names[c1]} column equals its {names[c2]} column at every row: the same "
                                f"quantity was written twice -- fix the column index in the script that writes this file, not the coupling.")
        return txt
    except Exception:                                    # noqa: BLE001
        return ""


_IFACE_MATCH_TOL = 1e-6
_DELIVERABLE_FLUX_TOL = 0.01


def _qc():
    """core.quality_checks, importable from any entry point."""
    try:
        from core import quality_checks as qc
    except Exception:                                      # noqa: BLE001
        import sys as _sys
        _here = str(Path(__file__).resolve().parents[1])
        if _here not in _sys.path:
            _sys.path.insert(0, _here)
        from core import quality_checks as qc
    return qc


def _interface_curve(points, allow_gaps=False):
    """quality_checks.interface_curve, importable from any entry point."""
    return _qc().interface_curve(points, allow_gaps=allow_gaps)


def _curve_legs(points, allow_gaps=False):
    """The straight legs of a BENT 2-D interface.

    (legs, None) when the points bend: two or more straight legs, each a dict of
    quality_checks.interface_curve ("idx", "t", "n", "c"). (None, None) when they
    lie on one straight line, or are not a set of (x, y) points: the one-line
    reading of the caller applies, as it always did. (None, why) when they bend
    and cannot be read as straight legs: the caller abstains and says why.
    MEASURED on an exactly right result set across a two-leg seam: read as one
    line, with one normal axis and one plane, the checks that use this called
    it wrong by 25-537 %. `allow_gaps` reads points that leave a stretch of the
    interface unsampled (probe points kept away from a corner) leg by leg too.
    """
    try:
        import numpy as np
        P = np.atleast_2d(np.asarray(points, float))
        if P.ndim != 2 or P.shape[0] < 3 or P.shape[1] != 2 or not np.isfinite(P).all():
            return None, None
        sv = np.linalg.svd(P - P.mean(axis=0), compute_uv=False)
    except Exception:                                      # noqa: BLE001
        return None, None
    if sv[0] <= 0 or sv[1] <= 1e-8 * sv[0]:
        return None, None
    try:
        cur, why = _interface_curve(P, allow_gaps=allow_gaps)
    except Exception:                                      # noqa: BLE001
        return None, None
    if cur is None:
        return None, f"do not form one curve ({why})"
    legs = [g for g in cur["legs"] if g["t"] is not None]
    if len(legs) < 2:
        return None, None
    if not all(g["straight"] for g in legs):
        return None, "bend, but the pieces between the bends are not straight (a curved interface)"
    return legs, None


def _leg_profiles(pts, flux):
    """Per straight leg of an exported interface: (centre, direction, normal,
    distance along the leg of each point, its flux, a tolerance). A point two
    legs share -- the corner -- is in neither: its recovered flux mixes the two
    legs' normals, and interpolating along one leg through it compares a
    different quantity. (quality_checks.leg_profiles, with the flux's first
    component read at each point.)"""
    import numpy as np
    P = np.atleast_2d(np.asarray(pts, float))
    if P.ndim != 2 or P.shape[1] < 2 or len(P) != len(flux):
        return []
    return [(g["c"], g["t"], g["n"], [float(v) for v in g["a"]],
             [float(flux[int(i)][0]) for i in g["rows"]], g["tol"]) for g in _qc().leg_profiles(P)]


def _on_one_leg(p, profiles):
    """The profile whose leg the point p lies on (inside its span), when exactly one."""
    import numpy as np
    hits = []
    for j, (c, t, n, a, _ys, tol) in enumerate(profiles):
        d = np.asarray(p, float)[:2] - c
        s = float(d @ t)
        if abs(float(d @ n)) <= tol and min(a) - tol <= s <= max(a) + tol:
            hits.append(j)
    return hits[0] if len(hits) == 1 else None


def _owner_side(name) -> str:
    """The side letter a participant folder stands for: side_A, sideA, A -> 'A'."""
    return re.sub(r"^side[_\-\s]*", "", str(name), flags=re.I).upper()


def _along_interface(points):
    """Which coordinate runs ALONG the interface, and its values.

    A deliverable is written on the probe points a task prescribes and an
    export on the solver's own interface nodes; they are the same line and
    almost never the same points (measured: 44 against 31 in the run this was
    written for). So they are compared as functions of the coordinate that
    varies, not point by point.
    """
    if not points:
        return None, []
    dims = len(points[0])
    spans = [max(p[d] for p in points) - min(p[d] for p in points)
             for d in range(dims)]
    axis = max(range(dims), key=lambda d: spans[d])
    return (axis, [float(p[axis]) for p in points]) if spans[axis] > 0 else (None, [])


def _interp_on(xs, ys, at):
    """Linear interpolation of (xs, ys) at `at`, or None outside their span.

    NEVER EXTRAPOLATES. A deliverable point beyond the exported span has no
    comparison available, and inventing one there is how a check of this kind
    produces its first false accusation.
    """
    pairs = sorted(zip(xs, ys))
    lo, hi = pairs[0][0], pairs[-1][0]
    if at < lo - 1e-12 or at > hi + 1e-12:
        return None
    for (x0, y0), (x1, y1) in zip(pairs, pairs[1:]):
        if x0 - 1e-12 <= at <= x1 + 1e-12:
            if x1 == x0:
                return y0
            w = (at - x0) / (x1 - x0)
            return y0 + w * (y1 - y0)
    return pairs[-1][1]


def _read_iface_by_header(path):
    """(points, values, fluxes) split by the file's OWN header names.

    The scalar heat layout x,y,u,qn is one case, not the definition: a
    thermo-mechanical interface carries x,y,T,ux,uy,qn,tx,ty, and reading it
    through the scalar layout takes ty for the flux. Coordinates are the
    leading x/y/z columns; flux components are the TRAILING run of columns
    whose names start with q or t (qn, tx, ty ...); everything between is the
    field trace. Header-less or unsplittable files fall back to the scalar
    reader so 4-column behaviour is unchanged. Returns None when unreadable.
    Note "T" (temperature) starts with t too -- which is why only the
    TRAILING run counts as flux: T sits before the displacement columns, so
    the trailing scan stops before it.
    """
    try:
        rows = [r for r in path.read_text(errors="replace").splitlines()
                if r.strip()]
        if len(rows) < 2:
            return None
        hdr = [c.strip().lower() for c in rows[0].split(",")]
        if any(ch.isdigit() for ch in rows[0].replace(",", " ").split()[0]):
            return None                        # no header line: caller falls back
        ncoord = 0
        for name in hdr:
            if name in _IFACE_COORD_NAMES:
                ncoord += 1
            else:
                break
        nflux = 0
        for name in reversed(hdr[ncoord:]):
            if name and name[0] in _IFACE_FLUX_PREFIXES and name not in ("t",):
                nflux += 1
            else:
                break
        if ncoord < 1 or nflux < 1 or ncoord + nflux >= len(hdr) + 1:
            return None
        nval = len(hdr) - ncoord - nflux
        pts, vals, flux = [], [], []
        for r in rows[1:]:
            parts = [c.strip() for c in r.split(",")]
            if len(parts) != len(hdr):
                return None
            nums = [float(c) for c in parts]
            pts.append(tuple(nums[:ncoord]))
            vals.append(nums[ncoord:ncoord + nval] or [0.0])
            flux.append(nums[ncoord + nval:])
        return pts, vals, flux
    except Exception:                          # noqa: BLE001
        return None


def _read_iface(path, _IF, want_flux=True):
    """Header-aware read with the scalar reader as the fallback."""
    got = _read_iface_by_header(path)
    if got is not None:
        return got
    try:
        g, _why = _IF.read_interface_csv(path, 2, 1, 1 if want_flux else 0)
    except Exception:                          # noqa: BLE001
        return None
    return g


def deliverable_flux_vs_export_findings(work: Path) -> list[dict]:
    """The flux handed in is not the flux the solver produced.

    MEASURED 2026-09-19 on a coupled run whose SOLVE AND COUPLING WERE BOTH
    SOUND. That run's exported fluxes cancel across the interface to 0.288% --
    bit-identical to the number a sibling run handed in that was correct --
    and its solution files are md5-identical to two correct siblings. It graded
    unphysical because its post-processing script never opened
    `side_<S>/interface_level<k>.csv`. It re-derived the flux from nodal values
    with a ten-nearest-node least-squares line fit, whose sample spans four x
    values and three y values, so the field's variation ALONG the interface was
    aliased into the normal derivative. Deviation from that side's own exported
    flux: 85%/241% at level 1, 70%/143% at level 2, 59%/100% at level 3.

    Every existing interface finding is TWO-SIDED -- it compares q_A with q_B --
    so a post-processing defect arrives dressed as a coupling failure. The lead
    finding that run actually read was "the coupling has not converged whatever
    the iteration history says", which was FALSE: it had converged to 8.3e-07.
    Its last five actions followed from believing that.

    This check is ONE-SIDED and needs no partner: it asks whether the flux
    column of a handed-in deliverable is the flux that side's own participant
    exported. Both are the caller's own files. Nothing is computed for them and
    nothing is compared against a reference.

    Matching is by nearest exported point, over BOTH participants and BOTH
    signs, keeping the best agreement -- so a deliverable whose sides are named
    the other way round, or whose sign convention is flipped, is recognised as
    matching rather than accused. Measured separation on the five runs of that
    round: 54.3% and 71.5% for the one that re-derived, and 9e-14 or less for
    the four that read the contract file. Twelve orders of magnitude, so the
    threshold is not a tuning question.

    IT ABSTAINS RATHER THAN GUESSES when there is no export to compare against,
    when the deliverable carries no flux column, or when a point cannot be
    matched -- and abstention is where a check like this quietly dies, so the
    reachability of the positive branch is pinned by its tests.
    """
    out: list[dict] = []
    try:
        from blind_eval import interface as _IF            # noqa: PLC0415
    except Exception:                                      # noqa: BLE001
        return out

    exported: dict = {}
    sides: list = []
    for d in sorted(p for p in work.iterdir() if p.is_dir()):
        if not ((d / "exports.json").is_file() or (d / "config.json").is_file()):
            continue
        sides.append(d)
        for q, kind, k, _s in _level_files(d):
            if _csv_role(q, kind) != "interface":
                continue
            got = _read_iface(q, _IF)
            if got and got[0] and got[2]:
                exported.setdefault(k, []).append((d.name, q, got[0], got[2]))

    # exports.json IS THE LAST SURVIVING WITNESS, and often the only one.
    #
    # The per-side interface dumps are what a deliverable should be built
    # from, but they are also what an agent deletes when something tells it to
    # tidy up -- which is exactly what happened in the run this check was
    # written for. exports.json holds the SAME quantity for the LAST level
    # solved, because the driver overwrites it each level, so it is attached to
    # the highest level index present and to no other.
    _levels_seen = {k for _q, _kind, k, s in _level_files(work)
                    if _q.parent == work and s
                    and _csv_role(_q, _kind) == "interface"}
    if _levels_seen:
        _finest = max(_levels_seen)
        for d in sides:
            if any(owner == d.name for owner, *_ in exported.get(_finest, [])):
                continue                      # a real per-level dump beats it
            try:
                blob = json.loads((d / "exports.json").read_text())
            except (OSError, ValueError):
                continue
            pts = [tuple(float(c) for c in row)
                   for row in (blob.get("coordinates") or [])]
            # A THERMO-MECHANICAL EXPORT IS A VECTOR PER POINT ([qn, tx, ty]);
            # float(list) raised here, the exception escaped audit(), and a
            # cell whose side-A files were byte-identical across levels with
            # a literal zero flux column received "[auto-audit unavailable:
            # TypeError]" twice instead of the two findings that named it.
            qn = [([float(c) for c in v] if isinstance(v, (list, tuple)) else [float(v)])
                  for v in (blob.get("normal_fluxes") or [])]
            if pts and qn and len(pts) == len(qn):
                exported.setdefault(_finest, []).append(
                    (d.name, d / "exports.json", pts, qn))

    for q, _kind, k, side in _level_files(work):
        if q.parent != work or _csv_role(q, _kind) != "interface" or not side:
            continue
        mine = _read_iface(q, _IF)
        if not mine or not mine[0] or not mine[2]:
            continue
        if not exported.get(k):
            continue
        # THE SAME SIDE'S EXPORT, WHERE THE FOLDERS SAY WHICH SIDE IS WHICH. Taken
        # over both participants, the finding on an exactly right bent set named
        # side A's export for side B's file (measured): with opposite outward
        # normals the two exports are each other's negative, both match about
        # equally, and the first one read wins. Both signs are still tried, so a
        # flipped convention is recognised; where no folder names this side,
        # every export is tried.
        cands = [e for e in exported[k] if _owner_side(e[0]) == str(side).upper()] or exported[k]
        # A BENT INTERFACE IS COMPARED LEG BY LEG: along one axis, the points of
        # its two legs interleave and one leg's flux is interpolated from the
        # other's (measured: 25-55 % on an exactly right set).
        if (_curve_legs([p[:2] for p in mine[0]])[0] is not None
                or any(_curve_legs([p[:2] for p in e[2]])[0] is not None for e in cands)):
            _dfind = _deliverable_by_leg(work, q, mine, cands, exported[k])
            if _dfind:
                out.append(_dfind)
            continue
        axis_mine, xs_mine = _along_interface(mine[0])
        if axis_mine is None:
            continue
        # A ZERO COLUMN IS NOT A STAND-IN, AND ROUND-OFF IS NOT A REFERENCE.
        # Measured: a deliverable whose heat flux was 0.0 at every point (its
        # flux-calc line sat on the wrong edge) was compared with the partner's
        # round-off and told it differed by 100% from "the flux your solver
        # produced" -- blaming a nodal stand-in for a dead channel, as the lead
        # finding. An all-zero column is named by zero_channel_findings; a
        # candidate whose own scale is round-off against the level's fluxes is
        # skipped like a zero one.
        if all(abs(float(fv[0])) == 0.0 for fv in mine[2]):
            continue
        _lvl_scale = max((abs(float(c)) for _o, _s, _p, _fl in exported[k] for f in _fl for c in f),
                         default=0.0)
        best = None
        for owner, src, pts, flux in cands:
            axis, xs = _along_interface(pts)
            if axis is None or axis != axis_mine:
                continue
            ys = [float(f[0]) for f in flux]
            if max((abs(v) for v in ys), default=0.0) <= 1e-9 * max(_lvl_scale, 1e-300):
                continue
            for sign in (1.0, -1.0):
                num = den = 0.0
                seen = 0
                for x, fv in zip(xs_mine, mine[2]):
                    b = _interp_on(xs, ys, x)
                    if b is None:
                        continue               # outside the exported span
                    seen += 1
                    b *= sign
                    num = max(num, abs(float(fv[0]) - b))
                    den = max(den, abs(b))
                # A handful of matched points is not a comparison; most of the
                # deliverable has to lie on the exported line before this may
                # say anything about it.
                if seen >= max(4, len(xs_mine) // 2) and den > 0:
                    rel = num / den
                    if best is None or rel < best[0]:
                        best = (rel, owner, src)
        if best is None or best[0] <= _DELIVERABLE_FLUX_TOL:
            continue
        out.append(_deliverable_finding(work, q, *best))
    return out


def _deliverable_finding(work: Path, q: Path, rel: float, owner, src) -> dict:
    """The finding of deliverable_flux_vs_export_findings for one handed-in file."""
    try:
        where = src.relative_to(work)
    except ValueError:
        where = src
    return {"sequence": f"deliverable flux {q.name}", "values": [rel],
            "priority": 25, "finding": (
        f"THE FLUX YOU HANDED IN IS NOT THE FLUX YOUR SOLVER PRODUCED. The "
        f"flux column of {q.name} differs by {rel:.0%} of its own size from "
        f"the outward flux {owner} exported at the same interface points "
        f"({where}, written by that side's own participant from its own "
        f"assembled system), so the column was not taken from it. Whatever "
        f"built it -- a recomputation from nodal values, an interpolation over "
        f"points that are not sorted along the interface, another level's file "
        f"each do this -- every two-sided interface finding below is then "
        f"measuring that post-processing rather than your coupling. Build "
        f"this column from that side's own "
        f"interface_level<k>.csv (or its exports.json), and keep those "
        f"files until the deliverable is written.")}


def _deliverable_by_leg(work: Path, q: Path, mine, cands, level_exports):
    """deliverable_flux_vs_export_findings on a BENT interface: each handed-in
    point is placed on the export leg it lies on and compared with that leg's
    flux, interpolated along the leg; a point on no leg, or beyond a leg's
    exported span, is not compared (never extrapolated). The corner is on no leg
    (see _leg_profiles). None when the column matches, or cannot be compared."""
    import numpy as np
    if all(abs(float(fv[0])) == 0.0 for fv in mine[2]):
        return None                     # a zero column: zero_channel_findings names it
    _lvl_scale = max((abs(float(c)) for _o, _s, _p, _fl in level_exports for f in _fl for c in f),
                     default=0.0)
    at = [np.asarray(p, float)[:2] for p in mine[0]]
    best = None
    for owner, src, pts, flux in cands:
        if max((abs(float(f[0])) for f in flux), default=0.0) <= 1e-9 * max(_lvl_scale, 1e-300):
            continue
        profiles = _leg_profiles(pts, flux)
        if not profiles:
            continue
        for sign in (1.0, -1.0):
            num = den = 0.0
            seen = 0
            for p, fv in zip(at, mine[2]):
                j = _on_one_leg(p, profiles)
                if j is None:
                    continue
                c, t, _n, a, ys, _tol = profiles[j]
                b = _interp_on(a, ys, float((p - c) @ t))
                if b is None:
                    continue
                seen += 1
                b *= sign
                num = max(num, abs(float(fv[0]) - b))
                den = max(den, abs(b))
            if seen >= max(4, len(at) // 2) and den > 0:
                rel = num / den
                if best is None or rel < best[0]:
                    best = (rel, owner, src)
    if best is None or best[0] <= _DELIVERABLE_FLUX_TOL:
        return None
    return _deliverable_finding(work, q, *best)


def _dudn_by_leg(ipts, legs, field_pts, field_vals, _IF):
    """-du/dn outward along each leg of a BENT interface, from the side's own field
    file: ([(leg label, row indices of `ipts` on the leg in order along it, -du/dn
    there)], None), or (None, why) when a leg runs along neither x nor y, or the
    side's field does not lie on one side of it.

    Per leg: its normal axis and plane, and only this side's field points beside
    it -- on its side of the leg, within the leg's span. With one axis and one
    plane for the whole interface, a notched side's own field lies on both sides of
    the line through a leg, and the derivative was taken across it (measured: an
    exactly right set read "does not follow from its own field" by 537 %)."""
    import numpy as np
    I = np.asarray(ipts, float)
    F = np.asarray(field_pts, float)
    if F.ndim != 2 or F.shape[1] != 2 or not len(F):
        return None, "the field file is not a set of (x, y) points"
    out = []
    for g in legs:
        t = g["t"]
        if min(abs(float(t[0])), abs(float(t[1]))) > 1e-6:
            return None, (f"its leg through ({g['c'][0]:.4g}, {g['c'][1]:.4g}) runs along "
                          f"neither x nor y")
        ax = 0 if abs(float(t[0])) < abs(float(t[1])) else 1        # the coordinate the leg fixes
        al = 1 - ax
        idx = sorted(set(g["idx"]), key=lambda i: float(I[i][al]))
        plane = float(np.median(I[idx, ax]))
        lo, hi = float(I[idx, al].min()), float(I[idx, al].max())
        nv = np.unique(np.round(F[:, ax], 9))
        tv = np.unique(np.round(F[:, al], 9))
        hn = float(np.median(np.diff(nv))) if len(nv) > 1 else 0.0
        ht = float(np.median(np.diff(tv))) if len(tv) > 1 else 0.0
        if hn <= 0:
            return None, "the field file has fewer than two columns of points"
        # which side: this side's points beside the leg, away from its two ends (a
        # point on the other leg, at the bend, is on neither side of this one)
        along = (F[:, al] >= lo + 0.25 * ht - 1e-9) & (F[:, al] <= hi - 0.25 * ht + 1e-9)
        dn = F[:, ax] - plane
        below = int(np.sum(along & (dn < -1e-9) & (dn >= -3.5 * hn)))
        above = int(np.sum(along & (dn > 1e-9) & (dn <= 3.5 * hn)))
        if (below > 0) == (above > 0):
            return None, (f"its field points lie on {'both sides' if below else 'neither side'} "
                          f"of its leg {'xy'[ax]} = {plane:.4g}")
        sign = 1.0 if below else -1.0                  # outward = +e_ax when the field is below
        mine = ((dn * -sign >= -1e-9) & (np.abs(dn) <= 6.0 * hn)
                & (F[:, al] >= lo - 3.0 * ht - 1e-9) & (F[:, al] <= hi + 3.0 * ht + 1e-9))
        sel = np.where(mine)[0]
        fp = [tuple(F[i]) for i in sel]
        fv = [field_vals[i] for i in sel]
        dudn = _IF.recover_normal_derivative(fp, fv, [tuple(I[i]) for i in idx], ax, plane, sign)
        out.append((f"{'xy'[ax]} = {plane:.4g}", idx, dudn))
    return out, None


_RATIO_ORDER = ("INCONSISTENT", "SIGN_CONVENTION", "MISFIT_SHRINKS", "CONSISTENT", "NOT_APPLICABLE",
                "NOT_ASSESSED")


def _ratio_words(r: dict) -> str:
    """One leg's (or side's) ratio result in words: verdict, implied coefficient, misfit."""
    pc = next(iter(r.get("per_component") or []), {})
    k, mf = pc.get("implied_coefficient"), pc.get("misfit", pc.get("spread"))
    return (str(r.get("verdict")) + (f", implied coefficient {k:.4g}" if isinstance(k, (int, float)) else "")
            + (f", misfit {mf:.0%}" if isinstance(mf, (int, float)) else ""))


def side_flux_ratio(ipts, iq, fpts, fvals, _IF, fallback_axis: int = 0) -> dict:
    """Is one side's flux, at one level, a constant multiple of its own field's normal
    derivative (flux_multiple_consistency)? Each interface point is read with its own
    normal: on a straight interface the axis its points hold constant, the
    plane there and the side its field lies on; on a BENT one leg by leg (each leg its
    own normal axis, plane and side, see _dudn_by_leg), a stretch left unsampled
    between two legs included. MEASURED: one axis for a two-leg interface took du/dx
    as the normal derivative on the leg whose normal is y, and a correct side was told
    it broke the sign convention and that its coefficient varied by 98670 %.

    Returns the dict of flux_multiple_consistency; on a bent interface its verdict is
    the worst leg's, with "per_leg" (one such dict per leg, "leg" naming it). The two
    points at each end of a leg (and of a straight interface) are left out where it has
    more than seven, as the audit's copy of this check does: at an interface end and at a
    corner a one-sided derivative is least accurate, and on node files those points alone
    read a right side's misfit at 30-90 % against 1-3 % without them (measured)."""
    import numpy as np

    def _inner(k):
        return list(range(2, k - 2)) if k > 7 else list(range(k))
    legs, bad = _curve_legs([tuple(p[:2]) for p in ipts], allow_gaps=True)
    if legs is None and bad is None:
        P = np.asarray(ipts, float)
        if P.ndim == 2 and P.shape[0] >= 2:
            ax = int(np.argmin(P.var(axis=0)))
            plane = float(P[:, ax].mean())
        else:
            ax = fallback_axis
            plane = float(ipts[0][ax]) if len(ipts) else 0.0
        try:
            sign = 1.0 if float(np.asarray(fpts, float)[:, ax].mean()) < plane else -1.0
        except Exception:                                    # noqa: BLE001
            sign = 1.0
        dn = _IF.recover_normal_derivative(fpts, fvals, ipts, ax, plane, sign)
        o = (list(np.argsort(P[:, 1 - ax], kind="stable")) if P.ndim == 2 and P.shape[1] == 2
             else list(range(len(ipts))))
        keep = [o[j] for j in _inner(len(o))]
        return _IF.flux_multiple_consistency([iq[i] for i in keep], [dn[i] for i in keep])
    if legs is None:
        return {"verdict": "NOT_ASSESSED",
                "detail": f"the interface points {bad}, so no normal could be taken along them"}
    per, why = _dudn_by_leg(ipts, legs, fpts, fvals, _IF)
    if per is None:
        return {"verdict": "NOT_ASSESSED",
                "detail": f"on this bent interface {why}, so no normal derivative could be taken leg by leg"}
    rows = []
    for lab, idx, dn in per:
        keep = _inner(len(idx))
        r = _IF.flux_multiple_consistency([iq[idx[j]] for j in keep], [dn[j] for j in keep])
        r["leg"] = lab
        rows.append(r)
    return _worst_leg(rows)


def _worst_leg(rows: list) -> dict:
    """A side's ratio result from its legs' (see side_flux_ratio): the worst leg's verdict."""
    worst = min(rows, key=lambda r: _RATIO_ORDER.index(r["verdict"]) if r["verdict"] in _RATIO_ORDER
                else len(_RATIO_ORDER))
    return {"verdict": worst["verdict"], "per_leg": rows,
            "detail": "read leg by leg along the bent interface -- " + "; ".join(
                f"leg {r['leg']}: {_ratio_words(r)}" for r in rows)}


def interface_sign_findings(work: Path) -> list[dict]:
    """The interface flux sign, from the agent's OWN files. Coupled problems
    only.

    WHY THIS IS IN THE AUDIT AND NOT ONLY IN A TOOL. The check was exposed as
    `verify_interface_flux`, described in the coupling must-read with the
    numbers from the runs it decided, and then called by ZERO of six runs in
    the next batch -- while the auto-audit on delivery reached five of those
    six. That is the same finding an earlier batch recorded for the audit
    itself: a calibrated check plus an instruction to run it was used by 1 of
    51 agents. Voluntary checking does not happen, so this one is not
    voluntary.

    What it asserts, needing no reference solution: for a flux really computed
    from your own solution, q_n(x) / (-du/dn)(x) equals k at every interface
    point, so the ratio is CONSTANT along the interface whatever k is -- and
    POSITIVE, because the task defines q_n with the OUTWARD normal.

    Measured on two real result sets, checked against an independent
    reference:
      one recorded run, complete but unphysical at order 1.2434 with
        BOTH prescribed codes proven to have run and the interface field
        matching to 0.000e+00: level 3 side B implied coefficient -250.8,
        against +198.1 and +200.3 at levels 1 and 2. It had the sign right on
        the coarse meshes and flipped it on the finest.
      the 4C+Kratos reference, verified correct at order 1.9796: all six sides
        positive, +0.98 to +1.30 where k = 1 and +200.4 to +206.7 where
        k = 200 -- the check recovering both conductivities from the
        delivered files alone.
    """
    import re as _re

    try:
        import sys as _sys
        _here = str(Path(__file__).resolve().parents[1])
        if _here not in _sys.path:
            _sys.path.insert(0, _here)
        from blind_eval import interface as _IF
    except Exception:
        return []

    def _index(files):
        out = {}
        for f in files:
            if _side_of(f):
                out[(_level_of(f), _side_of(f).upper())] = f
        return out

    ifs, sols = _index(_interface_files(work, sided=True)), _index(_field_files(work))
    if not ifs:
        return []                       # not a coupled result set: say nothing

    inverted, assessed, jumps = [], 0, {}
    ratio_rows: dict = {}           # side -> [(level, verdict, implied k, spread)]
    jumps_c: dict = {}              # level -> per-channel RMS jump, one per flux column
    chan_names: list = []
    vector_layout = 0
    bent_skipped: list = []         # (level, side, why) where a bent interface could not be read
    # A SIDE WHOSE CONDUCTIVITY IS A TENSOR: q_n / (-du/dn) is not one constant there
    # (see _k_tensor), so its sign and ratio are not judged; the two-sided checks
    # below still read its files.
    # A SIDE THAT STATES NO CONDUCTIVITY IS NOT JUDGED AS A SCALAR EITHER. Measured: two sides
    # whose config.json carried only level, nx and ny, and whose programs set K as a 2x2
    # matrix, were read as scalar sides, and a right result's flux "DOES NOT FOLLOW FROM ITS
    # OWN FIELD" (1314 % / 145 % / 561 %) led all three of its hand-in audits. Where the config
    # is silent the side's own program is read (see _k_in_program): a matrix there is a tensor,
    # a number a scalar, and where neither says, the coefficient is unknown and nothing that
    # needs it is judged.
    _tensor: dict = {}
    _unknown: dict = {}
    try:
        _said, _silent = set(), {}
        for _d in sorted({p.parent for p in work.glob("*/config.json")}
                         | {d for d in _side_dirs(work) if (d / "config.json").is_file()}):
            try:
                _cfg_d = json.loads((_d / "config.json").read_text() or "{}")
            except (OSError, ValueError):
                continue
            _why_t = _k_tensor(_cfg_d)
            if _why_t:
                _tensor.setdefault(_owner_side(_d.name), (_d.name, _why_t))
            elif not _k_unstated(_cfg_d):
                _said.add(_owner_side(_d.name))
            else:
                _form, _src = _k_in_program(_d)
                if _form == "tensor":
                    _tensor.setdefault(_owner_side(_d.name), (
                        _d.name, f"states no k, and {_src} there sets K as a matrix"))
                elif _form == "scalar":
                    _said.add(_owner_side(_d.name))
                else:
                    _silent.setdefault(_owner_side(_d.name), _d.name)
        _unknown = {s: n for s, n in _silent.items() if s not in _said and s not in _tensor}
    except Exception:                                    # noqa: BLE001
        _tensor, _unknown = {}, {}
    tensor_skipped: set = set()
    unknown_skipped: set = set()
    for (lvl, side), path in sorted(ifs.items()):
        gi = _read_iface(path, _IF)
        if gi is None:
            continue
        ipts, _giv, iq = gi
        if not (len(_giv) and len(iq)):
            continue                     # a file with no rows: one empty file cost every finding here
        # the k = q / (-du/dn) heuristic is defined for ONE scalar field and
        # ONE flux component; on a multi-component trace it pairs the wrong
        # columns, so it is skipped there (the ratio/mirror/zero branches
        # below handle every layout).
        if len(_giv[0]) != 1 or len(iq[0]) != 1:
            vector_layout += 1
            continue
        if str(side).upper() in _tensor:
            tensor_skipped.add(str(side).upper())
            continue
        if str(side).upper() in _unknown:
            unknown_skipped.add(str(side).upper())
            continue
        sp = sols.get((lvl, side))
        if sp is None:
            continue
        # A BENT INTERFACE IS JUDGED LEG BY LEG, each leg with its own normal axis,
        # plane and side, and each leg's ratio on its own: a sign flipped on one leg
        # is named on that leg (side "A|x = 0.4" below). The straight reading follows.
        _legs, _bad = _curve_legs(ipts, allow_gaps=True)
        if _legs is not None or _bad is not None:
            try:
                gs, _why = _IF.read_interface_csv(sp, 2, 1, 0)
                per, _bad2 = ((None, f"its interface points {_bad}") if _legs is None
                              else (None, "its field file could not be read") if gs is None
                              else _dudn_by_leg(ipts, _legs, gs[0], gs[1], _IF))
            except Exception as _e:                        # noqa: BLE001
                per, _bad2 = None, f"its field could not be read ({type(_e).__name__})"
            if per is None:
                bent_skipped.append((lvl, side, _bad2 or "its field file could not be read"))
                continue
            for _lab, _idx, _dn in per:
                _q = [iq[i] for i in _idx]
                try:
                    _qmax = max(abs(float(v[0])) for v in _q)
                    _dmax = max((abs(float(v[0])) for v in _dn if v is not None), default=0.0)
                except (TypeError, ValueError, IndexError):
                    _qmax, _dmax = 1.0, 0.0
                if _dmax > 0 and _qmax <= 1e-9 * _dmax:
                    continue                             # a zero flux column has no sign
                # THE READING verify_interface_flux GIVES: the misfit of the best constant
                # multiple, on the leg's interior. The spread of q / (-du/dn) did not fall
                # on right fields read leg by leg (130 % -> 204 % -> 43 %, measured): it is
                # ill-conditioned where -du/dn passes through zero and next to a corner.
                _kp = list(range(2, len(_idx) - 2)) if len(_idx) > 7 else list(range(len(_idx)))
                _res_in = _IF.flux_multiple_consistency([_q[i] for i in _kp], [_dn[i] for i in _kp])
                for comp in _res_in.get("per_component") or []:
                    k = comp.get("implied_coefficient")
                    if isinstance(k, (int, float)):
                        assessed += 1
                        # a flipped sign is a clean negative multiple, not any negative fit
                        if k < 0 and comp.get("verdict") == "SIGN_CONVENTION":
                            inverted.append((lvl, f"{side}|{_lab}", k))
                        if comp.get("component", 0) == 0:
                            ratio_rows.setdefault(f"{side}|{_lab}", []).append(
                                (lvl, str(comp.get("verdict")), float(k),
                                 float(comp.get("misfit") or 0.0), "misfit"))
            continue
        try:
            gs, _why = _IF.read_interface_csv(sp, 2, 1, 0)
            if gs is None:
                continue
            # THE GEOMETRY COMES FROM THE FILES, NOT FROM AN ASSUMPTION.
            # This call used to hard-code axis 0 (a vertical interface at
            # the first probe's x) and A-left/B-right. On a HORIZONTAL
            # interface it recovered du/dn along the wrong axis at a plane
            # that is not the interface and told a verified-correct
            # result set WRONG SIGN at every level -- and negating the flux
            # to obey it silenced the finding while corrupting the
            # result set (measured: implied k -0.4946 under the hard-coded
            # geometry; +1.060 and +2.092, both consistent, under the
            # files' own geometry). The interface axis is the coordinate
            # that is CONSTANT across the interface probes; the plane is
            # its value; the outward sign follows from which side of the
            # plane this side's own field points lie on.
            import numpy as _np
            _ip = _np.asarray(ipts, float)
            if _ip.ndim != 2 or _ip.shape[0] < 2:
                continue
            _axis = int(_np.argmin(_ip.var(axis=0)))
            _plane = float(_ip[:, _axis].mean())
            _sp_pts = _np.asarray(gs[0], float)
            _mean_side = float(_sp_pts[:, _axis].mean())
            sign = 1.0 if _mean_side < _plane else -1.0
            dudn = _IF.recover_normal_derivative(
                gs[0], gs[1], ipts, _axis, _plane, sign)
            res = _IF.flux_ratio_consistency(iq, dudn)
            # THE RATIO ROWS BELOW LEAVE OUT THE TWO END NODES AT EACH END: a partner
            # whose end-node flux went out raw (a corner reaction) made a right Neumann
            # side's delivered flux, which is what it applied, read "does not follow
            # from its own field" (measured). The sign test above keeps every point.
            _along = 1 - _axis
            _ord = sorted(range(len(ipts)), key=lambda i: float(ipts[i][_along]))
            _keep = _ord[2:-2] if len(_ord) > 7 else _ord
            res_inner = _IF.flux_ratio_consistency([iq[i] for i in _keep], [dudn[i] for i in _keep])
        except Exception:
            continue
        # A ZERO FLUX COLUMN HAS NO SIGN. Its implied coefficient is round-off of
        # either sign (measured: -8e-15), and calling it a wrong sign led the audit
        # above the true finding -- a channel that transmitted nothing -- and the run
        # negated its zeros.
        try:
            _qmax = float(_np.max(_np.abs(_np.asarray(iq, float))))
            _dmax = float(_np.max(_np.abs(_np.asarray(dudn, float))))
        except Exception:                                  # noqa: BLE001
            _qmax, _dmax = 1.0, 0.0
        if _dmax > 0 and _qmax <= 1e-9 * _dmax:
            continue
        for comp in res.get("per_component") or []:
            k = comp.get("implied_coefficient")
            if isinstance(k, (int, float)):
                assessed += 1
                if k < 0:
                    inverted.append((lvl, side, k))
        for comp in res_inner.get("per_component") or []:
            k = comp.get("implied_coefficient")
            if isinstance(k, (int, float)) and comp.get("component", 0) == 0:
                ratio_rows.setdefault(side, []).append(
                    (lvl, str(comp.get("verdict")), float(k), float(comp.get("spread") or 0.0),
                     "spread"))
    for lvl in sorted({l for l, _ in ifs}):
        a, b = ifs.get((lvl, "A")), ifs.get((lvl, "B"))
        if not (a and b):
            continue
        try:
            ga = _read_iface(a, _IF)
            gb = _read_iface(b, _IF)
            if ga and gb:
                ga, gb, _pw = _paired_iface(ga, gb)          # the same points, in one order
            if ga and gb:
                # two_sided_jumps returns (dict, msg); it is None on a mismatch.
                jd = _IF.two_sided_jumps(ga, gb)[0]
                if jd is not None:
                    jumps[lvl] = jd.get("jump_q_rel")
                    _pc = _per_channel_flux_jumps(ga, gb)
                    if _pc:
                        jumps_c[lvl] = _pc
                        if not chan_names:
                            chan_names = _iface_flux_names(a) or []
        except Exception:
            pass

    out: list[dict] = []
    # ONE SIDE'S FLUX ORDERS OF MAGNITUDE BELOW THE OTHER'S = NO TRANSMISSION.
    #
    # The Neumann side's outward flux must be the NEGATIVE of the Dirichlet
    # side's, so the two magnitudes are equal to discretisation error. A side
    # reporting a flux a hundred times smaller has not received its partner's
    # data at all -- in Kratos, the usual cause is FACE_HEAT_FLUX set on the
    # interface nodes with no ThermalFace2D2N condition to integrate it, which
    # is silent: same exit code, same convergence message, and exactly the
    # no-flux field. Measured: with the flux ignored the field is bit-identical
    # to the zero-flux run's.
    peaks = {}
    for (lvl, side), path in sorted(ifs.items()):
        try:
            g = _read_iface(path, _IF)
        except Exception:
            g = None
        if g is None:
            continue
        vals = [abs(c) for row in g[2] for c in row]
        if vals:
            peaks[(lvl, side)] = max(vals)
    # BOTH SIDES ZERO IS THE ONE CASE THE FAMILY ABOVE CANNOT SEE.
    #
    # The one-side-smaller check needs big > 0, the same-convention ratio
    # needs a nonzero sum, and the sign check needs a nonzero implied
    # coefficient -- so an interface whose flux column is identically zero on
    # BOTH sides slips every one of them. Measured: a real result set carried
    # max|q| = 0.000e+00 on both sides at all three levels while its fields
    # were plausible, and nothing here spoke. Physically a partitioned
    # interface with zero flux everywhere transmitted nothing: the two
    # subdomains were solved as if insulated from each other.
    # AND BIT-EXACT OPPOSITION IS ITS MIRROR. Two runs in one round exported
    # side B's flux as side A's negated to the last bit -- max|qA+qB| exactly
    # 0.0 at every level over |q| up to 76 -- which slips the same three
    # branches from the other direction: the sum is zero, so the convention
    # ratio is silent; both peaks are nonzero, so the all-zero branch is
    # silent; the implied coefficient is plausible, so the sign branch is
    # silent. Two independently solved subdomains never agree to the last
    # bit: a coupling iterated to a 1e-6 relative tolerance leaves a jump of
    # about that size. Reading "the fluxes are equal and opposite" as an
    # instruction to CONSTRUCT one side from the other leaves the claim that
    # two codes met at the interface with no support at all.
    mirror_lvls = []
    for lvl in sorted({l for l, _ in ifs}):
        a, b = ifs.get((lvl, "A")), ifs.get((lvl, "B"))
        if not (a and b):
            continue
        try:
            ga = _read_iface(a, _IF)
            gb = _read_iface(b, _IF)
            if not (ga and gb):
                continue
            ga, gb, _pw = _paired_iface(ga, gb)              # row i of each is the same point
            if not (ga and gb):
                continue
            qa = [c for row in ga[2] for c in row]
            qb = [c for row in gb[2] for c in row]
            n = min(len(qa), len(qb))
            if n and max(abs(qa[i]) for i in range(n)) > 0 and \
                    all(qa[i] + qb[i] == 0.0 for i in range(n)):
                mirror_lvls.append(lvl)
        except Exception:
            continue
    if mirror_lvls:
        out.append({"sequence": "interface flux constructed",
                    "values": mirror_lvls, "finding": (
            "SIDE B'S FLUX IS SIDE A'S NEGATED TO THE LAST BIT at level"
            + ("s " if len(mirror_lvls) > 1 else " ")
            + ", ".join(str(l) for l in mirror_lvls)
            + " (qA + qB is exactly 0.0 at every point). Two independently "
            "solved subdomains never agree to the last bit -- an iteration "
            "converged to a 1e-6 relative interface tolerance leaves a jump "
            "of about that size, not zero. 'Equal and opposite' is a "
            "statement about the CONVERGED PHYSICS, not an instruction to "
            "copy one side's column with a sign flip: computed this way, the "
            "file carries no evidence that the two solutions ever met at the "
            "interface. Compute each side's q_n = -(K grad u) . n_out from "
            "that side's OWN solution and its OWN material, and export what "
            "comes out; a small nonzero mismatch between the sides is the "
            "signature of a real coupling, not a defect to erase.")})
    zero_lvls = [lvl for lvl in sorted({l for l, _ in peaks})
                 if peaks.get((lvl, "A")) == 0.0 and peaks.get((lvl, "B")) == 0.0]
    if zero_lvls:
        out.append({"sequence": "interface flux all zero",
                    "values": zero_lvls, "finding": (
            "THE INTERFACE FLUX IS IDENTICALLY ZERO ON BOTH SIDES at level"
            + ("s " if len(zero_lvls) > 1 else " ")
            + ", ".join(str(l) for l in zero_lvls)
            + ". A coupled interface carries the flux that crosses it; a "
            "zero column on both sides means no transmission happened at "
            "all -- the two subdomains were solved as if insulated -- or the "
            "export never computed q_n = -(K grad u) . n_out from the "
            "solution. The field values themselves need no re-solve: recover "
            "the flux from your OWN existing solution (consistent nodal "
            "flux, or the gradient of your interpolant evaluated at the "
            "interface, times -K, dotted with the outward normal) and "
            "re-export. If the recovery also comes out zero, the two solves "
            "never exchanged data and the coupling itself did not run.")})
    # THE SMALLER FLUX IS NOT THE GUILTY ONE, AND THE RECEIVER IS NOT WHOEVER
    # IS SMALLER.
    #
    # This labelled the side with the smaller magnitude "the receiver", said it
    # "never RECEIVED its partner's flux", and appended Kratos advice
    # unconditionally. Measured on one run: side A was 4C, the DIRICHLET
    # side -- it receives VALUES, not flux -- and its flux was sound (0.69,
    # 0.80, 0.90 across the levels, matching a correct sibling). Side B's grew
    # 6.96, 38.3, 182: a one-sided finite difference of a field whose solve
    # had already failed, exported without a sign. The finding blamed A,
    # named Kratos, and the agent wrote "~200x larger ... consistent with
    # k=200 vs k=1" into its summary -- a rationalisation its own three
    # ratios (10x, 48x, 203x) refute. It stopped with fourteen minutes left.
    #
    # What the files can say without a role: WHICH side's magnitude changes
    # across the levels. A consistent flux settles under refinement; a flux
    # that grows with the mesh is a derivative taken by hand on a field that
    # is not converging. All levels are quoted so the reader sees the trend,
    # and the receiving side is named only where a config declares the role.
    _ratio_lvls = sorted({l for l, _ in peaks})
    _pairs = [(l, peaks.get((l, "A")), peaks.get((l, "B"))) for l in _ratio_lvls]
    _pairs = [(l, a, b) for l, a, b in _pairs if a is not None and b is not None]
    _bad = [(l, a, b) for l, a, b in _pairs
            if max(a, b) > 0 and min(a, b) < max(a, b) / 50.0]
    if _bad:
        def _drift(vals):
            vals = [v for v in vals if v > 0]
            return (max(vals) / min(vals)) if len(vals) > 1 else 1.0
        _dA = _drift([a for _l, a, _b in _pairs])
        _dB = _drift([b for _l, _a, b in _pairs])
        _ratios = ", ".join(f"level {l}: {a:.3g} on A against {b:.3g} on B "
                            f"({max(a, b) / max(min(a, b), 1e-300):.0f}x)"
                            for l, a, b in _pairs)
        _roles = {}
        try:
            for _sd, _op in _side_operators(work).items():
                _r = str(json.loads((_op["dir"] / "config.json").read_text() or "{}")
                         .get("side", "")).lower()
                if _r:
                    _roles[_sd[-1].upper() if _sd[-1].isalpha() else _sd] = _r
        except Exception:                                    # noqa: BLE001
            _roles = {}
        if _dA > 3.0 or _dB > 3.0:
            _suspect = "A" if _dA > _dB else "B"
            _lead = (f"SIDE {_suspect}'S INTERFACE FLUX CHANGES BY {max(_dA, _dB):.0f}x "
                     f"ACROSS THE LEVELS while its partner's settles. A consistent "
                     f"outward flux converges under refinement; one that grows "
                     f"with the mesh is a derivative taken by hand on a field that "
                     f"is not converging, or a load applied on the wrong "
                     f"geometry, on side {_suspect}. ")
        else:
            _lead = ("THE TWO SIDES' INTERFACE FLUXES DIFFER BY MORE THAN 50x AT "
                     "EVERY LEVEL and neither changes across them, so one side is "
                     "applying or recovering a different quantity from the other "
                     "-- a density where a total is expected, a wrong normal, a "
                     "wrong material. ")
        _rec = ""
        _neu = [k for k, v in _roles.items() if v == "neumann"]
        if _neu:
            _rec = (f"Side {_neu[0]} declares itself the Neumann side, so it is "
                    f"the one that RECEIVES the flux; check that the imported "
                    f"flux entered its assembled system as a boundary integral "
                    f"and not as a nodal value that is never integrated "
                    f"(measured: a nodal flux with no face condition is ignored, "
                    f"the solve exits 0, and the field is bit-identical to a "
                    f"zero-flux run). ")
        out.append({"sequence": "interface flux transmission",
                    "values": [b for _l, _a, b in _bad][:3], "finding": (
            _lead + "All levels, from your own two interface files: " + _ratios
            + ". The two sides' outward fluxes must be equal and opposite at "
            "convergence, and a mismatch this size is not a material contrast "
            "-- a coefficient ratio changes the FIELD, not the flux that crosses "
            "the seam. " + _rec
            + "Recover each side's flux from that side's own assembled system "
            "and export it with its outward-normal sign.")})
    # BOTH SIDES ON THE SAME CONVENTION, TESTED WITHOUT A DERIVATIVE.
    #
    # The `inverted` branch below needs recover_normal_derivative, and on the
    # NEUMANN side that recovery is ill-conditioned: measured on the furthest
    # openPASO run of the coupled problem, side B's implied coefficient came out
    # None, +63.6 and -128.0 across the three levels, so only the last one
    # tripped `k < 0` and the finding named level 3 alone. Its field near the
    # seam is ~3e-3 with k = 200, which is why.
    #
    # The same defect has a signal that needs no derivative, no material
    # coefficient and no mesh -- only the two interface files. With opposite
    # normals |qA + qB| is discretisation error and |qA - qB| is ~2|q|; on the
    # same convention the two swap. Measured, sum/diff per level:
    #
    #     that run, both sides negative       89.2   272.4   1036.3
    #     a reference result set verified
    #       correct at order 1.9796            0.03    0.00     0.00
    #
    # Three orders of separation, and on the wrong convention the ratio GROWS
    # under refinement because its denominator is the shrinking discretisation
    # error while its numerator stays O(1). A threshold of 4 sits far from both.
    #
    # The run this comes from had already worked the rest out: its own report
    # says "Both sides report negative fluxes of similar magnitude (~0.8),
    # giving qn_A + qn_B = -1.6", and it called that "a persistent flux sign
    # convention issue [that] prevents completion" -- it read the defect as
    # physics to repair rather than a sign on a value being written out. So the
    # finding carries the line, not just the diagnosis.
    same_conv = []
    for lvl in sorted({l for l, _ in ifs}):
        a, b = ifs.get((lvl, "A")), ifs.get((lvl, "B"))
        if not (a and b):
            continue
        try:
            ga = _read_iface(a, _IF)
            gb = _read_iface(b, _IF)
            if not (ga and gb):
                continue
            ga, gb, _pw = _paired_iface(ga, gb)              # row i of each is the same point
            if not (ga and gb):
                continue
            qa = [c for row in ga[2] for c in row]
            qb = [c for row in gb[2] for c in row]
            n = min(len(qa), len(qb))
            if n == 0:
                continue
            s = max(abs(qa[i] + qb[i]) for i in range(n))
            d = max(abs(qa[i] - qb[i]) for i in range(n))
            # d == 0 IS THE DEFECT AT ITS MOST BLATANT, NOT A REASON TO SKIP.
            #
            # The first version of this branch required d > 0 so the ratio
            # would be finite, which made it silent on the one case that needs
            # no interpretation at all: a flux column written IDENTICALLY into
            # both sides' files. Caught by its own test, on a synthetic pair
            # built to be exactly that. Reported with an infinite ratio.
            if s > 0 and (d == 0 or s > 4.0 * d):
                same_conv.append((lvl, (s / d) if d > 0 else float("inf")))
        except Exception:
            continue
    if same_conv:
        where = ", ".join(
            f"level {l} (|sum|/|difference| = "
            + ("identical, the difference is exactly zero)" if r == float("inf")
               else f"{r:.0f})")
            for l, r in same_conv)
        out.append({"sequence": "interface flux convention",
                    "values": [r for _l, r in same_conv], "finding": (
            "BOTH SIDES REPORTED THEIR FLUX WITH THE SAME SIGN at " + where
            + ". The two subdomains use OPPOSITE outward normals at the same "
            "physical point, so qA + qB must be zero to discretisation error "
            "and |qA - qB| must be about twice |q|. Here it is the other way "
            "round: the sum is the big number and the difference is tiny. Two "
            "ways lead here, fixed in different places: a sign flipped in what "
            "one side WRITES OUT (its field is right, its file is not), or the "
            "partner's flux APPLIED with the wrong sign (then that side's field "
            "itself answers the wrong flux, and negating a written column only "
            "hides it -- measured: a run did exactly that and handed in a field "
            "of the wrong size as converged). The fields decide: the flux each "
            "side's own FIELD carries, -k du/dn outward, cancels its partner's in "
            "the first case and adds in the second, and the own-field flux "
            "finding reads it from the field dumps. (Kratos's FACE_HEAT_FLUX is "
            "the heat ENTERING a side.) Check after the change: the files and "
            "the fields must both cancel, and the sum must SHRINK from level to "
            "level, not grow.")})
    # A FLIPPED SIGN READS ABOUT -k, THE SAME AT EVERY LEVEL. Measured on a coupled
    # run whose side had never solved its interior: implied k -33.09, -8.736, -0.889
    # against a stated k of 1, and this finding said "THE DEFECT IS IN YOUR RECOVERY
    # FUNCTION" -- the served recovery, right all along; the run negated, checked and
    # reverted. Implied values that are negative but nowhere near -k say the flux
    # does not follow from the field at all, and that is what is named then.
    _stated = {}
    try:
        for _nm, _op in _side_operators(work).items():
            if isinstance(_op.get("k"), (int, float)) and _op.get("k"):
                _stated[_nm.replace("side_", "").upper()] = float(_op["k"])
    except Exception:                                       # noqa: BLE001
        _stated = {}
    def _sk(s):                     # the side a key names: "A|x = 0.4" (a leg of A) -> "A"
        return str(s).split("|")[0]

    def _st(s):                     # "A", or "A (leg x = 0.4)" for one leg of a bent interface
        return str(s) if "|" not in str(s) else f"{_sk(s)} (leg {str(s).split('|', 1)[1]})"

    _not_a_flip = []
    for _sd in sorted({s for _l, s, _k in inverted}):
        _ks = [k for _l, s, k in inverted if s == _sd]
        _ref = _stated.get(_sk(_sd).upper())
        _flip = (max(abs(k) for k in _ks) < 1.5 * min(abs(k) for k in _ks) if _ref is None
                 else all(abs(-k / _ref - 1.0) < 0.5 for k in _ks))
        if not _flip:
            _not_a_flip.append(_sd)
    if _not_a_flip:
        where = ", ".join(f"level {l} side {_st(s)} (implied k = {k:.4g})"
                          for l, s, k in inverted if s in _not_a_flip)
        _nf = list(dict.fromkeys(_sk(x) for x in _not_a_flip))
        out.append({"sequence": "interface flux sign", "values": [], "finding": (
            "THE INTERFACE FLUX DOES NOT FOLLOW FROM THIS SIDE'S FIELD at " + where + ": "
            "q_n divided by the field's own normal derivative is negative, but a flux "
            "computed with the inward normal reads about -k, the same number at every "
            "level, and these do not"
            + (" (stated k = " + ", ".join(f"{_stated[str(x).upper()]:g}" for x in _nf
                                          if str(x).upper() in _stated) + ")"
               if any(str(x).upper() in _stated for x in _nf) else "")
            + ". So this is not a sign to flip: the field or the flux is wrong. A side "
            "whose interior was never solved exports a flux of nothing in particular; look "
            "at its own field_level<k>.csv dumps and its solve before touching the sign.")})
        inverted = [t for t in inverted if t[1] not in _not_a_flip]
    # A FLUX THAT DOES NOT FOLLOW FROM ITS SIDE'S OWN FIELD, AT EVERY LEVEL. The
    # ratio above was used for its sign alone, and on a physically wrong run it read
    # INCONSISTENT on both sides at all three levels (implied coefficients 0.044 /
    # 0.19 / 0.091 where the side states k = 1) while the run was handed in with its
    # convergence order in band (measured). Over the 146 judged sides of recorded runs
    # whose order came out right, the two rules below fire on that run's two sides and
    # on no other; they fire on 105 sides of runs that were not.
    try:
        _ops = {n.replace("side_", "").upper(): o for n, o in _side_operators(work).items()}
    except Exception:                                    # noqa: BLE001
        _ops = {}
    # ONE k IS NOT HELD AGAINST LEGS THAT DIFFER. Where the legs of one side imply
    # coefficients more than 30 % apart at its finest level, a conductivity that differs by
    # region along the interface reads the same as a flux off on one leg, so the stated k
    # is compared with neither (see consolidated._res_coefficient_check).
    _legs_differ = set()
    for _sd in {_sk(x) for x in ratio_rows if "|" in str(x)}:
        _by = {}
        for _key, _rows in ratio_rows.items():
            if _sk(_key) == _sd and "|" in str(_key):
                for _l, _v, _k, _sp, _x in _rows:
                    _by.setdefault(_l, []).append(_k)
        _fin = _by.get(max(_by)) if _by else []
        if len(_fin) >= 2 and min(_fin) > 0 and max(_fin) > 1.3 * min(_fin):
            _legs_differ.add(_sd)
    for _s, _rows in sorted(ratio_rows.items()):
        if len(_rows) < 2 or any(_k <= 0 for _l, _v, _k, _sp, _x in _rows):
            continue                                     # a negative one is the sign finding's
        _ks = (_ops.get(_sk(_s).upper()) or {}).get("k")
        _ks = (float(_ks) if isinstance(_ks, (int, float)) and _ks > 0
               and _sk(_s) not in _legs_differ else None)
        _spread = all(_v == "INCONSISTENT" for _l, _v, _k, _sp, _x in _rows)
        _off = _ks is not None and all(abs(_k / _ks - 1.0) > 0.3 for _l, _v, _k, _sp, _x in _rows)
        if not (_spread or _off):
            continue
        _mis = _rows[0][4] == "misfit"
        _lv = " / ".join(f"{_k:.3g}" for _l, _v, _k, _sp, _x in _rows)
        # ranked after "THE FLUX YOU HANDED IN IS NOT THE FLUX YOUR SOLVER PRODUCED" (25):
        # where the hand-in copy is the defect, that one names where it is
        out.append({"sequence": f"interface flux from field side {_st(_s)}", "priority": 26,
                    "values": [_k for _l, _v, _k, _sp, _x in _rows], "finding": (
            f"THE INTERFACE FLUX SIDE {_st(_s)} DELIVERS DOES NOT FOLLOW FROM ITS OWN FIELD, at "
            f"every level ({', '.join(str(_l) for _l, _v, _k, _sp, _x in _rows)}): "
            + ((f"q_n differs from the best constant multiple of -du/dn by "
                f"{' / '.join(f'{_sp:.0%}' for _l, _v, _k, _sp, _x in _rows)}" if _mis else
                f"q_n / (-du/dn) along the interface varies by "
                f"{' / '.join(f'{_sp:.0%}' for _l, _v, _k, _sp, _x in _rows)} of its median")
               if _spread else "")
            + ("; " if _spread and _off else "")
            + (f"its median reads {_lv} where the side's config.json states k = {_ks:g}"
               if _off else f" (median {_lv})")
            + ". For a flux computed from this side's field, that ratio is the conductivity "
            "at every interface point: constant along the interface, and the k the side "
            "states. The exchange checks cannot see this -- they compare the two sides' "
            "exports with each other.")})
    if inverted:
        where = ", ".join(f"level {l} side {_st(s)} (implied k = {k:.4g})"
                          for l, s, k in inverted)
        out.append({"sequence": "interface flux sign", "values": [], "finding": (
            "INTERFACE FLUX HAS THE WRONG SIGN at " + where + ". Your "
            "reported q_n is proportional to your own field's normal "
            "derivative but NEGATIVE: it does not carry the outward sign the "
            "task defines, q_n = -(K grad u) . n_out with n_out pointing OUT of "
            "the subdomain. Which number is wrong, the fields decide: if the "
            "flux each side's own FIELD carries cancels its partner's, only the "
            "written sign is wrong -- correct the code that writes the flux, at "
            "every level and both sides; if the fields' fluxes add (the "
            "own-field flux finding says so), the field itself answers a flux "
            "of the wrong sign and the partner's flux was APPLIED with the "
            "wrong sign (Kratos's FACE_HEAT_FLUX is the heat entering the "
            "side) -- then flipping the written sign only hides it. Measured: "
            "one run corrected only the level a warning named and failed on a "
            "level it never re-checked; another flipped its written sign over "
            "a field of the wrong size and handed it in as converged.")})
    # A FIXED PROBE GRID HAS THE SAME ROW COUNT AT EVERY LEVEL.
    #
    # MEASURED, one recorded run: its coupling genuinely converged
    # (9.8e-07 in 21 iterations at the finest level) and its interface files
    # carry 9, 17 and 33 rows across the three levels -- its own mesh nodes,
    # which change under refinement -- against a contract that fixes the probe
    # points once for all levels. Everything it computed was thrown away on
    # the sampling. The signal needs no task knowledge: rows that GROW with
    # the level are a mesh trace; a fixed grid cannot do that.
    rowcounts: dict = {}
    for (lvl, side), path in sorted(ifs.items()):
        try:
            g = _read_iface(path, _IF)
        except Exception:
            g = None
        if g is not None:
            rowcounts.setdefault(side, {})[lvl] = len(g[0])
    for side, per in sorted(rowcounts.items()):
        ns = [per[l] for l in sorted(per)]
        if len(ns) >= 2 and len(set(ns)) > 1 and all(
                b > a for a, b in zip(ns, ns[1:])):
            out.append({"sequence": f"interface rows side {side}",
                        "values": ns, "finding": (
                f"INTERFACE ROWS GROW WITH THE LEVEL on side {side}: "
                + ", ".join(str(n) for n in ns) + " rows across the levels. "
                "The interface probe points are FIXED -- the same points at "
                "every mesh level -- so every per-level interface file "
                "must have the SAME rows in the same order. A growing count "
                "means you wrote your own mesh nodes instead of evaluating "
                "(interpolating) your solution AT the prescribed points. "
                "Re-read the task's INTERFACE PROBE POINTS line and evaluate "
                "your existing solution there; no re-solve is needed.")})
            break
    # THE RESIDUAL YOU CONVERGED MUST BE THE DISAGREEMENT IN YOUR FILES.
    #
    # MEASURED, one recorded run: residual_level3.csv ends at 2.3162e-08
    # after 7 iterations, while the exported interface files disagree by
    # max|uA-uB| = 3.65e-03 -- IDENTICAL at all three levels -- and the flux
    # sum by ~0.8. Five orders between what the iteration measured and what
    # the result set contains means the iteration converged some OTHER
    # quantity (a different set of points, a previous iterate, one side's
    # internal state) than the fields that were written out. The observed
    # order was 0.1830 and nothing in the run said why.
    resid_final: dict = {}
    for f in _history_files(work):
        if _side_of(f):
            continue
        try:
            rows = [r for r in f.read_text(errors="replace").splitlines()
                    if r.strip()][1:]
            resid_final[_level_of(f)] = abs(float(rows[-1].split(",")[-1]))
        except Exception:
            continue
    mismatch = []
    columns: list = []
    for lvl, claimed in sorted(resid_final.items()):
        a, b = ifs.get((lvl, "A")), ifs.get((lvl, "B"))
        if not (a and b) or claimed <= 0:
            continue
        try:
            ga = _read_iface(a, _IF, want_flux=False)
            gb = _read_iface(b, _IF, want_flux=False)
            if not (ga and gb):
                continue
            ga, gb, _pw = _paired_iface(ga, gb)
            if not (ga and gb):
                continue
            ua = [v for row in ga[1] for v in row]
            ub = [v for row in gb[1] for v in row]
            n = min(len(ua), len(ub))
            if n == 0:
                continue
            # A FLOOR OF 1e-300 IS NOT A SCALE, IT IS A DIVISION BY ZERO WITH
            # THE TRACEBACK SUPPRESSED.
            #
            # On a Dirichlet-Neumann split the Dirichlet side exports no field
            # values at all -- `"values": []` by the served contract -- so `ua`
            # came back all zeros, the floor took over, and this check told an
            # agent its files disagreed by 1.05e+298 at all three levels
            # (measured on one run). A number like that is not a
            # finding; it is a defect wearing a finding's clothes, and the
            # agent spent four calls on it.
            #
            # No scale means no relative jump. Say nothing here rather than
            # something absurd: the one-sided checks and the flux comparison
            # still speak, and a silent check beats a check that invents 298
            # orders of magnitude.
            scale = max(abs(v) for v in ua[:n])
            if scale <= 1e-12:
                continue
            jump = max(abs(ua[i] - ub[i]) for i in range(n)) / scale
            if jump > 100.0 * claimed and jump > 1e-3:
                mismatch.append((lvl, claimed, jump))
                columns.append(_iface_column_diagnosis(a, b, ga, gb))
        except Exception:
            continue
    if mismatch:
        where = ", ".join(f"level {l}: claimed {c:.2e} vs measured {j:.2e}"
                          for l, c, j in mismatch)
        col_txt = next((c for c in columns if c), "")
        out.append({"sequence": "residual vs files", "values":
                    [j for _l, _c, j in mismatch], "finding": (
            "THE RESIDUAL YOUR ITERATION CONVERGED IS NOT THE DISAGREEMENT "
            "IN YOUR FILES (" + where + "). The relative field jump computed "
            "from your own two per-level interface files is orders of magnitude "
            "above the final value in that level's residual history, so the quantity "
            "your coupling loop measured is not the quantity you exported -- "
            "a different point set, a stale iterate, or one side's internal "
            "state. Recompute the mismatch FROM THE TWO FILES you are about to "
            "deliver -- max|uA-uB| over the interface rows, divided by "
            "max|uA| -- and iterate on THAT; if it does not match your "
            "loop's residual, your loop is reading different data than it "
            "writes. THIS IS NOT FIXED BY MAKING THE TWO COLUMNS THE SAME: "
            "writing one side's trace into both files sets this number to "
            "exactly zero and destroys the only evidence that the two codes "
            "ever agreed on anything (measured -- a run did that, its two "
            "interface files became bit-identical, and its coupling could no "
            "longer be shown at all). The two files are two SOLVERS' answers; "
            "they are supposed to differ by a little."
            + col_txt)})
    trend = [jumps[l] for l in sorted(jumps)
             if isinstance(jumps.get(l), (int, float))]
    if trend and max(trend) < _JUMP_ROUND_OFF:
        trend = []                       # round-off: the two fluxes agree, nothing to judge
    # ONE CHANNEL THAT MISBEHAVES IS NOT A RECOVERY DEFECT. The two findings
    # below read the jump of ALL flux columns as one number and, when it fell
    # too slowly, named the interface recovery -- a benign-sounding cause with
    # a recipe. Measured on a wrong thermo-mechanical run: the heat-flux jump
    # fell 4x per level (orders 2.10, 2.01) while the two traction jumps fell
    # 1.6x and 1.2x, then 1.1x -- and the run handed in its displacements,
    # ten times too large, with "first-order flux recovery, a known
    # limitation" copied from this audit. On the correct sibling every
    # channel fell 4x; on a third run the heat-flux jump GREW while the
    # tractions fell 4x. A recovery method acts on every column of a side
    # alike, so channels that part ways are not a recovery problem: the
    # channel that does not shrink carries a wrong sign, scaling or missing
    # term in ITS exchange. Say which, with the numbers, and do not offer the
    # recovery as the cause.
    _mixed = _mixed_channel_finding(jumps_c, chan_names)
    if _mixed:
        out.append(_mixed)
    elif len(trend) >= 2 and not all(trend[i + 1] < trend[i]
                                     for i in range(len(trend) - 1)):
        out.append({"sequence": "interface flux jump", "values": trend,
                    "finding": (
            "THE FLUX JUMP DOES NOT SHRINK under refinement (" +
            ", ".join(f"{v:.3e}" for v in trend) + "). A jump that does not "
            "shrink as h halves is not discretisation error, and a clean "
            "convergence order cannot reveal it. This measures the jump, not its "
            "cause. Causes on record: a wrong sign or scaling of an exchanged "
            "flux, the partner's data applied by row index instead of by position "
            "(row i of the partner's list is not point i of this side's interface "
            "when the two list their points in different orders), a boundary not "
            "held where the problem holds it, and a side "
            "whose linear system was not solved (a KSP 'preonly' without a direct "
            "factorisation applies one preconditioner sweep: |b - Ax|/|b| 0.6-0.9 "
            "on every level, measured). Compare, at one level, the flux each "
            "side's own field carries with the flux it exports.")})
    elif len(trend) >= 3 and all(trend[i + 1] < trend[i]
                                 for i in range(len(trend) - 1)):
        # THE JUMP SHRINKS, BUT TOO SLOWLY = A FIRST-ORDER INTERFACE
        # RECOVERY. On the prescribed halving a consistent (2nd-order)
        # recovery drops the two-sided flux jump ~4x per level; a jump that
        # only halves (~2x, order ~1) or worse means the flux one side
        # exports is not assembly-consistent with how it was applied --
        # the classic apply-with-one-quadrature / recover-with-another
        # mismatch (measured on a 4C-Neumann pair: applied Simpson nodal
        # loads on a QUAD4 solve, recovered by a P1-triangle re-assembly ->
        # jump fell only 1.5-1.8x per level, order ~0.8, and the coupled
        # field order was capped there). It converges cleanly to the wrong
        # fixed point, so neither the residual nor the sign check sees it.
        import math as _m
        orders = [_m.log2(trend[i] / trend[i + 1])
                  for i in range(len(trend) - 1) if trend[i + 1] > 0]
        med = sorted(orders)[len(orders) // 2] if orders else 0.0
        if med < 1.3:
            out.append({"sequence": "interface flux jump", "values": trend,
                        "finding": (
                "THE FLUX JUMP SHRINKS TOO SLOWLY (" +
                ", ".join(f"{v:.3e}" for v in trend) + f"; order ~{med:.2f} "
                "per halving, against ~2 for a consistent recovery). The "
                "flux one side exports is not assembly-consistent with how "
                "the partner applied it -- recover it with the SAME "
                "discretisation the solver used (the code's native boundary "
                "flux, or a re-assembly with the SAME element and quadrature "
                "as the solve), not a hand-rolled stand-in on a different "
                "element. A first-order interface recovery caps the coupled "
                "field's order however good the solves are.")})
    if tensor_skipped:
        _ts = sorted(tensor_skipped)
        out.append({"sequence": "interface flux sign", "values": [], "informational": True,
                    "finding": (
            f"THE FLUX-FROM-FIELD AND SIGN CHECKS DID NOT RUN (NOT CHECKED) for "
            f"{' and '.join('side ' + s for s in _ts)}: "
            + "; ".join(f"{_tensor[s][0]}'s config.json {_tensor[s][1]}" for s in _ts)
            + ". With a full tensor K the outward flux -(K grad u) . n takes in the "
            "derivative along the interface as well, so q_n / (-du/dn) is not one "
            "constant there, and neither the flux's sign nor its ratio to the field is "
            f"judged for {'that side' if len(_ts) == 1 else 'those sides'}. The checks "
            "that compare the two sides' files still do.")})
    _unknown_note = None
    if unknown_skipped:
        _us = sorted(unknown_skipped)
        _unknown_note = ({"sequence": "interface flux sign", "values": [], "informational": True,
                    "finding": (
            f"THE FLUX-FROM-FIELD AND SIGN CHECKS DID NOT RUN (NOT CHECKED) for "
            f"{' and '.join('side ' + s for s in _us)}: "
            + "; ".join(f"{_unknown[s]}'s config.json states no k, and its program does not "
                        f"say whether K is one number or a matrix" for s in _us)
            + ". q_n / (-du/dn) is one constant along the interface only where the conductivity "
            "is one number; with a tensor K the outward flux -(K grad u) . n takes in the "
            "derivative along the interface as well. Without the coefficient neither the flux's "
            f"sign nor its ratio to the field is judged for {'that side' if len(_us) == 1 else 'those sides'}. "
            "The checks that compare the two sides' files still do.")})
    # a side left unjudged for another reason is still said so, beside the note above
    _left = {str(s).upper() for _l, s in ifs} - tensor_skipped - unknown_skipped
    if not out and assessed == 0 and (_left or _unknown_note is None):
        if vector_layout:
            # Say WHY, accurately. Measured: a correct thermo-mechanical
            # result set with every prescribed file on disk was told to
            # "write solution_level<k>_<side>.csv" by this branch -- the
            # files existed; the sign heuristic had skipped every interface
            # because the trace is multi-component, and the message blamed
            # the wrong thing.
            out.append({"sequence": "interface flux sign", "values": [],
                        "informational": True,
                        "finding": (
                "THE SCALAR FLUX-SIGN HEURISTIC DOES NOT APPLY to this "
                "interface: its trace carries more than one field component, "
                "so k = q/(-du/dn) would pair the wrong columns and was not "
                "attempted. This is a statement of scope, not a defect in "
                "your result set: interface balance is still checked "
                "component-by-component against the two sides' files, and "
                "any finding from that check appears separately.")})
        elif bent_skipped:
            # A BENT INTERFACE THAT COULD NOT BE READ LEG BY LEG is a statement of
            # what this check can do, not of a missing file.
            _l, _s, _w = bent_skipped[0]
            out.append({"sequence": "interface flux sign", "values": [],
                        "informational": True,
                        "finding": (
                f"THE INTERFACE FLUX SIGN CHECK DID NOT RUN (NOT CHECKED) on this bent "
                f"interface: at level {_l}, side {_s}: {_w}. The check reads q_n / "
                f"(-du/dn) leg by leg, on legs along x or y with the side's own field "
                f"beside them, so nothing here is judged.")})
        else:
            out.append({"sequence": "interface flux sign", "values": [],
                        "finding": (
                "THE INTERFACE FLUX SIGN COULD NOT BE CHECKED: your interface "
                "files were read but the flux could not be compared with your "
                "own field. Write the per-level field file for every "
                "level and side on the prescribed probe grid, and this check "
                "becomes available. This is NOT a clean bill.")})
    if _unknown_note:
        out.append(_unknown_note)
    return out


def export_findings(work: Path) -> list[dict]:
    """Two defects that live in the EXPORT, not the solve, and cap the outcome.

    A perfect solve reported through a broken export reads as badly as a
    wrong solve, and neither the solver log nor a refinement study can see it.
    Both of these were reproduced by execution against real result sets.

    (1) NEAREST-NODE SAMPLING INSTEAD OF INTERPOLATION. This one is real, and
        it is the largest single recoverable defect measured across the
        recorded runs: 99 runs with openPASO and 86 without carry the
        fingerprint. The tasks prescribe FIXED probe points that are
        deliberately not mesh nodes. Answering with the value at the closest
        node is O(h) accurate, so it caps the reported order at 1 however good
        the solve is. Measured on one 4C problem: the same solve gave order
        +1.9516 read by bilinear interpolation and +1.0179 read by nearest
        node -- and +1.0179 is exactly what the result set reported.

        The fingerprint is free: with a mesh of N cells per side, nearest-node
        sampling can only ever return (N-1)^2 + 1 distinct interior values, so
        1936 probe points collapse onto 50, 226 and 962 distinct values at
        N = 8, 16, 32. Measured on four result sets from two tasks -- all four
        show exactly 50/1936, 226/1936, 962/1936. A result set that
        interpolates shows 1908/1928/1936.

    (2) ROW ORDER IS **NOT** CHECKED, AND MUST NOT BE. It looks like a defect
        and is not one. An independent check pairs each delivered row with
        the reference EVALUATED AT THAT ROW'S OWN COORDINATES -- the shared
        field-error check does `for p, v in zip(pts, vals)` -- so a
        transposed file is read point by point exactly like an ordered one.

        Verified against an independent reference, not argued: the 4C+Kratos
        reference is correct at order 1.9796; the SAME result set with the row
        order transposed is correct at order 1.9796, bit-identical. A check on
        row order would have flagged 146 runs with openPASO and 117 without -- a
        third of the recorded runs -- and sent every one of them to fix
        something that costs nothing, spending the action budget that is
        already the binding constraint. It was written, measured, and removed.
    """
    import csv as _csv
    import math as _math
    import re as _re

    findings: list[dict] = []
    per_level: dict[int, tuple] = {}
    for f in _field_files(work):
        m = _re.search(r"level(\d+)", f.name)
        if not m:
            continue
        try:
            rows = [r for r in _csv.reader(f.open())
                    if r and not r[0].strip().startswith(("x", "#"))]
        except OSError:
            continue
        xs, ys, vals = [], [], []
        for r in rows:
            if len(r) < 3:
                continue
            try:
                xs.append(float(r[0])); ys.append(float(r[1]))
                vals.append(float(r[2]))
            except ValueError:
                continue
        if len(vals) < 100:
            continue
        key = int(m.group(1))
        # keep the largest file per level, so a stale partial does not decide
        if key not in per_level or len(vals) > len(per_level[key][2]):
            per_level[key] = (xs, ys, vals, f.name)

    # REQUIRE THE ARITHMETIC SIGNATURE, AT MORE THAN ONE LEVEL.
    #
    # The first version fired whenever distinct*2 < n at ANY level, and that is
    # far too loose. Measured against an independent reference on 25 runs of
    # one task carrying that looser flag: EIGHT of them are verified correct
    # at order 2.0045, 1.9884, 1.9754 and 1.8426. Their real distinct counts
    # are 7974-8585 of 9261 -- 86 to 93
    # per cent -- and the flag came from ONE coarse level collapsing to a single
    # value, which a genuine nearest-node export never does. Nearest-node
    # sampling collapses EVERY level, and it collapses them lawfully: on a mesh
    # of N cells per side it returns exactly (N-1)^2+1 distinct interior values.
    # Two runs hit 50, 226, 962 out of 1936 -- 7^2+1, 15^2+1,
    # 31^2+1 -- at all three levels.
    #
    # So the test is the exact signature, at two levels or more. Accusing eight
    # correct result sets to catch four defective ones is a worse trade than
    # missing some, and it is the same error as the row-order check that was
    # written, measured and deleted.
    hits = []
    for lvl, (xs, ys, vals, name) in sorted(per_level.items()):
        n = len(vals)
        distinct = len({round(v, 12) for v in vals})
        root = _math.isqrt(max(distinct - 1, 1))
        if distinct > 1 and root * root + 1 == distinct and distinct * 4 < n:
            hits.append((lvl, name, distinct, n))
    if len(hits) < 2:
        hits = []
    for lvl, name, distinct, n in hits:
        if True:
            # (N-1)^2+1 for the mesh that would explain it, reported so the
            # agent can recognise its own mesh
            nn = int(_math.isqrt(max(distinct - 1, 1))) + 1
            findings.append({"sequence": name, "values": [distinct, n],
                             "finding": (
                f"ONLY {distinct} DISTINCT VALUES ACROSS {n} PROBE POINTS at "
                f"level {lvl}. The probe points are deliberately NOT mesh "
                f"nodes, so a correct export gives almost {n} distinct values; "
                f"{distinct} is what NEAREST-NODE SAMPLING returns on a mesh of "
                f"about {nn} cells per side, because it can only ever produce "
                f"(N-1)^2+1 interior values. Nearest-node lookup is O(h), so it "
                f"CAPS YOUR REPORTED ORDER AT 1 however good the solve is. "
                f"Verified against an independent reference: one coupled run "
                f"gives order 1.9796 by interpolation, and the SAME SOLVE "
                f"re-exported by nearest-node lookup -- nothing else changed -- "
                f"comes out at order 0.9815. A separate case gave "
                f"+1.9516 interpolated against +1.0179 nearest-node, and +1.0179 "
                f"is exactly what that run reported. Interpolate inside "
                f"the element that CONTAINS each probe point; this is a "
                f"post-processing fix and does not need the solver re-run.")})
    return findings


def probe_sampling_findings(work: Path) -> list[dict]:
    """Nearest-node sampling, read off the delivered files themselves.

    A probe file made by copying the nearest node's value holds at most
    (N-1)^2 + 1 distinct values for a mesh of N cells per side, however many
    probes it lists; interpolation with the element's shape functions gives
    close to one distinct value per probe. Measured on a coupled run whose
    every field on both sides self-converged at order ~1.0 with interface
    channels falling 4x per level: 36, 151 and 621 distinct values for 1936
    probes at the three levels, from an argmin over node distances -- the
    order was the sampler's, not the solve's, and the audit had named the
    interface recovery for it. The served knowledge tells an agent to count
    its distinct values; this counts them.
    """
    out: list[dict] = []
    seen_sides: set = set()
    for q in sorted(_field_files(work)):
        side = (_side_of(q) or "").upper()
        k = _level_of(q)
        try:
            rows = [r for r in q.read_text(errors="replace").splitlines() if r.strip()]
        except OSError:
            continue
        if len(rows) < 101:
            continue
        hdr = [c.strip() for c in rows[0].split(",")]
        try:
            cols = list(zip(*[[float(c) for c in r.split(",")] for r in rows[1:]]))
        except ValueError:
            continue
        if len(cols) != len(hdr):
            continue
        n = len(rows) - 1
        worst = None
        for name, col in zip(hdr, cols):
            if name.lower() in _IFACE_COORD_NAMES:
                continue
            d = len({round(v, 12) for v in col})
            if worst is None or d < worst[1]:
                worst = (name, d)
        if worst is None or worst[1] >= 0.5 * n:
            continue
        key = side or q.name
        if key in seen_sides:
            continue
        seen_sides.add(key)
        name, d = worst
        # A FIELD CONSTANT OVER A REGION HAS FEW VALUES TOO, and this finding
        # named nearest-node sampling for it. Measured on a coupled run whose
        # probes were interpolated linearly from the dumps: 382 distinct values
        # over 1936 probes because the side's interior was exactly 0.0 -- never
        # solved -- and most probes read that one value. A nearest-node file
        # spreads its probes over its node values instead (the measured one: 36
        # values for 1936 probes, none of them a large share).
        wcol = cols[hdr.index(name)]
        from collections import Counter as _Counter
        mode_v, mode_n = _Counter(round(v, 12) for v in wcol).most_common(1)[0]
        if mode_n >= 0.3 * n:
            out.append({"sequence": f"probe sampling {q.name}", "priority": 4,
                        "values": [float(d), float(n), float(mode_n)], "finding": (
                f"ONE VALUE FILLS {q.name}: {name} reads exactly {mode_v:g} at {mode_n} of "
                f"its {n} probe points (level {k}, side {side or '?'}), and only {d} distinct "
                f"values overall. A solved field with non-zero data is not constant over a "
                f"region; when that value is 0.0 the region was never solved (look at the "
                f"side's own field_level<k>.csv dump and the free-dof mask its solve "
                f"inverts on) or was overwritten after the solve. This is not a sampling "
                f"signature: interpolating such a field correctly still gives these values.")})
            continue
        out.append({"sequence": f"probe sampling {q.name}", "priority": 10,
                    "values": [float(d), float(n)], "finding": (
            f"NEAREST-NODE SAMPLING IN {q.name}: {name} takes only {d} distinct "
            f"values over {n} probe points (level {k}, side {side or '?'}). A "
            f"file interpolated with the element's shape functions carries close "
            f"to one distinct value per probe; one that copies the nearest node's "
            f"value carries at most (N-1)^2 + 1 for N cells per side, and its "
            f"error is O(h) whatever the solve's order -- every field, both "
            f"sides, one order, which is not a defect of the coupling. "
            f"Interpolate each probe inside its element (the field file plus "
            f"scipy.interpolate.griddata(..., method='linear') is enough for a "
            f"P1 field; the solver's own point evaluation is exact) and rewrite "
            f"every per-level file from the field, never from the nearest node.")})
    return out


def interface_point_set_findings(work: Path) -> list[dict]:
    """The interface files of one side must sample the SAME points at every
    level. The task's interface probes are one fixed set; a level reported at
    its own mesh nodes is comparable neither with the other levels nor with
    the prescribed points. Measured on a coupled run that had converged three
    real levels: levels 1 and 2 held 44 rows on the prescribed band, level 3
    held 31 rows at its mesh nodes over the whole interface, written by hand
    at the wall -- and the hand-in was refused for it while no check here had
    said a word (the row check fires only on growth, the ends check missed
    by half a spacing). Reads only the agent's own files."""
    by_side: dict = {}
    for q in _interface_files(work, sided=True):
        k, side = _level_of(q), _side_of(q)
        if k is None or not side:
            continue
        got = _read_iface(q, None) if False else _read_iface_by_header(q)
        if got is None:
            continue
        pts = {tuple(round(float(c), 9) for c in p) for p in got[0]}
        if pts:
            by_side.setdefault(side.upper(), {})[k] = pts
    out: list[dict] = []
    for side, levels in sorted(by_side.items()):
        if len(levels) < 2:
            continue
        ks = sorted(levels)
        ref = levels[ks[0]]
        odd = [k for k in ks[1:] if levels[k] != ref]
        if not odd:
            continue
        def _span(pts):
            ys = sorted({p[-1] for p in pts})
            return f"{len(pts)} points on [{ys[0]:.3g}, {ys[-1]:.3g}]"
        out.append({"sequence": f"interface point set side {side}", "priority": 20,
                    "values": [float(len(levels[k])) for k in ks], "finding": (
            f"YOUR INTERFACE FILES SAMPLE DIFFERENT POINTS AT DIFFERENT LEVELS on side "
            f"{side}: level {ks[0]} holds {_span(ref)}; "
            + "; ".join(f"level {k} holds {_span(levels[k])}" for k in odd)
            + ". The interface probes a task prescribes are ONE fixed set, the same at "
            f"every level; a level reported at its own mesh nodes is comparable neither "
            f"with the other levels nor with the prescribed points. Interpolate that "
            f"level's per-side interface dump (interface_level<k>.csv, trace and flux at "
            f"its nodes) along the interface to the same points the other levels use, "
            f"and never write a level's interface file by hand.")})
    return out


def interface_ends_findings(work: Path) -> list[dict]:
    """Interface rows that reach the ends of the interface, before delivery.

    Physics, not contract: where a partitioned interface meets the outer
    boundary the split problem has a Dirichlet-Neumann corner, and the
    recovered flux there does not converge under refinement — measured in
    this corpus (2.11x -> 2.51x the true value over a 4x refinement).
    Prescribed probe sets therefore exclude the interface ends. A file
    whose interface rows run to the very ends of the interface is the
    signature of a SELF-CHOSEN uniform sampling of the whole interface —
    measured twice: two converged couplings counted for nothing because all
    44 of their rows sat at self-chosen coordinates covering the full span,
    including the excluded ends.
    """
    import re as _re
    dom = {}
    for q in _field_files(work):
        try:
            rows = q.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for ln in rows[1:5000]:
            parts = ln.split(",")
            try:
                x, y = float(parts[0]), float(parts[1])
            except (ValueError, IndexError):
                continue
            for ax, v in ((0, x), (1, y)):
                lohi = dom.setdefault(ax, [v, v])
                lohi[0] = min(lohi[0], v); lohi[1] = max(lohi[1], v)
    if not dom:
        return []
    out = []
    for q in _interface_files(work, sided=True):
        try:
            rows = q.read_text(errors="replace").splitlines()[1:]
        except OSError:
            continue
        pts = []
        for ln in rows:
            parts = ln.split(",")
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except (ValueError, IndexError):
                continue
        if len(pts) < 4:
            continue
        # A BENT INTERFACE HAS ITS ENDS WHERE ITS LEGS END, not at one axis's extremes.
        # Measured: an exactly right two-leg set was told its rows run to the ends.
        try:
            _legs, _why = _curve_legs(pts)
        except Exception:                                # noqa: BLE001
            _legs, _why = None, None
        if _legs or _why:
            continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        var_ax = 0 if (max(xs) - min(xs)) > (max(ys) - min(ys)) else 1
        vals = xs if var_ax == 0 else ys
        lo, hi = dom.get(var_ax, (None, None))
        if lo is None:
            continue
        span = hi - lo
        if span <= 0:
            continue
        sv = sorted(vals)
        gaps = [b - a for a, b in zip(sv, sv[1:]) if b > a]
        row_dx = (sorted(gaps)[len(gaps)//2] if gaps else 0.02 * span)
        margin = 0.5 * row_dx      # only rows essentially AT the extremes
        if min(vals) < lo + margin or max(vals) > hi - margin:
            out.append({"sequence": q.name, "informational": True, "values":
                        [min(vals), max(vals), lo, hi], "finding": (
                f"YOUR INTERFACE ROWS RUN TO THE ENDS OF THE INTERFACE "
                f"({min(vals):.4g}..{max(vals):.4g} against a domain span "
                f"{lo:.4g}..{hi:.4g}). The recovered flux at the points "
                f"where the interface meets the outer boundary does not "
                f"converge (Dirichlet-Neumann corner — measured 2.1x-2.5x "
                f"the true value, worsening under refinement), so "
                f"probe prescriptions typically EXCLUDE the ends. If your task "
                f"prints an interface probe formula, re-check these rows "
                f"against it verbatim; rows at the extremes have twice been "
                f"the signature of a self-chosen uniform sampling on runs "
                f"whose coupling itself was sound.")})
            break
    return out


def unlaunched_participants_findings(work: Path) -> list[dict]:
    """Participants built, coupling never run — stated at DELIVERY, not only
    at give-up.

    Measured: two sessions wrote exports-contract participant scripts (4 and
    6 of them), never drove the coupling iteration, and DELIVERED — so the
    give-up gate, which already states exactly this, never saw them. The
    statement is structural and belongs on every path that reads the
    workdir: scripts implementing the exchange exist, no partitioned
    iteration history exists anywhere, therefore the one step that turns
    this work into coupling evidence was never taken.
    """
    if _history_files(work):
        return []
    pscripts = []
    for q in sorted(q for q in work.rglob("*.py") if not _is_backup(q)):
        try:
            c = q.read_text(errors="replace")
        except OSError:
            continue
        if "exports.json" in c and ("imports.json" in c
                                    or "InterfaceData" in c):
            pscripts.append(q.name)
    if len(pscripts) < 2:
        return []
    if not _interface_files(work) and not any(
            work.rglob("exports.json")):
        return []            # no sign this workdir is a coupling at all
    # WHAT IS MEASURED IS THE MISSING HISTORY, NOT A COUPLING THAT NEVER RAN.
    # couple() writes a history file only when it is given history_path; a
    # coupling that converged without one leaves none, and this finding then
    # told a run whose coupling had just converged that it "was never run".
    return [{"sequence": "no coupling history on disk", "values": [], "finding": (
        f"{len(pscripts)} script(s) implement the imports/exports participant "
        f"exchange ({', '.join(pscripts[:4])}) and no per-level residual "
        f"history exists anywhere in this directory, so nothing on disk shows "
        f"the coupling iteration over them ran. couple() writes that file only "
        f"when it is given history_path: couple each level with the history "
        f"file name your task gives (couple_levels takes it as history_pattern), "
        f"or, if the coupling has not run yet, run it now.")}]



def _halving_pair(n1: int, n2: int):
    """(a, b) with (a+1)(b+1) == n1 and (2a+1)(2b+1) == n2 -- the one way a
    2-D tensor grid of a x b cells halves into 2a x 2b -- or None. An even n2
    is not a halved tensor grid at all (2a+1 and 2b+1 are odd), so None there
    means "cannot say", not "wrong"."""
    if n1 <= 0 or n2 <= 0 or n2 % 2 == 0:
        return None
    p = 1
    while p * p <= n2:
        if n2 % p == 0 and (n2 // p) % 2 == 1:
            a, b = (p - 1) // 2, (n2 // p - 1) // 2
            if (a + 1) * (b + 1) == n1:
                return (a, b)
        p += 2
    return None


def _odd_grids(n: int) -> list:
    """The (cells_x, cells_y) tensor grids with n nodes, both counts even (a
    halved grid), squarest first; for naming what a node count reads as."""
    out = []
    p = 3
    while p * p <= n:
        if n % p == 0 and (n // p) % 2 == 1:
            out.append((p - 1, n // p - 1))
        p += 2
    return sorted(out, key=lambda ab: abs(ab[0] - ab[1]))


def exact_ladder_findings(work: Path) -> list[dict]:
    """A level that is provably not the halving of its neighbour, from the
    NDOF lines alone, where the same side proves it runs tensor grids.

    MEASURED on a coupled run whose every other verdict was clean and whose
    order still came out near 0.4. One side's run logs read NDOF 81, 255, 957.
    The two finer counts are the consecutive halvings of one base mesh;
    the coarsest, a square count, is a different mesh (the same shape with
    round numbers: 91 = 7 x 13 and 325 = 13 x 25 halve one base mesh, and
    64 = 8 x 8 is not on that ladder). No (a, b) satisfies (a+1)(b+1) = n1
    and (2a+1)(2b+1) = n2 there, so level 1 was not the mesh level 2 halves, and the
    order across levels 1-2 meant nothing. The band check above read the 3.15x
    step as fine: for coarse grids the +1 terms pull a true halving down to
    ~3.5x, and a band that admits that admits this. Over the nine CORRECT
    coupled cells of the two closed problems, 18 of 18 side-steps factor
    exactly; 9522's B1->2 is the one that does not.

    It leads the self-convergence finding (priority 55): when a level is off
    the ladder the order computed across it is meaningless, so "your interface
    recovery is first-order" is a misattribution -- measured on 9522, whose
    selfdiff read 1.32 and whose real defect is the 81-node level 1. It sits
    below the equation check (26): a field that does not solve its own PDE is
    the deeper defect.

    THE GUARD: a side is judged only where at least one of its own steps IS an
    exact halving -- proof that it runs tensor grids. An unstructured mesh's
    counts are arbitrary, its steps never factor, and it is never judged; the
    band check keeps covering it. Two levels only, one bad step, no proof:
    silent. Odd counts by chance cannot fire this: firing needs a consistent
    step beside the inconsistent one.
    """
    per_side: dict[str, dict[int, int]] = {}
    for q in _level_logs(work):
        try:
            txt = q.read_text(errors="replace")
        except OSError:
            continue
        nm = None
        for mm in _DOF_LINE.finditer(txt):
            nm = int(mm.group(1))
        if nm:
            lv = per_side.setdefault(_side_of(q), {})
            lv[_level_of(q)] = lv.get(_level_of(q), 0) + nm
    out = []
    for side, lv in sorted(per_side.items()):
        ks = sorted(lv)
        if len(ks) < 2:
            continue
        steps = [(a, b, _halving_pair(lv[a], lv[b])) for a, b in zip(ks, ks[1:])]
        if not any(pair for _a, _b, pair in steps):
            continue                       # nothing proves this side is a tensor grid
        who = f"side {side}" if side else "the run"
        for i, (a, b, pair) in enumerate(steps):
            if pair is not None or lv[b] % 2 == 0:
                continue
            reads = _odd_grids(lv[b])[:2]
            reads_txt = " or ".join(
                f"{cx}x{cy} cells ({cx + 1}x{cy + 1} nodes), which halves a "
                f"{cx // 2}x{cy // 2} mesh of {(cx // 2 + 1) * (cy // 2 + 1)} nodes"
                for cx, cy in reads) or "no tensor grid at all"
            # which of the two is the odd one out: the level whose OTHER
            # neighbour agrees with it is right, the other is wrong
            nxt = steps[i + 1] if i + 1 < len(steps) else None
            prv = steps[i - 1] if i > 0 else None
            if nxt and nxt[2] is not None:
                cx, cy = nxt[2]           # level b is cx x cy cells
                wrong, want = a, f"{cx // 2}x{cy // 2} cells ({(cx // 2 + 1) * (cy // 2 + 1)} nodes), half of level {b}'s {cx}x{cy} in each direction"
            elif prv and prv[2] is not None:
                cx, cy = prv[2]           # level a is 2cx x 2cy cells
                wrong, want = b, f"{4 * cx}x{4 * cy} cells ({(4 * cx + 1) * (4 * cy + 1)} nodes), double level {a}'s {2 * cx}x{2 * cy} in each direction"
            else:
                wrong, want = None, None
            finding = (
                f"YOUR OWN LOGS PROVE ONE LEVEL IS OFF THE LADDER: on {who}, level "
                f"{b} has {lv[b]} nodes, which reads as {reads_txt}; level {a} has "
                f"{lv[a]} nodes. No mesh halves from {lv[a]} nodes to {lv[b]}: the "
                f"other steps on this side halve exactly, this one cannot. "
                + (f"Level {wrong} is the one off the ladder. Re-run level {wrong} alone -- "
                   f"one couple call -- on {want}, and rewrite level {wrong}'s field, "
                   f"interface and history files from that run."
                   if wrong is not None else
                   f"With only these two levels the logs cannot say which of the two is "
                   f"off; the task's own ladder decides. Re-run the wrong one alone -- one "
                   f"couple call -- and rewrite its field, interface and history files.")
                + " A convergence order across a step that is not a halving is not "
                "an order -- measured: a run whose every other verdict was clean "
                "reported an order near 0.4 for exactly this.")
            out.append({"sequence": f"exact ladder {who}", "priority": 50,
                        "values": [lv[k] for k in ks], "finding": finding})
    return out


def ndof_ladder_findings(work: Path) -> list[dict]:
    """The mesh ladder the agent's own logs imply, stated before delivery.

    Measured three times in one development stretch: converged couplings
    (proven to have run at every level) read as malformed because the levels
    were solved on a SELF-CHOSEN mesh ladder rather than the one the task
    prescribes. The signal was in the agent's own run logs the whole time:
    a halved mesh multiplies the DOF count by ~2^dim per level, so NDOF
    growth factors far from that reveal a non-halved ladder before any
    independent check sees it. No task parsing: this states what the files
    imply and what halving would imply, and leaves the comparison to the
    reader who holds the task sheet.
    """
    import re as _re
    # PER SIDE, WHERE THE LOGS ARE SIDED. This summed both sides' counts at a
    # level (`per_level[k] += nm`), and a sum can sit inside the band while
    # one side does not. Measured on a coupled run that was correct: side B went
    # NDOF 99 -> 255 between levels 1 and 2, 2.58x, below the band, while side
    # A went 54 -> 187 (3.46x); summed, 153 -> 442 is 2.89x and passed, and
    # B's own convergence order was 0.24. Unsided logs -- one code, one log
    # per level -- keep the summed reading, so a single-code run reads
    # exactly what it read before.
    per_side: dict[str, dict[int, float]] = {}
    # ONE FILE PER NAME, THE SHALLOWEST. Measured: a run left three working
    # copies of one level's console in side_B/ beside the real per-level logs
    # at the top; this summed each copy into its level and read the ladder as
    # 981 -> 1530 (1.56x) for a mesh that went 216 -> 765 -> 2871.
    _by_name: dict[str, tuple[int, Path]] = {}
    for q in _level_logs(work):
        try:
            _d = len(q.relative_to(work).parts)
        except ValueError:
            _d = 99
        if q.name not in _by_name or _d < _by_name[q.name][0]:
            _by_name[q.name] = (_d, q)
    for _d, q in sorted(_by_name.values(), key=lambda t: str(t[1])):
        try:
            txt = q.read_text(errors="replace")
        except OSError:
            continue
        nm = None
        for mm in _DOF_LINE.finditer(txt):
            nm = int(mm.group(1))
        if nm:
            k = _level_of(q)
            lv = per_side.setdefault(_side_of(q), {})
            lv[k] = lv.get(k, 0) + nm
    ladders = {side: sorted(lv) for side, lv in per_side.items() if len(lv) >= 2}
    if not ladders:
        return []
    # The halving band is DIMENSION-AWARE, read from the agent's own files:
    # a 2D halving multiplies DOF by ~4, a 3D one by ~8. One loose band
    # admitted a 2D level that grew only 2.15x (a non-halved third level,
    # charged by the assessment) — measured the round after this check
    # landed.
    dim = 2
    for q in _field_files(work):
        try:
            lines = q.read_text(errors="replace").splitlines()
        except OSError:
            continue
        if not lines:
            continue
        cols = [c.strip().lower() for c in lines[0].split(",")]
        if cols[:3] == ["x", "y", "z"]:
            dim = 3
            break
        if cols and all(c.replace(".","",1).replace("-","",1)
                        .replace("e","",1).replace("+","",1).isdigit()
                        for c in cols[:1]):
            continue        # headerless: try the next file for a header
        break
    lo, hi = (2.6, 6.0) if dim == 2 else (5.2, 12.0)
    factors: dict[str, list] = {}
    bad: list = []                     # (side, a, b, ndof_a, ndof_b, factor)
    for side, ks in sorted(ladders.items()):
        lv = per_side[side]
        steps = [(a, b, lv[b] / lv[a]) for a, b in zip(ks, ks[1:]) if lv[a] > 0]
        if not steps:
            continue
        factors[side] = [f for _a, _b, f in steps]
        bad += [(side, a, b, lv[a], lv[b], f) for a, b, f in steps
                if not (lo <= f <= hi)]
    if not factors or not bad:
        return []
    # THE SIDE'S OWN DUMPS DECIDE WHETHER THE MESH OR THE LOG IS WRONG. Measured: a side whose
    # run logs carried a placeholder NDOF line was told "YOUR OWN LOGS IMPLY A MESH LADDER THAT
    # WAS NOT HALVED" and "double every cell count" while its dumps grew 3.46x and 3.71x -- a
    # right ladder, and a remedy that would have broken it had it been obeyed.
    def _dump_nodes(side: str, k: int):
        if not side:
            return None
        cands = [work / f"solution_level{k}_{side}.csv"]
        for d in sorted(p for p in work.iterdir() if p.is_dir()) if work.is_dir() else []:
            if d.name.lower().rstrip("_-").endswith(side.lower()) and d.name.lower().startswith("side"):
                cands.append(d / f"field_level{k}.csv")
        for q in cands:
            try:
                rows = [ln for ln in q.read_text(errors="replace").splitlines()[1:] if ln.strip()]
            except OSError:
                continue
            if rows:
                return len(rows)
        return None
    _log_wrong = []
    for item in list(bad):
        sd, a, b, na, nb, f = item
        da, db = _dump_nodes(sd, a), _dump_nodes(sd, b)
        if da and db and lo <= db / da <= hi:
            _log_wrong.append((sd, a, b, na, nb, f, da, db))
            bad.remove(item)
    _log_findings = [{"sequence": "ndof ladder", "values": [f], "per_side": {sd: [f]}, "finding": (
        f"THE NDOF LINES IN side {sd}'S LOGS DO NOT COUNT ITS OWN MESH: its logs say NDOF {na:.0f} -> "
        f"{nb:.0f} ({f:.2f}x) from level {a} to {b}, while its own dumps grow {da} -> {db} nodes "
        f"({db / da:.2f}x, a halving). A run log's NDOF line is the solver's own count for that level, "
        f"printed by that run; fix the line, not the mesh.")}
        for sd, a, b, na, nb, f, da, db in _log_wrong]
    if not bad:
        return _log_findings
    # NAME THE SIDE, THE LEVEL AND THE FIX. Measured on the honest build: a run
    # with three coupled levels, coupling proven, interface satisfied and both
    # codes proven read this finding twice (2.15x from level 2 to 3), and
    # still handed in -- the text told it to "re-check every level", not
    # which level was wrong or that one couple call on a doubled mesh would
    # have mended it.
    def _who(side: str) -> str:
        return f"side {side}" if side else "the run"
    named = "; ".join(
        f"{_who(sd)}: level {b} is NOT the halving of level {a} (NDOF {na:.0f} -> "
        f"{nb:.0f}, {f:.2f}x; halving gives ~{2 ** dim}x)"
        for sd, a, b, na, nb, f in bad)
    sd0, a0, b0, _na, _nb, _f = bad[0]
    _whose = "that side's" if sd0 else "its"
    fix = (f"Re-run level {b0} alone -- one couple call -- with {_who(sd0)}'s "
           f"mesh halving its level-{a0} h (double every cell count in "
           f"{_whose} config.json), and rewrite level {b0}'s field, interface "
           f"and history files from that run.")
    growth = "; ".join(
        (f"{_who(sd)} " if sd else "") + ", ".join(f"{f:.2f}x" for f in fs)
        for sd, fs in sorted(factors.items()))
    flat = [f for _sd, fs in sorted(factors.items()) for f in fs]
    return [{"sequence": "ndof ladder", "values": flat, "per_side": factors,
             "finding": (
        "YOUR OWN LOGS IMPLY A MESH LADDER THAT WAS NOT HALVED: NDOF per level "
        "grows by " + growth
        + f", while halving h multiplies the DOF count by ~4 in 2D and ~8 in "
        f"3D. {named}. {fix} A result set on a different ladder cannot be "
        "compared level-to-level however well it converged -- measured on "
        "runs whose coupling evidence was sound at every level and which "
        "were unusable for exactly this.")}] + _log_findings


def completeness_findings(work: Path, only_levels: set | None = None,
                          skip_participant_dirs: bool = False) -> list[dict]:
    """Members of the per-level x per-side deliverable set that are absent.

    `only_levels` narrows the expected set to the levels named, so the same
    body can be asked "what is missing for the level just finished?" while the
    ladder is still running. Left None nothing changes.

    Inferred from the agent's OWN files, no task parsing: the levels are
    every k seen in any *_level<k>* deliverable, the sides are every _A/_B
    suffix seen in any of them, and each family that uses sides is expected
    to have every (level, side) member once any of its members exists.
    MEASURED, the case this exists for: a run with coupling evidence at
    every level ended with 17 minutes of budget unused and two solution
    files never attempted -- nothing at delivery time enumerated the required
    set against the disk, and the auto-audit named quality defects but not
    absent files. A missing member makes the file set unusable (a missing
    subdomain file, a wrong level count), and it counts for nothing, the
    same as no result set at all.
    """
    import re as _re
    seen: dict[str, set] = {}
    levels: set[int] = set()
    sides: set[str] = set()
    # WHILE THE LADDER RUNS, A FILE IN A PARTICIPANT'S OWN FOLDER IS A WORKING FILE. Measured: a
    # copy named run_level1_A.log inside side_A/ made two later levels' replies lead with "THE
    # DELIVERABLE SET IS INCOMPLETE" before any deliverable had been written.
    def _in_participant_dir(q: Path) -> bool:
        d = q.parent
        if d == work:
            return False
        return ((d / "exports.json").is_file() or (d / "imports.json").is_file()
                or any(d.glob("participant*.py")))
    for ext in ("csv", "log"):
        for q, kind, k, s in _level_files(work, ext):
            if ext == "csv" and _csv_role(q, kind) == "raw":
                continue
            if skip_participant_dirs and _in_participant_dir(q):
                continue
            fam = f"{kind}.{ext}"
            levels.add(k)
            if s:
                sides.add(s)
            seen.setdefault(fam, set()).add((k, s or None))
    if only_levels is not None:
        levels &= set(only_levels)
    if not levels or not seen:
        return []
    missing = []
    for fam, members in seen.items():
        kind, ext = fam.rsplit(".", 1)
        fam_sides = sorted({s for _, s in members if s})
        for k in sorted(levels):
            if not fam_sides:
                if not any(lv == k for lv, _ in members):
                    missing.append(f"{kind}_level{k}.{ext}")
            else:
                for s in fam_sides:
                    if (k, s) not in members:
                        missing.append(f"{kind}_level{k}_{s}.{ext}")
    if not missing:
        return []
    return [{"sequence": "deliverable completeness", "values": [],
             "finding": (
        "THE DELIVERABLE SET IS INCOMPLETE: judged only by the levels and "
        "sides your OWN files establish, these members are absent: "
        + ", ".join(missing[:8])
        + (f" (+{len(missing)-8} more)" if len(missing) > 8 else "")
        + ". A result set missing a level or a side is malformed and counts "
        "as unusable however good the present members are. For members "
        "whose level already has a run log or residual history, write "
        "them from the numbers you already have; for a level with NO "
        "run evidence, solve it or remove its stale files -- never "
        "invent members.")}]


# ═══════════════════════════════════════════════════════════════════════════
# THE ALWAYS-RUN CORRECTIVE FUNNEL
#
# Everything below is reachable from BOTH routes an agent actually drives: the
# live `couple` reply (which computes the flux/continuity checks from this run's
# own exports, before any file is written) and the on-delivery audit (which
# reads the per-level files). Each check reads ONLY the agent's own output —
# its exports, its solution_/interface_/residual_level*.csv, its logs — plus
# the PUBLIC task text where one is explicitly supplied. None of it reads the
# sealed key: every path here is rooted at the agent's own work dir or at two
# strings
# the agent itself passed in. The point is corrective prominence — every finding
# NAMES THE FIX, and what_to_fix_next() surfaces the single highest-priority one
# at the top of the reply, where a weak agent that merely "acts" will read it.
# ═══════════════════════════════════════════════════════════════════════════


def _iface_module():
    """The shared interface module, or None. Same import path the sign check
    uses, so this audit and any independent check share ONE definition of a
    jump."""
    try:
        import sys as _sys
        _here = str(Path(__file__).resolve().parents[1])
        if _here not in _sys.path:
            _sys.path.insert(0, _here)
        from blind_eval import interface as _IF
        return _IF
    except Exception:                                       # noqa: BLE001
        return None


def _rows_of(arr) -> list[list[float]]:
    """Normalise an exports 'values'/'normal_fluxes' array to list[list[float]]
    (one row of components per interface point). Non-numeric entries drop the
    whole array rather than guess."""
    out: list[list[float]] = []
    if arr is None:
        return out
    try:
        for v in arr:
            if isinstance(v, (list, tuple)):
                out.append([float(x) for x in v])
            else:
                out.append([float(v)])
    except (TypeError, ValueError):
        return []
    return out


def _match_exports(a: dict, b: dict, with_points: bool = False):
    """Pair two participants' exported interface data BY WHERE THE POINTS ARE
    (quality_checks.pair_interface_points), whatever order each lists them in.

    Returns (uA, uB, qA, qB) as component-row lists over the shared points -- and,
    with `with_points`, the coordinates of the flux rows as a fifth entry -- or
    None when the two sides do not sample enough common points to compare
    (a non-matching interface — said elsewhere, not guessed at here)."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    ca, cb = a.get("coordinates") or [], b.get("coordinates") or []
    if not ca or not cb:
        return None
    ua, ub = _rows_of(a.get("values")), _rows_of(b.get("values"))
    qa, qb = _rows_of(a.get("normal_fluxes")), _rows_of(b.get("normal_fluxes"))
    try:
        pr = _qc().pair_interface_points(ca, cb)
    except Exception:                                      # noqa: BLE001
        pr = None
    if pr is None:
        return None
    mua, mub, mqa, mqb, pts = [], [], [], [], []
    for i, j in zip(pr["ia"], pr["ib"]):
        if i < len(ua) and j < len(ub):
            mua.append(ua[i]); mub.append(ub[j])
        if i < len(qa) and j < len(qb):
            mqa.append(qa[i]); mqb.append(qb[j])
            pts.append(ca[i])
    if len(mua) < 2 and len(mqa) < 2:
        return None
    return (mua, mub, mqa, mqb, pts) if with_points else (mua, mub, mqa, mqb)


def exchange_carried_nothing_finding(export_a: dict, export_b: dict,
                                     name_a: str = "A", name_b: str = "B"):
    """Did this converged coupling exchange any information at all?

    THE HOLE THIS FILLS, and why it lives here rather than in a participant.
    Both live content checks at this hook abstain on exactly this input:
    `flux_cancellation_finding` returns None when its denominator is zero, and
    `field_continuity_finding` does the same, deferring in a comment to "the
    near-zero check" -- which reads per-level CSVs that DO NOT EXIST YET when
    couple() runs. So a pair that exchanged nothing passed both.

    Measured on a live coupled run: three levels, both codes proven, the
    partitioned iteration converged in 7 steps to an interface residual of
    6.8e-11, and the traction columns read -3.04e-18 and 2.17e-19. Each side
    returned the answer it would have returned with no partner at all. A
    coupling that transmits nothing converges IMMEDIATELY and convincingly,
    because two sides exchanging nothing cannot disagree -- so every
    self-consistency measure the run reports about itself reads as success.

    WHY IT IS A FINDING AND NOT A REFUSAL. openPASO cannot know whether a
    subdomain is genuinely undriven; a free interface really can carry no
    traction. So this reports and never stops the run. The same restraint is
    why it belongs to the driver rather than to a participant's export
    self-check: that check ends the coupling when it fires, and on iteration 1
    imports.json is `{}` by contract, so a side legitimately recovers ~0 there.
    A fatal check must not fire on an input openPASO cannot interpret; a
    reporting one must.
    """
    m = _match_exports(export_a, export_b)
    if not m:
        return None
    ua, ub, qa, qb = m

    def _scale(rows_a, rows_b) -> float:
        vals = [abs(x) for r in (rows_a or []) for x in (r or [])]
        vals += [abs(x) for r in (rows_b or []) for x in (r or [])]
        if any(not math.isfinite(v) for v in vals):
            return float("nan")               # not finite: not zero either (named below)
        if not vals:
            return -1.0                       # absent, not zero: different defect
        return max(vals)

    # A NaN IS NOT A ZERO. The scale used to drop non-finite entries, and a run whose
    # flux channel was NaN at every point led with "identically zero on BOTH sides ...
    # the iteration converges at once and the residual looks excellent" for a coupling
    # that never converged (measured).
    nonfinite = []
    for label, (ra, rb) in (("interface value", (ua, ub)), ("interface flux", (qa, qb))):
        for nm, rows in ((name_a, ra), (name_b, rb)):
            vals = [x for r in (rows or []) for x in (r or [])]
            bad = sum(1 for x in vals if not math.isfinite(x))
            if bad:
                nonfinite.append(f"the {label} of {nm} ({bad} of {len(vals)} values)")
    if nonfinite:
        return {"sequence": "interface exchange not finite", "priority": 9, "values": [],
                "finding": (
            "THE EXCHANGE IS NOT FINITE: " + "; ".join(nonfinite) + " are NaN or infinite at "
            "the matched interface points. Nothing a partner computes from them is a coupling. "
            "Find where that side's own computation produces them -- a division by a zero weight "
            "or norm, a solve that failed, a field that was never solved -- and run that side "
            "standalone until its exports are finite.")}

    _ROUNDOFF = 1e-14
    dead = [label for label, scale in (("interface value", _scale(ua, ub)),
                                       ("interface flux", _scale(qa, qb)))
            if 0.0 <= scale <= _ROUNDOFF]
    if not dead:
        return None
    both = len(dead) == 2
    return {"sequence": "interface exchange carried nothing",
            "priority": 28,
            "values": [0.0],
            "finding": (
        f"THIS COUPLING TRANSMITTED NOTHING: the {' and the '.join(dead)} "
        f"channel{'s are' if both else ' is'} identically zero on BOTH sides "
        f"({name_a} and {name_b}) at the matched interface points. Two sides "
        f"that exchange nothing cannot disagree, so the iteration converges at "
        f"once and the residual looks excellent while each side returns the "
        f"answer it would have returned with no partner at all — that is what "
        f"an interface residual of 1e-11 means here, not a satisfied "
        f"transmission condition. Check that the imported field is actually "
        f"read from imports.json and enters the assembled system (the "
        f"condition that integrates it is the usual omission), and that the "
        f"exported quantity is recovered from your own solution rather than "
        f"initialised and never written. If this subdomain is genuinely "
        f"undriven, say so in your report — openPASO cannot tell that from the "
        f"numbers and is not asserting otherwise."
    )}


def flux_cancellation_finding(export_a: dict, export_b: dict,
                              name_a: str = "A", name_b: str = "B"):
    """CLASS 1, live from the two exports: max|q_A + q_B| / max(|q_A|,|q_B|) at
    matched interface points. ~2.0 means the two outward fluxes ADD instead of
    cancelling — the exact-opposite-convention signature — and the fix is one
    sign on the value written out, not a re-solve. Returns a finding dict or
    None. Needs no reference: the two subdomains share the seam with OPPOSITE
    outward normals, so a correct coupling has q_A + q_B ~ 0 by construction."""
    m = _match_exports(export_a, export_b, with_points=True)
    if not m:
        return None
    _, _, qa, qb, pq = m
    if len(qa) < 2 or len(qb) < 2 or not qa[0] or not qb[0]:
        return None
    ncomp = min(len(qa[0]), len(qb[0]))
    if not ncomp:
        return None
    num, worst_c = 0.0, 0
    for ra, rb in zip(qa, qb):
        for c in range(ncomp):
            if abs(ra[c] + rb[c]) > num:
                num, worst_c = abs(ra[c] + rb[c]), c
    den = max((abs(x) for r in qa for x in r[:ncomp]), default=0.0)
    den = max(den, max((abs(x) for r in qb for x in r[:ncomp]), default=0.0))
    if den <= 0:
        return None
    ratio = num / den
    if ratio < 1.5:
        return None
    # THE SIGN IS MEASURED, POINT BY POINT AND LEG BY LEG, BEFORE IT IS NAMED. Measured:
    # at 1.61 this said "both sides carry the SAME sign" of a pair whose legs carried
    # opposite signs; the defect was a trace one side did not hold.
    same = [ra[worst_c] * rb[worst_c] > 0 for ra, rb in zip(qa, qb)]
    both = [ra[worst_c] != 0 and rb[worst_c] != 0 for ra, rb in zip(qa, qb)]
    n_same, n_all = sum(same), sum(both)
    per_leg = ""
    try:
        legs, _bad = _curve_legs([tuple(float(v) for v in p[:2]) for p in pq])
        if legs:
            per_leg = " (" + "; ".join(
                f"{sum(same[i] for i in set(g['idx']))} of {len(set(g['idx']))} along the leg through "
                f"({g['c'][0]:.4g}, {g['c'][1]:.4g})" for g in legs) + ")"
    except Exception:                                      # noqa: BLE001
        per_leg = ""
    counted = f"they carry the same sign at {n_same} of {len(qa)} matched points{per_leg}"
    if n_all and n_same >= 0.9 * n_all:
        return {"sequence": "interface flux cancellation", "priority": 30,
                "values": [ratio], "finding": (
            f"YOUR TWO INTERFACE FLUXES ADD INSTEAD OF CANCELLING: "
            f"max|q_{name_a}+q_{name_b}| / max|q| = {ratio:.2f} at the matched "
            f"interface points, and {counted}: the signature of one sign convention "
            f"opposite to the other. The two subdomains "
            f"share the seam with OPPOSITE outward normals, so a correct "
            f"coupling has q_{name_a}+q_{name_b} ~ 0. Two ways lead here, and "
            f"they are fixed in different places: a sign flipped in what one "
            f"side WRITES OUT (its field is right, its export is not), or the "
            f"partner's flux APPLIED with the wrong sign (then that side's field "
            f"itself is wrong, and flipping the written sign only hides it). The "
            f"fields tell which: the flux each side's own field carries, -k du/dn "
            f"outward, cancels its partner's in the first case and adds in the "
            f"second -- the own-field flux finding reads exactly that from the "
            f"field dumps. Re-check after the change: the exports and the fields "
            f"must both cancel, and the imbalance must SHRINK as you refine.")}
    return {"sequence": "interface flux cancellation", "priority": 30,
            "values": [ratio], "finding": (
        f"YOUR TWO INTERFACE FLUXES DO NOT CANCEL: max|q_{name_a}+q_{name_b}| / max|q| "
        f"= {ratio:.2f} at the matched interface points, and {counted}, so this is not one "
        f"sign convention opposite to the other over the whole interface, and nothing here "
        f"says which side is off. The two subdomains share the seam with OPPOSITE outward "
        f"normals, so a correct coupling has q_{name_a}+q_{name_b} ~ 0 at every point. "
        f"Compare, point by point, the flux each side's own field carries (-k du/dn outward) "
        f"with the flux it exports, and the trace each side holds with the one it was sent "
        f"-- the own-field flux finding reads the first from the field dumps. Re-check after "
        f"any change: the exports and the fields must both cancel, and the imbalance must "
        f"SHRINK as you refine.")}


def field_continuity_finding(export_a: dict, export_b: dict,
                             name_a: str = "A", name_b: str = "B"):
    """CLASS 2, live from the two exports: max|u_A - u_B| / scale at matched
    interface points. Large => the two subdomains disagree at the seam, so the
    coupling has not PHYSICALLY converged whatever the iterate residual did.
    Skipped when the two sides declare DIFFERENT field names (a heterogeneous
    exchange — displacement against traction — where continuity of the trace is
    not the right statement), so an FSI-style pair is never false-charged."""
    fa = (export_a or {}).get("field_name")
    fb = (export_b or {}).get("field_name")
    if fa and fb and str(fa).strip().lower() != str(fb).strip().lower():
        return None
    m = _match_exports(export_a, export_b)
    if not m:
        return None
    ua, ub, _, _ = m
    if len(ua) < 2 or len(ub) < 2 or not ua[0] or not ub[0]:
        return None
    ncomp = min(len(ua[0]), len(ub[0]))
    if not ncomp:
        return None
    num = 0.0
    for ra, rb in zip(ua, ub):
        for c in range(ncomp):
            num = max(num, abs(ra[c] - rb[c]))
    scale = max((abs(x) for r in ua for x in r[:ncomp]), default=0.0)
    scale = max(scale, max((abs(x) for r in ub for x in r[:ncomp]), default=0.0))
    if scale <= 0:
        return None                         # near-zero owns this, not continuity
    rel = num / scale
    if rel < 0.25:
        return None
    return {"sequence": "interface field continuity", "priority": 35,
            "values": [rel], "finding": (
        f"YOUR TWO SUBDOMAINS DISAGREE AT THE INTERFACE: "
        f"max|u_{name_a} - u_{name_b}| / max|u| = {rel:.0%} of the field's own "
        f"scale at the matched interface points. THE COUPLING HAS NOT "
        f"PHYSICALLY CONVERGED -- a partitioned scheme is converged when the "
        f"two SIDES agree at the seam, and an iterate residual that fell to "
        f"your tolerance is NOT the same statement (it can fall to 1e-8 while "
        f"the two exported traces still disagree by ~100%). The quantity your "
        f"loop stops on must be computed FROM THE TWO TRACES it is about to "
        f"export -- max|u_A - u_B| over the shared interface points, divided by "
        f"max|u_A| -- and iterated until THAT is small. If it will not fall, "
        f"the two sides are enforcing different transmission conditions: check "
        f"the Dirichlet value one side APPLIES is exactly the trace the other "
        f"side EXPORTED (same points, same sign), not a stale or re-sampled "
        f"copy.")}


def interface_continuity_findings(work: Path) -> list[dict]:
    """CLASS 2 from the FILES: field continuity at the FINEST level, from the
    agent's own two per-level interface files, using the shared interface
    module's two_sided_jumps so a finding here mirrors the unsatisfied-
    interface outcome the run would meet after delivery. Fires only on a LARGE
    disagreement (>=25% of the field scale) so a correct-but-coarse level is
    never charged; the real small-model failures sit near 100%."""
    _IF = _iface_module()
    if _IF is None:
        return []
    import re as _re
    ifs: dict[int, dict[str, Path]] = {}
    for f in _interface_files(work, sided=True):
        ifs.setdefault(_level_of(f), {})[_side_of(f).upper()] = f
    worst = None
    unpaired = []                               # (level, why) where no common support exists
    for lvl in sorted(ifs):                     # coarse -> fine, keep the finest
        side = ifs[lvl]
        if len(side) != 2:
            continue
        sa, sb = sorted(side)
        ga = _read_iface(side[sa], _IF, want_flux=True)
        gb = _read_iface(side[sb], _IF, want_flux=True)
        if not (ga and gb):
            continue
        # THE SAME POINTS, WHEREVER EACH FILE LISTS THEM: two files of unequal length were
        # refused outright, and a seam one side lists without its two ends went unjudged.
        ga, gb, _pw = _paired_iface(ga, gb)
        if ga is None:
            unpaired.append((lvl, _pw))
            continue
        try:
            jd, _why = _IF.two_sided_jumps(ga, gb)
        except Exception:                                   # noqa: BLE001
            continue
        if not jd:
            continue
        ju = jd.get("jump_u_rel")
        if isinstance(ju, (int, float)) and ju == ju:
            worst = (lvl, float(ju))
    note = ([{"sequence": "interface field continuity", "values": [], "informational": True,
              "finding": (
        f"THE INTERFACE CONTINUITY CHECK DID NOT RUN (NOT CHECKED) at level(s) "
        f"{', '.join(str(l) for l, _w in unpaired)}: {unpaired[0][1]}. It compares the two "
        f"sides' traces at the points both files list, so nothing is judged there.")}]
            if unpaired else [])
    if worst is None or worst[1] < 0.25:
        return note
    lvl, ju = worst
    return note + [{"sequence": f"interface field continuity level {lvl}",
             "priority": 35, "values": [ju], "finding": (
        f"YOUR TWO SUBDOMAINS DISAGREE AT THE INTERFACE at level {lvl}: the two "
        f"exported traces differ by {ju:.0%} of the field's own scale "
        f"(max relative field jump, matched point-for-point). THE COUPLING HAS "
        f"NOT PHYSICALLY CONVERGED -- a partitioned scheme is converged when "
        f"the two SIDES agree, and an iterate residual falling to tolerance is "
        f"NOT the same statement (it can reach 1e-8 while the fields disagree, "
        f"which is what this result set shows: {ju:.0%}). Recompute the stop "
        f"criterion FROM THE TWO FILES you are about to deliver -- "
        f"max|u_A - u_B| over the shared interface rows, divided by max|u_A| -- "
        f"and iterate on THAT. If it will not fall, the Dirichlet value one "
        f"side applies is not the trace the other side exported: match them "
        f"point-for-point, same sign.")}]


def identical_solution_levels_findings(work: Path) -> list[dict]:
    """CLASS 3: <field>_level<i> == <field>_level<j> point-for-point => one
    mesh was saved to every level, so there is no refinement to measure. Skips
    a near-zero field (the near-zero check owns that) so the two are not both
    reported for the same files."""
    import re as _re
    groups: dict[str, dict[int, Path]] = {}
    for q in _field_files(work):
        side = _side_of(q).upper()
        lv = _level_of(q)
        prev = groups.get(side, {}).get(lv)
        # shallowest wins, so a build/ copy never shadows the real deliverable
        if prev is None or (len(q.relative_to(work).parts)
                            < len(prev.relative_to(work).parts)):
            groups.setdefault(side, {})[lv] = q
    out: list[dict] = []
    for side, byl in sorted(groups.items()):
        levels = sorted(byl)
        if len(levels) < 2:
            continue
        vecs: dict[int, list[float]] = {}
        for lv in levels:
            vv = _solution_value_vector(byl[lv])
            if vv is not None:
                vecs[lv] = vv
        ident = []
        lvs = sorted(vecs)
        for i in range(len(lvs)):
            for j in range(i + 1, len(lvs)):
                a, b = vecs[lvs[i]], vecs[lvs[j]]
                if len(a) != len(b) or len(a) < 4:
                    continue
                scale = max((abs(x) for x in a), default=0.0)
                if scale < 1e-8:
                    continue                # near-zero field: not this finding
                if all(abs(x - y) <= 1e-9 * scale for x, y in zip(a, b)):
                    ident.append((lvs[i], lvs[j]))
        if ident:
            tag = f" side {side}" if side else ""
            pairs = ", ".join(f"{i}&{j}" for i, j in ident)
            # AHEAD OF EVERY PER-LEVEL FINDING, BECAUSE THEY ARE ALL ABOUT
            # ONE MESH REPEATED. At priority 50 this sat sixth while the
            # equation check led at 5 with "the weak residual is 1.991e-03,
            # 1.991e-03, 1.991e-03 across the levels -- it does not fall.
            # Check the source term you implemented against the one your task
            # states" -- which is TRUE and is a CONSEQUENCE: the residual is
            # constant because it is the same field three times. Measured, C2
            # one run: the lead sent the agent hunting a
            # source-term bug that did not exist, and the finding that names
            # the cause was six lines down.
            out.append({"sequence": f"identical solution levels{tag}",
                        "priority": 4, "values": [], "finding": (
                f"YOUR SOLUTION IS IDENTICAL ACROSS DISTINCT MESH LEVELS{tag} "
                f"(level pair(s) {pairs} agree point-for-point, the difference "
                f"is exactly zero). A refinement study measures how the answer "
                f"CHANGES as the mesh is refined, so identical levels carry no "
                f"order at all -- log2(|L1-L2|/|L2-L3|) is 0/0 -- and the study "
                f"counts as NOT RUN however correct each level is. "
                + (lambda r: (
                    f"THE SOLVE DID REFINE -- {r[1]} -- so these level files were "
                    f"written from ONE output: write each level's file from that "
                    f"level's own dump. " if r[0] is True else
                    f"THIS SIDE'S OWN DUMPS ARE IDENTICAL TOO -- {r[1]} -- so these "
                    f"files come from one mesh or from one output: print the node/DOF "
                    f"count inside the solve at each level and confirm it changes, and "
                    f"write each level's file from that level's own run. " if r[0] is False else
                    f"Two ways lead here, measured: the level files written from one "
                    f"output, or a solve that ran one mesh at every level; this "
                    f"side keeps no per-level dumps to tell which. "))(
                    _own_dumps_refined(work, side))
                + f"Run three DISTINCT meshes (the prescribed coarsest, then halve, "
                f"then halve again) and save each level's OWN result.")})
    return out


def _own_dumps_refined(work: Path, side: str):
    """(True|False|None, detail) -- whether the side's OWN per-level dumps differ
    between levels: True when their node counts or values differ (the solve refined,
    so identical DELIVERED files were written from one output), False when the dumps
    are identical too (the solve itself ran one mesh), None when there are none."""
    import csv as _c
    dirs = [d for d in _side_dirs(work)
            if side and (d.name.endswith(side) or d.name.endswith("_" + side.lower()))]
    for d in dirs:
        sets = {}
        for q in sorted(d.glob("field_level*.csv")):
            m = re.fullmatch(r"field_level(\d+)\.csv", q.name)
            if not m:
                continue
            try:
                with q.open() as fh:
                    rows = [r for r in _c.reader(fh)][1:]
            except OSError:
                continue
            sets[int(m.group(1))] = rows
        if len(sets) >= 2:
            counts = {k: len(v) for k, v in sorted(sets.items())}
            same = all(v == sets[min(sets)] for v in sets.values())
            return (not same,
                    f"{d.name}/field_level<k>.csv hold " + ", ".join(f"{n} nodes at level {k}"
                                                                   for k, n in counts.items()))
    return None, ""


def _solution_value_vector(path: Path):
    """The concatenated non-coordinate columns of a solution CSV, in row order,
    for the identical-levels comparison. Requires a header (x,y[,z],...) so a
    headerless file is skipped rather than guessed at."""
    try:
        rows = [r for r in path.read_text(errors="replace").splitlines()
                if r.strip()]
    except OSError:
        return None
    if len(rows) < 2:
        return None
    first = rows[0].split(",")
    if not first or any(ch.isdigit() for ch in first[0]):
        return None                                     # headerless
    hdr = [c.strip().lower() for c in first]
    val_idx = [i for i, c in enumerate(hdr) if c not in ("x", "y", "z")]
    if not val_idx:
        return None
    vec: list[float] = []
    for r in rows[1:]:
        parts = r.split(",")
        if len(parts) < len(hdr):
            continue
        try:
            for i in val_idx:
                vec.append(float(parts[i]))
        except ValueError:
            return None
    return vec or None


def solution_rows_grow_findings(work: Path) -> list[dict]:
    """CLASS 4: a solution row count that GROWS with the level is a mesh trace,
    not a fixed probe grid. Every solution_level<k>.csv must carry the SAME
    prescribed probe points at every level; a growing count means the agent
    wrote its own mesh nodes instead of sampling the fixed points."""
    import re as _re
    per_side: dict[str, dict[int, int]] = {}
    for q in _field_files(work):
        side = _side_of(q).upper()
        lv = _level_of(q)
        try:
            n = sum(1 for _ in open(q, errors="ignore")) - 1     # minus header
        except OSError:
            continue
        cur = per_side.setdefault(side, {})
        if lv not in cur or n > cur[lv]:
            cur[lv] = n
    out: list[dict] = []
    for side, per in sorted(per_side.items()):
        ns = [per[l] for l in sorted(per)]
        if len(ns) >= 2 and len(set(ns)) > 1 and all(
                b > a for a, b in zip(ns, ns[1:])):
            tag = f" side {side}" if side else ""
            out.append({"sequence": f"solution rows grow{tag}", "priority": 55,
                        "values": ns, "finding": (
                f"YOUR SOLUTION ROW COUNT GROWS WITH THE LEVEL{tag}: "
                + ", ".join(str(n) for n in ns) + " rows across the levels. The "
                "task's SOLUTION PROBE POINTS are FIXED -- the same points at "
                "every mesh level -- so every per-level field file must carry "
                "the SAME rows in the same order. A count that grows with the "
                "mesh means you wrote your own MESH NODES instead of evaluating "
                "(interpolating) your solution AT the prescribed points. "
                "Re-read the task's probe-point list and sample your existing "
                "solution there; no re-solve is needed, and the points are "
                "deliberately NOT mesh nodes (interpolate inside the element "
                "that contains each one).")})
    return out


def pde_source_findings(pde_json: str, task_text: str = "") -> list[dict]:
    """OPTIONAL, public-only. Compares the agent's OWN declared source/equation
    string against the PUBLIC task source text it also supplied. Reads nothing
    sealed — both operands are strings the agent passed in. A silent wrong
    forcing (right shape, wrong function) converges cleanly to a different
    answer and no self-consistency check can see it, so this is the one place a
    declared/public mismatch can be named without any key.

    `pde_json` shape: {"A": {"source": "<what you implemented>",
                             "task_source": "<the task's stated source>"},
                       "B": {...}}  — task_text is a fallback task_source.
    """
    import json as _json
    import re as _re
    if not pde_json:
        return []
    try:
        spec = _json.loads(pde_json)
    except Exception:                                       # noqa: BLE001
        return [{"sequence": "declared pde", "informational": True, "finding": (
            "DECLARED PDE NOT CHECKED: the pde argument was not valid JSON. "
            "Pass {\"A\": {\"source\": \"...\", \"task_source\": \"...\"}} to "
            "have openPASO compare the forcing you implemented against the task's "
            "stated forcing (both PUBLIC strings you supply).")}]
    if not isinstance(spec, dict):
        return []

    def _norm(s: str) -> str:
        return _re.sub(r"\s+", "", str(s or "")).lower().replace("**", "^")

    out: list[dict] = []
    for side, d in spec.items():
        if not isinstance(d, dict):
            continue
        declared = d.get("source") or d.get("equation") or ""
        public = d.get("task_source") or task_text or ""
        nd, npub = _norm(declared), _norm(public)
        if not nd or not npub:
            continue
        if nd not in npub and npub not in nd:
            out.append({"sequence": f"declared source {side}", "priority": 45,
                        "finding": (
                f"THE SOURCE YOU DECLARED FOR SIDE {side} DOES NOT MATCH THE "
                f"PROBLEM STATEMENT YOU SUPPLIED: you declared "
                f"'{str(declared)[:120]}', which does not appear in the source "
                f"you gave: '{str(public)[:160]}'. Confirm you implemented "
                f"your problem's actual source term and coefficient, not a "
                f"paraphrase or a placeholder -- a different forcing converges "
                f"cleanly to a different answer, and no self-consistency check "
                f"can catch it. (This compares only the two strings you "
                f"supplied; openPASO reads nothing of its own.)")})
    return out


# Ordered high->low priority for the WHAT TO FIX NEXT lead. The FIRST row whose
# any-substring appears in a finding (or the finding's explicit `priority`)
# wins; lower rank = fix this FIRST = leads the reply. Matched on the STABLE
# uppercase headlines the findings already carry, so no existing finding has to
# be edited to be ranked.
_PRIORITY_TABLE = [
    (10, ("NOT COUPLED", "NEVER RECEIVED", "IDENTICALLY ZERO ON BOTH SIDES",
          "NEGATED TO THE LAST BIT", "SMALLER THAN ITS PARTNER",
          "TRANSMITTED NOTHING", "WAS NEVER RUN", "NEVER RUN",
          "PRODUCED BYTE-IDENTICAL", "NOT A FUNCTION OF ITS IMPORTS",
          "NOT COUPLED TO ITS PARTNER", "EXITED NON-ZERO", "TIMED OUT")),
    (12, ("NO LINE ANY SOLVER EMITS", "IS IDENTICALLY ZERO ON SIDE")),
    (11, ("NAMES", "FILE(S) THAT DO NOT EXIST")),
    (15, ("NO PER-LEVEL FIELD FILE AND NO INTERFACE FILE",)),
    (20, ("COUPLING HISTORY TOO SHORT", "NON-POSITIVE OR NON-FINITE RESIDUAL",
          "RESIDUAL BARELY MOVED", "CONSTANT RESIDUAL COLUMN",
          "WRITTEN-IN SEQUENCE", "IDENTICAL RESIDUAL HISTORY")),
    (30, ("ADD INSTEAD OF CANCELLING", "SAME SIGN", "WRONG SIGN",
          "FAIL TO CANCEL", "SIGN-CONVENTION")),
    (35, ("DISAGREE AT THE INTERFACE", "FIELD CONTINUITY")),
    (40, ("IS NOT THE DISAGREEMENT", "YOU REPORT ",
          "SHRINKS TOO SLOWLY", "DOES NOT SHRINK", "INCONSISTENT WITH YOUR "
          "SOLVE", "NOT CANCELLING")),
    (45, ("DOES NOT MATCH THE PROBLEM STATEMENT",)),
    (50, ("IDENTICAL ACROSS DISTINCT MESH LEVELS", "BIT-IDENTICAL")),
    (55, ("ROWS GROW WITH THE LEVEL", "ROW COUNT GROWS WITH THE LEVEL",
          "RUN TO THE ENDS OF THE INTERFACE", "DISTINCT VALUES ACROSS",
          "NEAREST-NODE")),
    (58, ("RUN LOG FROM THE WRONG LEVEL",)),
    (60, ("DELIVERABLE SET IS INCOMPLETE", "MISSING LEVEL", "DIFFERING COPY",
          "NO DOF-COUNT LINE", "MESH LADDER THAT WAS NOT HALVED",
          "SUMMARY FILE IS MISSING", "YOUR OWN NDOF IS")),
    (70, ("NEAR-ZERO FIELD", "FLOOR:")),
    (80, ("ORDER MISMATCH", "IMPROVE AT ONLY", "NON-MONOTONE", "FLUX JUMP",
          "FIRST-ORDER INTERFACE RECOVERY")),
]


def _rank(f: dict) -> int:
    p = f.get("priority")
    if isinstance(p, (int, float)):
        return int(p)
    t = " ".join((f.get("finding") or "").upper().split())
    for rank, subs in _PRIORITY_TABLE:
        if any(s in t for s in subs):
            return rank
    return 85


def what_to_fix_next(findings, *, converged: bool = True,
                     clean_msg: str | None = None) -> str:
    """The single leading line every route puts at the top of the reply.

    A weak agent reads the top of the message, not the thirty-first finding
    under a data dump, so the ONE highest-priority fix goes first, with its full
    corrective sentence. A clean funnel leads with the necessary-not-sufficient
    reminder — self-consistency cannot see a wrong-but-consistent answer.
    """
    real = [f for f in (findings or []) if not f.get("informational")]
    if clean_msg is None:
        clean_msg = (
            "your output is SELF-CONSISTENT (necessary, not sufficient) -- now "
            "check your FIELDS match the task: the right physics and boundary "
            "conditions, the prescribed interface/solution probe points, and "
            "the prescribed mesh levels. A self-consistency check reads only "
            "your own files and cannot see a wrong-but-consistent answer.")
    if not real:
        if not converged:
            return ("WHAT TO FIX NEXT: your coupling did NOT converge, so there "
                    "is no result yet -- a non-converged iteration is not a "
                    "solution. Fix convergence first (start relaxation at "
                    "theta=0.5, and confirm each side actually reads "
                    "imports.json and applies it), then re-run. Nothing "
                    "downstream matters until the iteration reaches tolerance.")
        return "WHAT TO FIX NEXT: " + clean_msg
    real.sort(key=lambda f: (_rank(f), str(f.get("sequence", ""))))
    top = real[0]
    others = real[1:]
    head = (f"WHAT TO FIX NEXT (highest priority of {len(real)} finding(s); "
            f"reads ONLY your own output files, never a reference solution):\n"
            f"  >> {top.get('sequence', '')}: {top.get('finding', '')}")
    if others:
        head += ("\n\nTHEN, in priority order: "
                 + "; ".join(str(o.get("sequence", "")) for o in others[:8])
                 + (f" (+{len(others) - 8} more)" if len(others) > 8 else "")
                 + ". Full corrective text for each is in the findings list.")
    return head



def _fourc_deck_state(side: Path, work: Path) -> dict | None:
    """What 4C left in a participant's directory, as the next sub-step: which deck it refused
    (its own error lines, the defects named from the deck text) or which runs finished.
    None when the directory holds no deck and no 4C console yet."""
    try:
        from tools.fourc_deck_lint import side_dir_report   # noqa: PLC0415
        rep = side_dir_report(side)
    except Exception:                                       # noqa: BLE001
        return None
    if not (rep["decks"] or rep["errors"] or rep["finished"] or rep.get("tracebacks") or rep.get("consoles")):
        return None
    # A PYTHON STOP ALONE DOES NOT MAKE A 4C SIDE. With no deck, no 4C console and
    # no 4C output, a traceback is the participant's own; the step that follows
    # keeps it in its brief but must not be headed "MAKE THE 4C DECK RUN"
    # (measured: served for an NGSolve side's ImportError).
    _is_4c = bool(rep["decks"] or rep["errors"] or rep["finished"] or rep.get("consoles"))
    if not _is_4c:
        try:
            _txt = " ".join(q.read_text(errors="ignore")[:20000]
                            for q in side.glob("*.py") if ".replaced-" not in q.name)
        except OSError:
            _txt = ""
        _is_4c = bool(re.search(r"\.4C\.yaml|FOURC_BIN|\b4C\b", _txt))
    try:
        rel = str(side.relative_to(work))
    except ValueError:
        rel = side.name
    parts, what = [], []
    if rep["finished"]:
        fin = ", ".join(f"{p} ({', '.join(k)})" for p, k in rep["finished"].items())
        what.append(f"finished run(s): {fin}")
        if rep["defects"]:
            # a run that finished on a defective deck solved a different problem (4C drops a
            # condition on an undefined E id silently and reports 'finished normally')
            parts.append(f"the run(s) with output prefix {', '.join(rep['finished'])} finished (VTU on disk), but a "
                         "run whose deck is named below solved the WRONG problem and has to be re-run after the fix")
        else:
            parts.append(f"the run(s) with output prefix {', '.join(rep['finished'])} finished (VTU on disk) -- leave them")
    if rep["monitors"]:
        parts.append(f"{len(rep['monitors'])} reaction-monitor file(s) exist")
    for lg, said in rep["errors"].items():
        what.append(f"4C stopped ({lg})")
        parts.append(f"4C's own error in {lg}: {said}")
    for dk, why in rep["defects"].items():
        what.append(f"{dk}: {len(why)} defect(s)")
        parts.append(f"deck {dk}: " + "; ".join(why))
    for lg, tb in rep.get("tracebacks", {}).items():
        what.append(f"the participant itself stopped in Python ({lg})")
        parts.append(f"the participant's own Python stop in {lg}: {tb} -- fix that line first")
    for lg, tail in rep.get("consoles", {}).items():
        # a 4C console with neither a finish nor a recognised error: its last lines are the verdict
        what.append(f"4C's console {lg} ends without a finish")
        parts.append(f"the console {lg} shows neither 'finished normally' nor an error block; it ends with: {tail} "
                     "-- read that log from the top, the cause is above its last lines")
    if not rep["errors"] and not rep["defects"] and not rep.get("tracebacks") and not rep.get("consoles"):
        if rep["finished"]:
            what.append("no exports.json")
            parts.append("every 4C run finished and no defect is named, so the participant stopped in its "
                         "recovery or export: run it again and read ITS stderr from the top (the served "
                         "check names the missing output)")
        else:
            what.append("deck(s) written, no 4C console found")
            parts.append("no 4C console log lies next to the deck(s): run the binary line-buffered "
                         "(stdbuf -oL -eL <bin> <deck> <prefix> > <deck>.log 2>&1) and read the log from the top")
    brief = (f"the directory {rel} holds " + "; ".join(parts) + ". Fix exactly the named deck against "
             "the grammar (`4C -p`, prepare_simulation(solver='fourc', physics=...)), run "
             "check_input(solver='fourc', input_path=<the deck>) until it names no defect, then re-run that "
             "deck until its VTU folder appears; a deck that ran is not touched.")
    return {"what": "; ".join(what) + ".", "brief": brief, "fourc": _is_4c}

def febio_deck_findings(work: Path) -> list[dict]:
    """The FEBio deck a side last handed to FEBio, judged from its text.

    MEASURED on three coupled rounds with a FEBio elastic side: the participant writes its .feb at run
    time and rewrites it every iteration, so the deck that made the delivered field sits in the side
    directory, and its decisive defects were ones FEBio runs to NORMAL TERMINATION without a word --
    u_z held on 146 of the 1386 nodes of a plane-strain slab (a three-level result whose error did not
    decay), a held set with 1178 interior nodes and interface tables with half their entries (two
    fields that came out zero). The deck is the one a script in the side names, else the newest.
    Reads only the agent's own files; one finding per side, the defects listed as the lint names them.
    """
    from tools.febio_deck_lint import lint_deck, looks_like_deck      # noqa: PLC0415
    out: list[dict] = []
    for side in _side_dirs(work):
        decks = [q for q in side.glob("*.feb") if q.is_file()]
        if not decks:
            continue
        try:
            named = " ".join(q.read_text(errors="ignore")[:400000] for q in side.glob("*.py")
                             if ".replaced-" not in q.name)
        except OSError:
            named = ""
        pick = [q for q in decks if q.name in named] or decks
        deck = max(pick, key=lambda q: q.stat().st_mtime)
        try:
            txt = deck.read_text(errors="ignore")
        except OSError:
            continue
        if not looks_like_deck(txt):
            continue
        found = lint_deck(txt)
        if not found:
            continue
        try:
            rel = str(deck.relative_to(work))
        except ValueError:
            rel = deck.name
        out.append({"sequence": f"febio deck {side.name}", "values": [], "priority": 7, "finding": (
            f"THE FEBio DECK {rel}, THE LAST ONE THIS SIDE HANDED TO FEBio, CARRIES {len(found)} DEFECT(S) "
            f"named from its text. FEBio runs some of these to NORMAL TERMINATION without a word, so the "
            f"field that came out of this deck is that deck's answer:\n  - " + "\n  - ".join(found[:8])
            + (f"\n  ... and {len(found) - 8} more" if len(found) > 8 else "")
            + "\nFix the script that writes the deck, run the ladder again, and read this check again.")})
    return out


def _side_dirs(work: Path) -> list:
    """Every participant folder under `work`, one or two levels down.

    ONE LEVEL WAS NOT ENOUGH. A run that kept its sides in work/coupled_run/side_A
    was audited as having no sides at all: the equation check found no config,
    said nothing, and a side whose interior was never solved went out unjudged.
    A folder counts when it holds a participant's own traffic (exports.json,
    imports.json or a field_level dump)."""
    out = []
    for d in sorted(set(list(work.glob("*/")) + list(work.glob("*/*/")))):
        if not d.is_dir() or d.name.startswith(".") or "simulation_outputs" in d.parts:
            continue
        if ((d / "exports.json").is_file() or (d / "imports.json").is_file()
                or any(d.glob("field_level*.csv"))):
            out.append(d)
    return out


# THE REMEDY NAMES THE HELD SET IN THE SIDE'S OWN CODE. Measured on a steady coupled round: a deal.II
# side held every dof inside its box (its hole 4 put them into boundary_values), and the remedy below
# named "the free-dof mask the solve inverts on" -- NGSolve's mechanism, which a deal.II program does
# not have; the cell spent 18 minutes on other causes. Each wording is the name the code's own served
# contract gives its held dofs; a side whose code its files do not show gets words of no code's.
_HELD_SET = {
    "dealii": ("the dofs boundary_values holds when the solve runs (hole 4 of the program fills it): a dof inside "
               "the box that it holds is never solved for"),
    "ngsolve": "the free-dof mask the solve inverts on (fes.FreeDofs(), or the BitArray handed to the solve)",
    "fenics": ("the dofs the dirichletbc entries of bcs hold: a dof inside the subdomain that one of them holds is "
               "never solved for"),
    "skfem": "the condensed set D: a dof inside the subdomain that is in D is never solved for",
    "kratos": "the nodes fixed before the solve: a node inside the subdomain that is fixed is never solved for",
}


def _held_set_words(side_dir: Path) -> str:
    """How the remedy names the dofs a side's solve holds, in the words of that side's own code."""
    try:
        from .participant_lint import _side_codes, _strip_strings_and_comments, looks_like_participant  # noqa: PLC0415
    except Exception:                                                                 # noqa: BLE001
        _side_codes = None
    codes: list = []
    if _side_codes is not None:
        files = sorted(Path(side_dir).glob("*.py")) + sorted(Path(side_dir).glob("*.cc"))
        texts = []
        for q in files:
            try:
                texts.append(q.read_text(errors="ignore"))
            except OSError:
                continue
        # a participant script speaks for the side before a probe beside it
        for t in sorted(texts, key=lambda t: not looks_like_participant(t)):
            codes += [c for c in _side_codes(_strip_strings_and_comments(t)) if c not in codes]
    held = [c for c in codes if c in _HELD_SET]
    if len(held) == 1:
        return _HELD_SET[held[0]]
    return "the dofs the solve holds at given values: a dof inside the subdomain that it holds is never solved for"


def unsolved_field_findings(work: Path, dirs=None, since=None, scan: bool = True) -> list[dict]:
    """A side whose own field dump is exactly 0.0 at most nodes inside its subdomain.

    MEASURED over every recorded field dump with at least eight interior nodes
    (1518): the share of interior nodes holding exactly 0.0 is under 10 % in 1411
    and at or above 50 % in 27 -- none of those 27 from a run whose answer was right; one
    was handed in as a converged coupling openPASO had called verified, its
    Dirichlet side having solved none of its interior (its mask was built by
    looping over a BitArray and held one bit). A field that solves a problem with non-zero data does not return
    exact zeros at interior nodes, so this needs no equation and no config. `scan=False` judges
    `dirs` alone (a ladder answers for its participants, not for a probe folder beside them)."""
    out: list[dict] = []
    seen, folders = set(), []
    for d in list(dirs or []) + (_side_dirs(work) if scan else []):
        try:
            key = Path(d).resolve()
        except OSError:
            continue
        if key not in seen and Path(d).is_dir():
            seen.add(key)
            folders.append(Path(d))
    for d in folders:
        for p in sorted(d.glob("field_level*.csv")):
            rows = []
            try:
                _cut = _cutoff_for(d, since)
                if _cut is not None and p.stat().st_mtime < _cut:
                    continue                 # the last run's dump, not this one's
                with p.open() as fh:
                    for r in _csv.reader(fh):
                        try:
                            v = [float(c) for c in r]
                        except ValueError:
                            continue
                        if len(v) >= 3:
                            rows.append(v)
            except OSError:
                continue
            if len(rows) < 12:
                continue
            xs, ys = [r[0] for r in rows], [r[1] for r in rows]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            tol = 1e-9 * max(x1 - x0, y1 - y0, 1e-30)
            inner = [r for r in rows if x0 + tol < r[0] < x1 - tol and y0 + tol < r[1] < y1 - tol]
            big = max(abs(v) for r in rows for v in r[2:])
            if len(inner) < 8 or not big > 0:
                continue
            zeros = sum(1 for r in inner if all(v == 0.0 for v in r[2:]))
            if zeros < 0.5 * len(inner):
                continue
            try:
                rel = str(p.relative_to(work))
            except ValueError:
                rel = p.name
            out.append({
                "sequence": f"unsolved field {rel}", "values": [zeros, len(inner)],
                "priority": 3,
                "finding": (
                    f"THIS SIDE'S FIELD WAS NOT SOLVED INSIDE ITS SUBDOMAIN: {rel} holds exactly "
                    f"0.0 at {zeros} of its {len(inner)} interior nodes while its largest value is "
                    f"{big:.3g}. A field that solves a problem with non-zero data is not exactly "
                    f"zero at interior nodes: those dofs were never solved for, or were zeroed "
                    f"after the solve. The coupling still converges on such a side -- it hands "
                    f"back what it was given -- so neither the residual history nor the "
                    f"exchange checks show it. Check {_held_set_words(d)}; and check that the solve's "
                    f"result is what the side exports.")})
    return out


def _read_named_csv(p: Path):
    """(header names, rows of floats) of a dump; rows that do not parse are skipped."""
    import csv as _csv
    with Path(p).open() as fh:
        rd = _csv.reader(fh)
        hdr = [h.strip().lower() for h in (next(rd, None) or [])]
        rows = []
        for r in rd:
            try:
                v = [float(c) for c in r]
            except ValueError:
                continue
            if len(v) == len(hdr):
                rows.append(v)
    return hdr, rows


def _flux_out(k, n_out, t, dn, dt):
    """-(K grad u) . n_out with grad u = dn n_out + dt t: k one number (then -k dn) or a 2x2 K."""
    import numpy as np
    if not isinstance(k, list):
        return -float(k) * dn
    K = np.asarray(k, float)
    return -(float(n_out @ K @ n_out) * dn + float(n_out @ K @ t) * dt)


def _own_field_flux(side_dir: Path, lvl: int, k, why: list | None = None):
    """(along-coordinates, outward flux -k du/dn from the side's OWN field dump) at its
    interface points, excluding the two end nodes and their neighbours; None when the
    dumps cannot say. The derivative is a one-sided difference over one interface spacing
    into the side's own field (exact on the first element row of a structured P1 grid).

    A BENT interface is taken leg by leg -- each leg's own normal, its own spacing, the
    two end nodes of each leg and their neighbours left out -- and comes back as
    {"legs": [...]}; where its legs cannot be read, None, with the reason in `why`."""
    import numpy as np
    try:
        from scipy.interpolate import griddata
        hf, rf = _read_named_csv(side_dir / f"field_level{lvl}.csv")
        hi, ri = _read_named_csv(side_dir / f"interface_level{lvl}.csv")
    except (OSError, ImportError, StopIteration):
        return None
    if not ({"x", "y"} <= set(hf) and {"x", "y"} <= set(hi)) or len(rf) < 16 or len(ri) < 7:
        return None
    vc = next((hf.index(c) for c in hf if c not in ("x", "y", "z")), None)
    if vc is None:
        return None
    F = np.asarray(rf, float)
    I = np.asarray(ri, float)[:, [hi.index("x"), hi.index("y")]]
    if not (np.isfinite(F).all() and np.isfinite(I).all()):
        return None
    xy, u = F[:, [hf.index("x"), hf.index("y")]], F[:, vc]
    legs, bad = _curve_legs(I)
    if bad is not None:
        if why is not None:
            why.append(f"{side_dir.name}'s interface points in interface_level{lvl}.csv {bad}")
        return None
    if legs is not None:
        return _own_field_flux_by_leg(side_dir, lvl, k, I, xy, u, legs, why)
    ax = int(np.argmin(np.ptp(I, axis=0)))           # the coordinate the interface fixes
    al = 1 - ax
    plane = float(np.median(I[:, ax]))
    inside = 1.0 if float(np.mean(xy[:, ax])) > plane else -1.0   # direction into the side
    order = np.argsort(I[:, al])
    pts = I[order]
    s = np.diff(pts[:, al])
    s = s[s > 0]
    if not s.size:
        return None
    delta = float(np.median(s))
    keep = pts[2:-2]
    if len(keep) < 3:
        return None
    p0 = keep.copy()
    p1 = keep.copy()
    p1[:, ax] = plane + inside * delta
    u0 = griddata(xy, u, p0, method="linear")
    u1 = griddata(xy, u, p1, method="linear")
    if np.isnan(u0).any() or np.isnan(u1).any():
        return None
    # outward normal points AWAY from the side: du/dn_out = (u0 - u1) / delta. With a 2x2 K the
    # flux also takes in the derivative along the interface, read off the trace itself.
    n_out = np.zeros(2)
    n_out[ax] = -inside
    t_al = np.zeros(2)
    t_al[al] = 1.0
    dt = np.gradient(u0, keep[:, al]) if isinstance(k, list) else 0.0
    return keep[:, al], _flux_out(k, n_out, t_al, (u0 - u1) / delta, dt)


def _own_field_flux_by_leg(side_dir: Path, lvl: int, k, I, xy, u, legs, why):
    """_own_field_flux on a bent interface: {"legs": [{"pts", "t", "n", "c", "delta",
    "q"}]}, one entry per leg long enough to leave three points once the two end
    nodes at each end are out. Which side of a leg is this side's is read from its
    own field nodes: they lie on one side of the leg and not on the other (a notched
    side lies on both sides of the LINE through a leg, so the mean position cannot
    say it). None, with the reason in `why`, when a leg cannot be read that way."""
    import numpy as np
    from scipy.interpolate import griddata
    out = []
    for g in legs:
        P = np.unique(I[g["idx"]], axis=0)
        a = (P - g["c"]) @ g["t"]
        o = np.argsort(a)
        P, a = P[o], a[o]
        s = np.diff(a)
        s = s[s > 0]
        if not s.size:
            continue
        delta = float(np.median(s))
        keep = P[2:-2]
        if len(keep) < 3:
            continue
        ak = (keep - g["c"]) @ g["t"]
        fa = (xy - g["c"]) @ g["t"]
        fb = (xy - g["c"]) @ g["n"]
        band = (fa >= ak.min() - 1e-9) & (fa <= ak.max() + 1e-9)
        plus = int(np.sum(band & (fb >= 0.25 * delta) & (fb <= 2.5 * delta)))
        minus = int(np.sum(band & (fb <= -0.25 * delta) & (fb >= -2.5 * delta)))
        if (plus > 0) == (minus > 0):
            if why is not None:
                why.append(f"{side_dir.name}'s field nodes in field_level{lvl}.csv lie on "
                           + ("both sides" if plus else "neither side")
                           + f" of its interface leg through ({g['c'][0]:.4g}, {g['c'][1]:.4g})")
            return None
        inward = 1.0 if plus else -1.0
        u0 = griddata(xy, u, keep, method="linear")
        u1 = griddata(xy, u, keep + inward * delta * g["n"], method="linear")
        if np.isnan(u0).any() or np.isnan(u1).any():
            return None
        dt = np.gradient(u0, ak) if isinstance(k, list) else 0.0
        out.append({"pts": keep, "t": g["t"], "n": g["n"], "c": g["c"], "delta": delta,
                    "q": _flux_out(k, -inward * np.asarray(g["n"], float), np.asarray(g["t"], float),
                                   (u0 - u1) / delta, dt)})
    return {"legs": out} if out else None


def _leg_pairs_balance(fa: dict, fb: dict):
    """max |qA + qB| over the legs the two sides share, over the larger flux there;
    None when fewer than three points could be compared. Legs are matched by their
    line (parallel, and the same offset to a fraction of the spacing); B's flux is
    interpolated along A's leg and never extrapolated beyond B's points."""
    import numpy as np
    qa_all, qb_all = [], []
    for la in fa["legs"]:
        for lb in fb["legs"]:
            if abs(la["t"][0] * lb["t"][1] - la["t"][1] * lb["t"][0]) > 1e-3:
                continue
            if abs(float((lb["c"] - la["c"]) @ la["n"])) > 0.25 * min(la["delta"], lb["delta"]):
                continue
            sa = (la["pts"] - la["c"]) @ la["t"]
            sb = (lb["pts"] - la["c"]) @ la["t"]
            o = np.argsort(sb)
            sb, qb = sb[o], lb["q"][o]
            inside = (sa >= sb.min()) & (sa <= sb.max())
            if not inside.any():
                continue
            qa_all.append(la["q"][inside])
            qb_all.append(np.interp(sa[inside], sb, qb))
    if not qa_all:
        return None
    qa, qb = np.concatenate(qa_all), np.concatenate(qb_all)
    if len(qa) < 3:
        return None
    scale = max(float(np.abs(qa).max()), float(np.abs(qb).max()))
    if scale <= 0:
        return None
    return float(np.abs(qa + qb).max()) / scale


def _implied_k_by_leg(side_dir: Path, lvl: int):
    """[(leg centre, implied k, misfit, share of points where the export and -du/dn
    agree in sign)] of a side on a BENT interface at one level: the best constant k with
    which the flux the side exports (its interface dump's qn) is -k du/dn of its own field
    (the one-sided difference of _own_field_flux), leg by leg, and how far the export lies
    from it. None where the legs or the flux column cannot be read."""
    import numpy as np
    f = _own_field_flux(side_dir, lvl, 1.0)
    if not isinstance(f, dict):
        return None
    try:
        hi, ri = _read_named_csv(side_dir / f"interface_level{lvl}.csv")
    except OSError:
        return None
    qcol = next((hi.index(c) for c in ("qn", "q", "q_n", "flux") if c in hi), None)
    if qcol is None or not ({"x", "y"} <= set(hi)):
        return None
    I = np.asarray(ri, float)
    P, Q = I[:, [hi.index("x"), hi.index("y")]], I[:, qcol]
    span = float(np.ptp(P, axis=0).max()) or 1.0
    out = []
    for g in f["legs"]:
        d, j = _qc()._nearest_rows(np.asarray(g["pts"], float), P)
        if (d > 1e-9 * span).any():
            continue
        qe, qu = Q[j], np.asarray(g["q"], float)
        if not (float(qu @ qu) > 0 and float(qe @ qe) > 0):
            continue
        k = float(qe @ qu / (qu @ qu))
        out.append((g["c"], k, float(np.linalg.norm(qe - k * qu) / np.linalg.norm(qe)),
                    float(np.mean(qe * qu > 0))))
    return out or None


def _k_differs_by_leg(side_dir: Path):
    """(level, [(leg centre, implied k, misfit, share)]) where a side's exported flux and
    its own field imply a coefficient that differs between the legs of a bent interface by
    more than 30 %, read at the finest level whose legs can be read with a misfit below
    50 %; None otherwise. MEASURED on coupled runs: the side with several materials
    read 2.3 between its two legs at the finest level, the single-material sides 0.9-1.12.
    Only where the export and the field agree in sign at nine points in ten or more on every
    leg: a field that answers the opposite flux on a leg is what the own-field check is for,
    not a second material."""
    if not _iface_bends(Path(side_dir)):
        return None
    lvls = sorted({int(m.group(1)) for q in Path(side_dir).glob("field_level*.csv")
                   for m in [re.fullmatch(r"field_level(\d+)\.csv", q.name)] if m}, reverse=True)
    for lvl in lvls:
        kk = _implied_k_by_leg(Path(side_dir), lvl)
        if not kk or len(kk) < 2 or any(m > 0.5 for _c, _k, m, _s in kk):
            continue
        if any(k <= 0 or sh < 0.9 for _c, k, _m, sh in kk):
            return None
        mags = [k for _c, k, _m, _s in kk]
        return (lvl, kk) if max(mags) > 1.3 * min(mags) else None
    return None


def _one_box_one_k_cannot_state(side_dir: Path) -> str:
    """'' unless a side's own files show what one box and one k cannot state; then that, in
    plain words. A coefficient that differs between the legs of a bent interface (see
    _k_differs_by_leg), or a bent interface on a mesh that leaves part of the box its own
    nodes span empty (a box with a corner cut out)."""
    try:
        got = _k_differs_by_leg(side_dir)
    except Exception:                                        # noqa: BLE001
        got = None
    if got:
        lvl, kk = got
        return (f"at level {lvl} its exported flux over its own field's -du/dn reads "
                + " and ".join(f"{k:.4g} along the leg through ({c[0]:.4g}, {c[1]:.4g})" for c, k, _m, _s in kk)
                + ", so the coefficient its field carries differs along the interface")
    try:
        if not _iface_bends(side_dir):
            return ""
        dumps = sorted(Path(side_dir).glob("field_level*.csv"),
                       key=lambda q: int(re.sub(r"\D", "", q.stem) or 0))
        h, r = _read_named_csv(dumps[-1])
        import numpy as np
        xy = np.asarray(r, float)[:, [h.index("x"), h.index("y")]]
        box = [(float(xy[:, 0].min()), float(xy[:, 0].max())), (float(xy[:, 1].min()), float(xy[:, 1].max()))]
        gaps = _box_fill(Path(side_dir), box)
    except Exception:                                        # noqa: BLE001
        return ""
    if gaps:
        lvl = max(gaps)
        return (f"its interface bends, and at level {lvl} its mesh leaves part of the box its own "
                f"nodes span without nodes ({_gap_words(gaps[lvl])}): the side is not one box")
    return ""


def _iface_bends(side_dir: Path) -> bool:
    """Whether the side's own interface dump (its highest level) bends: not one straight line."""
    import numpy as np
    files = sorted(side_dir.glob("interface_level*.csv"),
                   key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    if not files:
        return False
    try:
        hi, ri = _read_named_csv(files[-1])
    except (OSError, StopIteration):
        return False
    if not ({"x", "y"} <= set(hi)) or len(ri) < 3:
        return False
    I = np.asarray(ri, float)[:, [hi.index("x"), hi.index("y")]]
    legs, bad = _curve_legs(I)
    return legs is not None or bad is not None


_K_COMPONENT = re.compile(r"k_?(xx|xy|yx|yy|xz|yz|zx|zy|zz)", re.I)


def _k_tensor(cfg) -> str:
    """'' unless a side's config.json states its conductivity as a tensor; then how,
    in plain words. A tensor is k (or K, kappa, conductivity) given as a matrix or a
    list of more than one number, a table of tensor components (xx, xy, ...), or
    component keys beside it (kxx, kxy, kyy ...) -- whatever scalar k is stated too.

    WITH A FULL TENSOR THE FLUX IS NOT -k du/dn. The outward flux -(K grad u) . n takes
    in the derivative along the interface as well, so q_n / (-du/dn) is not one
    constant along the interface even for an exact flux (measured by a critical
    review: CONSISTENT for K_xy = 0, INCONSISTENT for K_xy = 0.3 and 0.8, spreads 347%
    and 760% on two grids). The grading side abstains there; so do these checks."""
    if not isinstance(cfg, dict):
        return ""
    for key in ("k", "K", "kappa", "conductivity"):
        v = cfg.get(key)
        if isinstance(v, (list, tuple)) and (len(v) > 1 or any(isinstance(x, (list, tuple)) for x in v)):
            return f"states {key} as a matrix"
        if isinstance(v, dict) and len(v) >= 2 and all(
                _K_COMPONENT.fullmatch("k" + str(c).lower().lstrip("k_")) for c in v):
            return f"states {key} by its tensor components"
    comps = sorted(str(c) for c in cfg if _K_COMPONENT.fullmatch(str(c)))
    if comps:
        return f"states the conductivity by its components ({', '.join(comps)})"
    return ""


def _k_unstated(cfg) -> bool:
    """True when a side's config.json states no conductivity at all, in any form."""
    return isinstance(cfg, dict) and not any(c in cfg for c in ("k", "K", "kappa", "conductivity")) \
        and not any(_K_COMPONENT.fullmatch(str(c)) for c in cfg)


def _k_in_program(side_dir: Path) -> tuple:
    """("scalar" | "tensor" | "", file) -- the form of the conductivity a side's own Python
    program sets at its top level (K = ..., or k = ... where no K is set), read without
    running it. Every served heat contract carries its coefficient as such a constant, and a
    side that keeps its data in code states it only there. A number is a scalar; a list of
    more than one number, or a matrix, a tensor; anything else (a name, a call, a function)
    says nothing about the form, and neither does a program that sets two forms."""
    import ast as _ast

    def _form(v):
        if isinstance(v, _ast.UnaryOp) and isinstance(v.op, (_ast.USub, _ast.UAdd)):
            return _form(v.operand)
        if isinstance(v, _ast.Constant):
            return ("scalar" if isinstance(v.value, (int, float)) and not isinstance(v.value, bool)
                    else "")
        if isinstance(v, (_ast.List, _ast.Tuple)):
            return ("tensor" if len(v.elts) > 1 or any(isinstance(e, (_ast.List, _ast.Tuple))
                                                       for e in v.elts) else "")
        if isinstance(v, _ast.Call):
            nm = getattr(v.func, "attr", None) or getattr(v.func, "id", "")
            if nm in ("array", "asarray", "float") and v.args:
                f = _form(v.args[0])
                return f if nm != "float" or f == "scalar" else ""
        return ""

    found: dict = {}
    for p in sorted(Path(side_dir).glob("*.py")):
        if ".replaced-" in p.name:
            continue
        try:
            body = _ast.parse(p.read_text(errors="replace")).body
        except (OSError, SyntaxError, ValueError):
            continue
        for name in ("K", "k"):
            forms = {_form(st.value) for st in body if isinstance(st, _ast.Assign)
                     and any(isinstance(t, _ast.Name) and t.id == name for t in st.targets)}
            if forms:
                found[p.name] = forms
                break
    forms = set().union(*found.values()) if found else set()
    if len(forms) != 1 or "" in forms:
        return "", ""
    form = forms.pop()
    return form, next(n for n, f in found.items() if form in f)


def _no_single_k(cfg) -> str:
    """How a side's config.json falls short of one conductivity, in plain words."""
    if not isinstance(cfg, dict) or "k" not in cfg:
        return "states no k"
    kv = cfg.get("k")
    if isinstance(kv, bool) or not isinstance(kv, (int, float)):
        what = {"dict": "a table", "list": "a list", "str": "text"}.get(type(kv).__name__, "no number")
        return f"gives k as {what}, not as one number"
    return f"states k = {kv:g}, not a positive number"


def own_field_flux_findings(work: Path, dirs=None, since=None, levels=None,
                            scan: bool = True) -> list[dict]:
    """The flux each side's OWN field carries across the interface, against its partner's.

    THE EXPORTS CAN BE MADE TO AGREE; THE FIELDS CANNOT. A side can write its flux with any
    sign, and the exchange checks compare what the sides write. Measured on a coupled run:
    its Neumann side applied the partner's flux with the wrong sign, so its field answered a
    flux opposite to the one it was sent; the exported fluxes added (~2.0), the reply said
    the fix was "one sign on the number you write out, not a defect in the solve", the run
    flipped its written sign, its files balanced, and it handed in a field 40 % of the right
    size as converged. From the fields themselves -- -k du/dn outward, from each side's own
    dump and the k its config states -- the two fluxes still added. On a right coupling
    they cancel, to the accuracy of a one-sided difference, shrinking with the mesh.

    A BENT INTERFACE IS READ LEG BY LEG. With one normal axis, one plane and one k for a
    two-leg seam, an exactly right result set read "DO NOT CARRY OPPOSITE FLUXES ... 156%
    -> 180% -> 190%" (measured). Each leg now has its own normal and spacing, and the two
    sides are compared on the legs they share. Where that cannot be done -- a side states
    no single k, or its legs cannot be read -- the check does not run, and the hand-in
    audit (the call with no level, folder or time filter, which lists notes apart from
    findings) is told so in one informational note; the per-level callers list defects
    only, so they get nothing."""
    out: list[dict] = []
    _notes = levels is None and not dirs and since is None and scan
    seen, folders = set(), []
    for d in list(dirs or []) + (_side_dirs(work) if scan else []):
        try:
            key = Path(d).resolve()
        except OSError:
            continue
        if key not in seen and Path(d).is_dir():
            seen.add(key)
            folders.append(Path(d))
    ks = {}
    cfgs = {}
    # A 3-D SIDE IS NOT JUDGED: the flux is read with a 2-D stencil (see _dumps_are_3d).
    _3d = [d for d in folders if _dumps_are_3d(d)]
    if _3d:
        if _notes:
            out.append({"sequence": "own-field flux balance", "values": [], "priority": 26,
                        "informational": True, "finding": (
                "THE OWN-FIELD FLUX CHECK WAS NOT JUDGED IN 3-D: "
                + " and ".join(d.name for d in _3d) + " carry a z coordinate in their dumps, and "
                "this check reads -k du/dn on a 2-D box, so it does not judge them; no key in a "
                "config.json changes that.")})
        return out
    for d in folders:
        try:
            cfg = json.loads((d / "config.json").read_text() or "{}")
        except (OSError, ValueError):
            continue
        cfgs[d] = cfg
        if isinstance(cfg, dict) and isinstance(cfg.get("k"), (int, float)) and cfg["k"] > 0:
            ks[d] = float(cfg["k"])
        elif isinstance(cfg, dict) and _k_as_matrix(cfg.get("k")) is not None:
            # A 2x2 k: the flux is -(K grad u) . n, the derivative along the interface included
            # (see _own_field_flux). Measured on a coupled round's own dumps: the four right
            # results read 85 % -> 41 % -> 23 %, falling like the one-sided difference it is,
            # and the side that had solved with K[0][0] alone 55 % -> 33 % -> 30 %.
            ks[d] = _k_as_matrix(cfg.get("k"))
    # A TENSOR STATED ANOTHER WAY (a table of components, component keys): not judged; the
    # hand-in audit is told why. A scalar k beside tensor component keys is not the flux's k
    # (it is usually kxx), so such a side is not judged with it either.
    for d, c in cfgs.items():
        if _k_tensor(c) and not isinstance(ks.get(d), list):
            ks.pop(d, None)
    _tens = [(d, _k_tensor(c)) for d, c in sorted(cfgs.items(), key=lambda kv: kv[0].name)
             if _k_tensor(c) and d not in ks]
    if _tens:
        if _notes and len(cfgs) == 2:
            out.append({"sequence": "own-field flux balance", "values": [], "priority": 26,
                        "informational": True, "finding": (
                "THE OWN-FIELD FLUX CHECK DID NOT RUN (NOT CHECKED): "
                + " and ".join(f"{d.name}'s config.json {why}" for d, why in _tens)
                + ". It takes -k du/dn from each side's own field dump, and with a full "
                "tensor K the outward flux -(K grad u) . n also takes in the derivative "
                "along the interface, so nothing here is judged.")})
        return out
    if len(ks) != 2:
        # two sides, each stating its k. On a bent interface the reason is said.
        if _notes and len(cfgs) == 2 and any(_iface_bends(d) for d in cfgs):
            _short = [f"{d.name}'s config.json {_no_single_k(c)}"
                      for d, c in sorted(cfgs.items(), key=lambda kv: kv[0].name) if d not in ks]
            out.append({"sequence": "own-field flux balance", "values": [], "priority": 26,
                        "informational": True, "finding": (
                "THE OWN-FIELD FLUX CHECK DID NOT RUN (NOT CHECKED) on this bent interface: "
                + " and ".join(_short) + ". It takes -k du/dn from each side's own field dump "
                "with the one k that side states, so nothing here is judged.")})
        return out
    (da, ka), (db, kb) = sorted(ks.items(), key=lambda kv: kv[0].name)
    import numpy as np
    # ONE k CANNOT READ A SIDE WHOSE CONDUCTIVITY DIFFERS BETWEEN THE LEGS. Measured: a side
    # with several materials, read with the one k its config states, gave its own-field flux
    # a factor of 2.3 wrong on one leg of a right field. Where a side's exported flux and its
    # own field imply a coefficient that differs between its legs, nothing is judged, and the
    # hand-in audit is told what was measured.
    _several = [(d, _k_differs_by_leg(d)) for d in (da, db)]
    _several = [(d, got) for d, got in _several if got]
    if _several:
        if _notes:
            d, (lv_k, kk) = _several[0]
            out.append({"sequence": "own-field flux balance", "values": [], "priority": 26,
                        "informational": True, "finding": (
                "THE OWN-FIELD FLUX CHECK DID NOT RUN (NOT CHECKED): at level " + str(lv_k) + " "
                + d.name + "'s exported flux over its own field's -du/dn reads "
                + " and ".join(f"{k:.4g} along the leg through ({c[0]:.4g}, {c[1]:.4g})"
                               for c, k, _m, _s in kk)
                + ". A conductivity that differs by region along the interface reads this way, "
                "and so does a flux scaled on one leg; the check takes -k du/dn with the one k "
                "the side's config states, so nothing here is judged.")})
        return out
    rows = []
    why: list = []
    by_leg = False
    _read_lv: list = []
    lvls = sorted({int(m.group(1)) for d in (da, db) for q in d.glob("field_level*.csv")
                   for m in [re.fullmatch(r"field_level(\d+)\.csv", q.name)] if m})
    for lvl in lvls:
        if levels is not None and lvl not in levels:
            continue
        try:
            _cut = [_cutoff_for(d, since) for d in (da, db)]
            if any(c is not None and (d / f"field_level{lvl}.csv").stat().st_mtime < c
                   for c, d in zip(_cut, (da, db))):
                continue
        except OSError:
            continue
        _read_lv.append(lvl)
        fa, fb = _own_field_flux(da, lvl, ka, why), _own_field_flux(db, lvl, kb, why)
        if fa is None or fb is None:
            continue
        if isinstance(fa, dict) or isinstance(fb, dict):
            if not (isinstance(fa, dict) and isinstance(fb, dict)):
                continue                             # one side bent, one straight: not one interface
            b = _leg_pairs_balance(fa, fb)
            if b is not None:
                rows.append((lvl, b))
                by_leg = True
            continue
        qb_at_a = np.interp(fa[0], fb[0], fb[1])
        inside = (fa[0] >= fb[0].min()) & (fa[0] <= fb[0].max())
        if inside.sum() < 3:
            continue
        qa, qb = fa[1][inside], qb_at_a[inside]
        scale = max(float(np.abs(qa).max()), float(np.abs(qb).max()))
        if scale <= 0:
            continue
        rows.append((lvl, float(np.abs(qa + qb).max()) / scale))
    if not rows:
        if _notes and why:
            out.append({"sequence": "own-field flux balance", "values": [], "priority": 26,
                        "informational": True, "finding": (
                f"THE OWN-FIELD FLUX CHECK DID NOT RUN (NOT CHECKED): {why[0]}, so no normal "
                f"could be taken along them from the field. Nothing here is judged.")})
        return out
    lv = [l for l, _ in rows]
    bal = [b for _, b in rows]
    txt = " -> ".join(f"{b:.0%}" for b in bal)
    added = any(b >= 1.5 for b in bal)
    # "DOES NOT SHRINK" IS JUDGED ON THE LAST REFINEMENT STEP, AND ONLY WHERE IT HOLDS. On a right
    # coupling the rest is the one-sided difference's error and falls with every step. Measured
    # over the recorded coupled runs that read three levels: the right results' last step keeps
    # 50-77 % of the level before, the defects' 90-116 %. The old rule fired when any step kept
    # 60 % or more: it called 76 % -> 38 % -> 23 % "does not shrink" (a step that kept 60.2 %,
    # where right results of the same problem read 85 % -> 41 % -> 23 %) and led a hand-in
    # audit with it.
    _kept = bal[-1] / bal[-2] if len(bal) >= 2 and bal[-2] > 0 else 0.0
    stays = len(bal) >= 3 and all(b > 0.2 for b in bal) and _kept >= 0.8
    # TWO READABLE LEVELS ARE ONE STEP, AND ONE STEP IS SAID AS ONE. Measured: a level-1 mesh
    # whose legs keep fewer than three points once their end nodes are out gives no reading,
    # so a three-level ladder had two readable levels (86 % -> 103 %) and a rule that asked
    # for three never spoke.
    two = len(bal) == 2 and all(b > 0.2 for b in bal) and _kept >= 0.8
    if not (added or stays or two):
        return out
    _unread = [l for l in _read_lv if l not in lv]
    _step = f"the last refinement step (level {lv[-2]} to level {lv[-1]}) " if len(bal) >= 2 else ""
    _trend = ((f" and it does not shrink with the mesh: {_step}took it from {bal[-2]:.0%} up to "
               f"{bal[-1]:.0%}." if bal[-1] >= bal[-2] else
               f" and it barely shrinks with the mesh: {_step}took it only from {bal[-2]:.0%} to "
               f"{bal[-1]:.0%}.") if _step else ".")
    out.append({
        "sequence": "own-field flux balance", "values": bal, "priority": 4,
        "finding": (
            f"THE TWO FIELDS DO NOT CARRY OPPOSITE FLUXES ACROSS THE INTERFACE: the flux each "
            f"side's own field carries out of it ("
            + ("-(K grad u) . n" if any(isinstance(q, list) for q in (ka, kb)) else "-k du/dn")
            + (", leg by leg along the bent interface" if by_leg else "")
            + f", from {da.name}/ and {db.name}/'s "
            f"field dumps and the {'K' if any(isinstance(q, list) for q in (ka, kb)) else 'k'} each "
            f"config states) fails to cancel its partner's by "
            f"{txt} at level(s) {', '.join(str(l) for l in lv)}"
            + (" -- about 200 %: the two fields' fluxes carry the SAME sign, so one side's "
               "field answers a flux of the opposite sign to the one its partner sent. That "
               "is a flux APPLIED with the wrong sign, and it is fixed in the solve, not in "
               "the number written out."
               if added else
               _trend + " On a right coupling the two cancel up to the error of a one-sided "
               "difference, which falls with every refinement; here the field on one side does "
               "not carry the flux its partner's field does."
               + (" Two readable levels are one refinement step: they show what this one step "
                  "did, not how a further level would move"
                  + (f" (level(s) {', '.join(str(l) for l in _unread)} gave no reading)"
                     if _unread else "")
                  + "." if two and not stays else ""))
            + " A side's exports can be made to balance while its field does not; this reads "
            "the fields.")})
    return out


def nonfinite_field_findings(work: Path, dirs=None, since=None, levels=None,
                             scan: bool = True) -> list[dict]:
    """A side's own field dump holding NaN or inf.

    INVISIBLE UNTIL NAMED. The equation check's reader drops a dump whose rebuild is
    NaN, and the audit then said "neither two or more of its own mesh dumps ... were
    found" of a side whose three dumps existed and were NaN at 90-97 % of their nodes;
    couple() called that level "CONVERGED -- THIS IS A RESULT" (measured: the side's
    bilinear form was assembled from the solution function instead of the trial
    function, an all-zero matrix, and its exports carried finite placeholders)."""
    out: list[dict] = []
    seen, folders = set(), []
    for d in list(dirs or []) + (_side_dirs(work) if scan else []):
        try:
            key = Path(d).resolve()
        except OSError:
            continue
        if key not in seen and Path(d).is_dir():
            seen.add(key)
            folders.append(Path(d))
    for d in folders:
        worst = None
        for p in sorted(d.glob("field_level*.csv")):
            m = re.fullmatch(r"field_level(\d+)\.csv", p.name)
            if not m or (levels is not None and int(m.group(1)) not in levels):
                continue
            try:
                _cut = _cutoff_for(d, since)
                if _cut is not None and p.stat().st_mtime < _cut:
                    continue
                n = bad = 0
                with p.open() as fh:
                    next(fh, None)
                    for line in fh:
                        parts = line.strip().split(",")
                        if len(parts) < 3:
                            continue
                        n += 1
                        try:
                            if not all(math.isfinite(float(v)) for v in parts[2:]):
                                bad += 1
                        except ValueError:
                            bad += 1
            except OSError:
                continue
            if n and bad and (worst is None or bad / n > worst[2] / worst[3]):
                worst = (int(m.group(1)), p.name, bad, n)
        if worst is None:
            continue
        try:
            name = str(d.relative_to(work))
        except ValueError:
            name = d.name
        lvl, fname, bad, n = worst
        out.append({"sequence": f"nonfinite field {name}", "values": [bad, n], "priority": 2,
                    "finding": (
            f"{name.upper()}'S FIELD IS NOT FINITE: its own {fname} holds NaN or inf at {bad} of "
            f"{n} nodes. Whatever it exported, the solve that produced this field failed -- a "
            f"singular or empty system (a form built from the solution instead of the trial "
            f"function assembles zero), a division by zero -- and nothing built on it is a "
            f"result. Run this side standalone and look at its field before coupling again.")})
    return out


_FLUX_COLUMNS = ("qn", "q", "flux", "normal_flux", "normal_fluxes", "q_n", "qn_out")


def nonfinite_interface_findings(work: Path, dirs=None, since=None, levels=None,
                                 scan: bool = True) -> list[dict]:
    """A side's own interface dump or exports.json holding NaN or inf, a flux column above all.

    MEASURED on a coupled round: two give-up screens led with NEAR-ZERO FIELD and its hint to
    check the source, while the side's interface_level<k>.csv held NaN in its flux column at
    every row (a Kratos mesh with flat tetrahedra returned NaN reactions with no message; the
    served export stop kept exports.json from being written; the field dump held the values the
    solve started from). Nothing read that column: nonfinite_field_findings reads field dumps
    only, and the exchange check needs both sides' exports. Named at the priority of a
    non-finite field, so it leads a near-zero or constant field that is the same failure."""
    out: list[dict] = []
    files: dict = {}
    for q in _interface_files(work):
        m = re.search(r"level_?(\d+)", q.name, re.I)
        if levels is not None and not (m and int(m.group(1)) in levels):
            continue
        if dirs is not None and not scan and q.parent not in [Path(d) for d in dirs]:
            continue
        files.setdefault(q.parent, []).append(q)
    for d in (list(dirs or []) + (_side_dirs(work) if scan else [])):
        files.setdefault(Path(d), [])
    for d, qs in files.items():
        worst, said = None, []
        for q in sorted(qs):
            try:
                _cut = _cutoff_for(d, since)
                if _cut is not None and q.stat().st_mtime < _cut:
                    continue
                hdr, rows = _read_named_csv(q)
            except (OSError, ValueError):
                continue
            for i, col in enumerate(hdr):
                if col in ("x", "y", "z") or not rows:
                    continue
                bad = sum(1 for r in rows if not math.isfinite(r[i]))
                if bad and (worst is None or (col in _FLUX_COLUMNS, bad / len(rows))
                            > (worst[1] in _FLUX_COLUMNS, worst[2] / worst[3])):
                    worst = (q.name, col, bad, len(rows))
        e = d / "exports.json"
        if e.is_file():
            try:
                _cut = _cutoff_for(d, since)
                ex = json.loads(e.read_text() or "{}") if (_cut is None or e.stat().st_mtime >= _cut) else {}
                for key in ("normal_fluxes", "values"):
                    vals = [v for v in _flat(ex.get(key) or []) if isinstance(v, (int, float))]
                    bad = sum(1 for v in vals if not math.isfinite(v))
                    if bad:
                        said.append(f"its exports.json in `{key}` at {bad} of {len(vals)} entries")
                        if worst is None:
                            worst = ("exports.json", key, bad, len(vals))
            except (OSError, ValueError, TypeError, AttributeError):
                pass
        if worst is None:
            continue
        fname, col, bad, n = worst
        if fname != "exports.json":
            said.insert(0, f"its own {fname} holds NaN or inf in column `{col}` at {bad} of {n} rows")
        try:
            name = str(d.relative_to(work))
        except ValueError:
            name = d.name
        who = "THE DELIVERED" if name in ("", ".") else f"{name.upper()}'S"
        flux = col in _FLUX_COLUMNS
        try:
            kratos = any("KratosMultiphysics" in p.read_text(errors="ignore")
                         for p in d.glob("*.py") if ".replaced-" not in p.name)
        except OSError:
            kratos = False
        out.append({"sequence": f"nonfinite interface {name}", "values": [bad, n], "priority": 2,
                    "finding": (
            f"{who} INTERFACE {'FLUX' if flux else 'VALUE'} IS NOT FINITE: "
            + "; ".join(said) + ". Whatever the other columns show -- a near-zero or constant "
            "field beside it is the same failure, not a small answer -- the side that wrote it did "
            f"not produce a usable interface {'flux' if flux else 'value'}, and nothing built on it is "
            "a result. "
            + ("Measured on this install: flat tetrahedra in a Kratos mesh give NaN fluxes with no "
               "message, and 'Error zero sum' leaves NaN too. " if kratos else "")
            + "Run this side standalone and read its own console before coupling again.")})
    return out


def exports_not_the_field_findings(work: Path, dirs=None, since=None, levels=None,
                                   scan: bool = True) -> list[dict]:
    """A side's interface dump does not carry its own field at the same points.

    BOTH FILES ARE THE SAME RUN'S. Every served participant writes its whole field
    (field_level<k>.csv, from the mesh's own nodes) and its interface trace and
    flux (interface_level<k>.csv, through the participant's interface list) from
    one solve, so at a point both files hold, the value columns they share are the
    same number. Measured on a coupled run: the interface list held the dofs of
    the mesh's first vertices, the trace was written onto those and exported from
    them, and the field was 0.0 at most of the true interface nodes -- 100 % of the
    trace at every level, while both interface files agreed with each other to
    1e-10 and every exchange check passed (its convergence order stayed in band
    because the partner's conductivity kept the trace small). Measured over the 550
    recorded coupled sides whose two dumps share points (312 runs): it names six --
    that side, a side that wrote an invented field beside its exports, and four
    sides of runs that failed -- and none of the other 91 sides of runs whose
    convergence order came out right.
    """
    out: list[dict] = []
    try:
        import numpy as _np
    except Exception:                                    # noqa: BLE001
        return out
    seen, folders = set(), []
    for d in list(dirs or []) + (_side_dirs(work) if scan else []):   # scan=False: `dirs` alone
        try:
            key = Path(d).resolve()
        except OSError:
            continue
        if key not in seen and Path(d).is_dir():
            seen.add(key)
            folders.append(Path(d))
    for d in folders:
        worst = None
        for p in sorted(d.glob("interface_level*.csv")):
            m = re.search(r"interface_level(\d+)\.csv$", p.name)
            q = d / f"field_level{m.group(1)}.csv" if m else None
            if not (m and q.is_file()):
                continue
            lvl = int(m.group(1))
            if levels is not None and lvl not in levels:
                continue
            try:
                _cut = _cutoff_for(d, since)
                if _cut is not None and min(p.stat().st_mtime, q.stat().st_mtime) < _cut:
                    continue                 # the last run's dump, not this one's
                hi, ri = _read_named_csv(p)
                hf, rf = _read_named_csv(q)
            except OSError:
                continue
            coords = [c for c in ("x", "y", "z") if c in hi and c in hf]
            vals = [c for c in hf if c not in ("x", "y", "z") and c in hi]
            if len(coords) < 2 or not vals or len(ri) < 3 or len(rf) < 3:
                continue
            I = _np.asarray(ri, float)
            F = _np.asarray(rf, float)
            ci, cf = [hi.index(c) for c in coords], [hf.index(c) for c in coords]
            vi, vf = [hi.index(c) for c in vals], [hf.index(c) for c in vals]
            span = float(_np.ptp(F[:, cf], axis=0).max()) or 1.0
            tol = 1e-8 * span
            gaps, matched = [], 0
            for row in I:
                dist = _np.abs(F[:, cf] - row[ci]).max(axis=1)
                hit = _np.where(dist <= tol)[0]
                if not hit.size:
                    continue
                matched += 1
                # a node listed more than once (a discontinuous field) is matched by its closest copy
                gaps.append(float(_np.abs(F[hit][:, vf] - row[vi]).max(axis=1).min()))
            if matched < max(3, 0.5 * len(I)):
                continue                     # the two files do not share their points: not judged
            at = _np.array([r for r in I], float)
            # THE FIELD'S OWN SIZE IS THE SCALE, NOT THE INTERFACE VALUES ALONE. Measured: two files
            # that were both round-off zero (a stub side with no load) read as a 100 % mismatch.
            _fpk = float(_np.abs(F[:, vf]).max()) if F.size else 0.0
            _ipk = float(_np.abs(at[:, vi]).max())
            if max(_fpk, _ipk) <= 1e-12:
                continue                     # both effectively zero: NEAR-ZERO FIELD is the finding there
            scale = max(_ipk, _fpk)
            g = _np.asarray(gaps, float)
            rel = float(g.max()) / scale
            if not rel > 1e-6:
                continue
            n_bad = int((g > 1e-6 * scale).sum())
            try:
                _dt = p.stat().st_mtime - q.stat().st_mtime        # > 0: the interface file is the newer
            except OSError:
                _dt = None
            if worst is None or rel > worst[0]:
                worst = (rel, lvl, n_bad, matched, float(g.max()), "/".join(vals), _dt)
        if worst is None:
            continue
        rel, lvl, n_bad, matched, gmax, names, _dt = worst
        # WHETHER THE TWO FILES ARE ONE RUN'S IS MEASURED, NOT ASSUMED. Measured: two
        # runs whose files came from different runs (an uncoupled re-run beside the
        # coupled one) were told "both files come from the same solve".
        if _dt is None:
            _when = "Their write times could not be read. "
        elif abs(_dt) > 5.0:
            _when = (f"The two files were written {abs(_dt):.0f} s apart (the "
                     f"{'field dump' if _dt > 0 else 'interface file'} first), so they are not "
                     f"one run's output: a file left by another run (an uncoupled re-run, an "
                     f"earlier level) does not describe this coupling. ")
        else:
            _when = (f"The two files were written {abs(_dt):.1f} s apart, by one run, and one "
                     f"run's two files hold the same number at a shared point. ")
        try:
            name = str(d.relative_to(work))
        except ValueError:
            name = d.name
        out.append({
            "sequence": f"exports not the field {name}", "values": [rel], "priority": 3,
            "finding": (
                f"{name}'S INTERFACE FILE DOES NOT CARRY ITS OWN FIELD: at level {lvl}, at "
                f"{n_bad} of the {matched} interface points that its field_level{lvl}.csv also "
                f"holds, the {names} there differs from interface_level{lvl}.csv by up to "
                f"{gmax:.3g} ({rel:.0%} of the interface values' own scale). " + _when
                + f"The interface values and fluxes this side exports are read through its "
                f"interface list; the field dump is read from the mesh's own nodes. Where they "
                f"differ, what the partner received is not what this side's field is at the "
                f"interface, and the exchange checks cannot see it: they compare the exports "
                f"with each other.")})
    return out


def _k_as_matrix(k):
    """[[kxx, kxy], [kyx, kyy]] as floats when a config's k is a 2x2 matrix -- nested, or four
    numbers row by row as the served deal.II wrapper reads them; None for anything else."""
    if not isinstance(k, (list, tuple)):
        return None
    if len(k) == 2 and all(isinstance(r, (list, tuple)) and len(r) == 2 for r in k):
        flat = [k[0][0], k[0][1], k[1][0], k[1][1]]
    elif len(k) == 4 and not any(isinstance(r, (list, tuple, dict)) for r in k):
        flat = list(k)
    else:
        return None
    try:
        v = [float(x) for x in flat]
    except (TypeError, ValueError):
        return None
    if isinstance(flat[0], bool) or not all(math.isfinite(x) for x in v):
        return None
    return [[v[0], v[1]], [v[2], v[3]]]


def _k_text(k) -> str:
    return (f"[[{k[0][0]:g}, {k[0][1]:g}], [{k[1][0]:g}, {k[1][1]:g}]]" if isinstance(k, list)
            else f"{float(k):g}")


def _dumps_are_3d(d: Path) -> bool:
    """True when a side's own dumps carry a z coordinate that varies, or its config states a
    z extent or nz: the side is three-dimensional. The equation, own-field flux, outer-boundary
    and stated-data checks read a field on a 2-D box and do not judge such a side.

    MEASURED on a correct 3-D result set (the served Kratos 3-D contract on both sides, coupled
    Dirichlet-Neumann to 1e-11): the equation check asked each side for "the subdomain box (x0,
    x1, y0, y1)", and with one stated it ran its 2-D stencil on the 3-D dumps; the own-field flux
    check did the same and said the two fields do not carry opposite fluxes."""
    files = [Path(d) / "config.json"] + [q for pat in ("field_level*.csv", "interface_level*.csv")
                                          for q in sorted(Path(d).glob(pat))[:2]]
    try:
        stamp = (str(Path(d).resolve()), tuple((q.name, q.stat().st_mtime) for q in files if q.is_file()))
    except OSError:
        stamp = None
    if stamp is not None and stamp in _THREE_D_SEEN:
        return _THREE_D_SEEN[stamp]
    found = _dumps_are_3d_uncached(d)
    if stamp is not None:
        _THREE_D_SEEN[stamp] = found
    return found


_THREE_D_SEEN: dict = {}


def _dumps_are_3d_uncached(d: Path) -> bool:
    try:
        cfg = json.loads((Path(d) / "config.json").read_text() or "{}")
        if isinstance(cfg, dict) and any(k in cfg for k in ("nz", "z0", "z1")):
            return True
    except (OSError, ValueError):
        pass
    for pat in ("field_level*.csv", "interface_level*.csv"):
        for q in sorted(Path(d).glob(pat))[:2]:
            try:
                with q.open() as fh:
                    hdr = [h.strip().lower() for h in (fh.readline() or "").split(",")]
                    if "z" not in hdr:
                        continue
                    iz, first = hdr.index("z"), None
                    for i, line in enumerate(fh):
                        parts = line.split(",")
                        try:
                            z = float(parts[iz])
                        except (IndexError, ValueError):
                            continue
                        if first is None:
                            first = z
                        elif abs(z - first) > 1e-12 * max(1.0, abs(first)):
                            return True
                        if i > 20000:
                            break
            except OSError:
                continue
    return False


def _not_judged_in_3d(side: str, what: str) -> str:
    """The one sentence a 3-D side gets from a check that reads a 2-D box."""
    return (f"SIDE {side}'S {what} WAS NOT JUDGED IN 3-D: its dumps carry a z coordinate, and this "
            f"check reads the field on a 2-D box (x and y only), so it does not judge this side; "
            f"no key in a config.json changes that.")


# A FLUID SIDE IS NOT A HEAT OR AN ELASTIC SIDE. Measured on a fluid-structure round: the equation and
# outer-boundary checks asked a fluid side for "the coefficient k ... and source_expr", "lam and mu"
# and the one value its outer edges hold -- data a flow does not have -- in couple() and in every
# give-up screen. No check here judges a flow, and the text did not say so.
_FLOW_WORDS = re.compile(r"\b(?:navier|stokes)\b", re.I)


def _fluid_side(d: Path) -> str:
    """What shows that the side in `d` solves a flow, or '': its exports carry the traction a flow
    exerts on a structure, or its participant script names Navier-Stokes / Stokes or solves on a
    mixed space with a pressure."""
    try:
        e = json.loads((Path(d) / "exports.json").read_text() or "{}")
        if isinstance(e, dict) and str(e.get("field_name") or "").startswith("traction_on_structure"):
            return "its exports.json carries field_name 'traction_on_structure', a flow's load on a structure"
    except Exception:                                        # noqa: BLE001
        pass
    for q in sorted(Path(d).glob("participant*.py")):
        if _is_backup(q) or ".replaced-" in q.name:
            continue
        try:
            t = q.read_text(errors="ignore")
        except OSError:
            continue
        if _FLOW_WORDS.search(t) or ("mixed_element" in t and re.search(r"\bpressure\b", t, re.I)):
            return f"{q.name} solves a flow"
    return ""


def _not_judged_as_fluid(lead: str, why: str, check: str) -> str:
    """The one sentence a fluid side gets from a check that judges a heat or an elastic field."""
    return (f"{lead}: it is a fluid side ({why}), and this check judges {check}, so it does not "
            f"judge this side; no key in its config.json changes that.")


def _is_dsmc_side(d: Path) -> bool:
    """True when the side in `d` is a DSMC side: its participant drives SPARTA (a deck with surface
    commands, or the binary), or SPARTA's log is there. Such a side has no finite element field,
    so the checks that ask for one -- its equation, its outer edges -- do not apply to it (measured:
    a SPARTA side was asked to state a conductivity, a source, a box and an outer value)."""
    try:
        if (Path(d) / "log.sparta").is_file():
            return True
        for q in Path(d).glob("participant*.py"):
            if _is_backup(q) or ".replaced-" in q.name:
                continue
            if re.search(r"\bsurf_collide\b|\bread_surf\b|\bspa_(?:serial|mpi)\b",
                         q.read_text(errors="ignore")):
                return True
    except OSError:
        return False
    return False


def _exchanges_a_vector(d: Path) -> bool:
    """A side whose own traffic is a vector per point: exports.json values of two or three
    numbers each, or a field dump whose header names ux and uy."""
    try:
        v = (json.loads((d / "exports.json").read_text() or "{}").get("values") or [None])[0]
        if isinstance(v, (list, tuple)) and len(v) in (2, 3):
            return True
    except Exception:                                        # noqa: BLE001
        pass
    for f in sorted(d.glob("field_level*.csv"))[:1]:
        try:
            with f.open() as fh:
                head = fh.readline().lower().replace(" ", "").strip().split(",")
        except OSError:
            continue
        if "ux" in head and "uy" in head:
            return True
    return False


def _side_operators(work: Path) -> dict:
    """Each side's operator, as that side's OWN config.json states it.

    The served contracts read ./config.json for the level keys and carry the
    subdomain's box, coefficient, reaction and source in the same file, so the
    operator is already on disk next to the participant that solved it. Nothing
    here comes from a task file or a key: it is the agent's own statement of
    the problem it implemented, which is what its own field can be judged
    against.
    """
    out = {}
    cfgs = sorted(work.glob("*/config.json"))
    # A SIDE ONE FOLDER DEEPER IS STILL A SIDE (see _side_dirs): a run that kept
    # its participants in work/coupled_run/side_A had no side here at all.
    cfgs += [d / "config.json" for d in _side_dirs(work)
             if d.parent.resolve() != work.resolve() and (d / "config.json").is_file()]
    for cfg_path in cfgs:
        try:
            cfg = json.loads(cfg_path.read_text() or "{}")
        except Exception:                                    # noqa: BLE001
            continue
        if not isinstance(cfg, dict):
            continue
        side = cfg_path.parent.name
        if side in out:
            side = str(cfg_path.parent.relative_to(work))
        missing = [k for k in ("x0", "x1", "y0", "y1", "k") if k not in cfg]
        entry = {"dir": cfg_path.parent, "missing": missing,
                 "three_d": _dumps_are_3d(cfg_path.parent), "fluid": _fluid_side(cfg_path.parent),
                 "dsmc": _is_dsmc_side(cfg_path.parent)}
        # THE BOX DOES NOT DEPEND ON THE CONDUCTIVITY. It was read only when k
        # was present too, so an elastic side -- which states lam and mu, or E
        # and nu, and no k -- had no box and its displacement was never judged.
        if all(_k in cfg for _k in ("x0", "x1", "y0", "y1")):
            try:
                entry["box"] = [(float(cfg["x0"]), float(cfg["x1"])),
                                (float(cfg["y0"]), float(cfg["y1"]))]
            except (TypeError, ValueError):
                pass
        if not missing:
            try:
                if "box" not in entry:
                    raise ValueError("box")
                # A 2x2 k IS ONE COEFFICIENT. The served NGSolve contract reads config.json's k as
                # [[kxx, kxy], [kyx, kyy]] and the deal.II wrapper as four numbers row by row, and
                # this reader asked for a number: every such side went unjudged (measured on a
                # coupled round -- a side solving with K[0][0] alone passed with nothing said).
                _kt = _k_as_matrix(cfg["k"])
                entry["k"] = _kt if _kt is not None else float(cfg["k"])
                entry["k_tensor"] = _kt is not None
                entry["reaction"] = float(cfg.get("reaction") or 0.0)
            except (TypeError, ValueError):
                entry["missing"] = ["a numeric x0, x1, y0, y1 and k"]
        entry["iface"] = str(cfg.get("iface") or "").strip().lower()
        entry["iface_axis"] = str(cfg.get("iface_axis") or "").strip().lower()[:1]
        if "outer" in cfg:
            try:
                entry["outer"] = float(cfg["outer"])
            except (TypeError, ValueError):
                pass
        # A THERMO-MECHANICAL SIDE STATES ITS HEAT SOURCE AS source_T. The
        # served thermo-elastic contracts read x0..y1, k and source_T (plus
        # the mechanical data) from config.json, not source_expr, so this
        # check judged NOTHING on any side of a whole coupled round while
        # its "not checked" note -- filtered out of the ladder reply -- named
        # a key those contracts never used. The temperature equation of such
        # a side is exactly -div(k grad T) = f_T, so source_T IS the operator's
        # source and the field to judge is the T column.
        src = str(cfg.get("source_expr") or cfg.get("source")
                  or cfg.get("source_T") or "").strip()
        entry["source"] = src.replace("^", "**")
        # THE MOMENTUM EQUATION OF A THERMO-ELASTIC SIDE is stated by the same
        # file: lam, mu, beta and the two body-force components. Read here so
        # the displacement is judged as the temperature is (see
        # _momentum_findings); a side without them is told so, not guessed at.
        _lm = None
        # "lambda" is how a task usually writes it; a side that stated it so
        # was read as having no elastic data at all
        _lam_key = next((_k for _k in ("lam", "lambda", "lmbda") if _k in cfg), None)
        try:
            if _lam_key and "mu" in cfg:
                _lm = (float(cfg[_lam_key]), float(cfg["mu"]))
            elif "E" in cfg and "nu" in cfg:
                # the elastic contracts accept E and nu as well; plane strain uses
                # the three-dimensional Lame constants
                _E, _nu = float(cfg["E"]), float(cfg["nu"])
                _lm = (_E * _nu / ((1.0 + _nu) * (1.0 - 2.0 * _nu)), _E / (2.0 * (1.0 + _nu)))
        except (TypeError, ValueError, ZeroDivisionError):
            _lm = None
        if _lm is not None:
            try:
                entry["elastic"] = {
                    "lam": _lm[0], "mu": _lm[1],
                    "beta": float(cfg.get("beta") or 0.0),
                    "fx": (None if cfg.get("source_ux") is None
                           else str(cfg["source_ux"]).strip().replace("^", "**")),
                    "fy": (None if cfg.get("source_uy") is None
                           else str(cfg["source_uy"]).strip().replace("^", "**"))}
            except (TypeError, ValueError):
                pass
        entry["field"] = ("T" if not (cfg.get("source_expr") or cfg.get("source"))
                          and cfg.get("source_T") else None)
        if not src and "elastic" not in entry:
            entry["missing"] = entry["missing"] + ["source_expr (or source_T for the temperature equation)"]
            entry["vector"] = _exchanges_a_vector(cfg_path.parent)
        # AN ISOTHERMAL ELASTIC SIDE HAS NO SCALAR EQUATION TO STATE: it is
        # judged by the momentum check alone, and asking it for a conductivity
        # and a scalar source would send it after keys its problem does not have.
        entry["scalar_na"] = "elastic" in entry and not src
        out[side] = entry
    return out


def _per_level_field_sets(work: Path, box: list) -> list:
    """Sets of the agent's own CSVs that are per-level fields on `box`.

    Found by SHAPE, never by name: files whose names differ only in one integer
    are one set, the integer is the level, and a file joins the set only if its
    rows are (x, y, value) inside the box AND form a grid of that box's cell
    midpoints -- the arrangement the identity's quadrature is exact on. A
    participant's own mesh dump is uniform but sits on the cell corners, and is
    refused there (see pde_consistency._detect_midpoint_grid), which is the
    point: it is the file that must not be judged.
    """
    from .pde_consistency import _detect_midpoint_grid

    groups: dict = {}
    seen = set()
    for path in sorted(list(work.glob("*.csv")) + list(work.glob("*/*.csv"))):
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        nums = re.findall(r"\d+", path.name)
        if not nums:
            continue
        rows = []
        try:
            with path.open() as fh:
                for row in _csv.reader(fh):
                    try:
                        vals = tuple(float(c) for c in row)
                    except ValueError:
                        continue                              # header
                    if len(vals) >= 3:
                        rows.append(vals[:3])
        except OSError:
            continue
        if len(rows) < 4:
            continue
        pts = [r[:2] for r in rows]
        (lo_x, hi_x), (lo_y, hi_y) = box
        pad = 1e-9 + 1e-6 * max(hi_x - lo_x, hi_y - lo_y)
        if not all(lo_x - pad <= p[0] <= hi_x + pad and lo_y - pad <= p[1] <= hi_y + pad
                   for p in pts):
            continue
        weight, _why = _detect_midpoint_grid(pts, box)
        if weight is None:
            continue
        key = re.sub(r"\d+", "#", path.name)
        groups.setdefault(key, {})[int(nums[0])] = rows
    return sorted((g for g in groups.values() if len(g) >= 2),
                  key=lambda g: (len(g), len(next(iter(g.values())))), reverse=True)


def _per_level_vector_sets(work: Path, box: list) -> list:
    """Sets of the agent's own CSVs that carry (x, y, T, ux, uy) on this box's
    cell midpoints -- the shape `_per_level_field_sets` finds for one scalar
    column, read here by header name for the three thermo-elastic columns."""
    from .pde_consistency import _detect_midpoint_grid

    groups: dict = {}
    seen = set()
    need = ("x", "y", "ux", "uy")             # T rides along when present, else 0
    for path in sorted(list(work.glob("*.csv")) + list(work.glob("*/*.csv"))):
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        nums = re.findall(r"\d+", path.name)
        if not nums:
            continue
        rows = []
        try:
            with path.open() as fh:
                rd = _csv.reader(fh)
                header = next(rd, None)
                if not header:
                    continue
                cols = [c.strip().strip('"').lower() for c in header]
                if not all(c in cols for c in need):
                    continue
                ix, iy, iux, iuy = (cols.index(c) for c in need)
                it = cols.index("t") if "t" in cols else None
                for row in rd:
                    try:
                        rows.append((float(row[ix]), float(row[iy]),
                                     float(row[it]) if it is not None else 0.0,
                                     float(row[iux]), float(row[iuy])))
                    except (ValueError, IndexError):
                        continue
        except OSError:
            continue
        if len(rows) < 4:
            continue
        pts = [r[:2] for r in rows]
        (lo_x, hi_x), (lo_y, hi_y) = box
        pad = 1e-9 + 1e-6 * max(hi_x - lo_x, hi_y - lo_y)
        if not all(lo_x - pad <= p[0] <= hi_x + pad and lo_y - pad <= p[1] <= hi_y + pad
                   for p in pts):
            continue
        weight, _why = _detect_midpoint_grid(pts, box)
        if weight is None:
            continue
        key = re.sub(r"\d+", "#", path.name)
        groups.setdefault(key, {})[int(nums[0])] = rows
    return sorted((g for g in groups.values() if len(g) >= 2),
                  key=lambda g: (len(g), len(next(iter(g.values())))), reverse=True)


def _source_is_zero(expr) -> bool:
    """True when the expression is identically zero on a few sample points
    ("0", "0.0", "0*x" alike); False when it cannot be evaluated."""
    if expr is None:
        return False
    try:
        from .pde_consistency import _eval_source
        import numpy as _np
        pts = [(0.13, 0.29), (0.61, 0.47), (0.83, 0.91), (0.37, 0.07)]
        return bool(_np.allclose(_eval_source(str(expr), pts, 2), 0.0))
    except Exception:                                    # noqa: BLE001
        return False


def _cutoff_for(side_dir, since):
    """The age below which a side's dump is the last run's: a number for every side,
    or a mapping from a side's folder to its own cutoff (see couple_levels: a dump
    is current when it is newer than the participant script and config that wrote
    it, so a level coupled in an earlier call of the same setup still counts)."""
    if isinstance(since, dict):
        try:
            return since.get(str(Path(side_dir).resolve()))
        except OSError:
            return None
    return since


def _dump_level_sets(side_dir: Path, box: list, kind: str = "scalar", field: str | None = None,
                     since: float | None = None) -> dict:
    """{level: rows at the cell midpoints of `box`} from a side's OWN mesh dumps.

    THE SOLVER'S FIELD, JUDGED WHERE THE QUADRATURE IS EXACT. A participant's
    field_level<k>.csv sits on its mesh nodes, where the midpoint rule is
    meaningless (see pde_consistency._detect_midpoint_grid), so it is
    interpolated -- linearly, exact for the P1 fields the served contracts write
    -- to the centres of a uniform grid over the box. This is the field the solve
    produced, before anything the agent did to it, so it can be judged at the
    moment a coupling returns, when no delivered file exists yet.

    MEASURED over the recorded coupled sides whose config states the operator
    (73 sides of correct runs with three dumped levels): judged this way 72 read
    CONSISTENT and one was refused as meaningless (residual above 1e6); the same
    fields scaled by 1.01 -- a source one percent off -- all read INCONSISTENT,
    and scaled by 1.005, 5 of 73 still pass; scaled by 0.99, about half of the
    sides on tensor grids still pass (99 of 184, measured 2026-09-25). One side whose delivered files had
    been called wrong (their third level rose) reads CONSISTENT on its own dumps.

    `kind` "scalar" gives rows (x, y, u) -- the column named `field` (e.g. "T"),
    else "u", else the only value column; "vector" gives (x, y, T, ux, uy) with
    T = 0 when the dump carries none. Dumps older than `since` are the last
    run's and are skipped. A dump that does not cover the box is not judged."""
    try:
        import numpy as np
        from scipy.interpolate import griddata
    except Exception:                                        # noqa: BLE001
        return {}
    out: dict = {}
    (x0, x1), (y0, y1) = box
    since = _cutoff_for(side_dir, since)
    for p in sorted(Path(side_dir).glob("field_level*.csv")):
        m = re.fullmatch(r"field_level(\d+)\.csv", p.name)
        if not m:
            continue
        try:
            if since is not None and p.stat().st_mtime < since:
                continue
            with p.open() as fh:
                rdr = _csv.reader(fh)
                head = [h.strip().lower() for h in next(rdr, [])]
                rows = []
                for r in rdr:
                    try:
                        rows.append([float(c) for c in r])
                    except ValueError:
                        continue
        except (OSError, StopIteration):
            continue
        if len(rows) < 8 or any(len(r) != len(rows[0]) for r in rows):
            continue
        a = np.asarray(rows, float)
        # ONE POINT SET, ONE ANSWER, WHATEVER THE ROW ORDER. The linear interpolant
        # triangulates the points, and on a structured grid the diagonal of each
        # square cell is a tie the triangulation breaks by input order: the same
        # field written in another row order gave other numbers (measured; the
        # verdict held). Rows are taken in coordinate order, each point once.
        try:
            _, _first = np.unique(np.round(a[:, :2], 12), axis=0, return_index=True)
            a = a[_first]
        except Exception:                                    # noqa: BLE001
            pass
        ncol = a.shape[1]
        def col(*names):
            for nm in names:
                if nm and nm.lower() in head:
                    return head.index(nm.lower())
            return None
        if kind == "scalar":
            c = col(field, "u") if field else col("u")
            if c is None and ncol == 3:
                c = 2
            if c is None:
                continue
            cols = [c]
        else:
            cx, cy = col("ux", "u_x"), col("uy", "u_y")
            if (cx is None or cy is None) and ncol == 4:
                cx, cy = 2, 3
            if cx is None or cy is None:
                continue
            cols = [col("t"), cx, cy]
        # THE GRID GROWS WITH THE MESH: a fixed grid puts a quadrature floor under
        # every level, and an exact field read FLAT on it (4.019e-04 at all three
        # levels) and was failed; 4 sqrt(N) cells a side falls with the mesh.
        n = max(16, min(512, int(round(4 * math.sqrt(len(a))))))
        xs = x0 + (np.arange(n) + 0.5) * (x1 - x0) / n
        ys = y0 + (np.arange(n) + 0.5) * (y1 - y0) / n
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        vals = []
        ok = True
        for c in cols:
            if c is None:
                vals.append(np.zeros(X.size))
                continue
            v = griddata(a[:, :2], a[:, c], (X, Y), method="linear").ravel()
            miss = np.isnan(v)
            if miss.mean() > 0.02:
                ok = False                   # the dump does not cover this box
                break
            if miss.any():
                v[miss] = griddata(a[:, :2], a[:, c], (X.ravel()[miss], Y.ravel()[miss]),
                                   method="nearest")
            vals.append(v)
        if not ok:
            continue
        pts = np.column_stack([X.ravel(), Y.ravel()] + vals)
        out[int(m.group(1))] = [tuple(float(z) for z in r) for r in pts]
    return out


def _dump_raw_levels(side_dir: Path, field: str | None = None, since=None) -> dict:
    """{level: [(x, y, u), ...]} straight from a side's own scalar dumps, no rebuild."""
    out: dict = {}
    cut = _cutoff_for(side_dir, since)
    for p in sorted(Path(side_dir).glob("field_level*.csv")):
        m = re.fullmatch(r"field_level(\d+)\.csv", p.name)
        if not m:
            continue
        try:
            if cut is not None and p.stat().st_mtime < cut:
                continue
            hdr, rows = _read_named_csv(p)
        except (OSError, StopIteration):
            continue
        if not rows or "x" not in hdr or "y" not in hdr:
            continue
        c = hdr.index(field.lower()) if field and field.lower() in hdr else (
            hdr.index("u") if "u" in hdr else (2 if len(hdr) == 3 else None))
        if c is None:
            continue
        ix, iy = hdr.index("x"), hdr.index("y")
        out[int(m.group(1))] = [(r[ix], r[iy], r[c]) for r in rows]
    return out


def _on_the_grid_itself(dres, raw: dict, op: dict):
    """A DOES NOT SATISFY read through the Delaunay rebuild, judged again on a tensor grid
    with bilinear cells (pde_consistency.check_levels_on_grid, rule 'q1').

    THE REBUILD ACCUSED A RIGHT SIDE. On a tensor grid every cell's four corners are
    cocircular, and the rebuild picks either diagonal cell by cell; a Kratos side that
    solved its own P1 system to 1e-12, with the source and k its config states, read
    5.990e-04 -> 5.826e-04 -> 2.177e-04, DOES NOT SATISFY, and the run gave up on it.
    Read with one uniform reconstruction the same field falls 2.0e-03 -> 9.8e-04 -> 2.7e-04.

    ONLY TO LIFT A REFUSAL, AND ONLY ON A CLEAN FALL. Measured over the recorded sides on
    tensor grids, as they are and scaled by 0.99 and 1.01: lifting a refusal when the
    bilinear reading falls by half at every step changes one verdict -- that side's -- and
    admits no scaled field; replacing the rebuild by that reading outright would change
    fifteen, and accuse a right side. Picking the most favourable of three reconstructions
    let a field scaled by 0.99 through on 50 of 119 right sides. Returns `dres` unchanged
    otherwise."""
    if not (dres and str(dres.get("verdict")) == "INCONSISTENT" and len(raw) >= 3):
        return dres
    try:
        from .pde_consistency import check_levels_on_grid, tensor_grid
        if any(tensor_grid(rows) is None for rows in raw.values()):
            return dres
        r = check_levels_on_grid(raw, op["source"], op["k"], op["box"],
                                 reaction=op["reaction"], rule="q1").as_dict()
        seq = [l.get("relative_weak_residual") for l in r.get("levels", [])]
        if (str(r.get("verdict")) == "CONSISTENT" and len(seq) >= 3
                and all(isinstance(x, float) for x in seq)
                and all(q < 0.5 * p for p, q in zip(seq, seq[1:]))):
            r["explanation"] = (str(r.get("explanation") or "") + " Read on the side's own grid "
                                "with bilinear cells: its nodes form a tensor grid, and the "
                                "scattered-node rebuild mixes the two diagonals of its cells.")
            return r
    except Exception:                                        # noqa: BLE001
        return dres
    return dres


def _judge_pair(check, dump_sets: dict, deliv_sets: list):
    """Run `check` on the side's own dumps and on its delivered files.
    Returns (dump_result or None, deliverable_result or None)."""
    dres = vres = None
    if len(dump_sets) >= 2:
        try:
            dres = check(dump_sets).as_dict()
        except Exception:                                    # noqa: BLE001
            dres = None
    if deliv_sets:
        try:
            vres = check({lvl: rows for lvl, rows in sorted(deliv_sets[0].items())}).as_dict()
        except Exception:                                    # noqa: BLE001
            vres = None
    return dres, vres


def _resid_text(res: dict) -> str:
    return ", ".join(f"{r:.3e}" for r in
                     [l.get("relative_weak_residual") for l in res.get("levels", [])]
                     if isinstance(r, float) and r == r)


def _momentum_findings(work: Path, side: str, op: dict, since=None) -> list[dict]:
    """The displacement of a thermo-elastic side against the momentum equation
    its own config states -- see pde_consistency.check_levels_thermoelastic.

    MEASURED on a delivered run: its configs stated NO body force beside the
    task's heat source, the displacement solved that other problem (400 times
    too small on both sides, self-converging at 1.95-1.97) while the
    temperature was right, and nothing here judged it.
    """
    out: list[dict] = []
    el = op.get("elastic")
    if not el or "box" not in op:
        return out
    fx, fy = el.get("fx"), el.get("fy")
    heat_stated = bool(op.get("source")) and not _source_is_zero(op["source"])
    no_body_force = (fx is None and fy is None) or (
        fx is not None and fy is not None and _source_is_zero(fx) and _source_is_zero(fy))
    if heat_stated and no_body_force:
        how = "absent from" if fx is None else "0 in"
        out.append({
            "sequence": f"mechanical source side {side}", "values": [], "priority": 40,
            "finding": (
                f"YOUR CONFIG STATES NO BODY FORCE FOR SIDE {side} (source_ux and source_uy {how} "
                f"./config.json) beside a heat source that is a polynomial. A manufactured "
                f"thermo-elastic problem states a body force for the momentum equation as it "
                f"states a heat source; with none the displacement solves a DIFFERENT problem "
                f"and converges cleanly to it, which no self-consistency check can see (the "
                f"momentum check judges against what you wrote). Confirm against the problem you were "
                f"given: if it states f_u, put both components in ./config.json as the task "
                f"writes them, on both sides.")})
    if fx is None or fy is None:
        # SAID, NOT SKIPPED. A side that states its material and no body force was
        # passed over with no note at all, and its audit read as a checked one.
        out.append({
            "sequence": f"momentum check side {side}", "values": [],
            "priority": 26, "informational": True,
            "finding": (
                f"SIDE {side}'S DISPLACEMENT WAS NOT CHECKED AGAINST ITS MOMENTUM EQUATION: its "
                f"./config.json states the material but not "
                + ("source_ux or source_uy" if fx is None and fy is None
                   else ("source_ux" if fx is None else "source_uy"))
                + ". State both body-force components as your task writes them (\"0\" when "
                f"there is none) and this audit judges the displacement against them.")})
        return out
    sets = _per_level_vector_sets(work, op["box"])
    dumps = _dump_level_sets(op["dir"], op["box"], "vector", since=since)
    if not sets and len(dumps) < 2:
        out.append({
            "sequence": f"momentum check side {side}", "values": [],
            "priority": 26, "informational": True,
            "finding": (
                f"SIDE {side}'S DISPLACEMENT WAS NOT CHECKED AGAINST ITS MOMENTUM EQUATION: "
                f"neither two or more of its own mesh dumps (field_level<k>.csv with ux, uy in "
                f"{op['dir'].name}/) nor a per-level file of yours holding x, y, T, ux, uy at "
                f"the cell MIDPOINTS of its box {op['box']} were found.")})
        return out
    op_txt = ("-div(sigma(u)) = f_u" + (f" - {el['beta']:g} grad T" if el["beta"] else "")
              + f" (lambda = {el['lam']:g}, mu = {el['mu']:g})")
    try:
        from .pde_consistency import check_levels_thermoelastic
        dres, vres = _judge_pair(
            lambda lv: check_levels_thermoelastic(lv, fx, fy, el["lam"], el["mu"], el["beta"],
                                                  op["box"]), dumps, sets)
    except Exception as exc:                              # noqa: BLE001
        out.append({"sequence": f"momentum check side {side}", "values": [],
                    "priority": 26, "informational": True,
                    "finding": (f"SIDE {side}'S MOMENTUM CHECK COULD NOT RUN: "
                                f"{type(exc).__name__}: {exc}")})
        return out
    # THE SOLVER'S OWN DUMPS DECIDE WHAT THE FIELD IS; the delivered files decide
    # only whether they carry it (see _verdict_finding). Where the dumps are newer and the
    # two disagree, both are said with their ages (see _timed_verdict).
    _dv = str((dres or {}).get("verdict")) if dres else None
    _vv = str((vres or {}).get("verdict")) if vres else None
    if (_dv in ("CONSISTENT", "INCONSISTENT") and _vv in ("CONSISTENT", "INCONSISTENT")
            and _dv != _vv and _dumps_newer_than_the_files(op["dir"], work)):
        out.append(_timed_verdict(side, "momentum", op_txt, dres, vres, op["dir"], work))
        return out
    if (dres and str(dres.get("verdict")) in ("CONSISTENT", "INCONSISTENT")
            and not (vres and _dumps_newer_than_the_files(op["dir"], work))):
        res, levels = dres, dumps
        on = f"its own mesh dumps ({op['dir'].name}/field_level<k>.csv)"
    elif vres:
        res, levels = vres, {lvl: rows for lvl, rows in sorted(sets[0].items())}
        on = "the fields you delivered"
    else:
        out.append(_verdict_finding(side, "momentum", op_txt, dres, vres, op["dir"].name))
        return out
    resid = [l.get("relative_weak_residual") for l in res.get("levels", [])]
    verdict = str(res.get("verdict"))
    shown = ", ".join(f"{r:.3e}" for r in resid if isinstance(r, float) and r == r)
    if verdict == "CONSISTENT":
        out.append({
            "sequence": f"momentum check side {side}", "values": resid,
            "priority": 26, "informational": True,
            "finding": (
                f"SIDE {side}'S DISPLACEMENT SATISFIES ITS OWN MOMENTUM EQUATION {op_txt} on "
                f"{on}: the weak residual falls {shown} across the levels. "
                f"Self-consistency, not a comparison with any reference: your displacement "
                f"solves the equation your config states, with the body force and the "
                f"temperature you gave it.")})
    elif verdict == "INCONSISTENT":
        # THE OPERATOR, MEASURED. The two wrong stiffness forms seen across the recorded
        # elastic runs each solve the right equation with a different material: mu
        # eps(u):eps(v) where 2 mu eps(u):eps(v) belongs is elasticity with mu/2, and
        # mu grad(u):grad(v) + lambda div u div v is elasticity with (lambda - mu, mu).
        # This verdict never named the form, which caused every such failure on record;
        # re-judging the same field with those two materials names it when it fits.
        form = ""
        for (lam2, mu2), what in (((el["lam"], 0.5 * el["mu"]),
                                   "mu eps(u):eps(v) where 2 mu eps(u):eps(v) belongs -- the "
                                   "factor 2 on mu is missing"),
                                  ((el["lam"] - el["mu"], el["mu"]),
                                   "mu grad(u):grad(v) + lambda div(u) div(v) -- a gradient "
                                   "inner product where the symmetric strain eps(u) = "
                                   "(grad u + grad u^T)/2 belongs")):
            try:
                alt = check_levels_thermoelastic(levels, fx, fy, lam2, mu2, el["beta"],
                                                 op["box"]).as_dict()
            except Exception:                              # noqa: BLE001
                continue
            if str(alt.get("verdict")) == "CONSISTENT":
                form = (f" MEASURED ON YOUR FIELD: it DOES satisfy the same equation with "
                        f"lambda = {lam2:g}, mu = {mu2:g}, which is what the stiffness form "
                        f"{what} solves. Check that form first.")
                break
        out.append({
            "sequence": f"momentum check side {side}", "values": resid,
            "priority": 5,
            "finding": (
                f"SIDE {side}'S DISPLACEMENT DOES NOT SATISFY THE MOMENTUM EQUATION ITS OWN "
                f"CONFIG STATES ({op_txt}), judged on {on}: the weak residual is {shown} across "
                f"the levels and does not fall at every refinement, which a field solving the "
                f"stated problem does. Refining will not fix this and your convergence study "
                f"cannot see it."
                + form
                + f" Check the stiffness form (2 mu eps(u):eps(v) + lambda div(u) div(v), with "
                f"eps the SYMMETRIC strain), the body force your solve applies against "
                f"source_ux, source_uy in your config, the thermal term if your side has one "
                f"(beta and the temperature it multiplies), lambda and mu and which subdomain "
                f"each belongs to, and whether the linear system was SOLVED at all.")})
    else:
        why = "; ".join(str(l.get("detail", ""))[:200] for l in res.get("levels", [])[:2])
        out.append({"sequence": f"momentum check side {side}", "values": resid,
                    "priority": 26, "informational": True,
                    "finding": (f"SIDE {side}'S MOMENTUM CHECK RETURNED {verdict}: {why}")})
    return out


def zero_channel_findings(work: Path) -> list[dict]:
    """A flux channel that is identically zero at every interface point while the field it belongs to varies.

    MEASURED on a delivered thermo-elastic run: side A's exported heat flux qn was 0.0 at all 31
    interface nodes at every level (its SCATRA FLUX CALC line condition did not sit on the interface
    line) while its tractions were right; the run's temperature converged cleanly to a function 13 %
    off on both sides, its displacement was right to 1e-7, the interface trace agreed, and the
    "transmitted nothing" check stayed silent because the traction channels were not zero. The agent
    saw the zeros, called them a limitation of the served recovery, and handed in. One dead channel is
    one dead exchange; it is named per side and per channel from the agent's own interface files.
    """
    out: list[dict] = []
    import csv as _csv2
    seen = {}
    for q in sorted(work.rglob("interface_level*.csv")):
        m = _LEVEL_FILE.match(q.name)
        if not m or not (m.group("side") or ""):
            continue
        try:
            with q.open() as fh:
                rd = _csv2.reader(fh)
                header = next(rd, None)
                if not header:
                    continue
                cols = [c.strip().strip('"').lower() for c in header]
                rows = []
                for row in rd:
                    try:
                        rows.append([float(x) for x in row])
                    except ValueError:
                        continue
        except OSError:
            continue
        if len(rows) < 3:
            continue
        flux_cols = [i for i, c in enumerate(cols) if c in ("qn", "q", "flux", "tx", "ty", "tz", "qx", "qy")]
        field_cols = [i for i, c in enumerate(cols) if c in ("t", "u", "ux", "uy", "uz", "p", "phi", "value")]
        if not flux_cols or not field_cols:
            continue
        import numpy as _np
        arr = _np.asarray(rows, dtype=float)
        varies = any(float(_np.ptp(arr[:, i])) > 0 for i in field_cols if i < arr.shape[1])
        if not varies:
            continue
        for i in flux_cols:
            if i < arr.shape[1] and float(_np.max(_np.abs(arr[:, i]))) == 0.0:
                key = (m.group("side"), cols[i])
                seen.setdefault(key, []).append(int(m.group("k")))
    for (side, ch), levels in sorted(seen.items()):
        out.append({"sequence": f"zero channel {ch} side {side}", "values": [], "priority": 12,
                    "finding": (
            f"CHANNEL {ch} IS IDENTICALLY ZERO ON SIDE {side} at level(s) {sorted(levels)} -- every "
            f"interface point, while the field beside it varies. Nothing was exchanged on that channel: "
            f"the partner solved with a zero {ch}, and a coupling that converges around a zero channel "
            f"converges to the wrong function on BOTH sides while its interface trace agrees. This is "
            f"not a limitation of the recovery: a boundary-flux output that is zero everywhere means the "
            f"flux condition does not sit on the interface (the flux-calc line's E id names a line other "
            f"than the interface, or its topology lists other nodes) or the exported column was never "
            f"filled. Fix the exchange before any level is refined.")})
    return out


def _iface_xy(p: Path):
    """The (x, y) rows of an interface dump, or None."""
    try:
        import numpy as _np
        h, r = _read_named_csv(p)
        if not ({"x", "y"} <= set(h)) or not r:
            return None
        return _np.asarray(r, float)[:, [h.index("x"), h.index("y")]]
    except Exception:                                        # noqa: BLE001
        return None


def _free_ends_on_bent(xy, u, own, partners) -> list:
    """[(end point, value there, held value, gap / peak)] of a side's field at the two
    ends of a BENT interface: for each end the side's mesh has a node at, an outer edge
    through it -- a line along x or y with every node of the side on one side of it --
    whose other nodes (off the interface) hold one value exactly, and an end node that
    departs from that value. The ends are the two ends of the curve the side's own
    interface points and its partners' trace together."""
    import numpy as _np
    pts = [own] + [p for p in partners if p is not None and len(p)]
    U = _np.unique(_np.vstack(pts), axis=0)
    span = float(_np.ptp(_np.vstack([xy, U]), axis=0).max()) or 1.0
    tol = 1e-9 * span
    scale = float(_np.abs(u).max())
    if scale <= 0.0:
        return []
    cur, _why = _interface_curve(U)
    if cur is None:
        return []
    ends = [cur["uv"][cur["order"][0]], cur["uv"][cur["order"][-1]]]
    near = _qc()._nearest_rows(xy, U)[0] <= tol              # this side's interface nodes
    out = []
    for E in ends:
        at = _np.where(_np.abs(xy - E).max(axis=1) <= tol)[0]
        if len(at) != 1:
            continue
        e, best = int(at[0]), None
        for a in (0, 1):
            dn = xy[:, a] - E[a]
            if not (_np.all(dn >= -tol) or _np.all(dn <= tol)):
                continue                                     # a line through the mesh, not its edge
            rest = [i for i in _np.where(_np.abs(dn) <= tol)[0] if i != e and not near[i]]
            if len(rest) < 3:
                continue
            c = float(_np.median(u[rest]))
            if _np.abs(u[rest] - c).max() > 1e-12 * scale:
                continue                                     # the edge is not held at one value
            gap = abs(float(u[e]) - c)
            if gap > 1e-6 * scale and (best is None or len(rest) > best[4]):
                best = (tuple(float(v) for v in xy[e]), float(u[e]), c, gap / scale, len(rest))
        if best is not None:
            out.append(best[:4])
    return out


def free_interface_end_findings(work: Path, dirs=None, levels=None) -> list[dict]:
    """An edge held at one value everywhere but at the node where it meets the interface.

    THE SHAPE OF A LOOP THAT SKIPS THE INTERFACE COLUMN. A side whose held edges are
    written by `for i in range(1, NX + 1)` leaves the node the interface ends on free;
    measured on coupled runs (hand-written sides, which carry no served check), those two
    nodes held 10-42 % of the side's peak while every other node of the edge held the
    outer value exactly, and nothing named it -- the audit's outer check exempts the
    interface's end points because on a Dirichlet side they carry the partner's value.
    Read from the side's own field dump alone: no config key is needed, and a side whose
    end node holds exactly what it imported there is named as carrying its partner's
    value, not as the one that left it free.

    A BENT INTERFACE HAS ITS ENDS WHERE THE CURVE ENDS, and needs no flag. Read along one
    axis and gated on a flag the heat contracts do not carry, this could not run on a
    two-leg interface: measured, four free interface ends of one coupled run (27 % -> 12 %
    -> 5 % and 76 % -> 37 % -> 17 % of each side's peak, falling at first order) went
    unnamed at every level, and a level read "SAVE IT NOW". The ends are the two ends of
    the curve the two sides' own interface dumps trace together; an outer edge through an
    end is a line along x or y that the side's mesh lies wholly on one side of; it is
    held when every one of its other nodes carries one value exactly. Only a side that
    states the edges are not held (full_outer_dirichlet false) is left alone."""
    out: list[dict] = []
    try:
        import numpy as _np
    except Exception:                                        # noqa: BLE001
        return out
    seen, folders = set(), []
    for d in list(dirs or []) + _side_dirs(work):
        try:
            key = Path(d).resolve()
        except OSError:
            continue
        if key not in seen and Path(d).is_dir():
            seen.add(key)
            folders.append(Path(d))
    # ONLY WHERE THE RUN SAYS THE EDGES ARE HELD. Whether the edges the interface ends on
    # carry the outer value is the problem's, and the run states it as full_outer_dirichlet
    # in a side's config.json; one side saying true covers the pair (it is one outer
    # boundary), any side saying false, or none saying, leaves this silent.
    # A SIDE THAT SAYS IT ONLY IN ITS SCRIPT SAYS IT TOO: the served contracts' knob is
    # FULL_OUTER_DIRICHLET in the file. Measured: a hand-written side set it True in code
    # and freed both interface corners, and this stayed silent for want of a config key;
    # over the record the same reading names free corners in runs whose fields were
    # otherwise right (38-41 % of the side's peak at level 1, halving per level) -- a
    # boundary condition the problem states, broken at two nodes, not a rounding matter.
    said = []
    for d in folders:
        try:
            cfg = json.loads((d / "config.json").read_text() or "{}")
        except (OSError, ValueError):
            cfg = {}
        if isinstance(cfg, dict) and isinstance(cfg.get("full_outer_dirichlet"), bool):
            said.append(cfg["full_outer_dirichlet"])
            continue
        for q in sorted(q for q in d.glob("participant*.py") if not _is_backup(q)):
            if ".replaced-" in q.name:
                continue
            try:
                m = re.search(r"^FULL_OUTER_DIRICHLET\s*=\s*(True|False)\b",
                              q.read_text(errors="ignore"), re.M)
            except OSError:
                m = None
            if m:
                said.append(m.group(1) == "True")
                break
    straight_ok, bent_ok = bool(said) and all(said), all(said)
    if not bent_ok:
        return out
    for d in folders:
        hits = []
        for p in sorted(d.glob("field_level*.csv")):
            m = re.fullmatch(r"field_level(\d+)\.csv", p.name)
            q = d / p.name.replace("field_", "interface_")
            if not (m and q.is_file()):
                continue
            lvl = int(m.group(1))
            if levels is not None and lvl not in levels:
                continue
            try:
                hf, rf = _read_named_csv(p)
                hi, ri = _read_named_csv(q)
            except OSError:
                continue
            if not ({"x", "y"} <= set(hf) and {"x", "y"} <= set(hi)) or len(rf) < 16 or len(ri) < 3:
                continue
            vc = next((hf.index(c) for c in hf if c not in ("x", "y", "z")), None)
            if vc is None:
                continue
            F = _np.asarray(rf, float)
            I = _np.asarray(ri, float)
            xy, u = F[:, [hf.index("x"), hf.index("y")]], F[:, vc]
            if not (_np.isfinite(xy).all() and _np.isfinite(u).all()):
                continue
            _legs, _bad = _curve_legs(I[:, [hi.index("x"), hi.index("y")]])
            if _legs is not None or _bad is not None:
                if _legs is not None:
                    hits += [(lvl,) + h for h in _free_ends_on_bent(
                        xy, u, I[:, [hi.index("x"), hi.index("y")]],
                        [_iface_xy(o / q.name) for o in folders if o != d])]
                continue
            if not straight_ok:
                continue
            iax = int(_np.argmin(_np.ptp(I[:, [hi.index("x"), hi.index("y")]], axis=0)))
            along = 1 - iax
            span = float(_np.ptp(xy, axis=0).max()) or 1.0
            tol = 1e-9 * span
            plane = float(_np.median(I[:, [hi.index("x"), hi.index("y")][iax]]))
            scale = float(_np.abs(u).max()) or 0.0
            if scale <= 0.0:
                continue
            for edge in (float(xy[:, along].min()), float(xy[:, along].max())):
                on = _np.where(_np.abs(xy[:, along] - edge) <= tol)[0]
                if len(on) < 4:
                    continue
                end = [i for i in on if abs(xy[i, iax] - plane) <= tol]
                rest = [i for i in on if i not in end]
                if len(end) != 1 or len(rest) < 3:
                    continue
                c = float(_np.median(u[rest]))
                if _np.abs(u[rest] - c).max() > 1e-12 * scale:
                    continue                                  # the edge is not held at one value
                gap = abs(float(u[end[0]]) - c)
                if gap > 1e-6 * scale:
                    hits.append((lvl, tuple(float(v) for v in xy[end[0]]), float(u[end[0]]), c, gap / scale))
        if not hits:
            continue
        # does this side hold, at those nodes, exactly what it imported there?
        imposed = False
        try:
            imp = json.loads((d / "imports.json").read_text() or "{}")
            blk = next(iter(imp.values()), {}) if isinstance(imp, dict) else {}
            ic = _np.asarray(blk.get("coordinates") or [], float)
            iv = _np.asarray(blk.get("values") or [], float).ravel()
            if ic.ndim == 2 and len(ic) == len(iv) and len(iv):
                lvl_last = max(h[0] for h in hits)
                last = [h for h in hits if h[0] == lvl_last]
                imposed = all(_np.abs(_np.hypot(ic[:, 0] - h[1][0], ic[:, 1] - h[1][1])).min() <= 1e-9 * max(span, 1.0)
                              and abs(iv[int(_np.argmin(_np.hypot(ic[:, 0] - h[1][0], ic[:, 1] - h[1][1])))] - h[2])
                              <= 1e-9 * max(abs(h[2]), scale) for h in last)
        except Exception:                                    # noqa: BLE001
            imposed = False
        try:
            name = str(d.relative_to(work))
        except ValueError:
            name = d.name
        lv = sorted({h[0] for h in hits})
        worst = max(hits, key=lambda h: h[4])
        share = " / ".join(f"{max(h[4] for h in hits if h[0] == l):.0%}" for l in lv)
        out.append({
            "sequence": f"free interface end {name}", "values": [h[4] for h in hits],
            "priority": 7 if not imposed else 27,
            "finding": (
                f"{name.upper()}'S FIELD DEPARTS AT THE NODE WHERE THE INTERFACE MEETS AN EDGE IT "
                f"HOLDS: every other node of that edge carries {worst[3]:.6g} exactly, and the "
                f"interface end node ({worst[1][0]:g}, {worst[1][1]:g}) carries {worst[2]:.3g} -- "
                f"{share} of the side's peak at level(s) {', '.join(str(l) for l in lv)}. "
                + ("This side holds there exactly the value it imported: it carries its partner's "
                   "end node, and the partner is where to look -- a partner whose held edges skip "
                   "the interface column leaves those nodes free."
                   if imposed else
                   "An edge held at one value is held end to end: a loop over the edge's nodes "
                   "that skips the interface column leaves exactly this node free. The error "
                   "shrinks with the mesh, but the boundary condition at that node is not the "
                   "problem's.")) })
    return out


def _same_dir(a, b) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return False


def outer_boundary_findings(work: Path, levels=None, dirs=None) -> list[dict]:
    """Does each side's field honour the outer condition its own config states?

    THE EQUATION CHECK CANNOT SEE A BOUNDARY CONDITION, BY CONSTRUCTION. Its
    test function is chosen so that value AND normal slope vanish on every face
    -- deliberately, because a coupled side's boundary carries the partner's
    data and any other choice refused 92 of 94 coupled sides. Both boundary
    terms therefore vanish "for any u at all", so a field solving the SAME
    interior equation with DIFFERENT boundary data passes it cleanly.

    MEASURED on one coupled run. Its Kratos side left the top and
    bottom outer edges natural instead of held at the prescribed value: its
    outer-boundary maximum is 1.03e-02 against 0.000e+00 in all four correct
    cells of the same round, essentially its own interior maximum, so the field
    was unanchored. It converged cleanly, its interface agreed, its divergence
    check passed at 1.5e-15 relative, and its convergence order came out 0.004.
    Nothing in this repository could see it.

    What CAN see it is the side's own declaration. The contract already has the
    caller transcribe `source_expr` into config.json and judges the field
    against it; `outer` is the same transcription for the value prescribed on
    the part of this subdomain's boundary that is NOT the interface. Comparing
    a field against a number the caller wrote down is not serving an answer --
    it is the same act as the equation check, one boundary out.

    ABSTAINS WITHOUT A DECLARATION. A side whose config states no `outer` gets
    nothing from this: a natural condition is a legitimate choice, and a check
    that guessed which edges were meant to be held would invent a defect.
    """
    out: list[dict] = []
    _EDGE = {"left": (0, 0), "right": (0, 1), "bottom": (1, 0), "top": (1, 1)}
    _all_ops = _side_operators(work)
    for side, op in sorted(_all_ops.items()):
        if op.get("dsmc"):
            continue                      # a DSMC side has no field and no held edges to judge
        if op.get("fluid"):
            out.append({"sequence": f"outer boundary {side}", "values": [], "priority": 26,
                        "informational": True, "finding": _not_judged_as_fluid(
                            f"SIDE {side}'S OUTER BOUNDARY WAS NOT CHECKED", op["fluid"],
                            "a field against the one value its non-interface edges hold")})
            continue
        if op.get("three_d"):
            out.append({"sequence": f"outer boundary {side}", "values": [], "priority": 26,
                        "informational": True, "finding": _not_judged_in_3d(side, "OUTER BOUNDARY")})
            continue
        if "outer" not in op or "box" not in op:
            # A SIDE THIS CHECK SKIPS IS SAID TO BE SKIPPED. Measured: two correct cells read "no
            # findings" while one side's outer boundary had never been judged (its config
            # stated no outer value).
            _miss = [w for w, have in (("\"outer\" (the value its non-interface edges hold)", "outer" in op),
                                       ("its box (x0, x1, y0, y1)", "box" in op)) if not have]
            out.append({"sequence": f"outer boundary {side}", "values": [], "priority": 26,
                        "informational": True, "finding": (
                f"SIDE {side}'S OUTER BOUNDARY WAS NOT CHECKED: its config.json states no "
                + " and no ".join(_miss) + ". A side that holds its outer edges at a value can state "
                "it there and this check compares its field with it; a side whose outer edges are "
                "natural has nothing here to check.")})
            continue
        # A SIDE THAT STATES NO iface HAS ITS INTERFACE WHERE ITS BOX MEETS ITS PARTNER'S.
        # Measured: three of five sides of one round stated outer and no iface and went
        # unjudged, while the partner's box named the shared edge exactly.
        if not op.get("iface"):
            (x0, x1), (y0, y1) = op["box"]
            _tol = 1e-9 * max(x1 - x0, y1 - y0)
            _shared = []
            for _other, _oo in _all_ops.items():
                if _other == side or "box" not in _oo:
                    continue
                (a0, a1), (b0, b1) = _oo["box"]
                for _name, _hit in (("right", abs(x1 - a0) < _tol), ("left", abs(x0 - a1) < _tol),
                                    ("top", abs(y1 - b0) < _tol), ("bottom", abs(y0 - b1) < _tol)):
                    if _hit:
                        _shared.append(_name)
            if len(set(_shared)) == 1:
                op = dict(op, iface=_shared[0])
        # WITHOUT `iface` THERE IS NO WAY TO KNOW WHICH EDGE TO LEAVE ALONE,
        # AND JUDGING THEM ALL ACCUSES THE INTERFACE. Measured on one coupled run:
        # a side that declared outer = 0 and no iface was told its level-1
        # field "departs from it by 100% of its own peak, worst at (0.625,
        # 0.5)" -- the interface edge, carrying the partner's trace exactly as
        # it should. That run was correct; the accusation was ours, and it
        # fired because the Kratos contract asked for `outer` without asking
        # for `iface`. A check that cannot tell the held edges from the
        # exchanged one has nothing to say, and says so.
        # AN INTERFACE STATED AS A COORDINATE NAMES ITS EDGE TOO. The NGSolve
        # contract takes iface as the interface line's coordinate (with
        # iface_axis), and two runs that wrote it so were told "not which edge".
        _ifc = op.get("iface")
        if _ifc and _ifc not in _EDGE:
            try:
                _c = float(_ifc)
                (_x0, _x1), (_y0, _y1) = op["box"]
                _tol = 1e-9 * max(_x1 - _x0, _y1 - _y0)
                _cands = {"left": ("x", _x0), "right": ("x", _x1),
                          "bottom": ("y", _y0), "top": ("y", _y1)}
                _hit = [e for e, (ax, v) in _cands.items()
                        if abs(_c - v) < _tol and op.get("iface_axis", "") in ("", ax)]
                if len(_hit) == 1:
                    op = dict(op, iface=_hit[0])
            except (TypeError, ValueError):
                pass
        if not op.get("iface") or op.get("iface") not in _EDGE:
            # A BENT INTERFACE IS NO ONE EDGE, so `iface` is not asked of it (measured: three
            # sides were told to add `iface` of a two-leg interface).
            if _iface_bends(op["dir"]):
                out.append({"sequence": f"outer boundary {side}", "values": [],
                            "priority": 26, "informational": True, "finding": (
                    f"SIDE {side}'S OUTER BOUNDARY WAS NOT CHECKED: its interface bends (its "
                    f"interface dump runs along two or more straight legs), and this check reads "
                    f"a box whose interface is one of its four edges, so it does not judge this "
                    f"side; no key in its config.json changes that. The ends of a bent interface "
                    f"are judged by the free-end check, from the field dumps.")})
                continue
            out.append({"sequence": f"outer boundary {side}", "values": [],
                        "priority": 26, "informational": True, "finding": (
                f"SIDE {side}'S OUTER BOUNDARY WAS NOT CHECKED: its config.json "
                f"states outer = {op['outer']:g} but not which edge is the "
                f"interface (`iface`: left, right, bottom or top, or the interface "
                f"line's coordinate on one of the box's edges). Without that "
                f"there is no telling the held edges from the exchanged one, so "
                f"nothing here is judged. Add `iface` beside `outer` and this "
                f"audit checks the field on the other three edges.")})
            continue
        # THE SIDE'S OWN NODAL DUMP, NOT THE PROBE FILE.
        #
        # The equation check reads per-level files on the cell MIDPOINTS,
        # because its quadrature is a midpoint rule -- and a midpoint grid
        # contains no boundary point at all, by construction. A boundary
        # question therefore has to be asked of the participant's own mesh
        # dump, which sits on the nodes and includes the faces: the very file
        # `_per_level_field_sets` refuses, for its own good reason. The two
        # checks want different files, and a real run writes both.
        if dirs is not None and not any(_same_dir(op["dir"], d) for d in dirs):
            continue                      # a folder beside the participants is not one of them
        chosen: dict = {}
        for q, kind, lvl, _sd in _level_files(op["dir"]):
            if levels is not None and lvl not in levels:
                continue                  # the caller asks about these levels only
            if _csv_role(q, kind) in ("raw", "field"):
                rows = []
                for line in q.read_text(errors="replace").splitlines()[1:]:
                    parts = line.split(",")
                    if len(parts) < 3:
                        continue
                    try:
                        rows.append((float(parts[0]), float(parts[1]),
                                     float(parts[2])))
                    except ValueError:
                        continue
                if rows:
                    chosen.setdefault(lvl, rows)
        if not chosen:
            continue
        (x0, x1), (y0, y1) = op["box"]
        span = max(x1 - x0, y1 - y0) or 1.0
        skip = _EDGE.get(op.get("iface", ""))
        # THE EDGES THE SIDE SAYS ARE NATURAL ARE NOT JUDGED. Measured on a coupled round: a
        # right side whose two edges at the interface's ends are insulated was told its field
        # "DOES NOT HOLD THE OUTER VALUE ... worst at" a point on one of them. Stated False
        # (full_outer_dirichlet / FULL_OUTER_DIRICHLET), only the edge opposite the interface
        # is judged; stated True, or not stated, every non-interface edge is.
        _held = _all_outer_held(op["dir"])
        only = (skip[0], 1 - skip[1]) if (_held is False and skip is not None) else None
        _how = ("" if _held is not None else
                " If your problem leaves the two edges the interface ends on natural (zero flux), "
                "say so in the participant, FULL_OUTER_DIRICHLET = False (or \"full_outer_dirichlet\": "
                "false in config.json), and this check judges only the edge opposite the interface.")
        worst, where, lvl_seen = 0.0, "", None
        by_lvl: dict = {}                 # level -> this level's worst departure
        for lvl in sorted(chosen):
            rows = chosen[lvl]
            scale = max((abs(r[2]) for r in rows), default=0.0)
            if scale <= 0:
                continue
            by_lvl[lvl] = 0.0
            for px, py, v in rows:
                # A CORNER BELONGS TO THE INTERFACE, NOT TO THE OUTER EDGE.
                # The two edges meet there, the interface carries the
                # partner's trace, and judging that point against the outer
                # value accuses every correct coupled side of exactly the
                # defect this check is for. Same end-node shape as the
                # quadrature rule on the interface itself.
                on_iface = False
                if skip is not None:
                    _ax, _end = skip
                    _c = px if _ax == 0 else py
                    _b = (x0, x1)[_end] if _ax == 0 else (y0, y1)[_end]
                    on_iface = abs(_c - _b) <= 1e-9 * span
                if on_iface:
                    continue
                for axis, (lo, hi) in ((0, (x0, x1)), (1, (y0, y1))):
                    c = px if axis == 0 else py
                    for end, bound in ((0, lo), (1, hi)):
                        if abs(c - bound) > 1e-9 * span:
                            continue
                        if only is not None and (axis, end) != only:
                            continue                  # an edge the side says is natural
                        dev = abs(v - op["outer"]) / scale
                        by_lvl[lvl] = max(by_lvl[lvl], dev)
                        if dev > worst:
                            worst, where, lvl_seen = dev, f"({px:.4g}, {py:.4g})", lvl
        # A SMALL DEPARTURE THAT DOES NOT SHRINK IS NOT DISCRETISATION. A held node
        # holds its value to round-off, and a weakly imposed one comes closer as the
        # mesh refines. Measured on a physically wrong run whose order read right: the
        # partner's trace sat on an outer edge at 0.46 % of the side's peak at every
        # level (its interface dofs were other nodes'), under the 2 % bar below.
        _ds = [by_lvl[k] for k in sorted(by_lvl)]
        flat = (len(_ds) >= 2 and all(d > 1e-6 for d in _ds)
                and all(b > 0.67 * a for a, b in zip(_ds, _ds[1:])))
        if flat and worst <= 0.02:
            out.append({"sequence": f"outer boundary {side}", "priority": 6,
                        "values": _ds, "finding": (
                f"SIDE {side}'S FIELD DOES NOT HOLD THE OUTER VALUE ITS OWN CONFIG STATES, and "
                f"the departure does not shrink with the mesh: config.json says outer = "
                f"{op['outer']:g} off the interface"
                + (f" (the interface is the {op['iface']} edge)" if op.get("iface") else "")
                + f", and the field departs from it by "
                + " / ".join(f"{d:.2%}" for d in _ds)
                + f" of its own peak at levels {', '.join(str(k) for k in sorted(by_lvl))}, worst "
                f"at {where}. A held node carries its value to round-off, and a weakly imposed "
                f"one comes closer as the mesh refines; a departure that stays is a value this "
                f"side put on its boundary. It solves the same interior equation, so the "
                f"equation check cannot see it. Check which nodes your participant writes "
                f"values into." + _how)})
        if worst > 0.02:
            out.append({"sequence": f"outer boundary {side}", "priority": 6,
                        "values": [worst], "finding": (
                f"SIDE {side}'S FIELD DOES NOT HOLD THE OUTER VALUE ITS OWN "
                f"CONFIG STATES. config.json says outer = {op['outer']:g} on "
                f"the part of this subdomain's boundary that is not the "
                f"interface"
                + (f" (the interface is the {op['iface']} edge)"
                   if op.get("iface") else "")
                + f", and your level-{lvl_seen} field departs from it by "
                f"{worst:.0%} of its own peak, worst at {where}. A field that "
                f"is not held where it should be held solves a DIFFERENT "
                f"problem: it satisfies the same interior equation -- the "
                f"equation check cannot see this, its test function is built "
                f"so every boundary term vanishes -- and converges cleanly to "
                f"the wrong function, which reads afterwards as a refinement "
                f"study that did not converge. Check which edges your "
                f"participant fixes against the ones it declares here." + _how)})
    return out


def _dumps_newer_than_the_files(side_dir: Path, work: Path) -> bool:
    """A side's dump rewritten after the per-level files were delivered is not what
    they show (measured: a level-1 dump overwritten by a later test run on another
    mesh, beside delivered files that held the right answer); the delivered files
    decide then."""
    try:
        dumps = [q.stat().st_mtime for q in Path(side_dir).glob("field_level*.csv")]
        files = [q.stat().st_mtime for q in Path(work).glob("solution_level*.csv")]
    except OSError:
        return False
    return bool(dumps and files) and max(dumps) > max(files) + 1.0


def _verdict_finding(side: str, what: str, op_txt: str, dres, vres, folder: str) -> dict:
    """One finding from the verdict on the side's own dumps (`dres`) and on its
    delivered files (`vres`). The dumps are the solver's field and decide what
    the FIELD is; the delivered files decide only whether they carry it."""
    seq = f"{what} check side {side}"
    dv = str((dres or {}).get("verdict")) if dres else None
    vv = str((vres or {}).get("verdict")) if vres else None
    judged = [r for r in (dres, vres) if r and str(r.get("verdict")) in ("CONSISTENT", "INCONSISTENT")]
    vals = [l.get("relative_weak_residual") for l in (judged[0] if judged else (dres or vres or {})).get("levels", [])]
    eq = "EQUATION" if what == "equation" else "MOMENTUM EQUATION"
    src_d = f"its own mesh dumps ({folder}/field_level<k>.csv)"
    if dv == "INCONSISTENT" or (dv not in ("CONSISTENT", "INCONSISTENT") and vv == "INCONSISTENT"):
        on = src_d if dv == "INCONSISTENT" else "the per-level files you delivered"
        r = dres if dv == "INCONSISTENT" else vres
        return {"sequence": seq, "values": vals, "priority": 5, "finding": (
            f"SIDE {side}'S FIELD DOES NOT SATISFY THE {eq} ITS OWN CONFIG STATES ({op_txt}), "
            f"judged on {on}: {r.get('explanation', '')} Check the source term you "
            f"implemented against the one your task states, the coefficient, which "
            f"subdomain each belongs to, and whether the linear system was SOLVED at all.")}
    if dv == "CONSISTENT" and vv == "INCONSISTENT":
        return {"sequence": seq, "values": vals, "priority": 6, "finding": (
            f"SIDE {side}'S SOLVER FIELD SATISFIES ITS {eq} ({op_txt}; {src_d}: "
            f"{_resid_text(dres)}), BUT THE PER-LEVEL FILES YOU DELIVERED DO NOT "
            f"({_resid_text(vres)}): the check passes on the field and fails on your files, "
            f"so the files do not carry that field faithfully. Rebuild them from the dumps "
            f"(scipy.interpolate.griddata(..., method='linear') at the prescribed points) "
            f"rather than changing the solver.")}
    if dv == "CONSISTENT" or (dv is None and vv == "CONSISTENT"):
        on = src_d + (" and on the per-level files you delivered" if vv == "CONSISTENT" else "") \
            if dv == "CONSISTENT" else "the per-level files you delivered"
        r = dres if dv == "CONSISTENT" else vres
        return {"sequence": seq, "values": vals, "priority": 26, "informational": True, "finding": (
            f"SIDE {side} SATISFIES ITS OWN {eq} {op_txt}, judged on {on}: "
            f"{r.get('explanation', '')} This is self-consistency, not a comparison with "
            f"any reference: the field solves the equation your config states, with the "
            f"source you gave it.")}
    unsettled = next((r for r in (dres, vres) if r and str(r.get("verdict")) == "UNSETTLED"), None)
    if unsettled:
        on = src_d if unsettled is dres else "the per-level files you delivered"
        return {"sequence": seq, "values": vals, "priority": 7, "finding": (
            f"SIDE {side}'S {eq} CHECK CANNOT SETTLE ({op_txt}), judged on {on}: "
            f"{unsettled.get('explanation', '')}")}
    why = "; ".join(str(x.get("explanation") or "")[:300] for x in (dres, vres) if x)
    return {"sequence": seq, "values": vals, "priority": 26, "informational": True, "finding": (
        f"SIDE {side}'S {eq} CHECK COULD NOT JUDGE IT (NOT CHECKED): {why}")}


def _timed_verdict(side: str, what: str, op_txt: str, dres, vres, side_dir: Path, work: Path) -> dict:
    """The verdict from the side's dumps and its delivered files, saying which is newer.

    DUMPS NEWER THAN THE FILES ARE NOT ALWAYS THE ONES TO BELIEVE, NOR ALWAYS THE ONES TO
    DROP. Measured both ways: a level-1 dump overwritten by a later test run on another
    mesh, beside delivered files that held the right answer; and delivered files left
    from before a re-run, which read "DOES NOT SATISFY" of a side whose current field
    satisfies its equation to round-off -- the run gave up on it. So both are judged and,
    where they disagree, both are said with their ages."""
    dv = str((dres or {}).get("verdict")) if dres else None
    vv = str((vres or {}).get("verdict")) if vres else None
    newer = _dumps_newer_than_the_files(side_dir, work)
    if dres and dv in ("CONSISTENT", "INCONSISTENT", "UNSETTLED") and not newer:
        return _verdict_finding(side, what, op_txt, dres, None, side_dir.name)
    if not (newer and dv in ("CONSISTENT", "INCONSISTENT") and vv in ("CONSISTENT", "INCONSISTENT")
            and dv != vv):
        return _verdict_finding(side, what, op_txt, None if vres else dres, vres, side_dir.name)
    if dv == "CONSISTENT" and vv == "INCONSISTENT":
        f = _verdict_finding(side, what, op_txt, dres, vres, side_dir.name)
        f["finding"] += (" The delivered files are OLDER than this side's latest dumps: they are "
                         "the last run's, not this one's.")
        return f
    eq = "EQUATION" if what == "equation" else "MOMENTUM EQUATION"
    say = {"CONSISTENT": "satisfy it", "INCONSISTENT": "do not satisfy it"}
    return {"sequence": f"{what} check side {side}", "values": [], "priority": 8, "finding": (
        f"SIDE {side}'S DELIVERED FILES AND ITS NEWER DUMPS DISAGREE ON ITS {eq} ({op_txt}): the "
        f"delivered files {say[vv]} ({_resid_text(vres)}); its own mesh dumps, written after them, "
        f"{say[dv]} ({_resid_text(dres)}). The two come from different runs. Write the delivered "
        f"files and the dumps from the same run, then check again.")}


def _transient_marks(side_dir: Path) -> list:
    """What says a side is time-dependent: time-step keys in its config.json, or the window
    constants (T_END, N_STEPS) the served transient contracts set in their scripts. The
    equation check models the steady operator only, and a transient side's final-time field
    judged against it would read as a wrong field."""
    marks = []
    try:
        cfg = json.loads((Path(side_dir) / "config.json").read_text() or "{}")
    except Exception:                                        # noqa: BLE001
        cfg = {}
    if isinstance(cfg, dict):
        # "dt" and "nt" only in lower case: "dT" is a temperature step on a steady side
        keys = [str(q) for q in cfg if q in ("dt", "nt") or str(q).lower() in (
            "t_end", "t_final", "n_steps", "nsteps", "num_steps", "time_steps")]
        if keys:
            marks.append("its config.json states " + ", ".join(keys))
    for q in sorted(q for q in Path(side_dir).glob("participant*.py") if not _is_backup(q)):
        if ".replaced-" in q.name:
            continue
        try:
            text = q.read_text(errors="replace")
        except OSError:
            continue
        names = [nm for nm in ("T_END", "N_STEPS") if re.search(rf"^{nm}\s*=", text, re.M)]
        if names:
            marks.append(f"{q.name} sets {' and '.join(names)}")
            break
    return marks


def _transient_note(side: str, marks: list) -> dict:
    return {"sequence": f"equation check side {side}", "values": [],
            "priority": 26, "informational": True, "finding": (
                f"SIDE {side}'S FIELD WAS NOT CHECKED AGAINST ITS OWN EQUATION: it is "
                f"time-dependent ({'; '.join(marks)}), and this check judges the steady equation "
                f"-div(k grad u) + c u = f only, so it does not judge this side.")}


def _k_not_one_number(side_dir: Path):
    """How a side's config.json states k when the check cannot judge it (a list that is no 2x2
    matrix, an object, a text), or None. The equation check judges one k over one box -- one
    number, or one 2x2 matrix, nested or four numbers row by row -- and says so rather than
    asking such a side for "a numeric k" it cannot give. A 2x2 matrix is judged, so it is never
    the reason: measured, a side whose config stated a 2x2 k and no source_expr was told its k
    was the reason it went unchecked."""
    try:
        k = json.loads((Path(side_dir) / "config.json").read_text() or "{}").get("k")
    except Exception:                                        # noqa: BLE001
        return None
    if k is None or _k_as_matrix(k) is not None:
        return None
    try:
        float(k)
        return None
    except (TypeError, ValueError):
        pass
    if isinstance(k, list):
        if k and all(isinstance(r, (list, tuple)) for r in k):
            return f"a {len(k)}x{len(k[0])} matrix (a tensor conductivity)"
        return f"a list of {len(k)} values"
    if isinstance(k, dict):
        return "an object with keys " + ", ".join(str(q) for q in list(k)[:4]) + (" ..." if len(k) > 4 else "")
    return f"the text {str(k)[:40]!r}"


def _f_src_in_script(side_dir: Path) -> bool:
    """True when a participant script of the side defines F_SRC, the source function the served
    wrappers carry beside the config keys."""
    for q in sorted(Path(side_dir).glob("participant*.py")):
        if _is_backup(q):
            continue
        try:
            if re.search(r"^def F_SRC\s*\(", q.read_text(errors="ignore"), re.M):
                return True
        except OSError:
            continue
    return False


def _box_fill(side_dir: Path, box: list, field=None, since=None) -> dict:
    """{level: measurement} for each current mesh dump of a side that leaves more than 2 % of
    the box its config.json states without mesh; levels that fill the box are left out.

    MEASURED ON THE GRID THE EQUATION CHECK READS (the midpoints _dump_level_sets builds):
    `outside` is the share of those points outside the dump's nodes (their convex hull, where
    linear interpolation has nothing), `apart` the share inside that outline but farther from
    every node than one cell -- a notch or removed cells, which linear interpolation bridges
    with values the side never computed. One cell is the larger of the mesh's typical cell
    (the median longest edge of the nodes' Delaunay triangles; in a filled mesh of square,
    stretched or equilateral cells no point is farther than about 0.6 of it from a node) and
    the nearest node's own size (the median length of the Delaunay edges at it), which keeps
    the coarse part of a graded mesh from reading as a gap. `cell` is the typical cell,
    `span` the nodes' extent."""
    try:
        import numpy as np
        from scipy.spatial import ConvexHull, Delaunay, cKDTree
    except Exception:                                        # noqa: BLE001
        return {}
    out: dict = {}
    (x0, x1), (y0, y1) = box
    tol = 1e-9 * max(x1 - x0, y1 - y0)
    for lvl, rows in sorted(_dump_raw_levels(side_dir, field, since).items()):
        if len(rows) < 8:
            continue
        pts = np.unique(np.asarray([(r[0], r[1]) for r in rows], float), axis=0)
        ux, uy = np.unique(pts[:, 0]), np.unique(pts[:, 1])
        if (len(ux) * len(uy) == len(pts) and ux[0] <= x0 + tol and ux[-1] >= x1 - tol
                and uy[0] <= y0 + tol and uy[-1] >= y1 - tol):
            continue                         # a whole tensor grid over the box fills it
        try:
            tri, hull = Delaunay(pts), ConvexHull(pts)
        except Exception:                                    # noqa: BLE001
            continue
        s = tri.simplices
        e = np.unique(np.sort(np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [2, 0]]]), axis=1), axis=0)
        ln = np.linalg.norm(pts[e[:, 0]] - pts[e[:, 1]], axis=1)
        at, ls = np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([ln, ln])
        o = np.lexsort((ls, at))
        cnt = np.bincount(at, minlength=len(pts))
        size = np.where(cnt > 0, ls[o][np.minimum(np.searchsorted(at[o], np.arange(len(pts)))
                                                  + cnt // 2, len(ls) - 1)], np.inf)
        n = max(16, min(512, int(round(4 * math.sqrt(len(rows))))))
        gx, gy = np.meshgrid(x0 + (np.arange(n) + 0.5) * (x1 - x0) / n,
                             y0 + (np.arange(n) + 0.5) * (y1 - y0) / n, indexing="ij")
        g = np.column_stack([gx.ravel(), gy.ravel()])
        inside = (g @ hull.equations[:, :2].T + hull.equations[:, 2] <= tol).all(axis=1)
        tp = pts[s]
        cell = float(np.median(np.linalg.norm(tp - np.roll(tp, 1, axis=1), axis=2).max(axis=1)))
        dist, near = cKDTree(pts).query(g)
        far = dist > np.maximum(size[near], cell)
        outside, apart = float(np.mean(~inside)), float(np.mean(inside & far))
        if outside > 0.02 or apart > 0.02:
            out[lvl] = {"outside": outside, "apart": apart, "cell": cell,
                        "span": (float(pts[:, 0].min()), float(pts[:, 0].max()),
                                 float(pts[:, 1].min()), float(pts[:, 1].max()))}
    return out


def _gap_words(g: dict) -> str:
    """The measured shares of one _box_fill level, in words."""
    return " and ".join(
        ([f"{g['outside']:.1%} of the box lies outside the dump's nodes"] if g["outside"] > 0.02 else [])
        + ([f"{g['apart']:.1%} lies inside their outline but farther than one mesh cell (about "
            f"{g['cell']:.3g}) from every node"] if g["apart"] > 0.02 else []))


def _all_outer_held(d: Path):
    """True/False where a side states whether the edges the interface ends on are held
    (config full_outer_dirichlet, else its script's FULL_OUTER_DIRICHLET); None otherwise."""
    try:
        cfg = json.loads((Path(d) / "config.json").read_text() or "{}")
    except (OSError, ValueError):
        cfg = {}
    if isinstance(cfg, dict) and isinstance(cfg.get("full_outer_dirichlet"), bool):
        return cfg["full_outer_dirichlet"]
    for q in sorted(q for q in Path(d).glob("participant*.py") if not _is_backup(q)):
        if ".replaced-" in q.name:
            continue
        try:
            m = re.search(r"^FULL_OUTER_DIRICHLET\s*=\s*(True|False)\b", q.read_text(errors="ignore"), re.M)
        except OSError:
            m = None
        if m:
            return m.group(1) == "True"
    return None


def data_compatibility_findings(work: Path) -> list[dict]:
    """Two conditions a smooth solution puts on the data a side states -- an ADVISORY.

    (i) Where two edges held at one constant meet at a right angle, u_xx = 0 along one
    and u_yy = 0 along the other, so -k lap(u) + c u = f leaves f = c u there. (ii) At an
    end of a straight interface on such a held edge, both sides have zero second
    derivative along the edge and share the trace's along the interface, so
    (f - c u)/k is the same on both sides there. MEASURED (a critical review, over every
    coupled result set on record whose configs state a box, a scalar k and a source: 168
    of them): both conditions flag the same 8, every one failed or wrong -- a constant
    of the source transcribed 10x too small or too large -- and no correct one. It reads
    only the data the configs state, never a task file. A problem whose solution has a
    genuine corner singularity also trips it, so it advises and never stops."""
    out: list[dict] = []
    try:
        import numpy as _np
        from .pde_consistency import _eval_source
    except Exception:                                        # noqa: BLE001
        return out
    # WHICH EDGES A SIDE HOLDS: stated (full_outer_dirichlet / FULL_OUTER_DIRICHLET with an outer
    # value), else read off the side's own finest field dump -- an edge along which it is constant.
    # Measured: the advisory never judged a 4C side (its scaffold states neither flag) nor a side
    # whose config states no outer value, so a source slip on either drew nothing.
    ops = {s: o for s, o in _side_operators(work).items()
           if o.get("box") and isinstance(o.get("k"), float) and o.get("source")
           and not o.get("three_d")}
    if not ops:
        return out
    _E = {"left": (0, 0), "right": (0, 1), "bottom": (1, 0), "top": (1, 1)}

    def _iface_edge(side, op):
        e = str(op.get("iface") or "")
        if e in _E:
            return e
        (x0, x1), (y0, y1) = op["box"]
        try:
            c = float(e)
            hits = [n for n, (ax, end) in _E.items()
                    if abs(c - ((x0, x1), (y0, y1))[ax][end]) < 1e-9 * max(x1 - x0, y1 - y0)]
            if len(hits) == 1:
                return hits[0]
        except (TypeError, ValueError):
            pass
        for other, oo in ops.items():              # the edge this box shares with a partner's box
            if other == side:
                continue
            (a0, a1), (b0, b1) = oo["box"]
            tol = 1e-9 * max(x1 - x0, y1 - y0)
            for name, hit in (("right", abs(x1 - a0) < tol), ("left", abs(x0 - a1) < tol),
                              ("top", abs(y1 - b0) < tol), ("bottom", abs(y0 - b1) < tol)):
                if hit:
                    return name
        return None

    def _dump_held(op, iface):
        """{edge: value} for each non-interface box edge along which the finest dump is constant."""
        d = Path(op["dir"])
        dumps = sorted(d.glob("field_level*.csv"), key=lambda q: _level_of(q) or 0)
        if not dumps:
            return {}
        try:
            a = _np.loadtxt(dumps[-1], delimiter=",", skiprows=1, ndmin=2)[:, :3]
        except Exception:                                    # noqa: BLE001
            return {}
        (x0, x1), (y0, y1) = op["box"]
        size = max(x1 - x0, y1 - y0)
        peak = float(_np.max(_np.abs(a[:, 2]))) if len(a) else 0.0
        if not peak > 0:
            return {}
        held = {}
        for name, (ax, end) in _E.items():
            if name == iface:
                continue
            at = ((x0, x1), (y0, y1))[ax][end]
            on = a[_np.abs(a[:, ax] - at) < 1e-6 * size, 2]
            if len(on) >= 3 and float(_np.ptp(on)) <= 1e-9 * peak:
                held[name] = float(_np.mean(on))
        return held

    def _f(op, x, y):
        v = _eval_source(op["source"], [(float(x), float(y))], 2)
        return float(_np.asarray(v).ravel()[0])

    ends = {}
    for side, op in sorted(ops.items()):
        edge = _iface_edge(side, op)
        if edge is None:
            continue
        stated = _all_outer_held(op["dir"])
        if stated is True and isinstance(op.get("outer"), float):
            held = {n: op["outer"] for n in _E if n != edge}
        elif stated is False:
            continue                                         # the side says its end edges are free
        else:
            held = _dump_held(op, edge)
            if isinstance(op.get("outer"), float):             # a stated outer value decides the value
                held = {n: v for n, v in held.items() if abs(v - op["outer"]) <= 1e-9 * max(1.0, abs(op["outer"]))}
        if not held:
            continue
        (x0, x1), (y0, y1) = op["box"]
        try:
            gx, gy = _np.meshgrid(_np.linspace(x0, x1, 41), _np.linspace(y0, y1, 41))
            fmax = float(_np.max(_np.abs(_eval_source(op["source"], list(zip(gx.ravel(), gy.ravel())), 2))))
        except Exception:                                    # noqa: BLE001
            continue
        if not fmax > 0:
            continue
        c = op.get("reaction", 0.0)
        for ex in ("left", "right"):                          # (i) every corner where two held edges meet
            for ey in ("bottom", "top"):
                if ex not in held or ey not in held or abs(held[ex] - held[ey]) > 1e-9 * max(1.0, abs(held[ex])):
                    continue
                px, py = (x0, x1)[_E[ex][1]], (y0, y1)[_E[ey][1]]
                try:
                    fc = _f(op, px, py)
                except Exception:                            # noqa: BLE001
                    continue
                want = c * held[ex]
                if abs(fc - want) > 1e-6 * fmax:
                    out.append({"sequence": f"data compatibility side {side}", "priority": 8,
                                "values": [abs(fc - want) / fmax], "finding": (
                        f"SIDE {side}'S STATED SOURCE IS NOT WHAT A SMOOTH SOLUTION NEEDS AT A HELD CORNER: at "
                        f"({px:g}, {py:g}), where two edges held at {held[ex]:g} meet, your config's source_expr "
                        f"gives f = {fc:.4g} ({abs(fc - want) / fmax:.0%} of its largest value on the box), and a "
                        f"solution smooth there needs f = {want:g} (u is constant along both edges, so "
                        f"-k(u_xx + u_yy) + c u = f leaves f = c u). Either the problem's solution is not smooth "
                        f"at that corner, or a term of source_expr differs from the problem statement: compare it "
                        f"term by term with your task, and the same terms in the participant script. This reads "
                        f"only the data your config states"
                        + ("" if stated is True else ", and which edges are held from your own field dump")
                        + ".")})
        ax, end = _E[edge]
        at = ((x0, x1), (y0, y1))[ax][end]
        for n, (bx, bend) in _E.items():                      # (ii) the interface's ends on held edges
            if n == edge or bx == ax or n not in held:
                continue
            t = ((x0, x1), (y0, y1))[bx][bend]
            px, py = (at, t) if ax == 0 else (t, at)
            try:
                ends.setdefault((round(px, 9), round(py, 9)), []).append(
                    (side, (_f(op, px, py) - c * held[n]) / op["k"]))
            except Exception:                                # noqa: BLE001
                continue
    for (px, py), vals in sorted(ends.items()):
        if len(vals) != 2:
            continue
        (sa, a), (sb, b) = vals
        scale = max(abs(a), abs(b))
        if scale > 0 and abs(a - b) > 1e-6 * scale:
            out.append({"sequence": "data compatibility interface", "priority": 8,
                        "values": [abs(a - b) / scale], "finding": (
                f"THE TWO SIDES' STATED DATA DISAGREE AT AN END OF THE INTERFACE, ({px:g}, {py:g}): a solution "
                f"smooth there needs (f - c u)/k to be the same on both sides (the held edge fixes u, and the "
                f"trace along the interface is shared), and your configs give {a:.4g} on side {sa} and {b:.4g} "
                f"on side {sb} ({abs(a - b) / scale:.0%} apart). Either the solution is not smooth there, or a "
                f"coefficient or a source term of one side differs from the problem statement: compare both "
                f"sides' source_expr and k term by term with your task.")})
    return out


class _Decided(Exception):
    """The discrete check has decided a side; the weak identity is not asked."""


def _op_text(op: dict) -> str:
    """The equation a side's config states, as its findings name it."""
    return ((f"-div(K grad u)" if op.get("k_tensor") else f"-div({op['k']:g} grad u)")
            + (f" + {op['reaction']:g} u" if op["reaction"] else "")
            + " = f" + (f" with K = {_k_text(op['k'])}" if op.get("k_tensor") else ""))


def _discrete_verdict(op: dict, since=None, gaps=(), level=None):
    """pde_consistency.check_levels_discrete on a side's own dumps (the levels that fill its box,
    one level when `level` is given), as a dict; None when there is nothing to judge."""
    raw = {lv: rows for lv, rows in _dump_raw_levels(op["dir"], op.get("field"), since).items()
           if lv not in gaps and (level is None or lv == level)}
    if not raw:
        return None
    try:
        from .pde_consistency import check_levels_discrete
        return check_levels_discrete(raw, op["source"], op["k"], op["box"], reaction=op["reaction"]).as_dict()
    except Exception:                                        # noqa: BLE001
        return None


def level_equation_findings(work: Path, dirs=None, since=None) -> list[dict]:
    """The sides whose field of this level its own discrete system condemns: A u one factor
    other than one times the load of the source its config states (pde_consistency.
    _one_factor_condemns). Blocking findings, for the level's own verdict in couple(); a side
    this cannot judge is left to equation_findings, which says why."""
    want = {Path(d).resolve() for d in dirs} if dirs else None
    out = []
    for side, op in sorted(_side_operators(work).items()):
        if op.get("missing") or op.get("scalar_na") or "box" not in op:
            continue
        if want is not None and Path(op["dir"]).resolve() not in want:
            continue
        try:
            if _transient_marks(op["dir"]) or _k_differs_by_leg(op["dir"]):
                continue
            gaps = _box_fill(op["dir"], op["box"], op.get("field"), since)
        except Exception:                                    # noqa: BLE001
            continue
        disc = _discrete_verdict(op, since, gaps)
        if disc and str(disc.get("verdict")) == "INCONSISTENT":
            out.append(_verdict_finding(side, "equation", _op_text(op), disc, None, op["dir"].name))
    return out


def equation_findings(work: Path, since=None) -> list[dict]:
    """Does each side's delivered field satisfy the equation that side states?

    NOT VOLUNTARY, FOR THE REASON THE SIGN CHECK IS NOT. `verify_pde_consistency`
    is the only check that separates a field converging cleanly to the RIGHT
    function from one converging just as cleanly to a wrong one -- a refinement
    study cannot, and the run's own mesh-independence verdict fires on half the
    correct runs. Measured across every coupled cell on disk: 1118 runs, the
    tool called in ONE. Two wordings of the invitation were tried and measured
    (2 calls against 14 asks, then 0 against 7). So it is computed here, from
    files the agent already wrote, and reported whether or not anyone asks.

    It judges the field against the operator the AGENT'S OWN config.json
    states, so it is a self-consistency check and leaks nothing: no task file,
    no key, no reference solution. A side whose config does not state the
    operator is told which key is missing rather than guessed at.
    """
    out: list[dict] = []
    _EQS = "the steady scalar equation -div(k grad u) + c u = f and the elastic momentum equation only"
    for side, op in sorted(_side_operators(work).items()):
        if op.get("fluid"):
            out.append({"sequence": f"equation check side {side}", "values": [], "priority": 26,
                        "informational": True, "finding": _not_judged_as_fluid(
                            f"SIDE {side}'S FIELD WAS NOT CHECKED AGAINST ITS OWN EQUATION", op["fluid"], _EQS)})
            continue
        if op.get("scalar_na") or op.get("dsmc"):
            continue
        if op.get("three_d"):
            out.append({"sequence": f"equation check side {side}", "values": [], "priority": 26,
                        "informational": True, "finding": _not_judged_in_3d(side, "FIELD")})
            continue
        _tm = _transient_marks(op["dir"])
        if _tm:
            out.append(_transient_note(side, _tm))
            continue
        if op.get("missing") and op.get("vector"):
            # A SIDE THAT EXCHANGES A DISPLACEMENT IS ASKED FOR ITS MOMENTUM DATA. Measured on three
            # coupled rounds: every FEBio elastic side was told its config.json lacked "k" and
            # "source_expr", keys of a heat equation its problem does not have.
            out.append({"sequence": f"equation check side {side}", "values": [], "priority": 26,
                        "informational": True, "finding": (
                f"SIDE {side}'S DISPLACEMENT WAS NOT CHECKED AGAINST ITS MOMENTUM EQUATION: it exchanges a "
                f"vector, and its ./config.json does not state its box (x0, x1, y0, y1), its material (E "
                f"and nu, or lam and mu) and its body force (source_ux, source_uy; \"0\" when there is "
                f"none) -- the numbers your code solves with. The elastic contracts read them from that "
                f"file; with them this audit judges the delivered displacement against them.")})
            continue
        if op.get("missing"):
            _kd = _k_not_one_number(op["dir"])
            # WHAT ONE BOX AND ONE k CANNOT STATE IS NOT ASKED FOR (measured: sides with
            # several materials were told to state "the coefficient k").
            _cs = "" if _kd else _one_box_one_k_cannot_state(op["dir"])
            if _cs:
                out.append({"sequence": f"equation check side {side}", "values": [],
                            "priority": 26, "informational": True, "finding": (
                    f"SIDE {side}'S FIELD WAS NOT CHECKED AGAINST ITS OWN EQUATION: {_cs}. This "
                    f"check judges -div(k grad u) = f with one k over one box, so it does not judge "
                    f"this side, and no key in its config.json changes that.")})
                continue
            out.append({
                "sequence": f"equation check side {side}", "values": [],
                "priority": 26, "informational": True,
                "finding": (
                    f"SIDE {side}'S FIELD WAS NOT CHECKED AGAINST ITS OWN EQUATION: "
                    f"its ./config.json states k as {_kd}. This check judges -div(K grad u) = f "
                    f"with one K over one box -- one number, or one 2x2 matrix [[kxx, kxy], [kyx, "
                    f"kyy]] (four numbers row by row read the same) -- so it does not judge a "
                    f"coefficient stated another way or one that varies by region."
                    if _kd else
                    f"SIDE {side}'S FIELD WAS NOT CHECKED AGAINST ITS OWN EQUATION: "
                    f"its ./config.json does not state {', '.join(op['missing'])}."
                    + (" Its script's F_SRC is not read by this check." if _f_src_in_script(op["dir"])
                       and any(m.startswith("source_expr") for m in op["missing"]) else "")
                    + f" State there the subdomain box (x0, x1, y0, y1), the coefficient "
                    f"k, the reaction (0 when there is none) and source_expr -- the "
                    f"numbers your code SOLVES with. The NGSolve, scikit-fem, FEniCSx, deal.II and "
                    f"Kratos-Neumann heat contracts read the box, k and source_expr from "
                    f"that file, and the NGSolve and deal.II ones the reaction too; a side that keeps them "
                    f"as constants in its code must state the same numbers there. With them "
                    f"this audit checks the delivered field against the equation you "
                    f"implemented, which is the one check that separates a field converging "
                    f"to the right function from one converging to a wrong one. It judges "
                    f"STEADY sides only; a time-dependent side is not judged. An ELASTIC "
                    f"side states lam and mu (or E and nu) and source_ux, source_uy instead, "
                    f"and is judged against its momentum equation.")})
            continue
        # ONE k OVER A SIDE WHOSE COEFFICIENT DIFFERS ALONG ITS INTERFACE is not judged either.
        try:
            _kl = _k_differs_by_leg(op["dir"])
        except Exception:                                    # noqa: BLE001
            _kl = None
        if _kl:
            out.append({"sequence": f"equation check side {side}", "values": [],
                        "priority": 26, "informational": True, "finding": (
                f"SIDE {side}'S FIELD WAS NOT CHECKED AGAINST ITS OWN EQUATION: at level {_kl[0]} "
                f"its exported flux over its own field's -du/dn reads "
                + " and ".join(f"{k:.4g} along the leg through ({c[0]:.4g}, {c[1]:.4g})" for c, k, _m, _s in _kl[1])
                + f", so the coefficient its field carries differs along the interface, and its "
                f"config.json states one k = {op['k']:g}. This check judges -div(k grad u) = f with "
                f"one k over one box, so it does not judge this side.")})
            continue
        sets = _per_level_field_sets(work, op["box"])
        dumps = _dump_level_sets(op["dir"], op["box"], "scalar", op.get("field"), since)
        # A MESH THAT IS NOT THE BOX IS NOT JUDGED ON THE BOX. The dumps are interpolated
        # to midpoints over the box the config states; where the side's mesh has no cells
        # (a notch, cells removed) linear interpolation fills the gap with values the side
        # never computed. Measured on the side's own nodes, and said with the numbers.
        _gaps = _box_fill(op["dir"], op["box"], op.get("field"), since)
        dumps = {lv: rows for lv, rows in dumps.items() if lv not in _gaps}
        # EACH LEVEL'S OWN DISCRETE SYSTEM IS JUDGED FROM THE FIRST LEVEL, and a load off by one
        # factor decides (pde_consistency._one_factor_condemns). Measured: a side whose load left
        # out its test function read A u = 3.87 times the load at level 1, and the level read
        # VERIFIED because this check waited for more levels.
        _disc = _discrete_verdict(op, since, _gaps)
        _factor = bool(_disc) and str(_disc.get("verdict")) == "INCONSISTENT"
        if _gaps and len(dumps) < 2 and not _factor:
            _lv = min(_gaps)
            _g = _gaps[_lv]
            _sp = _g["span"]
            out.append({
                "sequence": f"equation check side {side}", "values": [],
                "priority": 26, "informational": True,
                "finding": (
                    f"SIDE {side}'S FIELD WAS NOT CHECKED AGAINST ITS OWN EQUATION: its mesh "
                    f"dumps in {op['dir'].name}/ do not fill the box its config.json states, "
                    f"{op['box']}, at level(s) {', '.join(str(q) for q in sorted(_gaps))}. At "
                    f"level {_lv} the dump's nodes span x {_sp[0]:g} to {_sp[1]:g} and y "
                    f"{_sp[2]:g} to {_sp[3]:g}; {_gap_words(_g)}. This check judges -div(k grad u) = f over that whole box with one k, "
                    f"so it does not judge a side whose mesh is another shape (a box with "
                    f"cells removed, joined boxes), nor a box stated wrong.")})
            continue
        if not sets and len(dumps) < 2 and not _factor:
            _nf = nonfinite_field_findings(work, dirs=[op["dir"]], since=since, scan=False)
            _files = list(op["dir"].glob("field_level*.csv"))
            _have = len(_files)
            _cut = _cutoff_for(op["dir"], since)
            try:
                _old = sum(1 for q in _files if _cut is not None and q.stat().st_mtime < _cut)
            except OSError:
                _old = 0
            out.append({
                "sequence": f"equation check side {side}", "values": [],
                "priority": 26, "informational": True,
                "finding": (
                    f"SIDE {side}'S FIELD WAS NOT CHECKED AGAINST ITS OWN EQUATION: "
                    + (f"its {_have} mesh dump(s) in {op['dir'].name}/ hold non-finite values, "
                       f"so there is no field to judge (see the finding that names them)."
                       if _nf else
                       f"its {_have} mesh dump(s) in {op['dir'].name}/ give {len(dumps)} "
                       f"level(s) this check can use: "
                       + "; ".join(
                           ([f"{_old} older than this run's script and config"] if _old else [])
                           + ([f"{_have - _old - len(dumps)} not readable as rows of x, y and "
                               f"the field on its box {op['box']}"]
                              if _have - _old - len(dumps) > 0 else []))
                       + ". No per-level file of yours holds that side's field at the cell "
                       f"midpoints either."
                       if _have >= 2 else
                       f"neither two or more of its own mesh dumps (field_level<k>.csv in "
                       f"{op['dir'].name}/) nor a per-level file of yours holding that side's "
                       f"field at the cell MIDPOINTS of its box {op['box']} were found. The "
                       f"served contracts write the dumps at every level run; the check "
                       f"interpolates them to the midpoints, where its quadrature is exact."))})
            continue
        try:
            from .pde_consistency import check_levels
            if _factor:
                raise _Decided()
            # A CONSTANT K ACTS THROUGH ITS SYMMETRIC PART: div(K grad u) = sum K_ij d_i d_j u.
            # The weak identity refuses a K it is not handed symmetric, so it gets that part.
            _kw = op["k"]
            if op.get("k_tensor"):
                _kw = [[op["k"][0][0], 0.5 * (op["k"][0][1] + op["k"][1][0])],
                       [0.5 * (op["k"][0][1] + op["k"][1][0]), op["k"][1][1]]]
            dres, vres = _judge_pair(
                lambda lv: check_levels(lv, op["source"], _kw, op["box"],
                                        reaction=op["reaction"]), dumps, sets)
            dres = _on_the_grid_itself(dres, _dump_raw_levels(op["dir"], op.get("field"), since),
                                       dict(op, k=_kw))
            # A TENSOR SIDE IS ALSO JUDGED ON ITS OWN DISCRETE SYSTEM (pde_consistency.
            # check_levels_discrete): round-off there confirms it; anything else is said with
            # the numbers and the weak identity decides.
            if op.get("k_tensor") and _disc:
                if _disc.get("verdict") == "CONSISTENT":
                    dres = _disc
                elif dres and str(dres.get("verdict")) != "CONSISTENT":
                    dres["explanation"] = (str(dres.get("explanation") or "") + " The discrete "
                                           "check: " + str(_disc.get("explanation") or ""))
        except _Decided:
            dres, vres = _disc, None
        except Exception as exc:                              # noqa: BLE001
            out.append({"sequence": f"equation check side {side}", "values": [],
                        "priority": 26, "informational": True,
                        "finding": (f"SIDE {side}'S EQUATION CHECK COULD NOT RUN: "
                                    f"{type(exc).__name__}: {exc}")})
            continue
        out.append(_timed_verdict(side, "equation", _op_text(op), dres, vres, op["dir"], work))
        if _gaps and out[-1].get("finding"):
            out[-1]["finding"] += (
                f" Not judged: the mesh dump(s) of level(s) "
                f"{', '.join(str(q) for q in sorted(_gaps))}, which do not fill the box "
                f"{op['box']} (at level {min(_gaps)}, {_gap_words(_gaps[min(_gaps)])}).")
        # A CONFIG WRITTEN AFTER THE FIELD IT DESCRIBES is said, not assumed: measured, two
        # runs wrote side A's source_expr minutes after their last dump, and the verdict
        # judged the field against a statement the run did not carry when it solved.
        try:
            _cfg_t = (op["dir"] / "config.json").stat().st_mtime
            _dump_t = [q.stat().st_mtime for q in op["dir"].glob("field_level*.csv")]
            if _dump_t and _cfg_t > max(_dump_t) + 1.0 and out and out[-1].get("finding"):
                out[-1]["finding"] += (
                    f" Its config.json was written after these dumps "
                    f"({_cfg_t - max(_dump_t):.0f} s after the last): the operator judged is the "
                    f"one it states now, which is not proof it is the one the run solved.")
            # AND A SCRIPT CHANGED AFTER THEM (measured: a side's final script was not the
            # version that wrote the dumps judged here, and nothing said so).
            _scr = [q for q in op["dir"].glob("participant*.py") if ".replaced-" not in q.name]
            _scr_t = max((q.stat().st_mtime for q in _scr), default=0.0)
            if _dump_t and _scr_t > max(_dump_t) + 1.0 and out and out[-1].get("finding"):
                out[-1]["finding"] += (
                    f" Its participant script was changed after these dumps "
                    f"({_scr_t - max(_dump_t):.0f} s after the last): the field judged is the one "
                    f"an earlier version wrote.")
        except OSError:
            pass

    for side, op in sorted(_side_operators(work).items()):
        if not op.get("three_d") and not op.get("fluid"):
            out.extend(_momentum_findings(work, side, op, since))
    # A SIDE WITH NO config.json WAS PASSED OVER IN SILENCE. couple_levels hands
    # each level its keys through the environment and writes no file, so a side
    # whose problem data sit in its code has nothing here to be judged against
    # -- and a clean audit of it read as a checked one.
    stated = set(_side_operators(work))
    for d in _side_dirs(work):
        if d.name in stated or not (d / "exports.json").is_file():
            continue
        _fl = _fluid_side(d)
        if _fl:
            out.append({"sequence": f"equation check side {d.name}", "values": [], "priority": 26,
                        "informational": True, "finding": _not_judged_as_fluid(
                            f"SIDE {d.name}'S FIELD WAS NOT CHECKED AGAINST ITS OWN EQUATION", _fl, _EQS)})
            continue
        if _dumps_are_3d(d):
            out.append({"sequence": f"equation check side {d.name}", "values": [], "priority": 26,
                        "informational": True, "finding": _not_judged_in_3d(d.name, "FIELD")})
            continue
        _tm = _transient_marks(d)
        if _tm:
            out.append(_transient_note(d.name, _tm))
            continue
        _cs = _one_box_one_k_cannot_state(d)
        if _cs:
            out.append({"sequence": f"equation check side {d.name}", "values": [],
                        "priority": 26, "informational": True, "finding": (
                f"SIDE {d.name}'S FIELD WAS NOT CHECKED AGAINST ITS OWN EQUATION: {_cs}. This "
                f"check judges -div(k grad u) = f with one k over one box, so it does not judge "
                f"this side, and no key in a config.json changes that.")})
            continue
        out.append({
            "sequence": f"equation check side {d.name}", "values": [],
            "priority": 26, "informational": True,
            "finding": (
                f"SIDE {d.name}'S FIELD WAS NOT CHECKED AGAINST ITS OWN EQUATION: that "
                f"directory has no ./config.json stating the problem it solved, so this "
                f"audit had nothing to judge the field against. State there the subdomain box "
                f"(x0, x1, y0, y1) and the equation's data -- k, reaction and source_expr for a "
                f"scalar side; lam and mu (or E and nu) with source_ux and source_uy for an "
                f"elastic one. The NGSolve, scikit-fem, FEniCSx, deal.II and Kratos-Neumann heat "
                f"contracts read the box, k and source_expr from that file; the scikit-fem, "
                f"FEniCSx, NGSolve and DUNE elastic contracts read the box and those elastic "
                f"keys. Data typed only into the code cannot be checked. The check judges "
                f"STEADY sides only.")})
    return out


def wrong_level_run_log_findings(work: Path) -> list[dict]:
    """A run log that is a copy of ANOTHER level's console. Measured (round 29): four cells coupled
    three refined levels (consoles 54, 187, 693 dofs) and handed in run logs reading 54 at every
    level -- graded as an unchanged mesh. Compares each run_level<k>_<side>.log's NDOF line with the
    side's captured participant_output_level<k>.log and names the file to copy."""
    import re as _re
    out = []
    # the grader reads the FIRST canonical `NDOF = <n>` line of a run log; so does this check
    dof_any = _re.compile(r"^\s*NDOF\s*=\s*(\d+)\s*$", _re.M)
    consoles = {}
    for q in work.rglob("participant_output_level*.log"):
        if _SCRATCH & set(q.relative_to(work).parts[:-1]):
            continue
        m = _re.search(r"participant_output_level(\d+)\.log$", q.name)
        if not m:
            continue
        side_dir = q.parent.name
        try:
            vals = dof_any.findall(q.read_text(errors="ignore"))
        except OSError:
            continue
        if vals:
            consoles.setdefault(int(m.group(1)), {})[side_dir] = (vals[0], q)
    if not consoles:
        return out
    for q in _level_logs(work):
        m = _LEVEL_FILE.match(q.name)
        if not m or not m.group("side"):
            continue
        k, side = int(m.group("k")), m.group("side")
        try:
            vals = dof_any.findall(q.read_text(errors="ignore"))
        except OSError:
            continue
        if not vals or k not in consoles:
            continue
        # the side's console dir: a directory whose name ends with the side letter (side_A, A, sideA ...)
        match = [(d, v) for d, v in consoles[k].items() if d.lower().rstrip("/").endswith(side.lower())]
        if len(match) != 1:
            continue
        d, (ndof, cpath) = match[0]
        if vals[0] != ndof:
            other = [j for j, sides in consoles.items() if sides.get(d, (None,))[0] == vals[0]]
            out.append({"sequence": f"run log level {k} side {side}", "values": [], "priority": 20,
                        "finding": (f"RUN LOG FROM THE WRONG LEVEL: {q.relative_to(work)} carries NDOF {vals[0]} while "
                                    f"side {side}'s captured console for level {k} says NDOF {ndof}"
                                    + (f" -- it is level {other[0]}'s console" if other else "")
                                    + f". Copy {cpath.relative_to(work)} over it verbatim; a run log from another level "
                                    "reads as an unchanged mesh and sinks the whole mesh sequence.")})
    return out


_WORKER_SEES_ONLY_THE_BRIEF = (  # kept for reference; the briefs now carry a <...> block instead
    "THE WORKER SEES ONLY THIS BRIEF, NOT YOUR TASK: paste that subdomain's data from your task into it verbatim -- geometry and interface position, equations and coefficients, source terms as written, boundary values, the level-1 mesh, and the file names your task prescribes for this side. ")


def coupled_ladder(work: Path, converged_now: bool = False) -> dict | None:
    """The next unmet step of a partitioned coupled run, read from the files,
    phrased as the brief of the sub-agent that should do it.

    A small model that sees the whole job at once judges it too big and gives
    up (measured: 8 of 9 runs in one round, most with 25 minutes left); given
    ONE step whose end is a file it can see, it keeps the served contract and
    finishes the step (measured in single-step trials: 6 of 6 kept it against
    0 of 45 whole-problem runs). So every audit and every couple() reply names
    the next unmet step -- two participant scripts -> each exports standalone
    -> a coupling history per level -> that level's field and interface files
    for both sides -> that level's captured run logs -> the summary -- and
    returns it as a brief to hand to a sub-agent. No task knowledge: every
    check reads the agent's own files; the level count is the task's.
    """
    import re as _re
    try:
        from blind_eval.evidence import strip_terminal_noise
        sig_pats = solver_line_patterns()
    except Exception:                                   # noqa: BLE001
        sig_pats, strip_terminal_noise = [], (lambda t: t)
    scripts, exports = [], []
    for q in work.rglob("*"):
        if not q.is_file():
            continue
        try:
            rel = q.relative_to(work)
        except ValueError:
            continue
        if _SCRATCH & set(rel.parts[:-1]):
            continue
        if q.suffix == ".py" and ".replaced-" not in q.name:
            # (write_participant_contract's backup of a file it replaced is kept
            # for the agent to take back, not run: it was told to "RESTORE THE
            # SERVED CONTRACT" in it)
            try:
                t = q.read_text(errors="ignore")
            except OSError:
                continue
            if "imports.json" in t and "exports.json" in t and not _re.search(
                    r"run_|driver|coupl|orchestr|main_", q.name, _re.I):
                scripts.append(q)          # a participant, not the driver that launches them
        elif q.name == "exports.json":
            try:
                e = json.loads(q.read_text())
                if e.get("normal_fluxes") or e.get("values"):
                    exports.append(q)
            except Exception:                           # noqa: BLE001
                continue
    hist = _history_files(work)
    if not (scripts or exports):
        return None                    # no sign of a partitioned coupling here
    fields = _field_files(work)
    ifaces = _interface_files(work, sided=True)
    logs = _level_logs(work)

    def step(n, what, brief, as_is=True):
        hand = ("HAND THIS STEP TO A SUB-AGENT AS IS" if as_is else
                "HAND THIS STEP TO A SUB-AGENT WITH THE <...> BLOCK REPLACED BY YOUR TASK'S OWN WORDS -- the worker "
                "sees nothing else (measured: a worker briefed without them wrote placeholder source terms)")
        return {"step": n, "text": (
            f"LADDER STEP {n} OF 6 -- {what}\n"
            f"{hand}: spawn_subagent(role='worker', "
            f"task=\"{brief}\"). The step ends when its check passes on disk; "
            f"the sub-agent reports the exact error otherwise. Judge nothing about "
            f"the whole task -- only this step."), "brief": brief}

    if len(scripts) < 2:
        have = (f"{len(scripts)} participant script(s) found"
                + (f" ({scripts[0].relative_to(work)})" if scripts else ""))
        return step(1, f"WRITE THE {'SECOND' if scripts else 'FIRST'} PARTICIPANT: {have}.",
                    "Write the participant script of ONE code for ONE subdomain in its own directory ./side_<x>: "
                    "call knowledge(topic='coupling', solver=<that code>) and copy the served CONTRACT into "
                    "the file unchanged (imports.json handshake, sign convention, flux recovery, exports "
                    "schema, export self-check); fill only its marked hole(s) with the mesh, form, material, "
                    "source and solve for this subdomain from THIS DATA, which is all you know of the task: "
                    "<THAT SUBDOMAIN, COPIED FROM YOUR TASK WORD FOR WORD: geometry and interface position; equations and "
                    "coefficients; source terms as written; boundary values; the level-1 mesh; the file names the task "
                    "prescribes for this side>. Write ./config.json for level 1 and a "
                    "synthetic ./imports.json whose values and normal_fluxes are NOT zero (on a zero "
                    "import a Neumann side applies no load, and its EXPORT SELF-CHECK cannot fire); "
                    "if the code takes an input deck, run check_input(solver=<that code>, "
                    "input_path=<the deck>) until it names no defect before the binary runs; run the script with "
                    "that code's own interpreter until ./exports.json appears with finite values. CHECK: "
                    "./side_<x>/exports.json exists and the script exited 0.", as_is=False)
    # A PARTICIPANT WRITTEN FROM SCRATCH IS THE NEXT STEP, NOT A DETAIL -- JUDGED BY SERVED
    # LINES, the reference the write checks use (participant_lint._served_line_sets and
    # _is_served_contract), with the share it measured said. Measured on three cells: a Neumann
    # side that put zeros in every point load and reported the code could not apply the flux, a
    # Kratos side whose flux condition carried no nodal value and coupled as unresponsive, a
    # thermo-elastic side with no handshake at all; the served self-checks would have refused
    # each export. FOUR COMMENT MARKERS WERE NOT THAT REFERENCE: a side carrying 125 of its 241
    # code lines verbatim, which had run, held its trace and was correct, was told it "carries
    # none of the served contract's lines", and the run restored the empty contract over it. A
    # side whose current script ran and exported is kept. Fires only before the first
    # converged level: a coupling that already converged a level has working scripts.
    # converged_now is couple()'s own word for the call it is replying to, which
    # leaves no history file when it was given no history_path (measured: a run was
    # told to restore the served contract in the reply to its converged coupling).
    _converged_any = bool(converged_now)
    for q in hist:
        try:
            if _line_count(q) - 1 >= 3:
                _converged_any = True
                break
        except OSError:
            continue
    if not _converged_any:
        try:
            from .participant_lint import _is_served_contract, _served_line_sets   # noqa: PLC0415
            _sets = _served_line_sets()
        except Exception:                                   # noqa: BLE001
            _is_served_contract, _sets = None, ()

        def _ran_as_written(q: Path) -> bool:
            try:
                t0 = q.stat().st_mtime
                outs = [q.parent / "exports.json"] + list(q.parent.glob("field_level*.csv")) \
                    + list(q.parent.glob("interface_level*.csv"))
                return any(f.is_file() and f.stat().st_mtime >= t0 for f in outs)
            except OSError:
                return False
        unserved = []
        for q in scripts:
            try:
                t = q.read_text(errors="ignore")
            except OSError:
                continue
            if _is_served_contract is None or _is_served_contract(t) or _ran_as_written(q):
                continue
            have = {ln.strip() for ln in t.splitlines()}
            n, m = max(((len(ls & have), len(ls)) for ls in _sets), key=lambda nm: nm[0] / nm[1], default=(0, 0))
            unserved.append((q, n, m))
        if unserved:
            who = ", ".join(str(q.relative_to(work)) for q, _, _ in unserved[:2])
            share = "; ".join(
                (f"{q.relative_to(work)} carries {n} of the {m} served lines of the contract it is closest "
                 f"to ({n / m:.0%})" if m else f"{q.relative_to(work)} carries no served line")
                for q, n, m in unserved[:2])
            # THE CONTRACT EACH FILE IS CLOSEST TO, WITH THE CALL THAT WRITES IT, AND EVERY PHYSICS
            # WORD. Measured on a fluid-structure round: this brief named one word, 'thermoelastic';
            # two cells then called the writer with 'fsi', got the scalar heat contract over their
            # fluid file, and one wrote down that no fluid-structure contract exists.
            try:
                from .participant_lint import _side_roles                        # noqa: PLC0415
                from .coupling_knowledge import resolve_participant              # noqa: PLC0415
            except Exception:                                   # noqa: BLE001
                _side_roles = resolve_participant = None
            _calls = []
            for q, _n, _m in unserved[:2]:
                try:
                    # the role the write checks read off the file: its own code, then the role whose
                    # own lines it carries (the handshake helpers every contract shares tell none apart)
                    _roles = _side_roles(q.read_text(errors="ignore")) if _side_roles else []
                except OSError:
                    _roles = []
                if len(_roles) != 1:
                    continue                     # no one contract stands out: none is named
                _code, _role = _roles[0]
                _word = "" if _role == "base" else _role
                if resolve_participant is None or resolve_participant(_code, _word)[0] is None:
                    continue
                _calls.append(f"{q.relative_to(work)} is closest to the served {_code} contract"
                              + (f" (variant '{_word}')" if _word else "")
                              + f": write_participant_contract(solver='{_code}'"
                              + (f", variant='{_word}'" if _word else "")
                              + ", path=<absolute path of a NEW file beside it>) writes that contract")
            return step(1, f"RESTORE THE SERVED CONTRACT IN {who}: {share}. A copy of the served contract "
                           f"carries all of them; one written by hand or rewritten does not (measured: such "
                           f"scripts put zeros in every interface load, or coupled as unresponsive, and "
                           f"reported the code could not do it).",
                        f"For {who}: " + ("; ".join(_calls) + ". Otherwise call " if _calls else "call ")
                        + "knowledge(topic='coupling', solver=<that code>) with the word for what the "
                        "interface carries (physics='thermoelastic' for temperature and displacement together, "
                        "physics='elasticity' for a displacement, physics='transient' for a time-dependent "
                        "problem, physics='3d' for a three-dimensional domain, physics='fsi' for a fluid beside "
                        "a structure; none for a steady scalar one), copy the served CONTRACT for this side's "
                        "role into the file UNCHANGED, and "
                        "move only your mesh, deck/form, material, source and solve into its marked hole(s); keep "
                        "every served line, including the EXPORT SELF-CHECK block. Then write ./config.json for "
                        "level 1 and a synthetic ./imports.json whose values and normal_fluxes are NOT "
                        "zero (on a zero import a Neumann side applies no load, and its EXPORT SELF-CHECK "
                        "cannot fire), and run it with that code's own interpreter until "
                        "./exports.json appears. CHECK: the file contains the served EXPORT SELF-CHECK block "
                        "and the script exited 0 with ./exports.json beside it.")
    dirs_with_export = {q.parent for q in exports}
    # A SIDE THAT WROTE ITS PER-LEVEL DUMPS HAS RUN, whatever became of exports.json
    # (the driver deletes it before every iteration, and a run that deleted it by
    # hand was told "has not exported yet" with three levels on disk -- measured).
    try:
        dirs_with_export |= {d for d in _side_dirs(work)
                             if any(d.glob("field_level*.csv")) or any(d.glob("interface_level*.csv"))}
    except Exception:                                       # noqa: BLE001
        pass
    unrun = [q for q in scripts if q.parent not in dirs_with_export]
    if len(dirs_with_export) < 2 and unrun:
        who = ", ".join(str(q.relative_to(work)) for q in unrun[:2])
        # A 4C SIDE THAT ALREADY RAN THE BINARY: the step is the deck 4C refused, not the
        # whole participant. The side directory holds 4C's own verdict -- its error block
        # in the captured console, the VTU folder of every run that finished, the reaction
        # monitor files -- and the deck defects openPASO can name from the deck text. Put
        # them in the brief, so the worker starts from the defect (measured: a worker
        # handed the whole step again rewrote the deck from scratch and hit the same
        # section a second time). Reads only the agent's own files; writes nothing.
        deck_state = _fourc_deck_state(unrun[0].parent, work)
        if deck_state:
            return step(2, (f"MAKE THE 4C DECK RUN in {who}: " if deck_state.get("fourc", True)
                            else f"MAKE THE PARTICIPANT RUN in {who}: ") + deck_state['what'],
                        f"In the directory of {who}: {deck_state['brief']} Then run the participant again "
                        "with its interpreter. CHECK: ./exports.json appears beside the script with finite "
                        "values and the script exited 0.")
        return step(2, f"RUN EACH PARTICIPANT STANDALONE: {who} has not exported yet.",
                    f"In the directory of {who}: write a synthetic ./imports.json (the partner's field name, "
                    "coordinates along the interface, plausible values and normal_fluxes; the values and "
                    "normal_fluxes are NOT zero, since on a zero import a Neumann side applies no load and "
                    "its EXPORT SELF-CHECK cannot fire) and "
                    "./config.json for level 1; run the script with that code's own interpreter and a "
                    "generous timeout (first runs compile); read the traceback from the top, fix the script, "
                    "repeat. CHECK: ./exports.json appears beside the script with finite values and the "
                    "script exited 0.")
    rows_by_level: dict[int, int] = {}
    for q in hist:
        k = _level_of(q)
        if k is None:
            continue
        try:
            n = _line_count(q) - 1
        except OSError:
            n = 0
        rows_by_level[k] = max(rows_by_level.get(k, 0), n)
    done_levels = sorted(k for k, n in rows_by_level.items() if n >= 3)
    for k in done_levels:
        sides_f = {_side_of(q) for q in fields if _level_of(q) == k and _side_of(q)}
        sides_i = {_side_of(q) for q in ifaces if _level_of(q) == k}
        if len(sides_f) < 2 or len(sides_i) < 2:
            # A COUPLED LEVEL WITHOUT ITS PER-LEVEL DUMPS IS RE-RUN, NOT INTERPOLATED FROM NOTHING. Measured
            # (measured on a recorded run): level 1's scripts wrote no interface_level1.csv, the deliverables pass skipped
            # that level's interface files with a warning, and three second-order levels were read as no result.
            _gap = []
            for _sd in sorted({q.parent for q in scripts}):
                try:   # only a side whose script carries the served dump block is expected to leave dumps
                    if not any("field_level" in q.read_text(errors="ignore") for q in _sd.glob("*.py") if not _is_backup(q)):
                        continue
                except OSError:
                    continue
                for _stem in (f"field_level{k}", f"interface_level{k}"):    # the served name or the agent's suffixed variant
                    if not any(_sd.glob(f"{_stem}*.csv")):
                        _gap.append(f"{_sd.name}/{_stem}*.csv")
            if _gap:
                return step(3, f"LEVEL {k} WAS COUPLED BUT ITS PER-LEVEL DUMPS ARE MISSING ({', '.join(_gap[:4])}): the deliverables "
                               f"cannot be written from them; the participant that ran level {k} did not carry the served dump block.",
                            f"In {', '.join(sorted({str(q.relative_to(work)) for q in scripts}))}: make sure the served block that writes "
                            f"field_level<k>.csv and interface_level<k>.csv after exports.json is present and unchanged (it is part of the "
                            f"contract), then couple() level {k} again with the same participants so its dumps appear; CHECK: both sides' "
                            f"field_level{k}.csv and interface_level{k}.csv exist next to their exports.json.")
            # LEVELS FIRST, DELIVERABLES ONCE -- the ladder says the same as the couple() lead, or the
            # parent copies the ladder's ready-made spawn call and writes level 1's files (measured
            # 2026-09-12: 4 of 4 parents did exactly that while the lead above asked for couple_levels;
            # round 40's C3 6041 lost its third level to the same order). The step carries a <...>
            # block for the levels the task prescribes: the worker sees only the brief.
            nxt = max(done_levels) + 1
            # A SIDE WITHOUT LEVEL-TAGGED OUTPUT IS PRESERVED FIRST, INSIDE THE READY-MADE CALL. Measured
            # (a recorded run; parent test 2026-09-13, 2 of 4): a warning after the one-call text did
            # not move the parent; the spawn call it copies must carry the copy step itself.
            _untagged = []
            for _sd in sorted({q.parent for q in scripts}):
                try:
                    _has = [q.name for q in _sd.glob(f"*level{max(done_levels)}*") if not q.name.startswith("participant_output")]
                except OSError:
                    _has = ["?"]
                if not _has:
                    _untagged.append(_sd.name)
            _preserve = ("" if not _untagged else
                         f"FIRST, in {' and '.join(_untagged)}: this level's outputs carry no level tag (no file named with 'level{max(done_levels)}' "
                         f"beside exports.json), and the next level's run reuses the same names and overwrites them -- copy this level's field "
                         f"file(s), VTU folder and interface data to names containing 'level{max(done_levels)}' NOW (or add the served contract's dump "
                         f"block, which writes field_level<k>.csv and interface_level<k>.csv after exports.json); measured: a side whose runs shared "
                         f"one output prefix handed in identical solution files for every level and no interface file. THEN: ")
            return step(4, f"LEVEL {k}'S DELIVERABLES ARE NOT WRITTEN (coupled so far: levels {done_levels}; field files for "
                           f"{sorted(sides_f) or 'no'} side(s), interface files for {sorted(sides_i) or 'no'} side(s)). If your task "
                           f"prescribes levels beyond {max(done_levels)}, they come FIRST in ONE couple_levels call -- a level set with "
                           f"one level missing is no result at all -- and the deliverables of ALL levels follow in ONE pass.",
                        _preserve + f"<IF THE TASK PRESCRIBES LEVELS BEYOND {max(done_levels)}, START WITH THIS ONE CALL: "
                        f"couple_levels(participants=<the same list passed to couple>, levels=[{nxt}, ...every further level the "
                        f"task prescribes], history_pattern='<the task's per-level history file name with {{k}} in place of the level "
                        f"number>') -- every level keeps field_level<k>.csv, interface_level<k>.csv and participant_output_level<k>.log "
                        f"next to each exports.json; replace this block with the levels and file name, or delete it if level "
                        f"{max(done_levels)} is the last.> THEN, for EVERY coupled level k and EACH side, in ONE script that loops over "
                        "the levels: read that side's converged field (its per-level dumps "
                        "field_level<k>.csv, columns x,y,<field> nodal values, and interface_level<k>.csv, "
                        "columns x,y,<trace>,<flux> at its interface nodes -- NEVER exports.json, which the next "
                        "level overwrites: measured, a run that rebuilt level 1 from exports.json after level 2 "
                        "handed in two byte-identical levels), evaluate it at the probe points the task "
                        "prescribes by interpolating inside the element (never nearest node) and write the "
                        "per-level field file the task names; write the per-level interface file from that "
                        "side's own converged trace and its own outward flux at the prescribed interface "
                        "points -- BOTH from its interface_level<k>.csv (a Dirichlet side's trace there is the "
                        "value it imposed, identical to the partner's export; measured: a run that re-sampled "
                        "its field file at the boundary instead handed in a trace 2x off while its field was "
                        "right, so the result looked unphysical). The interpolation is four lines (measured on a served per-level dump): "
                        "a = numpy.loadtxt('field_level<k>.csv', delimiter=',', skiprows=1); "
                        "v = scipy.interpolate.griddata(a[:, :2], a[:, 2], P, method='linear') with P the "
                        "(n, 2) probe points -- one such call per value column (a[:, 2], a[:, 3], ... for a "
                        "field with several components, T then ux, uy for a thermo-elastic one); a NaN in v "
                        "is a probe outside this side's subdomain (leave it to the other side); write x, y "
                        "and the value columns in the task's order. Then each level's run log per side: a VERBATIM "
                        "copy of participant_output_level<k>.log next to that side's exports.json (never a summary). "
                        "CHECK: every coupled level has both sides' field files, interface files and run logs, and "
                        "audit_results(work_dir) reports no missing-fields and no run-log finding.", as_is=False)
        sides_l = set()
        for q in logs:
            if _level_of(q) != k or not _side_of(q):
                continue
            try:
                txt = strip_terminal_noise(q.read_text(errors="ignore"))
            except Exception:                           # noqa: BLE001
                continue
            if not sig_pats or any(_re.search(pp, txt, _re.IGNORECASE | _re.MULTILINE) for pp in sig_pats):
                sides_l.add(_side_of(q))
        if len(sides_l) < 2:
            return step(5, f"CAPTURE LEVEL {k}'S RUN LOGS: level {k} has a log holding the solver's own "
                           f"console output for {sorted(sides_l) or 'no'} side(s).",
                        f"For level {k}, for EACH side: the per-level run log the task names must hold that "
                        "solver's OWN console output (banner, iteration lines) plus the DOF-count line. The "
                        f"coupling tool kept each participant's captured console for THIS level as "
                        f"participant_output_level{k}.log next to its exports.json (participant_output.log is "
                        "only the latest level's) -- copy that file into the run log; never summarise or retype "
                        "it. CHECK: audit_results(work_dir) reports no run-log finding for this level.")
    next_level = (max(done_levels) + 1) if done_levels else 1
    short = (f" (its history so far has {rows_by_level.get(next_level, 0)} row(s))"
             if next_level in rows_by_level else "")
    if done_levels:
        what = (f"LEVELS {done_levels} ARE COMPLETE ON DISK. If your task prescribes more levels, run them ALL "
                f"in ONE call: couple_levels(participants=<the same list you passed to couple>, levels=[{next_level}, ...], "
                "history_pattern='<the task's per-level history file name with {k} in place of the level>'). "
                "Otherwise write the summary.")
        brief = (f"If the task prescribes a level {next_level}: call couple_levels(participants=<the same list you passed to "
                 f"couple>, levels=[{next_level}, ...every further level the task prescribes], history_pattern='<the task's "
                 "per-level history file name with {k} in place of the level number>') ONCE. It runs every remaining "
                 "level in that single call: each side gets its level's nx, ny in the environment (every cell count "
                 "doubled to halve h), each level warm-starts from the previous one, and each level's history file and "
                 "participant_output_level<k>.log are written; per-level couple() calls cost several calls a level. "
                 "Then do steps 4 and 5 for every new level. "
                 "If the task prescribes no further level: write the summary file naming ONLY files that exist, then run "
                 "audit_results(work_dir). CHECK: the audit reports no missing level, no invented name and no missing "
                 "field file.")
        return step(6, what, brief)
    if converged_now and not hist:
        return step(3, "THE COUPLING THIS REPLY ANSWERS CONVERGED, AND NO RESIDUAL-HISTORY FILE EXISTS ON DISK: couple() "
                       "writes one only when it is given history_path, and it was given none, so no level is on disk as "
                       "coupled yet.",
                    "Call couple again with the same participants and history_path=<absolute path of this level's "
                    "residual-history file, the name your task gives>, or run every level in one couple_levels("
                    "participants=<the same list you passed to couple>, levels=[...], history_pattern='<the per-level "
                    "history file name with {k} in place of the level number>') call. CHECK: the residual-history file "
                    "for this level exists with at least three rows and a falling residual.")
    return step(3, f"COUPLE LEVEL {next_level}{short}.",
                f"Set level={next_level} in both ./config.json; call couple(participants=[{{name, command, "
                f"work_dir (absolute), imports_from}} for both sides], max_iter from the served rho guidance, "
                "tol from the task, history_path=<absolute path of this level's residual-history file>); "
                "read the reply's WHAT TO FIX NEXT and fix the named side until converged is true. Once this "
                "level converges, every further level is ONE couple_levels(participants=<same>, levels=[...], "
                "history_pattern='<per-level history file name with {k}>') call. CHECK: "
                "the residual-history file for this level exists with at least three rows and a falling "
                "residual, and the couple() reply says converged.")


def missing_fields_findings(work: Path) -> list[dict]:
    """A coupling that ran and wrote no field is not a result yet.

    Measured twice on the honest build: a run coupled three real levels
    (driver histories, eight or nine iterations to 1e-7 each), wrote its run
    logs and its residual histories, and handed in with NO per-level field
    file and NO interface file -- once listing invented names, once honestly
    listing only what existed. Nothing in the audit named the missing piece,
    because generic discovery cannot see a family that was never written.
    The agent's own histories and logs are the evidence that the solves ran,
    so the absence of any field file is a finding, not an unknown.
    """
    hist = _history_files(work)
    logs = _level_logs(work)
    if not (hist or logs):
        return []
    if _field_files(work) or _interface_files(work):
        return []
    levels = sorted({_level_of(q) for q in hist + logs if _level_of(q) is not None})
    return [{"sequence": "no field files", "priority": 15, "values": [], "finding": (
        f"NO PER-LEVEL FIELD FILE AND NO INTERFACE FILE EXISTS, while "
        f"{len(hist)} residual history file(s) and {len(logs)} per-level run "
        f"log(s) do (levels {levels}). The coupling ran and nothing was "
        f"written at the points your task prescribes, so there is no result "
        f"to check yet. For EACH level and EACH side: evaluate that side's "
        f"converged field at the prescribed probe points (interpolate inside "
        f"the element, never nearest node) and write the per-level field "
        f"file; write the per-level interface file from that side's own "
        f"converged trace and flux at the prescribed interface points. Every "
        f"number you already have is on disk in your participants' per-level "
        f"dumps (field_level<k>.csv, interface_level<k>.csv; exports.json holds "
        f"only the LAST level); no re-solve is needed.")}]


def summary_names_findings(work: Path) -> list[dict]:
    """Every file the summary names must exist. A summary that lists files it
    never wrote reads as invented, whatever the numbers beside it say.

    Measured: one coupled run listed 21 deliverables in its summary -- six
    per-level field files and six interface files among them -- and had
    written none of the twelve; its three residual histories were real. The
    audit at hand-in said nothing about the names, so the run handed in a
    list. No task knowledge is used: the names come from the agent's own
    summary text, and existence is checked by basename anywhere under the
    working directory outside openPASO's scratch.
    """
    import re as _re
    out: list[dict] = []
    sf = _summary_file(work)
    if sf is None or not sf.is_file():
        return out
    try:
        text = sf.read_text(errors="ignore")
    except OSError:
        return out
    # A REPORT THAT SAYS THE WORK WAS NOT COMPLETED CLAIMS NO FILE. Measured: a run
    # whose honest could-not-complete report listed the files it had not made was told
    # its summary "names files that do not exist", and deleted the report twice.
    if _re.search(r"COULD[_ ]NOT[_ ]COMPLETE", text, _re.I):
        return out
    text = _re.sub(r"\b(?:https?|ftp)://\S+", " ", text)      # links are not files
    # A NAME IN A PLAN OR A PATTERN IS NOT A CLAIM. Measured: a summary's "what remains:
    # ... config.json" and the tail "_A.csv" of a pattern written field_level{k}_A.csv
    # were both listed as files the summary claimed and never wrote.
    _plan = _re.compile(r"\b(?:not (?:yet|written|produced|created|generated|saved|done)|never (?:written|"
                        r"produced|created)|missing|remains?|remaining|to ?do|would|will be|should be|"
                        r"planned|pending|could not|cannot|unable|failed to)\b", _re.I)
    names = set()
    for m in _re.finditer(r"[A-Za-z0-9_./-]+\.(?:csv|log|txt|json|vtu|vtk|pvd|yaml|yml|dat|npy|npz)\b", text):
        if m.start() > 0 and text[m.start() - 1] in "}>*]":
            continue                           # the tail of a pattern, not a file name
        _ls = text.rfind("\n", 0, m.start()) + 1
        _le = text.find("\n", m.end())
        if _plan.search(text[_ls:_le if _le >= 0 else len(text)]):
            continue                           # a line that says the file is not there, or not yet
        names.add(m.group(0).strip("`'\"(),;"))
    names = sorted(names)
    names = [n for n in names if n and n != sf.name and not n.startswith(("http", "www."))]
    if not names:
        return out
    present: set = set()
    for q in work.rglob("*"):
        if not q.is_file():
            continue
        try:
            if _SCRATCH & set(q.relative_to(work).parts[:-1]):
                continue
        except ValueError:
            continue
        present.add(q.name)
    missing = [n for n in names if Path(n).name not in present]
    if not missing:
        return out
    out.append({"sequence": "summary names", "priority": 10, "values": [],
                "finding": (
        f"YOUR SUMMARY FILE NAMES {len(missing)} FILE(S) THAT DO NOT EXIST "
        f"anywhere under your working directory: {', '.join(missing[:8])}"
        + (f" (+{len(missing) - 8} more)" if len(missing) > 8 else "")
        + ". A summary that lists files it never wrote is read as invented "
        "however real the rest of the work is. Write every file you name "
        "from the numbers you actually have -- or remove the name.")})
    return out


# THE LINES A SOLVER LIBRARY PRINTS ITSELF THAT THE EVIDENCE TABLE DOES NOT LIST. The served
# deal.II program prints deal.II's version and git revision, from the library's headers, through
# deal.II's own log stream (`DEAL::deal.II 9.8.0-pre, git revision <sha>`). With a direct solve
# it prints no other line of the library's, and the checks below called its console, passed
# through verbatim, "your own summary" 15 to 23 times per cell of one coupled round (measured).
LIBRARY_LINES = (r"^DEAL::deal\.II \d+\.\d+\.\d+[\w.+-]*, git revision [0-9a-f]*[ \t]*$",)


def solver_line_patterns() -> list:
    """The patterns of a line a solver emits itself: the evidence table's per-code
    signatures plus LIBRARY_LINES. Empty when the evidence table cannot be read."""
    try:
        from blind_eval.evidence import PER_CODE_SIGNATURES      # noqa: PLC0415
    except Exception:                                             # noqa: BLE001
        return []
    return [p for plist in PER_CODE_SIGNATURES.values() for p in plist] + list(LIBRARY_LINES)


def run_log_identity_findings(work: Path) -> list[dict]:
    """A per-level run log must carry the named solver's OWN console output.

    A side is credited to a code only when its log holds a line that code
    emits (a solver-iteration line, a banner with a number). A log of the
    agent's own summary -- `NDOF = 54`, `solve completed`, `max|u| = ...` --
    is prose, and a side whose log is prose cannot be credited to that code
    however right its numbers are. Measured on one real coupled run: a side
    labelled DUNE-fem that solved with scipy and wrote a three-line summary
    passed every other check here. Reads only the agent's own files.
    """
    out: list[dict] = []
    try:
        from blind_eval.evidence import strip_terminal_noise      # noqa: PLC0415
    except Exception:                                # signatures unavailable
        return out
    import re as _re
    pats = solver_line_patterns()
    if not pats:
        return out
    # Per-SIDE logs only (run_level<k>_<side>.log): on a coupled problem the task
    # asks each side's log to carry that code's own console output, because
    # that is what says WHICH code ran which subdomain. A single-code log is
    # judged by the evidence gate's canonical lines and is not charged here.
    def _copy_source(f: Path) -> str:
        """The driver's per-level console for this log's side, named as the file to copy."""
        try:
            _k = _level_of(f); _sd = _side_of(f)
            if _k and _sd:
                for cand in sorted(work.rglob(f"participant_output_level{_k}.log")):
                    if cand.parent.name.lower().endswith(_sd.lower()) or cand.parent.name.lower().endswith(f"_{_sd.lower()}"):
                        _t = strip_terminal_noise(cand.read_text(errors="replace"))
                        _stamped = re.search(r"^\[\d{4}-\d\d-\d\d[ T][\d:.]+\] \[(?:info|warning|debug|error)\]",
                                             _t, re.MULTILINE)
                        if any(re.search(pp, _t, re.IGNORECASE | re.MULTILINE) for pp in pats) or _stamped:
                            return (f" The coupling tool kept that side's console for level {_k} at {cand} ({len(_t)} bytes, with the "
                                    f"solver's own lines): copy that file over this one -- one command -- and keep its NDOF line.")
                        break
        except Exception:                                # noqa: BLE001
            return ""
        return ""

    def _console_lines(f: Path):
        """(path, solver lines) of the console the coupling tool kept for this log's side and
        level -- its stdout and stderr lines, without the driver's own header -- or (None, None)."""
        try:
            _k = _level_of(f); _sd = _side_of(f)
            if _k and _sd:
                for cand in sorted(work.rglob(f"participant_output_level{_k}.log")):
                    if cand.parent.name.lower().endswith(_sd.lower()):
                        _t = strip_terminal_noise(cand.read_text(errors="replace"))
                        return cand, [ln.strip() for ln in _t.splitlines() if ln.strip() and not _re.match(
                            r"^(?:iteration|command|returncode):|^--- (?:stdout|stderr) ---$", ln.strip())]
        except Exception:                                # noqa: BLE001
            pass
        return None, None

    for f in _level_logs(work):
        if not _side_of(f):
            continue
        try:
            text = strip_terminal_noise(f.read_text(errors="ignore"))
        except Exception:
            continue
        # A SOLVER'S OWN TIMESTAMPED LOG LINE IS A SOLVER LINE. The signature
        # table alone refused a real dolfinx console (its spdlog lines
        # `[2026-09-24 08:41:03.117] [info] ...`) three times in one correct
        # cell and once in another, while the copy-source branch below already
        # accepted exactly that shape. The log under judgement gets the same
        # rule as the candidate it would be told to copy.
        _stamped_log = _re.search(r"^\[\d{4}-\d\d-\d\d[ T][\d:.]+\] \[(?:info|warning|debug|error)\]",
                                  text, _re.MULTILINE)
        if _stamped_log or any(_re.search(p, text, _re.IGNORECASE | _re.MULTILINE)
                               for p in pats):
            # A BANNER STUB IS NOT A CONSOLE. Measured on a correct cell: six
            # side-A logs of ~500 bytes -- the solver's banner and the NDOF
            # line, the real consoles never written (stdout[:500]) -- passed
            # here on the banner alone. A capture cut short carries no step,
            # iteration or residual line; the whole console does.
            # A SHORT LOG IS A CUT CAPTURE ONLY WHERE THE WHOLE CONSOLE SAYS MORE. Measured: two
            # NGSolve consoles, whole, were told they were "what a capture cut short looks like";
            # that solver prints no step line. The driver's own console of that side and level
            # decides; with none on disk the log is not judged, and the note says so.
            _step = _re.search(r"(?im)^\s*(?:step|iter|time step|newton|finalised|residual|res-norm|solved?\b|converged)", text)
            if len(text) < 1200 and not _step:
                _cp, _cl = _console_lines(f)
                _have = {ln.strip() for ln in text.splitlines() if ln.strip()}
                _lack = [ln for ln in (_cl or []) if ln not in _have]
                if _cl is None:
                    out.append({"sequence": f"run log {f.name}", "priority": 90, "informational": True, "finding": (
                        f"{f.name}: carries no step, iteration or residual line ({len(text)} bytes), and no "
                        f"captured console of that side and level is on disk to tell a capture cut short "
                        f"from a solver that prints none; not judged.")})
                elif _lack:
                    out.append({"sequence": f"run log {f.name}", "priority": 58, "finding": (
                        f"{f.name}: THIS LOG IS A BANNER STUB ({len(text)} bytes): it carries no step, "
                        f"iteration or residual line, and the console the coupling tool captured for that "
                        f"side at that level ({_cp.relative_to(work)}) holds {len(_lack)} line(s) of solver "
                        f"output this log lacks. The task wants the console captured whole; copy that file "
                        f"over this one -- one command -- and keep its NDOF line.")})
            continue
        _src = _copy_source(f)
        out.append({"sequence": f"run log {f.name}", "finding": (
            f"{f.name}: THIS LOG CARRIES NO LINE ANY SOLVER EMITS "
            f"({len(text)} bytes of your own summary). The task wants that "
            "subdomain's solver console output, captured verbatim (its "
            "iteration lines, its banner), because that is what establishes "
            "WHICH code ran on that side; a side whose log is your own words "
            "cannot be credited to that code however right its numbers are. "
            "If you ran it through subprocess you already have the bytes: "
            "write result.stdout (and stderr) into this file, plus the NDOF "
            "line." + _src)})
    return out



_AXIS_SUFFIX = re.compile(r"^(?P<base>.+?)[ _\-]?(?P<ax>[xyz])$", re.I)


def _self_orders(view: dict) -> dict:
    """Median observed order of every self-difference sequence in `view`."""
    out = {}
    for label, seq in view.items():
        if not label.startswith("selfdiff_") or len(seq) < 2:
            continue
        try:
            orders = [math.log2(a / b) for a, b in zip(seq, seq[1:]) if b > 0 and a > 0]
        except (ValueError, ZeroDivisionError):
            continue
        if orders:
            out[label] = sorted(orders)[len(orders) // 2]
    return out


def _label_field_side(label: str) -> tuple:
    """('u', 'A') from 'selfdiff_solution_A_u|vector' and the like."""
    body = label[len("selfdiff_"):] if label.startswith("selfdiff_") else label
    tag, _, field = body.rpartition("_")
    side = tag.rpartition("_")[2] if "_" in tag else tag
    return field.replace("|vector", ""), side


def _second_order_peers(meds: dict, label: str) -> str:
    """The other fields on the same side whose own levels improve at second
    order (median order >= 1.7), named; '' when there are none."""
    _, side = _label_field_side(label)
    if not label.startswith("selfdiff_solution"):
        return ""
    peers = []
    for other, med in meds.items():
        if other == label or not other.startswith("selfdiff_solution"):
            continue
        f, s = _label_field_side(other)
        if s == side and med >= 1.7:
            peers.append(f)
    return ", ".join(sorted(peers))


def _peer_gap(meds: dict, label: str, med: float) -> float:
    """How far this field's order sits below the SLOWEST second-order peer on
    its side (0.0 when there is none). Measured on a wrong thermo-elastic run:
    the displacement of one side improved at 1.59 while its temperature, on
    the same mesh, improved at 1.99 -- a released constraint (a Dirichlet
    entry that freed u_z on the interface nodes) made the tractions first
    order -- and the 1.45 ceiling of the low-order branch let it pass in
    silence. The correct cells of the same family sit at 1.95-2.08 on every
    field of both sides."""
    _, side = _label_field_side(label)
    if not label.startswith("selfdiff_solution"):
        return 0.0
    peers = [m for other, m in meds.items()
             if other != label and other.startswith("selfdiff_solution")
             and _label_field_side(other)[1] == side and m >= 1.7]
    return max(0.0, min(peers) - med) if peers else 0.0


def _vector_order_view(seqs: dict) -> dict:
    """Replace the components of one vector field by the vector itself.

    AN ORDER IS A PROPERTY OF A FIELD IN A NORM, NOT OF ONE CARTESIAN
    COMPONENT. The interface exclusion directly below already learned half of
    this lesson; the other half is the minor component of a vector. On the first
    correct vector-valued coupled run, at order ~1.9 -- side
    A's ux self-difference improves at 1.92 and its uy, seven times smaller and
    sitting near the coupling iteration's own floor, improves at 0.96. The
    check fired on that component and told a correct agent its exchanged datum
    was O(h) and its coupling first-order. The vector formed from both
    components improves at 1.88, which is what the grader measures and what is
    true. Every scalar problem has one component per side, so its view is
    unchanged -- which is why this never showed on the 26 scalar CORRECT cells.

    Components join only when their own column names say so: identical after a
    trailing x/y/z is removed, with at least two distinct axes present. A field
    with no axis suffix always stands alone, so a temperature shipped beside a
    displacement (`T, ux, uy`, 43 files here) is never folded into it and can
    never be masked by it.
    """
    fam: dict = {}
    for label, seq in seqs.items():
        if not label.startswith("selfdiff_"):
            continue
        tag, _, field = label[len("selfdiff_"):].rpartition("_")
        if not tag:
            continue
        m = _AXIS_SUFFIX.match(field)
        if not m or not m.group("base"):
            continue
        fam.setdefault((tag, m.group("base")), {})[m.group("ax").lower()] = (
            label, seq)
    view = dict(seqs)
    for (tag, base), axes in fam.items():
        if len(axes) < 2:
            continue                      # one axis is not a vector
        members = [axes[a] for a in sorted(axes)]
        n = min(len(s) for _, s in members)
        if n < 2:
            continue
        for lab, _ in members:
            view.pop(lab, None)
        view[f"selfdiff_{tag}_{base}|vector"] = [
            math.sqrt(sum(s[i] ** 2 for _, s in members)) for i in range(n)]
    return view


def resolve_under_cell(raw) -> Path:
    """A relative path from a CALLER means a path in the CALLER's directory.

    THE MCP SERVER IS A SEPARATE PROCESS WITH ITS OWN WORKING DIRECTORY, which
    the caller can neither see nor control. `Path(".")` therefore resolved to
    the SERVER's cwd, and the tools that took a work_dir then reported, in
    perfect good faith, that the caller's files did not exist.

    MEASURED 2026-09-19 across the ten most recent coupled runs, four separate
    events in three tools:

        audit_results(work_dir='.')          -> "sequences_found": 0 and
                                                "YOUR SUMMARY FILE IS MISSING
                                                OR EMPTY" (RESULT.txt was on
                                                disk, 594 bytes)
        verify_interface_flux(...)           -> "REFUSED: no interface file
                                                could be read"
        verify_pde_consistency(...)          -> "solution_level1_A.csv does not
                                                exist" (it existed, 106 KB,
                                                written a minute earlier)

    The same directory by ABSOLUTE path returns 12 sequences and the real
    findings. One run spent its last three calls, with eleven minutes left,
    checking whether it had lost its own results; another lost 21% of its call
    budget to it. `couple` has refused a relative work_dir since the same
    defect bit it, with the same reason written above its check; these three
    doors never got it.
    """
    import os as _os
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    cell = _os.environ.get("OPENPASO_CELL_WORKDIR")
    return (Path(cell) / path) if cell else path


def _guarded(name: str, fn, work) -> list:
    """Run one audit check; a check that raises becomes ONE finding that names
    itself, and every other check still speaks. Measured: a TypeError in the
    exports.json fallback of one check escaped audit() and the agent read
    "[auto-audit unavailable: TypeError]" -- twelve findings lost, two of
    which named exactly its defects."""
    try:
        return list(fn(work) or [])
    except Exception as exc:                                  # noqa: BLE001
        return [{"sequence": f"audit check {name}", "values": [],
                 "priority": 90, "informational": True,
                 "finding": (f"AUDIT CHECK {name} COULD NOT RUN on your files "
                             f"({type(exc).__name__}: {str(exc)[:160]}); every "
                             f"other check below still did. This is a defect in "
                             f"the check, not a verdict on your result.")}]


def audit(work_dir: str, claimed_order: float | None = None,
          summary_path: str | None = None) -> dict:
    """The three questions, answered from the agent's own files.

    `summary_path` is the caller's hint for the agent's summary/answer file
    (a harness knows which file it just saw written); without it the file is
    discovered by name (result / summary / report / answer)."""
    work = resolve_under_cell(work_dir)
    if summary_path:
        _SUMMARY_HINT[str(work)] = summary_path
    findings: list[dict] = []
    findings.extend(_guarded("residual_findings", residual_findings, work))
    findings.extend(_guarded("diverged_exports_findings", diverged_exports_findings, work))
    findings.extend(_guarded("unsolved_field_findings", unsolved_field_findings, work))
    findings.extend(_guarded("nonfinite_field_findings", nonfinite_field_findings, work))
    findings.extend(_guarded("nonfinite_interface_findings", nonfinite_interface_findings, work))
    findings.extend(_guarded("own_field_flux_findings", own_field_flux_findings, work))
    findings.extend(_guarded("exports_not_the_field_findings", exports_not_the_field_findings, work))
    findings.extend(_guarded("completeness_findings", completeness_findings, work))
    findings.extend(_guarded("exact_ladder_findings", exact_ladder_findings, work))   # the provable one leads the band one
    findings.extend(_guarded("ndof_ladder_findings", ndof_ladder_findings, work))
    findings.extend(_guarded("interface_ends_findings", interface_ends_findings, work))
    findings.extend(_guarded("interface_point_set_findings", interface_point_set_findings, work))
    findings.extend(_guarded("probe_sampling_findings", probe_sampling_findings, work))
    findings.extend(_guarded("unlaunched_participants_findings", unlaunched_participants_findings, work))
    findings.extend(_guarded("run_log_identity_findings", run_log_identity_findings, work))
    findings.extend(_guarded("summary_names_findings", summary_names_findings, work))
    findings.extend(_guarded("missing_fields_findings", missing_fields_findings, work))
    seqs = _sequences_from_workdir(work)
    _mag_src: dict = {}                 # the finest-level file each magnitude was read from
    csvs = _sequences_from_level_csvs(work, sources=_mag_src)
    if "__ambiguous__" in csvs:
        # KEEP WHAT IS ALREADY KNOWN. This rebuilt the findings list from
        # scratch, so a residual history that had already been read and found
        # broken was dropped the moment two files collided on one per-level
        # slot. The two are independent — the residual check needs no per-level
        # field files at all, and the ambiguity is about which field file to
        # read — so reporting only the ambiguity told the agent to tidy its
        # filenames while saying nothing about a coupling that never converged.
        # THE INTERFACE FINDING SURVIVES AN AMBIGUOUS FIELD SET, for the
        # same reason the residual one does: it reads interface files, not
        # the per-level field slot that collided. So do the identity /
        # sampling / continuity checks — each reads its own files, none of
        # them the collided per-level field slot.
        # GUARDED HERE AS IN THE NORMAL BRANCH: called directly, one check that raised on a malformed
        # interface or log file ended audit() before the ambiguity finding and every later check.
        findings.extend(_guarded("interface_sign_findings", interface_sign_findings, work))
        findings.extend(_guarded("deliverable_flux_vs_export_findings", deliverable_flux_vs_export_findings, work))
        findings.extend(_guarded("outer_boundary_findings", outer_boundary_findings, work))
        findings.extend(_guarded("unparsed_level_files_findings", unparsed_level_files_findings, work))
        findings.extend(_guarded("solver_stopped_findings", solver_stopped_findings, work))
        findings.extend(_guarded("febio_deck_findings", febio_deck_findings, work))
        findings.extend(_guarded("interface_continuity_findings", interface_continuity_findings, work))
        findings.extend(_guarded("identical_solution_levels_findings", identical_solution_levels_findings, work))
        findings.extend(_guarded("wrong_level_run_log_findings", wrong_level_run_log_findings, work))
        findings.extend(_guarded("solution_rows_grow_findings", solution_rows_grow_findings, work))
        findings.extend(_guarded("data_compatibility_findings", data_compatibility_findings, work))
        findings.extend(_guarded("equation_findings", equation_findings, work))
        findings = findings + [
            {"sequence": "level files", "values": [],
             "finding": (
                 "AMBIGUOUS INPUT: more than one file matches "
                 "the per-level pattern at the top level (" +
                 ", ".join(csvs["__ambiguous__"][:4]) +
                 "). I will not guess which is your answer — "
                 "name your per-level files uniquely, or "
                 "remove the stale ones, and re-run this "
                 "check.")}]
        return {"sequences_found": 0, "clean": False,
                "what_to_fix_next": what_to_fix_next(findings),
                "findings": findings,
                "note": ("the per-level field check did not run: input was "
                         "ambiguous. Any other finding above DID run and "
                         "stands.")}
    seqs.update(csvs)
    # near-zero field: the loads may never have been applied at all
    _uniform_seen: set = set()
    for label, seq in list(seqs.items()):
        # "< 1e-8" must INCLUDE exact zero — the three 4C runs that wired
        # VAL: [0.0], FUNCT: [0] delivered fields of literal 0.0 everywhere,
        # and "0 < x" excluded precisely them.
        # ONE VALUE EVERYWHERE IS NOT A FIELD. Measured on a run whose 4C
        # side delivered a temperature of 8.5e13 at every probe point of every
        # level (a scalar-transport deck with no Dirichlet condition the field
        # could read): the audit led with a run-log format defect and never
        # mentioned the column, because nothing here asked whether a field
        # VARIES. The near-zero check below is this check's mirror image.
        if label.startswith("spread_") and seq and seq[0] < 1e-9:
            _tag, _, _f = label[len("spread_"):].rpartition("_")
            _mag = (seqs.get(f"magnitude_{_tag}_{_f}") or [0.0])[0]
            _key = (_f, f"{_mag:.6e}")
            if _key in _uniform_seen:
                continue                 # the same column read through another file family
            _uniform_seen.add(_key)
            _uf = _uniform_finding(work, label, seq, _mag_src, _f, _tag, _mag)
            if _uf:
                findings.append(_uf)
            continue
        if label.startswith("magnitude_") and seq and seq[0] < 1e-8:
            _nz = _near_zero_finding(work, label, seq, _mag_src)
            if _nz:
                findings.append(_nz)
    # A vector is one field; see _vector_order_view for why and for the
    # cell that proved it.
    _view = _vector_order_view(seqs)
    _meds = _self_orders(_view)
    for label, seq in _view.items():
        if label.startswith("magnitude_"):
            continue
        if len(seq) < 3 and not label.startswith("selfdiff_"):
            continue
        if len(seq) < 2:
            continue
        entry = {"sequence": label, "values": seq}
        # 1. floor detection FIRST — a flat sequence is precisely the case the
        # old monotonicity filter threw away before this check could see it,
        # which is why runs with errors flat at ~6e-7 sailed through.
        rel = [abs(a - b) / max(abs(a), 1e-300) for a, b in zip(seq, seq[1:])]
        if all(r < 0.05 for r in rel) and label.startswith("selfdiff_"):
            # A FLAT SELF-DIFFERENCE IS NOT A FLOOR, IT IS ORDER ZERO. The
            # FLOOR wording below is written for an error against a reference
            # that stops falling; on a sequence of level-to-level CHANGES it
            # said "refinement is changing nothing", and a run read that as
            # "converged" and handed in a traction whose change from level 1
            # to 2 (0.293) equalled its change from 2 to 3 (0.282).
            _fld, _sd = _label_field_side(label)
            entry["finding"] = (
                f"THE CHANGE BETWEEN YOUR LEVELS DOES NOT SHRINK for {_fld}"
                f"{' on side ' + _sd if _sd else ''}: {seq[0]:.3e} from one "
                f"level to the next, then {seq[-1]:.3e} -- within 5% of each "
                f"other. A converging quantity at least halves that change per "
                f"refinement; this one is at order ~0, so it is NOT converged "
                f"and a study that says otherwise is a wrong claim. Find what "
                f"drives it before refining again: for an interface quantity, "
                f"the recovery and the constraints on the interface nodes (a "
                f"released dof, a missing term); for a field, its equation or "
                f"the interface datum it receives.")
            findings.append(entry)
            continue
        if all(r < 0.05 for r in rel):
            entry["finding"] = (
                "FLOOR: the levels are within 5% of each other, so refinement "
                "is changing nothing. Whatever limits this number, it is not "
                "the mesh — check solver tolerances (a nonlinear/iterative "
                "solver left at its default stops around 1e-6-1e-7 and that "
                "floor becomes your 'error'), or a fixed post-processing step.")
            findings.append(entry)
            continue
        drops = sum(1 for a, b in zip(seq, seq[1:]) if b < a)
        if drops < len(seq) - 2:      # mostly non-decreasing: not error-like
            continue
        # 2/3. observed order vs the claim
        try:
            orders = [math.log2(a / b) for a, b in zip(seq, seq[1:]) if b > 0]
        except ValueError:
            continue
        if not orders:
            continue                     # nothing comparable; not a finding
        entry["observed_orders"] = [round(o, 2) for o in orders]
        med = sorted(orders)[len(orders) // 2]
        # THE CLAIMED ORDER IS THE FIELD'S, NOT THE INTERFACE TRACTION'S.
        #
        # Grouping by (kind, side) gave the audit interface_* sequences for the
        # first time, and the order check then compared them against the order
        # claimed in RESULT.txt. Those are different quantities: an interface
        # node that sits at the end of the interface has a one-sided boundary
        # weight and converges at order 1, measured 1.000/1.013/1.010 against a
        # known exact flux while the interior runs at 1.99. So a perfectly good
        # coupling shows interface self-differences improving at ~0.9-1.4.
        # It cost exactly one false alarm, and it was on the single coupled
        # run anyone had, at that point, verified correct against an
        # independent reference.
        _is_iface = label.startswith("selfdiff_interface") or \
            label.startswith("magnitude_interface")
        if _is_iface:
            continue
        # A FIELD THAT LAGS ITS SECOND-ORDER PEER BY 0.3 ON THE SAME MESH IS
        # NAMED EVEN ABOVE 1.45: see _peer_gap.
        _gap = _peer_gap(_meds, label, med)
        _coupled_low = ((0.5 <= med <= 1.45 or (0.5 <= med < 1.7 and _gap >= 0.3))
                        and bool(_interface_files(work, sided=True)))
        # A RECOVERY DEFECT DEGRADES EVERY FIELD OF A SIDE ALIKE. Measured on a
        # wrong thermo-mechanical run: its temperature self-converged at 1.95
        # on both sides while its displacement improved at 1.2-1.6, and this
        # branch named the interface recovery -- the agent handed in "first-
        # order recovery, a known limitation" over a flipped sign in one term
        # of its own weak form. When another field on the same side improves
        # at second order, the recovery is not the cause; the field that lags
        # is driven by something wrong in ITS equation or ITS exchange, and
        # that is what is said.
        _peers = _second_order_peers(_meds, label)
        _sampled = [f for f in findings if str(f.get("sequence", "")).startswith("probe sampling")]
        if (claimed_order is None and _coupled_low
                and not label.startswith("magnitude_") and _sampled):
            _fld, _sd = _label_field_side(label)
            entry["finding"] = (
                f"YOUR OWN LEVELS IMPROVE AT ONLY ~{med:.2f} FOR {_fld} ON SIDE {_sd} -- "
                f"and the probe files were made by nearest-node sampling (see the "
                f"NEAREST-NODE SAMPLING finding), which is O(h) on every field of "
                f"every side whatever the solve did. Fix the sampling first; the "
                f"order you measure after that is the one to judge.")
            findings.append(entry)
            continue
        if (claimed_order is None and _coupled_low
                and not label.startswith("magnitude_") and _peers):
            _fld, _sd = _label_field_side(label)
            entry["priority"] = 62
            entry["finding"] = (
                f"YOUR OWN LEVELS IMPROVE AT ONLY ~{med:.2f} FOR {_fld} ON "
                f"SIDE {_sd}, WHILE {_peers} ON THE SAME SIDE IMPROVE{'S' if ',' not in _peers else ''} AT "
                f"SECOND ORDER (~{med + _gap:.2f}). A first-order interface recovery or a coarse-mesh "
                f"discretisation error would degrade every coupled field of a side "
                f"alike, so neither is the cause here: the defect sits in what "
                f"drives {_fld} alone -- its source term, its material law, a "
                f"constraint released on some of its nodes (a Dirichlet entry whose "
                f"toggle frees a dof that another entry pinned; on a plane-strain "
                f"slab, u_z at the interface), or the "
                f"interface datum it receives (a sign, a scaling, a missing term "
                f"such as the thermal stress in a traction). Take ONE level and "
                f"compare what one side EXPORTS for that exchange against what "
                f"the partner APPLIES, point by point, before touching the "
                f"recovery. A field that does not converge is a wrong answer at "
                f"every level; do not hand it in as a known limitation.")
            findings.append(entry)
            continue
        if (claimed_order is None and _coupled_low
                and not label.startswith("magnitude_")):
            entry["finding"] = (
                f"YOUR OWN LEVELS IMPROVE AT ONLY ~{med:.2f}. On a COUPLED "
                "run whose coupling converged, several defects give an order "
                "near 1, and the order alone does not tell them apart. Among "
                "them: an exchanged datum recovered to first order only (the boundary "
                "trace of a P1 element gradient, a two-point nearest-node "
                "difference; the consistent residual q = -(A u - b_vol)/w on the "
                "interface rows of YOUR OWN system is second order), or a field "
                "converging cleanly to the solution of a different problem (an "
                "operator, source or boundary condition that is not your "
                "task's; a boundary not held where the task holds it is one). "
                "The equation check on each side's own config separates a wrong "
                "operator or source, and the outer-boundary check a boundary: "
                "read their verdicts before changing the recovery.")
            findings.append(entry)
        elif claimed_order is not None and med < claimed_order - 0.4:
            _msg = (
                f"ORDER MISMATCH: you are about to claim order "
                f"{claimed_order:g} but your own levels improve at ~{med:.2f}. "
                f"Common causes: an element degree lower than the task states; "
                f"a first-order time integrator behind a spatial study; "
                f"volumetric locking (near-incompressible material solved with "
                f"a pure displacement form — if your notes say 'mixed "
                f"formulation', check the assembled form actually uses it); "
                f"a low-order quadrature or projection in post-processing.")
            # THE COUPLED CAUSE WAS MISSING FROM THIS LIST. Measured three
            # times in one development day: couplings with converged
            # three-level evidence read as order 0.85, 0.96 and 0.97 against a
            # theoretical 2, each from a DIFFERENT first-order interface
            # recovery (a two-point nearest-node difference; a P1 element
            # gradient evaluated ON the boundary). Pattern checks cannot
            # enumerate the variants; the ORDER ITSELF is the signature, and
            # this is the one message every such run reads before delivering.
            if 0.5 <= med <= 1.45 and _interface_files(work, sided=True) and _peers:
                _fld, _sd = _label_field_side(label)
                _msg += (
                    f" On this COUPLED run {_fld} on side {_sd} improves at "
                    f"~{med:.2f} while {_peers} on the same side improves at "
                    f"second order, so the interface recovery is NOT the cause "
                    f"(it would degrade every field alike): look at what drives "
                    f"{_fld} alone -- its source term, its material law, the "
                    f"interface datum it receives (a sign, a scaling, a missing "
                    f"term).")
            elif 0.5 <= med <= 1.45 and _interface_files(work, sided=True):
                _msg += (
                    " On a COUPLED run whose coupling converged, two different "
                    "defects give an order near 1, and the order alone does not "
                    "tell them apart: an exchanged datum recovered to first "
                    "order only (the boundary trace of a P1 element gradient, a "
                    "two-point nearest-node difference; the consistent residual "
                    "q = -(A u - b_vol)/w on YOUR OWN interface rows is second "
                    "order), or a field converging to the solution of a "
                    "different problem. The equation check on each side's own "
                    "config separates the second.")
            entry["finding"] = _msg
            findings.append(entry)
        elif any(o < -0.1 for o in orders):
            entry["finding"] = (
                "NON-MONOTONE: at least one refinement made the answer WORSE. "
                "A converging study never does that outside noise; check for a "
                "mesh-dependent bug (wrong BC on the finer mesh, probe points "
                "outside the domain, a tolerance floor).")
            findings.append(entry)
    # THE RESOLVED PATH, LIKE EVERY OTHER CHECK. Measured: two cells called this
    # with work_dir='.' and were told "YOUR SUMMARY FILE IS MISSING OR EMPTY"
    # with RESULT.txt on disk; one burned four of its last calls on it.
    findings.extend(contract_findings(work))
    findings.extend(_guarded("interface_sign_findings", interface_sign_findings, work))
    findings.extend(_guarded("zero_channel_findings", zero_channel_findings, work))
    findings.extend(_guarded("deliverable_flux_vs_export_findings", deliverable_flux_vs_export_findings, work))
    findings.extend(_guarded("outer_boundary_findings", outer_boundary_findings, work))
    findings.extend(_guarded("free_interface_end_findings", free_interface_end_findings, work))
    findings.extend(_guarded("unparsed_level_files_findings", unparsed_level_files_findings, work))
    findings.extend(_guarded("solver_stopped_findings", solver_stopped_findings, work))
    findings.extend(_guarded("febio_deck_findings", febio_deck_findings, work))
    findings.extend(_guarded("export_findings", export_findings, work))
    findings.extend(_guarded("interface_continuity_findings", interface_continuity_findings, work))
    findings.extend(_guarded("identical_solution_levels_findings", identical_solution_levels_findings, work))
    findings.extend(_guarded("wrong_level_run_log_findings", wrong_level_run_log_findings, work))
    findings.extend(_guarded("solution_rows_grow_findings", solution_rows_grow_findings, work))
    # NOT VOLUNTARY: see equation_findings. The one check that separates a field
    # converging to the right function from one converging to a wrong one was
    # called in 1 of 1118 recorded coupled runs, so it is computed here from the
    # agent's own files and reported whether or not it was asked for.
    findings.extend(_guarded("data_compatibility_findings", data_compatibility_findings, work))
    findings.extend(_guarded("equation_findings", equation_findings, work))
    clean = not [f for f in findings if not f.get("informational")]
    # WHAT DID NOT RUN, SAID BESIDE "clean". A clean audit whose equation check
    # never ran was reported as "no findings -- self-consistent", the same words
    # a fully checked one gets.
    not_run = [f"{f['sequence']}: {f['finding']}" for f in findings
               if f.get("informational") and ("NOT CHECKED" in str(f.get("finding"))
                                               or "COULD NOT RUN" in str(f.get("finding")))]
    # THE LADDER: the next unmet step of a coupled run, from the files. It
    # leads a clean reply and closes a dirty one, so the agent always knows
    # the one thing to do next.
    try:
        _ladder = coupled_ladder(work)
    except Exception:                                   # noqa: BLE001
        _ladder = None
    # NO SUMMARY OVER A DEFECT OF THE RESULT. Measured: a hand-in audit led with a
    # located finding on a side's field and closed with the ladder's "write the
    # summary"; the run wrote the summary.
    _severe = sorted((int(f.get("priority", 99)), str(f.get("finding", ""))) for f in findings
                     if not f.get("informational") and int(f.get("priority", 99)) <= 5)
    if _severe and _ladder and _ladder.get("step") == 6:
        _ladder = dict(_ladder, text=(
            "LADDER STEP 6 OF 6 -- NOT YET: the finding that leads this reply is a defect of the "
            f"result itself (priority {_severe[0][0]}), and no summary is written over it. Fix the side "
            "it names, couple the levels it touches again, and run audit_results(work_dir) once more."))
    _lead = what_to_fix_next(findings, clean_msg=(_ladder["text"] if _ladder else None))
    if _ladder and _ladder["text"] not in _lead:
        _lead += "\n" + _ladder["text"]
    return {
        "sequences_found": len(seqs),
        "next_step": (_ladder or {}).get("text"),
        # THE SINGLE NEXT FIX, FIRST. A weak agent reads the top of the reply,
        # so the highest-priority finding leads with its full corrective text;
        # a clean funnel leads with the next ladder step.
        "what_to_fix_next": _lead,
        "findings": findings,
        "clean": clean,
        "not_run": not_run,
        "note": ("This audit uses ONLY your own files — no reference "
                 "solution. 'clean' means self-consistent, not correct."),
    }


# ── the proof that is owed NOW, not at hand-in ─────────────────────────────
#
# MEASURED, and it is the largest recoverable loss in the record. Of 516
# graded coupled cells, 112 were malformed and 92 of those carry PROVEN
# execution evidence: the runs happened. 34 of them reached three coupled
# levels with both codes proven, the coupling proven and the interface
# satisfied, and scored nothing. The single commonest reason is one line --
# no captured run log carrying `NDOF = <n>` for some (level, side).
#
# The check for that already exists and already runs. It runs when the
# submission is written, and by then the median cell has TWO tool calls left;
# one in five has none. It names work costing five to fifteen actions. The
# same observation was made once before from file mtimes -- five of six runs
# wrote their summary at 93-99% of their file-activity span -- and two checks
# were moved early in response. These two were not, and they are the two that
# decide the malformed bucket.
#
# So this selects the subset that is ALREADY OWED for the levels the agent has
# actually coupled, to be delivered as each level finishes. What it must not
# do is ask for work that is not due yet: `contract_findings` at level 1
# reports ONLY 1 LEVEL(S) DELIVERED, which is true at hand-in and pure noise
# to an agent who is on level 1 of three (measured on seeds 5622 and 7262).
# A gate that names an absent defect costs an action for nothing.

_DUE_PER_LEVEL = ("run-log contract", "coupling evidence", "discretisation size")


def deliverable_proof_due(work: Path, levels_done) -> list[dict]:
    """Findings already owed for the levels that have actually been coupled.

    Self-consistency only: every one of these reads the agent's own files and
    compares them against each other. Nothing here knows what any task asked
    for.
    """
    try:
        lv = {int(k) for k in (levels_done or [])}
    except (TypeError, ValueError):
        return []
    if not lv:
        return []
    out: list[dict] = []
    try:
        out += [f for f in contract_findings(work, only_levels=lv)
                if f.get("sequence") in _DUE_PER_LEVEL]
    except Exception:                                    # noqa: BLE001
        pass
    try:
        out += completeness_findings(work, only_levels=lv, skip_participant_dirs=True)
    except Exception:                                    # noqa: BLE001
        pass
    # run_log_identity_findings IS DELIBERATELY NOT HERE, and it was in the
    # first draft. Measured against the recorded runs it spoke on more than a
    # quarter of the correct ones -- runs whose logs are 76 to 242 bytes of the
    # agent's own words and which an independent check accepted anyway. A
    # mid-run surface that interrupts that much work that goes on to be right is not
    # precise enough to be worth an action. It stays in the hand-in audit,
    # where it costs less and where it belongs.
    return out


# ── what your own files already say about the ANSWER, per level ────────────
#
# The deliverable proof above asks whether the run can be shown to have
# happened. This asks whether what it produced is worth handing in, and it is
# the family that decides a correct run against a completed but unphysical one.
#
# MEASURED on a live run. It coupled three levels with both codes
# proven, wrote every deliverable, and came out completed but unphysical at order
# 0.16. Its own files said so three times over -- the field peaks below 1e-8,
# the levels sit within 5% of each other so refinement changes nothing, and
# the reported interface residual of 3.50e-15 is contradicted by its own two
# interface files, which differ by 127% of their own scale. Every one of those
# findings existed. NEAR-ZERO FIELD and FLOOR appear ZERO times in its
# trajectory, because they live only inside audit(), and audit() ran when the
# submission was written -- with two tool calls left.
#
# Same bodies, asked per level while the ladder is still running, and scoped
# to levels that have actually been coupled.

_MAG = "magnitude_"


def field_quality_due(work: Path, levels_done, converged: bool | None = None) -> list[dict]:
    """Answer-quality findings already visible in the agent's own files.

    Self-consistency only: a field measured against itself across levels, and
    the two sides' interface files measured against each other. `converged`, where
    the caller knows it, is whether the last coupling converged: the files a stalled
    coupling leaves are not read as a held trace (see _imported_trace_not_held).
    """
    try:
        lv = {int(k) for k in (levels_done or [])}
    except (TypeError, ValueError):
        return []
    if not lv:
        return []
    out: list[dict] = []
    try:
        seqs = dict(_sequences_from_workdir(work))
        csvs = _sequences_from_level_csvs(work)
        if "__ambiguous__" not in csvs:
            seqs.update(csvs)
    except Exception:                                    # noqa: BLE001
        seqs = {}
    for label, seq in seqs.items():
        # SOLUTION FIELDS ONLY. An interface TRACE may legitimately be zero --
        # a Dirichlet seam held at zero, a flux that genuinely vanishes -- and
        # two cells that went on to grade CORRECT carry exactly that (an
        # interface u of 0 and an interface q of 1.9e-16). A solution field
        # that peaks below 1e-8 is a solve that produced nothing, and that is
        # never right.
        if not label.startswith(_MAG + "solution") or not seq:
            continue
        if seq[0] < 1e-8:
            out.append({"sequence": label, "values": seq, "finding": (
                "NEAR-ZERO FIELD: this field peaks below 1e-8. This measures "
                "the field, not why it is small: a load that never entered the "
                "assembled system, boundary values that are all zero, and a "
                "truly small answer all give it. Check the assembled load "
                "vector and the boundary values now: a level that solves "
                "nothing costs the whole ladder built on top of it.")})
    # FLOOR needs two levels to mean anything, and only the self-difference
    # sequences carry it -- a magnitude that barely moves is often correct.
    for label, seq in _vector_order_view(seqs).items():
        if not label.startswith("selfdiff_") or len(seq) < 2:
            continue
        rel = [abs(a - b) / max(abs(a), 1e-300) for a, b in zip(seq, seq[1:])]
        if rel and all(r < 0.05 for r in rel):
            out.append({"sequence": label, "values": seq, "finding": (
                "FLOOR: successive levels are within 5% of each other, so "
                "refinement is changing nothing. Whatever limits this number "
                "it is not the mesh -- an iterative solver left at its default "
                "stops around 1e-6 to 1e-7 and that floor becomes your "
                "'error'. Refining further buys nothing until it moves.")})
            break                                        # one is the message
    # AND THE LADDER THAT IS GOING BACKWARDS. A refinement that makes the
    # answer worse is never noise at this size, and it is the opposite failure
    # to FLOOR: measured on a recorded cell whose self-differences grew about
    # fivefold per level while its interface residual read 1e-15 and every
    # other check reported clean.
    for label, seq in _vector_order_view(seqs).items():
        if not label.startswith("selfdiff_solution") or len(seq) < 2:
            continue
        if all(b > a * 1.5 for a, b in zip(seq, seq[1:])):
            out.append({"sequence": label, "values": seq, "finding": (
                "YOUR LADDER IS GOING BACKWARDS: each refinement made this "
                "quantity WORSE, not better. A converging study never does "
                "that outside noise, and a growing self-difference means the "
                "finer mesh is solving a different problem -- a boundary "
                "condition that moves with the mesh, an exchanged quantity "
                "that scales with the node count, or probe points outside the "
                "domain. Nothing built on top of this level will fix it.")})
            break
    # THE REPORTED-RESIDUAL-VS-FILES CHECK IS DELIBERATELY NOT HERE. It fires
    # on a run that went on to be correct (one case: the two interface
    # files disagree at level 3 and the result was right anyway), so it is not
    # precise enough to interrupt a run with. It stays in the hand-in audit.
    try:
        out += [f for f in (interface_continuity_findings(work) or [])
                if any(f" level {k}" in str(f.get("sequence", "")) for k in lv)]
    except Exception:                                    # noqa: BLE001
        pass
    try:
        out += nonfinite_field_findings(work, levels=lv)
    except Exception:                                    # noqa: BLE001
        pass
    try:
        out += nonfinite_interface_findings(work, levels=lv)
    except Exception:                                    # noqa: BLE001
        pass
    try:
        out += [f for f in own_field_flux_findings(work, levels=lv) if not f.get("informational")]
    except Exception:                                    # noqa: BLE001
        pass
    try:
        out += exports_not_the_field_findings(work, levels=lv)
    except Exception:                                    # noqa: BLE001
        pass
    try:
        out += free_interface_end_findings(work, levels=lv)
    except Exception:                                    # noqa: BLE001
        pass
    out += _interface_transmitted_nothing(work, lv)
    out += _imported_trace_not_held(work, converged=converged)
    out += _side_exported_nothing(work)
    return out


def _side_exported_nothing(work: Path) -> list[dict]:
    """One side's exports carry no field and no flux: all values exactly zero.

    THE UPSTREAM SUPPLY OF THE DEAD EXCHANGE. `_interface_transmitted_nothing`
    catches the pair once BOTH sides are silent and the per-level interface
    files exist; by then the iteration has already converged on nothing and the
    ladder is built on it. This reads one side's own exports.json, which exists
    the moment a participant runs once -- before any coupling, before any
    deliverable -- and says so while the run is still one side long.

    THE SERVED SELF-CHECK CANNOT SEE THIS. Its zero guard
    (coupling_knowledge.py, EXPORT SELF-CHECK) fires only for a Neumann side
    whose recovered flux is ~0 against a NONZERO imported flux. A side handed
    zeros -- the first coupling iteration, or a partner that is itself silent --
    skips that branch entirely and exports nothing without a word. Measured in
    single-side trials with the contract in hand and nothing to orchestrate:
    2 of 7 successful-looking exports were identically zero.

    Measured over the recorded runs: every run it flags had stopped short of a
    gradeable number, and it fires on none that produced one, correct or not.
    It fires on nothing that ever produced a gradeable number.

    Judged only where the side exported something at all: an absent or empty
    exports.json is a different defect with its own finding, and a side that
    exports values but no fluxes (or the reverse) is judged on what it wrote.
    """
    out: list[dict] = []
    try:
        import numpy as _np
    except Exception:                                    # noqa: BLE001
        return out
    for ep in sorted(work.rglob("exports.json")):
        try:
            e = json.loads(ep.read_text() or "{}")
        except (OSError, ValueError):
            continue
        if not isinstance(e, dict):
            continue
        v = _np.asarray(e.get("values") or [], float).ravel()
        q = _np.asarray(e.get("normal_fluxes") or [], float).ravel()
        if v.size == 0 and q.size == 0:
            continue                      # exported nothing at all: not this
        vmax = float(abs(v).max()) if v.size else 0.0
        qmax = float(abs(q).max()) if q.size else 0.0
        if vmax != 0.0 or qmax != 0.0:
            continue
        side = ep.parent.name or str(ep.parent)
        out.append({"sequence": f"side exported nothing {side}",
                    "values": [vmax, qmax], "finding": (
            f"{side} EXPORTED A FIELD AND A FLUX THAT ARE IDENTICALLY ZERO "
            f"({ep}). Every value and every flux is exactly 0.0, so this side "
            f"handed its partner nothing. On a driven problem that is the "
            f"answer a solve returns when the load never entered the assembled "
            f"system -- not a small number, the no-load answer. The export "
            f"self-check in the served contract cannot see it: its zero guard "
            f"only fires against a NONZERO imported flux, and a side handed "
            f"zeros skips that branch. Two sides in this state agree perfectly, "
            f"so the iteration converges at once and every self-consistency "
            f"number the run reports about itself reads as success. Check that "
            f"your source term and your boundary data actually reach the "
            f"assembled system before coupling on this.")})
        break                              # one side is the message
    return out


def _script_role(q: Path) -> str | None:
    """The role one participant script states: its SIDE line, else a contract's first
    lines "(NEUMANN side)" / "(DIRICHLET side)". None when it states neither."""
    try:
        txt = q.read_text(errors="ignore")
    except OSError:
        return None
    m = re.findall(r"^SIDE\s*=\s*[\"'](dirichlet|neumann)[\"']", txt, re.M)
    if m:
        return m[-1]
    # A CONTRACT THAT IS ONE ROLE BY CONSTRUCTION says so in its first line
    # and carries no SIDE: the served Kratos Neumann contract was read as
    # role-unknown and told it imposes a trace (measured; one run spent six
    # minutes replacing a working side).
    hm = re.search(r"\((NEUMANN|DIRICHLET) side\)", txt[:400], re.I)
    return hm.group(1).lower() if hm else None


def _stated_role(side_dir: Path, at: float | None = None) -> str | None:
    """'dirichlet' / 'neumann' where a side states its role: its config.json
    ("side" or "role", which the served contracts let override the constant),
    else its participant's SIDE line. None when it states neither.

    AS OF A FILE TIME, where `at` is given: only a config.json and a script written by
    then speak -- the script is the newest participant*.py not newer than `at`, a copy
    moved aside as .replaced- included. Measured: a side whose script was rewritten
    (SIDE = "dirichlet") after the exports it was judged on -- which its earlier
    "neumann" script, still on disk as a .replaced- copy, had written -- was told it
    exported a different trace from the one it imposes. Where every script on disk is
    newer than `at` and no config.json of that time states a role, "rewritten" comes
    back: the version that wrote those files is not on disk."""
    import json as _json

    def _cfg_role():
        try:
            cfg = _json.loads((side_dir / "config.json").read_text() or "{}")
        except Exception:                                # noqa: BLE001
            return None
        for key in ("side", "role"):
            v = str(cfg.get(key) or "").strip().lower() if isinstance(cfg, dict) else ""
            if v in ("dirichlet", "neumann"):
                return v
        return None
    scripts = sorted(side_dir.glob("participant*.py"))   # backups included: the role as of the exports
    if at is None:
        role = _cfg_role()
        if role:
            return role
        for q in scripts:
            if ".replaced-" in q.name:
                continue
            role = _script_role(q)
            if role:
                return role
        return None
    try:
        cfg_then = (side_dir / "config.json").is_file() and (side_dir / "config.json").stat().st_mtime <= at
        then = [q for q in scripts if q.stat().st_mtime <= at]
    except OSError:
        return None
    role = _cfg_role() if cfg_then else None
    if role:
        return role
    if then:
        return _script_role(max(then, key=lambda q: q.stat().st_mtime))
    return "rewritten" if scripts or _cfg_role() else None


def _field_gap_at(sd: Path, coords, values, scale: float, detail: bool = False):
    """The largest gap between a side's newest field dump and `values` at `coords`,
    relative to `scale`; None when the dump does not hold at least half the points.

    A TIME HISTORY IS COMPARED AT THE DUMP'S OWN TIME. `values` with more columns per point
    than the dump has value columns cannot be components of the dumped field: it is one
    column per time step (the transient contracts' (n_points, n_steps) exchange), and the
    dump is one instant of it. Measured: the first column against an end-of-window dump
    read a 99.7 % gap where the last column matched to 1e-12. Every column is compared
    and the best-matching one decides; with `detail` the result is (gap, {"columns": m,
    "last": the last column's gap}) -- m is 1 where the columns are not a time axis."""
    try:
        import numpy as _np
        dumps = sorted(sd.glob("field_level*.csv"), key=lambda q: q.stat().st_mtime)
        if not dumps:
            return None
        h, r = _read_named_csv(dumps[-1])
        if not ("x" in h and "y" in h) or len(h) < 3 or len(r) < 3:
            return None
        F = _np.asarray(r, float)
        xy = F[:, [h.index("x"), h.index("y")]]
        vcols = [i for i, c in enumerate(h) if c not in ("x", "y", "z")]
        vcol = vcols[0]
        C = _np.asarray(coords, float)
        if C.ndim != 2 or C.shape[1] < 2:
            return None
        V = _np.asarray(values, float).reshape(len(C), -1)
        cols = list(range(V.shape[1])) if V.shape[1] > len(vcols) else [0]
        tol = 1e-8 * (float(_np.ptp(xy, axis=0).max()) or 1.0)
        hits = [(i, _np.where(_np.abs(xy - c).max(axis=1) <= tol)[0]) for i, c in enumerate(C[:, :2])]
        hits = [(i, hit) for i, hit in hits if hit.size]
        if len(hits) < max(3, 0.5 * len(C)):
            return None
        per_col = [max(float(_np.abs(F[hit, vcol] - V[i, j]).min()) for i, hit in hits) / scale
                   for j in cols]
        best = min(per_col)
        return (best, {"columns": len(cols), "last": per_col[-1]}) if detail else best
    except Exception:                                    # noqa: BLE001
        return None


def _imported_trace_not_held(work: Path, converged: bool | None = None) -> list[dict]:
    """A side's exported interface values differ from the ones it imported.

    THE SAME DEFECT AS THE NGSOLVE-SPECIFIC WRITE GATE, SEEN FROM THE DATA AND
    THEREFORE TRUE OF EVERY BACKEND. On a Dirichlet-Neumann pair the side that
    imports a trace imposes it, so its own exported trace must come back
    unchanged. Where it does not, the solve overwrote what was handed to it and
    the two subdomains are answering different problems -- while the iteration
    still converges, because each side is self-consistent.

    FOUND BY ASKING WHAT SEPARATES A CORRECT RUN FROM A PHYSICALLY WRONG ONE.
    Same problem, same two codes, same recovery method, the same 44 interface
    points in the same order, three coupled levels each. The correct run's side B
    exports exactly what it imported (0.00%); the wrong run's differs by 127%. Five other
    hypotheses were ruled out first -- point correspondence, corner handling,
    the constrained dof set, the values array indexing, and whether
    skfem's solve(*condense(...)) restores constrained values (it does).

    Measured over the recorded runs: **no judgeable side of a correct run is
    flagged**, against many sides in failing ones.

    Judgeable only where a side imports and exports the same kind of values,
    which is what makes it the trace-holding side; anything else returns
    nothing rather than a guess.

    THE ROWS ARE PAIRED BY WHERE THEY ARE, AND ONLY FILES THAT BELONG TOGETHER ARE
    COMPARED. Sorted along one axis, a bent interface listed in two different row
    orders paired different points: two sides that exported their imports to
    round-off led two levels with "100 %" and "73 %" (measured). And an export
    older than the imports beside it was not made from them: a stalled level
    leaves the driver's next imports next to the last export. So the rows are
    paired by position (quality_checks.pair_interface_points), the export must
    have been written after the imports, and a caller that knows the last
    coupling did not converge (`converged=False`) gets nothing from this.
    """
    import json as _json
    out: list[dict] = []
    if converged is False:
        return out
    try:
        import numpy as _np
        _pair = _qc().pair_interface_points
    except Exception:                                    # noqa: BLE001
        return out
    for sd in sorted(work.glob("side_*")):
        try:
            ip, ep = sd / "imports.json", sd / "exports.json"
            if not (ip.is_file() and ep.is_file()):
                continue
            if ep.stat().st_mtime < ip.stat().st_mtime:
                continue                                 # this export was not made from these imports
            # A NEUMANN SIDE IMPOSES NO TRACE. On a vector pair it imports and
            # exports values of the same shape, and this check told one it
            # "EXPORTS A DIFFERENT TRACE FROM THE ONE IT IMPORTED" while the defect
            # sat in its partner (measured). The role is read where the side
            # states it; an unstated role keeps the check, worded conditionally.
            role = _stated_role(sd, at=ep.stat().st_mtime)
            if role == "rewritten":
                # THE PARTNER OF THAT TIME SAYS IT: in a Dirichlet-Neumann pair the side
                # whose partner was the Neumann side is the Dirichlet side.
                _mates = [o for o in sorted(work.glob("side_*")) if o != sd and (o / "exports.json").is_file()]
                if len(_mates) == 1:
                    _pr = _stated_role(_mates[0], at=(_mates[0] / "exports.json").stat().st_mtime)
                    role = {"neumann": "dirichlet", "dirichlet": "neumann"}.get(_pr, role)
            if role == "neumann":
                continue
            if role == "rewritten":
                out.append({"sequence": f"imported trace {sd.name}", "values": [], "informational": True,
                            "finding": (
                    f"THE IMPORTED-TRACE CHECK DID NOT RUN (NOT CHECKED) on {sd.name}: every "
                    f"participant script in it was written after its exports.json, and neither a "
                    f"config.json of that time nor its partner's files of that time state a role, so "
                    f"which role wrote those exports is not on disk. "
                    f"Nothing is judged; the next coupling run writes exports this check can read.")})
                continue
            imp = _json.loads(ip.read_text() or "{}")
            exp = _json.loads(ep.read_text() or "{}")
            src = next(iter(imp.values()), {}) if imp else {}
            iv = _np.array(src.get("values") or [], float)
            ev = _np.array(exp.get("values") or [], float)
            ic = _np.array(src.get("coordinates") or [], float)
            ec = _np.array(exp.get("coordinates") or [], float)
            if iv.size == 0 or ev.size == 0 or ic.ndim != 2 or ec.ndim != 2:
                continue
            if iv.size % len(ic) or ev.size % len(ec) or iv.size // len(ic) != ev.size // len(ec):
                continue                                 # not the trace-holding side
            iv, ev = iv.reshape(len(ic), -1), ev.reshape(len(ec), -1)
            scale = float(abs(iv).max())
            if scale == 0.0:
                continue
            pr = _pair(ic, ec)
            if pr is None or len(pr["ia"]) < 2 or (pr["only_a"] and pr["only_b"]):
                continue                                 # not the same points: no pointwise reading
            rel = float(abs(iv[pr["ia"]] - ev[pr["ib"]]).max() / scale)
            if rel < 0.01:
                # AN EXPORT THAT EQUALS THE IMPORTS IS NOT YET A HELD TRACE. Measured: a
                # side told this finding copied its imports into its exports, the finding
                # went quiet, and its field still held the outer value along the whole
                # interface. Where the side's newest field dump holds the interface points,
                # the field is read there too.
                _got = _field_gap_at(sd, ic, iv, scale, detail=True)
                _gap, _info = _got if _got is not None else (None, {})
                if _gap is not None and _gap > 0.01:
                    _what = (
                        f"its exports match its imports, a time history of {_info['columns']} columns "
                        f"per point, and its newest field dump holds none of its {_info['columns']} "
                        f"columns at the same interface points: the last column (the end of the "
                        f"window) differs by {_info['last']:.0%} of the imported scale, the closest "
                        f"by {_gap:.0%}. Either that dump is not this side's solved field at a step "
                        f"of this window, or the trace never reaches the field. "
                        if _info.get("columns", 1) > 1 else
                        f"its exports match its imports, while its newest field dump differs from "
                        f"the imported trace by {_gap:.0%} of the imported scale at the same "
                        f"interface points. ")
                    out.append({"sequence": f"imported trace {sd.name}", "values": [_gap],
                                "finding": (
                        f"{sd.name} EXPORTS THE TRACE IT IMPORTED, BUT ITS FIELD DOES NOT HOLD IT: "
                        + _what +
                        f"An export copied from the imports passes the exchange "
                        f"and leaves the field unchanged. The trace has to reach the interface "
                        f"nodes' own degrees of freedom, stay held there through the solve, and "
                        f"be read back from the solved field.")})
                continue
            out.append({"sequence": f"imported trace {sd.name}", "values": [rel],
                        "finding": (
                f"{sd.name} EXPORTS A DIFFERENT TRACE FROM THE ONE IT IMPORTED: "
                f"the two differ by {rel:.0%} of the imported scale at the same "
                f"interface points. "
                + ("This side is given the partner's trace and imposes it, so its field "
                   "must hold that trace at the interface, and what it exports back, "
                   "read from that field, must be that trace unchanged"
                   if role == "dirichlet" else
                   "If this side is the one that imposes the partner's trace (the "
                   "Dirichlet side; none is stated in its config.json or its "
                   "participant's SIDE), its field must hold that trace at the "
                   "interface, and what it exports back, read from that field, must "
                   "be that trace unchanged")
                + f". Copying the imports into the export silences this "
                f"finding and changes nothing in the field. The iteration converges anyway -- each side is "
                f"self-consistent -- and the two subdomains end up answering "
                f"different problems. The trace has to reach the interface "
                f"nodes' own degrees of freedom, stay held there through the "
                f"solve, and be read back from them: a trace written into other "
                f"dofs and held there makes this finding go quiet while the "
                f"interface itself stays free (measured). Measured across "
                f"recorded runs, no correct run has ever exported a trace that "
                f"differs from the one it was given.")})
        except Exception:                                # noqa: BLE001
            continue
    return out


def _interface_transmitted_nothing(work: Path, lv) -> list[dict]:
    """The same exchanged column identically zero on BOTH sides.

    NEITHER SIDE'S OWN SELF-CHECK CAN SEE THIS. Each participant guards
    "my recovered flux is ~0 against a NONZERO imported one" -- so when the
    exchange is zero in both directions, both guards are satisfied and both
    stay silent. The iteration then converges immediately, because a pair that
    exchanges nothing cannot disagree, and each side returns the answer it
    would have returned uncoupled.

    MEASURED on a live run: it coupled three levels with both codes
    proven and its two sides agreeing on displacement to the digit, while the
    traction columns read 9.5e-18 and 0.0. It came out completed but unphysical.
    Across the record the rule hits fourteen runs and none of them is correct. A seam
    legitimately at zero does not trip it: the test needs the SAME column dead
    on BOTH sides while the interface carries a nonzero scale elsewhere.
    """
    out: list[dict] = []
    try:
        import csv as _csv
        for fa in sorted(work.rglob("interface_level*_A.csv")):
            k = _level_of(fa)
            if k is None or k not in lv:
                continue
            fb = Path(str(fa).replace("_A.csv", "_B.csv"))
            if not fb.is_file():
                continue
            def _read(q):
                with q.open() as fh:
                    rd = _csv.reader(fh)
                    hdr = next(rd, None)
                    rows = []
                    for r in rd:
                        try:
                            rows.append([float(x) for x in r])
                        except ValueError:
                            pass
                return hdr, rows
            ha, A = _read(fa)
            hb, B = _read(fb)
            if not A or not B or len(A[0]) != len(B[0]):
                continue
            ncol = len(A[0])
            scale = max([abs(v) for r in A for v in r[1:]]
                        + [abs(v) for r in B for v in r[1:]] + [0.0])
            if scale <= 1e-12:
                continue
            for c in range(1, ncol):
                za = max(abs(r[c]) for r in A)
                zb = max(abs(r[c]) for r in B)
                if za < 1e-12 * scale and zb < 1e-12 * scale:
                    name = (ha[c] if ha and c < len(ha) else f"column {c}")
                    out.append({"sequence": f"interface transmitted nothing level {k}",
                                "values": [za, zb], "finding": (
                        f"YOUR TWO SIDES EXCHANGED NOTHING at level {k}: the "
                        f"`{name}` column is identically zero in BOTH "
                        f"{fa.name} and {fb.name} ({za:.2e} and {zb:.2e}) while "
                        f"the interface carries values of order {scale:.2e} "
                        f"elsewhere. Neither participant's export self-check can "
                        f"see this -- each only fires when its OWN recovery is ~0 "
                        f"against a NONZERO partner, and here both are zero, so "
                        f"both stay quiet. The iteration then converges at once, "
                        f"because two sides exchanging nothing cannot disagree, "
                        f"and each returns the answer it would have returned with "
                        f"no partner at all. Find where that quantity is recovered "
                        f"and why it comes out zero before running another level.")})
                    break
    except Exception:                                    # noqa: BLE001
        return out
    return out
