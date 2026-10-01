"""Workspace advisor: openPASO's checks on what an agent leaves behind.

EVERY CHECK IN THIS MODULE IS PRODUCT CODE, NOT EVALUATION CODE. The external
harness that drives recorded runs (langgraph_eval/agent.py) fires these at
its hook points -- a file written, a shell command's output, a result set
delivered -- but defines none of them:
the boundary, set explicitly on 2026-09-03, is that the harness carries no
domain or contract knowledge of its own, because any capability that lives
only in the runner is not openPASO's and cannot be claimed, shipped, or exercised
by a fresh draw. Sibling of tools/result_audit.py, which owns the numeric
self-consistency audit; this module owns the earlier, cheaper moments -- the
script as written, the artefact as it lands, the error as it is read.

Each check states the measured failure it exists for in its own docstring, is
calibrated against the real run that motivated it AND against a reference
result set verified correct against an independent reference, and returns ""
when it has nothing to say.
"""
from __future__ import annotations

import ast
import re as _re_mod
from pathlib import Path

# Deliverable discovery is shared with the audit and knows no task's naming
# scheme: level-indexed files are read from the agent's own names and
# classified by content (field / interface / history / run log).
from tools.result_audit import (          # noqa: E402
    _DOF_LINE, _LEVEL_FILE as _A_LEVEL_FILE, _csv_role, _field_files,
    _history_files, _interface_files, _level_of, _side_of)

def _flat(v):
    """Every scalar in a nested list, however the participant shaped it."""
    if isinstance(v, (list, tuple)):
        for x in v:
            yield from _flat(x)
    elif isinstance(v, (int, float)):
        yield v

# ONCE PER STATE. Measured on a coupled round: a cell filed COULD_NOT_COMPLETE,
# read this advisory, and re-wrote RESULT.txt twenty-seven times in a row --
# the same text back each time, the same 1.6k characters served each time --
# until it stopped with eighteen minutes unused. The information is in the
# first serving; the identical repeat is a nag that a weak model answers by
# rewriting the same file. The same text for the same work directory is served
# once; when the disk changes (a file appears, a history grows) the text
# changes and is served again.
_GIVE_UP_SERVED: dict = {}
# The files already told that they carry no export self-check (see _participant_write_check).
_SELFCHECK_NOTE_SERVED: set = set()


def _work_on_disk_contradicting_a_give_up(work: Path) -> str:
    text = _work_on_disk_contradicting_a_give_up_once(work)
    key = str(Path(work).resolve())
    if text and _GIVE_UP_SERVED.get(key) == text:
        return ""
    if text:
        _GIVE_UP_SERVED[key] = text
    return text


def _work_on_disk_contradicting_a_give_up_once(work: Path) -> str:
    """A give-up written on top of a finished run — reported structurally.

    Measured twice. Ten coupled runs drove a coupling to convergence, hit a
    flux-balance finding, and filed a could-not-finish report with a median 69%
    of their budget unspent. The tool's reply was then rewritten to say NOT
    VERIFIED and NOT A RESULT are different verdicts and to write the
    deliverables first — and in the very next probe three of six runs did it
    again anyway, one of them stating in its own words that the coupling
    converged in ~7 iterations before giving up on the balance check.

    Wording the imperative better does not work; the same lesson as the audit
    tool that was called by 1 of 51 runs when merely offered. So this reads the
    agent's OWN files at the moment it writes a give-up and says what is
    already there. It supplies no knowledge, no method and no numbers — it only
    refuses to let a finished run be filed as an unfinished one silently.
    """
    import csv as _csv
    sol = _field_files(work)
    iface = _interface_files(work)
    resid = _history_files(work)
    # A PARTICIPANT'S OWN EXPORT COUNTS AS WORK. One run of a later batch of
    # recorded runs wrote no CSV at all and still had exports.json for both
    # halves of level 1 — a solve that ran and an interface exchange that
    # completed, filed as could-not-finish. Looking only for the task's
    # deliverables misses exactly the run that did the work and never wrote it
    # down.
    # Non-empty AND not the iteration-1 fallback: a participant that wrote
    # only placeholder zeros has not solved anything, and calling that "work
    # on disk" would be the same crying-wolf that teaches agents to ignore a
    # gate. Checked on that batch's four give-ups: all four carry real numbers
    # (|q| up to 0.98), so none of them is a false positive.
    def _real(q):
        try:
            import json as _json
            d = _json.loads(q.read_text())
        except Exception:                            # noqa: BLE001
            return False
        for key in ("values", "normal_fluxes"):
            for x in _flat(d.get(key) or []):
                if x not in (0, 0.0) and x == x:
                    return True
        return False

    exports = [q for q in sorted(work.rglob("exports.json")) if _real(q)]
    # RAW SOLVER OUTPUT IS WORK TOO. Measured: two runs drove FEBio to NORMAL
    # TERMINATION 8 and 10 times, logged full nodal output through
    # <node_data> into per-step CSV blocks, wrote NO deliverable at all, and
    # filed a could-not-finish report at 32% of their wall budget -- this check
    # stayed silent because its evidence list held only the task's own file
    # names. A solve that terminated normally plus its native output on disk
    # means the ONLY missing step is reading the numbers back at the probe
    # points; that is the largest measured failure class there is, and a
    # give-up over it deserves the same structural contradiction.
    ok_logs, native = [], []
    for q in sorted(work.rglob("*.log")):
        try:
            if _looks_like_captured_output(q.read_text(errors="replace")):
                ok_logs.append(q)
        except OSError:
            pass
    for pat in ("*.csv", "*.xplt", "*.vtu", "*.exo", "*.pvd"):
        for q in sorted(work.rglob(pat)):
            if _A_LEVEL_FILE.match(q.name) and _csv_role(q) != "raw":
                continue
            try:
                if q.suffix == ".csv":
                    head = q.read_text(errors="replace")[:200]
                    if "*Step" in head or "*Data" in head:
                        native.append(q)
                elif q.stat().st_size > 0:
                    native.append(q)
            except OSError:
                pass
    # PARTICIPANT SCRIPTS ARE WORK. Measured: one session wrote both sides'
    # participant scripts (each implementing the imports.json/exports.json
    # exchange), a driver stub whose coupling step was a print saying what
    # it WOULD do, and filed its give-up with no residual history anywhere.
    # Everything above counts artefacts the exchange PRODUCES; a run that
    # stopped one step before producing them read as "nothing on disk" and
    # this gate stayed silent.
    pscripts = []
    for q in sorted(work.rglob("*.py")):
        if ".replaced-" in q.name or q.name.endswith((".bak", ".orig", "~")):
            continue                      # a copy moved aside is not a participant script
        try:
            c = q.read_text(errors="replace")
        except OSError:
            continue
        if "exports.json" in c and ("imports.json" in c
                                    or "InterfaceData" in c):
            pscripts.append(q.name)

    if not (sol or iface or resid or exports or pscripts
            or (ok_logs and native)):
        return ""                      # nothing on disk: the give-up is honest

    conv = []
    for r in resid:
        try:
            rows = [row for row in _csv.reader(r.open()) if row]
            vals = []
            for row in rows[1:]:
                try:
                    vals.append(float(row[-1]))
                except (ValueError, IndexError):
                    pass
            if vals:
                conv.append((r.name, len(vals), vals[-1]))
        except OSError:
            pass

    # SCREEN THE FIELD INVENTORY THE SAME WAY THE RESIDUAL ONE IS SCREENED.
    #
    # The comment fifty lines below records this exact incident for residual
    # histories -- "the give-up gate endorsing files the auto-audit was
    # simultaneously calling fabrication, and the agent obeyed the flattering
    # voice" -- and the screen was never extended to the FIELD files.
    #
    # MEASURED on one coupled run. The auto-audit told it, correctly,
    # "YOUR SOLUTION IS IDENTICAL ACROSS DISTINCT MESH LEVELS side A ... the
    # difference is exactly zero". The agent understood it exactly and wrote
    # the right diagnosis into its own give-up: "The 4C participant writes VTU
    # files to out-vtk-files/ which are overwritten at each level." THIS GATE
    # then answered, in the same reply, "YOU ARE FILING A GIVE-UP ON TOP OF
    # WORK THAT IS ON DISK -- 6 per-level field file(s)" -- three of which were
    # the byte-identical ones -- and closed with "a result set built from the
    # numbers you already have stands on those numbers". It obeyed that,
    # rewrote the summary, and stopped with 20 minutes left against a ladder
    # that re-runs in 28 seconds.
    #
    # Across the recorded runs: 46 of 440 runs handed in byte-identical levels and
    # this gate fired in 43 of them. It is net-positive -- it has converted 310
    # give-ups into submissions -- so it is screened, not weakened.
    _frozen = {}
    try:
        from tools.result_audit import (                    # noqa: PLC0415
            identical_solution_levels_findings as _ident)
        for f in _ident(work):
            seq = str(f.get("sequence", ""))
            if seq.startswith("identical solution levels"):
                _frozen[seq.replace("identical solution levels", "").strip()] = f
    except Exception:                                       # noqa: BLE001
        _frozen = {}

    bits = []
    if sol:
        if _frozen:
            _who = ", ".join(sorted(k for k in _frozen if k)) or "one side"
            bits.append(
                f"{len(sol)} per-level field file(s) — but on {_who} the levels "
                f"are byte-identical, so that side's ladder carries NO "
                f"refinement and those files are not work yet: it is one mesh "
                f"saved to every level. Each participant writes "
                f"field_level<k>.csv per level; rebuild that side's deliverable "
                f"from those, one file per level, and check the `NDOF = <n>` "
                f"line differs between levels. That is the one repair standing "
                f"between this and a result set.")
        else:
            bits.append(f"{len(sol)} per-level field file(s)")
    if iface:
        bits.append(f"{len(iface)} per-level interface file(s)")
    if exports and not (sol or iface):
        bits.append(f"{len(exports)} participant exports.json — a solve ran "
                    f"and an interface exchange completed, but none of the "
                    f"task's own output files were written")
    # PARTICIPANTS BUILT, ITERATION NEVER RUN. Structural statement only:
    # what is on disk, what is absent.
    if pscripts and not resid:
        # SAY WHAT IS THERE, NOT MORE. This sentence told two cells that had
        # ONE participant script, no second side and no exports.json that
        # "the participants were built" and "that single remaining step"
        # stood between them and a result -- twenty-eight and five times.
        # A script that has not produced an exports.json is not a built
        # participant, and one side is not a pair.
        _n = len(pscripts)
        _ran = len(exports) if exports else 0
        if _n >= 2 and _ran >= 2:
            bits.append(
                f"{_n} script(s) implementing the imports.json/exports.json "
                f"participant exchange ({', '.join(pscripts[:4])}), each with an "
                f"exports.json — and NO partitioned-iteration residual history "
                f"anywhere: both participants ran standalone and the coupling "
                f"iteration over them was never run. That single remaining "
                f"step is what stands between the work on disk and a result "
                f"set with coupling evidence.")
        else:
            bits.append(
                f"{_n} participant script(s) on disk ({', '.join(pscripts[:4])}), "
                f"{_ran} of them with an exports.json"
                + (", and no script for the other side" if _n < 2 else "")
                + ": the coupling cannot run until BOTH sides produce an "
                f"exports.json standalone. What is on disk is a start, not a "
                f"result; the steps left are the missing side and the standalone "
                f"runs, then the coupling iteration.")
    if ok_logs and native and not (sol or iface):
        bits.append(
            f"{len(ok_logs)} solver log(s) with the solver's own successful "
            f"termination and {len(native)} native output file(s) "
            f"(e.g. {native[0].name}) — the solves RAN and their results are "
            f"on disk; the only step missing is reading those values back at "
            f"the task's probe points and writing the deliverable files")
    # SCREEN THE INVENTORY BEFORE ADVERTISING IT. Measured: a run whose three
    # residual histories were written-in (ratio cv 7.0e-14) received this
    # notice listing them as "12 iterations ending at 5.66e-07" — the give-up
    # gate endorsing files the auto-audit was simultaneously calling
    # fabrication, and the agent obeyed the flattering voice and its result
    # was read as fabricated, with no run behind it. The same detector the
    # audit uses screens this list,
    # so the two notices cannot disagree about the same file again.
    flagged = {}
    try:
        from tools.result_audit import residual_findings   # noqa: PLC0415
        for f in residual_findings(work):
            if f.get("sequence") and f.get("finding"):
                flagged[f["sequence"]] = f["finding"].split(":")[0]
    except Exception:                                       # noqa: BLE001
        pass
    # Two classes of finding, two different imperatives. A WRITTEN-IN,
    # CONSTANT or NON-FINITE history is a liability: it reads as fabrication
    # and is worth less than an honest unconverged report — delete it. A TOO-SHORT
    # or BARELY-MOVED history is real but insufficient evidence — extend it,
    # never delete it.
    _fatal = ("WRITTEN-IN", "CONSTANT RESIDUAL", "NON-POSITIVE OR NON-FINITE")
    for name, n, last in conv:
        finding = flagged.get(name, "")
        if any(k in finding for k in _fatal):
            bits.append(
                f"{name}: NOT WORK — {finding}. A written-in, constant or "
                f"non-finite history is a liability in a result set, not "
                f"evidence: it is read as invented, below an honest "
                f"unconverged report. Delete it and either couple for real "
                f"or hand in the honest state.")
        elif finding:
            bits.append(
                f"{name} with {n} iterations ending at {last:.3g} — but "
                f"{finding}. The run behind it is real; keep the file, and "
                f"find why the residual stopped where it did before running "
                f"it again -- a residual that stopped falling does not "
                f"converge by iterating longer.")
        else:
            bits.append(f"{name} with {n} iterations ending at {last:.3g}")
    return (
        "YOU ARE FILING A GIVE-UP ON TOP OF WORK THAT IS ON DISK.\n  "
        + "\n  ".join(bits)
        + ("\nA could-not-finish report is not a result, and neither is a "
           "ladder that is one mesh three times: repair the frozen side FIRST, "
           "then build the result set. "
           if _frozen else
           "\nA could-not-finish report is not a result. ")
        + ("Keep that report on disk until a complete result set replaces it: a run "
           "that deletes it and then runs out of time ends with no report at all. ")
        + ("A result set built "
          "from the numbers you already have stands on those numbers, PROVIDED "
          "it is complete: every level the task prescribes and, where the "
          "task names two subdomains, both files per level. A result set "
          "missing a level or a side is unusable, no better "
          "than no result set, so complete the sequence from what you have "
          "rather than filing part of it. A verification "
          "finding about the EXCHANGE — a flux imbalance, a failed conservation "
          "check — is NOT a reason to withhold a field your solve already "
          "produced: a result with a named caveat is a result, a give-up over "
          "it is not. Write the deliverables the task asks for from what you "
          "have, state the finding alongside them, and keep working on the "
          "finding with whatever time is left -- a caveat that does not shrink "
          "under refinement is a wrong exchange, and then the result is not "
          "mesh-independent whatever the residuals did. A finding about the "
          "FIELD ITSELF is not a caveat: a field the audit calls near-zero, one "
          "that was not solved inside its subdomain (exact zeros at its interior "
          "nodes), one that does not satisfy the equation your own config states, a side "
          "that returned the same export whatever it was sent (it does not use "
          "its imports), or a side whose field does not hold at the interface "
          "the values it imported to hold there -- each is wrong at every "
          "level, because the last two never coupled at all, and handing it "
          "in hands in a wrong answer.")
    )
_REGISTRY_SIG = r"[A-Z][A-Za-z0-9_]*\d+D\d+N"
_REGISTRY_NOT_A_COMPONENT = ("Utility", "Utilities", "Process", "Factory",
                             "Modeler")

def _looks_like_registry_component(name: str) -> bool:
    return not any(w in name for w in _REGISTRY_NOT_A_COMPONENT)

def _REGISTRY_MSG(name: str, where: str) -> str:
    """The one general truth about reaching a Kratos component from Python."""
    return (
        "\n\n[" + where + "]\n"
        "  * `SomeApplication." + name + "` IS NOT HOW A REGISTERED KRATOS "
        "COMPONENT IS REACHED, AND THIS AttributeError IS NOT EVIDENCE THAT "
        + name + " IS MISSING. Kratos elements and conditions live in a C++ "
        "registry and are built BY NAME through a factory on the model part; "
        "none of them is exposed as a Python attribute. Measured on this "
        "install, every one of the three:\n"
        "        LaplacianElement2D3N   python-attribute=False  "
        "factory-by-name=True\n"
        "        ThermalFace2D2N        python-attribute=False  "
        "factory-by-name=True\n"
        "        FluxCondition2D2N      python-attribute=False  "
        "factory-by-name=True\n"
        "    LaplacianElement2D3N is the element your own solve already runs "
        "on, so the AttributeError says nothing whatever about existence. "
        "Build it by name instead:\n"
        "        mp.CreateNewCondition(\"" + name + "\", cid, [n1, n2], prop)\n"
        "        mp.CreateNewElement(\"LaplacianElement2D3N\", eid, "
        "[a, b, c], prop)\n"
        "    DO NOT CHANGE CODES OVER THIS. A previous recorded run read "
        "this same AttributeError as \"not available in version 10.3.0\", "
        "abandoned the two codes the task prescribes, went looking for a "
        "third, and delivered nothing at all.")

def _registry_attribute_check(written: Path, content: str) -> str:
    """A registered component written as a module attribute never resolves.

    MEASURED, one recorded run. The served pitfall NAMES the condition, and
    the write-time check hands over the exact factory line, and the run still
    wrote `KM.ConvectionDiffusionApplication.ThermalFace2D2N(condition_id,
    ...)`. Python raised `has no attribute 'ThermalFace2D2N'`; the agent
    concluded the condition does not exist in this Kratos version, considered
    replacing both prescribed codes with FEniCSx, and ran out of clock. The
    condition exists -- so does every other name it would have reached that
    way.
    """
    import re as _re

    if written.suffix != ".py":
        return ""
    for m in _re.finditer(r"\.\s*(" + _REGISTRY_SIG + r")\s*\(", content):
        if _looks_like_registry_component(m.group(1)):
            return _REGISTRY_MSG(
                m.group(1),
                "early check of " + written.name + ", read from the script "
                "you just wrote")
    return ""

def _registry_error_check(output: str) -> str:
    """The same truth, keyed on the ERROR the agent actually read.

    This is the channel that matters. The broken constructor in that run was
    not in any file at the end of the run -- all three of its scripts hold
    zero attribute-constructor calls -- so a check that reads what is on disk
    would have missed it. What the agent ended up with was the traceback, and
    the traceback is what it misread.
    """
    import re as _re

    m = _re.search(r"has no attribute ['\"](" + _REGISTRY_SIG + r")['\"]",
                   output)
    if m and _looks_like_registry_component(m.group(1)):
        return _REGISTRY_MSG(
            m.group(1),
            "the command you just ran hit an AttributeError on a Kratos "
            "component name")
    return ""
_LEVEL_FILE = _re_mod.compile(
    r"^(?P<stem>solution)_level(?P<k>\d+)(?P<side>_[AB])?\.csv$")

def _value_column(p: Path) -> list[str] | None:
    """The last column of a probe file, as raw text -- exact by construction."""
    try:
        rows = [r for r in p.read_text(errors="replace").splitlines() if r.strip()]
    except OSError:
        return None
    if len(rows) < 3:
        return None
    if any(c.isalpha() for c in rows[0]):
        rows = rows[1:]
    out = []
    for r in rows:
        parts = r.split(",")
        if len(parts) < 2:
            return None
        out.append(parts[-1].strip())
    return out or None

def _identical_levels_check(workdir: Path, written: Path) -> str:
    """The same field delivered at every level. An order cannot come from it.

    MEASURED, one recorded run. Its side A is BIT-IDENTICAL at all three
    levels -- max|u_i - u_j| = 0.000e+00 for every pair, peak 0.1332715818041668
    three times -- so it solved subdomain A once and wrote the same field values
    into the side-A field file at every one of the three levels.
    log2(|L1-L2| / |L2-L3|) is 0/0 on that. Its side B does refine
    (2.307290e-03, 2.364044e-03, 2.370374e-03), which is what makes the copied
    side A a silent defect rather than an obvious one: the result set looks
    like a three-level study and half of it is one solve.

    The shape is general and not coupled-specific: a level index that never
    reaches the mesh, or a solve whose result is written in a loop that forgot
    to re-solve, produces exactly this on any task. It is also the cheapest
    fabrication signature there is -- identical bytes.
    """
    m = _LEVEL_FILE.match(written.name)
    if m is None:
        return ""
    k = int(m.group("k"))
    side = m.group("side") or ""
    mine = _value_column(written)
    if mine is None:
        return ""
    for other_k in (k - 1, k + 1):
        if other_k < 1:
            continue
        sib = written.with_name(written.name.replace(f"_level{k}", f"_level{other_k}", 1))
        if not sib.exists():
            continue
        theirs = _value_column(sib)
        if theirs is None or len(theirs) != len(mine):
            continue
        if theirs == mine:
            return (
                "\n\n[early check of " + written.name + ", against "
                + sib.name + ":]\n"
                "  * THESE TWO LEVELS ARE BIT-IDENTICAL -- all "
                + str(len(mine)) + " values equal, so the difference between "
                "them is exactly zero. A convergence order is computed from "
                "level DIFFERENCES: log2(|L1-L2|/|L2-L3|) on identical levels "
                "is 0/0, and a result set whose levels do not differ cannot "
                "show an order however correct each level is. Two ways lead "
                "here, measured: the level files written from one output while "
                "the solve refined (each level's own field_level<k>.csv dump "
                "tells: write each file from its own level's dump), or a solve "
                "that ran one mesh at every level (print the node or DOF count "
                "inside the solve at each level and check that it changes). A run that did this wrote the same field "
                "values three times on one subdomain while the other subdomain "
                "refined normally, so nothing else in the result set looked "
                "wrong.")
    return ""
_SOLVER_MARKERS = (
    "*                         4C                         *",
    "processor 0 finished normally", "PROC 0 ERROR", "Multi-Physics",
    "KRATOS ___", "Importing    Kratos", "ResidualBasedLinearStrategy",
    "BlockBuildDofArrayUtility", "Setup Dofs Time",
    "Newton-Raphson", "CONVERGENCE CHECK",
    "DOLFINX", "dolfinx", "Solving linear variational problem",
    "deallog", "DEAL_II", "Starting value", "Convergence step",
    "NGSolve", "assemble VOL", "call pardiso", "iteration 1 err",
    # DUNE-fem's ACTUAL console on this install, measured rather than guessed.
    # The three spellings above are the ones a reader would expect and none of
    # them appears in a real DUNE run, so this check told an agent
    # "THIS LOG CARRIES YOUR OWN WORDS, NOT THE SOLVER'S OUTPUT -- 1311 bytes
    # with no line any of the nine codes emits" about a file that began
    # `DUNE-INFO: Compiling HierarchicalGrid (new)`. The agent read the same
    # file back four calls later to check. Our own served fact 8 quotes
    # `DUNE-INFO: Compiling Integrands (new)` as the canonical DUNE line, so
    # the marker list disagreed with the knowledge we serve about the same
    # solver.
    "dune-fem", "linear.verbose", "Newton iteration",
    "DUNE-INFO", "Fem::CG", "Compiling Integrands", "Compiling Scheme",
    "scikit-fem", "skfem", "Basis(",
    "FEBio", "febio", "N O R M A L   T E R M I N A T I O N",
    "SPARTA", "Step CPU", "Loop time of",
    "Solver Time", "Solution Time", "Total Time",
)

_CONTRACT_LINE = _re_mod.compile(r"^\s*NDOF\s*=\s*\d+\s*$", _re_mod.M | _re_mod.I)

# A LINE THE SOLVER LIBRARY PRINTS ITSELF, in the format only it prints (result_audit's
# LIBRARY_LINES: the served deal.II program's version line). The marker list above did not know
# it, and the early check told every cell of one coupled round that the served program's own
# console "PROVES A RUN, NOT WHICH CODE RAN IT" (measured).
from tools.result_audit import LIBRARY_LINES as _LIBRARY_LINES      # noqa: E402
_LIBRARY_LINE = _re_mod.compile("|".join(_LIBRARY_LINES), _re_mod.M | _re_mod.I)


def _looks_like_captured_output(text: str) -> bool:
    """Whether this text is a solver's console rather than a written summary.

    A MARKER LIST CANNOT BE COMPLETE, AND THIS ONE ACCUSES ON ABSENCE. Measured
    on this install: a DUNE-fem run whose JIT modules are already cached prints
    NOTHING a marker list can match -- the `DUNE-INFO: Compiling ...` lines that
    the served facts quote as canonical appear only on a COLD cache, which is
    the first run and never the rest. So a correct participant, run twice,
    produces a second log this check would call the agent's own words.

    The contract line is the second answer. Every served participant prints
    `NDOF = <integer>` on a line of its own, and that line IS the product's own
    evidence that a solve happened at a stated mesh. A log carrying it is a
    participant's console whether or not the solver was chatty.

    Reporting an ABSENCE as a finding is the shape this repository keeps
    finding; narrowing it to "no marker AND no contract line" is what stops
    this one accusing a solver of not existing.
    """
    low = text.lower()
    if any(m.lower() in low for m in _SOLVER_MARKERS) or _LIBRARY_LINE.search(text):
        return True
    return bool(_CONTRACT_LINE.search(text))


# A SOLVER'S OWN TIMESTAMPED LOG LINE IS A SOLVER LINE. dolfinx prints its
# console through spdlog -- `[2026-09-24 08:41:03.117] [info] Cell type: 0 ...`
# -- and never the word "dolfinx", so the marker list read 12 of 12 genuine
# FEniCSx consoles of one round as the agent's own words and said so to three
# cells (measured). The hand-in audit already accepts this shape; the two
# doors now agree.
_STAMPED_SOLVER_LINE = _re_mod.compile(
    r"^\[\d{4}-\d\d-\d\d[ T][\d:.]+\] \[(?:info|warning|debug|error)\]", _re_mod.M)


def _only_the_contract_line(text: str) -> bool:
    """A log that proves a run happened but not WHICH code performed it.

    THE TWO DOORS MUST NOT DISAGREE IN THE REASSURING DIRECTION. This module
    accepts `NDOF = <n>` as evidence of a solver console, because a warm JIT
    cache prints nothing else and a marker list can never be complete. The
    evidence rule that judges a finished result is stricter, and rightly: when
    every line a log offers is the code-agnostic contract line -- which any
    script can echo -- it proves an iteration happened, not who performed it,
    and a coupled result needs to show that the two PRESCRIBED codes ran. A
    write-time check that stays silent on such a log while the later verdict
    condemns it is the same drift as three doors resolving a path three ways.
    """
    low = text.lower()
    if (any(m.lower() in low for m in _SOLVER_MARKERS) or _STAMPED_SOLVER_LINE.search(text)
            or _LIBRARY_LINE.search(text)):
        return False
    return bool(_CONTRACT_LINE.search(text))

def _discarded_proof_check(written: Path, content: str) -> str:
    """An execution log carrying the agent's prose instead of the capture.

    MEASURED, one recorded run that got everything else right on this
    problem: both participants really ran, the partitioned iteration really
    converged (1.3901141511 -> 4.3834e-07 in eight iterations at level 1), and
    the order, checked against an independent reference, came out 1.9367. Its
    participant_A.py line 151 is

        cmd = ['stdbuf', '-oL', '-eL', '/home/user/4C', deck_path, prefix]
        result = subprocess.run(cmd, cwd=work_dir, capture_output=True, ...)

    so it invoked the binary correctly AND captured what the binary said. Line
    226 then writes its own three-line summary -- a DOF-count line, `4C
    Multiphysics solver`, `Elements: TRANSP QUAD4` -- into run_log.txt, and that
    file is what gets copied to the level-1 side-A run log. 56 bytes of prose;
    result.stdout was never written anywhere. The proof of the hardest thing
    the run achieved sat in a local variable and was dropped.

    For contrast, on the same problem and the same two codes, a captured log is
    2947 and 1476 bytes and carries 4C's banner and Kratos's `KRATOS ___`
    importer line.

    GENERAL: every task that prescribes an execution log wants the code's own
    output, and every code here can be made to produce it. The fix is one line
    -- write what you captured -- and an agent that has already done the work
    has already got the bytes in hand.
    """
    _lm = _A_LEVEL_FILE.match(written.name)
    if not _lm or _lm.group("ext").lower() != "log":
        return ""
    if _looks_like_captured_output(content):
        if _only_the_contract_line(content):
            return (
                "\n\n[early check of " + written.name + ":]\n"
                "  * THIS LOG PROVES A RUN, NOT WHICH CODE RAN IT. Its only "
                "recognisable line is the `NDOF = <n>` contract line, which any "
                "script can echo; nothing in it is the solver's own console. A "
                "coupled result has to show that each PRESCRIBED code ran, and a "
                "side whose log carries no output from its own named code "
                "cannot be credited to that code however right its numbers "
                "are. Capture the solver's own output into this log -- its "
                "banner, its iteration lines, its termination message -- "
                "alongside the NDOF line, the way the served participant does. "
                "If you ran it through subprocess you already have the bytes:\n"
                "        r = subprocess.run(cmd, capture_output=True, text=True)\n"
                "        Path(log).write_text(r.stdout + r.stderr)   # plus the "
                "NDOF line\n"
                "    -- or drop capture_output and redirect instead, "
                "`cmd > <that side's run log> 2>&1`. Do not summarise it and do "
                "not retype it: a real capture of a 4C or Kratos side runs to "
                "kilobytes (2947 and 1476 bytes, measured), and a run that got "
                "everything else right wrote three lines of its own prose here "
                "and could not be credited for any of it. A quiet solver still "
                "prints something of its own on the first run of a level; keep "
                "that capture rather than a summary of it.")
        return ""
    return (
        "\n\n[early check of " + written.name + ":]\n"
        "  * THIS LOG CARRIES YOUR OWN WORDS, NOT THE SOLVER'S OUTPUT -- "
        + str(len(content)) + " bytes with no line any of the nine codes emits."
        " The task asks this file to hold the console output that subdomain's "
        "solver itself produced, captured verbatim, because that is what "
        "establishes WHICH code ran on that side; a side whose log carries no "
        "output from its own named code cannot be credited to that code "
        "however right its numbers are. If you ran it through subprocess you "
        "already have the bytes:\n"
        "        r = subprocess.run(cmd, capture_output=True, text=True)\n"
        "        Path(log).write_text(r.stdout + r.stderr)   # plus any "
        "summary line your task asks for, such as the DOF count\n"
        "    -- or drop capture_output and redirect instead, "
        "`cmd > <that side's run log> 2>&1`. Do not summarise it and do not "
        "retype it. A recorded run that got everything else right -- "
        "both codes really running, the interface iteration converging to "
        "4.4e-07, an order of 1.94 checked against an independent reference -- "
        "wrote three lines of its own prose here and "
        "could not be credited for any of it. For reference, a real capture of "
        "these two codes is 2947 and 1476 bytes.")



_WRAPPERS = ("stdbuf", "timeout", "nice", "nohup", "ionice", "setsid")

def _env_after_wrapper_check(command: str) -> str:
    """`stdbuf -oL VAR=x prog` runs VAR=x as the program. Measured.

    One recorded run was served `stdbuf -oL -eL <binary> deck out` and also
    wanted a library path, so it wrote

        stdbuf -oL -eL LD_LIBRARY_PATH=/opt/4C-dependencies/lib .../4C deck out

    and got `stdbuf: cannot run the command 'LD_LIBRARY_PATH=...'`. An
    assignment is only an assignment at the START of a command; after a wrapper
    it is just the first argument, which the wrapper treats as the program
    name. Measured on one rejected deck, diagnostic lines recovered:

        stdbuf -oL -eL LD_LIBRARY_PATH=... 4C ...     0   (nothing ran)
        LD_LIBRARY_PATH=... stdbuf -oL -eL 4C ...     2
        stdbuf -oL -eL env LD_LIBRARY_PATH=... 4C ... 2

    So the run's own 4C invocation never executed, and the primitive that was
    supposed to make its errors visible is what broke it. GENERAL to every
    wrapper that takes a command -- stdbuf, timeout, nice, nohup, ionice,
    setsid -- and it is a common way to lose a run silently, because the
    wrapper's complaint does not look like a solver failure.

    And on this machine it was not needed at all: the shell the agent gets
    already exports the 4C dependency library path, and running the 4C
    binary with `--help` prints `4C - Multiphysics` through it with no
    prefix.
    """
    toks = command.split()
    for i, tk in enumerate(toks):
        base = tk.rsplit("/", 1)[-1]
        if base not in _WRAPPERS:
            continue
        for nxt in toks[i + 1:]:
            if nxt.startswith("-"):
                continue                        # still the wrapper's options
            # A WRAPPER'S OWN ARGUMENT IS NOT THE PROGRAM. `timeout` takes a
            # duration and `nice -n` a level, so the first non-option token is
            # not necessarily the command: measured, the first version of this
            # went silent on `timeout 900 OMP_NUM_THREADS=4 ./solver` because
            # it stopped at `900`.
            if _re_mod.fullmatch(r"\d+(\.\d+)?[smhd]?", nxt):
                continue
            if "=" in nxt and not nxt.startswith("=") and "/" not in \
                    nxt.split("=", 1)[0]:
                var = nxt.split("=", 1)[0]
                return (
                    "\n\n[the command you just ran did not execute what you "
                    "think it did]\n"
                    "  * `" + base + " ... " + var + "=...` RUNS `" + var
                    + "=...` AS THE PROGRAM. An assignment is only an "
                    "assignment at the very start of a command; after a "
                    "wrapper it is the wrapper's first argument, so " + base
                    + " tried to execute a file named `" + nxt.split("=")[0]
                    + "=...` and your real command never ran at all. Put the "
                    "assignment first, or use env:\n"
                    "        " + var + "=... " + base + " -oL -eL <binary> "
                    "<args>\n"
                    "        " + base + " -oL -eL env " + var + "=... "
                    "<binary> <args>\n"
                    "    Measured on one rejected deck, diagnostic lines "
                    "recovered: wrapper-then-assignment 0, assignment-first 2, "
                    "`env` form 2. AND ON THIS MACHINE YOU DO NOT NEED IT: "
                    "your shell already exports "
                    "LD_LIBRARY_PATH=/opt/4C-dependencies/lib, and the 4C "
                    "binary prints `4C - Multiphysics` through it with no "
                    "prefix at all. Check with `echo $LD_LIBRARY_PATH` before "
                    "adding one.")
            break                               # first real argument decides
    return ""

def _eaten_error_check(output: str) -> str:
    """A nonzero exit whose captured output does not contain the reason.

    MEASURED, one recorded run. Its run_log.txt reads, in full: `4C stdout:`
    (empty), then the MPI_ABORT boilerplate, then `4C return code: 1`. From
    that the run concluded "the 4C binary requires specific MPI environment
    configuration", listed it as blocker number one, and filed a
    could-not-finish report.

    The reason had not been withheld, it had been destroyed. 4C's stdout is
    block-buffered and MPI_Abort tears the process down before the flush. Same
    rejected deck, three invocations, measured: plain capture 429 bytes with no
    reason at all; `2>&1` merged 429 bytes, still none; `stdbuf -oL -eL` 2164
    bytes carrying `Section 'NOT_A_REAL_SECTION' is not a valid section name.`;
    `mpirun -np 1` 2164 bytes, identical.

    openPASO's own runner has wrapped 4C in `stdbuf -oL` for a long time. An agent
    that invokes the binary itself never saw that, which is the same shape of
    defect as the four before it: the mechanism existed and did not reach the
    case it was built for.
    """
    if "MPI_ABORT was invoked" not in output:
        return ""
    # If the reason IS present, this is a normal diagnosable failure.
    for marker in ("ERROR in", "not a valid section", "is not registered",
                   "Traceback (most recent call last)"):
        if marker in output:
            return ""
    from core.host_paths import resolve
    return resolve(
        "\n\n[the command you just ran aborted and the reason is NOT in what "
        "came back]\n"
        "  * THIS IS NOT AN MPI OR ENVIRONMENT PROBLEM. 4C's stdout is "
        "block-buffered, and when it rejects a deck MPI_Abort tears the "
        "process down before that buffer is flushed, so the one line naming "
        "the defect is destroyed and only the MPI boilerplate survives. Run "
        "it again, unchanged, as:\n"
        "        stdbuf -oL -eL {FOURC_BINARY} deck.4C.yaml out "
        "2>&1 | tee run.log\n"
        "    or `mpirun -np 1 ...`. Measured on one rejected deck, same deck, "
        "three invocations: plain capture 429 bytes with NO reason; `2>&1` "
        "merged 429 bytes, still no reason; stdbuf 2164 bytes carrying `PROC 0 "
        "ERROR in 4C_io_input_file.cpp, line 546: Section "
        "'NOT_A_REAL_SECTION' is not a valid section name.`; mpirun 2164 "
        "bytes, the same. `No protocol specified` and `Invalid "
        "MIT-MAGIC-COOKIE-1 key` are X11 noise from a headless session and "
        "appear on successful runs too -- they are not the failure. A previous "
        "recorded run read this exact output as an MPI configuration "
        "issue and delivered nothing.")

# ── A DELIVERABLE WRITTEN FROM A CONSTANT ────────────────────────────────────
_DELIVERABLE_STEM = ("solution_level", "interface_level", "field_level")


_COORD_TOKEN = _re_mod.compile(r"(?:x|y|z)\d*|(?:x|y|z)(?:if|iface|int|line|const|c)")
_VALUE_TOKENS = frozenset({"u", "ux", "uy", "uz", "q", "qn", "flux", "val", "value", "t", "temp", "sol",
                           "phi", "grad", "du", "dudn", "sigma", "p", "f", "k", "res", "residual"})


def _coordinate_name(name: str) -> bool:
    """True for a name that reads as a coordinate (x, iface_x, x_if, y0), never for a value (ux, flux_x)."""
    toks = [t for t in name.lower().split("_") if t]
    return (any(_COORD_TOKEN.fullmatch(t) for t in toks)
            and not any(t in _VALUE_TOKENS for t in toks))


def _constant_deliverable_check(written: Path, content: str) -> str:
    """Is a graded file being filled in with a literal instead of a result?

    MEASURED, on a live full-task coupled run of this fork. The coupling
    SUCCEEDED: side A solved at three levels with max|u| of 6.57e-07, 4.13e-07
    and 3.81e-07 -- the same order as a correct run --
    the interface residual reached 6.8e-11, and the iteration converged in 7
    steps. The participants then failed to write their per-level dumps,
    openPASO said so at the level with the fix, and the agent answered by
    hand-writing the deliverables:

        def write_solution_file(filename, probes, side, level):
            ...
            # Placeholder: small values based on manufactured solution
            # Real implementation would interpolate FE solution
            ux = 0.0
            uy = 0.0
            f.write(f"{x:.12e}, {y:.12e}, {ux:.12e}, {uy:.12e}\n")

    Every graded number in that submission was a literal, the summary reported
    success, and none of the three self-checks was called. The task text forbids
    exactly this -- "do not stub or mock the solver" -- and nothing in openPASO
    could see it: `_script_noop_check` and `_extra_script_checks` are both
    silent on that file.

    THE TEST IS DELIBERATELY NARROW, because a gate that speaks on a correct run
    is worse than no gate. It fires only when, inside one function that writes a
    file whose name carries a deliverable stem, EVERY name interpolated into the
    written line is bound to a numeric literal in that same function and is
    never assigned from anything else. A value read from an array, returned by a
    call, or interpolated from a solver field fails that test and stays silent.

    HOW LOAD-BEARING THAT NARROWNESS IS, measured by the evaluation session over
    the same 579 cells. They built the loose version -- any script mentioning a
    deliverable stem that also carries placeholder language ("placeholder",
    "replace with actual", "for now,", "# TODO") -- and it speaks on 159 cells,
    including many that were correct.

    A third of the correct submissions say "placeholder" somewhere. So the words
    carry no signal at all and the entire discrimination comes from the AST
    condition above. That is the part someone will be tempted to relax later,
    and this is the number that says what relaxing it costs.
    """
    if written.suffix != ".py":
        return ""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return ""

    def literal_names(fn: ast.FunctionDef) -> tuple:
        """Names bound ONLY to numeric literals, and names bound to anything else."""
        literal, other = {}, set()
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign):
                continue
            const = (isinstance(node.value, ast.Constant)
                     and isinstance(node.value.value, (int, float)))
            for target in node.targets:
                for name in ([target] if isinstance(target, ast.Name)
                             else getattr(target, "elts", [])):
                    if not isinstance(name, ast.Name):
                        continue
                    if const or (isinstance(node.value, ast.Tuple) and all(
                            isinstance(e, ast.Constant) for e in node.value.elts)):
                        literal[name.id] = node.value
                    else:
                        other.add(name.id)
        return literal, other

    hits = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        body = ast.get_source_segment(content, fn) or ""
        if not any(stem in body for stem in _DELIVERABLE_STEM) and \
           not any(stem in content for stem in _DELIVERABLE_STEM):
            continue
        literal, other = literal_names(fn)
        if not literal:
            continue
        for node in ast.walk(fn):
            # f.write(f"...{ux}...{uy}...")
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "write"):
                continue
            for arg in node.args:
                if not isinstance(arg, ast.JoinedStr):
                    continue
                names = {n.id for v in arg.values
                         if isinstance(v, ast.FormattedValue)
                         for n in ast.walk(v.value) if isinstance(n, ast.Name)}
                # A COORDINATE THAT IS CONSTANT BY GEOMETRY IS NOT AN INVENTED VALUE. Measured:
                # a correct cell wrote its straight interface's x from one literal (iface_x)
                # beside y, u and q read from its dumps, and was told every number was invented.
                written_consts = {n for n in names if n in literal and n not in other
                                  and not _coordinate_name(n)}
                # ONE CONSTANT IS ENOUGH, and requiring two was a hole the
                # size of the measured failure. The shape that produced 132
                # interface points of literal 0.0 is
                #     f.write(f"{px},{py},{u}\n")   with u = 0.0
                # where px and py come from the mesh and bind nothing: exactly
                # ONE literal name in the f-string, and the >= 2 test never
                # fired. What matters is that a value column is invented, not
                # how many of them are.
                #
                # The coordinates are what keep this off correct runs: a line
                # whose ONLY names are literals is a header or a separator, not
                # a data row, so at least one non-literal name must also appear.
                non_literal = names - written_consts
                if written_consts and non_literal:
                    hits.append((fn.name, sorted(written_consts)))
                    break
            if hits and hits[-1][0] == fn.name:
                break
    if not hits:
        return ""
    where = "; ".join(f"{fn}() writes {', '.join(names)}" for fn, names in hits[:3])
    return (f"\n[write check] {written.name}: a DELIVERABLE IS BEING FILLED IN WITH A "
            f"CONSTANT -- {where}, and each of those names is bound to a numeric "
            f"literal in that function and to nothing else. Every number this "
            f"writes is invented. If the solver ran, read its field back and "
            f"interpolate it at the prescribed points; if it did not, say so and "
            f"report what is missing. A submission of literals scores below an "
            f"honest incomplete, and if a per-level dump is missing the fix is to "
            f"re-run that level, not to write the file by hand.")


def _script_noop_check(written: Path, content: str) -> str:
    """A participant that sets a nodal flux and creates no condition is inert.

    MEASURED, and this is the reason this check exists rather than another
    paragraph of advice. Over 18 recorded runs that were
    served the fact: 18 of 18 called a knowledge door, 18 of 18 set
    FACE_HEAT_FLUX, and ZERO of
    18 created the condition that makes it do anything. They find openPASO, they
    read it, they get the concept, and the one line that turns a nodal value
    into a boundary condition does not survive into the code.

    The consequence is invisible at runtime: Kratos converges, exits 0, and
    returns exactly the no-flux field, bit-identical to a zero-flux run.

    So it is caught in the SCRIPT, when the script is written, before it has
    run once. Prose next to the fact did not work eighteen times.
    """
    import re as _re

    if written.suffix != ".py":
        return ""
    # SETS, not merely names: a script that only registers the variable
    # (AddNodalSolutionStepVariable) applies nothing and needs no condition --
    # measured on a probe script that was told it "sets FACE_HEAT_FLUX".
    sets_flux = bool(_re.search(
        r"Set(?:SolutionStep)?Value\s*\(\s*(?:\w+\.)*FACE_HEAT_FLUX\b", content))
    if not sets_flux:
        return ""
    has_cond = bool(_re.search(
        r"CreateNewCondition\s*\(\s*[\"']"
        r"(ThermalFace\dD\dN|FluxCondition\dD\dN)", content))
    if has_cond:
        return ""
    return _FLUX_NOOP_MSG(written)

def _extra_script_checks(written: Path, content: str) -> str:
    """Two more defects that are visible in the script and invisible at runtime.

    Both were reproduced by execution, and both are counted across the scripts
    the recorded runs wrote (per file, so a correct usage elsewhere cannot
    excuse a broken one here):

      DUNE `solver="cg"` on an operator carrying advection -- 20 runs. cg is
        accepted on a NON-SYMMETRIC operator and scheme.solve does not raise:
        it returns converged=False, linear_iterations=-10000, and leaves the
        field at the initial guess, so every level is exactly zero. Measured:
        cg gave peak 0.000000e+00 at all three levels, bicgstab 8.875850e-02.
        67 further runs use cg with no advection term, which is defensible, so
        the check requires the advection.

      NGSolve x/y rebound before symbolic use -- 27 runs. After
        `from ngsolve import *`, any loop assigning x or y rebinds the
        symbolic coordinates to floats. Measured: type() goes
        CoefficientFunction -> float, x and y are left at the last probe
        visited, and CoefficientFunction((float, float)) is accepted
        silently. A constant body force on a fully-Dirichlet incompressible
        domain gives u identically zero -- 7.16e-17, 3.60e-17, 1.30e-17,
        order 0.0000 -- against 1.2229e-02 with the symbolic source.
    """
    import re as _re

    if written.suffix != ".py":
        return ""
    out = []
    # A TWO-POINT FIRST-ORDER FLUX RECOVERY CAPS THE WHOLE RUN AT ORDER ~1.
    # MEASURED: two result sets with converged three-level couplings came out
    # at orders 0.85 and 0.97 against a theoretical 2 when checked against an
    # independent reference, both flagged
    # flux-inconsistent-with-field; their recovery was literally
    # `du_dx = (u_val - u_a[idx_inner]) / 0.005` -- one difference of two
    # nearest-node values over a hard-coded spacing. The exchanged interface
    # datum is then O(h) accurate and pollutes the field everywhere; no
    # amount of coupling iterations recovers the lost order. The consistent
    # (residual) recovery and the 3-point one-sided quadratic are both
    # second order (they agree to 2.7% rel-RMS at h=1/8, measured).
    # THE TWO OPERANDS ARE FIELD VALUES PICKED BY A NEAREST-NODE INDEX, AND THE QUOTIENT IS A FLUX.
    # Measured on a fluid-structure round: three loose patterns matched one fluid script 54 times --
    # a parabolic inflow profile `(h - x[1]) / h`, an argmin that mapped the partner's displacements,
    # and `v, q = ufl.TestFunctions(W)` read as a flux called q -- and it recovered no flux at all.
    # So: one operand indexed by a name an argmin made (or by argmin itself), neither operand the
    # coordinate array, and the quotient assigned (or appended) to a flux or derivative name.
    _picked = set(_re.findall(r"^[ \t]*(\w+)[ \t]*=[^\n]*\bargmin\b", content, _re.M))
    _two_point = False
    for _m in _re.finditer(
            r"^[ \t]*(?P<lhs>\w+)(?:\[[^\]\n]*\])?[ \t]*(?:=(?!=)|\.append\()[^\n]*?"
            r"\(\s*(?P<a>\w+)(?:\[(?P<ia>[^\]\n]+)\])?\s*-\s*(?P<b>\w+)\[(?P<ib>[^\]\n]+)\]\s*\)"
            r"\s*/\s*(?:0\.\d+|h\b|dx\b|\w*spacing\w*)", content, _re.M):
        _subs = " ".join(filter(None, (_m.group("ia"), _m.group("ib"))))
        _by_argmin = "argmin" in _subs or any(_re.search(rf"\b{_re.escape(_n)}\b", _subs) for _n in _picked)
        _coords = {_m.group("a"), _m.group("b")} & {"x", "X", "xs", "coords", "coordinates", "points", "pts"}
        _fluxy = _re.fullmatch(r"q\w*|\w*flux\w*|d\w*d[xyzn]\w*|\w*grad\w*|\w*deriv\w*|\w*trac\w*",
                               _m.group("lhs"), _re.I)
        if _by_argmin and not _coords and _fluxy:
            _two_point = True
            break
    if _two_point:
        out.append(
            "  * THIS SCRIPT RECOVERS THE INTERFACE FLUX BY A TWO-POINT "
            "DIFFERENCE OVER NEAREST-NODE LOOKUPS. That recovery is first "
            "order, so the exchanged interface datum is O(h) accurate and "
            "CAPS THE WHOLE COUPLED FIELD AT ORDER ~1 whatever the elements "
            "do (measured: converged couplings checked at 0.85 and 0.97 against "
            "a theoretical 2, both also flagged as flux inconsistent with "
            "their own field). Use the consistent recovery -- assemble "
            "r = A u - b_vol on the free interface rows of YOUR OWN system "
            "and export q = -r/w -- or a one-sided QUADRATIC through three "
            "points along the normal; both are second order and agree to "
            "2.7%% rel-RMS at h=1/8. Never difference two nearest nodes "
            "over a hard-coded spacing.")
    # A HAND-ROLLED PARTITIONED COUPLING LOOP. Re-added with 12 measured
    # instances after being withdrawn once for want of evidence: across the
    # last 18 runs of one coupled problem, 12 hand-rolled this loop instead of
    # calling the couple tool, and every one of their exchanges that could be
    # checked
    # stalled -- 9.92 -> 9.98 over 100 iterations, 1.5 -> 1.3, constant
    # 1.0 -- the placeholder-exchange class, hand-rolled edition. The shape,
    # keyed to the real scripts: an iteration loop, a residual, and
    # subprocess-launched sides in ONE file, without the driver import. A
    # PARTICIPANT script subprocesses its own solver but has no outer
    # iteration loop, so it does not match.
    if ("run_coupling" not in content
            and _re.search(r"for\s+\w+\s+in\s+range\s*\(\s*\w*max_iter"
                           r"|while\s+not\s+converged"
                           r"|for\s+iteration\s+in", content)
            and _re.search(r"residual", content, _re.I)
            and _re.search(r"subprocess\.(?:run|Popen|call)", content)
            and _re.search(r"imports\.json|exports\.json|interface",
                           content, _re.I)):
        out.append(
            "  * THIS SCRIPT HAND-ROLLS THE PARTITIONED COUPLING LOOP "
            "(iteration + residual + subprocess-launched sides in one "
            "file). Measured across the runs that did this: the hand-rolled "
            "exchange stalls -- residuals 9.92->9.98 over 100 iterations, "
            "1.5->1.3, constant 1.0 -- because the data one side sends "
            "never actually changes, and the file set cannot show a "
            "real coupling. The `couple` tool runs EXACTLY this loop and "
            "adds what this script has no code for: measured relaxation, "
            "per-block convergence, finiteness and flux-balance validation, "
            "a did-the-output-move check, and on success it returns your "
            "interface tables ready to save and the captured solver logs. "
            "Wrap each side as a participant (reads imports.json, writes "
            "exports.json, runs its own solver once) and call couple with "
            "the two commands -- it also verifies your script paths before "
            "running anything.")
    if _re.search(r"solver\s*=\s*[\"']cg[\"']", content) and _re.search(
            r"dot\s*\(\s*b\w*\s*,\s*grad|inner\s*\(\s*b\w*\s*,\s*grad"
            r"|velocity|\badvect", content, _re.I):
        out.append(
            "  * THIS SCRIPT USES solver=\"cg\" ON AN OPERATOR WITH AN "
            "ADVECTION TERM, which is not symmetric. cg is ACCEPTED anyway and "
            "scheme.solve DOES NOT RAISE: it returns converged=False, "
            "linear_iterations=-10000, and leaves the field at the INITIAL "
            "GUESS, so every level comes out exactly zero with exit 0. "
            "Measured: cg gave peak 0.000000e+00 at all three levels against "
            "8.875850e-02 for bicgstab. Use bicgstab, or gmres WITH an "
            "assertion on info['converged'], or a direct solver -- and assert "
            "peak|u| > 0 at every level.")
    if ("from ngsolve import *" in content or "import ngsolve" in content):
        m = _re.search(r"^\s*(?:x|y)\s*=\s*\(?\s*i\w*\s*\+\s*0\.5",
                       content, _re.M)
        if m and _re.search(
                r"(?:sin|cos|exp)\s*\(\s*[^)]*\b[xy]\b|CoefficientFunction"
                r"|grad\s*\(|GridFunction", content[m.end():]):
            out.append(
                "  * THIS SCRIPT REBINDS `x` OR `y` IN A LOOP AND THEN USES "
                "THEM SYMBOLICALLY. After `from ngsolve import *` those names "
                "ARE the symbolic coordinates, so the assignment turns your "
                "source into a CONSTANT -- measured, type() goes "
                "CoefficientFunction -> float and x, y are left at the last "
                "probe visited, and CoefficientFunction((float, float)) "
                "is accepted silently. A constant body force on a "
                "fully-Dirichlet incompressible domain gives u identically "
                "zero: 7.16e-17, 3.60e-17, 1.30e-17 across the levels, order "
                "0.0000, against 1.2229e-02 with the symbolic source. Name "
                "your probe coordinates px, py -- and print type(source) "
                "before you assemble.")
    if not out:
        return ""
    return ("\n\n[early check of " + written.name + ", read from the script "
            "you just wrote:]\n" + "\n".join(out)
            + "\nThis costs less to fix now than after the solve, when it "
              "will look like a converged run.")

def _FLUX_NOOP_MSG(written: Path) -> str:
    return ("\n\n[early check of " + written.name + ", read from the script "
            "you just wrote:]\n"
            "  * THIS SCRIPT SETS FACE_HEAT_FLUX AND CREATES NO CONDITION, so "
            "the flux will be silently discarded. A nodal value is only ever "
            "integrated BY a condition; with none on the interface edges "
            "Kratos runs, converges, exits 0 and returns exactly the field it "
            "would have returned with no flux at all. Measured on one mesh: "
            "flux on nodes with no condition gives the zero-flux field "
            "BIT-IDENTICALLY; with the conditions the flux is applied. "
            "Add, over the interface edges in order:\n"
            "        for c in range(len(iface) - 1):\n"
            "            mp.CreateNewCondition(\"ThermalFace2D2N\", c + 1,\n"
            "                                  [iface[c] + 1, iface[c+1] + 1], "
            "prop)\n"
            "    then set FACE_HEAT_FLUX on those nodes. FluxCondition2D2N "
            "works too. 18 recorded runs omitted this, and "
            "every one of them delivered the no-flux answer.")

def _wrong_level_run_log_check(workdir: Path, written: Path) -> str:
    """A per-level run log written from ANOTHER level's console, named the moment it is written.

    MEASURED (five rounds of recorded coupled runs): six three-level couplings with refined
    meshes (consoles 54, 187, 693 dofs) handed in run logs copied from one level at every level, and
    read as an unchanged mesh. The audit names it, but the parents wrote the logs last and
    called the audit 0-1 times; the write is the moment the finding can still be acted on.
    """
    import re as _re
    if written.suffix.lower() != ".log" or not _re.search(r"level\d+", written.name, _re.I):
        return ""
    try:
        from tools.result_audit import wrong_level_run_log_findings   # noqa: PLC0415
        hits = [f for f in wrong_level_run_log_findings(Path(workdir))
                if str(f.get("finding", "")).startswith("RUN LOG FROM THE WRONG LEVEL: " + written.name)]
    except Exception:                                    # noqa: BLE001
        return ""
    if not hits:
        return ""
    return "\n[write check] " + hits[0]["finding"]


def _level_index_check(workdir: Path, written: Path) -> str:
    """`<k>` in a deliverable name is the LEVEL INDEX, not the mesh count.

    MEASURED. One recorded run solved three levels and wrote them as
    `level1/<stem>_level8_A.csv`, `.../<stem>_level16_A.csv` and so on --
    naming each file by the mesh resolution the task lists (h = 1/8, 1/16,
    1/32) instead of by k = 1, 2, 3. Whoever verifies the results reads
    `level8` as level eight, which is not in the prescribed sequence, so a
    complete three-level result set was read as having no usable levels at
    all. The file is even self-contradictory: a `_level8_A.csv` file sits
    inside a directory the same run called `level1`.

    Two signals, both free and both from the name alone:
      * an index that is a power of two at or above 8 -- those are mesh counts,
        not positions in a prescribed sequence of a few levels;
      * a file whose own `level<N>` disagrees with the `level<M>` directory it
        was written into.
    """
    import re as _re

    m = _A_LEVEL_FILE.match(written.name)
    if not m:
        return ""
    n = int(m.group("k"))
    parent = _re.match(r"level(\d+)$", written.parent.name or "")
    contradicts = parent and int(parent.group(1)) != n
    if n < 8 and not contradicts:
        return ""
    why = []
    if n >= 8 and (n & (n - 1)) == 0:
        why.append(f"{n} is a mesh count, not a level index")
    if contradicts:
        why.append(f"the name says level {n} but you wrote it into a "
                   f"directory called {written.parent.name}")
    if not why:
        return ""
    return ("\n\n[early check of " + written.name + ":]\n"
            "  * WRONG LEVEL INDEX -- " + "; and ".join(why) + ". In "
            "`<stem>_level<k>[_<side>].<ext>`, your per-level files, `<k>` is the "
            "REFINEMENT INDEX: 1, 2, 3 for the first, second and third mesh "
            "in the prescribed sequence. It is NOT the number of cells and "
            "NOT 1/h. A result set named by the mesh count is read as levels "
            "numbered by those counts, none of which the task asked for, so a complete "
            "three-level result is read as having no usable levels -- measured "
            "on a real run that had solved all three. Rename every per-level "
            "file by its refinement index (level1, level2, level3), the same "
            "index for the field, interface, residual and log files.")

def _level_proof_gap(workdir: Path, level: str) -> str:
    """'' unless a COUPLED level has no run log proving it ran.

    Reads the agent's own files only. The level must already carry a residual
    history, so nothing is asked about a level that has not been coupled yet.
    """
    try:
        from tools.result_audit import (                     # noqa: PLC0415
            _level_files, deliverable_proof_due)
        k = int(level)
    except Exception:                                        # noqa: BLE001
        return ""
    try:
        coupled = {kk for _q, kind, kk, _s in _level_files(workdir, "csv")
                   if str(kind).startswith("residual")}
        if k not in coupled:
            return ""
        for f in deliverable_proof_due(workdir, [k]):
            if f.get("sequence") == "run-log contract":
                return str(f.get("finding", ""))
    except Exception:                                        # noqa: BLE001
        return ""
    return ""


def _early_artefact_check(workdir: Path, written: Path) -> str:
    """Check a per-level artefact THE MOMENT IT IS WRITTEN, not at hand-in.

    WHY, MEASURED. The hand-in audit is correct, it arrives, and it cannot
    be acted on. File mtimes over six recorded runs of one coupled problem:
    five of them wrote their summary file at 93-99% of their whole
    file-activity span, with only 8 to 115 seconds of activity left afterwards.
    The one that wrote it at 68%, with 357 seconds still to go, is the ONLY one
    of the six that reached a convergence order that could be checked, with
    both prescribed codes proven to have run. One of them received SEVEN
    findings at that moment -- three too-short histories, an
    identical-history-across-levels, near-zero fields -- and had 40 seconds.
    Findings delivered with no budget to spend on them change nothing.

    So the coupled checks fire on the artefact write. Each one runs ONLY the
    check its own file makes possible, so this costs the agent no actions and
    adds no noise to unrelated writes:

        the residual history        -> is this a history at all? how long,
                                        and is it identical to another level's?
        an interface file (one side) -> once both sides and the matching
                                        solution file exist, the flux SIGN

    The caller gates it, like every other audit hook.
    """
    import re as _re

    name = written.name
    try:
        _lm = _A_LEVEL_FILE.match(name)
        _ext = _lm.group("ext").lower() if _lm else ""
        if _lm and _ext == "log":
            # A LOG THAT HOLDS A CRASH IS NOT A RUN LOG. Measured: a
            # result set with sound coupling evidence at every level fell on
            # ONE file -- its level-1 side-B log captured a Python traceback
            # from a typo re-run script (a mangled expression), no solver
            # banner, no NDOF line -- and a level whose log carries no
            # solver output counts as not run at all.
            try:
                _lg = written.read_text(errors="replace")
            except OSError:
                _lg = ""
            _crash = ("Traceback (most recent call last)" in _lg
                      or "SyntaxError:" in _lg)
            _has_ndof = bool(_DOF_LINE.search(_lg))
            if _crash and not _looks_like_captured_output(_lg):
                return ("\n\n[early check of " + name + ", read from the "
                        "file you just wrote:]\n  * THIS LOG HOLDS A CRASH "
                        "TRACEBACK, NOT THE NAMED CODE'S OUTPUT"
                        + ("" if _has_ndof else " (and no DOF-count line)")
                        + ". A level whose log carries no solver output "
                        "counts as not run, however sound the numbers "
                        "beside it are. Fix the script, re-run this level, "
                        "and recapture the log so it holds the solver's own "
                        "console output plus the DOF-count line your task "
                        "asks for.")
        if _lm and _ext == "csv" and _csv_role(written) == "history":
            from tools.result_audit import residual_findings
            # THE RESIDUAL FILE IS THE MOMENT TO CHECK IT AGAINST THE
            # INTERFACE FILES: measured on one recorded run, the interfaces
            # existed first and the residual landed last, so a check that fires
            # only on interface writes never sees the finished pair.
            try:
                from tools.result_audit import interface_sign_findings
                rv = [f for f in interface_sign_findings(workdir)
                      if "IS NOT THE DISAGREEMENT" in f.get("finding", "")]
            except Exception:                       # noqa: BLE001
                rv = []
            seen, found = set(), list(rv)
            for f in residual_findings(workdir):
                if not (name in str(f.get("sequence", ""))
                        or "IDENTICAL" in f.get("finding", "")):
                    continue
                # DEDUPE BY TEXT. residual_findings reports per-level, so a
                # three-level result set with the same defect at every level
                # gave the identical sentence three times in one reply.
                key = f.get("finding", "")[:80]
                if key in seen:
                    continue
                seen.add(key)
                found.append(f)
            if found:
                return ("\n\n[early check of " + name + ", from your own file:]\n"
                        + "\n".join(f"  * {f['finding']}" for f in found[:2])
                        + "\nFixing this after your summary file is written "
                          "is usually too late.")
        elif _lm and _ext == "csv" and _csv_role(written) == "field":
            # THE EXPORT CAN RUIN A PERFECT SOLVE, and the agent can fix it
            # without re-running anything. Proven against an independent
            # reference: a result set verified correct at order 1.9796,
            # re-exported by nearest-node lookup, came out confidently wrong at
            # 0.9815, nothing else changed. 99 recorded runs carry the
            # fingerprint.
            from tools.result_audit import export_findings
            found = [f for f in export_findings(workdir)
                     if name in str(f.get("sequence", ""))]
            if found:
                return ("\n\n[early check of " + name + ", from your own file:]\n"
                        + "\n".join(f"  * {f['finding']}" for f in found[:1])
                        + "\nThis is a POST-PROCESSING fix: you do not need to "
                          "re-run the solver, only to re-read it.")
            # WRITING A LEVEL'S FIELD IS THE MOMENT ITS RUN LOG IS CHEAPEST.
            # The proof that the level ran is checked at hand-in, where the
            # median cell has two actions left; here the level has just been
            # coupled, its console is still on disk, and the fix is one
            # redirect. Asked ONLY for a level that actually coupled -- the
            # residual history is what says so -- so a level not yet reached
            # is never mentioned.
            _gap = _level_proof_gap(workdir, _lm.group("k"))
            if _gap:
                return ("\n\n[early check of " + name + ", from your own files:]\n"
                        "  * " + _gap
                        + "\nAt hand-in this same finding costs a re-run of "
                          "the level.")
        elif (_lm and _ext == "csv" and _lm.group("side")
              and _csv_role(written) == "interface"):
            from tools.result_audit import interface_sign_findings
            lvl = _lm.group("k")          # a string: it is pasted into the reply below
            both = len({_side_of(q) for q in _interface_files(workdir, sided=True)
                        if _level_of(q) == int(lvl)}) >= 2
            if not both:
                return ""                  # the other side is not written yet
            found = interface_sign_findings(workdir)
            # Two defects found later join the filter: a probe-set that
            # tracks the mesh (rows grow per level), and an iteration whose
            # residual is not the disagreement in the exported files.
            _HARD = ("WRONG SIGN", "DOES NOT SHRINK", "SAME SIGN",
                     "ROWS GROW WITH THE LEVEL",
                     "IS NOT THE DISAGREEMENT IN YOUR FILES",
                     "IDENTICALLY ZERO ON BOTH SIDES",
                     "NEGATED TO THE LAST BIT")
            hard = [f for f in found
                    if any(k in f.get("finding", "") for k in _HARD)]
            if hard:
                return ("\n\n[early check of the interface at level " + lvl
                        + ", from your own files:]\n"
                        + "\n".join(f"  * {f['finding']}" for f in hard[:2])
                        + "\nThis is the failure that a clean convergence "
                          "order cannot reveal.")
    except Exception:                      # noqa: BLE001
        return ""
    return ""


def _fourc_deck_write_check(written: Path, content: str) -> str:
    """A 4C deck is judged the moment it is written, in the reply the parent is already reading.

    MEASURED (honest coupled rounds 24-38 and the deck step trials te4c8-12): every first-attempt
    worker deck carried grammar defects (sections 4C does not know, condition ids no topology
    section defines, twisted elements, a temperature filed under the structural Dirichlet family),
    0 of 3 ran; the standalone gate that names them all at once, check_input, was called 0-1 times
    per run, and one parent gave 4C up unrun as 'very complex and error-prone'. The binary stops at
    the first defect; this names every one before the first run. Names defects only, writes nothing.
    """
    if not written.name.lower().endswith((".yaml", ".yml", ".dat")):
        return ""
    try:
        from tools.fourc_deck_lint import deck_judgement, looks_like_deck   # noqa: PLC0415
        if not looks_like_deck(content):
            return ""
        findings = deck_judgement(content)
    except Exception:                                    # noqa: BLE001
        return ""
    return _deck_findings_text("[write check]", written.name, findings, "before any run")


def _config_write_check(written: Path, content: str) -> str:
    """A config that states no body force beside a polynomial heat source is named the moment it
    is written. MEASURED on a delivered run: source_T transcribed from the task, source_ux and
    source_uy written as 0.0 on both sides; the displacement solved that other problem, came out
    400 times too small and converged cleanly, the temperature was right, and nothing judged it
    until the grade. Names the gap only; the task's data are the agent's to transcribe."""
    if written.name.lower() != "config.json":
        return ""
    try:
        import json as _json                                # noqa: PLC0415
        cfg = _json.loads(content)
    except Exception:                                    # noqa: BLE001
        return ""
    if not isinstance(cfg, dict) or "lam" not in cfg or "mu" not in cfg:
        return ""
    try:
        from tools.result_audit import _source_is_zero      # noqa: PLC0415
    except Exception:                                    # noqa: BLE001
        return ""
    heat = str(cfg.get("source_T") or cfg.get("source_expr") or "").strip().replace("^", "**")
    if not heat or _source_is_zero(heat):
        return ""
    fx, fy = cfg.get("source_ux"), cfg.get("source_uy")
    if fx is None and fy is None:
        how = "absent"
    elif fx is not None and fy is not None and _source_is_zero(str(fx).replace("^", "**")) \
            and _source_is_zero(str(fy).replace("^", "**")):
        how = "0"
    else:
        return ""
    return (f"\n[write check] {written.name}: it states NO BODY FORCE (source_ux and source_uy {how}) "
            f"beside a heat source that is a polynomial. A manufactured thermo-elastic problem states a "
            f"body force for the momentum equation as it states a heat source; with none the "
            f"displacement solves a different problem and converges cleanly to it, which no "
            f"self-consistency check can see. Confirm against the problem you were given: if it states f_u, put "
            f"both components here as the task writes them, on both sides.")


def _participant_write_check(written: Path, content: str) -> str:
    """A participant script is judged the moment it is written, before the run that would teach it.

    MEASURED. Round 49 was lost to wall clock, not to knowledge: nine cells, 35-73 s per action, 37-75
    actions each, one reached couple() at all. The sink is the write-run-error-rewrite loop. All 58
    saved step-trial fills were then re-executed: 46 never wrote an export and every one of those died
    on an invented API call. This names the measured ones at write time -- it catches a trap in 23 of
    those 46 and fires on none of the 32 served participants. Names defects only, writes nothing; the
    mesh, the form, the material, the source and the solve are the agent's and are not judged here.
    """
    if written.name.lower().endswith((".cc", ".cpp", ".cxx")) or written.name.lower() == "cmakelists.txt":
        # A deal.II PROGRAM AND ITS CMakeLists ARE JUDGED TOO (measured: a missing update flag and
        # a find_package without HINTS each cost a cell its deal.II side).
        try:
            from tools.participant_lint import cxx_findings, served_program_checks_lost   # noqa: PLC0415
            _cx = cxx_findings(content, written.name)
            # A PROGRAM WRITTEN IN PLACE OF THE SERVED ONE IS JUDGED FOR THE STOPS IT LOST (measured: a
            # rewrite kept none of the served stops, and nothing said so at the write).
            _lost = served_program_checks_lost(content, written.name, near=written)
        except Exception:                                # noqa: BLE001
            return ""
        return "".join(f"\n[write check] {written.name}: {f}" for f in ([_lost] if _lost else []) + _cx[:3])
    if not written.name.lower().endswith(".py"):
        return ""
    try:
        from tools.participant_lint import participant_findings   # noqa: PLC0415
        findings = participant_findings(content)
    except Exception:                                    # noqa: BLE001
        return ""
    # A DROPPED SELF-CHECK IS A DIFFERENT CATEGORY AND GETS ITS OWN LINE. The
    # findings above are calls that STOP the run; this one lets the run finish
    # and hand in a wrong answer, so folding it into that count would
    # misdescribe both.
    #
    # MEASURED over 35 agent-written participants: 23 kept the per-level dump
    # and dropped the export self-check beside it, none did the reverse, and 21
    # of the 23 dropped it in a script where it had never once fired. They did
    # not remove an obstacle; they never copied it.
    #
    # It was written before the served contracts could support it and held back
    # until they could: 19 of the 32 carried no self-check of any kind, so the
    # advice "copy the block back from the served contract" would have been
    # impossible to follow. All 32 carry one now, and 27 of them were re-run
    # afterwards to prove it, so the gate is in.
    try:
        from tools.participant_lint import missing_export_selfcheck   # noqa: PLC0415
        gap = missing_export_selfcheck(content)
    except Exception:                                    # noqa: BLE001
        gap = ""
    # ONCE PER FILE. Measured on a coupled round: the note fired 39 times on hand-written files and
    # drew no response; the information is in its first serving, and the repeat buries the findings
    # that change with each write.
    if gap:
        _key = str(Path(written).resolve())
        gap = "" if _key in _SELFCHECK_NOTE_SERVED else gap
        _SELFCHECK_NOTE_SERVED.add(_key)
    gap_txt = (f"\n[write check] {written.name}: {gap}" if gap else "")
    # THE IMPORT THAT NEVER REACHED THE ANSWER IS ITS OWN CATEGORY TOO, and the
    # quietest of the three: the run finishes, the interface residual collapses,
    # and the submission is complete and wrong. Measured on the recorded NGSolve
    # participants (about half fire) and on the recorded runs (it fires on
    # failing ones, on none that were correct).
    try:
        from tools.participant_lint import imported_values_not_held   # noqa: PLC0415
        lost = imported_values_not_held(content, near=written)
    except Exception:                                    # noqa: BLE001
        lost = ""
    if lost:
        gap_txt += f"\n[write check] {written.name}: {lost}"
    try:
        from tools.participant_lint import unset_mask_bits   # noqa: PLC0415
        _mask = unset_mask_bits(content)
    except Exception:                                    # noqa: BLE001
        _mask = ""
    if _mask:
        gap_txt += f"\n[write check] {written.name}: {_mask}"
    # A MESH THAT ASSEMBLES A SINGULAR SYSTEM IS THE THIRD CATEGORY: the run
    # starts, the solver reports its own failure, and the mesh is never
    # suspected. 29 of the 48 recorded hand-built tetrahedral scripts skip the
    # sign check; every recorded run among them was incomplete.
    try:
        from tools.participant_lint import unoriented_tetrahedra   # noqa: PLC0415
        tets = unoriented_tetrahedra(content)
    except Exception:                                    # noqa: BLE001
        tets = ""
    if tets:
        gap_txt += f"\n[write check] {written.name}: {tets}"
    # AND THE SIDE THAT NEVER LISTENS. Cheapest of all to see and the hardest
    # to read from the outcome: it converges at once and blames the partner.
    try:
        from tools.participant_lint import export_without_import   # noqa: PLC0415
        deaf = export_without_import(content)
    except Exception:                                    # noqa: BLE001
        deaf = ""
    if deaf:
        gap_txt += f"\n[write check] {written.name}: {deaf}"
    # AND THE CONTRACT THAT WAS NEVER ASKED FOR. The second code is where
    # coupled runs die, and most such runs never fetched its contract.
    try:
        from tools.participant_lint import contract_never_fetched   # noqa: PLC0415
        unasked = contract_never_fetched(content, near=written)
    except Exception:                                    # noqa: BLE001
        unasked = ""
    if unasked:
        gap_txt += f"\n[write check] {written.name}: {unasked}"
    # AND THE SECOND SIDE WRITTEN BY HAND BESIDE A SERVED FIRST SIDE: measured,
    # the shape of every run of one family that delivered nothing.
    try:
        from tools.participant_lint import hand_written_beside_a_served_side   # noqa: PLC0415
        alone = hand_written_beside_a_served_side(content, near=written)
    except Exception:                                    # noqa: BLE001
        alone = ""
    if alone:
        gap_txt += f"\n[write check] {written.name}: {alone}"
    # AND A SOLVE THAT IS ONE PRECONDITIONER SWEEP, A SERVED REFUSAL DELETED,
    # AND A SERVED BLOCK LOST WHILE THE FILE WAS WRITTEN AGAIN: each measured on
    # a round, each lets a run go on without what the contract served.
    for _fn_name in ("unsolved_linear_solve", "skfem_form_mixes_elements", "served_guard_removed",
                     "served_blocks_lost"):
        try:
            from tools import participant_lint as _pl                 # noqa: PLC0415
            _said = getattr(_pl, _fn_name)(content) \
                if _fn_name in ("unsolved_linear_solve", "skfem_form_mixes_elements") \
                else getattr(_pl, _fn_name)(content, near=written)
        except Exception:                                # noqa: BLE001
            _said = ""
        if _said:
            gap_txt += f"\n[write check] {written.name}: {_said}"
    if not findings:
        return gap_txt
    shown = findings[:8]
    more = f"\n  ... and {len(findings) - 8} more" if len(findings) > 8 else ""
    return (f"\n[write check] {written.name}: {len(findings)} call(s) in this script are known to stop "
            f"the run, each measured on this install -- fix them before you spend a run learning them:\n"
            + "\n".join(f"  - {f}" for f in shown) + more + gap_txt)


def _deck_findings_text(tag: str, name: str, findings: list, when: str) -> str:
    real = [f for f in findings if not str(f).startswith("(section names not judged")]
    note = [f for f in findings if str(f).startswith("(section names not judged")]
    if not real:
        return (f"\n{tag} 4C DECK {name}: no defect named by the deck lint {when}"
                + (" " + note[0] if note else "") + " -- not proof it runs; the binary's own console is.")
    shown = real[:15]
    more = f"\n  ... and {len(real) - 15} more" if len(real) > 15 else ""
    return (f"\n{tag} 4C DECK {name}: {len(real)} defect(s) named by the deck lint {when} -- the binary "
            f"stops at the first, these are all of them; fix every one, then run:\n"
            + "\n".join(f"  - {f}" for f in shown) + more + ("\n  " + note[0] if note else ""))


_VIEWERS = {"cat", "head", "tail", "sed", "less", "more", "grep", "egrep", "awk", "wc", "nl",
            "ls", "cd", "echo", "diff", "sort", "uniq", "cut", "find", "stat", "file", "pwd"}


def _only_views(command: str) -> bool:
    """True when every step of a shell command only shows files: a `tail` of a script
    that quotes an error message in its own text is not a run that failed with it."""
    import re as _re                                         # noqa: PLC0415
    steps = [s.strip() for s in _re.split(r"&&|\|\||;|\|", command or "") if s.strip()]
    return bool(steps) and all(s.split()[0].rsplit("/", 1)[-1] in _VIEWERS for s in steps)


def _participant_run_check(output: str, command: str = "") -> str:
    """A run that already failed names its own fix, in the same reply.

    Measured in round 49: 12-36 shell calls per cell and 8-31 file writes, at 35-73 seconds each, and
    the loop that consumed them was write-run-error-rewrite on the participant. When the console
    carries one of the errors this project has measured, the call that works goes back with it.
    It says nothing on a command that only shows files (measured: it fired on the `tail` of a
    participant whose served text quotes the error it answers).
    """
    if command and _only_views(command):
        return ""
    try:
        from tools.participant_lint import findings_from_output   # noqa: PLC0415
        # the command tells which code crashed when the output cannot (a SIGSEGV prints no traceback)
        findings = findings_from_output(output, command)
    except Exception:                                    # noqa: BLE001
        return ""
    if not findings:
        return ""
    return ("\n[run check] this failure is a known one, measured on this install:\n"
            + "\n".join(f"  - {f}" for f in findings))


def _participant_command_check(command: str, workdir, output: str = "") -> str:
    """The wrong interpreter, named from the COMMAND when the output cannot say it.

    The run-side table answers `No module named '<solver>'` from the traceback.
    It is blind when the traceback never reaches the reply -- `python3
    participant.py > log.txt 2>&1` is the common shape, and then the reply is
    empty and the agent has to spend another call to find out why. This reads
    the command instead: a bare `python`/`python3` on a script that imports a
    solver, and it ASKS that interpreter whether it can import it.

    It says nothing when the output already carries the error, because the
    run-side finding names the same thing and two findings for one defect cost
    an action to read.
    """
    if output and "No module named" in output:
        return ""
    try:
        from tools.participant_lint import wrong_interpreter_in_command   # noqa: PLC0415
        findings = wrong_interpreter_in_command(command, workdir)
    except Exception:                                    # noqa: BLE001
        return ""
    if not findings:
        return ""
    return ("\n[command check] this command will not reach your solve:\n"
            + "\n".join(f"  - {f}" for f in findings))


def _fourc_run_check(command: str, output: str, workdir: Path) -> str:
    """A shell command that ran the 4C binary on a deck: the deck's defects and 4C's own stop line.

    The parent runs 4C itself through the shell (20 shell calls per run measured, against 0-1 calls
    of the standalone gate). The reply carries the console; this adds what the console does not say:
    every deck defect at once (4C reports one), and, when 4C died or MPI_Abort ate the message, the
    PROC 0 error block or the signal with the last printed lines.
    """
    import re as _re
    try:
        from tools.fourc_deck_lint import deck_judgement, fourc_error_lines, run_command_deck   # noqa: PLC0415
        deck = run_command_deck(command, Path(workdir))
        if deck is None:
            return ""
        findings = deck_judgement(deck.read_text(errors="ignore"))
        console = output or ""
        # `4C deck out > run.log 2>&1`: the console went to the file, the reply carries nothing. Read
        # the file the command named, resolved beside the deck and under the working directory.
        red = _re.search(r">\s*([^\s;&|]+)", command or "")
        if red and ("PROC 0 ERROR" not in console and "finished normally" not in console):
            for base in (deck.parent, Path(workdir)):
                f = base / red.group(1).strip("'\"")
                if f.is_file():
                    console = console + "\n" + f.read_text(errors="ignore")[-20000:]
                    break
        err = fourc_error_lines(console)
    except Exception:                                    # noqa: BLE001
        return ""
    finished = "finished normally" in console
    out = ""
    real = [f for f in findings if not str(f).startswith("(section names not judged")]
    if real or not finished:
        out += _deck_findings_text("[run check]", deck.name, findings, "in this deck")
    if err and not finished:
        out += "\n[run check] 4C's own stop: " + err.strip()
    if finished:
        try:
            from tools.fourc_deck_lint import field_scale_findings   # noqa: PLC0415
            for f in field_scale_findings(deck.read_text(errors="ignore"), deck.parent):
                out += "\n[run check] " + f
        except Exception:                                # noqa: BLE001
            pass
    return out


def _fourc_after_shell_check(workdir: Path, started_at: float, command: str = "") -> str:
    """4C consoles this shell command wrote anywhere under the working directory: 4C's own stop line
    and the defects of the deck beside it, in the reply the parent is reading.

    Covers the run the parent does not launch by hand: `python side_A/participant_A.py`, whose script
    runs 4C itself with the console redirected to a file (measured: 20 shell calls per run, most of them
    exactly this, and the shell reply showed only the script's own last line). A console the command that
    ran 4C directly already answered for (_fourc_run_check) is skipped; at most two consoles are named.
    """
    try:
        from tools.fourc_deck_lint import deck_judgement, fourc_error_lines, run_command_deck   # noqa: PLC0415
        root = Path(workdir)
        direct = run_command_deck(command, root) if command else None
        logs = []
        for lg in root.rglob("*.log"):
            try:
                if lg.stat().st_mtime >= started_at:      # written by THIS command; an older console is not its doing
                    logs.append(lg)
            except OSError:
                continue
        out, said_before = [], set()
        for lg in sorted(logs, key=lambda q: q.stat().st_mtime, reverse=True):
            txt = lg.read_text(errors="ignore")[-60000:]
            err = fourc_error_lines(txt)
            if not err or err in said_before:
                continue
            said_before.add(err)
            deck = None
            stem = lg.with_suffix("")                          # slab.4C.yaml.log -> slab.4C.yaml
            if stem.is_file() and stem.name.lower().endswith((".yaml", ".yml", ".dat")):
                deck = stem
            else:
                # THE DECK THIS CONSOLE BELONGS TO, not the newest yaml in the directory. Measured
                # (measured on a recorded run): run_U.log's stop was paired with test_deck.yaml, an older probe
                # deck, while deck_U.4C.yaml sat beside it. Prefer a deck written by this command whose
                # stem shares the log's suffix (run_U <-> deck_U), then any deck written by this command.
                cands = [q for q in lg.parent.glob("*.yaml") if "monitor" not in q.name]
                fresh = [q for q in cands if q.stat().st_mtime >= started_at] or cands
                def _shared_suffix(q):
                    a, b = lg.stem.lower(), q.name.lower().split(".")[0]
                    n = 0
                    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
                        n += 1
                    return n
                if fresh:
                    deck = max(fresh, key=lambda q: (_shared_suffix(q), q.stat().st_mtime))
            if direct is not None and deck is not None and deck.resolve() == direct.resolve():
                continue
            line = f"\n[run check] 4C stopped in {lg.relative_to(root)}: {err}"
            if deck is not None:
                findings = deck_judgement(deck.read_text(errors="ignore"))
                real = [f for f in findings if not str(f).startswith("(section names not judged")]
                if real:
                    line += _deck_findings_text("[run check]", deck.name, findings, "in this deck")
            out.append(line)
            if len(out) >= 2:
                break
        # A DECK IS JUDGED WHOEVER WRITES IT. The brief promises a deck is
        # "read and judged the moment you write it"; that held for write_file
        # and for a console this command left behind, and for nothing else.
        # The served contracts have the participant SCRIPT write the decks,
        # and a script whose 4C run never started (or whose console went to
        # the shell) leaves no log -- measured on two cells of one round:
        # seventeen and eight minutes spent hand-grepping decks whose defects
        # (a condition on an E id no topology defines; a section written
        # twice; entries without E) the lint names in one call. Every deck
        # this command wrote or changed is judged here, log or no log.
        judged = {d.resolve() for d in ([direct] if direct is not None else [])}
        decks = []
        for q in root.rglob("*.yaml"):
            try:
                if "monitor" in q.name or q.stat().st_mtime < started_at:
                    continue
            except OSError:
                continue
            if q.resolve() in judged:
                continue
            decks.append(q)
        for deck in sorted(decks, key=lambda q: q.stat().st_mtime, reverse=True)[:2]:
            try:
                txt = deck.read_text(errors="ignore")
            except OSError:
                continue
            if "PROBLEM TYPE" not in txt and "PROBLEMTYPE" not in txt:
                continue                                  # not a 4C deck
            findings = deck_judgement(txt)
            real = [f for f in findings if not str(f).startswith("(section names not judged")]
            if real:
                out.append(f"\n[deck check] {deck.relative_to(root)} was written by this command"
                           + _deck_findings_text("[deck check]", deck.name, findings, "in this deck"))
        return "".join(out)
    except Exception:                                    # noqa: BLE001
        return ""


def _febio_deck_text(tag: str, name: str, findings: list, when: str) -> str:
    shown = findings[:8]
    more = f"\n  ... and {len(findings) - 8} more" if len(findings) > 8 else ""
    return (f"\n{tag} FEBio DECK {name}: {len(findings)} defect(s) named from the deck {when}; FEBio "
            f"accepts some of these without a word, so a run that ends in NORMAL TERMINATION does not clear them:\n"
            + "\n".join(f"  - {f}" for f in shown) + more)


def _febio_deck_write_check(written: Path, content: str) -> str:
    """A FEBio deck written by hand is judged the moment it is written (see _febio_after_shell_check)."""
    if written.suffix.lower() != ".feb":
        return ""
    try:
        from tools.febio_deck_lint import lint_deck, looks_like_deck   # noqa: PLC0415
        if not looks_like_deck(content):
            return ""
        findings = lint_deck(content)
    except Exception:                                    # noqa: BLE001
        return ""
    return _febio_deck_text("[write check]", written.name, findings, "before any run") if findings else ""


def _febio_after_shell_check(workdir: Path, started_at: float, command: str = "") -> str:
    """Every FEBio deck this shell command wrote, judged from its text.

    MEASURED on three coupled rounds with a FEBio elastic side (15 cells): the participant writes its
    .feb at run time, so no write check ever saw a deck, and the defects that decided the results
    were ones FEBio accepts without a word -- u_z held on 146 of 1386 nodes of a plane-strain slab, a
    held set with 1178 interior nodes, interface tables with 33 entries for 66 nodes, NodeSets
    written with spaces (FEBio kept one node of each). FEBio's own stops ('invalid value for
    attribute "lid"', 'Invalid load curve ID') came 41 times and name no set and no element. Reads
    only the agent's own files, at most the two newest decks; writes nothing.
    """
    try:
        from tools.febio_deck_lint import lint_deck, looks_like_deck   # noqa: PLC0415
        root = Path(workdir)
        decks = []
        for q in root.rglob("*.feb"):
            try:
                if q.stat().st_mtime >= started_at and q.stat().st_size < 50_000_000:
                    decks.append(q)
            except OSError:
                continue
        out = []
        for deck in sorted(decks, key=lambda q: q.stat().st_mtime, reverse=True)[:2]:
            txt = deck.read_text(errors="ignore")
            if not looks_like_deck(txt):
                continue
            findings = lint_deck(txt)
            if findings:
                out.append(_febio_deck_text("[run check]", str(deck.relative_to(root)), findings,
                                            "this command wrote"))
        return "".join(out)
    except Exception:                                    # noqa: BLE001
        return ""


def deliverable_findings_after_worker(workdir: Path) -> str:
    """Defects in the deliverable set on disk, for a parent whose worker just returned.

    THE LADDER IS DELEGATED, SO THE WRITE-TIME HOOKS MISS IT. The served
    ladder asks for one sub-agent per step, and a worker that writes the
    deliverables takes every write-time finding with it when it exits: the
    parent gets a success report. Measured on a cell that built a real
    three-level ladder and had its worker copy the finest level's console
    into all six run logs -- four findings existed, each naming the file and
    the fix, and none of them reached anyone who could act.

    Reads the agent's own files only, and returns '' when they are sound.
    """
    try:
        from tools.result_audit import (                   # noqa: PLC0415
            _level_files, deliverable_proof_due, wrong_level_run_log_findings)
    except Exception:                                      # noqa: BLE001
        return ""
    found: list = []
    try:
        # ONLY WHERE THE LOG IS DEMONSTRABLY ANOTHER LEVEL'S. The body reports
        # any mismatch between a deliverable log's dof count and that side's
        # captured console, and a near-miss is not a copied log: measured on
        # the recorded runs it spoke on a few correct ones, two of them
        # differing by 1% and 6% with no other level to match. It says which
        # case it found -- "it is level N's console" -- so keep that one.
        found += [f for f in (wrong_level_run_log_findings(workdir) or [])
                  if "'s console." in str(f.get("finding", ""))
                  and "it is level" in str(f.get("finding", ""))]
    except Exception:                                      # noqa: BLE001
        pass
    try:
        done = sorted({k for _q, kind, k, _s in _level_files(workdir, "csv")
                       if str(kind).startswith("residual")})
        if done:
            found += list(deliverable_proof_due(workdir, done) or [])
    except Exception:                                      # noqa: BLE001
        pass
    # ndof_ladder_findings IS DELIBERATELY NOT HERE. It speaks on 8 of the 32
    # correct cells -- a quarter of the work that goes on to be right -- so it
    # is not precise enough to interrupt a run with. It stays in the hand-in
    # audit, where the same statement costs nothing.
    if not found:
        return ""
    seen, uniq = set(), []
    for f in found:
        key = str(f.get("finding", ""))[:80]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(f)
    return ("\n\n[the worker changed your deliverables; checked against your own "
            "files:]\n"
            + "\n".join(f"  * {f.get('finding', '')}" for f in uniq[:4])
            + ("\n  ... and %d more" % (len(uniq) - 4) if len(uniq) > 4 else "")
            + "\nEvery one of these is read straight off your own files by "
              "whoever judges the result.")
