"""How a run ended, decided once.

The server, the run list and the saved record all read this. The interface
mirrors the same rule for records written before a turn carried its outcome.

The rule that matters: ``completed`` is a claim about the physics. It is not
"no exception escaped" and it is not "a solver tool was called". A solver tool
reports its own verdict, and only a result it reports as completed and
trustworthy is a finished result. A call that returned "Unknown solver: abaqus"
computed nothing; a result openPASO's attestation marks as not trustworthy ran
but is not a result anyone should report.
"""
from __future__ import annotations

import json
import re

# Tools whose return can be a solver result. run_bash is deliberately absent:
# a shell command that exits zero proves a shell command exited zero.
SOLVER_TOOLS = frozenset({
    "run_simulation", "run_with_generator", "coupled_solve",
    "couple", "couple_levels", "couple_precice", "verify_mesh_independence",
})

RUNNING = "running"
UNFINISHED = "unfinished"     # stopped without an end of its own (the server went down)
COMPLETED = "completed"        # a solver ran and openPASO verified its result
UNVERIFIED = "unverified"      # a solver ran, openPASO did not verify the result
NO_RESULT = "no_result"        # ended cleanly, no solver result
FAILED = "failed"
INTERRUPTED = "interrupted"


RESULT_LIMIT = 8000


def shorten(text: str, limit: int = RESULT_LIMIT) -> str:
    """A tool result, short enough to carry around but never cut where it
    matters. openPASO stamps its verification verdict at the END of a report,
    so taking the first 8000 characters of a long verified run threw away the
    one field that says the result can be trusted, and the run then showed as
    unverified. Keep both ends."""
    text = str(text)
    if len(text) <= limit:
        return text
    head, tail = int(limit * 0.7), limit - int(limit * 0.7)
    return (text[:head] + f"\n\n[… {len(text) - limit} characters left out of the middle …]\n\n"
            + text[-tail:])


def _payload(raw: str) -> dict | None:
    """The JSON object inside a tool result, which MCP wraps as a repr of text
    blocks. None when the result is not a JSON report (e.g. an error string)."""
    text = raw or ""
    m = re.search(r"'text':\s*(['\"])([\s\S]*?)\1\s*[,}]", text)
    if m:
        text = m.group(2).encode("utf-8").decode("unicode_escape", "ignore")
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        # Claude Code hands an MCP tool's text back as {"result": "<the reply>"}:
        # the report is the string inside, not the wrapper around it
        if isinstance(obj, dict) and set(obj) == {"result"} and isinstance(obj["result"], str):
            inner = _payload(obj["result"])
            if inner is not None:
                return inner
        return obj if isinstance(obj, dict) else None
    except ValueError:
        pass
    # The text is not a whole document — a record shortened for storage, or a
    # reply wrapped in something else. Read the two fields that matter, with
    # the structure unknown, so:
    #
    #  * trustworthy_result alone is enough. A coupling reply carries no
    #    "status" at all, and requiring one made every long coupling — which is
    #    most of them, they embed their histories and log paths — read as a
    #    call that computed nothing.
    #  * a mixture is never verified. These matches may belong to different
    #    objects, and pairing the first good one with the first bad one
    #    reported a failed run whose first participant looked fine as a
    #    verified result. If anything here says not trustworthy, or a status
    #    that is not "completed", the most that can honestly be said is that it
    #    ran.
    # the quotes may be escaped: a shortened copy of a reply wrapped as {"result": "..."}
    st = [m.lower() for m in re.findall(r'\\?"status\\?"\s*:\s*\\?"([A-Za-z_]+)\\?"', text)]
    tr = re.findall(r'\\?"trustworthy_result\\?"\s*:\s*(true|false)', text)
    if not st and not tr:
        return None
    out: dict = {}
    if st and len(set(st)) == 1:
        # one status in the whole text: it can only be this report's own
        out["status"] = st[0]
    elif st:
        # several, and no way to tell whose. Promoting one of them to the whole
        # report is how a failed run's healthy participant became the verdict,
        # so none is promoted and the verification flags decide: what can
        # honestly be said is that it ran.
        out["trustworthy_result"] = False
    if tr:
        out["trustworthy_result"] = all(v == "true" for v in tr) and out.get("trustworthy_result", True)
    return out


def _verdict_of(node: dict) -> str | None:
    """One report's verdict, or None when the node reports nothing.

    Two shapes exist. A run reports ``status`` and, from openPASO's
    verification gate, ``trustworthy_result``. A coupling reports no status at
    all: it carries the gate's ``trustworthy_result`` beside ``converged``.
    Reading only the first shape called every verified coupling a failure."""
    # Only an exact "completed" with the gate's flag is a verified result, so a
    # status the product might add later (say "completed_unphysical") reads as
    # unverified rather than as success. The prefix decides failure only.
    status = str(node.get("status", "")).lower()
    trusted = node.get("trustworthy_result")
    if status:
        if not status.startswith("completed"):
            return "failed"
        return "verified" if (status == "completed" and trusted is True) else "unverified"
    if trusted is True:
        return "verified"
    if trusted is False or "converged" in node or "all_levels_converged" in node:
        # it ran and reported on itself, but nothing here is a verified result
        return "failed" if node.get("error") else "unverified"
    if node.get("error"):
        return "failed"
    return None


def _walk(node, out: list[str]) -> None:
    """Every report inside a result, however deep. couple_levels returns one
    entry per level, each with its own verdict."""
    if isinstance(node, dict):
        v = _verdict_of(node)
        if v:
            out.append(v)
        for child in node.values():
            _walk(child, out)
    elif isinstance(node, list):
        for child in node:
            _walk(child, out)


# The legacy coupled_solve answers in prose: a convergence report followed by
# openPASO's verification note. Neither branch of that note is a machine-readable
# verdict — the tool itself says to use `couple` for one — so the most it can be
# is "it ran, and nothing verified it". Reading it as JSON made it "failed", and
# a coupling that really ran was reported as having computed nothing.
_NOTE = re.compile(r"\[openPASO verification:", re.I)
_BROKEN = re.compile(r"^(?:Backend not found|Unknown problem|Unknown solver|Error|Traceback)",
                     re.I | re.M)


def classify_solver_result(raw: str) -> str:
    """'verified', 'unverified' or 'failed' for one solver tool result.

    A report's own verdict decides it. Only when the top of the report says
    nothing about verification is the inside consulted: a ladder of couplings
    answers for the ladder, and "levels 1 and 2 verified, level 3 a null
    exchange" is a ladder that is NOT verified, however good its first levels
    were. Taking the best evidence anywhere in the tree would report exactly
    that ladder as finished."""
    p = _payload(raw)
    if not p:
        text = raw or ""
        if _NOTE.search(text) and not _BROKEN.search(text):
            return "unverified"
        return "failed"
    top = _verdict_of(p)
    if top:
        return top
    # No verdict at the top. Whatever is inside answers for its own part, not
    # for the whole, so the whole is verified only when nothing inside says
    # otherwise — the docstring's ladder, where one null level sinks it. The
    # best evidence anywhere would report exactly that ladder as finished.
    found: list[str] = []
    _walk(p, found)
    if not found:
        return "failed"
    if all(v == "verified" for v in found):
        return "verified"
    if "failed" in found and not any(v in ("verified", "unverified") for v in found):
        return "failed"
    return "unverified"


_RAISED = re.compile(r"^\s*([A-Za-z_][\w.]*(?:Error|Exception|Fault)): ?(.*)$", re.M)


def _what_broke(error: str) -> str:
    """The line of an error text that says what went wrong: a traceback's LAST exception
    line, not its banner; the text itself when it names no exception."""
    raised = _RAISED.findall(error or "")
    return f"{raised[-1][0]}: {raised[-1][1]}".strip() if raised else error


# A step a reply wrote out as text instead of taking it. Measured 2026-09-30: a
# run's history, rebuilt after a restart, handed the model its earlier steps as
# "[step] tool {args}" followed by "→ result"; its next reply wrote three steps in
# that form, results included, and called no tool.
_TYPED_STEP = re.compile(r"^\[(?:[\w/-]+ )?step\] (?P<tool>[\w.-]+)[^\n]*"
                         r"(?:\n→[^\n]*(?:\n(?!\n)[^\n]*)*)?", re.M)


def typed_steps(text: str) -> list[str]:
    """The tools a reply shows as steps it wrote out as text. None of them ran."""
    return [m.group("tool") for m in _TYPED_STEP.finditer(text or "")]


def turn_typed_steps(events) -> list[str]:
    """The steps a turn's messages wrote out as text, in the order written."""
    return [name for e in events if e.get("type") == "agent_msg"
            for name in typed_steps(e.get("text") or "")]


def without_typed_steps(text: str) -> str:
    """The reply with each step it wrote out as text replaced by a note that the
    step never ran: handed back as it was, it reads as a result."""
    return _TYPED_STEP.sub(lambda m: f"(Here the reply wrote a `{m.group('tool')}` step out as "
                                     "text. It never ran; the result written with it came from "
                                     "no tool.)", text or "")


def verdict_reason(raw: str) -> str:
    """Why, in the report's own words.

    A solver that verifies a result also says what it checked, and one that
    refuses says what is missing. Replacing that with "openPASO did not verify
    its result" throws away the answer: for a ladder whose levels are each
    sound but whose mesh never refined, the difference between "not verified"
    and "each level converged, but the mesh did not change between them, so the
    sequence shows nothing about convergence" is the difference between a
    person re-running the work and a person fixing it.

    A FAILED run is explained by what broke. Measured 2026-09-29: this read the
    verification sentence first, and for every failure that sentence is the same
    "NOT VERIFIED -- the solver run errored ...", so nine failed steps and the
    closing lines all said it and none said the error ("NameError: name
    'SplineGeometry' is not defined"), nor any fix openPASO names beside it."""
    p = _payload(raw) or {}
    if classify_solver_result(raw) == "failed":
        broke = p.get("error")
        if isinstance(broke, str) and broke.strip():
            said = " ".join(_what_broke(broke).split())
            named = p.get("what_the_error_is")
            if isinstance(named, str) and named.strip():
                said += " -- " + " ".join(named.split())
            return said
    for key in ("verification", "error", "what_to_fix_next", "message"):
        text = p.get(key)
        if isinstance(text, str) and text.strip():
            return " ".join(text.split())
    m = re.search(r'"(?:verification|error)"\s*:\s*"((?:[^"\\]|\\.){4,})"', raw or "")
    return " ".join(m.group(1).encode().decode("unicode_escape", "ignore").split()) if m else ""


def solver_verdict(events) -> str | None:
    """The best solver evidence in these events: 'verified', 'unverified',
    'failed' (a solver tool was called and computed nothing), or None.

    A verdict recorded on the event is the one that counts. It was read from the
    whole result, while the text beside it is a shortened copy that may no
    longer contain the field the verdict rests on — a coupling's
    trustworthy_result sits in the middle of a long reply, which is exactly the
    part that is cut. Re-reading the copy here made the run list and the
    downloaded record say a run computed nothing while its own transcript said
    it had finished: one fact, two readers, two answers."""
    seen = None
    for e in events:
        if e.get("type") == "tool_result" and e.get("tool") in SOLVER_TOOLS:
            v = e.get("verdict") or classify_solver_result(e.get("result") or "")
            if v == "verified":
                return v
            if v == "unverified" or seen is None:
                seen = v
    return seen


def solver_ran(events) -> bool:
    """True when some solver tool produced a result, verified or not."""
    return solver_verdict(events) in ("verified", "unverified")


def clean_outcome(events) -> str:
    """The outcome of a turn that ended without an error."""
    v = solver_verdict(events)
    return COMPLETED if v == "verified" else UNVERIFIED if v == "unverified" else NO_RESULT


def _turns(events):
    """Split the log into turns. A turn starts at `turn_start` (or, in records
    written before that event existed, at `user_msg`)."""
    turns, cur, has_marker = [], [], any(e.get("type") == "turn_start" for e in events)
    starter = "turn_start" if has_marker else "user_msg"
    for e in events:
        if e.get("type") == starter and cur:
            turns.append(cur)
            cur = []
        cur.append(e)
    if cur:
        turns.append(cur)
    return turns


def fold(events, *, live_default: str = RUNNING) -> str:
    """How the latest turn of a run ended, from its event log.

    Each turn is judged on its own. An error decides the turn, and an error that
    carries no outcome of its own is a failure. A clean end is judged by the
    solver evidence, never by what the log's `done` event claimed: records
    written before today said "completed" for turns that computed nothing."""
    turns = _turns(events)
    if not turns:
        return live_default
    turn = turns[-1]
    outcome = live_default
    decided = None                      # an error says how the turn ended
    for e in turn:
        t = e.get("type")
        if t == "error":
            decided = e.get("outcome") or FAILED
            outcome = decided
        elif t == "done":
            if decided:
                outcome = decided
                continue
            stated = e.get("outcome")
            outcome = stated if stated in (FAILED, INTERRUPTED, UNFINISHED) else clean_outcome(turn)
    return outcome
