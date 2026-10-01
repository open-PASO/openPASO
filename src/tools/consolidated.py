"""
Consolidated MCP tools for openPASO.

Reduces 48 tools → ~12 tools by combining related functionality.
Fewer tools = faster schema loading = faster agent response.
"""

import json
import os
import hashlib
import re
import time
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Context
from core.backend import detect_template_language, error_excerpt
from core.registry import get_backend, available_backends, all_backends
from core.fabrication_gate import inspect_result_artefacts, result_content_findings
from core.critic_gate import (CriticRegistry, CriticGateError,
                              setup_digest)
from core.quality_checks import check_result_files_finite, check_summary_finite
from core import pitfall_index

# Output locations come from core.output_paths, the single resolver — see
# the long note there. Keeping a private copy here is exactly how this
# defect survived its first repair.
from core.output_paths import output_dir as _output_dir
_OUTPUT_DIR = _output_dir("simulation_outputs")
_COUPLING_DIR = _output_dir("coupling")
FOURC_ROOT = Path(os.environ.get("FOURC_ROOT", ""))
_jobs: dict = {}

# HOE ablation toggle (MCP_NO_PITFALL_DB condition): when set, every
# knowledge surface strips pitfall-DB content so the agent operates as
# if the component were absent. "Pitfall DB" covers, in spirit:
#   - per-backend pitfall lists (incl. their Signal: failure anchors)
#   - general input-format pitfalls
#   - post-mortem records (the Signal-retrieval audit trail)
#   - the cross-backend collation catalog (backends/_cross.py)
# Off by default; never affects normal MCP usage.
_ABLATE_PITFALLS = os.environ.get("OFA_DISABLE_PITFALLS", "0") == "1"

_PITFALL_KEYS = ("pitfalls", "notes", "pitfall_db_entries",
                 "general_pitfalls", "common_pitfalls")

# Names that all reach the install / setup / build-config surface
# (backends/_setup.py). A caller who needs it is by definition one who
# has NOT got the backend working, so several plausible words are
# accepted rather than one canonical string.
_SETUP_TOPIC_ALIASES = frozenset({
    "install", "installation", "setup", "install_guide",
    "dependencies", "deps", "build_config", "build", "portability",
    "environment", "env",
})

# The MANDATORY pre-execution critic is unconditional.
#
# An OFA_DISABLE_CRITIC environment ablation used to lift it, stamping an
# unreviewed run VERIFIED. That is removed. An environment variable that turns a
# mandatory gate off is a bypass, and a gate with a bypass cannot support the
# claim that openPASO results are critic-reviewed: anything that sets the variable
# — a stray export, a harness default, a copied shell script — silently
# converts every verdict into an unreviewed one that still reads as VERIFIED.
# The comparison it existed for (running without the critic) is not run; the
# design is openPASO or no openPASO.

# The critic requirement was a boolean the AGENT passed: an audit showed a run
# stamped VERIFIED with critic_approved=True and no critic anywhere in the
# process. The server now keeps its own record. A review must be SUBMITTED for
# the exact setup being run, and the run tools consult this registry INSTEAD of
# trusting the flag.
_CRITIC_REGISTRY = CriticRegistry()


_PATHLIKE = re.compile(
    r"""["']([^"'\n]{3,200}?\.(?:py|yaml|yml|json|xml|msh|e|exo|vtu|vtk|dat|csv|txt|inp|prm|feb|mdpa))["']"""
)
_LOCAL_IMPORT = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", re.M)
_SYSPATH = re.compile(r"""sys\.path\.\w+\(\s*\d*\s*,?\s*["']([^"'\n]+)["']""")


def _at_sentence(text: str, limit: int) -> str:
    """`text` cut to at most `limit` characters at the end of a sentence; a cut
    with no sentence end in its second half ends at a word and says so."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    end = max(head.rfind(". "), head.rfind(".\n"))
    if end >= limit // 2:
        return head[:end + 1]
    return head[:head.rfind(" ")] + " [...]" if " " in head else head


def _cut_outside_commands(text: str, limit: int) -> str:
    """`text` cut to at most `limit` characters at a sentence or line end that lies outside every
    bracket -- never inside a served command such as spawn_subagent(...) or couple_levels(...).
    Measured: every level reply of a set of runs was cut at a flat 4000 characters, inside
    the served spawn command, in the middle of the run's own path. The cut is marked."""
    if len(text) <= limit:
        return text
    mark = " [cut here to fit this reply]"
    room = max(0, limit - len(mark))
    depth, best, first_open, ws = 0, -1, None, -1
    for i, ch in enumerate(text[:room]):
        if ch in "([{":
            if depth == 0:
                first_open = i
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
            if depth == 0:
                first_open = None
        if depth == 0:
            if ch == "\n" or (ch == "." and i + 1 < len(text) and text[i + 1] in " \n"):
                best = i + 1
            elif ch == " ":
                ws = i
    if best >= room // 4:
        return text[:best].rstrip() + mark
    cut = first_open if first_open is not None else ws
    cut = text.rfind(" ", 0, cut) if cut and cut > 0 else ws
    return (text[:cut].rstrip() if cut and cut > 0 else text[:room]) + mark


def _referenced_file_digest(setup_text: str) -> str:
    """Fingerprint the files a deck REFERENCES, not just the deck itself.

    THE BYPASS THIS CLOSES, which I demonstrated against my own gate. A deck may
    put its physics in a file it imports. The deck text then never changes, so
    the review digest matches, so the run verifies — while the equations are
    whatever the imported file now says. Measured: a reviewed deck importing
    `SOURCE = 1.0` returned max|u| = 0.0728 and VERIFIED; rewriting only the
    imported file to `SOURCE = 1000.0` returned 72.78 and VERIFIED again, with
    the same review. A thousandfold change in the physics through a review of
    something else.

    A sibling audit found the identical hole in `couple`, where the digest
    covered the participant COMMAND but never the contents of the script it
    named — a 225x different answer surviving its review. Same shape, two
    places, so it is the shape that has to be fixed rather than the instance.

    WHAT IS AND IS NOT COVERED, stated rather than implied. Covered: quoted
    paths with a known extension, local modules imported after a `sys.path`
    insertion or sitting next to the server's working directory, and the
    directories themselves so a shadowing module counts. An absent file is
    recorded AS absent, so creating it later invalidates the review — "write the
    file after review" is otherwise the same bypass again.

    NOT COVERED, AND NOT COVERABLE HERE. A coupling audit defeated the
    equivalent fix five ways and predicted this one would fall to the same
    routes; measured, it did. What remains open:

      * a module imported from a directory that does not exist yet at review
        time — a generator that writes its own helper into the job directory
        creates it AFTER the review, so no pre-run digest can see it;
      * `exec(open(path).read())`, a path from an environment variable,
        `os.path.join`, a glob — any path assembled at runtime.

    None of these can be closed by reading the deck, because closing them means
    knowing what the deck will do, which is the thing being gated. That is a
    real limit on the guarantee, so `_stamp_verification` states it in the
    VERDICT rather than here: a sibling audit found two limits that were
    documented only in a docstring and observed that this "is the same as
    nowhere". The agent reads the verdict.
    """
    # Roots to resolve a bare `from model import x` against. The first version
    # used ONLY `sys.path.insert` targets, so a deck with an ordinary sibling
    # import — no path manipulation, no adversarial intent — had NOTHING
    # fingerprinted. A coupling audit predicted this against my fix and it was
    # right: measured, the digest resolved only the output filename.
    roots = [Path(p) for p in _SYSPATH.findall(setup_text)]
    # The working directory resolves a bare `from model import x`, but it is
    # deliberately NOT hashed as a directory the way an explicit sys.path root
    # is. Hashing its listing made the digest depend on every unrelated file in
    # the cwd, so any file appearing between review and run invalidated the
    # review — a false-negative machine, and it turned four gate tests red the
    # moment I tried it. Only the named modules resolved against it are
    # fingerprinted.
    resolve_only = [Path.cwd()]
    seen: set[Path] = set()
    for raw in _PATHLIKE.findall(setup_text):
        seen.add(Path(raw))
        for r in roots:
            seen.add(r / raw)
    for mod in set(_LOCAL_IMPORT.findall(setup_text)):
        for r in roots + resolve_only:
            seen.add(r / f"{mod}.py")
            seen.add(r / mod / "__init__.py")
    # Explicit sys.path roots ARE hashed as directories, so adding a shadowing
    # module to one counts. The cwd is not, for the reason above.
    for r in roots:
        seen.add(r)

    parts: list[str] = []
    for p in sorted(seen, key=str):
        try:
            if p.is_file():
                parts.append(f"{p}={hashlib.sha256(p.read_bytes()).hexdigest()}")
            elif p.is_dir():
                # A directory's listing, so adding a shadowing module counts.
                names = sorted(x.name for x in p.iterdir() if x.is_file())
                parts.append(f"{p}/=" + hashlib.sha256(
                    "\n".join(names).encode()).hexdigest())
            else:
                parts.append(f"{p}=ABSENT")
        except OSError:
            parts.append(f"{p}=UNREADABLE")
    return "\n".join(parts)


def review_digest(solver: str, setup_text: str) -> str:
    """THE definition of what a critic review is bound to. One function.

    There were three places computing this — the issuing tool, the redeeming
    check, and a test helper — and they drifted. A coupling audit identified
    exactly that divergence as the root cause of its digest bypass: the sets of
    inputs each site folded in had stopped matching. Then my own test helper
    turned out to be a third copy and went stale the moment the definition
    changed, which is how a green suite can accompany a broken gate.

    Anything that needs to know what a review covers calls this.

    TWO BRANCHES BUILT A FILE BINDING AND THEY MUST NOT BE LAYERED.
    feature/anti-fabrication binds a single-solver setup by SCANNING the text
    for paths and hashing what it finds (`_referenced_file_digest`).
    feature/coupling-robustness binds a coupling by fingerprinting the
    participant spec's files EXPLICITLY and embedding the result in the setup
    text itself, under `__participant_files__` (see `_coupling_setup_text` and
    `_DIGEST_SCOPE_LIMIT`).

    Running both over a coupling made the gate unusable rather than stricter:
    the path scan also picks up each participant's `work_dir`, which the run
    itself writes into, so the digest computed when the review was issued no
    longer matched the digest computed after the coupling ran and every
    correctly-reviewed coupling came back NOT VERIFIED. That is a false
    negative in a gate whose whole job is to be believed.

    So when the text already carries an explicit file binding, that binding IS
    the answer and the scan is skipped. Nothing is unbound: a rewritten
    participant script still changes `__participant_files__`, which is exactly
    what test_coupling_robustness's fingerprint tests check.
    """
    if "__coupling_setup__" in setup_text:
        return setup_digest(solver, setup_text)
    return setup_digest(solver, setup_text, _referenced_file_digest(setup_text))


def _open_concerns(rec) -> str:
    """The critic points the review left open, named in the verdict.

    The served rule stops a review loop after two blocking rounds and tells the critic to
    submit what is still disputed as lines beginning "UNRESOLVED:" -- measured, a loop that
    never yields spends the whole run (four rounds, two of them contradicting each other, no
    solver called). Ending the loop is only honest if the disagreement survives into the
    result, so the verdict names it rather than reading as a clean approval.
    """
    lines = [ln.strip()[len("UNRESOLVED:"):].strip()
             for ln in (getattr(rec, "findings", "") or "").splitlines()
             if ln.strip().upper().startswith("UNRESOLVED:")]
    if not lines:
        return ""
    shown = "; ".join(lines[:3]) + (f" (+{len(lines) - 3} more)" if len(lines) > 3 else "")
    return f"; the critic left {len(lines)} point(s) open: {shown}"


def _earlier_rejections(rec, records) -> str:
    """How many times a critic turned down earlier versions of the file this approval names."""
    name = getattr(rec, "source_name", "")
    if not name:
        return ""
    n = sum(1 for r in records if getattr(r, "rejected", False) and getattr(r, "source_name", "") == name
            and r.digest != rec.digest and r.created <= rec.created)
    return (f"; approved after {n} rejection{'s' if n != 1 else ''} of earlier versions of {name}"
            if n else "")


def _critic_state(solver: str, setup_text: str, *, token: str = "",
                  job_id: str = "") -> tuple[bool, str]:
    """Has an independent critic reviewed THIS setup, on this server's record?

    Only a review submitted through `submit_critic_review` counts, and only for
    the exact deck being run: a digest mismatch means the input changed after it
    was reviewed, so "review a clean setup, then run a different one" is blocked.

    Two routes, both requiring a server-side record:

      * a TOKEN issued for this review — validated by the registry as known,
        unexpired, UNUSED, and matching this solver and this deck. Single-use,
        so it also blocks replaying one review across many runs.
      * no token — the deck is matched against submitted reviews by digest. This
        still proves a review of this exact setup exists; it does NOT bound how
        many runs one review covers. Callers that want that bound pass a token.

    What neither route can do is judge whether the critique was any GOOD; the
    server is not an oracle for review quality. It enforces that a substantive
    review of this setup happened and is auditable, which is the part that can
    be enforced in software rather than requested in a prompt.

    The agent's own `critic_approved` flag is deliberately not an input here: it
    is a self-report, and replacing the self-report is the entire point.
    """
    digest = review_digest(solver, setup_text)
    live = [r for r in _CRITIC_REGISTRY.records() if r.solver == solver and not r.expired()]
    turned_down = [r for r in live if getattr(r, "rejected", False) and r.digest == digest]
    if turned_down:
        # THE SAME TEXT CANNOT BE BOTH REJECTED AND APPROVED. A later critic's approval of
        # text an earlier critic turned down is not a second opinion that wins; it is the
        # first opinion asked again. The text has to change (a fix, or the rebuttal written
        # into it) and be reviewed as changed.
        first = (turned_down[-1].findings.strip().splitlines() or [""])[0][:200]
        return False, (f"a critic REJECTED this exact setup ({first}); it stays NOT VERIFIED "
                       f"until the text changes and a critic approves the changed text")
    approvals = [r for r in live if not getattr(r, "rejected", False)]
    if token:
        try:
            rec = _CRITIC_REGISTRY.consume(token, digest=digest, solver=solver,
                                           job_id=job_id)
            return True, ("reviewed (critic token redeemed; single use)"
                          + _open_concerns(rec) + _earlier_rejections(rec, live))
        except CriticGateError as exc:
            return False, f"critic review token refused: {exc}"
    for rec in approvals:
        if rec.digest == digest:
            return True, ("reviewed (submitted review matches this setup)" + _open_concerns(rec)
                          + _earlier_rejections(rec, live))
    near = approvals
    if near:
        # SAY WHICH PART CHANGED. "The input changed after it was reviewed" is
        # true and unusable: the caller has a participant list, seven scalar
        # arguments and the CONTENTS of every script in it, and is told that
        # one of them moved. Measured 2026-09-19: ten of ten coupled runs were
        # told exactly this and not one of them recovered the binding, because
        # nothing in the reply -- or in the record afterwards -- said what to
        # look at. The setup text is the caller's own arguments, so naming the
        # keys that differ gives nothing away.
        return False, ("a critic review exists for this solver but NOT for this "
                       "setup: the input changed after it was reviewed"
                       + _setup_difference(near, setup_text))
    return False, "no critic review is on record for this setup"


# A FILE NAME IS NOT A SETUP. Measured 2026-09-30 in the web interface: 14 of the 15 reviews
# filed on one run passed the script's FILE NAME as `setup` ("cylinder_final.py"). Each was
# accepted and bound to the digest of those few characters, which no run can match, so every
# run came back "the input changed after it was reviewed": a review bound to nothing, and a
# diagnosis that was not true. A name of a file in the run folder now stands for the file's
# text, and a name with no such file is refused before anything is recorded.
_A_FILE_NAME = re.compile(r"^[\w.~/+-]+\.[A-Za-z0-9]{1,10}$")
_SETUP_FILE_LIMIT = 20_000_000


def _text_of_named_file(raw: str) -> tuple[str | None, str | None]:
    """(text, error) when `raw` names a file instead of holding text; (None, None) when it
    is text, to be used as it is."""
    s = (raw or "").strip()
    if not s or "\n" in s or len(s) > 400:
        return None, None
    from tools.result_audit import resolve_under_cell
    path = resolve_under_cell(s)
    try:
        is_file = path.is_file()
    except OSError:
        is_file = False
    if not is_file:
        if _A_FILE_NAME.match(s):
            return None, (f"`{s}` names a file, and no such file is in this run's folder. Pass "
                          "the name of a file that exists, or the exact text that will run.")
        return None, None
    # INSIDE THE RUN FOLDER ONLY, links followed. The server may be able to read files the
    # model cannot (a sandboxed cell): a run of such a file prints its lines in the traceback.
    import os as _os
    root = Path(_os.environ.get("OPENPASO_CELL_WORKDIR") or Path.cwd()).resolve()
    real = path.resolve()
    if real != root and root not in real.parents:
        return None, (f"`{s}` is outside this run's folder (or links out of it). A review and a "
                      "run name a file inside the run folder; pass the exact text instead.")
    try:
        if path.stat().st_size > _SETUP_FILE_LIMIT:
            return None, f"`{s}` is larger than {_SETUP_FILE_LIMIT:,} bytes; pass a smaller file"
        return path.read_text(encoding="utf-8"), None
    except UnicodeDecodeError:
        return None, f"`{s}` is not a text file, so it cannot be a script or a deck"
    except OSError as exc:
        return None, f"`{s}` could not be read: {exc}"


def _run_text(text_arg: str, path_arg: str, text_name: str, path_name: str):
    """(text, error) for a run tool given its input as text or as a file in the run folder.

    Running the file is how a run matches the review of that file: the review is bound to the
    file's text, and a copy retyped into the call is a different setup wherever it differs."""
    if path_arg and text_arg:
        return None, f"pass {text_name} or {path_name}, not both"
    if not path_arg:
        if not text_arg:
            return None, (f"pass {text_name} (the text to run) or {path_name} (the name of a file "
                          f"in the run folder that holds it)")
        return text_arg, None
    text, err = _text_of_named_file(path_arg)
    if err:
        return None, err
    if text is None:
        return None, f"{path_name} names no file in the run folder: `{path_arg.strip()[:120]}`"
    return text, None


def _text_difference(records, setup_text: str) -> str:
    """Where a script differs from the closest review on record: its first changed line.

    Measured 2026-09-29/30 in the web interface: a model retyped a reviewed script into the
    run call and turned its time-derivative term `(a0 / dt) * ufl.inner(u, v)` into
    `(a0 / dt) * ufl.inner(ufl.grad(u), ufl.grad(v))`. The run was told only that "the input
    changed after it was reviewed", and it reported a drag coefficient of 53,288 at every
    step. The line that changed is the caller's own text, so naming it gives nothing away."""
    import difflib
    now = (setup_text or "").splitlines()
    best = None
    for rec in records:
        was = (getattr(rec, "setup_text", "") or "").splitlines()
        if not was:
            continue
        ops = [o for o in difflib.SequenceMatcher(a=was, b=now, autojunk=False).get_opcodes()
               if o[0] != "equal"]
        changed = sum(max(i2 - i1, j2 - j1) for _t, i1, i2, j1, j2 in ops)
        if ops and (best is None or changed < best[0]):
            best = (changed, was, ops[0])
    if best is None:
        return ""
    changed, was, (_t, i1, _i2, j1, _j2) = best

    def shown(lines, k):
        if k >= len(lines):
            return "(nothing: the text ends before this line)"
        t = lines[k].strip() or "(an empty line)"
        return t if len(t) <= 140 else t[:140] + " …"
    return (f" — the closest review on record is of a text that differs from this run's first at "
            f"its line {j1 + 1}: reviewed `{shown(was, i1)}`, this run `{shown(now, j1)}` "
            f"({changed} line(s) differ). Run the reviewed file itself (input_path / "
            f"generator_path) rather than a copy of it.")


def _setup_difference(records, setup_text: str) -> str:
    """Which keys differ between the closest review on record and this call."""
    try:
        now = json.loads(setup_text)
    except (TypeError, ValueError):
        return _text_difference(records, setup_text)
    if not isinstance(now, dict):
        return _text_difference(records, setup_text)
    best, best_diff = None, None
    for rec in records:
        try:
            was = json.loads(getattr(rec, "setup_text", "") or "")
        except (TypeError, ValueError):
            continue
        diff = sorted(k for k in set(was) | set(now) if was.get(k) != now.get(k))
        # A TIE GOES TO THE LATER REVIEW: the records come oldest first, and the
        # oldest of two equally close reviews was named (measured).
        if diff and (best_diff is None or len(diff) <= len(best_diff)):
            best, best_diff = was, diff
    if not best_diff:
        return ""
    parts = []
    for key in best_diff[:4]:
        if key == "__participant_files__":
            # the most common one, and the least obvious: a script was edited
            # after the review, so the review is of code that no longer runs
            changed = sorted(
                f"{name}/{fn}"
                for name, files in (now.get(key) or {}).items()
                for fn, h in files.items()
                if (best.get(key) or {}).get(name, {}).get(fn) != h)
            # A FILE THE REVIEW NEVER SAW DID NOT CHANGE. Measured: a review named its
            # work_dir "$(pwd)/side_A", which is not expanded here, so it found no
            # script at all, and this said the scripts' CONTENTS changed after it.
            unseen = sorted(
                f"{name}/{fn}"
                for name, files in (now.get(key) or {}).items()
                for fn, h in files.items()
                if (best.get(key) or {}).get(name, {}).get(fn) == "absent" and h != "absent")
            # A FILE THE REVIEWED SETUP NEVER NAMED is not a file that changed.
            absent_from_review = sorted(
                f"{name}/{fn}"
                for name, files in (now.get(key) or {}).items()
                for fn, h in files.items()
                if fn not in ((best.get(key) or {}).get(name) or {}))
            if unseen:
                parts.append("the review found no file at " + ", ".join(unseen[:3])
                             + " (the work_dir it named held none), so it is not a review "
                               "of these scripts")
            elif absent_from_review and len(absent_from_review) == len(changed):
                parts.append(", ".join(absent_from_review[:3]) + " was not in the reviewed setup")
            elif changed:
                parts.append("the CONTENTS of " + ", ".join(changed[:3])
                             + " changed after the review")
            else:
                parts.append("the participant files changed after the review")
            continue
        was_v, now_v = best.get(key), now.get(key)
        parts.append(f"{key}: reviewed {was_v!r}, now {now_v!r}"
                     if len(f"{was_v!r}{now_v!r}") < 160 else
                     f"{key} differs")
    return " — " + "; ".join(parts)


def _attest_run_quantities(work_dir, job_id: str) -> dict:
    """Compute the run's headline numbers from the run's OWN data output.

    An audit demonstrated the gap this closes: the gate bound its verdict to the
    RUN but never to a NUMBER, so a plausible invented value attached to a real,
    clean run passed everything. Nothing recomputed it, because nothing could —
    the number existed only in the agent's narration.

    So openPASO computes them itself, from the solver's data artefacts and never
    from anything the agent wrote: a value is derived from the mesh and nodal
    field the run actually produced, and carries the file it came from and that
    file's hash. The agent no longer has to state a number, which is the point —
    fabrication-by-assertion stops being a thing an agent can usefully do,
    because the authoritative value is already in the result next to its
    provenance.

    This is not the same as making fabrication impossible. It binds a number to
    a FILE, not to a PROBLEM: a field that is well-formed but does not solve the
    stated equations still attests fine. Separating solving from forging needs
    the discrete residual (core/residual_check.py), which requires the problem's
    source term and is therefore opt-in per run.

    Never raises: an unattestable run is reported as unattestable, with the
    reason, and the run still returns.
    """
    quantities = {}
    try:
        from core.attestation import AttestationError, attest_quantity
    except Exception as exc:                      # pragma: no cover
        return {"available": False,
                "why": f"attestation unavailable: {exc}"[:300]}
    for quantity in ("l2_norm", "max_abs"):
        try:
            att = attest_quantity(work_dir, job_id, quantity)
        except AttestationError as exc:
            quantities[quantity] = {"available": False, "why": str(exc)[:300]}
        except Exception as exc:
            quantities[quantity] = {
                "available": False,
                "why": f"could not be computed from the run's data: {exc}"[:300]}
        else:
            quantities[quantity] = {
                "available": True,
                "value": att.value,
                "field": att.field,
                "from_file": Path(att.source_file).name,
                "sha256": att.source_sha256[:16],
                "n_points": att.n_points,
                "computed_by": att.computed_by,
            }
    quantities["note"] = (
        "Computed by openPASO from this run's own data output. Report these "
        "rather than numbers read out of a script's print statements, and "
        "never a number you did not obtain from the run.")
    return quantities


def _check_declared_pde(spec: str, out_files) -> dict:
    """Does the field the run produced actually SOLVE the problem it declared?

    Every other check in this gate inspects the run: did it complete, did it
    write output, is that output finite, is the mesh structurally sane. All of
    them are satisfied by a field that is well-formed and wrong — an audit built
    one in eight lines that was MORE accurate than a genuine solve, ran 82x
    faster, and passed a mesh-independence study. As data it is impeccable. The
    only property that separates it from a solve is whether it satisfies the
    equations, and that is what this measures.

    Opt-in, because it needs the problem's source term and openPASO cannot infer
    one. That is not a leak: f is the problem statement, not its solution, and a
    residual is computed from f alone. A gate that needed the answer could not
    verify a real engineering problem, where there isn't one.

    Never raises — a gate an agent can disable by malforming its input is not a
    gate.
    """
    try:
        from core.residual_gate import check_run_residual
        return check_run_residual(spec, out_files)
    except Exception as exc:                       # pragma: no cover
        return {"verdict": "REFUSED",
                "detail": f"the residual check could not run: {exc}"[:300]}


def _critic_coverage_note() -> str:
    """State what the review is NOT bound to, in the verdict itself.

    A review is bound to the deck plus the files the deck statically names. It
    is not bound to a module the deck loads by a runtime-constructed path, and
    it cannot be: knowing that would mean knowing what the deck does, which is
    the thing being gated. An audit defeated the equivalent coupling fix through
    exactly those routes and predicted this one would fall to them too — it did.

    Served rather than left in a docstring, because that same audit found two
    limits documented only in a docstring and noted this "is the same as
    nowhere". A limit an agent never reads is not a disclosure.
    """
    return ("SCOPE OF THE REVIEW: it is bound to this deck and to the files the "
            "deck names outright. It is NOT bound to a module loaded by a path "
            "built at run time — from an environment variable, a glob, "
            "`exec(open(...))`, or a directory created after the review — so "
            "physics moved into such a file is outside what the critic saw.")


def _residual_coverage_note(result: dict) -> str:
    """Say, in the verdict itself, whether anything checked that this output
    solves anything.

    The residual check is opt-in — it needs the problem's source term, which
    only the caller has. So a run that skips it still passes every other check
    and still reads VERIFIED. If the verdict said nothing, those two cases would
    be indistinguishable in the one place an agent actually looks, and the
    strongest check in the gate would quietly become optional in practice rather
    than in principle. Naming the gap is what keeps it a gap instead of a hole.
    """
    verdict = (result.get("residual_check") or {}).get("verdict")
    if verdict == "SOLVES":
        return ("The output also SATISFIES the equations the run declared "
                "(relative residual "
                f"{result['residual_check'].get('relative_residual'):.2e}), so "
                "it was obtained by solving them rather than merely being a "
                "well-formed field.")
    if verdict == "INCONCLUSIVE":
        return ("NOTE: whether this output solves the declared problem could "
                "NOT be established — its residual falls in the band where a "
                "loosely-converged solve and a very fine-mesh analytic field "
                "are indistinguishable ("
                + str(result["residual_check"].get("detail", ""))[:200]
                + "). It is neither certified nor rejected.")
    if verdict in ("UNSUPPORTED", "REFUSED"):
        return ("NOTE: openPASO could not check whether this output solves the "
                "declared problem ("
                + str(result["residual_check"].get("detail", ""))[:160]
                + "), so this verdict covers the run, not the physics.")
    if "exports" in result and "graph" in result:
        # A COUPLING DECLARES ITS PROBLEM IN EACH SIDE'S config.json, and is checked against it
        # across levels. Measured: a converged couple() reply told a run whose two configs
        # stated the source and k that it "declared no problem to check against".
        return ("NOTE: this single couple() call did not check each side's field against its "
                "own equation. couple_levels does, from each side's config.json and per-level "
                "dumps (from the second level on), and audit_results does at hand-in.")
    return ("NOTE: nothing here checked whether this output satisfies any "
            "equations — the run declared no problem to check against. Pass "
            "verify_pde with the problem's source term to have openPASO assemble "
            "it and measure the residual; a field that is finite, structurally "
            "sane and solves nothing passes every other check in this gate.")


def _residual_blocks_verification(result: dict) -> bool:
    """True only when the residual check positively established the field does
    not solve the declared problem. UNSUPPORTED and REFUSED must never block —
    they mean openPASO did not check, and 'not checked' is not evidence of guilt
    any more than it is evidence of innocence."""
    return (result.get("residual_check") or {}).get("verdict") == "DOES_NOT_SOLVE"


# Qualifiers a request can carry that MUST NOT be silently dropped, each with
# the variant-name tokens that satisfy it and the ones that contradict it.
_VARIANT_QUALIFIERS = [
    ("3d", ("3d", "three_d", "3D"), ("2d", "1d")),
    ("2d", ("2d", "two_d", "plane"), ("3d", "1d")),
    ("transient", ("transient", "unsteady", "time_dependent", "dynamic"),
     ("steady", "stationary", "static")),
    ("steady", ("steady", "stationary", "static"),
     ("transient", "unsteady", "dynamic")),
    ("nonlinear", ("nonlinear", "non_linear"), ()),
    ("cylinder", ("cylinder",), ()),
]
_QUALIFIER_WORDS = {
    "3d": ("3d", "three-dimensional", "three dimensional"),
    "2d": ("2d", "two-dimensional", "plane stress", "plane strain"),
    # a wake that sheds vortices has no steady state: asking for the shedding, the
    # Strouhal number or a vortex street is asking for an unsteady run
    "transient": ("transient", "unsteady", "time-dependent", "time dependent",
                  "time-varying", "evolving", "vortex shedding", "shedding", "strouhal",
                  "vortex street", "wake"),
    "steady": ("steady", "stationary", "static", "steady-state"),
    "nonlinear": ("nonlinear", "non-linear", "large deformation", "finite strain"),
    "cylinder": ("cylinder", "obstacle", "bluff body"),
}


def _select_template_variant(query: str, variants: list[str]) -> tuple[str, str]:
    """Choose the template variant the request actually asked for.

    THE BUG THIS FIXES. Three call sites read `template_variants[0]` and nothing
    else, so `prepare_simulation(solver, "3d linear elasticity")` returned the
    2D plane-stress template and said nothing about it. A usability measurement
    found this on four of six realistic tasks: a correct deck already existed as
    a working generator and no tool could reach it. That is not a knowledge gap
    — adding prose cannot fix it, because a weak model handed a deck labelled
    "2D (plane stress)" for a 3D task will ship the 2D deck whatever text sits
    above it.

    Returns (variant, note). The note is never empty: it always names the
    alternatives, and when the request carried a qualifier that NO variant
    satisfies it says so in those words rather than substituting quietly.
    Substituting quietly is the failure mode — a weak model cannot detect it.
    """
    if not variants:
        return "", ""
    q = (query or "").lower()
    # WHOLE WORDS: "steady" is inside "unsteady", and a request for an unsteady run
    # was also read as a request for a steady one -- and told that no variant is
    # steady, so "adapt it rather than running it as-is".
    asked = [name for name, words in _QUALIFIER_WORDS.items()
             if any(re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", q) for w in words)]

    # THE VARIANT THAT SATISFIES THE MOST OF WHAT WAS ASKED. Each qualifier used to
    # overwrite the choice of the one before, so "an unsteady wake behind a cylinder"
    # ended on whichever cylinder variant came first -- the steady one -- although a
    # variant satisfying both was in the list. Ties keep catalog order.
    def _fits(v: str, name: str) -> bool:
        satisfies, contradicts = next((s, c) for n, s, c in _VARIANT_QUALIFIERS if n == name)
        if any(t in v.lower() for t in contradicts):
            return False
        if name == "steady" and "steady" not in v.lower():
            # a variant whose name says nothing about time is a steady one (channel_cylinder
            # is steady, and the transient twin says so in its name)
            return True
        return any(t in v.lower() for t in satisfies)
    chosen = max(variants, key=lambda v: sum(_fits(v, name) for name in asked))
    # what the CHOSEN variant does not provide, and whether another one does: a
    # request for "unsteady 3d flow" was told `3d` was selected "for: 3d, transient"
    missed = [name for name in asked if not _fits(chosen, name)]
    elsewhere = {name: [v for v in variants if v != chosen and _fits(v, name)] for name in missed}

    bits = []
    nowhere = [u for u in missed if not elsewhere[u]]
    if nowhere:
        bits.append(
            "⚠ You asked for " + " and ".join(f"**{u}**" for u in nowhere)
            + f", and no template variant provides it. Serving `{chosen}`, "
            f"which does NOT satisfy that — adapt it rather than running it "
            f"as-is.")
    for u in missed:
        if elsewhere[u]:
            bits.append(f"⚠ You asked for **{u}**, and `{chosen}` does NOT satisfy it: "
                        f"`{elsewhere[u][0]}` does -- request it by name.")
    satisfied = [name for name in asked if name not in missed]
    if satisfied:
        bits.append(f"Selected `{chosen}` for: {', '.join(satisfied)}.")
    if len(variants) > 1:
        others = [v for v in variants if v != chosen]
        bits.append(f"Other variants available: {', '.join(others)} — request "
                    f"one by name via `examples(action='template', "
                    f"variant='<name>')`.")
    return chosen, (" ".join(bits))


# Keys an agent needs FIRST to write a working input. Matched as substrings, so
# `required_sections`, `required_keys` and `requires` all hit "required".
_LOAD_BEARING_KEYS = (
    "description", "required", "section", "element", "space", "material",
    "weak_form", "code_skeleton", "skeleton", "syntax", "boundary", "bc",
    "solver", "deck", "commands", "start_here", "example", "template",
    "time_integration", "units",
)


def _fit_json_payload(payload: dict, limit: int) -> tuple[str, list[str]]:
    """Serialise as much of `payload` as fits, ALWAYS as valid JSON.

    Slicing a serialised dict at a character count is how this broke: three
    SPARTA physics were returning a `## Knowledge` block cut mid-string, so
    `json.loads` failed on what the agent received. Not merely losing content —
    handing a weak model something it cannot parse at all. And it gets worse as
    the knowledge grows, so every expansion was making the problem it was meant
    to fix slightly harder.

    Whole keys are dropped instead, load-bearing ones last, and the caller is
    told which went so it can say so rather than leaving a silent hole.
    """
    text = json.dumps(payload, indent=2, default=str)
    if len(text) <= limit:
        return text, []

    def rank(key: str) -> int:
        k = str(key).lower()
        return 0 if any(t in k for t in _LOAD_BEARING_KEYS) else 1

    ordered = sorted(payload.items(), key=lambda kv: (rank(kv[0]), len(
        json.dumps(kv[1], default=str))))
    kept: dict = {}
    dropped: list[str] = []
    for key, value in ordered:
        trial = dict(kept)
        trial[key] = value
        if len(json.dumps(trial, indent=2, default=str)) <= limit:
            kept = trial
        else:
            dropped.append(str(key))
    # Preserve the original key order among those that survived, so the payload
    # reads the way its author wrote it rather than in size order.
    kept = {k: v for k, v in payload.items() if k in kept}
    return json.dumps(kept, indent=2, default=str), dropped
def _file_fingerprint(path: Path) -> str:
    """Content hash of one file, or size+mtime when it is too big to hash cheaply.

    Either way a change is visible, which is all the digest needs.
    """
    import hashlib
    try:
        st = path.stat()
        if st.st_size > 4 << 20:
            return f"size:{st.st_size}:mtime:{st.st_mtime_ns}"
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as e:
        return f"unreadable:{type(e).__name__}"


def _participant_fingerprints(participants_json: str, monolithic_json: str = "") -> dict:
    """Fingerprint the FILES a coupling actually executes.

    THE BYPASS THIS CLOSES. A participant spec names a command
    (["python", "run.py"]) and a work_dir; the physics lives entirely in
    run.py, which the digest never saw. So a review of one coupling approved
    any other coupling that reused the same file names: reviewing a correct
    setup and then rewriting the participant script produced a completely
    different answer, stamped VERIFIED, with the note "submitted review matches
    this setup". Demonstrated with a 225x change in the coupled result.

    Every command token that resolves to an existing file (absolute, or
    relative to that participant's work_dir) is fingerprinted, as are its
    declared data_files and the monolithic reference command. A file that does
    not exist yet is recorded as absent, so creating it afterwards changes the
    digest — a participant whose script does not exist cannot have been
    reviewed.

    WHAT THIS STILL DOES NOT COVER, established by running it. The fingerprint
    reaches exactly the paths the SPEC NAMES. Anything a script opens at RUNTIME
    is invisible, and rewriting it leaves the digest identical while the coupled
    answer changes completely. Four routes were demonstrated, each taking a
    reviewed x=2.666667 to x=334.666665 with the verdict still reading VERIFIED,
    "a critic review of this exact setup is on record":

      * `from model import step` — an ordinary helper module beside run.py. No
        trickery at all, and the most likely shape of real code;
      * `exec(open("physics.py").read())`;
      * a path built from an environment variable, or by `os.path.join`;
      * a path found by `glob`.

    A symlink repointed at different content IS caught (the hash follows the
    link), as is a rewritten `data_files` entry and a rewritten monolithic
    reference. Closing the rest needs the participants' whole working trees
    fingerprinted, or an import trace; neither is done here. So `couple` states
    the scope in its served coverage (`_DIGEST_SCOPE_LIMIT`) rather than letting
    VERIFIED imply more than the digest supports, and declaring such a file in
    `data_files` is the supported way to bring it inside.
    """
    out: dict = {}
    try:
        specs = json.loads(participants_json) if participants_json else []
    except (json.JSONDecodeError, TypeError):
        return out
    if monolithic_json:
        try:
            m = json.loads(monolithic_json)
            if isinstance(m, dict):
                specs = list(specs) + [dict(m, name="__monolithic__")]
        except (json.JSONDecodeError, TypeError):
            pass
    if not isinstance(specs, list):
        return out
    for s in specs:
        if not isinstance(s, dict):
            continue
        name = str(s.get("name", "?"))
        wd = Path(str(s.get("work_dir", "."))).expanduser()
        seen: dict = {}
        # THE INTERPRETER IS NOT PART OF THE SETUP BEING REVIEWED.
        #
        # command[0] is an executable, normally an absolute path to a Python
        # that belongs to the machine rather than to the run. Hashing it put a
        # third-party binary inside the review digest, so an unrelated upgrade
        # would invalidate every review on the box -- and, because the miss
        # path joins the token onto the work_dir, it rendered as
        # `A//home/<user>/.../bin/python` with a doubled slash. Measured, C2
        # one run: "the CONTENTS of
        # A//home/<user>/<venv>/bin/python,
        # A/participant_A.py changed after the review".
        #
        # A command[0] INSIDE the participant's own directory is a different
        # thing -- a launcher script the caller wrote -- and stays in the
        # digest, which is why this tests the location rather than the index.
        _cmd = [str(t) for t in (s.get("command") or [])]
        if _cmd:
            try:
                _first = Path(_cmd[0]).expanduser()
                if _first.is_absolute() and not _first.is_relative_to(wd):
                    _cmd = _cmd[1:]
            except (OSError, ValueError):
                pass
        for token in _cmd + list(s.get("data_files") or []):
            tok = str(token)
            for cand in (Path(tok).expanduser(), wd / tok):
                try:
                    if cand.is_file():
                        seen[tok] = _file_fingerprint(cand)
                        break
                except OSError:
                    continue
            else:
                # Only record a miss for tokens that LOOK like files, so option
                # flags do not turn into noise that changes with argv order.
                if "/" in tok or "." in tok:
                    seen[tok] = "absent"
        if seen:
            out[name] = seen
    return out


def _coupling_work_dir_faults(parsed: dict) -> list:
    """Participants whose work_dir `couple` would refuse, named at review time.

    THE SAME RULE AS THE RUN DOOR, and deliberately no stricter: absolute, and
    inside this task's tree when the server was told what that tree is. A check
    here that refused something `couple` accepts would be worse than none --
    the agent would be unable to review a setup it can legally run.
    """
    import os as _os
    cell = _os.environ.get("OPENPASO_CELL_WORKDIR")
    root = Path(cell).resolve() if cell else None
    raw = parsed.get("participants")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []
    faults = []
    for spec in raw:
        if not isinstance(spec, dict):
            continue
        name = str(spec.get("name", "?"))
        wd = str(spec.get("work_dir", "") or "")
        if not wd:
            continue
        path = Path(wd)
        if not path.is_absolute():
            # THE THIRD DOOR WITH THIS RULE, AND THE RULE HAD DRIFTED AGAIN.
            # `couple` and `couple_levels` resolve a relative work_dir against
            # the declared task directory; this one still refused it, so a
            # setup spelled the way the served example shows could be RUN and
            # not REVIEWED. One rule, stated once, in each of the three places
            # that must agree.
            if root is None:
                faults.append(f"participant {name}: work_dir must be an "
                              f"ABSOLUTE path, got {wd!r} — nothing has told "
                              f"this server which directory a relative path "
                              f"would be relative TO")
            continue
        if root is not None and not path.resolve().is_relative_to(root):
            faults.append(f"participant {name}: work_dir {path.resolve()} is "
                          f"outside this task's working directory {root}, so "
                          f"the run you review here is not a run that can be "
                          f"made")
    return faults


def _coupling_setup_text(**kwargs) -> str:
    """Canonical text a coupling review is bound to.

    ONE definition, used by the coupling tools and by `submit_critic_review`, so
    the digest a review is issued for and the digest a run is checked against
    cannot drift apart. Every argument that changes what is solved belongs in
    here — including the CONTENTS of the participant scripts, which is what the
    coupling tools actually execute and what the spec only names.
    """
    payload = dict(kwargs)
    # Marks the text as a COUPLING setup whose file binding is explicit (the
    # `__participant_files__` fingerprints below). `review_digest` keys off this
    # to skip its path-scanning binding, which double-binds and, because it also
    # picks up each participant's work_dir, changes after the run and fails a
    # correctly-reviewed coupling. The marker rather than the fingerprints
    # themselves, so the rule still holds when a caller has no files to
    # fingerprint or a test strips them out.
    # ONE SPELLING PER SETUP, OR THE GATE CANNOT BE SATISFIED.
    #
    # The digest is a hash of this text, so two spellings of the SAME setup are
    # two setups. Measured 2026-09-19: `couple_levels` builds its digest with
    # `monolithic=""`, while THIS SERVER'S OWN `coupling_args_example` tells the
    # caller to send `"monolithic": false, "probe": null`. Of the twelve
    # combinations of the three falsy spellings for those two keys, exactly ONE
    # matches what `couple_levels` computes, and it is not the one we document.
    # The example also shows `participants` as a LIST while the tool takes the
    # JSON STRING, which is a second guaranteed mismatch on its own.
    #
    # Consequence, measured over every coupled run on this machine that reached
    # `couple_levels`: 10 of 10 were served "THIS LADDER IS NOT VERIFIED ... a
    # critic review exists for this solver but NOT for this setup" AFTER
    # submitting a review this server had accepted. So no coupled result in the
    # record carries a verification verdict, and the reason was our own example.
    #
    # Canonicalising here rather than at each door: absent, empty, null and
    # false all mean "no monolithic reference" and "no probe", so they must
    # hash alike; and a participant spec means the same thing whether it
    # arrives as a list or as the string of that list. This only MERGES
    # digests that already meant the same setup -- it can never make two
    # different setups collide, because the canonical form is a lossless
    # re-serialisation.
    for _key in ("monolithic", "probe"):
        if not payload.get(_key):
            payload[_key] = ""
    _specs = payload.get("participants")
    if _specs not in (None, "", []):
        try:
            _parsed = json.loads(_specs) if isinstance(_specs, str) else _specs
            # A work_dir MEANS THE SAME DIRECTORY whether it is written
            # relative to the task or in full, and `couple` now accepts both.
            # Left as typed, the two spellings are two setups: measured
            # 2026-09-19, the relative form also makes the fingerprinter
            # record every participant script as 'absent', because it looks
            # for them beside the SERVER. Two digests, one setup, and a
            # verification gate nothing can satisfy.
            if isinstance(_parsed, list):
                import os as _os
                _cell = _os.environ.get("OPENPASO_CELL_WORKDIR")
                for _spec in _parsed:
                    if not isinstance(_spec, dict):
                        continue
                    # A TIME LIMIT DOES NOT CHANGE WHAT IS SOLVED. Measured: a
                    # review submitted with "timeout" in each spec and a ladder
                    # called without it were two setups, and the ladder was told
                    # no review of it was on record.
                    _spec.pop("timeout", None)
                    _wd = str(_spec.get("work_dir", "") or "")
                    if _cell and _wd and not Path(_wd).is_absolute():
                        _spec["work_dir"] = str(Path(_cell) / _wd)
            payload["participants"] = json.dumps(_parsed, sort_keys=True,
                                                 separators=(",", ":"))
        except (TypeError, ValueError):
            payload["participants"] = str(_specs)
    for _key, _cast in (("max_iter", int), ("tol", float), ("theta", float)):
        if payload.get(_key) is not None:
            try:
                payload[_key] = _cast(payload[_key])
            except (TypeError, ValueError):
                pass
    if isinstance(payload.get("accelerator"), str):
        payload["accelerator"] = payload["accelerator"].strip().lower()
    # THE DIGEST COVERS WHAT IS SOLVED, AND NOTHING ELSE.
    #
    # `submit_critic_review` took whatever keys the caller sent, and our own
    # docstring tells them to "pass EVERY key for the tool you will call".
    # `couple_levels` takes arguments that do not change what is solved --
    # where the residual history is written, what it is named, the level list
    # it resolves ONE review for -- and its digest covers only the participants
    # and the seven solver arguments. So a caller who follows the instruction
    # reviews a payload the run can never reproduce.
    #
    # MEASURED on one coupled round, in all five runs, and readable
    # only because the mismatch now names its keys:
    #
    #     the input changed after it was reviewed — history_dir: reviewed '.',
    #     now None; history_pattern: reviewed 'residual_level{k}.csv', now None
    #
    # Restricting here rather than at the door, because this one function is
    # what both sides hash: an unknown key cannot change the digest, so it
    # cannot break a binding, and a key that DOES change what is solved is in
    # the list below and still does.
    # AN ITERATION CAP DOES NOT CHANGE WHAT IS SOLVED, any more than a time limit
    # does: the answer is the tolerance's, and a run that hits the cap says it did
    # not converge. Measured: a review of couple(max_iter=100) and the ladder
    # couple() itself recommends, run with couple_levels' own default of 150, were
    # two setups, and a right ladder read "no critic review of this coupling setup
    # is on record (max_iter: reviewed 100, now 150)".
    # AND HOW THE ITERATION IS RELAXED DOES NOT CHANGE WHAT IS SOLVED EITHER. theta,
    # the accelerator and the relaxation decide how fast the fixed point is reached,
    # not which one: a review at theta 0.001 and a right ladder run at 0.5 were two
    # setups, and the ladder read NOT VERIFIED (measured).
    _COVERED = ("participants", "monolithic", "probe", "tol",
                "problem", "solver_a",
                "solver_b", "nx", "ny", "params", "data", "exchanges",
                "scheme", "dimensions", "max_time", "time_window",
                "convergence_tol", "mapping")
    payload = {k: v for k, v in payload.items()
               if k in _COVERED or k.startswith("__")}
    payload["__coupling_setup__"] = True
    if payload.get("participants") or payload.get("monolithic"):
        fp = _participant_fingerprints(str(payload.get("participants") or ""),
                                       str(payload.get("monolithic") or ""))
        if fp:
            payload["__participant_files__"] = fp
    return json.dumps(payload, sort_keys=True)


_DIGEST_SCOPE_LIMIT = (
    "review-to-run binding SCOPE: the review is bound to the participant spec "
    "and to the CONTENTS of every file the spec names (each command token that "
    "is a file, every `data_files` entry, the monolithic reference). It is NOT "
    "bound to files a script opens at runtime — a helper module it imports, a "
    "path built from an environment variable or by os.path.join, a file found by "
    "glob. Rewriting one of those changes the coupled answer and leaves this "
    "verdict's digest identical, so VERIFIED here does not certify that part of "
    "the setup was reviewed. Declare such files in `data_files` to bring them "
    "inside the digest.")


_MONOLITHIC_NOT_SUPPLIED = (
    "monolithic consistency: NOT CHECKED — no un-split reference solve was "
    "supplied. Every other check here is internal to the coupling: it can tell "
    "you the iteration converged, conserved and stayed finite, and all of that "
    "is true of a coupling in which both sides consistently use the wrong units "
    "or apply the interface condition with the wrong sign. If this problem can "
    "be solved un-split in ONE code, pass `monolithic` and openPASO will compare "
    "the two answers; that is the strongest verification available here and it "
    "needs no external benchmark.")


def _run_monolithic_check(monolithic: str, exports: dict,
                          rtol: float = 0.05) -> tuple[dict, list[str], list[str]]:
    """Re-solve the coupled problem un-split, in one code, and compare.

    `monolithic` is a JSON {"command":[argv...], "work_dir":str, "timeout":int}.
    The command must write <work_dir>/monolithic.json in InterfaceData shape,
    sampled on the same interface the participants export. Every participant's
    exported `values` is then compared against the monolithic field interpolated
    onto that participant's own interface coordinates.

    Returns (report, findings, checks_not_run). A monolithic solve that itself
    fails is reported as NOT CHECKED, never as agreement: the coupling is not
    guilty because its reference could not be produced, and it is not innocent
    either.
    """
    import subprocess
    import numpy as _np
    from core.quality_checks import check_monolithic_consistency

    if not (monolithic or "").strip():
        return {"status": "not supplied"}, [], [_MONOLITHIC_NOT_SUPPLIED]
    try:
        spec = json.loads(monolithic)
        cmd = list(spec["command"])
        wd = Path(spec["work_dir"])
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return ({"status": "bad spec", "detail": str(e)}, [],
                [f"monolithic consistency: NOT CHECKED — the `monolithic` spec "
                 f"could not be read ({e}); expected JSON with `command` and "
                 "`work_dir`."])
    wd.mkdir(parents=True, exist_ok=True)
    out = wd / "monolithic.json"
    if out.exists():
        out.unlink()
    try:
        p = subprocess.run(cmd, cwd=str(wd), capture_output=True, text=True,
                           timeout=int(spec.get("timeout", 3600)), stdin=subprocess.DEVNULL)
    except (subprocess.TimeoutExpired, OSError, ValueError) as e:
        return ({"status": "reference solve failed", "detail": str(e)[:200]}, [],
                [f"monolithic consistency: NOT CHECKED — the un-split reference "
                 f"solve did not complete ({type(e).__name__}: {str(e)[:120]})."])
    if p.returncode != 0 or not out.exists():
        return ({"status": "reference solve failed",
                 "returncode": p.returncode,
                 "detail": (p.stderr or p.stdout or "")[-300:]}, [],
                [f"monolithic consistency: NOT CHECKED — the un-split reference "
                 f"solve exited {p.returncode} / wrote no monolithic.json, so "
                 "there is nothing to compare the coupled answer against."])
    try:
        ref = json.loads(out.read_text())
        # KEEP THE COMPONENT AXIS. `.ravel()` here made every VECTOR reference
        # unusable: with N points and 2 components the flattened size is 2N
        # while the coordinate count is N, so the shape guard below concluded
        # "not enough coordinates to map one onto the other" and the strongest
        # check in this tool reported NOT CHECKED on every vector coupling.
        ref_co = _np.atleast_2d(_np.asarray(ref.get("coordinates", []), float))
        _rv = _np.asarray(ref["values"], float)
        ref_vals = (_rv.reshape(len(ref_co), -1) if len(ref_co)
                    else _rv.reshape(-1, 1))
    except Exception as e:
        return ({"status": "reference unreadable", "detail": str(e)[:200]}, [],
                [f"monolithic consistency: NOT CHECKED — monolithic.json could "
                 f"not be read as InterfaceData ({e})."])
    if ref_vals.size == 0 or not _np.all(_np.isfinite(ref_vals)):
        return ({"status": "reference invalid"}, [],
                ["monolithic consistency: NOT CHECKED — the reference solve's "
                 "values are empty or non-finite."])

    ref_field = str(ref.get("field_name", "") or "")
    report: dict = {"status": "checked", "reference_points": int(len(ref_vals)),
                    "reference_components": int(ref_vals.shape[1]),
                    "reference_field": ref_field}
    findings: list[str] = []
    # Nothing here can tell an independent un-split solve from a script that
    # echoes the coupled answer back: both write the same file. Agreement is
    # therefore only as good as the reference, and that has to be said, because
    # a reference that is wrong in the same way as the coupling turns the
    # strongest check in this tool into a rubber stamp.
    not_run: list[str] = [
        "monolithic reference INDEPENDENCE: openPASO ran the command it was given "
        "and compared the numbers; it cannot tell a genuine un-split solve from "
        "one that re-reads or reproduces the coupled answer. Agreement below is "
        "evidence only if the reference solves the problem on its own."]
    for name, ex in exports.items():
        co = _np.atleast_2d(_np.asarray(ex.get("coordinates", []), float))
        _v = _np.asarray(ex.get("values", []), float)
        vals = _v.reshape(len(co), -1) if len(co) and _v.size else _v.reshape(-1, 1)
        if vals.size == 0:
            not_run.append(f"monolithic consistency for {name}: it exported no values")
            continue
        # Compare like with like. Participants on the two sides of an interface
        # do not always export the same quantity (a Dirichlet-Neumann pair
        # exports the same temperature; a force/displacement pair does not), and
        # comparing a displacement against a temperature reference would report
        # a large, entirely meaningless disagreement.
        got_field = str(ex.get("field_name", "") or "")
        if ref_field and got_field and got_field != ref_field:
            not_run.append(
                f"monolithic consistency for {name}: it exports {got_field!r} "
                f"while the reference solve provides {ref_field!r}, so there is "
                "nothing to compare. Have the reference write the field this "
                "participant exports if you want it covered.")
            continue
        target = vals
        # A SCALAR REFERENCE CANNOT CORROBORATE A VECTOR COUPLING and vice
        # versa: with different component counts there is no correspondence to
        # compare, and flattening both would line u_x up against u_y.
        if ref_vals.shape[1] != target.shape[1]:
            not_run.append(
                f"monolithic consistency for {name}: it exports "
                f"{target.shape[1]} component(s) per point while the reference "
                f"provides {ref_vals.shape[1]}, so there is nothing to compare. "
                "Have the reference write the same number of components.")
            continue
        if len(ref_vals) == len(target):
            ref_at = ref_vals
        else:
            if ref_co.size == 0 or co.size == 0 or len(ref_co) != len(ref_vals):
                not_run.append(
                    f"monolithic consistency for {name}: the reference has "
                    f"{len(ref_vals)} point(s) and this participant exports "
                    f"{len(target)}, and there are not enough coordinates to map "
                    "one onto the other")
                continue
            from core.field_transfer import InterfaceData, interpolate_to_points
            ref_at = _np.asarray(interpolate_to_points(
                InterfaceData(coordinates=ref_co, values=ref_vals,
                              field_name=str(ref.get("field_name", "ref"))),
                co), float).reshape(len(co), -1)
        if ref_at.shape != target.shape:
            not_run.append(f"monolithic consistency for {name}: shapes did not align")
            continue
        denom = float(_np.linalg.norm(ref_at)) or 1e-30
        rel_l2 = float(_np.linalg.norm(target - ref_at)) / denom
        entry = {"coupled_mean": [float(m) for m in target.mean(axis=0)],
                 "monolithic_mean": [float(m) for m in ref_at.mean(axis=0)],
                 "relative_l2": rel_l2}
        # PER COMPONENT AS WELL AS IN TOTAL. A displacement whose x component is
        # a thousand times its y component has a total relative L2 set entirely
        # by x, so a y component that is 100% wrong reads as 0.1% overall. That
        # is the vector form of the scale masking the per-block residuals exist
        # for, and the mean-based check below has the same hole.
        per: list[float] = []
        for c in range(target.shape[1]):
            d = float(_np.linalg.norm(ref_at[:, c])) or 1e-30
            rc = float(_np.linalg.norm(target[:, c] - ref_at[:, c])) / d
            per.append(rc)
            tag = f"{name} interface mean" + (f" [{c}]" if target.shape[1] > 1 else "")
            findings += check_monolithic_consistency(
                float(target[:, c].mean()), float(ref_at[:, c].mean()), rtol,
                qoi=tag)
            if rc > rtol:
                comp = f" component [{c}]" if target.shape[1] > 1 else ""
                findings.append(
                    f"{name}: coupled interface field{comp} differs from the "
                    f"un-split monolithic re-solve by {rc:.1%} in relative L2 > "
                    f"{rtol:.0%} — the coupled result is likely WRONG even "
                    "though the iteration converged.")
        if target.shape[1] > 1:
            entry["relative_l2_per_component"] = per
        report[name] = entry
    return report, findings, not_run


_COUPLING_WORDS = (
    "coupl",          # coupling, coupled, cosimulation spelt "co-coupling"
    "partition",      # partitioned
    "dirichlet_neumann", "dirichlet-neumann", "neumann_dirichlet",
    "fsi", "tsi",     # the two named multiphysics abbreviations
    "cosim", "co-sim",
    "precice",
    "two_code", "two-code", "two_way", "two-way",
    "interface_field", "staggered",
    # NAMED MULTIPHYSICS PROBLEMS THAT ARE COUPLED BY DEFINITION.
    #
    # A run asked knowledge(topic='conjugate_heat_transfer') — a correct name
    # for a coupled fluid-solid thermal problem — and got the usage message,
    # because every word above describes the METHOD and none names a PROBLEM.
    # An agent that has identified its problem correctly should not have to
    # guess our vocabulary. Still narrow: each of these is coupled by
    # definition, so no single-code request is diverted.
    "conjugate_heat", "conjugate heat",
    "thermomechanic", "thermo_mechanic", "thermo-mechanic",
    "poroelast",      # solid + fluid pressure
    "aeroelast",      # fluid + structure, the aero spelling of fsi
)


def _is_coupling_request(keyword: str) -> bool:
    """Would this keyword only ever be answered by coupled material?

    Deliberately narrow: it fires only when the keyword is ABOUT coupling, so
    an ordinary single-code search is never diverted. `fsi` and `tsi` are in
    because a monolithic FSI deck in one code's test suite is not an example of
    the partitioned two-code run the benchmark asks for, and an agent that gets
    one thinks its question was answered.
    """
    k = (keyword or "").strip().lower()
    if not k:
        return False
    return any(w in k for w in _COUPLING_WORDS)


def _coupling_example_pointer(keyword: str, solver: str) -> str:
    """Say where the coupled example IS, in one reply, without dumping it.

    Not the payload itself: the coupling knowledge is ~66k chars and this tool
    is often called early, when the agent has the fewest tokens spent and the
    most calls left to lose. What it needs here is the name of the call that
    works and the shape of the answer.
    """
    return (
        f"No FILE example for '{keyword}' in {solver}'s test suite — and there "
        f"cannot be one. A partitioned coupled run is TWO participant "
        f"programs plus a driver, so it lives in neither code's test tree.\n\n"
        f"WHERE IT ACTUALLY IS:\n"
        f"  * `knowledge(topic=\"coupling\", solver=\"{solver}\")` — returns the "
        f"participant CONTRACT for {solver}: the interface exchange, the sign "
        f"convention, the consistent flux recovery and the exports schema, "
        f"with the solve elided for you to write. Call it once per code you "
        f"need.\n"
        f"  * `knowledge(topic=\"coupling\")` with no solver — the driver "
        f"contract: imports.json / exports.json shapes, roles, relaxation, and "
        f"the failure table.\n"
        f"  * `couple(participants=...)` — runs the pair and returns a "
        f"verification verdict.\n\n"
        f"THE SHAPE OF A COUPLED RUN (so you can size the work now):\n"
        f"  1. one participant program per subdomain, each solving its own "
        f"subdomain in its own code;\n"
        f"  2. each writes `exports.json` (field_name, coordinates, values, "
        f"and normal_fluxes when a flux exists) and reads the partner's "
        f"`imports.json`;\n"
        f"  3. one side takes the DIRICHLET role (receives the field, returns "
        f"its outward-normal flux), the other the NEUMANN role;\n"
        f"  4. `couple` iterates them to a tolerance and reports the residual "
        f"history.\n\n"
        f"Do NOT take a monolithic {keyword} deck from a single code's tests "
        f"as an example of this: it solves the whole domain in one code, which "
        f"is the opposite of what a partitioned run does.")


def _couple_failure_reason(r, checks_ok: bool) -> str:
    """Name the clause that fired, and point the agent at the field that holds
    the cause.

    Three distinct situations were collapsed into one message, then two:

      * A CRASH. `converged` is False when a participant never ran at all, so a
        crash fell into the convergence branch and the agent was told to read
        `history` for "the residual per iteration" — an empty list. The one
        field carrying the cause, `error`, was never named, and neither were
        the per-participant `returncodes`. An agent shown an empty history and
        no cause has nothing to debug, and the dominant failure mode of coupled
        runs is stopping early: 63% end honestly incomplete at a median 30% of
        budget, with zero timeouts in 112 runs.
      * A GENUINE CONVERGENCE FAILURE, where `history` IS the thing to read.
      * A CONVERGED RUN that failed a downstream silent-wrong check. Telling
        that agent its coupling "may not have converged" is the cheapest
        possible reason to stop, and it is false in the same payload that says
        converged=True.

    Extracted from the tool body so the branches can be tested: nothing
    exercised them before, which is how the crash case survived.
    """
    if checks_ok:
        return ""
    err = str(getattr(r, "error", "") or "").strip()
    hist = list(getattr(r, "history", None) or [])
    rcs = getattr(r, "returncodes", None)
    codes = list(rcs.values()) if isinstance(rcs, dict) else list(rcs or [])
    # A run that iterated to its cap with every exit code 0 is NOT a crash, even
    # though the driver puts its "did not converge" text in `error`: one that
    # iterated 150 times was told "a participant failed to run ... nothing
    # iterated".
    ran_to_the_cap = (bool(hist) and not any(c not in (0, None) for c in codes)
                      and (not err or err.startswith("did not converge")))
    if (err or not hist) and not ran_to_the_cap:
        return (
            ("a participant failed to run after " + str(len(hist)) + " "
             "iteration(s), so the coupling stopped there. " if hist else
             "the coupling did not produce an iteration history — a "
             "participant failed to run, so there is no residual to read. ")
            + (f"The error was: {err[:400]} " if err else
               "No error text was captured. ")
            + (f"Per-participant exit codes: {rcs}. " if rcs else "")
            + ("Fix the participant that failed and re-run; `history` holds "
               "only the iterations before it failed." if hist else
               "Fix the participant that failed and re-run; do NOT read "
               "`history`, which is empty because nothing iterated."))
    if not getattr(r, "converged", False):
        return ("the coupling did not reach the requested tolerance: every "
                f"participant ran, and `history` holds the residual of each of "
                f"its {len(hist)} iteration(s)"
                + (f". The driver's words: {err[:300]}" if err else ""))
    return (
        "the coupling CONVERGED, and then failed one of openPASO's "
        "silent-wrong checks (see `validation`). This is a converged "
        "result with a caveat, NOT a failed run: report the numbers "
        "and the caveat. A downstream check can fail on discretisation "
        "error alone at the coarsest level and pass on the finer ones; "
        "whether this one did is decided by the next level's number, "
        "not assumed.")


def _interface_profile_trend(out_levels: list, shrink: float = 0.6):
    """(faults, notes, excused_levels) from the per-level POINT-BY-POINT flux
    mismatch, judged as the balance is (see _interface_balance_trend).

    MEASURED on a correct vector Dirichlet-Neumann ladder: the pointwise mismatch
    read 11.1% / 15.2% at level 1, 7.2% / 7.8% at level 2 and 5.8% / 3.9% at level
    3 -- the Neumann side's consistent recovery smoothing the flux it was given by
    the boundary mass matrix -- and the level-1 finding alone made the correct
    ladder NOT VERIFIED. A mismatch that shrinks, or ends inside tolerance, is
    that smoothing; one that does not is a mis-mapped exchange. Neither says
    anything about the operator: the equation check does.
    """
    return _interface_numbers_trend(out_levels, "interface_profile",
                                    "pointwise interface flux mismatch", shrink)


def _interface_numbers_trend(out_levels: list, key: str, what: str, shrink: float = 0.6):
    rows = []
    for x in out_levels:
        b = x.get(key) if isinstance(x, dict) else None
        if isinstance(b, dict) and b.get("rel") and not b.get("tautology"):
            try:
                rows.append((int(x.get("level")), [float(v) for v in b["rel"]],
                             float(b.get("rtol", 0.10))))
            except (TypeError, ValueError):
                continue
    if len(rows) < 2:
        return [], [], set()
    ncomp = min(len(r) for _, r, _ in rows)
    faults, notes, excused, held = [], [], set(), set()
    for c in range(ncomp):
        seq = [(k, r[c]) for k, r, _ in rows]
        rtol = rows[0][2]
        fired = [k for k, v in seq if v == v and v > rtol]
        if not fired:
            continue
        vals = [v for _, v in seq]
        ratios = [vals[i + 1] / vals[i] if vals[i] > 0 else 1.0
                  for i in range(len(vals) - 1)]
        name = f"flux component {c}" + (" (the only one)" if ncomp == 1 else "")
        path = " -> ".join(f"{v:.1%}" for v in vals)
        lv = ", ".join(str(k) for k, _ in seq)
        if all(r <= shrink for r in ratios) or vals[-1] <= rtol:
            notes.append(
                f"The {what} of {name}: {path} over levels {lv}, shrinking under "
                f"refinement or ending inside tolerance -- discretisation, and the "
                f"coarse-level caveat is not held against the ladder.")
            excused.update(fired)
        else:
            held.update(fired)
            faults.append(
                f"the {what} of {name} does NOT shrink under refinement ({path} over "
                f"levels {lv}): a mismatch that stays or grows is a mis-mapped or "
                f"inconsistent exchange of that component, and refinement will not "
                f"cure it")
    return faults, notes, excused - held


def _res_coefficient_check(res: dict, ks) -> None:
    """verify_interface_flux: one side's implied coefficient q_n / (-du/dn) against the
    one k its config states, written into `res` in place.

    LEG BY LEG ON A BENT INTERFACE, AND ONE k ONLY WHERE ONE k CAN SAY IT. Measured: a
    side with several materials was told "THE IMPLIED COEFFICIENT ... IS NOT THE k YOUR
    CONFIG STATES", from a ratio taken across both legs at once. Where the implied
    coefficient differs between legs, what was measured is said, and it is not held
    against one stated k: a conductivity that differs by region along the interface
    reads this way, and so does a flux scaled on one leg only."""
    rows = res.get("per_leg") or [res]
    ki = []
    for r in rows:
        for c in r.get("per_component") or []:
            k = c.get("implied_coefficient")
            if isinstance(k, (int, float)) and k > 0:
                ki.append((r.get("leg"), float(k), r.get("verdict")))
    if not ki:
        return
    by_leg = [(lab, k, v) for lab, k, v in ki if lab]
    if len(by_leg) >= 2 and max(k for _l, k, _v in by_leg) > 1.3 * min(k for _l, k, _v in by_leg):
        res["coefficient_check"] = (
            "NOT COMPARED WITH ONE k: the implied coefficient differs between the legs ("
            + ", ".join(f"{k:.4g} on leg {lab}" for lab, k, _v in by_leg)
            + (", steady along each" if all(v == "CONSISTENT" for _l, _k, v in by_leg) else "")
            + (f"; your config states k = {ks:g}" if ks else "") + "). A side whose conductivity "
            "differs by region along the interface reads this way, so one stated k is not held "
            "against it here. Where this side has one conductivity, the leg that departs from it "
            "is the one to look at.")
        return
    if ks and any(abs(k / ks - 1.0) > 0.3 for _l, k, _v in ki):
        res["stated_coefficient"] = ks
        res["coefficient_check"] = (
            f"THE IMPLIED COEFFICIENT ({', '.join(f'{k:.4g}' for _l, k, _v in ki)}) IS NOT "
            f"THE k YOUR CONFIG STATES ({ks:g}): a flux computed as -k du/dn from "
            f"this field would read about {ks:g}. The sign test passes, but the flux "
            f"this side delivers does not follow from this field with this k.")


def _ratio_misfit_trend(per_side: list, shrink: float = 0.6) -> list:
    """verify_interface_flux: the per-level trend of each side's (and leg's) misfit
    (flux_multiple_consistency; the spread of flux_ratio_consistency is read the same
    way), written into the per-side results in place; returns one plain statement per
    side or leg whose misfit went over tolerance at some level.

    MEASURED on a right field: the implied coefficient varied along the interface by
    31 % at level 1, 12 % at level 2 and 8 % at level 3, and level 1 alone read "the two
    were not computed from each other". A misfit (or spread) that falls at every level (by
    `shrink` or more per step, or into tolerance) is what discretisation error does: those
    levels read MISFIT_SHRINKS. One that does not keeps INCONSISTENT with its numbers; a
    single level says what one level cannot tell. No cause is named."""
    groups: dict = {}
    for res in per_side:
        for r in (res.get("per_leg") or [res]):
            pc = next(iter(r.get("per_component") or []), None)
            v = pc.get("misfit", pc.get("spread")) if isinstance(pc, dict) else None
            if not isinstance(v, (int, float)):
                continue
            groups.setdefault((str(res.get("side")), r.get("leg") or ""), []).append(
                (int(res.get("level") or 0), float(v), r,
                 float(r.get("misfit_tolerance") or r.get("spread_tolerance") or 0.30)))
    notes = []
    for (side, leg), seq in sorted(groups.items()):
        seq.sort(key=lambda t: t[0])
        bad = [r for _l, _s, r, _t in seq if r.get("verdict") == "INCONSISTENT"]
        if not bad:
            continue
        who = f"side {side}" + (f", leg {leg}" if leg else "")
        vals, lv, tol = [t[1] for t in seq], [str(t[0]) for t in seq], seq[0][3]
        path = " -> ".join(f"{v:.0%}" for v in vals)
        what = (f"{who}: the flux differs from the best constant multiple of this field's normal "
                f"derivative by")
        falls = len(seq) >= 2 and all(b < a for a, b in zip(vals, vals[1:]))
        if falls and (all(b <= shrink * a for a, b in zip(vals, vals[1:])) or vals[-1] <= tol):
            txt = (f"{what} {path} over levels {', '.join(lv)}. It falls at every level, as "
                   f"discretisation error does, so a level over the {tol:.0%} tolerance is not read "
                   f"as a verdict against the coupling. The trend is all this measures.")
            for r in bad:
                r["verdict"] = "MISFIT_SHRINKS"
                r["detail"] = txt
        elif len(seq) >= 2:
            txt = (f"{what} {path} over levels {', '.join(lv)}"
                   + (f": it falls at every level but stays over the {tol:.0%} tolerance at the finest"
                      if falls else " and does not fall at every level")
                   + ". A flux computed as -k du/dn from this field is a constant multiple of its normal "
                     "derivative. This measures the flux against the field, not why they differ.")
            for r in bad:
                r["detail"] = txt
        else:
            txt = (f"{what} {vals[0]:.0%} at level {lv[0]}, over the {tol:.0%} tolerance. One level "
                   f"cannot tell discretisation error from a flux that does not follow from this "
                   f"field: give every level's files, and a misfit that falls under refinement is "
                   f"discretisation.")
            for r in bad:
                r["detail"] = txt
        notes.append(txt)
    try:
        from .result_audit import _worst_leg
    except Exception:                                        # noqa: BLE001
        from tools.result_audit import _worst_leg
    for res in per_side:
        if res.get("per_leg"):
            w = _worst_leg(res["per_leg"])
            res["verdict"], res["detail"] = w["verdict"], w["detail"]
    return notes


_BLOCK_CAVEAT = "Global residual is NOT representative"
_CLEAN_HEAD = "this level's iteration CONVERGED and your files are self-consistent. "
_BLOCK_NAMED = re.compile(r"([A-Za-z0-9_.\-]+?\.(?:values|normal_fluxes)(?:\[\d+\])?)=([0-9.eE+\-]+)")


def _block_caveat_trend(out_levels: list, blocks_by_level: dict):
    """(faults, notes, excused_levels) from the per-block iteration caveat.

    The caveat measures how far one block's worst entry still moved, relative to its own
    value, at a level's LAST iteration: the iteration's precision at that level. Its own
    per-level trend -- the same block's last-step change at every level, from each level's
    block residuals -- decides: shrinking level by level, it is not a ladder fault and the
    note states the trend; otherwise the fault states the trend and the level to couple
    again with a smaller tol. Measured on the ladders of a steady problem: the caveat fired at the finest
    level only (1.74e-4 against a 1e-5 limit; the same block moved 1.4e-6 and 3.8e-6 at the
    two coarser levels, and the entry held 0.9 % of its block's largest value), and every
    correct ladder read NOT VERIFIED under a text that framed it as discretisation."""
    order = [int(x["level"]) for x in out_levels if isinstance(x, dict) and "level" in x]
    faults, notes, excused = [], [], set()
    for x in out_levels:
        k = x.get("level") if isinstance(x, dict) else None
        cav = [str(v) for v in (x.get("validation") or []) if _BLOCK_CAVEAT in str(v)] if k else []
        if not cav:
            continue
        lim = re.search(r"more than ([0-9.eE+\-]+) relative", cav[0])
        named = sorted({b for v in cav for b, _ in _BLOCK_NAMED.findall(v)})
        if not named:
            continue
        rows, all_shrink = [], True
        for b in named:
            seq = []
            for j in order:
                try:
                    v = float((blocks_by_level.get(j) or {}).get(b))
                except (TypeError, ValueError):
                    continue
                if v == v:
                    seq.append((j, v))
            shrinks = len(seq) >= 2 and all(b2 < a2 for (_, a2), (_, b2) in zip(seq, seq[1:]))
            all_shrink = all_shrink and shrinks
            rows.append(f"{b} {', '.join(f'{v:.1e}' for _, v in seq)} at levels "
                        f"{', '.join(str(j) for j, _ in seq)}")
        where = "; ".join(rows) + (f" (limit {lim.group(1)})" if lim else "")
        if all_shrink:
            excused.add(int(k))
            notes.append(f"The per-block iteration caveat at level {k} shrinks level by level: {where}. "
                         f"It is the iteration's precision at each level, not a ladder fault.")
        else:
            faults.append(f"level {k}: the per-block iteration caveat does not shrink across the ladder "
                          f"(the last-step change of its worst entry, relative to that entry's own value: "
                          f"{where}): the iteration at level {k} stopped before that block settled. That "
                          f"is the iteration's precision at that level, not a discretisation trend; couple "
                          f"level {k} again with a smaller tol.")
    return faults, notes, excused


def _interface_balance_trend(out_levels: list, shrink: float = 0.6):
    """(faults, notes, excused_levels) from the per-level interface balance.

    Reads the `interface_balance` numbers each level's couple() reply carried
    (per flux component: |net_A + net_B| / max|net|). A component whose
    imbalance exceeded its tolerance at some level is classified by how it
    moves across the levels: shrinking (every step at most `shrink` of the
    previous, or the last level inside tolerance) is discretisation error and
    the levels it fired on are EXCUSED; anything else is a fault. Levels whose
    numbers are a tautology (the two exports cannot disagree) carry no
    evidence and are skipped. Measured on a correct thermo-mechanical ladder
    the per-channel jump falls ~4x per halving; on a wrong one two channels
    fell 1.2x and 1.05x while the third fell 4x.
    """
    rows = []
    for x in out_levels:
        b = x.get("interface_balance") if isinstance(x, dict) else None
        if isinstance(b, dict) and b.get("rel") and not b.get("tautology"):
            try:
                rows.append((int(x.get("level")), [float(v) for v in b["rel"]],
                             float(b.get("rtol", 0.05))))
            except (TypeError, ValueError):
                continue
    if len(rows) < 2:
        return [], [], set()
    ncomp = min(len(r) for _, r, _ in rows)
    faults, notes, excused, held = [], [], set(), set()
    for c in range(ncomp):
        seq = [(k, r[c]) for k, r, _ in rows]
        rtol = rows[0][2]
        fired = [k for k, v in seq if v == v and v > rtol]
        if not fired:
            continue
        vals = [v for _, v in seq]
        ratios = [vals[i + 1] / vals[i] if vals[i] > 0 else 1.0
                  for i in range(len(vals) - 1)]
        name = f"flux component {c}" + (" (the only one)" if ncomp == 1 else "")
        path = " -> ".join(f"{v:.1%}" for v in vals)
        lv = ", ".join(str(k) for k, _ in seq)
        if all(r <= shrink for r in ratios) or vals[-1] <= rtol:
            # THE TREND, NOT A CAUSE. "discretisation error, as a consistent exchange
            # shows" was said of an imbalance of exactly h (12.5 % -> 6.2 % -> 3.1 %):
            # two zeroed end nodes against a partner's constant placeholder flux of
            # 1e-10 (measured). A shrinking imbalance is what discretisation error does,
            # and not the only thing that does it.
            notes.append(
                f"Interface balance of {name}: {path} over levels {lv}, shrinking under "
                f"refinement, as discretisation error does; the coarse-level caveat is not "
                f"held against the ladder. The trend is all this measures: an error that "
                f"lives on a fixed number of nodes shrinks the same way.")
            excused.update(fired)
        else:
            held.update(fired)
            faults.append(
                f"the interface balance of {name} does NOT shrink under "
                f"refinement ({path} over levels {lv}): discretisation error "
                f"falls by about 4x per mesh halving (2x for a first-order "
                f"recovery); an imbalance that stays or grows is a wrong sign, "
                f"scaling or missing term in what one side EXPORTS for that "
                f"component against what the partner APPLIES, and refinement "
                f"will not cure it")
    # EACH LEG OF A BENT INTERFACE, as the whole. Measured: the whole-interface net fell
    # 9.1 % -> 3.5 % -> 1.3 % and read as discretisation error while each leg's grew,
    # 15 % -> 18 % -> 22 % and 6 % -> 14 % -> 21 % (a flux applied by row index moved it
    # from one leg to the other). The worst leg is judged by the same rule, and where it
    # does not shrink, no level is excused and the whole-interface note is not served.
    leg_rows = []
    for x in out_levels:
        b = x.get("interface_balance") if isinstance(x, dict) else None
        lg = b.get("legs") if isinstance(b, dict) and not b.get("tautology") else None
        if isinstance(lg, dict) and lg.get("rel"):
            try:
                leg_rows.append((int(x.get("level")), [float(v) for v in lg["rel"]],
                                 float(lg.get("rtol", 0.05))))
            except (TypeError, ValueError):
                continue
    if len(leg_rows) >= 2:
        whole_path = {c: " -> ".join(f"{r[c]:.1%}" for _k, r, _t in rows if c < len(r))
                      for c in range(ncomp)} if rows else {}
        for c in range(min(len(r) for _, r, _ in leg_rows)):
            seq = [(k, r[c]) for k, r, _ in leg_rows]
            rtol = leg_rows[0][2]
            fired = [k for k, v in seq if v == v and v > rtol]
            if not fired:
                continue
            vals = [v for _, v in seq]
            ratios = [vals[i + 1] / vals[i] if vals[i] > 0 else 1.0 for i in range(len(vals) - 1)]
            path = " -> ".join(f"{v:.1%}" for v in vals)
            lv = ", ".join(str(k) for k, _ in seq)
            comp = f" (flux component {c})" if len(leg_rows[0][1]) > 1 else ""
            if all(r <= shrink for r in ratios) or vals[-1] <= rtol:
                notes.append(
                    f"Interface balance of the worst leg of the bent interface{comp}: {path} over "
                    f"levels {lv}, shrinking under refinement, as discretisation error does. The "
                    f"trend is all this measures.")
                excused.update(fired)
            else:
                held.update(fired)
                notes[:] = [t for t in notes if not t.startswith("Interface balance of flux component")]
                faults.append(
                    f"the interface balance of the worst leg of the bent interface{comp} does NOT "
                    f"shrink under refinement ({path} over levels {lv}, as a share of the flux the "
                    f"whole interface carries"
                    + (f"; the whole interface: {whole_path[c]}" if whole_path.get(c) else "")
                    + "): flux one side puts on one leg and its partner puts on another cancels in "
                    "the whole-interface sum, so a shrinking whole-interface imbalance does not say "
                    "the exchange is right, and refinement will not cure this")
    return faults, notes, excused - held


def _stamp_verification(result: dict, *, evidence_ok: bool, reason: str = "",
                        critic_approved: bool = False,
                        solver: str | None = None,
                        setup_text: str | None = None,
                        critic_token: str = "",
                        job_id: str = "",
                        field_defect: bool = False,
                        iteration_caveat: bool = False) -> dict:
    """Attach openPASO's verification-gate verdict to a run/coupling result in place.

    A result is trustworthy ONLY when it (1) passes the numerical checks — the
    run completed, produced output, and that output is finite / converged /
    balanced — AND (2) has been reviewed by openPASO's independent critic. openPASO
    *verifies* and checks integrity; it does not *validate* — physical validity
    stays the engineer's task.

    SCOPE, STATED HONESTLY. These checks bind the verdict to the RUN. They do
    NOT bind it to a reported NUMBER: nothing here recomputes a value the agent
    states, so a plausible invented number attached to a real run still passes.
    An audit demonstrated exactly that against a live backend. Earlier wording
    here claimed attestation "binds every reported number to run evidence";
    that was false and is removed rather than softened. Binding a value to the
    data it came from requires computing it from the run's own output (see
    core/attestation.py) and checking the field satisfies the discrete problem
    (see core/residual_check.py); until those are wired into this path, the
    verdict means "a real, clean run happened", not "this number came from it".

    Enforcement is by VERDICT, never by error: an unverified run still returns
    its output, but is never labelled trustworthy, so a confidently-wrong or
    fabricated claim can't be reported as a result. The critic is mandatory,
    unconditionally — there is no ablation switch that lifts it.

    THE CRITIC IS RESOLVED FROM THE SERVER'S RECORD, NOT FROM THE ARGUMENT.
    Pass `solver` and `setup_text` and this function asks the critic registry
    whether that exact deck was reviewed. A caller that does NOT identify its
    setup gets NOT VERIFIED: the gate fails closed, because a tool that cannot
    say what it ran cannot have had that thing reviewed. `critic_approved` is
    retained only to record what the agent CLAIMED, so a claim with no matching
    review can be named as such in the verdict.

    evidence_ok: True iff a real run backs this result AND the gate's numerical
        checks passed (execution completed, output/logs produced, converged,
        finite, interface balanced — as applicable to the calling tool).
    reason: short cause shown when evidence_ok is False.
    critic_approved: what the agent asserted. Recorded, never trusted.
    solver, setup_text: identify the deck whose review is being looked up.
    critic_token: optional single-use token from `submit_critic_review`.
    job_id: recorded against a redeemed token for audit.
    """
    if solver is not None and setup_text is not None:
        critic_ok, critic_note = _critic_state(
            solver, setup_text, token=critic_token, job_id=job_id)
    else:
        critic_ok, critic_note = False, (
            "not checked — this tool did not identify its setup to the "
            "verification gate, so no review could be looked up")
    if critic_approved and not critic_ok:
        critic_note += ("; the call declared critic_approved=True, which openPASO "
                        "does not accept as evidence — a review must be on "
                        "record via submit_critic_review")
    if not evidence_ok:
        result["trustworthy_result"] = False
        # A CONVERGENCE failure and a CONSERVATION finding on a converged run
        # are different situations and must not carry the same imperative.
        # Measured: 10 coupled runs drove a coupling to convergence, hit a
        # flux-balance finding, read "must NOT be reported as a result" in the
        # tool's own reply, and declared they could not finish with a median
        # 69% of their budget unspent — while the served knowledge said the
        # opposite ("a converged run with a failed conservation check is still
        # a result"). The agent obeys the imperative it is holding.
        # ROUTE ON THE PROPERTY, NOT ON THREE WORDS. The first router
        # recognised a converged-with-caveat run only when the reason contained
        # balance/conserv/flux. One recorded run's caveat read "the
        # coupling CONVERGED, and then failed one of openPASO's silent-wrong
        # checks ... report the numbers and the caveat" -- no listed word -- so
        # it fell to the else-branch, and the agent received both "report the
        # numbers and the caveat" AND "must NOT be reported as a result;
        # revise the setup and re-run" IN THE SAME PARAGRAPH. It obeyed the
        # harsher one, exported a residual its own files contradict, and
        # ended complete but unphysical. A converged run with a failed
        # downstream check is a result with a finding, whatever the finding is
        # called; the harsh imperative is for claims with no converged run
        # behind them.
        _low = (reason or "").lower()
        _conserv = any(w in _low for w in ("balance", "conserv", "flux",
                                           "caveat", "silent-wrong"))
        _converged = ("did not converge" not in _low
                      and "not converged" not in _low
                      and "not finite" not in _low)             # a NaN field is no result
        # THE FLAG DECIDES WHERE THERE IS ONE, NOT THE WORDS. Measured: a level that crashed
        # (converged False, residual NaN) was told "CONVERGED -- THIS IS A RESULT, SAVE IT NOW"
        # because its failure text said "flux".
        if result.get("converged") is False:
            _converged = False
        _says_converged = "converged" in _low and _converged
        if field_defect:
            # A DEFECT OF THE FIELD ITSELF IS NOT A CAVEAT THE NEXT LEVEL DECIDES.
            # Measured: a level whose side left part of a held edge free, and a
            # level whose sides exchanged nothing, were told "CONVERGED -- THIS IS A
            # RESULT, SAVE IT NOW" because the finding's text said "converges".
            result["verification"] = (
                "NOT VERIFIED -- " + (reason or "a side's field is defective")
                + ". This is a defect of what this level produced, not a "
                "caveat that refinement settles: do not save this level as a "
                "result. Fix the side named here, then couple this level again.")
        elif iteration_caveat and _converged:
            # AN ITERATION CAVEAT IS NOT A DISCRETISATION QUESTION. Measured: a per-block
            # caveat at a ladder's finest level was framed as "discretisation error or a wrong
            # transmission condition ... decided by the next level" on every correct ladder
            # of a steady round; it measures how far the iteration had settled.
            result["verification"] = (
                "CONVERGED WITH AN ITERATION CAVEAT (" + (reason or "a per-block caveat") + "). "
                "It measures how far one exchanged block had settled at the last iteration, "
                "not a discretisation trend. Coupling this level again with a smaller tol "
                "shows whether that block settles. "
                + ("The ladder verdict states that block's per-level trend." if _IN_LADDER.get() else
                   "Report it beside this level's numbers."))
        elif (_conserv or _says_converged) and _converged and _IN_LADDER.get():
            # INSIDE A LADDER THE NEXT LEVEL RUNS BY ITSELF, and the ladder's verdict
            # judges the caveat's trend. Measured: ladder replies read "THIS LADDER IS
            # NOT VERIFIED ... level 2: CONVERGED -- THIS IS A RESULT, SAVE IT NOW" and
            # "Fix that before the deliverables" in one breath (three cells of one round).
            result["verification"] = (
                "CONVERGED WITH A CAVEAT (" + (reason or "a conservation check") + "). "
                "Whether that is discretisation error or a wrong transmission "
                "condition is decided by the next level: discretisation error "
                "shrinks by about 4x per mesh halving (2x for a first-order "
                "recovery); a caveat that stays or grows is a wrong sign, scaling "
                "or missing term. The ladder verdict states its per-level trend.")
        elif (_conserv or _says_converged) and _converged:
            result["verification"] = (
                "CONVERGED — THIS IS A RESULT, SAVE IT NOW. The coupling "
                "reached its fixed point; a downstream check flagged a "
                "caveat (" + (reason or "a conservation check") + "). "
                "Whether that caveat is discretisation error or a wrong "
                "transmission condition is decided by the NEXT level, not "
                "assumed here: discretisation error shrinks by about 4x per "
                "mesh halving (2x for a first-order recovery); a caveat that "
                "stays or grows across levels is a wrong sign, scaling or "
                "missing term, and refinement will not cure it. The ladder "
                "verdict states the per-level trend of every check that "
                "fired, and one that does not shrink makes the ladder NOT "
                "VERIFIED whatever the residuals did. Do NOT "
                "discard this and do NOT hand-write a residual file: this "
                "reply already carries `residual_csv` (the measured "
                "iteration history), `interface_csv` (your interface tables) "
                "and `captured_solver_logs` (the log paths). SAVE THOSE "
                "VERBATIM as this level's deliverables, then run the NEXT "
                "level. A one-row or hand-written residual file reads as "
                "no coupling at all — the real multi-row history is in this "
                "reply. Report the caveat ALONGSIDE the saved files; it is "
                "not a reason to withhold them. (This is 'verification', "
                "not 'validation': physical validity is still yours to "
                "confirm — but the run is a RESULT.)")
        else:
            result["verification"] = (
                "NOT VERIFIED — "
                + (reason or "the result is not bound to a check-passing run")
                + ". Per openPASO attestation this claim must NOT be reported as "
                "a result; revise the setup and re-run.")
    elif not critic_ok:
        # THE CAUSE FIRST, AND WHAT "PASSED" COVERS. Measured 2026-09-29 in the web interface:
        # this read "the automated checks passed, but openPASO's MANDATORY independent critic
        # has not reviewed this setup" on runs whose review WAS on record -- filed for another
        # version of the script -- so the history showed an approved review and the verdict
        # denied one, and a reader took "the automated checks passed" for a physics check on a
        # run whose drag coefficient was 53,288 at every step.
        result["trustworthy_result"] = False
        result["verification"] = (
            "NOT VERIFIED — no critic review on record matches this setup ("
            + critic_note + "). openPASO's MANDATORY independent critic review is the gate "
            "still open, and openPASO treats no result as trustworthy until it is closed. The "
            "automated checks passed, and they are narrow: the run finished and wrote its "
            "output, every number in it is finite, the mesh is not degenerate and the result "
            "is not zero everywhere. They do not check that the physics is right. Spawn a "
            "critic to challenge the parameters, units, discretisation, problem "
            "statement and boundary conditions and to cross-check against "
            "literature/benchmarks, and give it the file to review: the critic files its "
            "own verdict with submit_critic_review, naming that file. Then run the file "
            "unchanged (input_path / generator_path). A review you write yourself is not a "
            "review. Asserting critic_approved=True does not work: "
            "openPASO looks the review up rather than taking your word for it.")
    else:
        result["trustworthy_result"] = True
        result["verification"] = (
            # WHO WROTE THE REVIEW IS NOT KNOWN HERE. Measured: all five cells of one round filed
            # the review themselves, no critic was spawned, and each read "an independent critic
            # reviewed this exact setup".
            "VERIFIED — a critic review of this exact setup is on record ("
            + critic_note + ") and the run passed openPASO's verification-gate "
            "numerical checks (it finished; its output is finite, on a mesh that is not "
            "degenerate, and not zero everywhere). This is verification, not validation: confirm "
            "physical validity against reality yourself. WHAT THE REVIEW PART DOES "
            "NOT PROVE: openPASO holds the review that was submitted and binds it to "
            "this exact setup, but it cannot see who wrote it. A model that composes a "
            "review of its own script and submits it reaches this line too — measured "
            "on a live run whose critic sub-agent returned nothing at all. So read this "
            "as 'a review exists and matches what ran', not as 'someone independent "
            "approved it'. The review covers the setup's own text; files it reads when it "
            "runs (a mesh, a module it imports) are not part of it. "
            "A VERIFIED arrangement is SETTLED: write its deliverable files "
            "now, from these numbers, then move to the next arrangement "
            "(finer level, next case). Re-running a verified arrangement "
            "re-proves what is already proven — measured: one session drove "
            "six VERIFIED couplings of the same mesh level, banked none of "
            "them, and ran out of budget with the remaining levels untouched; "
            "an incomplete level sequence is unusable however many times its "
            "first level was verified. "
            + _residual_coverage_note(result)
            + " " + _critic_coverage_note())
    # WHAT THE RESULT ITSELF SHOWS rides in the verdict, whatever the verdict is: a note in a
    # key of its own is a note nobody reads (the verdict is the line a reader and a model act on).
    _notes = result.get("result_notes") or []
    if _notes:
        result["verification"] += (" MEASURED IN THE RESULT, and worth a look before relying on "
                                   "it: " + "; ".join(_notes) + ".")
    result["critic_review"] = critic_note
    return result


def _solver_module_owners() -> dict:
    """Top-level module -> the solver it belongs to."""
    from tools.participant_lint import _MODULE_TO_BACKEND
    return {**_MODULE_TO_BACKEND, "basix": "fenics"}


def _solver_modules_in(script: str) -> dict:
    """The top-level modules a script imports that belong to one of openPASO's solvers,
    each with that solver's name."""
    import ast
    import re as _re
    try:
        tops = set()
        for node in ast.walk(ast.parse(script or "")):
            if isinstance(node, ast.Import):
                tops |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                tops.add(node.module.split(".")[0])
    except SyntaxError:
        tops = set(_re.findall(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", script or "", _re.M))
    owners = _solver_module_owners()
    return {m: owners[m] for m in tops if m in owners}


def _what_the_error_is(error: str, solver: str) -> str:
    """What a failed run already told us, with the call that works: the scrambled square, a
    script run under the wrong solver, and the measured table in participant_lint. Empty when
    openPASO has nothing measured to say about this failure.

    Measured 2026-09-29 in the web interface: an NGSolve run failed nine times in a row, and
    no reply named a fix -- the table of measured errors was handed only to coupled
    participants' shell runs, never to run_simulation, the call a single-solver run makes
    (and it did not yet hold those errors; they are in it now)."""
    import re as _re
    from tools.participant_lint import findings_from_output, scrambled_exponent_hint
    named = []
    hint = scrambled_exponent_hint(error or "")
    if hint:
        named.append(hint)
    # the whole dotted name: a missing SUBmodule (dolfinx.io.gmshio) is an API change, not
    # a missing solver, and the table below names it
    missing = _re.search(r"No module named '([A-Za-z_][\w.]*)'", error or "")
    owner = _solver_module_owners().get(missing.group(1)) if missing else None
    if owner and owner != solver:
        named.append(f"the script imports {missing.group(1)}, which is {owner}'s: run it with "
                     f"solver='{owner}'. Each solver runs in its own interpreter, and {solver}'s "
                     f"does not have {missing.group(1)}")
    elif owner:
        named.append(f"{solver}'s own interpreter cannot import {missing.group(1)} on this "
                     f"install: discover(query='list') shows the interpreter openPASO found "
                     f"for {solver}")
    # the table's "its own interpreter" entries tell a PARTICIPANT to change its command;
    # a run_simulation script has no command of its own, so the lines above say it instead
    named += [f for f in findings_from_output(error or "")
              if "this code has its own interpreter" not in f]
    return "\n".join(named)


def _short_reason(msg: str, limit: int = 240) -> str:
    """Collapse a multi-line availability/error message (often a raw import
    traceback) to a concise one-liner for user-facing surfaces. A backend that
    isn't installed should read as one clear line, not a 30-frame stack dump that
    floods the backend list. Keeps the actual exception (last non-empty line) and
    any install/try hint; the full trace stays available via `developer`/logs.
    """
    if not msg:
        return msg or ""
    lines = [ln.strip() for ln in str(msg).strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0][:limit]
    tail = lines[-1]
    if tail.lower().startswith("traceback"):
        tail = lines[-2] if len(lines) > 1 else tail
    hint = next((ln for ln in lines
                 if ln.lower().startswith(("install", "try", "run ", "set ", "conda ", "pip "))),
                "")
    out = tail if (not hint or hint == tail) else f"{tail}  ({hint})"
    return out[:limit]


# ─────────────────────────────────────────────────────────────────────────────
# JSON knowledge-block fitting
# ─────────────────────────────────────────────────────────────────────────────
# Every knowledge block an agent receives is rendered as a ```json fence. Until
# 2026-08-03 the size cap was applied with a raw string slice, `text[:LIMIT]`,
# which cuts in the middle of whatever value happens to sit at that offset. A
# sweep of all 206 (backend, physics) payloads found 12 over the cap, and ALL 12
# came back as INVALID JSON under the slice — 7 SPARTA rows and 5 FEniCSx rows,
# failing with "Unterminated string" or "Expecting property name enclosed in
# double quotes". A small model handed an unparseable payload has nothing to
# fall back on, so this is worse than serving less.
#
# _fit_json_block guarantees three things:
#   1. the returned text ALWAYS parses as JSON,
#   2. load-bearing entries are never removed,
#   3. whatever was removed is named, with the call that fetches it.
# It shrinks in two phases: first thin out the *inside* of the largest
# non-load-bearing container (so a big command reference degrades to a prefix
# rather than vanishing), then drop whole non-load-bearing entries. If only
# load-bearing entries remain it serves them in full rather than cutting.

_LOAD_BEARING_KEYS = frozenset({
    # Cross-backend: what a model needs to actually set the problem up.
    "description", "minimal_working_example", "worked_example",
    "function_space", "function_spaces", "weak_form", "weak_forms",
    "boundary_conditions", "solver", "verification",
    "problem_type", "required_sections", "input_format",
    # THE MEASURED TRAPS NEVER YIELD. The physics door fits its record to the
    # reply cap; before this, a 4C elasticity record came back with 1 of its 7
    # pitfalls and a note saying so (measured 2026-09-23).
    "pitfalls",
})


def _is_load_bearing(name: str) -> bool:
    """Entry names that must survive any shrink, on any backend.

    Anything holding runnable input counts: a small model copies an example,
    it does not reconstruct one from prose.
    """
    n = str(name).lower()
    return n in _LOAD_BEARING_KEYS or "example" in n or "template" in n


def _fit_json_block(payload: dict, limit: int, fetch_hint: str = "") -> tuple[str, str]:
    """Render ``payload`` as JSON within ``limit`` chars WITHOUT ever slicing.

    Returns ``(json_text, note)``. ``json_text`` always parses. ``note`` is a
    human-readable line naming what was thinned or dropped (empty if nothing).
    """
    def dump(obj):
        return json.dumps(obj, indent=2, default=str)

    text = dump(payload)
    if len(text) <= limit:
        return text, ""

    work = dict(payload)
    thinned, dropped = [], []
    droppable = [k for k in work if not _is_load_bearing(k)]

    # Phase 1 — thin the inside of oversized containers, largest first. Keeps a
    # usable prefix of e.g. a per-command reference instead of losing all of it.
    for key in sorted(droppable, key=lambda k: -len(dump(work[k]))):
        if len(dump(work)) <= limit:
            break
        val = work[key]
        if not isinstance(val, (dict, list)) or len(val) < 2:
            continue
        items = list(val.items()) if isinstance(val, dict) else list(enumerate(val))
        # Keep at least one sub-entry: "0 of 7 kept" is a drop wearing a
        # thinning label, and phase 2 reports drops properly.
        lo, hi, best = 1, len(items), None
        while lo <= hi:                       # binary search the longest prefix
            mid = (lo + hi) // 2
            if isinstance(val, dict):
                cand = dict(items[:mid])
                cand[f"__{len(items) - mid}_more_entries_omitted__"] = fetch_hint or "ask for this section by name"
            else:
                cand = [v for _, v in items[:mid]]
                cand.append(f"__{len(items) - mid}_more_entries_omitted__")
            probe = dict(work)
            probe[key] = cand
            if len(dump(probe)) <= limit:
                best, lo = cand, mid + 1
            else:
                hi = mid - 1
        if best is not None and len(best) - 1 < len(items):
            work[key] = best
            thinned.append(f"{key} ({len(best) - 1} of {len(items)} entries kept)")

    # Phase 2 — drop whole non-load-bearing entries, largest first.
    for key in sorted(droppable, key=lambda k: -len(dump(work.get(key, "")))):
        if len(dump(work)) <= limit:
            break
        if key in work:
            work.pop(key)
            thinned = [t for t in thinned if not t.startswith(f"{key} (")]
            dropped.append(key)

    text = dump(work)
    bits = []
    if dropped:
        bits.append("omitted entirely: " + ", ".join(dropped))
    if thinned:
        bits.append("shortened: " + "; ".join(thinned))
    note = ""
    if bits:
        # Say WHAT HAPPENED before saying what is gone. The reader of this note
        # is an agent deciding whether the block it just parsed is the whole
        # record; "Trimmed to fit" alone told it the payload was edited but not
        # that the copy in hand is INCOMPLETE, and a small model reads a JSON
        # object that parses as the complete answer. The phrase is fixed
        # wording, not decoration: tests/test_payload_stays_parseable.py scans
        # the served text for it to prove no section is dropped silently.
        head = ("this payload was too large to serve whole"
                if dropped else "this payload was trimmed to fit")
        note = ("\n\n[INCOMPLETE — " + head + ", so what follows the fence is "
                "not the full record: " + " | ".join(bits)
                + (f". Fetch the full record with {fetch_hint}" if fetch_hint else "")
                + "]")
    if len(text) > limit:
        note += (f"\n\n[Payload is {len(text)} chars, over the {limit} soft cap; "
                 f"served whole because every remaining entry is load-bearing "
                 f"and cutting it would produce invalid JSON.]")
    return text, note


def _strip_pitfalls(obj):
    """Recursively remove pitfall-DB keys from nested dicts/lists.

    No-op when _ABLATE_PITFALLS is False.
    """
    if not _ABLATE_PITFALLS:
        return obj
    if isinstance(obj, dict):
        return {k: _strip_pitfalls(v) for k, v in obj.items()
                if k not in _PITFALL_KEYS}
    if isinstance(obj, list):
        return [_strip_pitfalls(x) for x in obj]
    return obj


async def _run_with_progress(ctx: Context, coro, message_prefix: str = "Running"):
    """Run a coroutine while sending periodic MCP progress keepalives.

    This prevents the MCP client from timing out on long-running simulations
    (DUNE JIT compilation, 4C FSI, deal.II builds can take minutes).
    """
    import asyncio

    task = asyncio.create_task(coro)
    elapsed = 0
    try:
        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except asyncio.TimeoutError:
                elapsed += 5
                try:
                    await ctx.report_progress(
                        elapsed, total=None,
                        message=f"{message_prefix} ({elapsed}s elapsed)"
                    )
                except Exception:
                    pass  # progress reporting is best-effort
    except Exception:
        if not task.done():
            task.cancel()
        raise
    return task.result()


def _stub_template_tag(content: str, fmt: str) -> str:
    """Return a `" — ⚠ STUB"` marker if `content` looks like a
    placeholder template (a single comment line, or fewer than
    ~150 chars of non-comment body), otherwise empty.

    The catalog ships 9 fourc physics rows whose generators
    return only a one-line comment (`# Foo template — use ...`)
    because no full template has been written yet. Surfacing
    those as plain `## Template` sections in prepare_simulation
    output misleads the LLM: the heading promises a runnable
    template, but the body is a 50-80 char placeholder.

    Detection rule: strip every line that begins with `#`
    (YAML / Python comment) or is whitespace-only; if what
    remains is shorter than 150 chars, treat as a stub. The
    `fmt` argument tells us which comment character to honour
    — for the (rare) non-comment-character formats (`json`,
    `cpp`), we still apply the size heuristic but skip the
    comment-stripping step. (Audit 2026-06-02.)
    """
    if not isinstance(content, str):
        return ""
    # Central stub detection: catches print-and-exit (deal.II), availability-probe
    # (Kratos), <...>-placeholder decks (4C), and comment-only templates — fakes that
    # advertise physics but don't solve. (Extends the original size heuristic.)
    try:
        from core.quality_checks import is_stub_output
        reason = is_stub_output(content)
        if reason:
            return f" — ⚠ STUB (not a runnable deck: {reason})"
    except Exception:
        pass
    if fmt in ("yaml", "yml", "python", "py"):
        non_comment_lines = [
            ln for ln in content.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        body = "\n".join(non_comment_lines)
    else:
        body = content
    if len(body.strip()) < 150:
        return " — ⚠ STUB (catalog placeholder — no full template yet)"
    return ""


_PHYSICS_SYNONYMS = {
    # ── Added 2026-08-03: phrasings a weak model actually types for
    # the eight FEniCSx physics that had no route at all. Each target
    # is only returned when the backend really carries that key, so
    # these are inert for backends that do not.
    "porous_media": "stokes_darcy",
    "porous_flow": "stokes_darcy",
    "poroelastic": "stokes_darcy",
    "brinkman": "stokes_darcy",
    "free_flow_porous": "stokes_darcy",
    "thermoelastic": "thermal_structural",
    "thermoelasticity": "thermal_structural",
    "thermal_expansion": "thermal_structural",
    "thermal_stress": "thermal_structural",
    "phase_separation": "cahn_hilliard",
    "spinodal": "cahn_hilliard",
    "spinodal_decomposition": "cahn_hilliard",
    "binary_mixture": "cahn_hilliard",
    "species_transport": "reaction_diffusion",
    "multi_species": "reaction_diffusion",
    "turing": "reaction_diffusion",
    "eigenmodes": "eigenvalue",
    "modal_analysis": "eigenvalue",
    "natural_frequency": "eigenvalue",
    "obstacle_problem": "contact",
    "unilateral_contact": "contact",
    "signorini": "contact",
    "damage": "fracture",
    "griffith": "fracture",
    "free_surface": "multiphase",
    "level_set": "multiphase",

    # ── Heat / thermal conduction ──────────────────────────────────
    # canonical key 'heat' exists in: fourc, fenics, ngsolve, kratos,
    # dealii, dune, skfem, febio (all 8 backends)
    "thermal": "heat",
    "conduction": "heat",
    "temperature": "heat",
    "heat_transfer": "heat",
    "heat_conduction": "heat",
    "heat_flow": "heat",
    "thermal_conduction": "heat",
    "thermal_diffusion": "heat",
    "fourier": "heat",
    # transient flavour (canonical: heat_transient OR time_dependent_heat)
    "unsteady_heat": "heat_transient",
    "transient_heat": "heat_transient",
    "time_heat": "heat_transient",
    "dynamic_thermal": "heat_transient",
    "transient_thermal": "heat_transient",
    "time_dependent_thermal": "time_dependent_heat",

    # ── Linear elasticity / small-strain mechanics ─────────────────
    "elasticity": "linear_elasticity",
    "structural": "linear_elasticity",
    "structural_mechanics": "linear_elasticity",
    "structural_2d": "linear_elasticity",
    "structural_3d": "linear_elasticity",
    "solid": "linear_elasticity",
    "solid_mechanics": "linear_elasticity",
    "mechanics": "linear_elasticity",
    "small_strain": "linear_elasticity",
    "hooke": "linear_elasticity",
    "hookean": "linear_elasticity",
    "linear_solid": "linear_elasticity",
    "plane_strain": "linear_elasticity",
    "plane_stress": "linear_elasticity",
    "elastic": "linear_elasticity",
    "fea": "linear_elasticity",
    "statics": "linear_elasticity",
    "elasticity_2d": "linear_elasticity",
    "elasticity_3d": "linear_elasticity",

    # ── Hyperelasticity / large-deformation solid ──────────────────
    "nonlinear_elasticity": "hyperelasticity",
    "large_deformation": "hyperelasticity",
    "large_strain": "hyperelasticity",
    "neo_hookean": "hyperelasticity",
    "neohookean": "hyperelasticity",
    "mooney_rivlin": "hyperelasticity",
    "ogden": "hyperelasticity",
    "finite_strain": "hyperelasticity",
    "finite_deformation": "hyperelasticity",
    "hyperelastic": "hyperelasticity",
    "hyperelastic_solid": "hyperelasticity",
    "geometric_nonlinearity": "hyperelasticity",
    "nonlinear_solid": "hyperelasticity",

    # ── Plasticity / elasto-plastic ────────────────────────────────
    "elasto_plasticity": "plasticity",
    "elastoplasticity": "plasticity",
    "elasto_plastic": "plasticity",
    "yield": "plasticity",
    "yielding": "plasticity",
    "mohr_coulomb": "plasticity",
    "drucker_prager": "plasticity",
    "von_mises": "plasticity",
    "j2_plasticity": "plasticity",
    "soil_plasticity": "plasticity",
    "metal_plasticity": "plasticity",
    "return_mapping": "plasticity",
    "plastic_flow": "plasticity",

    # ── Stokes (creeping / mixed) flow ─────────────────────────────
    "stokes_flow": "stokes",
    "creeping_flow": "stokes",
    "mixed_stokes": "stokes",
    "taylor_hood": "stokes",
    "low_reynolds": "stokes",

    # ── Navier-Stokes / CFD ────────────────────────────────────────
    "cfd": "navier_stokes",
    "flow": "navier_stokes",
    "fluid_dynamics": "navier_stokes",
    "ns": "navier_stokes",
    "incompressible": "navier_stokes",
    "incompressible_flow": "navier_stokes",
    "viscous_flow": "navier_stokes",
    "fluid_flow": "navier_stokes",
    "fluid_mechanics": "navier_stokes",
    "internal_flow": "navier_stokes",
    "channel_flow": "navier_stokes",
    "external_flow": "navier_stokes",
    "laminar_flow": "navier_stokes",
    # transient flavour
    "transient_ns": "time_dependent_ns",
    "unsteady_ns": "time_dependent_ns",
    "unsteady_navier_stokes": "time_dependent_ns",
    "vortex_shedding": "time_dependent_ns",

    # ── Maxwell / electromagnetism ─────────────────────────────────
    "magnetostatics": "maxwell",
    "electromagnetics": "maxwell",
    "em": "maxwell",
    "magnetic": "maxwell",
    "eddy_current": "maxwell",
    "eddy_current_problem": "maxwell",
    "nedelec": "maxwell",
    "electromagnetic": "maxwell",
    "h_curl": "maxwell",
    "electric_field": "maxwell",
    "magnetic_field": "maxwell",
    "electrostatics": "maxwell",
    "electrodynamics": "maxwell",

    # ── Helmholtz / time-harmonic acoustics ────────────────────────
    "acoustics": "helmholtz",
    "acoustic": "helmholtz",
    "sound": "helmholtz",
    "frequency_domain": "helmholtz",
    "time_harmonic": "helmholtz",
    "scattering": "helmholtz",

    # ── Wave equation (second-order, time-domain) ──────────────────
    "wave_equation": "wave",
    "second_order_wave": "wave",
    "elastic_wave": "wave",
    "transient_wave": "time_dependent_wave",
    "unsteady_wave": "time_dependent_wave",

    # ── Eigenvalue / modal analysis ────────────────────────────────
    "vibration": "eigenvalue",
    "modal": "eigenvalue",
    "frequencies": "eigenvalue",
    "modes": "eigenvalue",
    "natural_frequencies": "eigenvalue",
    "eigenmode": "eigenvalue",
    "eigenfrequency": "eigenvalue",
    "buckling": "eigenvalue",
    "linear_buckling": "eigenvalue",

    # ── Poisson / Laplace / scalar elliptic ────────────────────────
    "diffusion": "poisson",
    "laplace": "poisson",
    "scalar": "poisson",
    "scalar_pde": "poisson",
    "steady_diffusion": "poisson",
    "electrostatic_field": "poisson",
    "elliptic": "poisson",

    # ── Convection-diffusion / scalar transport ────────────────────
    "transport": "convection_diffusion",
    "advection": "convection_diffusion",
    "advection_diffusion": "convection_diffusion",
    "scalar_transport": "convection_diffusion",
    "mass_transport": "convection_diffusion",
    "contaminant_transport": "convection_diffusion",
    "cd": "convection_diffusion",

    # ── DG (discontinuous Galerkin) ────────────────────────────────
    "discontinuous_galerkin": "dg_methods",
    "dg": "dg_methods",
    "ipdg": "dg_methods",
    "sipg": "dg_methods",
    "nipg": "dg_methods",
    "interior_penalty": "dg_methods",

    # ── Biharmonic / plate bending ─────────────────────────────────
    "plate": "biharmonic",
    "kirchhoff": "biharmonic",
    "kirchhoff_love": "biharmonic",
    "fourth_order": "biharmonic",
    "bending": "biharmonic",
    "kirchhoff_plate": "biharmonic",

    # ── Adaptive refinement (dealii / skfem / dune) ────────────────
    # canonical varies: adaptive_refinement (dealii), adaptive_poisson
    # (skfem, dune), hp_adaptive (dealii). _fuzzy_match_physics routes
    # the synonym only if it exists in this backend's catalog, so
    # mapping to adaptive_refinement first is safe — fall-through
    # picks the right one per backend.
    "amr": "hp_adaptive",
    "refinement": "hp_adaptive",
    "adaptive": "hp_adaptive",
    "h_refinement": "hp_adaptive",
    "p_refinement": "hp_adaptive",
    "hp_refinement": "hp_adaptive",
    "hp": "hp_adaptive",
    "error_estimator": "error_estimation",
    "kelly_estimator": "error_estimation",
    "kelly": "error_estimation",
    "adaptive_mesh": "hp_adaptive",
    "mesh_refinement": "hp_adaptive",

    # ── Phase field / fracture / damage ────────────────────────────
    "cahn_hilliard": "phase_field",
    "allen_cahn": "phase_field",
    "phase_field_fracture": "phase_field",
    "brittle_fracture": "fracture",
    "crack": "fracture",
    "crack_propagation": "fracture",
    "fracture_mechanics": "fracture",
    "damage_mechanics": "damage",
    "continuum_damage": "damage",

    # ── Topology / shape optimization ──────────────────────────────
    "topopt": "topology_optimization",
    "topology": "topology_optimization",
    "topology_opt": "topology_optimization",
    "shape_opt": "shape_optimization",
    "shape_optimisation": "shape_optimization",
    "structural_optimization": "topology_optimization",
    "compliance_minimization": "topology_optimization",

    # ── Contact / friction ─────────────────────────────────────────
    "friction": "contact",
    "contact_mechanics": "contact",
    "frictional_contact": "contact",
    "hertz": "contact",
    "mortar_contact": "contact",
    "node_to_surface": "contact",
    "surface_to_surface": "contact",

    # ── FSI / TSI / multiphysics coupling ──────────────────────────
    "fluid_structure": "fsi",
    "fluid_structure_interaction": "fsi",
    "thermo_structural": "thermal_structural",
    "thermomechanical": "thermal_structural",
    "multiphysics": "fsi",
    "coupling": "fsi",
    "thermal_solid_interaction": "tsi",
    "structural_thermal_interaction": "tsi",
    "soil_structure": "ssi",
    "structure_soil_interaction": "ssi",

    # ── Porous media / geomechanics ────────────────────────────────
    "poroelasticity": "porous_media",
    "poro": "porous_media",
    "consolidation": "porous_media",
    "terzaghi": "porous_media",
    "biot": "porous_media",
    "geomechanics": "porous_media",
    "saturated_porous": "porous_media",
    "unsaturated_porous": "porous_media",

    # ── Particle methods: peridynamics, SPH, DEM, MPM ──────────────
    "peridynamics": "particle_pd",
    "pd": "particle_pd",
    "bond_based": "particle_pd",
    "state_based": "particle_pd",
    "ordinary_state_based": "particle_pd",
    "non_ordinary_state_based": "particle_pd",
    "nosbpd": "particle_pd",
    "sph": "particle_sph",
    "smoothed_particle": "particle_sph",
    "smoothed_particle_hydrodynamics": "particle_sph",
    "discrete_element": "dem",
    "discrete_element_method": "dem",
    "granular": "dem",
    "material_point": "mpm",
    "material_point_method": "mpm",
    "particle_in_cell": "mpm",
    "pic": "mpm",
    "lagrangian_particles": "particle_sph",

    # ── Multiphase / free surface / VOF / level-set ────────────────
    "two_phase": "multiphase",
    "multi_phase": "multiphase",
    "vof": "multiphase",
    "volume_of_fluid": "multiphase",
    "immiscible": "multiphase",
    "interface": "multiphase",
    "free_surface_flow": "free_surface",
    "level_set_method": "level_set",
    "droplet": "droplet_dynamics",

    # ── Reaction-diffusion / chemical kinetics ─────────────────────
    "rd": "reaction_diffusion",
    "reaction_diffusion_system": "reaction_diffusion",
    "fitzhugh_nagumo": "reaction_diffusion",
    "gray_scott": "reaction_diffusion",
    "schnakenberg": "reaction_diffusion",
    "chemical_kinetics": "reaction_diffusion",

    # ── Structural dynamics / transient solid ──────────────────────
    "dynamics": "structural_dynamics",
    "transient_structural": "structural_dynamics",
    "dynamic_analysis": "structural_dynamics",
    "time_domain_structural": "structural_dynamics",
    "implicit_dynamics": "structural_dynamics",
    "explicit_dynamics": "structural_dynamics",
    "structural_transient": "structural_dynamics",

    # ── Schrödinger / quantum ──────────────────────────────────────
    "quantum": "schrodinger",
    "quantum_mechanics": "schrodinger",
    "wavefunction": "schrodinger",
    "eigenstate": "schrodinger",

    # ── MHD ────────────────────────────────────────────────────────
    "magnetohydrodynamics": "mhd",
    "magneto_hydrodynamics": "mhd",
    "plasma": "mhd",

    # ── Beams / shells / membranes ─────────────────────────────────
    "beam": "beams",
    "beam_element": "beams",
    "timoshenko": "beams",
    "euler_bernoulli": "beams",
    "shell_element": "shell",
    "kirchhoff_love_shell": "shell",
    "reissner_mindlin": "shell",
    "membrane_element": "membrane",

    # ── Cardiac / cardiovascular ───────────────────────────────────
    "cardiac": "cardiac_monodomain",
    "electrophysiology": "cardiac_monodomain",
    "monodomain": "cardiac_monodomain",
    "bidomain": "cardiac_monodomain",
    "cardiovascular": "cardiovascular0d",
    "windkessel": "cardiovascular0d",
    "lumped_parameter": "cardiovascular0d",
    "0d_model": "cardiovascular0d",

    # ── XFEM ───────────────────────────────────────────────────────
    "extended_fem": "xfem_fluid",
    "xfem": "xfem_fluid",
    "level_set_fem": "xfem_fluid",
    "embedded_interface": "xfem_fluid",

    # ── Reduced-order / multiscale ─────────────────────────────────
    "rom": "rom",
    "reduced_order": "rom",
    "reduced_order_modeling": "rom",
    "pod": "rom",
    "homogenization": "multiscale",
    "fe_squared": "multiscale",
    "fe2": "multiscale",

    # (Removed dead 'optimal_control' aliases — no backend provides that
    # canonical, even as reference knowledge, so they routed users to nothing.)

    # ── Matrix-free / multigrid (solver-level not physics, but
    #     dealii exposes them as physics keys) ─────────────────────
    "matrix_free_fe": "matrix_free",
    "geometric_multigrid": "multigrid",
    "algebraic_multigrid": "multigrid",
    "amg": "multigrid",
    "gmg": "multigrid",

    # ── HDG / HDivDiv / mixed methods ──────────────────────────────
    "hdg": "hdivdiv",
    "hybridizable_dg": "hdivdiv",
    "hellinger_reissner": "hdivdiv",
    "raviart_thomas": "mixed_poisson",
    "rt": "mixed_poisson",
    "bdm": "mixed_poisson",
    "mixed_finite_element": "mixed_poisson",
    "mixed_method": "mixed_poisson",
    "h_div_conforming": "mixed_poisson",

    # ── Hydraulics / shallow water ─────────────────────────────────
    "shallow_water_equations": "shallow_water",
    "saint_venant": "shallow_water",
    "swe": "shallow_water",

    # ── ALE ────────────────────────────────────────────────────────
    "arbitrary_lagrangian_eulerian": "ale",
    "moving_mesh": "ale",
}


# Queries shorter than this never participate in loose substring
# matches — short tokens collide with too many physics names /
# descriptions ('ns' is a substring of 'transient', 'em' of
# 'eigenvalue', 'pd' of 'pde'). For short tokens we trust ONLY
# exact-name and synonym-map matches. (Audit 2026-06-02.)
_MIN_LOOSE_MATCH_LEN = 4


def _fuzzy_match_physics(backend, query: str) -> str:
    """Fuzzy-match a physics query to an actual physics name in a backend.

    Resolution order (audit 2026-06-02):
      1. Empty -> return empty so caller can surface availables list.
      2. Exact physics-name match.
      3. Synonym map (e.g. 'ns' -> 'navier_stokes', 'em' -> 'maxwell',
         'thermal' -> 'heat'). Synonyms run BEFORE substring matching
         because short-token substrings collide constantly:
         'ns' is a substring of 'transient', 'em' of 'eigenvalue',
         'pd' of 'nonlinear_pde'. Without this ordering, LLMs that
         type the canonical shorthand silently got the wrong physics.
      4. Query is substring of a physics name (only if len >= 4).
      5. Physics name is substring of the query (only if the
         physics name is itself >= 4 chars — otherwise tiny names
         like 'pd' match every query containing those letters).
      6. Query is substring of a physics description (last resort,
         len >= 4).
      7. Fallthrough: return original so caller can produce a
         "no information found" message.
    """
    query_lower = query.lower().strip()

    # Empty / whitespace-only query — return verbatim so the
    # caller's "no information found" path can surface the
    # full available-physics list. Without this guard, the
    # next substring check matches "" to the FIRST physics
    # in the catalog (because "" is a substring of every
    # string) and the LLM silently sees prepare_simulation
    # output for a physics it never asked for. (Audit
    # 2026-06-01.)
    if not query_lower:
        return query_lower

    # 1. Direct match.
    for p in backend.supported_physics():
        if p.name == query_lower:
            return p.name

    # 1b. Separator-normalised direct match. Catalog keys use
    # underscores; an LLM writes the physics out in words. Before
    # 2026-08-03 the loose substring scan below then answered
    # 'navier stokes' with **stokes**, 'stokes darcy' with
    # **stokes**, and 'mixed poisson' with **poisson** — the wrong
    # physics, silently, with a plausible-looking payload. Mapping
    # spaces/hyphens/dots onto underscores and retrying the EXACT
    # name match (and the synonym map) can only ever tighten a
    # match, never loosen one, because it still requires equality
    # with a real catalog key.
    normalised = re.sub(r"[\s\-.]+", "_", query_lower)
    if normalised != query_lower:
        for p in backend.supported_physics():
            if p.name == normalised:
                return p.name

    # Method qualifiers outrank a shared generic noun. Without this, DUNE's
    # "advection diffusion ... SIPG" ties on one token with
    # reaction_diffusion and catalog order picks the reaction problem, while
    # 4C's "Crank-Nicolson transient heat" is captured by generic heat before
    # the transient One-Step-Theta recipe is considered.
    available = {p.name for p in backend.supported_physics()}
    tokens = set(normalised.split("_"))
    # 'convection' and 'advection' name the same transport term (Kratos
    # says convection, DUNE says advection). Without the alias the query
    # 'convection_diffusion' reaches the token-overlap fallback, ties
    # reaction_diffusion and dg_advection_diffusion at one shared token,
    # and catalog order picks the reaction problem (measured).
    if ("dg_advection_diffusion" in available
            and tokens.intersection({"advection", "convection"})
            and tokens.intersection({"diffusion", "sipg", "ipdg"})):
        return "dg_advection_diffusion"
    # AN ANISOTROPIC DIFFUSION IS A DIFFUSION WITH A TENSOR COEFFICIENT, NOT A
    # TRANSPORT PROBLEM. Measured: "anisotropic diffusion" resolved to
    # convection_diffusion on four codes and reaction_diffusion on DUNE-fem, and
    # two runs read transport knowledge for a steady conduction problem.
    if (tokens.intersection({"anisotropic", "orthotropic", "tensor"})
            and tokens.intersection({"diffusion", "conduction", "conductivity", "diffusivity"})
            and not tokens.intersection({"advection", "convection", "reaction", "transport"})):
        for _name in ("heat", "poisson"):
            if _name in available:
                return _name
    if ("thermo_transient_mms" in available
            and tokens.intersection({"heat", "thermal", "thermo"})
            and tokens.intersection({"transient", "crank", "theta"})):
        return "thermo_transient_mms"
    if ("nearly_incompressible_elasticity" in available
            and "elasticity" in tokens
            and (({"nearly", "incompressible"} <= tokens)
                 or ({"taylor", "hood"} <= tokens)
                 or ({"mixed", "pressure"} <= tokens))):
        return "nearly_incompressible_elasticity"

    # 2. Synonym map — BEFORE the substring scan so short
    # canonical shorthands ('ns', 'em', 'pd') route to the
    # right physics. Only return the synonym if it actually
    # exists in this backend's catalog; otherwise fall through
    # to the loose matchers (a backend that has 'maxwell' but
    # not the synonym should still match via substring).
    mapped = _PHYSICS_SYNONYMS.get(query_lower) or _PHYSICS_SYNONYMS.get(normalised)
    if mapped:
        for p in backend.supported_physics():
            if p.name == mapped:
                return p.name

    # 3. Loose substring of physics name (only for non-short
    # queries — see _MIN_LOOSE_MATCH_LEN rationale above).
    if len(query_lower) >= _MIN_LOOSE_MATCH_LEN:
        for p in backend.supported_physics():
            if query_lower in p.name.lower():
                return p.name

    # 4. Physics name is substring of query (only when the
    # physics name itself is non-trivial). Without the length
    # guard, a 2-char catalog entry like 'pd' matches every
    # query containing those letters, which is the same
    # collision class we just guarded the other direction
    # against.
    #
    # THE MOST SPECIFIC NAME WINS, and the words are read with their
    # separators normalised too. Measured 2026-09-28 in the web interface:
    # "transient incompressible Navier-Stokes" matched `stokes` -- a substring
    # of "navier-stokes", where the hyphen kept `navier_stokes` from matching --
    # and the run was served the Stokes deciding facts for a Navier-Stokes
    # problem. Catalog order decided which name a query holding two got.
    hits = [p.name for p in backend.supported_physics()
            if len(p.name) >= _MIN_LOOSE_MATCH_LEN
            and (p.name.lower() in query_lower or p.name.lower() in normalised)]
    if hits:
        return max(hits, key=len)

    # 5. Loose substring of physics description (last resort,
    # same length guard).
    if len(query_lower) >= _MIN_LOOSE_MATCH_LEN:
        for p in backend.supported_physics():
            if query_lower in p.description.lower():
                return p.name

    # 6. Token-overlap fallback. Compound synonyms like
    # 'rarefied_gas_dynamics' -> 'rarefied_flow' or 'hypersonic_cht'
    # -> 'hypersonic_flow' share a DISCRIMINATING token ('rarefied',
    # 'hypersonic', 'conjugate') but neither string is a substring of
    # the other, so steps 3-5 miss them. Match on shared significant
    # tokens (>= 5 chars to avoid 'flow'/'heat'/'grid' collisions) and
    # pick the physics with the most shared tokens. Discovered when a
    # 72B agent asked SPARTA for 'rarefied_gas_dynamics' and got a raw
    # "unknown physics" error instead of the worked rarefied_flow deck.
    q_tokens = {t for t in re.split(r"[^a-z0-9]+", query_lower) if len(t) >= 5}
    if q_tokens:
        best, best_n = None, 0
        for p in backend.supported_physics():
            p_tokens = {t for t in re.split(r"[^a-z0-9]+", p.name.lower())
                        if len(t) >= 5}
            n = len(q_tokens & p_tokens)
            if n > best_n:
                best, best_n = p.name, n
        if best:
            return best

    # Nothing matched — return original so the caller can
    # surface the "no information found" message with the
    # available-physics list.
    return query_lower


def _list_alternative_solvers(current_solver: str, physics: str) -> str:
    """List other backends that also support this physics (informational).

    This helps the agent know what alternatives exist if the chosen solver
    runs into issues, without being prescriptive about which to use.
    """
    alternatives = []
    for b in all_backends():
        if b.name() == current_solver:
            continue
        status, _ = b.check_availability()
        for p in b.supported_physics():
            if p.name == physics or physics in p.name or p.name in physics:
                # Tag unavailable backends so the LLM knows
                # they would need to be installed first. Hiding
                # them silently (the old available_backends()
                # behaviour) made dune-fem and febio
                # alternatives invisible. (Audit 2026-06-02.)
                tag = "" if status.value == "available" else f" *[{status.value}]*"
                alternatives.append(
                    f"- **{b.display_name()}**{tag}: {p.description}")
                break
    if not alternatives:
        return ""
    return "Other solvers that support this physics:\n" + "\n".join(alternatives)


def _narrow_coupling_by_signal(groups: dict, signal: str) -> dict:
    """Keep the coupling entries whose recorded symptom matches an observed one.

    DELEGATES to `core.pitfall_index` whenever that module is importable. That
    is the canonical matcher for the whole corpus — it folds quoting,
    whitespace and case, stems inflections, and carries a domain synonym table
    — and a second implementation of the same thing that drifted would be worse
    than none, because a symptom query that quietly matches differently for
    coupling than for every backend is a trap rather than a feature.

    The local fallback below exists only for trees where that module is not
    present yet. It mirrors the canonical matcher's TIERS — verbatim in the
    recorded symptom, verbatim anywhere in the entry, every distinctive query
    word present, and a labelled-weak majority overlap — because a fallback
    that is quietly STRICTER is the more dangerous kind of wrong. Measured
    here: a first version tested only for a full token subset, so paraphrased
    queries the canonical matcher surfaces as weak leads ("the interface flux
    balances to roundoff but the result is wrong") came back as "no recorded
    failure mode matches", and for a silent-wrong mode an authoritative-sounding
    absence is exactly the answer that gets an agent to trust a converged run.
    It carries no synonym table, so it can still never claim a match the
    canonical matcher would not.

    RESULTS ARE RANKED, and ties break towards the SHORTER entry. Unranked
    output meant the first entry of the first group won every tie, and the
    longest, most-general entry collects the most token matches — so the one
    entry that mentions everything was answering queries that belonged to its
    neighbours. Preferring the entry with the smaller vocabulary is a
    specificity tie-break, and the canonical matcher has the same tie problem
    (measured: a query of three generic words returned 19 candidates ranked by
    corpus order).

    Never returns an empty result silently: a query that matches nothing comes
    back with a note saying so, because an empty answer reads as "nothing is
    known about that", which is a different and much more dangerous claim.
    """
    try:                                       # canonical path
        from core import pitfall_index
    except ImportError:
        pitfall_index = None

    if pitfall_index is not None:
        result = pitfall_index.narrow(groups, signal=signal)
        kept = {}
        for e in result["entries"]:
            kept.setdefault(e["physics"], []).append(
                f"{e['text']}  <- match: {e.get('match', '?')}")
        if not kept:
            return {"no_match": [
                f"No recorded coupling failure mode matches {signal!r}. That is "
                "informative but not conclusive: it means this symptom is not "
                "catalogued, NOT that your coupling is right. "
                "knowledge(topic='pitfalls', solver='coupling') returns all "
                f"{result['total_available']} entries."]}
        modes = result.get("match_modes") or {}
        note = (f"signal={signal!r}: {result['shown']} of "
                f"{result['total_available']} entries, best match first. For "
                f"the complete set: knowledge(topic='pitfalls', "
                f"solver='coupling')")
        if modes and set(modes) <= {"some_tokens"}:
            note += (". EVERY match below is a partial word overlap, not a "
                     "match on a recorded symptom — treat them as leads to "
                     "read, not as an identification of your failure.")
        kept["_filter"] = [note]
        return kept

    import re as _re

    def _norm(s: str) -> str:
        s = s.lower().replace("’", "'").replace("‘", "'")
        s = s.replace("“", '"').replace("”", '"')
        s = _re.sub(r"[\\'\"`]+", "", s)
        return _re.sub(r"\s+", " ", s).strip()

    _STOP = set("a an the and or but if of in on at to for from with by is are "
                "was were be it its this that these those as not no so than "
                "then there when where which what how why all any both each "
                "more most other some such only same too very can will just "
                "should now use used using you your we our error warning "
                "message output file files line lines code".split())
    def _toks(s: str) -> set[str]:
        return {w for w in _re.findall(r"[a-z_][a-z0-9_]{2,}", s)
                if w not in _STOP}

    q = _norm(signal)
    qt = _toks(q)
    scored: list[tuple[float, int, str, str, str]] = []
    for group, entries in groups.items():
        for text in entries:
            whole = _norm(text)
            sig = whole.split("signal:", 1)[1] if "signal:" in whole else ""
            wt = _toks(whole)
            hit = qt & wt
            frac = len(hit) / len(qt) if qt else 0.0
            if sig and q in sig:
                score, mode = 1.0, "matches recorded symptom"
            elif q in whole:
                score, mode = 0.85, "matches entry text"
            elif qt and frac >= 1.0:
                score, mode = 0.7, "all query terms present"
            elif frac >= 0.6 and len(hit) >= 2:
                # Labelled WEAK, never presented as an identification. Below
                # 0.6, or on a single word, an overlap is a coincidence.
                score, mode = 0.3 + 0.3 * frac, "WEAK: partial term overlap"
            else:
                continue
            # Tie-break on entry vocabulary size: the shorter entry is the more
            # specific one, and without this the longest entry wins every tie.
            scored.append((score, len(wt), group, text, mode))
    scored.sort(key=lambda t: (-t[0], t[1]))
    kept: dict[str, list[str]] = {}
    for _score, _n, group, text, mode in scored:
        kept.setdefault(group, []).append(f"{text}  <- match: {mode}")
    total = sum(len(v) for v in groups.values())
    if not kept:
        return {"no_match": [
            f"No recorded coupling failure mode matches {signal!r}. That is "
            "informative but not conclusive: it means this symptom is not "
            "catalogued, NOT that your coupling is right. "
            f"knowledge(topic='pitfalls', solver='coupling') returns all "
            f"{total} entries."]}
    shown = sum(len(v) for v in kept.values())
    note = (f"signal={signal!r}: {shown} of {total} entries, best match first. "
            f"For the complete set: knowledge(topic='pitfalls', "
            f"solver='coupling')")
    if all(m.startswith("WEAK") for *_ , m in scored):
        note += (". EVERY match below is a partial word overlap, not a match on "
                 "a recorded symptom — treat them as leads to read, not as an "
                 "identification of your failure.")
    kept["_filter"] = [note]
    return kept


def _load_matching_postmortems(solver: str = "", physics: str = "",
                               signal: str = "") -> list[dict]:
    """Load post-mortem JSONs from data/postmortems/, filtered.

    The post-mortems directory is the audit trail behind the pitfall
    DB. Each record explains WHY a pitfall was added — the surface
    symptom that was observed, the root cause, the Table-1 category,
    the exact pitfall entries shipped, and the detection path the
    agent now has. The openPASO design paper's §3.2 / §5
    self-correction loop depends on the agent being able to retrieve
    these at planning time.

    Filters (any can be empty, treated as "match all"):
      * solver  — exact match against the post-mortem's `backend`
                  field (case-insensitive). NOT a fuzzy match because
                  the post-mortem's audit value depends on knowing
                  it's about THIS backend, not a similar one.
      * physics — substring match against the `physics` field. A
                  batch post-mortem like
                  "poisson, heat, helmholtz, eigenvalue" matches any
                  of its members.
      * signal  — substring match across each `pitfall_db_entries`
                  string. Useful when the post-execution critic
                  sees a specific error and wants to find the
                  matching post-mortem.

    Returns the post-mortems as parsed dicts. Sorted by `date`
    descending so the most-recent record comes first — typically
    the most-relevant for the current agent session.

    Files under ``data/postmortems/candidates/`` are NOT included
    here. Candidates are the pre-review staging area
    (the openPASO design paper §3.2 autonomous-growth path) — promotion to a
    formal post-mortem is a deliberate review step (#46).
    """
    from core.paths import data_dir as _data_dir
    pm_dir = _data_dir() / "postmortems"
    if not pm_dir.is_dir():
        return []
    solver_l = solver.lower().strip()
    physics_l = physics.lower().strip()
    signal_l = signal.lower().strip()
    out: list[dict] = []
    for path in pm_dir.glob("*.json"):
        if path.name.startswith("_"):
            # Skip schema / index files.
            continue
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        if solver_l and str(doc.get("backend", "")).lower() != solver_l:
            continue
        if physics_l and physics_l not in \
                str(doc.get("physics", "")).lower():
            continue
        if signal_l:
            entries = doc.get("pitfall_db_entries", []) or []
            if not any(signal_l in str(e).lower() for e in entries):
                continue
        out.append(doc)
    out.sort(key=lambda d: str(d.get("date", "")), reverse=True)
    return out


def _make_input_snapshot(input_content: str, solver: str = "",
                         extra: dict | None = None) -> dict:
    """Create a sanitised snapshot of simulation input for diff capture.

    Captures structure (length, line count, key patterns) without leaking content.
    """
    import hashlib
    snap = {
        "solver": solver,
        "input_length": len(input_content),
        "input_lines": input_content.count("\n") + 1,
        "input_hash": hashlib.sha256(input_content.encode()).hexdigest()[:12],
    }
    if extra:
        snap.update(extra)
    return snap


def _unsaved_work_notice(work_dir, out_files) -> str | None:
    """Warn, at the moment a solve SUCCEEDS, that nothing is written down yet.

    Measured over the recorded runs: the runs that fail most often are
    not the ones that fail to solve. They solve, keep working, and end with
    nothing a reader can find — three of the five failing single-code runs in
    one batch made 85-107 tool calls, produced solver output, and wrote no
    summary at all.

    Two static-text attempts did not change that: the rule filed in one
    backend's table, then the same rule appended to every knowledge payload.
    Both are read once, at the start, before there is anything to write. This
    notice is different in kind rather than in wording — it is stateful and
    just-in-time, raised at the one moment when numbers demonstrably exist and
    are demonstrably not saved.

    Deliberately GENERAL: it names no filename and assumes no benchmark. It
    reports what the run produced and that no human-readable summary sits
    beside it, which is true for any user of this server.
    """
    from pathlib import Path
    if not out_files:
        return None
    try:
        d = Path(work_dir)
        if not d.is_dir():
            return None
        summaries = [q for q in d.rglob("*")
                     if q.is_file()
                     and q.suffix.lower() in (".txt", ".md", ".csv", ".json")
                     and q.name not in ("imports.json", "exports.json")
                     and not q.name.startswith("trajectory")]
    except OSError:
        return None
    if summaries:
        return None
    return ("This run produced solver output and there is no written summary "
            "beside it yet. Whatever your task asks you to report, write it "
            "now, from the numbers you have, before doing anything else — "
            "including before investigating anything that looks wrong. A "
            "result that exists only in this conversation is not a result: if "
            "the session ends here, the run leaves nothing. If you have "
            "more levels or cases to run, write the summary now with what you "
            "have and an honest marker for what is missing, then rewrite it "
            "after each later run. Rewriting a small text file costs nothing "
            "next to a solve.")


# THE UNIVERSAL BLOCK MUST RIDE THE TOOL AGENTS ACTUALLY CALL. It lived only
# in knowledge.register_knowledge_tools, which src/server.py never registers —
# so every rule placed there over a long stretch of development
# (write-the-deliverable, the audit_results instruction, the degree rule) was
# served to NOBODY, while I "verified 9/9 backends" through
# get_physics_knowledge, a tool the live server does not expose. Agents call
# `knowledge` 2909 times in the development-run transcripts and
# get_physics_knowledge zero times.
import functools as _functools
from .knowledge import _PER_SIDE_NAMING  # one copy, served on both paths                                  # noqa: E402
from .knowledge import _physics_tail                            # noqa: E402
from .knowledge import _UNIVERSAL as _UNIVERSAL_BLOCK          # noqa: E402
from .knowledge import _UNIVERSAL_CORE as _UNIVERSAL_CORE      # noqa: E402


# THE FIRST 1500 CHARACTERS DECIDE THE RUN, SO THEY GO FIRST.
#
# Measured over every door an agent can open for a coupled task: 145,321
# characters served, of which the lines carrying a decisive fact total 571 --
# 0.4%. One prepare_simulation reply is 44,000 to 83,000 characters, and the
# coupled runs call it twice. File mtimes then show 5 of 6 delivering their
# files at 93-99% of their whole file-activity span; the only run that ever
# reached a verifiable order with both prescribed codes proven to have run is
# the one that delivered at 68%.
#
# The corpus is not removed -- it is reordered. An agent that reads only the
# top of the reply now gets the facts that separate a correct result from a
# confidently wrong one, each with the measurement behind it.
# One entry per live server: which backends prepare_simulation has seen.
# Keyed by id(mcp) so parallel servers in one process cannot bleed into
# each other; a second key (id, 'served') marks the must-read as sent.
_PREPARED_SOLVERS: dict = {}

# THE COUPLED MUST-READ IS SERVED ONCE PER SESSION. Measured on a small model:
# a seven-call sequence (discover, three knowledge calls, two
# prepare_simulation calls, one more knowledge call) carried FIVE copies of
# the ~24k-character must-read -- about 100k tokens of served text in all --
# and the run gave up citing complexity with 33 minutes of its 45 left. The
# rules do not change between calls; repeating them only fills the context
# the agent needs for its own work. One server process is one session, so a
# module flag is the session; `register_consolidated_tools` resets it.
_MUST_READ_STATE: dict = {"served": False}
# TRUE FOR A SUB-AGENT TOO. A sub-agent shares this server process but not its
# parent's context, and "was served earlier in this session" told one it had read
# a page it never saw (measured: a critic sub-agent then fetched the 75 kB page and
# its next reply ran to the output cap).
_MUST_READ_POINTER = (
    "THE COUPLED MUST-READ (the couple() call recipe, history_path, the "
    "fields-vs-evidence rule, the measured-not-modelled history rule, the "
    "sign convention, the rho budget, the silent no-flux trap) was served to "
    "this session -- to you, or to the agent that started you -- and is not "
    "repeated here. If your own conversation lacks it: "
    "knowledge(topic='coupling', signal='must-read').\n\n")


_DECIDING_FACTS = {
    "fourc": (
        "1. A standalone `Thermo` problem SILENTLY IGNORES every "
        "`DESIGN ... THERMO ...` Dirichlet/Neumann section: it parses, prints "
        "'processor 0 finished normally', exits 0, and the field is "
        "IDENTICALLY ZERO (measured max|T| = 0.000000000e+00). Use the PLAIN "
        "sections -- DESIGN SURF/LINE/POINT DIRICH, DESIGN VOL/SURF/LINE "
        "NEUMANN. With those the same deck matches an independent assembly to "
        "1.08e-15.\n"
        "2. The body source lives on the condition whose GEOMETRY TYPE MATCHES "
        "THE ELEMENT DIMENSION: LINE in 1D, SURF in 2D, VOL in 3D. The wrong "
        "one attaches to no element and contributes nothing, silently.\n"
        "3. Every load entry is ONOFF[i] * VAL[i] * FUNCT[i](x,t) -- FUNCT "
        "SCALES VAL. `VAL: 0` with a function set is an identically zero load "
        "that parses and exits 0 (measured: error flat at 1.0000 across all "
        "levels, order exactly 0.0000).\n"
        "4. STRUCTURE runtime-VTK carries `node_gid` (with NODE_GID: true), "
        "0-BASED while NODE "
        "COORDS ids are 1-based, and the .vtu holds one point PER ELEMENT "
        "CORNER (160 points, 54 distinct gids, for a 40-element mesh). "
        "Scatter results by gid: `out[int(gid)] = value`. Zipping instead "
        "gives a field with the RIGHT maximum and 67% pointwise error. "
        "The SCATRA runtime VTU has NO `node_gid` array at all -- its "
        "point_data is only `phi_1` (plus `flux_boundary_phi_1` when "
        "requested; measured KeyError on a 240-point scatra VTU) -- so on "
        "scatra output collapse the duplicated points BY COORDINATE "
        "instead, in one call: `_, first = np.unique(np.round(pts[:, :2], 10), "
        "axis=0, return_index=True)`, then pts[first] and phi[first] (a pairwise "
        "loop over the points is O(N^2); measured, 26 s per run).\n"
        "5. A spatially varying interface trace needs one DESIGN POINT DIRICH "
        "condition PER NODE -- no fitted FUNCT required.\n"
        "6. INVOKE IT AS `stdbuf -oL -eL /path/to/4C deck.4C.yaml out` or as `mpirun -np 1 /path/to/4C deck.4C.yaml out` "
        "-- from INSIDE your participant script: the participant you hand to couple() and to the critic review is that "
        "Python script, never the binary. "
        "The binary finds its libraries by itself (rpath-linked; measured to run with LD_LIBRARY_PATH unset), so add no prefix -- and if you ever do add one, an assignment must come BEFORE the wrapper: `stdbuf -oL VAR=x prog` makes stdbuf try to execute a file called `VAR=x` and your command never runs. "
        "4C's stdout is BLOCK-BUFFERED, and when a deck is "
        "rejected MPI_Abort tears the process down before that buffer is "
        "flushed, so the line naming the defect is destroyed and all you get "
        "back is the MPI boilerplate. Measured on one rejected deck: plain "
        "capture and `2>&1` both 429 bytes with NO reason; `stdbuf -oL -eL` "
        "and `mpirun -np 1` 2164 bytes carrying `PROC 0 ERROR in "
        "4C_io_input_file.cpp, line 546: Section 'NOT_A_REAL_SECTION' is not "
        "a valid section name.`. A bare `MPI_ABORT ... errorcode 1` with an "
        "empty stdout is NOT an MPI or environment problem -- it is your deck, "
        "and the reason is one flag away.\n"
        "7. `No protocol specified` and `Invalid MIT-MAGIC-COOKIE-1 key` on "
        "stderr are X11 noise from a headless session. They are not the "
        "failure and they appear on successful runs too."
        # Facts 8-9 measured by execution 2026-09-04 (4C 2026.2.0-dev,
        # 89519cfe76): CALCFLUX route, VTU step-0 trap. repr literals.
        '\n8. Boundary flux from a scatra solve: set CALCFLUX_BOUNDARY: "diffusive" in SCALAR TRANSPORT DYNAMIC **and** add a `SCATRA FLUX CALC LINE CONDITIONS:` entry (`- E: <line id>`; SURF in 3D) for every boundary line the flux is wanted on. Without that condition section 4C stops: \'Flux output requested without corresponding boundary condition specification!\'. The flux lands in the runtime VTU as point array `flux_boundary_phi_1` -- the diffusive flux VECTOR q = -k*grad(phi) at boundary nodes, NOT q.n (measured exact: (-1.6,0,0) on both x-faces of a P1-exact linear field; dot it with YOUR outward normal). THAT DESCRIPTION HOLDS ON DIRICHLET LINES. On a NEUMANN-loaded line the same array carries the consistent-residual echo of the APPLIED load (measured: exactly 0 under zero load while the true boundary gradient of the field was O(1e-3)) plus Dirichlet-corner reactions at the ends -- use it as your exported flux only on the side whose interface is Dirichlet. It agrees with one-sided quadratic differentiation of the field to 2.7% rel-RMS at h=1/8 on a smooth field.'
        "\n9. The runtime VTU numbered 00000 is the INITIAL state -- identically zero on a fresh scatra run; the solved field is the LAST step (00001 for stationary). Sampling step 0 yields an all-zero field and all-zero fluxes while the run exits 0 and prints 'finished normally'."
        # Facts 10-11 measured by execution 2026-09-04 during the two-material
        # conduction coupled walk (4C 2026.2.0-dev, 89519cfe76).
        "\n10. With CALCFLUX_BOUNDARY active, 4C also writes "
        "`<prefix>.boundaryflux_ScaTraFluxCalc_0scatra.txt` -- per flux-calc "
        "condition: area, integral and MEAN of the normal flux. It is a free "
        "cross-check on any nodal flux you exported (measured: nodal "
        "trapezoid integral matched it to 6 digits, -0.5558378 vs "
        "-0.555838).\n"
        "11. P1 triangles in scalar transport are spelled `<id> TRANSP TRI3 "
        "<n1> <n2> <n3> MAT <m> TYPE Std` in TRANSPORT ELEMENTS -- the "
        "corpus templates show only TRANSP QUAD4, and SOLID TRI3 is a "
        "structural element that is rejected against MAT_scatra."
        # Facts 12-13 measured by execution 2026-09-04 on the
        # thermo-mechanical coupled walk (4C 2026.2.0-dev, 89519cfe76).
        "\n12. STEADY THERMO-MECHANICS IN ONE 4C RUN: PROBLEMTYPE Thermo_Structure_Interaction with COUPALGO tsi_oneway, Statics in both STRUCTURAL DYNAMIC and THERMAL DYNAMIC, material MAT_Struct_ThermoStVenantK (stress C:(eps - alpha*(T-T0)*I), i.e. sigma_el - beta*T*I with beta=(3*lambda+2*mu)*alpha and T0 from INITTEMP) plus a CLONING MATERIAL MAP entry pairing it with a MAT_Fourier thermal material. A 2D plane-strain problem runs as a ONE-ELEMENT-THICK SOLIDSCATRA HEX8 slab with u_z=0 pinned on BOTH z-layers (per-node POINT DIRICH) -- that is exact plane strain, not an approximation. Thermal body sources go in as DESIGN VOL THERMO NEUMANN conditions. 4C's VTU carries NO mechanical reaction forces, but its DIRICHLET MONITOR does: `TAG: monitor_reaction` on each interface DESIGN POINT DIRICH entry plus an `IO/MONITOR STRUCTURE DBC` section (INTERVAL_STEPS 1, FILE_TYPE yaml, WRITE_CONDITION_INFORMATION true) writes <out>-<id>_monitor_dbc.yaml per condition with the node gid (ZERO-based: gid 17 is the deck's NODE 18) and the reaction f. At an interior interface node of the slab (f_layer0 + f_layer1)/(h*t_z) IS the traction in the flux convention -(sigma.n_out): measured 1.45e-2, 3.6e-3, 9.0e-4 relative at h = 1/10, 1/20, 1/40 against a manufactured thermo-elastic solution (order 2.0). Thermal DIRICH entries write no reaction file, so the consistent heat flux comes from a Scalar_Transport run with CALCFLUX_BOUNDARY on the same 2-D mesh; knowledge(topic='coupling', solver='fourc', physics='thermoelastic') serves the two-run contract's handshake and recovery; the decks are yours."
        '\n13. WHERE 4C EVALUATES A FUNCT LOAD DIFFERS BY PROBLEM TYPE, and the difference is O(h^2) in the solution: scatra SURF NEUMANN with a FUNCT source assembles the INTERPOLATED load M*f(nodes) (matches that discrete system to 1.7e-15); TSI VOL THERMO NEUMANN evaluates f at the 2x2 GAUSS POINTS (matches to ~1e-15). The two discrete solutions differ by 1.4e-2 at h=1/8, shrinking O(h^2). Consequence: a CALCFLUX boundary flux is the exact reaction of ITS OWN discrete system; compare it only against a re-assembly using the SAME load rule, or the mismatch (5.3e-2 at h=1/8 here) reads as a recovery bug that is not there.'
        "\n14. FIRST-ATTEMPT DECK TRAPS MEASURED ON TWELVE WORKER DECKS (2026-09-11), each one stops 4C in its input reader: (a) ONE topology section per kind -- every DNODE/DLINE/DSURF/DVOL entry of the deck goes into the single `DNODE-NODE TOPOLOGY` (etc.) section; a second section of the same name is 'defined more than once'; (b) the consistent heat flux comes from PROBLEMTYPE Scalar_Transport as in 8, never PROBLEMTYPE Thermo, which knows no CALCFLUX_BOUNDARY; (c) runtime output sections are exactly `IO/RUNTIME VTK OUTPUT`, `IO/RUNTIME VTK OUTPUT/STRUCTURE` and `THERMAL DYNAMIC/RUNTIME VTK OUTPUT` -- there is no .../SCATRA or .../THERMO variant (the scatra VTU comes from the plain section); (d) SOLIDSCATRA, WALL, SOLID are ELEMENT TYPES on the lines of `STRUCTURE ELEMENTS`, TRANSP of `TRANSPORT ELEMENTS`; a section named after the element type does not exist; (e) MAT_Struct_ThermoStVenantK takes YOUNGNUM 1 and YOUNG as a LIST ([E]), NUE, DENS, THEXPANS, INITTEMP and THERMOMAT <id of the MAT_Fourier>; (f) a temperature value belongs in the `... THERMO DIRICH CONDITIONS` family (NUMDOF 1); the plain `DESIGN POINT/LINE/SURF DIRICH CONDITIONS` family carries the 3 displacement dofs (NUMDOF 3); (g) 2-D element nodes run counter-clockwise: for node id = i + 1 + (NX + 1) j the quad of cell (i, j) is (id, id+1, id+NX+2, id+NX+1) -- a twisted quad has zero area and 4C says only 'determinant ... zero or negative'. Every one of these is named the moment you WRITE the deck: a deck written to a .yaml, .yml or .dat file is read and judged in the same reply, before the binary runs, so you do not have to ask for the check. Where a session also exposes it, check_input(solver='fourc', input_path=<deck>) is the same judgement on demand."
        # Fact 14 measured by execution 2026-09-05 (4C 2026.2.0-dev).
        + '\n15. A SAMPLED Neumann profile needs no polynomial fit: `DESIGN POINT NEUMANN CONDITIONS` works for Scalar_Transport with pre-integrated nodal loads -- per interior interface node F_i = h/6*(g_{i-1} + 4*g_i + g_{i+1}), FUNCT [0]. Delivery proven by the zero-vs-real load check (fields differ by 4.1e-3 at N=8) and the field converges at order ~1.95. The LINE NEUMANN + fitted-FUNCT route also works but silently smooths any profile the fit cannot represent.'),
    # Every line measured by execution on this install (dolfinx 0.10.0,
    # ufl 2025.2.1) on 2026-09-03. repr-generated literal: the measured
    # text contains brace/quote sequences that hand-escaping kept
    # breaking.
    "fenics": "1. `ufl.FiniteElement` NO LONGER EXISTS (dolfinx 0.10 / ufl 2025.2: AttributeError; the lowercase hint `ufl.finiteelement` is NOT what you want either). Build spaces the modern way -- fem.functionspace(mesh, ('Lagrange', 1)) with lowercase f, or basix.ufl.element('Lagrange', 'triangle', 1). Both measured working on this install.\n2. `LinearProblem` REQUIRES the keyword `petsc_options_prefix` on this install (TypeError without it). Measured working:\n       p = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc],\n           petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'},\n           petsc_options_prefix='run')\n       uh = p.solve()\n   Its solver is the PUBLIC `p.solver`; touching `p._solver` raises AttributeError (one run died on exactly that).\n3. `ufl.Constant` TAKES A DOMAIN AND A SHAPE, NOT A VALUE: ufl.Constant(0.0) raises AttributeError: 'float' object has no attribute 'ufl_domain', and ufl.Constant(mesh, np.array([0.0, 0.0])) reads the array as the SHAPE (measured). A constant INSIDE a form is fem.Constant(mesh, value).\n4. THE RECTANGLE CONSTRUCTOR TAKES THE CORNERS AS A LIST OF POINTS AND THE COUNTS AS A SEQUENCE: dmesh.create_rectangle(MPI.COMM_WORLD, [[X0, Y0], [X1, Y1]], [NX, NY], dmesh.CellType.triangle) -- measured signature (comm, points, n, cell_type, ...). Building a unit square and rescaling domain.geometry.x by hand is not the same thing and a worker lost a run to it.\n5. SELECT DOFS WITH fem.locate_dofs_topological(V, fdim, facets) (facets from mesh.locate_entities_boundary(domain, fdim, marker), fdim = domain.topology.dim - 1) or fem.locate_dofs_geometrical(V, marker). IT RETURNS THE ARRAY ITSELF FOR ONE SPACE -- do NOT index it with [0]. Measured: for a single space the result is an ndarray of shape (n,), and [0] is the first dof NUMBER, so everything downstream silently becomes one point; only when you pass a LIST of two spaces does it return a pair of arrays. A worker lost a run to exactly that [0]. There is NO V.subset_dofs -- AttributeError, measured, and one run invented exactly that. The condition is fem.dirichletbc(value, dofs, V), the value of the space's shape -- np.zeros(gdim, dtype=default_scalar_type) on a vector space, where a plain number raises Rank mismatch between Constant and function space -- or fem.dirichletbc(g, dofs) for a Function (with V too: incompatible function arguments). A block of a mixed space takes the Function WITH the sub-space: V0, _ = W.sub(0).collapse(); fem.dirichletbc(g, fem.locate_dofs_topological((W.sub(0), V0), fdim, facets), W.sub(0)), g on V0; a constant there raises 'Constant size is not equal to the block size'. All measured.\n6. dolfinx is SILENT by default: before creating the mesh, call dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO) -- the DOLFINX_LOGLEVEL environment variable is NOT honoured, and a run whose console output stays empty cannot show which code ran.\n7. Evaluate a Function at arbitrary points with the bb-tree route: bb = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim); cand = dolfinx.geometry.compute_collisions_points(bb, pts); cells = dolfinx.geometry.compute_colliding_cells(mesh, cand, pts); then uh.eval(pts, first_cell_per_point). Nearest-DOF lookup is the export defect that turns a converged solve into a wrong answer.\n8. FIRST USE COMPILES TOO: dolfinx JIT-compiles every new form with ffcx (a minute or more the first time, more under load). A short `timeout` around that first run, or a pipe into `head`, kills it mid-compile and looks like a crash. Run each participant once standalone with a generous timeout before coupling; the cached modules make later runs start in seconds.\n9. `fem.VectorFunctionSpace` DOES NOT EXIST on this install (AttributeError, measured dolfinx 0.10): a vector P1 space is fem.functionspace(mesh, ('Lagrange', 1, (2,))), and in its array component c of node n sits at index 2*n + c (tabulate_dof_coordinates() has one row per node). A vector fem.Function is interpolated from a callable returning shape (2, n) -- np.vstack((fx, fy)) -- and its TRANSPOSE fails with 'Interpolation data has the wrong shape/size' (measured).\n10. UFL arguments have no `.geometric_dimension()` (AttributeError, measured ufl 2025.2): write ufl.Identity(2) for plane strain; ufl.sym(ufl.grad(u)) is the strain. fem.dirichletbc takes (Function, dofs) or (Constant, dofs, V) on a plain space -- a Constant WITHOUT the space as third argument is a TypeError (measured).\n11. INTERPOLATION IS A METHOD OF THE FUNCTION: f = fem.Function(V); f.interpolate(lambda X: <expression of X[0], X[1]>) with X the (3, n) coordinate array. There is NO module-level fem.interpolate(callable, V) (AttributeError, measured), and a lambda with two arguments (x, y) is a TypeError.\n12. UFL FORMS ARE PYTHON EXPRESSIONS: scalar products are ufl.inner(a, b) (or ufl.dot), products are `*`, integrals are `<integrand> * ufl.dx` and `<integrand> * ds_measure`; the strain is ufl.sym(ufl.grad(u)), the trace ufl.tr(...), the divergence ufl.div(u). There is no `.` operator between UFL objects: a form written as `ufl.grad(u) . ufl.grad(v)` is Python attribute access and dies with \"'Grad' object has no attribute 'ufl'\" (measured on a worker script).\n13. A MIXED (TAYLOR-HOOD) SPACE COMES FROM basix, AND EVERY LEGACY SPELLING IS GONE. Measured on this install (dolfinx 0.10 / basix 0.10 / ufl 2025.2): `ufl.MixedElement` and `ufl.VectorElement` both raise AttributeError: module 'ufl' has no attribute 'MixedElement' / 'VectorElement', and `fem.FunctionSpace` with a capital F is TypeError: FunctionSpace.__init__() missing 1 required positional argument: 'cppV'. What this install accepts:\n       Ve = basix.ufl.element('Lagrange', mesh.basix_cell(), <deg_v>, shape=(gdim,))\n       Qe = basix.ufl.element('Lagrange', mesh.basix_cell(), <deg_q>)\n       W  = fem.functionspace(mesh, basix.ufl.mixed_element([Ve, Qe]))\n   Reaching the components: ufl.split(w) on a fem.Function, ufl.TrialFunctions(W) / ufl.TestFunctions(W) for the arguments, and w.split() on the Function. W.sub(i).collapse() returns a PAIR (space, dof indices), not a space -- unpack it -- and W.split() does not exist (AttributeError, measured). A velocity-pressure pair needs deg_v > deg_q to be stable; picking them is yours.\n14. FACET TAGS ARE A LOWERCASE FUNCTION IN THE MESH MODULE. `fem.MeshTags(...)` and `dolfinx.MeshTags` are both AttributeError (measured dolfinx 0.10): the constructor is dolfinx.mesh.meshtags(mesh, dim, indices, values) with indices and values as int32 arrays, and you tag facets with dim = mesh.topology.dim - 1. Pass it to a measure as ufl.Measure('ds', domain=mesh, subdomain_data=<tags>) and integrate one tag with ds(<value>). MEASURED AND NOT REQUIRED: the indices do NOT have to be sorted -- a shuffled index array gives the same integral to the last bit on this install, so do not spend a step sorting them. A connectivity the topology has not built raises RuntimeError: 'Connectivity between dimension 0 and 2 has not been computed', and that message names its own fix -- call mesh.topology.create_connectivity(<from>, <to>) once before the lookup.\n15. ASSEMBLING BY HAND (fem.petsc), MEASURED ON dolfinx 0.10: a UFL form goes through fem.form(...) first -- a bare one raises AttributeError: 'Form' object has no attribute '_cpp_object'. assemble_matrix(a_form, bcs=[bc]) zeroes the bc rows AND columns, so the right-hand side needs all three: fem.petsc.apply_lifting(b, [a_form], bcs=[[bc]]), b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE), fem.petsc.set_bc(b, [bc]). On a linear test: without them the boundary AND interior are wrong (4, 3.6), set_bc alone fixes the boundary only (0, 3.6), all three give 3e-15. A Function's PETSc vector is uh.x.petsc_vec (Function.vector is gone, and a PETSc Vec has no petsc_vec). A number times dx needs the domain: fem.Constant(mesh, value) * ufl.dx, or ufl.dx(domain=mesh); 1.0 * ufl.dx raises ValueError: This integral is missing an integration domain.",
    # Every line measured by execution on this install (dune-fem on
    # dune-py313) on 2026-09-03. repr-generated literal.
    "dune": "1. `ufl.Eq` NO LONGER EXISTS in this ufl (ImportError; five hits in one round). The lowercase `ufl.eq` does, and it is ONLY a conditional's test: ufl.conditional(ufl.eq(a, b), val_true, val_false). The equation handed to galerkin is written with Python's `==`: `scheme = galerkin([a == L, dbc], solver='cg')` -- passing `eq(a, L)` or a bare form dies with `ValueError: first argument should be a ufl equation (not only a form) or an 'integrands' model` (measured: one of three trial scripts read the `eq` line above as the equation builder).\n2. `DirichletBC` takes (functionSpace, value, subDomain=None) -- there is NO `marker` keyword (TypeError). Measured signature on this install.\n3. A VERTEX'S COORDINATES ARE vertex.geometry.center (or .corner(0)) -- there is no .geometry.point (AttributeError, measured; a run writing the 3-D side died on it). The dof array of a discrete function is u.as_numpy.\n4. A UFL Form HAS NO .copy() (AttributeError, measured). A form is immutable and cheap: build the second one by writing the expression again, e.g. keep the volume load as its own form rather than copying the full one.\n5. `scheme.solve(target=uh)` returns a DICT with keys converged, iterations, linear_iterations, timing -- read info['converged'], never info.converged (AttributeError on dict; one round hit it three times). Measured: a 4x4 Laplace solve returns converged=True with max|u| = 7.768e-02.\n6. Solver verbosity for a captured log: parameters={'linear.verbose': True} on galerkin(...) -- the old 'newton.linear.verbose' spelling is deprecated and warns.\n7. Create functions with a name -- space.interpolate(0.0, name='uh') -- because a plain UFL expression has no .name and downstream I/O that asks for one dies on AttributeError.\n8. FIRST USE COMPILES. 'DUNE-INFO: Compiling Integrands (new)' means dune-fem is JIT-compiling your UFL forms -- minutes on a loaded machine, and again for every new form. Do NOT wrap the run in a short `timeout` and do NOT pipe it into `head`: a PETSc 'Caught signal number 15 Terminate' printed after those lines means the process was killed from OUTSIDE (a timeout or a closed pipe), not that the solver crashed (measured: one run wrapped its participant in `timeout 120`, read the signal-15 message as 'DUNE crashes on every attempt' and gave up with a working install). Run each participant once standalone with a generous timeout so the compiled modules are cached; the coupling iterations then start in seconds.\n9. THE IMPORT LINES, EXACTLY -- COPY THIS BLOCK, DO NOT RECONSTRUCT IT. Every line below was executed on this install. Writing these by hand is the most expensive mistake made here: across 246 coupled runs on this machine, 141 of them raised an ImportError they had written themselves, 218 times in all, and each one costs a rewrite and a rerun.\n    import json\n    from pathlib import Path\n    import numpy as np\n    from dune.grid import structuredGrid, cartesianDomain\n    from dune.alugrid import aluConformGrid\n    from dune.fem import assemble, integrate\n    from dune.fem.space import lagrange\n    from dune.fem.scheme import galerkin\n    from dune.fem.function import uflFunction, gridFunction\n    from dune.ufl import DirichletBC, Constant\n    from ufl import (TrialFunction, TestFunction, SpatialCoordinate,\n                     FacetNormal, grad, inner, dot, div, dx, ds,\n                     conditional, lt, gt, le, ge, eq, And, Or,\n                     sin, cos, exp, sqrt, pi, as_vector)\n   THE THREE WRONG GUESSES THAT COST THE MOST, in order: `from ufl import abs` (71 times -- abs, min and max are NOT in ufl; Python's own built-ins work on a UFL expression), `from ufl import Eq` (45 -- only the lowercase `eq` exists, and the equation itself is written with Python's `==`), and `from dune.ufl import SpatialCoordinate` (39 -- it is in ufl; dune.ufl holds DirichletBC, Constant and cell, nothing else). Also measured absent: `dune.gdt` entirely, `aluConformGrid` in dune.grid (it is in dune.alugrid), `GridFunction` in dune.fem.function (the callables are `gridFunction` and `uflFunction`), and `SubDomain` in ufl. Of those two, `uflFunction(gridView, name=..., order=..., ufl=<expr>)` is DEPRECATED on this install and warns; the replacement takes its arguments in a different order -- `gridFunction(<expr>, gridView, name, order)`, expression FIRST (measured 2026-09-19).\n10. THE GRID CALL, EXACTLY: `gridView = structuredGrid([X0, Y0], [X1, Y1], [NX, NY])` -- lower corner, upper corner, cell counts, in that order (measured: a trial that passed the cell counts first died with `EquidistantOffsetCoordinates(...) Invoked with: array([6., 10.]), ...`).\n11. `abs`, `min`, `max` are NOT importable from ufl in this version (ImportError); the Python built-ins work on UFL expressions, and `ufl.conditional(ufl.lt(a, b), x, y)` is the branch.\n12. A grid entity has no `.index`: use `gridView.indexSet.index(entity)` (or `subIndex(entity, i, codim)` for its vertices); vertex coordinates come from `entity.geometry.center` / `corner(i)`, and the vertex-ordered nodal values of a Lagrange P1 function from `uh.as_numpy` (measured: `e.index` raises AttributeError on the generated Entity type).\n13. IN dune-fem's UFL THE SPACE CARRIES THE DOMAIN: `TrialFunction(space)`, `TestFunction(space)`, `SpatialCoordinate(space)`, and every integrand must contain one of them before `* dx` / `* ds`. Measured in three of three trial scripts: passing the grid gives `LeafGrid has no attribute ufl_domain`, a bare `dx` on a space-free expression gives `This integral is missing an integration domain`, and `space.domain` does not exist (`dimDomain` does).\n14. THE SPACE COUNTS ITS DOFS WITH .size (or len(space)) -- there is no space.dim and no space.dofCoordinates() (both AttributeErrors, measured; two runs writing the 3-D side died on exactly those two). Coordinates come from the grid, not the space: see the mesh access below.\n15. MESH ACCESS, EXACTLY (measured): `for v in gridView.vertices` / `for e in gridView.elements` iterate; `gridView.indexSet.index(v)` numbers a vertex and `gridView.indexSet.subIndex(e, i, 2)` numbers corner i of element e; `e.geometry.corners` is a TUPLE of the corner points (`len()` counts them) and `e.geometry.center` a point. There is no `gridView.entity(i)`, no `gridView.entitySet(...)`, no `gridView.corner(e, j)` and no `geometry.corner(i)` -- each was invented by a trial script and each raises AttributeError/TypeError. A SIMPLEX grid: `from dune.grid import cartesianDomain; from dune.alugrid import aluConformGrid; gridView = aluConformGrid(cartesianDomain([X0, Y0], [X1, Y1], [NX, NY]))` (6 x 10 -> 120 triangles, 77 vertices; `structuredGrid` makes 60 quadrilaterals). On this install a P1 Lagrange dof order equalled the vertex order on both grids, but never rely on it: map through interpolated coordinate fields.\n16. COEFFICIENTS GO IN AS `dune.ufl.Constant(value, name='k')`, never as a bare Python number: `0.0 * u * v * dx` (a zero reaction written as a float) folds to a domainless UFL Zero and dies with `This integral is missing an integration domain` (measured; the same fold hits a zero source).\n17. POWERS ARE `**`: `x[1]^2` is Python's XOR on a UFL expression and dies in as_tensor with `Expecting a tuple of Index objects` (measured).\n18. UFL CONDITIONS DO NOT COMBINE WITH `|` OR `&`: `lt(...) | lt(...)` dies with `unsupported operand type(s) for |: 'LT' and 'LT'` (measured). Build 0/1 indicators with `conditional(lt(abs(x[0] - X0), eps), 1, 0)`, ADD them for a union of edges, and take `1 - ind` for the complement (the outer boundary is `1 - <interface indicator>`); `ufl.Or(a, b)` / `ufl.And(a, b)` also exist. There is NO `SubDomain` in ufl (`from ufl import SubDomain` is an ImportError; two of three trial scripts reached for one) and none is needed: `DirichletBC(space, value, <that 0/1 UFL indicator>)` takes the indicator directly.",
    # Every line measured by execution on this install (FEBio 4.12,
    # febio4 binary) on 2026-09-04. repr-generated literal.
    "febio": '1. READING RESULTS BACK IS ONE DECK LINE, and a solve that is never read back is no result. Put inside <Output><logfile>:\n       <node_data data="x;y;z;ux;uy;uz" delim="," file="nodal_out.csv"/>\n   FEBio then writes one block PER TIME STEP, each headed *Step/*Time/*Data lines (measured: 51 blocks for 50 steps, header \'*Data  = x;y;z;ux;uy;uz\'); parse the LAST block for the final state and interpolate those nodal values at your probe points (scipy LinearNDInterpolator on the coordinate columns works). Runs that reached NORMAL TERMINATION and still delivered nothing all skipped this line.\n2. THE VISCOELASTIC WRAPPER FAMILY MUST MATCH THE NESTED ELASTIC\'S FAMILY (measured on FEBio 4.12): type="uncoupled viscoelastic" REFUSES a coupled child like isotropic elastic -- the error says \'Component ... needs to have property "elastic" defined\' even though <elastic> is present, because its FAMILY does not fit the slot. The coupled wrapper type="viscoelastic" accepts <elastic type="isotropic elastic"> (E, v) and runs to NORMAL TERMINATION. Uncoupled wrappers take uncoupled children (Mooney-Rivlin with k, etc.).\n3. \'negative jacobians detected\' during a solve is usually NOT the mesh: a hex8 grid whose first element has positive centroid jacobian can still invert under too-large load steps or a too-stiff/soft material pairing. Before rebuilding the mesh, halve the step (<time_steps> up, <step_size> down) and re-check the material family pairing of fact 2.\n4. FEBio prints its banner and \'N O R M A L   T E R M I N A T I O N\' letter-spaced -- grep for \'N O R M A L\', not \'NORMAL\'.',
    # Every line measured by execution on this install (NGSolve 6.2.2604)
    # on 2026-09-04. repr-generated literal.
    "ngsolve": "1. NEVER EVALUATE A COMPOUND-SPACE GridFunction DIRECTLY. On a product space (H1*H1, mixed), gfu(mesh(x,y)) either raises 'CompoundFESpace does not have an evaluator for VOL!' or -- measured, worse -- silently returns 0.0 while the field is nonzero. Evaluate the component: gfu.components[i](mesh(x,y)) (measured 0.2524 at the same point where the direct call returned 0.0).\n2. FORMS DO NOT SUPPORT -= (TypeError: unsupported operand). Subtract by adding the negated term at definition: a += (-1) * u * v * dx. Same for LinearForm.\n3. grad()/Grad() WORKS ON PROXIES AND GridFunctions, NOT ON ASSEMBLED CoefficientFunctions -- 'Operator grad not overloaded for CF ngfem::VectorialCoefficientFunction' (measured). Take grad(gfu.components[i]) and assemble what you need from those, or differentiate the symbolic expression BEFORE wrapping it in CoefficientFunction.\n4. Verbosity for a captured log: ngsolve.ngsglobals.msg_level = 3, or solvers.CG(..., printrates=True); NGSolve is otherwise quiet on success.\n5. THE 2-D GEOMETRY IS BUILT WITH netgen's SplineGeometry, AND IT HAS NEITHER AddVertex NOR AddRect (AttributeError, measured). A rectangle is one call: geo.AddRectangle((X0, Y0), (X1, Y1), bcs=(<bottom>, <right>, <top>, <left>)) -- FOUR BOUNDARY NAMES OF YOUR CHOOSING, IN THAT EDGE ORDER. ds('<name>') and H1(..., dirichlet='<name>') select by exactly those names, and a ds() over a name the mesh does not carry integrates over NOTHING with no error; several names are joined with |, as one regular expression, dirichlet='<a>|<b>': joined with commas they match no boundary and hold no dof, with no error (measured: 0 of 79 dofs held, against 24 with |); mesh with Mesh(geo.GenerateMesh(maxh=h)) and read the names back with mesh.GetBoundaries(). For a shape that is not a rectangle use AddPoint/AppendPoint plus Append/AddSegment -- and note both point calls take SEPARATE coordinates, AddPoint(x, y): handing them a tuple raises TypeError: AppendPoint(): incompatible function arguments (measured). Vertex coordinates come back as np.array([vert.point for vert in mesh.vertices]) -- a vertex of its own name: v is the test function.\n6. THE DOF OF A VERTEX COMES FROM THE SPACE, NOT FROM THE VERTEX. A MeshNode carries .nr (its number) and .point (its coordinates) and NOTHING ELSE you want here -- it has no .ndof and no .index (both AttributeErrors, measured), and the space has no fes.Dofs() either. So the two calls you need are vert.point for a vertex's coordinates and fes.GetDofNrs(NodeId(VERTEX, vert.nr))[0] for its dof; WHICH vertices are yours to choose. (For H1(order=1) that dof map is the identity on this install, measured over 87 vertices, but read it with GetDofNrs rather than assuming it for a higher order.) Mesh iterators are LOWERCASE properties: mesh.vertices, mesh.faces, mesh.edges -- mesh.Faces() and mesh.Vertices() are both AttributeErrors -- and a vertex's coordinates are vert.point, not vert.x. Boundary names come back from mesh.GetBoundaries(); mesh.Boundaries('interface') is a Region and a Region is NOT ITERABLE (TypeError, measured): iterate mesh.Boundaries('interface').Elements(), or take its .Mask() BitArray. A BitArray counts its set bits with NumSet() and has no Count (AttributeError, measured).\n7. A LinearForm MUST NOT CONTAIN THE TRIAL FUNCTION. Putting u * v * dx into one stops netgen with NgException: In MakeLinearFormIntegrator: must not have TrialFunction (measured) -- every term carrying u belongs in the BilinearForm, and the LinearForm carries only v.\n8. THE VECTORS ARE BaseVector, NOT ARRAYS AND NOT FORMS. A LinearForm's vector is f.vec and a BaseVector has no .vec of its own (AttributeError, measured -- a worker wrote f_vol.vec.vec). You cannot index one with a BitArray either (TypeError: __getitem__(): incompatible function arguments): take numbers out with vec.FV().NumPy() or np.array(vec). Assign THROUGH .data, never by rebinding: `vec.data = <expression>`, with a fresh vector from vec.CreateVector(). It is lowercase: reading vec.Data is an AttributeError, and vec.Data = ... raises nothing and leaves the vector as it was (measured). Matrix-vector products read a.mat * vec, and a constrained inverse is a.mat.Inverse(<free dofs>, inverse='sparsecholesky'). All of these shapes measured on this install; what you assemble into a and f, and how you constrain the solve, are yours.\n9. `inverse` IS A KEYWORD OF Inverse, NOT AN IMPORT. The solve is a.mat.Inverse(fes.FreeDofs(), inverse='sparsecholesky'); adding `inverse` to the `from ngsolve import (...)` list raises ImportError: cannot import name 'inverse' from 'ngsolve' (measured -- a worker edited it into the served import line).\n10. A PYTHON FUNCTION IS NOT A CoefficientFunction. CoefficientFunction(f) for a def/lambda is a TypeError ('incompatible constructor arguments', measured), and calling a NumPy-written source on ngsolve's symbolic x, y fails or -- np.zeros_like(x) -- silently returns a 0-d object array. Sample it at the mesh vertices instead (np.array([vert.point for vert in mesh.vertices])) into a P1 GridFunction on your space (gf.vec.FV().NumPy()[vertex_dofs] = f(vx, vy)); a GridFunction IS a CoefficientFunction and integrates as gf * v * dx, v the test function (the P1 interpolant of the source, O(h^2) like the discretisation itself).",
    # deal.II facts measured by execution on the coupled elasticity
    # walk of 2026-09-07 (deal.II 9.8.0-pre, ~/dealii/build). Re-measured and
    # extended 2026-09-29 on the same Release-only build: every right call below
    # compiled and ran in one probe program, and every wrong one was compiled on
    # its own, its first compiler error being the message quoted. They are the
    # calls the compile logs of coupled deal.II sides died on most often. Two
    # causes this entry used to give were NOT reproduced and are gone: swapping
    # the FEValues arguments gives "no matching function for call to
    # FEValues<2, 2>::FEValues(QGauss<2>, FE_Q<2>&, UpdateFlags)", not the
    # "unsubscribe" error (PreconditionSSOR<double> gives that, fact 6), and
    # FEEvaluation built like FEValues gives "no matching function" too.
    "dealii": (
        '1. deal.II prints NOTHING by default: a run whose log must carry the code own output needs BOTH deallog.depth_console(2); AND a SolverControl ctl(max_it, tol, true, true); (log_history, log_result) -- depth_console alone prints nothing. Then the console carries DEAL:cg lines per iteration. A direct solve prints none; the served program therefore opens its console with the library\'s own line, its version and git revision from its headers through deallog (DEAL::deal.II <version>, git revision <sha>).\n'
        '2. Print the DOF count yourself, on whatever line your task asks for: std::cout << "DOF count = " << dof_handler.n_dofs() << std::endl; -- nothing else emits it.\n'
        '3. FEValues AND THE FLAGS ITS LOOP READS. FEValues<2>(mapping, fe, quadrature, flags) evaluates the shape functions on one cell after its reinit(cell), and FEFaceValues<2> does the same on one face (fact 9). What a loop may read is what the flags asked for: shape_value(i, q) needs update_values, shape_grad(i, q) (a Tensor<1, 2>) update_gradients, quadrature_point(q) update_quadrature_points, JxW(q) update_JxW_values. A finite element vector\'s values at the quadrature points of the cell last given to reinit are get_function_values(vector, values), values a std::vector<double> of n_quadrature_points entries (it needs update_values), and its gradients are get_function_gradients(vector, gradients), gradients a std::vector<Tensor<1, 2>> (it needs update_gradients). THIS deal.II IS A RELEASE BUILD AND ASSERTS NOTHING: quadrature_point(q) without update_quadrature_points (or shape_value without update_values) exits with code 139, a segfault, and no message. To make it speak, add target_compile_definitions(<name> PRIVATE DEBUG) after deal_ii_setup_target and rebuild: the same run aborts with "An error occurred in line <2396> of file <.../fe_values_base.h>" and "requires the <update_quadrature_points> flag". A get_dof_indices vector left unsized segfaults with and without DEBUG: size it fe.n_dofs_per_cell(). FEEvaluation is the matrix-free class; built like FEValues it is "no matching function for call to".\n'
        '4. Evaluate the solution at arbitrary (off-node) points with VectorTools::point_value(dof_handler, solution, Point<2>(x, y)) -- nearest-vertex lookup is the export defect that turns a converged solve into a wrong answer.\n'
        '5. BUILD IT WITH CMAKE, AND THESE SIX LINES ARE THE WHOLE CMakeLists (re-measured 2026-09-14 on this install: configures and builds first try, ~20 s):\n'
        '       cmake_minimum_required(VERSION 3.13)\n'
        '       find_package(deal.II 9.0 REQUIRED HINTS ${DEAL_II_DIR})\n'
        '       deal_ii_initialize_cached_variables()\n'
        '       project(<name>)\n'
        '       add_executable(<name> <name>.cc)\n'
        '       deal_ii_setup_target(<name>)\n'
        '   then `cmake -S <dir> -B build -DDEAL_II_DIR=<the tree discover names for deal.II> -DCMAKE_BUILD_TYPE=Release && make -C build`. A COMPILE THAT READS /usr/include/deal.II IS BUILDING AGAINST THE SYSTEM PACKAGE (9.1.1 on the install this was measured on), whatever cmake printed: without HINTS, find_package ignores -DDEAL_II_DIR and says "Using the deal.II-9.1.1 installation found at /usr"; and target_link_libraries(<name> dealii::dealii) in place of deal_ii_setup_target, with no CMAKE_BUILD_TYPE, prints "Using the deal.II-9.8.0-pre build directory" and then fails in "/usr/include/deal.II/base/config.h:439: fatal error: mpi.h". static_assert(DEAL_II_VERSION_GTE(9, 5, 0), "...") after #include <deal.II/base/config.h> stops that build at its first line.\n'
        '6. READ A TENSOR WITH BRACKETS, MULTIPLY WITH vmult, AND GIVE THE SOLVERS THEIR TYPES. g[0] and K[i][j]: g(0) is "no match for call to \'(dealii::Tensor<1, 2>) (int)\'", and K = {{a, b}, {c, d}} "could not convert"; fill K[i][j] one entry at a time, or Tensor<2, 2> K(a) from const double a[2][2]. A.vmult(dst, src) multiplies a SparseMatrix: Vector has no vmult, and A * u is "no match for \'operator*\'". SolverCG takes the VECTOR type as its template argument, SolverCG<Vector<double>>, and is built from a SolverControl. PreconditionSSOR takes the MATRIX type, PreconditionSSOR<SparseMatrix<double>>, is built empty, and its initialize(matrix, relaxation) sets it up. SolverCG::solve takes (matrix, x, b, preconditioner). PreconditionSSOR<double> ends in "request for member \'unsubscribe\'" (its first error is "static assertion failed: This class can only be used if the first template argument is a class derived from \'EnableObserverPointer\'"), SolverCG<double> in "\'double\' is not a class, struct, or union type", ssor(A, 1.2) in "no matching function". The iteration count is control.last_step().\n'
        '7. AN OVERRIDE OF Function<2>::value MUST RESTATE THE DEFAULT: double value(const Point<2> &p, const unsigned int component = 0) const override. Without "= 0", f.value(p) is "no matching function for call to". A lambda becomes a Function as ScalarFunctionFromFunctionObject<2>([&](const Point<2> &p) { return ...; }). Catch with catch (std::exception &exc): ExceptionHandler "does not name a type".\n'
        '8. MESH, BOUNDARY IDS, DOFS. GridGenerator::subdivided_hyper_rectangle(tria, std::vector<unsigned int>{nx, ny}, Point<2>(x0, y0), Point<2>(x1, y1), colorize); nx and ny as two arguments is "no matching function". colorize = true (here and in hyper_rectangle) gives id 0 on x = x0, 1 on x = x1, 2 on y = y0, 3 on y = y1; false leaves every face at 0. for (const auto &face : cell->face_iterators()) if (face->at_boundary()) with face->boundary_id(), face->set_boundary_id(id), face->center() (boundary_indicator is not a member). A dof\'s position: DoFTools::map_dofs_to_support_points(mapping, dof_handler, points), the mapping required and points a std::vector<Point<2>> of n_dofs entries; a std::map keyed by Point<2> does not compile ("no match for \'operator<\'"). cell->vertex_index(v) is a VERTEX number, not a dof (they differed on 57 of 64 cell corners measured): the dof is cell->vertex_dof_index(v, 0). fe_values.dof_indices() runs 0..dofs_per_cell-1, LOCAL numbers: the global ones are cell->get_dof_indices(indices).\n'
        '9. FACE INTEGRALS. FEFaceValues<2>(mapping, fe, face quadrature, flags), the face quadrature a QGauss<1>, takes the flags its loop reads (update_values, update_quadrature_points, update_normal_vectors, update_JxW_values): without the flags argument it is "no matching function". Its reinit takes the cell and a face number, reinit(cell, face_no), and a cell\'s face numbers are cell->face_indices(), while cell->face(face_no)->at_boundary() and ->boundary_id() say where that face is. shape_value(i, q), quadrature_point(q), normal_vector(q) and JxW(q) all take (q): quadrature_points and normal_vectors are not members, and fv[0] is "ambiguous overload for \'operator[]\'". A DoFHandler<2> is built on the triangulation and then given distribute_dofs(fe) (it has no initialize).\n'
        '10. MatrixTools::apply_boundary_values(boundary_values, matrix, solution, right_hand_side) EMPTIES THE HELD ROWS of the matrix it is given down to the diagonal (measured: 2 off-diagonal entries in a held row before, 0 after), and AffineConstraints::distribute_local_to_global does the same to constrained rows. A consistent flux recovery r = A u - b reads exactly those rows, so keep A as assembled and hand such a call a copy: a SparseMatrix<double> built on the same SparsityPattern and filled by copy_from(A). A SparseMatrix copy-constructed from A does not copy it: deal.II leaves it EMPTY, and a library call that uses it (apply_boundary_values, condense, a direct solver\'s initialize) segfaults with no message, with DEBUG on and off (DEBUG in a program turns on the checks in deal.II\'s headers, not the ones compiled into its library, measured). SparseDirectUMFPACK (lac/sparse_direct.h) is a direct solver: its initialize(matrix) factorizes, and its vmult(x, b) writes the solution of matrix x = b into x. AffineConstraints<double> (ConstraintMatrix is gone): add_line(i) then set_inhomogeneity(i, v), or add_constraint(i, {}, v); VectorTools::interpolate_boundary_values(dof_handler, id, function, constraints); close() before use; after the solve, constraints.distribute(u), which takes the solution vector: handed a SparseMatrix it compiles and fails to LINK (undefined reference to ... distribute<dealii::SparseMatrix<double> >).\n'
        '11. HEADERS THAT DO NOT EXIST in this deal.II ("No such file or directory"): deal.II/base/deallog.h (deallog is in base/logstream.h), deal.II/lac/precondition_relaxation.h (lac/precondition.h), deal.II/base/affine_constraints.h (lac/affine_constraints.h), deal.II/numerics/transpose_copy.h (none), deal.II/fe/fe_face_values.h (FEFaceValues is in fe/fe_values.h), deal.II/lac/sparse_direct_umfpack.h (lac/sparse_direct.h), deal.II/numerics/affine_constraints.h (lac/affine_constraints.h), deal.II/base/vector_tools.h and deal.II/lac/vector_tools.h (numerics/vector_tools.h), deal.II/grid/boundary_descriptor.h (none). DynamicSparsityPattern needs lac/dynamic_sparsity_pattern.h, QGauss base/quadrature_lib.h, MappingQ fe/mapping_q.h: without them, "incomplete type" or "was not declared".\n'
        '12. A TWO-COMPONENT FIELD (a displacement) is FESystem<2> fe(FE_Q<2>(1), 2) (fe/fe_system.h). fe.n_dofs_per_cell() counts BOTH components, 8 on a degree-1 cell, and the component of local dof i is fe.system_to_component_index(i).first: a cell matrix is n_dofs_per_cell square, and an index i*2+d beyond it is written past the matrix without a word in this Release build (the heap is damaged and the run dies later, "malloc(): corrupted top size" or the like; with DEBUG on it stops at the write, "Index 8 is not in the half-open range [0,8)"). A class member FESystem is built in the constructor\'s initializer list, fe(FE_Q<2>(1), 2): left to its default constructor, or assigned afterwards, it is "use of deleted function". FEValuesExtractors::Vector disp(0) reads both components of shape function i: fe_values[disp].value(i, q) is a Tensor<1, 2>, fe_values[disp].symmetric_gradient(i, q) a SymmetricTensor<2, 2> (base/symmetric_tensor.h), fe_values[disp].divergence(i, q) a double, and scalar_product(s, e) contracts two SymmetricTensor<2, 2>; FEValues and FEFaceValues themselves have no value or gradient member ("has no member named"). A Function<2> of two components hands 2 to its base, Function<2>(2), and overrides value(p, component); interpolate_boundary_values(mapping, dof_handler, id, function, boundary_values) then holds both components, and fe.component_mask(FEValuesExtractors::Scalar(c)) as one more argument holds component c alone. A ComponentMask is ComponentMask(2, true) or that component_mask, and an entry changes through set(c, value): ComponentMask(2) is "no matching function", and mask[c] = true does not compile.'),
    # SPARTA facts measured by execution on this install (SPARTA 24 Sep 2025,
    # {SPARTA_BINARY}) on 2026-09-14, with
    # the two deck traps re-verified against the installed source. SPARTA also
    # had no deciding facts while it is one side of a coupled development
    # problem; a DSMC side is a binary driven by a text deck, so its traps are
    # deck traps and they stop the run rather than bend the answer.
    "sparta": "1. IT IS A TEXT-DECK BINARY: `spa_serial -in <deck>` run in the deck's directory. It prints its own banner (`SPARTA (24 Sep 2025)`), the grid/particle setup and the `Step CPU Np` stats table to STDOUT, and writes the same to `log.sparta` in the working directory. `-log none` suppresses the FILE, not the console; `stats N` sets how often the table prints (measured).\n"
              "2. A `fix ave/surf` WITH ONE VALUE IS A PER-SURF VECTOR, so the dump must reference it as `f_1`, NOT `f_1[1]`. The indexed form stops the run with `Dump surf fix does not compute per-surf array` (src/dump_surf.cpp, the `argindex[i] > 0 && size_per_surf_cols == 0` branch, read on this install).\n"
              "3. NO surf_collide STYLE TAKES A PRESCRIBED HEAT FLUX. The nine installed styles are adiabatic, cll, diffuse, impulsive, piston, specular, td, transparent and vanish; all four thermal ones prescribe a WALL TEMPERATURE. A flux can only be imposed indirectly, through `fix surf/temp`, which turns a per-surf energy flux into the wall temperature it implies -- and SPARTA's own doc for it states that it `does not check that the specified compute/fix calculates an energy flux`, so it will accept any per-surf quantity you hand it (doc/fix_surf_temp.html on this install).\n"
              "4. couple() COPIES THE FILES A PARTICIPANT LISTS IN data_files (absolute paths) into its work_dir before the first iteration, and stops before any iteration when one is missing; it does not read the deck for file names (measured on this install: a deck naming a file the participant did not list was left without it). SPARTA opens every file a deck names in the directory it runs in, so a standalone run needs the files there: the served contract copies its SURF_FILE, SPECIES and VSS from DATA_DIR, or from the SPARTA distribution's data directory, when they are missing.\n"
              "5. THE OUTPUT IS A MONTE-CARLO ESTIMATE, BUT IT IS REPRODUCIBLE: with the same `seed` and the same deck, repeat runs give the identical particle history (measured: four runs, identical Np at every stats step). A fixed-point iteration against a DSMC side therefore converges on the sampling noise you chose, not on noise that moves under you -- lengthen the averaging window (`fix ave/surf` over more steps) rather than tightening the coupling tolerance.",
    # scikit-fem facts measured by execution on this install (skfem 12.0.1,
    # {VENV}) on 2026-09-14, after
    # two of three runs died on lines 1 and 2 below. This was the
    # only backend in the corpus with NO deciding facts at all, and three
    # coupled development problems put scikit-fem on one side.
    "skfem": "1. A FORM TAKES ITS BASIS POSITIONALLY. `laplace.assemble(basis)` and `asm(laplace, basis)` both work; `laplace.assemble(basis=basis)` raises TypeError: BilinearForm._assemble() missing 1 required positional argument: 'ubasis' (measured).\n"
             "2. `basis.dofs` IS AN ATTRIBUTE, NOT A METHOD -- `basis.dofs(facet_indices=...)` raises TypeError: 'Dofs' object is not callable. Select dofs with `basis.get_dofs(facets=<facet indices>)` or `basis.get_dofs(lambda x: np.isclose(x[0], 0.6))`, then `.flatten()` for the index array (or `.nodal['u']`).\n"
             "3. `mesh.facets_satisfying(pred)` RETURNS BOUNDARY FACETS ONLY. An interface that runs THROUGH one mesh needs `mesh.facets_satisfying(pred, boundaries_only=False)`; without it the selection is silently empty and the condition is applied nowhere.\n"
             "4. EVALUATE AT OFF-NODE POINTS with `basis.probes(pts) @ x` -- `pts` is (dim, n_points) and `probes` returns the interpolation OPERATOR of shape (n_points, N), not values -- or `basis.interpolator(x)(pts)`. Both measured identical; nearest-vertex lookup is the export defect that turns a converged solve into a wrong answer.\n"
             "5. FOR P1 THE DOF ORDER IS THE MESH VERTEX ORDER (basis.doflocs == mesh.p, measured), so a nodal array can be indexed by vertex number.\n"
             "6. Solve with `solve(*condense(A, b, D=dirichlet_dofs))`, or with `x=` to impose non-zero values: `solve(*condense(A, b, x=x0, D=D))`. `enforce(A, b, D=D)` returns the modified pair instead. KEEP THE STAR: `condense(A, b, x=x0, D=D)` returns FOUR items (A_I, b_I, x, I), and `solve(A_I, b_I)` returns only the REDUCED vector (measured 49 entries on an 81-dof mesh) while the full `x` stays as x0 -- all zeros if x0 was zeros, with no error. If you unpack, write it back: `x[I] = solve(A_I, b_I)`.\n"
             "7. Verbosity for a captured log: logging.basicConfig(level=logging.INFO) makes skfem print \"Assembling 'laplace'.\" -- TO STDERR. Redirect with 2>&1 or the run log carries none of it.\n"
             "8. INSIDE A FORM THE COORDINATES ARE w.x[0] AND w.x[1]. There is no w.y -- AttributeError: Attribute 'y' not found in 'w' (measured; three runs died on it). `w.x` is the (dim, nelems, nqp) array of global quadrature-point coordinates, so a source written with it is evaluated where the rule needs it.\n"
             "9. A RECTANGLE IS MeshTri.init_tensor(np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1)). There is no MeshTri.init_rect (AttributeError: type object 'MeshTri1' has no attribute 'init_rect', measured).\n"
             "10. Basis(mesh, element) takes NO doforder keyword (TypeError: CellBasis.__init__() got an unexpected keyword argument 'doforder'), and a FacetBasis has NO find_dofs (AttributeError) -- it has get_dofs, like every other basis.\n"
             "11. THE MESH ARRAYS ARE p (2 x n_nodes), t (3 x n_elements) and facets (2 x n_facets), with t2f and f2t joining them; there is no mesh.f (AttributeError: 'MeshTri1' object has no attribute 'f', measured). The node indices of facet k are mesh.facets[:, k] and their coordinates mesh.p[:, mesh.facets[:, k]]; whole-boundary sets are mesh.boundary_facets() and mesh.boundary_nodes().\n"
             "12. A VECTOR BASIS SELECTS A COMPONENT BY NAME, NOT BY A component= KEYWORD. `get_dofs(..., component=0)` raises TypeError: AbstractBasis.get_dofs() got an unexpected keyword argument 'component' (measured, and a coupled run lost its budget to it twice). get_dofs returns a DofsView whose components are ONE-BASED STRINGS: d = basis.get_dofs(<facets or predicate>), then d.nodal['u^1'] and d.nodal['u^2'] are the x and y dof arrays, d.all('u^1') is the same thing, and d.all() / d.flatten() give both interleaved. There is no u^0.\n"
             "13. REFINE WITH mesh.refined(), NOT mesh.extend(). `'MeshTri1' object has no attribute 'extend'` is measured; mesh.refined() halves h once and mesh.refined(n) does it n times, each returning a NEW mesh (the meshes are immutable, so the return value is the refinement -- ignoring it leaves you on the coarse mesh).\n"
             "14. `FacetBasis(mesh, element)` WITHOUT `facets=` IS THE OUTER BOUNDARY ONLY (nelems == number of boundary facets, measured). Integrating over an interface that runs THROUGH the mesh with it gives zero weight at every interface node except where the line meets the boundary (measured: 2 of 9 nodes), so a flux recovered by dividing by those weights is silently zero. Pass the same facets you selected: `FacetBasis(mesh, element, facets=mesh.facets_satisfying(pred, boundaries_only=False))` (measured: 9 of 9).",
    "kratos": (
        "1. `LaplacianElement2D3N` exists; `LaplacianElement2D4N` does NOT "
        "('is not registered'). 2D is P1 TRIANGLES.\n"
        "2. CONDUCTIVITY, DENSITY, SPECIFIC_HEAT and HEAT_FLUX are read "
        "NODALLY through ConvectionDiffusionSettings, NOT from Properties. "
        "Setting conductivity only on Properties gives a singular system "
        "(measured: nodal 999 + Properties 1 -> 999). The same route works in 3-D with "
        "LaplacianElement3D4N (measured 2026-09-14). 'ATTENTION! setting the RHS to zero!' means "
        "the assembled right-hand side is exactly zero: Kratos skipped the solve and the field "
        "keeps its start values. Measured: all-zero data prints it (no source on the nodes, every "
        "held value zero); under a nonzero source, so does a hex split with flat tetrahedra (NaN "
        "fluxes) or with mixed signs (a finite, wrong field). Test the mesh first (check_tets(mp) "
        "in the served 3-D contract), then the source: HEAT_FLUX in AddNodalSolutionStepVariable "
        "BEFORE the mesh is built, set per node.\n"
        "3. The reaction-carrying DOF overload is `AddDof(var, reaction, mp)`; "
        "there is no AddDofWithReaction. Omitting the reaction and then asking "
        "the strategy for reactions aborts in EVERY worker thread.\n"
        "4. SETTING FACE_HEAT_FLUX ON NODES DOES NOTHING WITHOUT A CONDITION "
        "on those edges. The nodal value is only integrated BY a condition. "
        "This is the line, and it is the whole fix:\n"
        "       for c in range(len(iface) - 1):\n"
        "           mp.CreateNewCondition(\"ThermalFace2D2N\", c + 1,\n"
        "                                 [iface[c] + 1, iface[c+1] + 1], "
        "prop)\n"
        "   Measured on one mesh: flux on nodes with no condition gives the "
        "zero-flux field BIT-IDENTICALLY; with the conditions the flux is "
        "applied. FluxCondition2D2N works too. Two real "
        "runs died here, reporting the no-flux answer with their "
        "interface field matching to 0.000e+00, and a third quoted this very "
        "paragraph back in its give-up note without ever adding the line.\n"
        "4b. THE NAME GOES IN A STRING, THROUGH THE FACTORY. Elements and "
        "conditions live in a C++ registry and NONE is a Python attribute, so "
        "`SomeApplication.ThermalFace2D2N(...)` raises `has no attribute` for "
        "every name that exists. Measured on this install: "
        "LaplacianElement2D3N, ThermalFace2D2N and FluxCondition2D2N are all "
        "python-attribute=False and factory-by-name=True. That AttributeError "
        "is never evidence a component is missing -- one run read it as \"not "
        "available in version 10.3.0\" and went looking for a different code. "
        "A name that genuinely is absent fails differently, with `The "
        "Condition \"X\" is not registered!` from CreateNewCondition itself.\n"
        "5. FACE_HEAT_FLUX is the INWARD normal flux, while a task's q_n is "
        "OUTWARD -- so the number you REPORT is the negative of the one you "
        "APPLY.\n"
        "6. The volumetric HEAT_FLUX is the NODAL value at one centroid point "
        "(mean(f)*A/3); the exact mass matrix is 50% off.\n"
        "7. Kratos prints from C++ streams: an in-process stdout redirect "
        "captures ZERO bytes. Run the solve in a SUBPROCESS if the log has to "
        "show which code ran.\n"
        "8. THE MODEL PART IDIOM, EXACTLY (measured): `model = KM.Model(); mp = "
        "model.CreateModelPart('thermal')`, then `mp.AddNodalSolutionStepVariable(...)`, "
        "`mp.CreateNewNode(id, x, y, 0.0)` and `mp.CreateNewElement('LaplacianElement2D3N', "
        "id, [n1, n2, n3], mp.GetProperties()[1])`. There is no `KM.MainModelPart` and no "
        "`KM.ModelPart(...)` constructor to call yourself (`Module KratosMultiphysics has no "
        "attribute MainModelPart` -- a trial script died there twice, before and after a repair).\n"
        "9. THE SOLVE STACK LIVES IN THE CORE NAMESPACE, EXACTLY (measured): "
        "`KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)` (there is no `KM.AddDof`); "
        "`scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()`; "
        "`builder = KM.ResidualBasedBlockBuilderAndSolver(KM.SkylineLUFactorizationSolver())`; "
        "`strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder, True, False, False, False)`; "
        "`strategy.Initialize(); strategy.Solve()`. None of these is in StructuralMechanicsApplication "
        "(`cannot import name ResidualBasedLinearStrategy from ...StructuralMechanicsApplication` -- a "
        "trial script died there; its repair died on `KM.AddDof`).\n"
        "10. DIRICHLET VALUES ARE NODAL, NOT A CONDITION (measured): "
        "`n.SetSolutionStepValue(KM.TEMPERATURE, value); n.Fix(KM.TEMPERATURE)` on every outer-boundary "
        "node; there is no Dirichlet condition to create (`The Condition \"ThermalDirichlet2D2N\" is not "
        "registered!` -- a trial repair died there). Conditions are for FLUXES (ThermalFace2D2N / "
        "FluxCondition2D2N on the interface edges, fact 4) and for nothing else here.\n"
        "11. THE MODEL SETUP ORDER, EXACTLY (measured on the tested program): `mp.ProcessInfo[KM.DOMAIN_SIZE] = 2`; "
        "`settings = KM.ConvectionDiffusionSettings()` with `SetUnknownVariable(KM.TEMPERATURE)`, "
        "`SetDiffusionVariable(KM.CONDUCTIVITY)`, `SetVolumeSourceVariable(KM.HEAT_FLUX)`, "
        "`SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)` and `mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, settings)`; "
        "then `mp.AddNodalSolutionStepVariable(v)` for EACH of TEMPERATURE, CONDUCTIVITY, HEAT_FLUX, FACE_HEAT_FLUX, "
        "REACTION_FLUX BEFORE the first `CreateNewNode` (a node created earlier cannot hold them: `This container only "
        "can store the variables specified in its variables list ... CONDUCTIVITY` -- a trial repair died there); "
        "`mp.SetBufferSize(1)`; `props = mp.CreateNewProperties(1)`. The application is loaded by "
        "`import KratosMultiphysics.ConvectionDiffusionApplication` and referenced only through that module path -- "
        "a bare `ConvectionDiffusionApplication` is a NameError (measured). A variable list WITHOUT "
        "CONDUCTIVITY does not fail at setup: the solve SEGFAULTS with no message (exit 139, measured on a "
        "trial repair that listed the other four). `Properties` has no `AddProperty`/`Value`; nodal values "
        "carry the material (fact 2)."
        # Fact 12 measured 2026-09-11 on a coupled run (Kratos 10.x, ConvectionDiffusionApplication).
        "\n12. THE FLUX CONDITION READS ITS LOAD FROM THE NODES. `FluxCondition2D2N` integrates the NODAL "
        "value of the surface-source variable (FACE_HEAT_FLUX): `mp.CreateNewCondition('FluxCondition2D2N', "
        "id, [n1, n2], props)` WITHOUT `n.SetSolutionStepValue(KM.FACE_HEAT_FLUX, q)` on both of its nodes "
        "assembles exactly zero and the run exits 0 (measured: a Neumann side built that way exported the "
        "no-load answer at every iteration and couple() reported it UNRESPONSIVE; the served Neumann contract "
        "sets the nodal value first). The condition takes no flux property of its own."),
}
_DECIDING_FACTS["dune-fem"] = _DECIDING_FACTS["dune"]
_DECIDING_FACTS["dunefem"] = _DECIDING_FACTS["dune"]

_DECIDING_FACTS["fenicsx"] = _DECIDING_FACTS["fenics"]
_DECIDING_FACTS["dolfinx"] = _DECIDING_FACTS["fenics"]
_DECIDING_FACTS["scikit-fem"] = _DECIDING_FACTS["skfem"]
_DECIDING_FACTS["scikitfem"] = _DECIDING_FACTS["skfem"]
_DECIDING_FACTS["deal.ii"] = _DECIDING_FACTS["dealii"]
_DECIDING_FACTS["4c"] = _DECIDING_FACTS["fourc"]

# THE FACTS A FLUID-STRUCTURE SIDE NEEDS, served by the fluid-structure door (physics='fsi') after the
# code's own. Measured on a fluid-structure round: every FEniCSx fluid side spent 14-22 minutes in
# run-fix loops on dolfinx calls none of the served facts named -- tabulating a mixed space (5 crashes),
# a zero right-hand side (8), ufl.Constant given a value (7), NonlinearProblem (about 10), a mesh moved
# through dolfinx.mesh.move (2). They ride here, not in the code's general block, because that block is
# served with every reply of the code and each character added there pushes the reply's own prose past
# its cap (measured: 3k added there cut the per-side naming rule from the FEniCSx door's first reply).
# Each fact was run in that code's own interpreter on this install.
_FSI_FACTS = {
    "fenics": (
        "\nFOR A FLUID SIDE (measured on this install, dolfinx 0.10):"
        "\n16. A BOUNDARY CONDITION ON ONE BLOCK OF A MIXED SPACE takes a Function AND the sub-space (measured on "
        "a Taylor-Hood velocity block, where it held a parabolic profile to round-off): V0, _ = W.sub(0).collapse(); "
        "g = fem.Function(V0) (left zero for no-slip, else interpolated); dofs = "
        "fem.locate_dofs_topological((W.sub(0), V0), fdim, facets) -- a pair of arrays; bc = fem.dirichletbc(g, "
        "dofs, W.sub(0)). A constant there raises 'Constant size is not equal to the block size', even a zero; "
        "dofs located on V0 alone raise incompatible function arguments; and "
        "fem.locate_dofs_topological(W.sub(0).collapse(), ...) hands the (space, dofs) pair over as a space and "
        "raises 'numpy.ndarray' object has no attribute '_cpp_object'."
        "\n17. A MIXED SPACE CANNOT TABULATE: W.tabulate_dof_coordinates() raises RuntimeError: Cannot tabulate "
        "coordinates for a mixed FunctionSpace. Tabulate the collapsed space instead, V0.tabulate_dof_coordinates(), "
        "one row per node of V0."
        "\n18. A ZERO RIGHT-HAND SIDE IS BUILT FROM fem.Constant: 0 * v[0] * ufl.dx (0.0 times any expression) "
        "folds to UFL's Zero and raises ValueError: This integral is missing an integration domain at once; "
        "ufl.inner(fem.Constant(mesh, np.zeros(gdim)), v) * ufl.dx + fem.Constant(mesh, "
        "default_scalar_type(0.0)) * q * ufl.dx compiles."
        "\n19. A NONLINEAR SOLVE IS dolfinx.fem.petsc.NonlinearProblem(F, w, bcs=bcs, "
        "petsc_options_prefix='<name>', petsc_options={'snes_type': 'newtonls', ...}); problem.solve() fills w. "
        "Without petsc_options_prefix it is a TypeError, and solve() RETURNS when Newton did not converge too "
        "(reason -5 at the iteration cap), so read problem.solver.getConvergedReason() (> 0 converged) and "
        "problem.solver.getIterationNumber() (the Newton iteration count), or set "
        "'snes_error_if_not_converged': True to make it raise. `from dolfinx.fem.petsc import NewtonSolver` is "
        "an ImportError; NonlinearProblem needs none."
        "\n20. THE MESH MOVES THROUGH ITS GEOMETRY: there is no dolfinx.mesh.move (AttributeError). "
        "msh.geometry.x is the (n, 3) array of node coordinates, and forms assembled afterwards read it: for a "
        "displacement d on the mesh's own P1 vector space, msh.geometry.x[:, :gdim] += d.x.array.reshape(-1, "
        "gdim). Measured in serial on create_rectangle triangle meshes: that space's dof order equalled the node "
        "order, and the area integral changed by exactly the stretch applied; compare "
        "V1.tabulate_dof_coordinates()[:, :gdim] with msh.geometry.x[:, :gdim] before relying on it."
        "\n21. assemble_scalar IS dolfinx.fem.assemble_scalar(fem.form(...)); dolfinx.fem.petsc has none "
        "(AttributeError)."),
}
_FSI_FACTS["fenicsx"] = _FSI_FACTS["dolfinx"] = _FSI_FACTS["fenics"]
_FSI_FACTS["fourc"] = (
    "\nFOR A STRUCTURE SIDE (each run as a small deck on this install):"
    "\n16. A 2-D STRUCTURE IS MADE OF WALL ELEMENTS: `<id> WALL QUAD4 <n1> <n2> <n3> <n4> MAT 1 KINEM "
    "linear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 2 2` in STRUCTURE ELEMENTS, corners "
    "counter-clockwise (the binary's grammar, `4C -p`, also lists WALL QUAD8, QUAD9, TRI3, TRI6). SOLID "
    "takes 3-D cells only: SOLID QUAD4 stops 4C with \"Element 'SOLID' does not seem to know cell type "
    "'quad4'.\""
    "\n17. LINEAR ELASTICITY IS MAT_Struct_StVenantKirchhoff (YOUNG, NUE, DENS) with KINEM linear on the "
    "element: on a plane-strain block under an edge traction it reproduced the plane-strain Hooke solution "
    "to 2e-16. There is no MAT_Struct_LinearElastic: 4C stops with 'Failed to match specification in "
    "section MATERIALS'."
    "\n18. A TRACTION ON AN EDGE OF A 2-D BODY IS DESIGN LINE NEUMANN CONDITIONS on that edge's DLINE "
    "(NUMDOF 2, ONOFF, VAL, FUNCT, TYPE \"Live\"); with THICK 1.0 the load is VAL x FUNCT per unit length "
    "of the edge. DESIGN SURF NEUMANN in 2-D loads the whole surface per unit area, a body force: the same "
    "VAL there ran with no error and gave a different displacement."
    "\n19. THE DISPLACEMENT COMES BACK IN A VTU ONLY WHEN ASKED FOR: `IO/RUNTIME VTK OUTPUT` "
    "(INTERVAL_STEPS: 1, OUTPUT_DATA_FORMAT: \"ascii\") with `IO/RUNTIME VTK OUTPUT/STRUCTURE` "
    "(OUTPUT_STRUCTURE: true, DISPLACEMENT: true) writes out-vtk-files/structure-<step>-<rank>.vtu with "
    "the point array `displacement`, every node repeated once per element; step 00000 is the initial "
    "state, the last step the solution. Without those two sections the run finishes normally and writes "
    "no VTU. `node_gid` is written too only with NODE_GID: true."
    "\n20. A FUNCTn NO CONDITION LISTS IS EVALUATED NOWHERE: a deck with an unused FUNCT1 ran normally and "
    "gave the same displacement as one without it. The load a FUNCT builds reaches 4C only through the "
    "condition's FUNCT: [..] entry for that component.")
_FSI_FACTS["4c"] = _FSI_FACTS["fourc"]

_DECIDING_UNIVERSAL = (
    "* READ YOUR FIELD AT THE PROBE POINTS BY INTERPOLATION, NEVER BY NEAREST "
    "NODE. Measured: one solve exported two ways converged at second order by interpolation "
    "and at first order by nearest-node sampling. Free self-check: nearest-node sampling can only return "
    "(N-1)^2+1 distinct values on N cells per side, however many probes: "
    "count yours.\n"
    "* GATE BEFORE YOU HAND IN, at EVERY level: the solver REPORTED convergence "
    "(not merely that nothing raised), peak|u| > 0, and your load is not "
    "constant. Then compute log2(|L1-L2|/|L2-L3|) yourself.\n"
    "* ON A COUPLED SIDE, the consistent outward flux is "
    "q = -(K u - b_volume)/h with h the node's tributary length. The FACE "
    "load must NOT go into that residual -- include it and the reported flux "
    "comes out identically zero.")


# A DSMC SIDE GETS NO FINITE-ELEMENT RULE. Measured on a coupled round: SPARTA's deciding facts
# ended with the probe-interpolation, "your load is not constant" and q = -(K u - b_volume)/h
# rules, none of which a DSMC deck has, and the side that read them had no field to apply them to.
_NO_FEM_RULES = frozenset({"sparta"})


def _deciding_block(solver: str, physics: str, fsi: bool = False) -> str:
    """The lead block: what decides this run, before the corpus. `fsi`: a fluid-structure side, which
    gets the code's fluid-structure facts after its own (inside the block, so no cap cuts them)."""
    key = (solver or "").strip().lower()
    facts = _DECIDING_FACTS.get(key, "") + (_FSI_FACTS.get(key, "") if fsi else "")
    universal = "" if key in _NO_FEM_RULES else _DECIDING_UNIVERSAL
    if not facts and not universal:
        return ""
    head = (f"# WHAT DECIDES THIS RUN — {solver}/{physics}\n"
            f"# (read this before the corpus below; each line was measured by "
            f"execution)\n\n")
    body = "\n\n".join(x for x in (facts, universal) if x)
    return head + body + "\n\n" + "-" * 70 + "\n\n"


# EVERY KNOWLEDGE REPLY IS BOUNDED. Measured on coupled recorded runs
# with a small model: single knowledge replies of 55-145k characters, ten of
# them per run, 1-9 million input tokens per run, and the run gave up
# citing the size of the job. The coupling door already caps its head at
# _COUPLING_HEAD_LIMIT; this is the same discipline at the one place every
# knowledge reply leaves the server. The participant parts door is exempt
# (its chunks are bounded by construction and must arrive whole).
_KNOWLEDGE_REPLY_LIMIT = 48_000
_WORD_BOUND_CHARS = 6_000 * 6
_SIX_THOUSAND_CLAUSE = (" -- beyond roughly 6,000 words of instruction, measured, "
                        "models stop following instructions altogether")


def _true_to_its_length(text):
    """The closing sentence about six thousand words stays only on a reply that
    is under that bound. A label describing a cap that is not enforced is a
    label standing in for a guard: measured on a 109k prepare reply and on the
    physics door's 48k deal.II reply, both ending with the promise."""
    if isinstance(text, str) and len(text) > _WORD_BOUND_CHARS and _SIX_THOUSAND_CLAUSE in text:
        return text.replace(_SIX_THOUSAND_CLAUSE, "")
    return text


def _deciding_block_span(text: str, solver: str) -> tuple:
    """Where the deciding-facts block sits in a built reply, or (-1, -1).

    THE FACTS ARE REFERENCE, NOT INSTRUCTION, AND THE CAP WAS EATING THEM.
    Measured 2026-09-14 on knowledge(topic='coupling', solver='ngsolve'): the
    block began at 41,720 characters of a 47,874-character reply and the 48k
    cap cut FOUR of its nine lines off the end -- including the three added
    that morning after runs died on exactly those calls. The
    same rule the deck skeletons already have applies here: cap the corpus,
    never the measured facts. They are 2-4k per code.
    """
    if not isinstance(text, str) or not solver:
        return (-1, -1)
    head = f"# WHAT DECIDES THIS RUN \u2014 {solver}"
    i = text.find(head)
    if i < 0:
        i = text.find("# WHAT DECIDES THIS RUN")
        if i < 0:
            return (-1, -1)
    # THE CLOSING RULE, WITH OR WITHOUT ITS TRAILING BLANK LINE. The deck-grammar
    # branch rstrips the body before capping it, which ate the block's final
    # "\n\n" and made an exact match fail -- so FEBio's and SPARTA's facts were
    # still cut after the protection landed (measured 2026-09-14).
    rule = "\n" + "-" * 70
    j = text.find(rule, i)
    if j < 0:
        return (-1, -1)
    end = j + len(rule)
    while end < len(text) and text[end] in "\n":
        end += 1
    return (i, end)


def _probe_recipe_span(text: str) -> tuple:
    """Where the read-back recipe sits in a built reply, or (-1, -1).

    THE RECIPE EXISTS BECAUSE OF THE BIGGEST FAILURE BUCKET, AND THE CAP ATE IT.
    deal.II's backend adds a `probe_recipe` to every knowledge branch because a
    solve that worked and was never read back at the prescribed points is the
    largest single way a run is lost -- 60 recorded runs, 12.9 %. The
    backend comment says adding it to one branch would leave the other three
    silent. It was added to all four, and then the 48,000-character cap cut it
    off every one of them: measured on
    knowledge(topic='physics', solver='dealii', physics='heat'), the reply came
    to 47,731 characters and carried no `VectorTools::point_value` at all.

    Same rule as the deciding-facts block above: cap the corpus, never the
    short thing the agent is supposed to copy.
    """
    if not isinstance(text, str):
        return (-1, -1)
    head = "READING YOUR SOLUTION AT A POINT THAT IS NOT A MESH NODE"
    i = text.find(head)
    if i < 0:
        return (-1, -1)
    # Runs to the next top-level heading, or to the end of the reply.
    j = text.find("\n\n# ", i + len(head))
    return (i, len(text) if j < 0 else j)


def _cap_knowledge_reply(out: str, topic: str = "", solver: str = "",
                         physics: str = "", signal: str = "",
                         limit: int | None = None) -> str:
    limit = _KNOWLEDGE_REPLY_LIMIT if limit is None else limit
    if not isinstance(out, str) or len(out) <= limit:
        return out
    if (signal or "").strip().lower().startswith("participant"):
        return out
    # THE MEASURED FACTS ARE NEVER THE THING THAT GETS CUT. They are short
    # (2-4k), they are the reason a weak model's first attempt runs at all,
    # and _with_facts puts them at the END of a session's first reply -- which
    # is exactly where a cap bites. Measured 2026-09-14 on
    # knowledge(topic='coupling', solver='ngsolve'): the block started at
    # 41,720 of a 47,874-character reply and FOUR of its nine lines were cut,
    # including three added that morning after workers died on those calls.
    # So: lift the block out, cap the corpus around it, put it back where it
    # was. Same rule as the deck skeletons and the first contract block.
    # THE FIRST CONTRACT BLOCK IS NEVER CUT HERE EITHER. The front-loader keeps it whole and the 4C
    # branch keeps it whole, and this cap -- the last thing every reply passes through -- then cut it
    # anyway. Measured 2026-09-14 on knowledge(topic='coupling', solver='dune', physics='3d'): the
    # served text ended mid-block, with an opening ```python and no closing fence, so the worker was
    # handed a truncated participant. All three C10 cells of round 49 failed on exactly that side.
    # A block the agent copies is reference, not instruction; the budget bends around it.
    _end = _contract_block_end(out, 0)
    if _end > limit:
        limit = _end
        if len(out) <= limit:
            return out
    _pa, _pb = _probe_recipe_span(out)
    if _pa >= 0 and (_pb - _pa) < limit // 2:
        recipe = out[_pa:_pb]
        rest = _cap_knowledge_reply(out[:_pa] + out[_pb:], topic, solver, physics,
                                    signal, limit=limit - len(recipe))
        return (rest[:_pa] + recipe + rest[_pa:] if _pa <= len(rest)
                else rest + "\n\n" + recipe)
    # THE CONTRACT BLOCK RIDES OUTSIDE THE INSTRUCTION BUDGET, like the deck
    # skeletons and the measured facts. Keeping the end whole is not enough:
    # the block and the prose shared one window, so every line ADDED to a
    # contract silently cost a line of prose at the far end of the must-read.
    # Measured 2026-09-14 on knowledge(topic='coupling', solver='ngsolve'):
    # adding the per-level dump (1.6k) pushed "THE COUPLING HISTORY IS
    # MEASURED, NOT MODELLED" -- the anti-fabrication paragraph -- off the end
    # of the reply. A block the agent COPIES is reference; the prose it reads
    # is the instruction. They must not compete for the same bytes.
    _ca, _cb = _contract_block_span(out)
    if _ca >= 0 and _cb > _ca:
        blk = out[_ca:_cb]
        rest = _cap_knowledge_reply(out[:_ca] + out[_cb:], topic, solver, physics,
                                    signal, limit=limit)
        return rest[:_ca] + blk + rest[_ca:] if _ca <= len(rest) else rest + "\n\n" + blk
    _fa, _fb = _deciding_block_span(out, solver)
    if _fa >= 0 and (_fb - _fa) < limit // 2:
        facts = out[_fa:_fb]
        rest = _cap_knowledge_reply(out[:_fa] + out[_fb:], topic, solver, physics,
                                    signal, limit=limit - len(facts))
        return rest[:_fa] + facts + rest[_fa:] if _fa <= len(rest) else rest + "\n\n" + facts
    head = out[:limit]
    cuts = [head.rfind(m) for m in ("\n────", "\n\n#", "\n\n", "\n")]
    cut = max([c for c in cuts if c > limit // 2], default=-1)
    if cut > 0:
        head = head[:cut]
    asked = ", ".join(f"{k}={v!r}" for k, v in (("topic", topic), ("solver", solver),
                                                ("physics", physics), ("signal", signal)) if v)
    return (head + f"\n\n{'─' * 70}\n"
            f"THIS REPLY IS CUT AT {len(head):,} OF {len(out):,} CHARACTERS "
            f"({asked}). Nothing is lost: ask again with a NARROWER request -- "
            f"a physics= name, a signal= symptom (the error text you saw), or "
            f"index=True on the pitfalls topic -- and the part you need comes "
            f"back within the limit. Reading everything costs the actions you "
            f"need to solve the problem.\n")


# The levels a couple_levels call has still to run while it runs one of them: the files
# of those levels on disk are a previous attempt's, which this call is about to replace,
# and couple()'s reply for the level in hand does not lead with them (measured: a
# ladder's level-1 reply led with "COUPLING HISTORY TOO SHORT" on the previous
# ladder's residual_level2.csv, which the same call rewrote minutes later).
import contextvars as _contextvars                                  # noqa: E402
_LADDER_PENDING = _contextvars.ContextVar("openpaso_ladder_pending", default=())
# True while couple_levels runs one of its levels: the ladder checks each side against
# its own equation after its last level, so a level's reply does not say none was
# checked (measured: level 3 of a verified ladder led with the note that no level had
# been checked against its own equation, beside the ladder's own SATISFIES).
_IN_LADDER = _contextvars.ContextVar("openpaso_in_ladder", default=False)


# The interface files the healer wrote, by path, with the digest of what it wrote: a
# later run of the same level refreshes a file that is still the healer's, and a file
# the participant wrote or edited is never touched.
_HEALED_FILES: dict = {}


# HOW MUCH INTERFACE DATA ONE REPLY CARRIES INLINE. Measured: a transient side exported its
# whole time history per interface point, couple() inlined it as CSV rows, one level's block
# was 102,833 characters, and the couple_levels reply that carried it (249,047 characters in
# the tool record) was the last thing the run received -- the harness session died one second
# later. A block over the budget is named, never inlined: the numbers are the side's own
# exports.json already.
_IFACE_CSV_INLINE_MAX = 30_000           # characters of CSV rows in one couple() reply
_LADDER_IFACE_CSV_INLINE_MAX = 40_000    # the same, summed over every level of one couple_levels reply


def _iface_block_not_inlined(pname: str, block: dict, n_chars: int, n_points=None, n_cols=None,
                             level=None) -> dict:
    """The interface block that stays out of a reply: its file name and where its numbers are."""
    what = (f"{n_points} points x {n_cols} columns, " if n_points and n_cols else "")
    where = (f"{pname}'s own exports.json in its work_dir (overwritten by its next run)" if level is None else
             f"{pname}'s own interface dump of level {level} (interface_level{level}.csv in its work_dir; "
             f"openPASO writes it from the level's exports.json when the side does not, and says so under "
             f"interface_dump_written_for_you) -- its exports.json now holds a later level")
    return {"suggested_filename": block.get("suggested_filename"),
            "not_inlined": (f"{what}{n_chars:,} characters: more than one reply carries. The same numbers are "
                            f"{where}; write this file from there with a short script -- never by retyping "
                            f"numbers.")}


def _as_json_text(value):
    """A tool argument documented as a JSON list or object, as its JSON text: a list or a dict
    that arrives as such is serialised, a string is left as it is."""
    return value if isinstance(value, str) else json.dumps(value)


def _level_fault_is_field(text: str) -> bool:
    """True when a level's not-verified reason is a defect of its field -- not a caveat the next
    level decides and not a missing critic review."""
    if "critic review" in text or "critic has not reviewed" in text:
        return False
    return ("NOT VERIFIED --" in text or "NOT VERIFIED —" in text
            or "IS NOT A COUPLED RESULT" in text)


def _heal_missing_interface_dump(specs: list, k: int) -> dict:
    """Write the per-level interface file openPASO already holds the data for.

    HEALING, NOT WARNING, AND THIS IS THE CASE THAT EARNED IT. Measured on a
    live coupled run: the coupling SUCCEEDED -- three levels, both codes proven,
    interface residual 6.8e-11, max|u| of 6.57e-07 on side A -- and the
    participants wrote no per-level dumps. openPASO detected that and said so at
    the level, with the correct fix ("couple() this level AGAIN before anything
    is written from its dumps"). The agent answered by hand-writing the
    deliverables:

        # Placeholder: small values based on manufactured solution
        ux = 0.0
        uy = 0.0

    Every graded number was invented, after the physics was already right. A
    warning is only as good as the cheapest way around it, and hand-writing a
    file is very cheap.

    So for the one artefact openPASO ALREADY HAS, it stops asking. Each side's
    exports.json carries that level's interface coordinates, trace and recovered
    flux -- the exact content of interface_level<k>.csv -- and openPASO is
    holding it at the only moment it is correct. The served contract's own
    comment says why the agent cannot do this later: "exports.json is
    overwritten by the next level ... a run that rebuilt level 1's interface
    file from exports.json after level 2 got level 2's numbers".

    WHAT IT WILL NOT DO. It never writes field_level<k>.csv: that is the volume
    field and openPASO does not have it, only the participant does. It never
    overwrites a file the participant wrote. And it invents nothing -- every
    number transcribed here came out of the agent's own solver.

    A FILE IT WROTE IS ITS OWN TO KEEP CURRENT. Measured: a level coupled again
    after a fix kept the healer's file from the earlier, wrong run, and that
    stale file was handed in. A file still holding exactly what the healer wrote
    is rewritten from this run's exports; anything else is the participant's.
    """
    import csv as _csv
    import hashlib as _hl

    healed: dict = {}
    for spec in specs or []:
        name = str(spec.get("name") or "?")
        wd = spec.get("work_dir")
        if not wd:
            continue
        work = Path(wd)
        target = work / f"interface_level{k}.csv"
        if any(work.glob(f"interface_level{k}_*.csv")):
            continue                       # the participant wrote its own
        if target.exists():
            try:
                _mine = _HEALED_FILES.get(str(target.resolve())) == _hl.sha256(target.read_bytes()).hexdigest()
            except OSError:
                _mine = False
            if not _mine:
                continue                   # the participant wrote its own
        src = work / "exports.json"
        if not src.is_file():
            continue
        try:
            data = json.loads(src.read_text())
        except Exception:                                    # noqa: BLE001
            continue
        coords = data.get("coordinates") or []
        vals = data.get("values") or []
        flux = data.get("normal_fluxes") or []
        if not coords or len(vals) != len(coords):
            continue
        try:
            with target.open("w", newline="") as fh:
                writer = _csv.writer(fh)
                writer.writerow(["x", "y", "u", "qn"])
                for i, point in enumerate(coords):
                    xy = list(point) if isinstance(point, (list, tuple)) else [point]
                    u = vals[i]
                    q = flux[i] if i < len(flux) else 0.0
                    u = u[0] if isinstance(u, (list, tuple)) and u else u
                    q = q[0] if isinstance(q, (list, tuple)) and q else q
                    writer.writerow([f"{float(xy[0]):.11e}",
                                     f"{float(xy[1]):.11e}" if len(xy) > 1 else "0",
                                     f"{float(u):.11e}", f"{float(q):.11e}"])
        except (OSError, TypeError, ValueError, IndexError):
            # A PARTIAL FILE IS WORSE THAN NONE: it would look like a dump and
            # be read as one. Remove it and leave the gap visible.
            try:
                target.unlink()
            except OSError:
                pass
            continue
        try:
            _HEALED_FILES[str(target.resolve())] = _hl.sha256(target.read_bytes()).hexdigest()
        except OSError:
            pass
        healed[name] = target.name
    return healed


def _unread_blocks(sides: list) -> set:
    """The exported blocks no side of a stated Dirichlet-Neumann pair reads: the
    Dirichlet side reads its partner's `values`, the Neumann side its partner's
    `normal_fluxes`. Empty unless exactly two sides both state their role."""
    out: set = set()
    try:
        from . import result_audit as _ra_role
        roles = {name: _ra_role._stated_role(Path(str(wd))) for name, wd in sides}
        if len(sides) == 2 and all(roles.values()):
            (a, _), (b, _) = sides
            for me, other in ((a, b), (b, a)):
                out.add(f"{other}.{'normal_fluxes' if roles[me] == 'dirichlet' else 'values'}")
    except Exception:                                    # noqa: BLE001
        return set()
    return out


_NOT_OUTPUT = ("exports.json", "imports.json", "config.json")


def _outputs_at_risk(wd: Path, level: int, since: float) -> list:
    """Files this level's run wrote in `wd` (to two folders deep) whose path names no level:
    the next level's run reuses those names. Scripts, the exchange files, config and the
    driver's consoles are not outputs."""
    out = []
    try:
        for q in list(wd.glob("*")) + list(wd.glob("*/*")) + list(wd.glob("*/*/*")):
            if not q.is_file() or q.name in _NOT_OUTPUT or q.suffix in (".py", ".pyc"):
                continue
            if q.name.startswith("participant_output") or ".replaced-" in q.name:
                continue
            rel = str(q.relative_to(wd)).lower()
            if re.search(r"level[_-]?\d+|lvl[_-]?\d+", rel):
                continue
            if q.stat().st_mtime >= since:
                out.append(rel)
            if len(out) >= 20:
                break
    except OSError:
        return out
    return out


def _driver_measured(finding: str, driver_error: str) -> str:
    """A finding that asks for a repeat run, told what the driver's own repeat run
    in the same call found (measured: a reply asked a run to "MEASURE FIRST: run its
    participant twice" beside the driver's text saying both sides repeated)."""
    if "MEASURE FIRST, on each side" not in finding:
        return finding
    if "returned the same export when run again" in driver_error:
        said = ("every side returned the same export when run again on its last imports, so "
                "check the exchange:")
    elif "IS NOT A FUNCTION OF ITS IMPORTS" in driver_error:
        said = ("a side did not return the same export when run again on its last imports "
                "(`error` names it) -- fix that side first; then check the exchange:")
    else:
        return finding
    return re.sub(r"MEASURE FIRST, on each side:.*?If both repeat, check the exchange:",
                  "THE DRIVER MEASURED THIS IN THIS CALL: " + said, finding, flags=re.S)


def _level_not_a_coupled_result(rep: dict) -> str:
    """Why this level's own evidence is not a coupling, or "" when it is one.

    THE SUMMARY AND THE LEVEL SAID DIFFERENT THINGS, AND THE LOUDER ONE WAS
    WRONG. Measured 2026-09-17 on a coupled recorded run: one couple_levels
    call returned `"all_levels_converged": true` with `"iterations": 2`,
    `"residual": 0.0` and a history of ONE row at every level, and its
    next_step read "EVERY REQUESTED LEVEL CONVERGED ... write the task's
    per-level field file". couple()'s own reply for those same levels carried
    "IS NOT A COUPLED RESULT YET ... Do not write this level's deliverables
    from this run" -- couple_levels copied that sentence into what_to_fix_next
    and then computed its verdict from `converged` alone. The run wrote the
    deliverables the headline asked for and was read as never having coupled.

    Two sides that exchange nothing cannot disagree, so their iteration reaches
    its fixed point at the first step. That is what the three shapes below are:
    a participant whose export does not depend on its imports, a history too
    short to show an exchange, and a residual that is exactly zero rather than
    small. The same evidence couple() already weighs, weighed again here so the
    summary cannot contradict the level.
    """
    unresp = sorted(n for n, st in (rep.get("responsiveness") or {}).items()
                    if "unresponsive" in str(st).lower())
    if unresp:
        try:
            from core.quality_checks import unresponsive_clause as _uc
            _parts = [_uc(n, rep.get("responsiveness_detail"), rep.get("interface_sensitivity"))
                      for n in unresp]
        except Exception:                                    # noqa: BLE001
            _parts = []
        if _parts and all(_parts):
            return "; ".join(_parts) + ", so the coupling stood still"
        return (f"participant(s) {', '.join(unresp)} exported byte-identical "
                f"data while their imports changed, so nothing was transmitted")
    rows = len(rep.get("history") or [])
    if rows < 3:
        return (f"the iteration stopped after {rows} step(s): a fixed point "
                f"reached at once means the exchanged data never changed "
                f"between iterations")
    try:
        last = float(rep.get("residual"))
    except (TypeError, ValueError):
        return ""
    if last == 0.0:
        return ("the reported interface residual is exactly zero, which is "
                "what two sides that exchange nothing produce, not what a "
                "converging iteration produces")
    return ""


def _level_field_peak(specs: list, k: int) -> dict:
    """How big is the field each side just wrote for this level?

    WHY THIS IS REPORTED RATHER THAN OFFERED. `couple_levels` already ended by
    telling the agent to call audit_results. Measured over the graded
    coupled record, a self-check tool was called in 40 of 217 cells with files,
    and a live full-task run here wrote a field of LITERAL ZERO at all three
    levels, reported an interface residual of 6.8e-11 as evidence of success,
    and called none of the three checks it had. audit_results fires six
    NEAR-ZERO findings on that workspace in one call. The agent never made it.

    So the peak magnitude of the agent's OWN exported file is put in the reply
    it is already reading. It reads a file the agent wrote and reports a number
    from it: no source term, no reference solution, no answer. A field that
    peaks below 1e-8 on a driven problem almost always means the load never
    arrived, and that is worth saying at the level it happened rather than
    three levels later.
    """
    import csv as _csv

    out: dict = {}
    for spec in specs or []:
        name = str(spec.get("name") or "?")
        wd = spec.get("work_dir")
        if not wd:
            continue
        found = sorted(Path(wd).glob(f"field_level{k}*.csv"))
        if not found:
            continue
        peak = 0.0
        rows = 0
        try:
            with found[-1].open() as fh:
                for row in _csv.reader(fh):
                    vals = []
                    for cell in row:
                        try:
                            vals.append(float(cell))
                        except ValueError:
                            vals = []
                            break                      # a header line
                    if len(vals) > 2:
                        rows += 1
                        peak = max(peak, max(abs(v) for v in vals[2:]))
        except OSError:
            continue
        if rows:
            out[name] = {"peak_abs_value": peak, "points": rows,
                         "file": found[-1].name}
    return out


def _verify_elastic(solution_files: str, source_term: str, coefficient: str,
                    domain: str, equation: str) -> str:
    """The elasticity branch of verify_pde_consistency.

    Kept beside the scalar one rather than inside it because it needs three
    things the scalar path does not: TWO source expressions, TWO material
    constants, and four columns per row.
    """
    import csv as _csv
    import json as _json
    from pathlib import Path

    from tools.pde_consistency import check_levels_elastic

    def _fail(msg: str) -> str:
        return "REFUSED: " + msg

    # Two source expressions. A vector equation needs one per component, and
    # guessing which single expression the agent meant is how a check starts
    # answering about a problem it was not given.
    parts = None
    try:
        loaded = _json.loads(source_term)
        if isinstance(loaded, (list, tuple)) and len(loaded) == 2:
            parts = [str(e) for e in loaded]
    except Exception:                                        # noqa: BLE001
        pass
    if parts is None and ";" in str(source_term):
        halves = str(source_term).split(";")
        if len(halves) == 2:
            parts = [h.strip() for h in halves]
    if parts is None:
        return _fail(
            "your equation is vector elasticity, so this check needs BOTH "
            "components of f. Pass source_term as a JSON pair, "
            '\'["<f_x>", "<f_y>"]\', or as the two expressions separated by a '
            "semicolon. One expression cannot be matched to a component, and "
            "guessing is how a check answers about a problem it was not given.")

    lam = mu = None
    try:
        got = _json.loads(coefficient)
        if isinstance(got, dict):
            low = {str(k).strip().lower(): v for k, v in got.items()}
            for key in ("lambda", "lam", "lmbda", "l"):
                if key in low:
                    lam = float(low[key])
                    break
            for key in ("mu", "m", "g", "shear"):
                if key in low:
                    mu = float(low[key])
                    break
    except Exception:                                        # noqa: BLE001
        pass
    if lam is None or mu is None:
        return _fail(
            "elasticity needs both Lame constants, as "
            '\'{"lambda": <lambda>, "mu": <mu>}\'. They are stated in your task; '
            "this check will not infer them from the field, because a wrong "
            "pair would make a right answer look wrong.")

    try:
        box = [tuple(float(v) for v in pair) for pair in _json.loads(domain)]
    except Exception:                                        # noqa: BLE001
        return _fail(f"domain {domain!r} must be JSON like [[0,1],[0,1]]")
    if len(box) != 2:
        return _fail("this elasticity check is written for two dimensions")

    levels, problems = {}, []
    for i, raw in enumerate(str(solution_files).split(","), start=1):
        path = Path(raw.strip())
        if not path.is_file():
            problems.append(f"{path} does not exist")
            continue
        rows = []
        try:
            with path.open() as fh:
                for row in _csv.reader(fh):
                    try:
                        rows.append(tuple(float(c) for c in row))
                    except ValueError:
                        continue                              # header
        except OSError as exc:
            problems.append(f"{path}: {exc}")
            continue
        rows = [r[:4] for r in rows if len(r) >= 4]
        if not rows:
            problems.append(
                f"{path}: needs four numeric columns x, y, ux, uy — a vector "
                f"problem is checked on both components")
            continue
        levels[i] = rows
    if not levels:
        return _fail("no readable level files. " + "; ".join(problems))

    try:
        result = check_levels_elastic(levels, parts[0], parts[1], lam, mu, box)
    except Exception as exc:                                  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    out = result.as_dict()
    out["operator"] = "-div(sigma(u)) = f, constant lambda and mu"
    if problems:
        out["files_skipped"] = problems
    return _json.dumps(out, indent=2) + _UNIVERSAL_CORE


_SOLVER_BINARIES = {"4C": "fourc", "febio4": "febio", "febio3": "febio", "febio": "febio",
                    "spa_serial": "sparta", "spa_mpi": "sparta"}
_LAUNCH_WRAPPERS = {"stdbuf", "mpirun", "mpiexec", "srun", "timeout", "nice", "env", "time"}


def _solver_binary_as_participant(spec) -> str:
    """'' unless a coupling participant's command runs a solver binary directly.

    A coupling participant is a program that reads ./imports.json and writes
    ./exports.json each iteration; a solver binary does neither, so nothing can
    be exchanged. MEASURED: all four cells of one round that went on to complete
    first passed the 4C binary itself as a participant, two to four minutes each,
    while the served text said "Python wrapper + YAML".
    """
    try:
        cmd = [str(c) for c in (spec.get("command") or [])]
    except Exception:                                    # noqa: BLE001
        return ""
    exe = None
    for tok in cmd:
        base = Path(tok).name
        if base in _LAUNCH_WRAPPERS or tok.startswith("-") or tok.isdigit():
            continue
        exe = base
        break
    if exe not in _SOLVER_BINARIES:
        return ""
    code = _SOLVER_BINARIES[exe]
    return (f"participant {spec.get('name', '?')}: its command runs the solver binary {exe} directly. "
            f"A coupling participant is a program that reads ./imports.json and writes ./exports.json "
            f"every iteration, and a solver binary does neither, so nothing would be exchanged. Run the "
            f"served participant contract instead -- a Python script that writes the input and runs "
            f"{exe} itself: command [<python>, \"participant.py\"]. If that side's folder already holds "
            f"a participant script that ran (an exports.json beside it), name THAT script in the command; "
            f"otherwise write_participant_contract(solver='{code}', path=<a new file name>) writes one -- "
            f"never over a script that already works.")


def register_consolidated_tools(mcp: FastMCP):
    """Register all consolidated tools — ~12 tools instead of 48."""
    _MUST_READ_STATE["served"] = False
    # Registration is the start of a session. The prepared-solver set is keyed
    # by id(mcp), and a test harness that builds one server object per call
    # gets the same id back once the previous object is collected -- measured
    # 2026-09-23: a "first" fourc/poisson call arrived wearing the coupled
    # hand-off of a session that no longer existed, 54k against 41k. A server
    # registers once, so this costs it nothing.
    _PREPARED_SOLVERS.pop(id(mcp), None)
    _PREPARED_SOLVERS.pop((id(mcp), "served"), None)

    # Session journal — records events for knowledge capture
    from core.session_journal import get_journal as _get_journal

    # ═══════════════════════════════════════════════════════════
    # 1. KNOWLEDGE (replaces 13 separate knowledge tools)
    # ═══════════════════════════════════════════════════════════

    def _knowledge_body(topic: str, solver: str = "", physics: str = "",
                        signal: str = "", category: str = "",
                        index: bool = False) -> str:
        """Get knowledge about solvers, physics, materials, coupling,
        post-mortems, or input formats.

        START WITH THE INDEX when you want pitfalls and do not yet know
        what to ask for: `knowledge(topic='pitfalls', solver=...,
        index=True)` returns a one-screen map — how many entries exist
        per physics and per category, and the exact call for each slice.
        Then narrow. If you already have an error message, skip the
        index and pass it as `signal=` directly.

        This is the single entry point for ALL domain knowledge — the
        catalog, the pitfall database, AND the post-mortem record
        store. Wiring post-mortems through this same tool closes the
        self-improvement loop: every prepare_simulation call also
        surfaces the relevant post-mortems so the critic-gate can
        retrieve them at planning time (the openPASO design paper §3.2 / §5
        self-correction loop).

        Args:
            topic: What you want to know. Options:
                - "physics" — physics-specific knowledge + matching
                  post-mortems (needs solver + physics)
                - "pitfalls" — known pitfalls for a solver. Unfiltered
                  returns every entry (comprehensive by design). Narrow
                  with `signal=` (the error you saw), `physics=`, or
                  `category=`; `index=True` maps what exists first. A
                  narrowed answer always states how many entries it held
                  back and how to get them.
                - "postmortems" — openPASO's formal post-mortem records,
                  filtered by solver + physics + optional signal pattern
                  (read them here: they are inside openPASO, not files in
                  your working directory). These are the
                  audit-trail entries that record WHY each pitfall
                  exists; the critic-gate should retrieve them when
                  the agent's plan touches the matching (solver,
                  physics) area.
                - "materials" — material catalog for a solver
                - "coupling" — cross-solver coupling knowledge
                - "tsi" — thermo-structural interaction patterns
                - "precice" — preCICE comparison
                - "input_guide" — how to write input files for a solver
                - "solver_guidance" — which solver to use for a physics type
                - "hardware" — parallelism, GPU, and hardware acceleration capabilities
                - "overview" — backend-level reference catalog (element
                  families, mesh types, solver catalogue, unique
                  features). The content under the special "_general"
                  knowledge key — for dealii ~5 KB, fenics / ngsolve /
                  skfem / kratos / dune ~1-2 KB each. Needs solver=...
                - "cross_backend" — collation pitfalls that surface ONLY
                  when porting a problem between two backends (units
                  conventions, Tet10/Hex20 node ordering, 'linear
                  elastic' semantic drift across backends, Dirichlet
                  strong-vs-penalty enforcement, restart file
                  incompatibility, MPI launch idioms). Pass the
                  optional `physics` arg as a topic filter
                  (e.g. 'units', 'mesh', 'bc', 'restart', 'mpi') to
                  narrow the response. These pitfalls belong to no
                  single backend's catalog because they only fire on
                  the delta between two.
                - "install" — how to INSTALL a backend, how openPASO
                  finds it, which environment variables matter, the
                  first-run failures with the exact message each one
                  produces, and — importantly — which claims depend on
                  how the backend was COMPILED. Read this when a
                  backend reports not_installed, when a run fails
                  before any physics happens, or before trusting any
                  claim whose signal is an assertion message (deal.II
                  compiles those out in Release), a vendor linear
                  solver (FEBio without MKL), a complex scalar type
                  (dolfinx real vs complex builds) or an accelerator
                  style (SPARTA without KOKKOS). Optional solver=...
                  narrows it to one backend; with no solver you get
                  every backend plus the probe commands. Also
                  reachable as "setup", "dependencies", "build_config"
                  and "portability".
            solver: Backend name (e.g. 'fenics', 'fourc', 'dealii', 'ngsolve')
            physics: Physics type (e.g. 'poisson', 'linear_elasticity', 'navier_stokes')
            signal: The error text you actually observed. Paste it raw —
                quoting, case and whitespace differences are folded, and
                a paraphrase still matches on distinctive terms. Filters
                `pitfalls`, `postmortems` AND `coupling`. Every result
                states the match mode, so a partial word-overlap is
                labelled weak instead of being presented as an
                identification. No match means the failure mode is not
                catalogued for that backend — it does NOT mean the setup
                is right.
                With topic='coupling' it is free text describing what
                you SAW, ranked against the coupling failure entries.
                Describe the observation, not the mechanism — "it
                converged but the answer is wrong", "the residual
                stops falling and stays there", "the two sides
                stopped agreeing" all route to the right entry. This
                is the fast path when a coupling misbehaves: it
                returns the two or three entries that explain the
                symptom instead of the whole payload.
            category: Narrow pitfalls by kind. In use, commonest first:
                Numerical, API, Input, Syntax, Physics, Integration,
                Performance, Output, Mesh, Validation. Spelling variants
                are folded, so 'numerics' finds 'Numerical'.
            index: For topic='pitfalls', return the map instead of the
                content — entry counts per physics and per category plus
                the call that fetches each. Use it to choose a filter
                before pulling the full set.
        """
        _get_journal().record("knowledge_lookup", "knowledge",
                              solver=solver, physics=physics,
                              notes=f"topic={topic}")
        if topic == "physics" and solver and physics:
            backend = get_backend(solver)
            if not backend:
                return f"Unknown solver: {solver}"
            # FUZZY MATCH HERE TOO — the pairing was exactly inverted.
            # prepare_simulation has a seven-stage matcher and this had none,
            # so the tool WITH the matcher lacked the universal block and the
            # tool WITH the block dead-ended. Measured over 98 recorded runs:
            # knowledge(topic='physics', solver='fenics', physics='nonlinear')
            # returned 38 characters, 16 times across 13 runs, while
            # 'nonlinear' matches nonlinear_pde — the generator that builds
            # exactly the equation shape those runs were asked to solve.
            k = backend.get_knowledge(physics)
            if not k:
                _m = _fuzzy_match_physics(backend, physics)
                if _m:
                    k = backend.get_knowledge(_m)
                    if k:
                        physics = _m
            if not k:
                _avail = ", ".join(p.name for p in backend.supported_physics())
                return (f"No knowledge for '{physics}' in {solver}. "
                        f"Available: {_avail}")
            k = _strip_pitfalls(k)
            # A SECTION BY NAME. The fitter below tells a reader to ask for a
            # thinned or dropped section by name; this is the door that
            # answers. Exact key first, then a substring of one.
            if signal and isinstance(k, dict):
                _want = signal.strip().lower()
                _hit = (next((key for key in k if key.lower() == _want), None)
                        or next((key for key in k if _want in key.lower()), None))
                if _hit is not None:
                    return (json.dumps({_hit: k[_hit]}, indent=2, default=str)
                            + f"\n\n[section '{_hit}' of knowledge(topic='physics', "
                              f"solver='{solver}', physics='{physics}'), whole]"
                            + _physics_tail())
            from tools.knowledge import _find_reference_test_files
            ref = _find_reference_test_files(solver, physics)
            # Post-mortem BREADCRUMBS (ids only) — not full records — at plan
            # time. Rationale (senior-AI-scientist critic, 2026-05-31): full
            # post-mortems include surface_symptom / root_cause /
            # agent_detection_after_fix, which are diagnostic fields for human
            # review (#46), not pre-execution guidance. Auto-including them at
            # plan time produces linear token bloat in N_postmortems and
            # competes with the catalog for the agent's attention. The
            # pitfall_db_entries the catalog already exposes ARE the
            # pre-execution actionable content; the full post-mortem belongs
            # to the post-execution critic when it has a Signal: to match.
            # Agent can fetch the full record explicitly via
            # `knowledge(topic="postmortems", solver=..., signal=...)`.
            postmortems = ([] if _ABLATE_PITFALLS
                           else _load_matching_postmortems(solver, physics, ""))
            crumbs = ""
            if postmortems:
                breadcrumbs = [
                    {"id": pm.get("id", "?"),
                     "categories": pm.get("categories", []),
                     "date": pm.get("date", "")}
                    for pm in postmortems
                ]
                crumbs = (
                    f"\n\n## Post-mortem breadcrumbs "
                    f"({len(postmortems)} record"
                    f"{'' if len(postmortems) == 1 else 's'} — "
                    f"fetch full records via knowledge"
                    f"(topic='postmortems', solver=..., signal=...)"
                    f" when a post-execution Signal needs lookup):\n"
                    + json.dumps(breadcrumbs, indent=2))
            # THE RECORD IS FITTED, NEVER SLICED. This door is the full record
            # every other door points at (prepare's own trim note names it),
            # and the exit cap used to slice its JSON: measured 2026-09-23,
            # deal.II/poisson at 54,792 characters of JSON came back as
            # 48,432 -- cut mid-object, unparseable, the pitfalls key (last in
            # the record) gone, and a closing sentence promising six thousand
            # words. The JSON is fitted to the room the reply has: pitfalls
            # and runnable entries are load-bearing and never yield, large
            # prose entries thin and then drop, each named in a note that
            # says how to fetch it by name, and the cap downstream leaves the
            # JSON head alone.
            _tail = _physics_tail()
            _room = (_KNOWLEDGE_REPLY_LIMIT - len(_tail)
                     - (len(ref) + 2 if ref else 0) - len(crumbs) - 400)
            _hint = (f"knowledge(topic='physics', solver='{solver}', "
                     f"physics='{physics}', signal='<section name>')")
            if isinstance(k, dict):
                result, _note = _fit_json_block(k, max(8000, _room), fetch_hint=_hint)
                result += _note
            else:
                result = json.dumps(k, indent=2, default=str)
            if ref:
                result += f"\n\n{ref}"
            return result + crumbs + _tail

        elif topic == "postmortems":
            if _ABLATE_PITFALLS:
                return ("No post-mortems found. openPASO's post-mortem store is "
                        "the canonical one; absence here means the failure "
                        "mode has not yet been audited.")
            postmortems = _load_matching_postmortems(solver, physics, signal)
            if not postmortems:
                what = ", ".join(
                    f"{k}={v!r}" for k, v in
                    {"solver": solver, "physics": physics,
                     "signal": signal}.items() if v)
                return (f"No post-mortems found"
                        f"{' for ' + what if what else ''}. "
                        f"openPASO's post-mortem store is the canonical "
                        f"one; absence here means the failure mode "
                        f"has not yet been audited.")
            return json.dumps(postmortems, indent=2)

        elif topic == "pitfalls" and solver:
            # Ablation: when OFA_DISABLE_PITFALLS=1, refuse to surface
            # pitfalls so the agent has no shortcut to known-bug knowledge.
            if _ABLATE_PITFALLS:
                return f"No pitfalls available for {solver}"
            # COUPLING IS NOT A BACKEND, and that is exactly why it had no
            # pitfalls surface: `get_backend('coupling')` returns None, so this
            # branch fell straight through to "No pitfalls found for coupling"
            # while ~145 kB of coupling knowledge sat behind topic='coupling'
            # with no [Category] tag and no Signal: clause anywhere in it. An
            # agent whose coupling had just failed could not look its symptom
            # up — at the exact moment it most needed the knowledge, symptom
            # lookup returned nothing. Served here in the same shape every
            # backend uses ({group: [entry, ...]}), so the narrowing layer
            # applies to it unchanged when it lands.
            if solver.strip().lower() in ("coupling", "couple", "coupled"):
                from backends.coupling import get_coupling_pitfalls
                entries = get_coupling_pitfalls(physics or None)
                if signal:
                    entries = _narrow_coupling_by_signal(entries, signal)
                return json.dumps(entries, indent=2)
            # Backend is the source of truth for pitfalls (Table-1
            # promoted, post-execution-critic-actionable). The
            # deep_knowledge fallback was inverted historically —
            # it returned prose entries even for backends whose
            # generators had been Table-1 promoted, breaking the
            # alignment between prepare_simulation and discover.
            # Now backend is consulted FIRST; deep_knowledge is
            # only used as a supplement for physics the backend
            # does not enumerate (rare in practice).
            backend = get_backend(solver)
            all_pitfalls = {}
            # `guidance` holds entries that are real, useful advice but have
            # NO failure mode and NO observable — "filter radius should be
            # > 2-3x element edge length" and the like. They used to sit in
            # `pitfalls` carrying a boilerplate Signal clause they could not
            # deliver on, which made symptom lookup return them alongside
            # genuine pitfalls and diluted it. They are surfaced here under a
            # separate top-level key so nothing is lost, but they are not
            # pitfalls and must not be counted or matched as such.
            guidance = {}
            if backend:
                for p in backend.supported_physics():
                    k = backend.get_knowledge(p.name)
                    if k and "pitfalls" in k:
                        all_pitfalls[p.name] = k["pitfalls"]
                    if k and k.get("guidance"):
                        guidance[p.name] = k["guidance"]
            try:
                from tools.deep_knowledge import _4C_KNOWLEDGE, _FENICS_KNOWLEDGE
                dicts = {"fourc": _4C_KNOWLEDGE, "4c": _4C_KNOWLEDGE,
                         "fenics": _FENICS_KNOWLEDGE, "fenicsx": _FENICS_KNOWLEDGE}
                d = dicts.get(solver.lower(), {})
                for k, v in d.items():
                    if (isinstance(v, dict) and "pitfalls" in v
                            and k not in all_pitfalls):
                        all_pitfalls[k] = v["pitfalls"]
            except ImportError:
                pass
            if not backend:
                if all_pitfalls:
                    return json.dumps(all_pitfalls, indent=2)
            if backend:
                # Also include general input-format pitfalls (e.g., ExodusII
                # block IDs, FUNCT syntax, shared-node NUMDOF conflict)
                general_k = backend.get_knowledge("input_format")
                if isinstance(general_k, dict):
                    gp = general_k.get("general_pitfalls")
                    if gp:
                        all_pitfalls["general_input_format"] = gp
                    et = general_k.get("element_type_per_physics")
                    if et:
                        all_pitfalls["element_types"] = et
                # Include community-contributed knowledge
                community = _load_community_knowledge(solver)
                if community:
                    all_pitfalls["community_contributed"] = [
                        {"title": c["title"], "description": c.get("description", ""),
                         "category": c.get("category", ""), "confidence": c.get("confidence", 0)}
                        for c in community
                    ]
                # REFERENCE-ONLY AREAS. `supported_physics()` is a capability
                # claim — a generator can build a runnable input. Knowledge
                # coverage is wider. On kratos the gap was 21 of 41 areas
                # holding 65 verified entries (geomechanics, RANS, IGA, ROM,
                # topology optimisation, chimera, FEM-to-DEM, FSI). All 20
                # non-underscore ones were checked: no template, and
                # generate_input raises ValueError, so keeping them out of
                # supported_physics() is right. Nothing enumerated them though,
                # so an agent could not learn they existed — and for a small
                # model, undiscoverable is indistinguishable from absent.
                #
                # Served here under an explicit prefix so the label travels
                # with the content: knowing four FSI traps is worth a lot even
                # when openPASO cannot write the FSI input for you, but an agent
                # must not read their presence as "I can run this".
                # Only entries not ALREADY served are added. Backends alias
                # heavily — 4C resolves 154 area names onto 248 texts that the
                # advertised path already returns, so adding them by name would
                # have duplicated 942 entry slots and taught an agent that 4C
                # has 154 undiscovered subjects. It has none; kratos has 20 and
                # deal.II 2, carrying 66 genuinely unreachable entries between
                # them. Deduplicating by entry TEXT is what separates the two
                # cases, and it is why the first count of this was wrong.
                advertised = {p.name for p in backend.supported_physics()}
                already = set()
                for _v in all_pitfalls.values():
                    already.update(pitfall_index._strings_under(_v))
                for area in pitfall_index.reference_only_areas(solver,
                                                               advertised):
                    k = backend.get_knowledge(area)
                    if not (isinstance(k, dict) and k.get("pitfalls")):
                        continue
                    fresh = [s for s in pitfall_index._strings_under(
                        k["pitfalls"]) if "Signal:" in s and s not in already]
                    if not fresh:
                        continue  # an alias of something already shown
                    already.update(fresh)
                    all_pitfalls[
                        f"{area} [reference only — no generator; "
                        f"write the input yourself]"] = fresh

                # Install / setup / build-configuration pitfalls. Merged
                # in here as well as being reachable at topic='install',
                # because an agent debugging a failed run asks for
                # 'pitfalls' and would otherwise never see that the
                # backend's binary was never validated, or that the
                # claim it is reading only holds on a Debug build.
                try:
                    from backends._setup import get_setup_pitfalls
                    sp = get_setup_pitfalls(solver)
                    if sp:
                        all_pitfalls["install_and_build_config"] = sp
                except ImportError:
                    pass
                if guidance:
                    all_pitfalls["_guidance_not_pitfalls"] = guidance
                # NARROWING. The signature has always accepted `physics` and
                # `signal`; this branch read neither. `signal` was wired only
                # to topic='postmortems', and `physics` was accepted and
                # ignored, so the loop above collected every physics the
                # backend supports. An agent holding a stack trace therefore
                # had to pull the whole dump (87k chars for kratos) and triage
                # it unaided — affordable in a 200k window, but it spends the
                # attention of exactly the small model least able to spare it.
                #
                # An unfiltered call still returns everything as JSON — now
                # including the reference-only areas above, which is strictly
                # more knowledge than before, not less. Filters are additive
                # and always report what they held back, so narrowing can never
                # make knowledge unreachable or make a miss look like an empty
                # database.
                if index:
                    return pitfall_index.index_summary(all_pitfalls, solver)
                if physics or signal or category:
                    # RESOLVE THE AGENT'S WORDS TO THE BUCKET NAME FIRST.
                    #
                    # `all_pitfalls` is keyed by the backend's canonical
                    # physics names, and narrow() matches the requested string
                    # against those keys. A backend that aliases — ngsolve maps
                    # "anisotropic diffusion" onto its `poisson` knowledge —
                    # therefore answered the alias with nothing: measured on
                    # one NGSolve problem, physics='anisotropic_diffusion'
                    # served 5,685 chars against 7,047 unfiltered, and the entry
                    # that decides that problem was unreachable by any phrase in
                    # its task text.
                    #
                    # Identity of the returned dict is the test, so this works
                    # for any backend that aliases, without this file knowing
                    # any backend's alias table.
                    p_req = physics
                    if physics and backend:
                        try:
                            target = backend.get_knowledge(physics)
                            if target:
                                for _p in backend.supported_physics():
                                    if backend.get_knowledge(_p) is target:
                                        p_req = _p
                                        break
                        except Exception:            # noqa: BLE001
                            p_req = physics
                    narrowed = pitfall_index.narrow(
                        all_pitfalls, physics=p_req, signal=signal,
                        category=category)
                    if p_req != physics:
                        narrowed.setdefault("filters_applied", []).append(
                            f"physics={physics!r} resolved to the backend's "
                            f"{p_req!r} knowledge")
                    return pitfall_index.render(narrowed, solver)
                # AN UNFILTERED DUMP IS NOT A PAYLOAD, IT IS A DENIAL OF
                # SERVICE ON THE AGENT'S OWN CONTEXT.
                #
                # This returned every pitfall verbatim: 394,733 bytes for 4C,
                # about 99k tokens — 38% of the whole 262k window in ONE tool
                # result. Measured over 635 run ledgers, runs with these tools
                # carry a median 82,669 tokens of context per tool call against
                # 57,193 without them, and get 46 tool calls per run against 101.
                # On the same 45-minute clock, that is less than half the
                # actions, and this call is the single largest contributor.
                #
                # The index already exists and agents already use it. So an
                # unfiltered request now RETURNS the index plus the entry
                # count, and says exactly how to get the full text of what it
                # names. Nothing is hidden and nothing is truncated mid-entry:
                # the agent chooses what to spend its window on.
                _n = sum(len(v) if isinstance(v, list) else 1
                         for v in all_pitfalls.values()) \
                    if isinstance(all_pitfalls, dict) else len(all_pitfalls)
                _full = json.dumps(all_pitfalls, indent=2)
                if len(_full) <= 24000:
                    return _full
                # index_summary, NOT render(): render returns every entry in
                # full, which is the thing being avoided. (My first attempt
                # called render with index=True, which is not a parameter it
                # takes — the TypeError fell through to the full render and
                # the payload stayed at 388,714 characters.)
                _idx = pitfall_index.index_summary(all_pitfalls, solver)
                return (
                    f"{_idx}\n\n"
                    f"[{_n} pitfall entries for {solver}, {len(_full):,} "
                    f"characters in full — roughly {len(_full)//4:,} tokens, "
                    f"which would be a large fraction of your context window "
                    f"in a single reply. The index above names every entry. "
                    f"Fetch only what you need:\n"
                    f"  knowledge(topic='pitfalls', solver='{solver}', "
                    f"physics='<name>')     — one physics\n"
                    f"  knowledge(topic='pitfalls', solver='{solver}', "
                    f"signal='<error text>') — entries matching a symptom\n"
                    f"  knowledge(topic='pitfalls', solver='{solver}', "
                    f"category='<category>') — one category\n"
                    f"Any of those returns the full text of the entries it "
                    f"selects.]")
            return f"No pitfalls found for {solver}"

        elif topic == "materials" and solver:
            backend = get_backend(solver)
            if not backend:
                return f"Unknown solver: {solver}"
            materials = {}
            for p in backend.supported_physics():
                k = backend.get_knowledge(p.name)
                if k and "materials" in k:
                    materials[p.name] = k["materials"]
            return json.dumps(materials, indent=2) if materials else f"No material catalog for {solver}"

        elif topic == "overview" and solver:
            # Surface the backend-level "_general" reference catalog
            # (element families, mesh types, solver catalogue, unique
            # features). Discovered 2026-06-02: get_knowledge('_general')
            # returns substantive reference content for 6 of 8 backends
            # (dealii 5.2 KB; fenics/ngsolve/skfem/kratos/dune 1-2 KB
            # each) but was NOT exposed via any `knowledge(topic=...)`
            # surface — LLMs had no way to discover it existed.
            backend = get_backend(solver)
            if not backend:
                return f"Unknown solver: {solver}"
            general = backend.get_knowledge("_general")
            if not isinstance(general, dict) or not general or "error" in general:
                return (f"No backend-level overview catalog for "
                        f"{solver} (get_knowledge('_general') is "
                        "empty or returned an error).")
            return json.dumps({solver: general}, indent=2)

        elif topic == "coupling":
            return _get_coupling_knowledge(solver, signal, physics)

        elif topic == "tsi":
            return _get_tsi_knowledge()

        elif topic == "precice":
            return _get_precice_knowledge(solver)

        elif topic == "input_guide" and solver:
            from tools.examples_search import (
                _4C_INPUT_GUIDE, _FENICS_INPUT_GUIDE, _DEALII_INPUT_GUIDE,
                _FEBIO_INPUT_GUIDE, _DUNE_INPUT_GUIDE,
                _SPARTA_INPUT_GUIDE,
            )
            guides = {"fourc": _4C_INPUT_GUIDE, "4c": _4C_INPUT_GUIDE,
                      "fenics": _FENICS_INPUT_GUIDE, "dealii": _DEALII_INPUT_GUIDE,
                      "febio": _FEBIO_INPUT_GUIDE,
                      "dune": _DUNE_INPUT_GUIDE, "dune-fem": _DUNE_INPUT_GUIDE,
                      "dunefem": _DUNE_INPUT_GUIDE,
                      "sparta": _SPARTA_INPUT_GUIDE, "dsmc": _SPARTA_INPUT_GUIDE}
            return guides.get(solver.lower(), f"No input guide for {solver}")

        elif topic == "solver_guidance" and physics:
            # Show ALL registered backends so the LLM can learn
            # which solvers offer the physics in principle —
            # even when not installed yet — and decide whether
            # to install one. Tag unavailable backends so the
            # LLM does not try to run_simulation on them.
            # (Audit 2026-06-02; same hide-unavailable bug as
            # discover('list').)
            results = {}
            for b in all_backends():
                for p in b.supported_physics():
                    if p.name == physics:
                        status, _ = b.check_availability()
                        key = (b.display_name() if status.value == "available"
                               else f"{b.display_name()} [{status.value}]")
                        results[key] = {
                            "variants": p.template_variants,
                            "elements": p.element_types,
                            "dims": p.spatial_dims,
                        }
            return json.dumps(results, indent=2) if results else f"No solver supports '{physics}'"

        elif topic == "hardware":
            hw = {
                "FEniCSx (dolfinx)": {
                    "parallelism": "MPI (first-class, domain decomposition via PETSc)",
                    "gpu": "No native GPU. PETSc can use GPU backends (CUDA/HIP) for linear algebra if compiled with Kokkos/CUDA support, but this is not standard.",
                    "threading": "Limited — PETSc threading for assembly",
                    "typical_scale": "Millions of DOFs on HPC clusters",
                },
                "deal.II": {
                    "parallelism": "MPI (p4est for distributed meshes) + threading (TBB/std::thread)",
                    "gpu": "Yes — matrix-free GPU kernels via CUDA and portable backends. GPU support for matrix-free operators is a key feature (step-64 tutorial).",
                    "threading": "SharedMemory::TBB or std::thread for assembly",
                    "typical_scale": "Billions of DOFs demonstrated (matrix-free, GPU)",
                },
                "4C Multiphysics": {
                    "parallelism": "MPI (domain decomposition) + OpenMP threading",
                    "gpu": "No GPU for linear algebra (Epetra-based, CPU-only). Optional ArborX (Kokkos) for GPU-accelerated geometric search only. Tpetra (GPU-capable) not yet integrated.",
                    "threading": "OpenMP (set OMP_NUM_THREADS)",
                    "typical_scale": "Millions of DOFs on MPI clusters",
                    "note": "Trilinos 16.2.0 is the last supported version due to Epetra dependency",
                },
                "NGSolve": {
                    "parallelism": "MPI (via NGSolve's own parallel framework) + shared-memory task parallelism",
                    "gpu": "Experimental CUDA support for some operations. Not production-ready for most users.",
                    "threading": "Task-based parallelism (Netgen's built-in scheduler)",
                    "typical_scale": "Millions of DOFs",
                },
                "scikit-fem": {
                    "parallelism": "Serial only (no MPI). NumPy/SciPy vectorisation for assembly.",
                    "gpu": "No GPU support. Pure Python/NumPy.",
                    "threading": "NumPy BLAS threading only",
                    "typical_scale": "Tens of thousands of DOFs (prototyping)",
                },
                "Kratos Multiphysics": {
                    "parallelism": "MPI (Trilinos-based) + OpenMP for shared memory",
                    "gpu": "Limited — some GPU acceleration via Trilinos/Kokkos for linear algebra. Not all applications support it.",
                    "threading": "OpenMP",
                    "typical_scale": "Millions of DOFs",
                },
                "DUNE-fem": {
                    "parallelism": "MPI (DUNE grid parallelism via ALUGrid/YaspGrid)",
                    "gpu": "No native GPU support in DUNE-fem. DUNE-copasi has experimental GPU work.",
                    "threading": "Limited",
                    "typical_scale": "Moderate (research scale)",
                },
                "FEBio": {
                    "parallelism": (
                        "Shared-memory only (OpenMP). No MPI domain "
                        "decomposition: a single FEBio process drives "
                        "the whole simulation. Multi-physics coupling "
                        "(biphasic, multiphasic, fluid-solid mixture) "
                        "is monolithic in the solver, not via "
                        "subdomain decomposition."),
                    "gpu": (
                        "No GPU support. FEBio's linear-algebra "
                        "back-end is CPU only (PARDISO / MUMPS / "
                        "Skyline). GPU acceleration is on the wishlist "
                        "but not implemented as of FEBio 4.x."),
                    "threading": (
                        "OpenMP across element assembly + PARDISO's "
                        "internal threading. Set OMP_NUM_THREADS for "
                        "assembly; the linear solver uses its own "
                        "OMP_NUM_THREADS or MKL_NUM_THREADS pool."),
                    "typical_scale": (
                        "Hundreds of thousands of DOFs on a workstation; "
                        "millions are routinely run but FEBio targets "
                        "biomechanical models (single bones, soft "
                        "tissue, biphasic cartilage) rather than HPC "
                        "scale."),
                    "note": (
                        "FEBio's strength is biomechanics-specific "
                        "physics (biphasic / multiphasic mixtures, "
                        "active contraction, fiber materials, "
                        "growth-remodeling). It is NOT a general-"
                        "purpose FEM code; do not pick it for "
                        "Navier-Stokes / electromagnetics / "
                        "geomechanics."),
                },
                "SPARTA (DSMC)": {
                    "parallelism": (
                        "MPI-first: the simulation grid is spatially "
                        "decomposed across ranks; SPARTA scales to "
                        "thousands of cores and is a production HPC DSMC "
                        "code (run: mpirun -np N spa_mpi -in <script>)."),
                    "gpu": (
                        "Yes — via the Kokkos package (build spa_kokkos): "
                        "CUDA (NVIDIA) and HIP (AMD) backends run the "
                        "particle move/collide/surface kernels on GPU. "
                        "Enable with '-k on g 1 -sf kk' package flags."),
                    "threading": (
                        "OpenMP or Kokkos (OpenMP) for shared-memory "
                        "parallelism; typically combined with MPI (MPI+X)."),
                    "typical_scale": (
                        "Billions of simulator particles on HPC clusters; "
                        "particle count (fnum) trades statistical noise "
                        "against cost, not DOFs."),
                    "note": (
                        "SPARTA is a Direct Simulation Monte Carlo (DSMC) "
                        "rarefied-gas / particle code, NOT a FEM solver — "
                        "reachable through openPASO coupling (e.g. a continuum "
                        "FEM thermal wall coupled to DSMC gas)."),
                },
            }
            if solver:
                key_map = {"fourc": "4C Multiphysics", "4c": "4C Multiphysics",
                           "fenics": "FEniCSx (dolfinx)", "fenicsx": "FEniCSx (dolfinx)",
                           "dealii": "deal.II", "deal.ii": "deal.II",
                           "ngsolve": "NGSolve", "skfem": "scikit-fem", "scikit-fem": "scikit-fem",
                           "kratos": "Kratos Multiphysics", "dune": "DUNE-fem", "dune-fem": "DUNE-fem",
                           "febio": "FEBio", "sparta": "SPARTA (DSMC)"}
                name = key_map.get(solver.lower(), solver)
                if name in hw:
                    return json.dumps({name: hw[name]}, indent=2)
                return f"No hardware info for {solver}"
            return json.dumps(hw, indent=2)

        elif topic == "cross_backend":
            # Cross-backend collation pitfalls — failures that live
            # in the delta between two backends both claiming to
            # solve the same problem. See src/backends/_cross.py
            # for content + rationale. The `physics` arg here is
            # repurposed as a topic filter (e.g. 'units', 'mesh',
            # 'bc', 'restart', 'mpi') to narrow the response.
            if _ABLATE_PITFALLS:
                return "No cross-backend collation entries available."
            from backends._cross import get_cross_backend_pitfalls
            result = get_cross_backend_pitfalls(physics or signal or None)
            return json.dumps(result, indent=2)

        elif topic in _SETUP_TOPIC_ALIASES:
            # Install / setup / build-configuration knowledge.
            # Deliberately reachable under several names: a caller
            # who needs this is by definition one who has not got the
            # backend working yet, and making them guess the exact
            # topic string is the wrong place to be strict. See
            # src/backends/_setup.py.
            #
            # NOT gated on _ABLATE_PITFALLS. That flag exists to
            # withhold solver-behaviour knowledge; withholding
            # "your Kratos wheel cannot load against this glibc"
            # would not weaken the agent's physics reasoning, it
            # would just make the machine look broken.
            from backends._setup import get_setup_knowledge
            return json.dumps(get_setup_knowledge(solver or None), indent=2)

        else:
            # A COUPLING QUESTION UNDER AN UNLISTED NAME IS STILL A COUPLING
            # QUESTION.
            #
            # The topic list above is exact, so a physically reasonable request
            # like topic='conjugate_heat_transfer' — measured in development-run
            # transcripts — matched nothing and got the usage message. That is a
            # dead end handed to an agent that had correctly identified its problem
            # as coupled, and the run then had to guess the vocabulary.
            #
            # `_is_coupling_request` already existed for exactly this, and was
            # wired into `examples` only. It is deliberately narrow — it fires on
            # words that are ABOUT coupling — so a single-code request is never
            # diverted here.
            if _is_coupling_request(topic):
                return _get_coupling_knowledge(solver, signal, physics)

            # THE LONG FORM, BECAUSE THE COMPACT BLOCK PROMISES IT.
            #
            # The served universal block was cut from 34,814 to 12,624
            # characters and its closing line tells the agent it can ask for
            # the rest with knowledge(topic="universal_full"). Before this
            # branch existed that call returned the usage hint -- a promise
            # made in the payload and not kept, which is the same defect this
            # file records for 'postmortems' just below, and the same shape as
            # every other mechanism here that did not reach its case.
            if str(topic).strip().lower() in (
                    "universal_full", "universal", "rules_full"):
                from .knowledge import _UNIVERSAL_FULL
                return _UNIVERSAL_FULL

            # Topics list must match the docstring + dispatch
            # branches. Audit 2026-06-01: 'postmortems' was
            # documented in the docstring and implemented at
            # line 326 but missing from this usage hint, so
            # LLMs hitting an invalid topic never learned that
            # postmortems exists. (Same drift class as
            # session_insights' missing 'ingest'.)
            return (
                "Usage: knowledge(topic, solver, physics, signal='')\n"
                "  universal_full - the long form of the ten universal "
                "rules, with the full measured evidence (22,501 further "
                "characters; you probably do not need it)\n"
                "Topics: physics, pitfalls, postmortems, materials, "
                "overview, coupling, tsi, precice, input_guide, "
                "solver_guidance, hardware, cross_backend, install\n"
                "If a backend is not running yet, or you need to know "
                "whether a claim depends on how it was compiled, use "
                "topic='install' (solver=... optional)."
            )



    # EVERY return of _knowledge_body GETS THE CORE RULES — not just the one.
    #
    # `_UNIVERSAL_BLOCK` was appended on 1 of that function's 31 return paths
    # (topic="physics"). Measured over 995 knowledge calls from 193 development
    # runs: topic="pitfalls" 66.3%, topic="physics" 11.5% — so 75.6% of those
    # runs received NONE of the universal guidance, and every rule added during
    # development reached at most a quarter of its audience.
    #
    # Wrapping is deliberate: appending at each return would fix today's 30 paths
    # and leak again at the next one added. functools.wraps carries the docstring
    # to @mcp.tool(), which reads it at decoration time, so there stays ONE copy.
    # assigned=("__doc__",) AND NOT THE DEFAULT. functools.wraps copies
    # __name__ too, and FastMCP names the tool from __name__ at decoration
    # time, so the default registered this as `_knowledge_body` — renaming the
    # single most-used tool out from under every agent. Caught by the test that
    # asks what the agent ends up with rather than what the code does.
    @mcp.tool()
    @_functools.wraps(_knowledge_body, assigned=("__doc__",), updated=())
    def knowledge(topic: str, solver: str = "", physics: str = "",
                  signal: str = "", category: str = "",
                  index: bool = False) -> str:
        out = _knowledge_body(topic, solver, physics, signal, category, index)
        if not isinstance(out, str):
            return out
        # the physics path already carries the full block; never send both.
        # CAP THE BODY, THEN APPEND THE CORE: appended first, the core sat at
        # the end of every long reply and the cap cut it off -- measured
        # 2026-09-11, the coupling door served 48k characters with no core
        # rules at all. The total stays within the reply limit.
        if _UNIVERSAL_BLOCK in out:
            # the physics path: the rules block and the post-mortem
            # breadcrumbs sit at the END of a catalog that alone can exceed
            # the cap (deal.II/stokes: 48k of JSON) -- cap the catalog, keep
            # the tail whole
            i = out.index(_UNIVERSAL_BLOCK)
            j = out.find("\n\n## Post-mortem breadcrumbs")
            if 0 <= j < i:
                i = j                      # the breadcrumbs precede the block: keep both
            body, tail = out[:i], out[i:]
            # THE JSON HEAD RIDES OUTSIDE THE SLICING CAP. The physics branch
            # fitted it (parseable, pitfalls kept, or whole when every entry
            # is load-bearing); only what follows it is capped here.
            _je = body.find("\n}") + 2 if body.lstrip().startswith("{") else 0
            if _je > 2:
                head, rest = body[:_je], body[_je:]
                rest = _cap_knowledge_reply(rest, topic, solver, physics, signal,
                                            limit=max(2000, _KNOWLEDGE_REPLY_LIMIT - len(head) - len(tail)))
                return _true_to_its_length(head + rest + tail)
            body = _cap_knowledge_reply(body, topic, solver, physics, signal,
                                        limit=max(8000, _KNOWLEDGE_REPLY_LIMIT - len(tail)))
            return _true_to_its_length(body + tail)
        if (topic or "").strip().lower() == "postmortems" and out.lstrip().startswith(("[", "{")):
            # a JSON record set that callers parse as JSON: appending prose
            # to it broke every parser (measured: json 'Extra data')
            return _cap_knowledge_reply(out, topic, solver, physics, signal)
        # THE DECK SKELETONS RIDE OUTSIDE THE INSTRUCTION BUDGET. The cap exists because
        # instructions past ~6,000 words stop being followed; a deck skeleton is a table
        # the agent copies, not an instruction, and it sat at the END of every 4C coupling
        # reply. Measured 2026-09-12: the worker's own call (facts 10.6k + the thermo-elastic
        # contract block 29.9k) left 6k of the 48k for a 16.7k grammar, so the TSI skeleton
        # never reached a single worker that day (round 40's C1 cells spent their budgets on
        # exactly that grammar). Now: the text before the grammar is capped as before, the
        # two skeletons are appended whole, the grammar's prose notes only when they fit.
        _gm = "THE 4C DECK GRAMMAR"
        if _gm in out and (topic or "").strip().lower() == "coupling":
            try:
                from backends.fourc.deck_grammar import (FOURC_DECK_GRAMMAR as _G, FOURC_DECK_NOTES as _GN,
                                                         FOURC_DECK_SKELETONS as _GS)
                i = out.index(_gm)
                after = out[i + len(_G):] if out.startswith(_G, i) else ""
                body = out[:i].rstrip("\n")
                room = _KNOWLEDGE_REPLY_LIMIT - len(_UNIVERSAL_CORE) - len(after)
                # THE FIRST CONTRACT BLOCK RIDES WHOLE TOO. Measured 2026-09-13: the 4C thermo-elastic
                # contract grew to 32k with the pre-run deck checks, its block ended past the body's
                # budget, and this cap cut its last lines -- the derived leave-behind block and the
                # closing fence -- out of the worker's reply (the front loader had kept it whole;
                # the cap here cut it again). A block the worker copies is reference, not instruction.
                room = max(room, _contract_block_end(body, 0))
                body = _cap_knowledge_reply(body, topic, solver, physics, signal, limit=max(8000, room))
                gram = _G if len(body) + len(after) + len(_G) <= _KNOWLEDGE_REPLY_LIMIT - len(_UNIVERSAL_CORE) else _GS
                return body + "\n\n" + gram + after + _UNIVERSAL_CORE
            except Exception:                            # noqa: BLE001
                pass
        # THE SAME RULE FOR THE OTHER BINARIES. A FEBio or SPARTA side is driven by a
        # deck / input script exactly as a 4C side is; its grammar (3.4k / 2.8k) is
        # appended after the contract and would be the first thing this cap cut.
        _dg = _deck_grammar_text(solver) if (topic or "").strip().lower() == "coupling" else ""
        if _dg and _dg in out:
            body = out.replace(_dg, "").rstrip("\n")
            room = _KNOWLEDGE_REPLY_LIMIT - len(_UNIVERSAL_CORE) - len(_dg) - 2
            room = max(room, _contract_block_end(body, 0))
            body = _cap_knowledge_reply(body, topic, solver, physics, signal, limit=max(8000, room))
            return body + "\n\n" + _dg + _UNIVERSAL_CORE
        body = _cap_knowledge_reply(out, topic, solver, physics, signal,
                                    limit=_KNOWLEDGE_REPLY_LIMIT - len(_UNIVERSAL_CORE))
        return body + _UNIVERSAL_CORE

    # ═══════════════════════════════════════════════════════════
    # 2. DISCOVER (replaces 6 discovery tools)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def discover(query: str = "list", solver: str = "") -> str:
        """Discover available solvers and their capabilities.

        Args:
            query: What to discover. Options:
                - "list" — list all solvers with status
                - "physics" — list all physics types per solver
                - "capabilities" — full capabilities matrix
                - "recommend" — recommend solver for a physics (set solver= to physics name)
                - "coupling" — how to couple two codes together: which tool,
                  which backend can take which side, and the exact knowledge
                  calls that return each side's participant contract (handshake,
                  sign convention, flux recovery, exports schema; the solve is
                  yours to write)
            solver: Filter by solver name, or physics name for "recommend"
        """
        if query == "list":
            # Show ALL registered backends, not only the
            # installed ones. The MCP server instructions
            # advertise 8 backends; if discover('list') hides
            # the ones the user has not installed yet, an LLM
            # asking for (say) DUNE-fem or FEBio gets no entry,
            # no status, and no install hint — total dead end.
            # Surface every backend with its actual availability
            # status and the install hint that
            # check_availability() returns. (Audit 2026-06-02.)
            lines = []
            unavailable = []
            for b in all_backends():
                status, msg = b.check_availability()
                core = (f"- **{b.display_name()}** ({b.name()}): "
                        f"{status.value} — "
                        f"{b.input_format().value} input")
                if msg:
                    # Inline a ONE-LINE reason/hint (not a raw traceback) so the
                    # LLM does not have to call a second tool and the list stays
                    # readable.
                    #
                    # This used to fire only for UNAVAILABLE backends, so the one
                    # line that says WHERE an available backend lives — the
                    # interpreter or binary path check_availability() returns —
                    # was thrown away. Coupling needs exactly that: `couple`
                    # takes a `command` argv, and the coupling knowledge
                    # deliberately ships no host paths and sends the agent here
                    # to resolve them. Without the location this list is a dead
                    # end for the first argument of the first participant.
                    core += f"\n  *{_short_reason(msg)}*"
                    unavailable.append(b.name())
                lines.append(core)
            if not lines:
                return "No backends registered."
            if unavailable:
                # A one-line reason is rarely enough to fix an install, and
                # the truncated ImportError it comes from can actively
                # mislead — Kratos's own message blames LD_LIBRARY_PATH for
                # a glibc mismatch no path can fix. Point at the surface
                # that carries the real diagnosis, by name, so the agent
                # does not have to guess a topic string.
                lines.append(
                    "\nNot every backend above is usable. For the ones marked "
                    "otherwise, call knowledge(topic='install', solver='"
                    + unavailable[0] + "') — it gives the install route that "
                    "works, the exact first-run error messages, and which "
                    "environment variables are checked versus trusted blindly.")
            return "\n".join(lines)

        elif query == "physics":
            # Show physics for ALL registered backends (same
            # rationale as discover('list')) so an LLM can
            # learn what dune-fem or febio offer even before
            # installing them. Tag unavailable backends with
            # their status so the LLM does not call
            # run_simulation against a backend that will
            # error out on availability. (Audit 2026-06-02.)
            lines = []
            backends = [get_backend(solver)] if solver else all_backends()
            backends = [b for b in backends if b]
            for b in backends:
                status, _ = b.check_availability()
                tag = "" if status.value == "available" else f" *[{status.value}]*"
                lines.append(f"## {b.display_name()}{tag}")
                for p in b.supported_physics():
                    lines.append(f"- **{p.name}**: {p.description} (variants: {', '.join(p.template_variants)})")
                lines.append("")
            return "\n".join(lines)

        elif query == "capabilities":
            # Show ALL registered backends (see discover('list')
            # rationale above) so an LLM sees the full
            # capabilities matrix including not-yet-installed
            # backends. (Audit 2026-06-02.)
            lines = ["| Solver | Physics Count | Input | Status |",
                     "|--------|--------------|-------|--------|"]
            for b in all_backends():
                status, _ = b.check_availability()
                lines.append(f"| {b.display_name()} | {len(b.supported_physics())} | {b.input_format().value} | {status.value} |")
            return "\n".join(lines)

        elif query == "coupling":
            # Coupling had NO discover branch at all, so an agent could only
            # reach the coupling knowledge by already knowing the topic string
            # existed. That is a dead end for the multi-code capability that is
            # the reason this server has more than one backend.
            from tools.coupling_knowledge import coupling_sides_table
            lines = [
                "# Cross-code coupling on this install",
                "",
                "Two tools, and one deprecated one:",
                "- `couple(participants, max_iter, tol, accelerator, theta)` — THE "
                "general partitioned coupling. You write one solver script per "
                "subdomain; openPASO iterates, relaxes, checks convergence and "
                "conservation. Use this for any coupling.",
                "- `couple_precice(participants, data, exchanges, work_dir, scheme)` "
                "— the preCICE path, when each side is a real preCICE participant.",
                "- `coupled_solve(problem, solver_a, solver_b)` — DEPRECATED, fixed "
                "toy geometries only.",
                "",
                "READ THIS FIRST — the participant contract, the InterfaceData "
                "shapes, relaxation and the flux sign convention:",
                "    knowledge(topic='coupling')",
                "",
                "Then each side's participant CONTRACT (handshake, sign convention, "
                "flux recovery, exports schema; the solve is yours to write):",
                "    knowledge(topic='coupling', solver='<backend>')",
                "    knowledge(topic='precice',  solver='<backend>')   # preCICE path",
                "    knowledge(topic='tsi')                            # 4C-native TSI",
                "",
                coupling_sides_table(),
            ]
            avail = {b.name(): b.check_availability()[0].value
                     for b in all_backends()}
            lines += ["", "Availability on THIS install:"]
            lines += [f"- {n}: {s}" for n, s in sorted(avail.items())]
            # THE deal.II TREE TO BUILD AGAINST, NAMED. A deal.II side is a program the
            # agent compiles, and this line used to drop the tree check_availability
            # found: measured, two coupled sides compiled against a system /usr deal.II
            # 9.1.1 and never built (one lost 20 min).
            if avail.get("dealii") == "available":
                try:
                    from backends.dealii.backend import build_tree_note
                    _note = build_tree_note()
                except Exception:                              # noqa: BLE001
                    _note = ""
                if _note:
                    lines = [ln + f" -- {_note}" if ln.startswith("- dealii: ") else ln for ln in lines]
            return "\n".join(lines)

        elif query == "recommend":
            physics = solver  # in this case solver param holds the physics name
            # Empty / whitespace-only physics matches every
            # backend's first physics (substring-of-everything).
            # Same class of bug as the empty-physics
            # prepare_simulation match — reject it explicitly so
            # the LLM gets a clear usage hint instead of a fake
            # "all backends recommend this" result. Audit
            # 2026-06-01.
            if not physics or not physics.strip():
                return ("Empty physics name. Pass the physics "
                        "as the 'solver' parameter, e.g. "
                        "discover(query='recommend', "
                        "solver='poisson').")
            # Route the physics query through the canonical
            # fuzzy resolver per backend so short shorthands
            # ('ns', 'em', 'cfd', ...) hit the synonym map
            # before a loose substring scan. The raw
            # substring-on-name-or-description recommendation
            # silently matched 'ns' to heat in fenics; 'em' to
            # eigenvalue in fenics; 'pd' to nonlinear_pde in
            # fenics. (Audit 2026-06-02; same drift class as
            # the prepare_simulation fix.)
            #
            # Iterate ALL registered backends, not just the
            # installed ones, so the recommendation includes
            # backends the user has not installed yet. Tag
            # unavailable backends inline so the LLM knows
            # they need an install step. (Audit 2026-06-02;
            # same hide-unavailable bug as discover('list').)
            results = []
            for b in all_backends():
                matched = _fuzzy_match_physics(b, physics)
                if not matched:
                    continue
                for p in b.supported_physics():
                    if p.name == matched:
                        status, _ = b.check_availability()
                        tag = "" if status.value == "available" else f" *[{status.value}]*"
                        results.append(
                            f"- **{b.display_name()}**{tag}: {p.description}")
                        break
            return "\n".join(results) if results else f"No solver found for '{physics}'"

        return ("Usage: discover(query='list'|'physics'|'capabilities'|"
                "'recommend'|'coupling', solver='')")

    # ═══════════════════════════════════════════════════════════
    # 3. EXAMPLES (replaces 7 example/search tools)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def examples(keyword: str = "", solver: str = "fourc", action: str = "search",
                 max_results: int = 3, variant: str = "") -> str:
        """Find and retrieve example input files from solver test suites.

        IMPORTANT: Always call this before writing new input files to study
        real, validated configurations.

        Args:
            keyword: Search term (e.g. 'peridynamic', 'fsi', 'poisson', 'heat')
            solver: Backend name (default: 'fourc')
            action: What to do. Options:
                - "search" — find matching test files with content preview
                - "template" — get a generated template for this physics
                - "tutorials" — list available tutorials
            max_results: Maximum results (default 3)
            variant: For action="template": which of the physics' template
                variants to generate (e.g. '2d', '3d'). action="tutorials"
                lists them. With a variant and no keyword, the physics that
                has that variant is used.
        """
        if action == "search":
            # Empty / whitespace-only keyword matches every
            # filename (substring-of-everything) and silently
            # returns the first few random files in the test
            # tree. Surface a usage hint instead. Audit
            # 2026-06-01 (mirror of the empty-physics fix in
            # prepare_simulation).
            if not keyword or not keyword.strip():
                return ("Empty keyword. Provide a substring "
                        "to match, e.g. 'poisson', 'fluid', "
                        "'contact'.")
            from tools.examples_search import register_example_tools
            # Shared discovery with prepare_simulation —
            # discover_test_dirs returns local demo paths for all
            # backends; resolve_search_keywords applies the same
            # alias map (ngsolve hyperelasticity -> nonlin,
            # fenics navier_stokes -> navier-stokes, ...) so the
            # two LLM-facing tools surface the same content for
            # the same (solver, keyword) pair. Audit 2026-06-01.
            from tools.knowledge import (discover_test_dirs,
                                         resolve_search_keywords)
            results = []
            test_dirs = discover_test_dirs()
            solver_key = solver.lower()
            test_dir = test_dirs.get(solver_key)
            ext = "*.4C.yaml" if solver_key in ("fourc", "4c") else "*.cc" if solver_key == "dealii" else "*.py"

            if test_dir and test_dir.is_dir():
                # Apply solver-aware aliases on top of the raw
                # keyword (the LLM may have used the catalog
                # physics name verbatim, e.g. 'hyperelasticity'
                # — which doesn't match NGSolve's nonlin.py
                # demo without aliasing).
                kw_candidates = list(dict.fromkeys(
                    [keyword] + resolve_search_keywords(solver, keyword)))
                seen: set = set()
                for kw in kw_candidates:
                    for f in sorted(test_dir.rglob(ext)):
                        if kw.lower() not in f.name.lower():
                            continue
                        if f in seen:
                            continue
                        seen.add(f)
                        try:
                            content = f.read_text()[:5000]
                            rel = f.relative_to(test_dir)
                            results.append(f"### `{rel}`\n```\n{content}\n```\n")
                        except Exception:
                            pass
                        if len(results) >= max_results:
                            break
                    if len(results) >= max_results:
                        break

            # Also search templates. Route the keyword through the
            # canonical fuzzy resolver so short shorthands ('ns',
            # 'em', 'cfd', ...) resolve to the right physics via
            # the synonym map BEFORE a loose substring scan. The
            # old code did a raw substring match on name OR
            # description; keyword='ns' matched heat /
            # thermal_structural / reaction_diffusion /
            # multiphase / time_dependent_heat (all contain "ns"
            # somewhere) — five wrong templates and never a
            # navier_stokes one. (Audit 2026-06-02; same drift
            # class as the prepare_simulation fix.)
            #
            # Same 12000-char limit as prepare_simulation — the
            # harder Layer F templates (ngsolve hdivdiv /
            # nonlinear_elasticity, fenics navier_stokes) exceed
            # 3000 chars and lose their solver/output blocks if
            # truncated lower. (Audit 2026-06-01.)
            backend = get_backend(solver)
            if backend:
                EX_TEMPLATE_LIMIT = 12000
                matched = _fuzzy_match_physics(backend, keyword)
                for p in backend.supported_physics():
                    if p.name == matched:
                        _sel, _note = _select_template_variant(
                            keyword, list(p.template_variants))
                        for v in ([_sel] if _sel else []):
                            try:
                                content = backend.generate_input(p.name, v, {})
                                truncated = len(content) > EX_TEMPLATE_LIMIT
                                body = content[:EX_TEMPLATE_LIMIT]
                                suffix = (f"\n... [truncated {len(content) - EX_TEMPLATE_LIMIT} chars]"
                                          if truncated else "")
                                _n = f"{_note}\n\n" if _note else ""
                                results.append(f"### Template: `{p.name}/{v}`\n{_n}```\n{body}{suffix}\n```\n")
                            except Exception as exc:
                                # Same rationale as the
                                # prepare_simulation generator-
                                # failure surfacing: a silent
                                # except: pass made
                                # examples('search') return a
                                # "no template, no error" reply
                                # for any catalog regression.
                                # Now the failure is visible.
                                # (Audit 2026-06-02.)
                                results.append(
                                    f"### Template: `{p.name}/{v}`\n"
                                    f"⚠ Template generation FAILED: "
                                    f"`{type(exc).__name__}: {exc}`\n")
                        break

            if not results:
                # A COUPLED REQUEST MUST NOT DEAD-END HERE.
                #
                # This tool searches ONE backend's test suite by filename, so a
                # coupled example cannot exist in it by construction: a
                # partitioned run is two codes and a driver, not a file in
                # either code's tests. Measured: every coupling keyword —
                # `coupling`, `coupled`, `partitioned`, `dirichlet_neumann`,
                # and `fsi`, the flagship — returned "No examples found" for
                # both fourc and fenics, with no hint that the material exists
                # elsewhere. The docstring meanwhile tells the agent to ALWAYS
                # call this before writing input files.
                #
                # That is the worst place to be silent: coupled problems are
                # where an agent most needs a worked example, and the measured
                # failure of runs with these tools is running out of tool calls
                # (median 39 against 95 without them). A dead end costs a call and
                # returns nothing.
                if _is_coupling_request(keyword):
                    return _coupling_example_pointer(keyword, solver)
                return (f"No examples found for '{keyword}' in {solver}. This "
                        f"tool matches FILENAMES in {solver}'s own test suite "
                        f"and generated templates; if the physics is coupled "
                        f"(two codes exchanging an interface field), it cannot "
                        f"appear here — call "
                        f"`knowledge(topic=\"coupling\")` instead.")
            body = (f"## {len(results)} example(s) for '{keyword}' from "
                    f"{solver}\n\n" + "\n---\n".join(results))
            # AND THE FOUND-SOMETHING CASE IS THE DANGEROUS ONE.
            #
            # The branch above fires only when the filename search comes back
            # empty, which is true for fenics and false for fourc: 4C's test
            # suite matches every coupling keyword — `fsi` returns
            # fsi_dc3D_part_ait_ga_ost_xwall.4C.yaml, `partitioned` returns the
            # elch multiscale decks, `coupled` returns elch_onewaycoupled_ost.
            # Every one of them solves the whole problem inside 4C. They are
            # single-code multiphysics, not the two-code partitioned run with
            # exports.json/imports.json that the task asks for.
            #
            # So the guard existed and could not reach the case it was built
            # for, and the case it could not reach is the worse one: nothing is
            # more likely to stop an agent looking than a plausible answer. The
            # pointer is therefore appended whenever the REQUEST is about
            # coupling, found files or not.
            if _is_coupling_request(keyword):
                body += ("\n\n---\n⚠ READ THIS BEFORE COPYING THE ABOVE.\n"
                         + _coupling_example_pointer(keyword, solver))
            return body

        elif action == "template":
            backend = get_backend(solver)
            if not backend:
                return f"Unknown solver: {solver}"
            if not keyword or not keyword.strip():
                # A VARIANT NAMES ITS PHYSICS. Measured 2026-09-28: a run asked for
                # examples(action="template", solver="fenics", variant=...) with no
                # keyword; the keyword was a required argument, so the call died in
                # argument validation with "Field required" and the run lost a step.
                owners = [p.name for p in backend.supported_physics()
                          if variant and variant in p.template_variants]
                if len(owners) == 1:
                    keyword = owners[0]
                else:
                    listing = "; ".join(
                        f"{p.name} ({', '.join(p.template_variants) or 'no variants'})"
                        for p in backend.supported_physics())
                    head = (f"Variant `{variant}` belongs to several physics in {solver}: "
                            f"{', '.join(owners)}. Name one as keyword."
                            if len(owners) > 1 else
                            f"No physics in {solver} has a variant `{variant}`."
                            if variant else
                            "Empty keyword. Name the physics as keyword, e.g. keyword='navier_stokes'.")
                    return f"{head} Physics and their variants in {solver}: {listing}"
            # Route the keyword through the canonical fuzzy
            # resolver so short shorthands ('ns' -> navier_stokes,
            # 'em' -> maxwell, ...) route via the synonym map
            # first. The raw substring path here matched 'ns'
            # against 'transient_heat' and never against
            # 'navier_stokes' (no adjacent 'ns' substring in
            # 'navier_stokes' itself). (Audit 2026-06-02.)
            matched = _fuzzy_match_physics(backend, keyword)
            for p in backend.supported_physics():
                if p.name == matched:
                    avail = list(p.template_variants)
                    if variant and variant not in avail:
                        return (f"No variant `{variant}` for "
                                f"`{matched}` in {solver}. Available: "
                                + (", ".join(avail) if avail else "none"))
                    if variant:
                        chosen, note = variant, ""
                    else:
                        chosen, note = _select_template_variant(keyword, avail)
                        chosen = chosen or "2d"
                    try:
                        content = backend.generate_input(p.name, chosen, {})
                        fmt = detect_template_language(content, backend.input_format().value)
                        head = f"Template `{matched}/{chosen}`. {note}\n\n" if note else ""
                        return f"{head}```{fmt}\n{content}\n```"
                    except Exception as e:
                        return f"Error generating template: {e}"
            return f"No template for '{keyword}' in {solver}"

        elif action == "tutorials":
            backend = get_backend(solver)
            if not backend:
                return f"Unknown solver: {solver}"
            lines = [f"## {backend.display_name()} Templates\n"]
            for p in backend.supported_physics():
                lines.append(f"- **{p.name}**: {', '.join(p.template_variants)} — {p.description}")
            return "\n".join(lines)

        return ("Usage: examples(keyword, solver, action='search'|'template'|'tutorials', "
                "variant='...' for a template)")

    # ═══════════════════════════════════════════════════════════
    # 4. SIMULATE (replaces run_simulation + run_with_generator)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    async def submit_critic_review(solver: str, findings: str,
                                   setup: str = "", coupling_args: str = "",
                                   ttl_s: float = 3600.0) -> str:
        """Put an independent critic's review of a setup ON RECORD, so a run of
        that setup can be verified.

        REQUIRED, and the most common mistake: pass EXACTLY ONE of `setup` or
        `coupling_args`. `setup` is the deck text for run_simulation /
        run_with_generator / verify_mesh_independence; `coupling_args` is a
        JSON object for `couple` / `couple_precice` / `coupled_solve`. Passing
        neither — or both — is refused, and the refusal comes AFTER you have
        written the review, so the review is wasted. Measured over the
        recorded runs, half of all reviews handed in were rejected this way.

        openPASO's critic requirement is enforced, not requested. The run and
        coupling tools do not take your word for it: they look up whether THIS
        server holds a review of the EXACT setup being executed. Passing
        critic_approved=True without a matching review here leaves the result
        NOT VERIFIED, whatever else the run does.

        The workflow is: the agent doing the work spawns a sub-agent as an
        independent critic and gives it the file to review; the critic
        challenges the parameters, units, discretisation, problem statement and
        boundary conditions, cross-checks against literature and benchmarks,
        and CALLS THIS TOOL ITSELF with its verdict; the working agent then runs
        exactly the reviewed file. A review filed by the agent whose work it
        reviews is not a review, and an interface that can see who filed it
        refuses one.

        The review is bound to the setup by digest, so a setup edited after
        review no longer matches and must be reviewed again. That is deliberate:
        reviewing a clean deck and running a different one is the obvious way to
        defeat a critic requirement, and it is the route this closes.

        This server cannot judge whether a critique was any GOOD — it is not an
        oracle for review quality. It enforces that a substantive review of this
        setup exists and is auditable, and refuses a review too short to have
        said anything.

        Args:
            solver: the backend the run will use. For `couple` pass "couple",
                for `couple_precice` pass "couple_precice", and for the legacy
                `coupled_solve` pass "<solver_a>-><solver_b>".
            findings: what the critic actually checked and concluded. Substance
                is required; an empty approval is indistinguishable from no
                review and is refused.
            setup: for run_simulation / run_with_generator /
                verify_mesh_independence — the EXACT deck text that will run
                (input_content, generator_script, or input_template), or the
                NAME of the file in the run folder that holds it: the review is
                then bound to that file's text as it is now, and a run of that
                file (input_path / generator_path) matches it.
            coupling_args: for the coupling tools instead of `setup` — a JSON
                object of the arguments you will pass. Keys per tool:
                coupled_solve: problem, solver_a, solver_b, nx, ny, max_iter,
                tol, relaxation, params; couple AND couple_levels: participants,
                max_iter, tol, accelerator, theta, monolithic, probe -- the SAME
                seven for both, and for `couple_levels` you pass the values you
                will give it, NOT the per-level meshes, because one review
                covers the whole ladder; couple_precice:
                participants, data, exchanges, scheme, dimensions, max_time,
                time_window, max_iterations, convergence_tol, relaxation,
                mapping. Pass EVERY key for the tool you will call, with the
                values you will call it with — a missing or different key is a
                different setup and the run will come back NOT VERIFIED. For
                `couple` the CONTENTS of each participant's script are part of
                the setup too, so the scripts must already be written when the
                review is submitted, and editing one afterwards invalidates it.
            ttl_s: how long the review stays valid (default 1 hour).

        Returns: JSON with a `critic_token`. Passing it to run_simulation or
            run_with_generator makes the review single-use and binds it to that
            job; omitting it still works, since the deck is matched by digest.
        """
        _named_file = ""
        if bool(setup) == bool(coupling_args):
            # Say which mistake was made and what to send instead. The old
            # message stated the rule without saying which side was wrong, so
            # an agent that had passed NEITHER read it as a complaint about
            # passing both, and retried the same way. Your findings text is
            # preserved above — resubmit it unchanged with the argument added.
            both = bool(setup) and bool(coupling_args)
            return json.dumps({
                "accepted": False,
                "error": ("you passed BOTH `setup` and `coupling_args`; send "
                          "only the one that matches the tool you will call"
                          if both else
                          "you passed NEITHER `setup` nor `coupling_args`, so "
                          "there is nothing for the review to bind to"),
                "what_to_do": (
                    "resubmit the SAME findings text with exactly one of: "
                    "setup=<the exact deck/script text you will run> for "
                    "run_simulation, run_with_generator or "
                    "verify_mesh_independence; or coupling_args=<a JSON "
                    "object of the arguments you will pass> for couple, "
                    "couple_precice or coupled_solve."),
                # THIS EXAMPLE USED TO PRODUCE A DIGEST NOTHING COULD MATCH
                # (`monolithic: false` against couple_levels' `""`, and a LIST
                # where the tool takes the JSON string). The setup text is
                # canonicalised now, so all of those spellings hash alike --
                # the example is written the way the tool is actually called.
                "coupling_args_example": (
                    '{"participants": "[{\"name\": \"A\", ...}]", '
                    '"max_iter": 50, "tol": 1e-8, "accelerator": "auto", '
                    '"theta": 0.5, "monolithic": "", "probe": false}'),
                "findings_were_not_lost": True},
                indent=2)
        if coupling_args:
            try:
                parsed = json.loads(coupling_args)
            except json.JSONDecodeError as exc:
                return json.dumps({"accepted": False,
                                   "error": f"coupling_args is not JSON: {exc}"},
                                  indent=2)
            if not isinstance(parsed, dict):
                return json.dumps({
                    "accepted": False,
                    "error": "coupling_args must be a JSON OBJECT of the "
                             "arguments you will pass."}, indent=2)
            # A REVIEW OF A SETUP THAT CANNOT RUN IS WORSE THAN NO REVIEW.
            #
            # `couple` refuses a participant whose work_dir is outside this
            # task's tree. This door did not, so a review could be bound by
            # digest to a setup with an invented path -- and then the run the
            # agent actually makes, with the real path, is a DIFFERENT setup
            # whose digest matches nothing, so it comes back NOT VERIFIED with
            # a perfectly good review on record for a run nobody made.
            #
            # MEASURED on the five coupled runs of 2026-09-19: three of them
            # submitted a review whose participants sat in a sibling of the
            # served interpreter's own directory (.../open-fem-agent/side_A) --
            # a path the model can only have taken from the absolute
            # interpreter path openPASO itself serves. Two of those three then
            # handed in a result with no review matching it at all, and both
            # were correct, which is how "only verified output is handed in"
            # stops being true of the runs while staying true of the design.
            #
            # The same rule, at the earlier door, costs one call instead of the
            # whole verification chain.
            _bad = _coupling_work_dir_faults(parsed)
            _n_dir = len(_bad)
            # A SOLVER BINARY IS NOT A PARTICIPANT HERE EITHER: couple and couple_levels
            # refuse it, and this door accepted a review of it (measured).
            try:
                _specs_rv = json.loads(parsed.get("participants") or "[]") \
                    if isinstance(parsed.get("participants"), str) else (parsed.get("participants") or [])
                _bad += [t for t in (_solver_binary_as_participant(sp) for sp in _specs_rv
                                     if isinstance(sp, dict)) if t]
            except (TypeError, ValueError):
                pass
            if _bad:
                # NAME WHAT WAS REFUSED. Measured: every refusal said "fix the participants'
                # work_dir values first" when the refusal was a solver binary run as a
                # participant (five cells of one round).
                _what = " and ".join(w for w, on in (
                    ("the participants' work_dir values", _n_dir > 0),
                    ("each participant's command (your participant script, run by its "
                     "interpreter -- never the solver binary itself)", len(_bad) > _n_dir)) if on)
                return json.dumps({
                    "accepted": False,
                    "error": "; ".join(_bad),
                    "what_to_do": (
                        f"fix {_what} first, then "
                        "resubmit the SAME findings text. A review is bound to "
                        "the EXACT setup you pass here: reviewing one setup and "
                        "running another leaves the run unverified, whatever the "
                        "review said."),
                    "findings_were_not_lost": True}, indent=2)
            # Through the same single definition the run tools use, so the
            # participant-script fingerprints are part of both digests.
            setup_text = _coupling_setup_text(**parsed)
        else:
            _named_text, _named_err = _text_of_named_file(setup)
            if _named_err:
                return json.dumps({"accepted": False, "error": _named_err,
                                   "findings_were_not_lost": True}, indent=2)
            setup_text = setup if _named_text is None else _named_text
            if _named_text is not None:
                _named_file = setup.strip()
        # A REVIEW THAT REJECTS THE SETUP IS NOT A REVIEW OF RECORD FOR IT. Measured
        # on one set of coupled runs: two orchestrators filed their critics' verdicts
        # "REJECTED" and "NOT APPROVED", both were accepted, and both runs used the
        # token to verify what their own critics had turned down. The last verdict
        # stated in the text decides, so a review that approves after fixes passes.
        _verdicts = list(re.finditer(
            r"\b(?:VERDICT|DECISION|CONCLUSION|RESULT)\b[^\n\w]{0,6}\**\s*"
            r"(NOT[\s_-]+APPROVED|REJECTED|REJECT|DO[\s_-]+NOT[\s_-]+(?:RUN|PROCEED)|APPROVED)",
            str(findings), re.I))
        if _verdicts and not _verdicts[-1].group(1).upper().startswith("APPROVED"):
            # ...and it is KEPT: a rejection that left no trace let the working agent ask one
            # critic after another until one approved the same text (named by both peer
            # sessions, 2026-10-01). It issues no token and blocks that exact text.
            try:
                _CRITIC_REGISTRY.submit_review(
                    solver=solver, findings=findings,
                    digest=review_digest(solver, setup_text), ttl_s=ttl_s,
                    setup_text=setup_text, rejected=True, source_name=_named_file)
            except CriticGateError as exc:
                return json.dumps({"accepted": False, "error": str(exc),
                                   "findings_were_not_lost": True}, indent=2)
            return json.dumps({
                "accepted": False,
                "recorded": "rejection",
                "error": (f"THE REVIEW YOU FILED SAYS \"{_verdicts[-1].group(0).strip()}\": a setup its "
                          f"own critic turned down is not a reviewed setup, and no token is issued "
                          f"for it. Fix what the critic named, have it review the fixed setup, "
                          f"and file that review."),
                "note": ("The rejection is on record for this exact text: a run of it stays NOT "
                         "VERIFIED until the text changes and a critic approves the changed "
                         "text. Asking another critic about the same text does not undo it."),
                "findings_were_not_lost": True}, indent=2)
        try:
            rec = _CRITIC_REGISTRY.submit_review(
                solver=solver, findings=findings,
                digest=review_digest(solver, setup_text),
                ttl_s=ttl_s, setup_text=setup_text, source_name=_named_file)
        except CriticGateError as exc:
            return json.dumps({"accepted": False, "error": str(exc)}, indent=2)
        _get_journal().record("critic_review", "submit_critic_review",
                              solver=solver,
                              input_snapshot=_make_input_snapshot(
                                  setup_text, solver, {"type": "critic_review"}))
        # A SETUP WHOSE NAMED FILES DO NOT EXIST IS ON RECORD AS WRITTEN, AND SAID.
        # Measured: a review naming work_dir "$(pwd)/side_A" was accepted as "on
        # record for this exact setup" while it had found none of its scripts.
        _missing = []
        try:
            _fp = json.loads(setup_text).get("__participant_files__") or {}
            _missing = sorted(f"{_n}/{_f}" for _n, _files in _fp.items()
                              for _f, _h in (_files or {}).items() if _h == "absent")
        except (ValueError, TypeError, AttributeError):
            _missing = []
        reply = {
            "accepted": True,
            "critic_token": rec.token,
            "solver": solver,
            "valid_for_s": ttl_s,
            "note": ("This review is on record for this exact setup. Editing "
                     "the setup invalidates it. Pass critic_token to the run "
                     "tool to make the review single-use and bound to that "
                     "job."
                     + (f" NO FILE EXISTS AT {', '.join(_missing[:3])} as this setup names "
                        f"it: the review is of a setup whose scripts it never saw, and a "
                        f"run with the real paths is a different setup. A shell "
                        f"expression such as $(pwd) is not expanded here; name the "
                        f"absolute path." if _missing else "")),
        }
        if _named_file:
            reply["bound_to"] = {"file": _named_file, "characters": len(setup_text),
                                 "lines": len(setup_text.splitlines())}
            reply["note"] = (f"This review is on record for the text of {_named_file} as it is "
                             f"now ({len(setup_text):,} characters). Run that file unchanged -- "
                             f"run_simulation(solver=..., input_path='{_named_file}') or "
                             f"run_with_generator(solver=..., generator_path='{_named_file}') -- "
                             f"and the run matches it. Editing the file voids the review. Only "
                             f"this file's text is covered: files it reads when it runs (a mesh, a "
                             f"module it imports) are not.")
        return json.dumps(reply, indent=2)

    @mcp.tool()
    async def run_with_generator(solver: str, generator_script: str = "",
                                  generator_path: str = "",
                                  job_name: str = "", np: int = 1,
                                  critic_approved: bool = False,
                                  critic_token: str = "",
                                  verify_pde: str = "",
                                  ctx: Context = None) -> str:
        """Run a generator script that creates an input file, then execute the solver.

        Use this for solvers that need a COMPILED binary or separate input files:
        - 4C: generator creates .4C.yaml + mesh, then 4C binary runs on them
        - deal.II: generator creates main.cpp, then cmake + make + ./fem_solve
        - Kratos (with real binary): generator creates ProjectParameters.json +
          .mdpa + MainKratos.py, then Kratos Python runs MainKratos.py

        DO NOT use this for:
        - FEniCS, NGSolve, scikit-fem, DUNE-fem: use run_simulation() instead
        - Kratos manual-assembly scripts (numpy/scipy): use run_simulation()
          since those are standalone Python scripts, not input-file generators

        The generator script runs in the server's Python. It must produce an
        input file matching one of: *.4C.yaml, *.yaml, input.*, solve.py,
        MainKratos.py

        Args:
            solver: Backend name (fourc, dealii, kratos)
            generator_script: Python script that creates the input file
            generator_path: instead of generator_script, the name of a file in
                the run folder holding it. Use it to run a generator the critic
                reviewed: what runs is then exactly the reviewed text.
            job_name: Optional job directory name
            np: MPI processes (default 1)
            critic_approved: recorded, not trusted. The result is verified only
                if a critic review of THIS generator_script is on record — call
                submit_critic_review first.
            critic_token: optional token from submit_critic_review; makes the
                review single-use and binds it to this job.
            verify_pde: optional JSON declaring the problem being solved, so
                openPASO can check the result actually SATISFIES it rather than
                merely looking well-formed. Example:
                {"operator": "diffusion",
                 "source": "2*pi**2*sin(pi*x)*sin(pi*y)",
                 "coefficient": "1.0", "dim": 2, "domain_measure": 1.0}
                `source` and `coefficient` are numeric expressions in x, y, z.
                A field that does not satisfy the declared equations is NOT
                VERIFIED, whatever else the run did. Currently covers scalar
                diffusion on simplex meshes; anything else is reported as not
                checked, never as passed.
        """
        import subprocess
        import sys

        generator_script, _in_err = _run_text(generator_script, generator_path,
                                              "generator_script", "generator_path")
        if _in_err:
            return json.dumps({"status": "failed", "phase": "input", "error": _in_err,
                               "trustworthy_result": False}, indent=2)
        _journal = _get_journal()
        _snap = _make_input_snapshot(generator_script, solver, {"type": "generator"})
        _journal.record("tool_call", "run_with_generator", solver=solver,
                        input_snapshot=_snap)

        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        status, msg = backend.check_availability()
        if status.value != "available":
            _journal.record("tool_error", "run_with_generator", solver=solver,
                            error_message=f"Not available: {msg}",
                            input_snapshot=_snap)
            return f"Solver {solver} not available: {_short_reason(msg)}"

        # A PYTHON SOLVER'S SCRIPT IS ITS OWN INPUT. Measured 2026-09-29 in the web interface:
        # a model passed a complete FEniCSx script here. It ran in openPASO's own Python, which
        # has no dolfinx, the reply said "No module named 'dolfinx'", and the model concluded
        # FEniCSx was not installed and left a working script for another solver. Such a
        # script is refused before it runs, with the call that runs it.
        from core.backend import InputFormat
        _own = sorted(m for m, b in _solver_modules_in(generator_script).items() if b == solver)
        if backend.input_format() == InputFormat.PYTHON and _own:
            _msg = (f"This script imports {', '.join(_own)}: it is the {solver} input itself, not "
                    f"a generator. run_with_generator runs its script in openPASO's own Python, "
                    f"which is not {solver}'s, so it stops with \"No module named '{_own[0]}'\" "
                    f"although {solver} is installed. Pass the same script to "
                    f"run_simulation(solver='{solver}', input_content=<this script>): that runs "
                    f"it in {solver}'s own interpreter.")
            _journal.record("tool_error", "run_with_generator", solver=solver,
                            error_message=_msg[:300], input_snapshot=_snap)
            return json.dumps({"status": "failed", "phase": "routing", "error": _msg,
                               "next_step": f"run_simulation(solver='{solver}', "
                                            f"input_content=<this script>)"}, indent=2)

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = job_name or f"{solver}_gen_{ts}"
        work_dir = _OUTPUT_DIR / name
        work_dir.mkdir(parents=True, exist_ok=True)

        gen_path = work_dir / "generate_input.py"
        gen_path.write_text(generator_script)

        python = sys.executable
        gen_result = subprocess.run(
            [python, str(gen_path)],
            capture_output=True, text=True,
            cwd=str(work_dir),
            stdin=subprocess.DEVNULL,
        )

        if gen_result.returncode != 0:
            _journal.record("tool_error", "run_with_generator", solver=solver,
                            error_message=f"Generator failed: {gen_result.stderr[-200:]}",
                            input_snapshot=_snap)
            _failed = {
                "status": "failed", "phase": "generator",
                "error": gen_result.stderr[-500:],
                "work_dir": str(work_dir),
            }
            _missing = __import__("re").search(r"No module named '([A-Za-z_]\w*)",
                                               gen_result.stderr or "")
            if _missing and _missing.group(1) in _solver_module_owners():
                # the generator, not the solver, stopped: it ran in openPASO's own Python
                _failed["what_the_error_is"] = (
                    f"the GENERATOR stopped, in openPASO's own Python ({python}), which does not "
                    f"have {_missing.group(1)}; the solver never ran. A generator only writes the "
                    f"input files and needs no solver import. A script that is itself a solve in "
                    f"Python goes to run_simulation, which runs it in that solver's interpreter.")
            else:
                _named = _what_the_error_is(gen_result.stderr, solver)
                if _named:
                    _failed["what_the_error_is"] = _named
            return json.dumps(_failed, indent=2)

        from core.backend import find_generated_input
        input_file = find_generated_input(work_dir, backend)

        if not input_file:
            _journal.record("tool_error", "run_with_generator", solver=solver,
                            error_message="Generator did not produce an input file",
                            input_snapshot=_snap)
            return json.dumps({
                "status": "failed", "phase": "generator",
                "error": "Generator did not produce an input file",
                "work_dir": str(work_dir),
                "files": sorted(f.name for f in work_dir.iterdir())[:50],
            }, indent=2)

        input_content = input_file.read_text()
        # Update snapshot with the generated input's shape
        _snap_run = _make_input_snapshot(input_content, solver,
                                         {"type": "generated_input", "input_file": input_file.name})
        from core.backend import JobHandle
        run_coro = backend.run(input_content, work_dir, np=np, timeout=None)
        if ctx is not None:
            job = await _run_with_progress(ctx, run_coro, f"Running {solver}")
        else:
            job = await run_coro
        _jobs[job.job_id] = job

        if job.error:
            _journal.record("tool_error", "run_with_generator", solver=solver,
                            error_message=error_excerpt(job.error, 300),
                            input_snapshot=_snap_run)
        else:
            _journal.record("tool_success", "run_with_generator", solver=solver,
                            input_snapshot=_snap_run)

        result = {
            "job_id": job.job_id, "solver": solver,
            "status": job.status, "work_dir": str(job.work_dir),
            "elapsed": f"{job.elapsed:.2f}s" if job.elapsed else None,
            "input_file": input_file.name,
        }
        out_files = []
        # THE PROGRAM'S RETURN CODE, ALWAYS, AND A SIGNAL BY NAME. Measured: a run gave up
        # 0.4 min after a bare "status: failed" from a program a segfault had killed with
        # an empty stderr (a Release deal.II asserts nothing) -- a crash in its own code,
        # read as an install fault.
        if job.return_code is not None:
            result["return_code"] = job.return_code
            _sig = -job.return_code if job.return_code < 0 else (
                job.return_code - 128 if 128 < job.return_code < 160 else 0)
            if _sig:
                import signal as _signal_mod
                try:
                    result["killed_by"] = (f"{_signal_mod.Signals(_sig).name}: a crash inside your "
                                           f"program, not an install fault")
                except ValueError:
                    pass
        if job.error:
            result["error"] = error_excerpt(job.error)
            _named = _what_the_error_is(job.error, solver)
            if _named:
                result["what_the_error_is"] = _named
        nonfinite = []
        _empty = []
        _stdout_text = ""
        if job.status == "completed":
            out_files = backend.get_result_files(job)
            result["output_files"] = [f.name for f in out_files]
            stdout_log = work_dir / "stdout.log"
            if stdout_log.exists():
                _stdout_text = stdout_log.read_text()
                result["stdout_tail"] = (_stdout_text[-2000:]
                                         if len(_stdout_text) > 2000 else _stdout_text)
            # Attestation: a process that exits 0 but produces NO output files is
            # NOT a verified solve — the canonical silent failure. Flag it loudly.
            if not out_files:
                result["status"] = "completed_unverified"
            else:
                # ... and a file full of NaN/Inf is a fabricated-looking result.
                nonfinite = check_result_files_finite(out_files)
            # Also scan the headline numbers (results_summary.json + stdout).
            nonfinite += check_summary_finite(work_dir, _stdout_text)
            # Structural defects in the solver's own data output. The scans
            # above look at headline numbers and mesh files; a FIELD that is
            # mostly NaN, or a wholly degenerate mesh, passed both and was
            # stamped verified. (Anti-fabrication gate.)
            nonfinite += inspect_result_artefacts(out_files)
            if nonfinite:
                result.setdefault("validation", []).extend(nonfinite)
            # WHAT THE RESULT HOLDS. A result that is exactly zero everywhere is not a result;
            # a field that never changed in time, and the mesh the result lives on (cells,
            # extent, area), are stated beside it. Measured 2026-09-29: a run on a 95-cell mesh
            # of the obstacle's disc wrote zero velocity and pressure, and the gate said "the
            # automated checks passed" (core/fabrication_gate.result_content_findings).
            _empty, _notes, _mesh = result_content_findings(out_files)
            if _mesh:
                result["result_mesh"] = _mesh
            if _notes:
                result["result_notes"] = _notes
            if _empty:
                result.setdefault("validation", []).extend(_empty)
            # The 'finiteness not asserted' honesty note is a coverage gap,
            # not evidence of a bad number — it must not flip the verdict to
            # 'non-finite values' (FEBio .xplt / .bp-without-adios2 runs).
            nonfinite = [x for x in nonfinite
                         if not x.startswith("finiteness not asserted")]
            # openPASO computes the run's headline numbers from the run's own
            # data, so the agent never has to assert one of its own.
            if out_files:
                result["openpaso_computed"] = _attest_run_quantities(
                    work_dir, job.job_id)
                # …and, if the run declared what it is solving, whether that
                # data satisfies those equations at all.
                if verify_pde:
                    result["residual_check"] = _check_declared_pde(
                        verify_pde, out_files)
        # Verification gate: bind the verdict to run evidence (attestation).
        if job.error:
            reason = "the solver run errored, so no number is backed by a valid run"
        elif job.status != "completed":
            reason = f"the run did not complete (status={job.status})"
        elif not out_files:
            reason = ("the process exited cleanly but produced NO output files, "
                      "so no reported number is backed by run evidence")
        elif nonfinite:
            reason = ("the result contains non-finite (NaN/Inf) values, so it is "
                      "numerically invalid"
                      if any("non-finite" in x for x in nonfinite)
                      else "a result file is unreadable/corrupt, so the gate "
                           "could not assert the output's integrity")
        elif _empty:
            reason = _empty[0]
        elif _residual_blocks_verification(result):
            reason = ("the field this run produced does NOT satisfy the "
                      "equations it declared: "
                      + str(result["residual_check"].get("detail", "")))
        else:
            reason = ""
        _stamp_verification(result,
                            evidence_ok=(bool(out_files) and not job.error
                                         and not nonfinite and not _empty
                                         and not _residual_blocks_verification(result)),
                            reason=reason, critic_approved=critic_approved,
                            solver=solver, setup_text=generator_script,
                            critic_token=critic_token,
                            job_id=str(result.get("job_id", job_name or "")))
        notice = _unsaved_work_notice(getattr(job, "work_dir", "") or "",
                                      out_files)
        if notice:
            result["nothing_written_down_yet"] = notice
        return json.dumps(result, indent=2)

    @mcp.tool()
    def check_input(solver: str, input_path: str = "", input_content: str = "") -> str:
        """Name the defects of an input deck or script BEFORE it runs: the setup checks the run
        tools apply, as a standalone gate for a code you run yourself (a coupling participant's
        own deck). For 4C every section name is judged by the INSTALLED binary's own grammar
        (`4C -p`) and the closest known names are listed, every condition's E id is checked
        against the topology sections (4C drops a condition on an undefined id silently and
        finishes on the wrong problem), and the measured TSI / Scalar_Transport deck defects
        are named -- all of them in one call, where the binary stops at the first. Reads the
        file, writes nothing; the findings are advisory and name no fix beyond the defect.

        Args:
            solver: backend name (e.g. 'fourc')
            input_path: path of the deck or script to check (or pass input_content)
            input_content: the text itself, when no file exists yet
        """
        _get_journal().record("knowledge_lookup", "check_input", solver=solver, physics=input_path or "inline")
        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"
        text = input_content or ""
        if input_path:
            # A RELATIVE PATH IS THE CALLER'S, as at the other doors (resolve_under_cell). Measured:
            # "No such file" for ./side_B/participant_B.py, which existed -- it was resolved
            # against this server's own working directory.
            from .result_audit import resolve_under_cell
            _in = resolve_under_cell(input_path)
            if not _in.is_absolute():
                return ("check_input: a relative path cannot be resolved: this server does not know your "
                        "working directory. Pass the ABSOLUTE path of the file inside your working directory.")
            try:
                text = _in.read_text(errors="ignore")
            except OSError as e:
                return f"check_input: cannot read {_in}: {e}"
        if not text.strip():
            return "check_input: no input text (pass input_path or input_content)"
        findings = []
        try:
            findings += list(backend.validate_input(text) or [])
        except Exception as e:                          # noqa: BLE001
            findings.append(f"(setup check failed: {e!r})")
        if backend.name() == "fourc":
            from tools.fourc_deck_lint import deck_judgement   # noqa: PLC0415
            findings += deck_judgement(text)
        head = f"CHECK_INPUT ({solver}, {Path(input_path).name if input_path else 'inline text'}): "
        if not findings:
            return head + ("no defect named by the setup checks. That is not proof the deck runs; the "
                           "binary's own console is.")
        return head + f"{len(findings)} finding(s), fix every one before running:\n" + "\n".join(f"- {f}" for f in findings)

    @mcp.tool()
    async def run_simulation(solver: str, input_content: str = "",
                             input_path: str = "",
                             job_name: str = "", np: int = 1,
                             critic_approved: bool = False,
                             critic_token: str = "",
                             verify_pde: str = "",
                             ctx: Context = None) -> str:
        """Run a simulation directly with input content.

        Use this for Python-based solvers (FEniCS, NGSolve, scikit-fem, DUNE-fem)
        where the input IS a Python script. The tool routes through the correct
        Python environment automatically (e.g., conda env for FEniCS).

        For 4C/deal.II/Kratos where a separate input file must be generated
        first, use run_with_generator() instead.

        Args:
            solver: Backend name (best for: fenics, ngsolve, skfem, dune)
            input_content: The input content (Python script / YAML / C++ / XML)
            input_path: instead of input_content, the name of a file in the run
                folder whose text is the input. Use it to run a file the critic
                reviewed: what runs is then exactly the reviewed text.
            job_name: Optional job name
            np: MPI processes
            critic_approved: recorded, not trusted. The result is verified only
                if a critic review of THIS input_content is on record — call
                submit_critic_review first.
            critic_token: optional token from submit_critic_review; makes the
                review single-use and binds it to this job.
            verify_pde: optional JSON declaring the problem being solved, so
                openPASO can check the result actually SATISFIES it rather than
                merely looking well-formed. Example:
                {"operator": "diffusion",
                 "source": "2*pi**2*sin(pi*x)*sin(pi*y)",
                 "coefficient": "1.0", "dim": 2, "domain_measure": 1.0}
                `source` and `coefficient` are numeric expressions in x, y, z.
                A field that does not satisfy the declared equations is NOT
                VERIFIED, whatever else the run did. Currently covers scalar
                diffusion on simplex meshes; anything else is reported as not
                checked, never as passed.
        """
        input_content, _in_err = _run_text(input_content, input_path,
                                           "input_content", "input_path")
        if _in_err:
            return json.dumps({"status": "failed", "phase": "input", "error": _in_err,
                               "trustworthy_result": False}, indent=2)
        _journal = _get_journal()
        _snap = _make_input_snapshot(input_content, solver)
        _journal.record("tool_call", "run_simulation", solver=solver,
                        input_snapshot=_snap)

        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        status, msg = backend.check_availability()
        if status.value != "available":
            _journal.record("tool_error", "run_simulation", solver=solver,
                            error_message=f"Not available: {msg}",
                            input_snapshot=_snap)
            return f"Solver {solver} not available: {_short_reason(msg)}"

        # CP-4: validate the input BEFORE running (was skipped on the live path)
        _input_warnings = []
        try:
            _input_warnings = backend.validate_input(input_content) or []
        except Exception:
            pass

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = job_name or f"{solver}_{ts}"
        work_dir = _OUTPUT_DIR / name
        work_dir.mkdir(parents=True, exist_ok=True)

        run_coro = backend.run(input_content, work_dir, np=np, timeout=None)
        if ctx is not None:
            job = await _run_with_progress(ctx, run_coro, f"Running {solver}")
        else:
            job = await run_coro
        _jobs[job.job_id] = job

        if job.error:
            _journal.record("tool_error", "run_simulation", solver=solver,
                            error_message=error_excerpt(job.error, 300),
                            input_snapshot=_snap)
        else:
            _journal.record("tool_success", "run_simulation", solver=solver,
                            input_snapshot=_snap)

        result = {
            "job_id": job.job_id, "solver": solver,
            "status": job.status, "work_dir": str(job.work_dir),
            "elapsed": f"{job.elapsed:.2f}s" if job.elapsed else None,
        }
        out_files = []
        if job.error:
            result["error"] = error_excerpt(job.error)
            _named = _what_the_error_is(job.error, solver)
            if _named:
                result["what_the_error_is"] = _named
        if _input_warnings:
            result["input_validation_warnings"] = _input_warnings
        nonfinite = []
        _empty = []
        _stdout_text = ""
        if job.status == "completed":
            out_files = backend.get_result_files(job)
            result["output_files"] = [f.name for f in out_files]
            stdout_log = work_dir / "stdout.log"
            if stdout_log.exists():
                _stdout_text = stdout_log.read_text()
                result["stdout_tail"] = (_stdout_text[-2000:]
                                         if len(_stdout_text) > 2000 else _stdout_text)
            # CP-4: a process that exits 0 but produces NO output is NOT a verified
            # success — the canonical silent failure. Downgrade the status loudly.
            if not out_files:
                result["status"] = "completed_unverified"
                result["warning"] = ("Process exited cleanly but produced NO output files "
                                     "— this is NOT a verified solve. Do not treat as a result.")
                # WHERE THE FILES WENT, when the script says so. Measured 2026-09-29: a
                # 36-minute run changed directory with os.chdir, wrote every result there,
                # and was reported as having produced nothing; the reason was one line up.
                if re.search(r"\bos\.chdir\s*\(", input_content or ""):
                    result["warning"] += (
                        " The script changes its working directory (os.chdir), so its files "
                        "were written there, outside this job's folder, which is the only "
                        "place openPASO checks. Remove the chdir and write the results into "
                        "the current directory.")
            else:
                # ... and output full of NaN/Inf is a fabricated-looking result.
                nonfinite = check_result_files_finite(out_files)
            # Also scan the HEADLINE numbers (results_summary.json + stdout): a
            # summary can report max_value: Infinity while the mesh stays finite.
            nonfinite += check_summary_finite(work_dir, _stdout_text)
            # Structural defects in the solver's own data output. The scans
            # above look at headline numbers and mesh files; a FIELD that is
            # mostly NaN, or a wholly degenerate mesh, passed both and was
            # stamped verified. (Anti-fabrication gate.)
            nonfinite += inspect_result_artefacts(out_files)
            if nonfinite:
                result.setdefault("validation", []).extend(nonfinite)
            # WHAT THE RESULT HOLDS. A result that is exactly zero everywhere is not a result;
            # a field that never changed in time, and the mesh the result lives on (cells,
            # extent, area), are stated beside it. Measured 2026-09-29: a run on a 95-cell mesh
            # of the obstacle's disc wrote zero velocity and pressure, and the gate said "the
            # automated checks passed" (core/fabrication_gate.result_content_findings).
            _empty, _notes, _mesh = result_content_findings(out_files)
            if _mesh:
                result["result_mesh"] = _mesh
            if _notes:
                result["result_notes"] = _notes
            if _empty:
                result.setdefault("validation", []).extend(_empty)
            # The 'finiteness not asserted' honesty note is a coverage gap,
            # not evidence of a bad number — it must not flip the verdict to
            # 'non-finite values' (FEBio .xplt / .bp-without-adios2 runs).
            nonfinite = [x for x in nonfinite
                         if not x.startswith("finiteness not asserted")]
            # openPASO computes the run's headline numbers from the run's own
            # data, so the agent never has to assert one of its own.
            if out_files:
                result["openpaso_computed"] = _attest_run_quantities(
                    work_dir, job.job_id)
                # …and, if the run declared what it is solving, whether that
                # data satisfies those equations at all.
                if verify_pde:
                    result["residual_check"] = _check_declared_pde(
                        verify_pde, out_files)
        # Verification gate: attestation binds the verdict to run evidence.
        if job.error:
            reason = "the solver run errored, so no number is backed by a valid run"
        elif job.status != "completed":
            reason = f"the run did not complete (status={job.status})"
        elif not out_files:
            reason = ("the process exited cleanly but produced NO output files, "
                      "so no reported number is backed by run evidence")
        elif nonfinite:
            reason = ("the result contains non-finite (NaN/Inf) values, so it is "
                      "numerically invalid"
                      if any("non-finite" in x for x in nonfinite)
                      else "a result file is unreadable/corrupt, so the gate "
                           "could not assert the output's integrity")
        elif _empty:
            reason = _empty[0]
        elif _residual_blocks_verification(result):
            reason = ("the field this run produced does NOT satisfy the "
                      "equations it declared: "
                      + str(result["residual_check"].get("detail", "")))
        else:
            reason = ""
        _stamp_verification(result,
                            evidence_ok=(bool(out_files) and not job.error
                                         and not nonfinite and not _empty
                                         and not _residual_blocks_verification(result)),
                            reason=reason, critic_approved=critic_approved,
                            solver=solver, setup_text=input_content,
                            critic_token=critic_token,
                            job_id=str(result.get("job_id", job_name or "")))
        return json.dumps(result, indent=2)



    # write_participant_contract (below, by the knowledge tool) is the replacement
    # that does NOT walk through Option B: it writes the SAME elided text the
    # knowledge reply serves in parts -- one door, `_serve_participant`, which
    # fails closed -- and nothing else. Measured on a coupled round: two runs
    # spent 17 and 35 of their 45 minutes re-typing a 33k-character served
    # contract back into a file, one of them six times, and one reply of it hit
    # the harness's output cap. The text on disk is the text on the wire.
    #
    # materialize_participant WAS HERE, AND IT WALKED THROUGH OPTION B.
    #
    # It shutil-copied the COMPLETE participant file — solve included — into
    # the agent's work_dir. Option B elides the solve at SERVING time, so the
    # knowledge payload stopped handing over a solver while this tool went on
    # handing over the same file whole, by a different door. Measured on one batch of runs: agents called it 2-4 times per run and 8 to
    # 10 participant files landed in each workspace, two per run carrying the
    # full solve. Those runs are void.
    #
    # It is also the behaviour that was already reverted once, in 64a922af:
    # openPASO answers questions, it does not install runnable solvers into the
    # agent's workspace. Reverting the automatic delivery and leaving the
    # manual one standing fixed the door and not the room.
    #
    # That tool has no replacement. An agent that wants the participant reads it in
    # the coupling knowledge, minus the solve, or has that same elided text
    # written by write_participant_contract, and writes the solve itself.

    def _pde_consistency_body(solution_files: str, source_term: str,
                               coefficient: str = "1.0",
                               domain: str = "[[0,1],[0,1]]",
                               equation: str = "", reaction: str = "0.0") -> str:
        """Does your field actually satisfy the equation the task stated?

        A refinement study CANNOT answer this. Measured over 464 runs, result
        sets with a complete set of levels self-converge at a median order
        of 1.96 to 1.99 — the discretisation is fine — while a field that
        converges cleanly to the WRONG function looks identical in that study.
        Your own mesh-independence verdict does not separate them either: it
        catches three quarters of the wrong runs and also fires on half the
        correct ones.

        This checks the weak identity that any solution of
        -div(K grad u) = f obeys for a smooth v vanishing on the boundary:

            integral of u * (L* v)  ==  integral of f * v

        using ONLY the operator and source from your task and the values you
        already wrote. There is no reference solution in it.

        Measured separation, by execution: a field that solves the stated
        problem drives this residual down by roughly a factor of four per
        refinement, from 2.3e-02 to 2.6e-04 over four levels; a field that does
        not leaves it FLAT, near 6.4 at every level. The verdict is decided by
        whether it falls, not by any particular value.

        Args:
            solution_files: comma-separated paths to your solution CSVs, one
                per mesh level, each `x, y, u` (or `x, y, z, u`) with a header.
            source_term: the task's source term, in the task's own Python
                notation, e.g. "36*x**3*y - 20*x**3/3 + 8/3".
            coefficient: "2.5" for a scalar, or a symmetric tensor written as
                "[[3,-1],[-1,2]]". Must match what the task prescribes.
            domain: the box the problem lives on, "[[x0,x1],[y0,y1]]".
            equation: your task's equation, copied from its EQUATION line.
                REQUIRED. This check implements the second-order scalar
                diffusion form only, and it refuses anything else rather than
                answering about an operator it does not model.
        """
        # RECORDED LIKE ITS SIBLINGS. couple() has recorded itself in the
        # session journal since the journal existed and these two never did,
        # so openPASO could not see that a run had coupled level after level and
        # checked none of them against its own equation. Measured over every
        # recorded cell: the lead that asks for this check was served 14 times
        # in 10 cells and the tool was called twice, in one cell.
        try:
            _get_journal().record("tool_call", "verify_pde_consistency",
                                  solver="general", physics="coupling")
        except Exception:                                # noqa: BLE001
            pass
        import csv as _csv
        import json as _json
        from .pde_consistency import check_levels

        # IT MUST REFUSE OUTSIDE ITS OWN OPERATOR, AND IT DID NOT.
        #
        # Measured by handing real result sets to the unguarded version:
        #
        #   a linear-elasticity problem (Lame lambda/mu)
        #       -> "INCONSISTENT ... your field is converging to something that
        #          is not the solution of the stated problem"
        #   a biharmonic problem (lap(lap(u)) = f)
        #       -> "CONSISTENT ... Your field satisfies the equation you were
        #          given" — at rate 2.09, for an operator this check does not
        #          model at all
        #
        # The first tells an agent to throw away work that may be right. The
        # second BLESSES a field on evidence that does not exist, which is
        # worse. Of the 16 single-code development problems, 13 are outside the
        # implemented form — elasticity, Stokes, Navier-Stokes, biharmonic,
        # transient heat, a nonlinear a(u), a variable a(x,y) — and the run on
        # the nonlinear one was observed calling this tool.
        #
        # The universal core advertises it on 100% of knowledge calls, so an
        # unguarded verdict reaches every run that asks. A check that
        # answers about an operator it does not implement is not a weak check;
        # it is a source of wrong answers pointed at every run that asks.
        # Task and spec equation lines carry a trailing gloss in parentheses —
        # one steady-diffusion problem's is "-lap(u) = f  (steady diffusion,
        # unit conductivity)" — and that is one of the three problems this
        # check IS valid for. Strip a
        # trailing parenthetical before matching, or the guard refuses the
        # cases it exists to serve.
        import re as _re
        _raw = _re.sub(r"\s*\([^()]*\)\s*$", "", str(equation)).strip()
        _eq = "".join(_raw.split()).lower()
        # MATCH THE OPERATOR'S SHAPE, NOT A LIST OF SPELLINGS. A whitelist of
        # exact strings refused `-div(K grad T) = f in each subdomain` (two
        # coupled conduction problems) purely because the field is called T —
        # the same operator this check
        # implements. The coefficient token must be a bare name or absent: an
        # `a(x,y)` or `a(u)` carries parentheses and is refused, which is right,
        # because a coefficient varying in space or in the solution breaks the
        # constant-K adjoint this check uses.
        # A REACTION TERM IS INSIDE THIS OPERATOR, NOT OUTSIDE IT. -div(K grad u)
        # + c u with a constant c is self-adjoint exactly as constant-K diffusion
        # is, so the identity holds with + c v in the adjoint (see check_levels).
        # It was refused for the shape of its equation alone, which cost the
        # reacting side of every coupled conduction problem the only check that
        # separates a converged-right field from a converged-wrong one. The
        # coefficient of u must be a bare name or a number: `c(x)u` or `u**2`
        # carries something this identity does not model and stays refused.
        _OP = _re.compile(r"^-(?:div\(([a-z]*)grad([a-z]+)\)"
                          r"|lap(?:lacian)?\(([a-z]+)\))"
                          r"(?:\+[a-z0-9.]*\*?[a-z]+)?=f(.*)$")
        # A COUPLED DIFFUSION PROBLEM STATES THE SAME OPERATOR AND MUST NOT BE
        # REFUSED. One task reads "-div(k grad u) = f in each subdomain", which
        # is the implemented form applied per side — and exact-match alone
        # refused it. Prefix matching with a whitelist of benign qualifiers
        # keeps that case while still refusing "-div(a(u) grad u) = f", whose
        # coefficient depends on the solution, and "dT/dt - div(...)", which is
        # transient. The qualifier list is deliberately tiny: a phrase nobody
        # anticipated is a REFUSAL, not a guess.
        _QUAL = ("", "ineachsubdomain", "oneachsubdomain", "insubdomaina",
                 "insubdomainb", "inbothsubdomains")
        # `group(1) or ""` — on the `lap(u)` branch the coefficient group does
        # not participate and is None, which is not "" and silently refused
        # a steady-diffusion problem, one of the very ones this check is valid for.
        _m = _OP.match(_eq)
        _matched = bool(_m and (_m.group(1) or "") in ("", "k", "a")
                        and (_m.group(4) or "") in _QUAL)

        # LINEAR ELASTICITY IS A SECOND OPERATOR THIS CHECK NOW MODELS.
        #
        # It used to be the headline refusal, and rightly: the scalar adjoint
        # applied to a displacement field reported it converging to the wrong
        # solution, which meant nothing. But for CONSTANT lambda and mu the
        # elasticity operator is self-adjoint exactly as constant-K diffusion
        # is, so the same weak identity holds with L*v = -div(sigma(v)), and
        # the sin^2 test function kills both boundary terms for a coupled side
        # too. Calibration is in _adjoint_elastic_flat's docstring: on a
        # manufactured case the exact field falls 64.8x while a 3 % error in
        # ONE component stays flat.
        #
        # This matters beyond tidiness. The problem family this refusal
        # covered, coupled elasticity, is the one with the only recent correct run -- so the
        # single coupled problem that had started working was the one nothing
        # could check.
        _ELASTIC = _re.compile(
            r"^-div\(sigma\(([a-z]+)\)\)=f"
            r"(?:,sigma\(\1\)=.*)?$")
        _is_elastic = bool(_ELASTIC.match(_eq))
        if _is_elastic:
            return _verify_elastic(solution_files, source_term, coefficient,
                                   domain, equation)
        if not _eq:
            return (
                "REFUSED: pass `equation=` exactly as your task states it.\n\n"
                "This check implements ONE operator, the second-order scalar\n"
                "diffusion form -div(K grad u) = f with a CONSTANT symmetric\n"
                "K. It has no way to tell from your numbers alone whether that\n"
                "is your problem, and answering anyway is how it blesses a\n"
                "field it cannot judge: handed a biharmonic result set it\n"
                "reported CONSISTENT at rate 2.09, and handed an elasticity\n"
                "result set it reported that the field converges to the wrong\n"
                "solution. Neither verdict meant anything.\n\n"
                "If your equation is elasticity, Stokes, Navier-Stokes,\n"
                "biharmonic, transient, or has a coefficient depending on u or\n"
                "on position, this tool cannot check it and will say so. Use\n"
                "audit_results(work_dir=...) instead — it is operator-agnostic,\n"
                "reads only your own files, and catches a near-zero field, a\n"
                "tolerance floor, an order below the one you are claiming and a\n"
                "non-monotone sequence.")
        if not _matched:
            return (
                f"REFUSED: this check implements -div(K grad u) = f with a "
                f"constant symmetric K, and your equation is {equation!r}.\n\n"
                f"It is not a weaker check outside that form, it is a wrong "
                f"one: on a biharmonic result set it reported CONSISTENT at "
                f"rate 2.09, and on an elasticity result set it reported that "
                f"the field converges to the wrong solution. Both verdicts "
                f"were meaningless and both would have changed what the run "
                f"did next.\n\n"
                f"What still works on your problem, with no reference "
                f"solution: audit_results(work_dir=...) for a near-zero field, "
                f"a tolerance floor, an order below the one you are claiming, "
                f"and a non-monotone sequence; and for a vector problem, "
                f"checking each component's boundary values against the "
                f"prescribed ones.")
        try:
            coeff = _json.loads(coefficient)
        except Exception:
            try:
                coeff = float(coefficient)
            except Exception:
                return (f"coefficient {coefficient!r} is neither a number nor "
                        f"a JSON matrix like [[3,-1],[-1,2]]")
        try:
            box = [tuple(float(v) for v in pair)
                   for pair in _json.loads(domain)]
        except Exception:
            return (f"domain {domain!r} must be JSON like [[0,1],[0,1]] — one "
                    f"[low, high] pair per axis")
        levels, problems = {}, []
        for i, raw in enumerate(str(solution_files).split(","), start=1):
            path = Path(raw.strip())
            if not path.is_file():
                problems.append(f"{path} does not exist")
                continue
            rows = []
            try:
                with path.open() as fh:
                    for row in _csv.reader(fh):
                        try:
                            rows.append(tuple(float(c) for c in row))
                        except ValueError:
                            continue                      # header
            except OSError as exc:
                problems.append(f"{path}: {exc}")
                continue
            width = len(box) + 1
            rows = [r for r in rows if len(r) >= width]
            if not rows:
                problems.append(f"{path}: no rows with {width} numeric columns")
                continue
            levels[i] = [r[:width] for r in rows]
        if not levels:
            return ("no readable level files. " + "; ".join(problems))
        try:
            _c = float(str(reaction).strip() or 0.0)
        except ValueError:
            return (f"reaction {reaction!r} must be a number: the constant c of "
                    f"-div(K grad u) + c u = f, or 0 when the equation has no "
                    f"reaction term")
        try:
            result = check_levels(levels, source_term, coeff, box, reaction=_c)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        out = result.as_dict()
        if problems:
            out["files_skipped"] = problems
        return _json.dumps(out, indent=2)

    @mcp.tool()
    def verify_pde_consistency(solution_files: str, source_term: str,
                               coefficient: str = "1.0",
                               domain: str = "[[0,1],[0,1]]",
                               equation: str = "", reaction: str = "0.0") -> str:
        """Does your field actually satisfy the equation the task stated?

        The body is shared with couple(), which runs this same check on
        every converged level when it is handed the four arguments. Two
        wordings of the invitation to call this tool were measured and both
        failed -- 2 calls against 14 asks, then 0 against 7 -- so the check
        stopped being something to remember and became something couple()
        does with what the agent already gave it.
        """
        return _pde_consistency_body(solution_files, source_term, coefficient,
                                     domain, equation, reaction) + _UNIVERSAL_CORE



    @mcp.tool()
    def verify_interface_flux(interface_files: str,
                              solution_files: str = "",
                              interface_axis: int = 0) -> str:
        """On a COUPLED task: is your interface flux the right sign, and do
        your two sides actually agree there? Runs on YOUR OWN files, with no
        reference solution.

        THIS EXISTED ONLY INSIDE AN INDEPENDENT CHECK UNTIL NOW, which is why
        it is here. The check below is the one that decided one coupled
        recorded run: a result set whose two codes both genuinely ran,
        whose coupling genuinely iterated over three mesh levels, and whose
        interface FIELD matched to 0.000e+00 across the seam, was still
        complete but unphysical — because one side reported its flux with the
        INWARD normal. The relative flux jump came out 8.139e-01, 9.066e-01,
        9.530e-01 over the three levels: not shrinking, and growing. The
        agent had no way to see that before handing in. Now it does.

        WHAT IT CHECKS, all of it key-free:

        1. SIGN AND SELF-CONSISTENCY, per side. For a flux you really computed
           from your own solution,

               q_n(x) / (-du/dn)(x)  ==  k   at every interface point,

           so the flux is a CONSTANT multiple of -du/dn whatever k is — and
           POSITIVE. A NEGATIVE multiple means your normal points the
           wrong way: the task defines q_n = -(K grad u) . n_out with n_out
           pointing OUT of the subdomain. A flux far from every constant
           multiple, level after level, does not follow from the field you
           delivered; one that comes closer at every level is discretisation.

           The trap this catches most often: on the NEUMANN side the flux you
           IMPORT and the flux you REPORT have OPPOSITE signs. Kratos's
           FACE_HEAT_FLUX is the INWARD normal flux (measured against a closed
           form: u came out +0.875 where the inward reading predicts +0.875),
           so the number you write into each per-level interface file is the
           NEGATIVE of the one you applied.

        2. THE TWO-SIDED JUMP, and its refinement trend. The field trace must
           be continuous and the two fluxes must cancel, because the normals
           are anti-parallel. A jump that stays O(1) as h halves means the
           iteration converged to a fixed point of the WRONG transmission
           condition — and your observed order cannot see that, because
           convergence to a wrong answer is still convergence.

        Args:
            interface_files: comma-separated per-level interface files, one
                per side, in the shape `x, y, u, qn`; each name must carry
                `level<k>_<side>`, which is how level and side are read. Give
                BOTH sides and all levels; the trend is the informative part.
            solution_files: comma-separated per-level field files, one per
                side, `x, y, u`, named the same way. Needed for check 1 —
                without them the sign cannot be tested, only the jump.
            interface_axis: a fallback only. The normal is read from the files:
                the coordinate a straight interface holds constant, and on a bent
                interface each leg's own.

        Returns: per-side sign verdicts (leg by leg on a bent interface), per-level
            jumps, the refinement trend of each, and NOT_ASSESSED wherever a check
            could not look at anything — a check that could not run never reports
            success.
        """
        import re as _re
        from pathlib import Path as _RawPath
        from tools.result_audit import resolve_under_cell as _under_cell

        def _P(f):
            """Same resolution the audit uses: a relative path is the CALLER's.

            This door took `Path(f)` on whatever string it was handed. The MCP
            server is a separate process with its own working directory, so a
            relative name resolved against OURS and the tool then reported that
            the caller's file did not exist. Measured on two runs of
            2026-09-19, one of which spent its last three calls checking
            whether it had lost its own results.
            """
            return _RawPath(str(_under_cell(f)))


        try:
            from blind_eval import interface as _IF
        except Exception as exc:                       # pragma: no cover
            return f"REFUSED: the interface checker is unavailable: {exc}"

        def _split(spec):
            return [t.strip() for t in str(spec).split(",") if t.strip()]

        # A WIDE FILE IS READ BY ITS HEADER, NEVER BY POSITION. This read every
        # file as `x, y, u, qn`; a thermo-elastic interface file is `x, y, T, ux,
        # uy, qn, tx, ty`, so it took ux for the heat flux and served a false
        # SIGN_CONVENTION with a jump of 2.0 at every level -- which a correct
        # run copied into its hand-in (measured). A file of exactly the scalar
        # shape reads as before; a wider one is read by name for its one scalar
        # pair (T and qn, or u and qn); a file with no such pair is refused.
        def _read(path, want_flux):
            import csv as _csv
            with open(path, newline="", errors="ignore") as fh:
                rows = [r for r in _csv.reader(fh) if r and any(c.strip() for c in r)]
            if not rows:
                return None, "empty file"
            try:
                float(rows[0][0])
                has_head = False
            except (ValueError, IndexError):
                has_head = True
            narrow = 4 if want_flux else 3
            if not has_head or len(rows[0]) <= narrow:
                return _IF.read_interface_csv(path, 2, 1, 1 if want_flux else 0)
            head = [c.strip().strip('"').lower() for c in rows[0]]

            def _idx(*names):
                for n in names:
                    if n in head:
                        return head.index(n)
                return None
            ix, iy = _idx("x"), _idx("y")
            iu = _idx("t", "temperature", "u", "phi", "c")
            iq = _idx("qn", "q", "q_n", "flux") if want_flux else None
            if ix is None or iy is None or iu is None or (want_flux and iq is None):
                return None, (f"{len(head)} columns ({', '.join(head)}): this check reads ONE scalar "
                              f"field and its normal flux (T and qn, or u and qn) and there is no such "
                              f"pair here; a vector exchange is judged per channel in the couple_levels "
                              f"reply and by audit_results")
            pts, vals, flux = [], [], []
            for r in rows[1:]:
                try:
                    pts.append((float(r[ix]), float(r[iy])))
                    vals.append((float(r[iu]),))
                    flux.append((float(r[iq]),) if want_flux else ())
                except (ValueError, IndexError):
                    return None, "unparsable numeric value"
            return (pts, vals, flux), "ok"

        def _key(name):
            m = _re.search(r"level(\d+)_([AB])", name)
            return (int(m.group(1)), m.group(2)) if m else None

        iface, refused = {}, []
        for f in _split(interface_files):
            k = _key(_P(f).name)
            if k is None:
                refused.append(f"{f}: name carries no level<k>_<side>")
                continue
            try:
                # read_interface_csv returns (parsed, why) and parsed is the
                # TRIPLE (points, values, fluxes) -- unpacking it wrong is
                # what made the first version of this tool abstain on every
                # side with "type tuple doesn't define __round__".
                parsed, why = _read(_P(f), True)
                if parsed is None:
                    refused.append(f"{f}: {why}")
                else:
                    iface[k] = parsed
            except Exception as exc:
                refused.append(f"{f}: {type(exc).__name__}: {exc}")
        fields = {}
        for f in _split(solution_files):
            k = _key(_P(f).name)
            if k is None:
                continue
            try:
                parsed, why = _read(_P(f), False)
                if parsed is None:
                    refused.append(f"{f}: {why}")
                else:
                    fields[k] = (parsed[0], parsed[1])
            except Exception as exc:
                refused.append(f"{f}: {type(exc).__name__}: {exc}")

        if not iface:
            return json.dumps({
                "verdict": "REFUSED",
                "detail": ("no interface file could be read; give the "
                           "per-level interface file paths, one per side, "
                           "named with level<k>_<side>, in the shape "
                           "`x, y, u, qn` (a wider file is read by its header: "
                           "T and qn)"),
                "files_refused": refused}, indent=2) + _UNIVERSAL_CORE

        levels = sorted({k for k, _ in iface})
        out = {"levels_seen": levels, "per_side": [], "per_level": [],
               "files_refused": refused}

        # ---- check 1: sign and self-consistency, per side and level
        for lvl in levels:
            for side in ("A", "B"):
                got = iface.get((lvl, side))
                if got is None:
                    continue
                ipts, _iv, iq = got
                fld = fields.get((lvl, side))
                if fld is None:
                    out["per_side"].append({
                        "level": lvl, "side": side, "verdict": "NOT_ASSESSED",
                        "detail": ("no level-%d field file for side %s was given, so "
                                   "the reported flux cannot be compared with "
                                   "your own field and the SIGN IS UNTESTED"
                                   % (lvl, side))})
                    continue
                # GEOMETRY FROM THE FILES, NOT FROM AN ASSUMPTION. The
                # audit's copy of this check was fixed the same way after it
                # reported WRONG SIGN on a verified-correct horizontal-
                # interface result set (obeying it corrupted the data). The
                # axis is the coordinate constant across the interface
                # probes; the plane its value; the outward sign follows
                # from which side of the plane this side's own field lies --
                # and on a BENT interface each leg has its own (measured: one
                # axis for two legs told a correct side SIGN_CONVENTION and
                # INCONSISTENT by 98670 %, and the hand-in gave that as its cause).
                try:
                    from tools.result_audit import side_flux_ratio as _sfr
                    res = _sfr(ipts, iq, fld[0], fld[1], _IF, interface_axis)
                except Exception as exc:
                    res = {"verdict": "NOT_ASSESSED",
                           "detail": f"{type(exc).__name__}: {exc}"}
                res.update({"level": lvl, "side": side})
                # A CONSTANT RATIO IS NOT THE STATED COEFFICIENT. The ratio was
                # judged for spread and sign only, and a side whose implied
                # coefficient was 8.742 against a stated k of 1 read CONSISTENT
                # (measured, on a side that had never solved its interior).
                out["per_side"].append(res)
        # THE TREND DECIDES WHAT A COARSE MISFIT WAS. Measured on a right field: the
        # implied coefficient varied by 31 % along the interface at level 1, 12 % at
        # level 2 and 8 % at level 3, and level 1 alone read "the two were not
        # computed from each other".
        out["misfit_trend"] = _ratio_misfit_trend(out["per_side"])
        if not out["misfit_trend"]:
            del out["misfit_trend"]
        try:
            from tools.result_audit import _side_operators as _sops
            _wk = _RawPath(str(_under_cell(_split(interface_files)[0]))).parent
            _ops = _sops(_wk)
        except Exception:                                    # noqa: BLE001
            _ops = {}
        for res in out["per_side"]:
            try:
                _op = next((o for n, o in _ops.items()
                            if n.replace("side_", "").upper() == str(res.get("side")).upper()), None)
                _ks = float(_op["k"]) if _op and isinstance(_op.get("k"), (int, float)) else None
                _res_coefficient_check(res, _ks)
            except Exception:                                # noqa: BLE001
                pass

        # ---- check 2: the two-sided jump and its trend
        per_level = []
        for lvl in levels:
            a, b = iface.get((lvl, "A")), iface.get((lvl, "B"))  # triples
            if a is None or b is None:
                per_level.append({"level": lvl, "verdict": "NOT_ASSESSED",
                                  "detail": "only one side present"})
                continue
            try:
                # two_sided_jumps returns (dict | None, message) -- a TUPLE.
                # `dict(level=lvl, **that)` raises TypeError, the except below
                # swallowed it into NOT_ASSESSED, and so THIS CHECK HAS NEVER
                # RUN: jq stayed empty, `len(jq) >= 2` was never true, and the
                # flux_jump_trend block the docstring is written around was
                # silently absent from every reply, with nothing in its place
                # saying so. A report section that exists only in the healthy
                # case, whose absence reads as a pass.
                # THE SAME POINTS, WHEREVER EACH SIDE LISTS THEM: where the two files hold
                # different numbers of rows, the jump is taken on the points both hold.
                if len(a[0]) != len(b[0]):
                    from core.quality_checks import pair_interface_points as _pair
                    _pr = _pair(a[0], b[0])
                    if _pr is not None and len(_pr["ia"]) >= 2 and not (_pr["only_a"] and _pr["only_b"]):
                        a = tuple([t[i] for i in _pr["ia"]] for t in a)
                        b = tuple([t[j] for j in _pr["ib"]] for t in b)
                jumps, why = _IF.two_sided_jumps(a, b)
                if jumps is None:
                    per_level.append({"level": lvl, "verdict": "NOT_ASSESSED",
                                      "detail": why or "the two sides do not "
                                                       "line up at this level"})
                else:
                    per_level.append(dict(level=lvl, **jumps))
            except Exception as exc:
                per_level.append({"level": lvl, "verdict": "NOT_ASSESSED",
                                  "detail": f"{type(exc).__name__}: {exc}"})
        out["per_level"] = per_level
        jq = [p.get("jump_q_rel") for p in per_level
              if isinstance(p.get("jump_q_rel"), (int, float))]
        if len(jq) >= 2 and max(jq) < 1e-6:
            # the audit's floor (result_audit._JUMP_ROUND_OFF): no trend at round-off
            out["flux_jump_trend"] = {
                "values": jq, "shrinking": None,
                "detail": ("the flux jump is at round-off at every level: the two fluxes "
                           "agree to solver precision, and there is no trend to read")}
        elif len(jq) >= 2:
            shrinking = all(jq[i + 1] < jq[i] for i in range(len(jq) - 1))
            out["flux_jump_trend"] = {
                "values": jq,
                "shrinking": shrinking,
                "detail": ("the flux jump falls under refinement, which is "
                           "what a satisfied transmission condition looks like"
                           if shrinking else
                           "THE FLUX JUMP DOES NOT SHRINK under refinement, so "
                           "it is not discretisation error. This measures the "
                           "jump, not its cause. Causes on record: a wrong sign "
                           "or scaling of an exchanged flux (on the Neumann side "
                           "the flux you import and the flux you report are "
                           "opposite), the partner's data applied by row index "
                           "instead of by position (row i of the partner's list is "
                           "not point i of this side's interface when the two list "
                           "their points in different orders), a boundary not held "
                           "where the problem holds it, and a linear system that "
                           "was not solved.")}
        return json.dumps(out, indent=2) + _UNIVERSAL_CORE

    @mcp.tool()
    async def audit_results(work_dir: str, claimed_order: float = 0.0,
                            ctx: Context = None) -> str:
        """Check your OWN result files for the failures that most often sink a
        result set — BEFORE you hand it in. Uses only files you produced; no
        reference solution is involved, so a clean audit means self-consistent,
        not correct.

        What it catches, measured on 94 independently-checked correct
        result sets (no false alarm on any) and 102 complete-but-wrong ones
        (39 caught, about four in ten — more when the result set states its
        claimed convergence order, which the order check needs):

          * NEAR-ZERO FIELD - your finest solution peaks below 1e-8. On a
            driven problem check that the source/load reaches the solve (a
            defined function no condition references, a load curve never
            activated, boundary values all zero) before reading it as small.
          * FLOOR - successive refinement levels within 5% of each other:
            whatever limits your number, it is not the mesh. Usual cause is a
            solver tolerance (nonlinear/iterative defaults stop near 1e-6).
          * ORDER MISMATCH - your levels improve at a measurably lower rate
            than the order you are about to claim. Usual causes: element
            degree below what the task states, volumetric locking, a
            first-order integrator behind a spatial study.
          * NON-MONOTONE - a refinement made the answer worse.

        Call it on the directory holding your per-level outputs (it reads
        your summary file and your per-level field files), pass the
        convergence order you intend to claim, and treat any finding as a
        reason to look BEFORE handing in - each one names where to look.

        Args:
            work_dir: directory containing your results (searched recursively)
            claimed_order: the convergence order your result set will claim
                (0 = no order claim, order checks are skipped)
        """
        from . import result_audit
        try:
            r = result_audit.audit(work_dir,
                                   claimed_order=claimed_order or None)
        except Exception as e:                                # noqa: BLE001
            return json.dumps({"error": f"{type(e).__name__}: {e}"[:200]})
        return json.dumps(r, indent=2)

    @mcp.tool()
    async def verify_mesh_independence(
            solver: str, input_template: str, resolution: float,
            refinement_factor: float = 2.0, levels: int = 1,
            parameter_kind: str = "divisions", field: str = "",
            probe_points: str = "", rel_tol: float = 0.01,
            job_name: str = "", np: int = 1,
            critic_approved: bool = False, ctx: Context = None) -> str:
        """Heuristic mesh-independence study for problems WITHOUT an exact
        solution: re-run the SAME problem at successively refined
        resolutions and accept it as converged only if ALL monitored
        quantities stop changing materially.

        MMS convergence tests need a manufactured exact solution; real
        application problems have none. This tool automates the
        established recourse: halve the discretisation length (once by
        default, more via `levels`), then compare (a) a volume-weighted
        global L2 norm and the global max of the primary field, (b) the
        field value at probe points (auto-chosen from the mesh — field
        hotspot, domain centre, off-centre interior points — or supplied
        explicitly), and (c) any scalar QoIs the script writes to
        results_summary.json. Verdict: CONVERGED only if every monitored
        quantity changes by less than `rel_tol` on the finest refinement
        step; otherwise NOT CONVERGED, with all numbers in the report.

        The input template must contain the placeholder __RESOLUTION__
        where the characteristic discretisation parameter goes, e.g.
        `nx = __RESOLUTION__`. For Python-scripted solvers (fenics,
        ngsolve, skfem, dune) the template is the solve script itself; for
        compiled/file-input solvers (fourc, dealii, kratos, febio) it is a
        generator script that writes the input file, exactly as in
        run_with_generator. The solve must write the primary field as
        nodal data in a VTU/VTK/VTP result file.

        IMPORTANT — this tool checks discretisation convergence only. It
        does not validate the model physics; have the MANDATORY critic
        review the setup and pass critic_approved=True as with the run
        tools.

        Args:
            solver: Backend name (any registered backend).
            input_template: Solve/generator script containing __RESOLUTION__.
            resolution: Coarsest value of the discretisation parameter.
            refinement_factor: Refinement per level (default 2 = halving h).
            levels: Number of refinements (default 1; runs levels+1 cases).
            parameter_kind: 'divisions' (parameter counts elements; refining
                multiplies) or 'size' (parameter is h; refining divides).
            field: Field name to monitor (default: auto-select from result).
            probe_points: Optional JSON list of probe coordinates, e.g.
                "[[0.5, 0.5], [0.25, 0.75]]" (default: auto from the mesh).
            rel_tol: Acceptance threshold on relative change (default 0.01).
            job_name: Optional study directory name.
            np: MPI processes per run.
            critic_approved: True only after the critic approved the setup.
        """
        import subprocess
        import sys
        from core import mesh_independence as mi
        from core.backend import InputFormat, find_generated_input, sorted_by_step

        _journal = _get_journal()
        _snap = _make_input_snapshot(input_template, solver,
                                     {"type": "mesh_independence_template"})
        _journal.record("tool_call", "verify_mesh_independence", solver=solver,
                        input_snapshot=_snap)

        def _fail(msg: str) -> str:
            _journal.record("tool_error", "verify_mesh_independence",
                            solver=solver, error_message=msg[:300],
                            input_snapshot=_snap)
            res = {"tool": "verify_mesh_independence", "solver": solver,
                   "status": "failed", "error": msg}
            _stamp_verification(res, evidence_ok=False, reason=msg[:200],
                                critic_approved=critic_approved,
                                solver=solver, setup_text=input_template)
            return json.dumps(res, indent=2)

        # Structured failures for these early exits too (Copilot review,
        # PR #49): every failure path of THIS tool returns the same JSON
        # shape with the verification stamp and a tool_error journal
        # record — a client must never have to branch on plain strings.
        backend = get_backend(solver)
        if not backend:
            return _fail(f"Unknown solver: {solver}")
        status, msg = backend.check_availability()
        if status.value != "available":
            return _fail(
                f"Solver {solver} not available: {_short_reason(msg)}")

        try:
            resolutions = mi.refinement_resolutions(
                resolution, refinement_factor, levels, parameter_kind)
            mi.substitute_resolution(input_template, resolutions[0])
        except ValueError as e:
            return _fail(str(e))
        if not (0 < rel_tol < 1):
            return _fail(f"rel_tol must be in (0, 1), got {rel_tol}")

        user_probes = None
        if probe_points.strip():
            try:
                user_probes = json.loads(probe_points)
                if (not isinstance(user_probes, list) or not user_probes
                        or not all(isinstance(p, (list, tuple)) for p in user_probes)):
                    raise ValueError("expected a JSON list of coordinate lists")
            except (json.JSONDecodeError, ValueError) as e:
                return _fail(f"probe_points is not a JSON list of coordinates: {e}")

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        study_dir = _OUTPUT_DIR / (job_name or f"{solver}_meshcheck_{ts}")
        is_python = backend.input_format() == InputFormat.PYTHON

        level_reports = []      # user-facing per-level info
        level_metrics = []      # input to mi.compare_levels
        shared_probes = user_probes
        pinned_field = field
        for lvl, res_val in enumerate(resolutions):
            content = mi.substitute_resolution(input_template, res_val)
            work_dir = study_dir / f"level{lvl}_res{mi.format_resolution(res_val)}"
            work_dir.mkdir(parents=True, exist_ok=True)

            if not is_python:
                # generator path (fourc / dealii / kratos / febio / sparta):
                # the template writes the solver input file first.
                gen_path = work_dir / "generate_input.py"
                gen_path.write_text(content)
                gen = subprocess.run([sys.executable, str(gen_path)],
                                     capture_output=True, text=True,
                                     cwd=str(work_dir), stdin=subprocess.DEVNULL)
                if gen.returncode != 0:
                    return _fail(f"level {lvl} (resolution "
                                 f"{mi.format_resolution(res_val)}): generator "
                                 f"failed: {gen.stderr[-400:]}")
                input_file = find_generated_input(work_dir, backend)
                if not input_file:
                    return _fail(f"level {lvl}: generator produced no input file")
                content = input_file.read_text()

            run_coro = backend.run(content, work_dir, np=np, timeout=None)
            if ctx is not None:
                job = await _run_with_progress(
                    ctx, run_coro,
                    f"Mesh study level {lvl}/{levels} on {solver}")
            else:
                job = await run_coro
            _jobs[job.job_id] = job

            if job.status != "completed" or job.error:
                return _fail(
                    f"level {lvl} (resolution {mi.format_resolution(res_val)}) "
                    f"did not complete: {(job.error or job.status)[:400]}")

            # plain FILES only: dolfinx VTXWriter can emit a DIRECTORY named
            # *.vtu, which no mesh reader can open (agent-validation S1 hit
            # exactly this and burned an iteration on it)
            out_files = [f for f in backend.get_result_files(job)
                         if f.suffix.lower() in (".vtu", ".vtk", ".vtp")
                         and not f.name.endswith(".pvtu") and f.is_file()]
            if not out_files:
                return _fail(
                    f"level {lvl} exited cleanly but produced no readable "
                    f"result file (.vtu/.vtk/.vtp) — no number is backed by "
                    f"run evidence")
            result_file = sorted_by_step(out_files)[-1]

            try:
                metrics = mi.extract_level_metrics(
                    result_file, field=pinned_field, probe_points=shared_probes)
            except Exception as e:
                # unreadable/fieldless output is a verdict, never a crash
                return _fail(f"level {lvl}: {e}")
            if lvl == 0:
                # Pin the auto-chosen field and probe locations so every
                # level monitors the SAME quantities at the SAME points.
                pinned_field = metrics["field"]
                shared_probes = metrics["probe_points"]
            metrics["resolution"] = res_val
            metrics["qoi"] = mi.collect_qoi_scalars(work_dir)
            level_metrics.append(metrics)
            level_reports.append({
                "level": lvl, "resolution": res_val,
                "job_id": job.job_id, "work_dir": str(work_dir),
                "result_file": result_file.name,
                "elapsed": f"{job.elapsed:.2f}s" if job.elapsed else None,
                "n_points": metrics["n_points"], "n_cells": metrics["n_cells"],
                "global_l2": metrics["global_l2"],
                "global_max": metrics["global_max"],
                "probe_values": metrics["probe_values"],
                **({"qoi": metrics["qoi"]} if metrics["qoi"] else {}),
            })

        comparison = mi.compare_levels(level_metrics, rel_tol=rel_tol)
        result = {
            "tool": "verify_mesh_independence", "solver": solver,
            "status": "completed",
            "field": pinned_field,
            "norm_type": level_metrics[0]["norm_type"],
            "parameter_kind": parameter_kind,
            "refinement_factor": refinement_factor,
            "probe_points": shared_probes,
            "levels": level_reports,
            "refinement_steps": comparison["steps"],
            "rel_tol": rel_tol,
            "converged": comparison["converged"],
            "verdict": comparison["verdict"],
            "study_dir": str(study_dir),
        }
        if not comparison["converged"]:
            result["failures"] = comparison["failures"]

        _journal.record("tool_success", "verify_mesh_independence",
                        solver=solver, input_snapshot=_snap)
        # Verification gate: the runs are the evidence; the verdict is the
        # check. A study whose quantities still drift is NOT a verified
        # solution — stamp it so the coarse answer cannot be reported.
        _stamp_verification(
            result,
            evidence_ok=comparison["converged"],
            reason=("the mesh-independence study did NOT converge: "
                    + "; ".join(comparison["failures"])
                    + f" (threshold {rel_tol:.2%}). The solution still "
                    "depends on the mesh — refine further"),
            critic_approved=critic_approved,
            solver=solver, setup_text=input_template)
        return json.dumps(result, indent=2)

    # ═══════════════════════════════════════════════════════════
    # 5. COUPLING (general couple() + legacy coupled_solve)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    async def coupled_solve(
        problem: str = "heat_dd", solver_a: str = "fenics",
        solver_b: str = "fourc", nx: int = 32, ny: int = 32,
        max_iter: int = 20, tol: float = 1e-6,
        relaxation: float = 1.0, params: str = "{}",
        critic_approved: bool = False,
    ) -> str:
        """LEGACY cross-solver coupling — FIXED toy geometries only. PREFER `couple`.

        DEPRECATED: this tool only handles a fixed enum of benchmark problems on a
        hardcoded unit-square split at x=0.5 (heat_dd/poisson_dd/one_way/tsi_dd/...).
        For ANY real or non-benchmark coupling use the general `couple` tool, which is
        physics-agnostic and validates the result for silent-wrong (flux balance,
        convergence, finiteness). Only use coupled_solve to reproduce the legacy
        benchmarks.

        Domain A (Dirichlet at interface) supports: fenics, ngsolve, skfem, dune.
        Domain B (Neumann at interface) supports: fenics, fourc, ngsolve, skfem, dune.
        Any combination of these works for heat_dd and poisson_dd problems.

        Args:
            problem: 'heat_dd', 'poisson_dd', 'one_way',
                     'poisson_dd_study', 'l_bracket_tsi', 'heat_dd_precice'.
                     'tsi_dd' is REMOVED: it reported converged=True,
                     iterations=1, residual=0.0 on a run that did one thermal
                     solve and one one-way structural solve and never fed
                     anything back. Two-way TSI goes through `couple` — see the
                     shipped participant_tsi_* scripts.
            solver_a, solver_b: Backend names
            nx, ny: Elements per direction
            max_iter: Max iterations
            tol: Convergence tolerance
            relaxation: Under-relaxation parameter
            params: JSON with additional parameters
            critic_approved: Set True after critic review
        """
        _get_journal().record("tool_call", "coupled_solve",
                              solver=f"{solver_a}->{solver_b}",
                              physics=problem)
        # Import and delegate to the full coupling implementation
        from tools.coupling import register_coupling_tools
        # The coupling tools are complex — delegate to the original implementation
        from tools.coupling import (
            _heat_domain_decomposition, _poisson_domain_decomposition,
            _oneway_thermal_structural, _twoway_tsi_coupling,
            _relaxation_parameter_study, _l_bracket_tsi,
            _heat_dd_precice_comparison,
        )

        param_dict = json.loads(params)
        backend_a = get_backend(solver_a)
        backend_b = get_backend(solver_b)
        if not backend_a or not backend_b:
            return f"Backend not found: {solver_a} or {solver_b}"

        dispatch = {
            "heat_dd": lambda: _heat_domain_decomposition(backend_a, backend_b, nx, ny, max_iter, tol, relaxation, param_dict),
            "poisson_dd": lambda: _poisson_domain_decomposition(backend_a, backend_b, nx, ny, max_iter, tol, relaxation, param_dict),
            "one_way": lambda: _oneway_thermal_structural(backend_a, backend_b, nx, ny, param_dict),
            "tsi_dd": lambda: _twoway_tsi_coupling(backend_a, backend_b, nx, ny, max_iter, tol, relaxation, param_dict),
            "poisson_dd_study": lambda: _relaxation_parameter_study(backend_a, backend_b, nx, ny, max_iter, tol, param_dict),
            "l_bracket_tsi": lambda: _l_bracket_tsi(backend_a, backend_b, nx, ny, param_dict),
            "heat_dd_precice": lambda: _heat_dd_precice_comparison(backend_a, backend_b, nx, ny, max_iter, tol, relaxation, param_dict),
        }

        if problem not in dispatch:
            return f"Unknown problem: {problem}. Available: {list(dispatch.keys())}"

        out = await dispatch[problem]()
        # LEGACY path returns human-readable text, not a gated JSON verdict.
        # An audit found this tool accepted `critic_approved` and then never
        # read it — a dead parameter on a tool the server instructions list as
        # critic-gated, so an unreviewed run was indistinguishable from a
        # reviewed one. The critic state now governs the verdict shown here,
        # and it is resolved from the server's review record rather than from
        # the flag: every parameter that changes what is solved goes into the
        # digest, so reviewing one configuration does not approve another.
        critic_ok, critic_note = _critic_state(
            f"{solver_a}->{solver_b}",
            _coupling_setup_text(problem=problem, solver_a=solver_a,
                                 solver_b=solver_b, nx=nx, ny=ny,
                                 max_iter=max_iter, tol=tol,
                                 relaxation=relaxation, params=params))
        if critic_ok:
            note = ("\n\n[openPASO verification: LEGACY coupled_solve — critic-reviewed. "
                    "Trust is governed by the convergence report above; a "
                    "non-converged run is reported as failure, never a result. "
                    "For a machine-readable verification verdict use `couple`.]")
        else:
            note = ("\n\n[openPASO verification: NOT VERIFIED — openPASO's independent "
                    "critic has not reviewed this setup, and openPASO treats no "
                    "result as trustworthy until it has. Do NOT report the values "
                    "above as a result: have a critic challenge the parameters, "
                    "units, discretisation and boundary conditions, then call "
                    "submit_critic_review with what it found and re-run. "
                    f"({critic_note}.) Asserting critic_approved=True does not "
                    "work: openPASO looks the review up rather than taking your "
                    "word for it. For a machine-readable verdict use `couple`.]")
        return (out + note) if isinstance(out, str) else out

    @mcp.tool()
    async def couple(participants: str | list, max_iter: int = 50, tol: float = 1e-6,
                     accelerator: str = "auto", theta: float = 0.5,
                     monolithic: str = "", probe: bool = True,
                     critic_approved: bool = False, noise_replicates: int = 0,
                     noise_floor: float = 0.0, noise_block: int = 3,
                     history_path: str = "",
                     iface_level: int = 0, pde_sources: str = "",
                     pde_check: str = "") -> str:
        """GENERAL partitioned multi-code coupling — works for ANY physics/coupling.

        Have an independent critic review the setup before coupling; pass
        critic_approved=True only after that review (every simulation must be
        critic-reviewed first).

        Unlike coupled_solve (legacy, fixed toy geometries), this is physics-agnostic:
        you write one self-contained solver script per subdomain/participant and openPASO
        runs the fixed-point iteration, relaxation, convergence-or-fail, AND the
        silent-wrong validation a partitioned coupling needs, because a partitioned
        coupling's characteristic failure is not a crash — it is a clean, converged,
        confidently wrong number. openPASO checks, and reports in the verdict:
          * convergence, and per-block convergence (a large settled block, e.g. force,
            cannot hide a small moving one, e.g. displacement, inside one global norm);
          * finiteness of every exchanged array, including coordinates and fluxes;
          * interface flux balance, naming SIGN-CONVENTION and UNIT-MISMATCH signatures;
          * that every participant exited 0 — a diverged solver often writes its last
            iterate and then aborts;
          * that every participant's output actually MOVED when its imports moved —
            the test for a participant that exits 0 having done nothing, or re-serves
            a cached answer; such a run "converges" at iteration 2 with residual 0;
          * that the coupling graph is wired as declared — an `imports_from` name that
            matches no participant is REFUSED, not silently dropped into a one-way run;
          * whether the two interface discretisations match;
          * and, when you pass `monolithic`, that the coupled answer equals an
            independent un-split solve of the same problem. That last one is the only
            check that can catch a consistent unit error or a wrongly applied interface
            sign, so when it is not supplied the verdict SAYS it was not run.

        PARTICIPANT CONTRACT — each iteration the driver, per participant:
          1. writes <work_dir>/imports.json = {partner_name: InterfaceData} (boundary
             data this participant consumes; empty on iteration 1).
          2. runs your `command` in <work_dir>.
          3. reads <work_dir>/exports.json = the InterfaceData your script produced on
             the shared interface.
        Your script decides HOW to apply imports (Dirichlet/Neumann/Robin/traction/
        flux/...) and WHAT to export — opaque to the driver, so it generalizes.

        InterfaceData JSON shape (read imports, write exports):
          {"field_name": str, "n_points": N, "coordinates": [[x,y(,z)],...],
           "values": [...], "normal_fluxes": [...]  # optional, for conservation check}

        THE DRIVER IS JACOBI, NOT GAUSS-SEIDEL: within one iteration every
        participant reads the PREVIOUS iteration's exports, and the driver relaxes
        EVERY participant's export vector. A two-participant Dirichlet-Neumann loop
        therefore relaxes twice per cycle and converges geometrically — it does not
        finish in one step even for a linear problem. theta=1.0 (no relaxation)
        oscillates forever on a balanced interface; start at theta=0.5.

        FLUX SIGN: export `normal_fluxes` with respect to YOUR OWN outward normal.
        The two normals are anti-parallel, so the two participants' fluxes carry
        OPPOSITE signs and their sums cancel. The Dirichlet value you APPLY is the
        same number on both sides — the opposite rule. Getting this wrong is the
        single most common cause of the flux-balance finding.

        Args:
            participants: JSON list of {"name", "command":[argv...], "work_dir",
              "imports_from":[partner names], "timeout": seconds}. Every name in
              `imports_from` must be another participant's name.
            max_iter, tol: iteration controls.
            accelerator: "auto" (default: Aitken for a single-field exchange, Anderson for a
                multi-field one -- resolved from the first exports and reported in `theta.mode`),
                "aitken" (theta recomputed each iteration from the residual
                or "anderson" (Anderson mixing / interface quasi-Newton on the whole interface state, window 5:
                measured on a three-component thermo-elastic exchange to cut the iteration count several-fold)
              history, starting at `theta`) or "constant" (theta held at `theta` for
              the whole run). There is no per-field or per-participant theta.
            theta: the relaxation factor. Under-relaxation (theta < 1) is what makes
              a Dirichlet-Neumann or FSI coupling converge at all when the physical
              stiffness/density ratio makes the un-relaxed iteration diverge; 0.5 is
              a neutral default, not a recommendation for your problem.
            noise_replicates: for a STOCHASTIC participant (DSMC / Monte-Carlo /
              any sampled estimator). Use 4 or more — the floor is itself an
              estimate and three samples is a bad one. It makes the driver run
              every participant that many times on the SAME imports and MEASURE
              the residual noise floor — the residual a perfectly converged run
              would still report. Convergence is then judged against
              max(tol, floor), over a block mean, so a correct stochastic
              coupling is no longer reported as a failure just because tol sits
              under the sampler's own scatter. 0 (the default) switches the whole
              branch off. Deterministic participants measure a floor of exactly 0
              and are unaffected.
            noise_floor: declare a floor instead of measuring one (or raise a
              measured one), if you established it independently.
            noise_block: how many consecutive residuals must AVERAGE below the
              criterion before the run stops. Only in effect when a non-zero
              floor is; a single residual dipping into the noise means nothing.
            probe: after the iteration settles, spend ONE extra solve per
              participant perturbing its final imports and measuring how far its
              answer moves. This is the only check here that can tell a solver
              which reads its boundary data from one that merely looks as if it
              does; turn it off only if that solve is genuinely unaffordable, and
              the verdict will then record that the question was not asked.
            monolithic: OPTIONAL JSON {"command":[argv...], "work_dir": str,
              "timeout": int} — a solve of the SAME problem un-split, in ONE code,
              which writes <work_dir>/monolithic.json in InterfaceData shape on the
              same interface. Supplying it is the strongest verification available
              here and needs no external benchmark.
                        history_path: optional ABSOLUTE CSV path. When set, openPASO writes its
                            measured finite residuals there as
                            ``iteration,interface_residual``. Use the path of the per-level
                            residual-history file your task names; never retype the returned history.

        iface_level: optional level number stamped into the suggested_filename of the interface_csv blocks the reply carries on convergence (each participant's own final interface data, ready to save verbatim).

        pde_check: THE ONE CHECK THAT SEPARATES A RIGHT ANSWER FROM A CONVERGED WRONG ONE, run for you on every converged level. JSON keyed by participant name: {"A": {"solution_files": "<this side's per-level field files, comma-separated, in level order>", "equation": "<your task's EQUATION line, verbatim>", "source": "<that side's source, as your task wrote it>", "coefficient": "<that side's coefficient>", "domain": "[[x0,x1],[y0,y1]]"}, "B": {...}}. The files must be on the PROBE GRID your task prescribes, not your mesh nodes: the test is a midpoint quadrature and on a scatter it reports nothing rather than a misleading number. You name them; openPASO guesses no filename and opens no task file. Give it once and openPASO puts each side's own field back into that side's own equation at every level from the second on, and reports CONSISTENT or INCONSISTENT in `pde_consistency`. A refinement study cannot do this: a field that converges cleanly to the WRONG function is indistinguishable in one -- measured on three recorded coupled runs whose every self-consistency measure reported a converging run while the answer was wrong. Nothing here is read from any task file; these are YOUR strings, and openPASO neither stores nor supplies them. A side whose operator this check does not model is refused by name and the rest still run.

        pde_sources: OPTIONAL, public-only. JSON {"A": {"source": "<the forcing/coefficient you actually implemented>", "task_source": "<the task's stated source, verbatim>"}, "B": {...}}. When supplied, openPASO compares the two PUBLIC strings and flags a mismatch — a silent wrong forcing (right shape, wrong function) converges cleanly to a different answer and no self-consistency check can see it. Never required; openPASO reads no reference solution and never supplies the equation for you.

        Returns: JSON with converged, iterations, residual, per-block residuals,
            exports, the coupling graph, per-participant responsiveness and exit
            codes, a `validation` block, a `checks_not_run` block, and the verdict.
            A coupling that failed any check is reported as NOT VERIFIED, never as a
            trustworthy result — and one that could not be fully checked says so.
            With the stochastic branch on it also returns `noise_floor`,
            `tol_effective` and `stopped_at_noise_floor`. READ `noise_floor`
            BEFORE JUDGING CONVERGENCE: any tolerance applied to a result that
            carries one — including an acceptance tolerance — must be at least
            that floor, or it is measuring the sampler rather than the coupling.
        """
        from core.coupling_driver import Participant, run_coupling
        from core.quality_checks import (
            check_interface_balance, check_finite, check_convergence,
            ENDS_ONLY_MARK as _ENDS_ONLY_MARK,
            check_coupling_directionality, check_participant_responsiveness,
            check_interface_meshes, check_residual_blocks, check_returncodes,
            check_interface_flux_profile, check_interfaces_are_the_same_surface,
            check_interface_sensitivity,
        )
        _get_journal().record("tool_call", "couple", solver="general", physics="coupling")
        # AT FUNCTION SCOPE, NOT INSIDE THE CONVERGED BRANCH. These are read
        # where the reply is assembled, which every path reaches, while only a
        # converged run fills them. Defining them in the branch raised
        # "cannot access local variable '_pde_verdicts'" on the first
        # non-converged call and cost a live cell its coupling -- it gave up
        # on the tool and hand-rolled its own driver.
        _pde_verdicts: dict = {}
        _pde_notes: list = []
        participants = _as_json_text(participants)
        try:
            specs = json.loads(participants)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"invalid participants JSON: {e}"})
        if not isinstance(specs, list) or len(specs) < 2:
            return json.dumps({"error": "need a JSON list of >=2 participants"})
        parts = []
        cell_work = os.environ.get("OPENPASO_CELL_WORKDIR")
        cell_root = Path(cell_work).resolve() if cell_work else None
        for s in specs:
            try:
                wd = Path(s["work_dir"])
                # A RELATIVE work_dir IS THE CALLER'S, AND WE CAN RESOLVE IT
                # WHEN WE KNOW WHERE THEY ARE.
                #
                # This refused outright, because a relative path would
                # otherwise resolve against the SERVER process's cwd -- which
                # the caller can neither see nor control -- and the run would
                # land somewhere unpredictable while looking like it worked.
                # That reason still holds when nobody has told us the caller's
                # directory, and the refusal stays for that case.
                #
                # When the caller's directory IS declared, refusing costs a
                # call for no reason and pushes them towards writing an
                # absolute path from scratch. Measured 2026-09-19: the only
                # concrete absolute path anywhere in the served text is the
                # interpreter openPASO itself hands out, and SIX OF TEN coupled
                # runs built a work_dir out of its directory -- ".../
                # open-fem-agent/side_A" -- then had to be refused for being
                # outside the task tree, and the correction changed the setup
                # so their critic review no longer bound. We showed them the
                # only absolute path they had.
                if not wd.is_absolute():
                    if cell_root is None:
                        return json.dumps({"error":
                            f"participant {s.get('name','?')}: work_dir must be "
                            f"an ABSOLUTE path, got {s['work_dir']!r} — nothing "
                            f"has told this server which directory a relative "
                            f"path would be relative TO, and resolving it "
                            f"against the server's own would put your run "
                            f"somewhere you cannot see."})
                    wd = cell_root / wd
                wd = wd.resolve()
                if cell_root is not None and not wd.is_relative_to(cell_root):
                    return json.dumps({"error":
                        f"participant {s.get('name','?')}: work_dir {wd} is "
                        f"outside this task's working directory {cell_root}. "
                        f"Put every participant under the current task tree; "
                        f"private /tmp is disposable and sibling paths are "
                        f"isolated."})
                # DO NOT CREATE A PARTICIPANT'S DIRECTORY. It used to be
                # `wd.mkdir(parents=True, exist_ok=True)`, which turns a WRONG
                # PATH into a MISSING SCRIPT: the directory appears, empty, the
                # participant is launched in it, and the failure the caller
                # reads is "can't open file 'part.py'" -- which sends them to
                # look at their script, not at the path they passed. Measured
                # 2026-09-19: a work_dir of /testbed escaped as a raw
                # PermissionError traceback out of the tool, and an invented
                # sibling directory was silently created instead.
                #
                # A participant runs where its script already is. The contrast
                # with the participants whose directories DO exist is the whole
                # diagnosis, so it is printed.
                if not wd.is_dir():
                    _present = sorted(
                        {str(Path(str(o.get("work_dir", ""))).expanduser())
                         for o in specs if isinstance(o, dict)
                         and str(o.get("work_dir", ""))
                         and Path(str(o["work_dir"])).expanduser().is_dir()})
                    return json.dumps({"error":
                        f"participant {s.get('name','?')}: work_dir {wd} does "
                        f"not exist. A participant runs in a directory that "
                        f"ALREADY holds its script; this call does not create "
                        f"one, because an empty directory turns a wrong path "
                        f"into a missing script."
                        + (f" These participant directories DO exist: "
                           f"{_present}." if _present else
                           " None of this call's participant directories "
                           "exist.")})
                _bin = _solver_binary_as_participant(s)
                if _bin:
                    return json.dumps({"error": _bin})
                parts.append(Participant(name=s["name"], command=list(s["command"]),
                                         work_dir=wd, imports_from=s.get("imports_from", []),
                                         timeout=int(s.get("timeout", 3600)),
                                         data_files=list(s.get("data_files", [])),
                                         env=(dict(s["env"]) if isinstance(s.get("env"), dict) else None)))
            except (KeyError, TypeError) as e:
                return json.dumps({"error": f"bad participant spec {s!r}: {e}"})
            except OSError as e:
                # `/testbed` raised PermissionError straight out of the tool.
                return json.dumps({"error":
                    f"participant {s.get('name','?')}: work_dir "
                    f"{s.get('work_dir','?')!r} cannot be used: "
                    f"{type(e).__name__}: {e}"})
        if not (0.0 < theta <= 1.0):
            return json.dumps({"error": f"theta must be in (0, 1], got {theta}"})
        # The driver compares this string with ==, so "Aitken" or a typo used to
        # fall through to constant relaxation and run a different algorithm than
        # the one asked for, silently.
        accelerator = str(accelerator).strip().lower()
        if accelerator not in ("auto", "aitken", "anderson", "constant"):
            return json.dumps({"error": f"accelerator must be 'auto', 'aitken', 'anderson' or "
                                        f"'constant', got {accelerator!r}"})
        # The stochastic branch is opt-in and its arguments are checked here
        # rather than in the driver, so a bad one is a message instead of a run
        # that quietly behaves like the default.
        if noise_replicates and noise_replicates < 2:
            return json.dumps({"error":
                f"noise_replicates must be 0 (off) or >=2 — a floor is "
                f"measured BETWEEN independent replicate runs, so one run "
                f"cannot produce one. Got {noise_replicates}."})
        if noise_floor < 0.0:
            return json.dumps({"error": f"noise_floor must be >= 0, got "
                                        f"{noise_floor}"})
        if noise_block < 1:
            return json.dumps({"error": f"noise_block must be >= 1, got "
                                        f"{noise_block}"})
        # FIRST-CONTACT PREFLIGHT: the script a command names must exist
        # BEFORE anything runs. Measured: one session called this tool
        # EIGHT times, every call dying on "can't open file .../
        # participant_A.py: No such file or directory" — a work_dir/command
        # path mismatch it could have fixed in one step had the reply shown
        # what the directory actually holds. Checked here, not left to the
        # subprocess: an informed refusal beats eight blind retries.
        _missing = []
        for _p in parts:
            _wd = Path(_p.work_dir)
            _staged = {Path(x).name for x in
                       (getattr(_p, "data_files", None) or [])}
            _prev = ""
            for _arg in _p.command[1:]:
                _flagged = _prev in ("-c", "-e", "-m", "-x")
                _prev = str(_arg)
                if _flagged:
                    continue          # inline code / module, not a file path
                if not str(_arg).endswith((".py", ".sh", ".jl", ".cc")):
                    continue
                if Path(_arg).name in _staged:
                    continue          # the driver stages data_files itself
                _cand = Path(_arg)
                if not _cand.is_absolute():
                    _cand = _wd / _cand
                if not _cand.is_file():
                    try:
                        _have = sorted(q.name for q in _wd.iterdir())[:20]
                    except OSError:
                        _have = ["<work_dir not listable>"]
                    _missing.append(
                        f"participant {_p.name}: command references "
                        f"{_arg!r} which resolves to {_cand} and DOES NOT "
                        f"EXIST. Its work_dir {_wd} holds: {_have}")
        if _missing:
            return json.dumps({
                "error": "participant script(s) not found — nothing was run",
                "detail": _missing,
                "fix": ("Point each command at a script that exists: either "
                        "an ABSOLUTE path, or a path relative to that "
                        "participant's work_dir (the command runs WITH "
                        "work_dir as its current directory). Write the "
                        "script first, then call couple again.")},
                indent=2)
        try:
            _level_secs = None
            _t_level = __import__('time').perf_counter()
            _t_level_wall = __import__('time').time()      # the dumps this call writes are newer
            r = run_coupling(parts, max_iter=max_iter, tol=tol,
                             accelerator=accelerator, theta0=theta, probe=probe,
                             noise_replicates=int(noise_replicates),
                             noise_floor=(float(noise_floor) or None),
                             noise_block=int(noise_block))
            _level_secs = __import__('time').perf_counter() - _t_level
        except Exception as exc:                      # never raise out of a tool
            return json.dumps({"converged": False,
                               "error": f"coupling driver failed: "
                                        f"{type(exc).__name__}: {exc}"}, indent=2)

        history_file = None
        if history_path:
            destination = Path(history_path)
            if not destination.is_absolute():
                # A RELATIVE PATH MEANS THE TASK DIRECTORY, NOT THE SERVER'S.
                # This used to refuse ("history_path must be absolute"), for the
                # sound reason that a relative path would resolve against the
                # server's own cwd, which the agent cannot see. The refusal sat in
                # a field of a long reply that otherwise said converged, so the
                # driver silently wrote no history: 25 couple() calls and one
                # couple_levels(history_dir='.') in the recorded runs. The
                # measured cost was a complete three-level coupling whose agent
                # then wrote the residual files itself, from a different
                # normalisation, and handed in evidence contradicting the
                # driver's own verdict. Resolve against the task directory when
                # it is known, else beside the participants (couple_levels' own
                # default); the escape check below still refuses `../`.
                _base = (cell_root if cell_root is not None
                         else Path(specs[0]["work_dir"]).resolve().parent)
                destination = _base / destination
            if (cell_root is not None
                  and not destination.resolve().is_relative_to(cell_root)):
                history_file = {
                    "path": str(destination),
                    "error": (f"history_path is outside this task's working "
                              f"directory {cell_root}")}
            else:
                try:
                    import math as _math
                    rows = [(iteration, float(value))
                            for iteration, value in enumerate(r.history, 1)
                            if _math.isfinite(float(value))]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    # A RUN THAT PRODUCED NO RESIDUAL DOES NOT ERASE ONE THAT DID.
                    # A coupling that died at iteration 1 wrote a header-only file
                    # over the previous run's history (measured: 30 rows lost). The
                    # earlier file is kept beside it before an empty one replaces it.
                    if not rows and destination.is_file():
                        try:
                            _old_rows = len(destination.read_text().splitlines()) - 1
                        except OSError:
                            _old_rows = 0
                        if _old_rows > 0:
                            destination.replace(destination.with_name(
                                destination.stem + ".previous" + destination.suffix))
                    tmp = destination.with_suffix(destination.suffix + ".tmp")
                    tmp.write_text(
                        "iteration,interface_residual\n" +
                        "".join(f"{iteration},{value:.17g}\n"
                                for iteration, value in rows))
                    tmp.replace(destination)
                    history_file = {
                        "path": str(destination),
                        "rows_written": len(rows),
                        "nonfinite_omitted": len(r.history) - len(rows),
                        "driver_iterations": r.iterations,
                        "iteration_column": (
                            "original driver iteration numbers; iteration 1 "
                            "is absent because no residual exists before a "
                            "previous iterate"),
                    }
                except (OSError, TypeError, ValueError) as exc:
                    history_file = {
                        "path": str(destination),
                        "error": f"{type(exc).__name__}: {exc}"}

        # YOUR OWN VALIDATED INTERFACE DATA, RETURNED READY TO SAVE.
        # Twelve rounds of recorded runs measured the same terminal failure:
        # the coupling converges (proven to have run at every level) and the
        # agent then RE-DERIVES its interface files by hand -- first-order
        # recoveries, self-chosen probe coordinates, negated echoes -- and the
        # re-derivation, not the coupling, decides whether the result is right.
        # The serving
        # ladder (rule as text, function as text, function in the primary
        # door) converted 0 of 9 scripts. So the reply now carries each
        # participant's OWN final exports.json data -- the numbers its own
        # solver produced, already validated by the checks in this verdict --
        # as ready-to-save CSV text. openPASO writes no file and computes no
        # number here; the agent saves its own data verbatim.
        # EACH SIDE'S SOLVER CONSOLE, KEPT PER LEVEL. The driver retains each
        # participant's captured console in participant_output.log, which the
        # next level overwrites; measured on four cells, the per-level run
        # logs were then all written from the finest level's output and every
        # level carried the same DOF count. When the call names a level (the
        # iface_level argument, or a history_path like residual_level<k>.csv),
        # the same console is also kept as participant_output_level<k>.log.
        _lvl_for_log = int(iface_level) if iface_level else 0
        if not _lvl_for_log and history_path:
            _m_lvl = re.search(r"level[_-]?(\d+)", Path(history_path).name)
            if _m_lvl:
                _lvl_for_log = int(_m_lvl.group(1))
        if not _lvl_for_log:
            # AND FAILING BOTH, ASK THE PARTICIPANTS. The served per-level rule
            # has every side read its level from a ./config.json beside its
            # script, so by the time this runs the level is already on disk in a
            # place this contract defines -- there is no reason to lose the
            # console because the CALL did not repeat it.
            #
            # MEASURED on a coupled cell that reached three refined levels with
            # both codes and the coupling PROVEN: it called couple with no level
            # and no level-bearing history path, so no participant_output_level
            # file was ever kept, it had exactly ONE console to hand in, and its
            # three run logs came out byte-identical. Its field files refined
            # correctly the whole time (64 -> 215 -> 796 and 73 -> 256 -> 958);
            # the submission was read as one mesh solved three times.
            _lvls = set()
            for _p in parts:
                try:
                    _cfg = Path(_p.work_dir) / "config.json"
                    if _cfg.is_file():
                        _v = json.loads(_cfg.read_text() or "{}").get("level")
                        if _v is not None:
                            _lvls.add(int(_v))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    pass
            if len(_lvls) == 1:            # both sides agree; ambiguity is not a level
                _lvl_for_log = _lvls.pop()
        # ONLY A CONSOLE THIS CALL'S RUNS WROTE IS THIS LEVEL'S. Measured: side A stopped at
        # iteration 1, side B never ran, and B's console -- still the previous level's -- was
        # kept as this level's and read as "MESH UNCHANGED FROM LEVEL k on B". The driver's
        # exit codes name every side that ran in this call.
        _ran_here = set()
        for _p in parts:
            try:
                _src = Path(_p.work_dir) / "participant_output.log"
                if (_p.name in (getattr(r, "returncodes", None) or {}) and _src.is_file()
                        and _src.stat().st_mtime >= _t_level_wall - 1.0):
                    _ran_here.add(_p.name)
            except OSError:
                pass
        if _lvl_for_log:
            for _p in parts:
                if _p.name not in _ran_here:
                    continue
                try:
                    _src = Path(_p.work_dir) / "participant_output.log"
                    (Path(_p.work_dir) / f"participant_output_level{_lvl_for_log}.log").write_text(
                        _src.read_text(errors="replace"))
                except OSError:
                    pass
        # THE MESH MUST CHANGE BETWEEN LEVELS. Measured (round 27, a cell with time for three
        # levels): the same 693 / 957 dofs on all three, graded as no refinement at all. The
        # captured consoles carry each side's own NDOF line, so the tool compares this level's
        # with the previous level's and says so BEFORE the agent moves on. Reads the agent's own
        # files; changes nothing.
        _mesh_note = ""
        if _lvl_for_log and _lvl_for_log >= 2:
            try:
                # any "NDOF = n" on a console line (the run-log contract's bare line is the audit's business)
                _dof_any = re.compile(r"^\s*NDOF\s*=\s*(\d+)\s*$", re.M)   # the canonical contract line, FIRST match
                _same = []
                for _p in parts:
                    if _p.name not in _ran_here:
                        continue           # a side that did not run here has no console of this level
                    _cur = Path(_p.work_dir) / f"participant_output_level{_lvl_for_log}.log"
                    _prev = Path(_p.work_dir) / f"participant_output_level{_lvl_for_log - 1}.log"
                    if _cur.is_file() and _prev.is_file():
                        _mc = _dof_any.findall(_cur.read_text(errors="replace"))
                        _mp = _dof_any.findall(_prev.read_text(errors="replace"))
                        if _mc and _mp and _mc[0] == _mp[0]:
                            _same.append(f"{_p.name} (NDOF {_mc[0]} at both levels)")
                if _same:
                    _mesh_note = ("MESH UNCHANGED FROM LEVEL " + str(_lvl_for_log - 1) + " on " + ", ".join(_same)
                                  + ": this level is NOT a refinement and counts as not run. Halve h -- double nx "
                                  "and ny in that side's config; under couple_levels they arrive in the environment "
                                  "variable OPENPASO_CONFIG_JSON, which the script must merge over its ./config.json and "
                                  "build its mesh from -- and couple level " + str(_lvl_for_log)
                                  + " again before anything else.")
            except Exception:                                # noqa: BLE001
                _mesh_note = ""
        iface_csv = None
        if r.converged and (r.exports or {}):
          try:                       # a malformed export must not destroy
            iface_csv = {}           # a converged reply (reviewer finding)
            lvl = int(iface_level) if iface_level else 0
            suffix = f"_level{lvl}" if lvl else ""
            _inline_left = _IFACE_CSV_INLINE_MAX
            for pname, data in sorted((r.exports or {}).items()):
                co = data.get("coordinates") or []
                vals = data.get("values") or []
                qs = data.get("normal_fluxes") or []
                n = len(co)
                if not n or n > 500:
                    continue
                def _flatrow(v):
                    if isinstance(v, (list, tuple)):
                        return [float(x) for x in v]
                    return [float(v)]
                rows = []
                for i in range(n):
                    cells = [f"{x:.15e}" for x in _flatrow(co[i])]
                    if vals and i < len(vals):
                        cells += [f"{x:.15e}" for x in _flatrow(vals[i])]
                    if qs and i < len(qs):
                        cells += [f"{x:.15e}" for x in _flatrow(qs[i])]
                    rows.append(", ".join(cells))
                ncoord = len(co[0]) if co and co[0] else 2
                hdr = ", ".join(list("xyz"[:ncoord])
                                + (["u"] if vals else [])
                                + (["qn"] if qs else []))
                _text = hdr + chr(10) + chr(10).join(rows) + chr(10)
                _block = {"suggested_filename": f"interface{suffix}_{pname}.csv"}
                if len(_text) > _inline_left:
                    iface_csv[pname] = _iface_block_not_inlined(
                        pname, _block, len(_text), n, rows[0].count(",") + 1 if rows else None)
                    continue
                _inline_left -= len(_text)
                iface_csv[pname] = {**_block, "csv": _text}
            if iface_csv:
                iface_csv["_how_to_use"] = (
                    "Each block is that participant OWN interface data at "
                    "convergence, validated by this verdict. ONE DECISION "
                    "RULE: IF your participants exported AT the exact probe "
                    "points your task prescribes, save each csv field "
                    "verbatim with one write_file call and you are done. IF "
                    "they exported at their own mesh nodes, do NOT save "
                    "these blocks as your reported interface files -- "
                    "follow the reporting key in this same reply, which "
                    "interpolates each side converged solution onto the "
                    "prescribed points with its own material and outward "
                    "normal. Never retype numbers either way. The simplest "
                    "route is to export at the prescribed probe points "
                    "INSIDE your participant so these blocks carry them "
                    "directly.")
          except Exception as _e:                     # noqa: BLE001
            iface_csv = {"error": f"interface blocks not built: {_e}"}

        # ── silent-wrong validators ─────────────────────────────────
        # `val` holds findings (they flip the verdict); `not_run` holds checks
        # that could not look at anything (they never flip the verdict, and they
        # are printed in it, so "not checked" is never read as "checked and fine").
        val = list(r.warnings)
        not_run: list[str] = [_DIGEST_SCOPE_LIMIT]
        # THE CRITERION THE RUN WAS HELD TO goes in the COVERAGE channel, never
        # in the findings one. "Judged at the measured noise floor instead of at
        # your tol" must be printed — an agent that judges tighter than the
        # floor is judging the sampler — but it is not a fault, and `val` is the
        # list where anything at all means the coupling cannot be trusted. Left
        # in `val` it stamps NOT VERIFIED on every correct stochastic coupling,
        # measured: a converged run at a floor of 1.0e-02 came back "NOT
        # VERIFIED — the coupling did not converge, or failed one of openPASO's
        # silent-wrong checks", which is the verdict the whole branch exists to
        # stop being unavoidable. The driver hands these over in their own list
        # rather than being pattern-matched out of `warnings`, so a reworded
        # message cannot silently start or stop flipping the verdict.
        not_run += list(r.criterion_notes)
        # Judge against the criterion the run was actually held to. Quoting the
        # requested `tol` in a NOT CONVERGED message on a run that was judged
        # against a measured noise floor would name a threshold the driver never
        # applied.
        val += check_convergence(r.converged, r.residual,
                                 r.tol_effective if r.tol_effective else tol)
        for nm, ex in r.exports.items():
            val += check_finite(ex.get("values", []), label=f"{nm}.values")
            if ex.get("normal_fluxes") is not None:
                val += check_finite(ex["normal_fluxes"], label=f"{nm}.normal_fluxes")
            val += check_finite(ex.get("coordinates", []), label=f"{nm}.coordinates")
        f, n = check_returncodes(r.returncodes); val += f; not_run += n
        f, n = check_coupling_directionality(r.graph, max_iter); val += f; not_run += n
        f, n = check_participant_responsiveness(r.responsiveness,
                                                getattr(r, "responsiveness_detail", None),
                                                getattr(r, "sensitivity", None)); val += f; not_run += n
        if probe:
            # The measured floor is handed over, so the one branch that cannot
            # tell "stochastic" from "hidden state" reports coverage instead of
            # a finding when the run has an established floor. The branch that
            # catches a participant ignoring its imports outright (S below the
            # response floor) is unaffected and stays a finding either way.
            f, n = check_interface_sensitivity(r.sensitivity,
                                               noise_floor=r.noise_floor,
                                               exports=r.exports, graph=r.graph)
            val += f; not_run += n
        else:
            not_run.append(
                "interface sensitivity: NOT probed (probe=False). The one solve "
                "that would have established whether each participant's answer "
                "depends on its imports was skipped, so a solver that ignores "
                "imports.json is indistinguishable here from one that reads it.")
        # Only interesting when the GLOBAL norm claims convergence: that is the
        # case where a still-moving small block is invisible. When the global
        # residual already says NOT CONVERGED it says everything, and repeating
        # it per block just buries the findings that are specific.
        #
        # A MEASURED FLOOR DOES NOT TRANSFER TO THIS STATISTIC, so under one
        # this check reports coverage rather than a verdict. Two goes were
        # needed to get that right and the wrong one is worth recording. It
        # first compared the blocks against the requested `tol`, which faulted a
        # run held to 8.3e-03 for blocks moving by more than 1.0e-08. Feeding it
        # `tol_effective` instead looked like the fix and is not: the driver
        # measures the floor of ONE statistic, the global L2 residual over the
        # stacked export vector, while a block residual is the WORST ENTRY-WISE
        # relative change of one block. Those have different noise floors and
        # the second is far larger — a DSMC surface flux has entries near zero
        # whose relative change between samples is order 1, measured at
        # gas.normal_fluxes = 1.00e+00 against an effective limit of 6.9e-01 on
        # a coupling that is right.
        #
        # So there is no threshold here that means anything, and inventing one
        # would be the softening this whole branch refuses. Say what is missing
        # instead: scale masking was not ruled out, and it would take a per-block
        # floor that nothing measures.
        if r.converged and r.noise_floor:
            not_run.append(
                "per-block convergence: NOT CHECKED on a run judged at a "
                f"measured noise floor ({r.noise_floor:.2e}). That floor is for "
                "the GLOBAL residual — one L2 norm over the stacked export "
                "vector — while this check compares the worst ENTRY-WISE "
                "relative change of each block, a statistic with its own, much "
                "larger floor that nothing here measures. Comparing the two "
                "faults correct stochastic couplings, and picking a threshold "
                "would be guessing. Consequence: scale masking in the global "
                "residual has NOT been ruled out for this run — the per-block "
                "residuals are returned in `block_residuals`, and the decisive "
                "check remains an independent reference (`monolithic`)."
                + " Blocks: "
                + ", ".join(f"{k}={v:.2e}" for k, v in
                            sorted(r.block_residuals.items()) if v == v))
        elif r.converged:
            # A BLOCK NO SIDE READS DOES NOT DECIDE THE FIXED POINT. On a Dirichlet-
            # Neumann pair the Dirichlet side reads its partner's trace and the Neumann
            # side its partner's flux; the Dirichlet side's own exported trace is read by
            # no one and lags the iteration by one step. Measured: a right ladder was
            # denied by a caveat on exactly that block. Judged only where both sides
            # state their roles.
            _unread = _unread_blocks([(p.name, p.work_dir) for p in parts])
            # (a block of one column per point is named "<side>.<array>[<column>]": the column
            # belongs to its array -- measured, "B.normal_fluxes[k]" never matched and was judged)
            _unread_key = lambda _k: re.sub(r"\[\d+\]$", "", _k) in _unread      # noqa: E731
            _blocks = {k: v for k, v in (r.block_residuals or {}).items() if not _unread_key(k)}
            f, n = check_residual_blocks(_blocks or r.block_residuals, tol,
                                         fixed_point=getattr(r, "block_fixed_point", None),
                                         distance=getattr(r, "block_distance", None))
            val += f; not_run += n
            _skipped = {}
            for _k in (r.block_residuals or {}):
                if _unread_key(_k):
                    _a = re.sub(r"\[\d+\]$", "", _k)
                    _skipped[_a] = _skipped.get(_a, 0) + 1
            _skipped = [a + (f" ({n} columns)" if n > 1 else "") for a, n in sorted(_skipped.items())]
            if _skipped and _blocks:
                not_run.append("per-block convergence: " + ", ".join(_skipped) + " not judged -- read by no "
                               "side (a Dirichlet side reads its partner's trace, a Neumann side its partner's "
                               "flux), so their last step does not decide the fixed point")
        names = list(r.exports)
        _bal_numbers: dict = {}
        _prof_numbers: dict = {}
        if len(names) == 2:
            a, b = r.exports[names[0]], r.exports[names[1]]
            f, n = check_interfaces_are_the_same_surface(a, b, names[0], names[1])
            val += f; not_run += n
            f, n = check_interface_meshes(a, b, names[0], names[1]); val += f; not_run += n
            f, n = check_interface_flux_profile(a, b, names[0], names[1],
                                                numbers=_prof_numbers)
            val += f; not_run += n
            if a.get("normal_fluxes") is None or b.get("normal_fluxes") is None:
                not_run.append(
                    "interface flux balance: at least one participant exported no "
                    "`normal_fluxes`, so conservation across the interface was NOT "
                    "checked. Export the normal flux from both sides (each w.r.t. "
                    "its own outward normal) to enable the only conservation "
                    "evidence available here.")
            else:
                # A CHECK THAT CANNOT SPEAK IS COVERAGE, NOT A FINDING. On a
                # Dirichlet-Neumann pair the two exported fluxes are exact
                # opposites by construction (the Neumann side's consistent
                # recovery returns the flux it was given), and the check says
                # so instead of passing them. That sentence names a gap in
                # what THIS reply can verify, so it goes where the other
                # coverage gaps go; it must not make a level unverifiable for
                # a comparison no pair of this kind can ever offer.
                for _m in check_interface_balance(a, b, names[0], names[1],
                                                  numbers=_bal_numbers):
                    (not_run if ("cannot disagree" in _m or _m.startswith(_ENDS_ONLY_MARK)
                                 or "(the exchange is at round-off)" in _m) else val).append(_m)
        elif len(names) > 2:
            not_run.append(
                f"interface flux balance: {len(names)} participants — the pairwise "
                "conservation check only applies to a 2-participant interface, so "
                "conservation was NOT checked.")

        # ── monolithic consistency: the decisive silent-wrong detector ────────
        mono_block, f, n = _run_monolithic_check(monolithic, r.exports)
        val += f; not_run += n

        # `val` is the FINDINGS channel and `not_run` the COVERAGE channel, and
        # every check above puts its output in exactly one of them. So any
        # finding at all means the coupling cannot be trusted — no substring
        # matching on message text, which would silently stop working the moment
        # a message was reworded, and could equally raise a false alarm on a
        # benign sentence that happened to contain one of the keywords.
        # A non-matching interface is deliberately a coverage note rather than a
        # finding: it is a legitimate configuration whose consequence (unchecked
        # conservation) belongs in the coverage list, and whose failure mode is
        # caught by the flux balance.
        # A coupling in which nothing is exchanged converges instantly with a
        # zero residual and passes every other check — the most convincing
        # silent-wrong result this tool can produce. Name it. This runs BEFORE
        # the verdict is taken, and it splits its two outcomes deliberately:
        # NOT COUPLED is a finding and flips the verdict; ONE-WAY is a
        # legitimate master->slave configuration, so it goes to `not_run`,
        # where it is always printed and never flips anything.
        #
        # It used to append both to `val` and then RECOMPUTE checks_ok with a
        # keyword filter so that ONE-WAY would not flip it. That recomputation
        # silently discarded every finding from the validators above — the
        # same-surface, flux-profile, mesh, returncode, directionality,
        # responsiveness and monolithic checks all landed in `validation` and
        # none of them could make a coupling untrustworthy.
        deaf = [p.name for p in parts if not p.imports_from]
        if len(deaf) == len(parts):
            val.append(
                "NOT COUPLED: no participant lists `imports_from`, so none of "
                "them ever receives partner data. Nothing was exchanged and the "
                "run is not a coupling at all.")
        elif deaf:
            not_run.append(
                f"ONE-WAY: participant(s) {', '.join(deaf)} list no "
                f"`imports_from` and so never see their partners' data. That is "
                f"a legitimate master->slave coupling, so it is reported rather "
                f"than failed — but if the coupling is meant to be two-way, that "
                f"is the bug.")
        # The third branch that used to sit here — "converged in <= 2 iterations
        # with an exactly zero residual, therefore NOT COUPLED" — is deleted, not
        # moved. It is a real signal (a participant that ignores imports.json
        # settles instantly) but it cannot tell that case apart from a coupling
        # that is simply easy: a linear Dirichlet-Neumann pair with no source
        # lands on the interface value immediately, which the coupling knowledge
        # states in as many words. It failed exactly those correct couplings.
        # feature/coupling-robustness replaced the heuristic with two checks that
        # can actually discriminate — probe_interface_sensitivity perturbs an
        # import and watches whether the export moves, and
        # check_participant_responsiveness catches a byte-identical export — and
        # its own tests pin the distinction: with BOTH stubbed out, a do-nothing
        # participant is expected to verify, because those two are what detect it.
        checks_ok = r.converged and not val
        # DO NOT SPEND THE AGENT'S CONTEXT ON DATA IT ALREADY HAS ON DISK.
        #
        # `exports` was returned in full. Measured: at a level-3 interface the
        # whole return is 290,512 chars and `exports` is 143,813 of them —
        # 49.5% — while the verdict sat at char 287,419, i.e. 98.9% of the way
        # in. Every one of those arrays is already in the participant's own
        # work_dir/exports.json, which the agent wrote and can read.
        #
        # Why it matters more than it looks: a run with these tools gets a
        # median 39 tool calls per coupled run against 95 without them, at 91k
        # input tokens per call against 60k, and stops at 39% of the wall clock
        # with 5.8% timeouts against 19.5%. It runs out of ACTIONS, not
        # time — 60% write no output file at all against 31%. Text we
        # added to help (the deliverables-first imperative, the NaN rule) is
        # worth nothing while it arrives at 98.9% of a 290 kB payload.
        #
        # So: a summary per participant plus the path, and the verdict first.
        def _export_summary(exports):
            if not isinstance(exports, dict):
                return exports
            out = {}
            for name, ifd in exports.items():
                d = ifd if isinstance(ifd, dict) else {}
                vals = d.get("values") or []

                def _num(v):
                    # a multi-component exchange (temperature and displacement
                    # through one interface) carries a LIST per point; float()
                    # on it crashed every couple() call of the thermo-elastic
                    # cells (measured 2026-09-11: twelve calls in one cell)
                    if isinstance(v, (list, tuple)):
                        return [float(x) for x in v]
                    return float(v)
                out[name] = {
                    "field_name": d.get("field_name"),
                    "n_points": d.get("n_points", len(vals)),
                    "has_normal_fluxes": bool(d.get("normal_fluxes")),
                    "first_values": [_num(v) for v in vals[:3]],
                    "last_values": [_num(v) for v in vals[-3:]],
                    "full_arrays_are_on_disk":
                        f"<participant {name}'s work dir>/exports.json",
                }
            return out

        result = {"verification": None,          # filled by _stamp_verification
                  "validation": val, "checks_not_run": not_run,
                  # the balance numbers travel so a ladder can state their trend
                  "interface_balance": _bal_numbers or None,
                  "interface_profile": _prof_numbers or None,
                  "error": r.error,
                  "converged": r.converged, "iterations": r.iterations,
                  "residual": r.residual, "history": r.history,
                  "block_residuals": r.block_residuals,
                  "returncodes": r.returncodes,
                  "responsiveness": r.responsiveness,
                  **({"responsiveness_detail": r.responsiveness_detail}
                     if getattr(r, "responsiveness_detail", None) else {}),
                  "graph": r.graph, "relaxation": r.theta,
                  "interface_sensitivity": r.sensitivity,
                  "participant_output_logs": {
                      p.name: str(p.work_dir / "participant_output.log")
                      for p in parts
                      if (p.work_dir / "participant_output.log").is_file()},
                  "monolithic_check": mono_block,
                  "exports": _export_summary(r.exports)}
        if history_file is not None:
            result["history_file"] = history_file
        if iface_csv is not None:
            result["interface_csv"] = iface_csv
        # THE RESIDUAL HISTORY, READY TO SAVE. Measured: a session that had
        # CALLED this tool -- the real history sat in this very reply --
        # still hand-wrote per-level residual files containing a single
        # placeholder row of 1.0, and its otherwise-sound file set was
        # unusable. The transcription step is where placeholders creep in;
        # remove it the same way the interface tables were removed.
        try:
            import math as _math
            _hrows = [(_i, float(_v)) for _i, _v in
                      enumerate(r.history, 1) if _math.isfinite(float(_v))]
        except Exception:                                # noqa: BLE001
            _hrows = []
        if r.converged and len(_hrows) >= 2:
            _lvl = int(iface_level) if iface_level else 0
            _sfx = f"_level{_lvl}" if _lvl else ""
            result["residual_csv"] = {
                "suggested_filename": f"residual{_sfx}.csv",
                "csv": ("iteration,interface_residual" + chr(10)
                        + chr(10).join(f"{_i},{_v:.17g}"
                                       for _i, _v in _hrows) + chr(10)),
                "_how_to_use": (
                    "This is the measured iteration history of THIS "
                    "coupling, ready to save verbatim (adjust the filename "
                    "to your task naming). Never retype it and never write "
                    "a placeholder history: a single-row or constant "
                    "residual file reads as no coupling at all.")}
        # THE CAPTURED SOLVER LOGS ALREADY EXIST — POINT AT THEM. Measured
        # on a six-run read: the top kill (3 of 6) was couplings with
        # proven runs delivering levels without their captured solver
        # logs — reconstructing what the driver had already persisted. Each
        # participant's latest native output (command, returncode, stdout,
        # stderr) is on disk in its work_dir; a per-level log is a COPY of
        # that file plus the DOF-count line, never a reconstruction.
        _logs = {}
        for _p in parts:
            _lp = Path(_p.work_dir) / "participant_output.log"
            if _lp.is_file():
                _logs[_p.name] = str(_lp)
        if _logs:
            result["captured_solver_logs"] = {
                **_logs,
                "_how_to_use": (
                    "Each path holds that side's solver output as the driver "
                    "captured it (command, returncode, stdout, stderr). If "
                    "your task requires a per-level run log with the named "
                    "code's own output, COPY this file (plus the DOF-count "
                    "line your task asks for) -- do not retype or reconstruct "
                    "it. A level delivered without its captured log counts as "
                    "not run, however real the numbers beside it are.")}
        if r.noise_floor is not None:
            result["noise_floor"] = r.noise_floor
            result["tol_effective"] = r.tol_effective
            result["stopped_at_noise_floor"] = r.stopped_at_noise_floor
        if r.notes:
            # How the floor was measured, and the fixed-seed caveat. Kept out of
            # `validation` so a correct run's validation block stays empty, but
            # returned, because a floor of exactly 0 on a Monte-Carlo code is a
            # fixed seed and the agent has to see that.
            result["noise_notes"] = r.notes
        # NAME THE CLAUSE THAT FIRED. DO NOT TELL A CONVERGED RUN IT DIVERGED.
        #
        # This was a disjunction — "the coupling did not converge, or failed
        # one of openPASO's silent-wrong checks" — printed whenever checks_ok was
        # false, including when `converged` is True in the very same payload.
        # An agent that has just driven a real coupling to tolerance, at ~30%
        # of its budget, was told its coupling may not have converged. It is
        # the cheapest possible reason to stop, and measured coupled runs do
        # stop: 63% end honestly incomplete, at a median 30% of budget, with
        # zero timeouts in 112 runs.
        #
        # Section 3b of the served coupling text already says a converged run
        # with a failed downstream check IS a result — but it sits ~700 lines
        # away in a different payload from this message. So the operative
        # sentence is repeated here, where the agent is actually reading.
        reason = _couple_failure_reason(r, checks_ok)
        # A SIDE WHOSE OWN FIELD IS NOT FINITE IS NOT A RESULT, whatever its exports
        # say. Measured: a side whose form assembled to zero exported finite
        # placeholders over a NaN field, and this reply called the level "CONVERGED --
        # THIS IS A RESULT, SAVE IT NOW".
        try:
            from . import result_audit as _ra_nf
            _nf = _ra_nf.nonfinite_field_findings(
                Path(str(parts[0].work_dir)).parent, dirs=[Path(str(p.work_dir)) for p in parts],
                since=_t_level_wall - 1.0, scan=False)
        except Exception:                                    # noqa: BLE001
            _nf = []
        if _nf:
            checks_ok = False
            reason = _at_sentence(str(_nf[0].get("finding", "")), 400)
            val = list(val) + [str(f.get("finding", "")) for f in _nf]
            result["validation"] = val
        # NOR IS A SIDE THAT NEVER SOLVED ITS INTERIOR. Measured: every level of a
        # ladder whose side held exactly 0.0 at every interior node read
        # trustworthy_result true, under a ladder verdict that named the side.
        try:
            _us = _ra_nf.unsolved_field_findings(
                Path(str(parts[0].work_dir)).parent, dirs=[Path(str(p.work_dir)) for p in parts],
                since=_t_level_wall - 1.0, scan=False)
        except Exception:                                    # noqa: BLE001
            _us = []
        if _us:
            checks_ok = False
            reason = reason if _nf else _at_sentence(str(_us[0].get("finding", "")), 400)
            val = list(val) + [str(f.get("finding", "")) for f in _us]
            result["validation"] = val
        # NOR A SIDE WHOSE HELD BOUNDARY IS NOT WHAT ITS OWN CONFIG STATES, A LEVEL
        # RUN ON THE PREVIOUS LEVEL'S MESH, OR A LEVEL WHOSE SIDES EXCHANGED NOTHING.
        # Measured: a level whose side left 11 of 33 nodes of a held edge free, and
        # one whose interface corners were free, read trustworthy_result true while
        # the same reply's audit named them; a level whose sides exchanged nothing
        # was told to save itself as a result.
        _lvl_here = int(iface_level) if iface_level else (_lvl_for_log or None)
        _bf = []
        try:
            _root_here = Path(str(parts[0].work_dir)).parent
            _dirs_here = [Path(str(p.work_dir)) for p in parts]
            _bf += [f for f in _ra_nf.outer_boundary_findings(
                        _root_here, levels=[_lvl_here] if _lvl_here else None, dirs=_dirs_here)
                    if not f.get("informational")]
            _bf += [f for f in _ra_nf.free_interface_end_findings(
                        _root_here, dirs=_dirs_here, levels=[_lvl_here] if _lvl_here else None)
                    if not f.get("informational") and int(f.get("priority", 99)) <= 10]
        except Exception:                                    # noqa: BLE001
            _bf = []
        # NOR A SIDE WHOSE OWN DISCRETE SYSTEM CONDEMNS THIS LEVEL'S FIELD: A u one factor other than
        # one times the load of the source its config states. Measured: a side whose load left out its
        # test function read A u = 3.87 times the load at level 1, and this level read "VERIFIED ...
        # SETTLED" because the equation check waited for more levels.
        try:
            _bf += _ra_nf.level_equation_findings(
                Path(str(parts[0].work_dir)).parent, dirs=[Path(str(p.work_dir)) for p in parts],
                since=_t_level_wall - 1.0)
        except Exception:                                    # noqa: BLE001
            pass
        if _mesh_note:
            _bf.append({"finding": _mesh_note})
        _nc = _level_not_a_coupled_result(result) if r.converged else ""
        # (an exactly zero last residual is judged by the ladder and the audit, not here: a
        # deterministic pair whose accelerated step lands on its fixed point reproduces its
        # data bit for bit, and a single call cannot tell that from an exchange of nothing)
        if _nc.startswith("the reported interface residual is exactly zero"):
            _nc = ""
        if _nc:
            _bf.append({"finding": f"THIS LEVEL IS NOT A COUPLED RESULT YET: {_nc}"})
        if _bf:
            checks_ok = False
            reason = reason if (_nf or _us) else _at_sentence(str(_bf[0].get("finding", "")), 400)
            val = list(val) + [str(f.get("finding", "")) for f in _bf]
            result["validation"] = val
        # A LEVEL WHOSE ONLY FINDINGS ARE PER-BLOCK CAVEATS carries an iteration caveat, named
        # by the caveat itself (the generic converged-with-a-caveat reason frames it as
        # discretisation).
        _iter_caveat = (bool(r.converged) and bool(val) and not (_nf or _us or _bf)
                        and all(_BLOCK_CAVEAT in str(v) for v in val))
        if _iter_caveat:
            reason = _at_sentence(str(val[0]), 400)
        _stamp_verification(result, evidence_ok=checks_ok, reason=reason,
                            critic_approved=critic_approved,
                            field_defect=bool(_nf or _us or _bf),
                            iteration_caveat=_iter_caveat,
                            solver="couple",
                            setup_text=_coupling_setup_text(
                                participants=participants, max_iter=max_iter,
                                tol=tol, accelerator=accelerator, theta=theta,
                                monolithic=monolithic, probe=probe))
        if not_run:
            result["verification"] += (
                " COVERAGE — these checks could NOT run on this coupling, so the "
                "verdict above does not cover what they would have caught: "
                + " | ".join(not_run))
        # THE EXCHANGE POINTS ARE NOT THE REPORT POINTS, SAID WHERE THE AGENT
        # IS HOLDING THEM.
        #
        # MEASURED across every coupled recorded run: 11 result sets held an
        # interface file containing a point the task explicitly excludes, and 10
        # of the 11 were runs with these tools — 13.7% of coupled runs with the
        # tools against 1.2% of runs without them. Nine of them wrote exactly 9
        # rows, the level-1 mesh nodes at h = 1/8.
        #
        # The cause is this tool's own contract. Each participant writes
        # exports.json with `coordinates`, `values` and `normal_fluxes` at its
        # interface NODES, because that is what the exchange needs. Then the
        # deliverable asks for a fixed list of points that are deliberately NOT
        # nodes — and copying the file across is one line. One development
        # run's interface file is exports.json verbatim, down to the float noise
        # -2.6927850894701087e-18 where the node sits at y = 0. A run without
        # these tools cannot make this mistake because it has no such file.
        #
        # The corpus does say "the points you exchange are not the points you
        # report", ~700 lines away in a different payload. Saying it here, in
        # the reply that accompanies the data, is the only place it arrives at
        # the moment of the temptation.
        result["reporting"] = (
            "THE POINTS EXCHANGED ABOVE ARE NOT THE POINTS YOU REPORT. Each "
            "participant's exports.json holds values at ITS OWN interface "
            "NODES, which is what the exchange needs. If your task prescribes "
            "interface output at a fixed list of coordinates, those are "
            "deliberately not nodes: interpolate each side's converged "
            "solution onto the listed points, with that side's own material "
            "and its own outward normal. Copying exports.json into a "
            "per-level interface file is a common way an otherwise working "
            "coupled run is made unverifiable: the file is at the nodes, and "
            "the list is not. A task that lists interior "
            "points only has excluded the interface ENDS on purpose; do not "
            "complete the list with them.")

        # ── ALWAYS-RUN CORRECTIVE FUNNEL: lead the reply with the next fix ────
        # The driver verdict above says whether the ITERATION was sound; it does
        # not say whether the two PHYSICS match at the seam or whether the files
        # the agent will deliver are self-consistent. Those are computed here —
        # LIVE from this run's own exports (flux cancellation, field continuity)
        # and from any per-level files already on disk (identical levels, probe
        # sampling, fabricated/short residual history, missing deliverables) —
        # and the single highest-priority fix leads the reply, where a weak
        # agent that merely "acts" will read it. Advisory only: nothing here
        # touches `validation`, `verification` or `trustworthy_result`, and
        # every input is the agent's OWN output or a PUBLIC string it supplied —
        # never a reference solution. Wrapped so a fault in the funnel can never take
        # down a real coupling reply.
        _lead = None
        _compact: list = []
        try:
            from . import result_audit as _ra
            presub: list = []
            _names = list(r.exports or {})
            if len(_names) > 2:
                # SAY SO RATHER THAN BE SILENT. Every live content check here
                # is written for a PAIR, so a coupling with three or more
                # participants gets none of them -- and silence from a checker
                # is indistinguishable from a pass.
                presub.append({
                    "sequence": "interface content not checked",
                    "priority": 5, "values": [float(len(_names))], "finding": (
                        f"COVERAGE, NOT A VERDICT: this coupling has "
                        f"{len(_names)} participants and openPASO's live "
                        f"interface checks (exchange carried nothing, flux "
                        f"cancellation, field continuity) are written for a "
                        f"PAIR, so none of them ran. Nothing here says your "
                        f"interfaces are sound; it says they were not "
                        f"examined.")})
            if len(_names) == 2:
                _ea, _eb = r.exports[_names[0]], r.exports[_names[1]]
                # exchange_carried_nothing_finding goes FIRST because the
                # other two abstain on exactly its input -- both return None on
                # a zero denominator, so a pair that transmitted nothing used to
                # pass every live content check at this hook.
                for _fn in (_ra.exchange_carried_nothing_finding,
                            _ra.flux_cancellation_finding,
                            _ra.field_continuity_finding):
                    _f = _fn(_ea, _eb, _names[0], _names[1])
                    if _f:
                        presub.append(_f)
            # per-level files already written (this or an earlier couple call)
            _root = str(cell_root) if cell_root is not None else None
            if _root is None:
                try:
                    _root = os.path.commonpath([str(p.work_dir) for p in parts])
                except ValueError:
                    _root = None
            if _root:
                # couple() runs MID-coupling, often once per level, so the
                # whole-result-set completeness checks ("only 1 level yet",
                # "summary file missing") would nag about work not done yet.
                # Those belong to the audit at hand-in time; here keep the physics,
                # fabrication, identity and sampling findings that are true the
                # moment a file exists.
                _submit_only = ("level count", "level sequence",
                                "levels claimed", "deliverable completeness",
                                "summary file")
                _pending = set(_LADDER_PENDING.get())
                for _f in _ra.audit(_root).get("findings", []):
                    if any(s in str(_f.get("sequence", "")) for s in _submit_only):
                        continue
                    if str(_f.get("sequence", "")) == "no coupling history on disk":
                        continue           # this call IS the coupling; its own reply says what it wrote
                    _ml = re.search(r"level[_-]?(\d+)", str(_f.get("sequence", "")))
                    if _pending and _ml and int(_ml.group(1)) in _pending:
                        continue           # a file this ladder is about to rewrite
                    # A MEASUREMENT THE DRIVER MADE IN THIS CALL IS NOT ASKED FOR AGAIN.
                    _ft = str(_f.get("finding", ""))
                    _fm = _driver_measured(_ft, str(getattr(r, "error", "") or ""))
                    if _fm != _ft:
                        _f = dict(_f, finding=_fm)
                    presub.append(_f)
            if pde_sources:
                presub.extend(_ra.pde_source_findings(pde_sources))
            # A PARTICIPANT THAT DID NOT RUN OUTRANKS EVERYTHING ITS ABSENCE
            # CAUSED. With one side crashed the history is empty, the files
            # are missing and every downstream check fires -- but the one fix
            # is the traceback, so it leads (rank 10 in the priority table).
            # Measured: the lead said "COUPLING HISTORY TOO SHORT: 0
            # iteration(s)" while the AttributeError in side B sat three
            # fields lower, and the agent gave up.
            _rcs = getattr(r, "returncodes", None) or {}
            _err = str(getattr(r, "error", "") or "")
            _bad = sorted(n for n, c in _rcs.items() if c not in (0, None))
            if _bad or "wrote no exports.json" in _err:
                _m = re.search(r"participant (\S+) wrote no exports\.json", _err)
                _who = ", ".join(_bad) if _bad else (_m.group(1) if _m else "a participant")
                from core.coupling_driver import _stderr_tail   # noqa: PLC0415
                _tail = _stderr_tail(_err.split("stderr tail:", 1)[1].strip()
                                     if "stderr tail:" in _err else _err, 400)
                # WHAT THE FAILED SIDE LEFT ON DISK LEADS, NOT AN EMPTY STDERR TAIL.
                # Measured (e2e, 2026-09-12): a participant that wrote a deck, ran 4C
                # with its console redirected to a file and exited 1 got the lead
                # "stderr tail: ''. Fix that script" -- while the deck's defects and
                # 4C's own PROC 0 ERROR line sat in its directory. The same reader the
                # ladder uses puts them here, in the field the parent reads first.
                _disk = ""
                try:
                    from tools.result_audit import _fourc_deck_state   # noqa: PLC0415
                    for _n in (_bad or [_who]):
                        _wd = next((Path(_p.work_dir) for _p in parts if _p.name == _n), None)
                        if _wd is not None and _wd.is_dir():
                            _st = _fourc_deck_state(_wd, _wd.parent)
                            if _st:
                                _disk += f" ON DISK IN {_wd.name}: {_st['what']} {_st['brief']}"
                except Exception:                               # noqa: BLE001
                    _disk = ""
                # A STOP ON WHAT THE PARTNER EXPORTED NAMES THE PARTNER. Measured: side A's
                # trace-length guard said "partner 'B' exported a trace with 1 time levels"
                # and this lead told A to fix itself.
                _failed = set(_bad or [_who])
                _named = [_p.name for _p in parts if _p.name not in _failed
                          and re.search(r"partner\W{0,3}" + re.escape(_p.name) + r"\b", _tail)]
                _it = int(getattr(r, "iterations", 0) or 0)
                _head = (f"PARTICIPANT {_who} EXITED NON-ZERO AT ITERATION {_it} (exit codes {_rcs}): "
                         f"the coupling stopped there, so every other finding below is a consequence "
                         f"of this one." if _it > 1 else
                         f"PARTICIPANT {_who} EXITED NON-ZERO AND WAS NEVER RUN TO "
                         f"AN EXPORT (exit codes {_rcs}): nothing iterated, so every "
                         f"other finding below is a consequence of this one.")
                presub.insert(0, {"sequence": f"participant {_who}",
                                  "priority": 10, "finding": (
                    f"{_head} Its own stderr tail: {_tail!r}.{_disk} "
                    + (f"That stderr names partner {_named[0]}: {_who} stopped on what {_named[0]} "
                       f"exported. Read {_named[0]}'s exports.json and the part of its script that "
                       f"writes it before you edit {_who}, then call couple() again."
                       if _named and not _disk else
                       ("Fix the deck as named, then run that script once "
                        if _disk else "Fix that script, run it once ")
                       + "standalone until it writes exports.json, then call "
                       f"couple() again."))})
            # de-duplicate the same defect found twice (live vs file audit)
            _seen = set()
            for _f in presub:
                _k = (_f.get("finding") or "")[:80]
                if _k in _seen:
                    continue
                _seen.add(_k)
                # WITH ITS RANK: copied as sequence and text alone, a priority-3 unsolved
                # field went last behind two informational notes, and a VERIFIED ladder's
                # "SATISFIES ITS OWN EQUATION" was shown as what to fix next (measured).
                _compact.append({k: _f.get(k) for k in ("sequence", "finding", "priority",
                                                        "informational") if k in _f})
            # WHEN THE LEVEL CONVERGED AND THE FUNNEL IS CLEAN, the lead names
            # the three writes that turn a converged coupling into a result.
            # Measured: a run coupled three real levels (9 iterations to 1e-8
            # each) and never wrote a field file, then listed twelve files in
            # its summary that did not exist. Only files on disk count.
            # THE ONE-CALL MESH SEQUENCE LEADS THE CONVERGED REPLY. Measured in
            # round 25: six proven couplings ran couple() once per level (~8
            # minutes a level) and the wall cut them at level 1 or 2; the
            # couple_levels recipe sat in a ladder step the parents never
            # reached (0 audit_results calls). The per-level dumps every served
            # participant keeps (field_level<k>.csv, interface_level<k>.csv,
            # participant_output_level<k>.log) make it safe to couple every
            # remaining level first and write the deliverables afterwards.
            # "SELF-CONSISTENT" ONLY WHERE THE FILE CHECKS RAN: inside a ladder they run in the
            # ladder's own verdict, after its last level (measured: a level reply called files
            # self-consistent beside a finding of ours about their field).
            _lead = _ra.what_to_fix_next(
                _compact, converged=bool(r.converged), clean_msg=(
                    (_CLEAN_HEAD if not _IN_LADDER.get() else
                     "this level's iteration CONVERGED; its files are judged by the ladder's verdict "
                     "after the last level. ")
                    + "Only files on disk count, so for EACH level: "
                    "(1) evaluate EACH side's converged field at "
                    "the probe points your task prescribes and write that "
                    "side's per-level field file; (2) write each side's "
                    "per-level interface file at the prescribed interface "
                    "points from its own converged trace and flux; (3) make "
                    "each side's per-level run log the captured participant "
                    "log whose path this reply returns (the solver's own "
                    "console output), never a summary you type. A summary "
                    "that names files not on disk is read as invented."))
        except Exception:                                    # advisory only
            _lead, _compact = None, []
        if r.converged:
            # the one-call mesh sequence leads EVERY converged reply, whatever the funnel found
            # (round 25: six proven couplings ran couple() per level and the wall cut them)
            # THIS LEVEL'S RUN LOGS FIRST, FROM THE NAMED FILES. Measured (round 29): four cells
            # coupled three refined levels and handed in run logs that were all copies of the
            # level-1 console (NDOF 54 on every level while the consoles read 54, 187, 693); the
            # an audit reads them as an unchanged mesh. The copy is named here with its full path,
            # per side, and it comes BEFORE the next level, not after all of them.
            _logs = ""
            if _lvl_for_log:
                _logs = " ".join(f"side {_p.name}: {Path(_p.work_dir) / f'participant_output_level{_lvl_for_log}.log'};"
                                 for _p in parts)
            # A FIXED POINT REACHED AT ONCE IS NOT A COUPLED RESULT. Measured (round 39, a 4C-Kratos
            # cell): both participants re-served a stale exports.json, the residual was 2e-13 at
            # iteration 2, the funnel said "unresponsive" -- and this lead said CONVERGED, write the
            # deliverables. The parent saw the contradiction and spent its calls on it; the level was
            # read as no coupling (a history under 3 rows). The lead now says what the run is.
            _unresp = sorted(n for n, st in (getattr(r, "responsiveness", None) or {}).items()
                             if "unresponsive" in str(st).lower())
            _probe_hits = [str(x) for x in val if "do not respond" in str(x).lower() or "byte-identical" in str(x).lower()]
            _n_hist = len(getattr(r, "history", None) or [])
            _trivial = bool(_unresp or _probe_hits) or _n_hist < 3
            _lvl_txt = f"LEVEL {_lvl_for_log}" if _lvl_for_log else "THIS LEVEL"
            # THE PARTICIPANTS LIST, VERBATIM. Measured (parent test 2026-09-12): with "<the same list you
            # passed>" as the placeholder, 2 of 4 parents handed their worker bare names; the worker sees
            # only the brief. This reply ran with the list -- it is echoed, so the next call is a paste.
            try:
                _pjson = json.dumps([{"name": _p.name, "command": list(_p.command), "work_dir": str(_p.work_dir),
                                      "imports_from": list(getattr(_p, "imports_from", []) or [])} for _p in parts])
            except Exception:                                # noqa: BLE001
                _pjson = ""
            _plist = f"participants='{_pjson}'" if _pjson else "participants=<the same list you passed here>"
            try:
                from core.quality_checks import unresponsive_clause as _uc
            except Exception:                                # noqa: BLE001
                _uc = lambda _n, _d, _s=None: ""                      # noqa: E731
            _rdet = getattr(r, "responsiveness_detail", None)
            _unresp_txt = "; ".join(_uc(_n, _rdet, getattr(r, "sensitivity", None)) or f"participant {_n} exported byte-identical data while its imports changed"
                                    for _n in _unresp)
            _not_yet = (f"{_lvl_txt} IS NOT A COUPLED RESULT YET: the iteration stopped after {_n_hist} step(s)"
                        + (f"; {_unresp_txt}" if _unresp else "")
                        # the probe's finding, unless it says what the clause above already said
                        + (f"; {_at_sentence(_probe_hits[0], 300)}" if _probe_hits and not (
                            _unresp and _probe_hits[0][:60] in _unresp_txt) and not (
                            _unresp and _unresp_txt[:60] in _probe_hits[0]) else "")
                        + ". A fixed point reached at once means the exchanged data never changed between iterations. "
                          "Two ways are on record: a participant that does not read ./imports.json on every run, and one "
                          "that reads it but never lets the values reach its solve -- written into a vector the solve then "
                          "replaces, or onto dofs the solve is free to move -- so it exports the same answer whatever it "
                          "is sent. (The driver deletes exports.json before every run, so no old file stands in for this "
                          "one.) Fix that side, then couple() again. Do not write this level's deliverables from this run: "
                          "a history under 3 rows shows no coupling to anyone who reads it.")
            # LEVELS FIRST, DELIVERABLES ONCE. Measured (round 40, C3 6041): after level 1 the
            # parent spent ~30 calls on that level's deliverables (a worker, eleven read_file
            # calls into the dumps, six write-and-run scripts), coupled level 2 with couple()
            # again, and the wall fell at 1 min left on the level-3 call -- two proven levels,
            # no result. Partial levels are worth nothing to the reader of the result; a
            # deliverable written from the wrong level's console is now caught at write time
            # (the write check) and here (the deliverable findings), so the order that lost
            # rounds 29-33 is safe again: every remaining level in ONE couple_levels call, then
            # one script that writes every level's files from the per-level dumps.
            # THE PER-LEVEL DUMPS MUST EXIST BEFORE ANY DELIVERABLE IS WRITTEN FROM THEM. Measured on a
            # recorded run: level 1 ran with a script version that wrote no interface_level1.csv; the parent
            # edited the scripts, coupled levels 2 and 3, and the deliverables script skipped level 1's
            # interface files with a warning -- three second-order levels graded as no result for one
            # missing pair. The dumps are checked here, the moment the level converged.
            _dump_gap = []
            _no_tag = []
            if _lvl_for_log and not _trivial:
                for _p in parts:
                    _wd = Path(_p.work_dir)
                    try:
                        _carries = any("field_level" in q.read_text(errors="ignore") for q in _wd.glob("*.py"))
                        _tagged = [q.name for q in _wd.glob(f"*level{_lvl_for_log}*") if not q.name.startswith("participant_output")]
                    except OSError:
                        _carries, _tagged = False, ["?"]
                    if _carries:
                        # the served dump block, or the agent's suffixed variant (a recorded run: field_level1_A.csv)
                        for _stem in (f"field_level{_lvl_for_log}", f"interface_level{_lvl_for_log}"):
                            if not any(_wd.glob(f"{_stem}*.csv")):
                                _dump_gap.append(f"side {_p.name}: {_stem}*.csv")
                    elif not _tagged and _outputs_at_risk(_wd, _lvl_for_log, _t_level_wall - 1.0):
                        # A HAND-WRITTEN SIDE THAT KEEPS NO LEVEL-TAGGED OUTPUT. Measured on a recorded run: side A's
                        # 4C runs reused one output prefix, level 2 overwrote level 1's VTUs, side A's three solution
                        # files came out identical and no interface file could be written for it -- on the first C1
                        # cell whose fields were right. Named here, while this level's outputs are still on disk --
                        # and only where this level wrote output files the next level would reuse: a side that
                        # writes nothing but exports.json has nothing to lose.
                        _no_tag.append(_p.name)
            _dump_txt = ("" if not _dump_gap else
                         f"LEVEL {_lvl_for_log}'S PER-LEVEL DUMPS ARE MISSING ({'; '.join(_dump_gap)}): the served contract writes "
                         f"field_level<k>.csv and interface_level<k>.csv next to exports.json on every run, and no file of either name (with "
                         f"or without a suffix) is there, so the script that ran this level did not carry that block (an older version, or a "
                         f"rewrite). Fix the participant, then couple() this "
                         f"level AGAIN before anything is written from its dumps -- a deliverables pass silently skips a level whose "
                         f"dump is missing, and a level set with one level missing is no result. ")
            _tag_txt = ""
            if _no_tag:
                result["untagged_sides"] = list(_no_tag)      # couple_levels stops on it before overwriting
                _tag_txt = (f" SIDE {' AND '.join(_no_tag)} KEEPS NO LEVEL-TAGGED OUTPUT (no file named with 'level{_lvl_for_log}' beside its "
                              f"exports.json other than the driver's console copy): the next level's run reuses the same names and OVERWRITES this "
                              f"level's field, VTUs and interface data -- measured, a side whose runs shared one output prefix handed in three "
                              f"identical solution files and no interface file. Before the next level, copy this level's field and interface "
                              f"output to level-tagged names (or add the served contract's dump block, which writes field_level<k>.csv and "
                              f"interface_level<k>.csv after exports.json). If your script already keeps them under another name, ignore this. ")
            # HOW MANY LEVELS HAVE BEEN COUPLED WITHOUT ONE BEING CHECKED.
            #
            # Asking politely has been measured and does not work. This lead
            # was moved to the FRONT of the converged reply precisely because
            # the recorded runs had produced zero calls to the check; over every
            # recorded run since, the lead was served 14 times across 10 runs and
            # the tool was called twice, in one run. So after the first
            # unchecked level the sentence stops being an invitation and starts
            # being a count, which is what the rest of this reply already does
            # for every other defect.
            #
            # It reads openPASO's own session journal -- which tool names this run
            # has used -- and nothing about the task.
            # RUN IT, DO NOT ASK FOR IT. Two wordings of the invitation were
            # measured and both failed: 2 calls against 14 asks, then 0 against
            # 7 after the ask was moved to the front and given the evidence. So
            # when the agent hands over its own four strings once, openPASO puts
            # each side's field back into that side's equation at every level
            # itself. Nothing is read from a task file and nothing is stored.
            if pde_check:
                try:
                    _spec = json.loads(pde_check)
                except Exception as _e:                  # noqa: BLE001
                    _spec = {}
                    _pde_notes.append(f"pde_check was not readable JSON ({_e}), so no level was checked")
                for _p in parts:
                    _one = _spec.get(_p.name) if isinstance(_spec, dict) else None
                    if _one is None and isinstance(_spec, dict) and "equation" in _spec:
                        _one = _spec                     # one operator for both sides
                    if not isinstance(_one, dict):
                        continue
                    try:
                        # THE AGENT NAMES ITS OWN FILES. openPASO does not guess a
                        # deliverable name and reads no task file; it checks what
                        # it is handed. They must be the PRESCRIBED PROBE GRID and
                        # not the participant's mesh nodes: the test is a midpoint
                        # quadrature, so on a scatter it reports nothing rather
                        # than a misleading number. Measured on a verified-correct
                        # cell -- its probe files give CONSISTENT with residuals
                        # 0.054, 0.010, 0.0005, while its own mesh dumps at the
                        # same three levels give NOT_APPLICABLE.
                        _names = [x.strip() for x in
                                  str(_one.get("solution_files", "")).split(",") if x.strip()]
                        if not _names:
                            _pde_notes.append(
                                f"{_p.name}: give solution_files -- a comma-separated list of THIS "
                                f"side's per-level field files, in level order, on the probe grid "
                                f"your task prescribes. openPASO will not guess their names.")
                            continue
                        _root_p = Path(_root) if _root else Path(_p.work_dir)
                        _fl = []
                        for _n in _names:
                            _q = Path(_n)
                            if not _q.is_absolute():
                                _q = (_root_p / _n) if (_root_p / _n).is_file() else (Path(_p.work_dir) / _n)
                            _fl.append((len(_fl), _q))
                        if len(_fl) < 2:
                            # the verdict IS the decay across levels
                            _pde_notes.append(
                                f"{_p.name}: only {len(_fl)} file named -- this check reads the DECAY "
                                f"across levels, so it starts answering at level 2")
                            continue
                        _out = _pde_consistency_body(
                            ",".join(str(_q) for _, _q in _fl),
                            str(_one.get("source", "")),
                            str(_one.get("coefficient", "1.0")),
                            str(_one.get("domain", "[[0,1],[0,1]]")),
                            str(_one.get("equation", "")))
                        try:
                            _pde_verdicts[_p.name] = json.loads(_out)
                        except Exception:                # noqa: BLE001
                            _pde_verdicts[_p.name] = {"note": str(_out)[:400]}
                    except Exception as _e:              # noqa: BLE001
                        _pde_notes.append(f"{_p.name}: {type(_e).__name__}: {_e}")
            # WHAT THE CHECK SAID, IN PLACE OF THE PARAGRAPH THAT USED TO ASK
            # FOR IT. The old wording told the agent to call the tool itself
            # and was measured twice: 2 calls against 14 asks, then 0 against 7.
            _pde_said = ""
            if _pde_verdicts:
                _bits = []
                for _n, _v in _pde_verdicts.items():
                    _vd = str((_v or {}).get("verdict", "?")) if isinstance(_v, dict) else "?"
                    _rt = (_v or {}).get("observed_rate") if isinstance(_v, dict) else None
                    _bits.append(f"{_n}: {_vd}" + (f" (residual falling {_rt:.2f}x per level)"
                                                   if isinstance(_rt, (int, float)) else ""))
                _pde_said = ("YOUR FIELD AGAINST YOUR OWN EQUATION, this level and every one before it -- "
                             + "; ".join(_bits)
                             + ". CONSISTENT means the weak residual falls as the mesh refines, which a "
                               "field solving the stated problem does and a field solving a different one "
                               "does not, and it now answers BOTH sides of a coupled pair rather than "
                               "only the one whose boundary carries no partner data. REFUSED means the "
                               "check did NOT look and you have learned nothing about that side: it models "
                               "second-order scalar diffusion only, so an elasticity or reaction-diffusion "
                               "operator is refused by name. A CONSISTENT side is weaker evidence than an "
                               "INCONSISTENT one -- measured, the INCONSISTENT verdict never lands on a "
                               "right side, while plenty of wrong ones still read CONSISTENT. "
                               "This is the one statement here that is about being RIGHT rather than "
                               "being tidy. ")
            elif _pde_notes and pde_check:
                _pde_said = ("YOUR EQUATION CHECK DID NOT RUN: " + "; ".join(_pde_notes[:3]) + ". ")
            _unchecked = ""
            if not pde_check and not _IN_LADDER.get():
                # NO LONGER "UNTIL YOU HAND OVER THE FOUR STRINGS": couple_levels and the
                # hand-in audit judge each side from its own config.json and dumps. The weak
                # identity needs three levels; one level's own discrete system decides only a
                # load or stiffness off by one factor (result_audit.level_equation_findings).
                _unchecked = (
                    "NO LEVEL HAS BEEN CHECKED AGAINST ITS OWN EQUATION IN FULL YET: that takes three levels "
                    "of the same side. At this level each side whose config.json states its equation was "
                    "checked only for a load or a stiffness off by one factor, in its own discrete system "
                    "rebuilt from this level's dump. couple_levels and the hand-in audit check each side from its own "
                    "config.json (k, reaction, source_expr and the box x0, x1, y0, y1) and its "
                    "field_level<k>.csv dumps; a side whose config.json lacks those keys is reported NOT "
                    "CHECKED. It is the check that tells a field solving the stated problem from one "
                    "that converged cleanly to the wrong function, and it needs no reference answer. It "
                    "judges the field inside its subdomain only; the values the field holds on its "
                    "interface and outer boundary are judged by the checks that read it against what "
                    "the side exports. ")
            # A LEVEL WHOSE OWN FIELD IS DEFECTIVE IS NOT TOLD TO SAVE ITS LOGS AND MOVE ON.
            # Measured: a located outer-boundary finding sat at character 3254 of a
            # 4000-character lead, behind this preamble, and the parent dismissed it.
            _field_items = [str(_f.get("finding", "")) for _f in list(_nf) + list(_us) + [
                _x for _x in _bf if not str(_x.get("finding", "")).startswith(
                    ("MESH UNCHANGED", "THIS LEVEL IS NOT A COUPLED RESULT"))]]
            _defect_txt = ("" if not _field_items else
                           f"{_lvl_txt} CONVERGED, BUT ITS FIELD IS NOT A RESULT: "
                           + " | ".join(_at_sentence(_x, 700) for _x in _field_items[:2])
                           + " Fix that side and couple this level again before any deliverable or further "
                             "level; everything built on this level inherits the defect. ")
            # INSIDE couple_levels THE LADDER RUNS THE NEXT LEVEL ITSELF. Measured: every level of a
            # finished ladder still told the run to write this level's logs before the next level
            # and to raise the level and call couple again (two reviews of one round).
            _in_ladder_txt = ("The ladder runs its next level itself. This level's console and dumps stay "
                              "beside each exports.json (participant_output_level<k>.log, field_level<k>.csv, "
                              "interface_level<k>.csv); the deliverables of every level are written from them "
                              "after the last level. ")
            _one_call = (_not_yet if _trivial else _defect_txt if _defect_txt else
                         (f"LEVEL {_lvl_for_log} CONVERGED. " if _lvl_for_log else "LEVEL CONVERGED. ")
                         + _unchecked
                         + _pde_said
                         + (_in_ladder_txt if _IN_LADDER.get() else
                         "FIRST, WRITE THIS LEVEL'S RUN LOGS -- BEFORE THE NEXT LEVEL, NOT AT THE END. "
                           "They are a copy, "
                           "one per side, of the console this level just produced"
                         + (f" (this level: {_logs})" if _logs else " (participant_output_level<k>.log)")
                         + " -- into the per-level run log name your task gives for that side. It costs one action "
                           "now and is the step most often lost when it is left to the end: the consoles stay on "
                           "disk, the logs the task asks for are never written, and a coupling that was proven is "
                           "read as not run. Then write the remaining "
                           "deliverables -- the field and interface files per side from each level's dumps -- in "
                           "ONE pass at the end. A run log copied from another level's console reads as "
                           "an unchanged mesh and sinks the whole sequence, and the write check names it the "
                           "moment it is written; a level set with one level missing is read as no result at "
                           "all, so the levels come before any file. "
                         + "THEN, if the task prescribes further mesh levels, RUN THE NEXT LEVEL: raise "
                           "the level and DOUBLE the cell counts in BOTH sides' config.json, then call couple again. "
                           "Every level keeps its own dumps and console beside each exports.json "
                           "(field_level<k>.csv, interface_level<k>.csv, participant_output_level<k>.log), so the "
                           "coarse levels survive the fine ones. Where a session exposes it, "
                           f"couple_levels({_plist}, levels=[<next level>, ...every further level], "
                           "history_pattern='<the task's per-level history file name with {k} in place of the level "
                           "number>') does the whole remaining sequence in one call, warm-starting each level from "
                           "the last. ")
                         + (f" MEASURED COST: this level's coupling took {_level_secs:.0f} s of solver wall time in "
                            f"all, over {r.iterations} iteration(s). A finer level's cost is measured when it runs; "
                            "none is projected here." if _level_secs is not None else ""))
            # WHAT THE FUNNEL FOUND LEADS WHEN IT IS A DEFECT; the converged recipe follows it.
            _top = min((int(_f.get("priority", 99)) for _f in (_compact or [])
                        if isinstance(_f, dict) and not _f.get("informational")), default=99)
            if _lead and _top <= 10:
                _lead = _lead + "\n" + _dump_txt + _one_call + (_tag_txt if not _trivial else "")
            else:
                _lead = _dump_txt + _one_call + (_tag_txt if not _trivial else "") + ("\n" + _lead if _lead else "")
        if _mesh_note:
            _lead = _mesh_note + ("\n" + _lead if _lead else "")
        # THE LADDER RIDES ON EVERY couple() REPLY: the next unmet step, from
        # the files, as a sub-agent brief -- so the agent that just coupled a
        # level is told the one thing to do next instead of judging the job.
        _next = None
        try:
            if _root:
                _next = (_ra.coupled_ladder(Path(_root), converged_now=bool(r.converged)) or {}).get("text")
                if _next and _pjson:
                    # the ladder reads files and cannot know the list; this reply does
                    _next = _next.replace("participants=<the same list passed to couple>", f"participants='{_pjson}'") \
                                 .replace("participants=<the same list you passed to couple>", f"participants='{_pjson}'")
        except Exception:                                # advisory only
            _next = None
        # THE DELIVERABLE DEFECTS THE AUDIT NAMES RIDE ON couple() TOO. Measured (rounds 29-31):
        # parents call audit_results 0-1 times a run and couple() 3-8 times; the run logs copied
        # from another level and the field files written at the nodes instead of the prescribed
        # probe points were named by the audit nobody called. Reads the agent's own files.
        _deliv = []
        _fq: list = []                     # of those, the findings about what the field holds
        try:
            if _root:
                for _fn in (_ra.wrong_level_run_log_findings, _ra.solution_rows_grow_findings,
                            _ra.identical_solution_levels_findings):
                    try:
                        _deliv += list(_fn(Path(_root)) or [])
                    except Exception:                    # noqa: BLE001
                        pass
                # AND THE PROOF THAT THIS LEVEL RAN, WHILE THERE IS STILL BUDGET
                # TO PRODUCE IT. The same checks run at hand-in, and by then the
                # median cell has two tool calls left and one in five has none --
                # they name work costing five to fifteen actions. Asked here they
                # cost nothing: the level just finished, its console is still on
                # disk, and a missing log is one redirect away.
                #
                # Scoped to the levels that were actually COUPLED, which is what
                # the residual histories on disk record. Nothing is asked about a
                # level the agent has not reached: measured against the graded
                # set, of the cells this stays silent on, every one was missing
                # logs only for levels it never coupled, and it speaks on 0 of
                # the 32 correct cells.
                try:
                    # A LEVEL THIS CALL FAILED, OR WHOSE HISTORY HOLDS NO ROW, WAS NOT COUPLED: its
                    # deliverables are not owed yet (measured: a level that failed at iteration 1
                    # left a header-only history and was asked for its interface files).
                    _done = sorted({_k for _q, _kind, _k, _sd
                                    in _ra._level_files(Path(_root), "csv")
                                    if str(_kind).startswith("residual") and _ra._line_count(_q) > 1
                                    and not (not r.converged and _k == _lvl_for_log)})
                    if _done and not _IN_LADDER.get():
                        # (inside a couple_levels call nothing can be written between its levels:
                        # the proof owed is named by the ladder's own reply, after the last level)
                        _deliv += list(_ra.deliverable_proof_due(Path(_root), _done) or [])
                        # AND WHAT THE FILES ALREADY SAY ABOUT THE ANSWER.
                        # The proof above asks whether the run can be shown to
                        # have happened; this asks whether what it produced is
                        # worth building on, and it is the family that decides
                        # a correct run against a completed but unphysical
                        # one. Measured on a live run that coupled three levels,
                        # wrote every deliverable and came out unphysical: NEAR-ZERO FIELD
                        # and FLOOR appear ZERO times in its trajectory because
                        # they live only in the hand-in audit, which ran with
                        # two tool calls left.
                        _fq_all = list(_ra.field_quality_due(Path(_root), _done, converged=bool(r.converged)) or [])
                        _fq = [f for f in _fq_all if not f.get("informational")]
                        _deliv += _fq_all
                except Exception:                        # noqa: BLE001
                    pass
        except Exception:                                # noqa: BLE001
            _deliv = []
        # A CHECK THAT DID NOT RUN IS COVERAGE, NOT A DEFECT ON DISK.
        _deliv_notes = [_f for _f in _deliv if _f.get("informational")]
        _deliv = [_f for _f in _deliv if not _f.get("informational")]
        if _deliv_notes and isinstance(result.get("checks_not_run"), list):
            result["checks_not_run"].extend(_at_sentence(str(_f.get("finding", "")), 500)
                                            for _f in _deliv_notes)
        # A FINDING OF OURS ABOUT THE FIELD SINKS A VERDICT THAT CALLED THE LEVEL GOOD, and the
        # clean words go. Measured: one reply said "CONVERGED -- THIS IS A RESULT, SAVE IT NOW",
        # "your files are self-consistent" and "... ITS FIELD DOES NOT HOLD IT" together.
        _v_now = str(result.get("verification", ""))
        if _fq and r.converged and (result.get("trustworthy_result") or "SAVE IT NOW" in _v_now
                                    or "automated checks passed" in _v_now or "CAVEAT" in _v_now):
            result["trustworthy_result"] = False
            result["verification"] = (
                "NOT VERIFIED -- the check of your files in this reply reports: "
                + _at_sentence(str(sorted(_fq, key=_ra._rank)[0].get("finding", "")), 400)
                + " A level that a check of its own files speaks against is not saved as a result "
                  "until that is settled.")
        if _deliv and _lead:
            _lead = _lead.replace(_CLEAN_HEAD, "this level's iteration CONVERGED. ")
        if _deliv:
            # THE WORST FIRST, AND EACH CUT AT A SENTENCE. Three findings are shown, in
            # the order they were found; a finding cut mid-sentence ran on into the next
            # line of the reply (measured: "Measured across recorded runs, LEVEL 1 ...").
            try:
                _deliv = sorted(_deliv, key=_ra._rank)
            except Exception:                            # noqa: BLE001
                pass
            _dtxt = ("YOUR DELIVERABLES ON DISK HAVE DEFECTS ANY READER OF THE RESULT WILL SEE -- fix them before the next level: "
                     + " | ".join(_at_sentence(str(_f.get("finding", "")), 700) for _f in _deliv[:3]))
            # A FAILED CALL LEADS WITH ITS FAILURE; the deliverables follow it.
            _lead = (_dtxt + ("\n" + _lead if _lead else "")) if r.converged else (
                (_lead + "\n" if _lead else "") + _dtxt)
        if _next:
            _lead = (_lead + "\n" + _next) if _lead else _next
            result = {"next_step": _next, **result}
        # AN INCONSISTENT FIELD LEADS EVERYTHING. It is the only verdict here
        # that says the answer is wrong rather than untidy, and it is invisible
        # to every other check in this reply.
        if _pde_verdicts or _pde_notes:
            result = {"pde_consistency": {**_pde_verdicts,
                                          **({"notes": _pde_notes} if _pde_notes else {})},
                      **result}
            _bad = [n for n, v in _pde_verdicts.items()
                    if isinstance(v, dict) and str(v.get("verdict", "")).upper() == "INCONSISTENT"]
            if _bad:
                # REPORTED WITH ITS ERROR RATE, NOT AS A VERDICT ON THE RUN.
                # Calibrated on the recorded runs this check accepts: most of
                # the wrong runs land here, and so does one of the correct ones.
                # That one cannot be tuned away --
                # its residual sequence (0.0132, 0.0058, 0.0106) has the same
                # shape as genuinely wrong cells, and a non-monotonicity guard
                # that silences it also silences 8 of the 15 true catches. So
                # the finding leads, because it is the only thing here about
                # being right, but it says what it is: strong evidence, not a
                # proof, on one side of a two-sided problem.
                _lead = ("YOUR FIELD DOES NOT SATISFY THE EQUATION YOU GAVE, on side(s) "
                         + ", ".join(_bad)
                         + ": the weak residual does not fall as the mesh refines, which is what a "
                         "field solving the stated problem does and what a field solving a different "
                         "one does not. Fix that side -- its source first, then its coefficient "
                         "(the identity cannot see boundary conditions) -- BEFORE spending budget on "
                         "further levels, because everything built on this one inherits it. "
                         "Measured on the recorded coupled runs: this verdict lands on 25 sides "
                         "of runs whose answer was wrong and on none of the 118 judged sides of "
                         "runs whose answer was right. The reverse is not true -- a CONSISTENT "
                         "side is not a clean bill of health: 113 sides of wrong runs also read "
                         "CONSISTENT, because this identity cannot see a wrong condition on a "
                         "face and the wrong side of a run can be its partner.\n"
                         + (_lead or ""))
        # BETWEEN LADDER LEVELS THE DELIVERABLES ARE NOT DUE. Measured: levels 2 and 3 of a
        # running ladder led with "LEVEL k'S DELIVERABLES ARE NOT WRITTEN" before the ladder had
        # run its last level.
        if _lead and _IN_LADDER.get() and _LADDER_PENDING.get():
            _lead = "\n".join(_ln for _ln in str(_lead).split("\n")
                               if "DELIVERABLES ARE NOT WRITTEN" not in _ln[:120]) or None
        if _lead:
            result = {"what_to_fix_next": _lead,
                      "presubmission_findings": _compact[:12], **result}
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def couple_levels(participants: str | list, levels: str | list, critic_approved: bool = False,
                            max_iter: int = 150, tol: float = 1e-6, accelerator: str = "auto",
                            theta: float = 0.5, probe: bool = True, history_dir: str = "",
                            history_pattern: str = "coupling_history_level{k}.csv") -> str:
        """EVERY PRESCRIBED MESH LEVEL IN ONE CALL -- the same partitioned coupling as
        `couple`, run once per level of a task's mesh sequence.

        Measured over three rounds of recorded runs: six couplings that were proven at
        level 1 (both codes ran, the iteration converged) never reached level 3,
        because every level cost the agent ten more tool calls -- edit both
        config.json files, call couple, save the history, write the deliverables --
        and the wall clock ran out. This call does the per-level plumbing itself:

          * `levels` is a JSON list, one entry per level, with each side's mesh
            keys under that side's NAME, e.g. (your task's own numbers go here)
                [{"level": 1, "A": {"nx": <nA>, "ny": <mA>},  "B": {"nx": <nB>, "ny": <mB>}},
                 {"level": 2, "A": {"nx": <2*nA>, "ny": <2*mA>}, "B": {"nx": <2*nB>, "ny": <2*mB>}},
                 {"level": 3, "A": {"nx": <4*nA>, "ny": <4*mA>}, "B": {"nx": <4*nB>, "ny": <4*mB>}}]
            where the keys under each participant's NAME, plus "level", are
            handed to that participant's PROCESS in the environment variable
            OPENPASO_CONFIG_JSON (a JSON object; OPENPASO_LEVEL carries the level
            alone). openPASO writes NO file into your directories: most served
            contracts merge OPENPASO_CONFIG_JSON over their own ./config.json, and
            any script must read it the same way
            (json.loads(os.environ.get("OPENPASO_CONFIG_JSON", "{}")) merged over
            its config) or be run with one couple() call per level instead; a
            level that hands keys to a script that never reads them is refused
            before anything runs. Halve h per
            level as the task prescribes, i.e. double every cell count; a
            transient side's step count travels the same way ("n_steps", which
            the served transient contracts read), so both sides change it
            together;
          * each level starts from the previous level's converged interface
            state (the driver's warm start), which is why the levels must run in
            the same work directories;
          * each level writes its measured iteration history to
            <history_dir>/<history_pattern with {k} = the level> -- pass
            `history_pattern` as the per-level history file name YOUR TASK
            prescribes (it must contain "{k}"); the default is a neutral name
            you would have to rename -- and keeps each side's solver console
            as participant_output_level<k>.log next to its exports.json;
          * the reply carries, per level, the verdict, the iteration count, the
            history path and the interface tables ready to save. It stops at the
            first level that does not converge; fix that level and call again
            from it (the earlier levels' files stay).

        Everything `couple` checks is checked here too (it IS couple, per level).
        `history_dir` defaults to the task's working directory. THE CRITIC REVIEW
        IS THE ONE `couple` TAKES: submit_critic_review(solver='couple',
        coupling_args=<{"participants": ..., "max_iter": ..., "tol": ...,
        "accelerator": ..., "theta": ..., "probe": ...} exactly as passed here>),
        then critic_approved=True; openPASO looks that review up for every level.
        """
        import time as _t_ladder
        _ladder_t0 = _t_ladder.time()   # the deliverables judged below must be newer than this
        # A LIST IS TAKEN AS IT IS, not refused. Measured: one run sent `levels` as a list 16
        # times and lost 13.5 minutes to schema refusals, while this docstring calls it a list.
        participants, levels = _as_json_text(participants), _as_json_text(levels)
        try:
            lv = json.loads(levels)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"invalid levels JSON: {e}"})
        if not isinstance(lv, list) or not lv or not all(isinstance(x, dict) for x in lv):
            return json.dumps({"error": "levels must be a non-empty JSON list of objects "
                                        "{\"level\": k, \"<participant name>\": {<config keys>}, ...}"})
        try:
            specs = json.loads(participants)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"invalid participants JSON: {e}"})
        if not isinstance(specs, list) or len(specs) < 2:
            return json.dumps({"error": "need a JSON list of >=2 participants"})
        for _s in specs:
            _bin = _solver_binary_as_participant(_s) if isinstance(_s, dict) else ""
            if _bin:
                return json.dumps({"error": _bin})
        names = {s.get("name"): s for s in specs if isinstance(s, dict) and s.get("name")}
        cell_work = os.environ.get("OPENPASO_CELL_WORKDIR")
        _history_base = cell_work or str(Path(specs[0].get("work_dir", ".")).resolve().parent)
        if not history_dir:
            history_dir = _history_base
        elif not Path(history_dir).is_absolute():
            # history_dir='.' is what an agent naturally writes for "here". Joined
            # as-is it produced a relative history_path that couple() used to
            # refuse, so no level's history was written. Anchor it to the task.
            history_dir = str(Path(_history_base) / history_dir)
        if "{k}" not in (history_pattern or ""):
            return json.dumps({"error": "history_pattern must contain '{k}' (the level number), e.g. the "
                                        "per-level history file name your task prescribes"})
        # THE REVIEW IS RESOLVED HERE, from the server's own record, against the
        # same canonical text `couple` binds to (the per-level calls below bind
        # to it again): a self-reported flag decides nothing.
        _reviewed, _review_note = _critic_state(
            "couple", _coupling_setup_text(participants=participants, max_iter=max_iter, tol=tol,
                                           accelerator=accelerator, theta=theta, monolithic="",
                                           probe=probe))
        for nm, spec in names.items():
            wd = Path(str(spec.get("work_dir", "")))
            # THE SAME RULE AS `couple`, WHICH THIS TOOL CALLS. It refused a
            # relative work_dir outright while `couple` had just learned to
            # resolve one against the declared task directory -- so the form
            # the served example shows would have been accepted by one door
            # and refused by the other. A rule that differs between two doors
            # of the same tool is worse than either rule.
            if not wd.is_absolute():
                if not cell_work:
                    return json.dumps({"error":
                        f"participant {nm}: work_dir must be an ABSOLUTE path "
                        f"— nothing has told this server which directory a "
                        f"relative path would be relative TO."})
                wd = Path(cell_work) / wd
                spec["work_dir"] = str(wd)
            if cell_work and not wd.resolve().is_relative_to(Path(cell_work).resolve()):
                return json.dumps({"error": f"participant {nm}: work_dir {wd} is outside this task's "
                                            f"working directory {cell_work}"})
        # A SCRIPT THAT NEVER READS OPENPASO_CONFIG_JSON RUNS EVERY LEVEL ON ONE MESH.
        # Measured: a three-level ladder ran one side on a single mesh for 13 minutes,
        # because the file on disk read only its own ./config.json and the keys this
        # call hands each process never reached it. Refused here, before anything
        # runs, when a level hands keys to a participant whose Python script -- and
        # every module of its own directory that script imports -- never names
        # OPENPASO_CONFIG_JSON. A command that names no Python script is not judged.
        _unread = []
        for nm, spec in names.items():
            if not any(isinstance(e.get(nm), dict) and any(kk != "level" for kk in e.get(nm)) for e in lv):
                continue
            _cmd = spec.get("command")
            try:
                _args = (__import__("shlex").split(_cmd) if isinstance(_cmd, str)
                         else [str(a) for a in (_cmd or [])])
            except ValueError:
                continue
            _wd = Path(str(spec.get("work_dir", "")))
            _pys = []
            for _a in _args[1:]:
                if _a.endswith(".py"):
                    _c = Path(_a) if Path(_a).is_absolute() else _wd / _a
                    if _c.is_file():
                        _pys.append(_c)
            if not _pys:
                continue
            _texts, _seen = [], set()
            _todo = list(_pys)
            while _todo and len(_seen) < 20:
                _q = _todo.pop()
                if _q in _seen:
                    continue
                _seen.add(_q)
                try:
                    _src = _q.read_text(errors="replace")
                except OSError:
                    continue
                _texts.append(_src)
                for _mod in re.findall(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", _src, re.M):
                    _m = _q.parent / f"{_mod}.py"
                    if _m.is_file():
                        _todo.append(_m)
            if _texts and not any("OPENPASO_CONFIG_JSON" in _x for _x in _texts):
                _keys = sorted({kk for e in lv if isinstance(e.get(nm), dict) for kk in e.get(nm) if kk != "level"})
                _unread.append(f"{nm} ({_pys[0]}; keys {', '.join(_keys[:6])})")
        if _unread:
            return json.dumps({"error": (
                "NOTHING RAN: the script of " + "; ".join(_unread) + " never reads OPENPASO_CONFIG_JSON, so the keys "
                "this call hands it at each level would never reach it and every level would run on the mesh its "
                "own ./config.json states. Add, right after the line that reads ./config.json, "
                "cfg.update(json.loads(os.environ.get(\"OPENPASO_CONFIG_JSON\", \"{}\"))) (with the name your script "
                "gives that dict), and use the merged values for the mesh; or run one couple() call per level and "
                "edit that side's ./config.json between the calls.")})
        out_levels = []
        _blocks_by_level: dict = {}          # each level's per-block residuals, for the caveat's trend
        for entry in lv:
            try:
                k = int(entry.get("level", len(out_levels) + 1))
            except (TypeError, ValueError):
                return json.dumps({"error": f"bad level entry {entry!r}"})
            # THE LEVEL'S KEYS TRAVEL IN EACH PARTICIPANT'S ENVIRONMENT. openPASO
            # writes NO file of the agent's -- the served contracts read
            # OPENPASO_CONFIG_JSON and merge it over their own ./config.json.
            level_specs = []
            for spec in specs:
                sp = dict(spec)
                nm = sp.get("name")
                upd = entry.get(nm) if nm else None
                cfg = dict(upd) if isinstance(upd, dict) else {}
                cfg["level"] = k
                env = dict(sp.get("env") or {})
                env["OPENPASO_LEVEL"] = str(k)
                env["OPENPASO_CONFIG_JSON"] = json.dumps(cfg)
                sp["env"] = env
                level_specs.append(sp)
            _later = []
            for _e in lv[len(out_levels) + 1:]:
                try:
                    _later.append(int(_e.get("level")))
                except (TypeError, ValueError):
                    pass
            _pend_tok = _LADDER_PENDING.set(tuple(_later))
            _in_tok = _IN_LADDER.set(True)
            try:
                reply = await couple(json.dumps(level_specs), max_iter=max_iter, tol=tol, accelerator=accelerator,
                                     theta=theta, probe=probe, critic_approved=critic_approved,
                                     history_path=str(Path(history_dir) / history_pattern.replace("{k}", str(k))),
                                     iface_level=k)
            finally:
                _LADDER_PENDING.reset(_pend_tok)
                _IN_LADDER.reset(_in_tok)
            try:
                rep = json.loads(reply)
            except Exception:                                   # noqa: BLE001
                rep = {"error": "unparseable couple reply", "raw": str(reply)[:3000]}
            compact = {"level": k}
            for key in ("converged", "iterations", "residual", "error", "history_file",
                        "interface_csv", "captured_solver_logs", "validation", "relaxation",
                        "responsiveness", "interface_balance", "interface_profile"):
                if key in rep and rep[key] is not None:
                    compact[key] = rep[key]
            _blocks_by_level[k] = dict(rep.get("block_residuals") or {})
            # A LEVEL THAT STARTED COLD BECAUSE THE LAST LEVEL'S EXPORT DID NOT FIT IT, in one line.
            _cold = [str(_n) for _n in (rep.get("noise_notes") or [])
                     if "does not fit this level's window" in str(_n)]
            if _cold:
                compact["warm_start"] = _cold[0]
            # THE VERDICT TRAVELLED NOWHERE. couple() stamps the gate's verdict on
            # its own reply; this copy list did not carry it, and the aggregate had
            # none of its own, so a ladder whose every level was verified reached
            # every consumer with nothing saying so -- and the browser interface
            # honestly reported "a solver ran, but openPASO did not verify its
            # result" for work that had been verified level by level.
            #
            # The BOOLEAN on every level; the PROSE only where it does work. The
            # verification text is 700-1500 characters, and three copies of it
            # would add ~4.5k to a reply the parent reads in full -- the same
            # budget that has twice cut material the runs needed.
            #
            # THE LEVEL ANSWERS FOR ITS NUMBERS; THE REVIEW IS THE LADDER'S. A
            # level's own couple() reply can never say `trustworthy_result: true`
            # inside a ladder, and not because anything is wrong: couple_levels
            # injects that level's mesh keys into each participant's environment,
            # so the level's setup digest is no longer the digest the critic
            # reviewed, and the lookup misses every time. Reading couple()'s
            # verdict here would therefore publish "not verified" for every level
            # of every ladder forever, which is worse than publishing nothing.
            # The review is of the SETUP, and the levels are that same setup at
            # different meshes, so it is resolved ONCE below for the whole call.
            # What belongs to the level is its own numerical evidence, read from
            # its own reply: it converged, it exchanged something, and no
            # downstream check spoke against it.
            # (the level's own not-a-coupling evidence is read FIRST: it used to be
            # set below this line, so a level that exchanged nothing read trustworthy)
            _not_coupled = _level_not_a_coupled_result(rep) if rep.get("converged") else ""
            if _not_coupled:
                compact["coupled_evidence"] = (
                    f"LEVEL {k} IS NOT A COUPLED RESULT YET: {_not_coupled}. Two ways "
                    f"are on record: a participant that does not read ./imports.json on "
                    f"every run, and one that reads it but never lets the values reach "
                    f"its solve (written into a vector the solve replaces, or onto dofs "
                    f"it is free to move). The driver deletes exports.json before each "
                    f"run, so what it reads is that run's own. Do not write this level's "
                    f"deliverables from this run: a "
                    f"history under 3 rows shows no coupling to anyone who reads it.")
            _numbers_ok = (bool(rep.get("converged"))
                           and not compact.get("coupled_evidence")
                           and not (rep.get("validation") or []))
            compact["trustworthy_result"] = bool(_numbers_ok and _reviewed)
            if not compact["trustworthy_result"]:
                if not _numbers_ok and rep.get("verification"):
                    compact["verification"] = rep["verification"]
                elif not _reviewed:
                    compact["verification"] = (
                        "NOT VERIFIED — this level's numbers passed, but no critic "
                        "review of this coupling setup is on record (" + _review_note
                        + "). The review covers the whole ladder, so one "
                        "submit_critic_review for this setup verifies every level.")
            if rep.get("what_to_fix_next"):
                # 4000, not 2000: a failed 4C side's lead now carries the deck's defects and
                # 4C's own stop line, which the old cap cut off mid-list
                compact["what_to_fix_next"] = _cut_outside_commands(str(rep["what_to_fix_next"]), 4000)
            # THE SUMMARY MAY NOT SAY MORE THAN THE LEVEL'S OWN EVIDENCE (coupled_evidence, above).
            # only a level that is a coupled result gets its interface file written for it
            _healed = (_heal_missing_interface_dump(level_specs, k)
                       if rep.get("converged") and not compact.get("coupled_evidence") else {})
            if _healed:
                compact["interface_dump_written_for_you"] = _healed
                compact["interface_dump_note"] = (
                    "this level's interface file was MISSING and openPASO wrote "
                    "it for you from that side's own exports.json, which it was "
                    "holding at the only moment those numbers are correct -- "
                    "exports.json is overwritten by the next level. Nothing was "
                    "invented: every number came out of your solver. The VOLUME "
                    "field (field_level%d.csv) is still yours to write: add the "
                    "dump block to the participant and couple this level again "
                    "-- a standalone run of the participant reads an empty "
                    "imports.json, so its field is not the coupled one." % k)
            peaks = _level_field_peak(level_specs, k)
            if peaks:
                compact["field_peak"] = peaks
                flat = [nm for nm, v in peaks.items()
                        if v["peak_abs_value"] < 1e-8]
                if flat:
                    compact["field_finding"] = (
                        "NEAR-ZERO FIELD on side(s) " + ", ".join(sorted(flat))
                        + f": the level-{k} field this side just exported peaks "
                        "below 1e-8. This measures the field, not why: check "
                        "that the source reaches that side's solve (its form, "
                        "load vector or deck) and that its boundary values are "
                        "not all zero, before reading it as a small answer. A coupling whose "
                        "fields are zero converges immediately and tells you "
                        "nothing: two sides exchanging nothing cannot disagree.")
            # A LEVEL RUN ON THE PREVIOUS LEVEL'S MESH STOPS THE LADDER. Measured: a
            # ladder ran its third level for 4.1 minutes after couple() had said its
            # second "counts as not run".
            if any(str(_v).startswith("MESH UNCHANGED FROM LEVEL") for _v in (rep.get("validation") or [])):
                compact["mesh_unchanged"] = True
            # A SIDE THAT KEEPS NO LEVEL-TAGGED OUTPUT LOSES THIS LEVEL TO THE NEXT ONE.
            # Measured: inside one couple_levels call the warning that a side "keeps no
            # level-tagged output" arrived after every level had overwritten the one
            # before, and two right couplings were handed in with fields re-run without
            # the partner. The ladder stops before the next level runs over it.
            if rep.get("untagged_sides") and rep.get("converged") and len(out_levels) + 1 < len(lv):
                compact["untagged_sides"] = list(rep["untagged_sides"])
            out_levels.append(compact)
            if not rep.get("converged") or compact.get("mesh_unchanged") or compact.get("untagged_sides"):
                break
        _left = _LADDER_IFACE_CSV_INLINE_MAX
        for _x in reversed(out_levels):          # the finest level keeps its blocks first
            _blocks = _x.get("interface_csv")
            if not isinstance(_blocks, dict):
                continue
            for _pn, _b in list(_blocks.items()):
                if not isinstance(_b, dict) or not isinstance(_b.get("csv"), str):
                    continue
                if len(_b["csv"]) <= _left:
                    _left -= len(_b["csv"])
                else:
                    _blocks[_pn] = _iface_block_not_inlined(
                        _pn, _b, len(_b["csv"]), level=None if _x is out_levels[-1] else _x.get("level"))
        all_ok = (bool(out_levels) and len(out_levels) == len(lv)
                  and all(x.get("converged") and not x.get("coupled_evidence")
                          and not x.get("mesh_unchanged") for x in out_levels))
        # BOTH FACTS, NOT THE LOUDER ONE. A sequence can carry a level that is not a
        # coupling AND stop later on a level that did not converge, and the two need
        # different repairs. Naming only the first buries the other: the test fixture
        # that caught this has unresponsive participants at level 1 and a failure at
        # level 2, and an agent told only about level 1 would re-run the whole ladder
        # without knowing what it ran into.
        _null = [x for x in out_levels if x.get("coupled_evidence")]
        _stopped = (out_levels and not out_levels[-1].get("converged"))
        _same = [x for x in out_levels if x.get("mesh_unchanged")]
        _kept = [x for x in out_levels if x.get("untagged_sides")]
        _parts = []
        if _kept:
            _parts.append(
                f"THE LADDER STOPPED AFTER LEVEL {_kept[0]['level']} BEFORE THE NEXT LEVEL COULD OVERWRITE IT: side "
                f"{' and '.join(_kept[0]['untagged_sides'])} keeps no file named with 'level{_kept[0]['level']}' "
                f"beside its exports.json, so the next level's run would reuse the same names. Copy this level's "
                f"field and interface output to level-tagged names now (or add the served contract's dump block, "
                f"which writes field_level<k>.csv and interface_level<k>.csv after exports.json), then call "
                f"couple_levels again from level {_kept[0]['level'] + 1} on.")
        if _same:
            _parts.append(
                f"LEVEL {_same[0]['level']} RAN ON THE MESH OF THE LEVEL BEFORE IT (its sides printed the "
                f"same NDOF), so the ladder stopped there. The keys this call hands each side arrive in the "
                f"environment variable OPENPASO_CONFIG_JSON; that side's script must merge them over its "
                f"./config.json and build its mesh from the merged values. Fix it, then call couple_levels "
                f"again from level {_same[0]['level']} on.")
        if _null:
            first = _null[0]
            _parts.append(
                str(first["coupled_evidence"]) + " Fix the participant that is not "
                "reading its imports, then call couple_levels again from level "
                f"{first['level']} on. The levels before it keep their files and "
                "histories.")
        if _stopped:
            bad = out_levels[-1]["level"]
            _parts.append(
                f"LEVEL {bad} DID NOT CONVERGE (or errored): read its what_to_fix_next "
                f"and the participant logs, fix the cause, then call couple_levels again "
                f"with the levels from {bad} on -- the earlier levels' files and "
                f"histories stay.")
        if _parts:
            nxt = " ".join(_parts)
        elif all_ok:
            nxt = ("EVERY REQUESTED LEVEL CONVERGED. For EACH level k and EACH side: write the task's "
                   "per-level field file from that side's field_level<k>.csv (interpolated at the task's "
                   "probe points, one griddata call per value column), its per-level interface file from "
                   "interface_level<k>.csv (trace and flux at its interface nodes, interpolated along the "
                   "interface to the task's interface points with np.interp on the coordinate the interface "
                   "runs along -- griddata refuses points that lie on one line), and its per-level run log from "
                   "participant_output_level<k>.log plus the NDOF line; the coupling histories are already "
                   "at the history paths above. Then audit_results(work_dir) and the summary.")
        else:
            bad = out_levels[-1]["level"] if out_levels else "?"
            nxt = (f"LEVEL {bad} DID NOT CONVERGE (or errored): read its what_to_fix_next and the "
                   f"participant logs, fix the cause, then call couple_levels again with the levels "
                   f"from {bad} on -- the earlier levels' files and histories stay.")
        # THE EQUATION VERDICT, FROM THE FILES THIS LADDER WROTE. Each side's own
        # mesh dumps from this ladder (field_level<k>.csv, interpolated to cell
        # midpoints -- see result_audit._dump_level_sets) decide what the field
        # is; delivered per-level files, where they exist, are judged too, and
        # only for whether they carry that field. The audit on delivery computes
        # this again; here it reaches the agent while it can still act. It reads
        # each side's own config.json for the operator, never a task file or a key.
        _eq = []
        _cut = {}
        try:
            from .result_audit import equation_findings as _eqf
            # A DUMP IS CURRENT WHEN IT IS NEWER THAN WHAT WROTE IT, not only newer
            # than this call: a run that coupled level 1 with couple() and levels 2-3
            # with couple_levels had its level-1 dump set aside, and a three-level
            # result that satisfied its equation read "only two levels could be
            # checked" (measured). Each side's cutoff is the newest of its
            # participant script and its config.json.
            for _sp in specs:
                if not isinstance(_sp, dict) or not _sp.get("work_dir"):
                    continue
                _wd = Path(str(_sp["work_dir"]))
                _wd = _wd if _wd.is_absolute() else Path(history_dir) / _wd
                _stamps = [(_wd / "config.json")] + [
                    (_wd / _c if not Path(str(_c)).is_absolute() else Path(str(_c)))
                    for _c in (_sp.get("command") or []) if str(_c).endswith(".py")]
                _ts = [q.stat().st_mtime for q in _stamps if q.is_file()]
                try:
                    _cut[str(_wd.resolve())] = max(_ts) if _ts else _ladder_t0
                except OSError:
                    pass
            _all_eq = list(_eqf(Path(history_dir), since=_cut or _ladder_t0))
            _eq = [f["finding"] for f in _all_eq
                   if "NOT CHECKED" not in f.get("finding", "")]
            # THE GAP IS NAMED, IN ONE LINE. This filtered every NOT CHECKED
            # note out of the ladder reply, so a whole round of couplings whose
            # config.json named its source under a key the check did not read
            # heard nothing -- the one check that separates a field converging
            # to the right function from one converging to a wrong one ran on
            # no side of no cell, silently. The note that would have said so
            # is 300 characters.
            _eq_gaps = [_at_sentence(str(f["finding"]), 400) for f in _all_eq
                        if "NOT CHECKED" in f.get("finding", "")]
            # FILES OLDER THAN THIS LADDER ARE THE LAST RUN'S. After a re-run the
            # per-level deliverables on disk are still the previous ladder's until
            # they are rewritten, and this verdict was computed from them without
            # a word (measured on a second ladder whose files were the first's).
            _stale = sorted(q.name for q in Path(history_dir).glob("solution_level*.csv")
                            if q.is_file() and q.stat().st_mtime < _ladder_t0)
            if (_eq or _eq_gaps) and _stale:
                _eq.insert(0, f"THE EQUATION VERDICT BELOW WAS TAKEN FROM FILES WRITTEN BEFORE THIS "
                              f"LADDER STARTED ({', '.join(_stale[:4])}{' ...' if len(_stale) > 4 else ''}): "
                              f"it describes the previous run until you rewrite them from this "
                              f"ladder's dumps.")
        except Exception:                                    # noqa: BLE001
            _all_eq = []
            _eq = []
            _eq_gaps = []
        # THE LADDER'S OWN VERDICT, under the key names couple() uses so a reader
        # needs no mapping. It is STRICTLY STRONGER than the conjunction of the
        # levels: a level answers for one mesh, the ladder answers for the mesh
        # SEQUENCE, and two of its properties belong to no single level --
        # whether the mesh actually refined between them (three levels each
        # verified on the SAME mesh is not a verified ladder, and six three-level
        # couplings were lost to exactly that shape), and whether the coupling
        # transmitted anything at all. So the top level decides, and a consumer
        # must not reconstruct it from the levels.
        _ladder_faults = []
        if not _reviewed:
            _ladder_faults.append(
                "no critic review of this coupling setup is on record ("
                + _review_note + "); one submit_critic_review for this setup "
                "covers every level of the ladder")
        # THE TREND DECIDES WHAT A LEVEL-1 CAVEAT WAS. A balance finding at the
        # coarsest level is discretisation error when it shrinks under
        # refinement and a wrong transmission condition when it does not; the
        # level cannot know which, the ladder can. A level whose only finding
        # is a balance that the later levels show shrinking is not held against
        # the ladder; a balance that does not shrink is a ladder fault, named
        # with its numbers.
        _bal_faults, _bal_notes, _bal_excused = _interface_balance_trend(out_levels)
        _ladder_faults.extend(_bal_faults)
        _prof_faults, _prof_notes, _prof_excused = _interface_profile_trend(out_levels)
        _ladder_faults.extend(_prof_faults)
        # THE PER-BLOCK ITERATION CAVEAT BY ITS OWN PER-LEVEL TREND, stated with its numbers;
        # a level whose only findings are such caveats answers through it, not through its
        # level text (which carried the caveat's generic reason).
        _blk_faults, _blk_notes, _blk_excused = _block_caveat_trend(out_levels, _blocks_by_level)
        _ladder_faults.extend(_blk_faults)
        _blk_stated = {x["level"] for x in out_levels
                       if (x.get("validation") or []) and x.get("converged") and not x.get("coupled_evidence")
                       and all(_BLOCK_CAVEAT in str(v) for v in x["validation"])}
        _bal_notes = _bal_notes + _prof_notes + _blk_notes

        def _excused_finding(level: int, v) -> bool:
            v = str(v)
            return ((v.startswith("Interface flux NOT balanced") and level in _bal_excused)
                    or ("does NOT match POINT BY POINT" in v and level in _prof_excused)
                    or (_BLOCK_CAVEAT in v and level in _blk_excused))
        _tautology = any(isinstance(x.get("interface_balance"), dict)
                         and x["interface_balance"].get("tautology")
                         for x in out_levels)
        _unverified = [x["level"] for x in out_levels
                       if not x.get("trustworthy_result")
                       and not ((x["level"] in _bal_excused or x["level"] in _prof_excused
                                 or x["level"] in _blk_excused)
                                and all(_excused_finding(x["level"], v)
                                        for v in (x.get("validation") or []))
                                and x.get("converged") and not x.get("coupled_evidence")
                                and _reviewed)
                       and not (x["level"] in _blk_stated and _reviewed)]
        if _unverified:
            # A DEFECTIVE FIELD AND A CAVEAT OR REVIEW GAP ARE TWO FAULTS, named apart:
            # the first reason of several levels used to speak for all of them.
            def _why_of(_x):
                return (str(_x.get("coupled_evidence") or _x.get("verification")
                            or "it did not pass the verification gate")).strip()
            _bad = [x for x in out_levels if x["level"] in _unverified]
            _field_bad = [x for x in _bad if x.get("coupled_evidence") or _level_fault_is_field(_why_of(x))]
            for _grp in (_field_bad, [x for x in _bad if x not in _field_bad]):
                if not _grp:
                    continue
                _lv = ", ".join(str(x["level"]) for x in _grp)
                _ladder_faults.append(
                    (f"level {_lv} is not verified: " if len(_grp) == 1
                     else f"levels {_lv} are not verified; level {_grp[0]['level']}: ") + _why_of(_grp[0]))
        if len(out_levels) > 1:
            try:
                from .result_audit import ndof_ladder_findings as _ndof
                for _f in _ndof(Path(history_dir)):
                    _ladder_faults.append(_at_sentence(str(_f.get("finding", "")), 600))
            except Exception:                                # noqa: BLE001
                pass
        # TWO FAULTS NO EXCHANGE CHECK CAN SEE. Measured on a coupled run: a
        # Dirichlet side solved none of its interior (its mask was built by looping
        # over a BitArray and held one bit), exported the imported trace over a zero interior, converged
        # in ten iterations with every exchange check passing -- and this verdict
        # read VERIFIED LADDER, trustworthy_result: true, because nothing here
        # looked at either side's own field. A side that never solved its interior,
        # and a side whose field does not satisfy its own equation, now sink the
        # ladder by name.
        _part_dirs = []
        for _sp in specs:
            _wd = _sp.get("work_dir") if isinstance(_sp, dict) else None
            if _wd and _sp.get("name") != "__monolithic__":
                _pd = Path(str(_wd))
                _part_dirs.append(_pd if _pd.is_absolute() else Path(history_dir) / _pd)
        try:
            from .result_audit import unsolved_field_findings as _unsolved
            # THE LADDER ANSWERS FOR ITS PARTICIPANTS: a probe folder beside them holds
            # no dump of this sequence and must not sink it (scan=False).
            for _f in _unsolved(Path(history_dir), dirs=_part_dirs, since=_cut or _ladder_t0, scan=False):
                _ladder_faults.append(_at_sentence(str(_f.get("finding", "")), 500))
        except Exception:                                    # noqa: BLE001
            pass
        try:
            from .result_audit import own_field_flux_findings as _own_flux
            for _f in _own_flux(Path(history_dir), dirs=_part_dirs, since=_cut or _ladder_t0, scan=False):
                if not _f.get("informational"):          # an abstention note is no fault
                    _ladder_faults.append(_at_sentence(str(_f.get("finding", "")), 700))
        except Exception:                                    # noqa: BLE001
            pass
        try:
            from .result_audit import nonfinite_field_findings as _nonfinite
            for _f in _nonfinite(Path(history_dir), dirs=_part_dirs, since=_cut or _ladder_t0, scan=False):
                _ladder_faults.append(_at_sentence(str(_f.get("finding", "")), 500))
        except Exception:                                    # noqa: BLE001
            pass
        try:
            from .result_audit import exports_not_the_field_findings as _not_field
            for _f in _not_field(Path(history_dir), dirs=_part_dirs, since=_cut or _ladder_t0, scan=False):
                _ladder_faults.append(_at_sentence(str(_f.get("finding", "")), 700))
        except Exception:                                    # noqa: BLE001
            pass
        for _t in _eq:
            if "DOES NOT SATISFY" in str(_t):
                _ladder_faults.append(_at_sentence(str(_t), 600))
        # THE FIELD'S OWN BOUNDARY, TOO. Measured: a ladder whose side B left both
        # interface corners free read VERIFIED LADDER while every level led with the
        # free-corner finding, and another buried a located outer-boundary finding
        # under "this level's numbers passed".
        try:
            from .result_audit import free_interface_end_findings as _free_ends
            for _f in _free_ends(Path(history_dir), dirs=_part_dirs):
                if not _f.get("informational") and int(_f.get("priority", 99)) <= 10:
                    _ladder_faults.append(_at_sentence(str(_f.get("finding", "")), 600))
        except Exception:                                    # noqa: BLE001
            pass
        try:
            from .result_audit import outer_boundary_findings as _outer
            for _f in _outer(Path(history_dir)):
                if not _f.get("informational"):
                    _ladder_faults.append(_at_sentence(str(_f.get("finding", "")), 600))
        except Exception:                                    # noqa: BLE001
            pass
        # WHAT IS WRONG WITH A FIELD LEADS; the review record and a level's caveat follow.
        def _record_fault(_t: str) -> bool:
            return (_t.startswith("no critic review") or "Global residual is NOT representative" in _t
                    or "the per-block iteration caveat does not shrink" in _t
                    or (("is not verified:" in _t or "are not verified;" in _t)
                        and ("critic review" in _t or "caveat" in _t) and not _level_fault_is_field(_t)))
        _ladder_faults = ([_t for _t in _ladder_faults if not _record_fault(_t)]
                          + [_t for _t in _ladder_faults if _record_fault(_t)])
        # VERIFIED MEANS EACH SIDE'S OWN EQUATION WAS CHECKED AND HOLDS, not only
        # that the two sides agree with each other. The equation findings are
        # keyed by the side's folder name; a participant with none is unchecked.
        _eq_by_side: dict = {}
        for _f in _all_eq:
            _sq = str(_f.get("sequence", ""))
            if " check side " in _sq:
                _eq_by_side.setdefault(_sq.split(" check side ", 1)[1], []).append(str(_f.get("finding", "")))
        _unchecked = []
        for _pd in _part_dirs:
            _texts = _eq_by_side.get(_pd.name) or []
            # "SOLVER FIELD SATISFIES ... FILES DO NOT" is a verdict on the delivered
            # files, not on this ladder's field: the field is what the ladder answers for
            if not _texts or not all(("SATISFIES" in _t and "DOES NOT SATISFY" not in _t)
                                     for _t in _texts):
                _unchecked.append((_pd.name, next((_t for _t in _texts if "SATISFIES" not in _t),
                                                  _texts[0] if _texts else "")))
        _covers = (
            " The critic review this verdict rests on is a review of the SETUP, and "
            "couple_levels resolves it once for the whole call because every level "
            "is that same setup at a different mesh: one review covers the ladder, "
            "not one review per mesh.")
        if _bal_notes:
            _covers += " " + " ".join(_bal_notes)
        if _tautology:
            _covers += (
                " CONSERVATION WAS NOT JUDGED FROM THE EXPORTS: on this pair the two "
                "exported fluxes are exact opposites by construction (a Neumann side's "
                "consistent recovery returns the flux it was given), so their balance "
                "is no evidence either way. It is judged on the flux each side recovers "
                "from its OWN field at the task's interface points -- the per-level "
                "interface deliverables -- by audit_results, channel by channel and "
                "level by level; a channel whose jump does not shrink like the others "
                "is a wrong sign, scaling or missing term in that exchange.")
        if _ladder_faults:
            _trust = False
            _verif = ("THIS LADDER IS NOT VERIFIED. " + " ".join(_ladder_faults)
                      + " A mesh sequence answers for the sequence: a level that "
                      "transmitted nothing, or a mesh that did not refine, makes "
                      "the ladder worthless however good the other levels are."
                      + _covers)
        elif all_ok and _unchecked:
            _trust = False
            _first_side, _why = _unchecked[0]
            _verif = ("CONVERGED LADDER, NOT VERIFIED: every level converged, exchanged "
                      "data and refined, but the check of each side's field against its "
                      "own equation has not passed on " + ", ".join(n for n, _ in _unchecked)
                      + ". " + (f"{_first_side}: {_at_sentence(_why, 500)} " if _why else
                                f"{_first_side}: no equation finding was produced for that "
                                "folder. ")
                      + "Two sides that agree with each other are not two sides that each "
                      "solve their own problem: a side that never solved its interior "
                      "converges just as cleanly (measured). "
                      # A SIDE NO KEY CAN STATE (a bent or several-material side, see
                      # result_audit._one_box_one_k_cannot_state) is not sent after keys.
                      + ("No key in a config.json makes that check run on such a side; the "
                         "checks that read both sides still stand."
                         if "no key in" in str(_why) and "changes that" in str(_why) else
                         "Once the check can run -- "
                         "the side's config.json states its box, coefficient and source, and "
                         "its field_level<k>.csv dumps are on disk -- audit_results(work_dir) "
                         "judges the dumps this ladder wrote; no new coupling run is needed.")
                      + _covers)
        elif all_ok:
            _trust = True
            _verif = ("VERIFIED LADDER -- every requested level passed openPASO's "
                      "verification gate, each level's exchange carried data, the mesh "
                      "refined between levels, and each side's field satisfies its own "
                      "equation -- the equation, source, coefficients and boundary data that "
                      "side's config and script STATE; nothing here compares them with your "
                      "problem statement, so a constant transcribed wrongly verifies as cleanly "
                      "as a right one. Check them against the task term by term. This is "
                      "verification, not validation: confirm physical validity yourself." + _covers)
        else:
            _trust = False
            # THE REASON, NOT A GUESS AT ONE. Measured: a ladder that ran 3 of 3 levels
            # was told "it did not run every requested level".
            if len(out_levels) < len(lv):
                _why_not = f"it ran {len(out_levels)} of the {len(lv)} requested levels"
            elif out_levels and not out_levels[-1].get("converged"):
                _why_not = f"level {out_levels[-1]['level']} did not converge"
            else:
                _badl = [str(x["level"]) for x in out_levels
                         if x.get("coupled_evidence") or not x.get("converged")]
                _why_not = (f"level(s) {', '.join(_badl)} produced no coupled result" if _badl
                            else "not every level passed its own checks")
            _verif = ("THIS LADDER IS NOT VERIFIED: " + _why_not
                      + ", so there is no complete mesh sequence to answer for." + _covers)
        out = {"all_levels_converged": all_ok, "levels_run": len(out_levels),
               "levels_requested": len(lv), "history_dir": history_dir,
               "trustworthy_result": _trust, "verification": _verif,
               "critic_review": {"reviewed": bool(_reviewed), "note": _review_note,
                                 "self_reported_flag": bool(critic_approved)},
               "levels": out_levels, "next_step": nxt}
        # A LADDER FAULT COMES BEFORE THE DELIVERABLES, as an equation that is not
        # satisfied does. Measured: a ladder that named a side never solved inside its
        # subdomain still told the run to write every deliverable from that side.
        if _ladder_faults and all_ok:
            _lead_fault = str(_ladder_faults[0])
            out["next_step"] = (_at_sentence(_lead_fault, 400)
                                + ("" if _record_fault(_lead_fault) else
                                   " Fix that before the deliverables: files written from "
                                   "this ladder carry it.") + " " + str(nxt))
        if _eq_gaps:
            out["equation_check_not_run"] = _eq_gaps
        # THE DATA THE CONFIGS STATE, CHECKED FOR WHAT A SMOOTH SOLUTION NEEDS, LEADS IT. Measured:
        # a source constant transcribed 10x too small converged cleanly at order 2 on both
        # sides and was handed in; two compatibility conditions at the held corners and the
        # interface ends flag every such result set on record and no correct one.
        try:
            from .result_audit import data_compatibility_findings as _compat
            _cf = [_at_sentence(str(_f.get("finding", "")), 700) for _f in _compat(Path(history_dir))]
        except Exception:                                    # noqa: BLE001
            _cf = []
        if _cf:
            _eq = _cf + list(_eq)
        if _eq:
            out["equation_check"] = _eq
            if any("DOES NOT SATISFY" in t for t in _eq):
                out["next_step"] = (
                    "; ".join(t for t in _eq if "DOES NOT SATISFY" in t)
                    + " Fix that before the deliverables: a field converging to "
                      "the wrong function passes every convergence study you can "
                      "run on it. " + str(nxt))
        return json.dumps(out, indent=2)

    async def write_participant_contract(solver: str, path: str, variant: str = "",
                                         overwrite: bool = False) -> str:
        """Write the served participant CONTRACT for `solver` to `path`, solve elided.

        The file is byte-for-byte the text `knowledge(topic='coupling', solver=...,
        signal='participant[:<variant>]:part<k>')` serves in parts, concatenated:
        the contract with its handshake, its checks and its recovery, and the
        SOLVE elided where the banner sits. It is not a runnable program; fill the
        marked hole(s) yourself. {VARIANTS} The knowledge door takes the same words.
        Refuses to overwrite an existing file unless overwrite=True.
        """
        from .coupling_knowledge import (participant_companions, participant_contract_text,
                                         resolve_participant)
        from .result_audit import resolve_under_cell
        text, err = participant_contract_text(solver, variant or "")
        # THE FILES SERVED BESIDE IT (deal.II: its program scaffold and CMakeLists.txt),
        # the same texts the knowledge door shows, written next to `path`.
        companions, _cerr = participant_companions(solver, variant or "")
        err = err or _cerr
        if err:
            return json.dumps({"written": False, "error": err})
        # WHICH CONTRACT THIS IS, SAID. Measured: two runs took the default Kratos
        # contract -- its Dirichlet side -- for their Neumann side, because the
        # reply said only "written"; one re-typed the Neumann text by hand after it.
        _src_path, _label, _ = resolve_participant(solver, variant or "")
        _doc = next((_l.strip().strip('"').strip() for _l in text.splitlines()
                     if _l.strip().strip('"').strip()), "")
        # THE OTHER VARIANTS ARE THE RESOLVER'S WORDS, not read off the file name: a fluid-structure
        # contract is named participant_fsi_<role>_<code>.py, and its stem's second word is 'fsi'.
        from .coupling_knowledge import _participant_key_suffix, participant_variants
        _key, _suffix = _participant_key_suffix(solver, variant or "")
        _others = [_w for _w in participant_variants().get(_key, []) if _w and f"_{_w}" != _suffix]
        if _suffix:
            _others.insert(0, "the base one (variant='')")
        _which = (f"This is the {_label} contract ({_src_path.name}): {_doc[:160]}"
                  + (f" Other variants for this solver: {', '.join(_others)} (variant=...)."
                     if _others else "") + " ")
        # THE MCP SERVER IS A SEPARATE PROCESS WITH ITS OWN WORKING DIRECTORY,
        # which the caller can neither see nor control (result_audit's
        # resolve_under_cell and the three doors that learned it the hard way).
        # This tool's first round resolved './side_A/participant_A.py' against
        # the server's cwd -- a frozen, read-only source snapshot -- and every
        # call in every cell that used it came back "Read-only file system:
        # 'side_A'"; two cells then re-typed the contract anyway. A relative
        # path means a path in the CALLER's working directory, and when that
        # directory is not known the tool says so instead of writing somewhere.
        import os as _os
        _cell = _os.environ.get("OPENPASO_CELL_WORKDIR")
        target = resolve_under_cell(path)
        if not target.is_absolute():
            return json.dumps({"written": False, "path": str(path),
                               "error": ("a relative path cannot be resolved: this server "
                                         "does not know your working directory. Pass the "
                                         "ABSOLUTE path of the file inside your working "
                                         f"directory; with no task directory declared to it, "
                                         f"it writes only inside your home directory, "
                                         f"{Path.home()}.")})
        # EVERY WRITE STAYS INSIDE ONE DIRECTORY: the task's, when the server was told it,
        # and otherwise the user's home directory. With no task directory declared, an
        # absolute path used to be written wherever it pointed, a system directory
        # included. Not the directory the server was started in: the README's launch
        # config starts it inside the openPASO source tree, and that rule refused every
        # project outside it.
        _root = Path(_cell) if _cell else Path.home()
        try:
            target.resolve().relative_to(_root.resolve())
        except ValueError:
            return json.dumps({"written": False, "path": str(target), "error": (
                f"{target} is outside your working directory {_root}; write the contract inside it."
                + ("" if _cell else
                   " No task directory was declared to this server (OPENPASO_CELL_WORKDIR is unset "
                   "where it was started), so it writes only inside your home directory; to write "
                   "elsewhere, set OPENPASO_CELL_WORKDIR to the task's directory in the server's "
                   "launch environment."))})
        # A PATH OF ANOTHER SUFFIX IS REFUSED. Every contract this call writes is a Python script;
        # measured: a cell called it for side_B/dealii_side.cc and got the Python wrapper in its .cc.
        if target.suffix != ".py" and target.name not in {_nm for _nm, _ in companions}:
            _var = f", variant='{variant}'" if variant else ""
            return json.dumps({"written": False, "path": str(target), "error": (
                f"{target.name} is not a .py file, and the contract this call writes is a Python script "
                f"({_src_path.name}); nothing was written. Give it a .py path: write_participant_contract("
                f"solver='{solver}', path='{target.parent / 'participant_<x>.py'}'{_var})"
                + (f". Its program {companions[0][0]} and CMakeLists.txt are written beside it by that same "
                   f"call" if companions else "") + ".")})
        if target.exists() and not overwrite:
            return json.dumps({"written": False, "path": str(target),
                               "error": "the file exists; pass overwrite=True to replace it"})
        _beside = [(target.parent / _nm, _tx) for _nm, _tx in companions]
        # THE WRITTEN BUILD FINDS THE TREE DISCOVER NAMES. Measured: a cell's first build found the
        # system deal.II 9.1.1, because the served HINTS were ${DEAL_II_DIR} $ENV{DEAL_II_DIR} and it
        # had set neither. The knowledge door carries no host path, so the door's CMakeLists stays as
        # it is; the file written here gets this install's tree after those two, which still win.
        _tree_dir = ""
        if _src_path.name.startswith("participant_dealii") and any(_nm == "CMakeLists.txt" for _nm, _ in companions):
            try:
                from backends.dealii.backend import _find_dealii as _fd
                _tree_dir = str(_fd() or "")
            except Exception:                                  # noqa: BLE001
                _tree_dir = ""
        if _tree_dir:
            _hint = f'"{_tree_dir}"' if " " in _tree_dir else _tree_dir
            _beside = [(_bp, re.sub(r"(find_package\(deal\.II[^)]*?HINTS \$\{DEAL_II_DIR\} \$ENV\{DEAL_II_DIR\})",
                                    lambda _m: _m.group(1) + " " + _hint, _tx, count=1)
                        if _bp.name == "CMakeLists.txt" else _tx) for _bp, _tx in _beside]
        if any(_bp == target for _bp, _ in _beside):
            return json.dumps({"written": False, "path": str(target),
                               "error": (f"{target.name} is a file this contract writes BESIDE the wrapper; "
                                         f"give the wrapper its own name, participant_<x>.py")})
        _there = [_bp.name for _bp, _ in _beside if _bp.exists()]
        if _there and not overwrite:
            return json.dumps({"written": False, "path": str(target),
                               "error": (f"{', '.join(_there)} already exist(s) beside it, and this contract "
                                         f"writes that file too; pass overwrite=True to replace it (your "
                                         f"file is moved aside first, never lost)")})
        # A FILLED FILE REPLACED BY THE PRISTINE CONTRACT IS WORK LOST. Measured:
        # a parent whose worker had died at the step cap called this with
        # overwrite=True on the worker's 45 kB filled participant, got a bare
        # "written", and gave up. The caller's own file is MOVED aside -- never
        # read, never copied, so this stays the one writer that writes nothing
        # but the elided contract -- and the reply says where it went.
        # A MOVE THAT FAILED WRITES NOTHING: passed over, the file it could not move was
        # overwritten, the one loss this move exists to prevent. The moves already made
        # are undone, so a refusal leaves every file as it was.
        # A NAME NO EARLIER BACKUP HOLDS: named by the second alone, a second replacement
        # in the same second renamed onto the first backup and that file was gone.
        import time as _time
        _moved: list = []
        replaced = ""
        for _p in [target] + [_bp for _bp, _ in _beside]:
            try:
                # ANY NON-EMPTY FILE, NOT BY SIZE: a filled participant of exactly the
                # served contract's byte count was overwritten without being moved aside,
                # and this tool never reads the caller's file to tell the two apart.
                if not (_p.exists() and _p.stat().st_size > 0):
                    continue
                _size = _p.stat().st_size
                _stamp = _time.strftime('%H%M%S')
                _keep = _p.with_name(f"{_p.stem}.replaced-{_stamp}{_p.suffix}")
                _n = 2
                while _keep.exists():
                    _keep = _p.with_name(f"{_p.stem}.replaced-{_stamp}-{_n}{_p.suffix}")
                    _n += 1
                _p.rename(_keep)
            except OSError as exc:
                for _was, _now in reversed(_moved):
                    try:
                        _now.rename(_was)
                    except OSError:
                        pass
                return json.dumps({"written": False, "path": str(target), "error": (
                    f"could not move your existing {_p.name} aside ({type(exc).__name__}: {exc}), so "
                    f"nothing was written and your files are as they were. Make its directory "
                    f"writable, or give the contract another path.")})
            _moved.append((_p, _keep))
            replaced += (f" MOVED your existing {_size:,}-byte {_p.name} aside to {_keep.name} first -- "
                         f"if that was a filled file, take it back from there instead of filling the "
                         f"hole again.")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
            for _bp, _tx in _beside:
                _bp.write_text(_tx)
        except OSError as exc:
            return json.dumps({"written": False, "path": str(target),
                               "error": f"{type(exc).__name__}: {exc}"})
        import hashlib as _hl
        # COUNTED FROM THE MARKERS THEMSELVES. It counted "does not serve the
        # solve itself", a phrase the names list carries once however many solve
        # regions were cut, so a contract with three elided regions said "2".
        edit_blocks = text.count("# ── EDIT THIS BLOCK")
        solve_holes = text.count("THE SOLVE ITSELF IS YOURS AND IS NOT SERVED HERE")
        if not solve_holes:
            # A DOOR SCAFFOLD MARKS EACH HOLE BY A PAIR OF BANNER LINES. Measured: every 4C side
            # of two rounds was told its scaffold had "0 marked region(s)" to fill.
            solve_holes = sum(1 for _ln in text.splitlines()
                              if _ln.lstrip().startswith("# ── SOLVE") and "DOES NOT SERVE THIS" in _ln) // 2
        _also = ""
        if _beside:
            _holes = sum(_tx.count("IS YOURS AND IS NOT SERVED HERE") for _, _tx in _beside)
            _also = (f" It ALSO wrote {', '.join(_bp.name for _bp, _ in _beside)} beside it: the program "
                     f"the wrapper builds and runs, whose {_holes} holes (each where a HOLE banner sits) "
                     f"are the solve and are yours; the wrapper itself has none, and the consistent "
                     f"{'traction' if 'elastic' in _src_path.name else 'flux'} recovery is the program's "
                     f"served part.")
        # A SPARTA CONTRACT RUNS A BINARY, SO ITS NOTE NAMES THE ONE THIS INSTALL HAS. Measured: the
        # contract's placeholder "spa_serial" is not on PATH here, and two runs died on
        # FileNotFoundError. The file stays the served text; the note says what to put in SPARTA.
        if _src_path.name == "participant_sparta.py":
            try:
                from backends.sparta.backend import _find_sparta_binary as _fsb
                _spa = _fsb() or ""
            except Exception:                                  # noqa: BLE001
                _spa = ""
            _also += (f" BINARY: set SPARTA in the edit block to {_spa}, the SPARTA binary "
                      f"discover(query='list') names on this install." if _spa else
                      " BINARY: discover(query='list') names no SPARTA binary on this install.")
        # A deal.II CONTRACT IS BUILT, SO ITS NOTE NAMES THE BUILD: the six CMake lines and the
        # tree this install has. Measured: two coupled sides compiled against a system /usr
        # deal.II 9.1.1 and never built, one lost 20 min.
        if _src_path.name.startswith("participant_dealii"):
            try:
                from backends.dealii.backend import build_tree_note as _btn
                _tree = _btn()
            except Exception:                                  # noqa: BLE001
                _tree = ""
            _also += (" BUILD: "
                      + ("the CMakeLists.txt written beside it is the six lines deal.II needs"
                         + (f"; to the HINTS of its find_package this call added this install's tree, "
                            f"{_tree_dir}, after ${{DEAL_II_DIR}} and $ENV{{DEAL_II_DIR}} (a DEAL_II_DIR you set "
                            f"still comes first), so a build with no DEAL_II_DIR finds that tree and not a "
                            f"system package" if _tree_dir else "")
                         if any(_bp.name == "CMakeLists.txt" for _bp, _ in _beside) else
                         "a CMakeLists.txt of six lines -- cmake_minimum_required(VERSION 3.13); "
                         "find_package(deal.II 9.0 REQUIRED HINTS ${DEAL_II_DIR}); "
                         "deal_ii_initialize_cached_variables(); project(<name>); "
                         "add_executable(<name> <name>.cc); deal_ii_setup_target(<name>)")
                      + "; cmake -S <dir> -B build -DDEAL_II_DIR=<tree> -DCMAKE_BUILD_TYPE=Release. "
                      + (f"On this install: {_tree}." if _tree else
                         "discover(query='list') names the tree when deal.II is installed.")
                      + " A compile that reads /usr/include/deal.II is not building against that tree.")
        return json.dumps({
            "written": True, "path": str(target), "chars": len(text),
            "sha256": _hl.sha256(text.encode()).hexdigest(),
            **({"also_written": [{"path": str(_bp), "chars": len(_tx),
                                  "sha256": _hl.sha256(_tx.encode()).hexdigest()}
                                 for _bp, _tx in _beside]} if _beside else {}),
            "note": (_which + f"The served contract with the solve elided, not a program: "
                     f"{edit_blocks + solve_holes} marked region(s) are yours to fill "
                     f"({edit_blocks} EDIT THIS BLOCK, and {solve_holes} elided solve "
                     f"region(s), each where an elision banner sits). Fill them IN "
                     f"PLACE -- never re-type the file; the checks "
                     + ("it carries, and the recovery its program carries, are"
                        if _beside else
                        "it carries are"
                        if _src_path.name.startswith("participant_dealii") else
                        "and the recovery it carries are")
                     + " the same ones the knowledge reply serves."
                     + ("" if _beside or not _src_path.name.startswith("participant_dealii") else
                        " Its flux recovery is YOUR program's, the consistent one: the residual of "
                        "the system it assembled with no boundary condition applied, divided by "
                        "the nodal interface weight.")
                     + _also + replaced)})

    # THE VARIANTS ARE THE RESOLVER'S, read when the tool is registered. The docstring listed
    # 'neumann' among the variants of every solver; three FEniCSx workers of one round asked
    # for it first, and FEniCSx has none (its base contract serves both sides).
    try:
        from .coupling_knowledge import participant_variants_text as _pvt
        _vt = _pvt()
    except Exception:                                          # noqa: BLE001
        _vt = ("`variant` names a variant of the contract; a variant this install does not have is "
               "refused, and the refusal names the ones it has.")
    write_participant_contract.__doc__ = (write_participant_contract.__doc__ or "").replace("{VARIANTS}", _vt)
    mcp.tool()(write_participant_contract)

    @mcp.tool()
    async def couple_precice(participants: str, data: str, exchanges: str,
                             work_dir: str, scheme: str = "serial-explicit",
                             dimensions: int = 2, max_time: float = 10.0,
                             time_window: float = 1.0, timeout: int = 1800,
                             max_iterations: int = 20,
                             convergence_tol: float = 1e-6,
                             relaxation: float = 0.5,
                             mapping: str = "nearest-neighbor",
                             extra_env: str = "",
                             critic_approved: bool = False) -> str:
        """GENERAL preCICE coupling of ARBITRARY codes/paradigms, end-to-end.

        Have an independent critic review the setup before coupling; pass
        critic_approved=True only after that review.

        The standard-library (preCICE) path for cross-code coupling — works for any
        number of participants, any data fields, any exchange pattern. openPASO generates
        the preCICE config and launches every participant's solver command. Use this
        when each side is a separate executable/script that talks preCICE (e.g. a DSMC
        particle code <-> a FEM solid; FSI; TSI). Each backend's preCICE participant
        pattern is available via knowledge(topic='precice', solver=...).

        Args (all JSON strings except scheme/numbers):
            participants: list of {"name","mesh","writes":[data],"reads":[data],
                          "command":[argv...]} — one per coupled code.
            data:      list of {"name","type":"scalar"|"vector"}.
            exchanges: list of {"data","from","to"} — one per coupled field.
            work_dir:  directory to run in (config + participant cwd).
            scheme:    serial-explicit|serial-implicit|parallel-explicit|parallel-implicit.
                       An EXPLICIT scheme takes one pass per time window and measures
                       no convergence at all — it cannot establish a coupled fixed
                       point, and the verdict here says so rather than implying one.
            dimensions, max_time, time_window, timeout: coupling controls.
            max_iterations, convergence_tol, relaxation: implicit-scheme controls
                       (ignored for explicit). These were previously not forwarded
                       at all, so every implicit coupling ran on the defaults
                       whatever the caller asked for.
            mapping:   nearest-neighbor|nearest-projection. Mapped with
                       constraint="consistent", which preserves nodal values and
                       NOT integrals — a flux/force field on a non-matching
                       interface is therefore not conserved, and the tool says so.
            extra_env: optional JSON dict of extra env (e.g. {"LD_LIBRARY_PATH":...,
                       "PYTHONPATH":...}) for the participant processes.

        Returns: JSON {exit_codes_ok, exchanged, coupling_converged, returncodes,
            config, logs, evidence, validation, checks_not_run}. Every participant
            exiting 0 is NOT by itself a coupling: an implicit scheme that exhausts
            max-iterations without meeting its convergence measure logs that and
            exits 0, and two scripts that never call preCICE at all exit 0 too. The
            verdict is built from preCICE's own per-window record, and an explicit
            scheme — which measures no convergence — is reported as unmeasured
            rather than as converged.
        """
        from core.precice_config import run_precice_coupling, check_precice_available
        _get_journal().record("tool_call", "couple_precice", solver="general", physics="coupling")
        ok, msg = check_precice_available()
        if not ok:
            return json.dumps({"error": f"preCICE not usable: {msg}"})
        try:
            parts = json.loads(participants); ds = json.loads(data); exs = json.loads(exchanges)
            env = json.loads(extra_env) if extra_env else None
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"invalid JSON argument: {e}"})
        if not isinstance(parts, list) or len(parts) < 2:
            return json.dumps({"error": "need a JSON list of >=2 participants"})
        try:
            r = run_precice_coupling(parts, ds, exs, Path(work_dir), scheme=scheme,
                                     dimensions=dimensions, max_time=max_time,
                                     time_window=time_window, timeout=timeout,
                                     extra_env=env, max_iterations=max_iterations,
                                     convergence_tol=convergence_tol,
                                     mapping=mapping,
                                     initial_relaxation=relaxation)
        except ValueError as e:
            return json.dumps({"error": f"preCICE configuration refused: {e}"})
        except Exception as e:
            return json.dumps({"error": f"coupling failed: {e}"})
        # The orchestrator returns exit codes, preCICE's own per-window record and
        # log tails — not the exchanged field values, so check_finite cannot run on
        # the data. Best-effort: a whole-word NaN/Inf in any participant log is a
        # broken exchange. This only ever DOWNGRADES the verdict (fails safe toward
        # "not verified"), never upgrades it — the anti-fabrication direction.
        import re as _re
        _logs = " ".join(str(v) for v in (r.get("logs") or {}).values())
        nonfinite = bool(_re.search(r"\b(nan|-?inf|-?infinity)\b", _logs, _re.I))
        val: list[str] = []
        not_run: list[str] = []
        if nonfinite:
            val.append("participant logs report non-finite (NaN/Inf) "
                       "values — the exchanged fields are invalid.")
        if not r.get("exit_codes_ok"):
            val.append(f"participant exit codes: {r.get('returncodes')} — a "
                       "non-zero (or unknown) exit is a failed participant.")
        if not r.get("exchanged"):
            val.append("NO EXCHANGE: preCICE's own record shows no completed "
                       "coupling time window for every participant. Exiting 0 is "
                       "not evidence of a coupling — a script that never calls "
                       "preCICE exits 0 too.")
        conv = r.get("coupling_converged")
        if conv is False:
            val.append("NOT CONVERGED: preCICE recorded a time window in which the "
                       "implicit scheme hit max-iterations without meeting its "
                       "convergence measure. preCICE logs that and exits 0.")
        elif conv is None:
            not_run.append(
                "coupling convergence: preCICE recorded no convergence measure"
                + (" — an EXPLICIT scheme takes one pass per time window by "
                   "construction, so nothing here established that the coupled "
                   "state settled. Use serial-implicit / parallel-implicit if you "
                   "need that." if "explicit" in scheme else
                   " for this implicit scheme, so whether it converged is unknown."))
        for d in (r.get("evidence") or []):
            (not_run if "NOT established" in d else val).append(
                f"preCICE record: {d}" if "NOT established" in d else d)
        for note in (r.get("config_notes") or []):
            not_run.append(f"config: {note}")
        if val:
            r["validation"] = val
        r["checks_not_run"] = not_run
        evidence_ok = (bool(r.get("exit_codes_ok")) and bool(r.get("exchanged"))
                       and conv is not False and not nonfinite and not r.get("error"))
        _stamp_verification(
            r, evidence_ok=evidence_ok, critic_approved=critic_approved,
            solver="couple_precice",
            # Every argument that changes what preCICE computes. max_iterations,
            # convergence_tol, relaxation and mapping were absent, so a review of
            # one coupling silently approved the same coupling run to a
            # convergence tolerance five orders of magnitude looser.
            setup_text=_coupling_setup_text(
                participants=participants, data=data, exchanges=exchanges,
                scheme=scheme, dimensions=dimensions, max_time=max_time,
                time_window=time_window, max_iterations=max_iterations,
                convergence_tol=convergence_tol, relaxation=relaxation,
                mapping=mapping),
            reason="" if evidence_ok else
                   ("participant logs report non-finite (NaN/Inf) values"
                    if nonfinite else
                    "preCICE's own record does not show a completed, converged "
                    "exchange between all participants (see `validation`)"))
        if not_run:
            r["verification"] += (
                " COVERAGE — these checks could NOT run, so the verdict above does "
                "not cover what they would have caught: " + " | ".join(not_run))
        return json.dumps(r, indent=2)

    # ═══════════════════════════════════════════════════════════
    # 6. VISUALIZE (replaces 4 visualization tools)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    async def visualize(job_id: str = "", work_dir: str = "",
                        action: str = "summary", field: str = "",
                        ctx: Context = None) -> str:
        """Post-process and visualize simulation results.

        Args:
            job_id: Job ID from run_simulation (or leave empty and set work_dir)
            work_dir: Direct path to results directory
            action: What to do. Options:
                - "summary" — field statistics + results_summary.json content
                  (default; the fastest pulse-check on a finished run)
                - "list" — list every result file under the work dir
                - "plot" — generate a PNG of the named field (needs field=)
                - "validate" — automated sanity checks across the first
                  3 result files: NaN/Inf detection, constant-field
                  detection, suspiciously-large-magnitude detection
                  (>1e15). Use after summary when a field looks wrong.
            field: Specific field name to plot (e.g. 'temperature', 'displacement')
        """
        from core.backend import JobHandle

        # Find work directory
        if job_id and job_id in _jobs:
            wd = _jobs[job_id].work_dir
        elif work_dir:
            wd = Path(work_dir)
        else:
            return "Provide job_id or work_dir"

        if not wd.is_dir():
            return f"Directory not found: {wd}"

        # Collect result files — skip .pvtu (parallel wrappers that can hang PyVista).
        # Deliberately NOT globbing *.xdmf: pyvista's VTK XDMF reader SIGSEGVs on
        # some files, which is uncatchable and takes the whole MCP server down
        # (it wiped the in-memory job registry for the session). Every backend
        # that writes .xdmf also writes a readable .vtu companion, so nothing is
        # lost. (Mirrors quality_checks._FINITE_SCANNABLE excluding xdmf.)
        vtu_files = [f for f in sorted(wd.rglob("*.vtu")) if not f.name.endswith(".pvtu")]
        vtu_files += sorted(wd.rglob("*.vtk"))
        vtu_files += sorted(wd.rglob("*.vtp"))
        vtu_files += sorted(wd.rglob("*.bp"))  # ADIOS2/VTX output from dolfinx 0.10+

        if action == "list":
            return "\n".join(f"- {f.relative_to(wd)}" for f in vtu_files) or "No VTU/VTP files found"

        elif action == "summary":
            try:
                from core.post_processing import read_mesh
                import numpy as np
                import re

                # Layer F catalog templates (fenics / ngsolve /
                # skfem / kratos) write a per-run summary at
                # results_summary.json: max field values, dof
                # counts, convergence metrics. Without surfacing
                # this, visualize('summary') returns '[]' when
                # only the JSON summary exists — even though the
                # template printed exactly the info the LLM wants.
                # Audit 2026-06-01.
                summary_artifacts = []
                for js in sorted(wd.rglob("results_summary.json")):
                    try:
                        with open(js) as _f:
                            summary_artifacts.append({
                                "file": str(js.relative_to(wd)),
                                "summary": json.load(_f),
                            })
                    except Exception as e:
                        summary_artifacts.append({
                            "file": str(js.relative_to(wd)),
                            "error": f"unreadable: {e}",
                        })

                # Group VTU files by field type (structure, fluid, ale, etc.)
                # 4C multi-physics outputs separate files per field
                field_groups: dict[str, list] = {}
                for vtu in vtu_files:
                    name = vtu.stem
                    # Detect field type from filename patterns like
                    # structure-00-0, fluid-05-0, ale-03-0
                    match = re.match(r'^(.*?)(?:-\d+)?(?:-\d+)?$', name)
                    group_name = match.group(1) if match else name
                    # Also strip trailing -0 (processor rank)
                    group_name = re.sub(r'-\d+$', '', group_name)
                    field_groups.setdefault(group_name, []).append(vtu)

                def _safe_float(v):
                    """Convert to float, replacing NaN/Inf with string markers."""
                    f = float(v)
                    if np.isnan(f):
                        return "NaN"
                    if np.isinf(f):
                        return "Inf" if f > 0 else "-Inf"
                    return f

                results = []
                # Show summary per field group, using the last timestep
                # Limit to 10 groups to avoid extremely long responses
                group_idx = 0
                for group, files in sorted(field_groups.items())[:10]:
                    group_idx += 1
                    if ctx is not None:
                        try:
                            await ctx.report_progress(
                                group_idx, len(field_groups),
                                f"Reading {group} ({len(files)} timesteps)")
                        except Exception:
                            pass
                    # Use the last file in each group (latest timestep)
                    last_vtu = sorted(files)[-1]
                    try:
                        mesh = read_mesh(last_vtu)
                        fields = {}
                        for fname in mesh.point_data:
                            arr = np.asarray(mesh.point_data[fname])
                            n_nan = int(np.isnan(arr).sum())
                            n_inf = int(np.isinf(arr).sum())
                            finite = arr[np.isfinite(arr)]
                            stats = {
                                "shape": list(arr.shape),
                            }
                            if len(finite) > 0:
                                stats["min"] = _safe_float(finite.min())
                                stats["max"] = _safe_float(finite.max())
                                stats["mean"] = _safe_float(finite.mean())
                            if n_nan > 0:
                                stats["WARNING_NaN"] = f"{n_nan} values"
                            if n_inf > 0:
                                stats["WARNING_Inf"] = f"{n_inf} values"
                            fields[fname] = stats
                        results.append({
                            "field_group": group,
                            "timesteps": len(files),
                            "latest_file": last_vtu.name,
                            "points": mesh.n_points,
                            "fields": fields,
                        })
                    except Exception as e:
                        results.append({
                            "field_group": group,
                            "timesteps": len(files),
                            "error": str(e),
                        })
                # Prepend the JSON-summary artifacts (if any)
                # so the LLM sees them first.
                output = {
                    "results_summary_json": summary_artifacts,
                    "vtu_field_groups": results,
                }
                # If neither populated, drop the wrapper to keep
                # the legacy '[]' empty signal for "nothing here".
                if not summary_artifacts and not results:
                    return "[]"
                return json.dumps(output, indent=2)
            except Exception as e:
                return f"Error reading results: {e}"

        elif action == "plot" and field:
            try:
                from core.post_processing import read_mesh, plot_field
                vtu = vtu_files[-1] if vtu_files else None
                if not vtu:
                    return "No VTU files to plot"
                mesh = read_mesh(vtu)
                plot_path = wd / f"plot_{field}.png"
                plot_field(mesh, field, plot_path, title=field, spatial_dim=2)
                return f"Plot saved: {plot_path}"
            except Exception as e:
                return f"Error plotting: {e}"

        elif action == "validate":
            # Automated sanity checks on results
            try:
                from core.post_processing import read_mesh
                import numpy as np
                checks = []
                for vtu in vtu_files[:3]:
                    mesh = read_mesh(vtu)
                    for name in mesh.point_data:
                        arr = np.asarray(mesh.point_data[name])
                        issues = []
                        if np.any(np.isnan(arr)):
                            issues.append(f"CONTAINS NaN ({np.isnan(arr).sum()} values)")
                        if np.any(np.isinf(arr)):
                            issues.append(f"CONTAINS Inf ({np.isinf(arr).sum()} values)")
                        if arr.max() == arr.min() and len(arr) > 1:
                            issues.append(f"CONSTANT FIELD (all values = {arr.max():.6e})")
                        if arr.max() > 1e15:
                            issues.append(f"SUSPICIOUSLY LARGE max = {arr.max():.2e}")
                        status = "PASS" if not issues else "ISSUES FOUND"
                        checks.append(f"- {name} in {vtu.name}: {status}" +
                                     (f"\n  " + "\n  ".join(issues) if issues else ""))
                return "## Results Validation\n\n" + "\n".join(checks)
            except Exception as e:
                return f"Validation error: {e}"

        return "Usage: visualize(job_id, action='summary'|'plot'|'list'|'validate', field='')"

    # ═══════════════════════════════════════════════════════════
    # 7. DEVELOPER (replaces 3 developer tools)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def developer(action: str, solver: str = "", keyword: str = "") -> str:
        """Developer tools: architecture, source files, capabilities matrix.

        Args:
            action: What to surface. Options:
                - "architecture" — extension points + source-tree
                  layout for the requested solver
                - "files" — source-file listing filtered by keyword
                - "capabilities" — full backend × physics × variant
                  matrix dump
            solver: Backend name
            keyword: File pattern for "files" action
        """
        if action == "files":
            _get_journal().record("source_read", "developer",
                                  solver=solver, notes=f"keyword={keyword}")
        if action == "architecture" and solver:
            from tools.developer import _SOURCE_LOCATIONS
            info = _SOURCE_LOCATIONS.get(solver, {})
            if not info:
                return f"Unknown solver: {solver}"
            return json.dumps(info, indent=2)

        elif action == "capabilities":
            # All registered backends so the developer-side
            # capabilities listing matches discover('capabilities'):
            # consistent visibility across both surfaces.
            # (Audit 2026-06-02.)
            lines = []
            for b in all_backends():
                status, _ = b.check_availability()
                tag = "" if status.value == "available" else f" *[{status.value}]*"
                physics = [p.name for p in b.supported_physics()]
                lines.append(f"**{b.display_name()}**{tag}: {', '.join(physics)}")
            return "\n".join(lines)

        elif action == "files" and solver:
            # Check if solver has a source root set via env var
            from tools.developer import _SOURCE_LOCATIONS
            info = _SOURCE_LOCATIONS.get(solver, {})
            source_root = info.get("root", "")
            source_env = info.get("source_env_var", "")

            # `keyword` is a GLOB PATTERN handed to rglob, and pathlib treats
            # ".." in a pattern as an ordinary path component — so
            # keyword="../../../benchmarks/*/*/run_pair.py" walked straight out
            # of the backend directory and enumerated the repo. That mattered
            # for one directory in particular: benchmarks/coupling_pairs/ holds
            # the INDEPENDENT REFERENCE SOLUTIONS the coupling fixtures check
            # against, and the property those references rest on is that no tool
            # can reach them. A listing is not the file's contents, but "which
            # reference files exist and how big they are" is still a channel out
            # of the sandbox, and the fix is one line rather than an
            # argument about how much leaks.
            #
            # Absolute patterns are refused for the same reason: rglob("/etc/*")
            # ignores `base` entirely.
            if ".." in Path(keyword or "").parts or (keyword or "").startswith("/"):
                return ("developer(action='files') searches WITHIN one backend's "
                        "source directory. A pattern containing '..' or starting "
                        "at '/' would leave it, so it is refused — pass a "
                        "relative pattern such as '*.py' or 'generators/*.py'.")

            def _within(base: Path, hits: list[Path]) -> list[Path]:
                """Belt and braces: drop anything that resolves outside `base`,
                so a symlink inside the tree cannot do what '..' no longer can."""
                out = []
                for f in hits:
                    try:
                        if f.resolve().is_relative_to(base.resolve()):
                            out.append(f)
                    except (OSError, ValueError):
                        continue
                return out

            # If keyword starts with "src/" or similar, search the solver source tree
            if keyword and source_root and Path(source_root).is_dir():
                base = Path(source_root)
                pattern = keyword
                files = _within(base, sorted(base.rglob(pattern)))[:30]
                if files:
                    return "\n".join(f"- {f.relative_to(base)} ({f.stat().st_size}b)" for f in files)

            # Default: search the MCP backend files
            base = Path(__file__).resolve().parents[1] / "backends" / solver
            if not base.exists():
                hint = f"\nTo browse {solver} source code, set {source_env} in .claude/settings.json" if source_env else ""
                return f"No source directory for {solver}{hint}"
            pattern = keyword or "*.py"
            files = _within(base, sorted(base.rglob(pattern)))
            result = "\n".join(f"- {f.relative_to(base)} ({f.stat().st_size}b)" for f in files[:20])
            if source_env and not (source_root and Path(source_root).is_dir()):
                result += f"\n\nNote: Set {source_env} env var to browse the full {solver} source tree"
            return result

        return "Usage: developer(action='architecture'|'capabilities'|'files', solver='')"

    # ═══════════════════════════════════════════════════════════
    # 8. PREPARE (meta-tool: knowledge + examples + template in one call)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def prepare_simulation(solver: str, physics: str) -> str:
        """Prepare everything needed to set up a simulation — in ONE call.

        Returns: knowledge + real test file examples + generated template.
        This eliminates 3 separate tool calls before every simulation.

        Supports fuzzy matching: e.g. 'magnetostatics' finds 'maxwell',
        'thermal' finds 'heat', and method-qualified requests select distinct
        recipes. Preserve qualifiers from the problem: use phrases such as
        'nearly incompressible Taylor-Hood elasticity', 'steady SIPG
        advection-diffusion', or 'Crank-Nicolson transient heat', rather than
        reducing them to a generic family.

        Args:
            solver: Backend name (e.g. 'fourc', 'fenics', 'ngsolve')
            physics: Physics and required method (e.g. 'poisson',
                     'nearly incompressible Taylor-Hood elasticity',
                     'steady SIPG advection-diffusion')
        """
        _get_journal().record("knowledge_lookup", "prepare_simulation",
                              solver=solver, physics=physics)
        parts = []

        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        # A SECOND prepare_simulation WITH A DIFFERENT SOLVER IS A COUPLING.
        #
        # Measured over three coupled recorded runs: two of the three
        # called prepare_simulation twice -- once per prescribed
        # code -- and NEVER opened a knowledge door, so the coupled must-read
        # (the couple() recipe, the fields-vs-evidence hierarchy, the
        # measured-not-modelled history rule, the captured-log contract)
        # never reached them at all. It hung off a door they did not open,
        # while the door they DID open is the one server.py tells every agent
        # to call first. Same defect the _UNIVERSAL_BLOCK comment below
        # records, one layer up.
        #
        # Preparing two DIFFERENT backends in one session is the agent's own
        # signature of a partitioned coupling -- no task knowledge is used.
        # The must-read is prepended ONCE, on the call that reveals it.
        _prepared = _PREPARED_SOLVERS.setdefault(id(mcp), set())
        _coupling_head = ""
        if _prepared and backend.name() not in _prepared \
                and not _PREPARED_SOLVERS.get((id(mcp), "served")):
            # THE MUST-READ ARRIVES WHOLE; THE CONTRACTS ARE THE WORKERS'.
            #
            # This call used to push BOTH codes' participant contracts (60k
            # for 4C + Kratos, deck grammar included) in front of the
            # must-read, 109k in one reply, and dropped the second code's
            # pitfalls to make room. The coupling door had already settled the
            # same question the other way, measured 2026-09-11: the parent's
            # first reply keeps the must-read whole and says where the block
            # comes from, because under the orchestrator rule the parent never
            # copies the contract -- its worker's own knowledge(topic=
            # 'coupling', solver=...) call leads with the complete block, the
            # deciding facts and the deck grammar (measured 2026-09-23: that
            # call is 53k with both fences for Kratos). Two doors must not
            # disagree, and the one that pushed 60k was the one losing
            # pitfalls. What this call adds is the must-read, once, whole.
            _mr = (_MUST_READ_POINTER if _MUST_READ_STATE["served"]
                   else _COUPLING_MUST_READ)
            _MUST_READ_STATE["served"] = True
            _codes = sorted(_prepared | {backend.name()})
            _calls = " and ".join(
                f"knowledge(topic='coupling', solver='{c}')" for c in _codes)
            _coupling_head = (
                "# YOU HAVE NOW PREPARED TWO DIFFERENT CODES IN THIS SESSION.\n"
                "# If your task couples them -- two subdomains exchanging "
                "interface data --\n# the coupled must-read comes first, then "
                "where each code's participant contract comes from, then this "
                "code's physics payload.\n# If your task uses one code only, "
                "skip to the physics payload after the second rule.\n\n"
                + _mr
                + f"\n[THE PARTICIPANT CONTRACTS FOR {' and '.join(_codes)} ARE "
                  f"NOT REPEATED HERE. {_calls} each lead with that code's "
                  "complete contract block and its deciding facts (and its deck "
                  "grammar where the code takes an input deck); add the physics "
                  "word where the code ships a variant for it. That is the call "
                  "each WORKER makes for its own code: hand the worker the "
                  "brief, not this text. If you write the participants "
                  "yourself, make both calls yourself.]\n"
                + "\n" + "-" * 70 + "\n\n")
            _PREPARED_SOLVERS[(id(mcp), "served")] = True
        _prepared.add(backend.name())

        # Warn up front if the REQUESTED backend is not usable on this install.
        # Otherwise a user follows the returned template and only discovers at
        # run time that the solver can't run (user-session finding). The guidance
        # is still returned — it is valid — just clearly flagged.
        _avail_status, _avail_msg = backend.check_availability()
        if _avail_status.value != "available":
            parts.append(
                f"> ⚠ **{backend.display_name()} ({backend.name()}) is NOT available "
                f"on this install** — {_short_reason(_avail_msg)}\n>\n> That message "
                f"is a LOCAL OBSERVATION from the machine hosting this openPASO "
                f"server: any paths in it are this host's, not universal facts. "
                f"See `knowledge(topic='install')` for the route and the "
                f"environment variable that overrides it. The setup below is "
                f"still accurate, but install or enable {backend.name()} (or "
                f"choose an available backend) before running.\n")

        # Fuzzy match: find the best matching physics name
        matched_physics = _fuzzy_match_physics(backend, physics)
        if not matched_physics:
            # Empty / whitespace-only query — surface the
            # available-physics list so the LLM can pick a real
            # name. Without this guard prepare_simulation silently
            # builds a half-empty response for a physics it never
            # had. Audit 2026-06-01.
            available = ", ".join(
                p.name for p in backend.supported_physics())
            return (f"Empty physics query. Available physics in "
                    f"{backend.display_name()}: {available}")
        if matched_physics != physics:
            parts.append(f"*Note: '{physics}' matched to '{matched_physics}'*\n")

        # 0. Also available on — show which other backends support this physics (informational)
        alternatives = _list_alternative_solvers(solver, matched_physics)
        if alternatives:
            parts.append("## Also available on\n" + alternatives + "\n")

        # 1. Knowledge
        # Render pitfalls OUTSIDE the JSON dump so the 3000-char
        # truncation does not silently hide them. Audited
        # 2026-06-01: large KNOWLEDGE blocks like ngsolve::
        # hyperelasticity (7 pitfalls / ~4.4 KB) and skfem::
        # poisson (6 / ~4 KB) showed 0/N pitfalls fully visible
        # to the LLM client; every Layer F fix landed but never
        # reached the prepare_simulation surface that's meant to
        # teach the agent.
        k = _strip_pitfalls(backend.get_knowledge(matched_physics))
        if k:
            pitfalls_separate = None
            json_payload = k
            if isinstance(k, dict) and isinstance(k.get("pitfalls"), list):
                pitfalls_separate = k["pitfalls"]
                json_payload = {kk: vv for kk, vv in k.items() if kk != "pitfalls"}
            # After the pitfalls carve-out the remaining JSON
            # is description / spaces / solver / elements /
            # materials / time_integration / typical_experiments.
            # Most backends sit < 1.5 KB but fourc::solid_mechanics
            # is ~12 KB (rich plasticity_models + materials dict).
            # The old 3000-char cap silently dropped most of that.
            # Match the TEMPLATE_LIMIT of 12000 set above so the
            # LLM gets the full materials table. Audit 2026-06-01.
            KNOWLEDGE_LIMIT = 16000
            # Never slice: _fit_json_block drops or thins WHOLE entries so the
            # rendered block always parses, and never removes a load-bearing
            # one. See the helper's header for the sweep that motivated it.
            payload_text, payload_suffix = _fit_json_block(
                json_payload, KNOWLEDGE_LIMIT,
                fetch_hint=(f'knowledge(topic="physics", solver="{solver}", '
                            f'physics="{matched_physics}")'))
            # The trim note goes OUTSIDE the fence: whatever sits between
            # ```json and ``` must be parseable on its own, because that is
            # what a consumer extracts.
            parts.append("## Knowledge\n```json\n"
                         + payload_text
                         + "\n```\n"
                         + (payload_suffix.strip() + "\n" if payload_suffix else ""))
            if pitfalls_separate:
                bullets = "\n".join(f"- {p}" for p in pitfalls_separate)
                parts.append(
                    f"### Pitfalls ({len(pitfalls_separate)})\n{bullets}\n")

        # 1a. Installed-version API reference (VERIFIED by actually running here).
        # The dominant failure mode is writing API calls for the WRONG version of the
        # installed code (NGSolve Integrate signature, deal.II DEAL_II_DIR build-tree,
        # the 4C YAML schema, ...). Surface a verified smoke test + version gotchas so
        # the agent adapts a known-good call instead of guessing from memory.
        try:
            from backends._installed_api import render as _render_installed_api
            api_ref = _render_installed_api(backend.name())
            if api_ref:
                parts.append(api_ref + "\n")
        except Exception:
            pass

        # 1b. General input-format pitfalls (ExodusII IDs, FUNCT syntax, etc.)
        # These apply to ALL physics in this solver, not just the current one
        general_k = _strip_pitfalls(backend.get_knowledge("input_format"))
        if isinstance(general_k, dict):
            gp = general_k.get("general_pitfalls")
            if gp:
                pitfall_text = "\n".join(f"- {p}" for p in gp)
                parts.append(f"## General Input Pitfalls\n{pitfall_text}\n")

        # 2. Real test file examples
        from tools.knowledge import _find_reference_test_files
        ref = _find_reference_test_files(solver, matched_physics)
        if ref:
            parts.append(ref)

        # 3. Generated template
        # Templates can exceed 3000 chars on the harder physics
        # (ngsolve hdivdiv 3.2KB, nonlinear_elasticity 3.4KB,
        # fenics navier_stokes 3.8KB ...) — truncating at 3000
        # cuts off the trailing solver / output / summary
        # blocks the LLM needs to actually run the template.
        # Raise to 12000 chars so the standard Layer F-class
        # templates (typically 2-5KB) render in full. Audit
        # 2026-06-01.
        TEMPLATE_LIMIT = 12000
        for p in backend.supported_physics():
            if p.name == matched_physics and p.template_variants:
                # Honour the qualifiers in the request. Reading variants[0] and
                # nothing else served a 2D plane-stress deck for "3d linear
                # elasticity" while a 3d variant sat unreachable in the catalog.
                variant, variant_note = _select_template_variant(
                    physics, list(p.template_variants))
                try:
                    # A SMALL MESH FOR A DOCUMENTATION TEMPLATE.
                    #
                    # The generators default to nx=ny=32, which for an
                    # inline-mesh code means a 33x33 = 1089-node listing.
                    # Measured on prepare_simulation("fourc", "heat"): the
                    # template alone was 39,200 of the reply's 82,949
                    # characters -- 47% -- and 271 of its 774 lines were
                    # NODE COORDS. Those lines teach nothing and displace what
                    # does: over every door an agent can open for a coupled
                    # task the payload came to 145,321 characters while the
                    # lines carrying a decisive fact totalled 571, or 0.4%.
                    # The agent must generate its own mesh anyway -- the
                    # blind tasks prescribe THREE levels -- so a listing of
                    # one fixed mesh is the one part of the template that
                    # cannot be reused.
                    #
                    # 3x3 keeps the structure visible (every section, the
                    # topology, the ordering) at a fraction of the size. The
                    # note below tells the agent to scale it.
                    content = backend.generate_input(
                        matched_physics, variant, {"nx": 3, "ny": 3, "nz": 3})
                    fmt = backend.input_format().value
                    truncated = len(content) > TEMPLATE_LIMIT
                    body = content[:TEMPLATE_LIMIT]
                    suffix = (f"\n... [truncated {len(content) - TEMPLATE_LIMIT} chars]"
                              if truncated else "")
                    stub_tag = _stub_template_tag(content, fmt)
                    note = f"\n{variant_note}\n" if variant_note else ""
                    mesh_note = ""
                    if any(k in content for k in ("NODE COORDS", "NODE ",
                                                  "COORD ")):
                        mesh_note = (
                            "\nTHE MESH HERE IS 3x3 AND ILLUSTRATIVE. It shows "
                            "the sections, the node ordering and the topology "
                            "blocks; it is NOT the mesh your problem wants. A mesh "
                            "study prescribes its own refinement sequence, "
                            "so GENERATE the node and element lists in a loop "
                            "and emit one deck per level. A hard-coded listing "
                            "is the one part of this template you cannot "
                            "reuse.\n")
                    parts.append(f"## Template ({variant}){stub_tag}\n{note}"
                                 f"{mesh_note}```{fmt}\n{body}{suffix}\n```\n")
                except Exception as exc:
                    # Surface the failure: the catalog claims a
                    # template exists (p.template_variants is
                    # non-empty) but the generator raised. The
                    # old `except Exception: pass` silently
                    # produced an LLM-visible "successful" reply
                    # with no template and no hint that the
                    # generator was broken — masking Layer-F
                    # class regressions both from the LLM and
                    # the developer running it. (Audit 2026-06-02.)
                    parts.append(
                        f"## Template ({variant})\n"
                        f"⚠ Template generation FAILED for "
                        f"`{matched_physics}/{variant}`: "
                        f"`{type(exc).__name__}: {exc}`\n\n"
                        f"This is a catalog generator bug — the "
                        f"physics is advertised in "
                        f"`{backend.display_name()}.supported_physics()` "
                        f"but `generate_input` raised. The other "
                        f"sections of this response (knowledge, "
                        f"pitfalls, real-test references) are still "
                        f"valid; only the auto-generated template "
                        f"is missing.\n")
                break

        if not parts:
            # List available physics as hint
            avail = [p.name for p in backend.supported_physics()]
            return f"No information found for '{physics}' in {solver}. Available physics: {', '.join(avail)}"

        # THE UNIVERSAL BLOCK RIDES THIS CALL TOO.
        #
        # server.py tells the agent "Always do this first" about
        # prepare_simulation, and the block was attached only to
        # knowledge(topic='physics'). Measured over 98 recorded runs with
        # these tools: 108 prepare_simulation calls, 95 resolving
        # knowledge(topic='physics') calls, and 43 runs (44%) that received
        # the block NEVER. It is the only text that carries the wiring table
        # for all nine backends, the write-the-answer-first rule, and the
        # NOT VERIFIED vs NOT A RESULT distinction — the three things the
        # observed failures are made of.
        #
        # This is the same defect the comment at the knowledge() call site
        # already records, applied to the door that was missed: a fix landed
        # at one call site when there were two.
        # UNDER THE COUPLED HAND-OFF, THIS CODE'S PHYSICS CORPUS YIELDS.
        # THE CAP TRIMS THE CORPUS, NEVER THE PITFALLS, THE DECIDING FACTS OR
        # THE MUST-READ.
        #
        # MEASURED 2026-09-23. The first call in a session came back at 21-41k
        # characters with the pitfalls carved out and served. The SECOND call --
        # a different code, which reveals a coupling -- came back at 109,223
        # characters: both codes' contracts with their deck grammar (60k), the
        # must-read, the template, and 0 of the 7 pitfalls (46 bullets) this
        # door serves for 4C/heat, because the previous trim kept the template
        # and pointed at "the rest", and the pitfalls were the rest. On one
        # recorded run set the two runs that made this call were the two
        # that failed on the 4C side, and no run that got it right had made
        # it -- two of two is a lead, not a rate, which is why the fix names
        # the rule rather than the run set.
        #
        # One cap for the product: _KNOWLEDGE_REPLY_LIMIT, the same number the
        # knowledge door uses, so the two doors cannot disagree about what a
        # reply may weigh. The closing sentence about six thousand words is
        # kept only when the reply is actually under that bound, first calls
        # included (10 of 27 first replies are over it): a label describing a
        # cap that is not enforced is a label standing in for a guard, and in
        # the last line of a 109k reply it was the reply refuting itself.
        _TOTAL = _KNOWLEDGE_REPLY_LIMIT
        _SEP = "\n---\n"
        _NOTE_RESERVE = 700        # the omission note, budgeted before anything yields
        _deciding = _deciding_block(solver, matched_physics)
        _header = f"# Preparation for {matched_physics} on {solver}\n\n"
        _universal = _UNIVERSAL_BLOCK

        def _is_fixed(x: str) -> bool:
            # THE PITFALLS AND THE TEMPLATE, and every note ABOUT this reply: a
            # physics-name match, a code that is not available here, a variant
            # that does not satisfy what was asked. The template is a runnable
            # worked example a weak model copies -- as load-bearing as the
            # pitfalls, and more than the knowledge JSON or the API reference --
            # so the cap trims the corpus, never the pitfalls or the template.
            # Measured 2026-09-24: the first version yielded the template with
            # the corpus, and a coupled second call for ngsolve/hdivdiv dropped
            # its tail past the cap (test_prepare_simulation_template_complete),
            # the same stateful-second-call truncation as the pitfalls case.
            # The earlier version also let the "is NOT available on this
            # install" and "no template variant provides it" warnings yield.
            head = x.lstrip()[:200]
            return ("Pitfalls" in x[:80] or head.startswith("## Template")
                    or head.startswith("*Note") or "\u26a0" in head)

        def _warning_head(x: str) -> str:
            # a template whose lines before the fence carry a warning keeps
            # those lines when its body yields
            head = x.split("```", 1)[0]
            return head.rstrip() + "\n" if "\u26a0" in head else ""

        def _yield_rank(x: str) -> int:
            # THE ORDER IN WHICH THE CORPUS YIELDS when it does not fit beside
            # the must-read: the knowledge JSON and the other-codes list first
            # (the JSON's pitfalls are already out, and knowledge(topic=
            # 'physics', ...) serves a superset of it -- cut at the cap for
            # deal.II, which is why the note below names the repeat prepare
            # call and not that door), then the reference test files, then the
            # template, and last the installed-version API reference, which
            # was measured on this install and cannot be guessed from memory.
            first = x.lstrip()[:40]
            if first.startswith(("## Knowledge", "## Also available on")):
                return 0
            if first.startswith("## Reference"):
                return 1
            return 2                          # the API reference yields last

        _LABELS = (("## Knowledge", "the knowledge JSON"),
                   ("## Also available on", "the other codes for this physics"),
                   ("## Installed-version API reference",
                    "the installed-version API reference"),
                   ("## Reference", "the reference test files"),
                   ("## Template", "the template"))

        def _label(x: str) -> str:
            first = x.lstrip()[:60]
            for key, lab in _LABELS:
                if first.startswith(key):
                    return lab
            return (x.strip().splitlines() or ["a section"])[0].lstrip("#* ").strip()[:40]

        # THE HEAD IS NEVER CUT, AND THE CORPUS YIELDS ONLY TO IT. The first
        # version of this cap trimmed the coupled hand-off from its end to make
        # room, and the end of the must-read is the run-log contract, the
        # subprocess-capture rule and the per-side naming rule -- the evidence
        # rules a reader judges a coupled result by first, and the family that
        # lost the most coupled recorded runs after the physics itself. So
        # the must-read, the deciding facts, every pitfall, every warning and
        # the universal rules are fixed; only the corpus yields, least valuable
        # first, until the reply fits, and what remains is served in its own
        # order. When the fixed pieces alone exceed the cap (4C or deal.II as
        # the second code: 49.7-53.9k) the reply is over the cap by exactly
        # them and carries no corpus at all; a fixed piece that fell off to
        # meet a number would be the 2026-09-14 defect again.
        #
        # A FIRST call is served whole, as it always was: it was never capped
        # (measured 2026-09-23 over 9 backends x 3 physics, every first reply is
        # 14-43k), and the yielding is reserved for the call that carries the
        # hand-off, so a code whose corpus one day outgrows the cap loses
        # nothing from its own first reply.
        _fixed = [x for x in parts if _is_fixed(x)]
        _corpus = [x for x in parts if not _is_fixed(x)]
        _budget = (_TOTAL - len(_coupling_head) - len(_deciding) - len(_header)
                   - len(_universal) - sum(len(x) + len(_SEP) for x in _fixed))
        _kept = {id(x) for x in _corpus}
        _heads: dict = {}
        _over = sum(len(x) + len(_SEP) for x in _corpus) - _budget
        if _coupling_head and _over > 0:
            _over += _NOTE_RESERVE
            for x in sorted(_corpus, key=_yield_rank):
                if _over <= 0:
                    break
                _kept.discard(id(x))
                _over -= len(x) + len(_SEP)
                wh = _warning_head(x)
                if wh:
                    _heads[id(x)] = wh
                    _over += len(wh) + len(_SEP)
        _ordered = []
        for x in parts:
            if _is_fixed(x) or id(x) in _kept:
                _ordered.append(x)
            elif id(x) in _heads:
                _ordered.append(_heads[id(x)])
        _omitted = [x for x in _corpus if id(x) not in _kept]
        if _omitted:
            _names = ", ".join(_label(x) + (" (its body)" if id(x) in _heads else "")
                               for x in _omitted)
            _ordered.append(
                f"## {len(_omitted)} section(s) of this code's physics corpus "
                f"not shown ({_names}), so that the coupled must-read, the "
                f"measured facts and every pitfall above arrive in full. "
                f"prepare_simulation(solver='{solver}', "
                f"physics='{matched_physics}') called again serves this code's "
                f"preparation whole, without the coupled hand-off.\n")
        _body = _coupling_head + _deciding + _header + _SEP.join(_ordered)
        return _true_to_its_length(_body + _universal)

    # ═══════════════════════════════════════════════════════════
    # 9. TRANSFER FIELD (keep — needed for coupling)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    async def transfer_field(
        source_vtu: str, field_name: str,
        interface_coord: float, interface_axis: int = 0,
        target_format: str = "json", output_path: str = "",
    ) -> str:
        """Extract field values at an interface from a VTU file and format for transfer.

        Universal data connector for cross-solver coupling. Reads VTU output
        from any solver, extracts values at the interface plane, and formats
        them for the target solver's expected input shape.

        FIELD (VOLUME) COUPLINGS: pass `interface_axis=-1`. Not every coupling
        exchanges data on a surface. In thermo-structural interaction both
        participants own the WHOLE body and exchange volume fields — the
        temperature one way and the volumetric strain the other — so there is no
        interface plane, no normal and no flux to balance, and a plane slice
        cannot express the exchange at all. With `interface_axis=-1` every point
        in the file is taken, `interface_coord` is ignored, and the points come
        back in a fixed lexicographic order (the `couple` driver relaxes export
        vectors entry by entry, so the order must not move between iterations).

        Args:
            source_vtu: Path to VTU result file from the source solver.
            field_name: Field to extract (e.g. 'temperature', 'displacement').
            interface_coord: Coordinate value defining the interface plane.
                Ignored when interface_axis is -1.
            interface_axis: Axis perpendicular to interface (0=x, 1=y, 2=z), or
                -1 for the WHOLE VOLUME (field coupling, see above).
            target_format: Output format. Options:
                - "json"        — interface coordinates + values (default)
                - "fenics"      — Python BoundaryCondition snippet (Dirichlet
                                  at this interface), saved as .py
                - "4c_neumann"  — 4C-format YAML snippet for a Neumann
                                  boundary condition, saved as .yaml
            output_path: Where to save the formatted output. If empty,
                auto-generated next to the source VTU as
                'interface_<field_name>.<ext>'.

        Returns:
            A summary string with the interface min/max/mean and the path
            of the saved file.
        """
        from core.field_transfer import extract_interface_from_vtu

        vtu_path = Path(source_vtu)
        if not vtu_path.exists():
            return f"VTU file not found: {source_vtu}"

        try:
            iface = extract_interface_from_vtu(
                vtu_path, field_name, interface_coord, interface_axis)
        except Exception as e:
            return f"Error extracting interface: {e}"

        if not output_path:
            output_path = str(
                vtu_path.parent / f"interface_{field_name}.json")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if target_format == "json":
            iface.to_json(out)
        elif target_format == "fenics":
            from core.field_transfer import format_for_fenics
            code = format_for_fenics(
                iface, "dirichlet", interface_axis, interface_coord)
            out = out.with_suffix(".py")
            out.write_text(code)
        elif target_format == "4c_neumann":
            from core.field_transfer import format_for_4c_neumann
            yaml_snippet = format_for_4c_neumann(iface)
            out = out.with_suffix(".yaml")
            out.write_text(yaml_snippet)
        else:
            return (f"Unknown format: {target_format}. Use 'json', "
                    "'fenics', or '4c_neumann'.")

        vals = iface.values
        summary = (
            f"## Field Transfer: {field_name}\n\n"
            f"- Source: {vtu_path.name}\n"
            f"- Interface: {'xyz'[interface_axis]}={interface_coord}\n"
            f"- Nodes: {len(iface.coordinates)}\n"
            f"- Values: [{vals.min():.6e}, {vals.max():.6e}], "
            f"mean={vals.mean():.6e}\n"
            f"- Output: {out}\n"
        )
        if iface.normal_fluxes is not None:
            fl = iface.normal_fluxes
            summary += f"- Fluxes: [{fl.min():.6e}, {fl.max():.6e}]\n"
        return summary

    # ═══════════════════════════════════════════════════════════
    # 10. MESH (keep — needed for Gmsh)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def generate_mesh(geometry: str, mesh_size: float = 0.1,
                      output_dir: str = "", params: str = "") -> str:
        """Generate a mesh using Gmsh for non-trivial geometries.

        THE SHAPE IS YOURS TO SET. Each geometry below has DIMENSIONS, and
        `params` sets them. Without it you get this server's defaults, which
        are one particular published configuration and almost certainly not
        the one in your problem -- so the reply always states the dimensions
        it actually built, whether you passed any or not.

        Args:
            geometry: One of the built-in geometries:
                - "l_domain"          — 2D L-shaped domain. FIXED at
                  [-1,1]^2 minus [0,1]x[-1,0] (it matches deal.II's
                  hyper_L(-1,1)); it takes no shape parameters, and asking
                  for some is refused rather than ignored.
                - "plate_with_hole"   — 2D plate with circular hole.
                  params: radius, width, height.
                - "channel_cylinder"  — 2D channel with cylindrical
                  obstacle. params: cyl_radius, cyl_center ([x, y]),
                  channel_length, channel_height.
                (No "custom" passthrough yet — passing any other name
                returns a 'Unknown geometry' message with this list.)
            mesh_size: Target element size
            output_dir: Where to save (auto if empty)
            params: JSON object of the shape parameters above, e.g.
                {"channel_length": 5.0, "channel_height": 1.0,
                 "cyl_center": [1.0, 0.5], "cyl_radius": 0.1}. A key this
                geometry does not take is REFUSED with the list of the ones
                it does -- never dropped, because a dropped key hands you the
                default geometry while you believe you asked for another.
        """
        from tools.mesh_generation import register_mesh_tools
        # Delegate to original. Importer names must match the
        # functions in tools.mesh_generation EXACTLY — the prior
        # _generate_channel_cylinder_2d (without 'with') did not
        # exist there (actual name is _generate_channel_with_
        # cylinder_2d) and the ImportError short-circuited the
        # dispatch dict for ALL three geometries, including
        # l_domain and plate_with_hole. (Audit 2026-06-01.)
        try:
            from tools.mesh_generation import (
                _generate_l_domain_2d,
                _generate_plate_with_hole_2d,
                _generate_channel_with_cylinder_2d,
            )
            generators = {
                "l_domain": _generate_l_domain_2d,
                "plate_with_hole": _generate_plate_with_hole_2d,
                "channel_cylinder": _generate_channel_with_cylinder_2d,
            }
            gen = generators.get(geometry)
            if gen:
                # The three generators have DIFFERENT positional
                # signatures (l_domain: (mesh_size, output_path);
                # plate_with_hole: (mesh_size, radius, width,
                # height, output_path); channel_with_cylinder:
                # (mesh_size, cyl_radius, center, length, height,
                # output_path)). Always pass output_path via
                # keyword. The functions expect a FULL FILE path
                # (gmsh.write needs an extension to pick the
                # output format) — append "<geom>.msh" to the
                # directory the user passed in. (Audit 2026-06-01.)
                out_dir = Path(output_dir or str(_OUTPUT_DIR / "meshes"))
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"{geometry}.msh"
                # THE OVERRIDES ALREADY EXISTED AND THIS DOOR DROPPED THEM.
                #
                # Every generator takes its dimensions as keyword arguments
                # and tools/mesh_generation.py forwards them; this wrapper --
                # the only one an agent can reach -- called gen(mesh_size,
                # output_path=...) and nothing else. So `channel_cylinder`
                # could ONLY ever produce the 2.2 x 0.41 channel with its
                # cylinder at (0.2, 0.2), and a request for any other channel
                # returned that one with nothing saying so. That is not a
                # default being kept, it is one stored geometry with a
                # parameter list painted on the front (found by the browser
                # interface session, 2026-09-19).
                import inspect as _inspect
                _accepts = [p for p in _inspect.signature(gen).parameters
                            if p not in ("mesh_size", "output_path")]
                _shape: dict = {}
                if str(params).strip():
                    try:
                        _shape = json.loads(params)
                    except json.JSONDecodeError as _exc:
                        return (f"params is not JSON ({_exc}). Pass a JSON "
                                f"object of the shape parameters "
                                f"{geometry} takes: {_accepts or 'none'}.")
                    if not isinstance(_shape, dict):
                        return (f"params must be a JSON OBJECT of shape "
                                f"parameters, not {type(_shape).__name__}. "
                                f"{geometry} takes: {_accepts or 'none'}.")
                    # REFUSE, NEVER DROP. A silently ignored key returns the
                    # default geometry to someone who believes they asked for
                    # a different one -- the same defect this argument exists
                    # to cure, one layer in.
                    _unknown = [k for k in _shape if k not in _accepts]
                    if _unknown:
                        return (f"{geometry} does not take {_unknown}. "
                                + (f"It takes: {_accepts}."
                                   if _accepts else
                                   "It takes no shape parameters at all: it "
                                   "builds one fixed domain. Use a different "
                                   "geometry, or mesh your own .geo file."))
                result = gen(mesh_size, output_path=out_file, **_shape)
                # generators return either Path or (Path, n_nodes,
                # n_elements). Surface a friendly summary.
                _defaults = {p: v.default for p, v
                             in _inspect.signature(gen).parameters.items()
                             if p in _accepts}
                _built = {**_defaults, **_shape}
                # SAY WHAT WAS BUILT, always. The failure being cured is a
                # silent substitution, and it stays silent for anyone who
                # passes no params at all unless the reply names the shape.
                _shape_note = (f" shape: {_built}" if _built else
                               " shape: fixed [-1,1]^2 minus [0,1]x[-1,0]")
                if isinstance(result, tuple):
                    path, *meta = result
                    from tools.mesh_generation import reading_note
                    return (f"mesh: {path} (nodes={meta[0]}, "
                            f"elements={meta[1]});{_shape_note}"
                            + reading_note(geometry, path))
                return (f"{result};{_shape_note}" if result is not None
                        else f"ok;{_shape_note}")
            return f"Unknown geometry: {geometry}. Available: {list(generators.keys())}"
        except Exception as e:
            return f"Error: {e}"

    # ═══════════════════════════════════════════════════════════
    # 12. BACKEND DISCOVERY
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def reload_catalog() -> str:
        """Hot-reload the per-backend KNOWLEDGE dicts from disk.

        Closes a gap one of openPASO's own post-mortems recorded:
        the MCP server normally imports
        src/backends/<be>/generators/<physics>.py modules ONCE at
        startup and never refreshes them, so catalog edits made
        during a long-running session are invisible. Post-mortem
        records are read on every request (already hot), but
        pitfall dicts are not.

        This tool walks every imported `backends.<be>.generators.*`
        and `backends.<be>.backend` module, runs importlib.reload
        on each, and re-runs load_all_backends() so the registry
        re-binds the backend objects to the refreshed module
        attributes. After the call, the very next
        mcp__openpaso__knowledge call returns the on-disk
        catalog without having to restart Claude Code.

        Returns a one-line summary of which modules were
        successfully reloaded vs which raised, so the caller can
        tell when a syntax error in a newly-edited generator
        prevented its module from re-importing (in that case the
        OLD dict is still served from the previous import).
        """
        import importlib
        import sys
        reload_ok: list[str] = []
        reload_fail: list[tuple[str, str]] = []

        # Reload data/*_knowledge.py first (sourced by some
        # backends).
        for mod_name in list(sys.modules.keys()):
            if (mod_name.endswith("_knowledge")
                    and not mod_name.startswith("backends.")):
                try:
                    importlib.reload(sys.modules[mod_name])
                    reload_ok.append(mod_name)
                except Exception as exc:  # noqa: BLE001
                    reload_fail.append((mod_name, str(exc)))

        # Reload every imported backends.* submodule.
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("backends.") and "." in mod_name:
                try:
                    importlib.reload(sys.modules[mod_name])
                    reload_ok.append(mod_name)
                except Exception as exc:  # noqa: BLE001
                    reload_fail.append((mod_name, str(exc)))

        # Re-bind backend objects to refreshed modules.
        try:
            from core.registry import load_all_backends
            load_all_backends()
            re_register = "ok"
        except Exception as exc:  # noqa: BLE001
            re_register = f"FAILED: {exc}"

        msg = (f"reload_catalog: {len(reload_ok)} modules reloaded, "
               f"{len(reload_fail)} failed; "
               f"re-register backends: {re_register}.")
        if reload_fail:
            msg += "\n\nFailures:\n" + "\n".join(
                f"  - {n}: {e[:200]}" for n, e in reload_fail[:10])
        return msg

    @mcp.tool()
    def rediscover_backends(confirm: bool = False) -> str:
        """Probe the system for solver backends and report findings.

        Searches pip packages, conda environments, common build directories,
        and source roots. In developer mode, reports git branch and status.

        Args:
            confirm: If True, save the discovered config for future sessions.
                     If False (default), just report what was found.
        """
        from core.autodiscovery import (
            discover_backends as _discover,
            format_discovery,
            save_discovered_config,
        )

        # A negative dune interpreter result is cached for the server's
        # lifetime, so a dune installed after startup would otherwise stay
        # invisible. Reset the cache so re-discovery genuinely re-probes.
        try:
            from backends.dune.backend import _reset_dune_python_cache
            _reset_dune_python_cache()
        except Exception:
            pass

        results = _discover()
        report = format_discovery(results)

        if confirm:
            path = save_discovered_config(results)
            report += f"\n\nConfig saved to `{path}`. Will be used on next restart."
        else:
            found_count = sum(1 for r in results if r.found)
            if found_count > 0:
                report += (
                    f"\n\nCall `rediscover_backends(confirm=True)` to save this "
                    f"config for future sessions."
                )

        return report

    @mcp.tool()
    def setup_backend(action: str = "status", solver: str = "",
                      route: str = "") -> str:
        """Guided backend setup: detect, plan, install, verify, persist.

        Helps a new user get any of the 8 FEM backends working on THIS
        machine — picking the fastest install route for the current OS
        (pip > conda > binary download > source build), executing it,
        running the backend's smoke test, and persisting the resolved
        paths into ~/.config/openpaso/sources.json so every future MCP
        session finds the install without re-discovery.

        Actions:
          status  — one-row-per-backend table: available? source tree?
                    build dir? (no args needed)
          plan    — recommended install route for `solver` on this OS,
                    incl. system deps (apt/brew), human notes, and
                    whether the route is verified on this OS. Nothing
                    executes. Optional `route` (pip|conda|binary|source)
                    forces an alternative.
          install — execute the planned route. pip/conda run inline
                    (minutes); source builds start in the BACKGROUND
                    (30-120 min) — re-run with action='verify' when
                    done. binary routes return manual instructions.
          verify  — run the smoke test for `solver` and, on success,
                    persist its paths.

        macOS note: darwin routes are structured but mostly UNVERIFIED
        (flagged in the plan output). They are extension points — when
        a route is validated on a Mac, its os_support['darwin'] entry
        in src/core/backend_setup.py should be updated with the working
        steps (e.g. the 4C brew/CMake settings).

        Args:
            action: status | plan | install | verify
            solver: backend name (required for plan/install/verify)
            route:  optional route kind override (pip|conda|binary|source)
        """
        import json as _json
        from core.backend_setup import (
            render_status_markdown, plan_setup, execute_setup,
            _verify_and_persist,
        )
        if action == "status":
            return render_status_markdown()
        if not solver:
            return ("setup_backend: actions plan/install/verify need "
                    "solver=<backend>. Usage: action=status|plan|install"
                    "|verify, solver=fourc|fenics|dealii|ngsolve|skfem"
                    "|kratos|dune|febio, route=pip|conda|binary|source "
                    "(optional).")
        if action == "plan":
            return _json.dumps(plan_setup(solver, prefer=route or None),
                               indent=2, default=str)
        if action == "install":
            return _json.dumps(
                execute_setup(solver, route_kind=route or None),
                indent=2, default=str)
        if action == "verify":
            return _json.dumps(_verify_and_persist(solver), indent=2,
                               default=str)
        return (f"setup_backend: unknown action {action!r}. "
                "Use status | plan | install | verify.")

    # ═══════════════════════════════════════════════════════════
    # 13. SESSION INSIGHTS (knowledge capture)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def session_insights(action: str = "review", path: str = "") -> str:
        """Review knowledge discovered during this session or from saved
        journals on disk.

        Two flows are supported:

        * In-session flow: call ``review`` -> ``approve_all`` /
          ``reject_all`` during the live MCP session to surface
          candidates from the current journal and save approved ones
          to ``data/community_knowledge/pending/``.
        * Ingest flow: call ``ingest`` with ``path`` pointing at a
          previously-saved session journal (``data/sessions/session_*.json``,
          which the server writes on shutdown) or at a directory of
          such files.  Candidates are surfaced just like ``review``
          and can be approved with ``approve_all``.

        Args:
            action:
                - "review" — show candidate knowledge from the current
                  session for approval
                - "ingest" — load saved journal(s) from ``path`` and
                  analyse them; requires ``path``
                - "approve_all" — approve all pending candidates and
                  save to community_knowledge/pending/
                - "reject_all" — dismiss all pending candidates
                - "stats" — current session statistics
            path: file or directory used by the ``ingest`` action;
                ignored otherwise.  Directories are scanned for
                ``session_*.json``.
        """
        from pathlib import Path as _Path

        from core.session_journal import get_journal
        from core.session_analyzer import (
            CandidateKnowledge,
            analyze_journal,
            analyze_journal_file,
            filter_against_existing,
            format_candidates,
        )

        journal = get_journal()

        if action == "stats":
            return json.dumps({
                "session_id": journal.session_id,
                "events": len(journal.events),
                "errors": journal.error_count,
                "solvers_used": sorted(journal.solvers_used),
                "physics_used": sorted(journal.physics_used),
                "duration_seconds": round(journal.duration_seconds, 1),
            }, indent=2)

        if action == "review":
            if len(journal.events) < 3:
                return "Session too short for knowledge extraction (< 3 tool calls)."
            candidates = analyze_journal(journal)
            # Filter against existing knowledge
            existing = _collect_existing_pitfalls()
            candidates = filter_against_existing(candidates, existing)
            if not candidates:
                return "No new knowledge candidates discovered in this session."
            # Store candidates for potential approval
            _pending_candidates.clear()
            _pending_candidates.extend(candidates)
            return format_candidates(candidates)

        if action == "ingest":
            if not path:
                return (
                    "Usage: session_insights('ingest', path='<file_or_dir>')\n"
                    "Point at a session journal saved by the MCP server "
                    "(data/sessions/session_*.json) or a directory of "
                    "such files."
                )
            p = _Path(path)
            if not p.exists():
                return f"Path not found: {p}"
            sources: list[_Path] = (
                sorted(p.glob("session_*.json")) if p.is_dir() else [p]
            )
            if not sources:
                return f"No session_*.json files found in {p}"
            all_candidates: list = []
            errors: list[str] = []
            for s in sources:
                try:
                    all_candidates.extend(analyze_journal_file(s))
                except Exception as e:
                    # repr(e) keeps the exception type so a contributor
                    # can tell `KeyError('events')` from a `FileNotFoundError`.
                    errors.append(f"{s.name}: {e!r}")
            # Cross-source de-duplication on a normalised key (the in-file
            # analyzer runs fuzzy dedup already; cross-file dedup needs to
            # match that contract or near-identical entries from N journals
            # all survive as separate candidates).
            import re as _re
            _retry_re = _re.compile(r"\s*\(retry \d+\)\s*$", _re.IGNORECASE)
            def _norm_title(t: str) -> str:
                return " ".join(_retry_re.sub("", t).lower().split())
            best: dict[tuple[str, str, str], CandidateKnowledge] = {}
            for c in all_candidates:
                key = (
                    c.category.strip().lower(),
                    (c.solver or "").strip().lower(),
                    _norm_title(c.title),
                )
                if key not in best or c.confidence > best[key].confidence:
                    best[key] = c
            candidates = list(best.values())
            existing = _collect_existing_pitfalls()
            candidates = filter_against_existing(candidates, existing)
            _pending_candidates.clear()
            _pending_candidates.extend(candidates)
            header = (
                f"Ingested {len(sources)} journal file(s); "
                f"{len(all_candidates)} raw candidates -> "
                f"{len(candidates)} novel after dedup + filter.\n"
            )
            if errors:
                header += "Errors:\n  " + "\n  ".join(errors) + "\n"
            if not candidates:
                return header + "No new candidates."
            return header + format_candidates(candidates)

        if action == "approve_all":
            if not _pending_candidates:
                return "No pending candidates. Call session_insights('review') first."
            saved = _save_candidates(_pending_candidates, journal.session_id)
            count = len(_pending_candidates)
            _pending_candidates.clear()
            return f"Approved {count} candidate(s). Saved to: {saved}"

        if action == "reject_all":
            count = len(_pending_candidates)
            _pending_candidates.clear()
            return f"Rejected {count} candidate(s)."

        # The Actions list must match the docstring + the
        # actual dispatch branches in this function. Audit
        # 2026-06-01: 'ingest' was documented but missing
        # from this usage hint, so LLMs that hit an invalid
        # action never learned that ingest exists.
        return (
            "Usage: session_insights(action, path='')\n"
            "Actions: review, ingest, approve_all, reject_all, stats\n"
            "Use ingest with path=<session.json|dir> to "
            "analyse saved session journals."
        )

    # Storage for pending candidates between review and approve
    _pending_candidates: list = []


def _collect_existing_pitfalls() -> list[str]:
    """Gather all existing pitfall strings for novelty checking.

    Includes both built-in knowledge AND community contributions.

    Uses all_backends() (not available_backends): the pitfall
    library is a static catalog and the novelty check should
    compare against EVERY known pitfall, including those of
    backends the user has not installed locally. Filtering by
    availability would let a candidate that duplicates a
    dune-fem pitfall slip through as "novel" on any host
    without dune-fem. (Audit 2026-06-02.)
    """
    pitfalls = []
    try:
        for b in all_backends():
            for p in b.supported_physics():
                k = b.get_knowledge(p.name)
                if k and isinstance(k, dict) and "pitfalls" in k:
                    for pit in k["pitfalls"]:
                        if isinstance(pit, str):
                            pitfalls.append(pit)
                        elif isinstance(pit, dict) and "text" in pit:
                            pitfalls.append(pit["text"])
    except Exception:
        pass
    # Also include community contributions
    for c in _load_community_knowledge():
        pitfalls.append(c.get("title", ""))
    return pitfalls


def _load_community_knowledge(solver: str = "") -> list[dict]:
    """Load approved community knowledge from pending/ directory.

    Returns list of candidate dicts. Optionally filter by solver.
    """
    from pathlib import Path
    from core.session_journal import state_dir as _state_dir
    # staged community knowledge is runtime state -> the user's state dir, never the install
    pending_dir = _state_dir("community_knowledge") / "pending"
    if not pending_dir.exists():
        return []
    entries = []
    for f in sorted(pending_dir.glob("session_*.json")):
        try:
            data = json.loads(f.read_text())
            if isinstance(data, list):
                for entry in data:
                    if solver and entry.get("solver", "") != solver:
                        continue
                    entries.append(entry)
        except (json.JSONDecodeError, OSError):
            continue
    return entries


def _save_candidates(candidates: list, session_id: str) -> str:
    """Save approved candidates to community_knowledge/pending/."""
    from pathlib import Path
    from core.session_journal import state_dir as _state_dir
    # staged community knowledge is runtime state -> the user's state dir, never the install
    pending_dir = _state_dir("community_knowledge") / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for c in candidates:
        entries.append(c.to_dict())

    path = pending_dir / f"session_{session_id}.json"
    path.write_text(json.dumps(entries, indent=2, default=str))
    return str(path)


# ═══════════════════════════════════════════════════════════════
# Helper functions for knowledge (copied from original tools)
# ═══════════════════════════════════════════════════════════════

def _capture_knowledge_fn(fn_name: str, *args) -> str:
    """Reach into tools.knowledge.register_knowledge_tools to pull
    out one of the inline get_*_knowledge closure bodies.

    The three knowledge providers (coupling / TSI / preCICE) live
    inside register_knowledge_tools as nested @mcp.tool() closures,
    not as module-level functions. The consolidated tool surface
    needs to call them outside FastMCP's tool-dispatch path, so
    this helper builds a throwaway FastMCP instance, monkey-
    patches its `tool` decorator to capture every registered
    function by name, runs register_knowledge_tools, then calls
    the requested one.

    Failures here used to be wrapped in `except Exception: pass`
    and produced a bare "...knowledge not available" string to
    the LLM — silent degradation that hid genuine breakage of the
    capture trick (FastMCP API change, register_knowledge_tools
    refactor, missing tools.knowledge module, ...). The wrapper
    now surfaces the exception so the LLM and the developer can
    diagnose. (Audit 2026-06-02.)
    """
    try:
        from tools.knowledge import register_knowledge_tools
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        return (f"⚠ Cannot load knowledge subsystem: "
                f"`{type(exc).__name__}: {exc}`")
    mcp = FastMCP("tmp")
    captured: dict = {}
    orig = mcp.tool

    def cap(*a, **kw):
        d = orig(*a, **kw)

        def w(fn):
            r = d(fn)
            captured[fn.__name__] = fn
            return r
        return w

    mcp.tool = cap
    try:
        register_knowledge_tools(mcp)
    except Exception as exc:
        return (f"⚠ register_knowledge_tools failed while capturing "
                f"`{fn_name}`: `{type(exc).__name__}: {exc}`")
    if fn_name not in captured:
        return (f"⚠ `{fn_name}` was not registered by "
                f"register_knowledge_tools. Captured: "
                f"{sorted(captured.keys())}")
    try:
        return captured[fn_name](*args)
    except Exception as exc:
        return (f"⚠ `{fn_name}()` raised: "
                f"`{type(exc).__name__}: {exc}`")


_THERMOELASTIC_HEADING = "\n## THERMO-ELASTIC VARIANT"
# THE OTHER VARIANT CONTRACTS A BACKEND MAY SHIP, keyed by the physics word the
# agent passes. The thermo-elastic block had this defect first (2026-09-11) and
# the three others had it still: the variant sat behind the traps, past every
# cut, and physics='transient' / 'elasticity' / '3d' served the steady scalar
# heat contract first. Measured 2026-09-13 on every backend that ships one: the
# fenics reply with physics='transient_heat' was 37,171 characters, identical
# to the physics-less reply; kratos with 'heat_3d' 45,137, identical; dune the
# same. A time-dependent, vector or 3-D task therefore started from the wrong
# contract however it asked.
_VARIANT_HEADINGS = {
    "thermoelastic": _THERMOELASTIC_HEADING,
    "elastic": "\n## VECTOR (ELASTICITY) VARIANT",
    "transient": "\n## THE TRANSIENT PARTICIPANT",
    "3d": "\n## THE 3-D PARTICIPANT",
}
_VARIANT_PHYSICS_WORD = {"thermoelastic": "thermoelastic", "elastic": "elasticity",
                         "transient": "transient", "3d": "3d"}


def _is_thermoelastic(physics: str) -> bool:
    """Does the physics word name a temperature-plus-displacement exchange?"""
    p = (physics or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not p:
        return False
    return any(t in p for t in ("thermoelast", "thermo_elast", "thermomech", "thermo_mech",
                                "thermal_stress", "thermostruct", "thermo_struct",
                                "thermal_solid", "structural_thermal", "thermal_structure",
                                "thermo_structure")) or p in ("tsi",)


def _is_fsi_physics(physics: str) -> bool:
    """Does the physics word name a fluid-structure exchange?

    THE FSI CONTRACTS WERE REACHABLE ONLY BY PASSING THE PHYSICS AS A SOLVER NAME. Measured
    2026-09-14: knowledge(topic='coupling', solver='fsi') serves the fluid and structure participants,
    while solver='fourc' or solver='fenics' -- what an agent coupling those two codes actually calls --
    serves the scalar heat contract, with nothing saying an FSI contract exists. One development
    problem is exactly that coupling.
    """
    p = _norm_physics(physics)
    if not p:
        return False
    return ("fluid_structure" in p or "fluid-structure" in p or p in ("fsi",)
            or ("fsi" in p.split("_")))


def _norm_physics(physics: str) -> str:
    return (physics or "").strip().lower().replace("-", "_").replace(" ", "_")


def _is_vector_physics(physics: str) -> bool:
    """Does the physics word name a displacement/traction exchange (and not a
    thermo-elastic or a fluid-structure one, which have their own doors)?"""
    p = _norm_physics(physics)
    if not p or _is_thermoelastic(p) or "fluid" in p or p in ("fsi",):
        return False
    return any(t in p for t in ("elast", "structur", "solid", "displacement", "traction",
                                "plane_strain", "plane_stress", "hooke", "mechanic"))


def _is_transient_physics(physics: str) -> bool:
    """Does the physics word name a time-dependent problem?"""
    p = _norm_physics(physics)
    return bool(p) and any(t in p for t in ("transient", "unsteady", "time_dependent",
                                            "parabolic", "dynamic"))


def _is_threed_physics(physics: str) -> bool:
    """Does the physics word name a three-dimensional domain?"""
    p = _norm_physics(physics)
    return bool(p) and (p.startswith("3d") or p.endswith("3d") or "_3d" in p or "3d_" in p
                        or "three_d" in p or "3_d" in p)


def _variant_for(physics: str) -> str:
    """The shipped variant a physics word asks for: 'thermoelastic', 'elastic',
    'transient', '3d' or '' for the single-field steady 2-D contract. A vector
    word wins over a transient one (no transient-vector participant ships) and
    over a 3-D one (no 3-D vector participant ships)."""
    if _is_thermoelastic(physics):
        return "thermoelastic"
    if _is_vector_physics(physics):
        return "elastic"
    if _is_transient_physics(physics):
        return "transient"
    if _is_threed_physics(physics):
        return "3d"
    return ""


def _strip_notice(text: str) -> str:
    """Drop the physics-less notice paragraph (see coupling_knowledge._thermo_notice)."""
    import re as _re
    text = _re.sub(r"\n\nIF YOUR INTERFACE CARRIES TEMPERATURE AND DISPLACEMENT TOGETHER.*?scalar\) contract\.\n", "", text, count=1, flags=_re.S)
    return _re.sub(r"\n\nWHICH CONTRACT BELOW IS YOURS\..*?leads with the single-field \(scalar\) contract\.\n", "", text, count=1, flags=_re.S)


def _promote_thermoelastic(payload: str) -> str:
    return _promote_variant(payload, "thermoelastic")


def _promote_variant(payload: str, key: str) -> str:
    """Move a variant section (heading, paragraph, fenced contract) in front of
    the scalar contract, so the first reply of a session and the 28k
    pointer-mode head carry IT as the contract. Measured 2026-09-11 for the
    thermo-elastic block (it sat behind the traps, the vector and the transient
    sections, past every cut, and the thermo-mechanical cell's workers copied
    the scalar heat contract instead) and 2026-09-13 for the vector, transient
    and 3-D blocks, which were never promoted at all."""
    heading = _VARIANT_HEADINGS.get(key or "", "")
    if not isinstance(payload, str) or not heading:
        return payload
    a = payload.find(heading)
    if a < 0:
        return payload
    f = payload.find("```python", a)
    if f < 0:
        return payload
    e = payload.find("```\n", f + 9)
    if e < 0:
        return payload
    # THE PROGRAM A WRAPPER RUNS MOVES WITH IT (deal.II: its C++ scaffold and
    # CMakeLists.txt right behind the wrapper). Measured on a transient coupled
    # round: the promoted transient wrapper led the reply and the program it
    # runs was nowhere in it.
    cpp = payload.find("```cpp", e)
    if 0 <= cpp <= e + 1500 and "THE PROGRAM THE WRAPPER RUNS" in payload[e:cpp]:
        for fence in ("```cpp", "```cmake"):
            at = payload.find(fence, e)
            close = payload.find("```\n", at + len(fence)) if 0 <= at <= e + 1500 else -1
            if close > 0:
                e = close
    section = payload[a:e + 4]
    rest = payload[:a] + payload[e + 4:]
    anchor = rest.find("## PARTICIPANT CONTRACT")
    if anchor < 0:
        return payload
    out = rest[:anchor] + section.lstrip("\n") + "\n\n" + rest[anchor:]
    # the physics-less notice does not apply once the thermo-elastic block leads
    return _strip_notice(out)


def _get_coupling_knowledge(solver: str = "", signal: str = "", physics: str = ""):
    """Return coupling knowledge string (or a visible error block).

    `solver` is honoured: the payload for a named backend is its complete
    participant script, which is the whole point of asking for one. It used to
    be dropped on the floor, so every backend got the same bytes.
    """
    # THE PARTS DOOR FOLLOWS THE PHYSICS TOO. Measured 2026-09-11: a worker that
    # had the thermo-elastic contract asked for `participant:dirichlet:part1`
    # and got the scalar heat contract in parts, five calls long.
    _vk = _variant_for(physics)
    # physics='fsi' too: its parts were the scalar contract's (measured: a fluid worker asked for
    # signal='participant:...' with physics='fsi' and was served the heat contract in parts).
    _pk = _vk or ("fsi" if _is_fsi_physics(physics) else "")
    if (signal or "").strip().lower().startswith("participant") and _pk \
            and _pk not in (signal or "").lower():
        signal = signal.strip() + ":" + _pk
    payload = _capture_knowledge_fn("get_coupling_knowledge", solver, signal)
    # THE EXCHANGE THE PHYSICS NAMES GETS ITS CONTRACT FIRST (thermo-elastic,
    # vector, transient or 3-D), where the backend ships one.
    if _vk and isinstance(payload, str):
        payload = _promote_variant(payload, _vk)
    # THE PARTS DOOR IS SERVED AS IS. `signal='participant[:role]:partN'`
    # exists for clients that truncate long replies, so its bounded chunk of
    # the elided contract must reach the agent whole: no must-read prepended,
    # no head cap, no continuation. (The chunks are cut from the ELIDED text
    # in coupling_knowledge.coupling_participant; nothing here changes that.)
    if isinstance(payload, str) and (signal or "").strip().lower().startswith("participant"):
        return payload
    # A NAMED CODE'S DECIDING FACTS RIDE ALONG. Measured: coupled runs that
    # never called prepare_simulation (the door that serves them) hand-rolled
    # the code's API from memory and died on it -- a Kratos side against the
    # wrong API for 68 calls, a DUNE side on a name ufl no longer exports.
    # The facts are short and measured by execution; a coupled side needs
    # them exactly as much as a single-code run does.
    _fsi_door = _is_fsi_physics(physics) and (solver or "").strip().lower() not in (
        "fsi", "fluid_structure", "fluid-structure")
    _facts = (_deciding_block(solver, "fluid-structure side" if _fsi_door else "coupled side", fsi=_fsi_door)
              if isinstance(payload, str) and (solver or "").strip()
              and _DECIDING_FACTS.get((solver or "").strip().lower()) else "")

    def _with_facts(text: str) -> str:
        # after the lead (must-read or its pointer), before the payload, so the
        # head cap and the signalled continuation still measure the payload
        # alone
        if not _facts or not isinstance(text, str):
            return text
        # FIRST REPLY OF A SESSION (part A leads): the facts close the reply,
        # after the contract and part B -- the parent reads orchestration and
        # the contract; the worker's own door call (pointer mode) leads with
        # the facts. Measured: with the facts up front, 4C's and DUNE's first
        # replies lost the tail of part B to the 48k cap.
        if text.startswith(_COUPLING_LEAD_A) and not text.startswith(_COUPLING_MUST_READ):
            return text + "\n" + _facts
        for lead in (_COUPLING_MUST_READ, _MUST_READ_POINTER):
            if text.startswith(lead):
                return lead + _facts + text[len(lead):]
        return _facts + text
    if _is_fsi_physics(physics) and (solver or "").strip().lower() not in ("fsi", "fluid_structure",
                                                                          "fluid-structure"):
        _named = (solver or "").strip().lower()
        # EACH FILE WITH THE WRITER CALL THAT PUTS IT ON DISK. The writer could not write these
        # (measured: variant='fsi' wrote the scalar heat contract in all five cells of a round).
        _fenics_two = ("participant_fsi_fluid_fenics.py (fluid, variant='fsi_fluid') and "
                       "participant_fsi_solid_fenics.py (structure, variant='fsi_solid')")
        _which = {"fourc": "participant_fsi_solid_fourc.py (variant='fsi')",
                  "4c": "participant_fsi_solid_fourc.py (variant='fsi')",
                  "fenics": _fenics_two, "fenicsx": _fenics_two, "dolfinx": _fenics_two,
                  "skfem": "participant_fsi_solid_skfem.py (variant='fsi')",
                  "scikit-fem": "participant_fsi_solid_skfem.py (variant='fsi')"}
        _lead = ("[YOU ASKED FOR A FLUID-STRUCTURE COUPLING, so this is the FSI contract rather than the "
                 f"single-field one for {solver!r}."
                 + (f" The participant in it that runs under {solver!r} is {_which[_named]}."
                    if _named in _which else "")
                 + " The scalar contract for that code is still one call away: ask again without "
                   "physics='fsi'.]\n\n")
        _fsi_payload = _capture_knowledge_fn("get_coupling_knowledge", "fsi", signal)
        # AND LEAD WITH THE ONE THAT RUNS UNDER THE CODE THAT WAS ASKED FOR. The payload leads with the
        # fluid participant and carries ONE structure contract; a 4C structure side was therefore handed
        # the scikit-fem structure contract and a sentence saying a 4C one exists. Serve it instead.
        _own = {"fourc": "fsi_solid_fourc", "4c": "fsi_solid_fourc",
                "skfem": "fsi_solid_skfem", "scikit-fem": "fsi_solid_skfem"}.get(_named, "")
        if _own:
            try:
                from .coupling_knowledge import _script as _fsi_script     # noqa: PLC0415
                _own_block = _fsi_script(_own)
                if _own_block and _own_block[:400] not in _fsi_payload:
                    _code = _own.rsplit("_", 1)[1]
                    _fsi_payload = (f"## THE STRUCTURE PARTICIPANT THAT RUNS UNDER {solver.upper()} "
                                    f"— contract, solve elided; edit the marked block and write the "
                                    f"solve\n\nwrite_participant_contract(solver='{_code}', variant='fsi', "
                                    f"path=...) writes it.\n\n"
                                    f"```python\n{_own_block}```\n\n"
                                    # the partner's contract, named where this reply still has room
                                    f"## THE FLUID PARTICIPANT it couples with is FEniCSx's "
                                    f"participant_fsi_fluid_fenics.py: write_participant_contract("
                                    f"solver='fenics', variant='fsi_fluid', path=...) writes it, and "
                                    f"knowledge(topic='coupling', solver='fenics', physics='fsi') "
                                    f"serves it with its facts.\n\n" + _fsi_payload)
            except Exception:                                  # noqa: BLE001
                pass
        _fsi_first = not _MUST_READ_STATE["served"]
        _MUST_READ_STATE["served"] = True        # or every FSI call looks like the first reply, and
                                                 # the contract block is never repeated to the worker
        _text = _front_load_coupling(_lead + _fsi_payload, solver, must_read=_fsi_first)
        # THE SESSION'S FIRST REPLY IS THE ORCHESTRATOR'S: when the must-read fills it, it says that the
        # worker's own call "leads with the complete block, the deciding facts and the deck grammar" --
        # that call (pointer mode) carries the code's facts and its fluid-structure facts. Measured:
        # with the facts here too, the fluid-structure facts pushed the per-side naming rule out of
        # this reply for 4C and FEniCSx.
        if _fsi_first and "IS NOT REPEATED IN THIS FIRST REPLY" in _text:
            _facts = ""
        return _with_facts(_text)
    # THE ESCAPE HATCH HAD TO BE MADE REAL.
    #
    # The truncation notice tells the agent, verbatim, that "the rest is
    # available on request -- ask knowledge(topic='coupling', solver='...',
    # signal='<what you are stuck on>')", and the module comment above states
    # that "every section is still reachable by asking for it, which is what
    # the `signal` argument is for". Measured, neither was true: asking with a
    # signal returned a DIFFERENT, shorter payload of failure-table entries —
    # 11,529 characters carrying 5 lines the base did not already have — and
    # the material that had been cut was in neither. For solver='kratos' the
    # casualty is the participant script itself, cut mid-definition at
    # `def build_model()`, on the side a coupled task is most likely to assign
    # to Kratos.
    #
    # An unkept promise is worse than an honest cap: the agent spends a call,
    # believes it has asked for the missing piece, and reads a reply that does
    # not contain it. So a signalled request now really does return what the
    # head left behind, chosen by overlap with the signal.
    #
    # ORDER MATTERS: truncate FIRST, then append. Appending before the cut put
    # the continuation at the very end of a payload the front-loader trims from
    # the front, so the material the agent had just asked for was the first
    # thing dropped -- the request was answered and then silently unanswered.
    _want = ("must-read" in (signal or "").lower()
             or "must read" in (signal or "").lower())
    _mr = _want or not _MUST_READ_STATE["served"]
    _MUST_READ_STATE["served"] = True
    if signal and not _want:
        return _with_facts(_append_coupling_continuation(
            _front_load_coupling(payload, solver, must_read=_mr), solver, signal))
    return _with_facts(_front_load_coupling(payload, solver, must_read=_mr))


def _coupling_participant_script(solver: str, physics: str = "") -> str:
    """The participant CONTRACT for `solver` (solve elided) — the lead-in
    paragraph plus the first fenced contract block — pulled from that solver's
    own coupling payload. '' if the solver ships no served participant.

    PUSHED into prepare_simulation's coupled hand-off so a weak model receives
    the handshake, sign convention, flux recovery and exports schema through a
    call it already makes, instead of the opt-in knowledge(topic='coupling',
    solver=...) call it was measured never to issue (0 of 6 coupled runs made
    it). Hand-rolling that handshake is the dominant coupled failure — crash,
    unresponsive participant, wrong API — so the contract is handed over rather
    than pointed at. The solve stays the agent's own work.
    """
    try:
        payload = _capture_knowledge_fn("get_coupling_knowledge", solver, "")
    except Exception:
        return ""
    if not isinstance(payload, str) or not payload:
        return ""
    # A served participant is the fenced python block that carries the driver
    # contract: it reads imports.json and writes exports.json. Prefer the
    # config-driven one (reads config.json -> parameterises by level, which the
    # must-read requires); otherwise the first contract block.
    cands = []
    for m in re.finditer(r"```python", payload):
        fence = m.start()
        end = payload.find("```", fence + 9)
        if end < 0:
            continue
        block = payload[fence:end + 3]
        if "imports.json" in block and "exports.json" in block:
            cands.append((fence, block))
    if not cands:
        return ""
    if _is_thermoelastic(physics):
        te = [c for c in cands if '"field_name": "thermoelastic"' in c[1]]
        if te:
            cands = te
    _vk = _variant_for(physics)
    _vh = _VARIANT_HEADINGS.get(_vk, "") if _vk and _vk != "thermoelastic" else ""
    if _vh and payload.find(_vh) >= 0:
        # the variant's own block is the first fence after its heading
        _after = [c for c in cands if c[0] > payload.find(_vh)]
        if _after:
            cands = _after[:1]
    fence, block = next((c for c in cands if "config.json" in c[1]), cands[0])
    # THE PUSHED COPY IS THE LEAN ONE: comment blocks thinned to a line, code
    # untouched, so copying it into a file costs half the output tokens. The
    # fully annotated block stays behind knowledge(topic='coupling', solver=...).
    try:
        from .coupling_knowledge import lean_view
        block = lean_view(block)
    except Exception:
        pass
    # lead-in: the paragraph immediately before the fence (states the role and
    # the measured traps); capped so the push stays bounded.
    b2 = payload.rfind("\n\n", 0, fence)
    b1 = payload.rfind("\n\n", 0, b2) if b2 > 0 else -1
    lead = payload[b1 + 2:fence].strip()[-1000:] if b1 >= 0 else ""
    return (lead + "\n\n" + block).strip() if lead else block


# A COUPLED SIDE RUN BY A BINARY STILL NEEDS ITS DECK GRAMMAR.
#
# Measured on a two-material conduction problem whose side A is 4C: of the four deck-death
# diagnostics this project measured against the built binary, a coupling
# request for fourc served ZERO. They are not in the coupling corpus at all —
# they live in the backend's deck grammar, which reaches the agent only if it
# happens to ask topic='physics' with physics='heat' or 'thermo' (4/4 there,
# 2/4 for conjugate_heat_transfer, 0/4 on the pitfalls route).
#
# So an agent told to run one side of a coupling with 4C, asking the coupling
# topic about 4C, gets the handshake and none of the reasons its input file
# will die. The deck grammar is the run interface for a binary; a coupled task
# does not make it less necessary, it makes it necessary in a harder setting.
#
# Appended AFTER the front-loader, for the same reason the continuation is:
# material added before the cut is the first thing the head trims.
_DECK_DRIVEN = {"fourc": ("backends.fourc.deck_grammar", "FOURC_DECK_GRAMMAR"),
                "febio": ("backends.febio.deck_grammar", "FEBIO_DECK_GRAMMAR"),
                "sparta": ("backends.sparta.deck_grammar", "SPARTA_INPUT_GRAMMAR")}


def _deck_grammar_text(solver: str) -> str:
    """The deck grammar a coupling reply carries for a binary-driven side other
    than 4C (whose grammar and skeletons have their own budget rule), '' else."""
    key = (solver or "").strip().lower()
    where = _DECK_DRIVEN.get(key)
    if not where or key == "fourc":
        return ""
    try:
        mod = __import__(where[0], fromlist=[where[1]])
        block = getattr(mod, where[1], "")
    except Exception:                                   # noqa: BLE001
        return ""
    return block if isinstance(block, str) and block.strip() else ""


def _append_deck_grammar(payload: str, solver: str) -> str:
    key = (solver or "").strip().lower()
    where = _DECK_DRIVEN.get(key)
    if not where or not isinstance(payload, str):
        return payload
    try:
        mod = __import__(where[0], fromlist=[where[1]])
        block = getattr(mod, where[1], "")
    except Exception:                                   # noqa: BLE001
        return payload
    if not isinstance(block, str) or not block.strip() or block in payload:
        return payload
    return payload + "\n\n" + block


# How much of the cut material one signalled request may bring back. Sized so
# that must-read + reply + continuation stays near the same order as an
# ordinary coupling reply rather than dumping the whole 65k corpus, which is
# the behaviour the head limit exists to prevent.
_COUPLING_CONTINUATION_LIMIT = 14000


def _split_coupling_sections(text: str) -> list:
    """Break the corpus on the separators the front-loader also cuts on."""
    import re as _re
    parts, buf = [], []
    for line in text.splitlines(keepends=True):
        if _re.match(r"^(─{4,}|={4,}|#{1,3} )", line) and buf:
            parts.append("".join(buf)); buf = [line]
        else:
            buf.append(line)
    if buf:
        parts.append("".join(buf))
    return [p for p in parts if p.strip()]


def _append_coupling_continuation(payload: str, solver: str, signal: str) -> str:
    """Give back the sections the head cut off, ranked against the signal."""
    full = _capture_knowledge_fn("get_coupling_knowledge", solver, "")
    if not isinstance(full, str) or not isinstance(payload, str):
        return payload
    served_head = full[:max(_COUPLING_HEAD_LIMIT - len(_COUPLING_MUST_READ), 0)]
    tail = full[len(served_head):]
    if not tail.strip():
        return payload
    # RANK BY HOW RARE THE MATCHED WORD IS, NOT BY HOW MANY MATCHED.
    #
    # Counting raw hits ranks a section that happens to say "element",
    # "script" and "need" above the one that says "build_model" — measured: the
    # section actually carrying `def build_model()` scored 2 against three
    # generic sections scoring 4, which filled the budget and pushed out the
    # only section the agent had named. A word that occurs all over the corpus
    # says nothing about which section is wanted; a word that occurs in one
    # place says everything, so each match is weighted by its rarity.
    import math as _math
    words = {w for w in _re_words(signal) if len(w) > 3}
    sections = _split_coupling_sections(tail)
    lows = [sec.lower() for sec in sections]
    weight = {}
    for w in words:
        df = sum(1 for low in lows if w in low)
        weight[w] = 0.0 if df == 0 else _math.log(1.0 + len(sections) / df)
    scored = []
    for sec, low in zip(sections, lows):
        score = sum(weight[w] for w in words if w in low)
        scored.append((score, len(sec), sec))
    scored.sort(key=lambda t: (-t[0], t[1]))
    picked, used = [], 0
    for score, size, sec in scored:
        if used + size > _COUPLING_CONTINUATION_LIMIT:
            continue
        if score <= 0 and picked:
            break
        picked.append(sec); used += size
    if not picked:
        return payload
    banner = (f"\n\n{'─' * 70}\n"
              f"WHAT THE TRUNCATED REPLY LEFT OUT, MATCHED TO YOUR SIGNAL "
              f"({used:,} of {len(tail):,} remaining characters)\n"
              f"{'─' * 70}\n")
    return payload + banner + "".join(picked)


def _re_words(text: str) -> list:
    import re as _re
    return _re.findall(r"[a-z_]+", (text or "").lower())


# THE COUPLING PAYLOAD HAD NO CAP, WHILE prepare_simulation HAS ONE AT 16 kB.
#
# Measured: the generic coupling payload is 65,601 chars, and a coupled run
# typically reads it plus two solver payloads (dune 76,831, kratos 72,115,
# fenics 72,993) -- about 214,547 chars of prose before a solver runs. 220 of
# 224 coupled recorded runs read all three.
#
# The consequence is not that the text is unread; it is that the agent runs out
# of ACTIONS. Median tool calls: 39 with these tools, 95 without. Median input
# tokens per call: 91k against 60k. With the tools a run stops at 39% of the
# wall clock with 5.8% timeouts where runs without them hit 19.5% -- and writes
# NO output file in 60% of runs against 31%. Given the identical task and no
# help at all, the unassisted run produces a median 10 CSV files where the
# assisted one produces zero.
#
# So the fix is not more text and not better text. Sections the agent needs
# FIRST -- the hand-in contract, the deliverable half, the NaN rule, the
# pointer to audit_results -- are moved to the front, and the rest is offered
# rather than pushed. Nothing is deleted: every section is still reachable by
# asking for it, which is what the `signal` argument is for.
_COUPLING_HEAD_LIMIT = 28000


# WHAT MUST SURVIVE TRUNCATION, BECAUSE AN AGENT CANNOT ASK FOR WHAT IT DOES
# NOT KNOW EXISTS.
#
# Front-loading takes a PREFIX, so the "Choosing theta" section -- which sits
# past 24,000 characters -- stopped reaching agents entirely the moment this
# function was added. Measured: the served payload is 23,378 characters and
# contains no occurrence of the rho sizing, the divergence thresholds or the
# swap remedy. The truncation hint tells the agent to ask when it is stuck, but
# an agent that does not know its accelerator can DIVERGE reads a rising
# residual as its own modelling error and starts over.
#
# These numbers were measured by execution over 48 configurations of a
# two-subdomain conduction split at the tool's own tol = 1e-6, and they are
# mesh-independent to the iteration over a fourfold interface refinement:
#
#   rho          1/10   1     2     4      10     100
#   iterations   11     37    66    123    299    3311      (theta = 1/(1+rho))
#   theta = 0.5  18     37    90    DIVERGES from rho = 4 up
#   default      12     37    78    381    DIVERGES from rho = 10 up
#
# so the default max_iter = 50 is ALREADY SHORT at rho = 2, and the default
# accelerator diverges on exactly the severe-contrast problems seen in development.
_COUPLING_MUST_READ = """
YOU ARE THE ORCHESTRATOR OF THIS COUPLING, NOT ITS AUTHOR. Participant scripts
are written by role='worker' sub-agents, one ladder step each, never by you in
your own turn. Your NEXT action after reading this reply -- before any plan,
estimate, critic or file of your own -- is spawn_subagent(role='worker',
task=<the brief under "YOUR FIRST SUB-AGENT, NOW" below, with its <...>
block replaced by your task's data>).

START HERE -- THE WHOLE COUPLING IS ONE TOOL CALL. Write one script per
side that reads ./imports.json, runs its own solver once, writes
./exports.json. Then call:

    couple(participants='[
      {"name": "A", "command": ["<python>", "side_a.py"],
       "work_dir": "side_A", "imports_from": ["B"]},
      {"name": "B", "command": ["<python>", "side_b.py"],
       "work_dir": "side_B", "imports_from": ["A"]}]',
      max_iter=<from the rho guidance below; when unsure use 150>,
      tol=<the tolerance your task prescribes>)

Create side_A and side_B INSIDE your working directory (`mkdir -p side_A
side_B`, then `pwd` for the absolute paths); every script, log and
deliverable belongs under that directory, never under your home directory or
a project directory elsewhere -- a write outside it is refused, and refusal
is not a reason to stop.

Pass history_path="<absolute path of the per-level residual-history file your
task names>" in that call as well, so the tool writes the measured iteration
history straight to that file; never retype it.

A task that splits the DOMAIN into subdomains joined by interface continuity
conditions is this partitioned domain decomposition however many fields each
subdomain carries -- temperature and displacement together included. Exchange
every interface quantity per point: values = [T, ux, uy] and normal_fluxes =
[q_n, t_x, t_y] on both sides (one row per interface point, the same order
every iteration); the driver compares them component by component. A FIELD
coupling in which both codes own the whole body (the `tsi` route) is a
different setup and does not apply to a split domain.

WORK THE LADDER, ONE SUB-AGENT PER STEP. audit_results(work_dir) and every
couple() reply name the next unmet step, read from your own files, as a
ready sub-agent brief: two participants -> each exports standalone -> couple
level k -> level k's field and interface files -> its captured run logs ->
the summary. Hand the brief to spawn_subagent(role='worker', task=<brief>)
as is; the step ends when its check passes on disk. Never judge the whole
job at once -- every single step is small. A WORKER THAT REPORTS AN ERROR IS
NOT RE-BRIEFED BY YOU: call audit_results(work_dir) first. It reads that
worker's decks and consoles and names the defect and the step -- the code's
own error line, the sections the installed binary's grammar does not know
with the closest known names, a condition on an undefined id, the
participant's own Python stop -- so the next worker starts from the defect,
not from the whole job again.

YOUR FIRST SUB-AGENT, NOW -- before any plan, estimate or verdict. Copy the
call below and REPLACE the <...> block with your task's own words; the worker
sees nothing but this task= string (measured: a worker whose brief kept
"from the task" and no data wrote placeholder source terms):
    spawn_subagent(role='worker', task="Write side A's participant script in
    ./side_A: call knowledge(topic='coupling', solver='<side A's code>')
    (add physics='thermoelastic' when the interface carries temperature AND
    displacement together, physics='elasticity' when the exchanged field is a
    displacement, physics='transient' for a time-dependent problem, physics='3d'
    for a three-dimensional domain, physics='fsi' for a fluid beside a structure:
    the reply then leads with that contract) and
    write the served CONTRACT for the role the task gives side A to
    ./side_A/participant_A.py with write_participant_contract(solver=<code>,
    path='./side_A/participant_A.py', variant=<physics word above or ''>) --
    the knowledge reply's code with its comments, never re-typed; fill
    only its marked hole(s) IN PLACE -- never re-type the file -- with the mesh, form, material, source and solve
    for subdomain A from THIS DATA, which is all you know of the task:
    <SUBDOMAIN A, COPIED FROM YOUR TASK WORD FOR WORD: geometry and interface
    position; equations and coefficients; source terms as written; boundary
    values; the level-1 mesh; the file names the task prescribes for this
    side>. Write ./side_A/config.json for level 1 WITH THE TASK'S DATA for that
    side as the task writes them (x0, x1, y0, y1, iface, k, lam, mu, beta;
    source_T, source_ux, source_uy or source_expr as strings in x, y): both
    contracts read them and the audit judges the side against them. Then a
    synthetic ./side_A/imports.json whose values and normal_fluxes are
    NOT zero (on a zero import a Neumann side applies no load, and its
    EXPORT SELF-CHECK cannot fire). IF THE CODE TAKES AN INPUT
    DECK, that deck is READ AND JUDGED THE MOMENT YOU WRITE IT -- write it to a
    .yaml, .yml or .dat file and the defects come back in that reply, before
    the binary runs, so do not spend an action asking for the check. Run the
    script with that code's
    own interpreter (generous timeout, first runs compile) until
    ./side_A/exports.json appears with finite values. CHECK: exports.json
    exists and the script exited 0. Report DONE or the exact error.")
THEN A SECOND WORKER FOR SIDE B, WITH THE SAME CARE. Its brief carries side B's OWN geometry, equations, boundary values, level-1 mesh and file names, its own knowledge(topic='coupling', solver='<side B's code>', physics=...) call, and ITS interpreter -- a different binary from side A's in most pairs. The worker sees nothing but its task= string, so "the same as side A" tells it nothing. Same check: ./side_B/exports.json with finite values, exit 0. One side alone is not a coupling: a side with no exports.json never ran to completion, whichever code it runs.

THAT CODE'S OWN INTERPRETER IS PRINTED BY discover(query='list'), next to the
backend, and it is the argv `couple` needs as well. The default `python3` is
NOT it. Measured on this install: the default python carries exactly ONE of the
Python solvers here and none of the other five, so `python3 participant.py`
succeeds for that one code and raises ModuleNotFoundError for the rest -- which
is why the habit survives, and why it is the single most common way a
participant fails to start on this host. The critic reviews what a worker produced, not a
plan. Then audit_results(work_dir) names every further step.

The tool runs the whole iteration -- relaxation, convergence, validation --
and on success returns your interface tables READY TO SAVE plus the paths of
the captured solver logs. DO NOT hand-roll this loop yourself: measured
across the runs that did, the hand-rolled exchange stalls (residuals
9.92->9.98 over 100 iterations; constant 1.0) and cannot show two codes coupled.
One couple call per mesh level, on the exact levels your task prescribes --
or ONE couple_levels(participants=..., levels='[{"level": 1, "A": {"nx": ..,
"ny": ..}, "B": {...}}, ...]', history_pattern='<the per-level history file
name your task prescribes, with {k} for the level>') call for the whole
sequence: it hands each side its level's mesh keys in the environment
(OPENPASO_CONFIG_JSON; a script that never reads it is refused before anything
runs -- openPASO writes no file of yours), warm-starts each level from the previous one, and keeps every level's
history, console and interface tables (measured: couplings proven
at level 1 ran out of wall clock before level 3 when every level cost ten calls).

DO NOT WRITE THE PARTICIPANT'S HANDSHAKE FROM SCRATCH -- THE CONTRACT EXISTS
FOR YOUR CODE. For EACH of your two codes call `knowledge(topic='coupling',
solver='<that code>')` with the physics word the brief above names; the
reply then leads with that variant where the code ships one (the thermo-elastic
one takes [T, ux, uy] in and returns [qn, qx, qy]; the transient one marches
the WHOLE time window per call), and without the word with the steady scalar
contract. THE ROLES COME FROM THE TASK: when it says which
subdomain is the Dirichlet side and which the Neumann side, that is fixed.
Each served contract states which side it is (some carry both behind a SIDE
switch); take the one for the role your task gives that code, and where the
code ships none for that role, keep the other side's served handshake,
recovery and exports schema as the pattern.
It returns that code's participant CONTRACT: how to read
./config.json for the level and ./imports.json, the interface sign convention,
the consistent outward-flux recovery you apply to your OWN assembled system, the
exact ./exports.json schema, and the one-field-file-per-level rule. The mesh,
the weak form, the material, the source and the solve itself are deliberately
NOT in it: you write those, using `prepare_simulation(solver='<that code>',
physics='<your physics>')` for that code's API, gotchas and a generic worked
pattern. The same reply lists that code's measured traps. Write the
solve, keep the served handshake and recovery as given, and let the checks
below tell you what to fix.

THE NEUMANN SIDE'S IMPORTED FLUX IS SILENTLY IGNORED WITHOUT A CONDITION.

This is the single defect that has sunk the most nearly-correct coupled
runs, and it leaves no trace: the solver runs, converges, exits 0, and
returns exactly the answer it would have returned with no flux at all.

In Kratos, setting FACE_HEAT_FLUX on the interface NODES does nothing unless
`ThermalFace2D2N` conditions exist on the interface EDGES -- the nodal value is
only ever integrated BY a condition. Measured on one mesh, three runs differing
only in this:

    zero flux, conditions present     max|T| = T0
    flux on nodes, NO conditions      max|T| = T0, BIT-IDENTICAL
    flux on nodes AND conditions      a different field: the flux is applied

Two real runs died exactly here: side A correct to three digits, side B
reporting the no-flux answer, with the interface FIELD matching across the
seam exactly, so only the flux jump betrayed it, growing under refinement
instead of shrinking.

THE SHAPE IS GENERAL, not Kratos-specific: a boundary value attached to nodes
but never integrated over a facet contributes nothing. 4C has the same trap
twice -- the `DESIGN ... THERMO ...` condition sections are never evaluated in a
standalone Thermo problem, and a body source must sit on the condition whose
geometry type matches the ELEMENT DIMENSION (LINE 1D / SURF 2D / VOL 3D).

HOW TO CATCH IT IN ONE STEP, before any coupling iteration: solve the Neumann
side ONCE with the imported flux set to zero, then ONCE with your real flux,
and compare. If the two fields are identical, the flux never reached the
operator. That costs one extra solve and is the only check that sees this.

PARAMETERIZE BY LEVEL, OR THE BUDGET EATS YOU. Have each participant read
its mesh size from a tiny ./config.json ({"level": 1, "nx": <nx>, "ny": <ny>})
instead of hard-coding it; advancing a level is then: DOUBLE nx and ny
(halve h) and set the next level in each side's config, call couple again,
save that level's outputs. The level key is a label; only nx and ny change
the mesh (measured: a run that raised the label alone coupled the same
mesh three times). Measured
both ways: participants built this way ran all three levels in under two
minutes of compute; sessions that regenerated their meshes by hand inside
the participant spent their whole budget on level 1. Build the config
route FIRST -- it costs one extra minute at level 1 and buys the other
two levels.

THE FIELDS ARE THE RESULT; THE HISTORY IS THE EVIDENCE. Interface rows at points the task does not print are refused wholesale (measured: a coupling converged at every level was refused because every one of its interface rows sat at self-chosen coordinates). What counts is the field and interface files at the prescribed probe points, for every level and both sides -- a run that converges its coupling and writes no field files counts for NOTHING (measured: a run that converged its first level and delivered only that level's residual history). Alongside them, a task that names a per-level residual-history file wants one row per partitioned-iteration step, per mesh level. That file IS the evidence that two
codes iterated against each other; nothing else you deliver can show it.

THE INTERFACE FILE IS WRITTEN AT THE POINTS THE TASK LISTS, NOT AT YOUR NODES.

Its rows are the coordinates the task names, in that order, IDENTICAL at every
mesh level. Runs with both solvers and the iteration independently proven to
have run, and the residual down to 1e-7, were still counted as unusable for
writing their own interface MESH NODES instead:

    what was written   level 1: 7 rows   level 2: 15 rows   level 3: 23 rows
    what was asked     the same fixed list of points at EVERY level

An order compares the SAME quantity across levels, and node sets move under
refinement, so there is nothing to compare. Interpolate onto each listed
coordinate with the shape-function evaluation you already use for the solution
probes — not the nearest node — all of them, in order, nothing else.

THE ENDS OF THE INTERFACE ARE EXCLUDED ON PURPOSE: a Dirichlet-Neumann split
has a corner there whose recovered flux does not converge. Do not "complete"
the list with them.

CHECK EACH SUBDOMAIN AGAINST ITS OWN EQUATION FIRST. A converged interface
residual says the two sides AGREE, not that either is right, and the two
failures are independent: one run converged to 8e-07 in 16 steps with a field
TEN TIMES too small, whose error did not fall at all under refinement. Run
verify_pde_consistency(...) on each side, with that side's own source and
coefficient, before spending budget on the iteration.

IT ANSWERS BOTH SIDES NOW, and refuses for one reason only: an operator it
does not model. It implements second-order scalar diffusion, so an elasticity
form or an added reaction term is refused BY NAME on the equation string. Read
a refusal as having learned NOTHING about that side, not as a pass.

READ THE TWO VERDICTS ASYMMETRICALLY. INCONSISTENT is strong evidence against
a side, so fix it before spending budget on further levels. CONSISTENT is
weaker, because an identity whose test function vanishes on the boundary cannot see a
wrong condition ON that boundary. A clean verdict is not a clean bill of
health.

THE DIRICHLET SIDE RETURNS A MEASURED FLUX, NEVER A PLACEHOLDER. Recover it from your OWN system, in this order: (1) the CONSISTENT residual recovery q = -(A u - b_vol)/w on the interface rows -- second order, valid on BOTH sides; (2) the code's native boundary-flux output ONLY on a side whose interface is Dirichlet (on a Neumann-loaded line it echoes the applied load); (3) one-sided quadratic extrapolation of -k*du/dn from three field points along the normal as a CROSS-CHECK, not the exported value on a high-diffusivity side (measured: routes 1 and 3 agree to 2.7% rel-RMS at h=1/8 on the low-k side; route 3 alone missed by 6.5% on the high-k side). A constant or invented exchanged quantity turns the partitioned update into a no-op (measured: a hard-coded 0.0 flux never approached tolerance in 50 iterations; the real recovered flux converged in 4). If the residual is not contracting, check FIRST that the data you SEND changes between iterations.

THE RECOVERY, READY TO COPY (P1 triangles; split quads into two triangles first; works for any interface axis) -- verified by execution, 9.3e-4 max relative error against an analytic outward flux at h=1/16, second order under refinement. A first-order recovery here caps the whole coupled field at order ~1 however good the elements are. FOR ELASTICITY (vector interface) the same identity holds PER COMPONENT: t_c(x_i) = (K u - F_vol)_(i,c) / w_i with K the elastic stiffness assembled with NO boundary conditions, F_vol the volume load only, w_i the tributary interface length; export q = -t per the sign convention (measured on a manufactured plane-strain case: component orders 1.91-2.10 across the ladder):

def consistent_interface_flux(nodes, tris, k, u, f_vol, iface_ids, h_trib):
    # Second-order outward interface flux from YOUR OWN P1 system.
    # nodes (N,2); tris (M,3) int; k conductivity (scalar or per-node);
    # u solution (N,); f_vol volumetric source at nodes (N,);
    # iface_ids interior interface node indices; h_trib tributary length.
    # Returns q of len(iface_ids): q_i = -(K u - b_vol)_i / h_trib.
    import numpy as np
    resid = np.zeros(len(nodes))
    kn = np.full(len(nodes), float(k)) if np.isscalar(k) else np.asarray(k, float)
    for el in np.asarray(tris, int):
        P = nodes[el]
        area = 0.5 * abs((P[1,0]-P[0,0])*(P[2,1]-P[0,1]) - (P[1,1]-P[0,1])*(P[2,0]-P[0,0]))
        g = np.array([[P[1,1]-P[2,1], P[2,0]-P[1,0]],
                      [P[2,1]-P[0,1], P[0,0]-P[2,0]],
                      [P[0,1]-P[1,1], P[1,0]-P[0,0]]]) / (2.0*area)
        ke = float(kn[el].mean()) * area * (g @ g.T)
        resid[el] += ke @ u[el] - area/3.0 * float(np.mean(f_vol[el]))
    return np.array([-resid[i]/h_trib for i in iface_ids])


    couple(participants='[{"name": "A", "command": "<run side A>",
                           "work_dir": "side_A", "imports_from": ["B"]},
                          {"name": "B", "command": "<run side B>",
                           "work_dir": "side_B", "imports_from": ["A"]}]',
           max_iter=100, tol=1e-6,
           history_path="<the per-level residual-history file your task names — a path inside your working directory>")

returns `history` and writes its finite measured values directly to the requested
CSV. Report `iterations` (also copied
to `history_file.driver_iterations`) as your coupling-iteration count; `rows_written` is
normally one smaller because iteration 1 has no previous iterate and therefore
no residual. Get ONE participant writing exports.json standalone first, then the
second, then call couple: that order costs the fewest attempts.

A RESIDUAL THAT COLLAPSES IN A FEW STEPS IS USUALLY A DEAD PARTICIPANT.

If the driver reports a participant as UNRESPONSIVE, it means its exports were
BYTE-IDENTICAL across iterations in which its imports changed -- so its output
is not a function of its input, and the coupling is not coupled. The residual
then falls to ~1e-18 in three or four steps, because nothing is moving. That
looks like spectacular convergence and is worth nothing.

    a real coupling, even a very stiff one (a large k ratio), took 12 iterations to
    1.1e-22 with both sides reported responsive
    a dead one reached 1.5e-18 in 4 steps and was reported unresponsive

Three causes, in the order they occur: the script reads imports.json ONCE at
import time instead of on every invocation; it caches or re-serves the previous
exports.json rather than rewriting it; or it never finds the partner key at all
-- `imports.json` is `{{partner_name: ...}}`, and a wrong name silently yields
no data and the fallback is used forever. Print the first imported value each
iteration: if it never changes, that is the bug, not the physics. Do not raise
max_iter or change the accelerator for this -- neither touches it.

THE RESIDUAL YOU REPORT MUST MEASURE THE TWO SIDES, NOT AN ITERATE.

The interface residual you report is the disagreement between your two subdomains at the
shared interface probes, computed from the two profiles you exported:

    field:  max|u_A - u_B|  / max(|u_A|, |u_B|)
    flux:   max|q_A + q_B|  / max(|q_A|, |q_B|)   (outward normals, so they
                                                   must CANCEL, not match)
    the reported interface residual = the larger of the two.

An update norm, one side's own solver residual, or the driver's iterate
difference all fall to 1e-7 while the two codes still disagree completely.
Reporting one of those loses a coupled answer: measured, a 189% flux mismatch
between the two exported files sat behind a reported residual of 1.12e-07.
Recompute the number from the files you just wrote. If it is
not small, the coupling has not converged, whatever the iteration history says.

IF YOU DRIVE THE LOOP YOURSELF, IT MUST ACTUALLY ITERATE. A closed-form
sequence written into the residual-history file — 1.0, 0.5, 0.25, 0.125, … or
any r*q^k — is DETECTED and read as invented, not as a result. Two checks, both
stated here because an independent check applies them: the per-step ratio of a real
Dirichlet-Neumann iteration varies as the error's modal composition changes, so
a constant ratio is a formula; and the history depends on the discretisation, so
the SAME numbers at two mesh levels cannot both be measurements. A run that
honestly reports a diverging or stalling iteration is worth more than one that
reports a clean invented one.

BEFORE ANYTHING ELSE — HOW MANY ITERATIONS TO BUDGET, AND WHEN THIS DIVERGES.

rho = the interface conductance (or stiffness) of the DIRICHLET-side subdomain
divided by that of the Neumann-side one. Everything below is set by rho, NOT by
your mesh: measured counts did not move by a single iteration over a fourfold
interface refinement.

    rho          1/10   1     2     4      10     100
    iterations   11     37    66    123    299    3311

  * Set max_iter = 100 for rho <= 1 and 400 for rho <= 10, and size each
    participant's `timeout` for that many solves. The DEFAULT max_iter = 50 is
    already short at rho = 2.
  * theta = 0.5 with accelerator="constant" DIVERGES from rho = 4 upward.
    The DEFAULT accelerator DIVERGES from rho = 10 upward. A rising residual
    here is the scheme, not your model -- do not rebuild the setup.
  * With accelerator="constant", theta = 1/(1+rho) is the setting the counts
    above were measured with.
  * COMPUTE rho, DO NOT READ IT OFF THE MATERIAL CONTRAST. The interface
    conductance of a side is its coefficient divided by its own width, so a
    material contrast of N:1 on subdomains of different widths is NOT rho = N:
        rho = (k_dirichlet / width_dirichlet) / (k_neumann / width_neumann)
    Worked with arbitrary numbers: k = 3 across a width of 0.6 on the
    Dirichlet side and k = 45 across a width of 0.5 on the Neumann side give
    5 and 90, so rho = 0.056 where the material contrast reads 15:1, and
    a case that converges in a few iterations needing nothing special. An agent that reads the contrast as rho concludes the
    opposite and starts rebuilding a setup that was already fine. Put YOUR
    numbers in the formula.
  * IF rho > 10 AND THE TASK LEAVES THE ROLES TO YOU, the budget is the wrong
    knob: SWAP WHICH SIDE IS DIRICHLET. That replaces rho by 1/rho, and every
    ratio below 1 converged inside 25 iterations.
  * IF THE TASK PRESCRIBES WHICH SIDE IS DIRICHLET, DO NOT SWAP THEM. Wording
    like "these roles are prescribed: solve with them as stated rather than
    choosing your own" makes the assignment part of the problem, and a run that
    swaps has solved a different problem however well it converged. Raise
    max_iter and set theta = 1/(1+rho) instead. Read the task again before
    reaching for this: the remedy above is for the case where you are free to
    choose.


WHAT YOUR SOLVE MUST LEAVE BEHIND.

openPASO serves the participant with the mesh/form/solve region CUT OUT, and the
surviving code still uses the names that region defined. It is your solve, but
it has a contract: whatever you write must leave behind the interface degrees
of freedom, the assembled operator, and the solution vector the export block
reads -- the served text names them where the hole is. A participant that runs
but exports nothing has met the letter of the handshake and none of its point.

That is checked from your own files: `couple` reports a participant as
UNRESPONSIVE when its exports are byte-identical across iterations in which
its imports changed, and a residual that falls to ~1e-18 in three or four
steps is that, not convergence.

THE COUPLING HISTORY IS MEASURED, NOT MODELLED.

The per-level residual-history file your task names holds the mismatch your
iteration ACTUALLY measured at each step, written from inside the loop -- you
already compute that number every iteration to decide when to stop, so
appending it to the file is one line and costs nothing. Do not write a
plausible-looking decay instead: a modelled history has a step-to-step ratio
that is constant to machine precision, and that is checked -- three real runs
wrote 0.1*0.7^k, 8.5e-5*0.85^k scaled by 1/level, and one geometric sequence
repeated bit-identically at all three levels, and every one reads as invented,
which is worth LESS than an honest report that the iteration did not converge.
A real iteration's rate wanders; if your loop never computed a mismatch, it
never coupled, and the honest entry is a could-not-finish report plus your
best single-domain fields.

AND WRITE WHAT THE SOLVER SAID, NOT WHAT YOU KNOW IT DID.

Where a task asks for an execution log, it asks for the code's OWN console
output, captured verbatim, because that is the only thing that establishes
which code ran on which side. If you invoke the solver through subprocess you
already hold those bytes; the whole fix is not to drop them:

    r = subprocess.run(cmd, capture_output=True, text=True)
    Path(log).write_text(r.stdout + r.stderr)

or skip the capture and redirect, `cmd > <the per-level run log for that
code> 2>&1`.

DO NOT PREPEND A DOF LINE OF YOUR OWN. The served participant already prints
`NDOF = <integer>` on a line of its own, from the space it actually built, and
the contract line is read as the FIRST such line in the file. A hand-written
line above the capture therefore WINS over the real one, and a hand-written
number is where the wrong one comes from: measured on a coupled run whose mesh
really did refine, side A's log opened with `NDOF = 1` at every level and the
result set read as an unrefined mesh. If your own participant does not
print the line, add the print INSIDE it rather than in front of the log.
Measured: one run drove the interface to 4.4e-07 and reached order 1.94, then
wrote three lines of its own prose into the log and could not be credited with
any of it.
Some codes need one line to
print anything: FEniCSx `dolfinx.log.set_log_level(LogLevel.INFO)`,
deal.II `deallog.depth_console(2)` AND a SolverControl with log_history/
log_result, NGSolve `ngsglobals.msg_level = 3`, DUNE-fem
`parameters={"linear.verbose": True}`, scikit-fem `logging.basicConfig(
level=logging.INFO)`, which prints to STDERR. Kratos, 4C, FEBio and SPARTA
print by default.

AND THE DRIVER ALREADY KEPT EACH LEVEL'S CONSOLE FOR YOU. `couple` saves every
participant's captured output beside that side's exports.json as
participant_output_level<k>.log, one file per level, the moment the level
finishes. The per-level execution log your task asks for is a COPY OF THAT
FILE for that level and that side -- you do not have to re-run anything to
produce it, and you should write it WHEN THE LEVEL FINISHES rather than at
hand-in.

Leaving it to the end is what goes wrong. Measured: a coupled run that did
everything else right copied ONE console into all three level logs at the very
end, and three identical logs read as one mesh solved three times.

CHECK THE INTERFACE SIGN BEFORE YOU HAND IN.

audit_results(work_dir=<your results directory>) runs this check over the files
you have already written, and it is the route to prefer: it needs no arguments
beyond the directory, and it reports the sign finding alongside everything else
it names. A session that also exposes the dedicated tool can call it directly
for the same check on chosen files:

    verify_interface_flux(interface_files="<all per-level interface files, both sides>",
                          solution_files="<all per-level field files, both sides>")

Either way it needs no reference solution, and it reads each normal from your files
(on a bent interface, each leg's own). For a flux you really computed from your
own solution, q_n = k (-du/dn) at every interface point: a CONSTANT multiple,
POSITIVE. A NEGATIVE multiple means your normal points inward.

THE TRAP IT CATCHES: on the NEUMANN side the flux you IMPORT and the flux you
REPORT have OPPOSITE signs. Kratos's FACE_HEAT_FLUX is the INWARD normal flux,
while the task defines q_n = -(K grad u) . n_out with n_out pointing OUT of the
subdomain -- so the number you write into each per-level interface file is the
NEGATIVE of the one you applied.

Measured on the run described above whose field matched to 0.000e+00 across
the seam: one side's implied coefficient came out -1.25 times its stated k. On
a correct result set the same tool returns 0.98-1.30 times k on the low-k side
and 1.00-1.03 times k on the high-k side, without any reference solution. One
call would have told the run which of the two it was.

EACH SIDE'S PER-LEVEL RUN LOG MUST CARRY THAT SOLVER'S OWN OUTPUT.

A DOF-count line alone is code-agnostic: it cannot show WHICH code produced the
side, so a coupled claim built on it is unproven no matter how good the numbers
are. Capture the solver's console output into the log next to your DOF-count
line -- 4C's banner and git SHA, Kratos's strategy telemetry, whatever your code
prints. This was measured: a run whose numbers were genuinely second order was
credited to two prescribed codes it had invoked NEITHER of, on the strength of
fourteen ten-byte files carrying nothing but a DOF count.

Two ways that capture silently fails, both measured:
  * redirecting Python's stdout around an in-process solve captures ZERO bytes
    from a code that prints through C++ streams -- os.dup2 on fd 1 around a
    Kratos solve produced a 0-byte file. RUN THE SOLVE IN A SUBPROCESS and
    capture that subprocess's stdout; then the capture cannot miss.
  * a template that assembles and solves the system itself never prints
    anything from the named code, because the named code never ran. If a
    template you were given has no import of, or call to, the code the task
    names, it is the wrong artefact for a task that names it -- ask for the
    real route. An independent assembly is a fine CHECK on the solver's answer
    and a fatal substitute for it.

"""

# THE CONTRACT ARRIVES IN THE FIRST REPLY OF A SESSION. Measured 2026-09-11:
# the must-read had grown to 27k characters against a 28k head budget, so the
# first coupling call of every session -- the one the parent reads -- carried
# no participant contract at all (zero python fences in a 48k reply, every
# backend); only a later call, in pointer mode, did. The must-read is
# therefore split at the sentence that introduces the contract: part A (the
# orchestrator rule, the couple() recipe, the ladder, the first worker brief)
# leads, the code's own payload head follows with its contract inside the
# first 16k, and part B (the rules that decide convergence) closes the reply.
# THE PER-SIDE NAMING RULE JOINS THE MUST-READ BEFORE THE SPLIT. It was
# appended after part B had been cut, so the coupling door -- every first
# reply and signal='must-read' -- never served it (measured 2026-09-23 on
# kratos, fourc, dune: -1), and only the prepare hand-off did.
_COUPLING_MUST_READ += "\n" + _PER_SIDE_NAMING
_COUPLING_LEAD_SPLIT = "DO NOT WRITE THE PARTICIPANT'S HANDSHAKE FROM SCRATCH"
_COUPLING_LEAD_A = _COUPLING_MUST_READ[:_COUPLING_MUST_READ.index(_COUPLING_LEAD_SPLIT)]
_COUPLING_LEAD_B = _COUPLING_MUST_READ[_COUPLING_MUST_READ.index(_COUPLING_LEAD_SPLIT):]
_COUPLING_CONTRACT_HEAD = 11000     # floor for the payload head between the two parts when no contract block is found



def _contract_block_span(payload: str) -> tuple[int, int]:
    """(start, end) of the payload's first participant contract block, or
    (-1, -1). `_contract_block_end` gives the end alone; the cap needs the
    start as well, to lift the block out of the instruction budget."""
    if not isinstance(payload, str):
        return -1, -1
    pos = 0
    while True:
        i = payload.find("```python", pos)
        if i < 0:
            return -1, -1
        j = payload.find("```", i + 9)
        if j < 0:
            return -1, -1
        if "imports.json" in payload[i:j + 3] and "exports.json" in payload[i:j + 3]:
            return i, _contract_block_end(payload, j + 3)
        pos = j + 3


def _contract_block_end(payload: str, floor: int, with_program: bool = True) -> int:
    """Where the payload's first participant contract block closes (the fence
    after the first ```python block that carries the handshake), so the
    must-read reply can be cut behind the contract and never through it.
    `floor` when the payload has no such block. `with_program`: the program a
    wrapper runs, shown right behind it, belongs to the block (the worker's
    reply and the cap); the orchestrator's first reply leaves it out."""
    if not isinstance(payload, str):
        return floor
    pos = 0
    while True:
        i = payload.find("```python", pos)
        if i < 0:
            return floor
        j = payload.find("```", i + 9)
        if j < 0:
            return floor
        block = payload[i:j + 3]
        if "imports.json" in block and "exports.json" in block:
            end = j + 3
            # a second role's contract directly behind the first (the Neumann
            # side of a backend that ships both) belongs in the same head
            nxt = payload.find("```python", end)
            if 0 <= nxt <= end + 1500 and "NEUMANN-SIDE PARTICIPANT" in payload[end:nxt]:
                nj = payload.find("```", nxt + 9)
                if nj > 0 and "imports.json" in payload[nxt:nj]:
                    end = nj + 3
            # the program a wrapper runs, shown right behind it (deal.II: its C++
            # scaffold and CMakeLists.txt), is the same contract: a head cut
            # between the two, or inside the program, hands over half a file
            cpp = payload.find("```cpp", end)
            if with_program and 0 <= cpp <= end + 1500 and "THE PROGRAM THE WRAPPER RUNS" in payload[end:cpp]:
                for fence in ("```cpp", "```cmake"):
                    at = payload.find(fence, end)
                    close = payload.find("```", at + len(fence)) if 0 <= at <= end + 1500 else -1
                    if close > 0:
                        end = close + 3
            return max(floor, min(end + 200, len(payload)))
        pos = j + 3


def _front_load_coupling(payload: str, solver: str = "",
                         must_read: bool = True) -> str:
    # THE MUST-READ IS ATTACHED ALWAYS, NOT ONLY WHEN THE PAYLOAD IS TOO LONG.
    #
    # This returned `payload` untouched whenever it fitted inside the limit, so
    # the must-read — the couple() call, the forgery rules, the rho budget, and
    # where interface values go — rode along only as a SIDE EFFECT of
    # truncation. Measured: knowledge(topic='coupling') is long enough to be
    # cut and did carry it, but knowledge(topic='coupling', solver='fourc') and
    # any call narrowed by signal='...' are shorter, and carried none of it.
    #
    # Those are not exotic call shapes. The truncation notice below tells the
    # agent, in as many words, to come back with
    # knowledge(topic='coupling', solver='...', signal='<what you are stuck
    # on>') — so openPASO was directing agents at the one door that dropped the
    # text they were being sent to find.
    if not isinstance(payload, str):
        return _append_deck_grammar(payload, solver)
    _lead = _COUPLING_LEAD_A if must_read else _MUST_READ_POINTER
    _tail = _COUPLING_LEAD_B if must_read else ""
    if must_read:
        # In the first reply the must-read itself is present, so the payload's
        # own recap of it and the parts-door preamble are dead weight; drop
        # them, or 4C's 22k scaffold plus part B overruns the reply cap and
        # part B loses its tail (measured 2026-09-11).
        try:
            from .coupling_knowledge import _RECAP as _recap_text
            payload = payload.replace(_recap_text, "")
        except Exception:                                  # noqa: BLE001
            pass
        _cp = payload.find("# Coupling participant:")
        if _cp > 0:
            payload = payload[_cp:]
        # the head must hold the WHOLE first contract block (4C's lean
        # contract is ~22k, Kratos's ~7k), so its budget is the end of that
        # block plus the section boundary after it, never a fixed number
        # the orchestrator writes no participant: its first reply keeps the
        # wrapper contract and leaves the program behind it to the worker's call
        budget = _contract_block_end(payload, _COUPLING_CONTRACT_HEAD, with_program=False)
        if len(payload) <= budget:
            return _append_deck_grammar(_lead + payload + "\n" + _tail, solver)
        # THE PARENT'S FIRST REPLY MUST KEEP PART B WHOLE. Measured 2026-09-11 on the 4C
        # thermo-elastic door: lead A (5k) + the 31k contract block + part B (23k) ran to 83k,
        # the 48k reply cap cut part B at 9.6k of 23k (the rho budget, the interface-file
        # rule, the measured-history rule never reached the parent) and every fact with it.
        # Under the orchestrator rule the parent never copies the contract -- its worker's
        # own door call leads with it -- so when the block does not fit next to part B, the
        # first reply keeps the contract's prose and says where the block comes from.
        _fence = payload.find("```python")
        if 0 <= _fence < budget and len(_lead) + budget + len(_tail) > _KNOWLEDGE_REPLY_LIMIT:
            _phys = ""
            for _k, _h in _VARIANT_HEADINGS.items():
                if _h.strip() in payload[:_fence]:
                    _phys = f" physics='{_VARIANT_PHYSICS_WORD[_k]}',"
                    break
            note = (f"\n[THE SERVED CONTRACT BLOCK ({budget - _fence:,} characters) IS NOT REPEATED IN THIS "
                    f"FIRST REPLY so the must-read below arrives whole. Your WORKER's own call "
                    f"knowledge(topic='coupling', solver='{solver}',{_phys} ...) leads with the complete "
                    f"block, the deciding facts and the deck grammar; hand the worker the brief, not this text.]\n")
            return _append_deck_grammar(_lead + payload[:_fence] + note + "\n" + _tail, solver)
        limit = len(_lead) + budget
    else:
        if len(_lead) + len(payload) <= _COUPLING_HEAD_LIMIT:
            return _append_deck_grammar(_lead + payload, solver)
        # NEVER CUT INSIDE THE FIRST CONTRACT BLOCK. Measured 2026-09-11: the
        # thermo-elastic 4C block is 27k, the flat head 28k minus the lead cut
        # its exports tail off in the worker's own call, and the worker wrote
        # its own export code in place of the missing lines.
        budget = max(_COUPLING_HEAD_LIMIT - len(_lead), _contract_block_end(payload, 0))
        limit = len(_lead) + budget
    head = _lead + payload[:budget]
    # Cut on a section boundary so no instruction is truncated mid-sentence --
    # but take the LONGEST safe cut, not the first marker type that qualifies.
    #
    # This tried the markers in order and broke on the first whose LAST
    # occurrence sat past half the limit. Section separators are sparse, so
    # `\n────` typically last occurs well before the budget ends: measured, the
    # head collapsed from 24,000 to 14,585 characters and the served payload
    # from ~24,600 to 17,329. A third of the agent's coupling budget was being
    # discarded to avoid a mid-sentence cut that a later `\n\n` would have
    # avoided just as well.
    cuts = [head.rfind(m) for m in ("\n────", "\n\n#", "\n\n", "\n")]
    cut = max([c for c in cuts if c > limit // 2], default=-1)
    if cut > 0:
        head = head[:cut]
    rest = len(payload) - (len(head) - len(_lead))
    if must_read:
        hint = (f"\n\n{'─' * 70}\n"
                f"THIS PAYLOAD IS TRUNCATED HERE (this code's coupling text): {rest:,} further "
                f"characters (its traps, the other role, launch notes) exist and "
                f"are NOT lost -- knowledge(topic='coupling', solver='{solver}') "
                f"again, or with signal='<what you are stuck on>', returns them. "
                f"The must-read continues below.\n{'─' * 70}\n")
        return _append_deck_grammar(head + hint + "\n" + _tail, solver)
    hint = (f"\n\n{'─' * 70}\n"
            f"THIS PAYLOAD IS TRUNCATED HERE. {rest:,} further characters "
            f"exist and are NOT lost.\n"
            f"{'─' * 70}\n"
            f"You are reading the first {len(head):,} of {len(payload):,} "
            f"characters. The rest is available on request -- ask "
            f"knowledge(topic='coupling'"
            + (f", solver='{solver}'" if solver else "")
            + ", signal='<what you are stuck on>') and name the problem: a "
            f"diverging iteration, a sign convention, a flux that will not "
            f"balance, a participant that will not start.\n"
            f"WHY IT IS CUT. Reading all of it costs you the actions you need "
            f"to solve the problem: the text is not the binding constraint, "
            f"your budget is.\n"
            f"WHAT TO DO NEXT, in order: get ONE participant running "
            f"standalone until it writes exports.json; get the SECOND one "
            f"running; then call couple(); then write the deliverables. Ask "
            f"for more text only when a specific step has failed.\n")
    return _append_deck_grammar(head + hint + ("\n" + _tail if _tail else ""), solver)


def _get_tsi_knowledge():
    return _capture_knowledge_fn("get_tsi_knowledge")


def _get_precice_knowledge(solver: str = ""):
    return _capture_knowledge_fn("get_precice_knowledge", solver)


