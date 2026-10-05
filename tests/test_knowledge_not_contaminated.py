"""Guard: openPASO's knowledge must describe the CODE, never the ANSWER.

An audit found evaluation-specific content reachable through the very tool the
server instructions tell every agent to call first. Confirmed by execution:

  * campaign identifiers inside agent-readable pitfalls — "the E5 prototype
    run (2026-08-01)", "(T14 campaign fix.)";
  * MEASURED convergence orders shipped as knowledge — "L2 EOCs
    1.984/1.996/1.999", "observed orders 1.93 / 1.99 / 1.99", "Richardson
    orders 2.018/2.004";
  * pre-solved coupled cases — an exact solution and a tuned relaxation answer
    for the same geometry the evaluation uses.

Knowledge like that makes the evaluation measure itself: the agent can read the
answer to a convergence study out of the tool being assessed.

THE RULE THIS TEST ENFORCES
    Knowledge may state how the code behaves, how it fails, what a keyword
    means in the installed version, which element locks, which default changed.
    It may NOT state the answer to a problem: a measured convergence order we
    produced, an exact solution, a tuned parameter for a specific setup, or any
    reference to our own evaluation campaign.

A PUBLISHED REFERENCE VALUE IS AN ANSWER, CITED OR NOT.
    This reverses the carve-out this file used to carry -- "published
    literature benchmarks WITH a citation are expertise, not answers, and are
    deliberately allowed". That sentence was wrong and it was load-bearing:
    every stored DFG, Ghia and Schaefer-Turek number in the repository passed
    this suite because of it, so the audit that found them was arguing against
    a green build.

    A citation records where a number came from. It does not stop an agent
    reading the number instead of computing it, and it destroys our ability to
    tell afterwards which happened. openPASO stores "Strouhal 0.295-0.305" for
    the confined DFG cylinder; when a model reports a value in that interval we
    can no longer distinguish a retrieval from a recollection from a read of our
    own file. A product whose purpose is to stop a model asserting numbers it
    did not compute cannot keep those numbers on disk.

    openPASO's job is to name the benchmark, say how to drive the backend at it,
    and require the agent to fetch the reference at run time and cite what it
    fetched.

WHAT SURVIVES, AND IT IS MOST OF WHAT THESE ENTRIES SAY
    The lesson wrapped around the number stays; the number goes. "Different
    derived quantities converge at different rates, so one landing inside its
    published band is not evidence the run is resolved" is knowledge. The table
    of quantities and bands underneath it is the answer. Likewise "a reference
    value belongs to a geometry -- check which one before comparing" stays, and
    the two Strouhal numbers it compares go.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Where knowledge an agent can reach actually lives.
KNOWLEDGE_ROOTS = [REPO / "src" / "backends", REPO / "src" / "tools",
                   REPO / "data"]

# Identifiers of our own evaluation. None of these may appear in knowledge.
CAMPAIGN_TOKENS = [
    # 2026-09-23: the shapes that reached SERVED participant templates and shipped source comments
    # ("C2 excludes them", "C2 seed 9514 (graded CORRECT)"): a campaign name with a rule or a round,
    # a seed number, a grader verdict. None of these is knowledge about a solver.
    r"\bC[0-9]\s+(?:excludes|includes|round|iteration|iter|seed|cell|grader|grades)\b",
    r"\bseed\s+\d{4}\b", r"\bgraded\s+(?:CORRECT|WRONG|INCORRECT)\b", r"\bC[0-9]\s+iter\s*\d\b",
    # grader outcome classes are grading vocabulary, with or without a count
    r"\b(?:CONFIDENTLY_WRONG|HONEST_INCOMPLETE|COMPLETED_UNPHYSICAL|INVALID_INFRA)\b",
    r"\b(?:MALFORMED|UNPHYSICAL|FABRICATED|CORRECT|WRONG)\s+\d+/\d+\b",
    r"\b\d+\s+of\s+\d+\s+(?:CORRECT|WRONG|correct|unphysical|malformed|honest-incomplete|confidently wrong)\b",
    r"\b\d+\s+(?:honest-incomplete|malformed|confidently wrong)\b",
    r"\b\d+\s+graded\s+cells?\b", r"\bgraded\s+(?:cells?|record)\b", r"\bnone\s+CORRECT\b",
    r"\bE[1-5]\s+prototype\b", r"\bT1[0-9]\s+campaign\b", r"\bT[0-9]+\s+campaign\b",
    r"\bcampaign\s+fix\b", r"\bevaluation\s+campaign\b",
    r"\bB2/E[1-5]\b", r"\bheld-?out\s+(?:cell|instance)\b",
]
# NOTE: "held-out evaluation" is deliberately NOT a token here. Unlike a
# campaign identifier, the phrase legitimately appears in openPASO's own design
# prose (e.g. the verification gate explaining why an ablation flag exists).
# Flagging it would pressure a maintainer into deleting correct documentation,
# which is a worse outcome than the residual risk: an entry that named a
# specific held-out CELL or INSTANCE is still caught above, and the measured
# numbers such an entry would carry are caught by MEASURED_ORDER_PATTERNS.

# "we measured this order on this install" — the answer to a convergence study.
#
# The first four patterns were written from the contamination an audit had
# already found, and matched only that phrasing. An adversarial re-audit then
# found the same class of content passing straight through, in different words:
# `prepare_simulation('ngsolve','poisson')` was returning
# "1.669e-01 / 3.923e-02 / 8.013e-03 / 1.763e-03 (rates 2.09, 2.29, 2.18)" —
# a complete measured convergence table, handed to the agent by the tool being
# evaluated. It said "rates" rather than "observed orders" and that was the
# whole of its disguise.
#
# So these match the SHAPE of a measurement rather than one way of introducing
# it: a list of rates, a run of slash-separated error values, and a dated
# provenance stamp for one of our own runs.
MEASURED_ORDER_PATTERNS = [
    r"observed\s+orders?\s*[:=]?\s*\d+\.\d+",
    r"\bEOCs?\b[^.\n]{0,40}\d\.\d{2,}",
    r"Richardson\s+orders?\s+\d+\.\d+",
    r"verified\s+live[^.\n]{0,60}\d\.\d{2,}",
    # "(rates 2.09, 2.29, 2.18)" — a measured sequence, whatever it is called.
    r"\brates?\s+\d+\.\d{2,}\s*,\s*\d+\.\d{2,}",
    # "1.669e-01 / 3.923e-02 / 8.013e-03" — a convergence table. Three or more
    # slash-separated values in a row is our error series, not a citation.
    r"\d\.\d{2,}e[-+]\d+\s*/\s*\d\.\d{2,}e[-+]\d+\s*/\s*\d\.\d{2,}e[-+]\d+",
    # "gave orders 2.083 / 1.997 / 1.668" — the same measurement, slash-
    # separated instead of comma-separated, and introduced by "gave" rather
    # than "observed". Found still shipping after the first two patterns were
    # added, which is why these match the shape rather than the wording.
    r"\borders?\s+\d+\.\d{2,}\s*/\s*\d+\.\d{2,}",
]

# A PARAMETER TUNED FOR ONE SPECIFIC PROBLEM.
#
# This is the shape the other patterns all missed, and it is the one that gives
# most away: not a measured result but a settled ANSWER — "for the unit square
# split at x=0.5, use theta=0.42". An agent reading that skips the work the
# evaluation is asking it to do, and it compromises every future draw from the
# distribution, not just the instance it names.
#
# The discriminator is SPECIFIC GEOMETRY plus a CONCRETE VALUE. General guidance
# must survive untouched, because it is the knowledge we are trying to build:
#   * a formula is general        — "theta ~ 1/(1+rho)"
#   * a problem CLASS is general — "theta = 0.5 for problems with source terms"
#   * a named domain with numbers, next to a value, is not.
_TUNED_FOR_A_GEOMETRY = re.compile(
    r"(?:for|on)\s+(?:the\s+)?"
    r"(?:unit\s+(?:square|cube|interval)|\(0\s*,\s*[\d/.]+\)|"
    r"\d+\s*[x×]\s*\d+\s+(?:mesh|grid|domain)|"
    r"(?:square|cube|rectangle|box|domain)\s+split\s+at)"
    # The bridge must allow '.', because the geometry itself carries decimals —
    # "split at x=0.5". A first version excluded '.' to stop the match running
    # across a sentence boundary, and that made it miss the primary case while
    # catching the two easier ones. Bounded length does the same job.
    r"[^\n]{0,80}?"
    r"(?:use|set|take|choose|with)\s+\w{1,24}\s*(?:=|\bof\b|\bto\b)\s*"
    r"[-+]?\d*\.?\d+",
    re.IGNORECASE)

# TWO PATTERNS THAT WERE TRIED AND DELIBERATELY NOT KEPT
#
# A gate has to be precise or it gets switched off, and a gate that fires on
# correct content teaches people to ignore it. Both of these were measured
# against the purged branch before being rejected:
#
#   r"(?:verified empirically|live-verified|measured)\s+\d{4}-\d{2}-\d{2}"
#     — 98 hits on the ALREADY-PURGED branch. A date stamp records when we
#       checked something; it is untidy provenance, but an agent cannot read an
#       answer out of it. Making the merge gate fail in 98 places for that would
#       force a mass edit that buys no safety.
#
#   r"/home/[a-z][a-z0-9_-]*/"
#     — 16,435 hits, most of them in `src/backends/_installed_api.py`, a
#       generated manifest of where things are installed ON THIS MACHINE. That
#       file is supposed to contain local paths; that is its job. Six genuinely
#       misplaced paths in deal.II prose are a real finding, but they are a
#       code-review item, not something this pattern can isolate.
#
# What is kept above catches measured ANSWERS, and is clean on the purged
# branch — so a failure means new contamination, every time.

ALLOWED_SUFFIXES = {".py", ".json", ".md", ".txt", ".yaml", ".yml"}
# Tests and fixtures legitimately discuss our campaigns; knowledge does not.
# NOTE: these are matched against single path COMPONENTS. An earlier version
# listed "data/sessions", which contains a slash and therefore could never
# match one, and the separate substring check below covered "data/sessions" but
# not "data/webui_sessions" -- so every saved web UI run was inside the guard's
# corpus, and a user asking about a cylinder put benchmark numbers in it. Six of
# the first seven hits of the benchmark-value pattern were user runs.
SKIP_PARTS = {"tests", "scripts", "benchmarks", "__pycache__", ".git",
              "sessions", "webui_sessions", "postmortems"}


# The blind_eval modules the PRODUCT ships (make_product_tree keeps exactly these three; the rest
# of src/blind_eval is development-only and legitimately names the campaign). They ship, so they
# are held to the same no-campaign-vocabulary standard as knowledge -- comments included. Measured
# 2026-09-24: evidence.py carried "graded CORRECT" in two comments and stayed green because this
# scan did not look here; the sdist would have shipped it.
SHIPPED_BLIND_EVAL = [REPO / "src" / "blind_eval" / n
                      for n in ("interface.py", "evidence.py", "__init__.py")]


def _knowledge_files():
    for f in SHIPPED_BLIND_EVAL:
        if f.is_file():
            yield f
    for root in KNOWLEDGE_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix not in ALLOWED_SUFFIXES:
                continue
            rel = p.relative_to(REPO).as_posix()
            if any(part in rel.split("/") for part in SKIP_PARTS):
                continue
            if "data/sessions" in rel or "postmortems" in rel:
                continue
            yield p


def _served(text: str, suffix: str) -> str:
    """Return the text as an AGENT receives it, not as the source wraps it.

    Knowledge is written as adjacent Python string literals, so a single served
    sentence is stored across several source lines:

        "[Validation] The unsteady DFG 2D-2 case (Re=100) published set, "
        "which the Re=20 entry above does not cover: Cd_max 3.22-3.24, "

    Every pattern here forbids a newline inside a match, to stop it running
    across a sentence boundary. That made the whole file invisible to the
    guard: the benchmark name and its value sit either side of a seam. Joining
    the seams first is what the agent sees, and it is what we must test.
    Line numbers reported after joining are approximate by the number of seams
    above the hit, so they are printed with a "~".
    """
    if suffix == ".py":
        text = re.sub(r'"\s*\n\s*"', "", text)
        text = re.sub(r"'\s*\n\s*'", "", text)
    return text


def _hits(pattern: str):
    rx = re.compile(pattern, re.IGNORECASE)
    found = []
    for p in _knowledge_files():
        try:
            text = _served(p.read_text(errors="ignore"), p.suffix)
        except OSError:
            continue
        for m in rx.finditer(text):
            line = text[:m.start()].count("\n") + 1
            snippet = text[max(0, m.start() - 60):m.end() + 60].replace("\n", " ")
            found.append(f"{p.relative_to(REPO)}:{line}: …{snippet.strip()}…")
    return found


@pytest.mark.parametrize("pattern", CAMPAIGN_TOKENS)
def test_no_campaign_identifier_in_knowledge(pattern):
    """Knowledge must not reference our own evaluation."""
    hits = _hits(pattern)
    assert not hits, (
        "Evaluation-campaign reference found in agent-reachable knowledge.\n"
        "Knowledge describes the code, never our tests.\n  " + "\n  ".join(hits[:8]))


@pytest.mark.parametrize("pattern", MEASURED_ORDER_PATTERNS)
def test_no_measured_convergence_orders_in_knowledge(pattern):
    """The worst contamination: the answer to a convergence study, shipped.

    A general statement of a method's THEORETICAL order is fine — that is
    textbook code knowledge. Our measurements are not.
    """
    hits = _hits(pattern)
    assert not hits, (
        "Measured convergence order found in agent-reachable knowledge.\n"
        "An agent can read the answer to a convergence study out of the tool "
        "being evaluated.\n  " + "\n  ".join(hits[:8]))


def test_no_absolute_host_paths_in_served_knowledge_payloads():
    """A `*_knowledge.json` is handed to the agent wholesale. Absolute paths
    from the machine it was written on are wrong everywhere else.

    An audit found SPARTA's `installed_build` block shipping four absolute host
    paths through the live `knowledge` tool, for all ten physics — the only
    backend doing it.

    Deliberately narrow. A broader sweep over `src/backends` was measured and
    rejected: it fires 16,435 times, overwhelmingly inside the generated install
    manifest whose whole job is to record local paths, and on ordinary comments
    that mention a path while explaining something. Search-path fallbacks in a
    backend's own `backend.py` are likewise legitimate — the code has to look
    somewhere. What is not legitimate is a path baked into the knowledge PAYLOAD
    an agent is served as fact.
    """
    rx = re.compile(r"[\"'](/home/|/media/)[a-z][a-z0-9_/-]*")
    hits = []
    for root in KNOWLEDGE_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*_knowledge.json"):
            text = p.read_text(errors="ignore")
            for m in rx.finditer(text):
                line = text[:m.start()].count("\n") + 1
                hits.append(f"{p.relative_to(REPO)}:{line}: "
                            f"{text[m.start():m.start() + 90]}")
    assert not hits, (
        "An absolute host path is served to agents as knowledge. Describe how "
        "to locate the install; do not hard-code one machine's layout.\n  "
        + "\n  ".join(hits[:8]))


# A BOUNDARY CONDITION IS NOT A TUNED PARAMETER.
#
# Seam-joining (see _served) made this guard able to read entries that line
# wrapping had hidden from it, which is the point -- it immediately found two
# real hits. It also surfaced one false positive that had always been there:
#
#   "Verified on u_t = Lap(u) on the unit square with u=0 on the boundary and
#    u0 = sin(pi x) sin(pi y), whose exact amplitude decays as exp(-2 pi^2 t)"
#
# That is a MANUFACTURED SOLUTION being stated so the reader can reproduce the
# test, and stating one is method, not an answer -- the agent still has to run
# it. The guard matched because "unit square ... with u=0" has the shape of
# "unit square ... use theta=0.42".
#
# The discriminator is what is being set: a FIELD pinned to a value is a
# boundary or initial condition; a SOLVER PARAMETER given a value next to a
# geometry is the tuning this guard exists to catch. Deliberately narrow --
# a single field symbol only, so "use theta=0.42" and "set relaxation = 0.3"
# are untouched.
_A_BOUNDARY_CONDITION = re.compile(
    r"\b[a-z](?:_?[a-z0-9])?\s*=\s*[-+]?\d*\.?\d+\s+on\s+the\s+boundary"
    r"|\bon\s+the\s+boundary\s+and\s+[a-z]\d?\s*=",
    re.IGNORECASE)


def test_no_parameter_tuned_for_one_specific_problem():
    """The shape every other pattern missed, and the one that gives most away.

    Not a measured result but a settled ANSWER: "for the unit square split at
    x=0.5, use theta=0.42". An agent reading that skips the work the evaluation
    is asking it to do — and unlike a leaked instance, it compromises every
    future draw from the problem distribution, because the tuning is general.

    Calibrated in both directions, since general guidance is the knowledge we
    are trying to build and a guard that eats it would be worse than none:
    formulas ("theta ~ 1/(1+rho)"), problem CLASSES ("theta = 0.5 for problems
    with source terms"), qualitative rules ("put Dirichlet on the softer
    subdomain") and element facts ("P1 locks at nu = 0.4999") all pass.
    """
    hits = [h for h in _hits(_TUNED_FOR_A_GEOMETRY.pattern)
            if not _A_BOUNDARY_CONDITION.search(h)]
    assert not hits, (
        "a parameter tuned for one specific geometry is shipped in "
        "agent-reachable knowledge. State the rule or the formula, not the "
        "answer for one domain.\n  " + "\n  ".join(hits[:8]))


def test_the_tuned_parameter_guard_discriminates():
    """Proof the predicate works, because an untested guard is decoration."""
    for planted in ("for the unit square split at x=0.5 use theta=0.42",
                    "on the unit cube with 16 elements per side, set "
                    "relaxation = 0.3",
                    "for the 32x32 mesh choose dt of 0.001"):
        assert _TUNED_FOR_A_GEOMETRY.search(planted), planted
    for legitimate in ("theta ~ 1/(1+rho) with rho the conductance ratio",
                       "theta = 0.5 is required for problems WITH source terms",
                       "put Dirichlet on the softer subdomain",
                       "for nearly incompressible materials use a mixed "
                       "formulation",
                       "P1 elements lock at nu = 0.4999; use P2",
                       "the unit square is the usual test domain"):
        assert not _TUNED_FOR_A_GEOMETRY.search(legitimate), legitimate

    # A manufactured problem stated on the unit square is method, not tuning.
    assert _A_BOUNDARY_CONDITION.search(
        "Verified on u_t = Lap(u) on the unit square with u=0 on the boundary "
        "and u0 = sin(pi x) sin(pi y)")
    # ...but a tuned solver parameter next to a geometry still gets caught.
    assert not _A_BOUNDARY_CONDITION.search(
        "for the unit square split at x=0.5 use theta=0.42")


# Prose ABOUT exact solutions is legitimate and must not be flagged: the
# mesh-independence tool's docstring correctly says it is "for problems WITHOUT
# an exact solution", and a pitfall may warn that no closed form exists. Only a
# formula being SUPPLIED is contamination, so require a right-hand side that
# actually defines one (an equals/colon followed by an expression in x, t or a
# number) and exclude negating context.
_NEGATING = re.compile(
    r"(?:without|no|lacks?|absent|not\s+have|unavailable|if\s+an?)\s+"
    r"(?:an?\s+)?(?:known\s+|analytical\s+|closed[- ]form\s+)?exact\s+solution",
    re.IGNORECASE)


def test_no_exact_solution_shipped_for_a_coupled_case():
    """Pre-solved cases were reachable via knowledge(topic='coupling'):
    'Exact solution: T(x) = 100*(1-x)' with a per-solver error table."""
    rx = re.compile(r"[Ee]xact\s+solution\s*[:=]\s*([A-Za-z_]\w*\s*\([^)]*\)\s*=|"
                    r"[-+]?\d|[A-Za-z_]\w*\s*[-+*/^])")
    hits = []
    for f in _knowledge_files():
        try:
            text = _served(f.read_text(errors="ignore"), f.suffix)
        except OSError:
            continue
        for m in rx.finditer(text):
            ctx = text[max(0, m.start() - 120):m.end() + 40]
            if _NEGATING.search(ctx):
                continue                      # "for problems WITHOUT an exact solution"
            line = text[:m.start()].count("\n") + 1
            hits.append(f"{f.relative_to(REPO)}:{line}: …{ctx.replace(chr(10),' ').strip()[:150]}…")
    assert not hits, (
        "An exact solution is shipped in knowledge; the agent can read the "
        "answer instead of computing it.\n  " + "\n  ".join(hits[:8]))


# ── A PUBLISHED REFERENCE VALUE IS AN ANSWER ────────────────────────────────
#
# The three patterns below enforce the policy stated at the top of this file.
# Each was measured against the tree before being kept, because a guard that
# fires on correct content gets switched off:
#
#   P1  a named benchmark within one sentence of a decimal value.
#   P2  a claim that we HOLD reference values ("published bounds", "reference
#       lift/drag"), which is the assertion itself and gives the answer away
#       even when no number follows.
#   P3  a closed-form solution to a PDE, supplied with its formula.
#
# Calibration that had to be got right, and the entries that forced it:
#
#   * 4C uses "reference values" for ITS OWN regression-deck RESULT
#     DESCRIPTION values, and that is backend knowledge of exactly the kind we
#     want (fourc/generators/brownian_dynamics.py argues at length that they are
#     NOT portable between builds, which is a first-class fact). The negating
#     context therefore excludes regression/deck/binary/build/upstream.
#   * fourc/generators/low_mach.py REFUSES the Nusselt comparison and says the
#     literature value needs a non-dimensionalisation you do yourself. The
#     pattern deliberately does not match the bare phrase "literature value".
#   * deep_knowledge.py serves the closed-form (Cardano) eigenvalue formula for
#     a 3x3 symmetric tensor. That is how you compute principal stresses in
#     UFL -- an API fact, not an answer -- so P3 requires the solution be to a
#     named PDE or flow, not to any algebraic problem.
#
# Measured on the contaminated tree: 18 hit sites in 6 files, no false
# positives. After the purge it must be clean, so a failure means new
# contamination, every time.

_BENCHMARK_NAMES = (
    r"Sch[aä]fer[-\s]?Turek|DFG\s*2D|\bGhia\b|Kovasznay|Williamson|"
    r"Turek[-\s]?Hron|Kalthoff|Terzaghi"
)

# P1 — the benchmark and a number in the same sentence, either order.
_BENCHMARK_WITH_A_VALUE = re.compile(
    r"(?:" + _BENCHMARK_NAMES + r")[^.\n]{0,200}?[-+]?\d+\.\d{2,}"
    r"|[-+]?\d+\.\d{2,}[^.\n]{0,200}?(?:" + _BENCHMARK_NAMES + r")",
    re.IGNORECASE)

# P2 — the claim that a stored reference exists.
_CLAIMS_A_STORED_REFERENCE = re.compile(
    r"published\s+(?:value|bound|band|set|result)s?\b"
    r"|reference\s+(?:value|lift|drag)s?\b", re.IGNORECASE)

# P3 — a closed-form solution to a PDE, with its formula.
_CLOSED_FORM_PDE_SOLUTION = re.compile(
    r"closed[-\s]form[^.\n]{0,60}"
    r"(?:Navier[-\s]?Stokes|Stokes|Poisson|elasticity|flow|steady)"
    r"[^.\n]{0,40}solution[^.\n]{0,240}?[A-Za-z_]\w*\s*=\s*[^\s=]",
    re.IGNORECASE)

# An absence of a reference is the CORRECT framing and appears ~40 times
# ("needs no reference solution"); a solver's own regression values are
# legitimate backend knowledge. Neither is contamination.
_NOT_A_STORED_ANSWER = re.compile(
    r"(?:no|without|never|not|nothing|needs?\s+no|free\s+of|lacks?)\s+"
    r"(?:a\s+|an\s+|any\s+)?(?:reference|published)"
    r"|regression|\bdecks?\b|\bbinary\b|\bbuilds?\b|\bupstream\b"
    # An entry that sends the agent to FETCH the value is the behaviour we
    # want, and it has to be able to say so using these words. "Retrieve the
    # published envelope for your geometry" must not read as contamination
    # just because it contains "published". The discriminator is direction:
    # supplying a value versus directing the agent to obtain one.
    r"|retriev|look\s+up|\bfetch|not\s+stored\s+here|not\s+reproduced"
    r"|cite\s+what|only\s+evidence\s+when",
    re.IGNORECASE)


def _entry_around(text: str, start: int, end: int) -> str:
    """The whole served string the match sits in, not a character window.

    The unit of knowledge is the entry an agent is handed. After _served()
    joins the seams, that entry is one quoted literal, so a fixed +/-150
    window can land inside it and miss the sentence that decides the verdict:
    an entry opening "A REFERENCE VALUE BELONGS TO A GEOMETRY" and closing
    "establish which geometry your RETRIEVED reference describes" is teaching
    retrieval, and reading only its first line inverts that. Falls back to a
    window when the match is not inside a literal.
    """
    # Use the SAME quote character on both sides. Taking whichever quote came
    # first truncated an entry at an inner apostrophe -- "'Re=100 flow past a
    # cylinder' names at least two problems" ended the context three words in,
    # which cut off the sentence that made the entry legitimate.
    dq, sq = text.rfind('"', 0, start), text.rfind("'", 0, start)
    quote = '"' if dq > sq else "'"
    lo = max(dq, sq)
    hi = text.find(quote, end)
    if lo == -1 or hi == -1 or hi - lo > 4000:
        return text[max(0, start - 150):end + 150]
    return text[lo:hi]


def _hits_with_context(rx: re.Pattern, exclude_legitimate: bool):
    found = []
    for f in _knowledge_files():
        try:
            text = _served(f.read_text(errors="ignore"), f.suffix)
        except OSError:
            continue
        for m in rx.finditer(text):
            ctx = _entry_around(text, m.start(), m.end())
            if exclude_legitimate and _NOT_A_STORED_ANSWER.search(ctx):
                continue
            line = text[:m.start()].count("\n") + 1
            found.append(f"{f.relative_to(REPO)}:~{line}: "
                         f"…{' '.join(ctx.split())[:150]}…")
    return found


def test_no_published_benchmark_value_in_knowledge():
    """A named benchmark next to its number is the answer, citation or not."""
    hits = _hits_with_context(_BENCHMARK_WITH_A_VALUE, exclude_legitimate=False)
    assert not hits, (
        "A published benchmark value is shipped in agent-reachable knowledge. "
        "Name the benchmark and say how to drive the backend at it; the agent "
        "fetches the reference at run time and cites what it fetched. Keep the "
        "lesson, drop the number.\n  " + "\n  ".join(hits[:8]))


def test_no_claim_that_openpaso_holds_reference_values():
    """"Reference lift/drag at Re=20/100" asserts we hold the answer."""
    hits = _hits_with_context(_CLAIMS_A_STORED_REFERENCE,
                              exclude_legitimate=True)
    assert not hits, (
        "Knowledge claims openPASO holds published reference values. It does "
        "not and must not; the agent retrieves them.\n  " + "\n  ".join(hits[:8]))


def test_no_closed_form_pde_solution_shipped():
    """Kovasznay flow evaded the exact-solution guard by phrasing alone."""
    hits = _hits_with_context(_CLOSED_FORM_PDE_SOLUTION,
                              exclude_legitimate=True)
    assert not hits, (
        "A closed-form solution to a PDE is shipped with its formula. The "
        "METHOD (a test whose convective term does not vanish) is knowledge; "
        "the solution is the answer.\n  " + "\n  ".join(hits[:8]))


def test_the_published_value_guards_discriminate():
    """Proof the three predicates work in both directions.

    An untested guard is decoration, and a guard that eats correct knowledge is
    worse than none. The legitimate strings below are real entries from this
    repository that must keep passing.
    """
    for planted in (
            "should produce drag coefficient Cd around 5.57 (Schafer-Turek)",
            "differs from the Schäfer-Turek 1996 reference (Re=20: C_D ~ 5.58)",
            "the unsteady DFG 2D-2 published set: Cd_max 3.22-3.24",
            "an UNBOUNDED cylinder at the same Re sheds at St around 0.164 "
            "(Williamson 1996)"):
        assert _BENCHMARK_WITH_A_VALUE.search(planted), planted

    for planted in ("Driven-cavity benchmark; Ghia reference values at Re=100",
                    "Reference lift/drag at Re=20/100",
                    "to within published bounds"):
        assert _CLAIMS_A_STORED_REFERENCE.search(planted), planted

    assert _CLOSED_FORM_PDE_SOLUTION.search(
        "Kovasznay flow is a closed-form steady Navier-Stokes solution whose "
        "convective term is NOT zero, with lam = Re/2 - sqrt(Re^2/4)")

    # Naming a benchmark without giving its answer is expertise, and required:
    # an agent has to know which case it is being asked for.
    for legitimate in (
            "channel_with_cylinder is the Schäfer-Turek benchmark geometry",
            "Classic CFD benchmark (Ghia et al., 1982)",
            "the Turek-Hron FSI benchmark family",
            "DFG 2D-2 is the unsteady case; Umax=1.5 gives Ubar=1.0"):
        assert not _BENCHMARK_WITH_A_VALUE.search(legitimate), legitimate

    # A solver's own regression values, and the absence of a reference, are not
    # contamination. Both are checked through the same context filter the
    # tests use, not against the bare pattern.
    for legitimate in (
            "4C's own regression decks reproduce the reference values stored "
            "in them",
            "its reference values are NOT PORTABLE between builds",
            "this tool needs no reference solution"):
        assert _NOT_A_STORED_ANSWER.search(legitimate), legitimate

    # The Cardano eigenvalue formula is an API fact, not a physics answer.
    assert not _CLOSED_FORM_PDE_SOLUTION.search(
        "in 3D the same construction works with the three eigenvalues from "
        "the closed-form (Cardano) solution: qq = ufl.tr(A) / 3")


def test_the_scanner_cannot_see_its_own_fixtures():
    """The guard must not be able to read the strings it plants as proof.

    test_the_published_value_guards_discriminate deliberately contains real
    contamination -- "Cd around 5.57 (Schafer-Turek)", "Cd_max 3.22-3.24" --
    because a pattern nobody has seen fire is not a guard. If the scanner
    could reach this file it would report ITSELF as contaminated, stay red
    after a genuine purge, and teach everyone to ignore it. A guard that
    cannot go green is worse than no guard.

    This holds today by one word in SKIP_PARTS, which is a thin thing to rest
    on: the same set carried "data/sessions" for months, a string that can
    never match a path COMPONENT, so users' saved web UI runs sat inside the
    corpus and a person asking about a cylinder put benchmark numbers in it.
    That is exactly how a path widens without anyone deciding to widen it.
    """
    scanned = {p.resolve() for p in _knowledge_files()}
    assert Path(__file__).resolve() not in scanned, (
        "the contamination scanner can read its own test fixtures; its "
        "planted strings will report as contamination forever")
    assert not [p for p in scanned if "tests" in p.parts], \
        "tests are not knowledge and must stay outside the corpus"
    assert not [p for p in scanned if "webui_sessions" in p.parts], \
        "saved web UI runs are a user's own data, not served knowledge"
    # and the corpus must not be empty, or every test above passes vacuously
    assert len(scanned) > 100, f"only {len(scanned)} files scanned — the " \
        "corpus collapsed and every guard in this file is now vacuous"


# ── A SHIPPED DECK MUST NOT CARRY 4C'S OWN ANSWER BLOCK ────────────────────
#
# 4C's RESULT DESCRIPTION section is how a regression deck asserts expected
# nodal values -- "dispx 0.017936890424460145 at node 15, tolerance 1e-12".
# In an upstream test that is the mechanism. In a deck openPASO SHIPS it is
# the answer to that exact configuration, handed over before the agent has
# run anything.
#
# One was removed by hand when the porofluid deck was added. Nothing stopped
# it coming back: an adversarial audit appended the block above verbatim to
# that same file and the whole suite stayed green. It is the canonical 4C
# answer format, it is machine-readable, and re-adding it was one revert
# away. Every other pattern in this file looks for prose; this one looks for
# a section name.
_RESULT_DESCRIPTION = re.compile(r"^\s*RESULT DESCRIPTION\s*:", re.M)
# and the same content without the section header, which is what a
# hand-written block would look like
_ASSERTED_VALUE = re.compile(r"\bVALUE\s*:\s*[-+]?\d[^\n]*\n[^\n]*\bTOLERANCE\s*:",
                             re.M)


def _shipped_decks() -> list[Path]:
    return sorted((REPO / "src" / "backends").rglob("decks/*.yaml")) + \
           sorted((REPO / "src" / "backends").rglob("decks/*.4C.yaml"))


def test_no_shipped_deck_carries_an_expected_result_block():
    """A deck teaches the input grammar. It does not carry the answer."""
    decks = _shipped_decks()
    assert len(decks) > 20, (
        f"only found {len(decks)} shipped decks — the glob broke, and an "
        f"empty corpus would pass this test over nothing")
    bad: list[str] = []
    for d in decks:
        text = d.read_text(errors="ignore")
        for rx, what in ((_RESULT_DESCRIPTION, "a RESULT DESCRIPTION section"),
                         (_ASSERTED_VALUE, "an asserted VALUE/TOLERANCE pair")):
            m = rx.search(text)
            if m:
                line = text[:m.start()].count("\n") + 1
                bad.append(f"{d.relative_to(REPO)}:{line}: {what}")
    assert not bad, (
        "a shipped 4C deck carries expected results. That block is how a "
        "regression deck asserts nodal values; in a deck we hand out it is "
        "the answer to the configuration. Remove it and say in the title "
        "what the block is and how to write one.\n  " + "\n  ".join(bad))


def test_the_expected_result_guard_can_actually_fail():
    """Proof, because the audit that found this gap found it by planting."""
    planted = ('RESULT DESCRIPTION:\n  - STRUCTURE:\n      DIS: "structure"\n'
               '      NODE: 15\n      QUANTITY: "dispx"\n'
               '      VALUE: 0.017936890424460145\n      TOLERANCE: 1e-12\n')
    assert _RESULT_DESCRIPTION.search(planted)
    assert _ASSERTED_VALUE.search(planted)
    # a deck that merely EXPLAINS the block, as ours now do, must pass
    legitimate = ('  - "4C\'s RESULT DESCRIPTION block is deliberately absent."\n'
                  '  - "Add your own entries naming DIS, NODE, QUANTITY, VALUE"\n'
                  '  - "and TOLERANCE when you want 4C to self-check."\n')
    assert not _RESULT_DESCRIPTION.search(legitimate)
    assert not _ASSERTED_VALUE.search(legitimate)
