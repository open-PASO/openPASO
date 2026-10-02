"""Find the ONE pitfall that matches the error in front of you.

WHY THIS EXISTS
---------------
`knowledge(topic='pitfalls', solver=...)` returned every pitfall the backend
knows, for every physics it supports, unfiltered. Measured on this tree:

    kratos   87,109 chars   (~22k tokens)
    fourc    1,044,517 chars of knowledge behind it, 248 pitfall entries

The tool's signature accepts `physics` and `signal`, and the pitfalls branch
read NEITHER. `signal` was wired only to `topic='postmortems'`; `physics` was
accepted and ignored, so the branch looped over every supported physics. The
consequence is not that the payload is too big — at a 200k context window 22k
tokens is affordable, and the knowledge is meant to be comprehensive. The
consequence is that the two questions an agent most needs to ask had no answer:

    "I got THIS error text — which pitfall is it?"
    "I am solving Poisson — which of these 201 entries apply?"

An agent holding a stack trace had to pull all 87k chars and scan them itself.
A small model can do that, but it spends its attention on triage instead of on
the problem, and it is exactly the model least able to afford that.

WHAT THIS DOES NOT DO
---------------------
It does not shrink or summarise anything. An unfiltered call returns the same
full dump it always did, byte for byte. Narrowing is opt-in and additive: pass
a filter, get the matching subset PLUS a note saying how many entries were held
back and the exact call that returns them all. Nothing becomes unreachable.

THE FORMAT IT RELIES ON
-----------------------
Measured over all 1122 entries in the tree, the convention is near-universal:

    [Category] <the fact, and what to do about it, in prose>.
    Signal: <what you observe when it bites>. (Verified <date>, <version>)

    [Category] tag present   1121/1122  (100%)
    Signal: clause present   1122/1122  (100%)
    distinct categories      16, of which the top 8 cover 97.5%
    exact duplicate entries  0
    median entry length      398 chars

So `[Category]` and `Signal:` are the two fields that can be relied on, and
they are the two axes this module indexes. The fix is embedded in the prose
with no marker — deliberately left that way, because prose that explains why
is worth more to a weak model than a terse field, and re-cutting 1122 entries
while agents are concurrently adding hundreds more would collide for no gain.

MATCH HONESTY
-------------
A retrieval layer that guesses and does not say it guessed is worse than none:
the agent applies a fix for a different bug and trusts it because the tool
surfaced it. So every result carries the match mode that produced it, and a
weak match is labelled weak rather than presented as the answer.
"""
from __future__ import annotations

import re
from typing import Any

# The category vocabulary actually in use, measured from the tree rather than
# invented. Counts at time of writing, over 1121 tagged entries:
#   Numerical 477, API 232, Input 176, Syntax 71, Physics 57, Integration 51,
#   Performance 19, Output 14, Mesh 9, Validation 5, Numerics 4, Parallel 2,
#   Solver 1, Discretization 1, Environment 1, Workflow 1
# `Numerics` is a spelling variant of `Numerical`; the four singletons are
# one-offs. Both are normalised on read so a category filter does not silently
# miss entries, without rewriting the source entries themselves.
CANONICAL_CATEGORIES = {
    "numerical": "Numerical",
    "numerics": "Numerical",          # spelling variant, 4 entries
    "discretization": "Numerical",    # singleton, same axis
    "solver": "Numerical",            # singleton, same axis
    "api": "API",
    "input": "Input",
    "syntax": "Syntax",
    "physics": "Physics",
    "integration": "Integration",
    "environment": "Integration",     # singleton, same axis
    "performance": "Performance",
    "parallel": "Performance",        # 2 entries, same axis
    "output": "Output",
    "mesh": "Mesh",
    "validation": "Validation",
    "workflow": "Validation",         # singleton, same axis
    # SPARTA's dominant axis, 22 of its 79 entries. Not a synonym for anything
    # already here: getting a DSMC deck to START is a distinct failure surface
    # from Input (a malformed keyword) or Integration (a missing dependency) —
    # it covers grid/box/species declaration order and the mandatory commands a
    # run needs before it will begin at all. Without this key, category
    # filtering silently skipped SPARTA's largest group.
    "setup": "Setup",
    # Axes that arrived with the compound convention in `_cross.py` (below).
    # Both are real and distinct: a units mismatch and a boundary-condition
    # semantic difference are the two failure modes that converge cleanly and
    # give a confidently wrong answer, so they deserve their own filters.
    "units": "Units",
    "bc": "BC",
    # Sub-kinds that arrived with `_setup.py`, which uses the same compound
    # convention as `_cross.py`: [Integration][Install], [Integration][Discovery],
    # [Integration][FirstRun], [Integration][BuildConfig],
    # [Integration][Portability]. The axis is read from the LAST tag, so without
    # these keys the five sub-kinds fell outside the vocabulary and category
    # filtering skipped every install and build-configuration entry.
    #
    # They all fold onto Integration rather than earning separate axes: each is
    # a way the ENVIRONMENT rather than the model is wrong — the same axis
    # "environment" already maps to. Keeping them distinct in the tag preserves
    # the sub-kind for a reader while keeping the filter complete.
    "install": "Integration",
    "discovery": "Integration",
    "firstrun": "Integration",
    "buildconfig": "Integration",
    "portability": "Integration",
    # deal.II's [Build] arrived separately with knowledge/dealii-verify, as a
    # SINGLE tag rather than a compound one. Whether Assert is compiled in is a
    # build-time fact, so it sits on the same axis as [Integration][BuildConfig].
    "build": "Integration",
    # Sub-kinds that arrived with feature/tsi-coupling and feature/fsi-coupling,
    # which tagged four entries with their GROUP NAME in `pitfalls.py`
    # (silent_wrong, verification_limits, participant_contract,
    # capability_limits) rather than with an axis. Every other entry in those
    # same groups is tagged by axis — _SILENT_WRONG's own members carry
    # [Validation], [Numerical], [Mesh], [BC], [Units], [Integration] — so the
    # group is not the category and these four fell outside the vocabulary,
    # which silently dropped them from category filtering.
    #
    # Each folds onto the axis its own group predominantly uses, so filtering
    # returns them alongside the entries they belong with, while the tag keeps
    # the sub-kind visible to a reader:
    #
    #   Silent-Wrong  -> Validation   (_SILENT_WRONG is Validation-dominant; a
    #                                  coupling that converges to a wrong answer
    #                                  with nothing firing is a validation blind
    #                                  spot, which is what the group is about)
    #   Verification  -> Validation   (_CHECK_LIMITS is 5x Validation; these say
    #                                  what the checks CANNOT see)
    #   Contract      -> API          (_CONTRACT is API-dominant; it is how a
    #                                  participant script must be written)
    #   Capability    -> Integration  (_CAPABILITY_LIMITS's tool-fact axis: 4C's
    #                                  match_nodes octree tolerance is a fixed
    #                                  property of the build, not of the model)
    "silent-wrong": "Validation",
    "verification": "Validation",
    "contract": "API",
    "capability": "Integration",
}

# `_cross.py` carries the cross-backend collation pitfalls — the ones that only
# fire on the delta between two codes — and tags them COMPOUND:
#
#     [Cross-Backend][Units]  4      [Cross-Backend][Physics]      7
#     [Cross-Backend][Mesh]   5      [Cross-Backend][Numerical]    7
#     [Cross-Backend][BC]     3      [Cross-Backend][Output]       6
#     [Cross-Backend][API]    2      [Cross-Backend][Performance]  1
#
# That is a better scheme than a single tag, not a deviation from one: a
# namespace saying which corpus the entry belongs to, plus the axis it sits on.
# So the parser reads the LAST tag as the category and everything before it as
# namespace, which leaves single-tag entries behaving exactly as before.
_CATEGORY_RE = re.compile(r"^\s*((?:\[[^\]]{1,40}\])+)")
_ONE_TAG_RE = re.compile(r"\[([^\]]{1,40})\]")
NAMESPACES = {"cross-backend": "Cross-Backend"}
_SIGNAL_RE = re.compile(r"Signal:\s*(.*)", re.IGNORECASE | re.DOTALL)
# Trailing provenance: "(Verified empirically 2026-06-01 ...)" / "(Audit ...)".
# Kept in the entry text — it tells the agent the claim was executed and when —
# but excluded from the signal clause so a date cannot satisfy a symptom match.
_PROVENANCE_RE = re.compile(
    r"\((?:verified|audit|checked|measured|confirmed)\b[^()]*\)\s*$",
    re.IGNORECASE)

# Words too common to distinguish one pitfall from another. Matching on these
# alone produced hits on unrelated entries in every backend tried.
_STOP = frozenset("""
a an the and or but if of in on at to for from with without by is are was were
be been being it its this that these those as not no nor so than then there
here when where which who whom what how why all any both each few more most
other some such only own same too very can will just should now use used using
you your we our i my error warning message output file files line lines code
""".split())


def knowledge_keys(solver: str) -> list[str]:
    """Every knowledge area a backend HOLDS, not just the ones it advertises.

    `supported_physics()` is a capability claim: it means a generator can build
    a runnable input. Knowledge coverage is wider, and the gap is not small.
    Measured on kratos: 41 knowledge areas, 20 advertised, and the 21 unlisted
    ones hold 65 verified pitfall entries — geomechanics, RANS, IGA, ROM,
    topology optimisation, chimera, FEM-to-DEM and FSI among them. Every one
    was checked: none has a template and `generate_input` raises ValueError for
    all of them, so leaving them out of `supported_physics()` is CORRECT. What
    was wrong is that nothing enumerated them, so no agent could find out they
    exist. Undiscoverable knowledge is knowledge nobody has.

    Returns [] when the backend keeps its knowledge somewhere this cannot see,
    rather than guessing — a wrong key list would invent areas that do not
    exist, which is the failure this whole effort is against.

    DISCOVERY IS BEHAVIOUR-BASED, NOT NAME-BASED. The nine backends do not
    share a registry name: kratos uses `KNOWLEDGE`, 4C `_GENERATOR_SPECS`,
    deal.II `_KNOWLEDGE_SPECS`, FEniCSx `_PHYSICS_MODULES`, SPARTA `_PHYSICS`.
    A whitelist of those names would go stale the first time one is renamed and
    would silently start reporting "nothing hidden" — the exact shape of a gate
    that passes while the thing it guards rots. So candidates are gathered
    broadly and then CONFIRMED by asking `get_knowledge()`: a key counts only
    if the backend answers it with content. That cannot invent an area, and it
    cannot be defeated by renaming a constant.
    """
    import importlib
    from core.registry import get_backend

    be = get_backend(solver)
    if be is None:
        return []

    candidates: set[str] = set()
    for mod in (f"backends.{solver}.generators", f"backends.{solver}",
                f"backends.{solver}.backend"):
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        for attr in dir(m):
            # Any module-level mapping that could be a knowledge registry.
            # Over-collecting is safe; the confirmation step below discards
            # anything the backend does not actually serve.
            if not (attr.isupper() or attr.startswith("_")):
                continue
            value = getattr(m, attr, None)
            if isinstance(value, dict) and len(value) > 2:
                candidates.update(k for k in value if isinstance(k, str))
    # Module filenames too: several backends key knowledge by module name.
    from pathlib import Path
    be_dir = Path(__file__).resolve().parents[1] / "backends" / solver
    if be_dir.is_dir():
        for py in be_dir.rglob("*.py"):
            if not py.stem.startswith("__"):
                candidates.add(py.stem)

    confirmed = []
    for key in sorted(candidates):
        try:
            k = be.get_knowledge(key)
        except Exception:
            continue
        # A dict carrying real content. `error` keys are how some backends
        # signal an unknown physics, and must not be mistaken for an area.
        if isinstance(k, dict) and k and "error" not in k:
            confirmed.append(key)
    return confirmed


def reference_only_areas(solver: str, advertised: set[str]) -> list[str]:
    """Knowledge areas the backend holds but genuinely CANNOT run.

    Two things disqualify an area, and the second was learned the hard way.

    Not in `supported_physics()` is necessary but not sufficient: 4C keeps 158
    aliases (`elasticity` -> `structure`, `scatra`, `xfem`, ...) that resolve
    through `get_knowledge()` and that `generate_input()` builds real inputs
    for. Calling those reference-only was wrong twice over — it told an agent
    4C had 154 subjects it cannot run, when it can run all of them and already
    advertises them under their canonical names.

    So an area qualifies only when the backend also refuses to generate an
    input for it. That is the behavioural definition of "we know about this but
    cannot drive it", and it is checkable rather than declared.
    """
    keys = knowledge_keys(solver)
    adv = {a.lstrip("_").lower() for a in advertised}
    from core.registry import get_backend
    be = get_backend(solver)

    out = []
    for k in keys:
        if k.startswith("_") or k.lstrip("_").lower() in adv:
            continue
        if be is not None:
            try:
                built = be.generate_input(k, "default", {})
            except Exception:
                built = None      # no generator: genuinely reference-only
            if isinstance(built, str) and len(built) > 200 and \
                    "NOT a runnable" not in built:
                continue          # an alias of something runnable
        out.append(k)
    return out


def parse_entry(text: str) -> dict[str, Any]:
    """Split one pitfall string into the fields that can be relied on."""
    category_raw = ""
    namespace = ""
    m = _CATEGORY_RE.match(text)
    if m:
        tags = _ONE_TAG_RE.findall(m.group(1))
        # Last tag is the axis; anything before it is namespace. A single-tag
        # entry therefore parses exactly as it did before this was added.
        category_raw = tags[-1].strip()
        if len(tags) > 1:
            namespace = " ".join(
                NAMESPACES.get(t.strip().lower(), t.strip()) for t in tags[:-1])
    category = CANONICAL_CATEGORIES.get(category_raw.lower(), category_raw)

    body = text[m.end():].strip() if m else text.strip()

    signal = ""
    sm = _SIGNAL_RE.search(text)
    if sm:
        signal = _PROVENANCE_RE.sub("", sm.group(1)).strip()

    provenance = ""
    pm = _PROVENANCE_RE.search(text)
    if pm:
        provenance = pm.group(0).strip()

    return {
        "text": text,
        "category": category,
        "namespace": namespace,
        "category_as_written": category_raw,
        "signal": signal,
        "provenance": provenance,
        "advice": body,
    }


def _normalise(s: str) -> str:
    """Fold the differences a copy-pasted error message picks up in transit.

    Quote style, whitespace runs and case all vary between what a solver prints,
    what lands in a log, and what an agent pastes back. Matching on the raw
    string misses on all three.
    """
    s = s.lower()
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = re.sub(r"[\\'\"`]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# DOMAIN VOCABULARY. An agent and an entry routinely name the same thing
# differently, and the mismatch silently defeats token matching. Measured case
# that motivated this: the query "element locking hexahedral" scored 1/3 against
# an entry reading "SHEAR LOCKING: linear hex8 (3D8N) and quad4 (2D4N) elements
# lock in bending-dominated problems" — `element` missed `elements`, and
# `hexahedral` missed both `hex8` and `3D8N`. The agent asked exactly the right
# question and was told nothing was known, which is worse than an unfiltered
# dump: absence reads as authoritative.
#
# Every group below is one concept under the names actually used in this tree's
# entries and in the codes' own output.
_SYNONYMS: dict[str, str] = {}
for _canon, _names in {
    "hex": ("hex", "hexa", "hex8", "hex20", "hex27", "hexahedra", "hexahedral",
            "hexahedron", "3d8n", "3d20n", "3d27n", "brick"),
    "quad": ("quad", "quad4", "quad8", "quad9", "quadrilateral", "2d4n",
             "2d8n", "2d9n"),
    "tet": ("tet", "tet4", "tet10", "tetra", "tetrahedra", "tetrahedral",
            "tetrahedron", "3d4n", "3d10n"),
    "tri": ("tri", "tri3", "tri6", "triangle", "triangular", "2d3n", "2d6n"),
    "wedge": ("wedge", "prism", "prismatic", "penta", "3d6n"),
    # `overstiff` belongs here; a bare `overly` does NOT — it is a generic
    # adverb, and folding it in made every entry containing "overly" answer a
    # locking query. A synonym group may only hold words that name the concept.
    "lock": ("lock", "locking", "locks", "locked", "overstiff", "stiffening"),
    "converge": ("converge", "converges", "converged", "convergence",
                 "diverge", "diverges", "diverged", "divergence",
                 "nonconvergence"),
    "singular": ("singular", "singularity", "illconditioned", "conditioning",
                 "condition"),
    "dirichlet": ("dirichlet", "essential", "fixed", "clamped", "prescribed"),
    "neumann": ("neumann", "natural", "traction", "flux"),
    "nan": ("nan", "inf", "infinity", "notanumber", "indefinite"),
    "segfault": ("segfault", "segmentation", "sigsegv", "crash", "crashed",
                 "abort", "aborted"),
    "mesh": ("mesh", "meshes", "grid", "discretisation", "discretization"),
    "element": ("element", "elements", "cell", "cells"),
    "node": ("node", "nodes", "vertex", "vertices", "dof", "dofs"),
    "material": ("material", "materials", "constitutive"),
    "boundary": ("boundary", "boundaries", "bc", "bcs"),
    "timestep": ("timestep", "timesteps", "dt", "increment", "increments"),
    "solver": ("solver", "solvers", "solve", "solves", "linearsolver"),
    "precond": ("precond", "preconditioner", "preconditioning", "amg", "ilu"),
    "parallel": ("parallel", "mpi", "ranks", "rank", "processes", "procs"),
    "import": ("import", "imports", "importerror", "modulenotfounderror",
               "module"),
    "attribute": ("attribute", "attributeerror", "attr"),
    "key": ("key", "keyerror", "missing", "notfound"),
    # COUPLING VOCABULARY. Added after the coupling entries landed and 3 of 37
    # realistic symptom queries missed. Each group below is one concept under
    # the names that actually appear in coupling entries and driver output, and
    # each fixes a fold the stemmer cannot do on its own: `coupling` stems to
    # `coupl` while `couple` stays `couple`, `relaxation` strips no suffix at
    # all, and `mapping` becomes `mapp` while `map` stays `map`.
    "coupling": ("coupling", "coupled", "couple", "couples", "partitioned",
                 "staggered", "cosimulation"),
    "participant": ("participant", "participants", "partner", "partners",
                    "subdomain", "subdomains"),
    "relaxation": ("relaxation", "relax", "relaxed", "relaxes",
                   "underrelaxation", "theta", "omega"),
    "mapping": ("mapping", "map", "maps", "mapped", "interpolation",
                "interpolate", "interpolated", "nearestneighbour",
                "nearestneighbor", "nearestprojection", "rbf"),
    "precice": ("precice", "pyprecice", "libprecice", "fenicsprecice",
                "cyprecice"),
    "accelerator": ("accelerator", "acceleration", "aitken", "iqn",
                    "anderson", "quasinewton"),
    "checkpoint": ("checkpoint", "checkpoints", "checkpointing"),
    "subcycling": ("subcycling", "subcycle", "subcycles", "timewindow"),
    "monolithic": ("monolithic", "monolith", "unsplit", "unpartitioned"),
    # The backend's own names. `4c` was unmatchable entirely — see _tokens.
    "fourc": ("fourc", "4c", "baci"),
}.items():
    for _n in _names:
        _SYNONYMS[_n] = _canon


def _stem(w: str) -> str:
    """Fold the inflections that separate a query from an entry.

    Deliberately crude and conservative — only suffixes that change nothing
    about which pitfall is meant, and only when a real stem remains. An
    aggressive stemmer would collide unrelated identifiers, which in this
    setting means confidently surfacing the wrong fix.
    """
    if w in _SYNONYMS:
        return _SYNONYMS[w]
    for suf in ("ing", "ed", "es", "s"):
        if len(w) - len(suf) >= 4 and w.endswith(suf):
            cand = w[: -len(suf)]
            return _SYNONYMS.get(cand, cand)
    return w


def _tokens(s: str) -> set[str]:
    """Distinctive words of a query, for when the substring match misses.

    Numeric-suffixed identifiers are kept whole AND stemmed, so `hex8` matches
    an entry saying `hexahedral` without losing the ability to match `hex8`
    verbatim where an entry uses the code's own element name.

    THE PATTERN MUST ADMIT A LEADING DIGIT. The first version required
    `[a-z_]` first, so `4c` — the name of the largest backend — was dropped from
    every query and from every entry, making it impossible to match on. Two
    characters is also below the old 3-character floor, so both bounds had to
    move. Bare numbers are still excluded: a digit must be followed by a letter,
    which admits `4c` and `2d` while rejecting `35` and `1e-6`, where a number
    from one run is meaningless as a search term.
    """
    # `\d+[a-z]` alone also captures the mantissa of scientific notation:
    # `1e-6` yielded `1e`. Harmless (it matches nothing) but it is measurement
    # noise in the index, so require at least two trailing letters OR a letter
    # followed by a digit — admitting `4c`? no: `4c` is one letter. Handle it by
    # excluding a lone exponent marker instead, which is the actual case.
    words = re.findall(r"[a-z_][a-z0-9_]{2,}|\d+[a-z][a-z0-9_]*",
                       _normalise(s))
    words = [w for w in words if not re.fullmatch(r"\d+[ed]", w)]
    return {_stem(w) for w in words if _stem(w) not in _STOP}


def match_signal(entry: dict[str, Any], query: str) -> tuple[str, float]:
    """How well does this entry match an observed symptom?

    Returns (mode, score). Modes, strongest first:

      "substring"      the query appears verbatim (modulo quoting/whitespace)
                       inside the entry's Signal: clause. This is the case that
                       matters: the agent pasted the error and we found it.
      "substring_body" the query appears in the entry's prose but not its
                       Signal: clause — still a real hit, e.g. the agent pasted
                       an API name rather than an error.
      "all_tokens"     every distinctive word of the query is somewhere in the
                       entry. Paraphrased or partially-quoted errors land here.
      "some_tokens"    a fraction of them are. Reported, but labelled weak, and
                       never presented as the answer.
      ""               no relationship worth surfacing.
    """
    q = _normalise(query)
    if not q:
        return "", 0.0
    sig = _normalise(entry.get("signal", ""))
    whole = _normalise(entry.get("text", ""))

    if sig and q in sig:
        return "substring", 1.0
    if q in whole:
        return "substring_body", 0.85

    qt = _tokens(query)
    if not qt:
        return "", 0.0
    wt = _tokens(entry.get("text", ""))
    hit = qt & wt
    frac = len(hit) / len(qt)
    if frac >= 1.0:
        return "all_tokens", 0.7
    # Require both a majority of the query's words AND at least two of them, so
    # a one-word query cannot promote an unrelated entry on a single collision.
    if frac >= 0.6 and len(hit) >= 2:
        return "some_tokens", 0.3 + 0.3 * frac
    return "", 0.0


def flatten(all_pitfalls: dict[str, Any]) -> list[dict[str, Any]]:
    """Walk the {physics: [entry, ...]} shape the backends return.

    Values are not uniformly lists of strings — some physics carry dicts of
    sub-lists, and `element_types` carries a mapping. Anything that is not a
    string is kept and passed through under its own physics key rather than
    dropped, so narrowing can never lose content that an unfiltered call shows.
    """
    out: list[dict[str, Any]] = []
    for physics, value in all_pitfalls.items():
        for text in _strings_under(value):
            e = parse_entry(text)
            e["physics"] = physics
            out.append(e)
    return out


def _strings_under(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [s for v in value for s in _strings_under(v)]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings_under(v)]
    return []


# How many "closest related" entries a physics filter may add. Small on
# purpose: the point is to rescue the ONE decisive entry the agent could
# not name, not to hand back the corpus it was trying to narrow.
_RELATED_CAP = 8


def narrow(all_pitfalls: dict[str, Any], *, physics: str = "",
           signal: str = "", category: str = "") -> dict[str, Any]:
    """Apply the requested filters and report exactly what was held back.

    Returns a dict with the surviving entries and an accounting of the rest.
    Callers must render the accounting: a filtered answer that does not say it
    was filtered invites an agent to conclude the knowledge base is empty on a
    subject it simply did not ask the right way about.
    """
    entries = flatten(all_pitfalls)
    total = len(entries)
    kept = entries
    applied: list[str] = []

    if physics:
        want = physics.strip().lower()
        # Substring both ways: a backend's key may be `linear_elasticity` while
        # the agent asks for `elasticity`, or the reverse.
        sel = [e for e in kept
               if want in e["physics"].lower() or e["physics"].lower() in want]
        # TOKEN OVERLAP, because substring matching punishes the agent for
        # using the TASK's vocabulary. One task said "anisotropic diffusion";
        # the backend's buckets are `poisson` and `convection_diffusion`.
        # Neither substring test fires, so `physics='anisotropic_diffusion'`
        # selected NOTHING and the reply was 5,685 chars against 7,047 for the
        # unfiltered call — the agent got LESS knowledge for being specific in
        # the words it was given.
        if not sel:
            want_tok = {t for t in _tokens(want) if len(t) > 3}
            sel = [e for e in kept
                   if want_tok & {t for t in _tokens(e["physics"].lower())
                                  if len(t) > 3}]
        # Physics-independent buckets stay in — they apply to every problem, and
        # dropping them is how an agent misses the input-format pitfall that was
        # going to bite it.
        general = [e for e in kept
                   if e["physics"] in ("general_input_format", "element_types",
                                       "community_contributed")
                   and e not in sel]
        # A FILTER THAT MATCHES NOTHING MUST NOT EMPTY THE ANSWER.
        #
        # This function's own docstring promises that "narrowing can never make
        # knowledge unreachable or make a miss look like an empty database",
        # and the code broke that promise: an unrecognised physics name left
        # only the physics-independent buckets, so the agent read a near-empty
        # reply as "openPASO knows nothing about my problem". Measured on one
        # recorded run, where the one fact that decides a correct result
        # from a confidently wrong one lives under `poisson` and the task calls
        # the problem anisotropic diffusion.
        #
        # The first repair here returned EVERYTHING when nothing matched, which
        # traded a false-empty answer for a 130,283-char flood — the exact
        # failure this module's own comment calls "a denial of service on the
        # agent's own context". So the fallback is RANKED and BOUNDED: score
        # every remaining entry against the physics phrase with the same
        # signal matcher used below, keep the best few, and say plainly that
        # the name matched no bucket. Bounded and relevant beats both empty
        # and everything.
        related: list = []
        pool = [e for e in kept if e not in sel]
        scored_rel = []
        for e in pool:
            mode, score = match_signal(e, physics)
            if mode:
                scored_rel.append((score, len(_tokens(e.get("text", ""))), e))
        scored_rel.sort(key=lambda t: (-t[0], t[1]))
        related = [e for _s, _n, e in scored_rel[:_RELATED_CAP]]
        if not sel:
            kept = related + general
            applied.append(
                f"physics={physics!r} MATCHED NO BUCKET for this backend; "
                f"showing the {len(related)} entries whose text is closest to "
                f"that phrase, because a filter that matches nothing must not "
                f"look like an empty knowledge base — ask again with "
                f"index=True to see the bucket names")
        else:
            kept = sel + related + general
            applied.append(f"physics={physics!r}"
                           + (f" (plus {len(related)} closest related)"
                              if related else ""))

    if category:
        want = CANONICAL_CATEGORIES.get(category.strip().lower(),
                                        category.strip())
        kept = [e for e in kept if e["category"].lower() == want.lower()]
        applied.append(f"category={category!r}")

    match_report: dict[str, int] = {}
    if signal:
        scored = []
        for e in kept:
            mode, score = match_signal(e, signal)
            if mode:
                scored.append((score, mode, e))
                match_report[mode] = match_report.get(mode, 0) + 1
        # TIES MUST NOT RESOLVE BY CORPUS ORDER. A generic query such as
        # "converged at iteration 2 with residual 0.0" matched 19 of 32 coupling
        # entries at the same score, and a stable sort then handed first place
        # to whichever happened to be earliest in the file — in practice the
        # longest entry, because a long entry contains more words and so matches
        # more loosely. Break ties towards the SMALLER vocabulary: the entry
        # that matched the query using fewer other words is the more specific
        # one, and specificity is exactly what an agent needs from first place.
        scored.sort(key=lambda t: (-t[0], len(_tokens(t[2].get("text", "")))))
        kept = [dict(e, match=mode) for _, mode, e in scored]
        applied.append(f"signal={signal!r}")

    return {
        "entries": kept,
        "total_available": total,
        "shown": len(kept),
        "filters_applied": applied,
        "match_modes": match_report,
    }


def render(result: dict[str, Any], solver: str) -> str:
    """Format a narrowed result for an agent, honestly.

    Three things every rendering must carry, because each corresponds to a way
    an agent has been observed to go wrong:

      * how many entries exist that it is NOT seeing, and the call that shows
        them — otherwise a narrow miss reads as "nothing known";
      * the match mode per entry — otherwise a token-overlap guess reads as a
        confirmed identification;
      * for a signal query that found nothing, an explicit statement that no
        known pitfall matches, rather than an empty list which reads as an
        error in the tool.
    """
    shown = result["shown"]
    total = result["total_available"]
    filters = ", ".join(result["filters_applied"]) or "none"
    lines = [f"# {solver} pitfalls — {shown} of {total} entries",
             f"filters: {filters}"]

    if shown < total:
        lines.append(
            f"{total - shown} entries are not shown. For the complete set: "
            f"knowledge(topic='pitfalls', solver='{solver}')")

    modes = result.get("match_modes") or {}
    weak_only = modes and set(modes) <= {"some_tokens"}
    if weak_only:
        lines.append(
            "NOTE: every match below is a partial word-overlap, not a match on "
            "the recorded symptom text. Treat these as leads to read, not as an "
            "identification of your failure.")

    if not shown:
        if "signal" in filters:
            lines.append(
                "\nNo recorded pitfall matches that symptom. That is "
                "informative but not conclusive: it means this failure mode has "
                "not been catalogued for this backend, NOT that your setup is "
                "correct. Verify against the residual/attestation checks rather "
                "than reading silence as approval.")
        else:
            lines.append("\nNo entries match those filters.")
        return "\n".join(lines)

    by_physics: dict[str, list[dict[str, Any]]] = {}
    for e in result["entries"]:
        by_physics.setdefault(e["physics"], []).append(e)

    for physics, group in by_physics.items():
        lines.append(f"\n## {physics}")
        for e in group:
            tag = f"[{e['category']}]" if e["category"] else ""
            mark = ""
            if e.get("match"):
                label = {"substring": "matches recorded symptom",
                         "substring_body": "matches entry text",
                         "all_tokens": "all query terms present",
                         "some_tokens": "WEAK: partial term overlap"}
                mark = f"  <- {label.get(e['match'], e['match'])}"
            lines.append(f"- {tag} {e['advice']}{mark}")
    return "\n".join(lines)


def index_summary(all_pitfalls: dict[str, Any], solver: str) -> str:
    """A map of what exists, so an agent can narrow before it pulls.

    Cheap to read and enough to choose a filter: counts per physics and per
    category, with the exact calls that fetch each slice.
    """
    entries = flatten(all_pitfalls)
    by_p: dict[str, int] = {}
    by_c: dict[str, int] = {}
    for e in entries:
        by_p[e["physics"]] = by_p.get(e["physics"], 0) + 1
        if e["category"]:
            by_c[e["category"]] = by_c.get(e["category"], 0) + 1

    lines = [f"# {solver}: {len(entries)} pitfall entries",
             "",
             "Narrow before you pull. Three axes, any combination:",
             f"  by symptom   knowledge(topic='pitfalls', solver='{solver}', "
             f"signal='<paste the error text>')",
             f"  by physics   knowledge(topic='pitfalls', solver='{solver}', "
             f"physics='<name below>')",
             f"  by category  knowledge(topic='pitfalls', solver='{solver}', "
             f"category='<name below>')",
             "",
             "## physics"]
    for p, n in sorted(by_p.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {n:>4}  {p}")
    lines.append("\n## category")
    for c, n in sorted(by_c.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {n:>4}  {c}")
    lines.append(f"\nEverything at once: knowledge(topic='pitfalls', "
                 f"solver='{solver}') — {len(entries)} entries.")
    return "\n".join(lines)
