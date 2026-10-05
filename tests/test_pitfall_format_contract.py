"""The two fields retrieval depends on must be present on every entry.

WHY A CONTRACT AND NOT A STYLE GUIDE
------------------------------------
Knowledge is being added continuously and by many hands at once. A convention
that lives in a reviewer's memory decays under that load — not through
carelessness but because each writer sees only their own entries and the format
is invisible until something tries to retrieve across all of them.

Measured over the 1122 entries present when this was written, the convention was
already near-universal, which is why it is worth locking rather than redesigning:

    [Category] <the fact, and what to do about it, in prose>.
    Signal: <what you observe when it bites>. (Verified <date>, <version>)

    Signal: clause present   1122/1122  (100%)
    [Category] tag present   1121/1122  (100%)
    exact duplicate entries  0
    median entry length      398 chars

Retrieval is built on exactly those two fields — `signal=` matches the Signal
clause, `category=` filters the tag. An entry missing either is not a smaller
contribution; it is invisible to the query that would have surfaced it. So this
gate protects retrieval, not tidiness.

WHAT IS DELIBERATELY NOT ENFORCED
---------------------------------
No `Fix:` marker. The advice lives in prose on purpose: an explanation a weak
model can reason from is worth more than a terse field it can only copy, and
prose is what 1120 of the entries already are. Cutting them into fields would
be a large mechanical edit across files that concurrent work is actively
changing, for no retrieval gain.

Length is not capped either. The longest entries earn it — Kratos contact
integration needs every one of its 817 characters — and a cap would push
authors to drop the reasoning rather than tighten it.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from core.pitfall_index import CANONICAL_CATEGORIES, parse_entry  # noqa: E402

BACKENDS_DIR = REPO / "src" / "backends"
# Compound tags are legitimate: `_cross.py` writes [Cross-Backend][Units] and
# the axis is the LAST tag. The gate must read the convention the same way the
# retrieval layer does, or it fails entries that retrieval handles correctly.
_CATEGORY_RE = re.compile(r"^\s*(?:\[[^\]]{1,40}\])+")


def _entries() -> list[tuple[Path, str]]:
    """Every pitfall entry in the tree, with the file it came from.

    Deduplicated by (file, value): a sub-dict attached by reference to several
    physics keys is one entry to maintain, and counting it repeatedly once
    inflated a DUNE audit's total from 111 to 127.

    DOCSTRINGS ARE NOT ENTRIES. A pitfall entry is a data value — an item in a
    `pitfalls` list, a value in a knowledge dict. A module/class/function
    docstring is documentation, and several of them legitimately explain the
    convention using the literal token `Signal:` while describing it. Collecting
    those made the gate demand a [Category] tag on prose that must not carry
    one, and the only way to satisfy it was to reword correct documentation.
    Measured when this exclusion was added during the branch consolidation:
    1299 strings under src/backends contain `Signal:`, exactly 4 are docstrings,
    and none of the 4 is a pitfall — so this drops false positives and nothing
    else. (Found three separate times: _setup.py, dune/generators/__init__.py,
    and four SPARTA modules.)
    """
    seen: set[tuple[str, str]] = set()
    out: list[tuple[Path, str]] = []
    for py in sorted(BACKENDS_DIR.rglob("*.py")):
        try:
            tree = ast.parse(py.read_text(errors="ignore"))
        except SyntaxError:
            continue
        docstrings = set()
        for holder in ast.walk(tree):
            if isinstance(holder, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                   ast.AsyncFunctionDef)):
                body = getattr(holder, "body", None)
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    docstrings.add(id(body[0].value))
        # A REGEX PATTERN IS NOT AN ENTRY, AND NEITHER IS A WHOLE DOCUMENT.
        # Same reasoning as the docstring exclusion above: the gate demands a
        # [Category] tag on a pitfall entry, and the only way to satisfy it on
        # something that is not one is to damage correct code.
        #   * `re.search(r"Signal:\s*Segmentation fault|...", blob)` in
        #     fourc/backend.py is a pattern that MATCHES 4C's output. Tagging it
        #     would change what it matches.
        #   * FOURC_DECK_GRAMMAR in fourc/deck_grammar.py is the whole deck
        #     grammar document, which mentions the Signal: convention while
        #     explaining it. It is one constant, not one entry.
        patterns = set()
        for call in ast.walk(tree):
            if (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "re"):
                for arg in call.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        patterns.add(id(arg))
        documents = set()
        for assign in ast.walk(tree):
            if not isinstance(assign, ast.Assign):
                continue
            names = {tgt.id for tgt in assign.targets if isinstance(tgt, ast.Name)}
            if any(n.endswith(("_GRAMMAR", "_DOC", "_DOCUMENT", "_MANUAL"))
                   for n in names):
                if isinstance(assign.value, ast.Constant):
                    documents.add(id(assign.value))

        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and "Signal:" in node.value
                    and id(node) not in docstrings
                    and id(node) not in patterns
                    and id(node) not in documents):
                k = (str(py), node.value)
                if k not in seen:
                    seen.add(k)
                    out.append((py, node.value))
    return out


ENTRIES = _entries()


def test_there_are_entries_to_check():
    """Guard against the collector silently finding nothing.

    A format gate that examines zero entries passes perfectly and means
    nothing. This project has already shipped one green suite that was
    asserting against an empty set.
    """
    assert len(ENTRIES) > 900, (
        f"expected the full pitfall corpus, collected {len(ENTRIES)}. Either "
        f"the collector broke or the corpus shrank — both need looking at "
        f"before any format claim below can be believed.")


def test_every_entry_has_a_signal_clause():
    """Without it the entry cannot be found by the error the agent is holding."""
    bad = [(p.relative_to(REPO), t[:90]) for p, t in ENTRIES
           if not parse_entry(t)["signal"]]
    assert not bad, (
        f"{len(bad)} entries have no usable Signal: clause, so "
        f"knowledge(topic='pitfalls', signal=...) can never surface them:\n  "
        + "\n  ".join(f"{p}: {t}" for p, t in bad[:12]))


def test_every_entry_has_a_category_tag():
    """Without it the entry is invisible to category filtering."""
    bad = [(p.relative_to(REPO), t[:90]) for p, t in ENTRIES
           if not _CATEGORY_RE.match(t)]
    assert not bad, (
        f"{len(bad)} entries do not open with a [Category] tag:\n  "
        + "\n  ".join(f"{p}: {t}" for p, t in bad[:12])
        + "\n\nStart the entry with one of: "
        + ", ".join(sorted(set(CANONICAL_CATEGORIES.values()))))


def test_categories_come_from_the_known_vocabulary():
    """A new category name silently partitions the corpus.

    An agent filtering on `Numerical` will not see an entry tagged with a
    freshly-invented synonym, and nothing reports the miss. Spelling variants
    are folded on read (`Numerics` -> `Numerical`); this catches genuinely new
    values, which need a deliberate decision rather than arriving by accident.
    """
    unknown: dict[str, list[str]] = {}
    for p, t in ENTRIES:
        if not _CATEGORY_RE.match(t):
            continue
        raw = parse_entry(t)["category_as_written"]
        if raw.lower() not in CANONICAL_CATEGORIES:
            unknown.setdefault(raw, []).append(
                f"{p.relative_to(REPO)}: {t[:70]}")
    assert not unknown, (
        "category tags outside the known vocabulary:\n  "
        + "\n  ".join(f"[{k}] x{len(v)} — e.g. {v[0]}"
                      for k, v in sorted(unknown.items()))
        + "\n\nEither use an existing category or add the new one to "
          "CANONICAL_CATEGORIES in src/core/pitfall_index.py, mapping it to "
          "the axis it belongs on so filtering stays complete.")


def test_signal_clause_is_not_empty_boilerplate():
    """`Signal:` followed by nothing useful is the same as no Signal at all.

    Caught in review more than once: an entry whose Signal read only "see
    above" or "error message" — present, therefore passing a presence check,
    and matching nothing an agent could ever paste.
    """
    vague = {"", "see above", "error", "error message", "n/a", "none",
             "as described", "tbd", "unknown", "varies", "see description"}
    bad = []
    for p, t in ENTRIES:
        sig = parse_entry(t)["signal"].strip().rstrip(".").lower()
        if sig in vague or len(sig) < 15:
            bad.append(f"{p.relative_to(REPO)}: Signal: {sig!r}")
    assert not bad, (
        f"{len(bad)} entries have a Signal: clause too vague to match against:"
        f"\n  " + "\n  ".join(bad[:12])
        + "\n\nQuote what the software actually prints, or describe the "
          "observable numerically. A placeholder passes a presence check while "
          "making the entry unfindable.")


def test_entries_do_not_duplicate_each_other_verbatim():
    """Two identical entries mean one of them is unmaintained.

    Zero at the time of writing. Worth keeping there: when the same text sits in
    two places, a correction lands on one and the other keeps being served.
    """
    from collections import Counter
    c = Counter(t for _, t in ENTRIES)
    dupes = {t: n for t, n in c.items() if n > 1}
    assert not dupes, (
        f"{len(dupes)} entries appear verbatim in more than one place:\n  "
        + "\n  ".join(f"x{n}: {t[:80]}" for t, n in list(dupes.items())[:8])
        + "\n\nKeep one copy and reference it, so a fix cannot land on only "
          "one of them.")
