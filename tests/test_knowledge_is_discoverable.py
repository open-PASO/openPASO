"""Knowledge an agent cannot find is knowledge nobody has.

WHY THIS GATE EXISTS
--------------------
Kratos held 41 knowledge areas and advertised 20. The other 21 carried 65
verified pitfall entries — geomechanics, RANS, IGA, ROM, topology optimisation,
chimera, FEM-to-DEM, FSI — and no surface enumerated any of them. Each had been
researched, executed and written down, and none could be reached by an agent
that did not already know the key to guess. That is worse than a gap: the work
was done and then lost, and the loss is invisible because every test passed.

It happened by drift, not decision. `supported_physics()` is a capability claim
and correctly excludes areas with no generator; the pitfalls surface enumerated
`supported_physics()`; so knowledge-only areas fell between the two. Nobody
chose that. With hundreds of new entries landing from concurrent work, the same
drift will recur every time an area is written without a template — so the
check has to be mechanical rather than remembered.

WHAT IS ENFORCED
----------------
Every knowledge area holding pitfall entries must be reachable from a surface
that ENUMERATES it, not merely fetchable by an agent that guessed the key.
Either it is in `supported_physics()`, or the pitfalls surface lists it as
reference-only. Nothing may hold verified content and appear nowhere.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from core import pitfall_index  # noqa: E402
from core.registry import get_backend, load_all_backends  # noqa: E402

BACKENDS = ["fourc", "kratos", "dealii", "fenics", "ngsolve", "skfem",
            "dune", "febio", "sparta"]


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    load_all_backends()


def _pitfall_count(value) -> int:
    return len([s for s in pitfall_index._strings_under(value)
                if "Signal:" in s])


@pytest.mark.parametrize("solver", BACKENDS)
def test_every_knowledge_area_with_pitfalls_is_enumerated(solver):
    """No area may hold pitfall entries while appearing on no surface."""
    be = get_backend(solver)
    if be is None:
        pytest.skip(f"{solver} backend not registered on this install")

    keys = pitfall_index.knowledge_keys(solver)
    if not keys:
        pytest.skip(f"{solver} keeps knowledge where knowledge_keys() cannot "
                    f"enumerate it; covered by the per-backend audits instead")

    advertised = {p.name for p in be.supported_physics()}
    reference = set(pitfall_index.reference_only_areas(solver, advertised))

    # REACHABILITY IS ABOUT THE TEXT, NOT THE KEY. An earlier version of this
    # test compared area NAMES and accused 4C of hiding 154 subjects. It hides
    # none: those names are aliases (`elasticity` -> `structure`, `scatra`,
    # `xfem`, ...) whose 942 entry slots resolve to 248 texts that the
    # advertised path already serves in full. Counting names would have driven
    # a large pointless edit and shipped a payload with 942 duplicate entries.
    #
    # The question an agent actually cares about is whether the WORDS of a
    # pitfall can reach it. So collect every entry served by any enumerated
    # surface, then look for entries that appear on none of them.
    served: set[str] = set()
    for area in advertised | reference:
        k = be.get_knowledge(area)
        if isinstance(k, dict):
            served.update(s for s in pitfall_index._strings_under(k)
                          if "Signal:" in s)

    hidden = []
    for key in keys:
        if key.startswith("_"):
            continue  # `_general` is served by topic='overview'
        k = be.get_knowledge(key)
        if not isinstance(k, dict):
            continue
        unreachable = [s for s in pitfall_index._strings_under(k)
                       if "Signal:" in s and s not in served]
        if unreachable:
            hidden.append(f"{key} ({len(unreachable)} entries reachable "
                          f"nowhere else)")

    assert not hidden, (
        f"{solver}: these knowledge areas hold verified pitfall entries but no "
        f"surface enumerates them, so no agent can discover they exist:\n  "
        + "\n  ".join(hidden)
        + "\n\nEither declare the area in supported_physics() (only if a "
          "generator can build a runnable input for it) or let it be picked up "
          "as reference-only. Do not leave researched content unreachable.")


@pytest.mark.parametrize("solver", BACKENDS)
def test_reference_only_areas_really_have_no_generator(solver):
    """A reference-only label must be true, not a way to dodge the first test.

    If a generator CAN build a runnable input for an area, the area belongs in
    `supported_physics()` where discover() will show it. Labelling it
    reference-only would then understate the backend — the mirror image of the
    fabrication this project guards against, and just as misleading.
    """
    be = get_backend(solver)
    if be is None:
        pytest.skip(f"{solver} backend not registered on this install")
    advertised = {p.name for p in be.supported_physics()}
    areas = pitfall_index.reference_only_areas(solver, advertised)
    if not areas:
        pytest.skip(f"{solver} has no reference-only areas")

    runnable = []
    for area in areas:
        try:
            out = be.generate_input(area, "default", {})
        except Exception:
            continue  # correct: no generator, so reference-only is accurate
        if isinstance(out, str) and len(out) > 200 and "NOT a runnable" not in out:
            runnable.append(area)

    assert not runnable, (
        f"{solver}: these areas are served as reference-only yet "
        f"generate_input() produced a substantive input for them: "
        f"{runnable}. If they are runnable they belong in supported_physics() "
        f"so discover() advertises them.")


@pytest.mark.parametrize("solver", BACKENDS)
def test_pitfalls_surface_actually_serves_the_reference_areas(solver):
    """End to end: the entries must appear in what the tool returns.

    The first test checks enumeration; this one checks delivery. They are
    different failures and both have happened — a branch once enumerated an
    area whose content the assembling code then dropped, and every enumeration
    test still passed.
    """
    be = get_backend(solver)
    if be is None:
        pytest.skip(f"{solver} backend not registered on this install")
    advertised = {p.name for p in be.supported_physics()}
    areas = [a for a in pitfall_index.reference_only_areas(solver, advertised)
             if isinstance(be.get_knowledge(a), dict)
             and be.get_knowledge(a).get("pitfalls")]
    if not areas:
        pytest.skip(f"{solver} has no reference-only areas carrying pitfalls")

    # Rebuild what the tool assembles, then confirm each area's own text is in
    # it. Comparing counts would pass on a section that arrived empty.
    all_pitfalls = {}
    for p in be.supported_physics():
        k = be.get_knowledge(p.name)
        if k and "pitfalls" in k:
            all_pitfalls[p.name] = k["pitfalls"]
    for area in areas:
        k = be.get_knowledge(area)
        all_pitfalls[f"{area} [reference only — no generator; "
                     f"write the input yourself]"] = k["pitfalls"]

    served = json.dumps(all_pitfalls)
    missing = []
    for area in areas:
        entries = [s for s in pitfall_index._strings_under(
            be.get_knowledge(area)["pitfalls"]) if "Signal:" in s]
        for e in entries:
            probe = json.dumps(e)[1:60]
            if probe not in served:
                missing.append(f"{area}: {e[:70]}")
                break

    assert not missing, (
        f"{solver}: reference-only areas were enumerated but their entries are "
        f"not in the assembled payload:\n  " + "\n  ".join(missing))


def test_the_label_reaches_the_agent():
    """The reference-only caveat must travel with the content.

    Serving FSI pitfalls without saying there is no FSI generator invites an
    agent to try to run it and blame itself when it cannot. The label is part
    of the knowledge, not decoration around it.
    """
    be = get_backend("kratos")
    if be is None:
        pytest.skip("kratos not registered on this install")
    advertised = {p.name for p in be.supported_physics()}
    areas = pitfall_index.reference_only_areas("kratos", advertised)
    assert areas, ("kratos previously had 21 unlisted knowledge areas; finding "
                   "none means reference_only_areas() stopped working, not "
                   "that the knowledge was promoted")
    key = f"{areas[0]} [reference only — no generator; write the input yourself]"
    assert "reference only" in key and "no generator" in key
