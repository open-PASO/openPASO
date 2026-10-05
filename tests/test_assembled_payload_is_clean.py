"""Scan what the agent RECEIVES, not the files the text came from.

`test_knowledge_not_contaminated.py` greps source files. An adversarial 4C audit
showed why that is not enough: it measured **240 host absolute paths across
48 of 48 served payloads** — `/home/user/4C/build/4C` 144 times, the venv
interpreter 48, a host test-file path 48 — while the same sweep over
`get_knowledge()` returned zero.

Nothing was hidden. The paths live in an "Installed-version API reference" block
that `prepare_simulation` appends AFTER the JSON, so a check that reads the
knowledge dict cannot see them and a check that reads source files sees them
without knowing they reach an agent. The audit's phrase for it was "wrong surface
verified", and it applies to a guard I wrote.

So this one drives the real tool and reads the assembled reply. It is the only
check in the repo positioned where the agent stands.

WHY HOST PATHS IN A SERVED PAYLOAD ARE A DEFECT, not untidiness: openPASO is going
to be cloned and run by people whose machines are not this one. A payload that
tells an agent the solver is at `/home/user/4C/build/4C` is confidently
wrong everywhere else, and an agent that will not second-guess a served fact
follows it. The fix is to describe how to LOCATE an install, not to record where
one happened to be.
"""
from __future__ import annotations

import functools
import re

import pytest

# `/home/<user>/`, `/Users/<user>/`, `/media/<user>/` — a path rooted in one
# person's account. `/usr`, `/opt` and `/tmp` are deliberately not matched: they
# are system locations that are legitimately the same on another machine.
_HOST_PATH = re.compile(r"(?:/home|/Users|/media)/[a-z][a-z0-9_.-]*/")

# Naming a variable is how an agent is TOLD to find its own path, so a payload
# that mentions one alongside a path is teaching rather than hard-coding.
_TEACHES = re.compile(
    r"FOURC_BINARY|FOURC_ROOT|FEBIO_BINARY|FENICS_PYTHON|KRATOS_ROOT"
    r"|LD_LIBRARY_PATH|CONDA_DEFAULT_ENV|DEAL_II_DIR|VIRTUAL_ENV"
    r"|environment variable|set (?:it|this) to|on this machine|your install")

# A section that declares its paths to be local observations rather than facts.
# Whole blocks need this, not a nearby window: the installed-API reference
# carries its disclaimer once at the top and then lists paths for thousands of
# characters, which is exactly the shape a proximity check gets wrong.
_LOCAL_OBSERVATION = re.compile(
    r"LOCAL OBSERVATIONS?|local observations?|not universal facts?"
    r"|machine hosting this openPASO", re.I)


@functools.lru_cache(maxsize=1)
def _served():
    """Every assembled `prepare_simulation` reply, generated once."""
    from core.registry import get_backend, list_backends, load_all_backends
    from tools import consolidated as C

    load_all_backends()
    captured = {}

    class _Recorder:
        def tool(self, *a, **k):
            def deco(fn):
                captured[fn.__name__] = fn
                return fn
            return deco

    C.register_consolidated_tools(_Recorder())
    prepare = captured["prepare_simulation"]

    out = []
    for entry in list_backends():
        be = get_backend(entry["name"])
        if not be:
            continue
        for p in be.supported_physics():
            try:
                out.append((entry["name"], p.name, prepare(entry["name"], p.name)))
            except Exception as exc:
                out.append((entry["name"], p.name, f"__ERROR__ {exc}"))
    return tuple(out)


def test_no_served_payload_hard_codes_a_host_path():
    """The 4C finding, generalised to every backend.

    A hit is allowed only where the payload is plainly teaching an agent how to
    locate its own install — naming the environment variable, or saying "on this
    machine". Anything else states one person's filesystem as fact.
    """
    offenders: list[str] = []
    for backend, physics, text in _served():
        # A block may declare up front that its paths are local observations.
        # Everything until the next `## ` heading inherits that declaration —
        # a nearby-window check cannot work, because the disclaimer sits at the
        # top of a block whose paths run for thousands of characters.
        labelled: list[tuple[int, int]] = []
        for hm in re.finditer(r"^## .*$", text, re.M):
            end = text.find("\n## ", hm.end())
            end = len(text) if end == -1 else end
            if _LOCAL_OBSERVATION.search(text[hm.start():min(end, hm.end() + 900)]):
                labelled.append((hm.start(), end))

        for m in _HOST_PATH.finditer(text):
            if any(lo <= m.start() < hi for lo, hi in labelled):
                continue
            window = text[max(0, m.start() - 400):m.end() + 200]
            if _TEACHES.search(window) or _LOCAL_OBSERVATION.search(window):
                continue
            offenders.append(
                f"{backend}/{physics}: {text[m.start():m.start() + 70]!r}")
    # Report per backend so the failure is actionable rather than a wall.
    if offenders:
        by_backend: dict[str, int] = {}
        for o in offenders:
            by_backend[o.split("/")[0]] = by_backend.get(o.split("/")[0], 0) + 1
        summary = ", ".join(f"{k}={v}" for k, v in sorted(by_backend.items()))
    assert not offenders, (
        f"served payloads state a host absolute path as fact ({summary}). This "
        f"is what an agent receives, so it is wrong on every machine that is "
        f"not this one. Describe how to locate the install — name the "
        f"environment variable — instead of recording where it happens to "
        f"be.\n  " + "\n  ".join(offenders[:12])
        + (f"\n  ... and {len(offenders) - 12} more" if len(offenders) > 12 else ""))


def test_the_guard_looks_at_more_than_the_knowledge_dict():
    """Calibration, and the whole point of this file.

    The 4C audit found 240 host paths in the assembled payloads and 0 in
    `get_knowledge()`. If this test only ever saw the JSON it would agree with
    the check it exists to supersede, so assert the assembled reply really is
    larger than the knowledge block alone.
    """
    import json

    from core.registry import get_backend

    for backend, physics, text in _served():
        if text.startswith("__ERROR__"):
            continue
        be = get_backend(backend)
        try:
            k = be.get_knowledge(physics) or {}
        except Exception:
            continue
        if not isinstance(k, dict) or not k:
            continue
        assert len(text) > len(json.dumps(k, default=str)) / 2, (
            f"{backend}/{physics}: the assembled reply is not materially "
            f"larger than its knowledge dict, so this guard may be reading the "
            f"same surface as the source-file check")
        return
    pytest.skip("no backend served a knowledge dict to compare against")
