"""Providers: the upstream frameworks that produce cards and run the calculations a rule still requires.

openPASO does not depend on any provider. A provider is a separate package that declares an entry point:

    # in the provider's pyproject.toml
    [project.entry-points."openpaso.material_providers"]
    genius = "genius_openpaso:Provider"

and implements two calls:

    produce(request: dict) -> dict        # a new card (MaterialCard.to_dict() layout), e.g. from
                                          # {"material": "Al", "kind": "elastic.cubic", "budget_core_hours": 24}
    complete(card: dict, plan: list) -> dict
                                          # runs the planned calculations at the given settings and returns the
                                          # card with them added as evidence (variant_of / variant filled in)

GENIUS (an agentic framework for Quantum ESPRESSO workflows) is the first intended provider; its adapter lives
with GENIUS, not here. The split keeps the decision on the openPASO side: the provider computes, the rule decides
what has to be computed and whether the result may pass.
"""
from __future__ import annotations

from importlib.metadata import entry_points
from typing import Protocol

GROUP = "openpaso.material_providers"


class MaterialProvider(Protocol):
    def produce(self, request: dict) -> dict: ...

    def complete(self, card: dict, plan: list) -> dict: ...


def available() -> dict:
    """name -> provider instance, or name -> 'load error: ...' (a broken provider must not hide the others)."""
    found = {}
    for ep in entry_points(group=GROUP):
        try:
            obj = ep.load()
            found[ep.name] = obj() if isinstance(obj, type) else obj
        except Exception as e:  # noqa: BLE001  reported, not raised
            found[ep.name] = f"load error: {type(e).__name__}: {e}"
    return found
