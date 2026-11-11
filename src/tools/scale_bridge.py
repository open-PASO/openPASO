"""MCP tool for material data handed over from another framework: `material_handoff`.

A continuum run that uses constants from an upstream calculation inherits that calculation's errors. The tool lets
the agent (and the server) state what the upstream result rests on, derive from the producer's own settings which
calculations a check requires, see what is missing, have a provider run it, and only then turn the data into the
stiffness a continuum model uses. Logic lives in src/scale_bridge; this module only formats it for the protocol.
"""
from __future__ import annotations

import json
import re

from core.output_paths import output_dir
from scale_bridge import card as card_module
from scale_bridge import providers
from scale_bridge.card import MaterialCard
from scale_bridge.continuum import for_continuum
from scale_bridge.rules import RULES, check

ACTIONS = ("schema", "check", "plan", "continuum", "providers", "produce", "complete")


def _schema() -> str:
    kinds = "\n".join(f"- {k}: {m.DESCRIPTION}" for k, m in RULES.items())
    return f"{card_module.__doc__}\nKnown kinds and their rules:\n{kinds}\n"


def _label(text: str) -> str:
    """A provider's label as one safe file-name part: no separators, no leading dots."""
    return re.sub(r"[^A-Za-z0-9._-]+|\.{2,}", "_", str(text)).strip("._")[:40] or "card"


def _save(card: MaterialCard) -> str:
    d = output_dir("simulation_outputs") / "material_cards"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{_label(card.material)}_{_label(card.kind)}_{card.digest()[:10]}.json"
    p.write_text(json.dumps(card.to_dict(), indent=1), encoding="utf-8")
    return str(p)


def material_handoff_body(action: str, card: str = "", theta_deg: float = 0.0, provider: str = "",
                          request: str = "") -> str:
    if action not in ACTIONS:
        return f"Unknown action '{action}'. Actions: {', '.join(ACTIONS)}."
    if action == "schema":
        return _schema()
    if action == "providers":
        found = providers.available()
        if not found:
            return ("No material provider is installed. A provider is a separate package that registers an entry "
                    f"point in the group '{providers.GROUP}' (see scale_bridge/providers.py).")
        return json.dumps({k: (v if isinstance(v, str) else type(v).__name__) for k, v in found.items()}, indent=1)
    if action in ("produce", "complete"):
        prov = providers.available().get(provider)
        if prov is None or isinstance(prov, str):
            return f"Provider '{provider}' is not available: {prov or 'not installed'}. Call action='providers'."
    if action == "produce":
        try:
            new = MaterialCard.from_dict(prov.produce(json.loads(request or "{}")))
        except Exception as e:  # noqa: BLE001  the provider's failure is the answer
            return f"Provider '{provider}' failed to produce a card: {type(e).__name__}: {e}"
        return json.dumps({"card_path": _save(new), "check": check(new)}, indent=1, default=str)
    try:
        c = MaterialCard.load(card)
    except Exception as e:  # noqa: BLE001
        return f"Cannot read the card ({type(e).__name__}: {e}). Pass a card file path or the card's JSON text."
    report = check(c)
    if action == "check":
        return json.dumps(report, indent=1, default=str)
    if action == "plan":
        return json.dumps({"decision": report["decision"], "reason": report["reason"], "plan": report["plan"],
                           "reservation_core_hours": report["reservation_core_hours"]}, indent=1, default=str)
    if action == "continuum":
        if report["decision"] != "accept":
            return json.dumps({"passed": False, "decision": report["decision"], "reason": report["reason"],
                               "plan": report["plan"]}, indent=1, default=str)
        return json.dumps({"passed": True, "card_digest": report["card_digest"], "material": c.material,
                           **for_continuum(c.values, theta_deg)}, indent=1)
    # complete: the provider runs exactly what the rule still requires, then the card is checked again
    if report["decision"] != "incomplete":
        return json.dumps({"nothing_to_run": True, "decision": report["decision"], "reason": report["reason"]},
                          indent=1, default=str)
    try:
        new = MaterialCard.from_dict(prov.complete(c.to_dict(), report["plan"]))
    except Exception as e:  # noqa: BLE001
        return f"Provider '{provider}' failed to complete the card: {type(e).__name__}: {e}"
    return json.dumps({"card_path": _save(new), "check": check(new)}, indent=1, default=str)


def register_scale_bridge_tools(mcp):

    @mcp.tool()
    def material_handoff(action: str = "schema", card: str = "", theta_deg: float = 0.0, provider: str = "",
                         request: str = "") -> str:
        """Use material data from another framework (e.g. elastic constants from quantum-mechanical calculations)
        in a continuum model, but only once the calculations it rests on are complete.

        The card holds the values and every calculation behind them (call action='schema' for the layout). From
        the producer's OWN settings the rule derives which convergence calculations are required; a missing one
        counts as failed, never as passed.

        Args:
            action: 'schema' (card layout and rules), 'check' (accept / reject / incomplete, with the measured
                changes), 'plan' (the calculations still to run, with their settings and estimated core-hours),
                'continuum' (stiffness for the continuum model; refused unless the check accepts),
                'providers' (installed upstream frameworks), 'produce' (ask a provider for a new card),
                'complete' (ask a provider to run the planned calculations, then check again).
            card: path to a card file, or the card's JSON text (check, plan, continuum, complete).
            theta_deg: rotation of the crystal axes about z in the sheet plane (continuum).
            provider: provider name (produce, complete).
            request: JSON request for the provider (produce), e.g. {"material": "Al", "kind": "elastic.cubic"}.
        """
        return material_handoff_body(action, card, theta_deg, provider, request)
