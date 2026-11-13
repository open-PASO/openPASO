"""Rules: per kind of handoff data, which calculations are required and how they are compared.

A rule module provides ``KIND``, ``DESCRIPTION`` and ``ledger(card) -> Ledger``. To support new data (another
crystal symmetry, thermal conductivity, a plasticity model), add a module here and register its kind below; the
ledger, the plan and the MCP tool need no change.
"""
from __future__ import annotations

from scale_bridge.card import MaterialCard
from scale_bridge.rules import pw_dft_elastic

RULES = {pw_dft_elastic.KIND: pw_dft_elastic}


def rule_for(kind: str):
    if kind not in RULES:
        raise KeyError(f"no rule for kind '{kind}'; known kinds: {', '.join(sorted(RULES))}")
    return RULES[kind]


def check(card: MaterialCard) -> dict:
    rule = rule_for(card.kind)
    report = rule.ledger(card).report()
    report.update(kind=card.kind, rule=rule.DESCRIPTION, card_digest=card.digest(),
                  derived_values=rule.derived_values(card))
    return report


def plan(card: MaterialCard) -> list[dict]:
    return rule_for(card.kind).ledger(card).plan()
