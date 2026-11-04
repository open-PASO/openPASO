"""One list of required calculations decides acceptance, the plan of what is still missing, and its cost.

A rule turns the producer's settings into comparisons; each comparison requires named calculations and compares
named quantities against limits. The decision:

- reject: a required calculation failed (rerunning it gives the same end), a compared quantity is not finite, or a
  quantity exceeds its limit;
- incomplete: a required calculation is missing or was stopped from outside; the plan lists it with its settings;
- accept: every required calculation is complete and every quantity is within its limit.

A missing calculation never passes: "incomplete" is not "accept".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

OPEN, COMPLETE, IMPOSSIBLE = "open", "complete", "impossible"


@dataclass
class Obligation:
    name: str                      # the calculation it repeats (or a label for a structural requirement)
    settings: dict                 # the settings the required calculation must use
    estimate_core_hours: float = 0.0
    status: str = OPEN
    reason: str = ""
    satisfied_by: str = ""         # the card calculation that meets it


@dataclass
class Comparison:
    name: str
    description: str
    limits: dict                   # quantity -> limit
    obligations: list
    differences: dict = field(default_factory=dict)

    def state(self) -> str:
        if any(o.status == IMPOSSIBLE for o in self.obligations):
            return IMPOSSIBLE
        if any(o.status == OPEN for o in self.obligations) or set(self.differences) != set(self.limits):
            return OPEN
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in self.differences.values()):
            return IMPOSSIBLE
        return COMPLETE

    def exceeded(self) -> list[str]:
        if self.state() != COMPLETE:
            return []
        return [q for q, lim in self.limits.items() if self.differences[q] > lim]


class Ledger:
    def __init__(self, comparisons: list):
        self.comparisons = {c.name: c for c in comparisons}

    def decision(self) -> tuple[str, str]:
        states = {n: c.state() for n, c in self.comparisons.items()}
        impossible = [f"{n}/{o.name} ({o.reason})" for n, c in self.comparisons.items() for o in c.obligations
                      if o.status == IMPOSSIBLE]
        impossible += [f"{n}: a compared quantity is not finite" for n, s in states.items()
                       if s == IMPOSSIBLE and all(o.status != IMPOSSIBLE for o in self.comparisons[n].obligations)]
        if impossible:
            return "reject", "required evidence cannot be obtained: " + "; ".join(impossible)
        exceeded = [f"{n}:{q}" for n, c in self.comparisons.items() for q in c.exceeded()]
        if exceeded:
            return "reject", "limits exceeded: " + ", ".join(exceeded)
        open_ = [n for n, s in states.items() if s == OPEN]
        if open_:
            return "incomplete", "required calculations missing or stopped in: " + ", ".join(open_)
        return "accept", "every required calculation is complete and every compared quantity is within its limit"

    def plan(self) -> list[dict]:
        """The calculations still to run, with the settings each must use."""
        return [{"comparison": n, "repeats": o.name, "settings": o.settings, "status": o.status, "reason": o.reason,
                 "estimate_core_hours": round(o.estimate_core_hours, 2)}
                for n, c in self.comparisons.items() for o in c.obligations if o.status == OPEN]

    def reservation_core_hours(self) -> float:
        return sum(o.estimate_core_hours for c in self.comparisons.values() for o in c.obligations
                   if o.status == OPEN) if self.decision()[0] == "incomplete" else 0.0

    def report(self) -> dict:
        decision, reason = self.decision()
        return {"decision": decision, "reason": reason,
                "comparisons": {n: {"description": c.description, "state": c.state(), "limits": c.limits,
                                    "differences": c.differences,
                                    "calculations": {o.name: {"status": o.status, "reason": o.reason,
                                                              "satisfied_by": o.satisfied_by}
                                                     for o in c.obligations}}
                                for n, c in self.comparisons.items()},
                "plan": self.plan(), "reservation_core_hours": round(self.reservation_core_hours(), 2)}
