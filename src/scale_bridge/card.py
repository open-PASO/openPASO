"""The handoff card: material values with the calculations they rest on.

A card is plain JSON so that any producer can write it and any consumer can read it:

    {
      "material": "Al",
      "kind": "elastic.cubic",
      "values": {"C11": 112.1, "C12": 61.1, "C44": 33.1},
      "units": "GPa",
      "producer": {"name": "GENIUS", "version": "...", "program": "Quantum ESPRESSO 7.6"},
      "evidence": [
        {"name": "ref", "role": "reference", "status": "complete",
         "settings": {"ecutwfc_Ry": 40, "ecutrho_Ry": 320, "kmesh": [48, 48, 48], "smearing_width_Ry": 0.02},
         "strain": [[0,0,0],[0,0,0],[0,0,0]], "stress_GPa": [[...],[...],[...]], "core_hours": 1.2},
        {"name": "np005", "role": "normal", ...},
        {"name": "ref_cut", "role": "reference", "variant_of": "ref", "variant": "cutoff_k", ...}
      ]
    }

Stress is the Cauchy stress in GPa, tensile positive; strain is the small-strain tensor applied to the reference
cell. Status is ``complete`` (finished and converged), ``stopped`` (ended from outside, e.g. a time limit: it can be
run again) or ``failed`` (the program ended without success: running it again gives the same end). A calculation
that is not in the card is missing.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

STATUSES = ("complete", "stopped", "failed")


@dataclass
class Calculation:
    name: str
    role: str
    settings: dict
    strain: list | None = None
    stress_GPa: list | None = None
    status: str = "complete"
    variant_of: str = ""
    variant: str = ""
    core_hours: float | None = None
    note: str = ""


@dataclass
class MaterialCard:
    material: str
    kind: str
    values: dict
    units: str = "GPa"
    producer: dict = field(default_factory=dict)
    evidence: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "MaterialCard":
        known = set(Calculation.__dataclass_fields__)
        ev = [Calculation(**{k: v for k, v in c.items() if k in known}) for c in d.get("evidence", [])]
        return cls(material=d["material"], kind=d["kind"], values=dict(d["values"]), units=d.get("units", "GPa"),
                   producer=dict(d.get("producer", {})), evidence=ev)

    @classmethod
    def load(cls, source: str) -> "MaterialCard":
        """A path to a card file, or the card's JSON text."""
        text = source.strip()
        if not text.startswith("{"):
            text = Path(text).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))

    def to_dict(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def by_name(self) -> dict:
        return {c.name: c for c in self.evidence}

    def problems(self) -> list[str]:
        """Structural defects that make the card unreadable for any rule (not physics checks)."""
        out = []
        names = [c.name for c in self.evidence]
        if len(names) != len(set(names)):
            out.append("calculation names are not unique")
        for k, v in self.values.items():
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                out.append(f"value {k} is not a finite number")
        for c in self.evidence:
            if c.status not in STATUSES:
                out.append(f"{c.name}: status '{c.status}' is not one of {', '.join(STATUSES)}")
            if c.variant_of and c.variant_of not in names:
                out.append(f"{c.name}: variant_of '{c.variant_of}' is not in the card")
            for label, m in (("strain", c.strain), ("stress_GPa", c.stress_GPa)):
                if m is not None and (len(m) != 3 or any(len(row) != 3 for row in m)):
                    out.append(f"{c.name}: {label} is not 3x3")
        return out
