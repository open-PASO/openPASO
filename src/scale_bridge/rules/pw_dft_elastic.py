"""Cubic elastic constants from plane-wave DFT stress-strain calculations (e.g. Quantum ESPRESSO).

Evidence the card must carry, at the settings of the reference calculation:
- the unstrained reference cell, an opposite pair of normal strains along one cube axis (e.g. e_xx = +e, -e) and an
  opposite pair of shear strains in one off-diagonal component (e.g. e_yz = e_zy = +g/2, -g/2), all with identical
  settings. Any axis and any shear plane will do: in a cubic crystal they are equivalent.

Required comparisons, derived from the reference calculation's OWN settings (not from what the card offers):
- 'cutoff_k': the five calculations repeated with the plane-wave cutoff x1.25 and the k-point mesh x1.3;
- 'smearing': repeated with half the smearing width and the k-point mesh x1.5, whenever the reference uses smearing.
A comparison passes when C' = (C11 - C12)/2 and C44 change by at most 2 % and the pressure by at most 0.1 GPa.
'reported': the values on the card must equal those from the reference-setting stresses within 1 %.

A repeated calculation counts only if it applies the same strain as the calculation it repeats, its settings are at
least as tight as required and every other setting is identical (an unsmeared reference is repeated without
smearing); one that ended without success makes its comparison impossible; a missing or stopped one leaves it open.
Values must be in GPa, the reference must be unstrained, and every strain must be a pure normal or a symmetric pure
shear strain.
"""
from __future__ import annotations

import math

import numpy as np

from scale_bridge.card import MaterialCard
from scale_bridge.obligations import COMPLETE, IMPOSSIBLE, OPEN, Comparison, Ledger, Obligation

KIND = "elastic.cubic"
DESCRIPTION = __doc__.split("\n")[0]
LIMITS = {"Cp": 0.02, "C44": 0.02, "P_GPa": 0.1}
REPORTED_LIMIT = 0.01
TIGHTENED = ("ecutwfc_Ry", "ecutrho_Ry", "kmesh", "smearing_width_Ry")
VARIANTS = {
    "cutoff_k": ("cutoff x1.25 and k-point mesh x1.3", {"ecut": 1.25, "k": 1.3, "width": 1.0}),
    "smearing": ("half the smearing width and k-point mesh x1.5", {"ecut": 1.0, "k": 1.5, "width": 0.5}),
}
ESTIMATE_FACTOR = {"cutoff_k": 2.5, "smearing": 3.4}  # run time relative to the calculation it repeats


def variant_settings(base: dict, variant: str) -> dict:
    f = VARIANTS[variant][1]
    s = dict(base)
    for key in ("ecutwfc_Ry", "ecutrho_Ry"):
        if key in base:
            s[key] = base[key] * f["ecut"]
    s["kmesh"] = [math.ceil(round(n * f["k"], 6)) for n in base["kmesh"]]
    if base.get("smearing_width_Ry"):
        s["smearing_width_Ry"] = base["smearing_width_Ry"] * f["width"]
    return s


def satisfies(actual: dict, required: dict) -> bool:
    eps = 1e-9
    for key in ("ecutwfc_Ry", "ecutrho_Ry"):
        if key in required and not (_positive(actual.get(key)) and actual[key] >= required[key] * (1 - eps)):
            return False
    k = actual.get("kmesh")
    if not _kmesh_ok(k) or any(a < r for a, r in zip(k, required["kmesh"])):
        return False
    w = actual.get("smearing_width_Ry") or 0
    if required.get("smearing_width_Ry"):
        if not (_positive(w) and w <= required["smearing_width_Ry"] * (1 + eps)):
            return False
    elif w:
        return False                                      # an unsmeared reference is repeated without smearing
    other = lambda d: {k: v for k, v in d.items() if k not in TIGHTENED}  # noqa: E731
    return other(actual) == other(required)


def _positive(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def _kmesh_ok(k) -> bool:
    return isinstance(k, list) and len(k) == 3 and all(isinstance(n, int) and not isinstance(n, bool) and n > 0
                                                         for n in k)


def _settings_problem(s: dict) -> str:
    if not _positive(s.get("ecutwfc_Ry")):
        return "the reference calculation states no positive plane-wave cutoff (ecutwfc_Ry)"
    if "ecutrho_Ry" in s and not _positive(s["ecutrho_Ry"]):
        return "the reference calculation's ecutrho_Ry is not a positive number"
    if not _kmesh_ok(s.get("kmesh")):
        return "the reference calculation's kmesh is not three positive integers"
    w = s.get("smearing_width_Ry")
    if w is not None and not (w == 0 or _positive(w)):
        return "the reference calculation's smearing_width_Ry is not a non-negative number"
    return ""


def _component(c, role):
    """The one strain component a calculation applies, (a, a) for a normal strain or (i, j) with i < j for a
    shear; None if it applies none or several."""
    if c.strain is None:
        return None
    e = np.asarray(c.strain, dtype=float)
    if e.shape != (3, 3) or not np.all(np.isfinite(e)) or not np.allclose(e, e.T, rtol=0, atol=1e-12):
        return None
    nonzero = {(i, j) for i in range(3) for j in range(3) if abs(e[i, j]) > 1e-12}
    slots = [(a, a) for a in range(3)] if role == "normal" else [(0, 1), (0, 2), (1, 2)]
    for i, j in slots:
        if nonzero == {(i, j), (j, i)}:                    # nothing else strained: a pure normal or shear strain
            return i, j
    return None


def _pair(calcs, role):
    """The opposite pair with the smallest strain amplitude in one component; None if there is none."""
    def comp(c):
        i, j = _component(c, role)
        return float(c.strain[i][j])
    cand = [c for c in calcs if c.role == role and _component(c, role) is not None]
    pairs = [(p, m) for p in cand for m in cand if _component(p, role) == _component(m, role)
             and comp(p) > 0 and math.isclose(comp(p), -comp(m), rel_tol=1e-6)]
    return min(pairs, key=lambda pm: comp(pm[0]), default=None)


def base_set(card: MaterialCard):
    """(reference, normal+, normal-, shear+, shear-) at the reference settings, or (None, reason)."""
    if card.units != "GPa":
        return None, f"the values are in '{card.units}'; this rule compares them with stresses in GPa"
    own = [c for c in card.evidence if not c.variant_of]
    refs = [c for c in own if c.role == "reference"]
    if len(refs) != 1:
        return None, f"{len(refs)} reference calculations (exactly one required)"
    if not _unstrained(refs[0]):
        return None, "the reference calculation does not state a zero strain"
    problem = _settings_problem(refs[0].settings)
    if problem:
        return None, problem
    normal, shear = _pair(own, "normal"), _pair(own, "shear")
    if normal is None or shear is None:
        return None, "declared runs lack an opposite " + ("normal" if normal is None else "shear") + " strain pair"
    chosen = (refs[0], *normal, *shear)
    if any(c.settings != refs[0].settings for c in chosen):
        return None, "the strained calculations do not use the reference calculation's settings"
    return chosen, ""


def _unstrained(c) -> bool:
    if c.strain is None:
        return False
    e = np.asarray(c.strain, dtype=float)
    return e.shape == (3, 3) and bool(np.all(np.isfinite(e))) and bool(np.all(np.abs(e) <= 1e-12))


def _same_strain(a, b) -> bool:
    if a.strain is None or b.strain is None:
        return False
    ea, eb = np.asarray(a.strain, dtype=float), np.asarray(b.strain, dtype=float)
    return ea.shape == eb.shape == (3, 3) and np.allclose(ea, eb, rtol=1e-9, atol=1e-12)


def quantities(ref, np_, nm, sp, sm) -> dict:
    s = {c.name: np.asarray(c.stress_GPa, dtype=float) if c.stress_GPa is not None else np.full((3, 3), np.nan)
         for c in (ref, np_, nm, sp, sm)}
    a = _component(np_, "normal")[0]
    b, c = (k for k in range(3) if k != a)
    i, j = _component(sp, "shear")
    de = float(np_.strain[a][a]) - float(nm.strain[a][a])
    dg = 2 * (float(sp.strain[i][j]) - float(sm.strain[i][j]))
    c11 = (s[np_.name][a, a] - s[nm.name][a, a]) / de
    c12 = 0.5 * ((s[np_.name][b, b] - s[nm.name][b, b]) + (s[np_.name][c, c] - s[nm.name][c, c])) / de
    c44 = (s[sp.name][i, j] - s[sm.name][i, j]) / dg
    return {"C11": c11, "C12": c12, "C44": c44, "B": (c11 + 2 * c12) / 3, "Cp": (c11 - c12) / 2,
            "P_GPa": -float(np.trace(s[ref.name])) / 3}


def derived_values(card: MaterialCard) -> dict:
    if card.problems():
        return {}
    chosen, _ = base_set(card)
    if chosen is None or any(c.stress_GPa is None for c in chosen):
        return {}
    return {k: round(float(v), 4) for k, v in quantities(*chosen).items()}


def _status(c) -> tuple[str, str]:
    return {"complete": (COMPLETE, ""), "stopped": (OPEN, "stopped before completion; run it again"),
            "failed": (IMPOSSIBLE, c.note or "the program ended without success")}[c.status]


def ledger(card: MaterialCard) -> Ledger:
    problems = card.problems()
    chosen, why = base_set(card) if not problems else (None, "; ".join(problems))
    if chosen is None:
        return Ledger([Comparison("evidence", "the calculations the values rest on", {}, [
            Obligation("declared calculations", {}, status=IMPOSSIBLE, reason=why)])])
    ref = chosen[0]
    reported = Comparison("reported", "values on the card against the reference-setting stresses",
                          {k: REPORTED_LIMIT for k in ("C11", "C12", "C44")}, [])
    for c in chosen:
        st, reason = _status(c)
        reported.obligations.append(Obligation(c.name, c.settings, 0.0, st, reason, c.name if st == COMPLETE else ""))
    if all(o.status == COMPLETE for o in reported.obligations):
        q = quantities(*chosen)
        reported.differences = {k: abs(card.values[k] / q[k] - 1) if k in card.values and q[k] else float("nan")
                                for k in reported.limits}
    comparisons = [reported]
    wanted = ["cutoff_k"] + (["smearing"] if ref.settings.get("smearing_width_Ry") else [])
    for v in wanted:
        cmp = Comparison(v, VARIANTS[v][0], dict(LIMITS), [])
        repeats = []
        for base in chosen:
            need = variant_settings(base.settings, v)
            est = (base.core_hours or 0.0) * ESTIMATE_FACTOR[v]
            hits = [c for c in card.evidence if c.variant_of == base.name and c.variant == v]
            same = [c for c in hits if c.role == base.role and _same_strain(c, base)]
            fit = [c for c in same if satisfies(c.settings, need)]
            # a completed retry counts over an earlier stopped attempt; a failed one still rejects
            fit.sort(key=lambda c: {"complete": 0, "failed": 1, "stopped": 2}[c.status])
            if fit:
                st, reason = _status(fit[0])
                cmp.obligations.append(Obligation(base.name, need, est, st, reason, fit[0].name))
                repeats.append(fit[0])
            else:
                reason = ("missing" if not hits else
                          "present, but its role or strain differs from the calculation it repeats" if not same else
                          "present, but its settings are looser than required")
                cmp.obligations.append(Obligation(base.name, need, est, OPEN, reason))
        base_done = all(c.status == "complete" and c.stress_GPa is not None for c in chosen)
        if base_done and all(o.status == COMPLETE for o in cmp.obligations):
            qb, qv = quantities(*chosen), quantities(*repeats)
            cmp.differences = {"Cp": abs(qv["Cp"] / qb["Cp"] - 1), "C44": abs(qv["C44"] / qb["C44"] - 1),
                               "P_GPa": abs(qv["P_GPa"] - qb["P_GPa"])}
        comparisons.append(cmp)
    return Ledger(comparisons)
