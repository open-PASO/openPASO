"""The handoff check decides from the producer's own settings, and a missing calculation never passes.

Every case is built from a known cubic stiffness, so the expected decision follows from the construction alone:
complete evidence within the limits is accepted; a missing or stopped calculation leaves the card incomplete (with
the calculation in the plan); a failed calculation, a missing opposite strain pair, a non-finite stress, a change
beyond a limit, or reported values the stresses do not support reject it; a repeated calculation with looser or
different settings does not count.
"""
import json
import math

import numpy as np
import pytest

from scale_bridge.card import MaterialCard
from scale_bridge.continuum import cubic_voigt, plane_stress, rotate_about_z
from scale_bridge.rules import check
from scale_bridge.rules.pw_dft_elastic import variant_settings
from tools import scale_bridge as tool

C = {"C11": 110.0, "C12": 60.0, "C44": 33.0}
SMEARED = {"ecutwfc_Ry": 40.0, "ecutrho_Ry": 320.0, "kmesh": [24, 24, 24], "smearing_width_Ry": 0.02,
           "pseudopotential": "X.pbe.UPF"}
UNSMEARED = {k: v for k, v in SMEARED.items() if k != "smearing_width_Ry"}
STRAINS = {"ref": (None, 0.0), "np": ("normal", 0.005), "nm": ("normal", -0.005),
           "sp": ("shear", 0.01), "sm": ("shear", -0.01)}


def _calc(name, role, amp, settings, scale=(1.0, 1.0), pressure=0.01, **kw):
    """Stress from the stiffness C (C' and C44 scaled by `scale`) at the named strain."""
    c11 = C["C12"] + 2 * (C["C11"] - C["C12"]) / 2 * scale[0]
    c12, c44 = C["C12"], C["C44"] * scale[1]
    e, s = np.zeros((3, 3)), -pressure * np.eye(3)
    if role == "normal":
        e[0, 0] = amp
        s += np.diag([c11 * amp, c12 * amp, c12 * amp])
    elif role == "shear":
        e[1, 2] = e[2, 1] = amp / 2
        s[1, 2] = s[2, 1] = c44 * amp
    d = {"name": name, "role": role or "reference", "settings": settings, "strain": e.tolist(),
         "stress_GPa": s.tolist(), "status": "complete", "core_hours": 1.0}
    d.update(kw)
    return d


def card(settings=SMEARED, variants=("cutoff_k", "smearing"), scale=None, drop=(), edit=None, values=None):
    ev = [_calc(n, r, a, settings) for n, (r, a) in STRAINS.items() if n not in drop]
    for v in variants:
        for n, (r, a) in STRAINS.items():
            if n in drop:
                continue
            ev.append(_calc(f"{n}_{v}", r, a, variant_settings(settings, v), (scale or {}).get(v, (1.0, 1.0)),
                            variant_of=n, variant=v))
    if edit:
        edit(ev)
    return MaterialCard.from_dict({"material": "X", "kind": "elastic.cubic", "values": values or dict(C),
                                   "evidence": ev})


def test_complete_evidence_within_limits_is_accepted():
    r = check(card(scale={"cutoff_k": (1.005, 0.995), "smearing": (0.99, 1.01)}))
    assert r["decision"] == "accept", r["reason"]
    assert set(r["comparisons"]) == {"reported", "cutoff_k", "smearing"}
    assert math.isclose(r["derived_values"]["C44"], C["C44"], rel_tol=1e-9)


def test_missing_smearing_comparison_is_incomplete_and_planned():
    r = check(card(variants=("cutoff_k",)))
    assert r["decision"] == "incomplete"
    planned = [p for p in r["plan"] if p["comparison"] == "smearing"]
    assert len(planned) == 5
    assert all(p["settings"]["smearing_width_Ry"] == 0.01 and p["settings"]["kmesh"] == [36, 36, 36] for p in planned)
    assert r["reservation_core_hours"] == pytest.approx(5 * 3.4)


def test_without_smearing_only_the_cutoff_comparison_is_required():
    r = check(card(settings=UNSMEARED, variants=("cutoff_k",)))
    assert r["decision"] == "accept", r["reason"]
    assert "smearing" not in r["comparisons"]


def _permute(ev, perm):
    """The same calculations described in relabelled cube axes: cubic symmetry makes them equivalent."""
    P = np.eye(3)[list(perm)]
    for c in ev:
        for key in ("strain", "stress_GPa"):
            c[key] = (P @ np.asarray(c[key]) @ P.T).tolist()


@pytest.mark.parametrize("perm", [(1, 2, 0), (2, 0, 1)])
def test_normal_axis_and_shear_plane_may_be_any(perm):
    r = check(card(edit=lambda ev: _permute(ev, perm)))
    assert r["decision"] == "accept", r["reason"]
    assert math.isclose(r["derived_values"]["C11"], C["C11"], rel_tol=1e-9)
    assert math.isclose(r["derived_values"]["C12"], C["C12"], rel_tol=1e-9)
    assert math.isclose(r["derived_values"]["C44"], C["C44"], rel_tol=1e-9)


def test_change_beyond_a_limit_rejects():
    r = check(card(scale={"cutoff_k": (1.0, 1.03)}))
    assert r["decision"] == "reject" and "cutoff_k:C44" in r["reason"]


def test_failed_calculation_rejects():
    def fail(ev):
        next(c for c in ev if c["name"] == "sp_smearing").update(status="failed", note="too many k-points")
    r = check(card(edit=fail))
    assert r["decision"] == "reject" and "too many k-points" in r["reason"]


def test_stopped_calculation_is_incomplete():
    def stop(ev):
        next(c for c in ev if c["name"] == "np_cutoff_k").update(status="stopped")
    r = check(card(edit=stop))
    assert r["decision"] == "incomplete"
    assert [p["repeats"] for p in r["plan"]] == ["np"]


def test_missing_opposite_pair_rejects():
    r = check(card(drop=("sm",)))
    assert r["decision"] == "reject" and "opposite shear strain pair" in r["reason"]


def test_non_finite_stress_rejects():
    def nan(ev):
        next(c for c in ev if c["name"] == "sp_cutoff_k")["stress_GPa"][1][2] = float("nan")
    r = check(card(edit=nan))
    assert r["decision"] == "reject" and "not finite" in r["reason"]


def test_looser_or_different_settings_do_not_count():
    def loosen(ev):
        c = next(c for c in ev if c["name"] == "ref_cutoff_k")
        c["settings"] = dict(c["settings"], ecutwfc_Ry=44.0)
        c2 = next(c for c in ev if c["name"] == "np_smearing")
        c2["settings"] = dict(c2["settings"], pseudopotential="Y.pbe.UPF")
    r = check(card(edit=loosen))
    assert r["decision"] == "incomplete"
    assert {p["repeats"] for p in r["plan"]} == {"ref", "np"}
    assert all("looser" in p["reason"] for p in r["plan"])


def test_reported_values_must_follow_from_the_stresses():
    r = check(card(values=dict(C, C44=C["C44"] * 1.03)))
    assert r["decision"] == "reject" and "reported:C44" in r["reason"]


def test_plane_stress_of_an_isotropic_crystal_does_not_depend_on_the_angle():
    iso = cubic_voigt(200.0, 80.0, 60.0)
    q0, q30 = plane_stress(rotate_about_z(iso, 0.0)), plane_stress(rotate_about_z(iso, 30.0))
    assert np.allclose(q0, q30) and np.allclose(q0, q0.T)
    cub = cubic_voigt(C["C11"], C["C12"], C["C44"])
    assert np.allclose(rotate_about_z(cub, 90.0), cub)
    assert not np.allclose(rotate_about_z(cub, 45.0), cub)


class _FakeMCP:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def deco(f):
            self.tools[f.__name__] = f
            return f
        return deco


class _Provider:
    def produce(self, request):
        return card(variants=("cutoff_k",)).to_dict()

    def complete(self, card_dict, plan):
        full = card()
        assert {p["comparison"] for p in plan} == {"smearing"}
        return full.to_dict()


def test_tool_refuses_continuum_data_until_a_provider_completes_the_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENPASO_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(tool.providers, "available", lambda: {"fake": _Provider()})
    mcp = _FakeMCP()
    tool.register_scale_bridge_tools(mcp)
    run = mcp.tools["material_handoff"]
    assert "elastic.cubic" in run("schema")
    produced = json.loads(run("produce", provider="fake", request='{"material": "X"}'))
    assert produced["check"]["decision"] == "incomplete"
    refused = json.loads(run("continuum", card=produced["card_path"]))
    assert refused["passed"] is False and refused["plan"]
    completed = json.loads(run("complete", card=produced["card_path"], provider="fake"))
    assert completed["check"]["decision"] == "accept"
    data = json.loads(run("continuum", card=completed["card_path"], theta_deg=45.0))
    assert data["passed"] is True and len(data["plane_stress_xx_yy_xy"]) == 3
