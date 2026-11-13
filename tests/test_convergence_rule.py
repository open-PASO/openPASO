"""The coupling's convergence rule: one per-field measure for the stop test, the noise floor and the check."""
import json
import math
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from core.coupling_driver import (REL_FLOOR, Participant, field_arrays, field_change, run_coupling,
                                  statistic)
from core.field_transfer import InterfaceData
from core.quality_checks import check_residual_blocks

# ── the measure ──────────────────────────────────────────────────────────────


def test_a_field_that_did_not_move_changed_by_zero():
    a = np.array([1.0, 2.0, 3.0])
    assert field_change(a, a) == 0.0
    assert field_change(a, a * (1 + 0.5 * REL_FLOOR)) == 0.0


def test_the_change_is_the_l2_norm_of_the_difference_over_the_new_value():
    new, prev = np.array([3.0, 4.0]), np.array([3.0, 3.0])
    assert field_change(new, prev) == pytest.approx(1.0 / 5.0)


def test_a_new_value_of_exactly_zero_is_undefined_and_does_not_pass():
    assert math.isnan(field_change(np.zeros(3), np.ones(3)))
    assert math.isnan(field_change(np.zeros(3), np.zeros(3)))
    assert math.isnan(statistic({"a": 1e-20, "b": float("nan")}))
    assert not statistic({"a": 1e-20, "b": float("nan")}) < 1.0


def test_a_non_finite_value_never_passes():
    assert field_change(np.array([1.0, np.inf]), np.array([1.0, 1.0])) == float("inf")
    assert field_change(np.array([1.0, 1.0]), np.array([np.nan, 1.0])) == float("inf")
    assert statistic({"a": 0.0, "b": float("inf")}) == float("inf")


def test_huge_values_do_not_overflow():
    new, prev = np.array([1e300, 1e300]), np.array([1.1e300, 1e300])
    assert field_change(new, prev) == pytest.approx(0.1e300 / math.hypot(1e300, 1e300))


def test_the_statistic_is_the_largest_field():
    assert statistic({"a": 1e-3, "b": 1e-9}) == 1e-3


# ── which columns are a field ────────────────────────────────────────────────


def _ifd(values, names=None, fluxes=None, flux_names=None):
    v = np.asarray(values, float)
    return InterfaceData(coordinates=np.zeros((len(v), 2)), values=v, field_name="f",
                         normal_fluxes=None if fluxes is None else np.asarray(fluxes, float),
                         components=names, flux_components=flux_names)


def test_named_columns_make_vector_and_scalar_fields_and_coordinates_are_not_fields():
    f = field_arrays(_ifd([[0, 1, 2, 3]] * 4, ["x", "ux", "uy", "T"]))
    assert set(f) == {"values:u", "values:T"}
    assert f["values:u"].shape == (4, 2) and f["values:T"].shape == (4, 1)


def test_a_lone_component_name_is_its_own_field():
    assert set(field_arrays(_ifd([[1, 2]] * 3, ["ux", "T"]))) == {"values:ux", "values:T"}


def test_without_names_each_array_is_one_field_whatever_its_column_count():
    f = field_arrays(_ifd([[1, 2, 3]] * 3, fluxes=[[1, 2]] * 3))
    assert set(f) == {"values", "normal_fluxes"} and f["values"].shape == (3, 3)


def test_names_that_do_not_match_the_columns_are_refused():
    with pytest.raises(ValueError):
        _ifd([[1, 2, 3]] * 2, ["ux", "uy"])


def test_the_check_uses_the_same_measure_and_limit():
    f, _ = check_residual_blocks({"A.values:u": 1e-9, "A.values:T": 2e-6}, 1e-6)
    assert len(f) == 1 and "A.values:T" in f[0]
    f, _ = check_residual_blocks({"A.values:u": float("nan")}, 1e-6)
    assert f and "could not be judged" in f[0]
    assert check_residual_blocks({"A.values": 9e-7}, 1e-6) == ([], [])


# ── the driver: controls that must fail, and a coupling that must converge ──


def _side(tmp_path: Path, name: str, body: str, imports_from: list) -> Participant:
    wd = tmp_path / name
    wd.mkdir()
    (wd / "side.py").write_text(textwrap.dedent("""
        import json, os
        from pathlib import Path
        imp = json.loads(Path("imports.json").read_text() or "{}")
        k = int(Path("k").read_text()) + 1 if Path("k").exists() else 1
        Path("k").write_text(str(k))
        def partner(name, default):
            d = imp.get(name)
            return default if d is None else d["values"]
        def export(values, names=None):
            d = {"field_name": "f", "n_points": len(values), "coordinates": [[float(i), 0.0] for i in range(len(values))],
                 "values": values}
            if names:
                d["components"] = names
            Path("exports.json").write_text(json.dumps(d))
    """) + textwrap.dedent(body))
    return Participant(name=name, command=[sys.executable, "side.py"], work_dir=wd, imports_from=imports_from,
                       timeout=60)


def _echo(tmp_path):
    return _side(tmp_path, "B", """
        v = partner("A", [[1.0, 1.0]] * 3)
        export([[float(r[0]), float(r[1])] for r in v])
    """, ["A"])


def test_a_contracting_coupling_converges_below_tol(tmp_path):
    a = _side(tmp_path, "A", """
        v = partner("B", [[0.0, 0.0]] * 3)
        export([[0.5 * r[0] + 1.0, 0.5 * r[1] + 2.0] for r in v], ["ux", "uy"])
    """, ["B"])
    r = run_coupling([a, _echo(tmp_path)], max_iter=80, tol=1e-8, accelerator="constant", theta0=1.0, probe=False)
    assert r.converged, r.error
    assert r.residual < 1e-8 and set(r.block_residuals) == {"A.values:u", "B.values"}
    assert "ux, uy" in r.field_declarations["A"]


def test_a_field_still_moving_beside_a_settled_large_one_does_not_converge(tmp_path):
    # T is 1e3 and fixed; ux is 1e-5 and keeps moving by 1e-3 of itself: one lumped norm reads 1e-11
    a = _side(tmp_path, "A", """
        export([[1000.0, 1e-5 * (1 + 1e-3 * (-1) ** k)]] * 3, ["T", "ux"])
    """, [])
    r = run_coupling([a], max_iter=12, tol=1e-6, accelerator="constant", theta0=1.0, probe=False)
    assert not r.converged
    assert r.block_residuals["A.values:ux"] > 1e-4 and r.block_residuals["A.values:T"] == 0.0


def test_a_planted_error_in_a_small_field_does_not_converge(tmp_path):
    # the small field jumps by 10 % on every second iteration, the large one is exact
    a = _side(tmp_path, "A", """
        export([[5.0e4, 2.0e-3 * (1.1 if k % 2 else 1.0)]] * 4, ["p", "q"])
    """, [])
    r = run_coupling([a], max_iter=10, tol=1e-6, accelerator="constant", theta0=1.0, probe=False)
    assert not r.converged and r.block_residuals["A.values:q"] > 0.05


def test_a_diverging_coupling_does_not_converge(tmp_path):
    a = _side(tmp_path, "A", """
        v = partner("B", [[1.0, 1.0]] * 3)
        export([[2.0 * r[0] + 1.0, 2.0 * r[1] + 1.0] for r in v], ["ux", "uy"])
    """, ["B"])
    r = run_coupling([a, _echo(tmp_path)], max_iter=15, tol=1e-6, accelerator="constant", theta0=1.0,
                     probe=False)
    assert not r.converged


def test_a_field_exported_as_exactly_zero_never_counts_as_settled(tmp_path):
    a = _side(tmp_path, "A", """
        export([[1.0, 0.0]] * 3, ["T", "q"])
    """, [])
    r = run_coupling([a], max_iter=5, tol=1e-6, accelerator="constant", theta0=1.0, probe=False)
    assert not r.converged and math.isnan(r.block_residuals["A.values:q"])


def test_a_mesh_that_did_not_get_finer_is_still_named(tmp_path):
    from tools.result_audit import ndof_ladder_findings
    for k in (1, 2, 3):
        (tmp_path / f"run_level{k}.log").write_text("NDOF = 4624\n")
    assert ndof_ladder_findings(tmp_path)


def test_tol_has_no_default(tmp_path):
    a = _side(tmp_path, "A", "export([[1.0]] * 2)", [])
    r = run_coupling([a], max_iter=5)
    assert not r.converged and "tol is required" in r.error


def test_the_noise_floor_is_the_same_statistic_between_replicates(tmp_path):
    # a sampled side: the same answer plus scatter of about 1e-3 of itself
    a = _side(tmp_path, "A", """
        import random
        random.seed(os.urandom(8))
        export([[1.0 + 1e-3 * random.gauss(0, 1), 2.0 + 2e-3 * random.gauss(0, 1)] for _ in range(20)], ["ux", "uy"])
    """, [])
    r = run_coupling([a], max_iter=30, tol=1e-12, accelerator="constant", theta0=1.0, probe=False,
                     noise_replicates=4)
    assert r.noise_floor is not None and 1e-4 < r.noise_floor < 1e-2
    assert r.converged and r.stopped_at_noise_floor
