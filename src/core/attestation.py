"""Result attestation: make fabrication unreachable, not merely detectable.

WHY THIS EXISTS
---------------
openPASO's verification gate stamps a RUN: it confirms that a solver executed,
produced output files, and that those files are finite (``_stamp_verification``
in ``tools/consolidated.py``). What it did not do is bind a REPORTED NUMBER to
that run. The agent read the artefacts itself and wrote its own answer, so a
plausible invented value passed every gate. Detection was only possible outside
the tool, which means a general user of openPASO — one with nobody checking the
result independently — had no protection at all.

WHY THE OBVIOUS FIX DOES NOT WORK
---------------------------------
Attesting values that appear in the solver's *printed* output is worthless,
and this was demonstrated rather than assumed. The agent authors the solver
script, so it controls everything printed. A real skfem solve whose script
ended with

    print("L2_error = 1.0000e-12")     # hard-coded, unrelated to the solve

produced a genuine run, a genuine log, and a genuine artefact hash — and a
scan of that output happily attested the fabricated number. Binding a value to
a run proves only that a run happened.

WHAT THIS MODULE DOES INSTEAD
-----------------------------
Admissible evidence is the solver's DATA output only — the mesh and the nodal
field it wrote. openPASO reads that data with its own reader and computes the
requested quantity with its own arithmetic. Narration (stdout, logs, and any
plain-text file the agent could author) is never consulted. The agent supplies
a job and a field; it never supplies a number.

WHAT THIS GUARANTEES, AND WHAT IT DOES NOT
-----------------------------------------
Guaranteed: a reported number was computed BY OPENPASO, with its own arithmetic,
from structured field data that existed in the run's work directory inside the
run's execution window, and that has not been altered since (content hash).
Typing a number, printing one from the solver script, reporting a quantity the
data does not support, probing outside the mesh, or citing an artefact written
before or after the run are all refused. This holds for any problem, not only
ones with a manufactured solution.

NOT guaranteed: that the field data itself is the result of genuinely solving
the stated problem. The agent authors the solver script, so a script that
writes fabricated field values produces fabricated data that openPASO will then
faithfully read. An adversary who can derive the expected answer can in
principle emit a plausible field. Defending that requires evidence the data
satisfies the discrete problem — a residual or consistency check — which is
NOT implemented here. Do not describe this module as making fabrication
impossible; it makes fabrication-by-assertion impossible, which is a strictly
weaker and honest claim.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from .mesh_independence import (compute_global_l2, field_magnitude,
                                probe_field, read_nodal_mesh)


class AttestationError(Exception):
    """Raised when a quantity cannot be computed from a run's own data."""


# Structured field formats a solver writes natively. Deliberately excludes
# .txt/.log/.out/.csv/.json: those are narration or agent-authorable text.
DATA_SUFFIXES = (".vtu", ".vtk", ".pvtu", ".vtp", ".xdmf", ".xmf", ".bp",
                 ".msh", ".vtkhdf")

ATTESTABLE = ("l2_norm", "max_abs", "probe_values")


@dataclass(frozen=True)
class Attestation:
    job_id: str
    quantity: str
    value: float | None
    values: list | None
    source_file: str
    source_sha256: str
    field: str
    n_points: int
    computed_by: str
    created: float

    def to_record(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# A MESH FILE IS NOT THE RESULT, AND A TAG ARRAY IS NOT A FIELD. Measured 2026-09-29 in the
# web interface: three FEniCSx runs each had their headline numbers "computed by openPASO
# from the run's own data" out of the input mesh -- channel_cyl.msh sorts before
# pressure.vtu, and its first point array is gmsh's own `gmsh:dim_tags` -- so every reply
# carried max_abs 9.055 (a tag number) and told the model to report it. Mesh files go last,
# and bookkeeping arrays (gmsh's tags, VTK's ids and ghost flags) are never a field.
_MESH_SUFFIXES = (".msh",)
_BOOKKEEPING = ("gmsh:", "vtk")


def is_solution_field(name: str) -> bool:
    """False for arrays a mesh writer adds for its own bookkeeping."""
    return not str(name).lower().startswith(_BOOKKEEPING)


def find_data_artefacts(work_dir) -> list[Path]:
    """Solver-written field files only. Never narration. Mesh files last."""
    w = Path(work_dir)
    if not w.is_dir():
        return []
    found = [p for p in w.rglob("*")
             if p.is_file() and p.suffix.lower() in DATA_SUFFIXES]
    return sorted(found, key=lambda p: (p.suffix.lower() in _MESH_SUFFIXES, str(p)))


def _within_run_window(path: Path, started: float | None,
                       finished: float | None, slack: float = 5.0) -> bool:
    """True if the artefact was written while the run was executing.

    Blocks artefacts authored before or after the solver ran. It does NOT stop
    an agent whose own script writes fabricated data during the run — see the
    limits documented at the top of this module.
    """
    if started is None or finished is None:
        return True                      # no window recorded: cannot enforce
    m = path.stat().st_mtime
    return (started - slack) <= m <= (finished + slack)


def attest_quantity(work_dir, job_id: str, quantity: str, *,
                    field: str = "", probe_points=None,
                    result_file=None,
                    run_started: float | None = None,
                    run_finished: float | None = None) -> Attestation:
    """Compute `quantity` from the run's own data output and attest it.

    quantity: one of ATTESTABLE.
      * l2_norm      — volume-weighted L2 norm of the field over the mesh
      * max_abs      — maximum absolute nodal magnitude
      * probe_values — the field interpolated at `probe_points`

    Raises AttestationError if the run produced no admissible data output, or
    if the quantity cannot be computed from it. A value the run's data does not
    support cannot be reported.
    """
    if quantity not in ATTESTABLE:
        raise AttestationError(
            f"'{quantity}' is not attestable; openPASO computes only {ATTESTABLE}")

    candidates = ([Path(result_file)] if result_file
                  else find_data_artefacts(work_dir))
    candidates = [p for p in candidates if p.is_file()]
    if run_started is not None and run_finished is not None:
        kept = [p for p in candidates
                if _within_run_window(p, run_started, run_finished)]
        if candidates and not kept:
            raise AttestationError(
                f"job {job_id}: no data artefact was written during the run "
                f"itself; files present were authored outside the run window")
        candidates = kept
    if not candidates:
        raise AttestationError(
            f"job {job_id}: the run produced no solver data output "
            f"({'/'.join(DATA_SUFFIXES)}). Nothing can be attested: a number "
            f"not computable from the run's data must not be reported.")

    last_err = ""
    for path in candidates:
        try:
            points, cells, fields = read_nodal_mesh(path)
        except Exception as e:                       # unreadable artefact
            last_err = f"{path.name}: {e}"
            continue
        fields = {k: v for k, v in (fields or {}).items() if is_solution_field(k)}
        if not fields:
            last_err = f"{path.name}: no nodal solution field present"
            continue

        name = field if field in fields else next(iter(fields))
        values = fields[name]

        if quantity == "l2_norm":
            val, how = compute_global_l2(points, cells, values)
            out_val, out_vals = float(val), None
        elif quantity == "max_abs":
            mag = field_magnitude(values)
            out_val, out_vals = float(max(abs(m) for m in mag)), None
            how = "max |field| over the run's own nodal data"
        else:                                        # probe_values
            if not probe_points:
                raise AttestationError("probe_values requires probe_points")
            # probe_field extrapolates outside the mesh instead of failing, so
            # the domain must be checked explicitly: a probe the run's mesh does
            # not cover is not supported by its data. (Found by attack suite.)
            import numpy as _np
            _p = _np.asarray(points, float)
            lo, hi = _p.min(axis=0), _p.max(axis=0)
            for q in probe_points:
                qa = _np.asarray(q, float)[:len(lo)]
                if _np.any(qa < lo - 1e-9) or _np.any(qa > hi + 1e-9):
                    raise AttestationError(
                        f"job {job_id}: probe point {list(q)} lies outside the "
                        f"mesh this run produced; its data does not support that "
                        f"value")
            vals = probe_field(points, values, probe_points)
            if any(v is None or not math.isfinite(float(v)) for v in vals):
                raise AttestationError(
                    f"job {job_id}: the run's data does not cover every probe "
                    f"point; the requested values are not supported by it")
            out_val, out_vals = None, [float(v) for v in vals]
            how = "openPASO interpolation of the run's own nodal field"

        if out_val is not None and not math.isfinite(out_val):
            raise AttestationError(
                f"job {job_id}: computed '{quantity}' is not finite")

        return Attestation(
            job_id=job_id, quantity=quantity, value=out_val, values=out_vals,
            source_file=str(path), source_sha256=_sha256(path),
            field=name, n_points=len(points), computed_by=how,
            created=time.time())

    raise AttestationError(
        f"job {job_id}: no readable field artefact "
        f"({last_err or 'none found'}); nothing can be attested")


def verify_attestation(att: Attestation) -> tuple[bool, str]:
    """Re-check that the attested artefact has not changed since attestation."""
    p = Path(att.source_file)
    if not p.is_file():
        return False, "attested artefact no longer exists"
    if _sha256(p) != att.source_sha256:
        return False, "attested artefact changed after attestation"
    return True, "ok"


def require_attested(value_claimed, att: Attestation | None, *,
                     rel_tol: float = 1e-6) -> tuple[bool, str]:
    """Gate a claimed number against an attestation computed by openPASO.

    A claim with no attestation is refused outright. This is what makes
    fabrication unreachable rather than merely detectable.
    """
    if att is None:
        return False, ("REFUSED: no attestation. A reported value must be "
                       "computed by openPASO from a registered run's data output.")
    ok, why = verify_attestation(att)
    if not ok:
        return False, f"REFUSED: {why}"
    if att.value is None:
        return False, "REFUSED: attestation carries no scalar value"
    try:
        claimed = float(value_claimed)
    except (TypeError, ValueError):
        return False, "REFUSED: claimed value is not a number"
    if not math.isfinite(claimed):
        return False, "REFUSED: claimed value is not finite"
    # Scale by the LARGER magnitude, with only a tiny floor to avoid division
    # by zero. A floor of 1.0 made every pair of small numbers compare equal:
    # a claim 8x the attested value passed. (Found by attack suite, fixed.)
    denom = max(abs(att.value), abs(claimed), 1e-300)
    if abs(claimed - att.value) / denom > rel_tol:
        return False, (f"REFUSED: claimed {claimed!r} is not the value openPASO "
                       f"computed from the run's data ({att.value!r}, job "
                       f"{att.job_id}, {att.source_file})")
    return True, "attested"
