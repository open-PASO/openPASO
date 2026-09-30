"""Strengthened verification gate against fabricated results.

This consolidates the checks that an adversarial review showed were necessary.
Each one exists because a specific forgery got through without it; the comment
on each names the forgery.

DESIGN RULES LEARNED THE HARD WAY
---------------------------------
1. NEVER TRUST A RECORD — recompute. An attestation object carrying a value and
   a correct file hash was accepted with the value replaced, because the gate
   re-hashed the file and never recomputed the number. Anything an agent can
   construct in memory is not evidence.
2. NEVER GUESS WHICH ARTEFACT. Taking the alphabetically first readable file let
   an agent pre-empt the genuine result with a decoy, and misread honest
   transient runs by attesting the initial condition.
3. NEVER SUBSTITUTE SILENTLY. Falling back to "some other field" when the
   requested one is absent let a helper array be read as the solution.
4. A BOUNDING BOX IS NOT COVERAGE. Two slivers at opposite corners span the box
   while meshing 0.5% of it; one dangling node reopens the hole entirely.
5. SATISFYING THE EQUATIONS IS THE ONLY REAL DISCRIMINATOR. A field can be more
   accurate than a genuine solve and still be a forgery; what it cannot do is
   satisfy the discrete system the solver actually solved.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np


@dataclass
class GateVerdict:
    accepted: bool
    reasons: list[str] = dc_field(default_factory=list)
    checks: dict = dc_field(default_factory=dict)

    def refuse(self, why: str) -> "GateVerdict":
        self.accepted = False
        self.reasons.append(why)
        return self


# ── mesh and field sanity ─────────────────────────────────────────────────
def _cell_arrays(cells):
    """Normalise cell blocks to (type_name, connectivity).

    meshio returns CellBlock objects, which are NOT subscriptable. Indexing
    them raised TypeError that a broad exception handler swallowed, silently
    turning the whole structural check into a no-op. Handle both forms.
    """
    out = []
    for c in cells:
        name = getattr(c, "type", None)
        data = getattr(c, "data", None)
        if name is None or data is None:
            try:
                name, data = c[0], c[1]
            except (TypeError, IndexError, KeyError):
                continue
        try:
            arr = np.asarray(data, int)
        except (TypeError, ValueError):
            continue
        if arr.ndim == 2 and arr.size:
            out.append((str(name).lower(), arr))
    return out


def check_mesh_sanity(points, cells, *, dim: int,
                      domain_measure_expected: float | None = None,
                      measure_tol: float = 1e-6) -> GateVerdict:
    """Reject meshes that are not the stated domain, or that are degenerate.

    Forgeries this blocks: a four-node one-cell mesh hitting any target value;
    coordinates scaled so an unnormalised norm reaches a chosen number;
    zero-measure cells that push the norm onto a point-RMS fallback; nodes no
    cell references, which inflate a max or dilute a norm for free.
    """
    v = GateVerdict(True)
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[0] < 3:
        return v.refuse("mesh has too few points to be a discretisation")

    arrs = _cell_arrays(cells)
    top = [(n, a) for n, a in arrs
           if (dim == 2 and n.startswith("triangle"))
           or (dim == 3 and n.startswith("tetra"))]
    if not top:
        return v.refuse(f"no {dim}D cells: what was delivered is not a mesh of the domain")

    conn = np.vstack([a for _, a in top])
    if conn.max() >= p.shape[0] or conn.min() < 0:
        return v.refuse("cell connectivity references points that do not exist")

    # every point must belong to a cell
    referenced = np.unique(conn)
    if referenced.size != p.shape[0]:
        return v.refuse(
            f"{p.shape[0] - referenced.size} of {p.shape[0]} points are not "
            f"referenced by any cell")

    # no degenerate cells
    if dim == 2:
        a, b, c = p[conn[:, 0], :2], p[conn[:, 1], :2], p[conn[:, 2], :2]
        meas = 0.5 * np.abs((b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1])
                            - (c[:, 0]-a[:, 0])*(b[:, 1]-a[:, 1]))
    else:
        a, b, c, d = (p[conn[:, i], :3] for i in range(4))
        meas = np.abs(np.einsum("ij,ij->i", np.cross(b-a, c-a), d-a)) / 6.0
    total = float(meas.sum())
    if not np.all(meas > 1e-14 * max(total, 1e-30)):
        return v.refuse("mesh contains zero-measure (degenerate) cells")

    # A geometrically valid mesh can still be too coarse to be a discretisation:
    # four nodes and one constant cover the unit square exactly and hit any
    # target value. Require interior degrees of freedom — a solve with none is
    # not a solve of the interior problem.
    from collections import Counter
    facets = Counter()
    if dim == 2:
        for c in conn:
            for e in ((c[0], c[1]), (c[1], c[2]), (c[0], c[2])):
                facets[tuple(sorted(e))] += 1
    else:
        for c in conn:
            for f4 in ((c[0], c[1], c[2]), (c[0], c[1], c[3]),
                       (c[0], c[2], c[3]), (c[1], c[2], c[3])):
                facets[tuple(sorted(f4))] += 1
    boundary_nodes = {n for f, k in facets.items() if k == 1 for n in f}
    n_interior = p.shape[0] - len(boundary_nodes)
    v.checks["n_interior_nodes"] = int(n_interior)
    if n_interior < 1:
        return v.refuse(
            f"mesh has no interior nodes ({p.shape[0]} points, all on the "
            f"boundary): it cannot represent a solution of the interior problem")

    v.checks["domain_measure"] = total
    v.checks["n_cells"] = int(conn.shape[0])
    v.checks["n_points"] = int(p.shape[0])
    if domain_measure_expected is not None:
        rel = abs(total - domain_measure_expected) / max(abs(domain_measure_expected), 1e-30)
        if rel > measure_tol:
            return v.refuse(
                f"mesh covers measure {total:.6g}, the stated domain has "
                f"{domain_measure_expected:.6g}: this is not the problem's domain")
    return v


def check_field_sanity(values, *, n_points: int) -> GateVerdict:
    """The WHOLE field must be finite, not merely the scalar derived from it.

    Forgery this blocks: 99% NaN nodes whose norm still came out finite, and a
    max whose NaN-vs-number verdict depended on array order.
    """
    v = GateVerdict(True)
    a = np.asarray(values, float)
    if a.shape[0] != n_points:
        return v.refuse("field length does not match the number of mesh points")
    if not np.all(np.isfinite(a)):
        bad = int((~np.isfinite(a)).sum())
        return v.refuse(f"{bad} of {a.size} field values are not finite")
    return v


def select_field(fields: dict, requested: str) -> tuple[str | None, str]:
    """Never substitute silently, never read a metadata array as the solution.

    Forgeries this blocks: asking for 'error' and being given a helper array;
    a file whose first array is vtkGhostType being read as the solution.
    """
    usable = {k: v for k, v in fields.items() if not k.lower().startswith("vtk")}
    if requested:
        if requested not in usable:
            return None, (f"requested field '{requested}' is absent; present: "
                          f"{sorted(usable)}. The gate does not substitute.")
        return requested, "ok"
    if len(usable) == 1:
        return next(iter(usable)), "ok"
    return None, (f"field not specified and {len(usable)} candidates exist "
                  f"({sorted(usable)}); the gate does not guess")


def select_artefact(candidates: list[Path], explicit: Path | None) -> tuple[Path | None, str]:
    """Never guess which file is the result.

    Forgeries this blocks: a decoy named to sort first pre-empting the genuine
    solution. Also fixes an honest-run bug: a numbered time series sorts with
    the initial condition first, so the gate checked t=0 instead of the result.
    """
    if explicit is not None:
        return (explicit, "ok") if explicit.is_file() else (None, f"{explicit} not found")
    files = [c for c in candidates if c.is_file()]
    if not files:
        return None, "the run produced no data artefact"
    if len(files) > 1:
        return None, (f"{len(files)} data artefacts present "
                      f"({[f.name for f in files]}); the result file must be named "
                      f"explicitly — the gate does not choose")
    return files[0], "ok"


def check_probe_coverage(points, cells, probe_points, *, dim: int) -> GateVerdict:
    """Each probe must lie inside an actual cell.

    Forgery this blocks: slivers at opposite corners spanning the bounding box
    while meshing 0.5% of the domain, so probes were answered by nearest-
    neighbour extrapolation from a region with no elements.
    """
    v = GateVerdict(True)
    p = np.asarray(points, float)[:, :dim]
    arrs = [a for n, a in _cell_arrays(cells)
            if (dim == 2 and n.startswith("triangle"))
            or (dim == 3 and n.startswith("tetra"))]
    if not arrs:
        return v.refuse("no cells to test coverage against")
    conn = np.vstack(arrs)
    verts = p[conn]                                   # (ncell, dim+1, dim)
    outside = []
    for q in np.asarray(probe_points, float)[:, :dim]:
        if not _in_any_cell(q, verts, dim):
            outside.append(q.tolist())
    if outside:
        return v.refuse(
            f"{len(outside)} probe point(s) lie in no cell of the delivered mesh "
            f"(e.g. {outside[0]}): the mesh does not cover where the result is "
            f"claimed")
    return v


def _in_any_cell(q, verts, dim, tol=1e-9):
    """Barycentric point-in-simplex test, vectorised over cells."""
    v0 = verts[:, 0, :]
    mats = np.stack([verts[:, i + 1, :] - v0 for i in range(dim)], axis=-1)
    rhs = (q[None, :] - v0)[..., None]
    try:
        lam = np.linalg.solve(mats, rhs)[..., 0]
    except np.linalg.LinAlgError:
        return False
    inside = np.all(lam >= -tol, axis=1) & (lam.sum(axis=1) <= 1.0 + tol)
    return bool(np.any(inside))


# ── general-purpose entry point for the verification gate ─────────────────
# Deliberately CONSERVATIVE. The strict checks above (domain measure, interior
# nodes, unreferenced points, probe coverage) require knowing the problem, so
# they belong to an independent check that has the problem statement. What is
# wired into the run tools must never refuse an honest run from an unknown
# solver, so only unambiguous defects are reported here:
#   * a field containing non-finite values, at ANY node (the existing gate
#     scanned only summary headline numbers and the mesh file, so a field that
#     was mostly NaN could still be stamped verified);
#   * a mesh whose cells are entirely degenerate (zero measure).
def inspect_result_artefacts(result_files, max_files: int = 12) -> list[str]:
    """Structural defects in solver output. Returns human-readable warnings.

    Never raises: a reader failure is reported as an unverifiable artefact
    rather than crashing the run tool.
    """
    warnings: list[str] = []
    try:
        from .mesh_independence import read_nodal_mesh
    except Exception:
        return warnings
    for path in list(result_files)[:max_files]:
        p = Path(path)
        if p.suffix.lower() not in (".vtu", ".vtk", ".pvtu", ".vtp", ".msh", ".vtkhdf"):
            continue
        try:
            points, cells, fields = read_nodal_mesh(p)
        except BaseException:
            continue                      # unreadable: existing checks cover it
        if not fields:
            continue
        for name, values in fields.items():
            if name.lower().startswith("vtk"):
                continue
            arr = np.asarray(values, float)
            if arr.size and not np.all(np.isfinite(arr)):
                bad = int((~np.isfinite(arr)).sum())
                warnings.append(
                    f"{p.name}: field '{name}' has {bad}/{arr.size} non-finite "
                    f"values — the result is numerically invalid.")
        # Degeneracy must be tested DIRECTLY. Routing it through
        # check_mesh_sanity hid it: that function returns early on the first
        # problem it finds (e.g. unreferenced points), so a filter looking for
        # 'degenerate' never saw it and the wiring was a no-op.
        try:
            frac = degenerate_cell_fraction(points, cells,
                                            dim=3 if _looks_3d(points) else 2)
            if frac is not None and frac > 0.5:
                warnings.append(
                    f"{p.name}: {frac:.0%} of cells have zero measure — the "
                    f"mesh is degenerate and the field on it is not a solution.")
        except BaseException:
            pass
    return warnings


def _looks_3d(points) -> bool:
    p = np.asarray(points, float)
    return p.shape[1] >= 3 and float(np.ptp(p[:, 2])) > 1e-12


def degenerate_cell_fraction(points, cells, *, dim: int) -> float | None:
    """Fraction of top-dimensional cells with (near) zero measure.

    Independent of check_mesh_sanity's early-return chain, so a degenerate mesh
    is detected even when the mesh has other defects too.
    """
    p = np.asarray(points, float)
    arrs = [a for n, a in _cell_arrays(cells)
            if (dim == 2 and n.startswith("triangle"))
            or (dim == 3 and n.startswith("tetra"))]
    if not arrs:
        return None
    conn = np.vstack(arrs)
    if conn.size == 0 or conn.max() >= p.shape[0] or conn.min() < 0:
        return None
    if dim == 2:
        a, b, c = p[conn[:, 0], :2], p[conn[:, 1], :2], p[conn[:, 2], :2]
        meas = 0.5 * np.abs((b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1])
                            - (c[:, 0]-a[:, 0])*(b[:, 1]-a[:, 1]))
    else:
        a, b, c, d = (p[conn[:, i], :3] for i in range(4))
        meas = np.abs(np.einsum("ij,ij->i", np.cross(b-a, c-a), d-a)) / 6.0
    scale = float(meas.max()) if meas.size else 0.0
    if scale <= 0:
        return 1.0
    return float((meas <= 1e-12 * scale).mean())


# ── what the result holds: zero fields, a field that never moved, the mesh it lives on ──
# Measured 2026-09-29 in the web interface, three FEniCSx runs of one flow problem that
# the gate reported as "the automated checks passed":
#   * one wrote velocity and pressure that are EXACTLY 0 at every node, on a mesh of 95
#     cells covering [0.15, 0.25] x [0.15, 0.25] -- the obstacle's disc, not the fluid;
#   * one wrote a velocity identical, to 1e-15 of its size, at all 80 saved times of an
#     "unsteady" run (its time term had been replaced by a second viscous term);
#   * one was a real, changing flow.
# Finite and non-degenerate, all three: the scans above had nothing to say. What was
# missing is a look at the numbers themselves, from the run's own files.
_SOLUTION_SUFFIXES = (".vtu", ".vtk", ".vtp", ".xdmf", ".xmf")


def _solution_fields(fields: dict) -> dict:
    from .attestation import is_solution_field
    return {k: np.asarray(v, float) for k, v in (fields or {}).items() if is_solution_field(k)}


def _measure(points, cells) -> float | None:
    """Area (2-D) or volume (3-D) of the triangles/quads or tetrahedra a result lives on."""
    p = np.asarray(points, float)
    total, seen = 0.0, False
    for name, conn in _cell_arrays(cells):
        conn = np.asarray(conn)
        if conn.size == 0 or conn.max() >= p.shape[0]:
            continue
        if name.startswith("triangle"):
            a, b, c = p[conn[:, 0], :2], p[conn[:, 1], :2], p[conn[:, 2], :2]
            total += float(np.sum(0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                                               - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))))
            seen = True
        elif name.startswith("quad"):
            a, b, c, d = (p[conn[:, i], :2] for i in range(4))
            total += float(np.sum(0.5 * np.abs((a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1])
                                               + (b[:, 0] * c[:, 1] - c[:, 0] * b[:, 1])
                                               + (c[:, 0] * d[:, 1] - d[:, 0] * c[:, 1])
                                               + (d[:, 0] * a[:, 1] - a[:, 0] * d[:, 1]))))
            seen = True
        elif name.startswith("tetra"):
            a, b, c, d = (p[conn[:, i], :3] for i in range(4))
            total += float(np.sum(np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)) / 6.0))
            seen = True
    return total if seen else None


def _saved_times(path: Path, max_reads: int = 3):
    """(times, [(time, {field: array})]) of an XDMF time series: its first and last two saves."""
    import meshio
    with meshio.xdmf.TimeSeriesReader(str(path)) as reader:
        reader.read_points_cells()
        n = reader.num_steps
        picks = sorted({0, max(0, n - 2), n - 1})[:max_reads]
        frames = []
        for k in picks:
            t, point_data, _cell_data = reader.read_data(k)
            frames.append((float(t), _solution_fields(point_data)))
    return n, frames


def result_content_findings(result_files, max_files: int = 12):
    """(hard, notes, mesh) about what a run's output files hold.

    hard  -- the result is not a result: every solution field in every file is exactly 0.
    notes -- measured facts a reader needs, with no verdict: a field identical at the first and
             the last saved time of a time series.
    mesh  -- the mesh the result lives on: cells, points, bounding box, area or volume.
    Never raises.
    """
    hard, notes, mesh = [], [], None
    try:
        from .mesh_independence import read_nodal_mesh
    except Exception:
        return hard, notes, mesh
    zero, nonzero = [], []
    for path in list(result_files)[:max_files]:
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix not in _SOLUTION_SUFFIXES:
            continue
        if suffix in (".xdmf", ".xmf"):
            try:
                n, frames = _saved_times(p)
            except BaseException:
                continue
            if n >= 3 and len(frames) >= 2:
                (t0, first), (t1, last) = frames[0], frames[-1]
                for name, arr in last.items():
                    if name not in first or first[name].shape != arr.shape:
                        continue
                    size = float(np.max(np.abs(arr))) if arr.size else 0.0
                    if size == 0.0:
                        continue          # an all-zero field is judged below, from the .vtu
                    change = float(np.max(np.abs(arr - first[name]))) / size
                    if change <= 1e-12:
                        notes.append(
                            f"{p.name}: {n} saved times from t = {t0:.6g} to t = {t1:.6g}, and "
                            f"'{name}' is the same at the first and the last of them (they differ "
                            f"by {change:.1e} of its size): the field never changed in time")
            continue
        try:
            points, cells, fields = read_nodal_mesh(p)
        except BaseException:
            continue
        sol = _solution_fields(fields)
        if not sol:
            continue
        if mesh is None:
            pts = np.asarray(points, float)
            dim = 3 if _looks_3d(pts) else 2
            lo, hi = pts[:, :dim].min(axis=0), pts[:, :dim].max(axis=0)
            mesh = {"file": p.name, "cells": int(sum(len(np.asarray(c)) for _n, c in _cell_arrays(cells))),
                    "points": int(len(pts)),
                    "bounding_box": [[round(float(v), 6) for v in lo], [round(float(v), 6) for v in hi]]}
            meas = _measure(pts, cells)
            if meas is not None:
                mesh["area" if dim == 2 else "volume"] = round(meas, 8)
        for name, arr in sol.items():
            (zero if arr.size and not np.any(arr) else nonzero).append(f"{p.name}:{name}")
    if zero and not nonzero:
        hard.append("every solution field this run wrote is exactly 0 at every node ("
                    + ", ".join(zero) + "): the run computed no field")
    return hard, notes, mesh
