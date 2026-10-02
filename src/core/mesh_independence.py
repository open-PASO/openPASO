"""
Automated mesh-independence (grid-refinement) study.

MMS convergence tests need a problem whose exact solution can be
manufactured. Application problems from practice offer no such solution;
there the established recourse is a heuristic mesh-refinement study:
halve the discretisation length (once or more) and accept the solution
only once BOTH global norms AND values at selected points stop changing
materially. This module automates that heuristic, backend-generically:

  * :func:`substitute_resolution` — instantiate one input template at a
    given resolution (templates carry a ``__RESOLUTION__`` placeholder,
    never a hard-coded discretisation — the repo's no-anchoring rule).
  * :func:`refinement_resolutions` — the refinement ladder (configurable
    factor and number of refinements, default one halving of h).
  * :func:`extract_level_metrics` — read one level's result file (any
    meshio-readable format: .vtu/.vtk/.vtp) and compute the monitored
    quantities: a volume-weighted global L2 norm of the primary field,
    the global max magnitude, and interpolated values at probe points.
  * :func:`collect_qoi_scalars` — pick up scalar quantities of interest
    a solve script wrote to ``results_summary.json`` (the catalog
    templates' convention), so problem-specific QoIs are monitored too.
  * :func:`compare_levels` — relative change per refinement step and the
    verdict: CONVERGED only if ALL monitored quantities change below the
    threshold on the finest refinement step.

Everything here is pure post-processing arithmetic — no solver-specific
imports, no problem constants. The MCP tool ``verify_mesh_independence``
(src/tools/consolidated.py) drives the re-runs through the normal
backend layer and feeds the outputs to these functions.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("openpaso.mesh_independence")

#: Placeholder token a template must carry where the characteristic
#: discretisation parameter goes (e.g. number of divisions per side).
RESOLUTION_TOKEN = "__RESOLUTION__"

#: Default acceptance threshold: every monitored quantity must change by
#: less than this relative amount on the finest refinement step. 1% is the
#: strict end of the customary 1-2% band used in practice.
DEFAULT_REL_TOL = 0.01

#: Probe changes are normalised by max(|value|, floor) with
#: floor = PROBE_FLOOR_FRACTION * global field scale, so a probe sitting
#: in a near-zero region of the field cannot produce a spurious
#: "not converged" from noise around zero.
PROBE_FLOOR_FRACTION = 0.01

_TINY = 1e-300


# ── template instantiation ────────────────────────────────────────────────


def format_resolution(value) -> str:
    """Render a resolution for substitution: integers without a trailing
    ``.0`` (so ``nx = __RESOLUTION__`` stays a valid element count)."""
    f = float(value)
    if f.is_integer():
        return str(int(f))
    return repr(f)


def substitute_resolution(template: str, resolution,
                          token: str = RESOLUTION_TOKEN) -> str:
    """Replace every occurrence of ``token`` in ``template`` with the
    resolution value. Raises ``ValueError`` if the token is absent —
    running the identical script twice would fake a converged study."""
    if token not in template:
        raise ValueError(
            f"input_template does not contain the resolution placeholder "
            f"'{token}' — the study would re-run the identical problem and "
            f"the comparison would be meaningless. Put '{token}' where the "
            f"characteristic discretisation parameter goes.")
    return template.replace(token, format_resolution(resolution))


def refinement_resolutions(base, refinement_factor: float = 2.0,
                           levels: int = 1,
                           parameter_kind: str = "divisions") -> list:
    """The resolution ladder: base value plus ``levels`` refinements.

    parameter_kind:
      * ``"divisions"`` — the parameter counts elements/divisions; refining
        MULTIPLIES it by the factor and rounds to an integer.
      * ``"size"`` — the parameter is an element/cell size h; refining
        DIVIDES it by the factor (kept as float).
    """
    if levels < 1:
        raise ValueError("levels must be >= 1 (at least one refinement)")
    if refinement_factor <= 1.0:
        raise ValueError("refinement_factor must be > 1")
    if parameter_kind not in ("divisions", "size"):
        raise ValueError("parameter_kind must be 'divisions' or 'size'")
    # A non-positive base would silently produce an invalid ladder
    # ([0, 0, ...]) that only fails later in solver-specific ways
    # (Copilot review, PR #49).
    try:
        base_f = float(base)
    except (TypeError, ValueError):
        raise ValueError(f"resolution must be a number, got {base!r}")
    if not math.isfinite(base_f) or base_f <= 0:
        raise ValueError(
            f"resolution must be a positive number, got {base!r}")
    if parameter_kind == "divisions" and base_f < 1:
        raise ValueError(
            f"resolution counts divisions (parameter_kind='divisions') and "
            f"must be >= 1, got {base!r}; for an element SIZE use "
            f"parameter_kind='size'")
    out = []
    for lvl in range(levels + 1):
        if parameter_kind == "divisions":
            out.append(int(round(float(base) * refinement_factor ** lvl)))
        else:
            out.append(float(base) / refinement_factor ** lvl)
    return out


# ── global norm ───────────────────────────────────────────────────────────

# Corner-vertex count per meshio cell family (higher-order variants like
# triangle6 / tetra10 / hexahedron27 list corners FIRST in meshio/VTK
# ordering, so the first k columns are always the geometric corners).
_CORNERS = {"line": 2, "triangle": 3, "quad": 4, "tetra": 4,
            "hexahedron": 8, "wedge": 6}

# Simplex decompositions used for cell measures (indices into corner list).
_QUAD_TRIS = [(0, 1, 2), (0, 2, 3)]
_HEX_TETS = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6),
             (1, 4, 5, 6), (3, 4, 6, 7)]
_WEDGE_TETS = [(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)]


def _pad3(points: np.ndarray) -> np.ndarray:
    """Return points as (N, 3), zero-padding missing coordinates."""
    p = np.asarray(points, float)
    if p.ndim != 2:
        raise ValueError("points must be a 2-D array")
    if p.shape[1] >= 3:
        return p[:, :3]
    out = np.zeros((p.shape[0], 3))
    out[:, :p.shape[1]] = p
    return out


def _tri_areas(p0, p1, p2) -> np.ndarray:
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def _tet_volumes(p0, p1, p2, p3) -> np.ndarray:
    return np.abs(np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0),
                            p3 - p0)) / 6.0


def _cell_family(cell_type: str) -> Optional[str]:
    m = re.match(r"([a-z]+)", str(cell_type))
    fam = m.group(1) if m else ""
    return fam if fam in _CORNERS else None


def _cell_measures(family: str, pts: np.ndarray,
                   conn: np.ndarray) -> np.ndarray:
    """Measure (length/area/volume) of each cell in a connectivity block."""
    c = conn[:, :_CORNERS[family]]
    v = [pts[c[:, i]] for i in range(c.shape[1])]
    if family == "line":
        return np.linalg.norm(v[1] - v[0], axis=1)
    if family == "triangle":
        return _tri_areas(v[0], v[1], v[2])
    if family == "quad":
        return sum(_tri_areas(v[a], v[b], v[c2]) for a, b, c2 in _QUAD_TRIS)
    if family == "tetra":
        return _tet_volumes(v[0], v[1], v[2], v[3])
    if family == "hexahedron":
        return sum(_tet_volumes(v[a], v[b], v[c2], v[d])
                   for a, b, c2, d in _HEX_TETS)
    if family == "wedge":
        return sum(_tet_volumes(v[a], v[b], v[c2], v[d])
                   for a, b, c2, d in _WEDGE_TETS)
    raise ValueError(f"unsupported cell family: {family}")


def field_magnitude(values) -> np.ndarray:
    """Pointwise magnitude: identity for scalars, Euclidean norm per point
    for vector/tensor fields (flattened over trailing axes)."""
    a = np.asarray(values, float)
    if a.ndim == 1:
        return a
    return np.linalg.norm(a.reshape(a.shape[0], -1), axis=1)


def compute_global_l2(points, cells, values) -> tuple[float, str]:
    """Global L2 norm ``‖u‖_L2(Ω) = sqrt(∫|u|² dΩ)`` of a nodal field,
    computed with volume-weighted quadrature.

    "Volume-weighted" refers to the QUADRATURE, not to a normalisation:
    ``∫|u|² dΩ`` is approximated per cell as (cell measure) x (mean of
    |u|² over the cell's corner vertices) — a first-order rule applied
    identically at every refinement level, so the LIMIT the levels are
    compared against is well defined. The result is the UNNORMALISED
    norm: it is NOT divided by the domain measure, so a constant field
    ``u = c`` yields ``c * sqrt(|Ω|)`` (equal to ``c`` only on a
    unit-measure domain). This is deliberate — the mesh-independence
    verdict consumes only RELATIVE changes between refinement levels,
    which are invariant to any fixed normalisation, and the reported
    absolute value stays a standard, citable L2 norm.

    The highest-dimensional cell family present is used (surface/edge
    blocks in a volume mesh are ignored). Falls back to the plain
    point-wise RMS (flagged as ``"rms_point"``) when no supported cell
    block exists.

    Args:
        points: (N, dim) node coordinates.
        cells: iterable of (cell_type, connectivity) pairs or meshio
            CellBlock objects.
        values: (N,) or (N, k) nodal field.

    Returns:
        (norm, norm_type) with norm_type ``"volume_weighted_l2"`` (the
        quadrature label described above) or ``"rms_point"``.
    """
    pts = _pad3(points)
    mag2 = field_magnitude(values) ** 2

    _DIM = {"line": 1, "triangle": 2, "quad": 2,
            "tetra": 3, "hexahedron": 3, "wedge": 3}
    blocks = []  # (dim, family, conn)
    for cb in cells:
        if hasattr(cb, "type"):
            ctype, conn = cb.type, cb.data
        else:
            ctype, conn = cb
        fam = _cell_family(ctype)
        if fam is None:
            continue
        conn = np.asarray(conn)
        if conn.size == 0:
            continue
        blocks.append((_DIM[fam], fam, conn))

    if not blocks:
        rms = float(np.sqrt(np.mean(mag2))) if mag2.size else float("nan")
        return rms, "rms_point"

    top = max(d for d, _, _ in blocks)
    integral = 0.0
    measure = 0.0
    for d, fam, conn in blocks:
        if d != top:
            continue
        m = _cell_measures(fam, pts, conn)
        cell_mean = mag2[conn[:, :_CORNERS[fam]]].mean(axis=1)
        integral += float(np.sum(m * cell_mean))
        measure += float(np.sum(m))
    if measure <= _TINY:
        rms = float(np.sqrt(np.mean(mag2))) if mag2.size else float("nan")
        return rms, "rms_point"
    return float(np.sqrt(integral)), "volume_weighted_l2"


# ── probe points ──────────────────────────────────────────────────────────


def _active_axes(points: np.ndarray) -> list[int]:
    """Axes along which the mesh actually extends (drops the constant z of
    a 2-D mesh embedded in 3-D)."""
    pts = _pad3(points)
    spans = pts.max(axis=0) - pts.min(axis=0)
    ref = max(float(spans.max()), _TINY)
    return [ax for ax in range(3) if spans[ax] > 1e-10 * ref]


def _interior_fractions(n: int) -> list[float]:
    """``n`` deterministic interior fractions of the bounding box.

    The first two stay at the historical 35% / 65% so default behaviour
    is unchanged; further requests continue with a golden-ratio
    low-discrepancy sequence confined to [15%, 85%] of the box, so any
    ``n_extra`` yields distinct, reproducible interior probes (Copilot
    review, PR #49: values > 2 used to be silently ignored).
    """
    out = [0.35, 0.65][:max(0, n)]
    i = 1
    while len(out) < n:
        f = round(0.15 + ((i * 0.618033988749895) % 1.0) * 0.7, 6)
        if all(abs(f - g) > 1e-3 for g in out):
            out.append(f)
        i += 1
    return out


def default_probe_points(points, values=None, n_extra: int = 2) -> list[list[float]]:
    """Choose probe locations from the mesh itself (never from hard-coded
    problem dimensions): the location of the field's max magnitude (the
    hotspot, where local convergence matters most — only when ``values``
    is given), the bounding-box centre, and ``n_extra`` deterministic
    interior off-centre points (the first two at 35% / 65% of the box,
    further ones from a golden-ratio sequence within 15-85%). Nearby
    duplicates are dropped.
    """
    pts = _pad3(points)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = hi - lo
    diag = max(float(np.linalg.norm(span)), _TINY)
    axes = _active_axes(pts)

    def _frac(f):
        p = (lo + hi) / 2.0
        for ax in axes:
            p = p.copy()
            p[ax] = lo[ax] + f * span[ax]
        return p

    candidates = []
    if values is not None:
        mag = field_magnitude(values)
        if mag.size:
            candidates.append(pts[int(np.argmax(mag))])
    candidates.append((lo + hi) / 2.0)
    for f in _interior_fractions(n_extra):
        candidates.append(_frac(f))

    probes: list[list[float]] = []
    for c in candidates:
        if any(np.linalg.norm(c - np.asarray(p)) < 1e-6 * diag
               for p in probes):
            continue
        probes.append([float(x) for x in c])
    return probes


def probe_field(points, values, probe_points) -> list[float]:
    """Interpolate the field magnitude at the probe locations.

    Linear scattered-data interpolation (scipy griddata) over the mesh's
    active axes, with nearest-neighbour fallback where the probe falls
    outside the convex hull. 1-D meshes use np.interp. The interpolation
    error converges with the field itself, so probe DIFFERENCES between
    refinement levels remain a faithful local convergence signal.
    """
    pts = _pad3(points)
    mag = field_magnitude(values)
    probes = _pad3(np.asarray(probe_points, float))
    axes = _active_axes(pts)

    if not axes:  # degenerate single-point cloud
        return [float(mag[0])] * len(probes)
    if len(axes) == 1:
        ax = axes[0]
        order = np.argsort(pts[:, ax])
        vals = np.interp(probes[:, ax], pts[order, ax], mag[order])
        return [float(v) for v in vals]

    from scipy.interpolate import griddata
    src = pts[:, axes]
    tgt = probes[:, axes]
    vals = griddata(src, mag, tgt, method="linear")
    bad = ~np.isfinite(vals)
    if np.any(bad):
        vals[bad] = griddata(src, mag, tgt[bad], method="nearest")
    return [float(v) for v in vals]


# ── QoI pickup from results_summary.json ─────────────────────────────────


#: results_summary.json keys that describe the DISCRETISATION or the run,
#: not a physical quantity of interest. They change under refinement BY
#: CONSTRUCTION (ndofs quadruples, resolution doubles, wall time grows), so
#: monitoring them would flip every study to NOT CONVERGED — the false
#: negative one recorded run hit live: its
#: summary carried resolution/ndofs and the verdict failed on "QoI 'ndofs'
#: changed 74.61%" while every physical quantity had settled. Matched
#: case-insensitively against the LAST dotted key component.
_QOI_DENYLIST = frozenset({
    "resolution", "ndofs", "n_dofs", "dofs", "ndof", "n_dof", "num_dofs",
    "n_elements", "num_elements", "nelements", "n_cells", "num_cells",
    "ncells", "n_nodes", "num_nodes", "nnodes", "n_points", "num_points",
    "npoints", "n_vertices", "num_vertices", "nx", "ny", "nz",
    "mesh_size", "h", "dx", "n_refinements", "refinement_level", "level",
    "levels", "elapsed", "runtime", "wall_time", "cpu_time",
    "iterations", "n_iterations", "num_iterations",
})


def collect_qoi_scalars(work_dir, max_entries: int = 12) -> dict[str, float]:
    """Flatten every finite scalar in the run's ``results_summary.json``
    files into ``{dotted.key: value}``. Catalog templates write their
    quantities of interest there, so a problem-specific QoI (tip
    deflection, max temperature, ...) is monitored without the tool
    knowing the physics. Missing/unreadable files yield ``{}``.

    Discretisation descriptors and run metadata (``resolution``,
    ``ndofs``, ``n_elements``, ``wall_time``, ...) are EXCLUDED: they
    change under refinement by construction and would turn every study
    into a false NOT CONVERGED (see :data:`_QOI_DENYLIST`).
    """
    out: dict[str, float] = {}

    def _walk(obj, prefix: str):
        if len(out) >= max_entries:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{prefix}.{k}" if prefix else str(k))
        elif isinstance(obj, bool):
            return
        elif isinstance(obj, (int, float)):
            if prefix.rsplit(".", 1)[-1].lower() in _QOI_DENYLIST:
                return
            f = float(obj)
            if math.isfinite(f) and len(out) < max_entries:
                out[prefix] = f
        # lists are skipped: per-node arrays are not scalar QoIs

    try:
        for js in sorted(Path(work_dir).rglob("results_summary.json")):
            try:
                _walk(json.loads(js.read_text()), "")
            except Exception:
                continue
    except Exception:
        pass
    return out


# ── result-file reading ──────────────────────────────────────────────────

# VTK cell-type id → cell family, covering the classic linear/quadratic ids
# AND the arbitrary-order Lagrange ids that dolfinx's VTKFile writes (meshio
# cannot read those files — VTU version 2.2 — so they go through the
# pyvista fallback below). Corner vertices come FIRST in all of them, which
# is the only ordering property the metrics rely on.
_VTK_FAMILY = {
    3: "line", 21: "line", 68: "line",
    5: "triangle", 22: "triangle", 69: "triangle",
    9: "quad", 23: "quad", 28: "quad", 70: "quad",
    10: "tetra", 24: "tetra", 71: "tetra",
    12: "hexahedron", 25: "hexahedron", 29: "hexahedron", 72: "hexahedron",
    13: "wedge", 26: "wedge", 73: "wedge",
}


def _read_with_pyvista(result_file) -> tuple[np.ndarray, list, dict]:
    """Read (points, [(family, conn)], point_data) via pyvista/VTK.

    Used when meshio cannot parse the file (e.g. the VTU 2.2 format with
    Lagrange cells that dolfinx VTKFile writes). Connectivity is rebuilt
    from the raw celltypes/offsets arrays because pyvista's ``cells_dict``
    refuses arbitrary-order Lagrange cells.
    """
    import pyvista as pv

    grid = pv.read(str(result_file))
    if not hasattr(grid, "celltypes"):
        grid = grid.cast_to_unstructured_grid()
    points = np.asarray(grid.points, float)
    pdata = {k: np.asarray(grid.point_data[k]) for k in grid.point_data.keys()}

    celltypes = np.asarray(grid.celltypes)
    conn = np.asarray(grid.cell_connectivity)
    offsets = np.asarray(grid.offset)
    cells: list[tuple[str, np.ndarray]] = []
    # group consecutive same-(type, size) runs into rectangular blocks
    sizes = np.diff(offsets)
    for vtk_id in np.unique(celltypes):
        fam = _VTK_FAMILY.get(int(vtk_id))
        if fam is None:
            continue
        idx = np.nonzero(celltypes == vtk_id)[0]
        for sz in np.unique(sizes[idx]):
            sel = idx[sizes[idx] == sz]
            block = np.stack([conn[offsets[i]:offsets[i + 1]] for i in sel])
            cells.append((fam, block))
    return points, cells, pdata


def read_nodal_mesh(result_file) -> tuple[np.ndarray, list, dict]:
    """Read a result file into (points, cell blocks, point_data).

    meshio first (VTU/VTK/VTP it parses robustly), pyvista/VTK as the
    fallback for files meshio rejects. Raises ``ValueError`` when neither
    reader can parse the file — an unreadable result is not run evidence.
    """
    try:
        import meshio
        m = meshio.read(str(result_file))
        return (np.asarray(m.points, float), list(m.cells),
                dict(getattr(m, "point_data", {}) or {}))
    except BaseException as meshio_err:  # meshio can raise SystemExit
        try:
            return _read_with_pyvista(result_file)
        except BaseException as pv_err:
            raise ValueError(
                f"{Path(result_file).name}: unreadable by meshio "
                f"({meshio_err}) and by pyvista ({pv_err}) — the file "
                "cannot serve as run evidence for the study.") from pv_err


# ── level extraction ─────────────────────────────────────────────────────


def extract_level_metrics(result_file, field: str = "",
                          probe_points=None) -> dict:
    """Read ONE refinement level's result file and compute the monitored
    quantities. Raises ``ValueError`` with a precise reason when the file
    carries no usable nodal field or the requested field is missing.

    Returns::

        {"field", "n_points", "n_cells", "norm_type",
         "global_l2", "global_max",
         "probe_points", "probe_values"}
    """
    points, cell_blocks, pdata = read_nodal_mesh(result_file)
    if not pdata:
        raise ValueError(
            f"{Path(result_file).name}: no point_data fields — nothing to "
            "monitor. Write the primary field as nodal data.")
    if field:
        if field not in pdata:
            raise ValueError(
                f"{Path(result_file).name}: field '{field}' not present; "
                f"available: {sorted(pdata)}")
        fname = field
    else:
        # auto-select: never a vtk* metadata array (vtkGhostType,
        # vtkOriginalPointIds, ... that VTK writers add alongside the field)
        candidates = [k for k in sorted(pdata) if not k.startswith("vtk")]
        if not candidates:
            raise ValueError(
                f"{Path(result_file).name}: only vtk* metadata arrays "
                f"present ({sorted(pdata)}) — name the field explicitly.")
        fname = candidates[0]

    values = np.asarray(pdata[fname], float)
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{Path(result_file).name}: field '{fname}' contains non-finite "
            "values — the level's solve is invalid.")

    norm, norm_type = compute_global_l2(points, cell_blocks, values)
    mag = field_magnitude(values)
    gmax = float(mag.max()) if mag.size else float("nan")

    if probe_points is None:
        probe_points = default_probe_points(points, values)
    probe_vals = probe_field(points, values, probe_points)

    n_cells = int(sum(len(np.asarray(cb.data if hasattr(cb, "data") else cb[1]))
                      for cb in cell_blocks))
    return {
        "field": fname,
        "n_points": int(points.shape[0]),
        "n_cells": n_cells,
        "norm_type": norm_type,
        "global_l2": norm,
        "global_max": gmax,
        "probe_points": [[float(x) for x in p] for p in np.atleast_2d(probe_points)],
        "probe_values": probe_vals,
    }


# ── comparison + verdict ─────────────────────────────────────────────────


def relative_change(coarse: float, fine: float, floor: float = 0.0) -> float:
    """|fine - coarse| / max(|fine|, |coarse|, floor, tiny)."""
    denom = max(abs(fine), abs(coarse), floor, _TINY)
    return abs(fine - coarse) / denom


def compare_levels(levels: list[dict], rel_tol: float = DEFAULT_REL_TOL) -> dict:
    """Relative change of every monitored quantity per refinement step, and
    the verdict.

    ``levels`` is a list of per-level dicts (coarsest first), each with
    ``resolution``, ``global_l2``, ``global_max``, ``probe_values`` and
    optionally ``qoi`` (dict). The verdict is CONVERGED only if, on the
    FINEST refinement step, ALL monitored quantities change by less than
    ``rel_tol``:

      * the global L2 norm and global max of the field (relative to their
        own magnitude), plus every QoI present at both levels;
      * every probe value, normalised by max(|value|, 1% of the field's
        global max) so near-zero probes cannot produce spurious failures.

    Earlier steps are reported as trend but do not decide the verdict —
    the customary acceptance rule looks at the LAST halving.
    """
    if len(levels) < 2:
        raise ValueError("need at least two levels (one refinement) to compare")
    if not (0 < rel_tol < 1):
        raise ValueError("rel_tol must be in (0, 1)")

    steps = []
    for coarse, fine in zip(levels[:-1], levels[1:]):
        probe_floor = PROBE_FLOOR_FRACTION * max(
            abs(float(fine.get("global_max", 0.0))), _TINY)
        entry = {
            "from_resolution": coarse.get("resolution"),
            "to_resolution": fine.get("resolution"),
            "global_l2_rel_change": relative_change(
                float(coarse["global_l2"]), float(fine["global_l2"])),
            "global_max_rel_change": relative_change(
                float(coarse.get("global_max", 0.0)),
                float(fine.get("global_max", 0.0))),
            "probe_rel_changes": [
                relative_change(float(a), float(b), floor=probe_floor)
                for a, b in zip(coarse.get("probe_values", []),
                                fine.get("probe_values", []))],
        }
        qa, qb = coarse.get("qoi") or {}, fine.get("qoi") or {}
        shared = sorted(set(qa) & set(qb))
        if shared:
            entry["qoi_rel_changes"] = {
                k: relative_change(float(qa[k]), float(qb[k])) for k in shared}
        monitored = ([entry["global_l2_rel_change"],
                      entry["global_max_rel_change"]]
                     + entry["probe_rel_changes"]
                     + list(entry.get("qoi_rel_changes", {}).values()))
        entry["max_rel_change"] = max(monitored)
        steps.append(entry)

    last = steps[-1]
    failures = []
    if last["global_l2_rel_change"] >= rel_tol:
        failures.append(
            f"global L2 norm changed {last['global_l2_rel_change']:.2%}")
    if last["global_max_rel_change"] >= rel_tol:
        failures.append(
            f"global max changed {last['global_max_rel_change']:.2%}")
    for i, ch in enumerate(last["probe_rel_changes"]):
        if ch >= rel_tol:
            failures.append(f"probe {i} changed {ch:.2%}")
    for k, ch in last.get("qoi_rel_changes", {}).items():
        if ch >= rel_tol:
            failures.append(f"QoI '{k}' changed {ch:.2%}")

    converged = not failures
    return {
        "steps": steps,
        "rel_tol": rel_tol,
        "converged": converged,
        "failures": failures,
        "verdict": (
            f"CONVERGED — every monitored quantity changed by less than "
            f"{rel_tol:.2%} on the finest refinement step "
            f"(largest change {last['max_rel_change']:.3%})."
            if converged else
            f"NOT CONVERGED — {'; '.join(failures)} on the finest "
            f"refinement step (threshold {rel_tol:.2%}). Refine further "
            f"before reporting these results."),
    }
