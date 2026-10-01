"""
General-purpose simulation quality checks.

These checks provide warnings about common issues. They do NOT prescribe
specific numbers — the agent must determine appropriate resolution, time
steps, etc. based on the physics of each specific problem.
"""

import logging
from typing import Optional

logger = logging.getLogger("openpaso.quality")


def check_time_step(
    dt: float,
    h: float,
    wave_speed: Optional[float] = None,
    diffusivity: Optional[float] = None,
    scheme: str = "explicit",
) -> list[str]:
    """Check time step stability (CFL, Fourier number).

    These are mathematical stability conditions, not guidelines —
    violating them WILL cause the simulation to blow up.
    """
    warnings = []

    if scheme == "explicit":
        if wave_speed is not None and wave_speed > 0:
            cfl = dt * wave_speed / h
            if cfl > 1.0:
                warnings.append(
                    f"CFL = {cfl:.2f} > 1.0 — UNSTABLE for explicit scheme. "
                    f"Reduce dt to below {h / wave_speed:.2e}."
                )

        if diffusivity is not None and diffusivity > 0:
            fourier = dt * diffusivity / (h * h)
            if fourier > 0.5:
                warnings.append(
                    f"Fourier number = {fourier:.2f} > 0.5 — UNSTABLE for explicit diffusion. "
                    f"Reduce dt to below {0.5 * h * h / diffusivity:.2e}."
                )

    return warnings


def check_material_consistency(
    E: Optional[float] = None,
    nu: Optional[float] = None,
    density: Optional[float] = None,
) -> list[str]:
    """Check material parameter sanity — catches obvious errors."""
    warnings = []

    if nu is not None:
        if nu >= 0.5:
            warnings.append(
                f"Poisson ratio nu={nu} >= 0.5 — incompressible material. "
                f"Standard displacement formulations will lock. Use mixed method."
            )
        if nu < 0:
            warnings.append(f"Negative Poisson ratio nu={nu} — verify this is intended (auxetic).")
        if nu < -1.0 or nu > 0.5:
            warnings.append(f"Poisson ratio nu={nu} is outside physical range [-1, 0.5].")

    if E is not None and E <= 0:
        warnings.append(f"Non-positive Young's modulus E={E} — this is unphysical.")

    if density is not None and density <= 0:
        warnings.append(f"Non-positive density={density} — this is unphysical.")

    return warnings


def check_output_configured(solver: str, input_content: str) -> list[str]:
    """Check that the simulation will produce viewable output files."""
    warnings = []

    if solver == "fourc":
        if "IO/RUNTIME VTK OUTPUT" not in input_content:
            warnings.append(
                "No IO/RUNTIME VTK OUTPUT section found. "
                "Without it, no ParaView-readable output will be produced."
            )

    return warnings


# ── output-side validators (physics-agnostic; consume RESULTS, not setup) ──────
# Philosophy: catch silent-wrong results with checks that need NO physics knowledge
# and NO benchmark answer — finiteness, convergence honesty, conservation balance,
# and (when available) consistency against an independent monolithic re-solve.
# These feed the critic / result payload as warnings; they never hardcode a number
# tied to one physics (no Biot, no k*dt — those are problem-specific anchors).
import numpy as _np


def check_finite(values, label: str = "result") -> list[str]:
    """Flag NaN/Inf in a result array — a universal broken-run signal."""
    w = []
    a = _np.asarray(values, float)
    if a.size and not _np.all(_np.isfinite(a)):
        n = int((~_np.isfinite(a)).sum())
        w.append(f"{label}: {n}/{a.size} non-finite (NaN/Inf) values — result is invalid.")
    return w


# Field/mesh formats meshio reads ROBUSTLY and that carry numeric solution data.
# .xdmf/.xmf are deliberately excluded: meshio's XDMF reader can raise SystemExit
# on multi-grid files (killing the process), and solvers that emit XDMF also emit
# a companion .vtu here, so nothing is lost by scanning the .vtu instead.
_FINITE_SCANNABLE = (".vtu", ".vtk", ".vtp", ".pvtu", ".msh", ".vtkhdf")


def _scan_bp_finite(path) -> tuple[list[str], bool]:
    """Best-effort finiteness scan of an ADIOS2 .bp dataset (dolfinx VTXWriter).

    Mac stress audit 2026-07-18: an all-NaN field written ONLY via VTXWriter
    (.bp) was stamped VERIFIED because meshio cannot read .bp — the exact
    'fabricated result' the gate exists to catch (dolfinx Stokes/Taylor-Hood
    templates emit .bp exclusively). Scans via adios2 when importable.
    Returns (warnings, scanned?) — scanned=False when adios2 is unavailable
    so the caller can report that finiteness was NOT asserted.
    """
    try:
        import adios2
        import numpy as _np2
    except Exception:
        return [], False
    w = []
    try:
        with adios2.FileReader(str(path)) as f:
            for name in list(f.available_variables() or {}):
                try:
                    arr = _np2.asarray(f.read(name), float)
                except (ValueError, TypeError):
                    continue  # non-numeric variable (labels, connectivity strings)
                w += check_finite(arr, label=f"{getattr(path, 'name', path)}:{name}")
        return w, True
    except BaseException:
        # adios2 IS present but could not read the dataset: that is a CORRUPT
        # result file, not an unscannable format — report it as hard evidence
        # failure (verdict-flipping), unlike the missing-adios2 case above.
        return [
            f"{getattr(path, 'name', path)}: unreadable/corrupt result file — "
            "the gate could not read it to assert finiteness; the output "
            "cannot serve as verified run evidence."
        ], False



def _scan_point_cloud_vtu(p) -> "list[str] | None":
    """Findings for a zero-cell VTU (a point cloud), or None if the file is not one.

    Reads only the header in-process; the data are read by VTK in a subprocess with a
    timeout, because a VTK reader can segfault on malformed input and this gate runs
    inside the server.
    """
    import re as _re
    import subprocess as _sp
    import sys as _sys
    try:
        head = open(p, "rb").read(4096).decode("utf-8", "replace")
    except OSError:
        return None
    m = _re.search(r'NumberOfPoints="(\d+)"\s+NumberOfCells="(\d+)"', head)
    if not m or int(m.group(2)) != 0 or int(m.group(1)) == 0:
        return None
    code = (
        "import sys, json\n"
        "try:\n"
        "    import pyvista as pv, numpy as np\n"
        "    g = pv.read(sys.argv[1]); bad = []\n"
        "    for k in g.point_data.keys():\n"
        "        a = np.asarray(g.point_data[k], dtype=float)\n"
        "        n = int((~np.isfinite(a)).sum())\n"
        "        if n: bad.append([k, n, int(a.size)])\n"
        "    print(json.dumps({'ok': True, 'n_points': int(g.n_points), 'bad': bad}))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'ok': False, 'err': type(e).__name__ + ': ' + str(e)[:120]}))\n"
    )
    try:
        r = _sp.run([_sys.executable, "-c", code, str(p)], capture_output=True, text=True, timeout=60, stdin=_sp.DEVNULL)
        import json as _json
        line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("{")), None)
        res = _json.loads(line) if line else {"ok": False, "err": f"reader exited {r.returncode}"}
    except Exception as e:  # noqa: BLE001
        res = {"ok": False, "err": f"{type(e).__name__}: {e}"[:120]}
    name = getattr(p, "name", str(p))
    if not res.get("ok"):
        return [f"finiteness not asserted for {name}: a point-cloud VTU (no cells) that meshio "
                f"cannot read, and the VTK reader could not scan it ({res.get('err')})"]
    out = [f"{name}:{k}: {n}/{tot} non-finite (NaN/Inf) values — result is invalid." for k, n, tot in res["bad"]]
    return out


def check_result_files_finite(paths, max_files: int = 25) -> list[str]:
    """Best-effort finiteness scan of a run's OUTPUT files.

    Attestation binds a claim to run evidence, but "a file exists" is not enough:
    a solve can exit 0 and write an output full of NaN/Inf — a fabricated-looking
    result. This reads each result file with meshio (plus adios2 for .bp) and
    flags non-finite values in any point/cell data, so the verification gate can
    reject it. If NONE of the output files could be scanned, that is reported
    too — a VERIFIED verdict must not silently imply a finiteness check that
    never ran. This never raises.
    """
    w = []
    scannable_format_seen = False
    considered = 0
    try:
        import meshio
    except Exception:
        return w
    from pathlib import Path as _Path
    for p in list(paths)[:max_files]:
        p = p if hasattr(p, "suffix") else _Path(str(p))
        suffix = p.suffix.lower()
        if suffix == ".bp":
            considered += 1
            bp_w, bp_scanned = _scan_bp_finite(p)
            w += bp_w
            # .bp counts as scannable only when adios2 actually read it —
            # without adios2 the format is unscannable in this environment.
            scannable_format_seen = scannable_format_seen or bp_scanned
            continue
        considered += 1
        if suffix in (".pvtu", ".pvd"):
            # AN INDEX IS NOT A RESULT FILE. A .pvtu (parallel VTK) or .pvd (time
            # series) only lists the piece files that hold the data; meshio cannot
            # read it, and until 2026-09-23 this scan reported every one as
            # "unreadable/corrupt" -- a hard, verdict-flipping finding -- so every
            # 4C run with runtime VTK output was stamped NOT VERIFIED by this gate,
            # however correct the run was. Measured on 47 of 65 served 4C decks.
            # The pieces are scanned in their own right below; the index is noted.
            # The run gate reads any finding not prefixed "finiteness not asserted" as
            # hard, so the note carries that prefix: it is coverage information.
            w.append(f"finiteness not asserted for {p.name}: an index file that lists "
                     f"the piece files, which are scanned in their own right")
            continue
        if suffix not in _FINITE_SCANNABLE:
            continue
        try:
            m = meshio.read(str(p))
        except BaseException:
            # A POINT CLOUD IS NOT CORRUPT. 4C's particle output (SPH, DEM,
            # peridynamics, Brownian dynamics, beam-to-particle) is a valid VTU
            # with NumberOfCells="0"; meshio raises IndexError on it, and until
            # 2026-09-23 every such run was stamped NOT VERIFIED here as a corrupt
            # file. The XML header says what it is; a VTK reader in a subprocess
            # (VTK can segfault the process on bad input) scans the point data.
            pc = _scan_point_cloud_vtu(p)
            if pc is not None:
                w.extend(pc)
                if not any("not asserted" in x for x in pc):
                    scannable_format_seen = True          # a scanned point cloud is a scanned file
                continue
            # A best-effort scan must NEVER take down the run — some meshio
            # readers even raise SystemExit on malformed input. But a file with
            # a SCANNABLE suffix that fails to parse is a CORRUPT result file,
            # not an unscannable format (stress audit F1: a garbage-only .vtu
            # was stamped VERIFIED with the honesty note relegated to
            # 'validation'). Report it as a hard, verdict-flipping finding —
            # a result the gate cannot read is not verified run evidence.
            w.append(
                f"{p.name}: unreadable/corrupt result file — the gate could "
                "not read it to assert finiteness; the output cannot serve as "
                "verified run evidence.")
            continue
        # Only mark as scanned AFTER a successful read — otherwise an unreadable
        # .vtu would suppress the honesty note without any check having run.
        scannable_format_seen = True
        for name, arr in list(getattr(m, "point_data", {}).items()):
            w += check_finite(arr, label=f"{p.name}:{name}")
        for name, blocks in list(getattr(m, "cell_data", {}).items()):
            for i, arr in enumerate(blocks):
                w += check_finite(arr, label=f"{p.name}:{name}[{i}]")
    if considered and not scannable_format_seen and not w:
        # `not w`: when a hard corrupt-file finding was already emitted, the
        # note below would be misleading (the format IS scannable here — the
        # file is corrupt) and redundant (the hard finding flips the verdict).
        # Not a NaN finding — an honesty note: no output file was in a
        # format scannable in this environment (e.g. only .xplt, or .bp
        # without adios2), so finiteness is NOT asserted by the gate.
        w.append(
            "finiteness not asserted: none of the output files are in a "
            "scannable format (meshio: "
            + ", ".join(_FINITE_SCANNABLE)
            + "; .bp needs the adios2 python package) — verify field values "
            "independently.")
    return w


def _walk_nonfinite(obj, label: str) -> list[str]:
    import math
    w = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            w += _walk_nonfinite(v, f"{label}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            w += _walk_nonfinite(v, f"{label}[{i}]")
    elif isinstance(obj, float):
        if not math.isfinite(obj):
            w.append(f"{label}: non-finite value ({obj}) — result is invalid.")
    return w


# stdout NaN/Inf only where it clearly denotes a numeric RESULT (after = or :,
# optionally bracketed), so prose/paths can't trigger a false downgrade.
import re as _re
# Also matches arrow ("max(u) -> nan") and copula ("residual is nan") headline
# forms (stress audit F2) — still anchored to a result-introducing token so
# prose ("infinite domain", "information") cannot false-trigger.
_STDOUT_NONFINITE = _re.compile(
    r"(?:[=:]|->|→|\bis\b)\s*[\[(]?\s*[+-]?(?:nan|inf|infinity)\b", _re.I)


def check_summary_finite(work_dir, stdout_text: str = "") -> list[str]:
    """Scan a run's HEADLINE numbers — results_summary.json and stdout — for
    NaN/Inf. The mesh-file scan alone misses these: a summary can report
    ``"max_value": Infinity`` (the number the user actually reads) while the VTU
    field stays finite. json.loads parses bare Infinity/NaN to floats, which the
    walk then catches. Never raises.
    """
    import json as _json
    from pathlib import Path as _P
    w = []
    try:
        wd = _P(work_dir)
        for js in sorted(wd.rglob("results_summary.json")):
            try:
                w += _walk_nonfinite(_json.loads(js.read_text()), js.name)
            except Exception:
                continue
    except Exception:
        pass
    if stdout_text and _STDOUT_NONFINITE.search(stdout_text):
        w.append("stdout reports a non-finite (NaN/Inf) numeric result.")
    return w


def check_convergence(converged: bool, residual: float, tol: float) -> list[str]:
    """A non-converged coupled/iterative solve must NOT be reported as a result.
    The single most general silent-wrong guard."""
    w = []
    if not converged:
        w.append(
            f"NOT CONVERGED (residual {residual:.3e} > tol {tol:.1e}) — the reported "
            f"quantities are NOT trustworthy and must not be treated as a solution."
        )
    return w


# ── THE INTERFACE OF A 2-D SIDE IS A CURVE, AND A CURVE MAY BEND ──────────────
# Measured on an exactly balanced exchange across a two-leg seam, each side
# sampling it at its own points: a 2-D point set on a polyline has two nonzero
# singular values, so it was read as a SURFACE and given Delaunay area weights
# (the triangles fill the region between the two legs), and the balance read
# 26.9% -> 28.2% -> 28.8% "NOT balanced", not shrinking. The same straight-line
# reading took one normal, one plane and one axis for a whole bent interface in
# the audit's interface checks. A curve is ordered along itself and measured
# along its own length, leg by leg.
_BEND_DEG = 20.0      # a step turning more than this from its leg starts a new leg
_GAP_FACTOR = 2.5     # an interior step this many times BOTH neighbours is a break


def _angle(a, b) -> float:
    """Angle in degrees between two unit vectors."""
    return float(_np.degrees(_np.arccos(_np.clip(float(_np.dot(a, b)), -1.0, 1.0))))


def _in_plane(c):
    """(uv, None): the points as two coordinates, or (None, why). Two columns are
    taken as given; three only when one column is constant (a 2-D problem
    written with a z column), and that column is left out."""
    if c.shape[1] == 2:
        return c, None
    if c.shape[1] == 3:
        span = float(_np.max(_np.ptp(c, axis=0))) or 1.0
        flat = [j for j in range(3) if float(_np.ptp(c[:, j])) <= 1e-12 * span]
        if flat:
            return c[:, [j for j in range(3) if j != flat[-1]]], None
        return None, "the points do not lie in a plane of constant x, y or z"
    return None, f"the points have {c.shape[1]} coordinates"


def _chain(U):
    """(positions of the distinct points U along one curve, None), or (None, why).

    The minimum spanning tree of points spread along a curve is the curve itself:
    a path through every point, each joined to its neighbours along it. A tree
    that branches is not one curve, and a forest is several pieces."""
    n = len(U)
    if n <= 2:
        return list(range(n)), None
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import minimum_spanning_tree
        from scipy.spatial import cKDTree
        from scipy.spatial.distance import pdist, squareform
    except ImportError as e:
        return None, f"ordering the points needs scipy, which is not importable ({e})"
    if n <= 1500:
        G = squareform(pdist(U))
    else:
        k = 12
        d, j = cKDTree(U).query(U, k=k + 1)
        G = coo_matrix((d[:, 1:].ravel(), (_np.repeat(_np.arange(n), k), j[:, 1:].ravel())),
                       shape=(n, n)).tocsr()
        G = G.maximum(G.T)
    T = minimum_spanning_tree(G).tocoo()
    if T.nnz != n - 1:
        return None, "they fall apart into separate pieces"
    nb: list = [[] for _ in range(n)]
    for i, j in zip(T.row.tolist(), T.col.tolist()):
        nb[i].append(j)
        nb[j].append(i)
    br = [i for i in range(n) if len(nb[i]) > 2]
    if br:
        p = U[br[0]]
        return None, (f"they do not lie on one curve: at ({p[0]:.4g}, {p[1]:.4g}) the chain "
                      f"of nearest points branches")
    ends = [i for i in range(n) if len(nb[i]) == 1]
    cur = min(ends, key=lambda i: (float(U[i][0]), float(U[i][1])))
    order, prev = [cur], -1
    while len(order) < n:
        nxt = [j for j in nb[cur] if j != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        order.append(cur)
    if len(order) != n:
        return None, "the chain of nearest points does not reach every point"
    return order, None


def _line(P):
    """(centre, unit direction along the points) of the best straight line through P."""
    cen = P.mean(axis=0)
    _, _, vt = _np.linalg.svd(P - cen, full_matrices=False)
    t = vt[0] if float(_np.dot(vt[0], P[-1] - P[0])) >= 0 else -vt[0]
    return cen, t


def _through_corner(P, Q, line1, line2, step):
    """|P X| + |X Q|, X where the line of P's leg meets the line of Q's leg, when X
    lies ahead of P along the first leg and before Q along the second (and the path
    through it is not much longer than the step); None otherwise."""
    (c1, t1), (c2, t2) = line1, line2
    det = t1[0] * t2[1] - t1[1] * t2[0]
    if abs(det) < 1e-9:
        return None
    r = c2 - c1
    X = c1 + ((r[0] * t2[1] - r[1] * t2[0]) / det) * t1
    if float((X - P) @ t1) <= 0 or float((Q - X) @ t2) <= 0:
        return None
    d = float(_np.linalg.norm(X - P) + _np.linalg.norm(Q - X))
    return d if d <= 3.0 * step else None


def _legs_along(V, span):
    """(straight runs of chain positions, arc length of each step) for the ordered
    distinct points V.

    A run keeps its steps within _BEND_DEG of its first step. Where the chain
    crosses from one leg to the next, the step between them belongs to neither: a
    run's end point that lies off the line through the rest of its run is left to
    the neighbouring leg, and a single step joining two longer legs across their
    corner is no leg of its own. The length of such a step runs through the corner
    where the two legs' lines meet: a side with no point at the corner would
    otherwise cut it short."""
    d = _np.diff(V, axis=0)
    L = _np.linalg.norm(d, axis=1)
    u = d / L[:, None]
    runs, cur, ref = [], [0], None
    for j in range(len(u)):
        if ref is None:
            cur.append(j + 1)
            ref = u[j]
        elif _angle(u[j], ref) <= _BEND_DEG:
            cur.append(j + 1)
        else:
            runs.append(cur)
            cur, ref = [j, j + 1], u[j]
    runs.append(cur)
    tol = 1e-4 * span
    for k, r in enumerate(runs):
        if len(r) >= 4:
            cen, t = _line(V[r[1:-1]])
            n = _np.array([-t[1], t[0]])
            if abs(float((V[r[0]] - cen) @ n)) > tol:
                r = r[1:]
            if abs(float((V[r[-1]] - cen) @ n)) > tol:
                r = r[:-1]
            runs[k] = r
    keep = []
    for k, r in enumerate(runs):
        if (len(r) == 2 and 0 < k < len(runs) - 1 and len(runs[k - 1]) >= 3
                and len(runs[k + 1]) >= 3
                and _through_corner(V[r[0]], V[r[1]], _line(V[runs[k - 1]]),
                                    _line(V[runs[k + 1]]), L[r[0]]) is not None):
            continue
        keep.append(r)
    seg = L.copy()
    for r1, r2 in zip(keep, keep[1:]):
        i, j = r1[-1], r2[0]
        if j == i + 1 and len(r1) >= 2 and len(r2) >= 2:
            dc = _through_corner(V[i], V[j], _line(V[r1]), _line(V[r2]), L[i])
            if dc is not None:
                seg[i] = dc
    return keep, seg, L


def _leg_endpoints(leg):
    """(P0, P1) of one declared leg, given as ("x"|"y", position, from, to) -- the
    leg x = position (or y = position) running from `from` to `to` -- or as
    ((x0, y0), (x1, y1))."""
    if isinstance(leg, (list, tuple)) and len(leg) == 4 and str(leg[0]).lower() in ("x", "y"):
        pos, a, b = float(leg[1]), float(leg[2]), float(leg[3])
        if str(leg[0]).lower() == "x":
            return _np.array([pos, a]), _np.array([pos, b])
        return _np.array([a, pos]), _np.array([b, pos])
    p0, p1 = leg
    return _np.asarray(p0, float)[:2], _np.asarray(p1, float)[:2]


def _leg_record(P, idx, span):
    """One straight run of points: its direction t (along the curve), normal n
    (t turned by +90 degrees), centre c, length, and whether it is straight."""
    cen = P.mean(axis=0)
    if len(P) < 2 or float(_np.ptp(P, axis=0).max()) <= 0:
        return {"idx": idx, "t": None, "n": None, "c": cen, "length": 0.0, "straight": False}
    _, _, vt = _np.linalg.svd(P - cen, full_matrices=False)
    t = vt[0] if float(_np.dot(vt[0], P[-1] - P[0])) >= 0 else -vt[0]
    n = _np.array([-t[1], t[0]])
    resid = float(_np.max(_np.abs((P - cen) @ n)))
    # straight to 1e-4 of the curve's span: coordinates written with four decimals
    # stay straight, an arc between two bends (off its chord by ~1e-2) does not
    return {"idx": idx, "t": t, "n": n, "c": cen, "length": float(_np.ptp((P - cen) @ t)),
            "straight": resid <= 1e-4 * span + 1e-12}


def interface_curve(coords, legs=None, allow_gaps=False):
    """The interface points as ONE curve in their plane: (curve, None), or (None, why).

    The interface of a 2-D side is a curve whatever its shape: a straight line,
    a polyline of straight legs (one subdomain's corner cut out of the other), or
    an arc. Its points are ordered along it -- by the declared `legs` when the
    caller has them, else along the chain of nearest points -- and each gets its
    distance along the curve, measured through any corner a step cuts.

    `curve` holds: "uv" (the points as two coordinates), "order" (point indices
    along the curve), "s" (each point's distance along it), "length", "closed" and
    "close" (only declared legs that end where they began close; a chain of
    points is read as open), and "legs": one entry per straight run, with "idx"
    (its points, a corner point in both legs it joins), "t" (unit direction along
    the curve), "n" (t turned by +90 degrees), "c" (its centre), "length" and
    "straight".

    Only a point set that plainly is one curve is returned as one. Two rows of a
    surface strip, or a zig-zag through one, also chain into a path, so a chain
    that comes back within a spacing or two of itself far along the curve, or
    that turns at most of its points, is refused (a corner turns once, and its
    two legs part as they leave it).

    A chain with a step many spacings long in it falls apart and is refused, unless
    `allow_gaps`: then the pieces either side of such a step are read as the one
    curve with a stretch left unsampled (probe points that leave out the part of
    the interface near a corner are this shape), no leg crosses the gap, and
    curve["gaps"] lists each gap as the chain positions it joins. Only callers that
    read the legs ask for this; an integral over the gap would be a guess.
    """
    try:
        c = _np.asarray(coords, float)
    except (TypeError, ValueError) as e:
        return None, f"coordinates are not numeric ({e})"
    if c.ndim != 2 or len(c) < 2 or not _np.all(_np.isfinite(c)):
        return None, "coordinates are not an (N, dim) array of finite numbers"
    uv, why = _in_plane(c)
    if uv is None:
        return None, why
    span = float(_np.max(_np.ptp(uv, axis=0)))
    if span <= 0:
        return None, "every point is at the same location"
    # one entry per distinct location; a point listed twice shares its place
    keys: dict = {}
    grp: list = []
    for i, p in enumerate(uv):
        k = (int(round(p[0] / (1e-9 * span))), int(round(p[1] / (1e-9 * span))))
        if k not in keys:
            keys[k] = len(grp)
            grp.append([])
        grp[keys[k]].append(i)
    U = _np.array([uv[g[0]] for g in grp])
    m = len(U)
    if legs:
        return _curve_on_legs(uv, legs, span)
    pos, why = _chain(U)
    if pos is None:
        return None, why
    V = U[pos]
    runs, seg, L = _legs_along(V, span)
    gaps = []
    for j in range(1, len(L) - 1):
        if L[j] > _GAP_FACTOR * max(L[j - 1], L[j + 1]):
            if not allow_gaps:
                return None, (f"they fall apart: a step of {L[j]:.4g} from ({V[j][0]:.4g}, "
                              f"{V[j][1]:.4g}) against {L[j - 1]:.4g} and {L[j + 1]:.4g} either side")
            gaps.append(j)
    if gaps:
        # each piece between two gaps is read on its own, so no leg runs across a gap
        cuts = [0] + [j + 1 for j in gaps] + [len(V)]
        runs, seg = [], L.copy()
        for a, b in zip(cuts, cuts[1:]):
            if b - a >= 2:
                r_p, s_p, _ = _legs_along(V[a:b], span)
                runs += [[a + q for q in r] for r in r_p]
                seg[a:b - 1] = s_p
    one_step = sum(1 for r in runs if len(r) == 2)
    if len(runs) > 2 and one_step > max(1, len(runs) // 2):
        return None, (f"they turn at most of their points ({len(runs)} straight runs, "
                      f"{one_step} of them one step long), which is not one curve")
    s = _np.concatenate([[0.0], _np.cumsum(seg)])
    # A chain is taken as OPEN: a loop whose two ends lie one step apart cannot be
    # told from a narrow U with its two arms one step apart, and an open reading
    # of a loop leaves out one step of its length, not a whole leg.
    if m >= 5:
        loc = _np.maximum(_np.concatenate([[L[0]], L]), _np.concatenate([L, [L[-1]]]))
        if m <= 1500:
            ii, jj = _np.triu_indices(m, 4)
        else:
            from scipy.spatial import cKDTree
            pr = cKDTree(V).query_pairs(r=2.5 * float(loc.max()), output_type="ndarray")
            pr = pr.reshape(-1, 2)
            ii, jj = pr.min(axis=1), pr.max(axis=1)
            ii, jj = ii[jj - ii >= 4], jj[jj - ii >= 4]
        dij = _np.linalg.norm(V[ii] - V[jj], axis=1)
        arc = s[jj] - s[ii]
        near = (dij < 2.5 * _np.maximum(loc[ii], loc[jj])) & (dij < 0.35 * arc)
        if _np.any(near):
            k = int(_np.argmax(near))
            p, q = V[ii[k]], V[jj[k]]
            return None, (f"they come back within {dij[k]:.3g} of themselves, "
                          f"({p[0]:.4g}, {p[1]:.4g}) against ({q[0]:.4g}, {q[1]:.4g}), "
                          f"{arc[k]:.3g} apart along the chain: rows of a surface, not one curve")
    s_orig = _np.empty(len(uv))
    order: list = []
    for q_, p_ in enumerate(pos):
        s_orig[grp[p_]] = s[q_]
        order.extend(grp[p_])
    leg_recs = []
    for r in runs:
        idx = [i for q_ in r for i in grp[pos[q_]]]
        leg_recs.append(_leg_record(V[r], idx, span))
    return {"uv": uv, "order": _np.asarray(order, int), "s": s_orig, "length": float(s[-1]),
            "closed": False, "close": 0.0, "legs": leg_recs,
            "gaps": [(int(j), int(j) + 1) for j in gaps]}, None


def _curve_on_legs(uv, legs, span):
    """interface_curve along DECLARED legs: each point is placed on the nearest
    leg, and its distance along the curve is the lengths of the legs before it
    plus its distance from that leg's start. Consecutive legs need not meet: the
    distance between them is no part of the interface."""
    try:
        ends = [_leg_endpoints(lg) for lg in legs]
    except (TypeError, ValueError) as e:
        return None, f"the declared legs are not (axis, position, from, to) ({e})"
    lens = _np.array([float(_np.linalg.norm(b - a)) for a, b in ends])
    if not len(lens) or _np.any(lens <= 0):
        return None, "a declared leg has no length"
    base = _np.concatenate([[0.0], _np.cumsum(lens)])
    tol = 1e-6 * max(float(base[-1]), span)
    dist = _np.empty((len(uv), len(ends)))
    tpar = _np.empty((len(uv), len(ends)))
    for j, (a, b) in enumerate(ends):
        t = _np.clip(((uv - a) @ (b - a)) / lens[j] ** 2, 0.0, 1.0)
        dist[:, j] = _np.linalg.norm(uv - (a + t[:, None] * (b - a)), axis=1)
        tpar[:, j] = t
    near = _np.argmin(dist, axis=1)
    dmin = dist[_np.arange(len(uv)), near]
    if _np.any(dmin > tol):
        i = int(_np.argmax(dmin))
        return None, (f"the point ({uv[i][0]:.4g}, {uv[i][1]:.4g}) lies {dmin[i]:.3g} from "
                      f"every declared leg")
    s = base[near] + tpar[_np.arange(len(uv)), near] * lens[near]
    order = _np.argsort(s, kind="stable")
    closed = float(_np.linalg.norm(ends[-1][1] - ends[0][0])) <= tol and len(ends) > 1
    recs = []
    for j, (a, b) in enumerate(ends):
        idx = [int(i) for i in order if dist[i, j] <= tol]
        t = (b - a) / lens[j]
        recs.append({"idx": idx, "t": t, "n": _np.array([-t[1], t[0]]), "c": 0.5 * (a + b),
                     "length": float(lens[j]), "straight": True})
    close = (float(base[-1] - s[order[-1]] + s[order[0]]) if closed else 0.0)
    return {"uv": uv, "order": order, "s": s, "length": float(base[-1]),
            "closed": bool(closed), "close": close, "legs": recs}, None


def _curve_weights(curve) -> "_np.ndarray":
    """Trapezoid weights along a curve from interface_curve: sum_i w_i f_i is the
    integral of the piecewise-linear interpolant of f along it."""
    order, s = curve["order"], curve["s"]
    seg = _np.diff(s[order])
    ws = _np.zeros(len(order))
    ws[:-1] += 0.5 * seg
    ws[1:] += 0.5 * seg
    if curve.get("closed") and curve.get("close"):
        ws[0] += 0.5 * curve["close"]
        ws[-1] += 0.5 * curve["close"]
    w = _np.empty(len(order))
    w[order] = ws
    return w


def leg_profiles(pts, allow_gaps: bool = False) -> list:
    """Per straight leg of a curve of interface points: {"c", "t", "n" (the leg's
    centre, direction and normal), "a" (each point's distance along the leg, in
    increasing order), "rows" (their rows of `pts`), "tol"}. A point two legs share
    -- the corner -- is in neither: a flux recovered there mixes the two legs'
    normals, and interpolating along one leg through it compares a different
    quantity. [] when the points are not one curve in a plane."""
    P = _points(pts)
    if P is None or P.shape[1] < 2:
        return []
    if P.shape[1] > 2:
        if float(_np.ptp(P[:, 2:], axis=0).max()) > 1e-12 * max(float(_np.ptp(P, axis=0).max()), 1e-300):
            return []
        P = P[:, :2]
    cur, _why = interface_curve(P, allow_gaps=allow_gaps)
    if cur is None:
        return []
    legs = [g for g in cur["legs"] if g["t"] is not None and g["straight"]]
    count: dict = {}
    for g in legs:
        for i in set(g["idx"]):
            count[i] = count.get(i, 0) + 1
    out = []
    for g in legs:
        idx = sorted({i for i in g["idx"] if count[i] == 1})
        if len(idx) < 2:
            continue
        a = (P[idx] - g["c"]) @ g["t"]
        o = _np.argsort(a, kind="stable")
        sp = _np.diff(a[o])
        sp = sp[sp > 0]
        tol = 0.05 * float(_np.median(sp)) if sp.size else 1e-12
        out.append({"c": g["c"], "t": g["t"], "n": g["n"], "a": a[o],
                    "rows": _np.asarray(idx, int)[o], "tol": tol})
    return out


def on_one_leg(p, profiles):
    """The index of the profile whose leg the point p lies on (inside its span),
    when exactly one; None otherwise."""
    hits = []
    for j, g in enumerate(profiles):
        d = _np.asarray(p, float)[:2] - g["c"]
        s = float(d @ g["t"])
        if abs(float(d @ g["n"])) <= g["tol"] and g["a"][0] - g["tol"] <= s <= g["a"][-1] + g["tol"]:
            hits.append(j)
    return hits[0] if len(hits) == 1 else None


# NOTE (consolidation): this is feature/coupling-robustness's implementation.
# knowledge/coupling-revision had an arclength-INTEGRAL variant plus a private
# _interface_net_flux helper. That version was taken first, on the reasoning that
# the coupling fixtures depend on the arclength quantity — they do not:
# scripts/tier2_fixtures/coupling/_lib/couplinglib.py recomputes it itself and
# says so ("recomputed here so the fixture does not depend on that function being
# right"). Taking the arclength version instead failed 20 of the 69
# test_coupling_robustness tests, which need the per-component balance, the
# roundoff floor and the unit-ratio hint. The coverage reporting the other
# version carried inline ("conservation was NOT CHECKED") is not lost: it lives
# in couple()'s not_run list, which is also taken from that branch.
def interface_nodal_weights(coords, legs=None) -> tuple:
    """Quadrature weights for integrating a P1 nodal field over an interface.

    Returns (weights, dim, detail). `weights` is (N,) with sum = the measure of
    the interface, so `sum_i w_i * f_i` is the integral of the piecewise-linear
    interpolant of f. On failure it returns (None, None, reason). `legs`, when
    the caller has them, are the interface's declared straight legs in order,
    each ("x"|"y", position, from, to); without them a bent interface is ordered
    along the chain of its nearest points.

    WHY THIS EXISTS. Conservation across a coupling interface is a statement
    about INTEGRALS, and the two participants sample the same surface at
    different points, so the two sums are not comparable — only the two
    integrals are. The previous implementation formed the integral by sorting
    the points lexicographically, taking the segment lengths between
    consecutive points and applying a trapezoid rule. On a LINE that is right.
    On a SURFACE it is not an approximation of anything: the lexicographic
    order snakes row by row through the point cloud, and the "arclength" it
    accumulates is the length of that snake, which for a 20x20 patch of a
    1 m x 1 m surface comes out near 20 m instead of the 1 m^2 the integral
    needs. Multiplying a traction by that number produces a "net flux" whose
    magnitude is set by the mesh resolution, and the balance of two such
    numbers is meaningful only when both sides happen to snake the same way.

    So the DIMENSION of the interface is measured rather than assumed:

      dim 1  the points lie on a curve. Ordered along the curve's own principal
             direction (not by a coordinate axis, which fails for any interface
             that is not axis-aligned) and integrated with the trapezoid rule
             over the true segment lengths, so a curved interface is handled.
             A curve that BENDS -- the interface of a 2-D side whose corner is
             cut out of its partner -- has two nonzero singular values like a
             surface, and is told from one by interface_curve: ordered along its
             legs (declared, or the chain of nearest points) and integrated
             along its length, through the corner. Read as a surface, its
             triangles filled the region between the legs (measured: an exactly
             balanced exchange read 27-29 % out of balance).
      dim 2  the points lie on a surface. Triangulated in the surface's own
             best-fit plane and integrated exactly for a P1 field: node i gets
             one third of the area of every triangle it belongs to.
      dim 0  every point is at the same location — there is no interface to
             integrate over.
      dim 3  the points fill a volume; they are not a surface and no surface
             integral exists for them.

    The plane for dim 2 comes from an SVD of the centred coordinates, so it
    works for any orientation. A surface too curved to project without folding
    is REPORTED (a negative or wildly uneven weight is the symptom), never
    silently integrated.
    """
    try:
        c = _np.asarray(coords, float)
    except (TypeError, ValueError) as e:
        return None, None, f"coordinates are not numeric ({e})"
    if c.ndim == 1:
        c = c.reshape(-1, 1)
    if c.ndim != 2 or len(c) == 0:
        return None, None, "coordinates are not an (N, dim) array"
    if not _np.all(_np.isfinite(c)):
        return None, None, "coordinates contain non-finite entries"
    n = len(c)
    if n == 1:
        return None, 0, "a single interface point has no measure to integrate over"
    centred = c - c.mean(axis=0)
    try:
        sv = _np.linalg.svd(centred, compute_uv=False)
    except _np.linalg.LinAlgError as e:
        return None, None, f"the interface point cloud has no usable shape ({e})"
    if sv.size == 0 or sv[0] <= 0:
        return None, 0, "every exported interface point is at the same location"
    # A direction counts as real when it carries more than 1e-8 of the largest
    # extent: a straight interface's transverse singular value is roundoff.
    dim = int(_np.sum(sv > 1e-8 * sv[0]))
    if dim == 0:
        return None, 0, "every exported interface point is at the same location"
    if dim >= 3:
        return (None, 3,
                "the exported interface points fill a VOLUME (three independent "
                "directions), so they are not a surface and no surface integral "
                "over them exists")
    if dim == 1:
        _, _, vt = _np.linalg.svd(centred, full_matrices=False)
        s = centred @ vt[0]
        order = _np.argsort(s)
        cs = c[order]
        seg = _np.linalg.norm(_np.diff(cs, axis=0), axis=1)
        if float(_np.sum(seg)) <= 0:
            return None, 1, "the interface curve has zero length"
        w_sorted = _np.zeros(n)
        w_sorted[:-1] += 0.5 * seg
        w_sorted[1:] += 0.5 * seg
        w = _np.empty(n)
        w[order] = w_sorted
        return w, 1, f"arclength trapezoid over {n} points, length {seg.sum():.6g}"
    # dim == 2 in a plane of the coordinates: a bent CURVE when the points are
    # one, which only a chain along them can tell (see interface_curve). Anything
    # else keeps the surface reading below.
    if c.shape[1] in (2, 3):
        cur, why = interface_curve(c, legs=legs)
        if cur is not None:
            w = _curve_weights(cur)
            if float(_np.sum(w)) > 0 and _np.all(_np.isfinite(w)):
                return w, 1, (f"arclength trapezoid along a bent curve of "
                              f"{len([g for g in cur['legs'] if g['t'] is not None])} legs "
                              f"over {n} points, length {float(_np.sum(w)):.6g}")
        elif legs:
            return None, 1, (f"the interface points do not follow the declared legs "
                             f"({why}), so no integral along them can be formed")
    # dim == 2: triangulate in the surface's own best-fit plane.
    try:
        from scipy.spatial import Delaunay, QhullError
    except ImportError as e:
        return None, 2, (f"the interface is a SURFACE and integrating over it "
                         f"needs a triangulation; scipy is not importable ({e})")
    _, _, vt = _np.linalg.svd(centred, full_matrices=False)
    uv = centred @ vt[:2].T
    try:
        tri = Delaunay(uv)
    except (QhullError, ValueError) as e:
        return None, 2, (f"the interface surface could not be triangulated "
                         f"({type(e).__name__}: {str(e)[:80]})")
    simp = tri.simplices
    p0, p1, p2 = c[simp[:, 0]], c[simp[:, 1]], c[simp[:, 2]]
    # TRUE 3-D area, not the projected one: the projection is only used to
    # decide WHICH triangles, never how big they are, so a tilted or gently
    # curved interface keeps its real measure.
    cr = _np.cross(p1 - p0, p2 - p0)
    area = 0.5 * _np.linalg.norm(_np.atleast_2d(cr), axis=-1)
    w = _np.zeros(n)
    for k in range(3):
        _np.add.at(w, simp[:, k], area / 3.0)
    if float(_np.sum(area)) <= 0 or not _np.all(_np.isfinite(w)):
        return None, 2, "the interface triangulation has no area"
    # A point that no triangle touches contributes nothing to the integral,
    # which silently drops its flux. That is a folded or degenerate projection,
    # not a working quadrature.
    if _np.any(w <= 0):
        return (None, 2,
                f"{int(_np.sum(w <= 0))} of {n} interface points carry zero "
                "quadrature weight — the surface does not project onto its own "
                "best-fit plane without folding, so no reliable surface "
                "integral can be formed here")
    return w, 2, (f"surface quadrature over {len(simp)} triangles, "
                  f"area {area.sum():.6g}")


# ── WHICH ROWS OF TWO SIDES ARE THE SAME POINT ────────────────────────────────
# Two sides list their interface points each in its own order: one along the
# interface, another in its solver's dof order. Paired by row, or by a sort on one
# coordinate (which on a bent interface interleaves its two legs), different points
# were compared. Measured on coupled runs across a two-leg interface: a side that
# exported its imports to round-off was told its trace differed by 100 % and 73 %;
# the pointwise flux check said "the two sides do not export the same interface
# points" of two sides that exported the same points in another order, and stayed
# silent on a flux applied by row index that it reads as 143 % -> 128 % -> 118 %
# once the rows are paired by where they are; a conforming seam was called
# NON-MATCHING. Rows are paired here by position alone.
_PAIR_REL_TOL = 1e-6


def _points(c):
    """An (N, d) array of finite coordinates, or None. A flat list is N points on a line."""
    if c is None:
        return None
    try:
        a = _np.asarray(c, float)
    except (TypeError, ValueError):
        return None
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2 or not len(a) or not _np.all(_np.isfinite(a)):
        return None
    return a


def _shared_columns(A, B):
    """(A, B) in the columns both have, when every column only one of them has is
    constant (a 2-D side written with a z column); None when such a column varies."""
    if A.shape[1] == B.shape[1]:
        return A, B
    w = A if A.shape[1] > B.shape[1] else B
    k = min(A.shape[1], B.shape[1])
    sc = float(_np.max(_np.ptp(w, axis=0))) or 1.0
    if k < 1 or float(_np.max(_np.ptp(w[:, k:], axis=0))) > 1e-12 * sc:
        return None
    return A[:, :k], B[:, :k]


def _nearest_rows(P, Q):
    """(distance, row of Q) of the nearest row of Q to each row of P."""
    try:
        from scipy.spatial import cKDTree
        d, j = cKDTree(Q).query(P)
        return _np.asarray(d, float), _np.asarray(j, int)
    except ImportError:
        d, j = _np.empty(len(P)), _np.empty(len(P), int)
        for a in range(0, len(P), 256):
            D = _np.linalg.norm(P[a:a + 256, None, :] - Q[None, :, :], axis=2)
            j[a:a + 256] = _np.argmin(D, axis=1)
            d[a:a + 256] = D[_np.arange(len(D)), j[a:a + 256]]
        return d, j


def _closest_spacing(P) -> float:
    """The smallest distance between two rows of P (0.0 for fewer than two rows)."""
    if len(P) < 2:
        return 0.0
    try:
        from scipy.spatial import cKDTree
        d, _j = cKDTree(P).query(P, k=2)
        return float(_np.min(d[:, 1]))
    except ImportError:
        D = _np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        D[_np.arange(len(P)), _np.arange(len(P))] = _np.inf
        return float(D.min())


def _distinct_rows(P, tol):
    """The first row at each distinct location of P, in row order; rows within `tol`
    of each other are one location (a point a side lists twice is one point)."""
    n = len(P)
    root = list(range(n))

    def _find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i
    try:
        from scipy.spatial import cKDTree
        pairs = cKDTree(P).query_pairs(tol, output_type="ndarray") if n > 1 else []
    except ImportError:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
                 if float(_np.linalg.norm(P[i] - P[j])) <= tol]
    for i, j in pairs:
        ri, rj = _find(int(i)), _find(int(j))
        if ri != rj:
            root[max(ri, rj)] = min(ri, rj)
    return sorted({_find(i) for i in range(n)})


def pair_interface_points(ca, cb, rel_tol: float = _PAIR_REL_TOL):
    """Which rows of two interface point lists are the SAME point, whatever order
    each side lists them in. A dict, or None when the coordinates cannot be read:

      "ia", "ib"      the paired rows (A's row ia[k] is B's row ib[k]), in A's order
      "reps_a/b"      the first row at each distinct location of each side
      "only_a/b"      the distinct rows that have no partner point
      "tol"           the distance within which two points are one

    A row of A and a row of B pair when each is the other's nearest point and they
    lie within `rel_tol` of the interface's extent (and within a quarter of the
    closest spacing either side has, so a neighbour never stands in for a missing
    partner). Row order, and a sort on any one coordinate, play no part."""
    A, B = _points(ca), _points(cb)
    if A is None or B is None:
        return None
    got = _shared_columns(A, B)
    if got is None:
        return None
    A, B = got
    ext = float(_np.max(_np.ptp(_np.vstack([A, B]), axis=0)))
    if ext <= 0:
        return None
    tol = rel_tol * ext
    ra, rb = _distinct_rows(A, 1e-12 * ext), _distinct_rows(B, 1e-12 * ext)
    Ar, Br = A[ra], B[rb]
    for P in (Ar, Br):
        h = _closest_spacing(P)
        if h > 0:
            tol = min(tol, 0.25 * h)
    dab, jab = _nearest_rows(Ar, Br)
    _dba, jba = _nearest_rows(Br, Ar)
    ia, ib = [], []
    for i, (d, j) in enumerate(zip(dab, jab)):
        if d <= tol and int(jba[j]) == i:
            ia.append(ra[i])
            ib.append(rb[int(j)])
    pa, pb = set(ia), set(ib)
    return {"ia": ia, "ib": ib, "reps_a": ra, "reps_b": rb, "tol": tol,
            "only_a": [r for r in ra if r not in pa], "only_b": [r for r in rb if r not in pb]}


def interface_correspondence(ca, cb, rel_tol: float = _PAIR_REL_TOL) -> dict:
    """How two sides' interface points correspond, read by where they are.

    "kind" is
      "same"       every point either side lists, the other lists too;
      "common"     the points both list pair up, and each point only one side lists
                   lies beyond the other side's first or last point along the
                   interface (one side leaves out the interface's end points, as a
                   side whose end nodes keep the outer value does);
      "different"  a point one side lists inside the other's extent has no partner
                   point: two samplings of the interface;
      None         the coordinates could not be read ("why" says how).
    The pairs of pair_interface_points ride along ("ia", "ib", "only_a", "only_b")."""
    p = pair_interface_points(ca, cb, rel_tol)
    if p is None:
        return {"kind": None, "why": "the coordinates of the two sides could not be read as "
                                     "points in the same coordinates"}
    out = dict(p)
    if not p["only_a"] and not p["only_b"]:
        out.update(kind="same", why=None)
        return out
    if len(p["ia"]) < 2:
        out.update(kind="different", why=None)
        return out
    A, B = _shared_columns(_points(ca), _points(cb))
    U = _np.vstack([A[p["reps_a"]], B[p["only_b"]]])
    if U.shape[1] == 1:
        cur, why = {"s": U[:, 0] - U[:, 0].min(), "length": float(_np.ptp(U[:, 0]))}, None
    elif U.shape[1] in (2, 3):
        cur, why = interface_curve(U)
    else:
        cur, why = None, "they are not a curve in a plane"
    if cur is None:
        out.update(kind="different",
                   why=f"the points one side alone lists could not be placed along the interface ({why})")
        return out
    s = cur["s"]
    na = len(p["reps_a"])
    s_of_a = dict(zip(p["reps_a"], s[:na]))
    s_a = s[:na]
    s_b = _np.concatenate([[s_of_a[i] for i in p["ia"]], s[na:]])
    eps = 1e-9 * max(float(cur["length"]), 1e-300)
    beyond_b = all(v < s_b.min() - eps or v > s_b.max() + eps for v in (s_of_a[r] for r in p["only_a"]))
    beyond_a = all(v < s_a.min() - eps or v > s_a.max() + eps for v in s[na:])
    out.update(kind="common" if (beyond_a and beyond_b) else "different", why=None)
    return out


def _common_support(ca, cb):
    """Row indices (ia, ib) of the interface points the two sides SHARE, when
    one side's points are a subset of the other's; None otherwise (two
    genuinely different samplings, or fewer than two shared points). Rows are
    paired by where they are (pair_interface_points), in any order."""
    p = pair_interface_points(ca, cb)
    if p is None or len(p["ia"]) < 2 or (p["only_a"] and p["only_b"]):
        return None
    return list(p["ia"]), list(p["ib"])


def _restrict_export(e, idx):
    """A copy of an export dict with coordinates/values/normal_fluxes cut to
    the rows `idx`; keys whose length does not match the points are left out."""
    out = {}
    for k in ("coordinates", "normal_fluxes", "values"):
        v = e.get(k) if isinstance(e, dict) else getattr(e, k, None)
        if v is None:
            continue
        try:
            arr = _np.asarray(v, float)
        except (TypeError, ValueError):
            continue
        if arr.ndim >= 1 and len(arr) > max(idx):
            out[k] = arr[idx].tolist()
    return out


# A FLUX AT ROUND-OFF IS NOT A FLUX. Measured on a coupled run whose Dirichlet
# side ignored its imports and recovered no flux: both sides exported fluxes of
# about 1e-16 against an interface trace of order one, and the balance read "NOT
# balanced ... SIGN-CONVENTION" on nets of 9.2e-17 and 7.2e-17, the pointwise check
# "does NOT match POINT BY POINT" on values of 1e-17, and the sensitivity probe "NOT
# COUPLED" for the side that did apply its import. A relative measure of two
# round-off numbers is noise over noise. Measured against the trace the sides
# exchange: a flux below _ROUNDOFF of it carries nothing a relative check can read.
_ROUNDOFF = 1e-12


def _flux_and_trace(e):
    """(largest |normal_fluxes|, largest |values|) of one export; None for a key
    that is absent, empty or not numeric."""
    out = []
    for key in ("normal_fluxes", "values"):
        v = e.get(key) if isinstance(e, dict) else getattr(e, key, None)
        try:
            a = _np.asarray(v, float).ravel() if v is not None else _np.zeros(0)
        except (TypeError, ValueError):
            a = _np.zeros(0)
        a = a[_np.isfinite(a)]
        out.append(float(_np.max(_np.abs(a))) if a.size else None)
    return out[0], out[1]


def _exchange_at_roundoff(*exports):
    """(largest |flux|, largest |trace|) when every export carries fluxes and all of
    them are nonzero round-off against the largest trace value the exports carry;
    None otherwise (a real flux, an absent or empty flux, exactly zero fluxes --
    which are said as such elsewhere -- or no trace to measure against)."""
    fl = [_flux_and_trace(e) for e in exports]
    if not fl or any(f is None for f, _t in fl):
        return None
    fmax = max(f for f, _t in fl)
    tmax = max((t for _f, t in fl if t is not None), default=0.0)
    if tmax > 0 and 0 < fmax <= _ROUNDOFF * tmax:
        return fmax, tmax
    return None


def _interface_balance_core(export_a, export_b, label_a="A", label_b="B",
                            rtol: float = 0.05, floor: float = 0.0,
                            numbers: dict | None = None) -> list[str]:
    """Conservation across a coupling interface: the net flux leaving A should equal
    the net flux entering B (global balance). Pure arithmetic on the exchanged
    normal_fluxes — no physics. `export_*` are InterfaceData-like dicts/objects.

    An empty finding list means CHECKED AND BALANCED. Whether the check could
    run at all is answered here too, as a finding: a coupling that exchanges no
    fluxes, or whose two sides cannot be compared, gets an explicit NOT CHECKED
    entry rather than silence. This used to delegate the question to
    `interface_balance_coverage`, which was never written — it appeared in this
    sentence and nowhere else in the tree — so for every caller the caveat was
    unenforced and a coupling with no conservation evidence was indistinguishable
    from one that conserves exactly.

    `floor` is an absolute magnitude below which an imbalance is float noise
    rather than a finding. It exists because the per-component branch below
    compares each component on its OWN scale: a component that is zero on both
    sides — a tangential traction on a frictionless interface is exactly this,
    and it is the common case, not a corner one — then has a denominator of
    roundoff, and two ~1e-17 entries that should cancel report an imbalance of
    tens of percent on a coupling that is exactly right. Demonstrated: a normal
    traction of 1e5 cancelling exactly, with a 1e-17 tangential component on
    both sides, was reported as 91.6% non-conservative. The vector branch
    therefore derives one floor from the WHOLE interface and hands it down, so a
    component is only ever judged against the size of the flux actually being
    exchanged. Callers comparing a single scalar keep the old behaviour (0.0).
    """
    w = []
    _ro = _exchange_at_roundoff(export_a, export_b)
    if _ro is not None:
        if numbers is not None:
            numbers.update({"roundoff": True, "flux_scale": _ro[0], "trace_scale": _ro[1],
                            "rtol": float(rtol)})
        return [f"Interface flux balance NOT CHECKED (the exchange is at round-off): the "
                f"largest flux {label_a} or {label_b} exported is {_ro[0]:.1e}, against an "
                f"interface trace of {_ro[1]:.3g}, so both exported round-off rather than a "
                f"flux. A balance of two round-off numbers passes or fails by noise, so "
                f"nothing is judged here."]

    def _co(e):
        c = e.get("coordinates") if isinstance(e, dict) else getattr(e, "coordinates", None)
        if c is None:
            return None
        try:
            a = _np.asarray(c, float)
        except (TypeError, ValueError):
            return None
        return _np.atleast_2d(a) if a.ndim >= 1 else None

    # WHEN TO INTEGRATE. Two sides that sample the interface at exactly the SAME
    # points are directly comparable and their sums are the right quantity — one
    # quadrature rule applied to both, so a redistribution along the interface
    # cancels in the sum exactly, which is what makes it the flux-PROFILE
    # check's job to catch rather than this one's. Two sides that sample it
    # DIFFERENTLY are not comparable at all: 41 samples against 31 of the same
    # physical flux differ by the sampling, not by the physics, and only an
    # integral over the interface removes that. Measured on the shipped
    # participant pairs: nets of 313.8 against -240 for two fields that agree to
    # 1e-5 pointwise. So integrate exactly when the two discretisations differ —
    # which is a point-by-point comparison of the coordinates, not of their
    # count: two sides with the same number of points spread differently along
    # the interface are just as incomparable as two with different counts.
    # With no coordinates on either side there is nothing to integrate against
    # and nothing that says the two samplings differ, so the plain sum stands —
    # that is what the scalar unit tests and this function's own per-component
    # recursion hand in.
    ca, cb = _co(export_a), _co(export_b)
    _same_sampling = (ca is not None and cb is not None and ca.shape == cb.shape
                      and _np.allclose(ca, cb, rtol=1e-9, atol=1e-12))
    # THE COMMON SUPPORT, BEFORE ANY INTEGRAL. The shape the shipped participant
    # pairs actually produce is not two samplings of one interface but ONE
    # sampling seen through two windows: a 4C side exports the interface nodes
    # WITHOUT the two endpoints (outer nodes that keep the outer datum), its
    # partner exports every node including them. Each side integrated over its
    # own extent then sets an integral over the whole interface against one over
    # the interior segments, and the difference is the flux through the two end
    # segments -- a number fixed by the export windows, not by the physics.
    # Measured on a correct thermo-mechanical coupling: the two nets read 8-11%
    # apart at the coarsest level and cancelled to 1e-3 at the finest -- the
    # shrinking-with-h signature of a discretisation error, which is what the
    # verdict then called it, on the correct run AND on a wrong one, because at
    # level 1 the check could not tell them apart. When one side's points are a
    # subset of the other's, both sides are cut to the shared points and compared
    # there under one rule, so the extents agree by construction. The integral
    # path stays for two genuinely different samplings.
    _support_note = ""
    _support_kind = "same" if _same_sampling else "sums"
    _common_integrate = False
    _shared = (_common_support(ca, cb)
               if (ca is not None and cb is not None and not _same_sampling)
               else None)
    if _shared is not None:
        _ia, _ib = _shared
        _n_a, _n_b = len(ca), len(cb)
        _ra, _rb = _restrict_export(export_a, _ia), _restrict_export(export_b, _ib)
        _wts, _dim, _detail = interface_nodal_weights(ca[_ia])
        if _dim is not None and _dim >= 3:
            # a shared point set that fills a volume is not an interface; say so
            # rather than summing over it (the integral path says the same)
            return [f"Interface flux balance could NOT be evaluated: {_detail}. "
                    f"Conservation is unchecked."]
        if "normal_fluxes" in _ra and "normal_fluxes" in _rb:
            export_a, export_b = _ra, _rb
            ca, cb = ca[_ia], cb[_ib]
            _same_sampling = True
            _support_kind = "common"
            # ON THE SHARED POINTS' OWN WEIGHTS: summed plainly, a non-uniform shared
            # point set is not the interface integral (the weights were computed here
            # and never used).
            _common_integrate = _wts is not None
            if len(_ia) < max(_n_a, _n_b):
                _support_note = (
                    f" (compared on the {len(_ia)} interface points both sides "
                    f"export; {label_a} exports {_n_a}, {label_b} {_n_b}, and "
                    f"the points one side alone exports are outside the "
                    f"comparison)")
    _integrate = ca is not None and cb is not None and (not _same_sampling or _common_integrate)
    if _integrate and not _common_integrate:
        _support_kind = "integrated"
    # A NEUMANN SIDE'S CONSISTENT FLUX IS THE FLUX IT WAS GIVEN. On a
    # Dirichlet-Neumann pair the Neumann side applies the partner's flux as its
    # natural boundary condition, and a consistent recovery of that side's own
    # boundary flux returns the applied datum to solver precision -- so at the
    # shared points the two exports are exact opposites BY CONSTRUCTION, on a
    # correct coupling and on a wrong one alike (measured on three recorded
    # runs of one round, one correct, two wrong: 0.0% at every level once the
    # extents matched). Two copies of one number cannot disagree, so a pass
    # here would be no conservation evidence, and it must not be read as one.
    # Conservation is judged on the flux each side recovers from its OWN field
    # at the task's interface points -- the per-level interface deliverables,
    # which the audit compares channel by channel and level by level.
    if _same_sampling and ca is not None:
        try:
            _fa = _np.asarray(export_a.get("normal_fluxes") if isinstance(export_a, dict)
                              else getattr(export_a, "normal_fluxes", None), float)
            _fb = _np.asarray(export_b.get("normal_fluxes") if isinstance(export_b, dict)
                              else getattr(export_b, "normal_fluxes", None), float)
            _fa = _fa.reshape(len(ca), -1) if _fa.size == len(ca) * max(1, _fa.size // max(len(ca), 1)) else None
            _fb = _fb.reshape(len(cb), -1) if _fb is not None and _fb.size == len(cb) * max(1, _fb.size // max(len(cb), 1)) else None
        except (TypeError, ValueError, AttributeError):
            _fa = _fb = None
        if (_fa is not None and _fb is not None and _fa.shape == _fb.shape and _fa.size
                and _np.all(_np.isfinite(_fa)) and _np.all(_np.isfinite(_fb))):
            _sc = max(float(_np.max(_np.abs(_fa))), float(_np.max(_np.abs(_fb))))
            if _sc > 0:
                _pw = float(_np.max(_np.abs(_fa + _fb))) / _sc
                if _pw <= 1e-6:
                    if numbers is not None:
                        numbers.update({"rel": [0.0] * int(_fa.shape[1]),
                                        "rtol": float(rtol), "support": _support_kind,
                                        "n_points": int(len(ca)), "tautology": True,
                                        "pointwise_agreement": _pw})
                    return [
                        f"Interface flux balance NOT CHECKED (the two exports cannot "
                        f"disagree): at every one of the {len(ca)} shared interface "
                        f"points the two exported fluxes are exact opposites, to "
                        f"{_pw:.1e} relative -- closer than two independent solutions "
                        f"ever agree. That is the shape of a Neumann side whose "
                        f"consistent flux recovery returns the flux it was given (or "
                        f"of a re-export), so it is no conservation evidence either "
                        f"way. Conservation across this interface is judged on the "
                        f"flux each side recovers from its OWN field at the task's "
                        f"interface points -- the per-level interface deliverables, "
                        f"which audit_results compares channel by channel and level "
                        f"by level."]

    def _flux(e, co):
        f = e.get("normal_fluxes") if isinstance(e, dict) else getattr(e, "normal_fluxes", None)
        if f is None:
            return None, None
        a = _np.asarray(f, float)
        # A VECTOR interface flux (traction, momentum) must balance component by
        # component. Summing every component into one number lets a +x imbalance
        # cancel a -y one and report perfect conservation across an interface
        # that conserves nothing.
        v = (a.reshape(-1, a.shape[-1]) if a.ndim >= 2 and a.shape[-1] > 1
             else a.reshape(-1, 1))
        # THE SCALE OF A COMPONENT IS THE FLUX IT CARRIES, NOT ITS NET. A
        # tangential traction that changes sign along the interface (a
        # shear that is antisymmetric about the mid-point, measured on a
        # thermo-mechanical pair) sums to ~0 on both sides, and a ratio of
        # two nets that are both noise read 100% on a coupling that was
        # right. Each component's L1 magnitude rides along and becomes the
        # floor its net imbalance is judged against.
        if not _integrate:
            _l1[id(e)] = _np.abs(v).sum(axis=0)
            return v.sum(axis=0), None
        if co is None or len(co) != len(v):
            _l1[id(e)] = _np.abs(v).sum(axis=0)
            return (v.sum(axis=0),
                    "the two sides sample the interface differently and there "
                    "are no usable coordinates to integrate against")
        wts, dim, detail = interface_nodal_weights(co)
        if wts is None:
            _l1[id(e)] = _np.abs(v).sum(axis=0)
            return v.sum(axis=0), detail
        _l1[id(e)] = (wts[:, None] * _np.abs(v)).sum(axis=0)
        return (wts[:, None] * v).sum(axis=0), None

    _l1: dict = {}
    va, na = _flux(export_a, ca)
    vb, nb = _flux(export_b, cb)
    _l1a, _l1b = _l1.get(id(export_a)), _l1.get(id(export_b))
    if va is None or vb is None:
        # NOT CHECKED IS NOT PASSED.
        #
        # This returned `w` — empty — when a side exported no normal_fluxes,
        # so a coupling with NO conservation evidence produced exactly the same
        # finding list as one that conserves perfectly. The docstring above
        # already says an empty list "must never be read as conservation was
        # checked and is fine", and delegates the question to
        # `interface_balance_coverage`. That function does not exist: it is
        # named in that sentence and nowhere else in the tree. So the caveat
        # was never enforced anywhere, and every caller that asked only this
        # function got silence.
        #
        # Fourteen lines below, the SAME function reports "could NOT be
        # evaluated" as a finding when the two sides sample the interface
        # differently, with a comment saying that is better than "silently
        # returning []". This is that principle applied to the case it was
        # skipped for.
        missing = [lbl for lbl, v in ((label_a, va), (label_b, vb)) if v is None]
        return w + [
            f"Interface flux balance NOT CHECKED: "
            f"{' and '.join(missing)} exported no `normal_fluxes`, so there is "
            f"no conservation evidence for this coupling at all. This is not a "
            f"passing balance check — nothing was compared. Export the outward "
            f"normal flux from both participants, or state explicitly that "
            f"this coupling is not conservative by construction."]

    # THE SIBLINGS OF THE ABOVE, which the missing-key check does not cover.
    #
    # `is None` catches an ABSENT key. It does not catch a key that is present
    # and carries nothing, or carries nothing but zeros. Both of those sum to
    # zero on each side, the relative imbalance is 0/floor = 0, and the
    # function returned [] — indistinguishable from a perfectly conserving
    # coupling, and unlike the absent-key case with no coverage note either.
    #
    # These are not exotic. An empty array is a participant that wrote the key
    # and no data. All-zeros is the signature of a transfer that never ran, a
    # misspelled field name, or a buffer that was allocated and never filled.
    # Reporting those as "checked and balanced" is the whole defect this
    # function was audited for.
    def _n(v):
        try:
            return int(_np.asarray(v).size)
        except (TypeError, ValueError):
            return 0

    if _n(va) == 0 or _n(vb) == 0:
        empty = [lbl for lbl, v in ((label_a, va), (label_b, vb)) if _n(v) == 0]
        return w + [
            f"Interface flux balance NOT CHECKED: "
            f"{' and '.join(empty)} exported the `normal_fluxes` key with NO "
            f"data in it, so nothing was compared. An empty export is not a "
            f"zero flux — it is a participant that wrote the field and never "
            f"filled it."]

    if not _np.any(_np.abs(_np.asarray(va, float)) > 0) and \
       not _np.any(_np.abs(_np.asarray(vb, float)) > 0):
        return w + [
            f"Interface flux balance NOT CHECKED: both {label_a} and "
            f"{label_b} exported fluxes that are identically zero. That "
            f"balances trivially and says nothing about conservation — it is "
            f"the signature of a transfer that never ran, a wrong field name, "
            f"or an unfilled buffer. Confirm a non-zero flux is actually "
            f"being exchanged before reading any balance verdict."]
    # A FALLBACK TO THE PLAIN SUM ON DIFFERENT DISCRETISATIONS IS NOT A CHECK.
    # It compares 41 samples against 31 and its verdict is set by the meshes.
    # Reported as a finding, the way this function already reports a component
    # count mismatch and a non-finite net, rather than silently returning [].
    if na or nb:
        why = "; ".join(f"{lbl}: {m}" for lbl, m in ((label_a, na), (label_b, nb)) if m)
        return [f"Interface flux balance could NOT be evaluated: the two "
                f"participants sample the interface differently, so only an "
                f"integral over it would be comparable, and no quadrature could "
                f"be formed ({why}). Conservation is UNCHECKED — the plain sums "
                f"would have been set by the two meshes rather than by the "
                f"physics."]
    if va.shape != vb.shape:
        return [f"Interface flux balance could NOT be evaluated: {label_a} exports "
                f"{va.size} flux component(s) and {label_b} exports {vb.size}. "
                "Conservation is unchecked."]
    if va.size > 1:
        # ONE floor for the whole interface, from the largest component either
        # side actually exchanges. Without it each component is judged against
        # its own magnitude alone and a both-sides-zero component fails on
        # roundoff — see the `floor` note in the docstring.
        comp_floor = 1e-6 * max(float(_np.max(_np.abs(va[_np.isfinite(va)])))
                                if _np.any(_np.isfinite(va)) else 0.0,
                                float(_np.max(_np.abs(vb[_np.isfinite(vb)])))
                                if _np.any(_np.isfinite(vb)) else 0.0,
                                floor)
        if _l1a is not None and _l1b is not None:
            # the whole-interface floor also rides on the largest L1 magnitude,
            # so a component that is dead on both sides is still judged against
            # the flux the interface actually carries
            comp_floor = max(comp_floor, 1e-6 * float(max(_np.max(_l1a), _np.max(_l1b))))
        out = []
        rels = []
        for c in range(va.size):
            _nc: dict = {}
            _fl = comp_floor
            if _l1a is not None and _l1b is not None and c < len(_l1a) and c < len(_l1b):
                _fl = max(comp_floor, float(_l1a[c]), float(_l1b[c]))
            out += [m + (_support_note if m.startswith("Interface flux NOT balanced") else "")
                    for m in _interface_balance_core(
                        {"normal_fluxes": [float(va[c])]}, {"normal_fluxes": [float(vb[c])]},
                        f"{label_a}[{c}]", f"{label_b}[{c}]", rtol, _fl,
                        numbers=_nc)]
            rels.append(float(_nc.get("rel", [float("nan")])[0]))
        if numbers is not None:
            numbers.update({"rel": rels, "rtol": float(rtol),
                            "support": _support_kind,
                            "n_points": int(len(ca)) if ca is not None else None})
        return out
    fa, fb = float(va[0]), float(vb[0])
    if _l1a is not None and _l1b is not None and len(_l1a) == 1 and len(_l1b) == 1:
        floor = max(floor, float(_l1a[0]), float(_l1b[0]))
    # A non-finite net flux makes every comparison below False (nan > rtol is
    # False), so without this the check would report nothing at all on the most
    # broken data it can be handed.
    if not (_np.isfinite(fa) and _np.isfinite(fb)):
        return [f"Interface flux balance could NOT be evaluated: net({label_a})={fa}, "
                f"net({label_b})={fb} — a non-finite exchanged flux. Conservation "
                "is unchecked and the exchanged data is invalid."]
    denom = max(abs(fa), abs(fb), floor, 1e-30)
    rel = abs(fa + fb) / denom            # A exports +flux, B imports -flux → sum≈0
    if numbers is not None:
        numbers.update({"rel": [float(rel)], "rtol": float(rtol),
                        "support": _support_kind,
                        "n_points": int(len(ca)) if ca is not None else None,
                        "net": [[fa, fb]]})
    if rel > rtol:
        # Name the convention. The most common cause of this warning is not a
        # non-conservative coupling but both sides exporting their flux with
        # the SAME sign — which a correct coupling does if nobody said which
        # normal to use. Saying only "not balanced" sends the agent looking for
        # a physics bug that is not there.
        same_sign = fa * fb > 0 and abs(abs(fa) - abs(fb)) / denom <= rtol
        hint = (" The two magnitudes match but the signs agree, which is the "
                "signature of a SIGN-CONVENTION error rather than a "
                "conservation error: each participant must export the flux "
                "through the interface with respect to ITS OWN outward normal, "
                "and those normals are anti-parallel, so the two sums should "
                "cancel. Note this is the opposite of the BC value you APPLY, "
                "which is the same number on both sides."
                if same_sign else "")
        if not hint:
            # A CLEAN POWER OF TEN (or 60 / 3600) BETWEEN THE TWO MAGNITUDES is
            # named before the arrival test below: a side whose load never
            # arrived returns a flux set by roundoff, not one sitting exactly
            # a decade under its partner, so the ratio is the more specific
            # signal and wins when both would fire.
            hint = _unit_ratio_hint(fa, fb)
        if not hint:
            # ONE SIDE'S FLUX IS ~ZERO AGAINST A NONZERO PARTNER: the
            # imported interface load never entered that side's assembled
            # system (measured signature: a coupling that converges cleanly
            # while the receiving side keeps returning its no-load
            # solution). This is an ARRIVAL failure, not a conservation
            # error, and it has a one-step test and an exact per-code cure
            # in the receiving side's own served deciding facts.
            _mag_a, _mag_b = abs(fa), abs(fb)
            _big = max(_mag_a, _mag_b)
            if _big > 0 and min(_mag_a, _mag_b) < 0.02 * _big:
                _zero = label_a if _mag_a < _mag_b else label_b
                hint = (f" Side {_zero}'s net flux is ~ZERO against a "
                        f"nonzero partner: the imported load likely never "
                        f"entered {_zero}'s system (a run that converges "
                        f"while one side returns its no-load solution). "
                        f"Test it in one step: solve that side once with "
                        f"the import zeroed and once with the real import "
                        f"-- if the two fields match, the load is not "
                        f"being applied. The exact condition syntax for "
                        f"applying a sampled interface load is in that "
                        f"side's served deciding facts (prepare_simulation "
                        f"for that solver).")
        w.append(
            f"Interface flux NOT balanced: net({label_a})={fa:.4g}, net({label_b})={fb:.4g}, "
            f"imbalance {rel:.1%} > {rtol:.0%} — coupling may be non-conservative (silent error)."
            + _support_note + hint
        )
    return w


ENDS_ONLY_MARK = "Interface flux imbalance at the interface ENDS only"


def _bent_ends(pts):
    """(first, last): the rows of `pts` at the two ends of a BENT curve, taken
    along the curve; None for a straight set, or one that is not one curve (the
    callers then take the two extremes along the longest axis, which is right
    only on a straight interface: on a bent one, the extremes of one coordinate
    are a whole leg)."""
    try:
        p = _np.atleast_2d(_np.asarray(pts, float))
        sv = _np.linalg.svd(p - p.mean(axis=0), compute_uv=False)
    except (TypeError, ValueError, _np.linalg.LinAlgError):
        return None
    if p.shape[0] < 3 or sv.size < 2 or sv[0] <= 0 or sv[1] <= 1e-8 * sv[0]:
        return None
    cur, _why = interface_curve(p)
    if cur is None or cur["closed"]:
        return None
    return int(cur["order"][0]), int(cur["order"][-1])


def _interface_without_ends(export_a, export_b):
    """(interior_a, interior_b): both exports on their aligned points minus the
    two extreme points along the interface; None when the two sides are not
    aligned point for point or hold fewer than four shared points."""
    def _co(e):
        c = e.get("coordinates") if isinstance(e, dict) else getattr(e, "coordinates", None)
        try:
            a = _np.asarray(c, float)
        except (TypeError, ValueError):
            return None
        return _np.atleast_2d(a) if c is not None and a.ndim >= 1 else None
    ca, cb = _co(export_a), _co(export_b)
    if ca is None or cb is None:
        return None
    if ca.shape == cb.shape and _np.allclose(ca, cb, rtol=1e-9, atol=1e-12):
        ia = ib = list(range(len(ca)))
    else:
        shared = _common_support(ca, cb)
        if shared is None:
            return None
        ia, ib = shared
    if len(ia) < 4:
        return None
    pts = ca[ia]
    ends = _bent_ends(pts)
    if ends is not None:
        lo, hi = ends
    else:
        ax = int(_np.argmax(_np.var(pts, axis=0)))
        lo, hi = int(_np.argmin(pts[:, ax])), int(_np.argmax(pts[:, ax]))
    keep = [k for k in range(len(ia)) if k not in (lo, hi)]
    ra = _restrict_export(export_a, [ia[k] for k in keep])
    rb = _restrict_export(export_b, [ib[k] for k in keep])
    if "normal_fluxes" not in ra or "normal_fluxes" not in rb:
        return None
    return ra, rb


def _leg_nets(export_a, export_b):
    """Per straight leg of a BENT interface, the net flux each side carries through the
    stretch of the leg both sides sample: {"legs": [{"c", "t", "net_a", "net_b"}], "l1_a",
    "l1_b"} (per flux component; l1 is the integral of |q| over all those stretches).
    Each side is integrated from its own points on the leg with its own trapezoid weights
    along it, over the common stretch only (never beyond a side's last point there), the
    two ends of the interface left out (see check_interface_balance). None where the
    interface is straight, is not one curve in a plane, or the fluxes cannot be read."""
    def _get(e, k):
        return e.get(k) if isinstance(e, dict) else getattr(e, k, None)
    ca, cb = _points(_get(export_a, "coordinates")), _points(_get(export_b, "coordinates"))
    if ca is None or cb is None:
        return None
    got = _shared_columns(ca, cb)
    if got is None:
        return None
    ca, cb = got
    try:
        fa = _np.asarray(_get(export_a, "normal_fluxes"), float)
        fb = _np.asarray(_get(export_b, "normal_fluxes"), float)
    except (TypeError, ValueError):
        return None
    if (fa.size == 0 or fb.size == 0 or fa.size % len(ca) or fb.size % len(cb)
            or fa.size // len(ca) != fb.size // len(cb)):
        return None
    fa, fb = fa.reshape(len(ca), -1), fb.reshape(len(cb), -1)
    if not (_np.all(_np.isfinite(fa)) and _np.all(_np.isfinite(fb))):
        return None
    U = _np.unique(_np.vstack([ca, cb]), axis=0)
    if U.shape[1] == 3:
        sc = float(_np.max(_np.ptp(U, axis=0))) or 1.0
        flat = [j for j in range(3) if float(_np.ptp(U[:, j])) <= 1e-12 * sc]
        if not flat:
            return None
        keep = [j for j in range(3) if j != flat[-1]]
    elif U.shape[1] == 2:
        keep = [0, 1]
    else:
        return None
    ca, cb, U = ca[:, keep], cb[:, keep], U[:, keep]
    cur, _why = interface_curve(U)
    if cur is None:
        return None
    legs = [g for g in cur["legs"] if g["t"] is not None and g["straight"]]
    if len(legs) < 2:
        return None
    ends = [U[cur["order"][0]], U[cur["order"][-1]]]
    span = float(_np.max(_np.ptp(U, axis=0))) or 1.0
    tol = 1e-6 * span

    def _on(P, F, g, lo, hi):
        d = P - g["c"]
        s, off = d @ g["t"], _np.abs(d @ g["n"])
        m = (off <= 1e-4 * span) & (s >= lo - tol) & (s <= hi + tol)
        m &= ~_np.array([any(float(_np.linalg.norm(p - e)) <= tol for e in ends) for p in P])
        o = _np.argsort(s[m], kind="stable")
        return s[m][o], F[m][o]

    def _integral(s, f, lo, hi):
        br = _np.unique(_np.concatenate([[lo, hi], s[(s > lo) & (s < hi)]]))
        vals = _np.column_stack([_np.interp(br, s, f[:, c]) for c in range(f.shape[1])])
        seg = _np.diff(br)[:, None]
        return (0.5 * seg * (vals[:-1] + vals[1:])).sum(axis=0), \
               (0.5 * seg * (_np.abs(vals[:-1]) + _np.abs(vals[1:]))).sum(axis=0)
    out, l1a, l1b = [], _np.zeros(fa.shape[1]), _np.zeros(fa.shape[1])
    for g in legs:
        a = (U[g["idx"]] - g["c"]) @ g["t"]
        sa, qa = _on(ca, fa, g, float(a.min()), float(a.max()))
        sb, qb = _on(cb, fb, g, float(a.min()), float(a.max()))
        if len(sa) < 2 or len(sb) < 2:
            continue
        lo, hi = max(float(sa[0]), float(sb[0])), min(float(sa[-1]), float(sb[-1]))
        if hi - lo <= tol:
            continue
        na, la = _integral(sa, qa, lo, hi)
        nb, lb = _integral(sb, qb, lo, hi)
        l1a, l1b = l1a + la, l1b + lb
        out.append({"c": g["c"], "t": g["t"], "net_a": na, "net_b": nb})
    if len(out) < 2:
        return None
    return {"legs": out, "l1_a": l1a, "l1_b": l1b}


def check_interface_balance(export_a, export_b, label_a="A", label_b="B",
                            rtol: float = 0.05, floor: float = 0.0,
                            numbers: dict | None = None) -> list[str]:
    """The whole-interface balance (see _check_interface_balance_whole) and, on a BENT
    interface, the balance of each leg.

    A NET THAT SHRINKS WHILE EACH LEG'S GROWS IS A REDISTRIBUTION. Measured on a coupled
    run whose Neumann side applied its partner's flux by row index: the whole-interface
    imbalance read 9.1 % -> 3.5 % -> 1.3 % and the ladder called it discretisation
    error, while the two legs' nets read 15 % -> 18 % -> 22 % and 6 % -> 14 % -> 21 % of
    the flux the interface carries -- flux one side put on one leg, the other put on the
    other. Each leg is balanced with its own weights (_leg_nets), against the flux the
    whole interface carries so that a leg carrying little cannot inflate its ratio; the
    worst leg rides in numbers["legs"] so the ladder states its trend.
    """
    w = _check_interface_balance_whole(export_a, export_b, label_a, label_b, rtol, floor, numbers)
    if any(("NOT CHECKED" in str(m)) or ("could NOT be evaluated" in str(m)) for m in w):
        return w
    try:
        ln = _leg_nets(export_a, export_b)
    except Exception:                                        # noqa: BLE001
        ln = None
    if not ln:
        return w
    den = _np.maximum(_np.maximum(ln["l1_a"], ln["l1_b"]), 1e-300)
    rels = [(_np.abs(g["net_a"] + g["net_b"]) / den) for g in ln["legs"]]
    worst = [float(max(r[c] for r in rels)) for c in range(len(den))]
    if numbers is not None:
        numbers["legs"] = {"rel": worst, "per_leg": [[float(v) for v in r] for r in rels],
                           "rtol": float(rtol)}
    if max(worst) <= rtol:
        return w
    c = int(_np.argmax(worst))
    j = int(_np.argmax([r[c] for r in rels]))
    g = ln["legs"][j]
    whole = (numbers or {}).get("rel")
    comp = f" (component {c})" if len(den) > 1 else ""
    return list(w) + [
        f"Interface flux NOT balanced on one leg of the bent interface{comp}: through the leg "
        f"through ({', '.join(f'{v:.4g}' for v in g['c'])}) {label_a} carries {float(g['net_a'][c]):.4g} "
        f"and {label_b} {float(g['net_b'][c]):.4g}, an imbalance of {worst[c]:.1%} of the flux the "
        f"whole interface carries > {rtol:.0%}"
        + (f", while the whole interface reads {float(whole[c]):.1%}" if whole and c < len(whole) else "")
        + ". Flux one side puts on one leg and the other puts on another cancels in the "
        "whole-interface sum. At one level this cannot tell discretisation from a mis-mapped "
        "exchange; the trend over levels can."]


def _check_interface_balance_whole(export_a, export_b, label_a="A", label_b="B",
                                   rtol: float = 0.05, floor: float = 0.0,
                                   numbers: dict | None = None) -> list[str]:
    """The conservation check of `_interface_balance_core`, plus one diagnosis.

    AN IMBALANCE THAT SITS AT THE TWO ENDS OF THE INTERFACE IS NAMED AS SUCH, and
    no cause is asserted for it: a served cause ("a corner reaction mixes the
    outer condition in") was false on a later round whose end rows were copies
    of their neighbours. Measured on a correct vector coupling: the
    ladder was told "NOT VERIFIED ... a wrong sign, scaling or missing term" for
    21.6% -> 20.0% -> 17.4%, all of it at two corner rows; on the interior points
    the two exports agreed to 6e-8..4e-7. So when the whole interface fails, the
    interior is judged alone: if it holds (or is the Dirichlet-Neumann tautology),
    the finding names the ends and goes to the coverage channel, and the interior's
    numbers are what the ladder reads. The ends are never dropped from a balance
    that holds -- a correct coupling can carry real flux through its end nodes.
    """
    w = _interface_balance_core(export_a, export_b, label_a, label_b, rtol, floor, numbers)
    if not any(str(m).startswith("Interface flux NOT balanced") for m in w):
        return w
    inner = _interface_without_ends(export_a, export_b)
    if inner is None:
        return w
    n2: dict = {}
    w2 = _interface_balance_core(inner[0], inner[1], label_a, label_b, rtol, floor, n2)
    if any(str(m).startswith("Interface flux NOT balanced") for m in w2):
        return w
    if numbers is not None:
        numbers.clear()
        numbers.update(n2)
        numbers["ends_only"] = True
    whole = "; ".join(str(m).replace("Interface flux NOT balanced: ", "").split(" — ")[0]
                      for m in w if str(m).startswith("Interface flux NOT balanced"))[:300]
    # "WHICH THE EQUATION CHECK JUDGES" WAS FALSE OF A WRONG FIELD. Measured: a side
    # whose interface nodes held 0.0 while its exports carried the partner's trace
    # read this note at all three levels, passed the equation check (which judges the
    # interior only), and handed in "the flux mismatch is likely the corner handling".
    return [f"{ENDS_ONLY_MARK}: the whole interface reads {whole}, but on the interior points the two "
            f"sides' EXPORTS agree, so the imbalance in the exports sits at the two end points. That "
            f"is all this measures. On a Dirichlet-Neumann pair the interior agreement of the exports "
            f"holds by construction and says nothing about the fields themselves: whether a side's "
            f"field carries what it exports is judged by the checks that read the fields (its own "
            f"field dump against its interface file, its flux against its own field). Where those "
            f"hold, look at how each side treats the two nodes where the interface meets the outer "
            f"boundary."] + list(w2)


def check_interface_flux_profile(export_a, export_b, label_a="A", label_b="B",
                                 rtol: float = 0.10, numbers: dict | None = None,
                                 ends=None) -> tuple[list[str], list[str]]:
    """Does the flux match POINT BY POINT, not only in total?

    The net balance is a single number, and a single number is easy to satisfy
    by accident: one side can pile a large flux onto one node and take it off
    the others, and the two totals still cancel exactly while the two sides
    disagree about the interface everywhere. Demonstrated against this file —
    A exporting a uniform +0.67 against B exporting (-26.7, +8, +8, +8) summed
    to zero and was reported as conserved.

    THE POINTS ARE PAIRED BY WHERE THEY ARE (see pair_interface_points). Where
    the two sides list the same points in any order, or one side's points are
    among the other's, the flux is compared at the points both list. Where they
    sample the interface at different points, each point of the side that lists
    fewer is compared with the other side's flux read along the straight leg it
    lies on, never beyond that side's last point on the leg. Only where neither
    can be done does this say so, instead of passing.

    `ends`, where the caller knows them, are the two points where a straight interface ends on
    the sides' own meshes; only rows at those points are then left out (see below).
    """
    findings: list[str] = []

    def _get(e, k):
        v = e.get(k) if isinstance(e, dict) else getattr(e, k, None)
        return None if v is None else _np.asarray(v, float)

    _only_total = ("The net balance is then the ONLY conservation evidence, and it is a "
                   "single number that a wrong distribution can satisfy.")
    fa, fb = _get(export_a, "normal_fluxes"), _get(export_b, "normal_fluxes")
    if fa is None or fb is None or fa.size == 0 or fb.size == 0:
        return findings, ["pointwise interface flux profile: at least one side "
                          "exported no `normal_fluxes`, so only the total could "
                          "have been compared, and here not even that"]
    ca = _points(export_a.get("coordinates") if isinstance(export_a, dict)
                 else getattr(export_a, "coordinates", None))
    cb = _points(export_b.get("coordinates") if isinstance(export_b, dict)
                 else getattr(export_b, "coordinates", None))
    if ca is None or cb is None:
        who = " and ".join(lbl for lbl, c in ((label_a, ca), (label_b, cb)) if c is None)
        return findings, [
            f"pointwise interface flux profile: {who} exported no usable interface "
            f"coordinates, so its rows cannot be placed along the interface and the "
            f"flux could only be compared in total. " + _only_total]
    if fa.size % len(ca) or fb.size % len(cb) or fa.size // len(ca) != fb.size // len(cb):
        return findings, [
            f"pointwise interface flux profile: {label_a} exports {fa.size} flux number(s) "
            f"at {len(ca)} point(s) and {label_b} {fb.size} at {len(cb)}, so the flux rows "
            f"do not line up with the points or carry different components. " + _only_total]
    A_all, B_all = fa.reshape(len(ca), -1), fb.reshape(len(cb), -1)
    _ro = _exchange_at_roundoff(export_a, export_b)
    if _ro is not None:
        return findings, [
            f"pointwise interface flux profile: NOT CHECKED, the exchange is at round-off "
            f"(the largest flux either side exported is {_ro[0]:.1e}, against an interface "
            f"trace of {_ro[1]:.3g}), so there is no flux profile to compare point by point"]
    if not (_np.all(_np.isfinite(A_all)) and _np.all(_np.isfinite(B_all))):
        return findings, []
    cor = interface_correspondence(ca, cb)
    if cor.get("kind") is None:
        return findings, [f"pointwise interface flux profile: {cor.get('why')}, so the flux "
                          f"could only be compared in total. " + _only_total]
    ia, ib = list(cor["ia"]), list(cor["ib"])
    if len(ia) >= 3 and (not cor["only_a"] or not cor["only_b"]):
        # the points both sides list, whatever order each lists them in
        A, B, P = A_all[ia], B_all[ib], ca[ia]
        how = f"at the {len(ia)} interface points both sides export"
    else:
        # two samplings: the side with fewer points, against the other's flux along its legs
        few_a = len(cor["reps_a"]) <= len(cor["reps_b"])
        (cs, Fs, reps_s), (cd, Fd) = (((ca, A_all, cor["reps_a"]), (cb, B_all)) if few_a
                                      else ((cb, B_all, cor["reps_b"]), (ca, A_all)))
        profs = leg_profiles(cd)
        got_s, got_d, got_p = [], [], []
        for r in reps_s:
            j = on_one_leg(cs[r], profs) if profs else None
            if j is None:
                continue
            g = profs[j]
            s = float((cs[r][:2] - g["c"]) @ g["t"])
            if not (g["a"][0] - g["tol"] <= s <= g["a"][-1] + g["tol"]):
                continue
            got_s.append(Fs[r])
            got_d.append([_np.interp(s, g["a"], Fd[g["rows"], c]) for c in range(Fd.shape[1])])
            got_p.append(cs[r])
        if len(got_s) < 3:
            _why = (cor.get("why") or ("the side with more points does not form straight legs in a "
                                       "plane" if not profs else
                                       "fewer than three points of the side with fewer lie on a "
                                       "straight leg of the other inside its span"))
            return findings, [
                f"pointwise interface flux profile: the two sides sample the interface at "
                f"different points and {_why}, so the flux could only be compared in total. "
                + _only_total]
        Fs_, Fd_ = _np.asarray(got_s, float), _np.asarray(got_d, float)
        A, B = (Fs_, Fd_) if few_a else (Fd_, Fs_)
        P = _np.asarray(got_p, float)
        how = (f"at {len(P)} of {label_a if few_a else label_b}'s interface points, with "
               f"{label_b if few_a else label_a}'s flux read along the interface there (the two "
               f"sides sample it at different points)")
    # THE TWO END POINTS ARE LEFT OUT. Where the interface meets the outer
    # boundary the served contracts treat the flux differently by design -- one
    # copies the nearest interior value, another zeroes a corner that mixes the
    # outer reaction -- so the end rows measured the convention, not the
    # exchange (measured: they alone drove a "does NOT shrink" verdict whose
    # interior read 6.2 / 11.7 / 5.3 %). The ends are named by the balance check.
    # THE ENDS ARE WHERE THE INTERFACE ENDS, NOT THE FIRST AND LAST ROWS. Where both sides leave
    # the end nodes out of their lists, the first and last rows are the nodes NEXT to the ends,
    # and leaving them out hid the one defect there: measured on a coupled round, a Neumann side
    # that built its load from a list without the end vertices carried no flux on the two end
    # segments, its flux missed its partner's by 16-19 % at the end-adjacent nodes at every
    # level, and this check passed every level. Where the caller knows the ends (the sides' own
    # mesh dumps), only rows at them are left out.
    if len(A) >= 4:
        _ends = _bent_ends(P)
        _at = None
        if _ends is None and ends is not None:
            try:
                _E = _np.atleast_2d(_np.asarray(ends, float))[:, :P.shape[1]]
                _span = float(_np.ptp(P, axis=0).max()) or 1.0
                _at = [i for i in range(len(P))
                       if float(_np.min(_np.linalg.norm(_E - P[i], axis=1))) <= 1e-6 * _span]
            except (TypeError, ValueError):
                _at = None
        if _at is not None:
            _drop = set(_at)
        elif _ends is not None:
            _drop = set(_ends)
        else:
            _ax = int(_np.argmax(_np.var(P, axis=0)))
            _drop = {int(_np.argmin(P[:, _ax])), int(_np.argmax(P[:, _ax]))}
        _keep = [i for i in range(len(A)) if i not in _drop]
        A, B, P = A[_keep], B[_keep], P[_keep]
    # PER COMPONENT, for a vector interface flux. One scale taken over the whole
    # array is set by the largest component, so a tangential traction that is
    # two orders of magnitude below the normal one can be 100% wrong and read as
    # 1% of "the interface scale". Each component is judged against its own
    # magnitude instead — with a floor at 1e-9 of the largest component anywhere
    # on the interface, so a component that is legitimately zero on both sides
    # is not condemned on roundoff (the same trap the net balance's `floor`
    # exists for).
    S = A + B
    everywhere = max(float(_np.max(_np.abs(A))), float(_np.max(_np.abs(B))))
    worsts = []
    for c in range(S.shape[1]):
        scale = max(float(_np.max(_np.abs(A[:, c]))),
                    float(_np.max(_np.abs(B[:, c]))),
                    1e-9 * everywhere, 1e-30)
        worst = float(_np.max(_np.abs(S[:, c]))) / scale
        worsts.append(worst)
        if worst <= rtol:
            continue
        i = int(_np.argmax(_np.abs(S[:, c])))
        comp = f" component [{c}]" if S.shape[1] > 1 else ""
        where = ", ".join(f"{v:.4g}" for v in P[i])
        findings.append(
            f"Interface flux{comp} does NOT match POINT BY POINT, compared {how}: worst at "
            f"({where}) with {label_a}={A[i, c]:.4g} and {label_b}={B[i, c]:.4g} "
            f"(they should cancel), off by {worst:.1%} of that component's own "
            f"interface scale > {rtol:.0%}. The TOTALS may still balance — a "
            "redistribution along the interface cancels in the sum. At one level "
            "this cannot tell discretisation from a mis-mapped exchange: on a "
            "Dirichlet-Neumann pair the Neumann side's consistent recovery "
            "smooths the flux it was given by the boundary mass matrix, which "
            "shrinks under refinement, while a mis-mapped exchange does not.")
    if numbers is not None:
        numbers.update({"rel": worsts, "rtol": rtol, "n_points": int(len(A))})
    return findings, []


def check_interfaces_are_the_same_surface(export_a, export_b, label_a="A",
                                          label_b="B") -> tuple[list[str], list[str]]:
    """Are the two participants even talking about the same piece of geometry?

    Non-matching discretisation is routine. Two interfaces that do not OVERLAP
    at all are not a discretisation difference — they are two different surfaces,
    and every number exchanged between them is meaningless. Nothing else here
    looks at where the interface is: a B whose interface sat five metres away
    from A's was coupled, converged and stamped trustworthy.

    This can only see what the participants declare. A participant that reports
    the right coordinates for the wrong surface is beyond any check at this level
    — that is what the monolithic comparison is for.
    """
    def _co(e):
        c = e.get("coordinates") if isinstance(e, dict) else getattr(e, "coordinates", None)
        return None if c is None else _np.atleast_2d(_np.asarray(c, float))

    ca, cb = _co(export_a), _co(export_b)
    # The limit belongs in the SERVED coverage, not only in this docstring: an
    # agent reads the verdict, never the source. Stating it on every run is the
    # point — it is unconditional, and a reader who is told the interfaces
    # overlap would otherwise take that for "the right surface was used".
    _WRONG_SURFACE_LIMIT = (
        "interface identity: openPASO compared the coordinates the two participants "
        "REPORTED and they describe the same region of space. It cannot check "
        "that those coordinates are the surface each participant actually "
        "applied its boundary condition on — a participant that reports the "
        "right coordinates for the wrong surface is not detectable at this "
        "level, by any check here. Only the `monolithic` comparison can catch "
        "it.")
    if ca is None or cb is None or ca.size == 0 or cb.size == 0:
        return [], ["interface geometry: coordinates were not exported, so "
                    "whether the two sides describe the same surface is unknown"]
    if ca.shape[1] != cb.shape[1]:
        # A COORDINATE ONE SIDE ADDS AND HOLDS CONSTANT CARRIES NO GEOMETRY: a 2-D
        # side written as (x, y, 0) and its partner written as (x, y) describe the
        # same interface, and the column count alone read "not the same surface".
        # The two are compared in the columns they share when every extra column
        # is constant; an extra column that varies is a different dimension.
        _w, _n = (ca, cb) if ca.shape[1] > cb.shape[1] else (cb, ca)
        _extra = _w[:, _n.shape[1]:]
        _sc = float(_np.max(_np.ptp(_w, axis=0))) or 1.0
        if _n.shape[1] < 1 or float(_np.max(_np.ptp(_extra, axis=0))) > 1e-12 * _sc:
            return ([f"Interface geometry mismatch: {label_a} exports "
                     f"{ca.shape[1]}-D coordinates and {label_b} exports "
                     f"{cb.shape[1]}-D. These are not the same surface."], [])
        ca, cb = ca[:, :_n.shape[1]], cb[:, :_n.shape[1]]
    lo_a, hi_a = ca.min(axis=0), ca.max(axis=0)
    lo_b, hi_b = cb.min(axis=0), cb.max(axis=0)
    span = _np.maximum(hi_a - lo_a, hi_b - lo_b)
    tol = _np.maximum(span * 0.05, 1e-9)
    gap = _np.maximum(lo_a - hi_b, lo_b - hi_a)      # >0 in a disjoint direction
    if _np.any(gap > tol):
        d = int(_np.argmax(gap - tol))
        return ([f"Interfaces do NOT overlap: along axis {d}, {label_a} spans "
                 f"[{lo_a[d]:.4g}, {hi_a[d]:.4g}] and {label_b} spans "
                 f"[{lo_b[d]:.4g}, {hi_b[d]:.4g}] — a gap of {gap[d]:.4g}. These "
                 "are two different surfaces, so every value exchanged between "
                 "them was mapped onto geometry it does not belong to."], [])
    # OVERLAPPING BOXES ARE NOT ONE INTERFACE. On a bent interface the box of one
    # leg lies inside the box of both, so a side that listed one leg of two passed
    # (measured on two recorded sides). Where the interface is a curve, each
    # side's points are measured against the other's curve, both ways.
    cov_f, cov_n = _curve_coverage(ca, cb, label_a, label_b)
    if cov_f:
        return cov_f, []
    return [], cov_n + [_WRONG_SURFACE_LIMIT]


_COVERED = 0.9       # a side whose points lie along less of its partner's curve is named


def _covered_along(cx, cur_x, cur_y):
    """(fraction of X's curve length within about one point spacing of Y's curve,
    that uncovered length, X's length, the largest distance from Y's curve)."""
    Y = cur_y["uv"][cur_y["order"]]
    A, B = Y[:-1], Y[1:]
    ln = _np.linalg.norm(B - A, axis=1)
    ok = ln > 0
    A, B, ln = A[ok], B[ok], ln[ok]
    X = cur_x["uv"]
    if not len(ln):
        return None
    sx = cur_x["s"]
    ox = cur_x["order"]
    step = _np.diff(sx[ox])
    loc_sorted = _np.maximum(_np.concatenate([[step[0]], step]) if len(step) else [0.0],
                             _np.concatenate([step, [step[-1]]]) if len(step) else [0.0])
    loc = _np.empty(len(X))
    loc[ox] = loc_sorted
    dmin = _np.empty(len(X))
    spac = _np.empty(len(X))
    for a in range(0, len(X), 400):
        P = X[a:a + 400]
        d = P[:, None, :] - A[None, :, :]
        t = _np.clip(_np.einsum("pkj,kj->pk", d, B - A) / ln ** 2, 0.0, 1.0)
        dist = _np.linalg.norm(d - t[:, :, None] * (B - A)[None, :, :], axis=2)
        k = _np.argmin(dist, axis=1)
        dmin[a:a + 400] = dist[_np.arange(len(P)), k]
        spac[a:a + 400] = ln[k]
    covered = dmin <= 1.5 * _np.maximum(spac, loc)
    w = _curve_weights(cur_x)
    tot = float(_np.sum(w))
    if tot <= 0:
        return None
    return (float(_np.sum(w[covered])) / tot, float(_np.sum(w[~covered])), tot,
            float(_np.max(dmin)))


def _curve_coverage(ca, cb, label_a, label_b):
    """(findings, not_checked): each side's points against the other's curve.

    Runs where the interface is a curve in a plane of the coordinates (a 2-D
    side, or a 3-D one written with a constant coordinate); a surface keeps the
    box test above alone. Coordinates of a 2-D side that do not form one curve
    are said to be unchecked, never passed."""
    plain_2d = ca.shape[1] == 2
    if ca.shape[1] == 3:
        # one frame for both: leave out a coordinate only where BOTH hold it constant
        both = _np.vstack([ca, cb])
        sc = float(_np.max(_np.ptp(both, axis=0))) or 1.0
        flat = [j for j in range(3) if float(_np.ptp(both[:, j])) <= 1e-12 * sc]
        if not flat:
            return [], []
        keep = [j for j in range(3) if j != flat[-1]]
        ca, cb = ca[:, keep], cb[:, keep]
    elif ca.shape[1] != 2:
        return [], []
    cur_a, why_a = interface_curve(ca)
    cur_b, why_b = interface_curve(cb)
    if cur_a is None or cur_b is None:
        if plain_2d:
            who, why = (label_a, why_a) if cur_a is None else (label_b, why_b)
            return [], [f"interface coverage: {who}'s interface points do not form "
                        f"one curve ({why}), so whether each side lists the whole "
                        f"interface was not checked"]
        return [], []
    out = []
    for (lx, cx, kx), (ly, cy, ky) in (((label_a, ca, cur_a), (label_b, cb, cur_b)),
                                       ((label_b, cb, cur_b), (label_a, ca, cur_a))):
        got = _covered_along(cx, kx, ky)
        if got is None:
            continue
        frac, unc, tot, dmax = got
        if frac < _COVERED:
            out.append(
                f"Interfaces overlap only in part: {ly}'s interface points lie along "
                f"only {frac:.0%} of {lx}'s interface. {unc:.4g} of {lx}'s interface "
                f"length {tot:.4g} lies farther than one point spacing from every "
                f"point {ly} exports, up to {dmax:.4g} away, so over that part "
                f"{lx}'s points have no partner points to exchange with. The two "
                f"sides of one coupling list the same interface, every leg of a "
                f"bent one.")
    return out, []


# Ratios that show up when two participants agree on the physics and disagree on
# the units. Naming the suspect turns "your coupling is non-conservative" (which
# sends the agent hunting a physics bug that is not there) into one thing to
# check. Deliberately conservative: only near-exact ratios, and always phrased as
# a candidate, never as a diagnosis.
_UNIT_RATIOS = [
    (1e3, "a 1000x factor — the classic W/mW, m/mm, kg/g, kPa/Pa mix-up"),
    (1e-3, "a 1000x factor — the classic W/mW, m/mm, kg/g, kPa/Pa mix-up"),
    (1e6, "a 1e6 factor — e.g. Pa/MPa, m^2/mm^2, W/uW"),
    (1e-6, "a 1e6 factor — e.g. Pa/MPa, m^2/mm^2, W/uW"),
    (1e9, "a 1e9 factor — e.g. Pa/GPa, m^3/mm^3"),
    (1e-9, "a 1e9 factor — e.g. Pa/GPa, m^3/mm^3"),
    (1e4, "a 1e4 factor"), (1e-4, "a 1e4 factor"),
    (60.0, "a factor of 60 — a per-second / per-minute rate mismatch"),
    (1 / 60.0, "a factor of 60 — a per-second / per-minute rate mismatch"),
    (3600.0, "a factor of 3600 — a per-second / per-hour rate mismatch"),
    (1 / 3600.0, "a factor of 3600 — a per-second / per-hour rate mismatch"),
]


def _unit_ratio_hint(fa: float, fb: float, rtol: float = 0.02) -> str:
    """Name a UNIT MISMATCH when the two net fluxes differ by a suspicious factor.

    A sign error makes the magnitudes match; a unit error makes them differ by a
    clean power of ten (or 60 / 3600). Both look identical to a plain imbalance
    number, and the second one converges to a confidently wrong answer.
    """
    if fa == 0.0 or fb == 0.0:
        return ""
    ratio = abs(fb) / abs(fa)
    for target, what in _UNIT_RATIOS:
        if abs(ratio / target - 1.0) <= rtol:
            return (f" The magnitudes differ by {what}, not by a small amount: "
                    "that is the signature of a UNIT MISMATCH between the two "
                    "participants rather than a conservation error. Check that "
                    "both sides express the exchanged quantity in the same units "
                    "before looking for a physics bug.")
    return ""


def check_monolithic_consistency(coupled_qoi: float, monolithic_qoi: float,
                                 rtol: float = 0.05, qoi: str = "QoI") -> list[str]:
    """If the same problem can be solved un-split in one code, the coupled answer must
    match it. The most decisive silent-wrong detector — needs no external benchmark,
    only a monolithic re-solve. Returns a warning if they disagree beyond rtol.

    Unit mismatches, a wrongly applied interface sign, a participant that never
    reads its imports and a lossy mesh mapping all end at the same place: a
    coupled number that is clean, converged and wrong. This is the only check in
    the file that compares that number against an independent answer to the same
    question, which is why `couple` reports loudly when it was not run.
    """
    w = []
    if monolithic_qoi is None or coupled_qoi is None:
        return w
    if not (_np.isfinite(coupled_qoi) and _np.isfinite(monolithic_qoi)):
        return [f"{qoi}: coupled={coupled_qoi} vs monolithic re-solve="
                f"{monolithic_qoi} — a non-finite value, so the two could not be "
                "compared and the coupled result is not corroborated."]
    denom = max(abs(monolithic_qoi), 1e-30)
    rel = abs(coupled_qoi - monolithic_qoi) / denom
    if rel > rtol:
        w.append(
            f"{qoi}: coupled={coupled_qoi:.5g} vs monolithic re-solve={monolithic_qoi:.5g} "
            f"differ by {rel:.1%} > {rtol:.0%} — the coupled result is likely WRONG."
        )
    return w


# ── coupling-machinery checks (consume the driver's recorded evidence) ────────
# Each one answers a question a partitioned coupling can otherwise get wrong
# while reporting a clean convergence. They return (findings, not_checked):
# findings flip the verdict, not_checked is reported so that "this check could
# not look at anything" is never silently indistinguishable from "this check
# looked and was happy".

def check_coupling_directionality(graph: dict, max_iter: int = 0
                                  ) -> tuple[list[str], list[str]]:
    """Is the coupling wired the way the caller thinks it is?

    A partitioned coupling is a directed graph, and the two ways it goes wrong
    are silent: a participant that declares no partner at all, and an edge to a
    partner whose name is misspelled. Both make the iteration a one-way transfer
    that converges quickly and looks excellent — the residual really is zero,
    because nothing is feeding back.

    A deliberate one-way transfer is declared by asking for a single pass
    (max_iter=1). Iterating a one-way graph is the confusion this catches.
    """
    findings: list[str] = []
    not_checked: list[str] = []
    names = list(graph.get("participants") or [])
    edges = dict(graph.get("declared_edges") or {})
    if not names or not edges:
        return findings, ["coupling directionality: the participant graph was "
                          "not recorded, so one-way/two-way could not be checked"]
    unknown = {n: [s for s in srcs if s not in names] for n, srcs in edges.items()}
    unknown = {n: u for n, u in unknown.items() if u}
    if unknown:
        findings.append(
            f"Coupling graph names unknown participants: {unknown} — those edges "
            f"carry no data. Known participants: {names}.")
    isolated = [n for n in names if not edges.get(n)]
    if isolated and max_iter != 1:
        findings.append(
            f"ONE-WAY coupling: {isolated} import from nobody, so no information "
            "flows back to them and iterating to 'convergence' is meaningless — "
            "the residual falls to zero because nothing changes, not because the "
            "coupled problem was solved. If one-way IS intended, ask for a single "
            "pass (max_iter=1), which declares it; otherwise set imports_from on "
            "both sides.")
    elif isolated:
        not_checked.append(
            f"two-way convergence: {isolated} import from nobody and this was run "
            "as a declared single pass, so nothing was iterated and no coupled "
            "fixed point was established")
    return findings, not_checked


def unresponsive_clause(name: str, detail: dict | None, sensitivity: dict | None = None) -> str:
    """'' unless the driver saw `name`'s imports change only in some columns: then what
    the export's byte-identity does and does not say, in the columns' own names -- and,
    where the driver's probe nudged every number `name` imports, what that measured."""
    d = (detail or {}).get(name) or {}
    changed, same = list(d.get("changed") or []), list(d.get("unchanged") or [])
    if not changed and same:
        return (f"{name}'s imports changed only at round-off of their own scale "
                f"({', '.join(same)}): nothing it was given moved, so its byte-identical "
                f"export says nothing about whether it reads them")
    if not (changed and same):
        return ""
    head = (f"{name}'s export stayed byte-identical while its imports changed only in "
            f"{', '.join(changed)}; {', '.join(same)} arrived the same at every one of those "
            f"iterations. ")
    # THE PROBE HAS ALREADY ASKED THE SIDE ITSELF. Measured: this clause sent a run to its
    # partner's export while the probe had measured that the side ignored every number it
    # was sent (its export moved 7.6e-13 under the nudge).
    rec = (sensitivity or {}).get(name) if isinstance(sensitivity, dict) else None
    S = rec.get("S") if isinstance(rec, dict) else None
    if isinstance(S, (int, float)) and S == S:
        if S < 1e-9:
            return head + (f"The driver's probe nudged every number {name} imports and its export "
                           f"moved by {S:.1e} of the nudge: {name} itself does not apply what it is "
                           f"sent, whatever its partner exports. Its own script is where to look")
        return head + (f"The driver's probe nudged every number {name} imports and its export "
                       f"responded ({S:.2g} of the nudge), so {name} reads its imports; the column that "
                       f"never changed comes from the partner, and the partner's export is where to look")
    return head + (f"A side applies one of them (a Neumann side its partner's "
                   f"normal_fluxes, a Dirichlet side its values): if {name} applies an unchanged one, "
                   f"what it applied never changed, and the partner's export is where to look")


def check_participant_responsiveness(responsiveness: dict,
                                     detail: dict | None = None,
                                     sensitivity: dict | None = None) -> tuple[list[str], list[str]]:
    """Did every participant's answer actually depend on what it was given?

    This is the check for the participant that exits 0 having done nothing: it
    re-emits its initial condition (or a cached first answer) every iteration, so
    the export-vector change is exactly zero at iteration 2 and the coupling
    reports converged with a residual of 0.0 and no other complaint. A real solve
    handed different boundary data does not return byte-identical output.

    WHAT IT DOES NOT CATCH, stated plainly: the test is byte-identity, so a
    participant whose output depends on its imports only negligibly — a stale
    field with a token dependence added, or a solver whose interface condition is
    applied with a near-zero coefficient — reads as responsive. Byte-identity is
    what the ACCIDENTAL failures look like (a script that never opens
    imports.json, a cached result re-served); a deliberately disguised one needs
    the monolithic comparison, not this.
    """
    findings: list[str] = []
    not_checked: list[str] = []
    if not responsiveness:
        return findings, ["participant responsiveness: the driver recorded no "
                          "per-iteration trace, so a do-nothing participant "
                          "could not be ruled out"]
    dead = [n for n, s in responsiveness.items() if s == "unresponsive"]
    frozen = [n for n, s in responsiveness.items() if s == "imports never changed"]
    partial = [n for n in dead if unresponsive_clause(n, detail, sensitivity)]
    whole = [n for n in dead if n not in partial]
    if partial:
        findings.append(
            "; ".join(unresponsive_clause(n, detail, sensitivity) for n in partial)
            + ". Any convergence reported here is the coupling standing still, not a "
            "solution.")
    if whole:
        findings.append(
            f"Participant(s) {whole} produced byte-identical output while the data "
            "handed to them CHANGED — their answer does not depend on their "
            "imports. Ways measured to get here: the script never reads "
            "imports.json; it re-serves a cached or initial result; or it reads "
            "them and its solve never uses them (an interface held at one fixed "
            "value, imported values written where the solve does not look). Any "
            "convergence reported here is the coupling standing still, not a "
            "solution.")
    if frozen:
        not_checked.append(
            f"participant responsiveness for {frozen}: the data handed to them "
            "never changed during the run, so whether they read it could not be "
            "established")
    return findings, not_checked


def check_interface_meshes(export_a, export_b, label_a="A", label_b="B",
                           rtol: float = 1e-6) -> tuple[list[str], list[str]]:
    """Compare the two sides' interface discretisations.

    Non-matching interface meshes are legitimate and routine, so this is NOT an
    error — but it changes what the other numbers mean. Every exchange then goes
    through an interpolation that is lossy and does not conserve the integrated
    quantity unless the mapping was built to, and a converged residual is
    completely silent about that. Nothing here can inspect the caller's mapping,
    so the honest report is: say the interfaces do not match, and say that
    conservation across them is established by the flux balance or not at all.
    That belongs in the coverage list, not in the findings — reporting the
    geometry as a failure would be as wrong as reporting nothing.
    """
    findings: list[str] = []
    not_checked: list[str] = []

    def _co(e):
        c = e.get("coordinates") if isinstance(e, dict) else getattr(e, "coordinates", None)
        return None if c is None else _np.atleast_2d(_np.asarray(c, float))

    def _has_flux(e):
        f = e.get("normal_fluxes") if isinstance(e, dict) else getattr(e, "normal_fluxes", None)
        return f is not None and len(_np.asarray(f, float).ravel()) > 0

    ca, cb = _co(export_a), _co(export_b)
    if ca is None or cb is None or ca.size == 0 or cb.size == 0:
        return findings, ["interface mesh conformity: one or both participants "
                          "exported no interface coordinates, so matching / "
                          "non-matching discretisation could not be checked"]
    na, nb = len(ca), len(cb)
    if na == nb and ca.shape == cb.shape:
        span = float(_np.max(_np.abs(ca))) or 1.0
        if float(_np.max(_np.abs(ca - cb))) <= rtol * span:
            return findings, not_checked          # matching, node-for-node
    # MATCHING IS A MATTER OF WHERE THE POINTS ARE, NOT OF THEIR ORDER OR COUNT. The
    # same nodes listed in another order, or one side leaving out the interface's two
    # end points that its partner lists, is a conforming seam (measured: both were
    # called NON-MATCHING, and one run's hand-in listed the row order as a key issue).
    _raw = [e.get("coordinates") if isinstance(e, dict) else getattr(e, "coordinates", None)
            for e in (export_a, export_b)]
    if interface_correspondence(*_raw).get("kind") in ("same", "common"):
        return findings, not_checked
    both_flux = _has_flux(export_a) and _has_flux(export_b)
    note = (f"conservation across a NON-MATCHING interface ({label_a} exports {na} "
            f"point(s), {label_b} exports {nb}): every exchange passes through an "
            "interpolation, which is lossy and does not conserve the integrated "
            "quantity unless the mapping was built to — a nearest-neighbour or "
            "plain linear map is not. ")
    note += ("The interface flux balance is the only evidence here that it did "
             "conserve; the residual is silent about it."
             if both_flux else
             "Neither side exported `normal_fluxes`, so NOTHING here checked "
             "whether the interpolation conserved. Export the normal flux from "
             "both sides to make that checkable.")
    return findings, [note]


def check_residual_blocks(block_residuals: dict, tol: float,
                          slack: float = 10.0,
                          fixed_point: dict | None = None,
                          distance: dict | None = None,
                          scale_change: dict | None = None,
                          tiny_change: dict | None = None) -> tuple[list[str], list[str]]:
    """Is the reported global residual actually representative?

    The driver converges on ONE relative norm over every participant's stacked
    export vector. When the exchanged quantities live on different scales — the
    standard case in FSI (forces ~1e3, displacements ~1e-5) and TSI (temperature
    ~1e3, displacement ~1e-5) — the large block sets the denominator and the small
    block can still be moving by a large fraction of itself while the global
    number sits below tolerance. That is a converged-looking, wrong answer with no
    other symptom.
    """
    findings: list[str] = []
    not_checked: list[str] = []
    if not block_residuals:
        return findings, ["per-block convergence: the driver recorded no "
                          "per-block residuals, so scale masking in the global "
                          "residual could not be ruled out"]
    finite = {k: v for k, v in block_residuals.items() if v == v and abs(v) != float("inf")}
    if not finite:
        return findings, ["per-block convergence: every per-block residual was "
                          "non-finite or unavailable"]
    limit = tol * slack
    bad = {k: v for k, v in finite.items() if v > limit}
    # A LAST STEP IS NOT A DISTANCE. Under an accelerator a block can move a lot on its
    # last step and land on the fixed point; where the driver measured the block's own
    # fixed-point residual (raw output against the relaxed input), or estimated the
    # distance left from how fast its last two steps shrank, and either is inside the
    # limit, the block has converged whatever its last step was. With no relaxation the
    # fixed-point residual IS the last step, and only the distance can clear it.
    for est in (fixed_point, distance):
        if est:
            bad = {k: v for k, v in bad.items()
                   if not (isinstance(est.get(k), float) and est[k] == est[k] and est[k] <= limit)}
    # A BLOCK THAT MOVED WITHIN THE LIMIT ON ITS OWN SCALE HAS SETTLED. The entry-by-entry measure
    # reads an entry against its own value, so an entry of one field that holds a small share of
    # its block reads large for an absolute change at the iteration's tolerance (measured: 1.74e-4
    # at an entry holding 0.9 % of its block's largest value, on a right ladder's finest level). A
    # block is cleared when its last change against its largest value is within the limit and so
    # is that of each entry more than six orders below that value against itself (another quantity
    # sharing the array: the masking this check exists for; see coupling_driver._scaled_change).
    if scale_change and tiny_change:
        def _within(est, k):
            v = est.get(k)
            return isinstance(v, float) and v == v and v <= limit
        bad = {k: v for k, v in bad.items() if not (_within(scale_change, k) and _within(tiny_change, k))}
    if bad:
        worst = max(bad.items(), key=lambda kv: kv[1])
        # WHAT WAS MEASURED, SAID: the largest change of one entry of the block at the
        # LAST iteration, relative to that entry's own value -- the iteration's precision
        # at this level. Measured on a steady ladder: 1.74e-4 at an interface-end entry
        # holding 0.9 % of its block's largest value, whose block as a whole moved 8.7e-7.
        findings.append(
            "Global residual is NOT representative: block(s) "
            + ", ".join(f"{k}={v:.2e}" for k, v in sorted(bad.items()))
            + f" are still changing by more than {limit:.1e} relative (the entry that "
            "moved most at the last iteration, relative to its own value), while the "
            "global norm — which is dominated by the largest-magnitude block — "
            f"reports convergence. {worst[0]} is the one to look at. This measures the "
            "iteration at its last step, not a discretisation trend. Converge each "
            "exchanged quantity in its own units, or scale the blocks before taking "
            "the norm.")
    return findings, not_checked


def check_returncodes(returncodes: dict) -> tuple[list[str], list[str]]:
    """Every participant's LAST run must have exited 0.

    A solver that diverges commonly writes its last iterate and then aborts; the
    file handshake sees a perfectly well-formed exports.json and couples on it.
    """
    findings: list[str] = []
    if not returncodes:
        return findings, ["participant exit codes: none were recorded"]
    bad = {n: rc for n, rc in returncodes.items() if rc != 0}
    if bad:
        findings.append(
            f"Participant(s) exited non-zero: {bad} — the exchanged data on that "
            "iteration is the output of a FAILED solve, whatever the residual says.")
    return findings, []



def is_stub_output(content: str) -> str | None:
    """Detect a placeholder/stub generator output that advertises physics but does
    NOT produce a runnable, solving deck. Returns a reason string if stub, else None.

    Catches the silent-wrong catalog landmines the audits found across backends:
    deal.II print-and-exit placeholders, Kratos availability-probe stubs, 4C one-line
    comment templates, and `<...>`-placeholder decks. Turning these into a LOUD refusal
    (rather than fake output that passes validation) is the paper's own principle applied
    to openPASO itself.
    """
    if content is None:
        return "empty generator output"
    c = content.strip()
    if not c:
        return "empty generator output"
    low = c.lower()
    # one-line / comment-only templates (4C stubs like "# Membrane template — use ...")
    non_comment = [ln for ln in c.splitlines()
                   if ln.strip() and not ln.strip().startswith("#")]
    if not non_comment:
        return "template is comment-only — not a runnable deck (stub)"
    # explicit placeholder markers
    markers = [
        "see deal.ii tutorial for full implementation",  # dealii print-and-exit
        "placeholder: implement",                        # dealii NS / others
        "# placeholder", "// placeholder", "placeholder template",
        "not pip-installable", "not installed",          # kratos probe stubs
        '"note": "not installed"', "format template",     # kratos rom/iga/topology
        "use this as a starting point — not a self-contained",  # reduced_lung
    ]
    for m in markers:
        if m in low:
            return f"placeholder marker present ('{m}') — generator is a stub, not a real solve"
    # unfilled angle-bracket scalar placeholders (4C <...> YAML templates that abort).
    # Skip for XML (FEBio tags like <time_steps>) and C++ (deal.II templates <double>).
    import re
    is_xml = c.startswith("<") or "<?xml" in low or "</" in c
    is_cpp = "#include" in low or "int main" in low
    if not is_xml and not is_cpp and len(re.findall(r"<[a-z]+_[a-z_]+>", low)) >= 3:
        return "contains unfilled <...> placeholders — deck would not run (stub)"
    return None


def _imports_at_roundoff(name, exports, graph):
    """(largest |flux| `name` imports, largest |trace| on the interface) when the
    fluxes its partners export are nonzero round-off against the interface trace
    (its own export's and its partners'); None otherwise, or without the exports
    and the graph."""
    if not isinstance(exports, dict) or not isinstance(graph, dict):
        return None
    partners = [p for p in ((graph.get("declared_edges") or {}).get(name) or []) if p in exports]
    got = [_flux_and_trace(exports[p]) for p in partners]
    fl = [f for f, _t in got if f is not None]
    if not fl:
        return None
    tr = [t for _f, t in got if t is not None]
    if name in exports:
        own = _flux_and_trace(exports[name])[1]
        if own is not None:
            tr.append(own)
    fmax, tmax = max(fl), max(tr, default=0.0)
    if tmax > 0 and 0 < fmax <= _ROUNDOFF * tmax:
        return fmax, tmax
    return None


def check_interface_sensitivity(sensitivity: dict, floor: float = 1e-9,
                                noise_margin: float = 3.0,
                                noise_floor: float | None = None,
                                exports: dict | None = None,
                                graph: dict | None = None
                                ) -> tuple[list[str], list[str]]:
    """Did each participant's answer measurably depend on what it was handed?

    Consumes core.coupling_driver.probe_interface_sensitivity, which re-runs each
    participant twice after the coupling settles: once on exactly the imports it
    last had (NOISE — a solver that is a function of its boundary data returns
    the same answer) and once on those imports nudged by a known relative amount
    (SIGNAL). Two things are then decidable that watching the iteration cannot
    decide:

      * SIGNAL indistinguishable from NOISE: the participant's answer moves as
        much when nothing changed as when its boundary data did. It carries
        hidden state, or it is stochastic. Either way a fixed-point iteration
        over it has not converged to a coupled solution — and a participant that
        never opens imports.json while advancing a counter looks perfectly
        responsive during the iteration, which is how one was stamped
        trustworthy.

      * S = SIGNAL / perturbation below `floor`: the export is not a function of
        the import to any precision double arithmetic can express. The floor is
        far below anything physical — a rigid structure barely deflected by a
        fluid load still responds by roughly the stiffness ratio, and 1e-9 is a
        ratio of a billion.

    `exports` (participant -> its export) and `graph` (the driver's, with
    "declared_edges") let the NOT COUPLED branch see a round-off import; without
    them the probe's numbers are all there is.
    """
    findings: list[str] = []
    not_checked: list[str] = []
    if not sensitivity:
        return findings, [
            "interface sensitivity: NOT probed. Nothing established that the "
            "participants' answers depend on the data they were handed — a "
            "solver that never opens imports.json converges and passes every "
            "other check in this list."]
    for name, rec in sorted(sensitivity.items()):
        rec = rec if isinstance(rec, dict) else {}
        noise, signal, S = rec.get("noise"), rec.get("signal"), rec.get("S")
        why = rec.get("detail") or "the probe could not be carried out"
        if S is None:
            not_checked.append(
                f"interface sensitivity for {name}: NOT measured ({why}), so "
                "whether its answer depends on its partner is unknown")
            continue
        if noise is not None and noise > 0 and signal is not None \
                and signal <= noise * noise_margin:
            # THIS BRANCH CANNOT TELL ITS TWO CAUSES APART, and its own message
            # says so: "hidden state between calls, OR it is stochastic". When
            # the caller has DECLARED the coupling stochastic and the driver has
            # MEASURED a non-zero residual floor, the second cause is not a
            # suspicion — it is the established fact the run was judged against,
            # and a Monte-Carlo participant re-run on identical imports moves by
            # construction. Reported as coverage there, because it is exactly a
            # question this instrument could not answer; still a FINDING with no
            # floor in play, where "it is stochastic" would be an unevidenced
            # excuse. Measured before this split: a correct noisy pair converged
            # at a floor of 8.3e-03 with the probe reporting noise 2.29e-02
            # against signal 9.63e-03, and the finding stamped it NOT VERIFIED.
            # ...but ONLY when there is a response to be ambiguous about. A
            # participant whose export does not move AT ALL when its imports are
            # perturbed is not coupled to anything, and no amount of sampling
            # noise makes that acceptable — measured: with signal exactly 0 the
            # first guard alone routed it to coverage and the zero-response
            # finding below was never reached. So fall through to it.
            if noise_floor and S == S and S >= floor:
                not_checked.append(
                    f"interface sensitivity for {name}: NOT SEPARABLE. Re-running "
                    f"it on the SAME data moved its answer by {noise:.2e} "
                    f"relative and perturbing the data moved it by {signal:.2e}, "
                    f"which this probe cannot tell apart. With a measured "
                    f"residual noise floor of {noise_floor:.2e} that is what a "
                    f"sampled estimator looks like and is expected — but it is "
                    f"ALSO what a participant carrying hidden state looks like, "
                    f"and nothing here separates them. Compare the ANSWER "
                    f"against a reference (`monolithic`, or an independent "
                    f"solve) if you need that distinction. Note the response is "
                    f"still non-zero, so the participant is not ignoring its "
                    f"imports outright — that half is checked below.")
                continue
            findings.append(
                f"Participant {name} is NOT A FUNCTION of its imports: re-running "
                f"it on the SAME data moved its answer by {noise:.2e} relative, "
                f"and perturbing the data moved it by {signal:.2e} — the response "
                "cannot be told apart from its own run-to-run drift. It carries "
                "hidden state between calls, or it is stochastic. A partitioned "
                "fixed-point iteration over such a participant has not converged "
                "to a coupled solution, whatever the residual history shows. If "
                "a participant really is a sampled estimator, say so with "
                "`noise_replicates` — the driver then MEASURES its floor and "
                "this becomes a coverage note instead of a finding.")
            continue
        if not (S == S) or S < floor:
            # A ROUND-OFF IMPORT IS NUDGED BY ROUND-OFF. The probe moves each imported
            # number by a fraction of itself, so where the partner's flux is round-off
            # the nudge of it was too, and a side that applies that flux cannot answer
            # it (measured: a Neumann side that did integrate its import was called
            # NOT COUPLED on a 1e-17 import). Said, not judged, where the exports show it.
            _ri = _imports_at_roundoff(name, exports, graph)
            if _ri is not None:
                not_checked.append(
                    f"interface sensitivity for {name}: NOT measured. The flux it imports "
                    f"is at round-off ({_ri[0]:.1e}, against an interface trace of "
                    f"{_ri[1]:.3g}), and the probe nudges each imported number by a "
                    f"fraction of itself, so that nudge was round-off too; its answer moved "
                    f"by {S:.2e} of the perturbation. Whether {name} applies the flux it "
                    f"imports is not established here.")
                continue
            findings.append(
                f"Participant {name} is NOT COUPLED to its partner: perturbing "
                f"every number handed to it moved its answer by a relative "
                f"{S:.2e} of the perturbation. Its output does not depend on its "
                "imports to any precision double arithmetic can express, so the "
                "iteration converged to whatever it produces on its own — not to "
                "a coupled solution.")
            continue
        # A participant can respond in one block and be frozen in another: hold
        # the physics at a stale constant while echoing an imported quantity
        # back. The total then responds fully and the frozen half is invisible.
        blocks = rec.get("blocks") or {}
        dead = sorted(k for k, v in blocks.items()
                      if v is not None and v == v and v < floor)
        if dead and len(dead) < len(blocks):
            findings.append(
                f"Participant {name} exports block(s) {dead} that do NOT respond "
                "to its imports at all, while the rest of its export does. That "
                "part of its answer is a constant or a stale field carried "
                "through the iteration — the coupling converged around it "
                "without ever solving it.")
    # The frontier of this check, in the SERVED coverage rather than only in the
    # docstring above. Measured, not assumed: a participant whose export is
    # WRONG_CONSTANT + eps*import passes every check here for eps down to the
    # floor, because S is then eps and eps > floor. At eps=1e-6 and at eps=1e-8 a
    # participant frozen at a 50%-wrong value came back with no finding at all;
    # only at eps=1e-10 did the per-block test fire. A response that is real but
    # tiny is what stiff physics looks like, so no local measurement can separate
    # the two — the un-split `monolithic` comparison is the only thing that can.
    if any(isinstance(r, dict) and r.get("S") is not None
           for r in sensitivity.values()):
        not_checked.append(
            f"interface sensitivity FRONTIER: a measured response above the "
            f"floor ({floor:.0e}) establishes only that the export moves when "
            "the imports move — NOT that it moves by the right amount. A "
            "participant frozen at a badly wrong value that adds a token "
            "multiple of its import responds just enough to pass, and a genuinely "
            "stiff participant responds just as little, so the two are not "
            "separable by any measurement made here. Pass `monolithic` if you "
            "need that distinction.")
    return findings, not_checked
